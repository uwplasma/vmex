"""Effective-ripple adapter (state and wout to NEO_JAX, radial interpolation, gradient through the solve) and the NEO screen.

The bounce-integral physics tests live in NEO_JAX (``tests/unit/test_ripple.py``)."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vmex.core import neoclassical

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture
def transport_screen_backend(monkeypatch):
    """Exercise the screening adapter without a Boozer solve or a NEO run."""
    calls = {"runs": 0, "neo_calls": 0}
    valid = dict(rational_correction_valid=True, approximation_used=False,
                 rational_work_limit_exceeded=False)
    calls["results"] = [SimpleNamespace(s=s, epsilon_effective=value, diagnostics=valid.copy())
                        for s, value in [(.75, 3.e-4), (.25, 1.e-4)]]

    class Boozer:
        s_in = np.array([.25, .5, .75])

        def __init__(self, **kwargs):
            calls["booz_options"] = kwargs

        def read_wout_data(self, wout):
            calls["wout"] = wout

        def run(self, **kwargs):
            calls["runs"] += 1
            calls["run_options"] = kwargs
            calls["indices"] = list(self.compute_surfs)

    def config(**kwargs):
        calls["config_options"] = kwargs
        return SimpleNamespace(**kwargs)

    def run_neo(booz, **kwargs):
        calls["neo_calls"] += 1
        calls["neo_data"], calls["neo_options"] = booz, kwargs
        return calls["results"]

    monkeypatch.setitem(sys.modules, "booz_xform_jax", SimpleNamespace(Booz_xform=Boozer))
    monkeypatch.setattr(neoclassical, "_neo_imports", lambda: (config, run_neo))
    monkeypatch.setattr(neoclassical, "_neo_boozer_data",
                        lambda bx, indices: {"s_b": bx.s_in[indices]})
    return calls


@pytest.mark.parametrize("custom_config", [False, True])
def test_transport_screen_preserves_order_and_uses_strict_forward_settings(
    transport_screen_backend, custom_config,
):
    calls = transport_screen_backend
    config = object() if custom_config else None
    wout = object()
    values = neoclassical.screen_transport(wout, [.75, .25], mboz=8, nboz=6, config=config)
    assert calls["wout"] is wout
    assert calls["booz_options"] == dict(verbose=0, mboz=8, nboz=6)
    assert calls["indices"] == [2, 0] and calls["run_options"] == {}
    np.testing.assert_array_equal(calls["neo_data"]["s_b"], [.75, .25])
    np.testing.assert_array_equal(values, [3.e-4, 1.e-4])
    options = calls["neo_options"]
    assert options["use_jax"] and not options["progress"] and not options["jax_surface_scan"]
    if custom_config:
        assert options["config"] is config and "config_options" not in calls
    else:
        assert calls["config_options"] == dict(theta_n=192, phi_n=192, npart=80,
            multra=2, nstep_per=80, nstep_min=800, nstep_max=2000, acc_req=.005,
            ref_swi=2, max_rational_field_periods=100000, rational_surface_policy="error")


def test_transport_screen_rejects_snapping_before_running_either_backend(transport_screen_backend):
    calls = transport_screen_backend
    with pytest.raises(ValueError, match="requested half-mesh radii"):
        neoclassical.screen_transport(object(), [.251, .75])
    assert calls["runs"] == calls["neo_calls"] == 0


@pytest.mark.parametrize("value, diagnostic_update", [
    (np.nan, {}), (np.inf, {}), (0., {}), (-1.e-4, {}),
    (1.e-4, {"rational_correction_valid": False}),
    (1.e-4, {"approximation_used": True}),
    (1.e-4, {"rational_work_limit_exceeded": True}),
])
def test_transport_screen_rejects_an_invalid_surface_instead_of_a_partial_profile(
    transport_screen_backend, value, diagnostic_update,
):
    calls = transport_screen_backend
    calls["results"][1].epsilon_effective = value
    calls["results"][1].diagnostics.update(diagnostic_update)
    with pytest.raises(ValueError, match="Invalid NEO screen at s=0.25.*keep the incumbent"):
        neoclassical.screen_transport(object(), [.75, .25])
    assert calls["runs"] == calls["neo_calls"] == 1


def test_half_mesh_bracketing_interpolates_linearly_in_s():
    s_half = (np.arange(10) + 0.5) / 10
    targets = [0.3, 0.35, 0.01, 0.99]
    rows, lo, hi, w = neoclassical._bracket(s_half, targets)
    values = np.arange(rows.size, dtype=float) ** 2
    np.testing.assert_allclose((1 - w) * values[lo] + w * values[hi], np.interp(targets, s_half[rows], values))


@pytest.fixture(scope="module", autouse=False)
def ripple_backend():
    neo = pytest.importorskip("neo_jax")
    if not hasattr(neo, "effective_ripple_3_2"):
        pytest.skip("the effective ripple needs neo-jax >= 1.2.0")


@pytest.fixture(scope="module")
def qa_equilibrium(ripple_backend):
    import vmex as vj
    from vmex import optimize as opt

    return opt.solve_equilibrium(vj.VmecInput.from_file(DATA_DIR / "input.LandremanPaul2021_QA_lowres"))


def test_state_and_wout_routes_agree_and_a_good_qa_has_a_tiny_ripple(qa_equilibrium):
    """Both Boozer routes give ~1e-8 for this QA (a NEO default run returns noise there)."""
    eq, surfaces = qa_equilibrium, [0.3, 0.6]
    from_state = np.asarray(neoclassical.epsilon_effective_3_2_from_state(eq.state, eq.runtime, surfaces))
    from_wout = np.asarray(neoclassical.epsilon_effective_3_2_from_wout(eq.wout, surfaces))
    assert np.all((from_state > 1e-9) & (from_state < 1e-7))
    np.testing.assert_allclose(from_state, from_wout, rtol=0.1)


def test_boundary_gradient_matches_finite_differences_after_a_tight_resolve(qa_equilibrium):
    """Reverse mode through the implicit solve vs central differences of re-solved equilibria.

    The objective is the sum of ``epsilon_eff^(3/2)`` over three surfaces of the
    QA deck, differentiated with respect to its eight lowest boundary modes.
    """
    import jax
    import jax.numpy as jnp

    from vmex import optimize as opt

    inp = qa_equilibrium.inp
    surfaces = (0.25, 0.5, 0.75)
    loss = lambda st, rt: jnp.sum(neoclassical.epsilon_effective_3_2_from_state(st, rt, surfaces))  # noqa: E731
    probe = opt.VmecProblem.from_loss(inp, loss, max_mode=1, use_ess=True, restart_from=qa_equilibrium)
    sub0 = probe.subproblem(max_mode=1, x=probe.x0)
    # move off the sharp QA minimum (there a central difference only sees curvature), re-solve, restart there
    inp1 = probe.input_from_x(sub0.embed(np.asarray(sub0.x0) + 0.01 * np.asarray(sub0.scales)))
    base = opt.solve_equilibrium(inp1, initial_state=qa_equilibrium.state, forward_ftol=1e-13, forward_max_iterations=20000)
    problem = opt.VmecProblem.from_loss(inp1, loss, max_mode=1, use_ess=True, restart_from=base)
    sub = problem.subproblem(max_mode=1, x=problem.x0)
    x0, scales = np.asarray(sub.x0), np.asarray(sub.scales)
    _, grad = sub.value_and_grad(x0)

    def value(x):
        eq = opt.solve_equilibrium(problem.input_from_x(sub.embed(x)), initial_state=base.state,
                                   forward_ftol=1e-13, forward_max_iterations=20000)
        return float(jax.jit(loss)(eq.state, eq.runtime))

    k = int(np.argmax(np.abs(np.asarray(grad) * scales)))
    e = np.zeros_like(x0)
    e[k] = 1e-3 * scales[k]
    fd = (value(x0 + e) - value(x0 - e)) / 2e-3
    np.testing.assert_allclose(fd, float(grad[k]) * scales[k], rtol=0.05)


@pytest.mark.full  # ~3 min: an asymmetric solve, two ripple routes and re-solved central differences
def test_asymmetric_equilibrium_value_routes_and_gradient(ripple_backend):
    """A solved LASYM equilibrium (QA deck plus a small asymmetric boundary): finite, route-independent, differentiable."""
    import re

    import jax
    import jax.numpy as jnp

    import vmex as vj
    from vmex import optimize as opt

    text = re.sub(r"LASYM\s*=\s*\S+", "", (DATA_DIR / "input.LandremanPaul2021_QA_lowres").read_text(), flags=re.I).replace(
        "&INDATA", "&INDATA\n  LASYM = T\n  RBS(1,1) = 0.02\n  ZBC(1,1) = 0.02\n  RBS(0,1) = 0.01", 1)
    inp = vj.VmecInput.from_indata_text(text)
    eq = opt.solve_equilibrium(inp, forward_ftol=1e-13, forward_max_iterations=20000)
    surfaces = (0.3, 0.6)
    state_value = np.asarray(neoclassical.epsilon_effective_3_2_from_state(eq.state, eq.runtime, surfaces))
    assert np.all(np.isfinite(state_value)) and np.all(state_value > 1e-6)   # the asymmetry lifts the ripple far above the QA 1e-8
    np.testing.assert_allclose(state_value, neoclassical.epsilon_effective_3_2_from_wout(eq.wout, surfaces), rtol=2e-2)
    loss = lambda st, rt: jnp.sum(neoclassical.epsilon_effective_3_2_from_state(st, rt, surfaces))  # noqa: E731
    problem = opt.VmecProblem.from_loss(inp, loss, max_mode=1, use_ess=True, restart_from=eq)
    sub = problem.subproblem(max_mode=1, x=problem.x0)
    x0, scales = np.asarray(sub.x0), np.asarray(sub.scales)
    grad = np.asarray(sub.value_and_grad(x0)[1]) * scales
    k = int(np.argmax(np.abs(grad)))
    e = np.zeros_like(x0)
    e[k] = 1e-3 * scales[k]

    def value(x):
        eqx = opt.solve_equilibrium(problem.input_from_x(sub.embed(x)), initial_state=eq.state, forward_ftol=1e-13, forward_max_iterations=20000)
        return float(jax.jit(loss)(eqx.state, eqx.runtime))

    np.testing.assert_allclose((value(x0 + e) - value(x0 - e)) / 2e-3, grad[k], rtol=0.05)


def test_neo_summary_subset_keeps_all_sine_partners():
    from vmex.core.plotting import _neo_surface_subset

    data = {"ns_b": 5, "asym": True, "mode_first": True}
    for key in ("iota_b", "buco_b", "bvco_b", "s_b"):
        data[key] = np.arange(5)
    keys = ("rmnc_b", "zmns_b", "pmns_b", "bmnc_b",
            "rmns_b", "zmnc_b", "pmnc_b", "bmns_b")
    for key in keys:
        data[key] = np.arange(10).reshape(2, 5)
    subset = _neo_surface_subset(data)
    assert subset["ns_b"] == 3 and subset["mode_first"] and subset["asym"]
    for key in keys:
        np.testing.assert_array_equal(subset[key], data[key][:, [0, 2, 4]])


@pytest.mark.parametrize("trial, expected", [
    ([.8, .9, .9], True), ([.8, 1.01, .8], False),
    ([.999, .999, .999], False), ([0, .5, .5], False),
    ([float("nan"), .5, .5], False), ([.5, .5], False),
])
def test_transport_screen_rejects_radial_tradeoffs(trial, expected):
    from vmex.core.neoclassical import transport_improves

    assert transport_improves([1., 1., 1.], trial) is expected

"""Effective ripple: bounce-integral value, anchors, cross-code reference and gradients; NEO screen adapter."""

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


NFP = 2


def _torus(modes, iota=0.41, current=0.0, r=0.2, R0=5.0):
    """Large-aspect-ratio Boozer surface: |B| = 1 + sum a cos(m theta - n phi); circular section."""
    xm = np.array([0, 1] + [m for m, _, _ in modes], float)
    xn = np.array([0, 0] + [n for _, n, _ in modes], float)
    b, rm, zs = np.zeros(xm.size), np.zeros(xm.size), np.zeros(xm.size)
    b[0], rm[0], rm[1], zs[1] = 1.0, R0, r, r
    b[2:] = [a for _, _, a in modes]
    return dict(nfp_b=NFP, ixm_b=xm, ixn_b=xn, iota_b=np.array([iota]), buco_b=np.array([current]),
                bvco_b=np.array([R0]), rmnc_b=rm[None], zmns_b=zs[None], pmns_b=0 * zs[None],
                bmnc_b=b[None], s_b=np.array([0.5]))


def _eps(booz, **budget):
    return float(neoclassical.epsilon_effective_3_2_from_boozer(booz, **budget)[1][0])


def test_quasi_symmetric_fields_have_no_ripple_and_a_perturbation_scales_quadratically():
    """A single helicity (tokamak-like QA or QH, with a net current) gives exactly zero.

    With ``B = B(m theta - n phi)`` the geodesic integral ``I_1`` is the integral
    of an exact derivative between bounce points, so ``epsilon_eff`` vanishes
    (Nemov et al. 1999); a symmetry-breaking perturbation of relative size
    ``lam`` then gives ``epsilon_eff^(3/2) ~ lam^2``.
    """
    base = _eps(_torus([(1, 0, -0.1), (1, NFP, -0.02)]))
    for modes, current in (([(1, 0, -0.1)], 0.0), ([(1, NFP, -0.1)], 0.3)):
        assert _eps(_torus(modes, current=current)) < 1e-9 * base
    values = [_eps(_torus([(1, 0, -0.1), (1, NFP, -0.005 * lam)])) / lam**2 for lam in (1, 2)]
    np.testing.assert_allclose(values[1], values[0], rtol=0.05)


def test_value_is_invariant_to_angle_shifts_and_field_scale():
    """Shifting the Boozer angles (sine partners appear) or scaling ``B`` leaves the ripple unchanged."""
    booz = _torus([(1, 0, -0.1), (1, NFP, -0.03), (2, NFP, 0.01)], current=0.2)
    reference = _eps(booz)
    c, d = 0.7, 0.4
    delta = booz["ixm_b"] * c - booz["ixn_b"] * d
    shifted = dict(booz)
    for cos_key, sin_key in (("bmnc_b", "bmns_b"), ("rmnc_b", "rmns_b"), ("zmnc_b", "zmns_b"), ("pmnc_b", "pmns_b")):
        fc, fs = booz.get(cos_key, 0 * booz["bmnc_b"]), booz.get(sin_key, 0 * booz["bmnc_b"])
        shifted[cos_key] = fc * np.cos(delta) + fs * np.sin(delta)
        shifted[sin_key] = fs * np.cos(delta) - fc * np.sin(delta)
    np.testing.assert_allclose(_eps(shifted), reference, rtol=3e-3)
    scaled = dict(booz, bmnc_b=2.5 * booz["bmnc_b"], bvco_b=2.5 * booz["bvco_b"], buco_b=2.5 * booz["buco_b"])
    np.testing.assert_allclose(_eps(scaled), reference, rtol=1e-6)


def test_matches_converged_neo_on_the_qi_fixture_and_converges_in_the_budget():
    """Cross-code number and convergence on the nfp = 4 QI summary spectrum (iota ~ -1.3).

    The NEO_JAX values (``theta_n = phi_n = 128``, 96 particles, 80 steps per
    period, ``acc_req = 1e-3``) move by < 0.3 % against a ``theta_n = 64``
    run; the default budget agrees with them to 2 %, a 3x larger budget to 0.3 %.
    """
    with np.load(DATA_DIR.parent.parent / "tests" / "data" / "boozer_nfp4_QI_finite_beta.npz") as data:
        booz = {key: data[key] for key in data.files}
    booz["mode_first"] = True
    s, values = neoclassical.epsilon_effective_3_2_from_boozer(booz)
    np.testing.assert_allclose(s, [0.43, 0.49, 0.55], rtol=0, atol=1e-12)
    np.testing.assert_allclose(values, [5.2224e-4, 4.7922e-4, 4.5097e-4], rtol=0.03)
    one = {**booz, **{k: np.asarray(booz[k])[:, 1:2] for k in ("rmnc_b", "zmns_b", "pmns_b", "bmnc_b")},
           **{k: np.asarray(booz[k])[1:2] for k in ("iota_b", "buco_b", "bvco_b", "s_b")}}
    fine = _eps(one, nalpha=48, nturn=6, next=6, nper=48, nq=8, nquad=64)
    np.testing.assert_allclose(fine, 4.7922e-4, rtol=3e-3)


def test_gradient_forward_reverse_and_finite_difference_agree():
    """``jvp``, ``grad`` and central differences agree for the spectrum, ``iota`` and the current."""
    import jax

    booz = _torus([(1, 0, -0.1), (1, NFP, -0.03), (2, NFP, 0.01)], current=0.2)
    amp = jax.numpy.asarray(booz["bmnc_b"][0])

    def f(a, iota, current):
        b = dict(booz, bmnc_b=a[None], iota_b=iota[None], buco_b=current[None])
        return neoclassical.epsilon_effective_3_2_from_boozer(b, nalpha=16, nquad=32)[1][0]

    x0 = (amp, jax.numpy.asarray(0.41), jax.numpy.asarray(0.2))
    grads = jax.jit(jax.grad(f, argnums=(0, 1, 2)))(*x0)
    direction = (jax.numpy.asarray(np.random.default_rng(1).normal(size=amp.size)) * 1e-2, 1.0, 1.0)
    forward = jax.jvp(f, x0, direction)[1]
    reverse = sum(float(np.vdot(g, t)) for g, t in zip(grads, direction))
    np.testing.assert_allclose(forward, reverse, rtol=1e-8)
    h = 1e-4
    plus = f(*[x + h * t for x, t in zip(x0, direction)])
    minus = f(*[x - h * t for x, t in zip(x0, direction)])
    np.testing.assert_allclose((plus - minus) / (2 * h), reverse, rtol=5e-3)


def test_half_mesh_bracketing_interpolates_linearly_in_s():
    s_half = (np.arange(10) + 0.5) / 10
    targets = [0.3, 0.35, 0.01, 0.99]
    rows, lo, hi, w = neoclassical._bracket(s_half, targets)
    values = np.arange(rows.size, dtype=float) ** 2
    np.testing.assert_allclose(neoclassical._interpolate(values, lo, hi, w),
                               np.interp(targets, s_half[rows], values))


@pytest.fixture(scope="module")
def qa_equilibrium():
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

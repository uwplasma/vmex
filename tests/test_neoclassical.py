"""Neoclassical adapters, continuous-model enclosures and differentiable objectives."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_banded
import pytest

from vmex.core import neoclassical
from vmex.core.neoclassical import (
    ambipolar_current_interval,
    bounce_transport_bounds,
    conserving_transport_bounds,
    conserving_goal_interval,
    continuous_lorentz_heat_bounds,
    thermal_transport_bounds,
)

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
    assert calls["indices"] == [2, 0] and calls["run_options"] == dict(jit=True)
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


def test_epsilon_effective_boozer_adapter_preserves_surface_labels(monkeypatch):
    values = np.array([1.0e-4, 2.0e-4])
    calls = {}

    def run_neo(booz, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(eps_eff=values)

    monkeypatch.setattr(neoclassical, "_neo_imports", lambda: (lambda: None, run_neo))
    s, actual = neoclassical.epsilon_effective_from_boozer(
        {"s_b": np.array([0.25, 0.75])}, config=object())
    np.testing.assert_allclose(s, [0.25, 0.75])
    np.testing.assert_allclose(actual, values)
    assert calls == {
        "config": calls["config"], "use_jax": True, "progress": False,
        "jax_surface_scan": True}


def test_epsilon_effective_reads_both_neo_result_and_surface_conventions(monkeypatch):
    """NEO_JAX names the profile ``eps_eff`` or ``epsilon_effective``.

    ``BoozerData`` carries its own surface labels in ``es``; a mapping without
    ``s_b`` has none, and the adapter then falls back to surface indices so a
    plot never mislabels the radial axis.
    """
    values = np.array([3.0e-5, 4.0e-5])
    monkeypatch.setattr(
        neoclassical, "_neo_imports",
        lambda: (lambda: None,
                 lambda booz, **_kwargs: SimpleNamespace(
                     epsilon_effective=values)))
    surfaces, actual = neoclassical.epsilon_effective_from_boozer(
        SimpleNamespace(es=np.array([0.1, 0.9])))
    np.testing.assert_allclose(surfaces, [0.1, 0.9])
    np.testing.assert_allclose(actual, values)

    surfaces, _ = neoclassical.epsilon_effective_from_boozer({"bmnc_b": None})
    np.testing.assert_allclose(surfaces, [0.0, 1.0])


def test_epsilon_effective_default_is_library_safe():
    """A diagnostic call must not clear process-wide JAX caches by default.

    Cache release is memory *policy* and belongs to the caller (the CLI does
    it after all requested diagnostics); a library default of ``True`` would
    silently discard every warm executable of the surrounding program.
    """
    import inspect

    signature = inspect.signature(neoclassical.epsilon_effective_from_wout)
    assert signature.parameters["clear_jax_caches"].default is False


def test_diagnostic_config_is_the_bounded_summary_resolution(monkeypatch):
    """Summary figures ask NEO for a radial trend, not a transport number.

    The bounded settings (Nemov PoP 6, 4622 (1999) integrates along field
    lines until ``acc_req`` is met) resolve the summary Boozer spectrum
    without aliasing; publication numbers pass their own ``NeoConfig``.
    """
    monkeypatch.setattr(
        neoclassical, "_neo_imports", lambda: (SimpleNamespace, None))
    config = neoclassical.diagnostic_neo_config()
    assert (config.theta_n, config.phi_n, config.npart) == (32, 32, 24)
    assert (config.nstep_per, config.nstep_min, config.nstep_max) == (20, 200, 500)
    assert config.acc_req == 0.02
def test_diagnostic_config_matches_converged_neo_on_the_qi_deck():
    """The ``--plot`` resolution reproduces converged NEO without aliasing dips.

    The fixture is the summary Boozer transform (``mboz=16, nboz=12``) of the
    solved ``examples/data/input.nfp4_QI_finite_beta`` deck on s = 0.43, 0.49,
    0.55.  A 16 x 16 NEO spline grid aliases its m = 13-15 harmonics and gave
    5.9e-7 and 1.7e-9 on the outer surfaces of this trio.  The reference is
    NEO_JAX's default ``NeoConfig``, which moves by less than 5% when any
    resolution control is doubled.
    """
    pytest.importorskip("neo_jax")
    with np.load(Path(__file__).parent / "data" / "boozer_nfp4_QI_finite_beta.npz") as data:
        booz = {key: data[key] for key in data.files}
    s, values = neoclassical.epsilon_effective_from_boozer(
        booz, config=neoclassical.diagnostic_neo_config())
    np.testing.assert_allclose(s, [0.43, 0.49, 0.55], rtol=0, atol=1e-12)
    np.testing.assert_allclose(values, [5.239e-4, 4.838e-4, 4.557e-4], rtol=0.05)


@pytest.mark.parametrize("lasym", [False, True])
def test_epsilon_effective_from_wout_snaps_surfaces_and_negates_the_nu_table(
    monkeypatch, lasym,
):
    """The in-memory adapter is a pure relabelling of a booz_xform run.

    NEO's Boozer toroidal-angle table ``pmns_b`` is booz_xform's ``numns_b``
    with the opposite sign (NEO_JAX ``BoozerData``; booz_xform stores
    ``nu = phi_B - phi``), and requested surfaces are snapped to the Boozer
    surfaces that were actually transformed rather than interpolated.
    """
    import booz_xform_jax

    numns = np.array([[0.1, -0.2], [0.3, -0.4]])
    captured = {}

    class FakeBoozXform:
        asym = lasym
        rmns_b = zmnc_b = bmns_b = np.full((2, 2), .03)
        numnc_b = numns
        s_in = np.linspace(0.05, 0.95, 10)
        nfp = 3
        xm_b, xn_b = np.array([0, 1]), np.array([0, 3])
        iota = np.linspace(0.4, 0.5, 10)
        Boozer_I, Boozer_G = np.array([0.0, 0.0]), np.array([2.0, 2.0])
        rmnc_b, zmns_b = np.ones((2, 2)), np.zeros((2, 2))
        numns_b, bmnc_b = numns, np.array([[1.0, 0.05], [1.0, 0.06]])
        s_b = np.array([0.25, 0.75])

        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def read_wout_data(self, wout):
            captured["wout"] = wout

        def run(self):
            captured["surfs"] = list(self.compute_surfs)

    def run_neo(booz, **_kwargs):
        captured["booz"] = booz
        return SimpleNamespace(eps_eff=np.array([1.0e-4, 2.0e-4]))

    monkeypatch.setattr(booz_xform_jax, "Booz_xform", FakeBoozXform)
    monkeypatch.setattr(neoclassical, "_neo_imports", lambda: (dict, run_neo))

    wout = SimpleNamespace(lasym=lasym)
    s, values = neoclassical.epsilon_effective_from_wout(
        wout, surfaces=(0.24, 0.26, 0.77), mboz=8, nboz=6)

    assert captured["init"] == {"verbose": 0, "mboz": 8, "nboz": 6}
    assert captured["wout"] is wout
    # 0.24 and 0.26 snap to the same transformed surface: three requests, two runs
    assert captured["surfs"] == [2, 7]
    booz = captured["booz"]
    assert booz["ns_b"] == 2 and booz["nfp_b"] == 3
    assert booz["mode_first"] is True and booz["asym"] is lasym
    if lasym:
        np.testing.assert_allclose(booz["pmnc_b"], -numns)
        for key in ("rmns_b", "zmnc_b", "bmns_b"):
            np.testing.assert_allclose(booz[key], .03)
    np.testing.assert_allclose(booz["pmns_b"], -numns)
    np.testing.assert_allclose(booz["iota_b"], FakeBoozXform.iota[[2, 7]])
    np.testing.assert_allclose(s, FakeBoozXform.s_b)
    np.testing.assert_allclose(values, [1e-4, 2e-4])


@pytest.mark.full
def test_epsilon_effective_matches_neo_reference():
    """The in-memory adapter retains the summary QA profile.

    These values are within 1.4% of NEO_JAX's default ``NeoConfig`` on the
    same surfaces; the earlier 16 x 16 summary grid pinned values 7-20x larger.
    """
    pytest.importorskip("neo_jax")
    script = f"""
import json
import vmex as vj
from vmex import optimize as opt
from vmex.core import neoclassical
equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file({str(DATA_DIR / 'input.LandremanPaul2021_QA_lowres')!r}))
s, values = neoclassical.epsilon_effective_from_wout(
    equilibrium.wout, surfaces=[0.2, 0.5, 0.8],
    config=neoclassical.diagnostic_neo_config())
print(json.dumps([list(map(float, s)), list(map(float, values))]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True, timeout=90)
    s, values = json.loads(completed.stdout.splitlines()[-1])
    np.testing.assert_allclose(s, [0.19387755, 0.5, 0.80612245], rtol=0, atol=1e-7)
    np.testing.assert_allclose(
        values, [1.90173919e-8, 1.09848947e-8, 1.84568599e-8], rtol=5e-5)


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


# Under the stellarator reflection (theta, zeta, Z) -> (-theta, -zeta, -Z) the
# sine partner of a cosine-led table and the cosine partner of a sine-led one
# change sign; the mirrored equilibrium has the same effective ripple.
_MIRROR = ("rmns", "zmnc", "lmnc", "bmns", "bsubumns", "bsubvmns")


@pytest.mark.full  # nightly: two solves, five NEO evaluations
def test_epsilon_effective_lasym_identities():
    """End to end through booz_xform_jax and NEO_JAX with every sine partner.

    NEO's field-line integration at the diagnostic accuracy turns round-off in
    the Boozer tables into ~1 % changes, so identities hold to 3 %; the
    asymmetry itself moves the ripple 45x (measured), far outside that.
    """
    import dataclasses
    import re

    neo_jax = pytest.importorskip("neo_jax")
    if "bmns" not in neo_jax.BoozerData.__dataclass_fields__:
        pytest.skip("NEO_JAX without sine Boozer spectra")
    import vmex as vj
    from vmex import optimize as opt

    def ripple(wout):
        return neoclassical.epsilon_effective_from_wout(
            wout, surfaces=[0.25, 0.5, 0.8], config=neoclassical.diagnostic_neo_config())[1]

    text = (DATA_DIR / "input.LandremanPaul2021_QA_lowres").read_text()
    symmetric = opt.solve_equilibrium(vj.VmecInput.from_indata_text(text), verbose=False).wout
    padded = dataclasses.replace(symmetric, lasym=True, **{
        name: 0 * np.asarray(getattr(symmetric, partner)) for name, partner in zip(
            _MIRROR, ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc"))})
    np.testing.assert_allclose(ripple(padded), ripple(symmetric), rtol=3e-2)

    text = re.sub(r"LASYM\s*=\s*\S+", "", text, flags=re.I).replace(
        "&INDATA", "&INDATA\n  LASYM = T\n  RBS(1,1) = 0.02\n  ZBC(1,1) = 0.02\n  RBS(0,1) = 0.01", 1)
    asymmetric = opt.solve_equilibrium(vj.VmecInput.from_indata_text(text), verbose=False).wout
    mirrored = dataclasses.replace(asymmetric, **{
        name: -np.asarray(getattr(asymmetric, name)) for name in _MIRROR})
    reference = ripple(asymmetric)
    np.testing.assert_allclose(ripple(mirrored), reference, rtol=3e-2)
    assert np.all(reference > 10 * ripple(dataclasses.replace(asymmetric, lasym=False)))


@pytest.mark.parametrize("trial, expected", [
    ([.8, .9, .9], True), ([.8, 1.01, .8], False),
    ([.999, .999, .999], False), ([0, .5, .5], False),
    ([float("nan"), .5, .5], False), ([.5, .5], False),
])
def test_transport_screen_rejects_radial_tradeoffs(trial, expected):
    from vmex.core.neoclassical import transport_improves

    assert transport_improves([1., 1., 1.], trial) is expected

def _cell_functions(z):
    """Return z*coth(z), z*csch(z), tanh(z/2)/z, and its integral factor."""
    z = np.asarray(z, dtype=np.complex128)
    small = np.abs(z) < 0.03
    zsafe = np.where(small, 1.0 + 0j, z)
    em = np.exp(-zsafe)
    den = -np.expm1(-2.0 * zsafe)
    dc = zsafe * (1.0 + em * em) / den
    oc = 2.0 * zsafe * em / den
    tc = -np.expm1(-zsafe) / ((1.0 + em) * zsafe)
    rc = (1.0 - 2.0 * tc) / (zsafe * zsafe)
    z2 = z * z
    d0 = 1.0 + z2 / 3.0 - z2**2 / 45.0 + 2.0 * z2**3 / 945.0
    o0 = 1.0 - z2 / 6.0 + 7.0 * z2**2 / 360.0 - 31.0 * z2**3 / 15120.0
    t0 = 0.5 - z2 / 24.0 + z2**2 / 240.0 - 17.0 * z2**3 / 40320.0
    r0 = 1.0 / 12.0 - z2 / 120.0 + 17.0 * z2**2 / 20160.0 - 31.0 * z2**3 / 362880.0
    return tuple(np.where(small, a, b) for a, b in ((d0, dc), (o0, oc), (t0, tc), (r0, rc)))


def _exact_cell_transport(pitch_edges, weight, diffusion, precession, source, *, collision_frequency):
    """Solve -nu*(p*g')' + i*m*omega*w*g=w*s_m with g(a)=g'(b)=0.

    The source uses equally spaced alpha nodes and its finite odd-grid Fourier
    interpolant; output is the alpha mean of ``Re int w conj(s) g d lambda``.
    Cells may have arbitrary positive widths and diffusion, and precession may
    change sign. Returned per-mode entries include the factor two for m>0.
    """
    edges, w, p, omega, s = map(np.asarray, (pitch_edges, weight, diffusion, precession, source))
    h = np.diff(edges)
    if s.ndim != 2 or s.shape[0] % 2 != 1 or s.shape[1] != h.size:
        raise ValueError("source must have shape (odd nalpha, ncells)")
    if not (np.all(h > 0) and np.all(w > 0) and np.all(p > 0) and collision_frequency > 0):
        raise ValueError("positive cell widths, weights, diffusion and collision_frequency required")
    if not all(np.all(np.isfinite(x)) for x in (edges, w, p, omega, s)):
        raise ValueError("finite coefficients required")
    shat = np.fft.rfft(s, axis=0) / s.shape[0]
    a = collision_frequency * p
    mode_values, nodal_solutions = [], []
    n = h.size
    for mode, sm in enumerate(shat):
        b, f = 1j * mode * omega * w, w * sm
        z = np.sqrt(b / a) * h
        dc, oc, tc, rc = _cell_functions(z)
        diag, off, cell_load = a / h * dc, -a / h * oc, f * h * tc
        band = np.zeros((3, n), dtype=np.complex128)
        band[1] = diag
        band[1, :-1] += diag[1:]
        band[0, 1:] = off[1:]
        band[2, :-1] = off[1:]
        rhs = cell_load.copy()
        rhs[:-1] += cell_load[1:]
        nodal = np.concatenate((np.zeros(1, dtype=np.complex128), solve_banded((1, 1), band, rhs)))
        integral = (nodal[:-1] + nodal[1:]) * h * tc + (f / a) * h**3 * rc
        factor = 1.0 if mode == 0 else 2.0
        mode_values.append(factor * float(np.real(np.vdot(f, integral))))
        nodal_solutions.append(nodal)
    return {"transport": float(sum(mode_values)), "modes": np.asarray(mode_values), "nodal_solutions": np.asarray(nodal_solutions)}

def _case(n=24, nalpha=9):
    edges = np.linspace(0., 1., n + 1)**1.2
    x = (edges[:-1] + edges[1:]) / 2
    alpha = np.linspace(0, 2 * np.pi, nalpha, endpoint=False)
    w = 1 + .3 * np.cos(2 * x)
    p = .5 + .7 * x
    omega = 3 * (x - .5)
    source = ((1 + x)[None, :] * np.cos(alpha[:, None])
              + .2 * np.exp(-x)[None, :] * np.sin(2 * alpha[:, None]))
    return edges, w, p, omega, source


def _interval_contains(result, reference, rtol=2e-10):
    assert bool(result["valid"])
    eps = rtol * max(abs(reference), 1e-12)
    assert float(result["lower"]) <= reference + eps
    assert float(result["upper"]) >= reference - eps
    assert float(result["lower"]) >= 0
    assert float(result["upper"]) <= float(result.get("collision_only", np.inf)) + eps


@pytest.mark.parametrize("nu", [1e-4, 1e-2, 1., 1e2])
def test_variable_coefficients_enclose_exact_cell_ode(nu):
    args = _case()
    actual = bounce_transport_bounds(*args, collision_frequency=nu)
    exact = _exact_cell_transport(*args, collision_frequency=nu)
    _interval_contains(actual, exact["transport"])
    factors = np.array([1.] + [2.] * (args[-1].shape[0] // 2))
    assert np.all(factors * np.asarray(actual["mode_lower"]) <= exact["modes"] + 1e-10)
    assert np.all(factors * np.asarray(actual["mode_upper"]) >= exact["modes"] - 1e-10)


def test_nested_spaces_tighten_both_continuum_bounds():
    args = _case(n=32)
    results = [bounce_transport_bounds(*args, collision_frequency=.003,
               trial_indices=tuple(range(0, 33, step))) for step in (8, 4, 2, 1)]
    lo = np.array([float(r["lower"]) for r in results])
    hi = np.array([float(r["upper"]) for r in results])
    assert np.all(np.diff(lo) > -1e-11)
    assert np.all(np.diff(hi) < 1e-11)
    assert hi[-1] - lo[-1] < .1 * (hi[0] - lo[0])
    exact = _exact_cell_transport(*args, collision_frequency=.003)["transport"]
    for result in results:
        _interval_contains(result, exact)


@pytest.mark.parametrize("omega", [0., 1e-9, -1., 1., 1e4])
def test_constant_precession_analytic_mixed_boundary(omega):
    n, nu, length, w, p = 24, .01, 1.3, 1.2, .7
    edges = np.linspace(0., length, n + 1)
    alpha = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    source = np.cos(alpha[:, None]) * np.ones((1, n))
    args = (edges, np.full(n, w), np.full(n, p), np.full(n, omega), source)
    exact = _exact_cell_transport(*args, collision_frequency=nu)["transport"]
    if abs(omega) < 1e-8:
        analytical = .5 * w * w * length**3 / (3 * nu * p)
    else:
        kappa = np.sqrt(1j * omega * w / (nu * p))
        analytical = .5 * w * w * np.real((length - np.tanh(kappa * length) / kappa) / (1j * omega * w))
    assert exact == pytest.approx(analytical, rel=3e-11)
    out = bounce_transport_bounds(*args, collision_frequency=nu)
    _interval_contains(out, exact, rtol=1e-7)
    if omega == 0:
        assert float(out["upper"]) == pytest.approx(exact, rel=2e-13)


def test_alpha_independent_source_uses_exact_collision_inverse():
    edges, w, p, omega, source = _case()
    source[:] = 1 + .5 * (edges[:-1] + edges[1:])[None, :]
    args = edges, w, p, omega, source
    exact = _exact_cell_transport(*args, collision_frequency=.07)["transport"]
    out = bounce_transport_bounds(*args, collision_frequency=.07, trial_indices=(0, 24))
    assert float(out["lower"]) == pytest.approx(exact, rel=5e-13)
    assert float(out["upper"]) == pytest.approx(exact, rel=5e-13)
    assert float(out["relative_gap"]) == pytest.approx(0., abs=1e-15)


def test_zero_source_and_quasisymmetric_limit():
    args = list(_case())
    args[-1] = np.zeros_like(args[-1])
    out = bounce_transport_bounds(*args, collision_frequency=.002)
    assert bool(out["valid"])
    assert float(out["lower"]) == float(out["upper"]) == float(out["relative_gap"]) == 0.


def test_exact_mean_response_adds_to_nonzero_fourier_bounds():
    """A solved oscillatory response and an exact mean retain Fourier additivity."""
    edges, _, _, omega, source = _case(n=12, nalpha=5)
    edges = np.linspace(0., 1., 13)
    args = edges, np.ones(12), np.ones(12), omega
    base = bounce_transport_bounds(*args, source, collision_frequency=.7)
    mixed = bounce_transport_bounds(*args, source + 2., collision_frequency=.7)
    expected = 4 / (3 * .7)
    for name in ("lower", "upper", "collision_only"):
        np.testing.assert_allclose(mixed[name] - base[name], expected, rtol=2.e-13)
        np.testing.assert_allclose(mixed["mode_" + name][0], expected, rtol=2.e-13)
        np.testing.assert_allclose(mixed["mode_" + name][1:], base["mode_" + name][1:], rtol=2.e-13)


def test_affine_pitch_coordinate_invariance_and_its_derivative():
    """A shrinking/relabelled pitch interval represents the same differential equation."""
    edges, w, p, omega, source = map(jnp.asarray, _case(n=12, nalpha=5))
    expected = bounce_transport_bounds(edges, w, p, omega, source, collision_frequency=.2)

    def response(log_scale):
        scale = jnp.exp(log_scale)
        return bounce_transport_bounds(.7 + scale * edges, w / scale, scale * p,
                                       omega, source, collision_frequency=.2)

    for scale in (.03, 7.):
        actual = response(jnp.log(scale))
        for key in ("lower", "upper", "collision_only", "mode_lower", "mode_upper"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=2.e-11, atol=1.e-13)
    value, gradient = jax.jit(jax.value_and_grad(lambda z: response(z)["upper"]))(jnp.log(.03))
    np.testing.assert_allclose(gradient, 0., atol=1.e-11 * abs(value))


def test_jit_vmap_and_directional_gradient_through_all_coefficients():
    edges, w, p, omega, source = map(jnp.asarray, _case(n=8, nalpha=5))

    def objective(theta):
        result = bounce_transport_bounds(
            edges * jnp.exp(theta[0]), w * jnp.exp(theta[1] * edges[1:]),
            p * jnp.exp(theta[2] * edges[1:]), omega + theta[3],
            source * jnp.exp(theta[4]) + theta[5] * jnp.cos(jnp.arange(5)[:, None] * 2 * jnp.pi / 5),
            collision_frequency=.02 * jnp.exp(theta[6]), trial_indices=(0, 2, 4, 6, 8))
        return jnp.log(result["upper"]) + .3 * jnp.log(result["lower"])

    point = jnp.asarray([.02, -.03, .01, .13, -.04, .07, .06])
    direction = jnp.asarray([.2, -.4, .3, 1., -.2, .5, -.3])
    value, grad = jax.jit(jax.value_and_grad(objective))(point)
    step = 2e-5
    fd = (objective(point + step * direction) - objective(point - step * direction)) / (2 * step)
    assert np.isfinite(float(value))
    np.testing.assert_allclose(jnp.vdot(grad, direction), fd, rtol=2e-7, atol=1e-9)
    nus = jnp.asarray([.001, .01, .1])
    def fn(nu):
        return bounce_transport_bounds(edges, w, p, omega, source, collision_frequency=nu)["upper"]
    np.testing.assert_allclose(jax.jit(jax.vmap(fn))(nus), jnp.stack([fn(nu) for nu in nus]), rtol=1e-11)


@pytest.mark.parametrize("kind", ["weight", "diffusion", "precession", "source", "edges", "nu"])
def test_invalid_physical_inputs_are_not_certified(kind):
    args = [np.array(a) for a in _case(n=8)]
    nu = .01
    if kind == "weight":
        args[1][0] = 0.
    elif kind == "diffusion":
        args[2][0] = -.1
    elif kind == "precession":
        args[3][0] = np.nan
    elif kind == "source":
        args[4][0, 0] = np.inf
    elif kind == "edges":
        args[0][2] = args[0][1]
    else:
        nu = 0.
    out = bounce_transport_bounds(*args, collision_frequency=nu)
    assert not bool(out["valid"])
    assert float(out["lower"]) == 0.
    assert np.isinf(float(out["upper"]))


@pytest.mark.parametrize("trial", [(1, 8), (0, 4, 7), (0, 5, 4, 8), (0., 4., 8.), (0,)])
def test_invalid_static_trial_space_rejected(trial):
    with pytest.raises(ValueError, match="trial_indices"):
        bounce_transport_bounds(*_case(n=8), collision_frequency=.01, trial_indices=trial)


def test_incompatible_shapes_rejected():
    args = list(_case(n=8))
    args[1] = np.ones(7)
    with pytest.raises(ValueError, match="expected edges"):
        bounce_transport_bounds(*args, collision_frequency=.01)
    args = list(_case(n=8))
    args[-1] = np.ones((4, 8))
    with pytest.raises(ValueError, match="odd number"):
        bounce_transport_bounds(*args, collision_frequency=.01)


def test_thermal_matrix_loewner_order_and_zero_flux_heat():
    energy = np.array([.2, 1., 2.5, 4., 8.])
    weights = np.array([.12, .3, .4, .1, .01])
    lower = np.array([1., 2., .6, 1.3, 2.1])
    upper = lower * np.array([1.1, 1.6, 2., 1.3, 1.2])
    actual = lower + .43 * (upper - lower)
    bounds = thermal_transport_bounds(lower, upper, energy, weights, normalization=.7)
    middle = thermal_transport_bounds(actual, actual, energy, weights, normalization=.7)
    assert bool(bounds["valid"])
    for diff in (middle["matrix_lower"] - bounds["matrix_lower"], bounds["matrix_upper"] - middle["matrix_upper"]):
        assert np.linalg.eigvalsh(diff).min() >= -1e-14
    for key in ("heat", "zero_flux_heat"):
        assert float(bounds[key + "_lower"]) <= float(middle[key + "_lower"])
        assert float(bounds[key + "_upper"]) >= float(middle[key + "_upper"])
    singular = thermal_transport_bounds(np.zeros(5), upper, energy, weights)
    assert singular["zero_flux_heat_lower"] == 0
    assert singular["valid"]
    with pytest.raises(ValueError, match="matching"):
        thermal_transport_bounds(lower, upper[:-1], energy, weights)
    assert not bool(thermal_transport_bounds(lower, upper, energy, -weights)["valid"])
    assert not bool(thermal_transport_bounds(upper, lower, energy, weights)["valid"])


def test_ambipolar_signed_interval_equals_all_corner_extrema():
    from itertools import product
    lo, hi, c = np.array([1., 2., .1]), np.array([2., 3., 1.]), np.array([1.7, -.8, -2.])
    lower, upper = ambipolar_current_interval(lo, hi, c)
    corners = [np.dot(c, np.where(choice, hi, lo)) for choice in product([False, True], repeat=3)]
    assert float(lower) == pytest.approx(min(corners))
    assert float(upper) == pytest.approx(max(corners))


@pytest.mark.parametrize("k", [.2, 1., 3.])
@pytest.mark.parametrize("frequency", [1., 5.])
@pytest.mark.parametrize("source", [[1., 0.], [0., 1.]])
def test_conserving_bounds_retain_streaming_coupled_collision_invariants(k, frequency, source):
    # Dropping the undamped first mode gives zero upper for the first source,
    # or a false positive lower for the second source.
    c, inverse = np.diag([0., frequency]), np.diag([0., 1 / frequency])
    b = np.array([[0., k], [-k, 0.]])
    s = np.array(source)
    out = conserving_transport_bounds(c, b, s, s, inverse, invariants=np.eye(2)[:, :1],
                                     constraint_rows=(0,), constraint_columns=(1,))
    exact = s @ np.linalg.solve(c + b, s)
    assert exact == pytest.approx(frequency / k**2 if source[0] else 0.)
    _interval_contains(out, exact)
    np.testing.assert_allclose([out['lower'], out['upper']], exact, atol=1e-12)
    assert float(out['conservation_defect']) < 1e-13


def _conserving_case(parameter=0., ntrial=5, slack=0.):
    c = jnp.diag(jnp.array([0., 0., 1. + .1 * parameter, 2., 3.]))
    raw = jnp.array([[0., .3, .8, .2, .5], [0., 0., .1, 1., .3],
                     [0., 0., 0., .2, -.3], [0., 0., 0., 0., .4], [0., 0., 0., 0., 0.]])
    raw = raw.at[0, 2].add(.2 * parameter).at[1, 3].add(-.15 * parameter)
    k = raw - raw.T
    s = jnp.array([.3, -.2, .8, 1. + parameter, -.4])
    inverse = jnp.diag(jnp.array([0., 0., 1. / c[2, 2] + slack, .5 + slack, 1. / 3 + slack]))
    t = jnp.eye(5)[:, :ntrial]
    return (t.T @ c @ t, k @ t, s, t.T @ s, inverse), {
        'invariants': jnp.eye(5)[:, :2], 'constraint_rows': (0, 1), 'constraint_columns': (2, 3)}, c, k


def test_conserving_rectangular_images_and_nested_majorants_enclose_full_response():
    endpoints = []
    for ntrial, slack in [(4, .6), (5, .6), (5, .1), (5, 0.)]:
        args, kwargs, c, k = _conserving_case(ntrial=ntrial, slack=slack)
        out = conserving_transport_bounds(*args, **kwargs)
        exact = np.asarray(args[2]) @ np.linalg.solve(np.asarray(c + k), np.asarray(args[2]))
        _interval_contains(out, exact)
        endpoints.append([float(out['lower']), float(out['upper'])])
    endpoints = np.asarray(endpoints)
    assert np.all(np.diff(endpoints[:, 0]) >= -1e-12)
    assert np.all(np.diff(endpoints[:, 1]) <= 1e-12)
    np.testing.assert_allclose(endpoints[-1], exact, rtol=1e-12)


def test_conserving_chart_derivative_includes_moving_constraints():
    def objective(parameter):
        args, kwargs, _, _ = _conserving_case(parameter, ntrial=4, slack=.1)
        return conserving_transport_bounds(*args, **kwargs)['upper']

    at, step = .2, 2e-5
    eager, tangent = jax.jvp(objective, (jnp.asarray(at),), (jnp.asarray(1.),))
    compiled = jax.jit(objective)(at)
    _, pullback = jax.vjp(objective, jnp.asarray(at))
    fd = (objective(at + step) - objective(at - step)) / (2 * step)
    np.testing.assert_allclose(eager, compiled, rtol=2e-13)
    np.testing.assert_allclose([tangent, pullback(jnp.asarray(1.))[0]], fd, rtol=2e-8, atol=1e-10)
    values = jax.jit(jax.vmap(objective))(jnp.array([-.1, 0., .1]))
    assert np.isfinite(values).all()


def test_conserving_redundant_residual_generators_preserve_bounds_and_derivatives():
    # Store only the Gram form of (K trial_1, ..., K trial_n, S), even
    # when these generators are dependent in the physical residual space.
    def evaluate(parameter, compressed):
        args, kwargs, _, _ = _conserving_case(parameter, slack=.1)
        if compressed:
            c, b, s, f, g = args
            image = jnp.column_stack((b, s))
            args = (c, jnp.eye(6)[:, :5], jnp.eye(6)[:, -1], f, image.T @ g @ image)
            kwargs['invariants'] = image.T @ kwargs['invariants']
        out = conserving_transport_bounds(*args, **kwargs)
        return jnp.array([out['lower'], out['upper']]), out['valid']

    for compressed in (False, True):
        values, valid = jax.jit(lambda x: evaluate(x, compressed))(.2)
        assert bool(valid)
        np.testing.assert_allclose(values, evaluate(.2, False)[0], rtol=2e-12)
    np.testing.assert_allclose(jax.jacfwd(lambda x: evaluate(x, True)[0])(.2),
                               jax.jacrev(lambda x: evaluate(x, False)[0])(.2), rtol=2e-11)


def test_conserving_complex_fourier_response_and_complete_constraints():
    c = np.diag([0., 1., 2.])
    k = np.array([[.2j, 1., .3j], [-1., .5j, .4], [.3j, -.4, -.7j]])
    s = np.array([.3j, 1., -.4 + .2j])
    inverse = np.diag([0., 1., .5])
    out = conserving_transport_bounds(c, k, s, s, inverse, invariants=np.eye(3)[:, :1],
                                     constraint_rows=(0,), constraint_columns=(1,))
    exact = np.vdot(s, np.linalg.solve(c + k, s)).real
    _interval_contains(out, exact)
    np.testing.assert_allclose([out['lower'], out['upper']], exact, rtol=2e-12)
    # A complex moment covector with otherwise real forms must retain its dtype.
    # The single trial fixed by conservation leaves no free coordinates.
    out = conserving_transport_bounds([[1.]], [[1.], [0.]], [2., 0.], [0.],
                                     [[0., 0.], [0., 1.]], invariants=[[1j], [0.]],
                                     constraint_rows=(0,), constraint_columns=(0,))
    assert bool(out['valid'])
    np.testing.assert_allclose([out['lower'], out['upper']], [0., 4.])


def test_full_coulomb_heat_bracket_and_maxwell_flux_lift():
    # Exact degree-three Landau brackets for a unit-density, unit-mass Maxwellian
    # at T=1/2. Mode order: density, parallel momentum, energy, stress, heat, l=3.
    root = np.sqrt(2 / np.pi)
    collision = root * np.array([0., 0., 0., 8 / 5, 16 / 15, 12 / 5])
    maxwell = np.array([0., 0., 0., 12., 8., 18.])
    fifth = root * np.array([0., 0., 0., 1536 / 5, 1024 / 5, 2304 / 5])
    inverse = np.diag(np.divide(fifth, maxwell**2, out=np.zeros(6), where=maxwell > 0))
    for frequency in [.1, 1., 10.]:
        out = conserving_transport_bounds([[frequency * collision[4]]], np.zeros((6, 1)),
                                         np.eye(6)[4], [1.], inverse / frequency,
                                         invariants=np.eye(6)[:, :3])
        assert bool(out['valid'])
        np.testing.assert_allclose([out['lower'], out['upper']],
                                   [15 / (16 * root * frequency), 16 * root / (5 * frequency)], rtol=2e-13)
        # The one-mode collision inverse is a continuum lower bound only.
        assert float(out['upper'] / out['lower']) == pytest.approx(512 / (75 * np.pi))
    # The two-Sonine ion heat trial recovers the standard Braginskii 125/32
    # parallel-conductivity coefficient, with ion-electron collisions omitted.
    collision = root * np.array([[16 / 15, 8 * np.sqrt(7) / 35], [8 * np.sqrt(7) / 35, 12 / 7]])
    inverse = root * np.array([[16 / 5, -16 * np.sqrt(7) / 35], [-16 * np.sqrt(7) / 35, 304 / 105]])
    out = conserving_transport_bounds(collision, np.zeros((2, 2)), [1., 0.], [1., 0.], inverse)
    assert out['valid']
    np.testing.assert_allclose((10 / 3) * root * out['lower'], 125 / 32, rtol=2.e-14)


def test_signed_goal_retains_antisymmetric_response_missed_by_polarization():
    c = np.diag([0., 1.])
    k = np.array([[0., 1.], [-1., 0.]])
    s, j = np.eye(2)
    kwargs = dict(invariants=np.eye(2)[:, :1], constraint_rows=(0,), constraint_columns=(1,))
    source = conserving_transport_bounds(c, k, s, s, c, **kwargs)
    goal = conserving_transport_bounds(c, k, j, j, c, **kwargs)
    interval = conserving_goal_interval(c, k, s, j, source, goal)
    assert bool(interval['valid'])
    np.testing.assert_allclose([interval['lower'], interval['upper']], 1., atol=1e-13)
    inverse = np.linalg.inv(c + k)
    assert j @ ((inverse + inverse.T) / 2) @ s == pytest.approx(0.)
    assert j @ inverse @ s == pytest.approx(1.)
    # Promote integer forms before C+K; int32 addition would silently overflow.
    c = np.array([[2_000_000_000, 1_000_000_000], [1_000_000_000, 2_000_000_000]], dtype=np.int32)
    k = np.array([[0, 2_000_000_000], [-2_000_000_000, 0]], dtype=np.int32)
    s, j = np.array([3.e9, 2.e9]), np.array([1., 0.])
    inverse = np.linalg.inv(c.astype(float))
    source = conserving_transport_bounds(c, k, s, s, inverse)
    goal = conserving_transport_bounds(c, k, j, j, inverse)
    interval = conserving_goal_interval(c, k, s, j, source, goal)
    assert bool(interval['valid'])
    np.testing.assert_allclose([interval['lower'], interval['upper']], 0., atol=2e-12)


def test_signed_goal_interval_encloses_independent_full_matrix_solve_and_differentiates():
    goal_vector = jnp.array([.5, .1, -.2, .7, 1.])

    def evaluate(parameter):
        args, kwargs, c, k = _conserving_case(parameter, ntrial=4, slack=.1)
        source = conserving_transport_bounds(*args, **kwargs)
        goal = conserving_transport_bounds(args[0], args[1], goal_vector, goal_vector[:4], args[4], **kwargs)
        interval = conserving_goal_interval(args[0], k[:4, :4], args[3], goal_vector[:4], source, goal)
        return interval, jnp.real(goal_vector @ jnp.linalg.solve(c + k, args[2]))

    interval, direct = evaluate(.2)
    assert bool(interval['valid'])
    assert float(interval['lower']) <= float(direct) <= float(interval['upper'])
    compiled, _ = jax.jit(evaluate)(.2)
    np.testing.assert_allclose(compiled['center'], interval['center'], rtol=1e-12)
    def objective(x):
        return evaluate(x)[0]['upper']
    step = 2e-5
    np.testing.assert_allclose(jax.grad(objective)(.2), (objective(.2 + step) - objective(.2 - step)) / (2 * step),
                               rtol=3e-8, atol=1e-9)
    args, kwargs, _, k = _conserving_case(ntrial=4, slack=.1)
    source = conserving_transport_bounds(*args, **kwargs)
    with pytest.raises(ValueError, match='same trial basis'):
        conserving_goal_interval(args[0], k, args[3], args[3], source, source)
    for bad_orbit in (np.eye(4), 1e200 * np.eye(4)):
        invalid = conserving_goal_interval(args[0], bad_orbit, args[3], args[3], source, source)
        assert not bool(invalid['valid'])
        assert np.isneginf(invalid['lower']) and np.isposinf(invalid['upper'])


def test_signed_goal_covers_weak_orbit_integration_error():
    c, k = np.diag([0., 1.]), np.array([[0., 1.], [-1., 0.]])
    s, j = np.eye(2)
    kwargs = dict(invariants=np.eye(2)[:, :1], constraint_rows=(0,), constraint_columns=(1,))
    source = conserving_transport_bounds(c, k, s, s, c, **kwargs)
    goal = conserving_transport_bounds(c, k, j, j, c, **kwargs)
    error = np.array([[.04, .02], [.01, -.03]])
    bound = np.linalg.norm(error)
    def evaluate(scale):
        return conserving_goal_interval(c, k + scale * error, s, j, source, goal,
                                        orbit_error_bound=scale * bound)
    out = jax.jit(evaluate)(1.)
    assert bool(out['valid']) and out['lower'] <= 1. <= out['upper']
    primal = source['lower_trial'] + source['upper_trial']
    adjoint = goal['lower_trial'] - goal['upper_trial']
    assert float(out['orbit_error_radius']) == pytest.approx(bound * np.linalg.norm(primal) * np.linalg.norm(adjoint))
    assert float(out['radius']) == pytest.approx(float(out['orbit_error_radius']))
    step = 1e-5
    np.testing.assert_allclose(jax.grad(lambda x: evaluate(x)['upper'])(1.),
                               (evaluate(1. + step)['upper'] - evaluate(1. - step)['upper']) / (2 * step), rtol=1e-8)
    for invalid in (-1., np.nan, np.inf):
        out = conserving_goal_interval(c, k, s, j, source, goal, orbit_error_bound=invalid)
        assert not bool(out['valid']) and np.isneginf(out['lower']) and np.isposinf(out['upper'])
    for malformed in ([.1], .1j):
        with pytest.raises(ValueError, match='real scalar'):
            conserving_goal_interval(c, k, s, j, source, goal, orbit_error_bound=malformed)


@pytest.mark.parametrize("kind", ['nan', 'negative', 'nonsymmetric', 'gauge', 'pivot', 'omitted', 'inverse',
                                 'overflow_invariants', 'overflow_inverse'])
def test_invalid_conserving_algebra_is_not_certified(kind):
    c, b, s, g = np.eye(2), np.zeros((2, 2)), np.ones(2), np.eye(2)
    f = s
    kwargs = {}
    if kind == 'nan':
        s[0] = np.nan
    elif kind == 'negative':
        c[0, 0] = -1.
    elif kind == 'nonsymmetric':
        g[0, 1] = .3
    elif kind == 'gauge':
        c[0, 0] = 0.
    elif kind in ('pivot', 'omitted'):
        b = np.array([[0., 1.], [-1., 0.]])
        kwargs = {'invariants': np.eye(2)[:, :1]}
        if kind == 'pivot':
            kwargs.update(constraint_rows=(0,), constraint_columns=(0,))
        else:
            s[:] = 0.  # The chart must be valid beyond this one source.
    elif kind == 'overflow_invariants':
        c, b, s, f, g = np.eye(1), np.array([[1.], [0.]]), np.array([1., 2.]), np.array([2.]), np.diag([0., 1.])
        kwargs = {'invariants': np.array([[1e200], [0.]])}
    elif kind == 'overflow_inverse':
        c, b, s, f = np.array([[1e-200]]), np.zeros((2, 1)), np.array([1., np.sqrt(2.)]) * 1e-100, np.array([1e-100])
        g = np.diag([-1e200, 1e200])
    else:
        g *= .1
    out = conserving_transport_bounds(c, b, s, f, g, **kwargs)
    assert not bool(out['valid'])
    assert float(out['lower']) == 0. and np.isposinf(float(out['upper']))


@pytest.mark.parametrize("kwargs", [
    {'constraint_rows': (0,), 'constraint_columns': ()},
    {'constraint_rows': (0, 0), 'constraint_columns': (0, 1)},
    {'constraint_rows': (0,), 'constraint_columns': (2,)},
    {'constraint_rows': (.5,), 'constraint_columns': (0,)},
])
def test_malformed_conserving_chart_rejected(kwargs):
    with pytest.raises(ValueError, match='constraint'):
        conserving_transport_bounds(np.eye(2), np.eye(2), np.ones(2), np.ones(2), np.eye(2),
                                    invariants=np.ones((2, 1)), **kwargs)


@pytest.mark.parametrize("argument", ['collision', 'drift', 'source', 'source_load', 'inverse_bound', 'invariants'])
def test_conserving_shape_errors_are_explicit(argument):
    args = dict(collision=np.eye(2), drift=np.eye(2), source=np.ones(2),
                source_load=np.ones(2), inverse_bound=np.eye(2), invariants=np.ones((2, 1)))
    args[argument] = np.ones(3)
    with pytest.raises(ValueError, match='expected|invariants'):
        conserving_transport_bounds(**args)


def test_integer_pitch_edges_do_not_truncate_real_physics_inputs():
    # Array-like coordinates should not silently truncate weights and sources.
    edges = np.arange(4)
    args = (edges, np.full(3, 1.2), np.full(3, .7), np.full(3, .3), np.ones((3, 3)) * .8)
    out = bounce_transport_bounds(*args, collision_frequency=.2)
    exact = _exact_cell_transport(*args, collision_frequency=.2)["transport"]
    _interval_contains(out, exact)
    assert float(out["lower"]) == pytest.approx(exact, rel=1e-12)


def test_nonfinite_thermal_normalization_is_invalid():
    assert not bool(thermal_transport_bounds([1., 2.], [2., 3.], [.5, 3.], [.2, .4], normalization=np.inf)["valid"])


def test_thermal_normalization_cannot_broadcast_into_a_nonsymmetric_matrix():
    """A vector normalization would scale columns and destroy Loewner ordering."""
    with pytest.raises(ValueError, match="normalization must be a scalar"):
        thermal_transport_bounds([1., 2.], [2., 3.], [.5, 3.], [.2, .4], normalization=[1., 2.])

_BINS = (0., .25, .5, 1., 1.5, 2.5, 4., 6., 10., 16., 24., np.inf)


def _kinetic_geometry(n=12, *, constant=False):
    """A supplied continuous-pitch model with two resolved alpha harmonics."""
    edges = np.linspace(0, 1, n + 1)
    pitch = (edges[1:] + edges[:-1]) / 2
    alpha = 2 * np.pi * np.arange(5) / 5
    if constant:
        w, p, magnetic = np.ones(n), np.ones(n), np.zeros(n)
        source = np.cos(alpha)[:, None] * np.ones((1, n))
    else:
        w, p, magnetic = 1 + .3 * pitch, .7 + .4 * pitch, 2 * (pitch - .4)
        source = (1 + .2 * pitch) * np.cos(alpha)[:, None] + .1 * np.sin(2 * alpha)[:, None]
    return dict(pitch_edges=jnp.asarray(edges), weight=jnp.asarray(w), diffusion=jnp.asarray(p),
                source_factor=jnp.asarray(source), precession_factor=jnp.asarray(magnetic),
                electric_precession_factor=jnp.asarray(.7), normalization=jnp.asarray(1.), valid=jnp.asarray(True))


def _exact_heat(geometry, nu, er):
    """Adaptive full-axis energy integration of exact continuous-cell ODEs."""
    def integrand(x):
        if x == 0:
            return 0.
        response = _exact_cell_transport(
            geometry["pitch_edges"], geometry["weight"], geometry["diffusion"],
            x * np.asarray(geometry["precession_factor"]) + er * float(geometry["electric_precession_factor"]),
            x * np.asarray(geometry["source_factor"]), collision_frequency=nu * x**-1.5)["transport"]
        return float(geometry["normalization"]) * np.sqrt(x) * np.exp(-x) * (x - 2.5)**2 * response
    pieces = [quad(integrand, left, right, epsabs=2.e-10, epsrel=2.e-10, limit=150)
              for left, right in zip((0., 2.5), (2.5, np.inf))]
    return sum(item[0] for item in pieces), sum(item[1] for item in pieces)


def test_exact_maxwell_collision_integral_and_zero_fourier_mode():
    """The full collision-only heat moment is 270, including the infinite tail."""
    geometry = _kinetic_geometry(constant=True)
    for bins in ((0., np.inf), _BINS):
        result = continuous_lorentz_heat_bounds(
            geometry, temperature_ev=.5, collision_frequency=2., radial_electric_field=0., energy_edges=bins)
        assert result["valid"]
        # ∫mean_alpha(S A0^-1 S)dλ=1/6 for cos(alpha), w=p=1 and unit interval.
        np.testing.assert_allclose(result["upper"], 270 / (6 * 2), rtol=3.e-14)
        np.testing.assert_allclose(result["collision_only"], result["upper"], rtol=1.e-14)
        np.testing.assert_allclose(jnp.sum(result["bin_upper"]), result["upper"])
        assert result["lower"] <= result["upper"]
    # For the alpha-independent component, all precession disappears exactly.
    mean_source = geometry | dict(source_factor=jnp.ones_like(geometry["source_factor"]))
    exact = continuous_lorentz_heat_bounds(
        mean_source, temperature_ev=.5, collision_frequency=2., radial_electric_field=5.)
    np.testing.assert_allclose(exact["lower"], 45., rtol=3.e-14)
    np.testing.assert_allclose(exact["upper"], 45., rtol=3.e-14)


def test_exact_mean_heat_adds_without_a_zero_mode_trial_solve():
    """The mean Maxwell response adds independently of precessing harmonics."""
    geometry = _kinetic_geometry(8, constant=True)

    def response(mean):
        varied = geometry | {"source_factor": geometry["source_factor"] + mean}
        return continuous_lorentz_heat_bounds(
            varied, temperature_ev=.5, collision_frequency=.7,
            radial_electric_field=1.3, energy_edges=_BINS)

    base, mixed = response(0.), response(2.)
    for key in ("lower", "upper", "collision_only"):
        np.testing.assert_allclose(mixed[key] - base[key], 270 * 4 / (3 * .7), rtol=3.e-13)
    gradient = jax.jit(jax.grad(lambda mean: response(mean)["upper"]))(2.)
    np.testing.assert_allclose(gradient, 270 * 4 / (3 * .7), rtol=3.e-13)


def test_full_maxwell_bounds_preserve_affine_pitch_coordinates():
    """Continuous-energy certificates and AD are invariant to pitch reparameterization."""
    geometry = _kinetic_geometry(8)
    options = dict(temperature_ev=.5, collision_frequency=.7, radial_electric_field=1.3, energy_edges=_BINS)
    expected = continuous_lorentz_heat_bounds(geometry, **options)

    def response(log_scale):
        scale = jnp.exp(log_scale)
        varied = geometry | {"pitch_edges": .7 + scale * geometry["pitch_edges"],
                             "weight": geometry["weight"] / scale,
                             "diffusion": geometry["diffusion"] * scale}
        return continuous_lorentz_heat_bounds(varied, **options)

    for scale in (.03, 7.):
        actual = response(jnp.log(scale))
        for key in ("lower", "upper", "collision_only", "bin_lower", "bin_upper"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=2.e-11, atol=1.e-13)
    value, gradient = jax.jit(jax.value_and_grad(lambda z: response(z)["upper"]))(jnp.log(.03))
    np.testing.assert_allclose(gradient, 0., atol=1.e-11 * abs(value))


@pytest.mark.parametrize("nu,er", [(.1, -1.), (1., 0.), (10., 2.)])
def test_bounds_enclose_independent_continuous_pitch_and_energy_reference(nu, er):
    """An independent ODE solve and adaptive integral lie inside the interval."""
    geometry = _kinetic_geometry()
    result = continuous_lorentz_heat_bounds(
        geometry, temperature_ev=.5, collision_frequency=nu, radial_electric_field=er, energy_edges=_BINS)
    reference, error = _exact_heat(geometry, nu, er)
    assert result["valid"]
    assert float(result["lower"]) <= reference + 5 * error
    assert float(result["upper"]) >= reference - 5 * error
    assert result["upper"] <= result["collision_only"] * (1 + 1.e-12)


def test_nested_energy_and_pitch_trial_enrichment_tightens_continuous_bounds():
    """Energy refinement is a variational enrichment, not a changed velocity rule."""
    geometry = _kinetic_geometry()
    common = dict(temperature_ev=.5, collision_frequency=.5, radial_electric_field=-1.)
    coarse = continuous_lorentz_heat_bounds(geometry, energy_edges=_BINS, trial_indices=(0, 4, 8, 12), **common)
    pitch_fine = continuous_lorentz_heat_bounds(geometry, energy_edges=_BINS, **common)
    finite = np.asarray(_BINS[:-1])
    extra = (finite[:-1] + finite[1:]) / 2
    fine_bins = tuple(np.sort(np.r_[finite, extra, 32., np.inf]))
    both_fine = continuous_lorentz_heat_bounds(geometry, energy_edges=fine_bins, **common)
    assert coarse["lower"] <= pitch_fine["lower"] <= both_fine["lower"]
    assert coarse["upper"] >= pitch_fine["upper"] >= both_fine["upper"]
    assert both_fine["relative_gap"] < coarse["relative_gap"]
    np.testing.assert_allclose(coarse["collision_only"], both_fine["collision_only"], rtol=2.e-13)


def test_continuous_bound_gradients_and_charge_field_reversal():
    """JIT reverse/forward derivatives agree with finite differences and symmetry."""
    geometry = _kinetic_geometry(8)

    def objective(parameters):
        varied = geometry | {"source_factor": parameters[0] * geometry["source_factor"]}
        return continuous_lorentz_heat_bounds(
            varied, temperature_ev=.5, collision_frequency=parameters[2],
            radial_electric_field=parameters[1], energy_edges=_BINS)["upper"]

    evaluate = jax.jit(jax.value_and_grad(objective))
    point = jnp.asarray([1., .8, .7])
    value, gradient = evaluate(point)
    np.testing.assert_allclose(gradient[0], 2 * value, rtol=2.e-11)
    step = 1.e-5
    for axis in (1, 2):
        plus, _ = evaluate(point.at[axis].add(step))
        minus, _ = evaluate(point.at[axis].add(-step))
        np.testing.assert_allclose(gradient[axis], (plus - minus) / (2 * step), rtol=1.e-6)
    tangent = jnp.asarray([.2, .4, -.6])
    _, forward = jax.jvp(objective, (point,), (tangent,))
    np.testing.assert_allclose(forward, gradient @ tangent, rtol=2.e-11)
    reversed_case = continuous_lorentz_heat_bounds(
        geometry, temperature_ev=.5, collision_frequency=.7, radial_electric_field=-.8,
        charge_number=-1., energy_edges=_BINS)
    np.testing.assert_allclose(value, reversed_case["upper"], rtol=1.e-12)
    batched = jax.jit(jax.vmap(lambda nu: continuous_lorentz_heat_bounds(
        geometry, temperature_ev=.5, collision_frequency=nu, radial_electric_field=0.,
        energy_edges=_BINS)["upper"]))(jnp.asarray([.3, 1., 3.]))
    assert np.all(np.isfinite(batched))


@pytest.mark.parametrize("options", [
    {"temperature_ev": 0.}, {"collision_frequency": -1.}, {"radial_electric_field": np.nan},
    {"charge_number": 0.},
])
def test_invalid_operating_point_returns_infinite_upper(options):
    """Rejected scenarios cannot become zero-valued optimization candidates."""
    arguments = dict(temperature_ev=.5, collision_frequency=1., radial_electric_field=0.) | options
    result = continuous_lorentz_heat_bounds(_kinetic_geometry(), **arguments)
    assert not result["valid"] and result["lower"] == 0 and jnp.isinf(result["upper"])


@pytest.mark.parametrize("change", [
    {"valid": False}, {"normalization": -1.}, {"weight": jnp.ones(12).at[0].set(-1.)},
    {"source_factor": jnp.full((5, 12), np.nan)},
    {"pitch_edges": jnp.zeros(13)},
])
def test_invalid_geometry_is_not_certified(change):
    """Coefficient positivity and geometry validity are part of the contract."""
    result = continuous_lorentz_heat_bounds(
        _kinetic_geometry() | change, temperature_ev=.5, collision_frequency=1., radial_electric_field=0.)
    assert not result["valid"] and jnp.isinf(result["upper"])


@pytest.mark.parametrize("bins", [(0., 1.), (-1., np.inf), (0., .5, .4, np.inf),
                                   (0., np.nan, np.inf), [[0., 1.], [2., np.inf]],
                                   (0., 1000., np.inf)])
def test_energy_grid_must_cover_the_full_axis_with_representable_moments(bins):
    """Tail omission and unrepresentable energy moments raise instead of truncating."""
    with pytest.raises(ValueError, match="energy"):
        continuous_lorentz_heat_bounds(
            _kinetic_geometry(), temperature_ev=.5, collision_frequency=1., radial_electric_field=0., energy_edges=bins)


@pytest.mark.parametrize("change", [
    {"normalization": jnp.ones(2)}, {"weight": jnp.ones(11)},
    {"source_factor": jnp.ones((4, 12))},
])
def test_malformed_coefficient_shapes_raise(change):
    """Array broadcasting cannot silently redefine the kinetic model."""
    with pytest.raises(ValueError):
        continuous_lorentz_heat_bounds(
            _kinetic_geometry() | change, temperature_ev=.5, collision_frequency=1., radial_electric_field=0.)


def test_lost_positive_moment_gram_is_reported(monkeypatch):
    """A failed numerical Gram factor is reported, never repaired by clipping."""
    from vmex.core import neoclassical as module

    def fail_cholesky(matrix):
        raise np.linalg.LinAlgError("test the guard for an unresolved moment Gram matrix")

    monkeypatch.setattr(module.np.linalg, "cholesky", fail_cholesky)
    with pytest.raises(ValueError, match="positive definiteness"):
        continuous_lorentz_heat_bounds(
            _kinetic_geometry(), temperature_ev=.5, collision_frequency=1., radial_electric_field=0.,
            energy_edges=(0., 1.23456789, np.inf))

def _spectrum(chart="qi"):
    """A weak nonsymmetric perturbation of a single-well Boozer spectrum."""
    m, n, hm, hn = {"qi": ([0, 0, 1], [0, 2, 0], 0, 1),
                     "qa": ([0, 1, 1], [0, 0, 2], 1, 0),
                     "qh": ([0, 1, 1], [0, 4, 0], 1, 2)}[chart]
    return dict(bmnc_b=jnp.asarray([[1., .1, .001], [1.01, .11, .002], [1.02, .12, .003]]),
                xm_b=jnp.asarray(m), xn_b=jnp.asarray(n), iota_b=jnp.asarray([.4, .41, .42]),
                G_b=jnp.asarray([4.9, 5., 5.1]), I_b=jnp.zeros(3), nfp=2,
                psi_b=jnp.asarray([.4, .5, .6]), s_b=jnp.asarray([.4, .5, .6]),
                nalpha=5, points_per_period=32, quadrature_order=16,
                helicity_m=hm, helicity_n=hn, well_center=np.pi)


def _orbit_geometry(chart="qi", **overrides):
    """Full reference-sector coefficients, with optionally changed inputs."""
    return neoclassical.bounce_geometry_from_boozer(
        **(_spectrum(chart) | dict(pitch_fractions=jnp.linspace(0., 1., 9)) | overrides))


@pytest.mark.parametrize("chart", ["qi", "qa", "qh"])
def test_reference_chart_geometry_and_full_pitch_sector(chart):
    """Aligned charts preserve reference wells and the Clebsch multiplier."""
    geometry = _orbit_geometry(chart)
    assert bool(geometry["valid"]) and bool(geometry["alpha_alias_free"])
    np.testing.assert_allclose(geometry["pitch_band_fraction"], 1., atol=1.e-15)
    np.testing.assert_allclose(geometry["source_factor"].mean(axis=0), 0., atol=1.e-18)
    assert np.all(geometry["weight"] > 0) and np.all(geometry["diffusion"] > 0)
    assert np.any(np.abs(geometry["precession_factor"]) > 0)
    assert 0 < geometry["reference_perturbation_rms"] < .01
    c = 1. if chart == "qi" else -2 / (.41 - (4 if chart == "qh" else 0))
    np.testing.assert_allclose(geometry["chart_factor"], c)
    np.testing.assert_allclose(geometry["electric_precession_factor"], -c / np.sqrt(2))
    np.testing.assert_allclose(geometry["normalization"] * geometry["orbit_measure"], 1 / np.sqrt(np.pi))
    exact_reference = _spectrum(chart)
    exact_reference["bmnc_b"] = exact_reference["bmnc_b"].at[:, 2].set(0.)
    exact = _orbit_geometry(chart, **exact_reference)
    np.testing.assert_allclose(exact["source_factor"], 0., atol=2.e-15)


def test_seed_grid_matches_full_sector_and_supports_partial_physical_band():
    """Seed grids make full and restricted model coverage explicit."""
    inputs = _spectrum()
    grid = neoclassical.reference_pitch_grid_from_boozer(**inputs, ncells=8)
    assert len(grid["pitch_edges"]) == 9
    np.testing.assert_allclose(grid["well_center"], np.pi)
    np.testing.assert_allclose(grid["pitch_fractions"][[0, -1]], [0, 1])
    geometry = neoclassical.bounce_geometry_from_boozer(**inputs, pitch_edges=grid["pitch_edges"])
    np.testing.assert_allclose(geometry["weight"], _orbit_geometry()["weight"])
    partial = neoclassical.reference_pitch_grid_from_boozer(**inputs, ncells=6, pitch_margin=.1)
    geometry = neoclassical.bounce_geometry_from_boozer(**inputs, pitch_edges=partial["pitch_edges"])
    np.testing.assert_allclose(geometry["pitch_band_fraction"], .8)
    assert geometry["valid"]
    without_s = {k: v for k, v in inputs.items() if k != "s_b"}
    with_edge = neoclassical.bounce_geometry_from_boozer(**without_s, psi_edge=1., pitch_edges=grid["pitch_edges"])
    np.testing.assert_allclose(with_edge["source_factor"], _orbit_geometry()["source_factor"])
    signed = _orbit_geometry(psi_b=-inputs["psi_b"])
    np.testing.assert_allclose(signed["source_factor"], -_orbit_geometry()["source_factor"])


@pytest.mark.parametrize("sign", [-1., 1.])
def test_radius_stencil_reproduces_first_and_second_order_reference_terms(sign):
    """A separable quadratic reference has an exact action derivative in the radius chart."""
    radius = jnp.asarray([.25, .4, .6])
    amplitude = 1 + .3 * radius + .1 * radius**2
    g = 5 + .2 * radius**2
    inputs = _spectrum("qa") | dict(
        bmnc_b=jnp.stack((amplitude, .1 * amplitude, .002 * radius**2), axis=1),
        iota_b=jnp.full(3, .4), G_b=g, psi_b=sign * 2 * radius**2, s_b=radius**2)
    options = inputs | dict(pitch_fractions=jnp.linspace(0, 1, 9))
    actual = neoclassical.bounce_geometry_from_boozer(**options, radial_interpolation="radius")
    assert actual["valid"]
    action, tau = actual["reference_action"], 2 * actual["weight"]
    # For B=a(r)b(chi), the fixed-lambda identity Jr=(C'/C)J-(a'/a)(J+tau)/2
    # remains exact even when the orbit line element is interpolated separately.
    expected = actual["dr_dpsi"] * (
        .4 * radius[1] / g[1] * action
        - (.3 + .2 * radius[1]) / amplitude[1] * (action + tau) / 2)
    np.testing.assert_allclose(actual["radial_action_derivative"], expected, rtol=3.e-11, atol=3.e-11)
    legacy = neoclassical.bounce_geometry_from_boozer(**options)
    explicit = neoclassical.bounce_geometry_from_boozer(**options, radial_interpolation="flux")
    np.testing.assert_array_equal(legacy["precession_factor"], explicit["precession_factor"])
    assert not np.allclose(legacy["radial_action_derivative"], expected, rtol=1.e-3)
    grid = neoclassical.reference_pitch_grid_from_boozer(**inputs, ncells=8, radial_interpolation="radius")
    np.testing.assert_array_equal(grid["pitch_fractions"], options["pitch_fractions"])


def test_radius_stencil_rejects_inconsistent_flux_labels():
    """The chain rule assumes normalized flux is proportional to signed toroidal flux."""
    actual = _orbit_geometry(radial_interpolation="radius", psi_b=jnp.asarray([.4, .5, .61]))
    assert not actual["valid"]


def test_perturbation_diagnostic_resolves_shallow_reference_wells():
    """Small delta B/B does not imply small perturbations compared with trapping depth."""
    actual = _orbit_geometry()
    np.testing.assert_allclose(actual["reference_well_depth"], .22, rtol=1.e-13)
    np.testing.assert_allclose(actual["reference_perturbation_well_rms"], .002 / (.22 * np.sqrt(2)), rtol=1.e-12)
    np.testing.assert_allclose(actual["reference_perturbation_well_max"], .002 / .22, rtol=1.e-12)
    shallow = _orbit_geometry(bmnc_b=_spectrum()["bmnc_b"].at[:, 1].multiply(.01))
    np.testing.assert_allclose(shallow["reference_perturbation_well_max"],
                               100 * actual["reference_perturbation_well_max"], rtol=1.e-12)


def test_thermal_rule_preserves_positive_intervals_and_speed_outputs():
    """Positive quadrature carries monoenergetic bounds into thermal matrices."""
    geometry = _orbit_geometry()
    result = neoclassical.neoclassical_heat_bounds_from_geometry(
        geometry, temperature_ev=1000., collision_frequencies=[1., 10.],
        radial_electric_fields=[-100., 0., 100.], nenergy=3)
    assert result["upper"].shape == (2, 3)
    assert result["mono_upper"].shape == (2, 3, 3)
    assert np.all(result["valid"])
    assert np.all(result["upper"] >= result["lower"])
    assert np.all(result["collision_only"] >= result["upper"] - 1.e-10)
    np.testing.assert_allclose(result["collision_factors"], result["energy"]**-1.5)
    weights = result["normalization"] * result["energy_weights"] * (result["energy"] - 2.5)**2
    np.testing.assert_allclose(result["upper"], jnp.sum(result["mono_upper"] * weights, axis=-1))
    np.testing.assert_allclose(result["lower"], result["matrix_lower"][..., 1, 1])
    assert np.all(result["zero_flux_heat_upper"] >= result["zero_flux_heat_lower"] - 1.e-12)
    assert np.all(result["zero_flux_heat_upper"] <= result["upper"] + 1.e-12)
    opposite_charge = neoclassical.neoclassical_heat_bounds_from_geometry(
        geometry, temperature_ev=1000., charge_number=-1., collision_frequencies=[1., 10.],
        radial_electric_fields=[100., 0., -100.], nenergy=3)
    np.testing.assert_allclose(opposite_charge["upper"], result["upper"], rtol=2.e-12)


def test_collision_law_and_electric_precession_have_expected_limits():
    """The zero-precession case has 1/nu scaling and electric drift suppresses it."""
    geometry = _orbit_geometry() | {"precession_factor": jnp.zeros(8)}
    result = neoclassical.neoclassical_heat_bounds_from_geometry(
        geometry, temperature_ev=1000., collision_frequencies=[1., 10.],
        radial_electric_fields=[0., 100.], nenergy=3, collision_factors=jnp.ones(3))
    np.testing.assert_allclose(result["upper"][0, 0], 10 * result["upper"][1, 0], rtol=1.e-13)
    np.testing.assert_allclose(result["lower"][0, 0], 10 * result["lower"][1, 0], rtol=1.e-13)
    np.testing.assert_allclose(result["upper"][:, 0], result["collision_only"][:, 0], rtol=1.e-13)
    assert np.all(result["upper"][:, 1] < result["lower"][:, 0])


@pytest.mark.parametrize("radial_interpolation", ["flux", "radius"])
def test_shape_and_perturbation_gradients_include_moving_reference_pitch(radial_interpolation):
    """AD matches amplitude homogeneity and independent shape finite differences."""
    inputs = _spectrum()

    def upper(parameters):
        spectrum = inputs["bmnc_b"].at[:, 1].multiply(parameters[0]).at[:, 2].multiply(parameters[1])
        result = neoclassical.neoclassical_heat_bounds_from_boozer(
            **(inputs | {"bmnc_b": spectrum}), pitch_fractions=jnp.linspace(0., 1., 9),
            temperature_ev=1000., collision_frequencies=[1.], radial_electric_fields=[30.], nenergy=3,
            radial_interpolation=radial_interpolation)
        return jnp.sum(result["upper"])

    evaluate = jax.jit(jax.value_and_grad(upper))
    point = jnp.ones(2)
    value, gradient = evaluate(point)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    np.testing.assert_allclose(gradient[1], 2 * value, rtol=1.e-10)
    h = 1.e-6
    plus, _ = evaluate(point.at[0].add(h))
    minus, _ = evaluate(point.at[0].add(-h))
    np.testing.assert_allclose(gradient[0], (plus - minus) / (2 * h), rtol=1.e-5)
    tangent = jnp.asarray([.3, -.7])
    _, jvp = jax.jit(lambda x, v: jax.jvp(upper, (x,), (v,)))(point, tangent)
    np.testing.assert_allclose(jvp, gradient @ tangent, rtol=1.e-11)


@pytest.mark.parametrize("change", [
    {"pitch_fractions": jnp.linspace(-.2, .8, 9)},
    {"psi_b": jnp.asarray([.4, .4, .6])},
    {"minor_radius": -1.},
    {"s_b": jnp.asarray([.4, -.5, .6])},
    {"bmnc_b": jnp.asarray([[1., 0., .001], [1., 0., .002], [1., 0., .003]])},
    {"bmnc_b": jnp.full((3, 3), jnp.nan)},
    {"max_wells": 1},
])
def test_invalid_geometry_is_rejected_by_the_optimization_objective(change):
    """Invalid input never looks like a successful zero-transport candidate."""
    geometry = _orbit_geometry(**change)
    assert not geometry["valid"]
    result = neoclassical.neoclassical_heat_bounds_from_geometry(
        geometry, temperature_ev=1000., collision_frequencies=1., radial_electric_fields=0., nenergy=2)
    assert not result["valid"][0, 0]
    assert result["lower"][0, 0] == 0 and jnp.isinf(result["upper"][0, 0])
    assert jnp.isinf(neoclassical.smooth_worst_case_bound(result["upper"]))


def test_geometry_rejects_multiple_wells_and_resonant_helical_chart():
    """The one-well model cannot silently discard a second family."""
    double = _orbit_geometry(xn_b=jnp.asarray([0, 4, 0]))
    assert not double["valid"]
    resonant = _orbit_geometry("qa", iota_b=jnp.zeros(3))
    assert not resonant["valid"]
    aliased = _orbit_geometry(xm_b=jnp.asarray([0, 0, 3]))
    assert not aliased["alpha_alias_free"]
    assert not aliased["valid"]
    # A deliberately aliased interpolant model requires explicit opt-in.
    allowed = _orbit_geometry(xm_b=jnp.asarray([0, 0, 3]), allow_alpha_aliasing=True)
    assert allowed["valid"] and not allowed["alpha_alias_free"]


@pytest.mark.parametrize("change", [
    {"charge_number": 0.}, {"temperature_ev": -10.},
    {"collision_frequencies": [0.]}, {"radial_electric_fields": [jnp.nan]},
    {"collision_factors": [-1., 1., 1.]},
])
def test_invalid_species_and_scenarios_return_infinite_upper(change):
    """Species and operating-point failures propagate to the interval validity."""
    options = dict(temperature_ev=1000., collision_frequencies=[1.], radial_electric_fields=[0.], nenergy=3)
    result = neoclassical.neoclassical_heat_bounds_from_geometry(_orbit_geometry(), **(options | change))
    assert not result["valid"][0, 0] and jnp.isinf(result["upper"][0, 0])


def test_smooth_max_is_conservative_differentiable_and_rejects_invalid_cases():
    """The robust objective preserves the sampled maximum with a known offset."""
    values = jnp.asarray([.2, .5, 1.])
    total, gradient = jax.value_and_grad(neoclassical.smooth_worst_case_bound)(values, scale=.5, smoothing=.1)
    assert 2 <= total <= 2 + .1 * np.log(3)
    assert np.all(gradient >= 0)
    np.testing.assert_allclose(jnp.sum(gradient), 2.)
    for bad in (jnp.asarray([jnp.inf]), jnp.asarray([-1.]), jnp.asarray([jnp.nan])):
        assert jnp.isinf(neoclassical.smooth_worst_case_bound(bad))
    for options in ({"scale": 0.}, {"smoothing": -1.}):
        assert jnp.isinf(neoclassical.smooth_worst_case_bound(values, **options))


def test_state_and_class_compose_with_vmex_residual_contract(monkeypatch):
    """The state adapter snaps radial stencils and preserves surface/scenario axes."""
    inputs = _spectrum()
    fake_boozer = {k: v for k, v in inputs.items() if k in
                  ("bmnc_b", "xm_b", "xn_b", "iota_b", "G_b", "I_b", "nfp", "psi_b", "s_b")}
    fake_boozer["bmns_b"] = jnp.zeros_like(fake_boozer["bmnc_b"])
    rt = SimpleNamespace(setup=SimpleNamespace(s_full=jnp.linspace(0, 1, 11)))
    observed = []

    def transform(state, runtime, *, surfaces, **kwargs):
        observed.append(surfaces)
        return fake_boozer

    monkeypatch.setattr(neoclassical, "boozer_spectrum_state", transform)
    monkeypatch.setattr(neoclassical, "_aspect_scalars", lambda state, runtime: (jnp.asarray(1.), None, None, None))
    options = dict(temperature_ev=1000., collision_frequencies=[1.], radial_electric_fields=[0.],
                   nenergy=2, nalpha=5, points_per_period=32, quadrature_order=16)
    term = neoclassical.NeoclassicalHeatBound([.5], pitch_fractions=np.linspace(0, 1, 9), scale=1., **options)
    eq = SimpleNamespace(state=object(), runtime=rt)
    result = term.compute_state(eq.state, rt)
    assert result["upper"].shape == (1, 1, 1) and result["mono_upper"].shape == (1, 1, 1, 2)
    np.testing.assert_allclose(observed[0], [.35, .45, .55])
    np.testing.assert_allclose(term(eq)**2, result["total"])
    np.testing.assert_allclose(term.total(eq), result["total"])
    fixed = neoclassical.NeoclassicalHeatBound([.5], result["geometry"]["pitch_edges"], **options)
    np.testing.assert_allclose(fixed.total(eq), term.total(eq), rtol=1.e-12)
    with pytest.raises(ValueError, match="neighboring half-mesh"):
        neoclassical.neoclassical_heat_bounds_state(eq.state, rt, surfaces=[.01], pitch_fractions=[0., 1.], **options)
    with pytest.raises(ValueError, match="one row per target"):
        neoclassical.neoclassical_heat_bounds_state(eq.state, rt, surfaces=[.5], pitch_edges=np.ones((2, 4)), **options)


@pytest.mark.parametrize("change", [
    {"helicity_m": 2}, {"nalpha": 4}, {"points_per_period": 15},
    {"num_periods": 1}, {"nfp": 0}, {"bmnc_b": jnp.ones((2, 3))},
    {"radial_interpolation": "unknown"},
    {"pitch_fractions": jnp.ones((2, 3))}, {"s_b": jnp.ones(2)},
    {"pitch_edges": [.9, 1.], "pitch_fractions": [0., 1.]},
])
def test_geometry_shape_errors_are_explicit(change):
    """Malformed API inputs raise before a misleading numerical result."""
    with pytest.raises(ValueError):
        _orbit_geometry(**change)


def test_constructor_and_quadrature_validation():
    """Static shape and grid contracts fail with useful exceptions."""
    inputs = _spectrum()
    for options in ({"ncells": 0}, {"pitch_margin": .5}):
        with pytest.raises(ValueError):
            neoclassical.reference_pitch_grid_from_boozer(**inputs, **options)
    with pytest.raises(ValueError, match="positive magnetic well"):
        neoclassical.reference_pitch_grid_from_boozer(**(inputs | {"bmnc_b": jnp.zeros((3, 3))}))
    with pytest.raises(ValueError, match="s_b or psi_edge"):
        neoclassical.bounce_geometry_from_boozer(**{k: v for k, v in inputs.items() if k != "s_b"}, pitch_edges=[.95, 1.])
    with pytest.raises(ValueError, match="exactly one"):
        neoclassical.bounce_geometry_from_boozer(**inputs)
    base = dict(temperature_ev=1000., collision_frequencies=[1.], radial_electric_fields=[0.])
    for options in ({"nenergy": 1}, {"collision_factors": [1.]}, {"temperature_ev": [1., 2.]},
                    {"collision_frequencies": []}, {"radial_electric_fields": [[0.]]}):
        with pytest.raises(ValueError):
            neoclassical.neoclassical_heat_bounds_from_geometry(_orbit_geometry(), **(base | options))
    for options in ({"surfaces": []}, {"surfaces": [0.]}, {"pitch_fractions": [1., 0.]},
                    {"pitch_fractions": [-1., 1.]}, {"scale": 0.}, {"smoothing": 0.},
                    {"pitch_fractions": None}):
        with pytest.raises(ValueError):
            neoclassical.NeoclassicalHeatBound(**(base | dict(surfaces=[.5], pitch_fractions=[0., 1.]) | options))
    for options, count in (({}, 17), ({"helicity_m": 1, "helicity_n": 0}, 17),
                           ({"helicity_m": 1, "helicity_n": -1}, 33)):
        term = neoclassical.NeoclassicalHeatBound([.5], pitch_fractions=[0., 1.], **(base | options))
        assert term.options["nalpha"] == count


def test_public_exports_resolve_to_the_traceable_adapter():
    """Public VMEX and optimize imports expose the same objective and kernels."""
    import vmex
    from vmex import optimize

    for name in ("NeoclassicalHeatBound", "ConservingHeatBound", "neoclassical_heat_bounds_state"):
        assert getattr(vmex, name) is getattr(neoclassical, name)
        assert getattr(optimize, name) is getattr(neoclassical, name)
    assert vmex.neoclassical_heat_bounds_from_boozer is neoclassical.neoclassical_heat_bounds_from_boozer
    assert vmex.conserving_stellarator_proxy is neoclassical.conserving_stellarator_proxy


def test_full_coulomb_proxy_encloses_independent_angular_quadrature_and_has_gradients():
    """A dense quadrature and scalar minimizer independently check the feasible flow."""
    from scipy.optimize import minimize_scalar

    modes = np.array([[0, 0], [1, 0], [0, 1], [1, -1]])
    coefficients = jnp.asarray([1., .18, .013j, .006])
    options = dict(iota=.41, G_b=3., I_b=.02, nfp=2, collision_frequency=.3, electric_field=.01)
    grid = np.arange(257) * (2 * np.pi / 257)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    phase = np.exp(1j * (x[..., None] * modes[:, 0] + y[..., None] * modes[:, 1]))
    kd = modes @ np.array([.41, 2.])
    ke = modes @ np.array([3., -.04]) / (3. + .41 * .02)
    primitive = np.divide(ke, kd, out=np.zeros_like(kd), where=kd != 0) * coefficients
    J = np.real(phase @ coefficients)
    p, dj, ej, ep = [np.real(phase @ v) for v in
                     (primitive, 1j * kd * coefficients, 1j * ke * coefficients, 1j * ke * primitive)]
    s = np.sqrt(2 / np.pi)

    def objective(c):
        P = p + c
        return (.3 * 4 * s / 3 * np.mean(P**2)
                + 9 * (9368 * s / 2205) / (16 * .3) * np.mean((J * ej - P * dj)**2 / J**3)
                + .01**2 * 4 * s / .3 * np.mean((J * ep - .5 * P * ej)**2 / J**4))

    reference = minimize_scalar(objective, bracket=(-1., 1.), method="brent", options={"xtol": 1.e-12})
    coarse = neoclassical.conserving_stellarator_proxy(modes, coefficients, integration_order=0, **options)
    fine = neoclassical.conserving_stellarator_proxy(modes, coefficients, integration_order=24, **options)
    assert coarse["valid"] and fine["valid"] and coarse["upper"] > reference.fun
    np.testing.assert_allclose(fine["upper"], reference.fun, rtol=2.e-12)
    np.testing.assert_allclose(fine["optimal_heat_flow_constant"], reference.x, rtol=1.e-7)
    assert fine["lower"] == 0

    def response(values):
        return neoclassical.conserving_stellarator_proxy(
            modes, coefficients.at[1].set(values[0]), rho_bound=.3,
            **(options | dict(collision_frequency=values[1], electric_field=values[2])))["upper"]

    parameters, direction, step = jnp.array([.18, .3, .01]), jnp.array([.2, -.3, .04]), 1.e-5
    gradient = jax.jit(jax.grad(response))(parameters)
    finite = (response(parameters + step * direction) - response(parameters - step * direction)) / (2 * step)
    np.testing.assert_allclose(gradient @ direction, finite, rtol=3.e-8)


def test_full_coulomb_proxy_zero_source_symmetries_and_chart_rejection():
    """Zero drive, field reversal, broadcasting and invalid Fourier charts are explicit."""
    proxy = neoclassical.conserving_stellarator_proxy
    wave, c = np.array([[0, 0], [1, 0]]), jnp.array([1., .2])
    args = dict(iota=.4, G_b=1., I_b=0.)
    out = proxy(wave, c, collision_frequency=jnp.array([.1, 1.])[:, None],
                electric_field=jnp.array([-.01, 0., .01])[None, :], **args)
    assert np.all(out["valid"]) and out["upper"].shape == (2, 3)
    np.testing.assert_allclose(out["upper"][:, 0], out["upper"][:, 2])
    assert np.all(out["lower"][:, 1] > 0) and np.all(out["lower"][:, (0, 2)] == 0)
    uniform = proxy([[0, 0]], [1], **args)
    assert uniform["valid"] and uniform["upper"] == uniform["lower"] == uniform["relative_gap"] == 0
    overflow = proxy([[0, 0]], [1.], iota=1.e308, G_b=1., I_b=1.e308)
    assert not overflow["valid"] and jnp.isinf(overflow["upper"])
    for update in (dict(iota=0.), dict(G_b=0.), dict(collision_frequency=0.), dict(electric_field=np.nan),
                   dict(heat_inverse_bound=-1.), dict(rho_bound=.1), dict(rho_bound=1.)):
        invalid = proxy(wave, c, **(args | update))
        assert not invalid["valid"] and invalid["lower"] == 0 and jnp.isinf(invalid["upper"])
    for waves, coefficients in (([[1, 0]], [1.]), ([[0, 0], [0, 0]], [1., 1.]),
                                 ([[0, 0], [.5, 0]], [1., .2]), (wave, [1.])):
        with pytest.raises(ValueError):
            proxy(waves, coefficients, **args)
    for update in (dict(nfp=0), dict(helicity=1, nfp=2), dict(integration_order=-1), dict(iota=[.4]),
                   dict(collision_frequency=1j), dict(rho_bound=[.3])):
        with pytest.raises(ValueError):
            proxy(wave, c, **(args | update))


def test_full_coulomb_state_objective_flux_units_and_equilibrium_derivatives(monkeypatch):
    """Field scaling, nonlinear flux radius and the objective contract survive tracing."""
    def transform(amplitude, rt, **kwargs):
        return dict(bmnc_b=jnp.array([[2., amplitude, .001]]), bmns_b=jnp.zeros((1, 3)),
                    xm_b=np.array([0, 1, 1]), xn_b=np.array([0, 0, 2]),
                    iota_b=jnp.array([.4]), G_b=jnp.array([5.]), I_b=jnp.array([.02]), nfp=2,
                    psi_b=jnp.array([-.3]), psi_edge=jnp.array(-2.), s_b=jnp.array([.5]))
    monkeypatch.setattr(neoclassical.ConservingHeatBound, "_boozer", lambda self, state, rt: transform(state, rt))
    monkeypatch.setattr(neoclassical, "_aspect_scalars", lambda state, rt: (1., None, None, None))
    options = dict(temperature_ev=1000., collision_strengths=[100., 1000.], radial_electric_fields=[0., 30.],
                   geometry_order=(2, 1), fit_grid=32, integration_order=16, rho_bound=.4)
    term = neoclassical.ConservingHeatBound([.5], **options)
    out = term.compute_state(.2, None)
    assert out["upper"].shape == (1, 2, 2) and np.all(out["valid"])
    assert 0 < out["geometry_fit_error_sampled"][0] < .002
    np.testing.assert_allclose(term(SimpleNamespace(state=.2, runtime=None))**2, out["total"])
    value, grad = jax.jit(jax.value_and_grad(lambda a: term.total_state(a, None)))(.2)
    fd = (term.total_state(.200001, None) - term.total_state(.199999, None)) / .000002
    np.testing.assert_allclose(grad, fd, rtol=2.e-7)
    assert value > 0

    def scaled(amplitude, rt, **kwargs):
        data = transform(amplitude, rt, **kwargs)
        return data | {k: 2 * data[k] for k in ("bmnc_b", "G_b", "I_b", "psi_b", "psi_edge")}

    monkeypatch.setattr(neoclassical.ConservingHeatBound, "_boozer", lambda self, state, rt: scaled(state, rt))
    rescaled = term.compute_state(.2, None)
    np.testing.assert_allclose(rescaled["upper"][..., 0], out["upper"][..., 0] / 4, rtol=2.e-13)
    np.testing.assert_allclose(rescaled["lower"][..., 0], out["lower"][..., 0] / 4, rtol=2.e-13)
    undersampled = neoclassical.ConservingHeatBound([.5], **(options | dict(helicity=32)))
    with pytest.raises(ValueError, match="incoming Boozer"):
        undersampled.compute_state(.2, None)
    for update in (dict(surfaces=[]), dict(surfaces=[0.]), dict(temperature_ev=-1.),
                   dict(collision_strengths=[]), dict(radial_electric_fields=[np.nan]), dict(scale=0.),
                   dict(geometry_order=(1,)), dict(fit_grid=4), dict(charge_number=0.)):
        with pytest.raises(ValueError):
            neoclassical.ConservingHeatBound(**(options | dict(surfaces=[.5]) | update))


def test_full_coulomb_macro_matches_complete_velocity_polynomials():
    """Full degree-four velocity closure independently checks the reduced affine bound."""
    wave = np.array([[0, 0], [1, 0], [0, 1], [1, -1]])
    coefficient = np.array([1.2, .18, .013j, .006])
    options = dict(iota=.41, G_b=3., I_b=.02, nfp=2, collision_frequency=10., electric_field=.01)
    count, mean, e, nu = 97, 1.2, .012, 10.
    x, y = np.meshgrid(*(2 * np.pi * np.arange(count) / count,) * 2, indexing="ij")
    phase = np.exp(1j * (x[..., None] * wave[:, 0] + y[..., None] * wave[:, 1]))
    kd, ke = wave @ np.array([.41, 2.]), wave @ np.array([3., -.04]) / (3. + .41 * .02)
    pcoef = np.divide(ke, kd, out=np.zeros_like(kd), where=kd != 0) * coefficient
    J, p, dj, ej = [np.real(phase @ v) for v in (coefficient, pcoef, 1j * kd * coefficient, 1j * ke * coefficient)]
    fx, fy = np.meshgrid(*(np.rint(np.fft.fftfreq(count) * count),) * 2, indexing="ij")
    along, across = .41 * fx + 2 * fy, (3 * fx - .04 * fy) / (3. + .41 * .02)

    def derivative(value, frequency):
        factor = (1j * frequency).reshape((count, count) + (1,) * (value.ndim - 2))
        return np.fft.ifft2(np.fft.fft2(value, axes=(0, 1)) * factor, axes=(0, 1)).real

    def primitive(value):
        support = (abs(fx) <= 5) & (abs(fy) <= 5) & ((fx != 0) | (fy != 0))
        factor = np.divide(1., 1j * along, out=np.zeros_like(along, dtype=complex), where=support)
        return np.fft.ifft2(np.fft.fft2(value) * factor).real

    u = (J - mean) / mean
    phi = (1 - 3 * u + 6 * u**2) * primitive(p) / mean**3
    psi = J**3 * phi
    density, flow = -derivative(psi, along), e * derivative(psi, across)
    r = e * (3 * ej * phi + J * derivative(phi, across))
    force = -e * (J * derivative(r, across) + 1.5 * ej * r)
    correction = -4 * np.mean(force) / np.mean(dj**2)
    pressure = primitive(force + .25 * correction * dj**2)
    energy = 3 * pressure + correction * J * dj
    heat_flow = -e * primitive(derivative(energy, across))
    pe, po = np.zeros((count, count, 3, 6)), np.zeros((count, count, 3, 6))
    pe[..., 0, 0], pe[..., 0, 2] = density, np.sqrt(1.5) * (density - 2 * energy / 3)
    pe[..., 0, 3] = (3 * pressure - energy) / np.sqrt(3)
    po[..., 0, 1], po[..., 0, 4] = np.sqrt(2) * flow, np.sqrt(5) * (flow - 2 * heat_flow / 5)
    po[..., 1, 4], po[..., 2, 4] = np.sqrt(5) / 2, np.sqrt(5) * p / 2

    # Gaussian quadrature derives all streaming/mirror/source matrix entries,
    # including invariant and ell=4 components omitted by the reduced runtime.
    z, wz = np.polynomial.hermite.hermgauss(7)
    vperp2, wp = np.polynomial.laguerre.laggauss(7)
    z, vperp2 = np.meshgrid(z, vperp2, indexing="ij")
    z, vperp2, weight = z.ravel(), vperp2.ravel(), np.outer(wz, wp).ravel() / np.sqrt(np.pi)
    q, zero = z**2 + vperp2, np.zeros_like(z)
    basis = np.stack((np.ones_like(z), np.sqrt(2) * z, np.sqrt(2 / 3) * (1.5 - q),
        (3 * z**2 - q) / np.sqrt(3), 2 * z * (2.5 - q) / np.sqrt(5),
        np.sqrt(2 / 15) * z * (5 * z**2 - 3 * q), np.sqrt(2 / 15) * (q**2 - 5 * q + 15 / 4),
        np.sqrt(2 / 21) * (3 * z**2 - q) * (3.5 - q),
        (35 * z**4 - 30 * z**2 * q + 3 * q**2) / (2 * np.sqrt(105))), -1)
    mirror_values = -.5 * vperp2[:, None] * np.stack((zero, np.sqrt(2) * np.ones_like(z), zero,
        2 * np.sqrt(3) * z, 2 * (2.5 - q) / np.sqrt(5), 3 * np.sqrt(2 / 15) * (5 * z**2 - q)), -1)
    np.testing.assert_allclose(basis.T @ (weight[:, None] * basis), np.eye(9), atol=4.e-14)
    stream = basis.T @ ((weight * z)[:, None] * basis[:, :6])
    mirror = basis.T @ (weight[:, None] * mirror_values)
    source = basis.T @ (weight * .5 * (q + z**2) * (q - 2.5))
    embed = np.eye(9, 6)
    re = (J[..., None, None] * (derivative(po, along) @ stream.T)
          - .5 * dj[..., None, None] * (po @ (stream + mirror).T)
          + e * J[..., None, None] * (derivative(pe, across) @ embed.T))
    ro = (J[..., None, None]**2 * (derivative(pe, along) @ stream.T)
          - .5 * (J * dj)[..., None, None] * (pe @ mirror.T)
          + e * J[..., None, None] * (derivative(po, across) @ embed.T)
          - .5 * e * ej[..., None, None] * (po @ embed.T))
    re[..., 2, :] = -.5 * (J * ej)[..., None] * source - re[..., 2, :]
    ro[..., 2, :] *= -1
    assert max(np.max(abs(re[..., [0, 1, 2, 8]])), np.max(abs(ro[..., [0, 1, 2, 8]]))) < 3.e-11
    s = np.sqrt(2 / np.pi)
    metric = s * np.diag([0., 0., 0., 32 / 15, 16 / 5, 64 / 45, 16 / 5, 3856 / 1715, 6224 / 5145])
    metric[3, 7] = metric[7, 3] = -48 * np.sqrt(14) * s / 245

    def gram(values, matrix, angular_weight):
        return np.einsum("abiv,vw,abjw,ab->ij", values, matrix, values, angular_weight, optimize=True) / (count**2 * mean)

    collision = gram(pe, s * np.diag([0., 0., 0., 8 / 5, 0., 0.]), J)
    collision += gram(po, s * np.diag([0., 0., 0., 0., 16 / 15, 0.]), np.ones_like(J))
    inverse = gram(re, metric, J**-3) + gram(ro, metric, J**-4)
    trial = np.linalg.solve(nu * collision[:2, :2] + inverse[:2, :2] / nu,
                            inverse[:2, 2] / nu - nu * collision[:2, 2])
    plus, minus = np.r_[trial, 1.], np.r_[-trial, 1.]
    reference = nu * plus @ collision @ plus + minus @ inverse @ minus / nu
    result = neoclassical.conserving_stellarator_proxy(wave, coefficient, macro_order=2, integration_order=24, **options)
    coarse = neoclassical.conserving_stellarator_proxy(wave, coefficient, macro_order=2, integration_order=0, **options)
    assert result["valid"] and coarse["valid"] and coarse["upper"] > reference
    np.testing.assert_allclose(result["upper"], reference, rtol=3.e-11)
    assert 0 < result["upper"] < .99 * result["scalar_upper"]
    sharper = neoclassical.conserving_stellarator_proxy(wave, coefficient, macro_order=2, integration_order=24,
                                                       stress_inverse_bound=2.1, **options)
    np.testing.assert_allclose(sharper["upper"], result["upper"], rtol=2.e-14)
    assert sharper["scalar_upper"] < result["scalar_upper"]


def test_full_coulomb_macro_jvp_vjp_and_field_broadcasting():
    """Geometry, iota, collision and electric-field derivatives share one fixed chart."""
    wave, coefficients = np.array([[0, 0], [1, 0], [0, 1]]), jnp.array([1.2, .18, .013j])

    @jax.jit
    def response(v):
        return neoclassical.conserving_stellarator_proxy(wave, coefficients.at[1].set(v[0]), iota=v[1],
            G_b=3., I_b=.02, nfp=2, collision_frequency=v[2], electric_field=v[3],
            integration_order=16, macro_order=2, rho_bound=.3)["upper"]

    parameters, direction, step = jnp.array([.18, .41, 10., .01]), jnp.array([.02, .03, .7, .003]), 1.e-5
    value, forward = jax.jvp(response, (parameters,), (direction,))
    _, pullback = jax.vjp(response, parameters)
    reverse = pullback(jnp.asarray(1.))[0] @ direction
    finite = (response(parameters + step * direction) - response(parameters - step * direction)) / (2 * step)
    assert np.isfinite(value)
    np.testing.assert_allclose(forward, reverse, rtol=2.e-12)
    np.testing.assert_allclose(forward, finite, rtol=3.e-7)
    batched = neoclassical.conserving_stellarator_proxy(wave, coefficients, iota=.41, G_b=3., I_b=.02, nfp=2,
        collision_frequency=jnp.array([.1, 1., 10.])[:, None], electric_field=jnp.array([-.01, 0., .01])[None, :],
        integration_order=16, macro_order=2)
    assert np.all(batched["valid"]) and batched["upper"].shape == (3, 3)
    np.testing.assert_allclose(batched["upper"][:, 0], batched["upper"][:, 2], rtol=2.e-13)
    assert np.all(batched["upper"] <= batched["scalar_upper"] * (1 + 2.e-13))


def test_full_coulomb_macro_degenerate_sources_and_expanded_chart_guards():
    """Zero sources are regular; resonant or numerically singular trial charts reject."""
    proxy = neoclassical.conserving_stellarator_proxy
    options = dict(iota=.41, G_b=1., I_b=0., macro_order=2, integration_order=8)
    uniform = proxy([[0, 0]], [1.], electric_field=.01, **options)
    assert uniform["valid"] and uniform["upper"] == uniform["lower"] == uniform["scalar_upper"] == 0
    assert jax.grad(lambda x: proxy([[0, 0]], jnp.array([x]), **options)["upper"])(1.) == 0
    no_drive = proxy([[0, 0], [0, 1]], [1., .1], electric_field=.01, **options)
    assert no_drive["valid"] and no_drive["upper"] == 0
    resonant = proxy([[0, 0], [1, 0], [0, 1]], [1., .2, .01], **(options | dict(iota=.5)))
    assert not resonant["valid"] and jnp.isinf(resonant["upper"]) and jnp.isfinite(resonant["scalar_upper"])
    singular = proxy([[0, 0], [1, 0]], [1., 1.e-200], **options)
    assert not singular["valid"] and jnp.isinf(singular["upper"]) and singular["lower"] == 0
    invalid = proxy([[0, 0], [1, 0]], [1., .2], collision_frequency=0., **options)
    assert not invalid["valid"] and jnp.isinf(invalid["upper"]) and jnp.isinf(invalid["scalar_upper"])
    for order in (-1, .5, [2], 2j, np.inf):
        with pytest.raises(ValueError, match="macro_order"):
            proxy([[0, 0]], [1.], **(options | dict(macro_order=order)))


def test_full_coulomb_macro_state_adapter_preserves_diagnostic_units(monkeypatch):
    """The optional refinement and scalar comparison use the same physical units."""
    def transform(self, state, rt):
        return dict(bmnc_b=jnp.array([[2., state, .001]]), bmns_b=jnp.zeros((1, 3)),
                    xm_b=np.array([0, 1, 1]), xn_b=np.array([0, 0, 2]),
                    iota_b=jnp.array([.413]), G_b=jnp.array([5.]), I_b=jnp.array([.02]), nfp=2,
                    psi_b=jnp.array([-.3]), psi_edge=jnp.array(-2.), s_b=jnp.array([.5]))
    monkeypatch.setattr(neoclassical.ConservingHeatBound, "_boozer", transform)
    monkeypatch.setattr(neoclassical, "_aspect_scalars", lambda state, rt: (1., None, None, None))
    options = dict(temperature_ev=1000., collision_strengths=[100., 1000.], radial_electric_fields=[0., 30.],
                   geometry_order=(2, 1), fit_grid=32, integration_order=16, rho_bound=.4)
    scalar = neoclassical.ConservingHeatBound([.5], **options).compute_state(.2, None)
    term = neoclassical.ConservingHeatBound([.5], macro_order=2, **options)
    out = term.compute_state(.2, None)
    assert np.all(out["valid"]) and out["upper"].shape == out["scalar_upper"].shape == (1, 2, 2)
    np.testing.assert_allclose(out["scalar_upper"], scalar["upper"], rtol=3.e-12)
    assert np.all(out["upper"] <= out["scalar_upper"] * (1 + 3.e-13))
    np.testing.assert_allclose(term(SimpleNamespace(state=.2, runtime=None))**2, out["total"])


def test_continuous_energy_route_retains_geometry_gradients_and_its_distinct_schema():
    """The optional full-Maxwell branch is usable through the Boozer proxy API."""
    inputs = _spectrum()
    bins = (0., .5, 1., 2.5, 4., 8., 16., np.inf)
    options = dict(pitch_fractions=jnp.linspace(0, 1, 9), temperature_ev=1000.,
                   collision_frequencies=[1.], radial_electric_fields=[30.],
                   continuous_energy=True, energy_edges=bins)

    def response(amplitude):
        varied = inputs | {"bmnc_b": inputs["bmnc_b"].at[:, 2].multiply(amplitude)}
        return neoclassical.neoclassical_heat_bounds_from_boozer(**varied, **options)

    result = response(1.)
    assert result["upper"].shape == (1, 1) and result["bin_upper"].shape == (1, 1, len(bins) - 1)
    assert result["valid"][0, 0]
    assert "mono_upper" not in result and "matrix_upper" not in result
    value, derivative = jax.jit(jax.value_and_grad(lambda x: response(x)["upper"].sum()))(jnp.asarray(1.))
    np.testing.assert_allclose(derivative, 2 * value, rtol=2.e-10)
    geometry = _orbit_geometry()
    base = dict(temperature_ev=1000., collision_frequencies=[1.], radial_electric_fields=[0.])
    for update in ({"continuous_energy": True, "collision_factors": [1., 1.]},
                   {"continuous_energy": True, "collision_frequencies": []},
                   {"energy_edges": bins}):
        with pytest.raises(ValueError):
            neoclassical.neoclassical_heat_bounds_from_geometry(geometry, **(base | update))


def test_continuous_energy_objective_composes_with_state_interface(monkeypatch):
    """The class can optimize continuous-energy bounds without monoenergetic fields."""
    inputs = _spectrum()
    spectrum = {k: v for k, v in inputs.items() if k in
                ("bmnc_b", "xm_b", "xn_b", "iota_b", "G_b", "I_b", "nfp", "psi_b", "s_b")}
    spectrum["bmns_b"] = jnp.zeros_like(spectrum["bmnc_b"])
    monkeypatch.setattr(neoclassical, "boozer_spectrum_state", lambda state, rt, **kwargs: spectrum)
    monkeypatch.setattr(neoclassical, "_aspect_scalars", lambda state, rt: (1., None, None, None))
    runtime = SimpleNamespace(setup=SimpleNamespace(s_full=jnp.linspace(0, 1, 11)))
    term = neoclassical.NeoclassicalHeatBound(
        [.5], pitch_fractions=np.linspace(0, 1, 9), temperature_ev=1000.,
        collision_frequencies=[10.], radial_electric_fields=[0.], continuous_energy=True,
        nalpha=5, points_per_period=32, quadrature_order=16)
    result = term.compute_state(object(), runtime)
    assert result["upper"].shape == (1, 1, 1) and np.all(result["valid"])
    assert "mono_lower" not in result and np.isfinite(result["total"])
    np.testing.assert_allclose(result["residuals1d"]**2, result["total"])


def _near_axis_orbit(radius, kappa, b0=.2, b2=-.3):
    """Independent turning-point map for B=1-r*cos(chi)+r²*(b0+b2*cos(2chi))."""
    nodes, weights = np.polynomial.legendre.leggauss(160)
    t, weights = np.pi * nodes / 2, np.pi * weights / 2
    chi = 2 * np.arcsin(np.sqrt(kappa) * np.sin(t))
    cosine, turn_cosine = np.cos(chi), 1 - 2 * kappa
    field = 1 - radius * cosine + radius**2 * (b0 + b2 * np.cos(2 * chi))
    turn_field = 1 - radius * turn_cosine + radius**2 * (b0 + b2 * (2 * turn_cosine**2 - 1))
    pitch = 1 / turn_field
    dchi = 2 * np.sqrt(kappa) * np.cos(t) / np.sqrt(1 - kappa * np.sin(t)**2)
    parallel = np.sqrt(2 * pitch * radius * kappa) * np.cos(t)
    parallel *= np.sqrt(1 - 2 * radius * b2 * (turn_cosine + cosine))
    dl = dchi / field
    action = 2 * np.sum(weights * parallel * dl)
    tau = 2 * np.sum(weights * dl / parallel)
    diffusion = 2 * pitch * np.sum(weights * parallel * dl / field)
    field_psi = -cosine / radius + 2 * (b0 + b2 * np.cos(2 * chi))
    jpsi = -2 * np.sum(weights * field_psi / field * (1 - pitch * field / 2) * dl / parallel)
    return pitch, np.asarray([action, tau, diffusion, jpsi])


@pytest.mark.parametrize("kappa", [.15, .55, .9])
def test_near_axis_orbits_match_independent_singularity_removed_integrals(kappa):
    """First-/second-order field coefficients reproduce finite-radius bounce and drift kernels."""
    radius = .01 * np.asarray([.8, 1., 1.2])
    pitch, expected = _near_axis_orbit(radius[1], kappa)
    geometry = neoclassical.bounce_geometry_from_boozer(
        bmnc_b=np.column_stack((1 + .2 * radius**2, -radius, -.3 * radius**2)),
        xm_b=[0, 1, 2], xn_b=[0, 0, 0], iota_b=np.ones(3), G_b=np.ones(3), I_b=np.zeros(3),
        nfp=1, psi_b=radius**2 / 2, s_b=radius**2, helicity_m=1, helicity_n=0,
        pitch_edges=[pitch - .05 * radius[1], pitch + .05 * radius[1]],
        nalpha=3, well_center=0., points_per_period=1024, quadrature_order=256,
        radial_interpolation="radius")
    assert geometry["valid"]
    actual = [geometry["reference_action"][0], 2 * geometry["weight"][0],
              geometry["diffusion"][0], geometry["radial_action_derivative"][0]]
    np.testing.assert_allclose(actual, expected, rtol=3.e-4)
    np.testing.assert_allclose(geometry["precession_factor"][0], expected[3] / expected[1], rtol=3.e-4)
    np.testing.assert_allclose(geometry["source_factor"], 0., atol=1.e-11)

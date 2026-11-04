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
    assert float(result["upper"]) <= float(result["collision_only"]) + eps


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

    for name in ("NeoclassicalHeatBound", "neoclassical_heat_bounds_state"):
        assert getattr(vmex, name) is getattr(neoclassical, name)
        assert getattr(optimize, name) is getattr(neoclassical, name)
    assert vmex.neoclassical_heat_bounds_from_boozer is neoclassical.neoclassical_heat_bounds_from_boozer


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

"""Thin VMEX-to-NEO_JAX effective-ripple adapter checks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from vmex.core import neoclassical

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.mark.parametrize("lasym", [False, True])
def test_state_spectrum_adapter_preserves_live_geometry_and_ad(monkeypatch, lasym):
    """Every physical field survives layout conversion and both AD directions."""
    import jax
    import jax.numpy as jnp

    names = ("rmnc_b", "zmns_b", "pmns_b", "bmnc_b",
             "rmns_b", "zmnc_b", "pmnc_b", "bmns_b")
    reference = np.arange(1., 7.).reshape(2, 3) / 10
    config = object()

    def spectrum(x):
        fields = {name: (i + 1) * reference * x[i]
                  if lasym or i < 4 else jnp.zeros_like(reference)
                  for i, name in enumerate(names)}
        return dict(fields, nfp=3, xm_b=np.array([0, 1, 2]),
                    xn_b=np.array([0, 3, -3]), iota_b=x[8] * jnp.array([.4, .5]),
                    G_b=x[9] * jnp.array([2., 3.]), I_b=x[10] * jnp.array([.1, .2]),
                    s_b=jnp.array([.25, .75]))

    def run_neo(booz, **kwargs):
        assert booz["nfp_b"] == 3 and booz["ns_b"] == 2
        assert booz["mode_first"] is True and booz["asym"] is True
        assert kwargs == dict(config=config, use_jax=True, progress=False, jax_surface_scan=True)
        # Distinct mode and field coefficients detect a wrong transpose,
        # missing parity, or a changed sign of the toroidal-angle harmonics.
        profile = sum((i + 1) * jnp.array([1., 2., 4.]) @ booz[name]
                      for i, name in enumerate(names))
        profile += 3 * booz["iota_b"] + 5 * booz["buco_b"] + 7 * booz["bvco_b"]
        return SimpleNamespace(eps_eff=profile ** 2)

    monkeypatch.setattr(neoclassical, "_neo_imports", lambda: (None, run_neo))
    weights = jnp.array([.3, .7])

    def loss(x):
        surfaces, profile = neoclassical.epsilon_effective_from_spectrum(spectrum(x), config=config)
        return weights @ profile + 0 * jnp.sum(surfaces)

    def oracle(x):
        fields = spectrum(x)
        profile = sum((i + 1) * (fields[name] @ jnp.array([1., 2., 4.]))
                      for i, name in enumerate(names))
        return weights @ (profile + 3 * fields["iota_b"] + 5 * fields["I_b"] + 7 * fields["G_b"]) ** 2

    x = jnp.linspace(.8, 1.2, 11)
    direction = jnp.linspace(-.3, .4, 11)
    value, gradient = jax.jit(jax.value_and_grad(loss))(x)
    expected, expected_gradient = jax.value_and_grad(oracle)(x)
    np.testing.assert_allclose(value, expected, rtol=1e-13)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-13, atol=1e-13)
    _, tangent = jax.jit(lambda p: jax.jvp(loss, (p,), (direction,)))(x)
    _, pullback = jax.vjp(loss, x)
    np.testing.assert_allclose(tangent, pullback(jnp.array(1.))[0] @ direction, rtol=1e-13)
    step = 1e-5
    np.testing.assert_allclose(tangent, (oracle(x + step*direction) - oracle(x - step*direction))/(2*step), rtol=1e-8)


def test_state_ripple_wrapper_requests_full_shared_boozer_geometry(monkeypatch):
    from vmex.core import omnigenity

    calls = {}
    state, runtime, config, spectrum = object(), object(), object(), object()

    def transform(st, rt, **kwargs):
        assert st is state and rt is runtime
        calls.update(kwargs)
        return spectrum

    def evaluate(sp, **kwargs):
        assert sp is spectrum and kwargs == dict(config=config)
        return "actual_s", "physical_eps32"

    monkeypatch.setattr(omnigenity, "boozer_spectrum_state", transform)
    monkeypatch.setattr(neoclassical, "epsilon_effective_from_spectrum", evaluate)
    assert neoclassical.epsilon_effective_state(
        state, runtime, surfaces=[.3, .7], mboz=8, nboz=6, oversample=3,
        config=config) == ("actual_s", "physical_eps32")
    assert calls == dict(surfaces=[.3, .7], mboz=8, nboz=6, oversample=3, geometry=True)


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

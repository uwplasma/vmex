from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import jax, jnp
import vmec_jax as vj
from vmec_jax.namelist import InData
from vmec_jax.profiles import MU0
from vmec_jax import profiles as profile_mod


def test_profile_polynomial_scaled_and_pressure_match_manual_values():
    s = np.linspace(0.0, 1.0, 6)
    ne = vj.ProfilePolynomial(np.asarray([2.0, -0.5, 0.25]))
    te = vj.ProfilePolynomial(np.asarray([3.0, -1.0]))
    pressure = vj.ProfilePressure(ne, te, ne, te)
    pressure_pa = vj.ProfileScaled(pressure, vj.ELEMENTARY_CHARGE)

    ne_expected = 2.0 - 0.5 * s + 0.25 * s**2
    te_expected = 3.0 - s
    np.testing.assert_allclose(np.asarray(ne.f(s)), ne_expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(ne.dfds(s)), -0.5 + 0.5 * s, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        np.asarray(pressure.f(s)),
        2.0 * ne_expected * te_expected,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.asarray(pressure_pa.f(s)),
        vj.ELEMENTARY_CHARGE * 2.0 * ne_expected * te_expected,
        rtol=0.0,
        atol=1.0e-30,
    )


def test_profile_helpers_cover_vmec_pressure_conversion_edges():
    """Profile helpers should expose the same physics building blocks users script with."""

    base = vj.ProfilePolynomial(np.asarray([4.0]))
    scaled = vj.ProfileScaled(base, 2.5)
    pair_left = vj.ProfilePolynomial(np.asarray([2.0, -0.5]))
    pair_right = vj.ProfilePolynomial(np.asarray([3.0, 1.0]))
    pressure = vj.ProfilePressure(pair_left, pair_right, pair_left, pair_right)

    np.testing.assert_allclose(np.asarray(base.dfds([0.0, 0.7])), [0.0, 0.0])
    np.testing.assert_allclose(vj.profile_to_power_series_coeffs(scaled), [10.0])
    np.testing.assert_allclose(
        vj.profile_to_power_series_coeffs(pressure),
        2.0 * np.polynomial.polynomial.polymul([2.0, -0.5], [3.0, 1.0]),
    )

    am, pres_scale = vj.pressure_profile_to_vmec_am(
        vj.ProfilePolynomial(np.asarray([1.0, 2.0, 0.0, 1.0e-16])),
        pres_scale=2.0,
        trim_tol=1.0e-15,
    )
    assert pres_scale == 2.0
    np.testing.assert_allclose(am, [0.5, 1.0])

    indata = InData(scalars={"AM": [99.0], "PRES_SCALE": 0.0}, indexed={})
    updated = vj.with_pressure_profile(indata, vj.ProfilePolynomial(np.asarray([3.0, -1.0])), pres_scale=3.0)
    assert updated is not indata
    assert updated.scalars["PMASS_TYPE"] == "power_series"
    np.testing.assert_allclose(updated.scalars["AM"], [1.0, -1.0 / 3.0])

    with pytest.raises(ValueError, match="at least one"):
        vj.ProfilePressure()
    with pytest.raises(ValueError, match="even number"):
        vj.ProfilePressure(pair_left)
    with pytest.raises(ValueError, match="pres_scale"):
        vj.pressure_profile_to_vmec_am(base, pres_scale=0.0)
    with pytest.raises(TypeError, match="unsupported profile"):
        vj.profile_to_power_series_coeffs(object())  # type: ignore[arg-type]


def test_profile_pytrees_and_parser_helpers_are_stable():
    """The finite-beta profile classes are pytrees and parser helpers are deterministic."""

    with pytest.raises(NotImplementedError):
        profile_mod.Profile().f(0.0)
    with pytest.raises(NotImplementedError):
        profile_mod.Profile().dfds(0.0)

    polynomial = vj.ProfilePolynomial(jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    np.testing.assert_allclose(np.asarray(polynomial(0.25)), np.asarray(polynomial.f(0.25)))
    scaled = vj.ProfileScaled(polynomial, jnp.asarray(4.0, dtype=jnp.float64))
    pressure = vj.ProfilePressure(polynomial, polynomial)
    bundle = vj.standard_finite_beta_profiles(2.5)

    children, aux = polynomial.tree_flatten()
    restored_polynomial = vj.ProfilePolynomial.tree_unflatten(aux, children)
    np.testing.assert_allclose(np.asarray(restored_polynomial.coeffs), [1.0, 2.0])

    children, aux = scaled.tree_flatten()
    restored_scaled = vj.ProfileScaled.tree_unflatten(aux, children)
    np.testing.assert_allclose(np.asarray(restored_scaled.f(0.5)), np.asarray(scaled.f(0.5)))

    children, aux = pressure.tree_flatten()
    restored_pressure = vj.ProfilePressure.tree_unflatten(aux, children)
    np.testing.assert_allclose(np.asarray(restored_pressure.dfds(0.5)), np.asarray(pressure.dfds(0.5)))

    np.testing.assert_allclose(np.asarray(bundle.ne_coeffs), np.asarray(bundle.ne.coeffs))
    np.testing.assert_allclose(np.asarray(bundle.Te_coeffs), np.asarray(bundle.Te.coeffs))
    np.testing.assert_allclose(np.asarray(bundle.Ti_coeffs), np.asarray(bundle.Ti.coeffs))
    np.testing.assert_allclose(np.asarray(bundle.Zeff_coeffs), np.asarray(bundle.Zeff.coeffs))

    assert profile_mod._as_float_list(None) == []
    assert profile_mod._as_float_list([1, "2.5"]) == [1.0, 2.5]
    assert profile_mod._as_float_list(np.asarray([1.0, 2.0])) == [1.0, 2.0]
    assert profile_mod._as_float_list("bad-float") == "bad-float"
    assert profile_mod._lower(None, "Power_Series") == "Power_Series"
    assert profile_mod._lower([], "power_series") == "power_series"
    assert profile_mod._lower(["'akima_spline'"], "power_series") == "akima_spline"
    assert profile_mod._is_jax_tracer(1.0) is False
    np.testing.assert_allclose(np.asarray(profile_mod._coeff_array([1.0, 2.0], nmin=4)), [1.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(
        np.asarray(profile_mod._coeff_array(jnp.asarray([3.0, 4.0], dtype=jnp.float64), nmin=4)),
        [3.0, 4.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(profile_mod._coeffs_static_or_jax([5.0, 6.0]), [5.0, 6.0])


def test_standard_finite_beta_profiles_match_landreman_stage_one_scaling():
    beta_percent = 2.5
    bundle = vj.standard_finite_beta_profiles(beta_percent)
    scale = (beta_percent / 100.0) / 0.05
    ne0 = 3.0e20 * scale ** (1.0 / 3.0)
    te0 = 15.0e3 * scale ** (2.0 / 3.0)

    np.testing.assert_allclose(
        vj.profile_to_power_series_coeffs(bundle.ne),
        ne0 * np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, -0.99]),
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        vj.profile_to_power_series_coeffs(bundle.Te),
        te0 * np.asarray([1.0, -0.99]),
        rtol=1.0e-14,
    )
    s = np.asarray([0.0, 0.5, 1.0])
    expected_ne = ne0 * (1.0 - 0.99 * s**5)
    expected_te = te0 * (1.0 - 0.99 * s)
    expected_pressure_pa = vj.ELEMENTARY_CHARGE * 2.0 * expected_ne * expected_te
    np.testing.assert_allclose(np.asarray(bundle.pressure_pa.f(s)), expected_pressure_pa, rtol=2.0e-14)


def test_standard_pressure_profile_writes_vmec_am_and_eval_profiles_in_pa_and_internal_units():
    beta_percent = 2.5
    pressure_pa = vj.standard_pressure_profile(beta_percent)
    indata = InData(scalars={"AM": [0.0], "PRES_SCALE": 0.0}, indexed={})
    updated = vj.with_pressure_profile(indata, pressure_pa, pres_scale=1.0)

    assert updated.scalars["PMASS_TYPE"] == "power_series"
    assert updated.scalars["PRES_SCALE"] == 1.0
    assert len(updated.scalars["AM"]) == 7

    s = np.linspace(0.0, 1.0, 9)
    prof = vj.eval_profiles(updated, s)
    expected_pa = np.asarray(pressure_pa.f(s))
    np.testing.assert_allclose(np.asarray(prof["pressure_pa"]), expected_pa, rtol=2.0e-13, atol=1.0e-10)
    np.testing.assert_allclose(np.asarray(prof["pressure"]), MU0 * expected_pa, rtol=2.0e-13, atol=1.0e-16)


def test_standard_pressure_profile_is_differentiable_wrt_beta_percent():
    def axis_pressure(beta_percent):
        return vj.standard_pressure_profile(beta_percent).f(jnp.asarray([0.0], dtype=jnp.float64))[0]

    value, grad = jax.value_and_grad(axis_pressure)(jnp.asarray(2.5, dtype=jnp.float64))
    assert float(value) > 0.0
    assert np.isfinite(float(grad))
    assert float(grad) > 0.0


def test_redl_bootstrap_accepts_standard_profile_bundle_and_is_differentiable():
    def objective(beta_percent):
        bundle = vj.standard_finite_beta_profiles(beta_percent)
        jdotb, _details = vj.redl_bootstrap_jdotb(
            s=jnp.asarray([0.25, 0.5, 0.75], dtype=jnp.float64),
            G=jnp.asarray([1.8, 1.7, 1.6], dtype=jnp.float64),
            R=jnp.asarray([2.0, 2.1, 2.2], dtype=jnp.float64),
            iota=jnp.asarray([0.42, 0.45, 0.48], dtype=jnp.float64),
            epsilon=jnp.asarray([0.10, 0.12, 0.14], dtype=jnp.float64),
            f_t=jnp.asarray([0.45, 0.50, 0.55], dtype=jnp.float64),
            psi_edge=jnp.asarray(-0.03, dtype=jnp.float64),
            nfp=2,
            helicity_n=0,
            ne_coeffs=bundle.ne_coeffs,
            Te_coeffs=bundle.Te_coeffs,
            Ti_coeffs=bundle.Ti_coeffs,
            Zeff_coeffs=bundle.Zeff_coeffs,
        )
        return jnp.sum(jdotb * jdotb)

    value, grad = jax.value_and_grad(objective)(jnp.asarray(2.5, dtype=jnp.float64))
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))
    assert abs(float(grad)) > 0.0


def test_profile_pressure_matches_simsopt_when_available():
    simsopt_profiles = pytest.importorskip("simsopt.mhd.profiles")
    coeff_ne = np.asarray([3.0e20, 0.0, 0.0, 0.0, 0.0, -2.97e20])
    coeff_te = np.asarray([15.0e3, -14.85e3])
    s = np.linspace(0.0, 1.0, 11)

    ours_ne = vj.ProfilePolynomial(coeff_ne)
    ours_te = vj.ProfilePolynomial(coeff_te)
    ours = vj.ProfileScaled(vj.ProfilePressure(ours_ne, ours_te, ours_ne, ours_te), vj.ELEMENTARY_CHARGE)

    ref_ne = simsopt_profiles.ProfilePolynomial(coeff_ne)
    ref_te = simsopt_profiles.ProfilePolynomial(coeff_te)
    ref = simsopt_profiles.ProfileScaled(
        simsopt_profiles.ProfilePressure(ref_ne, ref_te, ref_ne, ref_te),
        vj.ELEMENTARY_CHARGE,
    )

    np.testing.assert_allclose(np.asarray(ours.f(s)), ref.f(s), rtol=2.0e-14)
    np.testing.assert_allclose(np.asarray(ours.dfds(s)), ref.dfds(s), rtol=2.0e-14)

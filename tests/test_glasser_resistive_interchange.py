from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax as vj
import vmec_jax.optimization_workflow as workflow
from vmec_jax._compat import jnp


def _upper_bound_softplus(values, *, maximum: float, softness: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return softness * np.logaddexp(0.0, (values - maximum) / softness)


def test_glasser_matches_landreman_jorge_iota_prime_relation():
    dmerc = jnp.asarray([0.0, 0.10, -0.05, 0.08, 0.0], dtype=jnp.float64)
    iota_prime = jnp.asarray([0.0, 0.30, -0.50, 1.10, 0.0], dtype=jnp.float64)
    shear = iota_prime / (2.0 * np.pi)
    h_term = jnp.asarray([0.0, 0.020, -0.015, 0.045, 0.0], dtype=jnp.float64)

    result = vj.glasser_resistive_interchange_from_mercier_terms(DMerc=dmerc, shear=shear, H=h_term)

    iota_prime_np = np.asarray(iota_prime)
    h_np = np.asarray(h_term)
    dmerc_np = np.asarray(dmerc)
    valid = iota_prime_np != 0.0
    expected_correction = np.zeros_like(dmerc_np)
    expected_correction[valid] = (
        4.0
        * np.pi**2
        / iota_prime_np[valid] ** 2
        * (h_np[valid] - iota_prime_np[valid] ** 2 / (8.0 * np.pi**2)) ** 2
    )
    expected_d_r = np.zeros_like(dmerc_np)
    expected_d_r[valid] = -dmerc_np[valid] + expected_correction[valid]

    np.testing.assert_allclose(np.asarray(result["D_R"]), expected_d_r, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(result["glasser_correction"]),
        expected_correction,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(np.asarray(result["H"]), np.asarray(h_term), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(np.asarray(result["glasser_shear_valid"]), valid)


def test_mercier_profile_integrals_prefer_jdotb_bdotb_H_reconstruction_over_dcurr_fallback():
    data = dict(
        s=np.linspace(0.0, 1.0, 5),
        phips=np.asarray([0.0, 1.0, 1.0, 1.0, 1.0]),
        iotas=np.asarray([0.0, 0.10, 0.22, 0.32, 0.38]),
        vp=np.asarray([0.0, 1.0, 1.1, 1.3, 1.6]),
        pres=np.asarray([0.0, 0.04, 0.03, 0.02, 0.0]),
        torcur=np.asarray([0.0, 0.05, 0.09, 0.12, 0.14]),
        tpp=np.asarray([0.0, 1.4, 1.5, 1.6, 0.0]),
        tbb=np.asarray([0.0, 0.70, 0.80, 0.90, 0.0]),
        tjb=np.asarray([0.0, 0.20, 0.18, 0.16, 0.0]),
        tjj=np.asarray([0.0, 0.05, 0.06, 0.07, 0.0]),
    )
    jdotb = np.asarray([0.0, 0.60, 0.40, 0.20, 0.0])
    bdotb = np.asarray([0.0, 2.00, 2.00, 1.00, 0.0])

    with_parallel_current = vj.mercier_terms_from_profile_integrals(**data, jdotb=jdotb, bdotb=bdotb)
    fallback = vj.mercier_terms_from_profile_integrals(**data)

    shear = np.asarray(with_parallel_current["shear"])
    ratio = np.divide(jdotb, bdotb, out=np.zeros_like(jdotb), where=bdotb != 0.0)
    expected_h_from_profiles = shear * (data["tjb"] - ratio * data["tbb"])
    expected_h_from_dcurr = -np.asarray(fallback["Dcurr"])

    np.testing.assert_allclose(np.asarray(with_parallel_current["H"]), expected_h_from_profiles, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(fallback["H"]), expected_h_from_dcurr, rtol=1e-13, atol=1e-13)
    assert not np.allclose(expected_h_from_profiles[1:-1], expected_h_from_dcurr[1:-1])

    for terms in (with_parallel_current, fallback):
        h = np.asarray(terms["H"])
        dmerc = np.asarray(terms["DMerc"])
        shear = np.asarray(terms["shear"])
        expected_d_r = np.zeros_like(dmerc)
        valid = shear != 0.0
        expected_d_r[valid] = -dmerc[valid] + (h[valid] - 0.5 * shear[valid] ** 2) ** 2 / shear[valid] ** 2
        np.testing.assert_allclose(np.asarray(terms["D_R"]), expected_d_r, rtol=1e-13, atol=1e-13)


def test_glasser_zero_shear_masking_and_regularization_are_explicit():
    dmerc = jnp.asarray([0.70, 0.10, 0.20, -0.30], dtype=jnp.float64)
    shear = jnp.asarray([0.0, 5.0e-4, 2.0e-3, -0.10], dtype=jnp.float64)
    h_term = jnp.asarray([0.020, 0.030, 0.040, 0.050], dtype=jnp.float64)

    strict = vj.glasser_resistive_interchange_from_mercier_terms(DMerc=dmerc, shear=shear, H=h_term)
    regularized = vj.glasser_resistive_interchange_from_mercier_terms(
        DMerc=dmerc,
        shear=shear,
        H=h_term,
        shear_epsilon=1.0e-3,
    )

    dmerc_np = np.asarray(dmerc)
    shear_np = np.asarray(shear)
    h_np = np.asarray(h_term)

    strict_valid = shear_np != 0.0
    expected_strict = np.zeros_like(dmerc_np)
    expected_strict[strict_valid] = (
        -dmerc_np[strict_valid]
        + (h_np[strict_valid] - 0.5 * shear_np[strict_valid] ** 2) ** 2 / shear_np[strict_valid] ** 2
    )
    np.testing.assert_allclose(np.asarray(strict["D_R"]), expected_strict, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(np.asarray(strict["glasser_shear_valid"]), strict_valid)
    np.testing.assert_allclose(np.asarray(strict["D_R"])[0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(strict["glasser_correction"])[0], 0.0, rtol=0.0, atol=0.0)

    eps = 1.0e-3
    expected_regularized_correction = (h_np - 0.5 * shear_np**2) ** 2 / (shear_np**2 + eps**2)
    expected_regularized = -dmerc_np + expected_regularized_correction
    expected_regularized_valid = shear_np**2 > eps**2
    np.testing.assert_allclose(
        np.asarray(regularized["glasser_correction"]),
        expected_regularized_correction,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(np.asarray(regularized["D_R"]), expected_regularized, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(np.asarray(regularized["glasser_shear_valid"]), expected_regularized_valid)
    assert np.isfinite(np.asarray(regularized["D_R"])).all()


def test_public_glasser_objective_uses_upper_bound_penalty_and_regularized_terms(monkeypatch):
    raw_terms = {
        "DMerc": jnp.asarray([1.0, 0.02, 0.04, 1.0], dtype=jnp.float64),
        "D_R": jnp.asarray([99.0, 0.30, -0.10, 99.0], dtype=jnp.float64),
        "H": jnp.asarray([0.0, 0.03, 0.04, 0.0], dtype=jnp.float64),
        "shear": jnp.asarray([0.0, 0.0, 0.20, 0.0], dtype=jnp.float64),
    }
    calls = []

    def fake_mercier_terms_from_state(**kwargs):
        calls.append(kwargs)
        return raw_terms

    monkeypatch.setattr(workflow, "mercier_terms_from_state", fake_mercier_terms_from_state)
    ctx = SimpleNamespace(static="static", indata="indata", signgs=-1)

    unregularized = vj.GlasserResistiveInterchange(maximum=0.05, softness=0.20, mmax_force=4, nmax_force=5)
    np.testing.assert_allclose(
        np.asarray(unregularized.J(ctx, "state")),
        _upper_bound_softplus([0.30, -0.10], maximum=0.05, softness=0.20),
        rtol=1e-13,
        atol=1e-13,
    )
    assert calls[-1]["state"] == "state"
    assert calls[-1]["static"] == "static"
    assert calls[-1]["indata"] == "indata"
    assert calls[-1]["signgs"] == -1
    assert calls[-1]["mmax_force"] == 4
    assert calls[-1]["nmax_force"] == 5

    regularized = vj.GlasserResistiveInterchange(maximum=0.05, softness=0.20, shear_epsilon=0.10)
    expected_terms = vj.glasser_resistive_interchange_from_mercier_terms(
        DMerc=raw_terms["DMerc"],
        shear=raw_terms["shear"],
        H=raw_terms["H"],
        shear_epsilon=0.10,
    )
    terms = regularized.terms(ctx, "state")
    np.testing.assert_allclose(np.asarray(terms["D_R"]), np.asarray(expected_terms["D_R"]), rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(
        np.asarray(terms["glasser_shear_valid"]),
        np.asarray(expected_terms["glasser_shear_valid"]),
    )
    np.testing.assert_allclose(
        np.asarray(regularized.J(ctx, "state")),
        _upper_bound_softplus(np.asarray(expected_terms["D_R"])[1:-1], maximum=0.05, softness=0.20),
        rtol=1e-13,
        atol=1e-13,
    )

    term = regularized.to_objective_term(target=0.0, residual_weight=3.0)
    assert term.name == "D_R"
    np.testing.assert_allclose(
        np.asarray(term.residual(ctx, "state")),
        3.0 * np.asarray(regularized.J(ctx, "state")),
        rtol=1e-13,
        atol=1e-13,
    )
    with pytest.raises(ValueError, match="target=0"):
        regularized.to_objective_term(target=1.0, residual_weight=1.0)

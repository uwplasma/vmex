from __future__ import annotations

import numpy as np
import pytest

from vmec_jax import profiles as profiles_mod
from vmec_jax._compat import has_jax, jax, jnp
from vmec_jax.namelist import InData
from vmec_jax.profiles import eval_profiles


def test_profile_private_helpers_and_fallbacks():
    assert profiles_mod._as_float_list(None) == []
    assert profiles_mod._as_float_list(3) == [3.0]
    assert profiles_mod._as_float_list(["2.5"]) == [2.5]
    assert profiles_mod._lower(["'Two_Power'"], "power_series") == "two_power"
    assert profiles_mod._lower([], "power_series") == "power_series"
    np.testing.assert_allclose(np.asarray(profiles_mod._coeff_array([], nmin=3)), np.zeros(3))

    indata = InData(
        scalars={
            "AC_AUX_S": [0.0, 0.4, 0.3, 1.0],
            "AC_AUX_F": [1.0, 2.0, 3.0, 4.0],
        },
        indexed={},
    )
    s_aux, f_aux = profiles_mod._aux_profile_arrays(indata, "AC")
    np.testing.assert_allclose(np.asarray(s_aux), [0.0, 0.4])
    np.testing.assert_allclose(np.asarray(f_aux), [1.0, 2.0])


def test_two_power_and_empty_current_profile_branches():
    indata = InData(
        scalars={
            "PMASS_TYPE": "two_power",
            "PIOTA_TYPE": "power_series",
            "PCURR_TYPE": "cubic_spline_i",
            "AM": [2.0, 2.0, 1.0],
            "AI": [0.0, 2.0],
            "AC": [1.0],
            "AC_AUX_S": [],
            "AC_AUX_F": [],
            "PRES_SCALE": 1.0,
            "BLOAT": 1.0,
            "SPRES_PED": 0.5,
            "LRFP": True,
            "NCURR": 1,
        },
        indexed={},
    )
    s = np.asarray([0.0, 0.25, 0.75])
    prof = eval_profiles(indata, s)
    assert np.isinf(np.asarray(prof["iota"])[0])
    np.testing.assert_allclose(np.asarray(prof["iota"])[1:], [2.0, 2.0 / 3.0])
    np.testing.assert_allclose(np.asarray(prof["current"]), np.zeros_like(s))


def test_current_profile_spline_i_and_ip_follow_vmec_integral_convention():
    s = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    base_scalars = {
        "PMASS_TYPE": "power_series",
        "PIOTA_TYPE": "power_series",
        "AM": [0.0],
        "AI": [],
        "AC": [1.0],
        "AC_AUX_S": [0.0, 1.0],
        "AC_AUX_F": [0.0, 2.0],
        "PRES_SCALE": 1.0,
        "BLOAT": 1.0,
        "SPRES_PED": 1.0,
        "NCURR": 1,
    }

    integrated = eval_profiles(
        InData(scalars={**base_scalars, "PCURR_TYPE": "cubic_spline_ip"}, indexed={}),
        s,
    )
    direct = eval_profiles(
        InData(scalars={**base_scalars, "PCURR_TYPE": "cubic_spline_i"}, indexed={}),
        s,
    )

    # VMEC's cubic_spline_ip parameterizes I'(s); with I'(s)=2s the enclosed
    # current is exactly I(s)=s^2. cubic_spline_i stores I(s) directly.
    np.testing.assert_allclose(np.asarray(integrated["current"]), s * s, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(direct["current"]), 2.0 * s, rtol=1.0e-12, atol=1.0e-12)


def test_concrete_jax_profile_grid_uses_host_equivalent_path():
    if not has_jax():
        pytest.skip("JAX is required for concrete JAX array profile coverage")

    indata = InData(
        scalars={
            "PMASS_TYPE": "two_power",
            "PIOTA_TYPE": "power_series",
            "PCURR_TYPE": "cubic_spline_ip",
            "AM": [4.0, 2.0, 1.0],
            "AI": [0.25, 0.5],
            "AC": [1.0],
            "AC_AUX_S": [0.0, 0.5, 1.0],
            "AC_AUX_F": [0.0, 1.0, 0.0],
            "PRES_SCALE": 2.0,
            "BLOAT": 1.0,
            "SPRES_PED": 1.0,
            "LRFP": False,
            "NCURR": 1,
        },
        indexed={},
    )
    s_np = np.linspace(0.0, 1.0, 7)
    prof_np = eval_profiles(indata, s_np)
    prof_jax = eval_profiles(indata, jnp.asarray(s_np))

    for key in ("pressure_pa", "pressure", "iota", "current"):
        np.testing.assert_allclose(np.asarray(prof_jax[key]), np.asarray(prof_np[key]), rtol=1.0e-12, atol=1.0e-12)


def test_profile_coeff_padding_stays_jit_differentiable_for_traced_current_coeffs():
    if not has_jax():
        pytest.skip("JAX is required for traced profile coefficient coverage")

    @jax.jit
    def enclosed_current(coeffs, x):
        padded = profiles_mod._coeff_array(coeffs, nmin=4)
        return profiles_mod._pcurr_power_series_ip(padded, x)

    coeffs = jnp.asarray([1.0, 2.0])
    x = jnp.asarray([0.0, 0.5, 1.0])
    current = enclosed_current(coeffs, x)
    grad = jax.grad(lambda c: jnp.sum(enclosed_current(c, x)))(coeffs)

    np.testing.assert_allclose(np.asarray(current), [0.0, 0.75, 2.0], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(grad), [1.5, 0.625], rtol=1.0e-12, atol=1.0e-12)

from __future__ import annotations

import numpy as np

from vmec_jax import profiles as profiles_mod
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

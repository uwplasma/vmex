from __future__ import annotations

import numpy as np
import pytest

import vmec_jax as vj
from vmec_jax import bootstrap_current as bc
from vmec_jax._compat import jax, jnp
from vmec_jax.namelist import InData
from vmec_jax.profiles import eval_profiles


def test_redl_current_low_beta_update_matches_manufactured_derivative():
    s = jnp.linspace(0.0, 1.0, 9)
    desired_ip = 1.0 + 2.0 * s
    bdotb = 2.5 + 0.5 * s
    dpsi_ds = -0.03
    jdotb = desired_ip * bdotb / (2.0 * np.pi * dpsi_ds)

    actual = vj.redl_current_derivative_update(
        s=s,
        jdotB_redl=jdotb,
        bdotb=bdotb,
        dpsi_ds=dpsi_ds,
        policy="low_beta",
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(desired_ip), rtol=1.0e-13, atol=1.0e-13)


def test_redl_current_lagged_pressure_update_removes_pressure_term():
    s = jnp.linspace(0.0, 1.0, 11)
    previous_current = s * s
    desired_ip = 0.5 + s
    bdotb = 3.0 + s
    dpds = -4.0 * jnp.ones_like(s)
    dpsi_ds = 0.02
    pressure_term = bc.MU0 * previous_current * dpds / bdotb
    rhs = desired_ip + pressure_term
    jdotb = rhs * bdotb / (2.0 * np.pi * dpsi_ds)

    actual = vj.redl_current_derivative_update(
        s=s,
        jdotB_redl=jdotb,
        bdotb=bdotb,
        dpsi_ds=dpsi_ds,
        dpds=dpds,
        previous_current=previous_current,
        policy="lagged_pressure",
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(desired_ip), rtol=1.0e-12, atol=1.0e-12)


def test_redl_current_integrating_factor_matches_constant_coefficient_solution():
    s = jnp.linspace(0.0, 1.0, 1001)
    a = 0.4
    rhs = 2.0 * jnp.ones_like(s)
    bdotb = jnp.ones_like(s)
    dpds = (a / bc.MU0) * jnp.ones_like(s)
    dpsi_ds = 0.07
    jdotb = rhs / (2.0 * np.pi * dpsi_ds)

    update = vj.redl_current_integrating_factor_update(
        s=s,
        jdotB_redl=jdotb,
        bdotb=bdotb,
        dpsi_ds=dpsi_ds,
        dpds=dpds,
    )

    expected_current = (2.0 / a) * (1.0 - np.exp(-a * np.asarray(s)))
    np.testing.assert_allclose(np.asarray(update["current"]), expected_current, rtol=3.0e-7, atol=3.0e-7)
    np.testing.assert_allclose(
        np.asarray(update["current_derivative"]),
        np.asarray(rhs - a * update["current"]),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_vmec_current_profile_conversion_and_indata_application_round_trip():
    s = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    current_derivative = 2.0 * s
    indata = InData(
        scalars={
            "PCURR_TYPE": "power_series",
            "CURTOR": 0.0,
            "NCURR": 0,
            "AC": [0.0],
            "AM": [0.0],
            "AI": [],
            "PRES_SCALE": 1.0,
            "PMASS_TYPE": "power_series",
            "PIOTA_TYPE": "power_series",
        },
        indexed={},
    )

    updated, profile = vj.bootstrap_current_update_to_indata(
        indata,
        s=s,
        current_derivative=current_derivative,
        signgs=-1,
    )

    assert indata.scalars["PCURR_TYPE"] == "power_series"
    assert updated.scalars["NCURR"] == 1
    assert updated.scalars["PCURR_TYPE"] == "cubic_spline_ip"
    assert updated.scalars["CURTOR"] == pytest.approx(-1.0)
    np.testing.assert_allclose(updated.scalars["AC_AUX_S"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(updated.scalars["AC_AUX_F"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(profile["current"]), [0.0, 0.25, 1.0])

    prof = eval_profiles(updated, np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]))
    np.testing.assert_allclose(np.asarray(prof["current"]), [0.0, 0.0625, 0.25, 0.5625, 1.0], rtol=1.0e-12)


def test_bootstrap_current_helpers_are_differentiable():
    s = jnp.linspace(0.0, 1.0, 101)
    bdotb = 2.0 + s
    dpds = -0.1 * jnp.ones_like(s)
    dpsi_ds = -0.04

    def edge_current(scale):
        jdotb = scale * (1.0 + s)
        update = vj.redl_current_integrating_factor_update(
            s=s,
            jdotB_redl=jdotb,
            bdotb=bdotb,
            dpsi_ds=dpsi_ds,
            dpds=dpds,
        )
        return update["current"][-1]

    value, grad = jax.value_and_grad(edge_current)(jnp.asarray(3.0, dtype=jnp.float64))
    assert np.isfinite(float(value))
    assert np.isfinite(float(grad))
    assert abs(float(grad)) > 0.0


def test_damping_and_input_validation_branches():
    np.testing.assert_allclose(np.asarray(vj.damp_current_profile([0.0, 2.0], [2.0, 4.0], 0.25)), [0.5, 2.5])
    with pytest.raises(ValueError, match="damping"):
        vj.damp_current_profile([0.0, 1.0], [1.0, 2.0], 1.5)
    with pytest.raises(ValueError, match="lagged_pressure"):
        vj.redl_current_derivative_update(
            s=[0.0, 1.0],
            jdotB_redl=[1.0, 1.0],
            bdotb=[1.0, 1.0],
            dpsi_ds=1.0,
            policy="lagged_pressure",
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        vj.apply_current_profile_to_indata(
            InData(scalars={}, indexed={}),
            ac_aux_s=[0.0, 0.5, 0.4],
            ac_aux_f=[1.0, 1.0, 1.0],
            curtor=1.0,
        )


def test_vmec_flux_derivative_sign_convention():
    assert float(vj.dpsi_ds_from_vmec_phiedge(0.2, signgs=1)) == pytest.approx(0.2 / (2.0 * np.pi))
    assert float(vj.dpsi_ds_from_vmec_phiedge(0.2, signgs=-1)) == pytest.approx(-0.2 / (2.0 * np.pi))

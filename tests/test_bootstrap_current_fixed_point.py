from __future__ import annotations

from types import SimpleNamespace

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
    with pytest.raises(ValueError, match="size mismatch"):
        vj.apply_current_profile_to_indata(
            InData(scalars={}, indexed={}),
            ac_aux_s=[0.0, 1.0],
            ac_aux_f=[1.0],
            curtor=1.0,
        )
    with pytest.raises(ValueError, match="at least two"):
        vj.apply_current_profile_to_indata(
            InData(scalars={}, indexed={}),
            ac_aux_s=[0.0],
            ac_aux_f=[1.0],
            curtor=1.0,
        )


def test_vmec_flux_derivative_sign_convention():
    assert float(vj.dpsi_ds_from_vmec_phiedge(0.2, signgs=1)) == pytest.approx(0.2 / (2.0 * np.pi))
    assert float(vj.dpsi_ds_from_vmec_phiedge(0.2, signgs=-1)) == pytest.approx(-0.2 / (2.0 * np.pi))


def test_pressure_derivative_and_current_profile_branch_coverage():
    s = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    dpds = bc._pressure_derivative_pa_from_profile_coeffs(
        s=s,
        ne_coeffs=jnp.asarray([2.0, 3.0]),
        Te_coeffs=jnp.asarray([5.0, 7.0]),
        Ti_coeffs=jnp.asarray([11.0, 13.0]),
        Zeff_coeffs=jnp.asarray([2.0, 0.5]),
    )
    assert np.all(np.isfinite(np.asarray(dpds)))
    assert np.asarray(dpds).shape == (3,)

    indata = InData(scalars={"AC": [1.0, 2.0], "PCURR_TYPE": "power_series"}, indexed={})
    np.testing.assert_allclose(
        np.asarray(bc._current_derivative_from_indata(indata, s, "power_series")),
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_allclose(
        np.asarray(bc._current_derivative_from_indata(InData(scalars={}, indexed={}), s, "unknown")),
        np.zeros(3),
    )

    profile = vj.vmec_current_profile_from_bootstrap_update(
        s=s,
        current=jnp.asarray([0.0, 0.2, 0.5]),
        signgs=1,
        pcurr_type="cubic_spline_i",
    )
    assert profile["curtor"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="current is required"):
        vj.vmec_current_profile_from_bootstrap_update(s=s, signgs=1, pcurr_type="cubic_spline_i")
    with pytest.raises(ValueError, match="current_derivative is required"):
        vj.vmec_current_profile_from_bootstrap_update(s=s, signgs=1, pcurr_type="cubic_spline_ip")
    with pytest.raises(ValueError, match="only cubic"):
        vj.vmec_current_profile_from_bootstrap_update(s=s, current=[0.0, 0.0, 0.0], signgs=1, pcurr_type="power")


def test_current_grid_update_samples_validation_and_exact_grid():
    options = vj.BootstrapCurrentOptions(helicity_n=0, n_current=3)
    out = bc._current_grid_update_samples(
        options=options,
        s=jnp.asarray([0.0, 0.5, 1.0]),
        jdotB_redl=jnp.asarray([1.0, 2.0, 3.0]),
        bdotb=jnp.asarray([4.0, 5.0, 6.0]),
        dpds=jnp.asarray([7.0, 8.0, 9.0]),
    )
    np.testing.assert_allclose(np.asarray(out["s"]), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(np.asarray(out["jdotB_redl"]), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="n_current"):
        bc._current_grid_update_samples(
            options=vj.BootstrapCurrentOptions(helicity_n=0, n_current=1),
            s=jnp.asarray([0.0, 1.0]),
            jdotB_redl=jnp.asarray([1.0, 1.0]),
            bdotb=jnp.asarray([1.0, 1.0]),
            dpds=jnp.asarray([0.0, 0.0]),
        )
    with pytest.raises(ValueError, match="must be 1D"):
        vj.redl_current_rhs(jdotB_redl=jnp.ones((2, 2)), bdotb=jnp.ones((2, 2)), dpsi_ds=1.0)
    with pytest.raises(ValueError, match="at least two"):
        vj.redl_current_rhs(jdotB_redl=[1.0], bdotb=[1.0], dpsi_ds=1.0)
    with pytest.raises(ValueError, match="length"):
        vj.redl_current_rhs(jdotB_redl=[1.0, 2.0], bdotb=[1.0, 2.0, 3.0], dpsi_ds=1.0)


def test_bootstrap_current_fixed_point_runs_callback_loop_to_convergence():
    s = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    target_derivative = jnp.asarray([0.5, 1.0, 1.5], dtype=jnp.float64)
    bdotb = 2.0 * jnp.ones_like(s)
    dpsi_ds = 0.03
    jdotb_redl = target_derivative * bdotb / (2.0 * np.pi * dpsi_ds)
    indata = InData(
        scalars={
            "PHIEDGE": 2.0 * np.pi * dpsi_ds,
            "PCURR_TYPE": "cubic_spline_ip",
            "CURTOR": 0.0,
            "NCURR": 1,
            "AC": [1.0],
            "AC_AUX_S": [0.0, 0.5, 1.0],
            "AC_AUX_F": [0.0, 0.0, 0.0],
        },
        indexed={},
    )
    solve_inputs = []

    class FakeRun:
        signgs = 1

    def solve_fn(current_indata):
        solve_inputs.append(tuple(float(x) for x in current_indata.scalars["AC_AUX_F"]))
        return FakeRun()

    def diagnostics_fn(_run, _current_indata):
        return {
            "s": s,
            "jdotB_redl": jdotb_redl,
            "bdotb": bdotb,
            "dpds": jnp.zeros_like(s),
            "dpsi_ds": dpsi_ds,
            "signgs": 1,
            "mismatch_norm": 0.0,
            "aspect": 7.0,
        }

    result = vj.bootstrap_current_fixed_point(
        indata,
        options=vj.BootstrapCurrentOptions(
            helicity_n=1,
            n_current=3,
            damping=1.0,
            current_tol=1.0e-12,
            mismatch_tol=1.0e-12,
            max_fixed_point_iter=4,
        ),
        solve_fn=solve_fn,
        diagnostics_fn=diagnostics_fn,
    )

    assert result.converged
    assert result.reason == "current_and_mismatch_tolerances"
    assert len(result.history) == 2
    assert solve_inputs[0] == (0.0, 0.0, 0.0)
    np.testing.assert_allclose(solve_inputs[1], np.asarray(target_derivative), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(result.indata.scalars["AC_AUX_F"], np.asarray(target_derivative))
    assert result.indata.scalars["CURTOR"] == pytest.approx(1.0)
    assert result.history[-1].current_update_norm == pytest.approx(0.0, abs=1.0e-14)
    assert result.history[-1].aspect == pytest.approx(7.0)


def test_bootstrap_current_fixed_point_limits_current_update_norm():
    s = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    target_derivative = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    bdotb = jnp.ones_like(s)
    dpsi_ds = 0.05
    indata = InData(
        scalars={
            "PHIEDGE": 2.0 * np.pi * dpsi_ds,
            "PCURR_TYPE": "cubic_spline_ip",
            "CURTOR": 0.0,
            "NCURR": 1,
            "AC": [1.0],
            "AC_AUX_S": [0.0, 0.5, 1.0],
            "AC_AUX_F": [0.0, 0.0, 0.0],
        },
        indexed={},
    )

    result = vj.bootstrap_current_fixed_point(
        indata,
        options=vj.BootstrapCurrentOptions(
            helicity_n=1,
            n_current=3,
            damping=1.0,
            max_current_update_norm=0.2,
            max_fixed_point_iter=1,
        ),
        solve_fn=lambda _current_indata: SimpleNamespace(signgs=1),
        diagnostics_fn=lambda _run, _current_indata: {
            "s": s,
            "jdotB_redl": target_derivative / (2.0 * np.pi * dpsi_ds),
            "bdotb": bdotb,
            "dpds": jnp.zeros_like(s),
            "dpsi_ds": dpsi_ds,
            "signgs": 1,
            "mismatch_norm": 1.0,
        },
    )

    np.testing.assert_allclose(result.indata.scalars["AC_AUX_F"], 0.2 * np.asarray(target_derivative), rtol=1.0e-12)
    assert result.history[0].current_update_limited is True
    assert result.history[0].effective_damping == pytest.approx(0.2)
    assert result.history[0].current_update_norm == pytest.approx(0.2)
    assert result.history[0].unlimited_current_update_norm == pytest.approx(1.0)
    assert result.history[0].max_current_update_norm == pytest.approx(0.2)


def test_bootstrap_current_fixed_point_limits_cubic_spline_i_current_profile():
    s = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    target_derivative = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    bdotb = jnp.ones_like(s)
    dpsi_ds = 0.05
    indata = InData(
        scalars={
            "PHIEDGE": 2.0 * np.pi * dpsi_ds,
            "PCURR_TYPE": "cubic_spline_i",
            "CURTOR": 0.0,
            "NCURR": 1,
            "AC": [1.0],
            "AC_AUX_S": [0.0, 0.5, 1.0],
            "AC_AUX_F": [0.0, 0.0, 0.0],
        },
        indexed={},
    )

    result = vj.bootstrap_current_fixed_point(
        indata,
        options=vj.BootstrapCurrentOptions(
            helicity_n=1,
            n_current=3,
            pcurr_type="cubic_spline_i",
            damping=1.0,
            max_current_update_norm=0.2,
            max_fixed_point_iter=1,
        ),
        solve_fn=lambda _current_indata: SimpleNamespace(signgs=1),
        diagnostics_fn=lambda _run, _current_indata: {
            "s": s,
            "jdotB_redl": target_derivative / (2.0 * np.pi * dpsi_ds),
            "bdotb": bdotb,
            "dpds": jnp.zeros_like(s),
            "dpsi_ds": dpsi_ds,
            "signgs": 1,
            "mismatch_norm": 1.0,
        },
    )

    limited_derivative = 0.2 * target_derivative
    expected_current = bc.integrate_current_derivative(s, limited_derivative)
    np.testing.assert_allclose(result.indata.scalars["AC_AUX_F"], np.asarray(expected_current), rtol=1.0e-12)
    assert result.indata.scalars["PCURR_TYPE"] == "cubic_spline_i"
    assert result.indata.scalars["CURTOR"] == pytest.approx(float(expected_current[-1]))
    assert result.history[0].current_update_limited is True
    assert result.history[0].effective_damping == pytest.approx(0.2)
    assert result.history[0].current_update_norm == pytest.approx(0.2)
    assert result.history[0].unlimited_current_update_norm == pytest.approx(1.0)
    assert result.history[0].max_current_update_norm == pytest.approx(0.2)


def test_bootstrap_current_fixed_point_can_return_best_evaluated_profile_on_budget():
    s = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    dpsi_ds = 0.05
    indata = InData(
        scalars={
            "PHIEDGE": 2.0 * np.pi * dpsi_ds,
            "PCURR_TYPE": "cubic_spline_ip",
            "CURTOR": 0.0,
            "NCURR": 1,
            "AC": [1.0],
            "AC_AUX_S": [0.0, 0.5, 1.0],
            "AC_AUX_F": [0.0, 0.0, 0.0],
        },
        indexed={},
    )
    targets = [jnp.ones_like(s), 2.0 * jnp.ones_like(s)]
    mismatches = [0.5, 0.2]
    calls = {"n": 0}

    def diagnostics_fn(_run, _current_indata):
        i = calls["n"]
        calls["n"] += 1
        return {
            "s": s,
            "jdotB_redl": targets[i] / (2.0 * np.pi * dpsi_ds),
            "bdotb": jnp.ones_like(s),
            "dpds": jnp.zeros_like(s),
            "dpsi_ds": dpsi_ds,
            "signgs": 1,
            "mismatch_norm": mismatches[i],
        }

    result = vj.bootstrap_current_fixed_point(
        indata,
        options=vj.BootstrapCurrentOptions(
            helicity_n=1,
            n_current=3,
            damping=1.0,
            current_tol=0.0,
            mismatch_tol=0.0,
            max_fixed_point_iter=2,
            return_best_evaluated_on_max_iter=True,
        ),
        solve_fn=lambda _current_indata: SimpleNamespace(signgs=1),
        diagnostics_fn=diagnostics_fn,
    )

    assert not result.converged
    assert result.returned_best_evaluated is True
    assert result.best_evaluated_iteration == 2
    assert result.best_evaluated_mismatch_norm == pytest.approx(0.2)
    np.testing.assert_allclose(result.indata.scalars["AC_AUX_F"], np.ones(3), rtol=1.0e-12)
    np.testing.assert_allclose(result.history[-1].ac_aux_f, 2.0 * np.ones(3), rtol=1.0e-12)


def test_bootstrap_current_fixed_point_extends_interior_redl_samples_to_full_current_grid():
    s_redl = jnp.asarray([0.25, 0.5, 0.75], dtype=jnp.float64)
    bdotb = jnp.ones_like(s_redl)
    dpsi_ds = 0.02
    target_derivative = 2.0 * jnp.ones_like(s_redl)
    indata = InData(
        scalars={
            "PHIEDGE": 2.0 * np.pi * dpsi_ds,
            "PCURR_TYPE": "cubic_spline_ip",
            "CURTOR": 0.0,
            "NCURR": 1,
            "AC": [1.0],
            "AC_AUX_S": [0.0, 1.0],
            "AC_AUX_F": [0.0, 0.0],
        },
        indexed={},
    )

    result = vj.bootstrap_current_fixed_point(
        indata,
        options=vj.BootstrapCurrentOptions(
            helicity_n=1,
            n_current=5,
            damping=1.0,
            max_fixed_point_iter=1,
        ),
        solve_fn=lambda _current_indata: type("FakeRun", (), {"signgs": 1})(),
        diagnostics_fn=lambda _run, _current_indata: {
            "s": s_redl,
            "jdotB_redl": target_derivative / (2.0 * np.pi * dpsi_ds),
            "bdotb": bdotb,
            "dpds": jnp.zeros_like(s_redl),
            "dpsi_ds": dpsi_ds,
            "signgs": 1,
        },
    )

    np.testing.assert_allclose(result.indata.scalars["AC_AUX_S"], np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(result.indata.scalars["AC_AUX_F"], 2.0 * np.ones(5), rtol=1.0e-12)
    assert result.indata.scalars["CURTOR"] == pytest.approx(2.0)


def test_bootstrap_current_fixed_point_low_beta_and_default_callbacks(monkeypatch):
    s = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    indata = InData(
        scalars={
            "PHIEDGE": 0.2,
            "PCURR_TYPE": "cubic_spline_ip",
            "CURTOR": 0.0,
            "NCURR": 1,
            "AC": [1.0],
            "AC_AUX_S": [0.0, 1.0],
            "AC_AUX_F": [0.0, 0.0],
        },
        indexed={},
    )

    def fake_solve(current_indata, *, run_kwargs=None):
        assert run_kwargs == {"max_iter": 4}
        return SimpleNamespace(signgs=-1)

    def fake_diag(_run, current_indata, **kwargs):
        assert kwargs["options"].helicity_n == 0
        assert current_indata.scalars["PCURR_TYPE"] == "cubic_spline_ip"
        return {
            "s": s,
            "jdotB_redl": jnp.asarray([1.0, 1.0]),
            "bdotb": jnp.asarray([2.0, 2.0]),
            "dpds": jnp.asarray([0.0, 0.0]),
            "mismatch_norm": 0.0,
            "signgs": -1,
            "beta_total": 0.01,
            "mean_iota": 0.4,
            "fsq_total": 1.0e-8,
        }

    monkeypatch.setattr(bc, "_default_bootstrap_solve_fn", fake_solve)
    monkeypatch.setattr(bc, "_default_redl_diagnostics_fn", fake_diag)
    result = vj.bootstrap_current_fixed_point(
        indata,
        options=vj.BootstrapCurrentOptions(
            helicity_n=0,
            n_current=2,
            policy="low_beta",
            damping=1.0,
            max_fixed_point_iter=1,
        ),
        ne_coeffs=[1.0, -0.5],
        Te_coeffs=[1.0, -0.5],
        run_kwargs={"max_iter": 4},
    )

    assert not result.converged
    assert result.history[0].beta_total == pytest.approx(0.01)
    assert result.history[0].mean_iota == pytest.approx(0.4)
    assert result.history[0].fsq_total == pytest.approx(1.0e-8)


def test_default_bootstrap_solve_fn_writes_temp_input(monkeypatch):
    import vmec_jax.driver as driver

    seen = {}

    def fake_run_fixed_boundary(path, **kwargs):
        seen["path"] = str(path)
        seen["kwargs"] = dict(kwargs)
        loaded = vj.read_indata(path)
        return SimpleNamespace(indata=loaded, signgs=1)

    monkeypatch.setattr(driver, "run_fixed_boundary", fake_run_fixed_boundary)
    run = bc._default_bootstrap_solve_fn(
        InData(scalars={"PHIEDGE": 0.3}, indexed={}),
        run_kwargs={"max_iter": 2},
    )

    assert run.signgs == 1
    assert run.indata.get_float("PHIEDGE") == pytest.approx(0.3)
    assert seen["kwargs"]["max_iter"] == 2
    assert seen["kwargs"]["verbose"] is False


def test_default_redl_diagnostics_fn_extracts_redl_and_solver_diagnostics(monkeypatch):
    import vmec_jax.finite_beta as finite_beta

    def fake_redl_bootstrap_mismatch_from_state(**_kwargs):
        return {
            "geometry": {
                "s": jnp.asarray([0.25, 0.75]),
                "fsa_B2": jnp.asarray([2.0, 3.0]),
            },
            "jdotB_redl": jnp.asarray([4.0, 5.0]),
            "residuals1d": jnp.asarray([0.3, 0.4]),
        }

    monkeypatch.setattr(finite_beta, "redl_bootstrap_mismatch_from_state", fake_redl_bootstrap_mismatch_from_state)
    run = SimpleNamespace(
        state=object(),
        static=object(),
        indata=InData(scalars={"PHIEDGE": 0.2}, indexed={}),
        signgs=1,
        result=SimpleNamespace(diagnostics={"final_fsqr": 1.0, "final_fsqz": 2.0, "final_fsql": 3.0}),
    )

    diag = bc._default_redl_diagnostics_fn(
        run,
        run.indata,
        options=vj.BootstrapCurrentOptions(helicity_n=0),
        ne_coeffs=[2.0, -1.0],
        Te_coeffs=[3.0, -1.0],
    )

    np.testing.assert_allclose(np.asarray(diag["s"]), [0.25, 0.75])
    np.testing.assert_allclose(np.asarray(diag["bdotb"]), [2.0, 3.0])
    assert float(diag["mismatch_norm"]) == pytest.approx(np.sqrt(0.3**2 + 0.4**2) / np.sqrt(2.0))
    assert diag["fsq_total"] == pytest.approx(6.0)


def test_bootstrap_current_fixed_point_rejects_invalid_iteration_controls():
    with pytest.raises(ValueError, match="max_fixed_point_iter"):
        vj.bootstrap_current_fixed_point(
            InData(scalars={}, indexed={}),
            options=vj.BootstrapCurrentOptions(helicity_n=0, max_fixed_point_iter=0),
            solve_fn=lambda _indata: object(),
            diagnostics_fn=lambda _run, _indata: {},
        )
    with pytest.raises(NotImplementedError, match="Anderson"):
        vj.bootstrap_current_fixed_point(
            InData(scalars={}, indexed={}),
            options=vj.BootstrapCurrentOptions(helicity_n=0, anderson_depth=1),
            solve_fn=lambda _indata: object(),
            diagnostics_fn=lambda _run, _indata: {},
        )
    with pytest.raises(ValueError, match="max_current_update_norm"):
        vj.bootstrap_current_fixed_point(
            InData(scalars={}, indexed={}),
            options=vj.BootstrapCurrentOptions(helicity_n=0, max_current_update_norm=0.0),
            solve_fn=lambda _indata: object(),
            diagnostics_fn=lambda _run, _indata: {},
        )


def test_bootstrap_current_fixed_point_requires_redl_profiles_for_default_diagnostics():
    with pytest.raises(ValueError, match="ne_coeffs and Te_coeffs"):
        vj.bootstrap_current_fixed_point(
            InData(scalars={}, indexed={}),
            options=vj.BootstrapCurrentOptions(helicity_n=0, max_fixed_point_iter=1),
            solve_fn=lambda _indata: object(),
        )

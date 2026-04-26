from __future__ import annotations

import os

import numpy as np
import pytest


def _require_slow() -> None:
    if os.environ.get("RUN_SLOW", "") != "1":
        pytest.skip("Set RUN_SLOW=1 to run slow quasisymmetry checks")


def test_quasisymmetry_ratio_residual_from_state_is_self_consistent(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.quasisymmetry import (
        quasisymmetry_diagnostics_from_state,
        quasisymmetry_ratio_residual_from_state,
        quasisymmetry_ratio_residual_from_wout,
    )
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))

    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-13)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces=True,
        use_scan=False,
        light_history=True,
    )

    diag = quasisymmetry_diagnostics_from_state(
        state=result.state,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    qs_state = quasisymmetry_ratio_residual_from_state(
        state=result.state,
        static=static,
        indata=indata,
        signgs=signgs,
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
    )
    qs_diag = quasisymmetry_ratio_residual_from_wout(
        diag,
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
    )

    assert np.asarray(diag.gmnc).ndim == 2
    assert np.asarray(diag.bmnc).ndim == 2
    assert np.asarray(diag.bsubumnc).ndim == 2
    assert np.asarray(diag.bsupumnc).ndim == 2
    np.testing.assert_allclose(
        np.asarray(qs_state["residuals1d"]),
        np.asarray(qs_diag["residuals1d"]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(np.asarray(qs_state["total"])) > 0.0


def test_as_jax_array_is_tracer_safe():
    pytest.importorskip("jax")

    import jax
    import jax.numpy as jnp

    from vmec_jax.quasisymmetry import _as_jax_array

    @jax.jit
    def traced(values):
        arr = _as_jax_array(values, dtype=np.float64)
        return jnp.sum(arr * arr)

    result = traced(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64))
    np.testing.assert_allclose(np.asarray(result), 14.0, rtol=0.0, atol=0.0)


def test_quasisymmetry_ratio_residual_returns_diagnostic_fields():
    pytest.importorskip("jax")

    from vmec_jax import load_wout
    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout

    root = os.path.dirname(os.path.dirname(__file__))
    wout = load_wout(os.path.join(root, "examples", "data", "wout_li383_low_res.nc"))
    qs = quasisymmetry_ratio_residual_from_wout(
        wout,
        surfaces=[0.5],
        helicity_m=1,
        helicity_n=1,
        ntheta=17,
        nphi=18,
    )

    for key in (
        "d_B_d_theta",
        "d_B_d_phi",
        "bsubu",
        "bsubv",
        "bsupu",
        "bsupv",
        "d_psi_d_s",
        "V_prime",
    ):
        assert key in qs

    np.testing.assert_allclose(
        np.asarray(qs["bsupu"] * qs["d_B_d_theta"] + qs["bsupv"] * qs["d_B_d_phi"]),
        np.asarray(qs["B_dot_grad_B"]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(
            qs["d_psi_d_s"]
            * (qs["bsubu"] * qs["d_B_d_phi"] - qs["bsubv"] * qs["d_B_d_theta"])
            / qs["sqrtg"]
        ),
        np.asarray(qs["B_cross_grad_B_dot_grad_psi"]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_scan_cache_lru_helpers_evict_oldest(monkeypatch):
    from collections import OrderedDict

    from vmec_jax.discrete_adjoint import _lru_cache_get, _lru_cache_put

    monkeypatch.setenv("VMEC_JAX_SCAN_CACHE_LIMIT", "2")
    cache = OrderedDict()
    _lru_cache_put(cache, ("a",), 1)
    _lru_cache_put(cache, ("b",), 2)
    assert list(cache.keys()) == [("a",), ("b",)]

    assert _lru_cache_get(cache, ("a",)) == 1
    assert list(cache.keys()) == [("b",), ("a",)]

    _lru_cache_put(cache, ("c",), 3)
    assert list(cache.keys()) == [("a",), ("c",)]

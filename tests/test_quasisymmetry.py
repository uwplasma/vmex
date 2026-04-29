from __future__ import annotations

import os
from types import SimpleNamespace

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


def test_quasisymmetry_surface_and_weight_helpers():
    pytest.importorskip("jax")

    from vmec_jax.quasisymmetry import _as_surface_array, _as_weight_array, _half_grid, _interp_half_grid

    np.testing.assert_allclose(np.asarray(_as_surface_array(0.5)), [0.5])
    np.testing.assert_allclose(np.asarray(_as_surface_array([0.25, 0.75])), [0.25, 0.75])
    np.testing.assert_allclose(np.asarray(_as_weight_array(None, 2)), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(_as_weight_array([2.0, 3.0], 2)), [2.0, 3.0])
    np.testing.assert_allclose(np.asarray(_half_grid(4, np.float64)), [1.0 / 6.0, 0.5, 5.0 / 6.0])

    samples = np.array([[10.0, 20.0], [30.0, 60.0], [50.0, 100.0]])
    s_half = np.array([0.0, 0.5, 1.0])
    interp = _interp_half_grid(samples, [0.25, 0.75], s_half)
    np.testing.assert_allclose(np.asarray(interp), [[20.0, 40.0], [40.0, 80.0]])

    single = _interp_half_grid(np.array([[7.0, 9.0]]), [0.25, 0.75], np.array([0.5]))
    np.testing.assert_allclose(np.asarray(single), [[7.0, 9.0], [7.0, 9.0]])

    with pytest.raises(ValueError, match="half-grid interpolation"):
        _interp_half_grid(np.zeros((0,)), [0.5], np.zeros((0,)))


def test_quasisymmetry_coefficient_shape_helpers():
    pytest.importorskip("jax")

    from vmec_jax.quasisymmetry import _optional_radial_mode_matrix, _radial_mode_matrix

    direct = _radial_mode_matrix(np.arange(6.0).reshape(3, 2), radial_count=3, mode_count=2)
    transposed = _radial_mode_matrix(np.arange(6.0).reshape(2, 3), radial_count=3, mode_count=2)

    np.testing.assert_allclose(np.asarray(direct), np.arange(6.0).reshape(3, 2))
    np.testing.assert_allclose(np.asarray(transposed), np.arange(6.0).reshape(2, 3).T)

    like = np.ones((3, 2))
    zeros = _optional_radial_mode_matrix(
        SimpleNamespace(lasym=False, bmns=np.ones((3, 2))),
        "bmns",
        radial_count=3,
        mode_count=2,
        like=like,
    )
    np.testing.assert_allclose(np.asarray(zeros), np.zeros((3, 2)))

    values = _optional_radial_mode_matrix(
        SimpleNamespace(lasym=True, bmns=np.arange(6.0).reshape(3, 2)),
        "bmns",
        radial_count=3,
        mode_count=2,
        like=like,
    )
    np.testing.assert_allclose(np.asarray(values), np.arange(6.0).reshape(3, 2))

    with pytest.raises(ValueError, match="expected a rank-2"):
        _radial_mode_matrix(np.ones((2, 2, 2)), radial_count=2, mode_count=2)
    with pytest.raises(ValueError, match="unexpected coefficient shape"):
        _radial_mode_matrix(np.ones((4, 4)), radial_count=3, mode_count=2)


def test_quasisymmetry_symoutput_split_reconstructs_half_grid():
    pytest.importorskip("jax")

    from vmec_jax.quasisymmetry import _vmec_symoutput_split_jax

    trig = SimpleNamespace(ntheta2=3, ntheta1=4)
    f = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)

    sym, asym = _vmec_symoutput_split_jax(f=f, trig=trig)
    rev_sym, rev_asym = _vmec_symoutput_split_jax(f=f, trig=trig, reversed_sym=True)

    np.testing.assert_allclose(np.asarray(sym + asym), f[:, :3, :])
    np.testing.assert_allclose(np.asarray(rev_sym + rev_asym), f[:, :3, :])
    np.testing.assert_allclose(np.asarray(rev_sym), np.asarray(asym))
    np.testing.assert_allclose(np.asarray(rev_asym), np.asarray(sym))


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


def test_quasisymmetry_ratio_residual_supports_lasym_wout():
    pytest.importorskip("jax")

    from vmec_jax import load_wout
    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout

    root = os.path.dirname(os.path.dirname(__file__))
    wout = load_wout(os.path.join(root, "examples", "data", "wout_basic_non_stellsym_simsopt.nc"))
    assert bool(wout.lasym)

    qs = quasisymmetry_ratio_residual_from_wout(
        wout,
        surfaces=[0.5],
        helicity_m=1,
        helicity_n=0,
        ntheta=13,
        nphi=14,
    )

    assert np.asarray(qs["residuals1d"]).ndim == 1
    assert np.all(np.isfinite(np.asarray(qs["residuals1d"])))
    assert np.isfinite(float(np.asarray(qs["total"])))
    assert float(np.linalg.norm(np.asarray(wout.bmns))) > 0.0


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

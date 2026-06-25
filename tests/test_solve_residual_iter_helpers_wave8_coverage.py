from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax._solve_runtime as solve_runtime
import vmec_jax.solve as solve
from vmec_jax.solvers.fixed_boundary import api as fixed_boundary_api
from vmec_jax.solvers.fixed_boundary.residual import iteration as residual_iteration
from vmec_jax.state import StateLayout, VMECState


def _state(ns: int = 4, k: int = 3) -> VMECState:
    layout = StateLayout(ns=ns, K=k, lasym=False)
    base = np.arange(ns * k, dtype=float).reshape(ns, k) + 1.0
    return VMECState(
        layout=layout,
        Rcos=base,
        Rsin=base + 10.0,
        Zcos=base + 20.0,
        Zsin=base + 30.0,
        Lcos=base + 40.0,
        Lsin=base + 50.0,
    )


def test_mask_scan_restart_force_payload_preserves_or_zeros_blocks():
    force_blocks = (
        np.asarray([1.0, -2.0, 3.0]),
        np.asarray([[4.0, 5.0], [6.0, 7.0]]),
    )

    kept, valid = solve._mask_scan_restart_force_payload(
        force_blocks=force_blocks,
        cache_valid=True,
        do_restart=False,
    )

    np.testing.assert_allclose(np.asarray(kept[0]), force_blocks[0])
    np.testing.assert_allclose(np.asarray(kept[1]), force_blocks[1])
    assert bool(np.asarray(valid)) is True

    masked, valid = solve._mask_scan_restart_force_payload(
        force_blocks=force_blocks,
        cache_valid=True,
        do_restart=True,
    )

    np.testing.assert_allclose(np.asarray(masked[0]), np.zeros_like(force_blocks[0]))
    np.testing.assert_allclose(np.asarray(masked[1]), np.zeros_like(force_blocks[1]))
    assert bool(np.asarray(valid)) is False


def test_coeff_column_and_mode_slice_helpers_cover_edges_and_invalid_indices():
    arr2 = np.arange(12.0).reshape(4, 3)

    np.testing.assert_allclose(np.asarray(solve._zero_coeff_column(arr2, idx=-1)), arr2)
    np.testing.assert_allclose(np.asarray(solve._zero_coeff_column(arr2, idx=99)), arr2)
    first_zeroed = np.asarray(solve._zero_coeff_column(arr2, idx=0))
    last_zeroed = np.asarray(solve._zero_coeff_column(arr2, idx=2))
    np.testing.assert_allclose(first_zeroed[:, 0], 0.0)
    np.testing.assert_allclose(first_zeroed[:, 1:], arr2[:, 1:])
    np.testing.assert_allclose(last_zeroed[:, :2], arr2[:, :2])
    np.testing.assert_allclose(last_zeroed[:, 2], 0.0)
    np.testing.assert_allclose(np.asarray(solve._zero_coeff_column(np.ones((2, 1)), idx=0)), np.zeros((2, 1)))

    arr3 = np.arange(24.0).reshape(4, 3, 2)
    replacement = np.asarray([[100.0, 101.0], [102.0, 103.0], [104.0, 105.0], [106.0, 107.0]])
    replaced = np.asarray(solve._replace_mode_slice(arr3, mode_idx=1, replacement=replacement))
    np.testing.assert_allclose(replaced[:, 0, :], arr3[:, 0, :])
    np.testing.assert_allclose(replaced[:, 1, :], replacement)
    np.testing.assert_allclose(replaced[:, 2, :], arr3[:, 2, :])
    assert solve._replace_mode_slice(None, mode_idx=0, replacement=replacement) is None
    np.testing.assert_allclose(np.asarray(solve._replace_mode_slice(arr3, mode_idx=8, replacement=replacement)), arr3)

    scale = np.asarray([1.0, 2.0, 3.0, 4.0])
    scaled = np.asarray(solve._scale_mode_slice(arr3, mode_idx=2, scale=scale))
    np.testing.assert_allclose(scaled[:, 2, :], arr3[:, 2, :] * scale[:, None])
    np.testing.assert_allclose(scaled[:, :2, :], arr3[:, :2, :])
    assert solve._scale_mode_slice(None, mode_idx=2, scale=scale) is None
    np.testing.assert_allclose(np.asarray(solve._scale_mode_slice(arr3, mode_idx=-2, scale=scale)), arr3)


def test_numpy_coeff_slice_helpers_copy_inputs_and_match_jax_variants():
    arr2 = np.arange(12.0).reshape(4, 3)
    arr3 = np.arange(24.0).reshape(4, 3, 2)
    arr2_original = arr2.copy()
    arr3_original = arr3.copy()

    np.testing.assert_allclose(
        solve._zero_coeff_column_np(arr2, idx=1),
        np.asarray(solve._zero_coeff_column(arr2, idx=1)),
    )
    np.testing.assert_allclose(arr2, arr2_original)
    np.testing.assert_allclose(solve._zero_coeff_column_np(arr2, idx=9), arr2_original)

    replacement = np.full((4, 2), -3.0)
    np.testing.assert_allclose(
        solve._replace_mode_slice_np(arr3, mode_idx=0, replacement=replacement),
        np.asarray(solve._replace_mode_slice(arr3, mode_idx=0, replacement=replacement)),
    )
    np.testing.assert_allclose(arr3, arr3_original)
    assert solve._replace_mode_slice_np(None, mode_idx=0, replacement=replacement) is None

    scale = np.asarray([0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(
        solve._scale_mode_slice_np(arr3, mode_idx=2, scale=scale),
        np.asarray(solve._scale_mode_slice(arr3, mode_idx=2, scale=scale)),
    )
    np.testing.assert_allclose(arr3, arr3_original)
    assert solve._scale_mode_slice_np(None, mode_idx=2, scale=scale) is None


def test_grad_and_preconditioner_tolerance_helpers_are_small_and_deterministic():
    grad = _state()
    expected = np.sqrt(
        np.mean(
            np.asarray(grad.Rcos) ** 2
            + np.asarray(grad.Rsin) ** 2
            + np.asarray(grad.Zcos) ** 2
            + np.asarray(grad.Zsin) ** 2
            + np.asarray(grad.Lcos) ** 2
            + np.asarray(grad.Lsin) ** 2
        )
    )
    assert solve._grad_rms_state(grad) == pytest.approx(expected)

    assert solve._resolve_grad_tol(1.0e-5, grad_rms0=10.0, dtype=np.float64) == pytest.approx(1.0e-5)
    with pytest.raises(ValueError, match="grad_tol must be >= 0"):
        solve._resolve_grad_tol(-1.0, grad_rms0=10.0, dtype=np.float64)
    assert solve._resolve_grad_tol(None, grad_rms0=4.0, dtype=np.float64) == pytest.approx(
        np.sqrt(np.finfo(np.float64).eps) * 4.0
    )

    assert solve._resolve_cg_tol(2.0e-4, current_obj=1.0, initial_obj=2.0, target_obj=0.5, dtype=np.float64) == pytest.approx(
        2.0e-4
    )
    with pytest.raises(ValueError, match="cg_tol must be > 0"):
        solve._resolve_cg_tol(0.0, current_obj=1.0, initial_obj=2.0, target_obj=0.5, dtype=np.float64)
    auto_cg = solve._resolve_cg_tol(None, current_obj=1.0, initial_obj=3.0, target_obj=0.5, dtype=np.float64)
    assert auto_cg == pytest.approx(0.25)

    assert solve._resolve_lbfgs_curvature_tol([1.0, 2.0], [3.0, 4.0]) == pytest.approx(
        np.finfo(np.float64).eps * np.linalg.norm([1.0, 2.0]) * np.linalg.norm([3.0, 4.0])
    )


def test_apply_preconditioner_mode_diag_radial_and_validation_branches():
    grad = _state(ns=4, k=3)
    static = SimpleNamespace(
        cfg=SimpleNamespace(nfp=2),
        modes=SimpleNamespace(m=np.asarray([0, 1, 2]), n=np.asarray([0, 1, -1])),
    )

    assert solve._apply_preconditioner(grad, static, kind="none") is grad
    mode_diag = solve._apply_preconditioner(grad, static, kind="mode_diag", exponent=1.0)
    weights = (1.0 + np.asarray([0.0, 1.0 + 4.0, 4.0 + 4.0])) ** -1.0
    np.testing.assert_allclose(np.asarray(mode_diag.Rcos), np.asarray(grad.Rcos) * weights[None, :])
    np.testing.assert_allclose(np.asarray(mode_diag.Lsin), np.asarray(grad.Lsin) * weights[None, :])

    radial = solve._apply_preconditioner(grad, static, kind="radial_tridi", radial_alpha=0.25)
    assert np.asarray(radial.Rcos).shape == np.asarray(grad.Rcos).shape
    np.testing.assert_allclose(np.asarray(radial.Rcos)[0], np.asarray(grad.Rcos)[0])
    np.testing.assert_allclose(np.asarray(radial.Rcos)[-1], np.asarray(grad.Rcos)[-1])

    both = solve._apply_preconditioner(
        grad,
        static,
        kind="mode_diag+radial_tridi",
        exponent=1.0,
        radial_alpha=0.25,
    )
    assert np.asarray(both.Rcos).shape == np.asarray(grad.Rcos).shape

    with pytest.raises(ValueError, match="exponent must be > 0"):
        solve._apply_preconditioner(grad, static, kind="mode_diag", exponent=0.0)
    with pytest.raises(ValueError, match="radial_alpha must be > 0"):
        solve._apply_preconditioner(grad, static, kind="radial_tridi", radial_alpha=0.0)
    with pytest.raises(ValueError, match="Unknown preconditioner"):
        solve._apply_preconditioner(grad, static, kind="bogus")


def test_solve_runtime_reexports_and_scan_chunk_wrapper_are_accessible(monkeypatch):
    assert solve._hash_array_bytes is solve_runtime._hash_array_bytes
    assert solve._tree_has_tracer is solve_runtime._tree_has_tracer
    assert solve._parse_iter_list is solve_runtime._parse_iter_list
    assert solve._default_scan_core is solve_runtime._default_scan_core
    assert solve._scan_fallback_policy is solve_runtime._scan_fallback_policy
    assert solve._residual_convergence_flags is solve_runtime._residual_convergence_flags

    assert solve._default_scan_core(scan_core_env="", scan_minimal=True, fsq_total_target=1.0e-8) is True
    assert solve._parse_iter_list("2,4-5,bad") == {2, 4, 5}
    assert solve._residual_convergence_flags(fsqr=1.0e-3, fsqz=2.0e-3, fsql=3.0e-3, ftol=1.0e-4, fsq_total_target=1.0e-2) == (
        False,
        True,
        True,
    )

    monkeypatch.setattr(solve, "_scan_backend_name", lambda: "gpu")
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)
    assert solve._scan_chunk_settings(max_iter_scan=11, nstep_screen=4, need_print=True, lthreed=True) == (4, False)

    monkeypatch.setenv("VMEC_JAX_SCAN_CHUNK_SIZE", "bad")
    assert solve._scan_chunk_settings(max_iter_scan=11, nstep_screen=4, need_print=False, lthreed=True) == (4, True)


def test_solve_facade_private_assignment_forwards_to_residual_iteration(monkeypatch):
    original = residual_iteration._scan_backend_name
    replacement = lambda: "synthetic-backend"

    monkeypatch.setattr(solve, "_scan_backend_name", replacement)

    assert residual_iteration._scan_backend_name is replacement

    monkeypatch.setattr(solve, "_scan_backend_name", original)


def test_solve_facade_public_assignment_forwards_to_fixed_boundary_api(monkeypatch):
    original = fixed_boundary_api.solve_lambda_gd
    replacement = lambda *args, **kwargs: "synthetic-result"

    monkeypatch.setattr(solve, "solve_lambda_gd", replacement)

    assert fixed_boundary_api.solve_lambda_gd is replacement

    monkeypatch.setattr(solve, "solve_lambda_gd", original)


def test_residual_iter_config_helpers_reexported_through_solve():
    badjac = solve._parse_bad_jacobian_config(
        {
            "VMEC_JAX_BADJAC_MODE": "state",
            "VMEC_JAX_DUMP_PTAU_STATE": "1",
            "VMEC_JAX_BADJAC_STATE_PROBE": "true",
            "VMEC_JAX_PTAU_TOL": "-1.0e-4",
            "VMEC_JAX_PTAU_TOL_REL": "2.0e-3",
        }
    )
    assert badjac.mode == "state"
    assert badjac.use_state is True
    assert badjac.dump_ptau_state is True
    assert badjac.state_probe is True
    assert solve._bad_jacobian_tau_tolerance(ptau_tol=badjac.ptau_tol, ptau_tol_rel=badjac.ptau_tol_rel, tau_scale=0.25) == pytest.approx(
        5.0e-4
    )

    debug = solve._resolve_debug_print_config(
        print_env="yes",
        mode_env="io_callback",
        ordered_env="false",
        io_callback_available=False,
    )
    assert debug.print_live is True
    assert debug.mode == "debug_print"
    assert debug.ordered is False
    assert solve._normalize_debug_print_mode("debug_callback") == "debug_callback"
    assert solve._normalize_debug_print_mode("not-real") == "debug_print"

    chunked = solve._resolve_chunked_scan_config(
        use_scan=True,
        state_has_tracer=True,
        scan_fallback_enabled=True,
        chunked_env="1",
    )
    assert chunked.force_chunked_scan is False
    assert chunked.scan_fallback_enabled is False
    assert chunked.differentiating_scan is True

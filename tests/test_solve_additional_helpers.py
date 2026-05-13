from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.solve import (
    _free_boundary_iter_controls,
    _grad_rms_state,
    _half_mesh_from_full_mesh,
    _jit_cache_get,
    _jit_cache_limit,
    _jit_cache_put,
    _mask_grad_for_constraints,
    _replace_mode_slice,
    _replace_mode_slice_np,
    _resolve_cg_tol,
    _resolve_grad_tol,
    _resolve_lbfgs_curvature_tol,
    _resolve_lm_damping,
    _s_half_from_full_mesh_s,
    _scale_mode_slice,
    _scale_mode_slice_np,
    _update_state_gd,
    _vmec_force_flux_profiles,
    _zero_coeff_column,
    _zero_coeff_column_np,
)
from vmec_jax.state import StateLayout, VMECState


def _state_from_value(value: float, *, ns: int = 3, k: int = 3) -> VMECState:
    layout = StateLayout(ns=ns, K=k, lasym=False)
    arr = np.full((ns, k), float(value), dtype=float)
    return VMECState(
        layout=layout,
        Rcos=arr.copy(),
        Rsin=arr.copy(),
        Zcos=arr.copy(),
        Zsin=arr.copy(),
        Lcos=arr.copy(),
        Lsin=arr.copy(),
    )


def test_jit_cache_limit_put_and_lru_policy(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "-4")
    assert _jit_cache_limit("VMEC_JAX_TEST_CACHE", 3) == 0

    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "not-an-int")
    assert _jit_cache_limit("VMEC_JAX_TEST_CACHE", 3) == 3

    cache: OrderedDict[tuple, object] = OrderedDict()
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "0")
    value = object()
    assert _jit_cache_put(cache, ("disabled",), value, env_name="VMEC_JAX_TEST_CACHE", default=2) is value
    assert cache == {}

    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "2")
    _jit_cache_put(cache, ("a",), "A", env_name="VMEC_JAX_TEST_CACHE", default=2)
    _jit_cache_put(cache, ("b",), "B", env_name="VMEC_JAX_TEST_CACHE", default=2)
    assert _jit_cache_get(cache, ("a",)) == "A"
    _jit_cache_put(cache, ("c",), "C", env_name="VMEC_JAX_TEST_CACHE", default=2)
    assert list(cache.keys()) == [("a",), ("c",)]
    assert _jit_cache_get(cache, ("missing",)) is None


def test_mode_slice_helpers_cover_invalid_none_and_singleton_branches():
    arr = np.arange(6, dtype=float).reshape(2, 3)
    np.testing.assert_allclose(np.asarray(_zero_coeff_column(arr, idx=-1)), arr)
    np.testing.assert_allclose(np.asarray(_zero_coeff_column(arr, idx=3)), arr)
    np.testing.assert_allclose(np.asarray(_zero_coeff_column(np.ones((2, 1)), idx=0)), np.zeros((2, 1)))
    np.testing.assert_allclose(_zero_coeff_column_np(arr, idx=1), np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]]))

    cube = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    repl = np.full((2, 2), -5.0)
    assert _replace_mode_slice(None, mode_idx=0, replacement=repl) is None
    assert _scale_mode_slice(None, mode_idx=0, scale=np.ones(2)) is None
    np.testing.assert_allclose(np.asarray(_replace_mode_slice(cube, mode_idx=9, replacement=repl)), cube)
    np.testing.assert_allclose(np.asarray(_scale_mode_slice(cube, mode_idx=-1, scale=np.ones(2))), cube)

    one_mode = cube[:, :1, :]
    np.testing.assert_allclose(np.asarray(_replace_mode_slice(one_mode, mode_idx=0, replacement=repl)), repl[:, None, :])
    np.testing.assert_allclose(_replace_mode_slice_np(one_mode, mode_idx=0, replacement=repl), repl[:, None, :])
    np.testing.assert_allclose(_scale_mode_slice_np(one_mode, mode_idx=0, scale=np.array([2.0, 3.0])), one_mode * np.array([2.0, 3.0])[:, None, None])


def test_state_update_mask_and_rms_helpers_are_componentwise():
    pytest.importorskip("jax")

    state = _state_from_value(2.0)
    grad = _state_from_value(1.0)
    updated = _update_state_gd(state, grad, step=0.25, scale_rz=2.0, scale_l=4.0)
    for field in ("Rcos", "Rsin", "Zcos", "Zsin"):
        np.testing.assert_allclose(np.asarray(getattr(updated, field)), 1.5)
    for field in ("Lcos", "Lsin"):
        np.testing.assert_allclose(np.asarray(getattr(updated, field)), 1.0)

    assert _grad_rms_state(grad) == pytest.approx(np.sqrt(6.0))

    static = SimpleNamespace(modes=SimpleNamespace(m=np.array([0, 1, 2])))
    masked = _mask_grad_for_constraints(grad, static, idx00=0, mask_lambda_axis=False)
    for field in ("Rcos", "Rsin", "Zcos", "Zsin"):
        got = np.asarray(getattr(masked, field))
        np.testing.assert_allclose(got[-1, :], 0.0)
        np.testing.assert_allclose(got[0, :], np.array([1.0, 0.0, 0.0]))
    for field in ("Lcos", "Lsin"):
        got = np.asarray(getattr(masked, field))
        np.testing.assert_allclose(got[:, 0], 0.0)
        np.testing.assert_allclose(got[0, 1:], 1.0)


def test_tolerance_resolvers_validate_explicit_values_and_scale_by_dtype():
    assert _resolve_grad_tol(0.0, grad_rms0=10.0, dtype=np.float64) == 0.0
    with pytest.raises(ValueError, match="grad_tol"):
        _resolve_grad_tol(-1.0, grad_rms0=10.0, dtype=np.float64)
    assert _resolve_grad_tol(None, grad_rms0=4.0, dtype=np.float32) == pytest.approx(
        np.sqrt(np.finfo(np.float32).eps) * 4.0
    )

    with pytest.raises(ValueError, match="cg_tol"):
        _resolve_cg_tol(0.0, current_obj=1.0, initial_obj=1.0, target_obj=0.0, dtype=np.float64)
    assert _resolve_cg_tol(None, current_obj=1.0, initial_obj=3.0, target_obj=0.0, dtype=np.float64) == pytest.approx(0.25)

    with pytest.raises(ValueError, match="damping"):
        _resolve_lm_damping(-1.0, curvature_scale=2.0, dtype=np.float64)
    assert _resolve_lm_damping(None, curvature_scale=2.0, dtype=np.float64) == pytest.approx(
        np.sqrt(np.finfo(np.float64).eps) * 2.0
    )

    assert _resolve_lbfgs_curvature_tol(np.array([3.0, 4.0]), np.array([0.0, 6.0])) == pytest.approx(
        np.finfo(float).eps * 30.0
    )


def test_mesh_flux_and_free_boundary_cadence_helpers():
    np.testing.assert_allclose(np.asarray(_s_half_from_full_mesh_s(np.array([0.0]))), np.array([0.0]))
    np.testing.assert_allclose(
        np.asarray(_s_half_from_full_mesh_s(np.array([0.0, 0.25, 1.0]))),
        np.array([0.0, 0.125, 0.625]),
    )
    np.testing.assert_allclose(
        np.asarray(_half_mesh_from_full_mesh(np.array([2.0, 4.0, 10.0]))),
        np.array([2.0, 3.0, 7.0]),
    )

    phipf_internal, chipf_internal, chips_eff = _vmec_force_flux_profiles(
        phipf=np.array([2.0, 4.0]),
        chipf=None,
        signgs=1,
        flux_is_internal=True,
    )
    np.testing.assert_allclose(np.asarray(phipf_internal), np.array([2.0, 4.0]))
    assert chipf_internal is None
    np.testing.assert_allclose(np.asarray(chips_eff), np.zeros(2))

    phipf_external, _, chips_iota = _vmec_force_flux_profiles(
        phipf=np.array([2.0 * np.pi, 4.0 * np.pi]),
        chipf=None,
        signgs=1,
        flux_is_internal=False,
        iotaf=np.array([3.0, 5.0]),
    )
    np.testing.assert_allclose(np.asarray(phipf_external), np.array([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(chips_iota), np.array([3.0, 10.0]))

    assert _free_boundary_iter_controls(iter2=5, iter1=1, nvacskip=0) == (1, 0)
    assert _free_boundary_iter_controls(iter2=6, iter1=1, nvacskip=4) == (2, 1)

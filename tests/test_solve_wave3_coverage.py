from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import has_jax
from vmec_jax.config import VMECConfig
from vmec_jax.solve import (
    _apply_preconditioner,
    _enforce_fixed_boundary_and_axis,
    _enforce_lambda_gauge,
    _grad_rms_state,
    _mask_grad_for_constraints,
    _mode00_index,
    _resolve_lm_damping,
    _update_state_gd,
    solve_fixed_boundary_gd,
    solve_fixed_boundary_gn_vmec_residual,
    solve_fixed_boundary_lbfgs,
    solve_fixed_boundary_lbfgs_vmec_residual,
    solve_fixed_boundary_residual_iter,
    solve_lambda_gd,
)
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.static import build_static


pytestmark = pytest.mark.skipif(not has_jax(), reason="solve helpers require JAX array updates")


def _small_static(*, ns: int = 5):
    cfg = VMECConfig(
        ns=ns,
        mpol=3,
        ntor=1,
        nfp=2,
        lasym=False,
        lthreed=True,
        lconm1=True,
        ntheta=8,
        nzeta=4,
    )
    return build_static(cfg)


def _state(static, *, start: float = 1.0) -> VMECState:
    layout = StateLayout(ns=int(static.cfg.ns), K=int(static.modes.K), lasym=bool(static.cfg.lasym))
    base = np.arange(layout.ns * layout.K, dtype=float).reshape(layout.ns, layout.K) + start
    return VMECState(
        layout=layout,
        Rcos=base,
        Rsin=base + 100.0,
        Zcos=base + 200.0,
        Zsin=base + 300.0,
        Lcos=base + 400.0,
        Lsin=base + 500.0,
    )


def _assert_state_allclose(actual: VMECState, expected: VMECState) -> None:
    for name in ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin"):
        np.testing.assert_allclose(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name)))


def _idx(static, m_value: int, n_value: int) -> int:
    for k, (m, n) in enumerate(zip(static.modes.m, static.modes.n)):
        if int(m) == m_value and int(n) == n_value:
            return k
    raise AssertionError(f"mode ({m_value}, {n_value}) not present in synthetic static")


def _solver_common_kwargs():
    return {
        "phipf": np.ones(3),
        "chipf": np.ones(3),
        "signgs": 1,
        "lamscale": 1.0,
        "verbose": False,
    }


def test_mode00_index_and_numpy_lambda_gauge_paths():
    modes_without_gauge = SimpleNamespace(m=np.array([1, 2]), n=np.array([0, 1]))
    assert _mode00_index(modes_without_gauge) is None

    lcos = np.arange(12.0).reshape(3, 4)
    lsin = lcos + 20.0
    same_lcos, same_lsin = _enforce_lambda_gauge(lcos, lsin, idx00=None)
    assert same_lcos is lcos
    assert same_lsin is lsin

    gauged_lcos, gauged_lsin = _enforce_lambda_gauge(lcos, lsin, idx00=2)
    np.testing.assert_allclose(gauged_lcos[:, 2], 0.0)
    np.testing.assert_allclose(gauged_lsin[:, 2], 0.0)
    np.testing.assert_allclose(lcos[:, 2], np.array([2.0, 6.0, 10.0]))
    np.testing.assert_allclose(lsin[:, 2], np.array([22.0, 26.0, 30.0]))


def test_enforce_fixed_boundary_and_axis_default_constraints():
    static = _small_static(ns=4)
    state = _state(static)
    idx00 = _mode00_index(static.modes)
    assert idx00 is not None

    k10 = _idx(static, 1, 0)
    edge_rcos = np.full(static.modes.K, -1.0)
    edge_rsin = np.full(static.modes.K, -2.0)
    edge_zcos = np.full(static.modes.K, -3.0)
    edge_zsin = np.full(static.modes.K, -4.0)

    out = _enforce_fixed_boundary_and_axis(
        state,
        static,
        edge_Rcos=edge_rcos,
        edge_Rsin=edge_rsin,
        edge_Zcos=edge_zcos,
        edge_Zsin=edge_zsin,
        idx00=idx00,
    )

    np.testing.assert_allclose(np.asarray(out.Rcos)[-1], edge_rcos)
    np.testing.assert_allclose(np.asarray(out.Rsin)[-1], edge_rsin)
    np.testing.assert_allclose(np.asarray(out.Zcos)[-1], edge_zcos)
    np.testing.assert_allclose(np.asarray(out.Zsin)[-1], edge_zsin)
    assert float(np.asarray(out.Rcos)[0, k10]) == 0.0
    assert float(np.asarray(out.Zsin)[0, k10]) == 0.0
    np.testing.assert_allclose(np.asarray(out.Lcos)[0], 0.0)
    np.testing.assert_allclose(np.asarray(out.Lsin)[0], 0.0)
    np.testing.assert_allclose(np.asarray(out.Lcos)[:, idx00], 0.0)
    np.testing.assert_allclose(np.asarray(out.Lsin)[:, idx00], 0.0)


def test_enforce_fixed_boundary_and_axis_disabled_constraints_are_noops():
    static = _small_static(ns=4)
    state = _state(static)
    k10 = _idx(static, 1, 0)

    out = _enforce_fixed_boundary_and_axis(
        state,
        static,
        edge_Rcos=np.full(static.modes.K, -1.0),
        edge_Rsin=np.full(static.modes.K, -2.0),
        edge_Zcos=np.full(static.modes.K, -3.0),
        edge_Zsin=np.full(static.modes.K, -4.0),
        enforce_axis=False,
        enforce_edge=False,
        enforce_lambda_axis=False,
        idx00=None,
    )

    _assert_state_allclose(out, state)
    assert float(np.asarray(out.Rcos)[0, k10]) == float(np.asarray(state.Rcos)[0, k10])
    assert float(np.asarray(out.Lcos)[0, 0]) == float(np.asarray(state.Lcos)[0, 0])


def test_mask_grad_constraint_switches_cover_lambda_axis_and_gauge_paths():
    static = _small_static(ns=4)
    grad = _state(static)
    idx00 = _mode00_index(static.modes)
    assert idx00 is not None
    k10 = _idx(static, 1, 0)

    masked = _mask_grad_for_constraints(grad, static, idx00=idx00)
    for name in ("Rcos", "Rsin", "Zcos", "Zsin"):
        arr = np.asarray(getattr(masked, name))
        assert float(arr[-1, k10]) == 0.0
        assert float(arr[0, k10]) == 0.0
    np.testing.assert_allclose(np.asarray(masked.Lcos)[0], 0.0)
    np.testing.assert_allclose(np.asarray(masked.Lsin)[0], 0.0)
    np.testing.assert_allclose(np.asarray(masked.Lcos)[:, idx00], 0.0)
    np.testing.assert_allclose(np.asarray(masked.Lsin)[:, idx00], 0.0)

    lambda_unmasked = _mask_grad_for_constraints(grad, static, idx00=None, mask_lambda_axis=False)
    np.testing.assert_allclose(np.asarray(lambda_unmasked.Lcos)[0], np.asarray(grad.Lcos)[0])
    np.testing.assert_allclose(np.asarray(lambda_unmasked.Lsin)[:, idx00], np.asarray(grad.Lsin)[:, idx00])


def test_grad_rms_and_update_state_use_distinct_rz_and_lambda_scales():
    static = _small_static(ns=3)
    state = _state(static)
    grad = _state(static, start=0.5)

    updated = _update_state_gd(state, grad, step=0.25, scale_rz=2.0, scale_l=4.0)
    np.testing.assert_allclose(np.asarray(updated.Rcos), np.asarray(state.Rcos) - 0.5 * np.asarray(grad.Rcos))
    np.testing.assert_allclose(np.asarray(updated.Zsin), np.asarray(state.Zsin) - 0.5 * np.asarray(grad.Zsin))
    np.testing.assert_allclose(np.asarray(updated.Lcos), np.asarray(state.Lcos) - np.asarray(grad.Lcos))
    np.testing.assert_allclose(np.asarray(updated.Lsin), np.asarray(state.Lsin) - np.asarray(grad.Lsin))

    manual = sum(np.asarray(getattr(grad, name)) ** 2 for name in ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin"))
    assert _grad_rms_state(grad) == pytest.approx(float(np.sqrt(np.mean(manual))))


def test_apply_preconditioner_noop_and_error_branches():
    static = _small_static(ns=4)
    grad = _state(static)

    assert _apply_preconditioner(grad, static, kind=" none ") is grad
    assert _apply_preconditioner(grad, static, kind=" + , ") is grad

    with pytest.raises(ValueError, match="exponent"):
        _apply_preconditioner(grad, static, kind="mode_diag", exponent=0.0)
    with pytest.raises(ValueError, match="radial_alpha"):
        _apply_preconditioner(grad, static, kind="radial_tridi", radial_alpha=0.0)
    with pytest.raises(ValueError, match="Unknown preconditioner"):
        _apply_preconditioner(grad, static, kind="bogus", radial_alpha=0.5)


def test_apply_preconditioner_mode_diagonal_scales_every_state_block():
    static = _small_static(ns=4)
    grad = _state(static)

    out = _apply_preconditioner(grad, static, kind="mode_diag", exponent=1.0)

    m = np.asarray(static.modes.m, dtype=float)
    n = np.asarray(static.modes.n, dtype=float)
    weights = (1.0 + m**2 + (n * float(static.cfg.nfp)) ** 2) ** -1.0
    for name in ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin"):
        np.testing.assert_allclose(np.asarray(getattr(out, name)), np.asarray(getattr(grad, name)) * weights[None, :])


def test_apply_preconditioner_radial_tridi_handles_short_and_single_interior_meshes():
    short_static = _small_static(ns=2)
    short_grad = _state(short_static)
    short_out = _apply_preconditioner(short_grad, short_static, kind="radial_tridi", radial_alpha=0.25)
    _assert_state_allclose(short_out, short_grad)

    single_static = _small_static(ns=3)
    single_grad = _state(single_static)
    alpha = 0.25
    single_out = _apply_preconditioner(single_grad, single_static, kind="radial_tridi", radial_alpha=alpha)
    rhs = np.asarray(single_grad.Rcos)
    expected_mid = (rhs[1] + alpha * rhs[0] + alpha * rhs[2]) / (1.0 + 2.0 * alpha)
    np.testing.assert_allclose(np.asarray(single_out.Rcos)[0], rhs[0])
    np.testing.assert_allclose(np.asarray(single_out.Rcos)[1], expected_mid)
    np.testing.assert_allclose(np.asarray(single_out.Rcos)[2], rhs[2])


def test_apply_preconditioner_radial_tridi_scan_branch_satisfies_system():
    static = _small_static(ns=5)
    grad = _state(static)
    alpha = 0.2

    out = _apply_preconditioner(grad, static, kind="radial_tridi", radial_alpha=alpha)

    rhs = np.asarray(grad.Rcos)
    x = np.asarray(out.Rcos)
    np.testing.assert_allclose(x[0], rhs[0])
    np.testing.assert_allclose(x[-1], rhs[-1])
    lhs = (1.0 + 2.0 * alpha) * x[1:-1] - alpha * x[:-2] - alpha * x[2:]
    np.testing.assert_allclose(lhs, rhs[1:-1], rtol=1e-12, atol=1e-12)


def test_apply_preconditioner_radial_tridi_rejects_non_matrix_blocks():
    static = _small_static(ns=4)
    bad = VMECState(
        layout=StateLayout(ns=1, K=int(static.modes.K), lasym=False),
        Rcos=np.arange(static.modes.K, dtype=float),
        Rsin=np.arange(static.modes.K, dtype=float),
        Zcos=np.arange(static.modes.K, dtype=float),
        Zsin=np.arange(static.modes.K, dtype=float),
        Lcos=np.arange(static.modes.K, dtype=float),
        Lsin=np.arange(static.modes.K, dtype=float),
    )

    with pytest.raises(ValueError, match="expected"):
        _apply_preconditioner(bad, static, kind="radial_tridi", radial_alpha=0.25)


def test_resolve_lm_damping_validation_and_adaptive_scale():
    assert _resolve_lm_damping(0.25, curvature_scale=2.0, dtype=np.float64) == pytest.approx(0.25)
    assert _resolve_lm_damping(None, curvature_scale=4.0, dtype=np.float64) == pytest.approx(
        np.sqrt(np.finfo(np.float64).eps) * 4.0
    )
    with pytest.raises(ValueError, match="damping"):
        _resolve_lm_damping(-1.0, curvature_scale=2.0, dtype=np.float64)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda state, static: solve_lambda_gd(
                state,
                static,
                **_solver_common_kwargs(),
                max_iter=0,
            ),
            "max_iter",
        ),
        (
            lambda state, static: solve_lambda_gd(
                state,
                static,
                **_solver_common_kwargs(),
                max_backtracks=-1,
            ),
            "max_backtracks",
        ),
        (
            lambda state, static: solve_lambda_gd(
                state,
                static,
                **_solver_common_kwargs(),
                bt_factor=1.0,
            ),
            "bt_factor",
        ),
        (
            lambda state, static: solve_lambda_gd(
                state,
                static,
                **_solver_common_kwargs(),
                preconditioner="radial_tridi",
            ),
            "Unknown preconditioner",
        ),
        (
            lambda state, static: solve_lambda_gd(
                state,
                static,
                **_solver_common_kwargs(),
                preconditioner="mode_diag",
                precond_exponent=0.0,
            ),
            "precond_exponent",
        ),
        (
            lambda state, static: solve_fixed_boundary_gd(
                state,
                static,
                **_solver_common_kwargs(),
                gamma=1.0,
            ),
            "gamma=1",
        ),
        (
            lambda state, static: solve_fixed_boundary_gd(
                state,
                static,
                **_solver_common_kwargs(),
                pressure=np.ones(2),
            ),
            "pressure",
        ),
        (
            lambda state, static: solve_fixed_boundary_lbfgs(
                state,
                static,
                **_solver_common_kwargs(),
                history_size=0,
            ),
            "history_size",
        ),
        (
            lambda state, static: solve_fixed_boundary_lbfgs(
                state,
                static,
                **_solver_common_kwargs(),
                pressure=np.ones(2),
            ),
            "pressure",
        ),
        (
            lambda state, static: solve_fixed_boundary_lbfgs_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                w_rz=-1.0,
            ),
            "nonnegative",
        ),
        (
            lambda state, static: solve_fixed_boundary_lbfgs_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                objective_scale=0.0,
            ),
            "objective_scale",
        ),
        (
            lambda state, static: solve_fixed_boundary_lbfgs_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                scale_l=0.0,
            ),
            "scale_rz and scale_l",
        ),
        (
            lambda state, static: solve_fixed_boundary_lbfgs_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                max_backtracks=-1,
            ),
            "max_backtracks",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                damping_increase=1.0,
            ),
            "damping_increase",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                damping_decrease=0.0,
            ),
            "damping_decrease",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                max_damping=0.0,
            ),
            "max_damping",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                max_retries=-1,
            ),
            "max_retries",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                zero_m1_iters=-1,
            ),
            "zero_m1_iters",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                zero_m1_fsqz_thresh=-1.0,
            ),
            "zero_m1_fsqz_thresh",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                w_l=-1.0,
            ),
            "nonnegative",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                max_iter=0,
            ),
            "max_iter",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                cg_maxiter=0,
            ),
            "cg_maxiter",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                bt_factor=0.0,
            ),
            "bt_factor",
        ),
        (
            lambda state, static: solve_fixed_boundary_gn_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                objective_scale=0.0,
            ),
            "objective_scale",
        ),
        (
            lambda state, static: solve_fixed_boundary_residual_iter(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                max_iter=0,
            ),
            "max_iter",
        ),
        (
            lambda state, static: solve_fixed_boundary_residual_iter(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                step_size=0.0,
            ),
            "step_size",
        ),
    ],
)
def test_solver_entry_validation_branches_short_circuit_before_kernels(call, match):
    static = _small_static(ns=3)
    state = _state(static)

    with pytest.raises(ValueError, match=match):
        call(state, static)

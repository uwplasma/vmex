from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.solve_lambda_optimizer import solve_lambda_gd_impl
from vmec_jax.solve import _resolve_lbfgs_curvature_tol
from vmec_jax.solve_optimizer_helpers import (
    ensure_descent_direction,
    lbfgs_curvature_tolerance,
    lbfgs_two_loop_direction,
)
from vmec_jax.state import StateLayout, VMECState


def test_lbfgs_two_loop_direction_empty_history_is_steepest_descent():
    g = jnp.asarray([1.0, -2.0, 3.0])

    direction = lbfgs_two_loop_direction(g, [], [])

    np.testing.assert_allclose(np.asarray(direction), [-1.0, 2.0, -3.0])


def test_lbfgs_two_loop_direction_uses_latest_curvature_pair_scaling():
    g = jnp.asarray([4.0, 3.0])
    s_hist = [jnp.asarray([1.0, 0.0])]
    y_hist = [jnp.asarray([2.0, 0.0])]

    direction = lbfgs_two_loop_direction(g, s_hist, y_hist)

    np.testing.assert_allclose(np.asarray(direction), [-2.0, -1.5])


def test_lbfgs_curvature_tolerance_tracks_dtype_and_solve_alias():
    s = np.asarray([3.0, 4.0], dtype=np.float32)
    y = np.asarray([0.0, 6.0], dtype=np.float32)
    expected = np.finfo(np.float32).eps * np.linalg.norm(s.ravel()) * np.linalg.norm(y.ravel())

    assert lbfgs_curvature_tolerance(s, y) == pytest.approx(expected)
    assert _resolve_lbfgs_curvature_tol(s, y) == pytest.approx(expected)


def test_ensure_descent_direction_preserves_descent_and_falls_back_otherwise():
    g = jnp.asarray([1.0, 2.0])
    descent = jnp.asarray([-0.5, -1.0])
    ascent = jnp.asarray([0.5, 1.0])

    direction, gtp, fallback = ensure_descent_direction(g, descent)
    np.testing.assert_allclose(np.asarray(direction), np.asarray(descent))
    assert gtp == pytest.approx(-2.5)
    assert fallback is False

    direction, gtp, fallback = ensure_descent_direction(g, ascent)
    np.testing.assert_allclose(np.asarray(direction), [-1.0, -2.0])
    assert gtp == pytest.approx(2.5)
    assert fallback is True


def test_lambda_optimizer_impl_runs_with_injected_tiny_quadratic_problem():
    jax = pytest.importorskip("jax")
    from types import SimpleNamespace

    layout = StateLayout(ns=3, K=2, lasym=False)
    zeros = jnp.zeros((3, 2), dtype=jnp.float64)
    lcos = zeros.at[1, 1].set(1.0).at[2, 1].set(0.5)
    state0 = VMECState(
        layout=layout,
        Rcos=zeros,
        Rsin=zeros,
        Zcos=zeros,
        Zsin=zeros,
        Lcos=lcos,
        Lsin=zeros,
    )
    static = SimpleNamespace(
        cfg=SimpleNamespace(nfp=1),
        modes=SimpleNamespace(m=jnp.asarray([0, 1]), n=jnp.asarray([0, 0])),
        basis=object(),
        s=jnp.asarray([0.0, 0.5, 1.0]),
        grid=SimpleNamespace(theta=jnp.asarray([0.0, jnp.pi]), zeta=jnp.asarray([0.0])),
    )

    def validate_options(**kwargs):
        return SimpleNamespace(
            max_iter=int(kwargs["max_iter"]),
            max_backtracks=int(kwargs["max_backtracks"]),
            bt_factor=float(kwargs["bt_factor"]),
            preconditioner=str(kwargs["preconditioner"]),
            precond_exponent=float(kwargs["precond_exponent"]),
        )

    def fake_eval_geom(state, _static):
        shape = (state.layout.ns, 1, 1)
        return SimpleNamespace(
            g_tt=jnp.ones(shape, dtype=jnp.float64),
            g_tp=jnp.zeros(shape, dtype=jnp.float64),
            g_pp=jnp.ones(shape, dtype=jnp.float64),
            sqrtg=jnp.ones(shape, dtype=jnp.float64),
        )

    def fake_dtheta(Lcos, _Lsin, _basis, *, coeffs_internal):
        del coeffs_internal
        return jnp.sum(jnp.asarray(Lcos), axis=1)[:, None, None]

    def fake_dzeta(_Lcos, Lsin, _basis, *, coeffs_internal):
        del coeffs_internal
        return jnp.sum(jnp.asarray(Lsin), axis=1)[:, None, None]

    def fake_bsup_from_sqrtg_lambda(**kwargs):
        return kwargs["lam_u"], kwargs["lam_v"]

    def enforce_lambda_gauge(Lcos, Lsin, *, idx00):
        return jnp.asarray(Lcos).at[:, idx00].set(0.0), jnp.asarray(Lsin).at[:, idx00].set(0.0)

    result = solve_lambda_gd_impl(
        state0,
        static,
        phipf=jnp.ones(3),
        chipf=jnp.zeros(3),
        signgs=1,
        lamscale=1.0,
        max_iter=3,
        step_size=0.2,
        grad_tol=0.0,
        max_backtracks=2,
        preconditioner="mode_diag",
        precond_exponent=0.5,
        verbose=False,
        has_jax_func=lambda: True,
        validate_options_func=validate_options,
        mode00_index_func=lambda _modes: 0,
        eval_geom_func=fake_eval_geom,
        eval_fourier_dtheta_func=fake_dtheta,
        eval_fourier_dzeta_phys_func=fake_dzeta,
        bsup_from_sqrtg_lambda_func=fake_bsup_from_sqrtg_lambda,
        angle_steps_func=lambda *, ntheta, nzeta: (1.0, 1.0),
        enforce_lambda_gauge_func=enforce_lambda_gauge,
        resolve_grad_tol_func=lambda grad_tol, *, grad_rms0, dtype: 0.0,
        jax_module=jax,
        jnp_module=jnp,
        jit_func=lambda f: f,
    )

    assert result.n_iter >= 1
    assert result.wb_history[-1] < result.wb_history[0]
    assert result.step_history.shape[0] >= 1
    assert result.diagnostics["idx00"] == 0
    np.testing.assert_allclose(np.asarray(result.state.Lcos)[:, 0], 0.0)


def test_lambda_optimizer_impl_reports_missing_jax():
    with pytest.raises(ImportError, match="solve_lambda_gd requires JAX"):
        solve_lambda_gd_impl(
            object(),
            object(),
            phipf=(),
            chipf=(),
            signgs=1,
            lamscale=1.0,
            has_jax_func=lambda: False,
            jax_module=object(),
            jnp_module=object(),
            jit_func=lambda f: f,
        )

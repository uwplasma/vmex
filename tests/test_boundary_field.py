from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.config import load_config
from vmec_jax.driver import example_paths
from vmec_jax.field import b_cartesian_from_state
from vmec_jax.grids import AngleGrid
from vmec_jax.state import zeros_state
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


def _small_static(cfg, *, ntheta: int = 8, nzeta: int = 1):
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, int(nzeta), endpoint=False)
    cfg_small = replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))
    return build_static(cfg_small, grid=AngleGrid(theta=theta, zeta=zeta, nfp=cfg.nfp))


def test_b_cartesian_from_state_matches_wout_path():
    pytest.importorskip("netCDF4")
    enable_x64(True)

    input_path, wout_path = example_paths("circular_tokamak")
    if wout_path is None:
        pytest.skip("No reference wout file available for circular_tokamak")

    cfg, indata = load_config(str(input_path))
    static = _small_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    b_input = np.asarray(
        b_cartesian_from_state(
            state,
            static,
            indata=indata,
            signgs=wout.signgs,
        )
    )
    b_wout = np.asarray(b_cartesian_from_state(state, static, wout=wout))
    b_stored = np.asarray(
        b_cartesian_from_state(
            state,
            static,
            wout=wout,
            use_wout_bsup=True,
        )
    )

    assert b_input.shape == (static.grid.ntheta, static.grid.nzeta, 3)
    np.testing.assert_allclose(b_input, b_wout, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(b_stored, b_wout, rtol=1.0e-7, atol=1.0e-9)


def test_b_cartesian_from_state_jvp_matches_finite_difference():
    pytest.importorskip("netCDF4")
    pytest.importorskip("jax")

    from vmec_jax._compat import jax, jnp

    enable_x64(True)

    input_path, wout_path = example_paths("circular_tokamak")
    if wout_path is None:
        pytest.skip("No reference wout file available for circular_tokamak")

    cfg, indata = load_config(str(input_path))
    static = _small_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)

    def objective(state_in):
        field = b_cartesian_from_state(
            state_in,
            static,
            indata=indata,
            signgs=wout.signgs,
        )
        return jnp.sum(field * field)

    zero = zeros_state(state.layout, like=jnp.asarray(state.Rcos))
    tangent_rcos = jnp.zeros_like(jnp.asarray(state.Rcos)).at[-1, 1].set(1.0e-3)
    tangent = replace(zero, Rcos=tangent_rcos)

    _, tangent_value = jax.jvp(objective, (state,), (tangent,))

    eps = 1.0e-4
    state_plus = replace(state, Rcos=jnp.asarray(state.Rcos) + eps * tangent_rcos)
    state_minus = replace(state, Rcos=jnp.asarray(state.Rcos) - eps * tangent_rcos)
    finite_difference = (objective(state_plus) - objective(state_minus)) / (2.0 * eps)

    np.testing.assert_allclose(
        np.asarray(tangent_value),
        np.asarray(finite_difference),
        rtol=5.0e-6,
        atol=1.0e-9,
    )


@pytest.mark.py311_coverage_only
def test_exact_optimizer_b_cartesian_tangents_and_scalar_cotangent_match_dense_jacobian():
    jax = pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.optimization import FixedBoundaryExactOptimizer, boundary_param_specs
    from vmec_jax.state import unpack_state

    enable_x64(True)

    input_path, _wout_path = example_paths("circular_tokamak")
    cfg, indata = load_config(str(input_path))
    static = build_static(cfg)
    field_static = _small_static(cfg, ntheta=4, nzeta=2)
    boundary = boundary_from_indata(indata, static.modes)
    specs = boundary_param_specs(
        boundary,
        static.modes,
        max_mode=1,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )[:2]
    params = np.zeros(len(specs))

    def residuals_fn(state):
        field = b_cartesian_from_state(
            state,
            field_static,
            indata=indata,
            signgs=exact_opt._signgs,
        )
        return jnp.ravel(field)

    def objective_value_and_cotangent_from_packed(packed_state, layout):
        packed_state = jnp.asarray(packed_state, dtype=jnp.float64)

        def objective(packed):
            residuals = residuals_fn(unpack_state(packed, layout))
            return 0.5 * jnp.vdot(residuals, residuals)

        return jax.value_and_grad(objective)(packed_state)

    residuals_fn._state_objective_value_and_cotangent_from_packed = (
        objective_value_and_cotangent_from_packed
    )

    exact_opt = FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        specs,
        residuals_fn,
        inner_max_iter=2,
        inner_ftol=1.0e-5,
    )
    field, tangents = exact_opt.b_cartesian_tangent_columns_fun(params, field_static)
    residuals = exact_opt.residual_fun(params)
    jacobian = exact_opt.jacobian_fun(params)
    cost, gradient = exact_opt.objective_and_gradient_fun(params)

    np.testing.assert_allclose(
        field.reshape(-1),
        residuals,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert cost == pytest.approx(0.5 * float(np.dot(residuals, residuals)), rel=1.0e-12)
    np.testing.assert_allclose(
        gradient,
        jacobian.T @ residuals,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    tangent_jacobian = tangents.reshape((-1, len(specs)))
    np.testing.assert_allclose(
        tangent_jacobian,
        jacobian,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.py311_coverage_only
def test_exact_optimizer_jacobian_matches_finite_difference_residual():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.optimization import FixedBoundaryExactOptimizer, boundary_param_specs

    enable_x64(True)

    input_path, _wout_path = example_paths("circular_tokamak")
    cfg, indata = load_config(str(input_path))
    static = build_static(cfg)
    field_static = _small_static(cfg, ntheta=5, nzeta=4)
    boundary = boundary_from_indata(indata, static.modes)
    specs = boundary_param_specs(
        boundary,
        static.modes,
        max_mode=1,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )[:2]
    params = np.zeros(len(specs))

    def residuals_fn(state):
        field = b_cartesian_from_state(
            state,
            field_static,
            indata=indata,
            signgs=exact_opt._signgs,
        )
        return jnp.ravel(field)

    exact_opt = FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        specs,
        residuals_fn,
        inner_max_iter=2,
        inner_ftol=1.0e-5,
        trial_max_iter=2,
        trial_ftol=1.0e-5,
    )
    jacobian = exact_opt.jacobian_fun(params)

    eps = 1.0e-5
    fd_columns = []
    for col in range(len(specs)):
        step = np.zeros_like(params)
        step[col] = eps
        plus = exact_opt.residual_fun(params + step)
        minus = exact_opt.residual_fun(params - step)
        fd_columns.append((plus - minus) / (2.0 * eps))
    fd_jacobian = np.stack(fd_columns, axis=1)

    diff = jacobian - fd_jacobian
    rel_norm = float(np.linalg.norm(diff) / np.linalg.norm(fd_jacobian))
    max_abs = float(np.max(np.abs(diff)))
    assert rel_norm < 2.0e-3
    assert max_abs < 1.2e-2


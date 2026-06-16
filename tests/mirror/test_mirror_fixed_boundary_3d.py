from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    run_mirror_fixed_boundary,
)
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.state import MirrorState3D
from vmec_jax.mirror.kernels.fields import evaluate_field_3d
from vmec_jax.mirror.kernels.forces import projected_energy_residual_3d
from vmec_jax.mirror.kernels.geometry import evaluate_geometry_3d
from vmec_jax.mirror.solvers.fixed_boundary.optimizers import (
    pack_reduced_state_3d,
    reduced_3d_energy_and_gradient,
    reduced_a_mask_3d,
)

pytestmark = pytest.mark.mirror


def _case():
    config = MirrorConfig(MirrorResolution(ns=5, ntheta=9, nxi=11, mpol=3), z_min=-1.3, z_max=1.3)
    grid = config.build_grid()
    boundary = MirrorBoundary.cosine_modulated_radius(r0=0.3, a2=-0.25, epsilon=0.04, theta_mode=2)
    base = MirrorState3D.from_boundary(grid, boundary)
    s = grid.s_full[:, None, None]
    theta = grid.theta[None, :, None]
    xi = grid.xi[None, None, :]
    perturbation = 0.01 * s * (1.0 - s) * (1.0 - xi**2) * np.cos(2.0 * theta)
    state = MirrorState3D(a=base.a * (1.0 + perturbation), lam=np.zeros_like(base.lam))
    return config, grid, boundary, state


def test_reduced_3d_lbfgs_gradient_matches_central_difference():
    pytest.importorskip("scipy.optimize")
    config, grid, boundary, state = _case()
    del config
    lam = 0.003 * grid.s_full[:, None, None] * (1.0 - grid.s_full[:, None, None]) * np.sin(
        2.0 * grid.theta[None, :, None]
    ) * (1.0 - grid.xi[None, None, :] ** 2)
    state = MirrorState3D(a=state.a, lam=lam)
    psi = PsiPrimeProfile.constant(0.01)
    current = IPrimeProfile.zero()
    pressure = PressureProfile.zero()
    vector = pack_reduced_state_3d(state, grid, boundary)
    value, gradient = reduced_3d_energy_and_gradient(
        vector,
        grid,
        boundary,
        psi_prime=psi,
        i_prime=current,
        pressure=pressure,
        mu0=1.0,
    )

    num_a = int(np.count_nonzero(reduced_a_mask_3d(grid)))
    indices = [0, num_a] if num_a < vector.size else [0]
    step = 1.0e-5
    for index in indices:
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        e_plus, _ = reduced_3d_energy_and_gradient(
            plus,
            grid,
            boundary,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            mu0=1.0,
        )
        e_minus, _ = reduced_3d_energy_and_gradient(
            minus,
            grid,
            boundary,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            mu0=1.0,
        )
        finite_difference = (e_plus - e_minus) / (2.0 * step)
        assert np.isfinite(value)
        assert gradient[index] == pytest.approx(finite_difference, rel=3.0e-2, abs=3.0e-5)


def test_nonaxisymmetric_fixed_boundary_solver_preserves_boundary_and_positive_jacobian():
    pytest.importorskip("scipy.optimize")
    config, grid, boundary, initial_state = _case()
    psi = PsiPrimeProfile.constant(0.01)
    current = IPrimeProfile.zero()
    pressure = PressureProfile.zero()
    initial_residual = projected_energy_residual_3d(
        initial_state,
        grid,
        psi_prime=psi,
        i_prime=current,
        pressure=pressure,
        mu0=1.0,
    )

    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=psi,
        i_prime=current,
        pressure=pressure,
        initial_state=initial_state,
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=6, tolerance=1.0e-10, mu0=1.0),
    )
    geometry = evaluate_geometry_3d(result.state, result.grid)
    field = evaluate_field_3d(result.state, result.grid, geometry, psi_prime=psi, i_prime=current)
    center_index = int(np.argmin(np.abs(result.grid.xi)))

    assert isinstance(result.state, MirrorState3D)
    assert result.final_trace.energy_total <= initial_residual.energy
    assert result.final_trace.residual_norm <= initial_residual.norm
    assert result.final_trace.min_sqrtg > 0.0
    assert np.min(geometry.sqrtg) > 0.0
    assert np.allclose(result.state.a[-1], boundary.radius_on_grid_3d(result.grid))
    assert np.allclose(result.state.a[:, :, 0], boundary.radius_on_grid_3d(result.grid)[:, 0][None, :])
    assert np.allclose(result.state.a[:, :, -1], boundary.radius_on_grid_3d(result.grid)[:, -1][None, :])
    assert np.mean(field.bmag[-1, :, 0]) > np.mean(field.bmag[-1, :, center_index])
    assert np.mean(field.bmag[-1, :, -1]) > np.mean(field.bmag[-1, :, center_index])

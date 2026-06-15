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
from vmec_jax.mirror.core.state import MirrorStateAxisym
from vmec_jax.mirror.kernels.forces import axisym_projected_energy_residual
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry

pytestmark = pytest.mark.mirror


def _perturbed_cylinder_case():
    config = MirrorConfig(MirrorResolution(ns=7, ntheta=1, nxi=13, mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    boundary = MirrorBoundary.constant_radius(0.3)
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    a = base.a * (1.0 + 0.02 * s * (1.0 - s) * (1.0 - xi**2))
    return config, grid, boundary, MirrorStateAxisym(a=a, lam=np.zeros_like(a))


def test_fixed_boundary_solver_decreases_energy_and_residual_for_perturbed_cylinder():
    config, grid, boundary, initial_state = _perturbed_cylinder_case()
    psi = PsiPrimeProfile.constant(0.01)
    current = IPrimeProfile.zero()
    pressure = PressureProfile.zero()
    initial_residual = axisym_projected_energy_residual(
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
        options=MirrorSolveOptions(maxiter=20, step_size=0.05, tolerance=1.0e-12, mu0=1.0),
    )

    assert result.final_trace.energy_total < initial_residual.energy
    assert result.final_trace.residual_norm < initial_residual.norm
    assert result.final_trace.min_sqrtg > 0.0
    assert np.allclose(result.state.a[-1], boundary.radius_on_grid(result.grid))
    assert np.allclose(result.state.a[:, 0], boundary.radius_on_grid(result.grid)[0])
    assert np.allclose(result.state.a[:, -1], boundary.radius_on_grid(result.grid)[-1])
    assert all(row.accepted for row in result.trace)


def test_fixed_boundary_solver_records_pressure_continuation_trace():
    config = MirrorConfig(MirrorResolution(ns=7, ntheta=1, nxi=13, mpol=0), z_min=-1.0, z_max=1.0)
    boundary = MirrorBoundary.polynomial_radius(r0=0.3, a2=0.1)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.polynomial([1.0, -1.0], gamma=2.0),
        options=MirrorSolveOptions(
            maxiter=4,
            step_size=1.0e-4,
            tolerance=1.0e-12,
            mu0=1.0,
            pressure_continuation=(0.0, 0.5, 1.0),
        ),
    )

    stages = sorted({row.stage_index for row in result.trace})
    scales = sorted({row.pressure_scale for row in result.trace})
    assert stages == [0, 1, 2]
    assert scales == [0.0, 0.5, 1.0]
    assert result.final_trace.min_sqrtg > 0.0
    assert result.final_trace.energy_total > result.trace[0].energy_total
    assert len(result.trace) == 3 * (4 + 1)


def test_fixed_boundary_solver_preserves_flared_boundary_and_positive_jacobian():
    config = MirrorConfig(MirrorResolution(ns=9, ntheta=1, nxi=17, mpol=0), z_min=-1.2, z_max=1.2)
    boundary = MirrorBoundary.polynomial_radius(r0=0.27, a2=0.16, a4=0.02)
    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.012),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(maxiter=8, step_size=0.02, tolerance=1.0e-12, mu0=1.0),
    )
    geometry = evaluate_axisym_geometry(result.state, result.grid)

    assert np.allclose(result.state.a[-1], boundary.radius_on_grid(result.grid))
    assert np.min(geometry.sqrtg) > 0.0
    assert result.final_trace.residual_norm <= result.trace[0].residual_norm

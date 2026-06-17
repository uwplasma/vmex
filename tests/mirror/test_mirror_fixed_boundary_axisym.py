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
from vmec_jax.mirror.solvers.fixed_boundary.optimizers import (
    OptimizerOptions,
    _lbfgs_options,
    axisym_reduced_coordinate_scale,
    axisym_reduced_a_mask,
    axisym_reduced_bounds,
    axisym_reduced_residual_preconditioner,
    axisym_residual_linear_maxiter,
    pack_axisym_reduced_state,
    reduced_axisym_energy_and_gradient,
)

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


def test_reduced_lbfgs_gradient_matches_central_difference():
    pytest.importorskip("scipy.optimize")
    config, grid, boundary, initial_state = _perturbed_cylinder_case()
    lam = 0.01 * grid.s_full[:, None] * (grid.xi[None, :] - np.mean(grid.xi))
    state = MirrorStateAxisym(a=initial_state.a, lam=lam)
    psi = PsiPrimeProfile.constant(0.01)
    current = IPrimeProfile.zero()
    pressure = PressureProfile.zero()
    vector = pack_axisym_reduced_state(state, grid, boundary)
    bounds = axisym_reduced_bounds(grid)
    value, gradient = reduced_axisym_energy_and_gradient(
        vector,
        grid,
        boundary,
        psi_prime=psi,
        i_prime=current,
        pressure=pressure,
        mu0=1.0,
    )
    assert len(bounds) == vector.size
    assert all(bound[0] is not None for bound in bounds[: int(np.count_nonzero(axisym_reduced_a_mask(grid)))])

    num_a = int(np.count_nonzero(axisym_reduced_a_mask(grid)))
    indices = [0, num_a] if num_a < vector.size else [0]
    step = 1.0e-5
    for index in indices:
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        e_plus, _ = reduced_axisym_energy_and_gradient(
            plus,
            grid,
            boundary,
            psi_prime=psi,
            i_prime=current,
            pressure=pressure,
            mu0=1.0,
        )
        e_minus, _ = reduced_axisym_energy_and_gradient(
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
        assert gradient[index] == pytest.approx(finite_difference, rel=2.0e-2, abs=2.0e-5)


def test_reduced_coordinate_scaling_matches_reduced_axisym_layout():
    config, grid, boundary, initial_state = _perturbed_cylinder_case()
    del config
    vector = pack_axisym_reduced_state(initial_state, grid, boundary)
    scale = axisym_reduced_coordinate_scale(initial_state, grid, boundary)
    identity = axisym_reduced_coordinate_scale(initial_state, grid, boundary, mode="none")
    num_a = int(np.count_nonzero(axisym_reduced_a_mask(grid)))
    boundary_radius = boundary.radius_on_grid(grid)

    assert scale.shape == vector.shape
    assert np.all(np.isfinite(scale))
    assert np.all(scale > 0.0)
    assert np.allclose(identity, 1.0)
    assert np.allclose(
        scale[:num_a], np.broadcast_to(boundary_radius[None, :], initial_state.a.shape)[axisym_reduced_a_mask(grid)]
    )
    assert np.allclose(scale[num_a:], np.median(boundary_radius))


def test_reduced_residual_preconditioner_preserves_axisym_layout_and_damps_high_frequency_vector():
    config, grid, boundary, initial_state = _perturbed_cylinder_case()
    del config
    vector = pack_axisym_reduced_state(initial_state, grid, boundary)
    alternating = np.where(np.arange(vector.size) % 2 == 0, 1.0, -1.0)

    identity = axisym_reduced_residual_preconditioner(alternating, grid, kind="none")
    smoothed = axisym_reduced_residual_preconditioner(
        alternating,
        grid,
        kind="radial_xi_tridi",
        radial_alpha=0.5,
        lambda_alpha=0.5,
        xi_alpha=0.5,
    )
    lambda_xi_smoothed = axisym_reduced_residual_preconditioner(
        alternating,
        grid,
        kind="radial_xi_lambda_xi_tridi",
        radial_alpha=0.5,
        lambda_alpha=0.5,
        xi_alpha=0.5,
    )
    num_a = int(np.count_nonzero(axisym_reduced_a_mask(grid)))

    assert identity.shape == alternating.shape
    assert smoothed.shape == alternating.shape
    assert lambda_xi_smoothed.shape == alternating.shape
    assert np.allclose(identity, alternating)
    assert np.linalg.norm(smoothed) < np.linalg.norm(alternating)
    assert np.linalg.norm(lambda_xi_smoothed[num_a:]) < np.linalg.norm(smoothed[num_a:])
    with pytest.raises(ValueError, match="expected"):
        axisym_reduced_residual_preconditioner(alternating[:-1], grid)
    with pytest.raises(ValueError, match="nonnegative"):
        axisym_reduced_residual_preconditioner(alternating, grid, radial_alpha=-0.1)


def test_lbfgs_solver_uses_reduced_optimizer_and_improves_perturbed_cylinder():
    pytest.importorskip("scipy.optimize")
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
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=8, tolerance=1.0e-10, ftol=1.0e-12, mu0=1.0),
    )

    assert len(result.trace) > 1
    assert len(result.optimizer_summaries) == 1
    assert result.optimizer_summaries[0].nit <= 8
    assert result.optimizer_summaries[0].message
    assert isinstance(result.optimizer_summaries[0].accepted, bool)
    assert result.optimizer_summaries[0].rejection_reason in {"", "accepted"}
    assert result.optimizer_summaries[0].candidate_energy_total is not None
    assert result.optimizer_summaries[0].candidate_min_sqrtg is not None
    assert result.final_trace.energy_total <= initial_residual.energy
    assert result.final_trace.residual_norm <= initial_residual.norm
    assert result.final_trace.min_sqrtg > 0.0
    assert np.allclose(result.state.a[-1], boundary.radius_on_grid(result.grid))
    assert np.allclose(result.state.a[:, 0], boundary.radius_on_grid(result.grid)[0])
    assert np.allclose(result.state.a[:, -1], boundary.radius_on_grid(result.grid)[-1])


def test_residual_newton_solver_reaches_tight_residual_for_perturbed_cylinder():
    config = MirrorConfig(MirrorResolution(ns=5, ntheta=1, nxi=9, mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    boundary = MirrorBoundary.constant_radius(0.3)
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    initial_state = MirrorStateAxisym(a=base.a * (1.0 + 0.01 * s * (1.0 - s) * (1.0 - xi**2)), lam=base.lam)
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
        options=MirrorSolveOptions(
            optimizer="residual_newton",
            maxiter=20,
            tolerance=1.0e-12,
            ftol=1.0e-14,
            line_search_steps=32,
            residual_linear_maxiter=64,
            mu0=1.0,
        ),
    )
    summary = result.optimizer_summaries[0]

    assert summary.success
    assert summary.accepted
    assert summary.optimizer == "residual_newton"
    assert summary.nit <= 6
    assert summary.residual_linear_maxiter_policy == "adaptive"
    assert summary.residual_linear_maxiter_effective_max is not None
    assert (
        summary.residual_linear_maxiter_effective_max
        == pack_axisym_reduced_state(
            result.state,
            result.grid,
            boundary,
        ).size
    )
    assert result.final_trace.residual_norm < 1.0e-12
    assert result.final_trace.residual_norm < initial_residual.norm
    assert result.final_trace.energy_total < initial_residual.energy
    assert result.final_trace.min_sqrtg > 0.0


def test_residual_newton_dense_lstsq_solver_improves_perturbed_cylinder():
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
        options=MirrorSolveOptions(
            optimizer="residual_newton",
            maxiter=4,
            tolerance=1.0e-10,
            ftol=1.0e-14,
            line_search_steps=32,
            residual_linear_solver="dense_lstsq",
            residual_preconditioner="none",
            mu0=1.0,
        ),
    )
    summary = result.optimizer_summaries[0]

    assert summary.accepted
    assert summary.optimizer == "residual_newton"
    assert summary.residual_linear_solver == "dense_lstsq"
    assert summary.residual_linear_maxiter_effective_max is None
    assert result.final_trace.residual_norm < initial_residual.norm
    assert result.final_trace.energy_total < initial_residual.energy
    assert result.final_trace.min_sqrtg > 0.0


def test_residual_newton_linear_maxiter_policy_preserves_fixed_and_expands_adaptive():
    config = MirrorConfig(MirrorResolution(ns=9, ntheta=1, nxi=17, mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    size = int(np.count_nonzero(axisym_reduced_a_mask(grid)) + grid.ns * (grid.nxi - 1))
    fixed = OptimizerOptions(
        optimizer="residual_newton",
        residual_linear_maxiter=16,
        residual_linear_maxiter_policy="fixed",
        tolerance=1.0e-12,
    )
    adaptive = OptimizerOptions(
        optimizer="residual_newton",
        residual_linear_maxiter=16,
        residual_linear_maxiter_policy="adaptive",
        tolerance=1.0e-12,
    )

    fixed_budget = axisym_residual_linear_maxiter(fixed, grid, vector_size=size, residual_norm=1.0e-3)
    adaptive_budget = axisym_residual_linear_maxiter(adaptive, grid, vector_size=size, residual_norm=1.0e-3)

    assert fixed_budget == 16
    assert adaptive_budget > fixed_budget
    assert adaptive_budget <= size
    assert adaptive_budget >= 5 * grid.nxi


def test_lbfgs_options_use_explicit_ftol_when_requested():
    options = OptimizerOptions(optimizer="lbfgs", maxiter=2000, tolerance=1.0e-12, ftol=1.0e-12)
    assert _lbfgs_options(options)["ftol"] == pytest.approx(1.0e-12)

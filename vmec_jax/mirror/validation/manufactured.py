"""Named manufactured mirror validation cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vmec_jax._compat import jax, jnp

from ..api import MirrorConfig, MirrorResolution
from ..core.boundary import MirrorBoundary
from ..core.profiles import IPrimeProfile, PressureProfile, PsiPrimeProfile
from ..core.state import MirrorStateAxisym
from ..kernels.constraints import project_axisym_state
from ..kernels.energy import MU0
from ..kernels.forces import axisym_total_energy_jax
from ..kernels.manufactured import ManufacturedAxisymCase, build_axisym_manufactured_case
from ..solvers.fixed_boundary.optimizers import (
    axisym_reduced_a_mask,
    axisym_reduced_coordinate_scale,
    pack_axisym_reduced_state,
    reduced_axisym_energy_and_gradient,
    unpack_axisym_reduced_state,
)


@dataclass(frozen=True)
class ManufacturedSolveTraceRow:
    """One reduced-coordinate manufactured fixed-boundary solve row."""

    iteration: int
    objective: float
    residual_norm: float
    fsq: float
    step_norm: float
    exact_error_norm: float


@dataclass(frozen=True)
class ManufacturedSolveResult:
    """Validation solve result for an axisymmetric manufactured mirror case."""

    case: ManufacturedAxisymCase
    state: MirrorStateAxisym
    trace: tuple[ManufacturedSolveTraceRow, ...]
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_nit: int
    optimizer_nfev: int
    optimizer_njev: int
    residual_norm: float
    fsq: float
    exact_error_norm: float

    @property
    def final_trace(self) -> ManufacturedSolveTraceRow:
        return self.trace[-1]


def _grid_from_resolution(resolution: MirrorResolution, *, z_min: float, z_max: float):
    return MirrorConfig(resolution=resolution, z_min=z_min, z_max=z_max).build_grid()


def make_mms_case(
    name: str,
    resolution: MirrorResolution | None = None,
    *,
    mu0: float = MU0,
) -> ManufacturedAxisymCase:
    """Create a named axisymmetric manufactured mirror case."""
    resolution = resolution or MirrorResolution(ns=7, ntheta=1, nxi=13, mpol=0)
    grid = _grid_from_resolution(resolution, z_min=-1.2, z_max=1.2)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]

    if name == "axisym_flared_polynomial":
        a0 = 0.28
        epsilon = 0.16
        alpha = 0.08
        a = a0 * (1.0 + epsilon * xi**2) * (1.0 + alpha * s * (1.0 - s) * (1.0 - xi**2))
        lam = np.zeros_like(a)
        boundary = MirrorBoundary.polynomial_radius(r0=a0, a2=epsilon)
        pressure = PressureProfile.zero()
        psi_prime = PsiPrimeProfile.constant(0.02)
        i_prime = IPrimeProfile.zero()
    elif name == "axisym_lambda":
        a0 = 0.3
        epsilon = 0.1
        lam0 = 0.015
        a = a0 * (1.0 + epsilon * xi**2) * np.ones_like(s)
        lam = lam0 * s * (1.0 - s) * (1.0 - xi**2) * xi
        boundary = MirrorBoundary.polynomial_radius(r0=a0, a2=epsilon)
        pressure = PressureProfile.zero()
        psi_prime = PsiPrimeProfile.constant(0.015)
        i_prime = IPrimeProfile.constant(0.01)
    elif name == "axisym_finite_pressure":
        a0 = 0.26
        epsilon = 0.12
        alpha = 0.05
        a = a0 * (1.0 + epsilon * xi**2) * (1.0 + alpha * s * (1.0 - s) * (1.0 - xi**2))
        lam = np.zeros_like(a)
        boundary = MirrorBoundary.polynomial_radius(r0=a0, a2=epsilon)
        pressure = PressureProfile.polynomial([500.0, -1000.0, 500.0])
        psi_prime = PsiPrimeProfile.constant(0.018)
        i_prime = IPrimeProfile.zero()
    elif name == "axisym_projected_fixed_boundary":
        a0 = 0.29
        epsilon = 0.11
        alpha = 0.04
        lam0 = 0.006
        a = a0 * (1.0 + epsilon * xi**2) * (1.0 + alpha * s * (1.0 - s) * (1.0 - xi**2))
        lam = lam0 * s * (1.0 - s) * xi * (1.0 - xi**2)
        boundary = MirrorBoundary.polynomial_radius(r0=a0, a2=epsilon)
        pressure = PressureProfile.zero()
        psi_prime = PsiPrimeProfile.constant(0.017)
        i_prime = IPrimeProfile.constant(0.004)
        state = project_axisym_state(MirrorStateAxisym(a=a, lam=lam), grid, boundary)
        return build_axisym_manufactured_case(
            name=name,
            grid=grid,
            state=state,
            boundary=boundary,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=mu0,
        )
    else:
        raise ValueError(f"unknown manufactured mirror case {name!r}")

    state = MirrorStateAxisym(a=a, lam=lam)
    return build_axisym_manufactured_case(
        name=name,
        grid=grid,
        state=state,
        boundary=boundary,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )


def _reduced_mms_objective_and_gradient(
    vector, case: ManufacturedAxisymCase, *, mu0: float
) -> tuple[float, np.ndarray]:
    return reduced_axisym_energy_and_gradient(
        vector,
        case.grid,
        case.boundary,
        psi_prime=case.psi_prime,
        i_prime=case.i_prime,
        pressure=case.pressure,
        source_a=case.source_a,
        source_lam=case.source_lam,
        mu0=mu0,
    )


def _mms_trace_row(
    *,
    iteration: int,
    vector: np.ndarray,
    previous_vector: np.ndarray,
    exact_vector: np.ndarray,
    case: ManufacturedAxisymCase,
    mu0: float,
) -> ManufacturedSolveTraceRow:
    value, gradient = _reduced_mms_objective_and_gradient(vector, case, mu0=mu0)
    residual_norm = float(np.linalg.norm(gradient))
    active_dof = max(1, int(gradient.size))
    return ManufacturedSolveTraceRow(
        iteration=int(iteration),
        objective=float(value),
        residual_norm=residual_norm,
        fsq=float(residual_norm**2 / active_dof),
        step_norm=float(np.linalg.norm(np.asarray(vector, dtype=float) - np.asarray(previous_vector, dtype=float))),
        exact_error_norm=float(np.linalg.norm(np.asarray(vector, dtype=float) - np.asarray(exact_vector, dtype=float))),
    )


def _unpack_reduced_axisym_state_jax(vector, case: ManufacturedAxisymCase):
    grid = case.grid
    boundary_radius = jnp.asarray(case.boundary.radius_on_grid(grid), dtype=jnp.asarray(vector).dtype)
    mask_i, mask_j = np.nonzero(axisym_reduced_a_mask(grid))
    num_a = int(mask_i.size)

    a = jnp.broadcast_to(boundary_radius[None, :], (grid.ns, grid.nxi))
    a = a.at[(mask_i, mask_j)].set(vector[:num_a])
    a = a.at[0, :].set(a[1, :])
    a = a.at[0, 0].set(boundary_radius[0])
    a = a.at[0, -1].set(boundary_radius[-1])

    lam_inner = jnp.reshape(vector[num_a:], (grid.ns, grid.nxi - 1))
    w_xi = jnp.asarray(grid.w_xi, dtype=jnp.asarray(vector).dtype)
    lam_last = -jnp.einsum("j,ij->i", w_xi[:-1], lam_inner) / w_xi[-1]
    lam = jnp.concatenate([lam_inner, lam_last[:, None]], axis=1)
    return a, lam


def _mms_objective_jax(vector, case: ManufacturedAxisymCase, *, mu0: float = MU0):
    a, lam = _unpack_reduced_axisym_state_jax(vector, case)
    energy = axisym_total_energy_jax(
        a,
        lam,
        case.grid,
        psi_prime=case.psi_prime,
        i_prime=case.i_prime,
        pressure=case.pressure,
        mu0=mu0,
    )
    source_a = jnp.asarray(case.source_a, dtype=a.dtype)
    source_lam = jnp.asarray(case.source_lam, dtype=lam.dtype)
    return energy - jnp.sum(source_a * a) - jnp.sum(source_lam * lam)


def solve_axisym_mms_fixed_boundary(
    case: ManufacturedAxisymCase,
    *,
    initial_state: MirrorStateAxisym | None = None,
    maxiter: int = 300,
    gtol: float = 1.0e-12,
    ftol: float = 1.0e-12,
    line_search_steps: int = 64,
    reduced_coordinate_scaling: str = "geometry",
    mu0: float = MU0,
) -> ManufacturedSolveResult:
    """Solve a sourced axisymmetric MMS fixed-boundary problem in reduced coordinates."""
    if jax is None:
        raise RuntimeError("JAX is required for solve_axisym_mms_fixed_boundary")
    state0 = project_axisym_state(initial_state or case.state, case.grid, case.boundary)
    x0 = pack_axisym_reduced_state(state0, case.grid, case.boundary)
    exact_x = pack_axisym_reduced_state(case.state, case.grid, case.boundary)
    scale = axisym_reduced_coordinate_scale(state0, case.grid, case.boundary, mode=reduced_coordinate_scaling)
    y0 = x0 / scale
    previous_x = x0.copy()
    trace: list[ManufacturedSolveTraceRow] = [
        _mms_trace_row(iteration=0, vector=x0, previous_vector=x0, exact_vector=exact_x, case=case, mu0=mu0)
    ]
    scale_jax = jnp.asarray(scale, dtype=jnp.asarray(x0).dtype)

    def objective_x(vector):
        return _mms_objective_jax(vector, case, mu0=mu0)

    value_and_grad_x = jax.value_and_grad(objective_x)
    hessian_x = jax.jacfwd(jax.grad(objective_x))

    def x_from_y(vector_y) -> np.ndarray:
        return np.asarray(vector_y, dtype=float) * scale

    def residual_norm_x(vector: np.ndarray) -> float:
        _value, gradient = value_and_grad_x(jnp.asarray(vector, dtype=scale_jax.dtype))
        return float(np.linalg.norm(np.asarray(gradient, dtype=float)))

    def record_vector(vector: np.ndarray) -> None:
        nonlocal previous_x
        if float(np.linalg.norm(vector - previous_x)) <= 1.0e-15:
            return
        trace.append(
            _mms_trace_row(
                iteration=len(trace),
                vector=vector,
                previous_vector=previous_x,
                exact_vector=exact_x,
                case=case,
                mu0=mu0,
            )
        )
        previous_x = vector.copy()

    y = np.asarray(y0, dtype=float).copy()
    success = False
    status = 0
    message = "maximum iterations reached"
    nfev = 0
    njev = 0

    for _iteration in range(1, int(maxiter) + 1):
        x = x_from_y(y)
        _value, grad_x = value_and_grad_x(jnp.asarray(x, dtype=scale_jax.dtype))
        nfev += 1
        grad_x = np.asarray(grad_x, dtype=float)
        residual_norm = float(np.linalg.norm(grad_x))
        if residual_norm <= float(gtol):
            success = True
            status = 1
            message = "`gtol` termination condition is satisfied."
            break

        hessian = np.asarray(hessian_x(jnp.asarray(x, dtype=scale_jax.dtype)), dtype=float)
        njev += 1
        jacobian_y = hessian * scale[:, None] * scale[None, :]
        residual_y = grad_x * scale
        try:
            step_y, *_ = np.linalg.lstsq(jacobian_y, -residual_y, rcond=1.0e-12)
        except np.linalg.LinAlgError:
            message = "linear reduced-Hessian solve failed"
            break
        if not np.all(np.isfinite(step_y)):
            message = "non-finite reduced-Hessian step"
            break

        step_norm = float(np.linalg.norm(step_y * scale))
        if step_norm <= float(ftol) * max(1.0, float(np.linalg.norm(x))):
            success = True
            status = 3
            message = "`xtol` termination condition is satisfied."
            break

        accepted = False
        alpha = 1.0
        for _ in range(int(line_search_steps)):
            y_trial = y + alpha * step_y
            x_trial = x_from_y(y_trial)
            state_trial = unpack_axisym_reduced_state(x_trial, case.grid, case.boundary)
            if np.all(np.asarray(state_trial.a) > 1.0e-10):
                trial_norm = residual_norm_x(x_trial)
                nfev += 1
                if np.isfinite(trial_norm) and trial_norm < residual_norm:
                    y = y_trial
                    record_vector(x_trial)
                    accepted = True
                    if trial_norm <= float(gtol):
                        success = True
                        status = 1
                        message = "`gtol` termination condition is satisfied."
                    break
            alpha *= 0.5
        if success:
            break
        if not accepted:
            message = "line search failed to reduce the manufactured residual"
            break

    final_x = x_from_y(y)
    record_vector(final_x)
    final_state = unpack_axisym_reduced_state(final_x, case.grid, case.boundary)
    final = trace[-1]
    return ManufacturedSolveResult(
        case=case,
        state=final_state,
        trace=tuple(trace),
        optimizer_success=bool(success),
        optimizer_status=int(status),
        optimizer_message=str(message),
        optimizer_nit=max(0, len(trace) - 1),
        optimizer_nfev=int(nfev),
        optimizer_njev=int(njev),
        residual_norm=float(final.residual_norm),
        fsq=float(final.fsq),
        exact_error_norm=float(final.exact_error_norm),
    )

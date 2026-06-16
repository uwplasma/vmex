"""Differentiable variational wrappers for mirror energy residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vmec_jax._compat import jax, jnp

from .energy import MU0


@dataclass(frozen=True)
class AxisymEnergyGradient:
    """Energy value and gradients with respect to ``a`` and ``lambda``."""

    energy: float
    grad_a: np.ndarray
    grad_lam: np.ndarray


@dataclass(frozen=True)
class AxisymProjectedResidual:
    """Unprojected and constraint-projected energy residuals."""

    energy: float
    grad_a: np.ndarray
    grad_lam: np.ndarray
    projected_a: np.ndarray
    projected_lam: np.ndarray
    norm: float
    fsq: float
    normalized_force: float
    active_dof: int


@dataclass(frozen=True)
class EnergyGradient3D:
    """Energy value and gradients with respect to 3D ``a`` and ``lambda``."""

    energy: float
    grad_a: np.ndarray
    grad_lam: np.ndarray


@dataclass(frozen=True)
class ProjectedResidual3D:
    """Unprojected and constraint-projected 3D energy residuals."""

    energy: float
    grad_a: np.ndarray
    grad_lam: np.ndarray
    projected_a: np.ndarray
    projected_lam: np.ndarray
    norm: float
    fsq: float
    normalized_force: float
    active_dof: int


def _require_jax() -> None:
    if jax is None:
        raise RuntimeError("JAX is required for differentiable mirror force wrappers")


def radial_derivative_matrix(s_full) -> np.ndarray:
    """Return the finite-difference matrix matching the current radial grid policy."""
    s_full = np.asarray(s_full, dtype=float)
    if s_full.ndim != 1 or s_full.size < 2:
        raise ValueError("s_full must be a one-dimensional grid with at least two points")
    if not np.allclose(np.diff(s_full), np.diff(s_full)[0]):
        raise ValueError("only uniform radial grids are supported by the first mirror AD wrapper")
    num_nodes = s_full.size
    spacing = s_full[1] - s_full[0]
    derivative = np.zeros((num_nodes, num_nodes), dtype=float)
    if num_nodes == 2:
        derivative[0, 0] = -1.0 / spacing
        derivative[0, 1] = 1.0 / spacing
        derivative[1, 0] = -1.0 / spacing
        derivative[1, 1] = 1.0 / spacing
        return derivative
    derivative[0, 0] = -1.5 / spacing
    derivative[0, 1] = 2.0 / spacing
    derivative[0, 2] = -0.5 / spacing
    derivative[-1, -1] = 1.5 / spacing
    derivative[-1, -2] = -2.0 / spacing
    derivative[-1, -3] = 0.5 / spacing
    for idx in range(1, num_nodes - 1):
        derivative[idx, idx - 1] = -0.5 / spacing
        derivative[idx, idx + 1] = 0.5 / spacing
    return derivative


def theta_derivative_matrix(theta_basis) -> np.ndarray:
    """Return the dense Fourier differentiation matrix for the theta grid."""
    ntheta = int(theta_basis.ntheta)
    if ntheta == 1:
        return np.zeros((1, 1), dtype=float)
    columns = [theta_basis.differentiate(np.eye(ntheta, dtype=float)[idx], axis=0) for idx in range(ntheta)]
    return np.asarray(columns, dtype=float).T


def _polyval_jnp(coefficients, s):
    coefficients = jnp.asarray(coefficients, dtype=s.dtype)
    powers = s[..., None] ** jnp.arange(coefficients.size, dtype=s.dtype)
    return powers @ coefficients


def axisym_total_energy_jax(
    a,
    lam,
    grid,
    *,
    psi_prime,
    i_prime,
    pressure,
    mu0: float = MU0,
):
    """Return differentiable axisymmetric mirror energy for fixed static grid/profile data."""
    a = jnp.asarray(a)
    lam = jnp.asarray(lam, dtype=a.dtype)
    rho = jnp.asarray(grid.rho_full, dtype=a.dtype)[:, None]
    s_full = jnp.asarray(grid.s_full, dtype=a.dtype)
    w_s = jnp.asarray(grid.w_s, dtype=a.dtype)
    w_xi = jnp.asarray(grid.w_xi, dtype=a.dtype)
    theta_weight = jnp.asarray(jnp.sum(jnp.asarray(grid.w_theta, dtype=a.dtype)), dtype=a.dtype)
    d_xi = jnp.asarray(grid.axial_basis.derivative_matrix, dtype=a.dtype)
    d_s = jnp.asarray(radial_derivative_matrix(grid.s_full), dtype=a.dtype)
    z_xi = jnp.asarray(grid.z_xi, dtype=a.dtype)

    r = rho * a
    r_xi = rho * (a @ d_xi.T)
    r_r_s = 0.5 * (d_s @ (r**2))
    sqrtg = r_r_s * z_xi
    g_thetatheta = r**2
    g_xixi = r_xi**2 + z_xi**2

    psi = _polyval_jnp(psi_prime.coefficients, s_full)[:, None]
    current = _polyval_jnp(i_prime.coefficients, s_full)[:, None]
    pressure_values = _polyval_jnp(pressure.coefficients, s_full)[:, None]
    lam_xi = lam @ d_xi.T

    b_sup_theta = (current - lam_xi) / sqrtg
    b_sup_xi = psi / sqrtg
    b2 = g_thetatheta * b_sup_theta**2 + g_xixi * b_sup_xi**2
    weights = w_s[:, None] * w_xi[None, :] * theta_weight
    magnetic = jnp.sum(weights * sqrtg * b2 / (2.0 * mu0))
    pressure_energy = jnp.sum(weights * sqrtg * pressure_values / (pressure.gamma - 1.0))
    return magnetic + pressure_energy


def total_energy_3d_jax(
    a,
    lam,
    grid,
    *,
    psi_prime,
    i_prime,
    pressure,
    mu0: float = MU0,
):
    """Return differentiable theta-dependent mirror energy for fixed static grid/profile data."""
    a = jnp.asarray(a)
    lam = jnp.asarray(lam, dtype=a.dtype)
    rho = jnp.asarray(grid.rho_full, dtype=a.dtype)[:, None, None]
    s_full = jnp.asarray(grid.s_full, dtype=a.dtype)
    w_s = jnp.asarray(grid.w_s, dtype=a.dtype)
    w_theta = jnp.asarray(grid.w_theta, dtype=a.dtype)
    w_xi = jnp.asarray(grid.w_xi, dtype=a.dtype)
    d_theta = jnp.asarray(theta_derivative_matrix(grid.theta_basis), dtype=a.dtype)
    d_xi = jnp.asarray(grid.axial_basis.derivative_matrix, dtype=a.dtype)
    d_s = jnp.asarray(radial_derivative_matrix(grid.s_full), dtype=a.dtype)
    z_xi = jnp.asarray(grid.z_xi, dtype=a.dtype)

    r = rho * a
    a_theta = jnp.einsum("ij,sjk->sik", d_theta, a)
    a_xi = jnp.einsum("ij,stj->sti", d_xi, a)
    r_theta = rho * a_theta
    r_xi = rho * a_xi
    r_r_s = 0.5 * jnp.einsum("ij,jtk->itk", d_s, r**2)
    sqrtg = r_r_s * z_xi

    g_thetatheta = r_theta**2 + r**2
    g_thetaxi = r_theta * r_xi
    g_xixi = r_xi**2 + z_xi**2

    psi = _polyval_jnp(psi_prime.coefficients, s_full)[:, None, None]
    current = _polyval_jnp(i_prime.coefficients, s_full)[:, None, None]
    pressure_values = _polyval_jnp(pressure.coefficients, s_full)[:, None, None]
    lam_theta = jnp.einsum("ij,sjk->sik", d_theta, lam)
    lam_xi = jnp.einsum("ij,stj->sti", d_xi, lam)

    b_sup_theta = (current - lam_xi) / sqrtg
    b_sup_xi = (psi + lam_theta) / sqrtg
    b2 = g_thetatheta * b_sup_theta**2 + 2.0 * g_thetaxi * b_sup_theta * b_sup_xi + g_xixi * b_sup_xi**2
    weights = w_s[:, None, None] * w_theta[None, :, None] * w_xi[None, None, :]
    magnetic = jnp.sum(weights * sqrtg * b2 / (2.0 * mu0))
    pressure_energy = jnp.sum(weights * sqrtg * pressure_values / (pressure.gamma - 1.0))
    return magnetic + pressure_energy


def axisym_flat_state_energy_jax(flat_state, shape, grid, *, psi_prime, i_prime, pressure, mu0: float = MU0):
    """Return differentiable energy for a packed ``[a, lambda]`` vector."""
    flat_state = jnp.asarray(flat_state)
    size = int(np.prod(shape))
    a = jnp.reshape(flat_state[:size], shape)
    lam = jnp.reshape(flat_state[size:], shape)
    return axisym_total_energy_jax(
        a,
        lam,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )


def flat_state_energy_3d_jax(flat_state, shape, grid, *, psi_prime, i_prime, pressure, mu0: float = MU0):
    """Return differentiable 3D energy for a packed ``[a, lambda]`` vector."""
    flat_state = jnp.asarray(flat_state)
    size = int(np.prod(shape))
    a = jnp.reshape(flat_state[:size], shape)
    lam = jnp.reshape(flat_state[size:], shape)
    return total_energy_3d_jax(
        a,
        lam,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )


def axisym_energy_value_and_gradient(
    state, grid, *, psi_prime, i_prime, pressure, mu0: float = MU0
) -> AxisymEnergyGradient:
    """Return energy and AD gradients with respect to ``a`` and ``lambda``."""
    _require_jax()

    def objective(a, lam):
        return axisym_total_energy_jax(
            a,
            lam,
            grid,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=mu0,
        )

    value, (grad_a, grad_lam) = jax.value_and_grad(objective, argnums=(0, 1))(state.a, state.lam)
    return AxisymEnergyGradient(
        energy=float(value),
        grad_a=np.asarray(grad_a),
        grad_lam=np.asarray(grad_lam),
    )


def energy_value_and_gradient_3d(state, grid, *, psi_prime, i_prime, pressure, mu0: float = MU0) -> EnergyGradient3D:
    """Return energy and AD gradients for theta-dependent mirror states."""
    _require_jax()

    def objective(a, lam):
        return total_energy_3d_jax(
            a,
            lam,
            grid,
            psi_prime=psi_prime,
            i_prime=i_prime,
            pressure=pressure,
            mu0=mu0,
        )

    value, (grad_a, grad_lam) = jax.value_and_grad(objective, argnums=(0, 1))(state.a, state.lam)
    return EnergyGradient3D(
        energy=float(value),
        grad_a=np.asarray(grad_a),
        grad_lam=np.asarray(grad_lam),
    )


def project_axisym_residual(grad_a, grad_lam, grid, *, fix_end_surfaces: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Project gradients onto admissible fixed-boundary/gauge variations."""
    projected_a = np.asarray(grad_a).copy()
    projected_lam = np.asarray(grad_lam).copy()
    projected_a[-1, :] = 0.0
    projected_a[0, :] = 0.0
    if fix_end_surfaces:
        projected_a[:, 0] = 0.0
        projected_a[:, -1] = 0.0
    average = np.tensordot(projected_lam, grid.w_xi, axes=([-1], [0])) / np.sum(grid.w_xi)
    projected_lam = projected_lam - average[:, None]
    return projected_a, projected_lam


def project_residual_3d(grad_a, grad_lam, grid, *, fix_end_surfaces: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D gradients onto admissible fixed-boundary/gauge variations."""
    projected_a = np.asarray(grad_a).copy()
    projected_lam = np.asarray(grad_lam).copy()
    projected_a[-1, :, :] = 0.0
    projected_a[0, :, :] = 0.0
    if fix_end_surfaces:
        projected_a[:, :, 0] = 0.0
        projected_a[:, :, -1] = 0.0
    average = np.sum(
        projected_lam * grid.w_theta[None, :, None] * grid.w_xi[None, None, :],
        axis=(1, 2),
    ) / (np.sum(grid.w_theta) * np.sum(grid.w_xi))
    projected_lam = projected_lam - average[:, None, None]
    return projected_a, projected_lam


def active_axisym_force_dof_count(grid, *, fix_end_surfaces: bool = True) -> int:
    """Return the number of active projected force degrees of freedom."""
    active_a = np.ones((grid.ns, grid.nxi), dtype=bool)
    active_a[-1, :] = False
    active_a[0, :] = False
    if fix_end_surfaces:
        active_a[:, 0] = False
        active_a[:, -1] = False
    # Lambda is gauge-projected by removing one weighted mean per radial surface.
    active_lam = grid.ns * max(grid.nxi - 1, 0)
    return int(np.count_nonzero(active_a) + active_lam)


def active_3d_force_dof_count(grid, *, fix_end_surfaces: bool = True) -> int:
    """Return the number of active projected 3D force degrees of freedom."""
    active_a = np.ones((grid.ns, grid.ntheta, grid.nxi), dtype=bool)
    active_a[-1, :, :] = False
    active_a[0, :, :] = False
    if fix_end_surfaces:
        active_a[:, :, 0] = False
        active_a[:, :, -1] = False
    # Lambda is gauge-projected by removing one theta-xi mean per radial surface.
    active_lam = grid.ns * max(grid.ntheta * grid.nxi - 1, 0)
    return int(np.count_nonzero(active_a) + active_lam)


def normalized_force_metrics(norm: float, energy: float, active_dof: int) -> tuple[float, float]:
    """Return mirror-native ``fsq`` and normalized projected-force norm."""
    active_dof = max(1, int(active_dof))
    fsq = float(norm) ** 2 / active_dof
    energy_scale = max(abs(float(energy)), np.finfo(float).tiny)
    return fsq, float(np.sqrt(fsq) / energy_scale)


def axisym_projected_energy_residual(
    state,
    grid,
    *,
    psi_prime,
    i_prime,
    pressure,
    mu0: float = MU0,
) -> AxisymProjectedResidual:
    """Return unprojected and constraint-projected AD energy residuals."""
    gradient = axisym_energy_value_and_gradient(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    projected_a, projected_lam = project_axisym_residual(gradient.grad_a, gradient.grad_lam, grid)
    norm = float(np.sqrt(np.sum(projected_a**2) + np.sum(projected_lam**2)))
    active_dof = active_axisym_force_dof_count(grid)
    fsq, normalized_force = normalized_force_metrics(norm, gradient.energy, active_dof)
    return AxisymProjectedResidual(
        energy=gradient.energy,
        grad_a=gradient.grad_a,
        grad_lam=gradient.grad_lam,
        projected_a=projected_a,
        projected_lam=projected_lam,
        norm=norm,
        fsq=fsq,
        normalized_force=normalized_force,
        active_dof=active_dof,
    )


def projected_energy_residual_3d(
    state,
    grid,
    *,
    psi_prime,
    i_prime,
    pressure,
    mu0: float = MU0,
) -> ProjectedResidual3D:
    """Return unprojected and constraint-projected 3D AD energy residuals."""
    gradient = energy_value_and_gradient_3d(
        state,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    projected_a, projected_lam = project_residual_3d(gradient.grad_a, gradient.grad_lam, grid)
    norm = float(np.sqrt(np.sum(projected_a**2) + np.sum(projected_lam**2)))
    active_dof = active_3d_force_dof_count(grid)
    fsq, normalized_force = normalized_force_metrics(norm, gradient.energy, active_dof)
    return ProjectedResidual3D(
        energy=gradient.energy,
        grad_a=gradient.grad_a,
        grad_lam=gradient.grad_lam,
        projected_a=projected_a,
        projected_lam=projected_lam,
        norm=norm,
        fsq=fsq,
        normalized_force=normalized_force,
        active_dof=active_dof,
    )


def central_difference_energy_component(
    state,
    grid,
    *,
    psi_prime,
    i_prime,
    pressure,
    component: str,
    index: tuple[int, int],
    step: float = 1.0e-6,
    mu0: float = MU0,
) -> float:
    """Return a central finite-difference derivative for one state component."""
    if component not in {"a", "lam"}:
        raise ValueError("component must be 'a' or 'lam'")
    a_plus = np.asarray(state.a, dtype=float).copy()
    a_minus = np.asarray(state.a, dtype=float).copy()
    lam_plus = np.asarray(state.lam, dtype=float).copy()
    lam_minus = np.asarray(state.lam, dtype=float).copy()
    if component == "a":
        a_plus[index] += step
        a_minus[index] -= step
    else:
        lam_plus[index] += step
        lam_minus[index] -= step
    e_plus = axisym_total_energy_jax(
        a_plus,
        lam_plus,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    e_minus = axisym_total_energy_jax(
        a_minus,
        lam_minus,
        grid,
        psi_prime=psi_prime,
        i_prime=i_prime,
        pressure=pressure,
        mu0=mu0,
    )
    return float((e_plus - e_minus) / (2.0 * step))

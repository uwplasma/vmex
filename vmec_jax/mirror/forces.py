"""Isotropic mirror energy and independent tensor-force diagnostics.

The pressure model follows VMEC's mass-conserving variational form.  A radial
mass profile ``M(s)`` defines ``p(s) = M(s) / V'(s)^gamma``, where
``V'(s) = integral J dtheta dxi``.  Varying the total energy therefore gives
the pressure force instead of treating pressure as geometry-independent.

The direct residual computes ``curl(B)/mu0 x B - grad(p)`` independently of
the energy gradient.  Agreement between these two routes is an M2 acceptance
gate and protects the solver from converging only an optimizer surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import pi
from typing import Any

import jax
import jax.numpy as jnp

from .geometry import (
    ContravariantField,
    MirrorGeometry,
    contravariant_field,
    evaluate_geometry,
    magnetic_field_squared,
    radial_derivative,
)
from .model import MirrorBoundary, MirrorState, project_fixed_boundary_state

Array = Any
MU0 = 4.0e-7 * pi


def _profile(values: Array, ns: int, dtype: Any, *, name: str) -> Array:
    values = jnp.asarray(values, dtype=dtype)
    if values.ndim == 0:
        values = jnp.broadcast_to(values, (ns,))
    if values.shape != (ns,):
        raise ValueError(f"{name} shape {values.shape} must be scalar or ({ns},)")
    return values


def _surface_integral(values: Array, grid: "MirrorGrid") -> Array:
    return jnp.einsum(
        "j,k,ijk->i",
        jnp.asarray(grid.theta_basis.weights),
        jnp.asarray(grid.axial_basis.weights),
        values,
    )


def _half_mesh_magnetic_terms(
    state: MirrorState,
    grid: "MirrorGrid",
    axial_flux_derivative: Array,
    current_derivative: Array,
) -> tuple[Array, Array]:
    """Return ``(B^2, J)`` on VMEC-style radial half cells.

    The cell Jacobian uses the conservative difference
    ``r*r_s = 0.5*d(r^2)/ds``.  Pairing this with a cell sum makes a
    self-similar cylinder exactly stationary under interior radial mapping
    variations, unlike a full-mesh one-sided derivative/trapezoid pairing.
    """

    a = jnp.asarray(state.radius_scale)
    s = jnp.asarray(grid.s)[:, None, None]
    radius_squared = s * a**2
    ds = float(grid.s[1] - grid.s[0])
    radius_squared_half = 0.5 * (radius_squared[1:] + radius_squared[:-1])
    radius_half = jnp.sqrt(jnp.maximum(radius_squared_half, 0.0))
    r_r_s_half = 0.5 * (radius_squared[1:] - radius_squared[:-1]) / ds
    d_radius_dtheta = grid.theta_basis.differentiate(radius_half, axis=1)
    d_radius_dxi = grid.axial_basis.differentiate(radius_half, axis=2)
    jacobian = r_r_s_half * float(grid.dz_dxi)

    g_thetatheta = d_radius_dtheta**2 + radius_half**2
    g_thetaxi = d_radius_dtheta * d_radius_dxi
    g_xixi = d_radius_dxi**2 + float(grid.dz_dxi) ** 2

    lambda_half = 0.5 * (state.lambda_stream[1:] + state.lambda_stream[:-1])
    d_lambda_dtheta = grid.theta_basis.differentiate(lambda_half, axis=1)
    d_lambda_dxi = grid.axial_basis.differentiate(lambda_half, axis=2)
    psi = _profile(axial_flux_derivative, grid.ns, a.dtype, name="axial_flux_derivative")
    current = _profile(current_derivative, grid.ns, a.dtype, name="current_derivative")
    psi_half = 0.5 * (psi[1:] + psi[:-1])[:, None, None]
    current_half = 0.5 * (current[1:] + current[:-1])[:, None, None]
    b_theta = (current_half - d_lambda_dxi) / jacobian
    b_xi = (psi_half + d_lambda_dtheta) / jacobian
    b_squared = g_thetatheta * b_theta**2 + 2.0 * g_thetaxi * b_theta * b_xi + g_xixi * b_xi**2
    return b_squared, jacobian


@dataclass(frozen=True)
class MirrorEnergy:
    """Total, magnetic, and pressure energy with radial profiles."""

    total: Array
    magnetic: Array
    pressure_energy: Array
    volume_derivative: Array
    pressure: Array
    b_squared: Array
    geometry: MirrorGeometry
    field: ContravariantField


@dataclass(frozen=True)
class IsotropicForceResidual:
    """Covariant physical force components and convergence diagnostics."""

    covariant_s: Array
    covariant_theta: Array
    covariant_xi: Array
    current_sup_s: Array
    current_sup_theta: Array
    current_sup_xi: Array
    physical_rms: Array
    normalized_rms: Array
    component_rms: Array


@dataclass(frozen=True)
class VariationalResidual:
    """Nondimensional generalized forces used for nonlinear convergence."""

    radius_gradient: Array
    lambda_gradient: Array
    radius_rms: Array
    lambda_rms: Array
    maximum: Array


for _cls in (MirrorEnergy, IsotropicForceResidual, VariationalResidual):
    jax.tree_util.register_dataclass(
        _cls,
        data_fields=[field.name for field in fields(_cls)],
        meta_fields=[],
    )


def mass_profile_from_pressure(
    pressure: Array,
    volume_derivative: Array,
    *,
    gamma: float = 5.0 / 3.0,
) -> Array:
    """Convert a reference pressure into VMEC's conserved mass profile."""

    pressure = jnp.asarray(pressure)
    volume_derivative = jnp.asarray(volume_derivative, dtype=pressure.dtype)
    return pressure * volume_derivative**float(gamma)


def mirror_energy(
    state: MirrorState,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    mu0: float = float(MU0),
) -> MirrorEnergy:
    """Evaluate mass-conserving isotropic mirror energy."""

    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    geometry = evaluate_geometry(state, grid)
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )
    b_squared = magnetic_field_squared(field, geometry)
    volume_derivative = _surface_integral(geometry.sqrt_g, grid)
    mass = _profile(mass_profile, grid.ns, b_squared.dtype, name="mass_profile")
    pressure = mass / volume_derivative**float(gamma)

    b_squared_half, jacobian_half = _half_mesh_magnetic_terms(
        state,
        grid,
        axial_flux_derivative,
        current_derivative,
    )
    volume_derivative_half = _surface_integral(jacobian_half, grid)
    mass_half = 0.5 * (mass[1:] + mass[:-1])
    pressure_half = mass_half / volume_derivative_half**float(gamma)
    magnetic_surface = _surface_integral(b_squared_half * jacobian_half, grid) / (2.0 * float(mu0))
    pressure_surface = pressure_half * volume_derivative_half / (float(gamma) - 1.0)
    ds = float(grid.s[1] - grid.s[0])
    magnetic = ds * jnp.sum(magnetic_surface)
    pressure_energy = ds * jnp.sum(pressure_surface)
    return MirrorEnergy(
        total=magnetic + pressure_energy,
        magnetic=magnetic,
        pressure_energy=pressure_energy,
        volume_derivative=volume_derivative,
        pressure=pressure,
        b_squared=b_squared,
        geometry=geometry,
        field=field,
    )


def fixed_boundary_energy_gradient(
    state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    **energy_kwargs: Any,
) -> MirrorState:
    """Differentiate total energy through fixed-boundary projection."""

    def objective(trial: MirrorState) -> Array:
        projected = project_fixed_boundary_state(trial, boundary, grid)
        return mirror_energy(projected, grid, **energy_kwargs).total

    return jax.grad(objective)(state)


def fixed_boundary_variational_residual(
    state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    **energy_kwargs: Any,
) -> VariationalResidual:
    """Return normalized fixed-boundary energy-gradient force components.

    This is the mirror analogue of VMEC's variational ``fsqr/fsqz/fsql``
    convergence residual.  The independently differenced tensor force from
    :func:`isotropic_force_residual` remains a discretization-verification
    diagnostic and should converge under grid refinement.
    """

    projected = project_fixed_boundary_state(state, boundary, grid)
    energy = mirror_energy(projected, grid, **energy_kwargs)
    gradient = fixed_boundary_energy_gradient(projected, boundary, grid, **energy_kwargs)
    energy_scale = jnp.maximum(jnp.abs(energy.total), jnp.finfo(energy.total.dtype).tiny)
    radius_scale = jnp.mean(jnp.asarray(boundary.radius_scale))
    radius_gradient = gradient.radius_scale * radius_scale / energy_scale

    psi = _profile(
        energy_kwargs["axial_flux_derivative"],
        grid.ns,
        energy.total.dtype,
        name="axial_flux_derivative",
    )
    flux_scale = jnp.maximum(jnp.max(jnp.abs(psi)), jnp.finfo(psi.dtype).tiny)
    lambda_gradient = gradient.lambda_stream * flux_scale / energy_scale

    radius_free = radius_gradient[1:-1, :, 1:-1]
    radius_rms = jnp.sqrt(jnp.mean(radius_free**2))
    lambda_rms = jnp.sqrt(jnp.mean(lambda_gradient**2))
    maximum = jnp.maximum(jnp.max(jnp.abs(radius_free)), jnp.max(jnp.abs(lambda_gradient)))
    return VariationalResidual(
        radius_gradient=radius_gradient,
        lambda_gradient=lambda_gradient,
        radius_rms=radius_rms,
        lambda_rms=lambda_rms,
        maximum=maximum,
    )


def isotropic_force_residual(
    energy: MirrorEnergy,
    grid: "MirrorGrid",
    *,
    mu0: float = float(MU0),
) -> IsotropicForceResidual:
    """Compute ``curl(B)/mu0 x B - grad(p)`` in mirror coordinates."""

    geometry, field = energy.geometry, energy.field
    bs, bt, bx = field.b_sup_s, field.b_sup_theta, field.b_sup_xi
    b_cov_s = geometry.g_ss * bs + geometry.g_stheta * bt + geometry.g_sxi * bx
    b_cov_theta = geometry.g_stheta * bs + geometry.g_thetatheta * bt + geometry.g_thetaxi * bx
    b_cov_xi = geometry.g_sxi * bs + geometry.g_thetaxi * bt + geometry.g_xixi * bx

    d_theta_b_xi = grid.theta_basis.differentiate(b_cov_xi, axis=1)
    d_xi_b_theta = grid.axial_basis.differentiate(b_cov_theta, axis=2)
    d_xi_b_s = grid.axial_basis.differentiate(b_cov_s, axis=2)
    d_theta_b_s = grid.theta_basis.differentiate(b_cov_s, axis=1)
    ds = float(grid.s[1] - grid.s[0])
    d_s_b_xi = radial_derivative(b_cov_xi, ds)
    d_s_b_theta = radial_derivative(b_cov_theta, ds)
    inverse_mu0_jac = 1.0 / (float(mu0) * geometry.sqrt_g)
    current_s = (d_theta_b_xi - d_xi_b_theta) * inverse_mu0_jac
    current_theta = (d_xi_b_s - d_s_b_xi) * inverse_mu0_jac
    current_xi = (d_s_b_theta - d_theta_b_s) * inverse_mu0_jac

    pressure_gradient_s = radial_derivative(energy.pressure, ds)[:, None, None]
    force_s = geometry.sqrt_g * (current_theta * bx - current_xi * bt) - pressure_gradient_s
    force_theta = geometry.sqrt_g * (current_xi * bs - current_s * bx)
    force_xi = geometry.sqrt_g * (current_s * bt - current_theta * bs)

    metric = jnp.stack(
        [
            jnp.stack([geometry.g_ss, geometry.g_stheta, geometry.g_sxi], axis=-1),
            jnp.stack([geometry.g_stheta, geometry.g_thetatheta, geometry.g_thetaxi], axis=-1),
            jnp.stack([geometry.g_sxi, geometry.g_thetaxi, geometry.g_xixi], axis=-1),
        ],
        axis=-2,
    )
    force_covariant = jnp.stack([force_s, force_theta, force_xi], axis=-1)
    # The cylindrical coordinate metric is singular at s=0; the physical
    # volume of that row vanishes, so norms use the regular interior rows.
    inverse_metric = jnp.linalg.inv(metric[1:])
    force_squared = jnp.einsum(
        "...i,...ij,...j->...",
        force_covariant[1:],
        inverse_metric,
        force_covariant[1:],
    )
    weights = (
        jnp.asarray(grid.radial_weights[1:])[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, :]
        * geometry.sqrt_g[1:]
    )
    physical_rms = jnp.sqrt(jnp.sum(weights * force_squared) / jnp.sum(weights))
    component_rms = jnp.sqrt(
        jnp.sum(weights[..., None] * force_covariant[1:] ** 2, axis=(0, 1, 2)) / jnp.sum(weights)
    )
    length = float(grid.z[-1] - grid.z[0])
    magnetic_force_scale = energy.b_squared[1:] / (float(mu0) * length)
    reference_rms = jnp.sqrt(jnp.sum(weights * magnetic_force_scale**2) / jnp.sum(weights))
    normalized_rms = physical_rms / jnp.maximum(reference_rms, jnp.finfo(physical_rms.dtype).tiny)
    return IsotropicForceResidual(
        covariant_s=force_s,
        covariant_theta=force_theta,
        covariant_xi=force_xi,
        current_sup_s=current_s,
        current_sup_theta=current_theta,
        current_sup_xi=current_xi,
        physical_rms=physical_rms,
        normalized_rms=normalized_rms,
        component_rms=component_rms,
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid

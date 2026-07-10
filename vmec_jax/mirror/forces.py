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
from .model import (
    AnisotropyIndicators,
    MirrorBoundary,
    MirrorState,
    PressureClosure,
    PressureMoments,
    anisotropy_indicators,
    project_fixed_boundary_state,
)

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
    """Return radially integrated ``(B^2, J)`` on each half cell.

    Two-point Gauss integration prevents the alternating full-mesh lambda
    mode that one-point midpoint quadrature cannot see. The returned arrays
    retain shape ``(ns-1, ntheta, nxi)``: ``J`` is the cell-average Jacobian
    and ``B^2`` is weighted so ``B^2*J`` is the cell-average magnetic density.
    """

    a = jnp.asarray(state.radius_scale)
    ds = float(grid.s[1] - grid.s[0])
    gauss = 0.5 + jnp.asarray([-1.0, 1.0]) / (2.0 * jnp.sqrt(3.0))
    fraction = gauss[:, None, None, None]
    s_left = jnp.asarray(grid.s[:-1])[None, :, None, None]
    s_quadrature = s_left + fraction * ds
    a_left, a_right = a[:-1][None], a[1:][None]
    a_quadrature = (1.0 - fraction) * a_left + fraction * a_right
    da_ds = (a_right - a_left) / ds
    radius_squared = s_quadrature * a_quadrature**2
    radius = jnp.sqrt(jnp.maximum(radius_squared, 0.0))
    r_r_s = 0.5 * (a_quadrature**2 + 2.0 * s_quadrature * a_quadrature * da_ds)
    d_radius_dtheta = grid.theta_basis.differentiate(radius, axis=2)
    d_radius_dxi = grid.axial_basis.differentiate(radius, axis=3)
    jacobian = r_r_s * float(grid.dz_dxi)

    g_thetatheta = d_radius_dtheta**2 + radius**2
    g_thetaxi = d_radius_dtheta * d_radius_dxi
    g_xixi = d_radius_dxi**2 + float(grid.dz_dxi) ** 2

    lam = jnp.asarray(state.lambda_stream)
    lambda_quadrature = (1.0 - fraction) * lam[:-1][None] + fraction * lam[1:][None]
    d_lambda_dtheta = grid.theta_basis.differentiate(lambda_quadrature, axis=2)
    d_lambda_dxi = grid.axial_basis.differentiate(lambda_quadrature, axis=3)
    psi = _profile(axial_flux_derivative, grid.ns, a.dtype, name="axial_flux_derivative")
    current = _profile(current_derivative, grid.ns, a.dtype, name="current_derivative")
    psi_quadrature = (
        (1.0 - gauss[:, None]) * psi[:-1][None] + gauss[:, None] * psi[1:][None]
    )[:, :, None, None]
    current_quadrature = (
        (1.0 - gauss[:, None]) * current[:-1][None] + gauss[:, None] * current[1:][None]
    )[:, :, None, None]
    b_theta = (current_quadrature - d_lambda_dxi) / jacobian
    b_xi = (psi_quadrature + d_lambda_dtheta) / jacobian
    b_squared = g_thetatheta * b_theta**2 + 2.0 * g_thetaxi * b_theta * b_xi + g_xixi * b_xi**2
    jacobian_cell = jnp.mean(jacobian, axis=0)
    magnetic_density_cell = jnp.mean(b_squared * jacobian, axis=0)
    return magnetic_density_cell / jacobian_cell, jacobian_cell


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
    bulk_normalized_rms: Array
    axis_normalized_rms: Array
    component_rms: Array


@dataclass(frozen=True)
class VariationalResidual:
    """Nondimensional generalized forces used for nonlinear convergence."""

    radius_gradient: Array
    lambda_gradient: Array
    radius_rms: Array
    lambda_rms: Array
    maximum: Array


@dataclass(frozen=True)
class AnisotropicMirrorEnergy:
    """ANIMEC energy and pressure moments on radial half cells."""

    total: Array
    magnetic: Array
    pressure_energy: Array
    b_squared_half: Array
    jacobian_half: Array
    moments_half: PressureMoments
    indicators_half: AnisotropyIndicators
    geometry: MirrorGeometry
    field: ContravariantField


@dataclass(frozen=True)
class AnisotropicForceResidual:
    """Cartesian ``J x B - div(P)`` and parallel-balance diagnostics."""

    force_xyz: Array
    radius_variation_projection: Array
    divergence_pressure_xyz: Array
    current_xyz: Array
    physical_rms: Array
    normalized_rms: Array
    component_rms: Array
    parallel_pressure_rms: Array


@dataclass(frozen=True)
class InterfaceResidual:
    """Lateral plasma-vacuum tangency and normal-stress residuals."""

    plasma_b_normal_rms: Array
    vacuum_b_normal_rms: Array
    normal_stress_rms: Array
    normal_stress_jump: Array


for _cls in (
    MirrorEnergy,
    IsotropicForceResidual,
    VariationalResidual,
    AnisotropicMirrorEnergy,
    AnisotropicForceResidual,
    InterfaceResidual,
):
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


def anisotropic_mirror_energy(
    state: MirrorState,
    grid: "MirrorGrid",
    closure: PressureClosure,
    *,
    axial_flux_derivative: Array,
    current_derivative: Array = 0.0,
    mu0: float = MU0,
) -> AnisotropicMirrorEnergy:
    """Evaluate the ANIMEC functional with consistent pressure moments."""

    geometry = evaluate_geometry(state, grid)
    field = contravariant_field(
        state,
        geometry,
        grid,
        axial_flux_derivative=axial_flux_derivative,
        current_derivative=current_derivative,
    )
    b_squared_half, jacobian_half = _half_mesh_magnetic_terms(
        state,
        grid,
        axial_flux_derivative,
        current_derivative,
    )
    b_half = jnp.sqrt(jnp.maximum(b_squared_half, 0.0))
    s_half = jnp.asarray(grid.s_half)[:, None, None]
    moments = closure.moments(s_half, b_half)
    indicators = anisotropy_indicators(closure, s_half, b_half, mu0=mu0)
    magnetic_surface = _surface_integral(b_squared_half * jacobian_half, grid) / (2.0 * float(mu0))
    pressure_surface = _surface_integral(moments.energy_density * jacobian_half, grid)
    ds = float(grid.s[1] - grid.s[0])
    magnetic = ds * jnp.sum(magnetic_surface)
    pressure_energy = ds * jnp.sum(pressure_surface)
    return AnisotropicMirrorEnergy(
        total=magnetic + pressure_energy,
        magnetic=magnetic,
        pressure_energy=pressure_energy,
        b_squared_half=b_squared_half,
        jacobian_half=jacobian_half,
        moments_half=moments,
        indicators_half=indicators,
        geometry=geometry,
        field=field,
    )


def anisotropic_fixed_boundary_energy_gradient(
    state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    closure: PressureClosure,
    **energy_kwargs: Any,
) -> MirrorState:
    """Differentiate the ANIMEC energy through fixed-boundary projection."""

    def objective(trial: MirrorState) -> Array:
        projected = project_fixed_boundary_state(trial, boundary, grid)
        return anisotropic_mirror_energy(
            projected,
            grid,
            closure,
            **energy_kwargs,
        ).total

    return jax.grad(objective)(state)


def _coordinate_basis(geometry: MirrorGeometry, grid: "MirrorGrid") -> Array:
    """Return Cartesian covariant basis vectors as ``[..., xyz, q]``."""

    radius = geometry.radius
    r_s = jnp.where(
        jnp.abs(radius) > jnp.finfo(radius.dtype).eps,
        geometry.d_radius_ds_regular / jnp.where(radius != 0.0, radius, 1.0),
        0.0,
    )
    r_s = r_s.at[0].set(r_s[1])
    theta = jnp.asarray(grid.theta)[None, :, None]
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    zeros = jnp.zeros_like(radius)
    e_s = jnp.stack([r_s * cosine, r_s * sine, zeros], axis=-1)
    e_theta = jnp.stack(
        [
            geometry.d_radius_dtheta * cosine - radius * sine,
            geometry.d_radius_dtheta * sine + radius * cosine,
            zeros,
        ],
        axis=-1,
    )
    e_xi = jnp.stack(
        [
            geometry.d_radius_dxi * cosine,
            geometry.d_radius_dxi * sine,
            jnp.full_like(radius, float(grid.dz_dxi)),
        ],
        axis=-1,
    )
    return jnp.stack([e_s, e_theta, e_xi], axis=-1)


def _current_contravariant(
    geometry: MirrorGeometry,
    field: ContravariantField,
    grid: "MirrorGrid",
    mu0: float,
) -> Array:
    """Return contravariant ``curl(B)/mu0`` components."""

    bs, bt, bx = field.b_sup_s, field.b_sup_theta, field.b_sup_xi
    b_cov_s = geometry.g_ss * bs + geometry.g_stheta * bt + geometry.g_sxi * bx
    b_cov_theta = geometry.g_stheta * bs + geometry.g_thetatheta * bt + geometry.g_thetaxi * bx
    b_cov_xi = geometry.g_sxi * bs + geometry.g_thetaxi * bt + geometry.g_xixi * bx
    ds = float(grid.s[1] - grid.s[0])
    inverse_mu0_jac = 1.0 / (float(mu0) * geometry.sqrt_g)
    return jnp.stack(
        [
            (
                grid.theta_basis.differentiate(b_cov_xi, axis=1)
                - grid.axial_basis.differentiate(b_cov_theta, axis=2)
            )
            * inverse_mu0_jac,
            (
                grid.axial_basis.differentiate(b_cov_s, axis=2)
                - radial_derivative(b_cov_xi, ds)
            )
            * inverse_mu0_jac,
            (
                radial_derivative(b_cov_theta, ds)
                - grid.theta_basis.differentiate(b_cov_s, axis=1)
            )
            * inverse_mu0_jac,
        ],
        axis=-1,
    )


def anisotropic_force_residual(
    state: MirrorState,
    energy: AnisotropicMirrorEnergy,
    grid: "MirrorGrid",
    closure: PressureClosure,
    *,
    mu0: float = MU0,
) -> AnisotropicForceResidual:
    """Evaluate the continuum anisotropic tensor-force residual."""

    geometry, field = energy.geometry, energy.field
    basis = _coordinate_basis(geometry, grid)
    b_contravariant = jnp.stack(
        [field.b_sup_s, field.b_sup_theta, field.b_sup_xi], axis=-1
    )
    b_xyz = jnp.einsum("...ai,...i->...a", basis, b_contravariant)
    b_magnitude = jnp.linalg.norm(b_xyz, axis=-1)
    unit_b = b_xyz / b_magnitude[..., None]
    s = jnp.asarray(grid.s)[:, None, None]
    moments = closure.moments(s, b_magnitude)

    metric = jnp.stack(
        [
            jnp.stack([geometry.g_ss, geometry.g_stheta, geometry.g_sxi], axis=-1),
            jnp.stack([geometry.g_stheta, geometry.g_thetatheta, geometry.g_thetaxi], axis=-1),
            jnp.stack([geometry.g_sxi, geometry.g_thetaxi, geometry.g_xixi], axis=-1),
        ],
        axis=-2,
    )
    # Cylindrical coordinates are singular at s=0.  Copying the first regular
    # metric row supplies the radial stencil; all reported norms exclude the
    # zero-volume axis row.
    metric_regular = metric.at[0].set(metric[1])
    inverse_metric = jnp.linalg.inv(metric_regular)
    anisotropic_pressure_contravariant = (
        (moments.parallel - moments.perpendicular)[..., None, None]
        * b_contravariant[..., :, None]
        * b_contravariant[..., None, :]
        / b_magnitude[..., None, None] ** 2
    )

    ds = float(grid.s[1] - grid.s[0])
    metric_derivatives = jnp.stack(
        [
            radial_derivative(metric_regular, ds),
            grid.theta_basis.differentiate(metric_regular, axis=1),
            grid.axial_basis.differentiate(metric_regular, axis=2),
        ],
        axis=-3,
    )
    christoffel = jnp.zeros(metric.shape[:-2] + (3, 3, 3), dtype=metric.dtype)
    for j in range(3):
        for k in range(3):
            derivative_combination = (
                metric_derivatives[..., j, :, k]
                + metric_derivatives[..., k, :, j]
                - metric_derivatives[..., :, j, k]
            )
            gamma_jk = 0.5 * jnp.einsum(
                "...il,...l->...i", inverse_metric, derivative_combination
            )
            christoffel = christoffel.at[..., :, j, k].set(gamma_jk)

    pressure_flux = geometry.sqrt_g[..., None, None] * anisotropic_pressure_contravariant
    d_s_flux = radial_derivative(pressure_flux, ds)
    d_theta_flux = grid.theta_basis.differentiate(pressure_flux, axis=1)
    d_xi_flux = grid.axial_basis.differentiate(pressure_flux, axis=2)
    coordinate_divergence = (
        d_s_flux[..., :, 0]
        + d_theta_flux[..., :, 1]
        + d_xi_flux[..., :, 2]
    ) / geometry.sqrt_g[..., None]
    coordinate_divergence += jnp.einsum(
        "...ijk,...jk->...i", christoffel, anisotropic_pressure_contravariant
    )
    divergence_anisotropic = jnp.einsum(
        "...ai,...i->...a", basis[1:], coordinate_divergence[1:]
    )
    perpendicular_derivatives = jnp.stack(
        [
            radial_derivative(moments.perpendicular, ds),
            grid.theta_basis.differentiate(moments.perpendicular, axis=1),
            grid.axial_basis.differentiate(moments.perpendicular, axis=2),
        ],
        axis=-1,
    )
    perpendicular_gradient_contravariant = jnp.einsum(
        "...ij,...j->...i", inverse_metric[1:], perpendicular_derivatives[1:]
    )
    perpendicular_gradient_xyz = jnp.einsum(
        "...ai,...i->...a", basis[1:], perpendicular_gradient_contravariant
    )
    divergence_pressure = perpendicular_gradient_xyz + divergence_anisotropic
    current_contravariant = _current_contravariant(geometry, field, grid, mu0)
    current_xyz = jnp.einsum(
        "...ai,...i->...a", basis[1:], current_contravariant[1:]
    )
    lorentz = jnp.cross(current_xyz, b_xyz[1:])
    force_xyz = lorentz - divergence_pressure
    theta = jnp.asarray(grid.theta)[None, :, None]
    radial_unit = jnp.stack(
        [jnp.cos(theta), jnp.sin(theta), jnp.zeros_like(theta)], axis=-1
    )
    radial_force = jnp.sum(force_xyz * radial_unit, axis=-1)
    variation_weights = (
        jnp.asarray(grid.radial_weights[1:])[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, :]
        * geometry.sqrt_g[1:]
        * jnp.sqrt(jnp.asarray(grid.s[1:]))[:, None, None]
    )
    radius_projection = jnp.zeros_like(geometry.radius).at[1:].set(
        -variation_weights * radial_force
    )

    # The side boundary and axial cuts are prescribed data, not active
    # Euler-Lagrange equations.  Keep their pointwise forces in ``force_xyz``
    # but exclude them from the equilibrium norm; interface/cut diagnostics
    # report those constraints separately.
    force_active = force_xyz[:-1, :, 1:-1]
    weights = (
        jnp.asarray(grid.radial_weights[1:-1])[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, 1:-1]
        * geometry.sqrt_g[1:-1, :, 1:-1]
    )
    force_squared = jnp.sum(force_active**2, axis=-1)
    physical_rms = jnp.sqrt(jnp.sum(weights * force_squared) / jnp.sum(weights))
    component_rms = jnp.sqrt(
        jnp.sum(weights[..., None] * force_active**2, axis=(0, 1, 2)) / jnp.sum(weights)
    )
    length = float(grid.z[-1] - grid.z[0])
    pressure_scale = (
        jnp.abs(moments.parallel[1:-1, :, 1:-1])
        + 2.0 * jnp.abs(moments.perpendicular[1:-1, :, 1:-1])
    )
    reference = (
        energy.b_squared_half[:-1, :, 1:-1] / float(mu0) + pressure_scale
    ) / length
    reference_rms = jnp.sqrt(jnp.sum(weights * reference**2) / jnp.sum(weights))
    parallel_pressure = jnp.sum(
        divergence_pressure[:-1, :, 1:-1] * unit_b[1:-1, :, 1:-1], axis=-1
    )
    parallel_pressure_rms = jnp.sqrt(
        jnp.sum(weights * parallel_pressure**2) / jnp.sum(weights)
    )
    return AnisotropicForceResidual(
        force_xyz=force_xyz,
        radius_variation_projection=radius_projection,
        divergence_pressure_xyz=divergence_pressure,
        current_xyz=current_xyz,
        physical_rms=physical_rms,
        normalized_rms=physical_rms
        / jnp.maximum(reference_rms, jnp.finfo(physical_rms.dtype).tiny),
        component_rms=component_rms,
        parallel_pressure_rms=parallel_pressure_rms,
    )


def interface_residual(
    *,
    perpendicular_pressure: Array,
    plasma_b_squared: Array,
    vacuum_b_squared: Array,
    plasma_b_normal: Array,
    vacuum_b_normal: Array,
    theta_weights: Array,
    axial_weights: Array,
    mu0: float = MU0,
) -> InterfaceResidual:
    """Evaluate anisotropic free-boundary interface conditions.

    The lateral interface requires both fields to be tangent and
    ``p_perp + B_plasma^2/(2*mu0) = B_vacuum^2/(2*mu0)``.
    Inputs are sampled on ``(theta, xi)``.
    """

    p_perp, bp2, bv2, bnp, bnv = jnp.broadcast_arrays(
        perpendicular_pressure,
        plasma_b_squared,
        vacuum_b_squared,
        plasma_b_normal,
        vacuum_b_normal,
    )
    weights = jnp.asarray(theta_weights)[:, None] * jnp.asarray(axial_weights)[None, :]
    denominator = jnp.sum(weights)
    plasma_b_normal_rms = jnp.sqrt(
        jnp.sum(weights * bnp**2 / jnp.maximum(bp2, jnp.finfo(bp2.dtype).tiny))
        / denominator
    )
    vacuum_b_normal_rms = jnp.sqrt(
        jnp.sum(weights * bnv**2 / jnp.maximum(bv2, jnp.finfo(bv2.dtype).tiny))
        / denominator
    )
    jump = p_perp + bp2 / (2.0 * float(mu0)) - bv2 / (2.0 * float(mu0))
    stress_scale = (
        jnp.abs(p_perp) + bp2 / (2.0 * float(mu0)) + bv2 / (2.0 * float(mu0))
    )
    normal_stress_rms = jnp.sqrt(
        jnp.sum(weights * (jump / jnp.maximum(stress_scale, jnp.finfo(bp2.dtype).tiny)) ** 2)
        / denominator
    )
    return InterfaceResidual(
        plasma_b_normal_rms=plasma_b_normal_rms,
        vacuum_b_normal_rms=vacuum_b_normal_rms,
        normal_stress_rms=normal_stress_rms,
        normal_stress_jump=jump,
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
    return _normalized_variational_residual(
        gradient,
        energy.total,
        boundary,
        grid,
        axial_flux_derivative=energy_kwargs["axial_flux_derivative"],
    )


def anisotropic_fixed_boundary_variational_residual(
    state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    closure: PressureClosure,
    **energy_kwargs: Any,
) -> VariationalResidual:
    """Return normalized generalized forces for the ANIMEC functional."""

    projected = project_fixed_boundary_state(state, boundary, grid)
    energy = anisotropic_mirror_energy(projected, grid, closure, **energy_kwargs)
    gradient = anisotropic_fixed_boundary_energy_gradient(
        projected, boundary, grid, closure, **energy_kwargs
    )
    return _normalized_variational_residual(
        gradient,
        energy.total,
        boundary,
        grid,
        axial_flux_derivative=energy_kwargs["axial_flux_derivative"],
    )


def _normalized_variational_residual(
    gradient: MirrorState,
    energy_total: Array,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
) -> VariationalResidual:
    """Scale an energy gradient into component-wise mirror force norms."""

    energy_scale = jnp.maximum(
        jnp.abs(energy_total), jnp.finfo(jnp.asarray(energy_total).dtype).tiny
    )
    radius_scale = jnp.mean(jnp.asarray(boundary.radius_scale))
    radius_gradient = gradient.radius_scale * radius_scale / energy_scale

    psi = _profile(
        axial_flux_derivative,
        grid.ns,
        jnp.asarray(energy_total).dtype,
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
    # The axis, side boundary, and end cuts are constrained rather than active
    # force equations. Keep their pointwise values but norm the free interior.
    force_active = force_covariant[1:-1, :, 1:-1]
    inverse_metric = jnp.linalg.inv(metric[1:-1, :, 1:-1])
    force_squared = jnp.einsum(
        "...i,...ij,...j->...",
        force_active,
        inverse_metric,
        force_active,
    )
    weights = (
        jnp.asarray(grid.radial_weights[1:-1])[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, 1:-1]
        * geometry.sqrt_g[1:-1, :, 1:-1]
    )
    physical_rms = jnp.sqrt(jnp.sum(weights * force_squared) / jnp.sum(weights))
    component_rms = jnp.sqrt(
        jnp.sum(weights[..., None] * force_active**2, axis=(0, 1, 2)) / jnp.sum(weights)
    )
    length = float(grid.z[-1] - grid.z[0])
    magnetic_force_scale = energy.b_squared[1:-1, :, 1:-1] / (float(mu0) * length)
    reference_rms = jnp.sqrt(jnp.sum(weights * magnetic_force_scale**2) / jnp.sum(weights))
    normalized_rms = physical_rms / jnp.maximum(reference_rms, jnp.finfo(physical_rms.dtype).tiny)
    active_s = jnp.asarray(grid.s[1:-1])

    def regional_normalized_rms(mask: Array) -> Array:
        regional_weights = weights * jnp.asarray(mask)[:, None, None]
        denominator = jnp.maximum(
            jnp.sum(regional_weights), jnp.finfo(physical_rms.dtype).tiny
        )
        regional_force = jnp.sqrt(jnp.sum(regional_weights * force_squared) / denominator)
        regional_reference = jnp.sqrt(
            jnp.sum(regional_weights * magnetic_force_scale**2) / denominator
        )
        return regional_force / jnp.maximum(
            regional_reference, jnp.finfo(physical_rms.dtype).tiny
        )

    bulk_normalized_rms = regional_normalized_rms(active_s >= 0.2)
    axis_normalized_rms = regional_normalized_rms(active_s < 0.2)
    return IsotropicForceResidual(
        covariant_s=force_s,
        covariant_theta=force_theta,
        covariant_xi=force_xi,
        current_sup_s=current_s,
        current_sup_theta=current_theta,
        current_sup_xi=current_xi,
        physical_rms=physical_rms,
        normalized_rms=normalized_rms,
        bulk_normalized_rms=bulk_normalized_rms,
        axis_normalized_rms=axis_normalized_rms,
        component_rms=component_rms,
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid

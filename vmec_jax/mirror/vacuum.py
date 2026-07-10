"""Open-annulus scalar-potential vacuum field for mirror equilibria.

The vacuum map uses ``rho in [0,1]`` between the plasma side boundary and a
fixed circular outer cylinder.  The correction field ``grad(nu)`` minimizes
the vacuum magnetic energy.  Fixed potential values on the outer cylinder
and axial cuts preserve the supplied external field there; the plasma-side
boundary is free, so stationarity supplies the natural ``B_vac dot n = 0``
condition.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator, cg

from .basis import ChebyshevBasis, ThetaBasis
from .forces import (
    InterfaceResidual,
    MirrorEnergy,
    interface_residual,
    mirror_energy,
)
from .model import MirrorBoundary, MirrorConfig, MirrorState

Array = Any
MU0 = 4.0e-7 * np.pi


@dataclass(frozen=True, eq=False)
class VacuumGrid:
    """CGL radial/axial and Fourier poloidal vacuum collocation grid."""

    radial_basis: ChebyshevBasis
    theta_basis: ThetaBasis
    axial_basis: ChebyshevBasis
    z: np.ndarray
    dz_dxi: float

    @property
    def nrho(self) -> int:
        return self.radial_basis.size

    @property
    def ntheta(self) -> int:
        return self.theta_basis.size

    @property
    def nxi(self) -> int:
        return self.axial_basis.size

    @property
    def rho(self) -> np.ndarray:
        return 0.5 * (self.radial_basis.nodes + 1.0)

    @property
    def radial_weights(self) -> np.ndarray:
        return 0.5 * self.radial_basis.weights

    @property
    def theta(self) -> np.ndarray:
        return self.theta_basis.nodes

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nrho, self.ntheta, self.nxi)

    def radial_derivative(self, values: Array) -> Array:
        """Differentiate with respect to ``rho in [0,1]``."""

        return 2.0 * self.radial_basis.differentiate(values, axis=0)


def build_vacuum_grid(plasma_grid: "MirrorGrid", *, nrho: int | None = None) -> VacuumGrid:
    """Build the Fourier-CGL annulus grid sharing plasma theta/end nodes."""

    return VacuumGrid(
        radial_basis=ChebyshevBasis.build(plasma_grid.ns if nrho is None else int(nrho)),
        theta_basis=plasma_grid.theta_basis,
        axial_basis=plasma_grid.axial_basis,
        z=np.asarray(plasma_grid.z),
        dz_dxi=float(plasma_grid.dz_dxi),
    )


@dataclass(frozen=True)
class VacuumGeometry:
    """Annular embedding, metric, basis, normal, and volume."""

    xyz: Array
    basis_xyz: Array
    inverse_metric: Array
    sqrt_g: Array
    inner_normal_xyz: Array
    volume: Array
    valid: Array


@dataclass(frozen=True)
class VacuumField:
    """Scalar-potential correction and total Cartesian vacuum field."""

    correction_xyz: Array
    total_xyz: Array
    b_normal_inner: Array
    correction_normal_outer: Array
    correction_normal_lower: Array
    correction_normal_upper: Array
    energy: Array
    laplacian: Array


@dataclass(frozen=True)
class VacuumSolveResult:
    """Solved potential and independent vacuum diagnostics."""

    potential: Array
    geometry: VacuumGeometry
    field: VacuumField
    variational_max: Array
    laplacian_rms: Array
    b_normal_rms: Array
    linear_residual: Array
    iterations: int
    converged: bool
    message: str


@dataclass(frozen=True)
class FreeBoundaryMirrorResult:
    """Joint axisymmetric plasma-boundary-vacuum equilibrium result."""

    boundary: MirrorBoundary
    plasma_state: MirrorState
    plasma_energy: MirrorEnergy
    vacuum_geometry: VacuumGeometry
    vacuum_field: VacuumField
    vacuum_potential: Array
    interface: InterfaceResidual
    history: Array
    variational_max: Array
    iterations: int
    converged: bool
    optimizer_success: bool
    message: str


for _cls in (VacuumGeometry, VacuumField, VacuumSolveResult, FreeBoundaryMirrorResult):
    data = [field.name for field in fields(_cls) if field.name not in {"iterations", "converged", "message"}]
    meta = [field.name for field in fields(_cls) if field.name in {"iterations", "converged", "message"}]
    jax.tree_util.register_dataclass(_cls, data_fields=data, meta_fields=meta)


def evaluate_vacuum_geometry(
    boundary: MirrorBoundary,
    grid: VacuumGrid,
    *,
    outer_radius: float,
) -> VacuumGeometry:
    """Map the plasma boundary to a fixed circular outer cylinder."""

    if outer_radius <= 0.0:
        raise ValueError("outer_radius must be positive")
    a = jnp.asarray(boundary.radius_scale)[None, :, :]
    rho = jnp.asarray(grid.rho)[:, None, None]
    gap = float(outer_radius) - a
    radius = a + rho * gap
    a_theta = grid.theta_basis.differentiate(a, axis=1)
    a_xi = grid.axial_basis.differentiate(a, axis=2)
    r_rho = jnp.broadcast_to(gap, radius.shape)
    r_theta = (1.0 - rho) * a_theta
    r_xi = (1.0 - rho) * a_xi
    length_scale = float(grid.dz_dxi)

    g_rr = r_rho**2
    g_rtheta = r_rho * r_theta
    g_rxi = r_rho * r_xi
    g_thetatheta = r_theta**2 + radius**2
    g_thetaxi = r_theta * r_xi
    g_xixi = r_xi**2 + length_scale**2
    metric = jnp.stack(
        [
            jnp.stack([g_rr, g_rtheta, g_rxi], axis=-1),
            jnp.stack([g_rtheta, g_thetatheta, g_thetaxi], axis=-1),
            jnp.stack([g_rxi, g_thetaxi, g_xixi], axis=-1),
        ],
        axis=-2,
    )
    inverse_metric = jnp.linalg.inv(metric)
    sqrt_g = radius * gap * length_scale

    theta = jnp.asarray(grid.theta)[None, :, None]
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    zeros = jnp.zeros_like(radius)
    e_rho = jnp.stack([r_rho * cosine, r_rho * sine, zeros], axis=-1)
    e_theta = jnp.stack(
        [r_theta * cosine - radius * sine, r_theta * sine + radius * cosine, zeros],
        axis=-1,
    )
    e_xi = jnp.stack(
        [r_xi * cosine, r_xi * sine, jnp.full_like(radius, length_scale)], axis=-1
    )
    basis = jnp.stack([e_rho, e_theta, e_xi], axis=-1)
    inner_area_vector = jnp.cross(e_theta[0], e_xi[0])
    inner_normal = inner_area_vector / jnp.linalg.norm(inner_area_vector, axis=-1)[..., None]
    z = jnp.asarray(grid.z)[None, None, :]
    xyz = jnp.stack(
        [radius * cosine, radius * sine, jnp.broadcast_to(z, radius.shape)], axis=-1
    )
    volume = jnp.einsum(
        "i,j,k,ijk->",
        jnp.asarray(grid.radial_weights),
        jnp.asarray(grid.theta_basis.weights),
        jnp.asarray(grid.axial_basis.weights),
        sqrt_g,
    )
    return VacuumGeometry(
        xyz=xyz,
        basis_xyz=basis,
        inverse_metric=inverse_metric,
        sqrt_g=sqrt_g,
        inner_normal_xyz=inner_normal,
        volume=volume,
        valid=jnp.all(gap > 0.0),
    )


def _potential_gradient_xyz(
    potential: Array, geometry: VacuumGeometry, grid: VacuumGrid
) -> Array:
    partial = jnp.stack(
        [
            grid.radial_derivative(potential),
            grid.theta_basis.differentiate(potential, axis=1),
            grid.axial_basis.differentiate(potential, axis=2),
        ],
        axis=-1,
    )
    contravariant = jnp.einsum("...ij,...j->...i", geometry.inverse_metric, partial)
    return jnp.einsum("...ai,...i->...a", geometry.basis_xyz, contravariant)


def vacuum_laplacian(
    potential: Array, geometry: VacuumGeometry, grid: VacuumGrid
) -> Array:
    """Evaluate ``div(grad(nu))`` in annular coordinates."""

    partial = jnp.stack(
        [
            grid.radial_derivative(potential),
            grid.theta_basis.differentiate(potential, axis=1),
            grid.axial_basis.differentiate(potential, axis=2),
        ],
        axis=-1,
    )
    flux = geometry.sqrt_g[..., None] * jnp.einsum(
        "...ij,...j->...i", geometry.inverse_metric, partial
    )
    divergence = (
        grid.radial_derivative(flux[..., 0])
        + grid.theta_basis.differentiate(flux[..., 1], axis=1)
        + grid.axial_basis.differentiate(flux[..., 2], axis=2)
    )
    return divergence / geometry.sqrt_g


def evaluate_vacuum_field(
    potential: Array,
    geometry: VacuumGeometry,
    grid: VacuumGrid,
    external_field_xyz: Array,
    *,
    mu0: float = MU0,
) -> VacuumField:
    """Evaluate correction field, total field, energy, and tangency."""

    potential = jnp.asarray(potential)
    external = jnp.broadcast_to(jnp.asarray(external_field_xyz), geometry.xyz.shape)
    correction = _potential_gradient_xyz(potential, geometry, grid)
    total = external + correction
    b_normal = jnp.sum(total[0] * geometry.inner_normal_xyz, axis=-1)
    e_rho = geometry.basis_xyz[..., :, 0]
    e_theta = geometry.basis_xyz[..., :, 1]
    e_xi = geometry.basis_xyz[..., :, 2]
    outer_area = jnp.cross(e_theta[-1], e_xi[-1])
    outer_normal = outer_area / jnp.linalg.norm(outer_area, axis=-1)[..., None]
    end_area = jnp.cross(e_rho, e_theta)
    end_normal = end_area / jnp.linalg.norm(end_area, axis=-1)[..., None]
    energy_density = jnp.sum(total**2, axis=-1) * geometry.sqrt_g / (2.0 * float(mu0))
    energy = jnp.einsum(
        "i,j,k,ijk->",
        jnp.asarray(grid.radial_weights),
        jnp.asarray(grid.theta_basis.weights),
        jnp.asarray(grid.axial_basis.weights),
        energy_density,
    )
    return VacuumField(
        correction_xyz=correction,
        total_xyz=total,
        b_normal_inner=b_normal,
        correction_normal_outer=jnp.sum(correction[-1] * outer_normal, axis=-1),
        correction_normal_lower=-jnp.sum(
            correction[:, :, 0] * end_normal[:, :, 0], axis=-1
        ),
        correction_normal_upper=jnp.sum(
            correction[:, :, -1] * end_normal[:, :, -1], axis=-1
        ),
        energy=energy,
        laplacian=vacuum_laplacian(potential, geometry, grid),
    )


def external_field_from_coils(coilset: Any, geometry: VacuumGeometry) -> Array:
    """Evaluate the differentiable clean-core Biot-Savart field on the annulus."""

    from vmec_jax.core.coils import biot_savart

    return biot_savart(coilset, geometry.xyz)


def _external_flux_boundary_functional(
    potential: Array,
    geometry: VacuumGeometry,
    grid: VacuumGrid,
    external_field_xyz: Array,
    *,
    mu0: float = MU0,
) -> Array:
    """Boundary source that leaves external flux unchanged at outer/end cuts."""

    external = jnp.broadcast_to(jnp.asarray(external_field_xyz), geometry.xyz.shape)
    e_rho = geometry.basis_xyz[..., :, 0]
    e_theta = geometry.basis_xyz[..., :, 1]
    e_xi = geometry.basis_xyz[..., :, 2]
    outer_area = jnp.cross(e_theta[-1], e_xi[-1])
    outer_flux = jnp.sum(external[-1] * outer_area, axis=-1)
    outer = jnp.einsum(
        "j,k,jk->",
        jnp.asarray(grid.theta_basis.weights),
        jnp.asarray(grid.axial_basis.weights),
        potential[-1] * outer_flux,
    )
    end_area = jnp.cross(e_rho, e_theta)
    lower_flux = -jnp.sum(external[:, :, 0] * end_area[:, :, 0], axis=-1)
    upper_flux = jnp.sum(external[:, :, -1] * end_area[:, :, -1], axis=-1)
    ends = jnp.einsum(
        "i,j,ij->",
        jnp.asarray(grid.radial_weights),
        jnp.asarray(grid.theta_basis.weights),
        potential[:, :, 0] * lower_flux + potential[:, :, -1] * upper_flux,
    )
    return (outer + ends) / float(mu0)


def vacuum_energy_functional(
    potential: Array,
    geometry: VacuumGeometry,
    grid: VacuumGrid,
    external_field_xyz: Array,
    *,
    boundary_condition: str = "fixed_external_flux",
) -> Array:
    """Quadratic vacuum functional used by host and future JAX solvers."""

    energy = evaluate_vacuum_field(
        potential, geometry, grid, external_field_xyz
    ).energy
    if boundary_condition == "fixed_external_flux":
        return energy - _external_flux_boundary_functional(
            potential, geometry, grid, external_field_xyz
        )
    if boundary_condition == "fixed_potential":
        return energy
    raise ValueError(
        "boundary_condition must be 'fixed_external_flux' or 'fixed_potential'"
    )


def solve_vacuum_potential(
    boundary: MirrorBoundary,
    grid: VacuumGrid,
    config: MirrorConfig,
    external_field_xyz: Array,
    fixed_potential: Array,
    *,
    outer_radius: float,
    initial_potential: Array = 0.0,
    boundary_condition: str = "fixed_external_flux",
) -> VacuumSolveResult:
    """Solve the quadratic open-annulus scalar-potential problem.

    ``fixed_external_flux`` is the production fixed-flux-cut policy: the
    correction has zero normal flux on the outer boundary and end cuts, and a
    single nodal value removes the constant gauge. ``fixed_potential`` keeps
    prescribed outer/end values and is useful for Dirichlet MMS cases.
    """

    geometry = evaluate_vacuum_geometry(boundary, grid, outer_radius=outer_radius)
    fixed = jnp.broadcast_to(jnp.asarray(fixed_potential), grid.shape)
    free_mask = np.ones(grid.shape, dtype=bool)
    if boundary_condition == "fixed_potential":
        free_mask[-1] = False
        free_mask[:, :, [0, -1]] = False
    elif boundary_condition == "fixed_external_flux":
        free_mask[-1, 0, 0] = False
        fixed = fixed.at[-1, 0, 0].set(0.0)
    else:
        raise ValueError(
            "boundary_condition must be 'fixed_external_flux' or 'fixed_potential'"
        )
    indices = tuple(np.asarray(index) for index in np.nonzero(free_mask))
    initial = np.broadcast_to(np.asarray(initial_potential, dtype=float), grid.shape)
    x0 = initial[indices]

    def unpack(vector: Array) -> Array:
        return fixed.at[indices].set(jnp.asarray(vector))

    def objective(vector: Array) -> Array:
        potential = unpack(vector)
        return vacuum_energy_functional(
            potential,
            geometry,
            grid,
            external_field_xyz,
            boundary_condition=boundary_condition,
        )

    initial_energy = max(abs(float(objective(jnp.asarray(x0)))), 1.0)
    normalized = lambda vector: objective(vector) / initial_energy
    gradient = jax.jit(jax.grad(normalized))
    hessian_vector = jax.jit(
        lambda point, direction: jax.jvp(gradient, (point,), (direction,))[1]
    )
    initial_gradient = np.asarray(gradient(jnp.asarray(x0)), dtype=float)

    if x0.size <= 512:
        hessian = np.asarray(jax.jacfwd(gradient)(jnp.asarray(x0)), dtype=float)
        step = np.linalg.solve(hessian, -initial_gradient)
        solution = x0 + step
        iterations = 1
        linear_residual = float(
            np.linalg.norm(hessian @ step + initial_gradient)
            / max(np.linalg.norm(initial_gradient), np.finfo(float).tiny)
        )
        message = "dense exact quadratic solve"
    else:
        operator = LinearOperator(
            (x0.size, x0.size),
            matvec=lambda direction: np.asarray(
                hessian_vector(jnp.asarray(x0), jnp.asarray(direction)), dtype=float
            ),
            dtype=float,
        )
        iterations = 0

        def count(_vector: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        step, info = cg(
            operator,
            -initial_gradient,
            rtol=min(1.0e-12, config.ftol),
            atol=0.0,
            maxiter=config.max_iterations,
            callback=count,
        )
        solution = x0 + step
        defect = operator.matvec(step) + initial_gradient
        linear_residual = float(
            np.linalg.norm(defect)
            / max(np.linalg.norm(initial_gradient), np.finfo(float).tiny)
        )
        message = "CG converged" if info == 0 else f"CG status {info}"

    potential = unpack(jnp.asarray(solution))
    field = evaluate_vacuum_field(potential, geometry, grid, external_field_xyz)
    final_gradient = gradient(jnp.asarray(solution))
    variational_max = jnp.max(jnp.abs(final_gradient))
    active_laplacian = field.laplacian[1:-1, :, 1:-1]
    laplacian_rms = jnp.sqrt(jnp.mean(active_laplacian**2))
    b_normal_rms = jnp.sqrt(jnp.mean(field.b_normal_inner[:, 1:-1] ** 2))
    converged = bool(float(variational_max) <= config.ftol and bool(geometry.valid))
    return VacuumSolveResult(
        potential=potential,
        geometry=geometry,
        field=field,
        variational_max=variational_max,
        laplacian_rms=laplacian_rms,
        b_normal_rms=b_normal_rms,
        linear_residual=jnp.asarray(linear_residual),
        iterations=iterations,
        converged=converged,
        message=message,
    )


def solve_axisymmetric_free_boundary_cli(
    initial_boundary: MirrorBoundary,
    plasma_grid: "MirrorGrid",
    vacuum_grid: VacuumGrid,
    config: MirrorConfig,
    coilset: Any,
    *,
    outer_radius: float,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    initial_state: MirrorState | None = None,
    initial_potential: Array | None = None,
    require_convergence: bool = False,
) -> FreeBoundaryMirrorResult:
    """Jointly solve an isotropic axisymmetric plasma and open vacuum.

    The active vector contains the lateral boundary away from the fixed cuts,
    the plasma interior map, and every vacuum-potential node except one gauge
    value. The direct coil field is reevaluated on each moving annulus.
    """

    if plasma_grid.ntheta != 1 or vacuum_grid.ntheta != 1:
        raise ValueError("axisymmetric free-boundary solve requires ntheta=1")
    if plasma_grid.nxi != vacuum_grid.nxi:
        raise ValueError("plasma and vacuum grids must share axial nodes")
    initial_boundary_radius = np.asarray(initial_boundary.radius_scale, dtype=float)
    if initial_boundary_radius.shape != (1, plasma_grid.nxi):
        raise ValueError("initial boundary does not match the axisymmetric grid")
    boundary_scale = float(np.mean(initial_boundary_radius))
    flux = np.asarray(axial_flux_derivative, dtype=float)
    flux_scale = max(float(np.max(np.abs(flux))), np.finfo(float).tiny)
    potential_scale = max(
        2.0 * flux_scale / boundary_scale**2 * float(plasma_grid.dz_dxi),
        np.finfo(float).tiny,
    )

    boundary_indices = np.arange(1, plasma_grid.nxi - 1)
    plasma_mask = np.zeros(plasma_grid.shape, dtype=bool)
    plasma_mask[1:-1, 0, 1:-1] = True
    plasma_indices = tuple(np.asarray(index) for index in np.nonzero(plasma_mask))
    vacuum_mask = np.ones(vacuum_grid.shape, dtype=bool)
    vacuum_mask[-1, 0, 0] = False
    vacuum_indices = tuple(np.asarray(index) for index in np.nonzero(vacuum_mask))
    nb = boundary_indices.size
    np_state = plasma_indices[0].size
    nv = vacuum_indices[0].size

    base_state = MirrorState.from_boundary(initial_boundary, plasma_grid) if initial_state is None else initial_state
    base_state.validate_shape(plasma_grid)
    if not np.allclose(np.asarray(base_state.radius_scale[-1]), initial_boundary_radius):
        raise ValueError("initial_state boundary must match initial_boundary")
    potential_seed = np.zeros(vacuum_grid.shape) if initial_potential is None else np.asarray(initial_potential)
    if potential_seed.shape != vacuum_grid.shape:
        raise ValueError(f"initial_potential shape {potential_seed.shape} must be {vacuum_grid.shape}")
    x0 = np.concatenate(
        [
            initial_boundary_radius[0, boundary_indices] / boundary_scale,
            np.asarray(base_state.radius_scale)[plasma_indices] / boundary_scale,
            potential_seed[vacuum_indices] / potential_scale,
        ]
    )
    geometric_upper = 0.98 * float(outer_radius) / boundary_scale
    if np.max(x0[:nb]) >= geometric_upper:
        raise ValueError("initial plasma boundary must lie inside the outer vacuum cylinder")
    lower = np.concatenate([np.full(nb + np_state, 0.2), np.full(nv, -np.inf)])
    upper = np.concatenate(
        [np.full(nb + np_state, geometric_upper), np.full(nv, np.inf)]
    )

    def unpack(vector: Array) -> tuple[MirrorBoundary, MirrorState, Array]:
        vector = jnp.asarray(vector)
        boundary_radius = jnp.asarray(initial_boundary_radius).at[
            0, jnp.asarray(boundary_indices)
        ].set(vector[:nb] * boundary_scale)
        boundary = MirrorBoundary(boundary_radius)
        radius = base_state.radius_scale.at[plasma_indices].set(
            vector[nb : nb + np_state] * boundary_scale
        )
        radius = radius.at[-1].set(boundary_radius)
        radius = radius.at[:, :, 0].set(boundary_radius[:, 0])
        radius = radius.at[:, :, -1].set(boundary_radius[:, -1])
        radius = radius.at[0].set(radius[1])
        state = MirrorState(radius, base_state.lambda_stream)
        potential = jnp.zeros(vacuum_grid.shape).at[vacuum_indices].set(
            vector[nb + np_state :] * potential_scale
        )
        return boundary, state, potential

    def components(vector: Array):
        boundary, state, potential = unpack(vector)
        plasma = mirror_energy(
            state,
            plasma_grid,
            axial_flux_derivative=axial_flux_derivative,
            mass_profile=mass_profile,
            current_derivative=current_derivative,
            gamma=gamma,
        )
        vacuum_geometry = evaluate_vacuum_geometry(
            boundary, vacuum_grid, outer_radius=outer_radius
        )
        external = external_field_from_coils(coilset, vacuum_geometry)
        vacuum_functional = vacuum_energy_functional(
            potential, vacuum_geometry, vacuum_grid, external
        )
        vacuum_field = evaluate_vacuum_field(
            potential, vacuum_geometry, vacuum_grid, external
        )
        return plasma, vacuum_geometry, vacuum_field, vacuum_functional

    initial_components = components(jnp.asarray(x0))
    plasma_scale = max(abs(float(initial_components[0].total)), 1.0)
    vacuum_scale = max(abs(float(initial_components[3])), 1.0)

    def plasma_objective(vector: Array) -> Array:
        return components(vector)[0].total / plasma_scale

    def vacuum_objective(vector: Array) -> Array:
        return components(vector)[3] / vacuum_scale

    def residual_function(vector: Array) -> Array:
        plasma, _, vacuum_field, _ = components(vector)
        plasma_gradient = jax.grad(plasma_objective)(vector)[nb : nb + np_state]
        vacuum_gradient = jax.grad(vacuum_objective)(vector)[nb + np_state :]
        plasma_b_squared = plasma.b_squared[-1, 0, 1:-1]
        vacuum_b_squared = jnp.sum(
            vacuum_field.total_xyz[0, 0, 1:-1] ** 2, axis=-1
        )
        pressure = jnp.broadcast_to(plasma.pressure[-1], plasma_b_squared.shape)
        jump = pressure + plasma_b_squared / (2.0 * MU0) - vacuum_b_squared / (
            2.0 * MU0
        )
        stress_scale = (
            jnp.abs(pressure)
            + plasma_b_squared / (2.0 * MU0)
            + vacuum_b_squared / (2.0 * MU0)
        )
        stress = jump / jnp.maximum(
            stress_scale, jnp.finfo(stress_scale.dtype).tiny
        )
        return jnp.concatenate([stress, plasma_gradient, vacuum_gradient])

    residual_jit = jax.jit(residual_function)
    jacobian_jit = jax.jit(jax.jacfwd(residual_function))

    history: list[tuple[float, float, float, float, float]] = []
    last_recorded: np.ndarray | None = None

    def residual_host(vector: np.ndarray) -> np.ndarray:
        nonlocal last_recorded
        residual = np.asarray(residual_jit(jnp.asarray(vector)), dtype=float)
        if last_recorded is None or not np.array_equal(vector, last_recorded):
            history.append(
                (
                    float(len(history)),
                    float(np.sqrt(np.mean(residual[:nb] ** 2))),
                    float(np.sqrt(np.mean(residual[nb : nb + np_state] ** 2))),
                    float(np.sqrt(np.mean(residual[nb + np_state :] ** 2))),
                    float(np.max(np.abs(residual))),
                )
            )
            last_recorded = np.array(vector, copy=True)
        return residual

    solve = least_squares(
        fun=residual_host,
        x0=x0,
        jac=lambda vector: np.asarray(jacobian_jit(jnp.asarray(vector)), dtype=float),
        bounds=(lower, upper),
        method="trf",
        ftol=1.0e-14,
        xtol=1.0e-14,
        gtol=1.0e-14,
        x_scale="jac",
        max_nfev=config.max_iterations,
    )
    solution = np.asarray(solve.x)

    boundary, state, potential = unpack(jnp.asarray(solution))
    plasma = mirror_energy(
        state,
        plasma_grid,
        axial_flux_derivative=axial_flux_derivative,
        mass_profile=mass_profile,
        current_derivative=current_derivative,
        gamma=gamma,
    )
    vacuum_geometry = evaluate_vacuum_geometry(
        boundary, vacuum_grid, outer_radius=outer_radius
    )
    external = external_field_from_coils(coilset, vacuum_geometry)
    vacuum_field = evaluate_vacuum_field(
        potential, vacuum_geometry, vacuum_grid, external
    )
    plasma_b_squared = plasma.b_squared[-1]
    vacuum_b_squared = jnp.sum(vacuum_field.total_xyz[0] ** 2, axis=-1)
    active_axial_weights = jnp.asarray(plasma_grid.axial_basis.weights).at[
        jnp.asarray([0, plasma_grid.nxi - 1])
    ].set(0.0)
    interface = interface_residual(
        perpendicular_pressure=jnp.broadcast_to(plasma.pressure[-1], plasma_b_squared.shape),
        plasma_b_squared=plasma_b_squared,
        vacuum_b_squared=vacuum_b_squared,
        plasma_b_normal=jnp.zeros_like(plasma_b_squared),
        vacuum_b_normal=vacuum_field.b_normal_inner,
        theta_weights=jnp.asarray(plasma_grid.theta_basis.weights),
        axial_weights=active_axial_weights,
    )
    final_residual = np.asarray(residual_jit(jnp.asarray(solution)), dtype=float)
    variational_max = float(np.max(np.abs(final_residual)))
    converged = bool(
        variational_max <= config.ftol
        and not bool(plasma.geometry.jacobian_sign_changed)
        and bool(vacuum_geometry.valid)
    )
    message = str(solve.message)
    if not converged:
        message += f"; variational force={variational_max:.3e}"
    result = FreeBoundaryMirrorResult(
        boundary=boundary,
        plasma_state=state,
        plasma_energy=plasma,
        vacuum_geometry=vacuum_geometry,
        vacuum_field=vacuum_field,
        vacuum_potential=potential,
        interface=interface,
        history=jnp.asarray(history),
        variational_max=jnp.asarray(variational_max),
        iterations=int(solve.nfev),
        converged=converged,
        optimizer_success=bool(solve.success),
        message=message,
    )
    if require_convergence and not converged:
        raise RuntimeError(message)
    return result


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid

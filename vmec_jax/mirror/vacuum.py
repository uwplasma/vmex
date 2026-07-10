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
from scipy.sparse.linalg import LinearOperator, cg

from .basis import ChebyshevBasis, ThetaBasis
from .model import MirrorBoundary, MirrorConfig

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


for _cls in (VacuumGeometry, VacuumField, VacuumSolveResult):
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
        energy = evaluate_vacuum_field(
            potential, geometry, grid, external_field_xyz
        ).energy
        if boundary_condition == "fixed_external_flux":
            energy -= _external_flux_boundary_functional(
                potential, geometry, grid, external_field_xyz
            )
        return energy

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


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid

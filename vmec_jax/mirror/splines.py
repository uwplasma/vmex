"""Clamped cubic B-spline coefficients for open mirror equilibria.

Knot locations are static NumPy data; coefficient evaluation and transfer are
differentiable JAX operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .basis import CubicBSplineBasis, MirrorGrid, ThetaBasis
from .geometry import evaluate_geometry, regularize_axis_stream_function
from .model import _regularize_axis_radius, MirrorBoundary, MirrorConfig, MirrorResolution, MirrorState

Array = Any


@dataclass(frozen=True, eq=False)
class _SplineEvaluationBasis:
    """Endpoint-augmented Gauss grid acting on evaluated spline values."""

    spline: CubicBSplineBasis
    nodes: np.ndarray
    weights: np.ndarray
    recovery_matrix: np.ndarray
    derivative_matrix: np.ndarray
    second_derivative_matrix: np.ndarray

    @classmethod
    def build(cls, spline: CubicBSplineBasis) -> "_SplineEvaluationBasis":
        nodes = np.concatenate(([spline.domain[0]], spline.quadrature_nodes, [spline.domain[1]]))
        weights = np.concatenate(([0.0], spline.quadrature_weights, [0.0]))
        values = np.asarray(spline.basis_matrix(nodes))
        recovery = np.linalg.pinv(values, rcond=1.0e-14)
        derivative = np.asarray(spline.basis_matrix(nodes, derivative=1)) @ recovery
        second = np.asarray(spline.basis_matrix(nodes, derivative=2)) @ recovery
        return cls(spline, nodes, weights, recovery, derivative, second)

    @property
    def size(self) -> int:
        return int(self.nodes.size)

    @staticmethod
    def _apply(matrix: Array, values: Array, axis: int) -> Array:
        values = jnp.asarray(values)
        moved = jnp.moveaxis(values, axis, 0)
        result = jnp.tensordot(jnp.asarray(matrix), moved, axes=((1,), (0,)))
        return jnp.moveaxis(result, 0, axis)

    def differentiate(self, values: Array, *, axis: int = -1) -> Array:
        return self._apply(self.derivative_matrix, values, axis)

    def differentiate_transpose(self, values: Array, *, axis: int = -1) -> Array:
        return self._apply(self.derivative_matrix.T, values, axis)

    def differentiate_twice(self, values: Array, *, axis: int = -1) -> Array:
        return self._apply(self.second_derivative_matrix, values, axis)

    def integrate(self, values: Array, *, axis: int = -1) -> Array:
        moved = jnp.moveaxis(jnp.asarray(values), axis, -1)
        return jnp.tensordot(moved, jnp.asarray(self.weights), axes=((-1,), (0,)))

    def interpolation_matrix(self, target_nodes: Array) -> np.ndarray:
        return np.asarray(self.spline.basis_matrix(target_nodes)) @ self.recovery_matrix

    def interpolate(self, values: Array, target_nodes: Array, *, axis: int = -1) -> Array:
        return self._apply(self.interpolation_matrix(target_nodes), values, axis)


@dataclass(frozen=True)
class SplineMirrorBoundary:
    """Lateral mirror boundary stored as axial B-spline coefficients."""

    radius_coefficients: Array


@dataclass(frozen=True)
class SplineMirrorState:
    """Geometry and stream-function B-spline coefficients."""

    radius_coefficients: Array
    lambda_coefficients: Array


@dataclass(frozen=True)
class SuppliedFieldInitialization:
    """Spline state and radial flux profile inferred from a vacuum field."""

    state: SplineMirrorState
    axial_flux_derivative: Array


def _integrate_poloidal_derivative(derivative: Array) -> Array:
    """Invert a resolved theta derivative with a zero-mean gauge."""

    ntheta = derivative.shape[1]
    if ntheta == 1:
        return jnp.zeros_like(derivative)
    modes = np.fft.fftfreq(ntheta, d=1.0 / ntheta)
    inverse = np.zeros(ntheta, dtype=complex)
    nonzero = modes != 0.0
    inverse[nonzero] = 1.0 / (1j * modes[nonzero])
    if ntheta % 2 == 0:
        inverse[ntheta // 2] = 0.0
    shape = (1, ntheta) + (1,) * (derivative.ndim - 2)
    return jnp.fft.ifft(
        jnp.fft.fft(derivative, axis=1) * jnp.asarray(inverse).reshape(shape),
        axis=1,
    ).real


@dataclass(frozen=True, eq=False)
class SplineMirrorDiscretization:
    """Coefficient-to-quadrature map for a fixed mirror configuration."""

    spline: CubicBSplineBasis
    grid: MirrorGrid
    evaluation_matrix: np.ndarray

    @staticmethod
    def _grid(resolution: MirrorResolution, axial: Any, z: np.ndarray, dz_dxi: float) -> MirrorGrid:
        """Build the radial and poloidal tensor factors shared by spline grids."""

        s = np.linspace(0.0, 1.0, resolution.ns)
        radial_weights = np.full(resolution.ns, 1.0 / (resolution.ns - 1))
        radial_weights[[0, -1]] *= 0.5
        return MirrorGrid(
            s=s,
            s_half=0.5 * (s[:-1] + s[1:]),
            radial_weights=radial_weights,
            theta_basis=ThetaBasis.build(resolution.ntheta, resolution.mpol),
            axial_basis=axial,
            z=z,
            dz_dxi=dz_dxi,
        )

    @classmethod
    def build(
        cls,
        config: MirrorConfig,
        *,
        elements: int,
        quadrature_order: int = 4,
    ) -> "SplineMirrorDiscretization":
        """Build a clamped spline and endpoint-augmented Gauss mirror grid."""

        elements = int(elements)
        if elements < 1:
            raise ValueError("spline discretization requires elements >= 1")
        spline = CubicBSplineBasis.clamped(np.linspace(-1.0, 1.0, elements + 1), quadrature_order=quadrature_order)
        axial = _SplineEvaluationBasis.build(spline)
        z_mid = 0.5 * (config.z_min + config.z_max)
        dz_dxi = 0.5 * (config.z_max - config.z_min)
        grid = cls._grid(config.resolution, axial, z_mid + dz_dxi * axial.nodes, dz_dxi)
        return cls(spline, grid, np.asarray(spline.basis_matrix(axial.nodes)))

    @classmethod
    def build_cgl(
        cls,
        config: MirrorConfig,
        *,
        elements: int,
        quadrature_order: int = 4,
    ) -> "SplineMirrorDiscretization":
        """Build coefficients evaluated on the CGL grid used by exterior panels."""

        elements = int(elements)
        if elements < 1:
            raise ValueError("spline discretization requires elements >= 1")
        spline = CubicBSplineBasis.clamped(
            np.linspace(-1.0, 1.0, elements + 1),
            quadrature_order=quadrature_order,
        )
        grid = config.build_grid()
        return cls(spline, grid, np.asarray(spline.basis_matrix(grid.xi)))

    @property
    def coefficient_count(self) -> int:
        return self.spline.size

    def evaluate_boundary(self, boundary: SplineMirrorBoundary) -> MirrorBoundary:
        """Evaluate boundary coefficients on the solver quadrature grid."""

        coefficients = jnp.asarray(boundary.radius_coefficients)
        expected = (self.grid.ntheta, self.coefficient_count)
        if coefficients.shape != expected:
            raise ValueError(f"boundary coefficient shape {coefficients.shape} must be {expected}")
        values = jnp.tensordot(coefficients, jnp.asarray(self.evaluation_matrix).T, axes=((-1,), (0,)))
        return MirrorBoundary(values)

    def evaluate_state(self, state: SplineMirrorState) -> MirrorState:
        """Evaluate state coefficients on the solver quadrature grid."""

        expected = (self.grid.ns, self.grid.ntheta, self.coefficient_count)
        if state.radius_coefficients.shape != expected or state.lambda_coefficients.shape != expected:
            raise ValueError(f"state coefficient arrays must have shape {expected}")
        matrix = jnp.asarray(self.evaluation_matrix)
        return MirrorState(
            radius_scale=jnp.tensordot(state.radius_coefficients, matrix.T, axes=((-1,), (0,))),
            lambda_stream=jnp.tensordot(state.lambda_coefficients, matrix.T, axes=((-1,), (0,))),
        )

    def fit_boundary(self, boundary: MirrorBoundary, source_grid: MirrorGrid) -> SplineMirrorBoundary:
        """Fit a nodal boundary once to initialize coefficient-native solves."""

        samples = source_grid.axial_basis.interpolate(boundary.radius_scale, self.spline.collocation_nodes, axis=-1)
        return SplineMirrorBoundary(self.spline.fit(samples, axis=-1))

    def fit_state(self, state: MirrorState, source_grid: MirrorGrid) -> SplineMirrorState:
        """Fit a nodal state once to initialize coefficient-native solves."""

        radius = source_grid.axial_basis.interpolate(state.radius_scale, self.spline.collocation_nodes, axis=-1)
        lam = source_grid.axial_basis.interpolate(state.lambda_stream, self.spline.collocation_nodes, axis=-1)
        return SplineMirrorState(
            self.spline.fit(radius, axis=-1),
            self.spline.fit(lam, axis=-1),
        )

    def project_fixed_boundary(
        self,
        state: SplineMirrorState,
        boundary: SplineMirrorBoundary,
    ) -> SplineMirrorState:
        """Apply side geometry, axis regularity, and the lambda gauge.

        Clamped endpoint coefficients remain the prescribed cut profiles from
        ``state``. The coefficient vectorizer excludes them from every solve.
        """

        radius = jnp.asarray(state.radius_coefficients)
        boundary_radius = jnp.asarray(boundary.radius_coefficients)
        radius = radius.at[-1].set(boundary_radius)
        radius = _regularize_axis_radius(radius)
        lam = jnp.asarray(state.lambda_coefficients).at[0].set(state.lambda_coefficients[1])
        evaluated = jnp.tensordot(lam, jnp.asarray(self.evaluation_matrix).T, axes=((-1,), (0,)))
        theta_weights = jnp.asarray(self.grid.theta_basis.weights)
        axial_weights = jnp.asarray(self.grid.axial_basis.weights)
        mean = jnp.einsum("j,k,ijk->i", theta_weights, axial_weights, evaluated)
        mean /= jnp.sum(theta_weights) * jnp.sum(axial_weights)
        return SplineMirrorState(radius, lam - mean[:, None, None])

    def impose_self_similar_cuts(
        self,
        state: SplineMirrorState,
        boundary: SplineMirrorBoundary,
    ) -> SplineMirrorState:
        """Fix both end cuts to scaled copies of their LCFS sections.

        ``radius_scale`` is independent of ``s`` on a self-similar cut, while
        the physical radius still scales as ``sqrt(s)``. The stream function
        remains supplied by the field initializer.
        """

        radius = jnp.asarray(state.radius_coefficients)
        edge = jnp.asarray(boundary.radius_coefficients)
        radius = radius.at[:, :, 0].set(edge[:, 0][None, :])
        radius = radius.at[:, :, -1].set(edge[:, -1][None, :])
        return self.project_fixed_boundary(
            SplineMirrorState(radius, state.lambda_coefficients),
            boundary,
        )

    def transfer_boundary(
        self,
        state: SplineMirrorState,
        source: SplineMirrorBoundary,
        target: SplineMirrorBoundary,
    ) -> SplineMirrorState:
        """Rescale a nested restart from ``source`` to ``target`` boundary."""

        nodes = jnp.asarray(self.spline.collocation_nodes)
        source_radius = self.spline.evaluate(source.radius_coefficients, nodes)
        target_radius = self.spline.evaluate(target.radius_coefficients, nodes)
        if bool(jnp.any(source_radius <= 0.0)) or bool(jnp.any(target_radius <= 0.0)):
            raise ValueError("boundary transfer requires positive source and target radii")
        radius = self.spline.evaluate(state.radius_coefficients, nodes)
        transferred = self.spline.fit(radius * target_radius[None] / source_radius[None])
        return self.project_fixed_boundary(
            SplineMirrorState(
                transferred,
                state.lambda_coefficients,
            ),
            target,
        )


def initialize_from_cartesian_field(
    initial_state: SplineMirrorState,
    boundary: SplineMirrorBoundary,
    discretization: SplineMirrorDiscretization,
    field: Array | Callable[[Array], Array],
) -> SuppliedFieldInitialization:
    """Project a supplied vacuum field into the open Clebsch representation.

    ``field`` is either Cartesian samples with shape ``grid.shape + (3,)`` or
    a callable from one Cartesian point to one field vector. The geometry is
    kept fixed. The surface-averaged axial flux density determines
    ``Psi'(s)``, and the remaining nonzero poloidal modes determine the
    stream function. The small component normal to the supplied flux surfaces
    is intentionally discarded.
    """

    state = discretization.project_fixed_boundary(initial_state, boundary)
    evaluated = discretization.evaluate_state(state)
    geometry = evaluate_geometry(evaluated, discretization.grid)
    if not isinstance(geometry.jacobian_sign_changed, jax.core.Tracer) and bool(geometry.jacobian_sign_changed):
        raise ValueError("supplied-field initialization requires a positive Jacobian")
    if callable(field):
        points = geometry.xyz.reshape((-1, 3))
        supplied = jax.vmap(field)(points).reshape(geometry.xyz.shape)
    else:
        supplied = jnp.asarray(field)
    if supplied.shape != geometry.xyz.shape:
        raise ValueError(f"supplied Cartesian field shape {supplied.shape} must be {geometry.xyz.shape}")
    if not isinstance(supplied, jax.core.Tracer) and not np.all(np.isfinite(np.asarray(supplied))):
        raise ValueError("supplied Cartesian field must be finite")

    covariant_theta = jnp.sum(supplied * geometry.e_theta_xyz, axis=-1)
    covariant_xi = jnp.sum(supplied * geometry.e_xi_xyz, axis=-1)
    determinant = geometry.g_thetatheta * geometry.g_xixi - geometry.g_thetaxi**2
    numerator = geometry.g_thetatheta * covariant_xi - geometry.g_thetaxi * covariant_theta
    safe_determinant = determinant.at[0].set(1.0)
    b_sup_xi = numerator / safe_determinant
    b_sup_xi = b_sup_xi.at[0].set(b_sup_xi[1])
    jac_b_xi = geometry.sqrt_g * b_sup_xi
    jac_b_xi = jac_b_xi.at[0].set(jac_b_xi[1])

    theta_weights = jnp.asarray(discretization.grid.theta_basis.weights)
    axial_weights = jnp.asarray(discretization.grid.axial_basis.weights)
    theta_average = jnp.einsum("j,ijk->ik", theta_weights, jac_b_xi)
    theta_average /= jnp.sum(theta_weights)
    axial_flux = jnp.einsum("k,ik->i", axial_weights, theta_average)
    axial_flux /= jnp.sum(axial_weights)
    axial_flux = axial_flux.at[0].set(axial_flux[1])

    derivative_theta = jac_b_xi - axial_flux[:, None, None]
    stream = _integrate_poloidal_derivative(derivative_theta)
    stream = stream.at[0].set(stream[1])
    stream_at_nodes = discretization.grid.axial_basis.interpolate(
        stream,
        discretization.spline.collocation_nodes,
        axis=-1,
    )
    coefficients = discretization.spline.fit(stream_at_nodes, axis=-1)
    initialized = discretization.project_fixed_boundary(
        SplineMirrorState(
            state.radius_coefficients,
            coefficients,
        ),
        boundary,
    )
    return SuppliedFieldInitialization(initialized, axial_flux)


@dataclass(frozen=True)
class SplineMirrorSolveResult:
    """Converged coefficient state and its evaluated mirror result."""

    coefficient_state: SplineMirrorState
    evaluated: Any


@dataclass(frozen=True)
class _SplineStateVectorizer:
    """Pack constrained spline coefficients into normalized solve variables."""

    base: SplineMirrorState
    evaluation_matrix: np.ndarray
    radius_indices: tuple[np.ndarray, np.ndarray, np.ndarray]
    radius_scale: float
    flux_scale: float
    lambda_axial_indices: np.ndarray
    lambda_free_indices: np.ndarray
    lambda_pivot: int
    lambda_weights: np.ndarray
    lambda_fixed_weighted_sum: np.ndarray
    solve_lambda: bool

    @classmethod
    def build(
        cls,
        state: SplineMirrorState,
        boundary: SplineMirrorBoundary,
        discretization: SplineMirrorDiscretization,
        *,
        axial_flux_derivative: Array,
        solve_lambda: bool,
    ) -> "_SplineStateVectorizer":
        base = discretization.project_fixed_boundary(state, boundary)
        boundary_values = discretization.evaluate_boundary(boundary).radius_scale
        radius_scale = float(np.mean(np.asarray(boundary_values)))
        if not np.isfinite(radius_scale) or radius_scale <= 0.0:
            raise ValueError("mean boundary radius must be positive and finite")
        flux = np.asarray(axial_flux_derivative, dtype=float)
        flux_scale = max(float(np.max(np.abs(flux))), np.finfo(float).tiny)

        shape = np.asarray(base.radius_coefficients).shape
        radius_mask = np.zeros(shape, dtype=bool)
        radius_mask[1:-1, :, 1:-1] = True
        lambda_axial_indices = np.arange(1, shape[2] - 1)
        radius_indices = tuple(np.asarray(index) for index in np.nonzero(radius_mask))

        coefficient_weights = np.asarray(discretization.evaluation_matrix).T @ np.asarray(
            discretization.grid.axial_basis.weights
        )
        interior_weights = (
            np.asarray(discretization.grid.theta_basis.weights)[:, None]
            * coefficient_weights[None, lambda_axial_indices]
        ).reshape(-1)
        if solve_lambda and interior_weights.size < 2:
            raise ValueError("lambda solve requires at least two interior coefficients")
        pivot = int(np.argmax(interior_weights)) if interior_weights.size else 0
        free_indices = np.delete(np.arange(interior_weights.size), pivot)
        endpoint_weights = np.zeros((shape[1], shape[2]))
        endpoint_weights[:, [0, -1]] = (
            np.asarray(discretization.grid.theta_basis.weights)[:, None]
            * coefficient_weights[None, [0, -1]]
        )
        fixed_sum = np.einsum("jk,ijk->i", endpoint_weights, np.asarray(base.lambda_coefficients)[1:])
        return cls(
            base=base,
            evaluation_matrix=np.asarray(discretization.evaluation_matrix),
            radius_indices=radius_indices,
            radius_scale=radius_scale,
            flux_scale=flux_scale,
            lambda_axial_indices=lambda_axial_indices,
            lambda_free_indices=free_indices,
            lambda_pivot=pivot,
            lambda_weights=interior_weights,
            lambda_fixed_weighted_sum=fixed_sum,
            solve_lambda=bool(solve_lambda),
        )

    @property
    def radius_size(self) -> int:
        return int(self.radius_indices[0].size)

    @property
    def radius_poloidal_nodes(self) -> int:
        """Return the active radial-shape coordinates per axial coefficient."""

        return int(self.base.radius_coefficients.shape[1])

    @property
    def lambda_size(self) -> int:
        """Return the number of gauge-free stream-function coefficients."""

        if not self.solve_lambda:
            return 0
        return int((self.base.radius_coefficients.shape[0] - 1) * self.lambda_free_indices.size)

    @property
    def block_slices(self) -> tuple[slice, ...]:
        """Return packed radius and lambda blocks that are present."""

        offset = self.radius_size
        blocks = [slice(0, offset)]
        if self.lambda_size:
            blocks.append(slice(offset, offset + self.lambda_size))
        return tuple(blocks)

    def pack(self) -> np.ndarray:
        """Pack the projected coefficient state."""

        radius = np.asarray(self.base.radius_coefficients)[self.radius_indices]
        radius = radius / self.radius_scale
        blocks = [radius]
        if not self.solve_lambda:
            return np.concatenate(blocks)
        interior = np.asarray(self.base.lambda_coefficients)[1:, :, self.lambda_axial_indices].reshape(
            self.base.radius_coefficients.shape[0] - 1, -1
        )
        lam = interior[:, self.lambda_free_indices].reshape(-1) / self.flux_scale
        blocks.append(lam)
        return np.concatenate(blocks)

    def unpack(self, vector: Array) -> SplineMirrorState:
        """Reconstruct constrained coefficients from normalized variables."""

        vector = jnp.asarray(vector)
        radius = self.base.radius_coefficients.at[self.radius_indices].set(
            vector[: self.radius_size] * self.radius_scale
        )
        radius = _regularize_axis_radius(radius)
        offset = self.radius_size
        if not self.solve_lambda:
            return SplineMirrorState(radius, self.base.lambda_coefficients)

        shape = self.base.lambda_coefficients.shape
        free = vector[offset:].reshape(shape[0] - 1, self.lambda_free_indices.size) * self.flux_scale
        interior = self.base.lambda_coefficients[1:, :, self.lambda_axial_indices].reshape(shape[0] - 1, -1)
        interior = interior.at[:, jnp.asarray(self.lambda_free_indices)].set(free)
        weighted_free = jnp.sum(
            free * jnp.asarray(self.lambda_weights[self.lambda_free_indices])[None, :],
            axis=1,
        )
        pivot_value = -(jnp.asarray(self.lambda_fixed_weighted_sum) + weighted_free) / float(
            self.lambda_weights[self.lambda_pivot]
        )
        interior = interior.at[:, self.lambda_pivot].set(pivot_value)
        lam = self.base.lambda_coefficients.at[1:, :, self.lambda_axial_indices].set(
            interior.reshape(shape[0] - 1, shape[1], self.lambda_axial_indices.size)
        )
        return SplineMirrorState(radius, lam.at[0].set(lam[1]))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative normalized coefficient bounds."""

        radius_lower = np.full(self.radius_size, 0.2)
        radius_upper = np.full(self.radius_size, 5.0)
        lower = np.concatenate(
            (
                radius_lower,
                np.full(self.lambda_size, -np.inf),
            )
        )
        upper = np.concatenate(
            (
                radius_upper,
                np.full(self.lambda_size, np.inf),
            )
        )
        return lower, upper

    def pullback_evaluated_gradient(self, gradient: MirrorState) -> np.ndarray:
        """Pull an evaluated-state gradient to the active coefficients."""

        matrix = np.asarray(self.evaluation_matrix)
        radius_coefficients = np.tensordot(np.asarray(gradient.radius_scale), matrix, axes=((-1,), (0,)))
        axis_gradient = np.fft.fft(radius_coefficients[0], axis=0)
        modes = np.rint(np.fft.fftfreq(radius_coefficients.shape[1], d=1.0 / radius_coefficients.shape[1])).astype(
            int
        )
        axis_gradient[np.abs(modes) % 2 == 1] = 0.0
        radius_coefficients[1] += np.fft.ifft(axis_gradient, axis=0).real
        radius = radius_coefficients[self.radius_indices]
        radius = radius * self.radius_scale
        blocks = [radius]
        if not self.solve_lambda:
            return np.concatenate(blocks)

        lambda_coefficients = np.tensordot(np.asarray(gradient.lambda_stream), matrix, axes=((-1,), (0,)))
        lambda_coefficients[1] += lambda_coefficients[0]
        interior = lambda_coefficients[1:, :, self.lambda_axial_indices].reshape(
            lambda_coefficients.shape[0] - 1,
            -1,
        )
        pivot_gradient = interior[:, self.lambda_pivot]
        free = interior[:, self.lambda_free_indices] - (
            pivot_gradient[:, None]
            * self.lambda_weights[self.lambda_free_indices][None, :]
            / self.lambda_weights[self.lambda_pivot]
        )
        blocks.append((free * self.flux_scale).reshape(-1))
        return np.concatenate(blocks)


def _packed_spline_layout(
    discretization: SplineMirrorDiscretization,
    vectorizer: _SplineStateVectorizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return channel, radial, and axial labels for solve variables."""

    channels = np.zeros(vectorizer.radius_size, dtype=int)
    radial = np.asarray(vectorizer.radius_indices[0], dtype=int)
    axial = np.asarray(vectorizer.radius_indices[2], dtype=int)
    if vectorizer.lambda_size:
        channels = np.concatenate(
            (channels, np.full(vectorizer.lambda_size, 3, dtype=int))
        )
        axial_count = vectorizer.lambda_axial_indices.size
        free = vectorizer.lambda_free_indices
        radial = np.concatenate(
            (radial, np.repeat(np.arange(1, discretization.grid.ns), free.size))
        )
        axial = np.concatenate(
            (
                axial,
                np.tile(
                    vectorizer.lambda_axial_indices[free % axial_count],
                    discretization.grid.ns - 1,
                ),
            )
        )
    return channels, radial, axial


def _packed_spline_preconditioner(
    discretization: SplineMirrorDiscretization,
    vectorizer: _SplineStateVectorizer,
) -> tuple[Any, np.ndarray, Any]:
    """Build tensor fallback and optional local sparse Hessian factor."""

    from .solver import SeparableMirrorPreconditioner

    derivative = np.asarray(
        discretization.spline.basis_matrix(discretization.grid.axial_basis.nodes, derivative=1)
    ) / float(discretization.grid.dz_dxi)
    weights = np.asarray(discretization.grid.axial_basis.weights)
    active_derivative = derivative[:, 1:-1]
    stiffness = active_derivative.T @ (weights[:, None] * active_derivative)
    geometry = SeparableMirrorPreconditioner.build_from_axial_stiffness(
        discretization.grid,
        stiffness,
        poloidal_nodes=vectorizer.radius_poloidal_nodes,
    )
    stream = None
    if vectorizer.lambda_size:
        stream = SeparableMirrorPreconditioner.build_from_axial_stiffness(
            discretization.grid,
            stiffness,
            radial_nodes=discretization.grid.ns - 1,
        )
    scales = np.ones(len(vectorizer.block_slices))
    lambda_start = vectorizer.radius_size

    def apply(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        result = np.array(vector, copy=True)
        result[: vectorizer.radius_size] = geometry.apply(vector[: vectorizer.radius_size]) * scales[0]
        if stream is not None:
            reduced = vector[lambda_start:]
            reduced = stream.apply_gauge_free(
                reduced,
                free_indices=vectorizer.lambda_free_indices,
                pivot=vectorizer.lambda_pivot,
                weights=vectorizer.lambda_weights,
            )
            result[lambda_start:] = reduced * scales[1]
        return result

    def build_local(matrix_columns: Callable[[np.ndarray], np.ndarray]) -> Any:
        """Factor a frozen sparse Hessian from chunked matrix-free columns."""

        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu

        size = vectorizer.radius_size + vectorizer.lambda_size
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        chunk_size = min(32, size)
        channels, radial, axial = _packed_spline_layout(discretization, vectorizer)
        for start in range(0, size, chunk_size):
            columns = np.arange(start, min(start + chunk_size, size))
            directions = np.zeros((columns.size, size))
            directions[np.arange(columns.size), columns] = 1.0
            responses = np.asarray(matrix_columns(directions), dtype=float)
            for local_index, column in enumerate(columns):
                axial_neighbors = np.abs(axial - axial[column]) <= 4
                if channels[column] == 3:
                    axial_neighbors = np.where(channels == 3, True, axial_neighbors)
                rows = np.flatnonzero(
                    (np.abs(radial - radial[column]) <= 2) & axial_neighbors
                )
                row_parts.append(rows)
                column_parts.append(np.full(rows.size, column, dtype=int))
                value_parts.append(responses[local_index, rows])
        matrix = coo_matrix(
            (
                np.concatenate(value_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(size, size),
        ).tocsc()
        matrix = 0.5 * (matrix + matrix.T)
        factor = splu(matrix)

        def solve(vector: np.ndarray) -> np.ndarray:
            return factor.solve(vector)

        solve.hessian_probe_count = size  # type: ignore[attr-defined]
        solve.hessian_column_count = size  # type: ignore[attr-defined]
        return solve

    local_builder = build_local if vectorizer.lambda_size else None
    return apply, scales, local_builder


def solve_fixed_boundary_cli(
    initial_state: SplineMirrorState,
    boundary: SplineMirrorBoundary,
    discretization: SplineMirrorDiscretization,
    config: MirrorConfig,
    *,
    axial_flux_derivative: Array,
    mass_profile: Array = 0.0,
    current_derivative: Array = 0.0,
    gamma: float = 5.0 / 3.0,
    solve_lambda: bool = False,
    gradient_tolerance: float = 1.0e-11,
    require_convergence: bool = False,
) -> SplineMirrorSolveResult:
    """Solve a scalar-pressure, fixed-cut open mirror equilibrium."""

    from .forces import (
        VariationalResidual,
        isotropic_force_residual,
        isotropic_staggered_energy_gradient,
        mirror_energy,
    )
    from .geometry import normalized_divergence_rms
    from .solver import (
        MirrorConvergenceError,
        MirrorSolveResult,
        _optimize_fixed_boundary,
        _valid_energy_objective,
    )

    grid = discretization.grid
    if grid.ns != config.resolution.ns or grid.ntheta != config.resolution.ntheta:
        raise ValueError("spline radial and poloidal resolution must match MirrorConfig")
    if gradient_tolerance <= 0.0:
        raise ValueError("gradient_tolerance must be positive")
    vectorizer = _SplineStateVectorizer.build(
        initial_state,
        boundary,
        discretization,
        axial_flux_derivative=axial_flux_derivative,
        solve_lambda=solve_lambda,
    )
    x0 = vectorizer.pack()
    lower_bounds, upper_bounds = vectorizer.bounds()
    energy_kwargs = {
        "axial_flux_derivative": axial_flux_derivative,
        "mass_profile": mass_profile,
        "current_derivative": current_derivative,
        "gamma": gamma,
    }

    def unpack_coefficients(vector: Array) -> SplineMirrorState:
        return vectorizer.unpack(vector)

    def unpack(vector: Array) -> MirrorState:
        return discretization.evaluate_state(unpack_coefficients(vector))

    def evaluate_energy(state: MirrorState):
        return mirror_energy(state, grid, **energy_kwargs)

    initial_evaluated = unpack(jnp.asarray(x0))
    initial_energy = evaluate_energy(initial_evaluated)
    energy_scale = max(abs(float(initial_energy.total)), np.finfo(float).tiny)

    def objective(vector: Array) -> Array:
        return _valid_energy_objective(evaluate_energy(unpack(vector)), energy_scale)

    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    cache_x: np.ndarray | None = None
    cache_value = 0.0
    cache_gradient = np.empty_like(x0)

    def evaluate(vector: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal cache_x, cache_value, cache_gradient
        if cache_x is None or not np.array_equal(vector, cache_x):
            value, gradient = value_and_gradient(jnp.asarray(vector))
            cache_x = np.array(vector, copy=True)
            cache_value = float(value)
            cache_gradient = np.asarray(gradient, dtype=float)
        return cache_value, cache_gradient

    def packed_variational(vector: Array, state: MirrorState) -> VariationalResidual:
        del state
        packed = evaluate(np.asarray(vector, dtype=float))[1]
        radius = packed[: vectorizer.radius_size]
        lam = packed[vectorizer.radius_size:]
        lambda_rms = np.sqrt(np.mean(lam**2)) if lam.size else 0.0
        return VariationalResidual(
            radius_gradient=jnp.asarray(radius),
            lambda_gradient=jnp.asarray(lam),
            radius_rms=jnp.asarray(np.sqrt(np.mean(radius**2))),
            lambda_rms=jnp.asarray(lambda_rms),
            maximum=jnp.asarray(np.max(np.abs(packed))),
        )

    def packed_weak(state: MirrorState) -> VariationalResidual:
        gradient = isotropic_staggered_energy_gradient(
            state,
            grid,
            **energy_kwargs,
        )
        packed = vectorizer.pullback_evaluated_gradient(gradient) / energy_scale
        radius = packed[: vectorizer.radius_size]
        lam = packed[vectorizer.radius_size:]
        return VariationalResidual(
            radius_gradient=jnp.asarray(radius),
            lambda_gradient=jnp.asarray(lam),
            radius_rms=jnp.asarray(np.sqrt(np.mean(radius**2))),
            lambda_rms=jnp.asarray(np.sqrt(np.mean(lam**2)) if lam.size else 0.0),
            maximum=jnp.asarray(np.max(np.abs(packed))),
        )

    def force_residual(state, energy):
        return isotropic_force_residual(
            energy,
            grid,
            state=state,
            **energy_kwargs,
        )

    history: list[tuple[float, float, float, float, float, float]] = []

    def record(iteration: int, vector: np.ndarray) -> None:
        state = unpack(jnp.asarray(vector))
        energy = evaluate_energy(state)
        variational = packed_variational(vector, state)
        force = force_residual(state, energy)
        history.append(
            (
                float(iteration),
                float(energy.total),
                float(variational.radius_rms),
                float(variational.lambda_rms),
                float(variational.maximum),
                float(force.normalized_rms),
            )
        )

    record(0, x0)
    if history[-1][4] <= config.ftol and not bool(initial_energy.geometry.jacobian_sign_changed):
        final_x = x0
        iterations = 0
        optimizer_success = True
        linear_iterations = 0
        final_linear_residual = 0.0
        message = "initial spline state satisfies physical ftol"
    else:
        optimization = _optimize_fixed_boundary(
            x0,
            lower_bounds,
            upper_bounds,
            objective=objective,
            evaluate=evaluate,
            packed_variational=packed_variational,
            unpack=unpack,
            record=record,
            config=config,
            gradient_tolerance=gradient_tolerance,
            matrix_free_context=(
                vectorizer,
                _packed_spline_preconditioner(discretization, vectorizer),
            ),
            # Direct Newton is fast only after continuation enters its local basin.
            start_with_newton=bool(
                solve_lambda and history[-1][4] <= 1.0e-4 and x0.size <= 4096
            ),
        )
        final_x = optimization.vector
        iterations = optimization.iterations
        optimizer_success = optimization.optimizer_success
        linear_iterations = optimization.linear_iterations
        final_linear_residual = optimization.final_linear_residual
        message = optimization.message

    coefficient_state = discretization.project_fixed_boundary(
        unpack_coefficients(jnp.asarray(final_x)),
        boundary,
    )
    final_state = regularize_axis_stream_function(
        discretization.evaluate_state(coefficient_state),
        grid,
        energy_kwargs["axial_flux_derivative"],
    )
    final_energy = evaluate_energy(final_state)
    final_variational = packed_variational(final_x, final_state)
    final_force = force_residual(final_state, final_energy)
    final_weak = packed_weak(final_state)
    record(iterations, final_x)
    converged = bool(
        float(final_variational.maximum) <= config.ftol and not bool(final_energy.geometry.jacobian_sign_changed)
    )
    if not converged:
        message += f"; variational force={float(final_variational.maximum):.3e}"
    result = MirrorSolveResult(
        state=final_state,
        energy=final_energy,
        variational=final_variational,
        force=final_force,
        staggered_weak_force=final_weak,
        normalized_divergence_rms=normalized_divergence_rms(final_energy.field, final_energy.geometry, grid),
        history=jnp.asarray(history),
        iterations=iterations,
        converged=converged,
        optimizer_success=optimizer_success,
        linear_iterations=linear_iterations,
        final_linear_residual=final_linear_residual,
        message=message,
    )
    if require_convergence and not converged:
        raise MirrorConvergenceError(result)
    return SplineMirrorSolveResult(coefficient_state, result)


jax.tree_util.register_dataclass(SplineMirrorBoundary, data_fields=["radius_coefficients"], meta_fields=[])
jax.tree_util.register_dataclass(
    SplineMirrorState,
    data_fields=["radius_coefficients", "lambda_coefficients"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    SplineMirrorSolveResult,
    data_fields=["coefficient_state", "evaluated"],
    meta_fields=[],
)
__all__ = [
    "CubicBSplineBasis",
    "SplineMirrorBoundary",
    "SplineMirrorDiscretization",
    "SplineMirrorSolveResult",
    "SplineMirrorState",
    "solve_fixed_boundary_cli",
]

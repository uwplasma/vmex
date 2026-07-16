"""Small JAX cubic B-spline bases for mirror geometry coefficients.

The open basis is clamped at both end cuts. The periodic basis folds uniform
cardinal splines onto a closed interval. Knot locations are static NumPy data;
coefficient evaluation and transfer are differentiable JAX operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .basis import CubicBSplineBasis, MirrorGrid, ThetaBasis
from .geometry import evaluate_geometry, regularize_axis_stream_function
from .model import _regularize_axis_radius, MirrorBoundary, MirrorConfig, MirrorResolution, MirrorState

Array = Any


@dataclass(frozen=True, eq=False)
class _SplineEvaluationBasis:
    """Gauss grid acting on evaluated open or periodic spline values."""

    spline: CubicBSplineBasis
    nodes: np.ndarray
    weights: np.ndarray
    recovery_matrix: np.ndarray
    derivative_matrix: np.ndarray
    second_derivative_matrix: np.ndarray

    @classmethod
    def build(cls, spline: CubicBSplineBasis) -> "_SplineEvaluationBasis":
        if spline.periodic:
            nodes = spline.quadrature_nodes
            weights = spline.quadrature_weights
        else:
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
    """Geometry and stream-function B-spline coefficients.

    Closed states optionally carry transverse section-center coefficients with
    shape ``(ns, 2, coefficient_count)``. Open states leave them ``None``.
    """

    radius_coefficients: Array
    lambda_coefficients: Array
    center_coefficients: Array | None = None


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
    closed: bool = False

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
        return cls(spline, grid, np.asarray(spline.basis_matrix(axial.nodes)), False)

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
        return cls(spline, grid, np.asarray(spline.basis_matrix(grid.xi)), False)

    @classmethod
    def build_closed(
        cls,
        resolution: MirrorResolution,
        *,
        coefficient_count: int,
        quadrature_order: int = 4,
    ) -> "SplineMirrorDiscretization":
        """Build a periodic spline grid for a closed mirror-hybrid axis."""

        spline = CubicBSplineBasis.periodic_uniform(
            coefficient_count,
            quadrature_order=quadrature_order,
        )
        axial = _SplineEvaluationBasis.build(spline)
        grid = cls._grid(resolution, axial, np.asarray(axial.nodes), 1.0)
        return cls(spline, grid, np.asarray(spline.basis_matrix(axial.nodes)), True)

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
        center = state.center_coefficients
        if self.closed:
            center_expected = (self.grid.ns, 2, self.coefficient_count)
            if center is not None and center.shape != center_expected:
                raise ValueError(f"center coefficient array must have shape {center_expected}")
        elif center is not None:
            raise ValueError("open spline states do not accept center coefficients")
        matrix = jnp.asarray(self.evaluation_matrix)
        return MirrorState(
            radius_scale=jnp.tensordot(state.radius_coefficients, matrix.T, axes=((-1,), (0,))),
            lambda_stream=jnp.tensordot(state.lambda_coefficients, matrix.T, axes=((-1,), (0,))),
            center_shift=(
                None
                if center is None
                else jnp.tensordot(center, matrix.T, axes=((-1,), (0,)))
            ),
        )

    def fit_boundary(self, boundary: MirrorBoundary, source_grid: MirrorGrid) -> SplineMirrorBoundary:
        """Fit a nodal boundary once to initialize coefficient-native solves."""

        samples = source_grid.axial_basis.interpolate(boundary.radius_scale, self.spline.collocation_nodes, axis=-1)
        return SplineMirrorBoundary(self.spline.fit(samples, axis=-1))

    def fit_state(self, state: MirrorState, source_grid: MirrorGrid) -> SplineMirrorState:
        """Fit a nodal state once to initialize coefficient-native solves."""

        radius = source_grid.axial_basis.interpolate(state.radius_scale, self.spline.collocation_nodes, axis=-1)
        lam = source_grid.axial_basis.interpolate(state.lambda_stream, self.spline.collocation_nodes, axis=-1)
        center = None
        if state.center_shift is not None:
            center_nodes = source_grid.axial_basis.interpolate(
                state.center_shift,
                self.spline.collocation_nodes,
                axis=-1,
            )
            center = self.spline.fit(center_nodes, axis=-1)
        return SplineMirrorState(
            self.spline.fit(radius, axis=-1),
            self.spline.fit(lam, axis=-1),
            center,
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
        center = state.center_coefficients
        if self.closed:
            center_shape = (self.grid.ns, 2, self.coefficient_count)
            if center is not None and center.shape != center_shape:
                raise ValueError(f"center coefficient array must have shape {center_shape}")
            if center is not None:
                center = jnp.asarray(center).at[-1].set(0.0)
        elif center is not None:
            raise ValueError("open spline states do not accept center coefficients")
        return SplineMirrorState(radius, lam - mean[:, None, None], center)

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
                state.center_coefficients,
            ),
            target,
        )

    def transfer_closed_state(
        self,
        state: SplineMirrorState,
        source: "SplineMirrorDiscretization",
        boundary: SplineMirrorBoundary,
    ) -> SplineMirrorState:
        """Transfer a periodic restart onto a nested radial and axial grid."""

        if not self.closed or not source.closed:
            raise ValueError("closed state transfer requires two periodic discretizations")
        if self.grid.theta_basis.mpol < source.grid.theta_basis.mpol:
            raise ValueError("closed state transfer cannot reduce poloidal resolution")
        if self.spline.domain != source.spline.domain:
            raise ValueError("closed state transfer requires matching periodic domains")

        def transfer_axial(values: Array) -> Array:
            _, transferred = source.spline.refine_periodic_uniform(
                values,
                self.coefficient_count,
                axis=-1,
            )
            return transferred

        def transfer_radial(values: Array) -> Array:
            moved = jnp.moveaxis(jnp.asarray(values), 0, -1)
            shape = moved.shape[:-1]
            rows = moved.reshape((-1, source.grid.ns))
            transferred = jax.vmap(
                lambda row: jnp.interp(self.grid.s, source.grid.s, row)
            )(rows)
            return jnp.moveaxis(transferred.reshape(shape + (self.grid.ns,)), -1, 0)

        def transfer_poloidal(values: Array) -> Array:
            return source.grid.theta_basis.interpolate(
                values,
                self.grid.theta,
                axis=1,
            )

        radius = transfer_poloidal(transfer_axial(state.radius_coefficients))
        lam = transfer_poloidal(transfer_axial(state.lambda_coefficients))
        center = state.center_coefficients
        if center is not None:
            center = transfer_axial(center)
        transferred = SplineMirrorState(
            transfer_radial(radius),
            transfer_radial(lam),
            None if center is None else transfer_radial(center),
        )
        return self.project_fixed_boundary(transferred, boundary)


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

    if discretization.closed:
        raise ValueError("supplied-field initialization requires an open spline grid")
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
            state.center_coefficients,
        ),
        boundary,
    )
    return SuppliedFieldInitialization(initialized, axial_flux)


def initialize_closed_vacuum_stream_function(
    state: SplineMirrorState,
    discretization: SplineMirrorDiscretization,
    axis: Any,
    *,
    axial_flux_derivative: Array,
) -> SplineMirrorState:
    """Seed the minimum-energy closed field for a prescribed flux profile.

    The axial average of ``sqrt(g) / g_uu`` gives the poloidal flux-density
    variation for a field with constant covariant axial component. For
    concentric circular surfaces this is proportional to ``1/R`` on each flux
    surface. A globally current-free field also requires the supplied flux
    profile to keep that covariant component constant between surfaces. The
    returned stream function is only an initializer; nonaxisymmetric
    equilibria still solve all lambda coefficients.
    """

    if not discretization.closed:
        raise ValueError("the closed vacuum initializer requires a periodic spline discretization")
    from .geometry import evaluate_closed_geometry

    evaluated = discretization.evaluate_state(state)
    geometry = evaluate_closed_geometry(evaluated, discretization.grid, axis)
    if bool(geometry.jacobian_sign_changed):
        raise ValueError("the closed vacuum initializer requires a positive Jacobian")
    flux = jnp.asarray(axial_flux_derivative, dtype=evaluated.radius_scale.dtype)
    if flux.ndim == 0:
        flux = jnp.broadcast_to(flux, (discretization.grid.ns,))
    if flux.shape != (discretization.grid.ns,):
        raise ValueError("axial_flux_derivative must be scalar or have one value per radial surface")

    axial_weights = jnp.asarray(discretization.grid.axial_basis.weights)
    theta_weights = jnp.asarray(discretization.grid.theta_basis.weights)
    metric_weight = geometry.sqrt_g / geometry.g_xixi
    metric_weight = jnp.einsum("ijk,k->ij", metric_weight, axial_weights)
    metric_weight /= jnp.sum(axial_weights)
    theta_mean = jnp.einsum("ij,j->i", metric_weight, theta_weights)
    theta_mean /= jnp.sum(theta_weights)
    if not np.all(np.isfinite(np.asarray(theta_mean))) or np.any(np.asarray(theta_mean) <= 0.0):
        raise ValueError("the closed vacuum metric weight must be positive and finite")
    target_derivative = flux[:, None] * (metric_weight / theta_mean[:, None] - 1.0)

    lam = _integrate_poloidal_derivative(target_derivative)
    surface_mean = jnp.einsum("ij,j->i", lam, theta_weights)
    lam -= (surface_mean / jnp.sum(theta_weights))[:, None]
    coefficients = jnp.broadcast_to(
        lam[:, :, None],
        state.lambda_coefficients.shape,
    )
    coefficients = coefficients.at[0].set(coefficients[1])
    return SplineMirrorState(
        state.radius_coefficients,
        coefficients,
        state.center_coefficients,
    )


@dataclass(frozen=True)
class SplineMirrorSolveResult:
    """Converged coefficient state and its evaluated mirror result."""

    coefficient_state: SplineMirrorState
    evaluated: Any


@dataclass(frozen=True)
class ClosedFieldLine:
    """One traced closed-spline field line and its rotational transform."""

    axial_parameter: Array
    theta: Array
    iota: Array


def trace_closed_field_line(
    field: "ContravariantField",
    discretization: SplineMirrorDiscretization,
    *,
    radial_index: int,
    theta0: float = 0.0,
    turns: int = 1,
    steps_per_turn: int = 256,
) -> ClosedFieldLine:
    """Integrate ``dtheta/du = B^theta/B^u`` on a closed spline surface."""

    if not discretization.closed:
        raise ValueError("closed field-line tracing requires a periodic discretization")
    radial_index = int(radial_index)
    turns = int(turns)
    steps_per_turn = int(steps_per_turn)
    if not 0 <= radial_index < discretization.grid.ns:
        raise ValueError("radial_index is outside the spline grid")
    if turns < 1 or steps_per_turn < 4:
        raise ValueError("turns must be positive and steps_per_turn must be at least four")

    denominator = field.jac_b_xi[radial_index]
    tiny = jnp.finfo(denominator.dtype).tiny
    ratio = field.jac_b_theta[radial_index] / jnp.where(jnp.abs(denominator) > tiny, denominator, jnp.inf)
    recovery = jnp.asarray(discretization.grid.axial_basis.recovery_matrix)
    axial_coefficients = jnp.tensordot(ratio, recovery.T, axes=((-1,), (0,)))
    modes = jnp.asarray(np.fft.fftfreq(discretization.grid.ntheta, d=1.0 / discretization.grid.ntheta))
    start, stop = discretization.spline.domain
    period = float(stop - start)
    step = period / steps_per_turn

    def pitch(theta, axial_parameter):
        samples = discretization.spline.evaluate(axial_coefficients, axial_parameter)
        coefficients = jnp.fft.fft(samples) / discretization.grid.ntheta
        return jnp.real(jnp.sum(coefficients * jnp.exp(1j * modes * theta)))

    def advance(theta, index):
        axial_parameter = float(start) + step * (index % steps_per_turn)
        k1 = pitch(theta, axial_parameter)
        k2 = pitch(theta + 0.5 * step * k1, axial_parameter + 0.5 * step)
        k3 = pitch(theta + 0.5 * step * k2, axial_parameter + 0.5 * step)
        k4 = pitch(theta + step * k3, axial_parameter + step)
        updated = theta + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return updated, updated

    count = turns * steps_per_turn
    indices = jnp.arange(count)
    final_theta, traced = jax.lax.scan(advance, jnp.asarray(theta0), indices)
    theta = jnp.concatenate((jnp.asarray([theta0]), traced))
    axial = float(start) + step * jnp.arange(count + 1)
    iota = (final_theta - float(theta0)) / (2.0 * jnp.pi * turns)
    return ClosedFieldLine(axial_parameter=axial, theta=theta, iota=iota)


@dataclass(frozen=True)
class _SplineStateVectorizer:
    """Pack constrained spline coefficients into normalized solve variables."""

    base: SplineMirrorState
    evaluation_matrix: np.ndarray
    radius_indices: tuple[np.ndarray, np.ndarray, np.ndarray]
    radius_theta_basis: np.ndarray | None
    radius_scale: float
    center_scale: float
    flux_scale: float
    lambda_axial_indices: np.ndarray
    lambda_free_indices: np.ndarray
    lambda_pivot: int
    lambda_weights: np.ndarray
    lambda_fixed_weighted_sum: np.ndarray
    lambda_local_gauge: bool
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
        if discretization.closed:
            radius_mask[1:-1] = True
            lambda_axial_indices = np.arange(shape[2])
        else:
            radius_mask[1:-1, :, 1:-1] = True
            lambda_axial_indices = np.arange(1, shape[2] - 1)
        radius_indices = tuple(np.asarray(index) for index in np.nonzero(radius_mask))
        radius_theta_basis = None
        if discretization.closed and base.center_coefficients is not None:
            if discretization.grid.ntheta < 3:
                raise ValueError("closed center-map solves require mpol >= 1")
            theta = np.asarray(discretization.grid.theta)
            columns = [np.ones(theta.size) / np.sqrt(theta.size)]
            normalization = np.sqrt(2.0 / theta.size)
            for mode in range(2, discretization.grid.theta_basis.mpol + 1):
                columns.extend(
                    (
                        normalization * np.cos(mode * theta),
                        normalization * np.sin(mode * theta),
                    )
                )
            radius_theta_basis = np.column_stack(columns)
            coordinates = jnp.einsum(
                "ta,rtn->ran",
                jnp.asarray(radius_theta_basis),
                base.radius_coefficients,
            )
            centered_radius = jnp.einsum(
                "ta,ran->rtn",
                jnp.asarray(radius_theta_basis),
                coordinates,
            )
            base = SplineMirrorState(
                base.radius_coefficients.at[1:-1].set(centered_radius[1:-1]),
                base.lambda_coefficients,
                base.center_coefficients,
            )

        coefficient_weights = np.asarray(discretization.evaluation_matrix).T @ np.asarray(
            discretization.grid.axial_basis.weights
        )
        interior_weights = (
            np.asarray(discretization.grid.theta_basis.weights)[:, None]
            * coefficient_weights[None, lambda_axial_indices]
        ).reshape(-1)
        if solve_lambda and interior_weights.size < 2:
            raise ValueError("lambda solve requires at least two interior coefficients")
        local_gauge = bool(discretization.closed and solve_lambda)
        pivot = 0 if local_gauge else int(np.argmax(interior_weights)) if interior_weights.size else 0
        free_indices = np.delete(np.arange(interior_weights.size), pivot)
        endpoint_weights = np.zeros((shape[1], shape[2]))
        if not discretization.closed:
            endpoint_weights[:, [0, -1]] = (
                np.asarray(discretization.grid.theta_basis.weights)[:, None] * coefficient_weights[None, [0, -1]]
            )
        fixed_sum = np.einsum("jk,ijk->i", endpoint_weights, np.asarray(base.lambda_coefficients)[1:])
        if local_gauge:
            coefficients = jnp.asarray(base.lambda_coefficients)
            constants = coefficients[1:, 0, 0]
            constants = jnp.concatenate((constants[:1], constants))
            base = SplineMirrorState(
                base.radius_coefficients,
                coefficients - constants[:, None, None],
                base.center_coefficients,
            )
        return cls(
            base=base,
            evaluation_matrix=np.asarray(discretization.evaluation_matrix),
            radius_indices=radius_indices,
            radius_theta_basis=radius_theta_basis,
            radius_scale=radius_scale,
            center_scale=radius_scale,
            flux_scale=flux_scale,
            lambda_axial_indices=lambda_axial_indices,
            lambda_free_indices=free_indices,
            lambda_pivot=pivot,
            lambda_weights=interior_weights,
            lambda_fixed_weighted_sum=fixed_sum,
            lambda_local_gauge=local_gauge,
            solve_lambda=bool(solve_lambda),
        )

    @property
    def radius_size(self) -> int:
        if self.radius_theta_basis is not None:
            shape = self.base.radius_coefficients.shape
            return int((shape[0] - 2) * self.radius_theta_basis.shape[1] * shape[2])
        return int(self.radius_indices[0].size)

    @property
    def radius_poloidal_nodes(self) -> int:
        """Return the active radial-shape coordinates per axial coefficient."""

        if self.radius_theta_basis is not None:
            return int(self.radius_theta_basis.shape[1])
        return int(self.base.radius_coefficients.shape[1])

    @property
    def lambda_size(self) -> int:
        """Return the number of gauge-free stream-function coefficients."""

        if not self.solve_lambda:
            return 0
        return int((self.base.radius_coefficients.shape[0] - 1) * self.lambda_free_indices.size)

    @property
    def center_size(self) -> int:
        """Return the number of active transverse center coefficients."""

        if self.base.center_coefficients is None:
            return 0
        shape = self.base.center_coefficients.shape
        return int((shape[0] - 1) * shape[1] * shape[2])

    @property
    def block_slices(self) -> tuple[slice, ...]:
        """Return packed radius, center, and lambda blocks that are present."""

        offset = self.radius_size
        blocks = [slice(0, offset)]
        if self.center_size:
            blocks.append(slice(offset, offset + self.center_size))
            offset += self.center_size
        if self.lambda_size:
            blocks.append(slice(offset, offset + self.lambda_size))
        return tuple(blocks)

    def pack(self) -> np.ndarray:
        """Pack the projected coefficient state."""

        if self.radius_theta_basis is None:
            radius = np.asarray(self.base.radius_coefficients)[self.radius_indices]
        else:
            radius = np.einsum(
                "ta,rtn->ran",
                self.radius_theta_basis,
                np.asarray(self.base.radius_coefficients)[1:-1],
            ).reshape(-1)
        radius = radius / self.radius_scale
        blocks = [radius]
        if self.center_size:
            center = np.asarray(self.base.center_coefficients)[:-1].reshape(-1) / self.center_scale
            blocks.append(center)
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
        if self.radius_theta_basis is None:
            radius = self.base.radius_coefficients.at[self.radius_indices].set(
                vector[: self.radius_size] * self.radius_scale
            )
        else:
            shape = self.base.radius_coefficients.shape
            coordinates = vector[: self.radius_size].reshape(
                shape[0] - 2,
                self.radius_theta_basis.shape[1],
                shape[2],
            )
            interior = jnp.einsum(
                "ta,ran->rtn",
                jnp.asarray(self.radius_theta_basis),
                coordinates * self.radius_scale,
            )
            radius = self.base.radius_coefficients.at[1:-1].set(interior)
        radius = _regularize_axis_radius(radius)
        offset = self.radius_size
        center = self.base.center_coefficients
        if self.center_size:
            center_shape = center.shape
            center = center.at[:-1].set(
                vector[offset : offset + self.center_size].reshape(center_shape[0] - 1, *center_shape[1:])
                * self.center_scale
            )
            offset += self.center_size
        if not self.solve_lambda:
            return SplineMirrorState(radius, self.base.lambda_coefficients, center)

        shape = self.base.lambda_coefficients.shape
        free = vector[offset:].reshape(shape[0] - 1, self.lambda_free_indices.size) * self.flux_scale
        interior = self.base.lambda_coefficients[1:, :, self.lambda_axial_indices].reshape(shape[0] - 1, -1)
        interior = interior.at[:, jnp.asarray(self.lambda_free_indices)].set(free)
        if not self.lambda_local_gauge:
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
        return SplineMirrorState(radius, lam.at[0].set(lam[1]), center)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative normalized coefficient bounds."""

        radius_lower = np.full(self.radius_size, 0.2)
        radius_upper = np.full(self.radius_size, 5.0)
        if self.radius_theta_basis is not None:
            shape = (
                self.base.radius_coefficients.shape[0] - 2,
                self.radius_theta_basis.shape[1],
                self.base.radius_coefficients.shape[2],
            )
            radius_lower = np.full(shape, -2.0)
            radius_upper = np.full(shape, 2.0)
            radius_lower[:, 0] = 0.2
            radius_upper[:, 0] = 5.0
            radius_lower = radius_lower.reshape(-1)
            radius_upper = radius_upper.reshape(-1)
        lower = np.concatenate(
            (
                radius_lower,
                np.full(self.center_size, -2.0),
                np.full(self.lambda_size, -np.inf),
            )
        )
        upper = np.concatenate(
            (
                radius_upper,
                np.full(self.center_size, 2.0),
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
        if self.radius_theta_basis is None:
            radius = radius_coefficients[self.radius_indices]
        else:
            radius = np.einsum(
                "ta,rtn->ran",
                self.radius_theta_basis,
                radius_coefficients[1:-1],
            ).reshape(-1)
        radius = radius * self.radius_scale
        blocks = [radius]
        if self.center_size:
            if gradient.center_shift is None:
                raise ValueError("closed evaluated gradient is missing its center map")
            center_coefficients = np.tensordot(
                np.asarray(gradient.center_shift),
                matrix,
                axes=((-1,), (0,)),
            )
            blocks.append((center_coefficients[:-1] * self.center_scale).reshape(-1))
        if not self.solve_lambda:
            return np.concatenate(blocks)

        lambda_coefficients = np.tensordot(np.asarray(gradient.lambda_stream), matrix, axes=((-1,), (0,)))
        lambda_coefficients[1] += lambda_coefficients[0]
        interior = lambda_coefficients[1:, :, self.lambda_axial_indices].reshape(
            lambda_coefficients.shape[0] - 1,
            -1,
        )
        if self.lambda_local_gauge:
            free = interior[:, self.lambda_free_indices]
            blocks.append((free * self.flux_scale).reshape(-1))
            return np.concatenate(blocks)
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
    if vectorizer.radius_theta_basis is None:
        radial = np.asarray(vectorizer.radius_indices[0], dtype=int)
        axial = np.asarray(vectorizer.radius_indices[2], dtype=int)
    else:
        poloidal = vectorizer.radius_theta_basis.shape[1]
        coefficient_count = discretization.coefficient_count
        radial = np.repeat(
            np.arange(1, discretization.grid.ns - 1),
            poloidal * coefficient_count,
        )
        axial = np.tile(
            np.tile(np.arange(coefficient_count), poloidal),
            discretization.grid.ns - 2,
        )
    if vectorizer.center_size:
        coefficient_count = discretization.coefficient_count
        center_channels = np.tile(
            np.repeat(np.arange(1, 3), coefficient_count),
            discretization.grid.ns - 1,
        )
        channels = np.concatenate((channels, center_channels))
        radial = np.concatenate(
            (
                radial,
                np.repeat(
                    np.arange(discretization.grid.ns - 1),
                    2 * coefficient_count,
                ),
            )
        )
        axial = np.concatenate(
            (
                axial,
                np.tile(
                    np.tile(np.arange(coefficient_count), 2),
                    discretization.grid.ns - 1,
                ),
            )
        )
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


def _closed_hessian_supports(
    discretization: SplineMirrorDiscretization,
    vectorizer: _SplineStateVectorizer,
) -> list[np.ndarray]:
    """Build the structural rows of the local closed-spline Hessian model."""

    channels, radial, axial = _packed_spline_layout(discretization, vectorizer)
    nodes = discretization.grid.axial_basis.nodes
    values = np.asarray(discretization.spline.basis_matrix(nodes))
    derivatives = np.asarray(discretization.spline.basis_matrix(nodes, derivative=1))
    scale = max(float(np.max(np.abs(derivatives))), 1.0)
    active = (np.abs(values) > 1.0e-13) | (np.abs(derivatives) > 1.0e-13 * scale)
    overlap = active.T.astype(np.int16) @ active.astype(np.int16) > 0
    supports = []
    for column in range(radial.size):
        radial_neighbors = np.abs(radial - radial[column]) <= 1

        # Axis regularization obtains lambda(0) from the first radius-shape
        # surface and uses it throughout the radial stream interpolation.
        axis_radius = (channels == 0) & (radial == 1)
        if channels[column] == 0 and radial[column] == 1:
            radial_neighbors[:] = True
        else:
            radial_neighbors |= axis_radius

        # The first half cell uses two off-axis stream surfaces. Its center
        # variables therefore span one more radial level than interior cells.
        if radial[column] == 0:
            radial_neighbors |= radial <= 2
        radial_neighbors |= (radial == 0) & (radial[column] <= 2)
        supports.append(
            np.flatnonzero(radial_neighbors & overlap[axial, axial[column]])
        )
    return supports


def _disjoint_support_groups(supports: list[np.ndarray], size: int) -> list[np.ndarray]:
    """Greedily group columns whose retained response rows cannot overlap."""

    occupied: list[np.ndarray] = []
    groups: list[list[int]] = []
    for column, rows in enumerate(supports):
        for color, used in enumerate(occupied):
            if not np.any(used[rows]):
                used[rows] = True
                groups[color].append(column)
                break
        else:
            used = np.zeros(size, dtype=bool)
            used[rows] = True
            occupied.append(used)
            groups.append([column])
    return [np.asarray(group, dtype=int) for group in groups]


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
    active_derivative = derivative if discretization.closed else derivative[:, 1:-1]
    stiffness = active_derivative.T @ (weights[:, None] * active_derivative)
    geometry = SeparableMirrorPreconditioner.build_from_axial_stiffness(
        discretization.grid,
        stiffness,
        poloidal_nodes=vectorizer.radius_poloidal_nodes,
    )
    center = None
    if vectorizer.center_size:
        center = SeparableMirrorPreconditioner.build_from_axial_stiffness(
            discretization.grid,
            stiffness,
            radial_nodes=discretization.grid.ns - 1,
            poloidal_nodes=1,
        )
    stream = None
    if vectorizer.lambda_size:
        stream = SeparableMirrorPreconditioner.build_from_axial_stiffness(
            discretization.grid,
            stiffness,
            radial_nodes=discretization.grid.ns - 1,
        )
    scales = np.ones(len(vectorizer.block_slices))
    center_start = vectorizer.radius_size
    lambda_start = center_start + vectorizer.center_size

    def apply(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        result = np.array(vector, copy=True)
        result[: vectorizer.radius_size] = geometry.apply(vector[: vectorizer.radius_size]) * scales[0]
        block = 1
        if center is not None:
            center_values = vector[center_start:lambda_start].reshape(
                discretization.grid.ns - 1,
                2,
                discretization.coefficient_count,
            )
            solved_center = np.empty_like(center_values)
            for component in range(2):
                solved_center[:, component] = center.apply(
                    center_values[:, component : component + 1].reshape(-1)
                ).reshape(discretization.grid.ns - 1, discretization.coefficient_count)
            result[center_start:lambda_start] = solved_center.reshape(-1) * scales[block]
            block += 1
        if stream is not None:
            reduced = vector[lambda_start:]
            if vectorizer.lambda_local_gauge:
                lifted = np.zeros((stream.active_shape[0], vectorizer.lambda_weights.size))
                lifted[:, vectorizer.lambda_free_indices] = reduced.reshape(
                    stream.active_shape[0], vectorizer.lambda_free_indices.size
                )
                solved = stream.apply(lifted.reshape(-1)).reshape(lifted.shape)
                reduced = solved[:, vectorizer.lambda_free_indices].reshape(reduced.shape)
            else:
                reduced = stream.apply_gauge_free(
                    reduced,
                    free_indices=vectorizer.lambda_free_indices,
                    pivot=vectorizer.lambda_pivot,
                    weights=vectorizer.lambda_weights,
                )
            result[lambda_start:] = reduced * scales[block]
        return result

    def build_local(matrix_columns: Callable[[np.ndarray], np.ndarray]) -> Any:
        """Factor a frozen sparse Hessian from chunked matrix-free columns."""

        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import splu

        size = vectorizer.radius_size + vectorizer.center_size + vectorizer.lambda_size
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        chunk_size = min(32, size)
        if discretization.closed:
            supports = _closed_hessian_supports(discretization, vectorizer)
            groups = _disjoint_support_groups(supports, size)
            for start in range(0, len(groups), chunk_size):
                batch = groups[start : start + chunk_size]
                directions = np.zeros((len(batch), size))
                for local_index, columns in enumerate(batch):
                    directions[local_index, columns] = 1.0
                responses = np.asarray(matrix_columns(directions), dtype=float)
                for local_index, columns in enumerate(batch):
                    for column in columns:
                        rows = supports[column]
                        row_parts.append(rows)
                        column_parts.append(np.full(rows.size, column, dtype=int))
                        value_parts.append(responses[local_index, rows])
        else:
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

        solve.hessian_probe_count = len(groups) if discretization.closed else size  # type: ignore[attr-defined]
        solve.hessian_column_count = size  # type: ignore[attr-defined]
        solve.rebuild_each_step = bool(  # type: ignore[attr-defined]
            discretization.closed and not vectorizer.center_size
        )
        return solve

    local_builder = build_local if vectorizer.lambda_size else None
    if local_builder is not None:
        local_builder.reuse_linearization = discretization.closed  # type: ignore[attr-defined]
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
    axis: Any | None = None,
    gradient_tolerance: float = 1.0e-11,
    require_convergence: bool = False,
) -> SplineMirrorSolveResult:
    """Solve an open or closed scalar-pressure spline fixed boundary.

    A closed discretization requires its evaluated periodic ``axis``. Open
    mirrors retain fixed end cuts and reject that argument.
    """

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
    if discretization.closed != (axis is not None):
        raise ValueError("closed spline discretizations require an axis; open ones do not")
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
        return mirror_energy(state, grid, axis=axis, **energy_kwargs)

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
        center_start = vectorizer.radius_size
        lambda_start = center_start + vectorizer.center_size
        center = packed[center_start:lambda_start]
        lam = packed[lambda_start:]
        center_rms = np.sqrt(np.mean(center**2)) if center.size else 0.0
        lambda_rms = np.sqrt(np.mean(lam**2)) if lam.size else 0.0
        return VariationalResidual(
            radius_gradient=jnp.asarray(radius),
            center_gradient=jnp.asarray(center),
            lambda_gradient=jnp.asarray(lam),
            radius_rms=jnp.asarray(np.sqrt(np.mean(radius**2))),
            center_rms=jnp.asarray(center_rms),
            lambda_rms=jnp.asarray(lambda_rms),
            maximum=jnp.asarray(np.max(np.abs(packed))),
        )

    def packed_weak(state: MirrorState) -> VariationalResidual:
        gradient = isotropic_staggered_energy_gradient(
            state,
            grid,
            axis=axis,
            **energy_kwargs,
        )
        packed = vectorizer.pullback_evaluated_gradient(gradient) / energy_scale
        radius = packed[: vectorizer.radius_size]
        center_start = vectorizer.radius_size
        lambda_start = center_start + vectorizer.center_size
        center = packed[center_start:lambda_start]
        lam = packed[lambda_start:]
        return VariationalResidual(
            radius_gradient=jnp.asarray(radius),
            center_gradient=jnp.asarray(center),
            lambda_gradient=jnp.asarray(lam),
            radius_rms=jnp.asarray(np.sqrt(np.mean(radius**2))),
            center_rms=jnp.asarray(np.sqrt(np.mean(center**2)) if center.size else 0.0),
            lambda_rms=jnp.asarray(np.sqrt(np.mean(lam**2)) if lam.size else 0.0),
            maximum=jnp.asarray(np.max(np.abs(packed))),
        )

    def force_residual(state, energy):
        return isotropic_force_residual(
            energy,
            grid,
            state=state,
            axis=axis,
            closed=discretization.closed,
            characteristic_length=None if axis is None else axis.arc_length,
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
            start_with_newton=(
                discretization.closed and not vectorizer.center_size and x0.size <= 512
            ),
            start_with_dense_root=bool(vectorizer.center_size and x0.size <= 2048),
            matrix_free_context=(
                vectorizer,
                _packed_spline_preconditioner(discretization, vectorizer),
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
    data_fields=["radius_coefficients", "lambda_coefficients", "center_coefficients"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    SplineMirrorSolveResult,
    data_fields=["coefficient_state", "evaluated"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    ClosedFieldLine,
    data_fields=["axial_parameter", "theta", "iota"],
    meta_fields=[],
)


__all__ = [
    "CubicBSplineBasis",
    "ClosedFieldLine",
    "SplineMirrorBoundary",
    "SplineMirrorDiscretization",
    "SplineMirrorSolveResult",
    "SplineMirrorState",
    "solve_fixed_boundary_cli",
    "trace_closed_field_line",
]


if TYPE_CHECKING:
    from .geometry import ContravariantField

"""Small JAX cubic B-spline bases for mirror geometry coefficients.

The open basis is clamped at both end cuts. The periodic basis folds uniform
cardinal splines onto a closed interval. Knot locations are static NumPy data;
coefficient evaluation and transfer are differentiable JAX operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .basis import MirrorGrid, ThetaBasis
from .model import MirrorBoundary, MirrorConfig, MirrorResolution, MirrorState

Array = Any
_DEGREE = 3


def _validate_breakpoints(breakpoints: Array) -> np.ndarray:
    values = np.asarray(breakpoints, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("breakpoints must be a one-dimensional array of length >= 2")
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
        raise ValueError("breakpoints must be finite and strictly increasing")
    return values


def _basis_levels(knots: Array, points: Array, degree: int) -> list[Array]:
    """Return Cox-de Boor basis levels from degree zero through ``degree``."""

    knots = jnp.asarray(knots)
    points = jnp.asarray(points).reshape(-1)
    evaluation_points = jnp.where(points == knots[-1], jnp.nextafter(knots[-1], -jnp.inf), points)
    level = ((evaluation_points[:, None] >= knots[:-1]) & (evaluation_points[:, None] < knots[1:])).astype(points.dtype)
    levels = [level]
    for order in range(1, degree + 1):
        count = knots.size - order - 1
        left_denominator = knots[order : order + count] - knots[:count]
        right_denominator = knots[order + 1 : order + count + 1] - knots[1 : count + 1]
        left = jnp.where(
            left_denominator > 0.0,
            (evaluation_points[:, None] - knots[:count]) / jnp.where(left_denominator > 0.0, left_denominator, 1.0),
            0.0,
        )
        right = jnp.where(
            right_denominator > 0.0,
            (knots[order + 1 : order + count + 1] - evaluation_points[:, None])
            / jnp.where(right_denominator > 0.0, right_denominator, 1.0),
            0.0,
        )
        level = left * level[:, :count] + right * level[:, 1 : count + 1]
        levels.append(level)
    if degree > 0:
        endpoint = points == knots[-1]
        levels[-1] = levels[-1].at[:, -1].set(jnp.where(endpoint, 1.0, levels[-1][:, -1]))
        levels[-1] = levels[-1].at[:, :-1].set(jnp.where(endpoint[:, None], 0.0, levels[-1][:, :-1]))
    return levels


def _basis_matrix(knots: Array, points: Array, degree: int, derivative: int = 0) -> Array:
    levels = _basis_levels(knots, points, degree)
    if derivative == 0:
        return levels[degree]
    count = jnp.asarray(knots).size - degree - 1
    knots = jnp.asarray(knots)
    lower = levels[degree - 1]
    left_denominator = knots[degree : degree + count] - knots[:count]
    right_denominator = knots[degree + 1 : degree + count + 1] - knots[1 : count + 1]
    left_scale = jnp.where(left_denominator > 0.0, degree / left_denominator, 0.0)
    right_scale = jnp.where(right_denominator > 0.0, degree / right_denominator, 0.0)
    first = left_scale * lower[:, :count] - right_scale * lower[:, 1 : count + 1]
    if derivative == 1:
        return first
    if derivative != 2:
        raise ValueError("only derivatives 0, 1, and 2 are supported")
    lower_count = count + 1
    degree_minus_one = degree - 1
    base = levels[degree - 2]
    lower_left_denominator = knots[degree_minus_one : degree_minus_one + lower_count] - knots[:lower_count]
    lower_right_denominator = (
        knots[degree_minus_one + 1 : degree_minus_one + lower_count + 1] - knots[1 : lower_count + 1]
    )
    lower_first = (
        jnp.where(lower_left_denominator > 0.0, degree_minus_one / lower_left_denominator, 0.0) * base[:, :lower_count]
        - jnp.where(lower_right_denominator > 0.0, degree_minus_one / lower_right_denominator, 0.0)
        * base[:, 1 : lower_count + 1]
    )
    return left_scale * lower_first[:, :count] - right_scale * lower_first[:, 1 : count + 1]


def _span_quadrature(breakpoints: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 1:
        raise ValueError("quadrature order must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    centers = 0.5 * (breakpoints[:-1] + breakpoints[1:])
    scales = 0.5 * np.diff(breakpoints)
    return (
        (centers[:, None] + scales[:, None] * nodes[None, :]).reshape(-1),
        (scales[:, None] * weights[None, :]).reshape(-1),
    )


@dataclass(frozen=True, eq=False)
class CubicBSplineBasis:
    """Static clamped or periodic cubic basis with JAX evaluation."""

    knots: np.ndarray
    breakpoints: np.ndarray
    periodic: bool
    size: int
    collocation_nodes: np.ndarray
    quadrature_nodes: np.ndarray
    quadrature_weights: np.ndarray

    @classmethod
    def clamped(cls, breakpoints: Array, *, quadrature_order: int = 4) -> "CubicBSplineBasis":
        """Build an open cubic basis with repeated endpoint knots."""

        breaks = _validate_breakpoints(breakpoints)
        knots = np.concatenate((np.repeat(breaks[0], _DEGREE + 1), breaks[1:-1], np.repeat(breaks[-1], _DEGREE + 1)))
        size = knots.size - _DEGREE - 1
        collocation = np.asarray([np.mean(knots[index + 1 : index + _DEGREE + 1]) for index in range(size)])
        quadrature_nodes, quadrature_weights = _span_quadrature(breaks, quadrature_order)
        return cls(knots, breaks, False, size, collocation, quadrature_nodes, quadrature_weights)

    @classmethod
    def periodic_uniform(
        cls,
        size: int,
        domain: tuple[float, float] = (0.0, 2.0 * np.pi),
        *,
        quadrature_order: int = 4,
    ) -> "CubicBSplineBasis":
        """Build ``size`` folded uniform cubic splines on a periodic interval."""

        size = int(size)
        start, stop = map(float, domain)
        if size < _DEGREE + 1:
            raise ValueError("periodic cubic basis requires size >= 4")
        if not stop > start:
            raise ValueError("periodic domain must have stop > start")
        spacing = (stop - start) / size
        knots = start + spacing * np.arange(-_DEGREE, size + _DEGREE + 1)
        breaks = np.linspace(start, stop, size + 1)
        collocation = breaks[:-1]
        quadrature_nodes, quadrature_weights = _span_quadrature(breaks, quadrature_order)
        return cls(knots, breaks, True, size, collocation, quadrature_nodes, quadrature_weights)

    @property
    def domain(self) -> tuple[float, float]:
        """Return the open or fundamental periodic interval."""

        return float(self.breakpoints[0]), float(self.breakpoints[-1])

    def basis_matrix(self, points: Array, *, derivative: int = 0) -> Array:
        """Evaluate basis values or derivatives at one-dimensional ``points``."""

        points = jnp.asarray(points)
        original_shape = points.shape
        evaluation_points = points.reshape(-1)
        if self.periodic:
            start, stop = self.domain
            period = stop - start
            evaluation_points = jnp.mod(evaluation_points - start, period) + start
            raw = _basis_matrix(self.knots, evaluation_points, _DEGREE, derivative)
            folded = jnp.zeros((evaluation_points.size, self.size), dtype=raw.dtype)
            for column in range(raw.shape[1]):
                folded = folded.at[:, (column - _DEGREE) % self.size].add(raw[:, column])
            matrix = folded
        else:
            matrix = _basis_matrix(self.knots, evaluation_points, _DEGREE, derivative)
        return matrix.reshape(original_shape + (self.size,))

    def evaluate(self, coefficients: Array, points: Array, *, derivative: int = 0, axis: int = -1) -> Array:
        """Evaluate spline coefficients along ``axis`` at arbitrary points."""

        coefficients = jnp.asarray(coefficients)
        if coefficients.shape[axis] != self.size:
            raise ValueError(f"coefficient axis has size {coefficients.shape[axis]}; expected {self.size}")
        moved = jnp.moveaxis(coefficients, axis, -1)
        values = jnp.tensordot(moved, self.basis_matrix(points, derivative=derivative), axes=((-1,), (-1,)))
        point_axes = tuple(range(values.ndim - jnp.ndim(points), values.ndim))
        target = tuple(range(axis % coefficients.ndim, axis % coefficients.ndim + len(point_axes)))
        return jnp.moveaxis(values, point_axes, target) if point_axes else values

    def fit(self, values: Array, *, nodes: Array | None = None, axis: int = -1) -> Array:
        """Interpolate nodal values to coefficients with a square collocation solve."""

        sample_nodes = self.collocation_nodes if nodes is None else jnp.asarray(nodes)
        matrix = self.basis_matrix(sample_nodes)
        if matrix.shape[0] != self.size:
            raise ValueError("fit requires exactly one independent sample per coefficient")
        values = jnp.asarray(values)
        moved = jnp.moveaxis(values, axis, 0)
        coefficients = jnp.linalg.solve(matrix, moved.reshape((self.size, -1)))
        coefficients = coefficients.reshape((self.size,) + moved.shape[1:])
        return jnp.moveaxis(coefficients, 0, axis)

    def integrate(self, coefficients: Array, *, axis: int = -1) -> Array:
        """Integrate a spline using per-span Gauss-Legendre quadrature."""

        values = self.evaluate(coefficients, self.quadrature_nodes, axis=axis)
        return jnp.tensordot(values, jnp.asarray(self.quadrature_weights), axes=((axis,), (0,)))

    def insert_knot(self, coefficients: Array, knot: float, *, axis: int = -1) -> tuple["CubicBSplineBasis", Array]:
        """Insert one open knot exactly with the Boehm coefficient update."""

        if self.periodic:
            raise ValueError("periodic knot insertion is not supported; refine the uniform basis")
        knot = float(knot)
        start, stop = self.domain
        if not start < knot < stop:
            raise ValueError("inserted knot must lie strictly inside the domain")
        if np.any(np.isclose(self.breakpoints, knot, rtol=0.0, atol=1.0e-14)):
            raise ValueError("inserted knot must be new; repeated-knot refinement is unsupported")
        span = int(np.searchsorted(self.knots, knot, side="right") - 1)
        multiplicity = int(np.count_nonzero(np.isclose(self.knots, knot, rtol=0.0, atol=1.0e-14)))
        values = jnp.moveaxis(jnp.asarray(coefficients), axis, 0)
        if values.shape[0] != self.size:
            raise ValueError(f"coefficient axis has size {values.shape[0]}; expected {self.size}")
        updated = jnp.zeros((self.size + 1,) + values.shape[1:], dtype=values.dtype)
        updated = updated.at[: span - _DEGREE + 1].set(values[: span - _DEGREE + 1])
        updated = updated.at[span - multiplicity + 1 :].set(values[span - multiplicity :])
        for index in range(span - _DEGREE + 1, span - multiplicity + 1):
            alpha = (knot - self.knots[index]) / (self.knots[index + _DEGREE] - self.knots[index])
            updated = updated.at[index].set(alpha * values[index] + (1.0 - alpha) * values[index - 1])
        new_breakpoints = np.sort(np.append(self.breakpoints, knot))
        refined = CubicBSplineBasis.clamped(
            new_breakpoints, quadrature_order=self.quadrature_weights.size // (self.breakpoints.size - 1)
        )
        return refined, jnp.moveaxis(updated, 0, axis)


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
    """Geometry and stream-function B-spline coefficients."""

    radius_coefficients: Array
    lambda_coefficients: Array


@dataclass(frozen=True, eq=False)
class SplineMirrorDiscretization:
    """Coefficient-to-quadrature map for a fixed mirror configuration."""

    spline: CubicBSplineBasis
    grid: MirrorGrid
    evaluation_matrix: np.ndarray
    closed: bool = False

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
        resolution = config.resolution
        s = np.linspace(0.0, 1.0, resolution.ns)
        ds = 1.0 / (resolution.ns - 1)
        radial_weights = np.full(resolution.ns, ds)
        radial_weights[[0, -1]] *= 0.5
        z_mid = 0.5 * (config.z_min + config.z_max)
        dz_dxi = 0.5 * (config.z_max - config.z_min)
        grid = MirrorGrid(
            s=s,
            s_half=0.5 * (s[:-1] + s[1:]),
            radial_weights=radial_weights,
            theta_basis=ThetaBasis.build(resolution.ntheta, resolution.mpol),
            axial_basis=axial,
            z=z_mid + dz_dxi * axial.nodes,
            dz_dxi=dz_dxi,
        )
        return cls(spline, grid, np.asarray(spline.basis_matrix(axial.nodes)), False)

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
        s = np.linspace(0.0, 1.0, resolution.ns)
        ds = 1.0 / (resolution.ns - 1)
        radial_weights = np.full(resolution.ns, ds)
        radial_weights[[0, -1]] *= 0.5
        grid = MirrorGrid(
            s=s,
            s_half=0.5 * (s[:-1] + s[1:]),
            radial_weights=radial_weights,
            theta_basis=ThetaBasis.build(resolution.ntheta, resolution.mpol),
            axial_basis=axial,
            z=np.asarray(axial.nodes),
            dz_dxi=1.0,
        )
        return cls(spline, grid, np.asarray(spline.basis_matrix(axial.nodes)), True)

    @property
    def coefficient_count(self) -> int:
        """Return the number of axial coefficients per scalar field."""

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
        return SplineMirrorState(self.spline.fit(radius, axis=-1), self.spline.fit(lam, axis=-1))

    def project_fixed_boundary(
        self,
        state: SplineMirrorState,
        boundary: SplineMirrorBoundary,
    ) -> SplineMirrorState:
        """Apply side/end geometry, axis regularity, and the lambda gauge in coefficient space."""

        radius = jnp.asarray(state.radius_coefficients)
        boundary_radius = jnp.asarray(boundary.radius_coefficients)
        radius = radius.at[-1].set(boundary_radius)
        if not self.closed:
            radius = radius.at[:, :, 0].set(boundary_radius[:, 0][None, :])
            radius = radius.at[:, :, -1].set(boundary_radius[:, -1][None, :])
        radius = radius.at[0].set(radius[1])
        lam = jnp.asarray(state.lambda_coefficients).at[0].set(state.lambda_coefficients[1])
        evaluated = jnp.tensordot(lam, jnp.asarray(self.evaluation_matrix).T, axes=((-1,), (0,)))
        theta_weights = jnp.asarray(self.grid.theta_basis.weights)
        axial_weights = jnp.asarray(self.grid.axial_basis.weights)
        mean = jnp.einsum("j,k,ijk->i", theta_weights, axial_weights, evaluated)
        mean /= jnp.sum(theta_weights) * jnp.sum(axial_weights)
        return SplineMirrorState(radius, lam - mean[:, None, None])

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
        return self.project_fixed_boundary(SplineMirrorState(transferred, state.lambda_coefficients), target)


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
        radius_indices = tuple(np.asarray(index) for index in np.nonzero(radius_mask))

        coefficient_weights = np.asarray(discretization.evaluation_matrix).T @ np.asarray(
            discretization.grid.axial_basis.weights
        )
        interior_weights = (
            np.asarray(discretization.grid.theta_basis.weights)[:, None] * coefficient_weights[None, 1:-1]
        ).reshape(-1)
        if solve_lambda and interior_weights.size < 2:
            raise ValueError("lambda solve requires at least two interior coefficients")
        pivot = int(np.argmax(interior_weights)) if interior_weights.size else 0
        free_indices = np.delete(np.arange(interior_weights.size), pivot)
        endpoint_weights = np.zeros((shape[1], shape[2]))
        endpoint_weights[:, [0, -1]] = (
            np.asarray(discretization.grid.theta_basis.weights)[:, None] * coefficient_weights[None, [0, -1]]
        )
        fixed_sum = np.einsum("jk,ijk->i", endpoint_weights, np.asarray(base.lambda_coefficients)[1:])
        return cls(
            base=base,
            evaluation_matrix=np.asarray(discretization.evaluation_matrix),
            radius_indices=radius_indices,
            radius_scale=radius_scale,
            flux_scale=flux_scale,
            lambda_free_indices=free_indices,
            lambda_pivot=pivot,
            lambda_weights=interior_weights,
            lambda_fixed_weighted_sum=fixed_sum,
            solve_lambda=bool(solve_lambda),
        )

    @property
    def radius_size(self) -> int:
        """Return the number of active geometry coefficients."""

        return int(self.radius_indices[0].size)

    @property
    def lambda_size(self) -> int:
        """Return the number of gauge-free stream-function coefficients."""

        if not self.solve_lambda:
            return 0
        return int((self.base.radius_coefficients.shape[0] - 1) * self.lambda_free_indices.size)

    def pack(self) -> np.ndarray:
        """Pack the projected coefficient state."""

        radius = np.asarray(self.base.radius_coefficients)[self.radius_indices] / self.radius_scale
        if not self.solve_lambda:
            return radius
        interior = np.asarray(self.base.lambda_coefficients)[1:, :, 1:-1].reshape(
            self.base.radius_coefficients.shape[0] - 1, -1
        )
        lam = interior[:, self.lambda_free_indices].reshape(-1) / self.flux_scale
        return np.concatenate((radius, lam))

    def unpack(self, vector: Array) -> SplineMirrorState:
        """Reconstruct constrained coefficients from normalized variables."""

        vector = jnp.asarray(vector)
        radius = self.base.radius_coefficients.at[self.radius_indices].set(
            vector[: self.radius_size] * self.radius_scale
        )
        radius = radius.at[0].set(radius[1])
        if not self.solve_lambda:
            return SplineMirrorState(radius, self.base.lambda_coefficients)

        shape = self.base.lambda_coefficients.shape
        free = vector[self.radius_size :].reshape(shape[0] - 1, self.lambda_free_indices.size) * self.flux_scale
        interior = self.base.lambda_coefficients[1:, :, 1:-1].reshape(shape[0] - 1, -1)
        interior = interior.at[:, jnp.asarray(self.lambda_free_indices)].set(free)
        weighted_free = jnp.sum(
            free * jnp.asarray(self.lambda_weights[self.lambda_free_indices])[None, :],
            axis=1,
        )
        pivot_value = -(jnp.asarray(self.lambda_fixed_weighted_sum) + weighted_free) / float(
            self.lambda_weights[self.lambda_pivot]
        )
        interior = interior.at[:, self.lambda_pivot].set(pivot_value)
        lam = self.base.lambda_coefficients.at[1:, :, 1:-1].set(interior.reshape(shape[0] - 1, shape[1], shape[2] - 2))
        return SplineMirrorState(radius, lam.at[0].set(lam[1]))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return conservative normalized coefficient bounds."""

        lower = np.concatenate((np.full(self.radius_size, 0.2), np.full(self.lambda_size, -np.inf)))
        upper = np.concatenate((np.full(self.radius_size, 5.0), np.full(self.lambda_size, np.inf)))
        return lower, upper

    def pullback_evaluated_gradient(self, gradient: MirrorState) -> np.ndarray:
        """Pull an evaluated-state gradient to the active coefficients."""

        matrix = np.asarray(self.evaluation_matrix)
        radius_coefficients = np.tensordot(np.asarray(gradient.radius_scale), matrix, axes=((-1,), (0,)))
        radius_coefficients[1] += radius_coefficients[0]
        radius = radius_coefficients[self.radius_indices] * self.radius_scale
        if not self.solve_lambda:
            return radius

        lambda_coefficients = np.tensordot(np.asarray(gradient.lambda_stream), matrix, axes=((-1,), (0,)))
        lambda_coefficients[1] += lambda_coefficients[0]
        interior = lambda_coefficients[1:, :, 1:-1].reshape(lambda_coefficients.shape[0] - 1, -1)
        pivot_gradient = interior[:, self.lambda_pivot]
        free = interior[:, self.lambda_free_indices] - (
            pivot_gradient[:, None]
            * self.lambda_weights[self.lambda_free_indices][None, :]
            / self.lambda_weights[self.lambda_pivot]
        )
        return np.concatenate((radius, (free * self.flux_scale).reshape(-1)))


def _packed_spline_preconditioner(
    discretization: SplineMirrorDiscretization,
    vectorizer: _SplineStateVectorizer,
) -> tuple[Any, np.ndarray]:
    """Build the existing tensor preconditioner on spline coefficients."""

    from .solver import SeparableMirrorPreconditioner

    derivative = np.asarray(
        discretization.spline.basis_matrix(discretization.grid.axial_basis.nodes, derivative=1)
    ) / float(discretization.grid.dz_dxi)
    weights = np.asarray(discretization.grid.axial_basis.weights)
    interior = derivative[:, 1:-1]
    stiffness = interior.T @ (weights[:, None] * interior)
    geometry = SeparableMirrorPreconditioner.build_from_axial_stiffness(discretization.grid, stiffness)
    stream = None
    if vectorizer.lambda_size:
        stream = SeparableMirrorPreconditioner.build_from_axial_stiffness(
            discretization.grid,
            stiffness,
            radial_nodes=discretization.grid.ns - 1,
        )
    scales = np.ones(2)

    def apply(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        result = np.array(vector, copy=True)
        result[: vectorizer.radius_size] = geometry.apply(vector[: vectorizer.radius_size]) * scales[0]
        if stream is not None:
            result[vectorizer.radius_size :] = (
                stream.apply_gauge_free(
                    vector[vectorizer.radius_size :],
                    free_indices=vectorizer.lambda_free_indices,
                    pivot=vectorizer.lambda_pivot,
                    weights=vectorizer.lambda_weights,
                )
                * scales[1]
            )
        return result

    return apply, scales


def solve_spline_fixed_boundary_cli(
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
    """Solve a scalar-pressure fixed boundary in native spline coefficients."""

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
        lam = packed[vectorizer.radius_size :]
        lambda_rms = np.sqrt(np.mean(lam**2)) if lam.size else 0.0
        return VariationalResidual(
            radius_gradient=jnp.asarray(radius),
            lambda_gradient=jnp.asarray(lam),
            radius_rms=jnp.asarray(np.sqrt(np.mean(radius**2))),
            lambda_rms=jnp.asarray(lambda_rms),
            maximum=jnp.asarray(np.max(np.abs(packed))),
        )

    def packed_weak(state: MirrorState) -> VariationalResidual:
        gradient = isotropic_staggered_energy_gradient(state, grid, **energy_kwargs)
        packed = vectorizer.pullback_evaluated_gradient(gradient) / energy_scale
        radius = packed[: vectorizer.radius_size]
        lam = packed[vectorizer.radius_size :]
        return VariationalResidual(
            radius_gradient=jnp.asarray(radius),
            lambda_gradient=jnp.asarray(lam),
            radius_rms=jnp.asarray(np.sqrt(np.mean(radius**2))),
            lambda_rms=jnp.asarray(np.sqrt(np.mean(lam**2)) if lam.size else 0.0),
            maximum=jnp.asarray(np.max(np.abs(packed))),
        )

    history: list[tuple[float, float, float, float, float, float]] = []

    def record(iteration: int, vector: np.ndarray) -> None:
        state = unpack(jnp.asarray(vector))
        energy = evaluate_energy(state)
        variational = packed_variational(vector, state)
        force = isotropic_force_residual(energy, grid)
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
        )
        final_x = optimization.vector
        iterations = optimization.iterations
        optimizer_success = optimization.optimizer_success
        linear_iterations = optimization.linear_iterations
        final_linear_residual = optimization.final_linear_residual
        message = optimization.message

    coefficient_state = unpack_coefficients(jnp.asarray(final_x))
    final_state = discretization.evaluate_state(coefficient_state)
    final_energy = evaluate_energy(final_state)
    final_variational = packed_variational(final_x, final_state)
    final_force = isotropic_force_residual(final_energy, grid)
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
    "solve_spline_fixed_boundary_cli",
]

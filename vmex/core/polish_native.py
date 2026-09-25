"""Native constrained force least squares: the polish solver for supported decks.

The ordinary VMEX solve converges a discrete projected variational problem;
its tiny FSQR/FSQZ/FSQL do not bound the continuum force
``F = J x B - grad p``.  This module corrects the independently evaluated
physical force on the native axis-regular spline representation
(``a_m(rho) = rho**|m| q(rho**2)``):

    r_i(c) = sqrt(w_i |sqrt g_i| / V) F_i(c) / F,    min 0.5 |r|^2  s.t.  C c = 0,

with packed coefficient corrections ``c`` in a fixed metric, ``C`` the frozen
linear tangential (relabeling) gauge, and exact structural boundary and axis
constraints.  Each chart takes Gauss-Newton KKT steps whose normal matrix is
assembled span by span from compressed forward products; charts are refined
exactly (angular zero padding, then knot insertion where the force is
largest).  A final exact-Hessian Newton step on the constrained stationarity
equations ``[A^T r + C^T nu, C c] = 0`` (GMRES, Gauss-Newton KKT factor as
preconditioner) brings the Frobenius-scaled projected gradient

    eta = |P A^T r| / (|A|_F |r|)

below ``stationarity_tolerance``.  Radial derivatives are formed from
coefficient differences and the base and correction jets are synthesized
separately, which keeps the evaluated stationarity accurate to ~1e-12 instead
of the ~1e-7 floor of contracting O(1) coefficients with derivative tables.

Scope: fixed boundary, axisymmetric, prescribed pressure and iota
(``NCURR = 0``, ``GAMMA = 0``), stellarator-symmetric.  Other decks keep the
collocation lane of :mod:`vmex.core.polish_driver`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .polish import apply_high_order_correction, make_native_correction_layout
from .polish_variational import (
    NativeGaugePlan,
    VariationalPlan,
    evaluate_tensorized_strong_force,
    make_native_gauge_plan,
    make_variational_plan,
    minimum_signed_jacobian,
    native_coordinate_scales,
    native_physical_force_residual,
    native_tangential_gauge_matrix,
    native_tangential_gauge_residual,
)
from .strong_force import (
    HighOrderEquilibriumState,
    append_high_order_state_modes,
    insert_high_order_state_knots,
)


@dataclass(frozen=True)
class NativePolishConfig:
    """Schedule of the native polish.

    ``force_tolerance`` bounds the quadrature force norm ``|r|`` (the
    volume-RMS force over ``force_scale``); ``stationarity_tolerance`` bounds
    ``eta``.  ``angular_padding`` exact zero-padded poloidal modes are added
    once; then up to ``max_refinements`` stages each insert ``insert_count``
    knots at the midpoints of the spans with the largest force.
    """

    force_tolerance: float = 1.0e-5
    stationarity_tolerance: float = 1.0e-8
    degree: int = 5
    angular_padding: int = 8
    insert_count: int = 16
    max_refinements: int = 6
    steps_per_chart: int = 2
    max_newton_steps: int = 4


class NativePolishResult(NamedTuple):
    """Polished state, final force norm and stationarity, and work counters."""

    state: HighOrderEquilibriumState
    force_norm: float
    stationarity: float
    converged: bool
    charts: int
    gauss_newton_steps: int
    newton_steps: int
    seconds: float


class _Chart(NamedTuple):
    base: HighOrderEquilibriumState
    plan: VariationalPlan
    layout: Any
    gauge: NativeGaugePlan
    scale: jax.Array
    constraint: sparse.csr_matrix
    force: Callable[[jax.Array], jax.Array]


def _chart(state, force_scale: float, volume_scale: float) -> _Chart:
    degree = int(state.radial_basis.degree)
    plan = make_variational_plan(state, radial_order=degree + 3, stable_derivatives=True)
    layout = make_native_correction_layout(state)
    gauge = make_native_gauge_plan(state, plan)
    scale = native_coordinate_scales(state, layout, plan)
    constraint = sparse.csr_matrix(native_tangential_gauge_matrix(state, layout, gauge, scale))

    def force(coordinates):
        return native_physical_force_residual(
            coordinates, state, layout, gauge, scale, force_scale, volume_scale
        )

    return _Chart(state, plan, layout, gauge, scale, constraint, force)


def _state(chart: _Chart, coordinates) -> HighOrderEquilibriumState:
    return apply_high_order_correction(
        chart.base, chart.layout.unpack(chart.scale * jnp.asarray(coordinates))
    )


@jax.jit
def _span_products(coordinates, directions, base, layout, gauge, scale, force_scale, volume_scale, span_plan):
    """Compressed force tangents on one radial span, shape (colors, rows)."""

    span_gauge = replace(gauge, variational=span_plan)

    def force(value):
        return native_physical_force_residual(
            value, base, layout, span_gauge, scale, force_scale, volume_scale
        )

    return jax.vmap(lambda direction: jax.jvp(force, (coordinates,), (direction,))[1])(directions)


def _span_plan(plan: VariationalPlan, start: int, stop: int) -> VariationalPlan:
    return replace(
        plan,
        rho=plan.rho[start:stop],
        radial_value=plan.radial_value[:, start:stop, :],
        radial_derivative=plan.radial_derivative[:, start:stop, :],
        radial_second_derivative=plan.radial_second_derivative[:, start:stop, :],
        profile_basis=plan.profile_basis[start:stop, :],
        profile_derivative=plan.profile_derivative[start:stop, :],
        quadrature_weights=plan.quadrature_weights[start:stop, ...],
        spline_value=plan.spline_value[start:stop],
        spline_first=plan.spline_first[start:stop],
        spline_second=plan.spline_second[start:stop],
        axis_factors=plan.axis_factors[:, start:stop],
    )


def _normal_system(chart: _Chart, coordinates, residual, force_scale, volume_scale):
    """Gauss-Newton normal matrix ``A^T A`` and gradient ``A^T r``, span by span.

    Spline support makes each radial span's Jacobian block dense on a few
    columns: two coefficients of one field and mode that are ``degree + 1``
    apart never share a span, so ``(field mode, index mod (degree+1))``
    colors the columns and one batched JVP per span recovers the block.  The
    global force Jacobian is never formed.
    """

    plan, layout = chart.plan, chart.layout
    active = np.asarray(layout.active_indices, dtype=np.int64)
    nbasis = int(layout.nbasis)
    width = int(chart.base.radial_basis.degree) + 1
    _, color = np.unique((active // nbasis) * width + (active % nbasis) % width, return_inverse=True)
    directions = np.zeros((int(color.max()) + 1, layout.size))
    directions[color, np.arange(layout.size)] = 1.0
    directions = jnp.asarray(directions)
    support = (
        (np.asarray(plan.radial_value) != 0.0)
        | (np.asarray(plan.radial_derivative) != 0.0)
        | (np.asarray(plan.radial_second_derivative) != 0.0)
    ).any(axis=0)[:, active % nbasis]
    spans = int(chart.base.radial_basis.breakpoints.size - 1)
    order = int(plan.rho.size) // spans
    rows_per_node = int(plan.theta.size * plan.zeta.size * 3)
    rows, cols, vals = [], [], []
    gradient = np.zeros(layout.size)
    residual = np.asarray(residual)
    for span in range(spans):
        start, stop = span * order, (span + 1) * order
        products = np.asarray(
            _span_products(
                jnp.asarray(coordinates), directions, chart.base, layout, chart.gauge, chart.scale,
                jnp.asarray(force_scale), jnp.asarray(volume_scale), _span_plan(plan, start, stop),
            )
        )
        columns = np.flatnonzero(support[start:stop].any(axis=0))
        # Column j of the block is its color's product; rows outside j's
        # radial support are structurally zero.
        block = products[color[columns]].T
        node_of_row = np.repeat(np.arange(order), rows_per_node)
        block = np.where(support[start:stop][node_of_row][:, columns], block, 0.0)
        rows.append(np.repeat(columns, columns.size))
        cols.append(np.tile(columns, columns.size))
        vals.append((block.T @ block).reshape(-1))
        gradient[columns] += block.T @ residual[start * rows_per_node : stop * rows_per_node]
    normal = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(layout.size, layout.size),
    ).tocsr()
    return normal, gradient


def _kkt_factor(normal, constraint):
    return sparse_linalg.splu(sparse.bmat([[normal, constraint.T], [constraint, None]], format="csc"))


class _Projector:
    """Euclidean projector onto ker C from one sparse factor per chart."""

    def __init__(self, constraint):
        n = constraint.shape[1]
        self.q = constraint.shape[0]
        self.lu = sparse_linalg.splu(
            sparse.bmat([[sparse.eye(n, format="csc"), constraint.T], [constraint, None]], format="csc")
        )

    def __call__(self, vector):
        vector = np.asarray(vector, dtype=float)
        return self.lu.solve(np.concatenate((vector, np.zeros(self.q))))[: vector.size]


def _admissible(chart: _Chart, candidate) -> bool:
    gauge = float(jnp.linalg.norm(native_tangential_gauge_residual(
        chart.base, chart.layout.unpack(chart.scale * candidate), chart.gauge)))
    return gauge < 1.0e-10 and float(minimum_signed_jacobian(_state(chart, candidate), chart.plan)) > 0.0


def _gauss_newton(chart: _Chart, coordinates, force_scale, volume_scale, steps: int):
    """Feasible Gauss-Newton KKT steps with backtracking on the force norm."""

    taken = 0
    for _ in range(steps):
        residual = np.asarray(chart.force(coordinates))
        norm = float(np.linalg.norm(residual))
        normal, gradient = _normal_system(chart, coordinates, residual, force_scale, volume_scale)
        defect = chart.constraint @ np.asarray(coordinates)
        step = _kkt_factor(normal, chart.constraint).solve(-np.concatenate((gradient, defect)))
        step = step[: chart.layout.size]
        for fraction in (1.0, 0.5, 0.25):
            candidate = coordinates + fraction * jnp.asarray(step)
            trial = float(jnp.linalg.norm(chart.force(candidate)))
            if trial < norm * (1.0 - 1.0e-8) and _admissible(chart, candidate):
                coordinates, taken = candidate, taken + 1
                break
        else:
            break
        if trial > norm * (1.0 - 1.0e-3):
            break
    return coordinates, taken


def _refine(state, force_scale, volume_scale, count: int):
    """Insert ``count`` knots at the midpoints of the largest-force spans."""

    spans = int(state.radial_basis.breakpoints.size - 1)
    count = min(int(count), spans - 1)
    order = int(state.radial_basis.degree) + 3
    plan = make_variational_plan(state, radial_order=order, stable_derivatives=True)
    samples = evaluate_tensorized_strong_force(state, plan)
    density = plan.quadrature_weights * state.jacobian_sign * samples.sqrt_g * jnp.sum(samples.force**2, axis=-1)
    scores = np.asarray(jnp.sum(density, axis=(1, 2))).reshape(spans, order).sum(axis=1)
    selected = np.sort(np.argsort(scores)[-count:])
    breaks = np.asarray(state.radial_basis.breakpoints)
    return insert_high_order_state_knots(state, 0.5 * (breaks[selected] + breaks[selected + 1]))


def _stationarity(chart: _Chart, coordinates, force_scale, volume_scale, config: NativePolishConfig):
    """Exact-Hessian constrained Newton on the stationarity equations."""

    gradient = jax.jit(jax.grad(lambda c: 0.5 * jnp.vdot(chart.force(c), chart.force(c))))
    projector = _Projector(chart.constraint)
    n, q = chart.constraint.shape[1], chart.constraint.shape[0]

    def measure(c):
        residual = np.asarray(chart.force(c))
        normal, _ = _normal_system(chart, c, residual, force_scale, volume_scale)
        frobenius = float(np.sqrt(max(normal.diagonal().sum(), 0.0)))
        g = np.asarray(gradient(c))
        eta = float(np.linalg.norm(projector(g))) / max(frobenius * float(np.linalg.norm(residual)), 1e-300)
        return eta, g, normal

    eta, g, normal = measure(coordinates)
    steps = 0
    for _ in range(config.max_newton_steps):
        if eta <= config.stationarity_tolerance:
            break
        hessian = jax.jit(lambda v, c=coordinates: jax.jvp(gradient, (c,), (v,))[1])
        factor = _kkt_factor(normal, chart.constraint)

        def apply(z):
            top = np.asarray(hessian(jnp.asarray(z[:n]))) + chart.constraint.T @ z[n:]
            return np.concatenate((top, chart.constraint @ z[:n]))

        rhs = -np.concatenate((g, chart.constraint @ np.asarray(coordinates)))
        solution, _ = sparse_linalg.gmres(
            sparse_linalg.LinearOperator((n + q, n + q), matvec=apply, dtype=float),
            rhs,
            M=sparse_linalg.LinearOperator((n + q, n + q), matvec=factor.solve, dtype=float),
            rtol=1.0e-10, atol=0.0, restart=40, maxiter=40,
        )
        candidate = coordinates + jnp.asarray(solution[:n])
        if not _admissible(chart, candidate):
            break
        trial_eta, trial_g, trial_normal = measure(candidate)
        if trial_eta >= eta:
            break
        coordinates, eta, g, normal, steps = candidate, trial_eta, trial_g, trial_normal, steps + 1
    return coordinates, eta, steps


def polish_native(
    state: HighOrderEquilibriumState,
    *,
    force_scale: float,
    volume_scale: float,
    config: NativePolishConfig | None = None,
    emit: Callable[[str], Any] | None = None,
) -> NativePolishResult:
    """Correct the physical force of a lifted native state (see module docstring).

    ``force_scale`` (N m^-3) and ``volume_scale`` (m^3) fix the metric of the
    residual; the returned ``force_norm`` is the quadrature volume-RMS force
    over ``force_scale``.  The returned state is the float64 sum of the final
    chart's base and correction; ``stationarity`` is evaluated on that
    (base, correction) pair, whose rounding to float64 coefficients is below
    every physical tolerance.
    """

    config = NativePolishConfig() if config is None else config
    started = perf_counter()
    say = (lambda _message: None) if emit is None else emit
    m_max = int(np.max(np.abs(np.asarray(state.m))))
    if config.angular_padding > 0:
        new_m = np.arange(m_max + 1, m_max + 1 + config.angular_padding)
        state = append_high_order_state_modes(state, new_m, np.zeros_like(new_m))
    charts = gauss_newton_steps = 0
    chart = coordinates = None
    force_norm = np.inf
    for stage in range(config.max_refinements + 1):
        if stage:
            state = _refine(_state(chart, coordinates), force_scale, volume_scale, config.insert_count)
        chart = _chart(state, force_scale, volume_scale)
        coordinates = jnp.zeros((chart.layout.size,))
        coordinates, taken = _gauss_newton(chart, coordinates, force_scale, volume_scale, config.steps_per_chart)
        charts, gauss_newton_steps = charts + 1, gauss_newton_steps + taken
        force_norm = float(jnp.linalg.norm(chart.force(coordinates)))
        say(f" native polish chart {charts}: basis {chart.base.radial_basis.size}, "
            f"{chart.layout.size} coordinates, |F|/F* = {force_norm:.3e}")
        if force_norm <= config.force_tolerance:
            break
    coordinates, eta, newton_steps = _stationarity(chart, coordinates, force_scale, volume_scale, config)
    force_norm = float(jnp.linalg.norm(chart.force(coordinates)))
    say(f" native polish stationarity: eta = {eta:.3e} after {newton_steps} Newton step(s)")
    return NativePolishResult(
        state=_state(chart, coordinates),
        force_norm=force_norm,
        stationarity=eta,
        converged=bool(force_norm <= config.force_tolerance and eta <= config.stationarity_tolerance),
        charts=charts,
        gauss_newton_steps=gauss_newton_steps,
        newton_steps=newton_steps,
        seconds=perf_counter() - started,
    )


def physical_scales(state: HighOrderEquilibriumState) -> tuple[float, float]:
    """``(F*, V)``: the wout ``volavgB**2 / (mu0 Aminor_p)`` and ``volume_p``.

    ``V = int |sqrt g|``, ``volavgB`` is the volume-RMS ``|B|`` and
    ``Aminor_p = sqrt(<area> / pi)`` with the cross-section area
    ``int |sqrt g| / R d rho d theta`` averaged over the toroidal angle,
    matching VMEC2000's definitions.
    """

    from .polish_variational import evaluate_variational_fields
    from .profiles import MU0

    plan = make_variational_plan(state, radial_order=int(state.radial_basis.degree) + 3)
    fields = evaluate_variational_fields(state, plan)
    weights = jnp.broadcast_to(plan.quadrature_weights, plan.shape).reshape(plan.rho.size, -1)
    jacobian = jnp.abs(fields.sqrt_g).reshape(weights.shape)
    radius = jnp.linalg.norm(fields.position[..., :2], axis=-1).reshape(weights.shape)
    volume = jnp.sum(weights * jacobian)
    b_rms = jnp.sqrt(jnp.sum(weights * jacobian * jnp.sum(fields.B**2, axis=-1).reshape(weights.shape)) / volume)
    # The weights integrate the full torus; dividing by 2 pi averages the
    # (rho, theta) area over the toroidal angle.
    area = jnp.sum(weights * jacobian / radius) / (2.0 * np.pi)
    minor_radius = jnp.sqrt(area / np.pi)
    return float(b_rms**2 / (MU0 * minor_radius)), float(volume)


def native_polish_supported(source) -> bool:
    """Whether a deck is inside the native polish's qualified scope."""

    return (
        not bool(getattr(source, "lasym", False))
        and int(getattr(source, "ntor", 0)) == 0
        and int(getattr(source, "ncurr", 0)) == 0
        and float(getattr(source, "gamma", 0.0)) == 0.0
        and not bool(getattr(source, "lfreeb", False))
    )


__all__ = ["NativePolishConfig", "NativePolishResult", "native_polish_supported", "polish_native"]

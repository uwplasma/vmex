"""Homotopy and pseudo-arclength strong-root correction (diagnostic route).

The production polish route is Gauss--Newton on the overdetermined physical
collocation residual: :func:`vmex.core.polish_driver.polish_legacy_solution`
calls :func:`~vmex.core.polish_driver.polish_collocation_least_squares`, and
:mod:`vmex.core.solver` reaches nothing in this module.  What lives here is the
earlier *square* route -- a homotopy in ``alpha`` from the legacy VMEC
endpoint to the strong-force operator, driven by pseudo-transient continuation
with a pseudo-arclength corrector -- kept for rank, branch and projection
diagnostics rather than for shipping equilibria.

Splitting it out is a code-organization change only.  The two routes share one
helper, :func:`vmex.core.polish_driver._corrected_state`, and the square route
reuses the driver's :class:`~vmex.core.polish_driver.PolishConfig`,
:class:`~vmex.core.polish_driver.PolishReport` and
:class:`~vmex.core.polish_driver.PolishResult` so that a diagnostic run is
reported in the same vocabulary as a production one.  Nothing in this module is
exported from :mod:`vmex`; it is reached explicitly, by benchmarks and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from time import perf_counter
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from solvax import gmres

from .errors import StrongForceCertificationError, StrongForceContinuationError
from .polish import (
    HighOrderCorrection,
    StrongPhysicalChart,
    StrongRootRuntime,
    _coordinate_gauge_residual_unscaled,
    _fit_regularized_channel,
    _physical_coordinate_blocks,
    _physical_equation_basis,
    _strong_residual_unscaled,
    apply_high_order_correction,
    strong_root_residual,
)
from .polish_driver import (
    PolishConfig,
    PolishReport,
    PolishResult,
    _corrected_state,
    _failed_certificate_checks,
    _normalization_fields,
)
from .strong_force import (
    StrongForceReport,
    certify_strong_force,
    evaluate_strong_force,
)

Array = Any


class PreconditionerQuality(NamedTuple):
    """True operator residual after one right-preconditioner application."""

    relative_residual: Array
    maximum: Array
    rms: Array


@dataclass(frozen=True)
class PreconditionerRefreshPolicy:
    """Thresholds for rebuilding a stored low-order factorization."""

    max_alpha_change: float = 0.25
    max_krylov_iterations: int = 80
    max_relative_residual: float = 0.5
    min_jacobian_margin_ratio: float = 0.7
    max_parameter_distance: float = 0.1

    def __post_init__(self) -> None:
        if self.max_alpha_change <= 0.0:
            raise ValueError("max_alpha_change must be positive")
        if self.max_krylov_iterations < 1:
            raise ValueError("max_krylov_iterations must be positive")
        if self.max_relative_residual <= 0.0:
            raise ValueError("max_relative_residual must be positive")
        if not 0.0 < self.min_jacobian_margin_ratio <= 1.0:
            raise ValueError("min_jacobian_margin_ratio must lie in (0, 1]")
        if self.max_parameter_distance <= 0.0:
            raise ValueError("max_parameter_distance must be positive")


@dataclass(frozen=True)
class PreconditionerSnapshot:
    """Cheap nonlinear-stage data used by the factor refresh policy."""

    alpha: float
    radial_degree: int
    radial_size: int
    krylov_iterations: int
    relative_residual: float
    jacobian_margin: float
    parameter_distance: float = 0.0
    transpose_converged: bool = True


class PreconditionerRefreshDecision(NamedTuple):
    """Host-side refresh decision with reviewer-visible reasons."""

    refresh: bool
    reasons: tuple[str, ...]


class StrongProjectionDiagnostics(NamedTuple):
    """How much solve-grid force content survives the square projection."""

    sampled_rms: Array
    reconstructed_rms: Array
    unresolved_rms: Array
    unresolved_fraction: Array
    angular_unresolved_fraction: Array
    radial_fit_unresolved_fraction: Array
    radial_unresolved_fraction: Array
    helical_unresolved_fraction: Array
    equation_discarded_fraction: Array
    projected_residual_rms: Array


@dataclass(frozen=True, eq=False)
class StrongModeBlockPreconditioner:
    """Bounded Fourier-mode factors for a strong-root Jacobian pencil."""

    indices: tuple[Array, ...]
    low_blocks: tuple[Array, ...]
    strong_blocks: tuple[Array, ...]
    build_seconds: float

    def apply(
        self,
        rhs: Array,
        alpha: Array = 1.0,
        dtau: Array | float = jnp.inf,
    ) -> Array:
        """Apply regularized block solves without a dense global Jacobian."""

        return self._apply(rhs, alpha, dtau, transpose=False)

    def apply_transpose(
        self,
        rhs: Array,
        alpha: Array = 1.0,
        dtau: Array | float = jnp.inf,
    ) -> Array:
        """Apply the exact transpose factors used by implicit adjoints."""

        return self._apply(rhs, alpha, dtau, transpose=True)

    def _apply(
        self,
        rhs: Array,
        alpha: Array,
        dtau: Array | float,
        *,
        transpose: bool,
    ) -> Array:
        rhs = jnp.asarray(rhs)
        alpha = jnp.asarray(alpha, dtype=rhs.dtype)
        inverse_dtau = jnp.where(
            jnp.isfinite(jnp.asarray(dtau)),
            1.0 / jnp.asarray(dtau, dtype=rhs.dtype),
            jnp.asarray(0.0, dtype=rhs.dtype),
        )
        result = jnp.zeros_like(rhs)
        for indices, low, strong in zip(
            self.indices, self.low_blocks, self.strong_blocks, strict=True
        ):
            matrix = (1.0 - alpha) * low + alpha * strong
            if transpose:
                matrix = matrix.T
            scale = jnp.maximum(jnp.linalg.norm(matrix, ord=jnp.inf), 1.0)
            regularization = jnp.where(
                inverse_dtau > 0.0,
                32.0 * jnp.finfo(rhs.dtype).eps * scale,
                0.0,
            )
            shifted = matrix + (
                inverse_dtau + regularization
            ) * jnp.eye(matrix.shape[0], dtype=rhs.dtype)
            result = result.at[indices].set(jnp.linalg.solve(shifted, rhs[indices]))
        return result


def _solvax_continuation_api() -> tuple[Any, ...]:
    """Load the continuation extension supplied by the companion SOLVAX PR."""

    try:
        from solvax import (
            ContinuationConfig,
            PseudoTransientConfig,
            adaptive_continuation,
            pseudo_arclength_corrector,
            pseudo_transient_continuation,
        )
    except ImportError as error:
        raise RuntimeError(
            "strong-force polishing requires a SOLVAX release containing "
            "adaptive continuation, pseudo-transient continuation, and "
            "pseudo-arclength correction (uwplasma/SOLVAX#87)"
        ) from error
    return (
        ContinuationConfig,
        PseudoTransientConfig,
        adaptive_continuation,
        pseudo_arclength_corrector,
        pseudo_transient_continuation,
    )


def _supports_keyword(function: Any, keyword: str) -> bool:
    """Return whether an installed SOLVAX callable exposes a new keyword."""

    try:
        return keyword in signature(function).parameters
    except (TypeError, ValueError):
        return False


def _residual_evaluations(result: Any) -> int:
    """Read exact work accounting, with a conservative pre-0.19 fallback."""

    nonlinear_steps = getattr(result, "nonlinear_steps", getattr(result, "steps", 0))
    return int(getattr(result, "residual_evaluations", nonlinear_steps + 1))


def strong_projection_diagnostics(
    vector: Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
) -> StrongProjectionDiagnostics:
    """Compare the square strong residual with its solve-grid force samples.

    The independent certificate deliberately uses shifted, overintegrated
    nodes. This diagnostic instead stays on the *solve* nodes and reports the
    content lost by the angular/radial fit and by the final square equation
    chart. It therefore distinguishes a projection mismatch from nonlinear
    solver failure without weakening or replacing the independent certificate.
    """

    from .strong_force import evaluate_strong_force

    full = chart.lift(vector)
    correction = runtime.layout.unpack(
        jnp.asarray(runtime.coordinate_scale) * full
    )
    state = apply_high_order_correction(runtime.native, correction)
    radial = jnp.asarray(runtime.radial_nodes)
    theta = jnp.asarray(runtime.theta)
    zeta = jnp.asarray(runtime.zeta)
    rr, tt, zz = jnp.meshgrid(radial, theta, zeta, indexing="ij")
    samples = evaluate_strong_force(state, rr, tt, zz)
    denominator = jnp.asarray(runtime.normalization_denominator)
    volume_weight = jnp.abs(samples.sqrt_g)
    radial_force = (
        2.0 * samples.signed_radial_force_density * volume_weight / denominator
    )
    helical_force = (
        2.0 * samples.signed_helical_force_density * volume_weight / denominator
    )
    radial_coefficients = _fit_regularized_channel(
        radial_force, runtime.cosine_projection, radial, runtime
    )
    helical_coefficients = _fit_regularized_channel(
        helical_force, runtime.sine_projection, radial, runtime
    )

    radial_basis = jnp.asarray(
        state.radial_basis.basis_matrix(radial * radial)
    )
    regularity = radial[:, None] ** jnp.abs(jnp.asarray(state.m))[None, :]
    radial_modes = (radial_basis @ radial_coefficients) * regularity
    helical_modes = (radial_basis @ helical_coefficients) * regularity
    # Reconstruct on the same flattened (theta, zeta) angular points the
    # runtime projections were built on.  Broadcasting the two 1-D grids
    # directly only typechecks when nzeta == 1, so the ntor = 0 benchmarks
    # never caught the missing mesh product.
    theta_mesh, zeta_mesh = jnp.meshgrid(theta, zeta, indexing="ij")
    phase = (
        jnp.asarray(state.m)[:, None]
        * theta_mesh.reshape(1, -1)
        - jnp.asarray(state.n)[:, None] * zeta_mesh.reshape(1, -1)
    )
    radial_angular_modes = jnp.einsum(
        "ra,ma->rm",
        radial_force.reshape((radial.size, -1)),
        jnp.asarray(runtime.cosine_projection),
    )
    helical_angular_modes = jnp.einsum(
        "ra,ma->rm",
        helical_force.reshape((radial.size, -1)),
        jnp.asarray(runtime.sine_projection),
    )
    radial_angular_reconstructed = jnp.einsum(
        "rm,ma->ra", radial_angular_modes, jnp.cos(phase)
    ).reshape(radial_force.shape)
    helical_angular_reconstructed = jnp.einsum(
        "rm,ma->ra", helical_angular_modes, jnp.sin(phase)
    ).reshape(helical_force.shape)
    radial_reconstructed = jnp.einsum(
        "rm,ma->ra", radial_modes, jnp.cos(phase)
    ).reshape(radial_force.shape)
    helical_reconstructed = jnp.einsum(
        "rm,ma->ra", helical_modes, jnp.sin(phase)
    ).reshape(helical_force.shape)

    def pair_rms(first: Array, second: Array) -> Array:
        return jnp.sqrt(jnp.mean(first * first + second * second))

    def relative(error: Array, reference: Array) -> Array:
        return jnp.linalg.norm(error) / jnp.maximum(
            jnp.linalg.norm(reference), jnp.finfo(reference.dtype).tiny
        )

    sampled_rms = pair_rms(radial_force, helical_force)
    reconstructed_rms = pair_rms(
        radial_reconstructed, helical_reconstructed
    )
    radial_error = radial_force - radial_reconstructed
    helical_error = helical_force - helical_reconstructed
    angular_error_r = radial_force - radial_angular_reconstructed
    angular_error_h = helical_force - helical_angular_reconstructed
    radial_fit_error_r = radial_angular_reconstructed - radial_reconstructed
    radial_fit_error_h = helical_angular_reconstructed - helical_reconstructed
    unresolved_rms = pair_rms(radial_error, helical_error)
    full_coefficients = (
        _strong_residual_unscaled(
            full,
            runtime,
            include_coordinate_gauge=False,
        )
        / jnp.asarray(runtime.strong_scale)
    )
    retained_coefficients = jnp.asarray(chart.equation_basis) @ (
        jnp.asarray(chart.equation_basis).T @ full_coefficients
    )
    projected = chart.project(full_coefficients)
    return StrongProjectionDiagnostics(
        sampled_rms=sampled_rms,
        reconstructed_rms=reconstructed_rms,
        unresolved_rms=unresolved_rms,
        unresolved_fraction=unresolved_rms
        / jnp.maximum(sampled_rms, jnp.finfo(sampled_rms.dtype).tiny),
        angular_unresolved_fraction=pair_rms(
            angular_error_r, angular_error_h
        )
        / jnp.maximum(sampled_rms, jnp.finfo(sampled_rms.dtype).tiny),
        radial_fit_unresolved_fraction=pair_rms(
            radial_fit_error_r, radial_fit_error_h
        )
        / jnp.maximum(sampled_rms, jnp.finfo(sampled_rms.dtype).tiny),
        radial_unresolved_fraction=relative(radial_error, radial_force),
        helical_unresolved_fraction=relative(helical_error, helical_force),
        equation_discarded_fraction=relative(
            full_coefficients - retained_coefficients, full_coefficients
        ),
        projected_residual_rms=jnp.linalg.norm(projected)
        / jnp.sqrt(float(chart.size)),
    )


def make_strong_physical_chart(
    runtime: StrongRootRuntime,
    *,
    relative_tolerance: float = 1.0e-10,
) -> StrongPhysicalChart:
    """Eliminate the exactly linear coordinate gauge from a strong root.

    The one-time dense factorization is restricted to the coordinate-gauge
    operator.  The nonlinear physical force and all subsequent JVP/VJP calls
    remain matrix-free.  ``relative_tolerance`` defines the numerical rank of
    the gauge operator and must leave at least one physical coordinate.
    """

    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    started = perf_counter()
    size = runtime.layout.size
    zero = jnp.zeros((size,), dtype=jnp.asarray(runtime.native.R_cos).dtype)
    gauge_operator = jax.jacfwd(
        lambda value: _coordinate_gauge_residual_unscaled(value, runtime)
    )(zero)
    _, singular_values, right_transpose = np.linalg.svd(
        np.asarray(jax.device_get(gauge_operator)),
        full_matrices=True,
    )
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("coordinate-gauge operator has no independent equations")
    gauge_rank = int(
        np.sum(singular_values > relative_tolerance * singular_values[0])
    )
    if gauge_rank <= 0 or gauge_rank >= size:
        raise ValueError(
            "coordinate-gauge rank must be positive and smaller than the root"
        )
    equation_basis = _physical_equation_basis(runtime.layout)
    physical_size = size - gauge_rank
    if equation_basis.shape != (size, physical_size):
        raise ValueError(
            "physical force-output equation count does not match gauge-free "
            f"coordinates: {equation_basis.shape[1]} != {physical_size}"
        )
    return StrongPhysicalChart(
        coordinate_basis=jnp.asarray(right_transpose[gauge_rank:].T),
        equation_basis=jnp.asarray(equation_basis),
        coordinate_scale=jnp.ones((physical_size,)),
        equation_scale=jnp.ones((physical_size,)),
        gauge_rank=gauge_rank,
        build_seconds=perf_counter() - started,
    )


@jax.jit
def strong_physical_residual(
    vector: Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
    alpha: Array = 1.0,
) -> Array:
    """Evaluate the square strong root in exact gauge-free coordinates."""

    full = chart.lift(vector)
    low = chart.project(strong_root_residual(full, runtime, 0.0))
    strong = chart.project(
        _strong_residual_unscaled(
            full,
            runtime,
            include_coordinate_gauge=False,
        )
        / jnp.asarray(runtime.strong_scale)
    )
    alpha = jnp.asarray(alpha, dtype=jnp.asarray(vector).dtype)
    return low + alpha * (strong - low)


def strong_root_rank(
    runtime: StrongRootRuntime,
    vector: Array | None = None,
    *,
    relative_tolerance: float = 1.0e-9,
) -> tuple[int, Array]:
    """Assemble a small diagnostic Jacobian and return numerical rank/SVD."""

    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    point = jnp.zeros((runtime.layout.size,)) if vector is None else jnp.asarray(vector)
    jacobian = jax.jacfwd(lambda value: strong_root_residual(value, runtime))(point)
    singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
    threshold = float(relative_tolerance) * singular_values[0]
    return int(jnp.sum(singular_values > threshold)), singular_values


def build_strong_mode_block_preconditioner(
    runtime: StrongRootRuntime,
    vector: Array | None = None,
    *,
    poloidal_bandwidth: int = 3,
) -> StrongModeBlockPreconditioner:
    """Probe bounded same-mode blocks at one reusable linearization point."""

    if poloidal_bandwidth < 1:
        raise ValueError("poloidal_bandwidth must be positive")
    started = perf_counter()
    base = (
        jnp.zeros(
            (runtime.layout.size,),
            dtype=jnp.asarray(runtime.native.R_cos).dtype,
        )
        if vector is None
        else jnp.asarray(vector)
    )
    if base.shape != (runtime.layout.size,):
        raise ValueError(
            f"block linearization has shape {base.shape}; "
            f"expected {(runtime.layout.size,)}"
        )
    grouped: dict[tuple[int, int], list[int]] = {}
    for group in runtime.layout.groups:
        key = (
            int(group.abs_n),
            int(group.m) // int(poloidal_bandwidth),
        )
        grouped.setdefault(key, []).extend(range(group.start, group.stop))
    indices = tuple(
        jnp.asarray(grouped[key], dtype=jnp.int32)
        for key in sorted(grouped)
    )
    low_blocks: list[Array] = []
    strong_blocks: list[Array] = []
    for block_indices in indices:
        local_zero = jnp.zeros((block_indices.size,), dtype=base.dtype)

        def block_residual(local: Array, alpha: float) -> Array:
            candidate = base.at[block_indices].add(local)
            return strong_root_residual(candidate, runtime, alpha)[block_indices]

        low_blocks.append(
            jax.jacfwd(lambda local: block_residual(local, 0.0))(local_zero)
        )
        strong_blocks.append(
            jax.jacfwd(lambda local: block_residual(local, 1.0))(local_zero)
        )
    jax.block_until_ready((low_blocks, strong_blocks))
    return StrongModeBlockPreconditioner(
        indices,
        tuple(low_blocks),
        tuple(strong_blocks),
        perf_counter() - started,
    )


def build_strong_physical_block_preconditioner(
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart,
    vector: Array | None = None,
    *,
    poloidal_bandwidth: int = 3,
) -> StrongModeBlockPreconditioner:
    """Probe bounded mode blocks directly in structured physical coordinates."""

    if poloidal_bandwidth < 1:
        raise ValueError("poloidal_bandwidth must be positive")
    started = perf_counter()
    base = (
        jnp.zeros(
            (chart.size,),
            dtype=jnp.asarray(runtime.native.R_cos).dtype,
        )
        if vector is None
        else jnp.asarray(vector)
    )
    if base.shape != (chart.size,):
        raise ValueError(
            f"physical block linearization has shape {base.shape}; "
            f"expected {(chart.size,)}"
        )
    indices = _physical_coordinate_blocks(
        runtime,
        chart,
        poloidal_bandwidth,
    )
    low_blocks: list[Array] = []
    strong_blocks: list[Array] = []
    for block_indices in indices:
        local_zero = jnp.zeros((block_indices.size,), dtype=base.dtype)

        def block_residual(local: Array, alpha: float) -> Array:
            candidate = base.at[block_indices].add(local)
            return strong_physical_residual(
                candidate, runtime, chart, alpha
            )[block_indices]

        low_blocks.append(
            jax.jacfwd(lambda local: block_residual(local, 0.0))(local_zero)
        )
        strong_blocks.append(
            jax.jacfwd(lambda local: block_residual(local, 1.0))(local_zero)
        )
    jax.block_until_ready((low_blocks, strong_blocks))
    return StrongModeBlockPreconditioner(
        indices,
        tuple(low_blocks),
        tuple(strong_blocks),
        perf_counter() - started,
    )


def preconditioner_quality(
    operator: Callable[[HighOrderCorrection], HighOrderCorrection],
    preconditioner: Callable[[HighOrderCorrection], HighOrderCorrection],
    probes: HighOrderCorrection,
) -> PreconditionerQuality:
    """Measure true relative residuals for a batch of leading-axis probes."""

    responses = jax.vmap(lambda rhs: operator(preconditioner(rhs)))(probes)
    residuals = jax.tree.map(jnp.subtract, responses, probes)

    def norms(tree):
        leaves = jax.tree.leaves(tree)
        squared = sum(jnp.sum(jnp.abs(leaf) ** 2, axis=tuple(range(1, leaf.ndim))) for leaf in leaves)
        return jnp.sqrt(squared)

    dtype = jax.tree.leaves(probes)[0].dtype
    relative = norms(residuals) / jnp.maximum(norms(probes), jnp.finfo(dtype).tiny)
    return PreconditionerQuality(
        relative_residual=relative,
        maximum=jnp.max(relative),
        rms=jnp.sqrt(jnp.mean(relative * relative)),
    )


def preconditioner_refresh_decision(
    previous: PreconditionerSnapshot,
    current: PreconditionerSnapshot,
    policy: PreconditionerRefreshPolicy | None = None,
) -> PreconditionerRefreshDecision:
    """Return whether nonlinear progress has invalidated stored factors."""

    policy = PreconditionerRefreshPolicy() if policy is None else policy
    reasons: list[str] = []
    if abs(current.alpha - previous.alpha) > policy.max_alpha_change:
        reasons.append("continuation-step")
    if (
        current.radial_degree != previous.radial_degree
        or current.radial_size != previous.radial_size
    ):
        reasons.append("radial-grid")
    if current.krylov_iterations > policy.max_krylov_iterations:
        reasons.append("krylov-work")
    if current.relative_residual > policy.max_relative_residual:
        reasons.append("linear-quality")
    reference_margin = max(abs(previous.jacobian_margin), np.finfo(float).tiny)
    if current.jacobian_margin < policy.min_jacobian_margin_ratio * reference_margin:
        reasons.append("jacobian-margin")
    if current.parameter_distance > policy.max_parameter_distance:
        reasons.append("parameter-distance")
    if not current.transpose_converged:
        reasons.append("transpose-certificate")
    return PreconditionerRefreshDecision(bool(reasons), tuple(reasons))


@dataclass(frozen=True)
class _IdentityPreconditioner:
    """Explicit identity action used to benchmark unpreconditioned JFNK."""

    build_seconds: float = 0.0

    def apply(self, rhs, alpha=1.0, dtau=jnp.inf):
        del alpha, dtau
        return rhs


def _build_mode_block_preconditioner(
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart | None = None,
) -> StrongModeBlockPreconditioner:
    """Backward-compatible private seam for the shared block builder."""

    if chart is None:
        return build_strong_mode_block_preconditioner(runtime)
    return build_strong_physical_block_preconditioner(runtime, chart)


def _solve_residual(
    vector: jax.Array,
    runtime: StrongRootRuntime,
    alpha: jax.Array,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    """Evaluate either the legacy full chart or the structured physical chart."""

    if chart is None:
        return strong_root_residual(vector, runtime, alpha)
    return strong_physical_residual(vector, runtime, chart, alpha)


def _full_solve_vector(
    vector: jax.Array,
    chart: StrongPhysicalChart | None,
) -> jax.Array:
    """Lift physical solve coordinates into the existing full root layout."""

    return jnp.asarray(vector) if chart is None else chart.lift(vector)


def _continuation_precondition(
    rhs: jax.Array,
    alpha: jax.Array,
    dtau: jax.Array,
    runtime: StrongRootRuntime,
    block_preconditioner: StrongModeBlockPreconditioner,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    """Use the exact legacy inverse early and mode bands near strong force."""

    if isinstance(block_preconditioner, _IdentityPreconditioner):
        return rhs
    if chart is not None:
        return block_preconditioner.apply(rhs, alpha, dtau)
    return jax.lax.cond(
        jnp.asarray(alpha) < 0.5,
        lambda value: _low_inverse(value, runtime),
        lambda value: block_preconditioner.apply(value, alpha, dtau),
        rhs,
    )


def _minimum_signed_jacobian(
    vector: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    state = _corrected_state(vector, runtime, chart)
    rr, tt, zz = jnp.meshgrid(
        jnp.asarray(runtime.radial_nodes),
        jnp.asarray(runtime.theta),
        jnp.asarray(runtime.zeta),
        indexing="ij",
    )
    samples = evaluate_strong_force(state, rr, tt, zz)
    signed = float(state.jacobian_sign) * samples.sqrt_g / jnp.maximum(rr, 1.0e-14)
    return jnp.min(signed)


def _low_inverse(rhs: jax.Array, runtime: StrongRootRuntime) -> jax.Array:
    """Invert the row-scaled low endpoint in reduced vector coordinates."""

    high_rhs = runtime.layout.unpack(
        jnp.asarray(rhs) / jnp.asarray(runtime.equation_scale)
    )
    low_rhs = runtime.transfer.restrict(high_rhs)
    solution = runtime.low_preconditioner.solve_scaled(low_rhs)
    return runtime.layout.pack(runtime.transfer.prolong(solution)) / jnp.asarray(
        runtime.coordinate_scale
    )


def _solve_low_inverse(
    rhs: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    """Apply the legacy inverse in the active solve-coordinate chart."""

    if chart is None:
        return _low_inverse(rhs, runtime)
    full_rhs = jnp.asarray(chart.equation_basis) @ (
        jnp.asarray(rhs) / jnp.asarray(chart.equation_scale)
    )
    full_solution = _low_inverse(full_rhs, runtime)
    return (
        jnp.asarray(chart.coordinate_basis).T @ full_solution
    ) / jnp.asarray(chart.coordinate_scale)


def _normalized_low_residual_norm(
    residual: jax.Array,
    runtime: StrongRootRuntime,
    chart: StrongPhysicalChart | None = None,
) -> jax.Array:
    """Return the low-endpoint RMS before numerical row equilibration."""

    equation_scale = (
        runtime.equation_scale if chart is None else chart.equation_scale
    )
    size = runtime.layout.size if chart is None else chart.size
    unscaled = jnp.asarray(residual) / jnp.asarray(equation_scale)
    return jnp.linalg.norm(unscaled) / np.sqrt(float(size))


def _ptc_config(config: PolishConfig, *, residual_scale: float) -> Any:
    _, PseudoTransientConfig, _, _, _ = _solvax_continuation_api()
    return PseudoTransientConfig(
        rtol=config.tolerance,
        # Couple the roundoff floor to a representative residual norm.  Both
        # tolerances then transform with a harmless positive equation scaling:
        # unlike a fixed absolute tolerance this cannot accept an unsolved
        # stage, while unlike ``atol=0`` it does not chase alpha-zero roundoff.
        # The independent dimensional certificate remains the final gate.
        atol=config.tolerance * float(residual_scale),
        max_steps=config.max_nonlinear_iterations,
        initial_dt=config.ptc_initial_dtau,
        max_dt=config.ptc_max_dtau,
        max_backtracks=config.max_backtracks,
        linear_restart=config.linear_restart,
        linear_max_restarts=config.linear_max_restarts,
    )


def _continuation_config(config: PolishConfig) -> Any:
    ContinuationConfig, _, _, _, _ = _solvax_continuation_api()
    return ContinuationConfig(
        target=1.0,
        initial_step=config.alpha_initial_step,
        min_step=config.alpha_min_step,
        max_step=config.alpha_max_step,
        max_stages=config.max_continuation_stages,
    )


def _branch_tangent(
    vector: jax.Array,
    alpha: float,
    runtime: StrongRootRuntime,
    config: PolishConfig,
    previous: tuple[jax.Array, jax.Array] | None,
    block_preconditioner: StrongModeBlockPreconditioner | None = None,
    chart: StrongPhysicalChart | None = None,
) -> tuple[jax.Array, jax.Array]:
    residual = lambda value: _solve_residual(  # noqa: E731
        value, runtime, alpha, chart
    )
    _, jvp = jax.linearize(residual, vector)
    parameter_direction = _solve_residual(
        vector, runtime, 1.0, chart
    ) - _solve_residual(vector, runtime, 0.0, chart)
    if previous is None:
        linear = gmres(
            jvp,
            -parameter_direction,
            precond=(
                (lambda value: _solve_low_inverse(value, runtime, chart))
                if block_preconditioner is None
                else lambda value: block_preconditioner.apply(value, alpha)
            ),
            restart=config.linear_restart,
            rtol=min(1.0e-6, config.tolerance),
            atol=config.tolerance,
            max_restarts=config.linear_max_restarts,
        )
        linear_x = linear.x
        linear_alpha = jnp.asarray(1.0, dtype=jnp.asarray(linear.x).dtype)
    else:
        previous_x, previous_alpha = previous

        def bordered(value):
            tangent_x, tangent_alpha = value
            physical = jax.tree.map(
                jnp.add,
                jvp(tangent_x),
                jax.tree.map(lambda item: item * tangent_alpha, parameter_direction),
            )
            normalization = (
                jnp.vdot(previous_x, tangent_x).real
                + previous_alpha * tangent_alpha
            )
            return physical, normalization

        linear = gmres(
            bordered,
            (jnp.zeros_like(vector), jnp.asarray(1.0, dtype=vector.dtype)),
            precond=lambda rhs: _bordered_preconditioner(
                runtime, previous, block_preconditioner, chart
            )((vector, jnp.asarray(alpha)), rhs, jnp.inf),
            restart=config.linear_restart,
            rtol=min(1.0e-6, config.tolerance),
            atol=config.tolerance,
            max_restarts=config.linear_max_restarts,
        )
        linear_x, linear_alpha = linear.x
    if not bool(linear.converged):
        raise StrongForceContinuationError(
            "pseudo-arclength tangent solve did not converge",
            hint="refine the radial representation or increase the linear budget",
            alpha=float(alpha),
            residual_norm=float(linear.residual_norm),
            linear_iterations=int(linear.iterations),
        )
    tangent_x = jnp.asarray(linear_x)
    tangent_alpha = jnp.asarray(linear_alpha, dtype=tangent_x.dtype)
    norm = jnp.sqrt(jnp.vdot(tangent_x, tangent_x).real + tangent_alpha**2)
    tangent_x, tangent_alpha = tangent_x / norm, tangent_alpha / norm
    if previous is not None:
        orientation = jnp.vdot(tangent_x, previous[0]).real + tangent_alpha * previous[1]
        sign = jnp.where(orientation < 0.0, -1.0, 1.0)
        tangent_x, tangent_alpha = sign * tangent_x, sign * tangent_alpha
    return tangent_x, tangent_alpha


def _bordered_preconditioner(
    runtime: StrongRootRuntime,
    tangent: tuple[jax.Array, jax.Array],
    block_preconditioner: StrongModeBlockPreconditioner | None = None,
    chart: StrongPhysicalChart | None = None,
):
    """Return a low-order block-elimination preconditioner for a bordered root."""

    def apply(state, rhs, dtau):
        return _apply_bordered_preconditioner(
            state,
            rhs,
            dtau,
            tangent,
            runtime,
            block_preconditioner,
            chart,
        )

    return apply


def _apply_bordered_preconditioner(
    state,
    rhs,
    dtau,
    tangent,
    runtime: StrongRootRuntime,
    block_preconditioner: StrongModeBlockPreconditioner | None = None,
    chart: StrongPhysicalChart | None = None,
):
    """Apply bordered block elimination with dynamic branch data."""

    vector, alpha = state
    rhs_x, rhs_alpha = rhs
    tangent_x, tangent_alpha = tangent
    parameter_direction = _solve_residual(
        vector, runtime, 1.0, chart
    ) - _solve_residual(vector, runtime, 0.0, chart)
    inverse = (
        (lambda value: _solve_low_inverse(value, runtime, chart))
        if block_preconditioner is None
        else lambda value: block_preconditioner.apply(value, alpha, dtau)
    )
    q_rhs = inverse(rhs_x)
    q_parameter = inverse(parameter_direction)
    schur = tangent_alpha - jnp.vdot(tangent_x, q_parameter).real
    tiny = jnp.sqrt(jnp.finfo(jnp.asarray(schur).dtype).eps)
    safe_schur = jnp.where(
        jnp.abs(schur) > tiny,
        schur,
        jnp.where(schur < 0.0, -tiny, tiny),
    )
    delta_alpha = (
        rhs_alpha - jnp.vdot(tangent_x, q_rhs).real
    ) / safe_schur
    return q_rhs - q_parameter * delta_alpha, delta_alpha


def _arclength_to_target(
    vector: jax.Array,
    alpha: float,
    runtime: StrongRootRuntime,
    config: PolishConfig,
    admissible,
    block_preconditioner: StrongModeBlockPreconditioner | None,
    initial_tangent: tuple[jax.Array, jax.Array] | None,
    chart: StrongPhysicalChart | None = None,
):
    _, _, _, pseudo_arclength_corrector, pseudo_transient_continuation = (
        _solvax_continuation_api()
    )
    tangent = (
        _branch_tangent(
            vector,
            alpha,
            runtime,
            config,
            None,
            block_preconditioner,
            chart,
        )
        if initial_tangent is None
        else initial_tangent
    )
    solve_size = runtime.layout.size if chart is None else chart.size
    residual_scale = np.sqrt(float(solve_size)) / float(
        runtime.operator_balance
    )
    nonlinear = _ptc_config(config, residual_scale=residual_scale)
    total_nonlinear = total_linear = total_evaluations = 0
    arclength_residual = lambda value, parameter: _solve_residual(  # noqa: E731
        value, runtime, parameter, chart
    )
    arclength_admissible = lambda value, parameter: admissible(  # noqa: E731
        value, parameter
    )
    parameterized_precondition = (  # noqa: E731
        lambda state, rhs, dtau, branch_tangent, predictor: (
            _apply_bordered_preconditioner(
                state,
                rhs,
                dtau,
                branch_tangent,
                runtime,
                block_preconditioner,
                chart,
            )
        )
    )
    for step in range(config.max_arclength_steps):
        predictor = (
            vector + config.arclength_step * tangent[0],
            jnp.asarray(alpha) + config.arclength_step * tangent[1],
        )
        if _supports_keyword(pseudo_arclength_corrector, "parameterized_precond"):
            corrector_kwargs = {
                "parameterized_precond": parameterized_precondition
            }
        elif _supports_keyword(pseudo_arclength_corrector, "precond"):
            corrector_kwargs = {
                "precond": _bordered_preconditioner(
                    runtime, tangent, block_preconditioner, chart
                )
            }
        else:
            corrector_kwargs = {}
        corrected = pseudo_arclength_corrector(
            arclength_residual,
            predictor,
            tangent=tangent,
            predictor=predictor,
            config=nonlinear,
            admissible=arclength_admissible,
            **corrector_kwargs,
        )
        total_nonlinear += int(corrected.steps)
        total_linear += int(corrected.linear_iterations)
        total_evaluations += _residual_evaluations(corrected)
        if not bool(corrected.converged) or not bool(corrected.linear_converged):
            return vector, alpha, step, total_nonlinear, total_linear, total_evaluations
        previous_alpha = alpha
        vector, alpha_array = corrected.x
        alpha = float(alpha_array)
        if (previous_alpha - 1.0) * (alpha - 1.0) <= 0.0:
            target = pseudo_transient_continuation(
                lambda value: _solve_residual(value, runtime, 1.0, chart),
                vector,
                precond=(
                    (
                        lambda state, rhs, dtau: _solve_low_inverse(
                            rhs, runtime, chart
                        )
                    )
                    if block_preconditioner is None
                    else lambda state, rhs, dtau: block_preconditioner.apply(
                        rhs, 1.0, dtau
                    )
                ),
                admissible=lambda value: admissible(value, 1.0),
                config=nonlinear,
            )
            total_nonlinear += int(target.steps)
            total_linear += int(target.linear_iterations)
            total_evaluations += _residual_evaluations(target)
            if bool(target.converged) and bool(target.linear_converged):
                return (
                    target.x,
                    1.0,
                    step + 1,
                    total_nonlinear,
                    total_linear,
                    total_evaluations,
                )
        tangent = _branch_tangent(
            vector,
            alpha,
            runtime,
            config,
            tangent,
            block_preconditioner,
            chart,
        )
    return (
        vector,
        alpha,
        config.max_arclength_steps,
        total_nonlinear,
        total_linear,
        total_evaluations,
    )


def polish_strong_root(
    runtime: StrongRootRuntime,
    *,
    config: PolishConfig | None = None,
    initial_certificate: StrongForceReport | None = None,
    chart: StrongPhysicalChart | None = None,
) -> PolishResult:
    """Follow the legacy-connected fixed-boundary branch to strong force.

    The square homotopy route, kept for rank and branch diagnostics; the
    public polishing path is :func:`polish_collocation_least_squares`.  The
    residual is the convex interpolation ``(1 - alpha) * low + alpha * strong``
    between the stored legacy raw-force endpoint and the strong-force
    endpoint.  The driver solves the ``alpha = 0`` endpoint with
    pseudo-transient continuation (skipping it when the endpoint residual is
    already at roundoff before row equilibration), runs SOLVAX adaptive
    continuation to ``alpha = 1``, and, if that stalls and
    ``use_pseudo_arclength`` is set, retries with bordered pseudo-arclength
    steps that can pass a fold in ``alpha``.  Every trial state must keep a
    signed Jacobian above the configured margin, so the branch cannot walk
    through a surface overlap.  Reaching ``alpha = 1`` is not acceptance: the
    corrected state is then handed to the independent certificate.

    Parameters
    ----------
    runtime:
        Prepared strong-root runtime from
        :func:`~vmex.core.polish.make_strong_root_runtime`.  It fixes the
        collocation grid, the constrained root layout, the row and column
        equilibration, the transfer to the legacy packing, and the low-order
        preconditioner.  ``runtime.native`` is the state being corrected.
    config:
        Controls; ``None`` uses ``PolishConfig()``.  This route reads the
        continuation, PTC, arclength, preconditioner, and Jacobian-margin
        knobs that the Gauss--Newton route ignores.
    initial_certificate:
        Certificate of ``runtime.native``, if the caller already has one.
        ``None`` evaluates it here.  When it already meets
        ``certificate_tolerance`` the driver returns immediately with
        ``termination_reason="already-certified"`` and a zero correction.
    chart:
        Optional gauge-free physical chart.  ``None`` solves in the full
        constrained root layout with ``strong_root_residual``; passing a chart
        solves the projected square system in ``chart.size`` coordinates
        instead, via ``strong_physical_residual``.

    Returns
    -------
    A :class:`PolishResult`.  Its ``context`` is always ``None`` — this route
    does not build the least-squares stationarity context that the implicit
    derivative entry points require.

    Raises
    ------
    StrongForceContinuationError
        If the branch does not reach ``alpha = 1`` and ``fail_policy`` is
        ``"raise"``.
    StrongForceCertificationError
        If ``alpha = 1`` is reached but the independent certificate exceeds
        ``certificate_tolerance`` and ``fail_policy`` is ``"raise"``.
    """

    config = PolishConfig() if config is None else config
    started = perf_counter()
    initial_certificate = (
        certify_strong_force(runtime.native)
        if initial_certificate is None
        else initial_certificate
    )
    solve_size = runtime.layout.size if chart is None else chart.size
    zero = jnp.zeros(
        (solve_size,), dtype=jnp.asarray(runtime.native.R_cos).dtype
    )
    full_zero = jnp.zeros((runtime.layout.size,), dtype=zero.dtype)
    if not _failed_certificate_checks(initial_certificate, config):
        report = PolishReport(
            converged=True,
            termination_reason="already-certified",
            final_alpha=1.0,
            initial_normalized_l2=float(initial_certificate.normalized_l2),
            final_normalized_l2=float(initial_certificate.normalized_l2),
            continuation_accepted=0,
            continuation_rejected=0,
            nonlinear_iterations=0,
            linear_iterations=0,
            residual_evaluations=0,
            arclength_steps=0,
            minimum_signed_jacobian=float(initial_certificate.minimum_signed_jacobian),
            factor_build_seconds=runtime.low_preconditioner.factor_build_seconds,
            solve_seconds=perf_counter() - started,
            radial_refinement_tolerance=config.radial_refinement_tolerance,
            **_normalization_fields(initial_certificate),
        )
        return PolishResult(runtime.native, initial_certificate, report, full_zero)
    _, _, adaptive_continuation, _, pseudo_transient_continuation = (
        _solvax_continuation_api()
    )
    initial_margin = float(_minimum_signed_jacobian(zero, runtime, chart))
    if config.preconditioner == "mode-block":
        block_preconditioner = _build_mode_block_preconditioner(runtime, chart)
    elif config.preconditioner == "none":
        block_preconditioner = _IdentityPreconditioner()
    else:
        block_preconditioner = None
    factor_build_seconds = (
        runtime.low_preconditioner.factor_build_seconds
        + (0.0 if block_preconditioner is None else block_preconditioner.build_seconds)
    )
    margin_floor = max(
        config.minimum_jacobian_floor,
        config.minimum_jacobian_ratio * initial_margin,
    )

    def admissible(vector, alpha):
        del alpha
        residual = _solve_residual(vector, runtime, 1.0, chart)
        return (
            jnp.all(jnp.isfinite(vector))
            & jnp.all(jnp.isfinite(residual))
            & (_minimum_signed_jacobian(vector, runtime, chart) >= margin_floor)
        )

    residual_scale = np.sqrt(float(solve_size)) / float(
        runtime.operator_balance
    )
    nonlinear = _ptc_config(config, residual_scale=residual_scale)
    precondition = (  # noqa: E731
        (lambda state, rhs, dtau: rhs)
        if config.preconditioner == "none"
        else lambda state, rhs, dtau: _solve_low_inverse(rhs, runtime, chart)
    )
    # The low homotopy endpoint subtracts the stored legacy defect, so zero is
    # its mathematical root.  Roundoff from restrict/project/prolong may leave
    # a tiny row-equilibrated remainder.  Check that remainder before row
    # scaling and avoid asking PTC to reduce it below floating-point noise.
    # A genuinely inconsistent endpoint still takes the globalized solve.
    endpoint_residual = _solve_residual(zero, runtime, 0.0, chart)
    endpoint_solved = float(
        _normalized_low_residual_norm(endpoint_residual, runtime, chart)
    ) <= config.tolerance
    if endpoint_solved:
        vector = zero
        nonlinear_iterations = 0
        linear_iterations = 0
        residual_evaluations = 1
        converged = True
    else:
        endpoint = pseudo_transient_continuation(
            lambda vector: _solve_residual(vector, runtime, 0.0, chart),
            zero,
            precond=precondition,
            admissible=lambda vector: admissible(vector, 0.0),
            config=nonlinear,
        )
        vector = endpoint.x
        nonlinear_iterations = int(endpoint.steps)
        linear_iterations = int(endpoint.linear_iterations)
        residual_evaluations = 1 + _residual_evaluations(endpoint)
        converged = bool(endpoint.converged) and bool(endpoint.linear_converged)
    steps: tuple[Any, ...] = ()
    arclength_steps = 0
    alpha = 0.0
    reason = "alpha-zero-failed"
    if converged:
        accepted_states: list[tuple[jax.Array, float]] = [(vector, 0.0)]

        def record_accepted_state(candidate, parameter, solution):
            del solution
            accepted_states.append((candidate, float(parameter)))
            return True

        # Kwargs for adaptive_continuation: the two spellings carry
        # different callable arities, so the annotation is deliberately Any.
        continuation_preconditioners: dict[str, Any] = (
            {"precond": precondition}
            if block_preconditioner is None
            or not _supports_keyword(
                adaptive_continuation, "parameterized_precond"
            )
            else {
                "parameterized_precond": (
                    lambda state, rhs, dtau, parameter: _continuation_precondition(
                        rhs,
                        parameter,
                        dtau,
                        runtime,
                        block_preconditioner,
                        chart,
                    )
                )
            }
        )
        continuation = adaptive_continuation(
            lambda value, parameter: _solve_residual(
                value, runtime, parameter, chart
            ),
            vector,
            alpha0=0.0,
            nonlinear_config=nonlinear,
            continuation_config=_continuation_config(config),
            admissible=admissible,
            accept_stage=record_accepted_state,
            **continuation_preconditioners,
        )
        steps = continuation.steps
        vector, alpha = continuation.x, continuation.alpha
        nonlinear_iterations += sum(stage.nonlinear_steps for stage in steps)
        linear_iterations += sum(stage.linear_iterations for stage in steps)
        residual_evaluations += sum(_residual_evaluations(stage) for stage in steps)
        converged = continuation.converged
        reason = "strong-root" if converged else "continuation-stalled"
        if not converged and config.use_pseudo_arclength:
            initial_tangent = None
            if len(accepted_states) >= 2:
                previous_vector, previous_alpha = accepted_states[-2]
                delta_vector = vector - previous_vector
                delta_alpha = jnp.asarray(alpha - previous_alpha)
                tangent_norm = jnp.sqrt(
                    jnp.vdot(delta_vector, delta_vector).real + delta_alpha**2
                )
                initial_tangent = (
                    delta_vector / tangent_norm,
                    delta_alpha / tangent_norm,
                )
            try:
                (
                    vector,
                    alpha,
                    arclength_steps,
                    arc_nonlinear,
                    arc_linear,
                    arc_evaluations,
                ) = _arclength_to_target(
                    vector,
                    alpha,
                    runtime,
                    config,
                    admissible,
                    block_preconditioner,
                    initial_tangent,
                    chart,
                )
            except StrongForceContinuationError:
                reason = "pseudo-arclength-tangent-failed"
            else:
                nonlinear_iterations += arc_nonlinear
                linear_iterations += arc_linear
                residual_evaluations += arc_evaluations
                converged = alpha == 1.0
                reason = (
                    "pseudo-arclength" if converged else "pseudo-arclength-stalled"
                )

    accepted = sum(stage.accepted for stage in steps)
    rejected = len(steps) - accepted
    if not converged:
        report = PolishReport(
            converged=False,
            termination_reason=reason,
            final_alpha=float(alpha),
            initial_normalized_l2=float(initial_certificate.normalized_l2),
            final_normalized_l2=float(initial_certificate.normalized_l2),
            continuation_accepted=accepted,
            continuation_rejected=rejected,
            nonlinear_iterations=nonlinear_iterations,
            linear_iterations=linear_iterations,
            residual_evaluations=residual_evaluations,
            arclength_steps=arclength_steps,
            minimum_signed_jacobian=float(
                _minimum_signed_jacobian(vector, runtime, chart)
            ),
            factor_build_seconds=factor_build_seconds,
            solve_seconds=perf_counter() - started,
            **_normalization_fields(initial_certificate),
        )
        if config.fail_policy == "raise":
            raise StrongForceContinuationError(
                "strong-force continuation did not reach alpha=1",
                hint="inspect the continuation report and refine the radial representation",
                alpha=float(alpha),
                residual_norm=float(
                    jnp.linalg.norm(
                        _solve_residual(vector, runtime, alpha, chart)
                    )
                ),
                nonlinear_iterations=nonlinear_iterations,
                linear_iterations=linear_iterations,
                accepted_stages=accepted,
                rejected_stages=rejected,
            )
        return PolishResult(runtime.native, initial_certificate, report, full_zero)

    state = _corrected_state(vector, runtime, chart)
    certificate = certify_strong_force(state)
    failed_checks = _failed_certificate_checks(certificate, config)
    certified = not failed_checks
    report = PolishReport(
        converged=certified,
        termination_reason="certified" if certified else "certification-failed",
        final_alpha=1.0,
        initial_normalized_l2=float(initial_certificate.normalized_l2),
        final_normalized_l2=float(certificate.normalized_l2),
        continuation_accepted=accepted,
        continuation_rejected=rejected,
        nonlinear_iterations=nonlinear_iterations,
        linear_iterations=linear_iterations,
        residual_evaluations=residual_evaluations,
        arclength_steps=arclength_steps,
        minimum_signed_jacobian=float(certificate.minimum_signed_jacobian),
        factor_build_seconds=factor_build_seconds,
        solve_seconds=perf_counter() - started,
        radial_refinement_tolerance=config.radial_refinement_tolerance,
        **_normalization_fields(initial_certificate, certificate),
    )
    if not certified and config.fail_policy == "raise":
        raise StrongForceCertificationError(
            "strong root failed the independent force certificate: "
            + "; ".join(failed_checks),
            hint="increase radial degree/resolution and retry once",
            solver_converged=True,
            normalized_l2=float(certificate.normalized_l2),
            tolerance=config.certificate_tolerance,
            radial_refinement=float(certificate.radial_refinement_difference),
            radial_refinement_tolerance=config.radial_refinement_tolerance,
        )
    if not certified:
        return PolishResult(runtime.native, initial_certificate, report, full_zero)
    return PolishResult(
        state,
        certificate,
        report,
        _full_solve_vector(vector, chart),
    )


__all__ = [
    "PreconditionerQuality",
    "PreconditionerRefreshDecision",
    "PreconditionerRefreshPolicy",
    "PreconditionerSnapshot",
    "StrongModeBlockPreconditioner",
    "StrongProjectionDiagnostics",
    "build_strong_mode_block_preconditioner",
    "build_strong_physical_block_preconditioner",
    "make_strong_physical_chart",
    "polish_strong_root",
    "preconditioner_quality",
    "preconditioner_refresh_decision",
    "strong_physical_residual",
    "strong_projection_diagnostics",
    "strong_root_rank",
]

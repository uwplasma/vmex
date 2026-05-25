"""Reusable teaching workflow helpers for fixed-boundary optimizations.

The functions in this module are intentionally small building blocks for the
standalone examples.  Users should still construct objective lists explicitly in
the scripts, as in SIMSOPT's ``LeastSquaresProblem.from_tuples`` workflow, but
the mechanical VMEC/JAX stage setup, mode continuation, saving, and plotting
live here instead of being repeated in every example.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Callable, Sequence

import numpy as np

from ._compat import enable_x64, jnp
from .boundary import boundary_from_indata, boundary_input_from_indata
from .config import config_from_indata
from .driver import run_fixed_boundary, write_wout_from_fixed_boundary_run
from .energy import flux_profiles_from_indata
from .field import b_cartesian_from_state, signgs_from_sqrtg
from .finite_beta import (
    finite_beta_scalars_from_state,
    magnetic_well_from_vp,
    mercier_terms_from_state,
    redl_bootstrap_mismatch_from_state,
)
from .geom import eval_geom
from .init_guess import initial_guess_from_boundary
from .optimization import (
    BoundaryParamSpec,
    FixedBoundaryExactOptimizer,
    boundary_param_names,
    boundary_param_specs,
    create_x_scale,
    extend_boundary_for_max_mode,
    rebuild_indata_with_resolution,
    smooth_min_abs_iota_residual,
    truncate_indata_boundary_modes,
)
from .modes import nyquist_mode_table_from_grid, vmec_mode_table
from .quasi_isodynamic import (
    _nearest_half_mesh_indices,
    lgradb_penalty_from_state,
    max_elongation_penalty_from_state,
    mirror_ratio_penalty_from_boozer_output,
    quasi_isodynamic_residual_from_state,
)
from .quasisymmetry import quasisymmetry_ratio_residual_from_state
from .static import build_static
from .wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


enable_x64(True)


_LINE_BUFFERING_ENABLED = False


def _enable_line_buffered_output() -> None:
    """Flush optimization progress promptly when examples run through pipes."""

    global _LINE_BUFFERING_ENABLED
    if _LINE_BUFFERING_ENABLED:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True)
        except (TypeError, ValueError):
            pass
    _LINE_BUFFERING_ENABLED = True


@dataclass(frozen=True)
class StageContext:
    """Objects needed by objective callbacks for one mode-continuation stage."""

    static: object
    indata: object
    boundary_input: object
    specs: Sequence[BoundaryParamSpec]
    signgs: int
    flux: object
    pressure: object


@dataclass(frozen=True)
class ObjectiveTerm:
    """One weighted least-squares objective block.

    The callback receives ``(ctx, state)`` and returns a scalar or vector.  The
    residual minimized by the optimizer is ``weight * (value - target)``.
    ``weight`` is the internal residual multiplier; public objective tuples use
    SIMSOPT semantics and are converted to ``sqrt(tuple_weight)`` by
    :meth:`LeastSquaresProblem.from_tuples`.
    """

    name: str
    evaluate: Callable[[StageContext, object], object]
    target: float | np.ndarray = 0.0
    weight: float = 1.0
    total: Callable[[StageContext, object], object] | None = None
    track_iota: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def residual(self, ctx: StageContext, state) -> object:
        value = _as_vector(self.evaluate(ctx, state))
        target = jnp.asarray(self.target, dtype=jnp.float64)
        if int(target.ndim) == 0:
            target = jnp.full_like(value, target)
        else:
            target = jnp.ravel(target)
        return float(self.weight) * (value - target)


@dataclass(frozen=True)
class FixedBoundaryObjectiveStage:
    """Prepared optimizer and metadata for one active boundary-mode stage."""

    mode: int
    ctx: StageContext
    optimizer: FixedBoundaryExactOptimizer
    specs: Sequence[BoundaryParamSpec]
    boundary_input: object


@dataclass(frozen=True)
class BoundaryModeLimits:
    """Boundary-parameter mode limits for one optimization stage.

    ``mode`` controls VMEC spectral-resolution extension and continuation
    bookkeeping.  ``max_m`` and ``max_n`` optionally restrict the free boundary
    coefficients independently, enabling schedules such as toroidal-first
    stages with ``max_m=1`` and ``max_n=4`` before a full ``max_m=max_n=4``
    cleanup.
    """

    mode: int
    max_m: int | None = None
    max_n: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class FixedBoundaryOptimizationResult:
    """Result returned by :func:`run_fixed_boundary_objective_optimization`."""

    stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]]
    final_optimizer: FixedBoundaryExactOptimizer
    final_result: dict
    stage_modes: list[int]

    @property
    def initial_stage(self) -> tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]:
        """First mode-continuation stage record.

        The tuple is ``(mode, optimizer, params0, result)``.  Examples keep this
        explicit so users can choose which stage to save or inspect.
        """

        return self.stage_records[0]

    @property
    def final_stage(self) -> tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]:
        """Last mode-continuation stage record."""

        return self.stage_records[-1]

    @property
    def initial_optimizer(self) -> FixedBoundaryExactOptimizer:
        """Optimizer object for the first stage."""

        return self.initial_stage[1]

    @property
    def initial_params(self) -> np.ndarray:
        """Initial boundary parameter vector for the first stage."""

        return np.asarray(self.initial_stage[2], dtype=float)

    @property
    def initial_result(self) -> dict:
        """Raw optimizer result dictionary for the first stage."""

        return self.initial_stage[3]

    @property
    def initial_state(self):
        """Initial VMEC state if the optimizer stored one."""

        return self.initial_result.get("_state_initial")

    @property
    def history(self) -> dict:
        """Final optimizer history dictionary written by ``save_history``."""

        return self.final_result.get("_history_dump", {})

    @property
    def history_entries(self) -> tuple[dict, ...]:
        """Per-callback objective samples from the full solve."""

        return tuple(self.history.get("history", ()))

    @property
    def stage_histories(self) -> tuple[dict, ...]:
        """Per-stage history dictionaries in mode-continuation order."""

        return tuple(
            result.get("_history_dump", {})
            for _mode, _optimizer, _params0, result in self.stage_records
        )

    @property
    def objective_history(self) -> np.ndarray:
        """Objective values over full-solve callbacks as a NumPy array."""

        return np.asarray(
            [entry.get("objective", np.nan) for entry in self.history_entries],
            dtype=float,
        )

    @property
    def final_params(self) -> np.ndarray:
        """Optimized boundary parameter vector for the final stage."""

        return np.asarray(self.final_result["x"], dtype=float)

    @property
    def final_state(self):
        """Final VMEC state if the optimizer stored one."""

        return self.final_result.get("_state_final")

    @property
    def stage_timing_summaries(self) -> tuple[dict[str, object], ...]:
        """Small timing/iteration summaries for each stage."""

        summaries = []
        for mode, _optimizer, _params0, result in self.stage_records:
            summary = _result_timing_summary(result)
            summary["mode"] = int(mode)
            summaries.append(summary)
        return tuple(summaries)

    @property
    def timing_summary(self) -> dict[str, object]:
        """Small timing/iteration summary for reports and examples."""

        summary = _result_timing_summary(self.final_result, history=self.history)
        summary["stages"] = self.stage_timing_summaries
        return summary


@dataclass(frozen=True)
class OptimizationOutputPaths:
    """Canonical files written by fixed-boundary optimization examples."""

    initial_input: Path
    final_input: Path
    initial_wout: Path
    final_wout: Path
    history: Path

    def as_dict(self) -> dict[str, Path]:
        """Return path names in the same order used by the example reports."""

        return {
            "initial_input": self.initial_input,
            "final_input": self.final_input,
            "initial_wout": self.initial_wout,
            "final_wout": self.final_wout,
            "history": self.history,
        }


@dataclass(frozen=True)
class QIObjectiveTerm:
    """One field-quality objective that shares a Boozer/QI field evaluation."""

    name: str
    evaluate: Callable[[StageContext, object, dict], tuple[object, object]]
    qi_options: "QuasiIsodynamicOptions | None" = None

    def residual_and_total(self, ctx: StageContext, state, field: dict) -> tuple[object, object]:
        residuals, total = self.evaluate(ctx, state, field)
        return _as_vector(residuals), total


@dataclass(frozen=True)
class FixedBoundaryVMEC:
    """Small fixed-boundary optimization object used by the examples.

    This object is intentionally lighter than SIMSOPT's full ``Vmec`` graph,
    but it plays the same role in the example workflow: it owns the VMEC input
    deck, resolution policy, active boundary parameterization, and output path.
    Objective objects are then assembled into a :class:`LeastSquaresProblem`
    and solved by :func:`least_squares_solve`.
    """

    input_file: Path
    cfg: object
    indata: object
    max_mode: int
    min_vmec_mode: int = 5
    output_dir: Path = Path("results/optimization")
    project_input_boundary_to_max_mode: bool = False
    include: tuple[str, ...] = ("rc", "zs")
    fix: tuple[str, ...] = ("rc00",)

    @classmethod
    def from_input(
        cls,
        input_file,
        *,
        max_mode: int,
        min_vmec_mode: int = 5,
        output_dir: Path | str = Path("results/optimization"),
        project_input_boundary_to_max_mode: bool = False,
        simple_seed: bool = False,
        simple_seed_perturbation: float = 1.0e-5,
        include: Sequence[str] = ("rc", "zs"),
        fix: Sequence[str] = ("rc00",),
    ) -> "FixedBoundaryVMEC":
        """Load a VMEC input file and apply the optimization resolution policy.

        ``simple_seed=True`` replaces the boundary by the standard
        near-circular three-coefficient seed plus deterministic tiny active
        higher-mode perturbations.  This is useful for stress-testing whether
        QA/QH/QP/QI examples can leave the zero-transform branch without
        changing the raw input deck on disk.
        """

        from . import load_config
        from .config import config_from_indata

        input_path = Path(input_file)
        cfg, indata = load_config(str(input_path))
        indata = rebuild_for_optimization_resolution(
            indata,
            max_mode=max_mode,
            min_vmec_mode=min_vmec_mode,
        )
        if bool(simple_seed):
            indata = simple_omnigenity_seed_indata(
                indata,
                max_mode=max_mode,
                include=include,
                fix=fix,
                perturbation=simple_seed_perturbation,
            )
        return cls(
            input_file=input_path,
            cfg=config_from_indata(indata),
            indata=indata,
            max_mode=int(max_mode),
            min_vmec_mode=int(min_vmec_mode),
            output_dir=Path(output_dir),
            project_input_boundary_to_max_mode=bool(project_input_boundary_to_max_mode),
            include=tuple(include),
            fix=tuple(fix),
        )


@dataclass(frozen=True)
class QuasiIsodynamicOptions:
    """Boozer/QI sampling options shared by QI objective terms."""

    surfaces: object
    mboz: int = 18
    nboz: int = 18
    nphi: int = 151
    nalpha: int = 31
    n_bounce: int = 51
    include_bounce_endpoints: bool = False
    softness: float = 2.0e-2
    width_weight: float = 1.0
    branch_width_weight: float = 0.5
    branch_width_softness: float = 2.0e-2
    profile_weight: float = 0.1
    shuffle_profile_weight: float = 1.0
    shuffle_profile_softness: float = 2.0e-2
    shuffle_profile_nphi_out: int | None = None
    weighted_shuffle_profile_weight: float = 0.0
    weighted_shuffle_profile_softness: float = 2.0e-2
    aligned_profile_weight: float = 0.0
    aligned_profile_softness: float = 2.0e-2
    aligned_profile_trap_level: float = 0.65
    aligned_profile_trap_softness: float = 5.0e-2
    phimin: float = 0.0
    jit_booz: bool = True


@dataclass
class _LeastSquaresProblemAssembly:
    """Mutable accumulator for SIMSOPT-style objective tuples."""

    objective_terms: list[ObjectiveTerm] = field(default_factory=list)
    qi_objective_terms: list[QIObjectiveTerm] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    qi_options: QuasiIsodynamicOptions | None = None

    def add_tuple(self, fn: Callable, target: float | np.ndarray, weight: float) -> None:
        tuple_weight = float(weight)
        if not math.isfinite(tuple_weight) or tuple_weight < 0.0:
            raise ValueError("Least-squares tuple weights must be finite and non-negative.")
        residual_weight = math.sqrt(tuple_weight)
        owner = getattr(fn, "__self__", None)
        if getattr(owner, "requires_qi_field", False):
            self._add_qi_field_objective(owner, target, residual_weight)
        elif hasattr(owner, "to_objective_term"):
            self._add_state_objective(owner, target, residual_weight)
        else:
            self._add_plain_callable(fn, target, residual_weight)

    def _add_qi_field_objective(self, owner, target: float | np.ndarray, residual_weight: float) -> None:
        if not _target_is_zero(target):
            raise ValueError("QI field objectives currently require target=0.")
        qi_term = owner.to_qi_term(residual_weight)
        if qi_term.qi_options is not None:
            if self.qi_options is not None and qi_term.qi_options is not self.qi_options:
                raise ValueError("QI field objectives in one problem must share one QuasiIsodynamicOptions object.")
            self.qi_options = qi_term.qi_options
        self.qi_objective_terms.append(qi_term)

    def _add_state_objective(self, owner, target: float | np.ndarray, residual_weight: float) -> None:
        term = owner.to_objective_term(target=target, residual_weight=residual_weight)
        self.metadata.update(term.metadata)
        self.objective_terms.append(term)

    def _add_plain_callable(self, fn: Callable, target: float | np.ndarray, residual_weight: float) -> None:
        name = getattr(fn, "__name__", "objective")
        self.objective_terms.append(
            ObjectiveTerm(
                name,
                lambda ctx, state, fn=fn: fn(ctx, state),
                target=target,
                weight=residual_weight,
            )
        )


@dataclass(frozen=True)
class LeastSquaresProblem:
    """Least-squares objective assembled from ``(function, target, weight)`` tuples.

    As in SIMSOPT, tuple ``weight`` is an objective weight.  Internally the
    residual is ``sqrt(weight) * (function - target)``.  The examples pass
    objective ``.J`` methods here, which keeps physics targets and weights in
    the tuple list rather than in the driver/solver call.
    """

    objective_terms: tuple[ObjectiveTerm, ...] = ()
    qi_objective_terms: tuple[QIObjectiveTerm, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    qi_options: QuasiIsodynamicOptions | None = None

    @classmethod
    def from_tuples(cls, tuples: Sequence[tuple[Callable, float | np.ndarray, float]]):
        """Create a problem from SIMSOPT-style ``(J, target, weight)`` tuples.

        ``J`` is usually an objective object's ``.J`` method, such as
        ``(AspectRatio().J, 6.0, 1.0)``.  Plain callables are also accepted when
        they use the workflow signature ``J(ctx, state)``.  ``weight`` is the
        least-squares objective weight, not the residual multiplier.
        """

        assembly = _LeastSquaresProblemAssembly()
        for item in tuples:
            if len(item) != 3:
                raise ValueError("Least-squares objective tuples must be (callable, target, weight).")
            fn, target, weight = item
            if not callable(fn):
                raise TypeError("Least-squares objective tuple first entry must be callable.")
            assembly.add_tuple(fn, target, weight)
        return cls(
            tuple(assembly.objective_terms),
            tuple(assembly.qi_objective_terms),
            metadata=assembly.metadata,
            qi_options=assembly.qi_options,
        )

    @property
    def is_qi(self) -> bool:
        """Whether the problem contains Boozer-space QI field objectives."""

        return bool(self.qi_objective_terms)


@dataclass(frozen=True)
class AugmentedLagrangianConstraint:
    """Wrap a non-negative violation objective as an augmented-Lagrangian term.

    The wrapped objective should expose a signed constraint residual ``g(x)``
    with feasibility ``g(x) <= 0``.  :class:`MirrorRatio` and
    :class:`MaxElongation` provide this signed form automatically through their
    constraint hooks while preserving their usual non-negative penalty behavior
    when used as ordinary least-squares terms.

    For an inequality constraint ``g(x) <= 0`` this wrapper adds the projected
    Powell-Hestenes-Rockafellar residual

    ``sqrt(mu) * max(g(x) + lambda / mu, 0)``

    where ``lambda`` is the current multiplier and ``mu`` is the penalty.  The
    constant term in the augmented Lagrangian is omitted because it does not
    affect minimizers.  Update multipliers only from exact accepted diagnostics
    using :meth:`updated`.
    """

    objective: object
    multiplier: float = 0.0
    penalty: float = 1.0
    softness: float = 0.0
    name: str | None = None

    @property
    def requires_qi_field(self) -> bool:
        return bool(getattr(self.objective, "requires_qi_field", False))

    def J(self, _ctx: StageContext, _state):
        raise RuntimeError("AugmentedLagrangianConstraint must be assembled through LeastSquaresProblem.")

    def updated(
        self,
        violation: float,
        *,
        penalty_growth: float = 1.0,
        max_penalty: float | None = None,
    ) -> "AugmentedLagrangianConstraint":
        """Return a copy with the inequality multiplier updated.

        ``violation`` should be ``max(g(x), 0)`` measured from the exact
        accepted final state, not a trial-point residual.  The multiplier is
        projected to be non-negative.
        """

        penalty = float(self.penalty)
        violation = max(float(violation), 0.0)
        multiplier = max(0.0, float(self.multiplier) + penalty * violation)
        new_penalty = penalty * float(penalty_growth)
        if max_penalty is not None:
            new_penalty = min(new_penalty, float(max_penalty))
        return AugmentedLagrangianConstraint(
            objective=self.objective,
            multiplier=multiplier,
            penalty=new_penalty,
            softness=self.softness,
            name=self.name,
        )

    def _projected_residual(self, residual):
        penalty = max(float(self.penalty), np.finfo(float).tiny)
        shifted = jnp.asarray(residual, dtype=jnp.float64) + float(self.multiplier) / penalty
        projected = _smooth_positive_part(shifted, softness=float(self.softness))
        return jnp.sqrt(jnp.asarray(penalty, dtype=jnp.float64)) * projected

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        if not _target_is_zero(target):
            raise ValueError("AugmentedLagrangianConstraint objective tuples require target=0.")
        if hasattr(self.objective, "to_constraint_term"):
            base = self.objective.to_constraint_term()
        elif hasattr(self.objective, "to_objective_term"):
            # Backward-compatible fallback for objectives that already return a
            # non-negative violation residual.  New constrained objectives
            # should prefer a signed ``to_constraint_term`` hook.
            base = self.objective.to_objective_term(target=0.0, residual_weight=1.0)
        else:
            raise ValueError("Wrapped augmented-Lagrangian objective must expose to_objective_term().")
        name = self.name or f"al_{base.name}"

        def _evaluate(ctx, state, base=base):
            return self._projected_residual(base.residual(ctx, state)) * float(residual_weight)

        def _total(ctx, state):
            residual = _evaluate(ctx, state)
            return jnp.sum(residual * residual)

        return ObjectiveTerm(
            name,
            _evaluate,
            target=0.0,
            weight=1.0,
            total=_total,
            track_iota=base.track_iota,
            metadata=dict(base.metadata),
        )

    def to_qi_term(self, residual_weight: float) -> QIObjectiveTerm:
        if hasattr(self.objective, "to_constraint_qi_term"):
            base = self.objective.to_constraint_qi_term()
        elif hasattr(self.objective, "to_qi_term"):
            # Backward-compatible fallback for QI objectives that already
            # return a non-negative violation residual.  New constrained QI
            # objectives should prefer a signed ``to_constraint_qi_term`` hook.
            base = self.objective.to_qi_term(1.0)
        else:
            raise ValueError("Wrapped augmented-Lagrangian QI objective must expose to_qi_term().")
        name = self.name or f"al_{base.name}"

        def _evaluate(ctx, state, field, base=base):
            residual, _total = base.residual_and_total(ctx, state, field)
            projected = self._projected_residual(residual) * float(residual_weight)
            return projected, jnp.sum(projected * projected)

        return QIObjectiveTerm(name, _evaluate, qi_options=base.qi_options)


class AspectRatio:
    """Aspect-ratio objective object."""

    name = "aspect"

    def J(self, ctx: StageContext, state):
        return equilibrium_aspect_ratio_from_state(state=state, static=ctx.static)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        return ObjectiveTerm(
            self.name,
            self.J,
            target=target,
            weight=residual_weight,
            metadata={"target_aspect": float(target)},
        )


class MeanIota:
    """Mean rotational-transform objective object."""

    name = "iota"

    def J(self, ctx: StageContext, state):
        return mean_iota(ctx, state)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        return ObjectiveTerm(
            self.name,
            self.J,
            target=target,
            weight=residual_weight,
            track_iota=True,
            metadata={"target_iota": float(target)},
        )


class AbsMeanIotaFloor:
    """Smooth lower-bound objective for ``abs(mean_iota)``."""

    name = "abs_iota_floor"

    def __init__(self, target: float, *, softness: float = 1.0e-3):
        self.target = float(target)
        self.softness = float(softness)

    def J(self, ctx: StageContext, state):
        return smooth_min_abs_iota_residual(mean_iota(ctx, state), self.target, softness=self.softness)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        del target
        return ObjectiveTerm(
            self.name,
            self.J,
            target=0.0,
            weight=residual_weight,
            track_iota=True,
            metadata={"iota_abs_min": self.target},
        )


class AbsMeanIotaCeiling:
    """Smooth upper-bound objective for ``abs(mean_iota)``."""

    name = "abs_iota_ceiling"

    def __init__(self, maximum: float, *, softness: float = 1.0e-3):
        self.maximum = float(maximum)
        self.softness = float(softness)

    def J(self, ctx: StageContext, state):
        return abs_mean_iota_ceiling_objective(
            self.maximum,
            weight=1.0,
            softness=self.softness,
        ).evaluate(ctx, state)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        del target
        term = abs_mean_iota_ceiling_objective(
            self.maximum,
            weight=residual_weight,
            softness=self.softness,
        )
        return ObjectiveTerm(
            self.name,
            term.evaluate,
            target=0.0,
            weight=1.0,
            track_iota=True,
            metadata={"iota_abs_max": self.maximum},
        )


class QuasisymmetryRatioResidual:
    """QS residual object for QA/QH/QP objectives."""

    name = "qs"

    def __init__(self, *, helicity_m: int, helicity_n: int, surfaces):
        self.helicity_m = int(helicity_m)
        self.helicity_n = int(helicity_n)
        self.surfaces = surfaces

    def _evaluate(self, ctx: StageContext, state):
        return quasisymmetry_ratio_residual_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            flux_local=ctx.flux,
            prof_local={"pressure": ctx.pressure},
            pressure_local=ctx.pressure,
            surfaces=self.surfaces,
            helicity_m=self.helicity_m,
            helicity_n=self.helicity_n,
        )

    def J(self, ctx: StageContext, state):
        return self._evaluate(ctx, state)["residuals1d"]

    def total(self, ctx: StageContext, state):
        return self._evaluate(ctx, state)["total"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        if not _target_is_zero(target):
            raise ValueError("Quasisymmetry residual objectives require target=0.")
        return ObjectiveTerm(
            self.name,
            self.J,
            target=0.0,
            weight=residual_weight,
            total=lambda ctx, state: float(residual_weight) ** 2 * self.total(ctx, state),
        )


class QuasiIsodynamicResidual:
    """Smooth QI residual object using a shared Boozer field evaluation."""

    name = "qi"
    requires_qi_field = True

    def __init__(self, options: QuasiIsodynamicOptions):
        self.options = options

    def J(self, _ctx: StageContext, _state):
        raise RuntimeError("QuasiIsodynamicResidual must be evaluated inside a QI solve.")

    def to_qi_term(self, residual_weight: float) -> QIObjectiveTerm:
        return quasi_isodynamic_field_objective(weight=residual_weight, qi_options=self.options)


class QuasiIsodynamicResidualCeiling:
    """Soft upper-bound objective for preserving a low-QI basin during cleanup."""

    name = "qi_ceiling"
    requires_qi_field = True

    def __init__(
        self,
        *,
        maximum: float,
        smooth_penalty: float = 0.0,
        qi_options: QuasiIsodynamicOptions | None = None,
    ):
        self.maximum = float(maximum)
        self.smooth_penalty = float(smooth_penalty)
        self.qi_options = qi_options

    def J(self, _ctx: StageContext, _state):
        raise RuntimeError("QuasiIsodynamicResidualCeiling must be evaluated inside a QI solve.")

    def to_qi_term(self, residual_weight: float) -> QIObjectiveTerm:
        return qi_residual_ceiling_objective(
            maximum=self.maximum,
            weight=residual_weight,
            smooth_penalty=self.smooth_penalty,
            qi_options=self.qi_options,
        )


class MirrorRatio:
    """Maximum mirror-ratio penalty object for QI solves."""

    name = "mirror_ratio"
    requires_qi_field = True

    def __init__(
        self,
        *,
        threshold: float,
        ntheta: int = 96,
        nphi: int = 96,
        surface_index: int | None = None,
        phimin: float = 0.0,
        smooth_extrema: float = 0.0,
        smooth_penalty: float = 0.0,
        normalize_surfaces: bool = True,
        qi_options: QuasiIsodynamicOptions | None = None,
    ):
        self.threshold = float(threshold)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)
        self.surface_index = None if surface_index is None else int(surface_index)
        self.phimin = float(phimin)
        self.smooth_extrema = float(smooth_extrema)
        self.smooth_penalty = float(smooth_penalty)
        self.normalize_surfaces = bool(normalize_surfaces)
        self.qi_options = qi_options

    def J(self, _ctx: StageContext, _state):
        raise RuntimeError("MirrorRatio must be evaluated inside a QI solve.")

    def to_qi_term(self, residual_weight: float) -> QIObjectiveTerm:
        return qi_mirror_ratio_objective(
            threshold=self.threshold,
            weight=residual_weight,
            ntheta=self.ntheta,
            nphi=self.nphi,
            surface_index=self.surface_index,
            phimin=self.phimin,
            smooth_extrema=self.smooth_extrema,
            smooth_penalty=self.smooth_penalty,
            normalize_surfaces=self.normalize_surfaces,
            qi_options=self.qi_options,
        )

    def to_constraint_qi_term(self) -> QIObjectiveTerm:
        return qi_mirror_ratio_constraint(
            threshold=self.threshold,
            ntheta=self.ntheta,
            nphi=self.nphi,
            surface_index=self.surface_index,
            phimin=self.phimin,
            smooth_extrema=self.smooth_extrema,
            normalize_surfaces=self.normalize_surfaces,
            qi_options=self.qi_options,
        )


class BoozerBTarget:
    """Boozer ``|B|`` spectrum-matching objective for QI steering.

    This term is intended as a differentiable homotopy/steering objective, not
    as a final QI diagnostic.  It compares the current Boozer ``|B|`` spectrum
    against a reference spectrum on the same Boozer mode grid.  By default each
    surface is normalized by its ``(m,n)=(0,0)`` coefficient so the term matches
    field shape rather than absolute field strength.
    """

    name = "boozer_b_target"
    requires_qi_field = True

    def __init__(
        self,
        *,
        target_bmnc,
        target_bmns=None,
        normalize: bool = True,
        include_b00: bool = False,
        qi_options: QuasiIsodynamicOptions | None = None,
    ):
        self.target_bmnc = np.asarray(target_bmnc, dtype=float)
        self.target_bmns = None if target_bmns is None else np.asarray(target_bmns, dtype=float)
        self.normalize = bool(normalize)
        self.include_b00 = bool(include_b00)
        self.qi_options = qi_options

    def J(self, _ctx: StageContext, _state):
        raise RuntimeError("BoozerBTarget must be evaluated inside a QI solve.")

    def to_qi_term(self, residual_weight: float) -> QIObjectiveTerm:
        return qi_boozer_b_target_objective(
            target_bmnc=self.target_bmnc,
            target_bmns=self.target_bmns,
            weight=residual_weight,
            normalize=self.normalize,
            include_b00=self.include_b00,
            qi_options=self.qi_options,
        )


class MaxElongation:
    """Maximum LCFS elongation penalty object for QI solves."""

    name = "max_elongation"
    requires_qi_field = True

    def __init__(
        self,
        *,
        threshold: float,
        ntheta: int = 48,
        nphi: int = 16,
        smooth_extrema: float = 0.0,
        smooth_penalty: float = 0.0,
        qi_options: QuasiIsodynamicOptions | None = None,
    ):
        self.threshold = float(threshold)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)
        self.smooth_extrema = float(smooth_extrema)
        self.smooth_penalty = float(smooth_penalty)
        self.qi_options = qi_options

    def J(self, _ctx: StageContext, _state):
        raise RuntimeError("MaxElongation must be evaluated inside a QI solve.")

    def to_qi_term(self, residual_weight: float) -> QIObjectiveTerm:
        return qi_max_elongation_objective(
            threshold=self.threshold,
            weight=residual_weight,
            ntheta=self.ntheta,
            nphi=self.nphi,
            smooth_extrema=self.smooth_extrema,
            smooth_penalty=self.smooth_penalty,
            qi_options=self.qi_options,
        )

    def to_constraint_qi_term(self) -> QIObjectiveTerm:
        return qi_max_elongation_constraint(
            threshold=self.threshold,
            ntheta=self.ntheta,
            nphi=self.nphi,
            smooth_extrema=self.smooth_extrema,
            qi_options=self.qi_options,
        )


class MagneticWell:
    """Smooth lower-bound objective for the vacuum magnetic-well proxy.

    The well follows the SIMSOPT/VMEC convention
    ``(dV/ds(0) - dV/ds(1)) / dV/ds(0)`` using the differentiable half-mesh
    volume derivative reconstructed from the VMEC state.  Positive values are
    favorable; this objective returns a smooth penalty when the well falls
    below ``minimum``.
    """

    name = "magnetic_well"

    def __init__(self, *, minimum: float = 0.0, softness: float = 1.0e-3):
        self.minimum = float(minimum)
        self.softness = float(softness)

    def well(self, ctx: StageContext, state):
        scalars = finite_beta_scalars_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
        )
        return magnetic_well_from_vp(scalars["vp"])

    def J(self, ctx: StageContext, state):
        deficit = float(self.minimum) - self.well(ctx, state)
        softness = jnp.asarray(float(self.softness), dtype=jnp.float64)
        return softness * jnp.logaddexp(jnp.asarray(0.0, dtype=jnp.float64), deficit / softness)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        if not _target_is_zero(target):
            raise ValueError("MagneticWell is a lower-bound penalty and requires target=0.")
        return ObjectiveTerm(self.name, self.J, target=0.0, weight=residual_weight)


class VolavgB:
    """Volume-averaged magnetic-field objective for finite-beta studies."""

    name = "volavgB"

    def J(self, ctx: StageContext, state):
        return finite_beta_scalars_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
        )["volavgB"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class BetaTotal:
    """Total-beta objective for finite-beta studies."""

    name = "betatotal"

    def J(self, ctx: StageContext, state):
        return finite_beta_scalars_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
        )["betatotal"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class DMerc:
    """Smooth lower-bound objective for VMEC Mercier stability.

    The residual is a per-surface smooth penalty for ``DMerc < minimum`` on
    interior radial surfaces.  It uses the differentiable state-level Mercier
    path for both stellarator-symmetric and LASYM equilibria.
    """

    name = "DMerc"

    def __init__(
        self,
        *,
        minimum: float = 0.0,
        softness: float = 1.0e-3,
        mmax_force: int | None = None,
        nmax_force: int | None = None,
    ):
        self.minimum = float(minimum)
        self.softness = float(softness)
        self.mmax_force = None if mmax_force is None else int(mmax_force)
        self.nmax_force = None if nmax_force is None else int(nmax_force)

    def terms(self, ctx: StageContext, state):
        return mercier_terms_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            mmax_force=self.mmax_force,
            nmax_force=self.nmax_force,
        )

    def J(self, ctx: StageContext, state):
        dmerc = jnp.asarray(self.terms(ctx, state)["DMerc"], dtype=jnp.float64)
        active = dmerc[1:-1] if int(dmerc.shape[0]) > 2 else jnp.zeros((0,), dtype=dmerc.dtype)
        deficit = float(self.minimum) - active
        softness = jnp.asarray(float(self.softness), dtype=jnp.float64)
        return softness * jnp.logaddexp(jnp.asarray(0.0, dtype=jnp.float64), deficit / softness)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        if not _target_is_zero(target):
            raise ValueError("DMerc is a lower-bound penalty and requires target=0.")
        return ObjectiveTerm(self.name, self.J, target=0.0, weight=residual_weight)


class _MercierProfileObjective:
    """Base object for differentiable VMEC JXBFORCE profile objectives."""

    name = "mercier_profile"
    profile_key = ""

    def __init__(
        self,
        *,
        surfaces: Sequence[float] | None = None,
        normalize: float = 1.0,
        mmax_force: int | None = None,
        nmax_force: int | None = None,
    ):
        self.surfaces = None if surfaces is None else tuple(float(s) for s in surfaces)
        self.normalize = float(normalize)
        self.mmax_force = None if mmax_force is None else int(mmax_force)
        self.nmax_force = None if nmax_force is None else int(nmax_force)

    def terms(self, ctx: StageContext, state):
        return mercier_terms_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            mmax_force=self.mmax_force,
            nmax_force=self.nmax_force,
        )

    def _select_profile(self, ctx: StageContext, profile):
        profile = jnp.asarray(profile, dtype=jnp.float64)
        if self.surfaces is None:
            return profile[1:-1] if int(profile.shape[0]) > 2 else jnp.zeros((0,), dtype=profile.dtype)
        s = np.asarray(getattr(ctx.static, "s"), dtype=float)
        indices = [int(np.argmin(np.abs(s - float(surface)))) for surface in self.surfaces]
        return profile[jnp.asarray(indices, dtype=jnp.int32)]

    def J(self, ctx: StageContext, state):
        profile = self.terms(ctx, state)[self.profile_key]
        values = self._select_profile(ctx, profile)
        return values / float(self.normalize)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class JDotB(_MercierProfileObjective):
    """VMEC ``jdotb`` profile objective from the differentiable JXBFORCE path."""

    name = "jdotb"
    profile_key = "jdotb"


class BDotB(_MercierProfileObjective):
    """VMEC ``bdotb`` profile objective from the differentiable JXBFORCE path."""

    name = "bdotb"
    profile_key = "bdotb"


class BDotGradV(_MercierProfileObjective):
    """VMEC ``bdotgradv`` profile objective from the differentiable JXBFORCE path."""

    name = "bdotgradv"
    profile_key = "bdotgradv"


class BVector:
    """Cartesian magnetic-field vector objective on one radial surface.

    The residual vector is ``(Bx, By, Bz)`` flattened over ``(theta, zeta)`` on
    ``ctx.static.grid``.  ``s_index=-1`` targets the boundary surface.
    """

    name = "B_vector"

    def __init__(self, *, s_index: int = -1, normalize: float = 1.0):
        self.s_index = int(s_index)
        self.normalize = float(normalize)

    def J(self, ctx: StageContext, state):
        field = b_cartesian_from_state(
            state,
            ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            s_index=self.s_index,
        )
        return jnp.ravel(jnp.asarray(field, dtype=jnp.float64)) / float(self.normalize)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class JVector(_MercierProfileObjective):
    """Flux-coordinate current-density vector objective from JXBFORCE channels.

    The returned vector contains ``(J^theta, J^zeta) = (itheta/sqrtg,
    izeta/sqrtg)`` flattened over the selected full-mesh surfaces and angular
    grid.  It is a VMEC-coordinate current-density diagnostic, not a Cartesian
    vector.
    """

    name = "J_vector"

    def J(self, ctx: StageContext, state):
        terms = mercier_terms_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            mmax_force=self.mmax_force,
            nmax_force=self.nmax_force,
            include_channels=True,
        )
        sqrtg = jnp.asarray(terms["sqrtg"], dtype=jnp.float64)
        sqrtg_safe = jnp.where(sqrtg != 0.0, sqrtg, jnp.asarray(1.0, dtype=sqrtg.dtype))
        jtheta = jnp.where(sqrtg != 0.0, jnp.asarray(terms["itheta"], dtype=jnp.float64) / sqrtg_safe, 0.0)
        jzeta = jnp.where(sqrtg != 0.0, jnp.asarray(terms["izeta"], dtype=jnp.float64) / sqrtg_safe, 0.0)
        vector = jnp.stack([jtheta, jzeta], axis=-1)
        values = self._select_profile(ctx, vector)
        return jnp.ravel(values) / float(self.normalize)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class ToroidalCurrent(_MercierProfileObjective):
    """Integrated toroidal-current profile from VMEC's Mercier path.

    The profile key is ``torcur`` and follows VMEC's Mercier normalization:
    ``signgs * 2*pi * <B_u>`` on the full radial mesh.  This is a
    state-derived current profile, not just the prescribed input ``ICURV``.
    """

    name = "torcur"
    profile_key = "torcur"


class ToroidalCurrentGradient(_MercierProfileObjective):
    """Radial derivative of ``ToroidalCurrent`` used by VMEC Mercier terms."""

    name = "torcur_prime"
    profile_key = "ip"


class RedlBootstrapMismatch(_MercierProfileObjective):
    """Redl bootstrap-current mismatch objective for finite-beta studies.

    Polynomial profile coefficients follow SIMSOPT ``ProfilePolynomial``
    ordering.  ``ne_coeffs`` are in ``m^-3`` and ``Te_coeffs``/``Ti_coeffs`` in
    eV.  The residual block is normalized as in SIMSOPT's
    ``VmecRedlBootstrapMismatch`` objective.
    """

    name = "redl_bootstrap_mismatch"

    def __init__(
        self,
        *,
        helicity_n: int,
        ne_coeffs,
        Te_coeffs,
        Ti_coeffs=None,
        Zeff_coeffs=1.0,
        surfaces: Sequence[float] | None = None,
        n_lambda: int = 32,
        mmax_force: int | None = None,
        nmax_force: int | None = None,
    ):
        super().__init__(surfaces=surfaces, normalize=1.0, mmax_force=mmax_force, nmax_force=nmax_force)
        self.helicity_n = int(helicity_n)
        self.ne_coeffs = tuple(float(x) for x in np.ravel(np.asarray(ne_coeffs, dtype=float)))
        self.Te_coeffs = tuple(float(x) for x in np.ravel(np.asarray(Te_coeffs, dtype=float)))
        if Ti_coeffs is None:
            self.Ti_coeffs = None
        else:
            self.Ti_coeffs = tuple(float(x) for x in np.ravel(np.asarray(Ti_coeffs, dtype=float)))
        self.Zeff_coeffs = tuple(float(x) for x in np.ravel(np.atleast_1d(np.asarray(Zeff_coeffs, dtype=float))))
        self.n_lambda = int(n_lambda)

    def _evaluate(self, ctx: StageContext, state):
        return redl_bootstrap_mismatch_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            helicity_n=self.helicity_n,
            ne_coeffs=self.ne_coeffs,
            Te_coeffs=self.Te_coeffs,
            Ti_coeffs=self.Ti_coeffs,
            Zeff_coeffs=self.Zeff_coeffs,
            surfaces=self.surfaces,
            n_lambda=self.n_lambda,
            mmax_force=self.mmax_force,
            nmax_force=self.nmax_force,
        )

    def J(self, ctx: StageContext, state):
        return self._evaluate(ctx, state)["residuals1d"]

    def total(self, ctx: StageContext, state):
        return self._evaluate(ctx, state)["total"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        if not _target_is_zero(target):
            raise ValueError("RedlBootstrapMismatch is already normalized and requires target=0.")
        return ObjectiveTerm(
            self.name,
            self.J,
            target=0.0,
            weight=residual_weight,
            total=lambda ctx, state: float(residual_weight) ** 2 * self.total(ctx, state),
        )


class LgradB:
    """Minimum-``L_grad_B`` penalty object usable in QS or QI examples."""

    name = "LgradB"

    def __init__(
        self,
        *,
        threshold: float,
        s_index: int = -1,
        ntheta: int = 9,
        nphi: int = 7,
        smooth_penalty: float = 0.0,
    ):
        self.threshold = float(threshold)
        self.s_index = int(s_index)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)
        self.smooth_penalty = float(smooth_penalty)

    def _evaluate(self, ctx: StageContext, state):
        return lgradb_penalty_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            flux_local=ctx.flux,
            threshold=self.threshold,
            s_index=self.s_index,
            ntheta=self.ntheta,
            nphi=self.nphi,
            smooth_penalty=self.smooth_penalty,
        )

    def J(self, ctx: StageContext, state):
        return self._evaluate(ctx, state)["residuals1d"]

    def total(self, ctx: StageContext, state):
        return self._evaluate(ctx, state)["total"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        if not _target_is_zero(target):
            raise ValueError("LgradB penalty objectives require target=0.")
        return ObjectiveTerm(
            self.name,
            self.J,
            target=0.0,
            weight=residual_weight,
            total=lambda ctx, state: float(residual_weight) ** 2 * self.total(ctx, state),
        )

    def to_qi_term(self, residual_weight: float) -> QIObjectiveTerm:
        return qi_lgradb_objective(
            threshold=self.threshold,
            weight=residual_weight,
            s_index=self.s_index,
            ntheta=self.ntheta,
            nphi=self.nphi,
            smooth_penalty=self.smooth_penalty,
        )


def aspect_objective(target: float, weight: float = 1.0) -> ObjectiveTerm:
    """Aspect-ratio least-squares objective."""

    def _evaluate(ctx: StageContext, state):
        return equilibrium_aspect_ratio_from_state(state=state, static=ctx.static)

    return ObjectiveTerm("aspect", _evaluate, target=target, weight=weight)


def mean_iota_objective(target: float, weight: float = 1.0) -> ObjectiveTerm:
    """Mean full-mesh rotational-transform objective."""

    return ObjectiveTerm(
        "iota",
        lambda ctx, state: mean_iota(ctx, state),
        target=target,
        weight=weight,
        track_iota=True,
    )


def abs_mean_iota_floor_objective(
    target: float,
    weight: float = 1.0,
    *,
    softness: float = 1.0e-3,
) -> ObjectiveTerm:
    """Smooth lower-bound penalty enforcing ``abs(mean_iota) >= target``."""

    def _evaluate(ctx: StageContext, state):
        return smooth_min_abs_iota_residual(
            mean_iota(ctx, state),
            float(target),
            softness=float(softness),
        )

    return ObjectiveTerm(
        "abs_iota_floor",
        _evaluate,
        target=0.0,
        weight=weight,
        track_iota=True,
    )


def abs_mean_iota_ceiling_objective(
    maximum: float,
    weight: float = 1.0,
    *,
    softness: float = 1.0e-3,
    abs_epsilon: float = 1.0e-12,
) -> ObjectiveTerm:
    """Smooth upper-bound penalty enforcing ``abs(mean_iota) <= maximum``."""

    maximum = float(maximum)

    def _evaluate(ctx: StageContext, state):
        iota = jnp.asarray(mean_iota(ctx, state), dtype=jnp.float64)
        smooth_abs_iota = jnp.sqrt(iota * iota + float(abs_epsilon) ** 2)
        return _smooth_positive_part(smooth_abs_iota - maximum, softness=float(softness))

    return ObjectiveTerm(
        "abs_iota_ceiling",
        _evaluate,
        target=0.0,
        weight=weight,
        track_iota=True,
    )


def quasisymmetry_objective(
    *,
    helicity_m: int,
    helicity_n: int,
    surfaces,
    weight: float = 1.0,
) -> ObjectiveTerm:
    """Quasisymmetry residual objective for QA, QH, or QP."""

    def _qs(ctx: StageContext, state):
        return quasisymmetry_ratio_residual_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            flux_local=ctx.flux,
            prof_local={"pressure": ctx.pressure},
            pressure_local=ctx.pressure,
            surfaces=surfaces,
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
        )

    return ObjectiveTerm(
        "qs",
        lambda ctx, state: _qs(ctx, state)["residuals1d"],
        target=0.0,
        weight=weight,
        total=lambda ctx, state: float(weight) ** 2 * _qs(ctx, state)["total"],
    )


def lgradb_objective(
    *,
    threshold: float,
    weight: float = 1.0,
    s_index: int = -1,
    ntheta: int = 9,
    nphi: int = 7,
    smooth_penalty: float = 0.0,
) -> ObjectiveTerm:
    """Differentiable minimum-``L_grad_B`` penalty objective."""

    def _lgradb(ctx: StageContext, state):
        return lgradb_penalty_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            flux_local=ctx.flux,
            threshold=float(threshold),
            s_index=int(s_index),
            ntheta=int(ntheta),
            nphi=int(nphi),
            smooth_penalty=float(smooth_penalty),
        )

    return ObjectiveTerm(
        "LgradB",
        lambda ctx, state: _lgradb(ctx, state)["residuals1d"],
        target=0.0,
        weight=weight,
        total=lambda ctx, state: float(weight) ** 2 * _lgradb(ctx, state)["total"],
    )


def quasi_isodynamic_field_objective(
    weight: float = 1.0,
    qi_options: QuasiIsodynamicOptions | None = None,
) -> QIObjectiveTerm:
    """Smooth QI residual term from ``quasi_isodynamic_residual_from_state``."""

    def _evaluate(_ctx: StageContext, _state, field: dict):
        return (
            jnp.asarray(field["residuals1d"], dtype=jnp.float64) * float(weight),
            float(weight) ** 2 * field["total"],
        )

    return QIObjectiveTerm("qi", _evaluate, qi_options=qi_options)


def _smooth_positive_part(value, *, softness: float):
    value = jnp.asarray(value, dtype=jnp.float64)
    softness = float(softness)
    if softness <= 0.0:
        return jnp.maximum(value, 0.0)
    return softness * jnp.logaddexp(value / softness, 0.0)


def qi_residual_ceiling_objective(
    *,
    maximum: float,
    weight: float = 1.0,
    smooth_penalty: float = 0.0,
    qi_options: QuasiIsodynamicOptions | None = None,
) -> QIObjectiveTerm:
    """Soft-wall objective that penalizes QI residuals above ``maximum``."""

    def _evaluate(_ctx: StageContext, _state, field: dict):
        qi_total = jnp.asarray(field["total"], dtype=jnp.float64)
        excess = _smooth_positive_part(qi_total - float(maximum), softness=float(smooth_penalty))
        residual = jnp.ravel(excess) * float(weight)
        return residual, jnp.sum(residual * residual)

    return QIObjectiveTerm("qi_ceiling", _evaluate, qi_options=qi_options)


def qi_mirror_ratio_objective(
    *,
    threshold: float,
    weight: float = 1.0,
    ntheta: int = 96,
    nphi: int = 96,
    surface_index: int | None = None,
    phimin: float = 0.0,
    smooth_extrema: float = 0.0,
    smooth_penalty: float = 0.0,
    normalize_surfaces: bool = True,
    qi_options: QuasiIsodynamicOptions | None = None,
) -> QIObjectiveTerm:
    """Mirror-ratio upper-bound objective evaluated from Boozer ``|B|`` modes."""

    def _evaluate(ctx: StageContext, _state, field: dict):
        mirror_booz = field["booz"] if surface_index is None else _slice_boozer_surfaces(field["booz"], int(surface_index))
        weights = None
        if bool(normalize_surfaces) and surface_index is None:
            nsurf = int(jnp.asarray(mirror_booz["bmnc_b"]).shape[0])
            weights = [1.0 / float(max(nsurf, 1))] * nsurf
        mirror = mirror_ratio_penalty_from_boozer_output(
            mirror_booz,
            nfp=int(ctx.static.cfg.nfp),
            threshold=float(threshold),
            weights=weights,
            ntheta=int(ntheta),
            nphi=int(nphi),
            phimin=float(phimin),
            smooth_extrema=float(smooth_extrema),
            smooth_penalty=float(smooth_penalty),
        )
        return (
            jnp.asarray(mirror["residuals1d"], dtype=jnp.float64) * float(weight),
            float(weight) ** 2 * mirror["total"],
        )

    return QIObjectiveTerm("mirror_ratio", _evaluate, qi_options=qi_options)


def qi_mirror_ratio_constraint(
    *,
    threshold: float,
    ntheta: int = 96,
    nphi: int = 96,
    surface_index: int | None = None,
    phimin: float = 0.0,
    smooth_extrema: float = 0.0,
    normalize_surfaces: bool = True,
    qi_options: QuasiIsodynamicOptions | None = None,
) -> QIObjectiveTerm:
    """Signed mirror-ratio constraint ``mirror_ratio - threshold <= 0``."""

    def _evaluate(ctx: StageContext, _state, field: dict):
        mirror_booz = field["booz"] if surface_index is None else _slice_boozer_surfaces(field["booz"], int(surface_index))
        weights = None
        if bool(normalize_surfaces) and surface_index is None:
            nsurf = int(jnp.asarray(mirror_booz["bmnc_b"]).shape[0])
            weights = [1.0 / float(max(nsurf, 1))] * nsurf
        mirror = mirror_ratio_penalty_from_boozer_output(
            mirror_booz,
            nfp=int(ctx.static.cfg.nfp),
            threshold=float(threshold),
            weights=weights,
            ntheta=int(ntheta),
            nphi=int(nphi),
            phimin=float(phimin),
            smooth_extrema=float(smooth_extrema),
            smooth_penalty=0.0,
        )
        weights_arr = jnp.ones_like(jnp.asarray(mirror["mirror_ratio"], dtype=jnp.float64))
        if weights is not None:
            weights_arr = jnp.asarray(weights, dtype=jnp.float64)
        residuals = (jnp.asarray(mirror["mirror_ratio"], dtype=jnp.float64) - float(threshold)) * jnp.sqrt(
            weights_arr
        )
        return residuals, jnp.sum(jnp.maximum(residuals, 0.0) ** 2)

    return QIObjectiveTerm("mirror_ratio_constraint", _evaluate, qi_options=qi_options)


def qi_boozer_b_target_objective(
    *,
    target_bmnc,
    target_bmns=None,
    weight: float = 1.0,
    normalize: bool = True,
    include_b00: bool = False,
    qi_options: QuasiIsodynamicOptions | None = None,
) -> QIObjectiveTerm:
    """Boozer ``|B|`` spectrum target evaluated on the shared QI field."""

    target_bmnc_arr = np.asarray(target_bmnc, dtype=float)
    target_bmns_arr = None if target_bmns is None else np.asarray(target_bmns, dtype=float)

    def _evaluate(_ctx: StageContext, _state, field: dict):
        booz = field["booz"]
        bmnc = jnp.asarray(booz["bmnc_b"], dtype=jnp.float64)
        target_c = jnp.asarray(target_bmnc_arr, dtype=jnp.float64)
        if bmnc.shape != target_c.shape:
            raise ValueError(
                "BoozerBTarget target_bmnc must have the same shape as the current Boozer bmnc_b "
                f"({target_c.shape} != {bmnc.shape})."
            )

        bmns_raw = booz.get("bmns_b")
        bmns = jnp.zeros_like(bmnc) if bmns_raw is None else jnp.asarray(bmns_raw, dtype=jnp.float64)
        target_s = jnp.zeros_like(target_c) if target_bmns_arr is None else jnp.asarray(target_bmns_arr, dtype=jnp.float64)
        if target_s.shape != bmnc.shape:
            raise ValueError(
                "BoozerBTarget target_bmns must have the same shape as the current Boozer bmnc_b "
                f"({target_s.shape} != {bmnc.shape})."
            )

        if bool(normalize):
            tiny = jnp.asarray(jnp.finfo(bmnc.dtype).tiny, dtype=bmnc.dtype)
            scale = jnp.maximum(jnp.abs(bmnc[:, :1]), tiny)
            target_scale = jnp.maximum(jnp.abs(target_c[:, :1]), tiny)
            bmnc = bmnc / scale
            bmns = bmns / scale
            target_c = target_c / target_scale
            target_s = target_s / target_scale

        diff_c = bmnc - target_c
        diff_s = bmns - target_s
        if not bool(include_b00):
            diff_c = diff_c.at[:, 0].set(0.0)
            diff_s = diff_s.at[:, 0].set(0.0)
        residuals = jnp.concatenate([jnp.ravel(diff_c), jnp.ravel(diff_s)])
        residuals = residuals * float(weight) / jnp.sqrt(jnp.asarray(max(int(residuals.size), 1), dtype=jnp.float64))
        return residuals, jnp.sum(residuals * residuals)

    return QIObjectiveTerm("boozer_b_target", _evaluate, qi_options=qi_options)


def qi_max_elongation_objective(
    *,
    threshold: float,
    weight: float = 1.0,
    ntheta: int = 48,
    nphi: int = 16,
    smooth_extrema: float = 0.0,
    smooth_penalty: float = 0.0,
    qi_options: QuasiIsodynamicOptions | None = None,
) -> QIObjectiveTerm:
    """Boundary elongation upper-bound objective."""

    def _evaluate(ctx: StageContext, state, _field: dict):
        elongation = max_elongation_penalty_from_state(
            state=state,
            static=ctx.static,
            threshold=float(threshold),
            ntheta=int(ntheta),
            nphi=int(nphi),
            smooth_extrema=float(smooth_extrema),
            smooth_penalty=float(smooth_penalty),
        )
        return (
            jnp.asarray(elongation["residuals1d"], dtype=jnp.float64) * float(weight),
            float(weight) ** 2 * elongation["total"],
        )

    return QIObjectiveTerm("max_elongation", _evaluate, qi_options=qi_options)


def qi_max_elongation_constraint(
    *,
    threshold: float,
    ntheta: int = 48,
    nphi: int = 16,
    smooth_extrema: float = 0.0,
    qi_options: QuasiIsodynamicOptions | None = None,
) -> QIObjectiveTerm:
    """Signed LCFS elongation constraint ``max_elongation - threshold <= 0``."""

    def _evaluate(ctx: StageContext, state, _field: dict):
        elongation = max_elongation_penalty_from_state(
            state=state,
            static=ctx.static,
            threshold=float(threshold),
            ntheta=int(ntheta),
            nphi=int(nphi),
            smooth_extrema=float(smooth_extrema),
            smooth_penalty=0.0,
        )
        residuals = jnp.asarray([elongation["max_elongation"] - float(threshold)], dtype=jnp.float64)
        return residuals, jnp.sum(jnp.maximum(residuals, 0.0) ** 2)

    return QIObjectiveTerm("max_elongation_constraint", _evaluate, qi_options=qi_options)


def qi_lgradb_objective(
    *,
    threshold: float,
    weight: float = 1.0,
    s_index: int = -1,
    ntheta: int = 9,
    nphi: int = 7,
    smooth_penalty: float = 0.0,
    qi_options: QuasiIsodynamicOptions | None = None,
) -> QIObjectiveTerm:
    """QI field-quality term penalizing small local ``L_grad_B``."""

    def _evaluate(ctx: StageContext, state, _field: dict):
        lgradb = lgradb_penalty_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            flux_local=ctx.flux,
            threshold=float(threshold),
            s_index=int(s_index),
            ntheta=int(ntheta),
            nphi=int(nphi),
            smooth_penalty=float(smooth_penalty),
        )
        return (
            jnp.asarray(lgradb["residuals1d"], dtype=jnp.float64) * float(weight),
            float(weight) ** 2 * lgradb["total"],
        )

    return QIObjectiveTerm("LgradB", _evaluate, qi_options=qi_options)


def mean_iota(ctx: StageContext, state):
    """Mean rotational transform on full-mesh surfaces, excluding the axis."""

    _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=ctx.static,
        indata=ctx.indata,
        signgs=ctx.signgs,
    )
    iotas = jnp.asarray(iotas, dtype=jnp.float64)
    return jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else jnp.mean(iotas[1:])


def objectives_track_iota(objectives: Sequence[ObjectiveTerm], target_iota: float | None = None) -> bool:
    """Return true when optimization history should record mean iota."""

    return target_iota is not None or any(term.track_iota for term in objectives)


def qs_stage_modes(
    *,
    max_mode: int,
    use_mode_continuation: bool,
    continuation_nfev: int,
) -> list[int]:
    """Repeated mode-continuation sequence used by the example scripts."""

    if bool(use_mode_continuation) and int(max_mode) > 1 and int(continuation_nfev) > 0:
        modes: list[int] = []
        for mode in range(1, int(max_mode) + 1):
            modes.extend([mode] * (2 if mode == 1 else 3))
        return modes
    return [int(max_mode)]


def repeated_stage_modes(
    *,
    max_mode: int,
    use_mode_continuation: bool,
    continuation_nfev: int,
    repeats: int = 5,
) -> list[int]:
    """Same-mode repeated continuation used by the QI example.

    Unlike :func:`qs_stage_modes`, repeated same-mode continuation has no
    lower-mode stages, so a zero ``continuation_nfev`` should not disable the
    repeated max-mode sequence.
    """

    del continuation_nfev
    if bool(use_mode_continuation) and int(max_mode) > 1:
        return [int(max_mode)] * max(1, int(repeats))
    return [int(max_mode)]


def boozer_b_target_from_wout(
    wout_path: str | Path,
    *,
    surfaces,
    mboz: int,
    nboz: int,
) -> dict[str, np.ndarray | int]:
    """Return Boozer ``|B|`` target spectra from a VMEC ``wout`` file.

    The returned ``bmnc_b``/``bmns_b`` arrays use the same surface-major shape
    as ``booz_xform_jax``'s differentiable API, so they can be passed directly
    to :class:`BoozerBTarget`.
    """

    from booz_xform_jax import Booz_xform

    bx = Booz_xform(verbose=0)
    bx.read_wout(str(wout_path), flux=False)
    s_in = np.asarray(bx.s_in, dtype=float)
    surface_indices = sorted({int(np.argmin(np.abs(s_in - float(surface)))) for surface in surfaces})
    bx.compute_surfs = surface_indices
    bx.mboz = int(mboz)
    bx.nboz = int(nboz)
    bx.mnboz = None
    bx.xm_b = None
    bx.xn_b = None
    bx._prepared = False
    bx.run()
    bmns_b = getattr(bx, "bmns_b", None)
    return {
        "bmnc_b": np.asarray(bx.bmnc_b, dtype=float).T,
        "bmns_b": None if bmns_b is None else np.asarray(bmns_b, dtype=float).T,
        "xm_b": np.asarray(bx.xm_b, dtype=int),
        "xn_b": np.asarray(bx.xn_b, dtype=int),
        "s_b": np.asarray(bx.s_b, dtype=float),
        "nfp": int(bx.nfp),
    }


def qs_stage_budget(
    *,
    stage_mode: int,
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
) -> int:
    """Outer residual/Jacobian budget for one fixed-boundary stage."""

    if int(max_nfev) <= 0:
        raise ValueError("max_nfev must be a positive integer for outer optimization stages.")
    if int(stage_mode) == int(max_mode):
        return int(max_nfev)
    # A zero continuation_nfev disables helper-generated lower-mode stages, but
    # users may still pass an explicit stage sequence for experiments such as
    # [2, 2, 3].  SciPy rejects max_nfev=0, so use the final-stage budget for
    # explicit lower-mode stages when no separate continuation budget is given.
    return int(continuation_nfev) if int(continuation_nfev) > 0 else int(max_nfev)


def normalize_boundary_mode_limits(stage_mode) -> BoundaryModeLimits:
    """Normalize an int/tuple/dict stage descriptor into mode limits.

    Accepted forms:

    - ``3``: square stage with ``max_m=max_n=3``.
    - ``(1, 4)``: anisotropic stage with ``max_m=1``, ``max_n=4`` and
      ``mode=4`` for VMEC resolution/budgeting.
    - ``(4, 1, 4)``: explicit ``(mode, max_m, max_n)``.
    - ``{"mode": 4, "max_m": 1, "max_n": 4, "label": "n-first"}``.
    """

    if isinstance(stage_mode, BoundaryModeLimits):
        return stage_mode
    if isinstance(stage_mode, dict):
        max_m = stage_mode.get("max_m")
        max_n = stage_mode.get("max_n")
        mode_raw = stage_mode.get("mode", stage_mode.get("max_mode"))
        if mode_raw is None:
            finite_limits = [value for value in (max_m, max_n) if value is not None]
            if not finite_limits:
                raise ValueError("Boundary mode-limit dictionaries require mode/max_mode or max_m/max_n.")
            mode_raw = max(int(value) for value in finite_limits)
        return BoundaryModeLimits(
            mode=int(mode_raw),
            max_m=None if max_m is None else int(max_m),
            max_n=None if max_n is None else int(max_n),
            label=stage_mode.get("label"),
        )
    if isinstance(stage_mode, (tuple, list)):
        if len(stage_mode) == 2:
            max_m, max_n = (None if value is None else int(value) for value in stage_mode)
            finite_limits = [value for value in (max_m, max_n) if value is not None]
            if not finite_limits:
                raise ValueError("At least one of max_m or max_n must be finite.")
            return BoundaryModeLimits(mode=max(finite_limits), max_m=max_m, max_n=max_n)
        if len(stage_mode) == 3:
            mode, max_m, max_n = stage_mode
            return BoundaryModeLimits(
                mode=int(mode),
                max_m=None if max_m is None else int(max_m),
                max_n=None if max_n is None else int(max_n),
            )
        raise ValueError("Boundary stage tuples must be (max_m, max_n) or (mode, max_m, max_n).")
    return BoundaryModeLimits(mode=int(stage_mode))


def describe_boundary_mode_limits(stage_mode) -> str:
    """Return a compact label for a boundary-mode stage descriptor."""

    limits = normalize_boundary_mode_limits(stage_mode)
    max_m = limits.mode if limits.max_m is None else limits.max_m
    max_n = limits.mode if limits.max_n is None else limits.max_n
    base = f"mode{limits.mode:02d}_m{int(max_m):02d}_n{int(max_n):02d}"
    return f"{base}_{limits.label}" if limits.label else base


def rebuild_for_optimization_resolution(indata, *, max_mode: int, min_vmec_mode: int = 5):
    """Set VMEC spectral resolution to at least ``max(min_vmec_mode, max_mode+2)``."""

    vmec_mpol = max(int(min_vmec_mode), int(max_mode) + 2)
    return rebuild_indata_with_resolution(indata, mpol=vmec_mpol, ntor=vmec_mpol)


def simple_omnigenity_seed_indata(
    indata,
    *,
    max_mode: int,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    perturbation: float = 1.0e-5,
    r0: float | None = None,
    rbc01: float | None = None,
    zbs01: float | None = None,
):
    """Return an input deck with a simple deterministic omnigenity seed boundary.

    The seed keeps only the near-circular base shape ``RBC(0,0)``,
    ``RBC(0,1)``, and ``ZBS(0,1)`` from the source deck unless explicit values
    are supplied.  Every other active optimizable boundary coefficient with
    ``max(abs(m), abs(n)) <= max_mode`` is set to a deterministic
    ``±perturbation`` value.  This avoids exactly-zero Jacobian columns when
    examples start far from a QA/QH/QP/QI warm-start boundary.
    """

    max_mode_i = int(max_mode)
    perturbation = float(perturbation)
    if max_mode_i < 0:
        raise ValueError("max_mode must be non-negative.")
    if not math.isfinite(perturbation) or perturbation < 0.0:
        raise ValueError("perturbation must be finite and non-negative.")

    boundary_keys = {"RBC", "RBS", "ZBC", "ZBS"}
    include_set = {str(item).lower() for item in include}
    fix_set = {str(item).lower() for item in fix}
    family_keys = {
        "rc": "RBC",
        "rs": "RBS",
        "zc": "ZBC",
        "zs": "ZBS",
    }
    kind_offsets = {"rc": 17, "rs": 29, "zc": 43, "zs": 59}

    def _source_value(key: str, index: tuple[int, int], default: float) -> float:
        return float(indata.indexed.get(key, {}).get(index, default))

    def _sign(kind: str, m_i: int, n_i: int) -> float:
        parity = (kind_offsets[kind] + 1009 * int(m_i) + 9176 * (int(n_i) + 8192)) % 2
        return 1.0 if parity == 0 else -1.0

    def _spec_name(kind: str, m_i: int, n_i: int) -> str:
        n_str = f"{int(n_i):+d}".replace("+", "")
        return f"{kind}{int(m_i)}{n_str}"

    out = copy.deepcopy(indata)
    out.indexed = {
        key: copy.deepcopy(values)
        for key, values in indata.indexed.items()
        if str(key).upper() not in boundary_keys
    }
    out.indexed["RBC"] = {
        (0, 0): _source_value("RBC", (0, 0), 1.0) if r0 is None else float(r0),
        (0, 1): _source_value("RBC", (0, 1), 0.2) if rbc01 is None else float(rbc01),
    }
    out.indexed["ZBS"] = {
        (0, 1): _source_value("ZBS", (0, 1), 0.2) if zbs01 is None else float(zbs01),
    }

    preserved = {("RBC", (0, 0)), ("RBC", (0, 1)), ("ZBS", (0, 1))}
    for m_i in range(0, max_mode_i + 1):
        n_values = range(0, max_mode_i + 1) if m_i == 0 else range(-max_mode_i, max_mode_i + 1)
        for n_i in n_values:
            if m_i == 0 and n_i == 0:
                continue
            for kind, key in family_keys.items():
                if kind not in include_set:
                    continue
                spec_name = _spec_name(kind, m_i, n_i)
                if spec_name.lower() in fix_set or (key, (n_i, m_i)) in preserved:
                    continue
                out.indexed.setdefault(key, {})[(n_i, m_i)] = _sign(kind, m_i, n_i) * perturbation

    return out


def prepare_simple_omnigenity_seed_input(
    input_file,
    output_dir,
    *,
    max_mode: int,
    min_vmec_mode: int = 5,
    enabled: bool = True,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    perturbation: float = 1.0e-5,
    filename: str = "input.simple_seed",
):
    """Write and return a simple omnigenity seed input path when enabled."""

    input_path = Path(input_file)
    if not bool(enabled):
        return input_path

    from .namelist import read_indata, write_indata

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indata = read_indata(input_path)
    indata = rebuild_for_optimization_resolution(
        indata,
        max_mode=max_mode,
        min_vmec_mode=min_vmec_mode,
    )
    seeded = simple_omnigenity_seed_indata(
        indata,
        max_mode=max_mode,
        include=include,
        fix=fix,
        perturbation=perturbation,
    )
    write_indata(output_path, seeded)
    return output_path


def interpolate_indata_boundary(
    seed_indata,
    reference_indata,
    lam: float,
    *,
    keys: Sequence[str] = ("RBC", "ZBS", "RBS", "ZBC"),
    max_mode: int | None = None,
    require_same_nfp: bool = True,
    preserve_seed_scalars: Sequence[str] = ("NFP", "LASYM"),
):
    """Interpolate selected VMEC boundary Fourier coefficients.

    This helper implements a deterministic global-to-local preconditioner for
    far-seed QI optimization.  ``lam=0`` preserves the seed boundary for the
    selected keys, while ``lam=1`` uses the reference boundary.  Scalar VMEC
    metadata remains seed-owned for entries in ``preserve_seed_scalars`` so a
    reference family can be used without accidentally changing the user's
    field-period count or symmetry flag.

    If ``max_mode`` is given, selected boundary coefficient dictionaries are
    projected to modes with ``abs(m) <= max_mode`` and ``abs(n) <= max_mode``.
    """

    lam = float(lam)
    if not math.isfinite(lam):
        raise ValueError("Boundary interpolation lambda must be finite.")
    if bool(require_same_nfp) and int(seed_indata.get_int("NFP", -1)) != int(reference_indata.get_int("NFP", -2)):
        raise ValueError(
            "Boundary interpolation requires same-NFP inputs; "
            f"got seed NFP={seed_indata.get_int('NFP')} and reference NFP={reference_indata.get_int('NFP')}."
        )

    out = copy.deepcopy(seed_indata)
    for key in preserve_seed_scalars:
        key = key.upper()
        if key in seed_indata.scalars:
            out.scalars[key] = copy.deepcopy(seed_indata.scalars[key])

    if "MPOL" in seed_indata.scalars or "MPOL" in reference_indata.scalars:
        out.scalars["MPOL"] = max(int(seed_indata.get_int("MPOL", 0)), int(reference_indata.get_int("MPOL", 0)))
    if "NTOR" in seed_indata.scalars or "NTOR" in reference_indata.scalars:
        out.scalars["NTOR"] = max(int(seed_indata.get_int("NTOR", 0)), int(reference_indata.get_int("NTOR", 0)))

    max_mode_i = None if max_mode is None else int(max_mode)
    for key in tuple(item.upper() for item in keys):
        seed_coeffs = seed_indata.indexed.get(key, {})
        ref_coeffs = reference_indata.indexed.get(key, {})
        indices = set(seed_coeffs) | set(ref_coeffs)
        interpolated = {}
        for idx in sorted(indices):
            if max_mode_i is not None and any(abs(int(i)) > max_mode_i for i in idx[:2]):
                continue
            seed_value = float(seed_coeffs.get(idx, 0.0))
            reference_value = float(ref_coeffs.get(idx, 0.0))
            interpolated[idx] = (1.0 - lam) * seed_value + lam * reference_value
        if interpolated or key in out.indexed:
            out.indexed[key] = interpolated
    return out


def build_fixed_boundary_objective_stage(
    cfg,
    indata,
    *,
    stage_mode: int,
    stage_max_m: int | None = None,
    stage_max_n: int | None = None,
    objectives: Sequence[ObjectiveTerm],
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    project_input_boundary_to_max_mode: bool = False,
    min_coeff: float = 0.0,
    inner_max_iter: int = 120,
    inner_ftol: float = 1.0e-9,
    trial_max_iter: int = 120,
    trial_ftol: float = 1.0e-9,
    solver_device: str | None = None,
    exact_path: str | None = None,
) -> FixedBoundaryObjectiveStage:
    """Build one VMEC/JAX optimization stage from an objective list."""

    stage_indata0 = (
        truncate_indata_boundary_modes(indata, max_mode=stage_mode)
        if bool(project_input_boundary_to_max_mode)
        else indata
    )
    static = build_static(cfg)
    boundary = boundary_from_indata(stage_indata0, static.modes, apply_m1_constraint=False)
    stage_indata, static, boundary = extend_boundary_for_max_mode(
        stage_indata0,
        static,
        boundary,
        stage_mode,
    )
    boundary_input = boundary_input_from_indata(stage_indata, static.modes)
    specs = boundary_param_specs(
        boundary_input,
        static.modes,
        max_mode=stage_mode,
        max_m=stage_max_m,
        max_n=stage_max_n,
        min_coeff=float(min_coeff),
        include=tuple(include),
        fix=tuple(fix),
    )

    guess = initial_guess_from_boundary(static, boundary, stage_indata, vmec_project=True)
    geom = eval_geom(guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    flux = flux_profiles_from_indata(stage_indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))
    ctx = StageContext(
        static=static,
        indata=stage_indata,
        boundary_input=boundary_input,
        specs=specs,
        signgs=signgs,
        flux=flux,
        pressure=pressure,
    )

    residuals_from_state = residuals_from_objectives(objectives, ctx)
    optimizer = FixedBoundaryExactOptimizer(
        static,
        stage_indata,
        boundary,
        specs,
        residuals_from_state,
        boundary_input=boundary_input,
        inner_max_iter=inner_max_iter,
        inner_ftol=inner_ftol,
        trial_max_iter=trial_max_iter,
        trial_ftol=trial_ftol,
        solver_device=solver_device,
        exact_path=exact_path,
    )
    return FixedBoundaryObjectiveStage(
        mode=int(stage_mode),
        ctx=ctx,
        optimizer=optimizer,
        specs=specs,
        boundary_input=boundary_input,
    )


def residuals_from_objectives(objectives: Sequence[ObjectiveTerm], ctx: StageContext):
    """Create the state residual callback consumed by ``FixedBoundaryExactOptimizer``."""

    def residuals_from_state(state, *, ctx=ctx):
        return jnp.concatenate([term.residual(ctx, state) for term in objectives])

    field_totals = tuple(term.total for term in objectives if term.total is not None)
    residuals_from_state._n_non_qs = sum(1 for term in objectives if term.total is None)
    residuals_from_state._qs_total_from_state = (
        lambda state, ctx=ctx, field_totals=field_totals: float(
            sum(float(total(ctx, state)) for total in field_totals)
        )
        if field_totals
        else lambda _state: 0.0
    )
    return residuals_from_state


def run_fixed_boundary_objective_optimization(
    *,
    cfg,
    indata,
    objectives: Sequence[ObjectiveTerm],
    stage_modes: Sequence[int],
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
    method: str,
    ftol: float,
    gtol: float,
    xtol: float,
    use_ess: bool,
    ess_alpha: float,
    output_dir: Path,
    label: str,
    use_mode_continuation: bool,
    target_aspect: float | None = None,
    target_iota: float | None = None,
    iota_abs_min: float | None = None,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    project_input_boundary_to_max_mode: bool = False,
    inner_max_iter: int = 120,
    inner_ftol: float = 1.0e-9,
    trial_max_iter: int = 120,
    trial_ftol: float = 1.0e-9,
    solver_device: str | None = None,
    exact_path: str | None = None,
    scipy_tr_solver: str | None = "lsmr",
    scipy_lsmr_maxiter: int | None = None,
    lbfgs_step_bound: float | None = None,
    scalar_step_bound: float | None = None,
    save_stage_inputs: bool = True,
    save_stage_wouts: bool = False,
    save_rerun_wouts: bool = False,
    save_final_outputs: bool = True,
) -> FixedBoundaryOptimizationResult:
    """Run a fixed-boundary objective list through one or more mode stages."""

    _enable_line_buffered_output()
    stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]] = []
    current_cfg = cfg
    current_indata = indata

    normalized_stage_modes = [normalize_boundary_mode_limits(stage_mode) for stage_mode in stage_modes]
    for stage_index, stage_limits in enumerate(normalized_stage_modes, start=1):
        stage = build_fixed_boundary_objective_stage(
            current_cfg,
            current_indata,
            stage_mode=int(stage_limits.mode),
            stage_max_m=stage_limits.max_m,
            stage_max_n=stage_limits.max_n,
            objectives=objectives,
            include=include,
            fix=fix,
            project_input_boundary_to_max_mode=project_input_boundary_to_max_mode,
            inner_max_iter=inner_max_iter,
            inner_ftol=inner_ftol,
            trial_max_iter=trial_max_iter,
            trial_ftol=trial_ftol,
            solver_device=solver_device,
            exact_path=exact_path,
        )
        x_scale = (
            create_x_scale(stage.specs, alpha=float(ess_alpha))
            if bool(use_ess)
            else np.ones(len(stage.specs), dtype=float)
        )
        # Each continuation stage is built from the previous stage's optimized
        # VMEC input, so the new optimization vector starts at zero increment.
        # This avoids reintroducing higher modes from the original deck when a
        # lower-mode stage intentionally projected them out.
        params0 = np.zeros(len(stage.specs), dtype=float)
        nfev = qs_stage_budget(
            stage_mode=int(stage_limits.mode),
            max_mode=int(max_mode),
            max_nfev=int(max_nfev),
            continuation_nfev=int(continuation_nfev),
        )
        iota_fn = (
            (lambda state, ctx=stage.ctx: float(mean_iota(ctx, state)))
            if objectives_track_iota(objectives, target_iota=target_iota) or iota_abs_min is not None
            else None
        )

        if int(stage_limits.mode) == int(max_mode):
            print_qs_problem_summary(
                method=method,
                max_nfev=nfev,
                use_mode_continuation=use_mode_continuation,
                use_ess=use_ess,
                ess_alpha=ess_alpha,
                objectives=objectives,
                specs=stage.specs,
                x_scale=np.asarray(x_scale, dtype=float),
                optimizer=stage.optimizer,
                params0=params0,
            )
        else:
            print(
                "Stage "
                f"{describe_boundary_mode_limits(stage_limits)} continuation seed "
                f"(budget={nfev}) ..."
            )

        result = stage.optimizer.run(
            params0,
            method=method,
            max_nfev=nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            x_scale=x_scale,
            verbose=1 if int(stage_limits.mode) == int(max_mode) else 0,
            iota_fn=iota_fn,
            target_iota=target_iota,
            target_aspect=target_aspect,
            scipy_tr_solver=scipy_tr_solver,
            scipy_lsmr_maxiter=scipy_lsmr_maxiter,
            lbfgs_step_bound=lbfgs_step_bound,
            scalar_step_bound=scalar_step_bound,
        )
        if iota_abs_min is not None:
            result["_history_dump"]["iota_abs_min"] = float(iota_abs_min)
        save_qs_stage_artifacts(
            stage_dir=output_dir / f"stage_{stage_index:02d}_{describe_boundary_mode_limits(stage_limits)}",
            optimizer=stage.optimizer,
            params_initial=params0,
            params_final=result["x"],
            result=result,
            save_inputs=save_stage_inputs,
            save_wouts=save_stage_wouts,
            save_rerun_wouts=save_rerun_wouts,
        )
        stage_records.append((int(stage_limits.mode), stage.optimizer, params0, result))
        current_indata = stage.optimizer._indata_from_params(result["x"])
        current_cfg = config_from_indata(current_indata)

    final_optimizer = stage_records[-1][1]
    final_result = stage_records[-1][3]
    combined_history = combine_qs_stage_histories(
        label=label,
        max_mode=max_mode,
        max_nfev=max_nfev,
        continuation_nfev=continuation_nfev,
        stage_modes=normalized_stage_modes,
        stage_records=stage_records,
    )
    if combined_history is not None:
        final_result["_history_dump"] = combined_history

    print_qs_final_summary(final_result, target_iota=target_iota, iota_abs_min=iota_abs_min)
    if save_final_outputs:
        save_qs_final_outputs(
            output_dir=output_dir,
            stage_records=stage_records,
            final_optimizer=final_optimizer,
            final_result=final_result,
            label=label,
            target_aspect=target_aspect,
            target_iota=target_iota,
            iota_abs_min=iota_abs_min,
            save_rerun_wouts=save_rerun_wouts,
        )
    else:
        annotate_qs_final_history(
            final_result,
            label=label,
            target_aspect=target_aspect,
            target_iota=target_iota,
            iota_abs_min=iota_abs_min,
        )
    return FixedBoundaryOptimizationResult(
        stage_records=stage_records,
        final_optimizer=final_optimizer,
        final_result=final_result,
        stage_modes=[int(stage_mode.mode) for stage_mode in normalized_stage_modes],
    )


def build_quasi_isodynamic_objective_stage(
    cfg,
    indata,
    *,
    stage_mode: int,
    stage_max_m: int | None = None,
    stage_max_n: int | None = None,
    scalar_objectives: Sequence[ObjectiveTerm],
    qi_objectives: Sequence[QIObjectiveTerm],
    surfaces,
    mboz: int,
    nboz: int,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    include_bounce_endpoints: bool,
    softness: float,
    width_weight: float,
    branch_width_weight: float,
    branch_width_softness: float,
    profile_weight: float,
    shuffle_profile_weight: float,
    shuffle_profile_softness: float,
    shuffle_profile_nphi_out: int | None = None,
    weighted_shuffle_profile_weight: float = 0.0,
    weighted_shuffle_profile_softness: float = 2.0e-2,
    aligned_profile_weight: float,
    aligned_profile_softness: float,
    aligned_profile_trap_level: float,
    aligned_profile_trap_softness: float,
    phimin: float,
    jit_booz: bool = True,
    project_input_boundary_to_max_mode: bool = True,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    inner_max_iter: int = 120,
    inner_ftol: float = 1.0e-9,
    trial_max_iter: int = 120,
    trial_ftol: float = 1.0e-9,
    solver_device: str | None = None,
    exact_path: str | None = None,
) -> FixedBoundaryObjectiveStage:
    """Build one QI stage while sharing one Boozer transform across QI terms."""

    stage_indata0 = (
        truncate_indata_boundary_modes(indata, max_mode=stage_mode)
        if bool(project_input_boundary_to_max_mode)
        else indata
    )
    static = build_static(cfg)
    boundary = boundary_from_indata(stage_indata0, static.modes, apply_m1_constraint=False)
    stage_indata, static, boundary = extend_boundary_for_max_mode(
        stage_indata0,
        static,
        boundary,
        stage_mode,
    )
    boundary_input = boundary_input_from_indata(stage_indata, static.modes)
    specs = boundary_param_specs(
        boundary_input,
        static.modes,
        max_mode=stage_mode,
        max_m=stage_max_m,
        max_n=stage_max_n,
        min_coeff=0.0,
        include=tuple(include),
        fix=tuple(fix),
    )

    guess = initial_guess_from_boundary(static, boundary, stage_indata, vmec_project=True)
    geom = eval_geom(guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    flux = flux_profiles_from_indata(stage_indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))
    ctx = StageContext(
        static=static,
        indata=stage_indata,
        boundary_input=boundary_input,
        specs=specs,
        signgs=signgs,
        flux=flux,
        pressure=pressure,
    )

    from booz_xform_jax import prepare_booz_xform_constants

    main_modes = vmec_mode_table(int(static.cfg.mpol), int(static.cfg.ntor))
    nyq_modes = nyquist_mode_table_from_grid(
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
    )
    booz_constants, booz_grids = prepare_booz_xform_constants(
        nfp=int(static.cfg.nfp),
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(static.cfg.lasym),
        xm=np.asarray(main_modes.m, dtype=int),
        xn=np.asarray(main_modes.n * int(static.cfg.nfp), dtype=int),
        xm_nyq=np.asarray(nyq_modes.m, dtype=int),
        xn_nyq=np.asarray(nyq_modes.n * int(static.cfg.nfp), dtype=int),
    )
    surface_indices = _nearest_half_mesh_indices(
        surfaces,
        n_half=max(int(np.asarray(static.s).shape[0]) - 1, 1),
    )

    def field_eval(state):
        return quasi_isodynamic_residual_from_state(
            state=state,
            static=static,
            indata=stage_indata,
            signgs=signgs,
            flux_local=flux,
            prof_local={"pressure": pressure},
            pressure_local=pressure,
            surfaces=surfaces,
            mboz=int(mboz),
            nboz=int(nboz),
            nphi=int(nphi),
            nalpha=int(nalpha),
            n_bounce=int(n_bounce),
            include_bounce_endpoints=bool(include_bounce_endpoints),
            softness=float(softness),
            width_weight=float(width_weight),
            branch_width_weight=float(branch_width_weight),
            branch_width_softness=float(branch_width_softness),
            profile_weight=float(profile_weight),
            shuffle_profile_weight=float(shuffle_profile_weight),
            shuffle_profile_softness=float(shuffle_profile_softness),
            shuffle_profile_nphi_out=shuffle_profile_nphi_out,
            weighted_shuffle_profile_weight=float(weighted_shuffle_profile_weight),
            weighted_shuffle_profile_softness=float(weighted_shuffle_profile_softness),
            aligned_profile_weight=float(aligned_profile_weight),
            aligned_profile_softness=float(aligned_profile_softness),
            aligned_profile_trap_level=float(aligned_profile_trap_level),
            aligned_profile_trap_softness=float(aligned_profile_trap_softness),
            phimin=float(phimin),
            jit_booz=bool(jit_booz),
            booz_constants=booz_constants,
            booz_grids=booz_grids,
            surface_indices=surface_indices,
        )

    def residuals_from_state(state, *, ctx=ctx):
        field = field_eval(state)
        scalar_parts = [term.residual(ctx, state) for term in scalar_objectives]
        qi_parts = [term.residual_and_total(ctx, state, field)[0] for term in qi_objectives]
        return jnp.concatenate([*scalar_parts, *qi_parts])

    residuals_from_state._n_non_qs = len(scalar_objectives)
    def _qs_total_from_state(state, *, ctx=ctx):
        field = field_eval(state)
        return float(sum(float(term.residual_and_total(ctx, state, field)[1]) for term in qi_objectives))

    residuals_from_state._qs_total_from_state = _qs_total_from_state

    optimizer = FixedBoundaryExactOptimizer(
        static,
        stage_indata,
        boundary,
        specs,
        residuals_from_state,
        boundary_input=boundary_input,
        inner_max_iter=inner_max_iter,
        inner_ftol=inner_ftol,
        trial_max_iter=trial_max_iter,
        trial_ftol=trial_ftol,
        solver_device=solver_device,
        exact_path=exact_path,
    )
    return FixedBoundaryObjectiveStage(
        mode=int(stage_mode),
        ctx=ctx,
        optimizer=optimizer,
        specs=specs,
        boundary_input=boundary_input,
    )


def run_quasi_isodynamic_objective_optimization(
    *,
    cfg,
    indata,
    scalar_objectives: Sequence[ObjectiveTerm],
    qi_objectives: Sequence[QIObjectiveTerm],
    stage_modes: Sequence[int],
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
    method: str,
    ftol: float,
    gtol: float,
    xtol: float,
    use_ess: bool,
    ess_alpha: float,
    output_dir: Path,
    label: str,
    use_mode_continuation: bool,
    surfaces,
    mboz: int,
    nboz: int,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    include_bounce_endpoints: bool,
    softness: float,
    width_weight: float,
    branch_width_weight: float,
    branch_width_softness: float,
    profile_weight: float,
    shuffle_profile_weight: float,
    shuffle_profile_softness: float,
    shuffle_profile_nphi_out: int | None = None,
    weighted_shuffle_profile_weight: float = 0.0,
    weighted_shuffle_profile_softness: float = 2.0e-2,
    aligned_profile_weight: float,
    aligned_profile_softness: float,
    aligned_profile_trap_level: float,
    aligned_profile_trap_softness: float,
    phimin: float,
    jit_booz: bool = True,
    target_aspect: float | None = None,
    iota_abs_min: float | None = None,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    project_input_boundary_to_max_mode: bool = True,
    inner_max_iter: int = 120,
    inner_ftol: float = 1.0e-9,
    trial_max_iter: int = 120,
    trial_ftol: float = 1.0e-9,
    solver_device: str | None = None,
    exact_path: str | None = None,
    scipy_tr_solver: str | None = "lsmr",
    scipy_lsmr_maxiter: int | None = None,
    lbfgs_step_bound: float | None = None,
    scalar_step_bound: float | None = None,
    save_stage_inputs: bool = True,
    save_stage_wouts: bool = False,
    save_final_outputs: bool = True,
) -> FixedBoundaryOptimizationResult:
    """Run a QI objective list through repeated or direct mode stages."""

    _enable_line_buffered_output()
    stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]] = []
    current_cfg = cfg
    current_indata = indata

    normalized_stage_modes = [normalize_boundary_mode_limits(stage_mode) for stage_mode in stage_modes]
    for stage_index, stage_limits in enumerate(normalized_stage_modes, start=1):
        stage = build_quasi_isodynamic_objective_stage(
            current_cfg,
            current_indata,
            stage_mode=int(stage_limits.mode),
            stage_max_m=stage_limits.max_m,
            stage_max_n=stage_limits.max_n,
            scalar_objectives=scalar_objectives,
            qi_objectives=qi_objectives,
            surfaces=surfaces,
            mboz=mboz,
            nboz=nboz,
            nphi=nphi,
            nalpha=nalpha,
            n_bounce=n_bounce,
            include_bounce_endpoints=include_bounce_endpoints,
            softness=softness,
            width_weight=width_weight,
            branch_width_weight=branch_width_weight,
            branch_width_softness=branch_width_softness,
            profile_weight=profile_weight,
            shuffle_profile_weight=shuffle_profile_weight,
            shuffle_profile_softness=shuffle_profile_softness,
            shuffle_profile_nphi_out=shuffle_profile_nphi_out,
            weighted_shuffle_profile_weight=weighted_shuffle_profile_weight,
            weighted_shuffle_profile_softness=weighted_shuffle_profile_softness,
            aligned_profile_weight=aligned_profile_weight,
            aligned_profile_softness=aligned_profile_softness,
            aligned_profile_trap_level=aligned_profile_trap_level,
            aligned_profile_trap_softness=aligned_profile_trap_softness,
            phimin=phimin,
            jit_booz=jit_booz,
            project_input_boundary_to_max_mode=project_input_boundary_to_max_mode,
            include=include,
            fix=fix,
            inner_max_iter=inner_max_iter,
            inner_ftol=inner_ftol,
            trial_max_iter=trial_max_iter,
            trial_ftol=trial_ftol,
            solver_device=solver_device,
            exact_path=exact_path,
        )
        x_scale = (
            create_x_scale(stage.specs, alpha=float(ess_alpha))
            if bool(use_ess)
            else np.ones(len(stage.specs), dtype=float)
        )
        # The stage input already contains the previous optimized boundary.
        # New modes therefore start from their deck values (usually zero after
        # projection) and all active coefficients are represented as increments.
        params0 = np.zeros(len(stage.specs), dtype=float)
        nfev = qs_stage_budget(
            stage_mode=int(stage_limits.mode),
            max_mode=int(max_mode),
            max_nfev=int(max_nfev),
            continuation_nfev=int(continuation_nfev),
        )
        iota_fn = (
            (lambda state, ctx=stage.ctx: float(mean_iota(ctx, state)))
            if objectives_track_iota(scalar_objectives) or iota_abs_min is not None
            else None
        )
        if int(stage_limits.mode) == int(max_mode):
            print_qs_problem_summary(
                method=method,
                max_nfev=nfev,
                use_mode_continuation=use_mode_continuation,
                use_ess=use_ess,
                ess_alpha=ess_alpha,
                objectives=scalar_objectives,
                specs=stage.specs,
                x_scale=np.asarray(x_scale, dtype=float),
                optimizer=stage.optimizer,
                params0=params0,
            )
            print("QI field objectives:")
            for term in qi_objectives:
                print(f"  - {term.name}")
        else:
            print(
                "Stage "
                f"{describe_boundary_mode_limits(stage_limits)} continuation seed "
                f"(budget={nfev}) ..."
            )

        result = stage.optimizer.run(
            params0,
            method=method,
            max_nfev=nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            x_scale=x_scale,
            verbose=1 if int(stage_limits.mode) == int(max_mode) else 0,
            iota_fn=iota_fn,
            target_aspect=target_aspect,
            scipy_tr_solver=scipy_tr_solver,
            scipy_lsmr_maxiter=scipy_lsmr_maxiter,
            lbfgs_step_bound=lbfgs_step_bound,
            scalar_step_bound=scalar_step_bound,
        )
        if iota_abs_min is not None:
            result["_history_dump"]["iota_abs_min"] = float(iota_abs_min)
        save_qs_stage_artifacts(
            stage_dir=output_dir / f"stage_{stage_index:02d}_{describe_boundary_mode_limits(stage_limits)}",
            optimizer=stage.optimizer,
            params_initial=params0,
            params_final=result["x"],
            result=result,
            save_inputs=save_stage_inputs,
            save_wouts=save_stage_wouts,
        )
        stage_records.append((int(stage_limits.mode), stage.optimizer, params0, result))
        current_indata = stage.optimizer._indata_from_params(result["x"])
        current_cfg = config_from_indata(current_indata)

    final_optimizer = stage_records[-1][1]
    final_result = stage_records[-1][3]
    combined_history = combine_qs_stage_histories(
        label=label,
        max_mode=max_mode,
        max_nfev=max_nfev,
        continuation_nfev=continuation_nfev,
        stage_modes=normalized_stage_modes,
        stage_records=stage_records,
    )
    if combined_history is not None:
        final_result["_history_dump"] = combined_history

    print_qs_final_summary(final_result, iota_abs_min=iota_abs_min)
    if save_final_outputs:
        save_qs_final_outputs(
            output_dir=output_dir,
            stage_records=stage_records,
            final_optimizer=final_optimizer,
            final_result=final_result,
            label=label,
            target_aspect=target_aspect,
            iota_abs_min=iota_abs_min,
        )
    else:
        annotate_qs_final_history(
            final_result,
            label=label,
            target_aspect=target_aspect,
            iota_abs_min=iota_abs_min,
        )
    return FixedBoundaryOptimizationResult(
        stage_records=stage_records,
        final_optimizer=final_optimizer,
        final_result=final_result,
        stage_modes=[int(stage_mode.mode) for stage_mode in normalized_stage_modes],
    )


def least_squares_solve(
    vmec: FixedBoundaryVMEC,
    problem: LeastSquaresProblem,
    *,
    stage_modes: Sequence[int],
    max_nfev: int,
    continuation_nfev: int,
    method: str = "scipy",
    ftol: float = 1.0e-4,
    gtol: float = 1.0e-4,
    xtol: float = 1.0e-4,
    use_ess: bool = False,
    ess_alpha: float = 1.2,
    label: str = "Fixed-boundary optimization",
    use_mode_continuation: bool = True,
    inner_max_iter: int = 120,
    inner_ftol: float = 1.0e-9,
    trial_max_iter: int = 120,
    trial_ftol: float = 1.0e-9,
    solver_device: str | None = None,
    exact_path: str | None = None,
    scipy_tr_solver: str | None = "lsmr",
    scipy_lsmr_maxiter: int | None = None,
    lbfgs_step_bound: float | None = None,
    scalar_step_bound: float | None = None,
    save_stage_inputs: bool = True,
    save_stage_wouts: bool = False,
    save_rerun_wouts: bool = False,
    save_final_outputs: bool = True,
) -> FixedBoundaryOptimizationResult:
    """Solve a SIMSOPT-style fixed-boundary least-squares problem.

    The examples use this as the common public workflow:

    1. create a :class:`FixedBoundaryVMEC`,
    2. assemble a :class:`LeastSquaresProblem` from ``(J, target, weight)``
       tuples,
    3. choose stage modes and optimizer settings,
    4. call this function.
    """

    _enable_line_buffered_output()
    metadata = dict(problem.metadata)
    target_aspect = _metadata_float(metadata, "target_aspect")
    target_iota = _metadata_float(metadata, "target_iota")
    iota_abs_min = _metadata_float(metadata, "iota_abs_min")

    if problem.is_qi:
        qi_options = problem.qi_options
        if qi_options is None:
            raise ValueError(
                "QI objectives require QuasiIsodynamicOptions on a QI objective, "
                "for example QuasiIsodynamicResidual(QI_OPTIONS)."
            )
        return run_quasi_isodynamic_objective_optimization(
            cfg=vmec.cfg,
            indata=vmec.indata,
            scalar_objectives=problem.objective_terms,
            qi_objectives=problem.qi_objective_terms,
            stage_modes=stage_modes,
            max_mode=vmec.max_mode,
            max_nfev=max_nfev,
            continuation_nfev=continuation_nfev,
            method=method,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            use_ess=use_ess,
            ess_alpha=ess_alpha,
            output_dir=vmec.output_dir,
            label=label,
            use_mode_continuation=use_mode_continuation,
            surfaces=qi_options.surfaces,
            mboz=qi_options.mboz,
            nboz=qi_options.nboz,
            nphi=qi_options.nphi,
            nalpha=qi_options.nalpha,
            n_bounce=qi_options.n_bounce,
            include_bounce_endpoints=qi_options.include_bounce_endpoints,
            softness=qi_options.softness,
            width_weight=qi_options.width_weight,
            branch_width_weight=qi_options.branch_width_weight,
            branch_width_softness=qi_options.branch_width_softness,
            profile_weight=qi_options.profile_weight,
            shuffle_profile_weight=qi_options.shuffle_profile_weight,
            shuffle_profile_softness=qi_options.shuffle_profile_softness,
            shuffle_profile_nphi_out=qi_options.shuffle_profile_nphi_out,
            weighted_shuffle_profile_weight=qi_options.weighted_shuffle_profile_weight,
            weighted_shuffle_profile_softness=qi_options.weighted_shuffle_profile_softness,
            aligned_profile_weight=qi_options.aligned_profile_weight,
            aligned_profile_softness=qi_options.aligned_profile_softness,
            aligned_profile_trap_level=qi_options.aligned_profile_trap_level,
            aligned_profile_trap_softness=qi_options.aligned_profile_trap_softness,
            phimin=qi_options.phimin,
            jit_booz=qi_options.jit_booz,
            target_aspect=target_aspect,
            iota_abs_min=iota_abs_min,
            include=vmec.include,
            fix=vmec.fix,
            project_input_boundary_to_max_mode=vmec.project_input_boundary_to_max_mode,
            inner_max_iter=inner_max_iter,
            inner_ftol=inner_ftol,
            trial_max_iter=trial_max_iter,
            trial_ftol=trial_ftol,
            solver_device=solver_device,
            exact_path=exact_path,
            scipy_tr_solver=scipy_tr_solver,
            scipy_lsmr_maxiter=scipy_lsmr_maxiter,
            lbfgs_step_bound=lbfgs_step_bound,
            scalar_step_bound=scalar_step_bound,
            save_stage_inputs=save_stage_inputs,
            save_stage_wouts=save_stage_wouts,
            save_final_outputs=save_final_outputs,
        )

    return run_fixed_boundary_objective_optimization(
        cfg=vmec.cfg,
        indata=vmec.indata,
        objectives=problem.objective_terms,
        stage_modes=stage_modes,
        max_mode=vmec.max_mode,
        max_nfev=max_nfev,
        continuation_nfev=continuation_nfev,
        method=method,
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
        use_ess=use_ess,
        ess_alpha=ess_alpha,
        output_dir=vmec.output_dir,
        label=label,
        use_mode_continuation=use_mode_continuation,
        target_aspect=target_aspect,
        target_iota=target_iota,
        iota_abs_min=iota_abs_min,
        include=vmec.include,
        fix=vmec.fix,
        project_input_boundary_to_max_mode=vmec.project_input_boundary_to_max_mode,
        inner_max_iter=inner_max_iter,
        inner_ftol=inner_ftol,
        trial_max_iter=trial_max_iter,
        trial_ftol=trial_ftol,
        solver_device=solver_device,
        exact_path=exact_path,
        scipy_tr_solver=scipy_tr_solver,
        scipy_lsmr_maxiter=scipy_lsmr_maxiter,
        lbfgs_step_bound=lbfgs_step_bound,
        scalar_step_bound=scalar_step_bound,
        save_stage_inputs=save_stage_inputs,
        save_stage_wouts=save_stage_wouts,
        save_rerun_wouts=save_rerun_wouts,
        save_final_outputs=save_final_outputs,
    )


def print_qs_problem_summary(
    *,
    method: str,
    max_nfev: int,
    use_mode_continuation: bool,
    use_ess: bool,
    ess_alpha: float,
    objectives: Sequence[ObjectiveTerm],
    specs: Sequence[BoundaryParamSpec],
    x_scale: np.ndarray,
    optimizer,
    params0,
) -> None:
    """Print the problem summary used by the standalone examples."""

    print(f"Parameter space ({len(specs)} DOFs): {boundary_param_names(specs)}")
    print("Objectives:")
    for term in objectives:
        print(f"  - {term.name}: target={term.target}, weight={term.weight}")
    if use_ess:
        print(f"ESS scales (alpha={ess_alpha}): min={x_scale.min():.3f}  max={x_scale.max():.3f}")
    else:
        print("ESS disabled - uniform scales.")
    print(f"Aspect ratio (initial):        {optimizer.aspect_ratio(params0):.6f}")
    print(f"Field objective (initial):     {optimizer.quasisymmetry_objective(params0):.6e}")
    print(f"Running {method} (max_nfev={max_nfev}, continuation={use_mode_continuation}) ...")


def print_qs_final_summary(
    result: dict,
    *,
    target_iota: float | None = None,
    iota_abs_min: float | None = None,
) -> None:
    """Print the final scalar diagnostics from an optimization result."""

    hist = result.get("_history_dump", {})
    print(f"\nTermination: {result['message']}")
    print(f"Aspect ratio (final):          {float(hist.get('aspect_final', float('nan'))):.6f}")
    if "iota_final" in hist:
        if target_iota is not None:
            target = f"  target={target_iota:.6f}"
        elif iota_abs_min is not None:
            target = f"  min |iota|={iota_abs_min:.6f}"
        else:
            target = ""
        print(f"Mean iota (final):             {float(hist['iota_final']):.6f}{target}")
    print(f"Field objective (final):       {float(hist.get('qs_final', float('nan'))):.6e}")
    print(f"Total objective (final):       {float(hist.get('objective_final', float('nan'))):.6e}")
    obj0 = hist.get("objective_initial")
    objf = hist.get("objective_final")
    if obj0 is not None and float(obj0) > 0.0 and objf is not None:
        print(f"Objective reduction:           {100.0 * (1.0 - float(objf) / float(obj0)):.1f}%")


def save_qs_stage_artifacts(
    *,
    stage_dir: Path,
    optimizer,
    params_initial,
    params_final,
    result,
    save_inputs: bool = True,
    save_wouts: bool = False,
    save_rerun_wouts: bool = False,
) -> None:
    """Save stage input files and optionally wout files."""

    stage_dir.mkdir(parents=True, exist_ok=True)
    if save_inputs:
        optimizer.save_input(stage_dir / "input.initial", params_initial)
        optimizer.save_input(stage_dir / "input.final", params_final)
    if save_wouts:
        optimizer.save_wout(stage_dir / "wout_initial.nc", params_initial, state=result.get("_state_initial"))
        optimizer.save_wout(stage_dir / "wout_final.nc", params_final, state=result.get("_state_final"))
    else:
        _remove_stale(stage_dir / "wout_initial.nc")
        _remove_stale(stage_dir / "wout_final.nc")
    if save_rerun_wouts:
        rerun = run_fixed_boundary(str(stage_dir / "input.initial"), verbose=False)
        write_wout_from_fixed_boundary_run(str(stage_dir / "wout_initial_rerun.nc"), rerun)
        rerun = run_fixed_boundary(str(stage_dir / "input.final"), verbose=False)
        write_wout_from_fixed_boundary_run(str(stage_dir / "wout_final_rerun.nc"), rerun)
    else:
        _remove_stale(stage_dir / "wout_initial_rerun.nc")
        _remove_stale(stage_dir / "wout_final_rerun.nc")


def save_qs_final_outputs(
    *,
    output_dir: Path,
    stage_records,
    final_optimizer,
    final_result: dict,
    label: str,
    target_aspect: float | None = None,
    target_iota: float | None = None,
    iota_abs_min: float | None = None,
    save_rerun_wouts: bool = False,
) -> None:
    """Save initial/final inputs, wouts, and history."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _initial_mode, initial_optimizer, initial_params0, initial_result = stage_records[0]
    initial_optimizer.save_input(output_dir / "input.initial", initial_params0)
    initial_optimizer.save_wout(
        output_dir / "wout_initial.nc",
        initial_params0,
        state=initial_result.get("_state_initial"),
    )
    if save_rerun_wouts:
        rerun = run_fixed_boundary(str(output_dir / "input.initial"), verbose=False)
        write_wout_from_fixed_boundary_run(str(output_dir / "wout_initial_rerun.nc"), rerun)
    else:
        _remove_stale(output_dir / "wout_initial_rerun.nc")

    final_optimizer.save_input(output_dir / "input.final", final_result["x"])
    final_optimizer.save_wout(
        output_dir / "wout_final.nc",
        final_result["x"],
        state=final_result.get("_state_final"),
    )
    if save_rerun_wouts:
        rerun = run_fixed_boundary(str(output_dir / "input.final"), verbose=False)
        write_wout_from_fixed_boundary_run(str(output_dir / "wout_final_rerun.nc"), rerun)
    else:
        _remove_stale(output_dir / "wout_final_rerun.nc")

    annotate_qs_final_history(
        final_result,
        label=label,
        target_aspect=target_aspect,
        target_iota=target_iota,
        iota_abs_min=iota_abs_min,
    )
    final_optimizer.save_history(output_dir / "history.json", final_result)


def optimization_output_paths(output_dir: str | Path) -> OptimizationOutputPaths:
    """Return the canonical final-artifact paths for an optimization run."""

    output_dir = Path(output_dir)
    return OptimizationOutputPaths(
        initial_input=output_dir / "input.initial",
        final_input=output_dir / "input.final",
        initial_wout=output_dir / "wout_initial.nc",
        final_wout=output_dir / "wout_final.nc",
        history=output_dir / "history.json",
    )


def save_optimization_result(
    result: FixedBoundaryOptimizationResult,
    *,
    output_dir: str | Path | None = None,
    paths: OptimizationOutputPaths | None = None,
) -> OptimizationOutputPaths:
    """Save initial/final inputs, wouts, and history from a solve result.

    The examples use this for the mechanical file writes only.  Diagnostics,
    plotting, and any extra exports should remain explicit in the user script.
    """

    if paths is None:
        if output_dir is None:
            raise ValueError("Either output_dir or paths must be provided.")
        paths = optimization_output_paths(output_dir)
    for path in paths.as_dict().values():
        path.parent.mkdir(parents=True, exist_ok=True)

    result.initial_optimizer.save_input(paths.initial_input, result.initial_params)
    result.initial_optimizer.save_wout(
        paths.initial_wout,
        result.initial_params,
        state=result.initial_state,
    )
    result.final_optimizer.save_input(paths.final_input, result.final_params)
    result.final_optimizer.save_wout(
        paths.final_wout,
        result.final_params,
        state=result.final_state,
    )
    result.final_optimizer.save_history(paths.history, result.final_result)
    return paths


def annotate_qs_final_history(
    final_result: dict,
    *,
    label: str,
    target_aspect: float | None = None,
    target_iota: float | None = None,
    iota_abs_min: float | None = None,
) -> None:
    """Attach final optimization metadata without writing artifacts."""

    history = final_result["_history_dump"]
    history["label"] = label
    if target_aspect is not None:
        history["target_aspect"] = float(target_aspect)
    if target_iota is not None:
        history["target_iota"] = float(target_iota)
    if iota_abs_min is not None:
        history["iota_abs_min"] = float(iota_abs_min)


def combine_qs_stage_histories(
    *,
    label: str,
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
    stage_modes,
    stage_records,
) -> dict | None:
    """Merge per-stage histories into one optimization history."""

    if len(stage_records) <= 1:
        return None
    normalized_stage_modes = [normalize_boundary_mode_limits(stage_mode) for stage_mode in stage_modes]

    combined_entries = []
    stage_boundaries = []
    wall_offset = 0.0
    nfev_total = 0
    njev_total = 0
    for idx, (_mode, _optimizer, _params0, result) in enumerate(stage_records):
        stage_hist = result["_history_dump"]
        entries = stage_hist["history"] if idx == 0 else stage_hist["history"][1:]
        for entry in entries:
            entry_copy = dict(entry)
            entry_copy["wall_time_s"] = float(entry_copy["wall_time_s"]) + wall_offset
            combined_entries.append(entry_copy)
        wall_offset = float(combined_entries[-1]["wall_time_s"]) if combined_entries else wall_offset
        stage_boundaries.append(len(combined_entries) - 1)
        nfev_total += int(stage_hist["nfev"])
        njev_total += int(stage_hist["njev"])

    final_hist = stage_records[-1][3]["_history_dump"]
    first_hist = stage_records[0][3]["_history_dump"]
    out = dict(final_hist)
    out.update(
        {
            "label": label,
            "max_nfev": int(
                sum(
                    qs_stage_budget(
                        stage_mode=int(mode.mode),
                        max_mode=int(max_mode),
                        max_nfev=int(max_nfev),
                        continuation_nfev=int(continuation_nfev),
                    )
                    for mode in normalized_stage_modes
                )
            ),
            "total_wall_time_s": float(wall_offset),
            "nfev": int(nfev_total),
            "njev": int(njev_total),
            "objective_initial": float(first_hist["objective_initial"]),
            "objective_final": float(final_hist["objective_final"]),
            "qs_initial": float(first_hist["qs_initial"]),
            "qs_final": float(final_hist["qs_final"]),
            "aspect_initial": float(first_hist["aspect_initial"]),
            "aspect_final": float(final_hist["aspect_final"]),
            "history": combined_entries,
            "stage_boundaries": stage_boundaries,
            "stage_mode_descriptors": [
                {
                    "mode": int(mode.mode),
                    "max_m": None if mode.max_m is None else int(mode.max_m),
                    "max_n": None if mode.max_n is None else int(mode.max_n),
                    "label": mode.label,
                }
                for mode in normalized_stage_modes
            ],
        }
    )
    if combined_entries and "iota" in combined_entries[0] and "iota" in combined_entries[-1]:
        out["iota_initial"] = float(combined_entries[0]["iota"])
        out["iota_final"] = float(combined_entries[-1]["iota"])
    return out


def _as_vector(value):
    arr = jnp.asarray(value, dtype=jnp.float64)
    return arr.reshape((1,)) if int(arr.ndim) == 0 else jnp.ravel(arr)


def _target_is_zero(target) -> bool:
    return bool(np.allclose(np.asarray(target, dtype=float), 0.0))


def _metadata_float(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    return None if value is None else float(value)


def _result_timing_summary(result: dict, *, history: dict | None = None) -> dict[str, object]:
    """Extract timing and optimizer call counts from a raw optimizer result."""

    hist = dict(result.get("_history_dump", {}) if history is None else history)
    return {
        "total_wall_time_s": hist.get("total_wall_time_s"),
        "nfev": hist.get("nfev", result.get("nfev")),
        "njev": hist.get("njev", result.get("njev")),
        "nit": hist.get("nit", result.get("nit")),
    }


def _remove_stale(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _slice_boozer_surfaces(booz: dict, surface_index: int) -> dict:
    bmnc = booz.get("bmnc_b")
    if bmnc is None:
        raise ValueError("Boozer output must include bmnc_b to slice surfaces.")
    nsurf = int(jnp.asarray(bmnc).shape[0])
    index = int(surface_index)
    if index < 0:
        index += nsurf
    if index < 0 or index >= nsurf:
        raise ValueError(f"surface_index {surface_index} is outside the Boozer surface range 0..{nsurf - 1}.")
    out = dict(booz)
    for key in ("bmnc_b", "bmns_b", "iota_b", "s_b"):
        value = out.get(key)
        if value is not None:
            out[key] = value[index : index + 1]
    return out


__all__ = [
    "AbsMeanIotaFloor",
    "AbsMeanIotaCeiling",
    "AspectRatio",
    "AugmentedLagrangianConstraint",
    "BVector",
    "BDotB",
    "BDotGradV",
    "BetaTotal",
    "BoozerBTarget",
    "BoundaryModeLimits",
    "DMerc",
    "FixedBoundaryVMEC",
    "FixedBoundaryObjectiveStage",
    "FixedBoundaryOptimizationResult",
    "JDotB",
    "JVector",
    "LeastSquaresProblem",
    "LgradB",
    "MagneticWell",
    "MaxElongation",
    "MeanIota",
    "MirrorRatio",
    "ObjectiveTerm",
    "OptimizationOutputPaths",
    "QuasiIsodynamicOptions",
    "QuasiIsodynamicResidual",
    "QuasiIsodynamicResidualCeiling",
    "QuasisymmetryRatioResidual",
    "QIObjectiveTerm",
    "RedlBootstrapMismatch",
    "StageContext",
    "ToroidalCurrent",
    "ToroidalCurrentGradient",
    "abs_mean_iota_floor_objective",
    "abs_mean_iota_ceiling_objective",
    "aspect_objective",
    "boozer_b_target_from_wout",
    "build_fixed_boundary_objective_stage",
    "build_quasi_isodynamic_objective_stage",
    "combine_qs_stage_histories",
    "describe_boundary_mode_limits",
    "interpolate_indata_boundary",
    "lgradb_objective",
    "least_squares_solve",
    "mean_iota",
    "mean_iota_objective",
    "normalize_boundary_mode_limits",
    "objectives_track_iota",
    "optimization_output_paths",
    "qs_stage_budget",
    "qs_stage_modes",
    "qi_lgradb_objective",
    "qi_max_elongation_constraint",
    "qi_boozer_b_target_objective",
    "qi_max_elongation_objective",
    "qi_mirror_ratio_constraint",
    "qi_mirror_ratio_objective",
    "qi_residual_ceiling_objective",
    "quasi_isodynamic_field_objective",
    "quasisymmetry_objective",
    "rebuild_for_optimization_resolution",
    "repeated_stage_modes",
    "prepare_simple_omnigenity_seed_input",
    "residuals_from_objectives",
    "run_fixed_boundary_objective_optimization",
    "run_quasi_isodynamic_objective_optimization",
    "save_optimization_result",
    "save_qs_final_outputs",
    "save_qs_stage_artifacts",
    "simple_omnigenity_seed_indata",
    "VolavgB",
]

"""Reusable teaching workflow helpers for fixed-boundary optimizations.

The functions in this module are intentionally small building blocks for the
standalone examples.  Users should still construct objective lists explicitly in
the scripts, as in SIMSOPT's ``LeastSquaresProblem.from_tuples`` workflow, but
the mechanical VMEC/JAX stage setup, mode continuation, saving, and plotting
live here instead of being repeated in every example.
"""

from __future__ import annotations

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
from .field import signgs_from_sqrtg
from .geom import eval_geom
from .init_guess import initial_guess_from_boundary
from .optimization import (
    BoundaryParamSpec,
    FixedBoundaryExactOptimizer,
    boundary_param_specs,
    create_x_scale,
    extend_boundary_for_max_mode,
    smooth_min_abs_iota_residual,
    truncate_indata_boundary_modes,
)
from .optimizers.fixed_boundary.objective_terms import FixedBoundaryObjectiveStage
from .optimizers.fixed_boundary.objective_terms import ObjectiveTerm
from .optimizers.fixed_boundary.objective_terms import QIObjectiveTerm
from .optimizers.fixed_boundary.objective_terms import StageContext
from .optimizers.fixed_boundary.objective_terms import as_vector
from .optimizers.fixed_boundary.objective_terms import attach_packed_state_autodiff_hooks as _attach_packed_state_autodiff_hooks
from .optimizers.fixed_boundary.objective_terms import residuals_from_objectives
from .optimizers.fixed_boundary.qi_objectives import BoozerBTarget
from .optimizers.fixed_boundary.qi_objectives import LgradB
from .optimizers.fixed_boundary.qi_objectives import MaxElongation
from .optimizers.fixed_boundary.qi_objectives import MirrorRatio
from .optimizers.fixed_boundary.qi_objectives import QuasiIsodynamicOptions
from .optimizers.fixed_boundary.qi_objectives import QuasiIsodynamicResidual
from .optimizers.fixed_boundary.qi_objectives import QuasiIsodynamicResidualCeiling
from .optimizers.fixed_boundary.qi_objectives import VMECMirrorRatio
from .optimizers.fixed_boundary.qi_objectives import boozer_b_target_from_wout
from .optimizers.fixed_boundary.qi_objectives import lgradb_objective
from .optimizers.fixed_boundary.qi_objectives import qi_boozer_b_target_objective
from .optimizers.fixed_boundary.qi_objectives import qi_lgradb_objective
from .optimizers.fixed_boundary.qi_objectives import qi_max_elongation_constraint
from .optimizers.fixed_boundary.qi_objectives import qi_max_elongation_objective
from .optimizers.fixed_boundary.qi_objectives import qi_mirror_ratio_constraint
from .optimizers.fixed_boundary.qi_objectives import qi_mirror_ratio_objective
from .optimizers.fixed_boundary.qi_objectives import qi_residual_ceiling_objective
from .optimizers.fixed_boundary.qi_objectives import quasi_isodynamic_field_objective
from .optimizers.fixed_boundary.finite_beta_objectives import BDotB
from .optimizers.fixed_boundary.finite_beta_objectives import BDotGradV
from .optimizers.fixed_boundary.finite_beta_objectives import BVector
from .optimizers.fixed_boundary.finite_beta_objectives import BetaTotal
from .optimizers.fixed_boundary.finite_beta_objectives import DMerc
from .optimizers.fixed_boundary.finite_beta_objectives import GlasserResistiveInterchange
from .optimizers.fixed_boundary.finite_beta_objectives import JDotB
from .optimizers.fixed_boundary.finite_beta_objectives import JVector
from .optimizers.fixed_boundary.finite_beta_objectives import MagneticWell
from .optimizers.fixed_boundary.finite_beta_objectives import RedlBootstrapMismatch
from .optimizers.fixed_boundary.finite_beta_objectives import ToroidalCurrent
from .optimizers.fixed_boundary.finite_beta_objectives import ToroidalCurrentGradient
from .optimizers.fixed_boundary.finite_beta_objectives import VolavgB
from .optimizers.fixed_boundary.parameterization import rebuild_indata_with_resolution
from .optimizers.fixed_boundary.seed_inputs import interpolate_indata_boundary
from .optimizers.fixed_boundary.seed_inputs import prepare_simple_omnigenity_seed_input
from .optimizers.fixed_boundary.seed_inputs import simple_omnigenity_seed_indata
from .optimizers.fixed_boundary.stage_policy import BoundaryModeLimits
from .optimizers.fixed_boundary.stage_policy import describe_boundary_mode_limits
from .optimizers.fixed_boundary.stage_policy import normalize_boundary_mode_limits
from .optimizers.fixed_boundary.stage_policy import qs_stage_budget
from .optimizers.fixed_boundary.stage_policy import qs_stage_modes
from .optimizers.fixed_boundary.stage_policy import repeated_stage_modes
from .optimizers.fixed_boundary.workflow_artifacts import FixedBoundaryOptimizationResult
from .optimizers.fixed_boundary.workflow_artifacts import OptimizationOutputPaths
from .optimizers.fixed_boundary.workflow_artifacts import optimization_output_paths
from .optimizers.fixed_boundary.workflow_artifacts import save_optimization_result
from .optimizers.fixed_boundary import workflow_outputs as _workflow_outputs
from .modes import nyquist_mode_table_from_grid, vmec_mode_table
from .quasi_isodynamic import (
    _nearest_half_mesh_indices,
    quasi_isodynamic_residual_from_state,
)
from .quasisymmetry import quasisymmetry_ratio_residual_from_state
from .static import build_static
from .wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


enable_x64(True)


_LINE_BUFFERING_ENABLED = False
_as_vector = as_vector


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
    vmec_mpol: int | None = None
    vmec_ntor: int | None = None
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
        vmec_mpol: int | None = None,
        vmec_ntor: int | None = None,
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
            vmec_mpol=vmec_mpol,
            vmec_ntor=vmec_ntor,
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
            vmec_mpol=None if vmec_mpol is None else int(vmec_mpol),
            vmec_ntor=None if vmec_ntor is None else int(vmec_ntor),
            output_dir=Path(output_dir),
            project_input_boundary_to_max_mode=bool(project_input_boundary_to_max_mode),
            include=tuple(include),
            fix=tuple(fix),
        )


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
        if tuple_weight == 0.0:
            return
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

    @property
    def scalar_objective_names(self) -> tuple[str, ...]:
        """Names of state-space objective terms assembled from tuples."""

        return tuple(term.name for term in self.objective_terms)

    @property
    def qi_objective_names(self) -> tuple[str, ...]:
        """Names of Boozer/QI objective terms assembled from tuples."""

        return tuple(term.name for term in self.qi_objective_terms)

    @property
    def objective_names(self) -> tuple[str, ...]:
        """All objective names in residual assembly order."""

        return self.scalar_objective_names + self.qi_objective_names

    @property
    def objective_count(self) -> int:
        """Total number of objective terms assembled from user tuples."""

        return len(self.objective_terms) + len(self.qi_objective_terms)

    @property
    def summary(self) -> dict[str, object]:
        """Compact description of the assembled least-squares problem."""

        return {
            "objective_count": self.objective_count,
            "scalar_objectives": self.scalar_objective_names,
            "qi_objectives": self.qi_objective_names,
            "is_qi": self.is_qi,
            "metadata": dict(self.metadata),
        }


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

        def _make_term(base_term):
            def _evaluate(ctx, state, base_term=base_term):
                return self._projected_residual(base_term.residual(ctx, state)) * float(residual_weight)

            def _total(ctx, state):
                residual = _evaluate(ctx, state)
                return jnp.sum(residual * residual)

            return ObjectiveTerm(
                name,
                _evaluate,
                target=0.0,
                weight=1.0,
                total=_total,
                track_iota=base_term.track_iota,
                metadata=dict(base_term.metadata),
            )

        if base.prepare is None:
            return _make_term(base)

        def _prepare(ctx):
            return _make_term(base.prepare(ctx))

        return ObjectiveTerm(
            name,
            lambda ctx, state: self._projected_residual(base.residual(ctx, state)) * float(residual_weight),
            target=0.0,
            weight=1.0,
            total=lambda ctx, state: jnp.sum(
                (self._projected_residual(base.residual(ctx, state)) * float(residual_weight)) ** 2
            ),
            track_iota=base.track_iota,
            metadata=dict(base.metadata),
            prepare=_prepare,
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
            metadata={
                "objective_family": "qs",
                "helicity_m": self.helicity_m,
                "helicity_n": self.helicity_n,
            },
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
        metadata={
            "objective_family": "qs",
            "helicity_m": int(helicity_m),
            "helicity_n": int(helicity_n),
        },
    )


def _smooth_positive_part(value, *, softness: float):
    value = jnp.asarray(value, dtype=jnp.float64)
    softness = float(softness)
    if softness <= 0.0:
        return jnp.maximum(value, 0.0)
    return softness * jnp.logaddexp(value / softness, 0.0)


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


def rebuild_for_optimization_resolution(
    indata,
    *,
    max_mode: int,
    min_vmec_mode: int = 5,
    vmec_mpol: int | None = None,
    vmec_ntor: int | None = None,
):
    """Set VMEC spectral resolution for an optimization run.

    This wrapper preserves the historical workflow-level
    ``rebuild_indata_with_resolution`` seam used by tests and diagnostics.  The
    reusable implementation lives in ``optimizers.fixed_boundary.seed_inputs``.
    """

    floor = max(int(min_vmec_mode), int(max_mode) + 2)
    mpol = max(1, int(vmec_mpol)) if vmec_mpol is not None else floor
    ntor = max(0, int(vmec_ntor)) if vmec_ntor is not None else floor
    return rebuild_indata_with_resolution(indata, mpol=mpol, ntor=ntor)


def _indata_get_int(indata, key: str, default: int) -> int:
    getter = getattr(indata, "get_int", None)
    if callable(getter):
        return int(getter(key, default))
    scalars = getattr(indata, "scalars", None)
    if isinstance(scalars, dict) and key in scalars:
        return int(scalars[key])
    return int(default)


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
    freeze_initial_axis: bool = False,
) -> FixedBoundaryObjectiveStage:
    """Build one VMEC/JAX optimization stage from an objective list."""

    stage_indata0 = (
        truncate_indata_boundary_modes(
            indata,
            max_mode=stage_mode,
            max_m=stage_max_m,
            max_n=stage_max_n,
        )
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
        active_max_m=stage_max_m,
        active_max_n=stage_max_n,
        min_mpol=_indata_get_int(stage_indata0, "MPOL", 5),
        min_ntor=_indata_get_int(stage_indata0, "NTOR", 5),
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
        freeze_initial_axis=freeze_initial_axis,
    )
    return FixedBoundaryObjectiveStage(
        mode=int(stage_mode),
        ctx=ctx,
        optimizer=optimizer,
        specs=specs,
        boundary_input=boundary_input,
    )


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
    scalar_cost_only_trials: bool | None = None,
    save_stage_inputs: bool = True,
    save_stage_wouts: bool = False,
    save_rerun_wouts: bool = False,
    save_final_outputs: bool = True,
) -> FixedBoundaryOptimizationResult:
    """Run a fixed-boundary objective list through one or more mode stages."""

    _enable_line_buffered_output()
    stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]] = []
    accepted_stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]] = []
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
            scalar_cost_only_trials=scalar_cost_only_trials,
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
        attempted_record = (int(stage_limits.mode), stage.optimizer, params0, result)
        stage_records.append(attempted_record)
        accepted_record = _select_nonworsening_stage_record(
            attempted_record,
            accepted_stage_records,
            stage_label=describe_boundary_mode_limits(stage_limits),
        )
        if accepted_record is attempted_record:
            accepted_stage_records.append(accepted_record)
        _accepted_mode, accepted_optimizer, _accepted_params0, accepted_result = accepted_record
        current_indata = accepted_optimizer._indata_from_params(accepted_result["x"])
        current_cfg = config_from_indata(current_indata)

    final_optimizer = accepted_stage_records[-1][1]
    final_result = accepted_stage_records[-1][3]
    combined_history = combine_qs_stage_histories(
        label=label,
        max_mode=max_mode,
        max_nfev=max_nfev,
        continuation_nfev=continuation_nfev,
        stage_modes=normalized_stage_modes,
        stage_records=accepted_stage_records,
    )
    if combined_history is not None:
        final_result["_history_dump"] = combined_history

    print_qs_final_summary(final_result, target_iota=target_iota, iota_abs_min=iota_abs_min)
    if save_final_outputs:
        save_qs_final_outputs(
            output_dir=output_dir,
            stage_records=accepted_stage_records,
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
        stage_records=accepted_stage_records,
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
    freeze_initial_axis: bool = True,
) -> FixedBoundaryObjectiveStage:
    """Build one QI stage while sharing one Boozer transform across QI terms."""

    stage_indata0 = (
        truncate_indata_boundary_modes(
            indata,
            max_mode=stage_mode,
            max_m=stage_max_m,
            max_n=stage_max_n,
        )
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
        active_max_m=stage_max_m,
        active_max_n=stage_max_n,
        min_mpol=_indata_get_int(stage_indata0, "MPOL", 5),
        min_ntor=_indata_get_int(stage_indata0, "NTOR", 5),
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

    bound_scalar_objectives = tuple(term.bind(ctx) for term in scalar_objectives)

    def residuals_from_state(state, *, ctx=ctx, scalar_objectives=bound_scalar_objectives):
        field = field_eval(state)
        scalar_parts = [term.residual(ctx, state) for term in scalar_objectives]
        qi_parts = [term.residual_and_total(ctx, state, field)[0] for term in qi_objectives]
        return jnp.concatenate([*scalar_parts, *qi_parts])

    residuals_from_state._n_non_qs = len(bound_scalar_objectives)
    def _qs_total_from_state(state, *, ctx=ctx):
        field = field_eval(state)
        return float(sum(float(term.residual_and_total(ctx, state, field)[1]) for term in qi_objectives))

    residuals_from_state._qs_total_from_state = _qs_total_from_state
    residuals_from_state._objective_family = "qi"
    _attach_packed_state_autodiff_hooks(residuals_from_state)

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
        freeze_initial_axis=freeze_initial_axis,
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
    scalar_cost_only_trials: bool | None = None,
    save_stage_inputs: bool = True,
    save_stage_wouts: bool = False,
    save_final_outputs: bool = True,
) -> FixedBoundaryOptimizationResult:
    """Run a QI objective list through repeated or direct mode stages."""

    _enable_line_buffered_output()
    stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]] = []
    accepted_stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]] = []
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
            scalar_cost_only_trials=scalar_cost_only_trials,
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
        attempted_record = (int(stage_limits.mode), stage.optimizer, params0, result)
        stage_records.append(attempted_record)
        accepted_record = _select_nonworsening_stage_record(
            attempted_record,
            accepted_stage_records,
            stage_label=describe_boundary_mode_limits(stage_limits),
        )
        if accepted_record is attempted_record:
            accepted_stage_records.append(accepted_record)
        write_qi_workflow_stage_checkpoint(
            output_dir=output_dir,
            stage_dir=output_dir / f"stage_{stage_index:02d}_{describe_boundary_mode_limits(stage_limits)}",
            stage_index=stage_index,
            stage_limits=stage_limits,
            result=result,
            completed_stage_modes=[record[0] for record in stage_records],
            requested_stage_modes=normalized_stage_modes,
        )
        _accepted_mode, accepted_optimizer, _accepted_params0, accepted_result = accepted_record
        current_indata = accepted_optimizer._indata_from_params(accepted_result["x"])
        current_cfg = config_from_indata(current_indata)

    final_optimizer = accepted_stage_records[-1][1]
    final_result = accepted_stage_records[-1][3]
    combined_history = combine_qs_stage_histories(
        label=label,
        max_mode=max_mode,
        max_nfev=max_nfev,
        continuation_nfev=continuation_nfev,
        stage_modes=normalized_stage_modes,
        stage_records=accepted_stage_records,
    )
    if combined_history is not None:
        final_result["_history_dump"] = combined_history

    print_qs_final_summary(final_result, iota_abs_min=iota_abs_min)
    if save_final_outputs:
        save_qs_final_outputs(
            output_dir=output_dir,
            stage_records=accepted_stage_records,
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
        stage_records=accepted_stage_records,
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
    scalar_cost_only_trials: bool | None = None,
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
            scalar_cost_only_trials=scalar_cost_only_trials,
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
        scalar_cost_only_trials=scalar_cost_only_trials,
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
    return _workflow_outputs.print_qs_problem_summary(
        method=method,
        max_nfev=max_nfev,
        use_mode_continuation=use_mode_continuation,
        use_ess=use_ess,
        ess_alpha=ess_alpha,
        objectives=objectives,
        specs=specs,
        x_scale=x_scale,
        optimizer=optimizer,
        params0=params0,
    )


def print_qs_final_summary(
    result: dict,
    *,
    target_iota: float | None = None,
    iota_abs_min: float | None = None,
) -> None:
    return _workflow_outputs.print_qs_final_summary(
        result,
        target_iota=target_iota,
        iota_abs_min=iota_abs_min,
    )


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
    return _workflow_outputs.save_qs_stage_artifacts(
        stage_dir=stage_dir,
        optimizer=optimizer,
        params_initial=params_initial,
        params_final=params_final,
        result=result,
        save_inputs=save_inputs,
        save_wouts=save_wouts,
        save_rerun_wouts=save_rerun_wouts,
        run_fixed_boundary_func=run_fixed_boundary,
        write_wout_from_fixed_boundary_run_func=write_wout_from_fixed_boundary_run,
    )


def _result_objective_final(result) -> float:
    return _workflow_outputs.result_objective_final(result)


def _select_nonworsening_stage_record(
    attempted_record,
    accepted_stage_records,
    *,
    stage_label: str,
):
    return _workflow_outputs.select_nonworsening_stage_record(
        attempted_record,
        accepted_stage_records,
        stage_label=stage_label,
    )


def write_qi_workflow_stage_checkpoint(
    *,
    output_dir: Path,
    stage_dir: Path,
    stage_index: int,
    stage_limits,
    result: dict,
    completed_stage_modes,
    requested_stage_modes,
) -> Path:
    return _workflow_outputs.write_qi_workflow_stage_checkpoint(
        output_dir=output_dir,
        stage_dir=stage_dir,
        stage_index=stage_index,
        stage_limits=stage_limits,
        result=result,
        completed_stage_modes=completed_stage_modes,
        requested_stage_modes=requested_stage_modes,
        normalize_boundary_mode_limits_func=normalize_boundary_mode_limits,
        describe_boundary_mode_limits_func=describe_boundary_mode_limits,
        stage_mode_checkpoint_descriptor_func=_stage_mode_checkpoint_descriptor,
    )


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
    return _workflow_outputs.save_qs_final_outputs(
        output_dir=output_dir,
        stage_records=stage_records,
        final_optimizer=final_optimizer,
        final_result=final_result,
        label=label,
        target_aspect=target_aspect,
        target_iota=target_iota,
        iota_abs_min=iota_abs_min,
        save_rerun_wouts=save_rerun_wouts,
        annotate_final_history_func=annotate_qs_final_history,
        run_fixed_boundary_func=run_fixed_boundary,
        write_wout_from_fixed_boundary_run_func=write_wout_from_fixed_boundary_run,
    )


def annotate_qs_final_history(
    final_result: dict,
    *,
    label: str,
    target_aspect: float | None = None,
    target_iota: float | None = None,
    iota_abs_min: float | None = None,
) -> None:
    return _workflow_outputs.annotate_qs_final_history(
        final_result,
        label=label,
        target_aspect=target_aspect,
        target_iota=target_iota,
        iota_abs_min=iota_abs_min,
    )


def combine_qs_stage_histories(
    *,
    label: str,
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
    stage_modes,
    stage_records,
) -> dict | None:
    return _workflow_outputs.combine_qs_stage_histories(
        label=label,
        max_mode=max_mode,
        max_nfev=max_nfev,
        continuation_nfev=continuation_nfev,
        stage_modes=stage_modes,
        stage_records=stage_records,
        normalize_boundary_mode_limits_func=normalize_boundary_mode_limits,
        qs_stage_budget_func=qs_stage_budget,
    )


def _target_is_zero(target) -> bool:
    return bool(np.allclose(np.asarray(target, dtype=float), 0.0))


def _metadata_float(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    return None if value is None else float(value)


def _remove_stale(path: Path) -> None:
    return _workflow_outputs.remove_stale(path)


def _stage_mode_checkpoint_descriptor(stage_mode) -> dict[str, object]:
    return _workflow_outputs.stage_mode_checkpoint_descriptor(
        stage_mode,
        normalize_boundary_mode_limits_func=normalize_boundary_mode_limits,
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    return _workflow_outputs.write_json_atomic(path, payload)


def _json_safe(value):
    return _workflow_outputs.json_safe(value)


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
    "GlasserResistiveInterchange",
    "JDotB",
    "JVector",
    "LeastSquaresProblem",
    "LgradB",
    "MagneticWell",
    "MaxElongation",
    "MeanIota",
    "MirrorRatio",
    "VMECMirrorRatio",
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
    "write_qi_workflow_stage_checkpoint",
    "simple_omnigenity_seed_indata",
    "VolavgB",
]

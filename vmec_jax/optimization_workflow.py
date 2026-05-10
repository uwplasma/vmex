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
from typing import Callable, Sequence

import numpy as np

from ._compat import enable_x64, jnp
from .boundary import boundary_from_indata, boundary_input_from_indata
from .driver import run_fixed_boundary, write_wout_from_fixed_boundary_run
from .energy import flux_profiles_from_indata
from .field import signgs_from_sqrtg
from .finite_beta import finite_beta_scalars_from_state, mercier_terms_from_state
from .geom import eval_geom
from .init_guess import initial_guess_from_boundary
from .optimization import (
    BoundaryParamSpec,
    FixedBoundaryExactOptimizer,
    boundary_param_names,
    boundary_param_specs,
    create_x_scale,
    extend_boundary_for_max_mode,
    lift_boundary_params,
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
class FixedBoundaryOptimizationResult:
    """Result returned by :func:`run_fixed_boundary_objective_optimization`."""

    stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]]
    final_optimizer: FixedBoundaryExactOptimizer
    final_result: dict
    stage_modes: list[int]


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
        include: Sequence[str] = ("rc", "zs"),
        fix: Sequence[str] = ("rc00",),
    ) -> "FixedBoundaryVMEC":
        """Load a VMEC input file and apply the optimization resolution policy."""

        from . import load_config
        from .config import config_from_indata

        input_path = Path(input_file)
        cfg, indata = load_config(str(input_path))
        indata = rebuild_for_optimization_resolution(
            indata,
            max_mode=max_mode,
            min_vmec_mode=min_vmec_mode,
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
    softness: float = 2.0e-2
    width_weight: float = 1.0
    branch_width_weight: float = 0.5
    branch_width_softness: float = 2.0e-2
    profile_weight: float = 0.1
    shuffle_profile_weight: float = 1.0
    shuffle_profile_softness: float = 2.0e-2
    aligned_profile_weight: float = 0.0
    aligned_profile_softness: float = 2.0e-2
    aligned_profile_trap_level: float = 0.65
    aligned_profile_trap_softness: float = 5.0e-2
    phimin: float = 0.0


@dataclass(frozen=True)
class LeastSquaresProblem:
    """Least-squares objective assembled from ``(function, target, weight)`` tuples.

    As in SIMSOPT, tuple ``weight`` is an objective weight.  Internally the
    residual is ``sqrt(weight) * (function - target)``.
    """

    objective_terms: tuple[ObjectiveTerm, ...] = ()
    qi_objective_terms: tuple[QIObjectiveTerm, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    qi_options: QuasiIsodynamicOptions | None = None

    @classmethod
    def from_tuples(cls, tuples: Sequence[tuple[Callable, float | np.ndarray, float]]):
        """Create a problem from ``(callable, target, weight)`` tuples."""

        objective_terms: list[ObjectiveTerm] = []
        qi_terms: list[QIObjectiveTerm] = []
        metadata: dict[str, object] = {}
        qi_options: QuasiIsodynamicOptions | None = None
        for fn, target, weight in tuples:
            residual_weight = math.sqrt(float(weight))
            owner = getattr(fn, "__self__", None)
            if getattr(owner, "requires_qi_field", False):
                if not _target_is_zero(target):
                    raise ValueError("QI field objectives currently require target=0.")
                qi_term = owner.to_qi_term(residual_weight)
                if qi_term.qi_options is not None:
                    if qi_options is not None and qi_term.qi_options is not qi_options:
                        raise ValueError("QI field objectives in one problem must share one QuasiIsodynamicOptions object.")
                    qi_options = qi_term.qi_options
                qi_terms.append(qi_term)
            elif hasattr(owner, "to_objective_term"):
                term = owner.to_objective_term(target=target, residual_weight=residual_weight)
                metadata.update(term.metadata)
                objective_terms.append(term)
            else:
                name = getattr(fn, "__name__", "objective")
                objective_terms.append(
                    ObjectiveTerm(
                        name,
                        lambda ctx, state, fn=fn: fn(ctx, state),
                        target=target,
                        weight=residual_weight,
                    )
                )
        return cls(tuple(objective_terms), tuple(qi_terms), metadata=metadata, qi_options=qi_options)

    @property
    def is_qi(self) -> bool:
        """Whether the problem contains Boozer-space QI field objectives."""

        return bool(self.qi_objective_terms)


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
        surface_index: int = 0,
        qi_options: QuasiIsodynamicOptions | None = None,
    ):
        self.threshold = float(threshold)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)
        self.surface_index = int(surface_index)
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
        qi_options: QuasiIsodynamicOptions | None = None,
    ):
        self.threshold = float(threshold)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)
        self.qi_options = qi_options

    def J(self, _ctx: StageContext, _state):
        raise RuntimeError("MaxElongation must be evaluated inside a QI solve.")

    def to_qi_term(self, residual_weight: float) -> QIObjectiveTerm:
        return qi_max_elongation_objective(
            threshold=self.threshold,
            weight=residual_weight,
            ntheta=self.ntheta,
            nphi=self.nphi,
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
        vp = jnp.abs(jnp.asarray(scalars["vp"], dtype=jnp.float64))
        dvol = vp[1:]
        if int(dvol.shape[0]) < 2:
            return jnp.asarray(0.0, dtype=jnp.float64)
        dvol_s0 = 1.5 * dvol[0] - 0.5 * dvol[1]
        dvol_s1 = 1.5 * dvol[-1] - 0.5 * dvol[-2]
        return jnp.where(dvol_s0 != 0.0, (dvol_s0 - dvol_s1) / dvol_s0, 0.0)

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
    interior radial surfaces.  It currently uses the differentiable
    stellarator-symmetric Mercier path; LASYM=True raises until that spectral
    derivative branch is wired.
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


def qi_mirror_ratio_objective(
    *,
    threshold: float,
    weight: float = 1.0,
    ntheta: int = 96,
    nphi: int = 96,
    surface_index: int = 0,
    qi_options: QuasiIsodynamicOptions | None = None,
) -> QIObjectiveTerm:
    """Mirror-ratio upper-bound objective evaluated from Boozer |B| modes."""

    def _evaluate(ctx: StageContext, _state, field: dict):
        mirror_booz = _slice_boozer_surfaces(field["booz"], int(surface_index))
        mirror = mirror_ratio_penalty_from_boozer_output(
            mirror_booz,
            nfp=int(ctx.static.cfg.nfp),
            threshold=float(threshold),
            ntheta=int(ntheta),
            nphi=int(nphi),
        )
        return (
            jnp.asarray(mirror["residuals1d"], dtype=jnp.float64) * float(weight),
            float(weight) ** 2 * mirror["total"],
        )

    return QIObjectiveTerm("mirror_ratio", _evaluate, qi_options=qi_options)


def qi_max_elongation_objective(
    *,
    threshold: float,
    weight: float = 1.0,
    ntheta: int = 48,
    nphi: int = 16,
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
        )
        return (
            jnp.asarray(elongation["residuals1d"], dtype=jnp.float64) * float(weight),
            float(weight) ** 2 * elongation["total"],
        )

    return QIObjectiveTerm("max_elongation", _evaluate, qi_options=qi_options)


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
    """Same-mode repeated continuation used by the QI example."""

    if bool(use_mode_continuation) and int(max_mode) > 1 and int(continuation_nfev) > 0:
        return [int(max_mode)] * int(repeats)
    return [int(max_mode)]


def qs_stage_budget(
    *,
    stage_mode: int,
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
) -> int:
    """Outer residual/Jacobian budget for one fixed-boundary stage."""

    return int(max_nfev) if int(stage_mode) == int(max_mode) else int(continuation_nfev)


def rebuild_for_optimization_resolution(indata, *, max_mode: int, min_vmec_mode: int = 5):
    """Set VMEC spectral resolution to at least ``max(min_vmec_mode, max_mode+2)``."""

    vmec_mpol = max(int(min_vmec_mode), int(max_mode) + 2)
    return rebuild_indata_with_resolution(indata, mpol=vmec_mpol, ntor=vmec_mpol)


def build_fixed_boundary_objective_stage(
    cfg,
    indata,
    *,
    stage_mode: int,
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
    scipy_tr_solver: str | None = "lsmr",
    scipy_lsmr_maxiter: int | None = None,
    save_stage_inputs: bool = True,
    save_stage_wouts: bool = False,
    save_rerun_wouts: bool = False,
) -> FixedBoundaryOptimizationResult:
    """Run a fixed-boundary objective list through one or more mode stages."""

    stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]] = []
    params_stage = None
    prev_specs = None

    for stage_index, stage_mode in enumerate(stage_modes, start=1):
        stage = build_fixed_boundary_objective_stage(
            cfg,
            indata,
            stage_mode=int(stage_mode),
            objectives=objectives,
            include=include,
            fix=fix,
            project_input_boundary_to_max_mode=project_input_boundary_to_max_mode,
            inner_max_iter=inner_max_iter,
            inner_ftol=inner_ftol,
            trial_max_iter=trial_max_iter,
            trial_ftol=trial_ftol,
            solver_device=solver_device,
        )
        x_scale = (
            create_x_scale(stage.specs, alpha=float(ess_alpha))
            if bool(use_ess)
            else np.ones(len(stage.specs), dtype=float)
        )
        params0 = (
            np.zeros(len(stage.specs), dtype=float)
            if params_stage is None
            else np.asarray(lift_boundary_params(prev_specs, params_stage, stage.specs), dtype=float)
        )
        nfev = qs_stage_budget(
            stage_mode=int(stage_mode),
            max_mode=int(max_mode),
            max_nfev=int(max_nfev),
            continuation_nfev=int(continuation_nfev),
        )
        iota_fn = (
            (lambda state, ctx=stage.ctx: float(mean_iota(ctx, state)))
            if objectives_track_iota(objectives, target_iota=target_iota) or iota_abs_min is not None
            else None
        )

        if int(stage_mode) == int(max_mode):
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
            print(f"Stage {stage_mode} -> {stage_mode + 1} continuation seed (budget={nfev}) ...")

        result = stage.optimizer.run(
            params0,
            method=method,
            max_nfev=nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            x_scale=x_scale,
            verbose=1 if int(stage_mode) == int(max_mode) else 0,
            iota_fn=iota_fn,
            target_iota=target_iota,
            target_aspect=target_aspect,
            scipy_tr_solver=scipy_tr_solver,
            scipy_lsmr_maxiter=scipy_lsmr_maxiter,
        )
        if iota_abs_min is not None:
            result["_history_dump"]["iota_abs_min"] = float(iota_abs_min)
        save_qs_stage_artifacts(
            stage_dir=output_dir / f"stage_{stage_index:02d}_mode{int(stage_mode):02d}",
            optimizer=stage.optimizer,
            params_initial=params0,
            params_final=result["x"],
            result=result,
            save_inputs=save_stage_inputs,
            save_wouts=save_stage_wouts,
            save_rerun_wouts=save_rerun_wouts,
        )
        stage_records.append((int(stage_mode), stage.optimizer, params0, result))
        prev_specs = stage.specs
        params_stage = result["x"]

    final_optimizer = stage_records[-1][1]
    final_result = stage_records[-1][3]
    combined_history = combine_qs_stage_histories(
        label=label,
        max_mode=max_mode,
        max_nfev=max_nfev,
        continuation_nfev=continuation_nfev,
        stage_modes=stage_modes,
        stage_records=stage_records,
    )
    if combined_history is not None:
        final_result["_history_dump"] = combined_history

    print_qs_final_summary(final_result, target_iota=target_iota, iota_abs_min=iota_abs_min)
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
    return FixedBoundaryOptimizationResult(
        stage_records=stage_records,
        final_optimizer=final_optimizer,
        final_result=final_result,
        stage_modes=[int(mode) for mode in stage_modes],
    )


def build_quasi_isodynamic_objective_stage(
    cfg,
    indata,
    *,
    stage_mode: int,
    scalar_objectives: Sequence[ObjectiveTerm],
    qi_objectives: Sequence[QIObjectiveTerm],
    surfaces,
    mboz: int,
    nboz: int,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    softness: float,
    width_weight: float,
    branch_width_weight: float,
    branch_width_softness: float,
    profile_weight: float,
    shuffle_profile_weight: float,
    shuffle_profile_softness: float,
    aligned_profile_weight: float,
    aligned_profile_softness: float,
    aligned_profile_trap_level: float,
    aligned_profile_trap_softness: float,
    phimin: float,
    project_input_boundary_to_max_mode: bool = True,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    inner_max_iter: int = 120,
    inner_ftol: float = 1.0e-9,
    trial_max_iter: int = 120,
    trial_ftol: float = 1.0e-9,
    solver_device: str | None = None,
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
            softness=float(softness),
            width_weight=float(width_weight),
            branch_width_weight=float(branch_width_weight),
            branch_width_softness=float(branch_width_softness),
            profile_weight=float(profile_weight),
            shuffle_profile_weight=float(shuffle_profile_weight),
            shuffle_profile_softness=float(shuffle_profile_softness),
            aligned_profile_weight=float(aligned_profile_weight),
            aligned_profile_softness=float(aligned_profile_softness),
            aligned_profile_trap_level=float(aligned_profile_trap_level),
            aligned_profile_trap_softness=float(aligned_profile_trap_softness),
            phimin=float(phimin),
            jit_booz=False,
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
    residuals_from_state._qs_total_from_state = lambda state, ctx=ctx: float(
        sum(float(term.residual_and_total(ctx, state, field_eval(state))[1]) for term in qi_objectives)
    )

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
    softness: float,
    width_weight: float,
    branch_width_weight: float,
    branch_width_softness: float,
    profile_weight: float,
    shuffle_profile_weight: float,
    shuffle_profile_softness: float,
    aligned_profile_weight: float,
    aligned_profile_softness: float,
    aligned_profile_trap_level: float,
    aligned_profile_trap_softness: float,
    phimin: float,
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
    scipy_tr_solver: str | None = "lsmr",
    scipy_lsmr_maxiter: int | None = None,
    save_stage_inputs: bool = True,
    save_stage_wouts: bool = False,
) -> FixedBoundaryOptimizationResult:
    """Run a QI objective list through repeated or direct mode stages."""

    stage_records: list[tuple[int, FixedBoundaryExactOptimizer, np.ndarray, dict]] = []
    params_stage = None
    prev_specs = None

    for stage_index, stage_mode in enumerate(stage_modes, start=1):
        stage = build_quasi_isodynamic_objective_stage(
            cfg,
            indata,
            stage_mode=int(stage_mode),
            scalar_objectives=scalar_objectives,
            qi_objectives=qi_objectives,
            surfaces=surfaces,
            mboz=mboz,
            nboz=nboz,
            nphi=nphi,
            nalpha=nalpha,
            n_bounce=n_bounce,
            softness=softness,
            width_weight=width_weight,
            branch_width_weight=branch_width_weight,
            branch_width_softness=branch_width_softness,
            profile_weight=profile_weight,
            shuffle_profile_weight=shuffle_profile_weight,
            shuffle_profile_softness=shuffle_profile_softness,
            aligned_profile_weight=aligned_profile_weight,
            aligned_profile_softness=aligned_profile_softness,
            aligned_profile_trap_level=aligned_profile_trap_level,
            aligned_profile_trap_softness=aligned_profile_trap_softness,
            phimin=phimin,
            project_input_boundary_to_max_mode=project_input_boundary_to_max_mode,
            include=include,
            fix=fix,
            inner_max_iter=inner_max_iter,
            inner_ftol=inner_ftol,
            trial_max_iter=trial_max_iter,
            trial_ftol=trial_ftol,
            solver_device=solver_device,
        )
        x_scale = (
            create_x_scale(stage.specs, alpha=float(ess_alpha))
            if bool(use_ess)
            else np.ones(len(stage.specs), dtype=float)
        )
        params0 = (
            np.zeros(len(stage.specs), dtype=float)
            if params_stage is None
            else np.asarray(lift_boundary_params(prev_specs, params_stage, stage.specs), dtype=float)
        )
        nfev = qs_stage_budget(
            stage_mode=int(stage_mode),
            max_mode=int(max_mode),
            max_nfev=int(max_nfev),
            continuation_nfev=int(continuation_nfev),
        )
        iota_fn = (
            (lambda state, ctx=stage.ctx: float(mean_iota(ctx, state)))
            if objectives_track_iota(scalar_objectives) or iota_abs_min is not None
            else None
        )
        if int(stage_mode) == int(max_mode):
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
            print(f"Stage {stage_mode} -> {stage_mode + 1} continuation seed (budget={nfev}) ...")

        result = stage.optimizer.run(
            params0,
            method=method,
            max_nfev=nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            x_scale=x_scale,
            verbose=1 if int(stage_mode) == int(max_mode) else 0,
            iota_fn=iota_fn,
            target_aspect=target_aspect,
            scipy_tr_solver=scipy_tr_solver,
            scipy_lsmr_maxiter=scipy_lsmr_maxiter,
        )
        if iota_abs_min is not None:
            result["_history_dump"]["iota_abs_min"] = float(iota_abs_min)
        save_qs_stage_artifacts(
            stage_dir=output_dir / f"stage_{stage_index:02d}_mode{int(stage_mode):02d}",
            optimizer=stage.optimizer,
            params_initial=params0,
            params_final=result["x"],
            result=result,
            save_inputs=save_stage_inputs,
            save_wouts=save_stage_wouts,
        )
        stage_records.append((int(stage_mode), stage.optimizer, params0, result))
        prev_specs = stage.specs
        params_stage = result["x"]

    final_optimizer = stage_records[-1][1]
    final_result = stage_records[-1][3]
    combined_history = combine_qs_stage_histories(
        label=label,
        max_mode=max_mode,
        max_nfev=max_nfev,
        continuation_nfev=continuation_nfev,
        stage_modes=stage_modes,
        stage_records=stage_records,
    )
    if combined_history is not None:
        final_result["_history_dump"] = combined_history

    print_qs_final_summary(final_result, iota_abs_min=iota_abs_min)
    save_qs_final_outputs(
        output_dir=output_dir,
        stage_records=stage_records,
        final_optimizer=final_optimizer,
        final_result=final_result,
        label=label,
        target_aspect=target_aspect,
        iota_abs_min=iota_abs_min,
    )
    return FixedBoundaryOptimizationResult(
        stage_records=stage_records,
        final_optimizer=final_optimizer,
        final_result=final_result,
        stage_modes=[int(mode) for mode in stage_modes],
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
    scipy_tr_solver: str | None = "lsmr",
    scipy_lsmr_maxiter: int | None = None,
    save_stage_inputs: bool = True,
    save_stage_wouts: bool = False,
    save_rerun_wouts: bool = False,
) -> FixedBoundaryOptimizationResult:
    """Solve a SIMSOPT-style fixed-boundary least-squares problem.

    The examples use this as the common public workflow:

    1. create a :class:`FixedBoundaryVMEC`,
    2. assemble a :class:`LeastSquaresProblem` from ``(J, target, weight)``
       tuples,
    3. choose stage modes and optimizer settings,
    4. call this function.
    """

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
            softness=qi_options.softness,
            width_weight=qi_options.width_weight,
            branch_width_weight=qi_options.branch_width_weight,
            branch_width_softness=qi_options.branch_width_softness,
            profile_weight=qi_options.profile_weight,
            shuffle_profile_weight=qi_options.shuffle_profile_weight,
            shuffle_profile_softness=qi_options.shuffle_profile_softness,
            aligned_profile_weight=qi_options.aligned_profile_weight,
            aligned_profile_softness=qi_options.aligned_profile_softness,
            aligned_profile_trap_level=qi_options.aligned_profile_trap_level,
            aligned_profile_trap_softness=qi_options.aligned_profile_trap_softness,
            phimin=qi_options.phimin,
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
            scipy_tr_solver=scipy_tr_solver,
            scipy_lsmr_maxiter=scipy_lsmr_maxiter,
            save_stage_inputs=save_stage_inputs,
            save_stage_wouts=save_stage_wouts,
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
        scipy_tr_solver=scipy_tr_solver,
        scipy_lsmr_maxiter=scipy_lsmr_maxiter,
        save_stage_inputs=save_stage_inputs,
        save_stage_wouts=save_stage_wouts,
        save_rerun_wouts=save_rerun_wouts,
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

    history = final_result["_history_dump"]
    history["label"] = label
    if target_aspect is not None:
        history["target_aspect"] = float(target_aspect)
    if target_iota is not None:
        history["target_iota"] = float(target_iota)
    if iota_abs_min is not None:
        history["iota_abs_min"] = float(iota_abs_min)
    final_optimizer.save_history(output_dir / "history.json", final_result)


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
                    int(max_nfev) if int(mode) == int(max_mode) else int(continuation_nfev)
                    for mode in stage_modes
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


def _remove_stale(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _slice_boozer_surfaces(booz: dict, surface_index: int) -> dict:
    index = int(surface_index)
    out = dict(booz)
    for key in ("bmnc_b", "bmns_b", "iota_b", "s_b"):
        value = out.get(key)
        if value is not None:
            out[key] = value[index : index + 1]
    return out


__all__ = [
    "AbsMeanIotaFloor",
    "AspectRatio",
    "BetaTotal",
    "DMerc",
    "FixedBoundaryVMEC",
    "FixedBoundaryObjectiveStage",
    "FixedBoundaryOptimizationResult",
    "LeastSquaresProblem",
    "LgradB",
    "MagneticWell",
    "MaxElongation",
    "MeanIota",
    "MirrorRatio",
    "ObjectiveTerm",
    "QuasiIsodynamicOptions",
    "QuasiIsodynamicResidual",
    "QuasisymmetryRatioResidual",
    "QIObjectiveTerm",
    "StageContext",
    "abs_mean_iota_floor_objective",
    "aspect_objective",
    "build_fixed_boundary_objective_stage",
    "build_quasi_isodynamic_objective_stage",
    "combine_qs_stage_histories",
    "lgradb_objective",
    "least_squares_solve",
    "mean_iota",
    "mean_iota_objective",
    "objectives_track_iota",
    "qs_stage_budget",
    "qs_stage_modes",
    "qi_lgradb_objective",
    "qi_max_elongation_objective",
    "qi_mirror_ratio_objective",
    "quasi_isodynamic_field_objective",
    "quasisymmetry_objective",
    "rebuild_for_optimization_resolution",
    "repeated_stage_modes",
    "residuals_from_objectives",
    "run_fixed_boundary_objective_optimization",
    "run_quasi_isodynamic_objective_optimization",
    "save_qs_final_outputs",
    "save_qs_stage_artifacts",
    "VolavgB",
]

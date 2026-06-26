"""QI, Boozer, mirror-ratio, elongation, and L_grad_B objectives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._compat import jnp
from ..field import b_cartesian_from_state
from ..modes import nyquist_mode_table_from_grid, vmec_mode_table
from .objectives import (
    lgradb_penalty_from_state,
    max_elongation_penalty_from_state,
    mirror_ratio_penalty_from_boozer_output,
    mirror_ratio_penalty_from_state,
)
from .objectives import _smooth_reduce_max, _smooth_reduce_min
from ..optimizers.fixed_boundary.objective_terms import ObjectiveTerm
from ..optimizers.fixed_boundary.objective_terms import QIObjectiveTerm
from ..optimizers.fixed_boundary.objective_terms import StageContext


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


def _as_sequence(value) -> tuple:
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _target_is_zero(target) -> bool:
    return bool(np.allclose(np.asarray(target, dtype=float), 0.0))


def _smooth_positive_part(value, *, softness: float):
    value = jnp.asarray(value, dtype=jnp.float64)
    softness = float(softness)
    if softness <= 0.0:
        return jnp.maximum(value, 0.0)
    return softness * jnp.logaddexp(value / softness, 0.0)


def _slice_boozer_surfaces(booz: dict, surface_index: int) -> dict:
    bmnc = booz.get("bmnc_b")
    if bmnc is None:
        raise ValueError("Boozer output must include bmnc_b to slice surfaces.")
    nsurf = int(np.asarray(bmnc).shape[0])
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
    """Maximum mirror-ratio penalty object for solved VMEC states."""

    name = "mirror_ratio"

    def __init__(
        self,
        *,
        threshold: float,
        surfaces=(1.0,),
        mboz: int = 18,
        nboz: int = 18,
        ntheta: int = 96,
        nphi: int = 96,
        surface_index: int | None = None,
        phimin: float = 0.0,
        smooth_extrema: float = 0.0,
        smooth_penalty: float = 0.0,
        normalize_surfaces: bool = True,
        jit_booz: bool = True,
        qi_options: QuasiIsodynamicOptions | None = None,
    ):
        self.threshold = float(threshold)
        self.surfaces = tuple(float(value) for value in _as_sequence(surfaces))
        self.mboz = int(mboz)
        self.nboz = int(nboz)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)
        self.surface_index = None if surface_index is None else int(surface_index)
        self.phimin = float(phimin)
        self.smooth_extrema = float(smooth_extrema)
        self.smooth_penalty = float(smooth_penalty)
        self.normalize_surfaces = bool(normalize_surfaces)
        self.jit_booz = bool(jit_booz)
        self.qi_options = qi_options

    @property
    def requires_qi_field(self) -> bool:
        return self.qi_options is not None

    def _selected_surfaces_and_weights(self) -> tuple[tuple[float, ...], list[float] | None]:
        surfaces = self.surfaces
        if not surfaces:
            raise ValueError("MirrorRatio surfaces must contain at least one surface.")
        if self.surface_index is not None:
            idx = int(self.surface_index)
            if idx < 0:
                idx += len(surfaces)
            if idx < 0 or idx >= len(surfaces):
                raise IndexError(
                    f"surface_index {self.surface_index} is outside MirrorRatio surface range 0..{len(surfaces) - 1}"
                )
            return (surfaces[idx],), None
        if bool(self.normalize_surfaces):
            return surfaces, [1.0 / float(max(len(surfaces), 1))] * len(surfaces)
        return surfaces, None

    def _prepare_boozer_constants(self, ctx: StageContext):
        try:
            from booz_xform_jax import prepare_booz_xform_constants
        except Exception as exc:  # pragma: no cover - optional dependency import
            raise ImportError(
                "MirrorRatio requires booz_xform_jax. Install it with `pip install booz_xform_jax`."
            ) from exc

        cfg = ctx.static.cfg
        main_modes = vmec_mode_table(int(cfg.mpol), int(cfg.ntor))
        nyq_modes = nyquist_mode_table_from_grid(
            mpol=int(cfg.mpol),
            ntor=int(cfg.ntor),
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
        )
        return prepare_booz_xform_constants(
            nfp=int(cfg.nfp),
            mboz=self.mboz,
            nboz=self.nboz,
            asym=bool(cfg.lasym),
            xm=np.asarray(main_modes.m, dtype=np.int32),
            xn=np.asarray(main_modes.n * int(cfg.nfp), dtype=np.int32),
            xm_nyq=np.asarray(nyq_modes.m, dtype=np.int32),
            xn_nyq=np.asarray(nyq_modes.n * int(cfg.nfp), dtype=np.int32),
        )

    def _evaluate_state(self, ctx: StageContext, state, *, booz_constants=None, booz_grids=None):
        surfaces, weights = self._selected_surfaces_and_weights()
        return mirror_ratio_penalty_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            surfaces=surfaces,
            weights=weights,
            mboz=self.mboz,
            nboz=self.nboz,
            ntheta=self.ntheta,
            nphi=self.nphi,
            phimin=self.phimin,
            smooth_extrema=self.smooth_extrema,
            smooth_penalty=self.smooth_penalty,
            flux_local=ctx.flux,
            prof_local={"pressure": ctx.pressure},
            pressure_local=ctx.pressure,
            jit_booz=self.jit_booz,
            booz_constants=booz_constants,
            booz_grids=booz_grids,
        )

    def J(self, ctx: StageContext, state):
        if self.qi_options is not None:
            raise RuntimeError("MirrorRatio with qi_options must be evaluated inside a QI solve.")
        return self._evaluate_state(ctx, state)["residuals1d"]

    def total(self, ctx: StageContext, state):
        if self.qi_options is not None:
            raise RuntimeError("MirrorRatio with qi_options must be evaluated inside a QI solve.")
        return self._evaluate_state(ctx, state)["total"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        if not _target_is_zero(target):
            raise ValueError("MirrorRatio is an upper-bound penalty and requires target=0.")

        def _prepare(ctx: StageContext):
            booz_constants, booz_grids = self._prepare_boozer_constants(ctx)

            def _evaluate(ctx_p: StageContext, state):
                return self._evaluate_state(
                    ctx_p,
                    state,
                    booz_constants=booz_constants,
                    booz_grids=booz_grids,
                )["residuals1d"]

            def _total(ctx_p: StageContext, state):
                return self._evaluate_state(
                    ctx_p,
                    state,
                    booz_constants=booz_constants,
                    booz_grids=booz_grids,
                )["total"]

            return ObjectiveTerm(
                self.name,
                _evaluate,
                target=0.0,
                weight=residual_weight,
                total=lambda ctx_p, state: float(residual_weight) ** 2 * _total(ctx_p, state),
            )

        return ObjectiveTerm(
            self.name,
            self.J,
            target=0.0,
            weight=residual_weight,
            total=lambda ctx, state: float(residual_weight) ** 2 * self.total(ctx, state),
            prepare=_prepare,
        )

    def to_constraint_term(self) -> ObjectiveTerm:
        def _evaluate(ctx: StageContext, state, *, booz_constants=None, booz_grids=None):
            surfaces, weights = self._selected_surfaces_and_weights()
            mirror = mirror_ratio_penalty_from_state(
                state=state,
                static=ctx.static,
                indata=ctx.indata,
                signgs=ctx.signgs,
                surfaces=surfaces,
                weights=weights,
                mboz=self.mboz,
                nboz=self.nboz,
                ntheta=self.ntheta,
                nphi=self.nphi,
                phimin=self.phimin,
                smooth_extrema=self.smooth_extrema,
                smooth_penalty=0.0,
                flux_local=ctx.flux,
                prof_local={"pressure": ctx.pressure},
                pressure_local=ctx.pressure,
                jit_booz=self.jit_booz,
                booz_constants=booz_constants,
                booz_grids=booz_grids,
            )
            residuals = jnp.asarray(mirror["mirror_ratio"], dtype=jnp.float64) - float(self.threshold)
            if weights is not None:
                residuals = residuals * jnp.sqrt(jnp.asarray(weights, dtype=jnp.float64))
            return residuals

        def _prepare(ctx: StageContext):
            booz_constants, booz_grids = self._prepare_boozer_constants(ctx)

            def _evaluate_prepared(ctx_p: StageContext, state):
                return _evaluate(
                    ctx_p,
                    state,
                    booz_constants=booz_constants,
                    booz_grids=booz_grids,
                )

            return ObjectiveTerm(f"{self.name}_constraint", _evaluate_prepared, target=0.0, weight=1.0)

        return ObjectiveTerm(f"{self.name}_constraint", _evaluate, target=0.0, weight=1.0, prepare=_prepare)

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


class VMECMirrorRatio:
    """Fast mirror-ratio penalty evaluated directly from VMEC ``|B|``."""

    name = "mirror_ratio"

    def __init__(
        self,
        *,
        threshold: float,
        surfaces=(1.0,),
        surface_index: int | None = None,
        ntheta: int | None = None,
        nphi: int | None = None,
        nzeta: int | None = None,
        smooth_extrema: float = 0.0,
        smooth_penalty: float = 0.0,
        normalize_surfaces: bool = True,
        bmag_floor: float = 1.0e-300,
    ):
        self.threshold = float(threshold)
        self.surfaces = tuple(float(value) for value in _as_sequence(surfaces))
        self.surface_index = None if surface_index is None else int(surface_index)
        self.smooth_extrema = float(smooth_extrema)
        self.smooth_penalty = float(smooth_penalty)
        self.normalize_surfaces = bool(normalize_surfaces)
        self.bmag_floor = float(bmag_floor)
        self.requested_ntheta = None if ntheta is None else int(ntheta)
        if nphi is not None and nzeta is not None and int(nphi) != int(nzeta):
            raise ValueError("VMECMirrorRatio accepts either nphi or nzeta, not conflicting values.")
        requested_nzeta = nzeta if nzeta is not None else nphi
        self.requested_nzeta = None if requested_nzeta is None else int(requested_nzeta)

    @property
    def requires_qi_field(self) -> bool:
        return False

    def _selected_surface_indices_and_weights(self, ctx: StageContext) -> tuple[list[int], jnp.ndarray]:
        surfaces = self.surfaces
        if not surfaces:
            raise ValueError("VMECMirrorRatio surfaces must contain at least one surface.")
        if self.surface_index is not None:
            idx = int(self.surface_index)
            if idx < 0:
                idx += len(surfaces)
            if idx < 0 or idx >= len(surfaces):
                raise IndexError(
                    f"surface_index {self.surface_index} is outside VMECMirrorRatio surface range 0..{len(surfaces) - 1}"
                )
            surfaces = (surfaces[idx],)
        s_grid = np.asarray(ctx.static.s, dtype=float)
        indices = [int(np.argmin(np.abs(s_grid - float(surface)))) for surface in surfaces]
        if bool(self.normalize_surfaces):
            weights = jnp.full((len(indices),), 1.0 / float(max(len(indices), 1)), dtype=jnp.float64)
        else:
            weights = jnp.ones((len(indices),), dtype=jnp.float64)
        return indices, weights

    def _evaluate_state(self, ctx: StageContext, state):
        indices, weights = self._selected_surface_indices_and_weights(ctx)
        ratios = []
        bmax_values = []
        bmin_values = []
        tiny = jnp.asarray(jnp.finfo(jnp.float64).tiny, dtype=jnp.float64)
        for s_index in indices:
            bcart = b_cartesian_from_state(
                state,
                ctx.static,
                indata=ctx.indata,
                signgs=ctx.signgs,
                s_index=int(s_index),
            )
            bcart = jnp.asarray(bcart, dtype=jnp.float64)
            bmag = jnp.sqrt(
                jnp.maximum(
                    jnp.sum(bcart * bcart, axis=-1),
                    jnp.asarray(self.bmag_floor, dtype=jnp.float64),
                )
            )
            bmax = _smooth_reduce_max(bmag, axis=(0, 1), softness=float(self.smooth_extrema))
            bmin = _smooth_reduce_min(bmag, axis=(0, 1), softness=float(self.smooth_extrema))
            bmin_positive = jnp.maximum(bmin, tiny)
            denom = jnp.maximum(bmax + bmin_positive, tiny)
            ratios.append((bmax - bmin_positive) / denom)
            bmax_values.append(bmax)
            bmin_values.append(bmin)
        mirror_ratio = jnp.asarray(ratios, dtype=jnp.float64)
        penalty = _smooth_positive_part(mirror_ratio - float(self.threshold), softness=float(self.smooth_penalty))
        residuals1d = penalty * jnp.sqrt(weights)
        total = jnp.sum(residuals1d * residuals1d)
        return {
            "residuals1d": residuals1d,
            "total": total,
            "penalty": penalty,
            "mirror_ratio": mirror_ratio,
            "bmax": jnp.asarray(bmax_values, dtype=jnp.float64),
            "bmin": jnp.asarray(bmin_values, dtype=jnp.float64),
            "threshold": jnp.asarray(float(self.threshold), dtype=jnp.float64),
        }

    def J(self, ctx: StageContext, state):
        return self._evaluate_state(ctx, state)["residuals1d"]

    def total(self, ctx: StageContext, state):
        return self._evaluate_state(ctx, state)["total"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        if not _target_is_zero(target):
            raise ValueError("VMECMirrorRatio is an upper-bound penalty and requires target=0.")
        return ObjectiveTerm(
            self.name,
            self.J,
            target=0.0,
            weight=residual_weight,
            total=lambda ctx, state: float(residual_weight) ** 2 * self.total(ctx, state),
        )


class BoozerBTarget:
    """Boozer ``|B|`` spectrum-matching objective for QI steering."""

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
    """Maximum LCFS elongation penalty object for solved VMEC states."""

    name = "max_elongation"

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

    @property
    def requires_qi_field(self) -> bool:
        return False

    def _evaluate_state(self, ctx: StageContext, state, *, smooth_penalty: float | None = None):
        return max_elongation_penalty_from_state(
            state=state,
            static=ctx.static,
            threshold=self.threshold,
            ntheta=self.ntheta,
            nphi=self.nphi,
            smooth_extrema=self.smooth_extrema,
            smooth_penalty=self.smooth_penalty if smooth_penalty is None else float(smooth_penalty),
        )

    def J(self, ctx: StageContext, state):
        return self._evaluate_state(ctx, state)["residuals1d"]

    def total(self, ctx: StageContext, state):
        return self._evaluate_state(ctx, state)["total"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        if not _target_is_zero(target):
            raise ValueError("MaxElongation is an upper-bound penalty and requires target=0.")
        return ObjectiveTerm(
            self.name,
            self.J,
            target=0.0,
            weight=residual_weight,
            total=lambda ctx, state: float(residual_weight) ** 2 * self.total(ctx, state),
        )

    def to_constraint_term(self) -> ObjectiveTerm:
        def _evaluate(ctx: StageContext, state):
            elongation = self._evaluate_state(ctx, state, smooth_penalty=0.0)
            return jnp.asarray([elongation["max_elongation"] - float(self.threshold)], dtype=jnp.float64)

        return ObjectiveTerm(f"{self.name}_constraint", _evaluate, target=0.0, weight=1.0)

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
    """Smooth QI residual term from a shared QI field evaluation."""

    def _evaluate(_ctx: StageContext, _state, field: dict):
        return (
            jnp.asarray(field["residuals1d"], dtype=jnp.float64) * float(weight),
            float(weight) ** 2 * field["total"],
        )

    return QIObjectiveTerm("qi", _evaluate, qi_options=qi_options)


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


def boozer_b_target_from_wout(
    wout_path: str | Path,
    *,
    surfaces,
    mboz: int,
    nboz: int,
) -> dict[str, np.ndarray | int]:
    """Return Boozer ``|B|`` target spectra from a VMEC ``wout`` file."""

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


__all__ = [
    "BoozerBTarget",
    "LgradB",
    "MaxElongation",
    "MirrorRatio",
    "QuasiIsodynamicOptions",
    "QuasiIsodynamicResidual",
    "QuasiIsodynamicResidualCeiling",
    "VMECMirrorRatio",
    "boozer_b_target_from_wout",
    "lgradb_objective",
    "qi_boozer_b_target_objective",
    "qi_lgradb_objective",
    "qi_max_elongation_constraint",
    "qi_max_elongation_objective",
    "qi_mirror_ratio_constraint",
    "qi_mirror_ratio_objective",
    "qi_residual_ceiling_objective",
    "quasi_isodynamic_field_objective",
]

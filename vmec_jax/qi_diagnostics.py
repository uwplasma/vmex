"""First-class quasi-isodynamic diagnostic records.

This module centralizes the unweighted QI diagnostics used by optimization
audits.  It deliberately wraps the existing smooth QI, legacy QI,
mirror-ratio, elongation, and LgradB implementations instead of defining a new
objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .qi_legacy import legacy_qi_branch_shuffle_diagnostic_from_boozer_output
from .quasi_isodynamic import (
    lgradb_penalty_from_state,
    max_elongation_penalty_from_state,
    mirror_ratio_penalty_from_boozer_output,
    quasi_isodynamic_residual_from_boozer_output,
    quasi_isodynamic_residual_from_state,
)
from .wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

QI_DIAGNOSTIC_VERSION = "qi_diagnostics.v1"


@dataclass(frozen=True)
class QIDiagnosticOptions:
    """Resolution and threshold controls for QI diagnostic records."""

    surfaces: object | None = None
    mboz: int | None = None
    nboz: int | None = None
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
    include_legacy: bool = True
    legacy_nphi_out: int | None = None
    mirror_threshold: float = 0.21
    mirror_ntheta: int = 128
    mirror_nphi: int = 128
    mirror_surface_index: int | None = None
    elongation_threshold: float = 8.0
    elongation_ntheta: int = 64
    elongation_nphi: int = 24
    include_lgradb: bool = False
    lgradb_threshold: float = 0.30
    lgradb_surface_index: int = -1
    lgradb_ntheta: int = 9
    lgradb_nphi: int = 7
    lgradb_smooth_penalty: float = 0.0
    fail_on_error: bool = False


@dataclass(frozen=True)
class QISeedSuitabilityTargets:
    """Promotion gates used to compare solved QI seed candidates.

    ``None`` disables a gate.  The defaults match the lightweight QI audit and
    optimization examples in this repository: a differentiable smooth-QI gate
    for optimization evidence, an independent legacy Goodman-style QI gate,
    then nonzero transform, aspect, mirror, and elongation cleanup gates.
    """

    smooth_qi_max: float | None = 3.0e-3
    legacy_qi_max: float | None = 2.0e-3
    target_aspect: float | None = 5.0
    aspect_relative_tolerance: float = 0.35
    aspect_max: float | None = None
    abs_iota_min: float | None = 0.41
    mirror_ratio_max: float | None = 0.21
    max_elongation: float | None = 8.0


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _as_options(options: QIDiagnosticOptions | None) -> QIDiagnosticOptions:
    return QIDiagnosticOptions() if options is None else options


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return float(arr.ravel()[0])


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    out = float(arr.ravel()[0])
    return out if np.isfinite(out) else None


def _max_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return float(np.max(arr))


def _min_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return float(np.min(arr))


def _list_or_none(value: Any) -> list[Any] | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return []
    if np.issubdtype(arr.dtype, np.integer):
        return [int(v) for v in arr.ravel()]
    return [float(v) for v in np.asarray(arr, dtype=float).ravel()]


def _mean_nonaxis_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    values = arr.ravel()[1:] if arr.size > 1 else arr.ravel()
    if values.size == 0:
        return None
    out = float(np.mean(values))
    return out if np.isfinite(out) else None


def _nfp_from_boozer_output(booz: dict[str, Any], nfp: int | None) -> int | None:
    if nfp is not None:
        return int(nfp)
    if "nfp_b" not in booz:
        return None
    arr = np.asarray(booz["nfp_b"])
    if arr.size == 0:
        return None
    return int(arr.ravel()[0])


def _infer_boozer_resolution(
    booz: dict[str, Any],
    *,
    nfp: int | None,
    options: QIDiagnosticOptions,
) -> tuple[int | None, int | None]:
    mboz = _maybe_int(options.mboz)
    nboz = _maybe_int(options.nboz)
    if mboz is None and "ixm_b" in booz:
        xm = np.asarray(booz["ixm_b"], dtype=float)
        if xm.size:
            mboz = int(np.max(np.abs(xm)))
    if nboz is None and "ixn_b" in booz:
        xn = np.asarray(booz["ixn_b"], dtype=float)
        if xn.size:
            nfp_abs = abs(int(nfp)) if nfp is not None and int(nfp) != 0 else None
            if nfp_abs is not None:
                normalized = xn / float(nfp_abs)
                if np.allclose(normalized, np.rint(normalized), rtol=0.0, atol=1.0e-12):
                    nboz = int(np.max(np.abs(np.rint(normalized))))
                else:
                    nboz = int(np.max(np.abs(xn)))
            else:
                nboz = int(np.max(np.abs(xn)))
    return mboz, nboz


def _base_record(
    *,
    source: str,
    options: QIDiagnosticOptions,
    nfp: int | None,
    mboz: int | None,
    nboz: int | None,
    surfaces: Any = None,
    surface_indices: Any = None,
) -> dict[str, Any]:
    return {
        "qi_diagnostic_version": QI_DIAGNOSTIC_VERSION,
        "qi_diagnostic_source": source,
        "qi_smooth_total": None,
        "qi_raw_total": None,
        "qi_legacy_total": None,
        "qi_mirror_ratio_max": None,
        "qi_mirror_ratio_target": float(options.mirror_threshold),
        "qi_mirror_excess_max": None,
        "qi_max_elongation": None,
        "qi_elongation_target": float(options.elongation_threshold),
        "qi_elongation_excess": None,
        "qi_lgradb_enabled": bool(options.include_lgradb),
        "qi_lgradb_min": None,
        "qi_lgradb_threshold": float(options.lgradb_threshold),
        "qi_lgradb_excess_max": None,
        "qi_phimin": float(options.phimin),
        "qi_nfp": None if nfp is None else int(nfp),
        "qi_mboz": mboz,
        "qi_nboz": nboz,
        "qi_boozer_resolution": {"mboz": mboz, "nboz": nboz},
        "qi_nphi": int(options.nphi),
        "qi_nalpha": int(options.nalpha),
        "qi_n_bounce": int(options.n_bounce),
        "qi_include_bounce_endpoints": bool(options.include_bounce_endpoints),
        "qi_shuffle_profile_nphi_out": None
        if options.shuffle_profile_nphi_out is None
        else int(options.shuffle_profile_nphi_out),
        "qi_legacy_nphi_out": None
        if options.legacy_nphi_out is None
        else int(options.legacy_nphi_out),
        "qi_mirror_ntheta": int(options.mirror_ntheta),
        "qi_mirror_nphi": int(options.mirror_nphi),
        "qi_mirror_surface_index": options.mirror_surface_index,
        "qi_elongation_ntheta": int(options.elongation_ntheta),
        "qi_elongation_nphi": int(options.elongation_nphi),
        "qi_lgradb_surface_index": int(options.lgradb_surface_index),
        "qi_lgradb_ntheta": int(options.lgradb_ntheta),
        "qi_lgradb_nphi": int(options.lgradb_nphi),
        "qi_surfaces": _list_or_none(surfaces if surfaces is not None else options.surfaces),
        "qi_surface_indices": _list_or_none(surface_indices),
        "aspect": None,
        "mean_iota": None,
    }


def _smooth_kwargs(options: QIDiagnosticOptions) -> dict[str, Any]:
    return {
        "nphi": int(options.nphi),
        "nalpha": int(options.nalpha),
        "n_bounce": int(options.n_bounce),
        "include_bounce_endpoints": bool(options.include_bounce_endpoints),
        "softness": float(options.softness),
        "width_weight": float(options.width_weight),
        "branch_width_weight": float(options.branch_width_weight),
        "branch_width_softness": float(options.branch_width_softness),
        "profile_weight": float(options.profile_weight),
        "shuffle_profile_weight": float(options.shuffle_profile_weight),
        "shuffle_profile_softness": float(options.shuffle_profile_softness),
        "shuffle_profile_nphi_out": options.shuffle_profile_nphi_out,
        "weighted_shuffle_profile_weight": float(options.weighted_shuffle_profile_weight),
        "weighted_shuffle_profile_softness": float(options.weighted_shuffle_profile_softness),
        "aligned_profile_weight": float(options.aligned_profile_weight),
        "aligned_profile_softness": float(options.aligned_profile_softness),
        "aligned_profile_trap_level": float(options.aligned_profile_trap_level),
        "aligned_profile_trap_softness": float(options.aligned_profile_trap_softness),
        "phimin": float(options.phimin),
    }


def _legacy_nphi_out(options: QIDiagnosticOptions) -> int:
    if options.legacy_nphi_out is not None:
        return int(options.legacy_nphi_out)
    return max(401, int(options.nphi))


def _handle_error(record: dict[str, Any], key: str, exc: Exception, *, fail_on_error: bool) -> None:
    if fail_on_error:
        raise exc
    record[key] = _format_error(exc)


def _surface_subset(booz: dict[str, Any], surface_index: int | None) -> dict[str, Any]:
    if surface_index is None:
        return booz
    bmnc = np.asarray(booz["bmnc_b"])
    if bmnc.ndim == 0:
        raise ValueError("bmnc_b must include a surface dimension")
    nsurf = int(bmnc.shape[0])
    index = int(surface_index)
    if index < 0:
        index += nsurf
    if index < 0 or index >= nsurf:
        raise ValueError(f"mirror_surface_index {surface_index} is outside the Boozer surface range 0..{nsurf - 1}")

    out: dict[str, Any] = {}
    for key, value in booz.items():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0 and int(shape[0]) == nsurf:
            out[key] = value[index : index + 1]
        else:
            out[key] = value
    return out


def _surface_subset_weights(
    weights: Iterable[float] | None,
    *,
    booz: dict[str, Any],
    surface_index: int | None,
) -> Iterable[float] | None:
    if weights is None or surface_index is None:
        return weights
    nsurf = int(np.asarray(booz["bmnc_b"]).shape[0])
    index = int(surface_index)
    if index < 0:
        index += nsurf
    weights_arr = np.asarray(list(weights), dtype=float)
    if weights_arr.shape[0] != nsurf:
        return [float(value) for value in weights_arr]
    return [float(weights_arr[index])]


def _record_smooth_total(
    record: dict[str, Any],
    smooth: dict[str, Any],
) -> None:
    total = _first_float(smooth.get("total"))
    record["qi_smooth_total"] = total
    # Backward-compatible alias for older sweep records.
    record["qi_raw_total"] = total


def _record_legacy(
    record: dict[str, Any],
    booz: dict[str, Any],
    *,
    options: QIDiagnosticOptions,
    nfp: int | None,
    weights: Iterable[float] | None,
) -> None:
    if not bool(options.include_legacy):
        return
    try:
        legacy = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
            booz,
            nfp=nfp,
            nphi=int(options.nphi),
            nalpha=int(options.nalpha),
            n_bounce=int(options.n_bounce),
            nphi_out=_legacy_nphi_out(options),
            phimin=float(options.phimin),
            weights=weights,
        )
        record["qi_legacy_total"] = float(legacy["total"])
        record["qi_legacy_residual_size"] = int(legacy.get("residual_size", 0))
        record["qi_legacy_nphi_out"] = _legacy_nphi_out(options)
    except Exception as exc:
        _handle_error(
            record,
            "qi_legacy_error",
            exc,
            fail_on_error=bool(options.fail_on_error),
        )


def _record_mirror(
    record: dict[str, Any],
    booz: dict[str, Any],
    *,
    options: QIDiagnosticOptions,
    nfp: int | None,
    weights: Iterable[float] | None,
) -> None:
    try:
        mirror_booz = _surface_subset(booz, options.mirror_surface_index)
        mirror_weights = _surface_subset_weights(
            weights,
            booz=booz,
            surface_index=options.mirror_surface_index,
        )
        mirror = mirror_ratio_penalty_from_boozer_output(
            mirror_booz,
            nfp=nfp,
            threshold=float(options.mirror_threshold),
            weights=mirror_weights,
            ntheta=int(options.mirror_ntheta),
            nphi=int(options.mirror_nphi),
            phimin=float(options.phimin),
        )
        mirror_max = _max_float(mirror.get("mirror_ratio"))
        record["qi_mirror_ratio_by_surface"] = _list_or_none(mirror.get("mirror_ratio"))
        record["qi_mirror_surface_index"] = options.mirror_surface_index
        record["qi_mirror_ratio_max"] = mirror_max
        record["qi_mirror_excess_max"] = None if mirror_max is None else max(
            0.0,
            mirror_max - float(options.mirror_threshold),
        )
    except Exception as exc:
        _handle_error(
            record,
            "qi_mirror_error",
            exc,
            fail_on_error=bool(options.fail_on_error),
        )


def _record_scalar_state_metrics(
    record: dict[str, Any],
    *,
    state: Any,
    static: Any,
    indata: Any,
    signgs: int,
    fail_on_error: bool,
) -> None:
    try:
        record["aspect"] = _first_float(equilibrium_aspect_ratio_from_state(state=state, static=static))
    except Exception as exc:
        _handle_error(record, "qi_aspect_error", exc, fail_on_error=fail_on_error)

    try:
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
        )
        record["mean_iota"] = _mean_nonaxis_float(iotas)
    except Exception as exc:
        _handle_error(record, "qi_iota_error", exc, fail_on_error=fail_on_error)


def _target_from_record(record: dict[str, Any], key: str, fallback: float | None) -> float | None:
    if fallback is None:
        return None
    value = _finite_float(record.get(key))
    return fallback if value is None else value


def _gate_excess(value: float | None, target: float | None, *, upper: bool) -> float | None:
    if target is None:
        return None
    if value is None:
        return None
    if upper:
        return max(0.0, value - target)
    return max(0.0, target - abs(value))


def _normalized_excess(excess: float | None, target: float | None) -> float | None:
    if excess is None:
        return None
    if target is None or target == 0.0:
        return float(excess)
    return float(excess / abs(target))


def _failure_message(name: str, value: float | None, target: float | None, *, upper: bool) -> str:
    if value is None:
        return f"{name} is unavailable"
    if target is None:
        return f"{name} gate is disabled"
    relation = "exceeds" if upper else "is below"
    return f"{name}={value:.6g} {relation} target {target:.6g}"


def annotate_qi_seed_suitability(
    record: dict[str, Any],
    *,
    targets: QISeedSuitabilityTargets | None = None,
) -> dict[str, Any]:
    """Return ``record`` with deterministic QI seed-ranking and gate fields.

    The ranking score intentionally combines the differentiable smooth QI
    residual with the legacy Goodman-style branch-shuffle diagnostic.  Missing
    core QI metrics are ranked last and are also reported in
    ``qi_gate_failures``/``qi_failure_reasons``.
    """

    targets = QISeedSuitabilityTargets() if targets is None else targets
    out = _with_result_aliases(record)

    smooth = _finite_float(out.get("qi_smooth_total"))
    legacy = _finite_float(out.get("qi_legacy_total"))
    aspect = _finite_float(out.get("aspect"))
    mean_iota = _finite_float(out.get("mean_iota"))
    mirror = _finite_float(out.get("qi_mirror_ratio_max"))
    elongation = _finite_float(out.get("qi_max_elongation"))

    smooth_target = targets.smooth_qi_max
    legacy_target = targets.legacy_qi_max
    aspect_target = _target_from_record(out, "target_aspect", targets.target_aspect)
    aspect_max = _target_from_record(out, "aspect_max", targets.aspect_max)
    iota_target = _target_from_record(out, "abs_iota_min", targets.abs_iota_min)
    mirror_target = _target_from_record(out, "qi_mirror_ratio_target", targets.mirror_ratio_max)
    elongation_target = _target_from_record(out, "qi_elongation_target", targets.max_elongation)

    aspect_relative_error = (
        None
        if aspect is None or aspect_target is None or aspect_target == 0.0
        else abs(aspect - aspect_target) / abs(aspect_target)
    )
    aspect_upper_excess = _gate_excess(aspect, aspect_max, upper=True)
    iota_shortfall = _gate_excess(mean_iota, iota_target, upper=False)
    smooth_excess = _gate_excess(smooth, smooth_target, upper=True)
    legacy_excess = _gate_excess(legacy, legacy_target, upper=True)
    mirror_excess = _gate_excess(mirror, mirror_target, upper=True)
    elongation_excess = _gate_excess(elongation, elongation_target, upper=True)

    failures: list[str] = []
    reasons: list[str] = []

    def require_gate(name: str, ok: bool, message: str) -> bool:
        if not ok:
            failures.append(name)
            reasons.append(message)
        return ok

    smooth_ok = True
    if smooth_target is not None:
        smooth_ok = require_gate(
            "smooth_qi",
            smooth is not None and smooth_excess == 0.0,
            _failure_message("smooth QI", smooth, smooth_target, upper=True),
        )
    legacy_ok = True
    if legacy_target is not None:
        legacy_ok = require_gate(
            "legacy_qi",
            legacy is not None and legacy_excess == 0.0,
            _failure_message("legacy QI", legacy, legacy_target, upper=True),
        )
    aspect_ok = True
    if aspect_max is not None:
        aspect_ok = require_gate(
            "aspect",
            aspect is not None and aspect_upper_excess == 0.0,
            _failure_message("aspect", aspect, aspect_max, upper=True),
        )
    elif aspect_target is not None:
        aspect_ok = require_gate(
            "aspect",
            aspect_relative_error is not None and aspect_relative_error <= targets.aspect_relative_tolerance,
            (
                "aspect is unavailable"
                if aspect_relative_error is None
                else (
                    f"aspect relative error={aspect_relative_error:.6g} exceeds "
                    f"tolerance {targets.aspect_relative_tolerance:.6g}"
                )
            ),
        )
    iota_ok = True
    if iota_target is not None:
        iota_ok = require_gate(
            "iota",
            mean_iota is not None and iota_shortfall == 0.0,
            _failure_message("abs(mean iota)", None if mean_iota is None else abs(mean_iota), iota_target, upper=False),
        )
    mirror_ok = True
    if mirror_target is not None:
        mirror_ok = require_gate(
            "mirror",
            mirror is not None and mirror_excess == 0.0,
            _failure_message("mirror ratio", mirror, mirror_target, upper=True),
        )
    elongation_ok = True
    if elongation_target is not None:
        elongation_ok = require_gate(
            "elongation",
            elongation is not None and elongation_excess == 0.0,
            _failure_message("max elongation", elongation, elongation_target, upper=True),
        )

    diagnostic_errors = sorted(key for key in out if key.startswith("qi_") and key.endswith("_error"))
    for key in diagnostic_errors:
        failures.append(key)
        reasons.append(f"{key}: {out[key]}")

    qi_core_complete = smooth is not None and legacy is not None
    qi_rank_score = float("inf") if not qi_core_complete else float(smooth + legacy)
    normalized_penalties = [
        0.0 if smooth_target is None else _normalized_excess(smooth_excess, smooth_target),
        0.0 if legacy_target is None else _normalized_excess(legacy_excess, legacy_target),
        0.0
        if aspect_target is None and aspect_max is None
        else _normalized_excess(aspect_upper_excess, aspect_max)
        if aspect_max is not None
        else None
        if aspect_relative_error is None
        else max(0.0, aspect_relative_error - targets.aspect_relative_tolerance),
        0.0 if iota_target is None else _normalized_excess(iota_shortfall, iota_target),
        0.0 if mirror_target is None else _normalized_excess(mirror_excess, mirror_target),
        0.0 if elongation_target is None else _normalized_excess(elongation_excess, elongation_target),
    ]
    constraint_penalties = [1.0 if value is None else float(value) for value in normalized_penalties]
    constraint_score = float(np.dot(constraint_penalties, constraint_penalties))

    out.update(
        {
            "target_aspect": aspect_target,
            "aspect_max": aspect_max,
            "abs_iota_min": iota_target,
            "qi_smooth_gate": smooth_target,
            "qi_legacy_gate": legacy_target,
            "qi_mirror_ratio_target": mirror_target,
            "qi_elongation_target": elongation_target,
            "abs_mean_iota": None if mean_iota is None else abs(mean_iota),
            "aspect_relative_error": None if aspect_relative_error is None else float(aspect_relative_error),
            "aspect_upper_excess": aspect_upper_excess,
            "iota_shortfall": iota_shortfall,
            "qi_smooth_excess": smooth_excess,
            "qi_legacy_excess": legacy_excess,
            "qi_mirror_excess_max": mirror_excess,
            "qi_elongation_excess": elongation_excess,
            "qi_diagnostic_errors": diagnostic_errors,
            "qi_gate_failures": failures,
            "failed_constraints": failures,
            "qi_failure_reasons": reasons,
            "qi_metric_gate_passed": bool(smooth_ok and legacy_ok and not diagnostic_errors),
            "qi_iota_gate_passed": bool(smooth_ok and legacy_ok and iota_ok and not diagnostic_errors),
            "qi_aspect_gate_passed": bool(aspect_ok and not diagnostic_errors),
            "qi_seed_gate_passed": bool(smooth_ok and legacy_ok and aspect_ok and iota_ok and not diagnostic_errors),
            "qi_mirror_gate_passed": bool(mirror_ok and not diagnostic_errors),
            "qi_engineering_gate_passed": bool(
                smooth_ok
                and legacy_ok
                and aspect_ok
                and iota_ok
                and mirror_ok
                and elongation_ok
                and not diagnostic_errors
            ),
            "qi_seed_suitability": "pass" if not failures else "needs_attention",
            "seed_suitability": "pass" if not failures else "needs_attention",
            "qi_rank_score": qi_rank_score,
            "qi_seed_score": qi_rank_score,
            "qi_constraint_score": constraint_score,
            "constraint_score": constraint_score,
        }
    )
    return out


def _with_result_aliases(record: dict[str, Any]) -> dict[str, Any]:
    """Return a QI record with common optimizer-summary aliases normalized."""

    out = dict(record)
    aliases = (
        ("aspect", "aspect_final"),
        ("mean_iota", "iota_final"),
        ("abs_iota_min", "iota_abs_min"),
        ("qi_smooth_total", "qi_raw_total"),
        ("qi_smooth_total", "smooth_total"),
        ("qi_legacy_total", "legacy_total"),
        ("qi_mirror_ratio_max", "mirror_ratio_max"),
        ("qi_mirror_ratio_target", "mirror_ratio_target"),
        ("qi_max_elongation", "max_elongation"),
        ("qi_elongation_target", "elongation_target"),
    )
    for canonical, alias in aliases:
        if (canonical not in out or _finite_float(out.get(canonical)) is None) and alias in out:
            out[canonical] = out[alias]
    return out


def qi_promotion_score(
    record: dict[str, Any],
    *,
    targets: QISeedSuitabilityTargets | None = None,
    require_legacy_source: bool = False,
    objective_key: str = "objective_final",
    wall_time_key: str = "total_wall_time_s",
) -> tuple[object, ...]:
    """Lexicographic score for promoting final QI optimization candidates.

    Seed audits intentionally rank QI-like starts before engineering cleanup so
    potentially useful seeds are not hidden.  Final promotion is stricter: a
    candidate that preserves QI while satisfying aspect/iota/mirror/elongation
    gates should beat a lower scalar objective that destroys an engineering
    gate.  Lower tuples are better.
    """

    row = _with_result_aliases(record)
    annotated = annotate_qi_seed_suitability(row, targets=targets)
    crashed = bool(row.get("crashed")) or row.get("success") is False
    legacy_source = str(row.get("qi_legacy_source", "legacy"))
    legacy_invalid = bool(require_legacy_source and legacy_source != "legacy")
    objective = _finite_float(row.get(objective_key))
    wall_time = _finite_float(row.get(wall_time_key))

    return (
        int(crashed),
        int(legacy_invalid),
        int(not annotated["qi_engineering_gate_passed"]),
        int(not annotated["qi_seed_gate_passed"]),
        int(not annotated["qi_metric_gate_passed"]),
        float(annotated.get("qi_rank_score", np.inf)),
        float(annotated.get("qi_constraint_score", np.inf)),
        np.inf if objective is None else float(objective),
        np.inf if wall_time is None else float(wall_time),
        str(row.get("label", row.get("case", row.get("output_dir", "")))),
    )


def qi_cleanup_candidate_promotable(
    candidate: dict[str, Any],
    *,
    reference: dict[str, Any] | None = None,
    targets: QISeedSuitabilityTargets | None = None,
    require_seed_gate: bool = True,
    require_engineering_gate: bool = False,
    require_mirror_improvement: bool = True,
    mirror_improvement_min: float = 0.0,
) -> dict[str, Any]:
    """Annotate whether a QI cleanup candidate should replace a reference.

    Mirror-ratio and elongation cleanup terms are engineering constraints, not
    definitions of QI.  This helper encodes the promotion rule used by the
    example optimizations: do not promote a candidate that improves mirror by
    destroying the smooth/legacy QI, aspect, or transform gates, and do not
    advance a mirror-ramp stage unless the mirror ratio actually decreases
    relative to the previously accepted state.
    """

    annotated = annotate_qi_seed_suitability(candidate, targets=targets)
    out = dict(annotated)
    reasons = list(out.get("qi_cleanup_rejection_reasons", []))

    if require_seed_gate and not bool(out.get("qi_seed_gate_passed")):
        seed_failure_names = {
            "smooth_qi",
            "legacy_qi",
            "aspect",
            "iota",
        }
        seed_failures = [
            str(item)
            for item in out.get("qi_gate_failures", ())
            if str(item) in seed_failure_names or (str(item).startswith("qi_") and str(item).endswith("_error"))
        ]
        failures = ", ".join(seed_failures) or "unknown"
        reasons.append(f"QI seed gate failed ({failures})")

    if require_engineering_gate and not bool(out.get("qi_engineering_gate_passed")):
        engineering_failures = ", ".join(str(item) for item in out.get("qi_gate_failures", ())) or "unknown"
        reasons.append(f"QI engineering gate failed ({engineering_failures})")

    candidate_mirror = _finite_float(out.get("qi_mirror_ratio_max"))
    reference_mirror: float | None = None
    if reference is not None:
        reference_mirror = _finite_float(reference.get("qi_mirror_ratio_max"))

    if require_mirror_improvement and reference is not None:
        if candidate_mirror is None:
            reasons.append("candidate mirror ratio is unavailable")
        elif reference_mirror is None:
            reasons.append("reference mirror ratio is unavailable")
        elif candidate_mirror > reference_mirror - float(mirror_improvement_min):
            reasons.append(
                "mirror ratio did not improve: "
                f"candidate={candidate_mirror:.6g}, reference={reference_mirror:.6g}"
            )

    out.update(
        {
            "qi_cleanup_candidate_mirror": candidate_mirror,
            "qi_cleanup_reference_mirror": reference_mirror,
            "qi_cleanup_promoted": not reasons,
            "qi_cleanup_rejection_reasons": reasons,
        }
    )
    return out


def rank_qi_seed_records(
    records: Iterable[dict[str, Any]],
    *,
    targets: QISeedSuitabilityTargets | None = None,
) -> list[dict[str, Any]]:
    """Annotate and rank QI seed records by smooth+legacy QI quality."""

    annotated = [annotate_qi_seed_suitability(record, targets=targets) for record in records]
    ranked = sorted(
        annotated,
        key=lambda row: (
            not np.isfinite(float(row.get("qi_rank_score", np.inf))),
            float(row.get("qi_rank_score", np.inf)),
            len(row.get("qi_gate_failures", [])),
            float(row.get("qi_constraint_score", np.inf)),
            str(row.get("label", row.get("case", ""))),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["qi_suitability_rank"] = index

    for key, rank_key in (
        ("qi_smooth_total", "qi_smooth_rank"),
        ("qi_legacy_total", "qi_legacy_rank"),
        ("qi_mirror_ratio_max", "qi_mirror_rank"),
        ("qi_constraint_score", "qi_constraint_rank"),
    ):
        finite = [row for row in ranked if _finite_float(row.get(key)) is not None]
        for index, row in enumerate(sorted(finite, key=lambda item: float(item[key])), start=1):
            row[rank_key] = index
    finite_iota = [row for row in ranked if _finite_float(row.get("abs_mean_iota")) is not None]
    for index, row in enumerate(sorted(finite_iota, key=lambda item: -float(item["abs_mean_iota"])), start=1):
        row["qi_iota_rank"] = index
    return ranked


def qi_diagnostics_from_boozer_output(
    booz: dict[str, Any],
    *,
    options: QIDiagnosticOptions | None = None,
    nfp: int | None = None,
    weights: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Evaluate QI diagnostics from an existing Boozer output dictionary.

    The returned record is flat and JSON/CSV-friendly.  State-only metrics
    (elongation and LgradB) are left as ``None`` by this Boozer-only entry
    point.
    """

    options = _as_options(options)
    nfp_local = _nfp_from_boozer_output(booz, nfp)
    mboz, nboz = _infer_boozer_resolution(booz, nfp=nfp_local, options=options)
    record = _base_record(
        source="boozer",
        options=options,
        nfp=nfp_local,
        mboz=mboz,
        nboz=nboz,
    )

    try:
        smooth = quasi_isodynamic_residual_from_boozer_output(
            booz,
            nfp=nfp_local,
            weights=weights,
            **_smooth_kwargs(options),
        )
        _record_smooth_total(record, smooth)
    except Exception as exc:
        _handle_error(
            record,
            "qi_smooth_error",
            exc,
            fail_on_error=bool(options.fail_on_error),
        )

    _record_mirror(record, booz, options=options, nfp=nfp_local, weights=weights)
    _record_legacy(record, booz, options=options, nfp=nfp_local, weights=weights)
    return record


def qi_diagnostics_from_state(
    *,
    state: Any,
    static: Any,
    indata: Any,
    signgs: int,
    surfaces: Any | None = None,
    options: QIDiagnosticOptions | None = None,
    weights: Iterable[float] | None = None,
    flux_local: Any = None,
    prof_local: Any = None,
    pressure_local: Any = None,
    jit_booz: bool | None = None,
    booz_constants: Any = None,
    booz_grids: Any = None,
    surface_indices: Any = None,
) -> dict[str, Any]:
    """Evaluate a complete QI diagnostic record from a solved VMEC state."""

    options = _as_options(options)
    jit_booz_local = bool(options.jit_booz if jit_booz is None else jit_booz)
    surfaces_local = surfaces if surfaces is not None else options.surfaces
    if surfaces_local is None:
        raise ValueError("surfaces must be supplied either as an argument or in QIDiagnosticOptions")
    nfp = int(static.cfg.nfp)
    mboz = _maybe_int(options.mboz) if options.mboz is not None else 12
    nboz = _maybe_int(options.nboz) if options.nboz is not None else 12
    record = _base_record(
        source="state",
        options=options,
        nfp=nfp,
        mboz=mboz,
        nboz=nboz,
        surfaces=surfaces_local,
        surface_indices=surface_indices,
    )
    record["qi_jit_booz"] = jit_booz_local

    booz = None
    try:
        smooth = quasi_isodynamic_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
            surfaces=surfaces_local,
            weights=weights,
            mboz=int(options.mboz if options.mboz is not None else 12),
            nboz=int(options.nboz if options.nboz is not None else 12),
            flux_local=flux_local,
            prof_local=prof_local,
            pressure_local=pressure_local,
            jit_booz=jit_booz_local,
            booz_constants=booz_constants,
            booz_grids=booz_grids,
            surface_indices=surface_indices,
            **_smooth_kwargs(options),
        )
        _record_smooth_total(record, smooth)
        booz = smooth.get("booz")
        record["qi_surfaces"] = _list_or_none(smooth.get("surfaces", surfaces_local))
        record["qi_surface_indices"] = _list_or_none(smooth.get("surface_indices", surface_indices))
    except Exception as exc:
        _handle_error(
            record,
            "qi_smooth_error",
            exc,
            fail_on_error=bool(options.fail_on_error),
        )

    if booz is not None:
        _record_mirror(record, booz, options=options, nfp=nfp, weights=weights)
        _record_legacy(record, booz, options=options, nfp=nfp, weights=weights)

    _record_scalar_state_metrics(
        record,
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        fail_on_error=bool(options.fail_on_error),
    )

    try:
        elongation = max_elongation_penalty_from_state(
            state=state,
            static=static,
            threshold=float(options.elongation_threshold),
            ntheta=int(options.elongation_ntheta),
            nphi=int(options.elongation_nphi),
        )
        elongation_max = _first_float(elongation.get("max_elongation"))
        record["qi_max_elongation"] = elongation_max
        record["qi_elongation_excess"] = None if elongation_max is None else max(
            0.0,
            elongation_max - float(options.elongation_threshold),
        )
    except Exception as exc:
        _handle_error(
            record,
            "qi_elongation_error",
            exc,
            fail_on_error=bool(options.fail_on_error),
        )

    if bool(options.include_lgradb):
        try:
            lgradb = lgradb_penalty_from_state(
                state=state,
                static=static,
                indata=indata,
                signgs=int(signgs),
                threshold=float(options.lgradb_threshold),
                s_index=int(options.lgradb_surface_index),
                ntheta=int(options.lgradb_ntheta),
                nphi=int(options.lgradb_nphi),
                smooth_penalty=float(options.lgradb_smooth_penalty),
                flux_local=flux_local,
            )
            record["qi_lgradb_min"] = _min_float(lgradb.get("L_grad_B"))
            record["qi_lgradb_excess_max"] = max(0.0, _max_float(lgradb.get("excess")) or 0.0)
        except Exception as exc:
            _handle_error(
                record,
                "qi_lgradb_error",
                exc,
                fail_on_error=bool(options.fail_on_error),
            )

    return record


__all__ = sorted(
    name
    for name, value in globals().items()
    if (
        getattr(value, "__module__", None) == __name__
        and not name.startswith("_")
    )
    or name == "QI_DIAGNOSTIC_VERSION"
)

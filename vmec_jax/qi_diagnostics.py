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

QI_DIAGNOSTIC_VERSION = "qi_diagnostics.v1"

__all__ = [
    "QI_DIAGNOSTIC_VERSION",
    "QIDiagnosticOptions",
    "qi_diagnostics_from_boozer_output",
    "qi_diagnostics_from_state",
]


@dataclass(frozen=True)
class QIDiagnosticOptions:
    """Resolution and threshold controls for QI diagnostic records."""

    surfaces: object | None = None
    mboz: int | None = None
    nboz: int | None = None
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
    }


def _smooth_kwargs(options: QIDiagnosticOptions) -> dict[str, Any]:
    return {
        "nphi": int(options.nphi),
        "nalpha": int(options.nalpha),
        "n_bounce": int(options.n_bounce),
        "softness": float(options.softness),
        "width_weight": float(options.width_weight),
        "branch_width_weight": float(options.branch_width_weight),
        "branch_width_softness": float(options.branch_width_softness),
        "profile_weight": float(options.profile_weight),
        "shuffle_profile_weight": float(options.shuffle_profile_weight),
        "shuffle_profile_softness": float(options.shuffle_profile_softness),
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
    jit_booz: bool = False,
    booz_constants: Any = None,
    booz_grids: Any = None,
    surface_indices: Any = None,
) -> dict[str, Any]:
    """Evaluate a complete QI diagnostic record from a solved VMEC state."""

    options = _as_options(options)
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
            jit_booz=bool(jit_booz),
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

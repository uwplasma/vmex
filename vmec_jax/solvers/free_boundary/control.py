"""Small free-boundary iteration-control helpers for VMEC solves."""

from __future__ import annotations

import os
from typing import Any, Callable, NamedTuple

import numpy as np

from ..fixed_boundary.residual.update import (
    scale_velocity_blocks,
    zero_velocity_blocks_like,
)


def free_boundary_iter_controls(iter2: int, iter1: int, nvacskip: int) -> tuple[int, int]:
    """Return the reduced legacy ``(ivac, ivacskip)`` free-boundary cadence."""

    nv = max(1, int(nvacskip))
    ivs = int((int(iter2) - int(iter1)) % nv)
    ivac = 1 if ivs == 0 else 2
    return ivac, ivs


def free_boundary_iter_controls_vmec(
    *,
    iter2: int,
    iter1: int,
    ivac: int,
    nvacskip: int,
    nvskip0: int,
    fsq_rz_prev: float,
    activate_fsq: float | None = None,
) -> tuple[int, int, int]:
    """VMEC2000-style ``ivac/ivacskip/nvacskip`` update for ``funct3d`` cadence."""

    i2 = int(iter2)
    i1 = int(iter1)
    iv = int(ivac)
    nv = max(1, int(nvacskip))
    nv0 = max(1, int(nvskip0))
    fs = float(fsq_rz_prev)
    if not np.isfinite(fs) or fs < 0.0:
        fs = 1.0

    if activate_fsq is None:
        activate_threshold = float(os.getenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "1.0e-3") or 1.0e-3)
    else:
        activate_threshold = float(activate_fsq)

    if i2 > 1 and fs <= activate_threshold:
        iv += 1

    if iv < 0:
        return iv, 0, nv

    ivs = int((i2 - i1) % nv)
    if iv <= 2:
        ivs = 0

    if ivs == 0:
        nv_est = int(1.0 / max(1.0e-1, 1.0e11 * fs))
        nv = max(nv0, max(1, nv_est))

    return iv, ivs, nv


def free_boundary_prev_rz_fsq_next(
    *,
    prev_fsq_before: float,
    fsq_rz_curr: float,
    turnon_restart: bool,
    preserve_turnon_restart: bool,
) -> float:
    """Optionally carry the pre-turn-on residual into the next cadence step."""

    if bool(turnon_restart) and bool(preserve_turnon_restart):
        return float(prev_fsq_before)
    return float(fsq_rz_curr)


def free_boundary_should_damp_constraint_baseline(
    *,
    freeb_ivac: int,
    freeb_turnon_iter: bool,
    lthreed: bool,
) -> bool:
    """Return whether VMEC should damp persistent free-boundary constraint baselines."""

    if not bool(lthreed):
        return int(freeb_ivac) >= 0
    return int(freeb_ivac) >= 0 and (not bool(freeb_turnon_iter))


def free_boundary_turnon_resets_iter1_immediately(*, lthreed: bool, lasym: bool) -> bool:
    """Return whether turn-on should immediately reset ``iter1`` for cadence."""

    return (not bool(lthreed)) or (not bool(lasym))


class FreeBoundaryNestorIterationResult(NamedTuple):
    """External-vacuum coupling state for one residual-loop iteration."""

    bsqvac_half_current: Any
    runtime: Any
    trace_arrays: Any
    reused: bool
    solve_time: float
    sample_time: float
    last_model: str
    last_diagnostics: dict[str, Any]
    ivac: int
    ivac_effective: int
    controls_cached: tuple[int, int, int] | None


def free_boundary_nestor_iteration_coupling(
    *,
    free_boundary_enabled: bool,
    freeb_couple_edge: bool,
    state: Any,
    static: Any,
    freeb_ivac: int,
    freeb_ivacskip: int,
    iter2: int,
    freeb_nestor_runtime: Any,
    freeb_plascur: float,
    external_field_provider_kind: str | None,
    external_field_provider_static: Any,
    external_field_provider_params: Any,
    collect_trace_arrays: bool,
    freeb_turnon_iter: bool,
    freeb_ivac_effective: int,
    freeb_nvacskip: int,
    controls_cached: tuple[int, int, int] | None,
    last_model: str,
    last_diagnostics: dict[str, Any],
    env_freeb_raise: bool,
    nestor_external_only_step_func: Callable[..., Any],
    edge_bsqvac_from_nestor_func: Callable[..., Any],
    source_reused_history: list[int],
    provider_allows_source_reuse_history: list[int],
    bnormal_rms_history: list[float],
    gsource_rms_history: list[float],
    bsqvac_rms_history: list[float],
) -> FreeBoundaryNestorIterationResult:
    """Run the VMEC-style free-boundary external-vacuum step when active."""

    bsqvac_half_current = None
    trace_arrays = None
    reused = False
    solve_time = 0.0
    sample_time = 0.0
    runtime_out = freeb_nestor_runtime
    model = str(last_model)
    diagnostics = dict(last_diagnostics)
    ivac = int(freeb_ivac)
    ivac_effective = int(freeb_ivac_effective)
    controls = controls_cached

    if not bool(free_boundary_enabled and freeb_couple_edge):
        return FreeBoundaryNestorIterationResult(
            bsqvac_half_current,
            freeb_nestor_runtime,
            trace_arrays,
            reused,
            solve_time,
            sample_time,
            model,
            diagnostics,
            ivac,
            ivac_effective,
            controls,
        )

    try:
        # VMEC enters NESTOR once control is active (`ivac >= 0`); vacuum.f
        # promotes ivac=0 -> 1 internally on the first turn-on.
        if ivac < 0:
            return FreeBoundaryNestorIterationResult(
                bsqvac_half_current,
                freeb_nestor_runtime,
                trace_arrays,
                reused,
                solve_time,
                sample_time,
                model,
                diagnostics,
                ivac,
                ivac_effective,
                controls,
            )
        nestor_res, runtime_out = nestor_external_only_step_func(
            state=state,
            static=static,
            ivac=ivac,
            ivacskip=int(freeb_ivacskip),
            iter_idx=int(iter2),
            runtime=freeb_nestor_runtime,
            extcur=tuple(getattr(static, "free_boundary_extcur", ()) or ()),
            plascur=float(freeb_plascur),
            external_field_provider_kind=external_field_provider_kind,
            external_field_provider_static=external_field_provider_static,
            external_field_provider_params=external_field_provider_params,
            collect_trace_arrays=bool(collect_trace_arrays),
        )
        model = str(getattr(nestor_res, "model", "spectral_poisson_external_only"))
        reused = bool(getattr(nestor_res, "reused", False))
        solve_time = float(getattr(nestor_res, "solve_time_s", 0.0))
        sample_time = float(getattr(nestor_res, "sample_time_s", 0.0))
        trace_arrays = getattr(nestor_res, "trace_arrays", None)
        diag_nestor = getattr(nestor_res, "diagnostics", None)
        if isinstance(diag_nestor, dict):
            diagnostics = dict(diag_nestor)
            source_reused_history.append(1 if bool(diag_nestor.get("source_reused", False)) else 0)
            provider_allows_source_reuse_history.append(
                1 if bool(diag_nestor.get("provider_allows_source_reuse", False)) else 0
            )
            for key, history in (
                ("bnormal_rms", bnormal_rms_history),
                ("gsource_rms", gsource_rms_history),
                ("bsqvac_rms", bsqvac_rms_history),
            ):
                try:
                    history.append(float(diag_nestor.get(key, float("nan"))))
                except Exception:
                    history.append(float("nan"))
        else:
            source_reused_history.append(0)
            provider_allows_source_reuse_history.append(0)
            bnormal_rms_history.append(float("nan"))
            gsource_rms_history.append(float("nan"))
            bsqvac_rms_history.append(float("nan"))
        bsqvac_half_current = edge_bsqvac_from_nestor_func(nestor_res, static)
        if freeb_turnon_iter:
            ivac = 1
            ivac_effective = 1
            controls = (ivac, int(freeb_ivacskip), int(freeb_nvacskip))
    except Exception:
        if env_freeb_raise:
            raise
        bsqvac_half_current = None
        trace_arrays = None
        reused = False
        solve_time = 0.0
        sample_time = 0.0

    return FreeBoundaryNestorIterationResult(
        bsqvac_half_current,
        runtime_out,
        trace_arrays,
        reused,
        solve_time,
        sample_time,
        model,
        diagnostics,
        ivac,
        ivac_effective,
        controls,
    )


_zero_velocity_blocks_like = zero_velocity_blocks_like
_scale_velocity_blocks = scale_velocity_blocks

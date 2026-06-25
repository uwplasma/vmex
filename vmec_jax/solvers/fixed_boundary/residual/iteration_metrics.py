"""Residual scalar metrics for the fixed-boundary iteration loop."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple


class PhysicalResidualMetricChannels(NamedTuple):
    """Physical residual channels sampled from current or cached norms."""

    norms_used: Any
    fsqr: Any
    fsqz: Any
    fsql: Any


def select_residual_norms_for_iteration(
    *,
    vmec2000_control: bool,
    precond_cache_valid: bool,
    need_bcovar_update: bool,
    cached_norms: Any,
    current_norms: Any,
) -> Any:
    """Select VMEC2000 cached norms when the ns4 preconditioner is reused."""

    if bool(vmec2000_control) and bool(precond_cache_valid) and (not bool(need_bcovar_update)):
        return cached_norms
    return current_norms


def physical_residual_metric_channels(
    *,
    gcr2: Any,
    gcz2: Any,
    gcl2: Any,
    norms_used: Any,
    host_update_assembly: bool,
    use_host_residual_metrics: bool,
    device_get_floats: Callable[..., tuple[float, ...]],
) -> PhysicalResidualMetricChannels:
    """Compute physical R, Z, and lambda residual channels.

    CPU host-update runs already have synchronized scalar channels, accelerator
    host-metric runs explicitly pull the required scalars once, and traced/device
    paths preserve array-valued residuals for AD.
    """

    if bool(host_update_assembly):
        gcr2_f = float(gcr2)
        gcz2_f = float(gcz2)
        gcl2_f = float(gcl2)
        fnorm_f = float(norms_used.fnorm)
        fnorm_l_f = float(norms_used.fnormL)
        r1_f = float(norms_used.r1)
        fsqr = r1_f * fnorm_f * gcr2_f
        fsqz = r1_f * fnorm_f * gcz2_f
        fsql = fnorm_l_f * gcl2_f
    elif bool(use_host_residual_metrics):
        (
            gcr2_f,
            gcz2_f,
            gcl2_f,
            fnorm_f,
            fnorm_l_f,
            r1_f,
        ) = device_get_floats(
            gcr2,
            gcz2,
            gcl2,
            norms_used.fnorm,
            norms_used.fnormL,
            norms_used.r1,
        )
        fsqr = r1_f * fnorm_f * gcr2_f
        fsqz = r1_f * fnorm_f * gcz2_f
        fsql = fnorm_l_f * gcl2_f
    else:
        fsqr = norms_used.r1 * norms_used.fnorm * gcr2
        fsqz = norms_used.r1 * norms_used.fnorm * gcz2
        fsql = norms_used.fnormL * gcl2

    return PhysicalResidualMetricChannels(
        norms_used=norms_used,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
    )

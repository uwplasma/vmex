"""Force-payload postprocessing seams for residual iteration."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from ...._compat import jnp
from .payload_blocks import (
    residual_force_payload_after_m1_scalxc,
    residual_force_payload_m1_scalxc_stages,
    zero_edge_rz_force_blocks,
)
from ....vmec_residue import vmec_gcx2_from_tomnsps
from ....vmec_tomnsp import TomnspsRZL

__all__ = [
    "ResidualForceMetricPayload",
    "ResidualForcePayloadResult",
    "force_z_channel_square_sums",
    "maybe_debug_force_z_channel_square_sums",
    "metric_force_payload_after_edge_policy",
    "residual_force_payload_from_kernels",
    "residual_force_payload_after_m1_scalxc_with_scan_debug",
    "residual_force_gcx2_after_edge_policy",
    "residual_force_z_nan_guard",
    "resolve_residual_force_mask_pack",
]


class ResidualForceMetricPayload(NamedTuple):
    """Metric payload and scalar VMEC force norms after edge policy."""

    frzl_metric: TomnspsRZL
    gcr2: Any
    gcz2: Any
    gcl2: Any


class ResidualForcePayloadResult(NamedTuple):
    """Raw/full residual force payloads and VMEC scalar metric payload."""

    include_edge_residual: bool
    mask_pack: Any | None
    frzl_raw: TomnspsRZL
    frzl_full: TomnspsRZL
    metric_payload: ResidualForceMetricPayload


def force_z_channel_square_sums(frzl: TomnspsRZL) -> tuple[Any, Any]:
    """Return squared sums of symmetric/asymmetric Z-force channels."""

    fzsc = jnp.asarray(frzl.fzsc)
    fzsc2 = jnp.sum(fzsc * fzsc)
    if frzl.fzcs is None:
        return fzsc2, jnp.asarray(0.0, dtype=fzsc.dtype)
    fzcs = jnp.asarray(frzl.fzcs)
    return fzsc2, jnp.sum(fzcs * fzcs)


def _debug_module_or_none() -> Any | None:
    try:
        from jax import debug as jax_debug  # type: ignore

        return jax_debug
    except Exception:
        return None


def maybe_debug_force_z_channel_square_sums(
    frzl: TomnspsRZL,
    *,
    enabled: bool,
    message: str,
    debug_module: Any | None = None,
) -> None:
    """Print Z-force channel square sums through ``jax.debug`` when enabled."""

    if not bool(enabled):
        return
    debug = _debug_module_or_none() if debug_module is None else debug_module
    if debug is None:
        return
    fzsc2, fzcs2 = force_z_channel_square_sums(frzl)
    debug.print(message, fzsc=fzsc2, fzcs=fzcs2)


def residual_force_payload_after_m1_scalxc_with_scan_debug(
    frzl: TomnspsRZL,
    *,
    s: Any,
    apply_m1_constraints: bool,
    lconm1: bool,
    zero_m1: Any,
    scan_debug_force_enabled: bool,
    debug_module: Any | None = None,
    stages_func: Callable[..., Any] = residual_force_payload_m1_scalxc_stages,
    final_func: Callable[..., TomnspsRZL] = residual_force_payload_after_m1_scalxc,
) -> TomnspsRZL:
    """Apply M1/zero/scalxc force-payload policy with optional scan diagnostics."""

    if not bool(scan_debug_force_enabled):
        return final_func(
            frzl,
            s=s,
            apply_m1_constraints=bool(apply_m1_constraints),
            lconm1=bool(lconm1),
            zero_m1=zero_m1,
        )

    force_stages = stages_func(
        frzl,
        s=s,
        apply_m1_constraints=bool(apply_m1_constraints),
        lconm1=bool(lconm1),
        zero_m1=zero_m1,
    )
    if bool(apply_m1_constraints):
        maybe_debug_force_z_channel_square_sums(
            force_stages.after_m1,
            enabled=True,
            message="[scan-debug-m1] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
            debug_module=debug_module,
        )
    maybe_debug_force_z_channel_square_sums(
        force_stages.after_zero_m1,
        enabled=True,
        message="[scan-debug-zero] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
        debug_module=debug_module,
    )
    maybe_debug_force_z_channel_square_sums(
        force_stages.after_scalxc,
        enabled=True,
        message="[scan-debug-scalxc] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
        debug_module=debug_module,
    )
    return force_stages.after_scalxc


def resolve_residual_force_mask_pack(
    static: Any,
    *,
    include_edge: bool,
    include_edge_residual: bool | None,
) -> tuple[bool, Any | None]:
    """Resolve residual-edge policy and the matching precomputed TOMNSP mask."""

    include_edge_residual_resolved = bool(include_edge if include_edge_residual is None else include_edge_residual)
    mask_pack = None
    if getattr(static, "tomnsps_masks", None) is not None:
        mask_pack = (
            getattr(static, "tomnsps_masks_edge")
            if bool(include_edge_residual_resolved)
            else getattr(static, "tomnsps_masks")
        )
    return include_edge_residual_resolved, mask_pack


def metric_force_payload_after_edge_policy(
    frzl: TomnspsRZL,
    *,
    include_edge: bool,
    zero_edge_rz_force_blocks_func: Callable[..., TomnspsRZL] = zero_edge_rz_force_blocks,
) -> TomnspsRZL:
    """Return the force payload used for R/Z metric scalars.

    The solver keeps the full residual payload for preconditioning and free-boundary
    parity, but optionally removes the LCFS contribution before forming physical
    R/Z force norms.  This helper isolates that policy from the iteration loop.
    """

    if bool(include_edge):
        return frzl
    return zero_edge_rz_force_blocks_func(frzl)


def residual_force_z_nan_guard(frzl: TomnspsRZL):
    """Return a scalar that preserves NaNs from Z-force channels.

    VMEC's scalar norm path can mask the edge contribution before reducing the
    arrays.  This guard keeps a NaN from the full Z-force payload visible in the
    final ``gcz2`` scalar while adding exactly zero for finite payloads.
    """

    z_force_dummy = jnp.sum(jnp.asarray(frzl.fzsc))
    if frzl.fzcs is not None:
        z_force_dummy = z_force_dummy + jnp.sum(jnp.asarray(frzl.fzcs))
    return jnp.where(
        jnp.isnan(z_force_dummy),
        z_force_dummy,
        jnp.asarray(0.0, dtype=jnp.asarray(z_force_dummy).dtype),
    )


def residual_force_gcx2_after_edge_policy(
    frzl: TomnspsRZL,
    *,
    include_edge: bool,
    lconm1: bool,
    s: Any,
    zero_edge_rz_force_blocks_func: Callable[..., TomnspsRZL] = zero_edge_rz_force_blocks,
    gcx2_func: Callable[..., tuple[Any, Any, Any]] = vmec_gcx2_from_tomnsps,
) -> ResidualForceMetricPayload:
    """Return force-norm scalars after solver edge masking and NaN policy."""

    frzl_metric = metric_force_payload_after_edge_policy(
        frzl,
        include_edge=bool(include_edge),
        zero_edge_rz_force_blocks_func=zero_edge_rz_force_blocks_func,
    )
    gcr2, gcz2, gcl2 = gcx2_func(
        frzl=frzl_metric,
        lconm1=bool(lconm1),
        apply_m1_constraints=False,
        include_edge=bool(include_edge),
        apply_scalxc=False,
        s=s,
    )
    return ResidualForceMetricPayload(
        frzl_metric=frzl_metric,
        gcr2=gcr2,
        gcz2=gcz2 + residual_force_z_nan_guard(frzl),
        gcl2=gcl2,
    )


def residual_force_payload_from_kernels(
    *,
    kernels: Any,
    static: Any,
    wout: Any,
    trig: Any,
    apply_lforbal: bool,
    include_edge: bool,
    include_edge_residual: bool | None,
    apply_m1_constraints: bool,
    lconm1: bool,
    zero_m1: Any,
    s: Any,
    scan_debug_force_enabled: bool,
    dump_hlo_force_tomnsps: bool = False,
    hlo_dump_func: Callable[..., None] | None = None,
    raw_tomnsps_callback: Callable[[TomnspsRZL], None] | None = None,
    gc_callback: Callable[[TomnspsRZL], None] | None = None,
    residual_func: Callable[..., TomnspsRZL] | None = None,
    postprocess_func: Callable[..., TomnspsRZL] = residual_force_payload_after_m1_scalxc_with_scan_debug,
    metric_func: Callable[..., ResidualForceMetricPayload] = residual_force_gcx2_after_edge_policy,
) -> ResidualForcePayloadResult:
    """Build residual force payloads from force kernels.

    This seam keeps the iteration loop focused on solver state updates while the
    TOMNSP force conventions stay in a separately tested module.  It preserves
    the VMEC ordering used in the original loop: resolve edge masks, assemble raw
    residuals, emit optional scan/HLO diagnostics, apply M1/zero/scalxc rules,
    then form metric force scalars with the selected edge policy.
    """

    include_edge_residual_resolved, mask_pack = resolve_residual_force_mask_pack(
        static,
        include_edge=bool(include_edge),
        include_edge_residual=include_edge_residual,
    )
    if residual_func is None:
        from ....vmec_forces import vmec_residual_internal_from_kernels as residual_func

    frzl_raw = residual_func(
        kernels,
        cfg_ntheta=int(static.cfg.ntheta),
        cfg_nzeta=int(static.cfg.nzeta),
        wout=wout,
        trig=trig,
        apply_lforbal=apply_lforbal,
        include_edge=bool(include_edge_residual_resolved),
        masks=mask_pack,
    )
    maybe_debug_force_z_channel_square_sums(
        frzl_raw,
        enabled=bool(scan_debug_force_enabled),
        message="[scan-debug-raw] fzsc2_raw={fzsc:.6e} fzcs2_raw={fzcs:.6e}",
    )
    if bool(dump_hlo_force_tomnsps) and hlo_dump_func is not None:
        try:

            def _tomnsps_only(k_in):
                frzl_hlo = residual_func(
                    k_in,
                    cfg_ntheta=int(static.cfg.ntheta),
                    cfg_nzeta=int(static.cfg.nzeta),
                    wout=wout,
                    trig=trig,
                    apply_lforbal=apply_lforbal,
                    include_edge=bool(include_edge_residual_resolved),
                    masks=mask_pack,
                )
                return (
                    frzl_hlo.frcc,
                    frzl_hlo.frss,
                    frzl_hlo.fzsc,
                    frzl_hlo.fzcs,
                    frzl_hlo.flsc,
                    frzl_hlo.flcs,
                )

            hlo_dump_func(label="tomnsps", fn=_tomnsps_only, args=(kernels,), kwargs={})
        except Exception:
            pass
    if raw_tomnsps_callback is not None:
        raw_tomnsps_callback(frzl_raw)

    frzl_full = postprocess_func(
        frzl_raw,
        s=s,
        apply_m1_constraints=bool(apply_m1_constraints),
        lconm1=bool(lconm1),
        zero_m1=zero_m1,
        scan_debug_force_enabled=bool(scan_debug_force_enabled),
    )
    if gc_callback is not None:
        gc_callback(frzl_full)

    metric_payload = metric_func(
        frzl_full,
        include_edge=bool(include_edge),
        lconm1=bool(lconm1),
        s=s,
    )
    return ResidualForcePayloadResult(
        include_edge_residual=include_edge_residual_resolved,
        mask_pack=mask_pack,
        frzl_raw=frzl_raw,
        frzl_full=frzl_full,
        metric_payload=metric_payload,
    )

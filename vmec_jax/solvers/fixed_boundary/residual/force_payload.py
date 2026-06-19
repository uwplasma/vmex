"""Force-payload postprocessing seams for residual iteration."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

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
    "ResidualForceEvaluationResult",
    "evaluate_residual_force_from_state",
    "force_z_channel_square_sums",
    "maybe_debug_force_z_channel_square_sums",
    "metric_force_payload_after_edge_policy",
    "residual_force_payload_from_kernels",
    "residual_force_payload_after_m1_scalxc_with_scan_debug",
    "residual_force_gcx2_after_edge_policy",
    "residual_force_z_nan_guard",
    "resolve_residual_force_mask_pack",
    "make_residual_force_evaluator",
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


class ResidualForceEvaluationResult(NamedTuple):
    """Force kernels, transformed residual blocks, norms, and preconditioner scales."""

    kernels: Any
    frzl_full: TomnspsRZL
    gcr2: Any
    gcz2: Any
    gcl2: Any
    rz_scale: Any
    l_scale: Any
    norms: Any


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


def evaluate_residual_force_from_state(
    *,
    state: Any,
    static: Any,
    wout_like: Any,
    trig: Any,
    s: Any,
    signgs: int,
    constraint_tcon0: Any,
    freeb_pres_scale: Any,
    apply_lforbal: bool,
    apply_m1_constraints: bool,
    include_edge: bool,
    include_edge_residual: bool | None,
    zero_m1: Any,
    freeb_bsqvac_half: Any | None = None,
    constraint_rcon0: Any | None = None,
    constraint_zcon0: Any | None = None,
    constraint_precond_diag: tuple[Any, Any] | None = None,
    constraint_tcon: Any | None = None,
    constraint_precond_active: Any | None = None,
    constraint_tcon_active: Any | None = None,
    iter_idx: int | None = None,
    scan_debug_force_enabled: bool = False,
    dump_hlo_force_tomnsps: bool = False,
    hlo_dump_func: Callable[..., None] | None = None,
    dump_hooks: Mapping[str, Callable[..., None]] | None = None,
    kernels_func: Callable[..., Any] | None = None,
    force_payload_func: Callable[..., ResidualForcePayloadResult] = residual_force_payload_from_kernels,
    norms_func: Callable[..., Any] | None = None,
    scale_func: Callable[..., tuple[Any, Any]] | None = None,
) -> ResidualForceEvaluationResult:
    """Evaluate one residual force payload and associated solver scalars.

    The residual controller needs one compact operation: ``state -> force
    blocks + norms + scale factors``.  This helper owns the surrounding VMEC
    force diagnostics so the iteration loop can focus on timestep/controller
    policy instead of every optional dump hook.
    """

    if kernels_func is None:
        from ....vmec_forces import vmec_forces_rz_from_wout as kernels_func

    if norms_func is None:
        from ....vmec_residue import vmec_force_norms_from_bcovar_dynamic as norms_func

    if scale_func is None:
        from ..preconditioning.operators import metric_surface_precond_from_bcovar_jax as scale_func

    hooks = {} if dump_hooks is None else dict(dump_hooks)
    kernels = kernels_func(
        state=state,
        static=static,
        wout=wout_like,
        indata=None,
        constraint_tcon0=constraint_tcon0,
        constraint_tcon=constraint_tcon,
        constraint_precond_diag=constraint_precond_diag,
        constraint_precond_active=constraint_precond_active,
        constraint_tcon_active=constraint_tcon_active,
        constraint_rcon0=constraint_rcon0,
        constraint_zcon0=constraint_zcon0,
        freeb_bsqvac_half=freeb_bsqvac_half,
        freeb_pres_scale=freeb_pres_scale,
        use_vmec_synthesis=True,
        trig=trig,
        iter_idx=iter_idx,
    )

    if iter_idx is not None:
        hook_kwargs = {"bc": kernels.bc, "static": static, "iter_idx": int(iter_idx)}
        if "bsube" in hooks:
            hooks["bsube"](**hook_kwargs)
        if "bsube_terms" in hooks:
            hooks["bsube_terms"](**hook_kwargs)
        if "bsubh" in hooks:
            hooks["bsubh"](**hook_kwargs)
        if "bsubs" in hooks:
            hooks["bsubs"](bc=kernels.bc, state=state, static=static, trig=trig, iter_idx=int(iter_idx), kernels=kernels)
        if "lulv" in hooks:
            hooks["lulv"](bc=kernels.bc, static=static, iter_idx=int(iter_idx), state=state, trig=trig)
        if "jacobian_terms" in hooks:
            hooks["jacobian_terms"](k=kernels, iter_idx=int(iter_idx))
        if "precond_inputs" in hooks:
            hooks["precond_inputs"](bc=kernels.bc, trig=trig, static=static, iter_idx=int(iter_idx), kernels=kernels)
        if "gmetric" in hooks:
            hooks["gmetric"](**hook_kwargs)
        if "force_kernels" in hooks:
            hooks["force_kernels"](k=kernels, static=static, iter_idx=int(iter_idx), label="raw")

    def _dump_force_tomnsps_hlo(*, label, fn, args, kwargs):
        if hlo_dump_func is not None:
            hlo_dump_func(label=label, fn=fn, args=args, kwargs=kwargs)

    raw_callback = (
        (lambda frzl: hooks["tomnsps"](frzl=frzl, static=static, iter_idx=int(iter_idx), label="raw"))
        if iter_idx is not None and "tomnsps" in hooks
        else None
    )
    gc_callback = (
        (lambda frzl: hooks["gc"](frzl=frzl, static=static, iter_idx=int(iter_idx), label="raw"))
        if iter_idx is not None and "gc" in hooks
        else None
    )
    force_payload = force_payload_func(
        kernels=kernels,
        static=static,
        wout=wout_like,
        trig=trig,
        apply_lforbal=apply_lforbal,
        include_edge=bool(include_edge),
        include_edge_residual=include_edge_residual,
        apply_m1_constraints=bool(apply_m1_constraints),
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        zero_m1=zero_m1,
        s=s,
        scan_debug_force_enabled=bool(scan_debug_force_enabled),
        dump_hlo_force_tomnsps=bool(dump_hlo_force_tomnsps),
        hlo_dump_func=_dump_force_tomnsps_hlo,
        raw_tomnsps_callback=raw_callback,
        gc_callback=gc_callback,
    )
    frzl_full = force_payload.frzl_full
    metric_payload = force_payload.metric_payload
    gcr2, gcz2, gcl2 = metric_payload.gcr2, metric_payload.gcz2, metric_payload.gcl2
    if iter_idx is not None and "gcx2" in hooks:
        hooks["gcx2"](
            gcr2=gcr2,
            gcz2=gcz2,
            gcl2=gcl2,
            iter_idx=int(iter_idx),
            include_edge=bool(include_edge),
            ns=int(static.cfg.ns),
        )
    norms = norms_func(bc=kernels.bc, trig=trig, s=s, signgs=signgs)
    if iter_idx is not None and "scalars" in hooks:
        hooks["scalars"](norms=norms, iter_idx=int(iter_idx), ns=int(static.cfg.ns))
    rz_scale, l_scale = scale_func(bc=kernels.bc, trig=trig)
    return ResidualForceEvaluationResult(
        kernels=kernels,
        frzl_full=frzl_full,
        gcr2=gcr2,
        gcz2=gcz2,
        gcl2=gcl2,
        rz_scale=rz_scale,
        l_scale=l_scale,
        norms=norms,
    )


def make_residual_force_evaluator(
    *,
    static: Any,
    wout_like: Any,
    trig: Any,
    s: Any,
    signgs: int,
    constraint_tcon0: Any,
    freeb_pres_scale: Any,
    apply_lforbal: bool,
    apply_m1_constraints: bool,
    runtime_env_enabled: Callable[[str], bool],
    getenv: Callable[[str, str], str],
    maybe_dump_hlo_kernel: Callable[..., None],
    dump_hooks: Mapping[str, Callable[..., None]],
    evaluate_force_func: Callable[..., ResidualForceEvaluationResult] = evaluate_residual_force_from_state,
) -> Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any, Any]]:
    """Build the solver's compact ``state -> force tuple`` evaluator.

    The residual iteration loop historically carried this wrapper as a nested
    function because it needs access to many diagnostic hooks.  Keeping the
    wrapper here makes the force path reusable while the caller still owns the
    concrete dump hooks and HLO policy.
    """

    def _compute_forces(
        state,
        *,
        include_edge: bool,
        include_edge_residual: bool | None = None,
        zero_m1: Any,
        freeb_bsqvac_half: Any | None = None,
        constraint_rcon0: Any | None = None,
        constraint_zcon0: Any | None = None,
        constraint_precond_diag: tuple[Any, Any] | None = None,
        constraint_tcon: Any | None = None,
        constraint_precond_active: Any | None = None,
        constraint_tcon_active: Any | None = None,
        iter_idx: int | None = None,
    ):
        scan_debug_force_enabled = getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0")
        dump_hlo_force_tomnsps = runtime_env_enabled(getenv("VMEC_JAX_DUMP_HLO_FORCE_TOMNSPS", ""))

        def _dump_force_tomnsps_hlo(*, label, fn, args, kwargs) -> None:
            maybe_dump_hlo_kernel(
                label=label,
                fn=fn,
                args=args,
                kwargs=kwargs,
                static=static,
                wout_like=wout_like,
                force=True,
            )

        force_eval = evaluate_force_func(
            state=state,
            static=static,
            wout_like=wout_like,
            trig=trig,
            s=s,
            signgs=signgs,
            constraint_tcon0=constraint_tcon0,
            freeb_pres_scale=freeb_pres_scale,
            apply_lforbal=apply_lforbal,
            apply_m1_constraints=bool(apply_m1_constraints),
            include_edge=bool(include_edge),
            include_edge_residual=include_edge_residual,
            zero_m1=zero_m1,
            freeb_bsqvac_half=freeb_bsqvac_half,
            constraint_rcon0=constraint_rcon0,
            constraint_zcon0=constraint_zcon0,
            constraint_precond_diag=constraint_precond_diag,
            constraint_tcon=constraint_tcon,
            constraint_precond_active=constraint_precond_active,
            constraint_tcon_active=constraint_tcon_active,
            iter_idx=iter_idx,
            scan_debug_force_enabled=bool(scan_debug_force_enabled),
            dump_hlo_force_tomnsps=bool(dump_hlo_force_tomnsps),
            hlo_dump_func=_dump_force_tomnsps_hlo,
            dump_hooks=dump_hooks,
        )
        return (
            force_eval.kernels,
            force_eval.frzl_full,
            force_eval.gcr2,
            force_eval.gcz2,
            force_eval.gcl2,
            force_eval.rz_scale,
            force_eval.l_scale,
            force_eval.norms,
        )

    return _compute_forces

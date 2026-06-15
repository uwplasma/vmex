"""Pure helpers for VMEC2000 scan force/restart payloads."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from ...._compat import jax, jnp
from ....vmec_residue import vmec_gcx2_from_tomnsps
from ....vmec_tomnsp import TomnspsRZL


class ScanForceBlocks(NamedTuple):
    frcc: Any
    frss: Any
    fzsc: Any
    fzcs: Any
    flsc: Any
    flcs: Any
    frsc: Any
    frcs: Any
    fzcc: Any
    fzss: Any
    flcc: Any
    flss: Any


class ScanForcePayload(NamedTuple):
    blocks: ScanForceBlocks
    fsqr: Any
    fsqz: Any
    fsql: Any
    fsqr1: Any
    fsqz1: Any
    fsql1: Any
    cache_precond_diag: Any
    cache_tcon: Any
    cache_norms: Any
    cache_rz_scale: Any
    cache_l_scale: Any
    cache_rz_norm: Any
    cache_f_norm1: Any
    cache_rz_mats: Any
    cache_lam_prec: Any
    cache_valid: Any


class ScanStepFields(NamedTuple):
    state: Any
    vRcc: Any
    vRss: Any
    vZsc: Any
    vZcs: Any
    vLsc: Any
    vLcs: Any
    vRsc: Any
    vRcs: Any
    vZcc: Any
    vZss: Any
    vLcc: Any
    vLss: Any
    inv_tau: Any
    fsq_prev: Any


def mask_scan_restart_force_payload(
    *, force_blocks: tuple[Any, ...], cache_valid: Any, do_restart: Any
) -> tuple[tuple[Any, ...], Any]:
    """Zero current-state scan forces on restart when checkpoint forces are skipped."""

    no_restart = jnp.logical_not(do_restart)
    masked_blocks = tuple(jnp.where(no_restart, block, jnp.zeros_like(block)) for block in force_blocks)
    cache_valid_masked = jnp.where(no_restart, cache_valid, jnp.asarray(False))
    return masked_blocks, cache_valid_masked


def _preconditioned_blocks(*, frzl_rz: TomnspsRZL, cache_lam_prec: Any) -> tuple[ScanForceBlocks, TomnspsRZL]:
    lam_prec = jnp.asarray(cache_lam_prec)
    frcc = jnp.asarray(frzl_rz.frcc)
    frss = frzl_rz.frss if frzl_rz.frss is not None else jnp.zeros_like(frcc)
    fzsc = jnp.asarray(frzl_rz.fzsc)
    fzcs = frzl_rz.fzcs if frzl_rz.fzcs is not None else jnp.zeros_like(fzsc)
    flsc = jnp.asarray(frzl_rz.flsc) * lam_prec
    flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * lam_prec)
    frsc = jnp.zeros_like(frcc)
    frcs = jnp.zeros_like(frcc)
    fzcc = jnp.zeros_like(fzsc)
    fzss = jnp.zeros_like(fzsc)
    flcc = jnp.zeros_like(flsc)
    flss = jnp.zeros_like(flsc)

    if getattr(frzl_rz, "frsc", None) is not None:
        frsc = jnp.asarray(frzl_rz.frsc)
    if getattr(frzl_rz, "frcs", None) is not None:
        frcs = jnp.asarray(frzl_rz.frcs)
    if getattr(frzl_rz, "fzcc", None) is not None:
        fzcc = jnp.asarray(frzl_rz.fzcc)
    if getattr(frzl_rz, "fzss", None) is not None:
        fzss = jnp.asarray(frzl_rz.fzss)
    if getattr(frzl_rz, "flcc", None) is not None:
        flcc = jnp.asarray(frzl_rz.flcc) * lam_prec
    if getattr(frzl_rz, "flss", None) is not None:
        flss = jnp.asarray(frzl_rz.flss) * lam_prec

    blocks = ScanForceBlocks(
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=flsc,
        flcs=flcs if flcs is not None else jnp.zeros_like(flsc),
        frsc=frsc,
        frcs=frcs,
        fzcc=fzcc,
        fzss=fzss,
        flcc=flcc,
        flss=flss,
    )
    frzl_pre = TomnspsRZL(
        frcc=blocks.frcc,
        frss=blocks.frss,
        fzsc=blocks.fzsc,
        fzcs=blocks.fzcs,
        flsc=blocks.flsc,
        flcs=flcs,
        frsc=blocks.frsc,
        frcs=blocks.frcs,
        fzcc=blocks.fzcc,
        fzss=blocks.fzss,
        flcc=blocks.flcc,
        flss=blocks.flss,
    )
    return blocks, frzl_pre


def _lambda_fsq1_from_blocks(
    *, frzl_pre: TomnspsRZL, delta_s: Any, optional_source: TomnspsRZL | None = None
) -> Any:
    gcl2_full = jnp.sum(jnp.asarray(frzl_pre.flsc)[1:] * jnp.asarray(frzl_pre.flsc)[1:])
    if frzl_pre.flcs is not None:
        flcs = jnp.asarray(frzl_pre.flcs)
        gcl2_full = gcl2_full + jnp.sum(flcs[1:] * flcs[1:])
    optional_blocks = optional_source if optional_source is not None else frzl_pre
    if getattr(optional_blocks, "flcc", None) is not None:
        flcc = jnp.asarray(optional_blocks.flcc)
        gcl2_full = gcl2_full + jnp.sum(flcc[1:] * flcc[1:])
    if getattr(optional_blocks, "flss", None) is not None:
        flss = jnp.asarray(optional_blocks.flss)
        gcl2_full = gcl2_full + jnp.sum(flss[1:] * flss[1:])
    return gcl2_full * delta_s


def _weighted_blocks(
    *,
    blocks: ScanForceBlocks,
    w_mode_mn: Any,
    lambda_update_scale_j: Any,
    apply_lambda_update_scale: bool,
) -> ScanForceBlocks:
    weights = jnp.asarray(w_mode_mn)[None, :, :]
    flsc = blocks.flsc * weights
    flcs = blocks.flcs * weights
    flcc = blocks.flcc * weights
    flss = blocks.flss * weights
    if bool(apply_lambda_update_scale):
        scale = jnp.asarray(lambda_update_scale_j)
        flsc = flsc * scale
        flcs = flcs * scale
        flcc = flcc * scale
        flss = flss * scale

    return ScanForceBlocks(
        frcc=blocks.frcc * weights,
        frss=blocks.frss * weights,
        fzsc=blocks.fzsc * weights,
        fzcs=blocks.fzcs * weights,
        flsc=flsc,
        flcs=flcs,
        frsc=blocks.frsc * weights,
        frcs=blocks.frcs * weights,
        fzcc=blocks.fzcc * weights,
        fzss=blocks.fzss * weights,
        flcc=flcc,
        flss=flss,
    )


def build_scan_force_payload(
    *,
    frzl_rz: TomnspsRZL,
    cache_lam_prec: Any,
    w_mode_mn: Any,
    lambda_update_scale_j: Any,
    apply_lambda_update_scale: bool,
    fsqr: Any,
    fsqz: Any,
    fsql: Any,
    f_norm1: Any,
    delta_s: Any,
    s: Any,
    lconm1: bool,
    cache_precond_diag: Any,
    cache_tcon: Any,
    cache_norms: Any,
    cache_rz_scale: Any,
    cache_l_scale: Any,
    cache_rz_norm: Any,
    cache_f_norm1: Any,
    cache_rz_mats: Any,
    cache_valid: Any,
    lambda_fsq1_optional_source: TomnspsRZL | None = None,
) -> ScanForcePayload:
    """Build the scan payload from preconditioned force blocks and cache fields."""

    pre_blocks, frzl_pre = _preconditioned_blocks(frzl_rz=frzl_rz, cache_lam_prec=cache_lam_prec)
    gcr2_p, gcz2_p, _gcl2_p = vmec_gcx2_from_tomnsps(
        frzl=frzl_pre,
        lconm1=bool(lconm1),
        apply_m1_constraints=False,
        include_edge=True,
        apply_scalxc=False,
        s=s,
    )
    fsqr1 = gcr2_p * f_norm1
    fsqz1 = gcz2_p * f_norm1
    fsql1 = _lambda_fsq1_from_blocks(
        frzl_pre=frzl_pre,
        delta_s=delta_s,
        optional_source=lambda_fsq1_optional_source,
    )
    weighted = _weighted_blocks(
        blocks=pre_blocks,
        w_mode_mn=w_mode_mn,
        lambda_update_scale_j=lambda_update_scale_j,
        apply_lambda_update_scale=apply_lambda_update_scale,
    )

    return ScanForcePayload(
        blocks=weighted,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        fsqr1=fsqr1,
        fsqz1=fsqz1,
        fsql1=fsql1,
        cache_precond_diag=cache_precond_diag,
        cache_tcon=cache_tcon,
        cache_norms=cache_norms,
        cache_rz_scale=cache_rz_scale,
        cache_l_scale=cache_l_scale,
        cache_rz_norm=cache_rz_norm,
        cache_f_norm1=cache_f_norm1,
        cache_rz_mats=cache_rz_mats,
        cache_lam_prec=cache_lam_prec,
        cache_valid=cache_valid,
    )


def current_scan_payload(**kwargs: Any) -> ScanForcePayload:
    return build_scan_force_payload(**kwargs)


def restart_scan_payload(**kwargs: Any) -> ScanForcePayload:
    return build_scan_force_payload(**kwargs)


def mask_scan_restart_payload(*, payload: ScanForcePayload, do_restart: Any) -> ScanForcePayload:
    masked_blocks, cache_valid = mask_scan_restart_force_payload(
        force_blocks=tuple(payload.blocks),
        cache_valid=payload.cache_valid,
        do_restart=do_restart,
    )
    return payload._replace(blocks=ScanForceBlocks(*masked_blocks), cache_valid=cache_valid)


def select_scan_force_payload(
    *,
    do_restart: Any,
    use_restart_payload: bool,
    restart_payload_fn: Callable[[Any], ScanForcePayload],
    current_payload_fn: Callable[[Any], ScanForcePayload],
    cond: Callable[..., ScanForcePayload] | None = None,
) -> ScanForcePayload:
    """Select restart/current payloads while preserving the no-restart fast path."""

    if bool(use_restart_payload):
        cond_fn = cond if cond is not None else jax.lax.cond
        return cond_fn(do_restart, restart_payload_fn, current_payload_fn, operand=None)
    return mask_scan_restart_payload(payload=current_payload_fn(None), do_restart=do_restart)


def select_scan_step_fields(
    *,
    vmec2000_control: bool,
    do_restart: Any,
    accept_step_fn: Callable[[Any], ScanStepFields],
    reject_step_fn: Callable[[Any], ScanStepFields],
    cond: Callable[..., ScanStepFields] | None = None,
) -> ScanStepFields:
    """Select accepted/rejected scan step fields with VMEC2000 retry semantics."""

    if bool(vmec2000_control):
        return accept_step_fn(None)
    cond_fn = cond if cond is not None else jax.lax.cond
    return cond_fn(do_restart, reject_step_fn, accept_step_fn, operand=None)


_current_payload = current_scan_payload
_restart_payload = restart_scan_payload

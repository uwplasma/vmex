"""Small helpers for residual-iteration force payload assembly."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np

from ...._compat import jnp
from ....vmec_residue import (
    vmec_apply_m1_constraints,
    vmec_apply_scalxc_to_tomnsps,
    vmec_zero_m1_zforce,
)
from ....vmec_tomnsp import TomnspsRZL


class ForceBlocks(NamedTuple):
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


class ResidualForcePayloadStages(NamedTuple):
    """Intermediate residual-force payloads through VMEC force conventions."""

    after_m1: TomnspsRZL
    after_zero_m1: TomnspsRZL
    after_scalxc: TomnspsRZL


def zero_edge_rz_force_block(a, *, preserve_numpy: bool = True):
    """Zero the LCFS row in an R/Z force block, leaving short meshes unchanged."""
    if a is None:
        return None
    if preserve_numpy and isinstance(a, np.ndarray):
        if a.shape[0] < 2:
            return a
        out = a.copy()
        out[-1] = np.zeros_like(a[-1])
        return out
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    return a.at[-1].set(jnp.zeros_like(a[-1]))


def zero_edge_rz_force_blocks(frzl: TomnspsRZL, *, preserve_numpy: bool = True) -> TomnspsRZL:
    """Zero LCFS rows for every R/Z block in a ``TomnspsRZL`` container."""
    return TomnspsRZL(
        frcc=zero_edge_rz_force_block(frzl.frcc, preserve_numpy=preserve_numpy),
        frss=zero_edge_rz_force_block(frzl.frss, preserve_numpy=preserve_numpy),
        fzsc=zero_edge_rz_force_block(frzl.fzsc, preserve_numpy=preserve_numpy),
        fzcs=zero_edge_rz_force_block(frzl.fzcs, preserve_numpy=preserve_numpy),
        flsc=frzl.flsc,
        flcs=frzl.flcs,
        frsc=zero_edge_rz_force_block(getattr(frzl, "frsc", None), preserve_numpy=preserve_numpy),
        frcs=zero_edge_rz_force_block(getattr(frzl, "frcs", None), preserve_numpy=preserve_numpy),
        fzcc=zero_edge_rz_force_block(getattr(frzl, "fzcc", None), preserve_numpy=preserve_numpy),
        fzss=zero_edge_rz_force_block(getattr(frzl, "fzss", None), preserve_numpy=preserve_numpy),
        flcc=getattr(frzl, "flcc", None),
        flss=getattr(frzl, "flss", None),
    )


def normalize_force_blocks(frzl: TomnspsRZL) -> TomnspsRZL:
    """Materialize force blocks after scalar transforms without changing values."""

    def _normalize_block(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        x = jnp.asarray(x)
        return jnp.where(jnp.isnan(x), x, x)

    values = {
        "frcc": _normalize_block(frzl.frcc),
        "frss": _normalize_block(frzl.frss),
        "fzsc": _normalize_block(frzl.fzsc),
        "fzcs": _normalize_block(frzl.fzcs),
        "flsc": _normalize_block(frzl.flsc),
        "flcs": _normalize_block(frzl.flcs),
        "frsc": _normalize_block(getattr(frzl, "frsc", None)),
        "frcs": _normalize_block(getattr(frzl, "frcs", None)),
        "fzcc": _normalize_block(getattr(frzl, "fzcc", None)),
        "fzss": _normalize_block(getattr(frzl, "fzss", None)),
        "flcc": _normalize_block(getattr(frzl, "flcc", None)),
        "flss": _normalize_block(getattr(frzl, "flss", None)),
    }
    try:
        return replace(frzl, **values)
    except Exception:
        return TomnspsRZL(**values)


def residual_force_payload_after_m1_scalxc(
    frzl: TomnspsRZL,
    *,
    s,
    apply_m1_constraints: bool,
    lconm1: bool,
    zero_m1,
) -> TomnspsRZL:
    """Apply residual force m=1, zeroing, and scalxc conventions."""
    return residual_force_payload_m1_scalxc_stages(
        frzl,
        s=s,
        apply_m1_constraints=apply_m1_constraints,
        lconm1=lconm1,
        zero_m1=zero_m1,
    ).after_scalxc


def residual_force_payload_m1_scalxc_stages(
    frzl: TomnspsRZL,
    *,
    s,
    apply_m1_constraints: bool,
    lconm1: bool,
    zero_m1,
) -> ResidualForcePayloadStages:
    """Return residual force payloads after m=1, zeroing, and scalxc stages."""

    if bool(apply_m1_constraints):
        frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(lconm1))
    after_m1 = frzl
    after_zero_m1 = vmec_zero_m1_zforce(frzl=after_m1, enabled=zero_m1)
    after_scalxc = normalize_force_blocks(vmec_apply_scalxc_to_tomnsps(frzl=after_zero_m1, s=s))
    return ResidualForcePayloadStages(
        after_m1=after_m1,
        after_zero_m1=after_zero_m1,
        after_scalxc=after_scalxc,
    )


def preconditioner_output_blocks_np(*, frzl_rz, lam_prec) -> ForceBlocks:
    """Apply lambda preconditioner factors to host preconditioner outputs."""
    lam = np.asarray(lam_prec)
    return ForceBlocks(
        frcc=np.asarray(frzl_rz.frcc),
        frss=None if frzl_rz.frss is None else np.asarray(frzl_rz.frss),
        fzsc=np.asarray(frzl_rz.fzsc),
        fzcs=None if frzl_rz.fzcs is None else np.asarray(frzl_rz.fzcs),
        flsc=np.asarray(frzl_rz.flsc) * lam,
        flcs=None if frzl_rz.flcs is None else np.asarray(frzl_rz.flcs) * lam,
        frsc=None if getattr(frzl_rz, "frsc", None) is None else np.asarray(frzl_rz.frsc),
        frcs=None if getattr(frzl_rz, "frcs", None) is None else np.asarray(frzl_rz.frcs),
        fzcc=None if getattr(frzl_rz, "fzcc", None) is None else np.asarray(frzl_rz.fzcc),
        fzss=None if getattr(frzl_rz, "fzss", None) is None else np.asarray(frzl_rz.fzss),
        flcc=None if getattr(frzl_rz, "flcc", None) is None else np.asarray(frzl_rz.flcc) * lam,
        flss=None if getattr(frzl_rz, "flss", None) is None else np.asarray(frzl_rz.flss) * lam,
    )


def preconditioner_output_blocks_jax(*, frzl_rz, lam_prec) -> ForceBlocks:
    """Apply lambda preconditioner factors to device preconditioner outputs."""
    lam = jnp.asarray(lam_prec)
    frcc = jnp.asarray(frzl_rz.frcc)
    fzsc = jnp.asarray(frzl_rz.fzsc)
    flsc = jnp.asarray(frzl_rz.flsc) * lam
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
        flcc = jnp.asarray(frzl_rz.flcc) * lam
    if getattr(frzl_rz, "flss", None) is not None:
        flss = jnp.asarray(frzl_rz.flss) * lam
    return ForceBlocks(
        frcc=frcc,
        frss=frzl_rz.frss,
        fzsc=fzsc,
        fzcs=frzl_rz.fzcs,
        flsc=flsc,
        flcs=None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * lam),
        frsc=frsc,
        frcs=frcs,
        fzcc=fzcc,
        fzss=fzss,
        flcc=flcc,
        flss=flss,
    )


def radial_preconditioner_output_blocks_jax(
    *,
    frzl,
    rz_scale,
    l_scale,
    precond_radial_alpha,
    precond_lambda_alpha,
    apply_radial_tridi_func,
) -> ForceBlocks:
    """Apply the scalar radial preconditioner policy to residual force blocks."""

    rz_weight = jnp.asarray(rz_scale)[:, None, None]
    l_weight = jnp.asarray(l_scale)[:, None, None]
    frcc = apply_radial_tridi_func(frzl.frcc * rz_weight, precond_radial_alpha)
    fzsc = apply_radial_tridi_func(frzl.fzsc * rz_weight, precond_radial_alpha)
    flsc = apply_radial_tridi_func(frzl.flsc * l_weight, precond_lambda_alpha)
    return ForceBlocks(
        frcc=frcc,
        frss=(
            apply_radial_tridi_func(frzl.frss * rz_weight, precond_radial_alpha)
            if frzl.frss is not None
            else None
        ),
        fzsc=fzsc,
        fzcs=(
            apply_radial_tridi_func(frzl.fzcs * rz_weight, precond_radial_alpha)
            if frzl.fzcs is not None
            else None
        ),
        flsc=flsc,
        flcs=(
            apply_radial_tridi_func(frzl.flcs * l_weight, precond_lambda_alpha)
            if frzl.flcs is not None
            else None
        ),
        frsc=(
            apply_radial_tridi_func(frzl.frsc * rz_weight, precond_radial_alpha)
            if getattr(frzl, "frsc", None) is not None
            else jnp.zeros_like(frcc)
        ),
        frcs=(
            apply_radial_tridi_func(frzl.frcs * rz_weight, precond_radial_alpha)
            if getattr(frzl, "frcs", None) is not None
            else jnp.zeros_like(frcc)
        ),
        fzcc=(
            apply_radial_tridi_func(frzl.fzcc * rz_weight, precond_radial_alpha)
            if getattr(frzl, "fzcc", None) is not None
            else jnp.zeros_like(fzsc)
        ),
        fzss=(
            apply_radial_tridi_func(frzl.fzss * rz_weight, precond_radial_alpha)
            if getattr(frzl, "fzss", None) is not None
            else jnp.zeros_like(fzsc)
        ),
        flcc=(
            apply_radial_tridi_func(frzl.flcc * l_weight, precond_lambda_alpha)
            if getattr(frzl, "flcc", None) is not None
            else jnp.zeros_like(flsc)
        ),
        flss=(
            apply_radial_tridi_func(frzl.flss * l_weight, precond_lambda_alpha)
            if getattr(frzl, "flss", None) is not None
            else jnp.zeros_like(flsc)
        ),
    )


_ForceBlocks = ForceBlocks
_zero_edge_rz_force_block = zero_edge_rz_force_block
_zero_edge_rz_force_blocks = zero_edge_rz_force_blocks
_normalize_force_blocks = normalize_force_blocks
_residual_force_payload_after_m1_scalxc = residual_force_payload_after_m1_scalxc
_residual_force_payload_m1_scalxc_stages = residual_force_payload_m1_scalxc_stages
_preconditioner_output_blocks_np = preconditioner_output_blocks_np
_preconditioner_output_blocks_jax = preconditioner_output_blocks_jax
_radial_preconditioner_output_blocks_jax = radial_preconditioner_output_blocks_jax

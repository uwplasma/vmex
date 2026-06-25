"""Force-block scaling and norm helpers for VMEC residual solves.

These helpers are split out of ``solve.py`` so residual iteration code can keep
its hot-path math testable without depending on the full solver controller.
"""

from __future__ import annotations

import numpy as np

from ...._compat import jnp
from .payload_blocks import ForceBlocks


def force_blocks_from_update_order(blocks) -> ForceBlocks:
    """Map update/velocity channel order to force-norm channel order."""

    return ForceBlocks(
        frcc=blocks.rcc,
        frss=blocks.rss,
        fzsc=blocks.zsc,
        fzcs=blocks.zcs,
        flsc=blocks.lsc,
        flcs=blocks.lcs,
        frsc=blocks.rsc,
        frcs=blocks.rcs,
        fzcc=blocks.zcc,
        fzss=blocks.zss,
        flcc=blocks.lcc,
        flss=blocks.lss,
    )


def mode_weight_force_blocks_np(
    blocks: ForceBlocks,
    *,
    w_mode_mn,
    zeros_coeff,
) -> ForceBlocks:
    """Scale preconditioned host force blocks by Fourier-mode weights.

    Missing optional blocks intentionally reuse ``zeros_coeff`` so the hot host
    path avoids repeated zero-array allocations.
    """

    weight = np.asarray(w_mode_mn)[None, :, :]
    zero = zeros_coeff

    def _optional_scale(a):
        return np.asarray(a) * weight if a is not None else zero

    return ForceBlocks(
        frcc=np.asarray(blocks.frcc) * weight,
        frss=_optional_scale(blocks.frss),
        fzsc=np.asarray(blocks.fzsc) * weight,
        fzcs=_optional_scale(blocks.fzcs),
        flsc=np.asarray(blocks.flsc) * weight,
        flcs=_optional_scale(blocks.flcs),
        frsc=_optional_scale(blocks.frsc),
        frcs=_optional_scale(blocks.frcs),
        fzcc=_optional_scale(blocks.fzcc),
        fzss=_optional_scale(blocks.fzss),
        flcc=_optional_scale(blocks.flcc),
        flss=_optional_scale(blocks.flss),
    )


def mode_weight_force_blocks_jax(
    blocks: ForceBlocks,
    *,
    w_mode_mn,
) -> ForceBlocks:
    """Scale preconditioned device force blocks by Fourier-mode weights."""

    weight = jnp.asarray(w_mode_mn)[None, :, :]

    def _optional_scale(a, like):
        return (a if a is not None else jnp.zeros_like(like)) * weight

    return ForceBlocks(
        frcc=jnp.asarray(blocks.frcc) * weight,
        frss=_optional_scale(blocks.frss, blocks.frcc),
        fzsc=jnp.asarray(blocks.fzsc) * weight,
        fzcs=_optional_scale(blocks.fzcs, blocks.fzsc),
        flsc=jnp.asarray(blocks.flsc) * weight,
        flcs=_optional_scale(blocks.flcs, blocks.flsc),
        frsc=jnp.asarray(blocks.frsc) * weight,
        frcs=jnp.asarray(blocks.frcs) * weight,
        fzcc=jnp.asarray(blocks.fzcc) * weight,
        fzss=jnp.asarray(blocks.fzss) * weight,
        flcc=jnp.asarray(blocks.flcc) * weight,
        flss=jnp.asarray(blocks.flss) * weight,
    )


def lambda_preconditioned_full_norm(frzl_pre, *, use_jax: bool):
    """Return VMEC2000 full-mesh lambda preconditioned residual norm."""

    xp = jnp if bool(use_jax) else np
    flsc = xp.asarray(frzl_pre.flsc)
    gcl2_full = xp.sum(flsc[1:] * flsc[1:])
    if frzl_pre.flcs is not None:
        flcs = xp.asarray(frzl_pre.flcs)
        gcl2_full = gcl2_full + xp.sum(flcs[1:] * flcs[1:])
    if getattr(frzl_pre, "flcc", None) is not None:
        flcc = xp.asarray(frzl_pre.flcc)
        gcl2_full = gcl2_full + xp.sum(flcc[1:] * flcc[1:])
    if getattr(frzl_pre, "flss", None) is not None:
        flss = xp.asarray(frzl_pre.flss)
        gcl2_full = gcl2_full + xp.sum(flss[1:] * flss[1:])
    return gcl2_full


def residual_fsq_from_norms(norms, *, gcr2, gcz2, gcl2):
    """Return VMEC residual ``(FSQR, FSQZ, FSQL)`` scalars from force norms."""

    fsqr = norms.r1 * norms.fnorm * gcr2
    fsqz = norms.r1 * norms.fnorm * gcz2
    fsql = norms.fnormL * gcl2
    return fsqr, fsqz, fsql


def safe_dt_from_force_blocks(
    *,
    dt_nominal: float,
    max_coeff_delta_rms: float,
    blocks: ForceBlocks,
) -> float:
    """Limit ``dt`` from force RMS when the stability guard is enabled."""

    frcc = jnp.asarray(blocks.frcc)
    frss = jnp.asarray(blocks.frss) if blocks.frss is not None else jnp.zeros_like(frcc)
    fzsc = jnp.asarray(blocks.fzsc)
    fzcs = jnp.asarray(blocks.fzcs) if blocks.fzcs is not None else jnp.zeros_like(fzsc)
    flsc = jnp.asarray(blocks.flsc)
    flcs = jnp.asarray(blocks.flcs) if blocks.flcs is not None else jnp.zeros_like(flsc)
    frsc = jnp.asarray(blocks.frsc) if blocks.frsc is not None else jnp.zeros_like(frcc)
    frcs = jnp.asarray(blocks.frcs) if blocks.frcs is not None else jnp.zeros_like(frcc)
    fzcc = jnp.asarray(blocks.fzcc) if blocks.fzcc is not None else jnp.zeros_like(fzsc)
    fzss = jnp.asarray(blocks.fzss) if blocks.fzss is not None else jnp.zeros_like(fzsc)
    flcc = jnp.asarray(blocks.flcc) if blocks.flcc is not None else jnp.zeros_like(flsc)
    flss = jnp.asarray(blocks.flss) if blocks.flss is not None else jnp.zeros_like(flsc)
    rms = jnp.sqrt(
        jnp.mean(
            frcc * frcc
            + frss * frss
            + frsc * frsc
            + frcs * frcs
            + fzsc * fzsc
            + fzcs * fzcs
            + fzcc * fzcc
            + fzss * fzss
            + flsc * flsc
            + flcs * flcs
            + flcc * flcc
            + flss * flss
        )
    )
    rms_f = float(np.asarray(rms))
    if not np.isfinite(rms_f) or rms_f <= 0.0:
        return max(float(dt_nominal), 1e-12)
    # With this integrator, first-step coefficient update is O(dt^2 * force).
    dt_lim = np.sqrt(float(max_coeff_delta_rms) / max(rms_f, 1e-30))
    dt_eff = min(float(dt_nominal), float(dt_lim))
    return max(dt_eff, 1e-12)

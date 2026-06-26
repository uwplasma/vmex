"""JAX ports of VMEC2000 1D (radial) preconditioner operators.

This module mirrors the reference NumPy implementation in
:mod:`vmec_jax.preconditioner_1d`, but uses JAX arrays/ops so the fixed-point
update loop stays JIT-able and differentiable.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
import os
import functools
from functools import partial
from types import SimpleNamespace

import numpy as np

from ._compat import jax, jnp, jit, has_jax
from .kernels.tomnsp import TomnspsRZL

_LAMBDA_PRECOND_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()

# Cache env-var flags at import time — these don't change during a run and
# os.getenv() + .strip().lower() was being called 7000+ times per solve.
_ENV_USE_PRECOMPUTED: bool | None = None
_ENV_USE_LAX_TRIDI: bool | None = None
_ENV_RZ_MATRIX_ASSEMBLY_JIT: bool | None = None
_ENV_RZ_MATRIX_FULL_JIT: bool | None = None


def _get_env_tridi_flags() -> tuple[bool, bool]:
    """Return (use_precomputed, use_lax_tridi) from env, cached after first call."""
    global _ENV_USE_PRECOMPUTED, _ENV_USE_LAX_TRIDI
    if _ENV_USE_PRECOMPUTED is None:
        _ENV_USE_PRECOMPUTED = os.getenv("VMEC_JAX_TRIDI_PRECOMPUTE", "0").strip().lower() not in (
            "", "0", "false", "no",
        )
    if _ENV_USE_LAX_TRIDI is None:
        _ENV_USE_LAX_TRIDI = os.getenv("VMEC_JAX_TRIDI_SOLVE", "").strip().lower() in (
            "1", "true", "yes", "lax", "force",
        )
    return _ENV_USE_PRECOMPUTED, _ENV_USE_LAX_TRIDI


def _rz_matrix_assembly_jit_enabled() -> bool:
    """Return whether to JIT only the small R/Z matrix assembly helper.

    The default stays on because fresh-process profiling shows the compiled
    helper is faster even for cold low-iteration 3D cases. Set
    ``VMEC_JAX_RZ_MATRIX_ASSEMBLY_JIT=0`` only for diagnostics.
    """

    global _ENV_RZ_MATRIX_ASSEMBLY_JIT
    if _ENV_RZ_MATRIX_ASSEMBLY_JIT is None:
        _ENV_RZ_MATRIX_ASSEMBLY_JIT = os.getenv("VMEC_JAX_RZ_MATRIX_ASSEMBLY_JIT", "1").strip().lower() not in (
            "",
            "0",
            "false",
            "no",
            "off",
        )
    return _ENV_RZ_MATRIX_ASSEMBLY_JIT


def _rz_matrix_full_jit_enabled() -> bool:
    """Return whether to JIT the full R/Z coefficient-and-assembly builder."""

    global _ENV_RZ_MATRIX_FULL_JIT
    if _ENV_RZ_MATRIX_FULL_JIT is None:
        _ENV_RZ_MATRIX_FULL_JIT = os.getenv("VMEC_JAX_RZ_MATRIX_FULL_JIT", "1").strip().lower() not in (
            "",
            "0",
            "false",
            "no",
            "off",
        )
    return _ENV_RZ_MATRIX_FULL_JIT


def _cache_allowed() -> bool:
    if not has_jax():
        return True
    try:
        from jax import core

        return bool(core.trace_ctx.is_top_level())
    except Exception:
        return False


def _jit_array_pytree_supported(value: Any) -> bool:
    """Return whether ``value`` can be passed as a dynamic JAX argument.

    Some test and host-control paths use ``types.SimpleNamespace`` containers
    for VMEC parity data.  Those are intentionally not registered as pytrees,
    so treating them as dynamic JIT arguments fails on newer JAX releases.
    """

    if not has_jax():
        return False
    try:
        leaves = jax.tree_util.tree_leaves(value)
    except Exception:
        return False
    try:
        for leaf in leaves:
            jnp.asarray(leaf)
    except Exception:
        return False
    return True


def _lambda_precond_cache_limit() -> int:
    raw = os.getenv("VMEC_JAX_PRECOND_CACHE_LIMIT", "16").strip()
    try:
        return max(1, int(raw))
    except Exception:
        return 16


def _lambda_precond_cache_get(key):
    cached = _LAMBDA_PRECOND_JIT_CACHE.get(key)
    if cached is not None:
        _LAMBDA_PRECOND_JIT_CACHE.move_to_end(key)
    return cached


def _lambda_precond_cache_put(key, value) -> None:
    _LAMBDA_PRECOND_JIT_CACHE[key] = value
    _LAMBDA_PRECOND_JIT_CACHE.move_to_end(key)
    limit = _lambda_precond_cache_limit()
    while len(_LAMBDA_PRECOND_JIT_CACHE) > limit:
        _LAMBDA_PRECOND_JIT_CACHE.popitem(last=False)


def clear_preconditioner_jit_caches() -> None:
    """Drop local preconditioner executable caches to limit late-run retention."""
    _LAMBDA_PRECOND_JIT_CACHE.clear()
    _make_rz_preconditioner_apply_jit.cache_clear()


def _contains_namespace_leaf(value: Any) -> bool:
    """Return True for test/lightweight namespace trees that JAX cannot abstract."""
    if isinstance(value, SimpleNamespace):
        return True
    if isinstance(value, dict):
        return any(_contains_namespace_leaf(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_namespace_leaf(item) for item in value)
    return False


def _sqrt_profiles_from_ns(ns: int, *, dtype) -> tuple[Any, Any]:
    ns = int(ns)
    if ns <= 0:
        z = jnp.zeros((0,), dtype=dtype)
        return z, z
    if ns == 1:
        return jnp.zeros((1,), dtype=dtype), jnp.zeros((0,), dtype=dtype)
    denom = jnp.asarray(float(ns - 1), dtype=dtype)
    full_pos = jnp.linspace(0.0, 1.0, ns, dtype=dtype)
    sqrt_sf = jnp.sqrt(jnp.maximum(full_pos, 0.0))
    half_pos = (jnp.arange(ns - 1, dtype=dtype) + 0.5) / denom
    sqrt_sh = jnp.sqrt(jnp.maximum(half_pos, 0.0))
    return sqrt_sf, sqrt_sh


def _sm_sp_from_profiles(sqrt_sf, sqrt_sh) -> tuple[Any, Any]:
    sqrt_sf = jnp.asarray(sqrt_sf)
    sqrt_sh = jnp.asarray(sqrt_sh)
    ns = int(sqrt_sf.shape[0])
    if ns < 2:
        z = jnp.zeros((0,), dtype=sqrt_sf.dtype)
        return z, z
    denom_outer = jnp.where(sqrt_sf[1:] != 0.0, sqrt_sf[1:], 1.0)
    sm = sqrt_sh / denom_outer
    denom_inner = jnp.where(sqrt_sf[:-1] != 0.0, sqrt_sf[:-1], 1.0)
    sp = sqrt_sh / denom_inner
    sp = sp.at[0].set(sm[0])
    return sm, sp


def _wint_from_config(*, cfg, dtype) -> Any:
    ntheta = int(cfg.ntheta)
    nzeta = int(cfg.nzeta)
    lasym = bool(cfg.lasym)
    ntheta_even = 2 * (ntheta // 2)
    ntheta_reduced = ntheta_even // 2 + 1
    ntheta_eff = ntheta_even if lasym else ntheta_reduced
    if ntheta_eff <= 0:
        return jnp.zeros((0,), dtype=dtype)
    if lasym:
        dnorm3 = 1.0 / (float(nzeta) * float(ntheta_even))
    else:
        # Reduced theta grid integrates the full even-theta domain via doubled interior weights.
        dnorm3 = 1.0 / (float(nzeta) * float(ntheta_reduced - 1))
    w_int = jnp.full((ntheta_eff,), dnorm3, dtype=dtype)
    if (not lasym) and ntheta_eff > 0:
        w_int = w_int.at[0].multiply(0.5)
        w_int = w_int.at[-1].multiply(0.5)
    return w_int


def lambda_preconditioner(
    *,
    bc,
    trig,
    s,
    cfg,
    damping_factor: float = 2.0,
    return_faclam: bool = False,
    return_debug: bool = False,
    r0scale: float | None = None,
) -> Any:
    """Compute the VMEC lambda preconditioner (n>=0 storage) in JAX.

    Parameters
    ----------
    bc:
        :class:`~vmec_jax.kernels.bcovar.VmecHalfMeshBcovar` (or compatible) providing
        ``guu/guv/gvv`` and ``jac.sqrtg`` on the *half mesh* and scalar ``lamscale``.
    trig:
        Unused placeholder to keep parity-call signatures consistent with the
        NumPy helper module.
    s:
        Full-mesh radial coordinate array.
    cfg:
        VMEC configuration (mpol/ntor/nfp/ntheta/nzeta/lasym/lthreed).
    damping_factor:
        Damping used in the diagonal lambda preconditioner.
    """
    if r0scale is None:
        r0scale = float(getattr(trig, "r0scale", 1.0)) if trig is not None else 1.0
    r0scale = float(r0scale)
    s = jnp.asarray(s)
    dtype = s.dtype

    guu = jnp.asarray(bc.guu, dtype=dtype)
    guv = jnp.asarray(bc.guv, dtype=dtype)
    gvv = jnp.asarray(bc.gvv, dtype=dtype)
    gsqrt = jnp.asarray(bc.jac.sqrtg, dtype=dtype)

    ns_full = int(guu.shape[0])
    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    if ns_full < 2:
        out = jnp.zeros((ns_full, mpol, nrange), dtype=dtype)
        if return_debug:
            debug = {
                "blam_pre": jnp.zeros((ns_full,), dtype=dtype),
                "clam_pre": jnp.zeros((ns_full,), dtype=dtype),
                "dlam_pre": jnp.zeros((ns_full,), dtype=dtype),
                "blam_post": jnp.zeros((ns_full,), dtype=dtype),
                "clam_post": jnp.zeros((ns_full,), dtype=dtype),
                "dlam_post": jnp.zeros((ns_full,), dtype=dtype),
            }
            if return_faclam:
                return out, jnp.zeros_like(out), debug
            return out, debug
        return (out, jnp.zeros_like(out)) if return_faclam else out

    w_int = _wint_from_config(cfg=cfg, dtype=dtype)

    w3 = w_int[None, :, None]
    gsqrt_safe = jnp.where(gsqrt != 0.0, gsqrt, 1.0)
    b_pre = jnp.sum(guu / gsqrt_safe * w3, axis=(1, 2))
    c_pre = jnp.sum(gvv / gsqrt_safe * w3, axis=(1, 2))
    if bool(cfg.lthreed):
        d_pre = jnp.sum(guv / gsqrt_safe * w3, axis=(1, 2))
    else:
        d_pre = jnp.zeros_like(b_pre)

    if ns_full >= 2:
        b_pre = b_pre.at[0].set(b_pre[1])
        c_pre = c_pre.at[0].set(c_pre[1])
        d_pre = d_pre.at[0].set(d_pre[1])

    b_post = b_pre
    c_post = c_pre
    d_post = d_pre
    if ns_full >= 2:
        b_next = jnp.concatenate([b_pre[2:], jnp.zeros((1,), dtype=dtype)])
        c_next = jnp.concatenate([c_pre[2:], jnp.zeros((1,), dtype=dtype)])
        d_next = jnp.concatenate([d_pre[2:], jnp.zeros((1,), dtype=dtype)])
        b_post = b_post.at[1:].set(0.5 * (b_pre[1:] + b_next))
        c_post = c_post.at[1:].set(0.5 * (c_pre[1:] + c_next))
        d_post = d_post.at[1:].set(0.5 * (d_pre[1:] + d_next))

    blam_pre = b_pre
    clam_pre = c_pre
    dlam_pre = d_pre
    blam_post = b_post
    clam_post = c_post
    dlam_post = d_post

    lamscale = jnp.asarray(bc.lamscale, dtype=dtype)
    p_factor = jnp.asarray(float(damping_factor), dtype=dtype) / (
        4.0 * (float(r0scale) ** 2) * lamscale * lamscale
    )

    sqrt_sf, _sqrt_sh = _sqrt_profiles_from_ns(ns_full, dtype=dtype)
    sqrt_sf = sqrt_sf.at[-1].set(1.0)

    m = jnp.arange(mpol, dtype=dtype)
    n = jnp.arange(nrange, dtype=dtype)
    nfp = float(cfg.nfp)
    tnn = (n * nfp) ** 2
    tmm = m * m
    tmn = 2.0 * m[:, None] * n[None, :] * nfp
    pwr = jnp.minimum(tmm / (16.0 * 16.0), 8.0)

    bF = b_post[:, None, None]
    cF = c_post[:, None, None]
    dF = d_post[:, None, None]
    faclam_raw = (tnn[None, None, :] * bF) + (tmn[None, :, :] * jnp.copysign(dF, bF)) + (tmm[None, :, None] * cF)
    faclam_raw = jnp.where(faclam_raw == 0.0, -1.0e-10, faclam_raw)

    sqrt_pow = (sqrt_sf[:, None, None]) ** (pwr[None, :, None])
    lam_prec = p_factor * sqrt_pow / faclam_raw

    # VMEC special-case m=n=0 preconditioner (chip/iota channel).
    b_safe = jnp.where(b_post != 0.0, b_post, jnp.asarray(-1.0e-10, dtype=dtype))
    p_factor00 = p_factor * (lamscale * lamscale)
    lam_prec = lam_prec.at[:, 0, 0].set(p_factor00 / b_safe)

    # VMEC jlam(m)=2 => js=1 (0-based index 0) is zero for all m,n except (0,0).
    if ns_full > 0:
        axis_mask = jnp.zeros((mpol, nrange), dtype=dtype).at[0, 0].set(1.0)
        lam_prec = lam_prec.at[0].set(lam_prec[0] * axis_mask)

    if return_debug:
        debug = {
            "blam_pre": blam_pre,
            "clam_pre": clam_pre,
            "dlam_pre": dlam_pre,
            "blam_post": blam_post,
            "clam_post": clam_post,
            "dlam_post": dlam_post,
        }
        if return_faclam:
            # VMEC dumps faclam as the preconditioner (not the raw denominator).
            return lam_prec, lam_prec, debug
        return lam_prec, debug
    if return_faclam:
        # VMEC dumps faclam as the preconditioner (not the raw denominator).
        return lam_prec, lam_prec
    return lam_prec


def lambda_preconditioner_cached(
    *,
    bc,
    trig,
    s,
    cfg,
    damping_factor: float = 2.0,
    return_faclam: bool = False,
    return_debug: bool = False,
    r0scale: float | None = None,
) -> Any:
    """Cached/JIT'd lambda preconditioner keyed by (cfg, r0scale, flags)."""
    if r0scale is None:
        r0scale = float(getattr(trig, "r0scale", 1.0)) if trig is not None else 1.0
    r0scale = float(r0scale)
    if not has_jax() or (not _cache_allowed()) or (not _jit_array_pytree_supported(bc)):
        return lambda_preconditioner(
            bc=bc,
            trig=trig,
            s=s,
            cfg=cfg,
            damping_factor=damping_factor,
            return_faclam=return_faclam,
            return_debug=return_debug,
            r0scale=r0scale,
        )
    if _contains_namespace_leaf(bc):
        # Newer JAX releases reject SimpleNamespace as a dynamic argument before
        # tracing. Production callers use registered data containers, while
        # tests and small diagnostics use namespaces; keep those on the exact
        # Python/JAX-array algebra path without caching a compiled executable.
        return lambda_preconditioner(
            bc=bc,
            trig=trig,
            s=s,
            cfg=cfg,
            damping_factor=damping_factor,
            return_faclam=return_faclam,
            return_debug=return_debug,
            r0scale=r0scale,
        )

    key = (
        int(cfg.mpol),
        int(cfg.ntor),
        int(cfg.ntheta),
        int(cfg.nzeta),
        int(cfg.nfp),
        bool(cfg.lasym),
        bool(cfg.lthreed),
        float(damping_factor),
        float(r0scale),
        bool(return_faclam),
        bool(return_debug),
    )
    cached = _lambda_precond_cache_get(key)
    if cached is None:
        def _fn(bc_in, s_in):
            return lambda_preconditioner(
                bc=bc_in,
                trig=None,
                s=s_in,
                cfg=cfg,
                damping_factor=damping_factor,
                return_faclam=return_faclam,
                return_debug=return_debug,
                r0scale=r0scale,
            )

        cached = jax.jit(_fn)
        _lambda_precond_cache_put(key, cached)
    return cached(bc, s)


def _compute_preconditioning_matrix(
    *,
    xs,
    xu12,
    xu_e,
    xu_o,
    x1_o,
    r12,
    total_pressure,
    tau,
    bsupv,
    sqrtg,
    w_int,
    sqrt_sh,
    sm,
    sp,
    delta_s: Any,
    ns_full: int | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """JAX port of VMEC `precondn` matrix assembly (axisymmetric subset)."""
    xs = jnp.asarray(xs)
    xu12 = jnp.asarray(xu12)
    xu_e = jnp.asarray(xu_e)
    xu_o = jnp.asarray(xu_o)
    x1_o = jnp.asarray(x1_o)
    r12 = jnp.asarray(r12)
    total_pressure = jnp.asarray(total_pressure)
    tau = jnp.asarray(tau)
    bsupv = jnp.asarray(bsupv)
    sqrtg = jnp.asarray(sqrtg)
    w_int = jnp.asarray(w_int)
    sqrt_sh = jnp.asarray(sqrt_sh)
    sm = jnp.asarray(sm)
    sp = jnp.asarray(sp)

    ns_half = int(xs.shape[0])
    if ns_half <= 0:
        z2 = jnp.zeros((0, 2), dtype=xs.dtype)
        z1 = jnp.zeros((0,), dtype=xs.dtype)
        return z2, z2, z2, z2, z1
    ns_full = int(ns_full) if ns_full is not None else ns_half

    pfactor = jnp.asarray(-4.0, dtype=xs.dtype)
    delta_s = jnp.asarray(delta_s, dtype=xs.dtype)
    gsqrt_safe = jnp.where(sqrtg != 0.0, sqrtg, 1.0)
    sqrt_sh_safe = jnp.where(sqrt_sh != 0.0, sqrt_sh, 1.0)

    # Broadcast helpers.
    sh = sqrt_sh_safe[:, None, None]
    w3 = w_int[None, :, None]

    p_tau = pfactor * r12 * r12 * total_pressure / gsqrt_safe * w3
    t1a = xu12 / delta_s
    xu_e_o = xu_e[1 : ns_half + 1]
    xu_e_i = xu_e[:ns_half]
    xu_o_o = xu_o[1 : ns_half + 1]
    xu_o_i = xu_o[:ns_half]
    t2a = 0.25 * (xu_e_o / sh + xu_o_o) / sh
    t3a = 0.25 * (xu_e_i / sh + xu_o_i) / sh

    ax0 = jnp.sum(p_tau * (t1a * t1a), axis=(1, 2))
    ax1 = jnp.sum(p_tau * (t1a + t2a) * (-t1a + t3a), axis=(1, 2))
    ax2 = jnp.sum(p_tau * (t1a + t2a) * (t1a + t2a), axis=(1, 2))
    ax3 = jnp.sum(p_tau * (-t1a + t3a) * (-t1a + t3a), axis=(1, 2))
    ax = jnp.stack([ax0, ax1, ax2, ax3], axis=1)

    x1_o_o = x1_o[1 : ns_half + 1]
    x1_o_i = x1_o[:ns_half]
    t1b = 0.5 * (xs + 0.5 / sh * x1_o_o)
    t2b = 0.5 * (xs + 0.5 / sh * x1_o_i)
    bx0 = jnp.sum(p_tau * t1b * t2b, axis=(1, 2))
    bx1 = jnp.sum(p_tau * t1b * t1b, axis=(1, 2))
    bx2 = jnp.sum(p_tau * t2b * t2b, axis=(1, 2))
    bx = jnp.stack([bx0, bx1, bx2], axis=1)

    cx = jnp.sum(0.25 * pfactor * (bsupv * bsupv) * sqrtg * w3, axis=(1, 2))

    axm0 = -ax[:, 0]
    axm1 = ax[:, 1] * sm * sp
    bxm0 = bx[:, 0]
    bxm1 = bx[:, 0] * sm * sp
    axm = jnp.stack([axm0, axm1], axis=1)
    bxm = jnp.stack([bxm0, bxm1], axis=1)

    # Full-grid accumulation (Fortran precondn loop over jf):
    # inner contribution uses jhi=jf-1 for jf>0, outer uses jho=jf for jf<ns_half.
    # Build full-length bases (ns_half+1) and truncate to ns_full.
    z = jnp.zeros((1,), dtype=xs.dtype)
    ax0_inner = jnp.concatenate([z, ax[:, 0]], axis=0)[:ns_full]
    ax1_inner = jnp.concatenate([z, ax[:, 2] * (sm * sm)], axis=0)[:ns_full]
    bx0_inner = jnp.concatenate([z, bx[:, 1]], axis=0)[:ns_full]
    bx1_inner = jnp.concatenate([z, bx[:, 1] * (sm * sm)], axis=0)[:ns_full]
    cx_inner = jnp.concatenate([z, cx], axis=0)[:ns_full]

    ax0_outer = jnp.concatenate([ax[:, 0], z], axis=0)[:ns_full]
    ax1_outer = jnp.concatenate([ax[:, 3] * (sp * sp), z], axis=0)[:ns_full]
    bx0_outer = jnp.concatenate([bx[:, 2], z], axis=0)[:ns_full]
    bx1_outer = jnp.concatenate([bx[:, 2] * (sp * sp), z], axis=0)[:ns_full]
    cx_outer = jnp.concatenate([cx, z], axis=0)[:ns_full]

    axd0 = ax0_inner + ax0_outer
    axd1 = ax1_inner + ax1_outer
    bxd0 = bx0_inner + bx0_outer
    bxd1 = bx1_inner + bx1_outer
    cxd = cx_inner + cx_outer
    axd = jnp.stack([axd0, axd1], axis=1)
    bxd = jnp.stack([bxd0, bxd1], axis=1)
    return axm, axd, bxm, bxd, cxd


def _resolve_tridi_flags(*, use_precomputed: bool | None, use_lax_tridi: bool | None) -> tuple[bool, bool]:
    if use_precomputed is None:
        use_precomputed = os.getenv("VMEC_JAX_TRIDI_PRECOMPUTE", "0").strip().lower() not in (
            "",
            "0",
            "false",
            "no",
        )
    if use_lax_tridi is None:
        use_lax_tridi = os.getenv("VMEC_JAX_TRIDI_SOLVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "lax",
            "force",
        )
    return bool(use_precomputed), bool(use_lax_tridi)


def _assemble_rz_preconditioner_matrices_impl_unjitted(
    *,
    arm,
    ard,
    brm,
    brd,
    azm,
    azd,
    bzm,
    bzd,
    cxd,
    delta_s,
    cfg,
    jmax_override: int | None = None,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
) -> tuple[dict[str, Any], Any, int]:
    """Assemble VMEC R/Z preconditioner matrices from cached parity coefficients."""
    ard = jnp.asarray(ard)
    ns = int(ard.shape[0])
    ns_f_default = max(ns - 1, 1)
    if jmax_override is None:
        ns_f = ns_f_default
    else:
        ns_f = int(max(1, min(int(jmax_override), ns)))
    dtype = ard.dtype

    arm = jnp.asarray(arm, dtype=dtype)
    brm = jnp.asarray(brm, dtype=dtype)
    brd = jnp.asarray(brd, dtype=dtype)
    azm = jnp.asarray(azm, dtype=dtype)
    azd = jnp.asarray(azd, dtype=dtype)
    bzm = jnp.asarray(bzm, dtype=dtype)
    bzd = jnp.asarray(bzd, dtype=dtype)
    cxd = jnp.asarray(cxd, dtype=dtype)
    delta_s = jnp.asarray(delta_s, dtype=dtype)

    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    nfp = float(cfg.nfp)
    m = jnp.arange(mpol, dtype=dtype)
    n = jnp.arange(nrange, dtype=dtype)
    m2 = (m * m)[None, :, None]
    n2 = ((n * nfp) ** 2)[None, None, :]
    m_par = (jnp.arange(mpol) % 2).astype(jnp.int32)

    ns_half = int(arm.shape[0])
    pad_rows = max(ns_f - ns_half, 0)
    if pad_rows > 0:
        z_arm = jnp.zeros((pad_rows, arm.shape[1]), dtype=dtype)
        z_azm = jnp.zeros((pad_rows, azm.shape[1]), dtype=dtype)
        arm_f = jnp.concatenate([arm, z_arm], axis=0)
        brm_f = jnp.concatenate([brm, z_arm], axis=0)
        azm_f = jnp.concatenate([azm, z_azm], axis=0)
        bzm_f = jnp.concatenate([bzm, z_azm], axis=0)
    else:
        arm_f = arm[:ns_f]
        brm_f = brm[:ns_f]
        azm_f = azm[:ns_f]
        bzm_f = bzm[:ns_f]
    ard_f = ard[:ns_f]
    brd_f = brd[:ns_f]
    azd_f = azd[:ns_f]
    bzd_f = bzd[:ns_f]
    cxd_f = cxd[:ns_f]

    arm_m = arm_f[:, m_par]
    brm_m = brm_f[:, m_par]
    ard_m = ard_f[:, m_par]
    brd_m = brd_f[:, m_par]
    azm_m = azm_f[:, m_par]
    bzm_m = bzm_f[:, m_par]
    azd_m = azd_f[:, m_par]
    bzd_m = bzd_f[:, m_par]

    ar = -(arm_m[:, :, None] + brm_m[:, :, None] * m2)
    az = -(azm_m[:, :, None] + bzm_m[:, :, None] * m2)
    dr = -(ard_m[:, :, None] + brd_m[:, :, None] * m2 + cxd_f[:, None, None] * n2)
    dz = -(azd_m[:, :, None] + bzd_m[:, :, None] * m2 + cxd_f[:, None, None] * n2)

    br = jnp.zeros_like(ar)
    bz = jnp.zeros_like(az)
    if ns_f > 1:
        br = br.at[1:].set(-(arm_m[:-1, :, None] + brm_m[:-1, :, None] * m2))
        bz = bz.at[1:].set(-(azm_m[:-1, :, None] + bzm_m[:-1, :, None] * m2))

    if ns_f > 0 and mpol > 1:
        ar = ar.at[0, 1:, :].set(0.0)
        az = az.at[0, 1:, :].set(0.0)
        dr = dr.at[0, 1:, :].set(0.0)
        dz = dz.at[0, 1:, :].set(0.0)

    if ns_f > 1 and mpol > 1:
        dr = dr.at[1, 1, :].add(br[1, 1, :])
        dz = dz.at[1, 1, :].add(bz[1, 1, :])

    if ns_f >= ns and ns > 0:
        edge_pedestal = jnp.asarray(0.05, dtype=dtype)
        fac = jnp.asarray(0.25, dtype=dtype)
        hs = jnp.asarray(delta_s, dtype=dtype)
        mult_fac = jnp.minimum(fac, fac * hs * jnp.asarray(15.0, dtype=dtype))
        edge_idx = ns - 1
        if mpol > 0:
            dr = dr.at[edge_idx, 0:1, :].multiply(1.0 + edge_pedestal)
            dz = dz.at[edge_idx, 0:1, :].multiply(1.0 + edge_pedestal)
        if mpol > 1:
            dr = dr.at[edge_idx, 1:2, :].multiply(1.0 + edge_pedestal)
            dz = dz.at[edge_idx, 1:2, :].multiply(1.0 + edge_pedestal)
        if mpol > 2:
            dr = dr.at[edge_idx, 2:, :].multiply(1.0 + 2.0 * edge_pedestal)
            dz = dz.at[edge_idx, 2:, :].multiply(1.0 + 2.0 * edge_pedestal)
        if mpol > 0 and nrange > 0:
            dz = dz.at[edge_idx, 0, 0].multiply((1.0 - mult_fac) / (1.0 + edge_pedestal))

    use_precomputed, use_lax_tridi = _resolve_tridi_flags(
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )
    cr = ir = cz = iz = None
    if use_precomputed:
        cr, ir = _tridi_precompute_coeffs(ar, dr, br)
        cz, iz = _tridi_precompute_coeffs(az, dz, bz)
    dlr_t = dr_t = dur_t = None
    dlz_t = dz_t = duz_t = None
    if use_lax_tridi:
        dlr_t, dr_t, dur_t = _tridi_pretranspose_for_lax(ar, dr, br)
        dlz_t, dz_t, duz_t = _tridi_pretranspose_for_lax(az, dz, bz)

    jmin_m = jnp.where(jnp.arange(mpol) > 0, 1, 0).astype(jnp.int32)
    jmin = jmin_m[:, None] * jnp.ones((mpol, nrange), dtype=jnp.int32)

    mats = {
        "ar": ar,
        "br": br,
        "dr": dr,
        "az": az,
        "bz": bz,
        "dz": dz,
        "arm_parity": arm,
        "ard_parity": ard,
        "brm_parity": brm,
        "brd_parity": brd,
        "azm_parity": azm,
        "azd_parity": azd,
        "bzm_parity": bzm,
        "bzd_parity": bzd,
        "cxd_full": cxd,
        "delta_s": delta_s,
    }
    if cr is not None:
        mats["cr"] = cr
    if ir is not None:
        mats["ir"] = ir
    if cz is not None:
        mats["cz"] = cz
    if iz is not None:
        mats["iz"] = iz
    if dlr_t is not None:
        mats["dlr_t"] = dlr_t
    if dr_t is not None:
        mats["dr_t"] = dr_t
    if dur_t is not None:
        mats["dur_t"] = dur_t
    if dlz_t is not None:
        mats["dlz_t"] = dlz_t
    if dz_t is not None:
        mats["dz_t"] = dz_t
    if duz_t is not None:
        mats["duz_t"] = duz_t
    return mats, jmin, int(ns_f)


_assemble_rz_preconditioner_matrices_impl_jit = partial(
    jit,
    static_argnames=("cfg", "jmax_override", "use_precomputed", "use_lax_tridi"),
)(_assemble_rz_preconditioner_matrices_impl_unjitted)


def _assemble_rz_preconditioner_matrices_impl(
    *,
    arm,
    ard,
    brm,
    brd,
    azm,
    azd,
    bzm,
    bzd,
    cxd,
    delta_s,
    cfg,
    jmax_override: int | None = None,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
) -> tuple[dict[str, Any], Any, int]:
    """Assemble R/Z preconditioner matrices, JITing only hashable configs."""
    if not _rz_matrix_assembly_jit_enabled():
        assemble = _assemble_rz_preconditioner_matrices_impl_unjitted
    else:
        try:
            hash(cfg)
        except TypeError:
            assemble = _assemble_rz_preconditioner_matrices_impl_unjitted
        else:
            assemble = _assemble_rz_preconditioner_matrices_impl_jit
    return assemble(
        arm=arm,
        ard=ard,
        brm=brm,
        brd=brd,
        azm=azm,
        azd=azd,
        bzm=bzm,
        bzd=bzd,
        cxd=cxd,
        delta_s=delta_s,
        cfg=cfg,
        jmax_override=jmax_override,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )


def _rz_preconditioner_matrices_impl(
    *,
    bc,
    k,
    s,
    cfg,
    jmax_override: int | None = None,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
) -> tuple[dict[str, Any], Any, int]:
    """Return VMEC R/Z radial preconditioner matrices (JAX, fixed-boundary)."""
    s = jnp.asarray(s)
    ns = int(s.shape[0])
    dtype = jnp.asarray(bc.guu).dtype
    w_int = _wint_from_config(cfg=cfg, dtype=dtype)

    r12 = jnp.asarray(bc.jac.r12, dtype=dtype)[1:]
    tau = jnp.asarray(bc.jac.tau, dtype=dtype)[1:]
    total_pressure = jnp.asarray(bc.bsq, dtype=dtype)[1:]
    bsupv = jnp.asarray(bc.bsupv, dtype=dtype)[1:]
    sqrtg = jnp.asarray(bc.jac.sqrtg, dtype=dtype)[1:]

    sqrt_sf, sqrt_sh = _sqrt_profiles_from_ns(ns, dtype=dtype)
    sm, sp = _sm_sp_from_profiles(sqrt_sf, sqrt_sh)
    delta_s = jnp.where(ns >= 2, s[1] - s[0], jnp.asarray(1.0, dtype=dtype))

    arm, ard, brm, brd, cxd = _compute_preconditioning_matrix(
        xs=jnp.asarray(bc.jac.zs, dtype=dtype)[1:],
        xu12=jnp.asarray(bc.jac.zu12, dtype=dtype)[1:],
        xu_e=jnp.asarray(k.pzu_even, dtype=dtype),
        xu_o=jnp.asarray(k.pzu_odd, dtype=dtype),
        x1_o=jnp.asarray(k.pz1_odd, dtype=dtype),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=sqrtg,
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns,
    )
    azm, azd, bzm, bzd, _cxd_z = _compute_preconditioning_matrix(
        xs=jnp.asarray(bc.jac.rs, dtype=dtype)[1:],
        xu12=jnp.asarray(bc.jac.ru12, dtype=dtype)[1:],
        xu_e=jnp.asarray(k.pru_even, dtype=dtype),
        xu_o=jnp.asarray(k.pru_odd, dtype=dtype),
        x1_o=jnp.asarray(k.pr1_odd, dtype=dtype),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=sqrtg,
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns,
    )
    return _assemble_rz_preconditioner_matrices_impl(
        arm=arm,
        ard=ard,
        brm=brm,
        brd=brd,
        azm=azm,
        azd=azd,
        bzm=bzm,
        bzd=bzd,
        cxd=cxd,
        delta_s=delta_s,
        cfg=cfg,
        jmax_override=jmax_override,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )


_rz_preconditioner_matrices_impl_full_jit = partial(
    jit,
    static_argnames=("cfg", "jmax_override", "use_precomputed", "use_lax_tridi"),
)(_rz_preconditioner_matrices_impl)


def rz_preconditioner_matrices(
    *,
    bc,
    k,
    trig,
    s,
    cfg,
    jmax_override: int | None = None,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
) -> tuple[dict[str, Any], Any, int]:
    """Return VMEC R/Z radial preconditioner matrices (JAX, fixed-boundary)."""
    del trig
    build = _rz_preconditioner_matrices_impl
    if _rz_matrix_full_jit_enabled() and _jit_array_pytree_supported((bc, k, s)):
        try:
            hash(cfg)
        except TypeError:
            build = _rz_preconditioner_matrices_impl
        else:
            build = _rz_preconditioner_matrices_impl_full_jit
    return build(
        bc=bc,
        k=k,
        s=s,
        cfg=cfg,
        jmax_override=jmax_override,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )


def _assemble_rz_preconditioner_matrices_numpy_host(
    *,
    arm,
    ard,
    brm,
    brd,
    azm,
    azd,
    bzm,
    bzd,
    cxd,
    delta_s,
    cfg,
    jmax_override: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Assemble the JAX R/Z matrix payload using NumPy host arrays.

    This mirrors ``_assemble_rz_preconditioner_matrices_impl_unjitted`` for
    concrete CPU host solves.  It intentionally omits precomputed/lax
    tridiagonal variants; callers fall back to the JAX helper for those
    diagnostic modes.
    """

    ard = np.asarray(ard)
    ns = int(ard.shape[0])
    ns_f_default = max(ns - 1, 1)
    ns_f = ns_f_default if jmax_override is None else int(max(1, min(int(jmax_override), ns)))
    dtype = ard.dtype

    arm = np.asarray(arm, dtype=dtype)
    brm = np.asarray(brm, dtype=dtype)
    azm = np.asarray(azm, dtype=dtype)
    bzm = np.asarray(bzm, dtype=dtype)
    brd = np.asarray(brd, dtype=dtype)
    azd = np.asarray(azd, dtype=dtype)
    bzd = np.asarray(bzd, dtype=dtype)
    cxd = np.asarray(cxd, dtype=dtype)
    delta_s = np.asarray(delta_s, dtype=dtype)

    mpol = int(cfg.mpol)
    nrange = int(cfg.ntor) + 1
    nfp = float(cfg.nfp)
    m = np.arange(mpol, dtype=dtype)
    n = np.arange(nrange, dtype=dtype)
    m2 = (m * m)[None, :, None]
    n2 = ((n * nfp) ** 2)[None, None, :]
    m_par = (np.arange(mpol) % 2).astype(np.int32)

    ns_half = int(arm.shape[0])
    pad_rows = max(ns_f - ns_half, 0)
    if pad_rows > 0:
        z_arm = np.zeros((pad_rows, arm.shape[1]), dtype=dtype)
        z_azm = np.zeros((pad_rows, azm.shape[1]), dtype=dtype)
        arm_f = np.concatenate([arm, z_arm], axis=0)
        brm_f = np.concatenate([brm, z_arm], axis=0)
        azm_f = np.concatenate([azm, z_azm], axis=0)
        bzm_f = np.concatenate([bzm, z_azm], axis=0)
    else:
        arm_f = arm[:ns_f]
        brm_f = brm[:ns_f]
        azm_f = azm[:ns_f]
        bzm_f = bzm[:ns_f]
    ard_f = ard[:ns_f]
    brd_f = brd[:ns_f]
    azd_f = azd[:ns_f]
    bzd_f = bzd[:ns_f]
    cxd_f = cxd[:ns_f]

    arm_m = arm_f[:, m_par]
    brm_m = brm_f[:, m_par]
    ard_m = ard_f[:, m_par]
    brd_m = brd_f[:, m_par]
    azm_m = azm_f[:, m_par]
    bzm_m = bzm_f[:, m_par]
    azd_m = azd_f[:, m_par]
    bzd_m = bzd_f[:, m_par]

    ar = -(arm_m[:, :, None] + brm_m[:, :, None] * m2)
    az = -(azm_m[:, :, None] + bzm_m[:, :, None] * m2)
    dr = -(ard_m[:, :, None] + brd_m[:, :, None] * m2 + cxd_f[:, None, None] * n2)
    dz = -(azd_m[:, :, None] + bzd_m[:, :, None] * m2 + cxd_f[:, None, None] * n2)

    br = np.zeros_like(ar)
    bz = np.zeros_like(az)
    if ns_f > 1:
        br[1:] = -(arm_m[:-1, :, None] + brm_m[:-1, :, None] * m2)
        bz[1:] = -(azm_m[:-1, :, None] + bzm_m[:-1, :, None] * m2)

    if ns_f > 0 and mpol > 1:
        ar[0, 1:, :] = 0.0
        az[0, 1:, :] = 0.0
        dr[0, 1:, :] = 0.0
        dz[0, 1:, :] = 0.0

    if ns_f > 1 and mpol > 1:
        dr[1, 1, :] += br[1, 1, :]
        dz[1, 1, :] += bz[1, 1, :]

    if ns_f >= ns and ns > 0:
        edge_pedestal = np.asarray(0.05, dtype=dtype)
        fac = np.asarray(0.25, dtype=dtype)
        hs = np.asarray(delta_s, dtype=dtype)
        mult_fac = np.minimum(fac, fac * hs * np.asarray(15.0, dtype=dtype))
        edge_idx = ns - 1
        if mpol > 0:
            dr[edge_idx, 0:1, :] *= 1.0 + edge_pedestal
            dz[edge_idx, 0:1, :] *= 1.0 + edge_pedestal
        if mpol > 1:
            dr[edge_idx, 1:2, :] *= 1.0 + edge_pedestal
            dz[edge_idx, 1:2, :] *= 1.0 + edge_pedestal
        if mpol > 2:
            dr[edge_idx, 2:, :] *= 1.0 + 2.0 * edge_pedestal
            dz[edge_idx, 2:, :] *= 1.0 + 2.0 * edge_pedestal
        if mpol > 0 and nrange > 0:
            dz[edge_idx, 0, 0] *= (1.0 - mult_fac) / (1.0 + edge_pedestal)

    jmin_m = np.where(np.arange(mpol) > 0, 1, 0).astype(np.int32)
    jmin = jmin_m[:, None] * np.ones((mpol, nrange), dtype=np.int32)

    mats = {
        "ar": ar,
        "br": br,
        "dr": dr,
        "az": az,
        "bz": bz,
        "dz": dz,
        "arm_parity": arm,
        "ard_parity": ard,
        "brm_parity": brm,
        "brd_parity": brd,
        "azm_parity": azm,
        "azd_parity": azd,
        "bzm_parity": bzm,
        "bzd_parity": bzd,
        "cxd_full": cxd,
        "delta_s": delta_s,
    }
    return mats, jmin, int(ns_f)


def rz_preconditioner_matrices_numpy_host(
    *,
    bc,
    k,
    trig,
    s,
    cfg,
    jmax_override: int | None = None,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
) -> tuple[dict[str, Any], Any, int]:
    """Return R/Z matrices with NumPy for concrete CPU host solves.

    The helper is only a performance mirror of the promoted JAX implementation.
    It is not used for traced/autodiff or accelerator paths.
    """

    del trig
    use_precomputed, use_lax_tridi = _resolve_tridi_flags(
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )
    if use_precomputed or use_lax_tridi:
        return rz_preconditioner_matrices(
            bc=bc,
            k=k,
            trig=None,
            s=s,
            cfg=cfg,
            jmax_override=jmax_override,
            use_precomputed=use_precomputed,
            use_lax_tridi=use_lax_tridi,
        )

    from vmec_jax.preconditioner_1d import (
        _compute_preconditioning_matrix as _compute_preconditioning_matrix_np,
        _sm_sp_from_profiles as _sm_sp_from_profiles_np,
        _sqrt_profiles_from_ns as _sqrt_profiles_from_ns_np,
        wint_from_config as _wint_from_config_np,
    )

    s_arr = np.asarray(s)
    ns = int(s_arr.shape[0])
    dtype = np.asarray(bc.guu).dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    w_int = np.asarray(_wint_from_config_np(cfg=cfg), dtype=dtype)

    r12 = np.asarray(bc.jac.r12, dtype=dtype)[1:]
    tau = np.asarray(bc.jac.tau, dtype=dtype)[1:]
    total_pressure = np.asarray(bc.bsq, dtype=dtype)[1:]
    bsupv = np.asarray(bc.bsupv, dtype=dtype)[1:]
    sqrtg = np.asarray(bc.jac.sqrtg, dtype=dtype)[1:]

    sqrt_sf, sqrt_sh = _sqrt_profiles_from_ns_np(ns)
    sqrt_sh = np.asarray(sqrt_sh, dtype=dtype)
    sm, sp = _sm_sp_from_profiles_np(np.asarray(sqrt_sf, dtype=dtype), sqrt_sh)
    sm = np.asarray(sm, dtype=dtype)
    sp = np.asarray(sp, dtype=dtype)
    delta_s = np.asarray(s_arr[1] - s_arr[0] if ns >= 2 else 1.0, dtype=dtype)

    arm, ard, brm, brd, cxd = _compute_preconditioning_matrix_np(
        xs=np.asarray(bc.jac.zs, dtype=dtype)[1:],
        xu12=np.asarray(bc.jac.zu12, dtype=dtype)[1:],
        xu_e=np.asarray(k.pzu_even, dtype=dtype),
        xu_o=np.asarray(k.pzu_odd, dtype=dtype),
        x1_o=np.asarray(k.pz1_odd, dtype=dtype),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=sqrtg,
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns,
    )
    azm, azd, bzm, bzd, _cxd_z = _compute_preconditioning_matrix_np(
        xs=np.asarray(bc.jac.rs, dtype=dtype)[1:],
        xu12=np.asarray(bc.jac.ru12, dtype=dtype)[1:],
        xu_e=np.asarray(k.pru_even, dtype=dtype),
        xu_o=np.asarray(k.pru_odd, dtype=dtype),
        x1_o=np.asarray(k.pr1_odd, dtype=dtype),
        r12=r12,
        total_pressure=total_pressure,
        tau=tau,
        bsupv=bsupv,
        sqrtg=sqrtg,
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=delta_s,
        ns_full=ns,
    )
    return _assemble_rz_preconditioner_matrices_numpy_host(
        arm=arm,
        ard=ard,
        brm=brm,
        brd=brd,
        azm=azm,
        azd=azd,
        bzm=bzm,
        bzd=bzd,
        cxd=cxd,
        delta_s=delta_s,
        cfg=cfg,
        jmax_override=jmax_override,
    )


def rz_preconditioner_matrices_reassemble(
    *,
    mats: dict[str, Any],
    cfg,
    jmax_override: int | None = None,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
) -> tuple[dict[str, Any], Any, int]:
    """Reassemble R/Z matrices from cached parity coefficients for a new jmax."""
    required = (
        "arm_parity",
        "ard_parity",
        "brm_parity",
        "brd_parity",
        "azm_parity",
        "azd_parity",
        "bzm_parity",
        "bzd_parity",
        "cxd_full",
        "delta_s",
    )
    missing = [key for key in required if key not in mats]
    if missing:
        raise KeyError(f"Missing cached preconditioner coefficients: {', '.join(missing)}")
    return _assemble_rz_preconditioner_matrices_impl(
        arm=mats["arm_parity"],
        ard=mats["ard_parity"],
        brm=mats["brm_parity"],
        brd=mats["brd_parity"],
        azm=mats["azm_parity"],
        azd=mats["azd_parity"],
        bzm=mats["bzm_parity"],
        bzd=mats["bzd_parity"],
        cxd=mats["cxd_full"],
        delta_s=mats["delta_s"],
        cfg=cfg,
        jmax_override=jmax_override,
        use_precomputed=use_precomputed,
        use_lax_tridi=use_lax_tridi,
    )


@partial(jit, static_argnames=("use_lax_tridi",))
def _tridi_solve_batched_jmin0(a, d, b, rhs, *, use_lax_tridi: bool = False) -> Any:
    """Batched Thomas solve for a/d/b with jmin=0 (axisymmetric)."""
    a = jnp.asarray(a)
    d = jnp.asarray(d)
    b = jnp.asarray(b)
    rhs = jnp.asarray(rhs)
    if rhs.ndim > d.ndim:
        expand = (1,) * (rhs.ndim - d.ndim)
        a = a.reshape(a.shape + expand)
        d = d.reshape(d.shape + expand)
        b = b.reshape(b.shape + expand)
    n = int(rhs.shape[0])
    if n == 0:
        return rhs

    # Prefer XLA's fused tridiagonal solver when explicitly enabled.
    if bool(use_lax_tridi) and n >= 3:
        # Map to tridiagonal_solve conventions: dl=subdiagonal, du=superdiagonal.
        # Our Thomas implementation treats `a` as the superdiagonal and `b` as
        # the subdiagonal, so swap here to preserve parity.
        dl = b
        du = a
        if dl.shape != d.shape:
            dl = jnp.broadcast_to(dl, d.shape)
        if du.shape != d.shape:
            du = jnp.broadcast_to(du, d.shape)
        dl = dl.at[0].set(jnp.asarray(0.0, dtype=dl.dtype))
        du = du.at[-1].set(jnp.asarray(0.0, dtype=du.dtype))
        rhs_in = rhs
        squeeze_rhs = False
        if rhs_in.ndim == d.ndim - 1:
            rhs_in = rhs_in[..., None]
            squeeze_rhs = True
        target_shape = rhs_in.shape
        pad_dims = d.ndim - (rhs_in.ndim - 1)
        if pad_dims > 0:
            rhs_in = rhs_in.reshape(rhs_in.shape[:-1] + (1,) * pad_dims + (rhs_in.shape[-1],))
        if rhs_in.shape[:-1] != d.shape:
            rhs_in = jnp.broadcast_to(rhs_in, d.shape + (rhs_in.shape[-1],))
        # tridiagonal_solve expects the system dimension on the last axis for
        # dl/d/du/d, and on the second-to-last axis for rhs.
        dl_t = jnp.moveaxis(dl, 0, -1)
        d_t = jnp.moveaxis(d, 0, -1)
        du_t = jnp.moveaxis(du, 0, -1)
        rhs_t = jnp.moveaxis(rhs_in, 0, -2)
        sol_t = jax.lax.linalg.tridiagonal_solve(dl_t, d_t, du_t, rhs_t)
        sol = jnp.moveaxis(sol_t, -2, 0)
        if sol.shape != target_shape:
            sol = sol.reshape(target_shape)
        return sol[..., 0] if squeeze_rhs else sol

    eps = jnp.asarray(1.0e-12, dtype=rhs.dtype)
    d0 = jnp.where(d[0] != 0.0, d[0], eps)
    a0 = a[0] / d0
    x0 = rhs[0] / d0

    def fwd(carry, inp):
        """Run the forward rule for the custom derivative."""
        a_prev, x_prev = carry
        aj, dj, bj, rj = inp
        denom = dj - a_prev * bj
        denom = jnp.where(denom != 0.0, denom, eps)
        a_new = aj / denom
        x_new = (rj - x_prev * bj) / denom
        return (a_new, x_new), (a_new, x_new)

    if n == 1:
        a_norm = a0[None, ...]
        x = x0[None, ...]
    else:
        inp = (a[1:], d[1:], b[1:], rhs[1:])
        (_, _), (a_rest, x_rest) = jax.lax.scan(fwd, (a0, x0), inp)
        a_norm = jnp.concatenate([a0[None, ...], a_rest], axis=0)
        x = jnp.concatenate([x0[None, ...], x_rest], axis=0)

    def bwd(carry, inp):
        """Run the transpose rule for the custom derivative."""
        x_next = carry
        a_j, x_j = inp
        x_new = x_j - a_j * x_next
        return x_new, x_new

    if n <= 1:
        return x
    # Backward substitution for indices n-2..0 (a_norm[n-1] unused).
    x_last = x[-1]
    inp_b = (a_norm[:-1], x[:-1])
    _, x_rev = jax.lax.scan(bwd, x_last, inp_b, reverse=True)
    x_out = jnp.concatenate([x_rev, x_last[None, ...]], axis=0)
    return x_out


@jit
def _tridi_precompute_coeffs(a, d, b) -> tuple[Any, Any]:
    """Precompute Thomas coefficients (c', inv_denom) for fixed tridiagonal."""
    a = jnp.asarray(a)
    d = jnp.asarray(d)
    b = jnp.asarray(b)
    n = int(d.shape[0])
    if n == 0:
        return a, d
    eps = jnp.asarray(1.0e-12, dtype=d.dtype)
    d0 = jnp.where(d[0] != 0.0, d[0], eps)
    inv0 = jnp.asarray(1.0, dtype=d.dtype) / d0
    cp0 = a[0] * inv0

    def fwd(cp_prev, inp):
        """Run the forward rule for the custom derivative."""
        aj, dj, bj = inp
        denom = dj - cp_prev * bj
        denom = jnp.where(denom != 0.0, denom, eps)
        inv = jnp.asarray(1.0, dtype=d.dtype) / denom
        cp = aj * inv
        return cp, (cp, inv)

    if n == 1:
        cp = cp0[None, ...]
        inv = inv0[None, ...]
    else:
        inp = (a[1:], d[1:], b[1:])
        _, (cp_rest, inv_rest) = jax.lax.scan(fwd, cp0, inp)
        cp = jnp.concatenate([cp0[None, ...], cp_rest], axis=0)
        inv = jnp.concatenate([inv0[None, ...], inv_rest], axis=0)
    return cp, inv


@jit
def _tridi_pretranspose_for_lax(a, d, b) -> tuple[Any, Any, Any]:
    """Pretranspose tridiagonal system for lax.tridiagonal_solve.

    Returns (dl_t, d_t, du_t) with system dimension on the last axis.
    """
    a = jnp.asarray(a)
    d = jnp.asarray(d)
    b = jnp.asarray(b)
    if a.shape != d.shape:
        a = jnp.broadcast_to(a, d.shape)
    if b.shape != d.shape:
        b = jnp.broadcast_to(b, d.shape)
    dl_t = jnp.moveaxis(b, 0, -1)
    d_t = jnp.moveaxis(d, 0, -1)
    du_t = jnp.moveaxis(a, 0, -1)
    dl_t = dl_t.at[..., 0].set(jnp.asarray(0.0, dtype=dl_t.dtype))
    du_t = du_t.at[..., -1].set(jnp.asarray(0.0, dtype=du_t.dtype))
    return dl_t, d_t, du_t


@jit
def _tridi_solve_batched_jmin0_lax_pretransposed(dl_t, d_t, du_t, rhs) -> Any:
    """Batched tridiagonal solve using pretransposed dl/d/du (system dim last)."""
    dl_t = jnp.asarray(dl_t)
    d_t = jnp.asarray(d_t)
    du_t = jnp.asarray(du_t)
    # Guard against zero diagonal entries (Thomas solver uses eps guards).
    eps = jnp.asarray(1.0e-12, dtype=d_t.dtype)
    d_t = jnp.where(d_t != 0.0, d_t, eps)
    rhs_in = jnp.asarray(rhs)
    expected_prefix = (d_t.shape[-1],) + d_t.shape[:-1]
    squeeze_rhs = rhs_in.ndim <= d_t.ndim
    if squeeze_rhs:
        pad_dims = d_t.ndim - rhs_in.ndim
        rhs_in = rhs_in.reshape(rhs_in.shape + (1,) * pad_dims + (1,))
    if rhs_in.shape[: d_t.ndim] != expected_prefix:
        rhs_in = jnp.broadcast_to(rhs_in, expected_prefix + rhs_in.shape[d_t.ndim :])
    rhs_shape = rhs_in.shape
    nrhs = functools.reduce(lambda x, y: x * y, rhs_shape[d_t.ndim :], 1)
    rhs_in = rhs_in.reshape(expected_prefix + (nrhs,))
    # rhs expects system dimension on second-to-last axis.
    rhs_t = jnp.moveaxis(rhs_in, 0, -2)
    sol_t = jax.lax.linalg.tridiagonal_solve(dl_t, d_t, du_t, rhs_t)
    sol = jnp.moveaxis(sol_t, -2, 0)
    sol = sol.reshape(rhs_shape)
    return sol[..., 0] if squeeze_rhs else sol


@jit
def _tridi_solve_precomputed(b, cp, inv, rhs) -> Any:
    """Solve tridiagonal using precomputed c' and inv_denom."""
    b = jnp.asarray(b)
    cp = jnp.asarray(cp)
    inv = jnp.asarray(inv)
    rhs = jnp.asarray(rhs)
    if rhs.ndim > inv.ndim:
        expand = (1,) * (rhs.ndim - inv.ndim)
        b = b.reshape(b.shape + expand)
        cp = cp.reshape(cp.shape + expand)
        inv = inv.reshape(inv.shape + expand)
    n = int(rhs.shape[0])
    if n == 0:
        return rhs

    dp0 = rhs[0] * inv[0]

    def fwd(dp_prev, inp):
        """Run the forward rule for the custom derivative."""
        bj, invj, rj = inp
        dp = (rj - bj * dp_prev) * invj
        return dp, dp

    if n == 1:
        dp = dp0[None, ...]
    else:
        inp = (b[1:], inv[1:], rhs[1:])
        _, dp_rest = jax.lax.scan(fwd, dp0, inp)
        dp = jnp.concatenate([dp0[None, ...], dp_rest], axis=0)

    def bwd(x_next, inp):
        """Run the transpose rule for the custom derivative."""
        cpj, dpj = inp
        x_new = dpj - cpj * x_next
        return x_new, x_new

    if n <= 1:
        return dp
    x_last = dp[-1]
    inp_b = (cp[:-1], dp[:-1])
    _, x_rev = jax.lax.scan(bwd, x_last, inp_b, reverse=True)
    x_out = jnp.concatenate([x_rev, x_last[None, ...]], axis=0)
    return x_out


@partial(
    jit,
    static_argnames=(
        "jmax",
        "use_precomputed",
        "use_lax_tridi",
        "use_rss",
        "use_rsc",
        "use_rcs",
        "use_zcs",
        "use_zcc",
        "use_zss",
    ),
)
def _rz_preconditioner_apply_arrays(
    *,
    ar,
    br,
    dr,
    cr,
    ir,
    az,
    bz,
    dz,
    cz,
    iz,
    dlr_t,
    dr_t,
    dur_t,
    dlz_t,
    dz_t,
    duz_t,
    frcc,
    frss,
    fzsc,
    fzcs,
    frsc,
    frcs,
    fzcc,
    fzss,
    jmax: int,
    use_precomputed: bool,
    use_lax_tridi: bool,
    use_rss: bool,
    use_rsc: bool,
    use_rcs: bool,
    use_zcs: bool,
    use_zcc: bool,
    use_zss: bool,
):
    # Ensure JAX .at[] support — numpy arrays pass through JIT boundary when JIT
    # is active, but when JIT is disabled (e.g. jax_disable_jit=True in tests)
    # they stay as plain numpy and .at[].set() would raise AttributeError.
    frcc_u = jnp.asarray(frcc)
    frss_u = jnp.asarray(frss)
    fzsc_u = jnp.asarray(fzsc)
    fzcs_u = jnp.asarray(fzcs)
    frsc_u = jnp.asarray(frsc)
    frcs_u = jnp.asarray(frcs)
    fzcc_u = jnp.asarray(fzcc)
    fzss_u = jnp.asarray(fzss)
    jmax = int(jmax)
    mpol = int(frcc.shape[1])
    nrange = int(frcc.shape[2])
    if jmax > 0:
        rhs_r = frcc[:jmax]
        rhs_z = fzsc[:jmax]
        rhs_rs = frss[:jmax]
        rhs_zc = fzcs[:jmax]
        rhs_rsc = frsc[:jmax]
        rhs_rcs = frcs[:jmax]
        rhs_zcc = fzcc[:jmax]
        rhs_zss = fzss[:jmax]

        r_blocks = [rhs_r]
        if use_rss:
            r_blocks.append(rhs_rs)
        if use_rsc:
            r_blocks.append(rhs_rsc)
        if use_rcs:
            r_blocks.append(rhs_rcs)
        rhs_r_stack = jnp.stack(r_blocks, axis=-1)

        z_blocks = [rhs_z]
        if use_zcs:
            z_blocks.append(rhs_zc)
        if use_zcc:
            z_blocks.append(rhs_zcc)
        if use_zss:
            z_blocks.append(rhs_zss)
        rhs_z_stack = jnp.stack(z_blocks, axis=-1)

        rhs_r0_stack = rhs_r_stack[:, 0, :]
        rhs_z0_stack = rhs_z_stack[:, 0, :]
        sol_rm_full = None
        sol_zm_full = None
        combined_precomputed = bool(use_precomputed) and (not bool(use_lax_tridi)) and mpol > 1 and jmax > 1
        if combined_precomputed:
            # Combine the m=0 and m>0 systems into one padded Thomas solve.
            # The m>0 radial block starts at j=1 in VMEC; row 0 is a harmless
            # identity equation with zero RHS, so the recurrence matches the
            # shorter j=1..jmax-1 solve but uses one accelerator scan.
            b0 = jnp.stack([br[:, 0, :], bz[:, 0, :]], axis=1)
            c0 = jnp.stack([cr[:, 0, :], cz[:, 0, :]], axis=1)
            i0 = jnp.stack([ir[:, 0, :], iz[:, 0, :]], axis=1)
            rhs0 = jnp.stack([rhs_r0_stack, rhs_z0_stack], axis=1)
            b0 = jnp.broadcast_to(b0, (jmax, 2, nrange))
            c0 = jnp.broadcast_to(c0, (jmax, 2, nrange))
            i0 = jnp.broadcast_to(i0, (jmax, 2, nrange))

            bm = jnp.stack([br[1:jmax, 1:, :], bz[1:jmax, 1:, :]], axis=1)
            cm = jnp.stack([cr[1:jmax, 1:, :], cz[1:jmax, 1:, :]], axis=1)
            im = jnp.stack([ir[1:jmax, 1:, :], iz[1:jmax, 1:, :]], axis=1)
            rhsm = jnp.stack([rhs_r_stack[1:, 1:, :], rhs_z_stack[1:, 1:, :]], axis=1)
            bm = jnp.broadcast_to(bm, (jmax - 1, 2, mpol - 1, nrange))
            cm = jnp.broadcast_to(cm, (jmax - 1, 2, mpol - 1, nrange))
            im = jnp.broadcast_to(im, (jmax - 1, 2, mpol - 1, nrange))

            pad_coeff = jnp.zeros((1, 2, mpol - 1, nrange), dtype=bm.dtype)
            pad_inv = jnp.ones((1, 2, mpol - 1, nrange), dtype=im.dtype)
            pad_rhs = jnp.zeros((1, 2, mpol - 1, nrange, rhs0.shape[-1]), dtype=rhsm.dtype)

            b_all = jnp.concatenate([b0[:, :, None, :], jnp.concatenate([pad_coeff, bm], axis=0)], axis=2)
            c_all = jnp.concatenate([c0[:, :, None, :], jnp.concatenate([pad_coeff, cm], axis=0)], axis=2)
            i_all = jnp.concatenate([i0[:, :, None, :], jnp.concatenate([pad_inv, im], axis=0)], axis=2)
            rhs_all = jnp.concatenate([rhs0[:, :, None, :, :], jnp.concatenate([pad_rhs, rhsm], axis=0)], axis=2)

            sol_all = _tridi_solve_precomputed(b_all, c_all, i_all, rhs_all)
            sol_r0_stack = sol_all[:, 0, 0, :, :]
            sol_z0_stack = sol_all[:, 1, 0, :, :]
            sol_rm_full = sol_all[:, 0, 1:, :, :]
            sol_zm_full = sol_all[:, 1, 1:, :, :]
        elif use_lax_tridi and (dlr_t is not None) and (dr_t is not None) and (dur_t is not None):
            dlr0 = dlr_t[0, :, :jmax]
            dr0 = dr_t[0, :, :jmax]
            dur0 = dur_t[0, :, :jmax]
            sol_r0_stack = _tridi_solve_batched_jmin0_lax_pretransposed(dlr0, dr0, dur0, rhs_r0_stack)
        else:
            sol_r0_stack = None
        if (not combined_precomputed) and use_lax_tridi and (dlz_t is not None) and (dz_t is not None) and (duz_t is not None):
            dlz0 = dlz_t[0, :, :jmax]
            dz0 = dz_t[0, :, :jmax]
            duz0 = duz_t[0, :, :jmax]
            sol_z0_stack = _tridi_solve_batched_jmin0_lax_pretransposed(dlz0, dz0, duz0, rhs_z0_stack)
        elif not combined_precomputed:
            sol_z0_stack = None

        if bool(use_precomputed) and (not bool(use_lax_tridi)) and (not combined_precomputed):
            # R and Z use independent tridiagonal coefficients but identical
            # radial lengths and block counts. Batch them through one Thomas
            # forward/backward scan to reduce nested scan work on accelerators.
            b0 = jnp.stack([br[:, 0, :], bz[:, 0, :]], axis=1)
            c0 = jnp.stack([cr[:, 0, :], cz[:, 0, :]], axis=1)
            i0 = jnp.stack([ir[:, 0, :], iz[:, 0, :]], axis=1)
            rhs0 = jnp.stack([rhs_r0_stack, rhs_z0_stack], axis=1)
            sol0 = _tridi_solve_precomputed(b0, c0, i0, rhs0)
            sol_r0_stack = sol0[:, 0, ...]
            sol_z0_stack = sol0[:, 1, ...]
        elif sol_r0_stack is None:
            sol_r0_stack = _tridi_solve_batched_jmin0(
                ar[:, 0, :], dr[:, 0, :], br[:, 0, :], rhs_r0_stack, use_lax_tridi=use_lax_tridi
            )
        if sol_z0_stack is None:
            sol_z0_stack = _tridi_solve_batched_jmin0(
                az[:, 0, :], dz[:, 0, :], bz[:, 0, :], rhs_z0_stack, use_lax_tridi=use_lax_tridi
            )
        idx = 0
        frcc_u = frcc_u.at[:jmax, 0, :].set(sol_r0_stack[..., idx])
        idx += 1
        if use_rss:
            frss_u = frss_u.at[:jmax, 0, :].set(sol_r0_stack[..., idx])
            idx += 1
        if use_rsc:
            frsc_u = frsc_u.at[:jmax, 0, :].set(sol_r0_stack[..., idx])
            idx += 1
        if use_rcs:
            frcs_u = frcs_u.at[:jmax, 0, :].set(sol_r0_stack[..., idx])
            idx += 1

        idx = 0
        fzsc_u = fzsc_u.at[:jmax, 0, :].set(sol_z0_stack[..., idx])
        idx += 1
        if use_zcs:
            fzcs_u = fzcs_u.at[:jmax, 0, :].set(sol_z0_stack[..., idx])
            idx += 1
        if use_zcc:
            fzcc_u = fzcc_u.at[:jmax, 0, :].set(sol_z0_stack[..., idx])
            idx += 1
        if use_zss:
            fzss_u = fzss_u.at[:jmax, 0, :].set(sol_z0_stack[..., idx])
            idx += 1

        if mpol > 1 and jmax > 1:
            a_r = ar[1:jmax, 1:, :]
            d_r = dr[1:jmax, 1:, :]
            b_r = br[1:jmax, 1:, :]
            a_z = az[1:jmax, 1:, :]
            d_z = dz[1:jmax, 1:, :]
            b_z = bz[1:jmax, 1:, :]

            rhs_rm_stack = rhs_r_stack[1:, 1:, :]
            if sol_rm_full is not None:
                sol_rm = sol_rm_full[1:, ...]
            elif use_lax_tridi and (dlr_t is not None) and (dr_t is not None) and (dur_t is not None):
                dlr_m = dlr_t[1:, :, : jmax - 1]
                dr_m = dr_t[1:, :, : jmax - 1]
                dur_m = dur_t[1:, :, : jmax - 1]
                sol_rm = _tridi_solve_batched_jmin0_lax_pretransposed(dlr_m, dr_m, dur_m, rhs_rm_stack)
            else:
                sol_rm = None

            rhs_zm_stack = rhs_z_stack[1:, 1:, :]
            if sol_zm_full is not None:
                sol_zm = sol_zm_full[1:, ...]
            elif use_lax_tridi and (dlz_t is not None) and (dz_t is not None) and (duz_t is not None):
                dlz_m = dlz_t[1:, :, : jmax - 1]
                dz_m = dz_t[1:, :, : jmax - 1]
                duz_m = duz_t[1:, :, : jmax - 1]
                sol_zm = _tridi_solve_batched_jmin0_lax_pretransposed(dlz_m, dz_m, duz_m, rhs_zm_stack)
            else:
                sol_zm = None

            if (sol_rm_full is None) and bool(use_precomputed) and (not bool(use_lax_tridi)):
                bm = jnp.stack([b_r, b_z], axis=1)
                cm = jnp.stack([cr[1:jmax, 1:, :], cz[1:jmax, 1:, :]], axis=1)
                im = jnp.stack([ir[1:jmax, 1:, :], iz[1:jmax, 1:, :]], axis=1)
                rhsm = jnp.stack([rhs_rm_stack, rhs_zm_stack], axis=1)
                solm = _tridi_solve_precomputed(bm, cm, im, rhsm)
                sol_rm = solm[:, 0, ...]
                sol_zm = solm[:, 1, ...]
            elif sol_rm is None:
                sol_rm = _tridi_solve_batched_jmin0(a_r, d_r, b_r, rhs_rm_stack, use_lax_tridi=use_lax_tridi)
            if sol_zm is None:
                sol_zm = _tridi_solve_batched_jmin0(a_z, d_z, b_z, rhs_zm_stack, use_lax_tridi=use_lax_tridi)

            pad_r = jnp.zeros((1, mpol - 1, nrange, sol_rm.shape[-1]), dtype=sol_rm.dtype)
            sol_rm_full = jnp.concatenate([pad_r, sol_rm], axis=0)
            pad_z = jnp.zeros((1, mpol - 1, nrange, sol_zm.shape[-1]), dtype=sol_zm.dtype)
            sol_zm_full = jnp.concatenate([pad_z, sol_zm], axis=0)
            idx = 0
            frcc_u = frcc_u.at[:jmax, 1:, :].set(sol_rm_full[..., idx])
            idx += 1
            if use_rss:
                frss_u = frss_u.at[:jmax, 1:, :].set(sol_rm_full[..., idx])
                idx += 1
            if use_rsc:
                frsc_u = frsc_u.at[:jmax, 1:, :].set(sol_rm_full[..., idx])
                idx += 1
            if use_rcs:
                frcs_u = frcs_u.at[:jmax, 1:, :].set(sol_rm_full[..., idx])
                idx += 1

            idx = 0
            fzsc_u = fzsc_u.at[:jmax, 1:, :].set(sol_zm_full[..., idx])
            idx += 1
            if use_zcs:
                fzcs_u = fzcs_u.at[:jmax, 1:, :].set(sol_zm_full[..., idx])
                idx += 1
            if use_zcc:
                fzcc_u = fzcc_u.at[:jmax, 1:, :].set(sol_zm_full[..., idx])
                idx += 1
            if use_zss:
                fzss_u = fzss_u.at[:jmax, 1:, :].set(sol_zm_full[..., idx])
                idx += 1

    return frcc_u, frss_u, fzsc_u, fzcs_u, frsc_u, frcs_u, fzcc_u, fzss_u


def rz_preconditioner_apply(
    *,
    frzl_in: TomnspsRZL,
    mats: dict[str, Any],
    jmax: int,
    cfg,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
) -> TomnspsRZL:
    """Apply cached VMEC R/Z preconditioner matrices.

    This is the matrix-application half of :func:`rz_preconditioner`,
    split out so callers (e.g. VMEC2000-style cached preconditioners) can
    reuse matrices across iterations without recomputing them.
    """
    ar = mats["ar"]
    br = mats["br"]
    dr = mats["dr"]
    cr = mats.get("cr", ar)
    ir = mats.get("ir", dr)
    az = mats["az"]
    bz = mats["bz"]
    dz = mats["dz"]
    cz = mats.get("cz", az)
    iz = mats.get("iz", dz)
    dlr_t = mats.get("dlr_t", None)
    dr_t = mats.get("dr_t", None)
    dur_t = mats.get("dur_t", None)
    dlz_t = mats.get("dlz_t", None)
    dz_t = mats.get("dz_t", None)
    duz_t = mats.get("duz_t", None)

    frcc = jnp.asarray(frzl_in.frcc)
    fzsc = jnp.asarray(frzl_in.fzsc)
    frss = jnp.asarray(frzl_in.frss) if frzl_in.frss is not None else jnp.zeros_like(frcc)
    fzcs = jnp.asarray(frzl_in.fzcs) if frzl_in.fzcs is not None else jnp.zeros_like(fzsc)
    frsc = jnp.asarray(getattr(frzl_in, "frsc", None)) if getattr(frzl_in, "frsc", None) is not None else jnp.zeros_like(frcc)
    frcs = jnp.asarray(getattr(frzl_in, "frcs", None)) if getattr(frzl_in, "frcs", None) is not None else jnp.zeros_like(frcc)
    fzcc = jnp.asarray(getattr(frzl_in, "fzcc", None)) if getattr(frzl_in, "fzcc", None) is not None else jnp.zeros_like(fzsc)
    fzss = jnp.asarray(getattr(frzl_in, "fzss", None)) if getattr(frzl_in, "fzss", None) is not None else jnp.zeros_like(fzsc)

    lthreed = bool(getattr(cfg, "lthreed", False))
    lasym = bool(getattr(cfg, "lasym", False))
    use_rss = bool(lthreed)
    use_rsc = bool(lasym)
    use_rcs = bool(lthreed and lasym)
    use_zcs = bool(lthreed)
    use_zcc = bool(lasym)
    use_zss = bool(lthreed and lasym)

    # NOTE: precomputed Thomas coefficients have shown parity drift in some
    # cases; default to the direct solver unless explicitly enabled.
    if use_precomputed is None:
        use_precomputed = os.getenv("VMEC_JAX_TRIDI_PRECOMPUTE", "0").strip().lower() not in (
            "",
            "0",
            "false",
            "no",
        )
    if ("cr" not in mats) or ("ir" not in mats) or ("cz" not in mats) or ("iz" not in mats):
        use_precomputed = False
    if use_lax_tridi is None:
        use_lax_tridi = os.getenv("VMEC_JAX_TRIDI_SOLVE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "lax",
            "force",
        )

    frcc_u, frss_u, fzsc_u, fzcs_u, frsc_u, frcs_u, fzcc_u, fzss_u = _rz_preconditioner_apply_arrays(
        ar=ar,
        br=br,
        dr=dr,
        cr=cr,
        ir=ir,
        az=az,
        bz=bz,
        dz=dz,
        cz=cz,
        iz=iz,
        dlr_t=dlr_t,
        dr_t=dr_t,
        dur_t=dur_t,
        dlz_t=dlz_t,
        dz_t=dz_t,
        duz_t=duz_t,
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        frsc=frsc,
        frcs=frcs,
        fzcc=fzcc,
        fzss=fzss,
        jmax=int(jmax),
        use_precomputed=bool(use_precomputed),
        use_lax_tridi=bool(use_lax_tridi),
        use_rss=bool(use_rss),
        use_rsc=bool(use_rsc),
        use_rcs=bool(use_rcs),
        use_zcs=bool(use_zcs),
        use_zcc=bool(use_zcc),
        use_zss=bool(use_zss),
    )

    frss_out = frss_u if frzl_in.frss is not None else None
    fzcs_out = fzcs_u if frzl_in.fzcs is not None else None
    frsc_out = frsc_u if getattr(frzl_in, "frsc", None) is not None else None
    frcs_out = frcs_u if getattr(frzl_in, "frcs", None) is not None else None
    fzcc_out = fzcc_u if getattr(frzl_in, "fzcc", None) is not None else None
    fzss_out = fzss_u if getattr(frzl_in, "fzss", None) is not None else None

    return TomnspsRZL(
        frcc=frcc_u,
        frss=frss_out,
        fzsc=fzsc_u,
        fzcs=fzcs_out,
        flsc=frzl_in.flsc,
        flcs=frzl_in.flcs,
        frsc=frsc_out,
        frcs=frcs_out,
        fzcc=fzcc_out,
        fzss=fzss_out,
        flcc=getattr(frzl_in, "flcc", None),
        flss=getattr(frzl_in, "flss", None),
    )


@functools.lru_cache(maxsize=32)
def _make_rz_preconditioner_apply_jit(
    jmax: int,
    lthreed: bool,
    lasym: bool,
    use_precomputed: bool,
    use_lax_tridi: bool,
    has_frss: bool,
    has_fzcs: bool,
    has_frsc: bool,
    has_frcs: bool,
    has_fzcc: bool,
    has_fzss: bool,
    has_lax_t: bool,
):
    """Return a JIT-compiled inner apply function for the given static config.

    This avoids the ~237 eager JAX dispatches per iteration that occur when
    rz_preconditioner_apply is called without JIT.  The static booleans key
    the cache so each distinct configuration compiles only once.
    """
    use_rss = bool(lthreed)
    use_rsc = bool(lasym)
    use_rcs = bool(lthreed and lasym)
    use_zcs = bool(lthreed)
    use_zcc = bool(lasym)
    use_zss = bool(lthreed and lasym)

    @partial(
        jit,
        static_argnames=(),
    )
    def _apply_jit(
        frcc, fzsc, frss, fzcs, frsc, frcs, fzcc, fzss,
        ar, br, dr, cr, ir,
        az, bz, dz, cz, iz,
        dlr_t, dr_t, dur_t,
        dlz_t, dz_t, duz_t,
        flsc, flcs, flcc, flss,
    ):
        frcc_u, frss_u, fzsc_u, fzcs_u, frsc_u, frcs_u, fzcc_u, fzss_u = _rz_preconditioner_apply_arrays(
            ar=ar, br=br, dr=dr, cr=cr, ir=ir,
            az=az, bz=bz, dz=dz, cz=cz, iz=iz,
            dlr_t=dlr_t, dr_t=dr_t, dur_t=dur_t,
            dlz_t=dlz_t, dz_t=dz_t, duz_t=duz_t,
            frcc=frcc, frss=frss, fzsc=fzsc, fzcs=fzcs,
            frsc=frsc, frcs=frcs, fzcc=fzcc, fzss=fzss,
            jmax=jmax,
            use_precomputed=use_precomputed,
            use_lax_tridi=use_lax_tridi,
            use_rss=use_rss, use_rsc=use_rsc, use_rcs=use_rcs,
            use_zcs=use_zcs, use_zcc=use_zcc, use_zss=use_zss,
        )
        return TomnspsRZL(
            frcc=frcc_u,
            frss=frss_u if has_frss else None,
            fzsc=fzsc_u,
            fzcs=fzcs_u if has_fzcs else None,
            flsc=flsc,
            flcs=flcs,
            frsc=frsc_u if has_frsc else None,
            frcs=frcs_u if has_frcs else None,
            fzcc=fzcc_u if has_fzcc else None,
            fzss=fzss_u if has_fzss else None,
            flcc=flcc,
            flss=flss,
        )

    return _apply_jit


def rz_preconditioner_apply_jit(
    *,
    frzl_in: TomnspsRZL,
    mats: dict[str, Any],
    jmax: int,
    cfg,
    use_precomputed: bool | None = None,
    use_lax_tridi: bool | None = None,
) -> TomnspsRZL:
    """JIT-cached wrapper for :func:`rz_preconditioner_apply`.

    Resolves all Python-level static parameters once per distinct
    configuration and caches a JIT-compiled apply function.  On subsequent
    calls with the same configuration (same lthreed/lasym/flags/jmax) only
    the JIT-compiled XLA computation is dispatched, eliminating ~237 eager
    JAX ops per call.
    """
    # --- resolve static booleans (Python-level, outside JIT) ---
    lthreed = bool(getattr(cfg, "lthreed", False))
    lasym = bool(getattr(cfg, "lasym", False))

    if use_precomputed is None or use_lax_tridi is None:
        _env_pre, _env_lax = _get_env_tridi_flags()
        if use_precomputed is None:
            use_precomputed = _env_pre
        if use_lax_tridi is None:
            use_lax_tridi = _env_lax
    has_cr_ir = ("cr" in mats) and ("ir" in mats) and ("cz" in mats) and ("iz" in mats)
    if not has_cr_ir:
        use_precomputed = False

    # Structural None checks determine output shape (static)
    has_frss = frzl_in.frss is not None
    has_fzcs = frzl_in.fzcs is not None
    has_frsc = getattr(frzl_in, "frsc", None) is not None
    has_frcs = getattr(frzl_in, "frcs", None) is not None
    has_fzcc = getattr(frzl_in, "fzcc", None) is not None
    has_fzss = getattr(frzl_in, "fzss", None) is not None

    has_lax_t = (
        ("dlr_t" in mats) and ("dr_t" in mats) and ("dur_t" in mats)
        and ("dlz_t" in mats) and ("dz_t" in mats) and ("duz_t" in mats)
    )

    _apply_jit = _make_rz_preconditioner_apply_jit(
        jmax=int(jmax),
        lthreed=lthreed,
        lasym=lasym,
        use_precomputed=bool(use_precomputed),
        use_lax_tridi=bool(use_lax_tridi and has_lax_t),
        has_frss=has_frss,
        has_fzcs=has_fzcs,
        has_frsc=has_frsc,
        has_frcs=has_frcs,
        has_fzcc=has_fzcc,
        has_fzss=has_fzss,
        has_lax_t=has_lax_t,
    )

    # --- extract arrays (dict access outside JIT, then pass as flat args) ---
    ar = mats["ar"]
    br = mats["br"]
    dr = mats["dr"]
    cr = mats.get("cr", ar)
    ir = mats.get("ir", dr)
    az = mats["az"]
    bz = mats["bz"]
    dz = mats["dz"]
    cz = mats.get("cz", az)
    iz = mats.get("iz", dz)
    dlr_t = mats.get("dlr_t", ar)  # placeholder (unused if has_lax_t=False)
    dr_t = mats.get("dr_t", ar)
    dur_t = mats.get("dur_t", ar)
    dlz_t = mats.get("dlz_t", az)
    dz_t = mats.get("dz_t", az)
    duz_t = mats.get("duz_t", az)

    frcc = frzl_in.frcc
    fzsc = frzl_in.fzsc
    # Inactive optional blocks are compile-time unused by _apply_jit.  Reuse
    # existing operands as placeholders instead of allocating NumPy zeros here:
    # on accelerators those host placeholders still become call arguments and
    # add per-iteration host->device transfer overhead.
    frss = frzl_in.frss if has_frss else frcc
    fzcs = frzl_in.fzcs if has_fzcs else fzsc
    frsc = getattr(frzl_in, "frsc", None) if has_frsc else frcc
    frcs = getattr(frzl_in, "frcs", None) if has_frcs else frcc
    fzcc = getattr(frzl_in, "fzcc", None) if has_fzcc else fzsc
    fzss = getattr(frzl_in, "fzss", None) if has_fzss else fzsc
    flsc = frzl_in.flsc
    flcs = frzl_in.flcs
    flcc = getattr(frzl_in, "flcc", None)
    flss = getattr(frzl_in, "flss", None)

    return _apply_jit(
        frcc, fzsc, frss, fzcs, frsc, frcs, fzcc, fzss,
        ar, br, dr, cr, ir,
        az, bz, dz, cz, iz,
        dlr_t, dr_t, dur_t,
        dlz_t, dz_t, duz_t,
        flsc, flcs, flcc, flss,
    )


def rz_preconditioner(
    *,
    frzl_in: TomnspsRZL,
    bc,
    k,
    trig,
    s,
    cfg,
) -> TomnspsRZL:
    """Apply the VMEC R/Z radial preconditioner in JAX."""
    mats, _jmin, jmax = rz_preconditioner_matrices(bc=bc, k=k, trig=trig, s=s, cfg=cfg)
    # Use the JIT-cached wrapper: resolves static booleans once per config and
    # caches the compiled _apply_jit function via lru_cache.  This eliminates
    # ~237 eager JAX dispatches per call that occurred with the plain
    # rz_preconditioner_apply path, reducing cold-start overhead significantly
    # when the preconditioner is called multiple times during setup.
    return rz_preconditioner_apply_jit(frzl_in=frzl_in, mats=mats, jmax=jmax, cfg=cfg)


# ---------------------------------------------------------------------------
# Pure-NumPy preconditioner apply (avoids JAX JIT dispatch overhead)
# ---------------------------------------------------------------------------

import numpy as _np_prec  # local alias to avoid shadowing jnp


def _tridi_fwd_np(b, inv, rhs) -> _np_prec.ndarray:
    """Forward sweep of Thomas algorithm (precomputed form) in pure NumPy.

    Parameters
    ----------
    b   : (n, ...) subdiagonal coefficients
    inv : (n, ...) inverse of reduced diagonal
    rhs : (n, ...) right-hand side
    Returns dp array of same shape as rhs.
    """
    n = rhs.shape[0]
    if n == 0:
        return _np_prec.empty_like(rhs)
    if rhs.ndim > inv.ndim:
        extra = (1,) * (rhs.ndim - inv.ndim)
        b   = b.reshape(b.shape + extra)
        inv = inv.reshape(inv.shape + extra)
    dp = _np_prec.empty_like(rhs)
    dp[0] = rhs[0] * inv[0]
    for j in range(1, n):
        dp[j] = (rhs[j] - b[j] * dp[j - 1]) * inv[j]
    return dp


def _tridi_bwd_np(cp, dp) -> _np_prec.ndarray:
    """Backward sweep of Thomas algorithm in pure NumPy.

    x[j] = dp[j] - cp[j] * x[j+1]
    """
    n = dp.shape[0]
    if n <= 1:
        return dp.copy()
    x = _np_prec.empty_like(dp)
    x[n - 1] = dp[n - 1]
    for j in range(n - 2, -1, -1):
        x[j] = dp[j] - cp[j] * x[j + 1]
    return x


def _tridi_solve_precomputed_np(b, cp, inv, rhs) -> _np_prec.ndarray:
    """Solve tridiagonal system using precomputed c'/inv coefficients (NumPy)."""
    b   = _np_prec.asarray(b)
    cp  = _np_prec.asarray(cp)
    inv = _np_prec.asarray(inv)
    rhs = _np_prec.asarray(rhs)
    if rhs.ndim > inv.ndim:
        extra = (1,) * (rhs.ndim - inv.ndim)
        b   = b.reshape(b.shape + extra)
        cp  = cp.reshape(cp.shape + extra)
        inv = inv.reshape(inv.shape + extra)
    dp = _tridi_fwd_np(b, inv, rhs)
    return _tridi_bwd_np(cp, dp)


def _tridi_solve_np(a, d, b, rhs) -> _np_prec.ndarray:
    """Full Thomas algorithm (no precomputed coefficients) in pure NumPy."""
    a   = _np_prec.asarray(a)
    d   = _np_prec.asarray(d)
    b   = _np_prec.asarray(b)
    rhs = _np_prec.asarray(rhs)
    n = rhs.shape[0]
    if n == 0:
        return rhs.copy()
    if rhs.ndim > d.ndim:
        extra = (1,) * (rhs.ndim - d.ndim)
        a = a.reshape(a.shape + extra)
        d = d.reshape(d.shape + extra)
        b = b.reshape(b.shape + extra)
    eps = 1e-12
    a_norm = _np_prec.empty_like(rhs)
    x_mod  = _np_prec.empty_like(rhs)
    d0 = _np_prec.where(d[0] != 0.0, d[0], eps)
    a_norm[0] = a[0] / d0
    x_mod[0]  = rhs[0] / d0
    for j in range(1, n):
        denom = d[j] - a_norm[j - 1] * b[j]
        denom = _np_prec.where(denom != 0.0, denom, eps)
        a_norm[j] = a[j] / denom
        x_mod[j]  = (rhs[j] - x_mod[j - 1] * b[j]) / denom
    x = _np_prec.empty_like(rhs)
    x[n - 1] = x_mod[n - 1]
    for j in range(n - 2, -1, -1):
        x[j] = x_mod[j] - a_norm[j] * x[j + 1]
    return x


def rz_preconditioner_apply_numpy(
    *,
    frzl_in: TomnspsRZL,
    mats: dict,
    jmax: int,
    cfg,
    use_precomputed: bool | None = None,
) -> TomnspsRZL:
    """Pure-NumPy version of ``rz_preconditioner_apply_jit``.

    Avoids all JAX JIT dispatch overhead by running the Thomas tridiagonal
    solve as a plain Python loop over radial surfaces.  This is faster than
    the JIT path for the non-scan CPU iteration loop because the overhead of
    converting NumPy→JAX arrays, dispatching to XLA, and converting back
    dominates the actual computation for small ns (10–50 surfaces).
    """
    np = _np_prec
    if use_precomputed is None:
        _env_pre, _ = _get_env_tridi_flags()
        use_precomputed = _env_pre

    has_cr_ir = ("cr" in mats) and ("ir" in mats) and ("cz" in mats) and ("iz" in mats)
    if not has_cr_ir:
        use_precomputed = False

    lthreed = bool(getattr(cfg, "lthreed", False))
    lasym   = bool(getattr(cfg, "lasym",   False))
    use_rss = lthreed
    use_rsc = lasym
    use_rcs = lthreed and lasym
    use_zcs = lthreed
    use_zcc = lasym
    use_zss = lthreed and lasym

    # Convert all mats to NumPy once.
    def _get(key, fallback):
        v = mats.get(key, None)
        return np.asarray(v) if v is not None else np.asarray(fallback)

    ar = np.asarray(mats["ar"])
    br = np.asarray(mats["br"])
    dr = np.asarray(mats["dr"])
    cr = _get("cr", ar)
    ir = _get("ir", dr)
    az = np.asarray(mats["az"])
    bz = np.asarray(mats["bz"])
    dz = np.asarray(mats["dz"])
    cz = _get("cz", az)
    iz = _get("iz", dz)

    def _to_np(x, like):
        return np.asarray(x) if x is not None else np.zeros_like(like)

    frcc = np.asarray(frzl_in.frcc)
    fzsc = np.asarray(frzl_in.fzsc)
    frss = _to_np(frzl_in.frss, frcc)
    fzcs = _to_np(frzl_in.fzcs, fzsc)
    frsc = _to_np(getattr(frzl_in, "frsc", None), frcc)
    frcs = _to_np(getattr(frzl_in, "frcs", None), frcc)
    fzcc = _to_np(getattr(frzl_in, "fzcc", None), fzsc)
    fzss = _to_np(getattr(frzl_in, "fzss", None), fzsc)

    has_frss = frzl_in.frss is not None
    has_fzcs = frzl_in.fzcs is not None

    jmax = int(jmax)

    frcc_u = frcc.copy()
    frss_u = frss.copy()
    fzsc_u = fzsc.copy()
    fzcs_u = fzcs.copy()
    frsc_u = frsc.copy()
    frcs_u = frcs.copy()
    fzcc_u = fzcc.copy()
    fzss_u = fzss.copy()

    if jmax > 0:
        # --- m = 0 block ---
        r_blocks  = [frcc[:jmax]]
        if use_rss: r_blocks.append(frss[:jmax])
        if use_rsc: r_blocks.append(frsc[:jmax])
        if use_rcs: r_blocks.append(frcs[:jmax])
        rhs_r0 = np.stack(r_blocks, axis=-1)[:, 0, :]  # (jmax, nrange, nb_r)

        z_blocks  = [fzsc[:jmax]]
        if use_zcs: z_blocks.append(fzcs[:jmax])
        if use_zcc: z_blocks.append(fzcc[:jmax])
        if use_zss: z_blocks.append(fzss[:jmax])
        rhs_z0 = np.stack(z_blocks, axis=-1)[:, 0, :]  # (jmax, nrange, nb_z)

        if use_precomputed:
            sol_r0 = _tridi_solve_precomputed_np(br[:, 0, :], cr[:, 0, :], ir[:, 0, :], rhs_r0)
            sol_z0 = _tridi_solve_precomputed_np(bz[:, 0, :], cz[:, 0, :], iz[:, 0, :], rhs_z0)
        else:
            sol_r0 = _tridi_solve_np(ar[:, 0, :], dr[:, 0, :], br[:, 0, :], rhs_r0)
            sol_z0 = _tridi_solve_np(az[:, 0, :], dz[:, 0, :], bz[:, 0, :], rhs_z0)

        idx = 0
        frcc_u[:jmax, 0, :] = sol_r0[..., idx]; idx += 1
        if use_rss: frss_u[:jmax, 0, :] = sol_r0[..., idx]; idx += 1
        if use_rsc: frsc_u[:jmax, 0, :] = sol_r0[..., idx]; idx += 1
        if use_rcs: frcs_u[:jmax, 0, :] = sol_r0[..., idx]; idx += 1

        idx = 0
        fzsc_u[:jmax, 0, :] = sol_z0[..., idx]; idx += 1
        if use_zcs: fzcs_u[:jmax, 0, :] = sol_z0[..., idx]; idx += 1
        if use_zcc: fzcc_u[:jmax, 0, :] = sol_z0[..., idx]; idx += 1
        if use_zss: fzss_u[:jmax, 0, :] = sol_z0[..., idx]; idx += 1

        # --- m >= 1 blocks (jmin = 1, so surface 0 is skipped) ---
        mpol = int(frcc.shape[1])
        if mpol > 1 and jmax > 1:
            rhs_rm = np.stack(r_blocks, axis=-1)[1:, 1:, :]  # (jmax-1, mpol-1, nrange, nb_r)
            rhs_zm = np.stack(z_blocks, axis=-1)[1:, 1:, :]  # (jmax-1, mpol-1, nrange, nb_z)

            if use_precomputed:
                sol_rm = _tridi_solve_precomputed_np(
                    br[1:jmax, 1:, :], cr[1:jmax, 1:, :], ir[1:jmax, 1:, :], rhs_rm)
                sol_zm = _tridi_solve_precomputed_np(
                    bz[1:jmax, 1:, :], cz[1:jmax, 1:, :], iz[1:jmax, 1:, :], rhs_zm)
            else:
                sol_rm = _tridi_solve_np(
                    ar[1:jmax, 1:, :], dr[1:jmax, 1:, :], br[1:jmax, 1:, :], rhs_rm)
                sol_zm = _tridi_solve_np(
                    az[1:jmax, 1:, :], dz[1:jmax, 1:, :], bz[1:jmax, 1:, :], rhs_zm)

            idx = 0
            frcc_u[1:jmax, 1:, :] = sol_rm[..., idx]; idx += 1
            if use_rss: frss_u[1:jmax, 1:, :] = sol_rm[..., idx]; idx += 1
            if use_rsc: frsc_u[1:jmax, 1:, :] = sol_rm[..., idx]; idx += 1
            if use_rcs: frcs_u[1:jmax, 1:, :] = sol_rm[..., idx]; idx += 1

            idx = 0
            fzsc_u[1:jmax, 1:, :] = sol_zm[..., idx]; idx += 1
            if use_zcs: fzcs_u[1:jmax, 1:, :] = sol_zm[..., idx]; idx += 1
            if use_zcc: fzcc_u[1:jmax, 1:, :] = sol_zm[..., idx]; idx += 1
            if use_zss: fzss_u[1:jmax, 1:, :] = sol_zm[..., idx]; idx += 1

    return TomnspsRZL(
        frcc=frcc_u,
        frss=frss_u if has_frss else None,
        fzsc=fzsc_u,
        fzcs=fzcs_u if has_fzcs else None,
        flsc=np.asarray(frzl_in.flsc),
        flcs=(np.asarray(frzl_in.flcs) if frzl_in.flcs is not None else None),
        frsc=(frsc_u if getattr(frzl_in, "frsc", None) is not None else None),
        frcs=(frcs_u if getattr(frzl_in, "frcs", None) is not None else None),
        fzcc=(fzcc_u if getattr(frzl_in, "fzcc", None) is not None else None),
        fzss=(fzss_u if getattr(frzl_in, "fzss", None) is not None else None),
        flcc=(np.asarray(frzl_in.flcc) if getattr(frzl_in, "flcc", None) is not None else None),
        flss=(np.asarray(frzl_in.flss) if getattr(frzl_in, "flss", None) is not None else None),
    )

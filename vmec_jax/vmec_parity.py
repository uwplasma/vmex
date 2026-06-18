"""VMEC parity helper kernels.

This module provides small kernels that reproduce specific VMEC discrete
conventions used in the reference ``wout_*.nc`` outputs, as needed for parity
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._compat import jnp
from .fourier import eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys


@dataclass(frozen=True)
class ParityRZL:
    """Real-space R/Z/L fields split into VMEC even/odd-m parity pieces."""

    R_even: Any
    R_odd: Any
    Z_even: Any
    Z_odd: Any
    L_even: Any
    L_odd: Any
    Rt_even: Any
    Rt_odd: Any
    Zt_even: Any
    Zt_odd: Any
    Lt_even: Any
    Lt_odd: Any
    Rp_even: Any
    Rp_odd: Any
    Zp_even: Any
    Zp_odd: Any
    Lp_even: Any
    Lp_odd: Any


def split_rzl_even_odd_m(state, basis, modes_m: np.ndarray) -> ParityRZL:
    """Evaluate real-space fields for even-m and odd-m subsets separately."""
    m = np.asarray(modes_m, dtype=int)
    dtype = jnp.asarray(state.Rcos).dtype
    mask_even = jnp.asarray((m % 2) == 0, dtype=dtype)
    mask_odd = (1.0 - mask_even).astype(dtype)

    coeff_cos_stack = jnp.stack([state.Rcos, state.Zcos, state.Lcos], axis=0)
    coeff_sin_stack = jnp.stack([state.Rsin, state.Zsin, state.Lsin], axis=0)
    mask_stack = jnp.stack([mask_even, mask_odd], axis=0)

    def _eval_stack(mask_stack):
        coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
        coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
        return eval_fourier(coeff_cos, coeff_sin, basis, coeffs_internal=True)

    def _eval_stack_dtheta(mask_stack):
        coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
        coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
        return eval_fourier_dtheta(coeff_cos, coeff_sin, basis, coeffs_internal=True)

    def _eval_stack_dphi(mask_stack):
        coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
        coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
        return eval_fourier_dzeta_phys(coeff_cos, coeff_sin, basis, coeffs_internal=True)

    stack = _eval_stack(mask_stack)
    stack_t = _eval_stack_dtheta(mask_stack)
    stack_p = _eval_stack_dphi(mask_stack)

    even = stack[0]
    odd = stack[1]
    even_t = stack_t[0]
    odd_t = stack_t[1]
    even_p = stack_p[0]
    odd_p = stack_p[1]

    R_even = even[0]
    R_odd = odd[0]
    Z_even = even[1]
    Z_odd = odd[1]
    L_even = even[2]
    L_odd = odd[2]

    Rt_even = even_t[0]
    Rt_odd = odd_t[0]
    Zt_even = even_t[1]
    Zt_odd = odd_t[1]
    Lt_even = even_t[2]
    Lt_odd = odd_t[2]

    Rp_even = even_p[0]
    Rp_odd = odd_p[0]
    Zp_even = even_p[1]
    Zp_odd = odd_p[1]
    Lp_even = even_p[2]
    Lp_odd = odd_p[2]

    return ParityRZL(
        R_even=R_even,
        R_odd=R_odd,
        Z_even=Z_even,
        Z_odd=Z_odd,
        L_even=L_even,
        L_odd=L_odd,
        Rt_even=Rt_even,
        Rt_odd=Rt_odd,
        Zt_even=Zt_even,
        Zt_odd=Zt_odd,
        Lt_even=Lt_even,
        Lt_odd=Lt_odd,
        Rp_even=Rp_even,
        Rp_odd=Rp_odd,
        Zp_even=Zp_even,
        Zp_odd=Zp_odd,
        Lp_even=Lp_even,
        Lp_odd=Lp_odd,
    )


def split_rzl_even_odd_lasym(state, basis) -> ParityRZL:
    """Split real-space fields into even/odd pieces for LASYM=True.

    For asymmetric runs, VMEC's symrzl decomposition corresponds to separating
    cosine (even) and sine (odd) helical contributions rather than even/odd
    m-parity. This helper mirrors that by evaluating the cosine-only and
    sine-only series separately on the full VMEC grid.
    """
    coeff_cos_stack = jnp.stack([state.Rcos, state.Zcos, state.Lcos], axis=0)
    coeff_sin_stack = jnp.stack([state.Rsin, state.Zsin, state.Lsin], axis=0)

    even = eval_fourier(coeff_cos_stack, jnp.zeros_like(coeff_sin_stack), basis, coeffs_internal=True)
    odd = eval_fourier(jnp.zeros_like(coeff_cos_stack), coeff_sin_stack, basis, coeffs_internal=True)

    even_t = eval_fourier_dtheta(
        coeff_cos_stack, jnp.zeros_like(coeff_sin_stack), basis, coeffs_internal=True
    )
    odd_t = eval_fourier_dtheta(
        jnp.zeros_like(coeff_cos_stack), coeff_sin_stack, basis, coeffs_internal=True
    )
    even_p = eval_fourier_dzeta_phys(
        coeff_cos_stack, jnp.zeros_like(coeff_sin_stack), basis, coeffs_internal=True
    )
    odd_p = eval_fourier_dzeta_phys(
        jnp.zeros_like(coeff_cos_stack), coeff_sin_stack, basis, coeffs_internal=True
    )

    return ParityRZL(
        R_even=even[0],
        R_odd=odd[0],
        Z_even=even[1],
        Z_odd=odd[1],
        L_even=even[2],
        L_odd=odd[2],
        Rt_even=even_t[0],
        Rt_odd=odd_t[0],
        Zt_even=even_t[1],
        Zt_odd=odd_t[1],
        Lt_even=even_t[2],
        Lt_odd=odd_t[2],
        Rp_even=even_p[0],
        Rp_odd=odd_p[0],
        Zp_even=even_p[1],
        Zp_odd=odd_p[1],
        Lp_even=even_p[2],
        Lp_odd=odd_p[2],
    )


def internal_odd_from_physical(
    phys_odd,
    s,
    *,
    axis: Literal["copy_js2", "zero"] = "copy_js2",
    eps: float = 1e-14,
):
    """Convert physical odd-m contribution to VMEC internal odd field.

    VMEC represents:

        X = X_even + sqrt(s) * X_odd_internal

    so:

        X_odd_internal = X_odd_physical / sqrt(s)
    """
    s = jnp.asarray(s)
    sh = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    mask = (sh > eps).astype(jnp.asarray(phys_odd).dtype)
    out = jnp.asarray(phys_odd) * mask / jnp.where(sh > eps, sh, 1.0)

    if out.shape[0] >= 2:
        if axis == "copy_js2":
            # VMEC's `totzsps` performs an origin extrapolation for the *m=1*
            # modes (jmin1(1)=1), defining the internal odd field at js=1 by
            # copying from js=2. For modes with m>=2, VMEC enforces jmin1(m>=2)=2,
            # i.e. the internal field is zero on axis. When callers provide a
            # mixed odd-m real-space sum, this distinction must be handled by
            # splitting modes before calling this helper.
            out = out.at[0].set(out[1])
        elif axis == "zero":
            out = out.at[0].set(jnp.zeros_like(out[0]))
        else:  # pragma: no cover
            raise ValueError(f"Unknown axis rule: {axis!r}")
    return out


def internal_odd_from_physical_vmec_m1(
    *,
    odd_m1_phys,
    odd_mge2_phys,
    s,
    eps: float = 1e-14,
):
    """VMEC-consistent internal odd field from split odd-m contributions.

    VMEC axis rules (vmec_params.f):
      - jmin1(m=1) = 1  => extrapolate to axis (copy from js=2)
      - jmin1(m>=2)= 2  => internal odd is zero on axis

    Callers must provide the physical odd-m real-space contributions split into:
      - the m=1 subset (any n),
      - the m>=3 subset (odd-m, any n).
    """
    s = jnp.asarray(s)
    # VMEC jmin1(m=1)=1 => copy js=2 to axis for m=1; odd m>=3 zero on axis.
    odd_m1_int = internal_odd_from_physical(odd_m1_phys, s, axis="copy_js2", eps=eps)
    odd_rest_int = internal_odd_from_physical(odd_mge2_phys, s, axis="zero", eps=eps)
    return odd_m1_int + odd_rest_int


def internal_odd_from_physical_vmec_jlam(
    *,
    odd_m1_phys,
    odd_mge2_phys,
    s,
    eps: float = 1e-14,
):
    """VMEC-consistent internal odd field for lambda (jlam).

    VMEC enforces jlam=2 for lambda updates but still applies the standard
    m=1 axis copy (js=2 -> js=1) before real-space synthesis.
    """
    s = jnp.asarray(s)
    odd_m1_int = internal_odd_from_physical(odd_m1_phys, s, axis="copy_js2", eps=eps)
    odd_rest_int = internal_odd_from_physical(odd_mge2_phys, s, axis="zero", eps=eps)
    return odd_m1_int + odd_rest_int


def _mn_index_maps(modes) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Build (m,n>=0) index maps for signed coefficients."""
    key = _mn_index_cache_key(modes)
    cached = _MN_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    if m_arr.size == 0:
        out = (0, 0, np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=int))
        _MN_INDEX_CACHE[key] = out
        return out
    mpol = int(m_arr.max()) + 1
    ntor = int(np.abs(n_arr).max())
    nrange = ntor + 1
    idx_pos = -np.ones((mpol, nrange), dtype=int)
    idx_neg = -np.ones((mpol, nrange), dtype=int)
    for k, (m_k, n_k) in enumerate(zip(m_arr, n_arr)):
        if n_k >= 0:
            idx_pos[int(m_k), int(n_k)] = int(k)
        else:
            idx_neg[int(m_k), int(-n_k)] = int(k)
    out = (mpol, ntor, idx_pos, idx_neg)
    _MN_INDEX_CACHE[key] = out
    return out


def _signed_to_mn_cos(coeffs, idx_pos, idx_neg):
    coeffs = jnp.asarray(coeffs)
    idx_pos = jnp.asarray(idx_pos, dtype=jnp.int32)
    idx_neg = jnp.asarray(idx_neg, dtype=jnp.int32)
    ns = int(coeffs.shape[0])
    mpol, nrange = idx_pos.shape
    if mpol == 0 or nrange == 0:
        z = jnp.zeros((ns, mpol, nrange), dtype=coeffs.dtype)
        return z, z

    mask_pos = idx_pos >= 0
    mask_neg = idx_neg >= 0
    idx_pos_safe = jnp.where(mask_pos, idx_pos, 0)
    idx_neg_safe = jnp.where(mask_neg, idx_neg, 0)

    pos = coeffs[:, idx_pos_safe] * mask_pos.astype(coeffs.dtype)
    neg = coeffs[:, idx_neg_safe] * mask_neg.astype(coeffs.dtype)

    rcc = pos + neg
    rss = pos - neg

    m = jnp.arange(mpol, dtype=jnp.int32)[:, None]
    n = jnp.arange(nrange, dtype=jnp.int32)[None, :]
    zero_mask = (m == 0) | (n == 0)
    rss = jnp.where(zero_mask[None, :, :], jnp.zeros_like(rss), rss)
    return rcc, rss


def _signed_to_mn_sin(coeffs, idx_pos, idx_neg):
    coeffs = jnp.asarray(coeffs)
    idx_pos = jnp.asarray(idx_pos, dtype=jnp.int32)
    idx_neg = jnp.asarray(idx_neg, dtype=jnp.int32)
    ns = int(coeffs.shape[0])
    mpol, nrange = idx_pos.shape
    if mpol == 0 or nrange == 0:
        z = jnp.zeros((ns, mpol, nrange), dtype=coeffs.dtype)
        return z, z

    mask_pos = idx_pos >= 0
    mask_neg = idx_neg >= 0
    idx_pos_safe = jnp.where(mask_pos, idx_pos, 0)
    idx_neg_safe = jnp.where(mask_neg, idx_neg, 0)

    pos = coeffs[:, idx_pos_safe] * mask_pos.astype(coeffs.dtype)
    neg = coeffs[:, idx_neg_safe] * mask_neg.astype(coeffs.dtype)

    sc = pos + neg
    cs = neg - pos

    m = jnp.arange(mpol, dtype=jnp.int32)[:, None]
    n = jnp.arange(nrange, dtype=jnp.int32)[None, :]
    n0 = n == 0
    m0 = m == 0
    cs = jnp.where(n0[None, :, :], jnp.zeros_like(cs), cs)
    sc = jnp.where(m0[None, :, :] & (~n0[None, :, :]), jnp.zeros_like(sc), sc)
    return sc, cs


def _signed_to_mn_cos_cached(coeffs, *, maps: SignedModeMaps):
    """Cached version of _signed_to_mn_cos using prebuilt masks."""
    coeffs = jnp.asarray(coeffs)
    if maps.mpol == 0 or maps.nrange == 0:
        z = jnp.zeros((int(coeffs.shape[0]), maps.mpol, maps.nrange), dtype=coeffs.dtype)
        return z, z
    idx_pos_safe = maps.idx_pos_safe_j if maps.idx_pos_safe_j is not None else jnp.asarray(maps.idx_pos_safe, dtype=jnp.int32)
    idx_neg_safe = maps.idx_neg_safe_j if maps.idx_neg_safe_j is not None else jnp.asarray(maps.idx_neg_safe, dtype=jnp.int32)
    mask_pos = maps.mask_pos_j if maps.mask_pos_j is not None else jnp.asarray(maps.mask_pos)
    mask_neg = maps.mask_neg_j if maps.mask_neg_j is not None else jnp.asarray(maps.mask_neg)
    if mask_pos.dtype != coeffs.dtype:
        mask_pos = mask_pos.astype(coeffs.dtype)
    if mask_neg.dtype != coeffs.dtype:
        mask_neg = mask_neg.astype(coeffs.dtype)
    pos = coeffs[:, idx_pos_safe] * mask_pos
    neg = coeffs[:, idx_neg_safe] * mask_neg
    rcc = pos + neg
    rss = pos - neg
    zero_mask = maps.zero_mask_j if maps.zero_mask_j is not None else jnp.asarray(maps.zero_mask)
    rss = jnp.where(zero_mask[None, :, :], jnp.zeros_like(rss), rss)
    return rcc, rss


def _signed_to_mn_cos_host(coeffs, *, maps: SignedModeMaps):
    """NumPy host-side version of `_signed_to_mn_cos_cached`."""
    coeffs = np.asarray(coeffs)
    if maps.mpol == 0 or maps.nrange == 0:
        z = np.zeros((int(coeffs.shape[0]), maps.mpol, maps.nrange), dtype=coeffs.dtype)
        return z, z
    mask_pos = maps.mask_pos.astype(coeffs.dtype, copy=False)
    mask_neg = maps.mask_neg.astype(coeffs.dtype, copy=False)
    pos = coeffs[:, maps.idx_pos_safe] * mask_pos[None, :, :]
    neg = coeffs[:, maps.idx_neg_safe] * mask_neg[None, :, :]
    rcc = pos + neg
    rss = pos - neg
    rss[:, maps.zero_mask] = 0
    return rcc, rss


def _signed_to_mn_sin_cached(coeffs, *, maps: SignedModeMaps):
    """Cached version of _signed_to_mn_sin using prebuilt masks."""
    coeffs = jnp.asarray(coeffs)
    if maps.mpol == 0 or maps.nrange == 0:
        z = jnp.zeros((int(coeffs.shape[0]), maps.mpol, maps.nrange), dtype=coeffs.dtype)
        return z, z
    idx_pos_safe = maps.idx_pos_safe_j if maps.idx_pos_safe_j is not None else jnp.asarray(maps.idx_pos_safe, dtype=jnp.int32)
    idx_neg_safe = maps.idx_neg_safe_j if maps.idx_neg_safe_j is not None else jnp.asarray(maps.idx_neg_safe, dtype=jnp.int32)
    mask_pos = maps.mask_pos_j if maps.mask_pos_j is not None else jnp.asarray(maps.mask_pos)
    mask_neg = maps.mask_neg_j if maps.mask_neg_j is not None else jnp.asarray(maps.mask_neg)
    if mask_pos.dtype != coeffs.dtype:
        mask_pos = mask_pos.astype(coeffs.dtype)
    if mask_neg.dtype != coeffs.dtype:
        mask_neg = mask_neg.astype(coeffs.dtype)
    pos = coeffs[:, idx_pos_safe] * mask_pos
    neg = coeffs[:, idx_neg_safe] * mask_neg
    sc = pos + neg
    cs = neg - pos
    n0 = maps.n0_mask_j if maps.n0_mask_j is not None else jnp.asarray(maps.n0_mask)
    m0 = maps.m0_mask_j if maps.m0_mask_j is not None else jnp.asarray(maps.m0_mask)
    cs = jnp.where(n0[None, :, :], jnp.zeros_like(cs), cs)
    sc = jnp.where(m0[None, :, :] & (~n0[None, :, :]), jnp.zeros_like(sc), sc)
    return sc, cs


def _signed_to_mn_sin_host(coeffs, *, maps: SignedModeMaps):
    """NumPy host-side version of `_signed_to_mn_sin_cached`."""
    coeffs = np.asarray(coeffs)
    if maps.mpol == 0 or maps.nrange == 0:
        z = np.zeros((int(coeffs.shape[0]), maps.mpol, maps.nrange), dtype=coeffs.dtype)
        return z, z
    mask_pos = maps.mask_pos.astype(coeffs.dtype, copy=False)
    mask_neg = maps.mask_neg.astype(coeffs.dtype, copy=False)
    pos = coeffs[:, maps.idx_pos_safe] * mask_pos[None, :, :]
    neg = coeffs[:, maps.idx_neg_safe] * mask_neg[None, :, :]
    sc = pos + neg
    cs = neg - pos
    cs = np.where(maps.n0_mask[None, :, :], np.zeros_like(cs), cs)
    sc = np.where((maps.m0_mask & (~maps.n0_mask))[None, :, :], np.zeros_like(sc), sc)
    return sc, cs


def _mn_cos_to_signed(rcc, rss, idx_pos, idx_neg, ncoeff: int):
    rcc = jnp.asarray(rcc)
    rss = jnp.asarray(rss)
    idx_pos = jnp.asarray(idx_pos, dtype=jnp.int32)
    idx_neg = jnp.asarray(idx_neg, dtype=jnp.int32)
    ns = int(rcc.shape[0])
    out = jnp.zeros((ns, ncoeff), dtype=rcc.dtype)
    if ncoeff == 0:
        return out

    mpol, nrange = idx_pos.shape
    m = jnp.arange(mpol, dtype=jnp.int32)[:, None]
    n = jnp.arange(nrange, dtype=jnp.int32)[None, :]
    m0 = m == 0
    n0 = n == 0

    pos = 0.5 * (rcc + rss)
    pos = jnp.where((m0 | n0)[None, :, :], rcc, pos)
    neg = 0.5 * (rcc - rss)

    idx_pos_flat = idx_pos.reshape(-1)
    idx_neg_flat = idx_neg.reshape(-1)
    mask_pos = idx_pos_flat >= 0
    mask_neg = idx_neg_flat >= 0
    idx_pos_safe = jnp.where(mask_pos, idx_pos_flat, 0)
    idx_neg_safe = jnp.where(mask_neg, idx_neg_flat, 0)

    pos_flat = pos.reshape(ns, -1) * mask_pos.astype(rcc.dtype)[None, :]
    neg_flat = neg.reshape(ns, -1) * mask_neg.astype(rcc.dtype)[None, :]

    out = out.at[:, idx_pos_safe].add(pos_flat)
    out = out.at[:, idx_neg_safe].add(neg_flat)
    return out


def _mn_sin_to_signed(sc, cs, idx_pos, idx_neg, ncoeff: int):
    sc = jnp.asarray(sc)
    cs = jnp.asarray(cs)
    idx_pos = jnp.asarray(idx_pos, dtype=jnp.int32)
    idx_neg = jnp.asarray(idx_neg, dtype=jnp.int32)
    ns = int(sc.shape[0])
    out = jnp.zeros((ns, ncoeff), dtype=sc.dtype)
    if ncoeff == 0:
        return out

    mpol, nrange = idx_pos.shape
    m = jnp.arange(mpol, dtype=jnp.int32)[:, None]
    n = jnp.arange(nrange, dtype=jnp.int32)[None, :]
    m0 = m == 0
    n0 = n == 0

    pos = 0.5 * (sc - cs)
    pos = jnp.where(n0[None, :, :], sc, pos)

    mask_neg = idx_neg >= 0
    mask_no_neg = (~mask_neg) & (~n0)
    pos = jnp.where(mask_no_neg[None, :, :] & m0[None, :, :], -cs, pos)
    pos = jnp.where(mask_no_neg[None, :, :] & (~m0[None, :, :]), sc, pos)

    neg = 0.5 * (sc + cs)

    idx_pos_flat = idx_pos.reshape(-1)
    idx_neg_flat = idx_neg.reshape(-1)
    mask_pos_flat = idx_pos_flat >= 0
    mask_neg_flat = idx_neg_flat >= 0
    idx_pos_safe = jnp.where(mask_pos_flat, idx_pos_flat, 0)
    idx_neg_safe = jnp.where(mask_neg_flat, idx_neg_flat, 0)

    pos_flat = pos.reshape(ns, -1) * mask_pos_flat.astype(sc.dtype)[None, :]
    neg_flat = neg.reshape(ns, -1) * mask_neg_flat.astype(sc.dtype)[None, :]

    out = out.at[:, idx_pos_safe].add(pos_flat)
    out = out.at[:, idx_neg_safe].add(neg_flat)
    return out


def _mn_cos_to_signed_cached(rcc, rss, *, maps: SignedModeMaps, ncoeff: int):
    """Cached version of _mn_cos_to_signed using prebuilt masks."""
    rcc = jnp.asarray(rcc)
    rss = jnp.asarray(rss)
    ns = int(rcc.shape[0])
    out = jnp.zeros((ns, ncoeff), dtype=rcc.dtype)
    if ncoeff == 0:
        return out
    m0 = maps.m0_mask_j if maps.m0_mask_j is not None else jnp.asarray(maps.m0_mask)
    n0 = maps.n0_mask_j if maps.n0_mask_j is not None else jnp.asarray(maps.n0_mask)
    pos = 0.5 * (rcc + rss)
    pos = jnp.where((m0 | n0)[None, :, :], rcc, pos)
    neg = 0.5 * (rcc - rss)

    idx_pos_safe = maps.idx_pos_safe_flat_j if maps.idx_pos_safe_flat_j is not None else jnp.asarray(maps.idx_pos_safe_flat, dtype=jnp.int32)
    idx_neg_safe = maps.idx_neg_safe_flat_j if maps.idx_neg_safe_flat_j is not None else jnp.asarray(maps.idx_neg_safe_flat, dtype=jnp.int32)
    mask_pos = maps.mask_pos_flat_j if maps.mask_pos_flat_j is not None else jnp.asarray(maps.mask_pos_flat)
    mask_neg = maps.mask_neg_flat_j if maps.mask_neg_flat_j is not None else jnp.asarray(maps.mask_neg_flat)
    if mask_pos.dtype != rcc.dtype:
        mask_pos = mask_pos.astype(rcc.dtype)
    if mask_neg.dtype != rcc.dtype:
        mask_neg = mask_neg.astype(rcc.dtype)

    pos_flat = pos.reshape(ns, -1) * mask_pos[None, :]
    neg_flat = neg.reshape(ns, -1) * mask_neg[None, :]

    idx_all = jnp.concatenate([idx_pos_safe, idx_neg_safe], axis=0)
    vals_all = jnp.concatenate([pos_flat, neg_flat], axis=1)
    out = out.at[:, idx_all].add(vals_all)
    return out


def _mn_cos_to_signed_host(rcc, rss, *, maps: SignedModeMaps, ncoeff: int):
    """NumPy host-side version of `_mn_cos_to_signed_cached`."""
    rcc = np.asarray(rcc)
    rss = np.asarray(rss)
    ns = int(rcc.shape[0])
    out = np.zeros((ns, ncoeff), dtype=rcc.dtype)
    if ncoeff == 0:
        return out
    pos = 0.5 * (rcc + rss)
    pos = np.where((maps.m0_mask | maps.n0_mask)[None, :, :], rcc, pos)
    neg = 0.5 * (rcc - rss)
    pos_flat = pos.reshape(ns, -1) * maps.mask_pos_flat.astype(rcc.dtype, copy=False)[None, :]
    neg_flat = neg.reshape(ns, -1) * maps.mask_neg_flat.astype(rcc.dtype, copy=False)[None, :]
    out[:, maps.idx_pos_safe_flat] += pos_flat
    out[:, maps.idx_neg_safe_flat] += neg_flat
    return out


def _mn_sin_to_signed_cached(sc, cs, *, maps: SignedModeMaps, ncoeff: int):
    """Cached version of _mn_sin_to_signed using prebuilt masks."""
    sc = jnp.asarray(sc)
    cs = jnp.asarray(cs)
    ns = int(sc.shape[0])
    out = jnp.zeros((ns, ncoeff), dtype=sc.dtype)
    if ncoeff == 0:
        return out
    m0 = maps.m0_mask_j if maps.m0_mask_j is not None else jnp.asarray(maps.m0_mask)
    n0 = maps.n0_mask_j if maps.n0_mask_j is not None else jnp.asarray(maps.n0_mask)

    pos = 0.5 * (sc - cs)
    pos = jnp.where(n0[None, :, :], sc, pos)

    mask_neg = maps.mask_neg_j if maps.mask_neg_j is not None else jnp.asarray(maps.mask_neg)
    mask_no_neg = (~mask_neg) & (~n0)
    pos = jnp.where(mask_no_neg[None, :, :] & m0[None, :, :], -cs, pos)
    pos = jnp.where(mask_no_neg[None, :, :] & (~m0[None, :, :]), sc, pos)

    neg = 0.5 * (sc + cs)

    idx_pos_safe = maps.idx_pos_safe_flat_j if maps.idx_pos_safe_flat_j is not None else jnp.asarray(maps.idx_pos_safe_flat, dtype=jnp.int32)
    idx_neg_safe = maps.idx_neg_safe_flat_j if maps.idx_neg_safe_flat_j is not None else jnp.asarray(maps.idx_neg_safe_flat, dtype=jnp.int32)
    mask_pos = maps.mask_pos_flat_j if maps.mask_pos_flat_j is not None else jnp.asarray(maps.mask_pos_flat)
    mask_neg_flat = maps.mask_neg_flat_j if maps.mask_neg_flat_j is not None else jnp.asarray(maps.mask_neg_flat)
    if mask_pos.dtype != sc.dtype:
        mask_pos = mask_pos.astype(sc.dtype)
    if mask_neg_flat.dtype != sc.dtype:
        mask_neg_flat = mask_neg_flat.astype(sc.dtype)

    pos_flat = pos.reshape(ns, -1) * mask_pos[None, :]
    neg_flat = neg.reshape(ns, -1) * mask_neg_flat[None, :]

    idx_all = jnp.concatenate([idx_pos_safe, idx_neg_safe], axis=0)
    vals_all = jnp.concatenate([pos_flat, neg_flat], axis=1)
    out = out.at[:, idx_all].add(vals_all)
    return out


def _mn_sin_to_signed_host(sc, cs, *, maps: SignedModeMaps, ncoeff: int):
    """NumPy host-side version of `_mn_sin_to_signed_cached`."""
    sc = np.asarray(sc)
    cs = np.asarray(cs)
    ns = int(sc.shape[0])
    out = np.zeros((ns, ncoeff), dtype=sc.dtype)
    if ncoeff == 0:
        return out
    pos = 0.5 * (sc - cs)
    pos = np.where(maps.n0_mask[None, :, :], sc, pos)
    mask_no_neg = (~maps.mask_neg) & (~maps.n0_mask)
    pos = np.where((mask_no_neg & maps.m0_mask)[None, :, :], -cs, pos)
    pos = np.where((mask_no_neg & (~maps.m0_mask))[None, :, :], sc, pos)
    neg = 0.5 * (sc + cs)
    pos_flat = pos.reshape(ns, -1) * maps.mask_pos_flat.astype(sc.dtype, copy=False)[None, :]
    neg_flat = neg.reshape(ns, -1) * maps.mask_neg_flat.astype(sc.dtype, copy=False)[None, :]
    out[:, maps.idx_pos_safe_flat] += pos_flat
    out[:, maps.idx_neg_safe_flat] += neg_flat
    return out


def vmec_m1_internal_to_physical_signed(
    *,
    Rcos,
    Zsin,
    Rsin,
    Zcos,
    modes,
    lthreed: bool,
    lasym: bool,
    lconm1: bool,
):
    """Undo VMEC's m=1 internal constraint in signed coefficient storage.

    VMEC stores m=1 symmetric (rss,zcs) and asymmetric (rsc,zcc) pairs in a
    constrained internal basis. For real-space synthesis, these must be
    converted back to the physical pair via:
        rss_phys = rss_int + zcs_int
        zcs_phys = rss_int - zcs_int
    (and similarly for rsc/zcc in the asymmetric sector).
    """
    if not bool(lconm1) or (not bool(lthreed) and not bool(lasym)):
        return Rcos, Zsin, Rsin, Zcos
    maps = signed_maps_from_modes(modes)
    if maps.mpol <= 1:
        return Rcos, Zsin, Rsin, Zcos
    ncoeff = int(jnp.asarray(Rcos).shape[1])

    Rcos_out = Rcos
    Zsin_out = Zsin
    Rsin_out = Rsin
    Zcos_out = Zcos

    if bool(lthreed):
        rcc, rss = _signed_to_mn_cos_cached(Rcos, maps=maps)
        zsc, zcs = _signed_to_mn_sin_cached(Zsin, maps=maps)
        rss_m1 = rss[:, 1, :]
        zcs_m1 = zcs[:, 1, :]
        rss = rss.at[:, 1, :].set(rss_m1 + zcs_m1)
        zcs = zcs.at[:, 1, :].set(rss_m1 - zcs_m1)
        Rcos_out = _mn_cos_to_signed_cached(rcc, rss, maps=maps, ncoeff=ncoeff)
        Zsin_out = _mn_sin_to_signed_cached(zsc, zcs, maps=maps, ncoeff=ncoeff)

    if bool(lasym):
        rsc, rcs = _signed_to_mn_sin_cached(Rsin, maps=maps)
        zcc, zss = _signed_to_mn_cos_cached(Zcos, maps=maps)
        rsc_m1 = rsc[:, 1, :]
        zcc_m1 = zcc[:, 1, :]
        rsc = rsc.at[:, 1, :].set(rsc_m1 + zcc_m1)
        zcc = zcc.at[:, 1, :].set(rsc_m1 - zcc_m1)
        Rsin_out = _mn_sin_to_signed_cached(rsc, rcs, maps=maps, ncoeff=ncoeff)
        Zcos_out = _mn_cos_to_signed_cached(zcc, zss, maps=maps, ncoeff=ncoeff)

    return Rcos_out, Zsin_out, Rsin_out, Zcos_out


def vmec_m1_internal_to_physical_signed_host(
    *,
    Rcos,
    Zsin,
    Rsin,
    Zcos,
    modes,
    lthreed: bool,
    lasym: bool,
    lconm1: bool,
):
    """NumPy host-side version of `vmec_m1_internal_to_physical_signed`."""
    if not bool(lconm1) or (not bool(lthreed) and not bool(lasym)):
        return np.asarray(Rcos), np.asarray(Zsin), np.asarray(Rsin), np.asarray(Zcos)
    maps = signed_maps_from_modes(modes)
    if maps.mpol <= 1:
        return np.asarray(Rcos), np.asarray(Zsin), np.asarray(Rsin), np.asarray(Zcos)
    Rcos = np.asarray(Rcos)
    Zsin = np.asarray(Zsin)
    Rsin = np.asarray(Rsin)
    Zcos = np.asarray(Zcos)
    ncoeff = int(Rcos.shape[1])

    Rcos_out = Rcos
    Zsin_out = Zsin
    Rsin_out = Rsin
    Zcos_out = Zcos

    if bool(lthreed):
        rcc, rss = _signed_to_mn_cos_host(Rcos, maps=maps)
        zsc, zcs = _signed_to_mn_sin_host(Zsin, maps=maps)
        rss_m1 = np.array(rss[:, 1, :], copy=True)
        zcs_m1 = np.array(zcs[:, 1, :], copy=True)
        rss[:, 1, :] = rss_m1 + zcs_m1
        zcs[:, 1, :] = rss_m1 - zcs_m1
        Rcos_out = _mn_cos_to_signed_host(rcc, rss, maps=maps, ncoeff=ncoeff)
        Zsin_out = _mn_sin_to_signed_host(zsc, zcs, maps=maps, ncoeff=ncoeff)

    if bool(lasym):
        rsc, rcs = _signed_to_mn_sin_host(Rsin, maps=maps)
        zcc, zss = _signed_to_mn_cos_host(Zcos, maps=maps)
        rsc_m1 = np.array(rsc[:, 1, :], copy=True)
        zcc_m1 = np.array(zcc[:, 1, :], copy=True)
        rsc[:, 1, :] = rsc_m1 + zcc_m1
        zcc[:, 1, :] = rsc_m1 - zcc_m1
        Rsin_out = _mn_sin_to_signed_host(rsc, rcs, maps=maps, ncoeff=ncoeff)
        Zcos_out = _mn_cos_to_signed_host(zcc, zss, maps=maps, ncoeff=ncoeff)

    return Rcos_out, Zsin_out, Rsin_out, Zcos_out


def vmec_m1_physical_to_internal_signed(
    *,
    Rcos,
    Zsin,
    Rsin,
    Zcos,
    modes,
    lthreed: bool,
    lasym: bool,
    lconm1: bool,
):
    """Apply VMEC's m=1 internal constraint in signed coefficient storage.

    This is the inverse of :func:`vmec_m1_internal_to_physical_signed`, mapping
    physical (output) rss/zcs (and rsc/zcc) pairs back to the constrained
    internal representation:

        rss_int = 0.5 * (rss_phys + zcs_phys)
        zcs_int = 0.5 * (rss_phys - zcs_phys)
    """
    if not bool(lconm1) or (not bool(lthreed) and not bool(lasym)):
        return Rcos, Zsin, Rsin, Zcos
    maps = signed_maps_from_modes(modes)
    if maps.mpol <= 1:
        return Rcos, Zsin, Rsin, Zcos
    ncoeff = int(jnp.asarray(Rcos).shape[1])

    Rcos_out = Rcos
    Zsin_out = Zsin
    Rsin_out = Rsin
    Zcos_out = Zcos

    if bool(lthreed):
        rcc, rss = _signed_to_mn_cos_cached(Rcos, maps=maps)
        zsc, zcs = _signed_to_mn_sin_cached(Zsin, maps=maps)
        rss_m1 = rss[:, 1, :]
        zcs_m1 = zcs[:, 1, :]
        rss = rss.at[:, 1, :].set(0.5 * (rss_m1 + zcs_m1))
        zcs = zcs.at[:, 1, :].set(0.5 * (rss_m1 - zcs_m1))
        Rcos_out = _mn_cos_to_signed_cached(rcc, rss, maps=maps, ncoeff=ncoeff)
        Zsin_out = _mn_sin_to_signed_cached(zsc, zcs, maps=maps, ncoeff=ncoeff)

    if bool(lasym):
        rsc, rcs = _signed_to_mn_sin_cached(Rsin, maps=maps)
        zcc, zss = _signed_to_mn_cos_cached(Zcos, maps=maps)
        rsc_m1 = rsc[:, 1, :]
        zcc_m1 = zcc[:, 1, :]
        rsc = rsc.at[:, 1, :].set(0.5 * (rsc_m1 + zcc_m1))
        zcc = zcc.at[:, 1, :].set(0.5 * (rsc_m1 - zcc_m1))
        Rsin_out = _mn_sin_to_signed_cached(rsc, rcs, maps=maps, ncoeff=ncoeff)
        Zcos_out = _mn_cos_to_signed_cached(zcc, zss, maps=maps, ncoeff=ncoeff)

    return Rcos_out, Zsin_out, Rsin_out, Zcos_out
_MN_INDEX_CACHE: dict[tuple, tuple[int, int, np.ndarray, np.ndarray]] = {}
_MN_SIGNED_MAP_CACHE: dict[tuple, "SignedModeMaps"] = {}


@dataclass(frozen=True)
class SignedModeMaps:
    """Precomputed index/mask maps for signed <-> (m,n>=0) conversions."""

    mpol: int
    nrange: int
    idx_pos: np.ndarray
    idx_neg: np.ndarray
    idx_pos_safe: np.ndarray
    idx_neg_safe: np.ndarray
    mask_pos: np.ndarray
    mask_neg: np.ndarray
    idx_pos_flat: np.ndarray
    idx_neg_flat: np.ndarray
    idx_pos_safe_flat: np.ndarray
    idx_neg_safe_flat: np.ndarray
    mask_pos_flat: np.ndarray
    mask_neg_flat: np.ndarray
    zero_mask: np.ndarray
    m0_mask: np.ndarray
    n0_mask: np.ndarray
    # Cached JAX arrays for performance (avoid per-call host->device copies).
    idx_pos_safe_j: any | None = None
    idx_neg_safe_j: any | None = None
    mask_pos_j: any | None = None
    mask_neg_j: any | None = None
    idx_pos_safe_flat_j: any | None = None
    idx_neg_safe_flat_j: any | None = None
    mask_pos_flat_j: any | None = None
    mask_neg_flat_j: any | None = None
    zero_mask_j: any | None = None
    m0_mask_j: any | None = None
    n0_mask_j: any | None = None


def _build_signed_maps(idx_pos: np.ndarray, idx_neg: np.ndarray) -> SignedModeMaps:
    """Build cached masks/safe indices for signed<->(m,n>=0) maps."""
    idx_pos = np.asarray(idx_pos, dtype=np.int32)
    idx_neg = np.asarray(idx_neg, dtype=np.int32)
    mpol, nrange = idx_pos.shape

    mask_pos = idx_pos >= 0
    mask_neg = idx_neg >= 0
    idx_pos_safe = np.where(mask_pos, idx_pos, 0).astype(np.int32)
    idx_neg_safe = np.where(mask_neg, idx_neg, 0).astype(np.int32)

    idx_pos_flat = idx_pos.reshape(-1)
    idx_neg_flat = idx_neg.reshape(-1)
    mask_pos_flat = idx_pos_flat >= 0
    mask_neg_flat = idx_neg_flat >= 0
    idx_pos_safe_flat = np.where(mask_pos_flat, idx_pos_flat, 0).astype(np.int32)
    idx_neg_safe_flat = np.where(mask_neg_flat, idx_neg_flat, 0).astype(np.int32)

    m = np.arange(mpol, dtype=np.int32)[:, None]
    n = np.arange(nrange, dtype=np.int32)[None, :]
    zero_mask = (m == 0) | (n == 0)
    m0_mask = m == 0
    n0_mask = n == 0

    return SignedModeMaps(
        mpol=mpol,
        nrange=nrange,
        idx_pos=idx_pos,
        idx_neg=idx_neg,
        idx_pos_safe=idx_pos_safe,
        idx_neg_safe=idx_neg_safe,
        mask_pos=mask_pos,
        mask_neg=mask_neg,
        idx_pos_flat=idx_pos_flat,
        idx_neg_flat=idx_neg_flat,
        idx_pos_safe_flat=idx_pos_safe_flat,
        idx_neg_safe_flat=idx_neg_safe_flat,
        mask_pos_flat=mask_pos_flat,
        mask_neg_flat=mask_neg_flat,
        zero_mask=zero_mask,
        m0_mask=m0_mask,
        n0_mask=n0_mask,
        idx_pos_safe_j=jnp.asarray(idx_pos_safe, dtype=jnp.int32),
        idx_neg_safe_j=jnp.asarray(idx_neg_safe, dtype=jnp.int32),
        mask_pos_j=jnp.asarray(mask_pos),
        mask_neg_j=jnp.asarray(mask_neg),
        idx_pos_safe_flat_j=jnp.asarray(idx_pos_safe_flat, dtype=jnp.int32),
        idx_neg_safe_flat_j=jnp.asarray(idx_neg_safe_flat, dtype=jnp.int32),
        mask_pos_flat_j=jnp.asarray(mask_pos_flat),
        mask_neg_flat_j=jnp.asarray(mask_neg_flat),
        zero_mask_j=jnp.asarray(zero_mask),
        m0_mask_j=jnp.asarray(m0_mask),
        n0_mask_j=jnp.asarray(n0_mask),
    )


def signed_maps_from_modes(modes) -> SignedModeMaps:
    """Cached SignedModeMaps keyed by ModeTable."""
    key = _mn_index_cache_key(modes)
    cached = _MN_SIGNED_MAP_CACHE.get(key)
    if cached is not None:
        return cached
    _mpol, _ntor, idx_pos, idx_neg = _mn_index_maps(modes)
    maps = _build_signed_maps(idx_pos, idx_neg)
    _MN_SIGNED_MAP_CACHE[key] = maps
    return maps


def _mn_index_cache_key(modes) -> tuple:
    m_arr = np.asarray(modes.m)
    n_arr = np.asarray(modes.n)
    return (
        m_arr.shape,
        n_arr.shape,
        str(m_arr.dtype),
        str(n_arr.dtype),
        m_arr.tobytes(),
        n_arr.tobytes(),
    )

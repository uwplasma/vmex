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
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    if m_arr.size == 0:
        return 0, 0, np.zeros((0, 0), dtype=int), np.zeros((0, 0), dtype=int)
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
    return mpol, ntor, idx_pos, idx_neg


def _signed_to_mn_cos(coeffs, idx_pos, idx_neg):
    coeffs = jnp.asarray(coeffs)
    ns = int(coeffs.shape[0])
    mpol, nrange = idx_pos.shape
    rcc = jnp.zeros((ns, mpol, nrange), dtype=coeffs.dtype)
    rss = jnp.zeros_like(rcc)
    for m_i in range(mpol):
        for n_i in range(nrange):
            kp = int(idx_pos[m_i, n_i])
            if kp < 0:
                continue
            pos = coeffs[:, kp]
            kn = int(idx_neg[m_i, n_i])
            neg = coeffs[:, kn] if kn >= 0 else jnp.zeros_like(pos)
            rcc = rcc.at[:, m_i, n_i].set(pos + neg)
            rss_val = pos - neg
            if n_i == 0 or m_i == 0:
                rss_val = jnp.zeros_like(pos)
            rss = rss.at[:, m_i, n_i].set(rss_val)
    return rcc, rss


def _signed_to_mn_sin(coeffs, idx_pos, idx_neg):
    coeffs = jnp.asarray(coeffs)
    ns = int(coeffs.shape[0])
    mpol, nrange = idx_pos.shape
    sc = jnp.zeros((ns, mpol, nrange), dtype=coeffs.dtype)
    cs = jnp.zeros_like(sc)
    for m_i in range(mpol):
        for n_i in range(nrange):
            kp = int(idx_pos[m_i, n_i])
            if kp < 0:
                continue
            pos = coeffs[:, kp]
            kn = int(idx_neg[m_i, n_i])
            neg = coeffs[:, kn] if kn >= 0 else jnp.zeros_like(pos)
            sc_val = pos + neg
            cs_val = neg - pos
            if n_i == 0:
                cs_val = jnp.zeros_like(pos)
            elif m_i == 0:
                sc_val = jnp.zeros_like(pos)
            sc = sc.at[:, m_i, n_i].set(sc_val)
            cs = cs.at[:, m_i, n_i].set(cs_val)
    return sc, cs


def _mn_cos_to_signed(rcc, rss, idx_pos, idx_neg, ncoeff: int):
    rcc = jnp.asarray(rcc)
    rss = jnp.asarray(rss)
    ns = int(rcc.shape[0])
    mpol, nrange = idx_pos.shape
    out = jnp.zeros((ns, ncoeff), dtype=rcc.dtype)
    for m_i in range(mpol):
        for n_i in range(nrange):
            kp = int(idx_pos[m_i, n_i])
            if kp < 0:
                continue
            if n_i == 0 or m_i == 0:
                pos = rcc[:, m_i, n_i]
            else:
                pos = 0.5 * (rcc[:, m_i, n_i] + rss[:, m_i, n_i])
            out = out.at[:, kp].set(pos)
            kn = int(idx_neg[m_i, n_i])
            if kn >= 0:
                neg = 0.5 * (rcc[:, m_i, n_i] - rss[:, m_i, n_i])
                out = out.at[:, kn].set(neg)
    return out


def _mn_sin_to_signed(sc, cs, idx_pos, idx_neg, ncoeff: int):
    sc = jnp.asarray(sc)
    cs = jnp.asarray(cs)
    ns = int(sc.shape[0])
    mpol, nrange = idx_pos.shape
    out = jnp.zeros((ns, ncoeff), dtype=sc.dtype)
    for m_i in range(mpol):
        for n_i in range(nrange):
            kp = int(idx_pos[m_i, n_i])
            if kp < 0:
                continue
            kn = int(idx_neg[m_i, n_i])
            if n_i == 0:
                pos = sc[:, m_i, n_i]
            elif kn >= 0:
                pos = 0.5 * (sc[:, m_i, n_i] - cs[:, m_i, n_i])
            else:
                pos = -cs[:, m_i, n_i] if m_i == 0 else sc[:, m_i, n_i]
            out = out.at[:, kp].set(pos)
            if kn >= 0:
                neg = 0.5 * (sc[:, m_i, n_i] + cs[:, m_i, n_i])
                out = out.at[:, kn].set(neg)
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
    mpol, ntor, idx_pos, idx_neg = _mn_index_maps(modes)
    if mpol <= 1:
        return Rcos, Zsin, Rsin, Zcos
    ncoeff = int(jnp.asarray(Rcos).shape[1])

    Rcos_out = Rcos
    Zsin_out = Zsin
    Rsin_out = Rsin
    Zcos_out = Zcos

    if bool(lthreed):
        rcc, rss = _signed_to_mn_cos(Rcos, idx_pos, idx_neg)
        zsc, zcs = _signed_to_mn_sin(Zsin, idx_pos, idx_neg)
        rss_m1 = rss[:, 1, :]
        zcs_m1 = zcs[:, 1, :]
        rss = rss.at[:, 1, :].set(rss_m1 + zcs_m1)
        zcs = zcs.at[:, 1, :].set(rss_m1 - zcs_m1)
        Rcos_out = _mn_cos_to_signed(rcc, rss, idx_pos, idx_neg, ncoeff=ncoeff)
        Zsin_out = _mn_sin_to_signed(zsc, zcs, idx_pos, idx_neg, ncoeff=ncoeff)

    if bool(lasym):
        rsc, rcs = _signed_to_mn_sin(Rsin, idx_pos, idx_neg)
        zcc, zss = _signed_to_mn_cos(Zcos, idx_pos, idx_neg)
        rsc_m1 = rsc[:, 1, :]
        zcc_m1 = zcc[:, 1, :]
        rsc = rsc.at[:, 1, :].set(rsc_m1 + zcc_m1)
        zcc = zcc.at[:, 1, :].set(rsc_m1 - zcc_m1)
        Rsin_out = _mn_sin_to_signed(rsc, rcs, idx_pos, idx_neg, ncoeff=ncoeff)
        Zcos_out = _mn_cos_to_signed(zcc, zss, idx_pos, idx_neg, ncoeff=ncoeff)

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
    mpol, _ntor, idx_pos, idx_neg = _mn_index_maps(modes)
    if mpol <= 1:
        return Rcos, Zsin, Rsin, Zcos
    ncoeff = int(jnp.asarray(Rcos).shape[1])

    Rcos_out = Rcos
    Zsin_out = Zsin
    Rsin_out = Rsin
    Zcos_out = Zcos

    if bool(lthreed):
        rcc, rss = _signed_to_mn_cos(Rcos, idx_pos, idx_neg)
        zsc, zcs = _signed_to_mn_sin(Zsin, idx_pos, idx_neg)
        rss_m1 = rss[:, 1, :]
        zcs_m1 = zcs[:, 1, :]
        rss = rss.at[:, 1, :].set(0.5 * (rss_m1 + zcs_m1))
        zcs = zcs.at[:, 1, :].set(0.5 * (rss_m1 - zcs_m1))
        Rcos_out = _mn_cos_to_signed(rcc, rss, idx_pos, idx_neg, ncoeff=ncoeff)
        Zsin_out = _mn_sin_to_signed(zsc, zcs, idx_pos, idx_neg, ncoeff=ncoeff)

    if bool(lasym):
        rsc, rcs = _signed_to_mn_sin(Rsin, idx_pos, idx_neg)
        zcc, zss = _signed_to_mn_cos(Zcos, idx_pos, idx_neg)
        rsc_m1 = rsc[:, 1, :]
        zcc_m1 = zcc[:, 1, :]
        rsc = rsc.at[:, 1, :].set(0.5 * (rsc_m1 + zcc_m1))
        zcc = zcc.at[:, 1, :].set(0.5 * (rsc_m1 - zcc_m1))
        Rsin_out = _mn_sin_to_signed(rsc, rcs, idx_pos, idx_neg, ncoeff=ncoeff)
        Zcos_out = _mn_cos_to_signed(zcc, zss, idx_pos, idx_neg, ncoeff=ncoeff)

    return Rcos_out, Zsin_out, Rsin_out, Zcos_out

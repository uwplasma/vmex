"""VMEC multigrid staging helpers (fixed-boundary parity work).

VMEC2000 typically uses multigrid in the radial direction via `NS_ARRAY`,
solving on a coarse `ns` first and then interpolating the Fourier coefficients
onto the next grid. The interpolation has a VMEC-specific convention:

- interpolate the **scaled** coefficients `x_old * scalxc_old`, where `scalxc`
  converts odd-m harmonics into VMEC's internal 1/sqrt(s) representation,
- then divide by `scalxc_new` to return to physical coefficients on the new grid,
- extrapolate odd-m values to the axis on the *scaled* array before interpolating,
  and zero odd-m coefficients on the axis on output.

This module ports the core of `STELLOPT/VMEC2000/Sources/TimeStep/interp.f`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._compat import jnp
from .state import StateLayout, VMECState


def _scalxc_vmec(*, ns: int, m: Any, dtype) -> Any:
    """VMEC `scalxc(js,m)` factors for each stored mode.

    Parameters
    ----------
    ns:
        Number of radial surfaces.
    m:
        Poloidal mode numbers per coefficient, shape (K,).
    dtype:
        Output dtype.
    """
    ns = int(ns)
    m = jnp.asarray(m)
    if ns <= 0:
        return jnp.zeros((0, int(m.shape[0])), dtype=dtype)

    s = jnp.linspace(0.0, 1.0, ns, dtype=dtype)
    sqrts = jnp.sqrt(jnp.maximum(s, 0.0))
    # VMEC sets sqrts(ns)=1 explicitly.
    if ns >= 1:
        sqrts = jnp.where(jnp.arange(ns, dtype=jnp.int32) == (ns - 1), jnp.asarray(1.0, dtype=dtype), sqrts)
    sq2 = sqrts[1] if ns >= 2 else jnp.asarray(1.0, dtype=dtype)
    scal_odd = 1.0 / jnp.maximum(sqrts, sq2)

    is_odd = ((m.astype(jnp.int32) % 2) == 1).astype(dtype)
    return jnp.where(is_odd[None, :] > 0, scal_odd[:, None], jnp.ones((ns, int(m.shape[0])), dtype=dtype))


def interp_vmec_radial_coeffs(
    x_old: Any,
    *,
    m: Any,
    ns_new: int,
) -> Any:
    """Interpolate a (ns_old, K) coefficient array onto a new VMEC radial grid.

    This reproduces VMEC2000's `interp.f` convention described in the module
    docstring.
    """
    x_old = jnp.asarray(x_old)
    ns_old, K = int(x_old.shape[0]), int(x_old.shape[1])
    ns_new = int(ns_new)
    if ns_old <= 0 or ns_new <= 0:
        return jnp.zeros((max(ns_new, 0), K), dtype=x_old.dtype)

    m = jnp.asarray(m)
    if int(m.shape[0]) != K:
        raise ValueError(f"m has shape {m.shape}, expected (K,) with K={K}")

    # Degenerate grids: fall back to a direct copy/truncate.
    if ns_old == ns_new:
        return x_old
    if ns_new == 1:
        return x_old[:1]
    if ns_old == 1:
        return jnp.broadcast_to(x_old[:1], (ns_new, K))

    dtype = x_old.dtype
    scal_old = _scalxc_vmec(ns=ns_old, m=m, dtype=dtype)
    scal_new = _scalxc_vmec(ns=ns_new, m=m, dtype=dtype)

    # Work in scaled (internal odd-m) representation.
    x_scaled = x_old * scal_old

    # Extrapolate odd-m modes over sqrt(s) to the axis on the scaled array:
    #   x(1) = 2*x(2) - x(3)   (Fortran, 1-based)
    if ns_old >= 3:
        is_odd = ((m.astype(jnp.int32) % 2) == 1).astype(dtype)
        axis_extrap = 2.0 * x_scaled[1] - x_scaled[2]
        axis_row = jnp.where(is_odd > 0, axis_extrap, x_scaled[0])
        x_scaled = jnp.concatenate([axis_row[None, :], x_scaled[1:]], axis=0)

    # Uniform-grid interpolation matching interp.f's js1/js2/xint construction.
    j = jnp.arange(ns_new, dtype=jnp.int32)
    num = j.astype(jnp.int64) * int(ns_old - 1)
    den = int(ns_new - 1)
    j1 = (num // den).astype(jnp.int32)
    j2 = jnp.minimum(j1 + 1, int(ns_old - 1))

    # xint = (sj - s1)/hsold with sj=j/(ns_new-1), s1=j1/(ns_old-1), hsold=1/(ns_old-1)
    # => xint = j*(ns_old-1)/(ns_new-1) - j1
    xint = (j.astype(dtype) * float(ns_old - 1) / float(ns_new - 1)) - j1.astype(dtype)
    xint = jnp.clip(xint, 0.0, 1.0)

    x1 = x_scaled[j1]
    x2 = x_scaled[j2]
    x_new_scaled = (1.0 - xint)[:, None] * x1 + xint[:, None] * x2

    # Unscale by scalxc on the new grid to return physical coefficients.
    x_new = x_new_scaled / scal_new

    # Zero odd-m modes on the axis (physical coefficients).
    is_odd = ((m.astype(jnp.int32) % 2) == 1).astype(dtype)
    axis_row = jnp.where(is_odd > 0, jnp.asarray(0.0, dtype=dtype), x_new[0])
    x_new = jnp.concatenate([axis_row[None, :], x_new[1:]], axis=0)
    return x_new


def interp_vmec_state(
    state_old: VMECState,
    *,
    m: Sequence[int] | np.ndarray | Any,
    n: Sequence[int] | np.ndarray | Any | None = None,
    lthreed: bool = True,
    lconm1: bool = True,
    ns_new: int,
) -> VMECState:
    """Interpolate a VMECState to a new radial resolution (ns_new)."""
    ns_new = int(ns_new)
    ns_old = int(state_old.layout.ns)
    K = int(state_old.layout.K)
    lasym = bool(state_old.layout.lasym)
    if ns_new <= 0:
        layout = StateLayout(ns=0, K=K, lasym=lasym)
        z = jnp.zeros((0, K), dtype=jnp.asarray(state_old.Rcos).dtype)
        return VMECState(layout=layout, Rcos=z, Rsin=z, Zcos=z, Zsin=z, Lcos=z, Lsin=z)

    m_arr = jnp.asarray(np.asarray(m, dtype=np.int32))
    if int(m_arr.shape[0]) != K:
        raise ValueError(f"m has shape {m_arr.shape}, expected (K,) with K={K}")

    n_arr_np = None
    if n is not None:
        n_arr_np = np.asarray(n, dtype=np.int32)
        if int(n_arr_np.shape[0]) != K:
            raise ValueError(f"n has shape {n_arr_np.shape}, expected (K,) with K={K}")

    layout = StateLayout(ns=ns_new, K=K, lasym=lasym)

    # For non-axisymmetric lasym=False stages, VMEC interpolates in internal
    # (m,n>=0) storage (interp.f) rather than directly on signed coefficients.
    # This avoids n-sign convention drift at stage boundaries.
    if (n_arr_np is not None) and (not lasym) and (int(np.max(np.abs(n_arr_np))) > 0):
        m_np = np.asarray(m_arr, dtype=np.int32)
        n_np = n_arr_np
        ns_old = int(state_old.layout.ns)
        mpol = int(np.max(m_np)) + 1
        ntor = int(np.max(np.abs(n_np)))
        nrange = ntor + 1

        idx_pos = -np.ones((mpol, nrange), dtype=np.int32)
        idx_neg = -np.ones((mpol, nrange), dtype=np.int32)
        for k, (mk, nk) in enumerate(zip(m_np, n_np)):
            if nk >= 0:
                idx_pos[int(mk), int(nk)] = int(k)
            else:
                idx_neg[int(mk), int(-nk)] = int(k)

        mscale = np.where(np.arange(mpol) == 0, 1.0, np.sqrt(2.0))
        nscale = np.where(np.arange(nrange) == 0, 1.0, np.sqrt(2.0))
        basis_norm = (1.0 / (mscale[:, None] * nscale[None, :])).astype(np.asarray(state_old.Rcos).dtype)
        mode_scale = (1.0 / basis_norm).astype(np.asarray(state_old.Rcos).dtype)

        def _signed_to_mn_cos(coeffs: Any) -> tuple[np.ndarray, np.ndarray]:
            coeffs_np = np.asarray(coeffs)
            rcc = np.zeros((ns_old, mpol, nrange), dtype=coeffs_np.dtype)
            rss = np.zeros_like(rcc)
            for mi in range(mpol):
                for ni in range(nrange):
                    kp = int(idx_pos[mi, ni])
                    if kp < 0:
                        continue
                    pos = coeffs_np[:, kp]
                    kn = int(idx_neg[mi, ni])
                    neg = coeffs_np[:, kn] if kn >= 0 else 0.0
                    rcc[:, mi, ni] = pos + neg
                    if (ni == 0) or (mi == 0):
                        rss[:, mi, ni] = 0.0
                    else:
                        rss[:, mi, ni] = pos - neg
            return rcc, rss

        def _signed_to_mn_sin(coeffs: Any) -> tuple[np.ndarray, np.ndarray]:
            coeffs_np = np.asarray(coeffs)
            zsc = np.zeros((ns_old, mpol, nrange), dtype=coeffs_np.dtype)
            zcs = np.zeros_like(zsc)
            for mi in range(mpol):
                for ni in range(nrange):
                    kp = int(idx_pos[mi, ni])
                    if kp < 0:
                        continue
                    pos = coeffs_np[:, kp]
                    kn = int(idx_neg[mi, ni])
                    neg = coeffs_np[:, kn] if kn >= 0 else 0.0
                    zsc_val = pos + neg
                    if ni == 0:
                        zcs[:, mi, ni] = 0.0
                    else:
                        zcs[:, mi, ni] = neg - pos
                        if mi == 0:
                            zsc_val = 0.0
                    zsc[:, mi, ni] = zsc_val
            return zsc, zcs

        def _mn_cos_to_signed(rcc: np.ndarray, rss: np.ndarray) -> np.ndarray:
            out = np.zeros((int(rcc.shape[0]), K), dtype=rcc.dtype)
            for mi in range(mpol):
                for ni in range(nrange):
                    kp = int(idx_pos[mi, ni])
                    if kp < 0:
                        continue
                    scale = mode_scale[mi, ni]
                    if (mi == 0) or (ni == 0):
                        pos = rcc[:, mi, ni] * scale
                    else:
                        pos = 0.5 * (rcc[:, mi, ni] + rss[:, mi, ni]) * scale
                    out[:, kp] = pos
                    kn = int(idx_neg[mi, ni])
                    if kn >= 0:
                        neg = 0.5 * (rcc[:, mi, ni] - rss[:, mi, ni]) * scale
                        out[:, kn] = neg
            return out

        def _mn_sin_to_signed(zsc: np.ndarray, zcs: np.ndarray) -> np.ndarray:
            out = np.zeros((int(zsc.shape[0]), K), dtype=zsc.dtype)
            for mi in range(mpol):
                for ni in range(nrange):
                    kp = int(idx_pos[mi, ni])
                    if kp < 0:
                        continue
                    scale = mode_scale[mi, ni]
                    kn = int(idx_neg[mi, ni])
                    if ni == 0:
                        pos = zsc[:, mi, ni] * scale
                    elif kn >= 0:
                        pos = 0.5 * (zsc[:, mi, ni] - zcs[:, mi, ni]) * scale
                    else:
                        pos = (-zcs[:, mi, ni] if mi == 0 else zsc[:, mi, ni]) * scale
                    out[:, kp] = pos
                    if kn >= 0:
                        neg = 0.5 * (zsc[:, mi, ni] + zcs[:, mi, ni]) * scale
                        out[:, kn] = neg
            return out

        def _interp_block(block: np.ndarray) -> np.ndarray:
            m_flat = np.repeat(np.arange(mpol, dtype=np.int32), nrange)
            interp = interp_vmec_radial_coeffs(block.reshape(ns_old, -1), m=m_flat, ns_new=ns_new)
            return np.asarray(interp).reshape(ns_new, mpol, nrange)

        rcc, rss = _signed_to_mn_cos(np.asarray(state_old.Rcos))
        zsc, zcs = _signed_to_mn_sin(np.asarray(state_old.Zsin))
        lsc, lcs = _signed_to_mn_sin(np.asarray(state_old.Lsin))

        if bool(lthreed) and bool(lconm1) and mpol > 1:
            rss_m1 = rss[:, 1, :].copy()
            zcs_m1 = zcs[:, 1, :].copy()
            rss[:, 1, :] = 0.5 * (rss_m1 + zcs_m1)
            zcs[:, 1, :] = 0.5 * (rss_m1 - zcs_m1)

        rcc = rcc * basis_norm[None, :, :]
        rss = rss * basis_norm[None, :, :]
        zsc = zsc * basis_norm[None, :, :]
        zcs = zcs * basis_norm[None, :, :]
        lsc = lsc * basis_norm[None, :, :]
        lcs = lcs * basis_norm[None, :, :]

        rcc_i = np.array(_interp_block(rcc), copy=True)
        rss_i = np.array(_interp_block(rss), copy=True)
        zsc_i = np.array(_interp_block(zsc), copy=True)
        zcs_i = np.array(_interp_block(zcs), copy=True)
        lsc_i = np.array(_interp_block(lsc), copy=True)
        lcs_i = np.array(_interp_block(lcs), copy=True)

        if bool(lthreed) and bool(lconm1) and mpol > 1:
            rss_m1 = rss_i[:, 1, :].copy()
            zcs_m1 = zcs_i[:, 1, :].copy()
            rss_i[:, 1, :] = rss_m1 + zcs_m1
            zcs_i[:, 1, :] = rss_m1 - zcs_m1

        Rcos_new = _mn_cos_to_signed(rcc_i, rss_i)
        Zsin_new = _mn_sin_to_signed(zsc_i, zcs_i)
        Lsin_new = _mn_sin_to_signed(lsc_i, lcs_i)

        return VMECState(
            layout=layout,
            Rcos=jnp.asarray(Rcos_new),
            Rsin=interp_vmec_radial_coeffs(state_old.Rsin, m=m_arr, ns_new=ns_new),
            Zcos=interp_vmec_radial_coeffs(state_old.Zcos, m=m_arr, ns_new=ns_new),
            Zsin=jnp.asarray(Zsin_new),
            Lcos=interp_vmec_radial_coeffs(state_old.Lcos, m=m_arr, ns_new=ns_new),
            Lsin=jnp.asarray(Lsin_new),
        )

    return VMECState(
        layout=layout,
        Rcos=interp_vmec_radial_coeffs(state_old.Rcos, m=m_arr, ns_new=ns_new),
        Rsin=interp_vmec_radial_coeffs(state_old.Rsin, m=m_arr, ns_new=ns_new),
        Zcos=interp_vmec_radial_coeffs(state_old.Zcos, m=m_arr, ns_new=ns_new),
        Zsin=interp_vmec_radial_coeffs(state_old.Zsin, m=m_arr, ns_new=ns_new),
        Lcos=interp_vmec_radial_coeffs(state_old.Lcos, m=m_arr, ns_new=ns_new),
        Lsin=interp_vmec_radial_coeffs(state_old.Lsin, m=m_arr, ns_new=ns_new),
    )

"""Mode-transform setup helpers for the VMEC residual iteration loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._compat import jnp

__all__ = [
    "ModeTransformHostProjection",
    "build_mode_transform_host_projection",
    "mn_cos_to_signed_host_projected",
    "mn_sin_to_signed_host_projected",
    "mode_diag_weights_mn",
    "mode_diag_weights_mn_np",
    "vmec_scalxc_from_s_np",
]


@dataclass(frozen=True)
class ModeTransformHostProjection:
    """Precomputed DGEMM projection matrices for host mode transforms."""

    ncoeff: int
    n_half: int
    A_cos: np.ndarray | None
    A_sin: np.ndarray | None
    AB_cos: np.ndarray | None
    AB_sin: np.ndarray | None


def build_mode_transform_host_projection(signed_maps: Any, *, ncoeff: int) -> ModeTransformHostProjection:
    """Build host projection matrices for VMEC ``mn`` to signed-mode maps."""

    ncoeff_i = int(ncoeff)
    if ncoeff_i <= 0:
        return ModeTransformHostProjection(
            ncoeff=ncoeff_i,
            n_half=0,
            A_cos=None,
            A_sin=None,
            AB_cos=None,
            AB_sin=None,
        )

    idx_pos_safe_np = np.asarray(signed_maps.idx_pos_safe_flat, dtype=np.int32)
    idx_neg_safe_np = np.asarray(signed_maps.idx_neg_safe_flat, dtype=np.int32)
    idx_all_np = np.concatenate([idx_pos_safe_np, idx_neg_safe_np], axis=0)
    n_flat_mn = len(idx_all_np)
    proj_mn_signed = np.zeros((n_flat_mn, ncoeff_i), dtype=np.float64)
    for j in range(n_flat_mn):
        ix = int(idx_all_np[j])
        if 0 <= ix < ncoeff_i:
            proj_mn_signed[j, ix] = 1.0

    n_half = n_flat_mn // 2  # = mpol * nrange
    m0_mask_np = np.asarray(signed_maps.m0_mask, dtype=bool)
    n0_mask_np = np.asarray(signed_maps.n0_mask, dtype=bool)
    mask_pos_flat_f64 = np.asarray(signed_maps.mask_pos_flat, dtype=np.float64)
    mask_neg_flat_f64 = np.asarray(signed_maps.mask_neg_flat, dtype=np.float64)

    # m0_mask has shape (mpol, 1), n0_mask has shape (1, nrange).
    # | broadcasts to (mpol, nrange); individual reshape would give wrong size.
    mn_bcast_shape = np.broadcast_shapes(m0_mask_np.shape, n0_mask_np.shape)
    m0n0_flat_1d = (m0_mask_np | n0_mask_np).reshape(-1)
    n0_flat_1d = np.broadcast_to(n0_mask_np, mn_bcast_shape).reshape(-1)
    m0_flat_1d = np.broadcast_to(m0_mask_np, mn_bcast_shape).reshape(-1)
    has_neg_flat = mask_neg_flat_f64 > 0.0
    mask_no_neg_flat = ~has_neg_flat & ~n0_flat_1d

    proj_pos = proj_mn_signed[:n_half]
    proj_neg = proj_mn_signed[n_half:]

    # cos: pos = 0.5*(cc+ss) unless m0|n0 -> pos = cc.
    cc_pos_fac = np.where(m0n0_flat_1d, 1.0, 0.5) * mask_pos_flat_f64
    ss_pos_fac = np.where(m0n0_flat_1d, 0.0, 0.5) * mask_pos_flat_f64
    A_cos = cc_pos_fac[:, None] * proj_pos + 0.5 * mask_neg_flat_f64[:, None] * proj_neg
    B_cos = ss_pos_fac[:, None] * proj_pos + (-0.5) * mask_neg_flat_f64[:, None] * proj_neg

    # sin: pos depends on n0/mask_no_neg/m0 category.
    sc_pos_fac = (
        np.where(
            n0_flat_1d,
            1.0,
            np.where(mask_no_neg_flat & m0_flat_1d, 0.0, np.where(mask_no_neg_flat & ~m0_flat_1d, 1.0, 0.5)),
        )
        * mask_pos_flat_f64
    )
    cs_pos_fac = (
        np.where(
            n0_flat_1d,
            0.0,
            np.where(mask_no_neg_flat & m0_flat_1d, -1.0, np.where(mask_no_neg_flat & ~m0_flat_1d, 0.0, -0.5)),
        )
        * mask_pos_flat_f64
    )
    A_sin = sc_pos_fac[:, None] * proj_pos + 0.5 * mask_neg_flat_f64[:, None] * proj_neg
    B_sin = cs_pos_fac[:, None] * proj_pos + 0.5 * mask_neg_flat_f64[:, None] * proj_neg

    return ModeTransformHostProjection(
        ncoeff=ncoeff_i,
        n_half=n_half,
        A_cos=A_cos,
        A_sin=A_sin,
        AB_cos=np.vstack([A_cos, B_cos]),
        AB_sin=np.vstack([A_sin, B_sin]),
    )


def mn_cos_to_signed_host_projected(cc: Any, ss: Any, projection: ModeTransformHostProjection) -> np.ndarray:
    """Apply a precomputed host DGEMM projection for cosine-like modes."""

    cc_np = np.asarray(cc, dtype=float)
    ns = int(cc_np.shape[0])
    if projection.ncoeff == 0:
        return np.zeros((ns, projection.ncoeff), dtype=cc_np.dtype)
    if ss is None:
        return cc_np.reshape(ns, -1) @ projection.A_cos
    cc_ss = np.concatenate([cc_np.reshape(ns, -1), np.asarray(ss, dtype=float).reshape(ns, -1)], axis=1)
    return cc_ss @ projection.AB_cos


def mn_sin_to_signed_host_projected(sc: Any, cs: Any, projection: ModeTransformHostProjection) -> np.ndarray:
    """Apply a precomputed host DGEMM projection for sine-like modes."""

    sc_np = np.asarray(sc, dtype=float)
    ns = int(sc_np.shape[0])
    if projection.ncoeff == 0:
        return np.zeros((ns, projection.ncoeff), dtype=sc_np.dtype)
    if cs is None:
        return sc_np.reshape(ns, -1) @ projection.A_sin
    sc_cs = np.concatenate([sc_np.reshape(ns, -1), np.asarray(cs, dtype=float).reshape(ns, -1)], axis=1)
    return sc_cs @ projection.AB_sin


def vmec_scalxc_from_s_np(s_in: Any, *, mpol: int, dtype: Any) -> np.ndarray:
    """NumPy VMEC ``scalxc`` for the non-differentiated CPU host-update path."""

    s_np = np.asarray(s_in, dtype=dtype)
    ns_local = int(s_np.shape[0])
    mpol_local = int(mpol)
    if ns_local == 0 or mpol_local <= 0:
        return np.zeros((ns_local, max(mpol_local, 0)), dtype=dtype)
    sqrts = np.sqrt(np.maximum(s_np, 0.0)).astype(dtype, copy=False)
    sqrts = np.array(sqrts, dtype=dtype, copy=True)
    sqrts[-1] = np.asarray(1.0, dtype=dtype)
    sq2 = sqrts[1] if ns_local >= 2 else np.asarray(1.0, dtype=dtype)
    scal_odd = 1.0 / np.maximum(sqrts, sq2)
    is_odd = (np.arange(mpol_local) % 2) == 1
    return np.where(is_odd[None, :], scal_odd[:, None], np.ones((ns_local, mpol_local), dtype=dtype))


def mode_diag_weights_mn(
    *,
    mpol: int,
    nrange: int,
    nfp: float,
    mode_diag_exponent: float,
    dtype: Any,
) -> Any:
    """Return JAX mode-diagonal weights on the VMEC ``m,n>=0`` grid."""

    m = jnp.arange(int(mpol), dtype=jnp.float64)
    n = jnp.arange(int(nrange), dtype=jnp.float64) * float(nfp)
    k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
    w = (1.0 + k2) ** (-float(mode_diag_exponent))
    return w.astype(dtype)


def mode_diag_weights_mn_np(
    *,
    mpol: int,
    nrange: int,
    nfp: float,
    mode_diag_exponent: float,
    dtype: Any,
) -> np.ndarray:
    """Return NumPy mode-diagonal weights on the VMEC ``m,n>=0`` grid."""

    m = np.arange(int(mpol), dtype=float)
    n = np.arange(int(nrange), dtype=float) * float(nfp)
    k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
    return ((1.0 + k2) ** (-float(mode_diag_exponent))).astype(dtype)

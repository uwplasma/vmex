"""Pure geometry helpers for residual-iteration solves."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ...._compat import jnp


def _m1_internal_to_physical_pair(rss, zcs, *, use_m1_pair_convert: bool):
    """Convert VMEC internal m=1 ``(rss, zcs)`` pair to physical coefficients."""
    if rss is None and zcs is None:
        return None, None
    if rss is None:
        zcs_arr = jnp.asarray(zcs)
        rss_arr = jnp.zeros_like(zcs_arr)
    else:
        rss_arr = jnp.asarray(rss)
    if zcs is None:
        zcs_arr = jnp.zeros_like(rss_arr)
    else:
        zcs_arr = jnp.asarray(zcs)
    if not use_m1_pair_convert:
        return rss_arr, zcs_arr
    tmp = rss_arr[:, 1, :]
    rss_arr = rss_arr.at[:, 1, :].set(tmp + zcs_arr[:, 1, :])
    zcs_arr = zcs_arr.at[:, 1, :].set(tmp - zcs_arr[:, 1, :])
    return rss_arr, zcs_arr


def _mn_sin_to_signed_physical_batch(
    sc,
    cs,
    *,
    scalxc_mn,
    mn_sin_to_signed_batch: Callable[[Any, Any], Any],
):
    """Scale sine/cosine blocks to VMEC physical coefficients, then map signed modes."""
    sc = jnp.asarray(sc) / scalxc_mn
    if cs is None:
        cs = jnp.zeros_like(sc)
    else:
        cs = jnp.asarray(cs) / scalxc_mn
    return mn_sin_to_signed_batch(sc, cs)


def _rz_norm_np(
    state,
    *,
    kp_idx_np,
    kn_idx_np,
    has_kn_np,
    m_idx_np,
    n_idx_np,
    include_rcc_np=None,
    lthreed: bool = True,
    lasym: bool = False,
) -> float:
    """Pure NumPy R/Z norm in VMEC internal signed-mode convention.

    This mirrors the JAX ``_rz_norm`` path used in ``solve.py`` while avoiding
    JAX dispatch on host-side preconditioner rebuilds.
    """
    kp_idx_np = np.asarray(kp_idx_np, dtype=np.int32)
    kn_idx_np = np.asarray(kn_idx_np, dtype=np.int32)
    has_kn_np = np.asarray(has_kn_np, dtype=bool)
    m_idx_np = np.asarray(m_idx_np, dtype=np.int32)
    n_idx_np = np.asarray(n_idx_np, dtype=np.int32)
    include_rcc_np = (
        np.asarray(include_rcc_np)
        if include_rcc_np is not None
        else ((m_idx_np > 0) | (n_idx_np > 0))
    )
    is_m0_1d = m_idx_np == 0
    is_n0_1d = n_idx_np == 0

    Rcos = np.asarray(state.Rcos)
    Zsin = np.asarray(state.Zsin)
    rpos = Rcos[:, kp_idx_np]
    zpos = Zsin[:, kp_idx_np]
    rneg = np.zeros_like(rpos)
    zneg = np.zeros_like(zpos)
    if bool(np.any(has_kn_np)):
        hk = has_kn_np
        rneg[:, hk] = Rcos[:, kn_idx_np[hk]]
        zneg[:, hk] = Zsin[:, kn_idx_np[hk]]

    rcc = rpos + np.where(has_kn_np, rneg, 0.0)
    zsc = np.where(has_kn_np, zpos + zneg, zpos)
    rss = np.where(is_n0_1d | is_m0_1d, 0.0, np.where(has_kn_np, rpos - rneg, 0.0))
    zsc = np.where((~is_n0_1d) & is_m0_1d, 0.0, zsc)
    zcs = np.where(is_n0_1d, 0.0, np.where(has_kn_np, zneg - zpos, -zpos))

    sl = slice(1, None)
    rz = float(
        np.dot(zsc[sl].ravel(), zsc[sl].ravel())
        + np.dot((include_rcc_np * rcc[sl]).ravel(), (include_rcc_np * rcc[sl]).ravel())
    )
    if lthreed:
        rz += float(np.dot(rss[sl].ravel(), rss[sl].ravel()) + np.dot(zcs[sl].ravel(), zcs[sl].ravel()))
    if lasym:
        Rsin = np.asarray(state.Rsin)
        Zcos = np.asarray(state.Zcos)
        rs_pos = Rsin[:, kp_idx_np]
        zc_pos = Zcos[:, kp_idx_np]
        rs_neg = np.zeros_like(rs_pos)
        zc_neg = np.zeros_like(zc_pos)
        if bool(np.any(has_kn_np)):
            hk = has_kn_np
            rs_neg[:, hk] = Rsin[:, kn_idx_np[hk]]
            zc_neg[:, hk] = Zcos[:, kn_idx_np[hk]]
        rsc = np.where(has_kn_np, rs_pos + rs_neg, np.where(is_n0_1d, rs_pos, np.where(is_m0_1d, 0.0, rs_pos)))
        rcs = np.where(has_kn_np, rs_neg - rs_pos, np.where(is_n0_1d, 0.0, np.where(is_m0_1d, -rs_pos, 0.0)))
        zcc = zc_pos + np.where(has_kn_np, zc_neg, 0.0)
        zss = np.where(is_n0_1d | is_m0_1d, 0.0, np.where(has_kn_np, zc_pos - zc_neg, 0.0))
        rz += float(
            np.dot(rsc[sl].ravel(), rsc[sl].ravel())
            + np.dot(rcs[sl].ravel(), rcs[sl].ravel())
            + np.dot(zcc[sl].ravel(), zcc[sl].ravel())
            + np.dot(zss[sl].ravel(), zss[sl].ravel())
        )
    return rz

"""Axis-current vacuum-field helpers for free-boundary VMEC.

The routines here model the net toroidal plasma current as a filament on the
magnetic axis.  They are kept separate from the NESTOR/controller code because
they are deterministic field-sampling utilities with their own VMEC2000 parity
tests.
"""

from __future__ import annotations

import numpy as np


def axis_current_field_simple(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    phi: np.ndarray,
    axis_r: np.ndarray,
    axis_z: np.ndarray,
    nfp: int,
    plascur: float,
    eps: float = 1.0e-18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-segment Biot-Savart axis-current field (VMEC++ simple path)."""

    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    phi = np.asarray(phi, dtype=float)
    axis_r = np.asarray(axis_r, dtype=float).reshape(-1)
    axis_z = np.asarray(axis_z, dtype=float).reshape(-1)

    if R.ndim != 2 or Z.shape != R.shape or phi.shape != R.shape:
        raise ValueError("R/Z/phi must be 2D arrays with matching shape")
    ntheta, nzeta = R.shape
    if axis_r.size != nzeta or axis_z.size != nzeta:
        raise ValueError(f"axis arrays must match nzeta={nzeta}: got {axis_r.size}, {axis_z.size}")

    mu0 = 4.0e-7 * np.pi
    # VMEC's `ctor/plascur` sign convention is opposite to the geometric
    # right-hand rule used by this explicit filament formula.
    current_amp = -float(plascur) / mu0
    if (not np.isfinite(current_amp)) or abs(current_amp) <= 0.0:
        z = np.zeros_like(R)
        return z, z, z

    phi_row = np.asarray(phi[0], dtype=float)
    x0 = axis_r * np.cos(phi_row)
    y0 = axis_r * np.sin(phi_row)
    z0 = axis_z

    nfp = max(1, int(nfp))
    axis_xyz = np.zeros((nfp * nzeta, 3), dtype=float)
    for p in range(nfp):
        ang = 2.0 * np.pi * float(p) / float(nfp)
        cp = np.cos(ang)
        sp = np.sin(ang)
        sl = slice(p * nzeta, (p + 1) * nzeta)
        axis_xyz[sl, 0] = cp * x0 - sp * y0
        axis_xyz[sl, 1] = sp * x0 + cp * y0
        axis_xyz[sl, 2] = z0
    axis_xyz = np.vstack([axis_xyz, axis_xyz[:1, :]])

    rq = R.reshape(-1)
    zq = Z.reshape(-1)
    pq = phi.reshape(-1)
    qxyz = np.stack([rq * np.cos(pq), rq * np.sin(pq), zq], axis=1)

    bxyz = np.zeros_like(qxyz, dtype=float)
    magnetic_field_scale = 1.0e-7 * current_amp * 2.0
    for sidx in range(axis_xyz.shape[0] - 1):
        p0 = axis_xyz[sidx]
        p1 = axis_xyz[sidx + 1]
        dseg = p1 - p0
        seg_len2 = float(np.dot(dseg, dseg))
        ri = qxyz - p0[None, :]
        rf = qxyz - p1[None, :]
        ri_norm = np.linalg.norm(ri, axis=1)
        rf_norm = np.linalg.norm(rf, axis=1)
        sum_rf = ri_norm + rf_norm
        denom = ri_norm * rf_norm * (sum_rf * sum_rf - seg_len2)
        mag = np.zeros_like(denom)
        mask = np.abs(denom) > float(eps)
        mag[mask] = magnetic_field_scale * sum_rf[mask] / denom[mask]
        bxyz += mag[:, None] * np.cross(dseg[None, :], ri)

    cp = np.cos(pq)
    sp = np.sin(pq)
    br = cp * bxyz[:, 0] + sp * bxyz[:, 1]
    bp = cp * bxyz[:, 1] - sp * bxyz[:, 0]
    bz = bxyz[:, 2]
    return br.reshape((ntheta, nzeta)), bp.reshape((ntheta, nzeta)), bz.reshape((ntheta, nzeta))


def axis_current_field_vmec_filament(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    axis_r: np.ndarray,
    axis_z: np.ndarray,
    nfp: int,
    plascur: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VMEC tolicu/belicu-equivalent axis-current field.

    This routine mirrors the VMEC call chain:
    - ``tolicu``: build axis filament points across field periods,
    - ``belicu``: evaluate ``bsc_b(single_coil, xpt, bvec)`` on boundary points.

    The Biot-Savart line-segment kernel follows ``bsc_b_coil_fil_loop`` from
    LIBSTELL ``bsc_T.f`` including ``eps_sq`` regularization.
    """

    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    axis_r = np.asarray(axis_r, dtype=float).reshape(-1)
    axis_z = np.asarray(axis_z, dtype=float).reshape(-1)

    if R.ndim != 2 or Z.shape != R.shape:
        raise ValueError("R/Z must be 2D arrays with matching shape")
    ntheta, nzeta = R.shape
    if axis_r.size != nzeta or axis_z.size != nzeta:
        raise ValueError(f"axis arrays must match nzeta={nzeta}: got {axis_r.size}, {axis_z.size}")

    mu0 = 4.0e-7 * np.pi
    # Match VMEC sign convention used in the legacy simple path and belicu call-chain.
    current = -float(plascur) / mu0
    if (not np.isfinite(current)) or abs(current) <= 0.0:
        z = np.zeros_like(R)
        return z, z, z

    nfper = max(1, int(nfp))
    nv = int(nzeta)
    # VMEC precal.f:
    #   if (nv == 1) then nvper = 64 else nvper = nfper
    # This keeps axis-current sampling non-degenerate in axisymmetric vacuum runs.
    nvper = 64 if nv == 1 else nfper
    alv = 2.0 * np.pi / float(max(1, nv))
    onp = 1.0 / float(nfper)
    alvp = onp * alv
    cosuv_1d = np.cos(alvp * np.arange(nv, dtype=float))
    sinuv_1d = np.sin(alvp * np.arange(nv, dtype=float))
    alp_per = 2.0 * np.pi / float(max(1, nvper))
    cosper = np.cos(alp_per * np.arange(nvper, dtype=float))
    sinper = np.sin(alp_per * np.arange(nvper, dtype=float))

    # tolicu.f: xpts(3,nvp), DO NOT CLOSE LOOP (wrap done in bsc_construct).
    xpts = np.zeros((3, nvper * nv), dtype=float)
    idx = 0
    for kper in range(nvper):
        cp = cosper[kper]
        sp = sinper[kper]
        for kv in range(nv):
            c = cosuv_1d[kv]
            s = sinuv_1d[kv]
            rr = axis_r[kv]
            xpts[0, idx] = rr * (cp * c - sp * s)
            xpts[1, idx] = rr * (sp * c + cp * s)
            xpts[2, idx] = axis_z[kv]
            idx += 1

    # bsc_construct_coil('fil_loop'): remove zero-length consecutive segments,
    # then wrap by appending the first point when needed.
    xnod_temp = np.zeros((3, xpts.shape[1] + 1), dtype=float)
    itemp = 1
    xnod_temp[:, 0] = xpts[:, 0]
    for i in range(1, xpts.shape[1]):
        vec = xnod_temp[:, itemp - 1] - xpts[:, i]
        if float(np.dot(vec, vec)) == 0.0:
            continue
        xnod_temp[:, itemp] = xpts[:, i]
        itemp += 1
    if itemp <= 1:
        z = np.zeros_like(R)
        return z, z, z
    vec_wrap = xnod_temp[:, itemp - 1] - xpts[:, 0]
    if float(np.dot(vec_wrap, vec_wrap)) == 0.0:
        itemp -= 1
    if itemp == 2:
        # Degenerate straight filament fallback.
        pass
    else:
        xnod_temp[:, itemp] = xpts[:, 0]
        itemp += 1
    xnod = np.asarray(xnod_temp[:, :itemp], dtype=float)
    nnode = int(xnod.shape[1])
    if nnode < 2:
        z = np.zeros_like(R)
        return z, z, z

    dxnod = xnod[:, 1:] - xnod[:, :-1]
    lsqnod = np.sum(dxnod * dxnod, axis=0)
    if not np.any(lsqnod > 0.0):
        z = np.zeros_like(R)
        return z, z, z
    eps_sq = np.finfo(float).eps * float(np.min(lsqnod[lsqnod > 0.0]))
    eps_sq = max(eps_sq, np.finfo(float).tiny)

    # belicu.f uses cosuv/sinuv tables, not arbitrary phi-grid values.
    cos1 = np.broadcast_to(cosuv_1d[None, :], (ntheta, nv)).reshape(-1)
    sin1 = np.broadcast_to(sinuv_1d[None, :], (ntheta, nv)).reshape(-1)
    rp = np.asarray(R, dtype=float).reshape(-1)
    zp = np.asarray(Z, dtype=float).reshape(-1)
    xobs = np.stack([rp * cos1, rp * sin1, zp], axis=1)

    # bsc_b_coil_fil_loop kernel, vectorized over observation points.
    capRv = xobs[:, None, :] - xnod.T[None, :, :]
    capR = np.sqrt(np.maximum(eps_sq, np.sum(capRv * capRv, axis=2)))
    R1p2 = capR[:, :-1] + capR[:, 1:]
    denom = np.maximum(R1p2 * R1p2 - lsqnod[None, :], eps_sq)
    Rfactor = 2.0 * R1p2 / (capR[:, :-1] * capR[:, 1:] * denom)
    crossv = np.cross(dxnod.T[None, :, :], capRv[:, :-1, :])
    braw = np.sum(crossv * Rfactor[:, :, None], axis=1)

    # bsc_b_coil: b = current * bsc_k2_def * braw, with bsc_k2_def = 1e-7.
    bxyz = (current * 1.0e-7) * braw
    br = cos1 * bxyz[:, 0] + sin1 * bxyz[:, 1]
    bp = -sin1 * bxyz[:, 0] + cos1 * bxyz[:, 1]
    bz = bxyz[:, 2]
    return br.reshape((ntheta, nv)), bp.reshape((ntheta, nv)), bz.reshape((ntheta, nv))


__all__ = ["axis_current_field_simple", "axis_current_field_vmec_filament"]

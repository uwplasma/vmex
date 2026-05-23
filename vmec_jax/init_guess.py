"""Initial guess construction.

VMEC builds its initial nested flux surfaces in `profil3d`, starting from
boundary Fourier coefficients and (optionally) axis curves. The logic is:

- For m>0 harmonics, scale boundary coefficients like rho**m with rho = sqrt(s).
- For m=0 harmonics, blend linearly between axis and boundary values in s.
- Lambda coefficients start at zero.

This module mirrors the `profil3d` behavior in VMEC2000 (external coefficient
convention, i.e. wout-like), while keeping the code path differentiable and
compatible with the rest of the JAX pipeline.
"""

from __future__ import annotations


import os
import numpy as np

from ._compat import jnp, has_jax
from .boundary import (
    BoundaryCoeffs,
    boundary_apply_vmec_m1_constraint,
    boundary_undo_vmec_m1_constraint,
)
from .grids import make_angle_grid
from .fourier import build_helical_basis, eval_fourier
from .namelist import InData
from .state import StateLayout, VMECState
from .static import VMECStatic
from .vmec_parity import internal_odd_from_physical_vmec_m1, vmec_m1_internal_to_physical_signed
from .vmec_realspace import (
    vmec_realspace_analysis,
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
)
from .vmec_tomnsp import vmec_trig_tables


def _read_axis_coeffs(indata: InData) -> dict[str, float | list[float]]:
    """Read axis arrays if present.

    VMEC supports axis series in a few naming conventions. For now we only
    look for the common modern VMEC names:

    - RAXIS_CC, RAXIS_CS
    - ZAXIS_CC, ZAXIS_CS

    Each may be a scalar or a list. We return the raw values.
    """
    out: dict[str, float | list[float]] = {}

    # Preferred explicit VMEC names.
    for key in ("RAXIS_CC", "RAXIS_CS", "ZAXIS_CC", "ZAXIS_CS"):
        v = indata.get(key, None)
        if v is not None:
            out[key] = v

    # Compatibility with legacy/short input names used by many cases,
    # where RAXIS maps to RAXIS_CC and ZAXIS maps to ZAXIS_CS.
    if "RAXIS_CC" not in out:
        v = indata.get("RAXIS", None)
        if v is not None:
            out["RAXIS_CC"] = v
    if "ZAXIS_CS" not in out:
        v = indata.get("ZAXIS", None)
        if v is not None:
            out["ZAXIS_CS"] = v
    return out


def _axis_array(values: float | list[float] | None, ntor: int, *, dtype):
    if values is None:
        return None
    if isinstance(values, list):
        arr = [float(v) for v in values]
    else:
        arr = [float(values)]
    if len(arr) < ntor + 1:
        arr = arr + [0.0] * (ntor + 1 - len(arr))
    else:
        arr = arr[: ntor + 1]
    return jnp.asarray(arr, dtype=dtype)


def _guess_axis_from_boundary(static: VMECStatic, boundary: BoundaryCoeffs):
    """Guess magnetic axis from boundary geometry.

    We use a simple VMEC-like heuristic: in each toroidal plane, set the axis
    to the midpoint of the (R,Z) bounding box of the LCFS cross-section, then
    Fourier-fit the resulting axis curve in zeta.
    """
    # Use a higher-resolution evaluation grid than the solver grid so the
    # mid-point estimate is consistent with VMEC's axis guess even on coarse
    # multigrid stages.
    ntheta_axis = max(256, int(static.cfg.ntheta) * 8)
    nzeta_axis = max(8, int(static.cfg.nzeta))
    grid = make_angle_grid(ntheta=ntheta_axis, nzeta=nzeta_axis, nfp=int(static.cfg.nfp))
    basis = build_helical_basis(static.modes, grid)
    Rb = np.asarray(eval_fourier(jnp.asarray(boundary.R_cos), jnp.asarray(boundary.R_sin), basis))
    Zb = np.asarray(eval_fourier(jnp.asarray(boundary.Z_cos), jnp.asarray(boundary.Z_sin), basis))

    R_axis = 0.5 * (Rb.min(axis=0) + Rb.max(axis=0))
    Z_axis = 0.5 * (Zb.min(axis=0) + Zb.max(axis=0))

    zeta = np.asarray(grid.zeta)
    n = np.arange(static.cfg.ntor + 1)
    cos_nz = np.cos(np.outer(n, zeta))
    sin_nz = np.sin(np.outer(n, zeta))

    delta = 2.0 / float(zeta.size)
    raxis_c = delta * (cos_nz @ R_axis)
    raxis_c[0] *= 0.5
    # VMEC convention: zaxis_cs uses a negative sine projection.
    zaxis_s = -delta * (sin_nz @ Z_axis)
    return jnp.asarray(raxis_c), jnp.asarray(zaxis_s)


def _boundary_cross_section_areas(static: VMECStatic, boundary: BoundaryCoeffs):
    basis = build_helical_basis(static.modes, static.grid)
    Rb = eval_fourier(jnp.asarray(boundary.R_cos), jnp.asarray(boundary.R_sin), basis)
    Zb = eval_fourier(jnp.asarray(boundary.Z_cos), jnp.asarray(boundary.Z_sin), basis)
    # signed polygon area, periodic closure, vectorized over zeta planes
    dA = Rb * jnp.roll(Zb, -1, axis=0) - jnp.roll(Rb, -1, axis=0) * Zb
    return 0.5 * jnp.sum(dA, axis=0)


def _boundary_is_traced(boundary: BoundaryCoeffs) -> bool:
    if not has_jax():
        return False
    try:
        import jax
    except Exception:
        return False
    tracer = jax.core.Tracer
    return isinstance(boundary.R_cos, tracer) or isinstance(boundary.R_sin, tracer) or isinstance(boundary.Z_cos, tracer) or isinstance(boundary.Z_sin, tracer)


def _any_value_is_traced(*values) -> bool:
    if not has_jax():
        return False
    try:
        import jax
    except Exception:
        return False
    tracer = jax.core.Tracer
    return any(isinstance(value, tracer) for value in values)


def _vmec_lflip_from_boundary(static: VMECStatic, boundary: BoundaryCoeffs) -> bool | None:
    """Return VMEC's initial lflip decision from boundary (or None if undecidable).

    VMEC's `readin.f` sets:

        lflip = (rtest*ztest < 0)

    with:
      - rtest = sum of m=1 `RBC` terms over n,
      - ztest = sum of m=1 `ZBS` terms over n.

    When either is (near) zero, the sign is numerically ambiguous; in that case
    we return None and let callers fall back to a more geometric heuristic.
    """
    m = np.asarray(static.modes.m, dtype=int)
    if not np.any(m == 1):
        return None
    mask = m == 1
    rtest = float(np.sum(np.asarray(boundary.R_cos, dtype=float)[mask]))
    ztest = float(np.sum(np.asarray(boundary.Z_sin, dtype=float)[mask]))
    if rtest == 0.0 or ztest == 0.0:
        return None
    return (rtest * ztest) < 0.0


def _vmec_lflip_from_boundary_jax(static: VMECStatic, boundary: BoundaryCoeffs):
    m_np = np.asarray(static.modes.m, dtype=int)
    idx = np.nonzero(m_np == 1)[0]
    if idx.size == 0:
        return jnp.asarray(False)
    idx = jnp.asarray(idx, dtype=jnp.int32)
    rtest = jnp.sum(jnp.take(jnp.asarray(boundary.R_cos), idx))
    ztest = jnp.sum(jnp.take(jnp.asarray(boundary.Z_sin), idx))
    is_ambig = jnp.logical_or(rtest == 0.0, ztest == 0.0)
    return jnp.where(is_ambig, False, rtest * ztest < 0.0)


def _flip_boundary_theta(static: VMECStatic, boundary: BoundaryCoeffs) -> BoundaryCoeffs:
    """VMEC `flip_theta`: apply θ -> π - θ to helical boundary coefficients."""
    m = np.asarray(static.modes.m)
    n = np.asarray(static.modes.n)
    key_to_k = {(int(mm), int(nn)): k for k, (mm, nn) in enumerate(zip(m, n))}

    R_cos = np.asarray(boundary.R_cos).copy()
    R_sin = np.asarray(boundary.R_sin).copy()
    Z_cos = np.asarray(boundary.Z_cos).copy()
    Z_sin = np.asarray(boundary.Z_sin).copy()

    R_cos_new = R_cos.copy()
    R_sin_new = R_sin.copy()
    Z_cos_new = Z_cos.copy()
    Z_sin_new = Z_sin.copy()

    for k, (mm, nn) in enumerate(zip(m, n)):
        m_i = int(mm)
        n_i = int(nn)
        if m_i == 0:
            continue
        k2 = key_to_k.get((m_i, -n_i))
        if k2 is None:
            continue
        # For the helical basis:
        #   cos(m(π-θ) - nζ) = (-1)^m cos(mθ + nζ) = (-1)^m cos(mθ - (-n)ζ)
        #   sin(m(π-θ) - nζ) = (-1)^(m+1) sin(mθ + nζ) = (-1)^(m+1) sin(mθ - (-n)ζ)
        fac_cos = -1.0 if (m_i % 2 == 1) else 1.0  # (-1)^m
        fac_sin = -fac_cos  # (-1)^(m+1)
        R_cos_new[k] = fac_cos * R_cos[k2]
        R_sin_new[k] = fac_sin * R_sin[k2]
        Z_cos_new[k] = fac_cos * Z_cos[k2]
        Z_sin_new[k] = fac_sin * Z_sin[k2]

    return BoundaryCoeffs(R_cos=R_cos_new, R_sin=R_sin_new, Z_cos=Z_cos_new, Z_sin=Z_sin_new)


def _flip_boundary_theta_arrays(static: VMECStatic, R_cos, R_sin, Z_cos, Z_sin):
    m_np = np.asarray(static.modes.m, dtype=int)
    n_np = np.asarray(static.modes.n, dtype=int)
    key_to_k = {(int(mm), int(nn)): k for k, (mm, nn) in enumerate(zip(m_np, n_np))}
    K = int(m_np.size)
    k2 = np.full((K,), -1, dtype=int)
    fac_cos = np.ones((K,), dtype=float)
    fac_sin = np.ones((K,), dtype=float)
    for k, (mm, nn) in enumerate(zip(m_np, n_np)):
        m_i = int(mm)
        if m_i == 0:
            continue
        k2_idx = key_to_k.get((m_i, -int(nn)))
        if k2_idx is None:
            continue
        k2[k] = int(k2_idx)
        fac_cos[k] = -1.0 if (m_i % 2 == 1) else 1.0
        fac_sin[k] = -fac_cos[k]

    k2_j = jnp.asarray(k2)
    fac_cos_j = jnp.asarray(fac_cos, dtype=R_cos.dtype)
    fac_sin_j = jnp.asarray(fac_sin, dtype=R_sin.dtype)
    mask = k2_j >= 0
    k2_safe = jnp.where(mask, k2_j, 0)

    R_cos_flip = jnp.where(mask, fac_cos_j * R_cos[k2_safe], R_cos)
    R_sin_flip = jnp.where(mask, fac_sin_j * R_sin[k2_safe], R_sin)
    Z_cos_flip = jnp.where(mask, fac_cos_j * Z_cos[k2_safe], Z_cos)
    Z_sin_flip = jnp.where(mask, fac_sin_j * Z_sin[k2_safe], Z_sin)
    return R_cos_flip, R_sin_flip, Z_cos_flip, Z_sin_flip


def _apply_m1_constraint(static: VMECStatic, boundary: BoundaryCoeffs) -> BoundaryCoeffs:
    """Apply VMEC m=1 constraint to boundary coefficients (internal basis)."""
    if not bool(getattr(static.cfg, "lconm1", True)):
        return boundary
    if int(static.cfg.ntor) == 0 and (not bool(static.cfg.lasym)):
        return boundary
    return boundary_apply_vmec_m1_constraint(
        boundary,
        static.modes,
        lthreed=int(static.cfg.ntor) > 0,
        lasym=bool(static.cfg.lasym),
    )


def _undo_m1_constraint_for_recompute(static: VMECStatic, boundary: BoundaryCoeffs) -> BoundaryCoeffs:
    """Undo VMEC's m=1 constraint for the boundary (used in axis recompute)."""
    if not bool(getattr(static.cfg, "lconm1", True)):
        return boundary
    if int(static.cfg.ntor) == 0 and (not bool(static.cfg.lasym)):
        return boundary
    return boundary_undo_vmec_m1_constraint(
        boundary,
        static.modes,
        lthreed=int(static.cfg.ntor) > 0,
        lasym=bool(static.cfg.lasym),
    )


def _blend_axis_m0_full(
    *,
    static: VMECStatic,
    s,
    Rcos,
    Rsin,
    Zcos,
    Zsin,
    Rcos_b,
    Rsin_b,
    Zcos_b,
    Zsin_b,
    raxis_cc,
    raxis_cs,
    zaxis_cc,
    zaxis_cs,
):
    """Blend m=0 modes between axis and boundary (profil3d convention)."""
    m0_idx = getattr(static, "m0_n_index", None)
    if m0_idx is None:
        m0_mask = static.modes.m == 0
        m0_idx = -np.ones((static.cfg.ntor + 1,), dtype=int)
        for k, (m_k, n_k) in enumerate(zip(static.modes.m, static.modes.n)):
            if m_k == 0 and n_k >= 0 and n_k < m0_idx.shape[0]:
                m0_idx[int(n_k)] = int(k)
    m0_idx = np.asarray(m0_idx, dtype=int)
    valid = m0_idx >= 0
    if not np.any(valid):
        return Rcos, Rsin, Zcos, Zsin
    k_idx = jnp.asarray(m0_idx[valid], dtype=jnp.int32)
    blend = s

    # Axis conventions mapped to vmec_jax helical storage:
    # - VMEC's internal `rcs/zcs` carry a minus sign from `raxis_cs/zaxis_cs`
    #   in `profil3d`, but vmec_jax stores helical sin-phase coefficients
    #   (sin(mθ-nζ)); for m=0 this introduces another minus sign.
    # - Net mapping in signed/helical storage:
    #     Rcos(m=0,n) <- +raxis_cc
    #     Rsin(m=0,n) <- +raxis_cs
    #     Zcos(m=0,n) <- +zaxis_cc
    #     Zsin(m=0,n) <- +zaxis_cs
    ax_Rcos = jnp.asarray(raxis_cc)[valid]
    ax_Rsin = jnp.asarray(raxis_cs)[valid]
    ax_Zcos = jnp.asarray(zaxis_cc)[valid]
    ax_Zsin = jnp.asarray(zaxis_cs)[valid]

    new_Rcos = (1.0 - blend)[:, None] * ax_Rcos[None, :] + blend[:, None] * Rcos_b[0, k_idx][None, :]
    new_Rsin = (1.0 - blend)[:, None] * ax_Rsin[None, :] + blend[:, None] * Rsin_b[0, k_idx][None, :]
    new_Zcos = (1.0 - blend)[:, None] * ax_Zcos[None, :] + blend[:, None] * Zcos_b[0, k_idx][None, :]
    new_Zsin = (1.0 - blend)[:, None] * ax_Zsin[None, :] + blend[:, None] * Zsin_b[0, k_idx][None, :]

    Rcos = jnp.asarray(Rcos).at[:, k_idx].set(new_Rcos)
    Rsin = jnp.asarray(Rsin).at[:, k_idx].set(new_Rsin)
    Zcos = jnp.asarray(Zcos).at[:, k_idx].set(new_Zcos)
    Zsin = jnp.asarray(Zsin).at[:, k_idx].set(new_Zsin)
    return Rcos, Rsin, Zcos, Zsin


def _recompute_axis_from_boundary(
    static: VMECStatic,
    boundary: BoundaryCoeffs,
    *,
    raxis_cc: np.ndarray,
    zaxis_cs: np.ndarray,
    signgs: int,
    n_grid: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Axis recompute heuristic to maximize min signed Jacobian per toroidal plane."""
    cfg = static.cfg
    boundary_recompute = _undo_m1_constraint_for_recompute(static, boundary)

    trig = vmec_trig_tables(
        ntheta=cfg.ntheta,
        nzeta=cfg.nzeta,
        nfp=cfg.nfp,
        mmax=cfg.mpol - 1,
        nmax=cfg.ntor,
        lasym=cfg.lasym,
    )
    ntheta1, ntheta2, ntheta3 = int(trig.ntheta1), int(trig.ntheta2), int(trig.ntheta3)

    coeff_cos = np.asarray(boundary_recompute.R_cos)[None, :]
    coeff_sin = np.asarray(boundary_recompute.R_sin)[None, :]
    R_lcfs_red = np.asarray(
        vmec_realspace_synthesis(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=static.modes, trig=trig)
    )[0]
    coeff_cos = np.asarray(boundary_recompute.Z_cos)[None, :]
    coeff_sin = np.asarray(boundary_recompute.Z_sin)[None, :]
    Z_lcfs_red = np.asarray(
        vmec_realspace_synthesis(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=static.modes, trig=trig)
    )[0]

    coeff_cos = np.asarray(boundary_recompute.R_cos)[None, :]
    coeff_sin = np.asarray(boundary_recompute.R_sin)[None, :]
    dR_dtheta_lcfs_red = np.asarray(
        vmec_realspace_synthesis_dtheta(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=static.modes, trig=trig)
    )[0]
    coeff_cos = np.asarray(boundary_recompute.Z_cos)[None, :]
    coeff_sin = np.asarray(boundary_recompute.Z_sin)[None, :]
    dZ_dtheta_lcfs_red = np.asarray(
        vmec_realspace_synthesis_dtheta(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=static.modes, trig=trig)
    )[0]

    ns = int(cfg.ns)
    ns12 = (ns + 1) // 2 - 1
    s_mid = ns12 / (ns - 1.0)
    delta_s = (ns - 1 - ns12) / (ns - 1.0)
    sqrt_s_mid = np.sqrt(s_mid)

    m = np.asarray(static.modes.m)
    scale_r = np.where(m > 0, sqrt_s_mid**m, 1.0)
    scale_other = np.where(m > 0, sqrt_s_mid**m, s_mid)

    Rcos_mid = scale_r * np.asarray(boundary.R_cos)
    Rsin_mid = scale_other * np.asarray(boundary.R_sin)
    Zcos_mid = scale_other * np.asarray(boundary.Z_cos)
    Zsin_mid = scale_other * np.asarray(boundary.Z_sin)

    # Blend m=0 using axis guess
    m0_mask = static.modes.m == 0
    for n in range(cfg.ntor + 1):
        k_candidates = np.where(m0_mask & (static.modes.n == n))[0]
        if k_candidates.size == 0:
            continue
        k = int(k_candidates[0])
        Rcos_mid[k] = (1.0 - s_mid) * raxis_cc[n] + s_mid * Rcos_mid[k]
        Zsin_mid[k] = (1.0 - s_mid) * zaxis_cs[n] + s_mid * Zsin_mid[k]

    R_half_red = np.asarray(
        vmec_realspace_synthesis(
            coeff_cos=Rcos_mid[None, :],
            coeff_sin=Rsin_mid[None, :],
            modes=static.modes,
            trig=trig,
        )
    )[0]
    Z_half_red = np.asarray(
        vmec_realspace_synthesis(
            coeff_cos=Zcos_mid[None, :],
            coeff_sin=Zsin_mid[None, :],
            modes=static.modes,
            trig=trig,
        )
    )[0]
    dR_dtheta_half_red = np.asarray(
        vmec_realspace_synthesis_dtheta(
            coeff_cos=Rcos_mid[None, :],
            coeff_sin=Rsin_mid[None, :],
            modes=static.modes,
            trig=trig,
        )
    )[0]
    dZ_dtheta_half_red = np.asarray(
        vmec_realspace_synthesis_dtheta(
            coeff_cos=Zcos_mid[None, :],
            coeff_sin=Zsin_mid[None, :],
            modes=static.modes,
            trig=trig,
        )
    )[0]

    dR_dtheta_half_red = 0.5 * (dR_dtheta_lcfs_red + dR_dtheta_half_red)
    dZ_dtheta_half_red = 0.5 * (dZ_dtheta_lcfs_red + dZ_dtheta_half_red)

    # Expand to full theta grid (ntheta1) for symmetric cases.
    R_lcfs = np.zeros((ntheta1, cfg.nzeta))
    Z_lcfs = np.zeros((ntheta1, cfg.nzeta))
    R_half = np.zeros((ntheta1, cfg.nzeta))
    Z_half = np.zeros((ntheta1, cfg.nzeta))
    dR_dtheta_lcfs = np.zeros((ntheta1, cfg.nzeta))
    dZ_dtheta_lcfs = np.zeros((ntheta1, cfg.nzeta))
    dR_dtheta_half = np.zeros((ntheta1, cfg.nzeta))
    dZ_dtheta_half = np.zeros((ntheta1, cfg.nzeta))

    # Fill reduced theta portion [0,pi].
    R_lcfs[:ntheta3, :] = R_lcfs_red
    Z_lcfs[:ntheta3, :] = Z_lcfs_red
    R_half[:ntheta3, :] = R_half_red
    Z_half[:ntheta3, :] = Z_half_red
    dR_dtheta_lcfs[:ntheta3, :] = dR_dtheta_lcfs_red
    dZ_dtheta_lcfs[:ntheta3, :] = dZ_dtheta_lcfs_red
    dR_dtheta_half[:ntheta3, :] = dR_dtheta_half_red
    dZ_dtheta_half[:ntheta3, :] = dZ_dtheta_half_red

    if not cfg.lasym:
        # Mirror: only interior points are reflected.
        # C++ equivalent:
        #   for l in [1, nThetaReduced-2]:
        #     l_reversed = (nThetaEven - l) % nThetaEven
        for iv in range(cfg.nzeta):
            ivminus = (cfg.nzeta - iv) % cfg.nzeta
            for iu_r in range(1, ntheta2 - 1):
                iu = (ntheta1 - iu_r) % ntheta1
                R_lcfs[iu, iv] = R_lcfs_red[iu_r, ivminus]
                Z_lcfs[iu, iv] = -Z_lcfs_red[iu_r, ivminus]
                R_half[iu, iv] = R_half_red[iu_r, ivminus]
                Z_half[iu, iv] = -Z_half_red[iu_r, ivminus]
                dR_dtheta_lcfs[iu, iv] = -dR_dtheta_lcfs_red[iu_r, ivminus]
                dZ_dtheta_lcfs[iu, iv] = dZ_dtheta_lcfs_red[iu_r, ivminus]
                dR_dtheta_half[iu, iv] = -dR_dtheta_half_red[iu_r, ivminus]
                dZ_dtheta_half[iu, iv] = dZ_dtheta_half_red[iu_r, ivminus]

    cosnv = np.asarray(trig.cosnv)  # (nzeta, nmax+1)
    sinnv = np.asarray(trig.sinnv)
    nscale = np.asarray(trig.nscale)
    r_axis = (raxis_cc[None, :] * cosnv / nscale[None, :]).sum(axis=1)
    z_axis = -(zaxis_cs[None, :] * sinnv / nscale[None, :]).sum(axis=1)

    new_r_axis = np.zeros_like(r_axis)
    new_z_axis = np.zeros_like(z_axis)

    ntheta, nzeta = R_lcfs.shape
    for k in range(nzeta // 2 + 1):
        rmin, rmax = float(R_lcfs[:, k].min()), float(R_lcfs[:, k].max())
        zmin, zmax = float(Z_lcfs[:, k].min()), float(Z_lcfs[:, k].max())

        dr = (rmax - rmin) / (n_grid - 1)
        dz = (zmax - zmin) / (n_grid - 1)

        r_guess = 0.5 * (rmax + rmin)
        z_guess = 0.5 * (zmax + zmin)

        dR_ds_half = (R_lcfs[:, k] - R_half[:, k]) / delta_s + r_axis[k]
        dZ_ds_half = (Z_lcfs[:, k] - Z_half[:, k]) / delta_s + z_axis[k]

        tau0 = dR_dtheta_half[:, k] * dZ_ds_half - dZ_dtheta_half[:, k] * dR_ds_half

        # Initialize to 0.0 so axis updates are only accepted when they improve
        # the minimum signed Jacobian above zero.
        min_tau_best = 0.0

        for iz in range(n_grid):
            z_grid = zmin + iz * dz
            if not cfg.lasym and (k == 0 or k == nzeta // 2):
                z_grid = 0.0
                if iz > 0:
                    break
            for ir in range(n_grid):
                r_grid = rmin + ir * dr
                tau = signgs * (tau0 - dR_dtheta_half[:, k] * z_grid + dZ_dtheta_half[:, k] * r_grid)
                min_tau = float(np.min(tau))
                if min_tau > min_tau_best:
                    min_tau_best = min_tau
                    r_guess = r_grid
                    z_guess = z_grid
                elif min_tau == min_tau_best and not cfg.lasym:
                    if abs(z_guess) > abs(z_grid):
                        z_guess = z_grid

        new_r_axis[k] = r_guess
        new_z_axis[k] = z_guess

    if not cfg.lasym:
        for k in range(1, nzeta // 2):
            k_rev = (nzeta - k) % nzeta
            new_r_axis[k_rev] = new_r_axis[k]
            new_z_axis[k_rev] = -new_z_axis[k]

    delta_v = 2.0 / float(nzeta)
    new_raxis_c = delta_v * (cosnv.T @ new_r_axis) / nscale
    new_zaxis_s = -delta_v * (sinnv.T @ new_z_axis) / nscale
    new_raxis_c[0] *= 0.5

    return new_raxis_c, new_zaxis_s


def extract_axis_override_from_state(state: VMECState, static: VMECStatic) -> dict[str, jnp.ndarray]:
    """Extract m=0 axis coefficients from a state in VMEC internal scaling."""
    m0_idx = getattr(static, "m0_n_index", None)
    if m0_idx is None:
        m0_idx = -np.ones((static.cfg.ntor + 1,), dtype=int)
        for k, (m_k, n_k) in enumerate(zip(static.modes.m, static.modes.n)):
            if int(m_k) == 0 and 0 <= int(n_k) < m0_idx.shape[0]:
                m0_idx[int(n_k)] = int(k)
    m0_idx = np.asarray(m0_idx, dtype=int)
    valid = m0_idx >= 0
    out = {
        "raxis_cc": jnp.zeros((static.cfg.ntor + 1,), dtype=jnp.asarray(state.Rcos).dtype),
        "raxis_cs": jnp.zeros((static.cfg.ntor + 1,), dtype=jnp.asarray(state.Rsin).dtype),
        "zaxis_cc": jnp.zeros((static.cfg.ntor + 1,), dtype=jnp.asarray(state.Zcos).dtype),
        "zaxis_cs": jnp.zeros((static.cfg.ntor + 1,), dtype=jnp.asarray(state.Zsin).dtype),
    }
    if np.any(valid):
        n_idx = jnp.asarray(np.nonzero(valid)[0], dtype=jnp.int32)
        k_idx = jnp.asarray(m0_idx[valid], dtype=jnp.int32)
        out["raxis_cc"] = out["raxis_cc"].at[n_idx].set(jnp.asarray(state.Rcos)[0, k_idx])
        out["raxis_cs"] = out["raxis_cs"].at[n_idx].set(jnp.asarray(state.Rsin)[0, k_idx])
        out["zaxis_cc"] = out["zaxis_cc"].at[n_idx].set(jnp.asarray(state.Zcos)[0, k_idx])
        out["zaxis_cs"] = out["zaxis_cs"].at[n_idx].set(jnp.asarray(state.Zsin)[0, k_idx])
    return out


def _recompute_axis_from_state_vmec(
    static: VMECStatic,
    *,
    pr1_even,
    pr1_odd,
    pz1_even,
    pz1_odd,
    pru_even,
    pru_odd,
    pzu_even,
    pzu_odd,
    signgs: int,
    n_grid: int = 61,
    trig=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port VMEC `guess_axis` from current parity fields.

    This mirrors `VMEC2000/Sources/Initialization_Cleanup/guess_axis.f`
    on the VMEC internal theta grid (`ntheta3`), including:
      - LCFS + mid-surface Jacobian proxy scan,
      - stellarator-symmetry extension over full theta when `lasym=False`,
      - per-zeta max-min Jacobian search on a `n_grid x n_grid` box,
      - Fourier reconstruction of axis coefficients with `nscale`.
    """
    cfg = static.cfg
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=cfg.ntheta,
            nzeta=cfg.nzeta,
            nfp=cfg.nfp,
            mmax=cfg.mpol - 1,
            nmax=cfg.ntor,
            lasym=cfg.lasym,
        )
    ntheta1, ntheta2, ntheta3 = int(trig.ntheta1), int(trig.ntheta2), int(trig.ntheta3)
    ns = int(cfg.ns)
    nzeta = int(cfg.nzeta)
    if ns < 2:
        raise ValueError("axis recompute requires ns >= 2")

    pr1_even = np.asarray(pr1_even, dtype=float)
    pr1_odd = np.asarray(pr1_odd, dtype=float)
    pz1_even = np.asarray(pz1_even, dtype=float)
    pz1_odd = np.asarray(pz1_odd, dtype=float)
    pru_even = np.asarray(pru_even, dtype=float)
    pru_odd = np.asarray(pru_odd, dtype=float)
    pzu_even = np.asarray(pzu_even, dtype=float)
    pzu_odd = np.asarray(pzu_odd, dtype=float)

    if pr1_even.ndim != 3 or pr1_even.shape[0] != ns:
        raise ValueError(f"Unexpected pr1_even shape {pr1_even.shape}; expected (ns, ntheta, nzeta)")
    if pr1_even.shape[2] != nzeta:
        raise ValueError(f"Unexpected pr1_even zeta size {pr1_even.shape[2]} != {nzeta}")
    ntheta_red = int(pr1_even.shape[1])
    if ntheta_red < ntheta3:
        raise ValueError(f"Unexpected reduced theta size {ntheta_red} < ntheta3={ntheta3}")

    hs = float(np.asarray(static.s)[1] - np.asarray(static.s)[0])
    sqrts = np.sqrt(np.maximum(np.asarray(static.s, dtype=float), 0.0))
    ns12 = (ns + 1) // 2 - 1
    ds = float((ns - 1 - ns12) * hs)

    ru0 = pru_even + sqrts[:, None, None] * pru_odd
    zu0 = pzu_even + sqrts[:, None, None] * pzu_odd

    r1b_red = pr1_even[ns - 1, :ntheta3, :] + pr1_odd[ns - 1, :ntheta3, :]
    z1b_red = pz1_even[ns - 1, :ntheta3, :] + pz1_odd[ns - 1, :ntheta3, :]
    r12_red = pr1_even[ns12, :ntheta3, :] + sqrts[ns12] * pr1_odd[ns12, :ntheta3, :]
    z12_red = pz1_even[ns12, :ntheta3, :] + sqrts[ns12] * pz1_odd[ns12, :ntheta3, :]
    ru12_red = 0.5 * (ru0[ns - 1, :ntheta3, :] + ru0[ns12, :ntheta3, :])
    zu12_red = 0.5 * (zu0[ns - 1, :ntheta3, :] + zu0[ns12, :ntheta3, :])

    r1b = np.zeros((ntheta1, nzeta), dtype=float)
    z1b = np.zeros((ntheta1, nzeta), dtype=float)
    r12 = np.zeros((ntheta1, nzeta), dtype=float)
    z12 = np.zeros((ntheta1, nzeta), dtype=float)
    ru12 = np.zeros((ntheta1, nzeta), dtype=float)
    zu12 = np.zeros((ntheta1, nzeta), dtype=float)

    r1b[:ntheta3, :] = r1b_red
    z1b[:ntheta3, :] = z1b_red
    r12[:ntheta3, :] = r12_red
    z12[:ntheta3, :] = z12_red
    ru12[:ntheta3, :] = ru12_red
    zu12[:ntheta3, :] = zu12_red

    if not bool(cfg.lasym):
        for iv in range(nzeta):
            ivminus = (nzeta - iv) % nzeta
            for iu in range(ntheta2, ntheta1):
                iu_r = ntheta1 - iu
                r1b[iu, iv] = r1b_red[iu_r, ivminus]
                z1b[iu, iv] = -z1b_red[iu_r, ivminus]
                r12[iu, iv] = r12_red[iu_r, ivminus]
                z12[iu, iv] = -z12_red[iu_r, ivminus]
                ru12[iu, iv] = -ru12_red[iu_r, ivminus]
                zu12[iu, iv] = zu12_red[iu_r, ivminus]

    rcom = np.zeros((nzeta,), dtype=float)
    zcom = np.zeros((nzeta,), dtype=float)
    axis_r0 = pr1_even[0, 0, :]
    axis_z0 = pz1_even[0, 0, :]

    lasym = bool(cfg.lasym)
    grid_count = max(int(n_grid), 0)
    grid_denom = float(max(grid_count - 1, 1))
    grid_frac = np.arange(grid_count, dtype=float) / grid_denom
    planes_to_compute = range(nzeta) if lasym else range(nzeta // 2 + 1)

    for iv in planes_to_compute:
        rmin = float(np.min(r1b[:, iv]))
        rmax = float(np.max(r1b[:, iv]))
        zmin = float(np.min(z1b[:, iv]))
        zmax = float(np.max(z1b[:, iv]))
        rbest = 0.5 * (rmax + rmin)
        zbest = 0.5 * (zmax + zmin)

        rs = (r1b[:, iv] - r12[:, iv]) / ds + axis_r0[iv]
        zs = (z1b[:, iv] - z12[:, iv]) / ds + axis_z0[iv]
        tau0 = ru12[:, iv] * zs - zu12[:, iv] * rs

        if grid_count > 0:
            r_grid = rmin + (rmax - rmin) * grid_frac
            if (not lasym) and (iv == 0 or iv == nzeta // 2):
                z_grid = np.zeros((1,), dtype=float)
            else:
                z_grid = zmin + (zmax - zmin) * grid_frac

            tau = int(signgs) * (
                tau0[None, None, :]
                - ru12[:, iv][None, None, :] * z_grid[:, None, None]
                + zu12[:, iv][None, None, :] * r_grid[None, :, None]
            )
            min_tau = np.min(tau, axis=2)
            max_tau = float(np.max(min_tau))

            if max_tau > 0.0:
                best_mask = min_tau == max_tau
                first_flat = int(np.argmax(best_mask.reshape(-1)))
                iz_best, ir_best = divmod(first_flat, int(r_grid.size))
                rbest = float(r_grid[ir_best])
                zbest = float(z_grid[iz_best])
                row_has_best = np.any(best_mask, axis=1)
                if np.any(row_has_best):
                    z_abs = np.abs(z_grid)
                    best_abs = float(np.min(z_abs[row_has_best]))
                    z_rows = np.nonzero(row_has_best & (z_abs == best_abs))[0]
                    if z_rows.size and abs(zbest) > abs(float(z_grid[int(z_rows[0])])):
                        zbest = float(z_grid[int(z_rows[0])])
            elif max_tau == 0.0:
                zero_rows = np.any(min_tau == 0.0, axis=1)
                if np.any(zero_rows):
                    z_abs = np.abs(z_grid)
                    better_rows = zero_rows & (z_abs < abs(zbest))
                    if np.any(better_rows):
                        best_abs = float(np.min(z_abs[better_rows]))
                        z_rows = np.nonzero(better_rows & (z_abs == best_abs))[0]
                        if z_rows.size:
                            zbest = float(z_grid[int(z_rows[0])])

        rcom[iv] = rbest
        zcom[iv] = zbest

    if not lasym:
        for iv in range(nzeta // 2 + 1, nzeta):
            src = nzeta - iv
            rcom[iv] = rcom[src]
            zcom[iv] = -zcom[src]

    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)
    nscale = np.asarray(trig.nscale, dtype=float)
    dzeta = 2.0 / float(nzeta)
    raxis_cc = dzeta * (cosnv.T @ rcom) / nscale
    zaxis_cs = -dzeta * (sinnv.T @ zcom) / nscale
    raxis_cs = -dzeta * (sinnv.T @ rcom) / nscale
    zaxis_cc = dzeta * (cosnv.T @ zcom) / nscale
    raxis_cc[0] *= 0.5
    zaxis_cc[0] *= 0.5
    if (nzeta % 2 == 0) and (nzeta // 2 <= int(cfg.ntor)):
        raxis_cc[nzeta // 2] *= 0.5
        zaxis_cc[nzeta // 2] *= 0.5
    return raxis_cc, raxis_cs, zaxis_cc, zaxis_cs


def _recompute_axis_from_state_vmec_jax(
    static: VMECStatic,
    *,
    pr1_even,
    pr1_odd,
    pz1_even,
    pz1_odd,
    pru_even,
    pru_odd,
    pzu_even,
    pzu_odd,
    signgs: int,
    n_grid: int = 61,
    trig=None,
):
    """JAX-compatible port of VMEC ``guess_axis`` from current parity fields."""
    cfg = static.cfg
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=cfg.ntheta,
            nzeta=cfg.nzeta,
            nfp=cfg.nfp,
            mmax=cfg.mpol - 1,
            nmax=cfg.ntor,
            lasym=cfg.lasym,
        )
    ntheta1, ntheta2, ntheta3 = int(trig.ntheta1), int(trig.ntheta2), int(trig.ntheta3)
    ns = int(cfg.ns)
    nzeta = int(cfg.nzeta)
    if ns < 2:
        raise ValueError("axis recompute requires ns >= 2")

    dtype = jnp.asarray(pr1_even).dtype
    pr1_even = jnp.asarray(pr1_even, dtype=dtype)
    pr1_odd = jnp.asarray(pr1_odd, dtype=dtype)
    pz1_even = jnp.asarray(pz1_even, dtype=dtype)
    pz1_odd = jnp.asarray(pz1_odd, dtype=dtype)
    pru_even = jnp.asarray(pru_even, dtype=dtype)
    pru_odd = jnp.asarray(pru_odd, dtype=dtype)
    pzu_even = jnp.asarray(pzu_even, dtype=dtype)
    pzu_odd = jnp.asarray(pzu_odd, dtype=dtype)

    hs = jnp.asarray(jnp.asarray(static.s)[1] - jnp.asarray(static.s)[0], dtype=dtype)
    sqrts = jnp.sqrt(jnp.maximum(jnp.asarray(static.s, dtype=dtype), jnp.asarray(0.0, dtype=dtype)))
    ns12 = (ns + 1) // 2 - 1
    ds = jnp.asarray((ns - 1 - ns12), dtype=dtype) * hs

    ru0 = pru_even + sqrts[:, None, None] * pru_odd
    zu0 = pzu_even + sqrts[:, None, None] * pzu_odd

    r1b_red = pr1_even[ns - 1, :ntheta3, :] + pr1_odd[ns - 1, :ntheta3, :]
    z1b_red = pz1_even[ns - 1, :ntheta3, :] + pz1_odd[ns - 1, :ntheta3, :]
    r12_red = pr1_even[ns12, :ntheta3, :] + sqrts[ns12] * pr1_odd[ns12, :ntheta3, :]
    z12_red = pz1_even[ns12, :ntheta3, :] + sqrts[ns12] * pz1_odd[ns12, :ntheta3, :]
    ru12_red = 0.5 * (ru0[ns - 1, :ntheta3, :] + ru0[ns12, :ntheta3, :])
    zu12_red = 0.5 * (zu0[ns - 1, :ntheta3, :] + zu0[ns12, :ntheta3, :])

    r1b = jnp.zeros((ntheta1, nzeta), dtype=dtype).at[:ntheta3, :].set(r1b_red)
    z1b = jnp.zeros((ntheta1, nzeta), dtype=dtype).at[:ntheta3, :].set(z1b_red)
    r12 = jnp.zeros((ntheta1, nzeta), dtype=dtype).at[:ntheta3, :].set(r12_red)
    z12 = jnp.zeros((ntheta1, nzeta), dtype=dtype).at[:ntheta3, :].set(z12_red)
    ru12 = jnp.zeros((ntheta1, nzeta), dtype=dtype).at[:ntheta3, :].set(ru12_red)
    zu12 = jnp.zeros((ntheta1, nzeta), dtype=dtype).at[:ntheta3, :].set(zu12_red)

    if not bool(cfg.lasym):
        for iv in range(nzeta):
            ivminus = (nzeta - iv) % nzeta
            for iu in range(ntheta2, ntheta1):
                iu_r = ntheta1 - iu
                r1b = r1b.at[iu, iv].set(r1b_red[iu_r, ivminus])
                z1b = z1b.at[iu, iv].set(-z1b_red[iu_r, ivminus])
                r12 = r12.at[iu, iv].set(r12_red[iu_r, ivminus])
                z12 = z12.at[iu, iv].set(-z12_red[iu_r, ivminus])
                ru12 = ru12.at[iu, iv].set(-ru12_red[iu_r, ivminus])
                zu12 = zu12.at[iu, iv].set(zu12_red[iu_r, ivminus])

    rcom = jnp.zeros((nzeta,), dtype=dtype)
    zcom = jnp.zeros((nzeta,), dtype=dtype)
    axis_r0 = pr1_even[0, 0, :]
    axis_z0 = pz1_even[0, 0, :]
    signgs_arr = jnp.asarray(signgs, dtype=dtype)
    grid_denom = max(int(n_grid) - 1, 1)
    frac = jnp.arange(int(n_grid), dtype=dtype) / jnp.asarray(grid_denom, dtype=dtype)

    planes_to_compute = range(nzeta) if bool(cfg.lasym) else range(nzeta // 2 + 1)
    for iv in planes_to_compute:
        rmin = jnp.min(r1b[:, iv])
        rmax = jnp.max(r1b[:, iv])
        zmin = jnp.min(z1b[:, iv])
        zmax = jnp.max(z1b[:, iv])
        r_mid = 0.5 * (rmax + rmin)
        z_mid = 0.5 * (zmax + zmin)

        r_grid = rmin + (rmax - rmin) * frac
        special_plane = (not bool(cfg.lasym)) and (iv == 0 or iv == nzeta // 2)
        if special_plane:
            z_grid = jnp.zeros((1,), dtype=dtype)
        else:
            z_grid = zmin + (zmax - zmin) * frac

        rs = (r1b[:, iv] - r12[:, iv]) / ds + axis_r0[iv]
        zs = (z1b[:, iv] - z12[:, iv]) / ds + axis_z0[iv]
        tau0 = ru12[:, iv] * zs - zu12[:, iv] * rs

        tau = signgs_arr * (
            tau0[None, None, :]
            - ru12[:, iv][None, None, :] * z_grid[:, None, None]
            + zu12[:, iv][None, None, :] * r_grid[None, :, None]
        )
        min_tau = jnp.min(tau, axis=2)
        max_tau = jnp.max(min_tau)
        best_mask = min_tau == max_tau
        if not bool(cfg.lasym):
            z_abs = jnp.abs(z_grid)[:, None]
            best_z_abs = jnp.min(jnp.where(best_mask, z_abs, jnp.inf))
            best_mask = jnp.logical_and(best_mask, z_abs == best_z_abs)
        flat_idx = jnp.argmax(best_mask.reshape(-1).astype(jnp.int32))
        iz_idx = flat_idx // int(r_grid.shape[0])
        ir_idx = flat_idx % int(r_grid.shape[0])
        # VMEC initializes each plane at the LCFS bounding-box midpoint and only
        # accepts scan points that improve the min-Jacobian proxy above zero. If
        # the best grid point is still negative, keep that midpoint instead of
        # moving to the least-bad point. Exact-zero ties may adjust z only.
        candidate_r = r_grid[ir_idx]
        candidate_z = z_grid[iz_idx]
        rbest = jnp.where(max_tau > 0.0, candidate_r, r_mid)
        zbest = jnp.where(max_tau >= 0.0, candidate_z, z_mid)
        rcom = rcom.at[iv].set(rbest)
        zcom = zcom.at[iv].set(zbest)

    if not bool(cfg.lasym):
        for iv in range(1, nzeta // 2):
            ivminus = (nzeta - iv) % nzeta
            rcom = rcom.at[ivminus].set(rcom[iv])
            zcom = zcom.at[ivminus].set(-zcom[iv])

    cosnv = jnp.asarray(trig.cosnv, dtype=dtype)
    sinnv = jnp.asarray(trig.sinnv, dtype=dtype)
    nscale = jnp.asarray(trig.nscale, dtype=dtype)
    dzeta = jnp.asarray(2.0 / float(nzeta), dtype=dtype)
    raxis_cc = dzeta * (cosnv.T @ rcom) / nscale
    zaxis_cs = -dzeta * (sinnv.T @ zcom) / nscale
    raxis_cs = -dzeta * (sinnv.T @ rcom) / nscale
    zaxis_cc = dzeta * (cosnv.T @ zcom) / nscale
    raxis_cc = raxis_cc.at[0].set(0.5 * raxis_cc[0])
    zaxis_cc = zaxis_cc.at[0].set(0.5 * zaxis_cc[0])
    if (nzeta % 2 == 0) and (nzeta // 2 <= int(cfg.ntor)):
        raxis_cc = raxis_cc.at[nzeta // 2].set(0.5 * raxis_cc[nzeta // 2])
        zaxis_cc = zaxis_cc.at[nzeta // 2].set(0.5 * zaxis_cc[nzeta // 2])
    return raxis_cc, raxis_cs, zaxis_cc, zaxis_cs


def _axis_parity_from_state_lasym(
    *,
    state: VMECState,
    static: VMECStatic,
    trig: VmecTrigTables | None = None,
) -> dict[str, np.ndarray]:
    """Compute VMEC parity pieces for LASYM axis recompute (cos/sin split)."""
    cfg = static.cfg
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=cfg.ntheta,
            nzeta=cfg.nzeta,
            nfp=cfg.nfp,
            mmax=cfg.mpol - 1,
            nmax=cfg.ntor,
            lasym=cfg.lasym,
        )
    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
        Rcos=Rcos,
        Zsin=Zsin,
        Rsin=Rsin,
        Zcos=Zcos,
        modes=static.modes,
        lthreed=bool(getattr(cfg, "lthreed", False)),
        lasym=bool(getattr(cfg, "lasym", False)),
        lconm1=bool(getattr(cfg, "lconm1", True)),
    )
    zcos = jnp.zeros_like(Rcos)
    zsin = jnp.zeros_like(Rsin)
    pr1_even = vmec_realspace_synthesis(
        coeff_cos=Rcos,
        coeff_sin=zsin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    pr1_odd = vmec_realspace_synthesis(
        coeff_cos=zcos,
        coeff_sin=Rsin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    pz1_even = vmec_realspace_synthesis(
        coeff_cos=Zcos,
        coeff_sin=zsin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    pz1_odd = vmec_realspace_synthesis(
        coeff_cos=zcos,
        coeff_sin=Zsin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    pru_even = vmec_realspace_synthesis_dtheta(
        coeff_cos=Rcos,
        coeff_sin=zsin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    pru_odd = vmec_realspace_synthesis_dtheta(
        coeff_cos=zcos,
        coeff_sin=Rsin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    pzu_even = vmec_realspace_synthesis_dtheta(
        coeff_cos=Zcos,
        coeff_sin=zsin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    pzu_odd = vmec_realspace_synthesis_dtheta(
        coeff_cos=zcos,
        coeff_sin=Zsin,
        modes=static.modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
    )
    return {
        "pr1_even": np.asarray(pr1_even),
        "pr1_odd": np.asarray(pr1_odd),
        "pz1_even": np.asarray(pz1_even),
        "pz1_odd": np.asarray(pz1_odd),
        "pru_even": np.asarray(pru_even),
        "pru_odd": np.asarray(pru_odd),
        "pzu_even": np.asarray(pzu_even),
        "pzu_odd": np.asarray(pzu_odd),
    }


def initial_guess_from_boundary(
    static: VMECStatic,
    boundary: BoundaryCoeffs,
    indata: InData | None = None,
    *,
    dtype=None,
    vmec_project: bool = False,
    infer_axis_if_missing: bool = True,
    axis_override: dict[str, object] | None = None,
) -> VMECState:
    """Build a VMECState initial guess from boundary coefficients.

    Parameters
    ----------
    static:
        Precomputed modes/grid/basis and radial coordinate.
    boundary:
        Boundary coefficients aligned with `static.modes`.
    indata:
        If provided, used to read optional axis specification. If absent or if
        the axis arrays are all zero, the axis is inferred from boundary m=0
        coefficients.
    dtype:
        Optional dtype for the returned arrays.
    vmec_project:
        If True, re-project the initial guess through VMEC's internal real-space
        grid (via ``vmec_realspace_synthesis`` + ``vmec_realspace_analysis``).
        This matches the VMEC grid/weighting used in parity diagnostics.
    axis_override:
        Optional explicit axis coefficients in VMEC internal scaling with keys
        ``raxis_cc``, ``raxis_cs``, ``zaxis_cc``, ``zaxis_cs``. When provided,
        these coefficients are used directly and missing-axis inference is
        skipped. This freezes the initialization branch choice for
        differentiated replay paths.
    """
    cfg = static.cfg
    K = static.modes.K
    layout = StateLayout(ns=cfg.ns, K=K, lasym=cfg.lasym)

    m = jnp.asarray(static.modes.m)
    s = jnp.asarray(static.s)
    if dtype is None:
        # Choose a dtype that avoids JAX warning spam.
        # VMEC expects float64; we default to float64 when x64 is enabled.
        if has_jax():
            try:
                import jax

                x64 = bool(jax.config.read("jax_enable_x64"))
            except Exception:
                x64 = True
            dtype = jnp.float64 if x64 else jnp.float32
        else:
            # numpy fallback: use float64 for VMEC parity
            import numpy as _np

            dtype = _np.float64

    boundary_use = boundary
    use_jax_boundary = bool(has_jax()) and (
        os.environ.get("VMEC_JAX_INIT_GUESS_JAX", "1").strip().lower() not in ("0", "false", "no")
    )
    if use_jax_boundary or _boundary_is_traced(boundary_use):
        lflip = _vmec_lflip_from_boundary_jax(static, boundary_use)
        R_cos_flip, R_sin_flip, Z_cos_flip, Z_sin_flip = _flip_boundary_theta_arrays(
            static,
            jnp.asarray(boundary_use.R_cos),
            jnp.asarray(boundary_use.R_sin),
            jnp.asarray(boundary_use.Z_cos),
            jnp.asarray(boundary_use.Z_sin),
        )
        boundary_use = BoundaryCoeffs(
            R_cos=jnp.where(lflip, R_cos_flip, jnp.asarray(boundary_use.R_cos)),
            R_sin=jnp.where(lflip, R_sin_flip, jnp.asarray(boundary_use.R_sin)),
            Z_cos=jnp.where(lflip, Z_cos_flip, jnp.asarray(boundary_use.Z_cos)),
            Z_sin=jnp.where(lflip, Z_sin_flip, jnp.asarray(boundary_use.Z_sin)),
        )
    else:
        lflip = _vmec_lflip_from_boundary(static, boundary_use)
        if lflip is None:
            # VMEC only flips when the m=1 rtest*ztest diagnostic is decisive.
            # If ambiguous, keep the input orientation.
            lflip = False
        if bool(lflip):
            boundary_use = _flip_boundary_theta(static, boundary_use)
    boundary_use = _apply_m1_constraint(static, boundary_use)


    # VMEC internal scaling: divide coefficients by mscale*nscale.
    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=cfg.ntheta,
            nzeta=cfg.nzeta,
            nfp=cfg.nfp,
            mmax=cfg.mpol - 1,
            nmax=cfg.ntor,
            lasym=cfg.lasym,
            dtype=dtype,
        )
    else:
        # Ensure cached trig tables match the requested resolution.
        if (int(trig.ntheta1) != int(cfg.ntheta)) or (int(trig.cosnv.shape[0]) != int(cfg.nzeta)):
            trig = vmec_trig_tables(
                ntheta=cfg.ntheta,
                nzeta=cfg.nzeta,
                nfp=cfg.nfp,
                mmax=cfg.mpol - 1,
                nmax=cfg.ntor,
                lasym=cfg.lasym,
                dtype=dtype,
            )
    if getattr(static, "mode_scale_internal", None) is not None:
        mode_scale = jnp.asarray(static.mode_scale_internal, dtype=dtype)
    else:
        m_idx = jnp.asarray(static.modes.m, dtype=jnp.int32)
        n_idx = jnp.asarray(static.modes.n, dtype=jnp.int32)
        n1 = jnp.abs(n_idx)
        mscale = jnp.asarray(trig.mscale, dtype=dtype)
        nscale = jnp.asarray(trig.nscale, dtype=dtype)
        mode_scale = (1.0 / (mscale[m_idx] * nscale[n1])).astype(dtype)  # internal scale

    # Base: broadcast boundary vectors to (ns,K) in internal convention.
    Rcos_b = (jnp.asarray(boundary_use.R_cos, dtype=dtype) * mode_scale)[None, :]
    Rsin_b = (jnp.asarray(boundary_use.R_sin, dtype=dtype) * mode_scale)[None, :]
    Zcos_b = (jnp.asarray(boundary_use.Z_cos, dtype=dtype) * mode_scale)[None, :]
    Zsin_b = (jnp.asarray(boundary_use.Z_sin, dtype=dtype) * mode_scale)[None, :]

    # Regularity scaling (profil3d): rho**m for m>0, s for m=0 (before axis blend).
    rho = jnp.sqrt(s)
    scale_r = jnp.where(m[None, :] > 0, rho[:, None] ** m[None, :], 1.0)
    scale_other = jnp.where(m[None, :] > 0, rho[:, None] ** m[None, :], s[:, None])
    Rcos = scale_r * Rcos_b
    Rsin = scale_other * Rsin_b
    Zcos = scale_other * Zcos_b
    Zsin = scale_other * Zsin_b

    # If user supplied a non-trivial axis spec, blend m=0 coefficients between
    # axis and boundary (linear in s).
    if indata is not None:
        ax = _read_axis_coeffs(indata)
        raxis_cc = _axis_array(ax.get("RAXIS_CC", None), cfg.ntor, dtype=dtype)
        raxis_cs = _axis_array(ax.get("RAXIS_CS", None), cfg.ntor, dtype=dtype)
        zaxis_cc = _axis_array(ax.get("ZAXIS_CC", None), cfg.ntor, dtype=dtype)
        zaxis_cs = _axis_array(ax.get("ZAXIS_CS", None), cfg.ntor, dtype=dtype)

        # Convert axis coefficients to VMEC internal scaling (1/(mscale*nscale)).
        n_arr = jnp.arange(cfg.ntor + 1, dtype=jnp.int32)
        nscale_axis = jnp.asarray(trig.nscale, dtype=dtype)
        axis_scale = (1.0 / nscale_axis[n_arr]).astype(dtype)
        if raxis_cc is not None:
            raxis_cc = raxis_cc * axis_scale
        if raxis_cs is not None:
            raxis_cs = raxis_cs * axis_scale
        if zaxis_cc is not None:
            zaxis_cc = zaxis_cc * axis_scale
        if zaxis_cs is not None:
            zaxis_cs = zaxis_cs * axis_scale

        if axis_override is not None:
            raxis_cc = jnp.asarray(axis_override.get("raxis_cc", jnp.zeros((cfg.ntor + 1,), dtype=dtype)), dtype=dtype)
            raxis_cs = jnp.asarray(axis_override.get("raxis_cs", jnp.zeros((cfg.ntor + 1,), dtype=dtype)), dtype=dtype)
            zaxis_cc = jnp.asarray(axis_override.get("zaxis_cc", jnp.zeros((cfg.ntor + 1,), dtype=dtype)), dtype=dtype)
            zaxis_cs = jnp.asarray(axis_override.get("zaxis_cs", jnp.zeros((cfg.ntor + 1,), dtype=dtype)), dtype=dtype)
            have_axis = True
            axis_from_indata = False
        else:
            # If axis arrays are all zero or missing, fall back to boundary-based axis.
            have_axis = False
            if raxis_cc is not None and np.any(np.asarray(raxis_cc) != 0.0):
                have_axis = True
            if raxis_cs is not None and np.any(np.asarray(raxis_cs) != 0.0):
                have_axis = True
            if zaxis_cc is not None and np.any(np.asarray(zaxis_cc) != 0.0):
                have_axis = True
            if zaxis_cs is not None and np.any(np.asarray(zaxis_cs) != 0.0):
                have_axis = True
            axis_from_indata = bool(have_axis)

        if not have_axis:
            if bool(infer_axis_if_missing):
                # VMEC-style axis guess (guess_axis) from the current parity fields.
                # Use the same VMEC trig tables and internal scaling as the solver.
                # VMEC sets signgs = -1 in readin and flips the boundary if needed.
                signgs_guess = -1

                # Undo the m=1 internal constraint before real-space synthesis.
                Rcos_phys, Zsin_phys, Rsin_phys, Zcos_phys = vmec_m1_internal_to_physical_signed(
                    Rcos=Rcos,
                    Zsin=Zsin,
                    Rsin=Rsin,
                    Zcos=Zcos,
                    modes=static.modes,
                    lthreed=bool(cfg.ntor > 0),
                    lasym=bool(cfg.lasym),
                    lconm1=bool(getattr(cfg, "lconm1", True)),
                )

                if getattr(static, "m_is_even", None) is not None:
                    mask_even = jnp.asarray(static.m_is_even, dtype=dtype)
                    mask_m1 = jnp.asarray(static.m_is_m1, dtype=dtype)
                    mask_odd_rest = jnp.asarray(static.m_is_odd_rest, dtype=dtype)
                else:
                    m_modes = np.asarray(static.modes.m, dtype=int)
                    mask_even = jnp.asarray((m_modes % 2) == 0, dtype=dtype)
                    mask_m1 = jnp.asarray(m_modes == 1, dtype=dtype)
                    mask_odd_rest = jnp.asarray((m_modes % 2 == 1) & (m_modes != 1), dtype=dtype)

                coeff_cos_stack = jnp.stack([Rcos_phys, Zcos_phys], axis=0)
                coeff_sin_stack = jnp.stack([Rsin_phys, Zsin_phys], axis=0)
                mask_stack = jnp.stack([mask_even, mask_m1, mask_odd_rest], axis=0)

                def _eval_stack(mask_stack):
                    coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
                    coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
                    return vmec_realspace_synthesis(
                        coeff_cos=coeff_cos,
                        coeff_sin=coeff_sin,
                        modes=static.modes,
                        trig=trig,
                        coeffs_internal=True,
                        apply_scalxc=False,
                        s=s,
                    )

                def _eval_stack_dtheta(mask_stack):
                    coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
                    coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
                    return vmec_realspace_synthesis_dtheta(
                        coeff_cos=coeff_cos,
                        coeff_sin=coeff_sin,
                        modes=static.modes,
                        trig=trig,
                        coeffs_internal=True,
                        apply_scalxc=False,
                        s=s,
                    )

                stack = _eval_stack(mask_stack)
                stack_t = _eval_stack_dtheta(mask_stack)

                pr1_even = stack[0][0]
                pz1_even = stack[0][1]
                pr1_m1 = stack[1][0]
                pz1_m1 = stack[1][1]
                pr1_rest = stack[2][0]
                pz1_rest = stack[2][1]

                pru_even = stack_t[0][0]
                pzu_even = stack_t[0][1]
                pru_m1 = stack_t[1][0]
                pzu_m1 = stack_t[1][1]
                pru_rest = stack_t[2][0]
                pzu_rest = stack_t[2][1]

                pr1_odd = internal_odd_from_physical_vmec_m1(
                    odd_m1_phys=pr1_m1,
                    odd_mge2_phys=pr1_rest,
                    s=s,
                )
                pz1_odd = internal_odd_from_physical_vmec_m1(
                    odd_m1_phys=pz1_m1,
                    odd_mge2_phys=pz1_rest,
                    s=s,
                )
                pru_odd = internal_odd_from_physical_vmec_m1(
                    odd_m1_phys=pru_m1,
                    odd_mge2_phys=pru_rest,
                    s=s,
                )
                pzu_odd = internal_odd_from_physical_vmec_m1(
                    odd_m1_phys=pzu_m1,
                    odd_mge2_phys=pzu_rest,
                    s=s,
                )

                if _any_value_is_traced(
                    pr1_even,
                    pr1_odd,
                    pz1_even,
                    pz1_odd,
                    pru_even,
                    pru_odd,
                    pzu_even,
                    pzu_odd,
                ):
                    raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = _recompute_axis_from_state_vmec_jax(
                        static,
                        pr1_even=pr1_even,
                        pr1_odd=pr1_odd,
                        pz1_even=pz1_even,
                        pz1_odd=pz1_odd,
                        pru_even=pru_even,
                        pru_odd=pru_odd,
                        pzu_even=pzu_even,
                        pzu_odd=pzu_odd,
                        signgs=signgs_guess,
                        trig=trig,
                    )
                    raxis_cc = jnp.asarray(raxis_cc, dtype=dtype) * axis_scale
                    raxis_cs = jnp.asarray(raxis_cs, dtype=dtype) * axis_scale
                    zaxis_cc = jnp.asarray(zaxis_cc, dtype=dtype) * axis_scale
                    zaxis_cs = jnp.asarray(zaxis_cs, dtype=dtype) * axis_scale
                    axis_from_indata = False
                else:
                    raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = _recompute_axis_from_state_vmec(
                        static,
                        pr1_even=pr1_even,
                        pr1_odd=pr1_odd,
                        pz1_even=pz1_even,
                        pz1_odd=pz1_odd,
                        pru_even=pru_even,
                        pru_odd=pru_odd,
                        pzu_even=pzu_even,
                        pzu_odd=pzu_odd,
                        signgs=signgs_guess,
                        trig=trig,
                    )

                    raxis_cc = jnp.asarray(raxis_cc, dtype=dtype) * axis_scale
                    raxis_cs = jnp.asarray(raxis_cs, dtype=dtype) * axis_scale
                    zaxis_cc = jnp.asarray(zaxis_cc, dtype=dtype) * axis_scale
                    zaxis_cs = jnp.asarray(zaxis_cs, dtype=dtype) * axis_scale
                    axis_from_indata = False
            else:
                # VMEC parity path: keep the explicit zero axis and let
                # `guess_axis`-style reset logic handle bad-Jacobian starts.
                raxis_cc = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
                raxis_cs = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
                zaxis_cc = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
                zaxis_cs = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
                axis_from_indata = True
            have_axis = True

        if have_axis:
            if raxis_cc is None:
                raxis_cc = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
            if raxis_cs is None:
                raxis_cs = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
            if zaxis_cc is None:
                zaxis_cc = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
            if zaxis_cs is None:
                zaxis_cs = jnp.zeros((cfg.ntor + 1,), dtype=dtype)

            Rcos, Rsin, Zcos, Zsin = _blend_axis_m0_full(
                static=static,
                s=s,
                Rcos=Rcos,
                Rsin=Rsin,
                Zcos=Zcos,
                Zsin=Zsin,
                Rcos_b=Rcos_b,
                Rsin_b=Rsin_b,
                Zcos_b=Zcos_b,
                Zsin_b=Zsin_b,
                raxis_cc=raxis_cc,
                raxis_cs=raxis_cs,
                zaxis_cc=zaxis_cc,
                zaxis_cs=zaxis_cs,
            )

            # VMEC only recomputes the axis when explicitly requested.
            if axis_from_indata and bool(indata.get_bool("LRECOMPUTE", False)):
                signgs_guess = -1
                new_raxis_cc = np.asarray(raxis_cc)
                new_zaxis_cs = np.asarray(zaxis_cs)
                for _ in range(3):
                    new_raxis_cc, new_zaxis_cs = _recompute_axis_from_boundary(
                        static,
                        boundary_use,
                        raxis_cc=new_raxis_cc,
                        zaxis_cs=new_zaxis_cs,
                        signgs=signgs_guess,
                    )
                raxis_cc = jnp.asarray(new_raxis_cc, dtype=dtype) * axis_scale
                zaxis_cs = jnp.asarray(new_zaxis_cs, dtype=dtype) * axis_scale
                Rcos, Rsin, Zcos, Zsin = _blend_axis_m0_full(
                    static=static,
                    s=s,
                    Rcos=Rcos,
                    Rsin=Rsin,
                    Zcos=Zcos,
                    Zsin=Zsin,
                    Rcos_b=Rcos_b,
                    Rsin_b=Rsin_b,
                    Zcos_b=Zcos_b,
                    Zsin_b=Zsin_b,
                    raxis_cc=raxis_cc,
                    raxis_cs=raxis_cs,
                    zaxis_cc=zaxis_cc,
                    zaxis_cs=zaxis_cs,
                )

    # Keep coefficients in VMEC's internal basis (mscale/nscale removed).

    if vmec_project:
        R_real = vmec_realspace_synthesis(
            coeff_cos=Rcos,
            coeff_sin=Rsin,
            modes=static.modes,
            trig=trig,
            coeffs_internal=True,
        )
        Z_real = vmec_realspace_synthesis(
            coeff_cos=Zcos,
            coeff_sin=Zsin,
            modes=static.modes,
            trig=trig,
            coeffs_internal=True,
        )
        # VMEC's symmetric projection keeps only the expected parity blocks.
        # For LASYM=True we must preserve both cos/sin components to avoid
        # wiping asymmetric contributions (e.g., Zcos, Rsin).
        parity_r = "both" if bool(cfg.lasym) else "cos"
        parity_z = "both" if bool(cfg.lasym) else "sin"
        Rcos, Rsin = vmec_realspace_analysis(f=R_real, modes=static.modes, trig=trig, parity=parity_r)
        Zcos, Zsin = vmec_realspace_analysis(f=Z_real, modes=static.modes, trig=trig, parity=parity_z)
        # vmec_realspace_analysis returns external (physical) coefficients;
        # convert back to VMEC internal scaling.
        Rcos = (Rcos * mode_scale[None, :]).astype(dtype)
        Rsin = (Rsin * mode_scale[None, :]).astype(dtype)
        Zcos = (Zcos * mode_scale[None, :]).astype(dtype)
        Zsin = (Zsin * mode_scale[None, :]).astype(dtype)

    Lcos = jnp.zeros((cfg.ns, K), dtype=dtype)
    Lsin = jnp.zeros((cfg.ns, K), dtype=dtype)

    return VMECState(layout=layout, Rcos=Rcos, Rsin=Rsin, Zcos=Zcos, Zsin=Zsin, Lcos=Lcos, Lsin=Lsin)

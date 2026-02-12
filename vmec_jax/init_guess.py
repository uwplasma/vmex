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

from dataclasses import dataclass

import numpy as np

from ._compat import jnp, has_jax
from .boundary import BoundaryCoeffs
from .grids import make_angle_grid
from .fourier import build_helical_basis, eval_fourier, eval_fourier_dtheta
from .namelist import InData
from .state import StateLayout, VMECState
from .static import VMECStatic
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
    zaxis_s = delta * (sin_nz @ Z_axis)
    return jnp.asarray(raxis_c), jnp.asarray(zaxis_s)


def _boundary_cross_section_areas(static: VMECStatic, boundary: BoundaryCoeffs) -> np.ndarray:
    basis = build_helical_basis(static.modes, static.grid)
    Rb = np.asarray(eval_fourier(jnp.asarray(boundary.R_cos), jnp.asarray(boundary.R_sin), basis))
    Zb = np.asarray(eval_fourier(jnp.asarray(boundary.Z_cos), jnp.asarray(boundary.Z_sin), basis))
    areas = []
    for k in range(Rb.shape[1]):
        R = Rb[:, k]
        Z = Zb[:, k]
        # signed polygon area, periodic closure
        area = 0.5 * np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z)
        areas.append(area)
    return np.asarray(areas)


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


def _apply_m1_constraint(static: VMECStatic, boundary: BoundaryCoeffs) -> BoundaryCoeffs:
    """Apply VMEC m=1 constraint to boundary coefficients (symmetric runs)."""
    if not bool(getattr(static.cfg, "lconm1", True)):
        return boundary
    if int(static.cfg.ntor) == 0:
        return boundary

    R_cos = np.asarray(boundary.R_cos).copy()
    R_sin = np.asarray(boundary.R_sin).copy()
    Z_cos = np.asarray(boundary.Z_cos).copy()
    Z_sin = np.asarray(boundary.Z_sin).copy()

    for k, m in enumerate(static.modes.m):
        if int(m) != 1:
            continue
        rbs = R_sin[k]
        zbc = Z_cos[k]
        R_sin[k] = 0.5 * (rbs + zbc)
        Z_cos[k] = 0.5 * (rbs - zbc)

    return BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)


def _undo_m1_constraint_for_recompute(static: VMECStatic, boundary: BoundaryCoeffs) -> BoundaryCoeffs:
    """Undo VMEC's m=1 constraint for the boundary (used in axis recompute)."""
    if not bool(getattr(static.cfg, "lconm1", True)):
        return boundary
    if int(static.cfg.ntor) == 0:
        return boundary

    R_cos = np.asarray(boundary.R_cos).copy()
    R_sin = np.asarray(boundary.R_sin).copy()
    Z_cos = np.asarray(boundary.Z_cos).copy()
    Z_sin = np.asarray(boundary.Z_sin).copy()

    for k, m in enumerate(static.modes.m):
        if int(m) != 1:
            continue
        rbs = R_sin[k]
        zbc = Z_cos[k]
        R_sin[k] = rbs + zbc
        Z_cos[k] = rbs - zbc

    return BoundaryCoeffs(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin)


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
    m0_mask = static.modes.m == 0
    blend = s
    for n in range(static.cfg.ntor + 1):
        k_candidates = jnp.where(m0_mask & (static.modes.n == n))[0]
        if k_candidates.size == 0:
            continue
        k = int(k_candidates[0])

        # profil3d axis sign conventions:
        #   rcc -> +raxis_cc
        #   rcs -> -raxis_cs
        #   zcc -> +zaxis_cc
        #   zcs -> -zaxis_cs
        ax_Rcos = raxis_cc[n]
        ax_Rsin = -raxis_cs[n]
        ax_Zcos = zaxis_cc[n]
        ax_Zsin = -zaxis_cs[n]

        new_Rcos = (1.0 - blend) * ax_Rcos + blend * Rcos_b[0, k]
        new_Rsin = (1.0 - blend) * ax_Rsin + blend * Rsin_b[0, k]
        new_Zcos = (1.0 - blend) * ax_Zcos + blend * Zcos_b[0, k]
        new_Zsin = (1.0 - blend) * ax_Zsin + blend * Zsin_b[0, k]

        if has_jax():
            Rcos = Rcos.at[:, k].set(new_Rcos)
            Rsin = Rsin.at[:, k].set(new_Rsin)
            Zcos = Zcos.at[:, k].set(new_Zcos)
            Zsin = Zsin.at[:, k].set(new_Zsin)
        else:
            Rcos = jnp.array(Rcos)
            Rsin = jnp.array(Rsin)
            Zcos = jnp.array(Zcos)
            Zsin = jnp.array(Zsin)
            Rcos[:, k] = new_Rcos
            Rsin[:, k] = new_Rsin
            Zcos[:, k] = new_Zcos
            Zsin[:, k] = new_Zsin
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

    for iv in range(nzeta):
        if (not bool(cfg.lasym)) and (iv > nzeta // 2):
            src = nzeta - iv
            rcom[iv] = rcom[src]
            zcom[iv] = -zcom[src]
            continue

        rmin = float(np.min(r1b[:, iv]))
        rmax = float(np.max(r1b[:, iv]))
        zmin = float(np.min(z1b[:, iv]))
        zmax = float(np.max(z1b[:, iv]))
        rbest = 0.5 * (rmax + rmin)
        zbest = 0.5 * (zmax + zmin)

        rs = (r1b[:, iv] - r12[:, iv]) / ds + axis_r0[iv]
        zs = (z1b[:, iv] - z12[:, iv]) / ds + axis_z0[iv]
        tau0 = ru12[:, iv] * zs - zu12[:, iv] * rs

        mintau = 0.0
        for iz in range(n_grid):
            zlim = zmin + (zmax - zmin) * float(iz) / float(max(n_grid - 1, 1))
            if (not bool(cfg.lasym)) and (iv == 0 or iv == nzeta // 2):
                zlim = 0.0
                if iz > 0:
                    break
            for ir in range(n_grid):
                rlim = rmin + (rmax - rmin) * float(ir) / float(max(n_grid - 1, 1))
                tau = int(signgs) * (tau0 - ru12[:, iv] * zlim + zu12[:, iv] * rlim)
                mintemp = float(np.min(tau))
                if mintemp > mintau:
                    mintau = mintemp
                    rbest = rlim
                    zbest = zlim
                elif mintemp == mintau:
                    if abs(zbest) > abs(zlim):
                        zbest = zlim

        rcom[iv] = rbest
        zcom[iv] = zbest

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


def initial_guess_from_boundary(
    static: VMECStatic,
    boundary: BoundaryCoeffs,
    indata: InData | None = None,
    *,
    dtype=None,
    vmec_project: bool = False,
    infer_axis_if_missing: bool = True,
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
    areas = _boundary_cross_section_areas(static, boundary_use)
    lflip = _vmec_lflip_from_boundary(static, boundary_use)
    if lflip is None:
        # VMEC only flips when the m=1 rtest*ztest diagnostic is decisive.
        # If ambiguous, keep the input orientation.
        lflip = False
    if bool(lflip):
        boundary_use = _flip_boundary_theta(static, boundary_use)
    boundary_use = _apply_m1_constraint(static, boundary_use)


    # VMEC internal scaling: divide coefficients by mscale*nscale.
    trig = vmec_trig_tables(
        ntheta=cfg.ntheta,
        nzeta=cfg.nzeta,
        nfp=cfg.nfp,
        mmax=cfg.mpol - 1,
        nmax=cfg.ntor,
        lasym=cfg.lasym,
        dtype=dtype,
    )
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
                # Heuristic fallback for non-VMEC workflows: infer the axis
                # from boundary geometry.
                raxis_cc, zaxis_cs = _guess_axis_from_boundary(static, boundary_use)
                # `_guess_axis_from_boundary` returns physical (wout-like) axis
                # coefficients; convert to VMEC internal scaling before blending.
                raxis_cc = jnp.asarray(raxis_cc, dtype=dtype) * axis_scale
                zaxis_cs = jnp.asarray(zaxis_cs, dtype=dtype) * axis_scale
                raxis_cs = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
                zaxis_cc = jnp.zeros((cfg.ntor + 1,), dtype=dtype)
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
                signgs_guess = 1 if np.median(areas) >= 0.0 else -1
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
                raxis_cc = jnp.asarray(new_raxis_cc, dtype=dtype)
                zaxis_cs = jnp.asarray(new_zaxis_cs, dtype=dtype)
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

    # Convert back to physical (wout) convention before returning.
    mode_scale_inv = jnp.where(mode_scale != 0, 1.0 / mode_scale, 0.0).astype(dtype)
    Rcos = Rcos * mode_scale_inv[None, :]
    Rsin = Rsin * mode_scale_inv[None, :]
    Zcos = Zcos * mode_scale_inv[None, :]
    Zsin = Zsin * mode_scale_inv[None, :]

    if vmec_project:
        if cfg.lasym:
            # Defer asymmetric support for the VMEC-grid projection.
            vmec_project = False
        else:
            R_real = vmec_realspace_synthesis(coeff_cos=Rcos, coeff_sin=Rsin, modes=static.modes, trig=trig)
            Z_real = vmec_realspace_synthesis(coeff_cos=Zcos, coeff_sin=Zsin, modes=static.modes, trig=trig)
            Rcos, Rsin = vmec_realspace_analysis(f=R_real, modes=static.modes, trig=trig, parity="cos")
            Zcos, Zsin = vmec_realspace_analysis(f=Z_real, modes=static.modes, trig=trig, parity="sin")
            Rcos = Rcos.astype(dtype)
            Rsin = Rsin.astype(dtype)
            Zcos = Zcos.astype(dtype)
            Zsin = Zsin.astype(dtype)

    Lcos = jnp.zeros((cfg.ns, K), dtype=dtype)
    Lsin = jnp.zeros((cfg.ns, K), dtype=dtype)

    return VMECState(layout=layout, Rcos=Rcos, Rsin=Rsin, Zcos=Zcos, Zsin=Zsin, Lcos=Lcos, Lsin=Lsin)

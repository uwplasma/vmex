"""Static (compile-time) data for vmec_jax.

This module defines a small "static" container that groups together data that
should be precomputed once per equilibrium problem and then reused inside
`jax.jit`'d kernels:

- mode table (m,n)
- angle grids (theta,zeta)
- helical basis tensors (cos/sin on the grid)
- radial grid (s in [0,1])

Keeping these pieces together prevents accidental recomputation and makes it
easier to write fast, end-to-end differentiable kernels later.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import TYPE_CHECKING

import numpy as np

from .config import VMECConfig
from .fourier import HelicalBasis, build_helical_basis
from .grids import AngleGrid, make_angle_grid
from .modes import ModeTable, vmec_mode_table

if TYPE_CHECKING:
    from .free_boundary import FreeBoundaryRuntimeState, MGridMetadata


def initial_free_boundary_state(cfg: VMECConfig):
    """Lazily construct free-boundary runtime state.

    Fixed-boundary runs dominate the CLI/examples path, so avoid importing the
    comparatively heavy free-boundary/NESTOR module unless ``LFREEB`` is active.
    The wrapper remains module-level so tests and downstream users can
    monkeypatch it as before.
    """

    from .free_boundary import initial_free_boundary_state as _impl

    return _impl(cfg)


@dataclass(frozen=True)
class VMECStatic:
    """Precomputed static data for a VMEC run."""

    cfg: VMECConfig
    modes: ModeTable
    grid: AngleGrid
    basis: HelicalBasis
    s: any  # (ns,) radial coordinate in [0,1]
    trig_vmec: any | None = None  # cached VMEC trig tables (fixaray parity)
    tomnsps_masks: any | None = None
    tomnsps_masks_edge: any | None = None
    # Cached mode arrays/masks for performance.
    m_np: np.ndarray | None = None
    n_np: np.ndarray | None = None
    m_is_even: np.ndarray | None = None
    m_is_odd: np.ndarray | None = None
    m_is_m0: np.ndarray | None = None
    m_is_m1: np.ndarray | None = None
    m_is_odd_rest: np.ndarray | None = None
    m_xmpq1: np.ndarray | None = None
    lambda_axis_copy_mask: np.ndarray | None = None
    m0_n_index: np.ndarray | None = None
    signed_maps: any | None = None
    mn_idx_m: np.ndarray | None = None
    mn_idx_n: np.ndarray | None = None
    mn_idx_kp: np.ndarray | None = None
    mn_idx_kn: np.ndarray | None = None
    mn_has_kn: np.ndarray | None = None
    mode_scale_internal: any | None = None
    free_boundary_state0: FreeBoundaryRuntimeState | None = None
    mgrid_metadata: MGridMetadata | None = None
    free_boundary_extcur: tuple[float, ...] | None = None


def build_static(
    cfg: VMECConfig,
    *,
    grid: AngleGrid | None = None,
    mgrid_metadata: MGridMetadata | None = None,
    free_boundary_extcur: tuple[float, ...] | None = None,
) -> VMECStatic:
    """Build the VMECStatic container from a parsed config.

    Parameters
    ----------
    grid:
        Optional override for the angular grid. This is used by parity kernels
        that must match VMEC's internal `ntheta1/2/3` conventions rather
        than the default `[0,2π)` endpoint-free grid.
    """
    modes = vmec_mode_table(cfg.mpol, cfg.ntor)
    if grid is None:
        grid = make_angle_grid(cfg.ntheta, cfg.nzeta, cfg.nfp, endpoint=False)
    basis = build_helical_basis(modes, grid)
    # Radial coordinate s = (i)/(ns-1). VMEC uses "s" = normalized toroidal flux.
    # Use a monotone [0,1] grid.
    if cfg.ns < 2:
        s = np.asarray([0.0], dtype=np.float64)
    else:
        s = np.linspace(0.0, 1.0, cfg.ns, dtype=np.float64)
    static_dtype = np.asarray(s).dtype
    tomnsps_masks = None
    tomnsps_masks_edge = None
    vmec_phase_stack = None
    vmec_phase_dtheta_stack = None
    vmec_phase_dzeta_stack = None
    try:
        from .vmec_tomnsp import vmec_trig_tables, tomnsps_masks as _tomnsps_masks

        trig_vmec = vmec_trig_tables(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(cfg.nfp),
            mmax=int(cfg.mpol) - 1,
            nmax=int(cfg.ntor),
            lasym=bool(cfg.lasym),
            dtype=static_dtype,
            cache=True,
        )
        tomnsps_masks = _tomnsps_masks(
            ns=int(cfg.ns),
            mpol=int(cfg.mpol),
            include_edge=False,
            dtype=static_dtype,
            cache=True,
        )
        tomnsps_masks_edge = _tomnsps_masks(
            ns=int(cfg.ns),
            mpol=int(cfg.mpol),
            include_edge=True,
            dtype=static_dtype,
            cache=True,
        )
        cache_phase = str(os.environ.get("VMEC_JAX_CACHE_VMEC_PHASE", "1")).lower() not in {"0", "false", "no"}
        if cache_phase and trig_vmec is not None:
            try:
                m = np.asarray(modes.m, dtype=int)
                n = np.asarray(modes.n, dtype=int)
                n1 = np.abs(n)
                sgn = np.where(n < 0, -1.0, 1.0)

                cosmu = np.asarray(trig_vmec.cosmu)
                sinmu = np.asarray(trig_vmec.sinmu)
                cosnv = np.asarray(trig_vmec.cosnv)
                sinnv = np.asarray(trig_vmec.sinnv)

                cosmu_m = cosmu[:, m].T
                sinmu_m = sinmu[:, m].T
                cosnv_n = cosnv[:, n1].T
                sinnv_n = sinnv[:, n1].T

                cos_phase = cosmu_m[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * sinnv_n[:, None, :]
                sin_phase = sinmu_m[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * sinnv_n[:, None, :]
                vmec_phase_stack = np.asarray(
                    np.concatenate([cos_phase, sin_phase], axis=0),
                    dtype=np.asarray(trig_vmec.cosmu).dtype,
                )

                cosmum = np.asarray(trig_vmec.cosmum)
                sinmum = np.asarray(trig_vmec.sinmum)
                dcos_phase = sinmum[:, m].T[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * cosmum[:, m].T[:, :, None] * sinnv_n[:, None, :]
                dsin_phase = cosmum[:, m].T[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * sinmum[:, m].T[:, :, None] * sinnv_n[:, None, :]
                vmec_phase_dtheta_stack = np.asarray(
                    np.concatenate([dcos_phase, dsin_phase], axis=0),
                    dtype=np.asarray(trig_vmec.cosmu).dtype,
                )

                cosnvn = np.asarray(trig_vmec.cosnvn)
                sinnvn = np.asarray(trig_vmec.sinnvn)
                dzcos_phase = cosmu_m[:, :, None] * sinnvn[:, n1].T[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * cosnvn[:, n1].T[:, None, :]
                dzsin_phase = sinmu_m[:, :, None] * sinnvn[:, n1].T[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * cosnvn[:, n1].T[:, None, :]
                vmec_phase_dzeta_stack = np.asarray(
                    np.concatenate([dzcos_phase, dzsin_phase], axis=0),
                    dtype=np.asarray(trig_vmec.cosmu).dtype,
                )

                trig_vmec = replace(
                    trig_vmec,
                    phase_stack=vmec_phase_stack,
                    phase_dtheta_stack=vmec_phase_dtheta_stack,
                    phase_dzeta_stack=vmec_phase_dzeta_stack,
                    phase_stack_m=modes.m,
                    phase_stack_n=modes.n,
                )
            except Exception:
                vmec_phase_stack = None
                vmec_phase_dtheta_stack = None
                vmec_phase_dzeta_stack = None
    except Exception:
        trig_vmec = None
        tomnsps_masks = None
        tomnsps_masks_edge = None
    m_np = np.asarray(modes.m, dtype=int)
    n_np = np.asarray(modes.n, dtype=int)
    m_is_even = (m_np % 2) == 0
    m_is_odd = ~m_is_even
    m_is_m0 = m_np == 0
    m_is_m1 = m_np == 1
    m_is_odd_rest = (m_np % 2 == 1) & (m_np != 1)
    m_xmpq1 = (m_np * (m_np - 1)).astype(float)
    m0_n_index = None
    signed_maps = None
    mn_idx_m = None
    mn_idx_n = None
    mn_idx_kp = None
    mn_idx_kn = None
    mn_has_kn = None
    try:
        nrange = int(cfg.ntor) + 1
        m0_n_index = -np.ones((nrange,), dtype=int)
        for k, (m_k, n_k) in enumerate(zip(m_np, n_np)):
            if m_k == 0 and n_k >= 0 and n_k < nrange:
                m0_n_index[int(n_k)] = int(k)
    except Exception:
        m0_n_index = None
    try:
        from .vmec_parity import signed_maps_from_modes

        signed_maps = signed_maps_from_modes(modes)
        idx_pos = np.asarray(signed_maps.idx_pos, dtype=np.int32)
        idx_neg = np.asarray(signed_maps.idx_neg, dtype=np.int32)
        mpol = int(cfg.mpol)
        nrange = int(cfg.ntor) + 1
        m_idx_list = []
        n_idx_list = []
        kp_idx_list = []
        kn_idx_list = []
        for m_i in range(mpol):
            for n_i in range(nrange):
                kp = int(idx_pos[m_i, n_i])
                if kp < 0:
                    continue
                m_idx_list.append(m_i)
                n_idx_list.append(n_i)
                kp_idx_list.append(kp)
                kn_idx_list.append(int(idx_neg[m_i, n_i]))
        mn_idx_m = np.asarray(m_idx_list, dtype=np.int32)
        mn_idx_n = np.asarray(n_idx_list, dtype=np.int32)
        mn_idx_kp = np.asarray(kp_idx_list, dtype=np.int32)
        mn_idx_kn = np.asarray(kn_idx_list, dtype=np.int32)
        mn_has_kn = mn_idx_kn >= 0
    except Exception:
        signed_maps = None
    lambda_axis_copy_mask = (m_np == 0) & (n_np > 0)
    return VMECStatic(
        cfg=cfg,
        modes=modes,
        grid=grid,
        basis=basis,
        s=s,
        trig_vmec=trig_vmec,
        tomnsps_masks=tomnsps_masks,
        tomnsps_masks_edge=tomnsps_masks_edge,
        m_np=m_np,
        n_np=n_np,
        m_is_even=m_is_even,
        m_is_odd=m_is_odd,
        m_is_m0=m_is_m0,
        m_is_m1=m_is_m1,
        m_is_odd_rest=m_is_odd_rest,
        m_xmpq1=m_xmpq1,
        lambda_axis_copy_mask=lambda_axis_copy_mask,
        m0_n_index=m0_n_index,
        signed_maps=signed_maps,
        mn_idx_m=mn_idx_m,
        mn_idx_n=mn_idx_n,
        mn_idx_kp=mn_idx_kp,
        mn_idx_kn=mn_idx_kn,
        mn_has_kn=mn_has_kn,
        mode_scale_internal=(
            None
            if trig_vmec is None
            else (
                1.0
                / (
                    np.asarray(trig_vmec.mscale, dtype=static_dtype)[m_np]
                    * np.asarray(trig_vmec.nscale, dtype=static_dtype)[np.abs(n_np)]
                )
            ).astype(static_dtype)
        ),
        free_boundary_state0=initial_free_boundary_state(cfg) if bool(cfg.lfreeb) else None,
        mgrid_metadata=mgrid_metadata,
        free_boundary_extcur=free_boundary_extcur,
    )

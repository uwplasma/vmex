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

from dataclasses import dataclass
import numpy as np

from ._compat import jnp
from .config import VMECConfig
from .fourier import HelicalBasis, build_helical_basis
from .grids import AngleGrid, make_angle_grid
from .modes import ModeTable, vmec_mode_table


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
    lambda_axis_copy_mask: np.ndarray | None = None


def build_static(cfg: VMECConfig, *, grid: AngleGrid | None = None) -> VMECStatic:
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
        s = jnp.asarray([0.0])
    else:
        s = jnp.linspace(0.0, 1.0, cfg.ns)
    tomnsps_masks = None
    tomnsps_masks_edge = None
    try:
        from .vmec_tomnsp import vmec_trig_tables, tomnsps_masks as _tomnsps_masks

        trig_vmec = vmec_trig_tables(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(cfg.nfp),
            mmax=int(cfg.mpol) - 1,
            nmax=int(cfg.ntor),
            lasym=bool(cfg.lasym),
            dtype=jnp.asarray(s).dtype,
            cache=True,
        )
        tomnsps_masks = _tomnsps_masks(
            ns=int(cfg.ns),
            mpol=int(cfg.mpol),
            include_edge=False,
            dtype=jnp.asarray(s).dtype,
            cache=True,
        )
        tomnsps_masks_edge = _tomnsps_masks(
            ns=int(cfg.ns),
            mpol=int(cfg.mpol),
            include_edge=True,
            dtype=jnp.asarray(s).dtype,
            cache=True,
        )
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
        lambda_axis_copy_mask=lambda_axis_copy_mask,
    )

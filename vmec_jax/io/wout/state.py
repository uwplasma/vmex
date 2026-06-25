"""Reconstruct VMEC solver state objects from WOUT files."""

from __future__ import annotations

from typing import Callable

import numpy as np

from vmec_jax.field import lamscale_from_phips
from vmec_jax.modes import vmec_mode_table
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.vmec_parity import vmec_m1_physical_to_internal_signed

from .flux import lambda_full_from_wout_half_mesh
from .schema import WoutData, assert_main_modes_match_wout


def state_from_wout(
    wout: WoutData,
    *,
    assert_main_modes_match_wout_func: Callable[..., None] = assert_main_modes_match_wout,
    lamscale_from_phips_func: Callable[..., float] = lamscale_from_phips,
) -> VMECState:
    """Build a :class:`~vmec_jax.state.VMECState` from WOUT Fourier coefficients.

    VMEC writes lambda in a backward-compatible half-mesh convention, so this
    reconstruction inverts ``wrout.f`` before building the internal solver state.
    """

    assert_main_modes_match_wout_func(wout=wout)
    layout = StateLayout(ns=wout.ns, K=int(wout.xm.size), lasym=bool(wout.lasym))

    ns = int(wout.ns)
    if ns < 2:
        s = np.asarray([0.0], dtype=float)
    else:
        s = np.linspace(0.0, 1.0, ns, dtype=float)
    lamscale = float(np.asarray(lamscale_from_phips_func(wout.phips, s)))

    # VMEC's `wout` stores phipf scaled by 2π*signgs. Internally, lambda scaling
    # uses the unscaled phipf (= phipf_internal). Align the reconstruction with
    # bcovar's bsupv formula by undoing the 2π*signgs factor here.
    scale = float(2.0 * np.pi * float(getattr(wout, "signgs", 1)))
    phipf_internal = (
        np.asarray(wout.phipf, dtype=float) / scale if scale != 0.0 else np.asarray(wout.phipf, dtype=float)
    )

    lmns_full = lambda_full_from_wout_half_mesh(
        lam_wout=np.asarray(wout.lmns),
        m_modes=np.asarray(wout.xm),
        s=s,
        phipf_internal=np.asarray(phipf_internal),
        lamscale=lamscale,
    )
    lmnc_full = lambda_full_from_wout_half_mesh(
        lam_wout=np.asarray(wout.lmnc),
        m_modes=np.asarray(wout.xm),
        s=s,
        phipf_internal=np.asarray(phipf_internal),
        lamscale=lamscale,
    )

    m_arr = np.asarray(wout.xm, dtype=int)
    n_arr = (np.asarray(wout.xn, dtype=int) // int(wout.nfp)).astype(int)
    sqrt2 = np.sqrt(2.0)
    mscale = np.where(m_arr == 0, 1.0, sqrt2)
    nscale = np.where(np.abs(n_arr) == 0, 1.0, sqrt2)
    mode_scale = (1.0 / (mscale * nscale))[None, :]

    Rcos = np.asarray(wout.rmnc) * mode_scale
    Rsin = np.asarray(wout.rmns) * mode_scale
    Zcos = np.asarray(wout.zmnc) * mode_scale
    Zsin = np.asarray(wout.zmns) * mode_scale

    modes = vmec_mode_table(wout.mpol, wout.ntor)
    lthreed = bool(int(wout.ntor) > 0)
    lasym = bool(wout.lasym)
    lconm1 = bool(lthreed or lasym)
    Rcos, Zsin, Rsin, Zcos = vmec_m1_physical_to_internal_signed(
        Rcos=Rcos,
        Zsin=Zsin,
        Rsin=Rsin,
        Zcos=Zcos,
        modes=modes,
        lthreed=lthreed,
        lasym=lasym,
        lconm1=lconm1,
    )

    return VMECState(
        layout=layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=np.asarray(lmnc_full) * mode_scale,
        Lsin=np.asarray(lmns_full) * mode_scale,
    )

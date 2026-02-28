"""JAX-native adapter from VMEC state to booz_xform_jax inputs."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._compat import jnp
from .energy import flux_profiles_from_indata
from .field import lamscale_from_phips
from .modes import vmec_mode_table, nyquist_mode_table_from_grid
from .profiles import eval_profiles
from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from .vmec_parity import vmec_m1_internal_to_physical_signed
from .vmec_realspace import vmec_realspace_analysis
from .vmec_tomnsp import vmec_trig_tables


@dataclass(frozen=True)
class BoozXformInputs:
    rmnc: Any
    zmns: Any
    lmns: Any
    bmnc: Any
    bsubumnc: Any
    bsubvmnc: Any
    iota: Any
    xm: Any
    xn: Any
    xm_nyq: Any
    xn_nyq: Any
    nfp: int
    bmns: Any | None = None
    bsubumns: Any | None = None
    bsubvmns: Any | None = None


def _mode_scale(m: Any, n: Any) -> Any:
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.asarray(m).dtype))
    mscale = jnp.where(m == 0, 1.0, sqrt2)
    nscale = jnp.where(jnp.abs(n) == 0, 1.0, sqrt2)
    return mscale * nscale


def _vmec_full_to_half(*, full: Any, m_modes: Any, s_full: Any) -> Any:
    full = jnp.asarray(full)
    m_modes = jnp.asarray(m_modes)
    s_full = jnp.asarray(s_full)
    ns_full = int(full.shape[0])
    if ns_full < 2:
        return full
    ns_in = ns_full - 1
    sqrt_s_full = jnp.sqrt(jnp.maximum(s_full, 0.0))
    sqrt_s_full = sqrt_s_full.at[0].set(1.0)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    sqrt_s_half = jnp.sqrt(jnp.maximum(s_half, 0.0))

    even_mask = (m_modes % 2) == 0
    odd_mask = ~even_mask

    even_val = 0.5 * (full[:-1, :] + full[1:, :])
    denom0 = sqrt_s_full[:-1, None]
    denom1 = sqrt_s_full[1:, None]
    odd_val = 0.5 * ((full[:-1, :] / denom0) + (full[1:, :] / denom1)) * sqrt_s_half[:, None]

    half = jnp.where(even_mask[None, :], even_val, odd_val)

    # m=1 axis extrapolation for the first half-mesh point.
    axis_mask = m_modes == 1
    if ns_full >= 3:
        rmnc_axis = (
            1.5 * full[1, :] / sqrt_s_full[1] - 0.5 * full[2, :] / sqrt_s_full[2]
        ) * sqrt_s_half[0]
        half0 = half[0, :]
        half0 = jnp.where(axis_mask, rmnc_axis, half0)
        half = half.at[0, :].set(half0)

    if half.shape[0] != ns_in:
        half = half[:ns_in, :]
    return half


def _lambda_wout_from_full_jax(
    *,
    lam_full: Any,
    m_modes: Any,
    phipf_internal: Any,
    lamscale: Any,
    s_full: Any,
) -> Any:
    lam_full = jnp.asarray(lam_full)
    m_modes = jnp.asarray(m_modes)
    phipf_internal = jnp.asarray(phipf_internal)
    s_full = jnp.asarray(s_full)
    ns = int(lam_full.shape[0])

    def _zero():
        return jnp.zeros_like(lam_full)

    def _nonzero():
        phipf_safe = jnp.where(phipf_internal == 0.0, 1.0, phipf_internal)
        lam_ext = lam_full * (lamscale / phipf_safe[:, None])

        if ns < 2:
            return jnp.zeros_like(lam_ext)

        hs = s_full[1] - s_full[0]
        idx = jnp.arange(ns + 1, dtype=lam_ext.dtype)
        sqrts_f = jnp.sqrt(jnp.maximum(hs * (idx - 1.0), 0.0))
        sqrts_f = sqrts_f.at[ns].set(1.0)
        shalf_f = jnp.sqrt(jnp.maximum(hs * jnp.abs(idx - 1.5), 0.0))

        sm_f = jnp.zeros_like(sqrts_f)
        sp_f = jnp.zeros_like(sqrts_f)
        i = jnp.arange(2, ns + 1)
        sm_val = jnp.where(sqrts_f[i] != 0.0, shalf_f[i] / sqrts_f[i], 0.0)
        sp_val = jnp.where(
            i < ns,
            jnp.where(sqrts_f[i] != 0.0, shalf_f[i + 1] / sqrts_f[i], 0.0),
            jnp.where(sqrts_f[i] != 0.0, 1.0 / sqrts_f[i], 0.0),
        )
        sm_f = sm_f.at[i].set(sm_val)
        sp_f = sp_f.at[i].set(sp_val)
        sp_f = sp_f.at[1].set(jnp.where(ns >= 2, sm_f[2], 0.0))

        lam_half = lam_ext
        mask_m_le1 = m_modes <= 1
        lam_half = lam_half.at[0, :].set(
            jnp.where(mask_m_le1, lam_half[1, :], lam_half[0, :])
        )

        even_mask = (m_modes % 2) == 0
        odd_mask = ~even_mask

        def body(js, arr):
            even_val = 0.5 * (arr[js, :] + arr[js - 1, :])
            odd_val = 0.5 * (sm_f[js + 1] * arr[js, :] + sp_f[js] * arr[js - 1, :])
            new_row = jnp.where(even_mask, even_val, odd_val)
            return arr.at[js, :].set(new_row)

        from jax import lax

        lam_half = lax.fori_loop(0, ns - 1, lambda i, a: body(ns - 1 - i, a), lam_half)
        lam_half = lam_half.at[0, :].set(jnp.zeros_like(lam_half[0, :]))
        return lam_half

    from jax import lax

    cond = jnp.asarray(lamscale).reshape(()) == 0.0
    return lax.cond(cond, lambda _: _zero(), lambda _: _nonzero(), operand=None)


def booz_xform_inputs_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    use_nyq_from_grid: bool = True,
) -> BoozXformInputs:
    """Construct booz_xform_jax inputs from a VMEC state using JAX kernels."""
    cfg = static.cfg
    nfp = int(cfg.nfp)

    main_modes = vmec_mode_table(int(cfg.mpol), int(cfg.ntor))
    if use_nyq_from_grid:
        nyq_modes = nyquist_mode_table_from_grid(
            mpol=int(cfg.mpol), ntor=int(cfg.ntor), ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta)
        )
    else:
        from .modes import nyquist_mode_table

        nyq_modes = nyquist_mode_table(int(cfg.mpol), int(cfg.ntor))

    m = jnp.asarray(main_modes.m)
    n = jnp.asarray(main_modes.n)
    mode_scale = _mode_scale(m, n)[None, :]

    lconm1 = bool(getattr(cfg, "lconm1", True))
    Rcos_use, Zsin_use, Rsin_use, Zcos_use = vmec_m1_internal_to_physical_signed(
        Rcos=state.Rcos,
        Zsin=state.Zsin,
        Rsin=state.Rsin,
        Zcos=state.Zcos,
        modes=main_modes,
        lthreed=bool(cfg.ntor > 0),
        lasym=bool(cfg.lasym),
        lconm1=lconm1,
    )

    rmnc_full = jnp.asarray(Rcos_use) * mode_scale
    zmns_full = jnp.asarray(Zsin_use) * mode_scale
    rmns_full = jnp.asarray(Rsin_use) * mode_scale
    zmnc_full = jnp.asarray(Zcos_use) * mode_scale

    lmnc_full = jnp.asarray(state.Lcos) * mode_scale
    lmns_full = jnp.asarray(state.Lsin) * mode_scale

    s_full = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s_full, signgs=signgs)
    lamscale = lamscale_from_phips(flux.phips, s_full)

    lmns_wout = _lambda_wout_from_full_jax(
        lam_full=lmns_full,
        m_modes=m,
        phipf_internal=flux.phipf,
        lamscale=lamscale,
        s_full=s_full,
    )
    lmnc_wout = _lambda_wout_from_full_jax(
        lam_full=lmnc_full,
        m_modes=m,
        phipf_internal=flux.phipf,
        lamscale=lamscale,
        s_full=s_full,
    )

    rmnc_half = _vmec_full_to_half(full=rmnc_full, m_modes=m, s_full=s_full)
    zmns_half = _vmec_full_to_half(full=zmns_full, m_modes=m, s_full=s_full)
    rmns_half = _vmec_full_to_half(full=rmns_full, m_modes=m, s_full=s_full)
    zmnc_half = _vmec_full_to_half(full=zmnc_full, m_modes=m, s_full=s_full)

    lmns_half = jnp.asarray(lmns_wout)[1:, :]
    lmnc_half = jnp.asarray(lmnc_wout)[1:, :]

    # iota on half mesh (with axis entry set to 0)
    s_half = jnp.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])], axis=0)
    prof = eval_profiles(indata, s_half)
    iotas = jnp.asarray(prof.get("iota", jnp.zeros_like(s_half)))
    iotas = iotas.at[0].set(0.0)
    iota_half = iotas[1:]

    wout_like = SimpleNamespace(
        phipf=flux.phipf,
        chipf=flux.chipf,
        phips=flux.phips,
        nfp=nfp,
        lasym=bool(cfg.lasym),
        signgs=int(signgs),
    )

    pres = jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))
    pres = pres.at[0].set(0.0)

    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        use_wout_bsup=False,
        use_wout_bsub_for_lambda=False,
        use_wout_bmag_for_bsq=False,
        use_vmec_synthesis=True,
        trig=None,
    )

    bsubu = jnp.asarray(bc.bsubu)
    bsubv = jnp.asarray(bc.bsubv)
    bsq = jnp.asarray(bc.bsq)
    pres_h = pres[:, None, None]
    bmod = jnp.sqrt(jnp.maximum(2.0 * (bsq - pres_h), 0.0))

    nyq_m = np.asarray(nyq_modes.m)
    nyq_n = np.asarray(nyq_modes.n)
    mmax = int(np.max(nyq_m)) if nyq_m.size else 0
    nmax = int(np.max(np.abs(nyq_n))) if nyq_n.size else 0
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        mmax=mmax,
        nmax=nmax,
        lasym=bool(cfg.lasym),
        dtype=jnp.asarray(bsubu).dtype,
        cache=True,
    )

    bsubumnc_full, bsubumns_full = vmec_realspace_analysis(
        f=bsubu, modes=nyq_modes, trig=trig, parity="both"
    )
    bsubvmnc_full, bsubvmns_full = vmec_realspace_analysis(
        f=bsubv, modes=nyq_modes, trig=trig, parity="both"
    )
    bmnc_full, bmns_full = vmec_realspace_analysis(
        f=bmod, modes=nyq_modes, trig=trig, parity="both"
    )

    bsubumnc = jnp.asarray(bsubumnc_full)[1:, :]
    bsubvmnc = jnp.asarray(bsubvmnc_full)[1:, :]
    bmnc = jnp.asarray(bmnc_full)[1:, :]

    bsubumns = jnp.asarray(bsubumns_full)[1:, :] if bool(cfg.lasym) else None
    bsubvmns = jnp.asarray(bsubvmns_full)[1:, :] if bool(cfg.lasym) else None
    bmns = jnp.asarray(bmns_full)[1:, :] if bool(cfg.lasym) else None

    return BoozXformInputs(
        rmnc=rmnc_half,
        zmns=zmns_half,
        lmns=lmns_half,
        bmnc=bmnc,
        bsubumnc=bsubumnc,
        bsubvmnc=bsubvmnc,
        iota=iota_half,
        xm=jnp.asarray(main_modes.m, dtype=jnp.int32),
        xn=jnp.asarray(main_modes.n * int(cfg.nfp), dtype=jnp.int32),
        xm_nyq=jnp.asarray(nyq_modes.m, dtype=jnp.int32),
        xn_nyq=jnp.asarray(nyq_modes.n * int(cfg.nfp), dtype=jnp.int32),
        nfp=nfp,
        bmns=bmns,
        bsubumns=bsubumns,
        bsubvmns=bsubvmns,
    )

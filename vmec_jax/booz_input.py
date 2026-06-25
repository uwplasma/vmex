"""JAX-native adapter from VMEC state to booz_xform_jax inputs."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._compat import jax, jnp
from .energy import FluxProfiles, _iotaf_from_iotas, flux_profiles_from_indata
from .field import lamscale_from_phips
from .modes import vmec_mode_table, nyquist_mode_table_from_grid
from .profiles import eval_profiles
from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from .vmec_parity import vmec_m1_internal_to_physical_signed
from .vmec_realspace import vmec_realspace_analysis
from .vmec_tomnsp import vmec_trig_tables


def _equilibrium_flux_profiles(
    *,
    state,
    static,
    indata,
    signgs: int,
    flux: FluxProfiles | None,
    profiles_half: dict | None,
):
    """Build Boozer inputs from equilibrium-consistent profiles when needed."""
    s_full = np.asarray(static.s)
    if int(s_full.shape[0]) < 2:
        s_half = s_full
    else:
        s_half = np.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])], axis=0)

    flux_local = flux if flux is not None else flux_profiles_from_indata(indata, s_full, signgs=signgs)
    # ``profiles_half`` is often used by optimization objectives to override
    # only pressure.  Do not treat it as a complete profile dictionary: Boozer
    # needs the equilibrium iota/current profiles too.  Merge caller-provided
    # values over the VMEC defaults so partial overrides cannot silently zero
    # iota and make field-line objectives evaluate the wrong trajectories.
    prof_default = eval_profiles(indata, s_half)
    if profiles_half is None:
        prof_local = prof_default
    else:
        prof_local = dict(prof_default)
        prof_local.update(profiles_half)
    pressure_local = prof_local.get("pressure", np.zeros_like(s_half))

    if int(indata.get_int("NCURR", 0)) != 1:
        return flux_local, prof_local

    from .driver import _final_flux_profiles_from_state

    flux_out, prof_out = _final_flux_profiles_from_state(
        indata=indata,
        static_in=static,
        state=state,
        signgs=int(signgs),
        flux_local=flux_local,
        prof_local=prof_local,
        pressure_local=pressure_local,
    )
    return flux_out, prof_out


@dataclass(frozen=True)
class BoozXformInputs:
    """Differentiable VMEC-to-Boozer input payload.

    The coefficient arrays follow the ``booz_xform_jax`` convention and are
    registered as a PyTree so QI/QS objectives can differentiate through the
    VMEC-to-Boozer preparation step.  Stellarator-symmetric channels are always
    present; ``lasym`` channels are optional and may be ``None``.
    """

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
    rmns: Any | None = None
    zmnc: Any | None = None
    lmnc: Any | None = None
    bmns: Any | None = None
    bsubumns: Any | None = None
    bsubvmns: Any | None = None

    def tree_flatten(self):
        children = (
            self.rmnc,
            self.zmns,
            self.lmns,
            self.bmnc,
            self.bsubumnc,
            self.bsubvmnc,
            self.iota,
            self.xm,
            self.xn,
            self.xm_nyq,
            self.xn_nyq,
            self.rmns,
            self.zmnc,
            self.lmnc,
            self.bmns,
            self.bsubumns,
            self.bsubvmns,
        )
        aux = int(self.nfp)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            rmnc,
            zmns,
            lmns,
            bmnc,
            bsubumnc,
            bsubvmnc,
            iota,
            xm,
            xn,
            xm_nyq,
            xn_nyq,
            rmns,
            zmnc,
            lmnc,
            bmns,
            bsubumns,
            bsubvmns,
        ) = children
        return cls(
            rmnc=rmnc,
            zmns=zmns,
            lmns=lmns,
            bmnc=bmnc,
            bsubumnc=bsubumnc,
            bsubvmnc=bsubvmnc,
            iota=iota,
            xm=xm,
            xn=xn,
            xm_nyq=xm_nyq,
            xn_nyq=xn_nyq,
            nfp=int(aux),
            rmns=rmns,
            zmnc=zmnc,
            lmnc=lmnc,
            bmns=bmns,
            bsubumns=bsubumns,
            bsubvmns=bsubvmns,
        )


if jax is not None:
    jax.tree_util.register_pytree_node_class(BoozXformInputs)


def _mode_scale(m: Any, n: Any) -> Any:
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.asarray(m).dtype))
    mscale = jnp.where(m == 0, 1.0, sqrt2)
    nscale = jnp.where(jnp.abs(n) == 0, 1.0, sqrt2)
    return mscale * nscale


def _jxbforce_nyquist_limits_from_trig(trig) -> tuple[int, int]:
    ntheta2 = int(getattr(trig, "ntheta2", 0))
    cosnv = jnp.asarray(getattr(trig, "cosnv"))
    nzeta = int(cosnv.shape[0]) if cosnv.ndim >= 1 else 0
    return max(ntheta2 - 1, 0), max(nzeta // 2, 0)


def _filter_bsubuv_jxbforce_parity_jax(
    *,
    bsubu_even: Any,
    bsubu_odd: Any,
    bsubv_even: Any,
    bsubv_odd: Any,
    trig,
    mmax_force: int,
    nmax_force: int,
    s: Any,
) -> tuple[Any, Any]:
    """JAX-friendly JXBFORCE low-pass filter for lasym=False bsub channels."""
    bsubu_even = jnp.asarray(bsubu_even)
    bsubu_odd = jnp.asarray(bsubu_odd)
    bsubv_even = jnp.asarray(bsubv_even)
    bsubv_odd = jnp.asarray(bsubv_odd)

    ns, ntheta, nzeta = bsubu_even.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        return bsubu_even[:, :nt2, :], bsubv_even[:, :nt2, :]

    bsubu_even_red = bsubu_even[:, :nt2, :]
    bsubu_odd_red = bsubu_odd[:, :nt2, :]
    bsubv_even_red = bsubv_even[:, :nt2, :]
    bsubv_odd_red = bsubv_odd[:, :nt2, :]

    cosmui = jnp.asarray(trig.cosmui)[:nt2, : mmax + 1]
    sinmui = jnp.asarray(trig.sinmui)[:nt2, : mmax + 1]
    cosmu = jnp.asarray(trig.cosmu)[:nt2, : mmax + 1]
    sinmu = jnp.asarray(trig.sinmu)[:nt2, : mmax + 1]
    cosnv = jnp.asarray(trig.cosnv)[:, : nmax + 1]
    sinnv = jnp.asarray(trig.sinnv)[:, : nmax + 1]

    r0scale = jnp.asarray(getattr(trig, "r0scale", 1.0), dtype=bsubu_even.dtype)
    dmult = jnp.full((mmax + 1, nmax + 1), 1.0, dtype=bsubu_even.dtype) / (r0scale**2)
    mnyq, nnyq = _jxbforce_nyquist_limits_from_trig(trig)
    if mnyq > 0 and mnyq <= mmax:
        dmult = dmult.at[mnyq, :].multiply(0.5)
    if nnyq > 0 and nnyq <= nmax:
        dmult = dmult.at[:, nnyq].multiply(0.5)

    s_full = jnp.asarray(s)
    pshalf = 0.5 * (s_full[1:] + s_full[:-1]) if int(s_full.shape[0]) > 1 else s_full
    pshalf = jnp.sqrt(jnp.maximum(jnp.concatenate([pshalf[:1], pshalf], axis=0) if int(s_full.shape[0]) > 1 else pshalf, 0.0))
    if int(pshalf.shape[0]) > 1:
        pshalf = pshalf.at[0].set(pshalf[1])

    odd_m = (jnp.arange(mmax + 1) % 2) == 1

    def _forward(f):
        f_theta_cos = jnp.einsum("sik,im->smk", f, cosmui, optimize=True)
        f_theta_sin = jnp.einsum("sik,im->smk", f, sinmui, optimize=True)
        coeff1 = jnp.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
        coeff2 = jnp.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)
        return coeff1 * dmult[None, :, :], coeff2 * dmult[None, :, :]

    def _inverse(coeff1, coeff2):
        tmp_cos = jnp.einsum("smn,im->sin", coeff1, cosmu, optimize=True)
        tmp_sin = jnp.einsum("smn,im->sin", coeff2, sinmu, optimize=True)
        return jnp.einsum("sin,kn->sik", tmp_cos, cosnv, optimize=True) + jnp.einsum(
            "sin,kn->sik", tmp_sin, sinnv, optimize=True
        )

    def _filter_field(f_even, f_odd):
        c1e, c2e = _forward(f_even)
        c1o, c2o = _forward(f_odd)
        scale = pshalf[:, None, None]
        odd_mask = odd_m[None, :, None]
        c1o = jnp.where(odd_mask, c1o / scale, c1o)
        c2o = jnp.where(odd_mask, c2o / scale, c2o)
        c1 = jnp.where(odd_mask, c1o, c1e)
        c2 = jnp.where(odd_mask, c2o, c2e)
        return _inverse(c1, c2)

    return _filter_field(bsubu_even_red, bsubu_odd_red), _filter_field(bsubv_even_red, bsubv_odd_red)


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


if jax is None:

    def _safe_sqrt_nonneg(x: Any) -> Any:
        return jnp.sqrt(jnp.maximum(x, 0.0))


else:

    @jax.custom_jvp
    def _safe_sqrt_nonneg(x: Any) -> Any:
        return jnp.sqrt(jnp.maximum(x, 0.0))

    @_safe_sqrt_nonneg.defjvp
    def _safe_sqrt_nonneg_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        y = _safe_sqrt_nonneg(x)
        safe = jnp.where(y > 0.0, 0.5 / y, 0.0)
        return y, t * safe


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
    trig: VmecTrigTables | None = None,
    flux: FluxProfiles | None = None,
    profiles_half: dict | None = None,
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
    flux, prof = _equilibrium_flux_profiles(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        flux=flux,
        profiles_half=profiles_half,
    )
    lamscale = getattr(flux, "lamscale", None)
    if lamscale is None:
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
    iotas = jnp.asarray(prof.get("iota", jnp.zeros_like(s_half)))
    iotas = iotas.at[0].set(0.0)
    iotaf = jnp.asarray(_iotaf_from_iotas(iotas, lrfp=bool(indata.get_bool("LRFP", False))))
    iota_half = iotas[1:]

    wout_like = SimpleNamespace(
        phipf=flux.phipf,
        chipf=flux.chipf,
        phips=flux.phips,
        iotas=iotas,
        iotaf=iotaf,
        nfp=nfp,
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        lasym=bool(cfg.lasym),
        signgs=int(signgs),
        ncurr=int(indata.get_int("NCURR", 0)),
        lcurrent=bool(int(indata.get_int("NCURR", 0)) == 1),
        flux_is_internal=True,
    )

    pres = jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))
    pres = pres.at[0].set(0.0)

    # Build or reuse trig tables for VMEC parity transforms.
    if trig is None:
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
            dtype=jnp.asarray(state.Rcos).dtype,
            cache=True,
        )

    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        use_wout_bsup=False,
        use_wout_bsub_for_lambda=False,
        use_wout_bmag_for_bsq=False,
        use_vmec_synthesis=True,
        trig=trig,
    )

    bsubu = jnp.asarray(bc.bsubu)
    bsubv = jnp.asarray(bc.bsubv)
    if not bool(cfg.lasym):
        pshalf = jnp.sqrt(jnp.maximum(0.5 * (s_full[1:] + s_full[:-1]), 0.0))
        pshalf = jnp.concatenate([pshalf[:1], pshalf], axis=0)
        if int(pshalf.shape[0]) > 1:
            pshalf = pshalf.at[0].set(pshalf[1])
        pshalf = pshalf[:, None, None]
        bsubu, bsubv = _filter_bsubuv_jxbforce_parity_jax(
            bsubu_even=bsubu,
            bsubu_odd=pshalf * bsubu,
            bsubv_even=bsubv,
            bsubv_odd=pshalf * bsubv,
            trig=trig,
            mmax_force=max(int(cfg.mpol) - 1, 0),
            nmax_force=int(cfg.ntor),
            s=s_full,
        )
    bsq = jnp.asarray(bc.bsq)
    pres_h = pres[:, None, None]
    bmod = _safe_sqrt_nonneg(2.0 * (bsq - pres_h))

    parity = "both" if bool(cfg.lasym) else "cos"
    bsubumnc_full, bsubumns_full = vmec_realspace_analysis(
        f=bsubu, modes=nyq_modes, trig=trig, parity=parity
    )
    bsubvmnc_full, bsubvmns_full = vmec_realspace_analysis(
        f=bsubv, modes=nyq_modes, trig=trig, parity=parity
    )
    bmnc_full, bmns_full = vmec_realspace_analysis(
        f=bmod, modes=nyq_modes, trig=trig, parity=parity
    )

    bsubumnc = jnp.asarray(bsubumnc_full)[1:, :]
    bsubvmnc = jnp.asarray(bsubvmnc_full)[1:, :]
    bmnc = jnp.asarray(bmnc_full)[1:, :]

    if not bool(cfg.lasym):
        mask_bsub = (jnp.asarray(nyq_modes.m) >= int(cfg.mpol)) | (jnp.abs(jnp.asarray(nyq_modes.n)) > int(cfg.ntor))
        bsubumnc = jnp.where(mask_bsub[None, :], 0.0, bsubumnc)
        bsubvmnc = jnp.where(mask_bsub[None, :], 0.0, bsubvmnc)

    bsubumns = jnp.asarray(bsubumns_full)[1:, :] if bool(cfg.lasym) else None
    bsubvmns = jnp.asarray(bsubvmns_full)[1:, :] if bool(cfg.lasym) else None
    bmns = jnp.asarray(bmns_full)[1:, :] if bool(cfg.lasym) else None
    rmns = rmns_half if bool(cfg.lasym) else None
    zmnc = zmnc_half if bool(cfg.lasym) else None
    lmnc = lmnc_half if bool(cfg.lasym) else None

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
        rmns=rmns,
        zmnc=zmnc,
        lmnc=lmnc,
        bmns=bmns,
        bsubumns=bsubumns,
        bsubvmns=bsubvmns,
    )

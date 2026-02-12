"""VMEC-style half-mesh metric + B-covariant ingredients.

This module ports the *core* algebra from VMEC2000's ``bcovar`` for
fixed-boundary parity work:

- Build half-mesh metric elements ``g_uu, g_uv, g_vv`` using VMEC's even/odd-m
  decomposition and half-mesh staggering.
- Build half-mesh Jacobian-related fields via :mod:`vmec_jax.vmec_jacobian`.
- Compute VMEC contravariant field components ``(B^u, B^v)`` and the covariant
  components ``(B_u, B_v)`` on the radial half mesh.
- Provide force-kernel inputs used by VMEC's ``forces`` routine.

The implementation here is intentionally limited to what's needed for validated
parity work. Use ``use_vmec_synthesis=True`` to switch the internal
R/Z/L synthesis to VMEC's symmetry-reduced theta grid (ntheta2/ntheta3) using
the ``fixaray`` trig tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._compat import jnp
from .field import TWOPI
from .field import lamscale_from_phips
from .field import chips_from_wout_chipf
from .vmec_jacobian import VmecHalfMeshJacobian, jacobian_half_mesh_from_parity
from .fourier import build_helical_basis, eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys
from .grids import AngleGrid
from .modes import ModeTable
from .vmec_parity import ParityRZL, internal_odd_from_physical_vmec_m1, split_rzl_even_odd_m
from .vmec_realspace import vmec_realspace_synthesis, vmec_realspace_synthesis_dtheta, vmec_realspace_synthesis_dzeta_phys
from .vmec_tomnsp import VmecTrigTables, vmec_trig_tables


@dataclass(frozen=True)
class VmecHalfMeshBcovar:
    """Half-mesh quantities used downstream by VMEC force/residue kernels."""

    jac: VmecHalfMeshJacobian

    # Half-mesh metric elements in cylindrical coordinates.
    guu: Any  # (ns, ntheta, nzeta)
    guv: Any  # (ns, ntheta, nzeta)
    gvv: Any  # (ns, ntheta, nzeta)

    # Half-mesh magnetic field components.
    bsupu: Any  # (ns, ntheta, nzeta)
    bsupv: Any  # (ns, ntheta, nzeta)
    bsubu: Any  # (ns, ntheta, nzeta)
    bsubv: Any  # (ns, ntheta, nzeta)

    # VMEC lambda force kernels (Fourier-space transform inputs) on full mesh.
    # These correspond to `bsubu_e/bsubv_e` in `bcovar.f` after the `-lamscale`
    # scaling, and are used as `(CLMN, BLMN)` in `tomnsps`.
    bsubu_e: Any  # (ns, ntheta, nzeta) unscaled full-mesh covariant B_u
    bsubv_e: Any  # (ns, ntheta, nzeta) unscaled full-mesh covariant B_v
    bsubu_e_scaled: Any  # (ns, ntheta, nzeta) scaled for tomnsps (VMEC bsubu_e)
    bsubv_e_scaled: Any  # (ns, ntheta, nzeta) scaled for tomnsps (VMEC bsubv_e)
    bsubu_tmp: Any  # (ns, ntheta, nzeta) pguv*bsupu term in bsubv_e
    bsubv_preblend: Any  # (ns, ntheta, nzeta) bsubv_e before blending
    bsubv_avg: Any  # (ns, ntheta, nzeta) averaged half-mesh bsubv
    clmn_even: Any  # (ns, ntheta, nzeta)
    clmn_odd: Any  # (ns, ntheta, nzeta)
    blmn_even: Any  # (ns, ntheta, nzeta)
    blmn_odd: Any  # (ns, ntheta, nzeta)

    # Lambda-force intermediate terms (full mesh), used for parity debugging.
    lu0_full: Any  # (ns, ntheta, nzeta) full-mesh LU even (scaled + wout.phipf)
    lu0_force: Any  # (ns, ntheta, nzeta) LU even used in lambda-force block
    lu1_full: Any  # (ns, ntheta, nzeta) full-mesh LU odd (scaled)
    lvv: Any  # (ns, ntheta, nzeta) phipog * gvv (full mesh)
    lvv_sh: Any  # (ns, ntheta, nzeta) lvv * pshalf
    phip_full: Any  # (ns,) full-mesh phipf used in LU_even
    phip_internal: Any  # (ns,) VMEC internal phipf = signgs*phipf/(2π)

    # bsq = |B|^2/2 + p on half mesh (VMEC convention).
    bsq: Any  # (ns, ntheta, nzeta)

    # Force-kernel inputs (VMEC `bcovar` post-processing).
    gij_b_uu: Any  # (ns, ntheta, nzeta) = (B^u B^u) * sqrt(g)
    gij_b_uv: Any  # (ns, ntheta, nzeta) = (B^u B^v) * sqrt(g)
    gij_b_vv: Any  # (ns, ntheta, nzeta) = (B^v B^v) * sqrt(g)
    lu_e: Any  # (ns, ntheta, nzeta) = R * bsq
    lv_e: Any  # (ns, ntheta, nzeta) = (sqrt(g)/R) * bsq  (tau * bsq)

    # Lambda derivatives on half mesh (scaled-lambda).
    lam_u: Any  # (ns, ntheta, nzeta)
    lam_v: Any  # (ns, ntheta, nzeta)

    # Scalar lambda scaling factor.
    lamscale: Any


def _pshalf_from_s(s: Any) -> Any:
    s = jnp.asarray(s)
    if s.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s, 0.0))
    sh = 0.5 * (s[1:] + s[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, 0.0))


def _half_mesh_from_even_odd(even, odd_int, *, s):
    """VMEC half-mesh staggering for fields of form X = X_even + sqrt(s) X_odd."""
    even = jnp.asarray(even)
    odd_int = jnp.asarray(odd_int)
    s = jnp.asarray(s)

    ns = int(s.shape[0])
    if ns < 2:
        return even

    pshalf = _pshalf_from_s(s)[:, None, None]
    out = jnp.zeros_like(even)
    out = out.at[1:].set(0.5 * (even[1:] + even[:-1] + pshalf[1:] * (odd_int[1:] + odd_int[:-1])))
    out = out.at[0].set(out[1])
    return out


def _metric_even_odd(*, a0, a1, b0, b1, s):
    """Even/odd decomposition of (a0 + sqrt(s)a1)^2 + (b0 + sqrt(s)b1)^2."""
    s = jnp.asarray(s)
    ss = s[:, None, None]
    even = a0 * a0 + b0 * b0 + ss * (a1 * a1 + b1 * b1)
    odd = 2.0 * (a0 * a1 + b0 * b1)
    return even, odd


def _metric_cross_even_odd(*, a0, a1, b0, b1, s):
    """Even/odd decomposition of (a0 + sqrt(s)a1)(b0 + sqrt(s)b1)."""
    s = jnp.asarray(s)
    ss = s[:, None, None]
    even = a0 * b0 + ss * (a1 * b1)
    odd = a0 * b1 + a1 * b0
    return even, odd


def _apply_vmec_lambda_axis_closure(
    *,
    Lsin: Any,
    m_modes: Any,
    n_modes: Any,
    lthreed: bool,
    ntor: int,
) -> Any:
    """Mirror VMEC ``totzsp_mod`` axis closure for symmetric 3D lambda modes.

    VMEC sets ``lmncs(1,n,m=0) = lmncs(2,n,m=0)`` for ``n>0`` in symmetric 3D
    runs. In vmec_jax signed storage this corresponds to ``Lsin[js=0,m=0,n>0]``
    being copied from ``js=1`` for force evaluation.
    """
    Lsin = jnp.asarray(Lsin)
    ns = int(Lsin.shape[0])
    if (not bool(lthreed)) or int(ntor) <= 0 or ns < 2:
        return Lsin

    m_modes_np = np.asarray(m_modes, dtype=int)
    n_modes_np = np.asarray(n_modes, dtype=int)
    axis_copy_mask_np = (m_modes_np == 0) & (n_modes_np > 0)
    if not np.any(axis_copy_mask_np):
        return Lsin

    axis_copy_mask = jnp.asarray(axis_copy_mask_np, dtype=Lsin.dtype)
    axis_row = jnp.where(axis_copy_mask != 0, Lsin[1, :], Lsin[0, :])
    return Lsin.at[0, :].set(axis_row)


def vmec_bcovar_half_mesh_from_wout(
    *,
    state,
    static,
    wout,
    pres: Any | None = None,
    use_wout_bsup: bool = False,
    use_wout_bsub_for_lambda: bool = False,
    use_wout_bmag_for_bsq: bool = False,
    use_vmec_synthesis: bool = False,
    trig: VmecTrigTables | None = None,
) -> VmecHalfMeshBcovar:
    """Compute VMEC-style half-mesh metric and B components for parity tests.

    Parameters
    ----------
    state:
        :class:`~vmec_jax.state.VMECState` (typically from :func:`~vmec_jax.wout.state_from_wout`).
    static:
        Static precomputations from :func:`~vmec_jax.static.build_static`.
    wout:
        :class:`~vmec_jax.wout.WoutData` providing ``phipf``, ``chipf``, ``phips``, ``signgs``.
    pres:
        Optional pressure profile on the *half mesh* in VMEC internal units (mu0*Pa).
        If omitted, uses ``wout.pres``.
    use_vmec_synthesis:
        If True, evaluate the R/Z/L real-space parity pieces and their derivatives
        using VMEC's ``fixaray`` trig tables (ntheta2/ntheta3 grids).
    use_wout_bsub_for_lambda:
        If True, build the lambda-force full-mesh kernels (``blmn/clmn``) from
        ``wout``-stored ``bsub*`` fields averaged to the full radial mesh. This
        is a reference-parity mode that avoids re-deriving ``bsubv_e`` from
        lambda derivatives.
    use_wout_bmag_for_bsq:
        If True, use ``wout`` Nyquist ``|B|`` to form ``bsq = |B|^2/2 + p`` in
        this parity path, instead of deriving ``|B|^2`` from
        ``bsup/bsub`` products.
    trig:
        Optional precomputed VMEC trig tables. If omitted and
        ``use_vmec_synthesis=True``, they are built internally.
    """
    s = jnp.asarray(static.s)
    ns = int(s.shape[0])
    # VMEC `totzsps` converts the m=1 (rss,zcs) internal pair to physical
    # coefficients before real-space synthesis when `lconm1` is enabled.
    # Apply the same conversion in signed storage for geometry parity.
    rsin_geom = jnp.asarray(state.Rsin)
    zcos_geom = jnp.asarray(state.Zcos)
    if bool(getattr(static.cfg, "lthreed", True)) and bool(getattr(static.cfg, "lconm1", True)) and int(getattr(static.cfg, "mpol", 0)) > 1:
        m_modes = np.asarray(static.modes.m, dtype=int)
        mask_m1 = jnp.asarray(m_modes == 1, dtype=rsin_geom.dtype)[None, :]
        rss_old = rsin_geom
        rsin_geom = jnp.where(mask_m1 != 0, rss_old + zcos_geom, rss_old)
        zcos_geom = jnp.where(mask_m1 != 0, rss_old - zcos_geom, zcos_geom)
    Lcos_force = jnp.asarray(state.Lcos)
    Lsin_force = _apply_vmec_lambda_axis_closure(
        Lsin=state.Lsin,
        m_modes=static.modes.m,
        n_modes=static.modes.n,
        lthreed=bool(getattr(static.cfg, "lthreed", False)),
        ntor=int(getattr(static.cfg, "ntor", 0)),
    )

    state_parity = SimpleNamespace(
        Rcos=state.Rcos,
        Rsin=rsin_geom,
        Zcos=zcos_geom,
        Zsin=state.Zsin,
        Lcos=Lcos_force,
        Lsin=Lsin_force,
    )

    if use_vmec_synthesis:
        if trig is None:
            mmax = int(np.max(static.modes.m))
            nmax = int(np.max(np.abs(static.modes.n)))
            trig = vmec_trig_tables(
                ntheta=int(static.cfg.ntheta),
                nzeta=int(static.cfg.nzeta),
                nfp=int(wout.nfp),
                mmax=mmax,
                nmax=nmax,
                lasym=bool(wout.lasym),
                dtype=jnp.asarray(state.Rcos).dtype,
            )

        def _eval_pair(cos, sin, mask):
            return vmec_realspace_synthesis(coeff_cos=cos * mask, coeff_sin=sin * mask, modes=static.modes, trig=trig)

        def _eval_pair_dtheta(cos, sin, mask):
            return vmec_realspace_synthesis_dtheta(coeff_cos=cos * mask, coeff_sin=sin * mask, modes=static.modes, trig=trig)

        def _eval_pair_dzeta(cos, sin, mask):
            return vmec_realspace_synthesis_dzeta_phys(coeff_cos=cos * mask, coeff_sin=sin * mask, modes=static.modes, trig=trig)

        m = np.asarray(static.modes.m, dtype=int)
        dtype = jnp.asarray(state.Rcos).dtype
        mask_even = jnp.asarray((m % 2) == 0).astype(dtype)
        mask_odd = (1.0 - mask_even).astype(dtype)

        parity = ParityRZL(
            R_even=_eval_pair(state_parity.Rcos, state_parity.Rsin, mask_even),
            R_odd=_eval_pair(state_parity.Rcos, state_parity.Rsin, mask_odd),
            Z_even=_eval_pair(state_parity.Zcos, state_parity.Zsin, mask_even),
            Z_odd=_eval_pair(state_parity.Zcos, state_parity.Zsin, mask_odd),
            L_even=_eval_pair(state_parity.Lcos, state_parity.Lsin, mask_even),
            L_odd=_eval_pair(state_parity.Lcos, state_parity.Lsin, mask_odd),
            Rt_even=_eval_pair_dtheta(state_parity.Rcos, state_parity.Rsin, mask_even),
            Rt_odd=_eval_pair_dtheta(state_parity.Rcos, state_parity.Rsin, mask_odd),
            Zt_even=_eval_pair_dtheta(state_parity.Zcos, state_parity.Zsin, mask_even),
            Zt_odd=_eval_pair_dtheta(state_parity.Zcos, state_parity.Zsin, mask_odd),
            Lt_even=_eval_pair_dtheta(state_parity.Lcos, state_parity.Lsin, mask_even),
            Lt_odd=_eval_pair_dtheta(state_parity.Lcos, state_parity.Lsin, mask_odd),
            Rp_even=_eval_pair_dzeta(state_parity.Rcos, state_parity.Rsin, mask_even),
            Rp_odd=_eval_pair_dzeta(state_parity.Rcos, state_parity.Rsin, mask_odd),
            Zp_even=_eval_pair_dzeta(state_parity.Zcos, state_parity.Zsin, mask_even),
            Zp_odd=_eval_pair_dzeta(state_parity.Zcos, state_parity.Zsin, mask_odd),
            Lp_even=_eval_pair_dzeta(state_parity.Lcos, state_parity.Lsin, mask_even),
            Lp_odd=_eval_pair_dzeta(state_parity.Lcos, state_parity.Lsin, mask_odd),
        )
    else:
        # Split real-space fields into even/odd-m subsets, then convert odd physical
        # contribution to VMEC's internal odd field by dividing by sqrt(s).
        parity = split_rzl_even_odd_m(state_parity, static.basis, static.modes.m)

    # VMEC axis convention (vmec_params.f: jmin1):
    # - m=1 odd-m internal fields are extrapolated to the axis (copy js=2),
    # - odd-m with m>=3 are zero on the axis.
    m_modes = np.asarray(static.modes.m, dtype=int)
    dtype = jnp.asarray(state.Rcos).dtype
    mask_m1 = jnp.asarray(m_modes == 1, dtype=dtype)
    mask_odd_rest = jnp.asarray((m_modes % 2 == 1) & (m_modes != 1), dtype=dtype)

    def _odd_internal_vmec(*, coeff_cos, coeff_sin, eval_fn):
        if use_vmec_synthesis:
            phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1)
            phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest)
        else:
            phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1, static.basis)
            phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest, static.basis)
        return internal_odd_from_physical_vmec_m1(odd_m1_phys=phys_m1, odd_mge2_phys=phys_rest, s=s)

    if use_vmec_synthesis:
        R1 = _odd_internal_vmec(coeff_cos=state_parity.Rcos, coeff_sin=state_parity.Rsin, eval_fn=lambda c, s: vmec_realspace_synthesis(coeff_cos=c, coeff_sin=s, modes=static.modes, trig=trig))
        Z1 = _odd_internal_vmec(coeff_cos=state_parity.Zcos, coeff_sin=state_parity.Zsin, eval_fn=lambda c, s: vmec_realspace_synthesis(coeff_cos=c, coeff_sin=s, modes=static.modes, trig=trig))
        Ru1 = _odd_internal_vmec(
            coeff_cos=state_parity.Rcos,
            coeff_sin=state_parity.Rsin,
            eval_fn=lambda c, s: vmec_realspace_synthesis_dtheta(coeff_cos=c, coeff_sin=s, modes=static.modes, trig=trig),
        )
        Zu1 = _odd_internal_vmec(
            coeff_cos=state_parity.Zcos,
            coeff_sin=state_parity.Zsin,
            eval_fn=lambda c, s: vmec_realspace_synthesis_dtheta(coeff_cos=c, coeff_sin=s, modes=static.modes, trig=trig),
        )
        Rv1 = _odd_internal_vmec(
            coeff_cos=state_parity.Rcos,
            coeff_sin=state_parity.Rsin,
            eval_fn=lambda c, s: vmec_realspace_synthesis_dzeta_phys(coeff_cos=c, coeff_sin=s, modes=static.modes, trig=trig),
        )
        Zv1 = _odd_internal_vmec(
            coeff_cos=state_parity.Zcos,
            coeff_sin=state_parity.Zsin,
            eval_fn=lambda c, s: vmec_realspace_synthesis_dzeta_phys(coeff_cos=c, coeff_sin=s, modes=static.modes, trig=trig),
        )

        Lu1 = _odd_internal_vmec(
            coeff_cos=state_parity.Lcos,
            coeff_sin=state_parity.Lsin,
            eval_fn=lambda c, s: vmec_realspace_synthesis_dtheta(coeff_cos=c, coeff_sin=s, modes=static.modes, trig=trig),
        )
        Lv1 = _odd_internal_vmec(
            coeff_cos=state_parity.Lcos,
            coeff_sin=state_parity.Lsin,
            eval_fn=lambda c, s: vmec_realspace_synthesis_dzeta_phys(coeff_cos=c, coeff_sin=s, modes=static.modes, trig=trig),
        )
    else:
        R1 = _odd_internal_vmec(coeff_cos=state_parity.Rcos, coeff_sin=state_parity.Rsin, eval_fn=eval_fourier)
        Z1 = _odd_internal_vmec(coeff_cos=state_parity.Zcos, coeff_sin=state_parity.Zsin, eval_fn=eval_fourier)
        Ru1 = _odd_internal_vmec(coeff_cos=state_parity.Rcos, coeff_sin=state_parity.Rsin, eval_fn=eval_fourier_dtheta)
        Zu1 = _odd_internal_vmec(coeff_cos=state_parity.Zcos, coeff_sin=state_parity.Zsin, eval_fn=eval_fourier_dtheta)
        Rv1 = _odd_internal_vmec(coeff_cos=state_parity.Rcos, coeff_sin=state_parity.Rsin, eval_fn=eval_fourier_dzeta_phys)
        Zv1 = _odd_internal_vmec(coeff_cos=state_parity.Zcos, coeff_sin=state_parity.Zsin, eval_fn=eval_fourier_dzeta_phys)

        Lu1 = _odd_internal_vmec(coeff_cos=state_parity.Lcos, coeff_sin=state_parity.Lsin, eval_fn=eval_fourier_dtheta)
        Lv1 = _odd_internal_vmec(coeff_cos=state_parity.Lcos, coeff_sin=state_parity.Lsin, eval_fn=eval_fourier_dzeta_phys)

    # Half-mesh Jacobian quantities from VMEC's discrete formula.
    jac = jacobian_half_mesh_from_parity(
        pr1_even=parity.R_even,
        pr1_odd=R1,
        pz1_even=parity.Z_even,
        pz1_odd=Z1,
        pru_even=parity.Rt_even,
        pru_odd=Ru1,
        pzu_even=parity.Zt_even,
        pzu_odd=Zu1,
        s=s,
    )

    # Metric elements on full mesh split into even/odd (internal) pieces, then
    # staggered to the half mesh using VMEC's pshalf convention.
    guu_e, guu_o = _metric_even_odd(a0=parity.Rt_even, a1=Ru1, b0=parity.Zt_even, b1=Zu1, s=s)
    guv_e, guv_o = _metric_cross_even_odd(a0=parity.Rt_even, a1=Ru1, b0=parity.Rp_even, b1=Rv1, s=s)
    guv_e2, guv_o2 = _metric_cross_even_odd(a0=parity.Zt_even, a1=Zu1, b0=parity.Zp_even, b1=Zv1, s=s)
    guv_e = guv_e + guv_e2
    guv_o = guv_o + guv_o2
    gvv_e, gvv_o = _metric_even_odd(a0=parity.Rp_even, a1=Rv1, b0=parity.Zp_even, b1=Zv1, s=s)

    # R^2 term in cylindrical metric: gvv <- gvv + R^2
    ss = s[:, None, None]
    R2_e = parity.R_even * parity.R_even + ss * (R1 * R1)
    R2_o = 2.0 * parity.R_even * R1

    guu = _half_mesh_from_even_odd(guu_e, guu_o, s=s)
    guv = _half_mesh_from_even_odd(guv_e, guv_o, s=s)
    gvv = _half_mesh_from_even_odd(gvv_e, gvv_o, s=s) + _half_mesh_from_even_odd(R2_e, R2_o, s=s)

    # Lambda derivatives on half mesh (scaled lambda).
    lam_u = _half_mesh_from_even_odd(parity.Lt_even, Lu1, s=s)
    lam_v = _half_mesh_from_even_odd(parity.Lp_even, Lv1, s=s)

    # NOTE: `lam_u/lam_v` are used downstream in VMEC's magnetic-field and
    # force pipeline, but the (guu,guv,gvv) metric elements used for bsub*
    # parity are those constructed above from (Ru,Zu,Rv,Zv) and the cylindrical
    # +R^2 term.

    # ---------------------------------------------------------------------
    # Contravariant B components (bsupu, bsupv) on the half mesh.
    # ---------------------------------------------------------------------
    # VMEC does not form bsupu/bsupv from pointwise (sqrtg, lam_u, lam_v) alone.
    # Instead, it:
    #   1) builds full-mesh LU = d(lambda)/du and LV = -d(lambda)/dv (even/odd-m),
    #   2) scales LU/LV by lamscale and adds phipf to LU_even,
    #   3) averages (LU,LV) from full -> half radial mesh using pshalf,
    #   4) adds the full-mesh flux function chips(js) via `add_fluxes`.
    #
    # See `VMEC2000/Sources/General/bcovar.f` and `add_fluxes.f90`.
    lamscale = lamscale_from_phips(wout.phips, s)
    signgs = int(getattr(wout, "signgs", 1))

    # VMEC adds the **full-mesh** flux function `chips(js)` to bsupu in
    # `add_fluxes`, while `wout` commonly stores the half-mesh array `chipf`.
    chipf_out = getattr(wout, "chipf", None)
    phipf_out = jnp.asarray(getattr(wout, "phipf"))
    signgs = int(getattr(wout, "signgs", 1))
    flux_is_internal = bool(getattr(wout, "flux_is_internal", False))
    if not flux_is_internal:
        scale = jnp.asarray(TWOPI, dtype=phipf_out.dtype) * jnp.asarray(signgs, dtype=phipf_out.dtype)
        phipf_internal = phipf_out / scale
        chipf_internal = None if chipf_out is None else (jnp.asarray(chipf_out) / scale)
    else:
        phipf_internal = phipf_out
        chipf_internal = None if chipf_out is None else jnp.asarray(chipf_out)
    if chipf_out is not None:
        chips_eff = chips_from_wout_chipf(
            chipf=chipf_internal,
            phipf=phipf_internal,
            iotaf=getattr(wout, "iotaf", None),
            iotas=getattr(wout, "iotas", None),
            # Solver-internal wout-like objects may omit iotaf/iotas and provide
            # half-mesh chipf; keep VMEC2000-compatible behavior in that case.
            assume_half_if_unknown=True,
        )
    else:
        chips_eff = jnp.asarray(getattr(wout, "iotaf", getattr(wout, "iotas", 0.0))) * jnp.asarray(phipf_internal)
    ncurr = int(getattr(wout, "ncurr", 0))
    lcurrent = bool(getattr(wout, "lcurrent", True))
    icurv = jnp.asarray(getattr(wout, "icurv", jnp.zeros((ns,), dtype=phipf_internal.dtype)))
    if int(icurv.shape[0]) != ns:
        icurv = jnp.zeros((ns,), dtype=phipf_internal.dtype)

    # VMEC bcovar: overg = 1 / (signgs * sqrtg).
    denom = int(signgs) * jac.sqrtg
    overg = jnp.where(denom != 0, 1.0 / denom, 0.0)

    # Full-mesh LU = d(lambda)/du and LV = -d(lambda)/dv in VMEC conventions.
    lu0_full = jnp.asarray(parity.Lt_even)
    lu1_full = jnp.asarray(Lu1)
    lv0_full = -jnp.asarray(parity.Lp_even)
    lv1_full = -jnp.asarray(Lv1)

    # Scale by lamscale and add wout phipf to LU_even (bsupv path).
    lu0_full = (lamscale * lu0_full) + jnp.asarray(phipf_internal)[:, None, None]
    lu1_full = lamscale * lu1_full
    lv0_full = lamscale * lv0_full
    lv1_full = lamscale * lv1_full

    pshalf = _pshalf_from_s(s)[:, None, None]

    # Radial full->half average (Fortran: for l=2..ns).
    bsupu = jnp.zeros_like(jac.sqrtg)
    bsupv = jnp.zeros_like(jac.sqrtg)
    if ns >= 2:
        avg_lu0 = lu0_full[1:] + lu0_full[:-1]
        avg_lu1 = lu1_full[1:] + lu1_full[:-1]
        avg_lv0 = lv0_full[1:] + lv0_full[:-1]
        avg_lv1 = lv1_full[1:] + lv1_full[:-1]

        bsupv = bsupv.at[1:].set(0.5 * overg[1:] * (avg_lu0 + pshalf[1:] * avg_lu1))
        bsupu = bsupu.at[1:].set(0.5 * overg[1:] * (avg_lv0 + pshalf[1:] * avg_lv1))

    if (ncurr == 1) and lcurrent and (ns >= 2):
        w_theta = jnp.asarray(trig.cosmui3[:, 0], dtype=bsupu.dtype) / jnp.asarray(trig.mscale[0], dtype=bsupu.dtype)
        w_ang = w_theta[:, None] * jnp.ones((int(overg.shape[2]),), dtype=bsupu.dtype)[None, :]
        pwint = jnp.broadcast_to(w_ang[None, :, :], overg.shape)
        pwint = pwint.at[0].set(jnp.zeros_like(pwint[0]))

        top = jnp.asarray(icurv, dtype=bsupu.dtype) - jnp.sum(
            pwint * ((guu * bsupu) + (guv * bsupv)),
            axis=(1, 2),
        )
        bot = jnp.sum(pwint * (overg * guu), axis=(1, 2))

        chips_dyn = jnp.asarray(chips_eff, dtype=bsupu.dtype)
        chips_new = jnp.where(bot != 0.0, top / bot, chips_dyn)
        chips_dyn = chips_dyn.at[0].set(jnp.asarray(0.0, dtype=chips_dyn.dtype))
        chips_dyn = chips_dyn.at[1:].set(chips_new[1:])
        chips_eff = chips_dyn

    # `add_fluxes`: bsupu += chips*overg (chips is a 1D full-mesh flux function).
    bsupu = bsupu + jnp.asarray(chips_eff)[:, None, None] * overg

    basis_nyq = None
    if bool(use_wout_bsup):
        # Replace with wout-stored Nyquist bsup (reference parity path).
        modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
        # Nyquist `wout` field coefficients (`bsup*`, `bsub*`, `bm*`) follow the
        # output transform conventions from `wrout` and are most consistent with
        # direct Fourier evaluation on the active angular grid.
        #
        # Using VMEC synthesis tables for these reference fields introduces a
        # small but systematic mismatch in the parity path for nfp>1 cases.
        grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
        basis_nyq = build_helical_basis(modes_nyq, grid)
        bsupu = jnp.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
        bsupv = jnp.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))

    # VMEC enforces axis bsup*=0 explicitly.
    if ns >= 1:
        bsupu = bsupu.at[0].set(jnp.zeros_like(bsupu[0]))
        bsupv = bsupv.at[0].set(jnp.zeros_like(bsupv[0]))

    # Align bsup sign convention with VMEC2000 (parity dumps).
    # VMEC uses signgs=-1 with a fixed orientation; our geometry currently
    # yields bsupu/bsupv with the opposite sign for computed fields.
    if not bool(use_wout_bsup):
        bsupu = -bsupu
        bsupv = -bsupv

    bsubu = guu * bsupu + guv * bsupv
    bsubv = guv * bsupu + gvv * bsupv

    # Optional reference parity path for lambda-force kernels.
    bsubu_lambda = bsubu
    bsubv_lambda = bsubv
    # VMEC internal phipf corresponds to dPhi/(2π) and includes signgs.
    # `flux_profiles_from_indata` already constructs phipf in that convention,
    # so use wout.phipf directly for the lambda-force block.
    phip_internal = jnp.asarray(phipf_internal)
    lu0_force = (lamscale * jnp.asarray(parity.Lt_even)) + phip_internal[:, None, None]

    if bool(use_wout_bsub_for_lambda):
        modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
        grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
        basis_nyq = build_helical_basis(modes_nyq, grid)
        bsubu_lambda = jnp.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
        bsubv_lambda = jnp.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))

    b2 = bsupu * bsubu + bsupv * bsubv
    if bool(use_wout_bmag_for_bsq):
        if basis_nyq is None:
            modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
            grid = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
            basis_nyq = build_helical_basis(modes_nyq, grid)
        bmag_ref = jnp.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))
        b2 = bmag_ref * bmag_ref
    pres_h = jnp.asarray(wout.pres if pres is None else pres)[:, None, None]
    bsq = 0.5 * b2 + pres_h

    # Force-kernel inputs matching what `forces.f` expects after `bcovar`.
    gij_b_uu = (bsupu * bsupu) * jac.sqrtg
    gij_b_uv = (bsupu * bsupv) * jac.sqrtg
    gij_b_vv = (bsupv * bsupv) * jac.sqrtg
    lu_e = bsq * jac.r12
    lv_e = bsq * jac.tau

    # ---------------------------------------------------------------------
    # Lambda force kernels (bcovar.f "lambda full mesh forces" block)
    # ---------------------------------------------------------------------
    # This reproduces the structure in `bcovar.f`:
    #   - compute an intermediate bsubv_e on the full radial mesh from LU and metrics
    #   - average (bsubuh,bsubvh) from the half mesh onto the full mesh
    #   - blend bsubv_e with averaged bsubvh using bdamp(s) for near-axis stability
    #
    # Inputs:
    # - LU (full mesh, parity-split): LU = phipf + lamscale*dλ/du
    # - lvv (half mesh): lvv = (g_vv / (signgs*sqrtg*2π))
    # - bsubu/bsubv (half mesh): covariant B components

    # Full-mesh LU parity pieces (odd is VMEC-internal 1/sqrt(s) representation).
    lu0 = (lamscale * parity.Lt_even) + jnp.asarray(wout.phipf)[:, None, None]
    lu1 = lamscale * Lu1

    # lvv on half mesh: phipog * gvv (bcovar.f uses phipog == 1/sqrtg).
    # NOTE: phipog does **not** include the 2π scaling used in `overg`.
    phipog = jnp.where(jac.sqrtg != 0, 1.0 / jac.sqrtg, 0.0)
    lvv = phipog * gvv

    if bool(use_wout_bsub_for_lambda):
        # Reference parity mode: use averaged wout bsub* directly.
        bsubu_e = jnp.zeros_like(bsubu_lambda)
        bsubv_e = jnp.zeros_like(bsubv_lambda)
        bsubu_tmp = jnp.zeros_like(bsubu_lambda)
        bsubv_preblend = jnp.zeros_like(bsubv_lambda)
        bsubv_avg = jnp.zeros_like(bsubv_lambda)
        if ns >= 2:
            bsubu_e = bsubu_e.at[:-1].set(0.5 * (bsubu_lambda[:-1] + bsubu_lambda[1:]))
            bsubu_e = bsubu_e.at[-1].set(0.5 * bsubu_lambda[-1])
            bsubv_e = bsubv_e.at[:-1].set(0.5 * (bsubv_lambda[:-1] + bsubv_lambda[1:]))
            bsubv_e = bsubv_e.at[-1].set(0.5 * bsubv_lambda[-1])
            bsubv_avg = bsubv_e
    else:
        # Intermediate full-mesh bsubv_e (before blending), following bcovar.f.
        bsubv_e = jnp.zeros_like(bsubv)
        if ns >= 2:
            bsubv_e = bsubv_e.at[:-1].set(0.5 * (lvv[:-1] + lvv[1:]) * lu0_force[:-1])
            bsubv_e = bsubv_e.at[-1].set(0.5 * lvv[-1] * lu0_force[-1])

        lvv_sh = lvv * pshalf
        bsubu_tmp = guv * bsupu  # bcovar: pguv*bsupu (sigma_an=1 isotropic)
        if ns >= 2:
            bsubv_e = bsubv_e.at[:-1].add(
                0.5 * ((lvv_sh[:-1] + lvv_sh[1:]) * lu1[:-1] + bsubu_tmp[:-1] + bsubu_tmp[1:])
            )
            bsubv_e = bsubv_e.at[-1].add(0.5 * (lvv_sh[-1] * lu1[-1] + bsubu_tmp[-1]))
        bsubv_preblend = bsubv_e

        # Average lambda forces onto full radial mesh (bsubu_e from bsubu half mesh).
        bsubu_e = jnp.zeros_like(bsubu)
        if ns >= 2:
            bsubu_e = bsubu_e.at[:-1].set(0.5 * (bsubu[:-1] + bsubu[1:]))
            bsubu_e = bsubu_e.at[-1].set(0.5 * bsubu[-1])

        # Blend bsubv_e with half-mesh bsubv average using bdamp(s) (VMEC: bdamp=2*pdamp*(1-s)).
        pdamp = 0.05
        bdamp = (2.0 * pdamp * (1.0 - s)).astype(jnp.asarray(bsubv_e).dtype)[:, None, None]
        if ns >= 2:
            bsubv_avg = jnp.zeros_like(bsubv_e)
            bsubv_avg = bsubv_avg.at[:-1].set(0.5 * (bsubv[:-1] + bsubv[1:]))
            bsubv_avg = bsubv_avg.at[-1].set(0.5 * bsubv[-1])
            bsubv_e = bdamp * bsubv_e + (1.0 - bdamp) * bsubv_avg
        else:
            bsubv_e = bdamp * bsubv_e + (1.0 - bdamp) * bsubv_e
            bsubv_avg = bsubv_e

    # Final scaling for tomnsps:
    # VMEC applies the "-lamscale" factor only for js>=2 (1-based). The axis (js=1)
    # is excluded so the lambda-force kernels do not introduce spurious constant
    # contributions from the copied/extrapolated half-mesh axis values.
    #
    # VMEC also exposes odd-m pieces as sqrt(s)*bsub*_e.
    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    clmn_even = jnp.zeros_like(bsubu_e)
    blmn_even = jnp.zeros_like(bsubv_e)
    if ns >= 1:
        # VMEC leaves the axis entries unscaled (the -lamscale factor is applied
        # only for js>=2). Preserve the raw axis values to match tomnsps dumps.
        clmn_even = clmn_even.at[0].set(bsubu_e[0])
        blmn_even = blmn_even.at[0].set(bsubv_e[0])
    if ns >= 2:
        clmn_even = clmn_even.at[1:].set(-lamscale * bsubu_e[1:])
        blmn_even = blmn_even.at[1:].set(-lamscale * bsubv_e[1:])
    clmn_odd = psqrts * clmn_even
    blmn_odd = psqrts * blmn_even

    bsubu_e_scaled = clmn_even
    bsubv_e_scaled = blmn_even

    return VmecHalfMeshBcovar(
        jac=jac,
        guu=guu,
        guv=guv,
        gvv=gvv,
        bsupu=bsupu,
        bsupv=bsupv,
        bsubu=bsubu,
        bsubv=bsubv,
        bsubu_e=bsubu_e,
        bsubv_e=bsubv_e,
        bsubu_e_scaled=bsubu_e_scaled,
        bsubv_e_scaled=bsubv_e_scaled,
        bsubu_tmp=bsubu_tmp,
        bsubv_preblend=bsubv_preblend,
        bsubv_avg=bsubv_avg,
        lu0_full=lu0_full,
        lu0_force=lu0_force,
        lu1_full=lu1_full,
        lvv=lvv,
        lvv_sh=lvv * pshalf,
        phip_full=jnp.asarray(wout.phipf),
        phip_internal=phip_internal,
        bsq=bsq,
        gij_b_uu=gij_b_uu,
        gij_b_uv=gij_b_uv,
        gij_b_vv=gij_b_vv,
        lu_e=lu_e,
        lv_e=lv_e,
        lam_u=lam_u,
        lam_v=lam_v,
        lamscale=lamscale,
        clmn_even=clmn_even,
        clmn_odd=clmn_odd,
        blmn_even=blmn_even,
        blmn_odd=blmn_odd,
    )

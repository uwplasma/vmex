"""A/B equivalence tests: ``vmec_jax.core.{forces,residuals}`` vs the legacy kernels.

Old implementations under test (left untouched; parity-proven vs VMEC2000):

- ``vmec_jax.kernels.forces.vmec_forces_rz_from_wout`` (forces.f + alias.f
  constraint pipeline, bcovar.f lambda kernels via ``k.bc``).
- ``vmec_jax.kernels.forces.vmec_residual_internal_from_kernels``
  (tomnsps/tomnspa + symforce, as called by the fixed-boundary solver in
  ``solvers/fixed_boundary/residual/iteration.py``).
- ``vmec_jax.kernels.residue`` (m=1 constraints, scalxc, gcx2 sums, norms).
- ``vmec_jax.preconditioner_1d_jax`` R/Z + lambda preconditioners and
  ``solvers/fixed_boundary/preconditioning/operators.scale_m1_precond_rhs_from_mats``
  (residue.f90 fsqr1/fsqz1/fsql1 lane).

New implementations:

- ``vmec_jax.core.forces``    (mhd_forces / lambda_force_kernels /
  constraint_force / alias_constraint_force / spectral_mhd_forces)
- ``vmec_jax.core.residuals`` (m1 mappings, m1_residue_rotation,
  zero_m1_z_force, scalxc_scale_force, force_residuals,
  scale_m1_preconditioner_rhs, apply_radial_preconditioner,
  apply_lambda_preconditioner, preconditioned_residuals)

Realistic spectral states come from short (unconverged) legacy driver runs on
sym 2D, sym 2D ncurr=1, sym 3D and lasym decks — same recipe as
``test_geometry_fields_ab.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

# Pin the legacy kernels to their CPU DFT lane (deterministic A/B reference).
os.environ.setdefault("VMEC_JAX_TOMNSPS_FFT", "0")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import vmec_jax as vj
from vmec_jax.field import chips_from_wout_chipf
from vmec_jax.kernels.forces import (
    vmec_forces_rz_from_wout,
    vmec_residual_internal_from_kernels,
)
from vmec_jax.kernels.parity import (
    _signed_to_mn_cos_cached,
    _signed_to_mn_sin_cached,
    signed_maps_from_modes,
    vmec_m1_internal_to_physical_signed,
    vmec_m1_physical_to_internal_signed,
)
from vmec_jax.kernels.residue import (
    vmec_apply_m1_constraints,
    vmec_apply_scalxc_to_tomnsps,
    vmec_force_norms_from_bcovar_dynamic,
    vmec_gcx2_from_tomnsps,
    vmec_zero_m1_zforce,
)
from vmec_jax.kernels.tomnsp import TomnspsRZL, vmec_trig_tables
from vmec_jax.preconditioner_1d_jax import (
    lambda_preconditioner as old_lambda_preconditioner,
    rz_preconditioner_apply as old_rz_preconditioner_apply,
    rz_preconditioner_matrices as old_rz_preconditioner_matrices,
)
from vmec_jax.solvers.fixed_boundary.preconditioning.operators import (
    scale_m1_precond_rhs_from_mats,
)
from vmec_jax.solvers.fixed_boundary.residual.force_norms import (
    lambda_preconditioned_full_norm,
)

from vmec_jax.core import forces as newf
from vmec_jax.core import preconditioner as newp
from vmec_jax.core import residuals as newr
from vmec_jax.core.fields import (
    constraint_scaling,
    energies_and_force_norms,
    magnetic_fields,
    metric_elements,
    preconditioned_force_norm,
)
from vmec_jax.core.fourier import Resolution, mode_table, trig_tables
from vmec_jax.core.geometry import (
    apply_lambda_axis_closure,
    half_mesh_jacobian,
    real_space_geometry,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "examples" / "data"

RTOL = 1e-12
ATOL = 1e-13

# The real-space kernels are compared with a small absolute floor: the
# (independently A/B-proven) geometry/field inputs agree to ~1e-12 *relative*,
# and the forward radial differences of forces.f cancel leading digits, so
# O(1) kernels can pick up a few-1e-12 absolute scatter without any
# order-of-operations difference in the port itself.  All spectral and scalar
# comparisons stay at the strict (RTOL, ATOL).
KERNEL_ATOL = 5e-12

TCON0 = 0.9

# name -> max_iter
CASES = {
    "solovev": 30,  # 2D sym, ncurr=0
    "cth_like_fixed_bdy": 25,  # 2D sym, nfp=5, ncurr=1
    "li383_low_res": 25,  # 3D sym (lthreed: crmn/czmn, m=1 constraint)
    "up_down_asymmetric_tokamak": 20,  # lasym (symforce + tomnspa)
}


def _allclose(new, old, name, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(
        np.asarray(new), np.asarray(old), rtol=rtol, atol=atol, err_msg=f"{name} mismatch"
    )


def _legacy_fnorm1(state, static, cfg):
    """Legacy fnorm1 reference (bcovar.f) via the old parity block conversions."""
    maps = signed_maps_from_modes(static.modes)
    rcc, rss = _signed_to_mn_cos_cached(jnp.asarray(state.Rcos), maps=maps)
    zsc, zcs = _signed_to_mn_sin_cached(jnp.asarray(state.Zsin), maps=maps)
    m_grid = np.arange(maps.mpol)[:, None]
    n_grid = np.arange(maps.nrange)[None, :]
    include_rcc = jnp.asarray(((m_grid > 0) | (n_grid > 0)).astype(float))
    sl = slice(1, None)
    rz_norm = jnp.sum(zsc[sl] * zsc[sl]) + jnp.sum(include_rcc * rcc[sl] * rcc[sl])
    if bool(cfg.lthreed):
        rz_norm = rz_norm + jnp.sum(rss[sl] * rss[sl]) + jnp.sum(zcs[sl] * zcs[sl])
    if bool(cfg.lasym):
        rsc, rcs = _signed_to_mn_sin_cached(jnp.asarray(state.Rsin), maps=maps)
        zcc, zss = _signed_to_mn_cos_cached(jnp.asarray(state.Zcos), maps=maps)
        rz_norm = (
            rz_norm
            + jnp.sum(rsc[sl] * rsc[sl])
            + jnp.sum(rcs[sl] * rcs[sl])
            + jnp.sum(zcc[sl] * zcc[sl])
            + jnp.sum(zss[sl] * zss[sl])
        )
    return 1.0 / rz_norm


def _old_residual_lane(case, *, zero_m1: float, include_edge: bool):
    """Legacy m1 -> zero_m1 -> scalxc -> gcx2 -> fsq chain (iteration.py order)."""
    frzl = case.frzl_raw_edge_old if include_edge else case.frzl_raw_old
    frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(case.cfg.lconm1))
    frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=jnp.asarray(zero_m1))
    frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=case.s)
    gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
        frzl=frzl,
        apply_m1_constraints=False,
        include_edge=include_edge,
        apply_scalxc=False,
        s=case.s,
    )
    norms = case.norms_old
    return SimpleNamespace(
        frzl_full=frzl,
        fsqr=norms.r1 * norms.fnorm * gcr2,
        fsqz=norms.r1 * norms.fnorm * gcz2,
        fsql=norms.fnormL * gcl2,
    )


def _old_preconditioned_lane(case, *, zero_m1: float):
    """Legacy scale_m1 -> rz precond -> faclam -> fsqr1/fsqz1/fsql1 chain."""
    frzl_full = _old_residual_lane(case, zero_m1=zero_m1, include_edge=False).frzl_full
    if bool(case.cfg.lthreed) or bool(case.cfg.lasym):
        frzl_rhs = scale_m1_precond_rhs_from_mats(
            frzl_full,
            case.mats_old,
            lconm1=bool(case.cfg.lconm1),
            mpol=int(case.cfg.mpol),
            host_update_assembly=False,
        )
    else:
        frzl_rhs = frzl_full
    frzl_rz = old_rz_preconditioner_apply(
        frzl_in=frzl_rhs,
        mats=case.mats_old,
        jmax=case.jmax_old,
        cfg=case.cfg,
        use_precomputed=False,
        use_lax_tridi=False,
    )
    lam = jnp.asarray(case.faclam_old)
    maybe = lambda x: None if x is None else jnp.asarray(x) * lam  # noqa: E731
    frzl_pre = TomnspsRZL(
        frcc=frzl_rz.frcc,
        frss=frzl_rz.frss,
        fzsc=frzl_rz.fzsc,
        fzcs=frzl_rz.fzcs,
        flsc=maybe(frzl_rz.flsc),
        flcs=maybe(frzl_rz.flcs),
        frsc=getattr(frzl_rz, "frsc", None),
        frcs=getattr(frzl_rz, "frcs", None),
        fzcc=getattr(frzl_rz, "fzcc", None),
        fzss=getattr(frzl_rz, "fzss", None),
        flcc=maybe(getattr(frzl_rz, "flcc", None)),
        flss=maybe(getattr(frzl_rz, "flss", None)),
    )
    gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
        frzl=frzl_pre,
        apply_m1_constraints=False,
        include_edge=True,
        apply_scalxc=False,
        s=case.s,
    )
    hs = case.s[1] - case.s[0]
    return SimpleNamespace(
        frzl_pre=frzl_pre,
        fsqr1=gcr2_p * case.fnorm1_old,
        fsqz1=gcz2_p * case.fnorm1_old,
        fsql1=lambda_preconditioned_full_norm(frzl_pre, use_jax=True) * hs,
    )


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request):
    """Run the legacy driver briefly; package old- and new-side chains."""
    name = request.param
    run = vj.run_fixed_boundary(str(DATA_DIR / f"input.{name}"), max_iter=CASES[name], verbose=False)
    cfg, state, static, flux, prof = run.cfg, run.state, run.static, run.flux, run.profiles
    signgs = int(run.signgs)
    s = jnp.asarray(static.s)
    ns = int(s.shape[0])
    hs = s[1] - s[0]

    pres = jnp.asarray(prof["pressure"])
    if ns > 0:
        pres = pres.at[0].set(0.0)
    ncurr = int(prof["ncurr"])
    chips = chips_from_wout_chipf(
        chipf=jnp.asarray(flux.chipf),
        phipf=jnp.asarray(flux.phipf),
        iotaf=None,
        iotas=None,
        assume_half_if_unknown=True,
    )
    wout_like = SimpleNamespace(
        phipf=jnp.asarray(flux.phipf),
        chipf=jnp.asarray(flux.chipf),
        phips=jnp.asarray(flux.phips),
        chips_eff=chips,
        pres=pres,
        nfp=int(cfg.nfp),
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        lasym=bool(cfg.lasym),
        signgs=signgs,
        ncurr=ncurr,
        lcurrent=(ncurr == 1),
        gamma=0.0,
        flux_is_internal=True,
    )

    # ------------------------------------------------------------------ old
    # Legacy tables are Fortran-faithful since the fixaray.f lasym dnorm fix
    # landed in vmec_jax/kernels/tomnsp.py; no rescaling compensation needed.
    trig_old = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        mmax=int(cfg.mpol) - 1,
        nmax=int(cfg.ntor),
        lasym=bool(cfg.lasym),
    )
    k_old = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        constraint_tcon0=TCON0,
        use_vmec_synthesis=True,
        trig=trig_old,
    )
    frzl_raw_old = vmec_residual_internal_from_kernels(
        k_old,
        cfg_ntheta=int(cfg.ntheta),
        cfg_nzeta=int(cfg.nzeta),
        wout=wout_like,
        trig=trig_old,
        include_edge=False,
    )
    frzl_raw_edge_old = vmec_residual_internal_from_kernels(
        k_old,
        cfg_ntheta=int(cfg.ntheta),
        cfg_nzeta=int(cfg.nzeta),
        wout=wout_like,
        trig=trig_old,
        include_edge=True,
    )
    norms_old = vmec_force_norms_from_bcovar_dynamic(
        bc=k_old.bc, trig=trig_old, s=s, signgs=signgs
    )
    mats_old, _jmin_old, jmax_old = old_rz_preconditioner_matrices(
        bc=k_old.bc, k=k_old, trig=None, s=s, cfg=cfg,
        use_precomputed=False, use_lax_tridi=False,
    )
    faclam_old = old_lambda_preconditioner(bc=k_old.bc, trig=None, s=s, cfg=cfg)
    fnorm1_old = _legacy_fnorm1(state, static, cfg)

    # ------------------------------------------------------------------ new
    res = Resolution(
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
        ns=ns,
    )
    trig_new = trig_tables(res)
    modes_new = mode_table(int(cfg.mpol), int(cfg.ntor))
    weights = newp.angular_integration_weights(
        ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), lasym=bool(cfg.lasym)
    )

    R_cos, Z_sin, R_sin, Z_cos = newr.m1_constrained_to_physical(
        jnp.asarray(state.Rcos),
        jnp.asarray(state.Zsin),
        jnp.asarray(state.Rsin),
        jnp.asarray(state.Zcos),
        modes=modes_new,
        lthreed=bool(cfg.lthreed),
        lasym=bool(cfg.lasym),
        lconm1=bool(cfg.lconm1),
    )
    new_inputs = dict(
        R_cos=R_cos,
        R_sin=R_sin,
        Z_cos=Z_cos,
        Z_sin=Z_sin,
        lambda_cos=jnp.asarray(state.Lcos),
        lambda_sin=apply_lambda_axis_closure(
            jnp.asarray(state.Lsin), modes=modes_new, ntor=int(cfg.ntor)
        ),
    )

    def new_chain(inputs, *, zero_m1, include_edge=False):
        geom = real_space_geometry(**inputs, modes=modes_new, trig=trig_new, s=s)
        jac = half_mesh_jacobian(geom, s=s)
        mets = metric_elements(geom, s=s)
        mf = magnetic_fields(
            geometry=geom,
            jacobian=jac,
            metrics=mets,
            trig=trig_new,
            s=s,
            phips=jnp.asarray(flux.phips),
            phipf=jnp.asarray(flux.phipf),
            chips=chips,
            signgs=signgs,
            gamma=0.0,
            pressure=pres,
            ncurr=ncurr,
        )
        en = energies_and_force_norms(
            jacobian=jac, metrics=mets, fields=mf, trig=trig_new, s=s, signgs=signgs
        )
        tcon = constraint_scaling(
            tcon0=TCON0, geometry=geom, jacobian=jac,
            total_pressure=mf.total_pressure, trig=trig_new, s=s,
        )
        forces = newf.mhd_forces(
            geometry=geom,
            jacobian=jac,
            metrics=mets,
            fields=mf,
            R_cos=inputs["R_cos"],
            R_sin=inputs["R_sin"],
            Z_cos=inputs["Z_cos"],
            Z_sin=inputs["Z_sin"],
            modes=modes_new,
            trig=trig_new,
            s=s,
            phipf=jnp.asarray(flux.phipf),
            tcon=tcon,
            signgs=signgs,
        )
        spectral = newf.spectral_mhd_forces(
            forces, mpol=int(cfg.mpol), ntor=int(cfg.ntor), trig=trig_new,
            include_edge=include_edge,
        )
        rotated = newr.m1_residue_rotation(spectral, lconm1=bool(cfg.lconm1))
        released = newr.zero_m1_z_force(rotated, jnp.asarray(zero_m1))
        scaled = newr.scalxc_scale_force(released, s=s)
        resid = newr.force_residuals(
            scaled, fnorm=en.fnorm, fnormL=en.fnormL, r1=en.r1, include_edge=include_edge
        )

        # Preconditioned residual lane (fixed boundary: jmax = ns-1).
        common = dict(
            r12_half=jac.r12[1:],
            bsq_half=mf.total_pressure[1:],
            bsupv_half=mf.bsupv[1:],
            sqrt_g_half=jac.sqrt_g[1:],
            angular_weight=weights,
            delta_s=hs,
            ns=ns,
        )
        coeffs_R = newp.precondn(
            dxds_half=jac.dZ_ds[1:],
            dxdu_half=jac.zu12[1:],
            dxdu_even_full=geom.dZ_dtheta_even,
            dxdu_odd_full=geom.dZ_dtheta_odd,
            x_odd_full=geom.Z_odd,
            **common,
        )
        coeffs_Z = newp.precondn(
            dxds_half=jac.dR_ds[1:],
            dxdu_half=jac.ru12[1:],
            dxdu_even_full=geom.dR_dtheta_even,
            dxdu_odd_full=geom.dR_dtheta_odd,
            x_odd_full=geom.R_odd,
            **common,
        )
        mat_kwargs = dict(
            delta_s=hs, mpol=int(cfg.mpol), ntor=int(cfg.ntor),
            nfp=int(cfg.nfp), ns=ns, jmax=None,
        )
        mats_R = newp.scalfor_matrices(coeffs_R, stabilize_edge_zc00=False, **mat_kwargs)
        mats_Z = newp.scalfor_matrices(coeffs_Z, stabilize_edge_zc00=True, **mat_kwargs)
        if bool(cfg.lthreed) or bool(cfg.lasym):
            rhs = newr.scale_m1_preconditioner_rhs(
                scaled, coefficients_R=coeffs_R, coefficients_Z=coeffs_Z,
                lconm1=bool(cfg.lconm1),
            )
        else:
            rhs = scaled
        solved = newr.apply_radial_preconditioner(
            rhs, matrices_R=mats_R, matrices_Z=mats_Z, jmax=ns - 1
        )
        faclam = newp.lamcal(
            guu_half=mets.guu,
            guv_half=mets.guv,
            gvv_half=mets.gvv,
            sqrt_g_half=jac.sqrt_g,
            lamscale=mf.lamscale,
            angular_weight=weights,
            mpol=int(cfg.mpol),
            ntor=int(cfg.ntor),
            nfp=int(cfg.nfp),
            lthreed=bool(cfg.lthreed),
        )
        preconditioned = newr.apply_lambda_preconditioner(solved, faclam)
        fnorm1 = preconditioned_force_norm(
            R_cos=jnp.asarray(state.Rcos),
            Z_sin=jnp.asarray(state.Zsin),
            modes=modes_new,
            R_sin=jnp.asarray(state.Rsin) if bool(cfg.lasym) else None,
            Z_cos=jnp.asarray(state.Zcos) if bool(cfg.lasym) else None,
        )
        pre = newr.preconditioned_residuals(preconditioned, fnorm1=fnorm1, delta_s=hs)
        return SimpleNamespace(
            geom=geom, jac=jac, mets=mets, mf=mf, en=en, tcon=tcon,
            forces=forces, spectral=spectral, scaled=scaled, resid=resid,
            preconditioned=preconditioned, pre=pre,
        )

    new0 = new_chain(new_inputs, zero_m1=0.0)

    return SimpleNamespace(
        name=name,
        cfg=cfg,
        s=s,
        signgs=signgs,
        state=state,
        static=static,
        modes_new=modes_new,
        trig_new=trig_new,
        new_inputs=new_inputs,
        new_chain=new_chain,
        new0=new0,
        k_old=k_old,
        frzl_raw_old=frzl_raw_old,
        frzl_raw_edge_old=frzl_raw_edge_old,
        norms_old=norms_old,
        mats_old=mats_old,
        jmax_old=int(jmax_old),
        faclam_old=faclam_old,
        fnorm1_old=fnorm1_old,
    )


# ---------------------------------------------------------------------------
# residuals.py: m=1 coefficient mappings (residue.f90 / readin.f)
# ---------------------------------------------------------------------------


def test_m1_constrained_to_physical_matches_old(case):
    old = vmec_m1_internal_to_physical_signed(
        Rcos=case.state.Rcos,
        Zsin=case.state.Zsin,
        Rsin=case.state.Rsin,
        Zcos=case.state.Zcos,
        modes=case.static.modes,
        lthreed=bool(case.cfg.lthreed),
        lasym=bool(case.cfg.lasym),
        lconm1=bool(case.cfg.lconm1),
    )
    new = (
        case.new_inputs["R_cos"],
        case.new_inputs["Z_sin"],
        case.new_inputs["R_sin"],
        case.new_inputs["Z_cos"],
    )
    for name, new_c, old_c in zip(("R_cos", "Z_sin", "R_sin", "Z_cos"), new, old):
        _allclose(new_c, old_c, f"m1 physical {name}")


def test_m1_physical_to_constrained_matches_old_and_roundtrips(case):
    kwargs = dict(
        modes=case.modes_new,
        lthreed=bool(case.cfg.lthreed),
        lasym=bool(case.cfg.lasym),
        lconm1=bool(case.cfg.lconm1),
    )
    back = newr.m1_physical_to_constrained(
        case.new_inputs["R_cos"],
        case.new_inputs["Z_sin"],
        case.new_inputs["R_sin"],
        case.new_inputs["Z_cos"],
        **kwargs,
    )
    old_back = vmec_m1_physical_to_internal_signed(
        Rcos=case.new_inputs["R_cos"],
        Zsin=case.new_inputs["Z_sin"],
        Rsin=case.new_inputs["R_sin"],
        Zcos=case.new_inputs["Z_cos"],
        modes=case.static.modes,
        lthreed=bool(case.cfg.lthreed),
        lasym=bool(case.cfg.lasym),
        lconm1=bool(case.cfg.lconm1),
    )
    originals = (case.state.Rcos, case.state.Zsin, case.state.Rsin, case.state.Zcos)
    for name, new_c, old_c, orig in zip(("R_cos", "Z_sin", "R_sin", "Z_cos"), back, old_back, originals):
        _allclose(new_c, old_c, f"m1 internal {name}")
        _allclose(new_c, orig, f"m1 roundtrip {name}")


# ---------------------------------------------------------------------------
# forces.py: real-space kernels (forces.f, bcovar.f lambda block, alias.f)
# ---------------------------------------------------------------------------


def test_real_space_force_kernels_match_old(case):
    f = case.new0.forces
    k = case.k_old
    pairs = [
        ("force_R_even", f.force_R_even, k.armn_e),
        ("force_R_odd", f.force_R_odd, k.armn_o),
        ("force_R_du_even", f.force_R_du_even, k.brmn_e),
        ("force_R_du_odd", f.force_R_du_odd, k.brmn_o),
        ("force_R_dv_even", f.force_R_dv_even, k.crmn_e),
        ("force_R_dv_odd", f.force_R_dv_odd, k.crmn_o),
        ("force_Z_even", f.force_Z_even, k.azmn_e),
        ("force_Z_odd", f.force_Z_odd, k.azmn_o),
        ("force_Z_du_even", f.force_Z_du_even, k.bzmn_e),
        ("force_Z_du_odd", f.force_Z_du_odd, k.bzmn_o),
        ("force_Z_dv_even", f.force_Z_dv_even, k.czmn_e),
        ("force_Z_dv_odd", f.force_Z_dv_odd, k.czmn_o),
        ("force_lambda_du_even", f.force_lambda_du_even, k.bc.blmn_even),
        ("force_lambda_du_odd", f.force_lambda_du_odd, k.bc.blmn_odd),
        ("force_lambda_dv_even", f.force_lambda_dv_even, k.bc.clmn_even),
        ("force_lambda_dv_odd", f.force_lambda_dv_odd, k.bc.clmn_odd),
        ("constraint_R_even", f.constraint_R_even, k.arcon_e),
        ("constraint_R_odd", f.constraint_R_odd, k.arcon_o),
        ("constraint_Z_even", f.constraint_Z_even, k.azcon_e),
        ("constraint_Z_odd", f.constraint_Z_odd, k.azcon_o),
    ]
    for name, new_arr, old_arr in pairs:
        _allclose(new_arr, old_arr, name, atol=KERNEL_ATOL)


def test_constraint_force_gcon_matches_old(case):
    _allclose(case.new0.forces.gcon, case.k_old.gcon, "gcon")
    _allclose(case.new0.forces.rcon0, case.k_old.constraint_rcon0, "rcon0")
    _allclose(case.new0.forces.zcon0, case.k_old.constraint_zcon0, "zcon0")


# ---------------------------------------------------------------------------
# forces.py -> transforms.tomnsps/tomnspa: spectral force coefficients
# ---------------------------------------------------------------------------

_BLOCK_MAP = [
    ("force_R_cc", "frcc"),
    ("force_R_ss", "frss"),
    ("force_Z_sc", "fzsc"),
    ("force_Z_cs", "fzcs"),
    ("force_lambda_sc", "flsc"),
    ("force_lambda_cs", "flcs"),
    ("force_R_sc", "frsc"),
    ("force_R_cs", "frcs"),
    ("force_Z_cc", "fzcc"),
    ("force_Z_ss", "fzss"),
    ("force_lambda_cc", "flcc"),
    ("force_lambda_ss", "flss"),
]


def _assert_spectral_blocks(new_force, old_frzl, label):
    for new_name, old_name in _BLOCK_MAP:
        new_block = getattr(new_force, new_name)
        old_block = getattr(old_frzl, old_name, None)
        if old_block is None:
            assert new_block is None, f"{label}: {new_name} expected None"
        else:
            assert new_block is not None, f"{label}: {new_name} missing"
            _allclose(new_block, old_block, f"{label}: {new_name} ({old_name})")


def test_spectral_force_coefficients_match_old(case):
    _assert_spectral_blocks(case.new0.spectral, case.frzl_raw_old, "tomnsps raw")


def test_spectral_force_coefficients_match_old_include_edge(case):
    new_edge = case.new_chain(case.new_inputs, zero_m1=0.0, include_edge=True)
    _assert_spectral_blocks(new_edge.spectral, case.frzl_raw_edge_old, "tomnsps edge")


# ---------------------------------------------------------------------------
# residuals.py: m1 rotation + zero_m1 + scalxc stages and fsqr/fsqz/fsql
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zero_m1", [0.0, 1.0], ids=["m1free", "m1zero"])
def test_processed_force_blocks_match_old(case, zero_m1):
    old = _old_residual_lane(case, zero_m1=zero_m1, include_edge=False)
    new = case.new0 if zero_m1 == 0.0 else case.new_chain(case.new_inputs, zero_m1=zero_m1)
    _assert_spectral_blocks(new.scaled, old.frzl_full, f"post-scalxc zero_m1={zero_m1}")


@pytest.mark.parametrize("zero_m1", [0.0, 1.0], ids=["m1free", "m1zero"])
def test_fsq_residuals_match_old(case, zero_m1):
    old = _old_residual_lane(case, zero_m1=zero_m1, include_edge=False)
    new = case.new0 if zero_m1 == 0.0 else case.new_chain(case.new_inputs, zero_m1=zero_m1)
    _allclose(new.resid.fsqr, old.fsqr, "fsqr")
    _allclose(new.resid.fsqz, old.fsqz, "fsqz")
    _allclose(new.resid.fsql, old.fsql, "fsql")
    # Edge rows are masked in this lane -> fedge must vanish identically.
    assert float(new.resid.fedge) == 0.0


def test_fsq_residuals_match_old_include_edge(case):
    old = _old_residual_lane(case, zero_m1=0.0, include_edge=True)
    new = case.new_chain(case.new_inputs, zero_m1=0.0, include_edge=True)
    _allclose(new.resid.fsqr, old.fsqr, "fsqr (edge)")
    _allclose(new.resid.fsqz, old.fsqz, "fsqz (edge)")
    _allclose(new.resid.fsql, old.fsql, "fsql (edge)")
    # fedge (residue.f90) against a direct edge-row reduction of the old blocks.
    edge2 = 0.0
    for name in ("frcc", "frss", "fzsc", "fzcs", "frsc", "frcs", "fzcc", "fzss"):
        block = getattr(old.frzl_full, name, None)
        if block is not None:
            edge2 = edge2 + jnp.sum(jnp.asarray(block)[-1] ** 2)
    fedge_old = case.norms_old.r1 * case.norms_old.fnorm * edge2
    _allclose(new.resid.fedge, fedge_old, "fedge")


def test_release_conditions_are_traced_values(case):
    zero = newr.m1_zero_condition(
        fsqz_previous=jnp.asarray(1e-7), iterations_since_restart=jnp.asarray(100)
    )
    keep = newr.m1_zero_condition(
        fsqz_previous=jnp.asarray(1e-3), iterations_since_restart=jnp.asarray(100)
    )
    startup = newr.m1_zero_condition(
        fsqz_previous=jnp.asarray(1e-3), iterations_since_restart=jnp.asarray(0)
    )
    assert bool(zero) and not bool(keep) and bool(startup)

    edge_on = newr.edge_force_condition(
        fsq_rz_previous=jnp.asarray(1e-7),
        iterations_since_restart=jnp.asarray(10),
        free_boundary=True,
    )
    edge_off_fixedb = newr.edge_force_condition(
        fsq_rz_previous=jnp.asarray(1e-7),
        iterations_since_restart=jnp.asarray(10),
        free_boundary=False,
    )
    edge_off_late = newr.edge_force_condition(
        fsq_rz_previous=jnp.asarray(1e-7),
        iterations_since_restart=jnp.asarray(60),
        free_boundary=True,
    )
    assert bool(edge_on) and not bool(edge_off_fixedb) and not bool(edge_off_late)
    # jit-compatible (traced masks, no Python branching on values).
    assert bool(
        jax.jit(lambda f, i: newr.m1_zero_condition(fsqz_previous=f, iterations_since_restart=i))(
            jnp.asarray(1e-7), jnp.asarray(100)
        )
    )


# ---------------------------------------------------------------------------
# residuals.py + preconditioner.py: fsqr1/fsqz1/fsql1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zero_m1", [0.0, 1.0], ids=["m1free", "m1zero"])
def test_preconditioned_residuals_match_old(case, zero_m1):
    old = _old_preconditioned_lane(case, zero_m1=zero_m1)
    new = case.new0 if zero_m1 == 0.0 else case.new_chain(case.new_inputs, zero_m1=zero_m1)
    _assert_spectral_blocks(new.preconditioned, old.frzl_pre, f"preconditioned zero_m1={zero_m1}")
    _allclose(new.pre.fsqr1, old.fsqr1, "fsqr1")
    _allclose(new.pre.fsqz1, old.fsqz1, "fsqz1")
    _allclose(new.pre.fsql1, old.fsql1, "fsql1")


# ---------------------------------------------------------------------------
# jit-compatibility and differentiability of the full chain
# ---------------------------------------------------------------------------


def _chain_scalars(case, inputs):
    out = case.new_chain(inputs, zero_m1=0.0)
    return (
        out.resid.fsqr,
        out.resid.fsqz,
        out.resid.fsql,
        out.pre.fsqr1,
        out.pre.fsqz1,
        out.pre.fsql1,
    )


def test_full_chain_is_jittable(case):
    eager = _chain_scalars(case, case.new_inputs)
    jitted = jax.jit(lambda inputs: _chain_scalars(case, inputs))(case.new_inputs)
    for name, a, b in zip(("fsqr", "fsqz", "fsql", "fsqr1", "fsqz1", "fsql1"), jitted, eager):
        _allclose(a, b, f"jit {name}", rtol=1e-11, atol=1e-14)


def test_grad_of_fsqr_wrt_R_cos(case):
    def fsqr_of_R_cos(R_cos):
        inputs = dict(case.new_inputs)
        inputs["R_cos"] = R_cos
        return case.new_chain(inputs, zero_m1=0.0).resid.fsqr

    grad = jax.grad(fsqr_of_R_cos)(case.new_inputs["R_cos"])
    grad_np = np.asarray(grad)
    assert grad_np.shape == np.asarray(case.new_inputs["R_cos"]).shape
    assert np.all(np.isfinite(grad_np))
    assert np.any(grad_np != 0.0)

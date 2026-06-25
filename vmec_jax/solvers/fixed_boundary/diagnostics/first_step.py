"""Single-step VMEC residual diagnostic helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from ...._compat import jnp
from ....state import VMECState
from ....vmec_tomnsp import TomnspsRZL


@dataclass(frozen=True)
class FirstStepDependencies:
    """Private helper hooks used by the first-step diagnostic."""

    mode00_index_func: Callable[..., int]
    half_mesh_from_full_mesh_func: Callable[..., Any]
    mass_half_mesh_from_indata_func: Callable[..., Any]
    pressure_half_mesh_from_indata_func: Callable[..., Any]
    icurv_full_mesh_from_indata_func: Callable[..., Any]
    vmec_force_flux_profiles_func: Callable[..., tuple[Any, Any, Any]]
    zero_edge_rz_force_blocks_func: Callable[..., Any]
    radial_tridi_smooth_dirichlet_func: Callable[..., Any]
    metric_surface_precond_scales_np_func: Callable[..., Any]
    wout_like_vmec_forces_cls: type
    metric_surface_precond_from_bcovar_np_func: Callable[..., tuple[Any, Any]]


@dataclass(frozen=True)
class FirstStepForceSetup:
    """VMEC force-balance setup shared by the first-step diagnostic body."""

    cfg: Any
    static_vmec: Any
    s: Any
    wout_like: Any
    trig: Any
    zero_m1: bool
    apply_lforbal: bool
    mask_pack: Any


@dataclass(frozen=True)
class FirstStepPreconditionedForces:
    """Preconditioned Fourier residual blocks."""

    frzl_pre: TomnspsRZL
    frcc: Any
    frss: Any
    fzsc: Any
    fzcs: Any
    flsc: Any
    flcs: Any


@dataclass(frozen=True)
class FirstStepWeightedForces:
    """Mode-weighted residual blocks and per-mode RMS diagnostics."""

    frcc_u: Any
    frss_u: Any
    fzsc_u: Any
    fzcs_u: Any
    flsc_u: Any
    flcs_u: Any
    frcc_mode: Any
    fzsc_mode: Any
    flsc_mode: Any


def _resolve_first_step_dependencies(
    *,
    has_jax_func: Callable[[], bool] | None,
    mode00_index_func: Callable[..., int] | None,
    half_mesh_from_full_mesh_func: Callable[..., Any] | None,
    mass_half_mesh_from_indata_func: Callable[..., Any] | None,
    pressure_half_mesh_from_indata_func: Callable[..., Any] | None,
    icurv_full_mesh_from_indata_func: Callable[..., Any] | None,
    vmec_force_flux_profiles_func: Callable[..., tuple[Any, Any, Any]] | None,
    zero_edge_rz_force_blocks_func: Callable[..., Any] | None,
    radial_tridi_smooth_dirichlet_func: Callable[..., Any] | None,
    metric_surface_precond_scales_np_func: Callable[..., Any] | None,
    wout_like_vmec_forces_cls: type | None,
) -> FirstStepDependencies:
    """Resolve injectable dependencies after the cheap JAX availability check."""

    if has_jax_func is None:
        from ...._compat import has_jax as has_jax_func

    if not has_jax_func():
        raise ImportError("first_step_diagnostics requires JAX (jax + jaxlib)")

    if mode00_index_func is None:
        from ..optimization.constraints import mode00_index as mode00_index_func
    if half_mesh_from_full_mesh_func is None:
        from ..profiles import _half_mesh_from_full_mesh as half_mesh_from_full_mesh_func
    if mass_half_mesh_from_indata_func is None:
        from ..profiles import _mass_half_mesh_from_indata as mass_half_mesh_from_indata_func
    if pressure_half_mesh_from_indata_func is None:
        from ..profiles import _pressure_half_mesh_from_indata as pressure_half_mesh_from_indata_func
    if icurv_full_mesh_from_indata_func is None:
        from ..profiles import _icurv_full_mesh_from_indata as icurv_full_mesh_from_indata_func
    if vmec_force_flux_profiles_func is None:
        from ..profiles import _vmec_force_flux_profiles as vmec_force_flux_profiles_func
    if zero_edge_rz_force_blocks_func is None:
        from ..residual.payload_blocks import (
            zero_edge_rz_force_blocks as zero_edge_rz_force_blocks_func,
        )
    if radial_tridi_smooth_dirichlet_func is None:
        from ..preconditioning.operators import (
            radial_tridi_smooth_dirichlet as radial_tridi_smooth_dirichlet_func,
        )
    if metric_surface_precond_scales_np_func is None:
        from ..preconditioning.operators import (
            metric_surface_precond_scales_np as metric_surface_precond_scales_np_func,
        )
    from ..preconditioning.operators import metric_surface_precond_from_bcovar_np
    if wout_like_vmec_forces_cls is None:
        from ..results import WoutLikeVmecForces as wout_like_vmec_forces_cls

    return FirstStepDependencies(
        mode00_index_func=mode00_index_func,
        half_mesh_from_full_mesh_func=half_mesh_from_full_mesh_func,
        mass_half_mesh_from_indata_func=mass_half_mesh_from_indata_func,
        pressure_half_mesh_from_indata_func=pressure_half_mesh_from_indata_func,
        icurv_full_mesh_from_indata_func=icurv_full_mesh_from_indata_func,
        vmec_force_flux_profiles_func=vmec_force_flux_profiles_func,
        zero_edge_rz_force_blocks_func=zero_edge_rz_force_blocks_func,
        radial_tridi_smooth_dirichlet_func=radial_tridi_smooth_dirichlet_func,
        metric_surface_precond_scales_np_func=metric_surface_precond_scales_np_func,
        wout_like_vmec_forces_cls=wout_like_vmec_forces_cls,
        metric_surface_precond_from_bcovar_np_func=metric_surface_precond_from_bcovar_np,
    )


def _build_first_step_force_setup(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    zero_m1: bool,
    deps: FirstStepDependencies,
) -> FirstStepForceSetup:
    """Build flux/profile/trig state needed to evaluate the first residual."""

    from ....boundary import boundary_from_indata
    from ....energy import flux_profiles_from_indata
    from ....static import build_static
    from ....vmec_tomnsp import vmec_angle_grid, vmec_trig_tables

    cfg = static.cfg
    grid_vmec = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    static_vmec = build_static(cfg, grid=grid_vmec)
    s = jnp.asarray(static_vmec.s)
    flux = flux_profiles_from_indata(indata, s, signgs=int(signgs))
    chipf_wout = jnp.asarray(flux.chipf)
    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    boundary = boundary_from_indata(indata, static_vmec.modes)
    idx00 = deps.mode00_index_func(static_vmec.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = deps.half_mesh_from_full_mesh_func(chipf_wout) if lrfp else None
    mass = deps.mass_half_mesh_from_indata_func(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )
    pres = deps.pressure_half_mesh_from_indata_func(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = deps.icurv_full_mesh_from_indata_func(indata=indata, s_full=s, signgs=int(signgs))
    phipf_internal, chipf_internal, chips_eff = deps.vmec_force_flux_profiles_func(
        phipf=jnp.asarray(flux.phipf),
        chipf=chipf_wout,
        signgs=int(signgs),
        flux_is_internal=True,
    )

    wout_like = deps.wout_like_vmec_forces_cls(
        nfp=int(cfg.nfp),
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        lasym=bool(cfg.lasym),
        signgs=int(signgs),
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
        phipf_internal=phipf_internal,
        chipf_internal=chipf_internal,
        chips_eff=chips_eff,
    )

    trig = getattr(static_vmec, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
    if not bool(wout_like.lasym):
        # For lasym=False keep Z-force intact in the first-step diagnostic.
        zero_m1 = False

    return FirstStepForceSetup(
        cfg=cfg,
        static_vmec=static_vmec,
        s=s,
        wout_like=wout_like,
        trig=trig,
        zero_m1=bool(zero_m1),
        apply_lforbal=bool(indata.get_bool("LFORBAL", False)) if indata is not None else False,
        mask_pack=getattr(static_vmec, "tomnsps_masks", None),
    )


def _precondition_first_step_forces(
    *,
    frzl: TomnspsRZL,
    k: Any,
    cfg: Any,
    trig: Any,
    s: Any,
    rz_scale: Any,
    l_scale: Any,
    deps: FirstStepDependencies,
    precond_radial_alpha: float,
    precond_lambda_alpha: float,
    use_axisymmetric_preconditioner: bool,
) -> FirstStepPreconditionedForces:
    """Apply either the axisymmetric or radial-tridiagonal first-step preconditioner."""

    def _apply_radial_tridi(rhs, alpha: float):
        return deps.radial_tridi_smooth_dirichlet_func(rhs, alpha=alpha, skip_nonpositive=True)

    if bool(use_axisymmetric_preconditioner) and (not bool(cfg.lthreed)) and (not bool(cfg.lasym)):
        from ....preconditioner_1d_jax import lambda_preconditioner, rz_preconditioner

        lam_prec = lambda_preconditioner(bc=k.bc, trig=trig, s=s, cfg=cfg)
        frzl_pre = rz_preconditioner(frzl_in=frzl, bc=k.bc, k=k, trig=trig, s=s, cfg=cfg)
        frcc = jnp.asarray(frzl_pre.frcc)
        frss = frzl_pre.frss
        fzsc = jnp.asarray(frzl_pre.fzsc)
        fzcs = frzl_pre.fzcs
        flsc = jnp.asarray(frzl_pre.flsc) * jnp.asarray(lam_prec)
        flcs = frzl_pre.flcs
        if not (jnp.all(jnp.isfinite(frcc)) and jnp.all(jnp.isfinite(fzsc)) and jnp.all(jnp.isfinite(flsc))):
            frcc = jnp.asarray(frzl.frcc)
            frss = frzl.frss
            fzsc = jnp.asarray(frzl.fzsc)
            fzcs = frzl.fzcs
            flsc = jnp.asarray(frzl.flsc)
            flcs = frzl.flcs
    else:
        frcc = _apply_radial_tridi(frzl.frcc * rz_scale[:, None, None], precond_radial_alpha)
        frss = (
            _apply_radial_tridi(frzl.frss * rz_scale[:, None, None], precond_radial_alpha)
            if frzl.frss is not None
            else None
        )
        fzsc = _apply_radial_tridi(frzl.fzsc * rz_scale[:, None, None], precond_radial_alpha)
        fzcs = (
            _apply_radial_tridi(frzl.fzcs * rz_scale[:, None, None], precond_radial_alpha)
            if frzl.fzcs is not None
            else None
        )
        flsc = _apply_radial_tridi(frzl.flsc * l_scale[:, None, None], precond_lambda_alpha)
        flcs = (
            _apply_radial_tridi(frzl.flcs * l_scale[:, None, None], precond_lambda_alpha)
            if frzl.flcs is not None
            else None
        )

    frzl_pre = TomnspsRZL(
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=flsc,
        flcs=flcs,
        frsc=getattr(frzl, "frsc", None),
        frcs=getattr(frzl, "frcs", None),
        fzcc=getattr(frzl, "fzcc", None),
        fzss=getattr(frzl, "fzss", None),
        flcc=getattr(frzl, "flcc", None),
        flss=getattr(frzl, "flss", None),
    )
    return FirstStepPreconditionedForces(
        frzl_pre=frzl_pre,
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=flsc,
        flcs=flcs,
    )


def _weight_first_step_forces(
    pre: FirstStepPreconditionedForces,
    *,
    cfg: Any,
    mode_diag_exponent: float,
) -> FirstStepWeightedForces:
    """Apply the diagnostic mode weights used for first-step update previews."""

    mpol = int(cfg.mpol)
    ntor = int(cfg.ntor)
    nrange = ntor + 1
    nfp = float(cfg.nfp)
    w_mode_mn = (1.0 + (jnp.arange(mpol)[:, None] ** 2 + (jnp.arange(nrange)[None, :] * nfp) ** 2)) ** (
        -float(mode_diag_exponent)
    )
    frcc_u = pre.frcc * w_mode_mn[None, :, :]
    frss_u = (pre.frss if pre.frss is not None else jnp.zeros_like(frcc_u)) * w_mode_mn[None, :, :]
    fzsc_u = pre.fzsc * w_mode_mn[None, :, :]
    fzcs_u = (pre.fzcs if pre.fzcs is not None else jnp.zeros_like(fzsc_u)) * w_mode_mn[None, :, :]
    flsc_u = pre.flsc * w_mode_mn[None, :, :]
    flcs_u = (pre.flcs if pre.flcs is not None else jnp.zeros_like(flsc_u)) * w_mode_mn[None, :, :]

    def _mode_rms(a):
        a = jnp.asarray(a)
        return jnp.sqrt(jnp.mean(a * a, axis=0))

    return FirstStepWeightedForces(
        frcc_u=frcc_u,
        frss_u=frss_u,
        fzsc_u=fzsc_u,
        fzcs_u=fzcs_u,
        flsc_u=flsc_u,
        flcs_u=flcs_u,
        frcc_mode=_mode_rms(frcc_u),
        fzsc_mode=_mode_rms(fzsc_u),
        flsc_mode=_mode_rms(flsc_u),
    )


def _first_step_update_preview(weighted: FirstStepWeightedForces, *, time_step: float) -> dict[str, Any]:
    """Return the VMEC-style first-step velocity/displacement preview arrays."""

    invtau = 0.15 / float(time_step)
    otav = invtau
    dtau = float(time_step) * otav / 2.0
    b1 = 1.0 - dtau
    fac = 1.0 / (1.0 + dtau)
    vRcc = fac * float(time_step) * weighted.frcc_u
    vRss = fac * float(time_step) * weighted.frss_u
    vZsc = fac * float(time_step) * weighted.fzsc_u
    vZcs = fac * float(time_step) * weighted.fzcs_u
    vLsc = fac * float(time_step) * weighted.flsc_u
    vLcs = fac * float(time_step) * weighted.flcs_u
    return {
        "dtau": float(dtau),
        "b1": float(b1),
        "fac": float(fac),
        "dRcc": np.asarray(float(time_step) * vRcc),
        "dRss": np.asarray(float(time_step) * vRss),
        "dZsc": np.asarray(float(time_step) * vZsc),
        "dZcs": np.asarray(float(time_step) * vZcs),
        "dLsc": np.asarray(float(time_step) * vLsc),
        "dLcs": np.asarray(float(time_step) * vLcs),
    }


def first_step_diagnostics_impl(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    step_size: float | None = None,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    precond_radial_alpha: float = 0.5,
    precond_lambda_alpha: float = 0.5,
    mode_diag_exponent: float = 1.0,
    include_edge: bool = True,
    zero_m1: bool = True,
    use_axisymmetric_preconditioner: bool = False,
    has_jax_func: Callable[[], bool] | None = None,
    mode00_index_func: Callable[..., int] | None = None,
    half_mesh_from_full_mesh_func: Callable[..., Any] | None = None,
    mass_half_mesh_from_indata_func: Callable[..., Any] | None = None,
    pressure_half_mesh_from_indata_func: Callable[..., Any] | None = None,
    icurv_full_mesh_from_indata_func: Callable[..., Any] | None = None,
    vmec_force_flux_profiles_func: Callable[..., tuple[Any, Any, Any]] | None = None,
    zero_edge_rz_force_blocks_func: Callable[..., Any] | None = None,
    radial_tridi_smooth_dirichlet_func: Callable[..., Any] | None = None,
    metric_surface_precond_scales_np_func: Callable[..., Any] | None = None,
    wout_like_vmec_forces_cls: type | None = None,
) -> Dict[str, Any]:
    """Return a first-step diagnostic bundle.

    The caller injects the solve-module helper aliases so existing tests and
    downstream private monkeypatch hooks keep the historical behavior while the
    implementation lives outside the solver monolith.
    """

    deps = _resolve_first_step_dependencies(
        has_jax_func=has_jax_func,
        mode00_index_func=mode00_index_func,
        half_mesh_from_full_mesh_func=half_mesh_from_full_mesh_func,
        mass_half_mesh_from_indata_func=mass_half_mesh_from_indata_func,
        pressure_half_mesh_from_indata_func=pressure_half_mesh_from_indata_func,
        icurv_full_mesh_from_indata_func=icurv_full_mesh_from_indata_func,
        vmec_force_flux_profiles_func=vmec_force_flux_profiles_func,
        zero_edge_rz_force_blocks_func=zero_edge_rz_force_blocks_func,
        radial_tridi_smooth_dirichlet_func=radial_tridi_smooth_dirichlet_func,
        metric_surface_precond_scales_np_func=metric_surface_precond_scales_np_func,
        wout_like_vmec_forces_cls=wout_like_vmec_forces_cls,
    )

    from ....vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from ....vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_gcx2_from_tomnsps,
        vmec_rz_norm_from_state,
        vmec_scalxc_from_s,
        vmec_wint_from_trig,
        vmec_zero_m1_zforce,
    )

    signgs = int(signgs)
    setup = _build_first_step_force_setup(
        state0,
        static,
        indata=indata,
        signgs=signgs,
        zero_m1=bool(zero_m1),
        deps=deps,
    )
    cfg = setup.cfg
    static_vmec = setup.static_vmec
    s = setup.s
    wout_like = setup.wout_like
    trig = setup.trig
    zero_m1 = setup.zero_m1
    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    apply_lforbal = bool(setup.apply_lforbal)

    def _metric_surface_precond_from_bcovar(bc):
        """Approximate radial preconditioner scaling from bcovar metrics."""

        return deps.metric_surface_precond_from_bcovar_np_func(
            bc=bc,
            trig=trig,
            wint_from_trig_func=vmec_wint_from_trig,
            scales_func=deps.metric_surface_precond_scales_np_func,
        )

    mask_pack = setup.mask_pack

    def _compute_forces(state: VMECState):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static_vmec,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        frzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(cfg.ntheta),
            cfg_nzeta=int(cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=False,
            masks=mask_pack,
        )
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)
        frzl_raw = frzl
        if bool(apply_m1_constraints):
            frzl = vmec_apply_m1_constraints(
                frzl=frzl,
                lconm1=bool(getattr(cfg, "lconm1", True)),
            )
        frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=jnp.asarray(float(bool(zero_m1))))
        frzl = deps.zero_edge_rz_force_blocks_func(frzl, preserve_numpy=False)
        gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
            frzl=frzl,
            lconm1=bool(getattr(cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=bool(include_edge),
            apply_scalxc=False,
            s=s,
        )
        gcr2_raw, gcz2_raw, gcl2_raw = vmec_gcx2_from_tomnsps(
            frzl=frzl_raw,
            lconm1=bool(getattr(cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=bool(include_edge),
            apply_scalxc=False,
            s=s,
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        fsqr = norms.r1 * norms.fnorm * gcr2
        fsqz = norms.r1 * norms.fnorm * gcz2
        fsql = norms.fnormL * gcl2
        rz_scale, l_scale = _metric_surface_precond_from_bcovar(k.bc)
        return k, frzl, fsqr, fsqz, fsql, rz_scale, l_scale, (gcr2_raw, gcz2_raw, gcl2_raw)

    k, frzl, fsqr, fsqz, fsql, rz_scale, l_scale, g_raw = _compute_forces(state0)
    gcr2_raw, gcz2_raw, gcl2_raw = g_raw

    pre = _precondition_first_step_forces(
        frzl=frzl,
        k=k,
        cfg=cfg,
        trig=trig,
        s=s,
        rz_scale=rz_scale,
        l_scale=l_scale,
        deps=deps,
        precond_radial_alpha=float(precond_radial_alpha),
        precond_lambda_alpha=float(precond_lambda_alpha),
        use_axisymmetric_preconditioner=bool(use_axisymmetric_preconditioner),
    )

    weighted = _weight_first_step_forces(pre, cfg=cfg, mode_diag_exponent=float(mode_diag_exponent))

    gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
        frzl=pre.frzl_pre,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        apply_m1_constraints=False,
        include_edge=True,
        apply_scalxc=False,
        s=s,
    )
    rz_norm = vmec_rz_norm_from_state(
        state=state0,
        static=static,
        s=s,
        apply_scalxc=False,
        ns_min=0,
        ns_max=int(jnp.asarray(s).shape[0]),
    )
    norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
    f_norm1 = jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=rz_norm.dtype))
    delta_s = jnp.asarray(s[1] - s[0], dtype=rz_norm.dtype)
    fsqr1 = gcr2_p * f_norm1
    fsqz1 = gcz2_p * f_norm1
    fsql1 = gcl2_p * delta_s

    if step_size is None:
        time_step = float(indata.get_float("DELT", 5e-3))
    else:
        time_step = float(step_size)
    update_preview = _first_step_update_preview(weighted, time_step=float(time_step))

    return {
        "fsqr": float(np.asarray(fsqr)),
        "fsqz": float(np.asarray(fsqz)),
        "fsql": float(np.asarray(fsql)),
        "fsqr1": float(np.asarray(fsqr1)),
        "fsqz1": float(np.asarray(fsqz1)),
        "fsql1": float(np.asarray(fsql1)),
        "gcr2_raw": float(np.asarray(gcr2_raw)),
        "gcz2_raw": float(np.asarray(gcz2_raw)),
        "gcl2_raw": float(np.asarray(gcl2_raw)),
        "rz_norm": float(np.asarray(rz_norm)),
        "f_norm1": float(np.asarray(f_norm1)),
        "f_norm_rz": float(np.asarray(norms.fnorm)),
        "f_norm_l": float(np.asarray(norms.fnormL)),
        "scalxc": np.asarray(vmec_scalxc_from_s(s=s, mpol=int(cfg.mpol))),
        "time_step": float(time_step),
        "dtau": update_preview["dtau"],
        "b1": update_preview["b1"],
        "fac": update_preview["fac"],
        "rz_scale": np.asarray(rz_scale),
        "l_scale": np.asarray(l_scale),
        "frzl": frzl,
        "frzl_pre": pre.frzl_pre,
        "frcc_u": np.asarray(weighted.frcc_u),
        "frss_u": np.asarray(weighted.frss_u),
        "fzsc_u": np.asarray(weighted.fzsc_u),
        "fzcs_u": np.asarray(weighted.fzcs_u),
        "flsc_u": np.asarray(weighted.flsc_u),
        "flcs_u": np.asarray(weighted.flcs_u),
        "frcc_mode_rms": np.asarray(weighted.frcc_mode),
        "fzsc_mode_rms": np.asarray(weighted.fzsc_mode),
        "flsc_mode_rms": np.asarray(weighted.flsc_mode),
        "dRcc": update_preview["dRcc"],
        "dRss": update_preview["dRss"],
        "dZsc": update_preview["dZsc"],
        "dZcs": update_preview["dZcs"],
        "dLsc": update_preview["dLsc"],
        "dLcs": update_preview["dLcs"],
        "bcovar": k.bc,
    }

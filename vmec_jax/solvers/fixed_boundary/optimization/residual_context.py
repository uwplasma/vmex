"""Shared setup for VMEC residual-objective optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ...._compat import jnp


@dataclass(frozen=True)
class ResidualForceContext:
    """Static data reused by residual-objective optimizers."""

    idx00: int
    signgs: int
    s: Any
    wout_like: Any
    trig: Any
    constraint_tcon0: float | None
    apply_lforbal: bool
    ftol_target: float
    edge_Rcos: Any
    edge_Rsin: Any
    edge_Zcos: Any
    edge_Zsin: Any
    mask_pack: Any


def resolve_residual_trig(
    *,
    state0,
    static,
    wout_like,
    vmec_trig_tables_func: Callable[..., Any],
    jnp_module: Any = jnp,
):
    """Return VMEC-grid trig tables compatible with one residual solve."""

    trig = getattr(static, "trig_vmec", None)
    needs_build = trig is None
    if not needs_build:
        try:
            needs_build = (
                int(trig.ntheta1) != int(static.cfg.ntheta)
                or int(trig.cosnv.shape[0]) != int(static.cfg.nzeta)
                or int(trig.cosmu.shape[1]) != int(wout_like.mpol)
                or int(trig.cosnv.shape[1]) != int(wout_like.ntor) + 1
            )
        except AttributeError:
            # Lightweight tests and monkeypatch workflows sometimes inject a
            # sentinel trig object.  Preserve that explicit dependency-injection
            # seam instead of forcing a rebuild.
            needs_build = False
    if needs_build:
        trig = vmec_trig_tables_func(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp_module.asarray(state0.Rcos).dtype,
        )
    return trig


def prepare_residual_force_context(
    state0,
    static,
    *,
    indata,
    signgs: int,
    idx00: int,
    include_constraint_force: bool,
    mode00_index_func: Callable[..., int] | None = None,
    half_mesh_from_full_mesh_func: Callable[..., Any] | None = None,
    mass_half_mesh_from_indata_func: Callable[..., Any] | None = None,
    pressure_half_mesh_from_indata_func: Callable[..., Any] | None = None,
    icurv_full_mesh_from_indata_func: Callable[..., Any] | None = None,
    vmec_force_flux_profiles_func: Callable[..., tuple[Any, Any, Any]] | None = None,
    wout_like_cls: type | None = None,
    flux_profiles_from_indata_func: Callable[..., Any] | None = None,
    boundary_from_indata_func: Callable[..., Any] | None = None,
    vmec_trig_tables_func: Callable[..., Any] | None = None,
    jnp_module: Any | None = None,
) -> ResidualForceContext:
    """Prepare flux/profile/trig data for residual-objective optimizers.

    The public wrappers in :mod:`vmec_jax.solve` pass their local helper aliases
    into this routine so tests that monkeypatch those compatibility seams keep
    exercising the same code path after refactors.
    """

    if jnp_module is None:
        jnp_module = jnp
    if mode00_index_func is None:
        from .constraints import mode00_index as mode00_index_func
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
    if wout_like_cls is None:
        from ..results import WoutLikeVmecForces as wout_like_cls
    if flux_profiles_from_indata_func is None:
        from ....energy import flux_profiles_from_indata as flux_profiles_from_indata_func
    if boundary_from_indata_func is None:
        from ....boundary import boundary_from_indata as boundary_from_indata_func
    if vmec_trig_tables_func is None:
        from ....kernels.tomnsp import vmec_trig_tables as vmec_trig_tables_func

    signgs = int(signgs)
    idx00 = int(idx00 if idx00 is not None else mode00_index_func(static.modes))
    s = jnp_module.asarray(static.s)

    flux = flux_profiles_from_indata_func(indata, s, signgs=signgs)
    chipf_wout = jnp_module.asarray(flux.chipf)

    phips = jnp_module.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    boundary = boundary_from_indata_func(indata, static.modes)
    r00 = float(np.asarray(boundary.R_cos)[idx00]) if idx00 >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = half_mesh_from_full_mesh_func(chipf_wout) if lrfp else None
    mass = mass_half_mesh_from_indata_func(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )

    pres = pressure_half_mesh_from_indata_func(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = icurv_full_mesh_from_indata_func(indata=indata, s_full=s, signgs=signgs)
    phipf_internal, chipf_internal, chips_eff = vmec_force_flux_profiles_func(
        phipf=jnp_module.asarray(flux.phipf),
        chipf=chipf_wout,
        signgs=signgs,
        flux_is_internal=True,
    )

    wout_like = wout_like_cls(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs,
        phipf=jnp_module.asarray(flux.phipf),
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

    trig = resolve_residual_trig(
        state0=state0,
        static=static,
        wout_like=wout_like,
        vmec_trig_tables_func=vmec_trig_tables_func,
        jnp_module=jnp_module,
    )

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))

    return ResidualForceContext(
        idx00=idx00,
        signgs=signgs,
        s=s,
        wout_like=wout_like,
        trig=trig,
        constraint_tcon0=constraint_tcon0,
        apply_lforbal=bool(indata.get_bool("LFORBAL", False)) if indata is not None else False,
        ftol_target=max(0.0, float(indata.get_float("FTOL", 0.0))),
        edge_Rcos=jnp_module.asarray(state0.Rcos)[-1, :],
        edge_Rsin=jnp_module.asarray(state0.Rsin)[-1, :],
        edge_Zcos=jnp_module.asarray(state0.Zcos)[-1, :],
        edge_Zsin=jnp_module.asarray(state0.Zsin)[-1, :],
        mask_pack=getattr(static, "tomnsps_masks", None),
    )


def residual_terms_from_force_context(
    *,
    context: ResidualForceContext,
    state,
    static,
    zero_m1_zforce: Any,
    w_rz: float,
    w_l: float,
    apply_m1_constraints: bool,
    zero_m1_after_m1_constraints: bool,
    include_edge: bool,
    zero_edge_rz_blocks: bool,
    objective_scale: float | None,
    assemble_residual_objective_terms_func: Callable[..., Any] | None = None,
    compute_jac_min: bool = False,
    jnp_module: Any | None = None,
) -> tuple[Any, Any | None]:
    """Evaluate VMEC residual objective terms from a prepared force context.

    L-BFGS and Gauss-Newton use different step policies, but the physics work is
    the same: build VMEC force kernels, convert them to residual blocks, compute
    VMEC normalization factors, then assemble the weighted objective terms.
    Keeping that seam here avoids policy-specific copies drifting apart.
    """

    if jnp_module is None:
        jnp_module = jnp
    if assemble_residual_objective_terms_func is None:
        from .residual_objective import assemble_residual_objective_terms as assemble_residual_objective_terms_func
    from ....kernels.forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from ....kernels.residue import vmec_force_norms_from_bcovar_dynamic

    kernels = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=context.wout_like,
        indata=None,
        constraint_tcon0=context.constraint_tcon0,
        use_vmec_synthesis=True,
        trig=context.trig,
    )
    rzl = vmec_residual_internal_from_kernels(
        kernels,
        cfg_ntheta=int(static.cfg.ntheta),
        cfg_nzeta=int(static.cfg.nzeta),
        wout=context.wout_like,
        trig=context.trig,
        apply_lforbal=context.apply_lforbal,
        include_edge=False,
        masks=context.mask_pack,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(
        bc=kernels.bc,
        trig=context.trig,
        s=context.s,
        signgs=context.signgs,
    )
    terms = assemble_residual_objective_terms_func(
        frzl=rzl,
        norms=norms,
        s=context.s,
        w_rz=w_rz,
        w_l=w_l,
        zero_m1_zforce=zero_m1_zforce,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        apply_m1_constraints=bool(apply_m1_constraints),
        zero_m1_after_m1_constraints=bool(zero_m1_after_m1_constraints),
        include_edge=bool(include_edge),
        apply_scalxc=True,
        zero_edge_rz_blocks=bool(zero_edge_rz_blocks),
        objective_scale=objective_scale,
    )

    jac_min = None
    if compute_jac_min:
        jac = context.signgs * jnp_module.asarray(kernels.bc.jac.sqrtg)
        jac_min = jnp_module.min(jac) if jac.shape[0] <= 1 else jnp_module.min(jac[1:, :, :])
    return terms, jac_min

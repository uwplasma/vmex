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
        from ....solve_profile_helpers import _half_mesh_from_full_mesh as half_mesh_from_full_mesh_func
    if mass_half_mesh_from_indata_func is None:
        from ....solve_profile_helpers import _mass_half_mesh_from_indata as mass_half_mesh_from_indata_func
    if pressure_half_mesh_from_indata_func is None:
        from ....solve_profile_helpers import _pressure_half_mesh_from_indata as pressure_half_mesh_from_indata_func
    if icurv_full_mesh_from_indata_func is None:
        from ....solve_profile_helpers import _icurv_full_mesh_from_indata as icurv_full_mesh_from_indata_func
    if vmec_force_flux_profiles_func is None:
        from ....solve_profile_helpers import _vmec_force_flux_profiles as vmec_force_flux_profiles_func
    if wout_like_cls is None:
        from ....solve_result_types import WoutLikeVmecForces as wout_like_cls

    from ....boundary import boundary_from_indata
    from ....energy import flux_profiles_from_indata
    from ....vmec_tomnsp import vmec_trig_tables

    signgs = int(signgs)
    idx00 = int(idx00 if idx00 is not None else mode00_index_func(static.modes))
    s = jnp_module.asarray(static.s)

    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = jnp_module.asarray(flux.chipf)

    phips = jnp_module.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    boundary = boundary_from_indata(indata, static.modes)
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

    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp_module.asarray(state0.Rcos).dtype,
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

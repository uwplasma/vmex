"""Finite-beta optimization diagnostics and residual helpers.

These helpers are intentionally VMEC-state based and JAX differentiable.  They
cover the global stage-one finite-beta quantities that are cheap and stable to
differentiate through the fixed-boundary discrete-adjoint path: aspect ratio,
iota bounds, volume-averaged field proxy, and total beta.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._compat import jnp
from .energy import flux_profiles_from_indata
from .profiles import eval_profiles
from .solve import _half_mesh_from_full_mesh, _icurv_full_mesh_from_indata, _mass_half_mesh_from_indata
from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from .vmec_residue import vmec_force_norms_from_bcovar_dynamic
from .wout import _chipf_from_chips, equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


@dataclass(frozen=True)
class FiniteBetaTargets:
    """Targets and weights for stage-one finite-beta fixed-boundary objectives."""

    aspect_ratio: float
    min_iota: float
    min_average_iota: float
    max_iota: float
    volavgB: float
    beta_total: float
    aspect_weight: float = 1.0
    iota_weight: float = 1.0
    max_iota_weight: float = 1.0
    volavgB_weight: float = 1.0
    beta_weight: float = 1.0


def _s_half_from_static(static):
    s = jnp.asarray(static.s)
    if int(s.shape[0]) < 2:
        return s
    return jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)


def _wout_like_for_state(*, state, static, indata, signgs: int):
    s = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=int(signgs))
    phips = jnp.asarray(flux.phips)
    if int(phips.shape[0]) > 0:
        phips = phips.at[0].set(0.0)

    s_half = _s_half_from_static(static)
    prof = eval_profiles(indata, s_half)
    pres = jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))
    if int(pres.shape[0]) > 0:
        pres = pres.at[0].set(0.0)

    chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    chipf = _chipf_from_chips(chips)

    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static.modes)
    mode_m = np.asarray(static.modes.m)
    mode_n = np.asarray(static.modes.n)
    idx00 = np.where((mode_m == 0) & (mode_n == 0))[0]
    r00 = float(np.asarray(boundary.R_cos)[int(idx00[0])]) if idx00.size else float(np.asarray(boundary.R_cos)[0])

    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips_half = _half_mesh_from_full_mesh(jnp.asarray(flux.chipf)) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips_half,
    )
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=int(signgs))

    wout_like = SimpleNamespace(
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=jnp.asarray(chipf),
        iotaf=jnp.asarray(iotaf),
        iotas=jnp.asarray(iotas),
        signgs=int(signgs),
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        flux_is_internal=True,
        ncurr=int(indata.get_int("NCURR", 0)),
        lcurrent=bool(indata.get_int("NCURR", 0) == 1),
        icurv=jnp.asarray(icurv),
        mass=jnp.asarray(mass),
        gamma=gamma,
    )
    return wout_like, pres


def finite_beta_scalars_from_state(*, state, static, indata, signgs: int) -> dict[str, Any]:
    """Return JAX-differentiable finite-beta scalar diagnostics from a VMEC state."""
    aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
    _chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    iotaf = jnp.asarray(iotaf, dtype=jnp.float64)

    wout_like, pres = _wout_like_for_state(state=state, static=static, indata=indata, signgs=int(signgs))
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
    norms = vmec_force_norms_from_bcovar_dynamic(
        bc=bc,
        trig=static.trig_vmec,
        s=jnp.asarray(static.s),
        signgs=int(signgs),
    )
    beta_total = jnp.where(norms.wb != 0.0, norms.wp / norms.wb, jnp.asarray(0.0, dtype=norms.wb.dtype))
    volavgB = jnp.sqrt(jnp.maximum(2.0 * norms.wb / jnp.maximum(norms.volume, 1e-300), 0.0))
    return {
        "aspect": aspect,
        "iotas": jnp.asarray(iotas, dtype=jnp.float64),
        "iotaf": iotaf,
        "mean_iota": jnp.mean(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0),
        "min_iota": jnp.min(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0),
        "max_iota": jnp.max(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0),
        "volavgB": volavgB,
        "betatotal": beta_total,
        "wb": norms.wb,
        "wp": norms.wp,
        "vp": getattr(norms, "vp", jnp.zeros_like(jnp.asarray(static.s))),
        "volume": norms.volume,
    }


def finite_beta_global_residuals_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    targets: FiniteBetaTargets,
) -> jnp.ndarray:
    """Build global finite-beta residuals for stage-one surface optimization."""
    scalars = finite_beta_scalars_from_state(state=state, static=static, indata=indata, signgs=int(signgs))
    aspect_res = jnp.maximum(scalars["aspect"] - float(targets.aspect_ratio), 0.0)
    min_iota_res = jnp.minimum(scalars["min_iota"] - float(targets.min_iota), 0.0)
    mean_iota_res = jnp.minimum(scalars["mean_iota"] - float(targets.min_average_iota), 0.0)
    max_iota_res = jnp.maximum(scalars["max_iota"] - float(targets.max_iota), 0.0)
    return jnp.asarray(
        [
            float(targets.aspect_weight) * aspect_res,
            float(targets.iota_weight) * min_iota_res,
            float(targets.iota_weight) * mean_iota_res,
            float(targets.max_iota_weight) * max_iota_res,
            float(targets.volavgB_weight) * (scalars["volavgB"] - float(targets.volavgB)),
            float(targets.beta_weight) * (scalars["betatotal"] - float(targets.beta_total)),
        ],
        dtype=jnp.float64,
    )

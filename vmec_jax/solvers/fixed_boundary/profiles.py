"""Pure profile and flux helper functions shared by solver entry points."""

from __future__ import annotations

import numpy as np

from ..._compat import jnp
from ...field import TWOPI, chips_from_wout_chipf


def _vmec_force_flux_profiles(*, phipf, chipf, signgs: int, flux_is_internal: bool, iotaf=None, iotas=None):
    phipf = jnp.asarray(phipf)
    chipf = None if chipf is None else jnp.asarray(chipf)
    if flux_is_internal:
        phipf_internal = phipf
        chipf_internal = chipf
    else:
        scale = jnp.asarray(TWOPI, dtype=phipf.dtype) * jnp.asarray(int(signgs), dtype=phipf.dtype)
        phipf_internal = phipf / scale
        chipf_internal = None if chipf is None else (chipf / scale)
    if chipf_internal is not None:
        chips_eff = chips_from_wout_chipf(
            chipf=chipf_internal,
            phipf=phipf_internal,
            iotaf=iotaf,
            iotas=iotas,
            assume_half_if_unknown=True,
        )
    else:
        iota = iotaf if iotaf is not None else iotas
        if iota is None:
            chips_eff = jnp.zeros_like(phipf_internal)
        else:
            chips_eff = jnp.asarray(iota, dtype=phipf_internal.dtype) * phipf_internal
    return phipf_internal, chipf_internal, chips_eff


def _s_half_from_full_mesh_s(s):
    s = jnp.asarray(s)
    if int(s.shape[0]) < 2:
        return s
    return jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)


def _half_mesh_from_full_mesh(x):
    x = jnp.asarray(x)
    if int(x.shape[0]) < 2:
        return x
    return jnp.concatenate([x[:1], 0.5 * (x[1:] + x[:-1])], axis=0)


def _pressure_half_mesh_from_indata(*, indata, s_full):
    from ...profiles import eval_profiles

    s_half = _s_half_from_full_mesh_s(s_full)
    prof = eval_profiles(indata, s_half)
    return jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))


def _mass_half_mesh_from_indata(*, indata, s_full, phips, r00, gamma, lrfp: bool = False, chips=None):
    """Compute VMEC mass profile on half mesh: mass = pmass * (|vnorm|*r00)^gamma."""
    from ...profiles import eval_profiles

    s_half = _s_half_from_full_mesh_s(s_full)
    prof = eval_profiles(indata, s_half)
    pmass = jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))
    vnorm = jnp.asarray(phips)
    if lrfp and (chips is not None):
        vnorm = jnp.asarray(chips)
    mass = pmass * (jnp.abs(vnorm) * jnp.asarray(r00, dtype=pmass.dtype)) ** jnp.asarray(gamma, dtype=pmass.dtype)
    if int(mass.shape[0]) > 0:
        mass = mass.at[0].set(jnp.asarray(0.0, dtype=mass.dtype))
    return mass


def _icurv_full_mesh_from_indata(*, indata, s_full, signgs: int):
    from ...profiles import eval_profiles

    s_full = jnp.asarray(s_full)
    ncurr = int(indata.get_int("NCURR", 0))
    if ncurr != 1:
        return jnp.zeros_like(s_full)

    curtor = float(indata.get_float("CURTOR", 0.0))
    if abs(curtor) <= np.finfo(float).eps:
        return jnp.zeros_like(s_full)

    # VMEC stores icurv on the half mesh (same indexing as phips/chips/iotas),
    # evaluated at s = (i-1.5)*hs for i>=2. Mirror that here.
    s_half = _s_half_from_full_mesh_s(s_full)
    prof = eval_profiles(indata, s_half)
    icurv_raw = jnp.asarray(prof.get("current", jnp.zeros_like(s_half)))
    if int(icurv_raw.shape[0]) != int(s_full.shape[0]):
        icurv_raw = jnp.zeros_like(s_half)

    # VMEC scales by pcurr(1) (edge), not the last half-mesh value.
    pedge_prof = eval_profiles(indata, jnp.asarray([1.0], dtype=s_full.dtype))
    pedge = jnp.asarray(pedge_prof.get("current", jnp.asarray([0.0], dtype=s_full.dtype)))[0]
    valid_pedge = jnp.abs(pedge) > jnp.asarray(abs(np.finfo(float).eps * curtor), dtype=s_full.dtype)

    mu0 = 4e-7 * np.pi
    currv = mu0 * curtor
    denom = jnp.where(valid_pedge, pedge, jnp.asarray(1.0, dtype=s_full.dtype))
    scale = jnp.asarray(float(signgs) * currv / (2.0 * np.pi), dtype=icurv_raw.dtype) / denom
    icurv = jnp.where(valid_pedge, scale * icurv_raw, jnp.zeros_like(s_full))
    if int(icurv.shape[0]) > 0:
        icurv = icurv.at[0].set(0.0)
    return icurv

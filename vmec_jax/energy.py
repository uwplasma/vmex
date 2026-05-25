"""Energy functionals.

This module provides a minimal, differentiable magnetic energy functional,
intended as the first objective for a fixed-boundary solver.

We match VMEC's reported `wb` normalization:

    wb = (1 / (2π)^2) * ∫ (B·B)/2 dV

where the integral is over the *full torus*.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Dict, Tuple


import numpy as np

from ._compat import jax, jnp
from .field import TWOPI, b2_from_bsup, bsup_from_geom, lamscale_from_phips
from .geom import eval_geom
from .grids import angle_steps
from .namelist import InData
from .profiles import eval_profiles


@dataclass(frozen=True)
class FluxProfiles:
    """Simple 1D flux functions."""

    phipf: Any  # (ns,)
    chipf: Any  # (ns,)
    phips: Any  # (ns,)
    signgs: int
    lamscale: Any  # scalar


def _as_float_list(x: Any) -> list[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    return [float(x)]


def _has_nonzero_profile_coeffs(values: Any) -> bool:
    """Return True when a namelist coefficient field has a nonzero entry."""

    try:
        coeffs = np.asarray(_as_float_list(values), dtype=float).reshape(-1)
    except Exception:
        return True
    return bool(coeffs.size > 0 and np.any(coeffs != 0.0))


def _poly_no_const(coeffs_1based, x):
    """Evaluate Σ_{i>=1} a_i x^i, where `coeffs_1based[i-1] == a_i`."""
    # Use np.asarray so coeffs[k] in the Python loop is a plain NumPy scalar, not
    # a JAX dynamic_slice that triggers an eager XLA compilation per iteration.
    a = np.asarray(coeffs_1based)
    x = jnp.asarray(x)
    if a.shape[0] == 0:
        return jnp.zeros_like(x)
    # Σ_{i=1..N} a_i x^i = x * Σ_{k=0..N-1} a_{k+1} x^k
    y = jnp.zeros_like(x, dtype=a.dtype)
    for k in range(len(a) - 1, -1, -1):
        y = y * x + a[k]
    return x * y


def _poly_no_const_deriv(coeffs_1based, x):
    """Derivative of Σ_{i>=1} a_i x^i."""
    # Use np.asarray to avoid eager XLA compilations for each c[k] indexing.
    a = np.asarray(coeffs_1based)
    x = jnp.asarray(x)
    if a.shape[0] == 0:
        return jnp.zeros_like(x)
    # d/dx Σ_{i=1..N} a_i x^i = Σ_{i=1..N} i a_i x^{i-1}
    c = a * np.arange(1, len(a) + 1, dtype=a.dtype)
    y = jnp.zeros_like(x, dtype=a.dtype)
    for k in range(len(c) - 1, -1, -1):
        y = y * x + c[k]
    return y


@functools.lru_cache(maxsize=64)
def _make_torflux_jit(aphi_tuple: tuple, lrfp: bool, indata_id: int):
    """Cache JIT'd torflux/torflux_deriv functions per unique (aphi, lrfp) key.

    ``indata_id`` is the Python id() of the InData object so that different
    indata instances (which may have different iota profiles) produce separate
    cache entries when lrfp=True.  For lrfp=False the iota profile is not
    used, so the id is ignored (any int works).
    """
    aphi_arr = jnp.asarray(aphi_tuple)

    # We cannot close over `indata` directly (not hashable), so the RFP branch
    # retrieves it via a lookup dictionary populated at call time.
    # For lrfp=False we only need aphi.
    if not lrfp:
        def _torflux_deriv_inner(x):
            return _poly_no_const_deriv(aphi_arr, x)

        def _torflux_inner(x):
            x = jnp.asarray(x)
            if x.ndim == 0:
                x = x[None]
            h = jnp.asarray(1e-2, dtype=x.dtype) * x
            grid = jnp.arange(101, dtype=x.dtype)[None, :]
            xi = h[:, None] * grid
            vals = _torflux_deriv_inner(xi)
            trap = jnp.sum(vals, axis=1) - 0.5 * (vals[:, 0] + vals[:, -1])
            return h * trap

        return jax.jit(_torflux_deriv_inner), jax.jit(_torflux_inner)

    # lrfp=True: cannot easily cache because eval_profiles depends on indata.
    # Return None to signal the caller should use the non-cached path.
    return None, None


# Registry: indata_id -> indata, for lrfp torflux_deriv calls.
# This avoids keeping a strong reference inside the lru_cache.
_indata_registry: dict = {}


def _iotaf_from_iotas(iotas, *, lrfp: bool) -> Any:
    """VMEC `add_fluxes` smoothing: build full-mesh `iotaf` from half-mesh `iotas`.

    See `VMEC2000/Sources/General/add_fluxes.f90` (non-RFP and RFP branches).
    """
    iotas = jnp.asarray(iotas)
    ns = int(iotas.shape[0])
    if ns <= 1:
        return iotas
    if ns == 2:
        # Not enough points for VMEC's 3-point axis closure. Fall back to a
        # constant extension from the first half-mesh value.
        return jnp.asarray([iotas[1], iotas[1]], dtype=iotas.dtype)

    out = jnp.zeros((ns,), dtype=iotas.dtype)
    if bool(lrfp):
        # Harmonic-mean variant used in RFP mode.
        eps = jnp.asarray(1e-30, dtype=iotas.dtype)

        def _safe_inv(x):
            return jnp.where(jnp.abs(x) > eps, 1.0 / x, 0.0)

        inv2 = _safe_inv(iotas[1])
        inv3 = _safe_inv(iotas[2])
        denom0 = 1.5 * inv2 - 0.5 * inv3
        out0 = jnp.where(jnp.abs(denom0) > eps, 1.0 / denom0, 0.0)

        invn = _safe_inv(iotas[-1])
        invn1 = _safe_inv(iotas[-2])
        denomN = 1.5 * invn - 0.5 * invn1
        outN = jnp.where(jnp.abs(denomN) > eps, 1.0 / denomN, 0.0)

        inv_a = _safe_inv(iotas[1:-1])
        inv_b = _safe_inv(iotas[2:])
        out_mid = jnp.where(jnp.abs(inv_a + inv_b) > eps, 2.0 / (inv_a + inv_b), 0.0)
    else:
        # Arithmetic-average variant (standard VMEC).
        out0 = 1.5 * iotas[1] - 0.5 * iotas[2]
        outN = 1.5 * iotas[-1] - 0.5 * iotas[-2]
        out_mid = 0.5 * (iotas[1:-1] + iotas[2:])

    out = out.at[0].set(out0)
    out = out.at[1:-1].set(out_mid)
    out = out.at[-1].set(outN)
    return out


def flux_profiles_from_indata(indata: InData, s, *, signgs: int) -> FluxProfiles:
    """Construct simple flux profiles (phipf/chipf) from &INDATA.

    This is a deliberately minimal port:

    - toroidal flux uses ``PHIEDGE`` and polynomial ``APHI`` (default
      ``aphi=[1]``), following ``magnetic_fluxes.f:torflux_deriv`` and
      ``profil1d.f``.
    - poloidal flux derivative follows ``magnetic_fluxes.f:polflux_deriv``,
      i.e. ``piota(tf) * torflux_deriv(s)`` for non-RFP
      (RFP uses ``polflux_deriv = 1``).
    """
    s = jnp.asarray(s)
    ns = int(s.shape[0])

    phiedge = float(indata.get_float("PHIEDGE", 1.0))

    aphi = _as_float_list(indata.get("APHI", []))
    if not aphi:
        aphi = [1.0]
    aphi_arr = jnp.asarray(aphi, dtype=s.dtype)

    lrfp = bool(indata.get_bool("LRFP", False))
    has_iota_profile = _has_nonzero_profile_coeffs(indata.get("AI", []))

    # Common CLI path: APHI is absent/default, so torflux(s)=s and
    # torflux_deriv=1. Avoid compiling tiny JIT helpers and, for current-driven
    # inputs without AI, avoid evaluating the full pressure/current profile just
    # to discover that the iota profile is zero.
    if (not bool(lrfp)) and len(aphi) == 1 and float(aphi[0]) == 1.0:
        torflux_edge = (
            jnp.asarray(signgs, dtype=s.dtype)
            * jnp.asarray(phiedge, dtype=s.dtype)
            / jnp.asarray(TWOPI, dtype=s.dtype)
        )
        phipf = torflux_edge * jnp.ones_like(s)
        if has_iota_profile:
            prof = eval_profiles(indata, s)
            chipf = torflux_edge * prof.get("iota", jnp.zeros_like(s))
        else:
            chipf = jnp.zeros_like(s)
        if ns < 2:
            s_half = s
        else:
            s_half = jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)
        phips = torflux_edge * jnp.ones_like(s_half)
        phips = phips.at[0].set(jnp.zeros_like(phips[0]))
        lamscale = lamscale_from_phips(phips, s)
        return FluxProfiles(phipf=phipf, chipf=chipf, phips=phips, signgs=int(signgs), lamscale=lamscale)

    # Use cached JIT'd functions when possible (non-RFP path).
    aphi_tuple = tuple(float(x) for x in aphi)
    indata_id = id(indata)
    cached_deriv, cached_torflux = _make_torflux_jit(aphi_tuple, lrfp, indata_id)

    if cached_deriv is not None:
        # Non-RFP: use cached JIT'd functions (no new compilation per multigrid level).
        _torflux_deriv = cached_deriv
        _torflux = cached_torflux

        def _polflux_deriv(x):
            if not has_iota_profile:
                return jnp.zeros_like(x)
            tf = _torflux(x)
            tf = jnp.minimum(tf, jnp.asarray(1.0, dtype=tf.dtype))
            prof = eval_profiles(indata, tf)
            iota_tf = prof.get("iota", jnp.zeros_like(tf))
            return iota_tf * _torflux_deriv(x)
    else:
        # RFP path: cannot cache (eval_profiles depends on non-hashable indata).
        def _torflux_deriv(x):
            prof = eval_profiles(indata, x)
            iota_x = prof.get("iota", jnp.zeros_like(x))
            iota_x = jnp.where(iota_x != 0, iota_x, jnp.asarray(float("inf"), dtype=iota_x.dtype))
            return 1.0 / iota_x

        def _torflux(x):
            x = jnp.asarray(x)
            if x.ndim == 0:
                x = x[None]
            h = jnp.asarray(1e-2, dtype=x.dtype) * x
            grid = jnp.arange(101, dtype=x.dtype)[None, :]
            xi = h[:, None] * grid
            vals = _torflux_deriv(xi)
            trap = jnp.sum(vals, axis=1) - 0.5 * (vals[:, 0] + vals[:, -1])
            return h * trap

        def _polflux_deriv(x):
            return jnp.ones_like(x)

    # VMEC: torflux_edge = signgs*phiedge/(2π) normalized by torflux(1).
    torflux_edge = jnp.asarray(signgs, dtype=s.dtype) * jnp.asarray(phiedge, dtype=s.dtype) / jnp.asarray(TWOPI, dtype=s.dtype)
    torflux_1 = _torflux(jnp.asarray(1.0, dtype=s.dtype))
    torflux_edge = jnp.where(torflux_1 != 0, torflux_edge / torflux_1, torflux_edge)

    # Full mesh.
    phipf = torflux_edge * _torflux_deriv(s)
    chipf = torflux_edge * _polflux_deriv(s)

    # Half mesh (VMEC: phips/chips).
    if ns < 2:
        s_half = s
    else:
        s_half = jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)
    phips = torflux_edge * _torflux_deriv(s_half)
    phips = phips.at[0].set(jnp.zeros_like(phips[0]))
    lamscale = lamscale_from_phips(phips, s)
    return FluxProfiles(phipf=phipf, chipf=chipf, phips=phips, signgs=int(signgs), lamscale=lamscale)


def flux_profiles_from_indata_host_default(indata: InData, s, *, signgs: int) -> FluxProfiles | None:
    """Return NumPy flux profiles for the common non-RFP default-APHI path.

    This helper is intentionally narrow: it avoids eager XLA compilation in
    CLI/driver finalization for input-only profiles, while leaving all
    non-default APHI, RFP, and explicit-iota cases on the differentiable JAX
    implementation above.
    """
    aphi = _as_float_list(indata.get("APHI", []))
    if not aphi:
        aphi = [1.0]
    if bool(indata.get_bool("LRFP", False)):
        return None
    if len(aphi) != 1 or float(aphi[0]) != 1.0:
        return None
    if _has_nonzero_profile_coeffs(indata.get("AI", [])):
        return None

    s_np = np.asarray(s, dtype=float)
    if s_np.ndim != 1:
        return None
    torflux_edge = float(signgs) * float(indata.get_float("PHIEDGE", 1.0)) / float(TWOPI)
    phipf = np.full_like(s_np, torflux_edge, dtype=float)
    chipf = np.zeros_like(s_np, dtype=float)
    if int(s_np.shape[0]) < 2:
        s_half = s_np
    else:
        s_half = np.concatenate([s_np[:1], 0.5 * (s_np[1:] + s_np[:-1])], axis=0)
    phips = np.full_like(s_half, torflux_edge, dtype=float)
    if phips.size:
        phips[0] = 0.0
    if phips.size < 2:
        lamscale = np.asarray(1.0, dtype=float)
    else:
        hs = float(s_np[1] - s_np[0])
        lamscale = np.asarray(np.sqrt(hs * np.sum(phips[1:] ** 2)), dtype=float)
    return FluxProfiles(phipf=phipf, chipf=chipf, phips=phips, signgs=int(signgs), lamscale=lamscale)


def integrate_volume_density(density, sqrtg, s, theta, zeta, *, nfp: int, signgs: int):
    """Integrate `density` over the full torus using `sqrtg` and grid spacings."""
    density = jnp.asarray(density)
    sqrtg = jnp.asarray(sqrtg)
    s = jnp.asarray(s)
    theta = jnp.asarray(theta)
    zeta = jnp.asarray(zeta)
    nfp = int(nfp)
    signgs = int(signgs)

    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    dphi = dzeta / int(nfp)

    jac = signgs * sqrtg
    per_period = jnp.sum(density * jac) * ds * dtheta * dphi
    return per_period * nfp


def magnetic_wb_from_state(state, static, indata: InData, *, signgs: int) -> Tuple[Any, Dict[str, Any]]:
    """Compute VMEC-style `wb` and a small diagnostics dict."""
    g = eval_geom(state, static)
    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    bsupu, bsupv = bsup_from_geom(g, phipf=flux.phipf, chipf=flux.chipf, nfp=static.cfg.nfp, signgs=signgs, lamscale=flux.lamscale)
    B2 = b2_from_bsup(g, bsupu, bsupv)
    E = integrate_volume_density(0.5 * B2, g.sqrtg, static.s, static.grid.theta, static.grid.zeta, nfp=static.cfg.nfp, signgs=signgs)
    wb = E / (TWOPI * TWOPI)
    diag = {"energy_total": E, "lamscale": flux.lamscale}
    return wb, diag

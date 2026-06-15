"""WOUT flux, lambda, and current-profile convention helpers."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ._compat import has_jax, jax, jnp
from .wout_diagnostics import lambda_half_mesh_weights


class CurrentProfileMetadata(NamedTuple):
    """VMECPlot2-compatible current/profile metadata persisted in WOUT files."""

    ac: np.ndarray
    ac_aux_s: np.ndarray
    ac_aux_f: np.ndarray
    pcurr_type: str
    piota_type: str


def wout_current_profile_metadata_from_indata(
    indata: Any,
    *,
    ndfmax: int = 101,
    min_ac_size: int = 21,
) -> CurrentProfileMetadata:
    """Return WOUT current-profile metadata from a VMEC input deck.

    VMECPlot2 expects a fixed-size polynomial ``AC`` profile plus auxiliary
    spline arrays even when the input deck does not provide spline data.  This
    helper centralizes those defaults so the WOUT builder does not open-code
    profile metadata normalization.
    """

    pcurr_type = indata.get("PCURR_TYPE", None)
    if pcurr_type is None:
        pcurr_type = "power_series"
    piota_type = indata.get("PIOTA_TYPE", None)
    if piota_type is None:
        piota_type = "power_series"

    ac_raw = indata.get("AC", [])
    if isinstance(ac_raw, (int, float, np.floating)):
        ac_vals = [float(ac_raw)]
    elif isinstance(ac_raw, list):
        ac_vals = [float(v) for v in ac_raw]
    else:
        ac_vals = []

    n_preset = max(int(min_ac_size), len(ac_vals) if ac_vals else 1)
    ac = np.zeros((n_preset,), dtype=float)
    for i, v in enumerate(ac_vals):
        if i >= n_preset:
            break
        ac[i] = v

    aux_size = int(ndfmax)
    return CurrentProfileMetadata(
        ac=ac,
        ac_aux_s=-np.ones((aux_size,), dtype=float),
        ac_aux_f=np.zeros((aux_size,), dtype=float),
        pcurr_type=str(pcurr_type),
        piota_type=str(piota_type),
    )


def lambda_wout_from_full_mesh(
    *,
    lam_full: np.ndarray,
    m_modes: np.ndarray,
    s: np.ndarray,
    phipf_internal: np.ndarray,
    lamscale: float,
) -> np.ndarray:
    """Convert internal full-mesh lambda coefficients to VMEC ``wout`` convention."""

    lam_full = np.asarray(lam_full, dtype=float)
    s_arr = np.asarray(s, dtype=float).reshape(-1)
    ns = int(s_arr.shape[0])
    if lam_full.ndim != 2 or lam_full.shape[0] != ns:
        raise ValueError("Expected lam_full with shape (ns, K)")
    m_modes = np.asarray(m_modes, dtype=int)
    if m_modes.ndim != 1 or m_modes.shape[0] != lam_full.shape[1]:
        raise ValueError("Expected m_modes with shape (K,)")
    phipf_internal = np.asarray(phipf_internal, dtype=float).reshape(-1)
    if phipf_internal.shape != (ns,):
        raise ValueError("Expected phipf_internal with shape (ns,)")

    if lamscale == 0.0:
        lam_ext = np.zeros_like(lam_full)
    else:
        phipf_safe = np.where(phipf_internal == 0.0, 1.0, phipf_internal)
        lam_ext = lam_full * (float(lamscale) / phipf_safe[:, None])

    if ns < 2:
        return np.zeros_like(lam_ext)

    sm_f, sp_f = lambda_half_mesh_weights(s_arr)
    lam_half = lam_ext.copy()
    mask_m_le1 = m_modes <= 1
    if np.any(mask_m_le1):
        lam_half[0, mask_m_le1] = lam_half[1, mask_m_le1]

    even_mask = (m_modes % 2) == 0
    odd_mask = ~even_mask
    for js_idx in range(ns - 1, 0, -1):
        if np.any(even_mask):
            lam_half[js_idx, even_mask] = 0.5 * (lam_half[js_idx, even_mask] + lam_half[js_idx - 1, even_mask])
        if np.any(odd_mask):
            sm_val = sm_f[js_idx + 1]
            sp_val = sp_f[js_idx]
            lam_half[js_idx, odd_mask] = 0.5 * (
                sm_val * lam_half[js_idx, odd_mask] + sp_val * lam_half[js_idx - 1, odd_mask]
            )

    lam_half[0, :] = 0.0
    return lam_half


def lambda_full_from_wout_half_mesh(
    *,
    lam_wout: np.ndarray,
    m_modes: np.ndarray,
    s: np.ndarray,
    phipf_internal: np.ndarray,
    lamscale: float,
) -> np.ndarray:
    """Recover internal full-mesh lambda coefficients from VMEC ``wout`` data."""

    lam_wout = np.asarray(lam_wout, dtype=float)
    s_arr = np.asarray(s, dtype=float).reshape(-1)
    ns = int(s_arr.shape[0])
    if lam_wout.ndim != 2 or lam_wout.shape[0] != ns:
        raise ValueError("Expected lam_wout with shape (ns, K)")
    m_modes = np.asarray(m_modes, dtype=int)
    if m_modes.ndim != 1 or m_modes.shape[0] != lam_wout.shape[1]:
        raise ValueError("Expected m_modes with shape (K,)")
    phipf_internal = np.asarray(phipf_internal, dtype=float).reshape(-1)
    if phipf_internal.shape != (ns,):
        raise ValueError("Expected phipf with shape (ns,)")
    if ns < 2:
        return lam_wout.copy()

    sm_f, sp_f = lambda_half_mesh_weights(s_arr)
    lam_full = np.zeros_like(lam_wout)
    is_m0 = m_modes == 0
    is_m1 = m_modes == 1
    lam_full[0, is_m0] = lam_wout[1, is_m0]
    denom_m1 = sm_f[2] + sp_f[1]
    if denom_m1 != 0.0:
        lam_full[0, is_m1] = 2.0 * lam_wout[1, is_m1] / denom_m1
    lam_full[0, ~(is_m0 | is_m1)] = 0.0

    for mval in range(0, int(np.max(m_modes)) + 1):
        mask = m_modes == mval
        if not np.any(mask):
            continue
        if (mval % 2) == 0:
            for js in range(2, ns + 1):
                lam_full[js - 1, mask] = 2.0 * lam_wout[js - 1, mask] - lam_full[js - 2, mask]
        else:
            for js in range(2, ns + 1):
                denom = sm_f[js]
                if denom == 0.0:
                    lam_full[js - 1, mask] = 0.0
                else:
                    lam_full[js - 1, mask] = (
                        2.0 * lam_wout[js - 1, mask] - sp_f[js - 1] * lam_full[js - 2, mask]
                    ) / denom

    lam_full = lam_full * phipf_internal[:, None]
    if lamscale != 0.0:
        lam_full = lam_full / float(lamscale)
    return lam_full


def chipf_from_chips(chips: np.ndarray) -> np.ndarray:
    """VMEC ``add_fluxes`` half-mesh ``chipf`` from full-mesh ``chips``."""

    is_traced = bool(has_jax()) and (isinstance(chips, jax.core.Tracer) or isinstance(chips, jax.Array))
    if is_traced:
        chips = jnp.asarray(chips, dtype=jnp.float64)
        ns = int(chips.shape[0])
        if ns <= 1:
            return chips
        chipf = jnp.zeros((ns,), dtype=chips.dtype)
        chipf = chipf.at[0].set((1.5 * chips[1] - 0.5 * chips[2]) if ns >= 3 else chips[1])
        if ns > 2:
            chipf = chipf.at[1:-1].set(0.5 * (chips[1:-1] + chips[2:]))
        chipf = chipf.at[-1].set(1.5 * chips[-1] - 0.5 * chips[-2])
        return chipf

    chips = np.asarray(chips, dtype=float)
    ns = int(chips.shape[0])
    if ns <= 1:
        return chips.copy()
    chipf = np.zeros((ns,), dtype=chips.dtype)
    if ns >= 3:
        chipf[0] = 1.5 * chips[1] - 0.5 * chips[2]
    else:
        chipf[0] = chips[1]
    if ns > 2:
        chipf[1:-1] = 0.5 * (chips[1:-1] + chips[2:])
    chipf[-1] = 1.5 * chips[-1] - 0.5 * chips[-2]
    return chipf


def icurv_full_mesh_from_indata(*, indata, s_full: np.ndarray, signgs: int) -> np.ndarray:
    """Return VMEC current profile on the full mesh from an input deck."""

    from .profiles import eval_profiles

    s_full = jnp.asarray(s_full)
    ncurr = int(indata.get_int("NCURR", 0))
    if ncurr != 1:
        return jnp.zeros_like(s_full)

    curtor = float(indata.get_float("CURTOR", 0.0))
    if abs(curtor) <= np.finfo(float).eps:
        return jnp.zeros_like(s_full)

    ns = int(s_full.shape[0])
    if ns < 2:
        s_half = s_full
    else:
        s_half = jnp.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])], axis=0)
    prof = eval_profiles(indata, s_half)
    icurv_raw = jnp.asarray(prof.get("current", jnp.zeros_like(s_half)))
    if int(icurv_raw.shape[0]) != ns:
        icurv_raw = jnp.zeros_like(s_half)

    pedge_prof = eval_profiles(indata, jnp.asarray([1.0], dtype=s_full.dtype))
    pedge = jnp.asarray(pedge_prof.get("current", jnp.asarray([0.0], dtype=s_full.dtype)))[0]
    valid_pedge = jnp.abs(pedge) > jnp.asarray(abs(np.finfo(float).eps * curtor), dtype=s_full.dtype)

    mu0 = 4e-7 * np.pi
    currv = mu0 * curtor
    denom = jnp.where(valid_pedge, pedge, jnp.asarray(1.0, dtype=s_full.dtype))
    scale = jnp.asarray(float(signgs) * currv / (2.0 * np.pi), dtype=icurv_raw.dtype) / denom
    icurv = jnp.where(valid_pedge, scale * icurv_raw, jnp.zeros_like(s_full))
    if ns > 0:
        icurv = jnp.where(jnp.arange(ns) == 0, jnp.asarray(0.0, dtype=icurv.dtype), icurv)
    return icurv


def wout_phi_profile_from_variables(variables, *, ns: int, phipf: np.ndarray) -> np.ndarray:
    """Read ``phi`` or synthesize the VMEC half-mesh toroidal-flux profile."""

    if "phi" in variables:
        return np.asarray(variables["phi"][:])

    from .integrals import cumrect_s_halfmesh

    if ns < 2:
        s = np.zeros((ns,), dtype=float)
    else:
        s = np.linspace(0.0, 1.0, ns, dtype=float)
    return np.asarray(cumrect_s_halfmesh(phipf, s))


__all__ = [
    "CurrentProfileMetadata",
    "chipf_from_chips",
    "icurv_full_mesh_from_indata",
    "lambda_full_from_wout_half_mesh",
    "lambda_wout_from_full_mesh",
    "wout_current_profile_metadata_from_indata",
    "wout_phi_profile_from_variables",
]

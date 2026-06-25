"""Static replay context and trace-shape helpers for free-boundary adjoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from vmec_jax._compat import jnp


def with_jax_nonsingular_replay_tables(
    *,
    basis: dict[str, Any],
    tables: dict[str, Any],
    nv: int,
) -> dict[str, Any]:
    """Add JAX NESTOR replay tables that depend only on static grid data."""

    if "iuv_grid" in tables:
        return tables

    mf = int(basis["mf"])
    nf = int(basis["nf"])
    onp = float(basis["onp"])
    cosv_tab = np.asarray(tables["cosv_tab"], dtype=float)
    sinv_tab = np.asarray(tables["sinv_tab"], dtype=float)
    cosui = np.asarray(tables["cosui"], dtype=float)
    sinui = np.asarray(tables["sinui"], dtype=float)
    nu_fourp = int(cosui.shape[1])
    if nu_fourp <= 0:
        raise ValueError("invalid nonsingular table shape")

    iuv_grid = (np.arange(nu_fourp, dtype=np.int32)[:, None] * int(nv)) + np.arange(int(nv), dtype=np.int32)[
        None, :
    ]
    imirr_full = np.asarray(basis["imirr_full"], dtype=np.int32)
    mf1 = int(mf + 1)
    idx_p_rows: list[int] = []
    idx_m_rows: list[int] = []
    negative_positions: list[int] = []
    flat_pos = 0
    for m in range(mf + 1):
        for n in range(nf + 1):
            idx_p_rows.append(int(m + (n + nf) * mf1))
            if n != 0 and m != 0:
                idx_m_rows.append(int(m + ((-n) + nf) * mf1))
                negative_positions.append(int(flat_pos))
            flat_pos += 1

    enriched = dict(tables)
    enriched.update(
        {
            "iuv_grid": np.asarray(iuv_grid, dtype=np.int32),
            "iref_grid": np.asarray(imirr_full[iuv_grid], dtype=np.int32),
            "cosv_modes": 0.5 * onp * np.asarray(cosv_tab[: nf + 1, :], dtype=float),
            "sinv_modes": 0.5 * onp * np.asarray(sinv_tab[: nf + 1, :], dtype=float),
            "idx_p_flat": np.asarray(idx_p_rows, dtype=np.int32),
            "idx_m_negative": np.asarray(idx_m_rows, dtype=np.int32),
            "negative_positions": np.asarray(negative_positions, dtype=np.int32),
            "sinm_sym": np.asarray(sinui[: mf + 1, :], dtype=float),
            "cosm_sym": -np.asarray(cosui[: mf + 1, :], dtype=float),
            "sinm_asym": np.asarray(cosui[: mf + 1, :], dtype=float),
            "cosm_asym": np.asarray(sinui[: mf + 1, :], dtype=float),
        }
    )
    return enriched


def direct_coil_boundary_replay_context_for_shape(
    static: Any,
    *,
    ntheta: int,
    nzeta: int,
) -> dict[str, Any]:
    """Build shape/static NESTOR replay data for accepted-boundary replay."""

    from vmec_jax.free_boundary import (
        _build_vmec_mode_basis,
        _ensure_vmec_nonsingular_kernel_tables,
        _vmec_boundary_wint,
    )

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    wint = _vmec_boundary_wint(static=static, ntheta=ntheta, nzeta=nzeta)
    basis = _build_vmec_mode_basis(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=int(static.cfg.nfp),
        mf=int(static.cfg.mpol) + 1,
        nf=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        wint=wint,
    )
    nvper = 64 if nzeta == 1 else max(1, int(static.cfg.nfp))
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=nzeta, nvper=nvper)
    tables = with_jax_nonsingular_replay_tables(basis=basis, tables=tables, nv=nzeta)
    return {
        "basis": basis,
        "tables": tables,
        "wint": wint,
        "nvper": nvper,
        "ntheta": ntheta,
        "nzeta": nzeta,
    }


def direct_coil_boundary_replay_context(
    static: Any,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Build static NESTOR replay data for an accepted boundary geometry."""

    R = geometry["R"]
    ntheta, nzeta = (int(v) for v in R.shape)
    return direct_coil_boundary_replay_context_for_shape(
        static,
        ntheta=ntheta,
        nzeta=nzeta,
    )


def direct_coil_trace_boundary_shape(trace: Mapping[str, Any]) -> tuple[int, int] | None:
    """Infer the active NESTOR boundary grid shape from accepted trace data."""

    nestor_trace = trace.get("freeb_nestor_trace")
    if isinstance(nestor_trace, Mapping):
        for key in ("br_axis", "bp_axis", "bz_axis"):
            axis = nestor_trace.get(key)
            if axis is None:
                continue
            shape = tuple(int(value) for value in np.shape(axis))
            if len(shape) == 2:
                return shape
    bsqvac = trace.get("freeb_bsqvac_half")
    if bsqvac is not None:
        shape = tuple(int(value) for value in np.shape(bsqvac))
        if len(shape) == 2:
            return shape
    return None


def direct_coil_trace_vacuum_field_override(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Return accepted vacuum-projection arrays from a production NESTOR trace."""

    nestor_trace = trace.get("freeb_nestor_trace", trace)
    if not isinstance(nestor_trace, Mapping):
        raise ValueError("trace must be a NESTOR trace or contain 'freeb_nestor_trace'")
    required_key_map = {
        "bnormal": ("bnormal",),
        "g_uu": ("g_uu",),
        "g_uv": ("g_uv",),
        "g_vv": ("g_vv",),
    }
    missing = tuple(
        source_keys[0] for source_keys in required_key_map.values() if not any(key in nestor_trace for key in source_keys)
    )
    if missing:
        raise ValueError(f"trace is missing vacuum-field override arrays: {missing}")
    out = {
        target_key: jnp.asarray(next(nestor_trace[source_key] for source_key in source_keys if source_key in nestor_trace))
        for target_key, source_keys in required_key_map.items()
    }
    zero_tangent = jnp.zeros_like(out["bnormal"])
    out["bu"] = (
        jnp.asarray(nestor_trace["bexu_ext"])
        if "bexu_ext" in nestor_trace
        else jnp.asarray(nestor_trace.get("bu", zero_tangent))
    )
    out["bv"] = (
        jnp.asarray(nestor_trace["bexv_ext"])
        if "bexv_ext" in nestor_trace
        else jnp.asarray(nestor_trace.get("bv", zero_tangent))
    )
    return out


__all__ = [
    "direct_coil_boundary_replay_context",
    "direct_coil_boundary_replay_context_for_shape",
    "direct_coil_trace_boundary_shape",
    "direct_coil_trace_vacuum_field_override",
    "with_jax_nonsingular_replay_tables",
]

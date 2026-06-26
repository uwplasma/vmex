"""Debug dump helpers for VMEC WOUT assembly.

These routines are intentionally side-effect-only.  Keeping them out of the
main WOUT synthesis function makes the numerical path easier to audit while
preserving the existing environment-variable diagnostics used for parity work.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "") not in ("", "0")


def _dump_dir() -> Path:
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _tag_suffix() -> str:
    tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
    return f"_{tag}" if tag else ""


def dump_bsub_parity_if_requested(*, s: np.ndarray, bc: Any) -> None:
    """Persist bcovar Bsub parity channels when ``VMEC_JAX_DUMP_BSUB_PARITY`` is set."""

    if not _truthy_env("VMEC_JAX_DUMP_BSUB_PARITY"):
        return
    np.savez(
        _dump_dir() / "bsub_parity_dump.npz",
        s=np.asarray(s, dtype=float),
        bsubu=np.asarray(getattr(bc, "bsubu"), dtype=float),
        bsubv=np.asarray(getattr(bc, "bsubv"), dtype=float),
        bsubu_e=np.asarray(getattr(bc, "bsubu_e"), dtype=float),
        bsubv_e=np.asarray(getattr(bc, "bsubv_e"), dtype=float),
        bsubu_e_scaled=np.asarray(getattr(bc, "bsubu_e_scaled"), dtype=float),
        bsubv_e_scaled=np.asarray(getattr(bc, "bsubv_e_scaled"), dtype=float),
        bsubu_parity_even=np.asarray(getattr(bc, "bsubu_parity_even"), dtype=float),
        bsubu_parity_odd=np.asarray(getattr(bc, "bsubu_parity_odd"), dtype=float),
        bsubv_parity_even=np.asarray(getattr(bc, "bsubv_parity_even"), dtype=float),
        bsubv_parity_odd=np.asarray(getattr(bc, "bsubv_parity_odd"), dtype=float),
    )


def dump_bsubh_if_requested(*, s: np.ndarray, bsupu: np.ndarray, bsupv: np.ndarray, bc: Any) -> None:
    """Persist Bsub/Bsup half-mesh channels when ``VMEC_JAX_DUMP_BSUBH`` is set."""

    if not _truthy_env("VMEC_JAX_DUMP_BSUBH"):
        return
    np.savez(
        _dump_dir() / "bsubh_wout.npz",
        s=np.asarray(s, dtype=float),
        bsupu=np.asarray(bsupu, dtype=float),
        bsupv=np.asarray(bsupv, dtype=float),
        bsubu=np.asarray(getattr(bc, "bsubu"), dtype=float),
        bsubv=np.asarray(getattr(bc, "bsubv"), dtype=float),
    )


def dump_bsub_sources_if_requested(*, bc: Any) -> None:
    """Persist raw and parity Bsub source channels for WOUT debugging."""

    if not _truthy_env("VMEC_JAX_DUMP_BSUB_SOURCES"):
        return
    payload = {
        "bsubu": np.asarray(getattr(bc, "bsubu"), dtype=float),
        "bsubv": np.asarray(getattr(bc, "bsubv"), dtype=float),
    }
    for key in (
        "bsubu_e",
        "bsubv_e",
        "bsubu_e_scaled",
        "bsubv_e_scaled",
        "bsubu_preblend",
        "bsubv_preblend",
        "bsubu_parity_even",
        "bsubu_parity_odd",
        "bsubv_parity_even",
        "bsubv_parity_odd",
    ):
        val = getattr(bc, key, None)
        if val is not None:
            payload[key] = np.asarray(val, dtype=float)
    np.savez(_dump_dir() / f"bsub_sources{_tag_suffix()}.npz", **payload)


def dump_bsub_pre_sym_if_requested(
    *,
    trig: Any,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    bsubs: np.ndarray,
) -> None:
    """Persist full-grid Bsub/Bsup values before VMEC symoutput splitting."""

    if not _truthy_env("VMEC_JAX_DUMP_BSUB_PRE_SYM"):
        return
    path = _dump_dir() / f"bsub_pre_sym_jax{_tag_suffix()}.dat"
    bsubu_dump = np.asarray(bsubu, dtype=float)
    bsubv_dump = np.asarray(bsubv, dtype=float)
    bsupu_dump = np.asarray(bsupu, dtype=float)
    bsupv_dump = np.asarray(bsupv, dtype=float)
    bsubs_dump = np.asarray(bsubs, dtype=float)
    ns_d, ntheta_d, nzeta_d = bsubu_dump.shape
    with path.open("w") as f:
        f.write("# bsub pre-symoutput dump (full grid)\n")
        f.write(f"ns={ns_d}\n")
        f.write(f"ntheta3={ntheta_d}\n")
        f.write(f"nzeta={nzeta_d}\n")
        f.write("columns: js lt lz bsubu bsubv bsupu bsupv bsubs\n")
        for lt in range(ntheta_d):
            for lz in range(nzeta_d):
                for js in range(ns_d):
                    f.write(
                        f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{bsubu_dump[js, lt, lz]:24.16E}"
                        f"{bsubv_dump[js, lt, lz]:24.16E}"
                        f"{bsupu_dump[js, lt, lz]:24.16E}"
                        f"{bsupv_dump[js, lt, lz]:24.16E}"
                        f"{bsubs_dump[js, lt, lz]:24.16E}\n"
                    )


def dump_bsub_parity_inputs_if_requested(
    *,
    bsubu_diag: np.ndarray,
    bsubv_diag: np.ndarray,
    bsubu_even: np.ndarray,
    bsubu_odd: np.ndarray,
    bsubv_even: np.ndarray,
    bsubv_odd: np.ndarray,
    use_bc_parity: bool,
) -> None:
    """Persist Bsub parity-filter inputs when requested."""

    if not _truthy_env("VMEC_JAX_DUMP_BSUB_PARITY_INPUTS"):
        return
    np.savez(
        _dump_dir() / f"bsub_parity_inputs{_tag_suffix()}.npz",
        bsubu_diag=np.asarray(bsubu_diag, dtype=float),
        bsubv_diag=np.asarray(bsubv_diag, dtype=float),
        bsubu_even=np.asarray(bsubu_even, dtype=float),
        bsubu_odd=np.asarray(bsubu_odd, dtype=float),
        bsubv_even=np.asarray(bsubv_even, dtype=float),
        bsubv_odd=np.asarray(bsubv_odd, dtype=float),
        odd_needs_shalf=np.asarray(use_bc_parity, dtype=np.int32),
    )


def dump_wrout_modes_if_requested(
    *,
    ns: int,
    nyq_modes: Any,
    gmnc: np.ndarray,
    gmns: np.ndarray,
    bmnc: np.ndarray,
    bmns: np.ndarray,
    bsubumnc: np.ndarray,
    bsubumns: np.ndarray,
    bsubvmnc: np.ndarray,
    bsubvmns: np.ndarray,
    bsubsmnc: np.ndarray,
    bsubsmns: np.ndarray,
    bsupumnc: np.ndarray,
    bsupumns: np.ndarray,
    bsupvmnc: np.ndarray,
    bsupvmns: np.ndarray,
) -> None:
    """Persist the full WOUT Nyquist coefficient table when requested."""

    if not _truthy_env("VMEC_JAX_DUMP_WROUT_MODES"):
        return
    dump_path = _dump_dir() / "wrout_modes_jax.dat"
    m_modes = np.asarray(nyq_modes.m, dtype=int)
    n_modes = np.asarray(nyq_modes.n, dtype=int)
    gmnc_np = np.asarray(gmnc, dtype=float)
    gmns_np = np.asarray(gmns, dtype=float)
    bmnc_np = np.asarray(bmnc, dtype=float)
    bmns_np = np.asarray(bmns, dtype=float)
    bsubumnc_np = np.asarray(bsubumnc, dtype=float)
    bsubumns_np = np.asarray(bsubumns, dtype=float)
    bsubvmnc_np = np.asarray(bsubvmnc, dtype=float)
    bsubvmns_np = np.asarray(bsubvmns, dtype=float)
    bsubsmnc_np = np.asarray(bsubsmnc, dtype=float)
    bsubsmns_np = np.asarray(bsubsmns, dtype=float)
    bsupumnc_np = np.asarray(bsupumnc, dtype=float)
    bsupumns_np = np.asarray(bsupumns, dtype=float)
    bsupvmnc_np = np.asarray(bsupvmnc, dtype=float)
    bsupvmns_np = np.asarray(bsupvmns, dtype=float)
    with dump_path.open("w") as f:
        f.write("# wrout Fourier-mode dump (vmec_jax)\n")
        f.write(f"ns={ns}\n")
        f.write(f"mnmax_nyq={m_modes.size}\n")
        f.write("cols: js mn m n\n")
        f.write(" gmnc gmns bmnc bmns\n")
        f.write(" bsubumnc bsubumns bsubvmnc bsubvmns\n")
        f.write(" bsubsmnc bsubsmns\n")
        f.write(" bsupumnc bsupumns bsupvmnc bsupvmns\n")
        for js_idx in range(ns):
            for mn_idx in range(m_modes.size):
                f.write(
                    f"{js_idx + 1:6d}{mn_idx + 1:6d}{int(m_modes[mn_idx]):6d}{int(n_modes[mn_idx]):6d}"
                    f"{gmnc_np[js_idx, mn_idx]:24.16E}{gmns_np[js_idx, mn_idx]:24.16E}"
                    f"{bmnc_np[js_idx, mn_idx]:24.16E}{bmns_np[js_idx, mn_idx]:24.16E}"
                    f"{bsubumnc_np[js_idx, mn_idx]:24.16E}{bsubumns_np[js_idx, mn_idx]:24.16E}"
                    f"{bsubvmnc_np[js_idx, mn_idx]:24.16E}{bsubvmns_np[js_idx, mn_idx]:24.16E}"
                    f"{bsubsmnc_np[js_idx, mn_idx]:24.16E}{bsubsmns_np[js_idx, mn_idx]:24.16E}"
                    f"{bsupumnc_np[js_idx, mn_idx]:24.16E}{bsupumns_np[js_idx, mn_idx]:24.16E}"
                    f"{bsupvmnc_np[js_idx, mn_idx]:24.16E}{bsupvmns_np[js_idx, mn_idx]:24.16E}\n"
                )


def print_wout_timing_if_requested(*, timing: dict[str, float], total_start: float | None) -> None:
    """Emit the WOUT timing line if timing instrumentation is active."""

    if total_start is None:
        return
    import time as _time

    timing["total_s"] = _time.perf_counter() - total_start
    try:
        parts = []
        for key in (
            "total_s",
            "trig_tables_s",
            "geom_synthesis_s",
            "forces_bcovar_s",
            "bsubs_half_s",
            "nyquist_coeffs_s",
            "equif_s",
            "beta_s",
            "mercier_s",
            "bsub_filter_s",
            "bsub_coeffs_s",
            "jxbforce_mercier_s",
        ):
            if key in timing:
                parts.append(f"{key}={timing[key]:.3e}")
        print("[vmec_jax wout timing] " + " ".join(parts), flush=True)
    except Exception:
        pass

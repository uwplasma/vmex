"""Diagnostic I/O helpers used by :mod:`vmec_jax.solve`.

This module intentionally preserves the legacy debug-dump truthiness rules used
by ``solve.py``. In particular, several dump flags only treat the empty string
and exact ``"0"`` as disabled.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .... import _solve_runtime


_dump_env_enabled = _solve_runtime._dump_env_enabled


def _normalize_resume_state_mode(resume_state_mode: str | None) -> str:
    """Normalize resume-state mode aliases used by scan and host solves."""
    if resume_state_mode is None:
        resume_state_mode = os.getenv("VMEC_JAX_RESUME_STATE_MODE", "full")
    mode = str(resume_state_mode).strip().lower() or "full"
    aliases = {
        "compact": "minimal",
        "light": "minimal",
        "off": "none",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("full", "minimal", "none"):
        raise ValueError("resume_state_mode must be one of {'full', 'minimal', 'none'}")
    return mode


def _pack_resume_state_record(*, base: dict[str, Any], heavy: dict[str, Any] | None = None, mode: str) -> dict | None:
    """Build the resume-state payload according to the requested detail level."""
    mode = _normalize_resume_state_mode(mode)
    if mode == "none":
        return None
    rec = dict(base)
    if mode == "full" and heavy:
        rec.update(heavy)
    return rec


def _vmec2000_cadence_selected(*, iter_idx: int, max_iter: int, nstep_screen: int) -> bool:
    """Return whether a VMEC2000-style row should be sampled on screen cadence."""
    i = int(iter_idx)
    if i <= 1:
        return True
    if i >= int(max_iter):
        return True
    return (i % max(1, int(nstep_screen))) == 0


def _should_print_vmec2000_row(
    *,
    iter_idx: int,
    max_iter: int,
    nstep_screen: int,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
) -> bool:
    if not (bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table)):
        return False
    return _vmec2000_cadence_selected(iter_idx=iter_idx, max_iter=max_iter, nstep_screen=nstep_screen)


def _format_vmec2000_iter_row(
    *,
    iter_idx: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    delt0r: float,
    r00: float,
    w_mhd: float,
    lasym: bool,
    z00: float | None = None,
) -> str:
    if bool(lasym):
        z_val = float("nan") if z00 is None else float(z00)
        return (
            f"{int(iter_idx):5d}"
            f"{float(fsqr):10.2E}{float(fsqz):10.2E}{float(fsql):10.2E}"
            f"{float(r00):11.3E}{z_val:11.3E}{float(delt0r):10.2E}{float(w_mhd):12.4E}"
        )
    return (
        f"{int(iter_idx):5d}"
        f"{float(fsqr):10.2E}{float(fsqz):10.2E}{float(fsql):10.2E}"
        f"{float(r00):11.3E}{float(delt0r):10.2E}{float(w_mhd):12.4E}"
    )


def _format_residual_physical_status_message(
    *,
    iter_idx: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    include_edge: bool,
) -> str:
    """Format the compact non-VMEC residual status line."""

    return (
        f"[solve_fixed_boundary_residual_iter] iter={int(iter_idx):03d} "
        f"fsqr={float(fsqr):.3e} fsqz={float(fsqz):.3e} "
        f"fsql={float(fsql):.3e} include_edge={bool(include_edge)}"
    )


def _format_residual_converged_message(
    *,
    fsqr: float,
    fsqz: float,
    fsql: float,
    target: float,
) -> str:
    """Format the compact non-VMEC convergence message."""

    return (
        "[solve_fixed_boundary_residual_iter] converged: "
        f"fsqr={float(fsqr):.3e} fsqz={float(fsqz):.3e} "
        f"fsql={float(fsql):.3e} target={float(target):.3e}"
    )


def _format_residual_iteration_update_message(
    *,
    iter_idx: int,
    dt_eff: float,
    update_rms: float,
    fsqr1: float,
    fsqz1: float,
    fsql1: float,
    step_status: str,
) -> str:
    """Format the compact non-VMEC accepted/restarted iteration message."""

    return (
        f"[solve_fixed_boundary_residual_iter] iter={int(iter_idx):03d} "
        f"dt_eff={float(dt_eff):.3e} update_rms={float(update_rms):.3e} "
        f"fsqr1={float(fsqr1):.3e} fsqz1={float(fsqz1):.3e} "
        f"fsql1={float(fsql1):.3e} step_status={step_status}"
    )


def _format_axis_coeff(val: float) -> str:
    text = f"{float(val):.16g}"
    if "e" in text:
        text = text.replace("e", "E")
    return text


def _format_time_control_log_row(
    *, iter_idx: int, fsq: float, fsq0: float, res0: float, res1: float, time_step: float
) -> str:
    return (
        f"iter={int(iter_idx)} fsq={float(fsq):.6e} fsq0={float(fsq0):.6e} "
        f"res0={float(res0):.6e} res1={float(res1):.6e} time_step={float(time_step):.6e}\n"
    )


def _format_time_control_trace_row(
    *,
    stage: str,
    iter2: int,
    iter1: int,
    fsq: float,
    fsq0: float,
    res0: float,
    res1: float,
    time_step: float,
    irst: int,
) -> str:
    return (
        f"{int(iter2):8d} {int(iter1):8d} "
        f"{float(fsq): .16e} {float(fsq0): .16e} "
        f"{float(res0): .16e} {float(res1): .16e} "
        f"{float(time_step): .16e} {int(irst):3d} {stage}\n"
    )


def _format_checkpoint_log_row(*, iter_idx: int, fsq: float, fsq0: float, res0: float, res1: float) -> str:
    return (
        f"iter={int(iter_idx)} fsq={float(fsq):.6e} fsq0={float(fsq0):.6e} "
        f"res0={float(res0):.6e} res1={float(res1):.6e}\n"
    )


def _format_freeb_control_trace_row(
    *,
    iter2: int,
    iter1: int,
    ivac: int,
    ivacskip: int,
    nvacskip: int,
    fsq_rz_prev: float,
    cached: bool,
) -> str:
    return (
        f"{int(iter2):8d} {int(iter1):8d} {int(ivac):8d} {int(ivacskip):8d} "
        f"{int(nvacskip):8d} {float(fsq_rz_prev): .16e} {1 if bool(cached) else 0:2d}\n"
    )


def _format_evolve_trace_row(
    *,
    iter2: int,
    iter1: int,
    ns: int,
    stage: str,
    fsq1: float,
    fsq_prev: float,
    time_step: float,
    dtau: float,
    b1: float,
    fac: float,
    xc_norm: float,
    v_norm: float,
    g_norm: float,
) -> str:
    return (
        f"{int(iter2):8d} {int(iter1):8d} {int(ns):8d} {stage} "
        f"{float(fsq1): .16e} {float(fsq_prev): .16e} "
        f"{float(time_step): .16e} {float(dtau): .16e} "
        f"{float(b1): .16e} {float(fac): .16e} "
        f"{float(xc_norm): .16e} {float(v_norm): .16e} {float(g_norm): .16e}\n"
    )


def _legacy_dump_record_path(*, enable_env: str, filename: str) -> Path | None:
    """Return a dump path for legacy flags that only treat empty/0 as disabled."""
    if not _dump_env_enabled(os.getenv(enable_env, "")):
        return None
    dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "")
    if not dump_dir:
        return None
    return Path(dump_dir) / filename


def _maybe_dump_time_control_record(
    *,
    iter_idx: int,
    fsq: float,
    fsq0: float,
    res0: float,
    res1: float,
    time_step: float,
) -> None:
    path = _legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_TIMECONTROL", filename="time_control.log")
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(
                _format_time_control_log_row(
                    iter_idx=iter_idx,
                    fsq=fsq,
                    fsq0=fsq0,
                    res0=res0,
                    res1=res1,
                    time_step=time_step,
                )
            )
    except Exception:
        return


def _dump_time_control_trace_record(
    *,
    stage: str,
    iter2: int,
    iter1: int,
    fsq: float,
    fsq0: float,
    res0: float,
    res1: float,
    time_step: float,
    irst: int,
) -> None:
    path = _legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_TIMECONTROL", filename="time_control_trace.log")
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(
                _format_time_control_trace_row(
                    stage=stage,
                    iter2=iter2,
                    iter1=iter1,
                    fsq=fsq,
                    fsq0=fsq0,
                    res0=res0,
                    res1=res1,
                    time_step=time_step,
                    irst=irst,
                )
            )
    except Exception:
        return


def _maybe_dump_checkpoint_record(*, iter_idx: int, fsq: float, fsq0: float, res0: float, res1: float) -> None:
    path = _legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_CHECKPOINT", filename="checkpoint.log")
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(_format_checkpoint_log_row(iter_idx=iter_idx, fsq=fsq, fsq0=fsq0, res0=res0, res1=res1))
    except Exception:
        return


def _dump_freeb_control_trace_record(
    *,
    iter2: int,
    iter1: int,
    ivac: int,
    ivacskip: int,
    nvacskip: int,
    fsq_rz_prev: float,
    cached: bool,
) -> None:
    path = _legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_FREEB_CONTROL", filename="freeb_control_trace.log")
    if path is None:
        return
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(
                _format_freeb_control_trace_row(
                    iter2=iter2,
                    iter1=iter1,
                    ivac=ivac,
                    ivacskip=ivacskip,
                    nvacskip=nvacskip,
                    fsq_rz_prev=fsq_rz_prev,
                    cached=cached,
                )
            )
    except Exception:
        return


def _dump_freeb_axis_trace_record(*, iter2: int, axis_r: np.ndarray, axis_z: np.ndarray) -> None:
    path = _legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_FREEB_AXIS", filename=f"freeb_axis_iter{int(iter2)}.npz")
    if path is None:
        return
    try:
        np.savez_compressed(
            path,
            iter2=int(iter2),
            axis_r=np.asarray(axis_r, dtype=float).reshape(-1),
            axis_z=np.asarray(axis_z, dtype=float).reshape(-1),
        )
    except Exception:
        return


def _maybe_dump_evolve_trace_record(
    *,
    static,
    iter2: int,
    iter1: int,
    stage: str,
    fsq1_val: float,
    fsq_prev_val: float,
    time_step_val: float,
    dtau_val: float,
    b1_val: float,
    fac_val: float,
    state_val: Any,
    vRcc_val,
    vRss_val,
    vZsc_val,
    vZcs_val,
    vLsc_val,
    vLcs_val,
    vRsc_val=None,
    vRcs_val=None,
    vZcc_val=None,
    vZss_val=None,
    vLcc_val=None,
    vLss_val=None,
    frcc_val=None,
    frss_val=None,
    fzsc_val=None,
    fzcs_val=None,
    flsc_val=None,
    flcs_val=None,
    frsc_val=None,
    frcs_val=None,
    fzcc_val=None,
    fzss_val=None,
    flcc_val=None,
    flss_val=None,
) -> None:
    path = _legacy_dump_record_path(enable_env="VMEC_JAX_DUMP_EVOLVE", filename="evolve_trace.log")
    if path is None:
        return
    try:
        from ....diagnostics import vmec_internal_mn_from_state, vmec_xc_from_mn_blocks

        blocks = vmec_internal_mn_from_state(
            state_val,
            static,
            apply_basis_norm=False,
            apply_m1_constraint=False,
        )
        xc_kwargs = {
            "rcc": blocks["rcc"],
            "rss": blocks["rss"],
            "zsc": blocks["zsc"],
            "zcs": blocks["zcs"],
            "lsc": blocks["lsc"],
            "lcs": blocks["lcs"],
        }
        if "rsc" in blocks:
            xc_kwargs.update(
                {
                    "rsc": blocks.get("rsc"),
                    "rcs": blocks.get("rcs"),
                    "zcc": blocks.get("zcc"),
                    "zss": blocks.get("zss"),
                    "lcc": blocks.get("lcc"),
                    "lss": blocks.get("lss"),
                }
            )
        xc_vec = np.asarray(vmec_xc_from_mn_blocks(cfg=static.cfg, **xc_kwargs), dtype=float)
        v_kwargs = {
            "rcc": np.asarray(vRcc_val, dtype=float),
            "rss": np.asarray(vRss_val, dtype=float),
            "zsc": np.asarray(vZsc_val, dtype=float),
            "zcs": np.asarray(vZcs_val, dtype=float),
            "lsc": np.asarray(vLsc_val, dtype=float),
            "lcs": np.asarray(vLcs_val, dtype=float),
        }
        if vRsc_val is not None:
            v_kwargs["rsc"] = np.asarray(vRsc_val, dtype=float)
        if vRcs_val is not None:
            v_kwargs["rcs"] = np.asarray(vRcs_val, dtype=float)
        if vZcc_val is not None:
            v_kwargs["zcc"] = np.asarray(vZcc_val, dtype=float)
        if vZss_val is not None:
            v_kwargs["zss"] = np.asarray(vZss_val, dtype=float)
        if vLcc_val is not None:
            v_kwargs["lcc"] = np.asarray(vLcc_val, dtype=float)
        if vLss_val is not None:
            v_kwargs["lss"] = np.asarray(vLss_val, dtype=float)
        v_vec = np.asarray(vmec_xc_from_mn_blocks(cfg=static.cfg, **v_kwargs), dtype=float)
        gnorm = 0.0
        if frcc_val is not None:
            g_kwargs = {
                "rcc": np.asarray(frcc_val, dtype=float),
                "rss": np.asarray(frss_val, dtype=float),
                "zsc": np.asarray(fzsc_val, dtype=float),
                "zcs": np.asarray(fzcs_val, dtype=float),
                "lsc": np.asarray(flsc_val, dtype=float),
                "lcs": np.asarray(flcs_val, dtype=float),
            }
            if frsc_val is not None:
                g_kwargs["rsc"] = np.asarray(frsc_val, dtype=float)
            if frcs_val is not None:
                g_kwargs["rcs"] = np.asarray(frcs_val, dtype=float)
            if fzcc_val is not None:
                g_kwargs["zcc"] = np.asarray(fzcc_val, dtype=float)
            if fzss_val is not None:
                g_kwargs["zss"] = np.asarray(fzss_val, dtype=float)
            if flcc_val is not None:
                g_kwargs["lcc"] = np.asarray(flcc_val, dtype=float)
            if flss_val is not None:
                g_kwargs["lss"] = np.asarray(flss_val, dtype=float)
            g_vec = np.asarray(vmec_xc_from_mn_blocks(cfg=static.cfg, **g_kwargs), dtype=float)
            gnorm = float(np.linalg.norm(g_vec))
        with path.open("a", encoding="utf-8") as f:
            f.write(
                _format_evolve_trace_row(
                    iter2=iter2,
                    iter1=iter1,
                    ns=int(static.cfg.ns),
                    stage=stage,
                    fsq1=fsq1_val,
                    fsq_prev=fsq_prev_val,
                    time_step=time_step_val,
                    dtau=dtau_val,
                    b1=b1_val,
                    fac=fac_val,
                    xc_norm=float(np.linalg.norm(xc_vec)),
                    v_norm=float(np.linalg.norm(v_vec)),
                    g_norm=gnorm,
                )
            )
    except Exception:
        return


def _finite_float_or_zero(value: Any) -> float:
    """Return a Python float, replacing NaN/Inf with zero for scalar diagnostics."""
    out = float(np.asarray(value))
    return out if np.isfinite(out) else 0.0


def _normalize_adjoint_trace_mode(adjoint_trace_mode: str) -> str:
    mode = str(adjoint_trace_mode).strip().lower() or "full"
    aliases = {
        "branch_local": "branch",
        "branch-local": "branch",
        "lean": "branch",
        "compact": "branch",
    }
    mode = aliases.get(mode, mode)
    if mode not in ("full", "dynamic", "branch"):
        raise ValueError("adjoint_trace_mode must be one of {'full', 'dynamic', 'branch'}")
    return mode


def _materialize_adjoint_trace_array(value, *, mode: str):
    """Return dynamic trace values as-is, but snapshot full trace arrays on host."""
    mode = _normalize_adjoint_trace_mode(mode)
    if mode == "dynamic":
        return value
    return np.asarray(value)


def _legacy_single_dump_iter_selected(*, dump_iter: str, iter_idx: int) -> bool:
    """Match legacy single-iteration dump filtering, including invalid-as-all."""
    if not dump_iter:
        return True
    try:
        return int(dump_iter) == int(iter_idx)
    except Exception:
        return True


def _pshalf_from_s_np(s_arr) -> np.ndarray:
    s_arr = np.asarray(s_arr, dtype=float)
    if s_arr.size < 2:
        return np.sqrt(np.maximum(s_arr, 0.0))
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    p = np.concatenate([sh[:1], sh], axis=0)
    return np.sqrt(np.maximum(p, 0.0))


def _maybe_dump_jacobian_terms_record(*, k, s, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_JACOBIAN_TERMS", "").strip()
    if not env or env in ("0", "false", "no", "False"):
        return
    dump_iter = os.getenv("VMEC_JAX_DUMP_ITER", "").strip()
    if not _legacy_single_dump_iter_selected(dump_iter=dump_iter, iter_idx=iter_idx):
        return
    outdir = os.getenv("VMEC_JAX_DUMP_DIR", "").strip() or "."
    outpath = Path(outdir).expanduser().resolve()
    outpath.mkdir(parents=True, exist_ok=True)
    fname = outpath / f"jacobian_terms_iter{int(iter_idx)}.dat"

    pr1_even = np.asarray(getattr(k, "pr1_even"))
    pr1_odd = np.asarray(getattr(k, "pr1_odd"))
    pz1_even = np.asarray(getattr(k, "pz1_even"))
    pz1_odd = np.asarray(getattr(k, "pz1_odd"))
    pru_even = np.asarray(getattr(k, "pru_even"))
    pru_odd = np.asarray(getattr(k, "pru_odd"))
    pzu_even = np.asarray(getattr(k, "pzu_even"))
    pzu_odd = np.asarray(getattr(k, "pzu_odd"))
    prv_even = np.asarray(getattr(k, "prv_even"))
    prv_odd = np.asarray(getattr(k, "prv_odd"))
    pzv_even = np.asarray(getattr(k, "pzv_even"))
    pzv_odd = np.asarray(getattr(k, "pzv_odd"))

    ns, ntheta3, nzeta = pr1_even.shape
    pshalf = _pshalf_from_s_np(np.asarray(s))
    if pshalf.shape[0] != ns:
        pshalf = np.resize(pshalf, (ns,))
    hs = float(np.asarray(s[1] - s[0])) if ns > 1 else 1.0
    ohs = 1.0 / hs if hs != 0.0 else 0.0
    dphids = 0.25

    with fname.open("w", encoding="utf-8") as f:
        f.write("# jacobian term dump\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta3}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write("columns: js lt lz pshalf\n")
        f.write(" pru_e pru_o pru_e_m1 pru_o_m1\n")
        f.write(" pz1_e pz1_o pz1_e_m1 pz1_o_m1\n")
        f.write(" pzu_e pzu_o pzu_e_m1 pzu_o_m1\n")
        f.write(" pr1_e pr1_o pr1_e_m1 pr1_o_m1\n")
        f.write(" prv_e prv_o prv_e_m1 prv_o_m1\n")
        f.write(" pzv_e pzv_o pzv_e_m1 pzv_o_m1\n")
        f.write(" ru12 pzs pzu12 prs pr12 ptau\n")
        f.write(" rv12 zv12\n")
        for lt in range(ntheta3):
            for lz in range(nzeta):
                for j in range(1, ns):
                    jm1 = j - 1
                    psh = pshalf[j]
                    psh_safe = psh if psh != 0.0 else 1.0
                    pru_e = pru_even[j, lt, lz]
                    pru_o = pru_odd[j, lt, lz]
                    pru_e_m1 = pru_even[jm1, lt, lz]
                    pru_o_m1 = pru_odd[jm1, lt, lz]
                    pz1_e = pz1_even[j, lt, lz]
                    pz1_o = pz1_odd[j, lt, lz]
                    pz1_e_m1 = pz1_even[jm1, lt, lz]
                    pz1_o_m1 = pz1_odd[jm1, lt, lz]
                    pzu_e = pzu_even[j, lt, lz]
                    pzu_o = pzu_odd[j, lt, lz]
                    pzu_e_m1 = pzu_even[jm1, lt, lz]
                    pzu_o_m1 = pzu_odd[jm1, lt, lz]
                    pr1_e = pr1_even[j, lt, lz]
                    pr1_o = pr1_odd[j, lt, lz]
                    pr1_e_m1 = pr1_even[jm1, lt, lz]
                    pr1_o_m1 = pr1_odd[jm1, lt, lz]
                    prv_e = prv_even[j, lt, lz]
                    prv_o = prv_odd[j, lt, lz]
                    prv_e_m1 = prv_even[jm1, lt, lz]
                    prv_o_m1 = prv_odd[jm1, lt, lz]
                    pzv_e = pzv_even[j, lt, lz]
                    pzv_o = pzv_odd[j, lt, lz]
                    pzv_e_m1 = pzv_even[jm1, lt, lz]
                    pzv_o_m1 = pzv_odd[jm1, lt, lz]

                    ru12 = 0.5 * (pru_e + pru_e_m1 + psh * (pru_o + pru_o_m1))
                    pzs = ohs * ((pz1_e - pz1_e_m1) + psh * (pz1_o - pz1_o_m1))
                    ptau = ru12 * pzs + dphids * (
                        pru_o * pz1_o + pru_o_m1 * pz1_o_m1 + (pru_e * pz1_o + pru_e_m1 * pz1_o_m1) / psh_safe
                    )
                    pzu12 = 0.5 * (pzu_e + pzu_e_m1 + psh * (pzu_o + pzu_o_m1))
                    prs = ohs * ((pr1_e - pr1_e_m1) + psh * (pr1_o - pr1_o_m1))
                    pr12 = 0.5 * (pr1_e + pr1_e_m1 + psh * (pr1_o + pr1_o_m1))
                    ptau = (
                        ptau
                        - prs * pzu12
                        - dphids
                        * (pzu_o * pr1_o + pzu_o_m1 * pr1_o_m1 + (pzu_e * pr1_o + pzu_e_m1 * pr1_o_m1) / psh_safe)
                    )
                    rv12 = 0.5 * (prv_e + prv_e_m1 + psh * (prv_o + prv_o_m1))
                    zv12 = 0.5 * (pzv_e + pzv_e_m1 + psh * (pzv_o + pzv_o_m1))

                    f.write(
                        f"{j + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{psh:24.16E}"
                        f"{pru_e:24.16E}{pru_o:24.16E}{pru_e_m1:24.16E}{pru_o_m1:24.16E}"
                        f"{pz1_e:24.16E}{pz1_o:24.16E}{pz1_e_m1:24.16E}{pz1_o_m1:24.16E}"
                        f"{pzu_e:24.16E}{pzu_o:24.16E}{pzu_e_m1:24.16E}{pzu_o_m1:24.16E}"
                        f"{pr1_e:24.16E}{pr1_o:24.16E}{pr1_e_m1:24.16E}{pr1_o_m1:24.16E}"
                        f"{prv_e:24.16E}{prv_o:24.16E}{prv_e_m1:24.16E}{prv_o_m1:24.16E}"
                        f"{pzv_e:24.16E}{pzv_o:24.16E}{pzv_e_m1:24.16E}{pzv_o_m1:24.16E}"
                        f"{ru12:24.16E}{pzs:24.16E}{pzu12:24.16E}{prs:24.16E}{pr12:24.16E}{ptau:24.16E}"
                        f"{rv12:24.16E}{zv12:24.16E}\n"
                    )

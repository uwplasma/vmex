"""Run a small convergence grid for the toroidal hybrid boundary."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

import vmec_jax as vj
from vmec_jax.namelist import write_indata
from vmec_jax.toroidal_hybrid import (
    evaluate_toroidal_hybrid_indata_boundary,
    sample_toroidal_stellarator_mirror_hybrid_boundary,
    toroidal_stellarator_mirror_hybrid_indata,
    toroidal_stellarator_mirror_hybrid_metrics,
)
from vmec_jax.vmec2000_exec import flatten_threed1, run_xvmec2000, threed1_fsq_total
from vmec_jax.wout import read_wout


def _parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_mode_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        left, sep, right = item.partition(":")
        if not sep:
            raise ValueError("mode pairs must use MPOL:NTOR, for example 5:4,6:5")
        pairs.append((int(left), int(right)))
    return pairs


_SHAPE_CASE_PRESETS = {
    "default": {},
    "sharp": {
        "side_minor_modulation": 0.16,
        "side_elongation": 0.35,
        "side_power": 2.0,
        "corner_amplitude": 0.025,
        "corner_power": 2.0,
    },
}

_VMEC_JAX_INITIALIZATION_POLICY = "vmec_jax_default_input_boundary"
_VMEC2000_INITIALIZATION_POLICY = "vmec2000_default_input_boundary"
_VMEC_JAX_AXIS_RAW_POLICY = "raw_input_axis_or_zero"
_VMEC_JAX_AXIS_INFERRED_POLICY = "boundary_inferred_missing_axis"


def _parse_shape_cases(text: str) -> list[str]:
    names = [item.strip() for item in str(text).split(",") if item.strip()]
    unknown = [name for name in names if name not in _SHAPE_CASE_PRESETS]
    if unknown:
        choices = ", ".join(sorted(_SHAPE_CASE_PRESETS))
        raise ValueError(f"unknown shape case(s) {unknown}; choices are {choices}")
    return names


def _vmec_jax_axis_initialization_policy(solver_mode: str) -> str:
    """Return the VMEC/JAX axis branch used by this runner's fixed-boundary call."""
    mode = str(solver_mode).strip().lower()
    infer_axis = mode != "parity"
    enable_env = os.getenv("VMEC_JAX_ENABLE_AXIS_INFER", "").strip().lower()
    disable_env = os.getenv("VMEC_JAX_DISABLE_AXIS_INFER", "").strip().lower()
    if enable_env in ("1", "true", "yes", "on"):
        infer_axis = True
    if disable_env in ("1", "true", "yes", "on"):
        infer_axis = False
    return _VMEC_JAX_AXIS_INFERRED_POLICY if infer_axis else _VMEC_JAX_AXIS_RAW_POLICY


def _base_sample_kwargs(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "major_radius": float(args.major_radius),
        "minor_radius": float(args.minor_radius),
        "axis_oval": float(args.axis_oval),
        "side_minor_modulation": float(args.side_minor_modulation),
        "side_elongation": float(args.side_elongation),
        "side_power": float(args.side_power),
        "corner_amplitude": float(args.corner_amplitude),
        "corner_helicity": int(args.corner_helicity),
        "corner_power": float(args.corner_power),
    }


def _shape_case_kwargs(args: argparse.Namespace) -> list[tuple[str, dict[str, float | int]]]:
    base = _base_sample_kwargs(args)
    names = _parse_shape_cases(args.shape_cases)
    if not names:
        return [("custom", base)]
    cases = []
    for name in names:
        kwargs = dict(base)
        kwargs.update(_SHAPE_CASE_PRESETS[name])
        cases.append((name, kwargs))
    return cases


def _import_matplotlib():
    try:
        mpl_cache = Path(tempfile.gettempdir()) / "vmec_jax_mplconfig"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit("This example requires matplotlib for plots.") from exc
    return plt


_CSV_COLUMNS = (
    "case",
    "shape_case",
    "ns",
    "mpol",
    "ntor",
    "nstep",
    "rbc_count",
    "zbs_count",
    "max_boundary_fit_error",
    "major_radius",
    "minor_radius",
    "axis_oval",
    "side_minor_modulation",
    "side_elongation",
    "side_power",
    "corner_amplitude",
    "corner_helicity",
    "corner_power",
    "initialization_policy",
    "vmec_jax_axis_initialization_policy",
    "ran_solve",
    "solver_mode",
    "use_scan",
    "full_solver_diagnostics",
    "diagnostic_light_history",
    "diagnostic_resume_state_mode",
    "diagnostic_stage_modes",
    "diagnostic_stage_niter",
    "diagnostic_stage_offsets",
    "diagnostic_step_history_size",
    "diagnostic_step_status_counts",
    "diagnostic_restart_reason_counts",
    "diagnostic_bcovar_updates",
    "diagnostic_initial_bcovar_update",
    "diagnostic_final_dt_eff",
    "diagnostic_max_update_rms",
    "diagnostic_final_update_rms",
    "requested_ftol",
    "fsq_total_target",
    "seconds",
    "n_iter",
    "direct_initial_residual_requested",
    "direct_initial_residual_source",
    "direct_initial_axis_initialization_policy",
    "direct_initial_fsq",
    "direct_initial_fsqr",
    "direct_initial_fsqz",
    "direct_initial_fsql",
    "direct_initial_fsq_ratio_vmec2000",
    "direct_initial_fsqr_ratio_vmec2000",
    "direct_initial_fsqz_ratio_vmec2000",
    "direct_initial_fsql_ratio_vmec2000",
    "direct_initial_error",
    "initial_residual_source",
    "initial_fsq",
    "best_fsq",
    "best_iter",
    "fsq_reduction",
    "final_fsq",
    "initial_fsqr",
    "initial_fsqz",
    "initial_fsql",
    "final_fsqr",
    "final_fsqz",
    "final_fsql",
    "best_fsqr",
    "best_fsqz",
    "best_fsql",
    "converged",
    "converged_strict",
    "converged_by_total_fsq",
    "aspect",
    "mean_iota",
    "magnetic_well",
    "ran_vmec2000",
    "vmec2000_initialization_policy",
    "vmec2000_returncode",
    "vmec2000_runtime_s",
    "vmec2000_n_rows",
    "vmec2000_initial_residual_source",
    "vmec2000_initial_fsq",
    "vmec2000_best_fsq",
    "vmec2000_best_iter",
    "vmec2000_fsq_reduction",
    "vmec2000_final_fsq",
    "vmec2000_initial_fsqr",
    "vmec2000_initial_fsqz",
    "vmec2000_initial_fsql",
    "vmec2000_final_fsqr",
    "vmec2000_final_fsqz",
    "vmec2000_final_fsql",
    "vmec2000_aspect",
    "vmec2000_mean_iota",
    "initial_fsq_ratio_vmec2000",
    "initial_fsqr_ratio_vmec2000",
    "initial_fsqz_ratio_vmec2000",
    "initial_fsql_ratio_vmec2000",
    "input",
    "wout",
    "vmec2000_wout",
    "vmec2000_threed1",
    "vmec2000_error",
)


def _csv_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_rows_csv(rows: list[dict[str, object]], *, outdir: Path) -> str:
    path = outdir / "toroidal_stellarator_mirror_hybrid_convergence.csv"
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(_CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_cell(row.get(name)) for name in _CSV_COLUMNS})
    return str(path)


def _summarize_fsq_history(
    values: np.ndarray, *, iterations: np.ndarray | None = None
) -> dict[str, float | int | None]:
    history = np.asarray(values, dtype=float).reshape(-1)
    if history.size == 0:
        return {
            "initial_fsq": None,
            "best_fsq": None,
            "best_iter": None,
            "fsq_reduction": None,
            "final_fsq": None,
        }
    if iterations is None:
        iter_values = np.arange(history.size, dtype=int)
    else:
        iter_values = np.asarray(iterations, dtype=int).reshape(-1)
        if iter_values.size != history.size:
            iter_values = np.arange(history.size, dtype=int)
    out: dict[str, float | int | None] = {
        "initial_fsq": float(history[0]),
        "best_fsq": None,
        "best_iter": None,
        "fsq_reduction": None,
        "final_fsq": float(history[-1]),
    }
    finite = np.isfinite(history)
    if np.any(finite):
        finite_values = np.where(finite, history, np.inf)
        best_idx = int(np.argmin(finite_values))
        best_fsq = float(finite_values[best_idx])
        out["best_fsq"] = best_fsq
        out["best_iter"] = int(iter_values[best_idx])
        out["fsq_reduction"] = float(history[0]) / best_fsq if best_fsq > 0.0 else None
    return out


def _safe_ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    num = float(numerator)
    den = float(denominator)
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return None
    return num / den


def _attach_initial_residual_comparison(row: dict[str, object]) -> None:
    """Attach VMEC/JAX-to-VMEC2000 first-row residual ratios when available."""
    row["initial_fsq_ratio_vmec2000"] = _safe_ratio(row.get("initial_fsq"), row.get("vmec2000_initial_fsq"))
    row["direct_initial_fsq_ratio_vmec2000"] = _safe_ratio(
        row.get("direct_initial_fsq"),
        row.get("vmec2000_initial_fsq"),
    )
    for name in ("fsqr", "fsqz", "fsql"):
        row[f"initial_{name}_ratio_vmec2000"] = _safe_ratio(
            row.get(f"initial_{name}"),
            row.get(f"vmec2000_initial_{name}"),
        )
        row[f"direct_initial_{name}_ratio_vmec2000"] = _safe_ratio(
            row.get(f"direct_initial_{name}"),
            row.get(f"vmec2000_initial_{name}"),
        )


def _row_history_iterations(row: dict[str, object], history_size: int) -> np.ndarray:
    """Return stored iteration labels for a row, or a one-based fallback."""
    labels = np.asarray(row.get("iter_history", []), dtype=int).reshape(-1)
    if labels.size != int(history_size):
        return np.arange(1, int(history_size) + 1, dtype=int)
    return labels


def _diag_float_list(diag: dict[str, object], key: str) -> list[float]:
    values = np.asarray(diag.get(key, []), dtype=float).reshape(-1)
    return [float(value) for value in values]


def _diag_int_list(diag: dict[str, object], key: str) -> list[int]:
    values = np.asarray(diag.get(key, []), dtype=int).reshape(-1)
    return [int(value) for value in values]


def _diag_str_list(diag: dict[str, object], key: str) -> list[str]:
    values = np.asarray(diag.get(key, []), dtype=object).reshape(-1)
    return [str(value) for value in values]


def _counts_json(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _solver_diagnostic_fields(diag: dict[str, object], *, fallback_size: int) -> dict[str, object]:
    """Return compact JSON-safe solver histories for trajectory audits."""
    step_status = _diag_str_list(diag, "step_status_history")
    restart_reason = _diag_str_list(diag, "restart_reason_history")
    pre_restart_reason = _diag_str_list(diag, "pre_restart_reason_history")
    dt_eff = _diag_float_list(diag, "dt_eff_history")
    update_rms = _diag_float_list(diag, "update_rms_history")
    w_curr = _diag_float_list(diag, "w_curr_history")
    w_try = _diag_float_list(diag, "w_try_history")
    w_try_ratio = _diag_float_list(diag, "w_try_ratio_history")
    terminal_size = max(
        len(step_status),
        len(restart_reason),
        len(pre_restart_reason),
        len(dt_eff),
        len(update_rms),
        len(w_curr),
        len(w_try),
        len(w_try_ratio),
        0,
    )
    iter2 = _diag_int_list(diag, "iter2_history")
    if terminal_size and len(iter2) != terminal_size:
        iter2 = [int(value) for value in range(1, terminal_size + 1)]
    elif not terminal_size and int(fallback_size) > 0:
        iter2 = []
    bcovar = _diag_int_list(diag, "bcovar_update_history")
    stage_modes = _diag_str_list(diag, "multigrid_stage_modes")
    stage_niter = _diag_int_list(diag, "multigrid_niter_stages")
    stage_offsets = _diag_int_list(diag, "multigrid_stage_offsets")
    return {
        "diagnostic_light_history": None if "light_history" not in diag else bool(diag.get("light_history")),
        "diagnostic_resume_state_mode": None
        if diag.get("resume_state_mode") is None
        else str(diag.get("resume_state_mode")),
        "diagnostic_stage_modes": stage_modes,
        "diagnostic_stage_niter": stage_niter,
        "diagnostic_stage_offsets": stage_offsets,
        "diagnostic_step_history_size": int(terminal_size),
        "diagnostic_step_iter_history": iter2,
        "diagnostic_step_status_history": step_status,
        "diagnostic_restart_reason_history": restart_reason,
        "diagnostic_pre_restart_reason_history": pre_restart_reason,
        "diagnostic_dt_eff_history": dt_eff,
        "diagnostic_update_rms_history": update_rms,
        "diagnostic_w_curr_history": w_curr,
        "diagnostic_w_try_history": w_try,
        "diagnostic_w_try_ratio_history": w_try_ratio,
        "diagnostic_bcovar_update_history": bcovar,
        "diagnostic_step_status_counts": _counts_json(step_status),
        "diagnostic_restart_reason_counts": _counts_json(restart_reason),
        "diagnostic_bcovar_updates": int(sum(1 for value in bcovar if int(value) != 0)),
        "diagnostic_initial_bcovar_update": None if not bcovar else bool(int(bcovar[0])),
        "diagnostic_final_dt_eff": None if not dt_eff else float(dt_eff[-1]),
        "diagnostic_max_update_rms": None if not update_rms else float(np.nanmax(update_rms)),
        "diagnostic_final_update_rms": None if not update_rms else float(update_rms[-1]),
    }


def _compute_direct_initial_residual(
    input_path: Path,
    *,
    solver_mode: str,
    use_scan: bool | None,
) -> dict[str, object]:
    """Evaluate force residual scalars on the VMEC/JAX initial state."""
    mode = str(solver_mode).strip().lower()
    run = vj.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode=str(solver_mode),
        use_scan=use_scan,
        max_iter=1,
        use_initial_guess=True,
        cli_fixed_boundary_mode=True,
        verbose=False,
    )
    wout = vj.wout_from_fixed_boundary_run(
        run,
        include_fsq=True,
        fast_bcovar=False if mode == "parity" else True,
    )
    fsqr = float(wout.fsqr)
    fsqz = float(wout.fsqz)
    fsql = float(wout.fsql)
    return {
        "direct_initial_residual_source": "vmec_jax_initial_guess_residual_scalars",
        "direct_initial_axis_initialization_policy": _vmec_jax_axis_initialization_policy(str(solver_mode)),
        "direct_initial_fsq": fsqr + fsqz + fsql,
        "direct_initial_fsqr": fsqr,
        "direct_initial_fsqz": fsqz,
        "direct_initial_fsql": fsql,
        "direct_initial_error": None,
    }


def _write_summary_plot(rows: list[dict[str, object]], *, outdir: Path) -> str:
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    labels = [f"ns={row['ns']}, {row['mpol']}:{row['ntor']}" for row in rows]
    solved_rows = any(row.get("best_fsq") is not None or row.get("final_fsq") is not None for row in rows)
    if solved_rows:
        values = [
            float(row["best_fsq"] if row.get("best_fsq") is not None else row["final_fsq"])
            if row.get("final_fsq") is not None
            else float(row["max_boundary_fit_error"])
            for row in rows
        ]
        ylabel = "best fsq"
    else:
        values = [float(row["max_boundary_fit_error"]) for row in rows]
        ylabel = "max boundary fit error"
    fig, ax = plt.subplots(1, 1, figsize=(max(7.0, 0.6 * len(rows)), 4.2), constrained_layout=True)
    ax.semilogy(np.arange(len(rows)), np.maximum(values, 1.0e-300), "o-", lw=1.5)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title("Toroidal hybrid convergence grid")
    ax.grid(True, which="both", alpha=0.25)
    path = outdir / "toroidal_hybrid_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_fsq_history_plot(rows: list[dict[str, object]], *, outdir: Path) -> str | None:
    history_rows = [row for row in rows if row.get("fsq_history") or row.get("vmec2000_fsq_history")]
    if not history_rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2), constrained_layout=True)
    for row in history_rows:
        direct_initial = row.get("direct_initial_fsq")
        if direct_initial is not None:
            ax.semilogy(
                [0],
                [max(float(direct_initial), 1.0e-300)],
                "*",
                ms=8,
                label=f"{row['case']} VMEC/JAX direct initial",
            )
        history = np.asarray(row.get("fsq_history", []), dtype=float).reshape(-1)
        if history.size:
            iters = _row_history_iterations(row, int(history.size))
            ax.semilogy(
                iters,
                np.maximum(history, 1.0e-300),
                "o-",
                lw=1.3,
                ms=3,
                label=f"{row['case']} VMEC/JAX",
            )
        vmec2000_history = np.asarray(row.get("vmec2000_fsq_history", []), dtype=float).reshape(-1)
        if vmec2000_history.size:
            vmec2000_iters = np.asarray(row.get("vmec2000_iter_history", []), dtype=int).reshape(-1)
            if vmec2000_iters.size != vmec2000_history.size:
                vmec2000_iters = np.arange(vmec2000_history.size, dtype=int)
            ax.semilogy(
                vmec2000_iters,
                np.maximum(vmec2000_history, 1.0e-300),
                "s--",
                lw=1.2,
                ms=3,
                label=f"{row['case']} VMEC2000",
            )
    ax.set_xlabel("iteration (0 is VMEC/JAX direct initial)")
    ax.set_ylabel("fsq")
    ax.set_title("Toroidal hybrid residual history")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    path = outdir / "toroidal_hybrid_fsq_history.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_step_diagnostic_plot(rows: list[dict[str, object]], *, outdir: Path) -> str | None:
    step_rows = [
        row
        for row in rows
        if row.get("diagnostic_dt_eff_history")
        or row.get("diagnostic_update_rms_history")
        or row.get("diagnostic_w_try_ratio_history")
    ]
    if not step_rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True, constrained_layout=True)
    for row in step_rows:
        label = str(row["case"])
        iters = np.asarray(row.get("diagnostic_step_iter_history", []), dtype=int).reshape(-1)
        for ax, key, ylabel in (
            (axes[0], "diagnostic_dt_eff_history", "dt effective"),
            (axes[1], "diagnostic_update_rms_history", "update RMS"),
            (axes[2], "diagnostic_w_try_ratio_history", "trial/current fsq"),
        ):
            values = np.asarray(row.get(key, []), dtype=float).reshape(-1)
            if values.size == 0:
                continue
            x = iters if iters.size == values.size else np.arange(1, values.size + 1, dtype=int)
            ax.semilogy(x, np.maximum(values, 1.0e-300), ".-", lw=1.1, ms=3, label=label)
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.25)
    axes[-1].set_xlabel("iteration")
    for ax in axes:
        if ax.lines:
            ax.legend(loc="best", fontsize=8)
    axes[0].set_title("Toroidal hybrid solver step diagnostics")
    path = outdir / "toroidal_hybrid_step_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_profile_plots(rows: list[dict[str, object]], *, outdir: Path) -> str | None:
    wout_rows = [row for row in rows if row.get("wout")]
    if not wout_rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    plotted = False
    for row in wout_rows:
        try:
            wout = read_wout(str(row["wout"]))
        except Exception:
            continue
        ns = int(getattr(wout, "ns", 0))
        if ns <= 0:
            continue
        s = np.linspace(0.0, 1.0, ns)
        label = str(row["case"])
        iotas = np.asarray(getattr(wout, "iotas", np.zeros((0,))), dtype=float).reshape(-1)
        if iotas.size == ns:
            axes[0].plot(s, iotas, ".-", lw=1.2, ms=3, label=label)
            plotted = True
        dwell = np.asarray(getattr(wout, "Dwell", np.zeros((0,))), dtype=float).reshape(-1)
        if dwell.size == ns:
            axes[1].plot(s, dwell, ".-", lw=1.2, ms=3, label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("iota")
    axes[0].set_title("iota profile")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("DWell")
    axes[1].set_title("Mercier well term")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    path = outdir / "toroidal_hybrid_profiles.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _write_parity_component_plot(rows: list[dict[str, object]], *, outdir: Path) -> str | None:
    parity_rows = [
        row
        for row in rows
        if row.get("final_fsqr") is not None
        and row.get("vmec2000_final_fsqr") is not None
        and row.get("vmec2000_final_fsqz") is not None
        and row.get("vmec2000_final_fsql") is not None
    ]
    if not parity_rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1,
        len(parity_rows),
        figsize=(max(5.0, 4.4 * len(parity_rows)), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    components = ("fsqr", "fsqz", "fsql")
    x = np.arange(len(components), dtype=float)
    width = 0.36
    for ax, row in zip(axes.ravel(), parity_rows, strict=False):
        jax_values = np.asarray([float(row[f"final_{name}"]) for name in components], dtype=float)
        vmec_values = np.asarray([float(row[f"vmec2000_final_{name}"]) for name in components], dtype=float)
        ax.bar(x - width / 2.0, np.maximum(jax_values, 1.0e-300), width=width, label="VMEC/JAX")
        ax.bar(x + width / 2.0, np.maximum(vmec_values, 1.0e-300), width=width, label="VMEC2000")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.set_ylabel("final residual component")
        ax.set_title(str(row["case"]))
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    path = outdir / "toroidal_hybrid_parity_components.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _row_case_name(*, ns: int, mpol: int, ntor: int, shape_case: str = "custom") -> str:
    base = f"ns{int(ns):03d}_mpol{int(mpol):02d}_ntor{int(ntor):02d}"
    return base if shape_case == "custom" else f"{shape_case}_{base}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default="results/toroidal_stellarator_mirror_hybrid_convergence")
    parser.add_argument("--ns-array", type=str, default="9,15")
    parser.add_argument("--mode-pairs", type=str, default="5:4")
    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--niter", type=int, default=80)
    parser.add_argument(
        "--nstep",
        type=int,
        default=25,
        help="VMEC print cadence written into NSTEP; use 1 for full VMEC2000 threed1 trajectories.",
    )
    parser.add_argument("--ftol", type=float, default=1.0e-9)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--major-radius", type=float, default=1.15)
    parser.add_argument("--minor-radius", type=float, default=0.18)
    parser.add_argument("--axis-oval", type=float, default=0.10)
    parser.add_argument("--side-minor-modulation", type=float, default=0.10)
    parser.add_argument("--side-elongation", type=float, default=0.28)
    parser.add_argument("--side-power", type=float, default=1.0)
    parser.add_argument("--corner-amplitude", type=float, default=0.035)
    parser.add_argument("--corner-helicity", type=int, default=1)
    parser.add_argument("--corner-power", type=float, default=1.0)
    parser.add_argument(
        "--shape-cases",
        type=str,
        default="",
        help="Comma-separated preset shape cases to scan; choices: default,sharp. Empty uses the explicit CLI shape.",
    )
    parser.add_argument("--ntheta-fit", type=int, default=64)
    parser.add_argument("--nzeta-fit", type=int, default=64)
    parser.add_argument("--run-solve", action="store_true")
    parser.add_argument("--solver-mode", choices=("default", "parity", "accelerated"), default="accelerated")
    parser.add_argument("--use-scan", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--direct-initial-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When solving, also evaluate VMEC/JAX residual scalars on the pre-iteration initial state.",
    )
    parser.add_argument(
        "--full-solver-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store full per-step VMEC/JAX solver histories instead of the quiet light-history path.",
    )
    parser.add_argument("--run-vmec2000", action="store_true")
    parser.add_argument("--vmec2000-exec", type=str, default="")
    parser.add_argument("--vmec2000-timeout-s", type=float, default=120.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns_values = _parse_ints(args.ns_array)
    mode_pairs = _parse_mode_pairs(args.mode_pairs)
    vmec2000_exec = Path(args.vmec2000_exec).expanduser() if str(args.vmec2000_exec).strip() else None
    shape_cases = _shape_case_kwargs(args)
    rows: list[dict[str, object]] = []

    for shape_case, sample_kwargs in shape_cases:
        samples = sample_toroidal_stellarator_mirror_hybrid_boundary(
            ntheta=int(args.ntheta_fit),
            nzeta=int(args.nzeta_fit),
            **sample_kwargs,
        )
        reference_metrics = toroidal_stellarator_mirror_hybrid_metrics(samples)
        for ns in ns_values:
            for mpol, ntor in mode_pairs:
                case = _row_case_name(ns=ns, mpol=mpol, ntor=ntor, shape_case=shape_case)
                case_dir = outdir / case
                case_dir.mkdir(parents=True, exist_ok=True)
                indata = toroidal_stellarator_mirror_hybrid_indata(
                    nfp=int(args.nfp),
                    mpol=int(mpol),
                    ntor=int(ntor),
                    ntheta_fit=int(args.ntheta_fit),
                    nzeta_fit=int(args.nzeta_fit),
                    ns_array=int(ns),
                    niter_array=int(args.niter),
                    ftol_array=float(args.ftol),
                    **sample_kwargs,
                )
                indata.scalars["NSTEP"] = int(args.nstep)
                input_path = case_dir / "input.toroidal_stellarator_mirror_hybrid"
                write_indata(input_path, indata)
                fitted = evaluate_toroidal_hybrid_indata_boundary(
                    indata,
                    ntheta=int(args.ntheta_fit),
                    nzeta=int(args.nzeta_fit),
                )
                max_fit_error = max(
                    float(np.max(np.abs(fitted.R - samples.R))),
                    float(np.max(np.abs(fitted.Z - samples.Z))),
                )
                row: dict[str, object] = {
                    "case": case,
                    "shape_case": shape_case,
                    "ns": int(ns),
                    "mpol": int(mpol),
                    "ntor": int(ntor),
                    "nstep": int(args.nstep),
                    "input": str(input_path),
                    "rbc_count": len(indata.indexed.get("RBC", {})),
                    "zbs_count": len(indata.indexed.get("ZBS", {})),
                    "max_boundary_fit_error": max_fit_error,
                    "major_radius": sample_kwargs["major_radius"],
                    "minor_radius": sample_kwargs["minor_radius"],
                    "axis_oval": sample_kwargs["axis_oval"],
                    "side_minor_modulation": sample_kwargs["side_minor_modulation"],
                    "side_elongation": sample_kwargs["side_elongation"],
                    "side_power": sample_kwargs["side_power"],
                    "corner_amplitude": sample_kwargs["corner_amplitude"],
                    "corner_helicity": sample_kwargs["corner_helicity"],
                    "corner_power": sample_kwargs["corner_power"],
                    "initialization_policy": _VMEC_JAX_INITIALIZATION_POLICY,
                    "vmec_jax_axis_initialization_policy": _vmec_jax_axis_initialization_policy(args.solver_mode),
                    "min_R": reference_metrics["min_R"],
                    "stellsym_R_error": reference_metrics["stellsym_R_error"],
                    "stellsym_Z_error": reference_metrics["stellsym_Z_error"],
                    "ran_solve": bool(args.run_solve),
                    "solver_mode": str(args.solver_mode),
                    "use_scan": None if args.use_scan is None else bool(args.use_scan),
                    "full_solver_diagnostics": bool(args.full_solver_diagnostics),
                    "diagnostic_light_history": None,
                    "diagnostic_resume_state_mode": None,
                    "diagnostic_stage_modes": [],
                    "diagnostic_stage_niter": [],
                    "diagnostic_stage_offsets": [],
                    "diagnostic_step_history_size": 0,
                    "diagnostic_step_iter_history": [],
                    "diagnostic_step_status_history": [],
                    "diagnostic_restart_reason_history": [],
                    "diagnostic_pre_restart_reason_history": [],
                    "diagnostic_dt_eff_history": [],
                    "diagnostic_update_rms_history": [],
                    "diagnostic_w_curr_history": [],
                    "diagnostic_w_try_history": [],
                    "diagnostic_w_try_ratio_history": [],
                    "diagnostic_bcovar_update_history": [],
                    "diagnostic_step_status_counts": {},
                    "diagnostic_restart_reason_counts": {},
                    "diagnostic_bcovar_updates": 0,
                    "diagnostic_initial_bcovar_update": None,
                    "diagnostic_final_dt_eff": None,
                    "diagnostic_max_update_rms": None,
                    "diagnostic_final_update_rms": None,
                    "requested_ftol": float(args.ftol),
                    "fsq_total_target": None,
                    "seconds": None,
                    "direct_initial_residual_requested": bool(args.direct_initial_residual),
                    "direct_initial_residual_source": None,
                    "direct_initial_axis_initialization_policy": None,
                    "direct_initial_fsq": None,
                    "direct_initial_fsqr": None,
                    "direct_initial_fsqz": None,
                    "direct_initial_fsql": None,
                    "direct_initial_fsq_ratio_vmec2000": None,
                    "direct_initial_fsqr_ratio_vmec2000": None,
                    "direct_initial_fsqz_ratio_vmec2000": None,
                    "direct_initial_fsql_ratio_vmec2000": None,
                    "direct_initial_error": None,
                    "initial_residual_source": None,
                    "initial_fsq": None,
                    "best_fsq": None,
                    "best_iter": None,
                    "fsq_reduction": None,
                    "final_fsq": None,
                    "initial_fsqr": None,
                    "initial_fsqz": None,
                    "initial_fsql": None,
                    "final_fsqr": None,
                    "final_fsqz": None,
                    "final_fsql": None,
                    "best_fsqr": None,
                    "best_fsqz": None,
                    "best_fsql": None,
                    "converged": None,
                    "converged_strict": None,
                    "converged_by_total_fsq": None,
                    "n_iter": None,
                    "aspect": None,
                    "mean_iota": None,
                    "magnetic_well": None,
                    "fsq_history": [],
                    "iter_history": [],
                    "fsqr_history": [],
                    "fsqz_history": [],
                    "fsql_history": [],
                    "wout": None,
                    "ran_vmec2000": bool(args.run_vmec2000),
                    "vmec2000_initialization_policy": _VMEC2000_INITIALIZATION_POLICY,
                    "vmec2000_returncode": None,
                    "vmec2000_runtime_s": None,
                    "vmec2000_n_rows": None,
                    "vmec2000_initial_residual_source": None,
                    "vmec2000_initial_fsq": None,
                    "vmec2000_best_fsq": None,
                    "vmec2000_best_iter": None,
                    "vmec2000_fsq_reduction": None,
                    "vmec2000_final_fsq": None,
                    "vmec2000_initial_fsqr": None,
                    "vmec2000_initial_fsqz": None,
                    "vmec2000_initial_fsql": None,
                    "vmec2000_final_fsqr": None,
                    "vmec2000_final_fsqz": None,
                    "vmec2000_final_fsql": None,
                    "vmec2000_aspect": None,
                    "vmec2000_mean_iota": None,
                    "vmec2000_iter_history": [],
                    "vmec2000_fsq_history": [],
                    "vmec2000_fsqr_history": [],
                    "vmec2000_fsqz_history": [],
                    "vmec2000_fsql_history": [],
                    "initial_fsq_ratio_vmec2000": None,
                    "initial_fsqr_ratio_vmec2000": None,
                    "initial_fsqz_ratio_vmec2000": None,
                    "initial_fsql_ratio_vmec2000": None,
                    "vmec2000_threed1": None,
                    "vmec2000_wout": None,
                    "vmec2000_error": None,
                }
                if bool(args.run_solve) and bool(args.direct_initial_residual):
                    try:
                        row.update(
                            _compute_direct_initial_residual(
                                input_path,
                                solver_mode=str(args.solver_mode),
                                use_scan=args.use_scan,
                            )
                        )
                    except Exception as exc:
                        row["direct_initial_error"] = str(exc)
                if bool(args.run_solve):
                    t0 = perf_counter()
                    run = vj.run_fixed_boundary(
                        input_path,
                        solver="vmec2000_iter",
                        solver_mode=str(args.solver_mode),
                        use_scan=args.use_scan,
                        max_iter=int(args.max_iter),
                        light_history=False if bool(args.full_solver_diagnostics) else None,
                        cli_fixed_boundary_mode=True,
                        verbose=False,
                    )
                    row["seconds"] = float(perf_counter() - t0)
                    diag = dict(run.result.diagnostics) if run.result is not None else {}
                    row["converged"] = bool(diag.get("converged", False))
                    row["converged_strict"] = bool(diag.get("converged_strict", False))
                    row["converged_by_total_fsq"] = bool(diag.get("converged_by_total_fsq", False))
                    if diag.get("requested_ftol") is not None:
                        row["requested_ftol"] = float(diag["requested_ftol"])
                    if diag.get("fsq_total_target") is not None:
                        row["fsq_total_target"] = float(diag["fsq_total_target"])
                    row["n_iter"] = int(getattr(run.result, "n_iter", -1)) if run.result is not None else None
                    best_component_index = None
                    fsq_history = np.zeros((0,), dtype=float)
                    if run.result is not None and getattr(run.result, "w_history", None) is not None:
                        fsq_history = np.asarray(run.result.w_history, dtype=float).reshape(-1)
                        row["fsq_history"] = [float(value) for value in fsq_history]
                        iter_history = np.asarray(diag.get("iter2_history", []), dtype=int).reshape(-1)
                        if iter_history.size == fsq_history.size:
                            row["iter_history"] = [int(value) for value in iter_history]
                        else:
                            row["iter_history"] = [int(value) for value in range(1, fsq_history.size + 1)]
                        row.update(_summarize_fsq_history(fsq_history))
                        if fsq_history.size and np.any(np.isfinite(fsq_history)):
                            best_component_index = int(
                                np.argmin(np.where(np.isfinite(fsq_history), fsq_history, np.inf))
                            )
                        row["initial_residual_source"] = "vmec_jax_solve_history_first_stored_row"
                        for source, history_key, initial_key, final_key, best_key in (
                            ("fsqr2_history", "fsqr_history", "initial_fsqr", "final_fsqr", "best_fsqr"),
                            ("fsqz2_history", "fsqz_history", "initial_fsqz", "final_fsqz", "best_fsqz"),
                            ("fsql2_history", "fsql_history", "initial_fsql", "final_fsql", "best_fsql"),
                        ):
                            component = np.asarray(getattr(run.result, source, []), dtype=float).reshape(-1)
                            row[history_key] = [float(value) for value in component]
                            if component.size:
                                row[initial_key] = float(component[0])
                                row[final_key] = float(component[-1])
                                if best_component_index is not None and 0 <= int(best_component_index) < component.size:
                                    row[best_key] = float(component[int(best_component_index)])
                    row.update(_solver_diagnostic_fields(diag, fallback_size=int(fsq_history.size)))
                    try:
                        row["aspect"] = float(
                            vj.equilibrium_aspect_ratio_from_state(state=run.state, static=run.static)
                        )
                    except Exception:
                        row["aspect"] = None
                    try:
                        _chips, iotas, _iotaf = vj.equilibrium_iota_profiles_from_state(
                            state=run.state,
                            static=run.static,
                            indata=run.indata,
                            signgs=int(run.signgs),
                        )
                        row["mean_iota"] = float(np.nanmean(np.asarray(iotas, dtype=float)))
                    except Exception:
                        row["mean_iota"] = None
                    try:
                        row["magnetic_well"] = float(
                            vj.magnetic_well_from_state(
                                state=run.state,
                                static=run.static,
                                indata=run.indata,
                                signgs=int(run.signgs),
                            )
                        )
                    except Exception:
                        row["magnetic_well"] = None
                    wout_path = case_dir / "wout_toroidal_stellarator_mirror_hybrid.nc"
                    vj.write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
                    row["wout"] = str(wout_path)
                if bool(args.run_vmec2000):
                    try:
                        vmec2000 = run_xvmec2000(
                            input_path,
                            exec_path=vmec2000_exec,
                            workdir=case_dir / "vmec2000",
                            timeout_s=float(args.vmec2000_timeout_s),
                            keep_workdir=True,
                        )
                        row["vmec2000_returncode"] = int(vmec2000.returncode)
                        row["vmec2000_runtime_s"] = float(vmec2000.runtime_s)
                        row["vmec2000_threed1"] = (
                            str(vmec2000.threed1_path) if vmec2000.threed1_path is not None else None
                        )
                        vmec2000_wouts = sorted(vmec2000.workdir.glob("wout*.nc"))
                        row["vmec2000_wout"] = str(vmec2000_wouts[0]) if vmec2000_wouts else None
                        if vmec2000_wouts:
                            try:
                                vmec2000_wout = read_wout(vmec2000_wouts[0])
                                row["vmec2000_final_fsqr"] = float(vmec2000_wout.fsqr)
                                row["vmec2000_final_fsqz"] = float(vmec2000_wout.fsqz)
                                row["vmec2000_final_fsql"] = float(vmec2000_wout.fsql)
                                row["vmec2000_aspect"] = float(vmec2000_wout.aspect)
                                row["vmec2000_mean_iota"] = float(
                                    np.nanmean(np.asarray(vmec2000_wout.iotas, dtype=float))
                                )
                            except Exception:
                                pass
                        vmec2000_rows = flatten_threed1(vmec2000.stages)
                        row["vmec2000_n_rows"] = len(vmec2000_rows)
                        if vmec2000_rows:
                            vmec2000_iters = np.asarray([item.it for item in vmec2000_rows], dtype=int)
                            vmec2000_fsq = threed1_fsq_total(vmec2000_rows)
                            row["vmec2000_iter_history"] = [int(value) for value in vmec2000_iters]
                            row["vmec2000_fsq_history"] = [float(value) for value in vmec2000_fsq]
                            vmec2000_summary = _summarize_fsq_history(vmec2000_fsq, iterations=vmec2000_iters)
                            row["vmec2000_initial_fsq"] = vmec2000_summary["initial_fsq"]
                            row["vmec2000_best_fsq"] = vmec2000_summary["best_fsq"]
                            row["vmec2000_best_iter"] = vmec2000_summary["best_iter"]
                            row["vmec2000_fsq_reduction"] = vmec2000_summary["fsq_reduction"]
                            row["vmec2000_final_fsq"] = vmec2000_summary["final_fsq"]
                            row["vmec2000_fsqr_history"] = [float(item.fsqr) for item in vmec2000_rows]
                            row["vmec2000_fsqz_history"] = [float(item.fsqz) for item in vmec2000_rows]
                            row["vmec2000_fsql_history"] = [float(item.fsql) for item in vmec2000_rows]
                            row["vmec2000_initial_residual_source"] = "vmec2000_threed1_first_row"
                            row["vmec2000_initial_fsqr"] = float(vmec2000_rows[0].fsqr)
                            row["vmec2000_initial_fsqz"] = float(vmec2000_rows[0].fsqz)
                            row["vmec2000_initial_fsql"] = float(vmec2000_rows[0].fsql)
                    except Exception as exc:
                        row["vmec2000_error"] = str(exc)
                _attach_initial_residual_comparison(row)
                rows.append(row)

    summary = {
        "shape_cases": [{"name": name, "sample_parameters": kwargs} for name, kwargs in shape_cases],
        "rows": rows,
        "csv": _write_rows_csv(rows, outdir=outdir),
        "figures": {},
    }
    if not bool(args.no_plots):
        summary["figures"]["convergence"] = _write_summary_plot(rows, outdir=outdir / "figures")
        fsq_history_plot = _write_fsq_history_plot(rows, outdir=outdir / "figures")
        if fsq_history_plot is not None:
            summary["figures"]["fsq_history"] = fsq_history_plot
        step_plot = _write_step_diagnostic_plot(rows, outdir=outdir / "figures")
        if step_plot is not None:
            summary["figures"]["step_diagnostics"] = step_plot
        profile_plot = _write_profile_plots(rows, outdir=outdir / "figures")
        if profile_plot is not None:
            summary["figures"]["profiles"] = profile_plot
        parity_plot = _write_parity_component_plot(rows, outdir=outdir / "figures")
        if parity_plot is not None:
            summary["figures"]["parity_components"] = parity_plot

    summary_path = outdir / "toroidal_stellarator_mirror_hybrid_convergence.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":  # pragma: no cover
    main()

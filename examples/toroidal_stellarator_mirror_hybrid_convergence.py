"""Run a small convergence grid for the toroidal hybrid boundary."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter
import tempfile

import numpy as np

import vmec_jax as vj
from vmec_jax.namelist import write_indata
from vmec_jax.toroidal_hybrid import (
    evaluate_toroidal_hybrid_indata_boundary,
    sample_toroidal_stellarator_mirror_hybrid_boundary,
    toroidal_stellarator_mirror_hybrid_indata,
    toroidal_stellarator_mirror_hybrid_metrics,
)
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
    "ns",
    "mpol",
    "ntor",
    "rbc_count",
    "zbs_count",
    "max_boundary_fit_error",
    "ran_solve",
    "seconds",
    "n_iter",
    "initial_fsq",
    "best_fsq",
    "best_iter",
    "fsq_reduction",
    "final_fsq",
    "converged",
    "aspect",
    "mean_iota",
    "magnetic_well",
    "input",
    "wout",
)


def _write_rows_csv(rows: list[dict[str, object]], *, outdir: Path) -> str:
    path = outdir / "toroidal_stellarator_mirror_hybrid_convergence.csv"
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(_CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in _CSV_COLUMNS})
    return str(path)


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
    history_rows = [row for row in rows if row.get("fsq_history")]
    if not history_rows:
        return None
    plt = _import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.2), constrained_layout=True)
    for row in history_rows:
        history = np.asarray(row["fsq_history"], dtype=float).reshape(-1)
        if history.size == 0:
            continue
        ax.semilogy(np.arange(history.size), np.maximum(history, 1.0e-300), "o-", lw=1.3, ms=3, label=str(row["case"]))
    ax.set_xlabel("VMEC/JAX iteration")
    ax.set_ylabel("fsq")
    ax.set_title("Toroidal hybrid residual history")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    path = outdir / "toroidal_hybrid_fsq_history.png"
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


def _row_case_name(*, ns: int, mpol: int, ntor: int) -> str:
    return f"ns{int(ns):03d}_mpol{int(mpol):02d}_ntor{int(ntor):02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=str, default="results/toroidal_stellarator_mirror_hybrid_convergence")
    parser.add_argument("--ns-array", type=str, default="9,15")
    parser.add_argument("--mode-pairs", type=str, default="5:4")
    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--niter", type=int, default=80)
    parser.add_argument("--ftol", type=float, default=1.0e-9)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--ntheta-fit", type=int, default=64)
    parser.add_argument("--nzeta-fit", type=int, default=64)
    parser.add_argument("--run-solve", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns_values = _parse_ints(args.ns_array)
    mode_pairs = _parse_mode_pairs(args.mode_pairs)
    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(
        ntheta=int(args.ntheta_fit),
        nzeta=int(args.nzeta_fit),
    )
    reference_metrics = toroidal_stellarator_mirror_hybrid_metrics(samples)
    rows: list[dict[str, object]] = []

    for ns in ns_values:
        for mpol, ntor in mode_pairs:
            case = _row_case_name(ns=ns, mpol=mpol, ntor=ntor)
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
            )
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
                "ns": int(ns),
                "mpol": int(mpol),
                "ntor": int(ntor),
                "input": str(input_path),
                "rbc_count": len(indata.indexed.get("RBC", {})),
                "zbs_count": len(indata.indexed.get("ZBS", {})),
                "max_boundary_fit_error": max_fit_error,
                "min_R": reference_metrics["min_R"],
                "stellsym_R_error": reference_metrics["stellsym_R_error"],
                "stellsym_Z_error": reference_metrics["stellsym_Z_error"],
                "ran_solve": bool(args.run_solve),
                "seconds": None,
                "initial_fsq": None,
                "best_fsq": None,
                "best_iter": None,
                "fsq_reduction": None,
                "final_fsq": None,
                "converged": None,
                "n_iter": None,
                "aspect": None,
                "mean_iota": None,
                "magnetic_well": None,
                "fsq_history": [],
                "wout": None,
            }
            if bool(args.run_solve):
                t0 = perf_counter()
                run = vj.run_fixed_boundary(
                    input_path,
                    solver="vmec2000_iter",
                    solver_mode="accelerated",
                    max_iter=int(args.max_iter),
                    cli_fixed_boundary_mode=True,
                    verbose=False,
                )
                row["seconds"] = float(perf_counter() - t0)
                diag = dict(run.result.diagnostics) if run.result is not None else {}
                row["converged"] = bool(diag.get("converged", False))
                row["n_iter"] = int(getattr(run.result, "n_iter", -1)) if run.result is not None else None
                if run.result is not None and getattr(run.result, "w_history", None) is not None:
                    fsq_history = np.asarray(run.result.w_history, dtype=float).reshape(-1)
                    row["fsq_history"] = [float(value) for value in fsq_history]
                    if fsq_history.size:
                        finite = np.isfinite(fsq_history)
                        row["initial_fsq"] = float(fsq_history[0])
                        row["final_fsq"] = float(fsq_history[-1])
                        if np.any(finite):
                            finite_values = np.where(finite, fsq_history, np.inf)
                            best_iter = int(np.argmin(finite_values))
                            best_fsq = float(finite_values[best_iter])
                            row["best_iter"] = best_iter
                            row["best_fsq"] = best_fsq
                            row["fsq_reduction"] = float(row["initial_fsq"]) / best_fsq if best_fsq > 0.0 else None
                try:
                    row["aspect"] = float(vj.equilibrium_aspect_ratio_from_state(state=run.state, static=run.static))
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
            rows.append(row)

    summary = {
        "rows": rows,
        "csv": _write_rows_csv(rows, outdir=outdir),
        "figures": {},
    }
    if not bool(args.no_plots):
        summary["figures"]["convergence"] = _write_summary_plot(rows, outdir=outdir / "figures")
        fsq_history_plot = _write_fsq_history_plot(rows, outdir=outdir / "figures")
        if fsq_history_plot is not None:
            summary["figures"]["fsq_history"] = fsq_history_plot
        profile_plot = _write_profile_plots(rows, outdir=outdir / "figures")
        if profile_plot is not None:
            summary["figures"]["profiles"] = profile_plot

    summary_path = outdir / "toroidal_stellarator_mirror_hybrid_convergence.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":  # pragma: no cover
    main()

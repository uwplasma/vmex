"""Benchmark fixed-boundary iteration runtime + residual traces (external backends).

Generates two README-friendly figures:
- runtime comparison (vmec2000 vs vmec_jax) for bundled inputs
- residual evolution over iterations for those inputs

Notes:
- This script can run the VMEC2000 executable (`xvmec2000`) for comparisons.
  If it is not available, the corresponding curves/bars are omitted.
- This is a benchmarking/communication script, not a regression test.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vmec_jax.vmec2000_exec import (
    find_vmec2000_exec,
    flatten_threed1,
    run_xvmec2000,
    threed1_fsq_total,
)


@dataclass(frozen=True)
class RunTrace:
    backend: str
    case: str
    iters: int
    seconds: float
    fsq_total: np.ndarray  # shape (iters,)


def _import_matplotlib():
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _distribute_iters(*, iters: int, nstep: int) -> list[int]:
    """Distribute a fixed total iteration budget across multigrid steps.

    VMEC2000 has per-multigrid-step iteration limits. For a fair
    "fixed budget" comparison, we distribute the *total* budget across steps so
    that sum(niter_array) == iters.
    """

    iters = int(iters)
    nstep = int(nstep)
    if iters <= 0:
        return [0]
    if nstep <= 1:
        return [iters]
    base, rem = divmod(iters, nstep)
    if base == 0:
        # If the user asks for fewer total iterations than there are steps,
        # collapse to a single step rather than setting any step to 0.
        return [iters]
    return [base + (1 if i < rem else 0) for i in range(nstep)]


def _run_vmec_jax(
    *,
    input_path: Path,
    case: str,
    iters: int,
    ns_override: int | None,
    warmup: bool,
    use_input_niter: bool,
    jit_forces: bool,
) -> RunTrace:
    import vmec_jax.api as vj

    # Warm up to exclude JAX compilation from timed region. Compilation cost is
    # shape-dependent, so we warm up per-case.
    # Note: some parts of the stack stage/jit by `max_iter`, so warm up with the
    # same iteration count.
    if warmup:
        warm = vj.run_fixed_boundary(
            input_path,
            solver="vmec2000_iter",
            max_iter=int(iters),
            multigrid_use_input_niter=bool(use_input_niter),
            verbose=False,
            ns_override=ns_override,
            jit_forces=bool(jit_forces),
        )
        try:
            warm_res = warm.result
            if warm_res is not None and hasattr(warm_res, "fsqr2_history"):
                h = getattr(warm_res, "fsqr2_history")
                if len(h) > 0:
                    _ = float(np.asarray(h)[-1])
        except Exception:
            pass

    t0 = time.perf_counter()
    run = vj.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        max_iter=int(iters),
        multigrid_use_input_niter=bool(use_input_niter),
        verbose=False,
        ns_override=ns_override,
        jit_forces=bool(jit_forces),
    )
    dt = time.perf_counter() - t0

    # solve_fixed_boundary_residual_iter stores invariant residuals as fsq histories.
    res = run.result
    if res is None or not hasattr(res, "fsqr2_history"):
        raise RuntimeError("vmec_jax run did not return residual histories")
    fsq = np.asarray(res.fsqr2_history) + np.asarray(res.fsqz2_history) + np.asarray(res.fsql2_history)
    fsq = fsq[:iters]
    if fsq.size < iters:
        fsq = np.pad(fsq, (0, iters - fsq.size), constant_values=np.nan)
    return RunTrace(backend="vmec_jax", case=case, iters=iters, seconds=float(dt), fsq_total=fsq)


def _run_vmec2000_exec(
    *,
    input_path: Path,
    case: str,
    iters: int,
    workdir: Path,
    exec_path: Path | None,
    timeout_s: float,
    nstep: int,
    use_input_niter: bool,
    ns_override: int | None,
) -> RunTrace | None:
    exec_path = exec_path or find_vmec2000_exec()
    if exec_path is None:
        return None

    indata_updates: dict[str, str] = {"NSTEP": str(int(nstep))}
    if ns_override is not None:
        indata_updates["NS_ARRAY"] = str(int(ns_override))
    if not use_input_niter:
        niter_steps = _distribute_iters(iters=int(iters), nstep=int(nstep))
        indata_updates["NITER"] = str(int(iters))
        indata_updates["NITER_ARRAY"] = ",".join(str(int(x)) for x in niter_steps)
        indata_updates["FTOL"] = "1e-14"
        indata_updates["FTOL_ARRAY"] = ",".join("1e-14" for _ in niter_steps)

    try:
        result = run_xvmec2000(
            input_path=input_path,
            exec_path=exec_path,
            workdir=workdir,
            timeout_s=float(timeout_s),
            indata_updates=indata_updates,
            keep_workdir=True,
        )
    except subprocess.TimeoutExpired:
        return None

    rows = flatten_threed1(result.stages)
    fsq = threed1_fsq_total(rows)
    if fsq.size == 0:
        return None
    if (not use_input_niter) and (fsq.size < iters):
        fsq = np.pad(fsq, (0, iters - fsq.size), constant_values=np.nan)
    return RunTrace(backend="vmec2000", case=case, iters=int(fsq.size), seconds=float(result.runtime_s), fsq_total=fsq)


def _plot_runtime(*, traces: list[RunTrace], outpath: Path) -> None:
    plt = _import_matplotlib()
    cases = sorted({t.case for t in traces})
    preferred = ["vmec2000", "vmec_jax"]
    present = sorted({t.backend for t in traces}, key=lambda b: (preferred.index(b) if b in preferred else 999, b))
    backends = [b for b in preferred if b in present] + [b for b in present if b not in preferred]
    data = {b: [] for b in backends}
    for case in cases:
        for b in backends:
            t = next((x for x in traces if x.case == case and x.backend == b), None)
            data[b].append(t.seconds if t is not None else np.nan)

    x = np.arange(len(cases))
    width = 0.8 / max(1, len(backends))
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    for i, b in enumerate(backends):
        ax.bar(x + (i - (len(backends) - 1) / 2.0) * width, data[b], width=width, label=b)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("seconds (wall, log scale)")
    ax.set_title("Fixed-boundary runtime for a fixed iteration budget")
    ax.legend(ncols=3, frameon=False)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_residuals(*, traces: list[RunTrace], outpath: Path) -> None:
    plt = _import_matplotlib()
    cases = sorted({t.case for t in traces})
    preferred = ["vmec2000", "vmec_jax"]
    present = sorted({t.backend for t in traces}, key=lambda b: (preferred.index(b) if b in preferred else 999, b))
    backends = [b for b in preferred if b in present] + [b for b in present if b not in preferred]
    n = len(cases)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    style = {
        "vmec2000": dict(
            ls="--",
            lw=1.8,
            marker="o",
            ms=3.0,
            mfc="none",
            mew=0.8,
            alpha=0.9,
            zorder=3,
        ),
        "vmec_jax": dict(ls="-", lw=1.6, marker=None, ms=0.0, alpha=0.9, zorder=2),
    }
    for idx, case in enumerate(cases):
        ax = axes[idx // ncols, idx % ncols]
        for b in backends:
            t = next((x for x in traces if x.case == case and x.backend == b), None)
            if t is None:
                continue
            y = np.asarray(t.fsq_total, dtype=float)
            x = np.arange(1, y.size + 1)
            st = style.get(b, {})
            ax.plot(x, y, label=b, **st)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("fsq_total")
        ax.set_title(case)
        ax.grid(True, which="both", alpha=0.3)
    # Hide unused axes.
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncols=min(3, len(handles)), frameon=False, loc="upper center")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_objective(*, traces: list[RunTrace], outpath: Path) -> None:
    plt = _import_matplotlib()
    cases = sorted({t.case for t in traces})
    preferred = ["vmec2000", "vmec_jax"]
    present = sorted({t.backend for t in traces}, key=lambda b: (preferred.index(b) if b in preferred else 999, b))
    backends = [b for b in preferred if b in present] + [b for b in present if b not in preferred]
    data = {b: [] for b in backends}
    for case in cases:
        for b in backends:
            t = next((x for x in traces if x.case == case and x.backend == b), None)
            if t is None or t.fsq_total.size == 0:
                data[b].append(np.nan)
            else:
                data[b].append(float(t.fsq_total[min(len(t.fsq_total), t.iters) - 1]))

    x = np.arange(len(cases))
    width = 0.8 / max(1, len(backends))
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    for i, b in enumerate(backends):
        ax.bar(x + (i - (len(backends) - 1) / 2.0) * width, data[b], width=width, label=b)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("fsq_total (final, log scale)")
    ax.set_title("Fixed-boundary objective after a fixed iteration budget")
    ax.legend(ncols=3, frameon=False)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases",
        nargs="*",
        # Keep defaults small; this script is for README figures, not profiling.
        default=["circular_tokamak", "shaped_tokamak_pressure", "solovev", "purely_toroidal_field"],
    )
    # Keep the default small; this script is for README figures, not profiling.
    p.add_argument("--iters", type=int, default=10, help="Fixed iteration budget for all backends.")
    p.add_argument(
        "--ns-override",
        type=int,
        default=13,
        help="Override ns for vmec_jax (vmec2000 backend uses input resolution).",
    )
    p.add_argument(
        "--disable-jit",
        action="store_true",
        help="Disable JAX JIT (reduces compilation overhead for quick runs).",
    )
    p.add_argument(
        "--jit-forces",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="JIT the force/residual pipeline in vmec_jax (best for performance).",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the warmup run (reduces runtime for quick checks).",
    )
    p.add_argument(
        "--run-vmec2000",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run the VMEC2000 executable (xvmec2000) for comparisons when available.",
    )
    p.add_argument(
        "--vmec2000-exec",
        default=None,
        help="Path to xvmec2000 (overrides VMEC2000_EXEC env).",
    )
    p.add_argument(
        "--vmec2000-timeout",
        type=float,
        default=60.0,
        help="Timeout (s) for each VMEC2000 executable run.",
    )
    p.add_argument(
        "--vmec2000-nstep",
        type=int,
        default=1,
        help="Override VMEC2000 NSTEP (printout cadence).",
    )
    p.add_argument(
        "--vmec2000-ns-override",
        type=int,
        default=13,
        help="Override VMEC2000 NS/NS_ARRAY (single-grid) for parity-focused traces.",
    )
    p.add_argument(
        "--vmec2000-use-input-niter",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use input NITER_ARRAY/FTOL_ARRAY (skip fixed-budget override).",
    )
    p.add_argument(
        "--jax-use-input-niter",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use input NITER_ARRAY/FTOL_ARRAY for vmec_jax staging.",
    )
    p.add_argument("--outdir", default="examples/outputs/bench_fixed_boundary", help="Output directory root.")
    p.add_argument(
        "--promote-docs-assets",
        action="store_true",
        help=(
            "Copy the rendered benchmark PNGs into docs/_static/figures. "
            "Leave disabled for routine benchmarking to keep generated artifacts out of git."
        ),
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Quick sanity run (reduces cases/iters and uses a smaller ns for vmec_jax).",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "examples" / "data"
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Make matplotlib cache writable and deterministic-ish.
    os.environ.setdefault("MPLCONFIGDIR", str(outdir / "_mplcache"))

    cases = list(args.cases)
    if bool(args.fast):
        if cases == ["circular_tokamak", "shaped_tokamak_pressure", "solovev", "purely_toroidal_field"]:
            cases = ["circular_tokamak"]
        if args.iters == 10:
            args.iters = 5
        if args.ns_override is None:
            args.ns_override = 9

    if bool(args.disable_jit):
        os.environ.setdefault("JAX_DISABLE_JIT", "1")

    traces: list[RunTrace] = []
    exec_path = Path(args.vmec2000_exec).expanduser() if args.vmec2000_exec else find_vmec2000_exec()
    if bool(args.run_vmec2000) and exec_path is None:
        print("[bench] VMEC2000 executable not found; skipping VMEC2000 traces.", flush=True)
        args.run_vmec2000 = False
    for case in cases:
        input_path = data_dir / f"input.{case}"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing bundled input: {input_path}")

        print(f"[bench] vmec_jax case={case} iters={args.iters} ns_override={args.ns_override}", flush=True)
        traces.append(
            _run_vmec_jax(
                input_path=input_path,
                case=case,
                iters=int(args.iters),
                ns_override=args.ns_override,
                warmup=not bool(args.no_warmup),
                use_input_niter=bool(args.jax_use_input_niter),
                jit_forces=bool(args.jit_forces),
            )
        )

        # External backends are optional and may not be installed.
        # Keep the overall script robust: skip a backend rather than failing the run.
        if bool(args.run_vmec2000):
            print(
                f"[bench] vmec2000 case={case} iters={args.iters} ns_override={args.vmec2000_ns_override}",
                flush=True,
            )
            try:
                tvm = _run_vmec2000_exec(
                    input_path=input_path,
                    case=case,
                    iters=int(args.iters),
                    workdir=outdir / "vmec2000" / case,
                    exec_path=exec_path,
                    timeout_s=float(args.vmec2000_timeout),
                    nstep=int(args.vmec2000_nstep),
                    use_input_niter=bool(args.vmec2000_use_input_niter),
                    ns_override=args.vmec2000_ns_override,
                )
            except Exception:
                tvm = None
            if tvm is not None:
                traces.append(tvm)
            else:
                raise RuntimeError(
                    f"VMEC2000 trace missing for case={case!r}. "
                    "Re-run with a larger --vmec2000-timeout or reduced --vmec2000-ns-override."
                )

    # Write machine-readable data for later reuse.
    payload: dict[str, Any] = {
        "iters": int(args.iters),
        "cases": list(cases),
        "ns_override": args.ns_override,
        "traces": [],
    }
    for t in traces:
        payload["traces"].append(
            {
                "backend": t.backend,
                "case": t.case,
                "iters": int(t.iters),
                "seconds": float(t.seconds),
                "fsq_total": np.asarray(t.fsq_total, dtype=float).tolist(),
            }
        )
    (outdir / "bench_fixed_boundary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fig_runtime = outdir / "bench_fixed_boundary_runtime.png"
    fig_resid = outdir / "bench_fixed_boundary_residual.png"
    fig_obj = outdir / "bench_fixed_boundary_objective.png"
    _plot_runtime(traces=traces, outpath=fig_runtime)
    _plot_residuals(traces=traces, outpath=fig_resid)
    _plot_objective(traces=traces, outpath=fig_obj)

    if bool(args.promote_docs_assets):
        docs_fig_dir = root / "docs" / "_static" / "figures"
        docs_fig_dir.mkdir(parents=True, exist_ok=True)
        (docs_fig_dir / fig_runtime.name).write_bytes(fig_runtime.read_bytes())
        (docs_fig_dir / fig_resid.name).write_bytes(fig_resid.read_bytes())
        (docs_fig_dir / fig_obj.name).write_bytes(fig_obj.read_bytes())
        print(f"[bench] promoted docs assets into: {docs_fig_dir}")

    print(f"[bench] wrote: {fig_runtime}")
    print(f"[bench] wrote: {fig_resid}")
    print(f"[bench] wrote: {fig_obj}")
    print(f"[bench] wrote: {outdir / 'bench_fixed_boundary.json'}")


if __name__ == "__main__":
    main()

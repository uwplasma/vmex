"""Benchmark fixed-boundary iteration runtime + residual traces (external backends).

Generates two README-friendly figures:
- runtime comparison (vmec2000 vs vmecpp vs vmec_jax) for 4 bundled inputs
- residual evolution over iterations for those inputs

Notes:
- This script depends on external backends (`vmec`, `vmecpp`). If they are not
  installed, the corresponding curves/bars are omitted.
- This is a benchmarking/communication script, not a regression test.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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


def _maybe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def _run_vmec_jax(*, input_path: Path, case: str, iters: int) -> RunTrace:
    import vmec_jax.api as vj

    t0 = time.perf_counter()
    run = vj.run_fixed_boundary(input_path, solver="vmecpp_iter", max_iter=int(iters), verbose=False)
    dt = time.perf_counter() - t0

    # solve_fixed_boundary_vmecpp_iter stores invariant residuals as fsq histories.
    res = run.result
    if res is None or not hasattr(res, "fsqr2_history"):
        raise RuntimeError("vmec_jax run did not return vmecpp_iter residual histories")
    fsq = np.asarray(res.fsqr2_history) + np.asarray(res.fsqz2_history) + np.asarray(res.fsql2_history)
    fsq = fsq[:iters]
    if fsq.size < iters:
        fsq = np.pad(fsq, (0, iters - fsq.size), constant_values=np.nan)
    return RunTrace(backend="vmec_jax", case=case, iters=iters, seconds=float(dt), fsq_total=fsq)


def _run_vmecpp(*, input_path: Path, case: str, iters: int, max_threads: int) -> RunTrace | None:
    vmecpp = _maybe_import("vmecpp")
    if vmecpp is None:
        return None

    inp = vmecpp.VmecInput.from_file(str(input_path))
    # Force a fixed iteration budget and return outputs even if not converged.
    # In VMEC++'s VMEC-style iteration, `fsqt` length is typically 2*niter_array[0].
    inp.return_outputs_even_if_not_converged = True
    n_half = int(np.ceil(iters / 2))
    # Use a plain Python list here. Some vmecpp builds are sensitive to numpy dtype/shape
    # when assigning Eigen-backed arrays via pybind11.
    inp.niter_array = [n_half, n_half]

    t0 = time.perf_counter()
    out = vmecpp.run(inp, verbose=False, max_threads=int(max_threads))
    dt = time.perf_counter() - t0

    fsq = np.asarray(out.wout.fsqt, dtype=float)
    fsq = fsq[:iters]
    if fsq.size < iters:
        fsq = np.pad(fsq, (0, iters - fsq.size), constant_values=np.nan)
    return RunTrace(backend="vmecpp", case=case, iters=iters, seconds=float(dt), fsq_total=fsq)


def _vmec_expected_wout_name(input_path: Path) -> str:
    # input.<name> -> wout_<name>.nc (VMEC convention)
    name = input_path.name
    if name.startswith("input."):
        name = name[len("input.") :]
    return f"wout_{name}.nc"


def _run_vmec2000(*, input_path: Path, case: str, iters: int, workdir: Path) -> RunTrace | None:
    vmec = _maybe_import("vmec")
    if vmec is None:
        return None

    # vmec.runvmec uses an MPI communicator; keep it local.
    try:
        from mpi4py import MPI  # type: ignore
    except Exception:
        return None

    restart_flag = 1
    readin_flag = 2
    timestep_flag = 4
    output_flag = 8

    ictrl = np.zeros(5, dtype=np.int32)
    reset_file = ""
    fcomm = MPI.COMM_SELF.py2f()

    wout_name = _vmec_expected_wout_name(input_path)

    old_cwd = Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    try:
        # Stage 1: read input only.
        ictrl[:] = 0
        ictrl[0] = restart_flag + readin_flag
        t_read = time.perf_counter()
        vmec.runvmec(ictrl, str(input_path), False, fcomm, reset_file)
        _ = time.perf_counter() - t_read

        # Override iteration controls for a fixed budget.
        vmec.vmec_input.niter = int(iters)
        vmec.vmec_input.ftol = 0.0

        # Stage 2: timestep + output (no re-read).
        ictrl[:] = 0
        ictrl[0] = restart_flag + timestep_flag + output_flag
        t0 = time.perf_counter()
        vmec.runvmec(ictrl, str(input_path), False, fcomm, reset_file)
        dt = time.perf_counter() - t0

        # Read residual trace from wout.
        from vmec_jax.wout import read_wout

        wout_path = Path(wout_name)
        if not wout_path.exists():
            # Some builds write wout_<name> (no .nc) or wout.<name>.nc; probe a few.
            probes = [
                Path(wout_name.replace(".nc", "")),
                Path("wout." + wout_name[len("wout_") :]),
            ]
            for p in probes:
                if p.exists():
                    wout_path = p
                    break
        wout = read_wout(wout_path)
        fsq = np.asarray(getattr(wout, "fsqt", np.zeros((0,), dtype=float)), dtype=float)
        if fsq.size == 0:
            # Fall back: use final invariant sum only.
            fsq = np.full((iters,), float(wout.fsqr + wout.fsqz + wout.fsql), dtype=float)
        else:
            fsq = fsq[:iters]
            if fsq.size < iters:
                fsq = np.pad(fsq, (0, iters - fsq.size), constant_values=np.nan)

        return RunTrace(backend="vmec2000", case=case, iters=iters, seconds=float(dt), fsq_total=fsq)
    finally:
        try:
            vmec.cleanup(True)
        except Exception:
            pass
        os.chdir(old_cwd)


def _plot_runtime(*, traces: list[RunTrace], outpath: Path) -> None:
    plt = _import_matplotlib()
    cases = sorted({t.case for t in traces})
    backends = ["vmec2000", "vmecpp", "vmec_jax"]
    data = {b: [] for b in backends}
    for case in cases:
        for b in backends:
            t = next((x for x in traces if x.case == case and x.backend == b), None)
            data[b].append(t.seconds if t is not None else np.nan)

    x = np.arange(len(cases))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    for i, b in enumerate(backends):
        ax.bar(x + (i - 1) * width, data[b], width=width, label=b)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15, ha="right")
    ax.set_ylabel("seconds (wall)")
    ax.set_title("Fixed-boundary runtime for a fixed iteration budget")
    ax.legend(ncols=3, frameon=False)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_residuals(*, traces: list[RunTrace], outpath: Path) -> None:
    plt = _import_matplotlib()
    cases = sorted({t.case for t in traces})
    backends = ["vmec2000", "vmecpp", "vmec_jax"]
    n = len(cases)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for idx, case in enumerate(cases):
        ax = axes[idx // ncols, idx % ncols]
        for b in backends:
            t = next((x for x in traces if x.case == case and x.backend == b), None)
            if t is None:
                continue
            y = np.asarray(t.fsq_total, dtype=float)
            x = np.arange(1, y.size + 1)
            ax.plot(x, y, lw=1.5, label=b)
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
        fig.legend(handles, labels, ncols=3, frameon=False, loc="upper center")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases",
        nargs="*",
        # Keep defaults to inputs known to run in vmecpp+vmec2000 wrappers.
        default=["circular_tokamak", "vmecpp_solovev", "cth_like_fixed_bdy", "nfp4_QH_warm_start"],
    )
    # Keep the default small; this script is for README figures, not profiling.
    p.add_argument("--iters", type=int, default=20, help="Fixed iteration budget for all backends.")
    p.add_argument("--max-threads", type=int, default=1, help="VMEC++ max_threads.")
    p.add_argument("--outdir", default="examples/outputs/bench_fixed_boundary", help="Output directory root.")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "examples" / "data"
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Make matplotlib cache writable and deterministic-ish.
    os.environ.setdefault("MPLCONFIGDIR", str(outdir / "_mplcache"))

    traces: list[RunTrace] = []
    for case in args.cases:
        input_path = data_dir / f"input.{case}"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing bundled input: {input_path}")

        traces.append(_run_vmec_jax(input_path=input_path, case=case, iters=int(args.iters)))

        # External backends are optional and may not be installed.
        # Keep the overall script robust: skip a backend rather than failing the run.
        try:
            tpp = _run_vmecpp(
                input_path=input_path, case=case, iters=int(args.iters), max_threads=int(args.max_threads)
            )
        except Exception:
            tpp = None
        if tpp is not None:
            traces.append(tpp)

        try:
            tvm = _run_vmec2000(
                input_path=input_path, case=case, iters=int(args.iters), workdir=outdir / "vmec2000" / case
            )
        except Exception:
            tvm = None
        if tvm is not None:
            traces.append(tvm)

    # Write machine-readable data for later reuse.
    payload: dict[str, Any] = {"iters": int(args.iters), "cases": list(args.cases), "traces": []}
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
    _plot_runtime(traces=traces, outpath=fig_runtime)
    _plot_residuals(traces=traces, outpath=fig_resid)

    # Copy into docs for README embedding.
    docs_fig_dir = root / "docs" / "_static" / "figures"
    docs_fig_dir.mkdir(parents=True, exist_ok=True)
    (docs_fig_dir / fig_runtime.name).write_bytes(fig_runtime.read_bytes())
    (docs_fig_dir / fig_resid.name).write_bytes(fig_resid.read_bytes())

    print(f"[bench] wrote: {fig_runtime}")
    print(f"[bench] wrote: {fig_resid}")
    print(f"[bench] wrote: {outdir / 'bench_fixed_boundary.json'}")


if __name__ == "__main__":
    main()

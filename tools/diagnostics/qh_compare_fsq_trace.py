"""Generate VMEC2000 vs vmec_jax fsq_total trace and runtime for QH."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vmec_jax.config import load_config
from vmec_jax.driver import run_fixed_boundary
from vmec_jax.vmec2000_exec import find_vmec2000_exec, run_xvmec2000


def _collect_vmec2000_trace(input_path: Path, *, niter: int, workdir: Path):
    exe = find_vmec2000_exec()
    if exe is None:
        raise SystemExit("xvmec2000 executable not found")
    cfg, _ = load_config(str(input_path))
    vmec = run_xvmec2000(
        input_path,
        exec_path=exe,
        workdir=workdir,
        timeout_s=300.0,
        indata_updates={
            "NITER": str(niter),
            "NS_ARRAY": f"{int(cfg.ns)}",
            "NITER_ARRAY": f"{niter}",
        },
        keep_workdir=True,
    )
    fsq = []
    for stage in vmec.stages:
        for row in stage.rows:
            fsq.append(float(row.fsqr + row.fsqz + row.fsql))
    return np.asarray(fsq), float(vmec.runtime_s)


def _collect_vmec_jax_trace(input_path: Path, *, niter: int):
    t0 = time.perf_counter()
    res = run_fixed_boundary(
        str(input_path),
        solver="vmec2000_iter",
        max_iter=int(niter),
        multigrid=False,
        multigrid_use_input_niter=False,
        verbose=False,
    )
    runtime = time.perf_counter() - t0
    fsq = np.asarray(res.result.fsqr2_history) + np.asarray(res.result.fsqz2_history) + np.asarray(
        res.result.fsql2_history
    )
    return fsq, float(runtime)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/input.nfp4_QH_warm_start"),
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "docs/_static/figures"),
    )
    p.add_argument("--niter", type=int, default=50)
    p.add_argument("--workdir", type=str, default=None, help="Workdir for VMEC2000 run (defaults under outdir).")
    args = p.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else (outdir / "qh_vmec2000_trace")
    workdir.mkdir(parents=True, exist_ok=True)

    fsq_vmec, t_vmec = _collect_vmec2000_trace(input_path, niter=int(args.niter), workdir=workdir)
    fsq_jax, t_jax = _collect_vmec_jax_trace(input_path, niter=int(args.niter))

    n = min(fsq_vmec.size, fsq_jax.size)
    it = np.arange(1, n + 1)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    ax.plot(it, fsq_vmec[:n], lw=2.0, label="VMEC2000")
    ax.plot(it, fsq_jax[:n], lw=2.0, label="vmec_jax")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fsq_total")
    ax.set_title("QH: fsq_total trace (50 iters)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    txt = f"VMEC2000: {t_vmec:.2f}s\\nvmec_jax: {t_jax:.2f}s"
    ax.text(
        0.98,
        0.95,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    outpath = outdir / "qh_compare_fsq_trace.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

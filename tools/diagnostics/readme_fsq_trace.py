"""Generate README fsq_total traces for axisymmetric and QH cases."""

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
        timeout_s=600.0,
        indata_updates={
            "NITER": str(niter),
            "NSTEP": "1",
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


def _plot_panel(ax, *, fsq_vmec, fsq_jax, title: str, t_vmec: float, t_jax: float):
    n = min(fsq_vmec.size, fsq_jax.size)
    it = np.arange(1, n + 1)
    ax.plot(it, fsq_vmec[:n], lw=2.8, linestyle="--", label="VMEC2000", zorder=2)
    ax.plot(it, fsq_jax[:n], lw=2.8, linestyle="-", label="vmec_jax", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fsq_total")
    ax.set_title(title)
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--axisym-input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/input.shaped_tokamak_pressure"),
    )
    p.add_argument(
        "--qh-input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/input.nfp4_QH_warm_start"),
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "docs/_static/figures"),
    )
    p.add_argument("--niter", type=int, default=100)
    args = p.parse_args()

    axisym_input = Path(args.axisym_input).expanduser().resolve()
    qh_input = Path(args.qh_input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    axisym_work = outdir / "readme_axisym_vmec2000_trace"
    qh_work = outdir / "readme_qh_vmec2000_trace"
    axisym_work.mkdir(parents=True, exist_ok=True)
    qh_work.mkdir(parents=True, exist_ok=True)

    fsq_vmec_a, t_vmec_a = _collect_vmec2000_trace(axisym_input, niter=int(args.niter), workdir=axisym_work)
    fsq_jax_a, t_jax_a = _collect_vmec_jax_trace(axisym_input, niter=int(args.niter))

    fsq_vmec_q, t_vmec_q = _collect_vmec2000_trace(qh_input, niter=int(args.niter), workdir=qh_work)
    fsq_jax_q, t_jax_q = _collect_vmec_jax_trace(qh_input, niter=int(args.niter))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    _plot_panel(
        axes[0],
        fsq_vmec=fsq_vmec_a,
        fsq_jax=fsq_jax_a,
        title=f"Axisymmetric fsq_total trace ({int(args.niter)} iters)",
        t_vmec=t_vmec_a,
        t_jax=t_jax_a,
    )
    _plot_panel(
        axes[1],
        fsq_vmec=fsq_vmec_q,
        fsq_jax=fsq_jax_q,
        title=f"QH fsq_total trace ({int(args.niter)} iters)",
        t_vmec=t_vmec_q,
        t_jax=t_jax_q,
    )
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    outpath = outdir / "readme_fsq_trace.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()

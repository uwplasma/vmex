"""Generate README fsq_total traces for optimized axisymmetric and stellarator cases."""

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
from vmec_jax.vmec2000_exec import _patch_indata, find_vmec2000_exec, run_xvmec2000


def _collect_vmec2000_trace(input_path: Path, *, niter: int, ftol: float, workdir: Path):
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
            "FTOL": f"{float(ftol):.3e}",
            "FTOL_ARRAY": f"{float(ftol):.3e}",
        },
        keep_workdir=True,
    )
    fsq = []
    for stage in vmec.stages:
        for row in stage.rows:
            fsq.append(float(row.fsqr + row.fsqz + row.fsql))
    return np.asarray(fsq), float(vmec.runtime_s)


def _collect_vmec_jax_trace(
    input_path: Path,
    *,
    niter: int,
    ftol: float,
    workdir: Path,
    solver_mode: str,
    cli_fixed_boundary_mode: bool,
):
    patched = _patch_indata(
        input_path.read_text(),
        updates={
            "NITER": str(niter),
            "NSTEP": "1",
            "FTOL": f"{float(ftol):.3e}",
            "FTOL_ARRAY": f"{float(ftol):.3e}",
        },
    )
    tmp_input = workdir / f"input_patched_{input_path.name}"
    tmp_input.write_text(patched)
    t0 = time.perf_counter()
    res = run_fixed_boundary(
        str(tmp_input),
        solver="vmec2000_iter",
        max_iter=int(niter),
        multigrid=False,
        multigrid_use_input_niter=False,
        verbose=False,
        solver_mode=str(solver_mode),
        performance_mode=bool(str(solver_mode) != "parity"),
        cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
    )
    runtime = time.perf_counter() - t0
    fsq = np.asarray(res.result.fsqr2_history) + np.asarray(res.result.fsqz2_history) + np.asarray(
        res.result.fsql2_history
    )
    return fsq, float(runtime)


def _plot_panel(ax, *, fsq_vmec, fsq_jax, title: str, t_vmec: float, t_jax: float, jax_label: str):
    n = min(fsq_vmec.size, fsq_jax.size)
    it = np.arange(1, n + 1)
    ax.plot(it, fsq_vmec[:n], lw=2.8, linestyle="--", label="VMEC2000", zorder=2)
    ax.plot(it, fsq_jax[:n], lw=2.8, linestyle="-", label=jax_label, zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fsq_total")
    ax.set_title(f"{title}\nVMEC2000 {t_vmec:.2f}s | {jax_label} {t_jax:.2f}s", fontsize=11)
    ax.grid(alpha=0.3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--axisym-input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/input.shaped_tokamak_pressure"),
    )
    p.add_argument(
        "--stellarator-input",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "examples/data/input.LandremanPaul2021_QA_lowres"),
    )
    p.add_argument(
        "--qh-input",
        type=str,
        default="",
        help="Deprecated alias for --stellarator-input.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "docs/_static/figures"),
    )
    p.add_argument("--niter", type=int, default=250)
    p.add_argument("--ftol", type=float, default=1e-14)
    p.add_argument("--solver-mode", type=str, default="accelerated")
    p.add_argument("--jax-label", type=str, default="vmec_jax optimized")
    p.add_argument(
        "--cli-fixed-boundary-mode",
        dest="cli_fixed_boundary_mode",
        action="store_true",
        help="Use the optimized CLI-style fixed-boundary controller for vmec_jax traces.",
    )
    p.add_argument(
        "--no-cli-fixed-boundary-mode",
        dest="cli_fixed_boundary_mode",
        action="store_false",
        help="Disable the CLI-style fixed-boundary controller for vmec_jax traces.",
    )
    p.set_defaults(cli_fixed_boundary_mode=True)
    args = p.parse_args()

    axisym_input = Path(args.axisym_input).expanduser().resolve()
    stellarator_input = Path(args.stellarator_input).expanduser().resolve()
    if args.qh_input:
        stellarator_input = Path(args.qh_input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    axisym_work = outdir / "readme_axisym_vmec2000_trace"
    st_work = outdir / "readme_stellarator_vmec2000_trace"
    axisym_work.mkdir(parents=True, exist_ok=True)
    st_work.mkdir(parents=True, exist_ok=True)

    fsq_vmec_a, t_vmec_a = _collect_vmec2000_trace(
        axisym_input, niter=int(args.niter), ftol=float(args.ftol), workdir=axisym_work
    )
    fsq_jax_a, t_jax_a = _collect_vmec_jax_trace(
        axisym_input,
        niter=int(args.niter),
        ftol=float(args.ftol),
        workdir=axisym_work,
        solver_mode=str(args.solver_mode),
        cli_fixed_boundary_mode=bool(args.cli_fixed_boundary_mode),
    )

    fsq_vmec_s, t_vmec_s = _collect_vmec2000_trace(
        stellarator_input, niter=int(args.niter), ftol=float(args.ftol), workdir=st_work
    )
    fsq_jax_s, t_jax_s = _collect_vmec_jax_trace(
        stellarator_input,
        niter=int(args.niter),
        ftol=float(args.ftol),
        workdir=st_work,
        solver_mode=str(args.solver_mode),
        cli_fixed_boundary_mode=bool(args.cli_fixed_boundary_mode),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    _plot_panel(
        axes[0],
        fsq_vmec=fsq_vmec_a,
        fsq_jax=fsq_jax_a,
        title=f"Axisymmetric fsq_total trace ({int(args.niter)} iters)",
        t_vmec=t_vmec_a,
        t_jax=t_jax_a,
        jax_label=str(args.jax_label),
    )
    _plot_panel(
        axes[1],
        fsq_vmec=fsq_vmec_s,
        fsq_jax=fsq_jax_s,
        title=f"LandremanPaul QA fsq_total trace ({int(args.niter)} iters)",
        t_vmec=t_vmec_s,
        t_jax=t_jax_s,
        jax_label=str(args.jax_label),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    outpath = outdir / "readme_fsq_trace.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()

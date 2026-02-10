"""Showcase: input -> fixed-boundary solve -> wout -> plots -> parity printout.

This script is the recommended starting point for new users.
Edit the `input.*` file to change the configuration; the script is intentionally
light on flags.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

import vmec_jax.api as vj
from vmec_jax.vmec2000_exec import find_vmec2000_exec, flatten_threed1, run_xvmec2000, threed1_fsq_total


def _import_matplotlib():
    try:
        # Ensure Matplotlib's cache/config lives in a writable directory (CI, sandboxed
        # environments, and some HPC setups may have a non-writable $HOME).
        mpl_cache = Path(tempfile.gettempdir()) / "vmec_jax_mplconfig"
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

        import matplotlib as mpl

        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("This example requires matplotlib (pip install -e .[plots]).") from e
    return plt


def _maybe_bmag_from_wout_physical(wout, *, theta: np.ndarray, phi: np.ndarray, s_index: int) -> np.ndarray | None:
    # vmec_jax currently writes minimal wouts for solver outputs. Those wouts may
    # not include Nyquist `bmnc/bmns` yet, so fall back to state-based evaluation.
    try:
        bmnc = np.asarray(getattr(wout, "bmnc", None))
    except Exception:
        bmnc = None
    if bmnc is None or bmnc.size == 0 or not np.any(bmnc):
        return None

    from vmec_jax.plotting import bmag_from_wout_physical

    return np.asarray(bmag_from_wout_physical(wout, theta=theta, phi=phi, s_index=int(s_index)))


def _write_plots(
    *,
    outdir: Path,
    run: vj.FixedBoundaryRun,
    wout_new,
    wout_ref,
    indata,
    vmec2000_fsq_total: np.ndarray | None,
) -> None:
    plt = _import_matplotlib()

    # Keep showcase plots reasonably fast; these are illustrative, not high-res exports.
    theta = vj.closed_theta_grid(128)
    phi = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    ns = int(wout_new.ns)
    s_index_lcfs = ns - 1
    s_indices = np.linspace(0, s_index_lcfs, 9).round().astype(int)

    # 1) Nested flux surfaces (phi=0) from reference and new wout.
    fig, ax = plt.subplots(1, 2 if wout_ref is not None else 1, figsize=(10, 4), constrained_layout=True)
    ax = np.atleast_1d(ax)
    for si in s_indices:
        Rn, Zn = vj.surface_rz_from_wout_physical(
            wout_new, theta=theta, phi=np.asarray([0.0]), s_index=int(si), nyq=False
        )
        ax[-1].plot(Rn[:, 0], Zn[:, 0], lw=1.2)
        if wout_ref is not None:
            Rr, Zr = vj.surface_rz_from_wout_physical(
                wout_ref, theta=theta, phi=np.asarray([0.0]), s_index=int(si), nyq=False
            )
            ax[0].plot(Rr[:, 0], Zr[:, 0], lw=1.2)
    if wout_ref is not None:
        ax[0].set_title("VMEC2000 (reference)")
        ax[1].set_title("vmec_jax (new wout)")
    else:
        ax[0].set_title("vmec_jax (new wout)")
    for a in ax:
        a.set_aspect("equal", "box")
        a.set_xlabel("R")
        a.set_ylabel("Z")
    fig.savefig(outdir / "surfaces_nested_phi0.png", dpi=180)
    plt.close(fig)

    # 2) |B| on LCFS.
    #
    # For reference VMEC wouts, prefer Nyquist `bmnc/bmns` evaluation (matches vmecPlot2).
    # For solver-produced minimal wouts, fall back to state-based evaluation if Nyquist
    # coefficients are not written yet.
    B_new = _maybe_bmag_from_wout_physical(wout_new, theta=theta, phi=phi, s_index=s_index_lcfs)
    if B_new is None:
        st_new = vj.state_from_wout(wout_new)
        B_new = vj.bmag_from_state_physical(
            st_new, run.static, indata=indata, theta=theta, phi=phi, s_index=s_index_lcfs
        )

    B_ref = None
    if wout_ref is not None:
        # For comparisons, use the same state-based pathway for both reference and
        # new wout, since vmec_jax solver outputs do not yet populate Nyquist
        # `bmnc/bmns` in `wout_*.nc`.
        st_ref = vj.state_from_wout(wout_ref)
        B_ref = vj.bmag_from_state_physical(
            st_ref, run.static, indata=indata, theta=theta, phi=phi, s_index=s_index_lcfs
        )

    fig, ax = plt.subplots(1, 2 if B_ref is not None else 1, figsize=(10, 4), constrained_layout=True)
    ax = np.atleast_1d(ax)
    vmin = float(np.min(B_new if B_ref is None else np.minimum(B_new, B_ref)))
    vmax = float(np.max(B_new if B_ref is None else np.maximum(B_new, B_ref)))
    im_new = ax[-1].imshow(B_new, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax[-1].set_title("vmec_jax |B| (LCFS)")
    if B_ref is not None:
        im_ref = ax[0].imshow(B_ref, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax[0].set_title("VMEC2000 |B| (LCFS)")
    for a in ax:
        a.set_xlabel("phi index")
        a.set_ylabel("theta index")
    if B_ref is not None:
        fig.colorbar(im_ref, ax=ax[0], shrink=0.85, pad=0.02)
    fig.colorbar(im_new, ax=ax[-1], shrink=0.85, pad=0.02)
    fig.savefig(outdir / "bmag_lcfs.png", dpi=180)
    plt.close(fig)

    # 3) 3D LCFS surface colored by |B| (new wout).
    th3 = vj.closed_theta_grid(80)
    ph3 = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    Rlcfs, Zlcfs = vj.surface_rz_from_wout_physical(wout_new, theta=th3, phi=ph3, s_index=s_index_lcfs, nyq=False)
    Blcfs = _maybe_bmag_from_wout_physical(wout_new, theta=th3, phi=ph3, s_index=s_index_lcfs)
    if Blcfs is None:
        st_new = vj.state_from_wout(wout_new)
        Blcfs = vj.bmag_from_state_physical(st_new, run.static, indata=indata, theta=th3, phi=ph3, s_index=s_index_lcfs)
    X = Rlcfs * np.cos(ph3[None, :])
    Y = Rlcfs * np.sin(ph3[None, :])
    fig = plt.figure(figsize=(6, 5), constrained_layout=True)
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot_surface(
        X,
        Y,
        Zlcfs,
        facecolors=plt.cm.viridis((Blcfs - Blcfs.min()) / max(float(np.ptp(Blcfs)), 1e-12)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    vj.fix_matplotlib_3d(ax3)
    ax3.set_title("LCFS 3D colored by |B| (vmec_jax)")
    fig.savefig(outdir / "lcfs_3d_bmag.png", dpi=180)
    plt.close(fig)

    # 4) Residual / objective trace (when available).
    res = getattr(run, "result", None)
    w_hist = None
    stage_offsets = None
    if res is not None:
        wh = getattr(res, "w_history", None)
        if wh is not None:
            w_hist = np.asarray(wh, dtype=float)
        diag = getattr(res, "diagnostics", {}) or {}
        if isinstance(diag, dict) and diag.get("multigrid_stage_offsets", None) is not None:
            stage_offsets = np.asarray(diag["multigrid_stage_offsets"], dtype=int)

    if w_hist is not None and w_hist.size > 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), constrained_layout=True)
        x = np.arange(w_hist.size, dtype=float)
        ax.semilogy(x, np.maximum(w_hist, 1e-300), lw=1.6, label="vmec_jax (solver metric)")
        if vmec2000_fsq_total is not None and vmec2000_fsq_total.size > 0:
            vmec_trace = vmec2000_fsq_total
            if vmec_trace.size > w_hist.size:
                vmec_trace = vmec_trace[: w_hist.size]
            x_vm = np.arange(vmec_trace.size, dtype=float)
            ax.semilogy(x_vm, np.maximum(vmec_trace, 1e-300), lw=1.2, ls="--", label="VMEC2000 fsq_total")
        elif wout_ref is not None:
            fsq_ref = float(wout_ref.fsqr + wout_ref.fsqz + wout_ref.fsql)
            if np.isfinite(fsq_ref) and fsq_ref > 0.0:
                ax.axhline(fsq_ref, color="k", ls="--", lw=1.0, alpha=0.6, label="VMEC2000 fsq_total")
        if stage_offsets is not None and stage_offsets.size:
            for off in stage_offsets[1:]:
                ax.axvline(float(off), color="k", lw=0.8, alpha=0.15)
        ax.set_xlabel("iteration")
        ax.set_ylabel("fsq_total")
        ax.set_title("Nonlinear solve trace (vmec2000_iter)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.savefig(outdir / "residual_trace.png", dpi=180)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="shaped_tokamak_pressure")
    p.add_argument(
        "--suite",
        action="store_true",
        help="Run the 3 bundled axisymmetric cases and print a parity summary.",
    )
    p.add_argument(
        "--solver",
        default="vmec2000_iter",
        choices=["vmec2000_iter", "vmec_gn", "gd", "lbfgs"],
    )
    p.add_argument("--max-iter", type=int, default=120)
    p.add_argument(
        "--verbose",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Print per-iteration solver trace (can be very verbose).",
    )
    p.add_argument(
        "--use-input-niter",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="For vmec2000_iter: respect NITER_ARRAY/FTOL_ARRAY staging (still capped by --max-iter).",
    )
    p.add_argument("--no-solve", action="store_true", help="Use initial guess only.")
    p.add_argument("--outdir", default=None, help="Defaults to examples/outputs/showcase/<case>/")
    p.add_argument(
        "--emit-readme-figures",
        action="store_true",
        help="Also copy key plots into docs/_static/figures for README embedding.",
    )
    p.add_argument(
        "--vmec2000-timeout",
        type=float,
        default=60.0,
        help="Timeout (s) for external VMEC2000 trace runs.",
    )
    p.add_argument(
        "--vmec2000-nstep",
        type=int,
        default=1,
        help="Override VMEC2000 NSTEP (printout cadence) for trace plots.",
    )
    p.add_argument(
        "--no-vmec2000-trace",
        action="store_true",
        help="Skip external VMEC2000 trace runs (residual plot uses final fsq only).",
    )
    args = p.parse_args()

    examples_dir = Path(__file__).resolve().parent
    data_dir = examples_dir / "data"

    if args.suite:
        cases = ["circular_tokamak", "shaped_tokamak_pressure", "solovev"]
        out_root = Path(args.outdir) if args.outdir else (examples_dir / "outputs" / "showcase" / "axisym_suite")
    else:
        cases = [str(args.case)]
        out_root = Path(args.outdir) if args.outdir else (examples_dir / "outputs" / "showcase" / str(args.case))

    out_root.mkdir(parents=True, exist_ok=True)

    suite_rows = []

    for case in cases:
        input_path = data_dir / f"input.{case}"
        ref_wout_path = data_dir / f"wout_{case}_reference.nc"
        outdir = out_root / case if args.suite else out_root
        outdir.mkdir(parents=True, exist_ok=True)

        cfg, indata = vj.load_input(input_path)
        run = vj.run_fixed_boundary(
            input_path,
            solver=str(args.solver),
            max_iter=int(args.max_iter),
            multigrid_use_input_niter=bool(args.use_input_niter),
            use_initial_guess=bool(args.no_solve),
            verbose=bool(args.verbose),
        )
        out_wout_path = outdir / f"wout_{case}_vmec_jax.nc"
        wout_new = vj.write_wout_from_fixed_boundary_run(out_wout_path, run, include_fsq=True)

        wout_ref = vj.read_wout(ref_wout_path) if ref_wout_path.exists() else None
        print(f"[vmec_jax] wrote wout: {out_wout_path}")
        if wout_ref is not None:
            fsq_ref = float(wout_ref.fsqr + wout_ref.fsqz + wout_ref.fsql)
            fsq_new = float(wout_new.fsqr + wout_new.fsqz + wout_new.fsql)
            suite_rows.append((case, fsq_ref, fsq_new))
            print(f"[vmec_jax] fsq_total: ref={fsq_ref:.3e} new={fsq_new:.3e}")

        vmec2000_trace = None
        if (not args.no_vmec2000_trace) and (find_vmec2000_exec() is not None):
            try:
                exec_res = run_xvmec2000(
                    input_path=input_path,
                    timeout_s=float(args.vmec2000_timeout),
                    indata_updates={"NSTEP": str(int(args.vmec2000_nstep))},
                )
                vmec2000_trace = threed1_fsq_total(flatten_threed1(exec_res.stages))
                if vmec2000_trace.size > 0:
                    print(f"[vmec2000] trace entries: {vmec2000_trace.size}")
            except Exception as exc:
                print(f"[vmec2000] trace failed: {exc}")

        _write_plots(
            outdir=outdir,
            run=run,
            wout_new=wout_new,
            wout_ref=wout_ref,
            indata=indata,
            vmec2000_fsq_total=vmec2000_trace,
        )
        print(f"[vmec_jax] wrote plots under: {outdir}")

        if args.emit_readme_figures and not args.suite:
            root = examples_dir.parent
            fig_dir = root / "docs" / "_static" / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            (fig_dir / f"showcase_{case}_surfaces.png").write_bytes((outdir / "surfaces_nested_phi0.png").read_bytes())
            (fig_dir / f"showcase_{case}_bmag_lcfs.png").write_bytes((outdir / "bmag_lcfs.png").read_bytes())
            (fig_dir / f"showcase_{case}_lcfs_3d_bmag.png").write_bytes((outdir / "lcfs_3d_bmag.png").read_bytes())
            residual_path = outdir / "residual_trace.png"
            if residual_path.exists():
                (fig_dir / f"showcase_{case}_residual.png").write_bytes(residual_path.read_bytes())
            print(f"[vmec_jax] updated README figures under: {fig_dir}")

    if args.suite and suite_rows:
        print("[vmec_jax] axisymmetric suite summary (fsq_total)")
        for case, fsq_ref, fsq_new in suite_rows:
            print(f"  {case:24s} ref={fsq_ref:.3e} new={fsq_new:.3e}")


if __name__ == "__main__":
    main()

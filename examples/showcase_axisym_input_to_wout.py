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

import vmec_jax as vj
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

    # Keep showcase plots reasonably fast and match vmecPlot2 conventions.
    theta = vj.closed_theta_grid(128)
    phi = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    ns_new = int(wout_new.ns)
    s_index_lcfs_new = ns_new - 1
    s_index_lcfs_ref = int(wout_ref.ns) - 1 if wout_ref is not None else None
    s_fracs = np.linspace(0.0, 1.0, 9)

    # 1) Nested flux surfaces (phi=0) from reference and new wout.
    fig, ax = plt.subplots(1, 2 if wout_ref is not None else 1, figsize=(10, 4), constrained_layout=True)
    ax = np.atleast_1d(ax)
    for sf in s_fracs:
        si_new = int(round(float(sf) * float(max(ns_new - 1, 0))))
        # Use vmecPlot2-style toroidal slices for surfaces.
        _, _zeta_surf, Rn, Zn = vj.vmecplot2_surface_grid(wout_new, s_index=si_new)
        ax[-1].plot(Rn[:, 0], Zn[:, 0], lw=1.2)
        if wout_ref is not None:
            ns_ref = int(wout_ref.ns)
            si_ref = int(round(float(sf) * float(max(ns_ref - 1, 0))))
            _, _zeta_surf, Rr, Zr = vj.vmecplot2_surface_grid(wout_ref, s_index=si_ref)
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

    # 2) |B| on LCFS using vmecPlot2-compatible grids.
    theta_b, zeta_b, B_new = vj.vmecplot2_bmag_grid(wout_new, s_index=s_index_lcfs_new)
    B_ref = None
    if wout_ref is not None:
        _, _, B_ref = vj.vmecplot2_bmag_grid(wout_ref, s_index=int(s_index_lcfs_ref))

    fig, ax = plt.subplots(1, 2 if B_ref is not None else 1, figsize=(10, 4), constrained_layout=True)
    ax = np.atleast_1d(ax)
    zeta2d, theta2d = np.meshgrid(zeta_b, theta_b)
    b_levels = 20
    im_new = ax[-1].contourf(zeta2d, theta2d, B_new, levels=b_levels)
    ax[-1].set_title("vmec_jax |B| (LCFS)")
    im_ref = None
    if B_ref is not None:
        im_ref = ax[0].contourf(zeta2d, theta2d, B_ref, levels=b_levels)
        ax[0].set_title("VMEC2000 |B| (LCFS)")

    def _fieldline(a, iota_val: float) -> None:
        if iota_val > 0:
            a.plot([0, zeta_b.max()], [0, zeta_b.max() * iota_val], "k")
        else:
            a.plot([0, zeta_b.max()], [-zeta_b.max() * iota_val, 0], "k")

    iota_new = float(np.asarray(wout_new.iotaf)[s_index_lcfs_new]) if hasattr(wout_new, "iotaf") else 0.0
    iota_ref = iota_new
    if wout_ref is not None and hasattr(wout_ref, "iotaf"):
        iota_ref = float(np.asarray(wout_ref.iotaf)[int(s_index_lcfs_ref)])
    for i, a in enumerate(ax):
        a.set_xlabel("zeta")
        a.set_ylabel("theta")
        _fieldline(a, iota_ref if (im_ref is not None and i == 0) else iota_new)
        a.set_xlim([0, 2 * np.pi])
        a.set_ylim([0, 2 * np.pi])
    if im_ref is not None:
        fig.colorbar(im_ref, ax=ax[0], shrink=0.85, pad=0.02)
    fig.colorbar(im_new, ax=ax[-1], shrink=0.85, pad=0.02)
    fig.savefig(outdir / "bmag_lcfs.png", dpi=180)
    plt.close(fig)

    # 3) 3D LCFS surface colored by |B| (vmecPlot2 defaults).
    th3, ph3, Rlcfs, Zlcfs, Blcfs = vj.vmecplot2_lcfs_3d_grid(wout_new, s_index=s_index_lcfs_new)
    X = Rlcfs * np.cos(ph3[None, :])
    Y = Rlcfs * np.sin(ph3[None, :])
    B_ref_3d = None
    if wout_ref is not None:
        _th3r, ph3r, Rr, Zr, Br = vj.vmecplot2_lcfs_3d_grid(wout_ref, s_index=int(s_index_lcfs_ref))
        Xr = Rr * np.cos(ph3r[None, :])
        Yr = Rr * np.sin(ph3r[None, :])
        B_ref_3d = (Xr, Yr, Zr, Br)

    if B_ref_3d is not None:
        fig = plt.figure(figsize=(10, 5))
        fig.patch.set_facecolor("white")
        ax3_ref = fig.add_subplot(1, 2, 1, projection="3d")
        ax3_new = fig.add_subplot(1, 2, 2, projection="3d")
        Xr, Yr, Zr, Br = B_ref_3d
        Br_rescaled = (Br - Br.min()) / max(float(Br.max() - Br.min()), 1e-12)
        ax3_ref.plot_surface(
            Xr,
            Yr,
            Zr,
            facecolors=plt.cm.jet(Br_rescaled),
            rstride=1,
            cstride=1,
            antialiased=False,
            shade=False,
        )
        ax3_ref.auto_scale_xyz([Xr.min(), Xr.max()], [Xr.min(), Xr.max()], [Xr.min(), Xr.max()])
        ax3_ref.set_title("VMEC2000 LCFS |B|")
        Bl_rescaled = (Blcfs - Blcfs.min()) / max(float(Blcfs.max() - Blcfs.min()), 1e-12)
        ax3_new.plot_surface(
            X,
            Y,
            Zlcfs,
            facecolors=plt.cm.jet(Bl_rescaled),
            rstride=1,
            cstride=1,
            antialiased=False,
            shade=False,
        )
        ax3_new.auto_scale_xyz([X.min(), X.max()], [X.min(), X.max()], [X.min(), X.max()])
        ax3_new.set_title("vmec_jax LCFS |B|")
    else:
        fig = plt.figure(figsize=(6, 5))
        fig.patch.set_facecolor("white")
        ax3 = fig.add_subplot(111, projection="3d")
        Bl_rescaled = (Blcfs - Blcfs.min()) / max(float(Blcfs.max() - Blcfs.min()), 1e-12)
        ax3.plot_surface(
            X,
            Y,
            Zlcfs,
            facecolors=plt.cm.jet(Bl_rescaled),
            rstride=1,
            cstride=1,
            antialiased=False,
            shade=False,
        )
        ax3.auto_scale_xyz([X.min(), X.max()], [X.min(), X.max()], [X.min(), X.max()])
    fig.tight_layout()
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
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument(
        "--single-ns",
        type=int,
        default=13,
        help="Run a single-grid parity pass at this NS (overrides NS_ARRAY/NITER_ARRAY).",
    )
    p.add_argument(
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Print per-iteration solver trace (can be very verbose).",
    )
    p.add_argument(
        "--status",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Print status messages about files/plots (disabled to match VMEC stdout).",
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
        cases = ["circular_tokamak", "purely_toroidal_field", "shaped_tokamak_pressure", "solovev"]
        out_root = Path(args.outdir) if args.outdir else (examples_dir / "outputs" / "showcase" / "axisym_suite")
    else:
        cases = [str(args.case)]
        out_root = Path(args.outdir) if args.outdir else (examples_dir / "outputs" / "showcase" / str(args.case))

    out_root.mkdir(parents=True, exist_ok=True)

    suite_rows = []

    def _status(msg: str) -> None:
        if bool(args.status):
            print(msg, flush=True)

    for case in cases:
        input_path = data_dir / f"input.{case}"
        ref_wout_path = data_dir / f"wout_{case}_reference.nc"
        outdir = out_root / case if args.suite else out_root
        outdir.mkdir(parents=True, exist_ok=True)

        cfg, indata = vj.load_input(input_path)
        use_input_niter = bool(args.use_input_niter) if args.single_ns is None else False
        ns_override = int(args.single_ns) if args.single_ns is not None else None
        _status(f"[vmec_jax] running case={case} solver={args.solver} max_iter={args.max_iter}")
        run = vj.run_fixed_boundary(
            input_path,
            solver=str(args.solver),
            max_iter=int(args.max_iter),
            multigrid_use_input_niter=use_input_niter,
            use_initial_guess=bool(args.no_solve),
            verbose=bool(args.verbose),
            ns_override=ns_override,
        )
        out_wout_path = outdir / f"wout_{case}_vmec_jax.nc"
        vj.write_wout_from_fixed_boundary_run(out_wout_path, run, include_fsq=True)
        wout_new = vj.load_wout(out_wout_path)

        wout_ref = vj.load_wout(ref_wout_path) if (args.single_ns is None and ref_wout_path.exists()) else None
        wout_ref_exec = None
        _status(f"[vmec_jax] wrote wout: {out_wout_path}")
        if wout_ref is not None:
            fsq_ref = float(wout_ref.fsqr + wout_ref.fsqz + wout_ref.fsql)
            fsq_new = float(wout_new.fsqr + wout_new.fsqz + wout_new.fsql)
            suite_rows.append((case, fsq_ref, fsq_new))
            _status(f"[vmec_jax] fsq_total: ref={fsq_ref:.3e} new={fsq_new:.3e}")

        vmec2000_trace = None
        if (not args.no_vmec2000_trace) and (find_vmec2000_exec() is not None):
            try:
                _status(f"[vmec2000] running xvmec2000 (timeout={args.vmec2000_timeout}s)")
                indata_updates = {"NSTEP": str(int(args.vmec2000_nstep))}
                if args.single_ns is not None:
                    ftol = float(indata.get_float("FTOL", 1e-10))
                    ns = int(args.single_ns)
                    indata_updates |= {
                        "NS_ARRAY": f"{ns}",
                        "NITER_ARRAY": f"{int(args.max_iter)}",
                        "FTOL_ARRAY": f"{ftol:.16e}",
                        "NITER": f"{int(args.max_iter)}",
                    }
                exec_res = run_xvmec2000(
                    input_path=input_path,
                    timeout_s=float(args.vmec2000_timeout),
                    indata_updates=indata_updates,
                    keep_workdir=True,
                )
                vmec2000_trace = threed1_fsq_total(flatten_threed1(exec_res.stages))
                if vmec2000_trace.size > 0:
                    _status(f"[vmec2000] trace entries: {vmec2000_trace.size}")
                if args.single_ns is not None:
                    wout_candidates = sorted(exec_res.workdir.glob("wout_*"))
                    if wout_candidates:
                        try:
                            wout_ref_exec = vj.load_wout(wout_candidates[0])
                        except Exception as exc:
                            _status(f"[vmec2000] wout load failed: {exc}")
            except Exception as exc:
                _status(f"[vmec2000] trace failed: {exc}")

        _write_plots(
            outdir=outdir,
            run=run,
            wout_new=wout_new,
            wout_ref=wout_ref_exec if wout_ref_exec is not None else wout_ref,
            indata=indata,
            vmec2000_fsq_total=vmec2000_trace,
        )
        _status(f"[vmec_jax] wrote plots under: {outdir}")

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
            _status(f"[vmec_jax] updated README figures under: {fig_dir}")

    if args.suite and suite_rows:
        _status("[vmec_jax] axisymmetric suite summary (fsq_total)")
        for case, fsq_ref, fsq_new in suite_rows:
            _status(f"  {case:24s} ref={fsq_ref:.3e} new={fsq_new:.3e}")


if __name__ == "__main__":
    main()

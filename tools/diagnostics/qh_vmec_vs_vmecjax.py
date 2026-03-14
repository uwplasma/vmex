"""Side-by-side VMEC2000 vs vmec_jax visualization for fixed-boundary cases.

This script:
1) Loads a VMEC2000 reference `wout`.
2) Uses either vmec_jax's solver/initial guess or the VMEC2000 wout coefficients.
3) Generates side-by-side figures for cross-sections (single phi plane, nested
   surfaces), 3D surface, |B| on LCFS, and iota profiles.

The vmec_jax side reflects the *current* solver capability unless
``--use-wout-state`` is enabled. For README-quality optimized comparisons, use
``--solve --solver vmec2000_iter --solver-mode accelerated
--cli-fixed-boundary-mode``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from within examples/ without installing.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

from vmec_jax.modes import vmec_mode_table
from vmec_jax.plotting import (
    axis_rz_from_state_physical,
    axis_rz_from_wout_physical,
    bmag_from_state_physical,
    bmag_from_state_vmec_realspace,
    bmag_from_wout_physical,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
    surface_rz_from_state_physical,
    surface_rz_from_wout_physical,
)
from vmec_jax.driver import run_fixed_boundary
from vmec_jax.geom import eval_geom
from vmec_jax.diagnostics import print_jacobian_stats
from vmec_jax.wout import read_wout, wout_minimal_from_fixed_boundary


def _resolve_wout_ref(*, input_path: Path, wout_ref: str) -> Path:
    if wout_ref:
        return Path(wout_ref).expanduser().resolve()
    case = input_path.name.split("input.", 1)[-1] if "input." in input_path.name else input_path.stem
    cand = input_path.parent / f"wout_{case}_reference.nc"
    if not cand.exists():
        cand = input_path.parent / f"wout_{case}.nc"
    return cand


def _load_vmec2000_wout(wout_path: Path):
    if not wout_path.exists():
        raise SystemExit(f"Missing wout reference: {wout_path}")
    return read_wout(wout_path)



def _plot_cross_sections(ax, *, R, Z, Raxis, Zaxis, title: str):
    nsurf = int(R.shape[0])
    colors = cm.viridis(np.linspace(0.15, 0.95, nsurf))
    for j in range(nsurf):
        ax.plot(R[j], Z[j], lw=1.8, color=colors[j])
    ax.plot(Raxis, Zaxis, "x", ms=6, color="black")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)


def _plot_bmag_surface(ax, *, B, theta, phi, title: str):
    phi2d, theta2d = np.meshgrid(phi, theta)
    cf = ax.contourf(phi2d, theta2d, B, levels=20, cmap="viridis")
    ax.set_xlabel("phi")
    ax.set_ylabel("theta")
    ax.set_title(title)
    return cf


def _plot_3d_surface(ax, *, R, Z, B, phi, title: str):
    phi2d, theta2d = np.meshgrid(phi, np.linspace(0, 2 * np.pi, num=R.shape[0]))
    X = R * np.cos(phi2d)
    Y = R * np.sin(phi2d)
    _ = theta2d
    # Robust color normalization avoids a mostly-flat blue surface when a few
    # points have very large |B| due to near-singular initial Jacobians.
    bvals = np.asarray(B, dtype=float)
    if not np.any(np.isfinite(bvals)):
        bvals = np.zeros_like(bvals)
    vmin = float(np.quantile(bvals, 0.01))
    vmax = float(np.quantile(bvals, 0.99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(bvals))
        vmax = float(np.nanmax(bvals))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    colors = cm.viridis(norm(bvals))
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.auto_scale_xyz([X.min(), X.max()], [X.min(), X.max()], [X.min(), X.max()])
    fix_matplotlib_3d(ax)


def _plot_iota(ax, *, s_vmec, iota_vmec, s_jax, iota_jax, label_jax: str):
    ax.plot(s_vmec, iota_vmec, lw=3.6, linestyle="--", label="VMEC2000", zorder=2)
    ax.plot(s_jax, iota_jax, lw=3.6, linestyle="-", label=label_jax, zorder=1)
    ax.set_xlabel("s")
    ax.set_ylabel("iota")
    ax.set_title("iota profile")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)


def _surface_indices(ns: int, n_surfaces: int) -> list[int]:
    if n_surfaces <= 1:
        return [max(ns - 1, 0)]
    idx = np.linspace(1, ns - 1, num=int(n_surfaces))
    idx = np.unique(np.clip(np.round(idx), 1, ns - 1).astype(int))
    return idx.tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "examples/data/input.nfp4_QH_warm_start"),
    )
    p.add_argument(
        "--wout-ref",
        type=str,
        default="",
        help="Path to VMEC2000 reference wout (defaults to wout_<case>[_reference].nc next to input).",
    )
    p.add_argument("--phi", type=float, default=0.0, help="Toroidal angle (radians) for cross-section plot.")
    p.add_argument("--n-surfaces", type=int, default=6, help="Number of flux surfaces in the cross-section plot.")
    p.add_argument("--jax-title", type=str, default="", help="Override title label for the vmec_jax panels.")
    p.add_argument("--prefix", type=str, default="qh", help="Output filename prefix.")
    p.add_argument("--outdir", type=str, default=str(REPO_ROOT / "docs/_static/figures"))
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--step-size", type=float, default=None)
    p.add_argument(
        "--solver",
        type=str,
        default="vmec2000_iter",
        help="gd, lbfgs, vmec_lbfgs, vmec_gn, or vmec2000_iter",
    )
    p.add_argument(
        "--solver-mode",
        type=str,
        default="accelerated",
        help="run_fixed_boundary solver_mode to use when --solve is active.",
    )
    p.add_argument(
        "--cli-fixed-boundary-mode",
        action="store_true",
        help="Use the optimized CLI-style fixed-boundary controller when solving.",
    )
    p.add_argument("--solve", action="store_true", help="Run the vmec_jax fixed-boundary solver (slower).")
    p.add_argument("--no-solve", action="store_true", help="Use the initial guess only (fast).")
    p.add_argument("--use-wout-state", action="store_true", help="Use VMEC2000 wout coefficients for vmec_jax state.")
    p.add_argument("--no-wout-state", action="store_true", help="Use vmec_jax solver/initial-guess state.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    wout_ref_path = _resolve_wout_ref(input_path=input_path, wout_ref=args.wout_ref)
    # VMEC2000 reference
    wout = _load_vmec2000_wout(wout_ref_path)
    prefix = str(args.prefix)

    # vmec_jax current output
    use_initial_guess = not bool(args.solve) or bool(args.no_solve)
    if bool(args.no_wout_state):
        use_wout_state = False
    elif bool(args.use_wout_state):
        use_wout_state = True
    else:
        use_wout_state = not bool(args.solve)

    if not use_initial_guess and not use_wout_state:
        print(
            "[vmec_jax] note: running a fresh solve from the input file; "
            "use --no-solve for a geometry-only visualization baseline."
        )

    step_size = args.step_size
    if step_size is None:
        step_size = 1.0 if str(args.solver).lower().endswith("_iter") else 1e-5

    if use_wout_state:
        from vmec_jax.config import load_config, VMECConfig
        from vmec_jax.modes import default_grid_sizes
        from vmec_jax.static import build_static
        from vmec_jax.wout import state_from_wout
        from vmec_jax.field import lamscale_from_phips

        cfg_in, indata = load_config(Path(args.input))
        ntheta, nzeta = default_grid_sizes(mpol=int(wout.mpol), ntor=int(wout.ntor), ntheta=0, nzeta=0)
        cfg = VMECConfig(
            mpol=int(wout.mpol),
            ntor=int(wout.ntor),
            ns=int(wout.ns),
            nfp=int(wout.nfp),
            lasym=bool(wout.lasym),
            lthreed=bool(int(wout.ntor) > 0),
            lconm1=bool(cfg_in.lconm1),
            ntheta=int(ntheta),
            nzeta=int(nzeta),
        )
        static = build_static(cfg)
        state = state_from_wout(wout)
        signgs = int(wout.signgs)
        flux_lamscale = float(np.asarray(lamscale_from_phips(wout.phips, static.s)))
        run = None
    else:
        run_kwargs = dict(
            solver=str(args.solver),
            step_size=step_size,
            use_initial_guess=use_initial_guess,
            solver_mode=str(args.solver_mode),
            cli_fixed_boundary_mode=bool(args.cli_fixed_boundary_mode),
        )
        if args.max_iter is not None:
            run_kwargs["max_iter"] = int(args.max_iter)
        run = run_fixed_boundary(Path(args.input), **run_kwargs)
        state = run.state
        static = run.static
        indata = run.indata
        signgs = run.signgs
        flux_lamscale = float(np.asarray(run.flux.lamscale))

    modes = vmec_mode_table(static.cfg.mpol, static.cfg.ntor)

    # --- Cross sections ---
    theta_cs = closed_theta_grid(200)
    phi_slices = np.asarray([float(args.phi)])
    idx_list = _surface_indices(int(wout.ns), int(args.n_surfaces))

    R_vmec_list = []
    Z_vmec_list = []
    for idx in idx_list:
        R_vm, Z_vm = surface_rz_from_wout_physical(wout, theta=theta_cs, phi=phi_slices, s_index=int(idx))
        R_vmec_list.append(R_vm[:, 0])
        Z_vmec_list.append(Z_vm[:, 0])
    R_vmec = np.stack(R_vmec_list, axis=0)
    Z_vmec = np.stack(Z_vmec_list, axis=0)
    Raxis_vmec, Zaxis_vmec = axis_rz_from_wout_physical(wout, phi=phi_slices)

    R_jax_list = []
    Z_jax_list = []
    for idx in idx_list:
        R_j, Z_j = surface_rz_from_state_physical(
            state, modes, theta=theta_cs, phi=phi_slices, s_index=int(idx), nfp=int(static.cfg.nfp)
        )
        R_jax_list.append(R_j[:, 0])
        Z_jax_list.append(Z_j[:, 0])
    R_jax = np.stack(R_jax_list, axis=0)
    Z_jax = np.stack(Z_jax_list, axis=0)
    Raxis_jax, Zaxis_jax = axis_rz_from_state_physical(state, modes, phi=phi_slices, nfp=int(static.cfg.nfp))

    if use_wout_state:
        jax_title_default = "vmec_jax (from wout)"
    else:
        jax_title_default = "vmec_jax (initial guess)" if use_initial_guess else "vmec_jax (solver)"
    jax_title = args.jax_title.strip() or jax_title_default

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    _plot_cross_sections(
        axes[0],
        R=R_vmec,
        Z=Z_vmec,
        Raxis=float(np.asarray(Raxis_vmec)[0]),
        Zaxis=float(np.asarray(Zaxis_vmec)[0]),
        title="VMEC2000",
    )
    _plot_cross_sections(
        axes[1],
        R=R_jax,
        Z=Z_jax,
        Raxis=float(np.asarray(Raxis_jax)[0]),
        Zaxis=float(np.asarray(Zaxis_jax)[0]),
        title=jax_title,
    )
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_compare_cross_sections.png", dpi=220)
    plt.close(fig)

    # --- B magnitude on LCFS ---
    theta_b = closed_theta_grid(30)
    phi_b = np.linspace(0.0, 2.0 * np.pi, num=65, endpoint=True)
    B_vmec = bmag_from_wout_physical(wout, theta=theta_b, phi=phi_b, s_index=int(wout.ns) - 1)
    sqrtg_floor = None
    if use_initial_guess and not use_wout_state:
        geom_init = eval_geom(state, static)
        abs_sg = np.abs(np.asarray(geom_init.sqrtg))
        floor = max(1e-3, 0.5 * float(np.median(abs_sg)))
        sqrtg_floor = floor
        print("[vmec_jax] initial guess Jacobian stats:")
        print_jacobian_stats(geom_init.sqrtg, indent="  ")
        print(f"[vmec_jax] initial guess sqrtg_floor={floor:.3e}")

    B_jax = bmag_from_state_physical(
        state,
        static,
        indata,
        theta=theta_b,
        phi=phi_b,
        s_index=int(static.cfg.ns) - 1,
        signgs=int(signgs),
        phipf=np.asarray(wout.phipf) if use_wout_state else np.asarray(run.flux.phipf),
        chipf=np.asarray(wout.chipf) if use_wout_state else np.asarray(run.flux.chipf),
        lamscale=float(np.asarray(flux_lamscale)),
        flux_is_internal=not use_wout_state,
        sqrtg_floor=sqrtg_floor,
    )
    B_jax_vmec = bmag_from_state_vmec_realspace(
        state,
        static,
        indata,
        s_index=int(static.cfg.ns) - 1,
        signgs=int(signgs),
        phipf=np.asarray(wout.phipf) if use_wout_state else np.asarray(run.flux.phipf),
        chipf=np.asarray(wout.chipf) if use_wout_state else np.asarray(run.flux.chipf),
        lamscale=float(np.asarray(flux_lamscale)),
        flux_is_internal=not use_wout_state,
        sqrtg_floor=sqrtg_floor,
    )
    print(
        f"[vmec_jax] B range (vmec_jax VMEC-grid) min={B_jax_vmec.min():.3e} max={B_jax_vmec.max():.3e}"
    )

    print(f"[vmec_jax] B range (VMEC2000) min={B_vmec.min():.3e} max={B_vmec.max():.3e}")
    print(f"[vmec_jax] B range (vmec_jax) min={B_jax.min():.3e} max={B_jax.max():.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cf0 = _plot_bmag_surface(axes[0], B=B_vmec, theta=theta_b, phi=phi_b, title="VMEC2000")
    cf1 = _plot_bmag_surface(axes[1], B=B_jax, theta=theta_b, phi=phi_b, title=jax_title)
    fig.colorbar(cf0, ax=axes[0], shrink=0.85, label="|B|")
    fig.colorbar(cf1, ax=axes[1], shrink=0.85, label="|B|")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_compare_bmag_surface.png", dpi=220)
    plt.close(fig)

    # --- Iota profiles ---
    prof_vmec = profiles_from_wout(wout)
    s_vmec = prof_vmec["s"]
    iota_vmec = prof_vmec["iotaf"]

    if use_wout_state:
        wout_jax_like = wout
        s_jax = s_vmec
        iota_jax = iota_vmec
    else:
        wout_jax_like = wout_minimal_from_fixed_boundary(
            path="dummy",
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
            fsqr=0.0,
            fsqz=0.0,
            fsql=0.0,
        )
        prof_jax = profiles_from_wout(wout_jax_like)
        s_jax = prof_jax["s"]
        iota_jax = prof_jax["iotaf"]

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.6))
    _plot_iota(
        ax,
        s_vmec=s_vmec,
        iota_vmec=iota_vmec,
        s_jax=s_jax,
        iota_jax=iota_jax,
        label_jax=jax_title,
    )
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_compare_iota.png", dpi=220)
    plt.close(fig)

    # --- 3D LCFS ---
    theta_3d = closed_theta_grid(80)
    nzeta = int(150 * int(wout.nfp))
    phi_3d = np.linspace(0.0, 2.0 * np.pi, num=nzeta, endpoint=True)

    R_vmec_3d, Z_vmec_3d = surface_rz_from_wout_physical(
        wout, theta=theta_3d, phi=phi_3d, s_index=int(wout.ns) - 1
    )
    B_vmec_3d = bmag_from_wout_physical(wout, theta=theta_3d, phi=phi_3d, s_index=int(wout.ns) - 1)

    R_jax_3d, Z_jax_3d = surface_rz_from_state_physical(
        state, modes, theta=theta_3d, phi=phi_3d, s_index=int(static.cfg.ns) - 1, nfp=int(static.cfg.nfp)
    )
    B_jax_3d = bmag_from_state_physical(
        state,
        static,
        indata,
        theta=theta_3d,
        phi=phi_3d,
        s_index=int(static.cfg.ns) - 1,
        signgs=int(signgs),
        phipf=np.asarray(wout.phipf) if use_wout_state else np.asarray(run.flux.phipf),
        chipf=np.asarray(wout.chipf) if use_wout_state else np.asarray(run.flux.chipf),
        lamscale=float(np.asarray(flux_lamscale)),
        flux_is_internal=not use_wout_state,
        sqrtg_floor=sqrtg_floor,
    )
    print(f"[vmec_jax] B range (vmec_jax 3D plot) min={B_jax_3d.min():.3e} max={B_jax_3d.max():.3e}")

    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_3d_surface(ax0, R=R_vmec_3d, Z=Z_vmec_3d, B=B_vmec_3d, phi=phi_3d, title="VMEC2000")
    _plot_3d_surface(ax1, R=R_jax_3d, Z=Z_jax_3d, B=B_jax_3d, phi=phi_3d, title=jax_title)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_compare_3d.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

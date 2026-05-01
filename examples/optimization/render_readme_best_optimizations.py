#!/usr/bin/env python
"""Render the compact README panel of best symmetric QA/QH/QP/QI runs.

The full optimization matrix lives in the documentation.  This script keeps the
README focused on one representative CPU, stellarator-symmetric result for each
target and evaluates the final |B| contours in Boozer coordinates through
``booz_xform_jax``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vmec_jax.plotting import fix_matplotlib_3d, prepare_matplotlib_3d, vmecplot2_lcfs_3d_grid
from vmec_jax.wout import read_wout


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SWEEP_ROOT = SCRIPT_DIR / "results" / "qs_ess_sweep"
FIGURE_DIR = REPO_ROOT / "docs" / "_static" / "figures"
SUMMARY_CSV = FIGURE_DIR / "qs_ess_summary_all.csv"
OUT_PNG = FIGURE_DIR / "readme_best_optimizations.png"
OUT_PDF = FIGURE_DIR / "readme_best_optimizations.pdf"
OUT_CSV = FIGURE_DIR / "readme_best_optimizations.csv"

PROBLEMS = ("qa", "qh", "qp", "qi")
PROBLEM_TITLES = {
    "qa": "QA",
    "qh": "QH",
    "qp": "QP",
    "qi": "QI",
}


@dataclass(frozen=True)
class BestRun:
    problem: str
    policy: str
    max_mode: int
    use_ess: bool
    objective_final: float
    aspect_final: float
    iota_final: float
    total_wall_time_s: float
    output_dir: Path


def _read_summary_rows() -> list[dict[str, str]]:
    with SUMMARY_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def _bool_value(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _path_from_summary(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _best_runs() -> list[BestRun]:
    rows = [
        row
        for row in _read_summary_rows()
        if row.get("backend", "").lower() == "cpu"
        and not _bool_value(row.get("stellarator_asymmetric"))
        and _bool_value(row.get("success"))
        and not _bool_value(row.get("crashed"))
        and row.get("output_dir")
    ]
    best: list[BestRun] = []
    for problem in PROBLEMS:
        candidates = [row for row in rows if row.get("problem") == problem]
        if not candidates:
            raise RuntimeError(f"No successful CPU symmetric rows found for {problem!r}")
        row = min(candidates, key=lambda item: float(item["objective_final"]))
        best.append(
            BestRun(
                problem=problem,
                policy=row["policy"],
                max_mode=int(row["max_mode"]),
                use_ess=_bool_value(row["use_ess"]),
                objective_final=float(row["objective_final"]),
                aspect_final=float(row["aspect_final"]),
                iota_final=float(row["iota_final"]),
                total_wall_time_s=float(row["total_wall_time_s"]),
                output_dir=_path_from_summary(row["output_dir"]),
            )
        )
    return best


def _lcfs_xyz(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return R * np.cos(phi[None, :]), R * np.sin(phi[None, :]), Z


def _plot_lcfs(ax, wout, title: str) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    theta, phi, R, Z, B = vmecplot2_lcfs_3d_grid(
        wout,
        s_index=int(wout.ns) - 1,
        ntheta=44,
        nzeta=max(80, 34 * int(wout.nfp)),
    )
    del theta
    X, Y, Zp = _lcfs_xyz(R, Z, phi)
    norm = Normalize(vmin=float(np.nanmin(B)), vmax=float(np.nanmax(B)))
    cmap = "viridis"
    colors = ScalarMappable(cmap=cmap, norm=norm).to_rgba(B)
    ax.plot_surface(
        X,
        Y,
        Zp,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("X", labelpad=-8)
    ax.set_ylabel("Y", labelpad=-8)
    ax.set_zlabel("Z", labelpad=-8)
    ax.tick_params(axis="both", which="major", labelsize=7, pad=-2)
    fix_matplotlib_3d(ax)
    ax.figure.colorbar(
        ScalarMappable(cmap=cmap, norm=norm),
        ax=ax,
        fraction=0.035,
        pad=0.01,
        shrink=0.62,
        label="|B|",
    )


def _plot_history(ax, run: BestRun) -> None:
    with (run.output_dir / "history.json").open() as f:
        data = json.load(f)
    history = data.get("history", [])
    if not history:
        raise RuntimeError(f"Missing history entries in {run.output_dir / 'history.json'}")
    wall_min = np.asarray([float(item.get("wall_time_s", 0.0)) / 60.0 for item in history])
    objective = np.asarray([float(item.get("objective", item.get("cost", np.nan))) for item in history])
    ax.plot(wall_min, objective, color="#1f4e79", linewidth=1.8)
    ax.scatter(wall_min[-1], objective[-1], s=18, color="#d95f02", zorder=3)
    for boundary in data.get("stage_boundaries", []) or []:
        try:
            idx = int(boundary)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(wall_min):
            ax.axvline(wall_min[idx], color="0.65", linestyle=":", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("Wall time (min)")
    ax.set_ylabel("Total objective")
    ax.set_title("Objective history", fontsize=9, pad=4)
    ax.grid(True, alpha=0.22, linestyle=":")


def _booz_xform_on_outer_surface(wout_path: Path):
    try:
        from booz_xform_jax import Booz_xform
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "render_readme_best_optimizations.py requires booz_xform_jax. "
            "Install it with `pip install .[qi]` from the repository root."
        ) from exc

    bx = Booz_xform(verbose=0)
    bx.read_wout(str(wout_path))
    bx.compute_surfs = [int(bx.ns_in) - 1]
    bx.mboz = int(bx.mpol)
    bx.nboz = int(bx.ntor)
    bx.run()
    return bx


def _booz_bmag_grid(bx, *, ntheta: int = 128, nphi: int = 192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi / float(bx.nfp), nphi)
    phi2d, theta2d = np.meshgrid(phi, theta)
    xm = np.asarray(bx.xm_b, dtype=float)
    xn = np.asarray(bx.xn_b, dtype=float)
    bmnc = np.asarray(bx.bmnc_b, dtype=float)[:, 0]
    B = np.tensordot(
        bmnc,
        np.cos(xm[:, None, None] * theta2d[None, :, :] - xn[:, None, None] * phi2d[None, :, :]),
        axes=(0, 0),
    )
    if bool(bx.asym) and bx.bmns_b is not None:
        bmns = np.asarray(bx.bmns_b, dtype=float)[:, 0]
        B = B + np.tensordot(
            bmns,
            np.sin(xm[:, None, None] * theta2d[None, :, :] - xn[:, None, None] * phi2d[None, :, :]),
            axes=(0, 0),
        )
    return theta, phi, np.asarray(B)


def _plot_boozer_bmag(ax, run: BestRun) -> None:
    bx = _booz_xform_on_outer_surface(run.output_dir / "wout_final.nc")
    theta, phi, B = _booz_bmag_grid(bx)
    PHI, THETA = np.meshgrid(phi, theta)
    vmin = float(np.nanmin(B))
    vmax = float(np.nanmax(B))
    if vmax <= vmin:
        pad = max(abs(vmin), 1.0) * 1e-12
        vmin -= pad
        vmax += pad
    levels = np.linspace(vmin, vmax, 24)
    cs = ax.contour(PHI, THETA, B, levels=levels, cmap="viridis", linewidths=0.9)
    ax.set_title(r"Final $|B|(\theta_B,\phi_B)$", fontsize=9, pad=4)
    ax.set_xlabel(r"Boozer $\phi_B$ (one field period)")
    ax.set_ylabel(r"Boozer $\theta_B$")
    ax.set_xlim(0.0, 2.0 * np.pi / float(bx.nfp))
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_yticks([0, np.pi, 2 * np.pi])
    ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.grid(True, alpha=0.18, linestyle=":")
    ax.figure.colorbar(cs, ax=ax, fraction=0.046, pad=0.018, label="|B|")


def _write_readme_summary(runs: list[BestRun]) -> None:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "problem",
                "policy",
                "max_mode",
                "ess",
                "objective_final",
                "aspect_final",
                "iota_final",
                "cpu_wall_time_min",
                "output_dir",
            ]
        )
        for run in runs:
            writer.writerow(
                [
                    run.problem,
                    run.policy,
                    run.max_mode,
                    "yes" if run.use_ess else "no",
                    f"{run.objective_final:.16e}",
                    f"{run.aspect_final:.16e}",
                    f"{run.iota_final:.16e}",
                    f"{run.total_wall_time_s / 60.0:.6f}",
                    str(run.output_dir),
                ]
            )


def _run_title(run: BestRun) -> str:
    return (
        f"{PROBLEM_TITLES[run.problem]} best symmetric CPU run: "
        f"{run.policy}, max_mode={run.max_mode}, "
        f"{'ESS' if run.use_ess else 'no ESS'}, "
        f"J={run.objective_final:.2e}, "
        f"A={run.aspect_final:.3f}, "
        f"iota={run.iota_final:.4f}, "
        f"{run.total_wall_time_s / 60.0:.1f} min"
    )


def _render_single_run(run: BestRun, out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(18, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=(1.05, 1.05, 1.0, 1.05))
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    wout_initial = read_wout(run.output_dir / "wout_initial.nc")
    wout_final = read_wout(run.output_dir / "wout_final.nc")
    _plot_lcfs(ax0, wout_initial, "Initial LCFS")
    _plot_lcfs(ax1, wout_final, "Final LCFS")
    _plot_history(ax2, run)
    _plot_boozer_bmag(ax3, run)
    fig.suptitle(_run_title(run), fontsize=13, x=0.01, y=1.02, ha="left")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _render_stacked_overview(runs: list[BestRun], image_paths: list[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.image as mpimg

    images = [mpimg.imread(path) for path in image_paths]
    fig, axes = plt.subplots(len(images), 1, figsize=(18, 4.9 * len(images)))
    axes = np.asarray(axes).ravel()
    for ax, image, run in zip(axes, images, runs):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(PROBLEM_TITLES[run.problem], loc="left", fontsize=12, pad=4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    runs = _best_runs()
    _write_readme_summary(runs)
    image_paths: list[Path] = []
    for run in runs:
        out_png = FIGURE_DIR / f"readme_best_optimization_{run.problem}.png"
        out_pdf = FIGURE_DIR / f"readme_best_optimization_{run.problem}.pdf"
        _render_single_run(run, out_png, out_pdf)
        image_paths.append(out_png)
        print(f"Wrote {out_png}")
        print(f"Wrote {out_pdf}")
    _render_stacked_overview(runs, image_paths)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

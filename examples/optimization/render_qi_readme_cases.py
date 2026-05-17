#!/usr/bin/env python
"""Render README/docs coverage for the two promoted QI_optimization inputs.

This renderer intentionally consumes existing reviewed outputs instead of
launching new optimization jobs.  The Boozer |B| panels use line contours only.
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
FIGURE_DIR = REPO_ROOT / "docs" / "_static" / "figures"
OUT_PNG = FIGURE_DIR / "readme_qi_optimization_cases.png"
OUT_CSV = FIGURE_DIR / "readme_qi_optimization_cases.csv"


@dataclass(frozen=True)
class QICase:
    label: str
    input_file: Path
    output_dir: Path
    initial_wout: Path
    note: str


CASES = (
    QICase(
        label="NFP=2 bundled QI",
        input_file=REPO_ROOT / "examples" / "data" / "input.nfp2_QI",
        output_dir=REPO_ROOT / "results" / "qi_opt" / "ess" / "nfp2_qi",
        initial_wout=REPO_ROOT / "results" / "qi_opt" / "ess" / "nfp2_qi" / "wout_initial.nc",
        note="default mirror-aware QI lane",
    ),
    QICase(
        label="NFP=3 seed 3127",
        input_file=REPO_ROOT / "examples" / "data" / "input.QI_stel_seed_3127",
        output_dir=REPO_ROOT / "results" / "qi_opt" / "ess" / "qi_stel_seed_3127_current_public_final",
        initial_wout=REPO_ROOT / "examples" / "data" / "wout_QI_stel_seed_3127.nc",
        note="curated reference-family baseline",
    ),
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _case_record(case: QICase) -> dict[str, str | float]:
    diagnostics = _load_json(case.output_dir / "diagnostics.json")
    history = _load_json(case.output_dir / "history.json")
    final_wout = case.output_dir / "wout_final.nc"
    for path in (case.input_file, case.initial_wout, final_wout):
        if not path.exists():
            raise FileNotFoundError(path)
    return {
        "case": case.label,
        "input_file": str(case.input_file.relative_to(REPO_ROOT)),
        "output_dir": str(case.output_dir.relative_to(REPO_ROOT)),
        "note": case.note,
        "objective_final": float(history["objective_final"]),
        "qi_smooth_total": float(diagnostics.get("qi_smooth_total", diagnostics["qi_raw_total"])),
        "qi_legacy_total": float(diagnostics["qi_legacy_total"]),
        "qi_mirror_ratio_max": float(diagnostics["qi_mirror_ratio_max"]),
        "qi_mirror_ratio_target": float(diagnostics["qi_mirror_ratio_target"]),
        "qi_max_elongation": float(diagnostics["qi_max_elongation"]),
        "qi_elongation_target": float(diagnostics["qi_elongation_target"]),
        "aspect": float(diagnostics["aspect"]),
        "target_aspect": float(diagnostics["target_aspect"]),
        "mean_iota": float(diagnostics["mean_iota"]),
        "qi_nfp": int(diagnostics["qi_nfp"]),
        "cpu_time_min": float(history["total_wall_time_s"]) / 60.0,
        "final_wout": str(final_wout.relative_to(REPO_ROOT)),
    }


def _write_csv(records: list[dict[str, str | float]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "input_file",
        "output_dir",
        "note",
        "objective_final",
        "qi_smooth_total",
        "qi_legacy_total",
        "qi_mirror_ratio_max",
        "qi_mirror_ratio_target",
        "qi_max_elongation",
        "qi_elongation_target",
        "aspect",
        "target_aspect",
        "mean_iota",
        "qi_nfp",
        "cpu_time_min",
        "final_wout",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _lcfs_xyz(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return R * np.cos(phi[None, :]), R * np.sin(phi[None, :]), Z


def _plot_lcfs(ax, wout_path: Path, title: str) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    wout = read_wout(wout_path)
    _theta, phi, R, Z, B = vmecplot2_lcfs_3d_grid(
        wout,
        s_index=int(wout.ns) - 1,
        ntheta=40,
        nzeta=max(72, 30 * int(wout.nfp)),
    )
    X, Y, Zp = _lcfs_xyz(R, Z, phi)
    norm = Normalize(vmin=float(np.nanmin(B)), vmax=float(np.nanmax(B)))
    colors = ScalarMappable(cmap="viridis", norm=norm).to_rgba(B)
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
    ax.set_title(title, fontsize=8, pad=4)
    ax.set_xlabel("X", labelpad=-8)
    ax.set_ylabel("Y", labelpad=-8)
    ax.set_zlabel("Z", labelpad=-8)
    ax.tick_params(axis="both", which="major", labelsize=6, pad=-2)
    fix_matplotlib_3d(ax)


def _plot_history(ax, history_path: Path) -> None:
    history = _load_json(history_path)
    entries = history.get("history", [])
    if not entries:
        raise RuntimeError(f"Missing history entries in {history_path}")
    wall_min = np.asarray([float(item.get("wall_time_s", 0.0)) / 60.0 for item in entries], dtype=float)
    objective = np.minimum.accumulate(
        np.asarray([max(float(item.get("objective", item.get("cost", np.nan))), 1.0e-16) for item in entries])
    )
    ax.semilogy(wall_min, objective, color="#1f4e79", linewidth=1.8)
    ax.scatter(wall_min[-1], objective[-1], s=18, color="#d95f02", zorder=3)
    ax.set_title("Objective history", fontsize=8, pad=4)
    ax.set_xlabel("Wall time (min)")
    ax.set_ylabel("Total objective")
    ax.grid(True, alpha=0.22, linestyle=":")


def _booz_xform_on_outer_surface(wout_path: Path):
    try:
        from booz_xform_jax import Booz_xform
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "render_qi_readme_cases.py requires booz_xform_jax. "
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


def _plot_boozer_bmag(ax, final_wout: Path, nfp: int) -> None:
    bx = _booz_xform_on_outer_surface(final_wout)
    theta, phi, B = _booz_bmag_grid(bx)
    PHI, THETA = np.meshgrid(phi, theta)
    vmin = float(np.nanmin(B))
    vmax = float(np.nanmax(B))
    if vmax <= vmin:
        pad = max(abs(vmin), 1.0) * 1.0e-12
        vmin -= pad
        vmax += pad
    levels = np.linspace(vmin, vmax, 24)
    cs = ax.contour(PHI, THETA, B, levels=levels, cmap="viridis", linewidths=0.9)
    ax.set_title(r"Final Boozer $|B|$ line contours", fontsize=8, pad=4)
    ax.set_xlabel(r"$\phi_B$ (one field period)")
    ax.set_ylabel(r"$\theta_B$")
    ax.set_xlim(0.0, 2.0 * np.pi / float(nfp))
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_yticks([0, np.pi, 2 * np.pi])
    ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.grid(True, alpha=0.18, linestyle=":")
    ax.figure.colorbar(cs, ax=ax, fraction=0.046, pad=0.018, label="|B|")


def _row_title(record: dict[str, str | float]) -> str:
    return (
        f"{record['case']} | J={record['objective_final']:.2e}, "
        f"QI={record['qi_legacy_total']:.2e}, smooth={record['qi_smooth_total']:.2e}, "
        f"mirror={record['qi_mirror_ratio_max']:.3f}, "
        f"elong={record['qi_max_elongation']:.2f}, "
        f"A={record['aspect']:.3f}, "
        f"iota={record['mean_iota']:.4f}, "
        f"{record['cpu_time_min']:.1f} CPU min"
    )


def _render(records: list[dict[str, str | float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(17.5, 8.6))
    gs = fig.add_gridspec(
        len(CASES),
        4,
        left=0.045,
        right=0.975,
        bottom=0.06,
        top=0.82,
        wspace=0.35,
        hspace=0.75,
        width_ratios=(1.05, 1.05, 1.0, 1.12),
    )
    fig.suptitle("QI_optimization coverage from bundled QI and seed-3127 inputs", fontsize=13, x=0.02, y=0.985, ha="left")
    for row, (case, record) in enumerate(zip(CASES, records, strict=True)):
        ax0 = fig.add_subplot(gs[row, 0], projection="3d")
        ax1 = fig.add_subplot(gs[row, 1], projection="3d")
        ax2 = fig.add_subplot(gs[row, 2])
        ax3 = fig.add_subplot(gs[row, 3])
        _plot_lcfs(ax0, case.initial_wout, "Raw input LCFS")
        _plot_lcfs(ax1, case.output_dir / "wout_final.nc", "Final LCFS")
        _plot_history(ax2, case.output_dir / "history.json")
        _plot_boozer_bmag(ax3, case.output_dir / "wout_final.nc", int(record["qi_nfp"]))
        title_y = 0.895 - 0.455 * row
        fig.text(0.045, title_y, _row_title(record), fontsize=10, ha="left", va="bottom")
        fig.text(
            0.045,
            title_y - 0.025,
            f"{record['input_file']} -> {record['output_dir']}",
            fontsize=7,
            ha="left",
            va="bottom",
        )
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prepare_matplotlib_3d()
    records = [_case_record(case) for case in CASES]
    _write_csv(records)
    _render(records)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_CSV}")
    for record in records:
        print(
            f"{record['case']}: QI={record['qi_legacy_total']:.6e} "
            f"smooth={record['qi_smooth_total']:.6e} "
            f"mirror={record['qi_mirror_ratio_max']:.6g} "
            f"elong={record['qi_max_elongation']:.6g} "
            f"iota={record['mean_iota']:.6g} "
            f"aspect={record['aspect']:.6g} "
            f"cpu_min={record['cpu_time_min']:.3f}"
        )


if __name__ == "__main__":
    main()

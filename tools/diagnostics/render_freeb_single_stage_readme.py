#!/usr/bin/env python
"""Render README figures for the free-boundary direct-coil provider lane."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "results" / "free_boundary_essos_coils_beta_scan_readme" / "summary.json"
DEFAULT_OUTDIR = REPO_ROOT / "docs" / "_static" / "figures"


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
        }
    )


def _load_runs(summary_path: Path) -> list[dict[str, Any]]:
    data = json.loads(summary_path.read_text())
    runs = list(data.get("runs", []))
    if not runs:
        raise ValueError(f"No runs found in {summary_path}")
    for run in runs:
        fsqr = float(run.get("fsqr") or 0.0)
        fsqz = float(run.get("fsqz") or 0.0)
        fsql = float(run.get("fsql") or 0.0)
        run["fsq_norm"] = float(np.sqrt(fsqr * fsqr + fsqz * fsqz + fsql * fsql))
    return runs


def _runs_by_backend(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["backend"]), []).append(run)
    for backend_runs in grouped.values():
        backend_runs.sort(key=lambda item: float(item["nominal_beta_percent"]))
    return grouped


def _write_csv(runs: list[dict[str, Any]], outdir: Path) -> Path:
    out = outdir / "freeb_single_stage_beta_scan_summary.csv"
    fields = [
        "backend",
        "nominal_beta_percent",
        "wall_s",
        "n_iter",
        "fsqr",
        "fsqz",
        "fsql",
        "fsq_norm",
        "aspect",
        "mean_iota",
        "pressure_scale",
        "max_pressure",
        "wp",
        "wb",
        "beta_proxy",
        "beta_proxy_percent",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (float(r["nominal_beta_percent"]), str(r["backend"]))):
            writer.writerow({field: run.get(field) for field in fields})
    return out


def render_architecture(outdir: Path) -> Path:
    """Render a compact architecture diagram for the README."""

    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.set_axis_off()

    boxes = [
        ("Coil parameters", "Fourier centerlines\nand coil currents", "#e0f2fe"),
        ("JAX Biot-Savart", "Differentiable direct\nexternal field", "#dcfce7"),
        ("Free-boundary VMEC", "Equilibrium solved from\ncoil field, no mgrid write", "#fef3c7"),
        ("Diagnostics", "wout, aspect, iota,\nBoozer/QS metrics", "#ede9fe"),
        ("Optimizer", "Coil-only objective\nupdates", "#fee2e2"),
    ]
    x_positions = np.linspace(0.07, 0.83, len(boxes))
    y = 0.53
    width = 0.15
    height = 0.28
    centers = []
    for x, (title, subtitle, color) in zip(x_positions, boxes, strict=True):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.25,
            facecolor=color,
            edgecolor="#334155",
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.67, title, ha="center", va="center", weight="bold", transform=ax.transAxes)
        ax.text(x + width / 2, y + height * 0.35, subtitle, ha="center", va="center", transform=ax.transAxes)
        centers.append((x + width / 2, y + height / 2))

    for (x0, y0), (x1, y1) in zip(centers[:-1], centers[1:], strict=True):
        ax.add_patch(
            FancyArrowPatch(
                (x0 + width / 2 - 0.005, y0),
                (x1 - width / 2 + 0.005, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#334155",
                transform=ax.transAxes,
            )
        )

    ax.add_patch(
        FancyArrowPatch(
            (centers[-1][0], y - 0.02),
            (centers[0][0], y - 0.02),
            connectionstyle="arc3,rad=-0.33",
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#991b1b",
            transform=ax.transAxes,
        )
    )
    ax.text(0.5, 0.24, "single-stage loop: only coil parameters are independent optimization variables", ha="center", transform=ax.transAxes)

    ax.plot([0.06, 0.94], [0.14, 0.14], color="#64748b", linewidth=0.8, linestyle="--", transform=ax.transAxes)
    ax.text(
        0.5,
        0.06,
        "Phase 1 in this branch: differentiable provider + forward free-boundary solve. "
        "Phase 2: full production custom adjoint through the vacuum/NESTOR solve.",
        ha="center",
        color="#475569",
        fontsize=9.5,
        transform=ax.transAxes,
    )
    fig.suptitle("Direct-coil free-boundary architecture", y=0.96, weight="bold")
    out = outdir / "freeb_single_stage_architecture.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_beta_scan(runs: list[dict[str, Any]], outdir: Path) -> Path:
    grouped = _runs_by_backend(runs)
    colors = {"mgrid": "#0f766e", "direct": "#c2410c"}
    labels = {"mgrid": "ESSOS coils -> mgrid", "direct": "Direct JAX coils"}
    markers = {"mgrid": "o", "direct": "s"}
    metrics = [
        ("pressure_scale", "VMEC PRES_SCALE", "linear"),
        ("beta_proxy_percent", r"$100 W_p/W_B$ (%)", "linear"),
        ("fsq_norm", "residual norm", "log"),
        ("aspect", "aspect ratio", "linear"),
        ("mean_iota", "mean iota", "linear"),
        ("wall_s", "wall time (s)", "linear"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    for ax, (metric, ylabel, scale) in zip(axes.ravel(), metrics, strict=True):
        for backend in ("mgrid", "direct"):
            if backend not in grouped:
                continue
            backend_runs = grouped[backend]
            x = np.array([float(run["nominal_beta_percent"]) for run in backend_runs])
            y = np.array([float(run[metric]) for run in backend_runs])
            ax.plot(
                x,
                y,
                marker=markers[backend],
                linewidth=1.8,
                markersize=5.5,
                color=colors[backend],
                label=labels[backend],
            )
        ax.set_xlabel("nominal beta (%)")
        ax.set_ylabel(ylabel)
        if scale == "log":
            ax.set_yscale("log")
            ax.set_ylim(5.0e-3, 2.0e-2)
        else:
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        if metric == "aspect":
            ax.axhline(6.0, color="#475569", linewidth=1.0, linestyle=":", label="A=6")
            ax.set_ylim(5.98, 6.02)
        if metric == "mean_iota":
            ax.set_ylim(0.385, 0.400)
        if metric == "beta_proxy_percent":
            ax.set_ylim(bottom=0.0)
        ax.margins(x=0.08)
    axes[0, 0].legend(frameon=False, loc="best")
    axes[1, 2].text(
        0.02,
        0.96,
        "First point includes cold-start overhead.",
        transform=axes[1, 2].transAxes,
        ha="left",
        va="top",
        color="#475569",
        fontsize=8.5,
    )
    fig.suptitle("Finite-pressure free-boundary scan from Landreman-Paul QA ESSOS coils", weight="bold")
    out = outdir / "freeb_single_stage_beta_scan.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_provider_parity(runs: list[dict[str, Any]], outdir: Path) -> Path:
    grouped_by_beta: dict[float, dict[str, dict[str, Any]]] = {}
    for run in runs:
        grouped_by_beta.setdefault(float(run["nominal_beta_percent"]), {})[str(run["backend"])] = run
    betas = sorted(beta for beta, row in grouped_by_beta.items() if {"mgrid", "direct"}.issubset(row))
    metrics = [("aspect", "aspect"), ("mean_iota", "mean iota"), ("fsq_norm", "residual norm")]
    floors = []
    rel_diffs: dict[str, list[float]] = {name: [] for name, _ in metrics}
    abs_diffs: dict[str, list[float]] = {name: [] for name, _ in metrics}
    for beta in betas:
        pair = grouped_by_beta[beta]
        ref = pair["mgrid"]
        direct = pair["direct"]
        for name, _label in metrics:
            a = float(ref[name])
            b = float(direct[name])
            abs_diff = abs(a - b)
            rel = abs_diff / max(abs(a), 1.0e-30)
            abs_diffs[name].append(abs_diff)
            rel_diffs[name].append(max(rel, 1.0e-16))
            floors.append(rel == 0.0)

    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    x = np.arange(len(betas))
    width = 0.23
    colors = ["#1d4ed8", "#0f766e", "#b45309"]
    for idx, (name, label) in enumerate(metrics):
        ax.bar(x + (idx - 1) * width, rel_diffs[name], width=width, color=colors[idx], label=label)
    ax.set_yscale("log")
    ax.set_ylim(5.0e-17, 2.0e-12)
    ax.set_xticks(x, [f"{beta:g}" for beta in betas])
    ax.set_xlabel("nominal beta (%)")
    ax.set_ylabel("relative difference: direct coils vs mgrid")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.text(
        0.02,
        0.08,
        "Bars at 1e-16 denote exact agreement to the precision recorded in the JSON summary.",
        transform=ax.transAxes,
        color="#475569",
        fontsize=9,
    )
    max_abs = max(max(values) for values in abs_diffs.values())
    ax.set_title(f"Provider parity in vmec_jax free-boundary solve (max absolute difference {max_abs:.1e})", weight="bold")
    out = outdir / "freeb_single_stage_provider_parity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    _setup_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(args.summary)
    outputs = [
        render_architecture(args.outdir),
        render_beta_scan(runs, args.outdir),
        render_provider_parity(runs, args.outdir),
        _write_csv(runs, args.outdir),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Render the validation-page strong-force comparison from the committed record.

Usage::

    python docs/_static/figures/sources/make_strong_force_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
from _provenance import file_sha256  # noqa: E402

BENCHMARKS = REPO / "benchmarks"
DEFAULT_FIGURE = REPO / "docs" / "_static" / "figures" / "readme_strong_force_comparison.webp"
DEFAULT_METADATA = BENCHMARKS / "strong_force_comparison_m4.json"
DEFAULT_ARTIFACT = BENCHMARKS / "strong_force_cases_m4.json"

SURFACE = "#fcfcfb"
INK = "#161616"
MUTED = "#66645f"
EDGE = "#d5d3ce"
COLORS = {
    "VMEX": "#1674d1",
    "VMEC2000": "#333333",
    "DESC": "#7357b8",
}


def _load(path: Path) -> dict[str, dict[str, dict]]:
    bundle = json.loads(path.read_text())
    if bundle["schema"] != "vmex.strong-force-comparison-cases/1":
        raise RuntimeError("unexpected strong-force comparison artifact schema")
    cases = bundle["cases"]
    for case_name, case in cases.items():
        for name, artifact in case["sources"].items():
            if artifact["measurement_dirty"]:
                raise RuntimeError(
                    f"{case_name}/{name} was measured from dirty source"
                )
            external = artifact.get("external_source")
            if external is not None and not external["success"]:
                raise RuntimeError(f"{case_name}/{name} external solve failed")
        polished = case["sources"].get("VMEX")
        if polished is None or "polish_report" not in polished:
            continue
        if not polished["polish_report"]["converged"]:
            raise RuntimeError("VMEX artifact is not independently certified")
        if polished["final_certificate"]["normalized_l2"] > polished["validation_tolerance"]:
            raise RuntimeError("VMEX polished force exceeds its validation gate")
    return cases


def _profile(name: str, artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    profile = (
        artifact["final_certificate"]["radial_profile"]
        if "final_certificate" in artifact
        else artifact["radial_profile"]
    )
    return (
        np.asarray(profile["rho"], dtype=float),
        np.asarray(profile["flux_surface_normalized_l2"], dtype=float),
    )


def _normalized(name: str, artifact: dict) -> float:
    return float(
        artifact["final_certificate"]["normalized_l2"]
        if "final_certificate" in artifact
        else artifact["metrics"]["normalized_l2"]
    )


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": EDGE,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
        }
    )


def _render_row(
    axes,
    artifacts: dict[str, dict],
    *,
    case_label: str,
    letters: tuple[str, str, str],
) -> None:
    artifacts = {name: artifacts[name] for name in COLORS}
    line_styles = {
        "VMEX": "-",
        "VMEC2000": (0, (5, 2)),
        "DESC": "-.",
    }
    ax = axes[0]
    for name, artifact in artifacts.items():
        rho, force = _profile(name, artifact)
        ax.plot(
            rho,
            np.maximum(force, 1.0e-30),
            color=COLORS[name],
            linestyle=line_styles[name],
            linewidth=2.5 if name == "VMEX" else 1.7,
            alpha=1.0 if name in ("VMEX", "DESC") else 0.88,
            label=name,
        )
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(
        r"normalized radius $\rho=\sqrt{s}$,  $s=\psi/\psi_B$"
    )
    ax.set_ylabel(
        "relative force error\n"
        r"$\epsilon_F=2|\mathbf{J}\!\times\!\mathbf{B}-\nabla p|/"
        r"(|\mathbf{J}\!\times\!\mathbf{B}|+|\nabla p|+F_{\rm floor})$"
    )
    ax.legend(loc="best", ncols=2)
    ax.set_title(
        f"({letters[0]}) {case_label}: radial profile",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    ax = axes[1]
    names = list(artifacts)
    values = [_normalized(name, artifacts[name]) for name in names]
    positions = np.arange(len(names))
    bars = ax.bar(
        positions,
        values,
        color=[COLORS[name] for name in names],
        width=0.72,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_yscale("log")
    ax.set_ylim(min(values) * 0.5, max(values) * 2.5)
    ax.set_xticks(positions)
    labels = {
        "VMEC2000": "VMEC\n2000",
    }
    ax.set_xticklabels([labels.get(name, name) for name in names])
    ax.set_ylabel("relative force error, volume L2")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * 1.10,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.set_title(
        f"({letters[1]}) Volume L2 error",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

def render(
    tokamak: dict[str, dict],
    output: Path,
) -> None:
    # Accuracy only, and only rows where certified polishing beats the legacy
    # codes on the independent force certificate (README figure policy). The
    # stellarator row returns when the production polish is tractable in 3D
    # (the captured-constants elimination now in progress); its bundle case
    # stays recorded for that day.
    _style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.9),
        gridspec_kw={"width_ratios": (1.7, 1.0)},
        dpi=180,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.86,
        wspace=0.38,
    )
    fig.suptitle(
        "Force balance across equilibrium solvers",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    _render_row(
        axes,
        tokamak,
        case_label="shaped tokamak, finite pressure",
        letters=("a", "b"),
    )
    fig.text(
        0.985,
        0.022,
        "One shared independent force-balance oracle on each code's exported equilibrium; VMEX is the certified polished result. Case: input.shaped_tokamak_pressure_polished.",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="webp", dpi=180, pil_kwargs={"lossless": True})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    cases = _load(args.artifact)
    render(
        cases["shaped_tokamak_pressure"]["sources"],
        args.output,
    )
    try:
        figure_path = args.output.relative_to(REPO).as_posix()
    except ValueError:
        figure_path = str(args.output)
    metadata = {
        "schema": "vmex.strong-force-readme-figure/4",
        "cases": {
            case: {
                "sources": {
                    name: {
                        "path": f"benchmarks/{args.artifact.name}",
                        "sha256": file_sha256(args.artifact),
                        "normalized_l2": _normalized(name, artifact),
                    }
                    for name, artifact in contents["sources"].items()
                }
            }
            for case, contents in cases.items()
        },
        "figure": figure_path,
        "figure_sha256": file_sha256(args.output),
        "timing_note": (
            "Accuracy only, and only rows where certified polishing beats "
            "the legacy codes on the independent force certificate, per the "
            "README figure policy (accuracy only). "
            "All errors from one shared independent oracle on each code's "
            "exported equilibrium (VMEX: the certified polished state). The "
            "stellarator row returns when the 3-D production polish is "
            "tractable."
        ),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(figure_path)


if __name__ == "__main__":
    main()

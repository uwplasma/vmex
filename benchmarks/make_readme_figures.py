#!/usr/bin/env python3
"""Regenerate the README figures from the benchmark and golden-parity data.

Produces (into ``docs/_static/figures/``):

- ``readme_runtime_compare.png``      — VMEC2000 vs vmec_jax (cold/warm CPU,
  GPU where comparable) vs VMEC++, from ``benchmarks/baseline.json`` and
  ``benchmarks/gpu_baseline.json``.  Run ``benchmarks/run_baseline.py`` first.
- ``readme_parity.png``               — iteration-for-iteration parity table
  vs the golden VMEC2000 fixtures (solves the five cases; needs the golden
  bundle, see ``tests/core_new/conftest.py``).
- ``readme_equilibrium_showcase.png`` — flux surfaces + boundary ``|B|`` of
  the bundled quick-start case (solves it in-process).

Usage:
    python benchmarks/make_readme_figures.py [--only runtime,parity,showcase]
        [--outdir docs/_static/figures]

Figures are written uncompressed; compress before committing:
    magick <fig>.png -colors 255 png8:<fig>.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data"

# --------------------------------------------------------------------------
# Shared style (dataviz conventions: near-white surface, recessive chrome,
# colorblind-safe categorical slots, direct labels over chartjunk)
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"        # vmec_jax warm (the hero series)
BLUE_LIGHT = "#86b6ef"  # vmec_jax cold (same hue, lighter step)
YELLOW = "#eda100"      # VMEC++
VIOLET = "#4a3aa7"      # GPU
RED = "#e34948"
GREEN_TEXT = "#006300"


def apply_style() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.labelsize": 10, "legend.frameon": False, "legend.fontsize": 9,
    })


CASE_LABELS = {
    "solovev": "Solovev",
    "DSHAPE": "DSHAPE",
    "circular_tokamak": "circular tokamak",
    "cth_like_fixed_bdy": "CTH-like",
    "li383_low_res": "LI383 (NCSX)",
    "LandremanPaul2021_QA_lowres": "Landreman-Paul QA",
    "LandremanPaul2021_QH_reactorScale_lowres": "Landreman-Paul QH reactor",
    "nfp4_QH_warm_start": "nfp4 QH",
    "NuhrenbergZille_1988_QHS": "Nuhrenberg-Zille QHS",
    "cth_like_free_bdy_lasym_small": "CTH-like free-bdy (lasym)",
}


# --------------------------------------------------------------------------
# 1. Runtime comparison
# --------------------------------------------------------------------------

def make_runtime_figure(out: Path) -> None:
    base = json.loads((REPO / "benchmarks" / "baseline.json").read_text())
    gpu = json.loads((REPO / "benchmarks" / "gpu_baseline.json").read_text())

    # GPU rows are single-grid first-stage runs; keep only rows whose
    # iteration count matches the CPU baseline (i.e. truly the same solve).
    gpu_rows = {
        case: vals["cuda/core_jit"]
        for case, vals in gpu.get("matrix", {}).items()
        if vals.get("cuda/core_jit", {}).get("ok")
    }

    rows = []
    for key, r in base.items():
        case, grid = key[:-1].split("[")
        v2k, cold = r.get("vmec2000", {}), r.get("vmec_jax_cold", {})
        warm, vpp = r.get("vmec_jax_warm", {}), r.get("vmecpp", {})
        if not (v2k.get("ok") and warm.get("ok")):
            continue
        label = CASE_LABELS.get(case, case)
        if grid == "multigrid":
            label += " (multigrid)"
        gr = None
        if grid == "input" and case in gpu_rows:
            if gpu_rows[case].get("iterations") == v2k.get("iterations"):
                gr = gpu_rows[case]["warm_wall_s"]
        rows.append(dict(label=label, ns=r.get("ns"), v2k=v2k.get("wall_s"),
                         cold=cold.get("wall_s"), warm=warm.get("warm_s"),
                         vpp=vpp.get("wall_s") if vpp.get("ok") else None,
                         gpu=gr))
    rows.sort(key=lambda r: r["v2k"])

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(rows) + 1.7), dpi=160)
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        xs = [v for v in (r["v2k"], r["cold"], r["warm"], r["vpp"], r["gpu"]) if v]
        ax.hlines(y, min(xs), max(xs), color=GRID, lw=1.0, zorder=1)

    mk = dict(s=52, zorder=3, linewidths=1.4, edgecolors=SURFACE)
    ax.scatter([r["v2k"] for r in rows], list(ys), color=INK2,
               label="VMEC2000 (Fortran)", **mk)
    ax.scatter([r["warm"] for r in rows], list(ys), color=BLUE,
               label="vmec_jax warm (in-process)", **mk)
    vpp_pts = [(r["vpp"], y) for y, r in zip(ys, rows) if r["vpp"]]
    ax.scatter([p[0] for p in vpp_pts], [p[1] for p in vpp_pts],
               color=YELLOW, label="VMEC++", **mk)
    gpu_pts = [(r["gpu"], y) for y, r in zip(ys, rows) if r["gpu"]]
    if gpu_pts:
        ax.scatter([p[0] for p in gpu_pts], [p[1] for p in gpu_pts],
                   color=VIOLET, marker="D", s=40, zorder=3, linewidths=1.4,
                   edgecolors=SURFACE, label="vmec_jax warm (GPU)")
    ax.scatter([r["cold"] for r in rows], list(ys), color=BLUE_LIGHT,
               label="vmec_jax cold (fresh CLI process)", **mk)

    xmax = max(max(v for v in (r["v2k"], r["cold"], r["vpp"] or 0, r["gpu"] or 0))
               for r in rows)
    for y, r in zip(ys, rows):
        sp = r["v2k"] / r["warm"]
        ax.annotate(f"{sp:,.0f}x" if sp >= 3 else f"{sp:.1f}x",
                    xy=(xmax * 2.6, y), va="center", ha="left",
                    fontsize=8.5, color=INK2, annotation_clip=False)
    ax.annotate("warm speedup\nvs VMEC2000", xy=(xmax * 2.45, len(rows) + 0.55),
                va="bottom", ha="left", fontsize=8, color=MUTED,
                annotation_clip=False)

    ax.set_xscale("log")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r['label']}  ·  ns={r['ns']}" for r in rows])
    ax.set_xlabel("wall-clock time (s, log scale)")
    ax.set_xlim(right=xmax * 2.4)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Full equilibrium solve: wall-clock time", loc="left",
                 pad=52, fontsize=13, color=INK)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), ncols=2,
              fontsize=8.5, columnspacing=1.4, handletextpad=0.25,
              borderaxespad=0.0, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# 2. Parity table (solves the five golden-fixture cases)
# --------------------------------------------------------------------------

PARITY_CASES = {
    # case -> (multigrid?, README description)
    "solovev": (False, "2D analytic tokamak"),
    "DSHAPE": (True, "D-shaped tokamak, multigrid"),
    "circular_tokamak": (True, "high-ns tokamak, multigrid"),
    "li383_low_res": (False, "3D stellarator, nfp=3"),
    "nfp4_QH_warm_start": (False, "3D quasi-helical, nfp=4"),
}


def _golden_final_iteration(stdout_path: Path) -> int:
    final = None
    for line in stdout_path.read_text().splitlines():
        tok = line.split()
        if len(tok) >= 6 and tok[0].isdigit() and "E" in tok[1]:
            final = int(tok[0])
    if final is None:
        raise ValueError(f"no iteration rows in {stdout_path}")
    return final


def collect_parity() -> dict:
    import sys
    import netCDF4

    sys.path.insert(0, str(REPO / "tests" / "core_new"))
    from conftest import resolve_golden_dir

    from vmec_jax.core.input import VmecInput
    from vmec_jax.core import solver
    from vmec_jax.core.multigrid import solve_multigrid

    golden = resolve_golden_dir()
    if golden is None:
        raise RuntimeError("golden VMEC2000 fixtures unavailable")

    out = {}
    for name, (multigrid, _desc) in PARITY_CASES.items():
        inp = VmecInput.from_file(str(DATA / f"input.{name}"))
        res = solve_multigrid(inp) if multigrid else solver.solve(inp)
        with netCDF4.Dataset(golden / name / f"wout_{name}.nc") as ds:
            wb_gold = float(ds.variables["wb"][()])
        out[name] = dict(
            iterations=int(res.iterations),
            golden_iterations=_golden_final_iteration(golden / name / "stdout.txt"),
            jacobian_resets=int(res.jacobian_resets),
            wb_rel_err=abs(float(res.wb) - wb_gold) / abs(wb_gold),
        )
        print(name, out[name], flush=True)
    return out


def make_parity_figure(out: Path) -> None:
    data = collect_parity()
    rows = [(k, data[k]) for k in PARITY_CASES if k in data]

    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(rows) + 1.9), dpi=160)
    ax.set_axis_off()
    cols = [0.02, 0.44, 0.60, 0.76, 0.99]
    ax.text(cols[0], 1.0, "case", fontsize=9.5, color=MUTED, va="center")
    ax.text(cols[1], 1.0, "VMEC2000\niterations", fontsize=9.5, color=MUTED,
            va="center", ha="center", linespacing=1.2)
    ax.text(cols[2], 1.0, "vmec_jax\niterations", fontsize=9.5, color=MUTED,
            va="center", ha="center", linespacing=1.2)
    ax.text(cols[3], 1.0, "match", fontsize=9.5, color=MUTED,
            va="center", ha="center")
    ax.text(cols[4], 1.0, "plasma energy wb\nrel. difference", fontsize=9.5,
            color=MUTED, va="center", ha="right", linespacing=1.2)

    n = len(rows)
    for i, (k, r) in enumerate(rows):
        y = 0.86 - 0.86 * (i + 0.5) / n
        if i % 2 == 0:
            ax.axhspan(y - 0.43 * 0.86 / n * 2, y + 0.43 * 0.86 / n * 2,
                       color="#f4f3ef", zorder=0)
        ax.text(cols[0], y, CASE_LABELS[k], fontsize=10.5, color=INK,
                va="center", fontweight="bold")
        ax.text(cols[0] + 0.205, y, PARITY_CASES[k][1], fontsize=8.5,
                color=MUTED, va="center")
        exact = r["iterations"] == r["golden_iterations"]
        ax.text(cols[1], y, f'{r["golden_iterations"]:,}', fontsize=11,
                color=INK2, va="center", ha="center", fontfamily="monospace")
        ax.text(cols[2], y, f'{r["iterations"]:,}', fontsize=11, color=BLUE,
                va="center", ha="center", fontfamily="monospace",
                fontweight="bold")
        ax.text(cols[3], y, "=" if exact else "-", fontsize=13,
                color=GREEN_TEXT if exact else RED, va="center", ha="center",
                fontweight="bold")
        wb_txt = "exact" if r["wb_rel_err"] == 0 else f'{r["wb_rel_err"]:.1e}'
        ax.text(cols[4], y, wb_txt, fontsize=11, color=INK2, va="center",
                ha="right", fontfamily="monospace")
        if r.get("jacobian_resets"):
            ax.text(cols[2] + 0.052, y, "*", fontsize=12, color=BLUE,
                    va="center", ha="center")

    ax.axhline(0.925, color=BASELINE, lw=0.8)
    ax.text(0.02, -0.10,
            "* including one mid-run jacobian reset, reproduced at the same iteration.",
            fontsize=8.5, color=MUTED, va="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.08)
    ax.set_title("Iteration-for-iteration parity with VMEC2000", loc="left",
                 fontsize=13, color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# 3. Equilibrium showcase (solves the bundled quick-start case)
# --------------------------------------------------------------------------

def make_showcase_figure(out: Path) -> None:
    from vmec_jax.core.input import VmecInput
    from vmec_jax.core.multigrid import solve_multigrid
    from vmec_jax.core.wout import wout_from_state
    from vmec_jax.core.plotting import surface_rz, surface_modB, axis_rz

    inp = VmecInput.from_file(str(DATA / "input.nfp4_QH_warm_start"))
    res = solve_multigrid(inp)
    wout = wout_from_state(inp=inp, state=res.state, niter=res.iterations,
                           fsqr=res.fsqr, fsqz=res.fsqz, fsql=res.fsql)
    ns, nfp = int(wout.ns), int(wout.nfp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=160,
                                   width_ratios=[1.0, 1.5])
    theta = np.linspace(0, 2 * np.pi, 181)
    phis = [0.0, np.pi / (2 * nfp), np.pi / nfp]
    phi_labels = ["$\\phi = 0$", "$\\phi = $ quarter period",
                  "$\\phi = $ half period"]
    ramp = ["#9ec5f4", "#5598e7", "#1c5cab"]
    for phi, c, lab in zip(phis, ramp, phi_labels):
        for j, f in enumerate((0.25, 0.5, 1.0)):
            si = max(1, round((ns - 1) * f))
            R, Z = surface_rz(wout, s_index=si, theta=theta, phi=np.array([phi]))
            ax1.plot(R[:, 0], Z[:, 0], color=c, lw=1.8 if f == 1.0 else 0.9,
                     alpha=1.0 if f == 1.0 else 0.6,
                     label=lab if j == 2 else None)
        Ra, Za = axis_rz(wout, np.array([phi]))
        ax1.plot(Ra, Za, marker="o", ms=4, color=c, mec=SURFACE, mew=1.0,
                 ls="none")
    ax1.set_aspect("equal")
    ax1.set_xlabel("R (m)")
    ax1.set_ylabel("Z (m)")
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=7.5,
               handlelength=1.3, labelspacing=0.3)
    ax1.set_title("Flux surfaces", loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    th = np.linspace(0, 2 * np.pi, 161)
    ph = np.linspace(0, 2 * np.pi / nfp, 161)
    modB = surface_modB(wout, s_index=ns - 1, theta=th, phi=ph)
    pc = ax2.contourf(ph * nfp / (2 * np.pi), th / (2 * np.pi), modB,
                      levels=24, cmap="Blues")
    ax2.contour(ph * nfp / (2 * np.pi), th / (2 * np.pi), modB, levels=8,
                colors=SURFACE, linewidths=0.45)
    cb = fig.colorbar(pc, ax=ax2, pad=0.02)
    cb.set_label("|B| (T)", color=INK2, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_visible(False)
    ax2.set_xlabel("toroidal angle (field periods)")
    ax2.set_ylabel("poloidal angle / 2$\\pi$")
    ax2.set_title("|B| on the boundary surface", loc="left", fontsize=11,
                  color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="runtime,parity,showcase")
    ap.add_argument("--outdir", default=str(REPO / "docs" / "_static" / "figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    apply_style()
    which = set(args.only.split(","))
    if "runtime" in which:
        make_runtime_figure(outdir / "readme_runtime_compare.png")
    if "parity" in which:
        make_parity_figure(outdir / "readme_parity.png")
    if "showcase" in which:
        make_showcase_figure(outdir / "readme_equilibrium_showcase.png")


if __name__ == "__main__":
    main()

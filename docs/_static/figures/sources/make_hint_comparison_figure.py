#!/usr/bin/env python3
"""Render the HINT/VMEX snapshot and relaxation history from compact records.

The current HINT state is still relaxing. VMEX uses native force-polished
surfaces and virtual casing outside them. These are different boundary
formulations, so the pictures do not establish matched equilibrium accuracy.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import maximum_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[4]
FIGURES = ROOT / "docs/_static/figures"
BLUE, ORANGE, PURPLE = "#2674ab", "#db572b", "#7553a3"


def _cloud(ax, counts, r, z, color, footprint=3):
    hits = maximum_filter(counts, size=footprint)
    rgba = np.empty((*hits.shape, 4))
    rgba[..., :3] = matplotlib.colors.to_rgb(color)
    rgba[..., 3] = np.where(hits > 0, np.minimum(.9, .68+.06*np.log1p(hits)), 0)
    ax.imshow(rgba, origin="lower", extent=(r[0], r[-1], z[0], z[-1]), interpolation="nearest")


def sections(counts, surfaces, r, z, exterior=None, core=None, axes=None, title="", subtitle=""):
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"}):
        fig, panels = plt.subplots(1, 2, figsize=(8.2, 6.3))
        for cut, ax in enumerate(panels):
            for i, curve in enumerate(surfaces[cut]):
                ax.plot(*curve.T, color=BLUE, lw=1.6 if i == len(surfaces[cut])-1 else .65, alpha=.85)
            if exterior is not None:
                _cloud(ax, exterior[cut], r, z, PURPLE, footprint=5)
            _cloud(ax, counts[cut], r, z, ORANGE)
            if axes is not None:
                ax.scatter(*axes[cut], marker="x", s=18, color=ORANGE, linewidths=1)
            ax.set(title=(r"$\phi = 0$", r"$\phi = \pi/4$")[cut], xlabel="R [m]",
                   ylabel="Z [m]", aspect="equal", xlim=(r[4], r[-5]), ylim=(z[4], z[-5]))
            ax.grid(alpha=.1)
        if core is not None and len(core):
            points = core[core[:, 0] == 1, 1:]
            centre = axes[1] if axes is not None else np.mean(points, axis=0)
            inset = panels[1].inset_axes([.04, .025, .45, .26])
            for curve in surfaces[1]:
                inset.plot(*curve.T, color=BLUE, lw=.5)
            inset.scatter(*points.T, s=1.5, color=ORANGE, alpha=.85, linewidths=0)
            inset.set(xlim=(centre[0]-.02, centre[0]+.02), ylim=(centre[1]-.025, centre[1]+.025), aspect="equal")
            inset.set_title("Axis region", fontsize=8)
            inset.tick_params(labelsize=6)
        handles = [Line2D([], [], color=BLUE), Line2D([], [], color=ORANGE, marker=".", ls="")]
        labels = ["VMEX surfaces", "HINT crossings"]
        if exterior is not None:
            handles.append(Line2D([], [], color=PURPLE, marker=".", ls="")); labels.append("VMEX extender (stopped traces)")
        fig.suptitle(title or "HINT / VMEX · QA 0.5% target", y=.98, fontsize=15)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .935), ncol=len(labels), frameon=False)
        comparison = "VMEX fixed boundary; exterior traces stop at mask/grid limits" if exterior is not None else "VMEX fixed boundary, force polished"
        fig.text(.5, .035, "HINT free boundary, still relaxing · "+comparison+"\n"+subtitle,
                 ha="center", fontsize=9, color="#444444")
        fig.subplots_adjust(left=.10, right=.97, bottom=.15, top=.82, wspace=.35)
        _check_layout(fig)
        return fig


def residuals(history):
    rows = history["records"]
    i = np.array([r["iteration"] for r in rows])
    force = np.array([r["force_residual"] if r["force_residual"] is not None else np.nan for r in rows])
    change = np.array([r["response_relative_change_per_iteration"] if r["response_relative_change_per_iteration"] is not None else np.nan for r in rows])
    restart = next((r["iteration"] for r in rows if r["damping_nu1"]), None)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, panels = plt.subplots(1, 2, figsize=(8.2, 2.7))
        for ax, y, color, label in zip(panels, (force, change), (ORANGE, BLUE),
                                      ("Relative force residual", "Relative plasma-field change")):
            ax.semilogy(i, y, color=color, marker=".", ms=3, lw=1.2)
            if restart is not None:
                ax.axvline(restart-.5, color="#888888", lw=.8, ls="--")
            ax.set(xlabel="Saved outer iteration", title=label)
            ax.grid(alpha=.15)
        fig.text(.5, .015, "Dashed line: response viscosity added · force uses positive-pressure support, not a traced LCFS",
                 ha="center", fontsize=8, color="#444444")
        fig.tight_layout(rect=(0, .04, 1, 1))
        _check_layout(fig)
        return fig


def _check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [*fig.texts, *fig.legends]
    for ax in fig.axes:
        labels.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            labels.extend(t.label1 for t in axis.get_major_ticks()
                          if min(limits) <= t.get_loc() <= max(limits))
    for text in labels:
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1


def main():
    with np.load(ROOT / "benchmarks/hint_poincare_sections.npz", allow_pickle=False) as data:
        meta = json.loads(data["metadata_json"].item())
        fig = sections(data["hint_counts"], data["vmex_surfaces"], data["R_edges_m"], data["Z_edges_m"],
                       exterior=data["extender_counts"], core=data["hint_core_points"], axes=data["hint_axes_RZ_m"],
                       subtitle=f"Relaxation time {meta['hint']['relaxation_time_code_units']:.2f}")
    fig.savefig(FIGURES / "readme_hint_comparison.webp", dpi=160, pil_kwargs={"lossless": True})
    plt.close(fig)
    history = json.loads((ROOT / "benchmarks/hint_relaxation.json").read_text())
    fig = residuals(history)
    fig.savefig(FIGURES / "readme_hint_relaxation.webp", dpi=160, pil_kwargs={"lossless": True})
    plt.close(fig)


if __name__ == "__main__":
    main()

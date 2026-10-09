#!/usr/bin/env python3
"""Figures of ``run_three_term_resolution.py``: four methods at every resolution against the 12 x 12 target.

    python benchmarks/plot_three_term_resolution.py OUT [--sections 6 8 10 12]

Reads ``OUT/m*/<method>.json`` and ``OUT/m*/wout_<method>.nc`` (any method or resolution still missing is left
out, so the figures can be redrawn as runs finish) and writes ``OUT/three_term_resolution.png`` (distance to the
target, the three interface conditions, wall time and peak GPU memory against mpol = ntor) and
``OUT/three_term_resolution_lcfs.png`` (LCFS cross-sections of every method on the target's at a few resolutions)
and ``OUT/three_term_resolution_surfaces.png`` (nested flux surfaces at ``--interior`` resolutions, the largest
distance of every surface to the target's at the same normalized flux, and the iota profiles).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

METHODS = {  # name: (label, colour, marker, line style)
    "target": ("fixed-boundary target (same field)", "k", "s", "-"),
    "three_term": ("three-term (VMEX, K = 0)", "tab:blue", "o", "-"),
    "desc": ("DESC free boundary (K = 0), L = 2M", "tab:green", "^", "-"),
    "desc_lm": ("DESC free boundary (K = 0), L = M", "tab:olive", "^", ":"),
    "nestor": ("VMEC + NESTOR (VMEX)", "tab:red", "v", "--"),
    "nestor_mgrid": ("VMEC + NESTOR (VMEX, mgrid)", "tab:orange", "<", "--"),
    "vmec2000": ("VMEC2000 + NESTOR (mgrid, CPU)", "tab:purple", "D", "--"),
    "start": ("initial guess (all methods)", "0.55", "x", ":"),
}


def lcfs(path, theta, phi, js=-1):
    """``R, Z`` of surface(s) ``js`` (row of ``rmnc``) at toroidal angle ``phi``."""
    with netCDF4.Dataset(path) as nc:
        xm, xn = np.asarray(nc["xm"][:], float), np.asarray(nc["xn"][:], float)
        rmnc, zmns = np.asarray(nc["rmnc"][:])[js], np.asarray(nc["zmns"][:])[js]
    a = np.outer(theta, xm) - xn * phi
    return np.cos(a) @ rmnc.T, np.sin(a) @ zmns.T


def surfaces(path):
    """Normalized flux of every surface, |iota| and the surface arrays of a wout."""
    with netCDF4.Dataset(path) as nc:
        ns = int(nc["ns"][:])
        return np.linspace(0.0, 1.0, ns), np.abs(np.asarray(nc["iotaf"][:], float))


def surface_mm(path_a, path_b, nfp, js):
    """Largest distance of each surface ``js`` of ``path_a`` to the same surface of ``path_b`` (four planes), mm."""
    theta = np.linspace(0, 2 * np.pi, 361)
    worst = np.zeros(len(js))
    for phi in np.linspace(0, np.pi / nfp, 4):
        (Ra, Za), (Rb, Zb) = lcfs(path_a, theta, phi, js), lcfs(path_b, theta, phi, js)
        for k in range(len(js)):
            d = np.hypot(Ra[:, k, None] - Rb[None, :, k], Za[:, k, None] - Zb[None, :, k])
            worst[k] = max(worst[k], float(np.max(np.min(d, axis=1))))
    return 1e3 * worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--sections", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    p.add_argument("--interior", type=int, nargs="+", default=[8, 12])
    a = p.parse_args()
    rows = [json.loads(f.read_text()) for f in sorted(a.out.glob("m*/*.json"))]
    by = {m: sorted((r for r in rows if r["method"] == m), key=lambda r: r["modes"]) for m in METHODS}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    panels = [("lcfs_mm_vs_truth", "largest LCFS distance to the 12 x 12 target [mm]", "linear"),
              ("bn", "B.n / |B|  (RMS)", "log"),
              ("pressure_balance", "pressure jump (|B_out|^2 - |B_in|^2 - 2 mu0 p) / 2|B|^2", "log"),
              ("sheet_current", "sheet current mu0 |K| / |B|", "log"),
              ("seconds", "wall time [s] (solve, compilation included)", "log"),
              ("peak_gpu_gib", "peak GPU memory [GiB] (VMEC2000: CPU, 8 MPI ranks)", "log")]
    for ax, (key, title, scale) in zip(axes.ravel(), panels):
        for m, (label, colour, marker, ls) in METHODS.items():
            pts = [(r["modes"], r[key]) for r in by[m] if r.get(key) is not None]
            if key in ("seconds", "peak_gpu_gib") and m in ("target", "start"):
                continue
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, ls, color=colour, marker=marker, label=label, lw=1.4, ms=6)
            if key == "lcfs_mm_vs_truth" and ("nestor" in m or m == "vmec2000"):  # open: stopped at the iteration cap
                for r in by[m]:
                    if r.get("converged") is False and r.get(key) is not None:
                        ax.plot(r["modes"], r[key], marker, color="white", mec=colour, ms=6, zorder=3)
        ax.set_yscale(scale)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("mpol = ntor  (ns = 51)", fontsize=8)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[0, 0].text(0.98, 0.55, "open markers: NESTOR not converged\n(iteration cap)", transform=axes[0, 0].transAxes,
                    ha="right", va="top", fontsize=7, color="tab:red")
    case = (a.out / "case.txt").read_text().strip() if (a.out / "case.txt").exists() else "beta"
    label = {"beta": "precise-QA beta 2.5%", "vacuum": "precise-QA vacuum"}.get(case, case)
    fig.suptitle(f"Free boundary of a known answer: {label}; the coil field is fitted to cancel B.n on the "
                 "12 x 12 fixed-boundary target (so the target's B.n there is the fit residual, while its pressure jump "
                 "and sheet current\nshow the 48 x 48 virtual-casing evaluation's own error, ~2e-4); every method starts "
                 "from the same boundary (minor radius x 0.97)", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out / "three_term_resolution.png", dpi=130)
    plt.close(fig)

    # ---- cross-sections ------------------------------------------------------------------------------------------
    truth = a.out / "truth.nc"
    with netCDF4.Dataset(truth) as nc:
        nfp = int(nc["nfp"][:])
    theta = np.linspace(0, 2 * np.pi, 361)
    planes = [0.0, np.pi / (2 * nfp), np.pi / nfp]
    sections = [m for m in a.sections if (a.out / f"m{m}").exists()]
    fig, axes = plt.subplots(len(sections), len(planes), figsize=(4.2 * len(planes), 3.6 * len(sections)),
                             squeeze=False)
    for i, modes in enumerate(sections):
        for j, phi in enumerate(planes):
            ax = axes[i, j]
            R, Z = lcfs(truth, theta, phi)
            ax.plot(R, Z, color="0.6", lw=4, label="12 x 12 target")
            for m, (label, colour, _, ls) in METHODS.items():
                w = a.out / f"m{modes}" / f"wout_{m}.nc"
                if m not in ("target", "start") and w.exists():
                    R, Z = lcfs(w, theta, phi)
                    ax.plot(R, Z, ls, color=colour, lw=1.3, label=label)
            ax.set_aspect("equal")
            ax.set_title(f"mpol = ntor = {modes},  phi = {phi * nfp / (2 * np.pi):.2f} field period", fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out / "three_term_resolution_lcfs.png", dpi=120)
    plt.close(fig)
    interior(a.out, truth, nfp, [m for m in a.interior if (a.out / f"m{m}").exists()])
    print("wrote", a.out / "three_term_resolution.png", a.out / "three_term_resolution_lcfs.png",
          a.out / "three_term_resolution_surfaces.png")


def interior(out, truth, nfp, resolutions):
    """Nested surfaces, their distance to the target's and the iota profiles at ``resolutions``."""
    s_truth, iota_truth = surfaces(truth)
    ns = s_truth.size
    shown = [round(f * (ns - 1)) for f in (0.04, 0.25, 0.5, 0.75, 1.0)]
    theta = np.linspace(0, 2 * np.pi, 361)
    planes = [0.0, np.pi / (2 * nfp), np.pi / nfp]
    methods = [m for m in METHODS if m not in ("target", "start")]
    fig, axes = plt.subplots(len(resolutions) + 1, 3, figsize=(13.5, 4.3 * (len(resolutions) + 1)), squeeze=False)
    for i, modes in enumerate(resolutions):
        for j, phi in enumerate(planes):
            ax = axes[i, j]
            R, Z = lcfs(truth, theta, phi, shown)
            ax.plot(R, Z, color="0.6", lw=3.5)
            ax.plot(*[c[:1, 0] for c in lcfs(truth, np.zeros(1), phi, [0])], "o", color="0.6", ms=7)
            for m in methods:
                w = out / f"m{modes}" / f"wout_{m}.nc"
                if not w.exists():
                    continue
                label, colour, _, ls = METHODS[m]
                with netCDF4.Dataset(w) as nc:
                    n = int(nc["ns"][:])
                rows = [round(f * (n - 1)) for f in (0.04, 0.25, 0.5, 0.75, 1.0)]
                R, Z = lcfs(w, theta, phi, rows)
                ax.plot(R, Z, ls, color=colour, lw=1.1)
                ax.plot(*[c[:1, 0] for c in lcfs(w, np.zeros(1), phi, [0])], "+", color=colour, ms=8)
            ax.set_aspect("equal")
            ax.set_title(f"mpol = ntor = {modes},  phi = {phi * nfp / (2 * np.pi):.2f} field period\n"
                         "s = 0.04, 0.25, 0.5, 0.75, 1 and the axis (grey: 12 x 12 target)", fontsize=8)
            ax.tick_params(labelsize=7)
    # distance of every surface to the target's, and the iota profiles
    ax_d, ax_i, ax_di = axes[-1]
    for modes, marker in zip(resolutions, ("o", "s", "^", "D")):
        for m in methods:
            w = out / f"m{modes}" / f"wout_{m}.nc"
            if not w.exists():
                continue
            label, colour, _, ls = METHODS[m]
            s_m, iota_m = surfaces(w)
            js = list(range(1, s_m.size))
            if s_m.size == ns:
                ax_d.plot(s_m[1:], surface_mm(w, truth, nfp, js), ls, color=colour, marker=marker, ms=3, lw=1,
                          markevery=5, label=f"{label}, {modes} x {modes}")
            ax_i.plot(s_m, iota_m, ls, color=colour, lw=1, marker=marker, ms=3, markevery=5)
            ax_di.plot(s_m, np.abs(iota_m - np.interp(s_m, s_truth, iota_truth)), ls, color=colour, lw=1,
                       marker=marker, ms=3, markevery=5)
    ax_i.plot(s_truth, iota_truth, color="0.4", lw=3, alpha=0.6, label="12 x 12 target")
    ax_d.set_ylabel("largest distance to the target's surface [mm]", fontsize=8)
    ax_i.set_ylabel("|iota|", fontsize=8)
    ax_di.set_ylabel("| |iota| - |iota_target| |", fontsize=8)
    ax_di.set_yscale("log")
    for ax in (ax_d, ax_i, ax_di):
        ax.set_xlabel("normalized toroidal flux s", fontsize=8)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(labelsize=7)
    ax_d.legend(fontsize=6)
    ax_i.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "three_term_resolution_surfaces.png", dpi=115)
    plt.close(fig)


if __name__ == "__main__":
    main()

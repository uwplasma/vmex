#!/usr/bin/env python3
"""Plot the retained near-circle comparison; no equilibrium solve is performed.

    python docs/_static/figures/sources/make_low_transport_figure.py

Read WOUT, Boozer, NEO and particle outputs from a qualification directory.
No simulation or transport dataset is stored in this repository. Use
--evidence /path/to/qualification to reproduce the retained comparison.
Rendering needs NumPy, Matplotlib, Pillow and SciPy.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "readme_low_transport.webp"
NAMES = ("seed2", "mapped2")
LABELS = ("Near-circular seed", "Optimized")
COLORS = ("#566a79", "#c56627")


def extract(root):
    from scipy.io import netcdf_file

    arrays, hashes = {}, {}

    def source(relative):
        path = root / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    for i, name in enumerate(NAMES):
        with np.load(source(f"{name}/neo/boozer-b32.npz")) as z:
            arrays[f"b{i}"] = z["bmnc_b"][:, 1]
            arrays["m"], arrays["n"] = z["ixm_b"], z["ixn_b"]
            arrays["s_contour"], arrays["nfp"] = z["s_b"][1], z["nfp_b"]
        neo = json.loads(source(f"{name}/neo/batch-result.json").read_text())
        arrays[f"Q{i}"] = np.array([r["epsilon_eff_3_over_2"] for r in neo["result"]])
        arrays["s"] = np.array(neo["actual_s"])
        with np.load(source(f"{name}/particles/trace.npz")) as z:
            arrays[f"loss_times{i}"], arrays[f"weights{i}"] = z["loss_times"], z["weights"]
            arrays[f"births{i}"] = z["births"]
    assert np.array_equal(arrays.pop("births0"), arrays.pop("births1"))
    with netcdf_file(source("mapped2/wout.nc"), mmap=False) as f:
        for key in ("rmnc", "zmns"):
            arrays[key] = f.variables[key].data[-1].copy()
        for key in ("xm", "xn"):
            arrays[key] = f.variables[key].data.copy()
    arrays["source_sha256"] = np.array(json.dumps(hashes, sort_keys=True))
    return arrays


def draw(z):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(12, 6.3), dpi=150, layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.12, 1))
    boundary = fig.add_subplot(grid[0, 0], projection="3d")
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 65), np.linspace(0, 2*np.pi, 193))
    phase = theta[..., None]*z["xm"]-phi[..., None]*z["xn"]
    R = np.sum(z["rmnc"]*np.cos(phase), axis=-1)
    Z = np.sum(z["zmns"]*np.sin(phase), axis=-1)
    X, Y = R*np.cos(phi), R*np.sin(phi)
    boundary.plot_surface(X, Y, Z, color="#689ba5", linewidth=0, antialiased=True,
                          rcount=193, ccount=65, shade=True)
    boundary.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    boundary.view_init(elev=35, azim=-45)
    boundary.set_axis_off()
    boundary.set_title(f"Optimized vacuum boundary, nfp = {int(z['nfp'])}", pad=0)

    theta, zeta = np.meshgrid(np.linspace(0, 2*np.pi, 161),
                              np.linspace(0, 2*np.pi/int(z["nfp"]), 161), indexing="ij")
    phase = theta[..., None]*z["m"]-zeta[..., None]*z["n"]
    fields = [np.sum(z[f"b{i}"]*np.cos(phase), axis=-1) for i in range(2)]
    levels = np.linspace(min(B.min() for B in fields), max(B.max() for B in fields), 14)[1:-1]
    contours = []
    for i in range(2):
        ax = fig.add_subplot(grid[0, i+1])
        c = ax.contour(zeta, theta, fields[i], levels=levels, cmap="viridis", linewidths=.85)
        ax.set(xlabel=r"Boozer $\zeta_B$ [rad]", ylabel=r"Boozer $\theta_B$ [rad]",
               title=f"{LABELS[i]}: |B|, s = {float(z['s_contour']):.3f}",
               xticks=np.linspace(0, 2*np.pi/int(z["nfp"]), 3),
               yticks=[0, np.pi, 2*np.pi], yticklabels=["0", r"$\pi$", r"$2\pi$"])
        contours.append(ax)
    fig.colorbar(c, ax=contours, label="|B| [T]", shrink=.85, pad=.02, format="%.2f")

    neo = fig.add_subplot(grid[1, 0])
    loss = fig.add_subplot(grid[1, 1:])
    time = np.linspace(0, .01, 801)
    for i in range(2):
        neo.semilogy(z["s"], z[f"Q{i}"], "o-", color=COLORS[i], label=LABELS[i], ms=4)
        w, t = z[f"weights{i}"], z[f"loss_times{i}"]
        curve = 100*np.sum(w[:, None]*((t[:, None] >= 0)&(t[:, None] <= time)), axis=0)/w.sum()
        loss.plot(time*1000, curve, color=COLORS[i], label=LABELS[i], lw=1.7)
        loss.annotate(f"{curve[-1]:.2f}%", xy=(10, curve[-1]), xytext=(-5, 5),
                      textcoords="offset points", ha="right", color=COLORS[i])
    neo.set(xlabel="Normalized toroidal flux s", ylabel=r"$Q=\epsilon_\mathrm{eff}^{3/2}$",
            title=f"Independent NEO: mean Q down {100*(1-z['Q1'].mean()/z['Q0'].mean()):.2f}%")
    loss.set(xlabel="Time [ms]", ylabel="Weighted lost particles [%]", xlim=(0, 10), ylim=(-2, 110),
             title="256 paired alphas · 3.52 MeV · s = 0.5 · volume-RMS 6 T / minor radius 1.7 m")
    for ax in (neo, loss):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, loc="center right")
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(256).save(
        OUT, lossless=True, method=6)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    draw(extract(args.evidence))

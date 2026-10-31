#!/usr/bin/env python3
"""Plot the retained near-circle comparison; no equilibrium solve is performed.

    python docs/_static/figures/sources/make_low_transport_figure.py

Read WOUT, Boozer, NEO and particle outputs from a qualification directory.
No simulation or transport dataset is stored in this repository. Use
--evidence /path/to/qualification to reproduce the retained comparison.
Rendering needs NumPy, Matplotlib, Pillow, SciPy and ESSOS.
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
LABELS = ("Near-circular seed", "pwO")
COLORS = ("#566a79", "#c56627")


def extract(root):
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
    arrays.update(boundary_data(source("mapped2/wout.nc")))
    arrays["source_sha256"] = np.array(json.dumps(hashes, sort_keys=True))
    return arrays


def boundary_data(path):
    from scipy.io import netcdf_file

    with netcdf_file(path, mmap=False) as f:
        z = {key: f.variables[key].data[-1].copy() for key in ("rmnc", "zmns")}
        z.update({key: f.variables[key].data.copy() for key in ("xm", "xn")})
        z["edge_B"] = 1.5*f.variables["bmnc"].data[-1].copy()-.5*f.variables["bmnc"].data[-2].copy()
        z["m_B"], z["n_B"] = (f.variables[k].data.copy() for k in ("xm_nyq", "xn_nyq"))
    return z


def draw_boundary(fig, boundary, z):
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 65), np.linspace(0, 2*np.pi, 193))
    phase = theta[..., None]*z["xm"]-phi[..., None]*z["xn"]
    R = np.sum(z["rmnc"]*np.cos(phase), axis=-1)
    Z = np.sum(z["zmns"]*np.sin(phase), axis=-1)
    X, Y = R*np.cos(phi), R*np.sin(phi)
    B = np.sum(z["edge_B"]*np.cos(theta[..., None]*z["m_B"]-phi[..., None]*z["n_B"]), axis=-1)
    norm = matplotlib.colors.Normalize(B.min(), B.max())
    boundary.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(norm(B)), linewidth=0,
                          rcount=193, ccount=65, shade=False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=boundary,
                 orientation="horizontal", shrink=.65, pad=0, label="|B| [T]")
    boundary.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))


def draw(z):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 9, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(12, 6.3), dpi=150, layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.12, 1))
    boundary = fig.add_subplot(grid[0, 0], projection="3d")
    draw_boundary(fig, boundary, z)
    boundary.view_init(elev=35, azim=-45)
    boundary.set_axis_off()
    boundary.set_title(f"pwO vacuum boundary, nfp = {int(z['nfp'])}", pad=0)

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
    save(fig, OUT)


def save(fig, path):
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(256).save(
        path, lossless=True, method=6)
    plt.close(fig)
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


def comparisons(root):
    """Show every NFP trial and the completed independent family measurements."""
    from scipy.io import netcdf_file

    fig, axes = plt.subplots(3, 4, figsize=(14, 10), dpi=150, layout="constrained")
    statuses = ("QA to pwO: NEO improved", "NEO improved", "QA to pwO: NEO improved", "NEO improved")
    theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 81), np.linspace(0, 2*np.pi, 81), indexing="ij")
    for col, nfp in enumerate(range(1, 5)):
        folder = root.parent / "mapped" / f"mirror-nfp{nfp}" / "low_transport"
        if nfp in (1, 3):
            folder = root.parent / "mapped" / {1: "nfp1-qa2-full-20261031",
                3: "qa3-pwo-warm-polish-20261031"}[nfp] / "backtrack"
        record = json.loads((folder / "summary.json").read_text())
        selected = nfp in (1, 3) or "selected_evaluation" in record
        stem = "trial" if selected else "terminal"
        picture = folder / (f"{stem}_boundary3d.png" if selected else f"terminal/{stem}_boundary3d.png")
        axes[0, col].imshow(plt.imread(picture)); axes[0, col].axis("off")
        axes[0, col].set_title(f"NFP {nfp}: {statuses[col]}")
        with netcdf_file(folder / f"wout_{stem}.nc", mmap=False) as f:
            m, n = (f.variables[k].data.copy() for k in ("xm_nyq", "xn_nyq"))
            coefficients = f.variables["bmnc"].data.copy()
        row = round((len(coefficients)-1)*.5+.5)
        B = np.sum(coefficients[row]*np.cos(theta[..., None]*m-phi[..., None]*n/nfp), axis=-1)
        contour = axes[1, col].contour(phi/nfp, theta, B, levels=14, cmap="viridis", linewidths=.8)
        axes[1, col].set(xlabel="VMEC toroidal angle", ylabel="VMEC poloidal angle",
                         title="Mid-radius |B| [T]")
        fig.colorbar(contour, ax=axes[1, col], shrink=.75)
        if nfp in (1, 3):
            points = sorted([(0., 1., True)]+[(r["alpha"], r["physics"]["raw"]/record["initial_pwO"],
                           r["accepted"]) for r in record["trials"]])
            axes[2, col].plot([r[0] for r in points], [r[1] for r in points], color="gray")
            for x, y, accepted in points:
                axes[2, col].scatter(x, y, color="#306d86" if accepted else "#bc3939", zorder=3)
            axes[2, col].set(xlabel="Fraction of proposed pwO step", ylabel="pwO variance / QA seed",
                             title="NEO backtracking (red: rejected)")
        else:
            cost = np.array([record["evaluations"][0]["cost"]]+[r["cost"] for r in record["iterations"]])
            axes[2, col].semilogy(np.arange(len(cost)), cost/cost[0], color="#306d86")
            axes[2, col].set(xlabel="Accepted iteration", ylabel="Total objective / initial",
                             title="Optimization history")
        axes[2, col].grid(alpha=.2)
    save(fig, OUT.with_name("readme_low_transport_nfp.webp"))

    data = json.loads((root / "summary.json").read_text())["cases"]
    for name, path in (("pwo1", "nfp1-qa2-full-20261031"),
                       ("pwo3", "qa3-pwo-warm-polish-20261031")):
        if name in data:
            continue
        folder = root.parent / "mapped" / path / "backtrack"
        row = json.loads((folder / "summary.json").read_text())
        with netcdf_file(folder / "wout_trial.nc", mmap=False) as f:
            iota = f.variables["iotaf"].data.copy().tolist()
        data[name] = dict(Q=row["accepted"]["Q"], actual_s=[.25, 16.5/34, .75],
                          refined_equilibrium=dict(full_signed_iota=iota), coarse=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), dpi=150)
    fig.subplots_adjust(left=.12, right=.98, top=.95, bottom=.20, hspace=.34)
    axes[2].set_prop_cycle(color=["#566a79", "#99a3af", "#2c7fb8", "#e34a33",
                                   "#7b3294", "#8c564b", "#c78a00", "#b85b20", "#007d63",
                                   "#afbecf", "#269fd0", "#242424"])
    handles, labels, missing = [], [], []
    for name, label in (("seed1", "Seed1"), ("seed4", "Seed4"), ("QA", "QA4"), ("QH", "QH4"),
            ("QP", "QP4"), ("QI", "QI4"), ("pwo1", "pwO1"), ("pwo3", "pwO3"), ("mapped4", "pwO4"),
            ("seed2", "Seed2"), ("QA_nfp2", "QA2"), ("mapped2", "pwO2")):
        row = data.get(name, {})
        if row.get("coarse"):
            label += " (NS35)"
        iota = row.get("refined_equilibrium", {}).get("full_signed_iota")
        if iota is None:
            iota = json.loads((root / name / "coarse-iota.json").read_text())["iotaf"]
            label += " (NS35)"
        line, = axes[2].plot(np.linspace(0, 1, len(iota)), iota,
                             ls="--" if "NS35" in label else "-")
        handles.append(line); labels.append(label)
        if not row.get("Q"):
            missing.append(label+": refined transport and particles"); continue
        axes[0].semilogy(row["actual_s"], np.array(row["Q"])**(2/3), "o-", color=line.get_color())
        particle = row.get("particles", {})
        if particle.get("status") == "completed":
            axes[1].plot(1000*np.array(particle["times_s"]), 100*np.array(particle["weighted_loss_curve"]),
                         color=line.get_color())
        else:
            missing.append(label+": particles")
    axes[0].set(xlabel="Normalized toroidal flux s", ylabel=r"Effective ripple $\epsilon_{\rm eff}$")
    axes[1].set(xlabel="Time [ms]", ylabel="Weighted alpha loss [%]", xlim=(0, 10), ylim=(-2, 103))
    axes[2].axhline(.41, color="black", ls=":", lw=1, label="Iota floor")
    axes[2].set(xlabel="Normalized toroidal flux s", ylabel=r"Rotational transform $\iota$")
    axes[2].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("pwO and symmetry controls at aspect ratio 6", fontsize=14)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .085), ncol=5, frameon=False)
    fig.text(.5, .055, "256 paired 3.52 MeV alphas; s=0.5; volume-RMS B=6 T; minor radius=1.7 m", ha="center")
    fig.text(.5, .025, "Not completed: "+"; ".join(missing), ha="center", fontsize=9)
    save(fig, OUT.with_name("readme_low_transport_comparison.webp"))


def coil_figure(root):
    """Render the accepted ESSOS coil set and its fixed-boundary target."""
    from essos.coils import Coils

    fig = plt.figure(figsize=(10, 5.5), dpi=150, layout="constrained")
    for index, (name, label) in enumerate((("public-warm-nfp4-final", "Vacuum"),
                                         ("public-warm-finite-tight", "Finite beta")), 1):
        folder = root.parent / "stage2" / name
        record = json.loads((folder / "single_stage_optimization_pwO_summary.json").read_text())
        free = record["free_boundary_verification"]
        coils = Coils.from_json(str(folder / "coils_single_stage_pwO_optimized.json"))
        coils.n_segments = 384
        curves = np.asarray(coils.gamma)
        z = boundary_data(folder / "wout_single_stage_pwO_free_check.nc")
        ax = fig.add_subplot(1, 2, index, projection="3d")
        draw_boundary(fig, ax, z)
        for curve in curves:
            curve = np.concatenate((curve, curve[:1]))
            ax.plot(*curve.T, color="#b65c32", lw=1.1)
        ax.set_box_aspect(np.ptp(curves.reshape(-1, 3), axis=0))
        ax.view_init(elev=35, azim=-45)
        ax.set_axis_off()
        ax.set_title(f"{label}: beta = {100*free['beta']:.3f}%\n"
                     f"Cold free boundary: {1000*free['sampled_boundary_distance_m']:.2f} mm from target")
    fig.suptitle("pwO4 equilibria held by two unique coils", fontsize=12)
    save(fig, OUT.with_name("readme_low_transport_coils.webp"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    draw(extract(args.evidence))
    comparisons(args.evidence)
    coil_figure(args.evidence)

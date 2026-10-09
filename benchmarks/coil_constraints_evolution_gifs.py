#!/usr/bin/env python
"""3D and 2D GIFs of the coils and the LCFS over an optimization's saved steps.

    JAX_PLATFORMS=cpu python benchmarks/coil_constraints_evolution_gifs.py <label> <prefix> <run dir>[:offset] ...

The runs are those of the coil-constraint single-stage examples
(``examples/optimization/single_stage_*_coil_constraints.py`` and
``single_stage_free_boundary_optimization_three_term.py``). Each run dir holds ``wout.stepK.nc``,
``coils.stepK.json`` and ``metrics.jsonl`` (``--save-every 1``); a restart's dir follows with ``:offset``, the
global step of its step 0 (which repeats the previous run's last step and is skipped). ``--stride N`` keeps every
Nth step. Writes ``<prefix>_3d.gif`` (the full LCFS and every coil) and ``<prefix>_2d.gif`` (LCFS cross-sections
at three toroidal angles with the coils crossing them, step 0 dashed, and the QA and sheet-current history).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("label")
parser.add_argument("prefix")
parser.add_argument("runs", nargs="+")
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--fps", type=int, default=8)
parser.add_argument("--only", choices=("2d", "3d"), help="write one of the two GIFs")
args = parser.parse_args()

from essos.coils import Coils  # noqa: E402  (after argparse: imports JAX)

frames = []  # (global step, wout path, coils path, metrics row)
for spec in args.runs:
    path, _, offset = spec.partition(":")
    path, offset = Path(path), int(offset or 0)
    rows = [json.loads(line) for line in open(path / "metrics.jsonl")]
    for row in rows:
        k = int(row["step"])
        if offset and k == 0:
            continue
        frames.append((offset + k, path / f"wout.step{k}.nc", path / f"coils.step{k}.json", row))
history = frames
frames = frames[:: args.stride] + ([frames[-1]] if (len(frames) - 1) % args.stride else [])
print(f"{len(frames)} frames from {len(history)} steps", flush=True)


def lcfs(path):
    with netCDF4.Dataset(path) as nc:
        return (np.asarray(nc["xm"][:]), np.asarray(nc["xn"][:]), np.asarray(nc["rmnc"][-1]),
                np.asarray(nc["zmns"][-1]), int(nc["nfp"][:]))


def rz(surface, theta, phi):
    xm, xn, rmnc, zmns, _ = surface
    angle = np.multiply.outer(theta, xm)[..., None, :] - np.multiply.outer(phi, xn)[None]
    return np.cos(angle) @ rmnc, np.sin(angle) @ zmns


coils = [np.asarray(Coils.from_json(str(f[2])).gamma) for f in frames]  # (coils, points, 3)
surfaces = [lcfs(f[1]) for f in frames]
nfp = surfaces[0][4]
theta = np.linspace(0, 2 * np.pi, 49)
phi3 = np.linspace(0, 2 * np.pi, 121)
# Modular coils span a narrow toroidal range each and avoid the symmetry planes: cut through the
# centroids of the first three coils, so each section shows its coil next to the plasma.
planes = np.array([np.arctan2(*np.mean(c[:, :2], axis=0)[::-1]) for c in coils[0][:3]])
first_rz = [rz(surfaces[0], np.linspace(0, 2 * np.pi, 361), np.array([p])) for p in planes]
extent = max(float(np.max(np.abs(c[..., :2]))) for c in coils) * 1.02
zext = max(float(np.max(np.abs(c[..., 2]))) for c in coils) * 1.05
qa_hist = np.array([max(h[3]["qa"], 1e-12) for h in history])
k_hist = np.array([h[3].get("sheet_current", np.nan) for h in history])
steps_hist = np.array([h[0] for h in history])


def title(i):
    step, _, _, row = frames[i]
    return (f"{args.label}  step {step}:  QA {row['qa']:.2e}  min|iota| {row['min_abs_iota']:.3f}  "
            f"aspect {row['aspect']:.2f}  K {row.get('sheet_current', float('nan')):.1e}")


# ---- 3D ----------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(7.5, 6.5)) if args.only != "2d" else None
ax = fig.add_subplot(projection="3d") if fig is not None else None


def draw3d(i):
    ax.clear()
    R, Z = rz(surfaces[i], theta, phi3)
    X, Y = R * np.cos(phi3)[None], R * np.sin(phi3)[None]
    ax.plot_surface(X, Y, Z, color="tab:orange", alpha=0.85, linewidth=0, antialiased=False, shade=True)
    for c, curve in enumerate(coils[i]):
        closed = np.vstack([curve, curve[:1]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=plt.cm.tab10(c % 4), lw=1.3)
    ax.set_xlim(-extent, extent), ax.set_ylim(-extent, extent), ax.set_zlim(-zext, zext)
    ax.set_box_aspect((1, 1, zext / extent))
    ax.view_init(elev=32, azim=-60 + 0.6 * i)
    ax.set_axis_off()
    ax.set_title(title(i), fontsize=8)
    return ()


if fig is not None:
    FuncAnimation(fig, draw3d, frames=len(frames), blit=False).save(f"{args.prefix}_3d.gif",
                                                                      writer=PillowWriter(fps=args.fps), dpi=85)
    plt.close(fig)
    print("3d done", flush=True)

# ---- 2D ----------------------------------------------------------------------------------------------------------
if args.only == "3d":
    raise SystemExit
fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
cuts, hist = [axes[0, 0], axes[0, 1], axes[1, 0]], axes[1, 1]


def crossings(curves, plane):
    """(R, Z) where the coils cross the toroidal angle ``plane`` (or plane + pi, the far side)."""
    out = []
    for curve in curves:
        c = np.vstack([curve, curve[:1]])
        phi = np.arctan2(c[:, 1], c[:, 0])
        for target in (plane,):
            d = np.angle(np.exp(1j * (phi - target)))
            idx = np.where((np.sign(d[:-1]) != np.sign(d[1:])) & (np.abs(d[:-1] - d[1:]) < np.pi))[0]
            for j in idx:
                t = d[j] / (d[j] - d[j + 1])
                p = c[j] + t * (c[j + 1] - c[j])
                out.append((np.hypot(p[0], p[1]), p[2]))
    return np.array(out).reshape(-1, 2)


all_R = np.concatenate([rz(s, theta, planes)[0].ravel() for s in surfaces[:: max(1, len(surfaces) // 20)]])
all_Z = np.concatenate([rz(s, theta, planes)[1].ravel() for s in surfaces[:: max(1, len(surfaces) // 20)]])
cross_all = np.vstack([crossings(c, p) for c in coils[:: max(1, len(coils) // 20)] for p in planes])
R_lim = (min(all_R.min(), cross_all[:, 0].min()) - 0.05, max(all_R.max(), cross_all[:, 0].max()) + 0.05)
Z_lim = (min(all_Z.min(), cross_all[:, 1].min()) - 0.05, max(all_Z.max(), cross_all[:, 1].max()) + 0.05)


def draw2d(i):
    tt = np.linspace(0, 2 * np.pi, 361)
    for a, p, (R0, Z0) in zip(cuts, planes, first_rz):
        a.clear()
        R, Z = rz(surfaces[i], tt, np.array([p]))
        a.plot(R0[:, 0], Z0[:, 0], "k--", lw=0.9, label="step 0")
        a.plot(R[:, 0], Z[:, 0], color="tab:orange", lw=1.8, label="LCFS")
        pts = crossings(coils[i], p)
        a.plot(pts[:, 0], pts[:, 1], "o", color="tab:blue", ms=4, label="coils")
        a.set_xlim(*R_lim), a.set_ylim(*Z_lim), a.set_aspect("equal")
        a.set_title(f"phi = {np.degrees(p):.1f} deg (coil {list(planes).index(p) + 1})", fontsize=9)
        a.set_xlabel("R [m]", fontsize=8), a.set_ylabel("Z [m]", fontsize=8)
        a.tick_params(labelsize=7)
    cuts[0].legend(fontsize=7, loc="upper right")
    hist.clear()
    hist.semilogy(steps_hist, qa_hist, color="tab:green", lw=1.2, label="QA")
    hist.semilogy(steps_hist, k_hist, color="tab:red", lw=1.2, label="K (sheet current)")
    hist.axvline(frames[i][0], color="k", lw=0.8)
    hist.set_xlabel("step", fontsize=8), hist.tick_params(labelsize=7), hist.legend(fontsize=7)
    fig.suptitle(title(i), fontsize=9)
    return ()


FuncAnimation(fig, draw2d, frames=len(frames), blit=False).save(f"{args.prefix}_2d.gif",
                                                                  writer=PillowWriter(fps=args.fps), dpi=80)
plt.close(fig)
print("2d done", flush=True)

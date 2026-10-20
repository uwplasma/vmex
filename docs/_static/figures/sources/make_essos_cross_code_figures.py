#!/usr/bin/env python3
"""Draw ``cross_code_losses.webp`` and ``cross_code_bars.webp`` from
``benchmarks/essos_cross_code.json``.

The record is ESSOS's ``examples/cross_code_benchmark/cross_code_benchmark.json``
(uwplasma/ESSOS#162): 256 collisionless 3.5 MeV alphas from s = 0.5 for 10 ms in
the Landreman-Paul QA, ARIES-CS and a scaled W7-X, traced by ESSOS (what
``vmex --trace`` runs), SIMPLE, FIRM3D, SIMSOPT and DESC.  Fate agreement is
taken over the particles on which SIMPLE and FIRM3D agree.

Usage::

    python docs/_static/figures/sources/make_essos_cross_code_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
RECORD = REPO / "benchmarks" / "essos_cross_code.json"
FIGURES = REPO / "docs" / "_static" / "figures"
ORDER = ("ESSOS", "SIMPLE", "FIRM3D", "SIMSOPT", "DESC")
COLORS = dict(zip(ORDER, ("#2a6fdb", "#e0803a", "#3a9e5c", "#8a5cc2", "#c94f6d")))


def agreement(rec):
    lost = {k: np.array([t is not None for t in c["loss_times"]]) for k, c in rec["codes"].items()}
    ref, agree = lost["SIMPLE"], lost["SIMPLE"] == lost["FIRM3D"]
    return {k: float((v == ref[:v.size])[agree[:v.size]].mean()) for k, v in lost.items()}


def save(fig, name):
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(32).save(
        FIGURES / name, lossless=True, method=6)


def main():
    data = json.loads(RECORD.read_text())
    fig, axes = plt.subplots(1, len(data), figsize=(3.2 * len(data), 2.6), sharey=True,
                             constrained_layout=True)
    for ax, (label, rec) in zip(axes, data.items()):
        for name in (k for k in ORDER if k in rec["codes"]):
            code = rec["codes"][name]
            tag = f"{name} ({code['particles']})" if code["particles"] < rec["particles"] else name
            ax.plot(1e3 * np.asarray(rec["times"]), 100 * np.asarray(code["curve"]), label=tag,
                    color=COLORS[name], lw=1.2, ls="--" if name == "ESSOS" else "-")
        ax.set(title=label, xlabel="t [ms]", xscale="log", xlim=(1e3 * rec["times"][1], 1e3 * rec["tmax"]))
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("alphas lost [%]")
    save(fig, "cross_code_losses.webp")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6), constrained_layout=True)
    width = 0.8 / len(ORDER)
    for i, (label, rec) in enumerate(data.items()):
        fates = agreement(rec)
        for k, name in enumerate(ORDER):
            code = rec["codes"].get(name)
            if code is None:
                continue
            x = i + (k - (len(ORDER) - 1) / 2) * width
            values = (code["run_s"] * rec["particles"] / code["particles"], code["energy_error"],
                      None if name in ("SIMPLE", "FIRM3D") else 100 * fates[name])
            for ax, v in zip(axes, values):
                if v is not None:
                    ax.bar(x, v, width, color=COLORS[name], label=name if i == 0 else None)
    for ax, title, scale in zip(axes, ("run time [s]", "max |E/E0 - 1|", "fate agreement [%]"),
                                ("log", "log", "linear")):
        ax.set(title=title, yscale=scale, xticks=range(len(data)), xticklabels=list(data))
    axes[2].set_ylim(80, 100)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "cross_code_bars.webp")


if __name__ == "__main__":
    main()

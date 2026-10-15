"""Render measured development fields from benchmarks/pwo_development.json.

Run from the repository root: python docs/_static/figures/sources/make_pwo_figure.py
This reconstructs stored spectra; it does not rerun the equilibrium optimization.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def main():
    record = json.loads((ROOT / "benchmarks/pwo_development.json").read_text())
    cases = record["cases"]
    spectra = [cases["canonical-vacuum"]["spectra"]["seed"],
               cases["canonical-vacuum"]["spectra"]["final"],
               cases["mapped-vacuum"]["spectra"]["final"]]
    angle = 2 * np.pi * np.arange(96) / 96
    fields = []
    for spectrum in spectra:
        theta, zeta = np.meshgrid(angle, angle / spectrum["nfp"], indexing="ij")
        phase = (theta[..., None] * np.asarray(spectrum["xm_b"])
                 - zeta[..., None] * np.asarray(spectrum["xn_b"]))
        fields.append(np.cos(phase) @ np.asarray(spectrum["bmnc_b"])[0]
                      + np.sin(phase) @ np.asarray(spectrum["bmns_b"])[0])
    levels = np.linspace(np.min(fields), np.max(fields), 30)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    for ax, field, title in zip(axes, fields, ["Seed", "Canonical fit", "Mapped fit"]):
        image = ax.contourf(angle / (2*np.pi), angle / (2*np.pi), field,
                            levels=levels, cmap="viridis")
        ax.set(xlabel=r"$N_{fp}\zeta_B/(2\pi)$", ylabel=r"$\theta_B/(2\pi)$", title=title)
    fig.colorbar(image, ax=list(axes), label="B [T]")
    fig.supxlabel("Vacuum, s=0.5; independently refined equilibria. "
                  "Calibrated canonical target; mapped mirror preserved. No confinement claim.", fontsize=8)
    fig.savefig(ROOT / "docs/_static/figures/pwo_development.webp", dpi=160,
                pil_kwargs={"lossless": True})
    plt.close(fig)


if __name__ == "__main__":
    main()

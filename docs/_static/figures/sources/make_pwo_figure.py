"""Reconstruct unfilled measured contours from the committed benchmark spectra."""

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
    angle = np.linspace(-np.pi, np.pi, 129)
    fields = []
    for spectrum in spectra:
        theta, zeta = np.meshgrid(angle, angle / spectrum["nfp"], indexing="ij")
        phase = (theta[..., None] * np.asarray(spectrum["xm_b"])
                 - zeta[..., None] * np.asarray(spectrum["xn_b"]))
        fields.append(np.cos(phase) @ np.asarray(spectrum["bmnc_b"])[0]
                      + np.sin(phase) @ np.asarray(spectrum["bmns_b"])[0])
    levels = np.linspace(np.min(fields), np.max(fields), 17)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    for ax, field, title in zip(axes, fields, ["Seed", "Canonical fit", "Mapped fit"]):
        image = ax.contour(angle / (2*np.pi), angle / (2*np.pi), field,
                            levels=levels, cmap="viridis", linewidths=0.8)
        ax.set(xlabel=r"$N_{fp}\zeta_B/(2\pi)$", ylabel=r"$\theta_B/(2\pi)$", title=title)
    fig.colorbar(image, ax=list(axes), label="B [T]")
    fig.supxlabel("Vacuum, s=0.5; independently refined equilibria. "
                  "Calibrated canonical target; mapped mirror preserved. No confinement claim.", fontsize=8)
    fig.savefig(ROOT / "docs/_static/figures/pwo_development.webp", dpi=160,
                pil_kwargs={"lossless": True})
    plt.close(fig)


if __name__ == "__main__":
    main()

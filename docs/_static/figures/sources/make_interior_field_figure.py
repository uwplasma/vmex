#!/usr/bin/env python3
"""Draw ``docs/_static/figures/readme_interior_field.webp``.

Reads only ``benchmarks/interior_field_vs_wout.json``: relative B and J errors,
against an exact equilibrium, of the fields stored in a WOUT file and of
:class:`~vmex.core.extender.VmecInteriorField` built from the same file.

Usage::

    python docs/_static/figures/sources/make_interior_field_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
RECORD = REPO / "benchmarks" / "interior_field_vs_wout.json"
OUT = REPO / "docs" / "_static" / "figures" / "readme_interior_field.webp"
WOUT, FIELD, MUTED = "#eb6834", "#2a78d6", "#6b6b68"


def main() -> None:
    rows = json.loads(RECORD.read_text())["surfaces"]
    s = [r["s"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    panels = (
        (axes[0], "B_wout_vmec2000", "B_interior_field", "Magnetic field B",
         "WOUT (VMEX, VMEC2000, VMEC++)"),
        (axes[1], "J_wout_vmec2000", "J_interior_field", "Current density J",
         "WOUT (VMEX, VMEC2000)"),
    )
    for ax, wout, field, title, label in panels:
        ax.semilogy(s, [r[wout] for r in rows], "s-", color=WOUT, lw=2, ms=6, label=label)
        ax.semilogy(s, [r[field] for r in rows], "o-", color=FIELD, lw=2, ms=6,
                    label="VmecInteriorField")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("normalized toroidal flux s")
        ax.grid(True, color="#e4e4e1", lw=0.8)
        ax.legend(frameon=False, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("error relative to the exact field")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, pil_kwargs={"lossless": True})
    print(OUT.name, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

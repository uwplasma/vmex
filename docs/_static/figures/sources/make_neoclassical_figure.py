#!/usr/bin/env python3
"""Regenerate ``readme_neoclassical_output.webp``, the README ``--neoclassical`` figure.

Solves the bundled Landreman-Paul QA beta = 2.5 % bootstrap deck, runs
``vmex --neoclassical`` (default preset, DKX, the deck's published n and T
profiles) on its WOUT and converts the
panel PNG to a 1000 px WebP.  Needs ``pip install "vmex[neoclassical]"``.

Usage::

    python docs/_static/figures/sources/make_neoclassical_figure.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from vmex.core.cli import main as vmex

REPO = Path(__file__).resolve().parents[4]
DECK = REPO / "examples" / "data" / "input.LandremanPaul2021_QA_beta2p5_bootstrap"
PROFILES = REPO / "examples" / "data" / f"kinetic_profiles.{DECK.name.removeprefix('input.')}.json"
OUT = REPO / "docs" / "_static" / "figures" / "readme_neoclassical_output.webp"

with tempfile.TemporaryDirectory() as tmp:
    assert vmex([str(DECK), "--outdir", tmp, "--quiet"]) == 0
    wout = Path(tmp) / f"wout_{DECK.name.removeprefix('input.')}.nc"
    assert vmex([str(wout), "--neoclassical", "--nc-profiles", str(PROFILES), "--outdir", tmp]) == 0
    image = Image.open(Path(tmp) / f"{DECK.name.removeprefix('input.')}_neoclassical.png").convert("RGB")
    image.resize((1000, round(1000 * image.height / image.width)), Image.LANCZOS).save(
        OUT, "WEBP", quality=50, method=6)
print(f"wrote {OUT}")

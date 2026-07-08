#!/usr/bin/env python
"""Coil-only quasi-helical free-boundary optimization example.

This script is a QH preset for ``free_boundary_QS_coil_optimization.py``.  It
uses the same complete-solve acceptance logic and optional validated
branch-local derivative proposal path as the generic QS example.

In ``vmec_jax`` the QH target uses field-period helicity ``m=1, n=-1``; the
full-torus toroidal mode is ``n * nfp`` internally.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "results" / "free_boundary_QH_coil_optimization"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.optimization.free_boundary_QS_coil_optimization import main as qs_main


def main(argv: list[str] | None = None) -> int:
    """Run the shared direct-coil optimizer with QH helicity defaults."""

    preset = [
        "--helicity-m",
        "1",
        "--helicity-n",
        "-1",
        "--outdir",
        str(DEFAULT_OUTDIR),
    ]
    return qs_main([*preset, *(argv or [])])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python
"""Coil-only quasi-axisymmetric free-boundary optimization example.

This script is a QA preset for ``free_boundary_QS_coil_optimization.py``.  It
keeps the example workflow in one implementation while making the common QA run
discoverable:

``direct coils -> complete free-boundary solves -> QA objective -> branch-local derivative proposal report``.

Complete solves remain the acceptance authority; the optional branch-local
derivative path is used only to propose coil steps under an unchanged accepted
branch fingerprint.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "results" / "free_boundary_QA_coil_optimization"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.optimization.free_boundary_QS_coil_optimization import main as qs_main


def main(argv: list[str] | None = None) -> int:
    """Run the shared direct-coil optimizer with QA helicity defaults."""

    preset = [
        "--helicity-m",
        "1",
        "--helicity-n",
        "0",
        "--outdir",
        str(DEFAULT_OUTDIR),
    ]
    return qs_main([*preset, *(argv or [])])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

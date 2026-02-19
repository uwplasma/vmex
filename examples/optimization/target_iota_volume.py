"""Compatibility shim for the explicit boundary optimization example."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name("explicit_target_iota_volume.py"), run_name="__main__")

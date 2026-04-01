from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":  # pragma: no cover
    runpy.run_path(str(Path(__file__).with_name("implicit_target_iota_volume.py")), run_name="__main__")
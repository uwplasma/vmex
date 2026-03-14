from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "tools" / "diagnostics" / "vmec2000_exec_stage_trace_compare.py"
    input_path = repo_root / "examples" / "data" / "input.LandremanPaul2021_QH_reactorScale_lowres"
    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")

    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(input_path),
        "--ns-array",
        "12 31 50",
        "--niter-array",
        "200 200 200",
        "--ftol-array",
        "1e-8 1e-10 1e-12",
        "--use-input-niter",
        "--max-iter",
        "600",
        "--dump-level",
        "none",
        "--rtol",
        "1e-4",
        "--atol",
        "1e-12",
        "--vmec-timeout",
        "60",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()

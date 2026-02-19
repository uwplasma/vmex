from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vmec_jax.vmec2000_exec import find_vmec2000_exec


@pytest.mark.vmec2000
def test_n3are_multigrid_override_stage_parity(tmp_path: Path):
    if os.environ.get("VMEC2000_INTEGRATION", "0") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 integration parity tests")

    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "diagnostics" / "vmec2000_exec_stage_trace_compare.py"
    input_path = repo_root.parent / "simsopt" / "tests" / "test_files" / "input.n3are_R7.75B5.7_lowres"
    if not input_path.exists():
        pytest.skip(f"Missing input file: {input_path}")

    cmd = [
        sys.executable,
        str(script),
        "--input",
        str(input_path),
        "--ns-array",
        "16 31 50",
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
        "--workdir",
        str(tmp_path / "parity"),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

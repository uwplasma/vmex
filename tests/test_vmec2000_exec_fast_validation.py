from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vmec_jax.vmec2000_exec import find_vmec2000_exec


pytestmark = pytest.mark.vmec2000


def test_fast_vmec2000_stage_trace_validation_cases():
    """Optional short executable-backed parity checks for fixed-boundary stages."""

    if os.environ.get("VMEC2000_INTEGRATION") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 executable validation")

    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "diagnostics" / "vmec2000_exec_stage_trace_compare.py"
    cases = [
        (repo_root / "examples" / "data" / "input.circular_tokamak", "1e-3"),
        (repo_root / "examples" / "data" / "input.basic_non_stellsym_pressure", "2e-3"),
    ]

    for input_path, rtol in cases:
        if not input_path.exists():
            pytest.skip(f"Missing bundled input: {input_path}")
        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--vmec2000",
            str(exe),
            "--single-ns",
            "13",
            "--max-iter",
            "2",
            "--vmec-timeout",
            "60",
            "--dump-level",
            "lite",
            "--rtol",
            rtol,
            "--atol",
            "1e-10",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

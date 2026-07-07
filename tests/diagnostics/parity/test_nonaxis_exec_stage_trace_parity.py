from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from vmec_jax.vmec2000_exec import find_vmec2000_exec


@pytest.mark.vmec2000
def test_nonaxis_stage_trace_parity_first_iters():
    if os.environ.get("VMEC2000_INTEGRATION", "0") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 integration parity tests")

    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "tools" / "diagnostics" / "vmec2000_exec_stage_trace_compare.py"
    simsopt_root = repo_root.parent / "simsopt" / "tests" / "test_files"
    cases = [
        (repo_root / "examples" / "data" / "input.nfp4_QH_warm_start", 1),
        (simsopt_root / "input.LandremanPaul2021_QA_reactorScale_lowres", 5),
        (simsopt_root / "input.LandremanPaul2021_QA_lowres", 10),
        (simsopt_root / "input.LandremanPaul2021_QH_reactorScale_lowres", 10),
    ]
    for input_path, max_iter in cases:
        if not input_path.exists():
            continue
        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--single-ns",
            "13",
            "--max-iter",
            str(max_iter),
            "--vmec-timeout",
            "60",
            "--dump-level",
            "lite",
            "--rtol",
            "1e-3",
            "--atol",
            "1e-10",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)

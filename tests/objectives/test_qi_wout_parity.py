from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest

from vmec_jax.vmec2000_exec import find_vmec2000_exec


def _parse_metric(text: str, name: str) -> float:
    pat = re.compile(rf"\s+{re.escape(name)}: max_abs=[0-9.Ee+-]+ max_rel=([0-9.Ee+-]+)")
    m = pat.search(text)
    if not m:
        raise AssertionError(f"Could not find metric '{name}' in comparator output")
    return float(m.group(1))


@pytest.mark.vmec2000
@pytest.mark.parametrize("ns", [35])
def test_qi_wout_parity_smoke(tmp_path, ns):
    if os.getenv("VMEC_JAX_RUN_QI_PARITY", "") not in ("1", "true", "TRUE"):
        pytest.skip("Set VMEC_JAX_RUN_QI_PARITY=1 to run external VMEC2000 QI parity checks.")

    root = Path(__file__).resolve().parents[2]
    input_path = root / "examples" / "data" / "input.nfp2_QI"
    if not input_path.exists():
        pytest.skip("Missing examples/data/input.nfp2_QI")

    vmec_exe_env = os.getenv("VMEC2000_EXEC", "") or os.getenv("VMEC2000_EXE", "")
    vmec_exe = Path(vmec_exe_env) if vmec_exe_env else find_vmec2000_exec(root=root.parent)
    if vmec_exe is None or not vmec_exe.exists():
        pytest.skip(f"Missing VMEC2000 executable: {vmec_exe}")

    cmp_script = root / "tools" / "diagnostics" / "vmec2000_exec_stage_trace_compare.py"
    assert cmp_script.exists()

    workdir = tmp_path / f"qi_parity_ns{ns}"
    cmd = [
        "python",
        str(cmp_script),
        "--input",
        str(input_path),
        "--vmec2000",
        str(vmec_exe),
        "--ns-array",
        str(ns),
        "--niter-array",
        "5000",
        "--ftol-array",
        "1e-11",
        "--max-iter",
        "5000",
        "--vmec-nstep",
        "200",
        "--dump-level",
        "none",
        "--no-fail-fast",
        "--radial-skip",
        "6",
        "--workdir",
        str(workdir),
    ]
    p = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if p.returncode != 0:
        raise AssertionError(f"Comparator failed (ns={ns}).\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")

    dmerc_rel = _parse_metric(p.stdout, "DMerc")
    dgeod_rel = _parse_metric(p.stdout, "Dgeod")
    bsubsmns_rel = _parse_metric(p.stdout, "bsubsmns")

    assert dmerc_rel <= 1e-3, f"DMerc parity regression at ns={ns}: {dmerc_rel}"
    assert dgeod_rel <= 1e-3, f"Dgeod parity regression at ns={ns}: {dgeod_rel}"
    assert bsubsmns_rel <= 1e-3, f"bsubsmns parity regression at ns={ns}: {bsubsmns_rel}"


@pytest.mark.full
@pytest.mark.vmec2000
@pytest.mark.parametrize("ns", [111])
def test_qi_wout_parity_nightly_high_resolution(tmp_path, ns):
    test_qi_wout_parity_smoke(tmp_path, ns)

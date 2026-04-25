from __future__ import annotations

import os
import sys
from pathlib import Path
import subprocess

import numpy as np
import pytest

from vmec_jax.cli import resolve_wout_path
from vmec_jax.config import load_config
from vmec_jax.vmec2000_exec import find_vmec2000_exec, run_xvmec2000
from vmec_jax.wout import read_wout

REPO_ROOT = Path(__file__).resolve().parents[1]
QA_INPUT_ENV = os.environ.get("VMEC_JAX_QA_INPUT", "")
QA_INPUT = Path(QA_INPUT_ENV).expanduser().resolve() if QA_INPUT_ENV else (REPO_ROOT / "examples/data/input.qa_signgs1")


def _rel_rms(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


@pytest.mark.vmec2000
@pytest.mark.parametrize(
    "input_path, is_3d",
    [
        (REPO_ROOT / "examples/data/input.circular_tokamak", False),
        (QA_INPUT, True),
    ],
)
def test_cli_matches_vmec2000_wout(tmp_path, input_path: Path, is_3d: bool):
    if os.environ.get("VMEC2000_INTEGRATION", "0") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 integration parity tests")

    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")

    if not input_path.exists():
        pytest.skip(f"Input file not found: {input_path}")

    cfg, _ = load_config(str(input_path))
    niter_env = os.environ.get("VMEC2000_CLI_NITER", "").strip()
    finite_step_mode = bool(niter_env)

    # VMEC2000 reference. By default this optional integration test compares
    # converged equilibria using the input deck's own multigrid schedule.
    # Setting VMEC2000_CLI_NITER switches to a short single-grid diagnostic.
    indata_updates = {}
    if finite_step_mode:
        niter = int(niter_env)
        indata_updates = {
            "NITER": str(niter),
            "NS_ARRAY": f"{int(cfg.ns)}",
            "NITER_ARRAY": f"{niter}",
        }
    vmec = run_xvmec2000(
        input_path,
        exec_path=exe,
        workdir=tmp_path / "vmec2000",
        timeout_s=240.0,
        indata_updates=indata_updates,
        keep_workdir=True,
    )

    case = input_path.name.split("input.", 1)[-1] if "input." in input_path.name else input_path.stem
    wout_ref_path = vmec.workdir / f"wout_{case}.nc"
    if not wout_ref_path.exists():
        pytest.skip("VMEC2000 wout not produced")

    # vmec_jax CLI run.
    outdir = tmp_path / "vmec_jax"
    cmd = [
        sys.executable,
        "-m",
        "vmec_jax",
        str(input_path),
        "--parity",
        "--outdir",
        str(outdir),
        "--quiet",
    ]
    if finite_step_mode:
        cmd.extend(["--max-iter", str(niter), "--no-multigrid", "--no-use-input-niter"])
    subprocess.run(cmd, check=True)

    wout_cli_path = resolve_wout_path(input_path=input_path, outdir=outdir, output=None)
    assert wout_cli_path.exists(), f"CLI did not write {wout_cli_path}"

    wref = read_wout(wout_ref_path)
    wcli = read_wout(wout_cli_path)

    rtol_axisym = float(os.environ.get("VMEC2000_WOUT_RTOL_AXISYM", "1e-3"))
    rtol_3d = float(os.environ.get("VMEC2000_WOUT_RTOL_3D", "5e-2"))
    rtol = rtol_3d if is_3d else rtol_axisym

    assert _rel_rms(wcli.rmnc, wref.rmnc) < rtol
    assert _rel_rms(wcli.zmns, wref.zmns) < rtol
    assert _rel_rms(wcli.lmns, wref.lmns) < rtol

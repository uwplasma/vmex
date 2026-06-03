from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_bootstrap_current_example_is_explicit_user_workflow() -> None:
    text = (Path(__file__).resolve().parents[1] / "examples" / "bootstrap_current_fixed_point.py").read_text()

    assert "FIXED_POINT_OPTIONS = vj.BootstrapCurrentOptions" in text
    assert "VMEC_RUN_KWARGS = {" in text
    assert "result = vj.bootstrap_current_fixed_point" in text
    assert "vj.write_indata(final_input, result.indata)" in text
    assert "history_json.write_text" in text
    assert "solver_device" in text


def test_bootstrap_current_example_help_does_not_run_workflow() -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "bootstrap_current_fixed_point.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage: bootstrap_current_fixed_point.py" in result.stdout
    assert "Run a bounded VMEC/Redl fixed-point loop" in result.stdout
    assert "Wrote final input" not in result.stdout

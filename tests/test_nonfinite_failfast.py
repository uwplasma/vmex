"""Regression tests for first-iteration NaN/Inf handling."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from vmex.core.errors import VmecNumericalError
from vmex.core.input import VmecInput
from vmex.core.solver import solve
from tools.diagnose_input import diagnose


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.mark.usefixtures("_module_jit_enabled")
def test_zero_effective_toroidal_flux_fails_on_first_iteration() -> None:
    """Zero PHIEDGE used to print NaNs until NITER was exhausted."""
    inp = VmecInput.from_file(DATA / "input.nfp2_QI")
    inp = dataclasses.replace(inp, phiedge=0.0, niter_array=[27])
    with pytest.raises(VmecNumericalError, match="NON-FINITE") as caught:
        solve(inp, max_iterations=27)
    assert caught.value.iteration == 1
    assert "PHIEDGE" in caught.value.hint


def test_shareable_diagnostic_redacts_input_details(capsys) -> None:
    """Default output must be safe to share for a confidential input deck."""
    path = DATA / "input.nfp2_QI"
    assert diagnose(path) == 0
    output = capsys.readouterr().out
    assert "input.nfp2_QI" not in output
    assert "phiedge=" not in output.lower()
    assert "nfp=" not in output.lower()
    assert "R(0)=" not in output
    assert "FSQR=" not in output
    assert "OK_FIRST_FORCE_PASS_FINITE" in output

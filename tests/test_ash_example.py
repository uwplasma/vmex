"""Missing physical evidence must prevent the expensive optimization path."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def example():
    path = Path(__file__).parents[1] / "examples/optimization/QI_optimization_helium_ash.py"
    spec = spec_from_file_location("ash_pilot_example", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_rejects_missing_and_truthy_non_boolean_evidence(example):
    gates = {key: True for key in example.REQUIRED_GATES}
    assert example.checked_verification(gates)["passed"]
    gates["alpha_independent_holdout"] = 1
    assert not example.checked_verification(gates)["passed"]
    del gates["kinetic_response"]
    assert "kinetic_response" in example.checked_verification(gates)["failed"]


def test_unqualified_real_response_records_failure_before_optimizer(example, monkeypatch, tmp_path):
    """A coarse negative-D kinetic response is a rejection, not a zero cost."""
    def transport(eq):
        raise ValueError("unqualified DKX helium diffusion: D=-0.219829 m^2/s; refine grid")
    provider = SimpleNamespace(transport=transport)
    monkeypatch.setattr(example, "load_provider", lambda spec, profile: provider)
    monkeypatch.setattr(example.vj.VmecInput, "from_file", lambda path: object())
    monkeypatch.setattr(example.vj, "read_wout", lambda path: object())
    def no_optimizer(*args, **kwargs):
        pytest.fail("an unqualified response reached the optimizer")
    monkeypatch.setattr(example.opt.VmecProblem, "from_tuples", no_optimizer)
    for name in ("input", "profile", "wout"):
        (tmp_path / name).write_text("fixture\n")
    output = tmp_path / "result"
    code = example.main([
        "--input", str(tmp_path / "input"), "--profile", str(tmp_path / "profile"),
        "--restart-wout", str(tmp_path / "wout"), "--response", "unused:factory",
        "--evaluate-wout-only", "--output", str(output),
    ])
    assert code == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "rejected_seed"
    assert "D=-0.219829" in report["error"]
    assert len(report["input_sha256"]) >= 3


def test_feasible_seed_is_not_reported_as_ash_improvement(example):
    passed = dict(passed=True, report={})
    radial = SimpleNamespace(valid=True, mean_occupation_s=2.0)
    assert example.candidate_status(radial, 2.0, passed)["status"] == "feasible_only"
    radial.mean_occupation_s = 1.0
    assert example.candidate_status(radial, 2.0, passed)["status"] == "feasible_only"
    passed["report"]["ash_relative_gain_lower95"] = 0.4
    assert example.candidate_status(radial, 2.0, passed)["status"] == "accepted"
    passed["passed"] = False
    assert example.candidate_status(radial, 2.0, passed)["status"] == "rejected_candidate"


@pytest.mark.parametrize("valid,cost", [(False, 1.0), (True, float("nan")), (True, -1.0)])
def test_invalid_candidate_never_receives_acceptance(example, valid, cost):
    radial = SimpleNamespace(valid=valid, mean_occupation_s=cost)
    report = dict(passed=True, report={"ash_relative_gain_lower95": 0.5})
    assert example.candidate_status(radial, 2.0, report)["status"] == "rejected_candidate"

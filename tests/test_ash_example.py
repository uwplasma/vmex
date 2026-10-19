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


@pytest.mark.parametrize("missing_metadata", [False, True])
def test_unqualified_real_response_records_failure_before_optimizer(
    example, monkeypatch, tmp_path, missing_metadata,
):
    """A coarse negative-D kinetic response is a rejection, not a zero cost."""
    if missing_metadata:
        from importlib.metadata import PackageNotFoundError

        installed_version = example.version

        def version(name):
            if name in ("essos", "dkx", "solvax"):
                raise PackageNotFoundError(name)
            return installed_version(name)

        monkeypatch.setattr(example, "version", version)
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
    if missing_metadata:
        assert all(report["versions"][name] is None for name in ("essos", "dkx", "solvax"))


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


def test_zero_cost_seed_records_rejection_before_normalization(example, monkeypatch, tmp_path):
    pytest.importorskip("essos.ash")
    # Zero cell cost is a valid radial solve, but cannot normalize an objective.
    provider = SimpleNamespace(transport=lambda eq: dict(
        radial_edges_m=[0., .5, 1.], cell_volume_m3=[.5, .5],
        volume_derivative_m2=[1., 1., 1.], diffusion_m2_s=[1., 1., 1.],
        convection_m_s=[0., 0., 0.], source_m3_s=[1., 1.],
        boundary_kappa_m_s=float("inf"), core_cost=[0., 0.],
    ))
    monkeypatch.setattr(example, "load_provider", lambda spec, profile: provider)
    monkeypatch.setattr(example.vj.VmecInput, "from_file", lambda path: object())
    monkeypatch.setattr(example.vj, "read_wout", lambda path: object())
    for name in ("input", "profile", "wout"):
        (tmp_path / name).write_text("fixture\n")
    output = tmp_path / "result"
    assert example.main([
        "--input", str(tmp_path / "input"), "--profile", str(tmp_path / "profile"),
        "--restart-wout", str(tmp_path / "wout"), "--response", "unused:factory",
        "--evaluate-wout-only", "--output", str(output),
    ]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "rejected_seed"
    assert "finite and positive" in report["error"]


@pytest.mark.parametrize("valid,cost", [(False, 1.0), (True, float("nan")), (True, -1.0)])
def test_invalid_candidate_never_receives_acceptance(example, valid, cost):
    radial = SimpleNamespace(valid=valid, mean_occupation_s=cost)
    report = dict(passed=True, report={"ash_relative_gain_lower95": 0.5})
    assert example.candidate_status(radial, 2.0, report)["status"] == "rejected_candidate"

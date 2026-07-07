from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from conftest import load_python_module


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "examples" / "optimization" / "free_boundary_QA_finite_beta_coil_optimization.py"


def _load_example_module():
    return load_python_module(SCRIPT_PATH, name="free_boundary_qa_finite_beta_coil_optimization", register=False)


def test_qa_finite_beta_wrapper_defaults_to_direct_coil_complete_solve_contract() -> None:
    module = _load_example_module()

    args = module.apply_example_defaults(module.build_parser().parse_args(["--smoke"]))

    assert args.input == ROOT / "examples" / "data" / "input.nfp2_QA_finite_beta"
    assert args.provider == "circle"
    assert args.beta == pytest.approx(2.5)
    assert args.pressure_profile == "standard"
    assert args.helicity_m == 1
    assert args.helicity_n == 0
    assert args.circle_current == pytest.approx(1.0e7)
    assert args.circle_radius == pytest.approx(10.0)
    assert args.max_current_vars == 1
    assert args.max_fourier_vars == 2
    assert args.max_evals == 2
    assert args.same_branch_report_vector_keys == "aspect,qs_total,mean_iota,lcfs_boundary_moment,betatotal"

    metadata = module.finite_beta_qa_metadata(args)
    assert metadata["complete_solve_acceptance_authority"] is True
    assert metadata["plasma_boundary_optimized"] is False
    assert metadata["same_branch_report_vector_keys"].endswith(",betatotal")
    assert "No exact adaptive full-loop gradients" in metadata["gradient_claim"]


def test_qa_finite_beta_wrapper_dry_run_smoke_writes_summary(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--smoke",
            "--dry-run",
            "--outdir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["phase"] == "qa-finite-beta-single-stage-direct-coil-validation"
    assert summary["dry_run"] is True
    assert summary["plasma_boundary_optimized"] is False
    assert summary["vmec_config"]["external_field_provider_kind"] == "direct_coils"
    assert summary["vmec_config"]["beta_percent"] == pytest.approx(2.5)
    assert summary["vmec_config"]["pressure_profile"] == "standard"
    assert summary["provider"]["provider"] == "circle"
    assert summary["provider"]["radius"] == pytest.approx(10.0)
    assert summary["finite_beta_qa_example"]["complete_solve_acceptance_authority"] is True
    assert summary["finite_beta_qa_example"]["helicity_n"] == 0
    assert summary["finite_beta_qa_example"]["same_branch_report_vector_keys"].endswith(",betatotal")
    assert (tmp_path / "input.direct_coil_qs").exists()

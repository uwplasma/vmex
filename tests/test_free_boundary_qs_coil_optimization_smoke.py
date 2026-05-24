from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "optimization" / "free_boundary_QS_coil_optimization.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("free_boundary_qs_coil_optimization_example", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_robust_smooth_cli_risk_maps_to_smooth_max():
    module = _load_example_module()

    assert module.robust_risk_method("mean") == "mean"
    assert module.robust_risk_method("mean_plus_std") == "mean_plus_std"
    assert module.robust_risk_method("smooth") == "smooth_max"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Full coil -> direct-coil free-boundary solve -> Boozer/QS exact "
        "gradient validation is phase 2; current tests validate provider, "
        "projection, dense-vacuum, and dense mode-space adjoint pieces."
    ),
)
def test_full_free_boundary_qs_exact_gradient_validation_phase2_marker():
    raise NotImplementedError("production NESTOR/QS exact-gradient validation is not promoted yet")


def test_robust_circle_smoke_uses_bounded_perturbed_scenarios(tmp_path, monkeypatch):
    pytest.importorskip("jax")
    module = _load_example_module()
    calls = []

    def fake_make_free_boundary_indata(_input_path, output_path, **_kwargs):
        output_path.write_text("&INDATA\n/\n")
        return output_path

    def fake_run_direct_free_boundary(input_path, params, *, vmec_max_iter, activate_fsq):
        calls.append(
            {
                "input_path": input_path,
                "current": float(np.asarray(params.base_currents)[0]),
                "vmec_max_iter": vmec_max_iter,
                "activate_fsq": activate_fsq,
            }
        )
        return SimpleNamespace(), 0.01

    def fake_summarize_run(_run, params, *, objective, wall_s, target_aspect, target_iota):
        current = float(np.asarray(params.base_currents)[0])
        return {
            "objective": objective,
            "wall_s": wall_s,
            "vmec_n_iter": 1,
            "fsqr": current,
            "fsqz": 0.0,
            "fsql": 0.0,
            "residual_proxy": current,
            "aspect": target_aspect,
            "target_aspect": target_aspect,
            "mean_iota": target_iota,
            "target_iota": target_iota,
            "coil_current_norm": abs(current),
            "mean_coil_length": 1.0,
            "vmec_history": {"w": [], "fsqr2": [], "fsqz2": [], "fsql2": []},
        }

    def fake_write_wout(path, _run, *, include_fsq):
        path.write_text(f"include_fsq={include_fsq}\n")

    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fake_run_direct_free_boundary)
    monkeypatch.setattr(module, "summarize_run", fake_summarize_run)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fake_write_wout)

    exit_code = module.main(
        [
            "--smoke",
            "--provider",
            "circle",
            "--max-evals",
            "1",
            "--max-iter",
            "1",
            "--robust-samples",
            "2",
            "--robust-risk",
            "mean_plus_std",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 3
    assert calls[0]["current"] == pytest.approx(2.0)
    assert any(call["current"] != pytest.approx(2.0) for call in calls[1:])

    history = json.loads((tmp_path / "history.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert len(history) == 1
    assert history[0]["summary"]["robust_samples"] == 2
    assert history[0]["summary"]["robust_risk"] == "mean_plus_std"
    assert len(history[0]["summary"]["scenario_objectives"]) == 3
    assert [scenario["scenario"] for scenario in history[0]["scenarios"]] == [
        "nominal",
        "perturbation_0",
        "perturbation_1",
    ]
    assert summary["robust_objective"]["samples"] == 2
    assert summary["robust_objective"]["risk"] == "mean_plus_std"
    assert (tmp_path / "wout_best_direct_coil_phase1.nc").read_text() == "include_fsq=True\n"

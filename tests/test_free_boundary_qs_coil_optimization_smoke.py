from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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


def test_objective_terms_report_weighted_proxy_components():
    module = _load_example_module()

    summary = {
        "residual_proxy": 2.0,
        "aspect": 5.5,
        "target_aspect": 6.0,
        "mean_iota": 0.3,
        "target_iota": 0.4,
    }

    terms = module.objective_terms_from_summary(
        summary,
        residual_weight=3.0,
        aspect_weight=0.5,
        iota_weight=10.0,
    )

    assert terms["residual"]["contribution"] == pytest.approx(6.0)
    assert terms["aspect"]["error"] == pytest.approx(-0.5)
    assert terms["aspect"]["contribution"] == pytest.approx(0.125)
    assert terms["mean_iota"]["contribution"] == pytest.approx(0.1)
    assert terms["total"] == pytest.approx(6.225)
    assert module.objective_from_summary(
        summary,
        residual_weight=3.0,
        aspect_weight=0.5,
        iota_weight=10.0,
    ) == pytest.approx(terms["total"])


def test_objective_terms_report_missing_unweighted_proxy_components():
    module = _load_example_module()

    terms = module.objective_terms_from_summary(
        {
            "residual_proxy": 2.0,
            "aspect": None,
            "target_aspect": 6.0,
            "mean_iota": None,
            "target_iota": 0.4,
        },
        residual_weight=3.0,
        aspect_weight=0.5,
        iota_weight=10.0,
    )

    assert terms["total"] == pytest.approx(6.0)
    assert terms["missing_unweighted_terms"] == ["aspect", "mean_iota"]
    assert terms["aspect"]["contribution"] == pytest.approx(0.0)
    assert terms["mean_iota"]["contribution"] == pytest.approx(0.0)


def test_circle_variable_manifest_and_apply_are_coil_only():
    module = _load_example_module()
    base_params, _metadata = module.make_circle_provider(current_scale=1.0)
    _x0, variables = module.select_coil_variables(
        base_params,
        max_current_vars=1,
        max_fourier_vars=1,
    )

    manifest = module.variable_records(
        variables,
        base_params,
        current_step=0.1,
        dof_step=0.5,
    )
    perturbed = module.apply_coil_variables(
        base_params,
        np.asarray([1.0, -2.0]),
        variables,
        current_step=0.1,
        dof_step=0.5,
    )

    assert [record["kind"] for record in manifest] == ["current", "fourier_dof"]
    assert all(record["kind"] in {"current", "fourier_dof"} for record in manifest)
    assert manifest[0]["parameterization"] == "multiplicative"
    assert manifest[0]["unit_x_delta"] == pytest.approx(0.2)
    assert manifest[1]["parameterization"] == "additive"
    assert manifest[1]["unit_x_delta"] == pytest.approx(0.5)
    assert float(np.asarray(perturbed.base_currents)[0]) == pytest.approx(2.2)
    assert float(np.asarray(base_params.base_currents)[0]) == pytest.approx(2.0)
    assert float(np.asarray(perturbed.base_curve_dofs)[variables[1][1]]) == pytest.approx(0.4)
    assert float(np.asarray(base_params.base_curve_dofs)[variables[1][1]]) == pytest.approx(1.4)
    assert perturbed.n_segments == base_params.n_segments
    assert perturbed.nfp == base_params.nfp
    assert perturbed.stellsym == base_params.stellsym


def test_circle_dry_run_writes_configuration_without_solves(tmp_path, monkeypatch):
    module = _load_example_module()

    def fake_make_free_boundary_indata(_input_path, output_path, **_kwargs):
        output_path.write_text("&INDATA\n/\n")
        return output_path

    def fail_run_direct_free_boundary(*_args, **_kwargs):
        raise AssertionError("dry-run must not call run_direct_free_boundary")

    def fail_sample_coil_perturbations(*_args, **_kwargs):
        raise AssertionError("dry-run must not sample robust perturbations")

    def fail_minimize(*_args, **_kwargs):
        raise AssertionError("dry-run must not call scipy.optimize.minimize")

    def fail_write_wout(*_args, **_kwargs):
        raise AssertionError("dry-run must not write a best wout")

    monkeypatch.setattr(module, "make_free_boundary_indata", fake_make_free_boundary_indata)
    monkeypatch.setattr(module, "run_direct_free_boundary", fail_run_direct_free_boundary)
    monkeypatch.setattr(module, "sample_coil_perturbations", fail_sample_coil_perturbations)
    monkeypatch.setattr(module, "write_wout_from_fixed_boundary_run", fail_write_wout)
    fake_scipy_optimize = ModuleType("scipy.optimize")
    fake_scipy_optimize.minimize = fail_minimize
    monkeypatch.setitem(sys.modules, "scipy.optimize", fake_scipy_optimize)

    exit_code = module.main(
        [
            "--smoke",
            "--dry-run",
            "--provider",
            "circle",
            "--robust-samples",
            "2",
            "--outdir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "history.json").exists()
    assert not (tmp_path / "wout_best_direct_coil_phase1.nc").exists()
    assert (tmp_path / "input.direct_coil_phase1_smoke").read_text() == "&INDATA\n/\n"

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "optimizer" not in summary
    assert "best" not in summary
    assert summary["scope"] == "coil-only direct-coil free-boundary scaffold"
    assert summary["dry_run"] is True
    assert summary["plasma_boundary_optimized"] is False
    assert any("Boozer/QS" in limitation for limitation in summary["wp11_limitations"])
    assert any("full-loop" in limitation for limitation in summary["wp11_limitations"])
    assert summary["provider"]["provider"] == "circle"
    assert summary["baseline_coils"]["n_base_coils"] == 1
    assert summary["vmec_config"]["vmec_max_iter"] == 2
    assert summary["vmec_config"]["jit_forces"] is True
    assert [record["kind"] for record in summary["optimized_variables"]] == ["current", "fourier_dof"]
    assert summary["optimized_variables"][0]["parameterization"] == "multiplicative"
    assert summary["optimized_variables"][1]["parameterization"] == "additive"
    assert summary["robust_objective"]["samples"] == 2
    assert summary["robust_objective"]["scenario_count_including_nominal"] == 3
    assert summary["objective_model"]["target_aspect"] == pytest.approx(6.0)


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

    def fake_run_direct_free_boundary(input_path, params, *, vmec_max_iter, activate_fsq, jit_forces=True):
        calls.append(
            {
                "input_path": input_path,
                "current": float(np.asarray(params.base_currents)[0]),
                "vmec_max_iter": vmec_max_iter,
                "activate_fsq": activate_fsq,
                "jit_forces": bool(jit_forces),
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
    assert all(call["jit_forces"] is True for call in calls)
    assert calls[0]["current"] == pytest.approx(2.0)
    assert any(call["current"] != pytest.approx(2.0) for call in calls[1:])

    history = json.loads((tmp_path / "history.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert len(history) == 1
    assert history[0]["summary"]["robust_samples"] == 2
    assert history[0]["summary"]["robust_risk"] == "mean_plus_std"
    assert len(history[0]["summary"]["scenario_objectives"]) == 3
    assert history[0]["summary"]["nominal_objective_terms"]["residual"]["contribution"] == pytest.approx(2.0)
    assert history[0]["summary"]["scenario_objective_max"] >= history[0]["summary"]["scenario_objective_min"]
    assert history[0]["variables"][0]["parameterization"] == "multiplicative"
    assert history[0]["coil_diagnostics"]["n_base_coils"] == 1
    assert [scenario["scenario"] for scenario in history[0]["scenarios"]] == [
        "nominal",
        "perturbation_0",
        "perturbation_1",
    ]
    assert history[0]["scenarios"][0]["summary"]["objective_terms"]["total"] == pytest.approx(2.0)
    assert summary["robust_objective"]["samples"] == 2
    assert summary["robust_objective"]["risk"] == "mean_plus_std"
    assert summary["dry_run"] is False
    assert summary["baseline_coils"]["n_base_coils"] == 1
    assert summary["optimized_variables"][0]["unit_x_delta"] == pytest.approx(0.04)
    assert (tmp_path / "wout_best_direct_coil_phase1.nc").read_text() == "include_fsq=True\n"

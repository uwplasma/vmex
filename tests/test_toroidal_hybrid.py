from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from importlib import import_module

import numpy as np
import pytest

from vmec_jax.namelist import read_indata, write_indata
import vmec_jax.toroidal_hybrid as toroidal_hybrid
from vmec_jax.toroidal_hybrid import (
    ToroidalHybridBoundarySamples,
    evaluate_toroidal_hybrid_indata_boundary,
    sample_toroidal_stellarator_mirror_hybrid_boundary,
    toroidal_stellarator_mirror_hybrid_indata,
    toroidal_stellarator_mirror_hybrid_metrics,
)


def test_toroidal_hybrid_boundary_is_stellarator_symmetric_and_corner_localized():
    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(ntheta=32, nzeta=32)
    metrics = toroidal_stellarator_mirror_hybrid_metrics(samples)

    assert metrics["min_R"] > 0.0
    assert metrics["stellsym_R_error"] < 1.0e-13
    assert metrics["stellsym_Z_error"] < 1.0e-13
    assert metrics["corner_weight_max"] == 1.0
    assert metrics["side_weight_max"] == 1.0

    side_cols = [0, samples.zeta.size // 2]
    corner_cols = [samples.zeta.size // 4, (3 * samples.zeta.size) // 4]
    side_m2 = np.mean(np.abs(samples.R[:, side_cols] - np.mean(samples.R[:, side_cols], axis=0)))
    corner_m2 = np.mean(np.abs(samples.R[:, corner_cols] - np.mean(samples.R[:, corner_cols], axis=0)))
    assert corner_m2 > 0.5 * side_m2


def test_toroidal_hybrid_localization_powers_sharpen_side_and_corner_regions():
    base = sample_toroidal_stellarator_mirror_hybrid_boundary(ntheta=32, nzeta=32)
    sharp = sample_toroidal_stellarator_mirror_hybrid_boundary(
        ntheta=32,
        nzeta=32,
        side_power=3.0,
        corner_power=3.0,
    )
    metrics = toroidal_stellarator_mirror_hybrid_metrics(sharp)

    assert metrics["stellsym_R_error"] < 1.0e-13
    assert metrics["stellsym_Z_error"] < 1.0e-13
    assert sharp.side_weight.max() == pytest.approx(1.0)
    assert sharp.corner_weight.max() == pytest.approx(1.0)
    assert sharp.side_weight.mean() < base.side_weight.mean()
    assert sharp.corner_weight.mean() < base.corner_weight.mean()


def test_toroidal_hybrid_indata_roundtrips_and_reconstructs_samples(tmp_path: Path):
    sample_kwargs = {
        "side_minor_modulation": 0.16,
        "side_elongation": 0.35,
        "side_power": 2.0,
        "corner_amplitude": 0.025,
        "corner_power": 2.0,
    }
    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(ntheta=32, nzeta=32, **sample_kwargs)
    indata = toroidal_stellarator_mirror_hybrid_indata(
        nfp=2,
        mpol=5,
        ntor=10,
        ntheta_fit=32,
        nzeta_fit=32,
        **sample_kwargs,
    )

    input_path = tmp_path / "input.hybrid"
    write_indata(input_path, indata)
    read_back = read_indata(input_path)
    reconstructed = evaluate_toroidal_hybrid_indata_boundary(read_back, ntheta=32, nzeta=32)

    np.testing.assert_allclose(reconstructed.R, samples.R, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(reconstructed.Z, samples.Z, rtol=0.0, atol=1.0e-12)
    assert read_back.get_int("NFP") == 2
    assert read_back.get_int("MPOL") == 5
    assert read_back.get_int("NTOR") == 10
    assert "RBS" not in read_back.indexed
    assert "ZBC" not in read_back.indexed


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ntheta": 7}, "at least 8"),
        ({"minor_radius": 0.0}, "major_radius"),
        ({"major_radius": 0.1, "minor_radius": 0.2}, "major_radius"),
        ({"corner_helicity": -1}, "nonnegative"),
        (
            {
                "major_radius": 0.3,
                "minor_radius": 0.29,
                "axis_oval": -0.2,
                "side_minor_modulation": 0.3,
                "corner_amplitude": 0.1,
            },
            "nonpositive cylindrical R",
        ),
        ({"side_power": 0.0}, "side_power"),
        ({"corner_power": np.inf}, "corner_power"),
    ],
)
def test_toroidal_hybrid_boundary_rejects_invalid_geometry(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sample_toroidal_stellarator_mirror_hybrid_boundary(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"nfp": 0}, "nfp"),
        ({"mpol": 2}, "mpol"),
        ({"ntor": 2}, "ntor"),
        ({"ntor": 3, "corner_helicity": 2}, "ntor"),
    ],
)
def test_toroidal_hybrid_indata_rejects_invalid_mode_extent(kwargs, match):
    with pytest.raises(ValueError, match=match):
        toroidal_stellarator_mirror_hybrid_indata(**kwargs)


def test_toroidal_hybrid_indata_rejects_non_stellarator_symmetric_samples(monkeypatch):
    def asymmetric_samples(*, ntheta, nzeta, **_kwargs):
        theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
        zeta = np.linspace(0.0, 2.0 * np.pi, int(nzeta), endpoint=False)
        theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")
        return ToroidalHybridBoundarySamples(
            theta=theta,
            zeta=zeta,
            R=1.2 + 0.05 * np.cos(theta2) + 0.01 * np.sin(theta2 + zeta2),
            Z=0.12 * np.sin(theta2),
            side_weight=np.cos(zeta2) ** 2,
            corner_weight=np.sin(zeta2) ** 2,
        )

    monkeypatch.setattr(
        toroidal_hybrid,
        "sample_toroidal_stellarator_mirror_hybrid_boundary",
        asymmetric_samples,
    )

    with pytest.raises(ValueError, match="not stellarator symmetric"):
        toroidal_hybrid.toroidal_stellarator_mirror_hybrid_indata(
            mpol=4,
            ntor=4,
            ntheta_fit=32,
            nzeta_fit=32,
            coeff_tol=1.0e-14,
        )


def test_toroidal_hybrid_example_runs_without_plots(tmp_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{Path.cwd()}{os.pathsep}{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid.py",
            "--outdir",
            str(tmp_path / "hybrid"),
            "--ntheta-fit",
            "32",
            "--nzeta-fit",
            "32",
            "--ntor",
            "10",
            "--side-minor-modulation",
            "0.16",
            "--side-elongation",
            "0.35",
            "--side-power",
            "2.0",
            "--corner-amplitude",
            "0.025",
            "--corner-power",
            "2.0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    metrics_path = Path(completed.stdout.strip())
    metrics = json.loads(metrics_path.read_text())
    assert Path(metrics["input"]).exists()
    assert metrics["figures"] == {}
    assert metrics["stellsym_R_error"] < 1.0e-13
    assert metrics["stellsym_Z_error"] < 1.0e-13
    assert metrics["rbc_count"] > 3
    assert metrics["zbs_count"] > 3
    assert metrics["sample_parameters"]["side_minor_modulation"] == 0.16
    assert metrics["sample_parameters"]["side_elongation"] == 0.35
    assert metrics["ntor"] == 10
    assert metrics["sample_parameters"]["side_power"] == 2.0
    assert metrics["sample_parameters"]["corner_amplitude"] == 0.025
    assert metrics["sample_parameters"]["corner_power"] == 2.0


def test_toroidal_hybrid_convergence_example_runs_without_solve(tmp_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{Path.cwd()}{os.pathsep}{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid_convergence.py",
            "--outdir",
            str(tmp_path / "hybrid_convergence"),
            "--ns-array",
            "7,9",
            "--mode-pairs",
            "5:10",
            "--ntheta-fit",
            "32",
            "--nzeta-fit",
            "32",
            "--side-power",
            "2.0",
            "--corner-power",
            "2.0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    summary_path = Path(completed.stdout.strip())
    summary = json.loads(summary_path.read_text())
    assert len(summary["rows"]) == 2
    assert Path(summary["csv"]).exists()
    assert summary["figures"] == {}
    assert all(not row["ran_solve"] for row in summary["rows"])
    assert all(row["cli_finish"] is True for row in summary["rows"])
    assert all(row["nstep"] == 25 for row in summary["rows"])
    assert all(row["full_solver_diagnostics"] is False for row in summary["rows"])
    assert all(row["diagnostic_step_history_size"] == 0 for row in summary["rows"])
    assert all(row["diagnostic_initial_axis_reset_attempted"] is None for row in summary["rows"])
    assert all(row["diagnostic_initial_axis_reset_reset"] is None for row in summary["rows"])
    assert all(row["initialization_policy"] == "vmec_jax_default_input_boundary" for row in summary["rows"])
    assert all(
        row["vmec_jax_axis_initialization_policy"] == "boundary_inferred_missing_axis" for row in summary["rows"]
    )
    assert all(row["vmec2000_initialization_policy"] == "vmec2000_default_input_boundary" for row in summary["rows"])
    assert all(row["direct_initial_residual_requested"] is True for row in summary["rows"])
    assert all(row["direct_initial_residual_source"] is None for row in summary["rows"])
    assert all(row["direct_initial_fsq"] is None for row in summary["rows"])
    assert all(row["direct_initial_fsq_ratio_vmec2000"] is None for row in summary["rows"])
    assert all(row["initial_fsq_ratio_direct_initial"] is None for row in summary["rows"])
    assert all(row["vmec2000_initial_fsq_ratio_direct_initial"] is None for row in summary["rows"])
    assert all(row["initial_residual_source"] is None for row in summary["rows"])
    assert all(row["vmec2000_initial_residual_source"] is None for row in summary["rows"])
    assert all(row["initial_fsq_ratio_vmec2000"] is None for row in summary["rows"])
    assert all(row["fsq_history"] == [] for row in summary["rows"])
    assert all(row["max_boundary_fit_error"] < 1.0e-12 for row in summary["rows"])
    assert [row["ns"] for row in summary["rows"]] == [7, 9]
    assert summary["shape_cases"][0]["sample_parameters"]["side_power"] == 2.0
    assert summary["shape_cases"][0]["sample_parameters"]["corner_power"] == 2.0
    with Path(summary["csv"]).open(newline="") as file_obj:
        csv_row = next(csv.DictReader(file_obj))
    assert csv_row["initialization_policy"] == "vmec_jax_default_input_boundary"
    assert csv_row["cli_finish"] == "True"
    assert csv_row["nstep"] == "25"
    assert csv_row["full_solver_diagnostics"] == "False"
    assert csv_row["diagnostic_step_history_size"] == "0"
    assert csv_row["diagnostic_initial_axis_reset_attempted"] == ""
    assert csv_row["diagnostic_initial_axis_reset_reset"] == ""
    assert csv_row["vmec_jax_axis_initialization_policy"] == "boundary_inferred_missing_axis"
    assert csv_row["vmec2000_initialization_policy"] == "vmec2000_default_input_boundary"
    assert csv_row["direct_initial_residual_requested"] == "True"
    assert csv_row["direct_initial_residual_source"] == ""
    assert csv_row["direct_initial_fsq_ratio_vmec2000"] == ""
    assert csv_row["initial_fsq_ratio_direct_initial"] == ""
    assert csv_row["vmec2000_initial_fsq_ratio_direct_initial"] == ""
    assert csv_row["initial_residual_source"] == ""
    assert csv_row["vmec2000_initial_residual_source"] == ""
    assert csv_row["initial_fsq_ratio_vmec2000"] == ""


def test_toroidal_hybrid_convergence_example_scans_shape_cases_without_solve(tmp_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{Path.cwd()}{os.pathsep}{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid_convergence.py",
            "--outdir",
            str(tmp_path / "hybrid_shape_scan"),
            "--ns-array",
            "7",
            "--mode-pairs",
            "5:10",
            "--ntheta-fit",
            "32",
            "--nzeta-fit",
            "32",
            "--shape-cases",
            "default,sharp",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    summary_path = Path(completed.stdout.strip())
    summary = json.loads(summary_path.read_text())
    rows = summary["rows"]
    assert [case["name"] for case in summary["shape_cases"]] == ["default", "sharp"]
    assert [row["shape_case"] for row in rows] == ["default", "sharp"]
    assert [row["case"] for row in rows] == ["default_ns007_mpol05_ntor10", "sharp_ns007_mpol05_ntor10"]
    assert rows[0]["side_power"] == 1.0
    assert rows[1]["side_power"] == 2.0
    assert rows[1]["corner_power"] == 2.0
    assert all(row["max_boundary_fit_error"] < 1.0e-12 for row in rows)


def test_toroidal_hybrid_convergence_history_summary_uses_iteration_labels():
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_convergence")
    summary = module._summarize_fsq_history(
        np.asarray([3.0, 2.0, 5.0]),
        iterations=np.asarray([1, 7, 11]),
    )

    assert summary["initial_fsq"] == 3.0
    assert summary["best_fsq"] == 2.0
    assert summary["best_iter"] == 7
    assert summary["fsq_reduction"] == 1.5
    assert summary["final_fsq"] == 5.0
    assert module._parse_shape_cases("default, sharp") == ["default", "sharp"]
    np.testing.assert_array_equal(
        module._row_history_iterations({"iter_history": [3, 5, 9]}, 3),
        np.asarray([3, 5, 9]),
    )
    np.testing.assert_array_equal(
        module._row_history_iterations({"iter_history": [3]}, 3),
        np.asarray([1, 2, 3]),
    )
    diag_fields = module._solver_diagnostic_fields(
        {
            "light_history": False,
            "resume_state_mode": "minimal",
            "multigrid_stage_modes": np.asarray(["accelerated"], dtype=object),
            "multigrid_niter_stages": np.asarray([8], dtype=int),
            "multigrid_stage_offsets": np.asarray([0], dtype=int),
            "iter2_history": np.asarray([1, 2], dtype=int),
            "step_status_history": np.asarray(["accepted", "accepted"], dtype=object),
            "restart_reason_history": np.asarray(["none", "bad_progress"], dtype=object),
            "dt_eff_history": np.asarray([0.1, 0.05], dtype=float),
            "update_rms_history": np.asarray([1.0e-3, 5.0e-4], dtype=float),
            "w_try_ratio_history": np.asarray([0.8, 1.2], dtype=float),
            "bcovar_update_history": np.asarray([1, 0], dtype=int),
            "initial_axis_reset_attempted": True,
            "initial_axis_reset_reset": True,
            "initial_axis_reset_bad_jacobian": True,
            "initial_axis_reset_force_reset": False,
            "initial_axis_reset_fsq": 7.0,
            "initial_axis_reset_ptau_min": -1.0,
            "initial_axis_reset_ptau_max": 2.0,
            "initial_axis_reset_state_tau_min": -0.5,
            "initial_axis_reset_state_tau_max": 1.5,
            "initial_axis_reset_error": None,
        },
        fallback_size=2,
    )
    assert diag_fields["diagnostic_light_history"] is False
    assert diag_fields["diagnostic_resume_state_mode"] == "minimal"
    assert diag_fields["diagnostic_stage_modes"] == ["accelerated"]
    assert diag_fields["diagnostic_step_iter_history"] == [1, 2]
    assert diag_fields["diagnostic_step_history_size"] == 2
    assert diag_fields["diagnostic_step_status_counts"] == {"accepted": 2}
    assert diag_fields["diagnostic_restart_reason_counts"] == {"none": 1, "bad_progress": 1}
    assert diag_fields["diagnostic_bcovar_updates"] == 1
    assert diag_fields["diagnostic_initial_bcovar_update"] is True
    assert diag_fields["diagnostic_final_dt_eff"] == 0.05
    assert diag_fields["diagnostic_max_update_rms"] == 1.0e-3
    assert diag_fields["diagnostic_final_update_rms"] == 5.0e-4
    assert diag_fields["diagnostic_initial_axis_reset_attempted"] is True
    assert diag_fields["diagnostic_initial_axis_reset_reset"] is True
    assert diag_fields["diagnostic_initial_axis_reset_bad_jacobian"] is True
    assert diag_fields["diagnostic_initial_axis_reset_force_reset"] is False
    assert diag_fields["diagnostic_initial_axis_reset_fsq"] == 7.0
    assert diag_fields["diagnostic_initial_axis_reset_ptau_min"] == -1.0
    assert diag_fields["diagnostic_initial_axis_reset_ptau_max"] == 2.0
    assert diag_fields["diagnostic_initial_axis_reset_state_tau_min"] == -0.5
    assert diag_fields["diagnostic_initial_axis_reset_state_tau_max"] == 1.5
    assert diag_fields["diagnostic_initial_axis_reset_error"] is None
    assert module._csv_cell({"accepted": 2}) == '{"accepted": 2}'
    with pytest.raises(ValueError, match="unknown shape"):
        module._parse_shape_cases("unknown")


def test_toroidal_hybrid_axis_initialization_policy_tracks_solver_mode_and_env(monkeypatch):
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_convergence")

    monkeypatch.delenv("VMEC_JAX_ENABLE_AXIS_INFER", raising=False)
    monkeypatch.delenv("VMEC_JAX_DISABLE_AXIS_INFER", raising=False)
    assert module._vmec_jax_axis_initialization_policy("parity") == "raw_input_axis_or_zero"
    assert module._vmec_jax_axis_initialization_policy("accelerated") == "boundary_inferred_missing_axis"

    monkeypatch.setenv("VMEC_JAX_ENABLE_AXIS_INFER", "1")
    assert module._vmec_jax_axis_initialization_policy("parity") == "boundary_inferred_missing_axis"

    monkeypatch.setenv("VMEC_JAX_DISABLE_AXIS_INFER", "1")
    assert module._vmec_jax_axis_initialization_policy("accelerated") == "raw_input_axis_or_zero"


def test_toroidal_hybrid_initial_residual_comparison_ratios():
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_convergence")
    row = {
        "direct_initial_fsq": 3.0,
        "direct_initial_fsqr": 2.0,
        "direct_initial_fsqz": 6.0,
        "direct_initial_fsql": None,
        "initial_fsq": 2.0,
        "vmec2000_initial_fsq": 4.0,
        "vmec2000_initial_fsqr": 2.0,
        "vmec2000_initial_fsqz": 0.0,
        "vmec2000_initial_fsql": 5.0,
        "initial_fsqr": 1.0,
        "initial_fsqz": 3.0,
        "initial_fsql": None,
    }

    module._attach_initial_residual_comparison(row)

    assert row["direct_initial_fsq_ratio_vmec2000"] == 0.75
    assert row["direct_initial_fsqr_ratio_vmec2000"] == 1.0
    assert row["direct_initial_fsqz_ratio_vmec2000"] is None
    assert row["direct_initial_fsql_ratio_vmec2000"] is None
    assert row["initial_fsq_ratio_direct_initial"] == pytest.approx(2.0 / 3.0)
    assert row["vmec2000_initial_fsq_ratio_direct_initial"] == pytest.approx(4.0 / 3.0)
    assert row["initial_fsq_ratio_vmec2000"] == 0.5
    assert row["initial_fsqr_ratio_vmec2000"] == 0.5
    assert row["initial_fsqz_ratio_vmec2000"] is None
    assert row["initial_fsql_ratio_vmec2000"] is None


def test_toroidal_hybrid_direct_initial_residual_helper(monkeypatch, tmp_path: Path):
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_convergence")
    calls = {}

    class DummyWout:
        fsqr = 1.0
        fsqz = 2.0
        fsql = 3.0

    def fake_run_fixed_boundary(path, **kwargs):
        calls["path"] = Path(path)
        calls["run_kwargs"] = kwargs
        return object()

    def fake_wout_from_fixed_boundary_run(run, **kwargs):
        calls["wout_run"] = run
        calls["wout_kwargs"] = kwargs
        return DummyWout()

    monkeypatch.setattr(module.vj, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(module.vj, "wout_from_fixed_boundary_run", fake_wout_from_fixed_boundary_run)

    fields = module._compute_direct_initial_residual(
        tmp_path / "input.case",
        solver_mode="parity",
        use_scan=False,
    )

    assert calls["path"] == tmp_path / "input.case"
    assert calls["run_kwargs"]["use_initial_guess"] is True
    assert calls["run_kwargs"]["solver"] == "vmec2000_iter"
    assert calls["run_kwargs"]["solver_mode"] == "parity"
    assert calls["run_kwargs"]["use_scan"] is False
    assert calls["wout_kwargs"] == {"include_fsq": True, "fast_bcovar": False}
    assert fields["direct_initial_residual_source"] == "vmec_jax_initial_guess_residual_scalars"
    assert fields["direct_initial_axis_initialization_policy"] == "raw_input_axis_or_zero"
    assert fields["direct_initial_fsq"] == 6.0
    assert fields["direct_initial_fsqr"] == 1.0
    assert fields["direct_initial_fsqz"] == 2.0
    assert fields["direct_initial_fsql"] == 3.0

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from vmec_jax.mirror import load_mirror_free_boundary_circular_coil_scan, load_mirror_output
from vmec_jax.mirror.plotting.bfield import mirror_boundary_field_line_data
from vmec_jax.mirror.plotting.diagnostics import mirror_field_line_pitch_profile_data

pytestmark = pytest.mark.mirror


def _load_run_case(script_name: str):
    script = Path("examples/mirror") / script_name
    return runpy.run_path(str(script))["run_case"]


def _load_root_example(script_name: str):
    script = Path("examples") / script_name
    return runpy.run_path(str(script))


def test_mirror_examples_write_readable_outputs_without_plots(tmp_path):
    cases = [
        ("fixed_cylinder.py", {"maxiter": 1, "write_plots": False}),
        ("fixed_flared_tube.py", {"maxiter": 1, "write_plots": False}),
        ("wham_vacuum_boundary.py", {"midplane_radius": 0.25, "maxiter": 1, "write_plots": False}),
        ("nonaxisymmetric_boundary.py", {"epsilon": 0.03, "maxiter": 1, "write_plots": False}),
    ]
    for script_name, kwargs in cases:
        run_case = _load_run_case(script_name)
        mout = run_case(tmp_path / script_name.removesuffix(".py"), **kwargs)
        output = load_mirror_output(mout)
        assert output.attributes["geometry_type"] == "mirror"
        assert output.diagnostics.min_sqrtg > 0.0
        assert output.diagnostics.mirror_ratio >= 1.0


def test_nonaxisymmetric_example_runs_as_standalone_script(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror/nonaxisymmetric_boundary.py",
            "--outdir",
            str(tmp_path / "standalone"),
            "--maxiter",
            "0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mout = Path(completed.stdout.strip())
    output = load_mirror_output(mout)
    assert output.ntheta > 1
    assert output.diagnostics.min_sqrtg > 0.0


def test_root_two_coil_axisym_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_two_coil_axisym.py",
            "--outdir",
            str(tmp_path / "two_coil"),
            "--maxiter",
            "0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mout = Path(completed.stdout.strip())
    output = load_mirror_output(mout)
    metrics = (mout.parent / "two_coil_axisym_metrics.json").read_text()
    assert output.ntheta == 1
    assert output.diagnostics.min_sqrtg > 0.0
    assert output.diagnostics.fsq >= 0.0
    assert output.diagnostics.active_force_dof > 0
    assert "axis_bz_relative_linf" in metrics
    assert "off_axis_br_relative_linf" in metrics


def test_root_finite_current_pitch_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_finite_current_pitch.py",
            "--outdir",
            str(tmp_path / "finite_current_pitch"),
            "--maxiter",
            "0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mout = Path(completed.stdout.strip())
    output = load_mirror_output(mout)
    metrics = json.loads((mout.parent / "finite_current_pitch_metrics.json").read_text())
    lines = mirror_boundary_field_line_data(output, num_lines=2)
    pitch = mirror_field_line_pitch_profile_data(output, num_lines=2)
    theta_advance = lines.theta[:, -1] - lines.theta[:, 0]
    assert output.ntheta == 1
    assert output.diagnostics.min_sqrtg > 0.0
    assert output.diagnostics.active_force_dof > 0
    assert output.profiles.i_prime[0] > 0.0
    assert np.min(theta_advance) > 1.0
    assert np.min(pitch.turns_mean) > 0.0
    assert metrics["field_line_theta_advance_mean"] > 1.0


def test_root_stellarator_hybrid_boundary_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_stellarator_hybrid_boundary.py",
            "--outdir",
            str(tmp_path / "hybrid"),
            "--ns",
            "5",
            "--ntheta",
            "13",
            "--nxi",
            "17",
            "--mpol",
            "4",
            "--maxiter",
            "0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mout = Path(completed.stdout.strip())
    output = load_mirror_output(mout)
    metrics = json.loads((mout.parent / "stellarator_hybrid_boundary_metrics.json").read_text())
    assert output.ntheta == 13
    assert output.diagnostics.min_sqrtg > 0.0
    assert metrics["mirror_end_theta_variation_max"] < 1.0e-12
    assert metrics["midplane_theta_variation"] > 0.01
    assert metrics["hybrid_symmetry_error"] < 1.0e-12
    assert metrics["figures"] == {}


def test_root_free_boundary_circular_coils_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "free_boundary_circular_coils"),
            "--ntheta",
            "8",
            "--nxi",
            "11",
            "--n-segments",
            "64",
            "--run-fixed-boundary-baseline",
            "--baseline-maxiter",
            "0",
            "--run-lcfs-pilot",
            "--lcfs-pilot-steps",
            "1",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    module = _load_root_example("mirror_free_boundary_circular_coils.py")
    schema = module["circular_coil_beta_scan_schema"]()
    module["validate_circular_coil_beta_scan_metrics"](metrics)
    assert metrics["metrics_schema"] == schema["metrics_schema"]
    assert metrics["metrics_schema_version"] == schema["metrics_schema_version"]
    assert metrics["workflow_status"] == "lcfs_pilot"
    assert metrics["free_boundary_solve_status"] == "lcfs_pilot_not_converged_free_boundary"
    assert metrics["external_field_provider_kind"] == "direct_coils"
    assert metrics["coil_format"] == "essos_compatible_circular_fourier"
    assert metrics["fixed_boundary_baseline_count"] == 3
    assert metrics["lcfs_pilot_requested"] is True
    assert metrics["lcfs_pilot_steps_requested"] == 1
    assert metrics["lcfs_pilot_rows_total"] == 3
    assert metrics["lcfs_pilot_accepted_rows_total"] == 3
    assert metrics["lcfs_pilot_skipped_rows_total"] == 0
    assert metrics["lcfs_pilot_target_merit"] == 0.0
    assert metrics["lcfs_pilot_stagnation_rtol"] == 0.0
    assert metrics["lcfs_pilot_fsq_growth_limit"] == 0.0
    assert metrics["lcfs_pilot_stop_reason_counts"] == {"max_steps": 3}
    assert schema["metrics_schema_version"] == "0.3"
    assert "workflow_status_values" in schema
    assert "free_boundary_status_values" in schema
    assert len(metrics["summary_rows"]) == 3
    assert all(set(schema["report_fields"]).issubset(row) for row in metrics["summary_rows"])
    bad_count = dict(metrics)
    bad_count["fixed_boundary_baseline_count"] = metrics["fixed_boundary_baseline_count"] + 1
    with pytest.raises(ValueError, match="fixed_boundary_baseline_count"):
        module["validate_circular_coil_beta_scan_metrics"](bad_count)
    bad_summary_rows = dict(metrics)
    bad_summary_rows["summary_rows"] = metrics["summary_rows"][:-1]
    with pytest.raises(ValueError, match="summary_rows"):
        module["validate_circular_coil_beta_scan_metrics"](bad_summary_rows)
    bad_summary_value = copy.deepcopy(metrics)
    bad_summary_value["summary_rows"][0]["baseline_final_fsq"] = -1.0
    with pytest.raises(ValueError, match="summary row 0 field baseline_final_fsq"):
        module["validate_circular_coil_beta_scan_metrics"](bad_summary_value)
    bad_pilot_summary = copy.deepcopy(metrics)
    bad_pilot_summary["fixed_boundary_baseline_rows"][0]["lcfs_pilot_rows_count"] += 1
    with pytest.raises(ValueError, match="baseline row 0 field lcfs_pilot_rows_count"):
        module["validate_circular_coil_beta_scan_metrics"](bad_pilot_summary)
    bad_pilot_total = dict(metrics)
    bad_pilot_total["lcfs_pilot_rows_total"] = metrics["lcfs_pilot_rows_total"] + 1
    with pytest.raises(ValueError, match="lcfs_pilot_rows_total"):
        module["validate_circular_coil_beta_scan_metrics"](bad_pilot_total)
    bad_stop_counts = dict(metrics)
    bad_stop_counts["lcfs_pilot_stop_reason_counts"] = {"max_steps": 2}
    with pytest.raises(ValueError, match="lcfs_pilot_stop_reason_counts"):
        module["validate_circular_coil_beta_scan_metrics"](bad_stop_counts)
    assert metrics["axis_bz_relative_linf"] < 1.0e-12
    assert metrics["boundary_bmag_min"] > 0.0
    assert metrics["beta_scan_requested_percent"] == [1.0, 3.0, 10.0]
    assert [case["beta_percent"] for case in metrics["beta_cases"]] == [1.0, 3.0, 10.0]
    assert set(schema["top_level_required_fields"]).issubset(metrics)
    assert Path(metrics["summary_csv"]).exists()
    with Path(metrics["summary_csv"]).open(newline="") as stream:
        report_rows = list(csv.DictReader(stream))
    assert [float(row["beta_percent"]) for row in report_rows] == [1.0, 3.0, 10.0]
    assert [float(row["beta_percent"]) for row in metrics["summary_rows"]] == [1.0, 3.0, 10.0]
    assert report_rows[0]["baseline_final_fsq"] == str(metrics["fixed_boundary_baseline_rows"][0]["final_fsq"])
    assert metrics["summary_rows"][0]["baseline_final_fsq"] == metrics["fixed_boundary_baseline_rows"][0]["final_fsq"]
    assert report_rows[0]["pilot_status"] == metrics["fixed_boundary_baseline_rows"][0]["lcfs_pilot_status"]
    assert report_rows[0]["last_accepted_step"] == "1"
    assert report_rows[0]["last_accepted_fsq"] == str(
        metrics["fixed_boundary_baseline_rows"][0]["lcfs_pilot_last_accepted_fsq"]
    )
    setup = load_mirror_free_boundary_circular_coil_scan(metrics["setup_json"])
    assert [case.beta_fraction for case in setup.beta_cases] == [0.01, 0.03, 0.10]
    assert len(metrics["fixed_boundary_baseline_rows"]) == 3
    assert all(set(schema["beta_row_required_fields"]).issubset(row) for row in metrics["fixed_boundary_baseline_rows"])
    assert all(Path(row["mout"]).exists() for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["final_fsq"] >= 0.0 for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_external_bnormal_rms"] >= 0.0 for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pressure_balance_rms"] >= 0.0 for row in metrics["fixed_boundary_baseline_rows"])
    assert all(
        row["lcfs_update_pressure_balance_rms_predicted"] <= row["lcfs_pressure_balance_rms"]
        for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(
        row["lcfs_update_max_relative_delta_radius"] <= 0.05 + 1.0e-14
        for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(row["lcfs_update_strategy"] == "scale_pressure" for row in metrics["fixed_boundary_baseline_rows"])
    assert all(
        row["lcfs_update_allowed_strategies"]
        == ["local_pressure", "scale_pressure", "bnormal_slope", "mixed_scale_bnormal", "noop"]
        for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(row["lcfs_update_rejection_reason"] is None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(
        {candidate["strategy"] for candidate in row["lcfs_update_candidate_summaries"]}
        == {"local_pressure", "scale_pressure", "bnormal_slope", "mixed_scale_bnormal", "noop"}
        for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(
        next(
            candidate["predicted_external_bnormal_rms"]
            for candidate in row["lcfs_update_candidate_summaries"]
            if candidate["strategy"] == "scale_pressure"
        )
        <= next(
            candidate["predicted_external_bnormal_rms"]
            for candidate in row["lcfs_update_candidate_summaries"]
            if candidate["strategy"] == "local_pressure"
        )
        for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(len(row["lcfs_pilot_rows"]) == 1 for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_status"] == "accepted" for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_rows_count"] == 1 for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_accepted_rows"] == 1 for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_skipped_rows"] == 0 for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_final_merit"] is not None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_best_merit"] is not None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_stop_reason"] == "max_steps" for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_final_fsq"] is not None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_best_fsq"] is not None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_final_fsq_growth_ratio"] is not None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_best_fsq_growth_ratio"] is not None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_last_accepted_step"] == 1 for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["lcfs_pilot_last_accepted_merit"] is not None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(
        row["lcfs_pilot_last_accepted_pressure_balance_rms"] is not None
        for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(row["lcfs_pilot_last_accepted_fsq"] is not None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(
        row["lcfs_pilot_last_accepted_fsq_growth_ratio"] is not None for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(
        row["lcfs_pilot_last_accepted_normalized_force"] is not None for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(row["lcfs_pilot_final_normalized_force"] is not None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(
        row["lcfs_pilot_final_pressure_balance_rms"] is not None for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(Path(row["lcfs_pilot_rows"][0]["mout"]).exists() for row in metrics["fixed_boundary_baseline_rows"])
    assert all(
        row["lcfs_pilot_rows"][0]["lcfs_pressure_balance_rms"] >= 0.0 for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(
        set(schema["pilot_row_required_fields"]).issubset(row["lcfs_pilot_rows"][0])
        for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(
        row["lcfs_pilot_rows"][0]["fsq_growth_ratio"] == pytest.approx(row["lcfs_pilot_final_fsq_growth_ratio"])
        for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(
        isinstance(row["lcfs_pilot_rows"][0]["accepted"], bool) for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(row["lcfs_pilot_rows"][0]["accepted"] for row in metrics["fixed_boundary_baseline_rows"])
    assert all(
        row["lcfs_pilot_rows"][0]["stop_reason"] == "max_steps" for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(
        row["lcfs_pilot_rows"][0]["lcfs_pressure_balance_rms"] <= row["lcfs_pressure_balance_rms"]
        for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert all(
        row["lcfs_pilot_rows"][0]["lcfs_merit"] <= row["lcfs_merit"] for row in metrics["fixed_boundary_baseline_rows"]
    )
    assert metrics["figures"] == {}


def test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "strict_guard"),
            "--betas",
            "1",
            "--ntheta",
            "8",
            "--nxi",
            "11",
            "--n-segments",
            "64",
            "--run-fixed-boundary-baseline",
            "--baseline-maxiter",
            "0",
            "--run-lcfs-pilot",
            "--lcfs-pilot-steps",
            "1",
            "--lcfs-require-bnormal-nonincrease",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    assert metrics["workflow_status"] == "lcfs_pilot"
    assert metrics["lcfs_pilot_rows_total"] == 1
    assert metrics["lcfs_pilot_accepted_rows_total"] == 0
    assert metrics["lcfs_pilot_skipped_rows_total"] == 1
    assert metrics["lcfs_pilot_stop_reason_counts"] == {"noop_candidate": 1}
    row = metrics["fixed_boundary_baseline_rows"][0]
    pilot = row["lcfs_pilot_rows"][0]
    assert row["lcfs_update_strategy"] == "noop"
    assert row["lcfs_pilot_status"] == "skipped"
    assert row["lcfs_pilot_stop_reason"] == "noop_candidate"
    assert row["lcfs_pilot_rows_count"] == 1
    assert row["lcfs_pilot_accepted_rows"] == 0
    assert row["lcfs_pilot_skipped_rows"] == 1
    assert row["lcfs_update_normal_field_guard"] is True
    assert row["lcfs_update_allowed_strategies"] == ["noop"]
    assert row["lcfs_update_rejection_reason"] == "normal_field_guard_no_candidate"
    assert pilot["skipped"] is True
    assert pilot["accepted"] is False
    assert pilot["rejection_reason"] == "normal_field_guard_no_candidate"
    assert pilot["stop_reason"] == "noop_candidate"
    assert pilot["fsq_growth_ratio"] is None
    assert row["lcfs_pilot_final_fsq_growth_ratio"] is None
    assert row["lcfs_pilot_best_fsq_growth_ratio"] is None
    assert row["lcfs_pilot_last_accepted_step"] is None
    assert row["lcfs_pilot_last_accepted_merit"] is None
    assert row["lcfs_pilot_last_accepted_fsq_growth_ratio"] is None
    assert pilot["lcfs_update_allowed_strategies_next"] == ["noop"]
    assert pilot["lcfs_update_rejection_reason_next"] == "normal_field_guard_no_candidate"
    assert pilot["mout"] is None


def test_root_free_boundary_circular_coils_pilot_stagnation_stops_early(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "stagnation_stop"),
            "--betas",
            "1",
            "--ntheta",
            "8",
            "--nxi",
            "11",
            "--n-segments",
            "64",
            "--run-fixed-boundary-baseline",
            "--baseline-maxiter",
            "0",
            "--run-lcfs-pilot",
            "--lcfs-pilot-steps",
            "3",
            "--lcfs-pilot-stagnation-rtol",
            "1.0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    row = metrics["fixed_boundary_baseline_rows"][0]
    assert metrics["lcfs_pilot_steps_requested"] == 3
    assert metrics["lcfs_pilot_stagnation_rtol"] == 1.0
    assert metrics["lcfs_pilot_fsq_growth_limit"] == 0.0
    assert metrics["lcfs_pilot_rows_total"] == 1
    assert metrics["lcfs_pilot_stop_reason_counts"] == {"merit_stagnation": 1}
    assert row["lcfs_pilot_rows_count"] == 1
    assert row["lcfs_pilot_stop_reason"] == "merit_stagnation"
    assert row["lcfs_pilot_final_fsq"] is not None
    assert row["lcfs_pilot_best_fsq"] is not None
    assert row["lcfs_pilot_final_fsq_growth_ratio"] is not None
    assert row["lcfs_pilot_best_fsq_growth_ratio"] is not None
    assert row["lcfs_pilot_last_accepted_step"] == 1
    assert row["lcfs_pilot_last_accepted_fsq_growth_ratio"] == pytest.approx(row["lcfs_pilot_final_fsq_growth_ratio"])
    assert row["lcfs_pilot_rows"][0]["stop_reason"] == "merit_stagnation"
    assert row["lcfs_pilot_rows"][0]["fsq_growth_ratio"] == pytest.approx(row["lcfs_pilot_final_fsq_growth_ratio"])
    assert row["lcfs_pilot_rows"][0]["lcfs_merit_improvement_fraction"] <= 1.0


def test_root_free_boundary_circular_coils_fsq_growth_guard_rejects_pilot(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "fsq_guard"),
            "--betas",
            "3",
            "--ntheta",
            "8",
            "--nxi",
            "11",
            "--n-segments",
            "64",
            "--run-fixed-boundary-baseline",
            "--baseline-maxiter",
            "5",
            "--run-lcfs-pilot",
            "--lcfs-pilot-steps",
            "5",
            "--lcfs-pilot-fsq-growth-limit",
            "1.0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    row = metrics["fixed_boundary_baseline_rows"][0]
    pilot = row["lcfs_pilot_rows"][0]
    assert metrics["lcfs_pilot_fsq_growth_limit"] == 1.0
    assert metrics["lcfs_pilot_rows_total"] == 1
    assert metrics["lcfs_pilot_accepted_rows_total"] == 0
    assert metrics["lcfs_pilot_stop_reason_counts"] == {"fsq_growth_guard": 1}
    assert row["lcfs_pilot_status"] == "rejected"
    assert row["lcfs_pilot_stop_reason"] == "fsq_growth_guard"
    assert pilot["accepted"] is False
    assert pilot["rejection_reason"] == "fsq_growth_guard"
    assert pilot["stop_reason"] == "fsq_growth_guard"
    assert pilot["final_fsq"] > row["final_fsq"]
    assert pilot["fsq_growth_ratio"] > 1.0
    assert row["lcfs_pilot_final_fsq_growth_ratio"] == pytest.approx(pilot["fsq_growth_ratio"])
    assert row["lcfs_pilot_best_fsq_growth_ratio"] == pytest.approx(pilot["fsq_growth_ratio"])
    assert row["lcfs_pilot_last_accepted_step"] is None
    assert row["lcfs_pilot_last_accepted_fsq_growth_ratio"] is None


def test_root_free_boundary_circular_coils_tolerant_fsq_guard_keeps_last_accepted(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "tolerant_fsq_guard"),
            "--betas",
            "3",
            "--ntheta",
            "8",
            "--nxi",
            "11",
            "--n-segments",
            "64",
            "--run-fixed-boundary-baseline",
            "--baseline-maxiter",
            "5",
            "--run-lcfs-pilot",
            "--lcfs-pilot-steps",
            "2",
            "--lcfs-pilot-fsq-growth-limit",
            "1.1",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    row = metrics["fixed_boundary_baseline_rows"][0]
    first, second = row["lcfs_pilot_rows"]
    assert metrics["lcfs_pilot_fsq_growth_limit"] == 1.1
    assert metrics["lcfs_pilot_rows_total"] == 2
    assert metrics["lcfs_pilot_accepted_rows_total"] == 1
    assert row["lcfs_pilot_status"] == "rejected"
    assert row["lcfs_pilot_accepted_rows"] == 1
    assert row["lcfs_pilot_last_accepted_step"] == 1
    assert first["accepted"] is True
    assert first["fsq_growth_ratio"] < 1.1
    assert second["accepted"] is False
    assert second["rejection_reason"] == "fsq_growth_guard"
    assert second["fsq_growth_ratio"] > 1.1
    assert row["lcfs_pilot_last_accepted_fsq"] == pytest.approx(first["final_fsq"])
    assert row["lcfs_pilot_last_accepted_fsq_growth_ratio"] == pytest.approx(first["fsq_growth_ratio"])
    assert row["lcfs_pilot_final_fsq"] == pytest.approx(second["final_fsq"])
    assert row["lcfs_pilot_final_fsq_growth_ratio"] == pytest.approx(second["fsq_growth_ratio"])
    summary = metrics["summary_rows"][0]
    assert summary["pilot_status"] == "rejected"
    assert summary["last_accepted_step"] == 1
    assert summary["last_accepted_fsq"] == pytest.approx(first["final_fsq"])
    assert summary["last_accepted_fsq_growth_ratio"] == pytest.approx(first["fsq_growth_ratio"])
    assert summary["final_trial_fsq"] == pytest.approx(second["final_fsq"])
    assert summary["final_trial_fsq_growth_ratio"] == pytest.approx(second["fsq_growth_ratio"])


def test_root_free_boundary_circular_coils_coupled_mode_scores_realized_trials(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "coupled_lcfs"),
            "--betas",
            "3",
            "--ntheta",
            "8",
            "--nxi",
            "11",
            "--n-segments",
            "64",
            "--run-fixed-boundary-baseline",
            "--baseline-maxiter",
            "0",
            "--run-lcfs-pilot",
            "--lcfs-pilot-steps",
            "1",
            "--lcfs-proposal-mode",
            "coupled",
            "--lcfs-coupled-fsq-weight",
            "1.0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    row = metrics["fixed_boundary_baseline_rows"][0]
    pilot = row["lcfs_pilot_rows"][0]
    trial_rows = pilot["coupled_trial_rows"]
    selected = [trial for trial in trial_rows if trial["selected"]]
    assert metrics["lcfs_coupled_fsq_weight"] == 1.0
    assert row["lcfs_pilot_status"] == "accepted"
    assert len(selected) == 1
    assert selected[0]["strategy"] == "bnormal_slope"
    assert selected[0]["score"] == pytest.approx(min(trial["score"] for trial in trial_rows))
    assert pilot["coupled_score"] == pytest.approx(selected[0]["score"])
    assert pilot["coupled_merit_ratio"] == pytest.approx(selected[0]["merit_ratio"])
    assert pilot["coupled_fsq_penalty"] == pytest.approx(selected[0]["fsq_penalty"])
    assert pilot["fsq_growth_ratio"] == pytest.approx(selected[0]["fsq_growth_ratio"])
    assert selected[0]["fsq_growth_ratio"] < 1.01
    assert selected[0]["accepted_by_merit"] is True
    assert selected[0]["mout"] == pilot["mout"]


def test_root_fixed_boundary_solve_diagnostic_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_fixed_boundary_solve_diagnostic.py",
            "--outdir",
            str(tmp_path / "solve_diagnostic"),
            "--ns-array",
            "7",
            "--nxi",
            "13",
            "--maxiter",
            "2",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    rows = json.loads(path.read_text())
    assert len(rows) == 1
    assert rows[0]["ns"] == 7
    assert rows[0]["optimizer"] == "lbfgs"
    assert rows[0]["residual_linear_maxiter"] == 16
    assert rows[0]["residual_linear_maxiter_policy"] == "adaptive"
    assert rows[0]["residual_linear_solver"] == "lsmr"
    assert "residual_linear_istop_last" in rows[0]
    assert rows[0]["residual_compare_dense_step"] is False
    assert rows[0]["residual_preconditioner"] == "radial_xi_tridi"
    assert rows[0]["residual_xi_alpha"] == pytest.approx(0.2)
    assert rows[0]["optimizer_nit"] <= 2
    assert "optimizer_rejection_reason" in rows[0]
    assert "optimizer_candidate_min_sqrtg" in rows[0]
    assert rows[0]["final_residual_norm"] >= 0.0
    assert Path(rows[0]["mout"]).exists()


def test_root_manufactured_fixed_boundary_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_manufactured_fixed_boundary.py",
            "--outdir",
            str(tmp_path / "manufactured"),
            "--ns",
            "5",
            "--nxi",
            "9",
            "--maxiter",
            "20",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert metrics["optimizer_success"]
    assert metrics["reached_projected_gtol"]
    assert metrics["final_residual_norm"] < 1.0e-12
    assert metrics["final_exact_error_norm"] < 1.0e-10


def test_root_implicit_sensitivity_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_implicit_sensitivity.py",
            "--outdir",
            str(tmp_path / "implicit_sensitivity"),
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert metrics["vector_size"] > 0
    assert metrics["root_residual_norm"] < 1.0e-12
    assert metrics["perturbed_root_success"]
    assert metrics["perturbed_residual_norm"] < 1.0e-10
    assert metrics["relative_error"] < 1.0e-3
    assert metrics["custom_vjp_adjoint_relative_error"] < 1.0e-8
    assert metrics["custom_vjp_directional_relative_error"] < 1.0e-3
    assert metrics["accepted"]
    assert metrics["figures"] == {}


def test_root_implicit_parameter_gradients_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_implicit_parameter_gradients.py",
            "--outdir",
            str(tmp_path / "implicit_parameter_gradients"),
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert metrics["vector_size"] > 0
    assert metrics["root_residual_norm"] < 1.0e-12
    assert metrics["families"] == ["source", "pressure", "current", "flux", "boundary"]
    assert metrics["accepted"]
    assert metrics["figures"] == {}
    for row in metrics["rows"]:
        assert row["accepted"]
        assert row["perturbed_residual_norm"] < 1.0e-10
        assert row["custom_vs_forward_relative_error"] < 1.0e-8
        assert row["custom_vs_finite_difference_relative_error"] < 1.0e-3


def test_root_implicit_solve_benchmark_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_implicit_solve_benchmark.py",
            "--outdir",
            str(tmp_path / "implicit_solve_benchmark"),
            "--ns-array",
            "5",
            "--nxi-array",
            "7",
            "--repeat",
            "1",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert metrics["accepted"]
    assert metrics["figures"] == {}
    assert Path(metrics["csv"]).exists()
    assert {row["method"] for row in metrics["rows"]} == {"dense", "matrix_free_cg"}
    assert all(row["vector_size"] == 45 for row in metrics["rows"])
    matrix_free_rows = [row for row in metrics["rows"] if row["method"] == "matrix_free_cg"]
    assert matrix_free_rows
    assert max(row["relative_error_vs_dense"] for row in matrix_free_rows) < 1.0e-5


def test_root_solver_comparison_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_solver_comparison.py",
            "--outdir",
            str(tmp_path / "solver_comparison"),
            "--cases",
            "cylinder,two_coil,manufactured",
            "--maxiter-gd",
            "1",
            "--maxiter-lbfgs",
            "2",
            "--maxiter-newton",
            "2",
            "--two-coil-ns",
            "5",
            "--two-coil-nxi",
            "9",
            "--residual-linear-maxiter",
            "12",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    row_keys = {(row["case"], row["optimizer"], row["solver_scope"]) for row in metrics["rows"]}
    assert ("cylinder", "gradient_descent", "production_fixed_boundary") in row_keys
    assert ("cylinder", "lbfgs", "production_fixed_boundary") in row_keys
    assert ("cylinder", "residual_newton", "production_fixed_boundary") in row_keys
    assert ("two_coil", "residual_newton", "production_fixed_boundary") in row_keys
    assert ("manufactured", "residual_newton", "manufactured_source_validation") in row_keys
    assert len(metrics["histories"]) == len(metrics["rows"])
    assert all(row["final_residual_norm"] >= 0.0 for row in metrics["rows"])
    production_rows = [row for row in metrics["rows"] if row["solver_scope"] == "production_fixed_boundary"]
    assert all(row["residual_preconditioner"] == "radial_xi_tridi" for row in production_rows)
    assert all(row["residual_linear_maxiter_policy"] == "adaptive" for row in production_rows)
    assert all(row["residual_linear_solver"] == "lsmr" for row in production_rows)
    assert all("residual_linear_iterations_total" in row for row in production_rows)
    assert all(row["residual_compare_dense_step"] is False for row in production_rows)


def test_root_residual_newton_convergence_grid_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_residual_newton_convergence_grid.py",
            "--outdir",
            str(tmp_path / "convergence_grid"),
            "--ns-array",
            "5",
            "--nxi-array",
            "9",
            "--maxiter-array",
            "2",
            "--residual-linear-maxiter-array",
            "8",
            "--preconditioners",
            "radial_xi_tridi",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert len(metrics["rows"]) == 1
    assert len(metrics["histories"]) == 1
    row = metrics["rows"][0]
    assert row["ns"] == 5
    assert row["nxi"] == 9
    assert row["optimizer"] == "residual_newton"
    assert row["residual_preconditioner"] == "radial_xi_tridi"
    assert row["residual_linear_maxiter"] == 8
    assert row["residual_linear_maxiter_policy"] == "fixed"
    assert row["residual_linear_solver"] == "lsmr"
    assert row["residual_linear_istop_last"] is not None
    assert row["residual_linear_iterations_last"] is not None
    assert row["residual_compare_dense_step"] is False
    assert row["residual_linear_maxiter_effective_max"] == 8
    assert row["final_residual_norm"] >= 0.0
    assert row["component_norm"] == pytest.approx(row["final_residual_norm"])
    assert row["component_active_dof"] > 0
    assert row["residual_a_norm"] >= 0.0
    assert row["residual_lam_norm"] >= 0.0
    assert row["residual_a_cap_adjacent_norm"] >= 0.0
    assert row["residual_lam_interior_xi_norm"] >= 0.0
    assert metrics["histories"][0]["row_id"] == row["row_id"]


def test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_residual_newton_convergence_grid.py",
            "--outdir",
            str(tmp_path / "finite_current_convergence_grid"),
            "--ns-array",
            "5",
            "--nxi-array",
            "9",
            "--maxiter-array",
            "1",
            "--residual-linear-maxiter-array",
            "8",
            "--residual-linear-maxiter-policy",
            "adaptive",
            "--i-prime",
            "0.01",
            "--preconditioners",
            "radial_xi_lambda_xi_tridi",
            "--residual-xi-alpha",
            "1.0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert metrics["case_label"] == "finite_current_two_coil"
    assert metrics["i_prime_value"] == pytest.approx(0.01)
    row = metrics["rows"][0]
    assert row["finite_current"]
    assert row["i_prime_value"] == pytest.approx(0.01)
    assert row["residual_preconditioner"] == "radial_xi_lambda_xi_tridi"
    assert row["residual_xi_alpha"] == pytest.approx(1.0)
    assert row["twist_proxy_i_prime_over_psi_prime"] > 0.0
    assert row["residual_lam_norm"] > 0.0
    assert row["residual_lam_fraction"] > 0.0
    assert row["residual_linear_maxiter_policy"] == "adaptive"
    assert row["residual_linear_solver"] == "lsmr"
    assert row["residual_linear_maxiter_effective_max"] >= 8
    assert row["residual_linear_iterations_last"] is not None
    assert row["residual_compare_dense_step"] is False

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


def _assert_nonblank_image(path: str | Path, image_module, *, min_std: float = 1.0e-4) -> None:
    pixels = image_module.imread(path)
    assert pixels.ndim in (2, 3)
    assert float(np.std(pixels)) > float(min_std)


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
    metrics = json.loads((mout.parent / "two_coil_axisym_metrics.json").read_text())
    assert output.ntheta == 1
    assert output.diagnostics.min_sqrtg > 0.0
    assert output.diagnostics.fsq >= 0.0
    assert output.diagnostics.active_force_dof > 0
    assert "axis_bz_relative_linf" in metrics
    assert "off_axis_br_relative_linf" in metrics
    assert metrics["boozer_like_surface_mirror_ratio_max"] >= 1.0
    assert metrics["boozer_like_field_line_turns_mean"] == pytest.approx(0.0)


def test_root_two_coil_axisym_example_writes_nonblank_benchmark_plots(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_two_coil_axisym.py",
            "--outdir",
            str(tmp_path / "two_coil_plots"),
            "--ns",
            "5",
            "--nxi",
            "9",
            "--maxiter",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mout = Path(completed.stdout.strip())
    metrics = json.loads((mout.parent / "two_coil_axisym_metrics.json").read_text())
    convergence_rows = json.loads((mout.parent / "two_coil_axisym_convergence.json").read_text())
    assert metrics["axis_bz_relative_linf"] < 1.0e-12
    assert "off_axis_br_relative_linf" in metrics
    assert "off_axis_bz_relative_linf" in metrics
    assert len(convergence_rows) == 3
    assert [row["ns"] for row in convergence_rows] == [7, 9, 11]
    assert [row["nxi"] for row in convergence_rows] == [17, 25, 33]

    figure_dir = mout.parent / "figures"
    for name in [
        "two_coil_axisym_axis_bz_comparison.png",
        "two_coil_axisym_geometry_with_coils.png",
        "two_coil_axisym_bmag_with_coils.png",
        "two_coil_axisym_off_axis_biot_savart_comparison.png",
        "two_coil_axisym_convergence.png",
        "two_coil_axisym_mirror_boundary_3d.png",
        "two_coil_axisym_mirror_boozer_like_diagnostics.png",
    ]:
        _assert_nonblank_image(figure_dir / name, image)


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
    assert metrics["boozer_like_field_line_turns_mean"] > 0.0
    assert metrics["boozer_like_contravariant_pitch_rms_max"] >= 0.0


def test_root_finite_current_pitch_example_writes_nonblank_field_line_plots(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_finite_current_pitch.py",
            "--outdir",
            str(tmp_path / "finite_current_pitch_plots"),
            "--ns",
            "5",
            "--nxi",
            "9",
            "--maxiter",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mout = Path(completed.stdout.strip())
    figure_dir = mout.parent / "figures"
    for name in [
        "finite_current_pitch_theta_advance.png",
        "finite_current_pitch_geometry_coils_field_lines.png",
        "finite_current_pitch_mirror_boozer_like_diagnostics.png",
    ]:
        path = figure_dir / name
        assert path.exists()
        assert path.stat().st_size > 4096
        _assert_nonblank_image(path, image, min_std=0.0)


def test_root_free_boundary_vector_ls_benchmark_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_vector_ls_benchmark.py",
            "--outdir",
            str(tmp_path / "vector_ls_benchmark"),
            "--nxi",
            "9",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    module = _load_root_example("mirror_free_boundary_vector_ls_benchmark.py")
    module["validate_vector_ls_benchmark_metrics"](metrics)
    rows = {row["name"]: row for row in metrics["rows"]}
    solve_rows = {row["name"]: row for row in metrics["solve_rows"]}

    assert metrics["metrics_schema_version"] == "0.4"
    assert set(rows) == {"finite_difference", "jax_forward", "jax_reverse", "jax_auto"}
    assert set(solve_rows) == set(rows)
    assert metrics["figures"] == {}
    assert all(row["accepted"] for row in rows.values())
    assert all(row["converged"] for row in solve_rows.values())
    assert all(row["line_search_factor"] == pytest.approx(1.0) for row in rows.values())
    assert all(row["stop_reason"] == "target_residual" for row in solve_rows.values())
    np.testing.assert_allclose(
        rows["jax_forward"]["coefficients_new"],
        rows["finite_difference"]["coefficients_new"],
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        rows["jax_reverse"]["coefficients_new"],
        rows["jax_forward"]["coefficients_new"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert all(row["residual_reduction_fraction"] > 0.0 for row in rows.values())
    assert rows["finite_difference"]["selected_jax_mode"] is None
    assert rows["jax_forward"]["selected_jax_mode"] == "forward"
    assert rows["jax_reverse"]["selected_jax_mode"] == "reverse"
    assert rows["jax_auto"]["selected_jax_mode"] == "forward"
    assert all(row["jacobian_rank"] == 3 for row in rows.values())
    assert all(row["jacobian_nullity"] == 0 for row in rows.values())
    assert all(row["jacobian_condition"] >= 1.0 for row in rows.values())
    assert all(row["actual_reduction_fraction"] > 0.0 for row in rows.values())
    assert all(row["ridge"] == pytest.approx(0.0) for row in rows.values())
    assert all(row["ridge_candidates"] == [0.0] for row in rows.values())
    assert set(solve_rows["jax_auto"]["selected_jax_mode_history"]) == {"forward"}
    assert all(row["jacobian_rank_history"] for row in solve_rows.values())
    assert all(row["ridge_history"] for row in solve_rows.values())
    assert all(row["ridge_candidates_history"] for row in solve_rows.values())
    np.testing.assert_allclose(
        solve_rows["jax_auto"]["coefficients_final"],
        solve_rows["finite_difference"]["coefficients_final"],
        rtol=1.0e-9,
        atol=1.0e-9,
    )


def test_root_free_boundary_vector_ls_benchmark_writes_nonblank_plots(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_vector_ls_benchmark.py",
            "--outdir",
            str(tmp_path / "vector_ls_benchmark_plots"),
            "--nxi",
            "9",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    module = _load_root_example("mirror_free_boundary_vector_ls_benchmark.py")
    module["validate_vector_ls_benchmark_metrics"](metrics)
    assert set(metrics["figures"]) == {"summary", "radius_profiles", "solve_residual_history"}
    for path in metrics["figures"].values():
        _assert_nonblank_image(path, image)


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
    assert metrics["hybrid_fixture_kind"] == "straight_axis_open_mirror_support_fixture"
    assert metrics["final_hybrid_target_kind"] == "toroidal_stellarator_mirror_hybrid"
    assert metrics["production_hybrid_claim"] is False
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
    assert schema["metrics_schema_version"] == "0.7"
    assert "workflow_status_values" in schema
    assert "free_boundary_status_values" in schema
    assert "ls_boundary_step_fields" in schema
    assert "ls_boundary_coupled_trial_fields" in schema
    assert "ls_boundary_coupled_loop_row_fields" in schema
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
    bad_ls_total = dict(metrics)
    bad_ls_total["ls_boundary_step_rows_total"] = metrics["ls_boundary_step_rows_total"] + 1
    with pytest.raises(ValueError, match="ls_boundary_step_rows_total"):
        module["validate_circular_coil_beta_scan_metrics"](bad_ls_total)
    bad_ls_trial_total = dict(metrics)
    bad_ls_trial_total["ls_boundary_coupled_trial_rows_total"] = metrics["ls_boundary_coupled_trial_rows_total"] + 1
    with pytest.raises(ValueError, match="ls_boundary_coupled_trial_rows_total"):
        module["validate_circular_coil_beta_scan_metrics"](bad_ls_trial_total)
    bad_ls_loop_total = dict(metrics)
    bad_ls_loop_total["ls_boundary_coupled_loop_rows_total"] = metrics["ls_boundary_coupled_loop_rows_total"] + 1
    with pytest.raises(ValueError, match="ls_boundary_coupled_loop_rows_total"):
        module["validate_circular_coil_beta_scan_metrics"](bad_ls_loop_total)
    assert metrics["axis_bz_relative_linf"] < 1.0e-12
    assert metrics["boundary_bmag_min"] > 0.0
    assert metrics["beta_scan_requested_percent"] == [1.0, 3.0, 10.0]
    assert metrics["ls_boundary_step_requested"] is False
    assert metrics["ls_boundary_step_rows_total"] == 0
    assert metrics["ls_boundary_coupled_trial_requested"] is False
    assert metrics["ls_boundary_coupled_trial_rows_total"] == 0
    assert metrics["ls_boundary_coupled_loop_requested"] is False
    assert metrics["ls_boundary_coupled_loop_steps_requested"] == 0
    assert metrics["ls_boundary_coupled_loop_target_merit"] is None
    assert metrics["ls_boundary_coupled_loop_stagnation_rtol"] is None
    assert metrics["ls_boundary_coupled_loop_fsq_growth_limit"] is None
    assert metrics["ls_boundary_coupled_loop_rows_total"] == 0
    assert metrics["ls_boundary_coupled_loop_accepted_rows_total"] == 0
    assert metrics["ls_boundary_coupled_loop_stop_reason_counts"] == {}
    assert metrics["ls_boundary_finite_difference_step"] is None
    assert metrics["ls_boundary_damping"] is None
    assert metrics["ls_boundary_max_relative_step"] is None
    assert metrics["ls_boundary_ridge"] is None
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
    assert all(row["ls_boundary_step"] is None for row in metrics["fixed_boundary_baseline_rows"])
    assert all(row["ls_boundary_coupled_loop_rows"] == [] for row in metrics["fixed_boundary_baseline_rows"])
    assert all(
        row["ls_boundary_coupled_loop_status"] == "not_requested" for row in metrics["fixed_boundary_baseline_rows"]
    )
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


def test_root_free_boundary_circular_coils_summary_reports_converged_free_boundary_statuses():
    module = _load_root_example("mirror_free_boundary_circular_coils.py")
    beta_summary = module["_beta_scan_summary"]
    common = {
        "run_fixed_boundary_baseline": True,
        "lcfs_pilot_steps": 1,
        "lcfs_pilot_target_merit": 1.0,
        "lcfs_pilot_stagnation_rtol": 0.0,
        "lcfs_pilot_fsq_growth_limit": 0.0,
        "run_ls_boundary_step": False,
        "run_ls_boundary_coupled_trial": False,
        "ls_boundary_coupled_loop_steps": 1,
        "ls_boundary_coupled_loop_target_merit": 1.0,
        "ls_boundary_coupled_loop_stagnation_rtol": 0.0,
        "ls_boundary_coupled_loop_fsq_growth_limit": 0.0,
        "ls_boundary_finite_difference_step": 1.0e-5,
        "ls_boundary_damping": 1.0,
        "ls_boundary_max_relative_step": 0.1,
        "ls_boundary_ridge": 1.0e-8,
    }

    pilot_summary = beta_summary(
        [
            {
                "lcfs_pilot_rows": [{"accepted": True, "skipped": False, "stop_reason": "target_merit"}],
                "lcfs_pilot_stop_reason": "target_merit",
                "ls_boundary_step": None,
                "ls_boundary_coupled_loop_rows": [],
            }
        ],
        run_lcfs_pilot=True,
        run_ls_boundary_coupled_loop=False,
        **common,
    )
    assert pilot_summary["free_boundary_solve_status"] == "lcfs_pilot_converged_free_boundary"

    loop_summary = beta_summary(
        [
            {
                "lcfs_pilot_rows": [],
                "ls_boundary_step": None,
                "ls_boundary_coupled_loop_rows": [{"accepted": True, "stop_reason": "target_merit"}],
                "ls_boundary_coupled_loop_stop_reason": "target_merit",
            }
        ],
        run_lcfs_pilot=False,
        run_ls_boundary_coupled_loop=True,
        **common,
    )
    assert loop_summary["free_boundary_solve_status"] == "ls_boundary_coupled_loop_converged_free_boundary"

    mixed_summary = beta_summary(
        [
            {
                "lcfs_pilot_rows": [{"accepted": True, "skipped": False, "stop_reason": "target_merit"}],
                "lcfs_pilot_stop_reason": "target_merit",
                "ls_boundary_step": None,
                "ls_boundary_coupled_loop_rows": [],
            },
            {
                "lcfs_pilot_rows": [{"accepted": True, "skipped": False, "stop_reason": "max_steps"}],
                "lcfs_pilot_stop_reason": "max_steps",
                "ls_boundary_step": None,
                "ls_boundary_coupled_loop_rows": [],
            },
        ],
        run_lcfs_pilot=True,
        run_ls_boundary_coupled_loop=False,
        **common,
    )
    assert mixed_summary["free_boundary_solve_status"] == "lcfs_pilot_not_converged_free_boundary"


def test_root_free_boundary_circular_coils_example_writes_nonblank_plots(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "free_boundary_circular_coils_plots"),
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    module = _load_root_example("mirror_free_boundary_circular_coils.py")
    module["validate_circular_coil_beta_scan_metrics"](metrics)
    assert metrics["workflow_status"] == "lcfs_pilot"
    assert metrics["fixed_boundary_baseline_count"] == 3
    assert metrics["lcfs_pilot_rows_total"] == 3
    assert metrics["beta_scan_requested_percent"] == [1.0, 3.0, 10.0]
    assert set(metrics["figures"]) == {"axis_bz", "boundary_bmag", "geometry", "beta_scan_summary"}
    for path in metrics["figures"].values():
        _assert_nonblank_image(path, image)

    row = metrics["fixed_boundary_baseline_rows"][0]
    pilot = row["lcfs_pilot_rows"][0]
    assert row["beta_percent"] == pytest.approx(1.0)
    assert row["lcfs_pilot_status"] == "accepted"
    assert pilot["accepted"] is True
    assert row["figures"]
    assert pilot["figures"]
    for figures in [row["figures"], pilot["figures"]]:
        for key in ["boundary_3d", "bfield_boundary", "lcfs_diagnostic", "boozer_like_diagnostics"]:
            _assert_nonblank_image(figures[key], image)


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


def test_root_free_boundary_circular_coils_ls_boundary_step_reports_reduction(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "ls_boundary_step"),
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
            "--run-ls-boundary-step",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    module = _load_root_example("mirror_free_boundary_circular_coils.py")
    schema = module["circular_coil_beta_scan_schema"]()
    module["validate_circular_coil_beta_scan_metrics"](metrics)
    row = metrics["fixed_boundary_baseline_rows"][0]
    ls_step = row["ls_boundary_step"]
    selected_rows = [trial for trial in ls_step["trial_rows"] if trial["selected"]]

    assert metrics["metrics_schema_version"] == "0.7"
    assert metrics["ls_boundary_step_requested"] is True
    assert metrics["ls_boundary_coupled_trial_requested"] is False
    assert metrics["ls_boundary_step_rows_total"] == 1
    assert metrics["ls_boundary_coupled_trial_rows_total"] == 0
    assert metrics["ls_boundary_finite_difference_step"] == 1.0e-5
    assert metrics["ls_boundary_damping"] == 1.0
    assert metrics["ls_boundary_max_relative_step"] == 0.1
    assert metrics["ls_boundary_ridge"] == 1.0e-8
    assert set(schema["ls_boundary_step_fields"]).issubset(ls_step)
    assert ls_step["accepted"] is True
    assert len(ls_step["coefficients_initial"]) == 3
    assert len(ls_step["coefficients_new"]) == 3
    assert ls_step["jacobian_shape"][1] == 3
    assert ls_step["residual_value_after"] <= ls_step["residual_value_before"]
    assert ls_step["lcfs_value_after"] <= ls_step["lcfs_value_before"]
    assert ls_step["equilibrium_rms_after"] == pytest.approx(ls_step["equilibrium_rms_before"])
    assert ls_step["figure"] is None
    assert ls_step["coupled_trial"] is None
    assert len(selected_rows) == 1
    assert selected_rows[0]["factor"] == pytest.approx(ls_step["line_search_factor"])
    assert selected_rows[0]["residual_value"] == pytest.approx(ls_step["residual_value_after"])
    assert row["figures"] == {}


def test_root_free_boundary_circular_coils_ls_boundary_coupled_trial_reports_realized_solve(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "ls_boundary_coupled_trial"),
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
            "--run-ls-boundary-step",
            "--run-ls-boundary-coupled-trial",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    module = _load_root_example("mirror_free_boundary_circular_coils.py")
    schema = module["circular_coil_beta_scan_schema"]()
    module["validate_circular_coil_beta_scan_metrics"](metrics)
    row = metrics["fixed_boundary_baseline_rows"][0]
    ls_step = row["ls_boundary_step"]
    trial = ls_step["coupled_trial"]

    assert metrics["metrics_schema_version"] == "0.7"
    assert metrics["ls_boundary_step_requested"] is True
    assert metrics["ls_boundary_coupled_trial_requested"] is True
    assert metrics["ls_boundary_step_rows_total"] == 1
    assert metrics["ls_boundary_coupled_trial_rows_total"] == 1
    assert set(schema["ls_boundary_coupled_trial_fields"]).issubset(trial)
    assert ls_step["accepted"] is True
    assert trial["status"] == "accepted"
    assert trial["accepted_by_merit"] is True
    assert trial["rejection_reason"] is None
    assert Path(trial["mout"]).exists()
    assert trial["final_fsq"] >= 0.0
    assert trial["final_residual_norm"] >= 0.0
    assert trial["final_normalized_force"] >= 0.0
    assert trial["fsq_growth_ratio"] <= 1.0
    assert trial["lcfs_merit_ratio"] <= 1.0
    assert trial["lcfs_external_bnormal_rms"] >= 0.0
    assert trial["lcfs_pressure_balance_rms"] >= 0.0
    assert trial["figures"] == {}


def test_root_free_boundary_circular_coils_ls_boundary_coupled_loop_reports_guarded_steps(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "ls_boundary_coupled_loop"),
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
            "--run-ls-boundary-coupled-loop",
            "--ls-boundary-coupled-loop-steps",
            "2",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    module = _load_root_example("mirror_free_boundary_circular_coils.py")
    schema = module["circular_coil_beta_scan_schema"]()
    module["validate_circular_coil_beta_scan_metrics"](metrics)
    row = metrics["fixed_boundary_baseline_rows"][0]
    loop_rows = row["ls_boundary_coupled_loop_rows"]
    first, second = loop_rows

    assert metrics["metrics_schema_version"] == "0.7"
    assert metrics["workflow_status"] == "ls_boundary_coupled_loop"
    assert metrics["free_boundary_solve_status"] == "ls_boundary_coupled_loop_not_converged_free_boundary"
    assert metrics["ls_boundary_coupled_loop_requested"] is True
    assert metrics["ls_boundary_coupled_loop_steps_requested"] == 2
    assert metrics["ls_boundary_coupled_loop_rows_total"] == 2
    assert metrics["ls_boundary_coupled_loop_accepted_rows_total"] == 1
    assert metrics["ls_boundary_coupled_loop_stop_reason_counts"] == {"None": 1, "ls_step_not_accepted": 1}
    assert row["ls_boundary_coupled_loop_status"] == "skipped"
    assert row["ls_boundary_coupled_loop_rows_count"] == 2
    assert row["ls_boundary_coupled_loop_accepted_rows"] == 1
    assert row["ls_boundary_coupled_loop_stop_reason"] == "ls_step_not_accepted"
    assert row["ls_boundary_coupled_loop_last_accepted_step"] == 1
    assert row["ls_boundary_coupled_loop_last_accepted_merit"] == pytest.approx(first["lcfs_merit"])
    assert row["ls_boundary_coupled_loop_last_accepted_fsq_growth_ratio"] == pytest.approx(first["fsq_growth_ratio"])
    assert set(schema["ls_boundary_coupled_loop_row_fields"]).issubset(first)
    assert first["status"] == "accepted"
    assert first["accepted"] is True
    assert first["rejection_reason"] is None
    assert first["stop_reason"] is None
    assert first["merit_improvement_fraction"] > 0.0
    assert first["fsq_growth_ratio"] <= 1.0
    assert first["lcfs_merit_ratio"] <= 1.0
    assert Path(first["mout"]).exists()
    assert first["ls_boundary_step"]["accepted"] is True
    assert first["ls_boundary_step"]["coupled_trial"]["status"] == "accepted"
    assert second["status"] == "skipped"
    assert second["accepted"] is False
    assert second["rejection_reason"] == "ls_step_not_accepted"
    assert second["stop_reason"] == "ls_step_not_accepted"
    assert second["mout"] is None
    assert second["ls_boundary_step"]["accepted"] is False
    assert second["ls_boundary_step"]["coupled_trial"] is None


def test_root_free_boundary_circular_coils_coupled_loop_reports_target_merit_convergence(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_free_boundary_circular_coils.py",
            "--outdir",
            str(tmp_path / "ls_boundary_coupled_loop_target"),
            "--betas",
            "1,3,10",
            "--ntheta",
            "8",
            "--nxi",
            "11",
            "--n-segments",
            "64",
            "--run-fixed-boundary-baseline",
            "--baseline-maxiter",
            "5",
            "--run-ls-boundary-coupled-loop",
            "--ls-boundary-coupled-loop-steps",
            "4",
            "--ls-boundary-coupled-loop-target-merit",
            "0.5",
            "--ls-boundary-coupled-loop-fsq-growth-limit",
            "1.5",
            "--ls-boundary-max-relative-step",
            "0.05",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    module = _load_root_example("mirror_free_boundary_circular_coils.py")
    module["validate_circular_coil_beta_scan_metrics"](metrics)

    assert metrics["free_boundary_solve_status"] == "ls_boundary_coupled_loop_converged_free_boundary"
    assert metrics["ls_boundary_coupled_loop_stop_reason_counts"] == {"target_merit": 3}
    assert metrics["ls_boundary_coupled_loop_rows_total"] == 3
    assert metrics["ls_boundary_coupled_loop_accepted_rows_total"] == 3
    for row in metrics["fixed_boundary_baseline_rows"]:
        assert row["ls_boundary_coupled_loop_status"] == "accepted"
        assert row["ls_boundary_coupled_loop_stop_reason"] == "target_merit"
        assert row["ls_boundary_coupled_loop_accepted_rows"] == 1
        assert row["ls_boundary_coupled_loop_final_merit"] <= 0.5
        assert row["ls_boundary_coupled_loop_final_fsq_growth_ratio"] <= 1.5
        assert len(row["ls_boundary_coupled_loop_rows"]) == 1
        assert row["ls_boundary_coupled_loop_rows"][0]["accepted"] is True


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


def test_root_fixed_boundary_solve_diagnostic_writes_nonblank_plots(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_fixed_boundary_solve_diagnostic.py",
            "--outdir",
            str(tmp_path / "solve_diagnostic_plots"),
            "--ns-array",
            "7",
            "--nxi",
            "13",
            "--maxiter",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = json.loads(Path(completed.stdout.strip()).read_text())
    assert len(rows) == 1
    row = rows[0]
    assert row["ns"] == 7
    assert row["nxi"] == 13
    assert row["optimizer"] == "lbfgs"
    assert row["optimizer_nit"] <= 2
    assert row["optimizer_accepted"] is True
    assert row["final_fsq"] >= 0.0
    assert row["final_normalized_force"] >= 0.0
    output = load_mirror_output(row["mout"])
    assert output.diagnostics.min_sqrtg > 0.0
    figure_dir = Path(row["mout"]).parent / "figures"
    for name in [
        "fixed_boundary_solve_ns7_nxi13_mirror_boundary_3d.png",
        "fixed_boundary_solve_ns7_nxi13_mirror_cross_sections.png",
        "fixed_boundary_solve_ns7_nxi13_mirror_bfield_boundary.png",
        "fixed_boundary_solve_ns7_nxi13_mirror_residual_history.png",
        "fixed_boundary_solve_ns7_nxi13_mirror_boozer_like_diagnostics.png",
    ]:
        _assert_nonblank_image(figure_dir / name, image)


def test_root_fixed_boundary_solve_diagnostic_residual_newton_reports_krylov_fields(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_fixed_boundary_solve_diagnostic.py",
            "--outdir",
            str(tmp_path / "solve_diagnostic_residual_newton"),
            "--ns-array",
            "5",
            "--nxi",
            "7",
            "--maxiter",
            "1",
            "--optimizer",
            "residual_newton",
            "--residual-linear-solver",
            "lsmr",
            "--residual-linear-maxiter",
            "4",
            "--residual-linear-maxiter-policy",
            "fixed",
            "--residual-preconditioner",
            "radial_xi_tridi",
            "--residual-compare-dense-step",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    rows = json.loads(path.read_text())
    assert len(rows) == 1
    row = rows[0]
    assert row["optimizer"] == "residual_newton"
    assert row["residual_linear_solver"] == "lsmr"
    assert row["residual_linear_maxiter_policy"] == "fixed"
    assert row["residual_linear_maxiter_effective_last"] == 4
    assert row["residual_linear_iterations_last"] >= 1
    assert row["residual_compare_dense_step"] is True
    assert row["residual_dense_step_cosine_last"] is not None
    assert row["residual_dense_step_relative_error_last"] is not None
    assert row["residual_preconditioner"] == "radial_xi_tridi"
    assert row["optimizer_accepted"] is True
    assert row["final_fsq"] >= 0.0
    assert Path(row["mout"]).exists()


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


def test_root_implicit_parameter_gradients_example_matrix_free_writes_plot(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_implicit_parameter_gradients.py",
            "--outdir",
            str(tmp_path / "implicit_parameter_gradients_matrix_free"),
            "--solve-method",
            "matrix_free_cg",
            "--families",
            "pressure,boundary",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert metrics["solve_method"] == "matrix_free_cg"
    assert metrics["families"] == ["pressure", "boundary"]
    assert metrics["accepted"]
    for row in metrics["rows"]:
        assert row["accepted"]
        assert row["perturbed_residual_norm"] < 1.0e-10
        assert row["custom_vs_forward_relative_error"] < 1.0e-8
        assert row["custom_vs_finite_difference_relative_error"] < 1.0e-3

    figures = metrics["figures"]
    assert set(figures) == {"directional_gradients"}
    _assert_nonblank_image(figures["directional_gradients"], image)


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


def test_root_implicit_solve_benchmark_writes_nonblank_plot(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_implicit_solve_benchmark.py",
            "--outdir",
            str(tmp_path / "implicit_solve_benchmark_plots"),
            "--ns-array",
            "5",
            "--nxi-array",
            "7",
            "--repeat",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    assert metrics["accepted"]
    assert Path(metrics["csv"]).exists()
    assert set(metrics["figures"]) == {"summary"}
    _assert_nonblank_image(metrics["figures"]["summary"], image)
    rows = {(row["method"], row["vector_size"]): row for row in metrics["rows"]}
    assert set(rows) == {("dense", 45), ("matrix_free_cg", 45)}
    assert rows[("dense", 45)]["relative_error_vs_dense"] == pytest.approx(0.0)
    assert rows[("matrix_free_cg", 45)]["relative_error_vs_dense"] < 1.0e-5
    assert rows[("matrix_free_cg", 45)]["linear_residual_relative"] < 1.0e-5


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


def test_root_solver_comparison_example_writes_nonblank_plots(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_solver_comparison.py",
            "--outdir",
            str(tmp_path / "solver_comparison_plots"),
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    figure_names = {Path(path).name for path in metrics["figures"]}
    assert figure_names == {
        "solver_comparison_residuals.png",
        "solver_comparison_final_residuals.png",
        "solver_comparison_boundaries.png",
    }
    for path in metrics["figures"]:
        _assert_nonblank_image(path, image)

    residual_newton_rows = {
        (row["case"], row["solver_scope"]): row for row in metrics["rows"] if row["optimizer"] == "residual_newton"
    }
    assert ("cylinder", "production_fixed_boundary") in residual_newton_rows
    assert ("two_coil", "production_fixed_boundary") in residual_newton_rows
    assert ("manufactured", "manufactured_source_validation") in residual_newton_rows
    for key in [("cylinder", "production_fixed_boundary"), ("two_coil", "production_fixed_boundary")]:
        row = residual_newton_rows[key]
        assert row["residual_linear_solver"] == "lsmr"
        assert row["residual_preconditioner"] == "radial_xi_tridi"
        assert row["residual_linear_maxiter_policy"] == "adaptive"
        assert row["residual_linear_iterations_total"] is not None
        assert row["final_residual_norm"] >= 0.0

    artifacts = {artifact["case"]: artifact for artifact in metrics["physical_artifacts"]}
    assert set(artifacts) == {"cylinder", "two_coil"}
    for case, artifact in artifacts.items():
        output = load_mirror_output(artifact["mout"])
        assert output.diagnostics.min_sqrtg > 0.0
        figure_dir = Path(artifact["figures"])
        for suffix in [
            "mirror_boundary_3d.png",
            "mirror_residual_history.png",
            "mirror_boozer_like_diagnostics.png",
        ]:
            _assert_nonblank_image(figure_dir / f"{case}_residual_newton_{suffix}", image)


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


def test_root_residual_newton_convergence_grid_writes_nonblank_plots(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_residual_newton_convergence_grid.py",
            "--outdir",
            str(tmp_path / "convergence_grid_plots"),
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    figure_names = {Path(path).name for path in metrics["figures"]}
    assert figure_names == {
        "residual_newton_convergence_resolution_heatmap.png",
        "residual_newton_convergence_budget.png",
        "residual_newton_convergence_history.png",
        "residual_newton_convergence_components.png",
    }
    for path in metrics["figures"]:
        _assert_nonblank_image(path, image)

    best = metrics["selected_artifacts"]["best_residual"]
    assert Path(best["mout"]).exists()
    figure_dir = Path(best["figures"])
    for name in [
        "best_two_coil_residual_newton_mirror_boundary_3d.png",
        "best_two_coil_residual_newton_mirror_boozer_like_diagnostics.png",
    ]:
        _assert_nonblank_image(figure_dir / name, image)


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


def test_root_residual_newton_convergence_grid_finite_current_writes_nonblank_plots(tmp_path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_residual_newton_convergence_grid.py",
            "--outdir",
            str(tmp_path / "finite_current_convergence_grid_plots"),
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
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    assert metrics["case_label"] == "finite_current_two_coil"
    assert metrics["i_prime_value"] == pytest.approx(0.01)
    figure_names = {Path(path).name for path in metrics["figures"]}
    assert figure_names == {
        "residual_newton_convergence_resolution_heatmap.png",
        "residual_newton_convergence_budget.png",
        "residual_newton_convergence_history.png",
        "residual_newton_convergence_components.png",
    }
    for path in metrics["figures"]:
        _assert_nonblank_image(path, image)

    row = metrics["rows"][0]
    assert row["finite_current"]
    assert row["residual_lam_norm"] > 0.0
    assert row["residual_lam_fraction"] > 0.0
    assert row["twist_proxy_i_prime_over_psi_prime"] > 0.0

    best = metrics["selected_artifacts"]["best_residual"]
    output = load_mirror_output(best["mout"])
    pitch = mirror_field_line_pitch_profile_data(output, num_lines=2)
    assert np.min(pitch.turns_mean) > 0.0
    figure_dir = Path(best["figures"])
    for name in [
        "best_finite_current_two_coil_residual_newton_mirror_boundary_3d.png",
        "best_finite_current_two_coil_residual_newton_mirror_bfield_boundary.png",
        "best_finite_current_two_coil_residual_newton_mirror_boozer_like_diagnostics.png",
        "best_finite_current_two_coil_residual_newton_mirror_radial_diagnostics.png",
    ]:
        _assert_nonblank_image(figure_dir / name, image)

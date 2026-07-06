"""Tests for compact fixed-boundary profile summary rendering."""

from __future__ import annotations

import json

from tools.diagnostics import summarize_fixed_boundary_profiles as summary


def test_summarize_profile_extracts_compile_and_scan_metrics(tmp_path) -> None:
    """Profile summaries should expose the M3A setup/compile fields."""

    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "wall_time_s": 4.2,
                "timing": {
                    "scan_total_s": 3.5,
                    "scan_initial_compute_forces_s": 0.4,
                    "scan_axis_reset_compute_forces_s": 0.1,
                    "scan_run_setup_s": 0.7,
                    "scan_preflight_s": 0.2,
                    "scan_device_run_s": 1.9,
                    "scan_device_dispatch_s": 1.8,
                    "scan_device_ready_s": 0.1,
                    "scan_runner_cache_lookup_s": 0.01,
                    "scan_runner_cache_build_s": 0.02,
                    "scan_runner_explicit_lower_s": 0.3,
                    "scan_runner_explicit_compile_s": 0.4,
                    "scan_runner_explicit_compile_count": 1,
                    "scan_runner_explicit_compile_failure_count": 0,
                    "scan_runner_explicit_hlo_line_count": 100,
                    "scan_runner_explicit_hlo_instruction_count": 80,
                    "scan_runner_explicit_hlo_failure_count": 0,
                    "scan_runner_arg_leaf_count": 40,
                    "scan_runner_arg_array_leaf_count": 35,
                    "scan_runner_arg_scalar_leaf_count": 5,
                    "scan_runner_arg_array_nbytes": 4096,
                    "scan_runner_arg_preconditioner_rz_mats_key_count": 8,
                    "scan_runner_arg_preconditioner_rz_mats_unexpected_key_count": 0,
                    "scan_runner_arg_preconditioner_rz_mats_missing_mandatory_key_count": 0,
                    "scan_runner_arg_preconditioner_rz_mats_compact_ok_count": 1,
                    "scan_runner_arg_path_arg0_state_leaf_count": 20,
                    "scan_runner_arg_path_arg0_state_array_nbytes": 2048,
                    "scan_runner_arg_path_arg0_cache_prec_rz_mats_leaf_count": 8,
                    "scan_runner_arg_path_arg0_cache_prec_rz_mats_array_nbytes": 1024,
                    "scan_runner_arg_category_state_leaf_count": 20,
                    "scan_runner_arg_category_state_array_nbytes": 2048,
                    "scan_runner_arg_category_preconditioner_leaf_count": 8,
                    "scan_runner_arg_category_preconditioner_array_nbytes": 1024,
                    "scan_history_none": 0,
                    "scan_history_leaf_count": 9,
                    "scan_history_array_leaf_count": 9,
                    "scan_history_scalar_leaf_count": 0,
                    "scan_history_array_nbytes": 288,
                    "scan_runner_cache_hit_count": 3,
                    "scan_runner_cache_miss_count": 1,
                    "scan_runner_cache_bypass_count": 0,
                    "scan_runner_cache_hit_dispatch_s": 0.3,
                    "scan_runner_cache_hit_ready_s": 0.03,
                    "scan_runner_cache_miss_dispatch_s": 1.5,
                    "scan_runner_cache_miss_ready_s": 0.07,
                    "scan_runner_cache_bypass_dispatch_s": 0.0,
                    "scan_runner_cache_bypass_ready_s": 0.0,
                    "scan_runner_cache_miss_category_cold_empty_count": 1,
                    "scan_runner_cache_miss_category_iteration_budget_count": 2,
                },
                "phase_timing": {
                    "cprofile_compile_summary": {
                        "backend_compile_and_load_call_count": 90,
                        "backend_compile_and_load_cumulative_s": 2.8,
                    }
                },
                "diagnostics": {
                    "use_scan_policy_source": "profile",
                    "use_scan_policy_detail": "bundled:short_case",
                    "fixed_boundary_execution_classification": "scan_cache_hit",
                },
                "result": {"final_residual": 1.25e-3},
            }
        ),
        encoding="utf-8",
    )

    row = summary.summarize_profile("case", path)

    assert row["label"] == "case"
    assert row["wall_s"] == 4.2
    assert row["scan_initial_compute_forces_s"] == 0.4
    assert row["scan_axis_reset_compute_forces_s"] == 0.1
    assert row["scan_run_setup_s"] == 0.7
    assert row["scan_preflight_s"] == 0.2
    assert row["scan_device_dispatch_s"] == 1.8
    assert row["scan_device_ready_s"] == 0.1
    assert row["scan_runner_cache_lookup_s"] == 0.01
    assert row["scan_runner_cache_build_s"] == 0.02
    assert row["scan_runner_explicit_lower_s"] == 0.3
    assert row["scan_runner_explicit_compile_s"] == 0.4
    assert row["scan_runner_explicit_compile_count"] == 1.0
    assert row["scan_runner_explicit_compile_failure_count"] == 0.0
    assert row["scan_runner_explicit_hlo_line_count"] == 100.0
    assert row["scan_runner_explicit_hlo_instruction_count"] == 80.0
    assert row["scan_runner_explicit_hlo_failure_count"] == 0.0
    assert row["scan_runner_arg_leaf_count"] == 40.0
    assert row["scan_runner_arg_array_leaf_count"] == 35.0
    assert row["scan_runner_arg_scalar_leaf_count"] == 5.0
    assert row["scan_runner_arg_array_nbytes"] == 4096.0
    assert row["scan_runner_arg_preconditioner_rz_mats_key_count"] == 8.0
    assert row["scan_runner_arg_preconditioner_rz_mats_unexpected_key_count"] == 0.0
    assert row["scan_runner_arg_preconditioner_rz_mats_missing_mandatory_key_count"] == 0.0
    assert row["scan_runner_arg_preconditioner_rz_mats_compact_ok_count"] == 1.0
    assert row["scan_history_none"] == 0.0
    assert row["scan_history_leaf_count"] == 9.0
    assert row["scan_history_array_leaf_count"] == 9.0
    assert row["scan_history_scalar_leaf_count"] == 0.0
    assert row["scan_history_array_nbytes"] == 288.0
    assert row["use_scan_policy_source"] == "profile"
    assert row["use_scan_policy_detail"] == "bundled:short_case"
    assert row["fixed_boundary_execution_classification"] == "scan_cache_hit"
    assert row["scan_runner_arg_top_leaf_paths"].startswith("arg0_state:20")
    assert "arg0_cache_prec_rz_mats:8" in row["scan_runner_arg_top_leaf_paths"]
    assert row["scan_runner_arg_top_nbytes_paths"].startswith("arg0_state:2048")
    assert "arg0_cache_prec_rz_mats:1024" in row["scan_runner_arg_top_nbytes_paths"]
    assert row["scan_runner_arg_top_leaf_categories"].startswith("state:20")
    assert "preconditioner:8" in row["scan_runner_arg_top_leaf_categories"]
    assert row["scan_runner_arg_top_nbytes_categories"].startswith("state:2048")
    assert "preconditioner:1024" in row["scan_runner_arg_top_nbytes_categories"]
    assert row["scan_runner_cache_hit_count"] == 3.0
    assert row["scan_runner_cache_miss_count"] == 1.0
    assert row["scan_runner_cache_bypass_count"] == 0.0
    assert row["scan_runner_cache_hit_dispatch_s"] == 0.3
    assert row["scan_runner_cache_hit_ready_s"] == 0.03
    assert row["scan_runner_cache_miss_dispatch_s"] == 1.5
    assert row["scan_runner_cache_miss_ready_s"] == 0.07
    assert row["scan_runner_cache_bypass_dispatch_s"] == 0.0
    assert row["scan_runner_cache_bypass_ready_s"] == 0.0
    assert row["scan_runner_cache_miss_categories"] == "cold_empty:1,iteration_budget:2"
    assert row["backend_compile_count"] == 90.0
    assert row["final_residual"] == 1.25e-3


def test_summarize_profile_uses_final_w_when_final_residual_is_absent(tmp_path) -> None:
    """Current profile_fixed_boundary reports final_w as the total residual."""

    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"result": {"final_w": 0.25}}), encoding="utf-8")

    assert summary.summarize_profile("case", path)["final_residual"] == 0.25


def test_render_markdown_includes_blank_missing_values() -> None:
    """Missing optional metrics should not break Markdown rendering."""

    text = summary.render_markdown(
        [
            {
                "label": "case",
                "path": "profile.json",
                "wall_s": 4.0,
                "scan_total_s": None,
                "scan_initial_compute_forces_s": None,
                "scan_axis_reset_compute_forces_s": None,
                "scan_run_setup_s": 0.5,
                "scan_preflight_s": None,
                "scan_device_run_s": None,
                "scan_device_dispatch_s": None,
                "scan_device_ready_s": None,
                "scan_runner_cache_lookup_s": None,
                "scan_runner_cache_build_s": None,
                "scan_runner_explicit_lower_s": None,
                "scan_runner_explicit_compile_s": None,
                "scan_runner_explicit_compile_count": None,
                "scan_runner_explicit_compile_failure_count": None,
                "scan_runner_explicit_hlo_line_count": None,
                "scan_runner_explicit_hlo_instruction_count": None,
                "scan_runner_explicit_hlo_failure_count": None,
                "scan_runner_arg_leaf_count": None,
                "scan_runner_arg_array_leaf_count": None,
                "scan_runner_arg_scalar_leaf_count": None,
                "scan_runner_arg_array_nbytes": None,
                "scan_runner_arg_preconditioner_rz_mats_key_count": None,
                "scan_runner_arg_preconditioner_rz_mats_unexpected_key_count": None,
                "scan_runner_arg_preconditioner_rz_mats_missing_mandatory_key_count": None,
                "scan_runner_arg_preconditioner_rz_mats_compact_ok_count": None,
                "scan_history_none": None,
                "scan_history_leaf_count": None,
                "scan_history_array_leaf_count": None,
                "scan_history_scalar_leaf_count": None,
                "scan_history_array_nbytes": None,
                "use_scan_policy_source": "",
                "use_scan_policy_detail": "",
                "fixed_boundary_execution_classification": "",
                "scan_runner_arg_top_leaf_paths": "",
                "scan_runner_arg_top_nbytes_paths": "",
                "scan_runner_cache_hit_count": None,
                "scan_runner_cache_miss_count": None,
                "scan_runner_cache_bypass_count": None,
                "scan_runner_cache_hit_dispatch_s": None,
                "scan_runner_cache_hit_ready_s": None,
                "scan_runner_cache_miss_dispatch_s": None,
                "scan_runner_cache_miss_ready_s": None,
                "scan_runner_cache_bypass_dispatch_s": None,
                "scan_runner_cache_bypass_ready_s": None,
                "scan_runner_cache_miss_categories": "",
                "backend_compile_count": 90.0,
                "backend_compile_s": 2.75,
                "final_residual": 1.0e-3,
            }
        ]
    )

    assert "| label | path | wall_s |" in text
    assert "| case | profile.json | 4 |  |  |  | 0.5 |" in text
    assert "|  |  |  | 90 | 2.75 | 0.001 |" in text


def test_scan_cache_miss_categories_are_stable_and_sorted() -> None:
    """Cache miss categories should render deterministically for CSV/Markdown."""

    text = summary._scan_cache_miss_categories(
        {
            "scan_runner_cache_miss_category_tolerance_count": 3,
            "scan_runner_cache_miss_category_geometry_count": 1,
            "scan_runner_cache_miss_category_empty_count": 0,
            "unrelated": 9,
        }
    )

    assert text == "geometry:1,tolerance:3"

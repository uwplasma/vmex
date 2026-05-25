from __future__ import annotations

from tools.benchmarks import bench_freeb_direct_coil_matrix as matrix


def _direct_solve_payload() -> dict:
    return {
        "cases": [
            {
                "label": "synthetic_direct_coil_solve",
                "status": "completed",
                "cold_or_compile_s": 1.23,
                "warm": {"min_s": 0.9, "mean_s": 1.0},
                "cold_solver_timing": {
                    "active_nestor_timing_summary": {"active_steps": 2},
                    "trial_nestor_timing_summary": {"recorded_calls": 3},
                },
                "warm_solver_timing": {
                    "active_nestor_timing_summary": {"active_steps": 1},
                    "trial_nestor_timing_summary": {"recorded_calls": 1},
                },
                "active_nestor_timing_improvement": {"sample_time_s": {"speedup": 10.0}},
                "trial_nestor_timing_improvement": {"sample_time_s": {"speedup": 2.0}},
                "free_boundary": {
                    "nestor_model": "vmec2000_like_dense_integral",
                    "last_provider_kind": "direct_coils",
                    "final_nestor_recompute_attempted": True,
                    "final_nestor_recompute_failed": False,
                    "final_nestor_sample_time_s": 0.012,
                    "final_nestor_solve_time_s": 0.027,
                    "last_nestor_diagnostics": {
                        "sample_points": 600,
                        "sample_time_s": 0.012,
                        "solve_time_s": 0.027,
                        "sample_setup_time_s": 0.001,
                        "sample_boundary_geometry_time_s": 0.002,
                        "sample_external_field_time_s": 0.003,
                        "sample_axis_field_time_s": 0.004,
                        "sample_projection_time_s": 0.005,
                        "sample_total_time_s": 0.015,
                        "cache_build_time_s": 0.008,
                        "source_time_s": 0.018,
                        "linear_solve_time_s": 0.00006,
                        "provider_jit_sampler": True,
                        "provider_coil_geometry_cached": True,
                        "provider_coil_count": 8,
                        "provider_segments_per_coil": 96,
                        "mode_matrix_lu_built": True,
                    },
                },
            }
        ]
    }


def test_matrix_timing_snapshot_preserves_compact_nestor_details() -> None:
    rows = matrix._timing_snapshot(_direct_solve_payload(), include_nestor=True)

    assert rows[0]["label"] == "synthetic_direct_coil_solve"
    assert rows[0]["cold_or_compile_s"] == 1.23
    assert rows[0]["warm_min_s"] == 0.9
    nestor = rows[0]["nestor"]
    assert nestor["model"] == "vmec2000_like_dense_integral"
    assert nestor["provider_kind"] == "direct_coils"
    assert nestor["active"]["cold"] == {"active_steps": 2}
    assert nestor["active"]["warm"] == {"active_steps": 1}
    assert nestor["active"]["improvement"]["sample_time_s"]["speedup"] == 10.0
    assert nestor["trial"]["cold"] == {"recorded_calls": 3}
    assert nestor["trial"]["warm"] == {"recorded_calls": 1}
    assert nestor["trial"]["improvement"]["sample_time_s"]["speedup"] == 2.0
    assert nestor["final_recompute"]["attempted"] is True
    assert nestor["final_recompute"]["sample_time_s"] == 0.012
    assert nestor["final_diagnostics"]["sample_points"] == 600
    assert nestor["final_diagnostics"]["sample_phase_time_s"]["setup"] == 0.001
    assert nestor["final_diagnostics"]["sample_phase_time_s"]["external_field"] == 0.003
    assert nestor["final_diagnostics"]["sample_phase_time_s"]["total"] == 0.015
    assert nestor["final_diagnostics"]["phase_time_s"]["cache_build"] == 0.008
    assert nestor["final_diagnostics"]["phase_time_s"]["source"] == 0.018
    assert nestor["final_diagnostics"]["phase_time_s"]["linear_solve"] == 0.00006
    assert nestor["final_diagnostics"]["provider"]["jit_sampler"] is True
    assert nestor["final_diagnostics"]["provider"]["geometry_cached"] is True
    assert nestor["final_diagnostics"]["provider"]["coil_count"] == 8
    assert nestor["final_diagnostics"]["provider"]["segments_per_coil"] == 96
    assert nestor["final_diagnostics"]["lu_built"]["mode_matrix"] is True


def test_matrix_timing_snapshot_keeps_provider_and_gradient_rows_compact() -> None:
    rows = matrix._timing_snapshot(_direct_solve_payload(), include_nestor=False)

    assert rows[0]["label"] == "synthetic_direct_coil_solve"
    assert "nestor" not in rows[0]


def test_matrix_timing_snapshot_skipped_case_is_not_noisy() -> None:
    payload = {"cases": [{"label": "essos_direct_coil_solve", "status": "skipped", "reason": "not_requested"}]}

    rows = matrix._timing_snapshot(payload, include_nestor=True)

    assert rows == [{"label": "essos_direct_coil_solve", "status": "skipped", "cold_or_compile_s": None, "reason": "not_requested"}]


def test_gpu_platform_name_prefers_concrete_jax_backend() -> None:
    assert matrix._gpu_platform_name({"platforms": ["cpu", "cuda"]}) == "cuda"
    assert matrix._gpu_platform_name({"platforms": ["gpu"], "devices": ["cuda:0"]}) == "cuda"
    assert matrix._gpu_platform_name({"platforms": ["rocm"]}) == "rocm"
    assert matrix._gpu_platform_name({"platforms": ["gpu"]}) == "gpu"


def test_cpu_gpu_comparison_matches_completed_cases_and_reports_nestor_ratios() -> None:
    def timing(cold: float, warm_min: float, *, sample: float, solve: float) -> dict:
        return {
            "label": "synthetic_direct_coil_solve",
            "status": "completed",
            "cold_or_compile_s": cold,
            "warm_min_s": warm_min,
            "warm_mean_s": warm_min * 2.0,
            "nestor": {
                "active": {
                    "warm": {
                        "sample_time_s": {"total_s": sample},
                        "solve_time_s": {"total_s": solve},
                    }
                },
                "trial": {"warm": {"sample_time_s": {"total_s": sample * 2.0}}},
                "final_recompute": {
                    "sample_time_s": sample * 3.0,
                    "solve_time_s": solve * 4.0,
                },
                "final_diagnostics": {
                    "sample_points": 64,
                    "sample_phase_time_s": {"external_field": sample * 5.0},
                    "provider": {"coil_count": 2},
                },
            },
        }

    comparisons = matrix._cpu_gpu_comparison(
        [
            {
                "label": "direct_solve",
                "backend": "cpu",
                "status": "completed",
                "timings": [timing(4.0, 2.0, sample=0.1, solve=0.2)],
            },
            {
                "label": "direct_solve",
                "backend": "gpu",
                "status": "completed",
                "timings": [timing(2.0, 0.5, sample=0.05, solve=0.1)],
            },
            {
                "label": "direct_solve",
                "backend": "gpu",
                "status": "failed",
                "timings": [timing(1.0, 1.0, sample=1.0, solve=1.0)],
            },
            {
                "label": "gradient",
                "backend": "cpu",
                "status": "completed",
                "timings": [timing(1.0, 1.0, sample=1.0, solve=1.0)],
            },
        ]
    )

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison["label"] == "direct_solve"
    assert comparison["case"] == "synthetic_direct_coil_solve"
    assert comparison["cpu"]["sample_points"] == 64
    assert comparison["gpu"]["provider"] == {"coil_count": 2}
    assert comparison["ratios_gpu_over_cpu"] == {
        "cold_or_compile": 0.5,
        "warm_min": 0.25,
        "warm_mean": 0.25,
        "active_nestor_warm_sample": 0.5,
        "active_nestor_warm_solve": 0.5,
        "final_recompute_sample": 0.5,
        "final_recompute_solve": 0.5,
        "final_external_field_sample": 0.5,
    }

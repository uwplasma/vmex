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
                        "cache_build_time_s": 0.008,
                        "source_time_s": 0.018,
                        "linear_solve_time_s": 0.00006,
                        "provider_jit_sampler": True,
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
    assert nestor["final_diagnostics"]["phase_time_s"]["cache_build"] == 0.008
    assert nestor["final_diagnostics"]["phase_time_s"]["source"] == 0.018
    assert nestor["final_diagnostics"]["phase_time_s"]["linear_solve"] == 0.00006
    assert nestor["final_diagnostics"]["provider"]["jit_sampler"] is True
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

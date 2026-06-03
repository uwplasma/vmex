from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.benchmarks import bench_freeb_direct_coil_matrix as matrix


def _direct_solve_payload() -> dict:
    solve_total_s = 0.7

    def phase(name: str, key: str, seconds: float, per_iter_s: float) -> dict:
        return {
            "name": name,
            "key": key,
            "seconds": seconds,
            "per_iter_s": per_iter_s,
            "fraction_of_solve": seconds / solve_total_s,
        }

    phase_entries = [
        phase("setup", "setup_total_s", 0.20, 0.10),
        phase("preconditioner", "preconditioner_s", 0.15, 0.075),
        phase("residual_metrics", "iteration_residual_metrics_s", 0.11, 0.055),
        phase("force_eval", "force_eval_s", 0.09, 0.045),
        phase("update", "update_s", 0.04, 0.02),
        phase("finalize", "finalize_s", 0.02, 0.01),
    ]
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
                    "timing": {
                        "solve_total_s": 0.7,
                        "iterations": 2,
                        "iteration_loop_s": 0.6,
                        "iteration_loop_unattributed_s": 0.1,
                        "setup_total_s": 0.08,
                        "setup_cache_key_hash_s": 0.03,
                        "setup_update_constants_s": 0.02,
                        "iteration_control_s": 0.04,
                        "iteration_control_fsq1_s": 0.02,
                        "iteration_control_fsq1_precond_norm_s": 0.004,
                        "iteration_control_fsq1_scalar_build_s": 0.002,
                        "iteration_control_fsq1_payload_get_s": 0.011,
                        "iteration_control_fsq1_direct_get_s": 0.003,
                        "iteration_control_fsq1_unattributed_s": 0.006,
                        "iteration_control_badjac_s": 0.01,
                        "iteration_control_badjac_ptau_get_s": 0.004,
                        "iteration_control_badjac_state_jacobian_s": 0.003,
                        "iteration_control_badjac_unattributed_s": 0.003,
                        "iteration_control_vmec_time_s": 0.005,
                        "iteration_control_restart_s": 0.003,
                        "iteration_control_evolve_s": 0.002,
                        "iteration_control_unattributed_s": 0.0,
                        "compute_forces_s": 0.2,
                        "preconditioner_s": 0.15,
                        "update_s": 0.05,
                        "update_state_s": 0.035,
                        "update_state_ready_s": 0.012,
                        "compute_forces_per_iter_s": 0.1,
                        "preconditioner_per_iter_s": 0.075,
                        "update_per_iter_s": 0.025,
                        "update_state_per_iter_s": 0.0175,
                        "update_state_ready_per_iter_s": 0.006,
                        "finalize_s": 0.02,
                        "finalize_nestor_recompute_s": 0.004,
                        "finalize_residual_recompute_s": 0.007,
                        "finalize_residual_device_get_s": 0.002,
                        "finalize_diag_build_s": 0.005,
                        "finalize_unattributed_s": 0.004,
                    },
                    "active_nestor_timing_summary": {"active_steps": 1},
                    "trial_nestor_timing_summary": {"recorded_calls": 1},
                },
                "phase_timing_comparison": {
                    "warm": {
                        "timing_available": True,
                        "solve_total_s": solve_total_s,
                        "iterations": 2,
                        "named_phase_total_s": 0.61,
                        "named_phase_fraction_of_solve": 0.61 / solve_total_s,
                        "top_named_phases": phase_entries[:3],
                        "all_named_phases": phase_entries,
                    },
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
    assert rows[0]["solver"]["warm"]["solve_total_s"] == 0.7
    assert rows[0]["solver"]["warm"]["setup_cache_key_hash_s"] == 0.03
    assert rows[0]["solver"]["warm"]["setup_update_constants_s"] == 0.02
    assert rows[0]["solver"]["warm"]["iteration_loop_unattributed_s"] == 0.1
    assert rows[0]["solver"]["warm"]["iteration_control_s"] == 0.04
    assert rows[0]["solver"]["warm"]["compute_forces_per_iter_s"] == 0.1
    assert rows[0]["solver"]["warm"]["update_state_ready_s"] == 0.012
    assert rows[0]["solver"]["warm"]["finalize_nestor_recompute_s"] == 0.004
    assert rows[0]["solver"]["warm"]["finalize_residual_device_get_s"] == 0.002
    assert rows[0]["solver"]["warm"]["finalize_unattributed_s"] == 0.004
    phase_timing = rows[0]["phase_timing"]["warm"]
    assert phase_timing["solve_total_s"] == 0.7
    assert phase_timing["named_phase_fraction_of_solve"] == 0.61 / 0.7
    assert [phase["name"] for phase in phase_timing["top_named_phases"]] == [
        "setup",
        "preconditioner",
        "residual_metrics",
    ]
    assert phase_timing["all_named_phases"][0]["key"] == "setup_total_s"


def test_matrix_timing_snapshot_keeps_provider_and_gradient_rows_compact() -> None:
    rows = matrix._timing_snapshot(_direct_solve_payload(), include_nestor=False)

    assert rows[0]["label"] == "synthetic_direct_coil_solve"
    assert "nestor" not in rows[0]
    assert "phase_timing" not in rows[0]


def test_matrix_timing_snapshot_skipped_case_is_not_noisy() -> None:
    payload = {"cases": [{"label": "essos_direct_coil_solve", "status": "skipped", "reason": "not_requested"}]}

    rows = matrix._timing_snapshot(payload, include_nestor=True)

    assert rows == [{"label": "essos_direct_coil_solve", "status": "skipped", "cold_or_compile_s": None, "reason": "not_requested"}]


def test_gpu_platform_name_prefers_concrete_jax_backend() -> None:
    assert matrix._gpu_platform_name({"platforms": ["cpu", "cuda"]}) == "cuda"
    assert matrix._gpu_platform_name({"platforms": ["gpu"], "devices": ["cuda:0"]}) == "cuda"
    assert matrix._gpu_platform_name({"platforms": ["rocm"]}) == "rocm"
    assert matrix._gpu_platform_name({"platforms": ["gpu"]}) == "gpu"


def test_gpu_available_probes_concrete_accelerator_platforms(monkeypatch) -> None:
    calls: list[str | None] = []

    class FakeDevice:
        def __init__(self, platform: str, name: str):
            self.platform = platform
            self._name = name

        def __str__(self) -> str:
            return self._name

    fake_jax = SimpleNamespace(
        default_backend=lambda: "cpu",
        devices=lambda platform=None: _fake_devices(platform, calls),
    )

    def _fake_devices(platform, calls):
        calls.append(platform)
        if platform is None:
            return [FakeDevice("cpu", "TFRT_CPU_0")]
        if platform == "cuda":
            return [FakeDevice("cuda", "cuda:0")]
        raise RuntimeError(f"{platform} unavailable")

    monkeypatch.setattr(matrix, "jax", fake_jax, raising=False)
    monkeypatch.setattr(matrix, "has_jax", lambda: True, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "vmec_jax._compat", SimpleNamespace(has_jax=lambda: True, jax=fake_jax))

    available, info = matrix._gpu_available()

    assert available is True
    assert calls[:2] == [None, "cuda"]
    assert info["default_backend"] == "cpu"
    assert "TFRT_CPU_0" in info["devices"]
    assert "cuda:0" in info["devices"]
    assert info["platforms"] == ["cpu", "cuda"]


def test_child_specs_can_add_badjac_probe0_direct_solve_rows(tmp_path) -> None:
    specs = matrix._child_specs(quick=True, outdir=tmp_path, backend="gpu", include_badjac_probe0=True)

    labels = [label for label, _, _, _ in specs]
    assert labels == [
        "provider",
        "direct_solve",
        "direct_solve_badjac_probe0",
        "direct_solve_jit_forces",
        "direct_solve_jit_forces_badjac_probe0",
        "gradient",
    ]
    probe_rows = [(label, out, args, env) for label, out, args, env in specs if label.endswith("_badjac_probe0")]
    assert {label for label, _, _, _ in probe_rows} == {
        "direct_solve_badjac_probe0",
        "direct_solve_jit_forces_badjac_probe0",
    }
    for label, out, args, env in probe_rows:
        assert out.name.endswith("_gpu_badjac_probe0.json")
        assert args[:4] == ["--max-iter", "2", "--warm-repeats", "1"]
        assert env == matrix.BADJAC_PROBE0_ENV
        assert matrix._script_for(label).name == "bench_freeb_direct_coil_solve.py"


def test_child_specs_can_add_timing_light_direct_solve_row(tmp_path) -> None:
    specs = matrix._child_specs(quick=True, outdir=tmp_path, backend="gpu", include_timing_light=True)

    labels = [label for label, _, _, _ in specs]
    assert labels == [
        "provider",
        "direct_solve",
        "direct_solve_jit_forces",
        "direct_solve_jit_forces_timing_light",
        "gradient",
    ]
    light = next(row for row in specs if row[0] == "direct_solve_jit_forces_timing_light")
    label, out, args, env = light
    assert label == "direct_solve_jit_forces_timing_light"
    assert out.name.endswith("_gpu_timing_light.json")
    assert args == ["--max-iter", "2", "--warm-repeats", "1", "--jit-forces"]
    assert env == {"VMEC_JAX_TIMING": "0", "VMEC_JAX_TIMING_DETAIL": "0"}
    assert matrix._script_for(label).name == "bench_freeb_direct_coil_solve.py"


def test_child_specs_can_add_host_policy_ablation_rows(tmp_path) -> None:
    specs = matrix._child_specs(quick=True, outdir=tmp_path, backend="gpu", include_policy_ablation=True)

    labels = [label for label, _, _, _ in specs]
    assert labels == [
        "provider",
        "direct_solve",
        "direct_solve_jit_forces",
        "direct_solve_jit_forces_no_residual_metrics",
        "direct_solve_jit_forces_no_fsq1_norms",
        "direct_solve_jit_forces_no_profile_setup",
        "direct_solve_jit_forces_host_policies_off",
        "gradient",
    ]
    ablations = {
        label: (out, args, env)
        for label, out, args, env in specs
        if label.startswith("direct_solve_jit_forces_") and label != "direct_solve_jit_forces_timing_light"
    }
    assert ablations["direct_solve_jit_forces_no_residual_metrics"][2] == {
        "VMEC_JAX_HOST_RESIDUAL_METRICS": "0"
    }
    assert ablations["direct_solve_jit_forces_no_fsq1_norms"][2] == {"VMEC_JAX_HOST_FSQ1_NORMS": "0"}
    assert ablations["direct_solve_jit_forces_no_profile_setup"][2] == {"VMEC_JAX_HOST_PROFILE_SETUP": "0"}
    assert ablations["direct_solve_jit_forces_host_policies_off"][2] == {
        "VMEC_JAX_HOST_RESIDUAL_METRICS": "0",
        "VMEC_JAX_HOST_FSQ1_NORMS": "0",
        "VMEC_JAX_HOST_PROFILE_SETUP": "0",
    }
    for label, (out, args, _env) in ablations.items():
        assert out.name.startswith("bench_freeb_direct_coil_solve_jit_forces_gpu_")
        assert args == ["--max-iter", "2", "--warm-repeats", "1", "--jit-forces"]
        assert matrix._script_for(label).name == "bench_freeb_direct_coil_solve.py"


def test_run_child_applies_and_records_badjac_probe0_env(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="child ok\n", stderr="")

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)
    monkeypatch.setattr(matrix, "_load_json", lambda path: {"status": "completed", "backend": "gpu", "cases": []})

    row = matrix._run_child(
        "direct_solve_jit_forces_badjac_probe0",
        tmp_path / "out.json",
        ["--max-iter", "2", "--jit-forces"],
        backend="gpu",
        timeout_s=1.0,
        jax_platform="cuda",
        env_overrides=matrix.BADJAC_PROBE0_ENV,
    )

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["VMEC_JAX_BADJAC_INITIAL_STATE_PROBE_ITERS"] == "0"
    assert row["env_overrides"] == matrix.BADJAC_PROBE0_ENV
    assert row["badjac_initial_state_probe_iters"] == "0"
    assert row["status"] == "completed"


def test_cpu_gpu_comparison_matches_completed_cases_and_reports_nestor_ratios() -> None:
    def timing(cold: float, warm_min: float, *, sample: float, solve: float) -> dict:
        return {
            "label": "synthetic_direct_coil_solve",
            "status": "completed",
            "cold_or_compile_s": cold,
            "warm_min_s": warm_min,
            "warm_mean_s": warm_min * 2.0,
            "solver": {
                "warm": {
                    "solve_total_s": warm_min + 1.0,
                    "iteration_loop_s": warm_min + 0.5,
                    "iteration_loop_unattributed_s": warm_min + 0.25,
                    "setup_total_s": warm_min + 0.5,
                    "setup_static_grid_rebuild_s": warm_min + 0.005,
                    "setup_freeb_policy_s": warm_min + 0.006,
                    "setup_boundary_profiles_s": warm_min + 0.007,
                    "setup_cache_key_hash_s": warm_min + 0.008,
                    "setup_ptau_constants_s": warm_min + 0.009,
                    "setup_index_constants_s": warm_min + 0.010,
                    "setup_update_constants_s": warm_min + 0.011,
                    "iteration_control_s": warm_min + 0.15,
                    "iteration_control_fsq1_s": warm_min + 0.05,
                    "iteration_control_fsq1_precond_norm_s": warm_min + 0.020,
                    "iteration_control_fsq1_scalar_build_s": warm_min + 0.018,
                    "iteration_control_fsq1_payload_get_s": warm_min + 0.025,
                    "iteration_control_fsq1_direct_get_s": warm_min + 0.015,
                    "iteration_control_fsq1_unattributed_s": warm_min + 0.010,
                    "iteration_control_badjac_s": warm_min + 0.04,
                    "iteration_control_badjac_ptau_get_s": warm_min + 0.035,
                    "iteration_control_badjac_state_jacobian_s": warm_min + 0.025,
                    "iteration_control_badjac_unattributed_s": warm_min + 0.020,
                    "iteration_control_vmec_time_s": warm_min + 0.03,
                    "iteration_control_restart_s": warm_min + 0.02,
                    "iteration_control_evolve_s": warm_min + 0.01,
                    "compute_forces_s": warm_min + 0.1,
                    "preconditioner_s": warm_min + 0.2,
                    "precond_refresh_s": warm_min + 0.365,
                    "precond_refresh_seed_s": warm_min + 0.155,
                    "precond_refresh_calls": 1,
                    "precond_reassemble_calls": 0,
                    "precond_cache_hit_count": 2,
                    "precond_refresh_seed_reuse_count": 1,
                    "precond_apply_s": warm_min + 0.375,
                    "update_s": warm_min + 0.3,
                    "update_state_s": warm_min + 0.28,
                    "update_state_ready_s": warm_min + 0.12,
                    "iteration_residual_metrics_s": warm_min + 0.125,
                    "finalize_s": warm_min + 0.25,
                    "finalize_nestor_recompute_s": warm_min + 0.04,
                    "finalize_residual_recompute_s": warm_min + 0.05,
                    "finalize_residual_device_get_s": warm_min + 0.02,
                    "finalize_diag_build_s": warm_min + 0.03,
                    "finalize_unattributed_s": warm_min + 0.01,
                    "compute_forces_per_iter_s": warm_min + 0.01,
                    "preconditioner_per_iter_s": warm_min + 0.02,
                    "update_per_iter_s": warm_min + 0.03,
                }
            },
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
    assert comparison["cpu"]["warm_solver_total_s"] == 3.0
    assert comparison["gpu"]["warm_solver_total_s"] == 1.5
    assert comparison["cpu"]["warm_compute_forces_per_iter_s"] == 2.01
    assert comparison["gpu"]["warm_compute_forces_per_iter_s"] == 0.51
    assert comparison["cpu"]["warm_setup_total_s"] == 2.5
    assert comparison["gpu"]["warm_setup_total_s"] == 1.0
    assert comparison["cpu"]["warm_setup_cache_key_hash_s"] == 2.008
    assert comparison["gpu"]["warm_setup_cache_key_hash_s"] == 0.508
    assert comparison["cpu"]["warm_iteration_residual_metrics_s"] == 2.125
    assert comparison["gpu"]["warm_iteration_residual_metrics_s"] == 0.625
    assert comparison["cpu"]["warm_finalize_s"] == 2.25
    assert comparison["gpu"]["warm_finalize_s"] == 0.75
    assert comparison["cpu"]["warm_precond_refresh_s"] == 2.365
    assert comparison["gpu"]["warm_precond_refresh_s"] == 0.865
    assert comparison["cpu"]["warm_precond_refresh_seed_s"] == 2.155
    assert comparison["gpu"]["warm_precond_refresh_seed_s"] == 0.655
    assert comparison["gpu"]["warm_precond_refresh_seed_reuse_count"] == 1
    assert comparison["cpu"]["warm_precond_apply_s"] == 2.375
    assert comparison["gpu"]["warm_precond_apply_s"] == 0.875
    assert comparison["cpu"]["warm_update_state_ready_s"] == 2.12
    assert comparison["gpu"]["warm_update_state_ready_s"] == 0.62
    assert comparison["cpu"]["warm_finalize_residual_recompute_s"] == 2.05
    assert comparison["gpu"]["warm_finalize_residual_recompute_s"] == 0.55
    assert comparison["ratios_gpu_over_cpu"] == pytest.approx({
        "cold_or_compile": 0.5,
        "warm_min": 0.25,
        "warm_mean": 0.25,
        "warm_solver_total": 0.5,
        "warm_iteration_loop": 1.0 / 2.5,
        "warm_iteration_loop_unattributed": 0.75 / 2.25,
        "warm_setup_total": 1.0 / 2.5,
        "warm_setup_static_grid_rebuild": 0.505 / 2.005,
        "warm_setup_freeb_policy": 0.506 / 2.006,
        "warm_setup_boundary_profiles": 0.507 / 2.007,
        "warm_setup_cache_key_hash": 0.508 / 2.008,
        "warm_setup_ptau_constants": 0.509 / 2.009,
        "warm_setup_index_constants": 0.510 / 2.010,
        "warm_setup_update_constants": 0.511 / 2.011,
        "warm_iteration_control": 0.65 / 2.15,
        "warm_iteration_control_fsq1": 0.55 / 2.05,
        "warm_iteration_control_fsq1_precond_norm": 0.520 / 2.020,
        "warm_iteration_control_fsq1_scalar_build": 0.518 / 2.018,
        "warm_iteration_control_fsq1_payload_get": 0.525 / 2.025,
        "warm_iteration_control_fsq1_direct_get": 0.515 / 2.015,
        "warm_iteration_control_fsq1_unattributed": 0.510 / 2.010,
        "warm_iteration_control_badjac": 0.54 / 2.04,
        "warm_iteration_control_badjac_ptau_get": 0.535 / 2.035,
        "warm_iteration_control_badjac_state_jacobian": 0.525 / 2.025,
        "warm_iteration_control_badjac_unattributed": 0.520 / 2.020,
        "warm_iteration_control_vmec_time": 0.53 / 2.03,
        "warm_iteration_control_restart": 0.52 / 2.02,
        "warm_iteration_control_evolve": 0.51 / 2.01,
        "warm_compute_forces": 0.6 / 2.1,
        "warm_preconditioner": 0.7 / 2.2,
        "warm_precond_refresh": 0.865 / 2.365,
        "warm_precond_refresh_seed": 0.655 / 2.155,
        "warm_precond_apply": 0.875 / 2.375,
        "warm_update": 0.8 / 2.3,
        "warm_update_state": 0.78 / 2.28,
        "warm_update_state_ready": 0.62 / 2.12,
        "warm_iteration_residual_metrics": 0.625 / 2.125,
        "warm_finalize": 0.75 / 2.25,
        "warm_finalize_nestor_recompute": 0.54 / 2.04,
        "warm_finalize_residual_recompute": 0.55 / 2.05,
        "warm_finalize_residual_device_get": 0.52 / 2.02,
        "warm_finalize_diag_build": 0.53 / 2.03,
        "warm_finalize_unattributed": 0.51 / 2.01,
        "active_nestor_warm_sample": 0.5,
        "active_nestor_warm_solve": 0.5,
        "final_recompute_sample": 0.5,
        "final_recompute_solve": 0.5,
        "final_external_field_sample": 0.5,
    })


def test_gpu_bottleneck_summary_ranks_warm_phase_overheads() -> None:
    comparisons = [
        {
            "label": "direct_solve_jit_forces",
            "case": "synthetic_direct_coil_solve",
            "cpu": {
                "warm_min_s": 0.05,
                "warm_solver_total_s": 0.04,
                "warm_compute_forces_s": 0.010,
                "warm_preconditioner_s": 0.002,
                "warm_iteration_residual_metrics_s": 0.001,
            },
            "gpu": {
                "warm_min_s": 0.15,
                "warm_solver_total_s": 0.11,
                "warm_compute_forces_s": 0.011,
                "warm_preconditioner_s": 0.020,
                "warm_iteration_residual_metrics_s": 0.030,
            },
            "ratios_gpu_over_cpu": {
                "warm_min": 3.0,
                "warm_solver_total": 2.75,
                "warm_compute_forces": 1.1,
                "warm_preconditioner": 10.0,
                "warm_iteration_residual_metrics": 30.0,
            },
        }
    ]

    rows = matrix._gpu_bottleneck_summary(comparisons, top_n=3)

    assert [row["phase"] for row in rows] == [
        "warm_min",
        "warm_solver_total",
        "warm_iteration_residual_metrics",
    ]
    assert rows[0]["ratio_gpu_over_cpu"] == 3.0
    assert abs(rows[0]["gpu_minus_cpu_s"] - 0.10) < 1.0e-15
    assert rows[2]["ratio_gpu_over_cpu"] == 30.0


def test_warm_phase_bottleneck_summary_ranks_single_backend_phase_overheads() -> None:
    timings = matrix._timing_snapshot(_direct_solve_payload(), include_nestor=True)
    rows = [
        {"label": "direct_solve", "backend": "cpu", "status": "completed", "timings": timings},
        {"label": "direct_solve", "backend": "gpu", "status": "failed", "timings": timings},
    ]

    summary = matrix._warm_phase_bottleneck_summary(rows, top_n=3)

    assert [(row["backend"], row["phase"]) for row in summary] == [
        ("cpu", "setup"),
        ("cpu", "preconditioner"),
        ("cpu", "residual_metrics"),
    ]
    assert summary[0]["seconds"] == 0.20
    assert summary[0]["fraction_of_solve"] == 0.20 / 0.7
    assert summary[1]["per_iter_s"] == 0.075

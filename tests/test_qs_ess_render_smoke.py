from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_renderer_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "optimization" / "render_qs_ess_publication_panel.py"
    spec = importlib.util.spec_from_file_location("render_qs_ess_publication_panel", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_sweep_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "optimization" / "generate_qs_ess_sweep.py"
    spec = importlib.util.spec_from_file_location("generate_qs_ess_sweep", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_readme_renderer_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "optimization" / "render_readme_best_optimizations.py"
    spec = importlib.util.spec_from_file_location("render_readme_best_optimizations", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_qi_renderer_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "optimization" / "render_qi_constrained_sweep.py"
    spec = importlib.util.spec_from_file_location("render_qi_constrained_sweep", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_qi_readme_cases_module():
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "optimization" / "render_qi_readme_cases.py"
    spec = importlib.util.spec_from_file_location("render_qi_readme_cases", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qs_ess_sweep_worker_jax_platform_policy():
    sweep = _load_sweep_module()

    assert sweep._default_worker_jax_platforms(None) is None
    assert sweep._default_worker_jax_platforms("auto") is None
    assert sweep._default_worker_jax_platforms("cpu") == "cpu"
    assert sweep._default_worker_jax_platforms("default") is None
    assert sweep._default_worker_jax_platforms("gpu") is None
    assert sweep._normalize_worker_jax_platforms(None) is None
    assert sweep._normalize_worker_jax_platforms("inherit") is None
    assert sweep._normalize_worker_jax_platforms("gpu") == "cuda"
    assert sweep._normalize_worker_jax_platforms("cpu,gpu") == "cpu,cuda"
    assert sweep._normalize_worker_jax_platforms("cuda") == "cuda"


def test_qs_ess_sweep_sets_missing_wall_time():
    sweep = _load_sweep_module()

    result = sweep.CaseResult(
        backend="cpu",
        problem="qa",
        max_mode=1,
        use_ess=False,
        success=False,
        crashed=True,
        message="timeout",
    )

    assert sweep._set_missing_wall_time(result, 12.5)
    assert result.total_wall_time_s == 12.5
    assert not sweep._set_missing_wall_time(result, 99.0)
    assert result.total_wall_time_s == 12.5


def test_qs_ess_sweep_profile_summary_fields():
    sweep = _load_sweep_module()
    summary = sweep._profile_summary_fields(
        {
            "profile": {
                "jacobian_total": {"count": 2, "wall_time_s": 7.5},
                "solve_forward_trial_total": {"count": 4, "wall_time_s": 3.0},
                "exact_tape_build": {"count": 2, "wall_time_s": 2.5},
                "exact_tape_build_solve_call": {"count": 2, "wall_time_s": 2.1},
                "exact_tape_build_unattributed": {"count": 2, "wall_time_s": 0.4},
                "exact_tape_solver_compute_forces_first": {"count": 2, "wall_time_s": 0.2},
                "exact_tape_solver_compute_forces_rest": {"count": 2, "wall_time_s": 0.8},
                "trial_solver_scan_total": {"count": 4, "wall_time_s": 1.75},
                "trial_solver_scan_runner_cache_lookup": {"count": 4, "wall_time_s": 0.05},
                "trial_solver_scan_runner_cache_build": {"count": 1, "wall_time_s": 0.30},
                "trial_solver_scan_runner_cache_hit_count": {"count": 1, "wall_time_s": 3.0},
                "trial_solver_scan_runner_cache_miss_count": {"count": 1, "wall_time_s": 1.0},
                "trial_solver_scan_runner_cache_bypass_count": {"count": 1, "wall_time_s": 0.0},
                "trial_solver_scan_runner_cache_hit_device_run": {
                    "count": 3,
                    "wall_time_s": 0.12,
                },
                "trial_solver_scan_runner_cache_miss_device_run": {
                    "count": 1,
                    "wall_time_s": 2.0,
                },
                "trial_solver_scan_runner_cache_bypass_device_run": {
                    "count": 1,
                    "wall_time_s": 0.0,
                },
                "trial_solver_scan_device_dispatch": {"count": 4, "wall_time_s": 0.25},
                "trial_solver_scan_device_ready": {"count": 4, "wall_time_s": 1.1},
                "trial_solver_scan_host_materialize": {"count": 4, "wall_time_s": 0.15},
                "write_wout": 0.25,
                "ignored": {"wall_time_s": "not-a-float"},
            }
        }
    )

    assert summary["profile_wall_time_s"] == pytest.approx(26.47)
    assert summary["profile_top_name"] == "jacobian_total"
    assert summary["profile_top_wall_time_s"] == pytest.approx(7.5)
    assert summary["profile_solve_forward_trial_total_wall_time_s"] == pytest.approx(3.0)
    assert summary["profile_exact_tape_build_wall_time_s"] == pytest.approx(2.5)
    assert summary["profile_exact_tape_build_solve_call_wall_time_s"] == pytest.approx(2.1)
    assert summary["profile_exact_tape_build_unattributed_wall_time_s"] == pytest.approx(0.4)
    assert summary["profile_exact_tape_solver_compute_forces_first_wall_time_s"] == pytest.approx(0.2)
    assert summary["profile_exact_tape_solver_compute_forces_rest_wall_time_s"] == pytest.approx(0.8)
    assert summary["profile_trial_solver_scan_total_wall_time_s"] == pytest.approx(1.75)
    assert summary["profile_trial_solver_scan_runner_cache_lookup_wall_time_s"] == pytest.approx(0.05)
    assert summary["profile_trial_solver_scan_runner_cache_build_wall_time_s"] == pytest.approx(0.30)
    assert summary["profile_trial_solver_scan_runner_cache_hit_count"] == pytest.approx(3.0)
    assert summary["profile_trial_solver_scan_runner_cache_miss_count"] == pytest.approx(1.0)
    assert summary["profile_trial_solver_scan_runner_cache_bypass_count"] == pytest.approx(0.0)
    assert summary[
        "profile_trial_solver_scan_runner_cache_hit_device_run_wall_time_s"
    ] == pytest.approx(0.12)
    assert summary[
        "profile_trial_solver_scan_runner_cache_miss_device_run_wall_time_s"
    ] == pytest.approx(2.0)
    assert summary[
        "profile_trial_solver_scan_runner_cache_bypass_device_run_wall_time_s"
    ] == pytest.approx(0.0)
    assert summary["profile_trial_solver_scan_device_dispatch_wall_time_s"] == pytest.approx(0.25)
    assert summary["profile_trial_solver_scan_device_ready_wall_time_s"] == pytest.approx(1.1)
    assert summary["profile_trial_solver_scan_host_materialize_wall_time_s"] == pytest.approx(0.15)
    assert summary["profile_jacobian_total_wall_time_s"] == pytest.approx(7.5)
    assert summary["profile_write_wout_wall_time_s"] == pytest.approx(0.25)

    empty = sweep._profile_summary_fields({})
    assert empty["profile_top_name"] is None
    assert empty["profile_wall_time_s"] is None


def test_qs_ess_sweep_worker_session_best_effort(monkeypatch):
    sweep = _load_sweep_module()
    calls = []

    monkeypatch.setattr(sweep.os, "setsid", lambda: calls.append("setsid"))
    sweep._start_worker_session()
    assert calls == ["setsid"]

    def fail_setsid():
        calls.append("fail")
        raise OSError("already session leader")

    monkeypatch.setattr(sweep.os, "setsid", fail_setsid)
    sweep._start_worker_session()
    assert calls == ["setsid", "fail"]


def test_qs_ess_sweep_terminates_worker_process_group(monkeypatch):
    sweep = _load_sweep_module()

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.alive = True
            self.terminated = False
            self.killed = False
            self.joins = []

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            self.joins.append(timeout)

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.alive = False

    signals = []
    proc = FakeProcess()

    def fake_killpg(pid, sig):
        signals.append((pid, sig))
        if sig == sweep.signal.SIGKILL:
            proc.alive = False

    monkeypatch.setattr(sweep.os, "killpg", fake_killpg)
    sweep._terminate_worker_process(proc, terminate_timeout_s=0.0)

    assert signals == [(12345, sweep.signal.SIGTERM), (12345, sweep.signal.SIGKILL)]
    assert proc.terminated is False
    assert proc.killed is False
    assert proc.joins == [0.0, None]


def test_qs_ess_sweep_cleans_group_when_direct_worker_already_exited(monkeypatch):
    sweep = _load_sweep_module()

    class FakeProcess:
        pid = 12345

        def is_alive(self):
            return False

        def join(self, timeout=None):
            self.join_timeout = timeout

    signals = []
    proc = FakeProcess()
    monkeypatch.setattr(sweep.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    sweep._terminate_worker_process(proc)

    assert signals == [(12345, sweep.signal.SIGTERM), (12345, sweep.signal.SIGKILL)]
    assert proc.join_timeout == 0.0


def test_qs_ess_sweep_terminates_worker_direct_child_when_group_kill_fails(monkeypatch):
    sweep = _load_sweep_module()

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.alive = True
            self.terminated = False
            self.killed = False

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            if self.terminated or self.killed:
                self.alive = False

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    monkeypatch.setattr(sweep.os, "killpg", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no pg")))
    proc = FakeProcess()
    sweep._terminate_worker_process(proc, terminate_timeout_s=0.0)

    assert proc.terminated is True
    assert proc.killed is False
    assert proc.alive is False


def test_qs_ess_sweep_lasym_copy_and_zero_asymmetric_seed():
    sweep = _load_sweep_module()
    indata = sweep.InData(
        scalars={"LASYM": False},
        indexed={"RBC": {(0, 0): 1.0}},
        source_path=Path("input.test"),
    )

    copied = sweep._copy_indata_with_lasym(indata, lasym=True)

    assert indata.get_bool("LASYM", True) is False
    assert copied.get_bool("LASYM", False) is True
    assert copied.indexed == indata.indexed
    assert sweep._boundary_include_for_indata(copied) == ("rc", "zs", "rs", "zc")

    boundary = SimpleNamespace(
        R_cos=[1.0, 0.0],
        R_sin=[0.0, 0.2],
        Z_cos=[0.0, 0.0],
        Z_sin=[0.0, 0.3],
    )
    specs = [
        SimpleNamespace(kind="rc", index=1),
        SimpleNamespace(kind="rs", index=0),
        SimpleNamespace(kind="rs", index=1),
        SimpleNamespace(kind="zc", index=0),
        SimpleNamespace(kind="zs", index=1),
    ]
    seeded = sweep._seed_zero_asymmetric_params(
        boundary_input=boundary,
        specs=specs,
        params=[0.0, 0.0, 0.0, 0.0, 0.0],
        seed=1.0e-7,
    )

    assert seeded.tolist() == [0.0, 1.0e-7, 0.0, 1.0e-7, 0.0]

    moved = seeded.copy()
    moved[1] += 2.0e-6
    moved[3] -= 3.0e-6
    stats = sweep._asymmetric_param_stats(specs, seeded, moved)

    assert stats["asymmetric_dof_count"] == 3
    assert stats["asymmetric_param_norm_delta"] > 0.0


def test_qs_ess_sweep_gpu_production_budgets_are_not_diagnostic_caps():
    sweep = _load_sweep_module()

    cfg = sweep._effective_problem_config(
        sweep.PROBLEM_CONFIGS["qh"],
        backend="gpu_prod",
        policy="direct",
        problem="qh",
        max_mode=2,
        use_ess=False,
    )

    assert cfg.max_nfev == sweep.PROBLEM_CONFIGS["qh"].max_nfev
    assert cfg.method == sweep.PROBLEM_CONFIGS["qh"].method
    assert cfg.inner_max_iter == min(sweep.PROBLEM_CONFIGS["qh"].inner_max_iter, sweep.GPU_PRODUCTION_INNER_MAX_ITER)
    assert cfg.inner_ftol == sweep.GPU_PRODUCTION_INNER_FTOL
    assert cfg.trial_max_iter == min(sweep.PROBLEM_CONFIGS["qh"].trial_max_iter, sweep.GPU_PRODUCTION_TRIAL_MAX_ITER)
    assert cfg.trial_ftol == sweep.GPU_PRODUCTION_TRIAL_FTOL

    cont_cfg = sweep._effective_problem_config(
        sweep.PROBLEM_CONFIGS["qa"],
        backend="gpu_prod",
        policy="continuation",
        problem="qa",
        max_mode=3,
        use_ess=True,
    )

    assert cont_cfg.max_nfev == sweep.PROBLEM_CONFIGS["qa"].max_nfev
    assert cont_cfg.continuation_nfev == sweep.PROBLEM_CONFIGS["qa"].continuation_nfev
    assert cont_cfg.method == sweep.PRODUCTION_AUTO_SCALAR_METHOD
    assert cont_cfg.inner_max_iter == min(
        sweep.PROBLEM_CONFIGS["qa"].inner_max_iter,
        sweep.GPU_PRODUCTION_INNER_MAX_ITER,
    )
    assert cont_cfg.inner_ftol == sweep.GPU_PRODUCTION_INNER_FTOL

    direct_qa_cfg = sweep._effective_problem_config(
        sweep.PROBLEM_CONFIGS["qa"],
        backend="gpu",
        policy="direct",
        problem="qa",
        max_mode=2,
        use_ess=False,
    )

    assert direct_qa_cfg.max_nfev == sweep.PROBLEM_CONFIGS["qa"].max_nfev

    direct_qh_ess_cfg = sweep._effective_problem_config(
        sweep.PROBLEM_CONFIGS["qh"],
        backend="gpu",
        policy="direct",
        problem="qh",
        max_mode=3,
        use_ess=True,
    )

    assert direct_qh_ess_cfg.max_nfev == sweep.PROBLEM_CONFIGS["qh"].max_nfev
    assert direct_qh_ess_cfg.method == sweep.PRODUCTION_AUTO_SCALAR_METHOD

    diag_cfg = sweep._effective_problem_config(
        sweep.PROBLEM_CONFIGS["qh"],
        backend="gpu",
        policy="direct",
        problem="qh",
        max_mode=2,
        use_ess=False,
        diagnostic_budgets=True,
    )

    assert diag_cfg.max_nfev == 4
    assert diag_cfg.method == sweep.PROBLEM_CONFIGS["qh"].method
    assert diag_cfg.inner_max_iter == 40
    assert diag_cfg.trial_max_iter == 40


def test_qs_ess_sweep_promotes_auto_scalar_for_production_high_mode_matrix():
    sweep = _load_sweep_module()

    for backend in ("cpu", "cpu_prod", "gpu", "gpu_prod"):
        for problem in ("qa", "qh", "qp", "qi"):
            cfg = sweep._effective_problem_config(
                sweep.PROBLEM_CONFIGS[problem],
                backend=backend,
                policy="continuation",
                problem=problem,
                max_mode=sweep.PRODUCTION_AUTO_SCALAR_MIN_MODE,
                use_ess=True,
            )
            assert cfg.method == sweep.PRODUCTION_AUTO_SCALAR_METHOD

    low_mode_cfg = sweep._effective_problem_config(
        sweep.PROBLEM_CONFIGS["qi"],
        backend="cpu",
        policy="direct",
        problem="qi",
        max_mode=sweep.PRODUCTION_AUTO_SCALAR_MIN_MODE - 1,
        use_ess=True,
    )
    assert low_mode_cfg.method == sweep.PROBLEM_CONFIGS["qi"].method

    diagnostic_cfg = sweep._effective_problem_config(
        sweep.PROBLEM_CONFIGS["qi"],
        backend="cpu",
        policy="direct",
        problem="qi",
        max_mode=5,
        use_ess=True,
        diagnostic_budgets=True,
    )
    assert diagnostic_cfg.method == sweep.PROBLEM_CONFIGS["qi"].method


def test_qs_ess_sweep_high_mode_stage_tags_residual_family_for_auto_scalar():
    sweep = _load_sweep_module()
    problem_cfg = sweep._effective_problem_config(
        sweep.PROBLEM_CONFIGS["qa"],
        backend="cpu",
        policy="direct",
        problem="qa",
        max_mode=3,
        use_ess=True,
    )
    cfg, indata = sweep._load_problem(problem_cfg, max_mode=3, stellarator_asymmetric=False)

    specs, opt, _iota_fn, _boundary_input = sweep._build_stage(
        problem_cfg,
        cfg,
        indata,
        3,
        solver_device="cpu",
    )

    assert max(max(abs(spec.m), abs(spec.n)) for spec in specs) == 3
    assert getattr(opt, "_objective_family") == "qs"
    assert getattr(opt, "_helicity_m") == sweep.PROBLEM_CONFIGS["qa"].helicity_m
    assert getattr(opt, "_helicity_n") == sweep.PROBLEM_CONFIGS["qa"].helicity_n
    method, _lsmr_maxiter, reason = opt._resolve_optimizer_method(problem_cfg.method, None)
    assert method == "scalar_trust"
    assert reason == "auto_scalar:high-mode-scalar-trust"


def test_qs_ess_sweep_cli_overrides_take_precedence_over_default_budgets():
    sweep = _load_sweep_module()

    override = sweep.CaseBudget(
        max_nfev=17,
        continuation_nfev=11,
        inner_max_iter=71,
        inner_ftol=2.0e-8,
        trial_max_iter=53,
        trial_ftol=3.0e-8,
    )

    cfg = sweep._effective_problem_config(
        sweep.PROBLEM_CONFIGS["qh"],
        backend="gpu",
        policy="direct",
        problem="qh",
        max_mode=5,
        use_ess=True,
        cli_budget=override,
        ess_alpha_override=2.5,
    )

    assert cfg.max_nfev == 17
    assert cfg.continuation_nfev == 11
    assert cfg.inner_max_iter == 71
    assert cfg.inner_ftol == pytest.approx(2.0e-8)
    assert cfg.trial_max_iter == 53
    assert cfg.trial_ftol == pytest.approx(3.0e-8)
    assert cfg.ess_alpha == pytest.approx(2.5)


def test_problem_configs_follow_current_seed_and_priority_policy():
    sweep = _load_sweep_module()

    qa_cfg = sweep.PROBLEM_CONFIGS["qa"]
    qh_cfg = sweep.PROBLEM_CONFIGS["qh"]
    qp_cfg = sweep.PROBLEM_CONFIGS["qp"]
    qi_cfg = sweep.PROBLEM_CONFIGS["qi"]

    assert qa_cfg.input_file.name == "input.nfp2_QA_omnigenity"
    assert qa_cfg.target_aspect == pytest.approx(5.0)
    assert qa_cfg.target_iota == pytest.approx(0.42)
    assert qa_cfg.iota_weight == pytest.approx(100.0)
    assert qa_cfg.lgradb_threshold == pytest.approx(0.30)
    assert qa_cfg.lgradb_weight == pytest.approx(0.0)
    assert qh_cfg.input_file.name == "input.nfp4_QH_warm_start"
    assert qh_cfg.target_aspect == pytest.approx(5.0)
    assert qh_cfg.iota_abs_min == pytest.approx(0.41)
    assert qh_cfg.iota_weight == pytest.approx(200.0)
    assert qh_cfg.lgradb_threshold == pytest.approx(0.30)
    assert qh_cfg.lgradb_weight == pytest.approx(0.0)
    assert qp_cfg.input_file.name == "input.nfp2_QI"
    assert qp_cfg.target_aspect == pytest.approx(5.0)
    assert qp_cfg.iota_abs_min == pytest.approx(0.41)
    assert qp_cfg.iota_weight == pytest.approx(200.0)
    assert qp_cfg.lgradb_threshold == pytest.approx(0.30)
    assert qp_cfg.lgradb_weight == pytest.approx(0.0)
    assert qp_cfg.project_input_boundary_to_max_mode
    assert qi_cfg.input_file.name == "input.minimal_seed_nfp2"
    assert qi_cfg.target_aspect == pytest.approx(5.0)
    assert qi_cfg.iota_abs_min == pytest.approx(0.41)
    assert qi_cfg.iota_weight == pytest.approx(200.0)
    assert qi_cfg.aspect_weight == pytest.approx(1.0)
    assert qi_cfg.qi_lgradb_weight == pytest.approx(0.0)
    assert qi_cfg.project_input_boundary_to_max_mode
    assert qi_cfg.qi_mboz == 18
    assert qi_cfg.qi_nboz == 18
    assert qi_cfg.qi_nphi == 151
    assert qi_cfg.qi_nalpha == 31
    assert qi_cfg.qi_n_bounce == 51
    assert qi_cfg.qi_profile_weight == pytest.approx(0.1)
    assert qi_cfg.qi_shuffle_profile_weight == pytest.approx(1.0)
    assert qi_cfg.qi_preseed_qi


def test_continuation_stage_modes_follow_omnigenity_repeated_policy(monkeypatch: pytest.MonkeyPatch):
    sweep = _load_sweep_module()

    assert sweep._stage_modes_for_problem(
        sweep.PROBLEM_CONFIGS["qa"],
        max_mode=3,
        use_mode_continuation=True,
    ) == [1, 1, 2, 2, 2, 3, 3, 3]
    assert sweep._stage_modes_for_problem(
        sweep.PROBLEM_CONFIGS["qh"],
        max_mode=2,
        use_mode_continuation=True,
    ) == [1, 1, 2, 2, 2]
    assert sweep._stage_modes_for_problem(
        sweep.PROBLEM_CONFIGS["qi"],
        max_mode=3,
        use_mode_continuation=True,
    ) == [1, 1, 2, 2, 2, 3, 3, 3]
    monkeypatch.setattr(sweep, "QI_STAGE_MODE_POLICY", "repeat")
    assert sweep._stage_modes_for_problem(
        sweep.PROBLEM_CONFIGS["qi"],
        max_mode=3,
        use_mode_continuation=True,
    ) == [3, 3, 3, 3, 3]
    assert sweep._stage_modes_for_problem(
        sweep.PROBLEM_CONFIGS["qa"],
        max_mode=3,
        use_mode_continuation=False,
    ) == [3]


def test_timed_out_partial_result_preserves_checkpoint_metrics():
    sweep = _load_sweep_module()
    result = sweep.CaseResult(
        backend="cpu",
        problem="qi",
        max_mode=3,
        use_ess=True,
        success=False,
        crashed=True,
        message="partial checkpoint after QI preseed mode 3; case still running",
        objective_final=1.2e-3,
        qs_final=1.0e-3,
        aspect_final=9.5,
        iota_final=0.43,
        nfev=17,
        njev=12,
        total_wall_time_s=321.0,
    )

    changed = sweep._mark_timed_out_result(result, elapsed_s=1200.2, case_timeout_s=1200.0)

    assert changed is True
    assert result.objective_final == pytest.approx(1.2e-3)
    assert result.qs_final == pytest.approx(1.0e-3)
    assert result.aspect_final == pytest.approx(9.5)
    assert result.iota_final == pytest.approx(0.43)
    assert result.nfev == 17
    assert result.njev == 12
    assert result.total_wall_time_s == pytest.approx(1200.2)
    assert result.crashed is True
    assert result.message.startswith("worker timed out after 1200.0 s")
    assert "partial checkpoint after QI preseed mode 3" in result.message


def test_stage_checkpoint_writes_partial_case_result_and_history(tmp_path):
    sweep = _load_sweep_module()

    class FakeOpt:
        _boundary_input = SimpleNamespace()

        def __init__(self):
            self.profile = {"exact_tape_build": {"count": 1, "wall_time_s": 0.25, "mean_wall_time_s": 0.25}}

        def save_input(self, path, params):
            Path(path).write_text(f"params={len(params)}\n")

        def save_wout(self, path, params, state=None):
            self.profile["write_wout"] = {"count": 1, "wall_time_s": 0.5, "mean_wall_time_s": 0.5}
            Path(path).write_text("wout\n")

        def _profile_dump(self):
            return dict(self.profile)

    history = {
        "history": [
            {"wall_time_s": 0.0, "objective": 2.0, "qs_objective": 1.5, "aspect": 5.0, "iota": 0.41},
            {"wall_time_s": 3.0, "objective": 1.0, "qs_objective": 0.8, "aspect": 5.1, "iota": 0.42},
        ],
        "nfev": 2,
        "njev": 1,
        "success": True,
        "message": "stage converged",
        "objective_initial": 2.0,
        "objective_final": 1.0,
        "qs_initial": 1.5,
        "qs_final": 0.8,
        "aspect_initial": 5.0,
        "aspect_final": 5.1,
        "total_wall_time_s": 3.0,
        "max_nfev": 4,
    }
    stage_result = {"x": np.asarray([], dtype=float), "_state_final": object(), "_history_dump": history}

    case_result = sweep._write_case_checkpoint(
        metadata=sweep.CaseRunMetadata(
            problem_cfg=sweep.PROBLEM_CONFIGS["qa"],
            cfg=SimpleNamespace(nfp=2),
            backend="cpu",
            problem="qa",
            max_mode=1,
            use_ess=False,
            output_dir=tmp_path,
            policy="continuation",
            solver_device="cpu",
            jax_platforms="cpu",
            jax_backend="cpu",
            jax_device_kind="CPU",
            stellarator_asymmetric=False,
        ),
        result_path=tmp_path / "case_result.json",
        stage_results=[("QA mode 1", 1, stage_result)],
        latest_specs=[],
        latest_opt=FakeOpt(),
        latest_params_final=np.asarray([], dtype=float),
        write_artifacts=True,
        success=False,
        crashed=True,
        message="partial checkpoint after QA mode 1; case still running",
    )

    saved_case = json.loads((tmp_path / "case_result.json").read_text())
    saved_history = json.loads((tmp_path / "history.json").read_text())
    saved_checkpoint = json.loads((tmp_path / "stage_checkpoint.json").read_text())

    assert case_result.crashed is True
    assert case_result.success is False
    assert case_result.objective_final == pytest.approx(1.0)
    assert case_result.iota_final == pytest.approx(0.42)
    assert case_result.profile_write_wout_wall_time_s == pytest.approx(0.5)
    assert saved_case["objective_final"] == pytest.approx(1.0)
    assert saved_case["crashed"] is True
    assert saved_history["stage_labels"] == ["QA mode 1"]
    assert saved_history["history"][-1]["stage"] == "QA mode 1"
    assert saved_checkpoint["partial"] is True
    assert saved_checkpoint["iota_final"] == pytest.approx(0.42)
    assert saved_checkpoint["history_path"] == "history.json"
    assert saved_checkpoint["case_result_path"] == "case_result.json"
    assert (tmp_path / "input.final").exists()
    assert (tmp_path / "wout_final.nc").exists()


def test_run_problem_stages_invokes_stage_checkpoint_callback(monkeypatch):
    sweep = _load_sweep_module()

    class FakeOpt:
        def __init__(self, mode):
            self.mode = mode

        def run(self, params0, **_kwargs):
            return {
                "x": np.ones(len(params0), dtype=float) * self.mode,
                "_history_dump": {
                    "history": [
                        {"wall_time_s": 0.0, "objective": 2.0, "qs_objective": 1.0, "aspect": 5.0},
                        {"wall_time_s": 1.0, "objective": 1.0, "qs_objective": 0.5, "aspect": 5.1},
                    ],
                    "nfev": 2,
                    "njev": 1,
                    "success": True,
                    "message": "ok",
                    "objective_initial": 2.0,
                    "objective_final": 1.0,
                    "qs_initial": 1.0,
                    "qs_final": 0.5,
                    "aspect_initial": 5.0,
                    "aspect_final": 5.1,
                    "total_wall_time_s": 1.0,
                    "max_nfev": 2,
                },
            }

    def fake_build_stage(_problem_cfg, _cfg, _indata, mode, *, solver_device):
        specs = [SimpleNamespace(kind="rc", index=idx) for idx in range(mode)]
        return specs, FakeOpt(mode), None, SimpleNamespace()

    callbacks = []
    monkeypatch.setattr(sweep, "_stage_modes_for_problem", lambda *_args, **_kwargs: [1, 2])
    monkeypatch.setattr(sweep, "_build_stage", fake_build_stage)
    monkeypatch.setattr(sweep.vj, "lift_boundary_params", lambda _prev, _params, specs: np.zeros(len(specs)))

    sweep._run_problem_stages(
        problem_cfg=sweep.PROBLEM_CONFIGS["qa"],
        problem="qa",
        max_mode=2,
        use_ess=False,
        use_mode_continuation=True,
        solver_device="cpu",
        cfg=SimpleNamespace(),
        indata=SimpleNamespace(),
        stage_label_prefix="QA",
        params_stage=None,
        prev_specs=None,
        stellarator_asymmetric=False,
        stage_completed_callback=lambda stage, specs, _opt, _params0, params: callbacks.append(
            (stage[0], stage[1], len(specs), params.tolist())
        ),
    )

    assert callbacks == [("QA mode 1", 1, 1, [1.0]), ("QA mode 2", 2, 2, [2.0, 2.0])]


def test_qi_renderer_plots_only_qi_refinement_after_qp_preseed():
    renderer = _load_qi_renderer_module()

    history = [
        {"stage": "QP preseed max_mode=1", "objective": 10.0},
        {"stage": "QP preseed max_mode=1", "objective": 1.0},
        {"stage": "QI max_mode=1", "objective": 0.2},
        {"stage": "QI max_mode=1", "objective": 0.05},
    ]

    segments = renderer._plotted_history_segments({"qi_qp_preseed": True}, history)

    assert len(segments) == 1
    assert [item["stage"] for item in segments[0]] == ["QI max_mode=1", "QI max_mode=1"]


def test_qi_readme_cases_concatenate_histories_and_record_lambda_scan(tmp_path):
    renderer = _load_qi_readme_cases_module()

    h1 = tmp_path / "stage1" / "history.json"
    h2 = tmp_path / "stage2" / "history.json"
    h1.parent.mkdir()
    h2.parent.mkdir()
    h1.write_text(
        json.dumps(
            {
                "label": "QI optimization (max_mode=3, ESS)",
                "history": [
                    {"wall_time_s": 0.0, "objective": 100.0},
                    {"wall_time_s": 2.0, "objective": 1.0},
                ],
            }
        )
    )
    h2.write_text(
        json.dumps(
            {
                "label": "QI boundary-reference baseline (max_mode=4)",
                "history": [
                    {"wall_time_s": 0.0, "objective": 0.5},
                    {"wall_time_s": 3.0, "objective": 0.4},
                ],
            }
        )
    )
    h3 = tmp_path / "history_duplicate.json"
    h3.write_text(h1.read_text())
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            [
                {"lambda": 0.99, "legacy_qi": 2.0e-3, "mirror": 0.2, "selected": False},
                {"lambda": 1.01, "legacy_qi": 5.0e-4, "mirror": 0.3, "selected": True},
            ]
        )
    )
    case = renderer.QICase(
        label="synthetic",
        input_file=tmp_path / "input.synthetic",
        output_dir=tmp_path,
        initial_wout=tmp_path / "wout_initial.nc",
        note="test",
        history_paths=(h1, h3, h2),
        preconditioner_summary=summary,
    )

    segments = renderer._history_segments(case)
    full_wall_s, stage_count, point_count = renderer._history_summary(case)
    precond_points, selected_lambda, selected_qi, selected_mirror = renderer._preconditioner_summary(case)

    assert stage_count == 2
    assert point_count == 4
    assert [segment["path"] for segment in segments] == [h1, h2]
    assert full_wall_s == pytest.approx(5.0)
    assert segments[1]["wall_time_s"][0] == pytest.approx(2.0)
    assert segments[0]["label"] == "seed solve"
    assert precond_points == 2
    assert selected_lambda == pytest.approx(1.01)
    assert selected_qi == pytest.approx(5.0e-4)
    assert selected_mirror == pytest.approx(0.3)
    best_so_far = renderer._stage_normalized_best_so_far(np.asarray([2.0, 3.0, 1.0, 1.5]))
    assert best_so_far.tolist() == pytest.approx([1.0, 1.0, 0.5, 0.5])


def test_qi_renderer_marks_raw_fallback_legacy_as_nonpromotable(tmp_path, monkeypatch):
    renderer = _load_qi_renderer_module()
    monkeypatch.setattr(renderer, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(renderer, "BEST_JSON", tmp_path / "best.json")

    fallback_dir = tmp_path / "cpu" / "continuation" / "qi" / "mode3" / "ess" / "fallback"
    legacy_dir = tmp_path / "cpu" / "continuation" / "qi" / "mode2" / "ess" / "legacy"
    fallback_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    for case_dir, record in (
        (
            fallback_dir,
            {
                "qi_raw_total": 1.0e-6,
                "objective_final": 1.0e-6,
                "max_mode": 3,
                "output_dir": str(fallback_dir),
            },
        ),
        (
            legacy_dir,
            {
                "qi_raw_total": 8.0e-3,
                "qi_legacy_total": 8.0e-3,
                "objective_final": 8.0e-3,
                "max_mode": 2,
                "output_dir": str(legacy_dir),
            },
        ),
    ):
        (case_dir / "case_result.json").write_text(
            json.dumps(
                {
                    "backend": "cpu",
                    "policy": "continuation",
                    "problem": "qi",
                    "use_ess": True,
                    "success": True,
                    "crashed": False,
                    "message": "ok",
                    "target_aspect": renderer.TARGET_ASPECT,
                    "input_nfp": renderer.QI_INPUT_NFP,
                    "qi_qp_preseed": False,
                    "qi_mirror_ratio_max": 0.21,
                    "qi_mirror_ratio_target": 0.21,
                    "qi_max_elongation": 5.0,
                    "qi_elongation_target": 8.0,
                    "aspect_final": renderer.TARGET_ASPECT,
                    "iota_final": renderer.TARGET_ABS_IOTA_MIN,
                    **record,
                }
            )
        )

    rows = renderer._discover_qi_results()
    by_mode = {row["max_mode"]: row for row in rows}

    assert by_mode[3]["qi_legacy_total"] == 1.0e-6
    assert by_mode[3]["qi_legacy_source"] == "raw_fallback"
    assert by_mode[2]["qi_legacy_source"] == "legacy"
    assert renderer._write_best(rows)["max_mode"] == 2


def test_qi_renderer_prefers_engineering_clean_candidate_over_lower_qi_mirror_failure(tmp_path, monkeypatch):
    renderer = _load_qi_renderer_module()
    monkeypatch.setattr(renderer, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(renderer, "BEST_JSON", tmp_path / "best.json")

    clean_dir = tmp_path / "cpu" / "continuation" / "qi" / "mode2" / "ess" / "clean"
    low_qi_dir = tmp_path / "cpu" / "continuation" / "qi" / "mode3" / "ess" / "low_qi_bad_mirror"
    non_qi_dir = tmp_path / "cpu" / "direct" / "qi" / "mode1" / "ess" / "non_qi_good_engineering"
    clean_dir.mkdir(parents=True)
    low_qi_dir.mkdir(parents=True)
    non_qi_dir.mkdir(parents=True)

    base = {
        "backend": "cpu",
        "policy": "continuation",
        "problem": "qi",
        "use_ess": True,
        "success": True,
        "crashed": False,
        "message": "ok",
        "target_aspect": renderer.TARGET_ASPECT,
        "input_nfp": renderer.QI_INPUT_NFP,
        "qi_qp_preseed": False,
        "qi_mirror_ratio_target": 0.21,
        "qi_elongation_target": 8.0,
        "iota_final": 0.45,
    }
    for case_dir, record in (
        (
            clean_dir,
            {
                "max_mode": 2,
                "output_dir": str(clean_dir),
                "objective_final": 2.0e-2,
                "qi_legacy_total": 2.0e-2,
                "qi_raw_total": 2.0e-2,
                "qi_mirror_ratio_max": 0.20,
                "qi_max_elongation": 7.5,
                "aspect_final": renderer.TARGET_ASPECT * 1.01,
            },
        ),
        (
            low_qi_dir,
            {
                "max_mode": 3,
                "output_dir": str(low_qi_dir),
                "objective_final": 1.0e-3,
                "qi_legacy_total": 1.0e-3,
                "qi_raw_total": 1.0e-3,
                "qi_mirror_ratio_max": 0.50,
                "qi_max_elongation": 7.0,
                "aspect_final": renderer.TARGET_ASPECT,
            },
        ),
        (
            non_qi_dir,
            {
                "max_mode": 1,
                "output_dir": str(non_qi_dir),
                "objective_final": 1.0e-1,
                "qi_legacy_total": 1.0e-1,
                "qi_raw_total": 1.0e-1,
                "qi_mirror_ratio_max": 0.10,
                "qi_max_elongation": 4.0,
                "aspect_final": renderer.TARGET_ASPECT,
            },
        ),
    ):
        (case_dir / "case_result.json").write_text(json.dumps({**base, **record}))

    rows = renderer._discover_qi_results()

    assert renderer._write_best(rows)["max_mode"] == 2


def test_qs_ess_renderer_ignores_legacy_backendless_records(tmp_path, monkeypatch):
    renderer = _load_renderer_module()
    monkeypatch.setattr(renderer, "OUTPUT_ROOT", tmp_path)

    legacy_dir = tmp_path / "qa" / "mode2" / "no_ess"
    explicit_dir = tmp_path / "cpu" / "continuation" / "qa" / "mode2" / "no_ess"
    direct_legacy_dir = tmp_path / "direct" / "qa" / "mode3" / "ess"
    legacy_dir.mkdir(parents=True)
    explicit_dir.mkdir(parents=True)
    direct_legacy_dir.mkdir(parents=True)

    legacy_record = {
        "problem": "qa",
        "max_mode": 2,
        "use_ess": False,
        "success": True,
        "crashed": False,
        "message": "legacy",
        "objective_final": 1.0,
    }
    explicit_record = {
        **legacy_record,
        "backend": "cpu",
        "policy": "continuation",
        "message": "explicit",
        "objective_final": 0.2,
    }
    direct_record = {
        "problem": "qa",
        "max_mode": 3,
        "use_ess": True,
        "success": False,
        "crashed": True,
        "message": "worker timed out after 5.0 s",
        "objective_final": None,
        "output_dir": "/remote/nonexistent/qs_case",
    }

    legacy_path = legacy_dir / "case_result.json"
    explicit_path = explicit_dir / "case_result.json"
    direct_path = direct_legacy_dir / "case_result.json"
    legacy_path.write_text(json.dumps(legacy_record))
    explicit_path.write_text(json.dumps(explicit_record))
    direct_path.write_text(json.dumps(direct_record))

    # Make the stale legacy file newer; explicit backend metadata should still win.
    os.utime(explicit_path, (100.0, 100.0))
    os.utime(legacy_path, (200.0, 200.0))

    discovered = renderer._discover_results()
    lookup = renderer._result_lookup(discovered)

    preferred = lookup[("cpu", False, "continuation", "qa", 2, False)]
    assert preferred.message == "explicit"
    assert preferred.objective_final == 0.2

    assert ("cpu", False, "direct", "qa", 3, True) not in lookup


def test_qs_ess_renderer_normalizes_gpu_backend_labels(tmp_path, monkeypatch):
    renderer = _load_renderer_module()
    monkeypatch.setattr(renderer, "OUTPUT_ROOT", tmp_path)

    case_dir = tmp_path / "gpu_prod" / "direct" / "qh" / "mode2" / "no_ess"
    case_dir.mkdir(parents=True)
    (case_dir / "case_result.json").write_text(
        json.dumps(
            {
                "backend": "gpu_prod",
                "policy": "direct",
                "problem": "qh",
                "max_mode": 2,
                "use_ess": False,
                "success": True,
                "crashed": False,
                "message": "ok",
                "objective_final": 1.0,
                "output_dir": str(case_dir),
            }
        )
    )

    discovered = renderer._discover_results()
    assert discovered[0].backend == "gpu"
    assert renderer._result_key(discovered[0]) == ("gpu", False, "direct", "qh", 2, False)


def test_qs_ess_renderer_separates_lasym_records(tmp_path, monkeypatch):
    renderer = _load_renderer_module()
    monkeypatch.setattr(renderer, "OUTPUT_ROOT", tmp_path)

    sym_dir = tmp_path / "cpu" / "continuation" / "qa" / "mode1" / "no_ess"
    asym_dir = tmp_path / "cpu" / "asymmetric" / "continuation" / "qa" / "mode1" / "no_ess"
    sym_dir.mkdir(parents=True)
    asym_dir.mkdir(parents=True)
    base_record = {
        "backend": "cpu",
        "policy": "continuation",
        "problem": "qa",
        "max_mode": 1,
        "use_ess": False,
        "success": True,
        "crashed": False,
        "message": "ok",
        "output_dir": str(sym_dir),
    }
    (sym_dir / "case_result.json").write_text(json.dumps({**base_record, "objective_final": 1.0}))
    (asym_dir / "case_result.json").write_text(
        json.dumps(
            {
                **base_record,
                "stellarator_asymmetric": True,
                "objective_final": 0.25,
                "asymmetric_dof_count": 8,
                "output_dir": str(asym_dir),
            }
        )
    )

    discovered = renderer._discover_results()
    lookup = renderer._result_lookup(discovered)

    assert lookup[("cpu", False, "continuation", "qa", 1, False)].objective_final == 1.0
    asym = lookup[("cpu", True, "continuation", "qa", 1, False)]
    assert asym.objective_final == 0.25
    assert asym.asymmetric_dof_count == 8
    assert ("cpu", True, "qa", "continuation") in renderer._available_row_specs(discovered)


def test_readme_renderer_uses_preoptimization_initial_wout(tmp_path, monkeypatch):
    renderer = _load_readme_renderer_module()

    input_file = tmp_path / "input.raw"
    input_file.write_text(
        "&INDATA\n"
        "  NFP = 2\n"
        "  RBC(0,0) = 1.0\n"
        "  RBC(0,1) = 0.2\n"
        "  ZBS(0,1) = 0.3\n"
        "/\n"
    )
    continuation_dir = tmp_path / "cpu" / "continuation" / "qh" / "mode3" / "ess"
    mode1_dir = tmp_path / "cpu" / "continuation" / "qh" / "mode1" / "ess"
    direct_qi_dir = tmp_path / "cpu" / "direct" / "qi" / "mode2" / "ess"
    direct_qa_dir = tmp_path / "cpu" / "direct" / "qa" / "mode3" / "ess"
    for directory in (continuation_dir, mode1_dir, direct_qi_dir, direct_qa_dir):
        directory.mkdir(parents=True)
        (directory / "wout_initial.nc").write_text("stage-local")
    (mode1_dir / "wout_initial.nc").write_text("pre-mode-1")
    (direct_qi_dir / "wout_original.nc").write_text("original")

    def fake_read_wout(_path):
        return SimpleNamespace(
            nfp=2,
            xm=np.asarray([0, 1], dtype=int),
            xn=np.asarray([0, 0], dtype=int),
            rmnc=np.asarray([[1.0, 0.2]], dtype=float),
            rmns=np.zeros((1, 2), dtype=float),
            zmnc=np.zeros((1, 2), dtype=float),
            zmns=np.asarray([[0.0, 0.3]], dtype=float),
        )

    monkeypatch.setattr(renderer, "read_wout", fake_read_wout)
    continuation = renderer.BestRun(
        problem="qh",
        policy="continuation",
        max_mode=3,
        use_ess=True,
        objective_final=1.0,
        aspect_final=7.0,
        iota_final=0.4,
        total_wall_time_s=1.0,
        output_dir=continuation_dir,
        input_file=input_file,
    )
    qi = renderer.BestRun(
        problem="qi",
        policy="direct",
        max_mode=2,
        use_ess=True,
        objective_final=1.0,
        aspect_final=7.0,
        iota_final=0.4,
        total_wall_time_s=1.0,
        output_dir=direct_qi_dir,
        input_file=input_file,
    )
    qa = renderer.BestRun(
        problem="qa",
        policy="direct",
        max_mode=3,
        use_ess=True,
        objective_final=1.0,
        aspect_final=6.0,
        iota_final=0.4,
        total_wall_time_s=1.0,
        output_dir=direct_qa_dir,
        input_file=input_file,
    )

    assert renderer._preoptimization_wout_path(continuation) == mode1_dir / "wout_initial.nc"
    assert renderer._preoptimization_wout_path(qi) == direct_qi_dir / "wout_original.nc"
    assert renderer._preoptimization_wout_path(qa) == direct_qa_dir / "wout_initial.nc"


def test_readme_renderer_derives_raw_wout_for_nonraw_qp_mode1_initial_wout(tmp_path, monkeypatch):
    renderer = _load_readme_renderer_module()

    input_file = tmp_path / "input.raw_qp"
    input_file.write_text(
        "&INDATA\n"
        "  NFP = 2\n"
        "  RBC(0,0) = 1.0\n"
        "  RBC(0,1) = 0.2\n"
        "  ZBS(0,1) = 0.3\n"
        "/\n"
    )
    continuation_dir = tmp_path / "cpu" / "continuation" / "qp" / "mode3" / "ess"
    mode1_dir = tmp_path / "cpu" / "continuation" / "qp" / "mode1" / "ess"
    continuation_dir.mkdir(parents=True)
    mode1_dir.mkdir(parents=True)
    (mode1_dir / "wout_initial.nc").write_text("optimized-mode1")
    derived_wout = continuation_dir / "wout_original.nc"

    def fake_read_wout(_path):
        return SimpleNamespace(
            nfp=2,
            xm=np.asarray([0, 1], dtype=int),
            xn=np.asarray([0, 0], dtype=int),
            rmnc=np.asarray([[1.0, 0.95]], dtype=float),
            rmns=np.zeros((1, 2), dtype=float),
            zmnc=np.zeros((1, 2), dtype=float),
            zmns=np.asarray([[0.0, 0.3]], dtype=float),
        )

    monkeypatch.setattr(renderer, "read_wout", fake_read_wout)
    derive_calls = []
    monkeypatch.setattr(
        renderer,
        "_derive_raw_initial_wout",
        lambda raw_input, output_dir: derive_calls.append((raw_input, output_dir)) or derived_wout,
    )
    run = renderer.BestRun(
        problem="qp",
        policy="continuation",
        max_mode=3,
        use_ess=True,
        objective_final=1.0,
        aspect_final=5.0,
        iota_final=0.4,
        total_wall_time_s=1.0,
        output_dir=continuation_dir,
        input_file=input_file,
    )

    assert renderer._preoptimization_wout_path(run) == derived_wout
    assert derive_calls == [(input_file, continuation_dir)]


def test_readme_best_renderer_accepts_vmec_canonical_phase_equivalent_wout(monkeypatch, tmp_path):
    renderer = _load_readme_renderer_module()
    input_file = tmp_path / "input.phase"
    input_file.write_text(
        "&INDATA\n"
        "  NFP = 3\n"
        "  RBC(0,0) = 1.0\n"
        "  RBC(1,0) = -0.2\n"
        "  ZBS(1,0) = -0.3\n"
        "  RBC(-2,1) = -0.04\n"
        "  ZBS(-2,1) = -0.05\n"
        "  RBC(-2,2) = 0.06\n"
        "  ZBS(-2,2) = -0.07\n"
        "/\n"
    )

    wout = SimpleNamespace(
        nfp=3,
        xm=np.asarray([0, 0, 1, 2], dtype=int),
        xn=np.asarray([0, 3, 6, 6], dtype=int),
        rmnc=np.asarray([[1.0, -0.2, 0.04, 0.06]], dtype=float),
        rmns=np.zeros((1, 4), dtype=float),
        zmnc=np.zeros((1, 4), dtype=float),
        zmns=np.asarray([[0.0, -0.3, -0.05, 0.07]], dtype=float),
    )
    monkeypatch.setattr(renderer, "read_wout", lambda _path: wout)

    assert renderer._boundary_mismatches(input_file, tmp_path / "wout.nc") == []


def test_readme_best_summary_records_initial_and_final_wout(tmp_path, monkeypatch):
    renderer = _load_readme_renderer_module()

    out_csv = tmp_path / "readme_best_optimizations.csv"
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    input_file = tmp_path / "input.raw"
    input_file.write_text("&INDATA\n  NFP = 2\n  RBC(0,0) = 1.0\n/\n")
    initial_wout = output_dir / "wout_initial.nc"
    final_wout = output_dir / "wout_final.nc"
    initial_wout.write_text("initial")
    final_wout.write_text("final")

    monkeypatch.setattr(renderer, "OUT_CSV", out_csv)
    monkeypatch.setattr(renderer, "_preoptimization_wout_path", lambda _run: initial_wout)
    run = renderer.BestRun(
        problem="qa",
        policy="direct",
        max_mode=1,
        use_ess=False,
        objective_final=1.0,
        aspect_final=6.0,
        iota_final=0.42,
        total_wall_time_s=60.0,
        output_dir=output_dir,
        input_file=input_file,
    )

    renderer._write_readme_summary([run])

    with out_csv.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["initial_wout"] == str(initial_wout)
    assert row["final_wout"] == str(final_wout)


def test_readme_renderer_filters_qi_rows_by_qi_target_aspect():
    renderer = _load_readme_renderer_module()

    current_qi = {
        "target_aspect": "5.0",
        "iota_abs_min": str(renderer.TARGET_ABS_IOTA_MIN),
        "aspect_final": "5.0",
    }
    legacy_qi = {
        "target_aspect": "10.0",
        "iota_abs_min": str(renderer.TARGET_ABS_IOTA_MIN),
        "aspect_final": "9.9",
    }

    assert renderer._is_current_qi_row(current_qi)
    assert not renderer._is_current_qi_row(legacy_qi)


def test_readme_renderer_can_use_dedicated_qi_result_dir(tmp_path):
    renderer = _load_readme_renderer_module()

    result_dir = tmp_path / "results" / "qi_opt" / "ess" / "nfp2_qi"
    result_dir.mkdir(parents=True)
    (result_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "target_aspect": 5.0,
                "aspect": 5.01,
                "mean_iota": -0.50,
                "qi_raw_total": 1.1e-3,
                "qi_legacy_total": 3.0e-4,
                "qi_mirror_ratio_max": 0.22,
                "qi_mirror_ratio_target": 0.30,
                "qi_max_elongation": 6.4,
                "qi_elongation_target": 8.2,
            }
        )
    )
    (result_dir / "history.json").write_text(
        json.dumps({"objective_final": 1.2e-2, "total_wall_time_s": 600.0})
    )

    row = renderer._qi_default_row_from_result_dir(result_dir)

    assert row is not None
    assert row["policy"] == "qi_default"
    assert float(row["target_aspect"]) == pytest.approx(5.0)
    assert float(row["qi_legacy_total"]) == pytest.approx(3.0e-4)


def test_qs_ess_renderer_keeps_partial_lasym_publication_groups():
    renderer = _load_renderer_module()

    sym = renderer.CaseResult(
        backend="cpu",
        policy="continuation",
        problem="qa",
        max_mode=1,
        use_ess=False,
        success=True,
        crashed=False,
        message="ok",
        stellarator_asymmetric=False,
    )
    partial_asym = renderer.CaseResult(
        backend="cpu",
        policy="direct",
        problem="qa",
        max_mode=1,
        use_ess=False,
        success=True,
        crashed=False,
        message="ok",
        stellarator_asymmetric=True,
    )
    complete_gpu_asym = [
        renderer.CaseResult(
            backend="gpu",
            policy="direct",
            problem=problem,
            max_mode=mode,
            use_ess=use_ess,
            success=True,
            crashed=False,
            message="ok",
            stellarator_asymmetric=True,
        )
        for problem in renderer.PROBLEMS
        for mode in renderer.MODES_BY_POLICY["direct"]
        for use_ess in renderer.ESS_OPTIONS
    ]

    filtered = renderer._publication_results([sym, partial_asym, *complete_gpu_asym])
    lookup = renderer._result_lookup(filtered)

    assert ("cpu", False, "continuation", "qa", 1, False) in lookup
    assert ("cpu", True, "direct", "qa", 1, False) in lookup
    assert ("gpu", True, "direct", "qa", 1, False) in lookup


def test_qs_ess_renderer_flags_nonpositive_bmag():
    renderer = _load_renderer_module()
    result = renderer.CaseResult(
        backend="cpu",
        policy="direct",
        problem="qi",
        max_mode=1,
        use_ess=False,
        success=True,
        crashed=False,
        message="ok",
        bmag_min=-1.0,
        bmag_max=2.0,
        bmag_nonpositive_fraction=0.1,
        bmag_finite=True,
    )

    assert renderer._status_label(result) == "bad |B|"


def _write_case(
    case_dir: Path,
    *,
    policy: str,
    problem: str,
    max_mode: int,
    use_ess: bool,
    objective_final: float,
    aspect_final: float,
    iota_final: float | None,
    wall_s: float,
    success: bool = True,
    nfp_wout: Path,
) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nfp_wout, case_dir / "wout_final.nc")
    history = {
        "history": [
            {"objective": 1.0, "wall_time_s": 0.0},
            {"objective": objective_final, "wall_time_s": wall_s},
        ],
        "stage_boundaries": [1],
        "success": success,
        "message": "`xtol` termination condition is satisfied." if success else "The maximum number of function evaluations is exceeded.",
        "nfev": 2,
        "njev": 1,
        "objective_initial": 1.0,
        "objective_final": objective_final,
        "qs_initial": 1.0,
        "qs_final": objective_final,
        "aspect_initial": aspect_final + 0.1,
        "aspect_final": aspect_final,
        "target_aspect": aspect_final,
    }
    if iota_final is not None:
        history["target_iota"] = iota_final
    (case_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {
        "problem": problem,
        "max_mode": max_mode,
        "use_ess": use_ess,
        "success": success,
        "crashed": False,
        "message": history["message"],
        "policy": policy,
        "objective_final": objective_final,
        "qs_final": objective_final,
        "aspect_final": aspect_final,
        "iota_final": iota_final,
        "nfev": 2,
        "njev": 1,
        "total_wall_time_s": wall_s,
        "output_dir": str(case_dir),
    }


def test_qs_ess_renderer_handles_partial_direct_matrix(tmp_path):
    pytest.importorskip("matplotlib")

    renderer = _load_renderer_module()
    root = Path(__file__).resolve().parents[1]
    wout_path = root / "examples" / "data" / "wout_circular_tokamak.nc"

    results = [
        renderer.CaseResult(
            **_write_case(
                tmp_path / "continuation" / "qa" / "mode1" / "no_ess",
                policy="continuation",
                problem="qa",
                max_mode=1,
                use_ess=False,
                objective_final=9.3e-3,
                aspect_final=6.0,
                iota_final=0.3942,
                wall_s=300.0,
                nfp_wout=wout_path,
            )
        ),
        renderer.CaseResult(
            **_write_case(
                tmp_path / "continuation" / "qa" / "mode1" / "ess",
                policy="continuation",
                problem="qa",
                max_mode=1,
                use_ess=True,
                objective_final=9.3e-3,
                aspect_final=6.0,
                iota_final=0.3942,
                wall_s=280.0,
                nfp_wout=wout_path,
            )
        ),
        renderer.CaseResult(
            **_write_case(
                tmp_path / "continuation" / "qh" / "mode1" / "no_ess",
                policy="continuation",
                problem="qh",
                max_mode=1,
                use_ess=False,
                objective_final=2.16e-1,
                aspect_final=7.0,
                iota_final=None,
                wall_s=130.0,
                nfp_wout=wout_path,
            )
        ),
        renderer.CaseResult(
            **_write_case(
                tmp_path / "continuation" / "qh" / "mode1" / "ess",
                policy="continuation",
                problem="qh",
                max_mode=1,
                use_ess=True,
                objective_final=2.16e-1,
                aspect_final=7.0,
                iota_final=None,
                wall_s=135.0,
                nfp_wout=wout_path,
            )
        ),
        renderer.CaseResult(
            **_write_case(
                tmp_path / "direct" / "qa" / "mode2" / "no_ess",
                policy="direct",
                problem="qa",
                max_mode=2,
                use_ess=False,
                objective_final=4.5e-4,
                aspect_final=5.999,
                iota_final=0.4066,
                wall_s=1110.0,
                nfp_wout=wout_path,
            )
        ),
        renderer.CaseResult(
            **_write_case(
                tmp_path / "direct" / "qa" / "mode2" / "ess",
                policy="direct",
                problem="qa",
                max_mode=2,
                use_ess=True,
                objective_final=1.58e-4,
                aspect_final=6.0,
                iota_final=0.4095,
                wall_s=893.0,
                success=False,
                nfp_wout=wout_path,
            )
        ),
    ]

    payloads = renderer._load_payloads(results)

    objective_png = tmp_path / "objective.png"
    objective_pdf = tmp_path / "objective.pdf"
    direct_png = tmp_path / "direct.png"
    direct_pdf = tmp_path / "direct.pdf"
    summary_png = tmp_path / "summary.png"
    summary_pdf = tmp_path / "summary.pdf"

    renderer._plot_objective_panel_all_policies(results, objective_png, objective_pdf)
    renderer._plot_state_atlas(results, payloads, policy="direct", outpath_png=direct_png, outpath_pdf=direct_pdf)
    renderer._plot_summary_tables(results, summary_png, summary_pdf)

    assert objective_png.exists()
    assert objective_pdf.exists()
    assert direct_png.exists()
    assert direct_pdf.exists()
    assert summary_png.exists()
    assert summary_pdf.exists()

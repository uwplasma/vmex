from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    script = ROOT / "examples" / "optimization" / "qi_staged_runner.py"
    spec = importlib.util.spec_from_file_location("qi_staged_runner_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qi_staged_runner_builds_external_input_cli_and_environment(tmp_path: Path) -> None:
    runner = _load_runner()
    config = runner.QIStagedCaseConfig(
        name="qi_nfp2",
        input_file=ROOT / "examples" / "data" / "input.minimal_seed_nfp2",
        output_dir=tmp_path / "out",
        max_mode=3,
        policy="direct",
        policy_case="nfp2_qi",
        reference_input=ROOT / "examples" / "data" / "input.nfp2_QI",
        reference_accept_as_baseline=True,
        backend_label="gpu",
        solver_device="gpu",
        worker_jax_platforms="gpu",
        use_ess=False,
        stage_mode_policy="repeat",
        max_nfev=5,
        continuation_nfev=4,
        inner_max_iter=21,
        inner_ftol=1.0e-8,
        trial_max_iter=22,
        trial_ftol=2.0e-8,
        ess_alpha=1.7,
        method="scipy",
        target_aspect=5.0,
        target_abs_iota_min=0.41,
        max_mirror_ratio=0.3,
        max_elongation=10.0,
        qi_mboz=10,
        qi_nboz=11,
        qi_nphi=61,
        qi_nalpha=13,
        qi_n_bounce=17,
        mirror_ramp_stages=({"name": "cleanup", "max_nfev": 5, "mirror_weight": 20.0},),
        make_plots=False,
    )

    env = runner._build_qi_staged_env(config)
    args = runner._build_qi_staged_args(config)
    joined = " ".join(str(item) for item in args)

    assert env["JAX_PLATFORMS"] == "cuda"
    assert "--input-file" in args
    assert "--ess-alpha" in args and "1.7" in args
    assert "--method" in args and "scipy" in args
    assert str(ROOT / "examples" / "data" / "input.minimal_seed_nfp2") in args
    assert str(tmp_path / "out") in args
    assert "--max-mode" in args and "3" in args
    assert "--no-use-mode-continuation" in args
    assert "--no-use-ess" in args
    assert "--no-make-plots" in args
    assert "--stage-mode-policy" in args and "repeat" in args
    assert "--max-nfev" in args and "5" in args
    assert "--continuation-nfev" in args and "4" in args
    assert "--inner-max-iter" in args and "21" in args
    assert "--trial-ftol" in args and "2e-08" in args
    assert "--target-aspect" in args and "6.0" in args
    assert "--target-abs-iota-min" in args and "0.41" in args
    assert "--max-mirror-ratio" in args and "0.3" in args
    assert "--max-elongation" in args and "8.2" in args
    assert "--qi-ceiling-max" in args and "0.02" in args
    assert "--qi-ceiling-smooth-penalty" in args and "0.002" in args
    assert "--mirror-weight" in args and "20.0" in args
    assert "--elongation-weight" in args and "10.0" in args
    assert "--qi-mboz" in args and "10" in args
    assert "--qi-nboz" in args and "11" in args
    assert "--qi-nphi" in args and "61" in args
    assert "--qi-nalpha" in args and "13" in args
    assert "--qi-n-bounce" in args and "17" in args
    assert "--solver-device" in args and "gpu" in args
    assert "--reference-input" in args
    assert "--accept-boundary-reference-baseline" in args
    assert str(ROOT / "examples" / "data" / "input.nfp2_QI") in args
    assert "--mirror-ramp-stages-json" in args
    stages_path = Path(args[args.index("--mirror-ramp-stages-json") + 1])
    assert stages_path == tmp_path / "out" / "mirror_ramp_stages.json"
    stages = json.loads(stages_path.read_text())
    assert stages == [{"max_nfev": 5, "method": "scipy", "mirror_weight": 20.0, "name": "cleanup"}]
    lambdas = tuple(float(value) for value in args[args.index("--reference-lambdas") + 1].split(","))
    assert lambdas == pytest.approx((0.99, 0.995, 1.0, 1.005, 1.01))
    reference_path = Path(args[args.index("--boundary-reference-json") + 1])
    reference = json.loads(reference_path.read_text())
    assert reference["reference_input"].endswith("input.nfp2_QI")
    assert reference["max_mode"] == 3
    assert reference["target_aspect"] == pytest.approx(6.0)
    assert reference["lambdas"] == pytest.approx([0.99, 0.995, 1.0, 1.005, 1.01])
    assert reference["prefer_aspect_candidates"] is True
    assert reference["prefer_lowest_qi_candidate"] is False
    assert reference["accept_as_baseline"] is True


def test_qi_staged_runner_passes_policy_qi_gates_and_audit_resolution(tmp_path: Path) -> None:
    runner = _load_runner()
    config = runner.QIStagedCaseConfig(
        name="qi_nfp3",
        input_file=ROOT / "examples" / "data" / "input.minimal_seed_nfp3",
        output_dir=tmp_path / "out",
        max_mode=4,
        policy_case="minimal_nfp3_qi",
        reference_input=ROOT / "examples" / "data" / "input.nfp3_QI_fixed_resolution_final",
        max_mirror_ratio=0.30,
        max_elongation=10.0,
        make_plots=False,
    )

    args = runner._build_qi_staged_args(config)

    assert args[args.index("--max-mirror-ratio") + 1] == "0.35"
    assert args[args.index("--max-elongation") + 1] == "8.2"
    assert args[args.index("--qi-gate-smooth-max") + 1] == "0.005"
    assert args[args.index("--qi-gate-legacy-max") + 1] == "0.002"
    assert args[args.index("--qi-mboz") + 1] == "5"
    assert args[args.index("--qi-nphi") + 1] == "31"
    assert args[args.index("--audit-qi-mboz") + 1] == "18"
    assert args[args.index("--audit-qi-nphi") + 1] == "151"
    lambdas = tuple(float(value) for value in args[args.index("--reference-lambdas") + 1].split(","))
    assert lambdas == pytest.approx((0.99, 0.995, 1.0, 1.005, 1.008, 1.01))


def test_qi_staged_runner_can_disable_reference_lambda_override(tmp_path: Path) -> None:
    runner = _load_runner()
    config = runner.QIStagedCaseConfig(
        name="qi_nfp2",
        input_file=ROOT / "examples" / "data" / "input.minimal_seed_nfp2",
        output_dir=tmp_path / "out",
        max_mode=3,
        reference_input=ROOT / "examples" / "data" / "input.nfp2_QI",
        reference_lambdas=None,
        make_plots=False,
    )

    args = runner._build_qi_staged_args(config)

    assert "--reference-lambdas" not in args
    reference_path = Path(args[args.index("--boundary-reference-json") + 1])
    assert "lambdas" not in json.loads(reference_path.read_text())


def test_qi_staged_runner_converts_artifacts_to_case_result(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    out = tmp_path / "out"
    out.mkdir()
    (out / "history.json").write_text(
        """
        {
          "success": true,
          "message": "synthetic optimizer success",
          "objective_final": 1.5,
          "qs_final": 2.5e-4,
          "aspect_final": 9.5,
          "iota_final": 0.52,
          "nfev": 7,
          "njev": 6,
          "total_wall_time_s": 12.0,
          "target_aspect": 10.0,
          "profile": {"jacobian_total": {"count": 2, "wall_time_s": 3.0}}
        }
        """
    )
    (out / "diagnostics.json").write_text(
        """
        {
          "qi_engineering_gate_passed": true,
          "qi_smooth_total": 1.0e-3,
          "qi_legacy_total": 1.5e-3,
          "qi_mirror_ratio_max": 0.28,
          "qi_mirror_ratio_target": 0.30,
          "qi_max_elongation": 6.5,
          "qi_elongation_target": 8.2
        }
        """
    )

    calls = []

    def _fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(runner, "_run_qi_subprocess", _fake_run)
    config = runner.QIStagedCaseConfig(
        name="qi_nfp2",
        input_file=ROOT / "examples" / "data" / "input.minimal_seed_nfp2",
        output_dir=out,
        max_mode=3,
        policy_case="nfp2_qi",
        solver_device="cpu",
        worker_jax_platforms="cpu",
        make_plots=False,
    )

    result = runner.run_qi_staged_case(config)

    assert calls
    assert result.success is True
    assert result.crashed is False
    assert result.problem == "qi"
    assert result.max_mode == 3
    assert result.objective_final == pytest.approx(1.5)
    assert result.qs_final == pytest.approx(2.5e-4)
    assert result.aspect_final == pytest.approx(9.5)
    assert result.iota_final == pytest.approx(0.52)
    assert result.nfev == 7
    assert result.total_wall_time_s == pytest.approx(12.0)
    assert result.profile_jacobian_total_wall_time_s == pytest.approx(3.0)
    assert result.input_nfp == 2
    assert result.qi_raw_total == pytest.approx(1.0e-3)
    assert result.qi_legacy_total == pytest.approx(1.5e-3)
    assert result.qi_mirror_ratio_max == pytest.approx(0.28)
    assert result.qi_max_elongation == pytest.approx(6.5)


def test_qi_staged_runner_preserves_partial_reference_metrics_on_timeout(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    out = tmp_path / "out"
    pre_dir = out / "boundary_reference_preconditioner"
    pre_dir.mkdir(parents=True)
    (pre_dir / "summary.json").write_text(
        """
        [
          {
            "lambda": 0.99,
            "selected": false,
            "score": 5.0,
            "smooth_qi": 4.0e-3,
            "legacy_qi": 3.0e-3,
            "mirror": 0.34,
            "elongation": 7.1,
            "mean_iota": 0.44,
            "aspect": 8.9
          },
          {
            "lambda": 1.01,
            "selected": true,
            "score": 1.0,
            "smooth_qi": 1.1e-3,
            "legacy_qi": 1.8e-3,
            "mirror": 0.29,
            "elongation": 6.4,
            "mean_iota": 0.47,
            "aspect": 9.8
          }
        ]
        """
    )

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout_s"))

    monkeypatch.setattr(runner, "_run_qi_subprocess", _timeout)
    config = runner.QIStagedCaseConfig(
        name="qi_nfp2",
        input_file=ROOT / "examples" / "data" / "input.minimal_seed_nfp2",
        output_dir=out,
        max_mode=3,
        policy_case="nfp2_qi",
        timeout_s=10.0,
        make_plots=False,
    )

    result = runner.run_qi_staged_case(config)

    assert result.success is False
    assert result.crashed is False
    assert "timed out" in result.message
    assert "partial boundary-reference metrics recorded" in result.message
    assert result.qs_final == pytest.approx(1.1e-3)
    assert result.qi_raw_total == pytest.approx(1.1e-3)
    assert result.qi_legacy_total == pytest.approx(1.8e-3)
    assert result.qi_mirror_ratio_max == pytest.approx(0.29)
    assert result.qi_max_elongation == pytest.approx(6.4)
    assert result.iota_final == pytest.approx(0.47)
    assert result.aspect_final == pytest.approx(9.8)


def test_qi_staged_subprocess_timeout_terminates_process_group(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    events = []

    class FakePopen:
        pid = 1234

        def __init__(self, *args, **kwargs):
            self.returncode = None
            events.append(("popen", bool(kwargs.get("start_new_session"))))

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout == 0.25:
                raise subprocess.TimeoutExpired(cmd=("fake",), timeout=timeout)
            self.returncode = -15
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: 4321)
    monkeypatch.setattr(runner.os, "killpg", lambda pgid, sig: events.append(("killpg", pgid, sig)))

    with (tmp_path / "stdout.log").open("w") as stdout, (tmp_path / "stderr.log").open("w") as stderr:
        with pytest.raises(subprocess.TimeoutExpired):
            runner._run_qi_subprocess(
                ["--synthetic"],
                env={},
                stdout=stdout,
                stderr=stderr,
                timeout_s=0.25,
            )

    assert ("popen", True) in events
    assert ("killpg", 4321, runner.signal.SIGTERM) in events
    assert ("killpg", 4321, runner.signal.SIGKILL) not in events


def test_qi_staged_runner_prefers_stage_checkpoint_metrics_on_timeout(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    out = tmp_path / "out"
    pre_dir = out / "boundary_reference_preconditioner"
    pre_dir.mkdir(parents=True)
    (pre_dir / "summary.json").write_text(
        """
        [
          {
            "lambda": 1.0,
            "selected": true,
            "score": 1.0,
            "smooth_qi": 9.0e-3,
            "legacy_qi": 8.0e-3,
            "mirror": 0.41,
            "elongation": 9.0,
            "mean_iota": 0.39,
            "aspect": 7.0
          }
        ]
        """
    )
    stage_dir = out / "mirror_ramp_02_cleanup"
    stage_dir.mkdir(parents=True)
    (stage_dir / "qi_stage_checkpoint.json").write_text(
        """
        {
          "schema_version": 1,
          "partial": true,
          "name": "cleanup",
          "history": {
            "objective_final": 2.25,
            "qs_final": 2.0e-3,
            "aspect_final": 10.1,
            "iota_final": 0.51,
            "nfev": 8,
            "njev": 7,
            "total_wall_time_s": 42.0
          },
          "diagnostics": {
            "qi_smooth_total": 1.9e-3,
            "qi_legacy_total": 1.2e-3,
            "qi_mirror_ratio_max": 0.27,
            "qi_mirror_ratio_target": 0.30,
            "qi_max_elongation": 6.2,
            "qi_elongation_target": 8.2,
            "mean_iota": 0.52,
            "aspect": 10.0
          }
        }
        """
    )

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout_s"))

    monkeypatch.setattr(runner, "_run_qi_subprocess", _timeout)
    config = runner.QIStagedCaseConfig(
        name="qi_nfp2",
        input_file=ROOT / "examples" / "data" / "input.minimal_seed_nfp2",
        output_dir=out,
        max_mode=3,
        policy_case="nfp2_qi",
        timeout_s=10.0,
        make_plots=False,
    )

    result = runner.run_qi_staged_case(config)

    assert result.success is False
    assert result.crashed is False
    assert "partial QI stage checkpoint metrics recorded" in result.message
    assert result.objective_final == pytest.approx(2.25)
    assert result.qs_final == pytest.approx(2.0e-3)
    assert result.qi_legacy_total == pytest.approx(1.2e-3)
    assert result.qi_mirror_ratio_max == pytest.approx(0.27)
    assert result.qi_max_elongation == pytest.approx(6.2)
    assert result.iota_final == pytest.approx(0.51)
    assert result.aspect_final == pytest.approx(10.1)


def test_qi_staged_runner_preserves_zero_checkpoint_metrics(tmp_path: Path) -> None:
    runner = _load_runner()
    out = tmp_path / "out"
    out.mkdir()
    (out / "stage_checkpoint.json").write_text(
        """
        {
          "schema_version": 1,
          "partial": true,
          "history": {
            "objective_final": 0.0,
            "qs_final": 0.0,
            "aspect_final": 0.0,
            "iota_final": 0.0
          },
          "diagnostics": {
            "qi_raw_total": 0.0,
            "qi_smooth_total": 4.0e-3,
            "qi_legacy_total": 0.0
          }
        }
        """
    )

    metrics = runner._stage_checkpoint_partial_metrics(out)

    assert metrics["objective_final"] == pytest.approx(0.0)
    assert metrics["qs_final"] == pytest.approx(0.0)
    assert metrics["aspect_final"] == pytest.approx(0.0)
    assert metrics["iota_final"] == pytest.approx(0.0)
    assert metrics["qi_raw_total"] == pytest.approx(0.0)
    assert metrics["qi_legacy_total"] == pytest.approx(0.0)


def test_qi_staged_runner_prefers_completed_stage_over_newer_pending_checkpoint(tmp_path: Path) -> None:
    runner = _load_runner()
    out = tmp_path / "out"
    stage1 = out / "mirror_ramp_01_matrix_free_mirror030_aspect0p35"
    stage2 = out / "mirror_ramp_02_matrix_free_mirror030_aspect0p75"
    stage1.mkdir(parents=True)
    stage2.mkdir(parents=True)
    completed_checkpoint = {
        "schema_version": 1,
        "partial": True,
        "role": "mirror_ramp",
        "history": {
            "objective_final": 1.989388186622862,
            "qs_final": 1.8673006069830484,
            "aspect_final": 6.399384783437412,
            "iota_final": -0.5006416479486586,
            "nfev": 70,
            "total_wall_time_s": 2822.126,
        },
        "diagnostics": {
            "qi_seed_gate_passed": True,
            "qi_engineering_gate_passed": False,
            "qi_smooth_total": 1.618419772147587e-3,
            "qi_legacy_total": 3.36274252862054e-4,
            "qi_mirror_ratio_max": 0.37172890836617173,
            "qi_max_elongation": 5.053531473933429,
            "aspect": 6.399384783437412,
            "mean_iota": -0.5006416479486586,
        },
    }
    pending_checkpoint = {
        "schema_version": 1,
        "partial": True,
        "role": "mirror_ramp_pending",
        "history": {},
        "diagnostics": {
            "qi_seed_gate_passed": False,
            "qi_engineering_gate_passed": False,
            "qi_smooth_total": 5.276621385462071e-3,
            "qi_legacy_total": 2.908884974702892e-3,
            "qi_mirror_ratio_max": 0.24068553823044567,
            "qi_max_elongation": 4.496529798672295,
            "aspect": 8.001941358966262,
            "mean_iota": -0.44727522386008184,
        },
        "promotion": {"stage_pending": True},
    }
    (stage1 / "qi_stage_checkpoint.json").write_text(json.dumps(completed_checkpoint))
    (stage2 / "qi_stage_checkpoint.json").write_text(json.dumps(pending_checkpoint))
    (out / "stage_checkpoint.json").write_text(json.dumps(pending_checkpoint))

    metrics = runner._stage_checkpoint_partial_metrics(out)

    assert metrics["objective_final"] == pytest.approx(1.989388186622862)
    assert metrics["qs_final"] == pytest.approx(1.8673006069830484)
    assert metrics["aspect_final"] == pytest.approx(6.399384783437412)
    assert metrics["iota_final"] == pytest.approx(-0.5006416479486586)
    assert metrics["qi_legacy_total"] == pytest.approx(3.36274252862054e-4)
    assert metrics["qi_mirror_ratio_max"] == pytest.approx(0.37172890836617173)
    assert metrics["qi_max_elongation"] == pytest.approx(5.053531473933429)


def test_qi_staged_runner_falls_back_from_invalid_root_checkpoint(tmp_path: Path) -> None:
    runner = _load_runner()
    out = tmp_path / "out"
    stage_dir = out / "mirror_ramp_01_cleanup"
    stage_dir.mkdir(parents=True)
    (out / "stage_checkpoint.json").write_text("{not json")
    (stage_dir / "qi_stage_checkpoint.json").write_text(
        """
        {
          "schema_version": 1,
          "partial": true,
          "history": {"objective_final": 1.0},
          "diagnostics": {"qi_smooth_total": 2.0e-3}
        }
        """
    )

    metrics = runner._stage_checkpoint_partial_metrics(out)

    assert metrics["objective_final"] == pytest.approx(1.0)
    assert metrics["qs_final"] == pytest.approx(2.0e-3)


def test_qi_staged_runner_sparse_stage_checkpoint_keeps_boundary_metrics(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    out = tmp_path / "out"
    pre_dir = out / "boundary_reference_preconditioner"
    pre_dir.mkdir(parents=True)
    (pre_dir / "summary.json").write_text(
        """
        [
          {
            "lambda": 1.0,
            "selected": true,
            "score": 1.0,
            "smooth_qi": 1.6e-3,
            "legacy_qi": 1.3e-3,
            "mirror": 0.26,
            "elongation": 6.1,
            "mean_iota": 0.48,
            "aspect": 8.5
          }
        ]
        """
    )
    stage_dir = out / "mirror_ramp_01_cleanup"
    stage_dir.mkdir(parents=True)
    (stage_dir / "qi_stage_checkpoint.json").write_text(
        """
        {
          "schema_version": 1,
          "partial": true,
          "history": {"objective_final": 0.5},
          "diagnostics": {}
        }
        """
    )

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout_s"))

    monkeypatch.setattr(runner, "_run_qi_subprocess", _timeout)
    config = runner.QIStagedCaseConfig(
        name="qi_nfp2",
        input_file=ROOT / "examples" / "data" / "input.minimal_seed_nfp2",
        output_dir=out,
        max_mode=3,
        policy_case="nfp2_qi",
        timeout_s=10.0,
        make_plots=False,
    )

    result = runner.run_qi_staged_case(config)

    assert result.objective_final == pytest.approx(0.5)
    assert result.qs_final == pytest.approx(1.6e-3)
    assert result.qi_legacy_total == pytest.approx(1.3e-3)
    assert result.qi_mirror_ratio_max == pytest.approx(0.26)
    assert result.qi_max_elongation == pytest.approx(6.1)
    assert result.iota_final == pytest.approx(0.48)
    assert result.aspect_final == pytest.approx(8.5)

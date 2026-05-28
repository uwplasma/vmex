from __future__ import annotations

import importlib.util
from dataclasses import fields, replace
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, script_name: str):
    script = ROOT / "examples" / "optimization" / script_name
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_minimal_seed_showcase_case_map_uses_three_coefficient_inputs() -> None:
    generator = _load_module("generate_minimal_seed_showcase", "generate_minimal_seed_showcase.py")

    case_fields = {field.name for field in fields(generator.MinimalSeedCase)}
    assert {"default_max_mode", "default_use_ess", "default_policy"}.isdisjoint(case_fields)
    assert generator.DEFAULT_CASE_ORDER == (
        "qi_circular_nfp1",
        "qi_nfp1",
        "qi_nfp2",
        "qi_nfp3",
        "qi_nfp4",
        "qa_nfp2",
        "qa_nfp3",
        "qh_nfp3",
        "qh_nfp4",
        "qp_nfp1",
        "qp_nfp2",
        "qp_nfp3",
    )
    for case_name in generator.DEFAULT_CASE_ORDER:
        case = generator.SHOWCASE_CASES[case_name]
        if case_name == "qi_circular_nfp1":
            assert case.input_file.name == "input.circular_tokamak"
            continue
        text = Path(case.input_file).read_text()
        assert f"NFP = {case.nfp}" in text
        assert "RBC(0,0)" in text
        assert "RBC(0,1)" in text
        assert "ZBS(0,1)" in text
        assert text.count("RBC(") == 2
        assert text.count("ZBS(") == 1
        assert "RBS(" not in text
        assert "ZBC(" not in text


def test_minimal_seed_showcase_target_helicity_seed_terms_are_deterministic() -> None:
    generator = _load_module("generate_minimal_seed_showcase_target_seed", "generate_minimal_seed_showcase.py")

    expected = (
        ("RBC", (1, 0), 1.0e-3),
        ("ZBS", (1, 0), 1.0e-3),
        ("RBC", (-1, 1), 1.0e-3),
        ("ZBS", (-1, 1), 1.0e-3),
        ("RBC", (1, 1), 1.0e-3),
        ("ZBS", (1, 1), 1.0e-3),
    )

    assert generator.TARGET_HELICITY_SEED_AMPLITUDE == pytest.approx(1.0e-3)
    assert generator._target_helicity_seed_terms(max_mode=0) == ()
    assert generator._target_helicity_seed_terms(max_mode=1) == expected
    assert generator._target_helicity_seed_terms(max_mode=1, amplitude=0.0) == ()


def test_minimal_seed_showcase_writes_per_run_seeded_input_without_mutating_raw_seed(tmp_path: Path) -> None:
    generator = _load_module("generate_minimal_seed_showcase_seed_writer", "generate_minimal_seed_showcase.py")
    case = generator.SHOWCASE_CASES["qh_nfp4"]
    raw_seed_text = case.input_file.read_text()

    seeded_input, inserted = generator._write_target_helicity_seeded_input(
        case.input_file,
        tmp_path,
        max_mode=1,
    )

    assert seeded_input == tmp_path / "input.target_helicity_seed"
    assert case.input_file.read_text() == raw_seed_text
    assert inserted == generator._target_helicity_seed_terms(max_mode=1)

    got = generator.read_indata(seeded_input)
    for family, index, value in inserted:
        assert got.indexed[family][index] == pytest.approx(value)


def test_minimal_seed_showcase_reference_preseed_is_per_run_and_bounded(tmp_path: Path) -> None:
    generator = _load_module("generate_minimal_seed_showcase_reference_preseed", "generate_minimal_seed_showcase.py")
    case = generator.SHOWCASE_CASES["qa_nfp2"]
    raw_seed_text = case.input_file.read_text()

    preseeded_input, changes = generator._write_reference_preseeded_input(
        case.input_file,
        case.reference_preseed_input,
        tmp_path,
        max_mode=1,
        blend=case.reference_preseed_blend,
    )

    assert preseeded_input == tmp_path / "input.reference_preseed"
    assert case.input_file.read_text() == raw_seed_text
    assert changes
    assert all(max(abs(change["n"]), abs(change["m"])) <= 1 for change in changes)
    assert all(not (change["family"] == "RBC" and change["n"] == 0 and change["m"] == 0) for change in changes)

    got = generator.read_indata(preseeded_input)
    assert got.indexed["RBC"][(0, 0)] == pytest.approx(1.0)
    assert got.indexed["RBC"][(1, 0)] == pytest.approx(0.25 * 0.009486394873335013)
    assert got.indexed["ZBS"][(1, 0)] == pytest.approx(0.25 * -0.001583241687798362)
    assert (2, 0) not in got.indexed.get("RBC", {})


def test_minimal_seed_showcase_config_patch_is_bounded_and_non_mutating() -> None:
    generator = _load_module("generate_minimal_seed_showcase_cfg", "generate_minimal_seed_showcase.py")
    budget = generator.MinimalSeedBudget(
        max_nfev=2,
        continuation_nfev=3,
        inner_max_iter=11,
        inner_ftol=1.0e-8,
        trial_max_iter=12,
        trial_ftol=2.0e-8,
    )
    case = generator.SHOWCASE_CASES["qi_nfp3"]
    original = generator.sweep.PROBLEM_CONFIGS["qi"]

    patched = generator._problem_config_for_case(case, max_mode=3, budget=budget)

    assert generator.sweep.PROBLEM_CONFIGS["qi"] is original
    assert patched.input_file.name == "input.minimal_seed_nfp3"
    assert patched.max_nfev == 2
    assert patched.continuation_nfev == 3
    assert patched.inner_max_iter == 11
    assert patched.trial_max_iter == 12
    assert patched.project_input_boundary_to_max_mode is True
    assert patched.min_vmec_mode >= 5
    assert patched.qi_preseed_qp is False
    assert case.qi_policy_case == "minimal_nfp3_qi"
    assert case.qi_reference_input.name == "input.nfp3_QI_fixed_resolution_final"


def test_minimal_seed_showcase_default_mode_matches_publication_matrix(monkeypatch) -> None:
    generator = _load_module("generate_minimal_seed_showcase_defaults", "generate_minimal_seed_showcase.py")

    monkeypatch.setattr(sys, "argv", ["generate_minimal_seed_showcase.py"])
    args = generator._parse_args()

    assert args.max_mode == 5


def test_minimal_seed_showcase_qi_stage_patch_inherits_requested_max_mode() -> None:
    generator = _load_module("generate_minimal_seed_showcase_qi_stage_patch", "generate_minimal_seed_showcase.py")
    budget = generator.MinimalSeedBudget(
        max_nfev=60,
        continuation_nfev=20,
        inner_max_iter=550,
        inner_ftol=1.0e-10,
        trial_max_iter=550,
        trial_ftol=1.0e-10,
    )

    stage = {
        "name": "cleanup",
        "max_nfev": 8,
        "stage_modes": (3,),
        "use_showcase_max_nfev": True,
        "use_showcase_max_mode": True,
    }
    patched = generator._patch_qi_stage_budget(stage, budget=budget, max_mode=5)

    assert patched["max_nfev"] == 60
    assert patched["stage_modes"] == (5,)
    assert patched["use_mode_continuation"] is False
    assert "use_showcase_max_nfev" not in patched
    assert "use_showcase_max_mode" not in patched


def test_renderer_uses_boundary_reference_wout_from_pending_checkpoint(tmp_path: Path) -> None:
    renderer = _load_module("render_minimal_seed_showcase_pending_wout", "render_minimal_seed_showcase.py")
    output_dir = tmp_path / "case"
    reference_dir = output_dir / "boundary_reference_preconditioner" / "lambda_0p950"
    reference_dir.mkdir(parents=True)
    reference_wout = reference_dir / "wout_interpolated.nc"
    reference_wout.write_text("wout\n")
    (output_dir / "stage_checkpoint.json").write_text(
        json.dumps(
            {
                "wout_path": str(output_dir / "boundary_reference_baseline" / "wout_final.nc"),
                "diagnostics_path": str(output_dir / "boundary_reference_baseline" / "diagnostics.json"),
                "diagnostics": {
                    "boundary_reference_wout_path": str(reference_wout),
                },
            }
        )
        + "\n"
    )
    record = renderer.ShowcaseRecord(
        case_name="qi_nfp1",
        nfp=1,
        problem="qi",
        output_dir=output_dir,
        success=False,
        crashed=False,
        message="partial",
        objective_final=None,
        aspect_final=6.8,
        iota_final=0.42,
        total_wall_time_s=None,
        policy="continuation",
        max_mode=5,
        use_ess=True,
    )

    assert renderer._final_wout_for_record(record) == reference_wout


def test_renderer_records_existing_wout_initial_when_original_is_absent(tmp_path: Path) -> None:
    renderer = _load_module("render_minimal_seed_showcase_initial_wout", "render_minimal_seed_showcase.py")
    output_dir = tmp_path / "case"
    output_dir.mkdir()
    initial_wout = output_dir / "wout_initial.nc"
    final_wout = output_dir / "wout_final.nc"
    initial_wout.write_text("initial\n")
    final_wout.write_text("final\n")
    record = renderer.ShowcaseRecord(
        case_name="qi_nfp1",
        nfp=1,
        problem="qi",
        output_dir=output_dir,
        success=True,
        crashed=False,
        message=None,
        objective_final=1.0,
        aspect_final=5.0,
        iota_final=0.4,
        total_wall_time_s=1.0,
        policy="direct",
        max_mode=5,
        use_ess=True,
    )

    provenance = renderer.provenance_for_record(record)

    assert provenance.initial_wout == initial_wout
    assert provenance.final_wout == final_wout


def test_renderer_case_filter_and_skip_missing_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    renderer = _load_module("render_minimal_seed_showcase_case_filter", "render_minimal_seed_showcase.py")
    output_root = tmp_path / "results"
    output_dir = output_root / "gpu" / "qi_nfp1" / "continuation" / "minimal_nfp1_qi" / "mode5" / "ess"
    output_dir.mkdir(parents=True)
    (output_dir / "case_result.json").write_text(
        json.dumps(
            {
                "problem": "qi",
                "success": False,
                "crashed": False,
                "message": "partial QI stage checkpoint metrics recorded",
                "policy": "continuation",
                "max_mode": 5,
                "use_ess": True,
                "qi_legacy_total": 1.0e-2,
                "qi_mirror_ratio_max": 0.25,
            }
        )
    )
    (output_dir / "showcase_case.json").write_text(
        json.dumps(
            {
                "minimal_seed_case": {
                    "name": "qi_nfp1",
                    "nfp": 1,
                    "qi_policy_case": "minimal_nfp1_qi",
                },
                "reference_preseed": {
                    "enabled": True,
                    "reference_input": "input.nfp1_QI",
                    "blend": 0.95,
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_minimal_seed_showcase.py",
            "--output-root",
            str(output_root),
            "--figure-dir",
            str(tmp_path / "figures"),
            "--cases",
            "qi_nfp1",
            "--skip-missing",
            "--summary-only",
        ],
    )

    renderer.main()

    captured = capsys.readouterr()
    assert "Missing current minimal-seed records" not in captured.out
    summary = tmp_path / "figures" / "minimal_seed_showcase_summary.csv"
    assert summary.exists()
    assert "qi_nfp1" in summary.read_text()


def test_renderer_rejects_unknown_case_filter(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    renderer = _load_module("render_minimal_seed_showcase_unknown_case", "render_minimal_seed_showcase.py")
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_minimal_seed_showcase.py", "--cases", "qi_nfp1,not_a_case"],
    )

    with pytest.raises(SystemExit):
        renderer._parse_args()

    assert "unknown --cases value(s): not_a_case" in capsys.readouterr().err


def test_renderer_skip_missing_selected_case_keeps_bounded_smoke_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    renderer = _load_module("render_minimal_seed_showcase_missing_case", "render_minimal_seed_showcase.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_minimal_seed_showcase.py",
            "--output-root",
            str(tmp_path / "empty"),
            "--figure-dir",
            str(tmp_path / "figures"),
            "--cases",
            "qi_nfp1",
            "--skip-missing",
            "--summary-only",
        ],
    )

    renderer.main()

    captured = capsys.readouterr()
    assert "Missing current minimal-seed records" not in captured.out
    assert "qi_nfp1" in captured.out


def test_minimal_seed_showcase_dispatches_qi_to_staged_runner(tmp_path: Path, monkeypatch) -> None:
    generator = _load_module("generate_minimal_seed_showcase_qi_staged", "generate_minimal_seed_showcase.py")
    budget = generator.MinimalSeedBudget(
        max_nfev=4,
        continuation_nfev=3,
        inner_max_iter=11,
        inner_ftol=1.0e-8,
        trial_max_iter=12,
        trial_ftol=2.0e-8,
    )
    captured = {}

    def _fake_staged_runner(config):
        captured["config"] = config
        return generator.sweep.CaseResult(
            backend=config.backend_label,
            problem="qi",
            max_mode=config.max_mode,
            use_ess=config.use_ess,
            success=True,
            crashed=False,
            message="synthetic staged result",
            policy=config.policy,
            output_dir=str(config.output_dir),
        )

    monkeypatch.setattr(generator.qi_staged_runner, "run_qi_staged_case", _fake_staged_runner)
    original_configs = dict(generator.sweep.PROBLEM_CONFIGS)
    result = generator._run_showcase_case(
        generator.SHOWCASE_CASES["qi_nfp2"],
        tmp_path / "case",
        backend_label="cpu",
        solver_device="cpu",
        worker_jax_platforms="cpu",
        policy="continuation",
        max_mode=3,
        use_ess=True,
        budget=budget,
        input_file=tmp_path / "input.target_helicity_seed",
    )

    config = captured["config"]
    assert result.problem == "qi"
    assert config.name == "qi_nfp2"
    assert config.input_file == tmp_path / "input.target_helicity_seed"
    assert config.max_mode == 3
    assert config.policy == "continuation"
    assert config.policy_case == "minimal_nfp2_qi"
    assert config.reference_input.name == "input.nfp2_QI"
    assert config.reference_accept_as_baseline is True
    assert config.reference_lambdas[:4] == pytest.approx((0.0, 0.1, 0.25, 0.5))
    assert config.reference_lambdas[-1] == pytest.approx(1.005)
    assert config.max_nfev == 4
    assert config.continuation_nfev == 3
    assert config.target_aspect == pytest.approx(5.0)
    assert config.inner_max_iter == 11
    assert config.trial_ftol == pytest.approx(2.0e-8)
    assert generator.sweep.PROBLEM_CONFIGS == original_configs


def test_minimal_seed_showcase_writes_target_helicity_seed_input(tmp_path: Path) -> None:
    generator = _load_module("generate_minimal_seed_showcase_seed", "generate_minimal_seed_showcase.py")
    case = generator.SHOWCASE_CASES["qa_nfp2"]

    seeded, terms = generator._write_target_helicity_seeded_input(
        case.input_file,
        tmp_path,
        max_mode=1,
        amplitude=1.0e-5,
    )

    text = seeded.read_text()
    assert seeded.name == "input.target_helicity_seed"
    assert len(terms) == 6
    assert "RBC(1,0)" in text
    assert "ZBS(1,0)" in text
    assert "RBC(-1,1)" in text
    assert "ZBS(-1,1)" in text
    assert "RBC(1,1)" in text
    assert "ZBS(1,1)" in text
    assert text.count("RBC(") == 5
    assert text.count("ZBS(") == 4

    disabled, disabled_terms = generator._write_target_helicity_seeded_input(
        case.input_file,
        tmp_path,
        max_mode=1,
        amplitude=0.0,
    )
    assert disabled == case.input_file
    assert disabled_terms == ()


def test_minimal_seed_showcase_metadata_records_seed_indices_as_n_m(tmp_path: Path) -> None:
    generator = _load_module("generate_minimal_seed_showcase_seed_metadata", "generate_minimal_seed_showcase.py")
    case = generator.SHOWCASE_CASES["qa_nfp2"]
    budget = generator.MinimalSeedBudget(
        max_nfev=1,
        continuation_nfev=1,
        inner_max_iter=2,
        inner_ftol=1.0e-7,
        trial_max_iter=2,
        trial_ftol=1.0e-7,
    )
    terms = (("RBC", (-1, 1), 1.0e-5),)

    generator._write_showcase_metadata(
        tmp_path,
        case=case,
        policy="continuation",
        max_mode=1,
        use_ess=True,
        budget=budget,
        seeded_input_file=tmp_path / "input.target_helicity_seed",
        seed_terms=terms,
    )

    metadata = json.loads((tmp_path / "showcase_case.json").read_text())
    assert metadata["target_helicity_seed"]["terms"] == [
        {"family": "RBC", "n": -1, "m": 1, "value": 1.0e-5}
    ]


def test_minimal_seed_worker_logs_and_records_crashes(tmp_path: Path, monkeypatch) -> None:
    generator = _load_module("generate_minimal_seed_showcase_worker", "generate_minimal_seed_showcase.py")

    started_session = []
    monkeypatch.setattr(generator.sweep, "_start_worker_session", lambda: started_session.append(True))

    def _raise_after_printing(*_args, **_kwargs):
        print("stdout marker")
        print("stderr marker", file=sys.stderr)
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(generator, "_run_showcase_case", _raise_after_printing)
    budget = generator.MinimalSeedBudget(
        max_nfev=1,
        continuation_nfev=1,
        inner_max_iter=2,
        inner_ftol=1.0e-7,
        trial_max_iter=2,
        trial_ftol=1.0e-7,
    )
    output_dir = tmp_path / "case"
    result_path = output_dir / "case_result.json"

    generator._worker(
        "qa_nfp2",
        str(output_dir),
        str(result_path),
        "cpu",
        "cpu",
        "cpu",
        "continuation",
        1,
        True,
        budget.__dict__,
        1.0e-5,
        None,
    )

    assert started_session == [True]
    record = json.loads(result_path.read_text())
    assert record["success"] is False
    assert record["crashed"] is True
    assert record["message"] == "RuntimeError: synthetic worker failure"
    assert record["input_nfp"] == 2
    assert "stdout marker" in (output_dir / "worker_stdout.log").read_text()
    assert "stderr marker" in (output_dir / "worker_stderr.log").read_text()
    assert "synthetic worker failure" in (output_dir / "traceback.txt").read_text()
    assert (output_dir / "showcase_case.json").exists()


def test_minimal_seed_showcase_passes_inner_qi_timeout(tmp_path: Path, monkeypatch) -> None:
    generator = _load_module("generate_minimal_seed_showcase_qi_timeout", "generate_minimal_seed_showcase.py")
    budget = generator.MinimalSeedBudget(
        max_nfev=4,
        continuation_nfev=3,
        inner_max_iter=11,
        inner_ftol=1.0e-8,
        trial_max_iter=12,
        trial_ftol=2.0e-8,
    )
    captured = {}

    def _fake_staged_runner(config):
        captured["timeout_s"] = config.timeout_s
        return generator.sweep.CaseResult(
            backend=config.backend_label,
            problem="qi",
            max_mode=config.max_mode,
            use_ess=config.use_ess,
            success=False,
            crashed=True,
            message="synthetic timeout",
            policy=config.policy,
            output_dir=str(config.output_dir),
        )

    monkeypatch.setattr(generator.qi_staged_runner, "run_qi_staged_case", _fake_staged_runner)

    generator._run_showcase_case(
        generator.SHOWCASE_CASES["qi_nfp2"],
        tmp_path / "case",
        backend_label="cpu",
        solver_device="cpu",
        worker_jax_platforms="cpu",
        policy="continuation",
        max_mode=3,
        use_ess=True,
        budget=budget,
        case_timeout_s=1200.0,
    )

    assert captured["timeout_s"] == pytest.approx(1140.0)


def test_minimal_seed_showcase_marks_qi_timeout_partials_as_non_crash(tmp_path: Path) -> None:
    generator = _load_module("generate_minimal_seed_showcase_qi_partial_timeout", "generate_minimal_seed_showcase.py")
    output_dir = tmp_path / "qi_case"
    stage_dir = output_dir / "cleanup"
    stage_dir.mkdir(parents=True)
    (stage_dir / "qi_stage_checkpoint.json").write_text(
        json.dumps(
            {
                "partial": True,
                "history": {
                    "objective_final": 3.0,
                    "qs_final": 1.7e-3,
                    "aspect_final": 9.2,
                    "iota_final": 0.46,
                },
                "diagnostics": {
                    "qi_legacy_total": 1.4e-3,
                    "qi_mirror_ratio_max": 0.31,
                    "qi_max_elongation": 6.9,
                },
            }
        )
    )
    result = generator._failure_result(
        generator.SHOWCASE_CASES["qi_nfp2"],
        output_dir,
        backend_label="cpu",
        solver_device="cpu",
        worker_jax_platforms="cpu",
        policy="continuation",
        max_mode=3,
        use_ess=True,
        message="worker timed out after 1200.0 s",
    )

    changed = generator._annotate_qi_partial_result(generator.SHOWCASE_CASES["qi_nfp2"], result, output_dir)

    assert changed is True
    assert result.success is False
    assert result.crashed is False
    assert "partial QI stage checkpoint metrics recorded" in result.message
    assert result.qs_final == pytest.approx(1.7e-3)
    assert result.qi_legacy_total == pytest.approx(1.4e-3)
    assert result.qi_mirror_ratio_max == pytest.approx(0.31)


def test_minimal_seed_physics_gate_rejects_zero_iota_and_bad_qi() -> None:
    generator = _load_module("generate_minimal_seed_showcase_physics_gate", "generate_minimal_seed_showcase.py")

    qa_result = generator.sweep.CaseResult(
        backend="cpu",
        problem="qa",
        max_mode=3,
        use_ess=True,
        success=True,
        crashed=False,
        message="optimizer success",
        objective_final=1764.0,
        iota_final=0.0,
    )
    changed = generator._apply_physics_gate(generator.SHOWCASE_CASES["qa_nfp2"], qa_result)

    assert changed is True
    assert qa_result.success is False
    assert "physics gate failed" in qa_result.message
    assert "iota" in qa_result.message

    qi_result = generator.sweep.CaseResult(
        backend="cpu",
        problem="qi",
        max_mode=3,
        use_ess=True,
        success=True,
        crashed=False,
        message="optimizer success",
        objective_final=10.0,
        iota_final=0.5,
        qi_legacy_total=1.0e-3,
        qi_mirror_ratio_max=0.2,
        qi_max_elongation=6.0,
    )
    changed = generator._apply_physics_gate(generator.SHOWCASE_CASES["qi_nfp2"], qi_result)

    assert changed is False
    assert qi_result.success is True

    qi_bad_mirror = generator.sweep.CaseResult(
        backend="cpu",
        problem="qi",
        max_mode=3,
        use_ess=True,
        success=True,
        crashed=False,
        message="optimizer success",
        iota_final=0.5,
        qi_legacy_total=1.0e-3,
        qi_mirror_ratio_max=0.8,
        qi_max_elongation=6.0,
    )

    assert generator._apply_physics_gate(generator.SHOWCASE_CASES["qi_nfp2"], qi_bad_mirror) is True
    assert qi_bad_mirror.success is False
    assert "mirror" in qi_bad_mirror.message

    qh_zero_iota = generator.sweep.CaseResult(
        backend="cpu",
        problem="qh",
        max_mode=3,
        use_ess=True,
        success=True,
        crashed=False,
        message="optimizer success",
        iota_final=0.0,
    )

    assert generator._apply_physics_gate(generator.SHOWCASE_CASES["qh_nfp4"], qh_zero_iota) is True
    assert qh_zero_iota.success is False
    assert "|iota|" in qh_zero_iota.message


def test_minimal_seed_rerun_clears_stale_case_artifacts(tmp_path: Path) -> None:
    generator = _load_module("generate_minimal_seed_showcase_prepare", "generate_minimal_seed_showcase.py")

    output_dir = tmp_path / "case"
    output_dir.mkdir()
    (output_dir / "case_result.json").write_text("{}")
    (output_dir / "history.json").write_text("stale")
    (output_dir / "wout_final.nc").write_text("stale")

    generator._prepare_output_dir_for_run(output_dir, rerun=True)

    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []

    (output_dir / "history.json").write_text("keep")
    generator._prepare_output_dir_for_run(output_dir, rerun=False)

    assert (output_dir / "history.json").read_text() == "keep"


def test_minimal_seed_renderer_loads_records_and_returns_monotone_segments(tmp_path: Path) -> None:
    generator = _load_module("generate_minimal_seed_showcase_for_renderer", "generate_minimal_seed_showcase.py")
    renderer = _load_module("render_minimal_seed_showcase", "render_minimal_seed_showcase.py")

    output_dir = tmp_path / "cpu" / "qa_nfp2" / "continuation" / "mode3" / "ess"
    output_dir.mkdir(parents=True)
    (output_dir / "input.reference_preseed").write_text("reference preseed")
    (output_dir / "input.target_helicity_seed").write_text("target helicity seed")
    (output_dir / "wout_original.nc").write_text("raw initial wout placeholder")
    (output_dir / "wout_final.nc").write_text("final wout placeholder")
    case = generator.SHOWCASE_CASES["qa_nfp2"]
    (output_dir / "showcase_case.json").write_text(
        json.dumps(
            {
                "minimal_seed_case": {
                    "name": case.name,
                    "problem": case.problem,
                    "nfp": case.nfp,
                    "input_file": str(case.input_file),
                    "reference_preseed_input": str(case.reference_preseed_input),
                    "reference_preseed_blend": case.reference_preseed_blend,
                },
                "reference_preseed": {
                    "enabled": True,
                    "blend": case.reference_preseed_blend,
                    "reference_input": str(case.reference_preseed_input),
                    "preseeded_input_file": str(output_dir / "input.reference_preseed"),
                    "changes": [{"family": "RBC", "n": 1, "m": 0, "old": 0.0, "new": 1e-3}],
                },
                "target_helicity_seed": {
                    "enabled": True,
                    "seeded_input_file": str(output_dir / "input.target_helicity_seed"),
                    "terms": [{"family": "RBC", "n": 1, "m": 0, "value": 1e-5}],
                },
                "policy": "continuation",
                "max_mode": 3,
                "use_ess": True,
            }
        )
    )
    (output_dir / "case_result.json").write_text(
        json.dumps(
            {
                "backend": "cpu",
                "problem": "qa",
                "max_mode": 3,
                "use_ess": True,
                "success": True,
                "crashed": False,
                "objective_final": 2.0,
                "aspect_final": 5.0,
                "iota_final": 0.42,
                "total_wall_time_s": 123.0,
                "policy": "continuation",
                "output_dir": str(output_dir),
            }
        )
    )
    (output_dir / "history.json").write_text(
        json.dumps(
            {
                "history": [
                    {"stage": "QA mode 1", "wall_time_s": 0.0, "objective": 10.0},
                    {"stage": "QA mode 1", "wall_time_s": 10.0, "objective": 8.0},
                    {"stage": "QA mode 1", "wall_time_s": 20.0, "objective": 9.0},
                    {"stage": "QA mode 3", "wall_time_s": 30.0, "objective": 7.0},
                    {"stage": "QA mode 3", "wall_time_s": 40.0, "objective": 7.5},
                    {"stage": "QA mode 3", "wall_time_s": 50.0, "objective": 4.0},
                ]
            }
        )
    )
    stage_a = output_dir / "stage_a"
    stage_b = output_dir / "stage_b"
    stage_a.mkdir()
    stage_b.mkdir()
    (stage_a / "history.json").write_text(
        json.dumps(
            {
                "history": [
                    {"stage": "QA staged A", "wall_time_s": 0.0, "objective": 6.0},
                    {"stage": "QA staged A", "wall_time_s": 10.0, "objective": 5.0},
                    {"stage": "QA staged A", "wall_time_s": 20.0, "objective": 7.0},
                ]
            }
        )
    )
    (stage_b / "history.json").write_text(
        json.dumps(
            {
                "history": [
                    {"stage": "QA staged B", "wall_time_s": 0.0, "objective": 4.0},
                    {"stage": "QA staged B", "wall_time_s": 10.0, "objective": 3.0},
                ]
            }
        )
    )

    records = renderer.best_records(renderer.load_records(tmp_path))
    assert len(records) == 1
    assert records[0].case_name == "qa_nfp2"
    promoted_candidate = replace(records[0], max_mode=5, policy="continuation", use_ess=True)
    assert renderer.publication_records([promoted_candidate]) == [promoted_candidate]
    assert renderer.publication_records([replace(promoted_candidate, max_mode=4)]) == []
    provenance = renderer.provenance_for_record(records[0])
    assert provenance.initial_kind == "raw_seed"
    assert provenance.initial_input.name == "input.minimal_seed_nfp2"
    assert provenance.stage_seed_kind == "reference_preseed+target_helicity_seed"
    assert provenance.stage_seed_input.name == "input.target_helicity_seed"
    assert provenance.initial_wout.name == "wout_original.nc"
    assert provenance.final_wout.name == "wout_final.nc"

    stale_dir = tmp_path / "cpu" / "qa_nfp2_old" / "continuation" / "mode3" / "ess"
    stale_dir.mkdir(parents=True)
    (stale_dir / "showcase_case.json").write_text(
        json.dumps(
            {
                "minimal_seed_case": {
                    "name": case.name,
                    "problem": case.problem,
                    "nfp": case.nfp,
                    "input_file": str(case.input_file),
                },
                "policy": "continuation",
                "max_mode": 3,
                "use_ess": True,
            }
        )
    )
    (stale_dir / "case_result.json").write_text((output_dir / "case_result.json").read_text())
    stale_records = renderer.load_records(stale_dir)
    assert stale_records[0].stale_reason is not None
    assert "reference-family preseed" in stale_records[0].stale_reason

    failed_dir = tmp_path / "cpu" / "qh_nfp4" / "continuation" / "mode3" / "ess"
    failed_dir.mkdir(parents=True)
    (failed_dir / "showcase_case.json").write_text(
        json.dumps(
            {
                "minimal_seed_case": {
                    "name": "qh_nfp4",
                    "problem": "qh",
                    "nfp": 4,
                    "input_file": "input.minimal_seed_nfp4",
                    "reference_preseed_input": None,
                    "reference_preseed_blend": 0.0,
                },
                "policy": "continuation",
                "max_mode": 3,
                "use_ess": True,
            }
        )
    )
    (failed_dir / "case_result.json").write_text(
        json.dumps(
            {
                "backend": "cpu",
                "problem": "qh",
                "max_mode": 3,
                "use_ess": True,
                "success": False,
                "crashed": True,
                "policy": "continuation",
                "output_dir": str(failed_dir),
            }
        )
    )
    all_records = renderer.best_records(renderer.load_records(tmp_path), successful_only=False)
    assert [record.case_name for record in all_records] == ["qa_nfp2", "qh_nfp4"]
    assert renderer.publication_records(renderer.load_records(tmp_path)) == []
    failed_record = next(record for record in all_records if record.case_name == "qh_nfp4")
    assert renderer.record_status(failed_record) == "failed"
    partial_record = replace(failed_record, crashed=True, message="partial checkpoint metrics recorded")
    assert renderer.record_status(partial_record) == "partial"

    segments = renderer.objective_segments(records[0])
    assert len(segments) == 2
    np.testing.assert_allclose(segments[0][1], [6.0, 5.0, 5.0])
    np.testing.assert_allclose(segments[1][1], [4.0, 3.0])
    assert segments[1][0][0] == pytest.approx(segments[0][0][-1])

    summary = tmp_path / "summary.csv"
    renderer.write_summary_csv(records, summary)
    summary_text = summary.read_text()
    assert "status" in summary_text
    assert "crashed" in summary_text
    assert "message" in summary_text
    assert "initial_kind" in summary_text
    assert "initial_input" in summary_text
    assert "stage_seed_kind" in summary_text
    assert "final_wout" in summary_text
    assert "reference_preseed+target_helicity_seed" in summary_text
    assert "qa_nfp2" in summary_text

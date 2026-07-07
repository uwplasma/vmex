from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "diagnostics"
    / "qi"
    / "qi_parameter_probe_harness.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("qi_parameter_probe_harness", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_cases_uses_supported_flags_and_keeps_metadata(tmp_path: Path) -> None:
    mod = _load_module()

    cases = mod.generate_cases(
        input_file=Path("input.test"),
        out_root=tmp_path,
        script=Path("examples/optimization/QI_optimization.py"),
        python_executable="python",
        max_modes=(2,),
        stage_mode_policies=("repeat",),
        stage_repeats=(2,),
        boundary_max_ms=(3,),
        boundary_max_ns=(4,),
        min_vmec_modes=(2,),
        vmec_mpol_values=(5,),
        vmec_ntor_values=(7,),
        qi_mboz_values=(6,),
        qi_nboz_values=(7,),
        qi_nphi_values=(31,),
        qi_nalpha_values=(5,),
        qi_n_bounce_values=(9,),
        max_nfev_values=(1,),
        methods=("scipy",),
        ess_alpha_values=(1.1,),
        inner_ftol_values=(1.0e-7,),
        inner_max_iter_values=(11,),
        trial_ftol_values=(1.0e-6,),
        trial_max_iter_values=(12,),
        weight_sets=({"mirror": 20.0, "qi_weight": 100.0},),
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.boundary_max_m == 3
    assert case.boundary_max_n == 4
    assert case.min_vmec_mode == 2
    assert case.vmec_mpol == 5
    assert case.vmec_ntor == 7
    assert case.stage_mode_limits_json is not None
    assert case.stage_mode_limits == (
        {"mode": 2, "max_m": 2, "max_n": 2, "label": "m2_n2"},
        {"mode": 2, "max_m": 2, "max_n": 2, "label": "m2_n2"},
    )
    assert case.unsupported_weight_keys == ("qi_weight",)
    assert "--stage-mode-limits-json" in case.command
    assert "--min-vmec-mode" in case.command
    assert "--vmec-mpol" in case.command
    assert "--vmec-ntor" in case.command
    assert "--mirror-weight" in case.command
    assert "--qi-weight" not in case.command
    assert "--max-mode" in case.command
    assert "--qi-mboz" in case.command
    assert "--audit-qi-mboz" in case.command


def test_cli_dry_run_writes_plan_and_commands(tmp_path: Path) -> None:
    mod = _load_module()

    rc = mod.main(
        [
            "--out-root",
            str(tmp_path),
            "--max-mode",
            "2,3",
            "--stage-mode-policy",
            "repeat",
            "--stage-repeats",
            "1",
            "--max-nfev",
            "1",
            "--method",
            "scipy",
            "--weights",
            "mirror=10,elongation=5",
            "--limit",
            "4",
            "--dry-run",
        ]
    )

    assert rc == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["kind"] == "qi_parameter_probe_harness"
    assert plan["execute"] is False
    assert plan["case_count"] == 2
    commands = (tmp_path / "commands.sh").read_text()
    assert "--max-mode 2" in commands
    assert "--max-mode 3" in commands
    assert "--no-make-plots" in commands


def test_cli_dry_run_writes_anisotropic_stage_mode_json(tmp_path: Path) -> None:
    mod = _load_module()

    rc = mod.main(
        [
            "--out-root",
            str(tmp_path),
            "--max-mode",
            "5",
            "--stage-mode-policy",
            "lower-repeat",
            "--stage-repeats",
            "1",
            "--boundary-max-m",
            "1",
            "--boundary-max-n",
            "5",
            "--min-vmec-mode",
            "1",
            "--vmec-mpol",
            "3",
            "--vmec-ntor",
            "7",
            "--max-nfev",
            "1",
            "--method",
            "scipy",
            "--limit",
            "2",
            "--dry-run",
        ]
    )

    assert rc == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    case = plan["cases"][0]
    assert case["min_vmec_mode"] == 1
    assert case["vmec_mpol"] == 3
    assert case["vmec_ntor"] == 7
    stage_json = Path(case["stage_mode_limits_json"])
    stages = json.loads(stage_json.read_text())
    assert stages == [
        {"label": "m1_n1", "max_m": 1, "max_n": 1, "mode": 1},
        {"label": "m1_n2", "max_m": 1, "max_n": 2, "mode": 2},
        {"label": "m1_n3", "max_m": 1, "max_n": 3, "mode": 3},
        {"label": "m1_n4", "max_m": 1, "max_n": 4, "mode": 4},
        {"label": "m1_n5", "max_m": 1, "max_n": 5, "mode": 5},
    ]
    commands = (tmp_path / "commands.sh").read_text()
    assert f"--stage-mode-limits-json {stage_json}" in commands
    assert "--min-vmec-mode 1" in commands
    assert "--vmec-mpol 3" in commands
    assert "--vmec-ntor 7" in commands


def test_summarize_output_dir_reads_diagnostics_and_history(tmp_path: Path) -> None:
    mod = _load_module()
    case_dir = tmp_path / "runs" / "case_a"
    case_dir.mkdir(parents=True)
    (case_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "qi_engineering_gate_passed": True,
                "qi_smooth_total": 1.0e-4,
                "qi_legacy_total": 2.0e-4,
                "qi_mirror_ratio_max": 0.2,
                "qi_max_elongation": 7.0,
                "mean_iota": 0.5,
                "aspect": 9.5,
                "qi_failure_reasons": [],
            }
        )
        + "\n"
    )
    (case_dir / "history.json").write_text(json.dumps({"objective_final": [3.0, 1.0], "wall_time_s": 2.5}) + "\n")

    records = [mod.summarize_output_dir(path) for path in mod.discover_output_dirs([tmp_path])]

    assert len(records) == 1
    assert records[0]["case_id"] == "case_a"
    assert records[0]["selection"] == "selected"
    assert records[0]["smooth_qi"] == 1.0e-4
    assert records[0]["objective_final"] == 1.0
    assert records[0]["wall_time_s"] == 2.5

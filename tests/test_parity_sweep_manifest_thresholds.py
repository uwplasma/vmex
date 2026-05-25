from __future__ import annotations

from pathlib import Path
import json
import sys

from tools.diagnostics.parity_sweep_manifest import (
    DEFAULT_MANIFEST,
    _build_freeb_scalpot_cmd,
    _build_stage_trace_cmd,
    _evaluate_freeb_thresholds,
    _evaluate_runtime_thresholds,
    _parse_manifest,
)


REPO_ROOT = DEFAULT_MANIFEST.parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_parity_manifest_smoke_tier_covers_required_physics_classes() -> None:
    """Guard the required parity matrix against accidental case removal."""

    _, cases = _parse_manifest(DEFAULT_MANIFEST)
    smoke_cases = [case for case in cases if case.get("enabled", True) and case.get("tier") == "smoke"]

    def has_case(*, lfreeb: bool, lasym: bool, axisymmetric: bool | None = None) -> bool:
        for case in smoke_cases:
            if bool(case.get("lfreeb")) != lfreeb:
                continue
            if bool(case.get("lasym")) != lasym:
                continue
            if axisymmetric is not None and bool(case.get("axisymmetric")) != axisymmetric:
                continue
            return True
        return False

    assert has_case(lfreeb=False, lasym=False, axisymmetric=True)
    assert has_case(lfreeb=False, lasym=True, axisymmetric=True)
    assert has_case(lfreeb=False, lasym=False, axisymmetric=False)
    assert has_case(lfreeb=False, lasym=True, axisymmetric=False)
    assert has_case(lfreeb=True, lasym=False)


def test_ci_parity_smoke_dry_run_covers_full_smoke_tier() -> None:
    """Keep required CI from silently sampling only part of the smoke manifest."""

    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "parity_sweep_manifest.py --tier smoke" in workflow
    assert "--max-cases" not in workflow


def test_parity_manifest_has_optional_bounded_freeb_lasym_true_case() -> None:
    """Keep a self-contained LASYM=true free-boundary case ready for optional parity."""

    _, cases = _parse_manifest(DEFAULT_MANIFEST)
    case = next(
        (
            case
            for case in cases
            if case.get("id") == "freeb_nonaxis_lasym_true_cth_like_local"
        ),
        None,
    )
    assert case is not None
    assert case.get("enabled") is True
    assert case.get("tier") == "planning"
    assert case.get("source") == "vmec_jax/examples"
    assert case.get("compare") == "freeb_scalpot"
    assert bool(case.get("lfreeb")) is True
    assert bool(case.get("lasym")) is True
    assert bool(case.get("axisymmetric")) is False
    assert case.get("input") == "examples/data/input.cth_like_free_bdy_lasym_small"
    assert float(case.get("max_runtime_s", 0.0)) > 0.0
    assert float(case.get("max_total_runtime_s", 0.0)) > 0.0
    assert set(case.get("runtime_thresholds_s_by_iter", {})) == {"80", "100"}
    assert case.get("metric_thresholds_rel_scaled_by_iter", {}).get("80", {}).get("source_sym") == 1.0e-2


def test_parity_manifest_smoke_cases_define_accuracy_and_runtime_contracts() -> None:
    """Keep fast parity cases explicit about tolerances, inputs, and cost gates."""

    _, cases = _parse_manifest(DEFAULT_MANIFEST)
    smoke_cases = [case for case in cases if case.get("enabled", True) and case.get("tier") == "smoke"]
    assert smoke_cases

    for case in smoke_cases:
        assert case["input"], case["id"]
        assert case.get("goal"), case["id"]
        assert case["compare"] in {"stage_trace", "freeb_scalpot"}, case["id"]
        assert float(case.get("atol", 0.0)) >= 0.0, case["id"]
        if case["compare"] == "stage_trace":
            assert float(case.get("rtol", 0.0)) > 0.0, case["id"]
            assert int(case.get("max_iter", 0)) > 0 or case.get("use_input_niter"), case["id"]
        if bool(case.get("lfreeb")):
            assert float(case.get("max_runtime_s", 0.0)) > 0.0, case["id"]
            assert float(case.get("max_total_runtime_s", 0.0)) > 0.0, case["id"]


def test_parity_manifest_enabled_local_inputs_exist() -> None:
    """Catch stale local manifest paths before optional sweeps are launched."""

    _, cases = _parse_manifest(DEFAULT_MANIFEST)
    local_cases = [
        case
        for case in cases
        if case.get("enabled", True) and str(case.get("source")) == "vmec_jax/examples"
    ]
    assert local_cases

    for case in local_cases:
        input_path = Path(case["input"])
        if not input_path.is_absolute():
            input_path = REPO_ROOT / input_path
        assert input_path.exists(), case["id"]


def test_parity_manifest_compare_modes_have_bounded_schema() -> None:
    """Keep manifest cases runnable by the sweep driver without broad VMEC runs."""

    _, cases = _parse_manifest(DEFAULT_MANIFEST)
    enabled_cases = [case for case in cases if case.get("enabled", True)]
    assert enabled_cases

    for case in enabled_cases:
        assert case["tier"] in {"smoke", "full", "planning"}, case["id"]
        assert case["compare"] in {"stage_trace", "freeb_scalpot"}, case["id"]
        assert isinstance(case.get("lfreeb"), bool), case["id"]
        assert isinstance(case.get("lasym"), bool), case["id"]
        assert isinstance(case.get("axisymmetric"), bool), case["id"]
        assert int(case.get("nfp", 0)) > 0, case["id"]
        assert int(case.get("ntor", -1)) >= 0, case["id"]
        assert float(case.get("vmec_timeout", 0.0)) > 0.0, case["id"]

        if case["compare"] == "stage_trace":
            assert "iter_list" not in case, case["id"]
            assert float(case.get("rtol", 0.0)) > 0.0, case["id"]
            assert float(case.get("atol", -1.0)) >= 0.0, case["id"]
            assert int(case.get("max_iter", 0)) > 0 or case.get("use_input_niter"), case["id"]
        else:
            assert bool(case.get("lfreeb")) is True, case["id"]
            iter_list = case.get("iter_list")
            assert isinstance(iter_list, list) and iter_list, case["id"]
            assert all(int(iter_idx) > 0 for iter_idx in iter_list), case["id"]
            assert int(case.get("max_iter", 0)) >= max(int(iter_idx) for iter_idx in iter_list), case["id"]
            assert case.get("metric_thresholds_rel_scaled") or case.get("metric_thresholds_rel_scaled_by_iter"), case[
                "id"
            ]


def test_evaluate_freeb_thresholds_global_pass() -> None:
    case = {"metric_thresholds_rel_scaled": {"source_sym": 1.0e-2, "amatrix": 2.0e-1}}
    runs = [
        {"iter": 53, "metrics_full": {"source_sym": {"rel_scaled": 5.0e-3}, "amatrix": {"rel_scaled": 1.0e-1}}},
        {"iter": 60, "metrics_full": {"amatrix": {"rel_scaled": 1.5e-1}}},
    ]
    ok, report = _evaluate_freeb_thresholds(case, runs)
    assert ok
    assert report["global"]["source_sym"]["pass"]
    assert report["global"]["amatrix"]["pass"]


def test_evaluate_freeb_thresholds_global_fail_on_limit() -> None:
    case = {"metric_thresholds_rel_scaled": {"source_sym": 1.0e-2}}
    runs = [{"iter": 53, "metrics_full": {"source_sym": {"rel_scaled": 2.0e-2}}}]
    ok, report = _evaluate_freeb_thresholds(case, runs)
    assert not ok
    assert not report["global"]["source_sym"]["pass"]


def test_evaluate_freeb_thresholds_by_iter_and_missing_metric_fail() -> None:
    case = {
        "metric_thresholds_rel_scaled_by_iter": {
            "53": {"source_sym": 1.0e-2},
            "54": {"potvac": 5.0e-1},
        }
    }
    runs = [{"iter": 53, "metrics_full": {"source_sym": {"rel_scaled": 5.0e-3}}}]
    ok, report = _evaluate_freeb_thresholds(case, runs)
    assert not ok
    assert report["by_iter"]["53"]["source_sym"]["pass"]
    assert not report["by_iter"]["54"]["potvac"]["pass"]


def test_evaluate_freeb_thresholds_bad_iter_key_fails() -> None:
    case = {"metric_thresholds_rel_scaled_by_iter": {"iter53": {"source_sym": 1.0e-2}}}
    runs = [{"iter": 53, "metrics_full": {"source_sym": {"rel_scaled": 5.0e-3}}}]
    ok, report = _evaluate_freeb_thresholds(case, runs)
    assert not ok
    assert report["by_iter"]["iter53"]["pass"] is False
    assert "error" in report["by_iter"]["iter53"]


def test_evaluate_runtime_thresholds_global_pass() -> None:
    case = {"max_runtime_s": 20.0, "max_total_runtime_s": 50.0}
    runs = [{"iter": 53, "runtime_s": 18.0}, {"iter": 54, "runtime_s": 17.0}]
    ok, report = _evaluate_runtime_thresholds(case, runs)
    assert ok
    assert report["max_runtime_s"]["pass"]
    assert report["max_total_runtime_s"]["pass"]


def test_evaluate_runtime_thresholds_by_iter_fail() -> None:
    case = {"runtime_thresholds_s_by_iter": {"53": {"max_runtime_s": 10.0}}}
    runs = [{"iter": 53, "runtime_s": 18.0}]
    ok, report = _evaluate_runtime_thresholds(case, runs)
    assert not ok
    assert not report["by_iter"]["53"]["max_runtime_s"]["pass"]


def test_evaluate_runtime_thresholds_bad_iter_key_fails() -> None:
    case = {"runtime_thresholds_s_by_iter": {"iter53": {"max_runtime_s": 10.0}}}
    runs = [{"iter": 53, "runtime_s": 8.0}]
    ok, report = _evaluate_runtime_thresholds(case, runs)
    assert not ok
    assert report["by_iter"]["iter53"]["pass"] is False


def test_stage_trace_command_contract_includes_accuracy_and_multigrid_overrides(tmp_path: Path) -> None:
    """Validate optional VMEC2000 parity wiring without launching VMEC2000."""

    case = {
        "input": "examples/data/input.minimal_seed_nfp2",
        "max_iter": 7,
        "rtol": 1.0e-7,
        "atol": 1.0e-10,
        "dump_level": "lite",
        "vmec_timeout": 12.5,
        "use_input_niter": True,
        "single_ns": 13,
        "ns_array": [7, 13],
        "niter_array": [30, 60],
        "ftol_array": [1.0e-8, 1.0e-10],
    }

    cmd = _build_stage_trace_cmd(case, vmec_exec=Path("/opt/vmec/xvmec2000"), workdir=tmp_path / "stage")

    assert Path(cmd[0]).name.startswith("python")
    assert "--input" in cmd and cmd[cmd.index("--input") + 1] == case["input"]
    assert "--vmec2000" in cmd and cmd[cmd.index("--vmec2000") + 1] == "/opt/vmec/xvmec2000"
    assert "--max-iter" in cmd and cmd[cmd.index("--max-iter") + 1] == "7"
    assert "--rtol" in cmd and cmd[cmd.index("--rtol") + 1] == "1e-07"
    assert "--atol" in cmd and cmd[cmd.index("--atol") + 1] == "1e-10"
    assert "--dump-level" in cmd and cmd[cmd.index("--dump-level") + 1] == "lite"
    assert "--vmec-timeout" in cmd and cmd[cmd.index("--vmec-timeout") + 1] == "12.5"
    assert "--workdir" in cmd and Path(cmd[cmd.index("--workdir") + 1]) == tmp_path / "stage"
    assert "--use-input-niter" in cmd
    assert "--single-ns" in cmd and cmd[cmd.index("--single-ns") + 1] == "13"
    assert "--ns-array" in cmd and cmd[cmd.index("--ns-array") + 1] == "7,13"
    assert "--niter-array" in cmd and cmd[cmd.index("--niter-array") + 1] == "30,60"
    assert "--ftol-array" in cmd and cmd[cmd.index("--ftol-array") + 1] == "1e-08,1e-10"


def test_free_boundary_scalpot_command_contract_is_iter_specific(tmp_path: Path) -> None:
    """Guard free-boundary parity commands and JSON output naming."""

    json_path = tmp_path / "summary_iter80.json"
    cmd = _build_freeb_scalpot_cmd(
        {"input": "examples/data/input.cth_like_free_bdy_small"},
        vmec_exec=Path("/opt/vmec/xvmec2000"),
        iter_idx=80,
        max_iter=100,
        workdir=tmp_path / "freeb",
        json_path=json_path,
    )

    assert "--input" in cmd and cmd[cmd.index("--input") + 1] == "examples/data/input.cth_like_free_bdy_small"
    assert "--vmec-exec" in cmd and cmd[cmd.index("--vmec-exec") + 1] == "/opt/vmec/xvmec2000"
    assert "--iter" in cmd and cmd[cmd.index("--iter") + 1] == "80"
    assert "--max-iter" in cmd and cmd[cmd.index("--max-iter") + 1] == "100"
    assert "--workdir" in cmd and Path(cmd[cmd.index("--workdir") + 1]) == tmp_path / "freeb"
    assert "--json" in cmd and Path(cmd[cmd.index("--json") + 1]) == json_path


def test_parity_manifest_dry_run_writes_executable_summary_without_vmec2000(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    """Dry-run should validate stage/free-boundary command wiring without external VMEC."""

    from tools.diagnostics import parity_sweep_manifest as manifest_runner

    input_path = tmp_path / "input.synthetic"
    input_path.write_text("&INDATA\n/\n", encoding="utf-8")
    manifest_path = tmp_path / "parity_manifest.toml"
    manifest_path.write_text(
        f"""
version = 7
name = "synthetic dry-run"

[[cases]]
id = "stage_case"
tier = "smoke"
enabled = true
compare = "stage_trace"
input = "{input_path}"
max_iter = 4
rtol = 1e-4
atol = 1e-12
dump_level = "lite"
vmec_timeout = 11

[[cases]]
id = "freeb_case"
tier = "smoke"
enabled = true
compare = "freeb_scalpot"
input = "{input_path}"
max_iter = 8
iter_list = [3, 8]
vmec_timeout = 22
max_runtime_s = 10.0
max_total_runtime_s = 30.0

[cases.metric_thresholds_rel_scaled_by_iter."3"]
source_sym = 1e-2

[cases.runtime_thresholds_s_by_iter."3"]
max_runtime_s = 5.0

[cases.runtime_thresholds_s_by_iter."8"]
max_runtime_s = 5.0
""".strip(),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_sweep_manifest.py",
            "--manifest",
            str(manifest_path),
            "--vmec-exec",
            str(tmp_path / "missing_xvmec2000"),
            "--tier",
            "smoke",
            "--dry-run",
            "--output-root",
            str(output_root),
        ],
    )

    assert manifest_runner.main() == 0

    stdout = capsys.readouterr().out
    assert "selected_cases=2" in stdout
    assert "DRY-RUN:" in stdout
    summary_path = next(output_root.glob("*/summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in summary["cases"]}

    assert summary["manifest_version"] == 7
    assert summary["failed_cases"] == 0
    assert summary["selected_case_count"] == 2
    assert cases["stage_case"]["status"] == "pass"
    assert cases["stage_case"]["runs"] == [{"returncode": 0, "runtime_s": 0.0, "stdout_path": ""}]
    assert "--max-iter" in cases["stage_case"]["cmd"]
    assert cases["stage_case"]["cmd"][cases["stage_case"]["cmd"].index("--max-iter") + 1] == "4"

    freeb = cases["freeb_case"]
    assert freeb["status"] == "pass"
    assert [run["iter"] for run in freeb["runs"]] == [3, 8]
    assert all(run["returncode"] == 0 for run in freeb["runs"])
    assert "metric_thresholds_rel_scaled" not in freeb
    assert freeb["runtime_thresholds_s"]["pass"] is True
    assert freeb["runtime_thresholds_s"]["observed_total_runtime_s"] == 0.0


def test_parity_manifest_explicit_ids_fail_closed_on_unknown_case(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Requested parity cases should not be silently skipped by a typo."""

    from tools.diagnostics import parity_sweep_manifest as manifest_runner

    input_path = tmp_path / "input.synthetic"
    input_path.write_text("&INDATA\n/\n", encoding="utf-8")
    manifest_path = tmp_path / "parity_manifest.toml"
    manifest_path.write_text(
        f"""
version = 1
name = "synthetic ids"

[[cases]]
id = "known_case"
tier = "smoke"
enabled = true
compare = "stage_trace"
input = "{input_path}"
max_iter = 4
rtol = 1e-4
atol = 1e-12
dump_level = "lite"
vmec_timeout = 11
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity_sweep_manifest.py",
            "--manifest",
            str(manifest_path),
            "--vmec-exec",
            str(tmp_path / "missing_xvmec2000"),
            "--ids",
            "known_case,typo_case",
            "--dry-run",
            "--output-root",
            str(tmp_path / "out"),
        ],
    )

    try:
        manifest_runner.main()
    except SystemExit as exc:
        assert str(exc) == "unknown case id(s): typo_case"
    else:  # pragma: no cover - defensive assertion for clearer failure output
        raise AssertionError("unknown explicit parity case id should fail closed")

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_qs_ess_sweep_worker_jax_platform_policy():
    sweep = _load_sweep_module()

    assert sweep._default_worker_jax_platforms(None) is None
    assert sweep._default_worker_jax_platforms("auto") is None
    assert sweep._default_worker_jax_platforms("cpu") == "cpu"
    assert sweep._default_worker_jax_platforms("default") is None
    assert sweep._default_worker_jax_platforms("gpu") is None


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
    assert cfg.inner_max_iter == sweep.GPU_PRODUCTION_INNER_MAX_ITER
    assert cfg.inner_ftol == sweep.GPU_PRODUCTION_INNER_FTOL
    assert cfg.trial_max_iter == sweep.GPU_PRODUCTION_TRIAL_MAX_ITER
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
    assert cont_cfg.inner_max_iter == sweep.GPU_PRODUCTION_INNER_MAX_ITER
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
    assert diag_cfg.inner_max_iter == 40
    assert diag_cfg.trial_max_iter == 40


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


def test_readme_renderer_uses_preoptimization_initial_wout(tmp_path):
    renderer = _load_readme_renderer_module()

    continuation_dir = tmp_path / "cpu" / "continuation" / "qh" / "mode3" / "ess"
    mode1_dir = tmp_path / "cpu" / "continuation" / "qh" / "mode1" / "ess"
    direct_qi_dir = tmp_path / "cpu" / "direct" / "qi" / "mode2" / "ess"
    direct_qa_dir = tmp_path / "cpu" / "direct" / "qa" / "mode3" / "ess"
    for directory in (continuation_dir, mode1_dir, direct_qi_dir, direct_qa_dir):
        directory.mkdir(parents=True)
        (directory / "wout_initial.nc").write_text("stage-local")
    (mode1_dir / "wout_initial.nc").write_text("pre-mode-1")
    (direct_qi_dir / "wout_original.nc").write_text("original")

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
    )

    assert renderer._preoptimization_wout_path(continuation) == mode1_dir / "wout_initial.nc"
    assert renderer._preoptimization_wout_path(qi) == direct_qi_dir / "wout_original.nc"
    assert renderer._preoptimization_wout_path(qa) == direct_qa_dir / "wout_initial.nc"


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

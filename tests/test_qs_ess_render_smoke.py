from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

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

    assert cfg.max_nfev == 12
    assert cfg.inner_max_iter == sweep.GPU_PRODUCTION_INNER_MAX_ITER
    assert cfg.inner_ftol == sweep.GPU_PRODUCTION_INNER_FTOL
    assert cfg.trial_max_iter == sweep.GPU_PRODUCTION_TRIAL_MAX_ITER
    assert cfg.trial_ftol == sweep.GPU_PRODUCTION_TRIAL_FTOL

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

    preferred = lookup[("cpu", "continuation", "qa", 2, False)]
    assert preferred.message == "explicit"
    assert preferred.objective_final == 0.2

    assert ("cpu", "direct", "qa", 3, True) not in lookup


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

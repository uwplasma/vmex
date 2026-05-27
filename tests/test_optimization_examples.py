from __future__ import annotations

import ast
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_OPTIMIZATION_SCRIPTS = (
    ROOT / "examples" / "optimization" / "QA_optimization.py",
    ROOT / "examples" / "optimization" / "QH_optimization.py",
    ROOT / "examples" / "optimization" / "QP_optimization.py",
    ROOT / "examples" / "optimization" / "QI_optimization.py",
)
FORBIDDEN_SOLVE_PHYSICS_KWARGS = {
    "target_aspect",
    "target_iota",
    "iota_abs_min",
    "qi_options",
    "plot",
    "print_outputs",
}


def _least_squares_solve_keyword_names(script: Path) -> list[set[str]]:
    tree = ast.parse(script.read_text(), filename=str(script))
    keyword_sets: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_solve = (
            isinstance(func, ast.Attribute)
            and func.attr == "least_squares_solve"
            or isinstance(func, ast.Name)
            and func.id == "least_squares_solve"
        )
        if is_solve:
            keyword_sets.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return keyword_sets


def test_fixed_boundary_qs_examples_are_standalone_workflows() -> None:
    scripts = [
        ROOT / "examples" / "optimization" / "QH_optimization.py",
        ROOT / "examples" / "optimization" / "QA_optimization.py",
        ROOT / "examples" / "optimization" / "QP_optimization.py",
    ]
    for script in scripts:
        text = script.read_text()
        assert 'if __name__ == "__main__"' not in text
        assert "FixedBoundaryQSConfig" not in text
        assert "run_quasi_isodynamic_objective_optimization(" not in text
        assert "run_fixed_boundary_objective_optimization(" not in text
        assert "build_qs_stage(" not in text
        assert "run_qs_stage(" not in text
        assert "run_qs_optimization(" not in text
        assert "FixedBoundaryVMEC.from_input(" in text
        assert "objective_tuples = [" in text
        assert "LeastSquaresProblem.from_tuples(" in text
        assert "problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)" in text
        assert "Assembled least-squares problem" in text
        assert "problem.objective_names" in text
        assert "problem.scalar_objective_names" in text
        assert "least_squares_solve(" in text
        assert "problem =" in text
        assert "The solve call only receives optimizer, continuation, device, and output" in text
        assert "SAVE_STAGE_INPUTS = True" in text
        assert "SAVE_STAGE_WOUTS = False" in text
        assert "USE_SIMPLE_SEED = True" in text
        assert "SIMPLE_SEED_PERTURBATION =" in text
        assert "prepare_simple_omnigenity_seed_input(" in text
        assert "perturbation=SIMPLE_SEED_PERTURBATION" in text
        assert "input.minimal_seed_nfp" in text
        assert "save_stage_inputs=SAVE_STAGE_INPUTS" in text
        assert "save_stage_wouts=SAVE_STAGE_WOUTS" in text
        assert "target_aspect=" not in text
        assert "target_iota=" not in text
        assert "iota_abs_min=" not in text
        assert "qi_options=" not in text
        assert "plot=" not in text
        assert "print_optimization_outputs" not in text
        assert "history = result.history" in text
        assert "objective_history = result.objective_history" in text
        assert "timing = result.timing_summary" in text
        assert "result_summary = result.summary" in text
        assert "saved_paths = vj.save_optimization_result(result, output_dir=OUTPUT_DIR)" in text
        assert "Files saved from result objects" in text
        assert "vj.load_wout(saved_paths.final_wout)" in text
        assert "vmecplot2_bmag_grid(" in text
        assert "plot_3d_boundary_comparison(" in text
        assert "plot_boozer_lcfs_bmag_comparison(" in text
        assert "plot_objective_history(" in text
        assert "Plotting is a normal post-processing block" in text
        assert "saved_paths.initial_wout" in text
        assert "saved_paths.history" in text


def test_qp_example_documents_zero_transform_escape_policy() -> None:
    text = (ROOT / "examples" / "optimization" / "QP_optimization.py").read_text()
    readme = (ROOT / "examples" / "optimization" / "README.md").read_text()
    docs = (ROOT / "docs" / "optimization.rst").read_text()

    assert "SIMPLE_SEED_PERTURBATION = 1.0e-5" in text
    assert "vj.qs_stage_modes(" in text
    assert "lower-mode continuation sequence" in text
    assert "zero-iota branch" in docs
    assert "zero-transform basin" in readme


def test_qs_examples_expose_reviewed_mode5_budgets() -> None:
    qa = (ROOT / "examples" / "optimization" / "QA_optimization.py").read_text()
    qh = (ROOT / "examples" / "optimization" / "QH_optimization.py").read_text()
    qp = (ROOT / "examples" / "optimization" / "QP_optimization.py").read_text()

    for text in (qa, qh, qp):
        assert "MAX_MODE = 5" in text
        assert "MAX_NFEV = 60" in text
        assert "ALPHA = 1.2" in text
        assert "TARGET_ASPECT = 5.0" in text

    assert "INNER_MAX_ITER = 120" in qa
    assert "TRIAL_MAX_ITER = 120" in qa
    assert "INNER_FTOL = 1.0e-9" in qa
    assert "TRIAL_FTOL = 1.0e-9" in qa

    assert "INNER_MAX_ITER = 180" in qh
    assert "TRIAL_MAX_ITER = 180" in qh
    assert "INNER_FTOL = 1.0e-9" in qh
    assert "TRIAL_FTOL = 1.0e-9" in qh

    assert "INNER_MAX_ITER = 180" in qp
    assert "TRIAL_MAX_ITER = 60" in qp
    assert "INNER_FTOL = 1.0e-9" in qp
    assert "TRIAL_FTOL = 1.0e-8" in qp


def test_qp_budget_probe_uses_vmec_space_engineering_terms() -> None:
    from tools.diagnostics import qs_budget_probe

    problem = qs_budget_probe._objective_problem(
        "qp",
        qs_budget_probe.PROBLEM_DEFAULTS["qp"],
    )

    assert problem.scalar_objective_names == (
        "aspect",
        "abs_iota_floor",
        "qs",
        "mirror_ratio",
        "max_elongation",
    )
    assert problem.qi_objective_names == ()
    assert problem.is_qi is False


def test_optimization_readme_and_docs_teach_visible_workflow_anatomy() -> None:
    readme = (ROOT / "examples" / "optimization" / "README.md").read_text()
    docs = (ROOT / "docs" / "optimization.rst").read_text()
    readme_flat = " ".join(readme.split())
    docs_flat = " ".join(docs.split())

    assert "Editable Workflow Anatomy" in readme
    assert "Editable workflow anatomy" in docs
    assert "`least_squares_solve` receives optimizer, continuation, device, and output controls only." in readme_flat
    assert "Do not pass physics shortcuts such as ``target_aspect`` or ``qi_options``" in docs_flat
    assert "Recommended workflow API" in docs
    assert "FixedBoundaryOptimizationResult" in docs
    assert "save_optimization_result" in readme
    assert "save_input" in docs
    assert "save_wout" in docs
    assert "save_history" in docs
    assert "save_final_outputs=False" in readme
    assert "plot_paths = {" in readme
    assert "vj.plot_3d_boundary_comparison(" in readme
    assert "vj.plot_boozer_lcfs_bmag_comparison(" in readme
    assert "vj.plot_objective_history(" in readme
    for obsolete_qi_driver_detail in (
        "VMEC_JAX_QI",
        "RUN_CASE",
        "qis.make_qi_optimization_context",
        "qi_optimization_support",
    ):
        assert obsolete_qi_driver_detail not in readme
        assert obsolete_qi_driver_detail not in docs


def test_primary_examples_use_direct_plotting_apis_not_generic_helpers() -> None:
    texts = [script.read_text() for script in PRIMARY_OPTIMIZATION_SCRIPTS]
    combined = "\n".join(texts + [(ROOT / "examples" / "optimization" / "README.md").read_text()])

    assert "plot_qh_optimization" not in combined
    assert "plot_qs_optimization" not in combined
    assert "plot_optimization_summary" not in combined

    for text in texts:
        assert "vj.plot_3d_boundary_comparison(" in text
        assert "vj.plot_boozer_lcfs_bmag_comparison(" in text
        assert "vj.plot_objective_history(" in text


def test_primary_example_solve_calls_do_not_take_physics_shortcuts() -> None:
    for script in PRIMARY_OPTIMIZATION_SCRIPTS:
        solve_keyword_sets = _least_squares_solve_keyword_names(script)
        assert solve_keyword_sets, f"{script.name} should call least_squares_solve"
        for keywords in solve_keyword_sets:
            assert not (keywords & FORBIDDEN_SOLVE_PHYSICS_KWARGS), script.name
            assert {"max_nfev", "method", "solver_device"} <= keywords
            assert "save_stage_inputs" in keywords
            assert "save_stage_wouts" in keywords
            assert "save_final_outputs" in keywords


def test_qi_example_uses_qi_problem_api() -> None:
    text = (ROOT / "examples" / "optimization" / "QI_optimization.py").read_text()
    cases_text = (ROOT / "examples" / "optimization" / "qi_optimization_cases.py").read_text()
    support_text = (ROOT / "vmec_jax" / "qi_optimization.py").read_text()
    compat_text = (ROOT / "examples" / "optimization" / "qi_optimization_support.py").read_text()

    assert "run_quasi_isodynamic_objective_optimization(" not in text
    assert "QI_CASES = {" in cases_text
    assert "from qi_optimization_cases import" not in text
    assert "import qi_optimization_support" not in text
    assert "from vmec_jax.qi_optimization import *" in compat_text
    assert "from tools.diagnostics" not in support_text.split("def _load_basin_prefilter_tools")[0]
    assert len(text.splitlines()) < 450
    assert "qis.configure(globals())" not in text
    assert "os.environ" not in text
    assert "RUN_CASE" not in text
    assert "CASE =" not in text
    assert "QI_CONTEXT = vj.make_qi_optimization_context(" in text
    assert "ctx=QI_CONTEXT" in text
    assert 'INPUT_FILE = DATA_DIR / "input.nfp2_QI"' in text
    assert 'INPUT_FILE = DATA_DIR / "input.QI_stel_seed_3127"' in text
    assert "USE_SIMPLE_SEED = False" in text
    assert "USE_TARGET_HELICITY_SEED = True" in text
    assert "USE_REFERENCE_FAMILY_SEED = False" in text
    assert "prepare_simple_omnigenity_seed_input(" in text
    assert "run_target_helicity_seed_preconditioner(" in text
    assert "run_boundary_reference_preconditioner(" in text
    assert "RAW_INPUT_FILE = INPUT_FILE" in text
    assert "QuasiIsodynamicOptions(" in text
    assert "QuasiIsodynamicResidual(QI_OPTIONS)" in text
    assert "QuasiIsodynamicResidualCeiling(" in text
    assert "VMECMirrorRatio(" in text
    assert "MaxElongation(" in text
    assert "objective_tuples = [" in text
    assert "LeastSquaresProblem.from_tuples(" in text
    assert "Assembled least-squares problem" in text
    assert "problem.objective_names" in text
    assert "problem.scalar_objective_names" in text
    assert "problem.qi_objective_names" in text
    assert "def make_vmec_for_stage(" not in text
    assert "least_squares_solve(" in text
    assert "SAVE_STAGE_INPUTS = True" in text
    assert "SAVE_STAGE_WOUTS = False" in text
    assert "save_stage_inputs=SAVE_STAGE_INPUTS" in text
    assert "save_stage_wouts=SAVE_STAGE_WOUTS" in text
    solve_call = text.split("return vj.least_squares_solve(", 1)[1].split(")\n", 1)[0]
    assert "target_aspect=" not in solve_call
    assert "iota_abs_min=" not in solve_call
    assert "qi_options=" not in solve_call
    assert "plot=" not in solve_call
    assert "print_optimization_outputs" not in text
    assert "def save_raw_seed_initial_artifacts(input_file, input_out, wout_out, *, ctx:" in support_text
    assert "vj.write_indata(input_out, vj.read_indata(input_file))" in support_text
    assert "vj.run_fixed_boundary(input_file, solver_device=_ctx(ctx, \"solver_device\"), verbose=False)" in support_text
    assert "vj.write_wout_from_fixed_boundary_run(wout_out, run)" in support_text
    assert "raw_initial_run = vj.save_raw_seed_initial_artifacts(" in text
    assert "INPUT_FILE," in text
    assert 'saved_paths["initial_input"],' in text
    assert 'saved_paths["initial_wout"],' in text
    assert "first_result_for_outputs" not in text
    assert "initial_result_for_outputs" not in text
    assert "result.final_optimizer" in text
    assert "result.final_result" in text
    assert "history = result.history" in text
    assert "objective_history = result.objective_history" in text
    assert "timing = result.timing_summary" in text
    assert "result_summary = result.summary" in text
    assert "Running the raw input deck once for initial comparison plots" in text
    assert "result.final_params" in text
    assert "result.final_state" in text
    assert "saved_paths = {" in text
    assert "result.final_optimizer.save_input(" in text
    assert "result.final_optimizer.save_wout(" in text
    assert "result.final_optimizer.save_history(" in text
    assert "Files saved for raw-seed/final comparison" in text
    assert 'vj.load_wout(saved_paths["final_wout"])' in text
    assert "vmecplot2_bmag_grid(" in text
    assert "plot_3d_boundary_comparison(" in text
    assert "plot_boozer_lcfs_bmag_comparison(" in text
    assert "plot_objective_history(" in text
    assert "Generating initial-vs-final LCFS |B| contour comparison in Boozer coordinates" in text
    assert "qi_diagnostics_from_state(" in text
    assert "qi_cleanup_candidate_promotable(" in support_text
    assert "require_engineering_gate=bool(stage.get(\"require_engineering_gate\", False))" in support_text
    assert "reference_diagnostics = None if accepted_result is None else qi_diagnostics_for_result(" in support_text
    assert "diagnostics.json" in text
    assert "json.dumps(vj.jsonable(diagnostics)" in text
    assert "qi_mirror_ratio_max" in text
    assert 'saved_paths["initial_wout"]' in text
    assert 'saved_paths["history"]' in text


def test_qi_case_resolver_respects_editable_default_and_env(monkeypatch) -> None:
    module_path = ROOT / "examples" / "optimization" / "qi_optimization_cases.py"
    spec = importlib.util.spec_from_file_location("qi_optimization_cases_for_test", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.delenv("VMEC_JAX_QI_RUN_CASE", raising=False)
    monkeypatch.delenv("VMEC_JAX_QI_INPUT", raising=False)
    monkeypatch.delenv("VMEC_JAX_QI_OUTPUT_DIR", raising=False)

    run_case, case = module.resolve_qi_case("nfp1_qi")
    assert run_case == "nfp1_qi"
    assert case["input_file"].name == "input.nfp1_QI"

    monkeypatch.setenv("VMEC_JAX_QI_RUN_CASE", "minimal_nfp2_qi")
    run_case, case = module.resolve_qi_case("nfp3_qi")
    assert run_case == "minimal_nfp2_qi"
    assert case["input_file"].name == "input.minimal_seed_nfp2"


def test_qi_example_keeps_mirror_cleanup_guarded_by_qi_ceiling() -> None:
    text = (ROOT / "examples" / "optimization" / "QI_optimization.py").read_text()
    cases_text = (ROOT / "examples" / "optimization" / "qi_optimization_cases.py").read_text()
    support_text = (ROOT / "vmec_jax" / "qi_optimization.py").read_text()

    assert '"mirror_weight": 20.0' in cases_text
    assert '"qi_ceiling_weight": 0.0' in cases_text
    assert '"mirror_ramp_stages": (' in cases_text
    assert '"name": "matrix_free_mirror030"' in cases_text
    assert '"require_mirror_improvement": False' in cases_text
    assert "stage_smooth_qi_max = float(stage.get(\"smooth_qi_max\", _ctx(ctx, \"qi_gate_smooth_max\")))" in support_text
    assert "stage_promotion_mirror_threshold = float(" in support_text
    assert "repeats=int(stage.get(\"stage_repeats\", _ctx(ctx, \"stage_repeats\")))" in support_text
    assert "method=str(stage.get(\"method\", _ctx(ctx, \"method\")))" in support_text
    assert '"use_showcase_max_nfev": True' in cases_text
    assert '"require_engineering_gate": True' in cases_text
    assert "qi_ceiling = vj.QuasiIsodynamicResidualCeiling(" in text
    assert "qi_options=QI_OPTIONS" in text
    assert "mirror = vj.VMECMirrorRatio(" in text
    assert "surface_index=MIRROR_SURFACE_INDEX" in text
    assert "stage_promotes_candidate(" in support_text
    assert "require_engineering_gate=bool(stage.get(\"require_engineering_gate\", False))" in support_text
    assert "_stage_value(stage, \"qi_ceiling_weight\", QI_CEILING_WEIGHT)" not in text
    assert "reference=reference_diagnostics" in support_text
    assert "objective_tuples.append((qi_ceiling.J, 0.0, QI_CEILING_WEIGHT))" in text
    assert "(mirror.J, 0.0, MIRROR_WEIGHT)" in text
    assert text.index("qi_ceiling = vj.QuasiIsodynamicResidualCeiling(") < text.index(
        "mirror = vj.VMECMirrorRatio("
    )
    assert text.index("QI_CEILING_WEIGHT > 0.0") > text.index("objective_tuples = [")


def test_qi_nfp4_case_is_explicit_nonpassing_stress_fixture() -> None:
    cases_text = (ROOT / "examples" / "optimization" / "qi_optimization_cases.py").read_text()
    docs = "\n".join(
        [
            (ROOT / "docs" / "optimization.rst").read_text(),
            (ROOT / "docs" / "optimization_sweep_results.rst").read_text(),
        ]
    )

    assert '"nfp4_qh_warm_to_qi"' in cases_text
    assert '"case_goal": "NFP=4 QH-to-QI non-passing stress fixture; audit only"' in cases_text
    assert '"expected_gate_status": "non_passing_stress_fixture"' in cases_text
    assert '"expected_gate_failures": ("smooth_qi", "legacy_qi", "mirror")' in cases_text
    assert '"known_best_nfp4_quick_audit": {' in cases_text
    assert "external_nfp4_qi_wfq0" in cases_text
    assert "NFP=4 QI" in docs
    assert "nfp3_qi" in docs
    assert "nfp4_qi_finite_beta" in docs
    assert (
        "finite-beta NFP=4 verification/stress lane" in docs
        or "finite-beta stress/verification lane" in docs
        or "finite-beta NFP=4 stress fixture" in docs
        or ("finite-beta NFP=4 stress" in docs and "fixture" in docs)
    )
    assert "non-passing stress fixture" in docs or "non-passing\nstress fixture" in docs
    assert "nfp4_qh_warm_to_qi" in docs


def test_qi_seed_robustness_optional_mirror_cleanup_contract_keeps_qi_guard() -> None:
    text = (ROOT / "examples" / "optimization" / "QI_seed_robustness.py").read_text()

    assert "VMEC_JAX_QI_SEED_INPUT" in text
    assert "VMEC_JAX_QI_SEED_OUTPUT_DIR" in text
    assert "VMEC_JAX_QI_SEED_MAX_NFEV" in text
    assert "QI seed robustness policy:" in text
    assert "# Optional engineering cleanup.  Mirror/elongation can be included after the" in text
    assert "# qi_ceiling = vj.QuasiIsodynamicResidualCeiling(" in text
    assert "#     (qi_ceiling.J, 0.0, 100.0)," in text
    assert "#     (mirror.J, 0.0, MIRROR_WEIGHT)," in text
    assert text.index("# qi_ceiling = vj.QuasiIsodynamicResidualCeiling(") < text.index("#     (mirror.J, 0.0, MIRROR_WEIGHT),")


def test_qi_objective_comparison_is_top_level_diagnostic() -> None:
    text = (ROOT / "examples" / "optimization" / "compare_omnigenity_qi_objective.py").read_text()

    assert "argparse" not in text
    assert "QI_VARIANTS" in text
    assert "PHIMIN_FACTORS" in text
    assert "QuasiIsodynamicResidual" in text
    assert "legacy_qi_branch_shuffle_diagnostic_from_boozer_output" in text
    assert "quasi_isodynamic_residual_from_state(" in text


def test_qs_sweep_reports_true_legacy_qi_metric() -> None:
    text = (ROOT / "examples" / "optimization" / "generate_qs_ess_sweep.py").read_text()

    assert "legacy_qi_branch_shuffle_diagnostic_from_boozer_output" in text
    assert '"qi_legacy_total": qi_total' not in text


def test_qs_sweep_qi_mirror_defaults_to_all_surfaces() -> None:
    from examples.optimization import generate_qs_ess_sweep as sweep

    text = (ROOT / "examples" / "optimization" / "generate_qs_ess_sweep.py").read_text()

    assert sweep.PROBLEM_CONFIGS["qi"].qi_mirror_surface_index is None
    assert sweep.PROBLEM_CONFIGS["qp"].qi_mirror_surface_index is None
    assert sweep.PROBLEM_CONFIGS["qi"].qi_ceiling_weight > 0.0
    assert sweep.PROBLEM_CONFIGS["qi"].qi_ceiling_max == pytest.approx(2.0e-3)
    assert "qi_ceiling_weight" in sweep.ProblemConfig.__dataclass_fields__
    assert "qi_mirror_objective = vj.VMECMirrorRatio(" in text
    assert "mirror = qi_mirror_objective._evaluate_state(qi_mirror_ctx, state)" in text

    booz = {
        "bmnc_b": np.arange(6.0).reshape(2, 3),
        "bmns_b": np.arange(6.0, 12.0).reshape(2, 3),
        "iota_b": np.asarray([0.4, 0.5]),
        "s_b": np.asarray([0.25, 0.75]),
    }
    assert sweep._mirror_boozer_surfaces(booz, None) is booz
    sliced = sweep._mirror_boozer_surfaces(booz, 1)
    np.testing.assert_allclose(sliced["bmnc_b"], booz["bmnc_b"][1:2])
    np.testing.assert_allclose(sliced["iota_b"], [0.5])


def test_qs_sweep_qi_mirror_residuals_use_vmec_mirror_ratio(monkeypatch) -> None:
    from examples.optimization import generate_qs_ess_sweep as sweep

    fake_booz = ModuleType("booz_xform_jax")
    fake_booz.prepare_booz_xform_constants = lambda **_kwargs: ("constants", "grids")
    monkeypatch.setitem(sys.modules, "booz_xform_jax", fake_booz)

    stage_indata = SimpleNamespace(get_bool=lambda *_args, **_kwargs: False)
    stage_boundary = SimpleNamespace()
    stage_boundary_input = SimpleNamespace()
    stage_static = SimpleNamespace(
        cfg=SimpleNamespace(mpol=2, ntor=1, ntheta=4, nzeta=4, nfp=2, lasym=False),
        modes="modes",
        s=np.asarray([0.0, 0.5, 1.0]),
    )
    mirror_calls = []

    class FakeOptimizer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.residuals_from_state = args[4]

    class FakeVMECMirrorRatio:
        def __init__(self, **kwargs):
            mirror_calls.append(("init", kwargs))

        def _evaluate_state(self, ctx, state):
            mirror_calls.append(("eval", ctx, state))
            assert ctx.static is stage_static
            assert ctx.indata is stage_indata
            assert ctx.signgs == -1
            assert state == "state"
            return {
                "residuals1d": sweep.jnp.asarray([0.2, 0.3], dtype=sweep.jnp.float64),
                "total": sweep.jnp.asarray(0.13, dtype=sweep.jnp.float64),
            }

    def fail_boozer_mirror(*_args, **_kwargs):
        raise AssertionError("QI sweep mirror residuals must not use the Boozer mirror penalty")

    def fail_unexpected_quality_term(**_kwargs):
        raise AssertionError("disabled QI engineering term should not be evaluated")

    monkeypatch.setattr(sweep.vj, "build_static", lambda _cfg: stage_static)
    monkeypatch.setattr(sweep.vj, "boundary_from_indata", lambda *_args, **_kwargs: stage_boundary)
    monkeypatch.setattr(
        sweep.vj,
        "extend_boundary_for_max_mode",
        lambda _indata, _static, _boundary, _max_mode: (stage_indata, stage_static, stage_boundary),
    )
    monkeypatch.setattr(sweep.vj, "boundary_input_from_indata", lambda *_args, **_kwargs: stage_boundary_input)
    monkeypatch.setattr(sweep.vj, "boundary_param_specs", lambda *_args, **_kwargs: ["spec"])
    monkeypatch.setattr(sweep.vj, "flux_profiles_from_indata", lambda *_args, **_kwargs: "flux")
    monkeypatch.setattr(sweep.vj, "FixedBoundaryExactOptimizer", FakeOptimizer)
    monkeypatch.setattr(sweep.vj, "VMECMirrorRatio", FakeVMECMirrorRatio)
    monkeypatch.setattr(sweep, "initial_guess_from_boundary", lambda *_args, **_kwargs: "guess")
    monkeypatch.setattr(sweep, "eval_geom", lambda *_args, **_kwargs: SimpleNamespace(sqrtg=np.ones((2, 2))))
    monkeypatch.setattr(sweep, "signgs_from_sqrtg", lambda *_args, **_kwargs: -1)
    monkeypatch.setattr(sweep, "equilibrium_aspect_ratio_from_state", lambda **_kwargs: 5.0)
    monkeypatch.setattr(
        sweep,
        "quasi_isodynamic_residual_from_state",
        lambda **_kwargs: {
            "residuals1d": sweep.jnp.asarray([0.11], dtype=sweep.jnp.float64),
            "total": sweep.jnp.asarray(0.0121, dtype=sweep.jnp.float64),
        },
    )
    monkeypatch.setattr(sweep, "mirror_ratio_penalty_from_boozer_output", fail_boozer_mirror)
    monkeypatch.setattr(sweep, "max_elongation_penalty_from_state", fail_unexpected_quality_term)
    monkeypatch.setattr(sweep, "lgradb_penalty_from_state", fail_unexpected_quality_term)

    problem_cfg = replace(
        sweep.PROBLEM_CONFIGS["qi"],
        target_aspect=5.0,
        target_iota=None,
        iota_abs_min=None,
        surfaces=np.asarray([0.25, 0.75]),
        qs_weight=2.0,
        qi_mirror_weight=5.0,
        qi_mirror_ntheta=13,
        qi_mirror_nphi=17,
        qi_ceiling_weight=0.0,
        qi_elongation_weight=0.0,
        qi_lgradb_weight=0.0,
        project_input_boundary_to_max_mode=False,
    )

    _stage_specs, stage_opt, _iota_fn, _stage_boundary_input = sweep._build_stage(
        problem_cfg,
        cfg=object(),
        indata0=stage_indata,
        max_mode=2,
        solver_device="cpu",
    )

    residuals = np.asarray(stage_opt.residuals_from_state("state"))

    np.testing.assert_allclose(residuals, [0.0, 0.22, 1.0, 1.5], rtol=1.0e-12, atol=1.0e-12)
    assert stage_opt.residuals_from_state._qs_total_from_state("state") == pytest.approx(3.2984)
    assert mirror_calls[0][0] == "init"
    assert mirror_calls[0][1]["threshold"] == pytest.approx(problem_cfg.qi_max_mirror_ratio)
    assert mirror_calls[0][1]["surfaces"] is problem_cfg.surfaces
    assert mirror_calls[0][1]["surface_index"] is None
    assert mirror_calls[0][1]["ntheta"] == 13
    assert mirror_calls[0][1]["nphi"] == 17
    assert [call[0] for call in mirror_calls].count("eval") == 2


def test_policy_matrix_plots_single_problem(tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")

    from examples.optimization import compare_qs_policy_matrix as matrix

    monkeypatch.setattr(matrix, "PROBLEMS", ("qa",))

    outpath = tmp_path / "one_problem_matrix.png"
    matrix._plot_policy_matrix_all([], outpath=outpath)

    assert outpath.exists()


def test_qs_sweep_history_merge_preserves_stage_profiles_and_traces() -> None:
    from examples.optimization.generate_qs_ess_sweep import PROBLEM_CONFIGS, _merge_stage_histories

    def stage_result(label: str, wall: float) -> dict:
        return {
            "_history_dump": {
                "history": [
                    {"wall_time_s": 0.0, "objective": 2.0, "qs_objective": 1.0, "aspect": 5.0},
                    {"wall_time_s": wall, "objective": 1.0, "qs_objective": 0.5, "aspect": 5.1},
                ],
                "nfev": 2,
                "njev": 1,
                "success": True,
                "message": label,
                "objective_initial": 2.0,
                "objective_final": 1.0,
                "qs_initial": 1.0,
                "qs_final": 0.5,
                "aspect_initial": 5.0,
                "aspect_final": 5.1,
                "max_nfev": 3,
                "profile": {
                    "exact_tape_build": {"count": 1, "wall_time_s": wall, "mean_wall_time_s": wall},
                    "trial_solve": {"count": 2, "wall_time_s": 2.0 * wall, "mean_wall_time_s": wall},
                },
                "callback_trace": {
                    "enabled": True,
                    "events": [{"index": 0, "kind": "jacobian", "source": "exact_tape_replay", "wall_time_s": wall}],
                    "summary": {"jacobian:exact_tape_replay": {"count": 1, "wall_time_s": wall}},
                },
            }
        }

    merged = _merge_stage_histories(
        [
            ("stage 1", 1, stage_result("one", 0.25)),
            ("stage 2", 2, stage_result("two", 0.5)),
        ],
        problem_cfg=PROBLEM_CONFIGS["qa"],
    )

    assert merged["profile"]["exact_tape_build"]["count"] == 2
    assert merged["profile"]["exact_tape_build"]["wall_time_s"] == 0.75
    assert merged["profile"]["trial_solve"]["count"] == 4
    assert merged["profile"]["trial_solve"]["wall_time_s"] == 1.5
    assert len(merged["stage_profiles"]) == 2
    assert merged["callback_trace"]["summary"]["jacobian:exact_tape_replay"]["count"] == 2
    assert merged["callback_trace"]["summary"]["jacobian:exact_tape_replay"]["wall_time_s"] == 0.75
    assert [event["stage"] for event in merged["callback_trace"]["events"]] == ["stage 1", "stage 2"]


def test_finite_beta_examples_plot_explicitly_after_solve() -> None:
    scripts = [
        ROOT / "examples" / "optimization" / "qa_optimization_finite_beta.py",
        ROOT / "examples" / "optimization" / "qh_optimization_finite_beta.py",
        ROOT / "examples" / "optimization" / "qi_optimization_finite_beta.py",
    ]
    for script in scripts:
        text = script.read_text()
        assert "save_final_outputs(" in text
        assert "plot=" not in text
        assert "vj.plot_3d_boundary_comparison(" in text
        assert "vj.plot_bmag_contours(" in text
        assert "vj.plot_objective_history(" in text


def test_custom_objective_term_residual_shape() -> None:
    from vmec_jax.optimization_workflow import ObjectiveTerm

    term = ObjectiveTerm(
        "custom",
        evaluate=lambda _ctx, _state: [1.0, 3.0],
        target=1.0,
        weight=2.0,
    )

    residual = term.residual(None, None)

    assert residual.shape == (2,)
    assert [float(x) for x in residual] == [0.0, 4.0]


def test_fixed_boundary_result_exposes_teaching_accessors() -> None:
    from vmec_jax.optimization_workflow import FixedBoundaryOptimizationResult

    initial_state = object()
    final_state = object()
    initial_record = (
        1,
        "initial_optimizer",
        np.asarray([0.0, 1.0]),
        {
            "_state_initial": initial_state,
            "_history_dump": {"history": [{"objective": 4.0}], "total_wall_time_s": 0.5},
        },
    )
    final_record = (
        2,
        "final_optimizer",
        np.asarray([2.0, 3.0]),
        {
            "x": np.asarray([3.0, 4.0]),
            "nfev": 7,
            "njev": 3,
            "_state_final": final_state,
            "_history_dump": {
                "history": [{"objective": 2.0}, {"objective": 0.25}],
                "total_wall_time_s": 12.5,
                "nfev": 7,
                "njev": 3,
            },
        },
    )

    result = FixedBoundaryOptimizationResult(
        stage_records=[initial_record, final_record],
        final_optimizer="final_optimizer",
        final_result=final_record[3],
        stage_modes=[1, 2],
    )

    assert result.initial_stage is initial_record
    assert result.final_stage is final_record
    assert result.initial_optimizer == "initial_optimizer"
    assert result.initial_result is initial_record[3]
    assert result.initial_state is initial_state
    assert result.final_state is final_state
    assert result.history is final_record[3]["_history_dump"]
    np.testing.assert_allclose(result.initial_params, [0.0, 1.0])
    np.testing.assert_allclose(result.objective_history, [2.0, 0.25])
    np.testing.assert_allclose(result.final_params, [3.0, 4.0])
    assert result.stage_histories == (
        {"history": [{"objective": 4.0}], "total_wall_time_s": 0.5},
        {
            "history": [{"objective": 2.0}, {"objective": 0.25}],
            "total_wall_time_s": 12.5,
            "nfev": 7,
            "njev": 3,
        },
    )
    assert result.stage_results == (initial_record[3], final_record[3])
    assert result.stage_optimizers == ("initial_optimizer", "final_optimizer")
    assert len(result.stage_initial_params) == 2
    np.testing.assert_allclose(result.stage_initial_params[0], [0.0, 1.0])
    np.testing.assert_allclose(result.stage_initial_params[1], [2.0, 3.0])
    assert result.stage_timing_summaries == (
        {"total_wall_time_s": 0.5, "nfev": None, "njev": None, "nit": None, "mode": 1},
        {"total_wall_time_s": 12.5, "nfev": 7, "njev": 3, "nit": None, "mode": 2},
    )
    assert result.timing_summary == {
        "total_wall_time_s": 12.5,
        "nfev": 7,
        "njev": 3,
        "nit": None,
        "stages": result.stage_timing_summaries,
    }
    assert result.summary == {
        "stage_modes": (1, 2),
        "objective_initial": None,
        "objective_final": None,
        "aspect_final": None,
        "iota_final": None,
        "field_objective_final": None,
        "timing": result.timing_summary,
    }


def test_least_squares_problem_uses_simsopt_weight_semantics() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem

    problem = LeastSquaresProblem.from_tuples([(lambda _ctx, _state: 2.0, 1.0, 4.0)])
    residual = problem.objective_terms[0].residual(None, None)

    assert residual.shape == (1,)
    assert float(residual[0]) == 2.0
    assert problem.scalar_objective_names == ("<lambda>",)
    assert problem.qi_objective_names == ()
    assert problem.objective_names == ("<lambda>",)
    assert problem.objective_count == 1
    assert problem.summary == {
        "objective_count": 1,
        "scalar_objectives": ("<lambda>",),
        "qi_objectives": (),
        "is_qi": False,
        "metadata": {},
    }


def test_least_squares_problem_routes_qi_terms() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem, QuasiIsodynamicOptions, QuasiIsodynamicResidual

    qi_options = QuasiIsodynamicOptions(surfaces=[0.5])
    qi = QuasiIsodynamicResidual(qi_options)
    problem = LeastSquaresProblem.from_tuples([(qi.J, 0.0, 9.0)])

    assert problem.is_qi
    assert len(problem.objective_terms) == 0
    assert len(problem.qi_objective_terms) == 1
    assert problem.qi_objective_terms[0].name == "qi"
    assert problem.qi_options is qi_options
    assert problem.scalar_objective_names == ()
    assert problem.qi_objective_names == ("qi",)
    assert problem.objective_names == ("qi",)


def test_least_squares_problem_rejects_nonzero_qi_targets() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem, QuasiIsodynamicOptions, QuasiIsodynamicResidual

    qi = QuasiIsodynamicResidual(QuasiIsodynamicOptions(surfaces=[0.5]))

    with pytest.raises(ValueError, match="target=0"):
        LeastSquaresProblem.from_tuples([(qi.J, 1.0, 1.0)])


def test_least_squares_problem_rejects_mixed_qi_options() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem, MirrorRatio, QuasiIsodynamicOptions, QuasiIsodynamicResidual

    qi_options = QuasiIsodynamicOptions(surfaces=[0.5])
    other_options = QuasiIsodynamicOptions(surfaces=[0.75])
    qi = QuasiIsodynamicResidual(qi_options)
    mirror = MirrorRatio(threshold=1.2, qi_options=other_options)

    with pytest.raises(ValueError, match="share one QuasiIsodynamicOptions"):
        LeastSquaresProblem.from_tuples([(qi.J, 0.0, 1.0), (mirror.J, 0.0, 1.0)])


def test_qi_field_objectives_raise_outside_qi_solve() -> None:
    from vmec_jax.optimization_workflow import (
        MirrorRatio,
        QuasiIsodynamicOptions,
        QuasiIsodynamicResidual,
        QuasiIsodynamicResidualCeiling,
    )

    qi_options = QuasiIsodynamicOptions(surfaces=[0.5])
    objectives = [
        QuasiIsodynamicResidual(qi_options),
        QuasiIsodynamicResidualCeiling(maximum=1.0e-2, qi_options=qi_options),
        MirrorRatio(threshold=1.2, qi_options=qi_options),
    ]

    for objective in objectives:
        with pytest.raises(RuntimeError, match="inside a QI solve"):
            objective.J(None, None)


def test_mirror_and_elongation_can_be_plain_state_objectives() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem, MaxElongation, MirrorRatio

    problem = LeastSquaresProblem.from_tuples(
        [
            (MirrorRatio(threshold=0.3, surfaces=[0.5, 1.0]).J, 0.0, 4.0),
            (MaxElongation(threshold=8.0).J, 0.0, 9.0),
        ]
    )

    assert not problem.is_qi
    assert problem.qi_options is None
    assert [term.name for term in problem.objective_terms] == ["mirror_ratio", "max_elongation"]


def test_mirror_ratio_objective_records_all_surface_smoothing_options() -> None:
    from vmec_jax.optimization_workflow import MirrorRatio, QuasiIsodynamicOptions

    qi_options = QuasiIsodynamicOptions(surfaces=[0.2, 0.8])
    mirror = MirrorRatio(
        threshold=0.21,
        ntheta=48,
        nphi=64,
        surface_index=None,
        phimin=0.1,
        smooth_extrema=2.0e-2,
        smooth_penalty=1.0e-2,
        qi_options=qi_options,
    )
    term = mirror.to_qi_term(3.0)

    assert mirror.surface_index is None
    assert mirror.phimin == pytest.approx(0.1)
    assert mirror.smooth_extrema == pytest.approx(2.0e-2)
    assert mirror.smooth_penalty == pytest.approx(1.0e-2)
    assert mirror.normalize_surfaces is True
    assert term.name == "mirror_ratio"
    assert term.qi_options is qi_options


def test_boozer_b_target_objective_matches_normalized_spectrum() -> None:
    from vmec_jax.optimization_workflow import BoozerBTarget, QuasiIsodynamicOptions

    qi_options = QuasiIsodynamicOptions(surfaces=[0.5])
    target = BoozerBTarget(
        target_bmnc=np.asarray([[2.0, 0.4, 0.2]]),
        target_bmns=np.asarray([[0.0, 0.2, 0.0]]),
        normalize=True,
        qi_options=qi_options,
    )
    term = target.to_qi_term(2.0)
    residual, total = term.residual_and_total(
        SimpleNamespace(static=SimpleNamespace(cfg=SimpleNamespace(nfp=2))),
        "state",
        {
            "booz": {
                "bmnc_b": np.asarray([[1.0, 0.1, 0.1]]),
                "bmns_b": np.asarray([[0.0, 0.1, 0.0]]),
            }
        },
    )

    expected = np.asarray([0.0, -0.1, 0.0, 0.0, 0.0, 0.0]) * 2.0 / np.sqrt(6.0)
    np.testing.assert_allclose(residual, expected)
    assert total == pytest.approx(float(np.sum(expected * expected)))
    assert term.qi_options is qi_options


def test_least_squares_problem_collects_problem_metadata() -> None:
    from vmec_jax.optimization_workflow import AbsMeanIotaCeiling, AbsMeanIotaFloor, AspectRatio, LeastSquaresProblem, MeanIota

    problem = LeastSquaresProblem.from_tuples(
        [
            (AspectRatio().J, 5.0, 1.0),
            (MeanIota().J, 0.42, 100.0),
            (AbsMeanIotaFloor(0.41).J, 0.0, 1.0),
            (AbsMeanIotaCeiling(0.65).J, 0.0, 1.0),
        ]
    )

    assert problem.metadata == {
        "target_aspect": 5.0,
        "target_iota": 0.42,
        "iota_abs_min": 0.41,
        "iota_abs_max": 0.65,
    }


def test_least_squares_problem_plain_tuple_uses_simsopt_weight_semantics() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem

    def vector_objective(ctx, state):
        assert ctx == "ctx"
        return np.asarray([state, state + 1.0], dtype=float)

    problem = LeastSquaresProblem.from_tuples(
        [
            (vector_objective, np.asarray([1.0, 3.0]), 9.0),
        ]
    )

    assert problem.is_qi is False
    assert problem.objective_terms[0].name == "vector_objective"
    np.testing.assert_allclose(
        np.asarray(problem.objective_terms[0].residual("ctx", 2.0)),
        [3.0, 0.0],
    )

    for bad_weight in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="finite and non-negative"):
            LeastSquaresProblem.from_tuples([(vector_objective, 0.0, bad_weight)])


def test_workflow_stage_policy_helpers_are_explicit() -> None:
    from vmec_jax.optimization_workflow import objectives_track_iota, qs_stage_budget, qs_stage_modes, repeated_stage_modes
    from vmec_jax.optimization_workflow import ObjectiveTerm

    assert qs_stage_modes(max_mode=3, use_mode_continuation=True, continuation_nfev=2) == [1, 1, 2, 2, 2, 3, 3, 3]
    assert qs_stage_modes(max_mode=3, use_mode_continuation=True, continuation_nfev=0) == [3]
    assert qs_stage_modes(max_mode=1, use_mode_continuation=True, continuation_nfev=2) == [1]
    assert repeated_stage_modes(max_mode=3, use_mode_continuation=True, continuation_nfev=0, repeats=4) == [3, 3, 3, 3]
    assert repeated_stage_modes(max_mode=3, use_mode_continuation=True, continuation_nfev=2, repeats=4) == [3, 3, 3, 3]
    assert repeated_stage_modes(max_mode=3, use_mode_continuation=False, continuation_nfev=2, repeats=4) == [3]
    assert qs_stage_budget(stage_mode=2, max_mode=3, max_nfev=30, continuation_nfev=5) == 5
    assert qs_stage_budget(stage_mode=3, max_mode=3, max_nfev=30, continuation_nfev=5) == 30
    assert qs_stage_budget(stage_mode=2, max_mode=3, max_nfev=30, continuation_nfev=0) == 30

    plain = ObjectiveTerm("plain", lambda _ctx, _state: 0.0)
    tracked = ObjectiveTerm("iota", lambda _ctx, _state: 0.0, track_iota=True)
    assert not objectives_track_iota([plain])
    assert objectives_track_iota([plain], target_iota=0.41)
    assert objectives_track_iota([plain, tracked])


def test_fixed_boundary_vmec_from_input_applies_resolution_policy(monkeypatch, tmp_path) -> None:
    import vmec_jax as package
    import vmec_jax.config as config_module
    import vmec_jax.optimization_workflow as workflow

    captured = {}

    def fake_rebuild(indata, *, mpol, ntor):
        captured.update({"indata": indata, "mpol": mpol, "ntor": ntor})
        return "resolved-indata"

    monkeypatch.setattr(package, "load_config", lambda path: ("raw-cfg", f"raw:{Path(path).name}"))
    monkeypatch.setattr(workflow, "rebuild_indata_with_resolution", fake_rebuild)
    monkeypatch.setattr(config_module, "config_from_indata", lambda indata: {"cfg_from": indata})

    vmec = workflow.FixedBoundaryVMEC.from_input(
        tmp_path / "input.test_case",
        max_mode=4,
        min_vmec_mode=5,
        output_dir=tmp_path / "out",
        project_input_boundary_to_max_mode=True,
        include=("rc", "zs", "rs", "zc"),
        fix=("rc00", "zs00"),
    )

    assert captured == {"indata": "raw:input.test_case", "mpol": 6, "ntor": 6}
    assert vmec.input_file == tmp_path / "input.test_case"
    assert vmec.cfg == {"cfg_from": "resolved-indata"}
    assert vmec.indata == "resolved-indata"
    assert vmec.max_mode == 4
    assert vmec.min_vmec_mode == 5
    assert vmec.output_dir == tmp_path / "out"
    assert vmec.project_input_boundary_to_max_mode is True
    assert vmec.include == ("rc", "zs", "rs", "zc")
    assert vmec.fix == ("rc00", "zs00")


def test_workflow_objective_wrappers_dispatch_to_state_helpers(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(
        static="static",
        indata="indata",
        signgs=-1,
        flux="flux",
        pressure="pressure",
    )

    monkeypatch.setattr(
        workflow,
        "equilibrium_aspect_ratio_from_state",
        lambda *, state, static: 7.0 if (state, static) == ("state", "static") else -1.0,
    )
    monkeypatch.setattr(workflow, "mean_iota", lambda ctx_arg, state: 0.35 if (ctx_arg, state) == (ctx, "state") else -1.0)

    def fake_iota_floor(value, target, *, softness):
        assert value == 0.35
        assert target == 0.41
        assert softness == 0.02
        return np.asarray([0.06])

    monkeypatch.setattr(workflow, "smooth_min_abs_iota_residual", fake_iota_floor)

    assert workflow.AspectRatio().J(ctx, "state") == 7.0
    assert workflow.MeanIota().J(ctx, "state") == 0.35
    np.testing.assert_allclose(workflow.AbsMeanIotaFloor(0.41, softness=0.02).J(ctx, "state"), [0.06])

    def fake_qs_residual_from_state(**kwargs):
        assert kwargs == {
            "state": "state",
            "static": "static",
            "indata": "indata",
            "signgs": -1,
            "flux_local": "flux",
            "prof_local": {"pressure": "pressure"},
            "pressure_local": "pressure",
            "surfaces": [0.25, 0.75],
            "helicity_m": 1,
            "helicity_n": -4,
        }
        return {"residuals1d": np.asarray([1.0, 2.0]), "total": 5.0}

    monkeypatch.setattr(workflow, "quasisymmetry_ratio_residual_from_state", fake_qs_residual_from_state)
    qs = workflow.QuasisymmetryRatioResidual(helicity_m=1, helicity_n=-4, surfaces=[0.25, 0.75])
    np.testing.assert_allclose(qs.J(ctx, "state"), [1.0, 2.0])
    assert qs.total(ctx, "state") == 5.0
    term = qs.to_objective_term(target=0.0, residual_weight=2.0)
    np.testing.assert_allclose(term.residual(ctx, "state"), [2.0, 4.0])
    assert term.total(ctx, "state") == 20.0
    with pytest.raises(ValueError, match="target=0"):
        qs.to_objective_term(target=1.0, residual_weight=1.0)

    factory_term = workflow.quasisymmetry_objective(
        helicity_m=1,
        helicity_n=-4,
        surfaces=[0.25, 0.75],
        weight=3.0,
    )
    np.testing.assert_allclose(factory_term.residual(ctx, "state"), [3.0, 6.0])
    assert factory_term.total(ctx, "state") == 45.0


def test_quasisymmetry_workflow_routing_jvp_and_vjp_match_finite_difference(monkeypatch) -> None:
    pytest.importorskip("jax")

    import jax

    from vmec_jax._compat import enable_x64, jnp
    import vmec_jax.optimization_workflow as workflow

    enable_x64(True)

    ctx = SimpleNamespace(
        static="static",
        indata="indata",
        signgs=-1,
        flux="flux",
        pressure="pressure",
    )

    def fake_qs_residual_from_state(**kwargs):
        assert kwargs["static"] == "static"
        assert kwargs["indata"] == "indata"
        assert kwargs["signgs"] == -1
        assert kwargs["flux_local"] == "flux"
        assert kwargs["pressure_local"] == "pressure"
        assert kwargs["surfaces"] == [0.25, 0.75]
        assert kwargs["helicity_m"] == 1
        assert kwargs["helicity_n"] == -4
        x = jnp.asarray(kwargs["state"], dtype=jnp.float64)
        residuals = jnp.stack([x**2 + 0.5 * x, 2.0 * x - 0.25])
        return {"residuals1d": residuals, "total": jnp.dot(residuals, residuals)}

    monkeypatch.setattr(workflow, "quasisymmetry_ratio_residual_from_state", fake_qs_residual_from_state)

    qs = workflow.QuasisymmetryRatioResidual(helicity_m=1, helicity_n=-4, surfaces=[0.25, 0.75])
    problem = workflow.LeastSquaresProblem.from_tuples([(qs.J, 0.0, 9.0)])
    combined = workflow.residuals_from_objectives(problem.objective_terms, ctx)

    assert not problem.is_qi
    assert problem.objective_terms[0].name == "qs"
    assert problem.objective_terms[0].total is not None
    assert combined._n_non_qs == 0

    def routed_residuals(x):
        return combined(jnp.asarray(x, dtype=jnp.float64))

    x0 = jnp.asarray(1.2, dtype=jnp.float64)
    direction = jnp.asarray(-0.4, dtype=jnp.float64)
    eps = jnp.asarray(1.0e-5, dtype=jnp.float64)

    residual0, jvp_ad = jax.jvp(routed_residuals, (x0,), (direction,))
    jvp_fd = (routed_residuals(x0 + eps * direction) - routed_residuals(x0 - eps * direction)) / (2.0 * eps)

    cotangent = jnp.asarray([0.3, -0.7], dtype=jnp.float64)
    _, pullback = jax.vjp(routed_residuals, x0)
    (vjp_ad,) = pullback(cotangent)
    vjp_fd = (
        jnp.vdot(cotangent, routed_residuals(x0 + eps))
        - jnp.vdot(cotangent, routed_residuals(x0 - eps))
    ) / (2.0 * eps)

    raw_total = float(jnp.dot(residual0 / 3.0, residual0 / 3.0))
    assert float(combined._qs_total_from_state(x0)) == pytest.approx(9.0 * raw_total)
    np.testing.assert_allclose(np.asarray(jvp_ad), np.asarray(jvp_fd), rtol=2.0e-7, atol=1.0e-9)
    np.testing.assert_allclose(np.asarray(vjp_ad), np.asarray(vjp_fd), rtol=2.0e-7, atol=1.0e-9)


def test_qi_problem_shared_field_terms_jvp_and_vjp_match_finite_difference() -> None:
    pytest.importorskip("jax")

    import jax

    from vmec_jax._compat import enable_x64, jnp
    from vmec_jax.optimization_workflow import (
        BoozerBTarget,
        LeastSquaresProblem,
        QuasiIsodynamicOptions,
        QuasiIsodynamicResidual,
    )

    enable_x64(True)

    qi_options = QuasiIsodynamicOptions(surfaces=[0.5])
    boozer_target = BoozerBTarget(
        target_bmnc=np.asarray([[1.0, 0.18, 0.07], [1.3, 0.08, 0.22]]),
        target_bmns=np.asarray([[0.0, 0.02, -0.01], [0.0, -0.03, 0.04]]),
        normalize=True,
        qi_options=qi_options,
    )
    problem = LeastSquaresProblem.from_tuples(
        [
            (lambda _ctx, state: jnp.asarray(state, dtype=jnp.float64) ** 2, 0.25, 4.0),
            (QuasiIsodynamicResidual(qi_options).J, 0.0, 9.0),
            (boozer_target.J, 0.0, 16.0),
        ]
    )
    ctx = SimpleNamespace(static=SimpleNamespace(cfg=SimpleNamespace(nfp=2)), indata=None, signgs=1)

    assert problem.is_qi
    assert len(problem.objective_terms) == 1
    assert [term.name for term in problem.qi_objective_terms] == ["qi", "boozer_b_target"]
    assert problem.qi_options is qi_options

    def synthetic_field(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        bmnc = jnp.stack(
            [
                jnp.stack([1.10 + 0.20 * x, 0.20 + 0.03 * x, 0.05 - 0.01 * x]),
                jnp.stack([1.40 + 0.10 * x, 0.10 - 0.02 * x, 0.30 + 0.04 * x]),
            ]
        )
        bmns = jnp.stack(
            [
                jnp.stack([0.0, 0.04 + 0.01 * x, -0.02 + 0.02 * x]),
                jnp.stack([0.0, -0.01 + 0.03 * x, 0.05 - 0.02 * x]),
            ]
        )
        qi_residuals = jnp.stack([0.40 + 0.10 * x, -0.20 + 0.05 * x])
        return {
            "residuals1d": qi_residuals,
            "total": jnp.dot(qi_residuals, qi_residuals),
            "booz": {"bmnc_b": bmnc, "bmns_b": bmns},
        }

    def routed_residuals(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        field = synthetic_field(x)
        scalar_parts = [term.residual(ctx, x) for term in problem.objective_terms]
        qi_parts = [term.residual_and_total(ctx, x, field)[0] for term in problem.qi_objective_terms]
        return jnp.concatenate([*scalar_parts, *qi_parts])

    x0 = jnp.asarray(0.35, dtype=jnp.float64)
    direction = jnp.asarray(0.6, dtype=jnp.float64)
    eps = jnp.asarray(1.0e-5, dtype=jnp.float64)

    residual0, jvp_ad = jax.jvp(routed_residuals, (x0,), (direction,))
    jvp_fd = (routed_residuals(x0 + eps * direction) - routed_residuals(x0 - eps * direction)) / (2.0 * eps)

    cotangent = jnp.linspace(-0.4, 0.5, int(residual0.size), dtype=jnp.float64)
    _, pullback = jax.vjp(routed_residuals, x0)
    (vjp_ad,) = pullback(cotangent)
    vjp_fd = (
        jnp.vdot(cotangent, routed_residuals(x0 + eps))
        - jnp.vdot(cotangent, routed_residuals(x0 - eps))
    ) / (2.0 * eps)

    assert np.all(np.isfinite(np.asarray(residual0)))
    np.testing.assert_allclose(np.asarray(jvp_ad), np.asarray(jvp_fd), rtol=1.0e-7, atol=1.0e-9)
    np.testing.assert_allclose(np.asarray(vjp_ad), np.asarray(vjp_fd), rtol=1.0e-7, atol=1.0e-9)


def test_qi_objective_factories_apply_weights_and_slice_shared_fields(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(static=SimpleNamespace(cfg=SimpleNamespace(nfp=5)), indata="indata", signgs=-1, flux="flux")
    qi_options = workflow.QuasiIsodynamicOptions(surfaces=[0.25, 0.75])

    field_term = workflow.quasi_isodynamic_field_objective(weight=2.0, qi_options=qi_options)
    residual, total = field_term.residual_and_total(
        ctx,
        "state",
        {"residuals1d": np.asarray([1.0, 2.0]), "total": 3.0},
    )
    np.testing.assert_allclose(residual, [2.0, 4.0])
    assert total == 12.0
    assert field_term.qi_options is qi_options

    ceiling_term = workflow.qi_residual_ceiling_objective(
        maximum=0.5,
        weight=4.0,
        smooth_penalty=0.0,
        qi_options=qi_options,
    )
    residual, total = ceiling_term.residual_and_total(ctx, "state", {"total": np.asarray(0.25)})
    np.testing.assert_allclose(residual, [0.0])
    assert total == 0.0
    residual, total = ceiling_term.residual_and_total(ctx, "state", {"total": np.asarray(0.75)})
    np.testing.assert_allclose(residual, [1.0])
    assert total == 1.0
    assert ceiling_term.qi_options is qi_options

    mirror_call_surfaces = []

    def fake_mirror_ratio_penalty_from_boozer_output(
        booz,
        *,
        nfp,
        threshold,
        weights,
        ntheta,
        nphi,
        phimin,
        smooth_extrema,
        smooth_penalty,
    ):
        assert nfp == 5
        assert threshold == 1.2
        assert weights is None
        assert ntheta == 8
        assert nphi == 9
        assert phimin == 0.1
        assert smooth_extrema == 0.0
        assert smooth_penalty == 0.0
        mirror_call_surfaces.append((np.asarray(booz["s_b"]), np.asarray(booz["iota_b"])))
        assert booz["untouched"] == "kept"
        return {"residuals1d": np.asarray([0.5]), "total": 0.25}

    monkeypatch.setattr(workflow, "mirror_ratio_penalty_from_boozer_output", fake_mirror_ratio_penalty_from_boozer_output)
    mirror_term = workflow.qi_mirror_ratio_objective(
        threshold=1.2,
        weight=3.0,
        ntheta=8,
        nphi=9,
        surface_index=1,
        phimin=0.1,
        qi_options=qi_options,
    )
    residual, total = mirror_term.residual_and_total(
        ctx,
        "state",
        {
            "booz": {
                "bmnc_b": np.zeros((2, 3)),
                "bmns_b": np.ones((2, 3)),
                "iota_b": np.asarray([0.2, 0.3]),
                "s_b": np.asarray([0.25, 0.75]),
                "untouched": "kept",
            }
        },
    )
    np.testing.assert_allclose(residual, [1.5])
    assert total == 2.25
    assert mirror_term.qi_options is qi_options
    np.testing.assert_array_equal(mirror_call_surfaces[-1][0], [0.75])
    np.testing.assert_array_equal(mirror_call_surfaces[-1][1], [0.3])

    negative_surface_term = workflow.qi_mirror_ratio_objective(
        threshold=1.2,
        weight=3.0,
        ntheta=8,
        nphi=9,
        surface_index=-1,
        phimin=0.1,
        qi_options=qi_options,
    )
    residual, total = negative_surface_term.residual_and_total(
        ctx,
        "state",
        {
            "booz": {
                "bmnc_b": np.zeros((2, 3)),
                "bmns_b": np.ones((2, 3)),
                "iota_b": np.asarray([0.2, 0.4]),
                "s_b": np.asarray([0.25, 1.0]),
                "untouched": "kept",
            }
        },
    )
    np.testing.assert_allclose(residual, [1.5])
    assert total == 2.25
    np.testing.assert_array_equal(mirror_call_surfaces[-1][0], [1.0])
    np.testing.assert_array_equal(mirror_call_surfaces[-1][1], [0.4])

    out_of_bounds_surface_term = workflow.qi_mirror_ratio_objective(
        threshold=1.2,
        surface_index=-3,
        qi_options=qi_options,
    )
    with pytest.raises(ValueError, match="outside the Boozer surface range"):
        out_of_bounds_surface_term.residual_and_total(
            ctx,
            "state",
            {"booz": {"bmnc_b": np.zeros((2, 3))}},
        )

    captured_weights = {}

    def fake_all_surface_mirror_ratio_penalty_from_boozer_output(
        booz,
        *,
        nfp,
        threshold,
        weights,
        ntheta,
        nphi,
        phimin,
        smooth_extrema,
        smooth_penalty,
    ):
        del booz, nfp, threshold, ntheta, nphi, phimin, smooth_extrema, smooth_penalty
        captured_weights["weights"] = weights
        return {"residuals1d": np.asarray([0.5, 0.5]), "total": 0.5}

    monkeypatch.setattr(
        workflow,
        "mirror_ratio_penalty_from_boozer_output",
        fake_all_surface_mirror_ratio_penalty_from_boozer_output,
    )
    all_surface_term = workflow.qi_mirror_ratio_objective(
        threshold=1.2,
        weight=1.0,
        surface_index=None,
        normalize_surfaces=True,
    )
    all_surface_term.residual_and_total(
        ctx,
        "state",
        {"booz": {"bmnc_b": np.zeros((2, 3))}},
    )
    np.testing.assert_allclose(captured_weights["weights"], [0.5, 0.5])

    def fake_max_elongation_penalty_from_state(
        *,
        state,
        static,
        threshold,
        ntheta,
        nphi,
        smooth_extrema=0.0,
        smooth_penalty=0.0,
    ):
        assert state == "state"
        assert static is ctx.static
        assert threshold == 4.0
        assert ntheta == 10
        assert nphi == 11
        assert smooth_extrema == 0.0
        assert smooth_penalty == 0.0
        return {"residuals1d": np.asarray([2.0]), "total": 4.0}

    monkeypatch.setattr(workflow, "max_elongation_penalty_from_state", fake_max_elongation_penalty_from_state)
    elongation_term = workflow.qi_max_elongation_objective(threshold=4.0, weight=0.5, ntheta=10, nphi=11)
    residual, total = elongation_term.residual_and_total(ctx, "state", {})
    np.testing.assert_allclose(residual, [1.0])
    assert total == 1.0

    def fake_lgradb_penalty_from_state(*, state, static, indata, signgs, flux_local, threshold, s_index, ntheta, nphi, smooth_penalty):
        assert state == "state"
        assert static is ctx.static
        assert indata == "indata"
        assert signgs == -1
        assert flux_local == "flux"
        assert threshold == 0.3
        assert s_index == -2
        assert ntheta == 12
        assert nphi == 13
        assert smooth_penalty == 0.01
        return {"residuals1d": np.asarray([4.0]), "total": 16.0}

    monkeypatch.setattr(workflow, "lgradb_penalty_from_state", fake_lgradb_penalty_from_state)
    lgradb_term = workflow.qi_lgradb_objective(
        threshold=0.3,
        weight=0.25,
        s_index=-2,
        ntheta=12,
        nphi=13,
        smooth_penalty=0.01,
    )
    residual, total = lgradb_term.residual_and_total(ctx, "state", {})
    np.testing.assert_allclose(residual, [1.0])
    assert total == 1.0


def test_lower_bound_and_lgradb_objective_edge_paths(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(static="static", indata="indata", signgs=1, flux="flux")
    monkeypatch.setattr(
        workflow,
        "finite_beta_scalars_from_state",
        lambda **_kwargs: {"vp": np.asarray([1.0, 2.0])},
    )
    well = workflow.MagneticWell(minimum=0.1, softness=0.01)
    assert float(well.well(ctx, "state")) == 0.0
    with pytest.raises(ValueError, match="target=0"):
        well.to_objective_term(target=1.0, residual_weight=1.0)
    with pytest.raises(ValueError, match="target=0"):
        workflow.DMerc().to_objective_term(target=1.0, residual_weight=1.0)
    with pytest.raises(ValueError, match="target=0"):
        workflow.LgradB(threshold=0.3).to_objective_term(target=1.0, residual_weight=1.0)

    def fake_lgradb_penalty_from_state(**kwargs):
        assert kwargs["state"] == "state"
        assert kwargs["threshold"] == 0.3
        assert kwargs["s_index"] == -1
        assert kwargs["ntheta"] == 9
        assert kwargs["nphi"] == 7
        assert kwargs["smooth_penalty"] == 0.02
        return {"residuals1d": np.asarray([2.0]), "total": 4.0}

    monkeypatch.setattr(workflow, "lgradb_penalty_from_state", fake_lgradb_penalty_from_state)
    term = workflow.LgradB(threshold=0.3, smooth_penalty=0.02).to_qi_term(residual_weight=5.0)
    residual, total = term.residual_and_total(ctx, "state", {})
    np.testing.assert_allclose(residual, [10.0])
    assert total == 100.0


def test_least_squares_solve_dispatches_regular_problem(monkeypatch, tmp_path) -> None:
    import vmec_jax.optimization_workflow as workflow
    from vmec_jax.optimization_workflow import AbsMeanIotaFloor, AspectRatio, LeastSquaresProblem, MeanIota

    captured = {}

    def fake_run_fixed_boundary_objective_optimization(**kwargs):
        captured.update(kwargs)
        return "regular-result"

    monkeypatch.setattr(workflow, "run_fixed_boundary_objective_optimization", fake_run_fixed_boundary_objective_optimization)
    vmec = SimpleNamespace(
        cfg="cfg",
        indata="indata",
        max_mode=2,
        output_dir=tmp_path,
        include=("rc", "zs"),
        fix=("rc00",),
        project_input_boundary_to_max_mode=True,
    )
    problem = LeastSquaresProblem.from_tuples(
        [
            (AspectRatio().J, 6.0, 1.0),
            (MeanIota().J, 0.42, 4.0),
            (AbsMeanIotaFloor(0.41).J, 0.0, 9.0),
        ]
    )

    result = workflow.least_squares_solve(
        vmec,
        problem,
        stage_modes=[1, 2],
        max_nfev=7,
        continuation_nfev=3,
        method="scipy",
        use_ess=True,
        ess_alpha=2.5,
        solver_device="cpu",
        save_stage_wouts=True,
        save_final_outputs=False,
    )

    assert result == "regular-result"
    assert captured["cfg"] == "cfg"
    assert captured["indata"] == "indata"
    assert captured["stage_modes"] == [1, 2]
    assert captured["max_mode"] == 2
    assert captured["max_nfev"] == 7
    assert captured["continuation_nfev"] == 3
    assert captured["use_ess"] is True
    assert captured["ess_alpha"] == 2.5
    assert captured["target_aspect"] == 6.0
    assert captured["target_iota"] == 0.42
    assert captured["iota_abs_min"] == 0.41
    assert captured["include"] == ("rc", "zs")
    assert captured["fix"] == ("rc00",)
    assert captured["solver_device"] == "cpu"
    assert captured["save_stage_wouts"] is True
    assert captured["save_final_outputs"] is False


def test_least_squares_solve_dispatches_qi_problem(monkeypatch, tmp_path) -> None:
    import vmec_jax.optimization_workflow as workflow
    from vmec_jax.optimization_workflow import LeastSquaresProblem, QuasiIsodynamicOptions, QuasiIsodynamicResidual

    captured = {}

    def fake_run_quasi_isodynamic_objective_optimization(**kwargs):
        captured.update(kwargs)
        return "qi-result"

    monkeypatch.setattr(workflow, "run_quasi_isodynamic_objective_optimization", fake_run_quasi_isodynamic_objective_optimization)
    qi_options = QuasiIsodynamicOptions(
        surfaces=[0.25, 0.75],
        mboz=8,
        nboz=7,
        nphi=21,
        nalpha=11,
        n_bounce=13,
        include_bounce_endpoints=True,
        softness=0.03,
        width_weight=2.0,
        shuffle_profile_nphi_out=41,
        weighted_shuffle_profile_weight=0.7,
        weighted_shuffle_profile_softness=0.04,
        phimin=0.1,
        jit_booz=False,
    )
    vmec = SimpleNamespace(
        cfg="cfg",
        indata="indata",
        max_mode=3,
        output_dir=tmp_path,
        include=("rc", "zs", "rs", "zc"),
        fix=("rc00",),
        project_input_boundary_to_max_mode=False,
    )
    problem = LeastSquaresProblem.from_tuples([(QuasiIsodynamicResidual(qi_options).J, 0.0, 16.0)])

    result = workflow.least_squares_solve(
        vmec,
        problem,
        stage_modes=[1, 1, 2, 3],
        max_nfev=9,
        continuation_nfev=4,
        use_mode_continuation=False,
        inner_max_iter=0,
        inner_ftol=0.0,
        trial_max_iter=0,
        trial_ftol=0.0,
        solver_device="gpu",
        save_final_outputs=False,
    )

    assert result == "qi-result"
    assert captured["cfg"] == "cfg"
    assert captured["indata"] == "indata"
    assert captured["stage_modes"] == [1, 1, 2, 3]
    assert captured["max_mode"] == 3
    assert captured["max_nfev"] == 9
    assert captured["continuation_nfev"] == 4
    assert captured["use_mode_continuation"] is False
    assert captured["surfaces"] == [0.25, 0.75]
    assert captured["mboz"] == 8
    assert captured["nboz"] == 7
    assert captured["nphi"] == 21
    assert captured["nalpha"] == 11
    assert captured["n_bounce"] == 13
    assert captured["include_bounce_endpoints"] is True
    assert captured["softness"] == 0.03
    assert captured["width_weight"] == 2.0
    assert captured["shuffle_profile_nphi_out"] == 41
    assert captured["weighted_shuffle_profile_weight"] == 0.7
    assert captured["weighted_shuffle_profile_softness"] == 0.04
    assert captured["phimin"] == 0.1
    assert captured["jit_booz"] is False
    assert captured["include"] == ("rc", "zs", "rs", "zc")
    assert captured["project_input_boundary_to_max_mode"] is False
    assert captured["inner_max_iter"] == 0
    assert captured["inner_ftol"] == 0.0
    assert captured["trial_max_iter"] == 0
    assert captured["trial_ftol"] == 0.0
    assert captured["solver_device"] == "gpu"
    assert captured["save_final_outputs"] is False


def test_quasi_isodynamic_options_default_to_jitted_boozer_path() -> None:
    from vmec_jax.optimization_workflow import QuasiIsodynamicOptions

    assert QuasiIsodynamicOptions(surfaces=[0.5]).jit_booz is True


def test_lgradb_tuple_stays_regular_state_objective() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem, LgradB

    lgradb = LgradB(threshold=0.30)
    problem = LeastSquaresProblem.from_tuples([(lgradb.J, 0.0, 0.01)])

    assert not problem.is_qi
    assert len(problem.objective_terms) == 1
    assert problem.objective_terms[0].name == "LgradB"
    assert len(problem.qi_objective_terms) == 0


def test_dmerc_tuple_stays_regular_state_objective() -> None:
    from vmec_jax.optimization_workflow import DMerc, LeastSquaresProblem

    problem = LeastSquaresProblem.from_tuples([(DMerc().J, 0.0, 0.25)])

    assert not problem.is_qi
    assert len(problem.objective_terms) == 1
    assert problem.objective_terms[0].name == "DMerc"
    assert len(problem.qi_objective_terms) == 0


def test_magnetic_well_tuple_stays_regular_state_objective(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow
    from vmec_jax import api
    from vmec_jax.optimization_workflow import LeastSquaresProblem, MagneticWell

    assert api.MagneticWell is MagneticWell
    assert api.DMerc is workflow.DMerc

    monkeypatch.setattr(
        workflow,
        "finite_beta_scalars_from_state",
        lambda **_kwargs: {"vp": np.asarray([0.0, 2.0, 1.5, 1.0])},
    )
    objective = MagneticWell(minimum=0.8, softness=1.0e-3)
    problem = LeastSquaresProblem.from_tuples([(objective.J, 0.0, 4.0)])
    term = problem.objective_terms[0]

    assert not problem.is_qi
    assert term.name == "magnetic_well"
    assert len(problem.qi_objective_terms) == 0
    assert float(term.residual(SimpleNamespace(static="static", indata="indata", signgs=1), "state")[0]) > 0.0


def test_public_api_reexports_example_optimization_contract() -> None:
    import vmec_jax as vj
    import vmec_jax.api as api
    import vmec_jax.booz as booz
    import vmec_jax.optimization_workflow as workflow
    import vmec_jax.plotting as plotting
    import vmec_jax.qi_diagnostics as qi_diagnostics
    from vmec_jax import finite_beta

    workflow_names = (
        "FixedBoundaryVMEC",
        "FixedBoundaryOptimizationResult",
        "LeastSquaresProblem",
        "BoundaryModeLimits",
        "ObjectiveTerm",
        "QIObjectiveTerm",
        "AspectRatio",
        "AugmentedLagrangianConstraint",
        "MeanIota",
        "AbsMeanIotaFloor",
        "QuasisymmetryRatioResidual",
        "QuasiIsodynamicOptions",
        "QuasiIsodynamicResidual",
        "QuasiIsodynamicResidualCeiling",
        "MirrorRatio",
        "MaxElongation",
        "LgradB",
        "BoozerBTarget",
        "boozer_b_target_from_wout",
        "qi_boozer_b_target_objective",
        "qi_max_elongation_constraint",
        "qi_mirror_ratio_constraint",
        "BetaTotal",
        "VolavgB",
        "BDotB",
        "BDotGradV",
        "BVector",
        "JDotB",
        "JVector",
        "ToroidalCurrent",
        "ToroidalCurrentGradient",
        "RedlBootstrapMismatch",
        "least_squares_solve",
        "prepare_simple_omnigenity_seed_input",
        "interpolate_indata_boundary",
        "qs_stage_modes",
        "repeated_stage_modes",
        "save_optimization_result",
    )
    for name in workflow_names:
        assert getattr(api, name) is getattr(workflow, name)
        assert getattr(vj, name) is getattr(workflow, name)

    for name in (
        "FiniteBetaTargets",
        "finite_beta_global_residuals_from_state",
        "finite_beta_scalars_from_state",
    ):
        assert getattr(api, name) is getattr(finite_beta, name)

    for name in (
        "QIDiagnosticOptions",
        "qi_diagnostics_from_boozer_output",
        "qi_diagnostics_from_state",
        "qi_cleanup_candidate_promotable",
        "rank_qi_seed_records",
    ):
        assert getattr(api, name) is getattr(qi_diagnostics, name)
        assert getattr(vj, name) is getattr(qi_diagnostics, name)

    for name in (
        "plot_3d_boundary_comparison",
        "plot_bmag_contours",
        "plot_boozmn",
        "plot_boozmn_bmag_contours",
        "plot_boozmn_mode_families",
        "plot_boozmn_spectrum",
        "plot_boozer_bmag_contours_from_state",
        "plot_boozer_lcfs_bmag_comparison",
        "plot_objective_history",
    ):
        assert getattr(api, name) is getattr(plotting, name)

    for name in (
        "BoozConfig",
        "parse_booz_surfaces",
        "read_booz_config",
        "resolve_boozmn_path",
        "run_booz_xform",
    ):
        assert getattr(api, name) is getattr(booz, name)
        assert getattr(vj, name) is getattr(booz, name)


def test_jxbforce_profile_tuple_stays_regular_state_objective() -> None:
    from vmec_jax.optimization_workflow import (
        BDotB,
        BDotGradV,
        BVector,
        JDotB,
        JVector,
        LeastSquaresProblem,
        RedlBootstrapMismatch,
        ToroidalCurrent,
        ToroidalCurrentGradient,
    )

    problem = LeastSquaresProblem.from_tuples(
        [
            (JDotB(surfaces=(0.25, 0.75)).J, 0.0, 0.25),
            (BDotB(surfaces=(0.5,)).J, 1.0, 0.10),
            (BDotGradV().J, 0.0, 0.05),
            (BVector(s_index=-1).J, 0.0, 0.05),
            (JVector(surfaces=(0.25,)).J, 0.0, 0.05),
            (ToroidalCurrent(surfaces=(0.75,)).J, 0.0, 0.20),
            (ToroidalCurrentGradient().J, 0.0, 0.30),
            (
                RedlBootstrapMismatch(
                    helicity_n=0,
                    ne_coeffs=[3.0e20, 0.0, -2.5e20],
                    Te_coeffs=[8.0e3, -6.0e3],
                    surfaces=(0.25, 0.75),
                ).J,
                0.0,
                0.40,
            ),
        ]
    )

    assert not problem.is_qi
    assert [term.name for term in problem.objective_terms] == [
        "jdotb",
        "bdotb",
        "bdotgradv",
        "B_vector",
        "J_vector",
        "torcur",
        "torcur_prime",
        "redl_bootstrap_mismatch",
    ]
    assert len(problem.qi_objective_terms) == 0


def test_finite_beta_objective_terms_expose_residuals_totals_and_metadata(monkeypatch) -> None:
    from vmec_jax._compat import jnp
    import vmec_jax.optimization_workflow as workflow
    from vmec_jax.optimization_workflow import (
        BetaTotal,
        DMerc,
        JDotB,
        JVector,
        LeastSquaresProblem,
        MagneticWell,
        RedlBootstrapMismatch,
        ToroidalCurrent,
        ToroidalCurrentGradient,
        VolavgB,
        residuals_from_objectives,
    )

    def fake_scalars_from_state(*, state, **_kwargs):
        x = jnp.asarray(state, dtype=jnp.float64)
        return {
            "volavgB": 2.0 + 0.2 * x,
            "betatotal": 0.04 + 0.01 * x,
            "vp": jnp.asarray([0.0, 1.2 + 0.01 * x, 1.0, 0.9], dtype=jnp.float64),
        }

    def fake_mercier_terms_from_state(*, state, include_channels=False, **_kwargs):
        x = jnp.asarray(state, dtype=jnp.float64)
        terms = {
            "DMerc": jnp.asarray([0.0, -0.20 + 0.01 * x, 0.10 + 0.02 * x, 0.0], dtype=jnp.float64),
            "jdotb": jnp.asarray([0.0, 10.0 + x, 20.0 + 2.0 * x, 0.0], dtype=jnp.float64),
            "torcur": jnp.asarray([0.0, 3.0 + 0.3 * x, 5.0 + 0.5 * x, 0.0], dtype=jnp.float64),
            "ip": jnp.asarray([0.0, 7.0 + 0.7 * x, 11.0 + 1.1 * x, 0.0], dtype=jnp.float64),
        }
        if include_channels:
            terms.update(
                {
                    "itheta": x * jnp.ones((4, 2, 3), dtype=jnp.float64),
                    "izeta": 2.0 * x * jnp.ones((4, 2, 3), dtype=jnp.float64),
                    "sqrtg": 4.0 * jnp.ones((4, 2, 3), dtype=jnp.float64),
                }
            )
        return terms

    def fake_redl_bootstrap_mismatch_from_state(**_kwargs):
        residuals = jnp.asarray([0.50, -0.25], dtype=jnp.float64)
        return {"residuals1d": residuals, "total": jnp.dot(residuals, residuals)}

    monkeypatch.setattr(workflow, "finite_beta_scalars_from_state", fake_scalars_from_state)
    monkeypatch.setattr(workflow, "mercier_terms_from_state", fake_mercier_terms_from_state)
    monkeypatch.setattr(workflow, "redl_bootstrap_mismatch_from_state", fake_redl_bootstrap_mismatch_from_state)

    objectives = [
        VolavgB(),
        BetaTotal(),
        MagneticWell(minimum=0.50, softness=0.05),
        DMerc(minimum=0.0, softness=0.05),
        JDotB(surfaces=(0.25, 0.75), normalize=10.0),
        ToroidalCurrent(surfaces=(0.25, 0.75), normalize=2.0),
        ToroidalCurrentGradient(surfaces=(0.75,), normalize=10.0),
        JVector(surfaces=(0.25,), normalize=2.0),
        RedlBootstrapMismatch(
            helicity_n=0,
            ne_coeffs=[3.0e20, 0.0, -2.5e20],
            Te_coeffs=[8.0e3, -6.0e3],
            surfaces=(0.25, 0.75),
        ),
    ]
    targets = [
        1.8,
        0.04,
        0.0,
        0.0,
        np.asarray([1.0, 2.0]),
        np.asarray([1.5, 2.5]),
        np.asarray([1.0]),
        0.0,
        0.0,
    ]
    objective_weights = [4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]

    problem = LeastSquaresProblem.from_tuples(
        [(objective.J, target, weight) for objective, target, weight in zip(objectives, targets, objective_weights)]
    )
    ctx = SimpleNamespace(static=SimpleNamespace(s=np.asarray([0.0, 0.25, 0.75, 1.0])), indata=None, signgs=1)
    state = jnp.asarray(1.0, dtype=jnp.float64)

    assert not problem.is_qi
    assert problem.metadata == {}
    assert [term.name for term in problem.objective_terms] == [
        "volavgB",
        "betatotal",
        "magnetic_well",
        "DMerc",
        "jdotb",
        "torcur",
        "torcur_prime",
        "J_vector",
        "redl_bootstrap_mismatch",
    ]
    np.testing.assert_allclose([term.weight for term in problem.objective_terms], np.sqrt(objective_weights))

    expected_blocks = []
    for term, objective, target, weight in zip(problem.objective_terms, objectives, targets, np.sqrt(objective_weights)):
        raw = workflow._as_vector(objective.J(ctx, state))
        target_arr = jnp.asarray(target, dtype=jnp.float64)
        if int(target_arr.ndim) == 0:
            target_arr = jnp.full_like(raw, target_arr)
        expected = float(weight) * (raw - jnp.ravel(target_arr))
        expected_blocks.append(expected)
        np.testing.assert_allclose(np.asarray(term.residual(ctx, state)), np.asarray(expected), rtol=1e-12, atol=1e-12)

    combined = residuals_from_objectives(problem.objective_terms, ctx)
    np.testing.assert_allclose(
        np.asarray(combined(state)),
        np.asarray(jnp.concatenate(expected_blocks)),
        rtol=1e-12,
        atol=1e-12,
    )
    redl_term = problem.objective_terms[-1]
    assert redl_term.total is not None
    np.testing.assert_allclose(float(redl_term.total(ctx, state)), 100.0 * (0.50**2 + (-0.25) ** 2))
    np.testing.assert_allclose(float(combined._qs_total_from_state(state)), float(redl_term.total(ctx, state)))


def test_finite_beta_workflow_objectives_are_jax_differentiable(monkeypatch) -> None:
    pytest.importorskip("jax")
    import jax

    from vmec_jax._compat import jnp
    import vmec_jax.optimization_workflow as workflow
    from vmec_jax.optimization_workflow import (
        BDotB,
        BDotGradV,
        BVector,
        BetaTotal,
        DMerc,
        JDotB,
        JVector,
        MagneticWell,
        RedlBootstrapMismatch,
        ToroidalCurrent,
        ToroidalCurrentGradient,
        VolavgB,
    )

    def fake_scalars_from_state(*, state, **_kwargs):
        scale = jnp.asarray(state, dtype=jnp.float64)
        return {
            "volavgB": 2.0 + 0.5 * scale,
            "betatotal": 0.03 + 0.01 * scale,
            "vp": jnp.asarray([0.0, 1.0 + scale, 0.8, 0.6], dtype=jnp.float64),
        }

    def fake_mercier_terms_from_state(*, state, **_kwargs):
        scale = jnp.asarray(state, dtype=jnp.float64)
        return {
            "DMerc": jnp.asarray(
                [0.0, 0.02 + 0.01 * scale, -0.03 + 0.02 * scale, 0.0],
                dtype=jnp.float64,
            ),
            "jdotb": jnp.asarray([0.0, 0.10 + 0.01 * scale, 0.20 + 0.02 * scale, 0.0], dtype=jnp.float64),
            "bdotb": jnp.asarray([0.0, 1.00 + 0.10 * scale, 1.20 + 0.20 * scale, 0.0], dtype=jnp.float64),
            "bdotgradv": jnp.asarray([0.0, 2.00 + 0.20 * scale, 2.20 + 0.30 * scale, 0.0], dtype=jnp.float64),
            "torcur": jnp.asarray([0.0, 0.40 + 0.04 * scale, 0.60 + 0.06 * scale, 0.0], dtype=jnp.float64),
            "ip": jnp.asarray([0.0, 1.40 + 0.14 * scale, 1.60 + 0.16 * scale, 0.0], dtype=jnp.float64),
            "itheta": scale * jnp.ones((4, 2, 3), dtype=jnp.float64),
            "izeta": (2.0 * scale) * jnp.ones((4, 2, 3), dtype=jnp.float64),
            "sqrtg": 4.0 * jnp.ones((4, 2, 3), dtype=jnp.float64),
        }

    def fake_redl_bootstrap_mismatch_from_state(*, state, **_kwargs):
        scale = jnp.asarray(state, dtype=jnp.float64)
        residuals = jnp.asarray([0.2 + 0.02 * scale, -0.1 + 0.01 * scale], dtype=jnp.float64)
        return {"residuals1d": residuals, "total": jnp.dot(residuals, residuals)}

    monkeypatch.setattr(workflow, "finite_beta_scalars_from_state", fake_scalars_from_state)
    monkeypatch.setattr(workflow, "mercier_terms_from_state", fake_mercier_terms_from_state)
    monkeypatch.setattr(workflow, "redl_bootstrap_mismatch_from_state", fake_redl_bootstrap_mismatch_from_state)
    monkeypatch.setattr(
        workflow,
        "b_cartesian_from_state",
        lambda state, *_args, **_kwargs: jnp.asarray(state, dtype=jnp.float64) * jnp.ones((2, 3, 3), dtype=jnp.float64),
    )
    ctx = SimpleNamespace(static=SimpleNamespace(s=np.asarray([0.0, 0.25, 0.75, 1.0])), indata=None, signgs=1)

    vol_value, vol_grad = jax.value_and_grad(lambda x: VolavgB().J(ctx, x))(jnp.asarray(1.0))
    beta_value, beta_grad = jax.value_and_grad(lambda x: BetaTotal().J(ctx, x))(jnp.asarray(1.0))
    well = MagneticWell(minimum=0.7, softness=1.0e-2)
    well_value, well_grad = jax.value_and_grad(lambda x: well.J(ctx, x))(jnp.asarray(1.0))
    dmerc = DMerc(minimum=0.0, softness=1.0e-2)
    dmerc_value, dmerc_grad = jax.value_and_grad(lambda x: jnp.sum(dmerc.J(ctx, x)))(jnp.asarray(1.0))
    jdotb_value, jdotb_grad = jax.value_and_grad(lambda x: jnp.sum(JDotB(surfaces=(0.25, 0.75)).J(ctx, x)))(
        jnp.asarray(1.0)
    )
    bdotb_value, bdotb_grad = jax.value_and_grad(lambda x: jnp.sum(BDotB().J(ctx, x)))(jnp.asarray(1.0))
    bdotgradv_value, bdotgradv_grad = jax.value_and_grad(lambda x: jnp.sum(BDotGradV().J(ctx, x)))(
        jnp.asarray(1.0)
    )
    b_vector_value, b_vector_grad = jax.value_and_grad(lambda x: jnp.sum(BVector().J(ctx, x)))(jnp.asarray(1.0))
    j_vector_value, j_vector_grad = jax.value_and_grad(lambda x: jnp.sum(JVector(surfaces=(0.25,)).J(ctx, x)))(
        jnp.asarray(1.0)
    )
    torcur_value, torcur_grad = jax.value_and_grad(lambda x: jnp.sum(ToroidalCurrent().J(ctx, x)))(jnp.asarray(1.0))
    torcur_prime_value, torcur_prime_grad = jax.value_and_grad(
        lambda x: jnp.sum(ToroidalCurrentGradient(surfaces=(0.25, 0.75)).J(ctx, x))
    )(jnp.asarray(1.0))
    redl_value, redl_grad = jax.value_and_grad(
        lambda x: jnp.sum(
            RedlBootstrapMismatch(
                helicity_n=0,
                ne_coeffs=[3.0e20, 0.0, -2.5e20],
                Te_coeffs=[8.0e3, -6.0e3],
                surfaces=(0.25, 0.75),
            ).J(ctx, x)
        )
    )(jnp.asarray(1.0))

    np.testing.assert_allclose(np.asarray(vol_value), 2.5)
    np.testing.assert_allclose(np.asarray(vol_grad), 0.5)
    np.testing.assert_allclose(np.asarray(beta_value), 0.04)
    np.testing.assert_allclose(np.asarray(beta_grad), 0.01)
    assert np.isfinite(np.asarray(well_value))
    assert np.isfinite(np.asarray(well_grad))
    assert np.isfinite(np.asarray(dmerc_value))
    assert np.isfinite(np.asarray(dmerc_grad))
    assert abs(float(np.asarray(dmerc_grad))) > 0.0
    np.testing.assert_allclose(np.asarray(jdotb_value), 0.33)
    np.testing.assert_allclose(np.asarray(jdotb_grad), 0.03)
    np.testing.assert_allclose(np.asarray(bdotb_value), 2.5)
    np.testing.assert_allclose(np.asarray(bdotb_grad), 0.3)
    np.testing.assert_allclose(np.asarray(bdotgradv_value), 4.7)
    np.testing.assert_allclose(np.asarray(bdotgradv_grad), 0.5)
    np.testing.assert_allclose(np.asarray(b_vector_value), 18.0)
    np.testing.assert_allclose(np.asarray(b_vector_grad), 18.0)
    np.testing.assert_allclose(np.asarray(j_vector_value), 4.5)
    np.testing.assert_allclose(np.asarray(j_vector_grad), 4.5)
    np.testing.assert_allclose(np.asarray(torcur_value), 1.10)
    np.testing.assert_allclose(np.asarray(torcur_grad), 0.10)
    np.testing.assert_allclose(np.asarray(torcur_prime_value), 3.30)
    np.testing.assert_allclose(np.asarray(torcur_prime_grad), 0.30)
    np.testing.assert_allclose(np.asarray(redl_value), 0.13)
    np.testing.assert_allclose(np.asarray(redl_grad), 0.03)


def test_jxbforce_and_current_objective_gradients_match_finite_difference(monkeypatch) -> None:
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    import vmec_jax.optimization_workflow as workflow
    from vmec_jax.optimization_workflow import BVector, JDotB, JVector, ToroidalCurrent

    def fake_mercier_terms_from_state(*, state, **_kwargs):
        x = jnp.asarray(state, dtype=jnp.float64)
        return {
            "jdotb": jnp.asarray([0.0, 0.1 + 0.02 * x**2, 0.2 + 0.03 * x**2, 0.0], dtype=jnp.float64),
            "torcur": jnp.asarray([0.0, 0.4 + 0.04 * x**2, 0.6 + 0.06 * x**2, 0.0], dtype=jnp.float64),
            "itheta": x**2 * jnp.ones((4, 2, 3), dtype=jnp.float64),
            "izeta": 2.0 * x**2 * jnp.ones((4, 2, 3), dtype=jnp.float64),
            "sqrtg": 4.0 * jnp.ones((4, 2, 3), dtype=jnp.float64),
        }

    monkeypatch.setattr(workflow, "mercier_terms_from_state", fake_mercier_terms_from_state)
    monkeypatch.setattr(
        workflow,
        "b_cartesian_from_state",
        lambda state, *_args, **_kwargs: jnp.asarray(state, dtype=jnp.float64) ** 2
        * jnp.ones((2, 3, 3), dtype=jnp.float64),
    )
    ctx = SimpleNamespace(static=SimpleNamespace(s=np.asarray([0.0, 0.25, 0.75, 1.0])), indata=None, signgs=1)

    def centered_fd(fn, x0, eps=1.0e-6):
        return (float(fn(x0 + eps)) - float(fn(x0 - eps))) / (2.0 * eps)

    import jax

    for objective in (
        JDotB(surfaces=(0.25, 0.75)),
        ToroidalCurrent(surfaces=(0.25, 0.75)),
        BVector(),
        JVector(surfaces=(0.25,)),
    ):
        fn = lambda x, objective=objective: jnp.sum(objective.J(ctx, jnp.asarray(x, dtype=jnp.float64)))
        ad_grad = float(jax.grad(fn)(jnp.asarray(1.3, dtype=jnp.float64)))
        fd_grad = centered_fd(fn, 1.3)
        np.testing.assert_allclose(ad_grad, fd_grad, rtol=1e-6, atol=1e-8)

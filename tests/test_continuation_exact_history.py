from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.namelist import InData
from vmec_jax.optimization import BoundaryParamSpec, FixedBoundaryExactOptimizer


def _stage_history(objective: float) -> dict:
    return {
        "history": [{"objective": float(objective), "wall_time_s": float(objective)}],
        "nfev": 1,
        "njev": 1,
        "objective_initial": float(objective),
        "objective_final": float(objective),
        "qs_initial": float(objective),
        "qs_final": float(objective),
        "aspect_initial": 1.0,
        "aspect_final": 1.0,
    }


@pytest.mark.parametrize(
    ("runner", "builder_name", "extra_kwargs"),
    [
        ("run_fixed_boundary_objective_optimization", "build_fixed_boundary_objective_stage", {"objectives": []}),
        (
            "run_quasi_isodynamic_objective_optimization",
            "build_quasi_isodynamic_objective_stage",
            {
                "scalar_objectives": [],
                "qi_objectives": [],
                "surfaces": (0.5,),
                "mboz": 4,
                "nboz": 4,
                "nphi": 9,
                "nalpha": 5,
                "n_bounce": 5,
                "include_bounce_endpoints": True,
                "softness": 2.0e-2,
                "width_weight": 1.0,
                "branch_width_weight": 0.5,
                "branch_width_softness": 2.0e-2,
                "profile_weight": 0.1,
                "shuffle_profile_weight": 1.0,
                "shuffle_profile_softness": 2.0e-2,
                "aligned_profile_weight": 0.0,
                "aligned_profile_softness": 2.0e-2,
                "aligned_profile_trap_level": 0.65,
                "aligned_profile_trap_softness": 5.0e-2,
                "phimin": 0.0,
            },
        ),
    ],
)
def test_repeated_and_higher_continuation_stages_use_previous_optimized_input(
    monkeypatch,
    tmp_path,
    runner,
    builder_name,
    extra_kwargs,
):
    import vmec_jax.optimization_workflow as workflow

    build_calls = []
    config_sources = []
    params0_by_stage = []
    nfev_by_stage = []

    class FakeOptimizer:
        def __init__(self, stage_index: int, stage_mode: int, n_params: int):
            self.stage_index = int(stage_index)
            self.stage_mode = int(stage_mode)
            self.n_params = int(n_params)

        def run(self, params0, **kwargs):
            params0_by_stage.append(np.asarray(params0, dtype=float).copy())
            nfev_by_stage.append(int(kwargs["max_nfev"]))
            x = np.full(self.n_params, 100.0 * self.stage_index + self.stage_mode, dtype=float)
            return {
                "x": x,
                "message": "synthetic",
                "_history_dump": _stage_history(float(10 - self.stage_index)),
            }

        def _indata_from_params(self, params):
            encoded = ",".join(f"{value:.1f}" for value in np.asarray(params, dtype=float))
            return InData(
                scalars={"MPOL": 5, "NTOR": 5},
                indexed={"RBC": {(0, 0): 1.0}},
                source_path=f"optimized-stage{self.stage_index}-mode{self.stage_mode}-{encoded}",
            )

    def fake_config_from_indata(indata):
        config_sources.append(indata.source_path)
        return SimpleNamespace(source_path=indata.source_path)

    def fake_build_stage(cfg, indata, *, stage_mode, **_kwargs):
        stage_index = len(build_calls) + 1
        n_params = int(stage_mode)
        build_calls.append((cfg.source_path, indata.source_path, int(stage_mode)))
        specs = [
            BoundaryParamSpec(f"rc{stage_mode}{idx}", "rc", idx, int(stage_mode), idx)
            for idx in range(n_params)
        ]
        return SimpleNamespace(
            mode=int(stage_mode),
            ctx=SimpleNamespace(),
            optimizer=FakeOptimizer(stage_index, int(stage_mode), n_params),
            specs=specs,
            boundary_input=None,
        )

    monkeypatch.setattr(workflow, "config_from_indata", fake_config_from_indata)
    monkeypatch.setattr(workflow, builder_name, fake_build_stage)
    monkeypatch.setattr(workflow, "print_qs_problem_summary", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "print_qs_final_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_stage_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_final_outputs", lambda **_kwargs: None)

    original = InData(
        scalars={"MPOL": 5, "NTOR": 5},
        indexed={"RBC": {(0, 0): 1.0, (2, 0): 0.5}},
        source_path="original-input",
    )

    result = getattr(workflow, runner)(
        cfg=SimpleNamespace(source_path="original-cfg"),
        indata=original,
        stage_modes=[1, 1, 2],
        max_mode=2,
        max_nfev=3,
        continuation_nfev=1,
        method="scipy",
        ftol=1.0e-6,
        gtol=1.0e-6,
        xtol=1.0e-6,
        use_ess=False,
        ess_alpha=0.0,
        output_dir=tmp_path,
        label="synthetic",
        use_mode_continuation=True,
        **extra_kwargs,
    )

    assert build_calls == [
        ("original-cfg", "original-input", 1),
        ("optimized-stage1-mode1-101.0", "optimized-stage1-mode1-101.0", 1),
        ("optimized-stage2-mode1-201.0", "optimized-stage2-mode1-201.0", 2),
    ]
    assert config_sources == [
        "optimized-stage1-mode1-101.0",
        "optimized-stage2-mode1-201.0",
        "optimized-stage3-mode2-302.0,302.0",
    ]
    assert [params.tolist() for params in params0_by_stage] == [[0.0], [0.0], [0.0, 0.0]]
    assert nfev_by_stage == [1, 1, 3]
    assert [record[0] for record in result.stage_records] == [1, 1, 2]


@pytest.mark.parametrize(
    ("runner", "builder_name", "extra_kwargs"),
    [
        ("run_fixed_boundary_objective_optimization", "build_fixed_boundary_objective_stage", {"objectives": []}),
        (
            "run_quasi_isodynamic_objective_optimization",
            "build_quasi_isodynamic_objective_stage",
            {
                "scalar_objectives": [],
                "qi_objectives": [],
                "surfaces": (0.5,),
                "mboz": 4,
                "nboz": 4,
                "nphi": 9,
                "nalpha": 5,
                "n_bounce": 5,
                "include_bounce_endpoints": True,
                "softness": 2.0e-2,
                "width_weight": 1.0,
                "branch_width_weight": 0.5,
                "branch_width_softness": 2.0e-2,
                "profile_weight": 0.1,
                "shuffle_profile_weight": 1.0,
                "shuffle_profile_softness": 2.0e-2,
                "aligned_profile_weight": 0.0,
                "aligned_profile_softness": 2.0e-2,
                "aligned_profile_trap_level": 0.65,
                "aligned_profile_trap_softness": 5.0e-2,
                "phimin": 0.0,
            },
        ),
    ],
)
def test_explicit_lower_mode_stages_get_valid_budget_when_continuation_budget_is_zero(
    monkeypatch,
    tmp_path,
    runner,
    builder_name,
    extra_kwargs,
):
    import vmec_jax.optimization_workflow as workflow

    nfev_by_stage = []

    class FakeOptimizer:
        def __init__(self, n_params: int):
            self.n_params = int(n_params)

        def run(self, params0, **kwargs):
            nfev_by_stage.append(int(kwargs["max_nfev"]))
            return {
                "x": np.ones(self.n_params, dtype=float),
                "message": "synthetic",
                "_history_dump": _stage_history(float(len(nfev_by_stage))),
            }

        def _indata_from_params(self, _params):
            return InData(scalars={"MPOL": 5, "NTOR": 5}, indexed={"RBC": {(0, 0): 1.0}}, source_path="next")

    def fake_build_stage(_cfg, _indata, *, stage_mode, **_kwargs):
        specs = [BoundaryParamSpec(f"rc{stage_mode}{idx}", "rc", idx, int(stage_mode), idx) for idx in range(stage_mode)]
        return SimpleNamespace(ctx=SimpleNamespace(), optimizer=FakeOptimizer(len(specs)), specs=specs)

    monkeypatch.setattr(workflow, "config_from_indata", lambda indata: SimpleNamespace(source_path=indata.source_path))
    monkeypatch.setattr(workflow, builder_name, fake_build_stage)
    monkeypatch.setattr(workflow, "print_qs_problem_summary", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "print_qs_final_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_stage_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_final_outputs", lambda **_kwargs: None)

    result = getattr(workflow, runner)(
        cfg=SimpleNamespace(source_path="original-cfg"),
        indata=InData(scalars={"MPOL": 5, "NTOR": 5}, indexed={"RBC": {(0, 0): 1.0}}, source_path="original"),
        stage_modes=[2, 2, 3],
        max_mode=3,
        max_nfev=4,
        continuation_nfev=0,
        method="scipy",
        ftol=1.0e-6,
        gtol=1.0e-6,
        xtol=1.0e-6,
        use_ess=False,
        ess_alpha=0.0,
        output_dir=tmp_path,
        label="synthetic",
        use_mode_continuation=True,
        **extra_kwargs,
    )

    assert nfev_by_stage == [4, 4, 4]
    assert result.history["max_nfev"] == 12


@pytest.mark.parametrize(
    ("runner", "builder_name", "extra_kwargs"),
    [
        ("run_fixed_boundary_objective_optimization", "build_fixed_boundary_objective_stage", {"objectives": []}),
        (
            "run_quasi_isodynamic_objective_optimization",
            "build_quasi_isodynamic_objective_stage",
            {
                "scalar_objectives": [],
                "qi_objectives": [],
                "surfaces": (0.5,),
                "mboz": 4,
                "nboz": 4,
                "nphi": 9,
                "nalpha": 5,
                "n_bounce": 5,
                "include_bounce_endpoints": True,
                "softness": 2.0e-2,
                "width_weight": 1.0,
                "branch_width_weight": 0.5,
                "branch_width_softness": 2.0e-2,
                "profile_weight": 0.1,
                "shuffle_profile_weight": 1.0,
                "shuffle_profile_softness": 2.0e-2,
                "aligned_profile_weight": 0.0,
                "aligned_profile_softness": 2.0e-2,
                "aligned_profile_trap_level": 0.65,
                "aligned_profile_trap_softness": 5.0e-2,
                "phimin": 0.0,
            },
        ),
    ],
)
def test_final_outputs_use_accepted_result_after_staged_continuation(
    monkeypatch,
    tmp_path,
    runner,
    builder_name,
    extra_kwargs,
):
    import vmec_jax.optimization_workflow as workflow

    calls = []
    optimizers = []

    def history(initial_objective: float, final_objective: float) -> dict:
        return {
            "history": [
                {"objective": float(initial_objective), "wall_time_s": 0.0},
                {"objective": float(final_objective), "wall_time_s": 1.0},
            ],
            "nfev": 2,
            "njev": 1,
            "objective_initial": float(initial_objective),
            "objective_final": float(final_objective),
            "qs_initial": float(initial_objective),
            "qs_final": float(final_objective),
            "aspect_initial": float(initial_objective),
            "aspect_final": float(final_objective),
        }

    class FakeOptimizer:
        def __init__(self, stage_index: int, n_params: int):
            self.stage_index = int(stage_index)
            self.n_params = int(n_params)
            self.accepted_params = np.arange(self.n_params, dtype=float) + 10.0 * self.stage_index
            self.stale_trial_params = np.arange(self.n_params, dtype=float) + 900.0 * self.stage_index
            self.accepted_state = SimpleNamespace(
                label=f"accepted-stage-{self.stage_index}",
                params=self.accepted_params.copy(),
            )
            self.stale_trial_state = SimpleNamespace(
                label=f"stale-trial-stage-{self.stage_index}",
                params=self.stale_trial_params.copy(),
            )

        def run(self, params0, **_kwargs):
            np.testing.assert_allclose(params0, np.zeros(self.n_params, dtype=float))
            return {
                "x": self.accepted_params.copy(),
                "message": "synthetic accepted",
                "_state_initial": SimpleNamespace(label=f"initial-stage-{self.stage_index}"),
                "_state_final": self.accepted_state,
                "_history_dump": history(
                    initial_objective=10.0 - self.stage_index,
                    final_objective=4.0 - self.stage_index,
                ),
            }

        def _indata_from_params(self, params):
            params_arr = np.asarray(params, dtype=float)
            calls.append(("indata", self.stage_index, tuple(params_arr)))
            np.testing.assert_allclose(params_arr, self.accepted_params)
            return InData(
                scalars={"MPOL": 5, "NTOR": 5},
                indexed={"RBC": {(0, 0): 1.0}},
                source_path=f"accepted-stage-{self.stage_index}",
            )

        def save_input(self, path, params):
            params_arr = np.asarray(params, dtype=float)
            calls.append(("input", self.stage_index, path.name, tuple(params_arr)))
            path.write_text("input")

        def save_wout(self, path, params, *, state=None):
            params_arr = np.asarray(params, dtype=float)
            calls.append(
                (
                    "wout",
                    self.stage_index,
                    path.name,
                    tuple(params_arr),
                    None if state is None else state.label,
                )
            )
            path.write_text("wout")

        def save_history(self, path, result):
            hist = result["_history_dump"]
            calls.append(
                (
                    "history",
                    self.stage_index,
                    path.name,
                    float(hist["objective_final"]),
                    float(hist["history"][-1]["objective"]),
                    tuple(hist["stage_boundaries"]),
                )
            )
            path.write_text("history")

    def fake_build_stage(_cfg, _indata, *, stage_mode, **_kwargs):
        stage_index = len(optimizers) + 1
        specs = [
            BoundaryParamSpec(f"rc{stage_mode}{idx}", "rc", idx, int(stage_mode), idx)
            for idx in range(int(stage_mode))
        ]
        optimizer = FakeOptimizer(stage_index, len(specs))
        optimizers.append(optimizer)
        return SimpleNamespace(ctx=SimpleNamespace(), optimizer=optimizer, specs=specs)

    monkeypatch.setattr(workflow, "config_from_indata", lambda indata: SimpleNamespace(source_path=indata.source_path))
    monkeypatch.setattr(workflow, builder_name, fake_build_stage)
    monkeypatch.setattr(workflow, "print_qs_problem_summary", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "print_qs_final_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_stage_artifacts", lambda **_kwargs: None)

    result = getattr(workflow, runner)(
        cfg=SimpleNamespace(source_path="original-cfg"),
        indata=InData(scalars={"MPOL": 5, "NTOR": 5}, indexed={"RBC": {(0, 0): 1.0}}, source_path="original"),
        stage_modes=[1, 2],
        max_mode=2,
        max_nfev=3,
        continuation_nfev=1,
        method="scipy",
        ftol=1.0e-6,
        gtol=1.0e-6,
        xtol=1.0e-6,
        use_ess=False,
        ess_alpha=0.0,
        output_dir=tmp_path,
        label="synthetic",
        use_mode_continuation=True,
        save_final_outputs=True,
        **extra_kwargs,
    )

    final_optimizer = optimizers[-1]
    expected_final = tuple(final_optimizer.accepted_params)
    unexpected_trial = tuple(final_optimizer.stale_trial_params)
    final_input = next(call for call in calls if call[:3] == ("input", 2, "input.final"))
    final_wout = next(call for call in calls if call[:3] == ("wout", 2, "wout_final.nc"))
    final_history = next(call for call in calls if call[:3] == ("history", 2, "history.json"))

    assert final_input[3] == expected_final
    assert final_wout[3] == expected_final
    assert final_wout[4] == "accepted-stage-2"
    assert final_input[3] != unexpected_trial
    assert final_wout[3] != unexpected_trial
    assert final_history[3:] == (2.0, 2.0, (1, 2))
    np.testing.assert_allclose(result.final_result["x"], final_optimizer.accepted_params)
    assert result.history["objective_final"] == 2.0
    assert result.history["history"][-1]["objective"] == 2.0


def test_best_exact_point_is_saved_when_trial_accepted_final_replays_worse(monkeypatch, tmp_path):
    import scipy.optimize
    import vmec_jax.optimization_workflow as workflow

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._solver_device_name = None
    opt._trial_residual_cache = {}
    opt._exact_cache = {}
    opt._exact_state_cache = {}
    opt._exact_residual_cache = {}
    opt._initial_tangent_cache = {}
    opt._last_jacobian_key = [None]
    opt._last_jacobian_residual = None
    opt._static = SimpleNamespace(cfg=SimpleNamespace())
    opt._indata = InData(scalars={}, indexed={}, source_path="synthetic")
    opt._flux = object()
    opt._signgs = 1
    opt._inner_max_iter = 0
    opt._inner_ftol = 0.0
    opt._trial_max_iter = 0
    opt._trial_ftol = 0.0
    opt._post_jacobian_clear = lambda *args, **_kwargs: None
    opt._base_params_vector = lambda: np.zeros(1, dtype=float)
    opt._exact_cache_key = lambda x: tuple(np.asarray(x, dtype=float).round(12))
    opt._aspect_target = 10.0
    opt._aspect_weight = 1.0
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._has_residual_block_metadata = True
    opt._qs_total_from_state_fn = None

    def exact_residual(params):
        value = float(np.asarray(params, dtype=float)[0])
        if np.isclose(value, 1.0):
            return np.asarray([0.2], dtype=float)
        if np.isclose(value, 2.0):
            return np.asarray([2.0], dtype=float)
        return np.asarray([1.0], dtype=float)

    def trial_residual(params):
        value = float(np.asarray(params, dtype=float)[0])
        if np.isclose(value, 2.0):
            return np.asarray([0.01], dtype=float)
        return exact_residual(params)

    def solve_exact(params, return_payload=False):
        value = float(np.asarray(params, dtype=float)[0])
        state = SimpleNamespace(label=f"exact-{value:.1f}", params=np.asarray(params, dtype=float).copy())
        return (state, {}) if return_payload else state

    def jacobian(params):
        residual = exact_residual(params)
        opt._last_jacobian_residual = residual.copy()
        opt._remember_exact_residual(opt._last_jacobian_key[0], residual)
        return np.asarray([[1.0]], dtype=float)

    def fake_least_squares(residuals_y, y0, *, jac, **_kwargs):
        np.testing.assert_allclose(residuals_y(np.asarray([2.0])), [0.01])
        jac(np.asarray([1.0]))
        jac(np.asarray([2.0]))
        return SimpleNamespace(
            x=np.asarray([2.0], dtype=float),
            cost=0.5 * 0.01**2,
            nfev=2,
            njev=2,
            success=True,
            status=1,
            message="synthetic trial-accepted final point",
        )

    opt.residual_fun = exact_residual
    opt.forward_residual_fun = trial_residual
    opt._evaluate_residuals_from_state = lambda state: exact_residual(state.params)
    opt._solve_exact_with_tape = solve_exact
    opt._solve_forward = lambda params, trial=True: solve_exact(params, return_payload=False)
    opt.jacobian_fun = jacobian

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)

    result = opt.run(np.asarray([0.0]), method="scipy", max_nfev=3, verbose=0)

    history = result["_history_dump"]
    objectives = [entry["objective"] for entry in history["history"]]
    np.testing.assert_allclose(result["x"], [1.0])
    assert result["cost"] == pytest.approx(0.5 * 0.2**2)
    assert result["objective"] == pytest.approx(0.2**2)
    assert result["_state_final"].label == "exact-1.0"
    assert history["selected_best_exact_point"] is True
    assert history["rejected_trial_exact_history_count"] == 1
    assert objectives == pytest.approx([1.0, 0.2**2, 0.2**2])
    assert all(next_obj <= obj + 1.0e-14 for obj, next_obj in zip(objectives, objectives[1:]))
    assert 2.0**2 not in objectives

    saved = []

    def fake_save_input(path, params):
        saved.append(("input", path.name, np.asarray(params, dtype=float).copy(), None))

    def fake_save_wout(path, params=None, *, state=None):
        saved.append(("wout", path.name, np.asarray(params, dtype=float).copy(), state))

    def fake_save_history(path, final_result):
        saved.append(("history", path.name, np.asarray(final_result["x"], dtype=float).copy(), None))

    opt.save_input = fake_save_input
    opt.save_wout = fake_save_wout
    opt.save_history = fake_save_history

    workflow.save_qs_final_outputs(
        output_dir=tmp_path,
        stage_records=[(1, opt, np.asarray([0.0]), result)],
        final_optimizer=opt,
        final_result=result,
        label="synthetic",
    )

    final_input = next(item for item in saved if item[:2] == ("input", "input.final"))
    final_wout = next(item for item in saved if item[:2] == ("wout", "wout_final.nc"))
    np.testing.assert_allclose(final_input[2], [1.0])
    np.testing.assert_allclose(final_wout[2], [1.0])
    assert final_wout[3] is result["_state_final"]
    assert final_wout[3].label == "exact-1.0"


def test_scipy_exception_returns_best_exact_point(monkeypatch):
    import scipy.optimize

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._solver_device_name = None
    opt._trial_residual_cache = {}
    opt._exact_cache = {}
    opt._exact_state_cache = {}
    opt._exact_residual_cache = {}
    opt._initial_tangent_cache = {}
    opt._last_jacobian_key = [None]
    opt._last_jacobian_residual = None
    opt._static = SimpleNamespace(cfg=SimpleNamespace())
    opt._indata = InData(scalars={}, indexed={}, source_path="synthetic")
    opt._flux = object()
    opt._signgs = 1
    opt._inner_max_iter = 0
    opt._inner_ftol = 0.0
    opt._trial_max_iter = 0
    opt._trial_ftol = 0.0
    opt._post_jacobian_clear = lambda *args, **_kwargs: None
    opt._base_params_vector = lambda: np.zeros(1, dtype=float)
    opt._exact_cache_key = lambda x: tuple(np.asarray(x, dtype=float).round(12))
    opt._aspect_target = 10.0
    opt._aspect_weight = 1.0
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._has_residual_block_metadata = False
    opt._qs_total_from_state_fn = None

    def exact_residual(params):
        value = float(np.asarray(params, dtype=float)[0])
        return np.asarray([0.1 if np.isclose(value, 1.0) else 1.0], dtype=float)

    def solve_exact(params, return_payload=False):
        value = float(np.asarray(params, dtype=float)[0])
        state = SimpleNamespace(label=f"exact-{value:.1f}", params=np.asarray(params, dtype=float).copy())
        return (state, {}) if return_payload else state

    def jacobian(params):
        residual = exact_residual(params)
        opt._last_jacobian_residual = residual.copy()
        opt._remember_exact_residual(opt._last_jacobian_key[0], residual)
        return np.asarray([[1.0]], dtype=float)

    def fake_least_squares(residuals_y, y0, *, jac, **_kwargs):
        residuals_y(np.asarray([1.0]))
        jac(np.asarray([1.0]))
        raise ValueError("array must not contain infs or NaNs")

    opt.residual_fun = exact_residual
    opt.forward_residual_fun = exact_residual
    opt._evaluate_residuals_from_state = lambda state: exact_residual(state.params)
    opt._solve_exact_with_tape = solve_exact
    opt._solve_forward = lambda params, trial=True: solve_exact(params, return_payload=False)
    opt.jacobian_fun = jacobian

    monkeypatch.setattr(scipy.optimize, "least_squares", fake_least_squares)

    result = opt.run(np.asarray([0.0]), method="scipy", max_nfev=3, verbose=0)

    np.testing.assert_allclose(result["x"], [1.0])
    assert result["success"] is False
    assert result["status"] == -1
    assert "scipy least_squares failed" in result["message"]
    assert result["_state_final"].label == "exact-1.0"
    assert result["_history_dump"]["selected_best_exact_point"] is True
    assert "array must not contain infs or NaNs" in result["_history_dump"]["optimizer_exception"]

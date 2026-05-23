import numpy as np
import jax.numpy as jnp
import pytest
from types import SimpleNamespace

from vmec_jax.namelist import InData
from vmec_jax.boundary import BoundaryCoeffs, boundary_input_from_indata
from vmec_jax.modes import ModeTable, vmec_mode_table
from vmec_jax.optimization import (
    BoundaryParamSpec,
    FixedBoundaryExactOptimizer,
    _indexed_boundary_maps_from_boundary,
    _linear_operator_matrix_arg,
    _linear_operator_vector_arg,
    _pressure_profile_for_static,
    apply_boundary_params,
    boundary_param_names,
    boundary_param_specs,
    create_x_scale,
    extend_boundary_for_max_mode,
    gauss_newton_least_squares,
    lift_boundary_params,
    make_qh_residuals_fn,
    make_qs_residuals_fn,
    parse_surface_list,
    prepare_fixed_boundary_context,
    rebuild_indata_with_resolution,
    smooth_min_abs_iota_residual,
    surface_indices_from_s,
    surface_indices_from_static,
    truncate_indata_boundary_modes,
)
from vmec_jax.optimization_workflow import (
    BoundaryModeLimits,
    describe_boundary_mode_limits,
    interpolate_indata_boundary,
    normalize_boundary_mode_limits,
)
from vmec_jax.state import pack_state


def test_boundary_param_specs_and_apply():
    modes = vmec_mode_table(mpol=2, ntor=1)
    k = modes.K
    boundary = BoundaryCoeffs(
        R_cos=np.linspace(1.0, 2.0, k),
        R_sin=np.zeros(k),
        Z_cos=np.zeros(k),
        Z_sin=np.linspace(0.1, 0.2, k),
    )

    specs = boundary_param_specs(
        boundary,
        modes,
        max_mode=1,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    names = boundary_param_names(specs)

    assert "rc00" not in names
    assert any(name.startswith("rc1") for name in names)
    assert any(name.startswith("zs1") for name in names)

    params = jnp.ones((len(specs),))
    updated = apply_boundary_params(boundary, specs, params)

    # rc00 should remain unchanged
    assert np.isclose(updated.R_cos[0], boundary.R_cos[0])
    # At least one other coefficient should change
    assert not np.allclose(np.asarray(updated.R_cos), np.asarray(boundary.R_cos))


def test_apply_boundary_params_rejects_unknown_kind():
    boundary = BoundaryCoeffs(
        R_cos=np.array([1.0]),
        R_sin=np.array([0.0]),
        Z_cos=np.array([0.0]),
        Z_sin=np.array([0.0]),
    )
    specs = [BoundaryParamSpec("bad", "bad", 0, 0, 0)]

    with pytest.raises(ValueError, match="Unknown boundary parameter kind"):
        apply_boundary_params(boundary, specs, jnp.asarray([1.0]))


def test_boundary_param_specs_cover_lasym_families_and_axis_filter():
    modes = ModeTable(
        m=np.array([0, 1, 1, 2], dtype=int),
        n=np.array([0, 0, 1, 2], dtype=int),
    )
    boundary = BoundaryCoeffs(
        R_cos=np.array([1.0, 0.2, 0.3, 0.4]),
        R_sin=np.array([0.0, 0.02, 0.03, 0.04]),
        Z_cos=np.array([0.0, 0.05, 0.06, 0.07]),
        Z_sin=np.array([0.0, 0.08, 0.09, 0.10]),
    )

    specs = boundary_param_specs(
        boundary,
        modes,
        max_m=1,
        max_n=1,
        include=("rc", "rs", "zc", "zs"),
        include_axis=True,
        fix=("rc00", "rs10"),
    )
    names = boundary_param_names(specs)

    assert "rc00" not in names
    assert "rs10" not in names
    assert "rc10" in names
    assert "zc10" in names
    assert "zs11" in names
    assert all("2" not in name for name in names)


def test_apply_boundary_params_updates_all_boundary_families():
    boundary = BoundaryCoeffs(
        R_cos=np.zeros(4),
        R_sin=np.zeros(4),
        Z_cos=np.zeros(4),
        Z_sin=np.zeros(4),
    )
    specs = [
        BoundaryParamSpec("rc10", "rc", 0, 1, 0),
        BoundaryParamSpec("rs10", "rs", 1, 1, 0),
        BoundaryParamSpec("zc10", "zc", 2, 1, 0),
        BoundaryParamSpec("zs10", "zs", 3, 1, 0),
    ]

    updated = apply_boundary_params(boundary, specs, jnp.asarray([1.0, 2.0, 3.0, 4.0]))

    np.testing.assert_allclose(np.asarray(updated.R_cos), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(updated.R_sin), [0.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(updated.Z_cos), [0.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(np.asarray(updated.Z_sin), [0.0, 0.0, 0.0, 4.0])


def test_surface_indices_from_s():
    s_half = np.array([0.1, 0.3, 0.5, 0.7])
    indices, selected = surface_indices_from_s(s_half, [0.28, 3])
    assert indices == [1, 2]
    np.testing.assert_allclose(selected, np.array([0.3, 0.5]))


def test_parse_surface_list_and_surface_indices_from_static():
    surfaces = parse_surface_list("1, 0.36, 3, 1e-1,")
    assert surfaces == [1, 0.36, 3, 0.1]

    static = SimpleNamespace(s=np.array([0.0, 0.2, 0.5, 1.0]))
    indices, selected = surface_indices_from_static(static, [0.36, 2])

    assert indices == [1, 1]
    np.testing.assert_allclose(selected, np.array([0.35, 0.35]))


def test_lift_boundary_params_maps_shared_names_and_zeros_new_modes():
    source_specs = [
        BoundaryParamSpec("rc10", "rc", 0, 1, 0),
        BoundaryParamSpec("zs10", "zs", 1, 1, 0),
    ]
    target_specs = [
        BoundaryParamSpec("rc10", "rc", 0, 1, 0),
        BoundaryParamSpec("zs10", "zs", 1, 1, 0),
        BoundaryParamSpec("rc21", "rc", 2, 2, 1),
    ]

    lifted = lift_boundary_params(source_specs, np.array([0.25, -0.5]), target_specs)

    np.testing.assert_allclose(lifted, np.array([0.25, -0.5, 0.0]))


def test_mode_continuation_rebuilds_from_previous_stage_input(monkeypatch, tmp_path):
    import vmec_jax.optimization_workflow as workflow

    calls = []
    configs = []

    class FakeOptimizer:
        def __init__(self, stage_index):
            self.stage_index = int(stage_index)

        def run(self, params0, **_kwargs):
            objective = float(self.stage_index)
            return {
                "x": np.asarray([10.0 + self.stage_index], dtype=float),
                "message": "synthetic",
                "_history_dump": {
                    "history": [{"objective": objective, "wall_time_s": objective}],
                    "nfev": 1,
                    "njev": 1,
                    "objective_initial": objective,
                    "objective_final": objective,
                    "qs_initial": objective,
                    "qs_final": objective,
                    "aspect_initial": objective,
                    "aspect_final": objective,
                },
            }

        def _indata_from_params(self, params):
            return InData(
                scalars={"MPOL": 5, "NTOR": 5},
                indexed={"RBC": {(0, 0): 1.0}},
                source_path=f"stage-{self.stage_index}-final-{float(np.asarray(params)[0]):.1f}",
            )

    def fake_config_from_indata(indata):
        configs.append(indata.source_path)
        return SimpleNamespace(source_path=indata.source_path)

    def fake_build_stage(cfg, indata, *, stage_mode, **_kwargs):
        calls.append((cfg.source_path, indata.source_path, int(stage_mode)))
        spec = BoundaryParamSpec(f"rc{stage_mode}0", "rc", 0, int(stage_mode), 0)
        return SimpleNamespace(
            mode=int(stage_mode),
            ctx=SimpleNamespace(),
            optimizer=FakeOptimizer(len(calls)),
            specs=[spec],
            boundary_input=None,
        )

    monkeypatch.setattr(workflow, "config_from_indata", fake_config_from_indata)
    monkeypatch.setattr(workflow, "build_fixed_boundary_objective_stage", fake_build_stage)
    monkeypatch.setattr(workflow, "print_qs_problem_summary", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "print_qs_final_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_stage_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_final_outputs", lambda **_kwargs: None)

    original = InData(
        scalars={"MPOL": 5, "NTOR": 5},
        indexed={"RBC": {(0, 0): 1.0, (2, 0): 0.5}},
        source_path="original-input",
    )

    result = workflow.run_fixed_boundary_objective_optimization(
        cfg=SimpleNamespace(source_path="original-cfg"),
        indata=original,
        objectives=[],
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
    )

    assert calls == [
        ("original-cfg", "original-input", 1),
        ("stage-1-final-11.0", "stage-1-final-11.0", 2),
    ]
    assert configs == ["stage-1-final-11.0", "stage-2-final-12.0"]
    assert [record[2].tolist() for record in result.stage_records] == [[0.0], [0.0]]


def test_qi_mode_continuation_rebuilds_from_previous_stage_input(monkeypatch, tmp_path):
    import vmec_jax.optimization_workflow as workflow

    calls = []
    limits = []

    class FakeOptimizer:
        def __init__(self, stage_index):
            self.stage_index = int(stage_index)

        def run(self, params0, **_kwargs):
            objective = float(self.stage_index)
            return {
                "x": np.asarray([20.0 + self.stage_index], dtype=float),
                "message": "synthetic",
                "_history_dump": {
                    "history": [{"objective": objective, "wall_time_s": objective}],
                    "nfev": 1,
                    "njev": 1,
                    "objective_initial": objective,
                    "objective_final": objective,
                    "qs_initial": objective,
                    "qs_final": objective,
                    "aspect_initial": objective,
                    "aspect_final": objective,
                },
            }

        def _indata_from_params(self, params):
            return InData(
                scalars={"MPOL": 5, "NTOR": 5},
                indexed={"RBC": {(0, 0): 1.0}},
                source_path=f"qi-stage-{self.stage_index}-final-{float(np.asarray(params)[0]):.1f}",
            )

    def fake_config_from_indata(indata):
        return SimpleNamespace(source_path=indata.source_path)

    def fake_build_stage(cfg, indata, *, stage_mode, stage_max_m=None, stage_max_n=None, **_kwargs):
        calls.append((cfg.source_path, indata.source_path, int(stage_mode)))
        limits.append((stage_max_m, stage_max_n))
        spec = BoundaryParamSpec(f"rc{stage_mode}0", "rc", 0, int(stage_mode), 0)
        return SimpleNamespace(
            mode=int(stage_mode),
            ctx=SimpleNamespace(),
            optimizer=FakeOptimizer(len(calls)),
            specs=[spec],
            boundary_input=None,
        )

    monkeypatch.setattr(workflow, "config_from_indata", fake_config_from_indata)
    monkeypatch.setattr(workflow, "build_quasi_isodynamic_objective_stage", fake_build_stage)
    monkeypatch.setattr(workflow, "print_qs_problem_summary", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "print_qs_final_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_stage_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_final_outputs", lambda **_kwargs: None)

    original = InData(
        scalars={"MPOL": 5, "NTOR": 5},
        indexed={"RBC": {(0, 0): 1.0, (2, 0): 0.5}},
        source_path="qi-original-input",
    )

    result = workflow.run_quasi_isodynamic_objective_optimization(
        cfg=SimpleNamespace(source_path="qi-original-cfg"),
        indata=original,
        scalar_objectives=[],
        qi_objectives=[],
        stage_modes=[1, {"mode": 2, "max_m": 1, "max_n": 2, "label": "nfirst"}],
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
        label="synthetic_qi",
        use_mode_continuation=True,
        surfaces=(0.5,),
        mboz=4,
        nboz=4,
        nphi=9,
        nalpha=5,
        n_bounce=5,
        include_bounce_endpoints=True,
        softness=2.0e-2,
        width_weight=1.0,
        branch_width_weight=0.5,
        branch_width_softness=2.0e-2,
        profile_weight=0.1,
        shuffle_profile_weight=1.0,
        shuffle_profile_softness=2.0e-2,
        weighted_shuffle_profile_weight=0.0,
        weighted_shuffle_profile_softness=2.0e-2,
        aligned_profile_weight=0.0,
        aligned_profile_softness=2.0e-2,
        aligned_profile_trap_level=0.65,
        aligned_profile_trap_softness=5.0e-2,
        phimin=0.0,
    )

    assert calls == [
        ("qi-original-cfg", "qi-original-input", 1),
        ("qi-stage-1-final-21.0", "qi-stage-1-final-21.0", 2),
    ]
    assert limits == [(None, None), (1, 2)]
    assert [record[2].tolist() for record in result.stage_records] == [[0.0], [0.0]]


def test_best_exact_point_history_guard_tracks_monotone_objectives():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._best_exact_params = None
    opt._best_exact_residual = None
    opt._best_exact_cost = np.inf

    opt._remember_best_exact_point(np.asarray([0.0]), np.asarray([2.0]))

    assert opt._best_exact_cost == pytest.approx(2.0)
    assert opt._exact_history_accepts(2.0 + 1.0e-12)
    assert not opt._exact_history_accepts(3.0)

    opt._remember_best_exact_point(np.asarray([1.0]), np.asarray([1.0]))

    assert opt._best_exact_cost == pytest.approx(0.5)
    np.testing.assert_allclose(opt._best_exact_params, [1.0])
    np.testing.assert_allclose(opt._best_exact_residual, [1.0])


def test_best_exact_point_ignores_nonfinite_residuals_and_copies_arrays():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._best_exact_params = None
    opt._best_exact_residual = None
    opt._best_exact_cost = np.inf

    params = np.asarray([1.0, 2.0], dtype=float)
    residual = np.asarray([0.25, -0.5], dtype=float)
    opt._remember_best_exact_point(params, residual, cost=0.15625)

    params[:] = 99.0
    residual[:] = 99.0
    np.testing.assert_allclose(opt._best_exact_params, [1.0, 2.0])
    np.testing.assert_allclose(opt._best_exact_residual, [0.25, -0.5])
    assert opt._best_exact_cost == pytest.approx(0.15625)

    opt._remember_best_exact_point(np.asarray([3.0, 4.0]), np.asarray([np.nan, 0.0]), cost=0.0)
    opt._remember_best_exact_point(np.asarray([5.0, 6.0]), np.asarray([0.0, 0.0]), cost=np.nan)

    np.testing.assert_allclose(opt._best_exact_params, [1.0, 2.0])
    np.testing.assert_allclose(opt._best_exact_residual, [0.25, -0.5])
    assert opt._best_exact_cost == pytest.approx(0.15625)


def test_stage_mode_helpers_handle_direct_zero_budget_and_repeated_qi_policy():
    import vmec_jax.optimization_workflow as workflow

    assert workflow.qs_stage_modes(max_mode=3, use_mode_continuation=False, continuation_nfev=5) == [3]
    assert workflow.qs_stage_modes(max_mode=3, use_mode_continuation=True, continuation_nfev=0) == [3]
    assert workflow.qs_stage_modes(max_mode=1, use_mode_continuation=True, continuation_nfev=5) == [1]

    assert workflow.repeated_stage_modes(max_mode=3, use_mode_continuation=False, continuation_nfev=5) == [3]
    assert workflow.repeated_stage_modes(max_mode=3, use_mode_continuation=True, continuation_nfev=0) == [
        3,
        3,
        3,
        3,
        3,
    ]
    assert workflow.repeated_stage_modes(max_mode=3, use_mode_continuation=True, continuation_nfev=5, repeats=4) == [
        3,
        3,
        3,
        3,
    ]

    assert workflow.qs_stage_budget(stage_mode=2, max_mode=3, max_nfev=30, continuation_nfev=7) == 7
    assert workflow.qs_stage_budget(stage_mode=3, max_mode=3, max_nfev=30, continuation_nfev=7) == 30
    assert workflow.qs_stage_budget(stage_mode=2, max_mode=3, max_nfev=30, continuation_nfev=0) == 30
    with pytest.raises(ValueError, match="max_nfev"):
        workflow.qs_stage_budget(stage_mode=3, max_mode=3, max_nfev=0, continuation_nfev=0)


def test_boundary_mode_limits_normalize_anisotropic_stage_descriptors():
    assert normalize_boundary_mode_limits(3) == BoundaryModeLimits(mode=3)
    assert normalize_boundary_mode_limits((1, 4)) == BoundaryModeLimits(mode=4, max_m=1, max_n=4)
    assert normalize_boundary_mode_limits((5, 2, 5)) == BoundaryModeLimits(mode=5, max_m=2, max_n=5)
    assert normalize_boundary_mode_limits({"mode": 6, "max_m": 1, "max_n": 6, "label": "nfirst"}) == (
        BoundaryModeLimits(mode=6, max_m=1, max_n=6, label="nfirst")
    )
    assert describe_boundary_mode_limits({"mode": 6, "max_m": 1, "max_n": 6, "label": "nfirst"}) == (
        "mode06_m01_n06_nfirst"
    )
    with pytest.raises(ValueError, match="mode/max_mode"):
        normalize_boundary_mode_limits({})


def test_create_x_scale_normalizes_lowest_level_and_decays_high_modes():
    specs = [
        BoundaryParamSpec("rc00", "rc", 0, 0, 0),
        BoundaryParamSpec("rc10", "rc", 0, 1, 0),
        BoundaryParamSpec("rc2m2", "rc", 1, 2, -2),
        BoundaryParamSpec("rc33", "rc", 2, 3, 3),
    ]

    np.testing.assert_allclose(create_x_scale(specs, alpha=0.0), np.ones(4))
    np.testing.assert_allclose(
        create_x_scale(specs, alpha=0.5),
        np.array([np.exp(0.5), 1.0, np.exp(-0.5), np.exp(-1.0)]),
    )


def test_smooth_min_abs_iota_residual_is_differentiable_floor():
    import jax

    floor = 0.41
    low = smooth_min_abs_iota_residual(jnp.asarray(0.30), floor, softness=1.0e-3)
    high = smooth_min_abs_iota_residual(jnp.asarray(0.43), floor, softness=1.0e-3)

    assert float(low) > 0.10
    assert float(high) < 1.0e-8

    grad_pos = jax.grad(lambda x: smooth_min_abs_iota_residual(x, floor, softness=1.0e-3))(0.30)
    grad_neg = jax.grad(lambda x: smooth_min_abs_iota_residual(x, floor, softness=1.0e-3))(-0.30)

    assert np.isfinite(float(grad_pos))
    assert np.isfinite(float(grad_neg))
    assert float(grad_pos) < 0.0
    assert float(grad_neg) > 0.0


def _assert_scalar_objective_hook_matches_residual_vector(residuals_fn, state):
    residuals = np.asarray(residuals_fn(state), dtype=float)
    packed = pack_state(state)
    value, cotangent = residuals_fn._state_objective_value_and_cotangent_from_packed(
        packed,
        state.layout,
    )

    expected_value = 0.5 * float(np.dot(residuals, residuals))
    assert float(value) == pytest.approx(expected_value, rel=1.0e-11, abs=1.0e-12)

    cotangent = np.asarray(cotangent)
    assert cotangent.shape == np.asarray(packed).shape
    assert np.all(np.isfinite(cotangent))


def test_qh_residual_factory_scalar_objective_hook_matches_residual_vector(
    load_case_qh_warm_start,
):
    _cfg, indata, static, _boundary, state = load_case_qh_warm_start
    residuals_fn = make_qh_residuals_fn(
        static,
        indata,
        helicity_m=1,
        helicity_n=-1,
        target_aspect=7.0,
        surfaces=[0.5],
        aspect_weight=0.3,
        qs_weight=2.0,
    )

    _assert_scalar_objective_hook_matches_residual_vector(residuals_fn, state)


def test_qs_residual_factory_scalar_objective_hook_sanitizes_iota_cotangent(
    load_case_qh_warm_start,
):
    _cfg, indata, static, _boundary, state = load_case_qh_warm_start
    residuals_fn = make_qs_residuals_fn(
        static,
        indata,
        helicity_m=1,
        helicity_n=-1,
        target_aspect=7.0,
        target_iota=0.41,
        surfaces=[0.5],
        aspect_weight=0.3,
        qs_weight=2.0,
        iota_weight=4.0,
    )

    _assert_scalar_objective_hook_matches_residual_vector(residuals_fn, state)


def test_qs_residual_factory_scalar_objective_hook_handles_iota_floor(
    load_case_qh_warm_start,
):
    _cfg, indata, static, _boundary, state = load_case_qh_warm_start
    residuals_fn = make_qs_residuals_fn(
        static,
        indata,
        helicity_m=0,
        helicity_n=1,
        min_abs_iota=0.41,
        surfaces=[0.5],
        qs_weight=2.0,
        iota_weight=4.0,
    )

    _assert_scalar_objective_hook_matches_residual_vector(residuals_fn, state)


@pytest.mark.full
def test_fixed_boundary_scan_exact_matches_tape_jacobian(load_case_circular_tokamak):
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    _cfg, indata, static, boundary, _state0 = load_case_circular_tokamak
    specs = boundary_param_specs(
        boundary,
        static.modes,
        max_mode=1,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )[:1]
    params = np.asarray([1.0e-4], dtype=float)

    def residuals_fn(state):
        aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
        return jnp.asarray([aspect], dtype=jnp.float64)

    def _optimizer(path: str):
        opt = FixedBoundaryExactOptimizer(
            static,
            indata,
            boundary,
            specs,
            residuals_fn,
            inner_max_iter=1,
            inner_ftol=1.0e-5,
            trial_max_iter=1,
            trial_ftol=1.0e-5,
            solver_device="cpu",
        )
        opt._scan_exact_path = path
        return opt

    tape_opt = _optimizer("tape")
    scan_opt = _optimizer("scan")

    np.testing.assert_allclose(
        scan_opt.residual_fun(params),
        tape_opt.residual_fun(params),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        scan_opt.jacobian_fun(params),
        tape_opt.jacobian_fun(params),
        rtol=1.0e-7,
        atol=1.0e-8,
    )


def test_indexed_boundary_maps_from_boundary_keep_first_duplicate_mode():
    modes = ModeTable(
        m=np.array([0, 1, 1, 1], dtype=int),
        n=np.array([0, 0, 0, -1], dtype=int),
    )
    boundary = BoundaryCoeffs(
        R_cos=np.array([1.0, 2.0, 99.0, 3.0]),
        R_sin=np.array([0.0, 0.2, 9.9, 0.3]),
        Z_cos=np.array([0.0, 0.4, 9.8, 0.5]),
        Z_sin=np.array([0.0, 0.6, 9.7, 0.7]),
    )

    maps = _indexed_boundary_maps_from_boundary(boundary, modes)

    assert maps["RBC"][(0, 0)] == 1.0
    assert maps["RBC"][(0, 1)] == 2.0
    assert maps["RBS"][(0, 1)] == 0.2
    assert maps["ZBC"][(0, 1)] == 0.4
    assert maps["ZBS"][(0, 1)] == 0.6
    assert maps["RBC"][(-1, 1)] == 3.0


def test_rebuild_indata_with_resolution_copies_scalars_without_mutating_original():
    indata = InData(
        scalars={"MPOL": 2, "NTOR": 1, "NFP": 4},
        indexed={"RBC": {(0, 0): 1.0}},
        source_path="input.test",
    )

    rebuilt = rebuild_indata_with_resolution(indata, mpol=6, ntor=5)

    assert indata.scalars["MPOL"] == 2
    assert indata.scalars["NTOR"] == 1
    assert rebuilt.scalars["MPOL"] == 6
    assert rebuilt.scalars["NTOR"] == 5
    assert rebuilt.scalars["NFP"] == 4
    assert rebuilt.indexed == indata.indexed
    assert rebuilt.source_path == indata.source_path


def test_truncate_indata_boundary_modes_projects_inactive_harmonics():
    indata = InData(
        scalars={"MPOL": 6, "NTOR": 6, "NFP": 2},
        indexed={
            "RBC": {(0, 0): 1.0, (1, 0): 0.2, (2, 0): 0.3, (-2, 1): 0.4},
            "ZBS": {(0, 1): 0.5, (1, 1): 0.6, (2, 2): 0.7},
            "AC": {(0,): 1.23},
        },
        source_path="input.nfp2_QI",
    )

    projected = truncate_indata_boundary_modes(indata, max_mode=1)

    assert projected.indexed["RBC"] == {(0, 0): 1.0, (1, 0): 0.2}
    assert projected.indexed["ZBS"] == {(0, 1): 0.5, (1, 1): 0.6}
    assert projected.indexed["AC"] == {(0,): 1.23}
    assert indata.indexed["RBC"][(2, 0)] == 0.3
    assert projected.source_path == indata.source_path


def test_truncate_indata_boundary_modes_none_returns_original():
    indata = InData(
        scalars={"NFP": 2},
        indexed={"RBC": {(2, 0): 0.3}},
        source_path="input.test",
    )

    assert truncate_indata_boundary_modes(indata, max_mode=None) is indata


def test_tiny_target_helicity_seed_terms_survive_active_mode_projection():
    """Document the optimization-time target-helicity seed convention."""

    amplitude = 1.0e-5
    indata = InData(
        scalars={"MPOL": 5, "NTOR": 5, "NFP": 4, "LASYM": False},
        indexed={
            "RBC": {
                (0, 0): 1.0,
                (0, 1): 0.2,
                (-1, 1): amplitude,
                (-2, 2): 9.0 * amplitude,
            },
            "ZBS": {
                (0, 1): 0.2,
                (-1, 1): -amplitude,
                (-2, 2): -9.0 * amplitude,
            },
        },
        source_path="input.minimal_seed_nfp4",
    )

    projected = truncate_indata_boundary_modes(indata, max_mode=1)

    assert projected.indexed["RBC"][(-1, 1)] == pytest.approx(amplitude)
    assert projected.indexed["ZBS"][(-1, 1)] == pytest.approx(-amplitude)
    assert (-2, 2) not in projected.indexed["RBC"]
    assert (-2, 2) not in projected.indexed["ZBS"]

    modes = vmec_mode_table(mpol=5, ntor=5)
    boundary = boundary_input_from_indata(projected, modes)
    target_index = next(
        k for k, (m_i, n_i) in enumerate(zip(modes.m, modes.n)) if int(m_i) == 1 and int(n_i) == -1
    )

    assert boundary.R_cos[target_index] == pytest.approx(amplitude)
    assert boundary.Z_sin[target_index] == pytest.approx(-amplitude)


def test_extend_boundary_for_max_mode_noops_when_resolution_sufficient():
    modes = vmec_mode_table(mpol=6, ntor=6)
    static = SimpleNamespace(modes=modes)
    boundary = BoundaryCoeffs(
        R_cos=np.zeros(modes.K),
        R_sin=np.zeros(modes.K),
        Z_cos=np.zeros(modes.K),
        Z_sin=np.zeros(modes.K),
    )
    indata = InData(scalars={"MPOL": 6, "NTOR": 6}, indexed={}, source_path=None)

    out_indata, out_static, out_boundary = extend_boundary_for_max_mode(
        indata,
        static,
        boundary,
        max_mode=3,
    )

    assert out_indata is indata
    assert out_static is static
    assert out_boundary is boundary


def test_extend_boundary_for_max_mode_rebuilds_resolution(monkeypatch):
    import vmec_jax.boundary as boundary_module
    import vmec_jax.config as config_module
    import vmec_jax.static as static_module

    modes = vmec_mode_table(mpol=2, ntor=1)
    rebuilt_modes = vmec_mode_table(mpol=5, ntor=5)
    static = SimpleNamespace(modes=modes)
    boundary = BoundaryCoeffs(
        R_cos=np.zeros(modes.K),
        R_sin=np.zeros(modes.K),
        Z_cos=np.zeros(modes.K),
        Z_sin=np.zeros(modes.K),
    )
    rebuilt_boundary = BoundaryCoeffs(
        R_cos=np.ones(rebuilt_modes.K),
        R_sin=np.zeros(rebuilt_modes.K),
        Z_cos=np.zeros(rebuilt_modes.K),
        Z_sin=np.zeros(rebuilt_modes.K),
    )
    cfg = SimpleNamespace(name="rebuilt")

    monkeypatch.setattr(config_module, "config_from_indata", lambda _indata: cfg)
    monkeypatch.setattr(static_module, "build_static", lambda _cfg: SimpleNamespace(modes=rebuilt_modes))
    monkeypatch.setattr(boundary_module, "boundary_from_indata", lambda _indata, _modes: rebuilt_boundary)

    indata = InData(
        scalars={"MPOL": 2, "NTOR": 1, "NFP": 2},
        indexed={"RBC": {(0, 0): 1.0}},
        source_path="input.lowres",
    )

    out_indata, out_static, out_boundary = extend_boundary_for_max_mode(
        indata,
        static,
        boundary,
        max_mode=3,
    )

    assert out_indata is not indata
    assert out_indata.scalars["MPOL"] == 5
    assert out_indata.scalars["NTOR"] == 5
    assert indata.scalars["MPOL"] == 2
    assert out_static.modes is rebuilt_modes
    assert out_boundary is rebuilt_boundary


def test_pressure_profile_for_static_defaults_to_zero(monkeypatch):
    import vmec_jax.optimization as opt_module

    static = SimpleNamespace(s=np.array([0.0, 0.5, 1.0]))
    monkeypatch.setattr(opt_module, "eval_profiles", lambda _indata, _s: {})

    pressure = _pressure_profile_for_static(SimpleNamespace(), static)

    np.testing.assert_allclose(np.asarray(pressure), np.zeros(3))


def test_prepare_fixed_boundary_context_uses_shared_precomputations(monkeypatch):
    import vmec_jax.optimization as opt_module

    state = SimpleNamespace(name="state")
    static = SimpleNamespace(s=np.array([0.0, 0.5, 1.0]))
    boundary = object()
    indata = object()
    flux = object()
    booz_inputs = object()

    monkeypatch.setattr(opt_module, "initial_guess_from_boundary", lambda *_args, **_kwargs: state)
    monkeypatch.setattr(opt_module, "eval_geom", lambda *_args, **_kwargs: SimpleNamespace(sqrtg=np.ones((2, 2))))
    monkeypatch.setattr(opt_module, "signgs_from_sqrtg", lambda *_args, **_kwargs: -1)
    monkeypatch.setattr(opt_module, "flux_profiles_from_indata", lambda *_args, **_kwargs: flux)
    monkeypatch.setattr(opt_module, "_pressure_profile_for_static", lambda *_args, **_kwargs: jnp.asarray([1.0, 2.0]))
    monkeypatch.setattr(opt_module, "booz_xform_inputs_from_state", lambda **_kwargs: booz_inputs)

    context = prepare_fixed_boundary_context(
        static=static,
        indata=indata,
        boundary=boundary,
        vmec_project=True,
    )

    assert context.st_guess is state
    assert context.signgs == -1
    assert context.flux is flux
    np.testing.assert_allclose(np.asarray(context.pressure), [1.0, 2.0])
    assert context.booz_inputs is booz_inputs


def test_gauss_newton_least_squares_solves_linear_problem():
    def residual(x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] - 1.0, 2.0 * x[1] - 2.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0, 0.0], dtype=float),
        max_nfev=5,
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
        verbose=0,
    )

    np.testing.assert_allclose(result["x"], np.array([1.0, 1.0]), atol=1e-12, rtol=0.0)
    assert result["success"]
    assert result["objective"] <= 1e-20


def test_gauss_newton_reports_nonfinite_optimality():
    def residual(x):
        return np.array([float(x[0]) - 1.0], dtype=float)

    def jacobian(_x):
        return np.array([[np.nan]], dtype=float)

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0], dtype=float),
        max_nfev=3,
        verbose=0,
    )

    assert not result["success"]
    assert result["message"] == "non-finite optimality encountered"
    assert result["njev"] == 1
    assert np.isfinite(result["cost"])


def test_gauss_newton_reports_line_search_failure():
    def residual(_x):
        return np.array([1.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0]], dtype=float)

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0], dtype=float),
        max_nfev=4,
        verbose=0,
    )

    assert not result["success"]
    assert result["message"] == "line search failed to reduce the objective"
    assert result["cost"] == 0.5


def test_gauss_newton_reports_lstsq_failure(monkeypatch):
    def residual(_x):
        return np.array([1.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0]], dtype=float)

    def raise_linalg(*_args, **_kwargs):
        raise np.linalg.LinAlgError("synthetic failure")

    monkeypatch.setattr(np.linalg, "lstsq", raise_linalg)

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0], dtype=float),
        max_nfev=3,
        verbose=0,
    )

    assert not result["success"]
    assert result["message"] == "linear least-squares solve failed"


def test_gauss_newton_reports_nonfinite_step(monkeypatch):
    def residual(_x):
        return np.array([1.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0]], dtype=float)

    monkeypatch.setattr(np.linalg, "lstsq", lambda *_args, **_kwargs: (np.array([np.nan]), None, None, None))

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0], dtype=float),
        max_nfev=3,
        verbose=0,
    )

    assert not result["success"]
    assert result["message"] == "non-finite Gauss-Newton step encountered"


def test_gauss_newton_reports_xtol_termination_for_zero_step(monkeypatch):
    def residual(_x):
        return np.array([1.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0]], dtype=float)

    monkeypatch.setattr(np.linalg, "lstsq", lambda *_args, **_kwargs: (np.array([0.0]), None, None, None))

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0], dtype=float),
        max_nfev=3,
        gtol=0.0,
        xtol=1.0e-12,
        verbose=0,
    )

    assert result["success"]
    assert result["message"] == "`xtol` termination condition is satisfied."
    assert result["step_norm"] == 0.0


def test_gauss_newton_post_jacobian_callback():
    """post_jacobian_callback is called once per jacobian evaluation."""
    call_counts = [0]

    def residual(x):
        return np.array([float(x[0]) - 1.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0]], dtype=float)

    def on_jac():
        call_counts[0] += 1

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0], dtype=float),
        max_nfev=5,
        post_jacobian_callback=on_jac,
        verbose=0,
    )

    assert result["success"]
    assert call_counts[0] == result["njev"]


def test_gauss_newton_exact_residual_after_jacobian():
    """exact_residual_after_jacobian_fun replaces the residual used for gradient."""
    # Set up a problem where forward_residual_fun gives a deliberately noisy
    # residual, but exact_residual_after_jacobian_fun provides the correct one.
    # The optimizer should still converge because the exact residual is used
    # for the gradient computation after each Jacobian call.
    rng = np.random.default_rng(42)
    noise_scale = 0.5

    def residual(x):
        return np.array([float(x[0]) - 1.0], dtype=float)

    def noisy_residual(x):
        return residual(x) + noise_scale * rng.standard_normal(1)

    # Track the most recent jacobian x so we can return the exact residual.
    last_x = [None]

    def jacobian(x):
        last_x[0] = float(x[0])
        return np.array([[1.0]], dtype=float)

    def exact_residual():
        if last_x[0] is None:
            return None
        return np.array([last_x[0] - 1.0], dtype=float)

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0], dtype=float),
        forward_residual_fun=noisy_residual,
        exact_residual_after_jacobian_fun=exact_residual,
        max_nfev=20,
        verbose=0,
    )

    assert result["success"]
    np.testing.assert_allclose(result["x"], np.array([1.0]), atol=1e-3, rtol=0.0)


def test_fixed_boundary_optimizer_exact_residual_reuses_jacobian_primal():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._last_jacobian_residual = np.array([3.0, 4.0])
    opt._last_jacobian_key = [b"accepted"]
    opt._exact_cache = {b"accepted": (object(), object())}

    def fail_residual(_state):
        raise AssertionError("cached Jacobian residual should be reused")

    opt._residuals_fn = fail_residual

    np.testing.assert_allclose(opt._exact_residual_after_jacobian(), [3.0, 4.0])


def test_lbfgs_adjoint_respects_scalar_evaluation_budget(monkeypatch):
    import scipy.optimize

    state = object()
    last_x = [np.array([0.0])]

    def fake_minimize(fun, y0, *, jac, method, bounds, options):
        assert jac is True
        assert method == "L-BFGS-B"
        assert bounds == [(-0.01, 0.01)]
        assert options["maxfun"] == 2
        fun(np.asarray(y0, dtype=float))
        fun(np.asarray([0.4], dtype=float))
        # The third line-search probe must be blocked by vmec_jax's hard
        # budget guard, not left to SciPy's soft maxfun accounting.
        fun(np.asarray([0.8], dtype=float))
        raise AssertionError("budget guard did not stop L-BFGS-B")

    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)
    monkeypatch.setattr(
        "vmec_jax.wout.equilibrium_aspect_ratio_from_state",
        lambda **_kwargs: 1.0,
    )

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._trial_residual_cache = {}
    opt._exact_cache = {}
    opt._static = SimpleNamespace()
    opt._solver_device_name = None
    opt._inner_max_iter = 0
    opt._inner_ftol = 0.0
    opt._trial_max_iter = 0
    opt._trial_ftol = 0.0
    opt._post_jacobian_clear = lambda *args, **kwargs: None
    opt._profile_dump = lambda: {}
    opt._cached_exact_state = lambda _x: None
    opt._base_params_vector = lambda: np.zeros(1)
    opt._exact_cache_key = lambda x: tuple(np.asarray(x, dtype=float).round(12))
    opt._qs_total_from_state = lambda _state, res: float(np.dot(res, res))

    def residual_fun(x):
        return np.asarray([float(np.asarray(x)[0]) - 1.0], dtype=float)

    def solve_exact(x, return_payload=False):
        last_x[0] = np.asarray(x, dtype=float)
        return (state, {}) if return_payload else state

    opt.residual_fun = residual_fun
    opt._evaluate_residuals_from_state = lambda _state: residual_fun(last_x[0])
    opt._solve_exact_with_tape = solve_exact
    opt._solve_forward = lambda x, trial=True: solve_exact(x, return_payload=False)

    objective_calls = []

    def objective_and_gradient(x):
        x = np.asarray(x, dtype=float)
        objective_calls.append(float(x[0]))
        residual = float(x[0]) - 1.0
        return 0.5 * residual * residual, np.asarray([residual], dtype=float)

    opt.objective_and_gradient_fun = objective_and_gradient

    result = opt.run(np.asarray([0.0]), method="lbfgs_adjoint", max_nfev=2, verbose=0)

    assert objective_calls == [0.0, 0.4]
    assert result["nfev"] == 2
    assert result["njev"] == 2
    assert not result["success"]
    assert result["message"] == "maximum number of scalar objective evaluations is exceeded"
    np.testing.assert_allclose(result["x"], np.asarray([0.4]))
    assert result["_history_dump"]["objective_final"] == pytest.approx(0.36)


def test_scalar_trust_improves_quadratic_with_hard_budget(monkeypatch):
    state = object()
    last_x = [np.array([0.0])]

    monkeypatch.setattr(
        "vmec_jax.wout.equilibrium_aspect_ratio_from_state",
        lambda **_kwargs: 1.0,
    )

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._trial_residual_cache = {}
    opt._exact_cache = {}
    opt._static = SimpleNamespace()
    opt._solver_device_name = None
    opt._inner_max_iter = 0
    opt._inner_ftol = 0.0
    opt._trial_max_iter = 0
    opt._trial_ftol = 0.0
    opt._post_jacobian_clear = lambda *args, **kwargs: None
    opt._profile_dump = lambda: {}
    opt._cached_exact_state = lambda _x: None
    opt._base_params_vector = lambda: np.zeros(1)
    opt._exact_cache_key = lambda x: tuple(np.asarray(x, dtype=float).round(12))
    opt._qs_total_from_state = lambda _state, res: float(np.dot(res, res))

    def residual_fun(x):
        return np.asarray([float(np.asarray(x)[0]) - 1.0], dtype=float)

    def solve_exact(x, return_payload=False):
        last_x[0] = np.asarray(x, dtype=float)
        return (state, {}) if return_payload else state

    opt.residual_fun = residual_fun
    opt._evaluate_residuals_from_state = lambda _state: residual_fun(last_x[0])
    opt._solve_exact_with_tape = solve_exact
    opt._solve_forward = lambda x, trial=True: solve_exact(x, return_payload=False)

    objective_calls = []

    def objective_and_gradient(x):
        x = np.asarray(x, dtype=float)
        objective_calls.append(float(x[0]))
        residual = float(x[0]) - 1.0
        return 0.5 * residual * residual, np.asarray([residual], dtype=float)

    opt.objective_and_gradient_fun = objective_and_gradient

    result = opt.run(
        np.asarray([0.0]),
        method="scalar_trust",
        max_nfev=5,
        scalar_step_bound=0.25,
        ftol=0.0,
        gtol=1e-12,
        verbose=0,
    )

    assert result["nfev"] <= 5
    assert result["njev"] == result["nfev"]
    assert result["cost"] < 1e-12
    assert result["nit"] == 4
    np.testing.assert_allclose(result["x"], np.asarray([1.0]), atol=1e-12, rtol=0.0)
    assert objective_calls == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert result["_history_dump"]["method"] == "scalar_trust"
    assert result["_history_dump"]["scalar_step_bound"] == pytest.approx(0.25)


def test_qs_total_prefers_metadata_function_over_residual_vector():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._n_qs = None
    opt._n_non_qs = 1

    opt._qs_total_from_state_fn = lambda _state: 7.0

    assert opt._qs_total_from_state(object(), np.array([10.0, 2.0, 3.0])) == 7.0


def test_qs_total_uses_supplied_residual_vector_without_metadata_function():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._n_qs = None
    opt._n_non_qs = 1
    opt._qs_total_from_state_fn = None

    assert opt._qs_total_from_state(object(), np.array([10.0, 2.0, 3.0])) == 13.0


def test_evaluate_residuals_from_state_uses_cached_eval_function():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._residuals_fn = lambda _state: np.array([-1.0])
    opt._residuals_eval_fn = lambda state: jnp.asarray([2.0, 3.0])

    np.testing.assert_allclose(opt._evaluate_residuals_from_state(object()), [2.0, 3.0])


def test_initial_tangent_cache_key_tracks_vmec_flip_branch():
    modes = vmec_mode_table(mpol=2, ntor=0)
    idx_m1 = int(np.nonzero(np.asarray(modes.m) == 1)[0][0])
    r_cos = np.zeros(modes.K)
    z_sin = np.zeros(modes.K)
    r_cos[idx_m1] = 1.0
    z_sin[idx_m1] = 1.0
    boundary = BoundaryCoeffs(
        R_cos=r_cos,
        R_sin=np.zeros(modes.K),
        Z_cos=np.zeros(modes.K),
        Z_sin=z_sin,
    )

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._boundary = boundary
    opt._boundary_input = None
    opt._static = SimpleNamespace(
        modes=modes,
        cfg=SimpleNamespace(lasym=False, ns=8),
    )
    opt._specs = [BoundaryParamSpec("rc10", "rc", idx_m1, 1, 0)]

    no_flip = opt._initial_tangent_cache_key(np.array([0.0]))
    flip = opt._initial_tangent_cache_key(np.array([-2.0]))

    assert no_flip is not None
    assert flip is not None
    assert no_flip != flip
    assert no_flip[1] is False
    assert flip[1] is True


def test_state_tangent_columns_cache_hit_skips_initial_linearization_setup(monkeypatch):
    import vmec_jax.discrete_adjoint as discrete_adjoint

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._layout = SimpleNamespace(size=3)
    opt._static = object()
    opt._profile = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._initial_tangent_cache = {"cached": jnp.asarray([[1.0, 2.0, 3.0]])}
    opt._initial_tangent_cache_key = lambda _params: "cached"
    opt._solve_exact_with_tape = lambda _params, return_payload=False: (
        "state",
        {
            "tape": "tape",
            # This would fail if the cache-hit path still converted the axis
            # override while setting up an unused initial-state linearization.
            "axis_override": {"bad": object()},
        },
    )
    opt._boundary_from_params = lambda _params: (_ for _ in ()).throw(
        AssertionError("cache hit should not rebuild the boundary")
    )
    opt._lasym_replay_column_chunk = lambda _n_params: None

    calls = []

    def fake_replay_columns(*, tape, static, initial_tangents, rebuild_preconditioner, column_chunk):
        calls.append((tape, static, bool(rebuild_preconditioner), column_chunk))
        return jnp.asarray(initial_tangents) + 1.0

    monkeypatch.setattr(
        discrete_adjoint,
        "checkpoint_tape_state_jvp_columns",
        fake_replay_columns,
    )

    state, final_tangents = opt._state_and_tangent_columns(
        np.asarray([0.0]),
        profile_prefix="jacobian",
    )

    assert state == "state"
    np.testing.assert_allclose(np.asarray(final_tangents), [[2.0, 3.0, 4.0]])
    assert calls == [("tape", opt._static, True, None)]
    assert opt._profile["jacobian_initial_tangents_cache_hit"]["count"] == 1


def test_state_tangent_columns_requests_jvp_only_exact_tape(monkeypatch):
    import vmec_jax.discrete_adjoint as discrete_adjoint

    monkeypatch.setenv("VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE", "1")
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._layout = SimpleNamespace(size=3)
    opt._static = object()
    opt._profile = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._initial_tangent_cache = {"cached": jnp.asarray([[1.0, 2.0, 3.0]])}
    opt._initial_tangent_cache_key = lambda _params: "cached"
    opt._lasym_replay_column_chunk = lambda _n_params: None
    solve_calls = []

    def fake_solve(_params, *, return_payload=False, jvp_only=False):
        solve_calls.append((bool(return_payload), bool(jvp_only)))
        return "state", {"tape": "jvp-tape", "axis_override": {}}

    opt._solve_exact_with_tape = fake_solve
    monkeypatch.setattr(
        discrete_adjoint,
        "checkpoint_tape_state_jvp_columns",
        lambda **kwargs: kwargs["initial_tangents"],
    )

    state, final_tangents = opt._state_and_tangent_columns(
        np.asarray([0.0]),
        profile_prefix="jacobian",
    )

    assert state == "state"
    np.testing.assert_allclose(np.asarray(final_tangents), [[1.0, 2.0, 3.0]])
    assert solve_calls == [(True, True)]


def test_state_tangent_columns_uses_full_exact_tape_by_default(monkeypatch):
    import vmec_jax.discrete_adjoint as discrete_adjoint

    monkeypatch.delenv("VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE", raising=False)
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._layout = SimpleNamespace(size=3)
    opt._static = object()
    opt._profile = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._initial_tangent_cache = {"cached": jnp.asarray([[1.0, 2.0, 3.0]])}
    opt._initial_tangent_cache_key = lambda _params: "cached"
    opt._lasym_replay_column_chunk = lambda _n_params: None
    solve_calls = []

    def fake_solve(_params, *, return_payload=False, jvp_only=False):
        solve_calls.append((bool(return_payload), bool(jvp_only)))
        return "state", {"tape": "full-tape", "axis_override": {}}

    opt._solve_exact_with_tape = fake_solve
    monkeypatch.setattr(
        discrete_adjoint,
        "checkpoint_tape_state_jvp_columns",
        lambda **kwargs: kwargs["initial_tangents"],
    )

    state, final_tangents = opt._state_and_tangent_columns(
        np.asarray([0.0]),
        profile_prefix="jacobian",
    )

    assert state == "state"
    np.testing.assert_allclose(np.asarray(final_tangents), [[1.0, 2.0, 3.0]])
    assert solve_calls == [(True, False)]


def test_gradient_callback_reuses_cached_initial_tangents(monkeypatch):
    import vmec_jax.discrete_adjoint as discrete_adjoint

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._layout = SimpleNamespace(size=3)
    opt._static = object()
    opt._profile = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._initial_tangent_cache = {
        "cached": jnp.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=jnp.float64,
        )
    }
    opt._initial_tangent_cache_key = lambda _params: "cached"
    opt._solve_exact_with_tape = lambda _params, return_payload=False: (
        "state",
        {
            "tape": "tape",
            # This would fail if the gradient path rebuilt an unused
            # initial-state VJP instead of using the cached tangent map.
            "axis_override": {"bad": object()},
        },
    )
    opt._boundary_from_params = lambda _params: (_ for _ in ()).throw(
        AssertionError("cache hit should not rebuild the boundary")
    )

    def residuals_fn(_state):
        raise AssertionError("objective cotangent hook should avoid residual VJP")

    def objective_value_and_cotangent_from_packed(packed_state, layout):
        np.testing.assert_allclose(np.asarray(packed_state), [7.0, 8.0, 9.0])
        assert layout is opt._layout
        return jnp.asarray(1.25, dtype=jnp.float64), jnp.asarray([10.0, 11.0, 12.0], dtype=jnp.float64)

    residuals_fn._state_objective_value_and_cotangent_from_packed = objective_value_and_cotangent_from_packed
    opt._residuals_fn = residuals_fn

    monkeypatch.setattr("vmec_jax.state.pack_state", lambda _state: jnp.asarray([7.0, 8.0, 9.0]))

    vjp_calls = []

    def fake_checkpoint_vjp(*, tape, static, final_cotangent, rebuild_preconditioner):
        vjp_calls.append((tape, static, bool(rebuild_preconditioner), np.asarray(final_cotangent)))
        return jnp.asarray([0.5, 1.0, 2.0], dtype=jnp.float64)

    monkeypatch.setattr(discrete_adjoint, "checkpoint_tape_state_vjp", fake_checkpoint_vjp)

    cost, grad = opt.objective_and_gradient_fun(np.asarray([0.0, 0.0]))

    assert cost == pytest.approx(1.25)
    np.testing.assert_allclose(grad, [8.5, 19.0])
    assert len(vjp_calls) == 1
    np.testing.assert_allclose(vjp_calls[0][3], [10.0, 11.0, 12.0])
    assert opt._profile["gradient_initial_tangents_cache_hit"]["count"] == 1


def test_tape_jacobian_remembers_residual_under_parameter_cache_key(monkeypatch):
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._scan_exact_path = "tape"
    opt._layout = SimpleNamespace(size=2)
    opt._discrete_jacobian_helper_cache = {}
    opt._exact_residual_cache = {}
    opt._last_jacobian_key = [b"accepted"]
    opt._last_jacobian_residual = None
    opt._profile = {}
    opt._exact_cache_key = lambda _params: b"accepted"
    opt._state_and_tangent_columns = lambda _params, profile_prefix: (
        SimpleNamespace(),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
    )
    opt._residuals_fn = lambda state: jnp.asarray(
        [state.foo + 1.0, 2.0 * state.bar],
        dtype=jnp.float64,
    )

    monkeypatch.setattr(
        "vmec_jax.state.pack_state",
        lambda _state: jnp.asarray([3.0, 4.0], dtype=jnp.float64),
    )
    monkeypatch.setattr(
        "vmec_jax.state.unpack_state",
        lambda packed, _layout: SimpleNamespace(foo=packed[0], bar=packed[1]),
    )

    jac = opt.jacobian_fun(np.asarray([0.1, 0.2], dtype=float))

    np.testing.assert_allclose(jac, np.asarray([[1.0, 0.0], [0.0, 2.0]]))
    np.testing.assert_allclose(opt._exact_residual_cache[b"accepted"], [4.0, 8.0])
    assert all(not isinstance(key, tuple) for key in opt._exact_residual_cache)


def test_tape_jacobian_cache_skips_same_point_replay(monkeypatch):
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._scan_exact_path = "tape"
    opt._layout = SimpleNamespace(size=2)
    opt._discrete_jacobian_helper_cache = {}
    opt._exact_residual_cache = {}
    opt._exact_jacobian_cache = {}
    opt._last_jacobian_key = [b"accepted"]
    opt._last_jacobian_residual = None
    opt._profile = {}
    opt._exact_cache_key = lambda _params: b"accepted"
    replay_calls = []

    def fake_state_and_tangent_columns(_params, profile_prefix):
        replay_calls.append(profile_prefix)
        return (
            SimpleNamespace(),
            jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        )

    opt._state_and_tangent_columns = fake_state_and_tangent_columns
    opt._residuals_fn = lambda state: jnp.asarray(
        [state.foo + 1.0, 2.0 * state.bar],
        dtype=jnp.float64,
    )

    monkeypatch.setattr(
        "vmec_jax.state.pack_state",
        lambda _state: jnp.asarray([3.0, 4.0], dtype=jnp.float64),
    )
    monkeypatch.setattr(
        "vmec_jax.state.unpack_state",
        lambda packed, _layout: SimpleNamespace(foo=packed[0], bar=packed[1]),
    )

    params = np.asarray([0.1, 0.2], dtype=float)
    jac1 = opt.jacobian_fun(params)
    jac2 = opt.jacobian_fun(params)
    jac2[0, 0] = 99.0
    jac3 = opt.jacobian_fun(params)

    assert replay_calls == ["jacobian"]
    np.testing.assert_allclose(jac1, np.asarray([[1.0, 0.0], [0.0, 2.0]]))
    np.testing.assert_allclose(jac3, np.asarray([[1.0, 0.0], [0.0, 2.0]]))
    np.testing.assert_allclose(opt._last_jacobian_residual, [4.0, 8.0])
    assert opt._profile["jacobian_cache_hit"]["count"] == 2
    assert opt._last_jacobian_source == "jacobian_cache_hit"


def test_gauss_newton_damped_fallback_recovers_from_oversized_step():
    """Damping should rescue cases where the raw GN step is unusably large."""

    def residual(x):
        return np.array([float(x[0]) - 1.0], dtype=float)

    def poor_scaled_jacobian(_x):
        return np.array([[1.0e-6]], dtype=float)

    result = gauss_newton_least_squares(
        residual,
        poor_scaled_jacobian,
        np.array([0.0], dtype=float),
        max_nfev=20,
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
        verbose=0,
    )

    assert result["success"]
    assert result["cost"] < 1e-10
    np.testing.assert_allclose(result["x"], np.array([1.0]), atol=1e-5, rtol=0.0)


def test_gauss_newton_helper_matches_scipy_linear_problem():
    """The standalone SciPy path should solve the same linear least-squares problem."""

    try:
        from scipy.optimize import least_squares
    except Exception:  # pragma: no cover - optional dependency
        return

    def residual(x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] - 1.0, 2.0 * x[1] - 2.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)

    result = least_squares(
        residual,
        np.array([0.0, 0.0], dtype=float),
        jac=jacobian,
        method="trf",
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )

    np.testing.assert_allclose(result.x, np.array([1.0, 1.0]), atol=1e-12, rtol=0.0)
    assert result.success


def test_fixed_boundary_optimizer_read_last_array_prefers_scalar_array_value():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._indata = InData(
        scalars={
            "NITER_ARRAY": 1500,
            "NITER": 10000,
            "FTOL_ARRAY": 1e-13,
            "FTOL": 1e-8,
        },
        indexed={},
        source_path=None,
    )

    assert opt._read_last_array("NITER_ARRAY", "NITER", 42, int) == 1500
    assert opt._read_last_array("FTOL_ARRAY", "FTOL", 1e-6, float) == 1e-13


def test_fixed_boundary_optimizer_read_last_array_handles_sequence_values():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._indata = InData(
        scalars={
            "NITER_ARRAY": [50, 75, 125],
            "NITER": 10000,
        },
        indexed={},
        source_path=None,
    )

    assert opt._read_last_array("NITER_ARRAY", "NITER", 42, int) == 125


def test_fixed_boundary_optimizer_solver_device_inherits_by_default():
    opt = object.__new__(FixedBoundaryExactOptimizer)

    assert opt._resolve_solver_device(None) is None
    assert opt._resolve_solver_device("auto") is None
    assert opt._resolve_solver_device("default") is None
    assert opt._resolve_solver_device("cpu") == "cpu"
    assert opt._resolve_solver_device("gpu") == "gpu"


def test_fixed_boundary_optimizer_trial_scan_default_and_env_override(monkeypatch):
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = "cpu"

    monkeypatch.delenv("VMEC_JAX_OPT_TRIAL_SCAN", raising=False)
    assert opt._use_scan_for_trial_solves() is True

    opt._solver_device_name = "gpu"
    assert opt._use_scan_for_trial_solves() is False

    monkeypatch.setenv("VMEC_JAX_OPT_TRIAL_SCAN", "0")
    assert opt._use_scan_for_trial_solves() is False

    monkeypatch.setenv("VMEC_JAX_OPT_TRIAL_SCAN", "1")
    assert opt._use_scan_for_trial_solves() is True


def test_fixed_boundary_optimizer_exact_path_is_device_aware(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_OPT_EXACT_PATH", raising=False)

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = "cpu"
    assert opt._select_exact_path() == "tape"

    opt._solver_device_name = "gpu"
    assert opt._select_exact_path() == "tape"

    monkeypatch.setenv("VMEC_JAX_OPT_EXACT_PATH", "tape")
    assert opt._select_exact_path() == "tape"

    monkeypatch.setenv("VMEC_JAX_OPT_EXACT_PATH", "scan")
    assert opt._select_exact_path() == "scan"


def test_solver_device_context_wraps_callback_once_and_restores_flag(monkeypatch):
    import vmec_jax._compat as compat

    events = []

    class FakeDeviceContext:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            events.append(("enter", self.device))
            return self

        def __exit__(self, *_exc):
            events.append(("exit", self.device))
            return False

    fake_jax = SimpleNamespace(
        devices=lambda name: [f"{name}:0"],
        default_device=lambda device: FakeDeviceContext(device),
    )
    monkeypatch.setattr(compat, "jax", fake_jax)

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = "gpu"
    opt._inside_solver_device_context = False

    def callback(value):
        events.append(("callback", opt._inside_solver_device_context))
        return opt._run_in_solver_device_context(
            lambda nested: ("nested", opt._inside_solver_device_context, nested),
            value + 1,
        )

    assert opt._run_in_solver_device_context(callback, 2) == ("nested", True, 3)
    assert events == [("enter", "gpu:0"), ("callback", True), ("exit", "gpu:0")]
    assert opt._inside_solver_device_context is False


def test_scan_exact_helpers_defer_to_solver_device_context_when_outside():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = "cpu"
    opt._inside_solver_device_context = False
    calls = []

    def run_in_context(fn, *args, **kwargs):
        calls.append((fn.__name__, args, kwargs))
        return f"wrapped:{fn.__name__}"

    opt._run_in_solver_device_context = run_in_context

    assert opt._scan_exact_helpers() == "wrapped:_scan_exact_helpers"
    assert opt._solve_scan_exact_state(np.asarray([1.0])) == "wrapped:_solve_scan_exact_state"
    assert calls[0] == ("_scan_exact_helpers", (), {})
    assert calls[1][0] == "_solve_scan_exact_state"
    np.testing.assert_allclose(calls[1][1][0], [1.0])
    assert calls[1][2] == {}


def test_linear_operator_argument_helpers_validate_vectors_and_matrices():
    np.testing.assert_allclose(
        _linear_operator_vector_arg(np.asarray([[1.0], [2.0]]), size=2, name="direction"),
        [1.0, 2.0],
    )
    np.testing.assert_allclose(
        _linear_operator_matrix_arg(np.arange(6.0), rows=2, name="directions"),
        np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
    )

    with pytest.raises(ValueError, match="direction expected 3 entries, got 2"):
        _linear_operator_vector_arg(np.asarray([1.0, 2.0]), size=3, name="direction")
    with pytest.raises(ValueError, match="directions with 5 entries cannot be reshaped to 2 rows"):
        _linear_operator_matrix_arg(np.arange(5.0), rows=2, name="directions")
    with pytest.raises(ValueError, match="directions expected 3 rows, got 2"):
        _linear_operator_matrix_arg(np.ones((2, 2)), rows=3, name="directions")


def test_residual_linear_operator_matvec_and_matmat_are_shape_checked(monkeypatch):
    pytest.importorskip("scipy.sparse.linalg")
    import vmec_jax.discrete_adjoint as discrete_adjoint
    import vmec_jax.init_guess as init_guess
    import vmec_jax.state as state_module

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._layout = SimpleNamespace(size=2)
    opt._static = object()
    opt._indata = object()
    opt._profile = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._exact_residual_cache = {}
    opt._exact_cache_key = lambda _params: b"operator"
    opt._boundary_from_params = lambda params: params
    opt._lasym_replay_column_chunk = lambda _n_params: None
    opt._solve_exact_with_tape = lambda _params, return_payload=False: (
        jnp.asarray([3.0, 4.0], dtype=jnp.float64),
        {"tape": "tape", "axis_override": {}},
    )
    opt._residuals_fn = lambda state: jnp.asarray(
        [state[0] + state[1], 2.0 * state[0] - state[1]],
        dtype=jnp.float64,
    )

    monkeypatch.setattr(init_guess, "initial_guess_from_boundary", lambda _static, boundary, _indata, **_kwargs: boundary)
    monkeypatch.setattr(state_module, "pack_state", lambda state: jnp.asarray(state, dtype=jnp.float64))
    monkeypatch.setattr(state_module, "unpack_state", lambda packed, _layout: packed)
    monkeypatch.setattr(
        discrete_adjoint,
        "checkpoint_tape_state_jvp",
        lambda *, tape, static, initial_tangent, rebuild_preconditioner: initial_tangent * 3.0,
    )
    monkeypatch.setattr(
        discrete_adjoint,
        "checkpoint_tape_state_jvp_columns",
        lambda *, tape, static, initial_tangents, rebuild_preconditioner, column_chunk: initial_tangents * 3.0,
    )

    op = opt.residual_linear_operator(np.asarray([0.0, 0.0]))

    assert op.shape == (2, 2)
    np.testing.assert_allclose(op.matvec(np.asarray([1.0, 2.0])), [9.0, 0.0])
    np.testing.assert_allclose(op.matmat(np.asarray([[1.0, 0.0], [0.0, 1.0]])), [[3.0, 3.0], [6.0, -3.0]])
    np.testing.assert_allclose(opt._exact_residual_cache[b"operator"], [7.0, 2.0])
    with pytest.raises(ValueError, match="matvec direction expected 2 entries, got 3"):
        op._matvec(np.ones(3))
    with pytest.raises(ValueError, match="matmat directions expected 2 rows, got 3"):
        op._matmat(np.ones((3, 1)))


def test_residual_linear_operator_reuses_cached_initial_tangents(monkeypatch):
    pytest.importorskip("scipy.sparse.linalg")
    import vmec_jax.discrete_adjoint as discrete_adjoint
    import vmec_jax.init_guess as init_guess
    import vmec_jax.state as state_module

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._layout = SimpleNamespace(size=3)
    opt._static = object()
    opt._indata = object()
    opt._profile = {}
    opt._discrete_jacobian_helper_cache = {}
    opt._exact_residual_cache = {}
    opt._exact_cache_key = lambda _params: b"operator"
    opt._initial_tangent_cache_key = lambda _params: "axis-branch"
    opt._initial_tangent_cache = {
        "axis-branch": jnp.asarray(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 4.0],
            ],
            dtype=jnp.float64,
        )
    }
    opt._boundary_from_params = lambda _params: (_ for _ in ()).throw(
        AssertionError("cached initial tangents should skip boundary reconstruction")
    )
    opt._lasym_replay_column_chunk = lambda _n_params: None
    opt._solve_exact_with_tape = lambda _params, return_payload=False: (
        jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float64),
        {"tape": "tape", "axis_override": {}},
    )
    opt._residuals_fn = lambda state: jnp.asarray(
        [state[0] + 2.0 * state[2], state[1] - state[2]],
        dtype=jnp.float64,
    )

    monkeypatch.setattr(
        init_guess,
        "initial_guess_from_boundary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached initial tangents should skip initial_guess")
        ),
    )
    monkeypatch.setattr(state_module, "pack_state", lambda state: jnp.asarray(state, dtype=jnp.float64))
    monkeypatch.setattr(state_module, "unpack_state", lambda packed, _layout: packed)
    monkeypatch.setattr(
        discrete_adjoint,
        "checkpoint_tape_state_jvp",
        lambda *, tape, static, initial_tangent, rebuild_preconditioner: initial_tangent,
    )
    monkeypatch.setattr(
        discrete_adjoint,
        "checkpoint_tape_state_jvp_columns",
        lambda *, tape, static, initial_tangents, rebuild_preconditioner, column_chunk: initial_tangents,
    )
    monkeypatch.setattr(
        discrete_adjoint,
        "checkpoint_tape_state_vjp",
        lambda *, tape, static, final_cotangent, rebuild_preconditioner: final_cotangent,
    )

    op = opt.residual_linear_operator(np.asarray([0.0, 0.0]))

    np.testing.assert_allclose(op.matvec(np.asarray([5.0, 7.0])), [81.0, -17.0])
    np.testing.assert_allclose(op.rmatvec(np.asarray([2.0, -3.0])), [16.0, 19.0])
    assert opt._profile["linear_operator_initial_tangents_cache_hit"]["count"] == 1
    assert "linear_operator_initial_tangents_cache_miss" not in opt._profile


def test_scan_exact_history_can_be_reconstructed_from_residuals():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "scan"
    opt._last_jacobian_key = [None]
    opt._last_jacobian_residual = None
    opt._exact_state_cache = {}
    opt._exact_cache = {}
    opt._history = []
    opt._wall_t0 = 0.0
    opt._iota_fn = None
    opt._aspect_target = 7.0
    opt._aspect_weight = 2.0
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._exact_cache_key = lambda _params: b"accepted"

    def fake_jacobian(_params):
        opt._last_jacobian_residual = np.asarray([0.5, 3.0, 4.0], dtype=float)
        return np.asarray([[1.0], [2.0], [3.0]], dtype=float)

    opt.jacobian_fun = fake_jacobian

    jac = opt._jacobian_fun_tracked(np.asarray([1.0]))

    np.testing.assert_allclose(jac, np.asarray([[1.0], [2.0], [3.0]]))
    assert len(opt._history) == 1
    entry = opt._history[0]
    assert entry["aspect"] == pytest.approx(7.25)
    assert entry["cost"] == pytest.approx(0.5 * (0.5**2 + 3.0**2 + 4.0**2))
    assert entry["qs_objective"] == pytest.approx(25.0)


def test_tape_exact_history_reuses_jacobian_residual_metadata_without_qs_callback():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._last_jacobian_key = [None]
    opt._last_jacobian_residual = None
    opt._exact_state_cache = {}
    opt._exact_cache = {b"accepted": (object(), object())}
    opt._exact_residual_cache = {}
    opt._history = []
    opt._wall_t0 = 0.0
    opt._iota_fn = None
    opt._aspect_target = 7.0
    opt._aspect_weight = 2.0
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._has_residual_block_metadata = True
    opt._exact_cache_key = lambda _params: b"accepted"
    opt._qs_total_from_state_fn = lambda _state: (_ for _ in ()).throw(
        AssertionError("QS total callback should not rerun for accepted history")
    )
    opt._evaluate_residuals_from_state = lambda _state: (_ for _ in ()).throw(
        AssertionError("residual callback should not rerun for accepted history")
    )

    def fake_jacobian(_params):
        opt._last_jacobian_residual = np.asarray([0.5, 3.0, 4.0], dtype=float)
        return np.asarray([[1.0], [2.0], [3.0]], dtype=float)

    opt.jacobian_fun = fake_jacobian

    jac = opt._jacobian_fun_tracked(np.asarray([1.0]))

    np.testing.assert_allclose(jac, np.asarray([[1.0], [2.0], [3.0]]))
    assert len(opt._history) == 1
    entry = opt._history[0]
    assert entry["aspect"] == pytest.approx(7.25)
    assert entry["cost"] == pytest.approx(0.5 * (0.5**2 + 3.0**2 + 4.0**2))
    assert entry["qs_objective"] == pytest.approx(25.0)


def test_exact_residual_after_jacobian_uses_cached_residual_without_state_eval():
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._last_jacobian_key = [b"accepted"]
    opt._last_jacobian_residual = None
    opt._exact_residual_cache = {b"accepted": np.asarray([3.0, 4.0], dtype=float)}
    opt._exact_cache = {b"accepted": (object(), object())}
    opt._profile = {}
    opt._evaluate_residuals_from_state = lambda _state: (_ for _ in ()).throw(
        AssertionError("state residual callback should not run on residual-cache hit")
    )

    np.testing.assert_allclose(opt._exact_residual_after_jacobian(), [3.0, 4.0])


def test_history_entry_uses_residual_block_metadata_for_qs_total(monkeypatch):
    monkeypatch.setattr(
        "vmec_jax.wout.equilibrium_aspect_ratio_from_state",
        lambda **_kwargs: 6.0,
    )

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._static = SimpleNamespace()
    opt._iota_fn = None
    opt._aspect_target = None
    opt._aspect_weight = 1.0
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._has_residual_block_metadata = True
    opt._qs_total_from_state_fn = lambda _state: (_ for _ in ()).throw(
        AssertionError("QS total callback should not rerun when residual blocks are known")
    )

    entry = opt._history_entry_from_state_or_residual(
        object(),
        np.asarray([10.0, 3.0, 4.0], dtype=float),
        wall_time_s=1.25,
    )

    assert entry["aspect"] == pytest.approx(6.0)
    assert entry["cost"] == pytest.approx(0.5 * (10.0**2 + 3.0**2 + 4.0**2))
    assert entry["qs_objective"] == pytest.approx(25.0)


def test_run_final_history_reuses_cached_jacobian_residual_metadata():
    state = object()
    residual = np.asarray([0.5, 3.0, 4.0], dtype=float)

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._scan_exact_path = "tape"
    opt._solver_device_name = None
    opt._inside_solver_device_context = False
    opt._trial_residual_cache = {}
    opt._exact_cache = {b"accepted": (state, {})}
    opt._exact_state_cache = {b"accepted": state}
    opt._exact_residual_cache = {}
    opt._initial_tangent_cache = {}
    opt._last_jacobian_key = [None]
    opt._last_jacobian_residual = None
    opt._static = SimpleNamespace()
    opt._inner_max_iter = 0
    opt._inner_ftol = 0.0
    opt._trial_max_iter = 0
    opt._trial_ftol = 0.0
    opt._post_jacobian_clear = lambda *args, **kwargs: None
    opt._exact_cache_key = lambda _params: b"accepted"
    opt._aspect_target = 7.0
    opt._aspect_weight = 2.0
    opt._n_non_qs = 1
    opt._n_qs = None
    opt._has_residual_block_metadata = True
    opt._qs_total_from_state_fn = lambda _state: (_ for _ in ()).throw(
        AssertionError("QS total callback should not rerun for final history")
    )
    opt._evaluate_residuals_from_state = lambda _state: (_ for _ in ()).throw(
        AssertionError("residual callback should not rerun for final history")
    )
    opt._solve_exact_with_tape = lambda _params, return_payload=False: (state, {}) if return_payload else state
    opt.residual_fun = lambda _params: residual.copy()
    opt.forward_residual_fun = lambda _params: (_ for _ in ()).throw(
        AssertionError("line-search trial residual should not run")
    )

    def fake_jacobian(_params):
        opt._last_jacobian_residual = residual.copy()
        opt._remember_exact_residual(b"accepted", residual)
        return np.zeros((3, 1), dtype=float)

    opt.jacobian_fun = fake_jacobian

    result = opt.run(np.asarray([0.0]), method="gauss_newton", max_nfev=1, verbose=0)

    assert result["success"]
    assert result["_history_dump"]["objective_final"] == pytest.approx(float(np.dot(residual, residual)))
    assert result["_history_dump"]["qs_final"] == pytest.approx(25.0)
    assert result["_history_dump"]["aspect_final"] == pytest.approx(7.25)
    assert result["_history_dump"]["history"][-1]["qs_objective"] == pytest.approx(25.0)


def test_gpu_replay_chunk_covers_symmetric_mode2(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK", raising=False)
    monkeypatch.delenv("VMEC_JAX_REPLAY_COLUMN_CHUNK", raising=False)
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=True))
    opt._solver_device_name = "gpu"

    assert opt._lasym_replay_column_chunk(23) is None
    assert opt._lasym_replay_column_chunk(24) is None
    assert opt._lasym_replay_column_chunk(96) is None

    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    assert opt._lasym_replay_column_chunk(24) is None


def test_lasym_replay_chunk_env_override(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK", "4")
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=True))
    opt._solver_device_name = "gpu"

    assert opt._lasym_replay_column_chunk(48) == 4


def test_lasym_replay_chunk_bad_env_falls_back_to_auto(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_REPLAY_COLUMN_CHUNK", raising=False)
    monkeypatch.setenv("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK", "bad")
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._static = SimpleNamespace(cfg=SimpleNamespace(lasym=False))
    opt._solver_device_name = "gpu"

    assert opt._lasym_replay_column_chunk(24) is None

    monkeypatch.setenv("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK", "off")
    assert opt._lasym_replay_column_chunk(24) is None


def test_fixed_boundary_optimizer_indata_from_params_updates_input_boundary(tmp_path):
    modes = vmec_mode_table(mpol=2, ntor=1)

    def _idx(m: int, n: int) -> int:
        for idx, (mm, nn) in enumerate(zip(np.asarray(modes.m), np.asarray(modes.n))):
            if int(mm) == m and int(nn) == n:
                return idx
        raise AssertionError((m, n))

    k00 = _idx(0, 0)
    k10 = _idx(1, 0)
    k11 = _idx(1, 1)
    r_cos = np.zeros(modes.K)
    z_sin = np.zeros(modes.K)
    r_cos[k00] = 1.0
    z_sin[k10] = 0.2
    boundary = BoundaryCoeffs(
        R_cos=r_cos,
        R_sin=np.zeros(modes.K),
        Z_cos=np.zeros(modes.K),
        Z_sin=z_sin,
    )
    specs = [
        BoundaryParamSpec("rc10", "rc", k10, 1, 0),
        BoundaryParamSpec("zs11", "zs", k11, 1, 1),
    ]

    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._boundary_input = boundary
    opt._boundary = boundary
    opt._specs = specs
    opt._static = SimpleNamespace(modes=modes)
    opt._indata = InData(
        scalars={"NFP": 2, "MPOL": 2, "NTOR": 1},
        indexed={"RBC": {(0, 0): 1.0}, "ZBS": {(0, 1): 0.2}},
        source_path=None,
    )

    updated = opt._indata_from_params(np.array([3e-3, -4e-3]))

    assert updated.indexed["RBC"][(0, 0)] == 1.0
    assert updated.indexed["RBC"][(0, 1)] == 3e-3
    assert updated.indexed["ZBS"][(1, 1)] == -4e-3


def test_fixed_boundary_optimizer_save_wout_uses_provided_state(monkeypatch, tmp_path):
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._static = SimpleNamespace(cfg=SimpleNamespace())
    opt._indata = InData(scalars={}, indexed={}, source_path=None)
    opt._flux = object()
    opt._signgs = 1
    opt._exact_cache = {}

    def fail_solve(*_args, **_kwargs):
        raise AssertionError("save_wout should not solve when state is provided")

    captured = {}

    def fake_write(path, run, **kwargs):
        captured["path"] = path
        captured["state"] = run.state
        captured["kwargs"] = kwargs

    opt._solve_forward = fail_solve
    monkeypatch.setattr("vmec_jax.driver.write_wout_from_fixed_boundary_run", fake_write)

    solved_state = object()
    out = tmp_path / "wout_final.nc"
    opt.save_wout(out, state=solved_state)

    assert captured["path"] == str(out)
    assert captured["state"] is solved_state
    assert captured["kwargs"]["fast_bcovar"] is True
    assert opt._profile["write_wout"]["count"] == 1


def test_fixed_boundary_optimizer_save_wout_reuses_state_cache(monkeypatch, tmp_path):
    opt = object.__new__(FixedBoundaryExactOptimizer)
    opt._static = SimpleNamespace(cfg=SimpleNamespace())
    opt._indata = InData(scalars={}, indexed={}, source_path=None)
    opt._flux = object()
    opt._signgs = 1
    opt._profile = {}
    params = np.array([0.1, -0.2])
    key = opt._exact_cache_key(params)
    solved_state = object()
    opt._exact_cache = {}
    opt._exact_state_cache = {key: solved_state}

    def fail_solve(*_args, **_kwargs):
        raise AssertionError("save_wout should reuse cached accepted state")

    captured = {}

    def fake_write(path, run, **kwargs):
        captured["path"] = path
        captured["state"] = run.state
        captured["kwargs"] = kwargs

    opt._solve_forward = fail_solve
    monkeypatch.setattr("vmec_jax.driver.write_wout_from_fixed_boundary_run", fake_write)

    out = tmp_path / "wout_cached.nc"
    opt.save_wout(out, params=params)

    assert captured["path"] == str(out)
    assert captured["state"] is solved_state
    assert captured["kwargs"]["fast_bcovar"] is True
    assert opt._profile["exact_state_cache_hit"]["count"] == 1
    assert opt._profile["write_wout"]["count"] == 1


def test_interpolate_indata_boundary_blends_selected_coefficients_and_projects_modes():
    seed = InData(
        scalars={"NFP": 3, "MPOL": 3, "NTOR": 3, "LASYM": False},
        indexed={
            "RBC": {(0, 0): 1.0, (1, 0): 0.2, (3, 0): 9.0},
            "ZBS": {(1, 0): 0.1},
            "AC": {(0,): 1.0},
        },
        source_path=None,
    )
    reference = InData(
        scalars={"NFP": 3, "MPOL": 6, "NTOR": 6, "LASYM": True},
        indexed={
            "RBC": {(0, 0): 2.0, (1, 0): 0.4, (2, 0): 0.6, (3, 0): 8.0},
            "ZBS": {(1, 0): 0.3, (2, 1): -0.5},
        },
        source_path=None,
    )

    out = interpolate_indata_boundary(seed, reference, 0.25, max_mode=2)

    assert out.scalars["NFP"] == 3
    assert out.scalars["LASYM"] is False
    assert out.scalars["MPOL"] == 6
    assert out.scalars["NTOR"] == 6
    assert out.indexed["RBC"][(0, 0)] == pytest.approx(1.25)
    assert out.indexed["RBC"][(1, 0)] == pytest.approx(0.25)
    assert out.indexed["RBC"][(2, 0)] == pytest.approx(0.15)
    assert (3, 0) not in out.indexed["RBC"]
    assert out.indexed["ZBS"][(1, 0)] == pytest.approx(0.15)
    assert out.indexed["ZBS"][(2, 1)] == pytest.approx(-0.125)
    assert out.indexed["AC"] == {(0,): 1.0}


def test_interpolate_indata_boundary_rejects_nfp_mismatch():
    seed = InData(scalars={"NFP": 2}, indexed={}, source_path=None)
    reference = InData(scalars={"NFP": 3}, indexed={}, source_path=None)

    with pytest.raises(ValueError, match="same-NFP"):
        interpolate_indata_boundary(seed, reference, 0.5)

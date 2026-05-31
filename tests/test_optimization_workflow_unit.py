from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.optimization import BoundaryParamSpec


def test_objective_factory_callbacks_dispatch_to_helpers(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(static="static", indata="indata", signgs=-1, flux="flux")
    monkeypatch.setattr(
        workflow,
        "equilibrium_aspect_ratio_from_state",
        lambda *, state, static: 8.5 if (state, static) == ("state", "static") else -1.0,
    )
    monkeypatch.setattr(workflow, "mean_iota", lambda ctx_arg, state: 0.33 if (ctx_arg, state) == (ctx, "state") else -1.0)

    def fake_iota_floor(value, target, *, softness):
        assert value == 0.33
        assert target == 0.41
        assert softness == 0.02
        return np.asarray([0.08])

    def fake_lgradb_penalty_from_state(**kwargs):
        assert kwargs == {
            "state": "state",
            "static": "static",
            "indata": "indata",
            "signgs": -1,
            "flux_local": "flux",
            "threshold": 0.25,
            "s_index": -2,
            "ntheta": 5,
            "nphi": 7,
            "smooth_penalty": 0.03,
        }
        return {"residuals1d": np.asarray([2.0, 3.0]), "total": 13.0}

    monkeypatch.setattr(workflow, "smooth_min_abs_iota_residual", fake_iota_floor)
    monkeypatch.setattr(workflow, "lgradb_penalty_from_state", fake_lgradb_penalty_from_state)

    aspect = workflow.aspect_objective(target=7.0, weight=2.0)
    iota = workflow.mean_iota_objective(target=0.4, weight=3.0)
    floor = workflow.abs_mean_iota_floor_objective(target=0.41, weight=4.0, softness=0.02)
    ceiling = workflow.abs_mean_iota_ceiling_objective(maximum=0.25, weight=5.0, softness=0.0)
    lgradb = workflow.lgradb_objective(
        threshold=0.25,
        weight=5.0,
        s_index=-2,
        ntheta=5,
        nphi=7,
        smooth_penalty=0.03,
    )

    np.testing.assert_allclose(aspect.residual(ctx, "state"), [3.0])
    np.testing.assert_allclose(iota.residual(ctx, "state"), [-0.21])
    np.testing.assert_allclose(floor.residual(ctx, "state"), [0.32])
    np.testing.assert_allclose(ceiling.residual(ctx, "state"), [0.4])
    np.testing.assert_allclose(lgradb.residual(ctx, "state"), [10.0, 15.0])
    assert lgradb.total(ctx, "state") == 325.0

    ceiling_object = workflow.AbsMeanIotaCeiling(0.25, softness=0.0)
    np.testing.assert_allclose(ceiling_object.J(ctx, "state"), 0.08, rtol=1.0e-12, atol=1.0e-12)
    ceiling_term = ceiling_object.to_objective_term(target=999.0, residual_weight=3.0)
    assert ceiling_term.metadata == {"iota_abs_max": 0.25}
    assert ceiling_term.track_iota is True


def test_enable_line_buffered_output_is_idempotent_and_tolerates_stream_errors(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    class RaisingStream:
        def __init__(self):
            self.calls = 0

        def reconfigure(self, **kwargs):
            self.calls += 1
            assert kwargs == {"line_buffering": True}
            raise TypeError("not supported")

    class PlainStream:
        pass

    stream = RaisingStream()
    monkeypatch.setattr(workflow, "_LINE_BUFFERING_ENABLED", False)
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", PlainStream())

    workflow._enable_line_buffered_output()
    workflow._enable_line_buffered_output()

    assert stream.calls == 1
    assert workflow._LINE_BUFFERING_ENABLED is True


def test_workflow_mode_limit_seed_and_summary_guard_branches(capsys) -> None:
    import vmec_jax.optimization_workflow as workflow
    from vmec_jax.namelist import minimal_fixed_boundary_indata

    assert workflow.normalize_boundary_mode_limits({"max_m": 1, "max_n": 3}).mode == 3
    with pytest.raises(ValueError, match="mode/max_mode"):
        workflow.normalize_boundary_mode_limits({})
    with pytest.raises(ValueError, match="At least one"):
        workflow.normalize_boundary_mode_limits((None, None))
    with pytest.raises(ValueError, match="Boundary stage tuples"):
        workflow.normalize_boundary_mode_limits((1, 2, 3, 4))

    seed = minimal_fixed_boundary_indata(nfp=2)
    with pytest.raises(ValueError, match="non-negative"):
        workflow.simple_omnigenity_seed_indata(seed, max_mode=-1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        workflow.simple_omnigenity_seed_indata(seed, max_mode=1, perturbation=float("nan"))
    with pytest.raises(ValueError, match="lambda must be finite"):
        workflow.interpolate_indata_boundary(seed, seed, float("nan"))
    with pytest.raises(ValueError, match="bmnc_b"):
        workflow._slice_boozer_surfaces({}, 0)

    workflow.print_qs_problem_summary(
        method="gauss-newton",
        max_nfev=3,
        use_mode_continuation=False,
        use_ess=False,
        ess_alpha=1.0,
        objectives=[],
        specs=[BoundaryParamSpec(name="rc01", kind="rc", index=0, m=0, n=1)],
        x_scale=np.asarray([1.0]),
        optimizer=SimpleNamespace(
            aspect_ratio=lambda _params: 6.0,
            quasisymmetry_objective=lambda _params: 1.0e-3,
        ),
        params0=np.asarray([0.0]),
    )
    workflow.print_qs_final_summary(
        {
            "message": "done",
            "_history_dump": {
                "aspect_final": 6.0,
                "iota_final": 0.42,
                "qs_final": 1.0e-4,
                "objective_final": 1.0e-3,
                "objective_initial": 2.0e-3,
            },
        }
    )
    out = capsys.readouterr().out
    assert "ESS disabled - uniform scales." in out
    assert "Mean iota (final):" in out


def test_least_squares_tuple_weights_are_simsopt_style() -> None:
    from vmec_jax.optimization_workflow import LeastSquaresProblem

    def residual_value(_ctx, _state):
        return np.asarray([3.0])

    problem = LeastSquaresProblem.from_tuples([(residual_value, 1.0, 4.0)])

    np.testing.assert_allclose(problem.objective_terms[0].residual(None, None), [4.0])
    with pytest.raises(ValueError, match="finite and non-negative"):
        LeastSquaresProblem.from_tuples([(residual_value, 0.0, -1.0)])
    with pytest.raises(ValueError, match="finite and non-negative"):
        LeastSquaresProblem.from_tuples([(residual_value, 0.0, np.inf)])


def test_qi_workflow_writes_stage_checkpoint_after_completed_stage(monkeypatch, tmp_path: Path) -> None:
    import vmec_jax.optimization_workflow as workflow

    class FakeOptimizer:
        def aspect_ratio(self, _params):
            return 5.0

        def quasisymmetry_objective(self, _params):
            return 1.0e-2

        def run(self, params0, **_kwargs):
            assert np.asarray(params0).shape == (1,)
            return {
                "x": np.asarray([0.125]),
                "message": "stage complete",
                "success": True,
                "nfev": 2,
                "njev": 1,
                "_history_dump": {
                    "history": [
                        {"objective": 4.0, "qs_objective": 3.0, "aspect": 5.2, "iota": 0.30, "wall_time_s": 0.0},
                        {"objective": 1.0, "qs_objective": 0.5, "aspect": 5.0, "iota": 0.42, "wall_time_s": 3.0},
                    ],
                    "objective_initial": 4.0,
                    "objective_final": 1.0,
                    "qs_initial": 3.0,
                    "qs_final": 0.5,
                    "aspect_initial": 5.2,
                    "aspect_final": 5.0,
                    "iota_initial": 0.30,
                    "iota_final": 0.42,
                    "nfev": 2,
                    "njev": 1,
                    "total_wall_time_s": 3.0,
                    "success": True,
                    "message": "stage complete",
                },
            }

        def save_input(self, path, _params):
            Path(path).write_text("input\n")

        def save_wout(self, path, _params, state=None):
            Path(path).write_text(f"wout {state}\n")

        def _indata_from_params(self, _params):
            return "next-indata"

    def fake_build_stage(*_args, **_kwargs):
        return SimpleNamespace(
            specs=[BoundaryParamSpec(name="rc01", kind="rc", index=0, m=0, n=1)],
            optimizer=FakeOptimizer(),
            ctx=SimpleNamespace(),
        )

    monkeypatch.setattr(workflow, "build_quasi_isodynamic_objective_stage", fake_build_stage)
    monkeypatch.setattr(workflow, "config_from_indata", lambda _indata: "next-cfg")

    result = workflow.run_quasi_isodynamic_objective_optimization(
        cfg="cfg",
        indata="indata",
        scalar_objectives=[],
        qi_objectives=[],
        stage_modes=[1],
        max_mode=1,
        max_nfev=2,
        continuation_nfev=0,
        method="scipy_matrix_free",
        ftol=1.0e-5,
        gtol=1.0e-5,
        xtol=1.0e-5,
        use_ess=False,
        ess_alpha=1.2,
        output_dir=tmp_path,
        label="QI checkpoint test",
        use_mode_continuation=True,
        surfaces=[0.5],
        mboz=2,
        nboz=2,
        nphi=5,
        nalpha=3,
        n_bounce=3,
        include_bounce_endpoints=True,
        softness=0.02,
        width_weight=1.0,
        branch_width_weight=0.5,
        branch_width_softness=0.02,
        profile_weight=0.1,
        shuffle_profile_weight=1.0,
        shuffle_profile_softness=0.02,
        aligned_profile_weight=0.0,
        aligned_profile_softness=0.02,
        aligned_profile_trap_level=0.65,
        aligned_profile_trap_softness=0.05,
        phimin=0.0,
        save_stage_inputs=True,
        save_stage_wouts=False,
        save_final_outputs=False,
    )

    stage_dir = tmp_path / "stage_01_mode01_m01_n01"
    checkpoint = json.loads((tmp_path / "stage_checkpoint.json").read_text())
    stage_checkpoint = json.loads((stage_dir / "qi_stage_checkpoint.json").read_text())
    stage_history = json.loads((stage_dir / "history.json").read_text())
    stage_diagnostics = json.loads((stage_dir / "diagnostics.json").read_text())

    assert result.stage_modes == [1]
    assert checkpoint == stage_checkpoint
    assert checkpoint["role"] == "mode_continuation"
    assert checkpoint["completed_stage_modes"] == [1]
    assert checkpoint["history"]["objective_final"] == 1.0
    assert stage_history["iota_final"] == 0.42
    assert stage_diagnostics["mean_iota"] == 0.42
    assert checkpoint["input_path"] == str(stage_dir / "input.final")


def test_simple_omnigenity_seed_indata_keeps_base_and_perturbs_active_modes() -> None:
    from vmec_jax.namelist import minimal_fixed_boundary_indata
    from vmec_jax.optimization_workflow import simple_omnigenity_seed_indata

    source = minimal_fixed_boundary_indata(nfp=2, r0=1.1, rbc01=0.17, zbs01=0.19)
    source.indexed["RBC"][(2, 2)] = 7.0
    source.indexed["RBS"] = {(1, 1): 8.0}

    seeded = simple_omnigenity_seed_indata(source, max_mode=1, perturbation=1.0e-5)

    assert source.indexed["RBC"][(2, 2)] == 7.0
    assert source.indexed["RBS"][(1, 1)] == 8.0
    assert "RBS" not in seeded.indexed
    assert "ZBC" not in seeded.indexed
    assert (2, 2) not in seeded.indexed["RBC"]
    assert seeded.indexed["RBC"][(0, 0)] == pytest.approx(1.1)
    assert seeded.indexed["RBC"][(0, 1)] == pytest.approx(0.17)
    assert seeded.indexed["ZBS"][(0, 1)] == pytest.approx(0.19)

    for index in ((1, 0), (-1, 1), (1, 1)):
        assert abs(seeded.indexed["RBC"][index]) == pytest.approx(1.0e-5)
        assert abs(seeded.indexed["ZBS"][index]) == pytest.approx(1.0e-5)


def test_prepare_simple_omnigenity_seed_input_writes_rebuilt_seed(tmp_path: Path) -> None:
    from vmec_jax.namelist import minimal_fixed_boundary_indata, read_indata, write_indata
    from vmec_jax.optimization_workflow import prepare_simple_omnigenity_seed_input

    source_path = tmp_path / "input.seed"
    write_indata(source_path, minimal_fixed_boundary_indata(nfp=3, mpol=2, ntor=2))

    disabled = prepare_simple_omnigenity_seed_input(source_path, tmp_path / "unused", max_mode=2, enabled=False)
    assert disabled == source_path

    output_path = prepare_simple_omnigenity_seed_input(
        source_path,
        tmp_path / "nested" / "case",
        max_mode=2,
        min_vmec_mode=5,
        perturbation=2.0e-5,
    )

    assert output_path == tmp_path / "nested" / "case" / "input.simple_seed"
    seeded = read_indata(output_path)
    assert seeded.scalars["MPOL"] == 5
    assert seeded.scalars["NTOR"] == 5
    assert abs(seeded.indexed["RBC"][(2, 2)]) == pytest.approx(2.0e-5)
    assert abs(seeded.indexed["ZBS"][(-2, 2)]) == pytest.approx(2.0e-5)


def test_fixed_boundary_vmec_from_input_can_apply_simple_seed(tmp_path: Path) -> None:
    from vmec_jax.namelist import minimal_fixed_boundary_indata, write_indata
    from vmec_jax.optimization_workflow import FixedBoundaryVMEC

    source = minimal_fixed_boundary_indata(nfp=2, r0=1.3, rbc01=0.11, zbs01=0.12, mpol=2, ntor=2)
    source.indexed["RBC"][(3, 3)] = 9.0
    input_path = tmp_path / "input.seed"
    write_indata(input_path, source)

    vmec = FixedBoundaryVMEC.from_input(
        input_path,
        max_mode=1,
        min_vmec_mode=5,
        output_dir=tmp_path / "out",
        simple_seed=True,
        simple_seed_perturbation=3.0e-5,
    )

    assert vmec.input_file == input_path
    assert vmec.indata.scalars["MPOL"] == 5
    assert vmec.indata.scalars["NTOR"] == 5
    assert vmec.indata.indexed["RBC"][(0, 0)] == pytest.approx(1.3)
    assert vmec.indata.indexed["RBC"][(0, 1)] == pytest.approx(0.11)
    assert vmec.indata.indexed["ZBS"][(0, 1)] == pytest.approx(0.12)
    assert (3, 3) not in vmec.indata.indexed["RBC"]
    assert abs(vmec.indata.indexed["RBC"][(1, 0)]) == pytest.approx(3.0e-5)


def test_mean_iota_handles_axis_only_and_full_profiles(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(static="static", indata="indata", signgs=1)
    profiles = iter(
        [
            (None, np.asarray([0.1]), None),
            (None, np.asarray([0.0, 0.3, 0.5]), None),
        ]
    )

    def fake_iota_profiles_from_state(**kwargs):
        assert kwargs == {"state": "state", "static": "static", "indata": "indata", "signgs": 1}
        return next(profiles)

    monkeypatch.setattr(workflow, "equilibrium_iota_profiles_from_state", fake_iota_profiles_from_state)

    assert float(workflow.mean_iota(ctx, "state")) == 0.0
    assert float(workflow.mean_iota(ctx, "state")) == pytest.approx(0.4)


def test_boozer_target_shape_guards_and_qi_runtime_errors(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    with pytest.raises(RuntimeError, match="inside a QI solve"):
        workflow.BoozerBTarget(target_bmnc=np.ones((1, 2))).J(None, None)

    options = workflow.QuasiIsodynamicOptions(surfaces=[0.5])
    with pytest.raises(RuntimeError, match="inside a QI solve"):
        workflow.MirrorRatio(threshold=0.3, qi_options=options).J(None, None)

    term = workflow.qi_boozer_b_target_objective(target_bmnc=np.ones((1, 3)))
    with pytest.raises(ValueError, match="target_bmnc"):
        term.residual_and_total(None, None, {"booz": {"bmnc_b": np.ones((2, 3))}})

    term = workflow.qi_boozer_b_target_objective(target_bmnc=np.ones((1, 3)), target_bmns=np.ones((1, 2)))
    with pytest.raises(ValueError, match="target_bmns"):
        term.residual_and_total(None, None, {"booz": {"bmnc_b": np.ones((1, 3))}})


def test_boozer_b_target_from_wout_uses_nearest_surfaces_and_transposes(monkeypatch, tmp_path) -> None:
    import vmec_jax.optimization_workflow as workflow

    instances = []

    class FakeBoozXform:
        def __init__(self, *, verbose):
            assert verbose == 0
            self.s_in = np.asarray([0.0, 0.2, 0.7, 1.0])
            self.bmnc_b = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            self.bmns_b = np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            self.xm_b = np.asarray([0, 1, 2])
            self.xn_b = np.asarray([0, -1, 1])
            self.s_b = np.asarray([0.2, 0.7])
            self.nfp = 5
            instances.append(self)

        def read_wout(self, path, *, flux):
            self.read_args = (path, flux)

        def run(self):
            self.ran = True
            self.xm_b = np.asarray([0, 1, 2])
            self.xn_b = np.asarray([0, -1, 1])

    monkeypatch.setitem(sys.modules, "booz_xform_jax", SimpleNamespace(Booz_xform=FakeBoozXform))

    out = workflow.boozer_b_target_from_wout(tmp_path / "wout_fake.nc", surfaces=[0.18, 0.72], mboz=6, nboz=7)

    bx = instances[0]
    assert bx.read_args == (str(tmp_path / "wout_fake.nc"), False)
    assert bx.compute_surfs == [1, 2]
    assert bx.mboz == 6
    assert bx.nboz == 7
    assert bx.mnboz is None
    assert bx.xm_b is not None
    assert bx.xn_b is not None
    assert bx._prepared is False
    assert bx.ran is True
    np.testing.assert_allclose(out["bmnc_b"], [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    np.testing.assert_allclose(out["bmns_b"], [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    np.testing.assert_array_equal(out["xm_b"], [0, 1, 2])
    assert out["nfp"] == 5


def test_boozer_b_target_from_wout_handles_missing_sine_modes(monkeypatch, tmp_path) -> None:
    import vmec_jax.optimization_workflow as workflow

    class FakeBoozXform:
        def __init__(self, *, verbose):
            self.s_in = np.asarray([0.0, 1.0])
            self.bmnc_b = np.asarray([[1.0]])
            self.xm_b = np.asarray([0])
            self.xn_b = np.asarray([0])
            self.s_b = np.asarray([1.0])
            self.nfp = 1

        def read_wout(self, _path, *, flux):
            assert flux is False

        def run(self):
            self.xm_b = np.asarray([0])
            self.xn_b = np.asarray([0])

    monkeypatch.setitem(sys.modules, "booz_xform_jax", SimpleNamespace(Booz_xform=FakeBoozXform))

    out = workflow.boozer_b_target_from_wout(tmp_path / "wout_fake.nc", surfaces=[0.9], mboz=2, nboz=3)

    assert out["bmns_b"] is None
    np.testing.assert_allclose(out["bmnc_b"], [[1.0]])


def test_build_quasi_isodynamic_stage_wires_shared_field_residuals(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    static = SimpleNamespace(
        modes="modes",
        s=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        cfg=SimpleNamespace(mpol=2, ntor=1, ntheta=5, nzeta=7, nfp=2, lasym=False),
    )
    captured = {}

    class FakeOptimizer:
        def __init__(
            self,
            static_arg,
            indata_arg,
            boundary_arg,
            specs_arg,
            residuals_from_state,
            *,
            boundary_input,
            inner_max_iter,
            inner_ftol,
            trial_max_iter,
            trial_ftol,
            solver_device,
        ):
            captured.update(
                {
                    "static": static_arg,
                    "indata": indata_arg,
                    "boundary": boundary_arg,
                    "specs": specs_arg,
                    "residuals_from_state": residuals_from_state,
                    "boundary_input": boundary_input,
                    "inner_max_iter": inner_max_iter,
                    "inner_ftol": inner_ftol,
                    "trial_max_iter": trial_max_iter,
                    "trial_ftol": trial_ftol,
                    "solver_device": solver_device,
                }
            )

    def fake_prepare_booz_xform_constants(**kwargs):
        captured["booz_kwargs"] = kwargs
        return "constants", "grids"

    def fake_quasi_isodynamic_residual_from_state(**kwargs):
        captured["field_kwargs"] = kwargs
        return {"value": 3.0, "total": 11.0}

    monkeypatch.setitem(
        sys.modules,
        "booz_xform_jax",
        SimpleNamespace(prepare_booz_xform_constants=fake_prepare_booz_xform_constants),
    )
    monkeypatch.setattr(workflow, "truncate_indata_boundary_modes", lambda indata, *, max_mode: f"{indata}-m{max_mode}")
    monkeypatch.setattr(workflow, "build_static", lambda cfg: static)
    monkeypatch.setattr(workflow, "boundary_from_indata", lambda indata, modes, *, apply_m1_constraint: f"boundary:{indata}")
    monkeypatch.setattr(
        workflow,
        "extend_boundary_for_max_mode",
        lambda indata, static_arg, boundary, stage_mode: (f"extended:{indata}", static_arg, f"extended:{boundary}"),
    )
    monkeypatch.setattr(workflow, "boundary_input_from_indata", lambda indata, modes: f"boundary_input:{indata}")
    monkeypatch.setattr(
        workflow,
        "boundary_param_specs",
        lambda *args, **kwargs: [BoundaryParamSpec("rc10", "rc", 0, 1, 0)],
    )
    monkeypatch.setattr(workflow, "initial_guess_from_boundary", lambda *args, **kwargs: "guess")
    monkeypatch.setattr(workflow, "eval_geom", lambda guess, static_arg: SimpleNamespace(sqrtg=np.ones((2, 2))))
    monkeypatch.setattr(workflow, "signgs_from_sqrtg", lambda sqrtg, *, axis_index: -1)
    monkeypatch.setattr(workflow, "flux_profiles_from_indata", lambda indata, s, *, signgs: f"flux:{indata}:{signgs}")
    monkeypatch.setattr(workflow, "quasi_isodynamic_residual_from_state", fake_quasi_isodynamic_residual_from_state)
    monkeypatch.setattr(workflow, "FixedBoundaryExactOptimizer", FakeOptimizer)

    scalar = workflow.ObjectiveTerm("scalar", lambda _ctx, _state: 2.0, target=0.5, weight=2.0)
    qi = workflow.QIObjectiveTerm(
        "qi_fast",
        lambda _ctx, _state, field: (np.asarray([field["value"], field["value"] + 1.0]), field["total"]),
    )

    stage = workflow.build_quasi_isodynamic_objective_stage(
        cfg="cfg",
        indata="indata",
        stage_mode=2,
        scalar_objectives=[scalar],
        qi_objectives=[qi],
        surfaces=[0.25, 0.75],
        mboz=4,
        nboz=5,
        nphi=9,
        nalpha=7,
        n_bounce=11,
        include_bounce_endpoints=True,
        softness=0.03,
        width_weight=1.5,
        branch_width_weight=0.6,
        branch_width_softness=0.04,
        profile_weight=0.2,
        shuffle_profile_weight=0.8,
        shuffle_profile_softness=0.05,
        weighted_shuffle_profile_weight=0.7,
        weighted_shuffle_profile_softness=0.06,
        aligned_profile_weight=0.4,
        aligned_profile_softness=0.07,
        aligned_profile_trap_level=0.55,
        aligned_profile_trap_softness=0.08,
        phimin=0.1,
        project_input_boundary_to_max_mode=True,
        inner_max_iter=3,
        inner_ftol=1.0e-4,
        trial_max_iter=5,
        trial_ftol=2.0e-4,
        solver_device="cpu",
    )

    assert stage.mode == 2
    assert stage.ctx.signgs == -1
    assert stage.boundary_input == "boundary_input:extended:indata-m2"
    assert captured["indata"] == "extended:indata-m2"
    assert captured["boundary"] == "extended:boundary:indata-m2"
    assert captured["boundary_input"] == "boundary_input:extended:indata-m2"
    assert captured["inner_max_iter"] == 3
    assert captured["solver_device"] == "cpu"
    assert captured["booz_kwargs"]["nfp"] == 2
    assert captured["booz_kwargs"]["mboz"] == 4

    residuals = captured["residuals_from_state"]("state")
    np.testing.assert_allclose(residuals, [3.0, 3.0, 4.0])
    assert captured["residuals_from_state"]._n_non_qs == 1
    assert captured["residuals_from_state"]._objective_family == "qi"
    assert captured["residuals_from_state"]._qs_total_from_state("state") == 11.0
    assert captured["field_kwargs"]["state"] == "state"
    assert captured["field_kwargs"]["surfaces"] == [0.25, 0.75]
    assert captured["field_kwargs"]["booz_constants"] == "constants"
    assert captured["field_kwargs"]["booz_grids"] == "grids"


def test_run_fixed_boundary_records_iota_abs_min_on_stage_history(monkeypatch, tmp_path) -> None:
    import vmec_jax.optimization_workflow as workflow

    run_kwargs = {}

    class FakeOptimizer:
        def run(self, params0, **kwargs):
            run_kwargs.update(kwargs)
            return {
                "x": np.asarray([1.0, 2.0]),
                "message": "ok",
                "_history_dump": {
                    "history": [{"objective": 1.0, "wall_time_s": 0.0}],
                    "nfev": 1,
                    "njev": 1,
                    "objective_initial": 1.0,
                    "objective_final": 0.5,
                    "qs_initial": 1.0,
                    "qs_final": 0.5,
                    "aspect_initial": 4.0,
                    "aspect_final": 4.5,
                },
            }

        def _indata_from_params(self, _params):
            return "next-indata"

    monkeypatch.setattr(workflow, "build_fixed_boundary_objective_stage", lambda *args, **kwargs: SimpleNamespace(
        ctx=SimpleNamespace(),
        optimizer=FakeOptimizer(),
        specs=[BoundaryParamSpec("rc10", "rc", 0, 1, 0), BoundaryParamSpec("zs10", "zs", 1, 1, 0)],
    ))
    monkeypatch.setattr(workflow, "config_from_indata", lambda indata: f"cfg:{indata}")
    monkeypatch.setattr(workflow, "print_qs_problem_summary", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "print_qs_final_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_stage_artifacts", lambda **_kwargs: None)
    final_save_calls = []
    monkeypatch.setattr(workflow, "save_qs_final_outputs", lambda **kwargs: final_save_calls.append(kwargs))

    result = workflow.run_fixed_boundary_objective_optimization(
        cfg="cfg",
        indata="indata",
        objectives=[],
        stage_modes=[2],
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
        label="fixed",
        use_mode_continuation=True,
        iota_abs_min=0.41,
        save_final_outputs=False,
    )

    assert run_kwargs["iota_fn"] is not None
    assert final_save_calls == []
    assert result.final_result["_history_dump"]["label"] == "fixed"
    assert result.final_result["_history_dump"]["iota_abs_min"] == 0.41


def test_run_qi_records_iota_abs_min_and_prints_qi_terms(monkeypatch, tmp_path, capsys) -> None:
    import vmec_jax.optimization_workflow as workflow

    run_kwargs = {}

    class FakeOptimizer:
        def run(self, params0, **kwargs):
            run_kwargs.update(kwargs)
            return {
                "x": np.asarray([0.0]),
                "message": "ok",
                "_history_dump": {
                    "history": [{"objective": 2.0, "wall_time_s": 0.0}],
                    "nfev": 1,
                    "njev": 1,
                    "objective_initial": 2.0,
                    "objective_final": 1.0,
                    "qs_initial": 2.0,
                    "qs_final": 1.0,
                    "aspect_initial": 4.0,
                    "aspect_final": 4.2,
                },
            }

        def _indata_from_params(self, _params):
            return "next-qi-indata"

    monkeypatch.setattr(workflow, "build_quasi_isodynamic_objective_stage", lambda *args, **kwargs: SimpleNamespace(
        ctx=SimpleNamespace(),
        optimizer=FakeOptimizer(),
        specs=[BoundaryParamSpec("rc10", "rc", 0, 1, 0)],
    ))
    monkeypatch.setattr(workflow, "config_from_indata", lambda indata: f"cfg:{indata}")
    monkeypatch.setattr(workflow, "print_qs_problem_summary", lambda **_kwargs: print("summary"))
    monkeypatch.setattr(workflow, "print_qs_final_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "save_qs_stage_artifacts", lambda **_kwargs: None)
    final_save_calls = []
    monkeypatch.setattr(workflow, "save_qs_final_outputs", lambda **kwargs: final_save_calls.append(kwargs))

    result = workflow.run_quasi_isodynamic_objective_optimization(
        cfg="cfg",
        indata="indata",
        scalar_objectives=[],
        qi_objectives=[workflow.QIObjectiveTerm("qi_one", lambda _ctx, _state, _field: ([0.0], 0.0))],
        stage_modes=[3],
        max_mode=3,
        max_nfev=4,
        continuation_nfev=1,
        method="scipy",
        ftol=1.0e-6,
        gtol=1.0e-6,
        xtol=1.0e-6,
        use_ess=False,
        ess_alpha=0.0,
        output_dir=tmp_path,
        label="qi",
        use_mode_continuation=True,
        surfaces=[0.5],
        mboz=4,
        nboz=4,
        nphi=9,
        nalpha=5,
        n_bounce=5,
        include_bounce_endpoints=False,
        softness=0.02,
        width_weight=1.0,
        branch_width_weight=0.5,
        branch_width_softness=0.02,
        profile_weight=0.1,
        shuffle_profile_weight=1.0,
        shuffle_profile_softness=0.02,
        weighted_shuffle_profile_weight=0.0,
        weighted_shuffle_profile_softness=0.02,
        aligned_profile_weight=0.0,
        aligned_profile_softness=0.02,
        aligned_profile_trap_level=0.65,
        aligned_profile_trap_softness=0.05,
        phimin=0.0,
        iota_abs_min=0.52,
        save_final_outputs=False,
    )

    out = capsys.readouterr().out
    assert "QI field objectives:" in out
    assert "  - qi_one" in out
    assert run_kwargs["iota_fn"] is not None
    assert final_save_calls == []
    assert result.final_result["_history_dump"]["label"] == "qi"
    assert result.final_result["_history_dump"]["iota_abs_min"] == 0.52


def test_least_squares_solve_rejects_qi_problem_without_options(tmp_path) -> None:
    import vmec_jax.optimization_workflow as workflow

    vmec = SimpleNamespace(cfg="cfg", indata="indata", max_mode=1, output_dir=tmp_path)
    problem = SimpleNamespace(
        metadata={},
        is_qi=True,
        qi_options=None,
        objective_terms=(),
        qi_objective_terms=(workflow.QIObjectiveTerm("qi", lambda _ctx, _state, _field: ([0.0], 0.0)),),
    )

    with pytest.raises(ValueError, match="QI objectives require QuasiIsodynamicOptions"):
        workflow.least_squares_solve(vmec, problem, stage_modes=[1], max_nfev=1, continuation_nfev=0)


def test_problem_and_final_summaries_print_diagnostic_branches(capsys) -> None:
    import vmec_jax.optimization_workflow as workflow

    specs = [BoundaryParamSpec("rc10", "rc", 0, 1, 0)]
    optimizer = SimpleNamespace(
        aspect_ratio=lambda params: 4.25,
        quasisymmetry_objective=lambda params: 1.25e-3,
    )
    workflow.print_qs_problem_summary(
        method="scipy",
        max_nfev=5,
        use_mode_continuation=True,
        use_ess=True,
        ess_alpha=1.5,
        objectives=[workflow.ObjectiveTerm("aspect", lambda _ctx, _state: 0.0, target=4.0, weight=2.0)],
        specs=specs,
        x_scale=np.asarray([0.25]),
        optimizer=optimizer,
        params0=np.asarray([0.0]),
    )
    summary = capsys.readouterr().out
    assert "Parameter space (1 DOFs): ['rc10']" in summary
    assert "ESS scales (alpha=1.5): min=0.250  max=0.250" in summary
    assert "Field objective (initial):     1.250000e-03" in summary

    workflow.print_qs_final_summary(
        {
            "message": "converged",
            "_history_dump": {
                "aspect_final": 4.1,
                "iota_final": 0.42,
                "qs_final": 0.01,
                "objective_initial": 4.0,
                "objective_final": 1.0,
            },
        },
        target_iota=0.43,
    )
    target_iota_summary = capsys.readouterr().out
    assert "Termination: converged" in target_iota_summary
    assert "Mean iota (final):             0.420000  target=0.430000" in target_iota_summary
    assert "Objective reduction:           75.0%" in target_iota_summary

    workflow.print_qs_final_summary(
        {"message": "stopped", "_history_dump": {"iota_final": -0.53}},
        iota_abs_min=0.5,
    )
    floor_summary = capsys.readouterr().out
    assert "Mean iota (final):             -0.530000  min |iota|=0.500000" in floor_summary


def test_save_stage_artifacts_handles_wouts_reruns_and_stale_files(monkeypatch, tmp_path) -> None:
    import vmec_jax.optimization_workflow as workflow

    calls = []

    class FakeOptimizer:
        def save_input(self, path, params):
            calls.append(("input", path.name, tuple(np.asarray(params))))
            path.write_text("input")

        def save_wout(self, path, params, *, state=None):
            calls.append(("wout", path.name, tuple(np.asarray(params)), state))
            path.write_text("wout")

    monkeypatch.setattr(workflow, "run_fixed_boundary", lambda path, *, verbose: {"path": path, "verbose": verbose})
    monkeypatch.setattr(
        workflow,
        "write_wout_from_fixed_boundary_run",
        lambda path, run: calls.append(("rerun_wout", Path(path).name, run["path"], run["verbose"])),
    )
    stage_dir = tmp_path / "stage"

    workflow.save_qs_stage_artifacts(
        stage_dir=stage_dir,
        optimizer=FakeOptimizer(),
        params_initial=np.asarray([1.0]),
        params_final=np.asarray([2.0]),
        result={"_state_initial": "state0", "_state_final": "statef"},
        save_inputs=True,
        save_wouts=True,
        save_rerun_wouts=True,
    )

    assert ("input", "input.initial", (1.0,)) in calls
    assert ("wout", "wout_initial.nc", (1.0,), "state0") in calls
    assert any(call[0] == "rerun_wout" and call[1] == "wout_final_rerun.nc" for call in calls)

    for stale in ("wout_initial.nc", "wout_final.nc", "wout_initial_rerun.nc", "wout_final_rerun.nc"):
        (stage_dir / stale).write_text("stale")

    workflow.save_qs_stage_artifacts(
        stage_dir=stage_dir,
        optimizer=FakeOptimizer(),
        params_initial=np.asarray([3.0]),
        params_final=np.asarray([4.0]),
        result={},
        save_inputs=False,
        save_wouts=False,
        save_rerun_wouts=False,
    )

    for stale in ("wout_initial.nc", "wout_final.nc", "wout_initial_rerun.nc", "wout_final_rerun.nc"):
        assert not (stage_dir / stale).exists()


def test_save_final_outputs_records_targets_and_reruns(monkeypatch, tmp_path) -> None:
    import vmec_jax.optimization_workflow as workflow

    calls = []

    class FakeOptimizer:
        def __init__(self, label):
            self.label = label
            self.saved_history = None

        def save_input(self, path, params):
            calls.append((self.label, "input", path.name, tuple(np.asarray(params))))
            path.write_text("input")

        def save_wout(self, path, params, *, state=None):
            calls.append((self.label, "wout", path.name, tuple(np.asarray(params)), state))
            path.write_text("wout")

        def save_history(self, path, result):
            calls.append((self.label, "history", path.name))
            self.saved_history = result["_history_dump"]
            path.write_text("history")

    initial_optimizer = FakeOptimizer("initial")
    final_optimizer = FakeOptimizer("final")
    monkeypatch.setattr(workflow, "run_fixed_boundary", lambda path, *, verbose: {"path": path, "verbose": verbose})
    monkeypatch.setattr(
        workflow,
        "write_wout_from_fixed_boundary_run",
        lambda path, run: calls.append(("rerun_wout", Path(path).name, run["path"], run["verbose"])),
    )
    final_result = {
        "x": np.asarray([9.0]),
        "_state_final": "statef",
        "_history_dump": {},
    }

    workflow.save_qs_final_outputs(
        output_dir=tmp_path,
        stage_records=[(1, initial_optimizer, np.asarray([1.0]), {"_state_initial": "state0"})],
        final_optimizer=final_optimizer,
        final_result=final_result,
        label="label",
        target_aspect=4.5,
        target_iota=0.42,
        iota_abs_min=0.4,
        save_rerun_wouts=True,
    )

    assert ("initial", "input", "input.initial", (1.0,)) in calls
    assert ("final", "wout", "wout_final.nc", (9.0,), "statef") in calls
    assert ("rerun_wout", "wout_initial_rerun.nc", str(tmp_path / "input.initial"), False) in calls
    assert ("rerun_wout", "wout_final_rerun.nc", str(tmp_path / "input.final"), False) in calls
    assert final_optimizer.saved_history == {
        "label": "label",
        "target_aspect": 4.5,
        "target_iota": 0.42,
        "iota_abs_min": 0.4,
    }


def test_save_optimization_result_writes_canonical_artifacts(tmp_path) -> None:
    import vmec_jax.optimization_workflow as workflow

    calls = []

    class FakeOptimizer:
        def __init__(self, label):
            self.label = label

        def save_input(self, path, params):
            calls.append((self.label, "input", path.name, tuple(np.asarray(params))))
            path.write_text("input")

        def save_wout(self, path, params, *, state=None):
            calls.append((self.label, "wout", path.name, tuple(np.asarray(params)), state))
            path.write_text("wout")

        def save_history(self, path, result):
            calls.append((self.label, "history", path.name, result["message"]))
            path.write_text("history")

    initial_optimizer = FakeOptimizer("initial")
    final_optimizer = FakeOptimizer("final")
    final_result = {
        "x": np.asarray([4.0, 5.0]),
        "message": "converged",
        "_state_final": "statef",
        "_history_dump": {"history": [{"objective": 1.0}], "total_wall_time_s": 2.0},
    }
    result = workflow.FixedBoundaryOptimizationResult(
        stage_records=[
            (
                1,
                initial_optimizer,
                np.asarray([1.0, 2.0]),
                {"_state_initial": "state0", "_history_dump": {}},
            )
        ],
        final_optimizer=final_optimizer,
        final_result=final_result,
        stage_modes=[1],
    )

    paths = workflow.save_optimization_result(result, output_dir=tmp_path)

    assert paths == workflow.OptimizationOutputPaths(
        initial_input=tmp_path / "input.initial",
        final_input=tmp_path / "input.final",
        initial_wout=tmp_path / "wout_initial.nc",
        final_wout=tmp_path / "wout_final.nc",
        history=tmp_path / "history.json",
    )
    assert paths.as_dict()["final_wout"] == tmp_path / "wout_final.nc"
    assert ("initial", "input", "input.initial", (1.0, 2.0)) in calls
    assert ("initial", "wout", "wout_initial.nc", (1.0, 2.0), "state0") in calls
    assert ("final", "input", "input.final", (4.0, 5.0)) in calls
    assert ("final", "wout", "wout_final.nc", (4.0, 5.0), "statef") in calls
    assert ("final", "history", "history.json", "converged") in calls
    for path in paths.as_dict().values():
        assert path.exists()


def test_save_optimization_result_requires_destination() -> None:
    import vmec_jax.optimization_workflow as workflow

    result = workflow.FixedBoundaryOptimizationResult(
        stage_records=[],
        final_optimizer=object(),
        final_result={},
        stage_modes=[],
    )

    with pytest.raises(ValueError, match="Either output_dir or paths"):
        workflow.save_optimization_result(result)


def test_combine_stage_histories_handles_single_stage_and_iota_boundaries() -> None:
    import vmec_jax.optimization_workflow as workflow

    single = [(1, "optimizer", np.asarray([0.0]), {"_history_dump": {}})]
    assert workflow.combine_qs_stage_histories(
        label="single",
        max_mode=1,
        max_nfev=3,
        continuation_nfev=1,
        stage_modes=[1],
        stage_records=single,
    ) is None

    def hist(entries, *, objective_initial, objective_final, qs_initial, qs_final, aspect_initial, aspect_final):
        return {
            "history": entries,
            "nfev": len(entries),
            "njev": len(entries) + 1,
            "objective_initial": objective_initial,
            "objective_final": objective_final,
            "qs_initial": qs_initial,
            "qs_final": qs_final,
            "aspect_initial": aspect_initial,
            "aspect_final": aspect_final,
        }

    stage_records = [
        (
            1,
            "opt1",
            np.asarray([0.0]),
            {
                "_history_dump": hist(
                    [
                        {"objective": 5.0, "wall_time_s": 1.0, "iota": 0.1},
                        {"objective": 4.0, "wall_time_s": 2.0, "iota": 0.2},
                    ],
                    objective_initial=5.0,
                    objective_final=4.0,
                    qs_initial=3.0,
                    qs_final=2.0,
                    aspect_initial=6.0,
                    aspect_final=5.0,
                )
            },
        ),
        (
            2,
            "opt2",
            np.asarray([0.0]),
            {
                "_history_dump": hist(
                    [
                        {"objective": 4.0, "wall_time_s": 0.0, "iota": 0.2},
                        {"objective": 1.0, "wall_time_s": 3.0, "iota": 0.7},
                    ],
                    objective_initial=4.0,
                    objective_final=1.0,
                    qs_initial=2.0,
                    qs_final=0.5,
                    aspect_initial=5.0,
                    aspect_final=4.0,
                )
            },
        ),
    ]

    combined = workflow.combine_qs_stage_histories(
        label="combined",
        max_mode=2,
        max_nfev=10,
        continuation_nfev=2,
        stage_modes=[1, 2],
        stage_records=stage_records,
    )

    assert combined["label"] == "combined"
    assert combined["max_nfev"] == 12
    assert combined["nfev"] == 4
    assert combined["njev"] == 6
    assert combined["total_wall_time_s"] == 5.0
    assert combined["stage_boundaries"] == [1, 2]
    assert combined["iota_initial"] == 0.1
    assert combined["iota_final"] == 0.7
    np.testing.assert_allclose([entry["wall_time_s"] for entry in combined["history"]], [1.0, 2.0, 5.0])


def test_least_squares_problem_assembly_handles_custom_state_and_qi_owners() -> None:
    import vmec_jax.optimization_workflow as workflow

    class StateOwner:
        def J(self, ctx, state):
            return np.asarray([2.0, 4.0])

        def to_objective_term(self, *, target, residual_weight):
            return workflow.ObjectiveTerm(
                "state_owner",
                self.J,
                target=target,
                weight=residual_weight,
                metadata={"target_custom": 3.0},
            )

    class QIOwner:
        requires_qi_field = True

        def __init__(self):
            self.options = workflow.QuasiIsodynamicOptions(surfaces=[0.4])

        def J(self, ctx, state):
            raise RuntimeError

        def to_qi_term(self, residual_weight):
            assert residual_weight == 2.0
            return workflow.QIObjectiveTerm("custom_qi", lambda _ctx, _state, _field: ([0.0], 0.0), self.options)

    state_owner = StateOwner()
    qi_owner = QIOwner()

    problem = workflow.LeastSquaresProblem.from_tuples(
        [
            (state_owner.J, np.asarray([1.0, 1.5]), 9.0),
            (qi_owner.J, np.asarray([0.0]), 4.0),
        ]
    )

    assert problem.metadata == {"target_custom": 3.0}
    assert problem.qi_options is qi_owner.options
    np.testing.assert_allclose(problem.objective_terms[0].residual(None, None), [3.0, 7.5])
    assert [term.name for term in problem.qi_objective_terms] == ["custom_qi"]


def test_lgradb_and_redl_object_state_paths_validate_and_scale(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(static="static", indata="indata", signgs=1, flux="flux")

    def fake_lgradb_penalty_from_state(**kwargs):
        assert kwargs["threshold"] == 0.2
        assert kwargs["s_index"] == -1
        return {"residuals1d": np.asarray([1.5]), "total": 2.25}

    monkeypatch.setattr(workflow, "lgradb_penalty_from_state", fake_lgradb_penalty_from_state)
    lgradb = workflow.LgradB(threshold=0.2)
    np.testing.assert_allclose(lgradb.J(ctx, "state"), [1.5])
    assert lgradb.total(ctx, "state") == 2.25
    lgradb_term = lgradb.to_objective_term(target=0.0, residual_weight=3.0)
    np.testing.assert_allclose(lgradb_term.residual(ctx, "state"), [4.5])
    assert lgradb_term.total(ctx, "state") == 20.25

    redl = workflow.RedlBootstrapMismatch(
        helicity_n=1,
        ne_coeffs=[1.0, 2.0],
        Te_coeffs=[3.0],
        Ti_coeffs=[4.0, 5.0],
        Zeff_coeffs=2.0,
    )
    assert redl.Ti_coeffs == (4.0, 5.0)
    with pytest.raises(ValueError, match="target=0"):
        redl.to_objective_term(target=1.0, residual_weight=1.0)


def test_fixed_boundary_result_properties_and_timing_summaries() -> None:
    import vmec_jax.optimization_workflow as workflow

    first_history = {
        "history": [{"objective": 5.0}, {"objective": 4.0}],
        "nfev": 2,
        "total_wall_time_s": 1.25,
    }
    final_history = {
        "history": [{"objective": 3.0}, {"not_objective": 0.0}],
        "njev": 4,
        "nit": 5,
    }
    first_result = {"_state_initial": "initial-state", "_history_dump": first_history}
    final_result = {
        "x": [9.0, 10.0],
        "_state_final": "final-state",
        "_history_dump": final_history,
        "nfev": 7,
        "njev": 8,
        "nit": 9,
    }
    first_record = (1, "optimizer-1", np.asarray([1.0, 2.0]), first_result)
    final_record = (2, "optimizer-2", np.asarray([3.0, 4.0]), final_result)

    result = workflow.FixedBoundaryOptimizationResult(
        stage_records=[first_record, final_record],
        final_optimizer="optimizer-2",
        final_result=final_result,
        stage_modes=[1, 2],
    )

    assert result.initial_stage is first_record
    assert result.final_stage is final_record
    assert result.initial_optimizer == "optimizer-1"
    np.testing.assert_allclose(result.initial_params, [1.0, 2.0])
    assert result.initial_result is first_result
    assert result.initial_state == "initial-state"
    assert result.history == final_history
    assert result.history_entries == ({"objective": 3.0}, {"not_objective": 0.0})
    assert result.stage_histories == (first_history, final_history)
    np.testing.assert_allclose(result.objective_history, [3.0, np.nan])
    np.testing.assert_allclose(result.final_params, [9.0, 10.0])
    assert result.final_state == "final-state"
    assert result.stage_timing_summaries[0]["mode"] == 1
    assert result.stage_timing_summaries[0]["nfev"] == 2
    assert result.timing_summary["nfev"] == 7
    assert result.timing_summary["njev"] == 4
    assert result.timing_summary["stages"][1]["mode"] == 2


def test_least_squares_problem_rejects_bad_tuples_and_mismatched_qi_options() -> None:
    import vmec_jax.optimization_workflow as workflow

    options = workflow.QuasiIsodynamicOptions(surfaces=np.asarray([0.5]))
    other_options = workflow.QuasiIsodynamicOptions(surfaces=np.asarray([0.25]))

    class FakeQIObjective:
        requires_qi_field = True

        def __init__(self, qi_options, name: str):
            self.qi_options = qi_options
            self.name = name

        def J(self, _ctx, _state):
            raise RuntimeError("assembled through LeastSquaresProblem")

        def to_qi_term(self, residual_weight: float):
            def _evaluate(_ctx, _state, field):
                residual = np.asarray(field["residual"], dtype=float) * float(residual_weight)
                return residual, np.sum(residual * residual)

            return workflow.QIObjectiveTerm(self.name, _evaluate, qi_options=self.qi_options)

    with pytest.raises(ValueError, match="must be \\(callable, target, weight\\)"):
        workflow.LeastSquaresProblem.from_tuples([(lambda ctx, state: 0.0, 0.0)])
    with pytest.raises(TypeError, match="first entry must be callable"):
        workflow.LeastSquaresProblem.from_tuples([(object(), 0.0, 1.0)])
    with pytest.raises(ValueError, match="finite and non-negative"):
        workflow.LeastSquaresProblem.from_tuples([(lambda ctx, state: 0.0, 0.0, -1.0)])
    with pytest.raises(ValueError, match="target=0"):
        workflow.LeastSquaresProblem.from_tuples([(FakeQIObjective(options, "qi").J, 1.0, 1.0)])

    problem = workflow.LeastSquaresProblem.from_tuples([(FakeQIObjective(options, "qi").J, 0.0, 4.0)])
    assert problem.is_qi
    assert problem.qi_options is options
    assert len(problem.qi_objective_terms) == 1

    with pytest.raises(ValueError, match="must share one"):
        workflow.LeastSquaresProblem.from_tuples(
            [
                (FakeQIObjective(options, "qi-a").J, 0.0, 1.0),
                (FakeQIObjective(other_options, "qi-b").J, 0.0, 1.0),
            ]
        )


def test_mirror_and_elongation_helpers_are_generic_unless_qi_field_is_requested() -> None:
    import vmec_jax.optimization_workflow as workflow

    options = workflow.QuasiIsodynamicOptions(surfaces=np.asarray([0.5]))

    generic_mirror = workflow.MirrorRatio(threshold=0.3, surfaces=(0.5, 1.0))
    vmec_mirror = workflow.VMECMirrorRatio(threshold=0.3, surfaces=(0.5, 1.0))
    generic_elongation = workflow.MaxElongation(threshold=5.0, qi_options=options)
    shared_qi_mirror = workflow.MirrorRatio(threshold=0.3, qi_options=options)

    assert generic_mirror.requires_qi_field is False
    assert vmec_mirror.requires_qi_field is False
    assert generic_elongation.requires_qi_field is False
    assert shared_qi_mirror.requires_qi_field is True

    generic_problem = workflow.LeastSquaresProblem.from_tuples(
        [
            (generic_mirror.J, 0.0, 1.0),
            (vmec_mirror.J, 0.0, 1.0),
            (generic_elongation.J, 0.0, 1.0),
        ]
    )
    assert generic_problem.is_qi is False
    assert generic_problem.qi_options is None
    assert generic_problem.objective_names == ("mirror_ratio", "mirror_ratio", "max_elongation")

    qi_problem = workflow.LeastSquaresProblem.from_tuples([(shared_qi_mirror.J, 0.0, 1.0)])
    assert qi_problem.is_qi is True
    assert qi_problem.qi_options is options
    assert qi_problem.qi_objective_names == ("mirror_ratio",)


def test_vmec_mirror_ratio_uses_vmec_field_without_boozer(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    calls = []

    def fake_b_cartesian_from_state(*args, **kwargs):
        calls.append(kwargs["s_index"])
        return np.asarray(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        )

    monkeypatch.setattr(workflow, "b_cartesian_from_state", fake_b_cartesian_from_state)
    ctx = SimpleNamespace(static=SimpleNamespace(s=np.asarray([0.0, 0.5, 1.0])), indata=object(), signgs=1)
    objective = workflow.VMECMirrorRatio(threshold=0.2, surfaces=(1.0,))

    residual = objective.J(ctx, state=object())

    np.testing.assert_allclose(np.asarray(residual), np.asarray([0.4]), rtol=1e-12, atol=1e-12)
    assert calls == [2]
    assert objective.total(ctx, state=object()) == pytest.approx(0.16)
    with pytest.raises(ValueError, match="target=0"):
        objective.to_objective_term(target=1.0, residual_weight=1.0)


def test_vmec_mirror_ratio_smooth_penalty_weights_multiple_surfaces(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    calls = []

    def fake_b_cartesian_from_state(*args, **kwargs):
        s_index = int(kwargs["s_index"])
        calls.append(s_index)
        bmag_by_surface = {
            0: np.asarray([[1.0, 2.0], [1.5, 1.25]]),
            2: np.asarray([[1.0, 4.0], [2.0, 3.0]]),
        }
        return np.stack(
            [
                bmag_by_surface[s_index],
                np.zeros_like(bmag_by_surface[s_index]),
                np.zeros_like(bmag_by_surface[s_index]),
            ],
            axis=-1,
        )

    monkeypatch.setattr(workflow, "b_cartesian_from_state", fake_b_cartesian_from_state)

    ctx = SimpleNamespace(static=SimpleNamespace(s=np.asarray([0.0, 0.5, 1.0])), indata=object(), signgs=1)
    objective = workflow.VMECMirrorRatio(
        threshold=0.45,
        surfaces=(0.0, 1.0),
        smooth_penalty=0.1,
        normalize_surfaces=True,
    )

    out = objective._evaluate_state(ctx, state=object())

    expected_ratios = np.asarray([1.0 / 3.0, 3.0 / 5.0])
    expected_penalty = 0.1 * np.logaddexp((expected_ratios - 0.45) / 0.1, 0.0)
    expected_residuals = expected_penalty * np.sqrt(0.5)

    assert calls == [0, 2]
    np.testing.assert_allclose(np.asarray(out["mirror_ratio"]), expected_ratios, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(out["penalty"]), expected_penalty, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(out["residuals1d"]), expected_residuals, rtol=1.0e-12, atol=1.0e-12)
    assert float(out["total"]) == pytest.approx(float(np.sum(expected_residuals**2)))


def test_vmec_mirror_ratio_accepts_boozer_style_sampling_aliases() -> None:
    import vmec_jax.optimization_workflow as workflow

    objective = workflow.VMECMirrorRatio(
        threshold=0.3,
        surfaces=(0.5, 1.0),
        ntheta=96,
        nphi=64,
    )

    assert objective.requires_qi_field is False
    assert objective.requested_ntheta == 96
    assert objective.requested_nzeta == 64

    objective = workflow.VMECMirrorRatio(
        threshold=0.3,
        surfaces=(0.5, 1.0),
        nzeta=32,
    )
    assert objective.requested_ntheta is None
    assert objective.requested_nzeta == 32

    with pytest.raises(ValueError, match="either nphi or nzeta"):
        workflow.VMECMirrorRatio(threshold=0.3, nphi=16, nzeta=32)


def test_mirror_ratio_objective_terms_cover_surface_selection_and_prepared_paths(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(
        static=SimpleNamespace(cfg=SimpleNamespace(mpol=2, ntor=1, ntheta=8, nzeta=8, nfp=2, lasym=False)),
        indata="indata",
        signgs=-1,
        flux="flux",
        pressure="pressure",
    )
    calls: list[dict[str, object]] = []

    def fake_penalty(**kwargs):
        calls.append(kwargs)
        return {
            "residuals1d": np.asarray([0.25, 0.5]),
            "total": 0.3125,
            "mirror_ratio": np.asarray([0.35, 0.45]),
        }

    monkeypatch.setattr(workflow, "mirror_ratio_penalty_from_state", fake_penalty)
    monkeypatch.setattr(
        workflow.MirrorRatio,
        "_prepare_boozer_constants",
        lambda self, ctx_arg: ("constants", "grids"),
    )

    with pytest.raises(ValueError, match="surfaces"):
        workflow.MirrorRatio(threshold=0.3, surfaces=())._selected_surfaces_and_weights()
    with pytest.raises(IndexError, match="outside"):
        workflow.MirrorRatio(threshold=0.3, surfaces=(0.25,), surface_index=3)._selected_surfaces_and_weights()

    selected = workflow.MirrorRatio(threshold=0.3, surfaces=(0.25, 0.5), surface_index=-1)
    assert selected._selected_surfaces_and_weights() == ((0.5,), None)
    unnormalised = workflow.MirrorRatio(threshold=0.3, surfaces=(0.25, 0.5), normalize_surfaces=False)
    assert unnormalised._selected_surfaces_and_weights() == ((0.25, 0.5), None)

    objective = workflow.MirrorRatio(threshold=0.3, surfaces=(0.25, 0.5), mboz=6, nboz=7, jit_booz=False)
    np.testing.assert_allclose(objective.J(ctx, "state"), [0.25, 0.5])
    assert objective.total(ctx, "state") == pytest.approx(0.3125)
    assert calls[-1]["surfaces"] == (0.25, 0.5)
    assert calls[-1]["weights"] == [0.5, 0.5]
    assert calls[-1]["mboz"] == 6
    assert calls[-1]["nboz"] == 7
    assert calls[-1]["jit_booz"] is False

    qi_bound = workflow.MirrorRatio(
        threshold=0.3,
        qi_options=workflow.QuasiIsodynamicOptions(surfaces=(0.5,)),
    )
    with pytest.raises(RuntimeError, match="inside a QI solve"):
        qi_bound.J(ctx, "state")
    with pytest.raises(RuntimeError, match="inside a QI solve"):
        qi_bound.total(ctx, "state")

    with pytest.raises(ValueError, match="target=0"):
        objective.to_objective_term(target=1.0, residual_weight=1.0)

    prepared = objective.to_objective_term(target=0.0, residual_weight=2.0).bind(ctx)
    np.testing.assert_allclose(prepared.residual(ctx, "state"), [0.5, 1.0])
    assert prepared.total(ctx, "state") == pytest.approx(1.25)
    assert calls[-1]["booz_constants"] == "constants"
    assert calls[-1]["booz_grids"] == "grids"

    constraint = objective.to_constraint_term().bind(ctx)
    np.testing.assert_allclose(constraint.residual(ctx, "state"), [0.03535534, 0.10606602], rtol=1e-6)


def test_vmec_mirror_ratio_surface_selection_errors_and_unnormalized_weights(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(static=SimpleNamespace(s=np.asarray([0.0, 0.5, 1.0])), indata=object(), signgs=1)
    objective = workflow.VMECMirrorRatio(threshold=0.2, surfaces=(0.0, 1.0), normalize_surfaces=False)

    monkeypatch.setattr(
        workflow,
        "b_cartesian_from_state",
        lambda *args, **kwargs: np.asarray([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
    )

    indices, weights = objective._selected_surface_indices_and_weights(ctx)
    assert indices == [0, 2]
    np.testing.assert_allclose(np.asarray(weights), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(objective.J(ctx, "state")), [1.0 / 3.0 - 0.2, 1.0 / 3.0 - 0.2])

    with pytest.raises(ValueError, match="surfaces"):
        workflow.VMECMirrorRatio(threshold=0.2, surfaces=())._selected_surface_indices_and_weights(ctx)
    with pytest.raises(IndexError, match="outside"):
        workflow.VMECMirrorRatio(threshold=0.2, surfaces=(0.5,), surface_index=-2)._selected_surface_indices_and_weights(ctx)


def test_max_elongation_objective_term_and_constraint_paths(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    calls: list[dict[str, object]] = []

    def fake_elongation(**kwargs):
        calls.append(kwargs)
        return {"residuals1d": np.asarray([0.2]), "total": 0.04, "max_elongation": 8.5}

    monkeypatch.setattr(workflow, "max_elongation_penalty_from_state", fake_elongation)
    ctx = SimpleNamespace(static="static")
    objective = workflow.MaxElongation(threshold=8.0, ntheta=9, nphi=7, smooth_extrema=0.1, smooth_penalty=0.2)

    np.testing.assert_allclose(objective.J(ctx, "state"), [0.2])
    assert objective.total(ctx, "state") == pytest.approx(0.04)
    assert calls[-1]["smooth_penalty"] == pytest.approx(0.2)
    term = objective.to_objective_term(target=0.0, residual_weight=3.0)
    np.testing.assert_allclose(term.residual(ctx, "state"), [0.6])
    assert term.total(ctx, "state") == pytest.approx(0.36)
    constraint = objective.to_constraint_term()
    np.testing.assert_allclose(constraint.residual(ctx, "state"), [0.5])
    assert calls[-1]["smooth_penalty"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="target=0"):
        objective.to_objective_term(target=1.0, residual_weight=1.0)


def test_qi_and_qs_object_wrappers_build_terms_without_solves(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    options = workflow.QuasiIsodynamicOptions(surfaces=np.asarray([0.5]))
    ctx = SimpleNamespace(static="static", indata="indata", signgs=-1, flux="flux", pressure="pressure")

    qi_wrappers = [
        (workflow.QuasiIsodynamicResidual(options), "qi"),
        (
            workflow.QuasiIsodynamicResidualCeiling(
                maximum=1.5,
                smooth_penalty=0.1,
                qi_options=options,
            ),
            "qi_ceiling",
        ),
        (
            workflow.MirrorRatio(
                threshold=0.2,
                ntheta=5,
                nphi=7,
                surface_index=-1,
                phimin=0.1,
                smooth_extrema=0.2,
                smooth_penalty=0.3,
                normalize_surfaces=False,
                qi_options=options,
            ),
            "mirror_ratio",
        ),
        (
            workflow.BoozerBTarget(
                target_bmnc=np.ones((1, 2)),
                target_bmns=np.zeros((1, 2)),
                normalize=False,
                include_b00=True,
                qi_options=options,
            ),
            "boozer_b_target",
        ),
        (
            workflow.MaxElongation(
                threshold=5.0,
                ntheta=6,
                nphi=4,
                smooth_extrema=0.2,
                smooth_penalty=0.3,
                qi_options=options,
            ),
            "max_elongation",
        ),
    ]
    for wrapper, expected_name in qi_wrappers:
        if expected_name != "max_elongation":
            with pytest.raises(RuntimeError, match="must be evaluated inside"):
                wrapper.J(ctx, "state")
        else:
            monkeypatch.setattr(
                workflow,
                "max_elongation_penalty_from_state",
                lambda **_kwargs: {
                    "residuals1d": workflow.jnp.asarray([0.0]),
                    "total": workflow.jnp.asarray(0.0),
                },
            )
            np.testing.assert_allclose(np.asarray(wrapper.J(ctx, "state")), [0.0])
        if hasattr(wrapper, "to_qi_term"):
            assert wrapper.to_qi_term(2.0).name == expected_name
        else:
            assert expected_name == "max_elongation"

    assert qi_wrappers[2][0].to_constraint_qi_term().name == "mirror_ratio_constraint"
    assert qi_wrappers[4][0].to_constraint_qi_term().name == "max_elongation_constraint"

    def fake_quasisymmetry_ratio_residual_from_state(**kwargs):
        assert kwargs["surfaces"] == (0.5,)
        assert kwargs["helicity_m"] == 1
        assert kwargs["helicity_n"] == -1
        return {"residuals1d": np.asarray([0.25, 0.5]), "total": 0.3125}

    monkeypatch.setattr(
        workflow,
        "quasisymmetry_ratio_residual_from_state",
        fake_quasisymmetry_ratio_residual_from_state,
    )
    qs = workflow.QuasisymmetryRatioResidual(helicity_m=1, helicity_n=-1, surfaces=(0.5,))
    np.testing.assert_allclose(np.asarray(qs.J(ctx, "state")), [0.25, 0.5])
    assert qs.total(ctx, "state") == pytest.approx(0.3125)
    with pytest.raises(ValueError, match="target=0"):
        qs.to_objective_term(target=1.0, residual_weight=1.0)
    term = qs.to_objective_term(target=0.0, residual_weight=2.0)
    np.testing.assert_allclose(np.asarray(term.residual(ctx, "state")), [0.5, 1.0])
    assert term.total(ctx, "state") == pytest.approx(1.25)

    lgradb = workflow.LgradB(threshold=0.3, s_index=-2, ntheta=3, nphi=5, smooth_penalty=0.1)
    with pytest.raises(ValueError, match="target=0"):
        lgradb.to_objective_term(target=1.0, residual_weight=1.0)
    assert lgradb.to_qi_term(1.5).name == "LgradB"


def test_state_objective_wrappers_use_monkeypatched_state_helpers(monkeypatch) -> None:
    import vmec_jax.optimization_workflow as workflow

    ctx = SimpleNamespace(
        static=SimpleNamespace(s=np.asarray([0.0, 0.5, 1.0])),
        indata="indata",
        signgs=1,
        flux="flux",
        pressure="pressure",
    )

    def fake_finite_beta_scalars_from_state(**kwargs):
        assert kwargs["state"] == "state"
        assert kwargs["static"] is ctx.static
        return {"vp": np.asarray([4.0, 2.0]), "volavgB": 2.5, "betatotal": 0.03}

    monkeypatch.setattr(workflow, "finite_beta_scalars_from_state", fake_finite_beta_scalars_from_state)
    monkeypatch.setattr(workflow, "magnetic_well_from_vp", lambda vp: 0.05)

    well = workflow.MagneticWell(minimum=0.10, softness=0.01)
    assert float(np.asarray(well.J(ctx, "state"))) > 0.0
    with pytest.raises(ValueError, match="target=0"):
        well.to_objective_term(target=1.0, residual_weight=1.0)
    np.testing.assert_allclose(
        np.asarray(workflow.VolavgB().to_objective_term(target=2.0, residual_weight=3.0).residual(ctx, "state")),
        [1.5],
    )
    np.testing.assert_allclose(
        np.asarray(workflow.BetaTotal().to_objective_term(target=0.01, residual_weight=10.0).residual(ctx, "state")),
        [0.2],
    )

    def fake_mercier_terms_from_state(**kwargs):
        assert kwargs["state"] == "state"
        return {
            "DMerc": np.asarray([0.0, -0.1, 0.2]),
            "D_R": np.asarray([0.0, 0.05, -0.02]),
            "H": np.asarray([0.0, 0.01, 0.02]),
            "shear": np.asarray([0.0, 0.4, 0.5]),
            "jdotb": np.asarray([1.0, 2.0, 3.0]),
            "bdotb": np.asarray([4.0, 5.0, 6.0]),
            "bdotgradv": np.asarray([7.0, 8.0, 9.0]),
            "sqrtg": np.asarray([[1.0, 0.0], [2.0, 4.0], [5.0, 6.0]]),
            "itheta": np.asarray([[2.0, 8.0], [2.0, 8.0], [10.0, 12.0]]),
            "izeta": np.asarray([[6.0, 16.0], [6.0, 16.0], [20.0, 24.0]]),
            "torcur": np.asarray([0.0, 1.0, 2.0]),
            "ip": np.asarray([3.0, 4.0, 5.0]),
        }

    monkeypatch.setattr(workflow, "mercier_terms_from_state", fake_mercier_terms_from_state)

    dmerc = workflow.DMerc(minimum=0.0, softness=0.01, mmax_force=2, nmax_force=3)
    assert np.asarray(dmerc.J(ctx, "state")).shape == (1,)
    with pytest.raises(ValueError, match="target=0"):
        dmerc.to_objective_term(target=1.0, residual_weight=1.0)

    glasser = workflow.GlasserResistiveInterchange(maximum=0.0, softness=0.01, mmax_force=2, nmax_force=3)
    assert np.asarray(glasser.J(ctx, "state")).shape == (1,)
    with pytest.raises(ValueError, match="target=0"):
        glasser.to_objective_term(target=1.0, residual_weight=1.0)

    np.testing.assert_allclose(np.asarray(workflow.JDotB(normalize=2.0).J(ctx, "state")), [1.0])
    np.testing.assert_allclose(
        np.asarray(workflow.BDotB(surfaces=(0.0, 1.0), normalize=2.0).J(ctx, "state")),
        [2.0, 3.0],
    )
    np.testing.assert_allclose(np.asarray(workflow.ToroidalCurrentGradient(normalize=2.0).J(ctx, "state")), [2.0])

    j_vector = workflow.JVector(surfaces=(0.0,), normalize=1.0)
    np.testing.assert_allclose(np.asarray(j_vector.J(ctx, "state")), [2.0, 6.0, 0.0, 0.0])

    monkeypatch.setattr(
        workflow,
        "b_cartesian_from_state",
        lambda state, static, **kwargs: np.asarray([[1.0, 2.0, 3.0]]) if state == "state" and static is ctx.static else None,
    )
    np.testing.assert_allclose(np.asarray(workflow.BVector(s_index=-2, normalize=2.0).J(ctx, "state")), [0.5, 1.0, 1.5])

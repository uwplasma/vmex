from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax as vj
import vmec_jax.qi_optimization as qio
from vmec_jax.namelist import InData


def _configure(tmp_path: Path) -> SimpleNamespace:
    qi_options = SimpleNamespace(
        mboz=7,
        nboz=8,
        nphi=9,
        nalpha=10,
        n_bounce=11,
        include_bounce_endpoints=True,
        softness=0.02,
        width_weight=1.0,
        branch_width_weight=2.0,
        branch_width_softness=0.03,
        profile_weight=0.4,
        shuffle_profile_weight=0.5,
        shuffle_profile_softness=0.06,
        shuffle_profile_nphi_out=13,
        weighted_shuffle_profile_weight=0.7,
        weighted_shuffle_profile_softness=0.08,
        aligned_profile_weight=0.9,
        aligned_profile_softness=0.1,
        aligned_profile_trap_level=0.2,
        aligned_profile_trap_softness=0.3,
        phimin=0.0,
    )
    qio.configure(
        {
            "ALPHA": 2.5,
            "CONTINUATION_NFEV": 2,
            "INNER_MAX_ITER": 3,
            "JIT_BOOZ": False,
            "MAX_ELONGATION": 6.0,
            "MAX_MIRROR_RATIO": 0.35,
            "MAX_MODE": 3,
            "MAX_NFEV": 4,
            "METHOD": "scipy_matrix_free",
            "MIN_VMEC_MODE": 5,
            "MIRROR_SURFACE_INDEX": -1,
            "MIRROR_WEIGHT": 2.0,
            "OPT_QI_RESOLUTION": {"nphi": 17},
            "OUTPUT_DIR": tmp_path,
            "QI_GATE_LEGACY_MAX": 2.0e-3,
            "QI_GATE_SMOOTH_MAX": 1.0e-3,
            "QI_OPTIONS": qi_options,
            "QI_WEIGHT": 10.0,
            "SOLVER_DEVICE": None,
            "STAGE_MODES": (1, 2, 3),
            "STAGE_REPEATS": 2,
            "SURFACES": np.asarray([0.5, 1.0]),
            "TARGET_ABS_IOTA_MIN": 0.41,
            "TARGET_ASPECT": 6.0,
            "TRIAL_FTOL": 1.0e-9,
            "USE_ESS": True,
            "USE_MODE_CONTINUATION": True,
        }
    )
    return qi_options


def test_qi_helper_scalar_parsers_and_scores(tmp_path: Path) -> None:
    _configure(tmp_path)

    assert qio._diagnostic_float({"x": 1.25}, "x") == 1.25
    assert np.isnan(qio._diagnostic_float({}, "missing"))
    assert qio._finite_or_inf("not-a-float") == float("inf")
    assert qio._finite_or_none("not-a-float") is None
    assert qio._parse_float_sequence("1, 2 3", name="x") == (1.0, 2.0, 3.0)
    assert qio._parse_float_sequence("", name="x") is None
    with pytest.raises(ValueError, match="x must be"):
        qio._parse_float_sequence("bad", name="x")

    assert qio.target_helicity_seed_terms(max_mode=0) == ()
    assert qio.target_helicity_seed_terms(max_mode=1, amplitude=0.0) == ()
    terms = qio.target_helicity_seed_terms(max_mode=1, amplitude=2.0e-5)
    assert ("RBC", (1, 0), 2.0e-5) in terms

    payload = qio._jsonable({"a": np.asarray([1.0, np.nan]), "p": Path("x"), "s": np.float64(2.0)})
    assert payload == {"a": [1.0, None], "p": "x", "s": 2.0}
    assert qio._partial_diagnostics_from_history({"objective_final": 1.0, "iota_final": 0.5}, {}) == {
        "objective_final": 1.0,
        "mean_iota": 0.5,
        "partial": True,
        "diagnostics_pending": True,
    }
    assert qio._partial_diagnostics_from_history({"objective_final": 1.0}, {"existing": 2.0}) == {
        "existing": 2.0,
        "objective_final": 1.0,
        "partial": True,
    }

    targets = SimpleNamespace(
        smooth_qi_max=1.0,
        legacy_qi_max=2.0,
        abs_iota_min=0.4,
        mirror_ratio_max=0.3,
        max_elongation=8.0,
        target_aspect=6.0,
    )
    score = qio.basin_prefilter_score(
        {
            "qi_smooth_total": 0.5,
            "qi_legacy_total": 1.0,
            "qi_mirror_ratio_max": 0.6,
            "qi_max_elongation": 10.0,
            "mean_iota": 0.2,
            "aspect": 7.0,
        },
        targets,
        {"iota_gap_weight": 2.0, "mirror_weight": 3.0, "elongation_weight": 4.0, "aspect_weight": 5.0},
    )
    assert score > 0.0
    assert qio.boundary_reference_preconditioner_score({"qi_rank_score": 1.0, "qi_constraint_score": 2.0}) > 100.0
    aspect_failed_score = qio.boundary_reference_preconditioner_score(
        {"qi_rank_score": 1.0, "qi_constraint_score": 0.0, "aspect_relative_error": 0.4}
    )
    aspect_ok_score = qio.boundary_reference_preconditioner_score(
        {"qi_rank_score": 1.0, "qi_constraint_score": 0.0, "aspect_relative_error": 0.02}
    )
    assert aspect_failed_score > aspect_ok_score
    assert qio.boundary_reference_record_is_qi_safe(
        {"mirror": 0.2, "mean_iota": -0.5, "aspect": 5.4},
        max_mirror_ratio=0.3,
        abs_iota_min=0.4,
        target_aspect=5.0,
    )
    assert not qio.boundary_reference_record_is_qi_safe(
        {"mirror": 0.4, "mean_iota": -0.5, "aspect": 5.4},
        max_mirror_ratio=0.3,
        abs_iota_min=0.4,
        target_aspect=5.0,
    )
    assert not qio.boundary_reference_record_is_qi_safe(
        {"mirror": 0.2, "mean_iota": -0.5, "aspect": 7.0},
        max_mirror_ratio=0.3,
        abs_iota_min=0.4,
        target_aspect=5.0,
    )


def test_qi_engineering_constraint_tuples_support_augmented_lagrangian() -> None:
    mirror = vj.VMECMirrorRatio(threshold=0.35, surfaces=(1.0,))
    elongation = vj.MaxElongation(threshold=8.0)

    regular = qio.qi_engineering_constraint_tuples(mirror, elongation, None, 2.0, 3.0)
    assert [term[0].__self__ for term in regular] == [mirror, elongation]
    assert [term[2] for term in regular] == [2.0, 3.0]

    al_terms = qio.qi_engineering_constraint_tuples(
        mirror,
        elongation,
        {
            "use_augmented_lagrangian_constraints": True,
            "al_mirror_multiplier": 0.1,
            "al_mirror_penalty": 25.0,
            "al_mirror_weight": 4.0,
            "al_elongation_multiplier": 0.2,
            "al_elongation_penalty": 9.0,
            "al_elongation_weight": 5.0,
            "al_constraint_softness": 1.0e-3,
        },
        0.0,
        0.0,
    )
    owners = [term[0].__self__ for term in al_terms]
    assert [owner.name for owner in owners] == ["al_mirror", "al_elongation"]
    assert [owner.penalty for owner in owners] == [25.0, 9.0]
    assert [owner.multiplier for owner in owners] == [0.1, 0.2]
    assert [term[2] for term in al_terms] == [4.0, 5.0]


def test_qi_mirror_objective_for_stage_can_use_vmec_or_boozer_backend() -> None:
    options = vj.QuasiIsodynamicOptions(surfaces=(0.5, 1.0), mboz=5, nboz=5)

    fast = qio.qi_mirror_objective_for_stage(
        None,
        qi_options=options,
        threshold=0.35,
        surfaces=options.surfaces,
        surface_index=None,
    )
    assert isinstance(fast, vj.VMECMirrorRatio)

    boozer = qio.qi_mirror_objective_for_stage(
        {"mirror_backend": "boozer", "mirror_surface_index": -1},
        qi_options=options,
        threshold=0.35,
        surfaces=options.surfaces,
        surface_index=None,
    )
    assert isinstance(boozer, vj.MirrorRatio)
    assert boozer.requires_qi_field
    assert boozer.surface_index == -1

    scalar = qio.qi_mirror_objective_for_stage(
        {"mirror_backend": "boozer_scalar", "mirror_mboz": 18, "mirror_nboz": 17},
        qi_options=options,
        threshold=0.35,
        surfaces=options.surfaces,
        surface_index=None,
    )
    assert isinstance(scalar, vj.MirrorRatio)
    assert not scalar.requires_qi_field
    assert (scalar.mboz, scalar.nboz) == (18, 17)


def test_qi_cli_override_loads_mirror_ramp_stages_json(tmp_path: Path) -> None:
    stages_path = tmp_path / "stages.json"
    stages_path.write_text('[{"name": "cleanup", "max_nfev": 5, "mirror_weight": 20.0}]')
    namespace = {
        "MAX_MODE": 3,
        "USE_MODE_CONTINUATION": True,
        "CONTINUATION_NFEV": 2,
        "STAGE_REPEATS": 1,
        "STAGE_MODE_POLICY": "lower",
    }

    qio.apply_qi_example_cli_overrides(namespace, ["--mirror-ramp-stages-json", str(stages_path)])

    assert namespace["MIRROR_RAMP_STAGES"] == ({"name": "cleanup", "max_nfev": 5, "mirror_weight": 20.0},)


def test_qi_lower_repeat_stage_policy_and_cli_controls() -> None:
    modes = qio.qi_stage_modes(
        max_mode=3,
        use_mode_continuation=True,
        continuation_nfev=2,
        repeats=2,
        policy="lower-repeat",
    )
    assert modes == [1, 1, 2, 2, 3, 3]

    direct = qio.qi_stage_modes(
        max_mode=3,
        use_mode_continuation=False,
        continuation_nfev=0,
        repeats=4,
        policy="lower-repeat",
    )
    assert direct == [3]

    namespace = {
        "MAX_MODE": 3,
        "USE_MODE_CONTINUATION": True,
        "CONTINUATION_NFEV": 2,
        "STAGE_REPEATS": 1,
        "STAGE_MODE_POLICY": "lower",
    }
    qio.apply_qi_example_cli_overrides(
        namespace,
        [
            "--stage-mode-policy",
            "lower-repeat",
            "--stage-repeats",
            "2",
            "--scipy-lsmr-maxiter",
            "5",
            "--scalar-cost-only-trials",
        ],
    )
    assert namespace["STAGE_MODES"] == [1, 1, 2, 2, 3, 3]
    assert namespace["SCIPY_LSMR_MAXITER"] == 5
    assert namespace["SCALAR_COST_ONLY_TRIALS"] is True


def test_explicit_qi_context_overrides_legacy_globals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    qi_options = _configure(tmp_path / "legacy")
    ctx = qio.make_qi_optimization_context(
        max_mode=5,
        output_dir=tmp_path / "explicit",
        opt_qi_resolution={"nphi": 19},
        surfaces=np.asarray([0.25]),
    )
    assert ctx.max_mode == 5
    assert ctx.output_dir == tmp_path / "explicit"

    monkeypatch.setattr(qio, "MAX_MODE", 99, raising=False)
    monkeypatch.setattr(qio, "STAGE_REPEATS", 99, raising=False)
    monkeypatch.setattr(qio, "OUTPUT_DIR", tmp_path / "poisoned", raising=False)
    monkeypatch.setattr(
        qio,
        "qi_stage_modes",
        lambda **kwargs: [kwargs["max_mode"], kwargs["repeats"], kwargs["policy"]],
    )

    assert qio.stage_modes_for({"stage_repeats": 2, "stage_mode_policy": "repeat"}, ctx=ctx) == [5, 2, "repeat"]
    opt = qio.make_basin_prefilter_options({}, ctx=ctx)
    assert opt.nphi == 19
    assert np.asarray(opt.surfaces).tolist() == [0.25]
    assert opt.include_bounce_endpoints == qi_options.include_bounce_endpoints

    qio.write_qi_stage_checkpoint(
        tmp_path / "stage",
        stage_index=1,
        stage_name="explicit",
        stage_modes=[1],
        stage_result=SimpleNamespace(history={"objective_final": 0.5}),
        diagnostics={},
        ctx=ctx,
    )
    assert (tmp_path / "explicit" / "stage_checkpoint.json").exists()
    assert not (tmp_path / "poisoned" / "stage_checkpoint.json").exists()


def test_target_helicity_seed_preconditioner_writes_seeded_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure(tmp_path)
    source = InData(
        scalars={"NFP": 2},
        indexed={"RBC": {(1, 0): "bad-existing"}, "ZBS": {}},
        source_path="input.seed",
    )
    written: dict[str, object] = {}

    monkeypatch.setattr(qio.vj, "read_indata", lambda _path: source)

    def fake_write(path, indata):
        written["path"] = Path(path)
        written["indata"] = indata

    monkeypatch.setattr(qio.vj, "write_indata", fake_write)

    out = qio.run_target_helicity_seed_preconditioner(
        "input.seed",
        tmp_path,
        {"terms": [("rbc", (1, 0), 1.0e-5), ("zbs", (1, 0), 2.0e-5)]},
    )

    assert out == tmp_path / "target_helicity_seed" / "input.target_helicity_seed"
    assert written["path"] == out
    seeded = written["indata"]
    assert seeded.indexed["RBC"][(1, 0)] == 1.0e-5
    assert seeded.indexed["ZBS"][(1, 0)] == 2.0e-5
    metadata = json.loads((tmp_path / "target_helicity_seed" / "metadata.json").read_text())
    assert metadata["inserted"][0] == {"family": "RBC", "m": 0, "n": 1, "value": 1.0e-5}
    assert qio.run_target_helicity_seed_preconditioner("input.seed", tmp_path, {"enabled": False}) == Path("input.seed")


def test_run_basin_prefilter_saves_candidate_inputs_and_records_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(tmp_path)

    class Candidate:
        def __init__(self, label: str, params: list[float]):
            self.label = label
            self.params = np.asarray(params, dtype=float)

        def as_record(self, names):
            return {
                "label": self.label,
                "params": self.params.tolist(),
                "named_params": dict(zip(names, self.params, strict=True)),
            }

    class Optimizer:
        def _solve_forward(self, params, *, trial):
            assert trial is True
            if float(params[0]) < 0.0:
                raise RuntimeError("boom")
            return SimpleNamespace(label="state")

        def save_input(self, path, params):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"params={np.asarray(params).tolist()}\n")

    stage = SimpleNamespace(
        specs=[SimpleNamespace(name="rc01")],
        optimizer=Optimizer(),
        ctx=SimpleNamespace(static="static", indata="indata", signgs=-1, flux="flux", pressure="pressure"),
    )

    monkeypatch.setattr(qio, "boundary_param_names", lambda _specs: ["rc01"])
    monkeypatch.setattr(qio, "create_x_scale", lambda _specs, *, alpha: np.asarray([alpha], dtype=float))
    monkeypatch.setattr(qio, "make_basin_prefilter_options", lambda _config, *, ctx=None: SimpleNamespace(surfaces=[1.0]))
    monkeypatch.setattr(
        qio,
        "_load_basin_prefilter_tools",
        lambda: (
            lambda **kwargs: SimpleNamespace(**kwargs),
            lambda **_kwargs: [Candidate("bad:one", [-1.0]), Candidate("good:two", [2.0])],
            lambda records, *, targets: records,
            lambda rows, path: Path(path).write_text(f"{len(rows)} rows\n"),
            lambda **_kwargs: stage,
        ),
    )
    monkeypatch.setattr(
        qio.vj,
        "qi_diagnostics_from_state",
        lambda **_kwargs: {
            "qi_smooth_total": 1.0e-4,
            "qi_legacy_total": 2.0e-4,
            "qi_mirror_ratio_max": 0.2,
            "qi_max_elongation": 4.0,
            "mean_iota": 0.5,
            "aspect": 6.0,
        },
    )

    selected = qio.run_basin_prefilter("input.seed", tmp_path, {"enabled": True, "alpha": 2.0})

    selected_path = tmp_path / "basin_prefilter" / "candidates" / "good_two" / "input.candidate"
    assert selected == selected_path
    assert selected_path.exists()
    rows = json.loads((tmp_path / "basin_prefilter" / "candidates.json").read_text())
    failed = next(row for row in rows if row["label"] == "bad:one")
    successful = next(row for row in rows if row["label"] == "good:two")
    assert failed["error"] == "RuntimeError: boom"
    assert successful["input_path"] == str(selected_path)
    assert successful["prefilter_rank"] == 1


def test_boundary_reference_preconditioner_raises_when_all_candidates_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(tmp_path)
    monkeypatch.setattr(qio.vj, "read_indata", lambda _path: InData(scalars={}, indexed={}))
    monkeypatch.setattr(
        qio.vj,
        "interpolate_indata_boundary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("interpolation failed")),
    )

    with pytest.raises(RuntimeError, match="no successful candidates"):
        qio.run_boundary_reference_preconditioner(
            "input.seed",
            tmp_path,
            {
                "enabled": True,
                "reference_input": "input.reference",
                "lambdas": (0.5, 1.0),
            },
        )

    summary = json.loads((tmp_path / "boundary_reference_preconditioner" / "summary.json").read_text())
    assert [row["lambda"] for row in summary] == [0.5, 1.0]
    for row in summary:
        assert row["selected"] is False
        assert row["score"] == float("inf")
        assert row["error"] == "RuntimeError: interpolation failed"


def test_make_options_and_diagnostics_helpers_use_configured_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    qi_options = _configure(tmp_path)
    built_options: list[SimpleNamespace] = []

    def fake_options(**kwargs):
        opt = SimpleNamespace(**kwargs)
        built_options.append(opt)
        return opt

    monkeypatch.setattr(qio.vj, "QIDiagnosticOptions", fake_options)
    monkeypatch.setattr(
        qio.vj,
        "qi_diagnostics_from_state",
        lambda **kwargs: {
            "qi_smooth_total": 5.0e-4,
            "qi_legacy_total": 6.0e-4,
            "qi_mirror_ratio_max": 0.2,
            "qi_max_elongation": 4.0,
            "mean_iota": 0.5,
            "aspect": 6.0,
        },
    )
    monkeypatch.setattr(
        qio,
        "annotate_qi_seed_suitability",
        lambda diagnostics, *, targets: {**diagnostics, "target_aspect": targets.target_aspect},
    )

    opt = qio.make_basin_prefilter_options({"mirror_ntheta": 12, "nphi": 99})
    assert opt.nphi == 17
    assert opt.mirror_ntheta == 12
    assert opt.include_bounce_endpoints == qi_options.include_bounce_endpoints

    stage_result = SimpleNamespace(
        final_optimizer=SimpleNamespace(static="static", indata="indata", signgs=-1),
        final_state="state",
    )
    assert qio.qi_diagnostics_for_result(stage_result, mirror_threshold=0.3, mirror_surface_index=-1)["target_aspect"] == 6.0
    run = SimpleNamespace(state="state", static="static", indata="indata", signgs=-1)
    assert (
        qio.qi_diagnostics_for_run(
            run,
            mirror_threshold=0.3,
            mirror_surface_index=-1,
            target_aspect=7.0,
            abs_iota_min=0.45,
            max_elongation=5.0,
            resolution={"mboz": 3},
        )["target_aspect"]
        == 7.0
    )
    assert built_options[-1].mboz == 3


def test_stage_checkpoint_modes_and_promotion_rules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure(tmp_path)
    monkeypatch.setattr(qio, "qi_stage_modes", lambda **kwargs: [kwargs["max_mode"], kwargs["repeats"]])

    assert qio.stage_modes_for({"stage_modes": [1, 3]}) == [1, 3]
    assert [mode.mode for mode in qio.stage_modes_for({"stage_mode_limits": [1]})] == [1]
    assert qio.stage_modes_for({"stage_repeats": 4}) == [3, 4]
    assert qio.promotion_score({"qi_rank_score": 1.0, "qi_constraint_score": 4.0}) == 112.0
    assert qio.engineering_promotion_score({"qi_rank_score": 1.0, "qi_constraint_score": 4.0, "qi_mirror_ratio_max": 0.5}) == 1103.0
    aspect_far = {"qi_rank_score": 1.0, "qi_constraint_score": 0.0, "aspect_relative_error": 0.36}
    aspect_near = {"qi_rank_score": 1.0, "qi_constraint_score": 0.0, "aspect_relative_error": 0.04}
    assert qio.promotion_score(aspect_near) < qio.promotion_score(aspect_far)
    assert qio.engineering_promotion_score(
        {**aspect_near, "qi_mirror_ratio_max": 0.3}
    ) < qio.engineering_promotion_score({**aspect_far, "qi_mirror_ratio_max": 0.3})

    promoted = qio.stage_promotes_candidate(
        {"accept_if_iota_improves": True, "iota_improvement_min": 0.05},
        {"mean_iota": 0.5, "qi_smooth_total": 1.0e-3, "qi_legacy_total": 1.0e-3},
        {"mean_iota": 0.4, "qi_smooth_total": 1.0e-3, "qi_legacy_total": 1.0e-3},
    )
    assert promoted["qi_cleanup_promoted"] is True
    rejected = qio.stage_promotes_candidate(
        {
            "accept_if_rank_improves": True,
            "accept_if_engineering_score_improves": True,
            "engineering_score_relax": 0.0,
            "mirror_improvement_min": 0.2,
        },
        {
            "qi_rank_score": 10.0,
            "qi_constraint_score": 0.0,
            "qi_mirror_ratio_max": 0.5,
            "qi_seed_gate_passed": True,
            "qi_engineering_gate_passed": True,
        },
        {
            "qi_rank_score": 1.0,
            "qi_constraint_score": 0.0,
            "qi_mirror_ratio_max": 0.45,
            "qi_seed_gate_passed": True,
            "qi_engineering_gate_passed": True,
        },
    )
    assert rejected["qi_cleanup_promoted"] is False
    assert any("rank score did not improve" in reason for reason in rejected["qi_cleanup_rejection_reasons"])
    assert any("mirror ratio did not improve enough" in reason for reason in rejected["qi_cleanup_rejection_reasons"])

    result = SimpleNamespace(history={"objective_final": 1.0}, final_result={})
    checkpoint = qio.write_qi_stage_checkpoint(
        tmp_path / "stage",
        stage_index=2,
        stage_name="cleanup",
        stage_modes=[1, 2],
        stage_result=result,
        diagnostics={},
        promotion={"ok": True},
    )
    assert checkpoint.exists()
    assert json.loads((tmp_path / "stage_checkpoint.json").read_text())["diagnostics"]["objective_final"] == 1.0


def test_materialize_qi_stage_inputs_uses_final_result_fallback(tmp_path: Path) -> None:
    saved: list[tuple[Path, list[float]]] = []

    class Optimizer:
        def save_input(self, path, params) -> None:
            path = Path(path)
            values = np.asarray(params, dtype=float).tolist()
            saved.append((path, values))
            path.write_text(json.dumps(values) + "\n")

    result = SimpleNamespace(
        initial_optimizer=Optimizer(),
        final_optimizer=Optimizer(),
        initial_params=np.asarray([1.0, 2.0]),
        final_result={"x": np.asarray([3.0, 4.0])},
    )

    final_path = qio.materialize_qi_stage_inputs(tmp_path / "stage", result)

    assert final_path == tmp_path / "stage" / "input.final"
    assert (tmp_path / "stage" / "input.initial").read_text().strip() == "[1.0, 2.0]"
    assert (tmp_path / "stage" / "input.final").read_text().strip() == "[3.0, 4.0]"
    assert saved == [
        (tmp_path / "stage" / "input.initial", [1.0, 2.0]),
        (tmp_path / "stage" / "input.final", [3.0, 4.0]),
    ]


def test_run_basin_prefilter_uses_lazy_tools_and_selects_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(tmp_path)

    class SurveyTargets:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Candidate:
        label = "axis:0"
        params = np.asarray([0.1, 0.2])

        def as_record(self, names):
            return {"label": self.label, "names": list(names), "params": self.params.tolist(), "score": 1.0}

    saved_inputs: list[tuple[Path, np.ndarray]] = []
    stage = SimpleNamespace(
        specs=["spec"],
        optimizer=SimpleNamespace(
            _solve_forward=lambda params, trial: "state",
            save_input=lambda path, params: saved_inputs.append((Path(path), np.asarray(params))),
        ),
        ctx=SimpleNamespace(static="static", indata="indata", signgs=-1, flux="flux", pressure="pressure"),
    )
    monkeypatch.setattr(qio, "boundary_param_names", lambda specs: ["rc01", "zs01"])
    monkeypatch.setattr(qio, "create_x_scale", lambda specs, alpha: np.ones(2))
    monkeypatch.setattr(
        qio,
        "make_basin_prefilter_options",
        lambda config, **_kwargs: SimpleNamespace(surfaces=np.asarray([1.0])),
    )
    monkeypatch.setattr(
        qio,
        "_load_basin_prefilter_tools",
        lambda: (
            SurveyTargets,
            lambda **kwargs: [Candidate()],
            lambda records, targets: records,
            lambda records, path: Path(path).write_text("csv\n"),
            lambda **kwargs: stage,
        ),
    )
    monkeypatch.setattr(
        qio.vj,
        "qi_diagnostics_from_state",
        lambda **kwargs: {
            "qi_smooth_total": 1.0e-4,
            "qi_legacy_total": 2.0e-4,
            "qi_mirror_ratio_max": 0.2,
            "qi_max_elongation": 4.0,
            "mean_iota": 0.5,
            "aspect": 6.0,
        },
    )

    selected = qio.run_basin_prefilter("input.seed", tmp_path, {"enabled": True, "save_candidate_inputs": False})

    assert selected == tmp_path / "basin_prefilter" / "input.prefilter_selected"
    assert saved_inputs[-1][0] == selected
    assert (tmp_path / "basin_prefilter" / "candidates.json").exists()
    assert (tmp_path / "basin_prefilter" / "candidates.csv").read_text() == "csv\n"


def test_boundary_reference_preconditioner_selects_safe_non_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(tmp_path)
    seed = InData(scalars={"NFP": 2}, indexed={})
    reference = InData(scalars={"NFP": 2}, indexed={})
    monkeypatch.setattr(qio.vj, "read_indata", lambda path: reference if "reference" in str(path) else seed)
    monkeypatch.setattr(qio.vj, "interpolate_indata_boundary", lambda seed, reference, lam, **kwargs: InData(scalars={"LAMBDA": lam}, indexed={}))
    monkeypatch.setattr(qio.vj, "rebuild_for_optimization_resolution", lambda candidate, **kwargs: candidate)
    monkeypatch.setattr(qio.vj, "write_indata", lambda path, indata: Path(path).write_text(str(indata.scalars.get("LAMBDA"))))
    monkeypatch.setattr(qio.vj, "run_fixed_boundary", lambda input_out, **kwargs: SimpleNamespace(path=str(input_out)))
    monkeypatch.setattr(qio.vj, "write_wout_from_fixed_boundary_run", lambda path, run: Path(path).write_text("wout"))

    def fake_diagnostics(run, **kwargs):
        lam = float(Path(run.path).read_text())
        return {
            "qi_smooth_total": 1.0e-4 + lam,
            "qi_legacy_total": 2.0e-4 + lam,
            "qi_mirror_ratio_max": 0.2 if lam < 1.0 else 0.6,
            "qi_max_elongation": 4.0,
            "mean_iota": 0.5,
            "aspect": 6.0,
            "qi_seed_gate_passed": True,
            "qi_engineering_gate_passed": True,
            "qi_failure_reasons": [],
            "qi_rank_score": lam,
            "qi_constraint_score": 0.0,
        }

    monkeypatch.setattr(qio, "qi_diagnostics_for_run", fake_diagnostics)

    selected = qio.run_boundary_reference_preconditioner(
        "input.seed",
        tmp_path,
        {
            "enabled": True,
            "reference_input": "input.reference",
            "lambdas": (0.5, 1.0),
            "prefer_non_endpoint": True,
        },
    )

    assert selected.name == "input.interpolated"
    assert "lambda_0p500" in str(selected)
    summary = json.loads((tmp_path / "boundary_reference_preconditioner" / "summary.json").read_text())
    assert any(record["selected"] for record in summary)
    assert qio.run_boundary_reference_preconditioner("input.seed", tmp_path, {"enabled": False}) == Path("input.seed")


def test_boundary_reference_preconditioner_prefers_aspect_pool_before_qi_rank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(tmp_path)
    seed = InData(scalars={"NFP": 1}, indexed={})
    reference = InData(scalars={"NFP": 1}, indexed={})
    monkeypatch.setattr(qio.vj, "read_indata", lambda path: reference if "reference" in str(path) else seed)
    monkeypatch.setattr(
        qio.vj,
        "interpolate_indata_boundary",
        lambda seed, reference, lam, **kwargs: InData(scalars={"LAMBDA": lam}, indexed={}),
    )
    monkeypatch.setattr(qio.vj, "rebuild_for_optimization_resolution", lambda candidate, **kwargs: candidate)
    monkeypatch.setattr(qio.vj, "write_indata", lambda path, indata: Path(path).write_text(str(indata.scalars.get("LAMBDA"))))
    monkeypatch.setattr(qio.vj, "run_fixed_boundary", lambda input_out, **kwargs: SimpleNamespace(path=str(input_out)))
    monkeypatch.setattr(qio.vj, "write_wout_from_fixed_boundary_run", lambda path, run: Path(path).write_text("wout"))

    def fake_diagnostics(run, **kwargs):
        lam = float(Path(run.path).read_text())
        aspect = 5.6 if lam < 0.5 else 7.0
        return {
            "qi_smooth_total": 1.0,
            "qi_legacy_total": 0.2 if lam < 0.5 else 0.001,
            "qi_mirror_ratio_max": 0.2,
            "qi_max_elongation": 4.0,
            "mean_iota": 0.5,
            "aspect": aspect,
            "aspect_relative_error": abs(aspect - 5.0) / 5.0,
            "qi_seed_gate_passed": False,
            "qi_engineering_gate_passed": False,
            "qi_failure_reasons": ["synthetic"],
            "qi_rank_score": 100.0 if lam < 0.5 else 0.001,
            "qi_constraint_score": 0.0,
        }

    monkeypatch.setattr(qio, "qi_diagnostics_for_run", fake_diagnostics)

    selected = qio.run_boundary_reference_preconditioner(
        "input.seed",
        tmp_path,
        {
            "enabled": True,
            "reference_input": "input.reference",
            "lambdas": (0.25, 1.0),
            "target_aspect": 5.0,
            "aspect_relative_tolerance": 0.35,
        },
    )

    assert "lambda_0p250" in str(selected)
    summary = json.loads((tmp_path / "boundary_reference_preconditioner" / "summary.json").read_text())
    selected_record = next(record for record in summary if record["selected"])
    assert selected_record["aspect"] == pytest.approx(5.6)


def test_boundary_reference_preconditioner_can_prefer_lowest_qi_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(tmp_path)
    seed = InData(scalars={"NFP": 1}, indexed={})
    reference = InData(scalars={"NFP": 1}, indexed={})
    monkeypatch.setattr(qio.vj, "read_indata", lambda path: reference if "reference" in str(path) else seed)
    monkeypatch.setattr(
        qio.vj,
        "interpolate_indata_boundary",
        lambda seed, reference, lam, **kwargs: InData(scalars={"LAMBDA": lam}, indexed={}),
    )
    monkeypatch.setattr(qio.vj, "rebuild_for_optimization_resolution", lambda candidate, **kwargs: candidate)
    monkeypatch.setattr(
        qio.vj,
        "write_indata",
        lambda path, indata: Path(path).write_text(str(indata.scalars.get("LAMBDA"))),
    )
    monkeypatch.setattr(qio.vj, "run_fixed_boundary", lambda input_out, **kwargs: SimpleNamespace(path=str(input_out)))
    monkeypatch.setattr(qio.vj, "write_wout_from_fixed_boundary_run", lambda path, run: Path(path).write_text("wout"))

    def fake_diagnostics(run, **kwargs):
        lam = float(Path(run.path).read_text())
        low_qi = lam > 0.5
        return {
            "qi_smooth_total": 1.0e-4 if low_qi else 1.0e-2,
            "qi_legacy_total": 1.0e-4 if low_qi else 1.0e-2,
            "qi_mirror_ratio_max": 0.3,
            "qi_max_elongation": 4.0,
            "mean_iota": 0.5,
            "aspect": 5.0,
            "aspect_relative_error": 0.0,
            "qi_seed_gate_passed": True,
            "qi_engineering_gate_passed": True,
            "qi_failure_reasons": [],
            "qi_rank_score": 100.0 if low_qi else 0.0,
            "qi_constraint_score": 0.0,
        }

    monkeypatch.setattr(qio, "qi_diagnostics_for_run", fake_diagnostics)

    selected = qio.run_boundary_reference_preconditioner(
        "input.seed",
        tmp_path,
        {
            "enabled": True,
            "reference_input": "input.reference",
            "lambdas": (0.25, 1.0),
            "prefer_lowest_qi_candidate": True,
        },
    )

    assert "lambda_1p000" in str(selected)
    summary = json.loads((tmp_path / "boundary_reference_preconditioner" / "summary.json").read_text())
    selected_record = next(record for record in summary if record["selected"])
    assert selected_record["smooth_qi"] == pytest.approx(1.0e-4)


def test_run_qi_stage_policy_no_ramp_writes_pre_diagnostic_checkpoint(tmp_path: Path) -> None:
    _configure(tmp_path)
    calls: list[dict[str, object]] = []

    def solve_qi_stage(input_file, output_dir, problem, **kwargs):
        calls.append({"input": input_file, "output": output_dir, "problem": problem, **kwargs})
        return SimpleNamespace(history={"objective_final": 0.25}, final_result={})

    result, promotion_log = qio.run_qi_stage_policy(
        "input.seed",
        tmp_path,
        solve_qi_stage=solve_qi_stage,
        make_qi_problem=lambda *args, **kwargs: {"objective": "qi"},
        boundary_reference_preconditioner={},
        mirror_ramp_stages=[],
    )

    assert result.history["objective_final"] == 0.25
    assert promotion_log == []
    assert calls[0]["max_nfev"] == 4
    assert json.loads((tmp_path / "stage_checkpoint.json").read_text())["role"] == "stage_pre_diagnostics"


def test_run_qi_stage_policy_writes_pending_checkpoint_before_long_stage(tmp_path: Path) -> None:
    _configure(tmp_path)
    pre_dir = tmp_path / "boundary_reference_preconditioner"
    pre_dir.mkdir()
    selected_input = pre_dir / "lambda_0p950" / "input.interpolated"
    selected_input.parent.mkdir()
    selected_input.write_text("input\n")
    (pre_dir / "summary.json").write_text(
        json.dumps(
            [
                {
                    "input": str(selected_input),
                    "lambda": 0.95,
                    "selected": True,
                    "smooth_qi": 1.2e-3,
                    "legacy_qi": 9.0e-4,
                    "mirror": 0.28,
                    "elongation": 4.0,
                    "mean_iota": 0.43,
                    "aspect": 5.2,
                    "aspect_relative_error": 0.04,
                    "qi_seed_gate_passed": True,
                    "qi_engineering_gate_passed": True,
                    "failure_reasons": [],
                }
            ]
        )
        + "\n"
    )

    def solve_qi_stage(*_args, **_kwargs):
        raise RuntimeError("simulated expensive stage failure")

    with pytest.raises(RuntimeError, match="simulated expensive stage failure"):
        qio.run_qi_stage_policy(
            selected_input,
            tmp_path,
            solve_qi_stage=solve_qi_stage,
            make_qi_problem=lambda stage=None: {"stage": stage},
            boundary_reference_preconditioner={"enabled": True},
            mirror_ramp_stages=[{"name": "long", "stage_modes": [5], "max_nfev": 60}],
        )

    root_checkpoint = json.loads((tmp_path / "stage_checkpoint.json").read_text())
    stage_checkpoint = json.loads((tmp_path / "mirror_ramp_01_long" / "qi_stage_checkpoint.json").read_text())

    assert root_checkpoint == stage_checkpoint
    assert root_checkpoint["role"] == "mirror_ramp_pending"
    assert root_checkpoint["promotion"]["stage_pending"] is True
    assert root_checkpoint["diagnostics"]["source"] == "boundary_reference_preconditioner"
    assert root_checkpoint["diagnostics"]["boundary_reference_input_path"] == str(selected_input)
    assert root_checkpoint["diagnostics"]["boundary_reference_wout_path"] is None
    assert root_checkpoint["diagnostics"]["qi_legacy_total"] == pytest.approx(9.0e-4)
    assert root_checkpoint["diagnostics"]["qi_mirror_ratio_max"] == pytest.approx(0.28)
    assert root_checkpoint["diagnostics"]["mean_iota"] == pytest.approx(0.43)
    assert root_checkpoint["diagnostics"]["aspect"] == pytest.approx(5.2)


def test_run_qi_stage_policy_mirror_ramp_promotes_guarded_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(tmp_path)
    calls: list[dict[str, object]] = []

    def solve_qi_stage(input_file, output_dir, problem, **kwargs):
        label = str(kwargs["label"])
        stage_result = SimpleNamespace(
            label=label,
            history={"objective_final": len(calls) + 1.0},
            final_result={},
            final_state="state",
            final_optimizer=SimpleNamespace(static="static", indata="indata", signgs=-1),
        )
        calls.append(
            {
                "input": input_file,
                "output": Path(output_dir),
                "problem": problem,
                "stage_modes": kwargs.get("stage_modes"),
                "method": kwargs.get("method"),
                "use_mode_continuation": kwargs.get("use_mode_continuation"),
                "label": label,
            }
        )
        return stage_result

    def fake_diagnostics(stage_result, **kwargs):
        label = stage_result.label
        if "baseline" in label:
            return {
                "label": "baseline",
                "qi_rank_score": 5.0,
                "qi_constraint_score": 1.0,
                "qi_seed_gate_passed": True,
                "qi_engineering_gate_passed": True,
                "qi_mirror_ratio_max": 0.6,
                "mean_iota": 0.35,
                "qi_smooth_total": 1.0e-3,
                "qi_legacy_total": 1.0e-3,
            }
        rank = 10.0 if "rough" in label else 1.0
        mirror = 0.7 if "rough" in label else 0.2
        return {
            "label": label,
            "qi_rank_score": rank,
            "qi_constraint_score": 0.5,
            "qi_seed_gate_passed": True,
            "qi_engineering_gate_passed": True,
            "qi_mirror_ratio_max": mirror,
            "mean_iota": 0.5,
            "qi_smooth_total": 8.0e-4,
            "qi_legacy_total": 8.0e-4,
        }

    def fake_promotable(stage_diagnostics, **kwargs):
        promoted = "polish" in str(stage_diagnostics["label"])
        return {
            **stage_diagnostics,
            "qi_cleanup_promoted": promoted,
            "qi_cleanup_rejection_reasons": [] if promoted else ["rough stage rejected"],
        }

    monkeypatch.setattr(qio, "qi_diagnostics_for_result", fake_diagnostics)
    monkeypatch.setattr(qio.vj, "qi_cleanup_candidate_promotable", fake_promotable)

    result, promotion_log = qio.run_qi_stage_policy(
        "input.seed",
        tmp_path,
        solve_qi_stage=solve_qi_stage,
        make_qi_problem=lambda stage=None: {"stage": None if stage is None else stage.get("name", "baseline")},
        boundary_reference_preconditioner={"enabled": True, "accept_as_baseline": True},
        mirror_ramp_stages=[
            {"name": "rough", "stage_modes": [1], "max_nfev": 2, "method": "scipy_matrix_free"},
            {"name": "polish", "stage_modes": [1, 2], "max_nfev": 3, "method": "lbfgs"},
        ],
    )

    assert result.label.endswith("polish (max_mode=3, ESS)")
    assert [entry["name"] for entry in promotion_log] == ["rough", "polish"]
    assert [entry["promoted"] for entry in promotion_log] == [False, True]
    assert calls[0]["output"] == tmp_path / "boundary_reference_baseline"
    assert calls[1]["stage_modes"] == [1]
    assert calls[2]["method"] == "lbfgs"
    assert (tmp_path / "mirror_ramp_02_polish" / "qi_stage_checkpoint.json").exists()


def test_run_qi_stage_policy_materializes_promoted_seed_for_next_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(tmp_path)
    calls: list[dict[str, object]] = []

    class DummyOptimizer:
        def __init__(self, label: str) -> None:
            self.label = label

        def save_input(self, path, params) -> None:
            Path(path).write_text(f"{self.label}: {np.asarray(params, dtype=float).tolist()}\n")

    def solve_qi_stage(input_file, output_dir, problem, **kwargs):
        call_index = len(calls)
        calls.append(
            {
                "input": input_file,
                "output": Path(output_dir),
                "problem": problem,
                "label": str(kwargs["label"]),
            }
        )
        return SimpleNamespace(
            label=str(kwargs["label"]),
            history={"objective_final": 0.5 / (call_index + 1)},
            initial_optimizer=DummyOptimizer(f"initial-{call_index}"),
            final_optimizer=DummyOptimizer(f"final-{call_index}"),
            initial_params=np.asarray([call_index], dtype=float),
            final_params=np.asarray([call_index + 10], dtype=float),
            final_result={"x": np.asarray([call_index + 20], dtype=float)},
            final_state="state",
        )

    def fake_diagnostics(stage_result, **_kwargs):
        return {
            "label": stage_result.label,
            "qi_rank_score": 1.0,
            "qi_constraint_score": 0.1,
            "qi_seed_gate_passed": True,
            "qi_engineering_gate_passed": True,
            "qi_mirror_ratio_max": 0.2,
            "mean_iota": 0.5,
            "qi_smooth_total": 8.0e-4,
            "qi_legacy_total": 8.0e-4,
        }

    monkeypatch.setattr(qio, "qi_diagnostics_for_result", fake_diagnostics)
    monkeypatch.setattr(
        qio.vj,
        "qi_cleanup_candidate_promotable",
        lambda stage_diagnostics, **_kwargs: {
            **stage_diagnostics,
            "qi_cleanup_promoted": True,
            "qi_cleanup_rejection_reasons": [],
        },
    )

    result, promotion_log = qio.run_qi_stage_policy(
        "input.seed",
        tmp_path,
        solve_qi_stage=solve_qi_stage,
        make_qi_problem=lambda stage=None: {"stage": None if stage is None else stage["name"]},
        boundary_reference_preconditioner={"enabled": False},
        mirror_ramp_stages=[
            {"name": "prefiltered_mirror_qi_iota_cleanup", "stage_modes": [5], "max_nfev": 2},
            {"name": "polish", "stage_modes": [5], "max_nfev": 2},
        ],
    )

    first_stage_final = tmp_path / "mirror_ramp_01_prefiltered_mirror_qi_iota_cleanup" / "input.final"
    first_stage_initial = tmp_path / "mirror_ramp_01_prefiltered_mirror_qi_iota_cleanup" / "input.initial"

    assert result.label.startswith("QI polish")
    assert [entry["promoted"] for entry in promotion_log] == [True, True]
    assert calls[0]["input"] == "input.seed"
    assert calls[1]["input"] == first_stage_final
    assert first_stage_initial.exists()
    assert first_stage_final.exists()
    assert first_stage_final.read_text().startswith("final-0:")


def test_run_qi_stage_policy_keeps_baseline_when_cleanup_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure(tmp_path)
    calls: list[SimpleNamespace] = []

    def solve_qi_stage(_input_file, _output_dir, _problem, **kwargs):
        result = SimpleNamespace(
            label=str(kwargs["label"]),
            history={"objective_final": len(calls) + 1.0},
            final_result={},
            final_state="state",
            final_optimizer=SimpleNamespace(static="static", indata="indata", signgs=1),
        )
        calls.append(result)
        return result

    def fake_diagnostics(stage_result, **_kwargs):
        if "baseline" in stage_result.label:
            return {
                "qi_rank_score": 1.0,
                "qi_constraint_score": 0.1,
                "qi_seed_gate_passed": True,
                "qi_engineering_gate_passed": True,
                "qi_mirror_ratio_max": 0.25,
                "mean_iota": 0.45,
                "qi_smooth_total": 8.0e-4,
                "qi_legacy_total": 8.0e-4,
            }
        return {
            "qi_rank_score": 10.0,
            "qi_constraint_score": 5.0,
            "qi_seed_gate_passed": False,
            "qi_engineering_gate_passed": False,
            "qi_mirror_ratio_max": 0.5,
            "mean_iota": 0.40,
            "qi_smooth_total": 2.0e-2,
            "qi_legacy_total": 2.0e-2,
        }

    monkeypatch.setattr(qio, "qi_diagnostics_for_result", fake_diagnostics)
    monkeypatch.setattr(
        qio.vj,
        "qi_cleanup_candidate_promotable",
        lambda stage_diagnostics, **_kwargs: {
            **stage_diagnostics,
            "qi_cleanup_promoted": False,
            "qi_cleanup_rejection_reasons": ["cleanup rejected"],
        },
    )

    result, promotion_log = qio.run_qi_stage_policy(
        "input.seed",
        tmp_path,
        solve_qi_stage=solve_qi_stage,
        make_qi_problem=lambda stage=None: {"stage": stage},
        boundary_reference_preconditioner={"enabled": True, "accept_as_baseline": True},
        mirror_ramp_stages=[{"name": "bad_cleanup", "stage_modes": [5], "max_nfev": 2}],
    )

    assert "boundary-reference baseline" in result.label
    assert len(calls) == 2
    assert promotion_log[0]["promoted"] is False
    assert promotion_log[0]["rejection_reasons"] == ["cleanup rejected"]

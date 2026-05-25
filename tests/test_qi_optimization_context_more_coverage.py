from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.qi_optimization as qio
from vmec_jax.qi_diagnostics import QIDiagnosticOptions


def _context_values(tmp_path: Path) -> dict[str, object]:
    return {
        "ALPHA": 2.5,
        "CONTINUATION_NFEV": 2,
        "INNER_MAX_ITER": 20,
        "JIT_BOOZ": False,
        "MAX_ELONGATION": 6.0,
        "MAX_MIRROR_RATIO": 0.28,
        "MAX_MODE": 3,
        "MAX_NFEV": 8,
        "METHOD": "scipy_matrix_free",
        "MIN_VMEC_MODE": 5,
        "MIRROR_SURFACE_INDEX": -1,
        "MIRROR_WEIGHT": 4.0,
        "OPT_QI_RESOLUTION": {"mboz": 11, "nphi": 17},
        "OUTPUT_DIR": tmp_path,
        "QI_GATE_LEGACY_MAX": 2.0e-3,
        "QI_GATE_SMOOTH_MAX": 1.0e-3,
        "QI_OPTIONS": QIDiagnosticOptions(
            surfaces=np.asarray([0.25, 1.0]),
            mboz=7,
            nboz=8,
            nphi=9,
            nalpha=10,
            n_bounce=11,
            include_bounce_endpoints=True,
            phimin=0.1,
        ),
        "QI_WEIGHT": 12.0,
        "SOLVER_DEVICE": None,
        "STAGE_MODES": (1, 2, 3),
        "STAGE_REPEATS": 2,
        "SURFACES": np.asarray([0.5, 1.0]),
        "TARGET_ABS_IOTA_MIN": 0.41,
        "TARGET_ASPECT": 6.0,
        "TRIAL_FTOL": 1.0e-8,
        "USE_ESS": True,
        "USE_MODE_CONTINUATION": True,
    }


def _context(tmp_path: Path, **overrides) -> qio.QIOptimizationContext:
    values = _context_values(tmp_path)
    values.update(overrides)
    return qio.make_qi_optimization_context(values)


def test_context_factory_accepts_uppercase_and_lowercase_overrides(tmp_path: Path) -> None:
    ctx = qio.make_qi_optimization_context(
        _context_values(tmp_path / "upper"),
        max_mode=5,
        output_dir=tmp_path / "lower",
        opt_qi_resolution={"nalpha": 23},
    )

    assert ctx.max_mode == 5
    assert ctx.output_dir == tmp_path / "lower"
    assert ctx.opt_qi_resolution == {"nalpha": 23}
    assert ctx.stage_modes == (1, 2, 3)
    assert ctx.target_abs_iota_min == pytest.approx(0.41)


def test_context_factory_strict_mode_does_not_read_module_globals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    values = _context_values(tmp_path)
    values.pop("MAX_MODE")
    monkeypatch.setattr(qio, "MAX_MODE", 99, raising=False)

    legacy = qio.make_qi_optimization_context(values)
    assert legacy.max_mode == 99

    with pytest.raises(KeyError, match="MAX_MODE"):
        qio.make_qi_optimization_context(values, strict=True)


def test_explicit_context_and_default_context_override_poisoned_globals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit = _context(tmp_path / "explicit", MAX_MODE=2)
    default = _context(tmp_path / "default", MAX_MODE=4)
    monkeypatch.setattr(qio, "_DEFAULT_CONTEXT", default)
    monkeypatch.setattr(qio, "MAX_MODE", 99, raising=False)

    assert qio._ctx(explicit, "max_mode") == 2
    assert qio._ctx(None, "max_mode") == 4

    monkeypatch.setattr(qio, "_DEFAULT_CONTEXT", None)
    assert qio._ctx(explicit, "max_mode") == 2
    assert qio._ctx(None, "max_mode") == 99


def test_seed_term_normalisation_uses_context_and_disables_cleanly(tmp_path: Path) -> None:
    ctx = _context(tmp_path, MAX_MODE=1)

    assert qio._normalise_seed_terms(None, ctx=ctx) == ()
    assert qio._normalise_seed_terms({"enabled": False, "amplitude": 1.0}, ctx=ctx) == ()
    assert qio._normalise_seed_terms({"max_mode": 0, "amplitude": 1.0e-5}, ctx=ctx) == ()
    assert qio._normalise_seed_terms((("rbc", ("1", "0"), "1e-5"),), ctx=ctx) == (
        ("RBC", (1, 0), 1.0e-5),
    )

    default_terms = qio._normalise_seed_terms({"amplitude": 2.0e-5}, ctx=ctx)
    assert ("RBC", (1, 0), 2.0e-5) in default_terms
    assert all(max(abs(n), abs(m)) <= 1 for _name, (n, m), _value in default_terms)


def test_parse_float_sequence_and_lazy_basin_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert qio._parse_float_sequence(None, name="radii") is None
    assert qio._parse_float_sequence("", name="radii") is None
    assert qio._parse_float_sequence("0.1, 0.2 3", name="radii") == (0.1, 0.2, 3.0)
    with pytest.raises(ValueError, match="radii must be"):
        qio._parse_float_sequence("0.1 bad", name="radii")

    real_import = builtins.__import__

    def blocked_import(name, globals_=None, locals_=None, fromlist=(), level=0):  # noqa: ANN001
        if name in {"tools.diagnostics.qi_basin_survey", "tools.diagnostics.qi_landscape_scan"}:
            raise ModuleNotFoundError(name)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="source checkout"):
        qio._load_basin_prefilter_tools()


def test_basin_prefilter_options_use_explicit_context(tmp_path: Path) -> None:
    ctx = _context(
        tmp_path,
        OPT_QI_RESOLUTION={"mboz": 13, "nboz": 14, "nphi": 15, "nalpha": 16, "n_bounce": 17},
        SURFACES=np.asarray([0.75]),
        MAX_MIRROR_RATIO=0.31,
        MAX_ELONGATION=5.5,
    )

    options = qio.make_basin_prefilter_options({"mirror_ntheta": 24, "elongation_nphi": 6}, ctx=ctx)

    assert options.mboz == 13
    assert options.nboz == 14
    assert options.nphi == 15
    assert options.nalpha == 16
    assert options.n_bounce == 17
    assert options.surfaces.tolist() == [0.75]
    assert options.mirror_threshold == pytest.approx(0.31)
    assert options.mirror_ntheta == 24
    assert options.elongation_threshold == pytest.approx(5.5)
    assert options.elongation_nphi == 6


def test_jsonable_history_and_partial_diagnostics_helpers() -> None:
    class NotArrayLike:
        def __array__(self):
            raise TypeError("not array-like")

        def __str__(self) -> str:
            return "not-array"

    assert qio._jsonable(
        {
            1: np.asarray([1.0, np.nan]),
            "path": Path("input.demo"),
            "finite": np.float64(2.5),
            "nonfinite": float("-inf"),
            "nested": (np.int64(3), NotArrayLike()),
        }
    ) == {
        "1": [1.0, None],
        "path": "input.demo",
        "finite": 2.5,
        "nonfinite": None,
        "nested": [3, "not-array"],
    }

    assert qio._stage_result_history(SimpleNamespace(history={"a": 1})) == {"a": 1}
    assert qio._stage_result_history({"_history_dump": {"b": 2}}) == {"b": 2}
    assert qio._stage_result_history(SimpleNamespace(final_result={"_history_dump": {"c": 3}})) == {"c": 3}
    assert qio._stage_result_history(SimpleNamespace()) == {}

    history = {
        "objective_final": np.float64(1.25),
        "qs_final": np.float64(2.0e-3),
        "aspect_final": np.float64(6.0),
        "iota_final": np.float64(0.43),
        "nfev": np.int64(7),
        "njev": np.int64(3),
        "total_wall_time_s": np.float64(12.5),
    }
    partial = qio._partial_diagnostics_from_history(history, {})
    assert partial == {
        "objective_final": np.float64(1.25),
        "qs_final": np.float64(2.0e-3),
        "aspect": np.float64(6.0),
        "mean_iota": np.float64(0.43),
        "nfev": np.int64(7),
        "njev": np.int64(3),
        "total_wall_time_s": np.float64(12.5),
        "partial": True,
        "diagnostics_pending": True,
    }
    assert qio._partial_diagnostics_from_history(history, {"existing": 1}) == {
        "existing": 1,
        "objective_final": np.float64(1.25),
        "qs_final": np.float64(2.0e-3),
        "aspect": np.float64(6.0),
        "mean_iota": np.float64(0.43),
        "nfev": np.int64(7),
        "njev": np.int64(3),
        "total_wall_time_s": np.float64(12.5),
        "partial": True,
    }


def test_stage_checkpoint_preserves_partial_history_and_context_root(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path / "root")
    stage_dir = tmp_path / "stage"
    stage_result = SimpleNamespace(
        final_result={
            "_history_dump": {
                "history": [{"objective": 3.0}, {"objective": 1.0}],
                "objective_final": 1.0,
                "qs_final": 8.0e-4,
                "aspect_final": 6.1,
                "iota_final": -0.45,
                "nfev": 4,
            }
        }
    )

    checkpoint_path = qio.write_qi_stage_checkpoint(
        stage_dir,
        stage_index=None,
        stage_name="pre_diagnostics",
        stage_modes=(1, 2),
        stage_result=stage_result,
        diagnostics={},
        promotion={"diagnostics_pending": True},
        role="stage_pre_diagnostics",
        ctx=ctx,
    )

    checkpoint = json.loads(checkpoint_path.read_text())
    root_checkpoint = json.loads((ctx.output_dir / "stage_checkpoint.json").read_text())
    diagnostics = json.loads((stage_dir / "diagnostics.json").read_text())
    history = json.loads((stage_dir / "history.json").read_text())

    assert checkpoint == root_checkpoint
    assert checkpoint["stage"] is None
    assert checkpoint["role"] == "stage_pre_diagnostics"
    assert [mode["mode"] for mode in checkpoint["stage_modes"]] == [1, 2]
    assert checkpoint["diagnostics"]["partial"] is True
    assert checkpoint["diagnostics"]["mean_iota"] == -0.45
    assert diagnostics["diagnostics_pending"] is True
    assert history["history"][-1]["objective"] == 1.0


def test_save_raw_seed_initial_artifacts_uses_context_solver_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path, SOLVER_DEVICE="gpu")
    calls: list[tuple[str, object]] = []
    sentinel_indata = object()
    sentinel_run = object()

    monkeypatch.setattr(qio.vj, "read_indata", lambda path: calls.append(("read", Path(path))) or sentinel_indata)
    monkeypatch.setattr(
        qio.vj,
        "write_indata",
        lambda path, indata: calls.append(("write_input", (Path(path), indata))) or Path(path).write_text("input\n"),
    )
    monkeypatch.setattr(
        qio.vj,
        "run_fixed_boundary",
        lambda path, *, solver_device, verbose: calls.append(("run", (Path(path), solver_device, verbose))) or sentinel_run,
    )
    monkeypatch.setattr(
        qio.vj,
        "write_wout_from_fixed_boundary_run",
        lambda path, run: calls.append(("write_wout", (Path(path), run))) or Path(path).write_text("wout\n"),
    )

    run = qio.save_raw_seed_initial_artifacts(
        tmp_path / "input.seed",
        tmp_path / "nested" / "input.initial",
        tmp_path / "nested" / "wout_initial.nc",
        ctx=ctx,
    )

    assert run is sentinel_run
    assert (tmp_path / "nested" / "input.initial").read_text() == "input\n"
    assert (tmp_path / "nested" / "wout_initial.nc").read_text() == "wout\n"
    assert calls == [
        ("read", tmp_path / "input.seed"),
        ("write_input", (tmp_path / "nested" / "input.initial", sentinel_indata)),
        ("run", (tmp_path / "input.seed", "gpu", False)),
        ("write_wout", (tmp_path / "nested" / "wout_initial.nc", sentinel_run)),
    ]


def test_basin_prefilter_disabled_returns_input_path(tmp_path: Path) -> None:
    assert qio.run_basin_prefilter(tmp_path / "input.seed", tmp_path / "out", {"enabled": False}) == tmp_path / "input.seed"


def test_stage_promotion_branches_use_context_gates(tmp_path: Path) -> None:
    ctx = _context(tmp_path, QI_GATE_SMOOTH_MAX=1.0e-3, QI_GATE_LEGACY_MAX=2.0e-3)
    iota_stage = {
        "accept_if_iota_improves": True,
        "iota_improvement_min": 0.05,
        "qi_relax_for_iota": 2.0,
    }
    promotion = {
        "qi_cleanup_promoted": False,
        "qi_cleanup_rejection_reasons": ["mirror ratio did not improve"],
        "mean_iota": 0.50,
        "qi_smooth_total": 1.8e-3,
        "qi_legacy_total": 3.0e-3,
    }
    reference = {
        "mean_iota": 0.40,
        "qi_smooth_total": 1.0e-3,
        "qi_legacy_total": 2.0e-3,
    }

    accepted = qio.stage_promotes_candidate(iota_stage, promotion, reference, ctx=ctx)
    assert accepted["qi_cleanup_promoted"] is True
    assert accepted["qi_cleanup_rejection_reasons"] == []
    assert "iota increased by" in accepted["qi_iota_promotion_reason"]

    rejected = qio.stage_promotes_candidate(
        iota_stage,
        {**promotion, "mean_iota": 0.41, "qi_smooth_total": 3.0e-3},
        reference,
        ctx=ctx,
    )
    assert rejected["qi_cleanup_promoted"] is False
    assert "relaxed QI promotion" in "\n".join(rejected["qi_cleanup_rejection_reasons"])

    rank_and_engineering_stage = {
        "accept_if_rank_improves": True,
        "rank_score_relax": 0.0,
        "accept_if_engineering_score_improves": True,
        "engineering_score_relax": 0.0,
        "mirror_improvement_min": 0.02,
    }
    worse_candidate = {
        "qi_cleanup_promoted": True,
        "qi_cleanup_rejection_reasons": [],
        "qi_seed_gate_passed": True,
        "qi_engineering_gate_passed": True,
        "qi_rank_score": 2.0,
        "qi_constraint_score": 1.0,
        "qi_mirror_ratio_max": 0.30,
    }
    better_reference = {
        "qi_seed_gate_passed": True,
        "qi_engineering_gate_passed": True,
        "qi_rank_score": 1.0,
        "qi_constraint_score": 1.0,
        "qi_mirror_ratio_max": 0.31,
    }
    out = qio.stage_promotes_candidate(
        rank_and_engineering_stage,
        worse_candidate,
        better_reference,
        ctx=ctx,
    )
    reasons = "\n".join(out["qi_cleanup_rejection_reasons"])
    assert out["qi_cleanup_promoted"] is False
    assert "rank score did not improve" in reasons
    assert "engineering score did not improve" in reasons
    assert "mirror ratio did not improve enough" in reasons

    first_baseline = qio.stage_promotes_candidate(
        rank_and_engineering_stage,
        {"qi_cleanup_promoted": True, "qi_cleanup_rejection_reasons": []},
        None,
        ctx=ctx,
    )
    assert first_baseline["qi_cleanup_promoted"] is True


def test_private_scalar_and_json_helpers_cover_invalid_inputs() -> None:
    assert np.isnan(qio._diagnostic_float({"value": None}, "value"))
    assert qio._diagnostic_float({"value": "1.25"}, "value") == pytest.approx(1.25)
    assert qio._finite_or_inf("not-a-number") == float("inf")
    assert qio._finite_or_inf(np.nan) == float("inf")
    assert qio._finite_or_none("not-a-number") is None
    assert qio._finite_or_none(np.inf) is None
    assert qio._parse_float_sequence(" ,  ", name="values") is None
    assert qio._jsonable(np.asarray(3.25)) == pytest.approx(3.25)

    class BadArray:
        def __array__(self, dtype=None):
            del dtype
            raise RuntimeError("no array view")

        def __str__(self):
            return "bad-array"

    assert qio._jsonable(BadArray()) == "bad-array"


def test_seed_term_normalisation_defaults_and_nonnumeric_existing_coefficients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path, MAX_MODE=1)

    default_terms = qio._normalise_seed_terms({"enabled": True, "amplitude": 2.0e-5}, ctx=ctx)
    assert default_terms
    assert {term[0] for term in default_terms} == {"RBC", "ZBS"}
    assert {term[2] for term in default_terms} == {2.0e-5}

    manual_terms = qio._normalise_seed_terms([("rbc", ("-1", "1"), "0.125")], ctx=ctx)
    assert manual_terms == (("RBC", (-1, 1), 0.125),)

    source = qio.InData(scalars={"NFP": 2}, indexed={"RBC": {(1, 0): "bad-existing-value"}})
    written = {}
    monkeypatch.setattr(qio.vj, "read_indata", lambda path: source)

    def fake_write_indata(path, indata):
        written["path"] = Path(path)
        written["indata"] = indata
        Path(path).write_text("seeded\n")

    monkeypatch.setattr(qio.vj, "write_indata", fake_write_indata)

    out = qio.run_target_helicity_seed_preconditioner(
        tmp_path / "input.seed",
        tmp_path,
        {"terms": [("RBC", (1, 0), 0.5)], "only_if_abs_below": 0.0},
        ctx=ctx,
    )

    assert out == tmp_path / "target_helicity_seed" / "input.target_helicity_seed"
    assert written["path"] == out
    assert written["indata"].indexed["RBC"][(1, 0)] == pytest.approx(0.5)
    metadata = json.loads((tmp_path / "target_helicity_seed" / "metadata.json").read_text())
    assert metadata["inserted"] == [{"family": "RBC", "m": 0, "n": 1, "value": 0.5}]


def test_qi_diagnostics_helpers_build_options_and_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path)
    calls = []
    annotated = []

    def fake_qi_diagnostics_from_state(**kwargs):
        calls.append(kwargs)
        return {"mean_iota": 0.44, "qi_smooth_total": 1.0e-4}

    def fake_annotate(diagnostics, *, targets):
        annotated.append((diagnostics, targets))
        out = dict(diagnostics)
        out["smooth_gate"] = targets.smooth_qi_max
        out["legacy_gate"] = targets.legacy_qi_max
        out["target_aspect"] = targets.target_aspect
        out["abs_iota_min"] = targets.abs_iota_min
        out["mirror_ratio_max"] = targets.mirror_ratio_max
        out["max_elongation"] = targets.max_elongation
        return out

    monkeypatch.setattr(qio.vj, "qi_diagnostics_from_state", fake_qi_diagnostics_from_state)
    monkeypatch.setattr(qio, "annotate_qi_seed_suitability", fake_annotate)

    stage_result = SimpleNamespace(
        final_state="stage-state",
        final_optimizer=SimpleNamespace(static="stage-static", indata="stage-indata", signgs=-1),
    )
    result_diagnostics = qio.qi_diagnostics_for_result(
        stage_result,
        mirror_threshold=0.33,
        mirror_surface_index=-1,
        ctx=ctx,
    )

    assert calls[0]["state"] == "stage-state"
    assert calls[0]["static"] == "stage-static"
    assert calls[0]["surfaces"] is ctx.surfaces
    assert calls[0]["options"].mirror_threshold == pytest.approx(0.33)
    assert calls[0]["options"].mirror_surface_index == -1
    assert calls[0]["options"].elongation_threshold == pytest.approx(ctx.max_elongation)
    assert result_diagnostics["smooth_gate"] == pytest.approx(ctx.qi_gate_smooth_max)
    assert result_diagnostics["legacy_gate"] == pytest.approx(ctx.qi_gate_legacy_max)

    run = SimpleNamespace(state="run-state", static="run-static", indata="run-indata", signgs=1)
    run_diagnostics = qio.qi_diagnostics_for_run(
        run,
        mirror_threshold=0.44,
        mirror_surface_index=0,
        target_aspect=7.0,
        abs_iota_min=0.52,
        max_elongation=9.0,
        resolution={"nphi": 33, "n_bounce": 35},
        smooth_qi_max=4.0e-3,
        legacy_qi_max=5.0e-3,
        ctx=ctx,
    )

    assert calls[1]["state"] == "run-state"
    assert calls[1]["static"] == "run-static"
    assert calls[1]["options"].nphi == 33
    assert calls[1]["options"].n_bounce == 35
    assert calls[1]["options"].nalpha == ctx.qi_options.nalpha
    assert calls[1]["options"].mirror_threshold == pytest.approx(0.44)
    assert calls[1]["options"].elongation_threshold == pytest.approx(9.0)
    assert run_diagnostics["smooth_gate"] == pytest.approx(4.0e-3)
    assert run_diagnostics["legacy_gate"] == pytest.approx(5.0e-3)
    assert run_diagnostics["target_aspect"] == pytest.approx(7.0)
    assert run_diagnostics["abs_iota_min"] == pytest.approx(0.52)
    assert run_diagnostics["mirror_ratio_max"] == pytest.approx(0.44)
    assert run_diagnostics["max_elongation"] == pytest.approx(9.0)
    assert len(annotated) == 2

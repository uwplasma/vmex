#!/usr/bin/env python
"""Run the robust QI staged script from sweep/showcase code.

``QI_optimization.py`` is intentionally written as a standalone example.  This
module provides a thin subprocess boundary so sweep drivers can reuse that
stronger staged/reference machinery without importing and executing the example
at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
TOOLS_OPTIMIZATION_DIR = ROOT / "tools" / "diagnostics" / "optimization"
QI_SCRIPT = SCRIPT_DIR / "QI_optimization.py"
DEFAULT_REFERENCE_LAMBDAS = (
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.975,
    0.995,
    1.0,
    1.005,
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_OPTIMIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_OPTIMIZATION_DIR))

import generate_qs_ess_sweep as sweep
from qi_optimization_cases import QI_CASES
from vmec_jax.namelist import read_indata


@dataclass(frozen=True)
class QIStagedCaseConfig:
    """Configuration for one subprocess-isolated QI staged optimization."""

    name: str
    input_file: Path
    output_dir: Path
    max_mode: int
    policy: str = "continuation"
    policy_case: str = "qi_stel_seed_3127"
    reference_input: Path | None = None
    reference_accept_as_baseline: bool = False
    backend_label: str = "cpu"
    solver_device: str | None = None
    worker_jax_platforms: str | None = None
    use_ess: bool = True
    stage_mode_policy: str = "lower"
    max_nfev: int | None = None
    continuation_nfev: int | None = None
    inner_max_iter: int | None = None
    inner_ftol: float | None = None
    trial_max_iter: int | None = None
    trial_ftol: float | None = None
    ess_alpha: float | None = None
    method: str | None = None
    target_aspect: float | None = None
    target_abs_iota_min: float | None = None
    max_mirror_ratio: float | None = None
    mirror_surface_index: int | None = None
    max_elongation: float | None = None
    qi_gate_smooth_max: float | None = None
    qi_gate_legacy_max: float | None = None
    qi_ceiling_max: float | None = None
    qi_ceiling_smooth_penalty: float | None = None
    mirror_weight: float | None = None
    elongation_weight: float | None = None
    qi_mboz: int | None = None
    qi_nboz: int | None = None
    qi_nphi: int | None = None
    qi_nalpha: int | None = None
    qi_n_bounce: int | None = None
    audit_qi_mboz: int | None = None
    audit_qi_nboz: int | None = None
    audit_qi_nphi: int | None = None
    audit_qi_nalpha: int | None = None
    audit_qi_n_bounce: int | None = None
    reference_lambdas: tuple[float, ...] | None = DEFAULT_REFERENCE_LAMBDAS
    mirror_ramp_stages: tuple[dict[str, Any], ...] | None = None
    make_plots: bool = True
    timeout_s: float | None = None


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _finite_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _first_finite_float(*values: Any) -> float | None:
    """Return the first finite float, preserving valid zero values."""

    for value in values:
        out = _finite_float(value)
        if out is not None:
            return out
    return None


def _profile_wall_time(profile: dict[str, Any], name: str) -> float | None:
    """Return profile wall time from nested or legacy flattened records."""

    record = profile.get(name)
    if isinstance(record, dict):
        return _finite_float(record.get("wall_time_s"))
    return _finite_float(profile.get(f"{name}_wall_time_s"))


def _input_nfp(input_file: Path) -> int | None:
    try:
        return _finite_int(read_indata(input_file).scalars.get("NFP"))
    except Exception:
        return None


def _prepend_pythonpath(env: dict[str, str], *paths: Path) -> None:
    current = env.get("PYTHONPATH")
    additions = [str(path) for path in paths]
    env["PYTHONPATH"] = os.pathsep.join(additions + ([current] if current else []))


def _policy_case(config: QIStagedCaseConfig) -> dict[str, Any]:
    case = QI_CASES.get(str(config.policy_case), {})
    return case if isinstance(case, dict) else {}


def _policy_boundary_reference(config: QIStagedCaseConfig) -> dict[str, Any]:
    boundary = _policy_case(config).get("boundary_reference_preconditioner", {})
    return boundary if isinstance(boundary, dict) else {}


def _policy_value(config: QIStagedCaseConfig, case_key: str, attr: str | None = None) -> Any:
    case = _policy_case(config)
    if case.get(case_key) is not None:
        return case[case_key]
    return getattr(config, attr or case_key)


def _policy_or_reference_value(config: QIStagedCaseConfig, case_key: str, reference_key: str, attr: str) -> Any:
    case = _policy_case(config)
    boundary = _policy_boundary_reference(config)
    if case.get(case_key) is not None:
        return case[case_key]
    if boundary.get(reference_key) is not None:
        return boundary[reference_key]
    return getattr(config, attr)


def _resolution_value(
    config: QIStagedCaseConfig,
    resolution_key: str,
    item_key: str,
    attr: str,
) -> int | None:
    resolution = _policy_case(config).get(resolution_key, {})
    if isinstance(resolution, dict) and resolution.get(item_key) is not None:
        return int(resolution[item_key])
    value = getattr(config, attr)
    return None if value is None else int(value)


def _jsonable_policy_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser())
    if isinstance(value, tuple):
        return [_jsonable_policy_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_policy_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_policy_value(item) for key, item in value.items()}
    return value


def _reference_input(config: QIStagedCaseConfig) -> Path | None:
    if config.reference_input is not None:
        return Path(config.reference_input)
    boundary = _policy_boundary_reference(config)
    if bool(boundary.get("enabled")) and boundary.get("reference_input") is not None:
        return Path(boundary["reference_input"])
    return None


def _reference_lambdas(config: QIStagedCaseConfig) -> tuple[float, ...] | None:
    if config.reference_lambdas is None:
        return None
    boundary = _policy_boundary_reference(config)
    if config.reference_lambdas == DEFAULT_REFERENCE_LAMBDAS and boundary.get("lambdas") is not None:
        return tuple(float(value) for value in boundary["lambdas"])
    return tuple(float(value) for value in config.reference_lambdas)


def _reference_preconditioner_overrides(config: QIStagedCaseConfig) -> dict[str, Any]:
    boundary = dict(_policy_boundary_reference(config))
    reference_input = _reference_input(config)
    if reference_input is not None:
        boundary["enabled"] = True
        boundary["reference_input"] = reference_input
    if boundary:
        boundary["max_mode"] = int(config.max_mode)
    target_aspect = _policy_value(config, "target_aspect")
    if boundary and target_aspect is not None:
        boundary["target_aspect"] = float(target_aspect)
    lambdas = _reference_lambdas(config)
    if config.reference_lambdas is None:
        boundary.pop("lambdas", None)
    elif lambdas is not None:
        boundary["lambdas"] = lambdas
    if boundary:
        boundary["accept_as_baseline"] = bool(
            boundary.get("accept_as_baseline", config.reference_accept_as_baseline)
            or config.reference_accept_as_baseline
        )
    return _jsonable_policy_value(boundary)


def _build_qi_staged_env(config: QIStagedCaseConfig) -> dict[str, str]:
    """Return process environment for the QI standalone subprocess."""

    env = dict(os.environ)
    _prepend_pythonpath(env, ROOT, SCRIPT_DIR)
    worker_jax_platforms = sweep._normalize_worker_jax_platforms(config.worker_jax_platforms)
    if worker_jax_platforms is not None:
        env["JAX_PLATFORMS"] = worker_jax_platforms
    return env


def _build_qi_staged_args(config: QIStagedCaseConfig) -> list[str]:
    """Return explicit CLI overrides for ``QI_optimization.py``.

    The QI example is intentionally editable by changing variables at the top.
    Sweep drivers should still pass explicit command-line overrides so they do
    not depend on stale environment-variable plumbing.
    """

    args = [
        str(QI_SCRIPT),
        "--input-file",
        str(Path(config.input_file).expanduser()),
        "--output-dir",
        str(Path(config.output_dir).expanduser()),
        "--max-mode",
        str(int(config.max_mode)),
        "--use-mode-continuation" if str(config.policy) == "continuation" else "--no-use-mode-continuation",
        "--use-ess" if bool(config.use_ess) else "--no-use-ess",
        "--make-plots" if bool(config.make_plots) else "--no-make-plots",
        "--stage-mode-policy",
        str(config.stage_mode_policy),
    ]
    reference_input = _reference_input(config)
    if reference_input is not None:
        args.extend(
            [
                "--use-reference-family-seed",
                "--reference-input",
                str(reference_input.expanduser()),
            ]
        )
        reference_lambdas = _reference_lambdas(config)
        if reference_lambdas is not None:
            args.extend(
                [
                    "--reference-lambdas",
                    ",".join(f"{float(value):.12g}" for value in reference_lambdas),
                ]
            )
        reference_overrides = _reference_preconditioner_overrides(config)
        if reference_overrides:
            reference_path = Path(config.output_dir).expanduser() / "boundary_reference_preconditioner.json"
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_text(json.dumps(reference_overrides, indent=2, sort_keys=True) + "\n")
            args.extend(["--boundary-reference-json", str(reference_path)])
        args.append(
            "--accept-boundary-reference-baseline"
            if bool(reference_overrides.get("accept_as_baseline", config.reference_accept_as_baseline))
            else "--no-accept-boundary-reference-baseline"
        )
    else:
        args.append("--no-use-reference-family-seed")
    if config.solver_device is not None:
        args.extend(["--solver-device", str(config.solver_device)])
    max_nfev = config.max_nfev
    if max_nfev is None:
        max_nfev = _policy_case(config).get("max_nfev")
    if max_nfev is not None:
        args.extend(["--max-nfev", str(int(max_nfev))])
    if config.continuation_nfev is not None:
        args.extend(["--continuation-nfev", str(int(config.continuation_nfev))])
    if config.inner_max_iter is not None:
        args.extend(["--inner-max-iter", str(int(config.inner_max_iter))])
    if config.inner_ftol is not None:
        args.extend(["--inner-ftol", str(float(config.inner_ftol))])
    if config.trial_max_iter is not None:
        args.extend(["--trial-max-iter", str(int(config.trial_max_iter))])
    if config.trial_ftol is not None:
        args.extend(["--trial-ftol", str(float(config.trial_ftol))])
    if config.ess_alpha is not None:
        args.extend(["--ess-alpha", str(float(config.ess_alpha))])
    if config.method is not None:
        args.extend(["--method", str(config.method)])
    physics_args = {
        "--target-aspect": _policy_value(config, "target_aspect"),
        "--target-abs-iota-min": _policy_value(config, "target_abs_iota_min"),
        "--max-mirror-ratio": _policy_or_reference_value(
            config, "mirror_threshold", "max_mirror_ratio", "max_mirror_ratio"
        ),
        "--max-elongation": _policy_or_reference_value(config, "max_elongation", "max_elongation", "max_elongation"),
        "--qi-gate-smooth-max": _policy_or_reference_value(
            config, "qi_gate_smooth_max", "smooth_qi_max", "qi_gate_smooth_max"
        ),
        "--qi-gate-legacy-max": _policy_or_reference_value(
            config, "qi_gate_legacy_max", "legacy_qi_max", "qi_gate_legacy_max"
        ),
        "--qi-ceiling-max": _policy_value(config, "qi_ceiling_max"),
        "--qi-ceiling-smooth-penalty": _policy_value(config, "qi_ceiling_smooth_penalty"),
        "--mirror-weight": _policy_value(config, "mirror_weight"),
        "--elongation-weight": _policy_value(config, "elongation_weight"),
    }
    for flag, value in physics_args.items():
        if value is not None:
            args.extend([flag, str(float(value))])
    mirror_surface_index = _policy_value(config, "mirror_surface_index")
    if mirror_surface_index is not None:
        args.extend(["--mirror-surface-index", str(int(mirror_surface_index))])
    qi_resolution_args = {
        "--qi-mboz": _resolution_value(config, "optimization_qi_resolution", "mboz", "qi_mboz"),
        "--qi-nboz": _resolution_value(config, "optimization_qi_resolution", "nboz", "qi_nboz"),
        "--qi-nphi": _resolution_value(config, "optimization_qi_resolution", "nphi", "qi_nphi"),
        "--qi-nalpha": _resolution_value(config, "optimization_qi_resolution", "nalpha", "qi_nalpha"),
        "--qi-n-bounce": _resolution_value(config, "optimization_qi_resolution", "n_bounce", "qi_n_bounce"),
        "--audit-qi-mboz": _resolution_value(config, "audit_qi_resolution", "mboz", "audit_qi_mboz"),
        "--audit-qi-nboz": _resolution_value(config, "audit_qi_resolution", "nboz", "audit_qi_nboz"),
        "--audit-qi-nphi": _resolution_value(config, "audit_qi_resolution", "nphi", "audit_qi_nphi"),
        "--audit-qi-nalpha": _resolution_value(config, "audit_qi_resolution", "nalpha", "audit_qi_nalpha"),
        "--audit-qi-n-bounce": _resolution_value(
            config, "audit_qi_resolution", "n_bounce", "audit_qi_n_bounce"
        ),
    }
    for flag, value in qi_resolution_args.items():
        if value is not None:
            args.extend([flag, str(int(value))])
    mirror_ramp_stages_config = config.mirror_ramp_stages
    if mirror_ramp_stages_config is None:
        policy_stages = _policy_case(config).get("mirror_ramp_stages")
        if policy_stages is not None:
            mirror_ramp_stages_config = tuple(policy_stages)
    if mirror_ramp_stages_config is not None:
        stages_path = Path(config.output_dir).expanduser() / "mirror_ramp_stages.json"
        stages_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_ramp_stages = [dict(stage) for stage in mirror_ramp_stages_config]
        if config.method is not None:
            # Stage JSON is authoritative inside QI_optimization.py, so patch it
            # along with --method to make showcase reruns reproducible.
            for stage in mirror_ramp_stages:
                stage["method"] = str(config.method)
        stages_path.write_text(json.dumps(mirror_ramp_stages, indent=2, sort_keys=True) + "\n")
        args.extend(["--mirror-ramp-stages-json", str(stages_path)])
    return args


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _history_metrics(history: dict[str, Any]) -> dict[str, Any]:
    profile = history.get("profile")
    profile = profile if isinstance(profile, dict) else {}
    return {
        "objective_final": _finite_float(history.get("objective_final")),
        "qs_final": _finite_float(history.get("qs_final")),
        "aspect_final": _finite_float(history.get("aspect_final")),
        "iota_final": _finite_float(history.get("iota_final")),
        "nfev": _finite_int(history.get("nfev")),
        "njev": _finite_int(history.get("njev")),
        "total_wall_time_s": _finite_float(history.get("total_wall_time_s")),
        "profile_wall_time_s": _finite_float(profile.get("total_wall_time_s")),
        "profile_solve_forward_trial_total_wall_time_s": _profile_wall_time(profile, "solve_forward_trial_total"),
        "profile_solve_forward_exact_total_wall_time_s": _profile_wall_time(profile, "solve_forward_exact_total"),
        "profile_exact_tape_build_wall_time_s": _profile_wall_time(profile, "exact_tape_build"),
        "profile_jacobian_total_wall_time_s": _profile_wall_time(profile, "jacobian_total"),
        "profile_write_wout_wall_time_s": _profile_wall_time(profile, "write_wout"),
    }


def _diagnostic_metrics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "qi_raw_total": _first_finite_float(diagnostics.get("qi_raw_total"), diagnostics.get("qi_smooth_total")),
        "qi_legacy_total": _finite_float(diagnostics.get("qi_legacy_total")),
        "qi_mirror_ratio_max": _finite_float(diagnostics.get("qi_mirror_ratio_max")),
        "qi_mirror_ratio_target": _finite_float(diagnostics.get("qi_mirror_ratio_target")),
        "qi_mirror_excess_max": _finite_float(diagnostics.get("qi_mirror_excess_max")),
        "qi_max_elongation": _finite_float(diagnostics.get("qi_max_elongation")),
        "qi_elongation_target": _finite_float(diagnostics.get("qi_elongation_target")),
        "qi_elongation_excess": _finite_float(diagnostics.get("qi_elongation_excess")),
        "qi_lgradb_min": _finite_float(diagnostics.get("qi_lgradb_min")),
        "qi_lgradb_threshold": _finite_float(diagnostics.get("qi_lgradb_threshold")),
        "qi_lgradb_excess_max": _finite_float(diagnostics.get("qi_lgradb_excess_max")),
    }


def _selected_boundary_reference_record(output_dir: Path) -> dict[str, Any]:
    """Return the selected boundary-reference candidate, if one was written."""

    summary_path = Path(output_dir) / "boundary_reference_preconditioner" / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        records = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(records, list):
        return {}
    selected = [record for record in records if isinstance(record, dict) and bool(record.get("selected"))]
    if selected:
        return selected[-1]
    finite = [
        record
        for record in records
        if isinstance(record, dict) and _finite_float(record.get("score")) is not None
    ]
    if not finite:
        return {}
    return min(finite, key=lambda record: float(record["score"]))


def _boundary_reference_partial_metrics(output_dir: Path) -> dict[str, Any]:
    """Map partial preconditioner metrics to sweep-result fields.

    A long QI staged solve can time out after the deterministic reference-family
    scan has already found a physically meaningful candidate but before the
    final history/diagnostics files are emitted.  Preserve those partial metrics
    so dashboards show what was achieved instead of a row of ``None`` values.
    """

    record = _selected_boundary_reference_record(output_dir)
    if not record:
        return {}
    return {
        "qs_final": _finite_float(record.get("smooth_qi")),
        "aspect_final": _finite_float(record.get("aspect")),
        "iota_final": _finite_float(record.get("mean_iota")),
        "qi_raw_total": _finite_float(record.get("smooth_qi")),
        "qi_legacy_total": _finite_float(record.get("legacy_qi")),
        "qi_mirror_ratio_max": _finite_float(record.get("mirror")),
        "qi_max_elongation": _finite_float(record.get("elongation")),
    }


def _stage_checkpoint_score(path: Path, record: dict[str, Any]) -> tuple[Any, ...]:
    """Rank QI stage checkpoints by diagnostic usefulness.

    Timeout handling should preserve the best completed stage evidence, not the
    newest file.  A later stage writes a pending checkpoint before its solve
    starts, and long runs can time out with that pending checkpoint copied to
    the root.  Prefer checkpoints with an actual completed stage history and
    exact diagnostics; use modification time only as a final tie-breaker.
    """

    history = record.get("history")
    diagnostics = record.get("diagnostics")
    promotion = record.get("promotion")
    history = history if isinstance(history, dict) else {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    promotion = promotion if isinstance(promotion, dict) else {}
    role = str(record.get("role", ""))

    has_history_result = any(
        _finite_float(history.get(key)) is not None
        for key in ("objective_final", "qs_final", "aspect_final", "iota_final", "total_wall_time_s")
    ) or _finite_int(history.get("nfev")) is not None
    has_exact_diagnostics = any(
        _finite_float(diagnostics.get(key)) is not None
        for key in (
            "qi_smooth_total",
            "qi_legacy_total",
            "qi_mirror_ratio_max",
            "qi_max_elongation",
        )
    )
    pending = role.endswith("_pending") or bool(promotion.get("stage_pending", False))
    pre_diagnostics = role.endswith("_pre_diagnostics") or bool(promotion.get("diagnostics_pending", False))
    passed_gate = bool(diagnostics.get("qi_engineering_gate_passed", False)) or bool(
        diagnostics.get("qi_seed_gate_passed", False)
    )
    smooth_qi = _first_finite_float(history.get("qs_final"), diagnostics.get("qi_smooth_total"))
    legacy_qi = _finite_float(diagnostics.get("qi_legacy_total"))
    mirror = _finite_float(diagnostics.get("qi_mirror_ratio_max"))
    mtime = path.stat().st_mtime

    return (
        int(has_history_result),
        int(has_exact_diagnostics),
        int(passed_gate),
        -int(pre_diagnostics),
        -int(pending),
        -float("inf") if smooth_qi is None else -float(smooth_qi),
        -float("inf") if legacy_qi is None else -float(legacy_qi),
        -float("inf") if mirror is None else -float(mirror),
        float(mtime),
    )


def _stage_checkpoint_record(output_dir: Path) -> dict[str, Any]:
    """Return the most useful QI stage checkpoint record, if available."""

    output_dir = Path(output_dir)
    root_checkpoint = output_dir / "stage_checkpoint.json"
    candidates = {
        path
        for path in (
            [root_checkpoint]
            + list(output_dir.glob("**/stage_checkpoint.json"))
            + list(output_dir.glob("**/qi_stage_checkpoint.json"))
        )
        if path.exists()
    }
    ranked_candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for path in candidates:
        record = _read_json(path)
        if record:
            ranked_candidates.append((_stage_checkpoint_score(path, record), record))
    if ranked_candidates:
        return max(ranked_candidates, key=lambda item: item[0])[1]
    return {}


def _stage_checkpoint_partial_metrics(output_dir: Path) -> dict[str, Any]:
    """Map the latest per-stage QI checkpoint to sweep-result fields."""

    record = _stage_checkpoint_record(output_dir)
    if not record:
        return {}
    history = record.get("history")
    diagnostics = record.get("diagnostics")
    history = history if isinstance(history, dict) else {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    return {
        "objective_final": _finite_float(history.get("objective_final")),
        "qs_final": _first_finite_float(history.get("qs_final"), diagnostics.get("qi_smooth_total")),
        "aspect_final": _first_finite_float(history.get("aspect_final"), diagnostics.get("aspect")),
        "iota_final": _first_finite_float(history.get("iota_final"), diagnostics.get("mean_iota")),
        "nfev": _finite_int(history.get("nfev")),
        "njev": _finite_int(history.get("njev")),
        "total_wall_time_s": _finite_float(history.get("total_wall_time_s")),
        "qi_raw_total": _first_finite_float(diagnostics.get("qi_raw_total"), diagnostics.get("qi_smooth_total")),
        "qi_legacy_total": _finite_float(diagnostics.get("qi_legacy_total")),
        "qi_mirror_ratio_max": _finite_float(diagnostics.get("qi_mirror_ratio_max")),
        "qi_mirror_ratio_target": _finite_float(diagnostics.get("qi_mirror_ratio_target")),
        "qi_max_elongation": _finite_float(diagnostics.get("qi_max_elongation")),
        "qi_elongation_target": _finite_float(diagnostics.get("qi_elongation_target")),
    }


def _has_partial_metrics(metrics: dict[str, Any]) -> bool:
    """Return True when a partial checkpoint carries at least one finite field."""

    return any(value is not None for value in metrics.values())


def _merge_partial_metrics(*metric_sets: dict[str, Any]) -> dict[str, Any]:
    """Merge partial metrics without letting missing later fields erase data."""

    merged: dict[str, Any] = {}
    for metrics in metric_sets:
        for key, value in metrics.items():
            if value is not None or key not in merged:
                merged[key] = value
    return merged


def annotate_case_result_from_partial_artifacts(result: sweep.CaseResult, output_dir: Path) -> bool:
    """Fill missing QI fields from partial staged artifacts.

    Returns ``True`` when any result field changed.  The success/crash status is
    intentionally left untouched: partial metrics explain a timeout but do not
    promote it to a passing optimization.
    """

    changed = False
    partial_metrics = _merge_partial_metrics(
        _boundary_reference_partial_metrics(Path(output_dir)),
        _stage_checkpoint_partial_metrics(Path(output_dir)),
    )
    for key, value in partial_metrics.items():
        if value is not None and getattr(result, key, None) is None:
            setattr(result, key, value)
            changed = True
    if changed and "partial" not in str(result.message):
        prefix = str(result.message).strip()
        suffix = "partial QI stage checkpoint metrics recorded"
        result.message = f"{prefix}; {suffix}" if prefix else suffix
    return changed


def _success_from_diagnostics(history: dict[str, Any], diagnostics: dict[str, Any], returncode: int) -> bool:
    if returncode != 0:
        return False
    if "qi_engineering_gate_passed" in diagnostics:
        return bool(diagnostics["qi_engineering_gate_passed"])
    if "qi_seed_gate_passed" in diagnostics:
        return bool(diagnostics["qi_seed_gate_passed"])
    return bool(history.get("success", False))


def _message_from_artifacts(history: dict[str, Any], diagnostics: dict[str, Any], returncode: int) -> str:
    pieces: list[str] = []
    if returncode != 0:
        pieces.append(f"QI staged subprocess exited with code {returncode}")
    if history.get("message"):
        pieces.append(str(history["message"]))
    failures = diagnostics.get("qi_gate_failures")
    if failures:
        pieces.append("QI gate failures: " + "; ".join(str(item) for item in failures))
    return "; ".join(pieces)


def _terminate_process_group(process: subprocess.Popen, *, grace_s: float = 5.0) -> None:
    """Terminate a staged QI subprocess and any children it spawned."""

    if process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=float(grace_s))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=float(grace_s))


def _run_qi_subprocess(
    cli_args: list[str],
    *,
    env: dict[str, str],
    stdout,
    stderr,
    timeout_s: float | None,
) -> int:
    """Run ``QI_optimization.py`` in a process group that cannot outlive us."""

    process = subprocess.Popen(
        [sys.executable, *cli_args],
        cwd=str(SCRIPT_DIR),
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    previous_handlers: dict[int, Any] = {}

    def _handle_parent_signal(signum, frame):  # noqa: ANN001
        _terminate_process_group(process, grace_s=1.0)
        previous = previous_handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
            return
        raise SystemExit(128 + int(signum))

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _handle_parent_signal)
    try:
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            raise
        return int(process.returncode)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def run_qi_staged_case(config: QIStagedCaseConfig) -> sweep.CaseResult:
    """Run ``QI_optimization.py`` and return a sweep-compatible result."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "qi_staged_stdout.log"
    stderr_path = output_dir / "qi_staged_stderr.log"
    env = _build_qi_staged_env(config)
    cli_args = _build_qi_staged_args(config)

    start = time.perf_counter()
    returncode = 0
    timeout_s = None if config.timeout_s in (None, 0) else float(config.timeout_s)
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            returncode = _run_qi_subprocess(
                cli_args,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout_s=timeout_s,
            )
        message_prefix = ""
    except subprocess.TimeoutExpired:
        returncode = 124
        message_prefix = f"QI staged subprocess timed out after {timeout_s:.1f} s"
    elapsed_s = time.perf_counter() - start

    history = _read_json(output_dir / "history.json")
    diagnostics = _read_json(output_dir / "diagnostics.json")
    history_metrics = _history_metrics(history)
    diagnostic_metrics = _diagnostic_metrics(diagnostics)
    stage_partial_metrics = _stage_checkpoint_partial_metrics(output_dir)
    boundary_partial_metrics = _boundary_reference_partial_metrics(output_dir)
    partial_metrics = _merge_partial_metrics(boundary_partial_metrics, stage_partial_metrics)
    for key, value in partial_metrics.items():
        if value is None:
            continue
        if key in history_metrics and history_metrics[key] is None:
            history_metrics[key] = value
        if key in diagnostic_metrics and diagnostic_metrics[key] is None:
            diagnostic_metrics[key] = value
    success = _success_from_diagnostics(history, diagnostics, returncode)
    message = _message_from_artifacts(history, diagnostics, returncode)
    if message_prefix:
        message = f"{message_prefix}; {message}" if message else message_prefix
    if partial_metrics and not success:
        suffix = (
            "partial QI stage checkpoint metrics recorded"
            if stage_partial_metrics
            else "partial boundary-reference metrics recorded"
        )
        message = f"{message}; {suffix}" if message else suffix
    if success and not message:
        message = "QI staged subprocess passed engineering gate"
    partial_metrics_available = _has_partial_metrics(partial_metrics)
    crashed = returncode != 0 and not (returncode == 124 and partial_metrics_available)

    wall_time_s = history_metrics.pop("total_wall_time_s")
    if wall_time_s is None:
        wall_time_s = elapsed_s
    return sweep.CaseResult(
        backend=str(config.backend_label),
        problem="qi",
        max_mode=int(config.max_mode),
        use_ess=bool(config.use_ess),
        success=bool(success),
        crashed=crashed,
        message=message,
        policy=str(config.policy),
        total_wall_time_s=wall_time_s,
        output_dir=str(output_dir),
        solver_device=config.solver_device,
        jax_platforms=sweep._normalize_worker_jax_platforms(config.worker_jax_platforms),
        input_file=str(config.input_file),
        input_nfp=_input_nfp(Path(config.input_file)),
        target_aspect=_finite_float(diagnostics.get("target_aspect") or history.get("target_aspect")),
        iota_abs_min=_finite_float(diagnostics.get("target_abs_iota_min")),
        qi_qp_preseed=False,
        qi_qi_preseed=True,
        qi_jit_booz=True,
        **history_metrics,
        **diagnostic_metrics,
    )

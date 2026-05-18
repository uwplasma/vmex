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
import subprocess
import sys
import time
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
QI_SCRIPT = SCRIPT_DIR / "QI_optimization.py"
DEFAULT_REFERENCE_LAMBDAS = (
    0.994,
    0.995,
    0.996,
    0.997,
    0.998,
    0.999,
    1.0,
    1.001,
    1.002,
    1.004,
    1.006,
    1.008,
    1.010,
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_qs_ess_sweep as sweep
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
    backend_label: str = "cpu"
    solver_device: str | None = None
    worker_jax_platforms: str | None = None
    use_ess: bool = True
    max_nfev: int | None = None
    inner_max_iter: int | None = None
    inner_ftol: float | None = None
    trial_max_iter: int | None = None
    trial_ftol: float | None = None
    ess_alpha: float | None = None
    reference_lambdas: tuple[float, ...] | None = DEFAULT_REFERENCE_LAMBDAS
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


def _input_nfp(input_file: Path) -> int | None:
    try:
        return _finite_int(read_indata(input_file).scalars.get("NFP"))
    except Exception:
        return None


def _prepend_pythonpath(env: dict[str, str], *paths: Path) -> None:
    current = env.get("PYTHONPATH")
    additions = [str(path) for path in paths]
    env["PYTHONPATH"] = os.pathsep.join(additions + ([current] if current else []))


def _bool_env(value: bool) -> str:
    return "1" if bool(value) else "0"


def _build_qi_staged_env(config: QIStagedCaseConfig) -> dict[str, str]:
    """Return environment variables for the QI standalone subprocess."""

    env = dict(os.environ)
    _prepend_pythonpath(env, ROOT, SCRIPT_DIR)
    env.update(
        {
            "VMEC_JAX_QI_INPUT": str(Path(config.input_file).expanduser()),
            "VMEC_JAX_QI_OUTPUT_DIR": str(Path(config.output_dir).expanduser()),
            "VMEC_JAX_QI_LABEL": str(config.name),
            "VMEC_JAX_QI_RUN_CASE": str(config.name),
            "VMEC_JAX_QI_POLICY_CASE": str(config.policy_case),
            "VMEC_JAX_QI_MAX_MODE": str(int(config.max_mode)),
            "VMEC_JAX_QI_USE_MODE_CONTINUATION": _bool_env(str(config.policy) == "continuation"),
            "VMEC_JAX_QI_USE_ESS": _bool_env(config.use_ess),
            "VMEC_JAX_QI_MAKE_PLOTS": _bool_env(config.make_plots),
        }
    )
    if config.reference_input is not None:
        env["VMEC_JAX_QI_REFERENCE_INPUT"] = str(Path(config.reference_input).expanduser())
        if config.reference_lambdas is not None:
            env["VMEC_JAX_QI_REFERENCE_LAMBDAS"] = ",".join(
                f"{float(value):.12g}" for value in config.reference_lambdas
            )
    if config.solver_device is not None:
        env["VMEC_JAX_QI_SOLVER_DEVICE"] = str(config.solver_device)
    if config.worker_jax_platforms is not None:
        env["JAX_PLATFORMS"] = str(config.worker_jax_platforms)
    if config.max_nfev is not None:
        env["VMEC_JAX_QI_MAX_NFEV"] = str(int(config.max_nfev))
    if config.inner_max_iter is not None:
        env["VMEC_JAX_QI_INNER_MAX_ITER"] = str(int(config.inner_max_iter))
    if config.inner_ftol is not None:
        env["VMEC_JAX_QI_INNER_FTOL"] = str(float(config.inner_ftol))
    if config.trial_max_iter is not None:
        env["VMEC_JAX_QI_TRIAL_MAX_ITER"] = str(int(config.trial_max_iter))
    if config.trial_ftol is not None:
        env["VMEC_JAX_QI_TRIAL_FTOL"] = str(float(config.trial_ftol))
    if config.ess_alpha is not None:
        env["VMEC_JAX_QI_ESS_ALPHA"] = str(float(config.ess_alpha))
    return env


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
        "profile_solve_forward_trial_total_wall_time_s": _finite_float(
            profile.get("solve_forward_trial_total_wall_time_s")
        ),
        "profile_solve_forward_exact_total_wall_time_s": _finite_float(
            profile.get("solve_forward_exact_total_wall_time_s")
        ),
        "profile_exact_tape_build_wall_time_s": _finite_float(profile.get("exact_tape_build_wall_time_s")),
        "profile_jacobian_total_wall_time_s": _finite_float(profile.get("jacobian_total_wall_time_s")),
        "profile_write_wout_wall_time_s": _finite_float(profile.get("write_wout_wall_time_s")),
    }


def _diagnostic_metrics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "qi_raw_total": _finite_float(diagnostics.get("qi_raw_total") or diagnostics.get("qi_smooth_total")),
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


def annotate_case_result_from_partial_artifacts(result: sweep.CaseResult, output_dir: Path) -> bool:
    """Fill missing QI fields from partial staged artifacts.

    Returns ``True`` when any result field changed.  The success/crash status is
    intentionally left untouched: partial metrics explain a timeout but do not
    promote it to a passing optimization.
    """

    changed = False
    for key, value in _boundary_reference_partial_metrics(Path(output_dir)).items():
        if value is not None and getattr(result, key, None) is None:
            setattr(result, key, value)
            changed = True
    if changed and "partial boundary-reference metrics" not in str(result.message):
        prefix = str(result.message).strip()
        suffix = "partial boundary-reference metrics recorded"
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


def run_qi_staged_case(config: QIStagedCaseConfig) -> sweep.CaseResult:
    """Run ``QI_optimization.py`` and return a sweep-compatible result."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "qi_staged_stdout.log"
    stderr_path = output_dir / "qi_staged_stderr.log"
    env = _build_qi_staged_env(config)

    start = time.perf_counter()
    returncode = 0
    timeout_s = None if config.timeout_s in (None, 0) else float(config.timeout_s)
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(
                [sys.executable, str(QI_SCRIPT)],
                cwd=str(SCRIPT_DIR),
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_s,
                check=False,
            )
        returncode = int(completed.returncode)
        message_prefix = ""
    except subprocess.TimeoutExpired:
        returncode = 124
        message_prefix = f"QI staged subprocess timed out after {timeout_s:.1f} s"
    elapsed_s = time.perf_counter() - start

    history = _read_json(output_dir / "history.json")
    diagnostics = _read_json(output_dir / "diagnostics.json")
    history_metrics = _history_metrics(history)
    diagnostic_metrics = _diagnostic_metrics(diagnostics)
    partial_metrics = _boundary_reference_partial_metrics(output_dir)
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
        suffix = "partial boundary-reference metrics recorded"
        message = f"{message}; {suffix}" if message else suffix
    if success and not message:
        message = "QI staged subprocess passed engineering gate"

    wall_time_s = history_metrics.pop("total_wall_time_s")
    if wall_time_s is None:
        wall_time_s = elapsed_s
    return sweep.CaseResult(
        backend=str(config.backend_label),
        problem="qi",
        max_mode=int(config.max_mode),
        use_ess=bool(config.use_ess),
        success=bool(success),
        crashed=returncode != 0,
        message=message,
        policy=str(config.policy),
        total_wall_time_s=wall_time_s,
        output_dir=str(output_dir),
        solver_device=config.solver_device,
        jax_platforms=config.worker_jax_platforms,
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

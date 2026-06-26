#!/usr/bin/env python
"""Constraint-filtered local search for far-seed QI optimization.

The basin survey and promotion tools show that scalar penalty objectives can
jump between incompatible basins: low-QI/low-iota or high-iota/bad-QI.  This
diagnostic uses a filter-search rule instead.  A trial step is accepted only
when it preserves the already-satisfied gates and improves the currently failed
gate: first QI, then iota, then engineering constraints.

This is not intended to replace differentiable local optimization.  It is a
bounded feasibility search used to decide whether a seed has a nearby path that
keeps QI while improving iota/mirror/elongation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "examples" / "data"
DEFAULT_INPUT = DATA_DIR / "input.QI_stel_seed_3127"
DEFAULT_OUTPUT_DIR = Path("results/diagnostics/qi_filter_search")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.qi_basin_survey import (  # noqa: E402
    SurveyTargets,
    _axis_order,
    _finite_float,
    _normalize_direction,
)
from tools.diagnostics.qi_landscape_scan import (  # noqa: E402
    DEFAULT_SURFACES,
    build_stage,
    parse_surfaces,
    resolve_input_path,
)
from vmec_jax._compat import enable_x64  # noqa: E402
from vmec_jax.optimization import boundary_param_names, create_x_scale  # noqa: E402
from vmec_jax.quasi_isodynamic.diagnostics import QIDiagnosticOptions, qi_diagnostics_from_state  # noqa: E402


enable_x64(True)

METRIC_KEYS = (
    "qi_smooth_total",
    "qi_legacy_total",
    "qi_mirror_ratio_max",
    "qi_max_elongation",
    "mean_iota",
    "aspect",
)


@dataclass(frozen=True)
class FilterSearchOptions:
    """Acceptance-filter controls for local QI feasibility search."""

    qi_relax: float = 1.5
    legacy_relax: float = 1.5
    mirror_relax: float = 1.25
    elongation_relax: float = 1.25
    aspect_tolerance: float = 2.0
    min_qi_gain: float = 1.0e-5
    min_iota_gain: float = 5.0e-3
    min_engineering_gain: float = 1.0e-3


@dataclass(frozen=True)
class FilterDecision:
    """Decision for one candidate according to the QI feasibility filter."""

    accepted: bool
    phase: str
    reason: str
    current_key: tuple[float, ...]
    candidate_key: tuple[float, ...]


def metric_value(metrics: dict[str, Any], key: str, *, default: float = float("inf")) -> float:
    value = _finite_float(metrics.get(key))
    return default if value is None else float(value)


def summarize_metrics(diagnostics: dict[str, Any]) -> dict[str, float | None]:
    """Extract JSON-friendly scalar metrics used by the filter."""

    return {key: _finite_float(diagnostics.get(key)) for key in METRIC_KEYS}


def _safe_limit(target: float, current: float, relax: float) -> float:
    if not np.isfinite(current):
        return float(target)
    return float(max(float(target), float(current) * float(relax)))


def gate_status(
    metrics: dict[str, Any],
    *,
    targets: SurveyTargets,
    options: FilterSearchOptions,
) -> dict[str, bool]:
    smooth = metric_value(metrics, "qi_smooth_total")
    legacy = metric_value(metrics, "qi_legacy_total")
    mirror = metric_value(metrics, "qi_mirror_ratio_max")
    elongation = metric_value(metrics, "qi_max_elongation")
    iota = abs(metric_value(metrics, "mean_iota", default=0.0))
    aspect = metric_value(metrics, "aspect")
    return {
        "smooth_qi": smooth <= float(targets.smooth_qi_max),
        "legacy_qi": legacy <= float(targets.legacy_qi_max),
        "qi": smooth <= float(targets.smooth_qi_max) and legacy <= float(targets.legacy_qi_max),
        "iota": iota >= float(targets.abs_iota_min),
        "mirror": mirror <= float(targets.mirror_ratio_max),
        "elongation": elongation <= float(targets.max_elongation),
        "aspect": abs(aspect - float(targets.target_aspect)) <= float(options.aspect_tolerance),
    }


def filter_phase(metrics: dict[str, Any], *, targets: SurveyTargets, options: FilterSearchOptions) -> str:
    """Return the next active gate to improve."""

    status = gate_status(metrics, targets=targets, options=options)
    if not status["qi"]:
        return "qi"
    if not status["iota"]:
        return "iota"
    if not (status["mirror"] and status["elongation"]):
        return "engineering"
    return "polish"


def engineering_violation(metrics: dict[str, Any], *, targets: SurveyTargets) -> float:
    mirror = metric_value(metrics, "qi_mirror_ratio_max")
    elongation = metric_value(metrics, "qi_max_elongation")
    return float(
        max(0.0, mirror - float(targets.mirror_ratio_max)) / max(float(targets.mirror_ratio_max), 1.0e-16)
        + max(0.0, elongation - float(targets.max_elongation)) / max(float(targets.max_elongation), 1.0e-16)
    )


def filter_key(metrics: dict[str, Any], *, phase: str, targets: SurveyTargets) -> tuple[float, ...]:
    smooth = metric_value(metrics, "qi_smooth_total")
    legacy = metric_value(metrics, "qi_legacy_total")
    iota_gap = max(0.0, float(targets.abs_iota_min) - abs(metric_value(metrics, "mean_iota", default=0.0)))
    engineering = engineering_violation(metrics, targets=targets)
    aspect_err = abs(metric_value(metrics, "aspect") - float(targets.target_aspect))
    if phase == "qi":
        return (smooth / max(float(targets.smooth_qi_max), 1.0e-16), legacy / max(float(targets.legacy_qi_max), 1.0e-16), iota_gap, engineering, aspect_err)
    if phase == "iota":
        return (iota_gap, smooth / max(float(targets.smooth_qi_max), 1.0e-16), legacy / max(float(targets.legacy_qi_max), 1.0e-16), engineering, aspect_err)
    if phase == "engineering":
        return (engineering, smooth / max(float(targets.smooth_qi_max), 1.0e-16), legacy / max(float(targets.legacy_qi_max), 1.0e-16), iota_gap, aspect_err)
    return (smooth + legacy + engineering + iota_gap + aspect_err,)


def candidate_preserves_required_gates(
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    phase: str,
    targets: SurveyTargets,
    options: FilterSearchOptions,
) -> tuple[bool, str]:
    """Check hard preservation constraints for the current filter phase."""

    aspect = metric_value(candidate, "aspect")
    if abs(aspect - float(targets.target_aspect)) > float(options.aspect_tolerance):
        return False, "aspect outside filter tolerance"
    smooth_limit = _safe_limit(targets.smooth_qi_max, metric_value(current, "qi_smooth_total"), options.qi_relax)
    legacy_limit = _safe_limit(targets.legacy_qi_max, metric_value(current, "qi_legacy_total"), options.legacy_relax)
    if phase != "qi":
        if metric_value(candidate, "qi_smooth_total") > smooth_limit:
            return False, "smooth QI would leave preserved basin"
        if metric_value(candidate, "qi_legacy_total") > legacy_limit:
            return False, "legacy QI would leave preserved basin"
    if phase == "engineering":
        if abs(metric_value(candidate, "mean_iota", default=0.0)) < float(targets.abs_iota_min):
            return False, "iota floor would be lost"
    if phase == "polish":
        status = gate_status(candidate, targets=targets, options=options)
        failed = [name for name, ok in status.items() if not ok]
        if failed:
            return False, "polish candidate would lose gates: " + ", ".join(failed)
    return True, "preserved required gates"


def filter_decision(
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    targets: SurveyTargets = SurveyTargets(),
    options: FilterSearchOptions = FilterSearchOptions(),
) -> FilterDecision:
    """Decide whether to accept one trial candidate."""

    phase = filter_phase(current, targets=targets, options=options)
    preserves, reason = candidate_preserves_required_gates(
        current,
        candidate,
        phase=phase,
        targets=targets,
        options=options,
    )
    current_key = filter_key(current, phase=phase, targets=targets)
    candidate_key = filter_key(candidate, phase=phase, targets=targets)
    if not preserves:
        return FilterDecision(False, phase, reason, current_key, candidate_key)
    if phase == "qi":
        gain = current_key[0] + current_key[1] - candidate_key[0] - candidate_key[1]
        if gain >= float(options.min_qi_gain):
            return FilterDecision(True, phase, "QI filter improved", current_key, candidate_key)
        return FilterDecision(False, phase, "QI gain below threshold", current_key, candidate_key)
    if phase == "iota":
        current_iota = abs(metric_value(current, "mean_iota", default=0.0))
        candidate_iota = abs(metric_value(candidate, "mean_iota", default=0.0))
        if candidate_iota - current_iota >= float(options.min_iota_gain):
            return FilterDecision(True, phase, "iota filter improved", current_key, candidate_key)
        return FilterDecision(False, phase, "iota gain below threshold", current_key, candidate_key)
    if phase == "engineering":
        gain = current_key[0] - candidate_key[0]
        if gain >= float(options.min_engineering_gain):
            return FilterDecision(True, phase, "engineering filter improved", current_key, candidate_key)
        return FilterDecision(False, phase, "engineering gain below threshold", current_key, candidate_key)
    if candidate_key < current_key:
        return FilterDecision(True, phase, "polish filter improved", current_key, candidate_key)
    return FilterDecision(False, phase, "polish key did not improve", current_key, candidate_key)


def generate_trial_directions(
    *,
    names: Sequence[str],
    x_scale: Sequence[float],
    axis_count: int,
    n_random: int,
    rng: np.random.Generator,
    direction_families: Sequence[str],
) -> list[tuple[str, np.ndarray]]:
    """Return normalized search directions in parameter space."""

    names = tuple(str(name) for name in names)
    x_scale_arr = np.asarray(x_scale, dtype=float)
    families = {str(item).strip().lower() for item in direction_families if str(item).strip()}
    directions: list[tuple[str, np.ndarray]] = []
    if "axes" in families:
        for idx in _axis_order(names, x_scale_arr)[: max(0, int(axis_count))]:
            direction = np.zeros(len(names), dtype=float)
            direction[idx] = 1.0
            directions.append((f"axis+:{names[idx]}", direction.copy()))
            directions.append((f"axis-:{names[idx]}", -direction.copy()))
    if "rademacher" in families:
        for sample in range(max(0, int(n_random))):
            directions.append((f"rademacher:{sample:03d}", rng.choice(np.asarray([-1.0, 1.0]), size=len(names))))
    if "gaussian" in families:
        for sample in range(max(0, int(n_random))):
            directions.append((f"gaussian:{sample:03d}", rng.normal(size=len(names))))
    return [(label, _normalize_direction(direction)) for label, direction in directions]


def _make_qi_options(args: argparse.Namespace) -> QIDiagnosticOptions:
    return QIDiagnosticOptions(
        surfaces=tuple(args.surfaces),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        nphi=int(args.nphi),
        nalpha=int(args.nalpha),
        n_bounce=int(args.n_bounce),
        include_bounce_endpoints=bool(args.include_bounce_endpoints),
        phimin=float(args.phimin),
        jit_booz=not bool(args.no_jit_booz),
        mirror_threshold=float(args.mirror_threshold),
        mirror_ntheta=int(args.mirror_ntheta),
        mirror_nphi=int(args.mirror_nphi),
        mirror_surface_index=args.mirror_surface_index,
        elongation_threshold=float(args.max_elongation),
        elongation_ntheta=int(args.elongation_ntheta),
        elongation_nphi=int(args.elongation_nphi),
    )


def evaluate_params(stage, params: np.ndarray, *, options: QIDiagnosticOptions, exact_solve: bool) -> dict[str, Any]:
    optimizer = stage.optimizer
    state = optimizer._solve_exact_with_tape(params) if bool(exact_solve) else optimizer._solve_forward(params, trial=True)
    diagnostics = qi_diagnostics_from_state(
        state=state,
        static=stage.ctx.static,
        indata=stage.ctx.indata,
        signgs=stage.ctx.signgs,
        surfaces=options.surfaces,
        options=options,
        flux_local=stage.ctx.flux,
        prof_local={"pressure": stage.ctx.pressure},
        pressure_local=stage.ctx.pressure,
    )
    return {"metrics": summarize_metrics(diagnostics), "diagnostics": diagnostics}


def write_history(history: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(history), indent=2, sort_keys=True) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as f:
        fields = [
            "iteration",
            "accepted",
            "phase",
            "label",
            "radius",
            "reason",
            *METRIC_KEYS,
        ]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in history:
            row = {field: record.get(field) for field in fields}
            row.update(record.get("metrics", {}))
            writer.writerow(row)


def run_filter_search(stage, *, args: argparse.Namespace, targets: SurveyTargets, options: FilterSearchOptions) -> dict[str, Any]:
    names = boundary_param_names(stage.specs)
    x_scale = create_x_scale(stage.specs, alpha=float(args.alpha))
    qi_options = _make_qi_options(args)
    rng = np.random.default_rng(int(args.rng_seed))
    params_current = np.zeros(len(names), dtype=float)
    radius = float(args.radius)
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current = evaluate_params(stage, params_current, options=qi_options, exact_solve=bool(args.exact_solve))
    current_metrics = dict(current["metrics"])
    history.append(
        {
            "iteration": 0,
            "accepted": True,
            "phase": filter_phase(current_metrics, targets=targets, options=options),
            "label": "initial",
            "radius": 0.0,
            "reason": "initial point",
            "params": params_current.tolist(),
            "metrics": current_metrics,
        }
    )

    accepted_count = 0
    for iteration in range(1, int(args.max_iterations) + 1):
        directions = generate_trial_directions(
            names=names,
            x_scale=x_scale,
            axis_count=int(args.axis_count),
            n_random=int(args.n_random),
            rng=rng,
            direction_families=args.directions,
        )
        max_trials = int(args.max_trials_per_iteration)
        if max_trials > 0:
            directions = directions[:max_trials]
        best: tuple[tuple[float, ...], dict[str, Any], np.ndarray] | None = None
        for label, direction in directions:
            params_trial = params_current + radius * np.asarray(x_scale, dtype=float) * direction
            try:
                evaluated = evaluate_params(stage, params_trial, options=qi_options, exact_solve=bool(args.exact_solve))
                trial_metrics = dict(evaluated["metrics"])
                decision = filter_decision(current_metrics, trial_metrics, targets=targets, options=options)
                record = {
                    "iteration": iteration,
                    "accepted": bool(decision.accepted),
                    "phase": decision.phase,
                    "label": label,
                    "radius": radius,
                    "reason": decision.reason,
                    "params": params_trial.tolist(),
                    "metrics": trial_metrics,
                    "current_key": list(decision.current_key),
                    "candidate_key": list(decision.candidate_key),
                }
            except Exception as exc:  # noqa: BLE001 - diagnostic search continues.
                record = {
                    "iteration": iteration,
                    "accepted": False,
                    "phase": filter_phase(current_metrics, targets=targets, options=options),
                    "label": label,
                    "radius": radius,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "params": params_trial.tolist(),
                    "metrics": {},
                }
                decision = None
            history.append(record)
            if bool(args.verbose):
                metrics = record.get("metrics", {})
                print(
                    f"iter={iteration} label={label} accepted={record['accepted']} "
                    f"phase={record['phase']} reason={record['reason']} "
                    f"QI={metrics.get('qi_smooth_total')} iota={metrics.get('mean_iota')}",
                    flush=True,
                )
            write_history(history, output_dir / "history.json")
            if decision is not None and decision.accepted:
                candidate_key = tuple(record["candidate_key"])
                if best is None or candidate_key < best[0]:
                    best = (candidate_key, record, params_trial)
        if best is None:
            radius *= float(args.shrink)
            write_history(history, output_dir / "history.json")
            if radius < float(args.min_radius):
                break
            continue
        _key, accepted_record, params_current = best
        current_metrics = dict(accepted_record["metrics"])
        accepted_count += 1
        if bool(args.save_accepted_inputs):
            stage.optimizer.save_input(Path(args.output_dir) / f"input.accepted_{accepted_count:03d}", params_current)
        write_history(history, output_dir / "history.json")
        if all(gate_status(current_metrics, targets=targets, options=options).values()):
            break
    final_phase = filter_phase(current_metrics, targets=targets, options=options)
    final_status = gate_status(current_metrics, targets=targets, options=options)
    stage.optimizer.save_input(output_dir / "input.final", params_current)
    report = {
        "kind": "qi_filter_search",
        "input": str(args.input),
        "output_dir": str(output_dir),
        "wall_time_s": time.perf_counter() - t0,
        "accepted_count": int(accepted_count),
        "final_phase": final_phase,
        "final_status": final_status,
        "final_metrics": current_metrics,
        "final_params": params_current.tolist(),
        "targets": asdict(targets),
        "options": asdict(options),
        "history_path": str(output_dir / "history.json"),
    }
    write_history(history, output_dir / "history.json")
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-mode", type=int, default=3)
    parser.add_argument("--min-vmec-mode", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--radius", type=float, default=2.0e-2)
    parser.add_argument("--min-radius", type=float, default=1.0e-3)
    parser.add_argument("--shrink", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-trials-per-iteration", type=int, default=0)
    parser.add_argument("--n-random", type=int, default=4)
    parser.add_argument("--axis-count", type=int, default=4)
    parser.add_argument("--rng-seed", type=int, default=20260515)
    parser.add_argument("--directions", default="axes,rademacher")
    parser.add_argument("--surfaces", type=parse_surfaces, default=DEFAULT_SURFACES)
    parser.add_argument("--mboz", type=int, default=8)
    parser.add_argument("--nboz", type=int, default=8)
    parser.add_argument("--nphi", type=int, default=41)
    parser.add_argument("--nalpha", type=int, default=9)
    parser.add_argument("--n-bounce", type=int, default=11)
    parser.add_argument("--include-bounce-endpoints", action="store_true")
    parser.add_argument("--mirror-threshold", type=float, default=0.35)
    parser.add_argument("--mirror-ntheta", type=int, default=32)
    parser.add_argument("--mirror-nphi", type=int, default=32)
    parser.add_argument("--mirror-surface-index", type=int, default=None)
    parser.add_argument("--max-elongation", type=float, default=8.0)
    parser.add_argument("--elongation-ntheta", type=int, default=24)
    parser.add_argument("--elongation-nphi", type=int, default=8)
    parser.add_argument("--target-aspect", type=float, default=10.0)
    parser.add_argument("--aspect-tolerance", type=float, default=2.0)
    parser.add_argument("--abs-iota-min", type=float, default=0.41)
    parser.add_argument("--smooth-qi-max", type=float, default=2.0e-3)
    parser.add_argument("--legacy-qi-max", type=float, default=2.0e-3)
    parser.add_argument("--qi-relax", type=float, default=1.5)
    parser.add_argument("--legacy-relax", type=float, default=1.5)
    parser.add_argument("--min-qi-gain", type=float, default=1.0e-5)
    parser.add_argument("--min-iota-gain", type=float, default=5.0e-3)
    parser.add_argument("--min-engineering-gain", type=float, default=1.0e-3)
    parser.add_argument("--phimin", type=float, default=0.0)
    parser.add_argument("--no-jit-booz", action="store_true")
    parser.add_argument("--exact-solve", action="store_true")
    parser.add_argument("--inner-max-iter", type=int, default=60)
    parser.add_argument("--trial-max-iter", type=int, default=60)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-8)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-8)
    parser.add_argument("--solver-device", default=None)
    parser.add_argument("--include", default="rc,zs")
    parser.add_argument("--fix", default="rc00")
    parser.add_argument("--project-input-boundary-to-max-mode", action="store_true")
    parser.add_argument("--save-accepted-inputs", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.directions = tuple(part.strip().lower() for part in str(args.directions).split(",") if part.strip())
    input_path = resolve_input_path(args.input)
    targets = SurveyTargets(
        smooth_qi_max=float(args.smooth_qi_max),
        legacy_qi_max=float(args.legacy_qi_max),
        mirror_ratio_max=float(args.mirror_threshold),
        max_elongation=float(args.max_elongation),
        abs_iota_min=float(args.abs_iota_min),
        target_aspect=float(args.target_aspect),
        aspect_tolerance=float(args.aspect_tolerance),
    )
    options = FilterSearchOptions(
        qi_relax=float(args.qi_relax),
        legacy_relax=float(args.legacy_relax),
        aspect_tolerance=float(args.aspect_tolerance),
        min_qi_gain=float(args.min_qi_gain),
        min_iota_gain=float(args.min_iota_gain),
        min_engineering_gain=float(args.min_engineering_gain),
    )
    plan = {
        "kind": "qi_filter_search",
        "input": str(input_path),
        "output_dir": str(args.output_dir),
        "execute": bool(args.execute),
        "targets": asdict(targets),
        "options": asdict(options),
        "max_iterations": int(args.max_iterations),
        "radius": float(args.radius),
        "directions": list(args.directions),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if not args.execute:
        print(f"Wrote QI filter-search plan: {output_dir / 'plan.json'}")
        return 0
    stage = build_stage(
        input_path=input_path,
        max_mode=args.max_mode,
        min_vmec_mode=args.min_vmec_mode,
        include=tuple(part.strip() for part in args.include.split(",") if part.strip()),
        fix=tuple(part.strip() for part in args.fix.split(",") if part.strip()),
        project_input_boundary_to_max_mode=args.project_input_boundary_to_max_mode,
        inner_max_iter=args.inner_max_iter,
        inner_ftol=args.inner_ftol,
        trial_max_iter=args.trial_max_iter,
        trial_ftol=args.trial_ftol,
        solver_device=args.solver_device,
    )
    report = run_filter_search(stage, args=args, targets=targets, options=options)
    print(f"Wrote QI filter-search summary: {output_dir / 'summary.json'}")
    print(
        "Final metrics: "
        f"QI={report['final_metrics'].get('qi_smooth_total')} "
        f"legacy={report['final_metrics'].get('qi_legacy_total')} "
        f"iota={report['final_metrics'].get('mean_iota')} "
        f"mirror={report['final_metrics'].get('qi_mirror_ratio_max')} "
        f"elongation={report['final_metrics'].get('qi_max_elongation')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

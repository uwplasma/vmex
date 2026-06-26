#!/usr/bin/env python
"""Bounded basin survey for far-seed QI optimization.

This tool probes larger boundary perturbations before launching a full local
QI optimization.  It is a diagnostic bridge between pure local least-squares
and expensive global optimizers: generate deterministic candidate jumps in the
active boundary-DOF space, optionally solve/evaluate them with the existing QI
diagnostics, then rank the candidates for local differentiable refinement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "examples" / "data"
DEFAULT_INPUT = DATA_DIR / "input.QI_stel_seed_3127"
DEFAULT_OUTPUT_DIR = Path("results/diagnostics/qi_basin_survey")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.optimization import boundary_param_names, create_x_scale
from vmec_jax.quasi_isodynamic.diagnostics import QIDiagnosticOptions, qi_diagnostics_from_state

from tools.diagnostics.qi_landscape_scan import (
    DEFAULT_SURFACES,
    _finite_float,
    build_stage,
    parse_surfaces,
    resolve_input_path,
)


enable_x64(True)

SUMMARY_FIELDS = (
    "rank",
    "label",
    "kind",
    "radius",
    "score",
    "qi_smooth_total",
    "qi_legacy_total",
    "qi_mirror_ratio_max",
    "qi_max_elongation",
    "mean_iota",
    "aspect",
    "input_path",
    "error",
)


@dataclass(frozen=True)
class SurveyTargets:
    """Acceptance/ranking targets for the far-seed QI basin survey."""

    smooth_qi_max: float = 2.0e-3
    legacy_qi_max: float = 2.0e-3
    mirror_ratio_max: float = 0.35
    max_elongation: float = 8.0
    abs_iota_min: float = 0.41
    target_aspect: float = 10.0
    aspect_tolerance: float = 2.0


@dataclass(frozen=True)
class BasinCandidate:
    """One boundary perturbation candidate in VMEC optimization coordinates."""

    label: str
    kind: str
    radius: float
    params: tuple[float, ...]
    dominant_dof: str | None = None

    def as_record(self, names: Sequence[str]) -> dict[str, Any]:
        params = list(float(v) for v in self.params)
        return {
            "label": self.label,
            "kind": self.kind,
            "radius": float(self.radius),
            "dominant_dof": self.dominant_dof,
            "params": params,
            "deltas": {name: value for name, value in zip(names, params) if abs(value) > 0.0},
        }


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float).reshape(-1)
    max_abs = float(np.max(np.abs(direction))) if direction.size else 0.0
    if max_abs == 0.0 or not np.isfinite(max_abs):
        return np.zeros_like(direction)
    return direction / max_abs


def _axis_order(names: Sequence[str], x_scale: np.ndarray) -> list[int]:
    order = np.argsort(-np.asarray(x_scale, dtype=float))
    return [int(idx) for idx in order if str(names[int(idx)]).lower() != "rc00"]


def generate_basin_candidates(
    *,
    names: Sequence[str],
    x_scale: Sequence[float],
    radii: Sequence[float],
    n_random: int,
    rng_seed: int,
    axis_count: int,
    directions: Sequence[str],
    include_zero: bool = True,
) -> list[BasinCandidate]:
    """Create deterministic large-step survey candidates.

    Candidate perturbations are scaled by ``x_scale`` so low-order modes receive
    larger physical jumps than high-order modes, matching the ESS local
    optimizer convention.
    """

    names = tuple(str(name) for name in names)
    scale = np.asarray(x_scale, dtype=float).reshape(-1)
    if scale.size != len(names):
        raise ValueError("x_scale length must match names length")
    if np.any(~np.isfinite(scale)):
        raise ValueError("x_scale must be finite")
    if any(float(radius) < 0.0 for radius in radii):
        raise ValueError("survey radii must be non-negative")
    direction_set = {str(item).strip().lower() for item in directions if str(item).strip()}
    candidates: list[BasinCandidate] = []
    seen: set[tuple[float, ...]] = set()

    def add(label: str, kind: str, radius: float, direction: np.ndarray, dominant_dof: str | None = None) -> None:
        normed = _normalize_direction(direction)
        params = tuple(float(v) for v in (float(radius) * scale * normed))
        key = tuple(round(value, 16) for value in params)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            BasinCandidate(
                label=label,
                kind=kind,
                radius=float(radius),
                params=params,
                dominant_dof=dominant_dof,
            )
        )

    if include_zero:
        add("zero", "baseline", 0.0, np.zeros(len(names), dtype=float))

    axis_indices = _axis_order(names, scale)[: max(0, int(axis_count))]
    if "axes" in direction_set:
        for radius in radii:
            for idx in axis_indices:
                direction = np.zeros(len(names), dtype=float)
                direction[idx] = 1.0
                add(f"axis+:{names[idx]}:{radius:g}", "axis_positive", float(radius), direction, names[idx])
                add(f"axis-:{names[idx]}:{radius:g}", "axis_negative", float(radius), -direction, names[idx])

    rng = np.random.default_rng(int(rng_seed))
    for radius in radii:
        if "rademacher" in direction_set:
            for sample in range(max(0, int(n_random))):
                direction = rng.choice(np.asarray([-1.0, 1.0]), size=len(names))
                add(f"rademacher:{sample:03d}:{radius:g}", "rademacher", float(radius), direction)
        if "gaussian" in direction_set:
            for sample in range(max(0, int(n_random))):
                direction = rng.normal(size=len(names))
                add(f"gaussian:{sample:03d}:{radius:g}", "gaussian", float(radius), direction)

    return candidates


def basin_score(metrics: dict[str, Any], targets: SurveyTargets = SurveyTargets()) -> float:
    """Return a finite ranking score; smaller is better."""

    values = {key: _finite_float(metrics.get(key)) for key in SUMMARY_FIELDS}
    smooth = values.get("qi_smooth_total")
    legacy = values.get("qi_legacy_total")
    mirror = values.get("qi_mirror_ratio_max")
    elongation = values.get("qi_max_elongation")
    iota = values.get("mean_iota")
    aspect = values.get("aspect")
    if any(value is None for value in (smooth, legacy, mirror, elongation, iota, aspect)):
        return 1.0e12
    smooth_score = max(0.0, float(smooth)) / max(float(targets.smooth_qi_max), 1.0e-16)
    legacy_score = max(0.0, float(legacy)) / max(float(targets.legacy_qi_max), 1.0e-16)
    mirror_score = max(0.0, float(mirror) - float(targets.mirror_ratio_max)) / max(
        float(targets.mirror_ratio_max), 1.0e-16
    )
    elongation_score = max(0.0, float(elongation) - float(targets.max_elongation)) / max(
        float(targets.max_elongation), 1.0e-16
    )
    iota_score = max(0.0, float(targets.abs_iota_min) - abs(float(iota))) / max(
        float(targets.abs_iota_min), 1.0e-16
    )
    aspect_score = max(0.0, abs(float(aspect) - float(targets.target_aspect)) - float(targets.aspect_tolerance)) / max(
        float(targets.target_aspect), 1.0e-16
    )
    return float(
        smooth_score
        + legacy_score
        + 2.0 * mirror_score
        + elongation_score
        + 4.0 * iota_score
        + 0.25 * aspect_score
    )


def rank_candidate_records(
    records: Sequence[dict[str, Any]],
    *,
    targets: SurveyTargets = SurveyTargets(),
) -> list[dict[str, Any]]:
    """Attach scores/ranks and return records sorted by survey score."""

    scored = []
    for record in records:
        out = dict(record)
        metrics = dict(out.get("metrics", {}))
        if out.get("error"):
            score = 1.0e12
        else:
            score = basin_score(metrics, targets=targets)
        out["score"] = float(score)
        scored.append(out)
    scored.sort(key=lambda item: (float(item["score"]), str(item.get("label", ""))))
    for rank, record in enumerate(scored, start=1):
        record["rank"] = rank
    return scored


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


def evaluate_candidate(stage, candidate: BasinCandidate, *, options: QIDiagnosticOptions, exact_solve: bool) -> dict[str, Any]:
    """Run one VMEC/QI diagnostic point and return JSON-friendly metrics."""

    optimizer = stage.optimizer
    params = np.asarray(candidate.params, dtype=float)
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
    metrics = {
        "qi_smooth_total": _finite_float(diagnostics.get("qi_smooth_total")),
        "qi_legacy_total": _finite_float(diagnostics.get("qi_legacy_total")),
        "qi_mirror_ratio_max": _finite_float(diagnostics.get("qi_mirror_ratio_max")),
        "qi_max_elongation": _finite_float(diagnostics.get("qi_max_elongation")),
        "mean_iota": _finite_float(diagnostics.get("mean_iota")),
        "aspect": _finite_float(diagnostics.get("aspect")),
    }
    return {"metrics": metrics, "diagnostics": diagnostics}


def _candidate_plan(
    *,
    input_path: Path,
    names: Sequence[str],
    candidates: Sequence[BasinCandidate],
    args: argparse.Namespace,
    targets: SurveyTargets,
) -> dict[str, Any]:
    return {
        "kind": "qi_basin_survey",
        "input": str(input_path),
        "max_mode": int(args.max_mode),
        "min_vmec_mode": int(args.min_vmec_mode),
        "alpha": float(args.alpha),
        "directions": list(args.directions),
        "radii": [float(radius) for radius in args.radius],
        "n_random": int(args.n_random),
        "axis_count": int(args.axis_count),
        "rng_seed": int(args.rng_seed),
        "execute": bool(args.execute),
        "exact_solve": bool(args.exact_solve),
        "targets": asdict(targets),
        "active_dofs": list(names),
        "candidates": [candidate.as_record(names) for candidate in candidates],
    }


def write_csv(records: Sequence[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            metrics = dict(record.get("metrics", {}))
            row = {field: record.get(field) for field in SUMMARY_FIELDS}
            row.update({key: metrics.get(key) for key in SUMMARY_FIELDS if key in metrics})
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="Run VMEC/QI solves. Default writes only a survey plan.")
    parser.add_argument("--max-mode", type=int, default=3)
    parser.add_argument("--min-vmec-mode", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=2.5, help="ESS spectral scaling alpha for candidate jumps.")
    parser.add_argument("--radius", type=float, action="append", default=None, help="Scaled jump radius; repeatable.")
    parser.add_argument("--n-random", type=int, default=8)
    parser.add_argument("--axis-count", type=int, default=4)
    parser.add_argument("--rng-seed", type=int, default=20260515)
    parser.add_argument(
        "--directions",
        default="axes,rademacher,gaussian",
        help="Comma-separated direction families: axes,rademacher,gaussian.",
    )
    parser.add_argument("--top-k", type=int, default=8)
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
    parser.add_argument("--phimin", type=float, default=0.0)
    parser.add_argument("--no-jit-booz", action="store_true")
    parser.add_argument("--exact-solve", action="store_true", help="Use exact accepted-point solves instead of trial solves.")
    parser.add_argument("--inner-max-iter", type=int, default=60)
    parser.add_argument("--trial-max-iter", type=int, default=60)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-8)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-8)
    parser.add_argument("--solver-device", default=None)
    parser.add_argument("--include", default="rc,zs")
    parser.add_argument("--fix", default="rc00")
    parser.add_argument("--project-input-boundary-to-max-mode", action="store_true")
    parser.add_argument("--save-candidate-inputs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    radii = tuple(args.radius) if args.radius is not None else (2.5e-2, 5.0e-2, 1.0e-1)
    args.radius = radii
    args.directions = tuple(part.strip().lower() for part in str(args.directions).split(",") if part.strip())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = resolve_input_path(args.input)
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
    names = boundary_param_names(stage.specs)
    x_scale = create_x_scale(stage.specs, alpha=float(args.alpha))
    candidates = generate_basin_candidates(
        names=names,
        x_scale=x_scale,
        radii=radii,
        n_random=int(args.n_random),
        rng_seed=int(args.rng_seed),
        axis_count=int(args.axis_count),
        directions=args.directions,
        include_zero=True,
    )
    targets = SurveyTargets(
        mirror_ratio_max=float(args.mirror_threshold),
        max_elongation=float(args.max_elongation),
        abs_iota_min=float(args.abs_iota_min),
        target_aspect=float(args.target_aspect),
        aspect_tolerance=float(args.aspect_tolerance),
    )
    plan = _candidate_plan(
        input_path=input_path,
        names=names,
        candidates=candidates,
        args=args,
        targets=targets,
    )
    (output_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if not args.execute:
        print(f"Wrote QI basin-survey plan: {output_dir / 'plan.json'}")
        return 0

    options = _make_qi_options(args)
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        record = candidate.as_record(names)
        try:
            evaluated = evaluate_candidate(stage, candidate, options=options, exact_solve=bool(args.exact_solve))
            record.update(evaluated)
            if bool(args.save_candidate_inputs):
                candidate_dir = output_dir / "candidates" / candidate.label.replace(":", "_")
                input_out = candidate_dir / "input.candidate"
                stage.optimizer.save_input(input_out, np.asarray(candidate.params, dtype=float))
                record["input_path"] = str(input_out)
        except Exception as exc:  # noqa: BLE001 - survey should continue to rank failures last.
            record["metrics"] = {}
            record["diagnostics"] = {}
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    ranked = rank_candidate_records(records, targets=targets)
    top = ranked[: max(1, int(args.top_k))]
    (output_dir / "candidates.json").write_text(json.dumps(ranked, indent=2, sort_keys=True) + "\n")
    (output_dir / "top_candidates.json").write_text(json.dumps(top, indent=2, sort_keys=True) + "\n")
    write_csv(ranked, output_dir / "candidates.csv")
    print(f"Wrote QI basin-survey candidates: {output_dir / 'candidates.json'}")
    print(f"Wrote QI basin-survey top candidates: {output_dir / 'top_candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

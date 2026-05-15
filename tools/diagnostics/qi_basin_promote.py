#!/usr/bin/env python
"""Promote QI basin-survey candidates into bounded local refinements.

``qi_basin_survey.py`` answers "which large perturbations look promising?"
This helper answers the next question: "if I start local differentiable QI
optimization from the top candidates, which basin survives the independent QI,
iota, mirror, elongation, and aspect gates?"

The script is intentionally a developer diagnostic.  It keeps strict per-case
budgets, writes a reviewable plan by default, and only launches VMEC when
``--execute`` is passed.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SURVEY = Path("results/diagnostics/qi_basin_survey/top_candidates.json")
DEFAULT_OUTPUT_DIR = Path("results/diagnostics/qi_basin_promotion")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.qi_constraint_policy_scan import (  # noqa: E402
    MAX_ELONGATION,
    QI_GATE_LEGACY_MAX,
    QI_GATE_SMOOTH_MAX,
    TARGET_ABS_IOTA_MIN,
    TARGET_ASPECT,
    ScanResolution,
    StagePolicy,
    _diagnose,
    _float_or_none,
    _make_problem,
)


@dataclass(frozen=True)
class PromotionPolicy:
    """One bounded local refinement policy applied to each surveyed seed."""

    name: str
    description: str
    stages: tuple[StagePolicy, ...]


SUMMARY_FIELDS = (
    "rank",
    "candidate_label",
    "policy",
    "selected",
    "selection_reason",
    "smooth_qi",
    "legacy_qi",
    "mirror",
    "elongation",
    "iota",
    "aspect",
    "wall_time_s",
    "output_dir",
)


def default_promotion_policies(*, max_nfev: int = 4) -> tuple[PromotionPolicy, ...]:
    """Return bounded policies for basin promotion.

    The first two policies prioritize finding a QI+iota basin.  The third
    policy adds a guarded augmented-Lagrangian cleanup stage for mirror and
    elongation after the candidate has been locally refined.
    """

    nfev = max(1, int(max_nfev))
    return (
        PromotionPolicy(
            "guarded_iota_ramp",
            "QI-first refinement, then guarded scalar-trust iota ramp before engineering cleanup.",
            (
                StagePolicy(
                    "qi_preserve",
                    method="scalar_trust",
                    max_nfev=max(2, nfev),
                    stage_modes=(3,),
                    aspect_weight=0.05,
                    iota_weight=0.0,
                    qi_weight=200.0,
                    qi_ceiling_weight=0.0,
                    branch_width_weight=0.5,
                    continue_if_qi_aspect_pass=True,
                ),
                StagePolicy(
                    "iota_soft_qi_guard",
                    method="scalar_trust",
                    max_nfev=max(2, nfev),
                    stage_modes=(3, 3),
                    aspect_weight=0.05,
                    iota_weight=50.0**2,
                    qi_weight=200.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=3.0e-3,
                    qi_ceiling_smooth_penalty=1.0e-3,
                    branch_width_weight=0.5,
                    continue_if_qi_aspect_pass=True,
                ),
                StagePolicy(
                    "mirror_soft_guard",
                    method="scalar_trust",
                    max_nfev=max(2, nfev),
                    stage_modes=(3,),
                    aspect_weight=0.05,
                    iota_weight=50.0**2,
                    qi_weight=200.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.35,
                    mirror_weight=2.0,
                    elongation_weight=1.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=5.0e-3,
                    qi_ceiling_smooth_penalty=1.0e-3,
                    branch_width_weight=0.5,
                ),
            ),
        ),
        PromotionPolicy(
            "direct_matrix_free",
            "Direct mode-3 matrix-free local QI+iota refinement.",
            (
                StagePolicy(
                    "direct_m3_qi_iota",
                    method="scipy_matrix_free",
                    max_nfev=nfev,
                    stage_modes=(3,),
                    branch_width_weight=0.5,
                ),
            ),
        ),
        PromotionPolicy(
            "repeat_continuation",
            "Repeated mode continuation to avoid under-refined local minima.",
            (
                StagePolicy(
                    "repeat_112233",
                    method="scipy_matrix_free",
                    max_nfev=nfev,
                    stage_modes=(1, 1, 2, 2, 3, 3),
                    branch_width_weight=0.5,
                ),
            ),
        ),
        PromotionPolicy(
            "qi_then_al_cleanup",
            "QI+iota refinement followed by augmented-Lagrangian mirror/elongation cleanup.",
            (
                StagePolicy(
                    "qi_iota_refine",
                    method="scipy_matrix_free",
                    max_nfev=nfev,
                    stage_modes=(1, 1, 2, 2, 3, 3),
                    branch_width_weight=0.5,
                ),
                StagePolicy(
                    "al_mirror_elongation",
                    method="scalar_trust",
                    max_nfev=max(2, nfev),
                    stage_modes=(3, 3),
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.35,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=10.0,
                    al_mirror_penalty=200.0,
                    al_elongation_multiplier=5.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=500.0,
                    qi_ceiling_max=5.0e-3,
                    branch_width_weight=0.5,
                ),
            ),
        ),
        PromotionPolicy(
            "soft_wall_cleanup",
            "Moderate mirror/elongation soft-wall cleanup with an active QI ceiling.",
            (
                StagePolicy(
                    "soft_wall",
                    method="scipy_matrix_free",
                    max_nfev=nfev,
                    stage_modes=(1, 1, 2, 2, 3, 3),
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.35,
                    mirror_weight=15.0,
                    elongation_weight=5.0,
                    qi_ceiling_weight=500.0,
                    qi_ceiling_max=5.0e-3,
                    branch_width_weight=0.5,
                ),
            ),
        ),
    )


def load_candidate_records(path: Path, *, top_n: int) -> list[dict[str, Any]]:
    """Load candidate records from ``qi_basin_survey.py`` output."""

    records = json.loads(Path(path).read_text())
    if not isinstance(records, list):
        raise ValueError("candidate JSON must contain a list")
    records = sorted(records, key=lambda item: (float(item.get("rank", 1.0e9)), float(item.get("score", 1.0e12))))
    out = []
    for record in records:
        input_path = record.get("input_path")
        if not input_path:
            continue
        input_file = Path(str(input_path)).expanduser()
        if not input_file.exists():
            continue
        out.append(record)
        if len(out) >= max(1, int(top_n)):
            break
    return out


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _stage_record_from_diagnostics(stage: StagePolicy, diagnostics: dict[str, Any], *, output_dir: Path, wall_time_s: float) -> dict[str, Any]:
    return {
        "stage_name": stage.name,
        "method": stage.method,
        "stage_modes": list(stage.stage_modes),
        "max_nfev": int(stage.max_nfev),
        "smooth_qi": _float_or_none(diagnostics.get("qi_smooth_total")),
        "legacy_qi": _float_or_none(diagnostics.get("qi_legacy_total")),
        "mirror": _float_or_none(diagnostics.get("qi_mirror_ratio_max")),
        "elongation": _float_or_none(diagnostics.get("qi_max_elongation")),
        "iota": _float_or_none(diagnostics.get("mean_iota")),
        "aspect": _float_or_none(diagnostics.get("aspect")),
        "qi_seed_gate_passed": bool(diagnostics.get("qi_seed_gate_passed")),
        "engineering_gate_passed": bool(diagnostics.get("qi_engineering_gate_passed")),
        "gate_failures": diagnostics.get("qi_gate_failures", []),
        "failure_reasons": diagnostics.get("qi_failure_reasons", []),
        "wall_time_s": float(wall_time_s),
        "output_dir": str(output_dir),
    }


def _should_continue_after_stage(stage: StagePolicy, diagnostics: dict[str, Any]) -> bool:
    if bool(diagnostics.get("qi_seed_gate_passed")):
        return True
    if not bool(getattr(stage, "continue_if_qi_aspect_pass", False)):
        return False
    failures = set(diagnostics.get("qi_gate_failures", []))
    # QI-preserve and iota-ramp stages are allowed to leave iota and
    # engineering gates for later stages, but they must not lose QI/aspect.
    return bool(failures) and failures <= {"iota", "mirror", "elongation"}


def run_promotion_policy(
    *,
    vj: Any,
    candidate: dict[str, Any],
    policy: PromotionPolicy,
    out_root: Path,
    resolution: ScanResolution,
    inner_max_iter: int,
    trial_max_iter: int,
    solver_device: str | None,
    save_wouts: bool,
) -> dict[str, Any]:
    """Run one promotion policy for one candidate input."""

    label = str(candidate.get("label") or f"rank_{candidate.get('rank', 'unknown')}")
    input_file = Path(str(candidate["input_path"])).expanduser()
    policy_dir = out_root / _safe_name(label) / policy.name
    active_input = input_file
    stage_records: list[dict[str, Any]] = []
    final_result = None
    final_diagnostics: dict[str, Any] | None = None
    t0 = time.perf_counter()

    for stage_index, stage in enumerate(policy.stages, start=1):
        stage_dir = policy_dir / f"{stage_index:02d}_{_safe_name(stage.name)}"
        problem, qi_options = _make_problem(vj, resolution, stage)
        vmec = vj.FixedBoundaryVMEC.from_input(
            active_input,
            max_mode=3,
            min_vmec_mode=6,
            output_dir=stage_dir,
            project_input_boundary_to_max_mode=True,
        )
        stage_t0 = time.perf_counter()
        result = vj.least_squares_solve(
            vmec,
            problem,
            stage_modes=list(stage.stage_modes),
            max_nfev=int(stage.max_nfev),
            continuation_nfev=0,
            method=str(stage.method),
            ftol=1.0e-4,
            gtol=1.0e-4,
            xtol=1.0e-8,
            use_ess=True,
            ess_alpha=1.2,
            label=f"{label}:{policy.name}:{stage.name}",
            inner_max_iter=int(inner_max_iter),
            inner_ftol=1.0e-8,
            trial_max_iter=int(trial_max_iter),
            trial_ftol=1.0e-8,
            solver_device=solver_device,
            scipy_tr_solver="lsmr",
            scipy_lsmr_maxiter=5,
            save_stage_inputs=True,
            save_stage_wouts=False,
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        result.final_optimizer.save_input(stage_dir / "input.final", result.final_params)
        result.final_optimizer.save_history(stage_dir / "history.json", result.final_result)
        if save_wouts:
            result.final_optimizer.save_wout(stage_dir / "wout_final.nc", result.final_params, state=result.final_state)
        mirror_gate = (
            float(stage.promotion_mirror_threshold)
            if stage.promotion_mirror_threshold is not None
            else float(stage.mirror_threshold)
        )
        diagnostics = _diagnose(vj, result, resolution, qi_options, mirror_threshold=mirror_gate)
        (stage_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
        stage_records.append(
            _stage_record_from_diagnostics(
                stage,
                diagnostics,
                output_dir=stage_dir,
                wall_time_s=time.perf_counter() - stage_t0,
            )
        )
        final_result = result
        final_diagnostics = diagnostics
        active_input = stage_dir / "input.final"

        # Do not spend cleanup budget if the candidate did not first find a
        # QI+iota basin.  Engineering gates are evaluated on the final record.
        if not _should_continue_after_stage(stage, diagnostics):
            break

    wall = time.perf_counter() - t0
    if final_result is not None and final_diagnostics is not None:
        policy_dir.mkdir(parents=True, exist_ok=True)
        final_result.final_optimizer.save_input(policy_dir / "input.final", final_result.final_params)
        if save_wouts:
            final_result.final_optimizer.save_wout(
                policy_dir / "wout_final.nc",
                final_result.final_params,
                state=final_result.final_state,
            )
        (policy_dir / "diagnostics.json").write_text(json.dumps(final_diagnostics, indent=2, sort_keys=True) + "\n")
    final_stage = stage_records[-1] if stage_records else {}
    selected = bool(final_stage.get("qi_seed_gate_passed")) and bool(final_stage.get("engineering_gate_passed"))
    if selected:
        reason = "QI+iota and engineering gates passed"
    else:
        reason = "; ".join(final_stage.get("failure_reasons", [])) if final_stage else "no stage completed"
    return {
        "rank": candidate.get("rank"),
        "candidate_label": label,
        "candidate_score": candidate.get("score"),
        "candidate_input": str(input_file),
        "policy": policy.name,
        "description": policy.description,
        "selected": selected,
        "selection_reason": reason,
        "smooth_qi": final_stage.get("smooth_qi"),
        "legacy_qi": final_stage.get("legacy_qi"),
        "mirror": final_stage.get("mirror"),
        "elongation": final_stage.get("elongation"),
        "iota": final_stage.get("iota"),
        "aspect": final_stage.get("aspect"),
        "wall_time_s": wall,
        "output_dir": str(policy_dir),
        "stages": stage_records,
    }


def _policy_subset(policies: tuple[PromotionPolicy, ...], names: Sequence[str] | None) -> tuple[PromotionPolicy, ...]:
    if not names:
        return policies
    wanted = set(names)
    available = {policy.name for policy in policies}
    missing = wanted - available
    if missing:
        raise ValueError(f"Unknown promotion policies: {', '.join(sorted(missing))}")
    return tuple(policy for policy in policies if policy.name in wanted)


def write_summary(records: Sequence[dict[str, Any]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        records,
        key=lambda item: (
            not bool(item.get("selected")),
            float("inf") if item.get("smooth_qi") is None else float(item.get("smooth_qi")),
            float("inf") if item.get("mirror") is None else float(item.get("mirror")),
        ),
    )
    (out_root / "promotion_summary.json").write_text(json.dumps(ranked, indent=2, sort_keys=True) + "\n")
    with (out_root / "promotion_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in ranked:
            writer.writerow({field: record.get(field) for field in SUMMARY_FIELDS})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_SURVEY)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--policy", action="append", help="Promotion policy to run; repeat for multiple.")
    parser.add_argument("--max-nfev", type=int, default=4)
    parser.add_argument("--inner-max-iter", type=int, default=60)
    parser.add_argument("--trial-max-iter", type=int, default=40)
    parser.add_argument("--solver-device", default=None)
    parser.add_argument("--save-wouts", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use a smaller QI diagnostic grid.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolution = ScanResolution()
    if args.quick:
        resolution = ScanResolution(surfaces=(0.5, 1.0), mboz=6, nboz=6, nphi=31, nalpha=7, n_bounce=9)
    policies = _policy_subset(default_promotion_policies(max_nfev=int(args.max_nfev)), args.policy)
    candidates = load_candidate_records(args.candidates, top_n=int(args.top_n))
    if not candidates:
        raise ValueError(
            "No promotable candidate inputs found. Run qi_basin_survey.py with --execute --save-candidate-inputs first."
        )
    args.out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "candidates_file": str(args.candidates),
        "out_root": str(args.out_root),
        "execute": bool(args.execute),
        "top_n": int(args.top_n),
        "resolution": asdict(resolution),
        "targets": {
            "target_aspect": TARGET_ASPECT,
            "target_abs_iota_min": TARGET_ABS_IOTA_MIN,
            "smooth_qi_max": QI_GATE_SMOOTH_MAX,
            "legacy_qi_max": QI_GATE_LEGACY_MAX,
            "max_elongation": MAX_ELONGATION,
        },
        "candidates": [
            {
                "rank": candidate.get("rank"),
                "label": candidate.get("label"),
                "score": candidate.get("score"),
                "input_path": candidate.get("input_path"),
                "metrics": candidate.get("metrics", {}),
            }
            for candidate in candidates
        ],
        "policies": [
            {
                "name": policy.name,
                "description": policy.description,
                "stages": [asdict(stage) for stage in policy.stages],
            }
            for policy in policies
        ],
    }
    (args.out_root / "promotion_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if not args.execute:
        print(f"Wrote QI basin-promotion plan: {args.out_root / 'promotion_plan.json'}")
        return 0

    import vmec_jax as vj
    from vmec_jax._compat import enable_x64

    enable_x64(True)
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        for policy in policies:
            try:
                print(f"\n[qi-basin-promote] {candidate.get('label')} :: {policy.name}", flush=True)
                records.append(
                    run_promotion_policy(
                        vj=vj,
                        candidate=candidate,
                        policy=policy,
                        out_root=args.out_root,
                        resolution=resolution,
                        inner_max_iter=int(args.inner_max_iter),
                        trial_max_iter=int(args.trial_max_iter),
                        solver_device=args.solver_device,
                        save_wouts=bool(args.save_wouts),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - continue matrix diagnostics.
                records.append(
                    {
                        "rank": candidate.get("rank"),
                        "candidate_label": candidate.get("label"),
                        "candidate_score": candidate.get("score"),
                        "candidate_input": candidate.get("input_path"),
                        "policy": policy.name,
                        "description": policy.description,
                        "selected": False,
                        "selection_reason": f"{type(exc).__name__}: {exc}",
                        "smooth_qi": None,
                        "legacy_qi": None,
                        "mirror": None,
                        "elongation": None,
                        "iota": None,
                        "aspect": None,
                        "wall_time_s": None,
                        "output_dir": str(args.out_root / _safe_name(str(candidate.get("label"))) / policy.name),
                        "stages": [],
                    }
                )
                print(f"[qi-basin-promote] failed: {type(exc).__name__}: {exc}", flush=True)
            write_summary(records, args.out_root)
    print(f"\nWrote QI basin-promotion summary: {args.out_root / 'promotion_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

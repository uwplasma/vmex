#!/usr/bin/env python
"""Bounded QI constraint-policy scan for ``input.QI_stel_seed_3127``.

This helper intentionally reuses the same public workflow as
``examples/optimization/QI_optimization.py`` while keeping the scan small:
low Boozer/QI grids, explicit ``max_nfev`` caps, and a short policy matrix.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "examples" / "data"
DEFAULT_INPUT = DATA_DIR / "input.QI_stel_seed_3127"
DEFAULT_OUT_ROOT = Path("/tmp/vmec_jax_qi_constraint_policy_scan")
TARGET_ASPECT = 10.0
TARGET_ABS_IOTA_MIN = 0.41
MAX_ELONGATION = 8.0
QI_GATE_SMOOTH_MAX = 2.0e-3
QI_GATE_LEGACY_MAX = 1.0e-3


@dataclass(frozen=True)
class ScanResolution:
    surfaces: tuple[float, ...] = (0.35, 0.7, 1.0)
    mboz: int = 8
    nboz: int = 8
    nphi: int = 41
    nalpha: int = 9
    n_bounce: int = 11
    mirror_ntheta: int = 32
    mirror_nphi: int = 32
    elongation_ntheta: int = 24
    elongation_nphi: int = 8


@dataclass(frozen=True)
class StagePolicy:
    name: str
    method: str = "scipy"
    max_nfev: int = 2
    stage_modes: tuple[int, ...] = (3,)
    aspect_weight: float = 0.25
    iota_weight: float = 200.0**2
    qi_weight: float = 10.0
    mirror_threshold: float = 0.21
    promotion_mirror_threshold: float | None = None
    mirror_weight: float = 0.0
    elongation_weight: float = 0.0
    use_augmented_lagrangian: bool = False
    al_mirror_multiplier: float = 0.0
    al_mirror_penalty: float = 1.0
    al_elongation_multiplier: float = 0.0
    al_elongation_penalty: float = 1.0
    qi_ceiling_weight: float = 0.0
    qi_ceiling_max: float = 2.0e-3
    qi_ceiling_smooth_penalty: float = 2.0e-3
    branch_width_weight: float = 0.5
    weighted_shuffle_profile_weight: float = 0.0
    continue_if_qi_aspect_pass: bool = False
    scalar_step_bound: float | None = None
    lbfgs_step_bound: float | None = None


@dataclass(frozen=True)
class Policy:
    name: str
    description: str
    stages: tuple[StagePolicy, ...]


def default_policies(*, max_nfev: int = 2) -> tuple[Policy, ...]:
    """Return the bounded policy matrix requested for the seed-robustness probe."""

    return (
        Policy(
            "scipy_qi_iota",
            "Baseline SciPy trust-region QI+iota/aspect objective; no engineering cleanup.",
            (StagePolicy("qi_iota", method="scipy", max_nfev=max_nfev),),
        ),
        Policy(
            "scalar_trust_qi_iota",
            "Scalar-adjoint safeguarded trust probe with the same QI+iota/aspect objective.",
            (StagePolicy("qi_iota", method="scalar_trust", max_nfev=max_nfev),),
        ),
        Policy(
            "matrix_free_qi_iota",
            "SciPy matrix-free trust-region probe with the same QI+iota/aspect objective.",
            (StagePolicy("qi_iota", method="scipy_matrix_free", max_nfev=max_nfev),),
        ),
        Policy(
            "large_mirror_weights",
            "Direct hard mirror cleanup pressure with large mirror and elongation weights.",
            (
                StagePolicy(
                    "large_mirror",
                    method="scipy",
                    max_nfev=max_nfev,
                    mirror_threshold=0.21,
                    mirror_weight=50.0,
                    elongation_weight=10.0,
                ),
            ),
        ),
        Policy(
            "staged_mirror_relax_tight",
            "Two short cleanup stages: relaxed mirror threshold, then tighter mirror threshold.",
            (
                StagePolicy(
                    "mirror_relaxed",
                    method="scipy",
                    max_nfev=max(1, max_nfev),
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.35,
                    mirror_weight=10.0,
                    elongation_weight=5.0,
                    qi_ceiling_weight=100.0,
                    qi_ceiling_max=2.0e-2,
                ),
                StagePolicy(
                    "mirror_tight",
                    method="scipy",
                    max_nfev=max(1, max_nfev),
                    mirror_threshold=0.21,
                    promotion_mirror_threshold=0.30,
                    mirror_weight=30.0,
                    elongation_weight=10.0,
                    qi_ceiling_weight=250.0,
                    qi_ceiling_max=1.0e-2,
                ),
            ),
        ),
        Policy(
            "softplus_barriers",
            "Smooth softplus QI ceiling plus moderate mirror/elongation penalties.",
            (
                StagePolicy(
                    "softplus_barrier",
                    method="scipy",
                    max_nfev=max_nfev,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_weight=15.0,
                    elongation_weight=5.0,
                    qi_ceiling_weight=250.0,
                    qi_ceiling_max=2.0e-2,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "balanced_qi_mirror",
            "High-QI-weight mirror cleanup: preserve the QI basin while applying stronger mirror pressure.",
            (
                StagePolicy(
                    "balanced_qi_mirror",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=250.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_weight=20.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "balanced_qi_mirror_tight",
            "Higher mirror pressure with the same QI ceiling, used to check whether mirror can be lowered before QI fails.",
            (
                StagePolicy(
                    "balanced_qi_mirror_tight",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=250.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "balanced_qi500_mirror_tight",
            "Same tight mirror policy with doubled QI weight to test whether the low-mirror basin can pass QI gates.",
            (
                StagePolicy(
                    "balanced_qi500_mirror_tight",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=500.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "balanced_qi1000_mirror_tight",
            "Same tight mirror policy with quadrupled QI weight for the high-QI/low-mirror tradeoff check.",
            (
                StagePolicy(
                    "balanced_qi1000_mirror_tight",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=1000.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "augmented_lagrangian_mirror",
            "Projected augmented-Lagrangian mirror/elongation constraints with a QI ceiling guard.",
            (
                StagePolicy(
                    "al_mirror_elongation",
                    method="scalar_trust",
                    max_nfev=max(2, max_nfev),
                    mirror_threshold=0.75,
                    promotion_mirror_threshold=0.75,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=20.0,
                    al_mirror_penalty=400.0,
                    al_elongation_multiplier=5.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=1000.0,
                    qi_ceiling_max=2.0e-3,
                ),
            ),
        ),
        Policy(
            "mode_continuation_repeat",
            "Cheap mode-continuation repeat: mode 2 warmup followed by two mode 3 passes.",
            (
                StagePolicy(
                    "mode_repeat",
                    method="scipy",
                    max_nfev=max(1, max_nfev),
                    stage_modes=(2, 3, 3),
                ),
            ),
        ),
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _build_qi_options(vj: Any, resolution: ScanResolution, stage: StagePolicy):
    return vj.QuasiIsodynamicOptions(
        surfaces=np.asarray(resolution.surfaces, dtype=float),
        mboz=int(resolution.mboz),
        nboz=int(resolution.nboz),
        nphi=int(resolution.nphi),
        nalpha=int(resolution.nalpha),
        n_bounce=int(resolution.n_bounce),
        include_bounce_endpoints=True,
        softness=2.0e-2,
        width_weight=1.0,
        branch_width_weight=float(stage.branch_width_weight),
        branch_width_softness=2.0e-2,
        profile_weight=0.1,
        shuffle_profile_weight=1.0,
        shuffle_profile_softness=2.0e-2,
        weighted_shuffle_profile_weight=float(stage.weighted_shuffle_profile_weight),
        weighted_shuffle_profile_softness=2.0e-2,
        phimin=0.0,
        jit_booz=True,
    )


def _make_problem(vj: Any, resolution: ScanResolution, stage: StagePolicy):
    qi_options = _build_qi_options(vj, resolution, stage)
    tuples = [
        (vj.AspectRatio().J, TARGET_ASPECT, float(stage.aspect_weight)),
        (vj.AbsMeanIotaFloor(TARGET_ABS_IOTA_MIN).J, 0.0, float(stage.iota_weight)),
        (vj.QuasiIsodynamicResidual(qi_options).J, 0.0, float(stage.qi_weight)),
    ]
    if stage.qi_ceiling_weight > 0.0:
        tuples.append(
            (
                vj.QuasiIsodynamicResidualCeiling(
                    maximum=float(stage.qi_ceiling_max),
                    smooth_penalty=float(stage.qi_ceiling_smooth_penalty),
                    qi_options=qi_options,
                ).J,
                0.0,
                float(stage.qi_ceiling_weight),
            )
        )
    if stage.mirror_weight > 0.0:
        mirror = vj.MirrorRatio(
            threshold=float(stage.mirror_threshold),
            ntheta=int(resolution.mirror_ntheta),
            nphi=int(resolution.mirror_nphi),
            surface_index=None,
            phimin=0.0,
            smooth_extrema=2.0e-2,
            smooth_penalty=2.0e-2,
            qi_options=qi_options,
        )
        if stage.use_augmented_lagrangian:
            mirror = vj.AugmentedLagrangianConstraint(
                mirror,
                multiplier=float(stage.al_mirror_multiplier),
                penalty=float(stage.al_mirror_penalty),
                softness=2.0e-2,
                name="al_mirror_ratio",
            )
        tuples.append(
            (
                mirror.J,
                0.0,
                float(stage.mirror_weight),
            )
        )
    if stage.elongation_weight > 0.0:
        elongation = vj.MaxElongation(
            threshold=MAX_ELONGATION,
            ntheta=int(resolution.elongation_ntheta),
            nphi=int(resolution.elongation_nphi),
            smooth_extrema=2.0e-2,
            smooth_penalty=2.0e-2,
            qi_options=qi_options,
        )
        if stage.use_augmented_lagrangian:
            elongation = vj.AugmentedLagrangianConstraint(
                elongation,
                multiplier=float(stage.al_elongation_multiplier),
                penalty=float(stage.al_elongation_penalty),
                softness=2.0e-2,
                name="al_max_elongation",
            )
        tuples.append(
            (
                elongation.J,
                0.0,
                float(stage.elongation_weight),
            )
        )
    return vj.LeastSquaresProblem.from_tuples(tuples), qi_options


def _diagnose(vj: Any, result: Any, resolution: ScanResolution, qi_options: Any, *, mirror_threshold: float) -> dict[str, Any]:
    from vmec_jax.qi_diagnostics import QISeedSuitabilityTargets, annotate_qi_seed_suitability

    options = vj.QIDiagnosticOptions(
        surfaces=np.asarray(resolution.surfaces, dtype=float),
        mboz=int(resolution.mboz),
        nboz=int(resolution.nboz),
        nphi=int(resolution.nphi),
        nalpha=int(resolution.nalpha),
        n_bounce=int(resolution.n_bounce),
        include_bounce_endpoints=True,
        softness=qi_options.softness,
        width_weight=qi_options.width_weight,
        branch_width_weight=qi_options.branch_width_weight,
        branch_width_softness=qi_options.branch_width_softness,
        profile_weight=qi_options.profile_weight,
        shuffle_profile_weight=qi_options.shuffle_profile_weight,
        shuffle_profile_softness=qi_options.shuffle_profile_softness,
        weighted_shuffle_profile_weight=qi_options.weighted_shuffle_profile_weight,
        weighted_shuffle_profile_softness=qi_options.weighted_shuffle_profile_softness,
        phimin=0.0,
        mirror_threshold=float(mirror_threshold),
        mirror_ntheta=int(resolution.mirror_ntheta),
        mirror_nphi=int(resolution.mirror_nphi),
        elongation_threshold=MAX_ELONGATION,
    )
    diagnostics = vj.qi_diagnostics_from_state(
        state=result.final_state,
        static=result.final_optimizer.static,
        indata=result.final_optimizer.indata,
        signgs=result.final_optimizer.signgs,
        surfaces=np.asarray(resolution.surfaces, dtype=float),
        options=options,
    )
    return annotate_qi_seed_suitability(
        diagnostics,
        targets=QISeedSuitabilityTargets(
            smooth_qi_max=QI_GATE_SMOOTH_MAX,
            legacy_qi_max=QI_GATE_LEGACY_MAX,
            target_aspect=TARGET_ASPECT,
            abs_iota_min=TARGET_ABS_IOTA_MIN,
            mirror_ratio_max=float(mirror_threshold),
            max_elongation=MAX_ELONGATION,
        ),
    )


def run_policy(
    policy: Policy,
    *,
    input_file: Path,
    out_root: Path,
    resolution: ScanResolution,
    inner_max_iter: int,
    trial_max_iter: int,
) -> dict[str, Any]:
    _ensure_repo_on_path()
    import vmec_jax as vj
    from vmec_jax._compat import enable_x64

    enable_x64(True)
    active_input = input_file
    policy_dir = out_root / policy.name
    stage_records: list[dict[str, Any]] = []
    selected_output = False
    selected_reason = "no stage completed"
    final_result = None
    final_diagnostics: dict[str, Any] | None = None
    final_qi_options = None
    start = time.perf_counter()

    for index, stage in enumerate(policy.stages, start=1):
        stage_dir = policy_dir / f"{index:02d}_{stage.name}"
        problem, qi_options = _make_problem(vj, resolution, stage)
        vmec = vj.FixedBoundaryVMEC.from_input(
            active_input,
            max_mode=3,
            min_vmec_mode=6,
            output_dir=stage_dir,
            project_input_boundary_to_max_mode=True,
        )
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
            label=f"{policy.name}:{stage.name}",
            inner_max_iter=int(inner_max_iter),
            inner_ftol=1.0e-8,
            trial_max_iter=int(trial_max_iter),
            trial_ftol=1.0e-8,
            solver_device="cpu",
            scipy_tr_solver="lsmr",
            scipy_lsmr_maxiter=5,
            save_stage_inputs=True,
            save_stage_wouts=False,
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        result.final_optimizer.save_input(stage_dir / "input.final", result.final_params)
        result.final_optimizer.save_wout(stage_dir / "wout_final.nc", result.final_params, state=result.final_state)
        result.final_optimizer.save_history(stage_dir / "history.json", result.final_result)
        mirror_gate = (
            float(stage.promotion_mirror_threshold)
            if stage.promotion_mirror_threshold is not None
            else float(stage.mirror_threshold)
        )
        diagnostics = _diagnose(vj, result, resolution, qi_options, mirror_threshold=mirror_gate)
        (stage_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
        stage_selected = bool(diagnostics.get("qi_seed_gate_passed"))
        stage_record = {
            "stage": index,
            "stage_name": stage.name,
            "method": stage.method,
            "stage_modes": list(stage.stage_modes),
            "max_nfev": stage.max_nfev,
            "output_dir": str(stage_dir),
            "selected": stage_selected,
            "smooth_qi": _float_or_none(diagnostics.get("qi_smooth_total")),
            "legacy_qi": _float_or_none(diagnostics.get("qi_legacy_total")),
            "mirror": _float_or_none(diagnostics.get("qi_mirror_ratio_max")),
            "elongation": _float_or_none(diagnostics.get("qi_max_elongation")),
            "iota": _float_or_none(diagnostics.get("mean_iota")),
            "aspect": _float_or_none(diagnostics.get("aspect")),
            "wall_time_s": _float_or_none(result.timing_summary.get("total_wall_time_s")),
            "gate_failures": diagnostics.get("qi_gate_failures", []),
        }
        stage_records.append(stage_record)
        final_result = result
        final_diagnostics = diagnostics
        final_qi_options = qi_options
        if stage_selected:
            selected_output = True
            selected_reason = "QI+iota seed gate passed"
            active_input = stage_dir / "input.final"
        else:
            selected_output = False
            selected_reason = "; ".join(diagnostics.get("qi_failure_reasons", [])) or "QI+iota seed gate failed"
            break

    if final_result is not None and final_diagnostics is not None and final_qi_options is not None:
        policy_dir.mkdir(parents=True, exist_ok=True)
        final_result.final_optimizer.save_input(policy_dir / "input.final", final_result.final_params)
        final_result.final_optimizer.save_wout(policy_dir / "wout_final.nc", final_result.final_params, state=final_result.final_state)
        (policy_dir / "diagnostics.json").write_text(json.dumps(final_diagnostics, indent=2, sort_keys=True) + "\n")

    wall = time.perf_counter() - start
    last = stage_records[-1] if stage_records else {}
    record = {
        "policy": policy.name,
        "description": policy.description,
        "selected": bool(selected_output),
        "selection": "selected" if selected_output else "rejected",
        "selection_reason": selected_reason,
        "smooth_qi": last.get("smooth_qi"),
        "legacy_qi": last.get("legacy_qi"),
        "mirror": last.get("mirror"),
        "elongation": last.get("elongation"),
        "iota": last.get("iota"),
        "aspect": last.get("aspect"),
        "wall_time_s": wall,
        "output_dir": str(policy_dir),
        "stages": stage_records,
    }
    (policy_dir / "policy_result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def write_summary(records: list[dict[str, Any]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    fields = [
        "policy",
        "smooth_qi",
        "legacy_qi",
        "mirror",
        "elongation",
        "iota",
        "aspect",
        "wall_time_s",
        "selection",
        "selection_reason",
        "output_dir",
    ]
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def _policy_subset(all_policies: tuple[Policy, ...], names: list[str] | None) -> tuple[Policy, ...]:
    if not names:
        return all_policies
    wanted = set(names)
    missing = wanted - {policy.name for policy in all_policies}
    if missing:
        raise ValueError(f"Unknown policies: {', '.join(sorted(missing))}")
    return tuple(policy for policy in all_policies if policy.name in wanted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--execute", action="store_true", help="Run optimizations. Default only writes the plan.")
    parser.add_argument("--policy", action="append", help="Run one named policy; repeat for multiple.")
    parser.add_argument("--max-nfev", type=int, default=2)
    parser.add_argument("--inner-max-iter", type=int, default=40)
    parser.add_argument("--trial-max-iter", type=int, default=25)
    parser.add_argument("--quick", action="store_true", help="Use an even smaller diagnostic grid.")
    args = parser.parse_args(argv)

    resolution = ScanResolution()
    if args.quick:
        resolution = ScanResolution(surfaces=(0.5, 1.0), mboz=6, nboz=6, nphi=31, nalpha=7, n_bounce=9)
    policies = _policy_subset(default_policies(max_nfev=max(1, int(args.max_nfev))), args.policy)

    args.out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "input": str(args.input),
        "out_root": str(args.out_root),
        "target_aspect": TARGET_ASPECT,
        "target_abs_iota_min": TARGET_ABS_IOTA_MIN,
        "resolution": asdict(resolution),
        "policies": [
            {
                "name": policy.name,
                "description": policy.description,
                "stages": [asdict(stage) for stage in policy.stages],
            }
            for policy in policies
        ],
    }
    (args.out_root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if not args.execute:
        print(f"Wrote bounded QI policy scan plan: {args.out_root / 'plan.json'}")
        return 0

    records: list[dict[str, Any]] = []
    for policy in policies:
        try:
            print(f"\n[qi-policy-scan] running {policy.name}", flush=True)
            records.append(
                run_policy(
                    policy,
                    input_file=args.input,
                    out_root=args.out_root,
                    resolution=resolution,
                    inner_max_iter=int(args.inner_max_iter),
                    trial_max_iter=int(args.trial_max_iter),
                )
            )
        except Exception as exc:  # noqa: BLE001 - policy scans should continue.
            records.append(
                {
                    "policy": policy.name,
                    "description": policy.description,
                    "selected": False,
                    "selection": "rejected",
                    "selection_reason": f"{type(exc).__name__}: {exc}",
                    "smooth_qi": None,
                    "legacy_qi": None,
                    "mirror": None,
                    "elongation": None,
                    "iota": None,
                    "aspect": None,
                    "wall_time_s": None,
                    "output_dir": str(args.out_root / policy.name),
                    "stages": [],
                }
            )
            print(f"[qi-policy-scan] {policy.name} failed: {type(exc).__name__}: {exc}", flush=True)
        write_summary(records, args.out_root)
    print(f"\nWrote QI policy scan summary: {args.out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

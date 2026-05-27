#!/usr/bin/env python3
"""Probe QA/QH/QP optimization inner/trial VMEC solve budgets.

This diagnostics runner mirrors the standalone QA/QH/QP examples but keeps
artifacts small and prints a compact JSON record for budget sweeps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64


enable_x64(True)

DATA_DIR = REPO_ROOT / "examples" / "data"


PROBLEM_DEFAULTS: dict[str, dict[str, Any]] = {
    "qa": {
        "warm_start": "input.nfp2_QA_omnigenity",
        "simple_seed": "input.minimal_seed_nfp2",
        "max_mode": 4,
        "method": "scipy",
        "scipy_tr_solver": "exact",
        "target_aspect": 5.0,
        "target_iota": 0.41,
        "helicity_m": 1,
        "helicity_n": 0,
        "aspect_weight": 1.0,
        "iota_weight": 10_000.0,
        "qs_weight": 1.0,
    },
    "qh": {
        "warm_start": "input.nfp4_QH_warm_start",
        "simple_seed": "input.minimal_seed_nfp4",
        "max_mode": 4,
        "method": "scipy",
        "scipy_tr_solver": "lsmr",
        "target_aspect": 5.0,
        "target_abs_iota_min": 0.41,
        "helicity_m": 1,
        "helicity_n": -1,
        "aspect_weight": 1.0,
        "iota_floor_weight": 40_000.0,
        "qs_weight": 1.0,
    },
    "qp": {
        "warm_start": "input.nfp2_QI",
        "simple_seed": "input.minimal_seed_nfp2",
        "max_mode": 5,
        "method": "scipy",
        "scipy_tr_solver": "lsmr",
        "target_aspect": 5.0,
        "target_abs_iota_min": 0.41,
        "helicity_m": 0,
        "helicity_n": -1,
        "aspect_weight": 1.0,
        "iota_floor_weight": 40_000.0,
        "qs_weight": 1.0,
        "project_input_boundary_to_max_mode": True,
        "max_mirror_ratio": 0.30,
        "max_elongation": 10.0,
        "mirror_weight": 20.0,
        "elongation_weight": 10.0,
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=sorted(PROBLEM_DEFAULTS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--max-nfev", type=int, default=6)
    parser.add_argument("--continuation-nfev", type=int, default=15)
    parser.add_argument("--max-mode", type=int, default=None)
    parser.add_argument("--min-vmec-mode", type=int, default=None)
    parser.add_argument("--inner-max-iter", type=int, default=550)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-10)
    parser.add_argument("--trial-max-iter", type=int, default=550)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-10)
    parser.add_argument("--ftol", type=float, default=1.0e-5)
    parser.add_argument("--gtol", type=float, default=1.0e-5)
    parser.add_argument("--xtol", type=float, default=1.0e-6)
    parser.add_argument("--solver-device", choices=("auto", "cpu", "gpu", "default"), default="auto")
    parser.add_argument("--use-ess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ess-alpha", type=float, default=1.2)
    parser.add_argument("--use-simple-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simple-seed-perturbation", type=float, default=1.0e-5)
    parser.add_argument("--use-mode-continuation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-stage-inputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-stage-wouts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-result-files", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _objective_problem(problem: str, defaults: dict[str, Any]) -> vj.LeastSquaresProblem:
    aspect = vj.AspectRatio()
    qs = vj.QuasisymmetryRatioResidual(
        helicity_m=int(defaults["helicity_m"]),
        helicity_n=int(defaults["helicity_n"]),
        surfaces=np.arange(0.0, 1.01, 0.1),
    )
    objective_tuples: list[tuple[Any, float, float]] = [
        (aspect.J, float(defaults["target_aspect"]), float(defaults["aspect_weight"])),
    ]
    if problem == "qa":
        iota = vj.MeanIota()
        objective_tuples.append((iota.J, float(defaults["target_iota"]), float(defaults["iota_weight"])))
    else:
        iota_floor = vj.AbsMeanIotaFloor(float(defaults["target_abs_iota_min"]))
        objective_tuples.append((iota_floor.J, 0.0, float(defaults["iota_floor_weight"])))
    objective_tuples.append((qs.J, 0.0, float(defaults["qs_weight"])))

    if problem == "qp":
        mirror = vj.VMECMirrorRatio(
            threshold=float(defaults["max_mirror_ratio"]),
            surfaces=np.linspace(0.1, 1.0, 6),
            smooth_extrema=2.0e-2,
            smooth_penalty=2.0e-2,
        )
        elongation = vj.MaxElongation(
            threshold=float(defaults["max_elongation"]),
            ntheta=48,
            nphi=16,
            smooth_extrema=2.0e-2,
            smooth_penalty=2.0e-2,
        )
        objective_tuples.extend(
            [
                (mirror.J, 0.0, float(defaults["mirror_weight"])),
                (elongation.J, 0.0, float(defaults["elongation_weight"])),
            ]
        )

    return vj.LeastSquaresProblem.from_tuples(objective_tuples)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def main() -> int:
    args = _build_parser().parse_args()
    defaults = PROBLEM_DEFAULTS[args.problem]

    max_mode = int(args.max_mode if args.max_mode is not None else defaults["max_mode"])
    min_vmec_mode = int(args.min_vmec_mode if args.min_vmec_mode is not None else max_mode + 2)
    use_mode_continuation = bool(not args.use_simple_seed if args.use_mode_continuation is None else args.use_mode_continuation)
    solver_device = None if args.solver_device in ("auto", "default") else args.solver_device

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_file = DATA_DIR / str(defaults["simple_seed" if args.use_simple_seed else "warm_start"])
    input_file = vj.prepare_simple_omnigenity_seed_input(
        input_file,
        args.output_dir,
        max_mode=max_mode,
        min_vmec_mode=min_vmec_mode,
        enabled=args.use_simple_seed,
        perturbation=args.simple_seed_perturbation,
    )
    stage_modes = vj.qs_stage_modes(
        max_mode=max_mode,
        use_mode_continuation=use_mode_continuation,
        continuation_nfev=args.continuation_nfev,
    )

    vmec = vj.FixedBoundaryVMEC.from_input(
        input_file,
        max_mode=max_mode,
        min_vmec_mode=min_vmec_mode,
        output_dir=args.output_dir,
        project_input_boundary_to_max_mode=bool(defaults.get("project_input_boundary_to_max_mode", False)),
    )
    problem = _objective_problem(args.problem, defaults)

    t0 = time.perf_counter()
    result = vj.least_squares_solve(
        vmec,
        problem,
        stage_modes=stage_modes,
        max_nfev=args.max_nfev,
        continuation_nfev=args.continuation_nfev,
        method=str(defaults["method"]),
        ftol=args.ftol,
        gtol=args.gtol,
        xtol=args.xtol,
        use_ess=args.use_ess,
        ess_alpha=args.ess_alpha,
        label=f"{args.problem.upper()} budget probe",
        use_mode_continuation=use_mode_continuation,
        inner_max_iter=args.inner_max_iter,
        inner_ftol=args.inner_ftol,
        trial_max_iter=args.trial_max_iter,
        trial_ftol=args.trial_ftol,
        solver_device=solver_device,
        scipy_tr_solver=str(defaults["scipy_tr_solver"]),
        save_stage_inputs=args.save_stage_inputs,
        save_stage_wouts=args.save_stage_wouts,
        save_final_outputs=False,
    )
    process_wall_time_s = time.perf_counter() - t0

    saved_paths = None
    if args.save_result_files:
        saved_paths = vj.save_optimization_result(result, output_dir=args.output_dir).as_dict()

    history = result.history
    timing = result.timing_summary
    record = {
        "problem": args.problem,
        "output_dir": args.output_dir,
        "max_mode": max_mode,
        "min_vmec_mode": min_vmec_mode,
        "stage_modes": list(stage_modes),
        "max_nfev": args.max_nfev,
        "continuation_nfev": args.continuation_nfev,
        "inner_max_iter": args.inner_max_iter,
        "inner_ftol": args.inner_ftol,
        "trial_max_iter": args.trial_max_iter,
        "trial_ftol": args.trial_ftol,
        "solver_device": solver_device,
        "use_ess": args.use_ess,
        "process_wall_time_s": process_wall_time_s,
        "solve_wall_time_s": float(timing.get("total_wall_time_s", process_wall_time_s)),
        "objective_final": float(history["objective_final"]),
        "qs_final": float(history["qs_final"]),
        "iota_final": float(history["iota_final"]),
        "aspect_final": float(history["aspect_final"]),
        "objective_history_tail": [float(x) for x in result.objective_history[-3:]],
        "summary": result.summary,
        "timing_summary": timing,
        "saved_paths": saved_paths,
    }
    record = _json_ready(record)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print("BUDGET_PROBE_JSON " + json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

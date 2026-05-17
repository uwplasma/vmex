#!/usr/bin/env python
"""Run bounded common-minimal-seed optimization showcase cases.

The README/docs showcase uses the same deliberately simple VMEC seed template
for each field period count:

- ``RBC(0,0)``
- ``RBC(0,1)``
- ``ZBS(0,1)``

This script maps that seed family to the requested targets:

- QI with NFP=1, 2, and 3
- QA with NFP=2
- QH with NFP=4
- QP with NFP=2

The implementation intentionally reuses ``generate_qs_ess_sweep.py`` so the
showcase follows the same exact optimizer, diagnostics, and per-case output
format as the larger benchmark matrix.  Defaults are bounded enough for local
representative runs; increase ``--max-nfev`` and ``--continuation-nfev`` for
publication-quality panels.

Examples:

  # One quick representative case.
  python examples/optimization/generate_minimal_seed_showcase.py --cases qa_nfp2 --max-nfev 2 --continuation-nfev 2

  # Full six-case README lane on CPU.
  python examples/optimization/generate_minimal_seed_showcase.py --cases all --backend-label cpu --solver-device cpu --max-nfev 30 --continuation-nfev 20 --case-timeout-s 1200

  # Render completed cases.
  python examples/optimization/render_minimal_seed_showcase.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "examples" / "data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "minimal_seed_showcase"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_qs_ess_sweep as sweep


@dataclass(frozen=True)
class MinimalSeedCase:
    """One target problem started from the common minimal boundary template."""

    name: str
    problem: str
    nfp: int
    input_file: Path
    qi_qp_preseed: bool | None = None
    qi_jit_booz: bool | None = None


SHOWCASE_CASES: dict[str, MinimalSeedCase] = {
    "qi_nfp1": MinimalSeedCase(
        name="qi_nfp1",
        problem="qi",
        nfp=1,
        input_file=DATA_DIR / "input.minimal_seed_nfp1",
        qi_qp_preseed=True,
        qi_jit_booz=True,
    ),
    "qi_nfp2": MinimalSeedCase(
        name="qi_nfp2",
        problem="qi",
        nfp=2,
        input_file=DATA_DIR / "input.minimal_seed_nfp2",
        qi_qp_preseed=True,
        qi_jit_booz=True,
    ),
    "qi_nfp3": MinimalSeedCase(
        name="qi_nfp3",
        problem="qi",
        nfp=3,
        input_file=DATA_DIR / "input.minimal_seed_nfp3",
        qi_qp_preseed=True,
        qi_jit_booz=True,
    ),
    "qa_nfp2": MinimalSeedCase(
        name="qa_nfp2",
        problem="qa",
        nfp=2,
        input_file=DATA_DIR / "input.minimal_seed_nfp2",
    ),
    "qh_nfp4": MinimalSeedCase(
        name="qh_nfp4",
        problem="qh",
        nfp=4,
        input_file=DATA_DIR / "input.minimal_seed_nfp4",
    ),
    "qp_nfp2": MinimalSeedCase(
        name="qp_nfp2",
        problem="qp",
        nfp=2,
        input_file=DATA_DIR / "input.minimal_seed_nfp2",
    ),
}

DEFAULT_CASE_ORDER = ("qi_nfp1", "qi_nfp2", "qi_nfp3", "qa_nfp2", "qh_nfp4", "qp_nfp2")


@dataclass(frozen=True)
class MinimalSeedBudget:
    """Bounded optimizer/VMEC budgets for a showcase run."""

    max_nfev: int
    continuation_nfev: int
    inner_max_iter: int
    inner_ftol: float
    trial_max_iter: int
    trial_ftol: float


def _parse_case_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not names or names == ("all",):
        return DEFAULT_CASE_ORDER
    unknown = sorted(set(names) - set(SHOWCASE_CASES))
    if unknown:
        known = ", ".join(DEFAULT_CASE_ORDER)
        raise ValueError(f"Unknown minimal-seed case(s): {', '.join(unknown)}. Known cases: {known}")
    return names


def _bool_from_choice(value: str) -> bool:
    return str(value).strip().lower() in {"on", "true", "1", "yes"}


def _case_output_dir(
    output_root: Path,
    *,
    case: MinimalSeedCase,
    backend_label: str,
    policy: str,
    max_mode: int,
    use_ess: bool,
) -> Path:
    qi_part = ""
    if case.problem == "qi":
        qi_part = "qp_preseed" if bool(case.qi_qp_preseed) else "no_qp_preseed"
    parts = [output_root, backend_label, case.name, policy]
    if qi_part:
        parts.append(qi_part)
    parts.extend([f"mode{int(max_mode)}", sweep._ess_label(bool(use_ess))])
    path = Path(parts[0])
    for part in parts[1:]:
        path = path / str(part)
    return path


def _problem_config_for_case(
    case: MinimalSeedCase,
    *,
    max_mode: int,
    budget: MinimalSeedBudget,
) -> sweep.ProblemConfig:
    """Return a sweep config patched to use the common minimal seed."""

    base = sweep.PROBLEM_CONFIGS[case.problem]
    min_vmec_mode = max(5, int(max_mode) + 2, int(base.min_vmec_mode))
    updates = {
        "input_file": case.input_file,
        "max_nfev": int(budget.max_nfev),
        "continuation_nfev": int(budget.continuation_nfev),
        "inner_max_iter": int(budget.inner_max_iter),
        "inner_ftol": float(budget.inner_ftol),
        "trial_max_iter": int(budget.trial_max_iter),
        "trial_ftol": float(budget.trial_ftol),
        "project_input_boundary_to_max_mode": True,
        "min_vmec_mode": min_vmec_mode,
    }
    if case.problem == "qi":
        updates.update(
            qi_preseed_qp=bool(case.qi_qp_preseed),
            qi_jit_booz=True if case.qi_jit_booz is None else bool(case.qi_jit_booz),
        )
    return replace(base, **updates)


def _qp_preseed_config_for_qi_case(
    case: MinimalSeedCase,
    *,
    max_mode: int,
    budget: MinimalSeedBudget,
) -> sweep.ProblemConfig:
    """Return a QP preseed config that uses the same NFP/minimal input as QI."""

    base = sweep.PROBLEM_CONFIGS["qp"]
    min_vmec_mode = max(5, int(max_mode) + 2, int(base.min_vmec_mode))
    return replace(
        base,
        input_file=case.input_file,
        max_nfev=int(budget.max_nfev),
        continuation_nfev=int(budget.continuation_nfev),
        inner_max_iter=int(budget.inner_max_iter),
        inner_ftol=float(budget.inner_ftol),
        trial_max_iter=int(budget.trial_max_iter),
        trial_ftol=float(budget.trial_ftol),
        project_input_boundary_to_max_mode=True,
        min_vmec_mode=min_vmec_mode,
    )


def _write_showcase_metadata(
    output_dir: Path,
    *,
    case: MinimalSeedCase,
    policy: str,
    max_mode: int,
    use_ess: bool,
    budget: MinimalSeedBudget,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "minimal_seed_case": asdict(case) | {"input_file": str(case.input_file)},
        "policy": str(policy),
        "max_mode": int(max_mode),
        "use_ess": bool(use_ess),
        "budget": asdict(budget),
    }
    (output_dir / "showcase_case.json").write_text(json.dumps(metadata, indent=2))


def _run_showcase_case(
    case: MinimalSeedCase,
    output_dir: Path,
    *,
    backend_label: str,
    solver_device: str | None,
    worker_jax_platforms: str | None,
    policy: str,
    max_mode: int,
    use_ess: bool,
    budget: MinimalSeedBudget,
) -> sweep.CaseResult:
    """Run one minimal-seed case with temporary sweep config overrides."""

    old_configs = dict(sweep.PROBLEM_CONFIGS)
    sweep.PROBLEM_CONFIGS[case.problem] = _problem_config_for_case(
        case,
        max_mode=max_mode,
        budget=budget,
    )
    if case.problem == "qi":
        sweep.PROBLEM_CONFIGS["qp"] = _qp_preseed_config_for_qi_case(
            case,
            max_mode=max_mode,
            budget=budget,
        )
    try:
        result = sweep._run_case(
            case.problem,
            int(max_mode),
            bool(use_ess),
            output_dir,
            use_mode_continuation=str(policy) == "continuation",
            policy=str(policy),
            backend=str(backend_label),
            solver_device=solver_device,
            jax_platforms=worker_jax_platforms,
            diagnostic_budgets=False,
            stellarator_asymmetric=False,
            qi_qp_preseed=case.qi_qp_preseed if case.problem == "qi" else None,
            qi_jit_booz=case.qi_jit_booz if case.problem == "qi" else None,
        )
    finally:
        sweep.PROBLEM_CONFIGS.clear()
        sweep.PROBLEM_CONFIGS.update(old_configs)
    return result


def _worker(
    case_name: str,
    output_dir_str: str,
    result_path_str: str,
    backend_label: str,
    solver_device: str | None,
    worker_jax_platforms: str | None,
    policy: str,
    max_mode: int,
    use_ess: bool,
    budget_dict: dict,
) -> None:
    output_dir = Path(output_dir_str)
    result_path = Path(result_path_str)
    case = SHOWCASE_CASES[case_name]
    budget = MinimalSeedBudget(**budget_dict)
    try:
        _write_showcase_metadata(
            output_dir,
            case=case,
            policy=policy,
            max_mode=max_mode,
            use_ess=use_ess,
            budget=budget,
        )
        result = _run_showcase_case(
            case,
            output_dir,
            backend_label=backend_label,
            solver_device=solver_device,
            worker_jax_platforms=worker_jax_platforms,
            policy=policy,
            max_mode=max_mode,
            use_ess=use_ess,
            budget=budget,
        )
    except Exception as exc:  # pragma: no cover - exercised by integration failures.
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "traceback.txt").write_text(traceback.format_exc())
        result = sweep.CaseResult(
            backend=str(backend_label),
            problem=case.problem,
            max_mode=int(max_mode),
            use_ess=bool(use_ess),
            success=False,
            crashed=True,
            message=f"{type(exc).__name__}: {exc}",
            policy=str(policy),
            output_dir=str(output_dir),
            solver_device=solver_device,
            jax_platforms=worker_jax_platforms,
            input_file=str(case.input_file),
            input_nfp=int(case.nfp),
            qi_qp_preseed=case.qi_qp_preseed if case.problem == "qi" else None,
            qi_jit_booz=case.qi_jit_booz if case.problem == "qi" else None,
        )
    result_path.write_text(json.dumps(asdict(result), indent=2))


def _read_result(path: Path) -> sweep.CaseResult:
    return sweep.CaseResult(**json.loads(path.read_text()))


def _write_showcase_summary(results: list[sweep.CaseResult], output_root: Path, summary_name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    records = [asdict(result) for result in results]
    (output_root / f"{summary_name}.json").write_text(json.dumps(records, indent=2))
    sweep._write_summary_csv(results, output_root / f"{summary_name}.csv")

    case_csv = output_root / f"{summary_name}_with_cases.csv"
    fieldnames = ["minimal_seed_case", "minimal_seed_nfp"] + list(records[0].keys()) if records else []
    if not fieldnames:
        case_csv.write_text("")
        return
    with case_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            row = asdict(result)
            output_dir = Path(str(result.output_dir or ""))
            meta_path = output_dir / "showcase_case.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            case_meta = meta.get("minimal_seed_case", {})
            row["minimal_seed_case"] = case_meta.get("name", "")
            row["minimal_seed_nfp"] = case_meta.get("nfp", "")
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cases", type=str, default="all", help="Comma-separated cases or 'all'.")
    parser.add_argument("--backend-label", type=str, default="cpu")
    parser.add_argument("--solver-device", type=str, default="cpu", help="Use 'cpu', 'gpu', or 'none'.")
    parser.add_argument("--worker-jax-platforms", type=str, default="cpu", help="Use 'inherit', 'cpu', or 'gpu'.")
    parser.add_argument("--policy", choices=("continuation", "direct"), default="continuation")
    parser.add_argument("--max-mode", type=int, default=3)
    parser.add_argument("--ess", choices=("on", "off"), default="on")
    parser.add_argument("--max-nfev", type=int, default=8)
    parser.add_argument("--continuation-nfev", type=int, default=8)
    parser.add_argument("--inner-max-iter", type=int, default=120)
    parser.add_argument("--inner-ftol", type=float, default=1e-9)
    parser.add_argument("--trial-max-iter", type=int, default=120)
    parser.add_argument("--trial-ftol", type=float, default=1e-9)
    parser.add_argument("--case-timeout-s", type=float, default=600.0)
    parser.add_argument("--rerun", action="store_true", help="Recompute cases even when case_result.json exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    case_names = _parse_case_names(args.cases)
    solver_device = None if str(args.solver_device).lower() in {"", "none", "default"} else str(args.solver_device)
    worker_jax_platforms = (
        None if str(args.worker_jax_platforms).lower() in {"", "none", "inherit"} else str(args.worker_jax_platforms)
    )
    budget = MinimalSeedBudget(
        max_nfev=int(args.max_nfev),
        continuation_nfev=int(args.continuation_nfev),
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
    )
    use_ess = _bool_from_choice(args.ess)
    max_mode = int(args.max_mode)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")

    results: list[sweep.CaseResult] = []
    for case_name in case_names:
        case = SHOWCASE_CASES[case_name]
        output_dir = _case_output_dir(
            output_root,
            case=case,
            backend_label=str(args.backend_label),
            policy=str(args.policy),
            max_mode=max_mode,
            use_ess=use_ess,
        )
        result_path = output_dir / "case_result.json"
        label = f"{case.name} problem={case.problem} nfp={case.nfp} policy={args.policy} mode={max_mode} ess={use_ess}"
        if result_path.exists() and not bool(args.rerun):
            result = _read_result(result_path)
            print(f"[{label}] skip existing success={result.success} crashed={result.crashed}", flush=True)
            results.append(result)
            continue
        if result_path.exists():
            result_path.unlink()

        old_platforms = os.environ.get("JAX_PLATFORMS")
        if worker_jax_platforms is not None:
            os.environ["JAX_PLATFORMS"] = worker_jax_platforms
        try:
            proc = ctx.Process(
                target=_worker,
                args=(
                    case_name,
                    str(output_dir),
                    str(result_path),
                    str(args.backend_label),
                    solver_device,
                    worker_jax_platforms,
                    str(args.policy),
                    max_mode,
                    use_ess,
                    asdict(budget),
                ),
            )
            t0 = time.perf_counter()
            proc.start()
        finally:
            if old_platforms is None:
                os.environ.pop("JAX_PLATFORMS", None)
            else:
                os.environ["JAX_PLATFORMS"] = old_platforms

        proc.join(timeout=None if args.case_timeout_s in (None, 0) else float(args.case_timeout_s))
        elapsed_s = time.perf_counter() - t0
        if proc.is_alive():
            sweep._terminate_worker_process(proc)
            result = sweep.CaseResult(
                backend=str(args.backend_label),
                problem=case.problem,
                max_mode=max_mode,
                use_ess=use_ess,
                success=False,
                crashed=True,
                message=f"worker timed out after {float(args.case_timeout_s):.1f} s",
                policy=str(args.policy),
                total_wall_time_s=elapsed_s,
                output_dir=str(output_dir),
                solver_device=solver_device,
                jax_platforms=worker_jax_platforms,
                input_file=str(case.input_file),
                input_nfp=int(case.nfp),
                qi_qp_preseed=case.qi_qp_preseed if case.problem == "qi" else None,
                qi_jit_booz=case.qi_jit_booz if case.problem == "qi" else None,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_showcase_metadata(
                output_dir,
                case=case,
                policy=str(args.policy),
                max_mode=max_mode,
                use_ess=use_ess,
                budget=budget,
            )
            result_path.write_text(json.dumps(asdict(result), indent=2))
        elif result_path.exists():
            result = _read_result(result_path)
        else:
            result = sweep.CaseResult(
                backend=str(args.backend_label),
                problem=case.problem,
                max_mode=max_mode,
                use_ess=use_ess,
                success=False,
                crashed=True,
                message=f"worker exit code {proc.exitcode} without result file",
                policy=str(args.policy),
                total_wall_time_s=elapsed_s,
                output_dir=str(output_dir),
                solver_device=solver_device,
                jax_platforms=worker_jax_platforms,
                input_file=str(case.input_file),
                input_nfp=int(case.nfp),
                qi_qp_preseed=case.qi_qp_preseed if case.problem == "qi" else None,
                qi_jit_booz=case.qi_jit_booz if case.problem == "qi" else None,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_showcase_metadata(
                output_dir,
                case=case,
                policy=str(args.policy),
                max_mode=max_mode,
                use_ess=use_ess,
                budget=budget,
            )
            result_path.write_text(json.dumps(asdict(result), indent=2))

        results.append(result)
        print(
            f"[{label}] success={result.success} crashed={result.crashed} objective={result.objective_final}",
            flush=True,
        )

    summary_name = f"summary_{args.backend_label}_{args.policy}_mode{max_mode}_{sweep._ess_label(use_ess)}"
    _write_showcase_summary(results, output_root, summary_name)
    print(f"Wrote {output_root / (summary_name + '.csv')}")


if __name__ == "__main__":
    main()

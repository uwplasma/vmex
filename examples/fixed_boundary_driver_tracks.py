"""Compare the conservative and optimized fixed-boundary driver tracks.

This example is intended for users who want to drive `vmec_jax` from Python
instead of the CLI while still understanding the difference between:

- the conservative VMEC2000-like parity path, and
- the optimized fixed-boundary CLI policy used by `vmec input.name`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import vmec_jax as vj


def _case_name(input_path: Path) -> str:
    name = input_path.name
    if name.startswith("input."):
        return name.split("input.", 1)[-1]
    if name.startswith("input_"):
        return name.split("input_", 1)[-1]
    return input_path.stem


def _track_summary(*, label: str, solver_mode: str, cli_fixed_boundary_mode: bool, run, seconds: float) -> dict[str, object]:
    diag = dict(run.result.diagnostics) if run.result is not None else {}
    fsq_total = float(np.asarray(run.result.w_history)[-1]) if run.result is not None else float("nan")
    return {
        "label": str(label),
        "solver_mode": str(solver_mode),
        "cli_fixed_boundary_mode": bool(cli_fixed_boundary_mode),
        "seconds": float(seconds),
        "converged": bool(diag.get("converged", False)),
        "fsq_total": float(fsq_total),
        "use_scan": bool(diag.get("use_scan", False)),
        "initial_policy": diag.get("cli_fixed_boundary_initial_policy", ""),
        "staged_followup_used": bool(diag.get("cli_fixed_boundary_staged_followup_used", False)),
        "full_parity_fallback": bool(diag.get("cli_fixed_boundary_full_parity_fallback", False)),
    }


def _print_summaries(rows: list[dict[str, object]]) -> None:
    print()
    print("Track summary")
    print("-" * 88)
    print(
        f"{'label':<14} {'seconds':>10} {'fsq_total':>14} {'conv':>6} "
        f"{'scan':>6} {'policy':<18} {'staged?':>8} {'fallback?':>10}"
    )
    for row in rows:
        print(
            f"{str(row['label']):<14} "
            f"{float(row['seconds']):>10.3f} "
            f"{float(row['fsq_total']):>14.6e} "
            f"{str(bool(row['converged'])):>6} "
            f"{str(bool(row['use_scan'])):>6} "
            f"{str(row['initial_policy']):<18} "
            f"{str(bool(row['staged_followup_used'])):>8} "
            f"{str(bool(row['full_parity_fallback'])):>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parity and optimized fixed-boundary tracks side by side.")
    parser.add_argument(
        "input",
        nargs="?",
        default=str(Path(__file__).resolve().parents[0] / "data" / "input.LandremanPaul2021_QA_lowres"),
        help="Path to an input.* file. Defaults to the bundled QA example.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[0] / "outputs" / "driver_tracks"),
        help="Directory for the generated wout files and optional JSON summary.",
    )
    parser.add_argument("--max-iter", type=int, default=None, help="Optional total iteration override.")
    parser.add_argument("--no-write-wout", action="store_true", help="Skip writing wout files.")
    parser.add_argument("--json", action="store_true", help="Write a JSON summary alongside the wout files.")
    parser.add_argument("--quiet", action="store_true", help="Silence solver progress output.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    case = _case_name(input_path)

    tracks = [
        ("parity", "parity", False),
        ("optimized_cli", "accelerated", True),
    ]
    rows: list[dict[str, object]] = []

    for label, solver_mode, cli_fixed_boundary_mode in tracks:
        print(
            f"[vmec_jax] running {label} track "
            f"(solver_mode={solver_mode}, cli_fixed_boundary_mode={bool(cli_fixed_boundary_mode)})",
            flush=True,
        )
        run_kwargs = dict(
            solver="vmec2000_iter",
            solver_mode=str(solver_mode),
            verbose=not bool(args.quiet),
            cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
        )
        if args.max_iter is not None:
            run_kwargs["max_iter"] = int(args.max_iter)

        t0 = perf_counter()
        run = vj.run_fixed_boundary(str(input_path), **run_kwargs)
        seconds = perf_counter() - t0
        rows.append(
            _track_summary(
                label=str(label),
                solver_mode=str(solver_mode),
                cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
                run=run,
                seconds=float(seconds),
            )
        )
        print(
            f"[vmec_jax] completed {label} track in {seconds:.3f}s "
            f"(fsq_total={rows[-1]['fsq_total']:.6e}, converged={rows[-1]['converged']})",
            flush=True,
        )
        if not bool(args.no_write_wout):
            wout_path = outdir / f"wout_{case}_{label}.nc"
            vj.write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)

    _print_summaries(rows)

    if bool(args.json):
        summary_path = outdir / f"{case}_driver_tracks_summary.json"
        summary_path.write_text(json.dumps(rows, indent=2))
        print()
        print(f"Wrote {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

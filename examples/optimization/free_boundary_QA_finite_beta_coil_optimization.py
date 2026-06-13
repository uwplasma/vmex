#!/usr/bin/env python
"""QA finite-beta single-stage free-boundary direct-coil optimization.

This is a deliberately small pedagogic wrapper around
``free_boundary_QS_coil_optimization.py``.  It uses the QA finite-beta input
deck and standard finite-beta pressure profile, then optimizes only direct-coil
degrees of freedom.  The plasma boundary is not part of the optimization
vector.

Every accepted trial is evaluated by a complete ``run_free_boundary`` solve
with the direct-coil provider.  Optional same-branch derivative reports remain
diagnostics/proposal evidence only; this example does not claim exact
gradients through adaptive free-boundary branch selection.

Minimal smoke from the repository root:

    python examples/optimization/free_boundary_QA_finite_beta_coil_optimization.py --smoke --dry-run

Run a tiny complete-solve smoke:

    python examples/optimization/free_boundary_QA_finite_beta_coil_optimization.py --smoke --max-evals 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from free_boundary_QS_coil_optimization import (
        SKIP_EXIT_CODE,
        SkipExample,
        apply_smoke_defaults as apply_base_smoke_defaults,
        build_parser as build_base_parser,
        optimize_coils as optimize_base_coils,
        write_json,
    )
except ModuleNotFoundError:
    from examples.optimization.free_boundary_QS_coil_optimization import (
        SKIP_EXIT_CODE,
        SkipExample,
        apply_smoke_defaults as apply_base_smoke_defaults,
        build_parser as build_base_parser,
        optimize_coils as optimize_base_coils,
        write_json,
    )


DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.nfp2_QA_finite_beta"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "free_boundary_QA_finite_beta_coil_optimization"
DEFAULT_BETA_PERCENT = 2.5
DEFAULT_PHIEDGE = 90.36222435222507
DEFAULT_CIRCLE_CURRENT = 3.0e7
DEFAULT_CIRCLE_RADIUS = 14.0
DEFAULT_TARGET_ASPECT = 5.0
DEFAULT_TARGET_IOTA = 0.4


def finite_beta_qa_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Return the small contract this wrapper adds to the base workflow."""

    return {
        "configuration": "QA finite-beta free-boundary direct-coil single-stage example",
        "input_template": str(args.input),
        "beta_percent": float(args.beta),
        "pressure_profile": str(args.pressure_profile),
        "helicity_m": int(args.helicity_m),
        "helicity_n": int(args.helicity_n),
        "optimized_dofs": "coil currents and selected direct-coil Fourier coefficients only",
        "plasma_boundary_optimized": False,
        "complete_solve_acceptance_authority": True,
        "gradient_claim": (
            "No exact adaptive full-loop gradients are promoted. Optional "
            "same-branch reports are fixed-accepted-branch diagnostics or "
            "proposal evidence; complete solves decide acceptance."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = __doc__
    parser.set_defaults(
        input=DEFAULT_INPUT,
        outdir=DEFAULT_OUTDIR,
        provider="circle",
        beta=DEFAULT_BETA_PERCENT,
        pressure_profile="standard",
        pressure_scale=0.0,
        phiedge=DEFAULT_PHIEDGE,
        helicity_m=1,
        helicity_n=0,
        target_aspect=DEFAULT_TARGET_ASPECT,
        target_iota=DEFAULT_TARGET_IOTA,
        qs_surfaces="0.25,0.5,0.75",
        circle_current=DEFAULT_CIRCLE_CURRENT,
        circle_radius=DEFAULT_CIRCLE_RADIUS,
        circle_n_segments=96,
        circle_nfp=1,
        circle_stellsym=False,
        max_current_vars=1,
        max_fourier_vars=2,
        current_step=0.03,
        dof_step=2.0e-2,
    )
    return parser


def apply_example_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args = apply_base_smoke_defaults(args)
    if args.smoke:
        args.max_evals = min(int(args.max_evals), 2)
    return args


def optimize_finite_beta_qa_coils(args: argparse.Namespace) -> dict[str, Any]:
    summary = optimize_base_coils(args)
    summary["phase"] = "qa-finite-beta-single-stage-direct-coil-validation"
    summary["scope"] = "QA finite-beta coil-only free-boundary optimization example"
    summary["finite_beta_qa_example"] = finite_beta_qa_metadata(args)
    summary["plasma_boundary_optimized"] = False
    summary["best_selection"] = "minimum objective among complete free-boundary solve evaluations"
    write_json(Path(summary["outdir"]) / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = apply_example_defaults(parser.parse_args(argv))
    try:
        optimize_finite_beta_qa_coils(args)
    except SkipExample as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        return SKIP_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

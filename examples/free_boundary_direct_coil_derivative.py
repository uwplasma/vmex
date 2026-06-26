#!/usr/bin/env python
"""Validate direct-coil free-boundary derivatives against complete-solve FD.

This example shows the public branch-local derivative workflow:

1. write a low-resolution free-boundary VMEC input with ``MGRID_FILE='DIRECT_COILS'``;
2. create differentiable Fourier coils in pure JAX;
3. choose a coil-current or coil-shape tangent;
4. compute complete-solve values and same-branch branch-local JVPs; and
5. compare the JVPs with central finite differences through complete solves.

The derivative path is intentionally conservative.  It differentiates a fixed
accepted free-boundary branch and records the complete-solve branch fingerprint;
hard accepted/rejected adaptive-branch changes remain nonsmooth events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.solvers.free_boundary import (
    FreeBoundaryDerivativeOptions,
    coil_direction,
    free_boundary_value_and_jvp,
)

from examples.optimization.free_boundary_QS_coil_optimization import (
    DEFAULT_FREE_BOUNDARY_PHIEDGE,
    DEFAULT_INPUT,
    make_circle_provider,
    make_free_boundary_indata,
)


DEFAULT_OUTDIR = REPO_ROOT / "results" / "free_boundary_direct_coil_derivative"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        return np.asarray(value).tolist()
    except Exception:
        return str(value)


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-ready derivative evidence without embedding solver payloads."""

    fd = report.get("fd_validation") or {}
    scalar_report = fd.get("scalar_report") or {}
    public_scalar_report = fd.get("public_scalar_report") or {}
    return {
        "contract": report["contract"],
        "differentiates_adaptive_controller": report["differentiates_adaptive_controller"],
        "differentiates_fixed_accepted_branch": report["differentiates_fixed_accepted_branch"],
        "values": report["values"],
        "directional_derivatives": report["directional_derivatives"],
        "base_abs_delta": report["base_abs_delta"],
        "fd_passed": scalar_report.get("passed"),
        "fd_scalars": public_scalar_report,
        "cotangent_vjp_fd_check": report.get("cotangent_vjp_fd_check"),
        "scalar_keys": report["scalar_keys"],
        "requested_outputs": report["requested_outputs"],
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the derivative report example."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--ns", type=int, default=12)
    parser.add_argument("--mpol", type=int, default=3)
    parser.add_argument("--ntor", type=int, default=2)
    parser.add_argument("--nzeta", type=int, default=4)
    parser.add_argument("--activate-fsq", type=float, default=1.0e99)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--helicity-m", type=int, default=1)
    parser.add_argument("--helicity-n", type=int, default=0)
    parser.add_argument("--qs-surfaces", default="0.5")
    parser.add_argument("--qs-ntheta", type=int, default=12)
    parser.add_argument("--qs-nphi", type=int, default=12)
    parser.add_argument(
        "--direction",
        choices=("current", "shape", "both"),
        default="both",
        help="Coil tangent used in the JVP and central-FD comparison.",
    )
    parser.add_argument("--current-tangent", type=float, default=0.05)
    parser.add_argument("--shape-tangent", type=float, default=1.0e-3)
    parser.add_argument("--no-fd", action="store_true", help="Skip central-FD validation and only compute the JVP.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the derivative example and write a compact report artifact."""

    args = build_parser().parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = make_free_boundary_indata(
        args.input,
        outdir / "input.direct_coil_derivative",
        vmec_max_iter=int(args.max_iter),
        ftol=float(args.ftol),
        ns=int(args.ns),
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        nzeta=int(args.nzeta),
        beta_percent=0.0,
        pressure_profile="linear-scale",
        pressure_scale=0.0,
        phiedge=DEFAULT_FREE_BOUNDARY_PHIEDGE,
    )
    params, metadata = make_circle_provider(current_scale=1.0, n_segments=64)
    direction = coil_direction(
        params,
        current=float(args.current_tangent) if args.direction in {"current", "both"} else 0.0,
        curve_dof=float(args.shape_tangent) if args.direction in {"shape", "both"} else 0.0,
        curve_index=(0, 0, 2) if args.direction in {"shape", "both"} else None,
    )
    options = FreeBoundaryDerivativeOptions(
        helicity_m=int(args.helicity_m),
        helicity_n=int(args.helicity_n),
        qs_surfaces=tuple(float(part) for part in str(args.qs_surfaces).replace(",", " ").split()),
        qs_ntheta=int(args.qs_ntheta),
        qs_nphi=int(args.qs_nphi),
        replay_kwargs={"use_stacked_step_controls": True, "use_accepted_only_fast_path": True},
    )
    solve_kwargs = {
        "max_iter": int(args.max_iter),
        "ftol": float(args.ftol),
        "vmec2000_control": True,
        "auto_flip_force": False,
        "use_direct_fallback": True,
        "verbose": False,
        "verbose_vmec2000_table": False,
        "jit_forces": True,
        "use_scan": False,
        "host_update_assembly": False,
        "adjoint_trace": True,
        "adjoint_trace_mode": "branch",
        "external_field_provider_kind": "direct_coils",
        "free_boundary_activate_fsq": float(args.activate_fsq),
    }
    report = free_boundary_value_and_jvp(
        input_path,
        params,
        direction_params=direction,
        outputs=("aspect", "mean_iota", "boundary_displacement", "bnormal_rms", "qs_residual"),
        cotangent={
            "aspect": 1.0,
            "mean_iota": 1.0,
            "boundary_displacement": 1.0,
            "bnormal_rms": 1.0,
            "qs_residual": 1.0,
        },
        options=options,
        solve_kwargs=solve_kwargs,
        validate_fd=not bool(args.no_fd),
        fd_epsilon=float(args.fd_epsilon),
        include_payload=False,
    )
    compact = _compact_report(report)
    compact["coil_provider"] = metadata
    compact["input_path"] = str(input_path)
    compact["direction"] = str(args.direction)
    report_path = outdir / "free_boundary_direct_coil_derivative_report.json"
    report_path.write_text(json.dumps(compact, indent=2, default=_json_default) + "\n")

    print("Direct-coil free-boundary derivative report")
    print(f"  input:   {input_path}")
    print(f"  report:  {report_path}")
    print("  branch differentiability: fixed accepted branch only")
    for name, value in compact["values"].items():
        deriv = compact["directional_derivatives"].get(name)
        print(f"  {name:22s} value={float(np.asarray(value)): .6e}  jvp={float(np.asarray(deriv)): .6e}")
    if compact["fd_passed"] is not None:
        print(f"  complete-solve FD validation passed: {compact['fd_passed']}")
    cotangent_check = compact.get("cotangent_vjp_fd_check") or {}
    if cotangent_check.get("available") and cotangent_check.get("fd_available"):
        print(
            "  cotangent VJP/FD check: "
            f"passed={cotangent_check['passed']} "
            f"ad={cotangent_check['ad_cotangent_directional']:.6e} "
            f"fd={cotangent_check['fd_cotangent_directional']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

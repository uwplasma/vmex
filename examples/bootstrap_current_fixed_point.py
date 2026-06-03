#!/usr/bin/env python
"""Compute a self-consistent Redl bootstrap-current VMEC input profile.

This example is intentionally explicit: the user selects the finite-beta
profiles, VMEC input, fixed-point controls, solver controls, output files, and
then calls ``vj.bootstrap_current_fixed_point``.  The plasma boundary is not an
optimization variable here; the loop only updates VMEC's current profile from
the Redl bootstrap-current formula.

Run from the repository root:

    PYTHONPATH=. python examples/bootstrap_current_fixed_point.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import vmec_jax as vj
from vmec_jax.driver import write_wout_from_fixed_boundary_run


REPO_ROOT = Path(__file__).resolve().parents[1]

# Physics/profile controls.  For stellarator studies set HELICITY_N to the
# target quasisymmetry helicity used by the Redl formula.
INPUT_PATH = REPO_ROOT / "examples" / "data" / "input.shaped_tokamak_pressure"
BETA_PERCENT = 1.0
HELICITY_N = 0
REDL_SURFACES = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
N_CURRENT = 32

# Fixed-point controls.  `integrating_factor` is the most faithful deterministic
# update; `low_beta` and `lagged_pressure` are useful diagnostics.
FIXED_POINT_OPTIONS = vj.BootstrapCurrentOptions(
    helicity_n=HELICITY_N,
    surfaces=REDL_SURFACES,
    n_current=N_CURRENT,
    policy="integrating_factor",  # alternatives: "low_beta", "lagged_pressure"
    damping=0.5,  # set closer to 1.0 for aggressive Picard updates
    max_fixed_point_iter=3,
    mismatch_tol=1.0e-2,
    current_tol=1.0e-2,
)

# VMEC solve controls for each fixed-point stage.
VMEC_RUN_KWARGS = {
    "max_iter": 250,
    "multigrid": False,
    "verbose": False,
    "jit_forces": "auto",
    "solver_device": None,  # use "cpu" or "gpu" to force a backend
}

RESULTS_DIR = REPO_ROOT / "results" / "bootstrap_current_fixed_point"
FINAL_INPUT = RESULTS_DIR / "input.bootstrap_current_final"
FINAL_WOUT = RESULTS_DIR / "wout_bootstrap_current_final.nc"
HISTORY_JSON = RESULTS_DIR / "history.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded VMEC/Redl fixed-point loop that updates the VMEC "
            "current profile from bootstrap-current diagnostics."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--beta-percent", type=float, default=BETA_PERCENT)
    parser.add_argument("--helicity-n", type=int, default=HELICITY_N)
    parser.add_argument("--max-fixed-point-iter", type=int, default=FIXED_POINT_OPTIONS.max_fixed_point_iter)
    parser.add_argument("--vmec-max-iter", type=int, default=VMEC_RUN_KWARGS["max_iter"])
    parser.add_argument(
        "--solver-device",
        choices=("cpu", "gpu"),
        default=None,
        help="Force VMEC solves to one backend. Omit to let JAX choose.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the selected configuration without running VMEC.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    results_dir = args.outdir
    final_input = results_dir / "input.bootstrap_current_final"
    final_wout = results_dir / "wout_bootstrap_current_final.nc"
    history_json = results_dir / "history.json"

    fixed_point_options = vj.BootstrapCurrentOptions(
        helicity_n=args.helicity_n,
        surfaces=REDL_SURFACES,
        n_current=N_CURRENT,
        policy=FIXED_POINT_OPTIONS.policy,
        damping=FIXED_POINT_OPTIONS.damping,
        max_fixed_point_iter=args.max_fixed_point_iter,
        mismatch_tol=FIXED_POINT_OPTIONS.mismatch_tol,
        current_tol=FIXED_POINT_OPTIONS.current_tol,
    )
    vmec_run_kwargs = {
        **VMEC_RUN_KWARGS,
        "max_iter": args.vmec_max_iter,
        "solver_device": args.solver_device,
    }

    if args.dry_run:
        print(f"Input:              {args.input}")
        print(f"Output directory:   {results_dir}")
        print(f"Beta percent:       {args.beta_percent}")
        print(f"Helicity N:         {args.helicity_n}")
        print(f"Fixed-point options:{fixed_point_options}")
        print(f"VMEC run kwargs:    {vmec_run_kwargs}")
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)

    base_indata = vj.read_indata(args.input)
    profiles = vj.standard_finite_beta_profiles(args.beta_percent)
    indata = vj.with_pressure_profile(base_indata, profiles.pressure_pa, pres_scale=1.0)

    result = vj.bootstrap_current_fixed_point(
        indata,
        options=fixed_point_options,
        ne_coeffs=profiles.ne_coeffs,
        Te_coeffs=profiles.Te_coeffs,
        Ti_coeffs=profiles.Ti_coeffs,
        Zeff_coeffs=profiles.Zeff_coeffs,
        run_kwargs=vmec_run_kwargs,
    )

    vj.write_indata(final_input, result.indata)
    if result.last_run is not None:
        write_wout_from_fixed_boundary_run(final_wout, result.last_run, include_fsq=True)

    history = [asdict(item) for item in result.history]
    history_json.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "final_input": str(final_input),
                "final_wout": str(final_wout),
                "beta_percent": args.beta_percent,
                "helicity_n": args.helicity_n,
                "converged": result.converged,
                "reason": result.reason,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )

    last = result.history[-1] if result.history else None
    print(f"Wrote final input: {final_input}")
    if result.last_run is not None:
        print(f"Wrote final WOUT:  {final_wout}")
    print(f"Wrote history:     {history_json}")
    print(f"Converged:         {result.converged} ({result.reason})")
    if last is not None:
        print(f"Iterations:        {last.iteration}")
        print(f"CURTOR:            {last.curtor:.6e}")
        print(f"Mismatch norm:     {last.mismatch_norm:.6e}")
        print(f"Current update:    {last.current_update_norm:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python

"""Plot results from the QH fixed-resolution exact-adjoint optimization.

Reads the wout files and history JSON produced by
``qh_fixed_resolution_jax.py`` (or the legacy ``qh_fixed_resolution_exact.py``)
and generates three figures in the output directory:

  - ``boundary_comparison.png``   3D LCFS coloured by |B| (initial vs final)
  - ``bmag_surface.png``          |B| contour lines on LCFS (initial vs final)
  - ``objective_history.png``     Objective value and aspect ratio vs iteration

Usage
-----
Run the optimisation first::

    python examples/optimization/qh_fixed_resolution_jax.py

Then plot (or regenerate after editing the plotting code)::

    python examples/optimization/plot_qh_optimization_results.py \\
        --output-dir results/qh_opt

All figures are saved inside ``--output-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vmec_jax as vj


def _parse_args():
    p = argparse.ArgumentParser(description="Plot QH optimization results")
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/qh_opt",
        help="Directory containing wout_initial.nc, wout_final.nc, history.json",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    outdir = Path(args.output_dir)

    wout_init_path = outdir / "wout_initial.nc"
    wout_final_path = outdir / "wout_final.nc"
    history_path = outdir / "history.json"

    for p in (wout_init_path, wout_final_path, history_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.  Run qh_fixed_resolution_jax.py first:\n"
                f"  python examples/optimization/qh_fixed_resolution_jax.py"
            )

    print(f"Generating plots from {outdir} …")
    paths = vj.plot_qh_optimization(
        wout_init_path,
        wout_final_path,
        history_path,
        outdir=outdir,
    )

    import json
    with open(history_path) as f:
        data = json.load(f)
    print()
    print(f"Summary: {data['nfev']} residual evals in {data['total_wall_time_s']:.1f} s")
    print(f"  Objective: {data['objective_initial']:.4f}  →  {data['objective_final']:.4f}")
    print(f"  QS total:  {data['qs_initial']:.4f}  →  {data['qs_final']:.4f}")
    print(f"  Aspect:    {data['aspect_initial']:.3f}  →  {data['aspect_final']:.3f}")
    print("Done.")


if __name__ == "__main__":
    main()

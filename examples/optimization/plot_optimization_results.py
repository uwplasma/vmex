#!/usr/bin/env python

"""Plot results from a fixed-boundary optimization.

Reads the wout files and history JSON produced by
the QA/QH/QP/QI optimization examples and generates three figures in the output
directory:

  - ``boundary_comparison.png``   3D LCFS coloured by |B| (initial vs final)
  - ``boozer_lcfs_bmag_comparison.png``
                                  |B| contour lines on the LCFS in Boozer coordinates
  - ``objective_history.png``     Objective value and aspect ratio vs iteration

Usage
-----
Run the optimisation first::

    python examples/optimization/QH_optimization.py

Then plot (or regenerate after editing the plotting code)::

    python examples/optimization/plot_optimization_results.py \\
        --output-dir results/qh_opt

All figures are saved inside ``--output-dir``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vmec_jax as vj


def _parse_args():
    p = argparse.ArgumentParser(description="Plot fixed-boundary optimization results")
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
                f"{p} not found. Run one of the optimization examples first, e.g.:\n"
                f"  python examples/optimization/QH_optimization.py"
            )

    print(f"Generating plots from {outdir} …")
    paths = {
        "boundary_comparison": vj.plot_3d_boundary_comparison(
            wout_init_path,
            wout_final_path,
            outdir=outdir,
        ),
        "boozer_lcfs_bmag_contours": vj.plot_boozer_lcfs_bmag_comparison(
            wout_init_path,
            wout_final_path,
            outdir=outdir,
        ),
        "objective_history": vj.plot_objective_history(
            history_path,
            outdir=outdir,
        ),
    }
    for name, path in paths.items():
        print(f"  {name}: {path}")

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

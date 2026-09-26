#!/usr/bin/env python
"""Bounded L-BFGS-B free-boundary single-stage optimization.

Use the scalar example's initialization, weighted objective and full
post-processing. Iota/radius feasibility is reported at the endpoint; L-BFGS-B
applies coil-parameter bounds, not nonlinear constraints. Currents stay fixed.
"""

from free_boundary_single_stage_optimization_scalar import main as run


def main(argv=None):
    """Run the shared production workflow with L-BFGS-B."""
    return run(argv, method="L-BFGS-B")


if __name__ == "__main__":
    raise SystemExit(main())

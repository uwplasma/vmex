#!/usr/bin/env python
"""Quasi-helical symmetry optimization with vmec_jax.

This mirrors the compact SIMSOPT example style: choose the VMEC input, choose
the boundary modes, build an objective list, choose an optimizer, and run.

To add a new objective, append an ``ObjectiveTerm`` to ``OBJECTIVES``.  The
callback receives ``(ctx, state)`` and returns a scalar or vector.  vmec_jax
minimizes ``weight * (value - target)`` with exact discrete-adjoint Jacobians.
"""

from pathlib import Path

import numpy as np

try:
    from fixed_boundary_qs_common import (
        FixedBoundaryQSConfig,
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        aspect_objective,
        quasisymmetry_objective,
        run_qs_optimization,
    )
except ModuleNotFoundError:
    from examples.optimization.fixed_boundary_qs_common import (
        FixedBoundaryQSConfig,
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        aspect_objective,
        quasisymmetry_objective,
        run_qs_optimization,
    )


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# User-editable run controls.
INPUT_FILE = DATA_DIR / "input.nfp4_QH_warm_start"
OUTPUT_DIR = Path("results/qh_opt")
VMEC_MPOL = 5
VMEC_NTOR = 5

MAX_MODE = 1
MAX_NFEV = 15
CONTINUATION_NFEV = 10
USE_MODE_CONTINUATION = True

METHOD = "scipy"  # "scipy", "gauss_newton", "lbfgs_adjoint", or "scipy_matrix_free"
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = None
FTOL = 1.0e-3
GTOL = 1.0e-3
XTOL = 1.0e-3

# 0 means use NITER/FTOL from the VMEC input deck for accepted exact points.
INNER_MAX_ITER = 0
INNER_FTOL = 0.0
TRIAL_MAX_ITER = 300
TRIAL_FTOL = 1.0e-10
SOLVER_DEVICE = None  # set to "cpu" or "gpu" to force one backend

# QH target and objective weights.
HELICITY_M = 1
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)
TARGET_ASPECT = 7.0
ASPECT_WEIGHT = 1.0
QS_WEIGHT = 1.0

# ESS preconditions high-mode boundary DOFs.  The full QA/QH/QP/QI sweep uses
# alpha=2.5, so changing USE_ESS here reproduces the panel policy.
USE_ESS = False
ALPHA = 2.5

# The objective list is the main teaching point.  Add custom terms here.
OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    quasisymmetry_objective(
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
        surfaces=SURFACES,
        weight=QS_WEIGHT,
    ),
    # Example custom scalar objective:
    # ObjectiveTerm(
    #     "major_radius",
    #     evaluate=lambda ctx, state: state.rmncc[0, 0],
    #     target=1.0,
    #     weight=0.1,
    # ),
]

CONFIG = FixedBoundaryQSConfig(
    input_file=INPUT_FILE,
    output_dir=OUTPUT_DIR,
    vmec_mpol=VMEC_MPOL,
    vmec_ntor=VMEC_NTOR,
    max_mode=MAX_MODE,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    use_mode_continuation=USE_MODE_CONTINUATION,
    use_ess=USE_ESS,
    ess_alpha=ALPHA,
    method=METHOD,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    target_aspect=TARGET_ASPECT,
    label=f"QH opt (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
)


if __name__ == "__main__":
    run_qs_optimization(CONFIG, OBJECTIVES)

#!/usr/bin/env python
"""Quasi-axisymmetric optimization with vmec_jax.

The script is intentionally compact: the user edits top-level variables,
constructs an objective list, and chooses the optimizer.  To add another target,
append an ``ObjectiveTerm`` to ``OBJECTIVES``.
"""

from pathlib import Path

import numpy as np

try:
    from fixed_boundary_qs_common import (
        FixedBoundaryQSConfig,
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        aspect_objective,
        mean_iota_objective,
        quasisymmetry_objective,
        run_qs_optimization,
    )
except ModuleNotFoundError:
    from examples.optimization.fixed_boundary_qs_common import (
        FixedBoundaryQSConfig,
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        aspect_objective,
        mean_iota_objective,
        quasisymmetry_objective,
        run_qs_optimization,
    )


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

INPUT_FILE = DATA_DIR / "input.nfp2_QA"
VMEC_MPOL = 5
VMEC_NTOR = 5

MAX_MODE = 1
MAX_NFEV = 15
CONTINUATION_NFEV = 10
USE_MODE_CONTINUATION = True

METHOD = "scipy"
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
SOLVER_DEVICE = None

# QA uses helicity N=0 and includes a target-iota residual.
HELICITY_M = 1
HELICITY_N = 0
SURFACES = np.arange(0.0, 1.01, 0.1)
TARGET_ASPECT = 6.0
TARGET_IOTA = 0.41

ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 1.0
QS_WEIGHT = 1.0

USE_ESS = False
ALPHA = 2.5

if MAX_MODE >= 2:
    # The higher-mode QA direct problem is sensitive to the lower-mode seed.
    CONTINUATION_NFEV = max(CONTINUATION_NFEV, 25)

OUTPUT_DIR = Path(f"results/qa_opt/{'ess' if USE_ESS else 'no_ess'}")

OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    mean_iota_objective(TARGET_IOTA, IOTA_WEIGHT),
    quasisymmetry_objective(
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
        surfaces=SURFACES,
        weight=QS_WEIGHT,
    ),
    # Example custom scalar objective:
    # ObjectiveTerm("custom", lambda ctx, state: your_metric(ctx, state), target=0.0, weight=1.0),
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
    target_iota=TARGET_IOTA,
    label=f"QA opt (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
)


if __name__ == "__main__":
    run_qs_optimization(CONFIG, OBJECTIVES)

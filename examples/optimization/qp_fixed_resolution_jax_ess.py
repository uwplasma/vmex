#!/usr/bin/env python
"""Quasi-poloidal symmetry optimization with vmec_jax.

QP is still a quasisymmetry problem: ``HELICITY_M = 0`` targets
``|B| ~ B(N zeta)``.  This example shows the same objective-list pattern as the
QA/QH scripts, with an additional lower-bound penalty on ``abs(mean_iota)``.
"""

from pathlib import Path

import numpy as np

try:
    from fixed_boundary_qs_common import (
        FixedBoundaryQSConfig,
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        abs_mean_iota_floor_objective,
        aspect_objective,
        quasisymmetry_objective,
        run_qs_optimization,
    )
except ModuleNotFoundError:
    from examples.optimization.fixed_boundary_qs_common import (
        FixedBoundaryQSConfig,
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        abs_mean_iota_floor_objective,
        aspect_objective,
        quasisymmetry_objective,
        run_qs_optimization,
    )


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

INPUT_FILE = DATA_DIR / "input.nfp4_QH_warm_start"
VMEC_MPOL = 5
VMEC_NTOR = 5

MAX_MODE = 3
MAX_NFEV = 20
CONTINUATION_NFEV = 0
USE_MODE_CONTINUATION = False

METHOD = "scipy"
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = None
FTOL = 1.0e-4
GTOL = 1.0e-4
XTOL = 1.0e-4

# QP remains exploratory; bounded VMEC budgets avoid spending minutes on poor
# rejected trial points from the QH seed.
INNER_MAX_ITER = 80
INNER_FTOL = 1.0e-8
TRIAL_MAX_ITER = 80
TRIAL_FTOL = 1.0e-8
SOLVER_DEVICE = None

HELICITY_M = 0
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)
TARGET_ASPECT = 7.0
TARGET_ABS_IOTA_MIN = 0.31
TARGET_IOTA_DISPLAY = -TARGET_ABS_IOTA_MIN

ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 20.0
QS_WEIGHT = 1.0

USE_ESS = True
ALPHA = 2.5

OUTPUT_DIR = Path(f"results/qp_opt/n{HELICITY_N:+d}/mode{MAX_MODE}/{'ess' if USE_ESS else 'no_ess'}")

OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    abs_mean_iota_floor_objective(TARGET_ABS_IOTA_MIN, IOTA_WEIGHT),
    quasisymmetry_objective(
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
        surfaces=SURFACES,
        weight=QS_WEIGHT,
    ),
    # Example custom vector objective:
    # ObjectiveTerm("custom_vector", lambda ctx, state: your_vector(ctx, state), target=0.0, weight=0.1),
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
    target_iota=TARGET_IOTA_DISPLAY,
    label=f"QP opt (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
)


if __name__ == "__main__":
    run_qs_optimization(CONFIG, OBJECTIVES)

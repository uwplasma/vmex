#!/usr/bin/env python
"""Quasi-helical symmetry optimization with vmec_jax."""

from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Problem parameters.  Edit these directly for a different case.
INPUT_FILE = DATA_DIR / "input.nfp4_QH_warm_start"
OUTPUT_DIR = Path("results/qh_opt/ess")
MAX_MODE = 3
MIN_VMEC_MODE = 6
USE_MODE_CONTINUATION = True
MAX_NFEV = 30
CONTINUATION_NFEV = 30
STAGE_MODES = vj.qs_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
)

# Optimizer parameters.
METHOD = "scipy"
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = None
FTOL = 1.0e-4
GTOL = 1.0e-4
XTOL = 1.0e-4
INNER_MAX_ITER = 120
INNER_FTOL = 1.0e-9
TRIAL_MAX_ITER = 120
TRIAL_FTOL = 1.0e-9
SOLVER_DEVICE = None

# Physics targets and least-squares objective weights.  The iota term is a
# differentiable lower bound on abs(mean_iota), not a target.
TARGET_ASPECT = 5.0
TARGET_ABS_IOTA_MIN = 0.41
HELICITY_M = 1
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)
ASPECT_WEIGHT = 1.0
IOTA_FLOOR_WEIGHT = 40_000.0
QS_WEIGHT = 1.0

USE_ESS = True
ALPHA = 1.2
PLOT = True


vmec = vj.FixedBoundaryVMEC.from_input(
    INPUT_FILE,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR,
)

aspect = vj.AspectRatio()
iota_floor = vj.AbsMeanIotaFloor(TARGET_ABS_IOTA_MIN)
qs = vj.QuasisymmetryRatioResidual(
    helicity_m=HELICITY_M,
    helicity_n=HELICITY_N,
    surfaces=SURFACES,
)
problem = vj.LeastSquaresProblem.from_tuples(
    [
        (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
        (iota_floor.J, 0.0, IOTA_FLOOR_WEIGHT),
        (qs.J, 0.0, QS_WEIGHT),
        # Optional:
        # (vj.LgradB(threshold=0.30).J, 0.0, 0.01),
    ]
)

result = vj.least_squares_solve(
    vmec,
    problem,
    stage_modes=STAGE_MODES,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    method=METHOD,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    use_ess=USE_ESS,
    ess_alpha=ALPHA,
    label=f"QH optimization (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
    use_mode_continuation=USE_MODE_CONTINUATION,
    target_aspect=TARGET_ASPECT,
    iota_abs_min=TARGET_ABS_IOTA_MIN,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    plot=PLOT,
)

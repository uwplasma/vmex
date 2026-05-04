#!/usr/bin/env python
"""Quasi-axisymmetric optimization with vmec_jax."""

from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Problem parameters.  These are intended to be edited directly, as in the
# SIMSOPT examples.
INPUT_FILE = DATA_DIR / "input.nfp2_QA_omnigenity"
OUTPUT_DIR = Path("results/qa_opt/ess")
MAX_MODE = 3
MIN_VMEC_MODE = 6
USE_MODE_CONTINUATION = True
MAX_NFEV = 60
CONTINUATION_NFEV = 60
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
SOLVER_DEVICE = None  # set to "cpu" or "gpu" to force one backend

# Physics targets and least-squares objective weights.  These are SIMSOPT-style
# tuple weights, so vmec_jax minimizes sqrt(weight) * (J - target).
TARGET_ASPECT = 5.0
TARGET_IOTA = 0.42
HELICITY_M = 1
HELICITY_N = 0
SURFACES = np.arange(0.0, 1.01, 0.1)
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 10_000.0
QS_WEIGHT = 1.0

# ESS scales high-mode boundary variables.  Set USE_ESS=False for an unscaled
# trust-region solve.
USE_ESS = True
ALPHA = 1.2
PLOT = True


# Optimizable VMEC object.
vmec = vj.FixedBoundaryVMEC.from_input(
    INPUT_FILE,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR,
)


# Objective function.  Add new terms by appending another
# (objective.J, target, weight) tuple.
aspect = vj.AspectRatio()
iota = vj.MeanIota()
qs = vj.QuasisymmetryRatioResidual(
    helicity_m=HELICITY_M,
    helicity_n=HELICITY_N,
    surfaces=SURFACES,
)
problem = vj.LeastSquaresProblem.from_tuples(
    [
        (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
        (iota.J, TARGET_IOTA, IOTA_WEIGHT),
        (qs.J, 0.0, QS_WEIGHT),
        # Optional:
        # (vj.LgradB(threshold=0.30).J, 0.0, 0.01),
        # (vj.MagneticWell(minimum=0.0).J, 0.0, 1.0),
        # Finite-beta examples can also add:
        # (vj.VolavgB().J, TARGET_VOLAVGB, VOLAVGB_WEIGHT),
        # (vj.BetaTotal().J, TARGET_BETA, BETA_WEIGHT),
        # DMerc is currently a wout diagnostic/parity gate; a differentiable
        # DMerc objective should be added in vmec_jax before uncommenting it.
    ]
)


# Optimization.
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
    label=f"QA optimization (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
    use_mode_continuation=USE_MODE_CONTINUATION,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    plot=PLOT,
)

vj.print_optimization_outputs(result, OUTPUT_DIR, plot=PLOT)

#!/usr/bin/env python
"""Quasi-isodynamic optimization with vmec_jax and booz_xform_jax."""

from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# QI uses the nfp=2 warm start from the omnigenity optimization examples.  The
# default is the best current QI lane: direct max_mode=3 with ESS, a low aspect
# weight, and the branch-width term active in the smooth QI residual.
INPUT_FILE = DATA_DIR / "input.nfp2_QI"
OUTPUT_DIR = Path("results/qi_opt/ess")
MAX_MODE = 3
MIN_VMEC_MODE = 6
USE_MODE_CONTINUATION = False
MAX_NFEV = 50
CONTINUATION_NFEV = 50
STAGE_REPEATS = 1
STAGE_MODES = [MAX_MODE] * STAGE_REPEATS if USE_MODE_CONTINUATION and MAX_MODE > 1 else [MAX_MODE]

METHOD = "scipy"
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = None
FTOL = 1.0e-4
GTOL = 1.0e-4
XTOL = 1.0e-8
INNER_MAX_ITER = 120
INNER_FTOL = 1.0e-9
TRIAL_MAX_ITER = 120
TRIAL_FTOL = 1.0e-9
SOLVER_DEVICE = None

# Scalar and field-quality targets.  The mirror/elongation terms are soft
# upper-bound penalties; uncomment LgradB if needed for additional shaping.
TARGET_ASPECT = 3.5
TARGET_ABS_IOTA_MIN = 0.40
MAX_MIRROR_RATIO = 0.21
MAX_ELONGATION = 8.0
SURFACES = np.linspace(0.1, 1.0, 6)
ASPECT_WEIGHT = 0.005
IOTA_FLOOR_WEIGHT = 200.0**2
QI_WEIGHT = 1.0
MIRROR_WEIGHT = 10.0
ELONGATION_WEIGHT = 10.0

# Boozer transform and smooth-QI residual resolution.
QI_OPTIONS = vj.QuasiIsodynamicOptions(
    surfaces=SURFACES,
    mboz=18,
    nboz=18,
    nphi=151,
    nalpha=31,
    n_bounce=51,
    softness=2.0e-2,
    width_weight=1.0,
    branch_width_weight=0.5,
    branch_width_softness=2.0e-2,
    profile_weight=0.0,
    aligned_profile_weight=0.0,
    aligned_profile_softness=2.0e-2,
    aligned_profile_trap_level=0.65,
    aligned_profile_trap_softness=5.0e-2,
)

USE_ESS = True
ALPHA = 1.2
PLOT = True


vmec = vj.FixedBoundaryVMEC.from_input(
    INPUT_FILE,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR,
    project_input_boundary_to_max_mode=True,
)

aspect = vj.AspectRatio()
iota_floor = vj.AbsMeanIotaFloor(TARGET_ABS_IOTA_MIN)
qi = vj.QuasiIsodynamicResidual(QI_OPTIONS)
mirror = vj.MirrorRatio(threshold=MAX_MIRROR_RATIO, ntheta=96, nphi=96, surface_index=0)
elongation = vj.MaxElongation(threshold=MAX_ELONGATION, ntheta=48, nphi=16)
problem = vj.LeastSquaresProblem.from_tuples(
    [
        (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
        (iota_floor.J, 0.0, IOTA_FLOOR_WEIGHT),
        (qi.J, 0.0, QI_WEIGHT),
        (mirror.J, 0.0, MIRROR_WEIGHT),
        (elongation.J, 0.0, ELONGATION_WEIGHT),
        # Optional:
        # (vj.LgradB(threshold=0.30).J, 0.0, 0.001),
        # (vj.MagneticWell(minimum=0.0).J, 0.0, 1.0),
        # Finite-beta examples can also add:
        # (vj.VolavgB().J, TARGET_VOLAVGB, VOLAVGB_WEIGHT),
        # (vj.BetaTotal().J, TARGET_BETA, BETA_WEIGHT),
        # DMerc is currently a wout diagnostic/parity gate; a differentiable
        # DMerc objective should be added in vmec_jax before uncommenting it.
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
    label=f"QI optimization (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
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

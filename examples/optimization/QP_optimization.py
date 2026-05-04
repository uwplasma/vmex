#!/usr/bin/env python
"""Quasi-poloidal symmetry optimization with vmec_jax.

QP is still a quasisymmetry target: ``HELICITY_M = 0`` makes the |B| spectrum
depend primarily on the Boozer toroidal angle.
"""

from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64
from vmec_jax.config import config_from_indata
from vmec_jax.optimization_workflow import (
    ObjectiveTerm,  # noqa: F401 - useful for custom terms below.
    abs_mean_iota_floor_objective,
    aspect_objective,
    qs_stage_modes,
    quasisymmetry_objective,
    rebuild_for_optimization_resolution,
    run_fixed_boundary_objective_optimization,
)


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# User parameters
INPUT_FILE = DATA_DIR / "input.nfp2_QI"
OUTPUT_DIR = Path("results/qp_opt/no_ess")

MAX_MODE = 3
MIN_VMEC_MODE = 6
MAX_NFEV = 40
CONTINUATION_NFEV = 30
USE_MODE_CONTINUATION = True

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

HELICITY_M = 0
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)
TARGET_ASPECT = 5.0
TARGET_ABS_IOTA_MIN = 0.41

ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 200.0
QS_WEIGHT = 1.0

USE_ESS = False
ALPHA = 1.2
LABEL = f"QP optimization (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})"

SAVE_STAGE_INPUTS = True
SAVE_STAGE_WOUTS = False
SAVE_RERUN_WOUTS = False
PLOT = True


# Objective function
OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    abs_mean_iota_floor_objective(TARGET_ABS_IOTA_MIN, IOTA_WEIGHT),
    quasisymmetry_objective(
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
        surfaces=SURFACES,
        weight=QS_WEIGHT,
    ),
    # Optional examples:
    # lgradb_objective(threshold=0.30, weight=0.1)
    # ObjectiveTerm("custom", lambda ctx, state: your_metric(ctx, state), target=0.0, weight=1.0)
]


# Problem setup
print(f"Loading {INPUT_FILE.name} ...")
cfg, indata = vj.load_config(str(INPUT_FILE))
indata = rebuild_for_optimization_resolution(indata, max_mode=MAX_MODE, min_vmec_mode=MIN_VMEC_MODE)
cfg = config_from_indata(indata)
stage_modes = qs_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
)


# Optimization
run_fixed_boundary_objective_optimization(
    cfg=cfg,
    indata=indata,
    objectives=OBJECTIVES,
    stage_modes=stage_modes,
    max_mode=MAX_MODE,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    method=METHOD,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    use_ess=USE_ESS,
    ess_alpha=ALPHA,
    output_dir=OUTPUT_DIR,
    label=LABEL,
    use_mode_continuation=USE_MODE_CONTINUATION,
    target_aspect=TARGET_ASPECT,
    iota_abs_min=TARGET_ABS_IOTA_MIN,
    project_input_boundary_to_max_mode=True,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    save_stage_inputs=SAVE_STAGE_INPUTS,
    save_stage_wouts=SAVE_STAGE_WOUTS,
    save_rerun_wouts=SAVE_RERUN_WOUTS,
    plot=PLOT,
)

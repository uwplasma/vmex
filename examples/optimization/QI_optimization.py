#!/usr/bin/env python
"""Quasi-isodynamic optimization with vmec_jax and booz_xform_jax."""

from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64
from vmec_jax.config import config_from_indata
from vmec_jax.optimization_workflow import (
    abs_mean_iota_floor_objective,
    aspect_objective,
    qi_lgradb_objective,  # noqa: F401 - shown in the optional objective below.
    qi_max_elongation_objective,
    qi_mirror_ratio_objective,
    quasi_isodynamic_field_objective,
    rebuild_for_optimization_resolution,
    repeated_stage_modes,
    run_quasi_isodynamic_objective_optimization,
)


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# User parameters
INPUT_FILE = DATA_DIR / "input.nfp2_QI"
OUTPUT_DIR = Path("results/qi_opt/ess")

MAX_MODE = 3
MIN_VMEC_MODE = 6
MAX_NFEV = 30
CONTINUATION_NFEV = 30
USE_MODE_CONTINUATION = True
QI_STAGE_REPEATS = 5

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

SURFACES = np.linspace(0.1, 1.0, 6)
TARGET_ASPECT = 5.0
TARGET_ABS_IOTA_MIN = 0.41

ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 200.0
QI_WEIGHT = 1.0
MIRROR_WEIGHT = np.sqrt(10.0)
ELONGATION_WEIGHT = np.sqrt(10.0)

# Boozer transform and smooth-QI residual resolution.
QI_MBOZ = 18
QI_NBOZ = 18
QI_NPHI = 151
QI_NALPHA = 31
QI_N_BOUNCE = 51
QI_SOFTNESS = 2.0e-2
QI_WIDTH_WEIGHT = 1.0
QI_BRANCH_WIDTH_WEIGHT = 1.0
QI_BRANCH_WIDTH_SOFTNESS = 2.0e-2
QI_PROFILE_WEIGHT = 0.0
QI_ALIGNED_PROFILE_WEIGHT = 0.0
QI_ALIGNED_PROFILE_SOFTNESS = 2.0e-2
QI_ALIGNED_PROFILE_TRAP_LEVEL = 0.65
QI_ALIGNED_PROFILE_TRAP_SOFTNESS = 5.0e-2

MAX_MIRROR_RATIO = 0.21
MAX_ELONGATION = 8.0

USE_ESS = True
ALPHA = 1.2
LABEL = f"QI optimization (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})"

SAVE_STAGE_INPUTS = True
SAVE_STAGE_WOUTS = False
PLOT = True


# Scalar objectives
SCALAR_OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    abs_mean_iota_floor_objective(TARGET_ABS_IOTA_MIN, IOTA_WEIGHT),
]

# QI field objectives.  The runner evaluates Boozer |B| once per VMEC state and
# shares that field among these terms.
QI_OBJECTIVES = [
    quasi_isodynamic_field_objective(weight=QI_WEIGHT),
    qi_mirror_ratio_objective(
        threshold=MAX_MIRROR_RATIO,
        weight=MIRROR_WEIGHT,
        ntheta=96,
        nphi=96,
        surface_index=0,
    ),
    qi_max_elongation_objective(
        threshold=MAX_ELONGATION,
        weight=ELONGATION_WEIGHT,
        ntheta=48,
        nphi=16,
    ),
    # Optional LgradB penalty, matching the omnigenity examples:
    # qi_lgradb_objective(threshold=0.30, weight=np.sqrt(0.001), ntheta=9, nphi=7),
]


# Problem setup
print(f"Loading {INPUT_FILE.name} ...")
cfg, indata = vj.load_config(str(INPUT_FILE))
indata = rebuild_for_optimization_resolution(indata, max_mode=MAX_MODE, min_vmec_mode=MIN_VMEC_MODE)
cfg = config_from_indata(indata)
stage_modes = repeated_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
    repeats=QI_STAGE_REPEATS,
)


# Optimization
run_quasi_isodynamic_objective_optimization(
    cfg=cfg,
    indata=indata,
    scalar_objectives=SCALAR_OBJECTIVES,
    qi_objectives=QI_OBJECTIVES,
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
    surfaces=SURFACES,
    mboz=QI_MBOZ,
    nboz=QI_NBOZ,
    nphi=QI_NPHI,
    nalpha=QI_NALPHA,
    n_bounce=QI_N_BOUNCE,
    softness=QI_SOFTNESS,
    width_weight=QI_WIDTH_WEIGHT,
    branch_width_weight=QI_BRANCH_WIDTH_WEIGHT,
    branch_width_softness=QI_BRANCH_WIDTH_SOFTNESS,
    profile_weight=QI_PROFILE_WEIGHT,
    aligned_profile_weight=QI_ALIGNED_PROFILE_WEIGHT,
    aligned_profile_softness=QI_ALIGNED_PROFILE_SOFTNESS,
    aligned_profile_trap_level=QI_ALIGNED_PROFILE_TRAP_LEVEL,
    aligned_profile_trap_softness=QI_ALIGNED_PROFILE_TRAP_SOFTNESS,
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
    plot=PLOT,
)

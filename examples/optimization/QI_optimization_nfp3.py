#!/usr/bin/env python
"""NFP=3 quasi-isodynamic optimization: circular seed -> QP -> QI."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Edit these ordinary variables for related NFP=3 experiments.
INPUT_FILE = DATA_DIR / "input.minimal_seed_nfp3"
OUTPUT_DIR = Path("results/qi_opt/simple_qp_then_qi/nfp3")
MAX_MODE = 4
MIN_VMEC_MODE = max(6, MAX_MODE + 3)
USE_SIMPLE_SEED = True
SIMPLE_SEED_PERTURBATION = 1.0e-5

# Optimizer controls.  These are intentionally the same kind of controls used
# in QA_optimization.py, QH_optimization.py, and QP_optimization.py.
QP_METHOD = "scipy"  # Try "auto", "gauss_newton", "scipy_matrix_free", or "scalar_trust".
QI_METHOD = "auto_scalar"  # Try "auto", "scipy", "lbfgs_adjoint", or "scalar_trust".
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = 4
FTOL = 1.0e-5
GTOL = 1.0e-5
XTOL = 1.0e-6
INNER_MAX_ITER = 450
INNER_FTOL = 1.0e-9
TRIAL_MAX_ITER = 450
TRIAL_FTOL = 1.0e-9
SOLVER_DEVICE = None  # None uses the JAX default; set "cpu" or "gpu" to force one backend.
USE_ESS = True
ALPHA = 1.2
QP_MAX_NFEV = 65
QI_MAX_NFEV = 80
MAKE_PLOTS = True

# Physics targets and weights.
TARGET_ASPECT = 6.0
TARGET_ABS_IOTA_MIN = 0.41
HELICITY_M = 0
HELICITY_N = -1
SURFACES = np.linspace(0.1, 1.0, 6)
ASPECT_WEIGHT = 0.25
IOTA_FLOOR_WEIGHT = 200.0**2
QP_WEIGHT = 1.0
QI_WEIGHT = 10.0
MAX_MIRROR_RATIO = 0.35
MAX_ELONGATION = 10.0
MIRROR_WEIGHT = 20.0
ELONGATION_WEIGHT = 10.0

# QI Boozer sampling.  Increase these for a final publication run.
QI_OPTIONS = vj.QuasiIsodynamicOptions(
    surfaces=SURFACES,
    mboz=18,
    nboz=18,
    nphi=151,
    nalpha=31,
    n_bounce=51,
    include_bounce_endpoints=True,
    softness=2.0e-2,
    width_weight=1.0,
    branch_width_weight=0.5,
    branch_width_softness=2.0e-2,
    profile_weight=0.1,
    shuffle_profile_weight=1.0,
    shuffle_profile_softness=2.0e-2,
    phimin=0.0,
    jit_booz=True,
)

# The raw input remains circular/minimal.  This generated deck only adds tiny
# active-mode perturbations so the optimizer does not start with zero columns.
SEED_INPUT = vj.prepare_simple_omnigenity_seed_input(
    INPUT_FILE,
    OUTPUT_DIR,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    enabled=USE_SIMPLE_SEED,
    perturbation=SIMPLE_SEED_PERTURBATION,
)

# Stage 1: make a quasi-poloidally symmetric basin from the circular seed.
qp_vmec = vj.FixedBoundaryVMEC.from_input(
    SEED_INPUT,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR / "qp_stage",
    project_input_boundary_to_max_mode=True,
)
aspect = vj.AspectRatio()
iota_floor = vj.AbsMeanIotaFloor(TARGET_ABS_IOTA_MIN)
qp = vj.QuasisymmetryRatioResidual(
    helicity_m=HELICITY_M,
    helicity_n=HELICITY_N,
    surfaces=SURFACES,
)
mirror = vj.VMECMirrorRatio(
    threshold=MAX_MIRROR_RATIO,
    surfaces=SURFACES,
    smooth_extrema=2.0e-2,
    smooth_penalty=2.0e-2,
)
elongation = vj.MaxElongation(
    threshold=MAX_ELONGATION,
    ntheta=48,
    nphi=16,
    smooth_extrema=2.0e-2,
    smooth_penalty=2.0e-2,
)
qp_objective_tuples = [
    (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
    (iota_floor.J, 0.0, IOTA_FLOOR_WEIGHT),
    (qp.J, 0.0, QP_WEIGHT),
    (mirror.J, 0.0, MIRROR_WEIGHT),
    (elongation.J, 0.0, ELONGATION_WEIGHT),
]
qp_problem = vj.LeastSquaresProblem.from_tuples(qp_objective_tuples)

print("\nStage 1: QP basin from circular/minimal seed")
print(f"  input:      {INPUT_FILE}")
print(f"  max_mode:   {MAX_MODE}")
print(f"  objectives: {', '.join(qp_problem.objective_names)}")
qp_result = vj.least_squares_solve(
    qp_vmec,
    qp_problem,
    stage_modes=[MAX_MODE],
    max_nfev=QP_MAX_NFEV,
    continuation_nfev=0,
    method=QP_METHOD,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    use_ess=USE_ESS,
    ess_alpha=ALPHA,
    label=f"NFP=3 QP pre-stage (max_mode={MAX_MODE})",
    use_mode_continuation=False,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    save_stage_inputs=True,
    save_stage_wouts=False,
    save_final_outputs=False,
)
qp_paths = vj.save_optimization_result(qp_result, output_dir=OUTPUT_DIR / "qp_stage")

# Stage 2: replace the QP residual with a QI residual and polish from the QP
# output.  This is the only handoff; no reference-family or global preconditioner
# is used.
qi_vmec = vj.FixedBoundaryVMEC.from_input(
    qp_paths.final_input,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR / "qi_stage",
    project_input_boundary_to_max_mode=True,
)
qi = vj.QuasiIsodynamicResidual(QI_OPTIONS)
qi_mirror = vj.MirrorRatio(
    threshold=MAX_MIRROR_RATIO,
    surfaces=QI_OPTIONS.surfaces,
    mboz=QI_OPTIONS.mboz,
    nboz=QI_OPTIONS.nboz,
    smooth_extrema=2.0e-2,
    smooth_penalty=2.0e-2,
    qi_options=QI_OPTIONS,
)
qi_objective_tuples = [
    (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
    (iota_floor.J, 0.0, IOTA_FLOOR_WEIGHT),
    (qi.J, 0.0, QI_WEIGHT),
    (qi_mirror.J, 0.0, MIRROR_WEIGHT),
    (elongation.J, 0.0, ELONGATION_WEIGHT),
]
qi_problem = vj.LeastSquaresProblem.from_tuples(qi_objective_tuples)

print("\nStage 2: QI polish from QP output")
print(f"  objectives: {', '.join(qi_problem.objective_names)}")
qi_result = vj.least_squares_solve(
    qi_vmec,
    qi_problem,
    stage_modes=[MAX_MODE],
    max_nfev=QI_MAX_NFEV,
    continuation_nfev=0,
    method=QI_METHOD,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    use_ess=USE_ESS,
    ess_alpha=ALPHA,
    label=f"NFP=3 QI polish (max_mode={MAX_MODE})",
    use_mode_continuation=False,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    save_stage_inputs=True,
    save_stage_wouts=False,
    save_final_outputs=False,
)
qi_paths = vj.save_optimization_result(qi_result, output_dir=OUTPUT_DIR / "qi_stage")

history = qi_result.history
timing = qi_result.timing_summary
print("\nFinal QI diagnostics from result.history:")
print(f"  aspect ratio:     {history['aspect_final']:.6g}")
print(f"  mean iota:        {history['iota_final']:.6g}")
print(f"  QI objective:     {history['qs_final']:.6e}")
print(f"  total objective:  {history['objective_final']:.6e}")
print(f"  wall time:        {timing['total_wall_time_s']:.2f} s")
print(f"  QP final input:   {qp_paths.final_input}")
print(f"  QI final input:   {qi_paths.final_input}")
print(f"  QI final WOUT:    {qi_paths.final_wout}")

if MAKE_PLOTS:
    plot_paths = {
        "seed_to_final_3d_boundary": vj.plot_3d_boundary_comparison(
            qp_paths.initial_wout,
            qi_paths.final_wout,
            outdir=OUTPUT_DIR,
        ),
        "seed_to_final_lcfs_boozer_bmag": vj.plot_boozer_lcfs_bmag_comparison(
            qp_paths.initial_wout,
            qi_paths.final_wout,
            outdir=OUTPUT_DIR,
        ),
        "qp_objective_history": vj.plot_objective_history(
            qp_paths.history,
            outdir=OUTPUT_DIR / "qp_stage",
        ),
        "qi_objective_history": vj.plot_objective_history(
            qi_paths.history,
            outdir=OUTPUT_DIR / "qi_stage",
        ),
    }
    print("\nPlot files:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

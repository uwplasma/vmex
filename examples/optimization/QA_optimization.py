#!/usr/bin/env python
"""Quasi-axisymmetric optimization with vmec_jax."""

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

# Problem parameters.  These are intended to be edited directly, as in the
# SIMSOPT examples.
WARM_START_INPUT_FILE = DATA_DIR / "input.nfp2_QA_omnigenity"
SIMPLE_SEED_INPUT_FILE = DATA_DIR / "input.minimal_seed_nfp2"
OUTPUT_DIR = Path("results/qa_opt/ess")
MAX_MODE = 3
MIN_VMEC_MODE = 6
USE_SIMPLE_SEED = True  # Start from near-circular RBC(0,0), RBC(0,1), ZBS(0,1).
INPUT_FILE = SIMPLE_SEED_INPUT_FILE if USE_SIMPLE_SEED else WARM_START_INPUT_FILE
INPUT_FILE = vj.prepare_simple_omnigenity_seed_input(
    INPUT_FILE,
    OUTPUT_DIR,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    enabled=USE_SIMPLE_SEED,
)
USE_MODE_CONTINUATION = not USE_SIMPLE_SEED
MAX_NFEV = 60
CONTINUATION_NFEV = 60
STAGE_MODES = vj.qs_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
)

# Optimizer parameters.
METHOD = "scipy"  # Try also "auto", "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
SCIPY_TR_SOLVER = "lsmr"  # For METHOD="scipy": "lsmr" is memory-light; "exact" is dense.
SCIPY_LSMR_MAXITER = None  # None lets SciPy choose; set an int to cap LSMR iterations.
FTOL = 1.0e-4  # Relative cost-reduction tolerance for the outer optimizer.
GTOL = 1.0e-4  # Gradient optimality tolerance for the outer optimizer.
XTOL = 1.0e-4  # Step-size tolerance for the outer optimizer.
INNER_MAX_ITER = 120  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
INNER_FTOL = 1.0e-9  # Accepted-point VMEC tolerance; 0 uses FTOL from the input deck.
TRIAL_MAX_ITER = 120  # Trial-point VMEC iterations; 0 follows the accepted/input budget.
TRIAL_FTOL = 1.0e-9  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.
SOLVER_DEVICE = None  # None uses JAX default; set "cpu" or "gpu" to force one backend.
EXACT_PATH = None  # None/"tape" favors cold starts; use "scan" only for long warm GPU runs.
USE_ESS = True  # Set False for an unscaled trust-region solve.
ALPHA = 1.2  # ESS high-mode scaling strength.
# Common alternatives:
# METHOD = "gauss_newton"
# METHOD = "lbfgs_adjoint"
# USE_SIMPLE_SEED = False
# USE_MODE_CONTINUATION = False
# STAGE_MODES = [MAX_MODE]
# USE_ESS = False

# Output controls.
SAVE_STAGE_INPUTS = True  # Keep per-stage input decks for continuation/debugging.
SAVE_STAGE_WOUTS = False  # Set True to also write per-stage WOUT files.
MAKE_PLOTS = True

# Physics targets and least-squares objective weights.  These are SIMSOPT-style
# tuple weights, so vmec_jax minimizes sqrt(weight) * (J - target).
TARGET_ASPECT = 6.0
TARGET_IOTA = 0.42
HELICITY_M = 1
HELICITY_N = 0
SURFACES = np.arange(0.0, 1.01, 0.1)
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 10_000.0
QS_WEIGHT = 1.0


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
objective_tuples = [
    (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
    (iota.J, TARGET_IOTA, IOTA_WEIGHT),
    (qs.J, 0.0, QS_WEIGHT),
    # Optional:
    # (vj.LgradB(threshold=0.30, smooth_penalty=1.0e-3).J, 0.0, 0.01),
    # (vj.MagneticWell(minimum=0.0).J, 0.0, 1.0),
    # Finite-beta examples can also add:
    # (vj.VolavgB().J, TARGET_VOLAVGB, VOLAVGB_WEIGHT),
    # (vj.BetaTotal().J, TARGET_BETA, BETA_WEIGHT),
    # (vj.DMerc(minimum=0.0, softness=1.0e-3).J, 0.0, DMERC_WEIGHT),
    # (vj.JDotB(surfaces=(0.25, 0.50, 0.75)).J, 0.0, JDOTB_WEIGHT),
    # (vj.BDotB(surfaces=(0.25, 0.50, 0.75)).J, TARGET_BDOTB, BDOTB_WEIGHT),
    # (vj.BDotGradV(surfaces=(0.25, 0.50, 0.75)).J, TARGET_BDOTGRADV, BDOTGRADV_WEIGHT),
    # (vj.ToroidalCurrent(surfaces=(0.25, 0.50, 0.75)).J, TARGET_TORCUR, TORCUR_WEIGHT),
    # (vj.ToroidalCurrentGradient(surfaces=(0.25, 0.50, 0.75)).J, TARGET_TORCUR_PRIME, TORCUR_PRIME_WEIGHT),
    # (vj.RedlBootstrapMismatch(helicity_n=HELICITY_N, ne_coeffs=NE_COEFFS, Te_coeffs=TE_COEFFS, surfaces=(0.25, 0.50, 0.75)).J, 0.0, BOOTSTRAP_WEIGHT),
    # (vj.BVector(s_index=-1).J, TARGET_B_VECTOR, B_VECTOR_WEIGHT),
    # (vj.JVector(surfaces=(0.25, 0.50, 0.75)).J, TARGET_J_VECTOR, J_VECTOR_WEIGHT),
]
problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)


# Optimization.
# The solve call only receives optimizer, continuation, device, and output
# controls.  Physics targets stay in objective_tuples above.
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
    exact_path=EXACT_PATH,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    save_stage_inputs=SAVE_STAGE_INPUTS,
    save_stage_wouts=SAVE_STAGE_WOUTS,
    save_final_outputs=False,
)

# Results are plain Python objects.  The call below only saves the standard
# artifacts; diagnostics and plots remain explicit in this script.
history = result.history
objective_history = result.objective_history
timing = result.timing_summary

saved_paths = vj.save_optimization_result(result, output_dir=OUTPUT_DIR)

print("\nFinal diagnostics from result.history:")
print(f"  aspect ratio:     {history['aspect_final']:.6g}")
print(f"  mean iota:        {history['iota_final']:.6g}")
print(f"  QS objective:     {history['qs_final']:.6e}")
print(f"  total objective:  {history['objective_final']:.6e}")
print(f"  wall time:        {timing['total_wall_time_s']:.2f} s")
print(f"  objective samples: {objective_history[:5]} ... {objective_history[-3:]}")

print("\nFiles saved from result objects:")
for name, path in saved_paths.as_dict().items():
    print(f"  {name}: {path}")

wout_final = vj.load_wout(saved_paths.final_wout)
theta, zeta, b_lcfs = vj.vmecplot2_bmag_grid(
    wout_final,
    s_index=-1,
    ntheta=64,
    nzeta=64,
    zeta_max=2.0 * np.pi / float(wout_final.nfp),
)
print("\nLCFS |B| data from vmecplot2_bmag_grid:")
print(f"  theta grid: {theta.shape}, zeta grid: {zeta.shape}, B grid: {b_lcfs.shape}")
print(f"  Bmin/Bmax:  {np.min(b_lcfs):.6g} / {np.max(b_lcfs):.6g}")

if MAKE_PLOTS:
    # Plotting is a normal post-processing block; add or remove entries here
    # instead of relying on hidden plotting side effects from the solve.
    plot_paths = {
        "boundary_comparison": vj.plot_3d_boundary_comparison(
            saved_paths.initial_wout,
            saved_paths.final_wout,
            outdir=OUTPUT_DIR,
        ),
        "bmag_contours": vj.plot_bmag_contours(
            saved_paths.initial_wout,
            saved_paths.final_wout,
            outdir=OUTPUT_DIR,
        ),
        "objective_history": vj.plot_objective_history(
            saved_paths.history,
            outdir=OUTPUT_DIR,
        ),
    }
    print("\nPlot files selected by this script:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

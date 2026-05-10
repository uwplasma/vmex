#!/usr/bin/env python
"""Quasi-poloidal symmetry optimization with vmec_jax."""

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

# QP is a quasisymmetry target with HELICITY_M=0.  We use the nfp=2 QI seed so
# QP and QI start from the same boundary family.
INPUT_FILE = DATA_DIR / "input.nfp2_QI"
OUTPUT_DIR = Path("results/qp_opt/no_ess")
MAX_MODE = 3
MIN_VMEC_MODE = 6
USE_MODE_CONTINUATION = True
MAX_NFEV = 40
CONTINUATION_NFEV = 30
STAGE_MODES = vj.qs_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
)

METHOD = "scipy"  # Try also "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
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

TARGET_ASPECT = 5.0
TARGET_ABS_IOTA_MIN = 0.41
HELICITY_M = 0
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)
ASPECT_WEIGHT = 1.0
IOTA_FLOOR_WEIGHT = 40_000.0
QS_WEIGHT = 1.0

USE_ESS = False
ALPHA = 1.2
MAKE_PLOTS = True


vmec = vj.FixedBoundaryVMEC.from_input(
    INPUT_FILE,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR,
    project_input_boundary_to_max_mode=True,
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
        # (vj.LgradB(threshold=0.30, smooth_penalty=1.0e-3).J, 0.0, 0.01),
        # (vj.MagneticWell(minimum=0.0).J, 0.0, 1.0),
        # Finite-beta examples can also add:
        # (vj.VolavgB().J, TARGET_VOLAVGB, VOLAVGB_WEIGHT),
        # (vj.BetaTotal().J, TARGET_BETA, BETA_WEIGHT),
        # (vj.DMerc(minimum=0.0, softness=1.0e-3).J, 0.0, DMERC_WEIGHT),
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
    label=f"QP optimization (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
    use_mode_continuation=USE_MODE_CONTINUATION,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
)

history = result.final_result["_history_dump"]
objective_history = np.asarray([entry["objective"] for entry in history["history"]])
print("\nFinal diagnostics from result.final_result['_history_dump']:")
print(f"  aspect ratio:     {history['aspect_final']:.6g}")
print(f"  mean iota:        {history['iota_final']:.6g}")
print(f"  QS objective:     {history['qs_final']:.6e}")
print(f"  total objective:  {history['objective_final']:.6e}")
print(f"  wall time:        {history['total_wall_time_s']:.2f} s")
print(f"  objective samples: {objective_history[:5]} ... {objective_history[-3:]}")

print("\nSaved files written by the solve:")
for path in (
    OUTPUT_DIR / "input.initial",
    OUTPUT_DIR / "input.final",
    OUTPUT_DIR / "wout_initial.nc",
    OUTPUT_DIR / "wout_final.nc",
    OUTPUT_DIR / "history.json",
):
    print(f"  {path}")

wout_final = vj.load_wout(OUTPUT_DIR / "wout_final.nc")
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
    plot_paths = {
        "boundary_comparison": vj.plot_3d_boundary_comparison(
            OUTPUT_DIR / "wout_initial.nc",
            OUTPUT_DIR / "wout_final.nc",
            outdir=OUTPUT_DIR,
        ),
        "bmag_contours": vj.plot_bmag_contours(
            OUTPUT_DIR / "wout_initial.nc",
            OUTPUT_DIR / "wout_final.nc",
            outdir=OUTPUT_DIR,
        ),
        "objective_history": vj.plot_objective_history(
            OUTPUT_DIR / "history.json",
            outdir=OUTPUT_DIR,
        ),
    }
    print("\nPlot files selected by this script:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

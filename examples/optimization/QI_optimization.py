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

METHOD = "scipy"  # Try also "gauss_newton", "scipy_matrix_free", or "lbfgs_adjoint".
SCIPY_TR_SOLVER = "lsmr"  # For METHOD="scipy": "lsmr" is memory-light; "exact" is dense.
SCIPY_LSMR_MAXITER = None  # None lets SciPy choose; set an int to cap LSMR iterations.
FTOL = 1.0e-4  # Relative cost-reduction tolerance for the outer optimizer.
GTOL = 1.0e-4  # Gradient optimality tolerance for the outer optimizer.
XTOL = 1.0e-8  # Step-size tolerance; QI often benefits from a tighter value.
INNER_MAX_ITER = 120  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
INNER_FTOL = 1.0e-9  # Accepted-point VMEC tolerance; 0 uses FTOL from the input deck.
TRIAL_MAX_ITER = 120  # Trial-point VMEC iterations; 0 follows the accepted/input budget.
TRIAL_FTOL = 1.0e-9  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.
SOLVER_DEVICE = None  # None uses JAX default; set "cpu" or "gpu" to force one backend.

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
    phimin=0.0,  # Set to np.pi / nfp if auditing a seed whose well starts there.
)

USE_ESS = True
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
        # (vj.LgradB(threshold=0.30, smooth_penalty=1.0e-3).J, 0.0, 0.001),
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
)

history = result.final_result["_history_dump"]
objective_history = np.asarray([entry["objective"] for entry in history["history"]])
print("\nFinal diagnostics from result.final_result['_history_dump']:")
print(f"  aspect ratio:     {history['aspect_final']:.6g}")
print(f"  mean iota:        {history['iota_final']:.6g}")
print(f"  QI objective:     {history['qs_final']:.6e}")
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

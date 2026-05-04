#!/usr/bin/env python
"""Quasi-poloidal symmetry optimization with vmec_jax."""

from pathlib import Path

import numpy as np

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
        # (vj.LgradB(threshold=0.30).J, 0.0, 0.01),
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
    plot_paths = vj.plot_qh_optimization(
        OUTPUT_DIR / "wout_initial.nc",
        OUTPUT_DIR / "wout_final.nc",
        OUTPUT_DIR / "history.json",
        outdir=OUTPUT_DIR,
    )
    print("\nPlot files selected by this script:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

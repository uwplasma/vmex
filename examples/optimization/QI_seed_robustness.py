#!/usr/bin/env python
"""QI seed-robustness probe from a near-axis stellarator seed.

This script is intentionally explicit: the user chooses the seed, constructs
objective tuples, runs the optimizer, then saves/prints/plots the result.
"""

from pathlib import Path
import os
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


# Seed and optimizer settings.  This default is the first short policy that
# gets the bundled seed onto a low-QI, nonzero-iota branch.  For a quick probe
# of another VMEC input deck, set VMEC_JAX_QI_SEED_INPUT and optionally
# VMEC_JAX_QI_SEED_OUTPUT_DIR; audit solved input/wout pairs first with
# audit_qi_seed_suitability.py because this script launches an optimization.
INPUT_FILE = _env_path("VMEC_JAX_QI_SEED_INPUT", DATA_DIR / "input.QI_stel_seed_3127")
OUTPUT_DIR = _env_path(
    "VMEC_JAX_QI_SEED_OUTPUT_DIR",
    Path("results/qi_seed_robustness/qi_stel_seed_3127/qiiota_aspect_mode3"),
)
MAX_MODE = 3
MIN_VMEC_MODE = 6
MAX_NFEV = _env_int("VMEC_JAX_QI_SEED_MAX_NFEV", 8)
METHOD = "scipy"  # Try "scalar_trust" for stricter monotone line-search probes.
SCIPY_TR_SOLVER = "lsmr"  # For METHOD="scipy": "lsmr" is memory-light; "exact" is dense.
FTOL = 1.0e-4
GTOL = 1.0e-4
XTOL = 1.0e-8
INNER_MAX_ITER = 120
INNER_FTOL = 1.0e-9
TRIAL_MAX_ITER = 120
TRIAL_FTOL = 1.0e-9
SOLVER_DEVICE = None  # Set "cpu" or "gpu" to force one backend.
USE_ESS = True
ALPHA = 1.2
MAKE_PLOTS = True
TARGET_ASPECT = 10.0
TARGET_ABS_IOTA_MIN = 0.41
MAX_MIRROR_RATIO = 0.21
MAX_ELONGATION = 8.0
ASPECT_WEIGHT = 0.25
IOTA_FLOOR_WEIGHT = 200.0**2
QI_WEIGHT = 10.0
MIRROR_WEIGHT = 10.0
ELONGATION_WEIGHT = 10.0
MIRROR_SMOOTH_EXTREMA = 2.0e-2
MIRROR_SMOOTH_PENALTY = 2.0e-2
QI_GATE_SMOOTH_MAX = 2.0e-3
QI_GATE_LEGACY_MAX = 2.0e-3

# QI residual settings.  These are the low-mode settings used for the bounded
# seed probe; increase mboz/nboz/nphi/nalpha/n_bounce for final publication runs.
surfaces = np.linspace(0.1, 1.0, 6)
qi_options = vj.QuasiIsodynamicOptions(
    surfaces=surfaces,
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
)

# Robust QI candidate objective.  QI alone can find a low-QI branch with
# near-zero transform, which is not an acceptable stellarator target.
aspect = vj.AspectRatio()
iota_floor = vj.AbsMeanIotaFloor(TARGET_ABS_IOTA_MIN)
qi = vj.QuasiIsodynamicResidual(qi_options)
objective_tuples = [
    (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
    (iota_floor.J, 0.0, IOTA_FLOOR_WEIGHT),
    (qi.J, 0.0, QI_WEIGHT),
]

print("\nQI seed robustness policy:")
print(f"  input file:      {INPUT_FILE}")
print(f"  output dir:      {OUTPUT_DIR}")
print(f"  max_mode:        {MAX_MODE}")
print(f"  min_vmec_mode:   {MIN_VMEC_MODE}")
print(f"  max_nfev:        {MAX_NFEV}")
print(f"  target aspect:   {TARGET_ASPECT}")
print(f"  abs iota floor:  {TARGET_ABS_IOTA_MIN}")
print(f"  mirror target:   {MAX_MIRROR_RATIO}")

# Optional engineering cleanup.  Mirror/elongation can be included after the
# QI+iota branch is established, but too much scalar pressure can destroy QI.
# mirror = vj.MirrorRatio(
#     threshold=MAX_MIRROR_RATIO,
#     ntheta=96,
#     nphi=96,
#     surfaces=qi_options.surfaces,
#     mboz=qi_options.mboz,
#     nboz=qi_options.nboz,
#     surface_index=None,  # all selected surfaces, matching the diagnostic gate
#     smooth_extrema=MIRROR_SMOOTH_EXTREMA,
#     smooth_penalty=MIRROR_SMOOTH_PENALTY,
# )
# elongation = vj.MaxElongation(threshold=MAX_ELONGATION, ntheta=48, nphi=16)
# qi_ceiling = vj.QuasiIsodynamicResidualCeiling(maximum=2.0e-2, smooth_penalty=2.0e-3, qi_options=qi_options)
# objective_tuples += [
#     (qi_ceiling.J, 0.0, 100.0),
#     (mirror.J, 0.0, MIRROR_WEIGHT),
#     (elongation.J, 0.0, ELONGATION_WEIGHT),
# ]

problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)
vmec = vj.FixedBoundaryVMEC.from_input(
    INPUT_FILE,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR,
    project_input_boundary_to_max_mode=True,
)

result = vj.least_squares_solve(
    vmec,
    problem,
    stage_modes=[MAX_MODE],
    max_nfev=MAX_NFEV,
    continuation_nfev=0,
    method=METHOD,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    use_ess=USE_ESS,
    ess_alpha=ALPHA,
    label=f"QI seed robustness ({INPUT_FILE.name}, max_mode={MAX_MODE})",
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    save_final_outputs=False,
)

history = result.history
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
paths = {
    "initial_input": OUTPUT_DIR / "input.initial",
    "final_input": OUTPUT_DIR / "input.final",
    "initial_wout": OUTPUT_DIR / "wout_initial.nc",
    "final_wout": OUTPUT_DIR / "wout_final.nc",
    "history": OUTPUT_DIR / "history.json",
}
result.initial_optimizer.save_input(paths["initial_input"], result.initial_params)
result.initial_optimizer.save_wout(
    paths["initial_wout"],
    result.initial_params,
    state=result.initial_state,
)
result.final_optimizer.save_input(paths["final_input"], result.final_params)
result.final_optimizer.save_wout(
    paths["final_wout"],
    result.final_params,
    state=result.final_state,
)
result.final_optimizer.save_history(paths["history"], result.final_result)

print("\nQI seed robustness result:")
print(f"  optimizer success: {result.final_result['success']}")
print(f"  optimizer message: {result.final_result['message']}")
print(f"  initial objective: {history['objective_initial']:.6e}")
print(f"  final objective:   {history['objective_final']:.6e}")
print(f"  final aspect:      {history['aspect_final']:.6g}")
if "iota_final" in history:
    print(f"  final mean iota:   {history['iota_final']:.6g}")
print(f"  wall time:         {result.timing_summary['total_wall_time_s']:.2f} s")

diagnostic_options = vj.QIDiagnosticOptions(
    surfaces=surfaces,
    mboz=18,
    nboz=18,
    nphi=151,
    nalpha=31,
    n_bounce=51,
    include_bounce_endpoints=True,
    phimin=0.0,
    mirror_threshold=MAX_MIRROR_RATIO,
    elongation_threshold=MAX_ELONGATION,
)
diagnostics = vj.qi_diagnostics_from_state(
    state=result.final_state,
    static=result.final_optimizer.static,
    indata=result.final_optimizer.indata,
    signgs=result.final_optimizer.signgs,
    surfaces=surfaces,
    options=diagnostic_options,
)
smooth_qi = float(diagnostics["qi_smooth_total"])
legacy_qi = float(diagnostics["qi_legacy_total"])
abs_iota = abs(float(history.get("iota_final", 0.0)))
mirror_ratio = float(diagnostics["qi_mirror_ratio_max"])
max_elongation = float(diagnostics["qi_max_elongation"])
qi_gate_passed = (
    smooth_qi <= QI_GATE_SMOOTH_MAX
    and legacy_qi <= QI_GATE_LEGACY_MAX
    and abs_iota >= TARGET_ABS_IOTA_MIN
)
engineering_gate_passed = (
    qi_gate_passed
    and mirror_ratio <= MAX_MIRROR_RATIO
    and max_elongation <= MAX_ELONGATION
)
print("\nIndependent QI promotion gate:")
print(f"  smooth QI:       {smooth_qi:.6e}  (limit {QI_GATE_SMOOTH_MAX:.1e})")
print(f"  legacy QI:       {legacy_qi:.6e}  (limit {QI_GATE_LEGACY_MAX:.1e})")
print(f"  abs(mean iota):  {abs_iota:.6g}  (minimum {TARGET_ABS_IOTA_MIN:.3g})")
print(f"  mirror ratio:    {mirror_ratio:.6g}  (target {MAX_MIRROR_RATIO:.3g})")
print(f"  mirror by surf:  {diagnostics.get('qi_mirror_ratio_by_surface')}")
print(f"  max elongation:  {max_elongation:.6g}  (target {MAX_ELONGATION:.3g})")
print(f"  QI+iota gate:    {qi_gate_passed}")
print(f"  full eng. gate:  {engineering_gate_passed}")
print("\nSaved files:")
for name, path in paths.items():
    print(f"  {name}: {path}")

if MAKE_PLOTS:
    plot_paths = {
        "boundary_comparison": vj.plot_3d_boundary_comparison(
            paths["initial_wout"],
            paths["final_wout"],
            outdir=OUTPUT_DIR,
        ),
        "bmag_contours": vj.plot_bmag_contours(
            paths["initial_wout"],
            paths["final_wout"],
            outdir=OUTPUT_DIR,
        ),
        "objective_history": vj.plot_objective_history(
            paths["history"],
            outdir=OUTPUT_DIR,
        ),
        "boozer_bmag_contours": vj.plot_boozer_bmag_contours_from_state(
            result.final_state,
            static=result.final_optimizer.static,
            indata=result.final_optimizer.indata,
            signgs=result.final_optimizer.signgs,
            outdir=OUTPUT_DIR,
            surfaces=(1.0,),
            mboz=18,
            nboz=18,
            title=f"{INPUT_FILE.name}: Boozer |B| contours on LCFS",
        ),
    }
    print("\nPlot files:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

#!/usr/bin/env python
"""Quasi-isodynamic optimization with vmec_jax and booz_xform_jax."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64
from vmec_jax.qi_diagnostics import QISeedSuitabilityTargets, annotate_qi_seed_suitability


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _diagnostic_float(record, key):
    value = record.get(key)
    return float(value) if value is not None else float("nan")

# Seed cases.  Pick one case by changing RUN_CASE; add another dictionary entry
# to use any external VMEC input deck.  The NFP is taken from the VMEC input
# file, so the same script can run the bundled NFP=2 QI seed, the stellarator
# seed, or the NFP=4 QH warm start without hard-coding field period count.
QI_CASES = {
    "nfp2_qi": {
        "case_goal": "default NFP=2 mirror-aware QI lane",
        "input_file": DATA_DIR / "input.nfp2_QI",
        "output_dir": Path("results/qi_opt/ess/nfp2_qi"),
        "max_mode": 3,
        "min_vmec_mode": 6,
        "use_mode_continuation": True,
        "stage_repeats": 5,
        "max_nfev": 12,
        "target_aspect": 5.0,
        "target_abs_iota_min": 0.41,
        "mirror_threshold": 0.21,
        "mirror_surface_index": None,
        "qi_ceiling_max": 2.0e-2,
        "qi_ceiling_smooth_penalty": 2.0e-3,
        "branch_width_weight": 5.0,
        "weighted_shuffle_profile_weight": 0.0,
        "phimin": 0.0,
        "mirror_weight": 10.0,
        "elongation_weight": 10.0,
        "qi_ceiling_weight": 100.0,
        "shuffle_profile_nphi_out": None,
    },
    "qi_stel_seed_3127": {
        "case_goal": "far-seed QI+iota robustness lane; low-mirror cleanup remains a gated follow-up",
        "input_file": DATA_DIR / "input.QI_stel_seed_3127",
        "output_dir": Path("results/qi_opt/ess/qi_stel_seed_3127"),
        "max_mode": 3,
        "min_vmec_mode": 6,
        "use_mode_continuation": False,
        "stage_repeats": 1,
        "max_nfev": 8,
        "target_aspect": 5.0,
        "target_abs_iota_min": 0.41,
        "mirror_threshold": 0.21,
        "mirror_surface_index": None,
        "qi_ceiling_max": 2.0e-3,
        "qi_ceiling_smooth_penalty": 2.0e-3,
        # First find a low-QI, nonzero-transform basin.  Current Boozer-target
        # and direct hard-mirror cleanups lower mirror but destroy QI for this
        # unrelated seed, so they remain opt-in experiments instead of the
        # public default.
        "branch_width_weight": 0.5,
        "weighted_shuffle_profile_weight": 0.0,
        "phimin": 0.0,
        "boozer_target_wout": None,
        "boozer_target_weight": 0.0,
        "boozer_target_normalize": True,
        "boozer_target_include_b00": False,
        "mirror_weight": 0.0,
        "elongation_weight": 0.0,
        "qi_ceiling_weight": 0.0,
        "shuffle_profile_nphi_out": None,
    },
    "nfp4_qh_warm_to_qi": {
        "case_goal": "NFP=4 QH-to-QI stress test; audit before promotion",
        "input_file": DATA_DIR / "input.nfp4_QH_warm_start",
        "output_dir": Path("results/qi_opt/ess/nfp4_qh_warm_to_qi"),
        "max_mode": 3,
        "min_vmec_mode": 6,
        "use_mode_continuation": True,
        "stage_repeats": 3,
        "max_nfev": 10,
        "target_aspect": 5.0,
        "target_abs_iota_min": 0.41,
        "mirror_threshold": 0.21,
        "mirror_surface_index": None,
        "qi_ceiling_max": 2.0e-2,
        "qi_ceiling_smooth_penalty": 2.0e-3,
        "branch_width_weight": 0.5,
        "weighted_shuffle_profile_weight": 0.0,
        "phimin": 0.0,
        "boozer_target_wout": None,
        "boozer_target_weight": 0.0,
        "boozer_target_normalize": True,
        "boozer_target_include_b00": False,
        "mirror_weight": 0.0,
        "elongation_weight": 0.0,
        "qi_ceiling_weight": 0.0,
        "shuffle_profile_nphi_out": None,
    },
    # Template for an arbitrary VMEC input deck:
    # "my_seed": {
    #     "case_goal": "describe the seed and acceptance intent",
    #     "input_file": Path("/absolute/path/to/input.my_seed"),
    #     "output_dir": Path("results/qi_opt/ess/my_seed"),
    #     "max_mode": 3,
    #     "min_vmec_mode": 6,
    #     "use_mode_continuation": True,
    #     "stage_repeats": 5,
    #     "max_nfev": 12,
    #     "target_aspect": 5.0,
    #     "target_abs_iota_min": 0.41,
    #     "mirror_threshold": 0.21,
    #     "mirror_surface_index": None,
    #     "qi_ceiling_max": 2.0e-2,
    #     "qi_ceiling_smooth_penalty": 2.0e-3,
    #     "branch_width_weight": 0.5,
    #     "weighted_shuffle_profile_weight": 0.0,
    #     "phimin": 0.0,
    #     # Optional homotopy target for far seeds.  Use a solved QI wout with
    #     # the same NFP to steer the Boozer |B| spectrum before QI cleanup.
    #     "boozer_target_wout": None,
    #     "boozer_target_weight": 0.0,
    #     "boozer_target_normalize": True,
    #     "boozer_target_include_b00": False,
    #     "mirror_weight": 0.0,
    #     "elongation_weight": 0.0,
    #     "qi_ceiling_weight": 0.0,
    #     "shuffle_profile_nphi_out": None,
    # },
}

RUN_CASE = "nfp2_qi"  # Try "qi_stel_seed_3127" or "nfp4_qh_warm_to_qi".
CASE = QI_CASES[RUN_CASE]

# Problem parameters.  The default case uses the bundled NFP=2 omnigenity seed
# because it gives the current best mirror-aware QI result in this repository.
# Users can change RUN_CASE or add a QI_CASES entry for any VMEC input deck
# while keeping the same objective-construction workflow below.
# The smooth metric is calibrated against the Goodman et al. branch-shuffle
# diagnostic: branch-width tracks bounce-width invariance, shuffle-profile
# compares against the branch-equalized well, and a small profile term keeps
# QH/QP-like false positives from ranking too favorably.
INPUT_FILE = CASE["input_file"]
OUTPUT_DIR = CASE["output_dir"]
MAX_MODE = int(CASE["max_mode"])
MIN_VMEC_MODE = int(CASE.get("min_vmec_mode", max(6, MAX_MODE + 3)))
USE_MODE_CONTINUATION = bool(CASE["use_mode_continuation"])  # Repeats the same max-mode stage for QI cleanup.
MAX_NFEV = int(CASE["max_nfev"])
CONTINUATION_NFEV = 0
QI_PREFINE = False
QI_PREFINE_NFEV = 30
STAGE_REPEATS = int(CASE["stage_repeats"])
STAGE_MODES = vj.repeated_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
    repeats=STAGE_REPEATS,
)

# Optimizer parameters.
METHOD = "scipy"  # Try also "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
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
USE_ESS = True  # Set False for an unscaled trust-region solve.
ALPHA = 1.2  # ESS high-mode scaling strength.
# Common alternatives:
# METHOD = "gauss_newton"
# METHOD = "lbfgs_adjoint"
# USE_MODE_CONTINUATION = False
# STAGE_MODES = [MAX_MODE]
# USE_ESS = False

# Output controls.
SAVE_STAGE_INPUTS = True  # Keep per-stage input decks for continuation/debugging.
SAVE_STAGE_WOUTS = False  # Set True to also write per-stage WOUT files.
MAKE_PLOTS = True

# Scalar and field-quality targets.  The default objective optimizes QI,
# aspect, and a nonzero-transform floor.  Mirror/elongation terms are useful
# engineering cleanup terms, but they are intentionally staged: the first solve
# should find a QI+iota basin, and mirror cleanup should then be added without
# dropping the QI and iota gates.
TARGET_ASPECT = float(CASE["target_aspect"])
TARGET_ABS_IOTA_MIN = float(CASE["target_abs_iota_min"])
MAX_MIRROR_RATIO = float(CASE.get("mirror_threshold", 0.21))
MIRROR_SURFACE_INDEX = CASE.get("mirror_surface_index", None)
MAX_ELONGATION = 8.0
SURFACES = np.linspace(0.1, 1.0, 6)
ASPECT_WEIGHT = 0.25
IOTA_FLOOR_WEIGHT = 200.0**2
QI_WEIGHT = 10.0
QI_CEILING_MAX = float(CASE.get("qi_ceiling_max", 2.0e-2))
QI_CEILING_WEIGHT = float(CASE["qi_ceiling_weight"])
QI_CEILING_SMOOTH_PENALTY = float(CASE.get("qi_ceiling_smooth_penalty", 2.0e-3))
MIRROR_WEIGHT = float(CASE["mirror_weight"])
ELONGATION_WEIGHT = float(CASE["elongation_weight"])
BOOZER_TARGET_WOUT = CASE.get("boozer_target_wout")
BOOZER_TARGET_WEIGHT = float(CASE.get("boozer_target_weight", 0.0))
MIRROR_SMOOTH_EXTREMA = 2.0e-2
MIRROR_SMOOTH_PENALTY = 2.0e-2
QI_GATE_SMOOTH_MAX = 2.0e-3
QI_GATE_LEGACY_MAX = 1.0e-3

# Boozer transform and smooth-QI residual resolution.
QI_OPTIONS = vj.QuasiIsodynamicOptions(
    surfaces=SURFACES,
    mboz=18,
    nboz=18,
    nphi=151,
    nalpha=31,
    n_bounce=51,
    include_bounce_endpoints=True,  # Matches the legacy Goodman-style QI level sampling.
    softness=2.0e-2,
    width_weight=1.0,
    # A branch-heavy QI residual tracks the Goodman-style branch diagnostic
    # better for the current mirror-aware QI lane than the historical 0.5 value.
    branch_width_weight=float(CASE["branch_width_weight"]),
    branch_width_softness=2.0e-2,
    profile_weight=0.1,
    shuffle_profile_weight=1.0,
    shuffle_profile_softness=2.0e-2,
    # Optional dense branch-shuffle output grid, closer to the legacy
    # omnigenity_optimization arr_out=True objective.  This is useful for
    # homotopy experiments but increases memory and runtime, so the default
    # published example keeps the base nphi grid.
    shuffle_profile_nphi_out=CASE.get("shuffle_profile_nphi_out"),
    # Optional closer-to-legacy weighted branch-shuffle term.  It is useful for
    # diagnostics and homotopy experiments, but the current best QI example
    # keeps it off because it did not improve the mirror-aware NFP=2 run.
    weighted_shuffle_profile_weight=float(CASE.get("weighted_shuffle_profile_weight", 0.0)),
    weighted_shuffle_profile_softness=2.0e-2,
    aligned_profile_weight=0.0,
    aligned_profile_softness=2.0e-2,
    aligned_profile_trap_level=0.65,
    aligned_profile_trap_softness=5.0e-2,
    phimin=float(CASE.get("phimin", 0.0)),  # Set to np.pi / nfp if auditing a seed whose well starts there.
)


# Objective function.  Add new terms by appending another
# (objective.J, target, weight) tuple.
aspect = vj.AspectRatio()
iota_floor = vj.AbsMeanIotaFloor(TARGET_ABS_IOTA_MIN)
qi = vj.QuasiIsodynamicResidual(QI_OPTIONS)
qi_ceiling = vj.QuasiIsodynamicResidualCeiling(
    maximum=QI_CEILING_MAX,
    smooth_penalty=QI_CEILING_SMOOTH_PENALTY,
    qi_options=QI_OPTIONS,
)
mirror = vj.MirrorRatio(
    threshold=MAX_MIRROR_RATIO,
    ntheta=96,
    nphi=96,
    # None means "all QI surfaces", matching QIDiagnosticOptions'
    # mirror-ratio gate.  Set an integer to optimize one surface only.
    surface_index=MIRROR_SURFACE_INDEX,
    smooth_extrema=MIRROR_SMOOTH_EXTREMA,
    smooth_penalty=MIRROR_SMOOTH_PENALTY,
    qi_options=QI_OPTIONS,
)
elongation = vj.MaxElongation(
    threshold=MAX_ELONGATION,
    ntheta=48,
    nphi=16,
    qi_options=QI_OPTIONS,
)
boozer_target = None
if BOOZER_TARGET_WOUT is not None and BOOZER_TARGET_WEIGHT > 0.0:
    target = vj.boozer_b_target_from_wout(
        BOOZER_TARGET_WOUT,
        surfaces=SURFACES,
        mboz=QI_OPTIONS.mboz,
        nboz=QI_OPTIONS.nboz,
    )
    boozer_target = vj.BoozerBTarget(
        target_bmnc=target["bmnc_b"],
        target_bmns=target["bmns_b"],
        normalize=bool(CASE.get("boozer_target_normalize", True)),
        include_b00=bool(CASE.get("boozer_target_include_b00", False)),
        qi_options=QI_OPTIONS,
    )

qi_only_objective_tuples = [
    # Optional diagnostic-only pre-refinement.  Do not promote this stage by
    # itself: QI-only can converge to a low-QI branch with near-zero iota.
    (qi.J, 0.0, QI_WEIGHT),
]
objective_tuples = [
    (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
    (iota_floor.J, 0.0, IOTA_FLOOR_WEIGHT),
    (qi.J, 0.0, QI_WEIGHT),
]
if boozer_target is not None:
    # Homotopy steering for far seeds: match a solved same-NFP QI Boozer
    # spectrum while the QI/iota terms keep the solve in a useful basin.
    objective_tuples.insert(2, (boozer_target.J, 0.0, BOOZER_TARGET_WEIGHT))
if QI_CEILING_WEIGHT > 0.0:
    # Soft-wall QI guard: mirror/elongation cleanup may trade scalar objective
    # against QI.  This term is inactive below QI_CEILING_MAX and grows once a
    # trial leaves the accepted QI basin.
    objective_tuples.append((qi_ceiling.J, 0.0, QI_CEILING_WEIGHT))
if MIRROR_WEIGHT > 0.0:
    # All-surface Boozer mirror cleanup.  Keep QI/iota terms active so this
    # engineering target cannot silently destroy omnigenity.
    objective_tuples.append((mirror.J, 0.0, MIRROR_WEIGHT))
if ELONGATION_WEIGHT > 0.0:
    objective_tuples.append((elongation.J, 0.0, ELONGATION_WEIGHT))
# Optional terms users can append in the same tuple style:
# objective_tuples.append((vj.LgradB(threshold=0.30, smooth_penalty=1.0e-3).J, 0.0, 0.001))
# objective_tuples.append((vj.MagneticWell(minimum=0.0).J, 0.0, 1.0))
# objective_tuples.append((vj.DMerc(minimum=0.0, softness=1.0e-3).J, 0.0, DMERC_WEIGHT))
# Finite-beta examples show VolavgB, BetaTotal, JDotB, RedlBootstrapMismatch,
# and profile-current targets without making this QI teaching script longer.
qi_only_problem = vj.LeastSquaresProblem.from_tuples(qi_only_objective_tuples)
problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)

print("\nQI optimization policy:")
print(f"  case:            {RUN_CASE}")
print(f"  case goal:       {CASE.get('case_goal', 'custom QI candidate')}")
print(f"  input file:      {INPUT_FILE}")
print(f"  output dir:      {OUTPUT_DIR}")
print(f"  max_mode:        {MAX_MODE}")
print(f"  min_vmec_mode:   {MIN_VMEC_MODE}")
print(f"  stage_modes:     {STAGE_MODES}")
print(f"  max_nfev:        {MAX_NFEV}")
print(f"  ESS:             {USE_ESS} (alpha={ALPHA})")
print(f"  target aspect:   {TARGET_ASPECT}")
print(f"  abs iota floor:  {TARGET_ABS_IOTA_MIN}")
print(f"  QI branch weight:{QI_OPTIONS.branch_width_weight}")
print(f"  QI weighted shuffle:{QI_OPTIONS.weighted_shuffle_profile_weight}")
print(f"  QI phimin:       {QI_OPTIONS.phimin}")
print(f"  Boozer target:   {BOOZER_TARGET_WOUT} (weight={BOOZER_TARGET_WEIGHT})")
print(f"  mirror target:   {MAX_MIRROR_RATIO} (surface={MIRROR_SURFACE_INDEX})")
print(f"  mirror weight:   {MIRROR_WEIGHT}")
print(f"  elongation wt:   {ELONGATION_WEIGHT}")
print(f"  QI ceiling:      {QI_CEILING_MAX} (weight={QI_CEILING_WEIGHT})")

active_input_file = INPUT_FILE
if QI_PREFINE:
    preseed_vmec = vj.FixedBoundaryVMEC.from_input(
        INPUT_FILE,
        max_mode=MAX_MODE,
        min_vmec_mode=MIN_VMEC_MODE,
        output_dir=OUTPUT_DIR / "qi_preseed",
        project_input_boundary_to_max_mode=True,
    )
    print("Running QI-only pre-refinement before applying scalar constraints ...")
    preseed_result = vj.least_squares_solve(
        preseed_vmec,
        qi_only_problem,
        stage_modes=STAGE_MODES,
        max_nfev=QI_PREFINE_NFEV,
        continuation_nfev=min(CONTINUATION_NFEV, QI_PREFINE_NFEV),
        method=METHOD,
        ftol=FTOL,
        gtol=GTOL,
        xtol=XTOL,
        use_ess=USE_ESS,
        ess_alpha=ALPHA,
        label=f"QI-only pre-refinement (max_mode={MAX_MODE})",
        use_mode_continuation=USE_MODE_CONTINUATION,
        inner_max_iter=INNER_MAX_ITER,
        inner_ftol=INNER_FTOL,
        trial_max_iter=TRIAL_MAX_ITER,
        trial_ftol=TRIAL_FTOL,
        solver_device=SOLVER_DEVICE,
        scipy_tr_solver=SCIPY_TR_SOLVER,
        scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
        save_stage_inputs=SAVE_STAGE_INPUTS,
        save_stage_wouts=SAVE_STAGE_WOUTS,
    )
    preseed_history = preseed_result.history
    print(f"QI-only pre-refinement final objective: {preseed_history['objective_final']:.6e}")
    active_input_file = OUTPUT_DIR / "qi_preseed" / "input.final"

vmec = vj.FixedBoundaryVMEC.from_input(
    active_input_file,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR,
    project_input_boundary_to_max_mode=True,
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
    save_stage_inputs=SAVE_STAGE_INPUTS,
    save_stage_wouts=SAVE_STAGE_WOUTS,
)

# Results are plain Python objects.  The solve writes these default artifacts
# for convenience; the explicit calls below show where to customize filenames
# or add additional exports in a SIMSOPT-style workflow.
initial_optimizer = result.initial_optimizer
final_optimizer = result.final_optimizer
final_result = result.final_result
history = result.history
objective_history = result.objective_history
timing = result.timing_summary

saved_paths = {
    "initial_input": OUTPUT_DIR / "input.initial",
    "final_input": OUTPUT_DIR / "input.final",
    "initial_wout": OUTPUT_DIR / "wout_initial.nc",
    "final_wout": OUTPUT_DIR / "wout_final.nc",
    "history": OUTPUT_DIR / "history.json",
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
initial_optimizer.save_input(saved_paths["initial_input"], result.initial_params)
initial_optimizer.save_wout(
    saved_paths["initial_wout"],
    result.initial_params,
    state=result.initial_state,
)
final_optimizer.save_input(saved_paths["final_input"], result.final_params)
final_optimizer.save_wout(
    saved_paths["final_wout"],
    result.final_params,
    state=result.final_state,
)
final_optimizer.save_history(saved_paths["history"], final_result)

print("\nFinal diagnostics from result.history:")
print(f"  aspect ratio:     {history['aspect_final']:.6g}")
print(f"  mean iota:        {history['iota_final']:.6g}")
print(f"  QI objective:     {history['qs_final']:.6e}")
print(f"  total objective:  {history['objective_final']:.6e}")
print(f"  wall time:        {timing['total_wall_time_s']:.2f} s")
print(f"  objective samples: {objective_history[:5]} ... {objective_history[-3:]}")

print("\nFiles saved from result objects:")
for name, path in saved_paths.items():
    print(f"  {name}: {path}")

wout_final = vj.load_wout(saved_paths["final_wout"])
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

diagnostic_options = vj.QIDiagnosticOptions(
    surfaces=SURFACES,
    mboz=QI_OPTIONS.mboz,
    nboz=QI_OPTIONS.nboz,
    nphi=QI_OPTIONS.nphi,
    nalpha=QI_OPTIONS.nalpha,
    n_bounce=QI_OPTIONS.n_bounce,
    include_bounce_endpoints=QI_OPTIONS.include_bounce_endpoints,
    softness=QI_OPTIONS.softness,
    width_weight=QI_OPTIONS.width_weight,
    branch_width_weight=QI_OPTIONS.branch_width_weight,
    branch_width_softness=QI_OPTIONS.branch_width_softness,
    profile_weight=QI_OPTIONS.profile_weight,
    shuffle_profile_weight=QI_OPTIONS.shuffle_profile_weight,
    shuffle_profile_softness=QI_OPTIONS.shuffle_profile_softness,
    shuffle_profile_nphi_out=QI_OPTIONS.shuffle_profile_nphi_out,
    weighted_shuffle_profile_weight=QI_OPTIONS.weighted_shuffle_profile_weight,
    weighted_shuffle_profile_softness=QI_OPTIONS.weighted_shuffle_profile_softness,
    aligned_profile_weight=QI_OPTIONS.aligned_profile_weight,
    aligned_profile_softness=QI_OPTIONS.aligned_profile_softness,
    aligned_profile_trap_level=QI_OPTIONS.aligned_profile_trap_level,
    aligned_profile_trap_softness=QI_OPTIONS.aligned_profile_trap_softness,
    phimin=float(QI_OPTIONS.phimin),
    mirror_threshold=MAX_MIRROR_RATIO,
    elongation_threshold=MAX_ELONGATION,
)
diagnostics = vj.qi_diagnostics_from_state(
    state=result.final_state,
    static=final_optimizer.static,
    indata=final_optimizer.indata,
    signgs=final_optimizer.signgs,
    surfaces=SURFACES,
    options=diagnostic_options,
)
diagnostics = annotate_qi_seed_suitability(
    diagnostics,
    targets=QISeedSuitabilityTargets(
        smooth_qi_max=QI_GATE_SMOOTH_MAX,
        legacy_qi_max=QI_GATE_LEGACY_MAX,
        target_aspect=TARGET_ASPECT,
        abs_iota_min=TARGET_ABS_IOTA_MIN,
        mirror_ratio_max=MAX_MIRROR_RATIO,
        max_elongation=MAX_ELONGATION,
    ),
)
smooth_qi = _diagnostic_float(diagnostics, "qi_smooth_total")
legacy_qi = _diagnostic_float(diagnostics, "qi_legacy_total")
aspect_ratio = _diagnostic_float(diagnostics, "aspect")
mean_iota = _diagnostic_float(diagnostics, "mean_iota")
abs_iota = abs(mean_iota)
mirror_ratio = _diagnostic_float(diagnostics, "qi_mirror_ratio_max")
max_elongation = _diagnostic_float(diagnostics, "qi_max_elongation")
qi_gate_passed = bool(diagnostics["qi_seed_gate_passed"])
engineering_gate_passed = bool(diagnostics["qi_engineering_gate_passed"])
print("\nIndependent QI promotion gate:")
print(f"  smooth QI:       {smooth_qi:.6e}  (limit {QI_GATE_SMOOTH_MAX:.1e})")
print(f"  legacy QI:       {legacy_qi:.6e}  (limit {QI_GATE_LEGACY_MAX:.1e})")
print(f"  aspect ratio:    {aspect_ratio:.6g}  (target {TARGET_ASPECT:.3g})")
print(f"  abs(mean iota):  {abs_iota:.6g}  (minimum {TARGET_ABS_IOTA_MIN:.3g})")
print(f"  mirror ratio:    {mirror_ratio:.6g}  (target {MAX_MIRROR_RATIO:.3g})")
print(f"  mirror by surf:  {diagnostics.get('qi_mirror_ratio_by_surface')}")
print(f"  max elongation:  {max_elongation:.6g}  (target {MAX_ELONGATION:.3g})")
print(f"  QI seed gate:    {qi_gate_passed}")
print(f"  full eng. gate:  {engineering_gate_passed}")
print(f"  rank score:      {diagnostics['qi_rank_score']:.6e}")
print(f"  failed gates:    {diagnostics['qi_gate_failures']}")
for reason in diagnostics["qi_failure_reasons"]:
    print(f"    - {reason}")
if engineering_gate_passed:
    print("\nVerdict: full QI engineering gate passed for this resolution and policy.")
elif qi_gate_passed:
    print(
        "\nVerdict: QI+iota gate passed, but mirror/elongation/aspect gates are not all "
        "satisfied. Treat this as a candidate basin and run a guarded cleanup or "
        "higher-resolution audit before promotion."
    )
else:
    print(
        "\nVerdict: QI+iota gate failed. Treat this as a diagnostic run, not a "
        "promoted QI optimization result."
    )

if MAKE_PLOTS:
    plot_paths = {
        "boundary_comparison": vj.plot_3d_boundary_comparison(
            saved_paths["initial_wout"],
            saved_paths["final_wout"],
            outdir=OUTPUT_DIR,
        ),
        "bmag_contours": vj.plot_bmag_contours(
            saved_paths["initial_wout"],
            saved_paths["final_wout"],
            outdir=OUTPUT_DIR,
        ),
        "objective_history": vj.plot_objective_history(
            saved_paths["history"],
            outdir=OUTPUT_DIR,
        ),
        "boozer_bmag_contours": vj.plot_boozer_bmag_contours_from_state(
            result.final_state,
            static=final_optimizer.static,
            indata=final_optimizer.indata,
            signgs=final_optimizer.signgs,
            outdir=OUTPUT_DIR,
            surfaces=(1.0,),
            mboz=QI_OPTIONS.mboz,
            nboz=QI_OPTIONS.nboz,
            title=f"{INPUT_FILE.name}: Boozer |B| contours on LCFS",
        ),
    }
    print("\nPlot files selected by this script:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

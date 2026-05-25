#!/usr/bin/env python
"""Quasi-isodynamic optimization with vmec_jax and booz_xform_jax.

This file is deliberately shaped like the QA/QH/QP examples: choose input and
controls, build the VMEC optimization object, construct objective tuples, solve,
then save and plot from the returned result.  QI needs extra staged promotion
logic for far seeds; that mechanical policy lives in ``qi_optimization_support``
so the scientific objective remains visible here.
"""

from pathlib import Path
import json
import os
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import vmec_jax as vj
from vmec_jax._compat import enable_x64
from vmec_jax.qi_diagnostics import QISeedSuitabilityTargets, annotate_qi_seed_suitability

from qi_optimization_cases import QI_CASES, resolve_qi_case
import qi_optimization_support as qis


enable_x64(True)

AVAILABLE_QI_CASES = tuple(sorted(QI_CASES))
RUN_CASE = "nfp2_qi"  # Try "nfp1_qi", "nfp3_qi", "nfp4_qi", "nfp4_qi_finite_beta", or "nfp4_qh_warm_to_qi".
RUN_CASE, CASE = resolve_qi_case(RUN_CASE)

# Problem parameters.  The default case uses the bundled NFP=2 omnigenity seed
# because it gives the current best mirror-aware QI result in this repository.
# Users can change RUN_CASE/VMEC_JAX_QI_RUN_CASE, pass VMEC_JAX_QI_INPUT, or add
# a QI_CASES entry in qi_optimization_cases.py for another VMEC input deck.
INPUT_FILE = CASE["input_file"]
OUTPUT_DIR = CASE["output_dir"]
EXPECTED_GATE_STATUS = str(CASE.get("expected_gate_status", "candidate"))
EXPECTED_GATE_FAILURES = tuple(CASE.get("expected_gate_failures", ()))
STRESS_FIXTURE_NOTES = tuple(CASE.get("stress_fixture_notes", ()))
KNOWN_BEST_NFP4_QUICK_AUDIT = dict(CASE.get("known_best_nfp4_quick_audit", {}))
MAX_MODE = int(os.environ.get("VMEC_JAX_QI_MAX_MODE", CASE["max_mode"]))
MIN_VMEC_MODE = int(os.environ.get("VMEC_JAX_QI_MIN_VMEC_MODE", CASE.get("min_vmec_mode", max(6, MAX_MODE + 3))))
USE_SIMPLE_SEED = os.environ.get("VMEC_JAX_QI_USE_SIMPLE_SEED", "1").strip().lower() not in {"0", "false", "no", "off"}
INPUT_FILE = vj.prepare_simple_omnigenity_seed_input(
    INPUT_FILE, OUTPUT_DIR, max_mode=MAX_MODE, min_vmec_mode=MIN_VMEC_MODE, enabled=USE_SIMPLE_SEED
)
_USE_MODE_CONTINUATION_ENV = os.environ.get("VMEC_JAX_QI_USE_MODE_CONTINUATION")
USE_MODE_CONTINUATION = (
    bool(CASE["use_mode_continuation"])
    if _USE_MODE_CONTINUATION_ENV is None
    else _USE_MODE_CONTINUATION_ENV.strip().lower() not in {"0", "false", "no", "off"}
)
_MAX_NFEV_ENV = os.environ.get("VMEC_JAX_QI_MAX_NFEV")
MAX_NFEV = int(_MAX_NFEV_ENV if _MAX_NFEV_ENV is not None else CASE["max_nfev"])
CONTINUATION_NFEV = 0
QI_PREFINE = False
QI_PREFINE_NFEV = 30
STAGE_REPEATS = int(os.environ.get("VMEC_JAX_QI_STAGE_REPEATS", CASE["stage_repeats"]))
STAGE_MODES = vj.repeated_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
    repeats=STAGE_REPEATS,
)
MIRROR_RAMP_STAGES = tuple(CASE.get("mirror_ramp_stages", ()))
if _MAX_NFEV_ENV is not None:
    budgeted_stages = []
    for stage in MIRROR_RAMP_STAGES:
        stage_max_nfev = int(stage.get("max_nfev", MAX_NFEV))
        if bool(stage.get("use_showcase_max_nfev", False)):
            stage_max_nfev = MAX_NFEV
        else:
            stage_max_nfev = min(stage_max_nfev, MAX_NFEV)
        budgeted_stages.append({**stage, "max_nfev": stage_max_nfev})
    MIRROR_RAMP_STAGES = tuple(budgeted_stages)
_TARGET_HELICITY_SEED_AMP_ENV = os.environ.get("VMEC_JAX_QI_TARGET_HELICITY_SEED_AMPLITUDE")
_TARGET_HELICITY_SEED_ENABLED_ENV = os.environ.get("VMEC_JAX_QI_USE_TARGET_HELICITY_SEED")
TARGET_HELICITY_SEED_ENABLED = (
    bool(CASE.get("target_helicity_seed_enabled", bool(CASE.get("target_helicity_seed_terms"))))
    if _TARGET_HELICITY_SEED_ENABLED_ENV is None
    else _TARGET_HELICITY_SEED_ENABLED_ENV.strip().lower() not in {"0", "false", "no", "off"}
)
TARGET_HELICITY_SEED_AMPLITUDE = float(
    _TARGET_HELICITY_SEED_AMP_ENV
    if _TARGET_HELICITY_SEED_AMP_ENV is not None
    else CASE.get("target_helicity_seed_amplitude", qis.TARGET_HELICITY_SEED_AMPLITUDE)
)
TARGET_HELICITY_SEED_TERMS = (
    qis.target_helicity_seed_terms(max_mode=MAX_MODE, amplitude=TARGET_HELICITY_SEED_AMPLITUDE)
    if _TARGET_HELICITY_SEED_AMP_ENV is not None or not CASE.get("target_helicity_seed_terms")
    else tuple(CASE.get("target_helicity_seed_terms", ()))
)
TARGET_HELICITY_SEED_CONFIG = {
    "enabled": TARGET_HELICITY_SEED_ENABLED,
    "terms": TARGET_HELICITY_SEED_TERMS,
    "only_if_abs_below": float(CASE.get("target_helicity_seed_only_if_abs_below", 0.0)),
}

# Optimizer parameters.  These are optimization controls only; physics targets
# stay in the objective tuples below, matching SIMSOPT's teaching workflow.
METHOD = os.environ.get("VMEC_JAX_QI_METHOD", CASE.get("method", "scipy"))  # Try "auto", "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
SCIPY_TR_SOLVER = "lsmr"  # For METHOD="scipy": "lsmr" is memory-light; "exact" is dense.
SCIPY_LSMR_MAXITER = None  # None lets SciPy choose; set an int to cap LSMR iterations.
FTOL = 1.0e-4  # Relative cost-reduction tolerance for the outer optimizer.
GTOL = 1.0e-4  # Gradient optimality tolerance for the outer optimizer.
XTOL = 1.0e-8  # Step-size tolerance; QI often benefits from a tighter value.
INNER_MAX_ITER = int(os.environ.get("VMEC_JAX_QI_INNER_MAX_ITER", CASE.get("inner_max_iter", 120)))  # 0 uses NITER from the input deck.
INNER_FTOL = float(os.environ.get("VMEC_JAX_QI_INNER_FTOL", CASE.get("inner_ftol", 1.0e-9)))  # 0 uses FTOL from the input deck.
TRIAL_MAX_ITER = int(os.environ.get("VMEC_JAX_QI_TRIAL_MAX_ITER", CASE.get("trial_max_iter", 120)))  # 0 follows accepted/input budget.
TRIAL_FTOL = float(os.environ.get("VMEC_JAX_QI_TRIAL_FTOL", CASE.get("trial_ftol", 1.0e-9)))  # 0 follows accepted/input tolerance.
_SOLVER_DEVICE_ENV = os.environ.get("VMEC_JAX_QI_SOLVER_DEVICE")
SOLVER_DEVICE = None if _SOLVER_DEVICE_ENV in (None, "", "none", "None") else _SOLVER_DEVICE_ENV  # Set "cpu" or "gpu" to force one backend.
_EXACT_PATH_ENV = os.environ.get("VMEC_JAX_QI_EXACT_PATH")
EXACT_PATH = None if _EXACT_PATH_ENV in (None, "", "none", "None", "auto") else _EXACT_PATH_ENV  # Set "scan" only for long warm GPU runs.
USE_ESS = os.environ.get("VMEC_JAX_QI_USE_ESS", "1").strip().lower() not in {"0", "false", "no", "off"}
ALPHA = float(os.environ.get("VMEC_JAX_QI_ESS_ALPHA", 1.2))  # ESS high-mode scaling strength.
STAGE_METHOD_OVERRIDE = os.environ.get("VMEC_JAX_QI_STAGE_METHOD")  # Override staged cleanup method for bounded experiments.
SCALAR_STEP_BOUND_OVERRIDE = os.environ.get("VMEC_JAX_QI_SCALAR_STEP_BOUND")  # Useful with METHOD/STAGE_METHOD="scalar_trust".
LBFGS_STEP_BOUND_OVERRIDE = os.environ.get("VMEC_JAX_QI_LBFGS_STEP_BOUND")  # Useful with METHOD/STAGE_METHOD="lbfgs_adjoint".
if STAGE_METHOD_OVERRIDE or SCALAR_STEP_BOUND_OVERRIDE or LBFGS_STEP_BOUND_OVERRIDE:
    overridden_stages = []
    for stage in MIRROR_RAMP_STAGES:
        stage_override = dict(stage)
        if STAGE_METHOD_OVERRIDE:
            stage_override["method"] = STAGE_METHOD_OVERRIDE
        if SCALAR_STEP_BOUND_OVERRIDE is not None:
            stage_override["scalar_step_bound"] = float(SCALAR_STEP_BOUND_OVERRIDE)
        if LBFGS_STEP_BOUND_OVERRIDE is not None:
            stage_override["lbfgs_step_bound"] = float(LBFGS_STEP_BOUND_OVERRIDE)
        overridden_stages.append(stage_override)
    MIRROR_RAMP_STAGES = tuple(overridden_stages)
# Common alternatives:
# METHOD = "gauss_newton"
# METHOD = "lbfgs_adjoint"
# USE_MODE_CONTINUATION = False
# STAGE_MODES = [MAX_MODE]
# USE_ESS = False

# Output controls.
SAVE_STAGE_INPUTS = True  # Keep per-stage input decks for continuation/debugging.
SAVE_STAGE_WOUTS = False  # Set True to also write per-stage WOUT files.
MAKE_PLOTS = os.environ.get("VMEC_JAX_QI_MAKE_PLOTS", "1").strip().lower() not in {"0", "false", "no", "off"}

# Physics targets and weights.  Edit these or append objective tuples to explore
# other QI/engineering tradeoffs.  The default public QI lane follows the
# QA/QH/QP examples at aspect 6; non-default robustness cases keep their
# case-specific target unless VMEC_JAX_QI_TARGET_ASPECT is set.
DEFAULT_TARGET_ASPECT = 6.0
_TARGET_ASPECT_ENV = os.environ.get("VMEC_JAX_QI_TARGET_ASPECT")
TARGET_ASPECT = float(
    _TARGET_ASPECT_ENV
    if _TARGET_ASPECT_ENV is not None
    else (DEFAULT_TARGET_ASPECT if RUN_CASE == "nfp2_qi" else CASE["target_aspect"])
)
TARGET_ABS_IOTA_MIN = float(CASE["target_abs_iota_min"])
MAX_MIRROR_RATIO = float(CASE.get("mirror_threshold", 0.21))
MIRROR_SURFACE_INDEX = CASE.get("mirror_surface_index", None)
MAX_ELONGATION = float(CASE.get("max_elongation", 8.0))
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
QI_GATE_SMOOTH_MAX = float(CASE.get("qi_gate_smooth_max", 2.0e-3))
QI_GATE_LEGACY_MAX = float(CASE.get("qi_gate_legacy_max", 2.0e-3))
JIT_BOOZ = bool(CASE.get("jit_booz", True))


def _resolution_value(resolution, key, default):
    return int(dict(resolution).get(key, default))


OPT_QI_RESOLUTION = dict(CASE.get("optimization_qi_resolution", {}))
AUDIT_QI_RESOLUTION = {**OPT_QI_RESOLUTION, **dict(CASE.get("audit_qi_resolution", {}))}
QI_OPTIONS = vj.QuasiIsodynamicOptions(
    surfaces=SURFACES,
    mboz=_resolution_value(OPT_QI_RESOLUTION, "mboz", 18),
    nboz=_resolution_value(OPT_QI_RESOLUTION, "nboz", 18),
    nphi=_resolution_value(OPT_QI_RESOLUTION, "nphi", 151),
    nalpha=_resolution_value(OPT_QI_RESOLUTION, "nalpha", 31),
    n_bounce=_resolution_value(OPT_QI_RESOLUTION, "n_bounce", 51),
    include_bounce_endpoints=True,
    softness=2.0e-2,
    width_weight=1.0,
    branch_width_weight=float(CASE["branch_width_weight"]),
    branch_width_softness=2.0e-2,
    profile_weight=0.1,
    shuffle_profile_weight=1.0,
    shuffle_profile_softness=2.0e-2,
    shuffle_profile_nphi_out=CASE.get("shuffle_profile_nphi_out"),
    weighted_shuffle_profile_weight=float(CASE.get("weighted_shuffle_profile_weight", 0.0)),
    weighted_shuffle_profile_softness=2.0e-2,
    aligned_profile_weight=0.0,
    aligned_profile_softness=2.0e-2,
    aligned_profile_trap_level=0.65,
    aligned_profile_trap_softness=5.0e-2,
    phimin=float(CASE.get("phimin", 0.0)),
    jit_booz=JIT_BOOZ,
)

# Objective function.  Add new terms by appending another
# (objective.J, target, weight) tuple.
aspect = vj.AspectRatio()
iota_floor = vj.AbsMeanIotaFloor(TARGET_ABS_IOTA_MIN)
qi = vj.QuasiIsodynamicResidual(QI_OPTIONS)
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

qi_only_objective_tuples = [(qi.J, 0.0, QI_WEIGHT)]
qi_only_problem = vj.LeastSquaresProblem.from_tuples(qi_only_objective_tuples)


def _stage_value(stage, key, default):
    return stage.get(key, CASE.get(key, default))


def make_qi_problem(stage=None):
    """Assemble the QI objective tuples for one optimization stage."""

    stage = {} if stage is None else dict(stage)
    qi_ceiling = vj.QuasiIsodynamicResidualCeiling(
        maximum=float(_stage_value(stage, "qi_ceiling_max", QI_CEILING_MAX)),
        smooth_penalty=float(_stage_value(stage, "qi_ceiling_smooth_penalty", QI_CEILING_SMOOTH_PENALTY)),
        qi_options=QI_OPTIONS,
    )
    mirror = vj.MirrorRatio(
        threshold=float(_stage_value(stage, "mirror_threshold", MAX_MIRROR_RATIO)),
        ntheta=96,
        nphi=96,
        surface_index=_stage_value(stage, "mirror_surface_index", MIRROR_SURFACE_INDEX),
        phimin=QI_OPTIONS.phimin,
        smooth_extrema=MIRROR_SMOOTH_EXTREMA,
        smooth_penalty=MIRROR_SMOOTH_PENALTY,
        qi_options=QI_OPTIONS,
    )
    elongation = vj.MaxElongation(
        threshold=MAX_ELONGATION,
        ntheta=48,
        nphi=16,
        smooth_extrema=2.0e-2,
        smooth_penalty=2.0e-2,
        qi_options=QI_OPTIONS,
    )
    if bool(_stage_value(stage, "use_augmented_lagrangian_constraints", False)):
        mirror = vj.AugmentedLagrangianConstraint(
            mirror,
            multiplier=float(_stage_value(stage, "al_mirror_multiplier", 0.0)),
            penalty=float(_stage_value(stage, "al_mirror_penalty", 1.0)),
            softness=MIRROR_SMOOTH_PENALTY,
            name="al_mirror_ratio",
        )
        elongation = vj.AugmentedLagrangianConstraint(
            elongation,
            multiplier=float(_stage_value(stage, "al_elongation_multiplier", 0.0)),
            penalty=float(_stage_value(stage, "al_elongation_penalty", 1.0)),
            softness=2.0e-2,
            name="al_max_elongation",
        )

    objective_tuples = [
        (aspect.J, TARGET_ASPECT, float(_stage_value(stage, "aspect_weight", ASPECT_WEIGHT))),
        (iota_floor.J, 0.0, float(_stage_value(stage, "iota_floor_weight", IOTA_FLOOR_WEIGHT))),
        (qi.J, 0.0, float(_stage_value(stage, "qi_weight", QI_WEIGHT))),
    ]
    if boozer_target is not None:
        objective_tuples.insert(2, (boozer_target.J, 0.0, BOOZER_TARGET_WEIGHT))
    qi_ceiling_weight = float(_stage_value(stage, "qi_ceiling_weight", QI_CEILING_WEIGHT))
    mirror_weight = float(_stage_value(stage, "mirror_weight", MIRROR_WEIGHT))
    elongation_weight = float(_stage_value(stage, "elongation_weight", ELONGATION_WEIGHT))
    if qi_ceiling_weight > 0.0:
        objective_tuples.append((qi_ceiling.J, 0.0, qi_ceiling_weight))
    if mirror_weight > 0.0:
        objective_tuples.append((mirror.J, 0.0, mirror_weight))
    if elongation_weight > 0.0:
        objective_tuples.append((elongation.J, 0.0, elongation_weight))
    return vj.LeastSquaresProblem.from_tuples(objective_tuples)


problem = make_qi_problem()
# Optional terms users can append in the same tuple style:
# objective_tuples.append((vj.LgradB(threshold=0.30, smooth_penalty=1.0e-3).J, 0.0, 0.001))
# objective_tuples.append((vj.MagneticWell(minimum=0.0).J, 0.0, 1.0))
# objective_tuples.append((vj.DMerc(minimum=0.0, softness=1.0e-3).J, 0.0, DMERC_WEIGHT))

print("\nQI optimization policy:")
print(f"  case:            {RUN_CASE}")
print(f"  case goal:       {CASE.get('case_goal', 'custom QI candidate')}")
print(f"  expected gates:  {EXPECTED_GATE_STATUS}")
print(f"  input file:      {INPUT_FILE}")
print(f"  output dir:      {OUTPUT_DIR}")
print(f"  max_mode:        {MAX_MODE}")
print(f"  min_vmec_mode:   {MIN_VMEC_MODE}")
print(f"  simple seed:     {USE_SIMPLE_SEED}")
print(f"  stage_modes:     {STAGE_MODES}")
print(f"  max_nfev:        {MAX_NFEV}")
print(f"  ESS:             {USE_ESS} (alpha={ALPHA})")
print(f"  solver/exact:    device={SOLVER_DEVICE or 'JAX default'}, exact_path={EXACT_PATH or 'default tape'}")
print(f"  target aspect:   {TARGET_ASPECT}")
print(f"  abs iota floor:  {TARGET_ABS_IOTA_MIN}")
print(f"  QI branch weight:{QI_OPTIONS.branch_width_weight}")
print(f"  QI weighted shuffle:{QI_OPTIONS.weighted_shuffle_profile_weight}")
print(f"  QI phimin:       {QI_OPTIONS.phimin}")
print(f"  QI opt grid:     mboz={QI_OPTIONS.mboz}, nboz={QI_OPTIONS.nboz}, nphi={QI_OPTIONS.nphi}, nalpha={QI_OPTIONS.nalpha}, n_bounce={QI_OPTIONS.n_bounce}")
print(f"  QI audit grid:   {AUDIT_QI_RESOLUTION or 'same as optimization'}")
print(f"  JIT Boozer path: {QI_OPTIONS.jit_booz}")
print(f"  Boozer target:   {BOOZER_TARGET_WOUT} (weight={BOOZER_TARGET_WEIGHT})")
print(f"  helicity seed:   {TARGET_HELICITY_SEED_ENABLED} (amp={TARGET_HELICITY_SEED_AMPLITUDE:.1e}, terms={len(TARGET_HELICITY_SEED_TERMS)})")
print(f"  boundary ref.:   {CASE.get('boundary_reference_preconditioner', {}).get('reference_input')} (enabled={CASE.get('boundary_reference_preconditioner', {}).get('enabled', False)})")
print(f"  mirror target:   {MAX_MIRROR_RATIO} (surface={MIRROR_SURFACE_INDEX})")
print(f"  mirror weight:   {MIRROR_WEIGHT}")
print(f"  elongation wt:   {ELONGATION_WEIGHT}")
print(f"  AL constraints:  {CASE.get('use_augmented_lagrangian_constraints', False)}")
print(f"  QI ceiling:      {QI_CEILING_MAX} (weight={QI_CEILING_WEIGHT})")
for note in STRESS_FIXTURE_NOTES:
    print(f"    - {note}")
if MIRROR_RAMP_STAGES:
    print("  mirror ramp:     guarded staged cleanup enabled")
    for index, stage in enumerate(MIRROR_RAMP_STAGES, start=1):
        print(f"    {index}: {stage['name']} mirror<={stage.get('mirror_threshold', MAX_MIRROR_RATIO)} w={stage.get('mirror_weight', MIRROR_WEIGHT)} nfev={stage.get('max_nfev', MAX_NFEV)} repeats={stage.get('stage_repeats', STAGE_REPEATS)}")

stages_without_guard = [
    stage["name"]
    for stage in MIRROR_RAMP_STAGES
    if float(stage.get("mirror_weight", MIRROR_WEIGHT)) > 0.0
    and float(stage.get("qi_ceiling_weight", QI_CEILING_WEIGHT)) <= 0.0
    and not bool(stage.get("require_engineering_gate", False))
]
if stages_without_guard:
    raise ValueError(
        "Mirror-ramp cleanup stages must include QuasiIsodynamicResidualCeiling "
        "or require the independent QI engineering gate; "
        f"missing guard in {stages_without_guard}."
    )


def make_vmec_for_stage(input_file, output_dir):
    return vj.FixedBoundaryVMEC.from_input(
        input_file,
        max_mode=MAX_MODE,
        min_vmec_mode=MIN_VMEC_MODE,
        output_dir=output_dir,
        project_input_boundary_to_max_mode=True,
    )


def solve_qi_stage(
    input_file,
    output_dir,
    stage_problem,
    *,
    max_nfev,
    label,
    stage_modes=STAGE_MODES,
    method=METHOD,
    use_mode_continuation=USE_MODE_CONTINUATION,
    scalar_step_bound=None,
    lbfgs_step_bound=None,
    save_final_outputs=True,
):
    # Small stage helper: physics is still assembled explicitly in
    # make_qi_problem(); this only forwards solve controls for one stage.
    vmec = make_vmec_for_stage(input_file, output_dir)
    return vj.least_squares_solve(
        vmec,
        stage_problem,
        stage_modes=stage_modes,
        max_nfev=max_nfev,
        continuation_nfev=CONTINUATION_NFEV,
        method=method,
        ftol=FTOL,
        gtol=GTOL,
        xtol=XTOL,
        use_ess=USE_ESS,
        ess_alpha=ALPHA,
        label=label,
        use_mode_continuation=use_mode_continuation,
        inner_max_iter=INNER_MAX_ITER,
        inner_ftol=INNER_FTOL,
        trial_max_iter=TRIAL_MAX_ITER,
        trial_ftol=TRIAL_FTOL,
        solver_device=SOLVER_DEVICE,
        exact_path=EXACT_PATH,
        scipy_tr_solver=SCIPY_TR_SOLVER,
        scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
        lbfgs_step_bound=lbfgs_step_bound,
        scalar_step_bound=scalar_step_bound,
        save_stage_inputs=SAVE_STAGE_INPUTS,
        save_stage_wouts=SAVE_STAGE_WOUTS,
        save_final_outputs=save_final_outputs,
    )


QI_CONTEXT = qis.make_qi_optimization_context(globals(), strict=True)
active_input_file = INPUT_FILE
active_input_file = qis.run_target_helicity_seed_preconditioner(
    active_input_file,
    OUTPUT_DIR,
    TARGET_HELICITY_SEED_CONFIG,
    ctx=QI_CONTEXT,
)
BOUNDARY_REFERENCE_PRECONDITIONER = dict(CASE.get("boundary_reference_preconditioner", {}))
active_input_file = qis.run_boundary_reference_preconditioner(
    active_input_file,
    OUTPUT_DIR,
    BOUNDARY_REFERENCE_PRECONDITIONER,
    ctx=QI_CONTEXT,
)
active_input_file = qis.run_basin_prefilter(
    active_input_file,
    OUTPUT_DIR,
    dict(CASE.get("basin_prefilter", {})),
    ctx=QI_CONTEXT,
)

if QI_PREFINE:
    print("Running QI-only pre-refinement before applying scalar constraints ...")
    preseed_result = solve_qi_stage(
        active_input_file,
        OUTPUT_DIR / "qi_preseed",
        qi_only_problem,
        max_nfev=QI_PREFINE_NFEV,
        label=f"QI-only pre-refinement (max_mode={MAX_MODE})",
    )
    print(f"QI-only pre-refinement final objective: {preseed_result.history['objective_final']:.6e}")
    active_input_file = OUTPUT_DIR / "qi_preseed" / "input.final"

result, promotion_log = qis.run_qi_stage_policy(
    active_input_file,
    OUTPUT_DIR,
    solve_qi_stage=solve_qi_stage,
    make_qi_problem=make_qi_problem,
    boundary_reference_preconditioner=BOUNDARY_REFERENCE_PRECONDITIONER,
    mirror_ramp_stages=MIRROR_RAMP_STAGES,
    ctx=QI_CONTEXT,
)

if result is None:
    raise RuntimeError("QI optimization did not produce a result.")

final_optimizer = result.final_optimizer
final_result = result.final_result
history = result.history
objective_history = result.objective_history
timing = result.timing_summary
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
saved_paths = {
    "initial_input": OUTPUT_DIR / "input.initial",
    "final_input": OUTPUT_DIR / "input.final",
    "initial_wout": OUTPUT_DIR / "wout_initial.nc",
    "final_wout": OUTPUT_DIR / "wout_final.nc",
    "history": OUTPUT_DIR / "history.json",
}
print("\nRunning the raw input deck once for initial comparison plots ...")
raw_initial_run = qis.save_raw_seed_initial_artifacts(
    INPUT_FILE,
    saved_paths["initial_input"],
    saved_paths["initial_wout"],
    ctx=QI_CONTEXT,
)
final_optimizer.save_input(saved_paths["final_input"], result.final_params)
final_optimizer.save_wout(saved_paths["final_wout"], result.final_params, state=result.final_state)
final_optimizer.save_history(saved_paths["history"], final_result)
if promotion_log:
    promotion_log_path = OUTPUT_DIR / "mirror_ramp_promotion_log.json"
    promotion_log_path.write_text(json.dumps(promotion_log, indent=2) + "\n")
    saved_paths["mirror_ramp_promotion_log"] = promotion_log_path

print("\nFinal diagnostics from result.history:")
print(f"  aspect ratio:     {history['aspect_final']:.6g}")
print(f"  mean iota:        {history['iota_final']:.6g}")
print(f"  QI objective:     {history['qs_final']:.6e}")
print(f"  total objective:  {history['objective_final']:.6e}")
print(f"  wall time:        {timing['total_wall_time_s']:.2f} s")
print(f"  objective samples: {objective_history[:5]} ... {objective_history[-3:]}")

print("\nFiles saved for raw-seed/final comparison:")
for name, path in saved_paths.items():
    print(f"  {name}: {path}")

wout_final = vj.load_wout(saved_paths["final_wout"])
theta, zeta, b_lcfs = vj.vmecplot2_bmag_grid(wout_final, s_index=-1, ntheta=64, nzeta=64, zeta_max=2.0 * np.pi / float(wout_final.nfp))
print("\nLCFS |B| data from vmecplot2_bmag_grid:")
print(f"  theta grid: {theta.shape}, zeta grid: {zeta.shape}, B grid: {b_lcfs.shape}")
print(f"  Bmin/Bmax:  {np.min(b_lcfs):.6g} / {np.max(b_lcfs):.6g}")

diagnostic_options = vj.QIDiagnosticOptions(
    surfaces=SURFACES,
    mboz=_resolution_value(AUDIT_QI_RESOLUTION, "mboz", QI_OPTIONS.mboz),
    nboz=_resolution_value(AUDIT_QI_RESOLUTION, "nboz", QI_OPTIONS.nboz),
    nphi=_resolution_value(AUDIT_QI_RESOLUTION, "nphi", QI_OPTIONS.nphi),
    nalpha=_resolution_value(AUDIT_QI_RESOLUTION, "nalpha", QI_OPTIONS.nalpha),
    n_bounce=_resolution_value(AUDIT_QI_RESOLUTION, "n_bounce", QI_OPTIONS.n_bounce),
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
diagnostics.update(
    {
        "qi_optimization_resolution": {"mboz": QI_OPTIONS.mboz, "nboz": QI_OPTIONS.nboz, "nphi": QI_OPTIONS.nphi, "nalpha": QI_OPTIONS.nalpha, "n_bounce": QI_OPTIONS.n_bounce},
        "qi_audit_resolution": {"mboz": diagnostic_options.mboz, "nboz": diagnostic_options.nboz, "nphi": diagnostic_options.nphi, "nalpha": diagnostic_options.nalpha, "n_bounce": diagnostic_options.n_bounce},
        "qi_case_expected_gate_status": EXPECTED_GATE_STATUS,
        "qi_case_stress_fixture": EXPECTED_GATE_STATUS == "non_passing_stress_fixture",
        "qi_case_expected_gate_failures": list(EXPECTED_GATE_FAILURES),
        "qi_case_stress_fixture_notes": list(STRESS_FIXTURE_NOTES),
        "qi_case_known_best_nfp4_quick_audit": KNOWN_BEST_NFP4_QUICK_AUDIT,
    }
)
expected_non_passing_stress = EXPECTED_GATE_STATUS == "non_passing_stress_fixture"
engineering_gate_passed = bool(diagnostics["qi_engineering_gate_passed"])
diagnostics["qi_case_expected_outcome_met"] = not engineering_gate_passed if expected_non_passing_stress else engineering_gate_passed
diagnostics_path = OUTPUT_DIR / "diagnostics.json"
diagnostics_path.write_text(json.dumps(qis.jsonable(diagnostics), indent=2, sort_keys=True) + "\n")
saved_paths["diagnostics"] = diagnostics_path

smooth_qi = qis.diagnostic_float(diagnostics, "qi_smooth_total")
legacy_qi = qis.diagnostic_float(diagnostics, "qi_legacy_total")
mean_iota = qis.diagnostic_float(diagnostics, "mean_iota")
print("\nIndependent QI promotion gate:")
print(f"  smooth QI:       {smooth_qi:.6e}  (limit {QI_GATE_SMOOTH_MAX:.1e})")
print(f"  legacy QI:       {legacy_qi:.6e}  (limit {QI_GATE_LEGACY_MAX:.1e})")
print(f"  aspect ratio:    {qis.diagnostic_float(diagnostics, 'aspect'):.6g}  (target {TARGET_ASPECT:.3g})")
print(f"  abs(mean iota):  {abs(mean_iota):.6g}  (minimum {TARGET_ABS_IOTA_MIN:.3g})")
print(f"  mirror ratio:    {qis.diagnostic_float(diagnostics, 'qi_mirror_ratio_max'):.6g}  (target {MAX_MIRROR_RATIO:.3g})")
print(f"  mirror by surf:  {diagnostics.get('qi_mirror_ratio_by_surface')}")
print(f"  max elongation:  {qis.diagnostic_float(diagnostics, 'qi_max_elongation'):.6g}  (target {MAX_ELONGATION:.3g})")
print(f"  QI seed gate:    {bool(diagnostics['qi_seed_gate_passed'])}")
print(f"  full eng. gate:  {engineering_gate_passed}")
print(f"  rank score:      {diagnostics['qi_rank_score']:.6e}")
print(f"  failed gates:    {diagnostics['qi_gate_failures']}")
print(f"  diagnostics:     {diagnostics_path}")
for reason in diagnostics["qi_failure_reasons"]:
    print(f"    - {reason}")

if MAKE_PLOTS:
    # Plotting is explicit post-processing.  QI includes Boozer |B| contours
    # because VMEC-angle contour plots alone are not a QI promotion gate.
    plot_paths = {
        "boundary_comparison": vj.plot_3d_boundary_comparison(saved_paths["initial_wout"], saved_paths["final_wout"], outdir=OUTPUT_DIR),
        "bmag_contours": vj.plot_bmag_contours(saved_paths["initial_wout"], saved_paths["final_wout"], outdir=OUTPUT_DIR),
        "objective_history": vj.plot_objective_history(saved_paths["history"], outdir=OUTPUT_DIR),
        "boozer_bmag_initial": vj.plot_boozer_bmag_contours_from_state(
            raw_initial_run.state,
            static=raw_initial_run.static,
            indata=raw_initial_run.indata,
            signgs=raw_initial_run.signgs,
            outdir=OUTPUT_DIR,
            filename="boozer_bmag_initial.png",
            surfaces=(1.0,),
            mboz=_resolution_value(AUDIT_QI_RESOLUTION, "mboz", QI_OPTIONS.mboz),
            nboz=_resolution_value(AUDIT_QI_RESOLUTION, "nboz", QI_OPTIONS.nboz),
            title=f"{INPUT_FILE.name}: initial seed Boozer |B| contours on LCFS",
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

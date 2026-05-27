#!/usr/bin/env python
"""Quasi-isodynamic optimization with vmec_jax and booz_xform_jax.

This script follows the same editable workflow as the QA/QH/QP examples:
choose a VMEC input, optionally prepare a better optimization seed, assemble
objective tuples, solve, then save/plot/print from the returned result.
"""

from pathlib import Path
import json
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Problem parameters. Edit these directly, as in the SIMSOPT examples.
INPUT_FILE = DATA_DIR / "input.nfp2_QI"
# Other useful seeds:
# INPUT_FILE = DATA_DIR / "input.nfp1_QI"
# INPUT_FILE = DATA_DIR / "input.QI_stel_seed_3127"
# INPUT_FILE = DATA_DIR / "input.minimal_seed_nfp4"
OUTPUT_DIR = Path("results/qi_opt/ess/nfp2_qi")
MAX_MODE = 5
MIN_VMEC_MODE = max(6, MAX_MODE + 3)

# Seed preparation. Leave all disabled to optimize directly from INPUT_FILE.
USE_SIMPLE_SEED = False  # True writes an input.simple_seed from RBC00/RBC01/ZBS01.
SIMPLE_SEED_PERTURBATION = 1.0e-5
USE_TARGET_HELICITY_SEED = True  # Adds tiny QI/QP-like modes when missing.
TARGET_HELICITY_SEED_AMPLITUDE = 1.0e-5
USE_REFERENCE_FAMILY_SEED = False  # True scans a nearby reference-family basin first.
REFERENCE_INPUT_FILE = DATA_DIR / "input.nfp2_QI"
REFERENCE_LAMBDAS = (0.995, 1.0, 1.005)

# Optimizer parameters.
METHOD = "scipy_matrix_free"  # Try "scipy", "gauss_newton", "lbfgs_adjoint", or "scalar_trust".
SCIPY_TR_SOLVER = "lsmr"  # For METHOD="scipy": "lsmr" is memory-light; "exact" is dense.
SCIPY_LSMR_MAXITER = None  # None lets SciPy choose; set an int to cap LSMR iterations.
FTOL = 1.0e-5  # Relative cost-reduction tolerance for the outer optimizer.
GTOL = 1.0e-5  # Gradient optimality tolerance for the outer optimizer.
XTOL = 1.0e-6  # Step-size tolerance for the outer optimizer.
INNER_MAX_ITER = 450  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
INNER_FTOL = 1.0e-9  # Accepted-point VMEC tolerance; 0 uses FTOL from the input deck.
TRIAL_MAX_ITER = 450  # Trial-point VMEC iterations; 0 follows the accepted/input budget.
TRIAL_FTOL = 1.0e-9  # Trial-point VMEC tolerance; 0 follows accepted/input tolerance.
SOLVER_DEVICE = None  # None uses JAX default; set "cpu" or "gpu" to force one backend.
USE_ESS = True  # Set False for an unscaled trust-region solve.
ALPHA = 1.2  # ESS high-mode scaling strength.
USE_MODE_CONTINUATION = True
CONTINUATION_NFEV = 10
MAX_NFEV = 60
STAGE_MODE_POLICY = "lower"  # "lower" stages 1..MAX_MODE; "repeat" repeats only MAX_MODE.
STAGE_REPEATS = 3  # Used only for STAGE_MODE_POLICY="repeat".
STAGE_MODES = vj.qi_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
    repeats=STAGE_REPEATS,
    policy=STAGE_MODE_POLICY,
)
# Common alternatives:
# STAGE_MODES = [1, 1, 2, 2, 3, 3]
# MIRROR_RAMP_STAGES = ()
# STAGE_MODE_POLICY = "repeat"
# METHOD = "lbfgs_adjoint"
# USE_REFERENCE_FAMILY_SEED = True
# SOLVER_DEVICE = "gpu"

# Output controls.
SAVE_STAGE_INPUTS = True
SAVE_STAGE_WOUTS = False
MAKE_PLOTS = True

# Physics targets and weights. Tuple weights use SIMSOPT semantics:
# residual = sqrt(weight) * (objective - target).
TARGET_ASPECT = 5.0
TARGET_ABS_IOTA_MIN = 0.41
MAX_MIRROR_RATIO = 0.30
MIRROR_SURFACE_INDEX = None
MAX_ELONGATION = 10.0
SURFACES = np.linspace(0.1, 1.0, 6)
ASPECT_WEIGHT = 0.25
IOTA_FLOOR_WEIGHT = 200.0**2
QI_WEIGHT = 10.0
QI_CEILING_MAX = 2.0e-2
QI_CEILING_WEIGHT = 0.0
QI_CEILING_SMOOTH_PENALTY = 2.0e-3
MIRROR_WEIGHT = 20.0
ELONGATION_WEIGHT = 10.0
QI_GATE_SMOOTH_MAX = 2.0e-3
QI_GATE_LEGACY_MAX = 2.0e-3
JIT_BOOZ = True
OPT_QI_RESOLUTION = {"mboz": 18, "nboz": 18, "nphi": 151, "nalpha": 31, "n_bounce": 51}
AUDIT_QI_RESOLUTION = dict(OPT_QI_RESOLUTION)


vj.apply_qi_example_cli_overrides(globals())

if "MIRROR_RAMP_STAGES" not in globals():
    MIRROR_RAMP_STAGES = (
        {
            "name": "matrix_free_qi_mirror_cleanup",
            "max_nfev": MAX_NFEV,
            "stage_modes": tuple(STAGE_MODES),
            "method": METHOD,
            "use_mode_continuation": USE_MODE_CONTINUATION,
            "mirror_threshold": MAX_MIRROR_RATIO,
            "promotion_mirror_threshold": MAX_MIRROR_RATIO,
            "mirror_weight": MIRROR_WEIGHT,
            "elongation_weight": ELONGATION_WEIGHT,
            "qi_ceiling_weight": QI_CEILING_WEIGHT,
            "require_mirror_improvement": False,
            "require_engineering_gate": True,
        },
    )

QI_OPTIONS = vj.QuasiIsodynamicOptions(
    surfaces=SURFACES,
    mboz=OPT_QI_RESOLUTION["mboz"],
    nboz=OPT_QI_RESOLUTION["nboz"],
    nphi=OPT_QI_RESOLUTION["nphi"],
    nalpha=OPT_QI_RESOLUTION["nalpha"],
    n_bounce=OPT_QI_RESOLUTION["n_bounce"],
    include_bounce_endpoints=True,
    softness=2.0e-2,
    width_weight=1.0,
    branch_width_weight=0.5,
    branch_width_softness=2.0e-2,
    profile_weight=0.1,
    shuffle_profile_weight=1.0,
    shuffle_profile_softness=2.0e-2,
    weighted_shuffle_profile_weight=0.0,
    weighted_shuffle_profile_softness=2.0e-2,
    phimin=0.0,
    jit_booz=JIT_BOOZ,
)

QI_CONTEXT = vj.make_qi_optimization_context(
    alpha=ALPHA,
    continuation_nfev=CONTINUATION_NFEV,
    inner_max_iter=INNER_MAX_ITER,
    jit_booz=JIT_BOOZ,
    max_elongation=MAX_ELONGATION,
    max_mirror_ratio=MAX_MIRROR_RATIO,
    max_mode=MAX_MODE,
    max_nfev=MAX_NFEV,
    method=METHOD,
    min_vmec_mode=MIN_VMEC_MODE,
    mirror_surface_index=MIRROR_SURFACE_INDEX,
    mirror_weight=MIRROR_WEIGHT,
    opt_qi_resolution=OPT_QI_RESOLUTION,
    output_dir=OUTPUT_DIR,
    qi_gate_legacy_max=QI_GATE_LEGACY_MAX,
    qi_gate_smooth_max=QI_GATE_SMOOTH_MAX,
    qi_options=QI_OPTIONS,
    qi_weight=QI_WEIGHT,
    solver_device=SOLVER_DEVICE,
    stage_modes=STAGE_MODES,
    stage_repeats=STAGE_REPEATS,
    surfaces=SURFACES,
    target_abs_iota_min=TARGET_ABS_IOTA_MIN,
    target_aspect=TARGET_ASPECT,
    trial_ftol=TRIAL_FTOL,
    use_ess=USE_ESS,
    use_mode_continuation=USE_MODE_CONTINUATION,
)

# Seed input. These helpers write new input files under OUTPUT_DIR; the source
# VMEC input is never modified.
RAW_INPUT_FILE = INPUT_FILE
INPUT_FILE = vj.prepare_simple_omnigenity_seed_input(
    INPUT_FILE,
    OUTPUT_DIR,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    enabled=USE_SIMPLE_SEED,
    perturbation=SIMPLE_SEED_PERTURBATION,
)
INPUT_FILE = vj.run_target_helicity_seed_preconditioner(
    INPUT_FILE,
    OUTPUT_DIR,
    {
        "enabled": USE_TARGET_HELICITY_SEED,
        "terms": vj.target_helicity_seed_terms(
            max_mode=MAX_MODE,
            amplitude=TARGET_HELICITY_SEED_AMPLITUDE,
        ),
        "only_if_abs_below": 0.0,
    },
    ctx=QI_CONTEXT,
)
INPUT_FILE = vj.run_boundary_reference_preconditioner(
    INPUT_FILE,
    OUTPUT_DIR,
    {
        "enabled": USE_REFERENCE_FAMILY_SEED,
        "reference_input": REFERENCE_INPUT_FILE,
        "lambdas": REFERENCE_LAMBDAS,
        "keys": ("RBC", "ZBS", "RBS", "ZBC"),
        "max_mode": MAX_MODE,
        "max_iter": INNER_MAX_ITER,
        "target_aspect": TARGET_ASPECT,
        "abs_iota_min": TARGET_ABS_IOTA_MIN,
        "max_mirror_ratio": MAX_MIRROR_RATIO,
        "max_elongation": MAX_ELONGATION,
        "smooth_qi_max": QI_GATE_SMOOTH_MAX,
        "legacy_qi_max": QI_GATE_LEGACY_MAX,
        "diagnostic_qi_resolution": AUDIT_QI_RESOLUTION,
    },
    ctx=QI_CONTEXT,
)

def _stage_value(stage, key, default):
    """Return a stage override while keeping the top-level script readable."""

    return default if stage is None or key not in stage else stage[key]


def make_qi_problem(stage=None):
    """Construct the editable least-squares objective for one QI stage."""

    aspect = vj.AspectRatio()
    iota_floor = vj.AbsMeanIotaFloor(TARGET_ABS_IOTA_MIN)
    qi = vj.QuasiIsodynamicResidual(QI_OPTIONS)
    qi_ceiling = vj.QuasiIsodynamicResidualCeiling(
        maximum=float(_stage_value(stage, "qi_ceiling_max", QI_CEILING_MAX)),
        smooth_penalty=float(_stage_value(stage, "qi_ceiling_smooth_penalty", QI_CEILING_SMOOTH_PENALTY)),
        qi_options=QI_OPTIONS,
    )
    mirror = vj.VMECMirrorRatio(
        threshold=float(_stage_value(stage, "mirror_threshold", MAX_MIRROR_RATIO)),
        surfaces=QI_OPTIONS.surfaces,
        ntheta=96,
        nphi=96,
        surface_index=_stage_value(stage, "mirror_surface_index", MIRROR_SURFACE_INDEX),
        smooth_extrema=2.0e-2,
        smooth_penalty=2.0e-2,
    )
    elongation = vj.MaxElongation(
        threshold=float(_stage_value(stage, "max_elongation", MAX_ELONGATION)),
        ntheta=48,
        nphi=16,
        smooth_extrema=2.0e-2,
        smooth_penalty=2.0e-2,
    )
    objective_tuples = [
        (aspect.J, TARGET_ASPECT, float(_stage_value(stage, "aspect_weight", ASPECT_WEIGHT))),
        (iota_floor.J, 0.0, float(_stage_value(stage, "iota_floor_weight", IOTA_FLOOR_WEIGHT))),
        (qi.J, 0.0, float(_stage_value(stage, "qi_weight", QI_WEIGHT))),
        (mirror.J, 0.0, float(_stage_value(stage, "mirror_weight", MIRROR_WEIGHT))),
        (elongation.J, 0.0, float(_stage_value(stage, "elongation_weight", ELONGATION_WEIGHT))),
        # Optional:
        # (qi_ceiling.J, 0.0, QI_CEILING_WEIGHT),
        # (vj.LgradB(threshold=0.30, smooth_penalty=1.0e-3).J, 0.0, 0.001),
        # (vj.MagneticWell(minimum=0.0).J, 0.0, 1.0),
        # (vj.DMerc(minimum=0.0, softness=1.0e-3).J, 0.0, DMERC_WEIGHT),
    ]
    qi_ceiling_weight = float(_stage_value(stage, "qi_ceiling_weight", QI_CEILING_WEIGHT))
    if stage is None and QI_CEILING_WEIGHT > 0.0:
        objective_tuples.append((qi_ceiling.J, 0.0, QI_CEILING_WEIGHT))
    elif qi_ceiling_weight > 0.0:
        objective_tuples.append((qi_ceiling.J, 0.0, qi_ceiling_weight))
    return vj.LeastSquaresProblem.from_tuples(objective_tuples)


problem = make_qi_problem()

print("\nAssembled least-squares problem:")
print(f"  objectives: {', '.join(problem.objective_names)}")
print(f"  scalar terms: {problem.scalar_objective_names}")
print(f"  QI terms: {problem.qi_objective_names}")
print("\nQI run controls:")
print(f"  raw input:      {RAW_INPUT_FILE}")
print(f"  active input:   {INPUT_FILE}")
print(f"  output dir:     {OUTPUT_DIR}")
print(f"  max_mode:       {MAX_MODE}")
print(f"  stage modes:    {STAGE_MODES}")
print(f"  ESS:            {USE_ESS} (alpha={ALPHA})")
print(f"  target aspect:  {TARGET_ASPECT}")
print(f"  min |iota|:     {TARGET_ABS_IOTA_MIN}")
print(f"  mirror target:  {MAX_MIRROR_RATIO}")
print(f"  max elongation: {MAX_ELONGATION}")


def solve_qi_stage(
    input_file,
    output_dir,
    stage_problem,
    *,
    max_nfev=MAX_NFEV,
    label=f"QI optimization (max_mode={MAX_MODE})",
    stage_modes=STAGE_MODES,
    method=METHOD,
    use_mode_continuation=USE_MODE_CONTINUATION,
    scalar_step_bound=None,
    lbfgs_step_bound=None,
    save_final_outputs=False,
):
    """Run one editable QI stage using the objective problem supplied above."""

    vmec = vj.FixedBoundaryVMEC.from_input(
        input_file,
        max_mode=MAX_MODE,
        min_vmec_mode=MIN_VMEC_MODE,
        output_dir=output_dir,
        project_input_boundary_to_max_mode=True,
    )
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
        scipy_tr_solver=SCIPY_TR_SOLVER,
        scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
        lbfgs_step_bound=lbfgs_step_bound,
        scalar_step_bound=scalar_step_bound,
        save_stage_inputs=SAVE_STAGE_INPUTS,
        save_stage_wouts=SAVE_STAGE_WOUTS,
        save_final_outputs=save_final_outputs,
    )


result, promotion_log = vj.run_qi_stage_policy(
    INPUT_FILE,
    OUTPUT_DIR,
    solve_qi_stage=solve_qi_stage,
    make_qi_problem=make_qi_problem,
    boundary_reference_preconditioner={"enabled": False},
    mirror_ramp_stages=MIRROR_RAMP_STAGES,
    ctx=QI_CONTEXT,
)

history = result.history
objective_history = result.objective_history
timing = result.timing_summary
result_summary = result.summary
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
saved_paths = {
    "initial_input": OUTPUT_DIR / "input.initial",
    "final_input": OUTPUT_DIR / "input.final",
    "initial_wout": OUTPUT_DIR / "wout_initial.nc",
    "final_wout": OUTPUT_DIR / "wout_final.nc",
    "history": OUTPUT_DIR / "history.json",
}
print("\nRunning the raw input deck once for initial comparison plots ...")
raw_initial_run = vj.save_raw_seed_initial_artifacts(
    RAW_INPUT_FILE,
    saved_paths["initial_input"],
    saved_paths["initial_wout"],
    ctx=QI_CONTEXT,
)
result.final_optimizer.save_input(saved_paths["final_input"], result.final_params)
result.final_optimizer.save_wout(saved_paths["final_wout"], result.final_params, state=result.final_state)
result.final_optimizer.save_history(saved_paths["history"], result.final_result)
if promotion_log:
    (OUTPUT_DIR / "mirror_ramp_promotion_log.json").write_text(json.dumps(promotion_log, indent=2) + "\n")

diagnostic_options = vj.QIDiagnosticOptions(
    surfaces=SURFACES,
    mboz=AUDIT_QI_RESOLUTION["mboz"],
    nboz=AUDIT_QI_RESOLUTION["nboz"],
    nphi=AUDIT_QI_RESOLUTION["nphi"],
    nalpha=AUDIT_QI_RESOLUTION["nalpha"],
    n_bounce=AUDIT_QI_RESOLUTION["n_bounce"],
    include_bounce_endpoints=QI_OPTIONS.include_bounce_endpoints,
    softness=QI_OPTIONS.softness,
    width_weight=QI_OPTIONS.width_weight,
    branch_width_weight=QI_OPTIONS.branch_width_weight,
    branch_width_softness=QI_OPTIONS.branch_width_softness,
    profile_weight=QI_OPTIONS.profile_weight,
    shuffle_profile_weight=QI_OPTIONS.shuffle_profile_weight,
    shuffle_profile_softness=QI_OPTIONS.shuffle_profile_softness,
    weighted_shuffle_profile_weight=QI_OPTIONS.weighted_shuffle_profile_weight,
    weighted_shuffle_profile_softness=QI_OPTIONS.weighted_shuffle_profile_softness,
    phimin=float(QI_OPTIONS.phimin),
    mirror_threshold=MAX_MIRROR_RATIO,
    elongation_threshold=MAX_ELONGATION,
)
diagnostics = vj.qi_diagnostics_from_state(
    state=result.final_state,
    static=result.final_optimizer.static,
    indata=result.final_optimizer.indata,
    signgs=result.final_optimizer.signgs,
    surfaces=SURFACES,
    options=diagnostic_options,
)
diagnostics = vj.annotate_qi_seed_suitability(
    diagnostics,
    targets=vj.QISeedSuitabilityTargets(
        smooth_qi_max=QI_GATE_SMOOTH_MAX,
        legacy_qi_max=QI_GATE_LEGACY_MAX,
        target_aspect=TARGET_ASPECT,
        abs_iota_min=TARGET_ABS_IOTA_MIN,
        mirror_ratio_max=MAX_MIRROR_RATIO,
        max_elongation=MAX_ELONGATION,
    ),
)
diagnostics_path = OUTPUT_DIR / "diagnostics.json"
diagnostics_path.write_text(json.dumps(vj.jsonable(diagnostics), indent=2, sort_keys=True) + "\n")

print("\nFinal diagnostics from result.history:")
print(f"  stages:           {result_summary['stage_modes']}")
print(f"  aspect ratio:     {history['aspect_final']:.6g}")
print(f"  mean iota:        {history['iota_final']:.6g}")
print(f"  QI objective:     {history['qs_final']:.6e}")
print(f"  total objective:  {history['objective_final']:.6e}")
print(f"  wall time:        {timing['total_wall_time_s']:.2f} s")
print(f"  objective samples: {objective_history[:5]} ... {objective_history[-3:]}")

print("\nIndependent QI diagnostics:")
print(f"  smooth QI:       {vj.diagnostic_float(diagnostics, 'qi_smooth_total'):.6e}")
print(f"  legacy QI:       {vj.diagnostic_float(diagnostics, 'qi_legacy_total'):.6e}")
print(f"  mirror ratio:    {vj.diagnostic_float(diagnostics, 'qi_mirror_ratio_max'):.6g}")
print(f"  max elongation:  {vj.diagnostic_float(diagnostics, 'qi_max_elongation'):.6g}")
print(f"  diagnostics:     {diagnostics_path}")

print("\nFiles saved for raw-seed/final comparison:")
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

if MAKE_PLOTS:
    print("\nGenerating initial-vs-final LCFS |B| contour comparison in Boozer coordinates:")
    plot_paths = {
        "boundary_comparison": vj.plot_3d_boundary_comparison(
            saved_paths["initial_wout"],
            saved_paths["final_wout"],
            outdir=OUTPUT_DIR,
        ),
        "initial_vs_final_lcfs_boozer_bmag_contours": vj.plot_boozer_lcfs_bmag_comparison(
            saved_paths["initial_wout"],
            saved_paths["final_wout"],
            outdir=OUTPUT_DIR,
        ),
        "objective_history": vj.plot_objective_history(saved_paths["history"], outdir=OUTPUT_DIR),
    }
    print("\nPlot files selected by this script:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

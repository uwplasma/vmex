#!/usr/bin/env python
"""Quasi-isodynamic optimization with vmec_jax and booz_xform_jax."""

from pathlib import Path
import json
import os
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from tools.diagnostics.qi_basin_survey import (
    SurveyTargets,
    generate_basin_candidates,
    rank_candidate_records,
    write_csv,
)
from tools.diagnostics.qi_landscape_scan import build_stage as build_diagnostic_stage
from vmec_jax._compat import enable_x64
from vmec_jax.optimization import boundary_param_names, create_x_scale
from vmec_jax.qi_diagnostics import QISeedSuitabilityTargets, annotate_qi_seed_suitability


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_QI_TARGET_ASPECT = 10.0


def _diagnostic_float(record, key):
    value = record.get(key)
    return float(value) if value is not None else float("nan")


def _finite_or_inf(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return out if np.isfinite(out) else float("inf")


def _finite_or_none(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None

# Seed cases.  Pick one case by changing RUN_CASE or setting
# VMEC_JAX_QI_RUN_CASE; add another dictionary entry, or set VMEC_JAX_QI_INPUT,
# to use an external VMEC input deck.  The NFP is taken from the VMEC input
# file, so the same script can run the bundled NFP=2 QI seed, the stellarator
# seed, or the NFP=4 QH warm start without hard-coding field period count.
QI_CASES = {
    "nfp2_qi": {
        "case_goal": "default NFP=2 mirror-aware QI lane",
        "input_file": DATA_DIR / "input.nfp2_QI",
        "output_dir": Path("results/qi_opt/ess/nfp2_qi"),
        "max_mode": 3,
        "min_vmec_mode": 6,
        "method": "scipy_matrix_free",
        "use_mode_continuation": True,
        "stage_repeats": 1,
        "max_nfev": 10,
        "target_aspect": DEFAULT_QI_TARGET_ASPECT,
        "target_abs_iota_min": 0.41,
        "max_elongation": 8.2,
        "mirror_threshold": 0.30,
        "mirror_surface_index": None,
        "qi_ceiling_max": 2.0e-2,
        "qi_ceiling_smooth_penalty": 2.0e-3,
        "branch_width_weight": 0.5,
        "weighted_shuffle_profile_weight": 0.0,
        "phimin": 0.0,
        "mirror_weight": 20.0,
        "elongation_weight": 10.0,
        "qi_ceiling_weight": 0.0,
        "shuffle_profile_nphi_out": None,
        # Guarded mirror-aware policy.  This matrix-free lane is the current
        # validated default: it obtains low smooth/legacy QI, nonzero transform,
        # and an all-surface mirror ratio below 0.30 from the bundled NFP=2 seed.
        "mirror_ramp_stages": (
            {
                "name": "matrix_free_mirror030",
                "max_nfev": 10,
                "stage_repeats": 1,
                "method": "scipy_matrix_free",
                "mirror_threshold": 0.21,
                "promotion_mirror_threshold": 0.30,
                "mirror_surface_index": None,
                "mirror_weight": 20.0,
                "elongation_weight": 10.0,
                "qi_ceiling_max": 2.0e-2,
                "qi_ceiling_weight": 0.0,
                "require_mirror_improvement": False,
                "require_engineering_gate": True,
            },
        ),
    },
    "qi_stel_seed_3127": {
        "case_goal": "far-seed staged QI robustness lane with explicit QI/iota/engineering gates",
        "input_file": DATA_DIR / "input.QI_stel_seed_3127",
        "output_dir": Path("results/qi_opt/ess/qi_stel_seed_3127"),
        "max_mode": 3,
        "min_vmec_mode": 6,
        "use_mode_continuation": False,
        "stage_repeats": 1,
        "max_nfev": 8,
        "target_aspect": DEFAULT_QI_TARGET_ASPECT,
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
        "optimization_qi_resolution": {"mboz": 5, "nboz": 5, "nphi": 31, "nalpha": 7, "n_bounce": 9},
        "audit_qi_resolution": {"mboz": 18, "nboz": 18, "nphi": 151, "nalpha": 31, "n_bounce": 51},
        "basin_prefilter": {
            "enabled": True,
            "radii": (0.025, 0.05, 0.1),
            "directions": ("axes", "rademacher"),
            "axis_count": 6,
            "n_random": 4,
            "max_candidates": 40,
            "trial_max_iter": 30,
            "trial_ftol": 1.0e-8,
            "top_k": 8,
            "iota_gap_weight": 120.0,
            "qi_weight": 1.0,
            "mirror_weight": 0.10,
            "elongation_weight": 0.05,
            "save_candidate_inputs": True,
        },
        "boozer_target_wout": None,
        "boozer_target_weight": 0.0,
        "boozer_target_normalize": True,
        "boozer_target_include_b00": False,
        "mirror_weight": 0.0,
        "elongation_weight": 0.0,
        "use_augmented_lagrangian_constraints": False,
        "al_mirror_multiplier": 0.0,
        "al_mirror_penalty": 1.0,
        "al_elongation_multiplier": 0.0,
        "al_elongation_penalty": 1.0,
        "qi_ceiling_weight": 0.0,
        "shuffle_profile_nphi_out": None,
        # Far seeds need more than one scalar least-squares call.  The policy
        # below first finds a QI-ish basin, then ramps transform with a QI
        # ceiling, and only then applies mirror/elongation cleanup.  Stages are
        # promoted by independent diagnostics, not by optimizer success flags.
        "mirror_ramp_stages": (
            {
                "name": "prefiltered_qi_iota_cleanup",
                "max_nfev": 8,
                "stage_repeats": 1,
                "stage_modes": (3,),
                "method": "scipy_matrix_free",
                "use_mode_continuation": False,
                "aspect_weight": 0.05,
                "iota_floor_weight": 50.0**2,
                "qi_weight": 250.0,
                "qi_ceiling_max": 6.0e-3,
                "qi_ceiling_weight": 2500.0,
                "mirror_weight": 0.0,
                "elongation_weight": 0.0,
                "require_seed_gate": False,
                "require_mirror_improvement": False,
                "require_engineering_gate": False,
                "accept_if_rank_improves": True,
            },
            {
                "name": "iota_ramp_qi_ceiling",
                "max_nfev": 4,
                "stage_repeats": 1,
                "stage_modes": (3, 3),
                "method": "scalar_trust",
                "scalar_step_bound": 5.0e-3,
                "aspect_weight": 0.05,
                "iota_floor_weight": 50.0**2,
                "qi_weight": 250.0,
                "qi_ceiling_max": 3.0e-3,
                "qi_ceiling_weight": 2500.0,
                "mirror_weight": 0.0,
                "elongation_weight": 0.0,
                "require_seed_gate": False,
                "require_mirror_improvement": False,
                "require_engineering_gate": False,
                "accept_if_rank_improves": True,
                "accept_if_iota_improves": True,
                "iota_improvement_min": 5.0e-2,
                "qi_relax_for_iota": 3.0,
            },
            {
                "name": "iota_ramp_large_step",
                "max_nfev": 6,
                "stage_repeats": 1,
                "stage_modes": (3,),
                "method": "scalar_trust",
                "scalar_step_bound": 1.0e-2,
                "aspect_weight": 0.05,
                "iota_floor_weight": 100.0**2,
                "qi_weight": 250.0,
                "qi_ceiling_max": 6.0e-3,
                "qi_ceiling_weight": 2500.0,
                "mirror_weight": 0.0,
                "elongation_weight": 0.0,
                "require_seed_gate": False,
                "require_mirror_improvement": False,
                "require_engineering_gate": False,
                "accept_if_rank_improves": False,
                "accept_if_iota_improves": True,
                "iota_improvement_min": 5.0e-2,
                "qi_relax_for_iota": 4.0,
            },
            {
                "name": "mirror_elongation_guard",
                "max_nfev": 4,
                "stage_repeats": 1,
                "stage_modes": (3,),
                "method": "scalar_trust",
                "scalar_step_bound": 3.0e-3,
                "aspect_weight": 0.05,
                "iota_floor_weight": 50.0**2,
                "qi_weight": 250.0,
                "qi_ceiling_max": 5.0e-3,
                "qi_ceiling_weight": 2500.0,
                "mirror_threshold": 0.35,
                "promotion_mirror_threshold": 0.35,
                "mirror_weight": 2.0,
                "elongation_weight": 1.0,
                "require_seed_gate": True,
                "require_mirror_improvement": False,
                "require_engineering_gate": False,
                "accept_if_rank_improves": False,
            },
        ),
    },
    "nfp4_qh_warm_to_qi": {
        "case_goal": "NFP=4 QH-to-QI staged stress test; audit before promotion",
        "input_file": DATA_DIR / "input.nfp4_QH_warm_start",
        "output_dir": Path("results/qi_opt/ess/nfp4_qh_warm_to_qi"),
        "max_mode": 3,
        "min_vmec_mode": 6,
        "use_mode_continuation": True,
        "stage_repeats": 3,
        "max_nfev": 10,
        "target_aspect": DEFAULT_QI_TARGET_ASPECT,
        "target_abs_iota_min": 0.41,
        "mirror_threshold": 0.21,
        "mirror_surface_index": None,
        "qi_ceiling_max": 2.0e-2,
        "qi_ceiling_smooth_penalty": 2.0e-3,
        "branch_width_weight": 0.5,
        "weighted_shuffle_profile_weight": 0.0,
        "phimin": 0.0,
        "optimization_qi_resolution": {"mboz": 5, "nboz": 5, "nphi": 31, "nalpha": 7, "n_bounce": 9},
        "audit_qi_resolution": {"mboz": 18, "nboz": 18, "nphi": 151, "nalpha": 31, "n_bounce": 51},
        "boozer_target_wout": None,
        "boozer_target_weight": 0.0,
        "boozer_target_normalize": True,
        "boozer_target_include_b00": False,
        "mirror_weight": 0.0,
        "elongation_weight": 0.0,
        "qi_ceiling_weight": 0.0,
        "shuffle_profile_nphi_out": None,
        "mirror_ramp_stages": (
            {
                "name": "qh_warm_qi_repeat112233",
                "max_nfev": 4,
                "stage_repeats": 1,
                "stage_modes": (1, 2, 3),
                "method": "scipy_matrix_free",
                "use_mode_continuation": True,
                "aspect_weight": 0.05,
                "iota_floor_weight": 50.0**2,
                "qi_weight": 250.0,
                "qi_ceiling_weight": 0.0,
                "mirror_weight": 0.0,
                "elongation_weight": 0.0,
                "require_seed_gate": False,
                "require_mirror_improvement": False,
                "require_engineering_gate": False,
                "accept_if_rank_improves": True,
            },
            {
                "name": "qh_warm_engineering_guard",
                "max_nfev": 4,
                "stage_repeats": 1,
                "stage_modes": (3,),
                "method": "scalar_trust",
                "scalar_step_bound": 3.0e-3,
                "aspect_weight": 0.05,
                "iota_floor_weight": 50.0**2,
                "qi_weight": 250.0,
                "qi_ceiling_max": 5.0e-3,
                "qi_ceiling_weight": 2500.0,
                "mirror_threshold": 0.35,
                "promotion_mirror_threshold": 0.35,
                "mirror_weight": 2.0,
                "elongation_weight": 1.0,
                "require_seed_gate": True,
                "require_mirror_improvement": False,
                "require_engineering_gate": False,
                "accept_if_rank_improves": False,
            },
        ),
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
    #     "target_aspect": DEFAULT_QI_TARGET_ASPECT,
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
    #     "mirror_ramp_stages": (),
    # },
}

RUN_CASE = "nfp2_qi"  # Try "qi_stel_seed_3127" or "nfp4_qh_warm_to_qi".
_EXTERNAL_INPUT = os.environ.get("VMEC_JAX_QI_INPUT")
if _EXTERNAL_INPUT:
    _external_label = os.environ.get(
        "VMEC_JAX_QI_RUN_CASE",
        os.environ.get("VMEC_JAX_QI_LABEL", Path(_EXTERNAL_INPUT).name.replace("input.", "")),
    )
    # External inputs use the far-seed robustness policy by default: first
    # establish a QI+iota basin, then add guarded engineering cleanup later.
    QI_CASES[_external_label] = {
        **QI_CASES["qi_stel_seed_3127"],
        "case_goal": "external VMEC input using the far-seed QI+iota robustness policy",
        "input_file": Path(_EXTERNAL_INPUT).expanduser(),
        "output_dir": Path(
            os.environ.get("VMEC_JAX_QI_OUTPUT_DIR", f"results/qi_opt/ess/{_external_label}")
        ).expanduser(),
    }
    RUN_CASE = _external_label
else:
    RUN_CASE = os.environ.get("VMEC_JAX_QI_RUN_CASE", RUN_CASE)
if RUN_CASE not in QI_CASES:
    raise KeyError(f"Unknown QI RUN_CASE {RUN_CASE!r}; available cases: {sorted(QI_CASES)}")
CASE = QI_CASES[RUN_CASE]
if os.environ.get("VMEC_JAX_QI_OUTPUT_DIR"):
    CASE = {
        **CASE,
        "output_dir": Path(os.environ["VMEC_JAX_QI_OUTPUT_DIR"]).expanduser(),
    }

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
MIRROR_RAMP_STAGES = tuple(CASE.get("mirror_ramp_stages", ()))

# Optimizer parameters.
METHOD = CASE.get("method", "scipy")  # Try also "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
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
QI_GATE_SMOOTH_MAX = 2.0e-3
QI_GATE_LEGACY_MAX = 1.0e-3
JIT_BOOZ = bool(CASE.get("jit_booz", True))  # Faster QI/Boozer path on current CPU/GPU diagnostics.


def _resolution_value(resolution, key, default):
    return int(dict(resolution).get(key, default))


OPT_QI_RESOLUTION = dict(CASE.get("optimization_qi_resolution", {}))
AUDIT_QI_RESOLUTION = {**OPT_QI_RESOLUTION, **dict(CASE.get("audit_qi_resolution", {}))}

# Boozer transform and smooth-QI residual resolution.
QI_OPTIONS = vj.QuasiIsodynamicOptions(
    surfaces=SURFACES,
    mboz=_resolution_value(OPT_QI_RESOLUTION, "mboz", 18),
    nboz=_resolution_value(OPT_QI_RESOLUTION, "nboz", 18),
    nphi=_resolution_value(OPT_QI_RESOLUTION, "nphi", 151),
    nalpha=_resolution_value(OPT_QI_RESOLUTION, "nalpha", 31),
    n_bounce=_resolution_value(OPT_QI_RESOLUTION, "n_bounce", 51),
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

qi_only_objective_tuples = [
    # Optional diagnostic-only pre-refinement.  Do not promote this stage by
    # itself: QI-only can converge to a low-QI branch with near-zero iota.
    (qi.J, 0.0, QI_WEIGHT),
]
# Optional terms users can append in the same tuple style:
# objective_tuples.append((vj.LgradB(threshold=0.30, smooth_penalty=1.0e-3).J, 0.0, 0.001))
# objective_tuples.append((vj.MagneticWell(minimum=0.0).J, 0.0, 1.0))
# objective_tuples.append((vj.DMerc(minimum=0.0, softness=1.0e-3).J, 0.0, DMERC_WEIGHT))
# Finite-beta examples show VolavgB, BetaTotal, JDotB, RedlBootstrapMismatch,
# and profile-current targets without making this QI teaching script longer.
qi_only_problem = vj.LeastSquaresProblem.from_tuples(qi_only_objective_tuples)


def _stage_value(stage, key, default):
    return stage.get(key, CASE.get(key, default))


def make_qi_problem(stage=None):
    """Assemble the QI objective tuples for one optimization stage."""

    stage = {} if stage is None else dict(stage)
    aspect_weight = float(_stage_value(stage, "aspect_weight", ASPECT_WEIGHT))
    iota_floor_weight = float(_stage_value(stage, "iota_floor_weight", IOTA_FLOOR_WEIGHT))
    qi_weight = float(_stage_value(stage, "qi_weight", QI_WEIGHT))
    qi_ceiling_weight = float(_stage_value(stage, "qi_ceiling_weight", QI_CEILING_WEIGHT))
    mirror_weight = float(_stage_value(stage, "mirror_weight", MIRROR_WEIGHT))
    elongation_weight = float(_stage_value(stage, "elongation_weight", ELONGATION_WEIGHT))
    use_al_constraints = bool(_stage_value(stage, "use_augmented_lagrangian_constraints", False))
    mirror_threshold = float(_stage_value(stage, "mirror_threshold", MAX_MIRROR_RATIO))
    mirror_surface_index = _stage_value(stage, "mirror_surface_index", MIRROR_SURFACE_INDEX)
    qi_ceiling_max = float(_stage_value(stage, "qi_ceiling_max", QI_CEILING_MAX))
    qi_ceiling_smooth_penalty = float(
        _stage_value(stage, "qi_ceiling_smooth_penalty", QI_CEILING_SMOOTH_PENALTY)
    )

    qi_ceiling = vj.QuasiIsodynamicResidualCeiling(
        maximum=qi_ceiling_max,
        smooth_penalty=qi_ceiling_smooth_penalty,
        qi_options=QI_OPTIONS,
    )
    mirror = vj.MirrorRatio(
        threshold=mirror_threshold,
        ntheta=96,
        nphi=96,
        # None means "all QI surfaces", matching QIDiagnosticOptions'
        # mirror-ratio gate.  Set an integer to optimize one surface only.
        surface_index=mirror_surface_index,
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
    if use_al_constraints:
        # Augmented-Lagrangian constraints are useful when plain high weights
        # on mirror/elongation lower those engineering metrics by destroying QI.
        # Multipliers should be updated only from exact accepted diagnostics
        # between explicit stages, never from SciPy trial residuals.
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
        (aspect.J, TARGET_ASPECT, aspect_weight),
        (iota_floor.J, 0.0, iota_floor_weight),
        (qi.J, 0.0, qi_weight),
    ]
    if boozer_target is not None:
        # Homotopy steering for far seeds: match a solved same-NFP QI Boozer
        # spectrum while the QI/iota terms keep the solve in a useful basin.
        objective_tuples.insert(2, (boozer_target.J, 0.0, BOOZER_TARGET_WEIGHT))
    if qi_ceiling_weight > 0.0:
        # Soft-wall QI guard: mirror/elongation cleanup may trade scalar
        # objective against QI.  This term is inactive below qi_ceiling_max and
        # grows once a trial leaves the accepted QI basin.
        objective_tuples.append((qi_ceiling.J, 0.0, qi_ceiling_weight))
    if mirror_weight > 0.0:
        # Boozer mirror cleanup.  Keep QI/iota terms active so this engineering
        # target cannot silently destroy omnigenity.
        objective_tuples.append((mirror.J, 0.0, mirror_weight))
    if elongation_weight > 0.0:
        objective_tuples.append((elongation.J, 0.0, elongation_weight))
    return vj.LeastSquaresProblem.from_tuples(objective_tuples)


problem = make_qi_problem()

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
print(
    "  QI opt grid:     "
    f"mboz={QI_OPTIONS.mboz}, nboz={QI_OPTIONS.nboz}, "
    f"nphi={QI_OPTIONS.nphi}, nalpha={QI_OPTIONS.nalpha}, n_bounce={QI_OPTIONS.n_bounce}"
)
print(f"  QI audit grid:   {AUDIT_QI_RESOLUTION or 'same as optimization'}")
print(f"  JIT Boozer path: {QI_OPTIONS.jit_booz}")
print(f"  Boozer target:   {BOOZER_TARGET_WOUT} (weight={BOOZER_TARGET_WEIGHT})")
print(f"  mirror target:   {MAX_MIRROR_RATIO} (surface={MIRROR_SURFACE_INDEX})")
print(f"  mirror weight:   {MIRROR_WEIGHT}")
print(f"  elongation wt:   {ELONGATION_WEIGHT}")
print(f"  AL constraints:  {CASE.get('use_augmented_lagrangian_constraints', False)}")
print(f"  QI ceiling:      {QI_CEILING_MAX} (weight={QI_CEILING_WEIGHT})")
if MIRROR_RAMP_STAGES:
    print("  mirror ramp:     guarded staged cleanup enabled")
    for index, stage in enumerate(MIRROR_RAMP_STAGES, start=1):
        print(
            f"    {index}: {stage['name']} "
            f"mirror<={stage.get('mirror_threshold', MAX_MIRROR_RATIO)} "
            f"w={stage.get('mirror_weight', MIRROR_WEIGHT)} "
            f"nfev={stage.get('max_nfev', MAX_NFEV)} "
            f"repeats={stage.get('stage_repeats', STAGE_REPEATS)}"
        )

if MIRROR_RAMP_STAGES:
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


def basin_prefilter_score(metrics, targets, config):
    """Rank prefilter candidates by QI/iota first, engineering second."""

    smooth = _finite_or_inf(metrics.get("qi_smooth_total"))
    legacy = _finite_or_inf(metrics.get("qi_legacy_total"))
    mirror = _finite_or_inf(metrics.get("qi_mirror_ratio_max"))
    elongation = _finite_or_inf(metrics.get("qi_max_elongation"))
    iota = abs(float(metrics.get("mean_iota") or 0.0))
    aspect = _finite_or_inf(metrics.get("aspect"))
    smooth_score = smooth / max(float(targets.smooth_qi_max), 1.0e-16)
    legacy_score = legacy / max(float(targets.legacy_qi_max), 1.0e-16)
    iota_score = max(0.0, float(targets.abs_iota_min) - iota) / max(float(targets.abs_iota_min), 1.0e-16)
    mirror_score = max(0.0, mirror - float(targets.mirror_ratio_max)) / max(float(targets.mirror_ratio_max), 1.0e-16)
    elongation_score = max(0.0, elongation - float(targets.max_elongation)) / max(float(targets.max_elongation), 1.0e-16)
    aspect_score = abs(aspect - float(targets.target_aspect)) / max(float(targets.target_aspect), 1.0e-16)
    qi_weight = float(config.get("qi_weight", 1.0))
    iota_weight = float(config.get("iota_gap_weight", 3.0))
    mirror_weight = float(config.get("mirror_weight", 0.25))
    elongation_weight = float(config.get("elongation_weight", 0.1))
    aspect_weight = float(config.get("aspect_weight", 0.1))
    return float(
        qi_weight * (smooth_score + legacy_score)
        + iota_weight * iota_score
        + mirror_weight * mirror_score
        + elongation_weight * elongation_score
        + aspect_weight * aspect_score
    )


def make_basin_prefilter_options(config):
    return vj.QIDiagnosticOptions(
        surfaces=SURFACES,
        mboz=_resolution_value(OPT_QI_RESOLUTION, "mboz", QI_OPTIONS.mboz),
        nboz=_resolution_value(OPT_QI_RESOLUTION, "nboz", QI_OPTIONS.nboz),
        nphi=_resolution_value(OPT_QI_RESOLUTION, "nphi", QI_OPTIONS.nphi),
        nalpha=_resolution_value(OPT_QI_RESOLUTION, "nalpha", QI_OPTIONS.nalpha),
        n_bounce=_resolution_value(OPT_QI_RESOLUTION, "n_bounce", QI_OPTIONS.n_bounce),
        include_bounce_endpoints=QI_OPTIONS.include_bounce_endpoints,
        phimin=float(QI_OPTIONS.phimin),
        jit_booz=JIT_BOOZ,
        mirror_threshold=float(config.get("mirror_threshold", MAX_MIRROR_RATIO)),
        mirror_ntheta=int(config.get("mirror_ntheta", 32)),
        mirror_nphi=int(config.get("mirror_nphi", 32)),
        mirror_surface_index=config.get("mirror_surface_index", MIRROR_SURFACE_INDEX),
        elongation_threshold=float(config.get("max_elongation", MAX_ELONGATION)),
        elongation_ntheta=int(config.get("elongation_ntheta", 24)),
        elongation_nphi=int(config.get("elongation_nphi", 8)),
    )


def run_basin_prefilter(input_file, output_dir, config):
    """Run a bounded large-step prefilter and return the selected input deck."""

    if not bool(config.get("enabled", False)):
        return Path(input_file)
    survey_dir = Path(output_dir) / "basin_prefilter"
    survey_dir.mkdir(parents=True, exist_ok=True)
    stage = build_diagnostic_stage(
        input_path=Path(input_file),
        max_mode=MAX_MODE,
        min_vmec_mode=MIN_VMEC_MODE,
        include=("rc", "zs"),
        fix=("rc00",),
        project_input_boundary_to_max_mode=True,
        inner_max_iter=int(config.get("inner_max_iter", 30)),
        inner_ftol=float(config.get("inner_ftol", 1.0e-8)),
        trial_max_iter=int(config.get("trial_max_iter", 30)),
        trial_ftol=float(config.get("trial_ftol", 1.0e-8)),
        solver_device=SOLVER_DEVICE,
    )
    names = boundary_param_names(stage.specs)
    x_scale = create_x_scale(stage.specs, alpha=float(config.get("alpha", ALPHA)))
    candidates = generate_basin_candidates(
        names=names,
        x_scale=x_scale,
        radii=tuple(float(radius) for radius in config.get("radii", (0.025, 0.05, 0.1))),
        n_random=int(config.get("n_random", 4)),
        rng_seed=int(config.get("rng_seed", 20260515)),
        axis_count=int(config.get("axis_count", 6)),
        directions=tuple(config.get("directions", ("axes", "rademacher"))),
        include_zero=True,
    )[: max(1, int(config.get("max_candidates", 24)))]
    options = make_basin_prefilter_options(config)
    targets = SurveyTargets(
        smooth_qi_max=QI_GATE_SMOOTH_MAX,
        legacy_qi_max=QI_GATE_LEGACY_MAX,
        mirror_ratio_max=float(config.get("mirror_threshold", MAX_MIRROR_RATIO)),
        max_elongation=float(config.get("max_elongation", MAX_ELONGATION)),
        abs_iota_min=TARGET_ABS_IOTA_MIN,
        target_aspect=TARGET_ASPECT,
        aspect_tolerance=2.0,
    )
    records = []
    for candidate in candidates:
        record = candidate.as_record(names)
        try:
            params = np.asarray(candidate.params, dtype=float)
            state = stage.optimizer._solve_forward(params, trial=True)
            diagnostics = vj.qi_diagnostics_from_state(
                state=state,
                static=stage.ctx.static,
                indata=stage.ctx.indata,
                signgs=stage.ctx.signgs,
                surfaces=options.surfaces,
                options=options,
                flux_local=stage.ctx.flux,
                prof_local={"pressure": stage.ctx.pressure},
                pressure_local=stage.ctx.pressure,
            )
            metrics = {
                "qi_smooth_total": _finite_or_none(diagnostics.get("qi_smooth_total")),
                "qi_legacy_total": _finite_or_none(diagnostics.get("qi_legacy_total")),
                "qi_mirror_ratio_max": _finite_or_none(diagnostics.get("qi_mirror_ratio_max")),
                "qi_max_elongation": _finite_or_none(diagnostics.get("qi_max_elongation")),
                "mean_iota": _finite_or_none(diagnostics.get("mean_iota")),
                "aspect": _finite_or_none(diagnostics.get("aspect")),
            }
            record["metrics"] = metrics
            record["diagnostics"] = diagnostics
            record["prefilter_score"] = basin_prefilter_score(metrics, targets, config)
            if bool(config.get("save_candidate_inputs", True)):
                candidate_dir = survey_dir / "candidates" / candidate.label.replace(":", "_")
                input_out = candidate_dir / "input.candidate"
                stage.optimizer.save_input(input_out, params)
                record["input_path"] = str(input_out)
        except Exception as exc:  # noqa: BLE001 - prefilter keeps failures ranked last.
            record["metrics"] = {}
            record["diagnostics"] = {}
            record["prefilter_score"] = float("inf")
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    ranked = sorted(
        rank_candidate_records(records, targets=targets),
        key=lambda row: (float(row.get("prefilter_score", float("inf"))), float(row.get("score", float("inf")))),
    )
    for rank, record in enumerate(ranked, start=1):
        record["prefilter_rank"] = rank
    top = ranked[: max(1, int(config.get("top_k", 8)))]
    (survey_dir / "candidates.json").write_text(json.dumps(ranked, indent=2, sort_keys=True) + "\n")
    (survey_dir / "top_candidates.json").write_text(json.dumps(top, indent=2, sort_keys=True) + "\n")
    write_csv(ranked, survey_dir / "candidates.csv")
    selected = top[0]
    selected_input = selected.get("input_path")
    if not selected_input:
        selected_input = str(survey_dir / "input.prefilter_selected")
        stage.optimizer.save_input(selected_input, np.asarray(selected["params"], dtype=float))
    print("\nBasin prefilter selected:")
    print(f"  label:          {selected.get('label')}")
    print(f"  prefilter score:{selected.get('prefilter_score')}")
    print(f"  metrics:        {selected.get('metrics')}")
    print(f"  input:          {selected_input}")
    return Path(selected_input)


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
):
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
        scipy_tr_solver=SCIPY_TR_SOLVER,
        scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
        lbfgs_step_bound=lbfgs_step_bound,
        scalar_step_bound=scalar_step_bound,
        save_stage_inputs=SAVE_STAGE_INPUTS,
        save_stage_wouts=SAVE_STAGE_WOUTS,
    )


def qi_diagnostics_for_result(
    stage_result,
    *,
    mirror_threshold,
    mirror_surface_index,
    smooth_qi_max=QI_GATE_SMOOTH_MAX,
    legacy_qi_max=QI_GATE_LEGACY_MAX,
):
    opt = stage_result.final_optimizer
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
        mirror_threshold=mirror_threshold,
        mirror_surface_index=mirror_surface_index,
        elongation_threshold=MAX_ELONGATION,
    )
    diagnostics = vj.qi_diagnostics_from_state(
        state=stage_result.final_state,
        static=opt.static,
        indata=opt.indata,
        signgs=opt.signgs,
        surfaces=SURFACES,
        options=diagnostic_options,
    )
    return annotate_qi_seed_suitability(
        diagnostics,
        targets=QISeedSuitabilityTargets(
            smooth_qi_max=smooth_qi_max,
            legacy_qi_max=legacy_qi_max,
            target_aspect=TARGET_ASPECT,
            abs_iota_min=TARGET_ABS_IOTA_MIN,
            mirror_ratio_max=mirror_threshold,
            max_elongation=MAX_ELONGATION,
        ),
    )


def stage_modes_for(stage):
    if "stage_modes" in stage:
        return [int(mode) for mode in stage["stage_modes"]]
    return vj.repeated_stage_modes(
        max_mode=MAX_MODE,
        use_mode_continuation=USE_MODE_CONTINUATION,
        continuation_nfev=CONTINUATION_NFEV,
        repeats=int(stage.get("stage_repeats", STAGE_REPEATS)),
    )


def promotion_score(record):
    """Lower score means a better exact-diagnostic QI candidate."""

    seed_penalty = 0.0 if bool(record.get("qi_seed_gate_passed")) else 100.0
    engineering_penalty = 0.0 if bool(record.get("qi_engineering_gate_passed")) else 10.0
    return (
        seed_penalty
        + engineering_penalty
        + _finite_or_inf(record.get("qi_rank_score"))
        + 0.25 * _finite_or_inf(record.get("qi_constraint_score"))
    )


def stage_promotes_candidate(stage, promotion, reference_diagnostics):
    """Apply the script's staged promotion rule to exact diagnostics."""

    reasons = list(promotion.get("qi_cleanup_rejection_reasons", []))
    if bool(stage.get("accept_if_iota_improves", False)) and reference_diagnostics is not None:
        candidate_iota = abs(_finite_or_inf(promotion.get("mean_iota")))
        reference_iota = abs(_finite_or_inf(reference_diagnostics.get("mean_iota")))
        iota_gain = candidate_iota - reference_iota
        qi_relax = float(stage.get("qi_relax_for_iota", 2.0))
        smooth_limit = qi_relax * max(
            QI_GATE_SMOOTH_MAX,
            _finite_or_inf(reference_diagnostics.get("qi_smooth_total")),
        )
        legacy_limit = qi_relax * max(
            QI_GATE_LEGACY_MAX,
            _finite_or_inf(reference_diagnostics.get("qi_legacy_total")),
        )
        if (
            iota_gain >= float(stage.get("iota_improvement_min", 0.0))
            and _finite_or_inf(promotion.get("qi_smooth_total")) <= smooth_limit
            and _finite_or_inf(promotion.get("qi_legacy_total")) <= legacy_limit
        ):
            out = dict(promotion)
            out["qi_cleanup_promoted"] = True
            out["qi_cleanup_rejection_reasons"] = []
            out["qi_iota_promotion_reason"] = (
                f"iota increased by {iota_gain:.6g} while QI stayed within "
                f"{qi_relax:.3g}x relaxed smooth/legacy limits"
            )
            return out
        reasons.append(
            "iota ramp did not satisfy relaxed QI promotion: "
            f"gain={iota_gain:.6g}, smooth_limit={smooth_limit:.6g}, legacy_limit={legacy_limit:.6g}"
        )
    if bool(stage.get("accept_if_rank_improves", False)) and reference_diagnostics is not None:
        candidate_score = promotion_score(promotion)
        reference_score = promotion_score(reference_diagnostics)
        tolerance = float(stage.get("rank_score_relax", 1.0e-12))
        if candidate_score >= reference_score - tolerance:
            reasons.append(
                "rank score did not improve: "
                f"candidate={candidate_score:.6g}, reference={reference_score:.6g}"
            )
    elif bool(stage.get("accept_if_rank_improves", False)):
        # The first staged far-seed result is allowed to become the baseline.
        pass
    if reasons:
        out = dict(promotion)
        out["qi_cleanup_promoted"] = False
        out["qi_cleanup_rejection_reasons"] = reasons
        return out
    return promotion

active_input_file = INPUT_FILE
active_input_file = run_basin_prefilter(
    active_input_file,
    OUTPUT_DIR,
    dict(CASE.get("basin_prefilter", {})),
)
if QI_PREFINE:
    print("Running QI-only pre-refinement before applying scalar constraints ...")
    preseed_result = solve_qi_stage(
        INPUT_FILE,
        OUTPUT_DIR / "qi_preseed",
        qi_only_problem,
        max_nfev=QI_PREFINE_NFEV,
        label=f"QI-only pre-refinement (max_mode={MAX_MODE})",
    )
    preseed_history = preseed_result.history
    print(f"QI-only pre-refinement final objective: {preseed_history['objective_final']:.6e}")
    active_input_file = OUTPUT_DIR / "qi_preseed" / "input.final"

first_result_for_outputs = None
promotion_log = []
if MIRROR_RAMP_STAGES:
    accepted_result = None
    accepted_seed_diagnostics = None
    best_result = None
    best_diagnostics = None
    for stage_index, stage in enumerate(MIRROR_RAMP_STAGES, start=1):
        stage_name = stage["name"]
        stage_output_dir = OUTPUT_DIR / f"mirror_ramp_{stage_index:02d}_{stage_name}"
        stage_problem = make_qi_problem(stage)
        stage_modes_i = stage_modes_for(stage)
        stage_result = solve_qi_stage(
            active_input_file,
            stage_output_dir,
            stage_problem,
            max_nfev=int(stage.get("max_nfev", MAX_NFEV)),
            label=f"QI {stage_name} (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
            stage_modes=stage_modes_i,
            method=str(stage.get("method", METHOD)),
            use_mode_continuation=bool(stage.get("use_mode_continuation", USE_MODE_CONTINUATION)),
            scalar_step_bound=stage.get("scalar_step_bound"),
            lbfgs_step_bound=stage.get("lbfgs_step_bound"),
        )
        if first_result_for_outputs is None:
            first_result_for_outputs = stage_result
        stage_mirror_threshold = float(stage.get("mirror_threshold", MAX_MIRROR_RATIO))
        stage_promotion_mirror_threshold = float(
            stage.get("promotion_mirror_threshold", stage_mirror_threshold)
        )
        stage_mirror_surface_index = stage.get("mirror_surface_index", MIRROR_SURFACE_INDEX)
        stage_smooth_qi_max = float(stage.get("smooth_qi_max", QI_GATE_SMOOTH_MAX))
        stage_legacy_qi_max = float(stage.get("legacy_qi_max", QI_GATE_LEGACY_MAX))
        reference_diagnostics = (
            None
            if accepted_result is None
            else qi_diagnostics_for_result(
                accepted_result,
                mirror_threshold=stage_promotion_mirror_threshold,
                mirror_surface_index=stage_mirror_surface_index,
                smooth_qi_max=stage_smooth_qi_max,
                legacy_qi_max=stage_legacy_qi_max,
            )
        )
        stage_diagnostics = qi_diagnostics_for_result(
            stage_result,
            mirror_threshold=stage_promotion_mirror_threshold,
            mirror_surface_index=stage_mirror_surface_index,
            smooth_qi_max=stage_smooth_qi_max,
            legacy_qi_max=stage_legacy_qi_max,
        )
        promotion = vj.qi_cleanup_candidate_promotable(
            stage_diagnostics,
            reference=reference_diagnostics,
            targets=QISeedSuitabilityTargets(
                smooth_qi_max=stage_smooth_qi_max,
                legacy_qi_max=stage_legacy_qi_max,
                target_aspect=TARGET_ASPECT,
                abs_iota_min=TARGET_ABS_IOTA_MIN,
                mirror_ratio_max=stage_promotion_mirror_threshold,
                max_elongation=MAX_ELONGATION,
            ),
            require_seed_gate=bool(stage.get("require_seed_gate", True)),
            require_mirror_improvement=bool(
                stage.get("require_mirror_improvement", accepted_seed_diagnostics is not None)
                and float(stage.get("mirror_weight", MIRROR_WEIGHT)) > 0.0
            ),
            require_engineering_gate=bool(stage.get("require_engineering_gate", False)),
            mirror_improvement_min=float(stage.get("mirror_improvement_min", 0.0)),
        )
        promotion = stage_promotes_candidate(stage, promotion, reference_diagnostics)
        promotion_log.append(
            {
                "stage": stage_index,
                "name": stage_name,
                "output_dir": str(stage_output_dir),
                "stage_modes": list(stage_modes_i),
                "method": str(stage.get("method", METHOD)),
                "promoted": bool(promotion["qi_cleanup_promoted"]),
                "smooth_qi": promotion.get("qi_smooth_total"),
                "legacy_qi": promotion.get("qi_legacy_total"),
                "mirror": promotion.get("qi_mirror_ratio_max"),
                "elongation": promotion.get("qi_max_elongation"),
                "mean_iota": promotion.get("mean_iota"),
                "rank_score": promotion.get("qi_rank_score"),
                "constraint_score": promotion.get("qi_constraint_score"),
                "iota_promotion_reason": promotion.get("qi_iota_promotion_reason"),
                "rejection_reasons": promotion.get("qi_cleanup_rejection_reasons", []),
            }
        )
        print(f"\nMirror-ramp stage {stage_index}: {stage_name}")
        print(f"  smooth QI:    {promotion.get('qi_smooth_total')}")
        print(f"  legacy QI:    {promotion.get('qi_legacy_total')}")
        print(f"  mirror ratio: {promotion.get('qi_mirror_ratio_max')}")
        print(f"  elongation:   {promotion.get('qi_max_elongation')}")
        print(f"  mean iota:    {promotion.get('mean_iota')}")
        print(f"  rank score:   {promotion.get('qi_rank_score')}")
        print(f"  promoted:     {promotion['qi_cleanup_promoted']}")
        if promotion.get("qi_iota_promotion_reason"):
            print(f"    - {promotion['qi_iota_promotion_reason']}")
        for reason in promotion.get("qi_cleanup_rejection_reasons", []):
            print(f"    - {reason}")

        if best_diagnostics is None or promotion_score(stage_diagnostics) < promotion_score(best_diagnostics):
            best_result = stage_result
            best_diagnostics = stage_diagnostics

        if promotion["qi_cleanup_promoted"]:
            accepted_result = stage_result
            accepted_seed_diagnostics = stage_diagnostics
            active_input_file = stage_output_dir / "input.final"
        else:
            if accepted_result is None:
                print(
                    f"Initial QI staged policy {stage_name!r} failed the promotion gate; "
                    "continuing with the best exact-diagnostic candidate recorded so far."
                )
                active_input_file = stage_output_dir / "input.final"
                continue
            print("  continuing from the last promoted stage.")
            continue
    result = accepted_result if accepted_result is not None else best_result
else:
    result = solve_qi_stage(
        active_input_file,
        OUTPUT_DIR,
        problem,
        max_nfev=MAX_NFEV,
        label=f"QI optimization (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})",
    )
    first_result_for_outputs = result

if result is None:
    raise RuntimeError("QI optimization did not produce a result.")

# Results are plain Python objects.  The solve writes these default artifacts
# for convenience; the explicit calls below show where to customize filenames
# or add additional exports in a SIMSOPT-style workflow.
initial_result_for_outputs = first_result_for_outputs if first_result_for_outputs is not None else result
initial_optimizer = initial_result_for_outputs.initial_optimizer
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
initial_optimizer.save_input(saved_paths["initial_input"], initial_result_for_outputs.initial_params)
initial_optimizer.save_wout(
    saved_paths["initial_wout"],
    initial_result_for_outputs.initial_params,
    state=initial_result_for_outputs.initial_state,
)
final_optimizer.save_input(saved_paths["final_input"], result.final_params)
final_optimizer.save_wout(
    saved_paths["final_wout"],
    result.final_params,
    state=result.final_state,
)
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
diagnostics["qi_optimization_resolution"] = {
    "mboz": QI_OPTIONS.mboz,
    "nboz": QI_OPTIONS.nboz,
    "nphi": QI_OPTIONS.nphi,
    "nalpha": QI_OPTIONS.nalpha,
    "n_bounce": QI_OPTIONS.n_bounce,
}
diagnostics["qi_audit_resolution"] = {
    "mboz": diagnostic_options.mboz,
    "nboz": diagnostic_options.nboz,
    "nphi": diagnostic_options.nphi,
    "nalpha": diagnostic_options.nalpha,
    "n_bounce": diagnostic_options.n_bounce,
}
diagnostics_path = OUTPUT_DIR / "diagnostics.json"
diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
saved_paths["diagnostics"] = diagnostics_path
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
print(f"  diagnostics:     {diagnostics_path}")
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

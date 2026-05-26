"""Case catalog and environment resolution for the QI optimization example."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_QI_TARGET_ASPECT = 5.0
DEFAULT_INNER_MAX_ITER = 450
DEFAULT_INNER_FTOL = 1.0e-9
DEFAULT_TRIAL_MAX_ITER = 450
DEFAULT_TRIAL_FTOL = 1.0e-9
SEED3127_REVIEWED_TARGET_ASPECT = DEFAULT_QI_TARGET_ASPECT
TARGET_HELICITY_SEED_AMPLITUDE = 1.0e-5
TARGET_HELICITY_SEED_TERMS = (
    ("RBC", (1, 0), TARGET_HELICITY_SEED_AMPLITUDE),
    ("ZBS", (1, 0), TARGET_HELICITY_SEED_AMPLITUDE),
    ("RBC", (-1, 1), TARGET_HELICITY_SEED_AMPLITUDE),
    ("ZBS", (-1, 1), TARGET_HELICITY_SEED_AMPLITUDE),
    ("RBC", (1, 1), TARGET_HELICITY_SEED_AMPLITUDE),
    ("ZBS", (1, 1), TARGET_HELICITY_SEED_AMPLITUDE),
)
MINIMAL_QI_LOCAL_STAGE_MIN_NFEV = 8


def _parse_float_sequence(value, *, name):
    """Parse a comma/space separated sequence used by subprocess wrappers."""

    if value in (None, ""):
        return None
    pieces = str(value).replace(",", " ").split()
    if not pieces:
        return None
    try:
        return tuple(float(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma- or space-separated float list: {value!r}") from exc


QI_CASES = {
    "nfp1_qi": {
        "case_goal": "NFP=1 mirror-aware QI lane",
        "input_file": DATA_DIR / "input.nfp1_QI",
        "output_dir": Path("results/qi_opt/ess/nfp1_qi_aspect5"),
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
        "target_helicity_seed_terms": TARGET_HELICITY_SEED_TERMS,
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
    "nfp2_qi": {
        "case_goal": "default NFP=2 mirror-aware QI lane",
        "input_file": DATA_DIR / "input.nfp2_QI",
        "output_dir": Path("results/qi_opt/ess/nfp2_qi_aspect5"),
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
        "target_helicity_seed_terms": TARGET_HELICITY_SEED_TERMS,
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
        "case_goal": "far-seed staged QI robustness lane with reference-family global preconditioning",
        "input_file": DATA_DIR / "input.QI_stel_seed_3127",
        "output_dir": Path("results/qi_opt/ess/qi_stel_seed_3127_aspect5"),
        "max_mode": 4,
        "min_vmec_mode": 6,
        "use_mode_continuation": False,
        "stage_repeats": 1,
        "max_nfev": 8,
        # The public robustness lane now uses the same aspect target as the
        # QA/QH/QP examples. Older aspect-4 artifacts remain archival only.
        "target_aspect": SEED3127_REVIEWED_TARGET_ASPECT,
        "target_abs_iota_min": 0.41,
        "mirror_threshold": 0.35,
        "mirror_surface_index": None,
        "qi_gate_smooth_max": 5.0e-3,
        "qi_gate_legacy_max": 2.0e-3,
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
        # Deterministic global-to-local preconditioner.  It makes a small
        # same-NFP reference-family scan before local optimization.  This is
        # the current robust path for the unrelated seed: the reviewed lambda
        # grid reaches the precise-QI basin, while purely local boundary steps
        # do not.
        "boundary_reference_preconditioner": {
            "enabled": True,
            "reference_input": DATA_DIR / "input.nfp3_QI_fixed_resolution_final",
            "lambdas": (0.99, 0.995, 1.0, 1.005, 1.008, 1.01, 1.012),
            "keys": ("RBC", "ZBS", "RBS", "ZBC"),
            "max_mode": 4,
            "max_iter": 80,
            "target_aspect": SEED3127_REVIEWED_TARGET_ASPECT,
            "abs_iota_min": 0.41,
            "max_mirror_ratio": 0.35,
            "max_elongation": 8.0,
            "smooth_qi_max": 5.0e-3,
            "legacy_qi_max": 2.0e-3,
            "diagnostic_qi_resolution": {"mboz": 18, "nboz": 18, "nphi": 151, "nalpha": 31, "n_bounce": 51},
            # Once a candidate passes the independent QI/iota/mirror/elongation
            # gates, prefer the lower-mirror branch.  This uses the aspect and
            # elongation margin of this seed without promoting non-QI states.
            "mirror_selection_weight": 10.0,
            "prefer_non_endpoint": True,
            "accept_as_baseline": True,
        },
        "basin_prefilter": {
            "enabled": False,
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
        "target_helicity_seed_terms": TARGET_HELICITY_SEED_TERMS,
        # Far seeds can still opt into the bounded basin prefilter above, but
        # the public seed-3127 lane now starts from the reference-family
        # preconditioner and only promotes cleanup stages that pass exact gates.
        # Mirror-balanced policies remain in tools/diagnostics because the
        # current all-surface mirror objective trades away the QI gate.
        "mirror_ramp_stages": (
            {
                "name": "prefiltered_mirror_qi_iota_cleanup",
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
                "name": "nfirst_qi_ceiling_mirror_cleanup",
                "max_nfev": 6,
                "stage_repeats": 1,
                # First unlock toroidal structure at fixed low poloidal
                # complexity, then unlock the full mode-4 boundary.  This is
                # cheaper and often safer than immediately moving all mode-4
                # poloidal/toroidal harmonics.
                "stage_mode_limits": (
                    {"mode": 4, "max_m": 1, "max_n": 4, "label": "nfirst"},
                    {"mode": 4, "max_m": 4, "max_n": 4, "label": "full"},
                ),
                "method": "scipy_matrix_free",
                "use_mode_continuation": True,
                "aspect_weight": 0.05,
                "iota_floor_weight": 50.0**2,
                "qi_weight": 350.0,
                "qi_ceiling_max": 5.0e-3,
                "qi_ceiling_weight": 5000.0,
                "mirror_threshold": 0.30,
                "promotion_mirror_threshold": 0.35,
                "mirror_weight": 8.0,
                "elongation_weight": 0.0,
                "require_seed_gate": True,
                "require_mirror_improvement": True,
                "require_engineering_gate": True,
                "accept_if_engineering_score_improves": True,
                "mirror_improvement_min": 1.0e-3,
            },
        ),
    },
    "nfp4_qi": {
        "case_goal": "NFP=4 minimal-seed QI lane with same-NFP reference-family preconditioner",
        "input_file": DATA_DIR / "input.minimal_seed_nfp4",
        "output_dir": Path("results/qi_opt/ess/minimal_nfp4_to_qi_finite_beta_reference"),
        "max_mode": 3,
        "min_vmec_mode": 5,
        "method": "scipy_matrix_free",
        "use_mode_continuation": True,
        "stage_repeats": 1,
        "max_nfev": 1,
        "inner_max_iter": DEFAULT_INNER_MAX_ITER,
        "inner_ftol": DEFAULT_INNER_FTOL,
        "trial_max_iter": DEFAULT_TRIAL_MAX_ITER,
        "trial_ftol": DEFAULT_TRIAL_FTOL,
        "target_aspect": DEFAULT_QI_TARGET_ASPECT,
        "target_abs_iota_min": 0.41,
        "max_elongation": 8.2,
        "mirror_threshold": 0.35,
        "mirror_surface_index": None,
        "qi_gate_smooth_max": 3.0e-3,
        "qi_gate_legacy_max": 2.0e-3,
        "qi_ceiling_max": 2.0e-2,
        "qi_ceiling_smooth_penalty": 2.0e-3,
        "branch_width_weight": 0.5,
        "weighted_shuffle_profile_weight": 0.0,
        "phimin": 0.0,
        "optimization_qi_resolution": {"mboz": 5, "nboz": 5, "nphi": 31, "nalpha": 7, "n_bounce": 9},
        "audit_qi_resolution": {"mboz": 18, "nboz": 18, "nphi": 151, "nalpha": 31, "n_bounce": 51},
        # This keeps the public source input as the common three-coefficient
        # minimal seed, but allows the driver to make a deterministic same-NFP
        # jump into a reviewed QI family before local differentiable cleanup.
        "boundary_reference_preconditioner": {
            "enabled": True,
            "reference_input": DATA_DIR / "input.nfp4_QI_finite_beta",
            "lambdas": (1.0,),
            "keys": ("RBC", "ZBS", "RBS", "ZBC"),
            "max_mode": 3,
            "max_iter": 80,
            "target_aspect": DEFAULT_QI_TARGET_ASPECT,
            "abs_iota_min": 0.41,
            "max_mirror_ratio": 0.35,
            "max_elongation": 8.2,
            "smooth_qi_max": 3.0e-3,
            "legacy_qi_max": 2.0e-3,
            "diagnostic_qi_resolution": {"mboz": 18, "nboz": 18, "nphi": 151, "nalpha": 31, "n_bounce": 51},
            "accept_as_baseline": True,
        },
        "basin_prefilter": {"enabled": False},
        "boozer_target_wout": None,
        "boozer_target_weight": 0.0,
        "boozer_target_normalize": True,
        "boozer_target_include_b00": False,
        "mirror_weight": 4.0,
        "elongation_weight": 1.0,
        "qi_ceiling_weight": 0.0,
        "shuffle_profile_nphi_out": None,
        "target_helicity_seed_terms": TARGET_HELICITY_SEED_TERMS,
        "mirror_ramp_stages": (
            {
                "name": "finite_beta_qi_audit_refine",
                "max_nfev": 1,
                "stage_repeats": 1,
                "method": "scipy_matrix_free",
                "mirror_threshold": 0.35,
                "promotion_mirror_threshold": 0.35,
                "mirror_weight": 4.0,
                "elongation_weight": 1.0,
                "require_mirror_improvement": False,
                "require_engineering_gate": True,
            },
        ),
    },
    "nfp4_qi_finite_beta": {
        "case_goal": "NFP=4 finite-beta QI stress fixture; not a README robustness lane",
        "input_file": DATA_DIR / "input.nfp4_QI_finite_beta",
        "output_dir": Path("results/qi_opt/ess/nfp4_qi_finite_beta"),
        "max_mode": 3,
        "min_vmec_mode": 5,
        "method": "scipy_matrix_free",
        "use_mode_continuation": True,
        "stage_repeats": 1,
        "max_nfev": 1,
        "inner_max_iter": DEFAULT_INNER_MAX_ITER,
        "inner_ftol": DEFAULT_INNER_FTOL,
        "trial_max_iter": DEFAULT_TRIAL_MAX_ITER,
        "trial_ftol": DEFAULT_TRIAL_FTOL,
        "target_aspect": DEFAULT_QI_TARGET_ASPECT,
        "target_abs_iota_min": 0.41,
        "max_elongation": 8.2,
        "mirror_threshold": 0.35,
        "mirror_surface_index": None,
        "qi_gate_smooth_max": 3.0e-3,
        "qi_gate_legacy_max": 2.0e-3,
        "qi_ceiling_max": 2.0e-2,
        "qi_ceiling_smooth_penalty": 2.0e-3,
        "branch_width_weight": 0.5,
        "weighted_shuffle_profile_weight": 0.0,
        "phimin": 0.0,
        "mirror_weight": 4.0,
        "elongation_weight": 1.0,
        "qi_ceiling_weight": 0.0,
        "shuffle_profile_nphi_out": None,
        "target_helicity_seed_terms": TARGET_HELICITY_SEED_TERMS,
        "optimization_qi_resolution": {"mboz": 5, "nboz": 5, "nphi": 31, "nalpha": 7, "n_bounce": 9},
        "audit_qi_resolution": {"mboz": 18, "nboz": 18, "nphi": 151, "nalpha": 31, "n_bounce": 51},
        "mirror_ramp_stages": (
            {
                "name": "finite_beta_qi_audit_refine",
                "max_nfev": 1,
                "stage_repeats": 1,
                "method": "scipy_matrix_free",
                "mirror_threshold": 0.35,
                "promotion_mirror_threshold": 0.35,
                "mirror_weight": 4.0,
                "elongation_weight": 1.0,
                "qi_ceiling_max": 2.0e-2,
                "qi_ceiling_weight": 0.0,
                "require_mirror_improvement": False,
                "require_engineering_gate": True,
            },
        ),
    },
    "nfp4_qh_warm_to_qi": {
        "case_goal": "NFP=4 QH-to-QI non-passing stress fixture; audit only",
        "input_file": DATA_DIR / "input.nfp4_QH_warm_start",
        "output_dir": Path("results/qi_opt/ess/nfp4_qh_warm_to_qi"),
        # This case is intentionally kept as a seed-robustness stress fixture,
        # not a promoted QI lane.  Bounded May 2026 diagnostics found no local
        # NFP=4 path satisfying the agreed smooth/legacy QI < 2e-3 gates.
        "expected_gate_status": "non_passing_stress_fixture",
        "expected_gate_failures": ("smooth_qi", "legacy_qi", "mirror"),
        "stress_fixture_notes": (
            "Bundled QH warm start and QH-to-QI local cleanup remain above the legacy QI gate.",
            "Archived external NFP=4 QI references improve QI but still miss the 2e-3 smooth/legacy gates.",
            "Do not promote this case without an independent diagnostics.json gate pass.",
        ),
        "known_best_nfp4_quick_audit": {
            "label": "external_nfp4_qi_wfq0",
            "smooth_qi": 8.421446105814759e-3,
            "legacy_qi": 5.205127302950363e-3,
            "mirror_ratio": 3.133889788613409e-1,
            "audit_command": "audit_qi_seed_suitability.py --quick --target-aspect 5 --max-mirror-ratio 0.35",
        },
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
        "target_helicity_seed_terms": TARGET_HELICITY_SEED_TERMS,
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

# Public convenience alias: the reviewed NFP=3 robustness lane is the
# ``input.QI_stel_seed_3127`` seed with same-NFP reference-family
# preconditioning.  Keep the historical descriptive case name, but also expose a
# symmetric ``nfp3_qi`` selector for README/docs workflows.
QI_CASES["nfp3_qi"] = {
    **QI_CASES["qi_stel_seed_3127"],
    "case_goal": "NFP=3 far-seed QI robustness lane",
}


def _minimal_or_circular_qi_case(
    *,
    base_case: str,
    case_goal: str,
    input_file: Path,
    output_dir: Path,
    reference_input: Path,
    max_mode: int,
    min_vmec_mode: int,
    reference_lambdas=(0.995, 1.0, 1.005),
):
    """Build a deterministic torus-like-seed QI case from a reviewed policy."""

    base = QI_CASES[base_case]
    boundary_reference = dict(base.get("boundary_reference_preconditioner", {}))
    boundary_reference.update(
        {
            "enabled": True,
            "reference_input": reference_input,
            "lambdas": tuple(float(value) for value in reference_lambdas),
            "keys": ("RBC", "ZBS", "RBS", "ZBC"),
            "max_mode": int(max_mode),
            "target_aspect": float(base.get("target_aspect", DEFAULT_QI_TARGET_ASPECT)),
            "abs_iota_min": float(base.get("target_abs_iota_min", 0.41)),
            "max_mirror_ratio": float(base.get("mirror_threshold", 0.35)),
            "max_elongation": float(base.get("max_elongation", 8.2)),
            "smooth_qi_max": float(base.get("qi_gate_smooth_max", 2.0e-3)),
            "legacy_qi_max": float(base.get("qi_gate_legacy_max", 2.0e-3)),
            # The reference scan is a deterministic proposal generator for
            # circular/minimal seeds.  Do not let its one-evaluation bookkeeping
            # solve become the final accepted result; the local cleanup stages
            # below must run and own promotion.
            "accept_as_baseline": False,
        }
    )
    local_stages = []
    for stage in base.get("mirror_ramp_stages", ()):
        stage_nfev = max(
            int(stage.get("max_nfev", base.get("max_nfev", MINIMAL_QI_LOCAL_STAGE_MIN_NFEV))),
            int(base.get("max_nfev", MINIMAL_QI_LOCAL_STAGE_MIN_NFEV)),
            MINIMAL_QI_LOCAL_STAGE_MIN_NFEV,
        )
        local_stages.append(
            {
                **stage,
                "max_nfev": stage_nfev,
                # Showcase and staged-runner --max-nfev should be the local
                # optimizer budget for these reference-seeded cases, not only a
                # ceiling over legacy one-evaluation audit stages.
                "use_showcase_max_nfev": True,
            }
        )
    return {
        **base,
        "case_goal": case_goal,
        "input_file": input_file,
        "output_dir": output_dir,
        "max_mode": int(max_mode),
        "min_vmec_mode": int(min_vmec_mode),
        "target_helicity_seed_terms": TARGET_HELICITY_SEED_TERMS,
        "boundary_reference_preconditioner": boundary_reference,
        "mirror_ramp_stages": tuple(local_stages),
    }


QI_CASES.update(
    {
        "minimal_nfp1_qi": _minimal_or_circular_qi_case(
            base_case="nfp1_qi",
            case_goal="NFP=1 common-minimal seed to QI with deterministic 1e-5 seeding",
            input_file=DATA_DIR / "input.minimal_seed_nfp1",
            output_dir=Path("results/qi_opt/ess/minimal_nfp1_to_qi_reference"),
            reference_input=DATA_DIR / "input.nfp1_QI",
            max_mode=3,
            min_vmec_mode=6,
            reference_lambdas=(0.99, 0.995, 1.0, 1.005, 1.01),
        ),
        "minimal_nfp2_qi": _minimal_or_circular_qi_case(
            base_case="nfp2_qi",
            case_goal="NFP=2 common-minimal seed to QI with deterministic 1e-5 seeding",
            input_file=DATA_DIR / "input.minimal_seed_nfp2",
            output_dir=Path("results/qi_opt/ess/minimal_nfp2_to_qi_reference"),
            reference_input=DATA_DIR / "input.nfp2_QI",
            max_mode=3,
            min_vmec_mode=6,
            reference_lambdas=(0.99, 0.995, 1.0, 1.005, 1.01),
        ),
        "minimal_nfp3_qi": _minimal_or_circular_qi_case(
            base_case="qi_stel_seed_3127",
            case_goal="NFP=3 common-minimal seed to QI with deterministic 1e-5 seeding",
            input_file=DATA_DIR / "input.minimal_seed_nfp3",
            output_dir=Path("results/qi_opt/ess/minimal_nfp3_to_qi_reference"),
            reference_input=DATA_DIR / "input.nfp3_QI_fixed_resolution_final",
            max_mode=4,
            min_vmec_mode=6,
            reference_lambdas=(0.99, 0.995, 1.0, 1.005, 1.008, 1.01),
        ),
        "minimal_nfp4_qi": _minimal_or_circular_qi_case(
            base_case="nfp4_qi",
            case_goal="NFP=4 common-minimal seed to finite-beta QI reference with deterministic 1e-5 seeding",
            input_file=DATA_DIR / "input.minimal_seed_nfp4",
            output_dir=Path("results/qi_opt/ess/minimal_nfp4_to_qi_reference"),
            reference_input=DATA_DIR / "input.nfp4_QI_finite_beta",
            max_mode=3,
            min_vmec_mode=5,
            reference_lambdas=(1.0,),
        ),
        "circular_nfp1_qi": _minimal_or_circular_qi_case(
            base_case="nfp1_qi",
            case_goal="NFP=1 circular torus seed to QI with deterministic 1e-5 seeding",
            input_file=DATA_DIR / "input.circular_tokamak",
            output_dir=Path("results/qi_opt/ess/circular_nfp1_to_qi_reference"),
            reference_input=DATA_DIR / "input.nfp1_QI",
            max_mode=3,
            min_vmec_mode=6,
            reference_lambdas=(0.99, 0.995, 1.0, 1.005, 1.01),
        ),
    }
)

RUN_CASE_DEFAULT = "nfp2_qi"


def resolve_qi_case(default_run_case: str | None = None):
    """Return ``(run_case, case)`` after applying QI example environment overrides."""

    RUN_CASE = RUN_CASE_DEFAULT if default_run_case is None else str(default_run_case)
    _EXTERNAL_INPUT = os.environ.get("VMEC_JAX_QI_INPUT")
    if _EXTERNAL_INPUT:
        _external_label = os.environ.get(
            "VMEC_JAX_QI_RUN_CASE",
            os.environ.get("VMEC_JAX_QI_LABEL", Path(_EXTERNAL_INPUT).name.replace("input.", "")),
        )
        _external_policy_case = os.environ.get("VMEC_JAX_QI_POLICY_CASE", "qi_stel_seed_3127")
        if _external_policy_case not in QI_CASES:
            raise KeyError(
                f"Unknown VMEC_JAX_QI_POLICY_CASE {_external_policy_case!r}; available cases: {sorted(QI_CASES)}"
            )
        _external_base_case = QI_CASES[_external_policy_case]
        _external_reference = os.environ.get("VMEC_JAX_QI_REFERENCE_INPUT")
        _external_boundary_reference = {"enabled": False}
        if _external_reference:
            _external_max_mode = int(os.environ.get("VMEC_JAX_QI_MAX_MODE", _external_base_case["max_mode"]))
            _reference_base = dict(
                _external_base_case.get(
                    "boundary_reference_preconditioner",
                    QI_CASES["qi_stel_seed_3127"]["boundary_reference_preconditioner"],
                )
            )
            _external_boundary_reference = {
                **_reference_base,
                "enabled": True,
                "reference_input": Path(_external_reference).expanduser(),
                "max_mode": _external_max_mode,
                "target_aspect": float(_external_base_case.get("target_aspect", DEFAULT_QI_TARGET_ASPECT)),
                "abs_iota_min": float(_external_base_case.get("target_abs_iota_min", 0.41)),
                "max_mirror_ratio": float(_external_base_case.get("mirror_threshold", 0.30)),
                "max_elongation": float(_external_base_case.get("max_elongation", 8.2)),
                "smooth_qi_max": float(_external_base_case.get("qi_gate_smooth_max", 2.0e-3)),
                "legacy_qi_max": float(_external_base_case.get("qi_gate_legacy_max", 2.0e-3)),
                "max_iter": int(os.environ.get("VMEC_JAX_QI_INNER_MAX_ITER", _reference_base.get("max_iter", 80))),
                "prefer_qi_safe_candidates": True,
            }
            _reference_lambdas = _parse_float_sequence(
                os.environ.get("VMEC_JAX_QI_REFERENCE_LAMBDAS"),
                name="VMEC_JAX_QI_REFERENCE_LAMBDAS",
            )
            if _reference_lambdas is not None:
                _external_boundary_reference["lambdas"] = _reference_lambdas
        # External inputs use the far-seed robustness policy by default: first
        # establish a QI+iota basin, then add guarded engineering cleanup later.
        # If the user supplies VMEC_JAX_QI_REFERENCE_INPUT, the same deterministic
        # global-to-local reference-family preconditioner is enabled for that seed.
        QI_CASES[_external_label] = {
            **_external_base_case,
            "case_goal": "external VMEC input using the far-seed QI+iota robustness policy",
            "input_file": Path(_EXTERNAL_INPUT).expanduser(),
            "output_dir": Path(
                os.environ.get("VMEC_JAX_QI_OUTPUT_DIR", f"results/qi_opt/ess/{_external_label}")
            ).expanduser(),
            "boundary_reference_preconditioner": _external_boundary_reference,
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
    return RUN_CASE, CASE

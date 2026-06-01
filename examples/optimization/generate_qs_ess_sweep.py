#!/usr/bin/env python
# ruff: noqa: E402
"""Run QA/QH/QP/QI ESS policy sweeps and build publication-style summary panels.

This script regenerates a reviewer-facing benchmark matrix for the standalone
``vmec_jax`` exact optimization path:

- problems: QA, QH, QP, and QI
- policies: continuation and direct-start mode expansion
- max_mode: 1, 2, 3, 4, 5 for continuation and direct start
- ESS: off and on
- backends: encoded by ``--backend-label`` (for example ``cpu`` or ``gpu``)

For each case it saves:

- ``input.initial`` / ``input.final``
- ``wout_initial.nc`` / ``wout_final.nc``
- ``history.json``
- ``boundary_comparison.png``
- ``bmag_surface.png``
- ``objective_history.png``

It also writes:

- ``summary_<backend>_<policy>.json`` and ``summary_<backend>_<policy>.csv``
- ``objective_panel_<backend>_<policy>.png`` / ``.pdf``
- ``geometry_atlas_<backend>_<policy>.png`` / ``.pdf``

The optimization itself remains pure in-process within each worker. The outer
matrix uses subprocess isolation only so one failed case cannot abort the full
sweep regeneration.

Examples:

  python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation
  JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy direct
  JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu_diag --solver-device gpu --policy direct --diagnostic-budgets
  python examples/optimization/render_qs_ess_publication_panel.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64, jnp
from vmec_jax.config import config_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.namelist import InData
from vmec_jax.optimization import rebuild_indata_with_resolution
from vmec_jax.quasi_isodynamic import (
    _nearest_half_mesh_indices,
    lgradb_penalty_from_state,
    max_elongation_penalty_from_state,
    mirror_ratio_penalty_from_boozer_output,
    quasi_isodynamic_residual_from_state,
)
from vmec_jax.qi_legacy import legacy_qi_branch_shuffle_diagnostic_from_boozer_output
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


enable_x64(True)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "examples" / "data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "qs_ess_sweep"

MIN_VMEC_MODE = 5
VMEC_MPOL = MIN_VMEC_MODE
VMEC_NTOR = MIN_VMEC_MODE
MODES = (1, 2, 3, 4, 5)
PROBLEMS = ("qa", "qh", "qp", "qi")
ESS_OPTIONS = (False, True)

USE_MODE_CONTINUATION = True
BACKEND_LABEL = "cpu"
SOLVER_DEVICE: str | None = None  # None uses JAX default; set "cpu" or "gpu" to force one backend.
SKIP_EXISTING = True
CASE_TIMEOUT_S: float | None = 1800.0
ESS_ALPHA = 1.2  # Try 1.2 for gentle ESS or 2.5 for stronger high-mode scaling.
QI_STAGE_MODE_POLICY = "lower"  # "lower" uses QS-style repeated ladder; "repeat" repeats only final mode.
TARGET_ASPECT = 5.0
QI_TARGET_ASPECT = TARGET_ASPECT
TARGET_ABS_IOTA_MIN = 0.41
HIGH_PRIORITY_IOTA_WEIGHT = 200.0
OPTIONAL_LGRADB_THRESHOLD = 0.30
OPTIONAL_LGRADB_WEIGHT = 0.0
OPTIONAL_QI_LGRADB_WEIGHT = 0.0
STELLARATOR_ASYMMETRIC = False
ASYMMETRIC_SEED = 1.0e-7
DIAGNOSTIC_BUDGETS = False
GPU_PRODUCTION_INNER_MAX_ITER = 180
GPU_PRODUCTION_INNER_FTOL = 1e-9
GPU_PRODUCTION_TRIAL_MAX_ITER = 180
GPU_PRODUCTION_TRIAL_FTOL = 1e-9
PRODUCTION_AUTO_SCALAR_MIN_MODE = 3
PRODUCTION_AUTO_SCALAR_METHOD = "auto_scalar"


@dataclass(frozen=True)
class CaseBudget:
    max_nfev: int | None = None
    continuation_nfev: int | None = None
    inner_max_iter: int | None = None
    inner_ftol: float | None = None
    trial_max_iter: int | None = None
    trial_ftol: float | None = None


# Bounded policies for known poor/runaway diagnostic cases. These are opt-in via
# ``--diagnostic-budgets`` and intentionally very small, so they remain useful
# for CI/render smoke tests without being mistaken for production results.
CASE_BUDGET_OVERRIDES: dict[tuple[str, str, str, int, bool], CaseBudget] = {
    # QA needs a moderately converged inner solve before the iota residual has a
    # useful derivative. The old 40-iteration GPU diagnostic cap kept QA on the
    # zero-iota branch. These budgets were selected to keep each GPU QA case
    # below the old short sweep timeout while moving continuation/ESS cases to the
    # target-iota basin.
    ("gpu", "continuation", "qa", 1, False): CaseBudget(
        max_nfev=12, continuation_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "continuation", "qa", 1, True): CaseBudget(
        max_nfev=12, continuation_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "continuation", "qa", 2, False): CaseBudget(
        max_nfev=8, continuation_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "continuation", "qa", 2, True): CaseBudget(
        max_nfev=8, continuation_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "continuation", "qa", 3, False): CaseBudget(
        max_nfev=6, continuation_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "continuation", "qa", 3, True): CaseBudget(
        max_nfev=6, continuation_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "direct", "qa", 1, False): CaseBudget(
        max_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "direct", "qa", 1, True): CaseBudget(
        max_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "direct", "qa", 2, False): CaseBudget(
        max_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "direct", "qa", 2, True): CaseBudget(
        max_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "direct", "qa", 3, False): CaseBudget(
        max_nfev=12, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "direct", "qa", 3, True): CaseBudget(
        max_nfev=24, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("cpu", "direct", "qa", 3, False): CaseBudget(
        max_nfev=24, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("cpu", "direct", "qa", 3, True): CaseBudget(
        max_nfev=24, inner_max_iter=120, inner_ftol=1e-8, trial_max_iter=120, trial_ftol=1e-8
    ),
    ("gpu", "direct", "qh", 2, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qh", 3, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qp", 2, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qp", 3, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qh", 3, True): CaseBudget(max_nfev=5, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qp", 3, True): CaseBudget(max_nfev=5, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
}


# GPU production sweeps use exact discrete-adjoint callbacks with moderately
# strict accepted/trial VMEC solves, but production quality should not be
# controlled by small per-case nfev caps.  Let the 30-minute case timeout decide
# whether high-mode LASYM cases are affordable; keep explicit low nfev caps only
# in ``CASE_BUDGET_OVERRIDES`` for opt-in ``--diagnostic-budgets`` runs.
GPU_PRODUCTION_BUDGET_OVERRIDES: dict[tuple[str, str, str, int, bool], CaseBudget] = {}


# CPU sweeps normally use the full scientific budgets for the published modes.
CPU_PRODUCTION_BUDGET_OVERRIDES: dict[tuple[str, str, str, int, bool], CaseBudget] = {}


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    input_file: Path
    method: str
    scipy_tr_solver: str | None
    scipy_lsmr_maxiter: int | None
    max_nfev: int
    continuation_nfev: int
    ftol: float
    gtol: float
    xtol: float
    ess_alpha: float
    target_aspect: float
    target_iota: float | None
    surfaces: np.ndarray
    helicity_m: int
    helicity_n: int
    inner_max_iter: int
    inner_ftol: float
    trial_max_iter: int
    trial_ftol: float
    iota_abs_min: float | None = None
    iota_floor_softness: float = 1.0e-3
    aspect_weight: float = 1.0
    iota_weight: float = 1.0
    iota_floor_weight: float | None = None
    qs_weight: float = 1.0
    lgradb_threshold: float = 0.30
    lgradb_weight: float = 0.0
    lgradb_ntheta: int = 9
    lgradb_nphi: int = 7
    lgradb_surface_index: int = -1
    lgradb_smooth_penalty: float = 1.0e-3
    objective_kind: str = "qs"
    qi_mboz: int = 12
    qi_nboz: int = 12
    qi_nphi: int = 101
    qi_nalpha: int = 21
    qi_n_bounce: int = 31
    qi_softness: float = 2.0e-2
    qi_width_weight: float = 1.0
    qi_branch_width_weight: float = 0.5
    qi_branch_width_softness: float = 1.0e-2
    qi_profile_weight: float = 0.1
    qi_shuffle_profile_weight: float = 0.0
    qi_shuffle_profile_softness: float = 2.0e-2
    qi_ceiling_max: float = 2.0e-3
    qi_ceiling_weight: float = 0.0
    qi_ceiling_smooth_penalty: float = 2.0e-3
    qi_aligned_profile_weight: float = 0.0
    qi_aligned_profile_softness: float = 2.0e-2
    qi_aligned_profile_trap_level: float = 0.65
    qi_aligned_profile_trap_softness: float = 5.0e-2
    qi_phimin: float = 0.0
    qi_jit_booz: bool = True
    qi_max_mirror_ratio: float = 0.21
    qi_mirror_weight: float = 10.0
    qi_mirror_ntheta: int = 96
    qi_mirror_nphi: int = 96
    qi_mirror_surface_index: int | None = None
    qi_max_elongation: float = 8.0
    qi_elongation_weight: float = 10.0
    qi_elongation_ntheta: int = 48
    qi_elongation_nphi: int = 16
    qi_lgradb_threshold: float = 0.30
    qi_lgradb_weight: float = 0.0
    qi_lgradb_ntheta: int = 9
    qi_lgradb_nphi: int = 7
    qi_lgradb_surface_index: int = -1
    qi_lgradb_smooth_penalty: float = 1.0e-3
    qi_preseed_qp: bool = False
    qi_preseed_qi: bool = False
    qi_preseed_qi_nfev: int = 30
    project_input_boundary_to_max_mode: bool = False
    min_vmec_mode: int = MIN_VMEC_MODE


PROBLEM_CONFIGS = {
    "qa": ProblemConfig(
        name="qa",
        input_file=DATA_DIR / "input.nfp2_QA_omnigenity",
        method="scipy",  # Try also "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
        scipy_tr_solver="lsmr",  # For method="scipy": "lsmr" is memory-light; "exact" is dense.
        scipy_lsmr_maxiter=None,  # None lets SciPy choose; set an int to cap LSMR work per step.
        max_nfev=70,  # Outer least-squares budget for the final stage.
        continuation_nfev=70,  # Per-stage budget when mode continuation is enabled.
        ftol=1e-4,  # Relative cost-reduction tolerance for the outer optimizer.
        gtol=1e-4,  # Gradient optimality tolerance for the outer optimizer.
        xtol=1e-4,  # Step-size tolerance for the outer optimizer.
        ess_alpha=ESS_ALPHA,
        target_aspect=TARGET_ASPECT,
        target_iota=0.42,
        iota_weight=100.0,
        surfaces=np.arange(0.0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=0,
        inner_max_iter=120,  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
        inner_ftol=1e-9,  # Accepted-point VMEC tolerance; 0 uses FTOL_ARRAY from the input deck.
        trial_max_iter=120,  # Trial-point VMEC iterations; lower this for faster diagnostics.
        trial_ftol=1e-9,  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.
        lgradb_threshold=OPTIONAL_LGRADB_THRESHOLD,
        lgradb_weight=OPTIONAL_LGRADB_WEIGHT,
        min_vmec_mode=6,
    ),
    "qh": ProblemConfig(
        name="qh",
        input_file=DATA_DIR / "input.nfp4_QH_warm_start",
        method="scipy",  # Try also "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
        scipy_tr_solver="lsmr",  # For method="scipy": "lsmr" is memory-light; "exact" is dense.
        scipy_lsmr_maxiter=None,  # None lets SciPy choose; set an int to cap LSMR work per step.
        max_nfev=70,  # Outer least-squares budget for the final stage.
        continuation_nfev=70,  # Per-stage budget when mode continuation is enabled.
        ftol=1e-4,  # Relative cost-reduction tolerance for the outer optimizer.
        gtol=1e-4,  # Gradient optimality tolerance for the outer optimizer.
        xtol=1e-4,  # Step-size tolerance for the outer optimizer.
        ess_alpha=ESS_ALPHA,
        target_aspect=TARGET_ASPECT,
        target_iota=None,
        iota_abs_min=TARGET_ABS_IOTA_MIN,
        surfaces=np.arange(0.0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
        inner_max_iter=120,  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
        inner_ftol=1e-9,  # Accepted-point VMEC tolerance; 0 uses FTOL_ARRAY from the input deck.
        trial_max_iter=120,  # Trial-point VMEC iterations; lower this for faster diagnostics.
        trial_ftol=1e-9,  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.
        lgradb_threshold=OPTIONAL_LGRADB_THRESHOLD,
        iota_weight=HIGH_PRIORITY_IOTA_WEIGHT,
        lgradb_weight=OPTIONAL_LGRADB_WEIGHT,
        min_vmec_mode=6,
    ),
    "qp": ProblemConfig(
        name="qp",
        input_file=DATA_DIR / "input.nfp2_QI",
        method="scipy",  # Try also "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
        scipy_tr_solver="lsmr",  # For method="scipy": "lsmr" is memory-light; "exact" is dense.
        scipy_lsmr_maxiter=None,  # None lets SciPy choose; set an int to cap LSMR work per step.
        max_nfev=70,  # Outer least-squares budget for the final stage.
        continuation_nfev=70,  # Per-stage budget when mode continuation is enabled.
        ftol=1e-4,  # Relative cost-reduction tolerance for the outer optimizer.
        gtol=1e-4,  # Gradient optimality tolerance for the outer optimizer.
        xtol=1e-4,  # Step-size tolerance for the outer optimizer.
        ess_alpha=ESS_ALPHA,
        target_aspect=TARGET_ASPECT,
        target_iota=None,
        surfaces=np.arange(0.0, 1.01, 0.1),
        helicity_m=0,
        helicity_n=-1,
        inner_max_iter=120,  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
        inner_ftol=1e-9,  # Accepted-point VMEC tolerance; 0 uses FTOL_ARRAY from the input deck.
        trial_max_iter=120,  # Trial-point VMEC iterations; lower this for faster diagnostics.
        trial_ftol=1e-9,  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.
        iota_abs_min=TARGET_ABS_IOTA_MIN,
        iota_weight=HIGH_PRIORITY_IOTA_WEIGHT,
        project_input_boundary_to_max_mode=True,
        lgradb_threshold=OPTIONAL_LGRADB_THRESHOLD,
        lgradb_weight=OPTIONAL_LGRADB_WEIGHT,
        min_vmec_mode=6,
    ),
    "qi": ProblemConfig(
        name="qi",
        input_file=DATA_DIR / "input.minimal_seed_nfp2",
        method="scipy_matrix_free",  # Try also "scipy", "gauss_newton", "lbfgs_adjoint", or "scalar_trust".
        scipy_tr_solver="lsmr",  # For method="scipy": "lsmr" is memory-light; "exact" is dense.
        scipy_lsmr_maxiter=None,  # None lets SciPy choose; set an int to cap LSMR work per step.
        max_nfev=70,  # Outer least-squares budget for the final stage.
        continuation_nfev=70,  # Per-stage budget when mode continuation is enabled.
        ftol=1e-4,  # Relative cost-reduction tolerance for the outer optimizer.
        gtol=1e-4,  # Gradient optimality tolerance for the outer optimizer.
        xtol=1e-4,  # Step-size tolerance for the outer optimizer.
        # The reference omnigenity workflow uses a gentler alpha than the QS
        # examples.  QI shares the common aspect-ratio target with QA/QH/QP so
        # README and sweep comparisons use the same compactness policy.
        ess_alpha=1.2,
        target_aspect=QI_TARGET_ASPECT,
        aspect_weight=1.0,
        target_iota=None,
        surfaces=np.linspace(0.1, 1.0, 6),
        helicity_m=0,
        helicity_n=0,
        inner_max_iter=120,  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
        inner_ftol=1e-9,  # Accepted-point VMEC tolerance; 0 uses FTOL_ARRAY from the input deck.
        trial_max_iter=120,  # Trial-point VMEC iterations; lower this for faster diagnostics.
        trial_ftol=1e-9,  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.
        iota_abs_min=TARGET_ABS_IOTA_MIN,
        iota_weight=HIGH_PRIORITY_IOTA_WEIGHT,
        objective_kind="qi",
        qi_mboz=18,
        qi_nboz=18,
        qi_nphi=151,
        qi_nalpha=31,
        qi_n_bounce=51,
        qi_softness=2.0e-2,
        qi_width_weight=1.0,
        qi_branch_width_weight=0.5,
        qi_branch_width_softness=2.0e-2,
        qi_profile_weight=0.1,
        qi_shuffle_profile_weight=1.0,
        qi_shuffle_profile_softness=2.0e-2,
        # Guard mirror/elongation cleanup so the sweep cannot promote a
        # mirror-clean state that leaves the accepted low-QI basin.
        qi_ceiling_max=2.0e-3,
        qi_ceiling_weight=100.0,
        qi_ceiling_smooth_penalty=2.0e-3,
        qi_aligned_profile_weight=0.0,
        qi_aligned_profile_softness=2.0e-2,
        qi_aligned_profile_trap_level=0.65,
        qi_aligned_profile_trap_softness=5.0e-2,
        qi_max_mirror_ratio=0.30,
        # Match the reference SIMSOPT QI script: MirrorRatioPen and
        # MaxElongationPen are scalar residuals with least-squares weight 1e1.
        qi_mirror_weight=math.sqrt(20.0),
        qi_max_elongation=8.2,
        qi_elongation_weight=math.sqrt(10.0),
        # Optional LgradB term.  Keep it inactive by default so symmetry/QI,
        # iota, aspect, mirror ratio, and elongation set the optimization path.
        qi_lgradb_threshold=OPTIONAL_LGRADB_THRESHOLD,
        qi_lgradb_weight=OPTIONAL_QI_LGRADB_WEIGHT,
        qi_lgradb_ntheta=9,
        qi_lgradb_nphi=7,
        qi_lgradb_surface_index=-1,
        qi_preseed_qp=False,
        qi_preseed_qi=True,
        qi_preseed_qi_nfev=30,
        project_input_boundary_to_max_mode=True,
        min_vmec_mode=6,
    ),
}


@dataclass
class CaseResult:
    backend: str
    problem: str
    max_mode: int
    use_ess: bool
    success: bool
    crashed: bool
    message: str
    policy: str = "continuation"
    objective_final: float | None = None
    qs_final: float | None = None
    aspect_final: float | None = None
    iota_final: float | None = None
    nfev: int | None = None
    njev: int | None = None
    total_wall_time_s: float | None = None
    profile_wall_time_s: float | None = None
    profile_top_name: str | None = None
    profile_top_wall_time_s: float | None = None
    profile_solve_forward_trial_total_wall_time_s: float | None = None
    profile_solve_forward_exact_total_wall_time_s: float | None = None
    profile_exact_tape_build_wall_time_s: float | None = None
    profile_exact_tape_build_solve_call_wall_time_s: float | None = None
    profile_exact_tape_build_unattributed_wall_time_s: float | None = None
    profile_exact_tape_solver_compute_forces_first_wall_time_s: float | None = None
    profile_exact_tape_solver_compute_forces_rest_wall_time_s: float | None = None
    profile_trial_solver_scan_total_wall_time_s: float | None = None
    profile_trial_solver_scan_runner_cache_lookup_wall_time_s: float | None = None
    profile_trial_solver_scan_runner_cache_build_wall_time_s: float | None = None
    profile_trial_solver_scan_runner_cache_hit_count: float | None = None
    profile_trial_solver_scan_runner_cache_miss_count: float | None = None
    profile_trial_solver_scan_runner_cache_bypass_count: float | None = None
    profile_trial_solver_scan_runner_cache_hit_device_run_wall_time_s: float | None = None
    profile_trial_solver_scan_runner_cache_miss_device_run_wall_time_s: float | None = None
    profile_trial_solver_scan_runner_cache_bypass_device_run_wall_time_s: float | None = None
    profile_trial_solver_scan_device_dispatch_wall_time_s: float | None = None
    profile_trial_solver_scan_device_ready_wall_time_s: float | None = None
    profile_trial_solver_scan_host_materialize_wall_time_s: float | None = None
    profile_jacobian_total_wall_time_s: float | None = None
    profile_write_wout_wall_time_s: float | None = None
    output_dir: str | None = None
    jax_backend: str | None = None
    jax_device_kind: str | None = None
    solver_device: str | None = None
    jax_platforms: str | None = None
    stellarator_asymmetric: bool = False
    asymmetry_seed: float = 0.0
    input_file: str | None = None
    input_nfp: int | None = None
    project_input_boundary_to_max_mode: bool | None = None
    target_aspect: float | None = None
    target_iota: float | None = None
    iota_abs_min: float | None = None
    iota_weight: float | None = None
    lgradb_weight: float | None = None
    qi_lgradb_weight: float | None = None
    asymmetric_dof_count: int = 0
    asymmetric_param_norm_initial: float | None = None
    asymmetric_param_norm_final: float | None = None
    asymmetric_param_norm_delta: float | None = None
    bmag_min: float | None = None
    bmag_max: float | None = None
    bmag_nonpositive_fraction: float | None = None
    bmag_finite: bool | None = None
    lgradb_min: float | None = None
    lgradb_threshold: float | None = None
    lgradb_excess_max: float | None = None
    lgradb_diagnostic_error: str | None = None
    qi_qp_preseed: bool | None = None
    qi_qi_preseed: bool | None = None
    qi_jit_booz: bool | None = None
    qi_raw_total: float | None = None
    qi_legacy_total: float | None = None
    qi_mirror_ratio_max: float | None = None
    qi_mirror_ratio_target: float | None = None
    qi_mirror_excess_max: float | None = None
    qi_max_elongation: float | None = None
    qi_elongation_target: float | None = None
    qi_elongation_excess: float | None = None
    qi_lgradb_min: float | None = None
    qi_lgradb_threshold: float | None = None
    qi_lgradb_excess_max: float | None = None
    qi_diagnostic_error: str | None = None


def _set_missing_wall_time(result: CaseResult, elapsed_s: float) -> bool:
    """Record outer worker elapsed time when the inner run could not report it."""
    if result.total_wall_time_s is not None:
        return False
    result.total_wall_time_s = float(elapsed_s)
    return True


def _mark_timed_out_result(result: CaseResult, *, elapsed_s: float, case_timeout_s: float | None) -> bool:
    """Mark an existing partial result as timed out without dropping metrics."""

    timeout_s = 0.0 if case_timeout_s is None else float(case_timeout_s)
    timeout_message = f"worker timed out after {timeout_s:.1f} s"
    message = str(result.message or "")
    if timeout_message not in message:
        result.message = timeout_message if not message else f"{timeout_message}; {message}"
    result.crashed = True
    if result.total_wall_time_s is None or float(result.total_wall_time_s) < float(elapsed_s):
        result.total_wall_time_s = float(elapsed_s)
    return True


def _atomic_write_json(path: Path, payload: object) -> None:
    """Atomically replace a JSON file so killed workers do not leave torn output."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _profile_wall_time(profile: dict, key: str) -> float | None:
    """Extract one profile bucket wall time from a history profile dictionary."""

    value = profile.get(key)
    if isinstance(value, dict):
        wall = value.get("wall_time_s")
    else:
        wall = value
    if wall is None:
        return None
    try:
        wall_f = float(wall)
    except (TypeError, ValueError):
        return None
    return wall_f if math.isfinite(wall_f) else None


def _profile_summary_fields(history: dict | None) -> dict[str, object]:
    """Return compact profile fields suitable for case_result/CSV summaries."""

    profile = history.get("profile") if isinstance(history, dict) else None
    if not isinstance(profile, dict) or not profile:
        return {
            "profile_wall_time_s": None,
            "profile_top_name": None,
            "profile_top_wall_time_s": None,
            "profile_solve_forward_trial_total_wall_time_s": None,
            "profile_solve_forward_exact_total_wall_time_s": None,
            "profile_exact_tape_build_wall_time_s": None,
            "profile_exact_tape_build_solve_call_wall_time_s": None,
            "profile_exact_tape_build_unattributed_wall_time_s": None,
            "profile_exact_tape_solver_compute_forces_first_wall_time_s": None,
            "profile_exact_tape_solver_compute_forces_rest_wall_time_s": None,
            "profile_trial_solver_scan_total_wall_time_s": None,
            "profile_trial_solver_scan_runner_cache_lookup_wall_time_s": None,
            "profile_trial_solver_scan_runner_cache_build_wall_time_s": None,
            "profile_trial_solver_scan_runner_cache_hit_count": None,
            "profile_trial_solver_scan_runner_cache_miss_count": None,
            "profile_trial_solver_scan_runner_cache_bypass_count": None,
            "profile_trial_solver_scan_runner_cache_hit_device_run_wall_time_s": None,
            "profile_trial_solver_scan_runner_cache_miss_device_run_wall_time_s": None,
            "profile_trial_solver_scan_runner_cache_bypass_device_run_wall_time_s": None,
            "profile_trial_solver_scan_device_dispatch_wall_time_s": None,
            "profile_trial_solver_scan_device_ready_wall_time_s": None,
            "profile_trial_solver_scan_host_materialize_wall_time_s": None,
            "profile_jacobian_total_wall_time_s": None,
            "profile_write_wout_wall_time_s": None,
        }

    entries: list[tuple[str, float]] = []
    for key in sorted(profile):
        wall = _profile_wall_time(profile, str(key))
        if wall is not None:
            entries.append((str(key), wall))
    top_name = None
    top_wall = None
    if entries:
        top_name, top_wall = max(entries, key=lambda item: item[1])
    return {
        "profile_wall_time_s": float(sum(wall for _key, wall in entries)) if entries else 0.0,
        "profile_top_name": top_name,
        "profile_top_wall_time_s": top_wall,
        "profile_solve_forward_trial_total_wall_time_s": _profile_wall_time(
            profile, "solve_forward_trial_total"
        ),
        "profile_solve_forward_exact_total_wall_time_s": _profile_wall_time(
            profile, "solve_forward_exact_total"
        ),
        "profile_exact_tape_build_wall_time_s": _profile_wall_time(profile, "exact_tape_build"),
        "profile_exact_tape_build_solve_call_wall_time_s": _profile_wall_time(
            profile, "exact_tape_build_solve_call"
        ),
        "profile_exact_tape_build_unattributed_wall_time_s": _profile_wall_time(
            profile, "exact_tape_build_unattributed"
        ),
        "profile_exact_tape_solver_compute_forces_first_wall_time_s": _profile_wall_time(
            profile, "exact_tape_solver_compute_forces_first"
        ),
        "profile_exact_tape_solver_compute_forces_rest_wall_time_s": _profile_wall_time(
            profile, "exact_tape_solver_compute_forces_rest"
        ),
        "profile_trial_solver_scan_total_wall_time_s": _profile_wall_time(profile, "trial_solver_scan_total"),
        "profile_trial_solver_scan_runner_cache_lookup_wall_time_s": _profile_wall_time(
            profile, "trial_solver_scan_runner_cache_lookup"
        ),
        "profile_trial_solver_scan_runner_cache_build_wall_time_s": _profile_wall_time(
            profile, "trial_solver_scan_runner_cache_build"
        ),
        "profile_trial_solver_scan_runner_cache_hit_count": _profile_wall_time(
            profile, "trial_solver_scan_runner_cache_hit_count"
        ),
        "profile_trial_solver_scan_runner_cache_miss_count": _profile_wall_time(
            profile, "trial_solver_scan_runner_cache_miss_count"
        ),
        "profile_trial_solver_scan_runner_cache_bypass_count": _profile_wall_time(
            profile, "trial_solver_scan_runner_cache_bypass_count"
        ),
        "profile_trial_solver_scan_runner_cache_hit_device_run_wall_time_s": _profile_wall_time(
            profile, "trial_solver_scan_runner_cache_hit_device_run"
        ),
        "profile_trial_solver_scan_runner_cache_miss_device_run_wall_time_s": _profile_wall_time(
            profile, "trial_solver_scan_runner_cache_miss_device_run"
        ),
        "profile_trial_solver_scan_runner_cache_bypass_device_run_wall_time_s": _profile_wall_time(
            profile, "trial_solver_scan_runner_cache_bypass_device_run"
        ),
        "profile_trial_solver_scan_device_dispatch_wall_time_s": _profile_wall_time(
            profile, "trial_solver_scan_device_dispatch"
        ),
        "profile_trial_solver_scan_device_ready_wall_time_s": _profile_wall_time(
            profile, "trial_solver_scan_device_ready"
        ),
        "profile_trial_solver_scan_host_materialize_wall_time_s": _profile_wall_time(
            profile, "trial_solver_scan_host_materialize"
        ),
        "profile_jacobian_total_wall_time_s": _profile_wall_time(profile, "jacobian_total"),
        "profile_write_wout_wall_time_s": _profile_wall_time(profile, "write_wout"),
    }


def _start_worker_session() -> None:
    """Put a worker in its own session/process group when the OS supports it."""

    setsid = getattr(os, "setsid", None)
    if setsid is None:
        return
    try:
        setsid()
    except OSError:
        # The worker still runs correctly without a private process group; the
        # parent falls back to killing the direct child PID if process-group
        # termination is unavailable.
        return


def _terminate_worker_process(proc: mp.Process, *, terminate_timeout_s: float = 10.0) -> None:
    """Terminate a worker and its process group without leaving GPU children alive."""

    pid = proc.pid

    def _signal_process_group(sig: int) -> bool:
        if pid is None or not hasattr(os, "killpg"):
            return False
        try:
            os.killpg(pid, sig)
            return True
        except ProcessLookupError:
            return True
        except OSError:
            return False

    if not proc.is_alive():
        # The direct multiprocessing child can exit while subprocesses it
        # launched remain alive in its process group.  Still signal the group
        # before returning so timeout cleanup does not leak GPU jobs.
        _signal_process_group(signal.SIGTERM)
        _signal_process_group(signal.SIGKILL)
        proc.join(timeout=0.0)
        return
    if not _signal_process_group(signal.SIGTERM):
        proc.terminate()
    proc.join(timeout=float(terminate_timeout_s))
    if not proc.is_alive():
        _signal_process_group(signal.SIGTERM)
        _signal_process_group(signal.SIGKILL)
        return
    if not _signal_process_group(signal.SIGKILL):
        try:
            proc.kill()
        except AttributeError:  # pragma: no cover - Python < 3.7 fallback.
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
    proc.join()


def _ess_label(use_ess: bool) -> str:
    return "ess" if use_ess else "no_ess"


def _panel_label(index: int) -> str:
    label = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def _ess_alpha_for_case(problem_cfg: ProblemConfig, problem: str, max_mode: int, use_ess: bool) -> float:
    if not use_ess:
        return 0.0
    return float(problem_cfg.ess_alpha)


def _sweep_backend_key(backend: str) -> str:
    backend_name = str(backend).strip().lower()
    if backend_name.startswith(("gpu", "cuda", "rocm")):
        return "gpu"
    if backend_name.startswith("cpu"):
        return "cpu"
    return backend_name


def _use_production_auto_scalar_method(
    problem_cfg: ProblemConfig,
    *,
    backend_key: str,
    max_mode: int,
    diagnostic_budgets: bool,
) -> bool:
    if bool(diagnostic_budgets):
        return False
    if backend_key not in {"cpu", "gpu"}:
        return False
    if int(max_mode) < PRODUCTION_AUTO_SCALAR_MIN_MODE:
        return False
    return str(problem_cfg.objective_kind) in {"qs", "qi"}


def _effective_problem_config(
    problem_cfg: ProblemConfig,
    *,
    backend: str,
    policy: str,
    problem: str,
    max_mode: int,
    use_ess: bool,
    diagnostic_budgets: bool = DIAGNOSTIC_BUDGETS,
    cli_budget: CaseBudget | None = None,
    ess_alpha_override: float | None = None,
) -> ProblemConfig:
    updates = {}
    backend_key = _sweep_backend_key(backend)
    if _use_production_auto_scalar_method(
        problem_cfg,
        backend_key=backend_key,
        max_mode=max_mode,
        diagnostic_budgets=diagnostic_budgets,
    ):
        updates["method"] = PRODUCTION_AUTO_SCALAR_METHOD
    if (not diagnostic_budgets) and backend_key == "gpu":
        updates.update(
            inner_max_iter=(
                GPU_PRODUCTION_INNER_MAX_ITER
                if int(problem_cfg.inner_max_iter) <= 0
                else min(int(problem_cfg.inner_max_iter), GPU_PRODUCTION_INNER_MAX_ITER)
            ),
            inner_ftol=max(float(problem_cfg.inner_ftol), GPU_PRODUCTION_INNER_FTOL),
            trial_max_iter=min(int(problem_cfg.trial_max_iter), GPU_PRODUCTION_TRIAL_MAX_ITER),
            trial_ftol=max(float(problem_cfg.trial_ftol), GPU_PRODUCTION_TRIAL_FTOL),
        )
        budget = GPU_PRODUCTION_BUDGET_OVERRIDES.get(
            (backend_key, str(policy), str(problem), int(max_mode), bool(use_ess))
        )
        if budget is not None:
            if budget.max_nfev is not None:
                updates["max_nfev"] = min(int(problem_cfg.max_nfev), int(budget.max_nfev))
            if budget.continuation_nfev is not None:
                updates["continuation_nfev"] = min(int(problem_cfg.continuation_nfev), int(budget.continuation_nfev))
            if budget.inner_max_iter is not None:
                updates["inner_max_iter"] = int(budget.inner_max_iter)
            if budget.inner_ftol is not None:
                updates["inner_ftol"] = float(budget.inner_ftol)
            if budget.trial_max_iter is not None:
                updates["trial_max_iter"] = int(budget.trial_max_iter)
            if budget.trial_ftol is not None:
                updates["trial_ftol"] = float(budget.trial_ftol)
    if (not diagnostic_budgets) and backend_key == "cpu":
        budget = CPU_PRODUCTION_BUDGET_OVERRIDES.get(
            (backend_key, str(policy), str(problem), int(max_mode), bool(use_ess))
        )
        if budget is not None:
            if budget.max_nfev is not None:
                updates["max_nfev"] = min(int(problem_cfg.max_nfev), int(budget.max_nfev))
            if budget.continuation_nfev is not None:
                updates["continuation_nfev"] = min(int(problem_cfg.continuation_nfev), int(budget.continuation_nfev))
            if budget.inner_max_iter is not None:
                updates["inner_max_iter"] = int(budget.inner_max_iter)
            if budget.inner_ftol is not None:
                updates["inner_ftol"] = float(budget.inner_ftol)
            if budget.trial_max_iter is not None:
                updates["trial_max_iter"] = int(budget.trial_max_iter)
            if budget.trial_ftol is not None:
                updates["trial_ftol"] = float(budget.trial_ftol)
    if diagnostic_budgets and backend_key == "gpu":
        # GPU callbacks are still cold-compile/dispatch dominated. Keep the
        # diagnostic panel finite and comparable by bounding every GPU case.
        updates.update(
            max_nfev=min(int(problem_cfg.max_nfev), 5),
            continuation_nfev=min(int(problem_cfg.continuation_nfev), 2),
            inner_max_iter=40
            if int(problem_cfg.inner_max_iter) == 0
            else min(int(problem_cfg.inner_max_iter), 40),
            inner_ftol=max(float(problem_cfg.inner_ftol), 1e-8),
            trial_max_iter=min(int(problem_cfg.trial_max_iter), 40),
            trial_ftol=max(float(problem_cfg.trial_ftol), 1e-8),
        )
    budget = None
    if diagnostic_budgets:
        budget = CASE_BUDGET_OVERRIDES.get(
            (backend_key, str(policy), str(problem), int(max_mode), bool(use_ess))
        )
    if budget is None and not updates:
        return problem_cfg
    if budget is not None and budget.max_nfev is not None:
        updates["max_nfev"] = int(budget.max_nfev)
    if budget is not None and budget.continuation_nfev is not None:
        updates["continuation_nfev"] = int(budget.continuation_nfev)
    if budget is not None and budget.inner_max_iter is not None:
        updates["inner_max_iter"] = int(budget.inner_max_iter)
    if budget is not None and budget.inner_ftol is not None:
        updates["inner_ftol"] = float(budget.inner_ftol)
    if budget is not None and budget.trial_max_iter is not None:
        updates["trial_max_iter"] = int(budget.trial_max_iter)
    if budget is not None and budget.trial_ftol is not None:
        updates["trial_ftol"] = float(budget.trial_ftol)
    if cli_budget is not None:
        if cli_budget.max_nfev is not None:
            updates["max_nfev"] = int(cli_budget.max_nfev)
        if cli_budget.continuation_nfev is not None:
            updates["continuation_nfev"] = int(cli_budget.continuation_nfev)
        if cli_budget.inner_max_iter is not None:
            updates["inner_max_iter"] = int(cli_budget.inner_max_iter)
        if cli_budget.inner_ftol is not None:
            updates["inner_ftol"] = float(cli_budget.inner_ftol)
        if cli_budget.trial_max_iter is not None:
            updates["trial_max_iter"] = int(cli_budget.trial_max_iter)
        if cli_budget.trial_ftol is not None:
            updates["trial_ftol"] = float(cli_budget.trial_ftol)
    if ess_alpha_override is not None:
        updates["ess_alpha"] = float(ess_alpha_override)
    return replace(problem_cfg, **updates)


def _copy_indata_with_lasym(indata: InData, *, lasym: bool) -> InData:
    """Return an input-deck copy with ``LASYM`` set explicitly."""

    scalars = dict(indata.scalars)
    scalars["LASYM"] = bool(lasym)
    indexed = {key: dict(values) for key, values in indata.indexed.items()}
    return InData(scalars=scalars, indexed=indexed, source_path=indata.source_path)


def _boundary_include_for_indata(indata: InData) -> tuple[str, ...]:
    """Boundary coefficient families to optimize for the input symmetry."""

    return ("rc", "zs", "rs", "zc") if bool(indata.get_bool("LASYM", False)) else ("rc", "zs")


def _spec_base_value(boundary, spec) -> float:
    arrays = {
        "rc": boundary.R_cos,
        "rs": boundary.R_sin,
        "zc": boundary.Z_cos,
        "zs": boundary.Z_sin,
    }
    return float(np.asarray(arrays[spec.kind], dtype=float)[int(spec.index)])


def _seed_zero_asymmetric_params(
    *,
    boundary_input,
    specs,
    params,
    seed: float,
) -> np.ndarray:
    """Deterministically excite zero ``RBS``/``ZBC`` modes for LASYM runs."""

    out = np.asarray(params, dtype=float).copy()
    if float(seed) == 0.0:
        return out
    for index, spec in enumerate(specs):
        if spec.kind not in ("rs", "zc"):
            continue
        if abs(_spec_base_value(boundary_input, spec) + float(out[index])) == 0.0:
            out[index] = float(seed)
    return out


def _asymmetric_param_stats(specs, params_initial, params_final) -> dict[str, float | int | None]:
    indices = [index for index, spec in enumerate(specs) if spec.kind in ("rs", "zc")]
    if not indices:
        return {
            "asymmetric_dof_count": 0,
            "asymmetric_param_norm_initial": None,
            "asymmetric_param_norm_final": None,
            "asymmetric_param_norm_delta": None,
        }
    idx = np.asarray(indices, dtype=int)
    initial = np.asarray(params_initial, dtype=float)[idx]
    final = np.asarray(params_final, dtype=float)[idx]
    return {
        "asymmetric_dof_count": int(idx.size),
        "asymmetric_param_norm_initial": float(np.linalg.norm(initial)),
        "asymmetric_param_norm_final": float(np.linalg.norm(final)),
        "asymmetric_param_norm_delta": float(np.linalg.norm(final - initial)),
    }


def _bmag_lcfs_stats(wout_path: Path) -> dict[str, float | bool | None]:
    """Return basic LCFS |B| reconstruction health metrics for a saved wout."""

    try:
        from vmec_jax.plotting import vmecplot2_bmag_grid
        from vmec_jax.wout import read_wout

        wout = read_wout(str(wout_path))
        ns = int(np.asarray(wout.ns))
        zeta_max = 2.0 * np.pi / int(np.asarray(wout.nfp))
        _theta, _zeta, bmag = vmecplot2_bmag_grid(
            wout,
            s_index=ns - 1,
            ntheta=64,
            nzeta=96,
            zeta_max=zeta_max,
        )
        finite = np.isfinite(bmag)
        if not bool(np.any(finite)):
            return {
                "bmag_min": None,
                "bmag_max": None,
                "bmag_nonpositive_fraction": None,
                "bmag_finite": False,
            }
        bmag_finite = np.asarray(bmag[finite], dtype=float)
        return {
            "bmag_min": float(np.min(bmag_finite)),
            "bmag_max": float(np.max(bmag_finite)),
            "bmag_nonpositive_fraction": float(np.mean(bmag_finite <= 0.0)),
            "bmag_finite": bool(np.all(finite)),
        }
    except Exception:
        return {
            "bmag_min": None,
            "bmag_max": None,
            "bmag_nonpositive_fraction": None,
            "bmag_finite": None,
        }


def _slice_boozer_surfaces(booz: dict, surface_index: int) -> dict:
    """Return a Boozer output dict restricted to one radial surface."""

    index = int(surface_index)
    out = dict(booz)
    for key in ("bmnc_b", "bmns_b", "iota_b", "s_b"):
        value = out.get(key)
        if value is not None:
            out[key] = value[index : index + 1]
    return out


def _mirror_boozer_surfaces(booz: dict, surface_index: int | None) -> dict:
    """Return all Boozer surfaces unless a single-surface ablation is requested."""

    if surface_index is None:
        return booz
    return _slice_boozer_surfaces(booz, int(surface_index))


def _lgradb_diagnostics_from_state(problem_cfg: ProblemConfig, opt, state) -> dict[str, float | str | None]:
    """Evaluate the independent LgradB diagnostic for QS targets."""

    if problem_cfg.objective_kind == "qi" or state is None or float(problem_cfg.lgradb_weight) == 0.0:
        return {}
    try:
        static = opt._static
        indata = opt._indata
        geom = eval_geom(state, static)
        signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
        flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
        lgradb = lgradb_penalty_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            flux_local=flux,
            threshold=problem_cfg.lgradb_threshold,
            s_index=problem_cfg.lgradb_surface_index,
            ntheta=problem_cfg.lgradb_ntheta,
            nphi=problem_cfg.lgradb_nphi,
            smooth_penalty=problem_cfg.lgradb_smooth_penalty,
        )
        lgradb_values = np.asarray(lgradb["L_grad_B"], dtype=float)
        lgradb_excess = np.asarray(lgradb["excess"], dtype=float)
        return {
            "lgradb_min": float(np.min(lgradb_values)),
            "lgradb_threshold": float(problem_cfg.lgradb_threshold),
            "lgradb_excess_max": max(0.0, float(np.max(lgradb_excess))),
        }
    except Exception as exc:
        return {
            "lgradb_min": None,
            "lgradb_threshold": float(problem_cfg.lgradb_threshold),
            "lgradb_excess_max": None,
            "lgradb_diagnostic_error": f"{type(exc).__name__}: {exc}",
        }


def _qi_diagnostics_from_state(problem_cfg: ProblemConfig, opt, state) -> dict[str, float | bool | None]:
    """Evaluate unweighted QI, mirror-ratio, and elongation diagnostics."""

    if problem_cfg.objective_kind != "qi" or state is None:
        return {}
    try:
        from booz_xform_jax import prepare_booz_xform_constants
        from vmec_jax.modes import nyquist_mode_table_from_grid, vmec_mode_table

        static = opt._static
        indata = opt._indata
        geom = eval_geom(state, static)
        signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
        flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
        pressure = jnp.zeros_like(jnp.asarray(static.s))
        main_modes = vmec_mode_table(int(static.cfg.mpol), int(static.cfg.ntor))
        nyq_modes = nyquist_mode_table_from_grid(
            mpol=int(static.cfg.mpol),
            ntor=int(static.cfg.ntor),
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
        )
        constants, grids = prepare_booz_xform_constants(
            nfp=int(static.cfg.nfp),
            mboz=problem_cfg.qi_mboz,
            nboz=problem_cfg.qi_nboz,
            asym=bool(static.cfg.lasym),
            xm=np.asarray(main_modes.m, dtype=int),
            xn=np.asarray(main_modes.n * int(static.cfg.nfp), dtype=int),
            xm_nyq=np.asarray(nyq_modes.m, dtype=int),
            xn_nyq=np.asarray(nyq_modes.n * int(static.cfg.nfp), dtype=int),
        )
        surface_indices = _nearest_half_mesh_indices(
            problem_cfg.surfaces,
            n_half=max(int(np.asarray(static.s).shape[0]) - 1, 1),
        )
        qi = quasi_isodynamic_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            flux_local=flux,
            prof_local={"pressure": pressure},
            pressure_local=pressure,
            surfaces=problem_cfg.surfaces,
            mboz=problem_cfg.qi_mboz,
            nboz=problem_cfg.qi_nboz,
            nphi=problem_cfg.qi_nphi,
            nalpha=problem_cfg.qi_nalpha,
            n_bounce=problem_cfg.qi_n_bounce,
            softness=problem_cfg.qi_softness,
            width_weight=problem_cfg.qi_width_weight,
            branch_width_weight=problem_cfg.qi_branch_width_weight,
            branch_width_softness=problem_cfg.qi_branch_width_softness,
            profile_weight=problem_cfg.qi_profile_weight,
            shuffle_profile_weight=problem_cfg.qi_shuffle_profile_weight,
            shuffle_profile_softness=problem_cfg.qi_shuffle_profile_softness,
            aligned_profile_weight=problem_cfg.qi_aligned_profile_weight,
            aligned_profile_softness=problem_cfg.qi_aligned_profile_softness,
            aligned_profile_trap_level=problem_cfg.qi_aligned_profile_trap_level,
            aligned_profile_trap_softness=problem_cfg.qi_aligned_profile_trap_softness,
            phimin=problem_cfg.qi_phimin,
            jit_booz=bool(problem_cfg.qi_jit_booz),
            booz_constants=constants,
            booz_grids=grids,
            surface_indices=surface_indices,
        )
        mirror_booz = _mirror_boozer_surfaces(qi["booz"], problem_cfg.qi_mirror_surface_index)
        mirror = mirror_ratio_penalty_from_boozer_output(
            mirror_booz,
            nfp=int(static.cfg.nfp),
            threshold=problem_cfg.qi_max_mirror_ratio,
            ntheta=problem_cfg.qi_mirror_ntheta,
            nphi=problem_cfg.qi_mirror_nphi,
        )
        elongation = max_elongation_penalty_from_state(
            state=state,
            static=static,
            threshold=problem_cfg.qi_max_elongation,
            ntheta=problem_cfg.qi_elongation_ntheta,
            nphi=problem_cfg.qi_elongation_nphi,
        )
        lgradb = lgradb_penalty_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            flux_local=flux,
            threshold=problem_cfg.qi_lgradb_threshold,
            s_index=problem_cfg.qi_lgradb_surface_index,
            ntheta=problem_cfg.qi_lgradb_ntheta,
            nphi=problem_cfg.qi_lgradb_nphi,
            smooth_penalty=problem_cfg.qi_lgradb_smooth_penalty,
        )
        mirror_values = np.asarray(mirror["mirror_ratio"], dtype=float)
        mirror_max = float(np.max(mirror_values))
        elongation_max = float(np.asarray(elongation["max_elongation"]))
        lgradb_values = np.asarray(lgradb["L_grad_B"], dtype=float)
        lgradb_min = float(np.min(lgradb_values))
        lgradb_excess = np.asarray(lgradb["excess"], dtype=float)
        qi_total = float(np.asarray(qi["total"]))
        legacy_qi = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
            qi["booz"],
            nfp=int(static.cfg.nfp),
            nphi=problem_cfg.qi_nphi,
            nalpha=problem_cfg.qi_nalpha,
            n_bounce=problem_cfg.qi_n_bounce,
            nphi_out=max(401, int(problem_cfg.qi_nphi)),
            phimin=problem_cfg.qi_phimin,
        )
        return {
            "qi_raw_total": qi_total,
            "qi_legacy_total": float(legacy_qi["total"]),
            "qi_mirror_ratio_max": mirror_max,
            "qi_mirror_ratio_target": float(problem_cfg.qi_max_mirror_ratio),
            "qi_mirror_excess_max": max(0.0, mirror_max - float(problem_cfg.qi_max_mirror_ratio)),
            "qi_max_elongation": elongation_max,
            "qi_elongation_target": float(problem_cfg.qi_max_elongation),
            "qi_elongation_excess": max(0.0, elongation_max - float(problem_cfg.qi_max_elongation)),
            "qi_lgradb_min": lgradb_min,
            "qi_lgradb_threshold": float(problem_cfg.qi_lgradb_threshold),
            "qi_lgradb_excess_max": max(0.0, float(np.max(lgradb_excess))),
        }
    except Exception as exc:
        return {
            "qi_raw_total": None,
            "qi_legacy_total": None,
            "qi_mirror_ratio_max": None,
            "qi_mirror_ratio_target": float(problem_cfg.qi_max_mirror_ratio),
            "qi_mirror_excess_max": None,
            "qi_max_elongation": None,
            "qi_elongation_target": float(problem_cfg.qi_max_elongation),
            "qi_elongation_excess": None,
            "qi_lgradb_min": None,
            "qi_lgradb_threshold": float(problem_cfg.qi_lgradb_threshold),
            "qi_lgradb_excess_max": None,
            "qi_diagnostic_error": f"{type(exc).__name__}: {exc}",
        }


def _vmec_resolution_for_max_mode(max_mode: int, *, minimum: int = MIN_VMEC_MODE) -> tuple[int, int]:
    resolution = max(int(minimum), int(max_mode) + 2)
    return resolution, resolution


def _stage_modes_for_problem(
    problem_cfg: ProblemConfig,
    *,
    max_mode: int,
    use_mode_continuation: bool,
) -> list[int]:
    """Return the mode policy used by the standalone optimization sweep.

    The omnigenity reference scripts do not use a single 1->2->3 pass.  Their
    robust path repeatedly re-solves at the same active mode before increasing
    the boundary space.  This reproduces that behavior for the continuation
    lane while keeping direct-start cases as a single max-mode solve.
    """

    max_mode = int(max_mode)
    if (not bool(use_mode_continuation)) or max_mode <= 1:
        return [max_mode]
    if problem_cfg.name in {"qa", "qh", "qp"}:
        modes: list[int] = []
        for mode in range(1, max_mode + 1):
            modes.extend([mode] * (2 if mode == 1 else 3))
        return modes
    if problem_cfg.name == "qi":
        return vj.qi_stage_modes(
            max_mode=max_mode,
            use_mode_continuation=use_mode_continuation,
            continuation_nfev=max(1, int(problem_cfg.continuation_nfev)),
            repeats=5,
            policy=QI_STAGE_MODE_POLICY,
        )
    return list(range(1, max_mode + 1))


def _load_problem(
    cfg: ProblemConfig,
    *,
    max_mode: int,
    stellarator_asymmetric: bool = STELLARATOR_ASYMMETRIC,
):
    cfg0, indata = vj.load_config(str(cfg.input_file))
    del cfg0
    mpol, ntor = _vmec_resolution_for_max_mode(max_mode, minimum=cfg.min_vmec_mode)
    indata = rebuild_indata_with_resolution(indata, mpol=mpol, ntor=ntor)
    if bool(stellarator_asymmetric):
        indata = _copy_indata_with_lasym(indata, lasym=True)
    config = config_from_indata(indata)
    return config, indata


def _jax_runtime_info() -> tuple[str | None, str | None]:
    try:
        import jax

        backend = str(jax.default_backend())
        device_kind = str(jax.devices()[0].device_kind) if jax.devices() else "unknown"
        return backend, device_kind
    except Exception:
        return None, None


def _default_worker_jax_platforms(solver_device: str | None) -> str | None:
    """Choose process-level JAX platform isolation for one worker case.

    Defaults must inherit the user's JAX installation/backend.  CPU-only worker
    isolation is opt-in through an explicit solver-device/platform request.
    """
    name = "auto" if solver_device is None else str(solver_device).strip().lower()
    if name == "cpu":
        return "cpu"
    return None


def _normalize_worker_jax_platforms(value: str | None) -> str | None:
    """Normalize user-facing platform names for the ``JAX_PLATFORMS`` variable.

    ``jax.default_backend()`` reports NVIDIA devices as ``gpu``, but
    ``JAX_PLATFORMS`` expects the concrete backend name ``cuda``.  Accept the
    user-facing ``gpu`` spelling here so worker subprocesses do not fail on
    CUDA-only JAX installs.
    """

    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in ("", "none", "inherit", "auto"):
        return None
    parts = [part.strip() for part in text.split(",")]
    normalized = ["cuda" if part.lower() == "gpu" else part for part in parts if part]
    return ",".join(normalized) if normalized else None


def _build_stage(problem_cfg: ProblemConfig, cfg, indata0, max_mode: int, *, solver_device: str | None):
    if bool(problem_cfg.project_input_boundary_to_max_mode):
        indata0 = vj.truncate_indata_boundary_modes(indata0, max_mode=max_mode)
    stage_static = vj.build_static(cfg)
    stage_boundary = vj.boundary_from_indata(indata0, stage_static.modes, apply_m1_constraint=False)
    stage_indata, stage_static, stage_boundary = vj.extend_boundary_for_max_mode(
        indata0, stage_static, stage_boundary, max_mode
    )
    stage_boundary_input = vj.boundary_input_from_indata(stage_indata, stage_static.modes)
    stage_specs = vj.boundary_param_specs(
        stage_boundary_input,
        stage_static.modes,
        max_mode=max_mode,
        min_coeff=0.0,
        include=_boundary_include_for_indata(stage_indata),
        fix=("rc00",),
    )

    stage_guess = initial_guess_from_boundary(stage_static, stage_boundary, stage_indata, vmec_project=True)
    stage_geom = eval_geom(stage_guess, stage_static)
    stage_signgs = int(signgs_from_sqrtg(np.asarray(stage_geom.sqrtg), axis_index=1))
    stage_flux = vj.flux_profiles_from_indata(stage_indata, stage_static.s, signgs=stage_signgs)
    stage_pressure = jnp.zeros_like(jnp.asarray(stage_static.s))
    qi_booz_constants = None
    qi_booz_grids = None
    qi_surface_indices = None
    if problem_cfg.objective_kind == "qi":
        from booz_xform_jax import prepare_booz_xform_constants
        from vmec_jax.modes import nyquist_mode_table_from_grid, vmec_mode_table

        main_modes = vmec_mode_table(int(stage_static.cfg.mpol), int(stage_static.cfg.ntor))
        nyq_modes = nyquist_mode_table_from_grid(
            mpol=int(stage_static.cfg.mpol),
            ntor=int(stage_static.cfg.ntor),
            ntheta=int(stage_static.cfg.ntheta),
            nzeta=int(stage_static.cfg.nzeta),
        )
        qi_booz_constants, qi_booz_grids = prepare_booz_xform_constants(
            nfp=int(stage_static.cfg.nfp),
            mboz=problem_cfg.qi_mboz,
            nboz=problem_cfg.qi_nboz,
            asym=bool(stage_static.cfg.lasym),
            xm=np.asarray(main_modes.m, dtype=int),
            xn=np.asarray(main_modes.n * int(stage_static.cfg.nfp), dtype=int),
            xm_nyq=np.asarray(nyq_modes.m, dtype=int),
            xn_nyq=np.asarray(nyq_modes.n * int(stage_static.cfg.nfp), dtype=int),
        )
        qi_surface_indices = _nearest_half_mesh_indices(
            problem_cfg.surfaces,
            n_half=max(int(np.asarray(stage_static.s).shape[0]) - 1, 1),
        )

    def stage_qs_eval(state):
        if problem_cfg.objective_kind == "qi":
            return quasi_isodynamic_residual_from_state(
                state=state,
                static=stage_static,
                indata=stage_indata,
                signgs=stage_signgs,
                flux_local=stage_flux,
                prof_local={"pressure": stage_pressure},
                pressure_local=stage_pressure,
                surfaces=problem_cfg.surfaces,
                mboz=problem_cfg.qi_mboz,
                nboz=problem_cfg.qi_nboz,
                nphi=problem_cfg.qi_nphi,
                nalpha=problem_cfg.qi_nalpha,
                n_bounce=problem_cfg.qi_n_bounce,
                softness=problem_cfg.qi_softness,
                width_weight=problem_cfg.qi_width_weight,
                branch_width_weight=problem_cfg.qi_branch_width_weight,
                branch_width_softness=problem_cfg.qi_branch_width_softness,
                profile_weight=problem_cfg.qi_profile_weight,
                shuffle_profile_weight=problem_cfg.qi_shuffle_profile_weight,
                shuffle_profile_softness=problem_cfg.qi_shuffle_profile_softness,
                aligned_profile_weight=problem_cfg.qi_aligned_profile_weight,
                aligned_profile_softness=problem_cfg.qi_aligned_profile_softness,
                aligned_profile_trap_level=problem_cfg.qi_aligned_profile_trap_level,
                aligned_profile_trap_softness=problem_cfg.qi_aligned_profile_trap_softness,
                phimin=problem_cfg.qi_phimin,
                jit_booz=bool(problem_cfg.qi_jit_booz),
                booz_constants=qi_booz_constants,
                booz_grids=qi_booz_grids,
                surface_indices=qi_surface_indices,
            )
        return quasisymmetry_ratio_residual_from_state(
            state=state,
            static=stage_static,
            indata=stage_indata,
            signgs=stage_signgs,
            flux_local=stage_flux,
            prof_local={"pressure": stage_pressure},
            pressure_local=stage_pressure,
            surfaces=problem_cfg.surfaces,
            helicity_m=problem_cfg.helicity_m,
            helicity_n=problem_cfg.helicity_n,
        )

    def mean_iota_raw(state):
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=state,
            static=stage_static,
            indata=stage_indata,
            signgs=stage_signgs,
        )
        iotas = jnp.asarray(iotas, dtype=jnp.float64)
        return jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else jnp.mean(iotas[1:])

    def iota_fn(state):
        return float(mean_iota_raw(state))

    track_iota = problem_cfg.target_iota is not None or problem_cfg.iota_abs_min is not None
    qi_mirror_ctx = SimpleNamespace(static=stage_static, indata=stage_indata, signgs=stage_signgs)
    qi_mirror_objective = vj.VMECMirrorRatio(
        threshold=problem_cfg.qi_max_mirror_ratio,
        surfaces=problem_cfg.surfaces,
        surface_index=problem_cfg.qi_mirror_surface_index,
        ntheta=problem_cfg.qi_mirror_ntheta,
        nphi=problem_cfg.qi_mirror_nphi,
    )

    def qi_field_quality_blocks(state, qi):
        blocks = [
            (
                jnp.asarray(qi["residuals1d"], dtype=jnp.float64) * problem_cfg.qs_weight,
                problem_cfg.qs_weight**2 * qi["total"],
            )
        ]
        if float(problem_cfg.qi_ceiling_weight) != 0.0:
            qi_total = jnp.asarray(qi["total"], dtype=jnp.float64)
            excess = qi_total - float(problem_cfg.qi_ceiling_max)
            softness = float(problem_cfg.qi_ceiling_smooth_penalty)
            if softness > 0.0:
                excess = softness * jnp.logaddexp(excess / softness, 0.0)
            else:
                excess = jnp.maximum(excess, 0.0)
            residual = jnp.ravel(excess) * float(problem_cfg.qi_ceiling_weight)
            blocks.append((residual, jnp.sum(residual * residual)))
        if float(problem_cfg.qi_mirror_weight) != 0.0:
            mirror = qi_mirror_objective._evaluate_state(qi_mirror_ctx, state)
            blocks.append(
                (
                    jnp.asarray(mirror["residuals1d"], dtype=jnp.float64) * problem_cfg.qi_mirror_weight,
                    problem_cfg.qi_mirror_weight**2 * mirror["total"],
                )
            )
        if float(problem_cfg.qi_elongation_weight) != 0.0:
            elongation = max_elongation_penalty_from_state(
                state=state,
                static=stage_static,
                threshold=problem_cfg.qi_max_elongation,
                ntheta=problem_cfg.qi_elongation_ntheta,
                nphi=problem_cfg.qi_elongation_nphi,
            )
            blocks.append(
                (
                    jnp.asarray(elongation["residuals1d"], dtype=jnp.float64) * problem_cfg.qi_elongation_weight,
                    problem_cfg.qi_elongation_weight**2 * elongation["total"],
                )
            )
        if float(problem_cfg.qi_lgradb_weight) != 0.0:
            lgradb = lgradb_penalty_from_state(
                state=state,
                static=stage_static,
                indata=stage_indata,
                signgs=stage_signgs,
                flux_local=stage_flux,
                threshold=problem_cfg.qi_lgradb_threshold,
                s_index=problem_cfg.qi_lgradb_surface_index,
                ntheta=problem_cfg.qi_lgradb_ntheta,
                nphi=problem_cfg.qi_lgradb_nphi,
                smooth_penalty=problem_cfg.qi_lgradb_smooth_penalty,
            )
            blocks.append(
                (
                    jnp.asarray(lgradb["residuals1d"], dtype=jnp.float64) * problem_cfg.qi_lgradb_weight,
                    problem_cfg.qi_lgradb_weight**2 * lgradb["total"],
                )
            )
        return tuple(blocks)

    def lgradb_quality_block(state):
        if float(problem_cfg.lgradb_weight) == 0.0:
            return None
        lgradb = lgradb_penalty_from_state(
            state=state,
            static=stage_static,
            indata=stage_indata,
            signgs=stage_signgs,
            flux_local=stage_flux,
            threshold=problem_cfg.lgradb_threshold,
            s_index=problem_cfg.lgradb_surface_index,
            ntheta=problem_cfg.lgradb_ntheta,
            nphi=problem_cfg.lgradb_nphi,
            smooth_penalty=problem_cfg.lgradb_smooth_penalty,
        )
        return (
            jnp.asarray(lgradb["residuals1d"], dtype=jnp.float64) * problem_cfg.lgradb_weight,
            problem_cfg.lgradb_weight**2 * lgradb["total"],
        )

    def stage_residuals_from_state(state):
        parts = []
        aspect = equilibrium_aspect_ratio_from_state(state=state, static=stage_static)
        parts.append(
            jnp.asarray(
                [problem_cfg.aspect_weight * (aspect - problem_cfg.target_aspect)],
                dtype=jnp.float64,
            )
        )
        if track_iota:
            iota = mean_iota_raw(state)
            if problem_cfg.target_iota is not None:
                parts.append(
                    jnp.asarray(
                        [problem_cfg.iota_weight * (iota - float(problem_cfg.target_iota))],
                        dtype=jnp.float64,
                    )
                )
            if problem_cfg.iota_abs_min is not None:
                iota_floor_weight = (
                    problem_cfg.iota_weight
                    if problem_cfg.iota_floor_weight is None
                    else float(problem_cfg.iota_floor_weight)
                )
                iota_residual = vj.smooth_min_abs_iota_residual(
                    iota,
                    problem_cfg.iota_abs_min,
                    softness=problem_cfg.iota_floor_softness,
                )
                parts.append(jnp.asarray([iota_floor_weight * iota_residual], dtype=jnp.float64))
        qs = stage_qs_eval(state)
        if problem_cfg.objective_kind == "qi":
            parts.extend(residuals for residuals, _total in qi_field_quality_blocks(state, qs))
        else:
            parts.append(jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * problem_cfg.qs_weight)
            lgradb_block = lgradb_quality_block(state)
            if lgradb_block is not None:
                parts.append(lgradb_block[0])
        return jnp.concatenate(parts)

    n_iota_terms = int(problem_cfg.target_iota is not None) + int(problem_cfg.iota_abs_min is not None)
    stage_residuals_from_state._n_non_qs = 1 + n_iota_terms
    stage_residuals_from_state._objective_family = str(problem_cfg.objective_kind)
    if problem_cfg.objective_kind == "qs":
        stage_residuals_from_state._helicity_m = int(problem_cfg.helicity_m)
        stage_residuals_from_state._helicity_n = int(problem_cfg.helicity_n)

    def stage_field_total_from_state(state):
        qs = stage_qs_eval(state)
        if problem_cfg.objective_kind == "qi":
            return float(sum(float(total) for _residuals, total in qi_field_quality_blocks(state, qs)))
        return float(problem_cfg.qs_weight) ** 2 * float(qs["total"])

    stage_residuals_from_state._qs_total_from_state = stage_field_total_from_state

    stage_opt = vj.FixedBoundaryExactOptimizer(
        stage_static,
        stage_indata,
        stage_boundary,
        stage_specs,
        stage_residuals_from_state,
        boundary_input=stage_boundary_input,
        inner_max_iter=problem_cfg.inner_max_iter,
        inner_ftol=problem_cfg.inner_ftol,
        trial_max_iter=problem_cfg.trial_max_iter,
        trial_ftol=problem_cfg.trial_ftol,
        solver_device=solver_device,
    )
    return stage_specs, stage_opt, iota_fn, stage_boundary_input


StageRecord = tuple[str, int, dict]
StageCheckpointCallback = Callable[[StageRecord, object, object, object, object], None]


def _merge_stage_histories(stage_results: list[StageRecord], *, problem_cfg: ProblemConfig) -> dict:
    combined_entries = []
    stage_boundaries = []
    stage_modes = []
    stage_labels = []
    stage_profiles = []
    profile_totals: dict[str, dict[str, float | int]] = {}
    callback_events = []
    callback_summary: dict[str, dict[str, float | int]] = {}
    callback_trace_enabled = False
    wall_offset = 0.0
    nfev_total = 0
    njev_total = 0
    max_nfev_total = 0
    for idx, (stage_label, _mode, stage_result) in enumerate(stage_results):
        stage_hist = stage_result["_history_dump"]
        stage_profile = stage_hist.get("profile")
        if isinstance(stage_profile, dict):
            stage_profiles.append(
                {
                    "stage": str(stage_label),
                    "mode": int(_mode),
                    "profile": dict(stage_profile),
                }
            )
            for key, value in stage_profile.items():
                entry = profile_totals.setdefault(str(key), {"count": 0, "wall_time_s": 0.0})
                if isinstance(value, dict):
                    entry["count"] = int(entry["count"]) + int(value.get("count", 0))
                    entry["wall_time_s"] = float(entry["wall_time_s"]) + float(value.get("wall_time_s", 0.0))
                elif isinstance(value, (int, float)):
                    entry["count"] = int(entry["count"]) + 1
                    entry["wall_time_s"] = float(entry["wall_time_s"]) + float(value)
        stage_trace = stage_hist.get("callback_trace")
        if isinstance(stage_trace, dict):
            callback_trace_enabled = callback_trace_enabled or bool(stage_trace.get("enabled", False))
            for event in stage_trace.get("events", []):
                if not isinstance(event, dict):
                    continue
                event_copy = dict(event)
                event_copy["index"] = len(callback_events)
                event_copy["stage"] = str(stage_label)
                event_copy["mode"] = int(_mode)
                callback_events.append(event_copy)
            for key, value in stage_trace.get("summary", {}).items():
                if not isinstance(value, dict):
                    continue
                entry = callback_summary.setdefault(str(key), {"count": 0, "wall_time_s": 0.0})
                entry["count"] = int(entry["count"]) + int(value.get("count", 0))
                entry["wall_time_s"] = float(entry["wall_time_s"]) + float(value.get("wall_time_s", 0.0))
        start_index = len(combined_entries)
        skip_first = idx != 0
        entries = stage_hist["history"] if idx == 0 else stage_hist["history"][1:]
        for entry in entries:
            entry_copy = dict(entry)
            entry_copy["wall_time_s"] = float(entry_copy["wall_time_s"]) + wall_offset
            entry_copy["stage"] = entry_copy.get("stage", stage_label)
            combined_entries.append(entry_copy)
        wall_offset = combined_entries[-1]["wall_time_s"]
        nfev_total += int(stage_hist["nfev"])
        njev_total += int(stage_hist["njev"])
        max_nfev_total += int(stage_hist.get("max_nfev", stage_hist["nfev"]))
        source_boundaries = stage_hist.get("stage_boundaries")
        if source_boundaries:
            for boundary in source_boundaries:
                adjusted = start_index + int(boundary) - (1 if skip_first else 0)
                if start_index <= adjusted < len(combined_entries):
                    stage_boundaries.append(adjusted)
        else:
            stage_boundaries.append(len(combined_entries) - 1)
        if stage_hist.get("stage_modes") and stage_hist.get("stage_labels"):
            stage_modes.extend(int(mode) for mode in stage_hist["stage_modes"])
            stage_labels.extend(str(label) for label in stage_hist["stage_labels"])
        else:
            stage_modes.append(int(_mode))
            stage_labels.append(str(stage_label))

    final_hist = stage_results[-1][2]["_history_dump"]
    first_hist = stage_results[0][2]["_history_dump"]
    merged = {
        "label": "Optimisation",
        "max_nfev": int(max_nfev_total),
        "ftol": problem_cfg.ftol,
        "gtol": problem_cfg.gtol,
        "xtol": problem_cfg.xtol,
        "total_wall_time_s": float(wall_offset),
        "nfev": int(nfev_total),
        "njev": int(njev_total),
        "success": bool(final_hist["success"]),
        "message": str(final_hist["message"]),
        "objective_initial": float(first_hist["objective_initial"]),
        "objective_final": float(final_hist["objective_final"]),
        "qs_initial": float(first_hist["qs_initial"]),
        "qs_final": float(final_hist["qs_final"]),
        "aspect_initial": float(first_hist["aspect_initial"]),
        "aspect_final": float(final_hist["aspect_final"]),
        "history": combined_entries,
        "target_aspect": problem_cfg.target_aspect,
        "stage_boundaries": stage_boundaries,
        "stage_modes": stage_modes,
        "stage_labels": stage_labels,
    }
    if problem_cfg.target_iota is not None:
        merged["target_iota"] = float(problem_cfg.target_iota)
    if problem_cfg.iota_abs_min is not None:
        merged["iota_abs_min"] = float(problem_cfg.iota_abs_min)
    if problem_cfg.target_iota is not None or problem_cfg.iota_abs_min is not None:
        if "iota" in combined_entries[0] and "iota" in combined_entries[-1]:
            merged["iota_initial"] = float(combined_entries[0]["iota"])
            merged["iota_final"] = float(combined_entries[-1]["iota"])
    if profile_totals:
        merged["profile"] = {
            key: {
                "count": int(value["count"]),
                "wall_time_s": float(value["wall_time_s"]),
                "mean_wall_time_s": float(value["wall_time_s"]) / int(value["count"])
                if int(value["count"])
                else 0.0,
            }
            for key, value in sorted(profile_totals.items())
        }
        merged["stage_profiles"] = stage_profiles
    if callback_trace_enabled or callback_events:
        merged["callback_trace"] = {
            "enabled": bool(callback_trace_enabled),
            "events": callback_events,
            "summary": dict(sorted(callback_summary.items())),
        }
    return merged


def _sync_result_profile_from_optimizer(result: dict, opt) -> None:
    """Refresh a result history profile after artifact writes update optimizer timings."""

    history = result.get("_history_dump") if isinstance(result, dict) else None
    profile_dump = getattr(opt, "_profile_dump", None)
    if not isinstance(history, dict) or not callable(profile_dump):
        return
    if "stage_profiles" in history:
        return
    try:
        history["profile"] = profile_dump()
    except Exception:
        return


def _input_nfp_or_none(cfg) -> int | None:
    try:
        return int(cfg.nfp)
    except Exception:
        return None


def _checkpoint_asymmetry_stats(
    *,
    specs,
    params_final,
    opt,
    stellarator_asymmetric: bool,
) -> dict[str, float | int | None]:
    if specs is None or params_final is None:
        return {}
    params_initial = np.zeros(len(specs), dtype=float)
    if bool(stellarator_asymmetric):
        boundary_input = getattr(opt, "_boundary_input", None)
        if boundary_input is not None:
            params_initial = _seed_zero_asymmetric_params(
                boundary_input=boundary_input,
                specs=specs,
                params=params_initial,
                seed=ASYMMETRIC_SEED,
            )
    return _asymmetric_param_stats(specs, params_initial, params_final)


def _iota_final_from_history(problem_cfg: ProblemConfig, history: dict) -> float | None:
    if problem_cfg.target_iota is None and problem_cfg.iota_abs_min is None:
        return None
    if history.get("iota_final") is not None:
        return _float_or_none(history.get("iota_final"))
    entries = history.get("history")
    if isinstance(entries, list) and entries and isinstance(entries[-1], dict):
        return _float_or_none(entries[-1].get("iota"))
    return None


def _case_result_from_history(
    *,
    history: dict,
    problem_cfg: ProblemConfig,
    cfg,
    backend: str,
    problem: str,
    max_mode: int,
    use_ess: bool,
    output_dir: Path,
    policy: str,
    solver_device: str | None,
    jax_platforms: str | None,
    jax_backend: str | None,
    jax_device_kind: str | None,
    stellarator_asymmetric: bool,
    specs,
    params_final,
    opt,
    success: bool,
    crashed: bool,
    message: str,
    extra_fields: dict | None = None,
) -> CaseResult:
    profile_summary = _profile_summary_fields(history)
    asym_stats = _checkpoint_asymmetry_stats(
        specs=specs,
        params_final=params_final,
        opt=opt,
        stellarator_asymmetric=stellarator_asymmetric,
    )
    fields = {
        "backend": str(backend),
        "problem": problem,
        "max_mode": int(max_mode),
        "use_ess": bool(use_ess),
        "success": bool(success),
        "crashed": bool(crashed),
        "message": str(message),
        "policy": str(policy),
        "objective_final": _float_or_none(history.get("objective_final")),
        "qs_final": _float_or_none(history.get("qs_final")),
        "aspect_final": _float_or_none(history.get("aspect_final")),
        "iota_final": _iota_final_from_history(problem_cfg, history),
        "nfev": _int_or_none(history.get("nfev")),
        "njev": _int_or_none(history.get("njev")),
        "total_wall_time_s": _float_or_none(history.get("total_wall_time_s")),
        "profile_wall_time_s": profile_summary["profile_wall_time_s"],
        "profile_top_name": profile_summary["profile_top_name"],
        "profile_top_wall_time_s": profile_summary["profile_top_wall_time_s"],
        "profile_solve_forward_trial_total_wall_time_s": profile_summary[
            "profile_solve_forward_trial_total_wall_time_s"
        ],
        "profile_solve_forward_exact_total_wall_time_s": profile_summary[
            "profile_solve_forward_exact_total_wall_time_s"
        ],
        "profile_exact_tape_build_wall_time_s": profile_summary["profile_exact_tape_build_wall_time_s"],
        "profile_exact_tape_build_solve_call_wall_time_s": profile_summary[
            "profile_exact_tape_build_solve_call_wall_time_s"
        ],
        "profile_exact_tape_build_unattributed_wall_time_s": profile_summary[
            "profile_exact_tape_build_unattributed_wall_time_s"
        ],
        "profile_exact_tape_solver_compute_forces_first_wall_time_s": profile_summary[
            "profile_exact_tape_solver_compute_forces_first_wall_time_s"
        ],
        "profile_exact_tape_solver_compute_forces_rest_wall_time_s": profile_summary[
            "profile_exact_tape_solver_compute_forces_rest_wall_time_s"
        ],
        "profile_trial_solver_scan_total_wall_time_s": profile_summary[
            "profile_trial_solver_scan_total_wall_time_s"
        ],
        "profile_trial_solver_scan_runner_cache_lookup_wall_time_s": profile_summary[
            "profile_trial_solver_scan_runner_cache_lookup_wall_time_s"
        ],
        "profile_trial_solver_scan_runner_cache_build_wall_time_s": profile_summary[
            "profile_trial_solver_scan_runner_cache_build_wall_time_s"
        ],
        "profile_trial_solver_scan_runner_cache_hit_count": profile_summary[
            "profile_trial_solver_scan_runner_cache_hit_count"
        ],
        "profile_trial_solver_scan_runner_cache_miss_count": profile_summary[
            "profile_trial_solver_scan_runner_cache_miss_count"
        ],
        "profile_trial_solver_scan_runner_cache_bypass_count": profile_summary[
            "profile_trial_solver_scan_runner_cache_bypass_count"
        ],
        "profile_trial_solver_scan_runner_cache_hit_device_run_wall_time_s": profile_summary[
            "profile_trial_solver_scan_runner_cache_hit_device_run_wall_time_s"
        ],
        "profile_trial_solver_scan_runner_cache_miss_device_run_wall_time_s": profile_summary[
            "profile_trial_solver_scan_runner_cache_miss_device_run_wall_time_s"
        ],
        "profile_trial_solver_scan_runner_cache_bypass_device_run_wall_time_s": profile_summary[
            "profile_trial_solver_scan_runner_cache_bypass_device_run_wall_time_s"
        ],
        "profile_trial_solver_scan_device_dispatch_wall_time_s": profile_summary[
            "profile_trial_solver_scan_device_dispatch_wall_time_s"
        ],
        "profile_trial_solver_scan_device_ready_wall_time_s": profile_summary[
            "profile_trial_solver_scan_device_ready_wall_time_s"
        ],
        "profile_trial_solver_scan_host_materialize_wall_time_s": profile_summary[
            "profile_trial_solver_scan_host_materialize_wall_time_s"
        ],
        "profile_jacobian_total_wall_time_s": profile_summary["profile_jacobian_total_wall_time_s"],
        "profile_write_wout_wall_time_s": profile_summary["profile_write_wout_wall_time_s"],
        "output_dir": str(output_dir),
        "jax_backend": jax_backend,
        "jax_device_kind": jax_device_kind,
        "solver_device": solver_device,
        "jax_platforms": jax_platforms,
        "stellarator_asymmetric": bool(stellarator_asymmetric),
        "asymmetry_seed": float(ASYMMETRIC_SEED if stellarator_asymmetric else 0.0),
        "input_file": str(problem_cfg.input_file),
        "input_nfp": _input_nfp_or_none(cfg),
        "project_input_boundary_to_max_mode": bool(problem_cfg.project_input_boundary_to_max_mode),
        "target_aspect": float(problem_cfg.target_aspect),
        "target_iota": (None if problem_cfg.target_iota is None else float(problem_cfg.target_iota)),
        "iota_abs_min": (None if problem_cfg.iota_abs_min is None else float(problem_cfg.iota_abs_min)),
        "iota_weight": float(problem_cfg.iota_weight),
        "lgradb_weight": float(problem_cfg.lgradb_weight),
        "qi_lgradb_weight": float(problem_cfg.qi_lgradb_weight),
        "qi_qp_preseed": (bool(problem_cfg.qi_preseed_qp) if problem_cfg.objective_kind == "qi" else None),
        "qi_qi_preseed": (bool(problem_cfg.qi_preseed_qi) if problem_cfg.objective_kind == "qi" else None),
        "qi_jit_booz": (bool(problem_cfg.qi_jit_booz) if problem_cfg.objective_kind == "qi" else None),
        **asym_stats,
    }
    if extra_fields:
        fields.update(extra_fields)
    return CaseResult(**fields)


def _stage_checkpoint_payload(
    *,
    stage_results: list[StageRecord],
    history: dict,
    partial: bool,
) -> dict[str, object]:
    latest_label, latest_mode, _latest_result = stage_results[-1]
    iota_final = _float_or_none(history.get("iota_final"))
    entries = history.get("history")
    if iota_final is None and isinstance(entries, list) and entries and isinstance(entries[-1], dict):
        iota_final = _float_or_none(entries[-1].get("iota"))
    stages = []
    for label, mode, result in stage_results:
        stage_history = result.get("_history_dump", {}) if isinstance(result, dict) else {}
        stages.append(
            {
                "label": str(label),
                "mode": int(mode),
                "success": bool(stage_history.get("success", False)),
                "message": str(stage_history.get("message", "")),
                "nfev": _int_or_none(stage_history.get("nfev")),
                "njev": _int_or_none(stage_history.get("njev")),
                "objective_final": _float_or_none(stage_history.get("objective_final")),
                "qs_final": _float_or_none(stage_history.get("qs_final")),
                "aspect_final": _float_or_none(stage_history.get("aspect_final")),
                "total_wall_time_s": _float_or_none(stage_history.get("total_wall_time_s")),
            }
        )
    return {
        "schema_version": 1,
        "partial": bool(partial),
        "stage_count": len(stage_results),
        "latest_stage_label": str(latest_label),
        "latest_stage_mode": int(latest_mode),
        "last_completed_stage": str(latest_label),
        "last_completed_stage_mode": int(latest_mode),
        "completed_stage_count": len(stage_results),
        "stage_labels": [str(label) for label in history.get("stage_labels", [])],
        "stage_modes": [int(mode) for mode in history.get("stage_modes", [])],
        "nfev": _int_or_none(history.get("nfev")),
        "njev": _int_or_none(history.get("njev")),
        "objective_final": _float_or_none(history.get("objective_final")),
        "qs_final": _float_or_none(history.get("qs_final")),
        "aspect_final": _float_or_none(history.get("aspect_final")),
        "iota_final": iota_final,
        "total_wall_time_s": _float_or_none(history.get("total_wall_time_s")),
        "history_path": "history.json",
        "case_result_path": "case_result.json",
        "input_path": "input.final",
        "wout_path": "wout_final.nc",
        "artifacts": {
            "input_final": "input.final",
            "wout_final": "wout_final.nc",
        },
        "stages": stages,
    }


def _write_case_checkpoint(
    *,
    output_dir: Path,
    result_path: Path,
    stage_results: list[StageRecord],
    problem_cfg: ProblemConfig,
    cfg,
    backend: str,
    problem: str,
    max_mode: int,
    use_ess: bool,
    policy: str,
    solver_device: str | None,
    jax_platforms: str | None,
    jax_backend: str | None,
    jax_device_kind: str | None,
    stellarator_asymmetric: bool,
    latest_specs,
    latest_opt,
    latest_params_final,
    write_artifacts: bool,
    success: bool,
    crashed: bool,
    message: str,
    extra_fields: dict | None = None,
    history_override: dict | None = None,
) -> CaseResult:
    """Write the latest bounded stage checkpoint and partial case result."""

    if not stage_results:
        raise ValueError("stage_results must contain at least one stage")
    output_dir = Path(output_dir)
    result_path = Path(result_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_stage_result = stage_results[-1][2]

    if write_artifacts and latest_opt is not None and latest_params_final is not None:
        latest_opt.save_input(output_dir / "input.final", latest_params_final)
        latest_opt.save_wout(
            output_dir / "wout_final.nc",
            latest_params_final,
            state=latest_stage_result.get("_state_final"),
        )
        _sync_result_profile_from_optimizer(latest_stage_result, latest_opt)

    history = (
        history_override
        if history_override is not None
        else _merge_stage_histories(stage_results, problem_cfg=problem_cfg)
    )
    _atomic_write_json(output_dir / "history.json", history)
    case_result = _case_result_from_history(
        history=history,
        problem_cfg=problem_cfg,
        cfg=cfg,
        backend=backend,
        problem=problem,
        max_mode=max_mode,
        use_ess=use_ess,
        output_dir=output_dir,
        policy=policy,
        solver_device=solver_device,
        jax_platforms=jax_platforms,
        jax_backend=jax_backend,
        jax_device_kind=jax_device_kind,
        stellarator_asymmetric=stellarator_asymmetric,
        specs=latest_specs,
        params_final=latest_params_final,
        opt=latest_opt,
        success=success,
        crashed=crashed,
        message=message,
        extra_fields=extra_fields,
    )
    _atomic_write_json(result_path, asdict(case_result))
    _atomic_write_json(
        output_dir / "stage_checkpoint.json",
        _stage_checkpoint_payload(stage_results=stage_results, history=history, partial=bool(crashed)),
    )
    print(f"  Wrote checkpoint {result_path}", flush=True)
    return case_result


def _boundary_params_from_indata(
    source_indata: InData,
    *,
    target_static,
    target_boundary_input,
    target_specs,
) -> np.ndarray:
    """Express ``source_indata`` boundary coefficients as params on target specs."""

    source_boundary = vj.boundary_input_from_indata(source_indata, target_static.modes)
    source_arrays = {
        "rc": np.asarray(source_boundary.R_cos, dtype=float),
        "rs": np.asarray(source_boundary.R_sin, dtype=float),
        "zc": np.asarray(source_boundary.Z_cos, dtype=float),
        "zs": np.asarray(source_boundary.Z_sin, dtype=float),
    }
    base_arrays = {
        "rc": np.asarray(target_boundary_input.R_cos, dtype=float),
        "rs": np.asarray(target_boundary_input.R_sin, dtype=float),
        "zc": np.asarray(target_boundary_input.Z_cos, dtype=float),
        "zs": np.asarray(target_boundary_input.Z_sin, dtype=float),
    }
    return np.asarray(
        [
            float(source_arrays[spec.kind][int(spec.index)] - base_arrays[spec.kind][int(spec.index)])
            for spec in target_specs
        ],
        dtype=float,
    )


def _resume_from_previous_continuation_case(
    *,
    problem_cfg: ProblemConfig,
    cfg,
    indata,
    output_dir: Path,
    max_mode: int,
    solver_device: str | None,
) -> tuple[list[StageRecord], object, np.ndarray] | None:
    """Load mode-(N-1) continuation output so mode-N can skip repeated stages."""

    if int(max_mode) <= 1:
        return None
    previous_dir = output_dir.parent.parent / f"mode{int(max_mode) - 1}" / output_dir.name
    case_path = previous_dir / "case_result.json"
    history_path = previous_dir / "history.json"
    input_path = previous_dir / "input.final"
    if not (case_path.exists() and history_path.exists() and input_path.exists()):
        return None
    try:
        previous_case = json.loads(case_path.read_text())
        if bool(previous_case.get("crashed", False)):
            return None
        previous_history = json.loads(history_path.read_text())
        _previous_cfg, previous_indata = vj.load_config(str(input_path))
    except Exception:
        return None

    previous_specs, _previous_opt, _previous_iota_fn, previous_boundary_input = _build_stage(
        problem_cfg,
        cfg,
        indata,
        int(max_mode) - 1,
        solver_device=solver_device,
    )
    previous_static = _previous_opt._static
    params_stage = _boundary_params_from_indata(
        previous_indata,
        target_static=previous_static,
        target_boundary_input=previous_boundary_input,
        target_specs=previous_specs,
    )
    print(
        f"  Reusing continuation seed from {previous_dir.relative_to(OUTPUT_ROOT)}",
        flush=True,
    )
    return [(f"{problem_cfg.name.upper()} modes 1-{int(max_mode) - 1}", int(max_mode) - 1, {"_history_dump": previous_history})], previous_specs, params_stage


def _plot_case_outputs(output_dir: Path) -> None:
    try:
        vj.plot_3d_boundary_comparison(
            output_dir / "wout_initial.nc",
            output_dir / "wout_final.nc",
            outdir=output_dir,
        )
        vj.plot_bmag_contours(
            output_dir / "wout_initial.nc",
            output_dir / "wout_final.nc",
            outdir=output_dir,
        )
        vj.plot_objective_history(output_dir / "history.json", outdir=output_dir)
    except Exception as exc:
        # Plotting is a post-processing convenience.  Do not mark an otherwise
        # valid optimization as failed because a remote/headless Matplotlib
        # environment has a broken 3D toolkit.
        (output_dir / "plotting_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        print(f"  Skipped case plots: {type(exc).__name__}: {exc}", flush=True)


def _save_case_outputs(
    output_dir: Path,
    opt,
    params_initial,
    params_final,
    result: dict,
    *,
    make_plots: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    opt.save_input(output_dir / "input.initial", params_initial)
    opt.save_wout(output_dir / "wout_initial.nc", params_initial, state=result.get("_state_initial"))
    opt.save_input(output_dir / "input.final", params_final)
    opt.save_wout(output_dir / "wout_final.nc", params_final, state=result.get("_state_final"))
    _sync_result_profile_from_optimizer(result, opt)
    opt.save_history(output_dir / "history.json", result)
    if make_plots:
        _plot_case_outputs(output_dir)


def _annotate_result_from_stage_checkpoint(result: CaseResult) -> CaseResult:
    """Fill failed/timeout result metrics from ``stage_checkpoint.json`` if present."""

    if result.output_dir is None:
        return result
    checkpoint_path = Path(result.output_dir) / "stage_checkpoint.json"
    if not checkpoint_path.exists():
        return result
    try:
        checkpoint = json.loads(checkpoint_path.read_text())
    except Exception:
        return result

    mapping = {
        "objective_final": "objective_final",
        "qs_final": "qs_final",
        "aspect_final": "aspect_final",
        "iota_final": "iota_final",
        "nfev": "nfev",
        "njev": "njev",
    }
    for checkpoint_key, result_key in mapping.items():
        if getattr(result, result_key) is None and checkpoint_key in checkpoint:
            setattr(result, result_key, checkpoint[checkpoint_key])
    if result.total_wall_time_s is None and "total_wall_time_s" in checkpoint:
        result.total_wall_time_s = checkpoint["total_wall_time_s"]
    stage = checkpoint.get("last_completed_stage") or checkpoint.get("latest_stage_label")
    if stage and "last checkpoint" not in result.message:
        result.message = f"{result.message}; last checkpoint: {stage}"
    return result


def _run_problem_stages(
    *,
    problem_cfg: ProblemConfig,
    problem: str,
    max_mode: int,
    use_ess: bool,
    use_mode_continuation: bool,
    solver_device: str | None,
    cfg,
    indata,
    stage_label_prefix: str,
    params_stage,
    prev_specs,
    stellarator_asymmetric: bool,
    stage_completed_callback: StageCheckpointCallback | None = None,
) -> tuple[list[StageRecord], object, object, object, object, dict]:
    stage_modes = _stage_modes_for_problem(
        problem_cfg,
        max_mode=max_mode,
        use_mode_continuation=use_mode_continuation,
    )
    stage_results: list[StageRecord] = []
    final_opt = None
    final_params0 = None
    final_result = None

    for stage_mode in stage_modes:
        stage_specs, stage_opt, iota_fn, stage_boundary_input = _build_stage(
            problem_cfg,
            cfg,
            indata,
            stage_mode,
            solver_device=solver_device,
        )
        stage_x_scale = (
            vj.create_x_scale(
                stage_specs,
                alpha=_ess_alpha_for_case(problem_cfg, problem, max_mode, use_ess),
            )
            if use_ess
            else np.ones(len(stage_specs))
        )
        params0_stage = (
            np.zeros(len(stage_specs), dtype=float)
            if params_stage is None
            else vj.lift_boundary_params(prev_specs, params_stage, stage_specs)
        )
        if bool(stellarator_asymmetric):
            params0_stage = _seed_zero_asymmetric_params(
                boundary_input=stage_boundary_input,
                specs=stage_specs,
                params=params0_stage,
                seed=ASYMMETRIC_SEED,
            )
        stage_budget = problem_cfg.max_nfev if stage_mode == max_mode else problem_cfg.continuation_nfev
        track_iota = problem_cfg.target_iota is not None or problem_cfg.iota_abs_min is not None
        stage_result = stage_opt.run(
            params0_stage,
            method=problem_cfg.method,
            max_nfev=stage_budget,
            ftol=problem_cfg.ftol,
            gtol=problem_cfg.gtol,
            xtol=problem_cfg.xtol,
            x_scale=stage_x_scale,
            verbose=0,
            iota_fn=iota_fn if track_iota else None,
            target_iota=problem_cfg.target_iota,
            target_aspect=problem_cfg.target_aspect,
            scipy_tr_solver=problem_cfg.scipy_tr_solver,
            scipy_lsmr_maxiter=problem_cfg.scipy_lsmr_maxiter,
        )
        if problem_cfg.iota_abs_min is not None:
            stage_result["_history_dump"]["iota_abs_min"] = float(problem_cfg.iota_abs_min)
        stage_record = (f"{stage_label_prefix} mode {stage_mode}", stage_mode, stage_result)
        stage_results.append(stage_record)
        if stage_completed_callback is not None:
            stage_completed_callback(stage_record, stage_specs, stage_opt, params0_stage, stage_result["x"])
        prev_specs = stage_specs
        params_stage = stage_result["x"]
        final_opt = stage_opt
        final_params0 = params0_stage
        final_result = stage_result

    assert final_opt is not None
    assert final_params0 is not None
    assert final_result is not None
    assert prev_specs is not None
    assert params_stage is not None
    return stage_results, prev_specs, params_stage, final_opt, final_params0, final_result


def _run_case(
    problem: str,
    max_mode: int,
    use_ess: bool,
    output_dir: Path,
    result_path: Path | None = None,
    *,
    use_mode_continuation: bool,
    policy: str,
    backend: str,
    solver_device: str | None,
    jax_platforms: str | None,
    diagnostic_budgets: bool = DIAGNOSTIC_BUDGETS,
    stellarator_asymmetric: bool = STELLARATOR_ASYMMETRIC,
    qi_qp_preseed: bool | None = None,
    qi_jit_booz: bool | None = None,
    cli_budget: CaseBudget | None = None,
    ess_alpha_override: float | None = None,
) -> CaseResult:
    result_path = Path(output_dir) / "case_result.json" if result_path is None else Path(result_path)
    problem_cfg = _effective_problem_config(
        PROBLEM_CONFIGS[problem],
        backend=backend,
        policy=policy,
        problem=problem,
        max_mode=max_mode,
        use_ess=use_ess,
        diagnostic_budgets=diagnostic_budgets,
        cli_budget=cli_budget,
        ess_alpha_override=ess_alpha_override,
    )
    if problem_cfg.objective_kind == "qi" and qi_qp_preseed is not None:
        problem_cfg = replace(problem_cfg, qi_preseed_qp=bool(qi_qp_preseed))
    if problem_cfg.objective_kind == "qi" and qi_jit_booz is not None:
        problem_cfg = replace(problem_cfg, qi_jit_booz=bool(qi_jit_booz))
    if problem_cfg.objective_kind == "qi" and problem_cfg.qi_preseed_qi and not bool(use_ess):
        # The QI-only preseed is an ESS-stabilized helper stage.  Running that
        # hidden stage without ESS can create ill-conditioned LSMR trust-region
        # subproblems before the visible QI stage starts, so keep no-ESS rows as
        # a true no-ESS baseline.
        problem_cfg = replace(problem_cfg, qi_preseed_qi=False)
    cfg, indata = _load_problem(
        problem_cfg,
        max_mode=max_mode,
        stellarator_asymmetric=stellarator_asymmetric,
    )
    jax_backend, jax_device_kind = _jax_runtime_info()

    stage_results: list[StageRecord] = []
    params_stage = None
    prev_specs = None
    use_mode_continuation_for_main = use_mode_continuation

    # High-mode continuation cases can be very expensive if each max_mode=N row
    # reruns stages 1..N from scratch.  When the matching mode-(N-1) row already
    # exists, use its final input deck as the seed and append only the new stage.
    # QI keeps its explicit QP pre-seed path below because the mode-N QP seed is
    # part of the QI policy being benchmarked.
    if (
        use_mode_continuation
        and int(max_mode) > 1
        and problem_cfg.objective_kind != "qi"
        and _stage_modes_for_problem(
            problem_cfg,
            max_mode=max_mode,
            use_mode_continuation=use_mode_continuation,
        )
        == list(range(1, max_mode + 1))
    ):
        resume = _resume_from_previous_continuation_case(
            problem_cfg=problem_cfg,
            cfg=cfg,
            indata=indata,
            output_dir=output_dir,
            max_mode=max_mode,
            solver_device=solver_device,
        )
        if resume is not None:
            resume_stage_results, prev_specs, params_stage = resume
            stage_results.extend(resume_stage_results)
            use_mode_continuation_for_main = False

    checkpoint_stage_results: list[StageRecord] = list(stage_results)

    def _checkpoint_completed_stage(
        stage_record: StageRecord,
        stage_specs,
        stage_opt,
        _params_initial,
        params_final,
    ) -> None:
        checkpoint_stage_results.append(stage_record)
        stage_label = stage_record[0]
        try:
            _write_case_checkpoint(
                output_dir=output_dir,
                result_path=result_path,
                stage_results=checkpoint_stage_results,
                problem_cfg=problem_cfg,
                cfg=cfg,
                backend=backend,
                problem=problem,
                max_mode=max_mode,
                use_ess=use_ess,
                policy=policy,
                solver_device=solver_device,
                jax_platforms=jax_platforms,
                jax_backend=jax_backend,
                jax_device_kind=jax_device_kind,
                stellarator_asymmetric=stellarator_asymmetric,
                latest_specs=stage_specs,
                latest_opt=stage_opt,
                latest_params_final=params_final,
                write_artifacts=True,
                success=False,
                crashed=True,
                message=f"partial checkpoint after {stage_label}; case still running",
            )
        except Exception as exc:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "checkpoint_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            print(f"  Skipped stage checkpoint after {stage_label}: {type(exc).__name__}: {exc}", flush=True)

    original_stage = None
    qp_seed_stage = None
    if problem_cfg.objective_kind == "qi" and problem_cfg.qi_preseed_qp:
        qp_cfg = _effective_problem_config(
            PROBLEM_CONFIGS["qp"],
            backend=backend,
            policy=policy,
            problem="qp",
            max_mode=max_mode,
            use_ess=use_ess,
            diagnostic_budgets=diagnostic_budgets,
            cli_budget=cli_budget,
            ess_alpha_override=ess_alpha_override,
        )
        qp_stage_results, prev_specs, params_stage, qp_opt, qp_params0, qp_result = _run_problem_stages(
            problem_cfg=qp_cfg,
            problem="qp",
            max_mode=max_mode,
            use_ess=use_ess,
            use_mode_continuation=use_mode_continuation,
            solver_device=solver_device,
            cfg=cfg,
            indata=indata,
            stage_label_prefix="QP preseed",
            params_stage=params_stage,
            prev_specs=prev_specs,
            stellarator_asymmetric=stellarator_asymmetric,
            stage_completed_callback=_checkpoint_completed_stage,
        )
        stage_results.extend(qp_stage_results)
        original_stage = (qp_opt, qp_params0, qp_result)
        qp_seed_stage = (qp_opt, qp_result["x"], qp_result)

    if problem_cfg.objective_kind == "qi" and problem_cfg.qi_preseed_qi:
        qi_seed_cfg = replace(
            problem_cfg,
            max_nfev=int(problem_cfg.qi_preseed_qi_nfev),
            continuation_nfev=min(int(problem_cfg.continuation_nfev), int(problem_cfg.qi_preseed_qi_nfev)),
            aspect_weight=0.0,
            target_iota=None,
            iota_abs_min=None,
            iota_weight=0.0,
            iota_floor_weight=0.0,
            qi_mirror_weight=0.0,
            qi_elongation_weight=0.0,
            qi_lgradb_weight=0.0,
            qi_preseed_qp=False,
            qi_preseed_qi=False,
        )
        qi_seed_stage_results, prev_specs, params_stage, _qi_seed_opt, _qi_seed_params0, _qi_seed_result = (
            _run_problem_stages(
                problem_cfg=qi_seed_cfg,
                problem="qi",
                max_mode=max_mode,
                use_ess=use_ess,
                use_mode_continuation=use_mode_continuation,
                solver_device=solver_device,
                cfg=cfg,
                indata=indata,
                stage_label_prefix="QI preseed",
                params_stage=params_stage,
                prev_specs=prev_specs,
                stellarator_asymmetric=stellarator_asymmetric,
                stage_completed_callback=_checkpoint_completed_stage,
            )
        )
        stage_results.extend(qi_seed_stage_results)

    qi_or_qs_stage_results, prev_specs, params_stage, final_opt, final_params0, final_result = _run_problem_stages(
        problem_cfg=problem_cfg,
        problem=problem,
        max_mode=max_mode,
        use_ess=use_ess,
        use_mode_continuation=use_mode_continuation_for_main,
        solver_device=solver_device,
        cfg=cfg,
        indata=indata,
        stage_label_prefix=problem.upper(),
        params_stage=params_stage,
        prev_specs=prev_specs,
        stellarator_asymmetric=stellarator_asymmetric,
        stage_completed_callback=_checkpoint_completed_stage,
    )
    stage_results.extend(qi_or_qs_stage_results)

    case_output_result = final_result
    if len(stage_results) > 1:
        case_output_result = dict(final_result)
        case_output_result["_history_dump"] = _merge_stage_histories(stage_results, problem_cfg=problem_cfg)

    _save_case_outputs(output_dir, final_opt, final_params0, final_result["x"], case_output_result, make_plots=False)
    hist = case_output_result["_history_dump"]
    try:
        _write_case_checkpoint(
            output_dir=output_dir,
            result_path=result_path,
            stage_results=checkpoint_stage_results,
            problem_cfg=problem_cfg,
            cfg=cfg,
            backend=backend,
            problem=problem,
            max_mode=max_mode,
            use_ess=use_ess,
            policy=policy,
            solver_device=solver_device,
            jax_platforms=jax_platforms,
            jax_backend=jax_backend,
            jax_device_kind=jax_device_kind,
            stellarator_asymmetric=stellarator_asymmetric,
            latest_specs=prev_specs,
            latest_opt=final_opt,
            latest_params_final=final_result["x"],
            write_artifacts=False,
            success=bool(hist["success"]),
            crashed=True,
            message=f"result metadata checkpoint before diagnostics: {hist['message']}",
            history_override=hist,
        )
    except Exception as exc:
        (output_dir / "checkpoint_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        print(f"  Skipped result metadata checkpoint: {type(exc).__name__}: {exc}", flush=True)
    _plot_case_outputs(output_dir)
    bmag_stats = _bmag_lcfs_stats(output_dir / "wout_final.nc")
    lgradb_stats = _lgradb_diagnostics_from_state(problem_cfg, final_opt, final_result.get("_state_final"))
    qi_stats = _qi_diagnostics_from_state(problem_cfg, final_opt, final_result.get("_state_final"))
    if original_stage is not None:
        original_opt, original_params0, original_result = original_stage
        original_opt.save_input(output_dir / "input.original", original_params0)
        original_opt.save_wout(output_dir / "wout_original.nc", original_params0, state=original_result.get("_state_initial"))
    if qp_seed_stage is not None:
        qp_opt, qp_params, qp_result = qp_seed_stage
        qp_opt.save_input(output_dir / "input.qp_seed", qp_params)
        qp_opt.save_wout(output_dir / "wout_qp_seed.nc", qp_params, state=qp_result.get("_state_final"))
    try:
        _atomic_write_json(
            output_dir / "stage_checkpoint.json",
            _stage_checkpoint_payload(stage_results=checkpoint_stage_results, history=hist, partial=False),
        )
    except Exception as exc:
        (output_dir / "checkpoint_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")

    return _case_result_from_history(
        history=hist,
        problem_cfg=problem_cfg,
        cfg=cfg,
        backend=backend,
        problem=problem,
        max_mode=max_mode,
        use_ess=use_ess,
        output_dir=output_dir,
        policy=policy,
        solver_device=solver_device,
        jax_platforms=jax_platforms,
        jax_backend=jax_backend,
        jax_device_kind=jax_device_kind,
        stellarator_asymmetric=stellarator_asymmetric,
        specs=prev_specs,
        params_final=final_result["x"],
        opt=final_opt,
        success=bool(hist["success"]),
        crashed=False,
        message=str(hist["message"]),
        extra_fields={**bmag_stats, **lgradb_stats, **qi_stats},
    )


def _worker(
    problem: str,
    max_mode: int,
    use_ess: bool,
    output_dir: str,
    result_path: str,
    use_mode_continuation: bool,
    policy: str,
    backend: str,
    solver_device: str | None,
    jax_platforms: str | None,
    diagnostic_budgets: bool,
    stellarator_asymmetric: bool,
    qi_qp_preseed: bool | None,
    qi_jit_booz: bool | None,
    cli_budget: CaseBudget | None = None,
    ess_alpha_override: float | None = None,
):
    _start_worker_session()
    try:
        case_result = _run_case(
            problem,
            max_mode,
            use_ess,
            Path(output_dir),
            Path(result_path),
            use_mode_continuation=use_mode_continuation,
            policy=policy,
            backend=backend,
            solver_device=solver_device,
            jax_platforms=jax_platforms,
            diagnostic_budgets=diagnostic_budgets,
            stellarator_asymmetric=stellarator_asymmetric,
            qi_qp_preseed=qi_qp_preseed,
            qi_jit_booz=qi_jit_booz,
            cli_budget=cli_budget,
            ess_alpha_override=ess_alpha_override,
        )
        _atomic_write_json(Path(result_path), asdict(case_result))
        stale_traceback = Path(output_dir) / "traceback.txt"
        if stale_traceback.exists():
            stale_traceback.unlink()
    except Exception as exc:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        try:
            failed = CaseResult(**json.loads(Path(result_path).read_text()))
            previous_message = str(failed.message or "")
            failed.message = (
                f"{type(exc).__name__}: {exc}"
                if not previous_message
                else f"{type(exc).__name__}: {exc}; {previous_message}"
            )
            failed.crashed = True
        except Exception:
            failed = CaseResult(
                backend=str(backend),
                problem=problem,
                max_mode=max_mode,
                use_ess=bool(use_ess),
                success=False,
                crashed=True,
                message=f"{type(exc).__name__}: {exc}",
                policy=policy,
                output_dir=str(output_dir),
                solver_device=solver_device,
                jax_platforms=jax_platforms,
                stellarator_asymmetric=bool(stellarator_asymmetric),
                asymmetry_seed=float(ASYMMETRIC_SEED if stellarator_asymmetric else 0.0),
                input_file=str(PROBLEM_CONFIGS[problem].input_file) if problem in PROBLEM_CONFIGS else None,
                input_nfp=None,
                project_input_boundary_to_max_mode=(
                    bool(PROBLEM_CONFIGS[problem].project_input_boundary_to_max_mode)
                    if problem in PROBLEM_CONFIGS
                    else None
                ),
                qi_qp_preseed=(bool(qi_qp_preseed) if problem == "qi" and qi_qp_preseed is not None else None),
                qi_jit_booz=(bool(qi_jit_booz) if problem == "qi" and qi_jit_booz is not None else None),
            )
        failed = _annotate_result_from_stage_checkpoint(failed)
        _atomic_write_json(Path(result_path), asdict(failed))
        Path(output_dir, "traceback.txt").write_text(traceback.format_exc())
        raise


def _case_key(result: CaseResult):
    return (result.problem, int(result.max_mode), bool(result.use_ess))


def _history_for(result: CaseResult) -> dict | None:
    if result.output_dir is None:
        return None
    hist_path = Path(result.output_dir) / "history.json"
    if not hist_path.exists():
        return None
    return json.loads(hist_path.read_text())


def _history_stage_segments(history: list[dict]) -> list[list[dict]]:
    segments: list[list[dict]] = []
    current: list[dict] = []
    current_stage = object()
    for item in history:
        stage = item.get("stage", "")
        if current and stage != current_stage:
            segments.append(current)
            current = []
        current.append(item)
        current_stage = stage
    if current:
        segments.append(current)
    return segments


def _style_publication():
    from vmec_jax.plotting import prepare_matplotlib_3d

    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    return plt


def _plot_objective_panel(results: list[CaseResult], outpath_png: Path, outpath_pdf: Path) -> None:
    plt = _style_publication()

    fig, axes = plt.subplots(
        len(PROBLEMS),
        len(MODES),
        figsize=(5.5 * len(MODES), 4.3 * len(PROBLEMS)),
        sharey="row",
        squeeze=False,
    )
    colors = {False: "#1f77b4", True: "#d95f02"}
    labels = {False: "No ESS", True: "ESS"}

    for row, problem in enumerate(PROBLEMS):
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            ax.set_title(f"{problem.upper()}  max_mode={mode}")
            panel_label = _panel_label(row * len(MODES) + col)
            ax.text(
                0.01,
                0.99,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                fontweight="bold",
            )
            annotation_lines = []
            for use_ess in ESS_OPTIONS:
                rec = next((r for r in results if _case_key(r) == (problem, mode, use_ess)), None)
                if rec is None:
                    continue
                hist = _history_for(rec)
                if hist is None:
                    continue
                segments = []
                start_index = 0
                for segment in _history_stage_segments(hist["history"]):
                    stop_index = start_index + len(segment)
                    segments.append(
                        (
                            np.arange(start_index, stop_index, dtype=float),
                            np.minimum.accumulate(
                                np.asarray([max(float(entry["objective"]), 1e-16) for entry in segment], dtype=float)
                            ),
                        )
                    )
                    start_index = stop_index
                linestyle = "-" if rec.success and not rec.crashed else "--"
                linewidth = 2.6 if use_ess else 2.1
                first_segment = True
                last_x = None
                last_y = None
                for x, y in segments:
                    ax.semilogy(
                        x,
                        y,
                        color=colors[use_ess],
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=labels[use_ess] if first_segment and (row == 0 and col == 0) else None,
                    )
                    first_segment = False
                    last_x = x[-1]
                    last_y = y[-1]
                if last_x is not None and last_y is not None:
                    ax.scatter(last_x, last_y, color=colors[use_ess], s=30, zorder=4)
                for boundary in hist.get("stage_boundaries", [])[:-1]:
                    ax.axvline(float(boundary), color="0.75", linestyle=":", linewidth=1.0, zorder=0)

                if rec.objective_final is None:
                    msg = f"{labels[use_ess]}: failed"
                else:
                    msg = f"{labels[use_ess]}: obj={float(rec.objective_final):.2e}"
                    if rec.aspect_final is not None:
                        msg += f", aspect={float(rec.aspect_final):.3f}"
                    if rec.iota_final is not None:
                        msg += f", iota={float(rec.iota_final):.4f}"
                annotation_lines.append(msg)

            if row == len(PROBLEMS) - 1:
                ax.set_xlabel("History index")
            if col == 0:
                ax.set_ylabel(f"{problem.upper()} total objective")
            ax.set_xlim(left=0)
            ax.grid(True, which="both", alpha=0.22)
            if annotation_lines:
                ax.text(
                    0.02,
                    0.02,
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8.8,
                    bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
                )

    handles, labels_list = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_list, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("QA/QH/QP/QI optimization sweep: objective histories with and without ESS", y=1.04, fontsize=14)
    fig.tight_layout()
    outpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_geometry_atlas(results: list[CaseResult], outpath_png: Path, outpath_pdf: Path) -> None:
    plt = _style_publication()
    import matplotlib.image as mpimg

    columns = [(mode, use_ess) for mode in MODES for use_ess in ESS_OPTIONS]
    row_specs = [
        ("qa", "boundary_comparison.png", "QA LCFS"),
        ("qa", "bmag_surface.png", "QA |B|"),
        ("qh", "boundary_comparison.png", "QH LCFS"),
        ("qh", "bmag_surface.png", "QH |B|"),
        ("qp", "boundary_comparison.png", "QP LCFS"),
        ("qp", "bmag_surface.png", "QP |B|"),
        ("qi", "boundary_comparison.png", "QI LCFS"),
        ("qi", "bmag_surface.png", "QI |B|"),
    ]

    fig, axes = plt.subplots(
        len(row_specs),
        len(columns),
        figsize=(3.1 * len(columns), 2.2 * len(row_specs)),
        squeeze=False,
    )
    for row, (problem, image_name, row_label) in enumerate(row_specs):
        for col, (mode, use_ess) in enumerate(columns):
            ax = axes[row, col]
            rec = next((r for r in results if _case_key(r) == (problem, mode, use_ess)), None)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"mode {mode}\n{_ess_label(use_ess).replace('_', ' ')}", fontsize=11)
            if col == 0:
                ax.text(
                    -0.06,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
            if rec is None or rec.output_dir is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                continue
            image_path = Path(rec.output_dir) / image_name
            if not image_path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                continue
            ax.imshow(mpimg.imread(image_path))
            ax.axis("off")

    fig.suptitle("Final equilibria across QA/QH/QP/QI, max_mode, and ESS settings", y=0.995, fontsize=14)
    fig.tight_layout()
    outpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)


def _write_summary_csv(results: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=[
                "policy",
                "backend",
                "problem",
                "max_mode",
                "use_ess",
                "success",
                "crashed",
                "objective_final",
                "qs_final",
                "aspect_final",
                "iota_final",
                "nfev",
                "njev",
                "total_wall_time_s",
                "profile_wall_time_s",
                "profile_top_name",
                "profile_top_wall_time_s",
                "profile_solve_forward_trial_total_wall_time_s",
                "profile_solve_forward_exact_total_wall_time_s",
                "profile_exact_tape_build_wall_time_s",
                "profile_exact_tape_build_solve_call_wall_time_s",
                "profile_exact_tape_build_unattributed_wall_time_s",
                "profile_exact_tape_solver_compute_forces_first_wall_time_s",
                "profile_exact_tape_solver_compute_forces_rest_wall_time_s",
                "profile_trial_solver_scan_total_wall_time_s",
                "profile_trial_solver_scan_runner_cache_lookup_wall_time_s",
                "profile_trial_solver_scan_runner_cache_build_wall_time_s",
                "profile_trial_solver_scan_runner_cache_hit_count",
                "profile_trial_solver_scan_runner_cache_miss_count",
                "profile_trial_solver_scan_runner_cache_bypass_count",
                "profile_trial_solver_scan_runner_cache_hit_device_run_wall_time_s",
                "profile_trial_solver_scan_runner_cache_miss_device_run_wall_time_s",
                "profile_trial_solver_scan_runner_cache_bypass_device_run_wall_time_s",
                "profile_trial_solver_scan_device_dispatch_wall_time_s",
                "profile_trial_solver_scan_device_ready_wall_time_s",
                "profile_trial_solver_scan_host_materialize_wall_time_s",
                "profile_jacobian_total_wall_time_s",
                "profile_write_wout_wall_time_s",
                "jax_backend",
                "jax_device_kind",
                "solver_device",
                "jax_platforms",
                "stellarator_asymmetric",
                "asymmetry_seed",
                "input_file",
                "input_nfp",
                "project_input_boundary_to_max_mode",
                "target_aspect",
                "target_iota",
                "iota_abs_min",
                "iota_weight",
                "lgradb_weight",
                "qi_lgradb_weight",
                "asymmetric_dof_count",
                "asymmetric_param_norm_initial",
                "asymmetric_param_norm_final",
                "asymmetric_param_norm_delta",
                "bmag_min",
                "bmag_max",
                "bmag_nonpositive_fraction",
                "bmag_finite",
                "lgradb_min",
                "lgradb_threshold",
                "lgradb_excess_max",
                "lgradb_diagnostic_error",
                "qi_qp_preseed",
                "qi_qi_preseed",
                "qi_jit_booz",
                "qi_raw_total",
                "qi_legacy_total",
                "qi_mirror_ratio_max",
                "qi_mirror_ratio_target",
                "qi_mirror_excess_max",
                "qi_max_elongation",
                "qi_elongation_target",
                "qi_elongation_excess",
                "qi_lgradb_min",
                "qi_lgradb_threshold",
                "qi_lgradb_excess_max",
                "qi_diagnostic_error",
                "message",
                "output_dir",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--backend-label", type=str, default=BACKEND_LABEL)
    parser.add_argument(
        "--solver-device",
        type=str,
        default=SOLVER_DEVICE,
        help="Passed to FixedBoundaryExactOptimizer: auto/default/cpu/gpu.",
    )
    parser.add_argument("--policy", choices=("continuation", "direct"), default="continuation")
    parser.add_argument("--problems", type=str, default=",".join(PROBLEMS))
    parser.add_argument("--modes", type=str, default=",".join(str(m) for m in MODES))
    parser.add_argument("--ess", choices=("both", "on", "off"), default="both")
    parser.add_argument(
        "--qi-qp-preseed",
        choices=("both", "on", "off"),
        default="off",
        help=(
            "For QI cases, choose whether to start from a same-mode QP preseed. "
            "Use 'both' to compare QP-preseed and direct-QI starts."
        ),
    )
    parser.add_argument(
        "--qi-jit-booz",
        choices=("on", "off"),
        default="on",
        help=(
            "For QI cases, use the jitted Boozer transform inside the QI "
            "objective. Default 'on' matches the user-facing QI optimization "
            "helpers; use 'off' only for diagnostics."
        ),
    )
    parser.add_argument(
        "--stellarator-asymmetric",
        action="store_true",
        default=STELLARATOR_ASYMMETRIC,
        help=(
            "Set LASYM=T in the in-memory VMEC input, optimize RBS/ZBC along "
            "with RBC/ZBS, and seed zero asymmetric modes by 1e-7."
        ),
    )
    parser.add_argument("--rerun", action="store_true", help="Recompute cases even if case_result.json exists.")
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=CASE_TIMEOUT_S,
        help=(
            "Wall-clock timeout per worker case. Use 0 to disable. Timed-out "
            "cases are recorded as crashed so large CPU/GPU sweeps can finish "
            "and render. Defaults to 1800 s."
        ),
    )
    parser.add_argument(
        "--worker-jax-platforms",
        type=str,
        default="inherit",
        help=(
            "Optional process-level JAX_PLATFORMS for worker subprocesses. "
            "The default 'inherit' preserves the user's CPU/GPU JAX backend; "
            "use 'cpu' only when an explicit CPU-only worker is desired."
        ),
    )
    parser.add_argument(
        "--diagnostic-budgets",
        action="store_true",
        default=DIAGNOSTIC_BUDGETS,
        help=(
            "Apply bounded per-case diagnostic budgets. By default CPU and GPU "
            "sweeps use the full optimization budgets."
        ),
    )
    parser.add_argument("--max-nfev", type=int, default=None, help="Override final-stage outer nfev budget.")
    parser.add_argument(
        "--continuation-nfev",
        type=int,
        default=None,
        help="Override per-lower-stage continuation nfev budget.",
    )
    parser.add_argument(
        "--inner-max-iter",
        type=int,
        default=None,
        help="Override accepted-point VMEC iteration budget; 0 uses input-deck NITER.",
    )
    parser.add_argument(
        "--inner-ftol",
        type=float,
        default=None,
        help="Override accepted-point VMEC tolerance; 0 uses input-deck FTOL.",
    )
    parser.add_argument(
        "--trial-max-iter",
        type=int,
        default=None,
        help="Override trial-point VMEC iteration budget; 0 follows accepted/input budget.",
    )
    parser.add_argument(
        "--trial-ftol",
        type=float,
        default=None,
        help="Override trial-point VMEC tolerance; 0 follows accepted/input tolerance.",
    )
    parser.add_argument(
        "--ess-alpha",
        type=float,
        default=None,
        help="Override ESS alpha for all selected cases.",
    )
    return parser.parse_args()


def main() -> None:
    global ESS_OPTIONS, MODES, PROBLEMS

    args = _parse_args()
    output_root = Path(args.output_root)
    backend_label = str(args.backend_label)
    solver_device = None if args.solver_device in (None, "", "none", "None") else str(args.solver_device)
    use_mode_continuation = args.policy == "continuation"
    problems = tuple(p.strip() for p in str(args.problems).split(",") if p.strip())
    modes = tuple(int(m.strip()) for m in str(args.modes).split(",") if m.strip())
    ess_options = {
        "both": ESS_OPTIONS,
        "on": (True,),
        "off": (False,),
    }[str(args.ess)]
    PROBLEMS = problems
    MODES = modes
    ESS_OPTIONS = ess_options
    qi_qp_preseed_options = {
        "both": (True, False),
        "on": (True,),
        "off": (False,),
    }[str(args.qi_qp_preseed)]
    qi_jit_booz = str(args.qi_jit_booz) == "on"
    symmetry_label = "asymmetric" if bool(args.stellarator_asymmetric) else "symmetric"
    case_timeout_s = None if args.case_timeout_s in (None, 0) else float(args.case_timeout_s)
    worker_jax_platforms_arg = str(args.worker_jax_platforms).strip()
    if worker_jax_platforms_arg.lower() in ("", "none", "inherit"):
        worker_jax_platforms = None
    elif worker_jax_platforms_arg.lower() == "auto":
        worker_jax_platforms = _default_worker_jax_platforms(solver_device)
    else:
        worker_jax_platforms = _normalize_worker_jax_platforms(worker_jax_platforms_arg)
    cli_budget = CaseBudget(
        max_nfev=args.max_nfev,
        continuation_nfev=args.continuation_nfev,
        inner_max_iter=args.inner_max_iter,
        inner_ftol=args.inner_ftol,
        trial_max_iter=args.trial_max_iter,
        trial_ftol=args.trial_ftol,
    )
    if all(getattr(cli_budget, field) is None for field in cli_budget.__dataclass_fields__):
        cli_budget = None

    output_root.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    results: list[CaseResult] = []

    for problem in problems:
        problem_qi_preseed_options: tuple[bool | None, ...] = (
            qi_qp_preseed_options if problem == "qi" else (None,)
        )
        for qi_qp_preseed in problem_qi_preseed_options:
            for max_mode in modes:
                for use_ess in ess_options:
                    output_base = output_root / backend_label
                    if bool(args.stellarator_asymmetric):
                        output_base = output_base / symmetry_label
                    output_dir = output_base / args.policy / problem
                    if problem == "qi":
                        output_dir = output_dir / ("qp_preseed" if qi_qp_preseed else "no_qp_preseed")
                    output_dir = output_dir / f"mode{max_mode}" / _ess_label(use_ess)
                    case_label = (
                        f"{backend_label} {symmetry_label} {args.policy} {problem}"
                        f"{' qp_preseed=' + str(qi_qp_preseed) if problem == 'qi' else ''} "
                        f"mode={max_mode} ess={use_ess}"
                    )
                    result_path = output_dir / "case_result.json"
                    if result_path.exists() and (not args.rerun):
                        record = json.loads(result_path.read_text())
                        if "backend" not in record:
                            record["backend"] = backend_label
                        result = CaseResult(**record)
                        existing_qi_jit_matches = True
                        if problem == "qi":
                            existing_qi_jit_matches = record.get("qi_jit_booz") is not None and bool(
                                record.get("qi_jit_booz")
                            ) == bool(qi_jit_booz)
                        if not bool(result.crashed):
                            if existing_qi_jit_matches:
                                results.append(result)
                                print(
                                    f"[{case_label}] skip existing success={result.success} "
                                    f"crashed={result.crashed} objective={result.objective_final}",
                                    flush=True,
                                )
                                continue
                            print(f"[{case_label}] rerun existing result with stale qi_jit_booz", flush=True)
                            result_path.unlink()
                        else:
                            print(
                                f"[{case_label}] rerun crashed existing success={result.success} "
                                f"crashed={result.crashed}",
                                flush=True,
                            )
                            result_path.unlink()
                    if result_path.exists() and args.rerun:
                        result_path.unlink()
                    stale_traceback = output_dir / "traceback.txt"
                    if stale_traceback.exists() and (args.rerun or not result_path.exists()):
                        stale_traceback.unlink()
                    proc = ctx.Process(
                        target=_worker,
                        args=(
                            problem,
                            max_mode,
                            use_ess,
                            str(output_dir),
                            str(result_path),
                            use_mode_continuation,
                            args.policy,
                            backend_label,
                            solver_device,
                            worker_jax_platforms,
                            bool(args.diagnostic_budgets),
                            bool(args.stellarator_asymmetric),
                            qi_qp_preseed,
                            qi_jit_booz,
                            cli_budget,
                            args.ess_alpha,
                        ),
                    )
                    case_t0 = time.perf_counter()
                    old_jax_platforms = os.environ.get("JAX_PLATFORMS")
                    if worker_jax_platforms is not None:
                        os.environ["JAX_PLATFORMS"] = worker_jax_platforms
                    try:
                        proc.start()
                    finally:
                        if old_jax_platforms is None:
                            os.environ.pop("JAX_PLATFORMS", None)
                        else:
                            os.environ["JAX_PLATFORMS"] = old_jax_platforms
                    proc.join(timeout=case_timeout_s)
                    elapsed_s = time.perf_counter() - case_t0
                    timed_out = proc.is_alive()
                    if timed_out:
                        _terminate_worker_process(proc)

                    result_loaded_from_path = False
                    if result_path.exists():
                        result = CaseResult(**json.loads(result_path.read_text()))
                        result_needs_write = _set_missing_wall_time(result, elapsed_s)
                        result_loaded_from_path = True
                    elif timed_out:
                        result = CaseResult(
                            backend=backend_label,
                            problem=problem,
                            max_mode=max_mode,
                            use_ess=bool(use_ess),
                            success=False,
                            crashed=True,
                            message=f"worker timed out after {case_timeout_s:.1f} s",
                            policy=args.policy,
                            output_dir=str(output_dir),
                            solver_device=solver_device,
                            jax_platforms=worker_jax_platforms,
                            total_wall_time_s=elapsed_s,
                            stellarator_asymmetric=bool(args.stellarator_asymmetric),
                            asymmetry_seed=float(ASYMMETRIC_SEED if args.stellarator_asymmetric else 0.0),
                            qi_qp_preseed=(
                                bool(qi_qp_preseed) if problem == "qi" and qi_qp_preseed is not None else None
                            ),
                            qi_jit_booz=(qi_jit_booz if problem == "qi" else None),
                        )
                        result_needs_write = True
                    else:
                        result = CaseResult(
                            backend=backend_label,
                            problem=problem,
                            max_mode=max_mode,
                            use_ess=bool(use_ess),
                            success=False,
                            crashed=True,
                            message=f"worker exit code {proc.exitcode} without result file",
                            policy=args.policy,
                            output_dir=str(output_dir),
                            solver_device=solver_device,
                            jax_platforms=worker_jax_platforms,
                            total_wall_time_s=elapsed_s,
                            stellarator_asymmetric=bool(args.stellarator_asymmetric),
                            asymmetry_seed=float(ASYMMETRIC_SEED if args.stellarator_asymmetric else 0.0),
                            qi_qp_preseed=(
                                bool(qi_qp_preseed) if problem == "qi" and qi_qp_preseed is not None else None
                            ),
                            qi_jit_booz=(qi_jit_booz if problem == "qi" else None),
                        )
                        result_needs_write = True
                    if not result_loaded_from_path:
                        result = _annotate_result_from_stage_checkpoint(result)
                    if timed_out:
                        result_needs_write = _mark_timed_out_result(
                            result,
                            elapsed_s=elapsed_s,
                            case_timeout_s=case_timeout_s,
                        )
                    elif proc.exitcode not in (0, None):
                        result.crashed = True
                        if "worker exit code" not in result.message:
                            result.message = f"exit code {proc.exitcode}; {result.message}"
                        result_needs_write = True
                    if result_needs_write or not result_path.exists():
                        output_dir.mkdir(parents=True, exist_ok=True)
                        _atomic_write_json(result_path, asdict(result))
                    results.append(result)
                    print(
                        f"[{case_label}] success={result.success} crashed={result.crashed} "
                        f"objective={result.objective_final}",
                        flush=True,
                    )

    summary = [asdict(r) for r in results]
    name_suffix = (
        f"{backend_label}_{symmetry_label}_{args.policy}"
        if bool(args.stellarator_asymmetric)
        else f"{backend_label}_{args.policy}"
    )
    summary_name = f"summary_{name_suffix}.json"
    csv_name = f"summary_{name_suffix}.csv"
    (output_root / summary_name).write_text(json.dumps(summary, indent=2))
    _write_summary_csv(results, output_root / csv_name)
    _plot_objective_panel(
        results,
        outpath_png=output_root / f"objective_panel_{name_suffix}.png",
        outpath_pdf=output_root / f"objective_panel_{name_suffix}.pdf",
    )
    _plot_geometry_atlas(
        results,
        outpath_png=output_root / f"geometry_atlas_{name_suffix}.png",
        outpath_pdf=output_root / f"geometry_atlas_{name_suffix}.pdf",
    )
    print(f"Wrote {output_root}")


if __name__ == "__main__":
    main()

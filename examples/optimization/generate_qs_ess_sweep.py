#!/usr/bin/env python
# ruff: noqa: E402
"""Run QA/QH/QP/QI ESS policy sweeps and build publication-style summary panels.

This script regenerates a reviewer-facing benchmark matrix for the standalone
``vmec_jax`` exact optimization path:

- problems: QA, QH, QP, and QI
- policies: continuation and direct-start mode expansion
- max_mode: 1, 2, 3, 4 for continuation and direct start
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
from dataclasses import asdict, dataclass, replace
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

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
from vmec_jax.quasi_isodynamic import _nearest_half_mesh_indices, quasi_isodynamic_residual_from_state
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


enable_x64(True)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "examples" / "data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "qs_ess_sweep"

VMEC_MPOL = 5
VMEC_NTOR = 5
MODES = (1, 2, 3, 4)
PROBLEMS = ("qa", "qh", "qp", "qi")
ESS_OPTIONS = (False, True)

USE_MODE_CONTINUATION = True
BACKEND_LABEL = "cpu"
SOLVER_DEVICE: str | None = None
SKIP_EXISTING = True
CASE_TIMEOUT_S: float | None = 1200.0
ESS_ALPHA = 2.5
STELLARATOR_ASYMMETRIC = False
ASYMMETRIC_SEED = 1.0e-7
DIAGNOSTIC_BUDGETS = False
GPU_PRODUCTION_INNER_MAX_ITER = 180
GPU_PRODUCTION_INNER_FTOL = 1e-9
GPU_PRODUCTION_TRIAL_MAX_ITER = 180
GPU_PRODUCTION_TRIAL_FTOL = 1e-9


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
# controlled by small per-case nfev caps.  Let the 20-minute case timeout decide
# whether high-mode LASYM cases are affordable; keep explicit low nfev caps only
# in ``CASE_BUDGET_OVERRIDES`` for opt-in ``--diagnostic-budgets`` runs.
GPU_PRODUCTION_BUDGET_OVERRIDES: dict[tuple[str, str, str, int, bool], CaseBudget] = {}


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
    objective_kind: str = "qs"
    qi_mboz: int = 6
    qi_nboz: int = 6
    qi_nphi: int = 41
    qi_nalpha: int = 13
    qi_n_bounce: int = 11
    qi_softness: float = 2.0e-2
    qi_profile_weight: float = 1.0
    qi_preseed_qp: bool = False


PROBLEM_CONFIGS = {
    "qa": ProblemConfig(
        name="qa",
        input_file=DATA_DIR / "input.nfp2_QA",
        method="scipy",
        scipy_tr_solver="lsmr",
        scipy_lsmr_maxiter=None,
        max_nfev=60,
        continuation_nfev=30,
        ftol=1e-6,
        gtol=1e-6,
        xtol=1e-6,
        # QA direct max_mode=3 can fall into a zero-iota stationary branch:
        # aspect and QS improve, but iota does not move.  ESS is still useful
        # for conditioning, but staged mode continuation is the reliable path
        # for the documented finite-iota QA minimum.
        ess_alpha=ESS_ALPHA,
        target_aspect=6.0,
        target_iota=0.41,
        iota_weight=10.0,
        surfaces=np.arange(0.0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=0,
        inner_max_iter=0,
        inner_ftol=0.0,
        trial_max_iter=300,
        trial_ftol=1e-10,
    ),
    "qh": ProblemConfig(
        name="qh",
        input_file=DATA_DIR / "input.nfp4_QH_warm_start",
        method="scipy",
        scipy_tr_solver="lsmr",
        scipy_lsmr_maxiter=None,
        max_nfev=60,
        continuation_nfev=30,
        ftol=1e-6,
        gtol=1e-6,
        xtol=1e-6,
        ess_alpha=ESS_ALPHA,
        target_aspect=7.0,
        target_iota=None,
        iota_abs_min=0.4,
        iota_weight=100.0,
        surfaces=np.arange(0.0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
        inner_max_iter=0,
        inner_ftol=0.0,
        trial_max_iter=300,
        trial_ftol=1e-10,
    ),
    "qp": ProblemConfig(
        name="qp",
        input_file=DATA_DIR / "input.nfp4_QH_warm_start",
        method="scipy",
        scipy_tr_solver="lsmr",
        scipy_lsmr_maxiter=None,
        max_nfev=50,
        continuation_nfev=20,
        ftol=1e-5,
        gtol=1e-5,
        xtol=1e-5,
        ess_alpha=ESS_ALPHA,
        target_aspect=7.0,
        target_iota=None,
        surfaces=np.arange(0.0, 1.01, 0.1),
        helicity_m=0,
        helicity_n=-1,
        inner_max_iter=120,
        inner_ftol=1e-9,
        trial_max_iter=120,
        trial_ftol=1e-9,
        iota_abs_min=0.4,
        iota_weight=100.0,
    ),
    "qi": ProblemConfig(
        name="qi",
        input_file=DATA_DIR / "input.nfp4_QH_warm_start",
        method="scipy",
        scipy_tr_solver="lsmr",
        scipy_lsmr_maxiter=None,
        max_nfev=30,
        continuation_nfev=12,
        ftol=1e-4,
        gtol=1e-4,
        xtol=1e-4,
        ess_alpha=ESS_ALPHA,
        target_aspect=7.0,
        target_iota=None,
        surfaces=np.linspace(0.2, 1.0, 5),
        helicity_m=0,
        helicity_n=0,
        inner_max_iter=120,
        inner_ftol=1e-9,
        trial_max_iter=120,
        trial_ftol=1e-9,
        iota_abs_min=0.4,
        iota_weight=100.0,
        objective_kind="qi",
        qi_mboz=6,
        qi_nboz=6,
        qi_nphi=41,
        qi_nalpha=13,
        qi_n_bounce=11,
        qi_softness=2.0e-2,
        qi_profile_weight=1.5,
        qi_preseed_qp=True,
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
    output_dir: str | None = None
    jax_backend: str | None = None
    jax_device_kind: str | None = None
    solver_device: str | None = None
    jax_platforms: str | None = None
    stellarator_asymmetric: bool = False
    asymmetry_seed: float = 0.0
    asymmetric_dof_count: int = 0
    asymmetric_param_norm_initial: float | None = None
    asymmetric_param_norm_final: float | None = None
    asymmetric_param_norm_delta: float | None = None
    bmag_min: float | None = None
    bmag_max: float | None = None
    bmag_nonpositive_fraction: float | None = None
    bmag_finite: bool | None = None


def _set_missing_wall_time(result: CaseResult, elapsed_s: float) -> bool:
    """Record outer worker elapsed time when the inner run could not report it."""
    if result.total_wall_time_s is not None:
        return False
    result.total_wall_time_s = float(elapsed_s)
    return True


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
    return ESS_ALPHA


def _effective_problem_config(
    problem_cfg: ProblemConfig,
    *,
    backend: str,
    policy: str,
    problem: str,
    max_mode: int,
    use_ess: bool,
    diagnostic_budgets: bool = DIAGNOSTIC_BUDGETS,
) -> ProblemConfig:
    updates = {}
    backend_key = "gpu" if str(backend).lower().startswith("gpu") else str(backend).lower()
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


def _load_problem(cfg: ProblemConfig, *, stellarator_asymmetric: bool = STELLARATOR_ASYMMETRIC):
    cfg0, indata = vj.load_config(str(cfg.input_file))
    del cfg0
    indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
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


def _build_stage(problem_cfg: ProblemConfig, cfg, indata0, max_mode: int, *, solver_device: str | None):
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
                profile_weight=problem_cfg.qi_profile_weight,
                jit_booz=False,
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
        parts.append(jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * problem_cfg.qs_weight)
        return jnp.concatenate(parts)

    n_iota_terms = int(problem_cfg.target_iota is not None) + int(problem_cfg.iota_abs_min is not None)
    stage_residuals_from_state._n_non_qs = 1 + n_iota_terms
    stage_residuals_from_state._qs_total_from_state = (
        lambda state: float(problem_cfg.qs_weight) ** 2 * float(stage_qs_eval(state)["total"])
    )

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


def _merge_stage_histories(stage_results: list[StageRecord], *, problem_cfg: ProblemConfig) -> dict:
    combined_entries = []
    stage_boundaries = []
    wall_offset = 0.0
    nfev_total = 0
    njev_total = 0
    max_nfev_total = 0
    for idx, (stage_label, _mode, stage_result) in enumerate(stage_results):
        stage_hist = stage_result["_history_dump"]
        entries = stage_hist["history"] if idx == 0 else stage_hist["history"][1:]
        for entry in entries:
            entry_copy = dict(entry)
            entry_copy["wall_time_s"] = float(entry_copy["wall_time_s"]) + wall_offset
            entry_copy["stage"] = stage_label
            combined_entries.append(entry_copy)
        wall_offset = combined_entries[-1]["wall_time_s"]
        nfev_total += int(stage_hist["nfev"])
        njev_total += int(stage_hist["njev"])
        max_nfev_total += int(stage_hist.get("max_nfev", stage_hist["nfev"]))
        stage_boundaries.append(len(combined_entries) - 1)

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
        "stage_modes": [int(mode) for _label, mode, _result in stage_results],
        "stage_labels": [str(label) for label, _mode, _result in stage_results],
    }
    if problem_cfg.target_iota is not None:
        merged["target_iota"] = float(problem_cfg.target_iota)
    if problem_cfg.iota_abs_min is not None:
        merged["iota_abs_min"] = float(problem_cfg.iota_abs_min)
    if problem_cfg.target_iota is not None or problem_cfg.iota_abs_min is not None:
        if "iota" in combined_entries[0] and "iota" in combined_entries[-1]:
            merged["iota_initial"] = float(combined_entries[0]["iota"])
            merged["iota_final"] = float(combined_entries[-1]["iota"])
    return merged


def _save_case_outputs(output_dir: Path, opt, params_initial, params_final, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    opt.save_input(output_dir / "input.initial", params_initial)
    opt.save_wout(output_dir / "wout_initial.nc", params_initial, state=result.get("_state_initial"))
    opt.save_input(output_dir / "input.final", params_final)
    opt.save_wout(output_dir / "wout_final.nc", params_final, state=result.get("_state_final"))
    opt.save_history(output_dir / "history.json", result)
    try:
        vj.plot_qh_optimization(
            output_dir / "wout_initial.nc",
            output_dir / "wout_final.nc",
            output_dir / "history.json",
            outdir=output_dir,
        )
    except Exception as exc:
        # Plotting is a post-processing convenience.  Do not mark an otherwise
        # valid optimization as failed because a remote/headless Matplotlib
        # environment has a broken 3D toolkit.
        (output_dir / "plotting_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        print(f"  Skipped case plots: {type(exc).__name__}: {exc}", flush=True)


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
) -> tuple[list[StageRecord], object, object, object, object, dict]:
    stage_modes = list(range(1, max_mode + 1)) if (use_mode_continuation and max_mode > 1) else [max_mode]
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
        stage_results.append((f"{stage_label_prefix} mode {stage_mode}", stage_mode, stage_result))
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
    *,
    use_mode_continuation: bool,
    policy: str,
    backend: str,
    solver_device: str | None,
    jax_platforms: str | None,
    diagnostic_budgets: bool = DIAGNOSTIC_BUDGETS,
    stellarator_asymmetric: bool = STELLARATOR_ASYMMETRIC,
) -> CaseResult:
    problem_cfg = _effective_problem_config(
        PROBLEM_CONFIGS[problem],
        backend=backend,
        policy=policy,
        problem=problem,
        max_mode=max_mode,
        use_ess=use_ess,
        diagnostic_budgets=diagnostic_budgets,
    )
    cfg, indata = _load_problem(
        problem_cfg,
        stellarator_asymmetric=stellarator_asymmetric,
    )
    jax_backend, jax_device_kind = _jax_runtime_info()

    stage_results: list[StageRecord] = []
    params_stage = None
    prev_specs = None

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
        )
        stage_results.extend(qp_stage_results)
        original_stage = (qp_opt, qp_params0, qp_result)
        qp_seed_stage = (qp_opt, qp_result["x"], qp_result)

    qi_or_qs_stage_results, prev_specs, params_stage, final_opt, final_params0, final_result = _run_problem_stages(
        problem_cfg=problem_cfg,
        problem=problem,
        max_mode=max_mode,
        use_ess=use_ess,
        use_mode_continuation=False if problem_cfg.objective_kind == "qi" and problem_cfg.qi_preseed_qp else use_mode_continuation,
        solver_device=solver_device,
        cfg=cfg,
        indata=indata,
        stage_label_prefix=problem.upper(),
        params_stage=params_stage,
        prev_specs=prev_specs,
        stellarator_asymmetric=stellarator_asymmetric,
    )
    stage_results.extend(qi_or_qs_stage_results)

    if len(stage_results) > 1:
        final_result["_history_dump"] = _merge_stage_histories(stage_results, problem_cfg=problem_cfg)

    _save_case_outputs(output_dir, final_opt, final_params0, final_result["x"], final_result)
    bmag_stats = _bmag_lcfs_stats(output_dir / "wout_final.nc")
    if original_stage is not None:
        original_opt, original_params0, original_result = original_stage
        original_opt.save_input(output_dir / "input.original", original_params0)
        original_opt.save_wout(output_dir / "wout_original.nc", original_params0, state=original_result.get("_state_initial"))
    if qp_seed_stage is not None:
        qp_opt, qp_params, qp_result = qp_seed_stage
        qp_opt.save_input(output_dir / "input.qp_seed", qp_params)
        qp_opt.save_wout(output_dir / "wout_qp_seed.nc", qp_params, state=qp_result.get("_state_final"))

    hist = final_result["_history_dump"]
    final_iota = None
    if (problem_cfg.target_iota is not None or problem_cfg.iota_abs_min is not None) and hist["history"]:
        final_iota = float(hist["history"][-1]["iota"])
    params_initial_for_stats = np.zeros(len(prev_specs), dtype=float)
    if bool(stellarator_asymmetric):
        params_initial_for_stats = _seed_zero_asymmetric_params(
            boundary_input=final_opt._boundary_input,
            specs=prev_specs,
            params=params_initial_for_stats,
            seed=ASYMMETRIC_SEED,
        )
    asym_stats = _asymmetric_param_stats(prev_specs, params_initial_for_stats, final_result["x"])

    return CaseResult(
        backend=str(backend),
        problem=problem,
        max_mode=max_mode,
        use_ess=bool(use_ess),
        success=bool(hist["success"]),
        crashed=False,
        message=str(hist["message"]),
        policy=policy,
        objective_final=float(hist["objective_final"]),
        qs_final=float(hist["qs_final"]),
        aspect_final=float(hist["aspect_final"]),
        iota_final=final_iota,
        nfev=int(hist["nfev"]),
        njev=int(hist["njev"]),
        total_wall_time_s=float(hist["total_wall_time_s"]),
        output_dir=str(output_dir),
        jax_backend=jax_backend,
        jax_device_kind=jax_device_kind,
        solver_device=solver_device,
        jax_platforms=jax_platforms,
        stellarator_asymmetric=bool(stellarator_asymmetric),
        asymmetry_seed=float(ASYMMETRIC_SEED if stellarator_asymmetric else 0.0),
        **asym_stats,
        **bmag_stats,
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
):
    try:
        case_result = _run_case(
            problem,
            max_mode,
            use_ess,
            Path(output_dir),
            use_mode_continuation=use_mode_continuation,
            policy=policy,
            backend=backend,
            solver_device=solver_device,
            jax_platforms=jax_platforms,
            diagnostic_budgets=diagnostic_budgets,
            stellarator_asymmetric=stellarator_asymmetric,
        )
        Path(result_path).write_text(json.dumps(asdict(case_result), indent=2))
        stale_traceback = Path(output_dir) / "traceback.txt"
        if stale_traceback.exists():
            stale_traceback.unlink()
    except Exception as exc:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
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
        )
        Path(result_path).write_text(json.dumps(asdict(failed), indent=2))
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
                y = np.asarray([max(float(entry["objective"]), 1e-16) for entry in hist["history"]], dtype=float)
                x = np.arange(len(y), dtype=int)
                linestyle = "-" if rec.success and not rec.crashed else "--"
                linewidth = 2.6 if use_ess else 2.1
                ax.semilogy(
                    x,
                    y,
                    color=colors[use_ess],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=labels[use_ess] if (row == 0 and col == 0) else None,
                )
                ax.scatter(x[-1], y[-1], color=colors[use_ess], s=30, zorder=4)
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
                "jax_backend",
                "jax_device_kind",
                "solver_device",
                "jax_platforms",
                "stellarator_asymmetric",
                "asymmetry_seed",
                "asymmetric_dof_count",
                "asymmetric_param_norm_initial",
                "asymmetric_param_norm_final",
                "asymmetric_param_norm_delta",
                "bmag_min",
                "bmag_max",
                "bmag_nonpositive_fraction",
                "bmag_finite",
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
            "and render. Defaults to 1200 s."
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
    return parser.parse_args()


def main() -> None:
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
    symmetry_label = "asymmetric" if bool(args.stellarator_asymmetric) else "symmetric"
    case_timeout_s = None if args.case_timeout_s in (None, 0) else float(args.case_timeout_s)
    worker_jax_platforms_arg = str(args.worker_jax_platforms).strip()
    if worker_jax_platforms_arg.lower() in ("", "none", "inherit"):
        worker_jax_platforms = None
    elif worker_jax_platforms_arg.lower() == "auto":
        worker_jax_platforms = _default_worker_jax_platforms(solver_device)
    else:
        worker_jax_platforms = worker_jax_platforms_arg

    output_root.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    results: list[CaseResult] = []

    for problem in problems:
        for max_mode in modes:
            for use_ess in ess_options:
                output_base = output_root / backend_label
                if bool(args.stellarator_asymmetric):
                    output_base = output_base / symmetry_label
                output_dir = output_base / args.policy / problem / f"mode{max_mode}" / _ess_label(use_ess)
                result_path = output_dir / "case_result.json"
                if result_path.exists() and (not args.rerun):
                    record = json.loads(result_path.read_text())
                    if "backend" not in record:
                        record["backend"] = backend_label
                    result = CaseResult(**record)
                    if bool(result.success) and not bool(result.crashed):
                        results.append(result)
                        print(
                            f"[{backend_label} {symmetry_label} {args.policy} {problem} mode={max_mode} ess={use_ess}] "
                            f"skip existing success={result.success} crashed={result.crashed} "
                            f"objective={result.objective_final}",
                            flush=True,
                        )
                        continue
                    print(
                        f"[{backend_label} {symmetry_label} {args.policy} {problem} mode={max_mode} ess={use_ess}] "
                        f"rerun failed existing success={result.success} crashed={result.crashed}",
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
                    proc.terminate()
                    proc.join(timeout=10.0)
                    if proc.is_alive():
                        try:
                            os.kill(proc.pid, 9)
                        except (OSError, TypeError):
                            pass
                        proc.join()

                if result_path.exists():
                    result = CaseResult(**json.loads(result_path.read_text()))
                    result_needs_write = _set_missing_wall_time(result, elapsed_s)
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
                    )
                    result_needs_write = True
                if proc.exitcode not in (0, None):
                    result.crashed = True
                    if "worker exit code" not in result.message:
                        result.message = f"exit code {proc.exitcode}; {result.message}"
                    result_needs_write = True
                if result_needs_write or not result_path.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(json.dumps(asdict(result), indent=2))
                results.append(result)
                print(
                    f"[{backend_label} {symmetry_label} {args.policy} {problem} mode={max_mode} ess={use_ess}] "
                    f"success={result.success} crashed={result.crashed} "
                    f"objective={result.objective_final}"
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

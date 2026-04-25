#!/usr/bin/env python
# ruff: noqa: E402
"""Run QA/QH/QP ESS policy sweeps and build publication-style summary panels.

This script regenerates a reviewer-facing benchmark matrix for the standalone
``vmec_jax`` exact optimization path:

- problems: QA, QH, and QP
- policies: continuation and direct-start mode expansion
- max_mode: 1, 2, 3 for continuation and direct start
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
import time
import traceback

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64, jnp
from vmec_jax.config import config_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.optimization import rebuild_indata_with_resolution
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


enable_x64(True)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "examples" / "data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "qs_ess_sweep"

VMEC_MPOL = 5
VMEC_NTOR = 5
MODES = (1, 2, 3)
PROBLEMS = ("qa", "qh", "qp")
ESS_OPTIONS = (False, True)

USE_MODE_CONTINUATION = True
BACKEND_LABEL = "cpu"
SOLVER_DEVICE: str | None = None
SKIP_EXISTING = True
CASE_TIMEOUT_S: float | None = None
ESS_ALPHA = 2.5


@dataclass(frozen=True)
class CaseBudget:
    max_nfev: int | None = None
    continuation_nfev: int | None = None
    inner_max_iter: int | None = None
    inner_ftol: float | None = None
    trial_max_iter: int | None = None
    trial_ftol: float | None = None


# Bounded policies for cases that are known to be poor/runaway diagnostics.
# This keeps full GPU panel regeneration finite while still recording the poor
# policy as a failed or weak result instead of blocking the whole sweep.
CASE_BUDGET_OVERRIDES: dict[tuple[str, str, str, int, bool], CaseBudget] = {
    ("gpu", "direct", "qa", 2, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qa", 3, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qh", 2, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qh", 3, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qp", 2, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qp", 3, False): CaseBudget(max_nfev=4, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qa", 3, True): CaseBudget(max_nfev=5, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qh", 3, True): CaseBudget(max_nfev=5, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
    ("gpu", "direct", "qp", 3, True): CaseBudget(max_nfev=5, inner_max_iter=40, trial_max_iter=40, trial_ftol=1e-8),
}


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
    aspect_weight: float = 1.0
    iota_weight: float = 1.0
    qs_weight: float = 1.0


PROBLEM_CONFIGS = {
    "qa": ProblemConfig(
        name="qa",
        input_file=DATA_DIR / "input.nfp2_QA",
        method="scipy",
        scipy_tr_solver="lsmr",
        scipy_lsmr_maxiter=None,
        max_nfev=40,
        continuation_nfev=25,
        ftol=1e-5,
        gtol=1e-5,
        xtol=1e-5,
        # QA direct max_mode=3 can fall into a zero-iota stationary branch:
        # aspect and QS improve, but iota does not move.  ESS is still useful
        # for conditioning, but staged mode continuation is the reliable path
        # for the documented finite-iota QA minimum.
        ess_alpha=ESS_ALPHA,
        target_aspect=6.0,
        target_iota=0.41,
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
        max_nfev=30,
        continuation_nfev=15,
        ftol=1e-3,
        gtol=1e-3,
        xtol=1e-3,
        ess_alpha=ESS_ALPHA,
        target_aspect=7.0,
        target_iota=None,
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
        max_nfev=20,
        continuation_nfev=8,
        ftol=1e-4,
        gtol=1e-4,
        xtol=1e-4,
        ess_alpha=ESS_ALPHA,
        target_aspect=7.0,
        target_iota=-0.31,
        surfaces=np.arange(0.0, 1.01, 0.1),
        helicity_m=0,
        helicity_n=-1,
        inner_max_iter=80,
        inner_ftol=1e-8,
        trial_max_iter=80,
        trial_ftol=1e-8,
        iota_abs_min=0.31,
        iota_weight=20.0,
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
) -> ProblemConfig:
    updates = {}
    if str(backend).lower() == "gpu":
        # GPU callbacks are still cold-compile/dispatch dominated. Keep the
        # full panel finite and comparable by bounding every GPU case; use CPU
        # runs for production-accuracy long budgets.
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
    budget = CASE_BUDGET_OVERRIDES.get(
        (str(backend), str(policy), str(problem), int(max_mode), bool(use_ess))
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


def _load_problem(cfg: ProblemConfig):
    cfg0, indata = vj.load_config(str(cfg.input_file))
    del cfg0
    indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
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
        include=("rc", "zs"),
        fix=("rc00",),
    )

    stage_guess = initial_guess_from_boundary(stage_static, stage_boundary, stage_indata, vmec_project=True)
    stage_geom = eval_geom(stage_guess, stage_static)
    stage_signgs = int(signgs_from_sqrtg(np.asarray(stage_geom.sqrtg), axis_index=1))
    stage_flux = vj.flux_profiles_from_indata(stage_indata, stage_static.s, signgs=stage_signgs)
    stage_pressure = jnp.zeros_like(jnp.asarray(stage_static.s))

    def stage_qs_eval(state):
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

    def stage_residuals_from_state(state):
        parts = []
        aspect = equilibrium_aspect_ratio_from_state(state=state, static=stage_static)
        parts.append(
            jnp.asarray(
                [problem_cfg.aspect_weight * (aspect - problem_cfg.target_aspect)],
                dtype=jnp.float64,
            )
        )
        if problem_cfg.target_iota is not None:
            iota = mean_iota_raw(state)
            if problem_cfg.iota_abs_min is None:
                iota_residual = iota - problem_cfg.target_iota
            else:
                iota_residual = jnp.minimum(jnp.abs(iota) - problem_cfg.iota_abs_min, 0.0)
            parts.append(
                jnp.asarray([problem_cfg.iota_weight * iota_residual], dtype=jnp.float64)
            )
        qs = stage_qs_eval(state)
        parts.append(jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * problem_cfg.qs_weight)
        return jnp.concatenate(parts)

    stage_residuals_from_state._n_non_qs = 2 if problem_cfg.target_iota is not None else 1
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
    return stage_specs, stage_opt, iota_fn


def _merge_stage_histories(stage_results: list[tuple[int, dict]], *, problem_cfg: ProblemConfig) -> dict:
    combined_entries = []
    stage_boundaries = []
    wall_offset = 0.0
    nfev_total = 0
    njev_total = 0
    for idx, (_mode, stage_result) in enumerate(stage_results):
        stage_hist = stage_result["_history_dump"]
        entries = stage_hist["history"] if idx == 0 else stage_hist["history"][1:]
        for entry in entries:
            entry_copy = dict(entry)
            entry_copy["wall_time_s"] = float(entry_copy["wall_time_s"]) + wall_offset
            combined_entries.append(entry_copy)
        wall_offset = combined_entries[-1]["wall_time_s"]
        nfev_total += int(stage_hist["nfev"])
        njev_total += int(stage_hist["njev"])
        stage_boundaries.append(len(combined_entries) - 1)

    final_hist = stage_results[-1][1]["_history_dump"]
    merged = {
        "label": "Optimisation",
        "max_nfev": int(
            sum(
                problem_cfg.continuation_nfev if mode != stage_results[-1][0] else problem_cfg.max_nfev
                for mode, _ in stage_results
            )
        ),
        "ftol": problem_cfg.ftol,
        "gtol": problem_cfg.gtol,
        "xtol": problem_cfg.xtol,
        "total_wall_time_s": float(wall_offset),
        "nfev": int(nfev_total),
        "njev": int(njev_total),
        "success": bool(final_hist["success"]),
        "message": str(final_hist["message"]),
        "objective_initial": float(stage_results[0][1]["_history_dump"]["objective_initial"]),
        "objective_final": float(final_hist["objective_final"]),
        "qs_initial": float(stage_results[0][1]["_history_dump"]["qs_initial"]),
        "qs_final": float(final_hist["qs_final"]),
        "aspect_initial": float(stage_results[0][1]["_history_dump"]["aspect_initial"]),
        "aspect_final": float(final_hist["aspect_final"]),
        "history": combined_entries,
        "target_aspect": problem_cfg.target_aspect,
        "stage_boundaries": stage_boundaries,
        "stage_modes": [int(mode) for mode, _ in stage_results],
    }
    if problem_cfg.target_iota is not None:
        merged["target_iota"] = float(problem_cfg.target_iota)
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
) -> CaseResult:
    problem_cfg = _effective_problem_config(
        PROBLEM_CONFIGS[problem],
        backend=backend,
        policy=policy,
        problem=problem,
        max_mode=max_mode,
        use_ess=use_ess,
    )
    cfg, indata = _load_problem(problem_cfg)
    jax_backend, jax_device_kind = _jax_runtime_info()

    stage_modes = list(range(1, max_mode + 1)) if (use_mode_continuation and max_mode > 1) else [max_mode]
    stage_results: list[tuple[int, dict]] = []
    params_stage = None
    prev_specs = None
    final_opt = None
    final_params0 = None
    final_result = None

    for stage_mode in stage_modes:
        stage_specs, stage_opt, iota_fn = _build_stage(
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
        if stage_mode == max_mode:
            stage_budget = problem_cfg.max_nfev
        else:
            stage_budget = problem_cfg.continuation_nfev
        stage_result = stage_opt.run(
            params0_stage,
            method=problem_cfg.method,
            max_nfev=stage_budget,
            ftol=problem_cfg.ftol,
            gtol=problem_cfg.gtol,
            xtol=problem_cfg.xtol,
            x_scale=stage_x_scale,
            verbose=0,
            iota_fn=iota_fn if problem_cfg.target_iota is not None else None,
            target_iota=problem_cfg.target_iota,
            target_aspect=problem_cfg.target_aspect,
            scipy_tr_solver=problem_cfg.scipy_tr_solver,
            scipy_lsmr_maxiter=problem_cfg.scipy_lsmr_maxiter,
        )
        stage_results.append((stage_mode, stage_result))
        prev_specs = stage_specs
        params_stage = stage_result["x"]
        final_opt = stage_opt
        final_params0 = params0_stage
        final_result = stage_result

    assert final_opt is not None
    assert final_params0 is not None
    assert final_result is not None

    if use_mode_continuation and len(stage_results) > 1:
        final_result["_history_dump"] = _merge_stage_histories(stage_results, problem_cfg=problem_cfg)

    _save_case_outputs(output_dir, final_opt, final_params0, final_result["x"], final_result)

    hist = final_result["_history_dump"]
    final_iota = None
    if problem_cfg.target_iota is not None and hist["history"]:
        final_iota = float(hist["history"][-1]["iota"])

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
        )
        Path(result_path).write_text(json.dumps(asdict(case_result), indent=2))
    except Exception as exc:
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
        )
        Path(result_path).write_text(json.dumps(asdict(failed), indent=2))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
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

    fig, axes = plt.subplots(len(PROBLEMS), 3, figsize=(16.5, 4.3 * len(PROBLEMS)), sharey="row")
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
    fig.suptitle("QA/QH/QP optimization sweep: objective histories with and without ESS", y=1.04, fontsize=14)
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
    ]

    fig, axes = plt.subplots(len(row_specs), len(columns), figsize=(19.0, 17.4))
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

    fig.suptitle("Final equilibria across QA/QH/QP, max_mode, and ESS settings", y=0.995, fontsize=14)
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
    parser.add_argument("--rerun", action="store_true", help="Recompute cases even if case_result.json exists.")
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=CASE_TIMEOUT_S,
        help=(
            "Optional wall-clock timeout per worker case. Timed-out cases are "
            "recorded as crashed so large CPU/GPU sweeps can finish and render."
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
                output_dir = output_root / backend_label / args.policy / problem / f"mode{max_mode}" / _ess_label(use_ess)
                result_path = output_dir / "case_result.json"
                if result_path.exists() and (not args.rerun):
                    record = json.loads(result_path.read_text())
                    if "backend" not in record:
                        record["backend"] = backend_label
                    result = CaseResult(**record)
                    results.append(result)
                    print(
                        f"[{backend_label} {args.policy} {problem} mode={max_mode} ess={use_ess}] "
                        f"skip existing success={result.success} crashed={result.crashed} "
                        f"objective={result.objective_final}",
                        flush=True,
                    )
                    continue
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
                    f"[{backend_label} {args.policy} {problem} mode={max_mode} ess={use_ess}] "
                    f"success={result.success} crashed={result.crashed} "
                    f"objective={result.objective_final}"
                )

    summary = [asdict(r) for r in results]
    summary_name = f"summary_{backend_label}_{args.policy}.json"
    csv_name = f"summary_{backend_label}_{args.policy}.csv"
    (output_root / summary_name).write_text(json.dumps(summary, indent=2))
    _write_summary_csv(results, output_root / csv_name)
    _plot_objective_panel(
        results,
        outpath_png=output_root / f"objective_panel_{backend_label}_{args.policy}.png",
        outpath_pdf=output_root / f"objective_panel_{backend_label}_{args.policy}.pdf",
    )
    _plot_geometry_atlas(
        results,
        outpath_png=output_root / f"geometry_atlas_{backend_label}_{args.policy}.png",
        outpath_pdf=output_root / f"geometry_atlas_{backend_label}_{args.policy}.pdf",
    )
    print(f"Wrote {output_root}")


if __name__ == "__main__":
    main()

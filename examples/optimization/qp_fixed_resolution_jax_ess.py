#!/usr/bin/env python
# ruff: noqa: E402
"""Quasi-poloidal symmetry optimization with vmec_jax.

This example starts from the QH warm-start input deck and targets a
quasi-poloidal symmetry residual with ``HELICITY_M = 0`` and user-selectable
``HELICITY_N``.  All controls are top-level Python variables, matching the
style of the QA/QH examples.
"""

from pathlib import Path

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


# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp4_QH_warm_start"

VMEC_MPOL = 5
VMEC_NTOR = 5

MAX_MODE = 3
MAX_NFEV = 20
CONTINUATION_NFEV = 0

METHOD = "scipy"
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = None

FTOL = 1e-4
GTOL = 1e-4
XTOL = 1e-4

# Exploratory QP runs are currently more robust with a fixed moderate VMEC
# budget.  Full input-deck accepted solves can trigger an early ``xtol`` stop
# from a poor QP trial step on this QH seed.
INNER_MAX_ITER = 80
INNER_FTOL = 1e-8
TRIAL_MAX_ITER = 80
TRIAL_FTOL = 1e-8

# Quasi-poloidal target: |B| ~ B(n*zeta).  HELICITY_N = +1 gives the same
# short-run objective as -1 for this stellarator-symmetric QH seed; keep -1 as
# the default and change this variable if you want to test the opposite sign.
HELICITY_M = 0
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)

TARGET_ASPECT = 7.0
TARGET_ABS_IOTA_MIN = 0.31
TARGET_IOTA_SIGN = -1  # display/plot branch; objective uses |iota| lower bound.
TARGET_IOTA_DISPLAY = TARGET_IOTA_SIGN * TARGET_ABS_IOTA_MIN

ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 20.0  # lower-bound penalty for |mean iota| >= TARGET_ABS_IOTA_MIN
QS_WEIGHT = 1.0

USE_ESS = True
ALPHA = 2.5  # short sweeps found this robust for the mode-3 QP direct start.
# Start directly from the QH warm-start boundary.  A low-mode QP continuation
# stage can move the seed away from its good aspect and |iota| before the
# higher-mode QP DOFs are available.
USE_MODE_CONTINUATION = False

OUTPUT_DIR = Path(f"results/qp_opt/n{HELICITY_N:+d}/mode{MAX_MODE}/{'ess' if USE_ESS else 'no_ess'}")
SAVE_STAGE_INPUTS = True
SAVE_STAGE_WOUTS = False


print(f"Loading {INPUT_FILE.name} …")
cfg, indata = vj.load_config(str(INPUT_FILE))
indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
cfg = config_from_indata(indata)


def _remove_stale(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _mean_iota(state, *, static, indata, signgs):
    _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    iotas = jnp.asarray(iotas, dtype=jnp.float64)
    return jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else jnp.mean(iotas[1:])


def _save_stage_artifacts(stage_dir: Path, stage_opt, params_initial, params_final, stage_result) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_STAGE_INPUTS:
        stage_opt.save_input(stage_dir / "input.initial", params_initial)
        stage_opt.save_input(stage_dir / "input.final", params_final)
    if SAVE_STAGE_WOUTS:
        stage_opt.save_wout(stage_dir / "wout_initial.nc", params_initial, state=stage_result.get("_state_initial"))
        stage_opt.save_wout(stage_dir / "wout_final.nc", params_final, state=stage_result.get("_state_final"))
    else:
        _remove_stale(stage_dir / "wout_initial.nc")
        _remove_stale(stage_dir / "wout_final.nc")


def _build_stage(max_mode: int):
    stage_static = vj.build_static(cfg)
    stage_boundary = vj.boundary_from_indata(indata, stage_static.modes, apply_m1_constraint=False)
    stage_indata, stage_static, stage_boundary = vj.extend_boundary_for_max_mode(
        indata, stage_static, stage_boundary, max_mode
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
            surfaces=SURFACES,
            helicity_m=HELICITY_M,
            helicity_n=HELICITY_N,
        )

    def stage_residuals_from_state(state):
        aspect = equilibrium_aspect_ratio_from_state(state=state, static=stage_static)
        iota = _mean_iota(state, static=stage_static, indata=stage_indata, signgs=stage_signgs)
        iota_shortfall = jnp.minimum(jnp.abs(iota) - TARGET_ABS_IOTA_MIN, 0.0)
        qs = stage_qs_eval(state)
        return jnp.concatenate(
            [
                jnp.asarray([ASPECT_WEIGHT * (aspect - TARGET_ASPECT)], dtype=jnp.float64),
                jnp.asarray([IOTA_WEIGHT * iota_shortfall], dtype=jnp.float64),
                jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * QS_WEIGHT,
            ]
        )

    stage_residuals_from_state._n_non_qs = 2
    stage_residuals_from_state._qs_total_from_state = (
        lambda state: float(QS_WEIGHT) ** 2 * float(stage_qs_eval(state)["total"])
    )

    stage_opt = vj.FixedBoundaryExactOptimizer(
        stage_static,
        stage_indata,
        stage_boundary,
        stage_specs,
        stage_residuals_from_state,
        boundary_input=stage_boundary_input,
        inner_max_iter=INNER_MAX_ITER,
        inner_ftol=INNER_FTOL,
        trial_max_iter=TRIAL_MAX_ITER,
        trial_ftol=TRIAL_FTOL,
    )
    stage_x_scale = vj.create_x_scale(stage_specs, alpha=ALPHA) if USE_ESS else np.ones(len(stage_specs))

    def iota_fn(state):
        return float(_mean_iota(state, static=stage_static, indata=stage_indata, signgs=stage_signgs))

    return stage_specs, stage_opt, stage_x_scale, iota_fn


stage_results = []
params_stage = None
prev_specs = None
stage_modes = list(range(1, MAX_MODE + 1)) if (USE_MODE_CONTINUATION and MAX_MODE > 1) else [MAX_MODE]

for stage_mode in stage_modes:
    stage_specs, stage_opt, stage_x_scale, iota_fn = _build_stage(stage_mode)
    params0_stage = (
        np.zeros(len(stage_specs), dtype=float)
        if params_stage is None
        else vj.lift_boundary_params(prev_specs, params_stage, stage_specs)
    )
    stage_budget = MAX_NFEV if stage_mode == MAX_MODE else CONTINUATION_NFEV
    if stage_mode == MAX_MODE:
        print(f"Parameter space ({len(stage_specs)} DOFs): {vj.boundary_param_names(stage_specs)}")
        print(f"Helicity: M={HELICITY_M}, N={HELICITY_N}")
        print(f"ESS: {USE_ESS}, alpha={ALPHA}")
        print(f"Aspect ratio (initial):        {stage_opt.aspect_ratio(params0_stage):.4f}")
        print(f"QS objective (initial):        {stage_opt.quasisymmetry_objective(params0_stage):.6f}")
        print(f"Running {METHOD} (max_nfev={stage_budget}, continuation={USE_MODE_CONTINUATION}) …")
    else:
        print(f"Stage {stage_mode} → {stage_mode + 1} continuation seed (budget={stage_budget}) …")

    stage_result = stage_opt.run(
        params0_stage,
        method=METHOD,
        max_nfev=stage_budget,
        ftol=FTOL,
        gtol=GTOL,
        xtol=XTOL,
        x_scale=stage_x_scale,
        verbose=1 if stage_mode == MAX_MODE else 0,
        iota_fn=iota_fn,
        target_iota=TARGET_IOTA_DISPLAY,
        target_aspect=TARGET_ASPECT,
        scipy_tr_solver=SCIPY_TR_SOLVER,
        scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    )
    _save_stage_artifacts(
        OUTPUT_DIR / f"stage_{stage_mode:02d}",
        stage_opt,
        params0_stage,
        stage_result["x"],
        stage_result,
    )
    stage_results.append((stage_mode, stage_specs, stage_opt, params0_stage, stage_result))
    prev_specs = stage_specs
    params_stage = stage_result["x"]


stage_mode, specs, opt, params0, result = stage_results[-1]
hist = result["_history_dump"]

print(f"\nTermination: {result['message']}")
print(f"Aspect ratio (final):          {hist['aspect_final']:.6f}")
print(f"Mean iota (final):             {hist['iota_final']:.6f}  |iota| minimum={TARGET_ABS_IOTA_MIN:.6f}")
print(f"QS objective (final):          {hist['qs_final']:.6e}")
print(f"Total objective (final):       {hist['objective_final']:.6e}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
initial_stage_mode, initial_specs, initial_opt, initial_params0, initial_result = stage_results[0]
initial_opt.save_input(OUTPUT_DIR / "input.initial", initial_params0)
initial_opt.save_wout(OUTPUT_DIR / "wout_initial.nc", initial_params0, state=initial_result.get("_state_initial"))
opt.save_input(OUTPUT_DIR / "input.final", result["x"])
opt.save_wout(OUTPUT_DIR / "wout_final.nc", result["x"], state=result.get("_state_final"))
opt.save_history(OUTPUT_DIR / "history.json", result)

print("\nGenerating plots …")
vj.plot_qh_optimization(
    OUTPUT_DIR / "wout_initial.nc",
    OUTPUT_DIR / "wout_final.nc",
    OUTPUT_DIR / "history.json",
    outdir=OUTPUT_DIR,
)
print("Done.")

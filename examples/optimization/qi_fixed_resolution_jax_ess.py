#!/usr/bin/env python
# ruff: noqa: E402
"""Quasi-isodynamic optimization with vmec_jax.

This example uses the differentiable smooth QI residual in
``vmec_jax.quasi_isodynamic``.  It first runs a same-mode QP preseed from the
QH warm-start input, then refines with the QI objective.  The QI residual
follows the same physical target as the branch/spline diagnostic in
``omnigenity_optimization``: magnetic well widths and normalized well profiles
should be weakly dependent on field-line label.  The implementation here uses
``booz_xform_jax`` so the path is compatible with JAX autodiff.

All user-facing parameters are top-level variables, matching the QA/QH/QP
examples.  Increase ``QI_MBOZ``, ``QI_NBOZ``, and the sampling grids for final
research runs; the defaults are intentionally modest for a first local run.
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
from vmec_jax.quasi_isodynamic import _nearest_half_mesh_indices, quasi_isodynamic_residual_from_state
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


enable_x64(True)


# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp4_QH_warm_start"

VMEC_MPOL = 5
VMEC_NTOR = 5

MAX_MODE = 1
MAX_NFEV = 12
CONTINUATION_NFEV = 6

METHOD = "scipy"
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = None

FTOL = 1e-3
GTOL = 1e-3
XTOL = 1e-3

INNER_MAX_ITER = 80
INNER_FTOL = 1e-8
TRIAL_MAX_ITER = 80
TRIAL_FTOL = 1e-8

SURFACES = np.linspace(0.2, 1.0, 5)
TARGET_ASPECT = 7.0
ASPECT_WEIGHT = 1.0
QI_WEIGHT = 1.0

# Boozer transform and smooth QI residual resolution.
QI_MBOZ = 6
QI_NBOZ = 6
QI_NPHI = 41
QI_NALPHA = 13
QI_N_BOUNCE = 11
QI_SOFTNESS = 2.0e-2
QI_PROFILE_WEIGHT = 1.5

USE_ESS = True
ALPHA = 2.5
USE_MODE_CONTINUATION = True
USE_QP_PRESEED = True
QP_PRESEED_MAX_NFEV = 20
QP_PRESEED_CONTINUATION_NFEV = 8
QP_HELICITY_M = 0
QP_HELICITY_N = -1
QP_SURFACES = np.arange(0.0, 1.01, 0.1)
QP_TARGET_ABS_IOTA_MIN = 0.31
QP_IOTA_WEIGHT = 20.0

OUTPUT_DIR = Path(f"results/qi_opt/mode{MAX_MODE}/{'ess' if USE_ESS else 'no_ess'}")
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


def _mean_iota(state, *, static, indata, signgs):
    _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    iotas = jnp.asarray(iotas, dtype=jnp.float64)
    return jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else jnp.mean(iotas[1:])


def _build_stage(max_mode: int, *, objective_kind: str):
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

    if objective_kind == "qi":
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
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            asym=bool(stage_static.cfg.lasym),
            xm=np.asarray(main_modes.m, dtype=int),
            xn=np.asarray(main_modes.n * int(stage_static.cfg.nfp), dtype=int),
            xm_nyq=np.asarray(nyq_modes.m, dtype=int),
            xn_nyq=np.asarray(nyq_modes.n * int(stage_static.cfg.nfp), dtype=int),
        )
        qi_surface_indices = _nearest_half_mesh_indices(
            SURFACES,
            n_half=max(int(np.asarray(stage_static.s).shape[0]) - 1, 1),
        )

        def stage_field_eval(state):
            return quasi_isodynamic_residual_from_state(
                state=state,
                static=stage_static,
                indata=stage_indata,
                signgs=stage_signgs,
                flux_local=stage_flux,
                prof_local={"pressure": stage_pressure},
                pressure_local=stage_pressure,
                surfaces=SURFACES,
                mboz=QI_MBOZ,
                nboz=QI_NBOZ,
                nphi=QI_NPHI,
                nalpha=QI_NALPHA,
                n_bounce=QI_N_BOUNCE,
                softness=QI_SOFTNESS,
                profile_weight=QI_PROFILE_WEIGHT,
                jit_booz=False,
                booz_constants=qi_booz_constants,
                booz_grids=qi_booz_grids,
                surface_indices=qi_surface_indices,
            )

    else:

        def stage_field_eval(state):
            return quasisymmetry_ratio_residual_from_state(
                state=state,
                static=stage_static,
                indata=stage_indata,
                signgs=stage_signgs,
                flux_local=stage_flux,
                prof_local={"pressure": stage_pressure},
                pressure_local=stage_pressure,
                surfaces=QP_SURFACES,
                helicity_m=QP_HELICITY_M,
                helicity_n=QP_HELICITY_N,
            )

    def stage_residuals_from_state(state):
        aspect = equilibrium_aspect_ratio_from_state(state=state, static=stage_static)
        field = stage_field_eval(state)
        parts = [jnp.asarray([ASPECT_WEIGHT * (aspect - TARGET_ASPECT)], dtype=jnp.float64)]
        if objective_kind == "qp":
            iota = _mean_iota(state, static=stage_static, indata=stage_indata, signgs=stage_signgs)
            iota_shortfall = jnp.minimum(jnp.abs(iota) - QP_TARGET_ABS_IOTA_MIN, 0.0)
            parts.append(jnp.asarray([QP_IOTA_WEIGHT * iota_shortfall], dtype=jnp.float64))
        parts.append(jnp.asarray(field["residuals1d"], dtype=jnp.float64) * QI_WEIGHT)
        return jnp.concatenate(parts)

    stage_residuals_from_state._n_non_qs = 2 if objective_kind == "qp" else 1
    stage_residuals_from_state._qs_total_from_state = (
        lambda state: float(QI_WEIGHT) ** 2 * float(stage_field_eval(state)["total"])
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

if USE_QP_PRESEED:
    print("Running QP preseed before QI refinement …")
    for stage_mode in stage_modes:
        stage_specs, stage_opt, stage_x_scale, iota_fn = _build_stage(stage_mode, objective_kind="qp")
        params0_stage = (
            np.zeros(len(stage_specs), dtype=float)
            if params_stage is None
            else vj.lift_boundary_params(prev_specs, params_stage, stage_specs)
        )
        stage_budget = QP_PRESEED_MAX_NFEV if stage_mode == MAX_MODE else QP_PRESEED_CONTINUATION_NFEV
        stage_result = stage_opt.run(
            params0_stage,
            method=METHOD,
            max_nfev=stage_budget,
            ftol=FTOL,
            gtol=GTOL,
            xtol=XTOL,
            x_scale=stage_x_scale,
            verbose=0,
            iota_fn=iota_fn,
            target_iota=-QP_TARGET_ABS_IOTA_MIN,
            target_aspect=TARGET_ASPECT,
            scipy_tr_solver=SCIPY_TR_SOLVER,
            scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
        )
        _save_stage_artifacts(
            OUTPUT_DIR / f"stage_qp_{stage_mode:02d}",
            stage_opt,
            params0_stage,
            stage_result["x"],
            stage_result,
        )
        stage_results.append((f"qp{stage_mode}", stage_mode, stage_specs, stage_opt, params0_stage, stage_result))
        prev_specs = stage_specs
        params_stage = stage_result["x"]

for stage_mode in [MAX_MODE] if USE_QP_PRESEED else stage_modes:
    stage_specs, stage_opt, stage_x_scale, _iota_fn = _build_stage(stage_mode, objective_kind="qi")
    params0_stage = (
        np.zeros(len(stage_specs), dtype=float)
        if params_stage is None
        else vj.lift_boundary_params(prev_specs, params_stage, stage_specs)
    )
    stage_budget = MAX_NFEV if stage_mode == MAX_MODE else CONTINUATION_NFEV
    if stage_mode == MAX_MODE:
        print(f"Parameter space ({len(stage_specs)} DOFs): {vj.boundary_param_names(stage_specs)}")
        print(f"ESS: {USE_ESS}, alpha={ALPHA}")
        print(f"Aspect ratio (initial):        {stage_opt.aspect_ratio(params0_stage):.4f}")
        print(f"QI objective (initial):        {stage_opt.quasisymmetry_objective(params0_stage):.6e}")
        print(
            f"Running {METHOD} (max_nfev={stage_budget}, continuation={USE_MODE_CONTINUATION}, "
            f"QP preseed={USE_QP_PRESEED}) …"
        )
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
        target_aspect=TARGET_ASPECT,
        scipy_tr_solver=SCIPY_TR_SOLVER,
        scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    )
    _save_stage_artifacts(
        OUTPUT_DIR / f"stage_qi_{stage_mode:02d}",
        stage_opt,
        params0_stage,
        stage_result["x"],
        stage_result,
    )
    stage_results.append((f"qi{stage_mode}", stage_mode, stage_specs, stage_opt, params0_stage, stage_result))
    prev_specs = stage_specs
    params_stage = stage_result["x"]


_label, stage_mode, specs, opt, params0, result = stage_results[-1]
hist = result["_history_dump"]

print(f"\nTermination: {result['message']}")
print(f"Aspect ratio (final):          {hist['aspect_final']:.6f}")
print(f"QI objective (final):          {hist['qs_final']:.6e}")
print(f"Total objective (final):       {hist['objective_final']:.6e}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_initial_label, initial_stage_mode, initial_specs, initial_opt, initial_params0, initial_result = stage_results[0]
initial_opt.save_input(OUTPUT_DIR / "input.initial", initial_params0)
initial_opt.save_wout(OUTPUT_DIR / "wout_initial.nc", initial_params0, state=initial_result.get("_state_initial"))
if USE_QP_PRESEED:
    qp_label, qp_stage_mode, qp_specs, qp_opt, qp_params0, qp_result = stage_results[-2]
    del qp_label, qp_stage_mode, qp_specs, qp_params0
    qp_opt.save_input(OUTPUT_DIR / "input.qp_seed", qp_result["x"])
    qp_opt.save_wout(OUTPUT_DIR / "wout_qp_seed.nc", qp_result["x"], state=qp_result.get("_state_final"))
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

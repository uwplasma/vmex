#!/usr/bin/env python
"""Quasi-helical symmetry optimization with vmec_jax.

This is written as a linear SIMSOPT-style example: edit parameters at the top,
build the objective list, construct the VMEC/JAX objects, create an optimizer,
run it, and save the result.  There is no argparse, no ``main()`` wrapper, and
no high-level run configuration object hiding the workflow.
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

try:
    from fixed_boundary_qs_common import (
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        StageContext,
        abs_mean_iota_floor_objective,
        aspect_objective,
        combine_qs_stage_histories,
        mean_iota,
        objectives_track_iota,
        print_qs_final_summary,
        print_qs_problem_summary,
        qs_stage_budget,
        qs_stage_modes,
        quasisymmetry_objective,
        save_qs_final_outputs,
        save_qs_stage_artifacts,
    )
except ModuleNotFoundError:
    from examples.optimization.fixed_boundary_qs_common import (
        ObjectiveTerm,  # noqa: F401 - shown in the commented custom-objective example.
        StageContext,
        abs_mean_iota_floor_objective,
        aspect_objective,
        combine_qs_stage_histories,
        mean_iota,
        objectives_track_iota,
        print_qs_final_summary,
        print_qs_problem_summary,
        qs_stage_budget,
        qs_stage_modes,
        quasisymmetry_objective,
        save_qs_final_outputs,
        save_qs_stage_artifacts,
    )


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# User parameters
INPUT_FILE = DATA_DIR / "input.nfp4_QH_warm_start"
OUTPUT_DIR = Path("results/qh_opt")

MAX_MODE = 1
VMEC_MPOL = max(6, MAX_MODE + 2)
VMEC_NTOR = VMEC_MPOL

MAX_NFEV = 30
CONTINUATION_NFEV = 30
USE_MODE_CONTINUATION = True

METHOD = "scipy"  # "scipy", "gauss_newton", "lbfgs_adjoint", or "scipy_matrix_free"
SCIPY_TR_SOLVER = "lsmr"
SCIPY_LSMR_MAXITER = None
FTOL = 1.0e-4
GTOL = 1.0e-4
XTOL = 1.0e-4

INNER_MAX_ITER = 120
INNER_FTOL = 1.0e-9
TRIAL_MAX_ITER = 120
TRIAL_FTOL = 1.0e-9
SOLVER_DEVICE = None  # set to "cpu" or "gpu" to force one backend

HELICITY_M = 1
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)
TARGET_ASPECT = 7.0
TARGET_IOTA = None
TARGET_ABS_IOTA_MIN = 0.41
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 200.0
QS_WEIGHT = 1.0

USE_ESS = False
ALPHA = 1.2
LABEL = f"QH opt (max_mode={MAX_MODE}, {'ESS' if USE_ESS else 'no ESS'})"

SAVE_STAGE_INPUTS = True
SAVE_STAGE_WOUTS = False
SAVE_RERUN_WOUTS = False
PLOT = True


# Objective function
# Add an objective by appending another ObjectiveTerm.  The callback receives
# (ctx, state) and returns a scalar or vector; vmec_jax minimizes
# weight * (value - target) in least-squares form.
OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    abs_mean_iota_floor_objective(TARGET_ABS_IOTA_MIN, IOTA_WEIGHT),
    quasisymmetry_objective(
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
        surfaces=SURFACES,
        weight=QS_WEIGHT,
    ),
    # Optional LgradB penalty:
    # import lgradb_objective from fixed_boundary_qs_common and append
    # lgradb_objective(threshold=0.30, weight=0.1)
    # ObjectiveTerm("major_radius", lambda ctx, state: state.rmncc[0, 0], target=1.0, weight=0.1),
]


# Problem setup
print(f"Loading {INPUT_FILE.name} ...")
cfg, indata = vj.load_config(str(INPUT_FILE))
indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
cfg = config_from_indata(indata)
stage_modes = qs_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
)


# Optimization
stage_records = []
params_stage = None
prev_specs = None

for stage_index, stage_mode in enumerate(stage_modes, start=1):
    # Build the fixed-boundary VMEC problem for this mode-continuation stage.
    static = vj.build_static(cfg)
    boundary = vj.boundary_from_indata(indata, static.modes, apply_m1_constraint=False)
    stage_indata, static, boundary = vj.extend_boundary_for_max_mode(
        indata,
        static,
        boundary,
        stage_mode,
    )
    boundary_input = vj.boundary_input_from_indata(stage_indata, static.modes)
    specs = vj.boundary_param_specs(
        boundary_input,
        static.modes,
        max_mode=stage_mode,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )

    guess = initial_guess_from_boundary(static, boundary, stage_indata, vmec_project=True)
    geom = eval_geom(guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    flux = vj.flux_profiles_from_indata(stage_indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))
    ctx = StageContext(
        static=static,
        indata=stage_indata,
        boundary_input=boundary_input,
        specs=specs,
        signgs=signgs,
        flux=flux,
        pressure=pressure,
    )

    def residuals_from_state(state, *, ctx=ctx):
        return jnp.concatenate([term.residual(ctx, state) for term in OBJECTIVES])

    qs_totals = tuple(term.total for term in OBJECTIVES if term.total is not None)
    residuals_from_state._n_non_qs = 0
    residuals_from_state._qs_total_from_state = (
        lambda state, ctx=ctx, qs_totals=qs_totals: float(
            sum(float(total(ctx, state)) for total in qs_totals)
        )
        if qs_totals
        else lambda _state: 0.0
    )

    optimizer = vj.FixedBoundaryExactOptimizer(
        static,
        stage_indata,
        boundary,
        specs,
        residuals_from_state,
        boundary_input=boundary_input,
        inner_max_iter=INNER_MAX_ITER,
        inner_ftol=INNER_FTOL,
        trial_max_iter=TRIAL_MAX_ITER,
        trial_ftol=TRIAL_FTOL,
        solver_device=SOLVER_DEVICE,
    )

    x_scale = vj.create_x_scale(specs, alpha=ALPHA) if USE_ESS else np.ones(len(specs), dtype=float)
    if params_stage is None:
        params0 = np.zeros(len(specs), dtype=float)
    else:
        params0 = np.asarray(vj.lift_boundary_params(prev_specs, params_stage, specs), dtype=float)
    nfev = qs_stage_budget(
        stage_mode=stage_mode,
        max_mode=MAX_MODE,
        max_nfev=MAX_NFEV,
        continuation_nfev=CONTINUATION_NFEV,
    )
    iota_fn = (lambda state, ctx=ctx: float(mean_iota(ctx, state))) if objectives_track_iota(OBJECTIVES) else None

    if stage_mode == MAX_MODE:
        print_qs_problem_summary(
            method=METHOD,
            max_nfev=MAX_NFEV,
            use_mode_continuation=USE_MODE_CONTINUATION,
            use_ess=USE_ESS,
            ess_alpha=ALPHA,
            objectives=OBJECTIVES,
            specs=specs,
            x_scale=np.asarray(x_scale, dtype=float),
            optimizer=optimizer,
            params0=params0,
        )
    else:
        print(f"Stage {stage_mode} -> {stage_mode + 1} continuation seed (budget={nfev}) ...")

    result = optimizer.run(
        params0,
        method=METHOD,
        max_nfev=nfev,
        ftol=FTOL,
        gtol=GTOL,
        xtol=XTOL,
        x_scale=x_scale,
        verbose=1 if stage_mode == MAX_MODE else 0,
        iota_fn=iota_fn,
        target_iota=TARGET_IOTA,
        target_aspect=TARGET_ASPECT,
        scipy_tr_solver=SCIPY_TR_SOLVER,
        scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    )
    save_qs_stage_artifacts(
        stage_dir=OUTPUT_DIR / f"stage_{stage_index:02d}_mode{stage_mode:02d}",
        optimizer=optimizer,
        params_initial=params0,
        params_final=result["x"],
        result=result,
        save_inputs=SAVE_STAGE_INPUTS,
        save_wouts=SAVE_STAGE_WOUTS,
        save_rerun_wouts=SAVE_RERUN_WOUTS,
    )
    stage_records.append((stage_mode, optimizer, params0, result))
    prev_specs = specs
    params_stage = result["x"]


# Output
final_optimizer = stage_records[-1][1]
final_result = stage_records[-1][3]
combined_history = combine_qs_stage_histories(
    label=LABEL,
    max_mode=MAX_MODE,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    stage_modes=stage_modes,
    stage_records=stage_records,
)
if combined_history is not None:
    final_result["_history_dump"] = combined_history

print_qs_final_summary(final_result, target_iota=TARGET_IOTA, iota_abs_min=TARGET_ABS_IOTA_MIN)
save_qs_final_outputs(
    output_dir=OUTPUT_DIR,
    stage_records=stage_records,
    final_optimizer=final_optimizer,
    final_result=final_result,
    label=LABEL,
    target_aspect=TARGET_ASPECT,
    target_iota=TARGET_IOTA,
    iota_abs_min=TARGET_ABS_IOTA_MIN,
    plot=PLOT,
    save_rerun_wouts=SAVE_RERUN_WOUTS,
)

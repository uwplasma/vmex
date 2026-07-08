#!/usr/bin/env python
"""Finite-beta stage-one QI fixed-boundary optimization with vmec_jax.

This script is intentionally linear, like the SIMSOPT stage-one examples:
choose top-level parameters, load the VMEC input, build the finite-beta and QI
objective directly, instantiate ``FixedBoundaryExactOptimizer``, run, and save
outputs.  There is no argparse, no ``main()`` wrapper, and no config object.
"""

from pathlib import Path

import numpy as np
from booz_xform_jax import prepare_booz_xform_constants

import vmec_jax as vj
from vmec_jax._compat import enable_x64, jnp
from vmec_jax.config import config_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.modes import nyquist_mode_table_from_grid, vmec_mode_table
from vmec_jax.optimization import rebuild_indata_with_resolution
from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_state
from vmec_jax.quasi_isodynamic.objectives import _nearest_half_mesh_indices

try:
    from finite_beta_stage1_common import (
        finite_beta_stage1_result,
        finite_beta_stage_budget,
        finite_beta_stage_modes,
        apply_finite_beta_pressure_profile,
        mean_abs_iota,
        pressure_profile,
        print_final_summary,
        save_final_outputs,
        save_stage_artifacts,
    )
except ModuleNotFoundError:
    from examples.optimization.finite_beta_stage1_common import (
        finite_beta_stage1_result,
        finite_beta_stage_budget,
        finite_beta_stage_modes,
        apply_finite_beta_pressure_profile,
        mean_abs_iota,
        pressure_profile,
        print_final_summary,
        save_final_outputs,
        save_stage_artifacts,
    )


enable_x64(True)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# User parameters
INPUT_FILE = DATA_DIR / "input.nfp4_QI_finite_beta"
OUTPUT_DIR = Path("results/qi_finite_beta")

VMEC_MPOL = 5
VMEC_NTOR = 5
MAX_MODE = 1

MAX_NFEV = 16
CONTINUATION_NFEV = 16
USE_MODE_CONTINUATION = True

METHOD = "scipy"  # Try also "auto", "auto_scalar", "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
SCIPY_TR_SOLVER = "lsmr"  # For METHOD="scipy": "lsmr" is memory-light; "exact" is dense.
SCIPY_LSMR_MAXITER = None  # For scipy_matrix_free, None uses vmec_jax's bounded cap of 4.
FTOL = 1.0e-3  # Relative cost-reduction tolerance for the outer optimizer.
GTOL = 1.0e-3  # Gradient optimality tolerance for the outer optimizer.
XTOL = 1.0e-3  # Step-size tolerance for the outer optimizer.

INNER_MAX_ITER = 0  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
INNER_FTOL = 0.0  # Accepted-point VMEC tolerance; 0 uses FTOL from the input deck.
TRIAL_MAX_ITER = 300  # Trial-point VMEC iterations; 0 follows the accepted/input budget.
TRIAL_FTOL = 1.0e-10  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.
SOLVER_DEVICE = None  # None uses JAX default; set "cpu" or "gpu" to force one backend.

SURFACES = tuple(np.linspace(0.0, 1.0, 10, endpoint=True))
TARGET_ASPECT = 6.0
MIN_IOTA = 1.04
MIN_AVERAGE_IOTA = 1.06
MAX_IOTA = 1.9
TARGET_VOLAVGB = 5.86461221551616
BETA_PERCENT = 2.5
TARGET_BETA = BETA_PERCENT / 100.0
STANDARD_PROFILES = vj.standard_finite_beta_profiles(BETA_PERCENT)

ASPECT_WEIGHT = 1.0e3
IOTA_WEIGHT = 1.0e5
MAX_IOTA_WEIGHT = 1.0e8
VOLAVGB_WEIGHT = 1.0e3
BETA_WEIGHT = 1.0e1
FIELD_WEIGHT = 2.0e5
BOOTSTRAP_WEIGHT = 0.0  # Set >0 to add the Redl bootstrap-current mismatch.
BOOTSTRAP_SURFACES = (0.25, 0.50, 0.75)
NE_COEFFS = vj.profile_to_power_series_coeffs(STANDARD_PROFILES.ne).tolist()  # m^-3, polynomial in s.
TE_COEFFS = vj.profile_to_power_series_coeffs(STANDARD_PROFILES.Te).tolist()  # eV; Ti defaults to Te.

# Boozer/QI residual resolution. These defaults are diagnostic-friendly for a
# first run; increase them for final research-quality QI refinements.
QI_MBOZ = 10
QI_NBOZ = 10
QI_NPHI = 32
QI_NALPHA = 8
QI_N_BOUNCE = 12
QI_SOFTNESS = 30.0
QI_PROFILE_WEIGHT = 0.15
QI_PHIMIN = 0.0  # Set to np.pi / nfp if the reference QI well starts there.

USE_ESS = True
ALPHA = 2.5
SAVE_STAGE_INPUTS = True
SAVE_STAGE_WOUTS = False
PLOT = True


# Problem setup
print(f"Loading {INPUT_FILE.name} ...")
base_cfg, indata = vj.load_config(str(INPUT_FILE))
indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
indata = apply_finite_beta_pressure_profile(indata, beta_percent=BETA_PERCENT)
base_cfg = config_from_indata(indata)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
stage_modes = finite_beta_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
)


# Optimization
stage_records = []
params_stage = None
prev_specs = None

for stage_mode in stage_modes:
    static = vj.build_static(base_cfg)
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
    pressure = pressure_profile(stage_indata, static)
    global_targets = vj.FiniteBetaTargets(
        aspect_ratio=TARGET_ASPECT,
        min_iota=MIN_IOTA,
        min_average_iota=MIN_AVERAGE_IOTA,
        max_iota=MAX_IOTA,
        volavgB=TARGET_VOLAVGB,
        beta_total=TARGET_BETA,
        aspect_weight=ASPECT_WEIGHT,
        iota_weight=IOTA_WEIGHT,
        max_iota_weight=MAX_IOTA_WEIGHT,
        volavgB_weight=VOLAVGB_WEIGHT,
        beta_weight=BETA_WEIGHT,
    )

    main_modes = vmec_mode_table(int(static.cfg.mpol), int(static.cfg.ntor))
    nyq_modes = nyquist_mode_table_from_grid(
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
    )
    booz_constants, booz_grids = prepare_booz_xform_constants(
        nfp=int(static.cfg.nfp),
        mboz=QI_MBOZ,
        nboz=QI_NBOZ,
        asym=bool(static.cfg.lasym),
        xm=np.asarray(main_modes.m, dtype=int),
        xn=np.asarray(main_modes.n * int(static.cfg.nfp), dtype=int),
        xm_nyq=np.asarray(nyq_modes.m, dtype=int),
        xn_nyq=np.asarray(nyq_modes.n * int(static.cfg.nfp), dtype=int),
    )
    surface_indices = _nearest_half_mesh_indices(
        SURFACES,
        n_half=max(int(np.asarray(static.s).shape[0]) - 1, 1),
    )

    def field_residual(
        state,
        *,
        static=static,
        stage_indata=stage_indata,
        signgs=signgs,
        flux=flux,
        pressure=pressure,
        booz_constants=booz_constants,
        booz_grids=booz_grids,
        surface_indices=surface_indices,
    ):
        return quasi_isodynamic_residual_from_state(
            state=state,
            static=static,
            indata=stage_indata,
            signgs=signgs,
            flux_local=flux,
            prof_local={"pressure": pressure},
            pressure_local=pressure,
            surfaces=SURFACES,
            mboz=QI_MBOZ,
            nboz=QI_NBOZ,
            nphi=QI_NPHI,
            nalpha=QI_NALPHA,
            n_bounce=QI_N_BOUNCE,
            softness=QI_SOFTNESS,
            profile_weight=QI_PROFILE_WEIGHT,
            phimin=QI_PHIMIN,
            jit_booz=False,
            booz_constants=booz_constants,
            booz_grids=booz_grids,
            surface_indices=surface_indices,
        )

    def residuals_from_state(
        state,
        *,
        static=static,
        stage_indata=stage_indata,
        signgs=signgs,
        global_targets=global_targets,
        field_residual=field_residual,
    ):
        global_res = vj.finite_beta_global_residuals_from_state(
            state=state,
            static=static,
            indata=stage_indata,
            signgs=signgs,
            targets=global_targets,
        )
        field = field_residual(state)
        residual_blocks = [
            global_res,
            jnp.asarray(field["residuals1d"], dtype=jnp.float64) * FIELD_WEIGHT,
        ]
        if BOOTSTRAP_WEIGHT > 0.0:
            redl = vj.redl_bootstrap_mismatch_from_state(
                state=state,
                static=static,
                indata=stage_indata,
                signgs=signgs,
                helicity_n=0,
                ne_coeffs=NE_COEFFS,
                Te_coeffs=TE_COEFFS,
                surfaces=BOOTSTRAP_SURFACES,
            )
            residual_blocks.append(jnp.asarray(redl["residuals1d"], dtype=jnp.float64) * BOOTSTRAP_WEIGHT)
        return jnp.concatenate(
            residual_blocks
        )

    residuals_from_state._n_non_qs = 6 + (len(BOOTSTRAP_SURFACES) if BOOTSTRAP_WEIGHT > 0.0 else 0)
    residuals_from_state._qs_total_from_state = (
        lambda state, field_residual=field_residual: FIELD_WEIGHT**2 * float(field_residual(state)["total"])
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
    params0 = (
        np.zeros(len(specs), dtype=float)
        if params_stage is None
        else np.asarray(vj.lift_boundary_params(prev_specs, params_stage, specs), dtype=float)
    )
    nfev = finite_beta_stage_budget(
        stage_mode=stage_mode,
        max_mode=MAX_MODE,
        max_nfev=MAX_NFEV,
        continuation_nfev=CONTINUATION_NFEV,
    )

    def iota_fn(state, *, static=static, stage_indata=stage_indata, signgs=signgs):
        return float(mean_abs_iota(state, static=static, indata=stage_indata, signgs=signgs))

    print(f"Running finite-beta QI stage mode={stage_mode}, nfev={nfev}")
    result = optimizer.run(
        params0,
        method=METHOD,
        max_nfev=nfev,
        ftol=FTOL,
        gtol=GTOL,
        xtol=XTOL,
        x_scale=x_scale,
        verbose=2,
        iota_fn=iota_fn,
        target_iota=MIN_AVERAGE_IOTA,
        target_aspect=TARGET_ASPECT,
        scipy_tr_solver=SCIPY_TR_SOLVER,
        scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    )
    save_stage_artifacts(
        stage_dir=OUTPUT_DIR / f"stage_mode{stage_mode}",
        optimizer=optimizer,
        params_initial=params0,
        params_final=result["x"],
        result=result,
        save_inputs=SAVE_STAGE_INPUTS,
        save_wouts=SAVE_STAGE_WOUTS,
    )
    stage_records.append((stage_mode, optimizer, params0, result))
    params_stage = result["x"]
    prev_specs = specs


final_optimizer = stage_records[-1][1]
final_result = stage_records[-1][3]
stage1_result = finite_beta_stage1_result(stage_records)
stage_summaries = stage1_result.stage_summaries
final_summary = stage1_result.final_summary
save_final_outputs(
    output_dir=OUTPUT_DIR,
    stage_records=stage_records,
    final_optimizer=final_optimizer,
    final_result=final_result,
)
print_final_summary(final_result)

if PLOT:
    try:
        plot_paths = {
            "boundary_comparison": vj.plot_3d_boundary_comparison(
                OUTPUT_DIR / "wout_initial.nc",
                OUTPUT_DIR / "wout_final.nc",
                outdir=OUTPUT_DIR,
            ),
            "bmag_contours": vj.plot_bmag_contours(
                OUTPUT_DIR / "wout_initial.nc",
                OUTPUT_DIR / "wout_final.nc",
                outdir=OUTPUT_DIR,
            ),
            "objective_history": vj.plot_objective_history(
                OUTPUT_DIR / "history.json",
                outdir=OUTPUT_DIR,
            ),
        }
        print("Plot files:")
        for path in plot_paths.values():
            print(f"  {path}")
    except Exception as exc:
        (OUTPUT_DIR / "plotting_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        print(f"Plotting failed: {type(exc).__name__}: {exc}")

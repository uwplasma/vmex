#!/usr/bin/env python
# ruff: noqa: E402
"""Quasi-helical symmetry optimization with vmec_jax.

This script is intentionally written in the same teaching style as SIMSOPT's
``QH_fixed_resolution.py``:

1. choose the VMEC input and resolution directly in Python,
2. choose the boundary parameter space directly,
3. construct the objective blocks directly in the script,
4. choose the optimizer directly,
5. run the solve, save outputs, and plot the results.

No finite differences are used. The Jacobian comes from vmec_jax's exact
discrete-adjoint path through :class:`vmec_jax.FixedBoundaryExactOptimizer`.
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
from vmec_jax.wout import equilibrium_aspect_ratio_from_state

# ── 0. Floating-point precision ───────────────────────────────────────────────
enable_x64(True)

# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp4_QH_warm_start"

# Choose the VMEC solver resolution directly in the script.  The QH examples
# and SIMSOPT comparisons in this repo use mpol=ntor=5 for consistent scaling.
VMEC_MPOL = 5
VMEC_NTOR = 5

# Boundary parameterization.
MAX_MODE = 2
MAX_NFEV = 15
CONTINUATION_NFEV = 15

# Outer optimizer: "gauss_newton" or "scipy".
#
# For standalone QH continuation, the exact SciPy trust-region path is the
# robust default.  The cheaper Gauss-Newton branch can still be useful for
# short experiments, but the seeded mode-2/mode-3 continuation path is more
# reliable with SciPy on the current exact callbacks.
METHOD = "scipy"

FTOL = 1e-3
GTOL = 1e-3
XTOL = 1e-3

HELICITY_M = 1
HELICITY_N = -1
SURFACES = np.arange(0.0, 1.01, 0.1)

TARGET_ASPECT = 7.0
ASPECT_WEIGHT = 1.0
QS_WEIGHT = 1.0
OBJECTIVE_TUPLES = [
    ("aspect", TARGET_ASPECT, ASPECT_WEIGHT),
    ("qs", 0.0, QS_WEIGHT),
]

# Optional exponential spectral scaling and staged continuation.  For QH,
# continuation is mainly useful once MAX_MODE > 2.
USE_ESS = False
ALPHA = 0.8
USE_MODE_CONTINUATION = True

OUTPUT_DIR = Path("results/qh_opt")

print(f"Loading {INPUT_FILE.name} …")
cfg, indata = vj.load_config(str(INPUT_FILE))
indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
cfg = config_from_indata(indata)
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
        parts = []
        for name, target, weight in OBJECTIVE_TUPLES:
            if name == "aspect":
                aspect = equilibrium_aspect_ratio_from_state(state=state, static=stage_static)
                parts.append(jnp.asarray([float(weight) * (aspect - float(target))], dtype=jnp.float64))
            elif name == "qs":
                qs = stage_qs_eval(state)
                parts.append(jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * float(weight))
            else:
                raise ValueError(f"Unknown objective block '{name}'")
        return jnp.concatenate(parts)

    stage_residuals_from_state._n_non_qs = 1
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
    )
    stage_x_scale = vj.create_x_scale(stage_specs, alpha=ALPHA) if USE_ESS else np.ones(len(stage_specs))
    return stage_indata, stage_static, stage_boundary_input, stage_specs, stage_opt, stage_x_scale


stage_results = []
params_stage = None
prev_specs = None
stage_modes = list(range(1, MAX_MODE + 1)) if (USE_MODE_CONTINUATION and MAX_MODE > 1) else [MAX_MODE]

for stage_mode in stage_modes:
    stage_indata, stage_static, stage_boundary_input, stage_specs, stage_opt, stage_x_scale = _build_stage(stage_mode)
    params0_stage = (
        np.zeros(len(stage_specs))
        if params_stage is None
        else vj.lift_boundary_params(prev_specs, params_stage, stage_specs)
    )
    stage_budget = MAX_NFEV if stage_mode == MAX_MODE else CONTINUATION_NFEV

    if stage_mode == MAX_MODE:
        print(f"Parameter space ({len(stage_specs)} DOFs): {vj.boundary_param_names(stage_specs)}")
        if USE_ESS:
            print(f"ESS scales (alpha={ALPHA}): min={stage_x_scale.min():.3f}  max={stage_x_scale.max():.3f}")
        else:
            print("ESS disabled — uniform scales.")
        print(f"\nAspect ratio (initial):        {stage_opt.aspect_ratio(params0_stage):.4f}")
        print(f"QS objective (initial):        {stage_opt.quasisymmetry_objective(params0_stage):.6f}")
        print(f"\nRunning {METHOD} (max_nfev={MAX_NFEV}, continuation={USE_MODE_CONTINUATION}) …")
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
    )
    stage_results.append((stage_mode, stage_specs, stage_opt, params0_stage, stage_result))
    prev_specs = stage_specs
    params_stage = stage_result["x"]

stage_mode, specs, opt, params0, result = stage_results[-1]

combined_history = None
if USE_MODE_CONTINUATION and len(stage_results) > 1:
    combined_entries = []
    wall_offset = 0.0
    nfev_total = 0
    njev_total = 0
    for idx, (_mode, _specs, _opt, _params0, _result) in enumerate(stage_results):
        stage_hist = _result["_history_dump"]
        entries = stage_hist["history"] if idx == 0 else stage_hist["history"][1:]
        for entry in entries:
            entry_copy = dict(entry)
            entry_copy["wall_time_s"] = float(entry_copy["wall_time_s"]) + wall_offset
            combined_entries.append(entry_copy)
        wall_offset = combined_entries[-1]["wall_time_s"]
        nfev_total += int(stage_hist["nfev"])
        njev_total += int(stage_hist["njev"])
    combined_history = {
        "label": "Optimisation",
        "max_nfev": int(sum(CONTINUATION_NFEV if m != MAX_MODE else MAX_NFEV for m in stage_modes)),
        "ftol": FTOL,
        "gtol": GTOL,
        "xtol": XTOL,
        "total_wall_time_s": float(wall_offset),
        "nfev": int(nfev_total),
        "njev": int(njev_total),
        "success": bool(result["_history_dump"]["success"]),
        "message": str(result["_history_dump"]["message"]),
        "objective_initial": float(stage_results[0][4]["_history_dump"]["objective_initial"]),
        "objective_final": float(result["_history_dump"]["objective_final"]),
        "qs_initial": float(stage_results[0][4]["_history_dump"]["qs_initial"]),
        "qs_final": float(result["_history_dump"]["qs_final"]),
        "aspect_initial": float(stage_results[0][4]["_history_dump"]["aspect_initial"]),
        "aspect_final": float(result["_history_dump"]["aspect_final"]),
        "history": combined_entries,
        "target_aspect": TARGET_ASPECT,
    }

print(f"\nTermination: {result['message']}")
print(f"Aspect ratio (final):          {opt.aspect_ratio(result['x']):.4f}")
print(f"QS objective (final):          {opt.quasisymmetry_objective(result['x']):.6f}")
print(
    f"Objective reduction:           "
    f"{100*(1 - result['objective']/result['_history_dump']['objective_initial']):.1f}%"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
opt.save_wout(OUTPUT_DIR / "wout_initial.nc", params0)
opt.save_wout(OUTPUT_DIR / "wout_final.nc", result["x"])
if combined_history is not None:
    result["_history_dump"] = combined_history
opt.save_history(OUTPUT_DIR / "history.json", result)

print("\nGenerating plots …")
vj.plot_qh_optimization(
    OUTPUT_DIR / "wout_initial.nc",
    OUTPUT_DIR / "wout_final.nc",
    OUTPUT_DIR / "history.json",
    outdir=OUTPUT_DIR,
)
print("Done.")

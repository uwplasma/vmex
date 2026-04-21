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
import os

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

# Outer optimizer: "gauss_newton" or "scipy".
METHOD = "gauss_newton"

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

OUTPUT_DIR = Path("results/qh_opt")

# Optional environment overrides for benchmarking without editing the file.
MAX_MODE = int(os.environ.get("VMEC_JAX_QH_MAX_MODE", str(MAX_MODE)))
MAX_NFEV = int(os.environ.get("VMEC_JAX_QH_MAX_NFEV", str(MAX_NFEV)))
METHOD = os.environ.get("VMEC_JAX_QH_METHOD", METHOD)
OUTPUT_DIR = Path(os.environ.get("VMEC_JAX_QH_OUTPUT_DIR", str(OUTPUT_DIR)))

print(f"Loading {INPUT_FILE.name} …")
cfg, indata = vj.load_config(str(INPUT_FILE))
indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
cfg = config_from_indata(indata)
static = vj.build_static(cfg)
boundary = vj.boundary_from_indata(indata, static.modes, apply_m1_constraint=False)
indata, static, boundary = vj.extend_boundary_for_max_mode(indata, static, boundary, MAX_MODE)
boundary_input = vj.boundary_input_from_indata(indata, static.modes)

specs = vj.boundary_param_specs(
    boundary_input,
    static.modes,
    max_mode=MAX_MODE,
    min_coeff=0.0,
    include=("rc", "zs"),
    fix=("rc00",),
)
params0 = np.zeros(len(specs))
print(f"Parameter space ({len(specs)} DOFs): {vj.boundary_param_names(specs)}")

state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
geom = eval_geom(state_guess, static)
signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
pressure = jnp.zeros_like(jnp.asarray(static.s))


def residuals_from_state(state):
    parts = []
    for name, target, weight in OBJECTIVE_TUPLES:
        if name == "aspect":
            aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
            parts.append(jnp.asarray([float(weight) * (aspect - float(target))], dtype=jnp.float64))
        elif name == "qs":
            qs = quasisymmetry_ratio_residual_from_state(
                state=state,
                static=static,
                indata=indata,
                signgs=signgs,
                flux_local=flux,
                prof_local={"pressure": pressure},
                pressure_local=pressure,
                surfaces=SURFACES,
                helicity_m=HELICITY_M,
                helicity_n=HELICITY_N,
            )
            parts.append(jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * float(weight))
        else:
            raise ValueError(f"Unknown objective block '{name}'")
    return jnp.concatenate(parts)


def qs_total_from_state(state):
    qs = quasisymmetry_ratio_residual_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        flux_local=flux,
        prof_local={"pressure": pressure},
        pressure_local=pressure,
        surfaces=SURFACES,
        helicity_m=HELICITY_M,
        helicity_n=HELICITY_N,
    )
    return float(OBJECTIVE_TUPLES[1][2]) ** 2 * float(qs["total"])


residuals_from_state._n_non_qs = 1
residuals_from_state._qs_total_from_state = qs_total_from_state

opt = vj.FixedBoundaryExactOptimizer(
    static,
    indata,
    boundary,
    specs,
    residuals_from_state,
    boundary_input=boundary_input,
)

print(f"\nAspect ratio (initial):        {opt.aspect_ratio(params0):.4f}")
print(f"QS objective (initial):        {opt.quasisymmetry_objective(params0):.6f}")
print(f"\nRunning {METHOD} (max_nfev={MAX_NFEV}) …")

result = opt.run(
    params0,
    method=METHOD,
    max_nfev=MAX_NFEV,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    verbose=1,
    target_aspect=TARGET_ASPECT,
)

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
opt.save_history(OUTPUT_DIR / "history.json", result)

print("\nGenerating plots …")
vj.plot_qh_optimization(
    OUTPUT_DIR / "wout_initial.nc",
    OUTPUT_DIR / "wout_final.nc",
    OUTPUT_DIR / "history.json",
    outdir=OUTPUT_DIR,
)
print("Done.")

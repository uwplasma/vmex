#!/usr/bin/env python
# ruff: noqa: E402
"""Quasi-helical symmetry optimisation with vmec_jax — fixed-resolution exact adjoint.

This script mirrors the workflow of SIMSOPT's ``QH_fixed_resolution.py`` but
stays entirely within vmec_jax.  Rather than argparse, all user-facing
parameters live as top-level variables so you can read, copy, and modify the
script without ever touching a CLI.

Workflow (read top to bottom):

1.  **Load configuration** — parse a VMEC namelist and build the static grid.
2.  **Define boundary DOFs** — choose which Fourier coefficients to optimise
    and up to what mode number.
3.  **Construct the least-squares problem** — select objectives (aspect ratio +
    quasisymmetry) and their relative weights via
    ``vj.make_qh_residuals_fn``.
4.  **Build the optimizer** — ``vj.FixedBoundaryExactOptimizer`` wraps the
    discrete-adjoint Jacobian, exact-solve caching, and Gauss-Newton loop.
5.  **Run the optimisation** — ``opt.run()`` returns an SciPy-like result dict.
6.  **Save outputs** — wout NetCDF files and a JSON history file.
7.  **Plot** — ``vj.plot_qh_optimization`` generates three publication-quality
    figures.

Reference: SIMSOPT's ``QH_fixed_resolution.py`` does the same task but calls
VMEC2000 as a subprocess and computes the Jacobian via finite differences.
Here the Jacobian is exact (discrete-adjoint), no subprocess is needed, and
the full workflow runs in a single Python process.
"""

from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64

# ── 0.  Floating-point precision ──────────────────────────────────────────────
# VMEC requires 64-bit floating point throughout.
enable_x64(True)

# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS  (edit these — no argparse needed)
# ─────────────────────────────────────────────────────────────────────────────

# Path to the VMEC namelist input file.
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp4_QH_warm_start"

# Maximum |m|, |n| mode number for the boundary parameter space.
# Increasing MAX_MODE adds higher harmonics to the optimisation DOFs.
# For this input the effective mode space saturates at MAX_MODE=2 because
# the initial boundary only carries modes up to m=1; MAX_MODE=3 gives the
# same parameter set.
MAX_MODE = 2

# Maximum number of residual + Jacobian evaluations combined.
MAX_NFEV = 15

# Convergence tolerances (relative cost reduction / gradient / step norm).
FTOL = 1e-3
GTOL = 1e-3
XTOL = 1e-3

# Quasi-helical symmetry helicity: |B| ~ B(m*theta - n*zeta).
HELICITY_M = 1
HELICITY_N = -1  # negative → quasi-helical (QH); 0 → quasi-axisymmetric (QA)

# Target aspect ratio (penalised as (aspect - TARGET_ASPECT)^2 in the objective).
TARGET_ASPECT = 7.0

# Flux surfaces on which to evaluate quasisymmetry (s ∈ [0, 1]).
SURFACES = np.arange(0.0, 1.01, 0.1)

# Relative weights for the two objective blocks.
# The combined residual is [ASPECT_WEIGHT*(aspect-target), QS_WEIGHT*qs_residuals...].
ASPECT_WEIGHT = 1.0
QS_WEIGHT = 1.0

# Output directory for wout files, history JSON, and figures.
OUTPUT_DIR = Path("results/qh_opt")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load configuration
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE.name} …")
cfg, indata = vj.load_config(str(INPUT_FILE))
static = vj.build_static(cfg)
boundary_input = vj.boundary_input_from_indata(indata, static.modes)
boundary = vj.boundary_from_indata(indata, static.modes, apply_m1_constraint=False)

# If MAX_MODE exceeds the modes available in the input file, extend the
# static grid and boundary so that all requested DOFs actually exist
# (initialised to zero), matching SIMSOPT's fixed_range() behaviour.
indata, static, boundary = vj.extend_boundary_for_max_mode(indata, static, boundary, MAX_MODE)
boundary_input = vj.boundary_input_from_indata(indata, static.modes)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Define boundary degrees of freedom (DOFs)
#
# boundary_param_specs selects which Fourier coefficients to optimise.
#   • include=("rc", "zs")  — R_cos and Z_sin coefficients (stellarator-symmetric)
#   • fix=("rc00",)         — keep the major radius fixed
#   • max_mode=MAX_MODE     — limit to |m|, |n| ≤ MAX_MODE
# ─────────────────────────────────────────────────────────────────────────────
specs = vj.boundary_param_specs(
    boundary_input,
    static.modes,
    max_mode=MAX_MODE,
    min_coeff=0.0,          # include all coefficients, even zero-valued ones
    include=("rc", "zs"),
    fix=("rc00",),          # major radius is not a DOF
)
params0 = np.zeros(len(specs))   # start at the reference boundary (zero perturbation)

print(f"Parameter space ({len(specs)} DOFs): {vj.boundary_param_names(specs)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Construct the least-squares problem
#
# make_qh_residuals_fn returns a residuals_from_state(VMECState) → 1-D array
# function that combines:
#   • one aspect-ratio residual: ASPECT_WEIGHT * (aspect - TARGET_ASPECT)
#   • one QS residual per surface (from quasisymmetry_ratio_residual_from_state)
# ─────────────────────────────────────────────────────────────────────────────
residuals_fn = vj.make_qh_residuals_fn(
    static,
    indata,
    helicity_m=HELICITY_M,
    helicity_n=HELICITY_N,
    target_aspect=TARGET_ASPECT,
    surfaces=SURFACES,
    aspect_weight=ASPECT_WEIGHT,
    qs_weight=QS_WEIGHT,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Build the optimizer
#
# FixedBoundaryExactOptimizer encapsulates:
#   • Forward solve (tight for accepted steps, relaxed for line-search trials)
#   • Discrete-adjoint Jacobian via checkpoint-tape replay + batched JVP
#   • Single-entry cache to avoid double tape builds at the same x
#   • History tracking per Jacobian evaluation
# ─────────────────────────────────────────────────────────────────────────────
opt = vj.FixedBoundaryExactOptimizer(
    static,
    indata,
    boundary,
    specs,
    residuals_fn,
    boundary_input=boundary_input,
)

print(f"\nAspect ratio (initial):        {opt.aspect_ratio(params0):.4f}")
print(f"QS objective (initial):        {opt.quasisymmetry_objective(params0):.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Run the optimisation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRunning Gauss-Newton (max_nfev={MAX_NFEV}) …")
result = opt.run(
    params0,
    max_nfev=MAX_NFEV,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    verbose=1,
)

print(f"\nTermination: {result['message']}")
print(f"Aspect ratio (final):          {opt.aspect_ratio(result['x']):.4f}")
print(f"QS objective (final):          {opt.quasisymmetry_objective(result['x']):.6f}")
print(f"Objective reduction:           {100*(1 - result['objective']/result['_history_dump']['objective_initial']):.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save outputs
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

opt.save_wout(OUTPUT_DIR / "wout_initial.nc", params0)
opt.save_wout(OUTPUT_DIR / "wout_final.nc", result["x"])
opt.save_history(OUTPUT_DIR / "history.json", result)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plot results
#
# plot_qh_optimization produces three figures saved to OUTPUT_DIR:
#   • boundary_comparison.png   3-D LCFS coloured by |B| (before/after)
#   • bmag_surface.png          |B| contour lines on LCFS (before/after)
#   • objective_history.png     Objective and aspect ratio vs iteration
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots …")
vj.plot_qh_optimization(
    OUTPUT_DIR / "wout_initial.nc",
    OUTPUT_DIR / "wout_final.nc",
    OUTPUT_DIR / "history.json",
    outdir=OUTPUT_DIR,
)
print("Done.")

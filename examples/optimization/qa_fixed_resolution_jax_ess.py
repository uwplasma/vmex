#!/usr/bin/env python
# ruff: noqa: E402
"""Quasi-axisymmetric (QA) optimisation with exponential spectral scaling (ESS).

This script is the QA counterpart of ``qh_fixed_resolution_jax.py``.  It
optimises an nfp=2 quasi-axisymmetric equilibrium for three objectives:

* **Aspect ratio**: ``ASPECT_WEIGHT * (aspect - TARGET_ASPECT)``
* **Mean iota**:   ``IOTA_WEIGHT   * (mean_iota - TARGET_IOTA)``
* **QA symmetry**: ``QS_WEIGHT     * quasisymmetry_ratio_residuals(m=1, n=0)``

A toggle ``USE_ESS`` enables *exponential spectral scaling* (ESS): each boundary
DOF is pre-scaled by ``exp(-ALPHA * max(|m|, |n|)) / exp(-ALPHA)`` so that the
Gauss-Newton step favours low-order harmonics over fine-scale ones.  This often
improves convergence when the boundary has many DOFs at high mode numbers.

All user-facing parameters are top-level variables — no argparse needed.

Workflow
--------
1. Load configuration (namelist → static grid).
2. Define boundary DOFs up to ``MAX_MODE``.
3. Build ``x_scale`` with/without ESS.
4. Construct the least-squares problem via ``vj.make_qs_residuals_fn``.
5. Build ``vj.FixedBoundaryExactOptimizer`` and run Gauss-Newton.
6. Save wout + history JSON.
7. Generate figures.
"""

from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64

# ── 0.  Floating-point precision ──────────────────────────────────────────────
enable_x64(True)

# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS  (edit these — no argparse needed)
# ─────────────────────────────────────────────────────────────────────────────

# Path to the VMEC namelist input file.
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp2_QA"

# Maximum |m|, |n| mode number for the boundary parameter space.
# max_mode=2 → 24 DOFs; max_mode=3 → 48 DOFs (significantly longer JIT time).
MAX_MODE = 2

# Maximum number of residual + Jacobian evaluations combined.
MAX_NFEV = 15

# Convergence tolerances (relative cost reduction / gradient / step norm).
FTOL = 1e-3
GTOL = 1e-3
XTOL = 1e-3

# Quasi-axisymmetric symmetry helicity: |B| ~ B(m*theta - n*zeta), n=0 → QA.
HELICITY_M = 1
HELICITY_N = 0   # 0 → quasi-axisymmetric (QA)

# Target aspect ratio.
TARGET_ASPECT = 6.0

# Target mean rotational transform (iota).  Mean is taken over all full-mesh surfaces.
TARGET_IOTA = 0.41

# Flux surfaces on which to evaluate quasisymmetry (s ∈ [0, 1]).
SURFACES = np.arange(0.0, 1.01, 0.1)

# Objective weights.
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT   = 1.0
QS_WEIGHT     = 1.0

# ── ESS settings ──────────────────────────────────────────────────────────────
# If True, boundary DOFs are scaled by exp(-ALPHA * max(|m|, |n|)) / exp(-ALPHA)
# so that high-mode-number harmonics are smaller in the scaled parameter space,
# encouraging the optimizer to first improve low-order shape.
USE_ESS = True
ALPHA   = 1.0   # ESS decay rate; larger → stronger suppression of high modes

# Output directory — subdirectory name reflects whether ESS was used.
_tag       = "ess" if USE_ESS else "no_ess"
OUTPUT_DIR = Path(f"results/qa_opt/{_tag}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load configuration
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE.name} …")
cfg, indata = vj.load_config(str(INPUT_FILE))
static = vj.build_static(cfg)
boundary = vj.boundary_from_indata(indata, static.modes)

# Extend modes if MAX_MODE exceeds what the input file provides.
indata, static, boundary = vj.extend_boundary_for_max_mode(indata, static, boundary, MAX_MODE)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Define boundary degrees of freedom (DOFs)
# ─────────────────────────────────────────────────────────────────────────────
specs = vj.boundary_param_specs(
    boundary,
    static.modes,
    max_mode=MAX_MODE,
    min_coeff=0.0,
    include=("rc", "zs"),
    fix=("rc00",),
)
params0 = np.zeros(len(specs))

print(f"Parameter space ({len(specs)} DOFs): {vj.boundary_param_names(specs)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-DOF scale vector (ESS or uniform)
#
# x_scale[i] controls the step size for parameter i in the Gauss-Newton loop.
# With ESS=True, high-mode DOFs are down-weighted relative to low-mode ones.
# With ESS=False, all DOFs are treated uniformly (x_scale = ones).
# ─────────────────────────────────────────────────────────────────────────────
if USE_ESS:
    x_scale = vj.create_x_scale(specs, alpha=ALPHA)
    print(f"ESS scales (alpha={ALPHA}): min={x_scale.min():.3f}  max={x_scale.max():.3f}")
else:
    x_scale = np.ones(len(specs))
    print("ESS disabled — uniform scales.")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Construct the least-squares problem
#
# make_qs_residuals_fn returns a residuals_from_state(VMECState) → 1-D array
# combining:
#   • one aspect-ratio residual (if target_aspect is not None)
#   • one mean-iota residual   (if target_iota is not None)
#   • one QS residual per surface
# ─────────────────────────────────────────────────────────────────────────────
residuals_fn = vj.make_qs_residuals_fn(
    static,
    indata,
    helicity_m=HELICITY_M,
    helicity_n=HELICITY_N,
    target_aspect=TARGET_ASPECT,
    target_iota=TARGET_IOTA,
    surfaces=SURFACES,
    aspect_weight=ASPECT_WEIGHT,
    iota_weight=IOTA_WEIGHT,
    qs_weight=QS_WEIGHT,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Build the optimizer
# ─────────────────────────────────────────────────────────────────────────────
opt = vj.FixedBoundaryExactOptimizer(static, indata, boundary, specs, residuals_fn)

print(f"\nAspect ratio (initial):        {opt.aspect_ratio(params0):.4f}")
print(f"QS objective (initial):        {opt.quasisymmetry_objective(params0):.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Run the optimisation
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRunning Gauss-Newton (max_nfev={MAX_NFEV}, ESS={USE_ESS}) …")
result = opt.run(
    params0,
    max_nfev=MAX_NFEV,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    x_scale=x_scale,
    verbose=1,
)

print(f"\nTermination: {result['message']}")
print(f"Aspect ratio (final):          {opt.aspect_ratio(result['x']):.4f}")
print(f"QS objective (final):          {opt.quasisymmetry_objective(result['x']):.6f}")
_hist = result.get("_history_dump", {})
_obj0 = _hist.get("objective_initial", None)
if _obj0 is not None and _obj0 > 0.0:
    print(f"Objective reduction:           {100*(1 - result['objective']/_obj0):.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Save outputs
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Annotate history dump with metadata for plotting
_ess_tag = f"ESS α={ALPHA}" if USE_ESS else "no ESS"
result["_history_dump"]["label"] = f"QA opt (max_mode={MAX_MODE}, {_ess_tag})"
result["_history_dump"]["target_aspect"] = TARGET_ASPECT

opt.save_wout(OUTPUT_DIR / "wout_initial.nc", params0)
opt.save_wout(OUTPUT_DIR / "wout_final.nc", result["x"])
opt.save_history(OUTPUT_DIR / "history.json", result)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Plot results
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots …")
vj.plot_qh_optimization(
    OUTPUT_DIR / "wout_initial.nc",
    OUTPUT_DIR / "wout_final.nc",
    OUTPUT_DIR / "history.json",
    outdir=OUTPUT_DIR,
)
print(f"Done.  Results saved to {OUTPUT_DIR}/")

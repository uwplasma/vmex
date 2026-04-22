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

When ``MAX_MODE > 1`` and ``USE_MODE_CONTINUATION = True``, the script first
solves the lower-mode QA problem and then lifts that solution into the richer
boundary space before running the final stage.  For the 24-DOF QA case this
continuation is the difference between a worse local minimum and the expected
improvement over ``max_mode=1``.

All user-facing parameters are top-level variables — no argparse needed.

Workflow
--------
1. Load configuration (namelist → static grid).
2. Define boundary DOFs up to ``MAX_MODE``.
3. Build ``x_scale`` with/without ESS.
4. Construct the least-squares problem via ``vj.make_qs_residuals_fn``.
5. Optionally solve lower-mode continuation stages.
6. Build ``vj.FixedBoundaryExactOptimizer`` and run the final stage.
7. Save wout + history JSON.
8. Generate figures.
"""

from pathlib import Path
import os

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64
from vmec_jax.config import config_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.optimization import rebuild_indata_with_resolution
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

# ── 0.  Floating-point precision ──────────────────────────────────────────────
enable_x64(True)

# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS  (edit these — no argparse needed)
# ─────────────────────────────────────────────────────────────────────────────

# Path to the VMEC namelist input file.
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp2_QA"

# Choose the VMEC solver resolution directly in the script.
VMEC_MPOL = 5
VMEC_NTOR = 5

# Maximum |m|, |n| mode number for the boundary parameter space.
# max_mode=2 → 24 DOFs; max_mode=3 → 48 DOFs (significantly longer JIT time).
MAX_MODE = 2

# Maximum number of residual + Jacobian evaluations combined.
# When MAX_MODE > 1, the final stage is warm-started from the previous mode.
MAX_NFEV = 40
CONTINUATION_NFEV = 15

# Outer least-squares method. "scipy" uses exact residuals + exact
# discrete-adjoint Jacobians through scipy.optimize.least_squares.
METHOD = "scipy"

# Convergence tolerances (relative cost reduction / gradient / step norm).
FTOL = 1e-3
GTOL = 1e-3
XTOL = 1e-3

# VMEC inner solve budget used for accepted points and line-search trials.
INNER_MAX_ITER = 1
INNER_FTOL = 1e-13
TRIAL_MAX_ITER = 1
TRIAL_FTOL = 1e-10

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
OBJECTIVE_TUPLES = [
    ("aspect", TARGET_ASPECT, ASPECT_WEIGHT),
    ("iota", TARGET_IOTA, IOTA_WEIGHT),
    ("qs", 0.0, QS_WEIGHT),
]

# ── ESS settings ──────────────────────────────────────────────────────────────
# If True, boundary DOFs are scaled by exp(-ALPHA * max(|m|, |n|)) / exp(-ALPHA)
# so that high-mode-number harmonics are smaller in the scaled parameter space,
# encouraging the optimizer to first improve low-order shape.  The max_mode=3
# QA problem benefits from a stronger ESS profile than max_mode=2, so the
# script automatically promotes ALPHA and continuation budget when MAX_MODE >= 3
# unless the user overrides them explicitly.
USE_ESS = True
ALPHA   = 0.8
USE_MODE_CONTINUATION = True

# Output directory — subdirectory name reflects whether ESS was used.
_tag       = "ess" if USE_ESS else "no_ess"
OUTPUT_DIR = Path(f"results/qa_opt/{_tag}")

# Optional environment overrides for benchmarking without editing the file.
MAX_MODE = int(os.environ.get("VMEC_JAX_QA_MAX_MODE", str(MAX_MODE)))
MAX_NFEV = int(os.environ.get("VMEC_JAX_QA_MAX_NFEV", str(MAX_NFEV)))
CONTINUATION_NFEV = int(os.environ.get("VMEC_JAX_QA_CONTINUATION_NFEV", str(CONTINUATION_NFEV)))
METHOD = os.environ.get("VMEC_JAX_QA_METHOD", METHOD)
FTOL = float(os.environ.get("VMEC_JAX_QA_FTOL", str(FTOL)))
GTOL = float(os.environ.get("VMEC_JAX_QA_GTOL", str(GTOL)))
XTOL = float(os.environ.get("VMEC_JAX_QA_XTOL", str(XTOL)))
USE_ESS = os.environ.get("VMEC_JAX_QA_USE_ESS", str(USE_ESS)).lower() in {"1", "true", "yes", "on"}
ALPHA = float(os.environ.get("VMEC_JAX_QA_ALPHA", str(ALPHA)))
USE_MODE_CONTINUATION = os.environ.get(
    "VMEC_JAX_QA_USE_CONTINUATION", str(USE_MODE_CONTINUATION)
).lower() in {"1", "true", "yes", "on"}
OUTPUT_DIR = Path(os.environ.get("VMEC_JAX_QA_OUTPUT_DIR", str(OUTPUT_DIR)))

# The 48-DOF QA problem needs a stronger continuation seed than max_mode=2.
# When the user switches MAX_MODE to 3 without specifying tuning overrides,
# promote the continuation budget and ESS alpha automatically so the script
# lands in the better basin instead of the early shallow one.
if MAX_MODE >= 3:
    if "VMEC_JAX_QA_CONTINUATION_NFEV" not in os.environ:
        CONTINUATION_NFEV = max(CONTINUATION_NFEV, 25)
    if USE_ESS and "VMEC_JAX_QA_ALPHA" not in os.environ:
        ALPHA = max(ALPHA, 1.6)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load configuration
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_FILE.name} …")
cfg, indata = vj.load_config(str(INPUT_FILE))
import vmec_jax._compat as _compat
_jnp = _compat.jnp
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
    stage_pressure = _jnp.zeros_like(_jnp.asarray(stage_static.s))

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

    def stage_residuals_fn(state):
        parts = []
        for name, target, weight in OBJECTIVE_TUPLES:
            if name == "aspect":
                aspect = equilibrium_aspect_ratio_from_state(state=state, static=stage_static)
                parts.append(_jnp.asarray([float(weight) * (aspect - float(target))], dtype=_jnp.float64))
            elif name == "iota":
                _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
                    state=state, static=stage_static, indata=stage_indata, signgs=stage_signgs
                )
                iotas = _jnp.asarray(iotas, dtype=_jnp.float64)
                mean_iota = _jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else _jnp.mean(iotas[1:])
                parts.append(_jnp.asarray([float(weight) * (mean_iota - float(target))], dtype=_jnp.float64))
            elif name == "qs":
                qs = stage_qs_eval(state)
                parts.append(_jnp.asarray(qs["residuals1d"], dtype=_jnp.float64) * float(weight))
            else:
                raise ValueError(f"Unknown objective block '{name}'")
        return _jnp.concatenate(parts)

    stage_residuals_fn._n_non_qs = 2
    stage_residuals_fn._qs_total_from_state = lambda state: float(QS_WEIGHT) ** 2 * float(stage_qs_eval(state)["total"])
    stage_opt = vj.FixedBoundaryExactOptimizer(
        stage_static,
        stage_indata,
        stage_boundary,
        stage_specs,
        stage_residuals_fn,
        boundary_input=stage_boundary_input,
        inner_max_iter=INNER_MAX_ITER,
        inner_ftol=INNER_FTOL,
        trial_max_iter=TRIAL_MAX_ITER,
        trial_ftol=TRIAL_FTOL,
    )
    stage_x_scale = (
        vj.create_x_scale(stage_specs, alpha=ALPHA) if USE_ESS else np.ones(len(stage_specs))
    )
    return stage_indata, stage_static, stage_boundary_input, stage_specs, stage_opt, stage_x_scale


def _iota_fn(state, *, stage_static, stage_indata, stage_opt):
    _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
        state=state, static=stage_static, indata=stage_indata, signgs=stage_opt._signgs)
    iotas = _jnp.asarray(iotas, dtype=_jnp.float64)
    if int(iotas.shape[0]) <= 1:
        return 0.0
    return float(_jnp.mean(iotas[1:]))

stage_results = []
params_stage = None
stage_modes = list(range(1, MAX_MODE + 1)) if (USE_MODE_CONTINUATION and MAX_MODE > 1) else [MAX_MODE]

for stage_mode in stage_modes:
    stage_indata, stage_static, stage_boundary_input, stage_specs, stage_opt, stage_x_scale = _build_stage(stage_mode)
    params0_stage = np.zeros(len(stage_specs)) if params_stage is None else vj.lift_boundary_params(
        prev_specs, params_stage, stage_specs
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
        print(f"\nRunning {METHOD} least-squares (max_nfev={MAX_NFEV}, ESS={USE_ESS}) …")
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
        iota_fn=lambda state, s=stage_static, i=stage_indata, o=stage_opt: _iota_fn(
            state, stage_static=s, stage_indata=i, stage_opt=o
        ),
        target_iota=TARGET_IOTA,
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
    }

print(f"\nTermination: {result['message']}")
print(f"Aspect ratio (final):          {opt.aspect_ratio(result['x']):.4f}")
print(f"QS objective (final):          {opt.quasisymmetry_objective(result['x']):.6f}")
_hist = result.get("_history_dump", {})
_obj0 = _hist.get("objective_initial", None)
_obj_f = _hist.get("objective_final", None)
if _obj0 is not None and _obj0 > 0.0 and _obj_f is not None:
    print(f"Objective reduction:           {100*(1 - _obj_f/_obj0):.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Save outputs
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Annotate history dump with metadata for plotting
_ess_tag = f"ESS α={ALPHA}" if USE_ESS else "no ESS"
history_dump = result["_history_dump"] if combined_history is None else combined_history
label_suffix = ", continuation" if combined_history is not None else ""
history_dump["label"] = f"QA opt (max_mode={MAX_MODE}, {_ess_tag}{label_suffix})"
history_dump["target_aspect"] = TARGET_ASPECT
history_dump["target_iota"] = TARGET_IOTA
result["_history_dump"] = history_dump

opt.save_wout(OUTPUT_DIR / "wout_initial.nc", np.zeros(len(specs)))
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

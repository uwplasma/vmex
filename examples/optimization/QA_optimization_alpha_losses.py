#!/usr/bin/env python
"""Optimize a two-period boundary for quasi-axisymmetry and fusion-alpha confinement.

Two stages. The first is the quasi-axisymmetry least-squares problem of
``QA_optimization.py`` (quasisymmetry ratio, aspect ratio, iota floor), with
VMEX's exact Jacobian; it removes most of the losses of the rotating-ellipse
seed. The second refines the boundary against the alphas themselves: each
evaluation solves the equilibrium, scales it to ARIES-CS size and traces
fusion-born alphas in Boozer coordinates through ESSOS (the ``vmex --trace``
tracer). Its cost is

    f = L + w_A (A - A*)^2 + w_i max(iota* - min|iota|, 0)^2,

where ``L`` is the loss weighted by how early each alpha leaves,
``mean over alphas of (1 - t_loss / t_max)`` (zero for confined alphas). The
weighting makes ``L`` move before an alpha crosses from lost to confined,
which the plain lost count does not. That count still has no useful gradient,
so SciPy's derivative-free COBYQA drives the second stage. Every evaluation
starts from the same stage-1 equilibrium and traces the same random births,
so ``f`` is deterministic.

The seed, the QA stage and the final boundary are then traced with a larger,
longer ensemble and a different seed, and compared in one figure. Measured on
a 10-core laptop in about 3.5 min: 30.3 % +- 1.5 % of 1000 alphas lost within
5 ms for the seed, 5.5 % +- 0.7 % after the QA stage. The alpha stage lowers
the loss of its own 1 ms ensemble from 2.5 % to 1.6 %, but reads 5.7 % on the
independent check, so at this budget quasi-axisymmetry does the work and the
alpha stage only refines within the noise. Needs ``pip install "vmex[coils]"``.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

import vmex as vj
from vmex import optimize as opt

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed, as in QA_optimization.py:
SEED_PERTURBATION = 0.05

# Stage 1, quasi-axisymmetry: surfaces of the QS residual, the mode ladder
# (highest boundary mode varied per stage, and its residual evaluations), and
# step control, as in QA_optimization.py:
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES = [1, 2]
MAX_NFEV = [10, 15]
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0

# Targets shared by both stages:
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
MAGNETIC_WELL_TARGET = 0.01       # stage 1 only

# Stage 2, alpha losses: boundary coefficients (m, n) varied in RBC and ZBS,
# and the penalty weights of the cost above:
LOSS_MODES = [(0, 1), (1, -1), (1, 0), (1, 1)]
ASPECT_WEIGHT = 0.01
IOTA_WEIGHT = 10.0

# Alpha ensemble per stage-2 evaluation: particles, horizon [s], birth surface.
# The tracer scales the equilibrium to ARIES-CS size in memory first:
N_ALPHAS = 1000
T_MAX = 1e-3
BIRTH_S = 0.3
SEED = 1

# COBYQA: evaluations, and initial and final trust-region radii [m]:
MAX_EVALUATIONS = 14
INITIAL_STEP = 1e-3
FINAL_STEP = 1e-4

# Verification ensemble, with a different seed:
FINAL_ALPHAS = 1000
FINAL_T_MAX = 5e-3

# Every output file name contains this:
OUTPUT_NAME = "QA_alpha_losses_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [3]
    N_ALPHAS, T_MAX, MAX_EVALUATIONS = 32, 1e-4, 3
    FINAL_ALPHAS, FINAL_T_MAX = 32, 1e-4

###############################################################################
# End of input parameters.
###############################################################################

# ESSOS shards the alphas over the JAX devices: one CPU device per core, as
# ``vmex --trace`` does. This must run before JAX starts its backend.
jax.config.update("jax_num_cpu_devices", os.cpu_count() or 1)

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
# Boundary arrays are indexed [n + ntor, m].
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
mpol = max(max(MAX_MODES) + 2, 5)
seed_input = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
seed_equilibrium = opt.solve_equilibrium(seed_input)

### Stage 1: quasi-axisymmetry ################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|."""
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
problem = opt.VmecProblem.from_tuples(
    seed_input, [(qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT_TARGET, 1.0), (iota_floor, 0.0, 10.0),
                 (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0)],
    max_mode=max(MAX_MODES), use_ess=True, restart_from=seed_equilibrium)
x = problem.x0
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== Stage 1: quasi-axisymmetry, max_mode = {max_mode} =====")
    stage = problem.subproblem(max_mode=max_mode, x=x)
    step = PARAMETER_STEP * stage.scales
    result = least_squares(
        stage.residual, stage.x0, jac=stage.residual_jac, x_scale=step, max_nfev=max_nfev,
        bounds=(stage.x0 - MAX_PARAMETER_CHANGE * step, stage.x0 + MAX_PARAMETER_CHANGE * step),
        ftol=1e-6, xtol=1e-10, verbose=1)
    x = stage.embed(result.x)
qa_input = problem.input_from_x(x)
qa_equilibrium = problem.equilibrium_from_x(x)
print(f"QS total {float(qs.total_state(qa_equilibrium.state, qa_equilibrium.runtime)):.3e}")

### Stage 2: alpha losses #####################################################

def input_from_x(x):
    rbc, zbs = qa_input.rbc.copy(), qa_input.zbs.copy()
    for k, (m, n) in enumerate(LOSS_MODES):
        rbc[n + qa_input.ntor, m], zbs[n + qa_input.ntor, m] = x[k], x[k + len(LOSS_MODES)]
    return replace(qa_input, rbc=rbc, zbs=zbs)


def evaluate(x, n_alphas=N_ALPHAS, t_max=T_MAX, seed=SEED):
    """Solve from the stage-1 state, trace, and return (cost, terms, equilibrium, alphas)."""
    equilibrium = opt.solve_equilibrium(input_from_x(x), initial_state=qa_equilibrium.solution)
    alphas = vj.trace_alphas(equilibrium.wout, nparticles=n_alphas, tmax=t_max,
                             s=BIRTH_S, seed=seed, times_to_trace=100)
    lost = alphas.lost_times >= 0
    weighted_loss = float(np.where(lost, 1.0 - alphas.lost_times / t_max, 0.0).mean())
    aspect = float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime))
    iota = float(opt.min_abs_iota(equilibrium.state, equilibrium.runtime))
    cost = (weighted_loss + ASPECT_WEIGHT * (aspect - ASPECT_TARGET) ** 2
            + IOTA_WEIGHT * max(IOTA_FLOOR - iota, 0.0) ** 2)
    return cost, dict(loss=alphas.loss_fraction, weighted_loss=weighted_loss,
                      aspect=aspect, min_iota=iota), equilibrium, alphas


history = []


def objective(x):
    start = time.perf_counter()
    try:
        cost, terms, _, _ = evaluate(x)
    except Exception as error:     # an unsolvable boundary: steer away from it
        print(f"[{len(history) + 1:3d}] solve failed ({type(error).__name__}); cost 10")
        history.append(dict(cost=10.0))
        return 10.0
    history.append(dict(cost=cost, **terms))
    print(f"[{len(history):3d}] cost {cost:.4e}  lost {100 * terms['loss']:5.1f} %  "
          f"weighted {terms['weighted_loss']:.4f}  aspect {terms['aspect']:.3f}  "
          f"min|iota| {terms['min_iota']:.3f}  ({time.perf_counter() - start:.1f} s)", flush=True)
    return cost


print(f"\n===== Stage 2: alpha losses, RBC and ZBS at (m, n) = {LOSS_MODES} =====")
x0 = np.array([qa_input.rbc[n + qa_input.ntor, m] for m, n in LOSS_MODES]
              + [qa_input.zbs[n + qa_input.ntor, m] for m, n in LOSS_MODES])
result = minimize(objective, x0, method="COBYQA",
                  options=dict(maxfev=MAX_EVALUATIONS, initial_tr_radius=INITIAL_STEP,
                               final_tr_radius=FINAL_STEP))
x_best = result.x if result.fun <= history[0]["cost"] else x0   # never return worse than the start
final_input = input_from_x(x_best)

### Check the result ##########################################################

# The optimizer saw N_ALPHAS alphas for T_MAX with one seed. Quote the seed,
# the QA stage and the final boundary on a larger, longer ensemble instead.
checks = {}
for label, case in (("seed", seed_input), ("QA stage", qa_input), ("alpha stage", final_input)):
    equilibrium = opt.solve_equilibrium(case, initial_state=qa_equilibrium.solution)
    alphas = vj.trace_alphas(equilibrium.wout, nparticles=FINAL_ALPHAS, tmax=FINAL_T_MAX,
                             s=BIRTH_S, seed=SEED + 1, times_to_trace=100)
    checks[label] = (alphas, equilibrium)
    print(f"{label:>11}: {FINAL_ALPHAS} alphas for {1e3 * FINAL_T_MAX:g} ms lose "
          f"{100 * alphas.loss_fraction:.1f} % +- {100 * alphas.loss_fraction_sigma:.1f} %, "
          f"aspect {float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.3f}, "
          f"min|iota| {float(opt.min_abs_iota(equilibrium.state, equilibrium.runtime)):.3f}, "
          f"QS total {float(qs.total_state(equilibrium.state, equilibrium.runtime)):.2e}")

### Print, plot and save ######################################################

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", checks["alpha stage"][1].wout)
print(f"Wrote {input_path}\nWrote {wout_path}")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, (ax_loss, ax_cost) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
for label, (alphas, _) in checks.items():
    t, f = np.asarray(alphas.times), np.asarray(alphas.loss_fractions)
    keep = t > 0
    ax_loss.plot(t[keep], 100 * f[keep], label=f"{label}: {100 * alphas.loss_fraction:.1f} %")
ax_loss.set(xscale="log", xlabel="time [s]", ylabel="alphas lost [%]",
            title=f"{FINAL_ALPHAS} alphas born on s = {BIRTH_S}")
ax_loss.legend(frameon=False)
costs = np.array([h["cost"] for h in history])
ax_cost.semilogy(np.arange(1, len(costs) + 1), costs, ".", color="0.6", label="evaluation")
ax_cost.semilogy(np.arange(1, len(costs) + 1), np.minimum.accumulate(costs), "-", label="best so far")
ax_cost.set(xlabel="stage-2 evaluation", ylabel="cost", title="alpha-loss stage")
ax_cost.legend(frameon=False)
fig.savefig(f"{OUTPUT_NAME}_losses.png", dpi=150)
print(f"Wrote {OUTPUT_NAME}_losses.png")
for path in vj.plot_tracing(checks["alpha stage"][0], ".", name=OUTPUT_NAME).values():
    print(f"Wrote {path}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")

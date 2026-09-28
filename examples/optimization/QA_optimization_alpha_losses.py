#!/usr/bin/env python
"""Optimize a two-period boundary for fusion-alpha confinement.

Each objective evaluation solves the equilibrium, scales it to ARIES-CS size,
and traces fusion-born alphas in Boozer coordinates through ESSOS (the
``vmex --trace`` tracer). The cost is

    f = L + w_A (A - A*)^2 + w_i max(iota* - min|iota|, 0)^2,

where ``L`` is the loss fraction weighted by how early each alpha is lost,
``mean over alphas of (1 - t_loss / t_max)`` (zero for confined alphas), ``A``
is the aspect ratio and ``min|iota|`` the smallest rotational transform.
Weighting by loss time makes ``L`` move with the boundary before an alpha
crosses from lost to confined, which the plain lost count does not.

Loss counts have no useful gradient (ESSOS #61 measures an identically zero
one), so SciPy's derivative-free COBYQA drives the boundary. Every evaluation
uses the same random seed, so ``f`` is deterministic. It stays piecewise
constant at the scale of one alpha, which is why the ensemble is kept at a few
hundred. Needs ``pip install "vmex[coils]"``.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.tracing import trace_alphas

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed, as in QA_optimization.py:
SEED_PERTURBATION = 0.05

# Boundary coefficients (m, n) varied in RBC and ZBS; RBC(0,0) stays fixed:
MODES = [(0, 1), (1, -1), (1, 0), (1, 1)]

# Alpha ensemble per evaluation: particles, horizon [s] and birth surface.
# The tracer scales the equilibrium to ARIES-CS size in memory first:
N_ALPHAS = 300
T_MAX = 2e-3
BIRTH_S = 0.3
SEED = 1

# Targets and penalty weights:
ASPECT_TARGET = 6.0
ASPECT_WEIGHT = 0.01
IOTA_FLOOR = 0.3
IOTA_WEIGHT = 10.0

# Optimizer: evaluations, and the initial and final trust-region radii of
# COBYQA in metres of boundary coefficient:
MAX_EVALUATIONS = 60
INITIAL_STEP = 0.02
FINAL_STEP = 1e-3
BOUND = 0.15                      # no coefficient moves further than this

# Verification: a larger ensemble for longer, on the optimized boundary:
FINAL_ALPHAS = 1000
FINAL_T_MAX = 1e-2

# Every output file name contains this:
OUTPUT_NAME = "QA_alpha_losses_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    N_ALPHAS, T_MAX, MAX_EVALUATIONS = 64, 2e-4, 4
    FINAL_ALPHAS, FINAL_T_MAX = 64, 2e-4

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
# Boundary arrays are indexed [n + ntor, m].
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)


def x_from_input(inp):
    """The decision vector: RBC then ZBS at MODES."""
    return np.array([inp.rbc[n + inp.ntor, m] for m, n in MODES]
                    + [inp.zbs[n + inp.ntor, m] for m, n in MODES])


def input_from_x(x):
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    for k, (m, n) in enumerate(MODES):
        rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = x[k], x[k + len(MODES)]
    return replace(inp, rbc=rbc, zbs=zbs)


### Set up the objective ######################################################

history = []
warm_start = {"state": None}


def evaluate(x, n_alphas=N_ALPHAS, t_max=T_MAX):
    """Solve, trace and return (cost, equilibrium, tracing result)."""
    equilibrium = opt.solve_equilibrium(input_from_x(x), initial_state=warm_start["state"])
    warm_start["state"] = equilibrium.solution
    alphas = trace_alphas(equilibrium.wout, nparticles=n_alphas, tmax=t_max,
                          s=BIRTH_S, seed=SEED, times_to_trace=100)
    lost = alphas.lost_times >= 0
    weighted_loss = float(np.where(lost, 1.0 - alphas.lost_times / t_max, 0.0).mean())
    aspect = float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime))
    iota = float(opt.min_abs_iota(equilibrium.state, equilibrium.runtime))
    cost = (weighted_loss + ASPECT_WEIGHT * (aspect - ASPECT_TARGET) ** 2
            + IOTA_WEIGHT * max(IOTA_FLOOR - iota, 0.0) ** 2)
    return cost, dict(loss=alphas.loss_fraction, weighted_loss=weighted_loss,
                      aspect=aspect, min_iota=iota), equilibrium


def objective(x):
    start = time.perf_counter()
    try:
        cost, terms, _ = evaluate(x)
    except Exception as error:     # an unsolvable boundary: steer away from it
        print(f"[{len(history) + 1:3d}] solve failed ({type(error).__name__}); cost 10")
        history.append(dict(cost=10.0))
        return 10.0
    history.append(dict(cost=cost, **terms))
    print(f"[{len(history):3d}] cost {cost:.4e}  lost {100 * terms['loss']:5.1f} %  "
          f"weighted {terms['weighted_loss']:.4f}  aspect {terms['aspect']:.3f}  "
          f"min|iota| {terms['min_iota']:.3f}  ({time.perf_counter() - start:.1f} s)", flush=True)
    return cost


### Run the optimization ######################################################

x0 = x_from_input(inp)
print(f"Varying RBC and ZBS at (m, n) = {MODES}: {x0.size} coefficients")
result = minimize(objective, x0, method="COBYQA",
                  bounds=list(zip(x0 - BOUND, x0 + BOUND)),
                  options=dict(maxfev=MAX_EVALUATIONS, initial_tr_radius=INITIAL_STEP,
                               final_tr_radius=FINAL_STEP, disp=True))

### Check the result ##########################################################

# The optimizer saw N_ALPHAS alphas for T_MAX with one seed. Quote the start
# and the optimum on a larger, longer ensemble with a different seed instead.
SEED += 1
final = {}
for label, x in (("initial", x0), ("optimized", result.x)):
    warm_start["state"] = None
    _, terms, equilibrium = evaluate(x, FINAL_ALPHAS, FINAL_T_MAX)
    sigma = np.sqrt(terms["loss"] * (1 - terms["loss"]) / FINAL_ALPHAS)
    final[label] = (terms, equilibrium)
    print(f"{label:>9}: {FINAL_ALPHAS} alphas for {1e3 * FINAL_T_MAX:g} ms lose "
          f"{100 * terms['loss']:.1f} % +- {100 * sigma:.1f} %, aspect {terms['aspect']:.3f}, "
          f"min|iota| {terms['min_iota']:.3f}")

### Print, plot and save ######################################################

final_input = input_from_x(result.x)
input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final["optimized"][1].wout)
print(f"Wrote {input_path}\nWrote {wout_path}")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

costs = [h["cost"] for h in history]
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
ax.semilogy(np.arange(1, len(costs) + 1), costs, ".", color="0.6", label="evaluation")
ax.semilogy(np.arange(1, len(costs) + 1), np.minimum.accumulate(costs), "-", label="best so far")
ax.set(xlabel="evaluation", ylabel="cost", title="alpha-loss optimization")
ax.legend(frameon=False)
fig.savefig(f"{OUTPUT_NAME}_objectives.png", dpi=150)
print(f"Wrote {OUTPUT_NAME}_objectives.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")

#!/usr/bin/env python
"""Directly minimize fusion-alpha losses from a non-optimized NFP=2 seed.

The only design terms are traced loss fraction and aspect ratio. No
quasisymmetry, omnigenity or iota target is used. Each trial samples births
from its own field with the same RNG seed; a longer, independent trace checks
the result.

Run ``python examples/optimization/alpha_particle_optimization.py`` after
``pip install 'vmex[coils]'``. ``VMEX_EXAMPLES_CI=1`` runs a short wiring test.
This is a research example: increase the particle and evaluation budgets for
converged optimization and check loss at the intended reactor time horizon.
Direct loss optimization methods are discussed in
https://arxiv.org/abs/2302.11369.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt

DATA = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2"
MODES = [(0, 1), (1, -1), (1, 0), (1, 1)]
ASPECT_WEIGHT = 0.05
N_PARTICLES, T_MAX, MAX_EVAL = 2000, 2e-3, 30
CHECK_PARTICLES, CHECK_T_MAX = 2000, 5e-3
BIRTH_S, SEED = 0.3, 1
NAME = "alpha_particle_optimized"
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    N_PARTICLES, T_MAX, MAX_EVAL = 32, 1e-4, 10
    CHECK_PARTICLES, CHECK_T_MAX = 32, 1e-4

jax.config.update("jax_num_cpu_devices", os.cpu_count() or 1)

inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = 0.17
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.03, 0.03
seed_input = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
    mpol=5, ntor=5, ntheta=16, nzeta=14)
seed_eq = opt.solve_equilibrium(seed_input)
aspect_target = float(opt.aspect_ratio(seed_eq.state, seed_eq.runtime))


def input_from_x(x):
    rbc, zbs = seed_input.rbc.copy(), seed_input.zbs.copy()
    for k, (m, n) in enumerate(MODES):
        rbc[n + seed_input.ntor, m] = x[k]
        zbs[n + seed_input.ntor, m] = x[k + len(MODES)]
    return replace(seed_input, rbc=rbc, zbs=zbs)


x0 = np.array([seed_input.rbc[n + seed_input.ntor, m] for m, n in MODES]
              + [seed_input.zbs[n + seed_input.ntor, m] for m, n in MODES])
history = []


def evaluate(x):
    eq = opt.solve_equilibrium(input_from_x(x), initial_state=seed_eq.solution)
    trace = vj.trace_alphas(eq.wout, nparticles=N_PARTICLES, tmax=T_MAX,
                            s=BIRTH_S, seed=SEED, times_to_trace=20)
    if trace.particles_failed or not np.isfinite(trace.energy_error).all():
        raise ValueError("unreliable alpha trace: failed orbit or nonfinite energy error")
    aspect = float(opt.aspect_ratio(eq.state, eq.runtime))
    cost = float(trace.loss_fraction + ASPECT_WEIGHT * (aspect - aspect_target) ** 2)
    return cost, trace.loss_fraction, aspect


def objective(x):
    tic = time.perf_counter()
    try:
        cost, loss, aspect = evaluate(x)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        print(f"{len(history) + 1:3d}: invalid evaluation ({exc}); cost 10", flush=True)
        cost, loss, aspect = 10.0, np.nan, np.nan
    history.append((cost, loss, aspect))
    print(f"{len(history):3d}: cost {cost:.5f}, lost {loss:.1%}, "
          f"aspect {aspect:.3f}, {time.perf_counter() - tic:.1f} s", flush=True)
    return cost


print("Direct alpha-loss optimization (no symmetry stage or symmetry objective)", flush=True)
fit = minimize(objective, x0, method="COBYQA",
               options=dict(maxfev=MAX_EVAL, initial_tr_radius=1e-2,
                            final_tr_radius=1e-4))
best = fit.x if fit.fun < history[0][0] else x0
final_input = input_from_x(best)

# Fresh particles and a longer horizon are never seen by the optimizer.
checks = {}
for label, case in (("seed", seed_input), ("optimized", final_input)):
    eq = opt.solve_equilibrium(case, initial_state=seed_eq.solution)
    trace = vj.trace_alphas(eq.wout, nparticles=CHECK_PARTICLES,
                            tmax=CHECK_T_MAX, s=BIRTH_S, seed=SEED + 1,
                            times_to_trace=100)
    if trace.particles_failed or not np.isfinite(trace.energy_error).all():
        raise ValueError(f"unreliable {label} holdout trace")
    checks[label] = trace
    print(f"{label}: {CHECK_PARTICLES} alphas for {1e3 * CHECK_T_MAX:g} ms lose "
          f"{100 * trace.loss_fraction:.1f}% ± {100 * trace.loss_fraction_sigma:.1f}%; "
          f"aspect {float(opt.aspect_ratio(eq.state, eq.runtime)):.3f}", flush=True)
    if label == "optimized":
        wout_path = vj.write_wout(f"wout_{NAME}.nc", eq.wout)
print(f"Wrote {final_input.to_indata(f'input.{NAME}')} and {wout_path}")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
for label, trace in checks.items():
    ax.plot(trace.times[1:], 100 * trace.loss_fractions[1:], label=label)
ax.set(xscale="log", xlabel="time [s]", ylabel="alphas lost [%]")
ax.legend(frameon=False)
fig.savefig(f"{NAME}_losses.png", dpi=150)
print(f"Wrote {NAME}_losses.png")

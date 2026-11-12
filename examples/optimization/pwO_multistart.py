#!/usr/bin/env python
"""Optimize a piecewise-omnigenous boundary from several deterministic starts and keep the best.

Each start perturbs one or two boundary modes of the circular seed. All starts run the first
stage of the pwO mode ladder; the one or two with the lowest cost go on to the next stage, and
the winner reaches the last. A start that sits in a poor basin shows it in the first-stage
cost (about ten times higher), so screening costs one cheap stage per start. The winner is
re-solved at ns 71 with ftol 1e-13. See ``pwO_optimization.py`` for the objective terms.
Requires vmex[neoclassical].
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.pwo import FreeMapProblem, MappedPwOResidual

# Seed deck (NFP 3), and the starts: each is a list of (m, n, amplitude) that is subtracted
# from RBC(m, n) and added to ZBS(m, n). Amplitudes may be large; no random numbers:
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp3"
STARTS = [[(1, -1, amplitude)] for amplitude in (0.03, 0.05, 0.08, -0.05)] + [
    [(1, -1, 0.05), (2, -1, amplitude)] for amplitude in (0.01, -0.01, 0.02)]

# Mode ladder, residual evaluations per stage, and how many starts survive each stage:
MAX_MODES = [1, 2, 3]
MAX_NFEV = [15, 15, 15]
KEEP = [2, 1, 1]

# Objective, as in pwO_optimization.py:
SURFACES = np.linspace(0.1, 0.9, 8)
MBOZ = 16
MAP_PARAMETERS = dict(a=0.3, b=0.0, c=-2.0, eta_min=2.8)
FREE_MAP = ("a", "b", "c", "eta_min")
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.42
MIRROR_LIMIT = 0.16
PWO_WEIGHT = 1.0e3
PARAMETER_STEP = 0.05
MAX_PARAMETER_CHANGE = 5.0

# Equilibrium resolution of the ladder and of the final solve (stepped: a jump flips the Jacobian):
MPOL = max(MAX_MODES[-1] + 2, 5)
LADDER_GRID = dict(delt=0.5, ns_array=np.array([35]), niter_array=np.array([5000]), ftol_array=np.array([1e-12]))
FINAL_NS = [45, 55, 71]
FINAL_FTOL = 1.0e-13

OUTPUT_NAME = "pwO_multistart_nfp3"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    STARTS, MAX_MODES, MAX_NFEV, KEEP = STARTS[:2], [1], [3], [1]
    FINAL_NS, FINAL_FTOL = [31], 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

pwo = MappedPwOResidual(SURFACES, mboz=MBOZ, nboz=MBOZ, oversample=2, limit_map=True, **MAP_PARAMETERS)


def max_mirror(equilibrium_state, solver_context):
    return pwo.diagnostics_state(equilibrium_state, solver_context)[3]


terms = [
    (lambda state, context: pwo.spectrum_state(state, context), 0.0, 1.0),  # must stay first
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.Hinge(opt.min_abs_iota, floor=IOTA_FLOOR), 0.0, 1000.0),
    (opt.Hinge(max_mirror, ceiling=MIRROR_LIMIT), 0.0, 100.0),
    (opt.Hinge(opt.axis_excursion, ceiling=0.25), 0.0, 100.0),
    (opt.Hinge(opt.edge_jacobian_ratio, floor=0.25), 0.0, 100.0),
    (opt.Hinge(opt.max_elongation, ceiling=9.0), 0.0, 10.0),
]
report = opt.EquilibriumReporter(
    ("pwO total", pwo.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("min iota", opt.min_abs_iota, ".4f"), ("max mirror", lambda s, c: jnp.max(max_mirror(s, c)), ".3f"))

### Seed every start ##########################################################

base = vj.VmecInput.from_file(INPUT_FILE)
pool = []
for modes in STARTS:
    rbc, zbs = base.rbc.copy(), base.zbs.copy()
    for m, n, amplitude in modes:
        rbc[base.ntor + n, m] -= amplitude
        zbs[base.ntor + n, m] += amplitude
    inp = replace(base, rbc=rbc, zbs=zbs, **LADDER_GRID).change_resolution(
        mpol=MPOL, ntor=MPOL, ntheta=2 * MPOL + 6, nzeta=2 * MPOL + 4)
    equilibrium = opt.solve_equilibrium(inp)
    if not equilibrium.result.converged:
        print(f"start {modes}: no converged seed equilibrium, dropped")
        continue
    problem = opt.VmecProblem.from_tuples(
        inp, terms, max_mode=MAX_MODES[-1], use_ess=True, ess_alpha=1.2, restart_from=equilibrium)
    pool.append(dict(modes=modes, inp=inp, equilibrium=equilibrium, problem=problem, x=problem.x0,
                     map=dict(MAP_PARAMETERS), cost=np.inf, nfev=0))

### Run the ladder, dropping the worst starts after every stage ################

for max_mode, max_nfev, keep in zip(MAX_MODES, MAX_NFEV, KEEP):
    for start in pool:
        print(f"\n===== start {start['modes']}, max_mode = {max_mode} =====")
        stage = start["problem"].subproblem(max_mode=max_mode, x=start["x"])
        pwo.options.update(start["map"])
        joint = FreeMapProblem(stage, pwo, start["equilibrium"], start["map"], FREE_MAP, weight=PWO_WEIGHT)
        step = np.concatenate((PARAMETER_STEP * stage.scales, np.ones(len(FREE_MAP))))
        change = np.concatenate((MAX_PARAMETER_CHANGE * step[:joint.nx], np.full(len(FREE_MAP), np.inf)))
        result = least_squares(
            joint.residual, joint.x0, jac=joint.residual_jac, x_scale=step, max_nfev=max_nfev,
            bounds=(joint.x0 - change, joint.x0 + change), ftol=1e-6, xtol=1e-10)
        start["x"] = stage.embed(result.x[:joint.nx])
        start["map"] = joint.values(result.x)
        start["inp"] = start["problem"].input_from_x(start["x"])
        start["equilibrium"] = start["problem"].equilibrium_from_x(start["x"])
        start["cost"], start["nfev"] = result.cost, start["nfev"] + result.nfev
        print(f"cost {result.cost:.4e} after {start['nfev']} evaluations")
    pool = sorted(pool, key=lambda start: start["cost"])[:keep]
    print(f"\nkept after max_mode {max_mode}: {[(start['modes'], float(start['cost'])) for start in pool]}")
best = pool[0]
pwo.options.update(best["map"])

### Check the winner at the final resolution ###################################

final_equilibrium = best["equilibrium"]
for ns in FINAL_NS:
    final_input = replace(best["inp"], ns_array=np.array([ns]), ftol_array=np.array([FINAL_FTOL]),
                          niter_array=np.array([30000]))
    final_equilibrium = opt.solve_equilibrium(
        final_input, initial_state=final_equilibrium.solution, verbose=not ci_smoke,
        raise_on_max_iterations=True)
report(f"final, start {best['modes']}", final_equilibrium)
input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")

#!/usr/bin/env python
"""One start of the multistart study: the whole mode ladder from one deterministic perturbation.

START indexes STARTS, a deterministic grid of one- and two-mode boundary perturbations
(rotating ellipse of either sign and three amplitudes, and the 0.05 ellipse plus an m = 2
helical mode). The script runs the family's mode ladder with plain SciPy least squares, records
the cost, residual evaluations and wall time after every stage, then re-solves the result on
the final grid (ns 71, ftol 1e-13) and writes a JSON record and the WOUT.
Best-of-K screening keeps the start with the lowest stage-1 cost and reads the rest of its
ladder from this record, so one process per start gives the single-start and the multistart
numbers. Run it from outside the repository root, one process per (family, start)::

    FAMILY=pwO START=3 python benchmarks/optimizer_multistart.py
"""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

FAMILY = os.environ.get("FAMILY", "pwO")     # QA, pwO or QI
START = int(os.environ.get("START", "0"))
OUTPUT = Path(os.environ.get("OUTPUT", "."))
SCREEN = os.environ.get("SCREEN") == "1"       # stage 1 only, pwO at 8 x 8 Boozer modes without oversampling
DATA = Path(vj.__file__).resolve().parents[1] / "examples" / "data"

# (m, n, amplitude): rbc(m, n) -= amplitude and zbs(m, n) += amplitude for every mode listed.
ELLIPSE = (1, -1)
STARTS = [[(*ELLIPSE, a)] for a in (0.03, 0.05, 0.08, -0.03, -0.05, -0.08)] + [
    [(*ELLIPSE, 0.05), (2, -1, b)] for b in (0.01, -0.01, 0.02, -0.02)]


def perturbed(base, modes):
    """The base boundary with every listed mode shifted."""
    rbc, zbs = base.rbc.copy(), base.zbs.copy()
    for m, n, amplitude in modes:
        rbc[base.ntor + n, m] -= amplitude
        zbs[base.ntor + n, m] += amplitude
    return replace(base, rbc=rbc, zbs=zbs)


if FAMILY in ("QA", "pwO"):
    nfp = 2 if FAMILY == "QA" else 3
    base = vj.VmecInput.from_file(DATA / f"input.minimal_seed_nfp{nfp}")
    ladder, budgets, parameter_step, prep = (
        ([1, 2], [10, 15], 0.02, dict(delt=0.5)) if FAMILY == "QA" else
        ([1, 2, 3], [15] * 3, 0.05, dict(delt=0.5, ns_array=np.array([35]), niter_array=np.array([5000]),
                                          ftol_array=np.array([1e-12]))))
else:
    base = vj.VmecInput.from_file(DATA / "input.QI_nfp2_initial")
    base = replace(base, ns_array=np.array([31]), ftol_array=np.array([1e-12]), niter_array=np.array([5500]))
    ladder, budgets, parameter_step, prep = [1, 3], [10, 15], None, dict(delt=0.5)
FINAL_NS = [45, 55, 71] if FAMILY == "pwO" else [71]  # restarted in steps: a large jump flips the Jacobian
final_grid = dict(ftol_array=np.array([1e-13]), niter_array=np.array([30000]))

### Objectives (as in the shipped examples) ###################################

if FAMILY == "QA":
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), helicity_m=1, helicity_n=0)
    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0),
             (lambda s, r: jnp.maximum(0.42 - opt.min_abs_iota(s, r), 0.0), 0.0, 10.0),
             (opt.magnetic_well, 0.01, 1.0)]
    metrics = dict(objective=qs.total, aspect=opt.aspect_ratio, min_iota=opt.min_abs_iota,
                   well=opt.magnetic_well)
elif FAMILY == "pwO":
    from vmex.core.pwo import FreeMapProblem, MappedPwOResidual
    surfaces = np.linspace(0.1, 0.9, 8)
    map_values, free_map = dict(a=0.3, b=0.0, c=-2.0, eta_min=2.8), ("a", "b", "c", "eta_min")
    pwo = MappedPwOResidual(surfaces, mboz=8 if SCREEN else 16, nboz=8 if SCREEN else 16, oversample=1 if SCREEN else 2, limit_map=True, **map_values)

    def max_mirror(state, runtime):
        """Full-surface mirror ratio on the pwO surfaces."""
        return pwo.diagnostics_state(state, runtime)[3]

    terms = [(lambda s, r: pwo.spectrum_state(s, r), 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0),
             (opt.Hinge(opt.min_abs_iota, floor=0.42), 0.0, 1000.0),
             (opt.Hinge(max_mirror, ceiling=0.16), 0.0, 100.0),
             (opt.Hinge(opt.axis_excursion, ceiling=0.25), 0.0, 100.0),
             (opt.Hinge(opt.edge_jacobian_ratio, floor=0.25), 0.0, 100.0),
             (opt.Hinge(opt.max_elongation, ceiling=9.0), 0.0, 10.0)]
    metrics = dict(objective=pwo.total, aspect=opt.aspect_ratio, min_iota=opt.min_abs_iota,
                   elongation=opt.max_elongation, axis=opt.axis_excursion,
                   mirror=lambda s, r: jnp.max(max_mirror(s, r)))
else:
    from vmex.core.qi import ConstructedQIResidual
    qi = ConstructedQIResidual(np.linspace(0.1, 1.0, 6), mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
    terms = [(qi, 0.0, 10.0), (opt.aspect_ratio, 8.0, 0.01),
             (lambda s, r: jnp.maximum(0.51 - opt.min_abs_iota(s, r), 0.0), 0.0, 10.0),
             (lambda s, r: jnp.maximum(opt.mirror_ratio(s, r) - 0.99 * 0.21, 0.0), 0.0, 1000.0),
             (lambda s, r: jnp.maximum(opt.max_elongation(s, r) - 8.0, 0.0), 0.0, 10.0)]
    metrics = dict(objective=qi.total, aspect=opt.aspect_ratio, min_iota=opt.min_abs_iota,
                   mirror=opt.mirror_ratio, elongation=opt.max_elongation)
report = opt.EquilibriumReporter(*[(name, fun, ".6e") for name, fun in metrics.items()], stream=None)

### Ladder ####################################################################

top = max(ladder)
mpol = max(top + 2, 5)
name = f"{FAMILY}_{START}" + ("_screen" if SCREEN else "")
if SCREEN:
    ladder, budgets = ladder[:1], budgets[:1]
record = dict(family=FAMILY, start=START, modes=STARTS[START], ok=False, stages=[])
began = time.time()
try:
    inp = replace(perturbed(base, STARTS[START]), **prep).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    equilibrium = opt.solve_equilibrium(inp)
    result = equilibrium.result
    record.update(seed_converged=bool(result.converged), seed_metrics=report("", equilibrium))
    problem = opt.VmecProblem.from_tuples(
        inp, terms, max_mode=top, use_ess=True, ess_alpha=1.2,
        **(dict(restart_from=equilibrium) if FAMILY != "QI" else {}))
    x = problem.x0
    for max_mode, max_nfev in zip(ladder, budgets):
        stage_began = time.time()
        stage = target = problem.subproblem(max_mode=max_mode, x=x)
        nx = stage.x0.size
        step = stage.scales if parameter_step is None else parameter_step * stage.scales
        limit = None if parameter_step is None else 5.0
        if FAMILY == "pwO":  # the map parameters are free, sigmoid-scaled and unbounded
            target = FreeMapProblem(stage, pwo, equilibrium, map_values, free_map, weight=1e3)
            step = np.concatenate((step, np.ones(len(free_map))))
            limit = np.concatenate((np.full(nx, 5.0), np.full(len(free_map), np.inf)))
        bounds = (-np.inf, np.inf) if limit is None else (target.x0 - limit * step, target.x0 + limit * step)
        result = least_squares(target.residual, target.x0, jac=target.residual_jac, x_scale=step,
                               max_nfev=max_nfev, bounds=bounds, ftol=1e-6, xtol=1e-10)
        if FAMILY == "pwO":
            map_values = target.values(result.x)
            pwo.options.update(map_values)
        x = stage.embed(result.x[:nx])
        inp, equilibrium = problem.input_from_x(x), problem.equilibrium_from_x(x)
        record["stages"].append(dict(max_mode=max_mode, nfev=int(result.nfev), cost=float(result.cost),
                                     wall=time.time() - stage_began, metrics=report("", equilibrium)))
    final = equilibrium
    for ns in [] if SCREEN else FINAL_NS:
        final_input = replace(inp, ns_array=np.array([ns]), **final_grid)
        final = opt.solve_equilibrium(final_input, initial_state=final.solution, raise_on_max_iterations=True)
    if not SCREEN:
        final_input.to_indata(str(OUTPUT / f"input.{name}"))
        vj.write_wout(str(OUTPUT / f"wout_{name}.nc"), final.wout)
    record.update(ok=True, final=report("", final))
except Exception as error:  # noqa: BLE001 - the table counts failures
    record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
record["wall"] = time.time() - began
(OUTPUT / f"{name}.json").write_text(json.dumps(record, indent=1, default=float))
print(name, json.dumps({k: record.get(k) for k in ("ok", "final", "wall", "error")}, default=float))

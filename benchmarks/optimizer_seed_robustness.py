#!/usr/bin/env python
"""Seed robustness of the boundary-optimization ladders (QA, pwO, QI).

Each seed perturbs the example's start boundary at random, runs the example's mode ladder with
the chosen stage driver, re-solves the optimized boundary on a finer radial grid and writes one
JSON record per (family, seed, driver): final metrics on that grid, residual evaluations,
rejected trials and wall time. Seed 0 is the unperturbed example; seeds above 9 are the stress
set (random-sign rotating ellipse down to nearly circular, more noise). A perturbed boundary
whose cold start has no converged equilibrium is redrawn, as a user would re-seed.

Run it from outside the repository root, one process per (family, seed, driver)::

    FAMILY=QA SEED=3 DRIVER=run_stage python benchmarks/optimizer_seed_robustness.py
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

FAMILY = os.environ.get("FAMILY", "QA")        # QA, pwO or QI
SEED = int(os.environ.get("SEED", "1"))
DRIVER = os.environ.get("DRIVER", "run_stage")  # run_stage, or scipy (the call the examples used to make)
OUTPUT = Path(os.environ.get("OUTPUT", "."))

DATA = Path(vj.__file__).resolve().parents[1] / "examples" / "data"
rng = np.random.default_rng(1000 + SEED)


def perturb(base, sigma, modes=2, ellipse=0.0):
    """Gaussian noise on RBC/ZBS for 1 <= m <= modes and |n| <= modes, plus a rotating ellipse."""
    rbc, zbs, ntor = base.rbc.copy(), base.zbs.copy(), base.ntor
    for m in range(1, modes + 1):
        for n in range(-modes, modes + 1):
            rbc[ntor + n, m] += sigma * rng.standard_normal()
            zbs[ntor + n, m] += sigma * rng.standard_normal()
    rbc[ntor - 1, 1] -= ellipse
    zbs[ntor - 1, 1] += ellipse
    return replace(base, rbc=rbc, zbs=zbs)


def seeded(base):
    """Seed 0: the example start; 1-9: moderate noise; 10 and up: the stress set."""
    if SEED == 0:
        return replace(base, rbc=base.rbc.copy(), zbs=base.zbs.copy())
    if SEED < 10:
        return perturb(base, 0.01, ellipse=0.05 * rng.uniform(0.4, 1.6))
    return perturb(base, 0.007, ellipse=rng.choice([-1, 1]) * rng.uniform(0.005, 0.12))


if FAMILY in ("QA", "pwO"):
    nfp = 2 if FAMILY == "QA" else 3
    base = vj.VmecInput.from_file(DATA / f"input.minimal_seed_nfp{nfp}")
    ladder, budgets, parameter_step, prep = (
        ([1, 2], [10, 15], 0.02, dict(delt=0.5)) if FAMILY == "QA" else
        ([1, 2, 3], [15] * 3, 0.05, dict(delt=0.5, ns_array=np.array([35]), niter_array=np.array([5000]),
                                          ftol_array=np.array([1e-12]))))
    final_grid = dict(ns_array=np.array([71 if FAMILY == "QA" else 55]),
                      ftol_array=np.array([1e-14 if FAMILY == "QA" else 1e-10]),
                      niter_array=np.array([8000 if FAMILY == "QA" else 30000]))
else:
    base = vj.VmecInput.from_file(DATA / "input.QI_nfp2_initial")
    base = replace(base, ns_array=np.array([31]), ftol_array=np.array([1e-12]), niter_array=np.array([5500]))
    ladder, budgets, parameter_step, prep = [1, 3], [10, 15], None, dict(delt=0.5)
    final_grid = dict(ns_array=np.array([71]), ftol_array=np.array([1e-14]), niter_array=np.array([8000]))

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
    pwo = MappedPwOResidual(surfaces, mboz=16, nboz=16, oversample=2, limit_map=True, **map_values)

    def max_mirror(state, runtime):
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

### Seed ######################################################################

mpol = max(max(ladder) + 2, 5)


def resolved(inp):
    return replace(inp, **prep).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)


def converged(equilibrium):
    result = equilibrium.result
    fsq = float(result.fsqr) + float(result.fsqz) + float(result.fsql)
    return bool(result.converged) or fsq <= 100 * float(equilibrium.inp.ftol_array[-1])


record = dict(family=FAMILY, seed=SEED, driver=DRIVER, ok=False, stages=[])
start = time.time()
try:
    for redraw in range(30):
        inp = resolved(seeded(base))
        try:
            equilibrium = opt.solve_equilibrium(inp)
        except Exception as error:  # noqa: BLE001 - a start boundary without an initial Jacobian
            if "JACOBIAN" not in str(error).upper() or SEED == 0:
                raise
            continue
        if converged(equilibrium) or SEED == 0:
            break
    record.update(redraws=redraw, seed_metrics=report("", equilibrium))
    problem = opt.VmecProblem.from_tuples(
        inp, terms, max_mode=max(ladder), use_ess=True, ess_alpha=1.2,
        **(dict(restart_from=equilibrium) if FAMILY != "QI" else {}))
    x = problem.x0
    rejected = 0
    for max_mode, max_nfev in zip(ladder, budgets):
        began = time.time()
        stage = target = problem.subproblem(max_mode=max_mode, x=x)
        nx = stage.x0.size
        step = stage.scales if parameter_step is None else parameter_step * stage.scales
        limit = None if parameter_step is None else 5.0
        if FAMILY == "pwO":  # the map parameters are free, sigmoid-scaled and unbounded
            target = FreeMapProblem(stage, pwo, equilibrium, map_values, free_map, weight=1e3)
            step = np.concatenate((step, np.ones(len(free_map))))
            limit = np.concatenate((np.full(nx, 5.0), np.full(len(free_map), np.inf)))
        solve = target.residual

        def residual(point, solve=solve):
            global rejected
            value = solve(point)
            rejected += int(np.max(np.abs(value)) >= 1e5)  # the failed-trial penalty rows
            return value

        if DRIVER == "run_stage":
            target.residual = residual
            result = opt.run_stage(target, max_nfev=2 * max_nfev, step=step, change=limit)
        else:
            bounds = (-np.inf, np.inf) if limit is None else (
                target.x0 - limit * step, target.x0 + limit * step)
            result = least_squares(residual, target.x0, jac=target.residual_jac, x_scale=step,
                                   max_nfev=max_nfev, bounds=bounds, ftol=1e-6, xtol=1e-10)
        if FAMILY == "pwO":
            map_values = target.values(result.x)
            pwo.options.update(map_values)
        x = stage.embed(result.x[:nx])
        inp, equilibrium = problem.input_from_x(x), problem.equilibrium_from_x(x)
        record["stages"].append(dict(
            max_mode=max_mode, nfev=int(result.nfev), cost=float(result.cost), rejected=rejected,
            wall=time.time() - began, metrics=report("", equilibrium)))
    final = opt.solve_equilibrium(
        replace(inp, **final_grid), initial_state=equilibrium.solution, raise_on_max_iterations=True)
    record.update(ok=True, final=report("", final))
except Exception as error:  # noqa: BLE001 - the table counts failures
    record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
record["wall"] = time.time() - start
record["cpu"] = time.process_time()
path = OUTPUT / f"{FAMILY}_{SEED}_{DRIVER}.json"
path.write_text(json.dumps(record, indent=1, default=float))
print(path, json.dumps({k: record.get(k) for k in ("ok", "final", "wall", "error")}, default=float))

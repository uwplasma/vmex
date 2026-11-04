#!/usr/bin/env python
"""Optimize a QA boundary with a differentiable heat bound.

An implicit equilibrium adjoint supplies the gradient. A re-solved finite
difference checks it, and finer endpoint evaluations test the improvement.
Set VMEX_HEAT_BOUND_MODEL=coulomb for the full-Coulomb stellarator proxy;
the default is the bounce-averaged Lorentz reference. Each has its stated orbit
model; see the neoclassical section of docs/explanation/confinement.rst.
"""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.neoclassical import _state_stencils, reference_pitch_grid_from_boozer

# Seed, operating points and boundary variables:
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.LandremanPaul2021_QA_lowres"
OUTPUT = Path(os.environ.get("VMEX_HEAT_BOUNDS_OUTPUT", "."))
SURFACES = (.5,)
TEMPERATURE_EV = 1000.
MODEL = os.environ.get("VMEX_HEAT_BOUND_MODEL", "lorentz")
COLLISION_FREQUENCIES = (100., 1000.)          # thermal Lorentz rates, s^-1
COLLISION_STRENGTHS = (1.e4, 1.e5)            # Coulomb Gamma*n/(m²*vth³), s^-1
RADIAL_ELECTRIC_FIELDS = (-250., 0., 250.)     # Er = -dPhi/dr, V/m
ACTIVE_MODES = ("RBC(1,0)", "RBC(-1,1)", "ZBS(-1,1)")
SEED_PERTURBATION = .005                     # metres, RBC(1,0)
PARAMETER_STEP, MAX_PARAMETER_CHANGE = .002, 3.
MAXITER = 8

# Trial and verification resolutions; these are independent error controls:
PITCH_CELLS, ENERGY_POINTS = 32, 6
VERIFY_PITCH_CELLS, VERIFY_ENERGY_POINTS = 96, 12
FINAL_NS, FINAL_FTOL = 26, 1.e-12
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER = 2
    PITCH_CELLS, ENERGY_POINTS = 16, 4
    VERIFY_PITCH_CELLS, VERIFY_ENERGY_POINTS = 32, 6

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium and objective ######################################
OUTPUT.mkdir(parents=True, exist_ok=True)
inp = vj.VmecInput.from_file(INPUT_FILE).change_resolution(
    mpol=6, ntor=6, ntheta=20, nzeta=20)
rbc = inp.rbc.copy()
rbc[inp.ntor + 1, 0] += SEED_PERTURBATION
inp = replace(inp, rbc=rbc, ns_array=np.array([14]),
              ftol_array=np.array([1.e-11]), niter_array=np.array([4000]))
seed = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
shared = dict(temperature_ev=TEMPERATURE_EV,
              radial_electric_fields=RADIAL_ELECTRIC_FIELDS)
if MODEL == "coulomb":
    shared.update(collision_strengths=COLLISION_STRENGTHS)
    term = vj.ConservingHeatBound(SURFACES, geometry_order=(3, 1), fit_grid=64,
                                 mboz=6, nboz=6, **shared)
elif MODEL == "lorentz":
    booz, _ = _state_stencils(seed.state, seed.runtime, SURFACES, 6, 6, 2)
    grid = reference_pitch_grid_from_boozer(**booz, ncells=PITCH_CELLS, helicity_m=1, helicity_n=0)
    shared.update(collision_frequencies=COLLISION_FREQUENCIES,
                  helicity_m=1, helicity_n=0, well_center=grid["well_center"])
    term = vj.NeoclassicalHeatBound(
        SURFACES, pitch_fractions=np.linspace(0., 1., PITCH_CELLS + 1),
        points_per_period=128, quadrature_order=32, nenergy=ENERGY_POINTS,
        mboz=6, nboz=6, nalpha=13, **shared)
else:
    raise ValueError("VMEX_HEAT_BOUND_MODEL must be lorentz or coulomb")
initial = term.compute_state(seed.state, seed.runtime)
if not np.all(initial["valid"]):
    raise RuntimeError("invalid seed reference geometry")
term.scale = float(np.max(initial["upper"]))
aspect_seed = float(opt.aspect_ratio(seed.state, seed.runtime))
iota_seed = float(opt.mean_iota(seed.state, seed.runtime))


def loss(state, runtime):
    """Upper response plus dimensionless aspect and transform constraints."""
    return (term.total_state(state, runtime)
            + 20 * (opt.aspect_ratio(state, runtime) / aspect_seed - 1)**2
            + 20 * (opt.mean_iota(state, runtime) / iota_seed - 1)**2)


problem = opt.VmecProblem.from_loss(
    inp, loss, max_mode=1, vary_major_radius=False, use_ess=False,
    restart_from=seed, forward_ftol=1.e-11, forward_max_iterations=4000,
    progress=True, evaluation_progress=False).subproblem(ACTIVE_MODES)
x0 = problem.x0.copy()


def value_and_gradient(y):
    """Scale boundary variables in metres before the L-BFGS-B step."""
    value, gradient = problem.value_and_grad(x0 + PARAMETER_STEP * y)
    print(f"objective={float(value):.9g}", flush=True)
    return value, PARAMETER_STEP * gradient


### Check the derivative and optimize #########################################
seed_value, seed_gradient = value_and_gradient(np.zeros_like(x0))
direction = np.array([1., -.7, .4])
direction /= np.linalg.norm(direction)
fd_step = 2.e-3
fd = float((problem.fun(x0 + PARAMETER_STEP * fd_step * direction)
            - problem.fun(x0 - PARAMETER_STEP * fd_step * direction)) / (2 * fd_step))
ad = float(np.dot(seed_gradient, direction))
gradient_error = abs(ad - fd) / max(abs(ad), abs(fd), 1.e-14)
if not np.isfinite(gradient_error) or gradient_error > .03:
    raise RuntimeError(f"directional derivative discrepancy: {gradient_error:.3%}")
seed = problem.equilibrium_from_x(x0)
result = minimize(
    value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
    bounds=[(-MAX_PARAMETER_CHANGE, MAX_PARAMETER_CHANGE)] * len(x0),
    options={"maxiter": MAXITER, "gtol": 1.e-6, "ftol": 1.e-10, "maxls": 15})
candidate = problem.equilibrium_from_x(x0 + PARAMETER_STEP * result.x)
if not np.isfinite(result.fun) or float(result.fun) > seed_value:
    raise RuntimeError("the reference objective did not improve")

### Verify matched endpoints and save #########################################
if MODEL == "coulomb":
    verify = vj.ConservingHeatBound(SURFACES, geometry_order=(6, 4), fit_grid=128,
        integration_order=40, mboz=6 if ci_smoke else 10, nboz=6 if ci_smoke else 10,
        scale=term.scale, **shared)
else:
    verify = vj.NeoclassicalHeatBound(
        SURFACES, pitch_fractions=np.linspace(0., 1., VERIFY_PITCH_CELLS + 1),
        points_per_period=1024, quadrature_order=256, nenergy=VERIFY_ENERGY_POINTS,
        mboz=6 if ci_smoke else 10, nboz=6 if ci_smoke else 10,
        nalpha=13 if ci_smoke else 21, scale=term.scale, **shared)
records = {}
for label, equilibrium in (("seed", seed), ("candidate", candidate)):
    refined_input = replace(equilibrium.inp, ns_array=np.array([FINAL_NS]),
                            ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([5000]))
    refined = opt.solve_equilibrium(refined_input, initial_state=equilibrium.solution,
                                   raise_on_max_iterations=True)
    response = verify.compute_state(refined.state, refined.runtime)
    if not np.all(response["valid"]):
        raise RuntimeError(f"{label}: invalid refined reference geometry")
    records[label] = {key: np.asarray(response[key]).tolist()
                      for key in ("lower", "upper", "relative_gap")}
    diagnostic = (response["geometry_fit_error_sampled"] if MODEL == "coulomb" else
                  response["geometry"]["reference_perturbation_well_max"])
    records[label]["geometry_diagnostic"] = np.asarray(diagnostic).tolist()
    vj.write_wout(OUTPUT / f"wout_optimization_{label}.nc", refined.wout)
    if label == "candidate":
        refined.inp.to_indata(OUTPUT / "input.QA_heat_bound_optimized")

seed_lower = float(np.max(records["seed"]["lower"]))
seed_upper = float(np.max(records["seed"]["upper"]))
candidate_upper = float(np.max(records["candidate"]["upper"]))
if candidate_upper >= (seed_upper if MODEL == "coulomb" else seed_lower):
    raise RuntimeError("refined upper objective did not decrease" if MODEL == "coulomb" else
                       "refined intervals do not establish an improved worst response")
records.update(
    model=("single-species full Coulomb; incompressible ExB; no tangential magnetic drift" if MODEL == "coulomb"
           else "single-reference-well Lorentz; positive discrete speed rule"),
    gradient_check={"ad": ad, "fd": fd, "relative_error": gradient_error},
    optimizer={"iterations": int(result.nit), "converged": bool(result.success),
               "message": str(result.message), "initial_loss": float(seed_value),
               "final_loss": float(result.fun)},
    upper_bound_decrease=1 - candidate_upper / seed_upper,
    guaranteed_worst_response_decrease=(1 - candidate_upper / seed_lower if candidate_upper < seed_lower else None))
(OUTPUT / "neoclassical_heat_bound_optimization.json").write_text(
    json.dumps(records, indent=2, allow_nan=False) + "\n")
print(f"Upper-bound decrease: {1 - candidate_upper / seed_upper:.2%}")
print(f"Separated-interval response decrease: {records['guaranteed_worst_response_decrease']}")
print(f"Optimizer: {result.message}; fuller kinetic acceptance remains separate.")

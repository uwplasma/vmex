#!/usr/bin/env python
r"""Single-stage free-boundary optimization at finite beta with VMEX + ESSOS.

``single_stage_free_boundary_optimization.py`` at volume-average beta 0.5%:
the same seed, targets, weights, coils and budget, with a pressure profile
p ~ (1 - s) and no net toroidal current (NCURR = 1, CURTOR = 0, AC = 0). The
coils are the only variables; every trial runs a NESTOR free-boundary solve
in their Biot-Savart field and VMEX differentiates it implicitly. The
free-boundary solve balances the plasma pressure against the coil field
itself, so neither a normal-field nor a pressure-balance term is needed.

The objective is that of ``single_stage_free_boundary_optimization.py`` plus

    (1/2) (<beta> / TARGET_BETA - 1)^2

which holds beta at its target while the coils, and so the field strength and
the plasma volume, change.

The coils start as in the zero-beta example: fitted to the fixed-boundary
seed with no equilibrium solve, the enclosed toroidal flux read off the
fitted field, and the pressure scaled so that the fixed-boundary seed sits at
the target beta. One free-boundary solve then checks that the fitted coils
confine that plasma before any optimization starts. Self-consistent bootstrap
current and current-profile optimization are not part of this example.

Run it with ``VMEX_EXAMPLES_CI=1`` for a short smoke pass that reports and
exits 0. Otherwise a run that misses a target says which and exits 1.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core import implicit as im

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import SurfaceRZFourier, surfacerzfourier_from_boundary

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.05, 1.0, 6)

# Seed boundary the coils are fitted to: the rotating ellipse of
# single_stage_optimization.py.
SEED_MINOR_RADIUS = 0.195
SEED_ELLIPSE = 0.10

# Volume-average beta, pressure profile p(s) = PRES_SCALE * (1 - s) as VMEC
# power-series coefficients, and no net toroidal current:
TARGET_BETA = 0.005
PRESSURE_PROFILE = [1.0, -1.0]
BETA_TOLERANCE = 0.1              # relative, for the final check

# Targets the finished design is checked against:
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile
ASPECT_LIMIT = 6.0                # maximum aspect ratio
COIL_SURFACE_DISTANCE_LIMIT = 0.15
COIL_DISTANCE_LIMIT = 0.17
CURVATURE_LIMIT = 7.0

# Values the optimizer is given instead, tighter than the limits above because
# a quadratic penalty settles just inside the threshold it is handed. The
# normal-field value is used only by the coil pre-fit: on a free boundary
# B.n = 0 holds by construction.
IOTA_CONSTRAINT = 0.43
ASPECT_CONSTRAINT = 5.97
NORMAL_FIELD_CONSTRAINT = 0.008
CURVATURE_OBJECTIVE_LIMIT = 6.9
COIL_DISTANCE_CONSTRAINT = 0.19
COIL_SURFACE_DISTANCE_CONSTRAINT = 0.16

# Coils: number of unique shapes, Fourier order, and the circle they start on.
N_COILS = 3
COIL_ORDER = 4
COIL_MAJOR_RADIUS = 1.0
COIL_MINOR_RADIUS = 0.5
COIL_CURRENT = 2.7e5
N_SEGMENTS = 64
STELLSYM = True

# Weights, and the longest coil the optimizer may build. The length term is
# one-sided: coils shorter than LENGTH_TARGET cost nothing, so it never pulls
# against quasisymmetry or the normal field.
LENGTH_TARGET = 5.5
LENGTH_WEIGHT = 0.5
CURVATURE_WEIGHT = 10.0
COIL_DISTANCE_WEIGHT = 1.0e4
COIL_SURFACE_DISTANCE_WEIGHT = 1.0e4
CONSTRAINT_WEIGHT = 1.0e3

# Bound on each scaled coil variable:
PARAMETER_BOUND = 3.0

# Free-boundary solve used on every trial: radial surfaces, force tolerance and
# iteration cap.
NS = 31
FTOL = 1.0e-10
NITER = 4000

# Budgets. One trial is one free-boundary solve plus one adjoint.
MAXITER = 50
MAX_TRIALS = 100
COIL_FIT_MAXITER = 200            # coil-only pre-fit, no equilibrium solves

# Surface grid the coil terms use:
NPHI, NTHETA = 37, 32

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 1, 2, 2
    N_SEGMENTS, COIL_ORDER, NPHI, NTHETA = 24, 2, 8, 8
    NS, FTOL = 12, 1.0e-8

###############################################################################
# End of input parameters.
###############################################################################

started = time.perf_counter()

###############################################################################
### Set up the seed boundary and fit the coils to it ##########################
###############################################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = SEED_MINOR_RADIUS
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_ELLIPSE, -SEED_ELLIPSE
am = np.zeros(21)
am[:len(PRESSURE_PROFILE)] = PRESSURE_PROFILE
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am,
              pres_scale=1.0, ncurr=1, curtor=0.0, ac=np.zeros(21)).change_resolution(
    mpol=5, ntor=5, ntheta=16, nzeta=14)
surface_seed = surfacerzfourier_from_boundary(
    jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), NFP, nphi=NPHI, ntheta=NTHETA)

curves0 = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
coils_circular = Coils(curves0, np.full(N_COILS, COIL_CURRENT))
x_circular = np.asarray(curves0.dofs).ravel()
scales = np.full(x_circular.size, 0.05)


def normal_field_rms(coils, surface):
    """Area-weighted RMS of B.n/|B| on a surface."""
    field = BiotSavart(coils)
    B = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    normal_field = jnp.sum(B * surface.unitnormal, axis=2) / jnp.linalg.norm(B, axis=2)
    weights = surface.area_element / jnp.sum(surface.area_element)
    return jnp.sqrt(jnp.sum(weights * normal_field**2))


def hinge(constraints):
    """Quadratic penalty for constraints written as c >= 0, normalized by their limits."""
    return 0.5 * CONSTRAINT_WEIGHT * jnp.sum(jnp.maximum(-constraints, 0.0)**2)


def coil_costs(coils, surface):
    """Coil length, curvature, separation and clearance to ``surface``."""
    length = jnp.sqrt(LENGTH_WEIGHT) * jnp.maximum(coils.length[:N_COILS] - LENGTH_TARGET, 0.0)
    curvature = jnp.sqrt(CURVATURE_WEIGHT) * jnp.maximum(
        coils.curvature[:N_COILS] - CURVATURE_OBJECTIVE_LIMIT, 0.0)
    return jnp.asarray([
        0.5 * jnp.vdot(length, length),
        0.5 * jnp.vdot(curvature, curvature),
        0.5 * COIL_DISTANCE_WEIGHT * loss_coil_separation(
            coils, COIL_DISTANCE_CONSTRAINT, block_size=32),
        0.5 * COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
            coils, surface, COIL_SURFACE_DISTANCE_CONSTRAINT, block_size=32),
    ])


def coils_from_x(x):
    return coils_circular.with_dofs(jnp.concatenate((x, coils_circular.dofs_currents)))


def coil_fit_objective(u):
    """The coil half of single_stage_optimization.py with the boundary frozen.

    At 0.5% beta the plasma's own normal field is small, so the vacuum
    condition B_coils.n = 0 is a good enough target for a starting point.
    """
    coils = coils_from_x(jnp.asarray(x_circular) + jnp.asarray(scales) * u)
    normal_field = 1.0 - normal_field_rms(coils, surface_seed) / NORMAL_FIELD_CONSTRAINT
    return jnp.sum(coil_costs(coils, surface_seed)) + hinge(normal_field[None])


print("Running single_stage_free_boundary_optimization_finite_beta.py")
coil_fit_value_and_grad = jax.jit(jax.value_and_grad(coil_fit_objective))
coil_fit = minimize(
    lambda u: tuple(map(np.asarray, coil_fit_value_and_grad(jnp.asarray(u)))),
    np.zeros_like(x_circular), jac=True, method="L-BFGS-B",
    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x_circular.size,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
x0 = x_circular + scales * coil_fit.x
coils0 = coils_from_x(jnp.asarray(x0))
print(f"[coil fit] {coil_fit.nit} L-BFGS-B iterations, no equilibrium solves: B.n/B RMS "
      f"= {100 * float(normal_field_rms(coils0, surface_seed)):.3f}% on the seed")

###############################################################################
### Check that the fitted coils confine the seed ##############################
###############################################################################

# The plasma a free-boundary solve finds depends on the toroidal flux it is
# told to enclose. Take it from the fitted field itself, through the seed's
# phi = 0 cross-section (Gauss-Legendre in radius, uniform in angle), so the
# seed boundary is the one the coils are asked to confine.
rho, rho_weights = np.polynomial.legendre.leggauss(24)
rho, rho_weights = 0.5 * (rho + 1.0), 0.5 * rho_weights
theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
# At phi = 0 every toroidal harmonic has phase m*theta, so sum over n.
m = np.arange(inp.rbc.shape[1])
rbc_phi0, zbs_phi0 = inp.rbc.sum(axis=0), inp.zbs.sum(axis=0)
R_edge = np.cos(np.outer(theta, m)) @ rbc_phi0
Z_edge = np.sin(np.outer(theta, m)) @ zbs_phi0
dR_edge = -np.sin(np.outer(theta, m)) @ (m * rbc_phi0)
dZ_edge = np.cos(np.outer(theta, m)) @ (m * zbs_phi0)
R_center = rbc_phi0[0]
R = R_center + rho[:, None] * (R_edge - R_center)
Z = rho[:, None] * Z_edge
area_element = rho[:, None] * ((R_edge - R_center) * dZ_edge - Z_edge * dR_edge)
points = np.stack([R, np.zeros_like(R), Z], axis=-1).reshape(-1, 3)
field0 = BiotSavart(coils0)
B_phi = np.asarray(jax.vmap(field0.B)(jnp.asarray(points)))[:, 1].reshape(R.shape)
toroidal_flux = float(np.sum(rho_weights[:, None] * B_phi * area_element)
                      * (2.0 * np.pi / theta.size))
# VMEC counts the toroidal angle clockwise seen from above, so its PHIEDGE
# has the opposite sign to the flux measured along +phi.
inp = replace(inp, phiedge=-toroidal_flux)
# Beta is proportional to the pressure at fixed boundary and flux, so one
# fixed-boundary solve of the seed at PRES_SCALE = 1 gives the scale.
beta_at_unit_pressure = float(opt.solve_equilibrium(inp).wout.betatotal)
inp = replace(inp, pres_scale=TARGET_BETA / beta_at_unit_pressure, lfreeb=True,
              mgrid_file="direct ESSOS field", ns_array=np.array([NS]),
              ftol_array=np.array([FTOL]), niter_array=np.array([NITER]))
print(f"[flux] PHIEDGE = {inp.phiedge:.6f} Wb from the fitted coils; "
      f"PRES_SCALE = {inp.pres_scale:.1f} Pa for beta {100 * TARGET_BETA:.2f}%")


def field_from_u(u):
    return BiotSavart(coils_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u))


params = im.params_from_input(inp)
config = vj.make_free_boundary_config(
    inp, field0, ns=NS, ftol=FTOL, max_iterations=NITER,
    adjoint_tol=1.0e-8, adjoint_solver="boundary_schur",
    field_from_parameters=field_from_u)
solver_context = im.runtime_from_params(params, config.implicit)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)


def boundary_surface(equilibrium_state, nphi=NPHI, ntheta=NTHETA):
    """The free boundary of a solved state, as an ESSOS surface."""
    rmnc, _, _, zmns = im._edge_physical(equilibrium_state, solver_context)
    modes = solver_context.modes
    ntor = int(np.max(np.abs(modes.n)))
    rows = jnp.asarray(np.asarray(modes.n) + ntor)
    cols = jnp.asarray(np.asarray(modes.m))
    shape = (2 * ntor + 1, int(np.max(modes.m)) + 1)
    rbc = jnp.zeros(shape).at[rows, cols].set(rmnc)
    zbs = jnp.zeros(shape).at[rows, cols].set(zmns)
    return surfacerzfourier_from_boundary(rbc, zbs, NFP, nphi=nphi, ntheta=ntheta)


def plasma_values(equilibrium_state):
    return {"QA total": float(qs.total_state(equilibrium_state, solver_context)),
            "beta": float(opt.volume_average_beta(equilibrium_state, solver_context)),
            "aspect": float(opt.aspect_ratio(equilibrium_state, solver_context)),
            "mean iota": float(opt.mean_iota(equilibrium_state, solver_context)),
            "min |iota|": float(opt.min_abs_iota(equilibrium_state, solver_context))}


state0, status0, _, _ = vj.solve_free_boundary_implicit_status(
    params, jnp.zeros_like(x0), config)
if int(status0) != 0:
    raise SystemExit("The fitted coils do not hold a converged free boundary "
                     f"(status {int(status0)}); raise COIL_FIT_MAXITER.")
free_seed = boundary_surface(state0)
boundary_shift = float(jnp.max(jnp.linalg.norm(free_seed.gamma - surface_seed.gamma, axis=-1)))
seed_values = plasma_values(state0)
print("[seed] " + ", ".join(f"{k} = {v:.4g}" for k, v in seed_values.items())
      + f"; free boundary within {boundary_shift:.4f} m of the fitted seed")

###############################################################################
### Set up the objective ######################################################
###############################################################################

COIL_TERMS = ("coil length", "coil curvature", "coil separation", "coil-surface separation")


@jax.jit
def accepted_terms(equilibrium_state, u):
    """Every term after the solve, compiled once and reused by each trial."""
    qs_rows = qs.residuals_state(equilibrium_state, solver_context)
    beta_row = opt.volume_average_beta(equilibrium_state, solver_context) / TARGET_BETA - 1.0
    penalty = hinge(jnp.stack([
        opt.min_abs_iota(equilibrium_state, solver_context) / IOTA_CONSTRAINT - 1.0,
        1.0 - opt.aspect_ratio(equilibrium_state, solver_context) / ASPECT_CONSTRAINT]))
    coils = coils_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u)
    costs = coil_costs(coils, boundary_surface(equilibrium_state))
    value = 0.5 * jnp.vdot(qs_rows, qs_rows) + 0.5 * beta_row**2 + penalty + jnp.sum(costs)
    return value, (qs_rows, beta_row, penalty, costs)


def objective(u):
    """One free-boundary solve and its adjoint; the solve runs on the host."""
    equilibrium_state, status, _, _ = vj.solve_free_boundary_implicit_status(params, u, config)

    def accepted(_):
        value, aux = accepted_terms(equilibrium_state, u)
        return value, (*aux, status)

    def rejected(_):
        # A smooth, finite wall lets SciPy backtrack after a failed trial. Its
        # derivative is explicit here; the failed equilibrium contributes zero.
        aux = accepted_terms(equilibrium_state, u)[1]
        wall = 1.0e3 * (1.0 + jnp.sqrt(1.0e-12 + jnp.vdot(u, u)))**2
        return wall, (*map(jnp.zeros_like, aux), status)

    return jax.lax.cond(status == 0, accepted, rejected, operand=None)


value_and_grad_jax = jax.value_and_grad(objective, has_aux=True)
monitor = opt.OptimizationMonitor()
counts = {"trials": 0, "rejected": 0}
# SciPy returns its last point, which can be a rejected trial carrying the
# wall value; the result is the best trial whose free boundary converged.
best = {"value": np.inf, "u": None}


def value_and_grad(u):
    counts["trials"] += 1
    (value, (qs_rows, beta_row, penalty, costs, status)), gradient = \
        value_and_grad_jax(jnp.asarray(u))
    qs_rows = np.asarray(qs_rows)
    terms = {"quasisymmetry": 0.5 * float(qs_rows @ qs_rows),
             "beta": 0.5 * float(beta_row)**2,
             "iota and aspect penalty": float(penalty)}
    terms.update(zip(COIL_TERMS, map(float, np.asarray(costs))))
    counts["rejected"] += int(status) != 0
    if int(status) == 0 and float(value) < best["value"]:
        best.update(value=float(value), u=np.array(u, dtype=float))
    terms["rejected trial"] = float(value) if int(status) else 0.0
    return monitor.cache_evaluation(u, value, gradient, terms)


###############################################################################
### Run the optimization ######################################################
###############################################################################

print(f"True NESTOR free boundary + ESSOS at beta {100 * TARGET_BETA:.2f}%: "
      f"{x0.size} coil variables, "
      "no boundary variables and no mgrid file")
free_problem = vj.FunctionProblem.from_functions(
    np.zeros_like(x0), value_and_grad=value_and_grad, evaluation_progress=not ci_smoke)
first = free_problem.compile_value_and_gradient(progress=not ci_smoke, report_interval=10.0)
initial_value = float(first.value)
result = minimize(free_problem.value_and_grad, np.zeros_like(x0), jac=True,
                  method="L-BFGS-B", callback=monitor,
                  bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size,
                  options={"maxiter": MAXITER, "maxfun": MAX_TRIALS, "maxcor": 20,
                           "maxls": 20, "ftol": 1e-12, "gtol": 1e-8})
u, final_value = best["u"], best["value"]
if u is None:
    raise RuntimeError("no trial converged a free boundary; see the trial log")
print(f"[solve] {result.nit} L-BFGS-B iterations, {counts['trials']} trials "
      f"({counts['rejected']} rejected), status {result.status}: {result.message}",
      flush=True)
optimization_seconds = time.perf_counter() - started

###############################################################################
### Check the result against the targets ######################################
###############################################################################

# Re-solve the optimized coils' free boundary independently of the optimizer,
# on a radial ladder that ends finer than the trial solves, and check every
# target on that solve.
coils_final = coils_from_x(jnp.asarray(x0 + scales * u))
field_final = BiotSavart(coils_final)
final_ns = [NS] if ci_smoke else [16, 51]
final_input = replace(inp, ns_array=np.array(final_ns),
                      ftol_array=np.full(len(final_ns), FTOL if ci_smoke else 1.0e-12),
                      niter_array=np.full(len(final_ns), 8000))
free_result = vj.solve_free_boundary_multigrid(
    final_input, external_field=field_final, verbose=not ci_smoke,
    raise_on_max_iterations=False)
final_converged = bool(np.all(np.asarray(free_result.converged)))
final_context = im.runtime_from_params(
    im.params_from_input(final_input),
    vj.make_free_boundary_config(final_input, field_final, ns=final_ns[-1]).implicit)
final_values = {
    "QA total": float(qs.total_state(free_result.state, final_context)),
    "beta": float(opt.volume_average_beta(free_result.state, final_context)),
    "aspect": float(opt.aspect_ratio(free_result.state, final_context)),
    "mean iota": float(opt.mean_iota(free_result.state, final_context)),
    "min |iota|": float(opt.min_abs_iota(free_result.state, final_context))}
wout = vj.wout_from_state(
    inp=final_input, state=free_result.state, fsqr=free_result.fsqr,
    fsqz=free_result.fsqz, fsql=free_result.fsql, niter=free_result.iterations,
    converged=free_result.converged, vacuum_output=free_result.vacuum)

surface_final = SurfaceRZFourier.from_wout_file(
    vj.write_wout("wout_single_stage_free_boundary_finite_beta_optimized.nc", wout),
    nphi=61, ntheta=64)
gamma = np.asarray(coils_final.gamma)
points = np.asarray(surface_final.gamma).reshape(-1, 3)
coil_surface_distance = min(
    float(np.linalg.norm(c[:, None] - points[None], axis=2).min()) for c in gamma)
coil_distance = min(
    float(np.linalg.norm(gamma[i][:, None] - gamma[j][None], axis=2).min())
    for i in range(len(gamma)) for j in range(i + 1, len(gamma)))
maximum_curvature = float(np.max(np.asarray(coils_final.curvature)))
minimum_iota, final_aspect = final_values["min |iota|"], final_values["aspect"]
final_beta = final_values["beta"]

###############################################################################
### Print, plot and save ######################################################
###############################################################################

print("[final] " + ", ".join(f"{k} = {v:.4g}" for k, v in final_values.items()))
print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} after "
      f"{result.nit} L-BFGS-B iterations and {counts['trials']} trials")
print(f"Coil lengths = {np.asarray(coils_final.length[:N_COILS])}")
print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m "
      f"(target >= {COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
print(f"Minimum coil-coil distance = {coil_distance:.4f} m "
      f"(target >= {COIL_DISTANCE_LIMIT:.4f} m)")
print(f"Maximum curvature = {maximum_curvature:.4f} 1/m "
      f"(target <= {CURVATURE_LIMIT:.4f} 1/m)")
print(f"Minimum |iota| = {minimum_iota:.4f} (target >= {IOTA_FLOOR:.4f})")
print(f"Aspect ratio = {final_aspect:.4f} (target <= {ASPECT_LIMIT:.4f})")
print(f"Volume-average beta = {100 * final_beta:.4f}% (target {100 * TARGET_BETA:.2f}% "
      f"within {100 * BETA_TOLERANCE:.0f}%)")

checks = (
    ("minimum |iota|", minimum_iota, IOTA_FLOOR, "below"),
    ("aspect ratio", final_aspect, ASPECT_LIMIT, "above"),
    ("minimum coil-surface distance", coil_surface_distance, COIL_SURFACE_DISTANCE_LIMIT, "below"),
    ("minimum coil-coil distance", coil_distance, COIL_DISTANCE_LIMIT, "below"),
    ("maximum curvature", maximum_curvature, CURVATURE_LIMIT, "above"),
    ("beta", final_beta, TARGET_BETA * (1 - BETA_TOLERANCE), "below"),
    ("beta", final_beta, TARGET_BETA * (1 + BETA_TOLERANCE), "above"),
)
unmet = [f"{name} {value:.4g} {side} the {limit:.4g} limit"
         for name, value, limit, side in checks
         if (value < limit if side == "below" else value > limit)]
if not final_converged:
    unmet.append("the verification free-boundary solve did not converge")
if unmet:
    print("\nThis run did NOT meet its stated targets: " + "; ".join(unmet) + ".")
    if ci_smoke:
        print("Smoke mode caps the budget far below what the targets need; exit status 0.")
else:
    print("\nAll stated targets met.")

summary = {
    "example": "single_stage_free_boundary_optimization_finite_beta.py", "smoke": ci_smoke,
    "optimization_seconds": round(optimization_seconds, 1),
    "coil_fit_iterations": int(coil_fit.nit), "trials": counts["trials"],
    "rejected_trials": counts["rejected"], "lbfgsb_iterations": int(result.nit),
    # one per trial, plus the seed check and the verification solve
    "free_boundary_solves": counts["trials"] + 2,
    "phiedge": float(inp.phiedge), "pres_scale": float(inp.pres_scale), "seed boundary shift": boundary_shift,
    "seed": seed_values,
    "final": {**final_values,
              "coil-surface distance": coil_surface_distance,
              "coil-coil distance": coil_distance,
              "maximum curvature": maximum_curvature,
              "verification solve converged": final_converged},
    "targets": {"min |iota| >=": IOTA_FLOOR, "aspect <=": ASPECT_LIMIT,
                "coil-surface distance >=": COIL_SURFACE_DISTANCE_LIMIT,
                "coil-coil distance >=": COIL_DISTANCE_LIMIT,
                "maximum curvature <=": CURVATURE_LIMIT,
                "beta": [TARGET_BETA * (1 - BETA_TOLERANCE),
                         TARGET_BETA * (1 + BETA_TOLERANCE)]},
    "unmet": unmet, "met": not unmet,
}
Path("single_stage_free_boundary_optimization_finite_beta_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n")

input_path = final_input.to_indata("input.single_stage_free_boundary_finite_beta_optimized")
coils_final.to_json("coils_single_stage_free_boundary_finite_beta_optimized.json")
surface_initial = surfacerzfourier_from_boundary(
    jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), NFP, nphi=60, ntheta=60)
surface_initial.to_vtk("surface_single_stage_free_boundary_finite_beta_initial", field=field0)
coils0.to_vtk("coils_single_stage_free_boundary_finite_beta_initial")
surface_final.to_vtk("surface_single_stage_free_boundary_finite_beta_optimized", field=field_final)
coils_final.to_vtk("coils_single_stage_free_boundary_finite_beta_optimized")
print(f"Wrote {input_path}\nWrote wout_single_stage_free_boundary_finite_beta_optimized.nc")

print("Plotting results...")
monitor.save("single_stage_free_boundary_finite_beta_objectives.csv")
monitor.plot("single_stage_free_boundary_finite_beta_objectives.png",
             title="Finite-beta free-boundary single-stage objective terms")
vj.plot_optimization_objects("single_stage_free_boundary_finite_beta_optimization.png",
                             ("Initial", surface_initial, coils0),
                             ("Optimized", surface_final, coils_final))
print("Wrote single_stage_free_boundary_finite_beta_optimization.png and the objective history")
if unmet and not ci_smoke:
    raise SystemExit(1)

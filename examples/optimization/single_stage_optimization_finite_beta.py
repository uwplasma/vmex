#!/usr/bin/env python
r"""Single-stage fixed-boundary optimization at finite beta with VMEX + ESSOS.

``single_stage_optimization.py`` at volume-average beta 0.5%: the same seed,
targets, weights and coils, with a pressure profile p ~ (1 - s) and
no net toroidal current (NCURR = 1, CURTOR = 0, AC = 0). The plasma boundary
and the coils are optimized together; VMEX solves the fixed-boundary
equilibrium on every trial and differentiates it implicitly.

At finite beta the plasma currents make a field of their own, so the coils
no longer have to be tangent to the boundary by themselves: the condition is
that the TOTAL field is, (B_coils + B_plasma).n = 0. Virtual casing computes
B_plasma on the boundary from the converged equilibrium on every trial, and
the normal-field limit applies to that total. With zero edge pressure the
normal condition is the whole interface condition, so no pressure-balance
term is needed.

The coil field also sets the field strength, which now matters: the coils
are fitted to the seed first (as in ``single_stage_optimization.py``), the
toroidal flux PHIEDGE is read off the fitted field, and the pressure is
scaled so the seed equilibrium sits at the target beta. A beta term in the
objective keeps it there as the boundary moves.

The objective is that of ``single_stage_optimization.py`` plus

    (1/2) (<beta> / TARGET_BETA - 1)^2

with the normal-field limit applied to the total field. Self-consistent
bootstrap current and current-profile optimization are not part of this
example.

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
from vmex.core import virtual_casing as vc

from essos.coils import Coils, CreateEquallySpacedCurves
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import surfacerzfourier_from_boundary

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Highest boundary Fourier mode number that is varied:
MAX_MODE = 2

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.05, 1.0, 6)

# Seed boundary: a rotating ellipse of aspect ratio about 5.97. RBC(1,1) and
# ZBS(1,1) carry opposite signs; equal signs give a circle and no transform.
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
NORMAL_FIELD_LIMIT = 0.01         # area-weighted RMS of B.n/|B| on the boundary
COIL_SURFACE_DISTANCE_LIMIT = 0.15
COIL_DISTANCE_LIMIT = 0.17
CURVATURE_LIMIT = 7.0

# Values the optimizer is given instead. They are tighter than the limits above
# for two reasons: the check re-solves on a finer radial and surface grid, and a
# quadratic penalty settles just inside the threshold it is handed.
IOTA_CONSTRAINT = 0.43
ASPECT_CONSTRAINT = 5.97
NORMAL_FIELD_CONSTRAINT = 0.008
CURVATURE_OBJECTIVE_LIMIT = 6.9
COIL_DISTANCE_CONSTRAINT = 0.19
COIL_SURFACE_DISTANCE_CONSTRAINT = 0.16

# Coils: number of unique shapes, Fourier order, and the circle they start on.
# The aspect-6 seed reaches about 0.3 m from the circle R = 1, so radius 0.5
# starts the coils about 0.2 m out, above the 0.15 m clearance limit.
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

# Bound on each scaled variable. They keep a trial boundary inside the range
# the seed solve converges on; L-BFGS-B rather than BFGS because BFGS cannot
# represent a bound.
PARAMETER_BOUND = 3.0

# Budgets. One trial is one equilibrium solve, one virtual-casing evaluation
# and one adjoint. The script's own end-of-run check passes at iterations 12,
# 14, 16 and 20 and fails at 10 (aspect 6.002), measured; 14 iterations took 19
# trials. Raise MAXITER to go further.
MAXITER = 14
MAX_TRIALS = 35
COIL_FIT_MAXITER = 200            # coil-only pre-fit, no equilibrium solves

# Surface grid the coil objective uses. A toroidal count commensurate with the
# coil number aliases narrow B.n/B structure, so 37 rather than 36.
NPHI, NTHETA = 37, 32
# Significant digits of the virtual-casing plasma field:
VC_DIGITS = 4

# A compact GIF of the accepted iterates. Off by default: with "absB" colour
# every frame re-solves its equilibrium, which took longer than the whole
# 14-iteration optimization (measured).
MAKE_MOVIE = False
# Surface colour in that GIF: None, "absB", "B.n/B", or a callable
# ``(x, objects) -> values``.
MOVIE_SURFACE_COLOR = "absB"

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER, MAX_TRIALS, COIL_FIT_MAXITER = 2, 4, 2
    N_SEGMENTS, COIL_ORDER, NPHI, NTHETA = 24, 2, 8, 8

###############################################################################
# End of input parameters.
###############################################################################

started = time.perf_counter()

###############################################################################
### Set up the seed and fit the coils to it ###################################
###############################################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = SEED_MINOR_RADIUS
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_ELLIPSE, -SEED_ELLIPSE
mpol = max(MAX_MODE + 2, 5)
am = np.zeros(21)
am[:len(PRESSURE_PROFILE)] = PRESSURE_PROFILE
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am,
              pres_scale=1.0, ncurr=1, curtor=0.0, ac=np.zeros(21)).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
surface_seed = surfacerzfourier_from_boundary(
    jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), NFP, nphi=NPHI, ntheta=NTHETA)

curves0 = CreateEquallySpacedCurves(
    N_COILS, COIL_ORDER, COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS,
    n_segments=N_SEGMENTS, nfp=NFP, stellsym=STELLSYM)
coils_circular = Coils(curves0, np.full(N_COILS, COIL_CURRENT))
x_coils_circular = np.asarray(curves0.dofs).ravel()
coil_scales = np.full(x_coils_circular.size, 0.05)


def coils_from_dofs(x_coils):
    return coils_circular.with_dofs(jnp.concatenate((x_coils, coils_circular.dofs_currents)))


def normalized_normal_field(coils, surface):
    """B.n / |B| of the coil field alone on a surface."""
    field = BiotSavart(coils)
    B = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
    return jnp.sum(B * surface.unitnormal, axis=2) / jnp.linalg.norm(B, axis=2)


def normal_field_rms(coils, surface):
    weights = surface.area_element / jnp.sum(surface.area_element)
    return jnp.sqrt(jnp.sum(weights * normalized_normal_field(coils, surface)**2))


def coil_field(coils):
    """The coil Biot-Savart field as a batched callable, for virtual casing and plots."""
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)


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


def coil_fit_objective(u_coils):
    """The coil half of the objective with the boundary frozen at the seed.

    At 0.5% beta the plasma's own normal field is small, so the vacuum
    condition B_coils.n = 0 is a good enough target for a starting point.
    """
    coils = coils_from_dofs(jnp.asarray(x_coils_circular) + jnp.asarray(coil_scales) * u_coils)
    normal_field = 1.0 - normal_field_rms(coils, surface_seed) / NORMAL_FIELD_CONSTRAINT
    return jnp.sum(coil_costs(coils, surface_seed)) + hinge(normal_field[None])


print("Running single_stage_optimization_finite_beta.py")
coil_fit_value_and_grad = jax.jit(jax.value_and_grad(coil_fit_objective))
coil_fit = minimize(
    lambda u: tuple(map(np.asarray, coil_fit_value_and_grad(jnp.asarray(u)))),
    np.zeros_like(x_coils_circular), jac=True, method="L-BFGS-B",
    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x_coils_circular.size,
    options={"maxiter": COIL_FIT_MAXITER, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
x_coils0 = x_coils_circular + coil_scales * coil_fit.x
coils0 = coils_from_dofs(jnp.asarray(x_coils0))
print(f"[coil fit] {coil_fit.nit} L-BFGS-B iterations, no equilibrium solves: B.n/B RMS "
      f"= {100 * float(normal_field_rms(coils0, surface_seed)):.3f}% on the seed")

###############################################################################
### Set up the equilibrium ####################################################
###############################################################################

# The toroidal flux the coils put through the seed's phi = 0 cross-section
# (Gauss-Legendre in radius, uniform in angle) fixes the field strength, and
# with it the pressure a given beta needs. At phi = 0 every toroidal harmonic
# has phase m*theta, so the cross-section sums the boundary over n.
rho, rho_weights = np.polynomial.legendre.leggauss(24)
rho, rho_weights = 0.5 * (rho + 1.0), 0.5 * rho_weights
theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
m = np.arange(inp.rbc.shape[1])
rbc_phi0, zbs_phi0 = inp.rbc.sum(axis=0), inp.zbs.sum(axis=0)
R_edge = np.cos(np.outer(theta, m)) @ rbc_phi0
Z_edge = np.sin(np.outer(theta, m)) @ zbs_phi0
dR_edge = -np.sin(np.outer(theta, m)) @ (m * rbc_phi0)
dZ_edge = np.cos(np.outer(theta, m)) @ (m * zbs_phi0)
R = rbc_phi0[0] + rho[:, None] * (R_edge - rbc_phi0[0])
Z = rho[:, None] * Z_edge
area_element = rho[:, None] * ((R_edge - rbc_phi0[0]) * dZ_edge - Z_edge * dR_edge)
points = np.stack([R, np.zeros_like(R), Z], axis=-1).reshape(-1, 3)
B_phi = np.asarray(coil_field(coils0)(jnp.asarray(points)))[:, 1].reshape(R.shape)
toroidal_flux = float(np.sum(rho_weights[:, None] * B_phi * area_element)
                      * (2.0 * np.pi / theta.size))
# VMEC counts the toroidal angle the other way round, hence the sign.
inp = replace(inp, phiedge=-toroidal_flux)
# Beta is proportional to the pressure at fixed boundary and flux, so one seed
# solve at PRES_SCALE = 1 gives the scale.
beta_at_unit_pressure = float(opt.solve_equilibrium(inp).wout.betatotal)
inp = replace(inp, pres_scale=TARGET_BETA / beta_at_unit_pressure)
print(f"[flux] PHIEDGE = {inp.phiedge:.6f} Wb from the fitted coils; "
      f"PRES_SCALE = {inp.pres_scale:.1f} Pa for beta {100 * TARGET_BETA:.2f}%")


def relative_beta(equilibrium_state, solver_context):
    return opt.volume_average_beta(equilibrium_state, solver_context) / TARGET_BETA


# A VmecProblem owns the boundary-mode convention, the dof scaling and the
# equilibrium solve. Quasisymmetry and beta are its objective; the limits are
# added below.
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
plasma_problem = opt.VmecProblem.from_tuples(
    inp, [(qs.residuals_state, 0.0, 1.0), (relative_beta, 1.0, 1.0)], max_mode=MAX_MODE,
    use_ess=True, progress=not ci_smoke)

# One variable vector holds both halves. SciPy works in dimensionless
# increments u, with x = x0 + scales * u, so every dof moves on a similar scale.
x_boundary0 = plasma_problem.x0
rbc0, zbs0 = plasma_problem.boundary_from_x(x_boundary0)
x0 = np.concatenate([x_boundary0, x_coils0])
scales = np.concatenate([0.02 * plasma_problem.scales, coil_scales])
n_boundary = x_boundary0.size


def objects_from_x(x):
    """The boundary surface and the coils implied by one variable vector."""
    rbc, zbs = plasma_problem.boundary_from_x(x[:n_boundary])
    surface = surfacerzfourier_from_boundary(rbc, zbs, NFP, nphi=NPHI, ntheta=NTHETA)
    return surface, coils_from_dofs(x[n_boundary:])


# Virtual casing picks its quadrature once, on the concrete seed, so that the
# plasma field stays differentiable in the boundary on every trial.
seed_equilibrium = plasma_problem.equilibrium_from_x(x_boundary0)
precision = vc.plan_vc_precision(vc.surface_field_data_from_state(
    inp, seed_equilibrium.solution, runtime=seed_equilibrium.solver_context,
    nphi=NPHI, ntheta=NTHETA), digits=VC_DIGITS)


def total_normal_field(input_, coils, equilibrium_state, solver_context, nphi, ntheta,
                       precision):
    """(B_coils + B_plasma).n / |B| on the boundary, and its area weights."""
    surface_data = vc.surface_field_data_from_state(
        input_, equilibrium_state, runtime=solver_context, nphi=nphi, ntheta=ntheta)
    interface = vc.PlasmaVacuumInterface.from_surface_data(
        surface_data, digits=VC_DIGITS, precision=precision)
    B = jnp.linalg.norm(surface_data.B_total, axis=0)
    return interface.bnormal_residual(coil_field(coils)) / B, interface.weights


###############################################################################
### Set up the objective ######################################################
###############################################################################

def plasma_objective(u):
    """Quasisymmetry, beta, the transform floor, the aspect and normal-field limits.

    One equilibrium solve, one virtual-casing evaluation and one adjoint.
    Floor the profile minimum rather than its average: a mean target is
    satisfiable while an interior surface sits near zero transform.
    """
    x = jnp.asarray(x0) + jnp.asarray(scales) * u
    _, coils = objects_from_x(x)

    def limits(state, ctx):
        normal_field, weights = total_normal_field(
            inp, coils, state, ctx, NPHI, NTHETA, precision)
        rms = jnp.sqrt(jnp.sum(weights * normal_field**2))
        return hinge(jnp.stack([
            opt.min_abs_iota(state, ctx) / IOTA_CONSTRAINT - 1.0,
            1.0 - opt.aspect_ratio(state, ctx) / ASPECT_CONSTRAINT,
            1.0 - rms / NORMAL_FIELD_CONSTRAINT]))

    return plasma_problem.jax_objective_from_state(x[:n_boundary], limits, n_extra_terms=1)


def coil_objective(u):
    """Coil regularization. Pure JAX, no solve."""
    surface, coils = objects_from_x(jnp.asarray(x0) + jnp.asarray(scales) * u)
    costs = coil_costs(coils, surface)
    return jnp.sum(costs), costs


plasma_value_and_grad = jax.jit(jax.value_and_grad(plasma_objective, has_aux=True))
coil_value_and_grad = jax.jit(jax.value_and_grad(coil_objective, has_aux=True))

# The monitor prints one row per evaluation and records the history for the
# plot and the movie at the end.
monitor = opt.OptimizationMonitor(plasma_problem)
counts, cached = {"trials": 0}, {}
COIL_TERMS = ("coil length", "coil curvature", "coil separation", "coil-surface separation")
n_qs_rows = plasma_problem.metadata["term_slices"][0][2]


def value_and_grad(u):
    """What SciPy calls: the two halves evaluated once and added."""
    u = np.asarray(u, dtype=float)
    if cached.get("key") == u.tobytes():          # SciPy re-asks at the same point
        return cached["value"], cached["gradient"].copy()
    counts["trials"] += 1
    (plasma_value, (rows, plasma_penalty)), plasma_gradient = \
        plasma_value_and_grad(jnp.asarray(u))
    (coil_value, coil_costs_), coil_gradient = coil_value_and_grad(jnp.asarray(u))
    rows = np.asarray(rows)
    terms = {"quasisymmetry": 0.5 * float(rows[:n_qs_rows] @ rows[:n_qs_rows]),
             "beta": 0.5 * float(rows[n_qs_rows:] @ rows[n_qs_rows:]),
             "iota/aspect/B.n penalty": float(np.asarray(plasma_penalty)[0])}
    terms.update(zip(COIL_TERMS, map(float, np.asarray(coil_costs_))))
    value, gradient = monitor.cache_evaluation(
        u, plasma_value + coil_value, plasma_gradient + coil_gradient, terms)
    cached.update(key=u.tobytes(), value=value, gradient=gradient)
    return value, gradient.copy()


###############################################################################
### Run the optimization ######################################################
###############################################################################

print(f"Fixed-boundary VMEX + virtual casing + ESSOS at beta {100 * TARGET_BETA:.2f}%: "
      f"{n_boundary} boundary and {x_coils0.size} coil variables, exact reverse-mode "
      "derivatives")
report = opt.EquilibriumReporter(
    ("QA total", qs.total, ".6e"), ("beta", opt.volume_average_beta, ".4%"),
    ("aspect", opt.aspect_ratio, ".4f"), ("mean iota", opt.mean_iota, ".4f"),
    ("min |iota|", opt.min_abs_iota, ".4f"))
seed_values = report("seed", seed_equilibrium)

u = np.zeros_like(x0)
vj.FunctionProblem.from_functions(u, value_and_grad=value_and_grad).compile_value_and_gradient(
    report_interval=10.0)
initial_value = monitor.records[0].cost
result = minimize(value_and_grad, u, jac=True, method="L-BFGS-B", callback=monitor,
                  bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size,
                  options={"maxiter": MAXITER, "maxfun": MAX_TRIALS, "maxcor": 20,
                           "maxls": 20, "ftol": 1e-12, "gtol": 1e-8})
u, final_value = result.x, float(result.fun)
print(f"[solve] {result.nit} L-BFGS-B iterations, {counts['trials']} trials, "
      f"status {result.status}: {result.message}", flush=True)
optimization_seconds = time.perf_counter() - started

###############################################################################
### Check the result against the targets ######################################
###############################################################################

# A lower objective proves nothing on its own: a boundary with no transform
# makes the quasisymmetry residual trivially small. Re-solve on a finer radial
# and surface grid and compare every target explicitly.
x_final = x0 + scales * u
_, coils_final = objects_from_x(jnp.asarray(x_final))
equilibrium = plasma_problem.equilibrium_from_x(x_final[:n_boundary])
# FTOL 1e-12 rather than tighter: 1e-13 is out of reach even at ns=50 here, and
# a verification solve that cannot converge verifies nothing.
final_input = replace(plasma_problem.input_from_x(x_final[:n_boundary]),
                      ns_array=np.array([31 if ci_smoke else 101]),
                      ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-12]),
                      niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution, verbose=not ci_smoke)

surface_final = surfacerzfourier_from_boundary(
    jnp.asarray(final_input.rbc), jnp.asarray(final_input.zbs), NFP, nphi=61, ntheta=64)
# The total field's normal component, with virtual casing replanned on the
# finer grid of the optimized boundary.
final_data = vc.surface_field_data_from_state(
    final_input, final_equilibrium.solution, runtime=final_equilibrium.solver_context,
    nphi=61, ntheta=64)
normal_field, area = map(np.asarray, total_normal_field(
    final_input, coils_final, final_equilibrium.solution, final_equilibrium.solver_context,
    61, 64, vc.plan_vc_precision(final_data, digits=VC_DIGITS)))
normal_field_rms_final = float(np.sqrt(np.sum(area * normal_field**2)))
normal_field_max = float(np.max(np.abs(normal_field)))
gamma = np.asarray(coils_final.gamma)
points = np.asarray(surface_final.gamma).reshape(-1, 3)
coil_surface_distance = min(
    float(np.linalg.norm(c[:, None] - points[None], axis=2).min()) for c in gamma)
coil_distance = min(
    float(np.linalg.norm(gamma[i][:, None] - gamma[j][None], axis=2).min())
    for i in range(len(gamma)) for j in range(i + 1, len(gamma)))
maximum_curvature = float(np.max(np.asarray(coils_final.curvature)))
minimum_iota = float(opt.min_abs_iota(final_equilibrium.state, final_equilibrium.runtime))
final_aspect = float(opt.aspect_ratio(final_equilibrium.state, final_equilibrium.runtime))
final_beta = float(opt.volume_average_beta(final_equilibrium.state, final_equilibrium.runtime))
final_converged = bool(np.all(np.asarray(final_equilibrium.result.converged)))

###############################################################################
### Print, plot and save ######################################################
###############################################################################

final_values = report("final", final_equilibrium)
print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} after "
      f"{result.nit} L-BFGS-B iterations and {counts['trials']} trials")
print(f"Coil lengths = {np.asarray(coils_final.length[:N_COILS])}")
print(f"(B_coils + B_plasma).n/B: area-weighted RMS = {100 * normal_field_rms_final:.3f}%, "
      f"max = {100 * normal_field_max:.3f}% (target RMS <= "
      f"{100 * NORMAL_FIELD_LIMIT:.1f}%; the maximum is reported, not optimized)")
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
    ("B.n/B RMS", normal_field_rms_final, NORMAL_FIELD_LIMIT, "above"),
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
    unmet.append("the ns=101 verification solve did not converge")
if unmet:
    print("\nThis run did NOT meet its stated targets: " + "; ".join(unmet) + ".")
    if ci_smoke:
        print("Smoke mode caps the budget far below what the targets need; exit status 0.")
else:
    print("\nAll stated targets met.")

summary = {
    "example": "single_stage_optimization_finite_beta.py", "smoke": ci_smoke,
    "optimization_seconds": round(optimization_seconds, 1),
    "coil_fit_iterations": int(coil_fit.nit), "trials": counts["trials"],
    "lbfgsb_iterations": int(result.nit),
    "equilibrium_solves": monitor.records[-1].equilibrium_solves,
    "phiedge": float(inp.phiedge), "pres_scale": float(inp.pres_scale),
    "seed": seed_values,
    "final": {**final_values, "min |iota|": minimum_iota, "aspect": final_aspect,
              "B.n/B RMS": normal_field_rms_final, "B.n/B max": normal_field_max,
              "coil-surface distance": coil_surface_distance,
              "coil-coil distance": coil_distance,
              "maximum curvature": maximum_curvature, "beta": final_beta,
              "verification solve converged": final_converged},
    "targets": {"min |iota| >=": IOTA_FLOOR, "aspect <=": ASPECT_LIMIT,
                "B.n/B RMS <=": NORMAL_FIELD_LIMIT,
                "coil-surface distance >=": COIL_SURFACE_DISTANCE_LIMIT,
                "coil-coil distance >=": COIL_DISTANCE_LIMIT,
                "maximum curvature <=": CURVATURE_LIMIT,
                "beta": [TARGET_BETA * (1 - BETA_TOLERANCE),
                         TARGET_BETA * (1 + BETA_TOLERANCE)]},
    "unmet": unmet, "met": not unmet,
}
Path("single_stage_optimization_finite_beta_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

input_path = final_input.to_indata("input.single_stage_finite_beta_optimized")
wout_path = vj.write_wout("wout_single_stage_finite_beta_optimized.nc", final_equilibrium.wout)
coils_final.to_json("coils_single_stage_finite_beta_optimized.json")
# ESSOS writes |B| and B.n/B on the surface and the filaments for ParaView.
surface_initial = surfacerzfourier_from_boundary(rbc0, zbs0, NFP, nphi=60, ntheta=60)
surface_initial.to_vtk("surface_single_stage_finite_beta_initial", field=BiotSavart(coils0))
coils0.to_vtk("coils_single_stage_finite_beta_initial")
surface_final.to_vtk("surface_single_stage_finite_beta_optimized", field=BiotSavart(coils_final))
coils_final.to_vtk("coils_single_stage_finite_beta_optimized")
print(f"Wrote {input_path}\nWrote {wout_path}")
print("Wrote coils_single_stage_finite_beta_optimized.json and single_stage_optimization_finite_beta_summary.json")
print("Wrote initial and optimized surface/coils VTK files")

print("Plotting results...")
vj.plot_optimization_objects("single_stage_finite_beta_optimization.png",
                             ("Initial", surface_initial, coils0),
                             ("Optimized", surface_final, coils_final))
monitor.save("single_stage_finite_beta_objectives.csv")
monitor.plot("single_stage_finite_beta_objectives.png", title="Finite-beta single-stage objective terms")
print("Wrote single_stage_finite_beta_optimization.png")
print("Wrote single_stage_finite_beta_objectives.csv and single_stage_finite_beta_objectives.png")
if MAKE_MOVIE:
    print("Making movie of accepted iterates...")
    monitor.movie_surface_coils(
        "single_stage_finite_beta_optimization.gif", objects_from_x, x0=x0, scales=scales,
        surface_color=MOVIE_SURFACE_COLOR, plasma_problem=plasma_problem,
        external_field=lambda objects: coil_field(objects[1]), nphi=NPHI, ntheta=NTHETA,
        cmap="jet")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
if unmet and not ci_smoke:
    raise SystemExit(1)

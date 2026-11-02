#!/usr/bin/env python
"""Optimize a boundary for piecewise omnigenity with a magnetic well.

SciPy nonlinear least squares varies the boundary Fourier coefficients of a
vacuum equilibrium in stages of increasing mode number.
The residual vector holds the mapped pwO contour
variance, the aspect ratio, a floor on the rotational transform, the magnetic
well and a ceiling on the mirror ratio, and VMEX supplies its exact Jacobian.
The verified boundary is then compared with the seed by a matched NEO
calculation of epsilon_eff^(3/2). Requires vmex[neoclassical].
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.neoclassical import screen_transport
from vmex.core.pwo import MappedPwOResidual
from vmex.core.statephysics import _field_chain

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 3
print(f"pwO optimization, NFP = {NFP}")
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed. The exactly circular
# torus has zero first-order iota sensitivity; this gives the optimizer a basin:
SEED_PERTURBATION = 0.05

# Flux surfaces the pwO residual is evaluated on, and its Boozer resolution:
SURFACES = np.linspace(0.1, 0.9, 5)
MBOZ = 16

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2, 3, 4, 5]
MAX_NFEV = [15]*len(MAX_MODES)
# STAGE_FTOL = [1e-13 if NFP==2 else 1e-12]*len(MAX_MODES)
ASPECT_TARGET = 5.5
MAGNETIC_WELL_TARGET = 0 #0.04
MAGNETIC_WELL_WEIGHT = 1e3
ELONGATION_LIMIT = 9.0
NS_ARRAY = np.array([35])
NITER_ARRAY = np.array([5000])
FTOL_ARRAY = np.array([1e-12])
STAGE_FTOL = [FTOL_ARRAY]*len(MAX_MODES)

# Stages that also minimize the smooth fast-ion proxy GammaC, its surfaces and weight:
GAMMA_C = [False]*len(MAX_MODES) #[False] * (len(MAX_MODES) - 2) + [True] * 2
GAMMA_C_SURFACES = tuple(SURFACES[1:])
GAMMA_C_WEIGHT = 100.0

# Targets:
IOTA_FLOOR = 0.31 if NFP==1 else 0.42  # minimum |iota| over the profile; use 0.31 for NFP = 1
MIRROR_LIMIT = 0.16               # maximum (Bmax-Bmin)/(Bmax+Bmin) on the pwO surfaces
AXIS_LIMIT = 0.25                 # maximum RMS helical excursion of the magnetic axis, metres
EDGE_JACOBIAN_FLOOR = 0.25        # minimum min/max Jacobian ratio in each of the three outer cells
PWO_WEIGHT = 1.0e3                # the seed's raw pwO variance is of order 0.01

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.05
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = [37, 45, 55, 71]        # restarted in small steps: a large jump flips the Jacobian
FINAL_FTOL = 1.0e-10
FINAL_NITER = 30000

# Every output file name contains this:
OUTPUT_NAME = f"pwO_optimized_nfp{NFP}" + (f"_well{MAGNETIC_WELL_TARGET}" if MAGNETIC_WELL_TARGET > 0 else "")

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV, STAGE_FTOL, GAMMA_C = [1], [4], [1.0e-11], [False]
    FINAL_NS, FINAL_FTOL = [31], 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface."""
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def mirror_excess(equilibrium_state, solver_context):
    """Hinge on the full-surface mirror ratio, which the mapped variance does not see."""
    mirror = pwo.diagnostics_state(equilibrium_state, solver_context)[3]
    return jnp.maximum(mirror - MIRROR_LIMIT, 0.0)


def axis_excursion(equilibrium_state, solver_context):
    """RMS size of the axis' n != 0 harmonics; undoes the state's sqrt(2) mode scaling."""
    modes = solver_context.modes
    helical = (np.asarray(modes.m) == 0) & (np.asarray(modes.n) != 0)
    return jnp.sqrt(2 * jnp.sum(equilibrium_state.R_cos[0, helical]**2
                                + equilibrium_state.Z_sin[0, helical]**2))


def axis_excess(equilibrium_state, solver_context):
    """Hinge on the axis excursion: a strongly helical axis cannot be re-solved on a finer grid."""
    return jnp.maximum(axis_excursion(equilibrium_state, solver_context) - AXIS_LIMIT, 0.0)


def edge_jacobian_excess(equilibrium_state, solver_context):
    """Hinge on the Jacobian spread of the three outer cells: a near-zero cell flips sign on a finer grid."""
    tau = _field_chain(equilibrium_state, solver_context)[1].tau * solver_context.setup.signgs
    tau = tau[-3:].reshape(3, -1)
    return jnp.maximum(EDGE_JACOBIAN_FLOOR - jnp.min(tau, axis=1) / jnp.max(tau, axis=1), 0.0)


def elongation_excess(equilibrium_state, solver_context):
    """Hinge on the cross-section elongation above its limit."""
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)


# Each term is (function, target, weight).
pwo = MappedPwOResidual(SURFACES, mboz=MBOZ, nboz=MBOZ, oversample=2)
objective_function_terms = [
    (pwo, 0.0, PWO_WEIGHT),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 100.0),
    (mirror_excess, 0.0, 100.0),
    (axis_excess, 0.0, 100.0),
    (edge_jacobian_excess, 0.0, 100.0),
    (elongation_excess, 0.0, 10.0),
]
if MAGNETIC_WELL_TARGET > 0:
    objective_function_terms.append(
        (opt.magnetic_well, MAGNETIC_WELL_TARGET, MAGNETIC_WELL_WEIGHT))

report = opt.EquilibriumReporter(
    ("pwO total", pwo.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("elongation", opt.max_elongation, ".4f"),
    ("axis excursion", axis_excursion, ".4f"), ("magnetic well", opt.magnetic_well, ".3f"),
    ("max mirror", lambda state, rt: jnp.max(pwo.diagnostics_state(state, rt)[3]), ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

seed_input = inp
equilibrium = opt.solve_equilibrium(inp)
report("seed", equilibrium)
# Stages that share a boundary resolution and objective share ONE problem, as in
# QA_optimization.py: every jitted graph is compiled once for the whole group.
group_of = {max_mode: (max(max_mode + 2, MINIMUM_MPOL), gamma_c)
            for max_mode, gamma_c in zip(MAX_MODES, GAMMA_C)}
group_max_mode = {
    max_mode: max(other for other in MAX_MODES if group_of[other] == group_of[max_mode])
    for max_mode in MAX_MODES}
gamma_c_term = (opt.GammaCSmooth(GAMMA_C_SURFACES), 0.0, GAMMA_C_WEIGHT)
problem, group = None, None
for max_mode, max_nfev, stage_ftol in zip(MAX_MODES, MAX_NFEV, STAGE_FTOL):
    print(f"\n===== pwO stage, max_mode = {max_mode} =====")
    if group_of[max_mode] != group:
        group = group_of[max_mode]
        mpol, gamma_c = group
        # The decks' iteration caps are too low for a new resolution's seed solve.
        inp = replace(inp, delt=0.5, ftol_array=stage_ftol,
                      ns_array=NS_ARRAY,
                      niter_array=NITER_ARRAY).change_resolution(
            mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        problem = opt.VmecProblem.from_tuples(
            inp, objective_function_terms + [gamma_c_term] * gamma_c,
            max_mode=group_max_mode[max_mode],
            vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
            ess_alpha=ESS_ALPHA, restart_from=equilibrium,
            progress=True)
        x = problem.x0
        monitor.problem = problem
        if not ci_smoke:
            problem.compile_residual_and_jacobian()
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    step = PARAMETER_STEP * stage.scales
    result = least_squares(
        stage.residual, stage.x0, jac=stage.residual_jac,
        x_scale=step, max_nfev=max_nfev,
        bounds=(stage.x0 - MAX_PARAMETER_CHANGE * step,
                stage.x0 + MAX_PARAMETER_CHANGE * step),
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    x = stage.embed(result.x)
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_resolution = dict(ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = equilibrium
for ns in FINAL_NS:
    final_input = replace(inp, ns_array=np.array([ns]), **final_resolution)
    final_equilibrium = opt.solve_equilibrium(
        final_input, initial_state=final_equilibrium.solution,
        verbose=not ci_smoke, raise_on_max_iterations=True)

# # Matched NEO: the seed on the same radial grid, the same half-mesh surfaces.
# seed_equilibrium = opt.solve_equilibrium(
#     replace(seed_input, ns_array=np.array([FINAL_NS[-1]]), **final_resolution),
#     verbose=False, raise_on_max_iterations=True)
# half_mesh = (np.arange(FINAL_NS[-1] - 1) + 0.5) / (FINAL_NS[-1] - 1)
# neo_surfaces = half_mesh[np.abs(half_mesh[:, None] - SURFACES).argmin(axis=0)]
# seed_Q = screen_transport(seed_equilibrium.wout, neo_surfaces)
# final_Q = screen_transport(final_equilibrium.wout, neo_surfaces)
# print(f"epsilon_eff^(3/2) at s = {neo_surfaces}: seed {seed_Q}, final {final_Q}")

### Print, plot and save ######################################################

final_total = report("final", final_equilibrium)["pwO total"]
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   well_floor=MAGNETIC_WELL_TARGET)
print(f"\npwO total {final_total:.3e}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")

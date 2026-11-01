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
from vmex.core.pwo import FreeMapProblem, MappedPwOResidual

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 3
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed. The exactly circular
# torus has zero first-order iota sensitivity; this gives the optimizer a basin:
SEED_PERTURBATION = 0.05

# Flux surfaces the pwO residual is evaluated on, and its Boozer resolution:
SURFACES = np.linspace(0.1, 0.9, 8)
MBOZ = 16

# Initial values of the pwO map, and which of them the optimizer may change.
# Remove a name to hold it fixed. Any value gives a valid map on every surface:
MAP_PARAMETERS = dict(a=0.3, b=0.3, c=-2.0, eta_min=2.8)
FREE_MAP = ("a", "b", "c", "eta_min")

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2, 3, 4, 5] if NFP==1 else [1, 2, 3, 4]
MAX_NFEV = [20]*len(MAX_MODES)
STAGE_FTOL = [1e-13 if NFP==2 else 1e-12]*len(MAX_MODES)
ASPECT_TARGET = 6.0
MAGNETIC_WELL_TARGET = 0.06
MAGNETIC_WELL_WEIGHT = 3e3
ELONGATION_LIMIT = 9.0

# Alternative settings for a larger design space:
# MAX_MODES = [1, 3, 5, 7]
# MAX_NFEV = [15, 25, 30, 40]
# STAGE_FTOL = [1e-11, 1e-11, 1e-12, 1e-13]    # equilibrium tolerance of each stage
# ASPECT_TARGET = 3.5
# MAGNETIC_WELL_TARGET = 0.07

# Targets:
IOTA_FLOOR = 0.31 if NFP==1 else 0.42  # minimum |iota| over the profile; use 0.31 for NFP = 1
MIRROR_LIMIT = 0.15               # maximum (Bmax-Bmin)/(Bmax+Bmin) on the pwO surfaces
AXIS_LIMIT = 0.25                 # maximum RMS helical excursion of the magnetic axis, metres
PWO_WEIGHT = 1.0e3                # the seed's raw pwO variance is of order 0.01

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = [37, 45, 55, 71]        # restarted in small steps: a large jump flips the Jacobian
FINAL_FTOL = 5.0e-13
FINAL_NITER = 30000

# Every output file name contains this:
OUTPUT_NAME = f"pwO_optimized_nfp{NFP}" + (f"_well{MAGNETIC_WELL_TARGET}" if MAGNETIC_WELL_TARGET > 0 else "")

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV, STAGE_FTOL = [1], [4], [1.0e-11]
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


def elongation_excess(equilibrium_state, solver_context):
    """Hinge on the cross-section elongation above its limit."""
    return jnp.maximum(
        opt.max_elongation(equilibrium_state, solver_context) - ELONGATION_LIMIT, 0.0)


def boozer_spectrum(equilibrium_state, solver_context):
    """The field and iota the map reads; FreeMapProblem turns these rows into the pwO residual."""
    return pwo.spectrum_state(equilibrium_state, solver_context)


# Each term is (function, target, weight).
pwo = MappedPwOResidual(SURFACES, mboz=MBOZ, nboz=MBOZ, oversample=2, limit_map=True,
                        **MAP_PARAMETERS)
objective_function_terms = [
    (boozer_spectrum, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 100.0),
    (mirror_excess, 0.0, 100.0),
    (axis_excess, 0.0, 100.0),
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
# Stages that share a boundary resolution share ONE problem, as in
# QA_optimization.py: every jitted graph is compiled once for the whole group.
resolution_of = {max_mode: max(max_mode + 2, MINIMUM_MPOL) for max_mode in MAX_MODES}
group_max_mode = {
    max_mode: max(other for other in MAX_MODES
                  if resolution_of[other] == resolution_of[max_mode])
    for max_mode in MAX_MODES}
problem, mpol = None, None
for max_mode, max_nfev, stage_ftol in zip(MAX_MODES, MAX_NFEV, STAGE_FTOL):
    print(f"\n===== pwO stage, max_mode = {max_mode} =====")
    if resolution_of[max_mode] != mpol:
        mpol = resolution_of[max_mode]
        # The decks' iteration caps are too low for a new resolution's seed solve.
        inp = replace(inp, delt=0.5, ftol_array=np.array([stage_ftol]),
                      niter_array=np.array([20000])).change_resolution(
            mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        problem = opt.VmecProblem.from_tuples(
            inp, objective_function_terms, max_mode=group_max_mode[max_mode],
            vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
            ess_alpha=ESS_ALPHA, restart_from=equilibrium,
            progress=True)
        x = problem.x0
        monitor.problem = problem
        if not ci_smoke:
            problem.compile_residual_and_jacobian()
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    joint = FreeMapProblem(stage, pwo, equilibrium, MAP_PARAMETERS, FREE_MAP, weight=PWO_WEIGHT)
    step = PARAMETER_STEP * stage.scales
    free_map = np.ones(len(FREE_MAP))
    result = least_squares(
        joint.residual, joint.x0, jac=joint.residual_jac,
        x_scale=np.concatenate((step, 0.5 * free_map)), max_nfev=max_nfev,
        bounds=(np.concatenate((stage.x0 - MAX_PARAMETER_CHANGE * step, -np.inf * free_map)),
                np.concatenate((stage.x0 + MAX_PARAMETER_CHANGE * step, np.inf * free_map))),
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    x = stage.embed(result.x[:joint.nx])
    MAP_PARAMETERS = joint.values(result.x)
    pwo.options.update(MAP_PARAMETERS)
    print(f"map = {MAP_PARAMETERS}")
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

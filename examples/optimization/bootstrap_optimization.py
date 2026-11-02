#!/usr/bin/env python
"""Minimize the bootstrap current of a vacuum boundary with no symmetry target.

SciPy nonlinear least squares varies the boundary Fourier coefficients of a
vacuum equilibrium in stages of increasing mode number, starting from a
near-circular torus. The main residual is the collisionless (1/nu)
Shaing-Callen bootstrap geometric factor -iota Delta_G / f_t on a few flux
surfaces: 1 for a quasi-axisymmetric field, iota/(iota - N) for a
quasi-helically symmetric one and 0 for an exactly omnigenous field with
poloidally closed |B| contours. No quasisymmetry, QI or pwO residual is used;
the remaining rows are the aspect ratio, a floor on the rotational transform,
an edge-Jacobian guard and optional mirror, axis and elongation ceilings.

A zero factor does not make the field omnigenous. At NFP 3 and 4 the optimum
is an egg-crate of isolated |B| maxima and minima whose QA-like and QH-like
contributions cancel; epsilon_eff^(3/2) stays near 0.02. Without the mirror
ceiling the factor also vanishes, through a mirror ratio near 0.5 and
epsilon_eff^(3/2) of order 1. At NFP 1 and 2 the targets are not met.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 3
print(f"Bootstrap optimization, NFP = {NFP}")
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed. The exactly circular
# torus has zero first-order iota sensitivity; this gives the optimizer a basin:
SEED_PERTURBATION = 0.05

# Flux surfaces the bootstrap factor is evaluated on, and its Boozer resolution.
# The factor is unreliable on low-order rational surfaces (iota m = n NFP):
SURFACES = np.linspace(0.15, 0.85, 5)
MBOZ = 16
# Width of the resonances iota m = n NFP (BOOTSJ damp_bs); wider ones let iota cross them:
DAMPING = 0.1

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2, 3, 4, 5]
MAX_NFEV = [15] * len(MAX_MODES)
NS_ARRAY = np.array([35])
NITER_ARRAY = np.array([5000])
FTOL_ARRAY = np.array([1e-12])
STAGE_FTOL = [FTOL_ARRAY] * len(MAX_MODES)

# Stages that also minimize the smooth fast-ion proxy GammaC, its surfaces and weight:
GAMMA_C = [False] * len(MAX_MODES)
GAMMA_C_SURFACES = tuple(SURFACES[1:])
GAMMA_C_WEIGHT = 100.0

# Targets. A limit set to None drops its row:
BOOTSTRAP_WEIGHT = 100.0          # the seed's ratio is about 1 on every surface
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.31 if NFP == 1 else 0.42  # minimum |iota| over the profile
EDGE_JACOBIAN_FLOOR = 0.25        # minimum min/max Jacobian ratio in each of the three outer cells
MIRROR_LIMIT = 0.2                # maximum (Bmax-Bmin)/(Bmax+Bmin) on the boundary
AXIS_LIMIT = None                 # maximum RMS helical excursion of the magnetic axis, metres
ELONGATION_LIMIT = 9.0
LIMIT_WEIGHT = 100.0              # weight of the iota, Jacobian and limit hinges

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
OUTPUT_NAME = f"bootstrap_optimized_nfp{NFP}"

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

# Each term is (function, target, weight).
bootstrap = opt.BootstrapGeometricResidual(SURFACES, mboz=MBOZ, nboz=MBOZ, oversample=2,
                                           damping=DAMPING)
limits = [(opt.mirror_ratio, MIRROR_LIMIT), (opt.axis_excursion, AXIS_LIMIT),
          (opt.max_elongation, ELONGATION_LIMIT)]
objective_function_terms = [
    (bootstrap, 0.0, BOOTSTRAP_WEIGHT),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.Hinge(opt.min_abs_iota, floor=IOTA_FLOOR), 0.0, LIMIT_WEIGHT),
    (opt.Hinge(opt.edge_jacobian_ratio, floor=EDGE_JACOBIAN_FLOOR), 0.0, LIMIT_WEIGHT),
] + [(opt.Hinge(metric, ceiling=limit), 0.0, LIMIT_WEIGHT) for metric, limit in limits if limit is not None]


def bootstrap_ratios(equilibrium):
    """-iota Delta_G / f_t on each surface: 1 is QA-like, 0 is no bootstrap current."""
    return np.asarray(bootstrap.compute_state(equilibrium.solution, equilibrium.solver_context)["ratio"])


report = opt.EquilibriumReporter(
    ("bootstrap total", bootstrap.total, ".4e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("min |iota|", opt.min_abs_iota, ".4f"), ("elongation", opt.max_elongation, ".4f"),
    ("axis excursion", opt.axis_excursion, ".4f"), ("edge mirror", opt.mirror_ratio, ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = opt.solve_equilibrium(inp)
report("seed", equilibrium)
seed_ratios = bootstrap_ratios(equilibrium)
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
    print(f"\n===== bootstrap stage, max_mode = {max_mode} =====")
    if group_of[max_mode] != group:
        group = group_of[max_mode]
        mpol, gamma_c = group
        # The decks' iteration caps are too low for a new resolution's seed solve.
        inp = replace(inp, delt=0.5, ftol_array=stage_ftol, ns_array=NS_ARRAY,
                      niter_array=NITER_ARRAY).change_resolution(
            mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        problem = opt.VmecProblem.from_tuples(
            inp, objective_function_terms + [gamma_c_term] * gamma_c,
            max_mode=group_max_mode[max_mode],
            vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
            ess_alpha=ESS_ALPHA, restart_from=equilibrium, progress=True)
        x = problem.x0
        monitor.problem = problem
        if not ci_smoke:
            problem.compile_residual_and_jacobian()
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    step = PARAMETER_STEP * stage.scales
    result = least_squares(
        stage.residual, stage.x0, jac=stage.residual_jac, x_scale=step, max_nfev=max_nfev,
        bounds=(stage.x0 - MAX_PARAMETER_CHANGE * step, stage.x0 + MAX_PARAMETER_CHANGE * step),
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

### Print, plot and save ######################################################

report("final", final_equilibrium)
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   mirror_limit=MIRROR_LIMIT, elongation_limit=ELONGATION_LIMIT)
print(f"\nbootstrap ratio -iota Delta_G/f_t at s = {np.round(SURFACES, 3)}"
      f"\n  seed  {np.array2string(seed_ratios, precision=4)}"
      f"\n  final {np.array2string(bootstrap_ratios(final_equilibrium), precision=4)}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")

#!/usr/bin/env python
"""Optimize a boundary for quasi-isodynamic confinement with shape limits.

SciPy nonlinear least squares varies the boundary Fourier coefficients of a
vacuum equilibrium in stages of increasing mode number, starting from a
near-circular torus with a rotating ellipse (no QI-shaped warm start). The
residual vector holds the constructed-QI residual, the aspect ratio, and
one-sided limits on the rotational transform, the mirror ratio and the
elongation; VMEX supplies its exact Jacobian. The first stage multiplies the
shape terms so the optimizer reaches the aspect, transform and mirror limits
before the QI residual shapes the boundary.

The QI residual is evaluated on a Boozer grid whose resolution is a knob of
its own. The run optimizes on QI_OPTIONS and re-scores the finished boundary
on the finer VALIDATION_OPTIONS, so an optimum that only exists at the
optimizer's grid resolution is visible as a gap between the two numbers.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.qi import ConstructedQIResidual

# Seed: the circular torus of this field-period count, with these boundary
# coefficients added, keyed (family, n, m). A rotating ellipse gives the torus
# the first-order iota response it lacks. Amplitudes 0.03, 0.04, 0.05, 0.06 and
# 0.07 reach QI 1.4e-3 to 1.6e-3, as does 0.05 with a 0.2 helical axis
# excursion ("zbs", 1, 0). Amplitude 0.045 and a 0.1 axis excursion stop at
# 5e-3 to 6e-3, and the opposite ellipse sign settles at aspect 12 with a
# mirror ratio of 0.27:
NFP = 2
SEED_MODES = {("rbc", -1, 1): -0.05, ("zbs", -1, 1): 0.05}
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Radial grid every optimizer trial is solved on:
STAGE_NS = 31
STAGE_FTOL = 1.0e-12
STAGE_NITER = 5500

# Flux surfaces the QI residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 6)

# Stages: highest boundary mode number varied, residual evaluations spent, and the
# factor on the shape rows. The QI residual is rough, so a stage with a large
# factor first moves the boundary to the aspect, transform and mirror limits;
# without it some seeds never reach them. Three 30-evaluation stages reach QI
# 1e-3 to 2e-3; 15-evaluation stages stop near 5e-3:
STAGES = [(1, 15, 30.0), (1, 30, 1.0), (2, 30, 1.0), (3, 30, 1.0)]

# Boozer resolution the QI residual is optimized on, and the finer one the
# finished boundary is re-scored with. Optimizing on mboz = nboz = 8, nphi = 41,
# nalpha = 12, n_bounce = 13 ends at fine-grid QI 9.6e-3, seven times worse:
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
VALIDATION_OPTIONS = dict(mboz=14, nboz=14, nphi=101, nalpha=29, n_bounce=31)

# Weights on the QI residual and the shape terms (the objective is the sum of
# weight * residual^2):
QI_WEIGHT, ASPECT_WEIGHT, IOTA_WEIGHT, MIRROR_WEIGHT, ELONGATION_WEIGHT = 300.0, 0.3, 10.0, 1000.0, 10.0

# Targets and limits:
ASPECT_TARGET = 8.0               # 6.0 stalls at aspect 6.6 and QI 7.6e-3
IOTA_FLOOR = 0.51                 # minimum |iota| over the profile
MIRROR_LIMIT = 0.21               # 0.16 also converges, to QI 2.0e-3
ELONGATION_LIMIT = 8.0

# A finite-weight limit settles just above its threshold, and the final
# FINAL_NS solve reads the mirror ratio about 4e-4 above the stage grid, so
# the mirror limit starts pulling this far below the limit:
MIRROR_LIMIT_FRACTION = 0.99

# Step control:
ESS_ALPHA = 1.2                   # lower only after a low-mode QI basin has converged
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 71
FINAL_FTOL = 1.0e-13
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = "QI_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    VALIDATION_OPTIONS = QI_OPTIONS
    STAGES = [(2, 5, 1.0)]
    FINAL_NS, FINAL_FTOL = 31, 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
for (family, n, m), amplitude in SEED_MODES.items():
    {"rbc": rbc, "zbs": zbs}[family][inp.ntor + n, m] += amplitude
inp = replace(inp, rbc=rbc, zbs=zbs, ns_array=np.array([STAGE_NS]),
              ftol_array=np.array([STAGE_FTOL]), niter_array=np.array([STAGE_NITER]))

### Set up the objective ######################################################

# Each term is (function, target, weight); the shape terms are the last rows.
qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)
objective_function_terms = [
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.min_abs_iota, opt.at_least(IOTA_FLOOR), IOTA_WEIGHT),
    (opt.mirror_ratio, opt.at_most(MIRROR_LIMIT_FRACTION * MIRROR_LIMIT), MIRROR_WEIGHT),
    (opt.max_elongation, opt.at_most(ELONGATION_LIMIT), ELONGATION_WEIGHT),
]
qi_terms = [(qi, 0.0, QI_WEIGHT), *objective_function_terms]

report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("min |iota|", opt.min_abs_iota, ".4f"),
    ("mirror", opt.mirror_ratio, ".4f"),
    ("elongation", opt.max_elongation, ".4f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

# Every stage shares one resolution, so one problem built at the largest
# max_mode serves them all: problem.subproblem() frees only the stage's
# harmonics and every jitted residual and Jacobian is compiled once. A fresh
# stage still restarts SciPy's trust-region model. A RuntimeWarning about
# uncertified Jacobian columns is expected once the optimizer leaves the seed
# and needs no action; see examples/README.md.
top_mode = max(mode for mode, *_ in STAGES)
mpol = max(top_mode + 2, MINIMUM_MPOL)
inp = replace(inp, delt=0.5).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
problem = opt.VmecProblem.from_tuples(
    inp, qi_terms, max_mode=top_mode, use_ess=True,
    progress=not ci_smoke, evaluation_progress=not ci_smoke,
    ess_alpha=ESS_ALPHA, vary_major_radius=VARY_MAJOR_RADIUS)
monitor.problem = problem
if not ci_smoke:
    problem.compile_residual_and_jacobian()
x = problem.x0
for max_mode, max_nfev, shape_factor in STAGES:
    print(f"\n===== QI stage, max_mode = {max_mode}, shape rows x {shape_factor:g} =====")
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    row_scale = np.ones(stage.residual(stage.x0).size)
    row_scale[-len(objective_function_terms):] = np.sqrt(shape_factor)
    result = least_squares(
        lambda u: row_scale * stage.residual(u), stage.x0,
        jac=lambda u: row_scale[:, None] * stage.residual_jac(u),
        x_scale=stage.scales, max_nfev=max_nfev,
        ftol=1.0e-6, xtol=1.0e-10, verbose=2, callback=monitor)
    x = stage.embed(result.x)
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"QI mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid, then re-score QI on the finer Boozer grid too.
final_input = replace(
    inp, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

qi_final = report("final", final_equilibrium)["constructed QI"]
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   mirror_limit=MIRROR_LIMIT, elongation_limit=ELONGATION_LIMIT)
qi_validation = ConstructedQIResidual(SURFACES, **VALIDATION_OPTIONS)
qi_validation_total = float(qi_validation.total(final_equilibrium))
print(f"\nQI total {qi_final:.3e}; independent fine-grid validation {qi_validation_total:.3e}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}")
print(f"Wrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")

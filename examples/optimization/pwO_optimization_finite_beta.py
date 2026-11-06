#!/usr/bin/env python
"""Continue a vacuum pwO boundary to finite beta, with p proportional to 1-s.

Run ``pwO_optimization.py`` first with its magnetic-well target enabled
(and ELONGATION_LIMIT = 6.0 at NFP 1):
from a vacuum magnetic hill this continuation does not reach a well. Pressure is calibrated at fixed toroidal flux, then the
workflow is that of the vacuum example with beta and Mercier stability added
to the residual vector. The optimized boundary is re-solved on a finer
radial ladder before it is saved.
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
from vmex.core.restart import restart_state

# Number of field periods, and the vacuum result this script continues:
NFP = 3
print(f"finite-beta pwO optimization, NFP = {NFP}")
VACUUM_NAME = f"pwO_optimized_nfp{NFP}_well0.04"
INPUT_FILE = Path(f"input.{VACUUM_NAME}")
WOUT_FILE = Path(f"wout_{VACUUM_NAME}.nc")

# Volume-averaged beta the pressure is calibrated to:
TARGET_BETA = 0.005
BETA_WEIGHT = 100.0               # on the relative beta error

# Flux surfaces the pwO residual is evaluated on, and its Boozer resolution:
SURFACES = np.linspace(0.1, 0.9, 5)
MBOZ = 16

# Initial values of the pwO map (a, b, c, eta_min), and which of them the
# optimizer may change; remove a name to hold it fixed:
MAP_PARAMETERS = dict(a=0.3, b=0.0, c=-2.0, eta_min=2.8)
FREE_MAP = ("a", "b", "c", "eta_min")

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend. Higher vacuum modes stay fixed:
MAX_MODES = [1, 2, 3]
MAX_NFEV = [20, 20, 40]
NS_ARRAY = np.array([35])
NITER_ARRAY = np.array([20000])
FTOL_ARRAY = np.array([1e-12])

# Targets:
ASPECT_TARGET = 6.0
ASPECT_WEIGHT = 10.0
IOTA_FLOOR = 0.31 if NFP == 1 else 0.42  # minimum |iota| over the profile
IOTA_WEIGHT = 1.0e4
MAGNETIC_WELL_FLOOR = 0.01
MAGNETIC_WELL_WEIGHT = 1.0e3
MERCIER_S_MIN = 0.2               # DMerc inward of this is set by the radial grid
MERCIER_MARGIN = 0.1              # DMerc floor on the optimizer grid; the final grid differs
MERCIER_WEIGHT = 100.0
MIRROR_LIMIT = 0.16               # maximum (Bmax-Bmin)/(Bmax+Bmin) on the pwO surfaces
AXIS_LIMIT = 0.25                 # maximum RMS helical excursion of the magnetic axis, metres
ELONGATION_LIMIT = 9.0
EDGE_JACOBIAN_FLOOR = 0.25        # minimum min/max Jacobian ratio in each of the three outer cells
PWO_WEIGHT = 1.0e3                # the seed's raw pwO variance is of order 0.01

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.05
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below the vacuum deck's or MINIMUM_MPOL. DMerc needs more modes
# than the boundary: at NFP 1 its geodesic term is 10x too large at mpol 7.
MINIMUM_MPOL = 13 if NFP == 1 else 7

# Verification solve of the optimized boundary:
FINAL_NS = [37, 45, 55, 71]        # restarted in small steps: a large jump flips the Jacobian
FINAL_FTOL = 1.0e-10
FINAL_NITER = 30000

# Every output file name contains this:
OUTPUT_NAME = f"pwO_finite_beta_nfp{NFP}"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]
    FINAL_NS, FINAL_FTOL = [31], 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = vj.VmecInput.from_file(INPUT_FILE)
mpol = max(inp.mpol, MINIMUM_MPOL)
inp = replace(inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4),
              am=np.array([1.0, -1.0]), pres_scale=1.0, ns_array=NS_ARRAY,
              ftol_array=FTOL_ARRAY, niter_array=NITER_ARRAY)
calibration = opt.solve_equilibrium(
    inp, initial_state=restart_state(vj.read_wout(WOUT_FILE), inp),
    raise_on_max_iterations=True)
inp = replace(inp, pres_scale=TARGET_BETA / float(calibration.wout.betatotal))

### Set up the objective ######################################################

def max_mirror(equilibrium_state, solver_context):
    """Full-surface mirror ratio on the pwO surfaces, which the mapped variance does not see."""
    return pwo.diagnostics_state(equilibrium_state, solver_context)[3]


def boozer_spectrum(equilibrium_state, solver_context):
    """The field and iota the map reads; FreeMapProblem turns these rows into the pwO residual."""
    return pwo.spectrum_state(equilibrium_state, solver_context)


def dmerc(equilibrium_state, solver_context):
    """DMerc from MERCIER_S_MIN to inside the edge; positive is stable."""
    values = opt.d_merc_state(equilibrium_state, solver_context)
    s = np.linspace(0.0, 1.0, values.shape[0])
    return values[(s >= MERCIER_S_MIN) & (s < 1.0)]


def mercier(equilibrium_state, solver_context):
    """DMerc hinge, log-bounded: DMerc diverges at zero shear and would swamp every other row."""
    return jnp.log1p(opt.Hinge(dmerc, floor=MERCIER_MARGIN)(equilibrium_state, solver_context))


# Each term is (function, target, weight).
pwo = MappedPwOResidual(SURFACES, mboz=MBOZ, nboz=MBOZ, oversample=2, limit_map=True,
                        **MAP_PARAMETERS)
objective_function_terms = [
    (boozer_spectrum, 0.0, 1.0),  # must stay first: FreeMapProblem reads these rows
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
    (opt.Hinge(opt.min_abs_iota, floor=IOTA_FLOOR), 0.0, IOTA_WEIGHT),
    (opt.Hinge(opt.magnetic_well, floor=MAGNETIC_WELL_FLOOR), 0.0, MAGNETIC_WELL_WEIGHT),
    (mercier, 0.0, MERCIER_WEIGHT),
    (opt.Hinge(max_mirror, ceiling=MIRROR_LIMIT), 0.0, 100.0),
    (opt.Hinge(opt.axis_excursion, ceiling=AXIS_LIMIT), 0.0, 100.0),
    (opt.Hinge(opt.edge_jacobian_ratio, floor=EDGE_JACOBIAN_FLOOR), 0.0, 100.0),
    (opt.Hinge(opt.max_elongation, ceiling=ELONGATION_LIMIT), 0.0, 10.0),
    (opt.volume_average_beta, TARGET_BETA, BETA_WEIGHT / TARGET_BETA**2),
]

report = opt.EquilibriumReporter(
    ("pwO total", pwo.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("beta", opt.volume_average_beta, ".3%"), ("min |iota|", opt.min_abs_iota, ".4f"),
    ("magnetic well", opt.magnetic_well, ".4f"),
    ("min DMerc", lambda state, rt: jnp.min(dmerc(state, rt)), ".3e"),
    ("axis excursion", opt.axis_excursion, ".4f"),
    ("max mirror", lambda state, rt: jnp.max(max_mirror(state, rt)), ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = seed = opt.solve_equilibrium(inp, initial_state=calibration.solution)
report("seed", equilibrium)
# Stages that share a boundary resolution share ONE problem, as in
# QA_optimization.py: every jitted graph is compiled once for the whole group.
# The resolution never drops below the deck's, which holds the higher vacuum modes.
resolution_of = {max_mode: max(max_mode + 2, inp.mpol) for max_mode in MAX_MODES}
group_max_mode = {
    max_mode: max(other for other in MAX_MODES
                  if resolution_of[other] == resolution_of[max_mode])
    for max_mode in MAX_MODES}
problem, mpol = None, None
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== finite-beta pwO stage, max_mode = {max_mode} =====")
    if resolution_of[max_mode] != mpol:
        mpol = resolution_of[max_mode]
        inp = replace(inp, delt=0.5).change_resolution(
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
    # Map variables are sigmoid-scaled, so a unit step spans their bounds.
    step = np.concatenate((PARAMETER_STEP * stage.scales, np.full(len(FREE_MAP), 0.5)))
    change = np.concatenate((MAX_PARAMETER_CHANGE * step[:joint.nx], np.full(len(FREE_MAP), np.inf)))
    result = least_squares(
        joint.residual, joint.x0, jac=joint.residual_jac,
        x_scale=step, max_nfev=max_nfev, bounds=(joint.x0 - change, joint.x0 + change),
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

### Print, plot and save ######################################################

final = report("final", final_equilibrium)
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   well_floor=0.0,
                   extra=[("min DMerc", final["min DMerc"], 0.0, "min"),
                          ("beta", final["beta"], 0.9 * TARGET_BETA, "min"),
                          ("beta", final["beta"], 1.1 * TARGET_BETA, "max")])
print(f"\npwO total {final['pwO total']:.3e}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")

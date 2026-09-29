#!/usr/bin/env python
"""Optimize a quasi-axisymmetric boundary against a quasilinear heat flux.

``QA_optimization.py`` with one more objective tuple: GKX's mixing-length
quasilinear heat flux ``gamma * W_Q / <k_perp^2>`` of the most unstable
gyrokinetic eigenmode, on a set of flux tubes chosen by physical radius ``s``
(normalized toroidal flux) and field-line label ``alpha``, reduced by a mean
or, with SOFTMAX_TEMPERATURE, by a smooth maximum. ``W_Q`` is the eigenmode's
heat flux per unit field energy and ``<k_perp^2>`` its field-weighted
perpendicular wavenumber, so the objective sees the mode structure, not only
its growth rate (``QA_optimization_turbulence_linear.py``).

Both weights depend on the eigenvector of a non-symmetric operator. GKX
differentiates the eigenpair implicitly (a bordered tangent system) in forward
and reverse mode, so SciPy receives VMEX's exact implicit Jacobian of every
residual row. Needs ``pip install 'vmex[turbulence]'``.

The mixing-length rule is a model of saturation with an uncalibrated
amplitude; ``QA_optimization_turbulence_nonlinear.py`` optimizes the heat flux
of the saturated nonlinear state instead.

Measured on a shared 36-core Xeon host (CPU only, 12 cores, load 50-80 from
other jobs, JAX 0.10.2, GKX main ahead of 2.4.2), default settings: 19 min end
to end, 6.1 GB peak memory. Compiling the residual and Jacobian took 318 s;
stage 1 then took 227 s for 5 Jacobian and 10 residual evaluations (45 s per
Jacobian), stage 2 317 s for 6 and 15 (53 s). The quasilinear heat flux at
s = 0.5 went 2.57 -> 0.94 while the aspect ratio came from 11.5 to 7.7 and the
quasisymmetry error from 0.039 to 0.045; the mean |iota| fell to 0.12, below
the 0.42 floor, which these weights do not enforce. On a loaded host run with
OPENBLAS_NUM_THREADS=1: there threaded LAPACK took 67 s for the dense
eigenvalue solve that one thread does in 5 s.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.turbulence import quasilinear_flux_proxy

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed. The exactly circular
# torus has zero first-order iota sensitivity; this gives the optimizer a QA basin:
SEED_PERTURBATION = 0.05

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 10)

# Mode ladder: highest boundary mode number varied in each stage, and the
# residual evaluations each stage may spend:
MAX_MODES = [1, 2]
MAX_NFEV = [10, 15]

# Targets:
ASPECT_TARGET = 6.0
MAGNETIC_WELL_TARGET = 0.01
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile

# Alternative settings for a larger design space:
#   MAX_MODES = [1, 3, 5, 7, 9]
#   MAX_NFEV = [15, 25, 30, 40, 50]
#   ASPECT_TARGET = 3.5
#   MAGNETIC_WELL_TARGET = 0.07

# Flux tubes: every (s, alpha) pair is one tube. s is the normalized toroidal
# flux, the same radius at every radial resolution. Add radii or lines here:
TUBE_S = [0.5]
TUBE_ALPHAS = [0.0]
SOFTMAX_TEMPERATURE = None        # None: mean over tubes; e.g. 0.01: smooth max

# Gyrokinetic model on each tube: drive in a/L, one binormal wavenumber
# ky = 2 pi / LY (in rho_ref units), and the Hermite-Laguerre resolution:
A_OVER_LT, A_OVER_LN = 3.0, 1.0
LY = 12.0
N_LAGUERRE, N_HERMITE = 4, 8
NTHETA = 32                       # parallel grid points over one poloidal turn

# Weight of the quasilinear heat-flux term, relative to its seed value (1 makes the
# seed's term cost 0.5, against 15 for the seed's aspect-ratio error):
FLUX_WEIGHT = 10.0

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
FINAL_NS = 71
FINAL_FTOL = 1.0e-14
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = "QA_turbulence_quasilinear_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]
    N_LAGUERRE, N_HERMITE, NTHETA = 2, 3, 16
    FINAL_NS, FINAL_FTOL = 31, 1.0e-10

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
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface.

    opt.mean_iota targets the average instead; opt.soft_min_abs_iota is the smooth minimum.
    """
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def heat_flux(state, runtime):
    """Mixing-length quasilinear heat flux, reduced over the flux tubes."""
    fluxes = jnp.stack([
        quasilinear_flux_proxy(
            state, runtime, s=s, alpha=alpha, ntheta=NTHETA,
            n_laguerre=N_LAGUERRE, n_hermite=N_HERMITE, ly=LY,
            a_over_lt=A_OVER_LT, a_over_ln=A_OVER_LN)
        for s in TUBE_S for alpha in TUBE_ALPHAS])
    if SOFTMAX_TEMPERATURE is None:
        return jnp.mean(fluxes)
    return SOFTMAX_TEMPERATURE * logsumexp(fluxes / SOFTMAX_TEMPERATURE)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"),
    ("QL heat flux", heat_flux, ".5f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = opt.solve_equilibrium(inp)
seed_flux = report("seed", equilibrium)["QL heat flux"]

# Each term is (function, target, weight): QA_optimization.py's list plus the
# turbulence term, normalized by its seed value.
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
    (heat_flux, 0.0, FLUX_WEIGHT / max(abs(seed_flux), 1.0e-6) ** 2),
]
# Stages that share a boundary resolution share ONE problem. A max_mode stage
# only frees more of the same boundary harmonics, so problem.subproblem() cuts
# the stage out of a problem built at the largest max_mode of its resolution
# group and freezes the rest; every jitted residual, Jacobian and predictor
# graph is then traced and compiled once for the whole group. Rebuilding per
# stage instead is a cache miss by construction -- the decision vector changes
# length, although MINIMUM_MPOL keeps every array shape inside the solve
# identical -- and on the shipped ladder that recompilation is about half the
# run. A stage that raises mpol still starts a new group.
resolution_of = {max_mode: max(max_mode + 2, MINIMUM_MPOL) for max_mode in MAX_MODES}
group_max_mode = {
    max_mode: max(other for other in MAX_MODES
                  if resolution_of[other] == resolution_of[max_mode])
    for max_mode in MAX_MODES}
problem, mpol = None, None
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QA stage, max_mode = {max_mode} =====")
    if resolution_of[max_mode] != mpol:
        mpol = resolution_of[max_mode]
        inp = replace(inp, delt=0.5).change_resolution(
            mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        # A RuntimeWarning about uncertified Jacobian columns is expected once
        # the optimizer leaves the seed and needs no action; see
        # examples/README.md.
        problem = opt.VmecProblem.from_tuples(
            inp, objective_function_terms, max_mode=group_max_mode[max_mode],
            vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
            ess_alpha=ESS_ALPHA, restart_from=equilibrium)
        x = problem.x0
        monitor.problem = problem
        if not ci_smoke:
            problem.compile_residual_and_jacobian()
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    step = PARAMETER_STEP * stage.scales
    start = time.perf_counter()
    result = least_squares(
        stage.residual, stage.x0, jac=stage.residual_jac,
        x_scale=step, max_nfev=max_nfev,
        bounds=(stage.x0 - MAX_PARAMETER_CHANGE * step,
                stage.x0 + MAX_PARAMETER_CHANGE * step),
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    print(f"stage wall time {time.perf_counter() - start:.0f} s for {result.nfev} "
          f"residual and {result.njev} Jacobian evaluations")
    x = stage.embed(result.x)
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"mode {max_mode}", equilibrium)

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(
    inp, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

final_flux = report("final", final_equilibrium)["QL heat flux"]
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   well_floor=MAGNETIC_WELL_TARGET,
                   extra=[("QL heat flux", final_flux, seed_flux, "max")])
print(f"\nQL heat flux {seed_flux:.5f} -> {final_flux:.5f} at NS = {FINAL_NS}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")

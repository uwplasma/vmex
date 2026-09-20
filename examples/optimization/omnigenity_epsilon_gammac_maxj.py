#!/usr/bin/env python
"""Combined effective-ripple / Gamma_c / maximum-J confinement optimization.

Fast-ion refinement of the finite-beta Landreman--Buller--Drevlak QA: one
scalar objective adds the quasisymmetry ratio residual (the differentiable
proxy that controls effective ripple), the derivative-safe ``GammaCSmooth``
surrogate, and the outer-volume maximum-J residual at fixed physical pitch.
Each confinement term is normalized by its seed value so the weights are
dimensionless preferences, one L-BFGS-B call drives the boundary through one
reverse implicit adjoint per gradient, and force balance is a constraint by
construction: every evaluation is a converged hot-restarted equilibrium.

Policy: optimize the surrogate, report the hard values.  Hard ``Gamma_c``
and NEO_JAX ``epsilon_eff^(3/2)`` are evaluated before and after at the
shared optimization resolution.  Minimizing the surrogate does not promise
zero prompt orbit losses -- certify a design with collisionless tracing
(``examples/vmex_essos_workflow.py``) before quoting loss fractions.  The
bootstrap-consistent current profile is frozen during this refinement; see
``QA_maxJ_continuation.py`` for the self-consistent Redl loop.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.gammac import GammaC, GammaCSmooth
from vmex.core.maxj import MaximumJResidual, common_trapped_pitches_state

# The seed deck: the finite-beta Landreman-Buller-Drevlak QA:
INPUT_FILE = (Path(__file__).resolve().parents[1] / "data"
              / "input.LandremanPaul2021_QA_beta2p5_bootstrap")

# Flux surfaces each term is evaluated on. The quasisymmetry ratio is the
# differentiable proxy for effective ripple; Gamma_c is sampled on fewer
# surfaces because it is expensive; maximum-J is an outer-volume quantity,
# where pressure can reverse the precession; EPS_SURFACES is the hard NEO_JAX
# validation, never differentiated:
QS_SURFACES = np.linspace(0.1, 0.9, 8)
GC_SURFACES = (0.35, 0.6, 0.85)
MAXJ_SURFACES = np.array([0.6, 0.7, 0.8, 0.9])
EPS_SURFACES = (0.25, 0.5, 0.75)

# Highest boundary Fourier mode number varied, and the L-BFGS-B iterations:
MAX_MODE = 2
MAXITER = 20

# Weights of the three seed-normalized confinement terms:
W_EPS = 1.0
W_GC = 1.0
W_MAXJ = 1.0

# Constraints. The transform floor guards the seed rather than pushing it: it
# sits IOTA_MARGIN below the seed minimum, so it activates only on degradation:
IOTA_MARGIN = 0.95
MAXJ_TARGET = -0.01
TRAPPING_DEPTHS = (0.4, 0.8)

# Gamma_c surrogate temperature and quadrature, and the bounce-ACTION plan:
GC_TEMPERATURE = 0.15
GC_BUDGET = dict(nalpha=7, num_transit=3, points_per_transit=64,
                 num_pitch=24, quadrature_order=32)
ACTION = dict(nalpha=7, points_per_period=32, num_periods=8,
              max_wells=20, quadrature_order=24)
ACTION_MBOZ = 10

# Research resolution. The hard Gamma_c carries 10-20 % scatter at the shipped
# budgets, so quote the budget with any number taken from this script:
#   MAX_MODE, MAXITER = 3, 40
#   GC_BUDGET = dict(nalpha=9, num_transit=5, points_per_transit=128,
#                    num_pitch=48, quadrature_order=32)
#   ACTION = dict(nalpha=9, points_per_period=48, num_periods=10,
#                 max_wells=24, quadrature_order=32)

# NEO_JAX controls for the hard effective-ripple validation:
NEO_CONFIG_ARGS = dict(theta_n=24, phi_n=24, npart=12, multra=1, no_bins=20,
                       nstep_per=6, nstep_min=30, nstep_max=60, acc_req=0.1,
                       max_rational_field_periods=100000)

# Step control:
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
STAGE_MAX_ITERATIONS = 3000       # forward solve cap inside an optimizer trial

# Radial grid every optimizer trial is solved on; the deck's own is used at the
# shipped settings:
STAGE_OVERRIDES = {}

# Every output file name contains this:
OUTPUT_NAME = "QA_eps_gammac_maxJ"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODE, MAXITER = 1, 2
    QS_SURFACES, GC_SURFACES = np.linspace(0.2, 0.8, 4), (0.5,)
    GC_TEMPERATURE = 0.2
    GC_BUDGET = dict(nalpha=5, num_transit=2, points_per_transit=32,
                     num_pitch=12, quadrature_order=16)
    # The bounce-ACTION trace cannot be shortened further: a coarser plan loses
    # the matched wells on the outer surfaces and returns NaN slopes.
    ACTION = dict(nalpha=7, points_per_period=32, num_periods=8,
                  max_wells=20, quadrature_order=16)
    NEO_CONFIG_ARGS = dict(theta_n=16, phi_n=16, npart=8, multra=1, no_bins=12,
                           nstep_per=4, nstep_min=20, nstep_max=40, acc_req=0.2,
                           max_rational_field_periods=100000)
    STAGE_MAX_ITERATIONS = 100
    STAGE_OVERRIDES = dict(ns_array=np.array([13]), ftol_array=np.array([1e-9]),
                           niter_array=np.array([3000]))

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE), **STAGE_OVERRIDES)
equilibrium = opt.solve_equilibrium(inp)
state0, rt0 = equilibrium.solution, equilibrium.solver_context

### Set up the objective ######################################################

# One physical lambda must describe the same trapped particles on every
# surface and field-line label; select it once at the seed and keep the
# pitch grid static for the whole stage.
pitch = np.asarray(common_trapped_pitches_state(
    state0, rt0, MAXJ_SURFACES, TRAPPING_DEPTHS,
    mboz=ACTION_MBOZ, nboz=ACTION_MBOZ, nalpha=ACTION["nalpha"],
    points_per_period=ACTION["points_per_period"],
    num_periods=ACTION["num_periods"]))

qs = opt.QuasisymmetryRatioResidual(QS_SURFACES, helicity_m=1, helicity_n=0)
gamma_c_smooth = GammaCSmooth(GC_SURFACES, temperature=GC_TEMPERATURE, **GC_BUDGET)
gamma_c_hard = GammaC(GC_SURFACES, **GC_BUDGET)  # report-only, never differentiated
maximum_j = MaximumJResidual(MAXJ_SURFACES, pitch, mboz=ACTION_MBOZ,
                             nboz=ACTION_MBOZ, target=MAXJ_TARGET, **ACTION)

SCALES = {"QS": float(qs.total_state(state0, rt0)),
          "GammaCSmooth": float(gamma_c_smooth.total_state(state0, rt0)),
          "maxJ": float(maximum_j.total_state(state0, rt0))}
for name, scale in SCALES.items():
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"seed {name} total {scale} cannot normalize the objective")
ASPECT0 = float(opt.aspect_ratio(state0, rt0))
BETA0 = float(opt.volume_average_beta(state0, rt0))
# Guard the seed's transform rather than push it: the floor sits just below
# the seed minimum so the constraint activates only on degradation.
IOTA_FLOOR = IOTA_MARGIN * float(opt.min_abs_iota(state0, rt0))


def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


constraint_terms = [
    (opt.aspect_ratio, ASPECT0, 1.0),
    (iota_floor, 0.0, 10.0),
    (opt.volume_average_beta, BETA0, 1.0 / BETA0**2),
]


def normalized_terms(equilibrium_state, solver_context):
    return (qs.total_state(equilibrium_state, solver_context) / SCALES["QS"],
            gamma_c_smooth.total_state(equilibrium_state, solver_context)
            / SCALES["GammaCSmooth"],
            maximum_j.total_state(equilibrium_state, solver_context) / SCALES["maxJ"])


def loss(equilibrium_state, solver_context):
    eps_proxy, gc_term, mj_term = normalized_terms(equilibrium_state, solver_context)
    rows = opt.residuals_from_tuples(
        equilibrium_state, solver_context, constraint_terms)
    return (W_EPS * eps_proxy + W_GC * gc_term + W_MAXJ * mj_term
            + 0.5 * jnp.vdot(rows, rows))


def print_terms(label, eq):
    eps_proxy, gc_term, mj_term = (
        float(v) for v in normalized_terms(eq.solution, eq.solver_context))
    rows = np.asarray(opt.residuals_from_tuples(
        eq.solution, eq.solver_context, constraint_terms))
    print(f"[{label}] w_eps*QS/QS0 = {W_EPS * eps_proxy:.4f}, "
          f"w_gc*GammaCSmooth/G0 = {W_GC * gc_term:.4f}, "
          f"w_maxj*maxJ/J0 = {W_MAXJ * mj_term:.4f}, "
          f"constraints = {0.5 * float(rows @ rows):.4e}")


def hard_confinement(eq):
    """Hard validation metrics; the surrogate never appears in a quoted number."""
    row = {"gamma_c": np.asarray(
        gamma_c_hard.compute_state(eq.solution, eq.solver_context)["gamma_c"])}
    maxj = maximum_j.compute_state(eq.solution, eq.solver_context)
    row["maxj_total"] = float(maxj["total"])
    row["maxj_fraction"] = float(maxj["maximum_j_fraction"])
    row["qs_total"] = float(qs.total_state(eq.solution, eq.solver_context))
    try:
        from neo_jax import NeoConfig  # optional: pip install vmex[neoclassical]

        config = NeoConfig(**NEO_CONFIG_ARGS)
        row["eps32"] = np.asarray(vj.epsilon_effective_from_wout(
            eq.wout, surfaces=EPS_SURFACES, config=config)[1])
    except ImportError as error:  # state the reason; never report a zero
        print(f"epsilon_eff unavailable: {error}")
        row["eps32"] = None
    return row


report = opt.EquilibriumReporter(
    ("QS", qs.total, ".4e"), ("GammaCSmooth", gamma_c_smooth.total, ".4e"),
    ("maxJ", maximum_j.total, ".4e"), ("aspect", opt.aspect_ratio, ".3f"),
    ("min |iota|", opt.min_abs_iota, ".3f"),
    ("beta", opt.volume_average_beta, ".3%"))
monitor = opt.OptimizationMonitor()
report("seed", equilibrium)
print_terms("seed", equilibrium)
before = hard_confinement(equilibrium)

### Run the optimization ######################################################

problem = opt.VmecProblem.from_loss(
    inp, loss, max_mode=MAX_MODE, use_ess=True, ess_alpha=ESS_ALPHA,
    restart_from=equilibrium,
    forward_max_iterations=STAGE_MAX_ITERATIONS)
print(f"dof_names = {problem.dof_names}")
monitor.problem = problem
problem.compile_value_and_gradient()

x0, step = problem.x0, PARAMETER_STEP * problem.scales


def value_and_gradient(y):
    value, gradient = problem.value_and_grad(x0 + step * y)
    evaluation_costs.append(float(value))
    return monitor.cache_evaluation(x0 + step * y, value, step * gradient)


def monitor_y(intermediate_result):
    monitor({"x": x0 + step * intermediate_result.x,
             "fun": intermediate_result.fun})


evaluation_costs = []
result = minimize(
    value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
    bounds=[(-MAX_PARAMETER_CHANGE, MAX_PARAMETER_CHANGE)] * x0.size,
    callback=monitor_y,
    options={"maxiter": MAXITER, "gtol": 1.0e-8, "ftol": 1.0e-12,
             "maxls": 20, "maxcor": 20})
# The scalar cost is evaluated on the optimizer's internal jitted re-solve.
# At a loose smoke-lane ftol that re-solve can sit on a different point of
# the ftol ball than the seed solve, so this line need not start at exactly
# W_EPS + W_GC + W_MAXJ; every quoted physics number below comes from the
# materialized equilibria, whose before/after lineage is one hot-restart
# chain from the seed.
print(f"optimizer scalar cost: {evaluation_costs[0]:.16e} -> {float(result.fun):.16e}")

x_final = x0 + step * result.x
final_input = problem.input_from_x(x_final)
final_equilibrium = problem.equilibrium_from_x(x_final)
report("final", final_equilibrium)
print_terms("final", final_equilibrium)

### Check the result ##########################################################

# Hard before/after table, at the same radial and sampling resolution.
after = hard_confinement(final_equilibrium)
gc_before, gc_after = before["gamma_c"].mean(), after["gamma_c"].mean()
print(f"\nhard confinement, before -> after (s = {list(GC_SURFACES)}):")
print(f"hard Gamma_c per surface = {np.array2string(before['gamma_c'], precision=4)} -> "
      f"{np.array2string(after['gamma_c'], precision=4)}")
print(f"hard Gamma_c mean = {gc_before:.4e} -> {gc_after:.4e}")
if before["eps32"] is not None and after["eps32"] is not None:
    print(f"epsilon_eff^(3/2) mean = {before['eps32'].mean():.4e} -> "
          f"{after['eps32'].mean():.4e} (s = {list(EPS_SURFACES)})")
print(f"maximum-J residual = {before['maxj_total']:.4e} -> {after['maxj_total']:.4e}, "
      f"maximum-J fraction = {before['maxj_fraction']:.1%} -> {after['maxj_fraction']:.1%}")
print(f"QS ratio total = {before['qs_total']:.4e} -> {after['qs_total']:.4e}")

### Print, plot and save ######################################################

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".", j_pitch=float(pitch[0])).values():
    print(f"Wrote {path}")

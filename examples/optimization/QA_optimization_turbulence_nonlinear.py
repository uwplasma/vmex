#!/usr/bin/env python
"""Optimize a quasi-axisymmetric boundary against the nonlinear GKX heat flux.

The objective list of ``QA_optimization.py`` plus one tuple: the ion heat flux
of a saturated nonlinear gyrokinetic simulation, averaged over a finite window
of WINDOW_STEPS time steps, on a set of flux tubes chosen by physical radius
``s`` (normalized toroidal flux) and field-line label ``alpha``, reduced by a
mean or, with SOFTMAX_TEMPERATURE, by a smooth maximum.

Each tube is run to saturation once per stage, outside the derivative, and the
run stops only when GKX's saturation gate accepts the heat-flux trace (at least
256 retained samples spanning 20 autocorrelation times, relative standard
error <= 5 %, both half-windows agreeing); a tube that does not saturate
within MAX_SATURATION_STEPS stops the script. Stages after the first restart
from the previous stage's saturated state. The saturated state is then held
fixed and the window is differentiated exactly (GKX's checkpointed discrete
adjoint) through the flux-tube geometry. That is the derivative of a finite
window from a fixed start, not of the long-time mean: accepted designs need an
independent cold-start check before a transport reduction is claimed.

A least-squares Jacobian would push one forward tangent per boundary
coefficient through the whole window. This script instead sums the residual
rows into one scalar, as ``QA_optimization_scalar.py`` does, so SciPy L-BFGS-B
receives a value and a gradient from one reverse sweep through the
checkpointed window and one reverse equilibrium adjoint. Needs
``pip install 'vmex[turbulence]'``.

Measured on a shared 36-core Xeon host (CPU only, 12 cores, load ~60 from
other jobs, JAX 0.10.2, GKX 2.4.0), default settings: 70 min end to end,
5.5 GB peak memory. The cold saturation of the seed took 35,696 steps
(1,083 s); the warm restarts after each stage 2,480 and 6,296 steps (99 s,
136 s). One value and gradient took 69 s in stage 1 (22 evaluations, 1,527 s)
and 60 s in stage 2 (19, 1,139 s). The gated saturated heat flux at s = 0.5
went 53.0 +- 2.6 (seed) -> 36.3 +- 1.8 (stage 1) -> 42.6 +- 2.1 (stage 2,
which traded flux for quasisymmetry: QS error 2.29 -> 0.75), with the aspect
ratio 11.5 -> 6.0. That is one tube, one seed and no held-out check: it shows
the pipeline works, not a certified transport reduction.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import logsumexp
from scipy.optimize import minimize

import gkx
from gkx.diagnostics.saturation import SaturationStopConfig, saturation_stop_decision
import vmex as vj
from vmex import optimize as opt
from vmex.core.turbulence import flux_tube_geometry

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed:
SEED_PERTURBATION = 0.05

# Flux surfaces the quasisymmetry residual is evaluated on:
SURFACES = np.linspace(0.1, 1.0, 10)

# Mode ladder: highest boundary mode number varied in each stage, and the
# L-BFGS-B iterations each stage may spend:
MAX_MODES = [1, 2]
MAXITER = [8, 8]

# Targets:
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile

# Flux tubes: every (s, alpha) pair is one tube, saturated separately:
TUBE_S = [0.5]
TUBE_ALPHAS = [0.0]
SOFTMAX_TEMPERATURE = None        # None: mean over tubes

# Nonlinear gyrokinetic model: ion drive in a/L, box, resolution, time step:
A_OVER_LT, A_OVER_LN = 3.0, 1.0
NX, NY, NZ = 8, 8, 16
LX, LY = 62.8, 62.8
N_LAGUERRE, N_HERMITE = 4, 8
DT = 0.05

# Saturation: a heat-flux sample every SAMPLE_STEPS steps, at most
# MAX_SATURATION_STEPS steps per tube and stage, and the relative standard
# error of the saturated mean the gate requires:
SAMPLE_STEPS = 8
MAX_SATURATION_STEPS = 60_000
SATURATION_REL_SEM = 0.05

# Differentiated post-saturation window (below GKX's 1024-step divergence knee):
WINDOW_STEPS = 512

# Weight of the heat-flux term, relative to its seed value (1 makes the
# seed's term cost 0.5, against 15 for the seed's aspect-ratio error):
FLUX_WEIGHT = 10.0

# Step control, as in QA_optimization_scalar.py:
PARAMETER_STEP = 0.02
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2

# Equilibrium resolution: mode numbers max_mode + 2, never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Every output file name contains this:
OUTPUT_NAME = "QA_turbulence_nonlinear_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs; it wires
# every piece together but does not reach saturation:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAXITER = [1], [2]
    NX, NY, NZ, N_LAGUERRE, N_HERMITE = 4, 4, 8, 2, 3
    SAMPLE_STEPS, MAX_SATURATION_STEPS, WINDOW_STEPS = 2, 8, 2

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

### Set up the gyrokinetic model ##############################################

TUBES = [(s, alpha) for s in TUBE_S for alpha in TUBE_ALPHAS]
grid = gkx.build_spectral_grid(gkx.GridConfig(Nx=NX, Ny=NY, Nz=NZ, Lx=LX, Ly=LY))
terms = gkx.TermConfig(collisions=1.0, hypercollisions=1.0, hyperdiffusion=1.0,
                       end_damping=1.0, nonlinear=1.0, apar=0.0, bpar=0.0)


def tube_geometry(state, runtime, tube):
    s, alpha = tube
    return flux_tube_geometry(state, runtime, s=s, alpha=alpha, ntheta=NZ)


def gk_parameters(geometry):
    """Ion drive plus the velocity-space and grid-scale dissipation GKX needs."""
    return gkx.LinearParams(
        tprim=A_OVER_LT, fprim=A_OVER_LN, nu=0.01,
        kpar_scale=geometry.gradpar_value, nu_hermite=1.0, nu_laguerre=2.0,
        p_hyper_m=float(min(20, max(N_HERMITE // 2, 1))),
        hypercollisions_const=0.0, hypercollisions_kz=1.0,
        D_hyper=0.05, p_hyper_kperp=2.0,
        damp_ends_amp=0.1, damp_ends_widthfrac=0.125)


def noise_seed():
    """Small Hermitian density perturbation to grow the turbulence from."""
    state = jnp.zeros((1, N_LAGUERRE, N_HERMITE, NY, NX, NZ), dtype=jnp.complex64)
    profile = 1.0e-3 * (1.0 + 0.2 * jnp.cos(grid.z))
    for ky, kx, amplitude in ((1, 0, 1.0 + 0.2j), (1, 1, 0.3 - 0.1j)):
        state = state.at[0, 0, 0, ky, kx].set(amplitude * profile)
        state = state.at[0, 0, 0, -ky, -kx].set(np.conj(amplitude) * profile)
    return state


def window_flux(saturated_state, geometry, steps):
    return gkx.nonlinear_heat_flux_window(
        saturated_state, grid, geometry, gk_parameters(geometry), DT, steps,
        terms=terms, method="rk3")


saturation_traces = []           # (label, time, heat flux, gate decision) per run


def saturate(equilibrium, seeds, label):
    """Run every tube to an accepted saturated state (outside the derivative)."""
    states = []
    for tube, seed in zip(TUBES, seeds):
        geometry = tube_geometry(equilibrium.state, equilibrium.runtime, tube)
        state, fluxes, start = seed, [], time.perf_counter()
        for chunk in range(MAX_SATURATION_STEPS // SAMPLE_STEPS):
            state = gkx.integrate_nonlinear(
                state, grid, geometry, gk_parameters(geometry), DT, SAMPLE_STEPS,
                method="rk3", terms=terms, return_fields=False)
            fluxes.append(float(window_flux(state, geometry, 1)))
            times = DT * SAMPLE_STEPS * np.arange(1, len(fluxes) + 1)
            decision = saturation_stop_decision(
                times, fluxes, config=SaturationStopConfig(rel_sem=SATURATION_REL_SEM))
            if decision["saturated"]:
                break
        steps = (chunk + 1) * SAMPLE_STEPS
        print(f"tube s={tube[0]}, alpha={tube[1]}: {steps} steps, "
              f"{time.perf_counter() - start:.1f} s, "
              f"saturated={decision['saturated']}, "
              f"Q = {decision['mean']} +- {decision['sem']}", flush=True)
        if not decision["saturated"] and not ci_smoke:
            raise RuntimeError(
                f"tube {tube} did not saturate in {MAX_SATURATION_STEPS} steps: "
                f"{decision['reasons']}")
        saturation_traces.append(
            (f"{label}, s={tube[0]}, alpha={tube[1]}", times, fluxes, decision))
        states.append(state)
    return states


### Set up the objective ######################################################

def iota_floor(state, runtime):
    """Hinge on the profile minimum of |iota|."""
    return jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0.0)


def heat_flux(state, runtime):
    """Post-saturation window heat flux, reduced over the flux tubes."""
    fluxes = jnp.stack([
        window_flux(saturated, tube_geometry(state, runtime, tube), WINDOW_STEPS)
        for tube, saturated in zip(TUBES, saturated_states)])
    if SOFTMAX_TEMPERATURE is None:
        return jnp.mean(fluxes)
    return SOFTMAX_TEMPERATURE * logsumexp(fluxes / SOFTMAX_TEMPERATURE)


qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("heat flux", heat_flux, ".5e"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

equilibrium = opt.solve_equilibrium(inp)
saturated_states = saturate(equilibrium, [noise_seed() for _ in TUBES], "seed")
seed_flux = report("seed", equilibrium)["heat flux"]
flux_scale = max(abs(seed_flux), 1.0e-6)

# Each term is (function, target, weight); the flux is normalized by its seed.
objective_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (heat_flux, 0.0, FLUX_WEIGHT / flux_scale**2),
]

for max_mode, maxiter in zip(MAX_MODES, MAXITER):
    print(f"\n===== QA + nonlinear heat flux stage, max_mode = {max_mode} =====")

    # Defined per stage: a new function is traced afresh, so it sees this
    # stage's saturated states rather than a compiled copy of the last ones.
    def loss(state, runtime):
        """The scalar that is differentiated: 0.5 * r.T @ r over all residual rows."""
        rows = opt.residuals_from_tuples(state, runtime, objective_terms)
        return 0.5 * jnp.vdot(rows, rows)

    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    problem = opt.VmecProblem.from_loss(
        inp, loss, max_mode=max_mode, use_ess=True, ess_alpha=ESS_ALPHA,
        restart_from=equilibrium)
    monitor.problem = problem
    x0 = problem.x0
    step = PARAMETER_STEP * problem.scales

    def value_and_gradient(y):
        """What SciPy calls. The monitor caches it, so its callback never re-solves."""
        start = time.perf_counter()
        value, gradient = problem.value_and_grad(x0 + step * y)
        print(f"  value and gradient: {time.perf_counter() - start:.1f} s", flush=True)
        return monitor.cache_evaluation(x0 + step * y, value, step * gradient)

    def record(intermediate_result):
        monitor({"x": x0 + step * intermediate_result.x, "fun": intermediate_result.fun})

    start = time.perf_counter()
    initial_value = float(value_and_gradient(np.zeros_like(x0))[0])
    result = minimize(
        value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
        bounds=[(-MAX_PARAMETER_CHANGE, MAX_PARAMETER_CHANGE)] * x0.size,
        callback=record, options={"maxiter": maxiter, "maxls": 10})
    print(f"optimizer scalar cost: {initial_value:.12e} -> {float(result.fun):.12e}")
    print(f"stage wall time {time.perf_counter() - start:.0f} s for "
          f"{result.nfev + 1} value-and-gradient evaluations")
    x = x0 + step * result.x
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"mode {max_mode}, stage state", equilibrium)
    # Restart every tube from its last saturated state in the new geometry.
    saturated_states = saturate(equilibrium, saturated_states, f"mode {max_mode}")
    report(f"mode {max_mode}, re-saturated", equilibrium)

### Print, plot and save ######################################################

final_flux = report("final", equilibrium)["heat flux"]
opt.report_targets(equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   extra=[("heat flux", final_flux, seed_flux, "max")])
print(f"\nwindow heat flux {seed_flux:.5e} -> {final_flux:.5e} "
      "(one differentiated window each, a noisy estimate)")
# The number to quote is the gated saturated mean with its standard error.
for label, _, _, decision in saturation_traces:
    print(f"saturated Q [{label}] = {decision['mean']} +- {decision['sem']}")

input_path = inp.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
figure, axis = plt.subplots(figsize=(7, 4))
for label, times, fluxes, _ in saturation_traces:
    axis.plot(times, fluxes, label=label)
axis.set(xlabel="time (a / v_thi, from each restart)", ylabel="ion heat flux Q",
         title="Saturation runs, one per tube and stage")
axis.legend(fontsize=8)
figure.savefig(f"{OUTPUT_NAME}_saturation.png", dpi=120, bbox_inches="tight")
print(f"Wrote {OUTPUT_NAME}_saturation.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")

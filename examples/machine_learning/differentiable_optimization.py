#!/usr/bin/env python
"""Reshape a vacuum stellarator with JAX gradients and a selectable optimizer.

Run from the repository root: python examples/machine_learning/differentiable_optimization.py
The Adam option needs the optimizer extra: pip install 'vmex[optimizers]'.
The six normalized inputs move selected boundary Fourier coefficients; VMEX
solves force balance and differentiates that converged equilibrium implicitly.
This example lowers aspect ratio from six to five while targeting rotational
transform. It reports the resulting loss of quasi-axisymmetry explicitly: a
compactness objective does not optimize confinement. The grid remains coarse.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.plotting import surface_rz

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.LandremanPaul2021_QA_lowres"
OUTPUT_DIR = Path("results/machine_learning")
DOF_NAMES = ["RBC(0,1)", "ZBS(0,1)", "RBC(1,0)", "ZBS(1,0)", "RBC(1,1)", "ZBS(1,1)"]
MAX_MODE = 1                    # raise this when adding higher-order DOF_NAMES
BOUNDARY_STEP = 0.05             # metres per unit of the dimensionless input x
X_LIMIT = 1.0                    # each selected coefficient can move by up to 5 cm
IOTA_TARGET, ASPECT_TARGET = 0.45, 5.0
IOTA_SCALE, ASPECT_SCALE = 0.05, 0.5
METHOD = "L-BFGS-B"              # "Adam" keeps the explicit Optax update loop
LBFGSB_MAXITER, ADAM_STEPS, LEARNING_RATE = 20, 50, 0.08
QS_SURFACES = (0.25, 0.5, 0.75)  # report QA degradation; not part of the target loss
NS, NS_VERIFY, MPOL, NTOR = 21, 31, 5, 5
FTOL, MAX_ITERATIONS = 1e-12, 5000
FD_STEPS = (0.0003, 0.001, 0.003) # dimensionless; 15--150 micrometres in coefficient space

# Fix the major radius RBC(0,0), toroidal flux and zero pressure/current.
# Only the six named coefficients vary; no uniform size or magnetic-field scale is learned.
inp = vj.VmecInput.from_file(INPUT_FILE)
inp = replace(
    inp, lfreeb=False, pres_scale=0.0, curtor=0.0, ncurr=1, ac=np.zeros_like(inp.ac),
    ns_array=np.array([NS]), ftol_array=np.array([FTOL]),
    niter_array=np.array([MAX_ITERATIONS]), delt=0.5,
).change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=2 * NTOR + 4)


def physical_loss(state, runtime):
    """A dimensionless scalar loss evaluated on the solved equilibrium."""
    return 0.5 * (
        ((opt.mean_iota(state, runtime) - IOTA_TARGET) / IOTA_SCALE) ** 2
        + ((opt.aspect_ratio(state, runtime) - ASPECT_TARGET) / ASPECT_SCALE) ** 2)


def metrics(equilibrium):
    """Physical labels in the order [mean_iota, aspect_ratio]."""
    return np.array([float(f(equilibrium.state, equilibrium.runtime))
                     for f in (opt.mean_iota, opt.aspect_ratio)])


def plot_training(losses, values, equilibria, path):
    """Plot the target loss and the independently verified boundary change."""
    fig, grid = plt.subplots(2, 3, figsize=(11, 6.4), constrained_layout=True)
    axes = grid[0]
    axes[0].semilogy(losses, "o-")
    axes[0].set(xlabel=f"{METHOD} iteration", ylabel="Dimensionless loss")
    for index, (label, target) in enumerate(zip(("Mean rotational transform", "Aspect ratio"),
                                             (IOTA_TARGET, ASPECT_TARGET))):
        axes[index + 1].plot(np.asarray(values)[:, index], "o-")
        axes[index + 1].axhline(target, color="black", linestyle="--", linewidth=1)
        axes[index + 1].set(xlabel=f"{METHOD} iteration", ylabel=label)
    theta = np.linspace(0, 2 * np.pi, 200)
    wouts = [equilibrium.wout for equilibrium in equilibria]
    radius, height = [], []
    for ax, fraction in zip(grid[1], (0, 0.25, 0.5)):
        for wout, label, style in zip(wouts, ("Initial", "Selected"), ("-", "--")):
            r, z = surface_rz(wout, s_index=wout.ns-1, theta=theta,
                              phi=np.array([fraction * 2 * np.pi / wout.nfp]))
            ax.plot(r[:, 0], z[:, 0], style, label=label)
            radius.extend(r[:, 0]); height.extend(z[:, 0])
        ax.set(xlabel="R [m]", ylabel="Z [m]", title=f"Toroidal period fraction {fraction:g}", aspect="equal")
        ax.legend(frameon=False, fontsize=8)
    for ax in grid[1]:
        ax.set(xlim=(min(radius)-0.02, max(radius)+0.02), ylim=(min(height)-0.02, max(height)+0.02))
    fig.savefig(path, dpi=160)
    plt.close(fig)

started = perf_counter()
problem = opt.VmecProblem.from_loss(
    inp, physical_loss, max_mode=MAX_MODE, vary_major_radius=False,
    use_ess=False, progress=True).subproblem(active=DOF_NAMES)
assert problem.dof_names == tuple(DOF_NAMES)
x_seed = jnp.asarray(problem.x0)


def loss(x):
    """x -> boundary -> converged equilibrium -> scalar objective."""
    return problem.jax_fun(x_seed + BOUNDARY_STEP * x)


# This is ordinary JAX differentiation. VMEX supplies the equilibrium VJP;
# it solves one adjoint system rather than storing all equilibrium iterations.
value_and_grad = jax.value_and_grad(loss)
x = jnp.zeros_like(x_seed)
tick = perf_counter()
value, gradient = value_and_grad(x)
jax.block_until_ready((value, gradient))
cold_gradient_seconds = perf_counter() - tick
tick = perf_counter()
value, gradient = value_and_grad(x)
jax.block_until_ready((value, gradient))
warm_gradient_seconds = perf_counter() - tick
print(f"First value/gradient: {cold_gradient_seconds:.2f} s; warm: {warm_gradient_seconds:.2f} s")

# Check a directional derivative using two separate, cold equilibrium solves.
# Neither finite-difference value calls value_and_grad or reuses an adjoint.
if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
    raise FloatingPointError("Initial loss or gradient is non-finite; inspect the equilibrium")
if np.linalg.norm(gradient) == 0:
    raise ValueError("Initial gradient is zero; choose targets away from the initial equilibrium")
direction = np.asarray(gradient) / np.linalg.norm(gradient)
ad = float(jnp.vdot(gradient, direction))
fd_derivatives = []
for step in FD_STEPS:
    fd_values = []
    for sign in (+1, -1):
        perturbed_input = problem.input_from_x(x_seed + BOUNDARY_STEP * sign * step * direction)
        equilibrium = opt.solve_equilibrium(perturbed_input, raise_on_max_iterations=True)
        fd_values.append(float(physical_loss(equilibrium.state, equilibrium.runtime)))
    fd_derivatives.append((fd_values[0] - fd_values[1]) / (2 * step))
fd_errors = np.abs(ad / np.asarray(fd_derivatives) - 1)
print(f"Directional derivative AD={ad:.8e}; FD steps={FD_STEPS}, relative errors={fd_errors}")
np.testing.assert_allclose(fd_derivatives, ad, rtol=1e-3, atol=1e-6)

# The solver and differentiable loss are shared by both optimizers.
x_history, loss_history, metrics_history = [], [], []


def record(candidate):
    """Record accepted optimizer points, with a force-balance check."""
    candidate = np.asarray(candidate)
    if x_history and np.array_equal(candidate, x_history[-1]):
        return
    value = float(loss(jnp.asarray(candidate)))
    equilibrium = problem.equilibrium_from_x(x_seed + BOUNDARY_STEP * candidate)
    fsq = float(equilibrium.result.fsqr + equilibrium.result.fsqz + equilibrium.result.fsql)
    if not np.isfinite(value) or not np.isfinite(fsq) or (
            not equilibrium.result.converged and fsq > 100 * FTOL):
        raise RuntimeError(f"Uncertified optimizer point; loss={value}, FSQ={fsq:.3e}")
    x_history.append(candidate.copy())
    loss_history.append(value)
    metrics_history.append(metrics(equilibrium))
    print(f"iteration {len(x_history)-1:2d}: loss={value:.6e}, "
          f"mean iota={metrics_history[-1][0]:.6f}, aspect={metrics_history[-1][1]:.6f}")


def objective_and_gradient(candidate):
    """The ordinary (scalar, gradient-vector) contract used by optimizers."""
    value, gradient = value_and_grad(jnp.asarray(candidate))
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("Non-finite loss or gradient; reduce the step size or design box")
    return float(value), np.asarray(gradient)


record(x)
if METHOD == "L-BFGS-B":
    result = minimize(
        objective_and_gradient, np.asarray(x), jac=True, method=METHOD,
        bounds=[(-X_LIMIT, X_LIMIT)] * len(x), callback=record,
        options=dict(maxiter=LBFGSB_MAXITER, maxls=12, ftol=1e-10, gtol=1e-6))
    record(result.x)
    optimizer_evaluations, optimizer_message = result.nfev, str(result.message)
elif METHOD == "Adam":
    import optax
    # Replace this transform with any Optax optimizer; the VMEX loss is unchanged.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
    optimizer_state = optimizer.init(x)
    for iteration in range(ADAM_STEPS):
        value, gradient = objective_and_gradient(x)
        updates, optimizer_state = optimizer.update(jnp.asarray(gradient), optimizer_state, x)
        x = jnp.clip(optax.apply_updates(x, updates), -X_LIMIT, X_LIMIT)
        record(x)
    optimizer_evaluations, optimizer_message = ADAM_STEPS, "Fixed Adam step budget completed"
else:
    raise ValueError("METHOD must be 'L-BFGS-B' or 'Adam'")
print(f"{METHOD}: {optimizer_message}; optimizer gradient evaluations={optimizer_evaluations}")

# Return the best recorded point (Adam need not decrease loss at every step),
# and independently re-solve both endpoints to check the reported improvement.
best_index = int(np.argmin(loss_history))
qs = opt.QuasisymmetryRatioResidual(QS_SURFACES, helicity_m=1, helicity_n=0)
verification_metrics, verification_loss, verification_qs, verification_equilibria = [], [], [], []
for candidate in (x_history[0], x_history[best_index]):
    equilibrium = opt.solve_equilibrium(
        problem.input_from_x(x_seed + BOUNDARY_STEP * candidate), raise_on_max_iterations=True)
    verification_equilibria.append(equilibrium)
    verification_metrics.append(metrics(equilibrium))
    verification_qs.append(float(qs.total_state(equilibrium.state, equilibrium.runtime)))
    verification_loss.append(float(physical_loss(equilibrium.state, equilibrium.runtime)))
assert verification_loss[1] < verification_loss[0]
refined_metrics, refined_loss, refined_qs = [], [], []
for candidate in (x_history[0], x_history[best_index]):
    refined_input = replace(problem.input_from_x(x_seed + BOUNDARY_STEP * candidate),
                            ns_array=np.array([NS_VERIFY]))
    equilibrium = opt.solve_equilibrium(refined_input, raise_on_max_iterations=True)
    refined_metrics.append(metrics(equilibrium))
    refined_qs.append(float(qs.total_state(equilibrium.state, equilibrium.runtime)))
    refined_loss.append(float(physical_loss(equilibrium.state, equilibrium.runtime)))
print(f"Radial refinement ns={NS_VERIFY}: initial/final loss={refined_loss}")
print(f"QA residual at ns={NS_VERIFY}: {refined_qs[0]:.6e} -> {refined_qs[1]:.6e}; "
      "the compactness loss does not preserve quasisymmetry")
provenance = dict(
    input=asdict(inp), source_input=str(INPUT_FILE), vmex_version=vj.__version__,
    jax_version=jax.__version__, devices=[str(d) for d in jax.devices()], dof_names=DOF_NAMES,
    boundary_step=BOUNDARY_STEP, max_mode=MAX_MODE, x_limit=X_LIMIT, targets=[IOTA_TARGET, ASPECT_TARGET],
    scales=[IOTA_SCALE, ASPECT_SCALE], optimizer=METHOD, optimizer_message=optimizer_message,
    lbfgsb_maxiter=LBFGSB_MAXITER, adam_steps=ADAM_STEPS, learning_rate=LEARNING_RATE,
    qs_surfaces=QS_SURFACES, optimizer_evaluations=int(optimizer_evaluations),
    fd_steps=FD_STEPS, ns_verify=NS_VERIFY)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez(
    OUTPUT_DIR / "differentiation.npz", x_history=x_history, loss_history=loss_history,
    metrics_history=metrics_history, dof_names=DOF_NAMES, step=BOUNDARY_STEP,
    targets=[IOTA_TARGET, ASPECT_TARGET], scales=[IOTA_SCALE, ASPECT_SCALE],
    best_index=best_index, verification_metrics=verification_metrics, verification_loss=verification_loss,
    directional_ad=ad, directional_fd=fd_derivatives[1], fd_step=FD_STEPS[1],
    fd_steps=FD_STEPS, fd_derivatives=fd_derivatives, fd_errors=fd_errors,
    refined_metrics=refined_metrics, refined_loss=refined_loss, ns_verify=NS_VERIFY,
    verification_qs=verification_qs, refined_qs=refined_qs, method=METHOD,
    optimizer_evaluations=optimizer_evaluations, optimizer_message=optimizer_message,
    provenance=json.dumps(provenance, default=lambda a: a.tolist()),
    cold_gradient_seconds=cold_gradient_seconds, warm_gradient_seconds=warm_gradient_seconds,
)
plot_training(loss_history, metrics_history, verification_equilibria, OUTPUT_DIR / "differentiation.png")
print(f"Independent loss check: {verification_loss[0]:.6e} -> {verification_loss[1]:.6e}")
print(f"Wrote {OUTPUT_DIR / 'differentiation.npz'} and differentiation.png in {perf_counter() - started:.1f} s")

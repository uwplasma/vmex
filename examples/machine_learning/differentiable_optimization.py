#!/usr/bin/env python
"""Shape a vacuum stellarator with JAX gradients and Optax Adam.

Run from the repository root: python examples/machine_learning/differentiable_optimization.py
Requires the optional optimizer extra: pip install 'vmex[optimizers]'.
The six normalized inputs move selected boundary Fourier coefficients; VMEX
solves force balance and differentiates that converged equilibrium implicitly.
This small example targets rotational transform and aspect ratio on a coarse
grid. Its loss improvement is a learning example, not a design certificate.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import vmex as vj
from vmex import optimize as opt

INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.LandremanPaul2021_QA_lowres"
OUTPUT_DIR = Path("results/machine_learning")
DOF_NAMES = ["RBC(0,1)", "ZBS(0,1)", "RBC(1,0)", "ZBS(1,0)", "RBC(1,1)", "ZBS(1,1)"]
MAX_MODE = 1                    # raise this when adding higher-order DOF_NAMES
BOUNDARY_STEP = 0.003             # metres per unit of the dimensionless input x
X_LIMIT = 1.0                    # local box: each coefficient moves at most 3 mm
IOTA_TARGET, ASPECT_TARGET = 0.45, 6.0
IOTA_SCALE, ASPECT_SCALE = 0.05, 0.5
STEPS, LEARNING_RATE = 15, 0.15
NS, NS_VERIFY, MPOL, NTOR = 21, 31, 5, 5
FTOL, MAX_ITERATIONS = 1e-12, 3000
FD_STEPS = (0.003, 0.01, 0.03)    # dimensionless; 9--90 micrometres in coefficient space

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


def plot_training(losses, values, path):
    """Plot the training loss and physical targets in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    axes[0].semilogy(losses, "o-")
    axes[0].set(xlabel="Adam step", ylabel="Dimensionless loss")
    for index, (label, target) in enumerate(zip(("Mean rotational transform", "Aspect ratio"),
                                             (IOTA_TARGET, ASPECT_TARGET))):
        axes[index + 1].plot(np.asarray(values)[:, index], "o-")
        axes[index + 1].axhline(target, color="black", linestyle="--", linewidth=1)
        axes[index + 1].set(xlabel="Adam step", ylabel=label)
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

# Replace this transform with any Optax optimizer; the VMEX loss is unchanged.
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
optimizer_state = optimizer.init(x)
x_history, loss_history, metrics_history = [], [], []
for iteration in range(STEPS + 1):
    value, gradient = value_and_grad(x)
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError(f"Non-finite loss or gradient at step {iteration}; reduce the step size")
    equilibrium = problem.equilibrium_from_x(x_seed + BOUNDARY_STEP * x)
    fsq = float(equilibrium.result.fsqr + equilibrium.result.fsqz + equilibrium.result.fsql)
    if not np.isfinite(fsq) or (not equilibrium.result.converged and fsq > 100 * FTOL):
        raise RuntimeError(f"Uncertified equilibrium at step {iteration}; FSQ={fsq:.3e}")
    x_history.append(np.asarray(x))
    loss_history.append(float(value))
    metrics_history.append(metrics(equilibrium))
    print(f"step {iteration:2d}: loss={float(value):.6e}, "
          f"mean iota={metrics_history[-1][0]:.6f}, aspect={metrics_history[-1][1]:.6f}")
    if iteration < STEPS:
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, x)
        x = jnp.clip(optax.apply_updates(x, updates), -X_LIMIT, X_LIMIT)

# Adam need not decrease the loss at every step. Return the best evaluated point,
# and independently re-solve both endpoints to check the reported improvement.
best_index = int(np.argmin(loss_history))
verification_metrics, verification_loss = [], []
for candidate in (x_history[0], x_history[best_index]):
    equilibrium = opt.solve_equilibrium(
        problem.input_from_x(x_seed + BOUNDARY_STEP * candidate), raise_on_max_iterations=True)
    verification_metrics.append(metrics(equilibrium))
    verification_loss.append(float(physical_loss(equilibrium.state, equilibrium.runtime)))
assert verification_loss[1] < verification_loss[0]
refined_metrics, refined_loss = [], []
for candidate in (x_history[0], x_history[best_index]):
    refined_input = replace(problem.input_from_x(x_seed + BOUNDARY_STEP * candidate),
                            ns_array=np.array([NS_VERIFY]))
    equilibrium = opt.solve_equilibrium(refined_input, raise_on_max_iterations=True)
    refined_metrics.append(metrics(equilibrium))
    refined_loss.append(float(physical_loss(equilibrium.state, equilibrium.runtime)))
print(f"Radial refinement ns={NS_VERIFY}: initial/final loss={refined_loss}")
provenance = dict(
    input=asdict(inp), source_input=str(INPUT_FILE), vmex_version=vj.__version__,
    jax_version=jax.__version__, devices=[str(d) for d in jax.devices()], dof_names=DOF_NAMES,
    boundary_step=BOUNDARY_STEP, max_mode=MAX_MODE, x_limit=X_LIMIT, targets=[IOTA_TARGET, ASPECT_TARGET],
    scales=[IOTA_SCALE, ASPECT_SCALE], optimizer="Optax Adam, global gradient clip 1",
    optax_version=optax.__version__, learning_rate=LEARNING_RATE, steps=STEPS,
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
    provenance=json.dumps(provenance, default=lambda a: a.tolist()),
    cold_gradient_seconds=cold_gradient_seconds, warm_gradient_seconds=warm_gradient_seconds,
)
plot_training(loss_history, metrics_history, OUTPUT_DIR / "differentiation.png")
print(f"Independent loss check: {verification_loss[0]:.6e} -> {verification_loss[1]:.6e}")
print(f"Wrote {OUTPUT_DIR / 'differentiation.npz'} and differentiation.png in {perf_counter() - started:.1f} s")

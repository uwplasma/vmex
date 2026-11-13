"""Gradient-based design: differentiate through the equilibrium solve with JAX.

Six boundary coefficients of a vacuum QA seed are driven toward mean iota 0.45 and
aspect ratio 5. `jax.value_and_grad` differentiates the converged equilibrium
implicitly (one adjoint solve, no stored iterations) and feeds SciPy's L-BFGS-B.
The gradient is first checked against central differences of independent solves,
and both endpoints are re-solved on a finer radial grid.

A compactness objective does not optimize confinement: the QA residual is
reported alongside the loss and typically gets worse.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.plotting import surface_rz

INPUT_FILE = Path(__file__).resolve().parents[1] / "data/input.LandremanPaul2021_QA_lowres"
OUTPUT_DIR = Path("results/machine_learning")
DOF_NAMES = ["RBC(0,1)", "ZBS(0,1)", "RBC(1,0)", "ZBS(1,0)", "RBC(1,1)", "ZBS(1,1)"]
MAX_MODE = 1                    # boundary modes |m|,|n| <= MAX_MODE are packed
STEP = 0.05                     # coefficient = seed + STEP * x, with x in [-1, 1]
NS, NS_FINE, MPOL, NTOR = 21, 31, 5, 5
MAXITER = 20
FD_STEP = 1e-3                  # in units of x
SURFACES = (0.25, 0.5, 0.75)    # normalized toroidal flux for the QA residual
IOTA_TARGET, IOTA_SCALE = 0.45, 0.05
ASPECT_TARGET, ASPECT_SCALE = 5.0, 0.5

if os.environ.get("VMEX_EXAMPLES_CI") == "1":     # smoke pass: coarser, shorter
    NS_FINE, MPOL, NTOR, MAXITER = 25, 3, 3, 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
source = vj.VmecInput.from_file(INPUT_FILE)
seed = replace(
    source, lfreeb=False, pres_scale=0.0, curtor=0.0, ncurr=1, ac=np.zeros_like(source.ac),
    ns_array=np.array([NS]), ftol_array=np.array([1e-12]), niter_array=np.array([5000]), delt=0.5,
).change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=2 * NTOR + 4)
qa = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)


def physical_loss(state, runtime):
    """Dimensionless distance to the iota and aspect-ratio targets."""
    return 0.5 * (((opt.mean_iota(state, runtime) - IOTA_TARGET) / IOTA_SCALE) ** 2
                  + ((opt.aspect_ratio(state, runtime) - ASPECT_TARGET) / ASPECT_SCALE) ** 2)


problem = opt.VmecProblem.from_loss(
    seed, physical_loss, max_mode=MAX_MODE, vary_major_radius=False, use_ess=False,
    progress=True).subproblem(active=DOF_NAMES)
x_seed = jnp.asarray(problem.x0)


def loss(x):
    """x -> boundary -> converged equilibrium -> scalar loss."""
    return problem.jax_fun(x_seed + STEP * x)


def solve(x, ns=NS):
    """Independent equilibrium solve at x on an ns-point radial grid."""
    inp = replace(problem.input_from_x(x_seed + STEP * np.asarray(x)), ns_array=np.array([ns]))
    return opt.solve_equilibrium(inp, raise_on_max_iterations=True)


# Directional derivative along the gradient: adjoint versus two independent solves.
x0 = jnp.zeros_like(x_seed)
value_and_grad = jax.value_and_grad(loss)
value, gradient = value_and_grad(x0)
direction = np.asarray(gradient) / np.linalg.norm(gradient)
ad = float(np.asarray(gradient) @ direction)
plus, minus = (float(physical_loss(eq.state, eq.runtime))
               for eq in (solve(s * FD_STEP * direction) for s in (+1, -1)))
fd = (plus - minus) / (2 * FD_STEP)
print(f"Directional derivative: adjoint {ad:.6e}, central difference {fd:.6e}")
np.testing.assert_allclose(fd, ad, rtol=1e-3)

history = [np.asarray(x0)]
result = minimize(lambda x: tuple(np.asarray(v, float) for v in value_and_grad(jnp.asarray(x))),
                  history[0], jac=True, method="L-BFGS-B", bounds=[(-1, 1)] * len(x0),
                  callback=lambda x: history.append(np.copy(x)),
                  options=dict(maxiter=MAXITER, maxls=12, ftol=1e-10, gtol=1e-6))
print(f"L-BFGS-B: {result.message}; {result.nfev} gradient evaluations")
x_final = result.x

# Independent re-solves of both endpoints on the design grid and a finer one.
for ns in (NS, NS_FINE):
    before, after = [], []
    for label, x, record in (("seed", history[0], before), ("final", x_final, after)):
        eq = solve(x, ns)
        record.append(float(physical_loss(eq.state, eq.runtime)))
        print(f"ns={ns} {label}: loss {record[0]:.4e}, "
              f"iota {float(opt.mean_iota(eq.state, eq.runtime)):.4f}, "
              f"aspect {float(opt.aspect_ratio(eq.state, eq.runtime)):.4f}, "
              f"QA residual {float(qa.total_state(eq.state, eq.runtime)):.4e}")
    assert after[0] < before[0], f"loss did not decrease at ns={ns}"

losses = [float(loss(jnp.asarray(x))) for x in history]
np.savez(OUTPUT_DIR / "differentiation.npz", x_history=history, loss_history=losses,
         dof_names=DOF_NAMES, directional_ad=ad, directional_fd=fd)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
axes[0].semilogy(losses, "o-")
axes[0].set(xlabel="L-BFGS-B iteration", ylabel="Loss", title="Target loss")
for ax, fraction in zip(axes[1:], (0, 0.5)):
    for x, style, label in ((history[0], "-", "Seed"), (x_final, "--", "Final")):
        wout = solve(x).wout
        r, z = surface_rz(wout, s_index=wout.ns - 1, theta=np.linspace(0, 2 * np.pi, 200),
                          phi=np.array([fraction * 2 * np.pi / wout.nfp]))
        ax.plot(r[:, 0], z[:, 0], style, label=label)
    ax.set(xlabel="R [m]", ylabel="Z [m]", aspect="equal", title=f"Boundary at {fraction:g} field periods")
    ax.legend(frameon=False)
fig.savefig(OUTPUT_DIR / "differentiation.png", dpi=150)
print(f"Wrote {OUTPUT_DIR.resolve()}")

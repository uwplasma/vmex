"""Fixed-boundary mirror equilibrium and exact implicit gradients."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vmec_jax.core.plotting import plot_mout  # noqa: E402
from vmec_jax.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    fixed_boundary_adjoint,
    fixed_boundary_parameters,
    mout_from_result,
    project_fixed_boundary_state,
    solve_fixed_boundary_cli,
    write_mout,
)

# Inputs: edit these values, then run this file directly.
NS, NXI = 3, 5
FTOL = 1.0e-12
MAX_ITERATIONS = 500
AXIAL_FLUX_DERIVATIVE = 0.1
FINITE_DIFFERENCE_STEP = 1.0e-4
OUTPUT_DIR = Path("results/mirror_fixed_boundary_gradients")

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config = MirrorConfig(
    resolution=MirrorResolution(ns=NS, nxi=NXI),
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
)
grid = config.build_grid()
xi, s = jnp.asarray(grid.xi), jnp.asarray(grid.s)
boundary = MirrorBoundary.from_radius(0.3 * (1.0 + 0.12 * (1.0 - xi**2)), grid)
mass_profile = 2.0e3 * (1.0 - s)
current_derivative = 3.0e-2 * s

result = solve_fixed_boundary_cli(
    MirrorState.from_boundary(boundary, grid),
    boundary,
    grid,
    config,
    axial_flux_derivative=AXIAL_FLUX_DERIVATIVE,
    mass_profile=mass_profile,
    current_derivative=current_derivative,
    solve_lambda=True,
    require_convergence=True,
)
parameters = fixed_boundary_parameters(
    boundary,
    axial_flux_derivative=AXIAL_FLUX_DERIVATIVE,
    mass_profile=mass_profile,
    current_derivative=current_derivative,
)


def quantity(state, _energy):
    """Interior midplane radius used for the gradient demonstration."""

    return state.radius_scale[1, 0, grid.nxi // 2]


adjoint = fixed_boundary_adjoint(
    result,
    parameters,
    grid,
    quantity,
    solve_lambda=True,
    rtol=1.0e-10,
)
directions = fixed_boundary_parameters(
    MirrorBoundary(0.002 * (1.0 - xi**2)[None, :]),
    axial_flux_derivative=0.3,
    mass_profile=5.0e4 * (1.0 - s),
    current_derivative=0.1 * s,
)
contributions = np.asarray(
    [
        jnp.vdot(adjoint.gradient.boundary_radius, directions.boundary_radius),
        adjoint.gradient.axial_flux_derivative * directions.axial_flux_derivative,
        jnp.vdot(adjoint.gradient.mass_profile, directions.mass_profile),
        jnp.vdot(adjoint.gradient.current_derivative, directions.current_derivative),
    ],
    dtype=float,
)
finite_difference_values = []
for sign in (-1.0, 1.0):
    varied_boundary = MirrorBoundary(
        boundary.radius_scale
        + sign * FINITE_DIFFERENCE_STEP * directions.boundary_radius
    )
    varied = solve_fixed_boundary_cli(
        project_fixed_boundary_state(result.state, varied_boundary, grid),
        varied_boundary,
        grid,
        config,
        axial_flux_derivative=(
            AXIAL_FLUX_DERIVATIVE
            + sign * FINITE_DIFFERENCE_STEP * directions.axial_flux_derivative
        ),
        mass_profile=mass_profile + sign * FINITE_DIFFERENCE_STEP * directions.mass_profile,
        current_derivative=(
            current_derivative
            + sign * FINITE_DIFFERENCE_STEP * directions.current_derivative
        ),
        solve_lambda=True,
        require_convergence=True,
    )
    finite_difference_values.append(float(quantity(varied.state, varied.energy)))
finite_difference = np.diff(finite_difference_values)[0] / (2.0 * FINITE_DIFFERENCE_STEP)
predicted = float(np.sum(contributions))
relative_error = abs(predicted - finite_difference) / abs(finite_difference)

mout_path = write_mout(
    OUTPUT_DIR / "mout_mirror_fixed_gradient.nc",
    mout_from_result(
        result,
        grid,
        config,
        boundary=boundary,
        axial_flux_derivative=AXIAL_FLUX_DERIVATIVE,
        current_derivative=current_derivative,
        closure="isotropic",
    ),
)
plot_mout(mout_path, OUTPUT_DIR, name="mirror_fixed_gradient")

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
axes[0].plot(grid.z, np.asarray(adjoint.gradient.boundary_radius[0]), "o-", color="#0072B2")
axes[0].set(title="Boundary shape gradient", xlabel="Axial position z [m]", ylabel="dq/da")
labels = ["boundary", "flux", "pressure", "current"]
axes[1].bar(labels, contributions, color=["#0072B2", "#009E73", "#CC79A7", "#D55E00"])
axes[1].axhline(0.0, color="0.25", lw=0.8)
axes[1].set(title="Directional contributions", ylabel="dq/dalpha")
history = np.asarray(result.history)
axes[2].semilogy(history[:, 0], np.maximum(history[:, 4], 1.0e-18), "o-", color="#0072B2")
axes[2].axhline(FTOL, color="0.25", ls="--", label="ftol")
axes[2].set(title="Primal convergence", xlabel="Iteration", ylabel="Maximum residual")
axes[2].legend()
for axis in axes:
    axis.grid(alpha=0.22)
fig.savefig(OUTPUT_DIR / "mirror_fixed_gradient_sensitivity.png", dpi=140)
plt.close(fig)

summary = {
    "quantity": float(adjoint.value),
    "adjoint_iterations": adjoint.iterations,
    "adjoint_relative_residual": adjoint.relative_residual,
    "predicted_directional_derivative": predicted,
    "finite_difference_directional_derivative": float(finite_difference),
    "relative_error": float(relative_error),
}
(OUTPUT_DIR / "mirror_fixed_gradient_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))

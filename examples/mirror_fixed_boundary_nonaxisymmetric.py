"""Native-spline nonaxisymmetric fixed-boundary mirror equilibria."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    mout_from_result,
    plot_mout,
    write_mout,
)
from vmec_jax.mirror.analytic import (  # noqa: E402
    RotatingEllipseParaxial,
    StraightFieldLineMirror,
)
from vmec_jax.mirror.forces import staggered_field_strength  # noqa: E402
from vmec_jax.mirror.geometry import magnetic_field_xyz  # noqa: E402
from vmec_jax.mirror.splines import (  # noqa: E402
    SplineMirrorDiscretization,
    solve_spline_fixed_boundary_cli,
)

# Inputs: edit these constants, then run this file directly.
CASES = ("rotating_ellipse", "straight_field_line")
NS, MPOL, NTHETA, SOURCE_NXI = 5, 2, 12, 17
SPLINE_ELEMENTS = 4
SHAPE_STAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
FTOL = 1.0e-12
MAX_ITERATIONS = 1000
OUTPUT_DIR = Path("results/mirror_fixed_boundary_nonaxisymmetric")

RADIUS = {"rotating_ellipse": 0.05, "straight_field_line": 0.03}
AXIAL_FLUX_DERIVATIVE = {"rotating_ellipse": 0.01, "straight_field_line": 0.005}

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config = MirrorConfig(
    resolution=MirrorResolution(ns=NS, mpol=MPOL, ntheta=NTHETA, nxi=SOURCE_NXI),
    z_min=-1.0,
    z_max=1.0,
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
)
source_grid = config.build_grid()
discretization = SplineMirrorDiscretization.build(config, elements=SPLINE_ELEMENTS)
theta = jnp.asarray(source_grid.theta)[:, None]
z = jnp.asarray(source_grid.z)[None, :]


def boundary_for(case: str, stage: float) -> MirrorBoundary:
    """Return one analytic continuation boundary on the source grid."""

    radius = RADIUS[case]
    if case == "rotating_ellipse":
        fixture = RotatingEllipseParaxial(
            half_length=1.0,
            mirror_strength=0.2 * stage,
            elongation=1.0 + 0.5 * stage,
            rotation=0.5 * jnp.pi * stage,
        )
        values = fixture.boundary_radius(radius, theta, z)
    elif case == "straight_field_line":
        fixture = StraightFieldLineMirror(center_field=1.0, axial_scale=2.5)
        values = fixture.boundary_radius(radius, theta, stage * z)
    else:
        raise ValueError(f"unknown mirror case {case!r}")
    return MirrorBoundary.from_radius(values, source_grid)


def plot_validation(case: str, result, coefficient_state, boundary_state, path: Path) -> dict[str, float]:
    """Plot the independent symmetry or field-direction check."""

    grid = discretization.grid
    boundary = discretization.evaluate_boundary(boundary_state).radius_scale
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    theta_dense = np.linspace(0.0, 2.0 * np.pi, 257)
    poloidal_modes = np.fft.fftfreq(grid.ntheta, d=1.0 / grid.ntheta)
    for axial_index, color in zip(
        (0, int(np.argmin(np.abs(grid.z))), grid.nxi - 1),
        ("#0072B2", "#009E73", "#D55E00"),
        strict=True,
    ):
        polar_samples = np.asarray(boundary[:, axial_index])
        polar = np.real(
            np.exp(1j * theta_dense[:, None] * poloidal_modes[None, :])
            @ (np.fft.fft(polar_samples) / grid.ntheta)
        )
        axes[0].plot(
            polar * np.cos(theta_dense),
            polar * np.sin(theta_dense),
            color=color,
            label=f"z={grid.z[axial_index]:.2f}",
        )
    axes[0].set(title="Solved LCFS sections", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    axes[0].legend()

    history = np.asarray(result.history)
    axes[1].semilogy(history[:, 0], np.maximum(history[:, 4], 1.0e-18), "o-", color="#0072B2")
    axes[1].axhline(FTOL, color="0.25", ls="--", label="ftol")
    axes[1].set(title="Final-stage convergence", xlabel="Iteration", ylabel="Maximum residual")
    axes[1].legend()

    diagnostics: dict[str, float] = {}
    if case == "rotating_ellipse":
        mod_b = np.asarray(
            staggered_field_strength(
                result.state,
                grid,
                axial_flux_derivative=AXIAL_FLUX_DERIVATIVE[case],
            )
        )
        center = int(np.argmin(np.abs(grid.z)))
        modes = np.fft.rfft(mod_b[:, :, center], axis=1) / grid.ntheta
        m1 = 2.0 * np.abs(modes[:, 1])
        m2 = 2.0 * np.abs(modes[:, 2])
        axes[2].plot(grid.s, m1, "o-", color="#D55E00", label="m=1")
        axes[2].plot(grid.s, m2, "s-", color="#0072B2", label="m=2")
        axes[2].set(title="Midplane |B| modes", xlabel="Normalized flux s", ylabel="Amplitude [T]")
        axes[2].legend()
        diagnostics["forbidden_m1_max"] = float(np.max(m1))
    else:
        fixture = StraightFieldLineMirror(center_field=1.0, axial_scale=2.5)
        solved = np.asarray(magnetic_field_xyz(result.energy.field, result.energy.geometry))
        points = np.asarray(result.energy.geometry.xyz)
        cosine_by_z = []
        for axial_index in range(grid.nxi):
            analytic = np.asarray(jax.vmap(fixture.field)(jnp.asarray(points[1:, :, axial_index].reshape(-1, 3))))
            sampled = solved[1:, :, axial_index].reshape(-1, 3)
            cosine = np.sum(sampled * analytic, axis=1)
            cosine /= np.linalg.norm(sampled, axis=1) * np.linalg.norm(analytic, axis=1)
            cosine_by_z.append(np.mean(cosine))
        axes[2].plot(grid.z, cosine_by_z, "o-", color="#009E73")
        lower = min(cosine_by_z)
        axes[2].set_ylim(lower - 0.08 * (1.0 - lower), 1.00005)
        axes[2].set(title="Analytic field direction", xlabel="Axial position z [m]", ylabel="Mean cosine")
        diagnostics["minimum_mean_direction_cosine"] = float(np.min(cosine_by_z))

    for axis in axes:
        axis.grid(alpha=0.22)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    diagnostics["lambda_max"] = float(jnp.max(jnp.abs(coefficient_state.lambda_coefficients)))
    return diagnostics


summaries = {}
for case in CASES:
    initial_boundary = boundary_for(case, 0.0)
    previous_boundary = discretization.fit_boundary(initial_boundary, source_grid)
    coefficient_state = discretization.fit_state(MirrorState.from_boundary(initial_boundary, source_grid), source_grid)
    stage_iterations = []
    for stage in SHAPE_STAGES:
        final_boundary = discretization.fit_boundary(boundary_for(case, stage), source_grid)
        coefficient_state = discretization.transfer_boundary(coefficient_state, previous_boundary, final_boundary)
        spline_result = solve_spline_fixed_boundary_cli(
            coefficient_state,
            final_boundary,
            discretization,
            config,
            axial_flux_derivative=AXIAL_FLUX_DERIVATIVE[case],
            solve_lambda=True,
            gradient_tolerance=FTOL,
            require_convergence=True,
        )
        coefficient_state = spline_result.coefficient_state
        previous_boundary = final_boundary
        result = spline_result.evaluated
        stage_iterations.append(result.iterations)

    evaluated_boundary = discretization.evaluate_boundary(final_boundary)
    mout_path = write_mout(
        OUTPUT_DIR / f"mout_{case}.nc",
        mout_from_result(
            result,
            discretization.grid,
            config,
            boundary=evaluated_boundary,
            axial_flux_derivative=AXIAL_FLUX_DERIVATIVE[case],
            closure="isotropic",
        ),
    )
    plot_mout(mout_path, OUTPUT_DIR, name=case)
    validation = plot_validation(
        case,
        result,
        coefficient_state,
        final_boundary,
        OUTPUT_DIR / f"{case}_validation.png",
    )
    summaries[case] = {
        "stage_iterations": stage_iterations,
        "variational_max": float(result.variational.maximum),
        "staggered_weak_max": float(result.staggered_weak_force.maximum),
        "normalized_divergence_rms": float(result.normalized_divergence_rms),
        **validation,
    }

(OUTPUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
print(json.dumps(summaries, indent=2))

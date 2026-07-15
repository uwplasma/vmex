#!/usr/bin/env python3
"""Benchmark genuine-3D exterior mirror equilibria at beta 0 and 50%."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from vmec_jax.mirror import (
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    SplineMirrorDiscretization,
    solve_beta_scan_cli,
)
from vmec_jax.mirror.output import (
    boundary_fourier_amplitudes,
    summarize_axisymmetric_beta_scan,
    summarize_nonaxisymmetric_beta_scan,
)

jax.config.update("jax_enable_x64", True)


def _two_coil_dofs(*, axisymmetric: bool) -> np.ndarray:
    """Return two circular ESSOS coils, offset only for the 3-D fixture."""

    dofs = np.zeros((2, 3, 3))
    dofs[:, 0, 2] = 0.9
    dofs[:, 1, 1] = 0.9
    dofs[:, 2, 0] = [-1.0, 1.0]
    if not axisymmetric:
        dofs[:, 0, 0] = [0.04, -0.04]
    return dofs


def _axisymmetric_field_preflight(external_field, z: np.ndarray) -> dict[str, float]:
    """Verify exact on-axis and azimuthal symmetry before an axisymmetric solve."""

    z = np.asarray(z, dtype=float)
    axis_points = np.stack((np.zeros_like(z), np.zeros_like(z), z), axis=-1)
    axis_field = np.asarray(external_field(jnp.asarray(axis_points)))
    analytic = sum(
        4.0e-7 * np.pi * 2.0e5 * 0.9**2 / (2.0 * (0.9**2 + (z - position) ** 2) ** 1.5) for position in (-1.0, 1.0)
    )
    axis_relative_error = float(np.max(np.abs(axis_field[:, 2] - analytic) / analytic))
    axis_transverse_relative = float(np.max(np.linalg.norm(axis_field[:, :2], axis=1) / analytic))

    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = np.asarray(
        [
            (radius * np.cos(angle), radius * np.sin(angle), axial)
            for axial in (-0.4, 0.0, 0.4)
            for radius in (0.05, 0.15)
            for angle in theta
        ]
    )
    field = np.asarray(external_field(jnp.asarray(points))).reshape(3, 2, 8, 3)
    radial = np.stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)), axis=-1)
    toroidal = np.stack((-np.sin(theta), np.cos(theta), np.zeros_like(theta)), axis=-1)
    field_radial = np.einsum("...tc,tc->...t", field, radial)
    field_toroidal = np.einsum("...tc,tc->...t", field, toroidal)
    field_axial = field[..., 2]
    scale = float(np.max(np.linalg.norm(field, axis=-1)))
    azimuthal_relative_error = float(
        max(
            np.max(np.ptp(field_radial, axis=-1)),
            np.max(np.ptp(field_axial, axis=-1)),
            np.max(np.abs(field_toroidal)),
        )
        / scale
    )
    metrics = {
        "axis_relative_error": axis_relative_error,
        "axis_transverse_relative": axis_transverse_relative,
        "azimuthal_relative_error": azimuthal_relative_error,
    }
    if max(metrics.values()) >= 1.0e-10:
        raise ValueError(f"axisymmetric ESSOS field preflight failed: {metrics}")
    return metrics


def _axisymmetric_flux_surface_radii(
    external_field,
    z: np.ndarray,
    axial_flux_derivative: float,
    initial_radii: np.ndarray,
    *,
    order: int = 16,
) -> tuple[np.ndarray, dict[str, float]]:
    """Trace one axisymmetric flux surface from a supplied field callable."""

    nodes, weights = np.polynomial.legendre.leggauss(order)
    radial_nodes = 0.5 * (nodes + 1.0)
    radial_weights = 0.5 * weights

    def enclosed_flux(radius: float, axial: float) -> float:
        points = np.column_stack((radius * radial_nodes, np.zeros(order), np.full(order, axial)))
        bz = np.asarray(external_field(jnp.asarray(points)))[:, 2]
        return float(radius**2 * np.sum(radial_weights * radial_nodes * bz))

    radii = []
    for axial, initial in zip(np.asarray(z), np.asarray(initial_radii), strict=True):
        upper = 2.0 * float(initial)
        while enclosed_flux(upper, float(axial)) < axial_flux_derivative:
            upper *= 2.0
            if upper > 16.0 * float(initial):
                raise ValueError("failed to bracket the supplied-field flux surface")
        radii.append(
            brentq(
                lambda radius: enclosed_flux(radius, float(axial)) - axial_flux_derivative,
                0.0,
                upper,
                xtol=1.0e-13,
                rtol=1.0e-13,
            )
        )
    radii = np.asarray(radii)
    flux_error = max(
        abs(enclosed_flux(radius, axial) / axial_flux_derivative - 1.0)
        for radius, axial in zip(radii, np.asarray(z), strict=True)
    )
    return radii, {
        "quadrature_order": order,
        "maximum_relative_flux_error": float(flux_error),
        "maximum_paraxial_radius_correction": float(np.max(np.abs(radii / np.asarray(initial_radii) - 1.0))),
    }


def run(
    ns: int,
    ntheta: int,
    nxi: int,
    *,
    spline_elements: int,
    axisymmetric: bool,
    center_radius: float | None = None,
    exterior_order: int = 6,
    beta_values: tuple[float, ...] = (0.0, 0.50),
) -> dict:
    """Run one beta sequence and return machine-readable diagnostics."""

    if not axisymmetric and ntheta % 2 == 0:
        raise ValueError("nonaxisymmetric mirror collocation requires odd ntheta")
    if center_radius is None:
        center_radius = 0.25 if axisymmetric else 0.2
    if center_radius <= 0.0:
        raise ValueError("center_radius must be positive")
    if exterior_order < 2:
        raise ValueError("exterior_order must be at least two")

    config = MirrorConfig(
        resolution=MirrorResolution(
            ns=ns,
            mpol=0 if axisymmetric else (ntheta - 1) // 2,
            nxi=nxi,
        ),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    source_grid = config.build_grid()
    discretization = SplineMirrorDiscretization.build_cgl(config, elements=spline_elements)
    grid = discretization.grid
    dofs = _two_coil_dofs(axisymmetric=axisymmetric)
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart

    coils = Coils(
        Curves(jnp.asarray(dofs), n_segments=128, nfp=1, stellsym=False),
        jnp.asarray([2.0e5, 2.0e5]),
    )
    biot_savart = BiotSavart(coils)

    def external_field(points):
        points = jnp.asarray(points)
        return jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)

    field_preflight = _axisymmetric_field_preflight(external_field, np.asarray(grid.z)) if axisymmetric else None
    z = jnp.asarray(grid.z)
    on_axis = sum(
        4.0e-7 * jnp.pi * 2.0e5 * 0.9**2 / (2.0 * (0.9**2 + (z - position) ** 2) ** 1.5) for position in (-1.0, 1.0)
    )
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * center_radius**2
    paraxial = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    flux_preflight = None
    if axisymmetric:
        exact_radii, flux_preflight = _axisymmetric_flux_surface_radii(
            external_field,
            np.asarray(grid.z),
            float(flux),
            np.asarray(paraxial.radius_scale[0]),
        )
        boundary = MirrorBoundary.from_radius(exact_radii, grid)
    else:
        boundary = MirrorBoundary(
            paraxial.radius_scale + 0.03 * jnp.asarray(grid.xi)[None, :] * jnp.cos(jnp.asarray(grid.theta)[:, None])
        )
    betas = jnp.asarray(beta_values)
    start = time.perf_counter()
    results = solve_beta_scan_cli(
        discretization.fit_boundary(boundary, source_grid),
        discretization,
        config,
        external_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        current_derivative=(0.0 if axisymmetric else 1.0e-3 * jnp.asarray(grid.s)),
        exterior_ntheta=ntheta,
        exterior_order=exterior_order,
        exterior_spectral_side_density=True,
    )
    wall = time.perf_counter() - start
    summarize = summarize_axisymmetric_beta_scan if axisymmetric else summarize_nonaxisymmetric_beta_scan
    diagnostics = summarize(results, betas, grid, reference_field=float(on_axis[center]))
    rows = []
    for beta, result, diagnostic in zip(betas, results, diagnostics, strict=True):
        axial_weights = np.asarray(grid.axial_basis.weights).copy()
        axial_weights[[0, -1]] = 0.0
        lateral_weights = np.asarray(grid.theta_basis.weights)[:, None] * axial_weights[None, :]
        vacuum_b_normal = np.asarray(result.vacuum_field.lateral_b_normal)
        if vacuum_b_normal.ndim == 1:
            vacuum_b_normal = vacuum_b_normal[None, :]
        stress_jump = np.asarray(result.interface.normal_stress_jump)

        def lateral_rms(values: np.ndarray) -> float:
            return float(np.sqrt(np.sum(lateral_weights * values**2) / np.sum(lateral_weights)))

        row = {
            "beta": float(beta),
            "converged": result.converged,
            "iterations": result.iterations,
            "residual": float(result.variational_max),
            "normal_stress_rms": float(result.interface.normal_stress_rms),
            "vacuum_b_normal_rms": float(result.interface.vacuum_b_normal_rms),
            "staggered_weak_max": float(result.plasma_staggered_weak_force.maximum),
            "pointwise_force_rms": float(result.plasma_force.normalized_rms),
            "pointwise_force_axis_rms": float(result.plasma_force.axis_normalized_rms),
            "pointwise_force_first_row_rms": float(result.plasma_force.first_row_normalized_rms),
            "pointwise_force_bulk_rms": float(result.plasma_force.bulk_normalized_rms),
            "pointwise_force_end_collar_rms": float(result.plasma_force.end_collar_normalized_rms),
            "axis_field_nonuniformity": float(result.plasma_force.axis_field_nonuniformity),
            "normalized_divb": float(result.normalized_divergence_rms),
            "lambda_max": float(jnp.max(jnp.abs(result.plasma_state.lambda_stream))),
            "compatibility": float(result.vacuum_field.neumann_result.compatibility_error),
            "raw_compatibility": float(result.vacuum_field.neumann_result.raw_compatibility_error),
            "condition_number": float(result.vacuum_field.neumann_result.condition_number),
            "vacuum_b_normal_T_rms": lateral_rms(vacuum_b_normal),
            "normal_stress_jump_Pa_rms": lateral_rms(stress_jump),
            "achieved_beta": float(diagnostic.achieved_reference_beta),
            "volume_beta": float(diagnostic.volume_averaged_beta),
        }
        if axisymmetric:
            axis_field = np.sqrt(np.asarray(result.plasma_b_squared)[0, 0])
            row.update(
                center_radius_m=float(diagnostic.center_radius),
                center_field_T=float(diagnostic.center_axis_field),
                field_ratio=float(diagnostic.diamagnetic_field_ratio),
                mirror_ratio=float(np.max(axis_field) / axis_field[center]),
            )
        else:
            row.update(
                mean_center_radius_m=float(diagnostic.center_mean_radius),
                mean_center_field_T=float(diagnostic.center_mean_field),
                m1_radius_m=float(diagnostic.center_boundary_modes[1]),
                m1_axial_l2_m=float(diagnostic.boundary_mode_l2[1]),
                m1_axial_max_m=float(diagnostic.boundary_mode_max[1]),
                m1_core_l2_m=float(diagnostic.boundary_mode_core_l2[1]),
                m1_core_max_m=float(diagnostic.boundary_mode_core_max[1]),
                m1_midplane_leakage_fraction=float(
                    diagnostic.center_boundary_modes[1] / diagnostic.boundary_mode_max[1]
                ),
                m1_axial_profile_m=np.asarray(boundary_fourier_amplitudes(result.boundary)[1]).tolist(),
                volume_m3=float(diagnostic.plasma_volume),
                energy_J=float(diagnostic.plasma_energy),
            )
        rows.append(row)
    return {
        "grid": {"ns": ns, "ntheta": ntheta, "nxi": nxi},
        "axisymmetric": axisymmetric,
        "field_preflight": field_preflight,
        "flux_surface_preflight": flux_preflight,
        "initial_center_radius_m": center_radius,
        "exterior_order": exterior_order,
        "spline_elements": spline_elements,
        "device": str(jax.devices()[0]),
        "wall_s_pair": wall,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024.0**2 if sys.platform == "darwin" else 1024.0),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, required=True)
    parser.add_argument("--ntheta", type=int, required=True)
    parser.add_argument("--nxi", type=int, required=True)
    parser.add_argument("--spline-elements", type=int, required=True)
    parser.add_argument("--axisymmetric", action="store_true")
    parser.add_argument("--center-radius", type=float)
    parser.add_argument("--exterior-order", type=int, default=6)
    beta_group = parser.add_mutually_exclusive_group()
    beta_group.add_argument("--beta-zero-only", action="store_true")
    beta_group.add_argument("--beta-values", nargs="+", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    beta_values = (
        tuple(args.beta_values) if args.beta_values is not None else ((0.0,) if args.beta_zero_only else (0.0, 0.50))
    )
    result = run(
        args.ns,
        args.ntheta,
        args.nxi,
        spline_elements=args.spline_elements,
        axisymmetric=args.axisymmetric,
        center_radius=args.center_radius,
        exterior_order=args.exterior_order,
        beta_values=beta_values,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

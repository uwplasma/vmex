#!/usr/bin/env python3
"""Benchmark genuine-3D exterior mirror equilibria at beta 0 and 50%."""

from __future__ import annotations

import argparse
import json
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from vmec_jax.mirror import (
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    build_vacuum_grid,
    solve_beta_scan_cli,
)
from vmec_jax.mirror.output import (
    boundary_fourier_amplitudes,
    summarize_axisymmetric_beta_scan,
    summarize_nonaxisymmetric_beta_scan,
)

jax.config.update("jax_enable_x64", True)


def run(
    ns: int,
    ntheta: int,
    nxi: int,
    *,
    jacobian_chunk_size: int,
    axisymmetric: bool,
    beta_values: tuple[float, ...] = (0.0, 0.50),
) -> dict:
    """Run one endpoint pair and return machine-readable diagnostics."""

    config = MirrorConfig(
        resolution=MirrorResolution(
            ns=ns,
            mpol=0 if axisymmetric else 1,
            ntheta=1 if axisymmetric else ntheta,
            nxi=nxi,
        ),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(grid, nrho=ns)
    dofs = np.zeros((2, 3, 3))
    dofs[:, 0, 2] = 0.9
    dofs[:, 1, 1] = 0.9
    dofs[:, 2, 0] = [-1.0, 1.0]
    dofs[:, 0, 0] = [0.04, -0.04]
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

    z = jnp.asarray(grid.z)
    on_axis = sum(
        4.0e-7 * jnp.pi * 2.0e5 * 0.9**2
        / (2.0 * (0.9**2 + (z - position) ** 2) ** 1.5)
        for position in (-1.0, 1.0)
    )
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.2**2
    base = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    boundary = base
    if not axisymmetric:
        boundary = MirrorBoundary(
            base.radius_scale + 0.03 * jnp.asarray(grid.xi)[None, :] * jnp.cos(jnp.asarray(grid.theta)[:, None])
        )
    betas = jnp.asarray(beta_values)
    start = time.perf_counter()
    results = solve_beta_scan_cli(
        boundary,
        grid,
        vacuum_grid,
        config,
        external_field,
        betas,
        outer_radius=0.1,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        current_derivative=(0.0 if axisymmetric else 1.0e-3 * jnp.asarray(grid.s)),
        vacuum_backend="exterior",
        exterior_ntheta=ntheta,
        exterior_order=6,
        exterior_spectral_side_density=True,
        exterior_jacobian_chunk_size=jacobian_chunk_size,
    )
    wall = time.perf_counter() - start
    summarize = summarize_axisymmetric_beta_scan if axisymmetric else summarize_nonaxisymmetric_beta_scan
    diagnostics = summarize(results, betas, grid, reference_field=float(on_axis[center]))
    rows = []
    for beta, result, diagnostic in zip(betas, results, diagnostics, strict=True):
        row = {
            "beta": float(beta),
            "converged": result.converged,
            "iterations": result.iterations,
            "residual": float(result.variational_max),
            "normal_stress_rms": float(result.interface.normal_stress_rms),
            "vacuum_b_normal_rms": float(result.interface.vacuum_b_normal_rms),
            "staggered_weak_max": float(
                result.plasma_staggered_weak_force.maximum
            ),
            "normalized_divb": float(result.normalized_divergence_rms),
            "lambda_max": float(jnp.max(jnp.abs(result.plasma_state.lambda_stream))),
            "compatibility": float(result.vacuum_field.neumann_result.compatibility_error),
            "condition_number": float(result.vacuum_field.neumann_result.condition_number),
            "achieved_beta": float(diagnostic.achieved_reference_beta),
            "volume_beta": float(diagnostic.volume_averaged_beta),
        }
        if axisymmetric:
            row.update(
                center_radius_m=float(diagnostic.center_radius),
                center_field_T=float(diagnostic.center_axis_field),
                field_ratio=float(diagnostic.diamagnetic_field_ratio),
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
                m1_axial_profile_m=np.asarray(
                    boundary_fourier_amplitudes(result.boundary)[1]
                ).tolist(),
                volume_m3=float(diagnostic.plasma_volume),
                energy_J=float(diagnostic.plasma_energy),
            )
        rows.append(row)
    return {
        "grid": {"ns": ns, "ntheta": ntheta, "nxi": nxi},
        "axisymmetric": axisymmetric,
        "jacobian_chunk_size": jacobian_chunk_size,
        "device": str(jax.devices()[0]),
        "wall_s_pair": wall,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, required=True)
    parser.add_argument("--ntheta", type=int, required=True)
    parser.add_argument("--nxi", type=int, required=True)
    parser.add_argument("--jacobian-chunk-size", type=int, default=1)
    parser.add_argument("--axisymmetric", action="store_true")
    parser.add_argument("--beta-zero-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(
        args.ns,
        args.ntheta,
        args.nxi,
        jacobian_chunk_size=args.jacobian_chunk_size,
        axisymmetric=args.axisymmetric,
        beta_values=(0.0,) if args.beta_zero_only else (0.0, 0.50),
    )
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

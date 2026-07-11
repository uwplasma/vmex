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

from vmec_jax.core.coils import CoilSet, two_coil_on_axis_bz
from vmec_jax.mirror import (
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    build_vacuum_grid,
    solve_beta_scan_cli,
    summarize_nonaxisymmetric_beta_scan,
)

jax.config.update("jax_enable_x64", True)


def run(ns: int, ntheta: int, nxi: int, *, high_order: bool) -> dict:
    """Run one endpoint pair and return machine-readable diagnostics."""

    config = MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=1, ntheta=ntheta, nxi=nxi),
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
    coils = CoilSet(
        base_curve_dofs=jnp.asarray(dofs),
        base_currents=jnp.asarray([2.0e5, 2.0e5]),
        n_segments=128,
    )
    on_axis = two_coil_on_axis_bz(
        jnp.asarray(grid.z), coil_radius=0.9, separation=2.0, current=2.0e5
    )
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.2**2
    base = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    boundary = MirrorBoundary(
        base.radius_scale
        + 0.03
        * jnp.asarray(grid.xi)[None, :]
        * jnp.cos(jnp.asarray(grid.theta)[:, None])
    )
    betas = jnp.asarray([0.0, 0.50])
    start = time.perf_counter()
    results = solve_beta_scan_cli(
        boundary,
        grid,
        vacuum_grid,
        config,
        coils,
        betas,
        outer_radius=0.1,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        current_derivative=1.0e-3 * jnp.asarray(grid.s),
        vacuum_backend="exterior",
        exterior_order=6,
        exterior_spectral_side_density=True,
        exterior_high_order_cap_panels=high_order,
        exterior_curved_side_geometry=True,
    )
    wall = time.perf_counter() - start
    diagnostics = summarize_nonaxisymmetric_beta_scan(
        results, betas, grid, reference_field=float(on_axis[center])
    )
    rows = []
    for beta, result, diagnostic in zip(betas, results, diagnostics, strict=True):
        rows.append(
            {
                "beta": float(beta),
                "converged": result.converged,
                "iterations": result.iterations,
                "residual": float(result.variational_max),
                "normal_stress_rms": float(result.interface.normal_stress_rms),
                "vacuum_b_normal_rms": float(result.interface.vacuum_b_normal_rms),
                "compatibility": float(
                    result.vacuum_field.neumann_result.compatibility_error
                ),
                "condition_number": float(
                    result.vacuum_field.neumann_result.condition_number
                ),
                "achieved_beta": float(diagnostic.achieved_reference_beta),
                "volume_beta": float(diagnostic.volume_averaged_beta),
                "mean_center_radius_m": float(diagnostic.center_mean_radius),
                "mean_center_field_T": float(diagnostic.center_mean_field),
                "m1_radius_m": float(diagnostic.center_boundary_modes[1]),
                "volume_m3": float(diagnostic.plasma_volume),
                "energy_J": float(diagnostic.plasma_energy),
            }
        )
    return {
        "grid": {"ns": ns, "ntheta": ntheta, "nxi": nxi},
        "high_order_cap_panels": high_order,
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
    parser.add_argument("--linear-cap", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run(args.ns, args.ntheta, args.nxi, high_order=not args.linear_cap)
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    circular_loop_field_rz,
    mirror_boundary_from_on_axis_bz,
    mirror_boundary_from_two_coil_flux_tube,
    on_axis_mirror_ratio,
    run_mirror_fixed_boundary,
    two_coil_field_rz,
    two_coil_on_axis_bz,
    two_coil_on_axis_mirror_ratio,
)
from vmec_jax.mirror.kernels.fields import evaluate_axisym_field
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry

pytestmark = pytest.mark.mirror


def _quadrature_circular_loop_field_rz(radius, z, *, loop_radius: float, current: float, num_points: int = 8192):
    phi = np.linspace(0.0, 2.0 * np.pi, int(num_points), endpoint=False)
    dphi = 2.0 * np.pi / int(num_points)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    radius = np.asarray(radius, dtype=float)
    z = np.asarray(z, dtype=float)
    radius, z = np.broadcast_arrays(radius, z)
    rr = radius[..., None]
    zz = z[..., None]
    distance2 = rr**2 + float(loop_radius) ** 2 - 2.0 * rr * float(loop_radius) * cos_phi + zz**2
    distance3 = distance2**1.5
    br_integrand = float(loop_radius) * zz * cos_phi / distance3
    bz_integrand = (float(loop_radius) ** 2 - float(loop_radius) * rr * cos_phi) / distance3
    scale = 4.0e-7 * np.pi * float(current) / (4.0 * np.pi)
    return scale * np.sum(br_integrand, axis=-1) * dphi, scale * np.sum(bz_integrand, axis=-1) * dphi


def test_two_coil_on_axis_formula_matches_full_loop_on_axis_branch():
    z = np.linspace(-1.0, 1.0, 17)
    radius = 0.35
    separation = 2.0
    current = 1.0e6

    analytic = two_coil_on_axis_bz(z, coil_radius_m=radius, separation_m=separation, current_a=current)
    full_loop = circular_loop_field_rz(
        np.zeros_like(z), z + 0.5 * separation, loop_radius_m=radius, current_a=current
    ).bz
    full_loop += circular_loop_field_rz(
        np.zeros_like(z), z - 0.5 * separation, loop_radius_m=radius, current_a=current
    ).bz

    assert np.allclose(analytic, full_loop, rtol=2.0e-15, atol=2.0e-15)
    assert two_coil_on_axis_mirror_ratio(coil_radius_m=radius, separation_m=separation, current_a=current) > 1.0


def test_circular_loop_off_axis_field_matches_direct_biot_savart_quadrature():
    radius = np.asarray([0.015, 0.025, 0.04])
    z = np.asarray([-0.45, 0.0, 0.35])
    loop_radius = 0.35
    current = 1.0e6

    field = circular_loop_field_rz(radius, z, loop_radius_m=loop_radius, current_a=current)
    br_quad, bz_quad = _quadrature_circular_loop_field_rz(
        radius,
        z,
        loop_radius=loop_radius,
        current=current,
    )

    assert np.allclose(field.br, br_quad, rtol=2.0e-10, atol=2.0e-10)
    assert np.allclose(field.bz, bz_quad, rtol=2.0e-10, atol=2.0e-10)


def test_two_coil_flux_tube_boundary_helper_matches_explicit_on_axis_construction():
    config = MirrorConfig(MirrorResolution(ns=7, ntheta=1, nxi=17, mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    coil_radius = 0.35
    separation = 2.0
    current = 1.0e6
    psi_value = 0.012
    bz_axis = two_coil_on_axis_bz(
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )

    explicit = mirror_boundary_from_on_axis_bz(psi_value, grid.z, bz_axis)
    helper = mirror_boundary_from_two_coil_flux_tube(
        psi_value,
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )

    assert np.allclose(helper.radius_on_grid(grid), explicit.radius_on_grid(grid), rtol=0.0, atol=0.0)


def test_two_coil_flux_tube_mirror_axis_field_matches_analytic_bz():
    coil_radius = 0.35
    separation = 2.0
    current = 1.0e6
    midplane_radius = 0.3
    config = MirrorConfig(MirrorResolution(ns=9, ntheta=1, nxi=33, mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    analytic_bz = two_coil_on_axis_bz(
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )
    psi_value = 0.5 * abs(
        float(two_coil_on_axis_bz(0.0, coil_radius_m=coil_radius, separation_m=separation, current_a=current))
    )
    psi_value *= midplane_radius**2
    boundary = mirror_boundary_from_on_axis_bz(psi_value, grid.z, analytic_bz)

    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(psi_value),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=0, tolerance=1.0e-10, mu0=1.0),
    )
    geometry = evaluate_axisym_geometry(result.state, result.grid)
    field = evaluate_axisym_field(
        result.state, result.grid, geometry, psi_prime=result.psi_prime, i_prime=result.i_prime
    )
    mirror_bz = field.b_z[0]
    low_radius_index = 1
    off_axis = two_coil_field_rz(
        geometry.r[low_radius_index],
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )

    assert np.allclose(mirror_bz, analytic_bz, rtol=3.0e-13, atol=3.0e-13)
    assert np.max(np.abs(field.b_r[low_radius_index] - off_axis.br)) < 0.002
    assert np.max(np.abs(field.b_z[low_radius_index] - off_axis.bz)) < 0.01
    assert on_axis_mirror_ratio(mirror_bz) == pytest.approx(on_axis_mirror_ratio(analytic_bz), rel=3.0e-13)
    assert result.final_trace.mirror_ratio > 1.0

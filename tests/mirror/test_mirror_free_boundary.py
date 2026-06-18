from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.mirror import (
    MirrorBoundary,
    MirrorCircularCoils,
    initial_mirror_boundary_from_circular_coil_scan,
    load_mirror_free_boundary_circular_coil_scan,
    make_mirror_free_boundary_beta_cases,
    make_mirror_free_boundary_circular_coil_scan,
    make_mirror_grid,
    mirror_boundary_from_on_axis_bz,
    mirror_circular_coils_to_direct_params,
    sample_mirror_axis_external_field,
    sample_mirror_boundary_external_field,
    two_coil_on_axis_bz,
    write_mirror_free_boundary_circular_coil_scan,
)

pytestmark = pytest.mark.mirror


def test_mirror_circular_coils_build_essos_compatible_direct_params():
    coils = MirrorCircularCoils.symmetric_pair(
        coil_radius_m=0.35,
        separation_m=1.4,
        current_a=1.0e6,
        n_segments=64,
        regularization_epsilon=1.0e-6,
        chunk_size=17,
    )

    params = mirror_circular_coils_to_direct_params(coils)
    dofs = np.asarray(params.base_curve_dofs)

    assert dofs.shape == (2, 3, 3)
    np.testing.assert_allclose(dofs[:, 0, 2], [0.35, 0.35])
    np.testing.assert_allclose(dofs[:, 1, 1], [0.35, 0.35])
    np.testing.assert_allclose(dofs[:, 2, 0], [-0.7, 0.7])
    np.testing.assert_allclose(params.base_currents, [1.0e6, 1.0e6])
    assert params.n_segments == 64
    assert params.nfp == 1
    assert params.stellsym is False
    assert params.regularization_epsilon == 1.0e-6
    assert params.chunk_size == 17


def test_mirror_axis_direct_circular_coils_match_analytic_two_coil_field():
    enable_x64(True)
    coil_radius = 0.35
    separation = 1.2
    current = 1.0e6
    grid = make_mirror_grid(ns=5, ntheta=1, nxi=17, z_min=-0.6, z_max=0.6)
    coils = MirrorCircularCoils.symmetric_pair(
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
        n_segments=256,
    )

    sample = sample_mirror_axis_external_field(grid, coils)
    expected_bz = two_coil_on_axis_bz(
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )

    np.testing.assert_allclose(np.asarray(sample.br), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(sample.btheta), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(sample.bz), expected_bz, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(sample.bmag), np.abs(expected_bz), rtol=1.0e-12, atol=1.0e-12)


def test_mirror_boundary_external_field_sampling_has_boundary_shape():
    enable_x64(True)
    grid = make_mirror_grid(ns=5, ntheta=9, nxi=11, mpol=3, z_min=-0.5, z_max=0.5)
    boundary = MirrorBoundary.cosine_modulated_radius(r0=0.1, a2=0.2, epsilon=0.05, theta_mode=2)
    coils = MirrorCircularCoils.symmetric_pair(
        coil_radius_m=0.4,
        separation_m=1.0,
        current_a=8.0e5,
        n_segments=192,
    )

    sample = sample_mirror_boundary_external_field(grid, boundary, coils)

    assert sample.br.shape == (grid.ntheta, grid.nxi)
    assert sample.btheta.shape == sample.br.shape
    assert sample.bz.shape == sample.br.shape
    assert sample.bmag.shape == sample.br.shape
    np.testing.assert_allclose(np.asarray(sample.r), boundary.radius_on_grid_3d(grid))
    assert np.all(np.asarray(sample.bmag) > 0.0)


def test_mirror_free_boundary_beta_cases_default_to_requested_scan_points():
    cases = make_mirror_free_boundary_beta_cases(pressure_scale_for_one_percent=12.5)

    assert [case.beta_percent for case in cases] == [1.0, 3.0, 10.0]
    assert [case.beta_fraction for case in cases] == [0.01, 0.03, 0.10]
    assert [case.pressure_scale for case in cases] == [12.5, 37.5, 125.0]

    with pytest.raises(ValueError, match="nonnegative"):
        make_mirror_free_boundary_beta_cases((-1.0,))


def test_mirror_free_boundary_circular_coil_scan_json_roundtrip(tmp_path):
    coils = MirrorCircularCoils.symmetric_pair(
        coil_radius_m=0.4,
        separation_m=1.6,
        current_a=9.0e5,
        n_segments=96,
        regularization_epsilon=1.0e-5,
    )
    scan = make_mirror_free_boundary_circular_coil_scan(
        coils,
        beta_percent=(1.0, 3.0, 10.0),
        pressure_scale_for_one_percent=4.0,
    )

    path = write_mirror_free_boundary_circular_coil_scan(tmp_path / "scan.json", scan)
    loaded = load_mirror_free_boundary_circular_coil_scan(path)

    np.testing.assert_allclose(loaded.coils.radii_m, coils.radii_m)
    np.testing.assert_allclose(loaded.coils.z_centers_m, coils.z_centers_m)
    np.testing.assert_allclose(loaded.coils.currents_a, coils.currents_a)
    assert loaded.coils.n_segments == 96
    assert loaded.coils.regularization_epsilon == pytest.approx(1.0e-5)
    assert [case.beta_percent for case in loaded.beta_cases] == [1.0, 3.0, 10.0]
    assert [case.pressure_scale for case in loaded.beta_cases] == [4.0, 12.0, 40.0]


def test_initial_mirror_boundary_from_circular_coil_scan_matches_analytic_flux_tube():
    enable_x64(True)
    coil_radius = 0.35
    separation = 1.2
    current = 1.0e6
    midplane_radius = 0.25
    grid = make_mirror_grid(ns=5, ntheta=1, nxi=17, z_min=-0.6, z_max=0.6)
    coils = MirrorCircularCoils.symmetric_pair(
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
        n_segments=256,
    )
    scan = make_mirror_free_boundary_circular_coil_scan(coils)

    boundary = initial_mirror_boundary_from_circular_coil_scan(
        grid,
        scan,
        midplane_radius=midplane_radius,
    )
    analytic_bz = two_coil_on_axis_bz(
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )
    midplane_bz = float(two_coil_on_axis_bz(0.0, coil_radius_m=coil_radius, separation_m=separation, current_a=current))
    expected = mirror_boundary_from_on_axis_bz(0.5 * midplane_bz * midplane_radius**2, grid.z, analytic_bz)

    np.testing.assert_allclose(boundary.radius_on_grid(grid), expected.radius_on_grid(grid), rtol=1.0e-12)

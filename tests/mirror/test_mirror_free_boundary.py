from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.mirror import (
    MirrorBoundary,
    MirrorCircularCoils,
    MirrorExternalFieldSample,
    MirrorFreeBoundaryBetaCase,
    MirrorFreeBoundaryCircularCoilScan,
    MirrorFreeBoundaryResidual,
    initial_mirror_boundary_from_circular_coil_scan,
    load_mirror_free_boundary_circular_coil_scan,
    make_mirror_free_boundary_beta_cases,
    make_mirror_free_boundary_circular_coil_scan,
    make_mirror_grid,
    mirror_boundary_from_external_axis_field,
    mirror_boundary_from_on_axis_bz,
    mirror_circular_coils_to_direct_params,
    mirror_external_bnormal,
    mirror_external_pressure_balance_response,
    mirror_free_boundary_least_squares_step,
    mirror_free_boundary_residual,
    mirror_free_boundary_residual_jacobian_finite_difference,
    mirror_lcfs_diagnostic,
    mirror_lcfs_diagnostic_from_arrays,
    mirror_lcfs_merit,
    mirror_lcfs_residual,
    propose_axisymmetric_mirror_lcfs_update,
    propose_axisymmetric_mirror_lcfs_scale_update,
    propose_axisymmetric_mirror_lcfs_noop_update,
    propose_axisymmetric_mirror_lcfs_bnormal_update,
    propose_axisymmetric_mirror_lcfs_mixed_update,
    propose_axisymmetric_mirror_lcfs_candidate_set,
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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "radii_m": [[0.3]],
                "z_centers_m": [0.0],
                "currents_a": [1.0],
            },
            "one-dimensional",
        ),
        (
            {
                "radii_m": [0.3, 0.4],
                "z_centers_m": [0.0],
                "currents_a": [1.0, 1.0],
            },
            "same shape",
        ),
        (
            {
                "radii_m": [],
                "z_centers_m": [],
                "currents_a": [],
            },
            "at least one",
        ),
        (
            {
                "radii_m": [0.0],
                "z_centers_m": [0.0],
                "currents_a": [1.0],
            },
            "positive",
        ),
        (
            {
                "radii_m": [0.3],
                "z_centers_m": [0.0],
                "currents_a": [1.0],
                "n_segments": 7,
            },
            "at least 8",
        ),
        (
            {
                "radii_m": [0.3],
                "z_centers_m": [0.0],
                "currents_a": [1.0],
                "regularization_epsilon": -1.0,
            },
            "nonnegative",
        ),
        (
            {
                "radii_m": [0.3],
                "z_centers_m": [0.0],
                "currents_a": [1.0],
                "chunk_size": 0,
            },
            "chunk_size",
        ),
    ],
)
def test_mirror_circular_coils_reject_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MirrorCircularCoils(**kwargs)


def test_mirror_circular_coils_reject_nonpositive_symmetric_pair_separation():
    with pytest.raises(ValueError, match="separation_m"):
        MirrorCircularCoils.symmetric_pair(
            coil_radius_m=0.35,
            separation_m=0.0,
            current_a=1.0e6,
        )


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
    with pytest.raises(ValueError, match="pressure_scale_for_one_percent"):
        make_mirror_free_boundary_beta_cases(pressure_scale_for_one_percent=0.0)


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
    payload = json.loads(path.read_text())

    np.testing.assert_allclose(loaded.coils.radii_m, coils.radii_m)
    np.testing.assert_allclose(loaded.coils.z_centers_m, coils.z_centers_m)
    np.testing.assert_allclose(loaded.coils.currents_a, coils.currents_a)
    assert loaded.coils.n_segments == 96
    assert loaded.coils.regularization_epsilon == pytest.approx(1.0e-5)
    assert [case.beta_percent for case in loaded.beta_cases] == [1.0, 3.0, 10.0]
    assert [case.pressure_scale for case in loaded.beta_cases] == [4.0, 12.0, 40.0]
    assert payload["status"] == "setup_only_no_lcfs_solve"


def test_mirror_free_boundary_scan_rejects_empty_cases_and_beta_case_defaults():
    coils = MirrorCircularCoils.symmetric_pair(
        coil_radius_m=0.4,
        separation_m=1.6,
        current_a=9.0e5,
    )
    case = MirrorFreeBoundaryBetaCase.from_dict({"beta_percent": 2.5, "pressure_scale": 7.0})

    assert case.beta_fraction == pytest.approx(0.025)
    with pytest.raises(ValueError, match="at least one beta case"):
        MirrorFreeBoundaryCircularCoilScan(coils=coils, beta_cases=())


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


def test_mirror_boundary_from_external_axis_field_rejects_invalid_inputs():
    grid = make_mirror_grid(ns=3, ntheta=1, nxi=5, z_min=-1.0, z_max=1.0)

    with pytest.raises(ValueError, match="axis_bz"):
        mirror_boundary_from_external_axis_field(grid, np.ones(grid.nxi + 1), midplane_radius=0.25)
    with pytest.raises(ValueError, match="midplane_radius"):
        mirror_boundary_from_external_axis_field(grid, np.ones(grid.nxi), midplane_radius=0.0)
    with pytest.raises(ValueError, match="nonzero"):
        mirror_boundary_from_external_axis_field(grid, np.zeros(grid.nxi), midplane_radius=0.25)


def test_mirror_lcfs_diagnostic_reports_side_boundary_targets():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 5)
    boundary_r = np.full((theta.size, z.size), 0.25)
    internal_bmag = np.full((2, theta.size, z.size), 2.0)
    external_bmag = np.ones_like(boundary_r)
    output = SimpleNamespace(
        theta=theta,
        z=z,
        geometry=SimpleNamespace(boundary_r=boundary_r),
        field=SimpleNamespace(bmag=internal_bmag),
        profiles=SimpleNamespace(pressure=np.asarray([0.2, 0.0])),
    )
    sample = MirrorExternalFieldSample(
        r=boundary_r,
        theta=theta,
        z=z,
        br=np.zeros_like(boundary_r),
        btheta=np.zeros_like(boundary_r),
        bz=np.ones_like(boundary_r),
        bmag=external_bmag,
    )

    diagnostic = mirror_lcfs_diagnostic(output, sample, mu0=1.0)
    diagnostic_from_arrays = mirror_lcfs_diagnostic_from_arrays(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        edge_internal_bmag=internal_bmag[-1],
        external_sample=sample,
        edge_pressure=0.0,
        mu0=1.0,
    )

    np.testing.assert_allclose(diagnostic.boundary_dr_dz, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(diagnostic.external_bnormal, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(diagnostic.pressure_balance, 1.5)
    np.testing.assert_allclose(diagnostic_from_arrays.external_bnormal, diagnostic.external_bnormal)
    np.testing.assert_allclose(diagnostic_from_arrays.pressure_balance, diagnostic.pressure_balance)
    assert diagnostic.external_bnormal_rms == pytest.approx(0.0)
    assert diagnostic.external_bnormal_max == pytest.approx(0.0)
    assert diagnostic.pressure_balance_rms == pytest.approx(1.5)
    assert diagnostic.pressure_balance_max == pytest.approx(1.5)


def test_mirror_lcfs_diagnostic_from_arrays_rejects_invalid_shapes():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 3)
    boundary_r = np.ones((theta.size, z.size))
    sample = MirrorExternalFieldSample(
        r=boundary_r,
        theta=theta,
        z=z,
        br=np.zeros_like(boundary_r),
        btheta=np.zeros_like(boundary_r),
        bz=np.ones_like(boundary_r),
        bmag=np.ones_like(boundary_r),
    )

    with pytest.raises(ValueError, match="boundary_r"):
        mirror_lcfs_diagnostic_from_arrays(
            theta=theta,
            z=z,
            boundary_r=np.ones((2, z.size)),
            edge_internal_bmag=boundary_r,
            external_sample=sample,
            edge_pressure=0.0,
        )
    with pytest.raises(ValueError, match="at least two"):
        mirror_lcfs_diagnostic_from_arrays(
            theta=theta,
            z=np.asarray([0.0]),
            boundary_r=np.ones((1, 1)),
            edge_internal_bmag=np.ones((1, 1)),
            external_sample=MirrorExternalFieldSample(
                r=np.ones((1, 1)),
                theta=theta,
                z=np.asarray([0.0]),
                br=np.zeros((1, 1)),
                btheta=np.zeros((1, 1)),
                bz=np.ones((1, 1)),
                bmag=np.ones((1, 1)),
            ),
            edge_pressure=0.0,
        )
    with pytest.raises(ValueError, match="external field sample"):
        mirror_lcfs_diagnostic_from_arrays(
            theta=theta,
            z=z,
            boundary_r=boundary_r,
            edge_internal_bmag=boundary_r,
            external_sample=MirrorExternalFieldSample(
                r=boundary_r,
                theta=theta,
                z=z,
                br=np.zeros((1, z.size + 1)),
                btheta=np.zeros_like(boundary_r),
                bz=np.ones_like(boundary_r),
                bmag=np.ones_like(boundary_r),
            ),
            edge_pressure=0.0,
        )
    with pytest.raises(ValueError, match=r"internal edge \|B\|"):
        mirror_lcfs_diagnostic_from_arrays(
            theta=theta,
            z=z,
            boundary_r=boundary_r,
            edge_internal_bmag=np.ones((1, z.size + 1)),
            external_sample=sample,
            edge_pressure=0.0,
        )
    with pytest.raises(ValueError, match="mu0"):
        mirror_lcfs_diagnostic_from_arrays(
            theta=theta,
            z=z,
            boundary_r=boundary_r,
            edge_internal_bmag=boundary_r,
            external_sample=sample,
            edge_pressure=0.0,
            mu0=0.0,
        )


def test_mirror_external_bnormal_is_zero_for_axial_field_on_cylinder():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 7)
    boundary_r = np.full((theta.size, z.size), 0.25)
    sample = MirrorExternalFieldSample(
        r=boundary_r,
        theta=theta,
        z=z,
        br=np.zeros_like(boundary_r),
        btheta=np.zeros_like(boundary_r),
        bz=np.ones_like(boundary_r),
        bmag=np.ones_like(boundary_r),
    )

    bnormal, dr_dz = mirror_external_bnormal(boundary_r, z, sample, return_dr_dz=True)

    np.testing.assert_allclose(dr_dz, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(bnormal, 0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    ("boundary_r", "z", "sample_br", "match"),
    [
        (np.ones(3), np.linspace(-1.0, 1.0, 3), np.zeros((1, 3)), "boundary_r"),
        (np.ones((1, 3)), np.ones((1, 3)), np.zeros((1, 3)), "z must be"),
        (np.ones((1, 1)), np.asarray([0.0]), np.zeros((1, 1)), "at least two"),
        (np.ones((1, 3)), np.linspace(-1.0, 1.0, 3), np.zeros((1, 4)), "external field sample"),
    ],
)
def test_mirror_external_bnormal_rejects_invalid_shapes(boundary_r, z, sample_br, match):
    sample = MirrorExternalFieldSample(
        r=np.ones_like(sample_br),
        theta=np.asarray([0.0]),
        z=z,
        br=sample_br,
        btheta=np.zeros_like(sample_br),
        bz=np.ones_like(sample_br),
        bmag=np.ones_like(sample_br),
    )

    with pytest.raises(ValueError, match=match):
        mirror_external_bnormal(boundary_r, z, sample)


def test_mirror_lcfs_merit_combines_pressure_and_normal_field():
    diagnostic = SimpleNamespace(
        pressure_balance_rms=2.0,
        external_bnormal_rms=0.3,
        external_bmag=np.full((1, 4), 3.0),
    )

    merit = mirror_lcfs_merit(diagnostic, pressure_scale=2.0, bnormal_scale=3.0, bnormal_weight=4.0)

    assert merit.pressure_balance_rms == pytest.approx(2.0)
    assert merit.external_bnormal_rms == pytest.approx(0.3)
    assert merit.external_bmag_rms == pytest.approx(3.0)
    assert merit.value == pytest.approx(np.sqrt(1.0 + 4.0 * 0.1**2))


def test_mirror_lcfs_residual_vector_matches_merit_components():
    pressure_balance = np.asarray([[3.0, 4.0, 0.0]])
    external_bnormal = np.asarray([[0.5, -0.5, 1.0]])
    diagnostic = SimpleNamespace(
        pressure_balance=pressure_balance,
        pressure_balance_rms=float(np.sqrt(np.mean(pressure_balance**2))),
        external_bnormal=external_bnormal,
        external_bnormal_rms=float(np.sqrt(np.mean(external_bnormal**2))),
        external_bmag=np.full_like(pressure_balance, 2.0),
    )

    residual = mirror_lcfs_residual(
        diagnostic,
        pressure_scale=5.0,
        bnormal_scale=2.0,
        bnormal_weight=4.0,
    )
    merit = mirror_lcfs_merit(
        diagnostic,
        pressure_scale=5.0,
        bnormal_scale=2.0,
        bnormal_weight=4.0,
    )

    expected_pressure = pressure_balance / 5.0
    expected_bnormal = external_bnormal
    np.testing.assert_allclose(residual.pressure_component, expected_pressure)
    np.testing.assert_allclose(residual.bnormal_component, expected_bnormal)
    np.testing.assert_allclose(
        residual.vector,
        np.concatenate([expected_pressure.ravel(), expected_bnormal.ravel()]),
    )
    assert residual.value == pytest.approx(merit.value)
    assert residual.pressure_balance_rms == pytest.approx(diagnostic.pressure_balance_rms)
    assert residual.external_bnormal_rms == pytest.approx(diagnostic.external_bnormal_rms)


def test_mirror_free_boundary_residual_combines_equilibrium_and_lcfs_blocks():
    diagnostic = SimpleNamespace(
        pressure_balance=np.asarray([[1.0, -1.0]]),
        pressure_balance_rms=1.0,
        external_bnormal=np.asarray([[0.25, -0.25]]),
        external_bnormal_rms=0.25,
        external_bmag=np.ones((1, 2)),
    )
    lcfs = mirror_lcfs_residual(
        diagnostic,
        pressure_scale=2.0,
        bnormal_scale=1.0,
        bnormal_weight=4.0,
    )
    equilibrium = np.asarray([2.0, -4.0, 1.0])

    combined = mirror_free_boundary_residual(
        equilibrium,
        lcfs,
        equilibrium_scale=2.0,
        equilibrium_weight=4.0,
        lcfs_weight=9.0,
    )

    expected_equilibrium = 2.0 * equilibrium / 2.0
    expected_lcfs = 3.0 * lcfs.vector
    np.testing.assert_allclose(combined.equilibrium_component, expected_equilibrium)
    np.testing.assert_allclose(combined.lcfs_component, expected_lcfs)
    np.testing.assert_allclose(combined.vector, np.concatenate([expected_equilibrium, expected_lcfs]))
    assert combined.value == pytest.approx(np.sqrt(np.mean(expected_equilibrium**2) + np.mean(expected_lcfs**2)))
    assert combined.equilibrium_rms == pytest.approx(np.sqrt(np.mean(equilibrium**2)))
    assert combined.lcfs_value == pytest.approx(lcfs.value)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"equilibrium_residual": [], "equilibrium_scale": 1.0}, "equilibrium_residual"),
        ({"equilibrium_residual": [1.0], "equilibrium_scale": 0.0}, "equilibrium_scale"),
        ({"equilibrium_residual": [1.0], "equilibrium_weight": -1.0}, "equilibrium_weight"),
        ({"equilibrium_residual": [1.0], "lcfs_weight": -1.0}, "lcfs_weight"),
    ],
)
def test_mirror_free_boundary_residual_rejects_invalid_inputs(kwargs, match):
    diagnostic = SimpleNamespace(
        pressure_balance=np.asarray([[1.0]]),
        pressure_balance_rms=1.0,
        external_bnormal=np.asarray([[0.0]]),
        external_bnormal_rms=0.0,
        external_bmag=np.ones((1, 1)),
    )
    lcfs = mirror_lcfs_residual(diagnostic, pressure_scale=1.0, bnormal_scale=1.0)
    equilibrium = kwargs.pop("equilibrium_residual")

    with pytest.raises(ValueError, match=match):
        mirror_free_boundary_residual(equilibrium, lcfs, **kwargs)


def test_mirror_free_boundary_residual_rejects_nonfinite_blocks():
    lcfs = SimpleNamespace(vector=np.asarray([0.0]), value=0.0)

    with pytest.raises(ValueError, match="equilibrium_residual must be finite"):
        mirror_free_boundary_residual([np.inf], lcfs, equilibrium_scale=1.0)
    with pytest.raises(ValueError, match="lcfs_residual.vector"):
        mirror_free_boundary_residual([1.0], SimpleNamespace(vector=np.asarray([]), value=0.0))
    with pytest.raises(ValueError, match="lcfs_residual.vector must be finite"):
        mirror_free_boundary_residual([1.0], SimpleNamespace(vector=np.asarray([np.nan]), value=0.0))


def _free_boundary_residual_from_vector(vector):
    vector = np.asarray(vector, dtype=float).ravel()
    value = 0.0 if vector.size == 0 else float(np.sqrt(np.mean(vector**2)))
    return MirrorFreeBoundaryResidual(
        vector=vector,
        equilibrium_component=vector,
        lcfs_component=np.asarray([], dtype=float),
        value=value,
        equilibrium_rms=value,
        lcfs_value=0.0,
        equilibrium_scale=1.0,
        equilibrium_weight=1.0,
        lcfs_weight=0.0,
    )


def _linear_synthetic_free_boundary_residual(coefficients):
    target = np.asarray([0.4, -0.2])
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ]
    )
    vector = matrix @ (np.asarray(coefficients, dtype=float) - target)
    diagnostic = SimpleNamespace(
        pressure_balance=vector[2:3][None, :],
        pressure_balance_rms=float(np.sqrt(np.mean(vector[2:3] ** 2))),
        external_bnormal=vector[3:4][None, :],
        external_bnormal_rms=float(np.sqrt(np.mean(vector[3:4] ** 2))),
        external_bmag=np.ones((1, 1)),
    )
    lcfs = mirror_lcfs_residual(diagnostic, pressure_scale=1.0, bnormal_scale=1.0)
    return mirror_free_boundary_residual(
        vector[:2],
        lcfs,
        equilibrium_scale=1.0,
        equilibrium_weight=1.0,
        lcfs_weight=1.0,
    )


def test_mirror_free_boundary_residual_jacobian_finite_difference_matches_linear_model():
    coefficients = np.asarray([0.1, 0.3])
    residual, jacobian, steps = mirror_free_boundary_residual_jacobian_finite_difference(
        coefficients,
        _linear_synthetic_free_boundary_residual,
        finite_difference_step=1.0e-6,
    )

    expected_jacobian = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ]
    )
    assert residual.vector.shape == (4,)
    np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(steps, [1.0e-6, 1.0e-6])


def test_mirror_free_boundary_least_squares_step_reduces_linear_combined_residual():
    coefficients = np.asarray([0.1, 0.3])

    step = mirror_free_boundary_least_squares_step(
        coefficients,
        _linear_synthetic_free_boundary_residual,
        max_relative_step=2.0,
        line_search_factors=(1.0, 0.5),
    )

    assert step.accepted is True
    assert step.line_search_factor == pytest.approx(1.0)
    np.testing.assert_allclose(step.new_coefficients, [0.4, -0.2], rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(step.predicted_vector, 0.0, atol=1.0e-10)
    assert step.trial_residual.value < 1.0e-10
    assert step.trial_residual.value < step.residual.value


def test_mirror_free_boundary_least_squares_step_backtracks_nonlinear_residual():
    def nonlinear_residual(coefficients):
        c0 = float(np.asarray(coefficients, dtype=float)[0])
        equilibrium = np.asarray([c0 - 1.0 + 4.0 * c0**2])
        diagnostic = SimpleNamespace(
            pressure_balance=np.zeros((1, 1)),
            pressure_balance_rms=0.0,
            external_bnormal=np.zeros((1, 1)),
            external_bnormal_rms=0.0,
            external_bmag=np.ones((1, 1)),
        )
        lcfs = mirror_lcfs_residual(diagnostic, pressure_scale=1.0, bnormal_scale=1.0)
        return mirror_free_boundary_residual(equilibrium, lcfs, equilibrium_scale=1.0)

    step = mirror_free_boundary_least_squares_step(
        np.asarray([0.0]),
        nonlinear_residual,
        max_relative_step=2.0,
        line_search_factors=(1.0, 0.5, 0.25),
    )

    assert step.accepted is True
    assert step.line_search_factor == pytest.approx(0.5)
    np.testing.assert_allclose(step.new_coefficients, [0.5], atol=1.0e-10)
    assert step.trial_residual.value < step.residual.value


def test_mirror_free_boundary_least_squares_step_can_select_better_accepted_factor():
    def nonlinear_residual(coefficients):
        c0 = float(np.asarray(coefficients, dtype=float)[0])
        return _free_boundary_residual_from_vector([c0 - 1.0 + c0**2])

    step = mirror_free_boundary_least_squares_step(
        np.asarray([0.0]),
        nonlinear_residual,
        max_relative_step=2.0,
        line_search_factors=(1.0, 0.5),
    )

    assert step.accepted is True
    assert step.line_search_factor == pytest.approx(0.5)
    assert step.trial_residual.value < step.residual.value


def test_mirror_free_boundary_least_squares_step_rejects_all_worse_trials_with_ridge():
    def nonlinear_residual(coefficients):
        c0 = float(np.asarray(coefficients, dtype=float)[0])
        return _free_boundary_residual_from_vector([c0 - 1.0 + 100.0 * c0**2])

    step = mirror_free_boundary_least_squares_step(
        np.asarray([0.0]),
        nonlinear_residual,
        max_relative_step=2.0,
        ridge=0.1,
        line_search_factors=(1.0, 0.5),
    )

    assert step.accepted is False
    assert step.line_search_factor == pytest.approx(0.0)
    np.testing.assert_allclose(step.new_coefficients, [0.0])
    np.testing.assert_allclose(step.trial_residual.vector, step.residual.vector)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coefficients": []}, "coefficients"),
        ({"coefficients": [np.inf]}, "finite"),
        ({"finite_difference_step": 0.0}, "finite_difference_step"),
        ({"damping": 0.0}, "damping"),
        ({"max_relative_step": 0.0}, "max_relative_step"),
        ({"ridge": -1.0}, "ridge"),
        ({"accept_tolerance": -1.0}, "accept_tolerance"),
        ({"line_search_factors": ()}, "line_search_factors"),
        ({"line_search_factors": (1.0, -0.5)}, "line_search_factors"),
    ],
)
def test_mirror_free_boundary_least_squares_step_rejects_invalid_inputs(kwargs, match):
    coefficients = kwargs.pop("coefficients", [0.0, 0.0])

    with pytest.raises(ValueError, match=match):
        mirror_free_boundary_least_squares_step(
            coefficients,
            _linear_synthetic_free_boundary_residual,
            **kwargs,
        )


def test_mirror_free_boundary_residual_jacobian_rejects_non_residual_return():
    with pytest.raises(TypeError, match="MirrorFreeBoundaryResidual"):
        mirror_free_boundary_residual_jacobian_finite_difference(
            np.asarray([0.0]),
            lambda coefficients: coefficients,
        )


@pytest.mark.parametrize(
    ("coefficients", "match"),
    [
        ([], "coefficients"),
        ([np.inf], "finite"),
    ],
)
def test_mirror_free_boundary_residual_jacobian_rejects_invalid_coefficients(coefficients, match):
    with pytest.raises(ValueError, match=match):
        mirror_free_boundary_residual_jacobian_finite_difference(
            coefficients,
            _linear_synthetic_free_boundary_residual,
        )


@pytest.mark.parametrize(
    ("residual", "match"),
    [
        (_free_boundary_residual_from_vector([]), "empty residual vector"),
        (_free_boundary_residual_from_vector([np.inf]), "non-finite residual vector"),
    ],
)
def test_mirror_free_boundary_residual_jacobian_rejects_invalid_residual_vectors(residual, match):
    with pytest.raises(ValueError, match=match):
        mirror_free_boundary_residual_jacobian_finite_difference(
            np.asarray([0.0]),
            lambda coefficients: residual,
            residual=residual,
        )


def test_mirror_free_boundary_residual_jacobian_rejects_varying_vector_shape():
    base = _free_boundary_residual_from_vector([0.0])

    with pytest.raises(ValueError, match="fixed shape"):
        mirror_free_boundary_residual_jacobian_finite_difference(
            np.asarray([0.0]),
            lambda coefficients: _free_boundary_residual_from_vector([0.0, float(np.asarray(coefficients)[0])]),
            residual=base,
        )


def test_mirror_lcfs_residual_has_boundary_coefficient_finite_difference_jacobian():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 7)
    xi = z.copy()
    br = 0.1 * z[None, :]
    bz = np.ones((theta.size, z.size))
    external_sample = MirrorExternalFieldSample(
        r=np.ones((theta.size, z.size)),
        theta=np.broadcast_to(theta[:, None], (theta.size, z.size)),
        z=np.broadcast_to(z[None, :], (theta.size, z.size)),
        br=br,
        btheta=np.zeros_like(bz),
        bz=bz,
        bmag=np.sqrt(br**2 + bz**2),
    )
    edge_internal_bmag = np.ones((theta.size, z.size))

    def residual_vector(coefficients):
        r0, a2, a4 = np.asarray(coefficients, dtype=float)
        boundary_r = r0 * (1.0 + a2 * xi**2 + a4 * xi**4)[None, :]
        sample = MirrorExternalFieldSample(
            r=boundary_r,
            theta=external_sample.theta,
            z=external_sample.z,
            br=external_sample.br,
            btheta=external_sample.btheta,
            bz=external_sample.bz,
            bmag=external_sample.bmag,
        )
        diagnostic = mirror_lcfs_diagnostic_from_arrays(
            theta=theta,
            z=z,
            boundary_r=boundary_r,
            edge_internal_bmag=edge_internal_bmag,
            external_sample=sample,
            edge_pressure=0.0,
            mu0=1.0,
        )
        return mirror_lcfs_residual(
            diagnostic,
            pressure_scale=1.0,
            bnormal_scale=1.0,
            bnormal_weight=1.0,
        ).vector

    coefficients = np.asarray([0.3, 0.08, -0.02])
    step = 1.0e-6
    jacobian = np.column_stack(
        [
            (residual_vector(coefficients + step * basis) - residual_vector(coefficients - step * basis)) / (2.0 * step)
            for basis in np.eye(coefficients.size)
        ]
    )
    direction = np.asarray([0.02, -0.03, 0.01])
    directional = (
        residual_vector(coefficients + step * direction) - residual_vector(coefficients - step * direction)
    ) / (2.0 * step)

    assert jacobian.shape == (2 * z.size, coefficients.size)
    assert np.all(np.isfinite(jacobian))
    assert np.linalg.norm(jacobian) > 0.0
    np.testing.assert_allclose(jacobian @ direction, directional, rtol=1.0e-7, atol=5.0e-10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pressure_scale": 0.0}, "pressure_scale"),
        ({"bnormal_scale": 0.0}, "bnormal_scale"),
        ({"bnormal_weight": -1.0}, "bnormal_weight"),
    ],
)
def test_mirror_lcfs_merit_rejects_invalid_scales(kwargs, match):
    diagnostic = SimpleNamespace(
        pressure_balance_rms=2.0,
        external_bnormal_rms=0.3,
        external_bmag=np.full((1, 4), 3.0),
    )

    with pytest.raises(ValueError, match=match):
        mirror_lcfs_merit(diagnostic, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mu0": 0.0}, "mu0"),
        ({"radius_step_fraction": 0.0}, "radius_step_fraction"),
        ({"radius_step_min": 0.0}, "radius_step_min"),
        ({"radius_floor": 0.0}, "radius_floor"),
    ],
)
def test_mirror_external_pressure_balance_response_rejects_invalid_steps(kwargs, match):
    diagnostic = SimpleNamespace()

    with pytest.raises(ValueError, match=match):
        mirror_external_pressure_balance_response(diagnostic, provider_params=None, **kwargs)


def test_mirror_external_pressure_balance_response_samples_direct_circular_coils():
    enable_x64(True)
    theta = np.asarray([0.0])
    z = np.linspace(-0.5, 0.5, 3)
    boundary_r = np.full((theta.size, z.size), 0.2)
    diagnostic = SimpleNamespace(theta=theta, z=z, boundary_r=boundary_r)
    coils = MirrorCircularCoils.symmetric_pair(
        coil_radius_m=0.4,
        separation_m=1.2,
        current_a=8.0e5,
        n_segments=32,
    )

    response = mirror_external_pressure_balance_response(
        diagnostic,
        coils,
        radius_step_fraction=1.0e-3,
        radius_step_min=1.0e-5,
    )

    assert response.shape == boundary_r.shape
    assert np.all(np.isfinite(response))
    assert np.linalg.norm(response) > 0.0


def test_axisymmetric_lcfs_update_reduces_synthetic_pressure_imbalance():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 5)
    boundary_r = np.ones((theta.size, z.size))
    pressure_balance = np.asarray([[0.0, 0.2, -0.1, 0.2, 0.0]])
    diagnostic = SimpleNamespace(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        pressure_balance=pressure_balance,
    )
    pressure_response = np.ones_like(pressure_balance)

    proposal = propose_axisymmetric_mirror_lcfs_update(
        diagnostic,
        pressure_response,
        damping=1.0,
        max_relative_step=0.5,
        preserve_caps=True,
        cap_taper_power=0.0,
        smoothing_passes=0,
    )

    assert proposal.pressure_balance_rms_predicted < proposal.pressure_balance_rms_before
    np.testing.assert_allclose(proposal.pressure_balance_predicted, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(proposal.delta_radius[[0, -1]], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(proposal.new_radius, [1.0, 0.8, 1.1, 0.8, 1.0])
    np.testing.assert_allclose(proposal.boundary.radius(proposal.xi), proposal.new_radius)


def test_axisymmetric_lcfs_update_tapers_near_caps():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 9)
    boundary_r = np.ones((theta.size, z.size))
    pressure_balance = np.full((theta.size, z.size), 0.2)
    pressure_balance[:, 0] = 0.0
    pressure_balance[:, -1] = 0.0
    diagnostic = SimpleNamespace(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        pressure_balance=pressure_balance,
    )

    untapered = propose_axisymmetric_mirror_lcfs_update(
        diagnostic,
        np.ones_like(pressure_balance),
        damping=1.0,
        max_relative_step=0.5,
        preserve_caps=True,
        cap_taper_power=0.0,
        smoothing_passes=0,
    )
    tapered = propose_axisymmetric_mirror_lcfs_update(
        diagnostic,
        np.ones_like(pressure_balance),
        damping=1.0,
        max_relative_step=0.5,
        preserve_caps=True,
        cap_taper_power=2.0,
        smoothing_passes=1,
    )

    assert abs(tapered.delta_radius[1]) < abs(untapered.delta_radius[1])
    assert abs(tapered.delta_radius[-2]) < abs(untapered.delta_radius[-2])
    np.testing.assert_allclose(tapered.delta_radius[[0, -1]], 0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"damping": 0.0}, "damping"),
        ({"max_relative_step": 0.0}, "max_relative_step"),
        ({"radius_floor": 0.0}, "radius_floor"),
        ({"cap_taper_power": -1.0}, "cap_taper_power"),
        ({"smoothing_passes": -1}, "smoothing_passes"),
    ],
)
def test_axisymmetric_lcfs_update_rejects_invalid_options(kwargs, match):
    diagnostic = SimpleNamespace(
        theta=np.asarray([0.0]),
        z=np.linspace(-1.0, 1.0, 5),
        boundary_r=np.ones((1, 5)),
        pressure_balance=np.ones((1, 5)),
    )

    with pytest.raises(ValueError, match=match):
        propose_axisymmetric_mirror_lcfs_update(diagnostic, np.ones((1, 5)), **kwargs)


@pytest.mark.parametrize(
    ("diagnostic", "pressure_response", "match"),
    [
        (
            SimpleNamespace(
                theta=np.asarray([0.0]),
                z=np.asarray([0.0, -1.0]),
                boundary_r=np.ones((1, 2)),
                pressure_balance=np.ones((1, 2)),
            ),
            np.ones((1, 2)),
            "strictly increasing",
        ),
        (
            SimpleNamespace(
                theta=np.asarray([0.0]),
                z=np.linspace(-1.0, 1.0, 5),
                boundary_r=np.ones((1, 5)),
                pressure_balance=np.ones((1, 5)),
            ),
            np.ones((2, 5)),
            "pressure_response",
        ),
        (
            SimpleNamespace(
                theta=np.asarray([0.0]),
                z=np.linspace(-1.0, 1.0, 5),
                boundary_r=np.full((1, 5), np.nan),
                pressure_balance=np.ones((1, 5)),
            ),
            np.ones((1, 5)),
            "finite",
        ),
    ],
)
def test_axisymmetric_lcfs_update_rejects_invalid_arrays(diagnostic, pressure_response, match):
    with pytest.raises(ValueError, match=match):
        propose_axisymmetric_mirror_lcfs_update(diagnostic, pressure_response)


def test_axisymmetric_lcfs_scale_update_reduces_synthetic_pressure_imbalance():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 5)
    boundary_r = np.ones((theta.size, z.size))
    pressure_balance = np.full((theta.size, z.size), 0.2)
    diagnostic = SimpleNamespace(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        pressure_balance=pressure_balance,
    )

    proposal = propose_axisymmetric_mirror_lcfs_scale_update(
        diagnostic,
        np.ones_like(pressure_balance),
        max_relative_step=0.5,
    )

    assert proposal.strategy == "scale_pressure"
    assert proposal.pressure_balance_rms_predicted < proposal.pressure_balance_rms_before
    np.testing.assert_allclose(proposal.delta_radius, -0.2)
    np.testing.assert_allclose(proposal.pressure_balance_predicted, 0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_relative_step": 0.0}, "max_relative_step"),
        ({"radius_floor": 0.0}, "radius_floor"),
    ],
)
def test_axisymmetric_lcfs_scale_update_rejects_invalid_options(kwargs, match):
    diagnostic = SimpleNamespace(
        theta=np.asarray([0.0]),
        z=np.linspace(-1.0, 1.0, 5),
        boundary_r=np.ones((1, 5)),
        pressure_balance=np.ones((1, 5)),
    )

    with pytest.raises(ValueError, match=match):
        propose_axisymmetric_mirror_lcfs_scale_update(diagnostic, np.ones((1, 5)), **kwargs)


@pytest.mark.parametrize(
    ("diagnostic", "pressure_response", "match"),
    [
        (
            SimpleNamespace(
                theta=np.asarray([0.0]),
                z=np.asarray([0.0, -1.0]),
                boundary_r=np.ones((1, 2)),
                pressure_balance=np.ones((1, 2)),
            ),
            np.ones((1, 2)),
            "strictly increasing",
        ),
        (
            SimpleNamespace(
                theta=np.asarray([0.0]),
                z=np.linspace(-1.0, 1.0, 5),
                boundary_r=np.ones((1, 5)),
                pressure_balance=np.ones((1, 5)),
            ),
            np.ones((2, 5)),
            "pressure_response",
        ),
        (
            SimpleNamespace(
                theta=np.asarray([0.0]),
                z=np.linspace(-1.0, 1.0, 5),
                boundary_r=np.full((1, 5), np.nan),
                pressure_balance=np.ones((1, 5)),
            ),
            np.ones((1, 5)),
            "finite",
        ),
    ],
)
def test_axisymmetric_lcfs_scale_update_rejects_invalid_arrays(diagnostic, pressure_response, match):
    with pytest.raises(ValueError, match=match):
        propose_axisymmetric_mirror_lcfs_scale_update(diagnostic, pressure_response)


def test_axisymmetric_lcfs_noop_update_preserves_boundary_and_residual():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 5)
    boundary_r = np.linspace(0.8, 1.2, z.size)[None, :]
    pressure_balance = np.linspace(-0.1, 0.2, z.size)[None, :]
    diagnostic = SimpleNamespace(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        pressure_balance=pressure_balance,
    )

    proposal = propose_axisymmetric_mirror_lcfs_noop_update(diagnostic)

    assert proposal.strategy == "noop"
    np.testing.assert_allclose(proposal.new_radius, boundary_r[0])
    np.testing.assert_allclose(proposal.delta_radius, 0.0)
    np.testing.assert_allclose(proposal.pressure_balance_predicted, pressure_balance[0])


def test_axisymmetric_lcfs_noop_update_rejects_invalid_arrays():
    diagnostic = SimpleNamespace(
        theta=np.asarray([0.0]),
        z=np.asarray([0.0, -1.0]),
        boundary_r=np.ones((1, 2)),
        pressure_balance=np.ones((1, 2)),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        propose_axisymmetric_mirror_lcfs_noop_update(diagnostic)

    diagnostic.z = np.linspace(-1.0, 1.0, 5)
    diagnostic.boundary_r = np.ones((1, 5))
    diagnostic.pressure_balance = np.ones((1, 5))
    with pytest.raises(ValueError, match="pressure_response"):
        propose_axisymmetric_mirror_lcfs_noop_update(diagnostic, pressure_response=np.ones((2, 5)))


def test_axisymmetric_lcfs_bnormal_update_reduces_synthetic_normal_field():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 9)
    boundary_r = np.ones((theta.size, z.size))
    pressure_balance = np.zeros_like(boundary_r)
    br = 0.1 * z[None, :]
    sample = MirrorExternalFieldSample(
        r=boundary_r,
        theta=theta,
        z=z,
        br=br,
        btheta=np.zeros_like(boundary_r),
        bz=np.ones_like(boundary_r),
        bmag=np.sqrt(1.0 + br**2),
    )
    diagnostic = SimpleNamespace(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        pressure_balance=pressure_balance,
    )
    before = mirror_external_bnormal(boundary_r, z, sample)

    proposal = propose_axisymmetric_mirror_lcfs_bnormal_update(
        diagnostic,
        sample,
        np.zeros_like(boundary_r),
        max_relative_step=0.5,
        smoothing_passes=0,
    )
    after = mirror_external_bnormal(proposal.new_radius[None, :], z, sample)

    assert proposal.strategy == "bnormal_slope"
    assert float(np.sqrt(np.mean(after**2))) < 1.0e-12
    assert float(np.sqrt(np.mean(after**2))) < float(np.sqrt(np.mean(before**2)))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_relative_step": 0.0}, "max_relative_step"),
        ({"radius_floor": 0.0}, "radius_floor"),
        ({"slope_limit": 0.0}, "slope_limit"),
        ({"smoothing_passes": -1}, "smoothing_passes"),
    ],
)
def test_axisymmetric_lcfs_bnormal_update_rejects_invalid_options(kwargs, match):
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 5)
    boundary_r = np.ones((1, 5))
    sample = MirrorExternalFieldSample(
        r=boundary_r,
        theta=theta,
        z=z,
        br=np.zeros_like(boundary_r),
        btheta=np.zeros_like(boundary_r),
        bz=np.ones_like(boundary_r),
        bmag=np.ones_like(boundary_r),
    )
    diagnostic = SimpleNamespace(theta=theta, z=z, boundary_r=boundary_r, pressure_balance=np.zeros_like(boundary_r))

    with pytest.raises(ValueError, match=match):
        propose_axisymmetric_mirror_lcfs_bnormal_update(diagnostic, sample, np.ones_like(boundary_r), **kwargs)


def test_axisymmetric_lcfs_bnormal_update_rejects_invalid_arrays_and_smooths():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 5)
    boundary_r = np.ones((1, 5))
    diagnostic = SimpleNamespace(theta=theta, z=z, boundary_r=boundary_r, pressure_balance=np.zeros_like(boundary_r))
    sample = MirrorExternalFieldSample(
        r=boundary_r,
        theta=theta,
        z=z,
        br=np.linspace(-0.1, 0.1, z.size)[None, :],
        btheta=np.zeros_like(boundary_r),
        bz=np.ones_like(boundary_r),
        bmag=np.ones_like(boundary_r),
    )

    bad_z = SimpleNamespace(
        theta=theta, z=np.asarray([0.0, -1.0]), boundary_r=np.ones((1, 2)), pressure_balance=np.ones((1, 2))
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        propose_axisymmetric_mirror_lcfs_bnormal_update(bad_z, sample, np.ones((1, 2)))
    bad_sample = MirrorExternalFieldSample(
        r=np.ones((1, 4)),
        theta=theta,
        z=z,
        br=np.ones((1, 4)),
        btheta=np.zeros((1, 4)),
        bz=np.ones((1, 4)),
        bmag=np.ones((1, 4)),
    )
    with pytest.raises(ValueError, match="external field sample"):
        propose_axisymmetric_mirror_lcfs_bnormal_update(diagnostic, bad_sample, np.ones_like(boundary_r))
    with pytest.raises(ValueError, match="pressure_response"):
        propose_axisymmetric_mirror_lcfs_bnormal_update(diagnostic, sample, np.ones((2, 5)))

    default_response = propose_axisymmetric_mirror_lcfs_bnormal_update(
        diagnostic,
        sample,
        smoothing_passes=0,
    )
    np.testing.assert_allclose(default_response.pressure_response, 0.0)

    proposal = propose_axisymmetric_mirror_lcfs_bnormal_update(
        diagnostic,
        sample,
        np.ones_like(boundary_r),
        smoothing_passes=1,
    )
    assert proposal.strategy == "bnormal_slope"
    assert np.all(np.isfinite(proposal.new_radius))


def test_axisymmetric_lcfs_mixed_update_improves_pressure_without_increasing_normal_field():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 9)
    boundary_r = np.ones((theta.size, z.size))
    pressure_balance = np.full_like(boundary_r, 0.2)
    br = 0.1 * z[None, :]
    sample = MirrorExternalFieldSample(
        r=boundary_r,
        theta=theta,
        z=z,
        br=br,
        btheta=np.zeros_like(boundary_r),
        bz=np.ones_like(boundary_r),
        bmag=np.sqrt(1.0 + br**2),
    )
    diagnostic = SimpleNamespace(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        pressure_balance=pressure_balance,
    )
    before = mirror_external_bnormal(boundary_r, z, sample)

    proposal = propose_axisymmetric_mirror_lcfs_mixed_update(
        diagnostic,
        sample,
        np.ones_like(boundary_r),
        scale_fractions=(1.0,),
        bnormal_fractions=(1.0,),
        max_relative_step=0.5,
        smoothing_passes=0,
    )
    after = mirror_external_bnormal(proposal.new_radius[None, :], z, sample)

    assert proposal.strategy == "mixed_scale_bnormal"
    assert proposal.pressure_balance_rms_predicted < proposal.pressure_balance_rms_before
    assert float(np.sqrt(np.mean(after**2))) <= float(np.sqrt(np.mean(before**2))) + 1.0e-14


def test_axisymmetric_lcfs_candidate_set_returns_standard_strategies():
    theta = np.asarray([0.0])
    z = np.linspace(-1.0, 1.0, 9)
    boundary_r = np.ones((theta.size, z.size))
    pressure_balance = np.full_like(boundary_r, 0.2)
    br = 0.1 * z[None, :]
    sample = MirrorExternalFieldSample(
        r=boundary_r,
        theta=theta,
        z=z,
        br=br,
        btheta=np.zeros_like(boundary_r),
        bz=np.ones_like(boundary_r),
        bmag=np.sqrt(1.0 + br**2),
    )
    diagnostic = SimpleNamespace(
        theta=theta,
        z=z,
        boundary_r=boundary_r,
        pressure_balance=pressure_balance,
    )

    candidates = propose_axisymmetric_mirror_lcfs_candidate_set(
        diagnostic,
        sample,
        np.ones_like(boundary_r),
        max_relative_step=0.5,
        smoothing_passes=0,
    )

    assert [candidate.strategy for candidate in candidates] == [
        "local_pressure",
        "scale_pressure",
        "bnormal_slope",
        "mixed_scale_bnormal",
        "noop",
    ]

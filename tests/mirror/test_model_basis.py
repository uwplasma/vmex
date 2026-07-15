"""M0/M1 contracts and spectral identities for the mirror backend."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import vmec_jax.mirror as mirror_api  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    EndCondition,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    solve_free_boundary_cli,
)
from vmec_jax.mirror.basis import ChebyshevBasis, ThetaBasis  # noqa: E402
from vmec_jax.mirror.forces import mirror_energy  # noqa: E402
from vmec_jax.mirror.model import (  # noqa: E402
    MIRROR_INPUT_SCHEMA,
    MIRROR_OUTPUT_SCHEMA,
    project_fixed_boundary_state,
)
from vmec_jax.mirror.output import (  # noqa: E402
    boundary_fourier_amplitudes,
    boundary_fourier_norms,
    summarize_axisymmetric_beta_scan,
    summarize_nonaxisymmetric_beta_scan,
)


def test_public_api_keeps_numerical_kernels_in_owning_modules() -> None:
    required = {
        "MirrorConfig",
        "MirrorState",
        "solve_fixed_boundary_cli",
        "solve_free_boundary_cli",
        "solve_beta_scan_cli",
        "solve_fixed_boundary_implicit",
        "spline_fixed_boundary_adjoint",
        "spline_fixed_boundary_tangent",
        "write_mout",
        "plot_mout",
    }
    internal = {
        "ChebyshevBasis",
        "SeparableMirrorPreconditioner",
        "isotropic_force_residual",
        "solve_reduced_exterior_laplace_neumann",
    }
    assert required <= set(mirror_api.__all__)
    assert internal.isdisjoint(mirror_api.__all__)
    assert len(mirror_api.__all__) == 20


def test_mirror_config_freezes_supported_end_and_convergence_contract() -> None:
    config = MirrorConfig()
    assert config.end_condition is EndCondition.FIXED_FLUX_CUT
    assert config.ftol == 1.0e-12
    assert config.max_iterations == 2000
    assert MIRROR_INPUT_SCHEMA == "vmec_jax.mirror.input/2"
    assert MIRROR_OUTPUT_SCHEMA == "vmec_jax.mirror.mout/1"

    assert MirrorResolution(mpol=0).ntheta == 1
    assert MirrorResolution(mpol=4).ntheta == 9
    with pytest.raises(TypeError, match="ntheta"):
        MirrorResolution(ns=5, mpol=2, ntheta=5, nxi=9)
    with pytest.raises(ValueError, match="z_max"):
        MirrorConfig(z_min=1.0, z_max=1.0)
    with pytest.raises(ValueError, match="ftol"):
        MirrorConfig(ftol=0.0)


def test_grid_and_state_shapes_are_explicit_and_pytree_compatible() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=3, nxi=13),
        z_min=-2.0,
        z_max=3.0,
    )
    grid = config.build_grid()
    assert grid.shape == (7, 7, 13)
    assert grid.xi[0] == -1.0 and grid.xi[-1] == 1.0
    assert np.all(np.diff(grid.xi) > 0.0)
    assert np.isclose(grid.z[0], -2.0) and np.isclose(grid.z[-1], 3.0)
    assert np.isclose(np.sum(grid.radial_weights), 1.0)
    assert np.isclose(np.sum(grid.theta_basis.weights), 2.0 * np.pi)
    assert np.isclose(np.sum(grid.axial_basis.weights), 2.0)

    radius = 0.3 * (1.0 + 0.1 * jnp.asarray(grid.xi) ** 2)
    boundary = MirrorBoundary.from_radius(radius, grid)
    state = MirrorState.from_boundary(boundary, grid)
    state.validate_shape(grid)
    assert state.radius_scale.shape == grid.shape
    assert state.lambda_stream.shape == grid.shape
    leaves, structure = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 2
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    np.testing.assert_allclose(rebuilt.radius_scale, state.radius_scale)


def test_cgl_derivative_and_quadrature_are_polynomial_exact() -> None:
    basis = ChebyshevBasis.build(18)
    x = basis.nodes
    for power in range(basis.size):
        values = jnp.asarray(x**power)
        expected = np.zeros_like(x) if power == 0 else power * x ** (power - 1)
        np.testing.assert_allclose(basis.differentiate(values), expected, rtol=2.0e-11, atol=2.0e-11)

    assert np.all(basis.weights > 0.0)
    assert np.isclose(np.sum(basis.weights), 2.0)
    for power in range(basis.size):
        expected = 0.0 if power % 2 else 2.0 / (power + 1)
        assert np.isclose(
            float(basis.integrate(jnp.asarray(x**power))),
            expected,
            rtol=4.0e-13,
            atol=4.0e-13,
        )


def test_cgl_operators_obey_integration_by_parts_and_spectral_interpolation() -> None:
    coarse = ChebyshevBasis.build(17)
    x = jnp.asarray(coarse.nodes)
    f = 1.0 + x + 0.3 * x**5
    g = 0.7 - 0.2 * x**2 + x**4
    lhs = coarse.integrate(coarse.differentiate(f) * g + f * coarse.differentiate(g))
    rhs = f[-1] * g[-1] - f[0] * g[0]
    np.testing.assert_allclose(lhs, rhs, rtol=2.0e-13, atol=2.0e-13)

    fine = ChebyshevBasis.build(65)
    values = jnp.exp(x) + 0.25 * jnp.cos(3.0 * x)
    interpolated = coarse.interpolate(values, fine.nodes)
    expected = np.exp(fine.nodes) + 0.25 * np.cos(3.0 * fine.nodes)
    np.testing.assert_allclose(interpolated, expected, rtol=5.0e-11, atol=5.0e-11)
    roundtrip = fine.interpolate(interpolated, coarse.nodes)
    np.testing.assert_allclose(roundtrip, values, rtol=2.0e-13, atol=2.0e-13)

    rng = np.random.default_rng(17)
    left = jnp.asarray(rng.normal(size=coarse.size))
    right = jnp.asarray(rng.normal(size=coarse.size))
    np.testing.assert_allclose(
        jnp.vdot(left, coarse.differentiate(right)),
        jnp.vdot(coarse.differentiate_transpose(left), right),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_theta_fft_derivative_and_quadrature_resolve_requested_modes() -> None:
    basis = ThetaBasis.build(ntheta=13, mpol=6)
    theta = jnp.asarray(basis.nodes)
    values = 0.7 + 0.4 * jnp.cos(3.0 * theta) - 0.2 * jnp.sin(5.0 * theta)
    expected = -1.2 * jnp.sin(3.0 * theta) - 1.0 * jnp.cos(5.0 * theta)
    np.testing.assert_allclose(basis.differentiate(values), expected, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(basis.integrate(values), 1.4 * np.pi, rtol=2.0e-13, atol=2.0e-13)

    left = 0.3 + jnp.sin(2.0 * theta) - 0.1 * jnp.cos(4.0 * theta)
    np.testing.assert_allclose(
        jnp.vdot(left, basis.differentiate(values)),
        jnp.vdot(basis.differentiate_transpose(left), values),
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    axisym = ThetaBasis.build(ntheta=1, mpol=0)
    np.testing.assert_array_equal(axisym.differentiate(jnp.asarray([3.0])), jnp.asarray([0.0]))


def _grid(*, ntheta: int = 1, nxi: int = 5):
    return MirrorConfig(
        resolution=MirrorResolution(ns=3, mpol=(ntheta - 1) // 2, nxi=nxi)
    ).build_grid()


def test_model_constructors_reject_invalid_static_contracts() -> None:
    for arguments, message in (
        ({"ns": 2}, "ns"),
        ({"mpol": -1}, "mpol"),
        ({"nxi": 1}, "nxi"),
    ):
        with pytest.raises(ValueError, match=message):
            MirrorResolution(**arguments)
    assert MirrorResolution().axisymmetric
    with pytest.raises(ValueError, match="end condition"):
        MirrorConfig(end_condition="invalid")
    with pytest.raises(ValueError, match="max_iterations"):
        MirrorConfig(max_iterations=0)

    grid = _grid()
    integer_boundary = MirrorBoundary.from_radius(1, grid)
    assert jnp.issubdtype(integer_boundary.radius_scale.dtype, jnp.inexact)
    with pytest.raises(ValueError, match="radius shape"):
        MirrorBoundary.from_radius(jnp.ones((2, 2)), grid)
    with pytest.raises(ValueError, match="on_axis_bz"):
        MirrorBoundary.from_axis_field(0.1, jnp.ones(2), grid)
    with pytest.raises(ValueError, match="scalar"):
        MirrorBoundary.from_axis_field(jnp.ones(2), jnp.ones(grid.nxi), grid)
    with pytest.raises(ValueError, match="boundary shape"):
        MirrorState.from_boundary(MirrorBoundary(jnp.ones((2, 2))), grid)

    state = MirrorState.from_boundary(integer_boundary, grid)
    with pytest.raises(ValueError, match="radius_scale"):
        MirrorState(jnp.ones(2), state.lambda_stream).validate_shape(grid)
    with pytest.raises(ValueError, match="lambda_stream"):
        MirrorState(state.radius_scale, jnp.ones(2)).validate_shape(grid)
    with pytest.raises(ValueError, match="boundary shape"):
        project_fixed_boundary_state(state, MirrorBoundary(jnp.ones((2, 2))), grid)


def test_fourier_and_beta_diagnostics_validate_inputs() -> None:
    grid = _grid()
    with pytest.raises(ValueError, match="shape"):
        boundary_fourier_amplitudes(MirrorBoundary(jnp.ones(5)))
    even_theta = jnp.linspace(0.0, 2.0 * jnp.pi, 4, endpoint=False)
    nyquist = boundary_fourier_amplitudes(MirrorBoundary(0.2 + 0.01 * jnp.cos(2.0 * even_theta)[:, None]))
    np.testing.assert_allclose(nyquist[2], 0.01)
    with pytest.raises(ValueError, match="axial size"):
        boundary_fourier_norms(MirrorBoundary(jnp.ones((1, 4))), grid)
    with pytest.raises(ValueError, match="central_fraction"):
        boundary_fourier_norms(MirrorBoundary(jnp.ones((1, 5))), grid, central_fraction=1.0)

    with pytest.raises(ValueError, match="one value"):
        summarize_axisymmetric_beta_scan((), [0.0], grid, reference_field=1.0)
    with pytest.raises(ValueError, match="at least one"):
        summarize_axisymmetric_beta_scan((), [], grid, reference_field=1.0)
    with pytest.raises(ValueError, match="ntheta=1"):
        summarize_axisymmetric_beta_scan((object(),), [0.0], _grid(ntheta=3), reference_field=1.0)
    with pytest.raises(ValueError, match="one value"):
        summarize_nonaxisymmetric_beta_scan((), [0.0], _grid(ntheta=3), reference_field=1.0)
    with pytest.raises(ValueError, match="at least one"):
        summarize_nonaxisymmetric_beta_scan((), [], _grid(ntheta=3), reference_field=1.0)
    with pytest.raises(ValueError, match="ntheta > 1"):
        summarize_nonaxisymmetric_beta_scan((object(),), [0.0], grid, reference_field=1.0)


@pytest.mark.parametrize("ntheta", [1, 3])
def test_beta_diagnostics_evaluate_solved_state(ntheta: int) -> None:
    grid = _grid(ntheta=ntheta)
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    radius = 0.3 + (0.0 if ntheta == 1 else 0.01 * xi * jnp.cos(theta))
    boundary = MirrorBoundary.from_radius(radius, grid)
    state = MirrorState.from_boundary(boundary, grid)
    energy = mirror_energy(state, grid, axial_flux_derivative=0.1)
    pressure = jnp.broadcast_to(energy.pressure[:, None, None], grid.shape)
    result = SimpleNamespace(
        boundary=boundary,
        plasma_energy=energy,
        plasma_b_squared=energy.b_squared,
        pressure=pressure,
        vacuum_field=SimpleNamespace(lateral_field_xyz=jnp.ones((grid.nxi, 3))),
    )
    if ntheta == 1:
        diagnostic = summarize_axisymmetric_beta_scan((result,), [0.0], grid, reference_field=1.0)[0]
        assert float(diagnostic.center_radius) > 0.0
        assert float(diagnostic.center_vacuum_side_field) > 0.0
    else:
        diagnostic = summarize_nonaxisymmetric_beta_scan((result,), [0.0], grid, reference_field=1.0)[0]
        assert float(diagnostic.plasma_volume) > 0.0
        assert float(diagnostic.boundary_mode_core_l2[1]) > 0.0


def test_free_boundary_rejects_inconsistent_static_inputs() -> None:
    grid = _grid()
    boundary = MirrorBoundary.from_radius(0.2, grid)
    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    common = dict(
        initial_boundary=boundary,
        plasma_grid=grid,
        config=config,
        external_field=object(),
        axial_flux_derivative=0.1,
    )

    with pytest.raises(ValueError, match="chunk"):
        solve_free_boundary_cli(**common, exterior_jacobian_chunk_size=0)
    with pytest.raises(ValueError, match="target_central_pressure"):
        solve_free_boundary_cli(**common, target_central_pressure=0.0)
    with pytest.raises(ValueError, match="initial_mass_scale"):
        solve_free_boundary_cli(**common, initial_mass_scale=0.0)
    with pytest.raises(ValueError, match="initial boundary"):
        solve_free_boundary_cli(**{**common, "initial_boundary": MirrorBoundary(jnp.ones((2, 5)))})


def test_free_boundary_rejects_inconsistent_initial_guesses() -> None:
    grid = _grid()
    boundary = MirrorBoundary.from_radius(0.2, grid)
    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    common = dict(
        initial_boundary=boundary,
        plasma_grid=grid,
        config=config,
        external_field=object(),
        axial_flux_derivative=0.1,
    )
    wrong_boundary = MirrorBoundary.from_radius(0.3, grid)
    with pytest.raises(ValueError, match="initial_state boundary"):
        solve_free_boundary_cli(**common, initial_state=MirrorState.from_boundary(wrong_boundary, grid))

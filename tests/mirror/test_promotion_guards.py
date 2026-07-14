"""Input and diagnostic contracts used by the mirror promotion audit."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    BiMaxwellianPressureClosure,
    IsotropicPressureClosure,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    TabulatedPressureClosure,
    build_vacuum_grid,
    solve_free_boundary_cli,
)
from vmec_jax.mirror.diagnostics import (  # noqa: E402
    boundary_fourier_amplitudes,
    boundary_fourier_norms,
    summarize_axisymmetric_beta_scan,
    summarize_nonaxisymmetric_beta_scan,
)
from vmec_jax.mirror.forces import mirror_energy  # noqa: E402
from vmec_jax.mirror.model import project_fixed_boundary_state  # noqa: E402


def _grid(*, ntheta: int = 1, nxi: int = 5):
    return MirrorConfig(
        resolution=MirrorResolution(
            ns=3, mpol=0 if ntheta == 1 else 1, ntheta=ntheta, nxi=nxi
        )
    ).build_grid()


def test_model_constructors_reject_invalid_static_contracts() -> None:
    for arguments, message in (
        ({"ns": 2}, "ns"),
        ({"mpol": -1}, "mpol"),
        ({"nxi": 1}, "nxi"),
        ({"mpol": 1, "ntheta": 2}, "resolve"),
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
        project_fixed_boundary_state(
            state, MirrorBoundary(jnp.ones((2, 2))), grid
        )

    with pytest.raises(ValueError, match="greater than one"):
        IsotropicPressureClosure(jnp.ones(1), gamma=1.0)
    for arguments, message in (
        ({"temperature_ratio": 0.0, "critical_field": 1.0}, "temperature_ratio"),
        ({"temperature_ratio": 0.5, "critical_field": 0.0}, "critical_field"),
        (
            {"temperature_ratio": 0.5, "critical_field": 1.0, "gamma": 1.0},
            "gamma",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            BiMaxwellianPressureClosure(
                jnp.ones(1), jnp.ones(1), **arguments
            )


@pytest.mark.parametrize(
    ("s_nodes", "b_nodes", "values", "gamma", "message"),
    [
        ([[0.0, 1.0]], [1.0, 2.0], np.ones((2, 2)), 0.0, "one-dimensional"),
        ([0.0], [1.0, 2.0], np.ones((1, 2)), 0.0, "at least two"),
        ([0.0, 1.0], [1.0, 2.0], np.ones((3, 2)), 0.0, "shape"),
        ([0.0, 0.0], [1.0, 2.0], np.ones((2, 2)), 0.0, "increasing"),
        ([0.0, 1.0], [1.0, 2.0], np.ones((2, 2)), 1.0, "gamma"),
    ],
)
def test_tabulated_closure_validates_tables(
    s_nodes, b_nodes, values, gamma, message
) -> None:
    with pytest.raises(ValueError, match=message):
        TabulatedPressureClosure(s_nodes, b_nodes, values, gamma=gamma)


def test_fourier_and_beta_diagnostics_validate_inputs() -> None:
    grid = _grid()
    with pytest.raises(ValueError, match="shape"):
        boundary_fourier_amplitudes(MirrorBoundary(jnp.ones(5)))
    even_theta = jnp.linspace(0.0, 2.0 * jnp.pi, 4, endpoint=False)
    nyquist = boundary_fourier_amplitudes(
        MirrorBoundary(0.2 + 0.01 * jnp.cos(2.0 * even_theta)[:, None])
    )
    np.testing.assert_allclose(nyquist[2], 0.01)
    with pytest.raises(ValueError, match="axial size"):
        boundary_fourier_norms(MirrorBoundary(jnp.ones((1, 4))), grid)
    with pytest.raises(ValueError, match="central_fraction"):
        boundary_fourier_norms(
            MirrorBoundary(jnp.ones((1, 5))), grid, central_fraction=1.0
        )

    with pytest.raises(ValueError, match="one value"):
        summarize_axisymmetric_beta_scan((), [0.0], grid, reference_field=1.0)
    with pytest.raises(ValueError, match="at least one"):
        summarize_axisymmetric_beta_scan((), [], grid, reference_field=1.0)
    with pytest.raises(ValueError, match="ntheta=1"):
        summarize_axisymmetric_beta_scan(
            (object(),), [0.0], _grid(ntheta=3), reference_field=1.0
        )
    with pytest.raises(ValueError, match="one value"):
        summarize_nonaxisymmetric_beta_scan(
            (), [0.0], _grid(ntheta=3), reference_field=1.0
        )
    with pytest.raises(ValueError, match="at least one"):
        summarize_nonaxisymmetric_beta_scan(
            (), [], _grid(ntheta=3), reference_field=1.0
        )
    with pytest.raises(ValueError, match="ntheta > 1"):
        summarize_nonaxisymmetric_beta_scan(
            (object(),), [0.0], grid, reference_field=1.0
        )


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
        perpendicular_pressure=pressure,
        vacuum_field=SimpleNamespace(
            lateral_field_xyz=jnp.ones((grid.nxi, 3))
        ),
    )
    if ntheta == 1:
        diagnostic = summarize_axisymmetric_beta_scan(
            (result,), [0.0], grid, reference_field=1.0
        )[0]
        assert float(diagnostic.center_radius) > 0.0
        assert float(diagnostic.center_vacuum_side_field) > 0.0
    else:
        diagnostic = summarize_nonaxisymmetric_beta_scan(
            (result,), [0.0], grid, reference_field=1.0
        )[0]
        assert float(diagnostic.plasma_volume) > 0.0
        assert float(diagnostic.boundary_mode_core_l2[1]) > 0.0


def test_free_boundary_rejects_inconsistent_static_inputs() -> None:
    grid = _grid()
    vacuum_grid = build_vacuum_grid(grid, nrho=3)
    boundary = MirrorBoundary.from_radius(0.2, grid)
    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    common = dict(
        initial_boundary=boundary,
        plasma_grid=grid,
        vacuum_grid=vacuum_grid,
        config=config,
        external_field=object(),
        outer_radius=0.4,
        axial_flux_derivative=0.1,
    )

    with pytest.raises(ValueError, match="vacuum_backend"):
        solve_free_boundary_cli(**common, vacuum_backend="invalid")
    with pytest.raises(ValueError, match="curved side"):
        solve_free_boundary_cli(
            **common, vacuum_backend="exterior", exterior_curved_side_geometry=True
        )
    with pytest.raises(ValueError, match="chunk"):
        solve_free_boundary_cli(
            **common, vacuum_backend="exterior", exterior_jacobian_chunk_size=0
        )
    with pytest.raises(ValueError, match="target_central_pressure"):
        solve_free_boundary_cli(**common, target_central_pressure=0.0)
    with pytest.raises(ValueError, match="initial_mass_scale"):
        solve_free_boundary_cli(**common, initial_mass_scale=0.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        solve_free_boundary_cli(
            **common,
            mass_profile=1.0,
            pressure_closure=IsotropicPressureClosure(jnp.asarray([1.0])),
        )
    with pytest.raises(ValueError, match="initial boundary"):
        solve_free_boundary_cli(
            **{**common, "initial_boundary": MirrorBoundary(jnp.ones((2, 5)))}
        )

    mismatched_theta = _grid(ntheta=3)
    with pytest.raises(ValueError, match="theta nodes"):
        solve_free_boundary_cli(
            **{**common, "vacuum_grid": build_vacuum_grid(mismatched_theta, nrho=3)}
        )
    mismatched_axial = _grid(nxi=7)
    with pytest.raises(ValueError, match="axial nodes"):
        solve_free_boundary_cli(
            **{**common, "vacuum_grid": build_vacuum_grid(mismatched_axial, nrho=3)}
        )
    nonaxisymmetric = _grid(ntheta=3)
    with pytest.raises(ValueError, match="nonaxisymmetric"):
        solve_free_boundary_cli(
            initial_boundary=MirrorBoundary.from_radius(0.2, nonaxisymmetric),
            plasma_grid=nonaxisymmetric,
            vacuum_grid=build_vacuum_grid(nonaxisymmetric, nrho=3),
            config=MirrorConfig(
                resolution=MirrorResolution(
                    ns=3, mpol=1, ntheta=3, nxi=5
                )
            ),
            external_field=object(),
            outer_radius=0.4,
            axial_flux_derivative=0.1,
        )


def test_free_boundary_rejects_inconsistent_initial_guesses() -> None:
    grid = _grid()
    vacuum_grid = build_vacuum_grid(grid, nrho=3)
    boundary = MirrorBoundary.from_radius(0.2, grid)
    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    common = dict(
        initial_boundary=boundary,
        plasma_grid=grid,
        vacuum_grid=vacuum_grid,
        config=config,
        external_field=object(),
        outer_radius=0.4,
        axial_flux_derivative=0.1,
    )
    wrong_boundary = MirrorBoundary.from_radius(0.3, grid)
    with pytest.raises(ValueError, match="initial_state boundary"):
        solve_free_boundary_cli(
            **common, initial_state=MirrorState.from_boundary(wrong_boundary, grid)
        )
    with pytest.raises(ValueError, match="initial_potential shape"):
        solve_free_boundary_cli(**common, initial_potential=jnp.zeros(2))
    with pytest.raises(ValueError, match="inside the outer"):
        solve_free_boundary_cli(
            **{**common, "initial_boundary": MirrorBoundary.from_radius(0.5, grid)}
        )

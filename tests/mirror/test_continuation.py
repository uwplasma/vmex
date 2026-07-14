"""Resolution-continuation tests for mirror states."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    build_vacuum_grid,
)
from vmec_jax.mirror.continuation import (  # noqa: E402
    interpolate_fixed_boundary_state,
    solve_axisymmetric_beta_scan_cli,
)
import vmec_jax.mirror.continuation as continuation  # noqa: E402
from vmec_jax.mirror.model import project_fixed_boundary_state  # noqa: E402
from vmec_jax.mirror.output import FreeBoundaryRestart  # noqa: E402


def _grid(ns: int, nxi: int):
    return MirrorConfig(resolution=MirrorResolution(ns=ns, mpol=1, ntheta=3, nxi=nxi)).build_grid()


def test_fixed_boundary_state_interpolation_roundtrips_and_preserves_constraints() -> None:
    coarse, fine = _grid(7, 9), _grid(11, 13)

    def fields(grid):
        s = jnp.asarray(grid.s)[:, None, None]
        theta = jnp.asarray(grid.theta)[None, :, None]
        xi = jnp.asarray(grid.xi)[None, None, :]
        radius = 0.2 + 0.1 * s + 0.01 * jnp.cos(theta) * (1.0 - xi**2)
        lam = s * jnp.cos(theta) * (1.0 - xi**2)
        boundary = MirrorBoundary.from_radius(radius[-1], grid)
        state = project_fixed_boundary_state(MirrorState(radius, lam), boundary, grid)
        return boundary, state

    coarse_boundary, coarse_state = fields(coarse)
    fine_boundary, _ = fields(fine)
    interpolated = interpolate_fixed_boundary_state(coarse_state, coarse, fine_boundary, fine)
    roundtrip = interpolate_fixed_boundary_state(interpolated, fine, coarse_boundary, coarse)

    # Axis closure deliberately flattens the first radial interval, so only
    # surfaces outside that interval are an exact coarse-fine round trip.
    np.testing.assert_allclose(roundtrip.radius_scale[2:], coarse_state.radius_scale[2:], atol=3.0e-14)
    np.testing.assert_allclose(roundtrip.lambda_stream[2:], coarse_state.lambda_stream[2:], atol=3.0e-14)
    np.testing.assert_allclose(interpolated.radius_scale[-1], fine_boundary.radius_scale)
    np.testing.assert_allclose(interpolated.radius_scale[0], interpolated.radius_scale[1])
    np.testing.assert_allclose(interpolated.lambda_stream[0], interpolated.lambda_stream[1])
    np.testing.assert_allclose(interpolated.lambda_stream[:, :, [0, -1]], 0.0, atol=2.0e-15)


def test_fixed_boundary_state_interpolation_is_differentiable() -> None:
    coarse, fine = _grid(5, 7), _grid(7, 9)
    boundary = MirrorBoundary.from_radius(0.3, coarse)
    state = MirrorState.from_boundary(boundary, coarse)
    fine_boundary = MirrorBoundary.from_radius(0.3, fine)

    tangent = jnp.ones(coarse.shape).at[:, :, [0, -1]].set(0.0)

    def interpolate_lambda(scale):
        varied = MirrorState(state.radius_scale, state.lambda_stream + scale * tangent)
        return interpolate_fixed_boundary_state(varied, coarse, fine_boundary, fine).lambda_stream

    derivative = jax.jacfwd(interpolate_lambda)(1.0)
    assert bool(jnp.all(jnp.isfinite(derivative)))
    assert float(jnp.linalg.norm(derivative)) > 0.0


def test_beta_scan_propagates_restart_mass_scale(monkeypatch) -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=5))
    grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(grid, nrho=3)
    reference = MirrorBoundary.from_radius(0.3, grid)
    restart_boundary = MirrorBoundary.from_radius(0.31, grid)
    restart_state = MirrorState.from_boundary(restart_boundary, grid)
    restart = FreeBoundaryRestart(restart_boundary, restart_state, jnp.zeros(vacuum_grid.shape), 2.5)
    received = []

    def fake_solve(boundary, *_args, **kwargs):
        received.append(
            (
                boundary,
                kwargs["initial_state"],
                kwargs["initial_mass_scale"],
                kwargs["exterior_spectral_side_density"],
            )
        )
        return SimpleNamespace(
            boundary=boundary,
            plasma_state=kwargs["initial_state"],
            vacuum_potential=kwargs["initial_potential"],
            mass_scale=jnp.asarray(kwargs["initial_mass_scale"] + 0.5),
        )

    monkeypatch.setattr(continuation, "solve_axisymmetric_free_boundary_cli", fake_solve)
    solve_axisymmetric_beta_scan_cli(
        reference,
        grid,
        vacuum_grid,
        config,
        object(),
        jnp.asarray([0.0, 0.0]),
        outer_radius=0.6,
        axial_flux_derivative=0.1,
        reference_field=1.0,
        initial_restart=restart,
        exterior_spectral_side_density=True,
    )

    assert received[0][0] is restart_boundary
    assert received[0][1] is restart_state
    assert received[0][2] == 2.5
    assert received[1][2] == 3.0
    assert all(item[3] is True for item in received)

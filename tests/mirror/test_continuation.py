"""Resolution-continuation tests for mirror states."""

from __future__ import annotations

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
    interpolate_fixed_boundary_state,
    project_fixed_boundary_state,
)


def _grid(ns: int, nxi: int):
    return MirrorConfig(
        resolution=MirrorResolution(ns=ns, mpol=1, ntheta=3, nxi=nxi)
    ).build_grid()


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
    interpolated = interpolate_fixed_boundary_state(
        coarse_state, coarse, fine_boundary, fine
    )
    roundtrip = interpolate_fixed_boundary_state(
        interpolated, fine, coarse_boundary, coarse
    )

    # Axis closure deliberately flattens the first radial interval, so only
    # surfaces outside that interval are an exact coarse-fine round trip.
    np.testing.assert_allclose(
        roundtrip.radius_scale[2:], coarse_state.radius_scale[2:], atol=3.0e-14
    )
    np.testing.assert_allclose(
        roundtrip.lambda_stream[2:], coarse_state.lambda_stream[2:], atol=3.0e-14
    )
    np.testing.assert_allclose(interpolated.radius_scale[-1], fine_boundary.radius_scale)
    np.testing.assert_allclose(interpolated.radius_scale[0], interpolated.radius_scale[1])
    np.testing.assert_allclose(interpolated.lambda_stream[0], 0.0, atol=2.0e-15)
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

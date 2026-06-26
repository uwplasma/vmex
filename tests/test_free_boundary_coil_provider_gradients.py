from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import CoilFieldParams
from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import direct_coil_boundary_bnormal_rms_jax


def _accepted_boundary_fixture():
    from vmec_jax._compat import jnp

    base_dofs = jnp.zeros((1, 3, 3), dtype=float)
    base_dofs = base_dofs.at[0, 0, 2].set(1.55)
    base_dofs = base_dofs.at[0, 1, 1].set(1.55)
    base_currents = jnp.asarray([2.5e7], dtype=float)

    return {
        "base_dofs": base_dofs,
        "base_currents": base_currents,
        "R": jnp.asarray([[0.78, 0.86, 0.91], [0.83, 0.94, 0.80]], dtype=float),
        "Z": jnp.asarray([[0.16, -0.13, 0.21], [-0.18, 0.24, -0.11]], dtype=float),
        "phi": jnp.asarray([[0.05, 0.45, 0.90], [1.20, 1.55, 2.10]], dtype=float),
        "Ru": jnp.asarray([[0.03, -0.04, 0.02], [0.05, -0.03, 0.01]], dtype=float),
        "Zu": jnp.asarray([[0.22, 0.24, 0.21], [0.23, 0.20, 0.25]], dtype=float),
        "Rv": jnp.asarray([[0.04, 0.01, -0.03], [0.05, -0.02, 0.03]], dtype=float),
        "Zv": jnp.asarray([[0.02, -0.03, 0.06], [-0.01, 0.05, -0.02]], dtype=float),
        "br_add": jnp.asarray([[0.03, -0.02, 0.01], [0.015, -0.025, 0.02]], dtype=float),
        "bp_add": jnp.asarray([[-0.04, 0.01, 0.035], [0.02, -0.03, 0.015]], dtype=float),
        "bz_add": jnp.asarray([[0.02, 0.015, -0.025], [-0.01, 0.03, -0.02]], dtype=float),
    }


def test_direct_coil_accepted_boundary_replay_gradients_include_background_field():
    """AD/FD gate for the accepted-boundary replay primitive.

    This is intentionally a fixed-boundary replay test: it validates
    differentiability of the direct-coil field, cylindrical boundary
    projection, and additive non-coil background-field channels used by the
    free-boundary replay path.  It does not claim differentiation through the
    nonlinear VMEC iteration loop.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)
    data = _accepted_boundary_fixture()

    def params_for(current_scale, geometry_scale):
        return CoilFieldParams(
            base_curve_dofs=data["base_dofs"].at[0, 0, 2].add(0.02 * geometry_scale),
            base_currents=data["base_currents"] * (1.0 + 0.03 * current_scale),
            n_segments=96,
            regularization_epsilon=1.0e-9,
        )

    def objective(current_scale=0.0, geometry_scale=0.0, background_scale=0.0):
        background_factor = 1.0 + 0.07 * background_scale
        return direct_coil_boundary_bnormal_rms_jax(
            params_for(current_scale, geometry_scale),
            R=data["R"],
            Z=data["Z"],
            phi=data["phi"],
            Ru=data["Ru"],
            Zu=data["Zu"],
            Rv=data["Rv"],
            Zv=data["Zv"],
            br_add=background_factor * data["br_add"],
            bp_add=background_factor * data["bp_add"],
            bz_add=background_factor * data["bz_add"],
        )

    eps = 1.0e-4
    cases = (
        lambda scale: objective(current_scale=scale),
        lambda scale: objective(geometry_scale=scale),
        lambda scale: objective(background_scale=scale),
    )
    for scalar_objective in cases:
        exact = jax.grad(scalar_objective)(0.0)
        fd = (scalar_objective(eps) - scalar_objective(-eps)) / (2.0 * eps)
        assert np.isfinite(float(exact))
        assert abs(float(exact)) > 1.0e-8
        np.testing.assert_allclose(exact, fd, rtol=5.0e-7, atol=1.0e-10)


def test_direct_coil_accepted_boundary_replay_gradient_wrt_boundary_geometry():
    """AD/FD gate for accepted-boundary geometry sensitivity.

    The full nonlinear free-boundary adjoint needs the accepted plasma
    boundary's dependence on coil controls.  This smaller gate keeps that
    boundary fixed but verifies that the replay primitive is differentiable
    with respect to accepted ``R``/``Z`` coordinates themselves.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    data = _accepted_boundary_fixture()
    params = CoilFieldParams(
        base_curve_dofs=data["base_dofs"],
        base_currents=data["base_currents"],
        n_segments=96,
        regularization_epsilon=1.0e-9,
    )
    dR = jnp.asarray([[0.012, -0.009, 0.007], [-0.011, 0.006, -0.004]], dtype=float)
    dZ = jnp.asarray([[-0.006, 0.008, -0.005], [0.009, -0.007, 0.004]], dtype=float)

    def objective(boundary_scale):
        return direct_coil_boundary_bnormal_rms_jax(
            params,
            R=data["R"] + boundary_scale * dR,
            Z=data["Z"] + boundary_scale * dZ,
            phi=data["phi"],
            Ru=data["Ru"],
            Zu=data["Zu"],
            Rv=data["Rv"],
            Zv=data["Zv"],
            br_add=data["br_add"],
            bp_add=data["bp_add"],
            bz_add=data["bz_add"],
        )

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-5
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    assert np.isfinite(float(exact))
    assert abs(float(exact)) > 1.0e-9
    np.testing.assert_allclose(exact, fd, rtol=2.0e-7, atol=1.0e-10)

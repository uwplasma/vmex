import numpy as np
import jax.numpy as jnp

from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.modes import vmec_mode_table
from vmec_jax.optimization import (
    BoundaryParamSpec,
    apply_boundary_params,
    boundary_param_names,
    boundary_param_specs,
    gauss_newton_least_squares,
    lift_boundary_params,
    surface_indices_from_s,
)


def test_boundary_param_specs_and_apply():
    modes = vmec_mode_table(mpol=2, ntor=1)
    k = modes.K
    boundary = BoundaryCoeffs(
        R_cos=np.linspace(1.0, 2.0, k),
        R_sin=np.zeros(k),
        Z_cos=np.zeros(k),
        Z_sin=np.linspace(0.1, 0.2, k),
    )

    specs = boundary_param_specs(
        boundary,
        modes,
        max_mode=1,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    names = boundary_param_names(specs)

    assert "rc00" not in names
    assert any(name.startswith("rc1") for name in names)
    assert any(name.startswith("zs1") for name in names)

    params = jnp.ones((len(specs),))
    updated = apply_boundary_params(boundary, specs, params)

    # rc00 should remain unchanged
    assert np.isclose(updated.R_cos[0], boundary.R_cos[0])
    # At least one other coefficient should change
    assert not np.allclose(np.asarray(updated.R_cos), np.asarray(boundary.R_cos))


def test_surface_indices_from_s():
    s_half = np.array([0.1, 0.3, 0.5, 0.7])
    indices, selected = surface_indices_from_s(s_half, [0.28, 3])
    assert indices == [1, 2]
    np.testing.assert_allclose(selected, np.array([0.3, 0.5]))


def test_lift_boundary_params_maps_shared_names_and_zeros_new_modes():
    source_specs = [
        BoundaryParamSpec("rc10", "rc", 0, 1, 0),
        BoundaryParamSpec("zs10", "zs", 1, 1, 0),
    ]
    target_specs = [
        BoundaryParamSpec("rc10", "rc", 0, 1, 0),
        BoundaryParamSpec("zs10", "zs", 1, 1, 0),
        BoundaryParamSpec("rc21", "rc", 2, 2, 1),
    ]

    lifted = lift_boundary_params(source_specs, np.array([0.25, -0.5]), target_specs)

    np.testing.assert_allclose(lifted, np.array([0.25, -0.5, 0.0]))


def test_gauss_newton_least_squares_solves_linear_problem():
    def residual(x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] - 1.0, 2.0 * x[1] - 2.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0, 0.0], dtype=float),
        max_nfev=5,
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
        verbose=0,
    )

    np.testing.assert_allclose(result["x"], np.array([1.0, 1.0]), atol=1e-12, rtol=0.0)
    assert result["success"]
    assert result["objective"] <= 1e-20


def test_gauss_newton_post_jacobian_callback():
    """post_jacobian_callback is called once per jacobian evaluation."""
    call_counts = [0]

    def residual(x):
        return np.array([float(x[0]) - 1.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0]], dtype=float)

    def on_jac():
        call_counts[0] += 1

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0], dtype=float),
        max_nfev=5,
        post_jacobian_callback=on_jac,
        verbose=0,
    )

    assert result["success"]
    assert call_counts[0] == result["njev"]


def test_gauss_newton_exact_residual_after_jacobian():
    """exact_residual_after_jacobian_fun replaces the residual used for gradient."""
    # Set up a problem where forward_residual_fun gives a deliberately noisy
    # residual, but exact_residual_after_jacobian_fun provides the correct one.
    # The optimizer should still converge because the exact residual is used
    # for the gradient computation after each Jacobian call.
    rng = np.random.default_rng(42)
    noise_scale = 0.5

    def residual(x):
        return np.array([float(x[0]) - 1.0], dtype=float)

    def noisy_residual(x):
        return residual(x) + noise_scale * rng.standard_normal(1)

    # Track the most recent jacobian x so we can return the exact residual.
    last_x = [None]

    def jacobian(x):
        last_x[0] = float(x[0])
        return np.array([[1.0]], dtype=float)

    def exact_residual():
        if last_x[0] is None:
            return None
        return np.array([last_x[0] - 1.0], dtype=float)

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0], dtype=float),
        forward_residual_fun=noisy_residual,
        exact_residual_after_jacobian_fun=exact_residual,
        max_nfev=20,
        verbose=0,
    )

    assert result["success"]
    np.testing.assert_allclose(result["x"], np.array([1.0]), atol=1e-3, rtol=0.0)


def test_gauss_newton_helper_matches_scipy_linear_problem():
    """The standalone SciPy path should solve the same linear least-squares problem."""

    try:
        from scipy.optimize import least_squares
    except Exception:  # pragma: no cover - optional dependency
        return

    def residual(x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] - 1.0, 2.0 * x[1] - 2.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)

    result = least_squares(
        residual,
        np.array([0.0, 0.0], dtype=float),
        jac=jacobian,
        method="trf",
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )

    np.testing.assert_allclose(result.x, np.array([1.0, 1.0]), atol=1e-12, rtol=0.0)
    assert result.success

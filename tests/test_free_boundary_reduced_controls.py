from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.solvers.free_boundary import (
    ReducedControlMap,
    reduced_control_decode,
    reduced_control_least_squares_step,
    reduced_control_pullback,
)


def test_reduced_control_least_squares_step_is_public() -> None:
    import vmec_jax as vj
    import vmec_jax.api as public_api

    assert vj.ReducedControlMap is ReducedControlMap
    assert public_api.ReducedControlMap is ReducedControlMap
    assert vj.reduced_control_decode is reduced_control_decode
    assert public_api.reduced_control_decode is reduced_control_decode
    assert vj.reduced_control_least_squares_step is reduced_control_least_squares_step
    assert public_api.reduced_control_least_squares_step is reduced_control_least_squares_step
    assert vj.reduced_control_pullback is reduced_control_pullback
    assert public_api.reduced_control_pullback is reduced_control_pullback


def test_reduced_control_least_squares_step_reports_exact_and_uncontrolled_parts() -> None:
    jacobian = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )
    target = np.asarray([3.0, 4.0, 5.0])

    step = reduced_control_least_squares_step(jacobian, target, labels=("side", "corner"))

    np.testing.assert_allclose(step.control_delta, [3.0, 2.0])
    np.testing.assert_allclose(step.predicted_delta, [3.0, 4.0, 0.0])
    np.testing.assert_allclose(step.residual_after, [0.0, 0.0, 5.0])
    assert step.control_delta_by_label == {"side": 3.0, "corner": 2.0}
    assert step.rank == 2
    assert step.condition_number == pytest.approx(2.0)
    assert step.residual_l2 == pytest.approx(5.0)
    assert step.residual_rel == pytest.approx(5.0 / np.sqrt(50.0))
    assert step.to_dict()["control_delta_by_label"] == {"side": 3.0, "corner": 2.0}


def test_reduced_control_least_squares_step_supports_ridge_damping() -> None:
    step = reduced_control_least_squares_step([[1.0]], [2.0], ridge=3.0)

    np.testing.assert_allclose(step.control_delta, [0.5])
    np.testing.assert_allclose(step.predicted_delta, [0.5])
    np.testing.assert_allclose(step.residual_after, [1.5])
    assert step.ridge == pytest.approx(3.0)
    assert step.trust_scale == pytest.approx(1.0)


def test_reduced_control_least_squares_step_supports_trust_radius() -> None:
    step = reduced_control_least_squares_step(np.eye(2), [3.0, 4.0], trust_radius=2.5)

    np.testing.assert_allclose(step.control_delta, [1.5, 2.0])
    np.testing.assert_allclose(step.predicted_delta, [1.5, 2.0])
    assert step.control_l2 == pytest.approx(2.5)
    assert step.trust_scale == pytest.approx(0.5)


def test_reduced_control_map_encodes_decodes_and_projects_boundary_values() -> None:
    control_map = ReducedControlMap(
        initial=np.asarray([10.0, -1.0, 0.5]),
        jacobian=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        ),
        labels=("side", "corner"),
        rcond=1.0e-12,
    )
    full_values = np.asarray([13.0, 3.0, 7.5])

    step = control_map.encode(full_values)
    decoded = control_map.decode(step.control_delta)
    projected = control_map.project(full_values)
    pulled = control_map.pullback([1.0, 2.0, 3.0])
    payload = control_map.to_dict()

    np.testing.assert_allclose(step.control_delta, [3.0, 2.0])
    np.testing.assert_allclose(decoded, [13.0, 3.0, 0.5])
    np.testing.assert_allclose(projected, decoded)
    np.testing.assert_allclose(step.residual_after, [0.0, 0.0, 7.0])
    assert step.control_delta_by_label == {"side": 3.0, "corner": 2.0}
    assert control_map.full_size == 3
    assert control_map.control_count == 2
    np.testing.assert_allclose(pulled, [1.0, 4.0])
    assert payload["rank"] == 2
    assert payload["rank_deficient"] is False
    assert payload["labels"] == ["side", "corner"]


def test_reduced_control_decode_matches_host_map_and_jacobian() -> None:
    import jax

    initial = np.asarray([10.0, -1.0, 0.5])
    jacobian = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )
    controls = jnp.asarray([3.0, 2.0])
    control_map = ReducedControlMap(initial=initial, jacobian=jacobian, labels=("side", "corner"))

    decoded = reduced_control_decode(initial, jacobian, controls)
    decoded_from_map = control_map.decode_jax(controls)
    derivative = jax.jacfwd(lambda values: reduced_control_decode(initial, jacobian, values))(controls)

    np.testing.assert_allclose(np.asarray(decoded), control_map.decode(np.asarray(controls)), atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(decoded_from_map), np.asarray(decoded), atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(derivative), jacobian, atol=1.0e-14)


def test_reduced_control_pullback_matches_decode_vjp() -> None:
    import jax

    initial = np.asarray([10.0, -1.0, 0.5])
    jacobian = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [0.5, -1.0],
        ]
    )
    controls = jnp.asarray([3.0, 2.0])
    full_adjoint = jnp.asarray([7.0, 11.0, 13.0])
    control_map = ReducedControlMap(initial=initial, jacobian=jacobian, labels=("side", "corner"))

    pulled = reduced_control_pullback(jacobian, full_adjoint)
    pulled_from_map = control_map.pullback_jax(full_adjoint)
    _decoded, vjp_fun = jax.vjp(lambda values: reduced_control_decode(initial, jacobian, values), controls)
    pulled_from_vjp = vjp_fun(full_adjoint)[0]

    expected = jacobian.T @ np.asarray(full_adjoint)
    np.testing.assert_allclose(control_map.pullback(np.asarray(full_adjoint)), expected, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(pulled), expected, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(pulled_from_map), expected, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(pulled_from_vjp), expected, atol=1.0e-14)


@pytest.mark.parametrize(
    ("jacobian", "target", "kwargs", "match"),
    [
        ([[1.0, 2.0]], [1.0], {"labels": ("a",)}, "labels length"),
        ([[1.0, 2.0]], [1.0, 2.0], {}, "row count"),
        ([[1.0, np.nan]], [1.0], {}, "finite"),
        ([[1.0]], [1.0], {"ridge": -1.0}, "ridge"),
        ([[1.0]], [1.0], {"rcond": -1.0}, "rcond"),
        ([[1.0]], [1.0], {"trust_radius": 0.0}, "trust_radius"),
    ],
)
def test_reduced_control_least_squares_step_rejects_invalid_inputs(
    jacobian, target, kwargs, match
) -> None:
    with pytest.raises(ValueError, match=match):
        reduced_control_least_squares_step(jacobian, target, **kwargs)


@pytest.mark.parametrize(
    ("initial", "jacobian", "kwargs", "match"),
    [
        ([0.0], [[[1.0]]], {}, "two-dimensional"),
        ([0.0, 0.0], [[1.0]], {}, "row count"),
        ([0.0], [[]], {}, "at least one control"),
        ([np.nan], [[1.0]], {}, "finite"),
        ([0.0], [[1.0]], {"labels": ("a", "b")}, "labels length"),
        ([0.0], [[1.0]], {"rcond": -1.0}, "rcond"),
    ],
)
def test_reduced_control_map_rejects_invalid_inputs(initial, jacobian, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        ReducedControlMap(initial=initial, jacobian=jacobian, **kwargs)


def test_reduced_control_map_rejects_mismatched_decode_and_encode_sizes() -> None:
    control_map = ReducedControlMap(initial=[0.0, 0.0], jacobian=np.eye(2))

    with pytest.raises(ValueError, match="full_values size"):
        control_map.encode([1.0])
    with pytest.raises(ValueError, match="control_delta size"):
        control_map.decode([1.0])
    with pytest.raises(ValueError, match="full_values size"):
        control_map.pullback([1.0])
    with pytest.raises(ValueError, match="finite"):
        control_map.decode([1.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        control_map.pullback([1.0, np.nan])
    with pytest.raises(ValueError, match="control_delta size"):
        reduced_control_decode([0.0, 0.0], np.eye(2), [1.0])
    with pytest.raises(ValueError, match="full_values size"):
        reduced_control_pullback(np.eye(2), [1.0])

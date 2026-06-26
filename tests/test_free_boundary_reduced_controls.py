from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.solvers.free_boundary import reduced_control_least_squares_step


def test_reduced_control_least_squares_step_is_public() -> None:
    import vmec_jax as vj
    import vmec_jax.api as public_api

    assert vj.reduced_control_least_squares_step is reduced_control_least_squares_step
    assert public_api.reduced_control_least_squares_step is reduced_control_least_squares_step


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

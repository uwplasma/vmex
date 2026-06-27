from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.solvers.free_boundary import (
    FreeBoundaryNativeSplineState,
    FreeBoundaryReducedEdgeState,
    ReducedControlMap,
    ReducedControlState,
    free_boundary_reduced_edge_state_from_vmec_state,
    free_boundary_reduced_edge_state_to_vmec_state,
    reduced_control_decode,
    reduced_control_least_squares_step,
    reduced_control_pullback,
)
from vmec_jax.config import VMECConfig
from vmec_jax.namelist import InData
from vmec_jax.solvers.free_boundary.control import _prepare_freeb_edge_control_projection
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.static import build_static


def test_reduced_control_least_squares_step_is_public() -> None:
    import vmec_jax as vj
    import vmec_jax.api as public_api

    assert vj.ReducedControlMap is ReducedControlMap
    assert vj.ReducedControlState is ReducedControlState
    assert vj.FreeBoundaryNativeSplineState is FreeBoundaryNativeSplineState
    assert vj.FreeBoundaryReducedEdgeState is FreeBoundaryReducedEdgeState
    assert public_api.ReducedControlMap is ReducedControlMap
    assert public_api.ReducedControlState is ReducedControlState
    assert public_api.FreeBoundaryNativeSplineState is FreeBoundaryNativeSplineState
    assert public_api.FreeBoundaryReducedEdgeState is FreeBoundaryReducedEdgeState
    assert (
        vj.free_boundary_reduced_edge_state_from_vmec_state
        is free_boundary_reduced_edge_state_from_vmec_state
    )
    assert (
        public_api.free_boundary_reduced_edge_state_from_vmec_state
        is free_boundary_reduced_edge_state_from_vmec_state
    )
    assert (
        vj.free_boundary_reduced_edge_state_to_vmec_state
        is free_boundary_reduced_edge_state_to_vmec_state
    )
    assert (
        public_api.free_boundary_reduced_edge_state_to_vmec_state
        is free_boundary_reduced_edge_state_to_vmec_state
    )
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


def test_reduced_control_state_tracks_native_coordinates() -> None:
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

    state = ReducedControlState.from_full_values(control_map, [13.0, 3.0, 7.5])
    updated = state.update([0.25, -0.5])
    payload = updated.to_dict()

    np.testing.assert_allclose(state.control_delta, [3.0, 2.0])
    np.testing.assert_allclose(state.decode(), [13.0, 3.0, 0.5])
    np.testing.assert_allclose(updated.control_delta, [3.25, 1.5])
    np.testing.assert_allclose(updated.decode(), [13.25, 2.0, 0.5])
    np.testing.assert_allclose(np.asarray(updated.decode_jax()), updated.decode())
    assert updated.control_delta_by_label == {"side": 3.25, "corner": 1.5}
    assert payload["reduced_unknown_size"] == 2
    assert payload["unknown_by_label"] == {"side": 3.25, "corner": 1.5}


def test_free_boundary_reduced_edge_state_encodes_vmec_lcfs_edge_and_pullback() -> None:
    import jax

    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=True,
        ntheta=8,
        nzeta=1,
    )
    static = build_static(cfg)
    layout = StateLayout(ns=3, K=static.modes.K, lasym=False)
    zeros = np.zeros((3, static.modes.K), dtype=float)
    anchor_rcos = zeros.copy()
    anchor_rcos[-1, 0] = 3.0
    state0 = VMECState(
        layout=layout,
        Rcos=anchor_rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    indata = InData(
        scalars={"MPOL": 2, "NTOR": 0, "NS_ARRAY": [3], "NFP": 1, "LASYM": False, "LCONM1": True},
        indexed={"RBC": {(0, 0): 3.0}},
    )
    jacobian = np.zeros((4 * static.modes.K, 1), dtype=float)
    jacobian[0, 0] = 1.0
    projection = _prepare_freeb_edge_control_projection(
        {
            "enabled": True,
            "basis_symmetry": "test",
            "labels": ["R00"],
            "control_jacobian": jacobian,
            "update_mode": "native_coordinate",
        },
        indata=indata,
        static=static,
        state0=state0,
        free_boundary_enabled=True,
    )

    rcos = zeros.copy()
    rcos[-1, 0] = 3.25
    rcos[-1, 1] = 0.5
    state = VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )

    reduced_edge = free_boundary_reduced_edge_state_from_vmec_state(state, projection)
    native_spline_state = FreeBoundaryNativeSplineState.from_vmec_state(state, projection)
    decoded_state = free_boundary_reduced_edge_state_to_vmec_state(
        reduced_edge,
        state,
        projection,
        host_update=True,
    )
    decoded_state_jax = free_boundary_reduced_edge_state_to_vmec_state(
        reduced_edge,
        state,
        projection,
        host_update=False,
    )
    updated_state = free_boundary_reduced_edge_state_to_vmec_state(
        reduced_edge.update([0.5]),
        state,
        projection,
        host_update=True,
    )
    native_decoded_state = native_spline_state.to_vmec_state(host_update=True)
    native_decoded_state_jax = native_spline_state.to_vmec_state(host_update=False)
    native_updated_state = native_spline_state.update_edge([0.5]).to_vmec_state(host_update=True)
    force_dR = zeros.copy()
    force_dR[-1, 0] = 2.0
    force_deltas = (
        force_dR,
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
    )
    full_adjoint = jnp.asarray([2.0, 7.0, *([0.0] * (4 * static.modes.K - 2))])
    _decoded, vjp_fun = jax.vjp(
        lambda values: reduced_edge.control_state.control_map.decode_jax(values),
        jnp.asarray(reduced_edge.control_delta),
    )

    assert projection["enabled"] is True
    assert native_spline_state.control_delta_by_label == {"R00": pytest.approx(0.25)}
    assert native_spline_state.to_dict()["mode"] == "free_boundary_native_spline_state"
    assert native_spline_state.to_dict()["full_edge_size"] == 4 * static.modes.K
    assert reduced_edge.control_delta_by_label == {"R00": pytest.approx(0.25)}
    assert reduced_edge.fit_residual_linf > 0.0
    assert reduced_edge.to_dict()["mode"] == "native_reduced_lcfs_edge_state"
    np.testing.assert_allclose(reduced_edge.decode_edge_values()[0], 3.25)
    np.testing.assert_allclose(np.asarray(reduced_edge.decode_edge_values_jax())[0], 3.25)
    assert np.asarray(decoded_state.Rcos)[-1, 0] == pytest.approx(3.25)
    assert np.asarray(decoded_state.Rcos)[-1, 1] == pytest.approx(0.0)
    np.testing.assert_allclose(np.asarray(decoded_state_jax.Rcos), np.asarray(decoded_state.Rcos))
    assert np.asarray(updated_state.Rcos)[-1, 0] == pytest.approx(3.75)
    np.testing.assert_allclose(np.asarray(native_decoded_state.Rcos), np.asarray(decoded_state.Rcos))
    np.testing.assert_allclose(
        np.asarray(native_decoded_state_jax.Rcos),
        np.asarray(decoded_state.Rcos),
    )
    assert np.asarray(native_updated_state.Rcos)[-1, 0] == pytest.approx(3.75)
    np.testing.assert_allclose(native_spline_state.pullback_delta_tuple(force_deltas), [2.0])
    np.testing.assert_allclose(
        np.asarray(reduced_edge.pullback_jax(full_adjoint)),
        np.asarray(vjp_fun(full_adjoint)[0]),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(reduced_edge.pullback(np.asarray(full_adjoint)), [2.0])


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
    with pytest.raises(ValueError, match="control_delta size"):
        ReducedControlState(control_map=control_map, control_delta=[1.0])
    with pytest.raises(ValueError, match="control_update size"):
        ReducedControlState(control_map=control_map, control_delta=[1.0, 2.0]).update([1.0])

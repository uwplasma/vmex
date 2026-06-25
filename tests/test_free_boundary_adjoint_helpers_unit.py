from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.free_boundary_adjoint as fba
from vmec_jax._compat import jnp
from vmec_jax.solvers.free_boundary.adjoint import objectives as objective_helpers
from vmec_jax.solvers.free_boundary.adjoint import branch_local as branch_local_helpers
from vmec_jax.solvers.free_boundary.adjoint import direct_coil_replay as direct_coil_replay_helpers
from vmec_jax.solvers.free_boundary.adjoint import pytrees as pytree_helpers
from vmec_jax.solvers.free_boundary.adjoint import replay_plan as replay_plan_helpers
from vmec_jax.solvers.free_boundary.adjoint import runtime as runtime_helpers
from vmec_jax.solvers.free_boundary.adjoint import trace_controls
from vmec_jax.solvers.free_boundary.adjoint import trace_fingerprint
from vmec_jax.solvers.free_boundary.adjoint import trace_metadata
from vmec_jax.solvers.free_boundary.adjoint import trace_stack


def test_free_boundary_adjoint_runtime_helpers_sync_and_scope_fallbacks() -> None:
    assert runtime_helpers.block_until_ready_for_timing(
        {"value": 1.0},
        jax_module=None,
        tree_util_module=None,
    ) == {"value": 1.0}

    class ReadyModule:
        @staticmethod
        def block_until_ready(value):
            if isinstance(value, dict):
                raise TypeError("top-level container unsupported")
            return f"{value}-ready"

    class TreeUtilModule:
        @staticmethod
        def tree_map(fn, value):
            return {key: fn(item) for key, item in value.items()}

    assert runtime_helpers.block_until_ready_for_timing(
        {"leaf": "x"},
        jax_module=ReadyModule(),
        tree_util_module=TreeUtilModule(),
    ) == {"leaf": "x-ready"}

    with runtime_helpers.jax_named_scope("fallback", jax_module=None):
        pass

    class ScopeContext:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    class ScopeModule:
        @staticmethod
        def named_scope(name):
            assert name == "named"
            return ScopeContext()

    with runtime_helpers.jax_named_scope(
        "named",
        jax_module=ScopeModule(),
        nullcontext_factory=lambda: pytest.fail("named_scope should be used"),
    ):
        pass


def test_free_boundary_branch_local_helpers_validate_payloads_and_report_flags() -> None:
    init = SimpleNamespace(static=SimpleNamespace(cfg=SimpleNamespace(nfp=2, mpol=2, ntor=1, lasym=False)), signgs=1)
    trace = {
        "step_status": "accepted",
        "freeb_bsqvac_half": None,
        "freeb_nestor_trace": None,
        "preconditioner_use_precomputed_tridi": False,
        "preconditioner_use_lax_tridi": True,
        "precond_jmax": 3,
        "precond_mats": {"a": np.ones((1,))},
        "lam_prec": np.ones((2,)),
        "w_mode_mn": np.ones((2, 3)),
    }
    payload = {"init": init, "params": {"current": 1.0}, "traces": [trace]}

    branch_payload = branch_local_helpers.prepare_branch_local_payload(
        input_path=None,
        params=None,
        complete_payload=payload,
        init_kwargs=None,
        solve_kwargs=None,
        require_active_trace=False,
        complete_solve_trace_func=lambda *_args, **_kwargs: pytest.fail("complete solve should not run"),
    )
    assert branch_payload.params == {"current": 1.0}
    assert branch_payload.traces == (trace,)
    assert "payload_copy_wall_s" in branch_payload.timings

    with pytest.raises(RuntimeError, match="active free-boundary trace"):
        branch_local_helpers.prepare_branch_local_payload(
            input_path=None,
            params={"current": 1.0},
            complete_payload=payload,
            init_kwargs=None,
            solve_kwargs=None,
            require_active_trace=True,
            complete_solve_trace_func=lambda *_args, **_kwargs: payload,
        )

    values, source = branch_local_helpers.evaluate_branch_local_production_values(
        payload=payload,
        scalar_fn=lambda _payload: {"objective": 2.0, "aspect": 5.0},
        production_values=None,
        timings=branch_payload.timings,
    )
    assert source == "scalar_fn"
    assert values == {"objective": 2.0, "aspect": 5.0}
    assert branch_local_helpers.select_branch_local_scalar_key(values, None) == "objective"
    assert branch_local_helpers.select_branch_local_scalar_keys(
        all_values=values,
        replay_scalar_fns={"objective": object(), "aspect": object()},
        scalar_keys=["aspect"],
    ) == ("aspect",)

    replay_setup = branch_local_helpers.prepare_branch_local_replay_setup(
        init=init,
        traces=(trace,),
        replay_kwargs={"use_accepted_only_fast_path": False, "nestor_operator_tol": 1.0e-9},
        replay_payload={"slim": True},
        payload=payload,
        replay_plan={"plan": "cached"},
        use_replay_plan=True,
        include_replay_graph_metadata=False,
        timings=branch_payload.timings,
    )
    assert replay_setup.replay_payload == {"slim": True}
    assert replay_setup.replay_payload_source == "user"
    assert replay_setup.replay_plan == {"plan": "cached"}
    assert replay_setup.graph_metadata["omitted"]
    assert replay_setup.replay_branch_metadata["n_steps"] == 1

    flags = branch_local_helpers.branch_local_replay_option_flags(
        replay_setup.replay_options,
        replay_plan=replay_setup.replay_plan,
        ad_mode="direct",
        extra={"directional_jvp_fast_path": "none"},
    )
    assert flags["use_replay_plan"]
    assert not flags["use_accepted_only_fast_path"]
    assert flags["nestor_operator_tol"] == 1.0e-9
    assert flags["directional_jvp_fast_path"] == "none"


@pytest.mark.py311_coverage_only
def test_free_boundary_adjoint_timing_and_dense_helper_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover asset-free fallback helpers used by branch-local adjoint reports."""

    real_jax = fba.jax

    monkeypatch.setattr(fba, "jax", None)
    assert fba._block_until_ready_for_timing({"value": 1.0}) == {"value": 1.0}
    with fba._jax_named_scope("no_jax"):
        pass

    class SelectiveReady:
        @staticmethod
        def block_until_ready(value):
            if isinstance(value, dict):
                raise TypeError("top-level container unsupported")
            return value

    monkeypatch.setattr(fba, "jax", SelectiveReady())
    ready = fba._block_until_ready_for_timing({"value": np.asarray([1.0, 2.0])})
    np.testing.assert_allclose(ready["value"], np.asarray([1.0, 2.0]))

    class NoNamedScope:
        @staticmethod
        def block_until_ready(value):
            return value

    monkeypatch.setattr(fba, "jax", NoNamedScope())
    with fba._jax_named_scope("missing_named_scope"):
        pass
    monkeypatch.setattr(fba, "jax", real_jax)

    jac = fba._finite_difference_jacobian(
        lambda x: jnp.asarray([x[0] + x[1], x[0] * x[1]]),
        jnp.asarray([1.0, 2.0]),
        eps=1.0e-6,
    )
    np.testing.assert_allclose(np.asarray(jac), np.asarray([[1.0, 1.0], [2.0, 1.0]]), rtol=1.0e-6, atol=1.0e-6)

    with pytest.raises(ValueError, match="same shape"):
        fba.dense_fixed_point_solve_jax(
            lambda state, _params: jnp.asarray(state)[:1],
            jnp.ones(2),
            params=None,
            max_iter=1,
        )


@pytest.mark.py311_coverage_only
def test_free_boundary_adjoint_trace_stackability_error_paths() -> None:
    """Cover trace stackability guards without running a free-boundary solve."""

    assert fba.direct_coil_accepted_trace_status_masks is trace_controls.direct_coil_accepted_trace_status_masks
    assert (
        fba.direct_coil_accepted_trace_controller_controls_jax
        is trace_controls.direct_coil_accepted_trace_controller_controls_jax
    )
    assert fba._accepted_trace_effective_controller_masks is trace_controls.accepted_trace_effective_controller_masks
    assert (
        fba._accepted_trace_segment_is_unconditionally_accepted
        is trace_controls.accepted_trace_segment_is_unconditionally_accepted
    )
    assert fba._unique_shape_list is trace_metadata.unique_shape_list
    assert fba._compact_segment_summaries is trace_metadata.compact_segment_summaries
    assert fba._json_safe_fingerprint_value is trace_metadata.json_safe_fingerprint_value
    assert fba._fingerprint_has_rejected_controller_slot is trace_metadata.fingerprint_has_rejected_controller_slot
    assert (
        fba.direct_coil_accepted_trace_controller_slot_summary
        is trace_metadata.direct_coil_accepted_trace_controller_slot_summary
    )
    assert fba._trace_scalar is trace_fingerprint.trace_scalar
    assert fba._trace_bool is trace_fingerprint.trace_bool
    assert fba._trace_pack_size is trace_fingerprint.trace_pack_size
    assert fba._trace_array_size is trace_fingerprint.trace_array_size
    assert fba._trace_pytree_shape_signature is trace_fingerprint.trace_pytree_shape_signature
    assert (
        fba.direct_coil_accepted_trace_fingerprint
        is trace_fingerprint.direct_coil_accepted_trace_fingerprint
    )
    assert (
        fba.direct_coil_accepted_trace_fingerprint_delta
        is trace_fingerprint.direct_coil_accepted_trace_fingerprint_delta
    )
    assert (
        fba.direct_coil_accepted_trace_fingerprint_delta_summary
        is trace_fingerprint.direct_coil_accepted_trace_fingerprint_delta_summary
    )
    assert fba._extract_adjoint_step_trace is replay_plan_helpers.extract_adjoint_step_trace
    assert fba._slice_replay_controls is replay_plan_helpers.slice_replay_controls
    assert fba._stackability_probe is replay_plan_helpers.stackability_probe
    assert fba._weighted_half_norm is objective_helpers.weighted_half_norm
    assert fba._static_weight_is_zero is objective_helpers.static_weight_is_zero
    assert fba._tree_weighted_half_norm is objective_helpers.tree_weighted_half_norm
    assert fba._pytree_batched_directional_vdot_jax is pytree_helpers.pytree_batched_directional_vdot_jax
    assert fba._pytree_pullback_basis_jax is pytree_helpers.pytree_pullback_basis_jax
    assert fba._pytree_unstack_leading_axis_jax is pytree_helpers.pytree_unstack_leading_axis_jax
    assert (
        fba._accepted_step_policy_signature_for_complete_payload
        is replay_plan_helpers.accepted_step_policy_signature_for_complete_payload
    )
    assert (
        fba._accepted_step_policy_layout_for_complete_payload
        is replay_plan_helpers.accepted_step_policy_layout_for_complete_payload
    )
    assert (
        fba._accepted_step_policy_summary_for_complete_payload
        is replay_plan_helpers.accepted_step_policy_summary_for_complete_payload
    )
    assert fba._complete_solve_objective_values is replay_plan_helpers.complete_solve_objective_values
    assert fba._stack_trace_control_field is trace_stack.stack_trace_control_field
    assert fba._stack_trace_pytree_field is trace_stack.stack_trace_pytree_field
    assert fba._stack_optional_trace_pytree_field is trace_stack.stack_optional_trace_pytree_field
    assert fba._trace_preconditioner_policy_value is trace_stack.trace_preconditioner_policy_value
    assert fba._trace_preconditioner_static_signature is trace_stack.trace_preconditioner_static_signature
    assert fba._trace_static_value_shape_signature is trace_stack.trace_static_value_shape_signature
    assert fba._trace_optional_presence_signature is trace_stack.trace_optional_presence_signature
    assert fba._stack_trace_nestor_axis_controls is trace_stack.stack_trace_nestor_axis_controls
    assert fba._trace_step_policy_static_signature is trace_stack.trace_step_policy_static_signature
    assert (
        fba.direct_coil_accepted_trace_step_policy_segments
        is trace_stack.direct_coil_accepted_trace_step_policy_segments
    )
    assert (
        fba.direct_coil_accepted_trace_step_policy_segment_summary
        is trace_stack.direct_coil_accepted_trace_step_policy_segment_summary
    )
    assert (
        fba.direct_coil_accepted_trace_preconditioner_policy_segments
        is trace_stack.direct_coil_accepted_trace_preconditioner_policy_segments
    )
    assert (
        fba.direct_coil_accepted_trace_preconditioner_policy_segment_summary
        is trace_stack.direct_coil_accepted_trace_preconditioner_policy_segment_summary
    )
    assert fba.direct_coil_accepted_trace_scalar_controls_jax is trace_stack.direct_coil_accepted_trace_scalar_controls_jax
    assert fba.direct_coil_accepted_trace_array_controls_jax is trace_stack.direct_coil_accepted_trace_array_controls_jax
    assert (
        fba.direct_coil_accepted_trace_preconditioner_controls_jax
        is trace_stack.direct_coil_accepted_trace_preconditioner_controls_jax
    )
    assert fba.direct_coil_accepted_trace_step_controls_jax is trace_stack.direct_coil_accepted_trace_step_controls_jax
    assert trace_metadata.unique_shape_list([(2, 3), (2, 3), (1,)]) == [[2, 3], [1]]
    assert trace_metadata.compact_segment_summaries(
        [{"count": 1, "signature_repr": "large"}, {"count": 2, "tag": "kept"}]
    ) == [{"count": 1}, {"count": 2, "tag": "kept"}]
    json_safe = trace_metadata.json_safe_fingerprint_value(
        {
            "array": np.asarray([1.0, np.inf]),
            "scalar": np.float64(2.0),
            "bad": float("nan"),
            3: (np.asarray(4),),
        }
    )
    assert json_safe == {"array": [1.0, None], "scalar": 2.0, "bad": None, "3": [4]}

    class BadToList:
        def tolist(self):
            raise RuntimeError("synthetic conversion failure")

    bad_to_list = BadToList()
    assert trace_metadata.json_safe_fingerprint_value(bad_to_list) is bad_to_list
    assert trace_metadata.json_safe_fingerprint_value(b"bytes") == b"bytes"
    slot_summary = trace_metadata.direct_coil_accepted_trace_controller_slot_summary(
        {
            "n_steps": 3,
            "masks": {
                "accepted": [True, False, False],
                "rejected": [False, True, False],
                "done": [False, False, True],
                "active": [True, True, False],
                "has_active_freeb_replay": [True, True, False],
            },
        }
    )
    assert slot_summary == {
        "n_steps": 3,
        "active_slots": 2,
        "accepted_slots": 1,
        "rejected_slots": 1,
        "done_markers": 1,
        "active_free_boundary_slots": 2,
        "accepted_free_boundary_slots": 1,
        "fixed_rejected_controller_slot_present": True,
    }
    assert trace_metadata.fingerprint_has_rejected_controller_slot({"accept_mask": np.asarray([1, 0])})
    assert trace_metadata.fingerprint_has_rejected_controller_slot({"step_status": ("restart_bad_jacobian",)})
    assert trace_metadata.fingerprint_has_rejected_controller_slot({"step_status": ("rejected",)})
    assert not trace_metadata.fingerprint_has_rejected_controller_slot({"accept_mask": np.asarray([1, 1])})
    assert not trace_metadata.fingerprint_has_rejected_controller_slot("not-a-fingerprint")
    assert replay_plan_helpers.complete_solve_objective_values(2.5) == {"objective": 2.5}
    assert replay_plan_helpers.complete_solve_objective_values({"a": np.asarray([1.5]), 2: 3.0}) == {
        "a": 1.5,
        "2": 3.0,
    }
    with pytest.raises(ValueError, match="empty mapping"):
        replay_plan_helpers.complete_solve_objective_values({})
    with pytest.raises(ValueError, match="mapping entry 'a' must be scalar"):
        replay_plan_helpers.complete_solve_objective_values({"a": np.asarray([1.0, 2.0])})
    with pytest.raises(ValueError, match="scalar or a mapping"):
        replay_plan_helpers.complete_solve_objective_values(np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(
        np.asarray(objective_helpers.weighted_half_norm(np.asarray([1.0, 2.0]), 2.0)),
        5.0,
    )
    np.testing.assert_allclose(
        np.asarray(objective_helpers.tree_weighted_half_norm({"a": np.asarray([1.0, 2.0]), "b": np.asarray([3.0])}, 1.0)),
        7.0,
    )
    np.testing.assert_allclose(np.asarray(objective_helpers.tree_weighted_half_norm({}, 1.0)), 0.0)
    assert objective_helpers.static_weight_is_zero(np.zeros(2))
    assert not objective_helpers.static_weight_is_zero(np.asarray([0.0, 1.0]))
    assert not objective_helpers.static_weight_is_zero(np.asarray([]))
    replay_result = objective_helpers.accepted_controller_replay_result(
        run={
            "state": "state-object",
            "history": {
                "accepted": jnp.asarray([1.0, 0.0, 1.0]),
                "force": jnp.asarray([2.0, 100.0, 4.0]),
                "bsqvac": jnp.asarray([0.5, 100.0, 1.5]),
            },
        },
        controls={
            "has_active_freeb_replay": jnp.asarray([True, False, True]),
            "reset_to_trace_pre": np.asarray([False, True, False]),
        },
        scalar_controls={"dt": "scalar"},
        array_controls={"v": "array"},
        step_controls={"state_pre": "step"},
        preconditioner_controls={"mats": "precond"},
        preconditioner_controls_stacked=True,
        preconditioner_policy_segments=({"start": 0, "stop": 3},),
        preconditioner_policy_segment_summary={"n": 1},
        step_policy_segments=({"start": 0, "stop": 3},),
        step_policy_segment_summary={"n": 1},
        segment_preconditioner_controls_stacked=(True,),
        use_preconditioner_policy_segments=True,
        use_stacked_step_controls=False,
        accepted_only_fast_path_segments=(True,),
        state_weight=0.0,
        include_replay_aux=True,
        state_only_replay=False,
    )
    np.testing.assert_allclose(np.asarray(replay_result["objective_components"]["state"]), 0.0)
    np.testing.assert_allclose(np.asarray(replay_result["objective_components"]["force"]), 6.0)
    np.testing.assert_allclose(np.asarray(replay_result["objective_components"]["bsqvac"]), 2.0)
    np.testing.assert_allclose(np.asarray(replay_result["objective"]), 8.0)
    assert replay_result["used_preconditioner_policy_segments"] is True
    assert replay_result["used_accepted_only_fast_path"] is True
    assert replay_result["state_reset_flags"] == (False, True, False)
    stripped_replay_result = objective_helpers.accepted_controller_replay_result(
        run=replay_result,
        controls=replay_result["controls"],
        scalar_controls={},
        array_controls={},
        step_controls={},
        preconditioner_controls={},
        preconditioner_controls_stacked=False,
        preconditioner_policy_segments=(),
        preconditioner_policy_segment_summary={},
        step_policy_segments=(),
        step_policy_segment_summary={},
        segment_preconditioner_controls_stacked=(),
        use_preconditioner_policy_segments=False,
        use_stacked_step_controls=False,
        accepted_only_fast_path_segments=(),
        state_weight=0.0,
        include_replay_aux=False,
        state_only_replay=True,
    )
    assert set(stripped_replay_result["controls"]) == {"has_active_freeb_replay"}
    np.testing.assert_allclose(np.asarray(stripped_replay_result["objective_components"]["force"]), 0.0)

    class BadArray:
        def __array__(self, _dtype=None):
            raise TypeError("synthetic bad array")

    assert not objective_helpers.static_weight_is_zero(BadArray())
    jacobian_tree = {"x": jnp.asarray([[1.0, 2.0], [3.0, 4.0]])}
    direction_tree = {"x": jnp.asarray([10.0, -2.0])}
    np.testing.assert_allclose(
        np.asarray(pytree_helpers.pytree_batched_directional_vdot_jax(jacobian_tree, direction_tree, 2)),
        np.asarray([6.0, 22.0]),
    )
    np.testing.assert_allclose(
        np.asarray(pytree_helpers.pytree_batched_directional_vdot_jax({}, {}, 3)),
        np.zeros(3),
    )
    unstacked = pytree_helpers.pytree_unstack_leading_axis_jax({"x": jnp.asarray([[1.0], [2.0]])}, 2)
    np.testing.assert_allclose(np.asarray(unstacked[0]["x"]), np.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(unstacked[1]["x"]), np.asarray([2.0]))
    assert fba._accepted_trace_reset_flags([]) == ()
    assert fba._accepted_trace_reset_flags([{}, {}]) == (False, False)
    trace0 = {
        "state_pre": np.asarray([0.0, 0.0]),
        "state_post": np.asarray([1.0, 1.0]),
    }
    trace1_continuous = {
        "state_pre": np.asarray([1.0, 1.0]),
        "state_post": np.asarray([2.0, 2.0]),
    }
    trace1_reset = {
        "state_pre": np.asarray([9.0, 9.0]),
        "state_post": np.asarray([2.0, 2.0]),
    }
    continuous = trace_fingerprint.direct_coil_accepted_trace_fingerprint([trace0, trace1_continuous])
    reset = trace_fingerprint.direct_coil_accepted_trace_fingerprint([trace0, trace1_reset])
    np.testing.assert_array_equal(continuous["state_reset_flags"], np.asarray([0]))
    np.testing.assert_array_equal(reset["state_reset_flags"], np.asarray([1]))
    reset_delta = trace_fingerprint.direct_coil_accepted_trace_fingerprint_delta(
        [trace0, trace1_continuous],
        [trace0, trace1_reset],
    )
    assert not reset_delta["compatible"]
    assert "state_reset_flags" in reset_delta["changed_fields"]

    with pytest.raises(ValueError, match="at least one"):
        fba._stack_trace_control_field((), "dt_eff")
    with pytest.raises(KeyError, match="missing control field"):
        fba._stack_trace_control_field(({},), "dt_eff")
    with pytest.raises(ValueError, match="consistent shape"):
        fba._stack_trace_control_field(
            ({"dt_eff": np.asarray([1.0])}, {"dt_eff": np.asarray([1.0, 2.0])}),
            "dt_eff",
        )

    with pytest.raises(ValueError, match="at least one"):
        fba._stack_trace_pytree_field((), "precond_mats")
    with pytest.raises(KeyError, match="missing control field"):
        fba._stack_trace_pytree_field(({},), "precond_mats")
    with pytest.raises(ValueError, match="inconsistent structure"):
        fba._stack_trace_pytree_field(
            ({"precond_mats": {"a": np.asarray([1.0])}}, {"precond_mats": {"b": np.asarray([1.0])}}),
            "precond_mats",
        )
    with pytest.raises(ValueError, match="consistent leaf shapes"):
        fba._stack_trace_pytree_field(
            (
                {"precond_mats": {"a": np.asarray([1.0])}},
                {"precond_mats": {"a": np.asarray([1.0, 2.0])}},
            ),
            "precond_mats",
        )

    assert fba._trace_preconditioner_policy_value({}, "preconditioner_use_lax_tridi") == -1
    assert fba._trace_preconditioner_policy_value(
        {"preconditioner_use_lax_tridi": np.asarray([])},
        "preconditioner_use_lax_tridi",
    ) == -1
    assert fba._trace_preconditioner_policy_value(
        {"preconditioner_use_lax_tridi": np.asarray([True])},
        "preconditioner_use_lax_tridi",
    ) == 1


@pytest.mark.py311_coverage_only
def test_free_boundary_trace_fingerprint_fallback_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover branch-fingerprint fallback paths without a free-boundary solve."""

    assert trace_fingerprint.trace_scalar({"value": None}, "value", default=3.0) == 3.0
    assert trace_fingerprint.trace_scalar({"value": np.asarray([])}, "value", default=4.0) == 4.0
    assert trace_fingerprint.trace_bool({"flag": None}, "flag") == 0
    assert trace_fingerprint.trace_bool({"flag": np.asarray([])}, "flag") == 0
    assert trace_fingerprint.trace_array_size(None) == 0
    assert trace_fingerprint.trace_pack_size(None) == 0

    real_tree_util = trace_fingerprint.tree_util

    class RaisingTreeUtil:
        @staticmethod
        def tree_leaves(_value):
            raise TypeError("synthetic non-pytree")

    monkeypatch.setattr(trace_fingerprint, "tree_util", RaisingTreeUtil())
    assert trace_fingerprint.trace_pytree_shape_signature(np.ones((2, 3))) == ((2, 3),)
    monkeypatch.setattr(trace_fingerprint, "tree_util", real_tree_util)

    empty = trace_fingerprint.direct_coil_accepted_trace_fingerprint([])
    assert empty["n_steps"] == 0
    assert empty["step_status"] == ()
    np.testing.assert_array_equal(empty["accept_mask"], np.asarray([], dtype=int))
    np.testing.assert_array_equal(empty["done_mask"], np.asarray([], dtype=int))

    trace0 = {
        "dt_eff": np.asarray(0.5),
        "fac": np.asarray(1.0),
        "flip_sign": False,
        "freeb_bsqvac_half": np.ones((2, 2)),
        "state_pre": np.asarray([0.0, 0.0]),
        "state_post": np.asarray([1.0, 1.0]),
    }
    trace1 = {
        **trace0,
        "dt_eff": np.asarray(0.25),
        "fac": np.asarray(0.75),
        "flip_sign": True,
        "freeb_bsqvac_half": np.ones((3, 2)),
        "state_pre": np.asarray([1.0, 1.0]),
        "state_post": np.asarray([2.0, 2.0]),
    }
    truncated = trace_fingerprint.direct_coil_accepted_trace_fingerprint([trace0, trace1], max_steps=1)
    assert truncated["n_steps"] == 1
    np.testing.assert_array_equal(truncated["freeb_sizes"], np.asarray([4]))

    changed = trace_fingerprint.direct_coil_accepted_trace_fingerprint_delta([trace0], [trace0, trace1])
    assert not changed["compatible"]
    assert "n_steps" in changed["changed_fields"]
    assert "freeb_sizes" in changed["changed_fields"]
    assert "flags.flip_sign" in changed["changed_fields"]
    assert "scalars.dt_eff" in changed["changed_fields"]

    scalar_changed_trace0 = {**trace0, "dt_eff": np.asarray(0.75)}
    scalar_changed = trace_fingerprint.direct_coil_accepted_trace_fingerprint_delta(
        [trace0],
        [scalar_changed_trace0],
    )
    assert not scalar_changed["compatible"]
    assert "scalars.dt_eff" in scalar_changed["changed_fields"]
    assert scalar_changed["max_abs_scalar_delta"] > 0.0
    assert scalar_changed["max_rel_scalar_delta"] > 0.0

    json_changed = trace_fingerprint.direct_coil_accepted_trace_fingerprint_delta_summary(
        [trace0, trace1],
        [trace0, trace1],
        max_steps=1,
    )
    assert json_changed["compatible"]
    assert json_changed["reference"]["n_steps"] == 1

    bad_state = object()
    bad_reset = trace_fingerprint.direct_coil_accepted_trace_fingerprint(
        [
            {"state_post": bad_state},
            {"state_pre": bad_state},
        ]
    )
    np.testing.assert_array_equal(bad_reset["state_reset_flags"], np.asarray([0]))


@pytest.mark.py311_coverage_only
def test_free_boundary_adjoint_trace_signature_and_extraction_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover static-signature and trace-extraction branches used by gates."""

    object_signature = fba._trace_static_value_shape_signature(np.asarray([{"label": "branch"}], dtype=object))
    assert object_signature[0][0] == (1,)
    assert object_signature[0][1] == "object"

    real_tree_util = fba.tree_util

    class RaisingTreeUtil:
        @staticmethod
        def tree_leaves(_value):
            raise TypeError("not a pytree")

    monkeypatch.setattr(fba, "tree_util", RaisingTreeUtil())
    fallback_signature = fba._trace_static_value_shape_signature("raw")
    assert fallback_signature[0][0] == ()
    monkeypatch.setattr(fba, "tree_util", real_tree_util)

    trace = {"step": 1}
    assert fba._extract_adjoint_step_trace({"adjoint_step_trace": [trace]}) == (trace,)
    assert fba._extract_adjoint_step_trace({"diagnostics": {"adjoint_step_trace": [trace]}}) == (trace,)
    assert fba._extract_adjoint_step_trace(SimpleNamespace(diagnostics={"adjoint_step_trace": [trace]})) == (trace,)
    assert fba._extract_adjoint_step_trace(
        SimpleNamespace(result=SimpleNamespace(diagnostics={"adjoint_step_trace": [trace]}))
    ) == (trace,)
    assert fba._extract_adjoint_step_trace([trace]) == (trace,)

    with pytest.raises(RuntimeError, match="No adjoint_step_trace"):
        fba._extract_adjoint_step_trace("missing")
    with pytest.raises(RuntimeError, match="No adjoint_step_trace"):
        fba._extract_adjoint_step_trace(object())
    with pytest.raises(RuntimeError, match="No adjoint_step_trace"):
        fba._extract_adjoint_step_trace([1, 2])


@pytest.mark.py311_coverage_only
def test_free_boundary_adjoint_segment_fast_path_guards() -> None:
    """Cover accepted-segment fast-path guards for active/rejected/done masks."""

    masks = {
        "active": np.asarray([True, True]),
        "accepted": np.asarray([True, True]),
        "rejected": np.asarray([False, False]),
        "done": np.asarray([False, True]),
    }
    assert fba._accepted_trace_segment_is_unconditionally_accepted(masks, start=0, stop=2)
    assert not fba._accepted_trace_segment_is_unconditionally_accepted(masks, start=0, stop=0)

    inactive = {**masks, "active": np.asarray([True, False])}
    assert not fba._accepted_trace_segment_is_unconditionally_accepted(inactive, start=0, stop=2)

    rejected = {**masks, "rejected": np.asarray([False, True])}
    assert not fba._accepted_trace_segment_is_unconditionally_accepted(rejected, start=0, stop=2)

    early_done = {**masks, "done": np.asarray([True, False])}
    assert not fba._accepted_trace_segment_is_unconditionally_accepted(early_done, start=0, stop=2)


@pytest.mark.py311_coverage_only
def test_direct_coil_trace_shape_and_frozen_vacuum_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover trace-shape inference and the frozen-vacuum wrapper without a solve."""

    shape = (2, 3)
    assert fba._direct_coil_trace_boundary_shape(
        {"freeb_nestor_trace": {"br_axis": np.ones(shape)}}
    ) == shape
    assert fba._direct_coil_trace_boundary_shape({"freeb_bsqvac_half": np.ones(shape)}) == shape
    assert fba._direct_coil_trace_boundary_shape({"freeb_nestor_trace": {"br_axis": np.ones((2, 3, 4))}}) is None

    calls: list[dict[str, object]] = []

    def fake_dense_vmec_nestor_mode_solve_jax(**kwargs):
        calls.append(
            {
                "bexni": kwargs["bexni"],
                "include_phi_flat": kwargs["include_phi_flat"],
                "include_residual": kwargs["include_residual"],
            }
        )
        return {"mode_coeffs": jnp.zeros_like(kwargs["bexni"])}

    def fake_vacuum_boundary_fields_from_mode_coeffs_jax(_mode_coeffs, *, bu_ext, bv_ext, g_uu, g_uv, g_vv, basis):
        del basis
        return {"bsqvac": jnp.asarray(bu_ext) - jnp.asarray(bv_ext) + jnp.asarray(g_uu) + jnp.asarray(g_uv) + jnp.asarray(g_vv)}

    monkeypatch.setattr(
        direct_coil_replay_helpers,
        "dense_vmec_nestor_mode_solve_jax",
        fake_dense_vmec_nestor_mode_solve_jax,
    )
    monkeypatch.setattr(
        direct_coil_replay_helpers,
        "vacuum_boundary_fields_from_mode_coeffs_jax",
        fake_vacuum_boundary_fields_from_mode_coeffs_jax,
    )

    grid = jnp.ones(shape)
    geometry = {
        "R": grid,
        "Z": 2.0 * grid,
        "phi": 3.0 * grid,
        "Ru": grid,
        "Zu": grid,
        "Rv": grid,
        "Zv": grid,
        "ruu": grid,
        "ruv": grid,
        "rvv": grid,
        "zuu": grid,
        "zuv": grid,
        "zvv": grid,
    }
    trace = {
        "freeb_nestor_trace": {
            "bnormal": 2.0 * grid,
            "g_uu": grid,
            "g_uv": 2.0 * grid,
            "g_vv": 3.0 * grid,
            "bu": 4.0 * grid,
            "bv": 5.0 * grid,
            "br_axis": 6.0 * grid,
            "bp_axis": 7.0 * grid,
            "bz_axis": 8.0 * grid,
        }
    }
    out = fba.direct_coil_boundary_bsqvac_from_trace_jax(
        params=None,
        geometry=geometry,
        trace=trace,
        basis={},
        tables={},
        signgs=1,
        nvper=1,
        wint=0.25 * grid,
        include_diagnostics=False,
        include_mode_diagnostics=False,
        freeze_vacuum_field=True,
    )
    np.testing.assert_allclose(np.asarray(out["bsqvac"]), np.asarray(5.0 * grid))
    np.testing.assert_allclose(np.asarray(calls[-1]["bexni"]).reshape(shape), np.asarray(-0.5 * grid * ((2.0 * np.pi) ** 2)))
    assert calls[-1]["include_phi_flat"] is False
    assert calls[-1]["include_residual"] is False

    with pytest.raises(ValueError, match="NESTOR trace"):
        fba.direct_coil_boundary_bsqvac_from_trace_jax(
            params=None,
            geometry=geometry,
            trace={"freeb_nestor_trace": object()},
            basis={},
            tables={},
            signgs=1,
            nvper=1,
            wint=grid,
        )


@pytest.mark.py311_coverage_only
def test_accepted_trace_control_metadata_and_stack_contracts() -> None:
    """Cover fixed-branch controller metadata without a VMEC solve."""

    def trace(step: int, *, active_freeb: bool = False, precond_jmax: int = 2) -> dict[str, object]:
        scalar = np.asarray(1.0 + step)
        vector = np.asarray([1.0 + step, 2.0 + step])
        payload: dict[str, object] = {
            "dt_eff": scalar,
            "b1": scalar,
            "fac": scalar,
            "force_scale": scalar,
            "max_update_rms_pre": scalar,
            "lambda_update_scale": scalar,
            "flip_sign": np.asarray(False),
            "limit_update_rms": np.asarray(True),
            "divide_by_scalxc_for_update": np.asarray(False),
            "preconditioner_use_precomputed_tridi": np.asarray(precond_jmax > 0),
            "preconditioner_use_lax_tridi": np.asarray(False),
            "precond_jmax": precond_jmax,
            "precond_mats": {"main": vector},
            "lam_prec": vector,
            "w_mode_mn": vector,
            "state_pre": {"x": vector},
            "force_state_pre": {"x": 0.5 * vector},
            "freeb_pres_scale": np.asarray([0.25 + step]),
            "constraint_tcon": np.asarray([0.0 + step]),
            "wout_like": {"tag": np.asarray([step])},
            "trig": {"tag": np.asarray([step])},
            "zero_m1": np.asarray([False]),
        }
        for key in fba._ACCEPTED_TRACE_REQUIRED_ARRAY_CONTROL_KEYS:
            payload[key] = vector
        for key in fba._ACCEPTED_TRACE_OPTIONAL_ARRAY_CONTROL_KEYS:
            payload[key] = vector
        if active_freeb:
            axis = np.ones((2, 2)) * (1.0 + step)
            payload["freeb_bsqvac_half"] = axis
            payload["freeb_nestor_trace"] = {
                "br_axis": axis,
                "bp_axis": 2.0 * axis,
                "bz_axis": 3.0 * axis,
            }
        return payload

    traces = (trace(0, active_freeb=True), trace(1, active_freeb=False))

    with pytest.raises(ValueError, match="accept_mask"):
        fba.direct_coil_accepted_trace_controller_controls_jax(traces, accept_mask=np.asarray([True]))
    with pytest.raises(ValueError, match="done_mask"):
        fba.direct_coil_accepted_trace_controller_controls_jax(traces, done_mask=np.asarray([False]))

    controls = fba.direct_coil_accepted_trace_controller_controls_jax(
        traces,
        accept_mask=np.asarray([True, False]),
        done_mask=np.asarray([False, True]),
    )
    masks = fba._accepted_trace_effective_controller_masks(controls)
    assert np.asarray(masks["accepted"]).tolist() == [True, False]
    assert np.asarray(masks["rejected"]).tolist() == [False, True]

    scalar_controls = fba.direct_coil_accepted_trace_scalar_controls_jax(traces)
    array_controls = fba.direct_coil_accepted_trace_array_controls_jax(traces)
    preconditioner_controls = fba.direct_coil_accepted_trace_preconditioner_controls_jax(traces)
    step_controls = fba.direct_coil_accepted_trace_step_controls_jax(traces)
    assert scalar_controls["dt_eff"].shape == (2,)
    assert array_controls["vRcc_before"].shape == (2, 2)
    assert preconditioner_controls["precond_mats"]["main"].shape == (2, 2)
    assert step_controls["freeb_nestor_axes"]["br_axis"].shape == (2, 2, 2)
    assert step_controls["force_state_pre"]["x"].shape == (2, 2)

    missing_optional = (trace(0), trace(1))
    assert fba._stack_optional_trace_pytree_field(missing_optional, "freeb_pres_scale") is not None
    partial_optional = (trace(0), {**trace(1), "freeb_pres_scale": None})
    assert fba._stack_optional_trace_pytree_field(partial_optional, "freeb_pres_scale") is None
    inconsistent_optional = (trace(0), {**trace(1), "freeb_pres_scale": np.ones(2)})
    with pytest.raises(ValueError, match="optional field"):
        fba._stack_optional_trace_pytree_field(inconsistent_optional, "freeb_pres_scale")

    assert fba._stack_trace_nestor_axis_controls(missing_optional) is None
    bad_axis = (trace(0, active_freeb=True), trace(1, active_freeb=True))
    bad_axis[1]["freeb_nestor_trace"] = {"br_axis": np.ones((3, 2)), "bp_axis": np.ones((2, 2)), "bz_axis": np.ones((2, 2))}
    with pytest.raises(ValueError, match="NESTOR axis field"):
        fba._stack_trace_nestor_axis_controls(bad_axis)

    metadata = fba.direct_coil_accepted_trace_branch_metadata(
        traces,
        accept_mask=np.asarray([True, False]),
        done_mask=np.asarray([False, True]),
        json_safe=True,
    )
    assert metadata["n_steps"] == 2
    assert metadata["n_free_boundary_replay_steps"] == 1
    graph = fba.direct_coil_accepted_trace_replay_graph_metadata(
        traces,
        accept_mask=np.asarray([True, False]),
        done_mask=np.asarray([False, True]),
        json_safe=True,
    )
    assert graph["active_free_boundary_replay_steps"] == 1
    assert graph["inferred_boundary_shape"] == [2, 2]
    sliced = replay_plan_helpers.slice_replay_controls(
        {"a": np.asarray([1, 2, 3]), "nested": {"b": np.asarray([[1], [2], [3]])}},
        start=1,
        stop=3,
    )
    np.testing.assert_array_equal(np.asarray(sliced["a"]), np.asarray([2, 3]))
    np.testing.assert_array_equal(np.asarray(sliced["nested"]["b"]), np.asarray([[2], [3]]))

    plan = fba.direct_coil_accepted_trace_controller_replay_plan(
        missing_optional,
        static=SimpleNamespace(),
        use_preconditioner_policy_segments=True,
        use_segment_preconditioner_controls=True,
        use_stacked_step_controls=False,
    )
    assert plan["segment_source"] == "preconditioner_policy"
    assert plan["preconditioner_controls_stacked"]
    inherited_context = {"sentinel": object()}
    inherited_plan = fba.direct_coil_accepted_trace_controller_replay_plan(
        traces,
        static=SimpleNamespace(),
        use_preconditioner_policy_segments=True,
        boundary_replay_contexts_by_shape={(2, 2): inherited_context},
    )
    assert inherited_plan["boundary_replay_contexts_by_shape"][(2, 2)] is inherited_context

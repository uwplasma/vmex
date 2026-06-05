from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.free_boundary_adjoint as fba
from vmec_jax._compat import jnp


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

    assert fba._accepted_trace_reset_flags([]) == ()
    assert fba._accepted_trace_reset_flags([{}, {}]) == (False, False)

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

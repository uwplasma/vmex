from __future__ import annotations

from collections import OrderedDict

from vmec_jax.solve_jit_cache_helpers import jit_cache_get, jit_cache_put
from vmec_jax.solvers.fixed_boundary.residual.force_cache import (
    compute_forces_jit_cache_key,
    select_compute_forces_callable,
)


def test_compute_forces_jit_cache_key_is_structural() -> None:
    key = compute_forces_jit_cache_key(
        static_key=("static", 1),
        wout_key=("wout", "float64"),
        signgs=-1,
        apply_m1_constraints=True,
    )

    assert key == ("compute_forces_v1", ("static", 1), ("wout", "float64"), -1, True)


def test_select_compute_forces_callable_uses_cache_for_primal_solves(monkeypatch) -> None:
    monkeypatch.setenv("VMEC_JAX_TEST_COMPUTE_FORCES_CACHE", "1")
    cache: OrderedDict[tuple, object] = OrderedDict()
    jit_calls = []

    def fake_jit(fn, *, static_argnames):
        jit_calls.append((fn, static_argnames))
        return {"fn": fn, "static_argnames": static_argnames}

    fn_a = lambda **_kwargs: "a"
    fn_b = lambda **_kwargs: "b"
    key_a = ("a",)
    key_b = ("b",)

    first = select_compute_forces_callable(
        fn_a,
        differentiating_scan=False,
        cache=cache,
        cache_key=key_a,
        jit_func=fake_jit,
        cache_get=jit_cache_get,
        cache_put=jit_cache_put,
        cache_env_name="VMEC_JAX_TEST_COMPUTE_FORCES_CACHE",
        cache_default=32,
    )
    second = select_compute_forces_callable(
        fn_a,
        differentiating_scan=False,
        cache=cache,
        cache_key=key_a,
        jit_func=fake_jit,
        cache_get=jit_cache_get,
        cache_put=jit_cache_put,
        cache_env_name="VMEC_JAX_TEST_COMPUTE_FORCES_CACHE",
        cache_default=32,
    )
    third = select_compute_forces_callable(
        fn_b,
        differentiating_scan=False,
        cache=cache,
        cache_key=key_b,
        jit_func=fake_jit,
        cache_get=jit_cache_get,
        cache_put=jit_cache_put,
        cache_env_name="VMEC_JAX_TEST_COMPUTE_FORCES_CACHE",
        cache_default=32,
    )

    assert first is second
    assert third is not first
    assert len(jit_calls) == 2
    assert list(cache) == [key_b]
    assert jit_calls[0][1] == ("include_edge", "include_edge_residual")


def test_select_compute_forces_callable_does_not_store_differentiating_scans() -> None:
    cache: OrderedDict[tuple, object] = OrderedDict()
    jit_calls = []

    def fake_jit(fn, *, static_argnames):
        jit_calls.append(static_argnames)
        return ("jitted", fn)

    got = select_compute_forces_callable(
        lambda **_kwargs: "value",
        differentiating_scan=True,
        cache=cache,
        cache_key=("trace",),
        jit_func=fake_jit,
        cache_get=jit_cache_get,
        cache_put=jit_cache_put,
    )

    assert got[0] == "jitted"
    assert jit_calls == [("include_edge", "include_edge_residual")]
    assert cache == {}

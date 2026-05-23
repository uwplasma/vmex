from dataclasses import dataclass
import gc
import numpy as np

import vmec_jax._compat as compat
import vmec_jax.solve as solve_mod
import vmec_jax.solve_force_payload_helpers as payload_helpers_mod
import vmec_jax.vmec_forces as vf
from vmec_jax.vmec_numpy_forces import (
    _NP_STACK_CACHE,
    _NP_STACK_CACHE_LIMIT,
    _NP_STACK_CACHE_MAX_BYTES,
    _NP_MODULE,
    _NumpyJaxShim,
    _NumpyLaxShim,
    _NpArray,
    _NpFftModule,
    _NpModule,
    _idx_has_fancy,
    _numpy_module_patch,
    _np_einsum,
    _prune_np_stack_cache,
    _to_numpy_recursive,
    clear_numpy_force_caches,
    compute_forces_numpy,
)


def test_np_stack_cache_keeps_small_long_lived_inputs():
    clear_numpy_force_caches()
    a = np.ones((4,), dtype=float)
    b = np.zeros((4,), dtype=float)

    first = _NpModule.stack([a, b], axis=0)
    second = _NpModule.stack([a, b], axis=0)

    assert second is first
    assert len(_NP_STACK_CACHE) == 1
    clear_numpy_force_caches()


def test_np_stack_cache_skips_large_force_path_inputs():
    clear_numpy_force_caches()
    n = _NP_STACK_CACHE_MAX_BYTES // np.dtype(float).itemsize + 1
    a = np.ones((n,), dtype=float)
    b = np.zeros((n,), dtype=float)

    out = _NpModule.stack([a, b], axis=0)

    assert out.shape == (2, n)
    assert len(_NP_STACK_CACHE) == 0


def test_np_at_indexer_preserves_functional_set_and_accumulates_duplicate_fancy_indices():
    arr = _NpModule.asarray(np.arange(6.0).reshape(2, 3))
    view_before_set = arr[:, 1]
    updated = arr.at[:, 1].set([-10.0, -20.0])

    np.testing.assert_allclose(arr, np.arange(6.0).reshape(2, 3))
    np.testing.assert_allclose(view_before_set, [1.0, 4.0])
    np.testing.assert_allclose(updated[:, 1], [-10.0, -20.0])

    accumulated = _NpModule.zeros((3,), dtype=float)
    returned = accumulated.at[np.asarray([0, 0, 2])].add(np.asarray([1.0, 2.0, 4.0]))
    assert returned is accumulated
    np.testing.assert_allclose(accumulated, [3.0, 0.0, 4.0])

    accumulated.at[1].add(5.0)
    np.testing.assert_allclose(accumulated, [3.0, 5.0, 4.0])
    assert _idx_has_fancy((slice(None), np.asarray([0, 1])))
    assert not _idx_has_fancy((slice(None), np.asarray(1)))


def test_np_stack_cache_prunes_dead_and_excess_entries():
    clear_numpy_force_caches()
    _prune_np_stack_cache()
    assert len(_NP_STACK_CACHE) == 0

    def add_dead_entry() -> None:
        a = np.ones((2,), dtype=float)
        b = np.zeros((2,), dtype=float)
        _NpModule.stack([a, b], axis=0)

    add_dead_entry()
    gc.collect()
    _prune_np_stack_cache()
    assert len(_NP_STACK_CACHE) == 0

    arrays = []
    for idx in range(_NP_STACK_CACHE_LIMIT + 2):
        arrays.append((np.full((1,), idx, dtype=float), np.full((1,), -idx, dtype=float)))
        _NpModule.stack(arrays[-1], axis=0)
    assert len(_NP_STACK_CACHE) <= _NP_STACK_CACHE_LIMIT
    clear_numpy_force_caches()


def test_np_module_array_and_elementwise_wrappers_match_numpy():
    x = _NpModule.array([[1, 2], [3, 4]], dtype=float)
    y = _NpModule.asarray(x, dtype=float)
    z = _NpModule.array([[1, 2], [3, 4]])
    assert y is x
    assert isinstance(z, _NpArray)
    np.testing.assert_allclose(_NpModule.zeros((2,), dtype=int), np.zeros((2,), dtype=int))
    np.testing.assert_allclose(_NpModule.ones((2,)), np.ones((2,)))
    assert _NpModule.empty_like(x).shape == x.shape
    np.testing.assert_allclose(_NpModule.full((2,), 3.0), [3.0, 3.0])
    np.testing.assert_allclose(_NpModule.full_like(x, 7.0), np.full_like(np.asarray(x), 7.0))
    np.testing.assert_allclose(_NpModule.arange(3), np.arange(3))
    np.testing.assert_allclose(_NpModule.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    np.testing.assert_allclose(_NpModule.reshape(x, (4,)), np.reshape(np.asarray(x), (4,)))
    np.testing.assert_allclose(_NpModule.concatenate([x, x], axis=0), np.concatenate([np.asarray(x)] * 2))
    np.testing.assert_allclose(_NpModule.where(x > 2), np.where(np.asarray(x) > 2))
    np.testing.assert_allclose(_NpModule.where(x > 2, x, 0.0), np.where(np.asarray(x) > 2, np.asarray(x), 0.0))
    np.testing.assert_allclose(_NpModule.maximum(x, 2.0), np.maximum(np.asarray(x), 2.0))
    np.testing.assert_allclose(_NpModule.minimum(x, 2.0), np.minimum(np.asarray(x), 2.0))
    np.testing.assert_allclose(_NpModule.abs(-x), np.abs(-np.asarray(x)))
    np.testing.assert_allclose(_NpModule.sqrt(x), np.sqrt(np.asarray(x)))
    np.testing.assert_allclose(_NpModule.sum(x, axis=0), np.sum(np.asarray(x), axis=0))
    np.testing.assert_allclose(_NpModule.all(x > 0), np.all(np.asarray(x) > 0))
    np.testing.assert_allclose(_NpModule.any(x > 3), np.any(np.asarray(x) > 3))
    np.testing.assert_array_equal(_NpModule.isfinite([1.0, np.inf]), np.isfinite([1.0, np.inf]))
    np.testing.assert_array_equal(_NpModule.isnan([1.0, np.nan]), np.isnan([1.0, np.nan]))
    np.testing.assert_array_equal(_NpModule.isinf([1.0, np.inf]), np.isinf([1.0, np.inf]))
    np.testing.assert_allclose(_NpModule.mean(x, axis=1), np.mean(np.asarray(x), axis=1))
    np.testing.assert_allclose(_NpModule.max(x), np.max(np.asarray(x)))
    np.testing.assert_allclose(_NpModule.min(x), np.min(np.asarray(x)))
    np.testing.assert_allclose(_NpModule.einsum("ij,j->i", x, [1.0, 2.0]), np.einsum("ij,j->i", x, [1.0, 2.0]))
    np.testing.assert_allclose(_NpModule.take(x, [0], axis=0), np.take(np.asarray(x), [0], axis=0))
    np.testing.assert_allclose(_NpModule.broadcast_to([1.0], (2,)), np.broadcast_to([1.0], (2,)))
    np.testing.assert_allclose(_NpModule.moveaxis(np.ones((1, 2, 3)), 0, -1), np.moveaxis(np.ones((1, 2, 3)), 0, -1))
    np.testing.assert_allclose(_NpModule.expand_dims([1.0, 2.0], 0), np.expand_dims([1.0, 2.0], 0))
    np.testing.assert_allclose(_NpModule.squeeze(np.ones((1, 2, 1))), np.squeeze(np.ones((1, 2, 1))))
    np.testing.assert_allclose(_NpModule.squeeze(np.ones((1, 2, 1)), axis=0), np.squeeze(np.ones((1, 2, 1)), axis=0))
    np.testing.assert_allclose(_NpModule.transpose(x), np.transpose(np.asarray(x)))
    np.testing.assert_allclose(_NpModule.clip(x, 2.0, 3.0), np.clip(np.asarray(x), 2.0, 3.0))
    np.testing.assert_allclose(_NpModule.sin(x), np.sin(np.asarray(x)))
    np.testing.assert_allclose(_NpModule.cos(x), np.cos(np.asarray(x)))
    np.testing.assert_allclose(_NpModule.exp(x), np.exp(np.asarray(x)))
    np.testing.assert_allclose(_NpModule.log(x), np.log(np.asarray(x)))
    np.testing.assert_allclose(_NpModule.sign(x - 2.5), np.sign(np.asarray(x) - 2.5))
    np.testing.assert_allclose(_NpModule.dot([1.0, 2.0], [3.0, 4.0]), np.dot([1.0, 2.0], [3.0, 4.0]))
    np.testing.assert_allclose(_NpModule.matmul(x, x), np.matmul(np.asarray(x), np.asarray(x)))
    np.testing.assert_allclose(_NpModule.vstack([x, x]), np.vstack([np.asarray(x)] * 2))
    np.testing.assert_allclose(_NpModule.hstack([x, x]), np.hstack([np.asarray(x)] * 2))
    np.testing.assert_allclose(_NpModule.roll(x, 1, axis=0), np.roll(np.asarray(x), 1, axis=0))
    np.testing.assert_allclose(_NpModule.pad([1.0], (1, 1), mode="constant"), np.pad([1.0], (1, 1)))
    np.testing.assert_allclose(_NpModule.sort([3.0, 1.0]), np.sort([3.0, 1.0]))
    np.testing.assert_allclose(_NpModule.argsort([3.0, 1.0]), np.argsort([3.0, 1.0]))
    np.testing.assert_allclose(_NpModule.floor([1.2]), np.floor([1.2]))
    np.testing.assert_allclose(_NpModule.ceil([1.2]), np.ceil([1.2]))
    np.testing.assert_allclose(_NpModule.round([1.25], decimals=1), np.round([1.25], decimals=1))
    np.testing.assert_allclose(_NpModule.prod(x, axis=0), np.prod(np.asarray(x), axis=0))
    np.testing.assert_allclose(_NpModule.cumsum([1.0, 2.0]), np.cumsum([1.0, 2.0]))
    np.testing.assert_allclose(_NpModule.real([1.0 + 2.0j]), np.real([1.0 + 2.0j]))
    np.testing.assert_allclose(_NpModule.imag([1.0 + 2.0j]), np.imag([1.0 + 2.0j]))


def test_np_fft_and_jax_lax_shims_match_numpy():
    data = np.asarray([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(_NpFftModule.fft(data), np.fft.fft(data))
    np.testing.assert_allclose(_NpFftModule.ifft(np.fft.fft(data)), data)
    np.testing.assert_allclose(_NpFftModule.rfft(data), np.fft.rfft(data))
    np.testing.assert_allclose(_NpFftModule.irfft(np.fft.rfft(data), n=data.size), data)

    assert _NumpyLaxShim.cond(True, lambda x: x + 1, lambda x: x - 1, 3) == 4
    assert _NumpyLaxShim.cond(False, lambda x: x + 1, lambda x: x - 1, 3) == 2
    lhs = np.arange(6.0).reshape(2, 3)
    rhs = np.arange(12.0).reshape(3, 4)
    out = _NumpyLaxShim.dot_general(lhs, rhs, (((1,), (0,)), ((), ())))
    np.testing.assert_allclose(out, lhs @ rhs)
    with _NumpyJaxShim.named_scope("unit"):
        assert _NumpyJaxShim.default_backend() == "cpu"
    with _NumpyJaxShim.profiler.TraceAnnotation("unit"):
        assert _NumpyJaxShim.block_until_ready(5) == 5


def test_numpy_module_patch_restores_force_path_modules_and_numpy_mode():
    original_jnp = vf.jnp
    original_solve_jnp = solve_mod.jnp
    original_solve_jax = solve_mod.jax
    original_payload_jnp = payload_helpers_mod.jnp
    assert compat.has_jax()

    with _numpy_module_patch():
        assert vf.jnp is _NP_MODULE
        assert solve_mod.jnp is _NP_MODULE
        assert isinstance(solve_mod.jax, _NumpyJaxShim)
        assert payload_helpers_mod.jnp is _NP_MODULE
        assert not compat.has_jax()

    assert vf.jnp is original_jnp
    assert solve_mod.jnp is original_solve_jnp
    assert solve_mod.jax is original_solve_jax
    assert payload_helpers_mod.jnp is original_payload_jnp
    assert compat.has_jax()


def test_numpy_module_patch_deletes_attrs_that_were_absent(monkeypatch):
    import vmec_jax.vmec_numpy_forces as vmn

    class DummyModule:
        pass

    dummy = DummyModule()
    monkeypatch.setattr(vmn, "_PATCHES", [(dummy, [("temporary_jnp", _NP_MODULE)])])

    with _numpy_module_patch():
        assert dummy.temporary_jnp is _NP_MODULE
    assert not hasattr(dummy, "temporary_jnp")


def test_np_einsum_fast_paths_match_generic_einsum():
    rng = np.random.default_rng(1)
    cases = [
        ("psmk,kn->psmn", rng.normal(size=(2, 3, 4, 5)), rng.normal(size=(5, 6))),
        ("sik,im->smk", rng.normal(size=(2, 3, 4)), rng.normal(size=(3, 5))),
        ("smk,kn->smn", rng.normal(size=(2, 3, 4)), rng.normal(size=(4, 5))),
        ("smn,kn->smk", rng.normal(size=(2, 3, 4)), rng.normal(size=(5, 4))),
        ("smk,im->sik", rng.normal(size=(2, 3, 4)), rng.normal(size=(5, 3))),
        ("...k,kij->...ij", rng.normal(size=(2, 3)), rng.normal(size=(3, 4, 5))),
        ("...ij,kij->...k", rng.normal(size=(2, 4, 5)), rng.normal(size=(3, 4, 5))),
        ("sij,kij->sk", rng.normal(size=(2, 4, 5)), rng.normal(size=(3, 4, 5))),
        ("...k,tkij->t...ij", rng.normal(size=(2, 3)), rng.normal(size=(4, 3, 5, 6))),
    ]
    for expr, a, b in cases:
        np.testing.assert_allclose(_np_einsum(expr, a, b), np.einsum(expr, a, b))
    np.testing.assert_allclose(_np_einsum("ij,ij->", np.ones((2, 2)), np.ones((2, 2))), 4.0)


def test_to_numpy_recursive_handles_nested_dataclasses_and_fallbacks():
    from dataclasses import dataclass

    class BadArray:
        def __array__(self, dtype=None, copy=None):
            raise TypeError("not array-like")

    @dataclass(frozen=True)
    class Inner:
        arr: object
        label: str

    @dataclass(frozen=True)
    class Outer:
        inner: Inner
        scalar: int
        optional: object = None

    obj = Outer(Inner([1.0, 2.0], "a"), 3)
    converted = _to_numpy_recursive(obj)
    assert isinstance(converted.inner.arr, np.ndarray)
    assert converted.inner.label == "a"
    assert converted.scalar == 3
    np.testing.assert_allclose(_to_numpy_recursive([4.0, 5.0]), [4.0, 5.0])
    bad = BadArray()
    assert _to_numpy_recursive(bad) is bad


@dataclass(frozen=True)
class _State:
    Rcos: object
    Rsin: object
    Zcos: object
    Zsin: object
    Lcos: object
    Lsin: object


def test_compute_forces_numpy_converts_state_and_constraint_inputs():
    state = _State(
        Rcos=np.asarray([[1.0]]),
        Rsin=np.asarray([[2.0]]),
        Zcos=np.asarray([[3.0]]),
        Zsin=np.asarray([[4.0]]),
        Lcos=np.asarray([[5.0]]),
        Lsin=np.asarray([[6.0]]),
    )
    calls = []

    def fake_compute_forces_impl(state_np, **kwargs):
        calls.append((state_np, kwargs))
        assert isinstance(state_np.Rcos, _NpArray)
        assert isinstance(kwargs["constraint_precond_diag"][0], _NpArray)
        assert isinstance(kwargs["constraint_precond_diag"][1], _NpArray)
        assert isinstance(kwargs["constraint_tcon"], _NpArray)
        assert isinstance(kwargs["constraint_rcon0"], _NpArray)
        assert isinstance(kwargs["constraint_zcon0"], _NpArray)
        assert kwargs["constraint_precond_active"] is True
        assert kwargs["constraint_tcon_active"] is False
        assert kwargs["zero_m1"] == 0.25
        assert vf.jnp is _NP_MODULE
        return "forces-result"

    result = compute_forces_numpy(
        fake_compute_forces_impl,
        state,
        include_edge=True,
        include_edge_residual=False,
        zero_m1=np.asarray(0.25),
        freeb_bsqvac_half=np.asarray([[[7.0]]]),
        constraint_rcon0=np.asarray([[[1.0]]]),
        constraint_zcon0=np.asarray([[[2.0]]]),
        constraint_precond_diag=(np.asarray([3.0]), np.asarray([4.0])),
        constraint_tcon=np.asarray([5.0]),
        constraint_precond_active=np.asarray(True),
        constraint_tcon_active=np.asarray(False),
        iter_idx=9,
    )

    assert result == "forces-result"
    assert len(calls) == 1
    forwarded = calls[0][1]
    assert forwarded["include_edge"] is True
    assert forwarded["include_edge_residual"] is False
    np.testing.assert_allclose(forwarded["freeb_bsqvac_half"], [[[7.0]]])
    assert forwarded["iter_idx"] == 9


def test_compute_forces_numpy_falls_back_when_optional_inputs_cannot_convert():
    class BadState:
        Rcos = Rsin = Zcos = Zsin = Lcos = Lsin = object()

    class BadArray:
        def __array__(self, dtype=None, copy=None):
            raise TypeError("not array-like")

    bad = BadArray()
    seen = {}

    def fake_compute_forces_impl(state_np, **kwargs):
        seen["state"] = state_np
        seen["kwargs"] = kwargs
        return "fallback-result"

    result = compute_forces_numpy(
        fake_compute_forces_impl,
        BadState(),
        include_edge=False,
        include_edge_residual=False,
        constraint_rcon0=bad,
        constraint_zcon0=bad,
        constraint_precond_diag=(bad, bad),
        constraint_tcon=bad,
        constraint_precond_active=bad,
        constraint_tcon_active=bad,
        zero_m1=0.0,
    )

    assert result == "fallback-result"
    assert isinstance(seen["state"], BadState)
    assert seen["kwargs"]["constraint_rcon0"] is bad
    assert seen["kwargs"]["constraint_zcon0"] is bad
    assert seen["kwargs"]["constraint_precond_diag"] == (bad, bad)
    assert seen["kwargs"]["constraint_tcon"] is bad
    assert seen["kwargs"]["constraint_precond_active"] is bad
    assert seen["kwargs"]["constraint_tcon_active"] is bad

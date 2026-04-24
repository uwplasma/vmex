import numpy as np

from vmec_jax.vmec_numpy_forces import (
    _NP_STACK_CACHE,
    _NP_STACK_CACHE_MAX_BYTES,
    _NpModule,
    clear_numpy_force_caches,
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

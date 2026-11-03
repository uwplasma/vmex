"""AotLane: serialized executables reload bit-identically and never block a call."""

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import aot

_SCRIPT = textwrap.dedent("""
    import sys, jax, jax.numpy as jnp, numpy as np
    from pathlib import Path
    from vmex.core import aot
    jax.config.update("jax_num_cpu_devices", int(sys.argv[3]))
    aot._directory = lambda: Path(sys.argv[1]) / "aot"
    jax.config.update("jax_compilation_cache_dir", sys.argv[1] + "/xla")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    @jax.tree_util.register_pytree_node_class
    class Field:                 # computes in its unflatten, like an ESSOS field
        def __init__(self, y):
            self.y, self.scale = y, jnp.asarray(y).sum()
        def tree_flatten(self):
            return (self.y,), None
        @classmethod
        def tree_unflatten(cls, _, leaves):
            return cls(*leaves)
    def fn(x, f, *, k=2):
        # The solve is a LAPACK call, which a loaded executable must find bound.
        return jax.lax.fori_loop(0, 5, lambda i, c: jnp.linalg.solve(2 * jnp.eye(4), c) * k + f.y, x)
    lane = aot.AotLane(fn, static_argnames=("k",))
    if sys.argv[2] == "build":   # the persistent cache already holds this program
        jax.jit(fn, static_argnames=("k",))(jnp.arange(4.0), Field(jnp.ones(4)), k=3)
        jax.clear_caches()
    if sys.argv[2] == "load":
        lane.jit = None          # a cached lane must not trace or lower
    x, y = jnp.arange(4.0), Field(jnp.ones(4))
    got = lane(x, y, k=3)
    np.testing.assert_array_equal(got, jax.jit(fn, static_argnames=("k",))(x, y, k=3)
                                  if sys.argv[2] == "build" else np.load(sys.argv[1] + "/want.npy"))
    np.save(sys.argv[1] + "/want.npy", got)
""")


@pytest.mark.parametrize("devices", [1, 4])
def test_executable_reloads_in_a_fresh_process(tmp_path, devices):
    """Also when the build was served by the persistent cache (its blobs fail to run),
    on a backend with several devices, for a pytree that computes in its unflatten,
    and for a program whose first call in the process is a LAPACK kernel."""
    for mode in ("build", "load"):
        subprocess.run([sys.executable, "-c", _SCRIPT, str(tmp_path), mode, str(devices)],
                       check=True, capture_output=True)
    assert len(list((tmp_path / "aot").glob("*.aot"))) == 1


def test_fallbacks_never_block_a_call(tmp_path, monkeypatch):
    monkeypatch.setattr(aot, "_directory", lambda: tmp_path / "aot")

    def fn(x, y):
        return x * 2 + y

    lane = aot.AotLane(fn)
    x, y = jnp.arange(4.0), jnp.ones(4)
    want = jax.jit(fn)(x, y)
    np.testing.assert_array_equal(jax.jit(lambda a: lane(a, y))(x), want)   # tracers
    assert not (tmp_path / "aot").exists()
    np.testing.assert_array_equal(lane(x, y), want)
    for f in (tmp_path / "aot").glob("*.aot"):
        f.write_bytes(b"corrupt")
    np.testing.assert_array_equal(aot.AotLane(fn)(x, y), want)              # corrupt entry
    monkeypatch.setattr(aot, "_directory", lambda: None)                    # cache off
    np.testing.assert_array_equal(aot.AotLane(fn)(x, y), want)


def test_donating_lane_matches_jit_and_never_reuses_donated_inputs(tmp_path, monkeypatch):
    """Repeat calls, a rejected executable and a post-donation failure behave like the jit lane."""
    monkeypatch.setattr(aot, "_directory", lambda: tmp_path / "aot")

    def fn(c, y):
        return c * 2 + y

    lane = aot.AotLane(fn, donate_argnums=(0,))
    ref = jax.jit(fn, donate_argnums=(0,))
    y = jnp.ones(4)
    for _ in range(2):                              # build, then in-memory executable
        c, c_ref = jnp.arange(4.0), jnp.arange(4.0)
        np.testing.assert_array_equal(lane(c, y), ref(c_ref, y))
        assert c.is_deleted() == c_ref.is_deleted()
        assert not y.is_deleted()

    def rejecting(*_a):
        raise TypeError("argument drift")           # raised before any donation

    monkeypatch.setattr(lane, "executable", lambda *a, **k: rejecting)
    c = jnp.arange(4.0)
    np.testing.assert_array_equal(lane(c, y), ref(jnp.arange(4.0), y))     # fallback

    def failing_after_donation(c, _y):
        c.delete()
        raise RuntimeError("device failure")

    monkeypatch.setattr(lane, "executable", lambda *a, **k: failing_after_donation)
    with pytest.raises(RuntimeError, match="device failure"):
        lane(jnp.arange(4.0), y)


def test_target_device_follows_jit_placement():
    """Committed leaves fix the device, then a default-device context; two devices skip the cache."""
    cpu = jax.devices("cpu")[0]

    class Spread:                                   # committed to two devices
        committed, shape, dtype, weak_type = True, (2,), np.dtype("float64"), False

        def devices(self):
            return {"d0", "d1"}

    assert aot._device([Spread()]) is None
    assert aot._structure((jnp.ones(2), Spread()), {}) is None
    assert aot._device([jax.device_put(jnp.ones(2), cpu)]) == cpu
    with jax.default_device(cpu):
        assert aot._device([np.ones(2)]) == cpu

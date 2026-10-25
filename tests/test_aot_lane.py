"""AotLane: serialized executables reload bit-identically and never block a call."""

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

from vmex.core import aot

_SCRIPT = textwrap.dedent("""
    import sys, jax, jax.numpy as jnp, numpy as np
    from pathlib import Path
    from vmex.core import aot
    aot._directory = lambda: Path(sys.argv[1]) / "aot"
    def fn(x, y, *, k=2):
        return jax.lax.fori_loop(0, 5, lambda i, c: c * k + y, x)
    lane = aot.AotLane(fn, static_argnames=("k",))
    if sys.argv[2] == "load":
        lane.jit = None          # a cached lane must not trace or lower
    x, y = jnp.arange(4.0), jnp.ones(4)
    got = lane(x, y, k=3)
    np.testing.assert_array_equal(got, jax.jit(fn, static_argnames=("k",))(x, y, k=3)
                                  if sys.argv[2] == "build" else np.load(sys.argv[1] + "/want.npy"))
    np.save(sys.argv[1] + "/want.npy", got)
""")


def test_executable_reloads_in_a_fresh_process(tmp_path):
    for mode in ("build", "load"):
        subprocess.run([sys.executable, "-c", _SCRIPT, str(tmp_path), mode],
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

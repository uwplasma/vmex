"""``jax.jit(jax.value_and_grad(f))`` through the implicit path, first call.

The first call of an outer jit builds the host-side runtime template and the
boundary packing tables from inside the trace.  Both are cached, so they must
come out as concrete arrays (never tracers) and the setup lane must not be
dispatched as a nested jit carrying ``compiler_options``.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from vmex.core import implicit as im
from vmex.core import setup as setup_mod
from vmex.core.input import VmecInput

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_jit_value_and_grad_first_call_matches_eager():
    inp = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    p0 = im.params_from_input(inp)

    def f(p):
        return im.run(inp, p, ftol=1e-12, max_iterations=2000).wb

    # Cold host caches: the traced first call must populate them itself.
    im._template_runtime.cache_clear()
    im._PACK_TABLE_CACHE.clear()
    setup_mod._setup_lane.cache_clear()
    v_jit, g_jit = jax.jit(jax.value_and_grad(f))(p0)
    for leaf in jax.tree.leaves(list(im._PACK_TABLE_CACHE.values())):
        assert not isinstance(leaf, jax.core.Tracer)

    v_eager, g_eager = jax.value_and_grad(f)(p0)
    np.testing.assert_allclose(float(v_jit), float(v_eager), rtol=1e-12)
    for a, b in zip(jax.tree.leaves(g_jit), jax.tree.leaves(g_eager)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-8, atol=1e-12)
    assert any(np.any(np.asarray(a) != 0.0) for a in jax.tree.leaves(g_jit))

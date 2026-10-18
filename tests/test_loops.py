"""Loops over derivative computations (vmex.core.loops).

XLA:GPU can miscompile a bare ``lax.map`` / ``lax.scan`` / ``fori_loop``
whose body takes a derivative (wrong tangents, right primals; see
:mod:`vmex.core.loops`).  These tests keep such loops out of ``vmex/``
statically, and check :func:`~vmex.core.loops.map_columns` -- values,
``jvp``, ``vjp`` and ``jacrev`` through it -- against the unbatched map
and central differences on whatever device the suite runs on (the GPU lane
is the one that matters).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from vmex.core.loops import map_columns

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
_DERIVATIVES = {"jvp", "vjp", "linearize", "grad", "value_and_grad", "jacfwd", "jacrev", "hessian"}
_HELPER = ROOT / "vmex" / "core" / "loops.py"


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _differentiates(node) -> bool:
    return any(isinstance(n, ast.Call) and _dotted(n.func).split(".")[-1] in _DERIVATIVES
               for n in ast.walk(node))


def _bare_loops(path: Path) -> list[str]:
    """``file:line`` of every ``lax.map`` / ``chunk_map`` (maps go through ``map_columns``: a batch of one,
    bare or as an uneven tail, is the miscompiled program), and of every ``scan``/``fori_loop``/``while_loop``
    whose body (a lambda or a function defined in the same module) differentiates without being unrolled."""
    tree = ast.parse(path.read_text(), filename=str(path))
    defs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        keywords = {k.arg for k in node.keywords}
        where = f"{path.relative_to(ROOT)}:{node.lineno}"
        if name.endswith("lax.map") or name.split(".")[-1] == "chunk_map":
            found.append(f"{where} {name.split('.')[-1]} (use map_columns)")
        elif name.split(".")[-1] in ("scan", "fori_loop", "while_loop") and "lax" in name and "unroll" not in keywords:
            bodies = [a for a in node.args if isinstance(a, ast.Lambda)]
            bodies += [defs[a.id] for a in node.args if isinstance(a, ast.Name) and a.id in defs]
            if any(_differentiates(body) for body in bodies):
                found.append(f"{where} {name.split('.')[-1]} body differentiates")
    return found


def test_no_bare_loop_over_derivatives_in_vmex():
    """Every map goes through ``map_columns``; no rolled scan/fori/while body differentiates."""
    found = [hit for path in sorted((ROOT / "vmex").rglob("*.py")) if path != _HELPER for hit in _bare_loops(path)]
    assert not found, "bare loops (XLA:GPU miscompile, see vmex.core.loops):\n" + "\n".join(found)


def test_the_lint_sees_the_patterns(tmp_path):
    """The static check flags the forms it is meant to (so a silent parser change cannot pass it vacuously)."""
    source = tmp_path / "vmex" / "probe.py"
    source.parent.mkdir()
    source.write_text(
        "import jax\nfrom jax import lax\n"
        "def body(c, x):\n    return c, jax.jvp(f, (x,), (x,))[1]\n"
        "a = jax.lax.map(f, xs)\n"
        "b = lax.map(f, xs, batch_size=4)\n"
        "b2 = map_columns(f, xs, batch_size=4)\n"
        "c = jax.lax.scan(body, 0, xs)\n"
        "d = jax.lax.scan(body, 0, xs, unroll=True)\n"
        "e = jax.lax.fori_loop(0, 3, lambda i, v: jax.grad(g)(v), v0)\n"
        "f2 = jax.lax.scan(lambda c, x: (c + x, None), 0, xs)\n"
        "g2 = im.chunk_map(f, xs, chunk_size=8)\n")
    global ROOT
    saved, ROOT = ROOT, tmp_path
    try:
        hits = _bare_loops(source)
    finally:
        ROOT = saved
    assert [h.split(" ", 1)[0] for h in hits] == [
        "vmex/probe.py:5", "vmex/probe.py:6", "vmex/probe.py:8", "vmex/probe.py:10", "vmex/probe.py:12"], hits


def _field(x, p):
    """A smooth nonlinear map whose tangent mixes a reduction and a product, as ``B . n``."""
    b = jnp.tanh(jnp.outer(x, p)) @ jnp.cos(p)
    n = jnp.sin(x) / jnp.sqrt(1.0 + jnp.sum(x**2))
    return jnp.sum(b * n) * x + jnp.exp(-p[: x.size] ** 2)


@pytest.mark.parametrize("batch_size", [1, 3, None])
def test_map_columns_matches_the_unbatched_map_and_differences(batch_size):
    rng = np.random.default_rng(0)
    x0, p0 = jnp.asarray(rng.normal(size=6)), jnp.asarray(rng.normal(size=9))
    columns = jnp.asarray(rng.normal(size=(7, 6)))

    def tangent(v):
        return jax.jvp(lambda x: _field(x, p0), (x0,), (v,))[1]

    reference = jax.vmap(tangent)(columns)
    got = jax.jit(lambda c: map_columns(tangent, c, batch_size=batch_size))(columns)
    np.testing.assert_allclose(got, reference, rtol=1e-13, atol=1e-13)
    h = 1e-6
    fd = np.stack([(_field(x0 + h * v, p0) - _field(x0 - h * v, p0)) / (2 * h) for v in columns])
    np.testing.assert_allclose(got, fd, rtol=1e-7, atol=1e-8)

    _, pullback = jax.vjp(lambda x: _field(x, p0), x0)
    rows = jax.jit(lambda c: map_columns(lambda u: pullback(u)[0], c, batch_size=batch_size))(columns)
    np.testing.assert_allclose(rows, jax.vmap(lambda u: pullback(u)[0])(columns), rtol=1e-13, atol=1e-13)

    # reverse mode *through* the loop, as a driver's jacrev of a mapped distance
    def mapped(p):
        return map_columns(lambda v: jnp.sum(_field(v, p) ** 2), columns, batch_size=batch_size)

    jac = jax.jit(jax.jacrev(mapped))(p0)
    fd = np.stack([(mapped(p0 + h * e) - mapped(p0 - h * e)) / (2 * h) for e in np.eye(p0.size)], axis=1)
    np.testing.assert_allclose(jac, fd, rtol=1e-6, atol=1e-8)


def test_map_columns_never_compiles_a_batch_of_one(monkeypatch):
    """Every vmapped body sees a batch of at least two, padding included (one column, uneven tails)."""
    original, batches = jax.vmap, []

    def spy(function):
        def mapped(xs):
            batches.append({int(jnp.shape(a)[0]) for a in jax.tree.leaves(xs)})
            return original(function)(xs)
        return mapped

    monkeypatch.setattr(jax, "vmap", spy)
    for count, batch_size in ((1, 1), (1, None), (2, 1), (5, 1), (7, 3), (9, 4), (49, 8), (4, None)):
        batches.clear()
        out = jax.jit(lambda xs, b=batch_size: map_columns(lambda c: 2.0 * c, xs, batch_size=b))(
            jnp.arange(float(count)))
        np.testing.assert_array_equal(out, 2.0 * np.arange(count))
        sizes = set().union(*batches)
        assert sizes and min(sizes) >= 2, (count, batch_size, sizes)
        if batch_size is not None and count > 1:
            assert max(sizes) <= max(2, batch_size), (count, batch_size, sizes)


def test_map_columns_pytrees_zero_size_leaves_and_errors():
    columns = (jnp.arange(5.0), jnp.zeros((5, 0)))
    out = map_columns(lambda c: {"a": 2.0 * c[0], "empty": c[1]}, columns, batch_size=2)
    np.testing.assert_array_equal(out["a"], 2.0 * np.arange(5.0))
    assert out["empty"].shape == (5, 0)
    assert map_columns(jnp.sin, jnp.zeros((0, 4))).shape == (0, 4)
    with pytest.raises(ValueError, match="batch_size"):
        map_columns(jnp.sin, jnp.zeros(3), batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        map_columns(jnp.sin, jnp.zeros(3), batch_size=True)
    with pytest.raises(ValueError, match="disagree"):
        map_columns(lambda c: c, (jnp.zeros(3), jnp.zeros(4)))

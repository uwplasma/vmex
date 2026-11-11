"""Shipped examples call the vmex API they import: every call binds its signature."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").rglob("*.py"))


def _resolve(node, namespace):
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Attribute):
        return getattr(_resolve(node.value, namespace), node.attr, None)
    return None


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_calls_match_vmex_signatures(path):
    """A keyword an example passes (pwO_optimization.py: band_weight) exists in the installed API."""
    tree = ast.parse(path.read_text())
    namespace, dicts = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
            except ImportError:  # optional extra not installed on this lane
                pass
        elif (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
              and getattr(node.value.func, "id", "") == "dict"):
            dicts[node.targets[0].id] = [k.arg for k in node.value.keywords]
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        target = _resolve(call.func, namespace)
        if not (inspect.isclass(target) or inspect.isfunction(target)) or not target.__module__.startswith("vmex"):
            continue
        keys = [k.arg for k in call.keywords if k.arg]
        for k in call.keywords:  # **NAME expands a top-level dict(...)
            keys += dicts.get(getattr(k.value, "id", ""), []) if k.arg is None else []
        args = [None] * sum(not isinstance(a, ast.Starred) for a in call.args)
        try:
            inspect.signature(target).bind_partial(*args, **dict.fromkeys(keys))
        except TypeError as error:
            pytest.fail(f"{path.name}:{call.lineno} {target.__qualname__}: {error}")

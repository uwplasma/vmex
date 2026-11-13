"""Shipped examples call the vmex API they import: every call binds its signature."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import vmex

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").rglob("*.py"))


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


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_names_exist_in_vmex(path):
    """Every vmex name an example imports, or reaches through ``vmex.``, exists."""
    missing = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "vmex":
            try:
                module = importlib.import_module(node.module)
            except ImportError:
                missing.append(node.module)
                continue
            for alias in node.names:
                if not hasattr(module, alias.name):
                    try:
                        importlib.import_module(f"{node.module}.{alias.name}")
                    except ImportError:
                        missing.append(f"{node.module}.{alias.name}")
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id == "vmex" and not hasattr(vmex, node.attr)):
            missing.append(f"vmex.{node.attr}")
    assert missing == []


def test_pwo_example_optimization_stage_runs_and_lowers_its_cost(tmp_path):
    """The shipped pwO script, up to its verification solve, at the CI budget."""
    script = ROOT / "examples" / "optimization" / "pwO_optimization.py"
    source = script.read_text().split("### Check the result")[0].replace("__file__", repr(str(script)))
    code = source + "\nprint('COST', result.cost, 0.5 * np.sum(joint.residual(joint.x0) ** 2))\n"
    env = dict(os.environ, VMEX_EXAMPLES_CI="1", JAX_ENABLE_X64="1")
    run = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    final, initial = map(float, run.stdout.split("COST")[-1].split())
    assert np.isfinite(final) and final < initial

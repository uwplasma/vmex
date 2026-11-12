"""Every vmex name an example imports or calls through ``vmex.`` exists.

An example that names an API its installed release lacks fails only when it is
run; this resolves the names statically, in every lane that imports vmex.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import vmex

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").rglob("*.py"))


def _missing(path: Path) -> list[str]:
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
    return missing


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_names_exist_in_vmex(path):
    assert _missing(path) == []

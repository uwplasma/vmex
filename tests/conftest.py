from __future__ import annotations

"""Pytest configuration.

Allows running tests directly from the repo without requiring an editable
install, silences XLA/absl C++ noise, disables jit globally (unit tests cover
correctness on small arrays; compilation dominates runtime — tests that need
the jit lane re-enable it explicitly), and gates ``full``-marked tests behind
``RUN_FULL=1``.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Keep the test suite fast: avoid JAX compilation in unit tests.
try:  # pragma: no cover
    import jax

    jax.config.update("jax_disable_jit", True)
except Exception:  # pragma: no cover
    pass


def pytest_collection_modifyitems(config, items):
    run_full = os.environ.get("RUN_FULL", "") == "1"
    for item in items:
        if item.get_closest_marker("full") is not None and not run_full:
            item.add_marker(pytest.mark.skip(reason="Full tests disabled. Set RUN_FULL=1."))

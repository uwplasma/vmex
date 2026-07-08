"""Compatibility re-exports for the staged QI optimization example.

Reusable bounded-QI support now lives in :mod:`vmec_jax.quasi_isodynamic.optimization`.
Keep this module so existing example and test imports continue to work.
"""

from __future__ import annotations

import vmec_jax.quasi_isodynamic.optimization as _qio
from vmec_jax.quasi_isodynamic.optimization import *  # noqa: F401,F403
from vmec_jax.quasi_isodynamic.optimization import (  # noqa: F401
    _diagnostic_float,
    _finite_or_inf,
    _finite_or_none,
    _jsonable,
    _parse_float_sequence,
    _partial_diagnostics_from_history,
    _stage_result_history,
    _write_json_atomic,
)


def configure(context: dict) -> None:
    """Install script-level constants on the public module and this shim."""

    _qio.configure(context)
    globals().update(context)

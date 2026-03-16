from __future__ import annotations

"""Pytest configuration.

Allows running tests directly from the repo without requiring an editable install.
"""


import sys
from pathlib import Path

import pytest
import os


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Keep the test suite fast: avoid JAX compilation in unit tests.
# Tests cover correctness on small arrays; compilation dominates runtime.
try:  # pragma: no cover
    import jax

    jax.config.update("jax_disable_jit", True)
except Exception:  # pragma: no cover
    pass


def require_slow() -> None:
    """Skip tests marked as slow unless RUN_SLOW=1 is set."""
    if os.environ.get("RUN_SLOW", "") != "1":
        pytest.skip("Set RUN_SLOW=1 to run slow gradient/implicit tests")



_ASSET_SENTINEL = _ROOT / "examples" / "data" / "wout_circular_tokamak_reference.nc"


def _assets_available() -> bool:
    return _ASSET_SENTINEL.exists()


def pytest_collection_modifyitems(config, items):
    has_assets = _assets_available()
    run_full = os.environ.get("RUN_FULL", "") == "1"
    if run_full and not has_assets:
        raise pytest.UsageError("RUN_FULL=1 but example assets are missing. Run tools/fetch_assets.py")
    for item in items:
        if item.get_closest_marker("full") is not None:
            if not run_full:
                item.add_marker(pytest.mark.skip(reason="Full tests disabled. Set RUN_FULL=1."))
            elif not has_assets:
                item.add_marker(pytest.mark.skip(reason="Missing example assets. Run tools/fetch_assets.py"))

@pytest.fixture(scope="session")
def load_case_qa_reactorscale_lowres():
    """Load the bundled QA reactor-scale low-res input used in examples."""
    from vmec_jax._compat import has_jax, enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import initial_guess_from_boundary

    if has_jax():
        enable_x64(True)

    inpath = _ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)
    return cfg, indata, static, bdy, st0


@pytest.fixture(scope="session")
def load_case_circular_tokamak():
    """Load the bundled circular tokamak (axisymmetric, lasym=False) input used in examples."""
    from vmec_jax._compat import has_jax, enable_x64
    from vmec_jax.config import load_config
    from vmec_jax.static import build_static
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.init_guess import initial_guess_from_boundary

    if has_jax():
        enable_x64(True)

    inpath = _ROOT / "examples" / "data" / "input.circular_tokamak"
    cfg, indata = load_config(str(inpath))
    static = build_static(cfg)
    bdy = boundary_from_indata(indata, static.modes)
    st0 = initial_guess_from_boundary(static, bdy, indata)
    return cfg, indata, static, bdy, st0

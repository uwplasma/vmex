from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


from vmec_jax.driver import load_example
from vmec_jax.kernels.forces import vmec_forces_rz_from_wout
pytestmark = pytest.mark.full


def test_freeb_bsqvac_edge_slice_matches_full_half_mesh():
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    wout_path = root / "examples" / "data" / "wout_circular_tokamak_reference.nc"
    if not wout_path.exists():
        pytest.skip("Missing example assets. Run tools/fetch_assets.py")
    ex = load_example("circular_tokamak", root=root, with_wout=True)

    kernels_ref = vmec_forces_rz_from_wout(
        state=ex.state,
        static=ex.static,
        wout=ex.wout,
        use_vmec_synthesis=True,
    )
    vac_edge = np.full(np.asarray(kernels_ref.pr1_even[-1]).shape, 0.125, dtype=float)
    vac_full = np.zeros((int(ex.cfg.ns),) + vac_edge.shape, dtype=float)
    vac_full[-1, :, :] = vac_edge

    kernels_full = vmec_forces_rz_from_wout(
        state=ex.state,
        static=ex.static,
        wout=ex.wout,
        freeb_bsqvac_half=vac_full,
        use_vmec_synthesis=True,
    )
    kernels_edge = vmec_forces_rz_from_wout(
        state=ex.state,
        static=ex.static,
        wout=ex.wout,
        freeb_bsqvac_half=vac_edge,
        use_vmec_synthesis=True,
    )

    np.testing.assert_allclose(np.asarray(kernels_full.gcon), np.asarray(kernels_edge.gcon))
    np.testing.assert_allclose(np.asarray(kernels_full.armn_e), np.asarray(kernels_edge.armn_e))
    np.testing.assert_allclose(np.asarray(kernels_full.azmn_e), np.asarray(kernels_edge.azmn_e))

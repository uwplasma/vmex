from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.kernels.forces import vmec_forces_rz_from_wout
from vmec_jax.kernels.tomnsp import vmec_angle_grid
from vmec_jax.wout import read_wout, state_from_wout
pytestmark = pytest.mark.full


def test_c_kernels_nonzero_for_axisym_reference():
    """VMEC2000 computes C-kernels for axisym cases; ensure vmec_jax does too."""
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples/data/input.circular_tokamak"
    wout_path = root / "examples/data/wout_circular_tokamak_reference.nc"
    assert input_path.exists()
    assert wout_path.exists()

    cfg, indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    st = state_from_wout(wout)

    grid = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
    )
    static = build_static(cfg, grid=grid)

    k = vmec_forces_rz_from_wout(state=st, static=static, wout=wout, indata=indata)

    crmn_e = np.asarray(k.crmn_e)
    czmn_e = np.asarray(k.czmn_e)
    assert np.isfinite(crmn_e).all()
    assert np.isfinite(czmn_e).all()
    assert float(np.max(np.abs(crmn_e))) > 0.0
    assert float(np.max(np.abs(czmn_e))) > 0.0

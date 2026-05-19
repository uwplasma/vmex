from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_tomnsp import vmec_angle_grid
from vmec_jax.wout import read_wout, state_from_wout, wout_minimal_from_fixed_boundary


def _rms(value) -> float:
    arr = np.asarray(value, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def _abs_rms(got, expected) -> float:
    return _rms(np.asarray(got, dtype=float) - np.asarray(expected, dtype=float))


def test_up_down_lasym_reference_state_bsubvmns_uses_iequi_asym_source(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples_single_grid/data/input.up_down_asymmetric_tokamak"
    wout_path = root / "examples_single_grid/data/wout_up_down_asymmetric_tokamak_reference.nc"
    if not input_path.exists() or not wout_path.exists():
        pytest.skip("Missing bundled up_down_asymmetric_tokamak VMEC2000 reference")

    cfg, indata = load_config(str(input_path))
    wref = read_wout(wout_path)
    cfg = replace(
        cfg,
        ns=int(wref.ns),
        mpol=int(wref.mpol),
        ntor=int(wref.ntor),
        nfp=int(wref.nfp),
        lasym=bool(wref.lasym),
        lthreed=bool(int(wref.ntor) > 0),
    )
    grid = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    static = build_static(cfg, grid=grid)
    state = state_from_wout(wref)

    wnew = wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_up_down_asymmetric_tokamak_vmec_jax.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=int(wref.signgs),
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
    )

    assert bool(wnew.lasym) is True
    assert _rms(wref.bsubvmns) < 1.0e-5
    assert _abs_rms(wnew.bsubvmns, wref.bsubvmns) < 5.0e-5

    # Guard channels that regressed when the full LASYM bsubu/bsubv output was
    # switched to the IEQUI source instead of only the asymmetric bsubv channel.
    assert _abs_rms(wnew.bsubvmnc, wref.bsubvmnc) < 5.0e-4
    assert _abs_rms(wnew.bsubumns, wref.bsubumns) < 1.0e-12
    assert _abs_rms(wnew.bsupumns, wref.bsupumns) < 1.0e-12

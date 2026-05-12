from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.field import chips_from_wout_chipf, half_mesh_avg_from_full_mesh
from vmec_jax.wout import read_wout


CASES = (
    "wout_circular_tokamak.nc",
    "wout_shaped_tokamak_pressure.nc",
    "wout_nfp4_QH_warm_start.nc",
    "wout_LandremanPaul2021_QA_lowres.nc",
    "wout_nfp3_QI_fixed_resolution_final.nc",
)


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.sqrt(np.mean(np.asarray(b, dtype=float) ** 2))), 1.0e-30)
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)) / scale)


@pytest.mark.parametrize("wout_name", CASES)
def test_bundled_vmec2000_chipf_is_detected_as_half_mesh(wout_name: str) -> None:
    """Real bundled wouts should use VMEC2000 half-mesh ``chipf`` convention."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    wout = read_wout(_data_dir() / wout_name)
    chipf = np.asarray(wout.chipf, dtype=float)
    phipf = np.asarray(wout.phipf, dtype=float)
    iotaf = np.asarray(wout.iotaf, dtype=float)
    iotas = np.asarray(wout.iotas, dtype=float)

    rel_half = _relative_rms(chipf, iotaf * phipf)
    rel_full = _relative_rms(chipf, iotas * phipf)

    assert rel_half < 1.0e-12
    assert rel_full > 1.0e-3


@pytest.mark.parametrize("wout_name", CASES)
def test_bundled_wout_chips_round_trips_to_vmec2000_chipf(wout_name: str) -> None:
    """The parity helper should return full-mesh chips whose VMEC average is chipf."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    wout = read_wout(_data_dir() / wout_name)
    chipf = np.asarray(wout.chipf, dtype=float)
    chips = np.asarray(
        chips_from_wout_chipf(
            chipf=chipf,
            phipf=np.asarray(wout.phipf, dtype=float),
            iotaf=np.asarray(wout.iotaf, dtype=float),
            iotas=np.asarray(wout.iotas, dtype=float),
            assume_half_if_unknown=True,
        ),
        dtype=float,
    )

    assert chips.shape == chipf.shape
    assert chips[0] == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(np.asarray(half_mesh_avg_from_full_mesh(chips)), chipf, rtol=1.0e-13, atol=1.0e-13)
    assert _relative_rms(chips, chipf) > 1.0e-5

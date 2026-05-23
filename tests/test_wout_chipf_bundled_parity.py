from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.field import chips_from_wout_chipf, half_mesh_avg_from_full_mesh
from vmec_jax.wout import read_wout


CASES = (
    ("axisym_vacuum", "examples/data/wout_circular_tokamak.nc"),
    ("axisym_finite_beta", "examples/data/wout_shaped_tokamak_pressure.nc"),
    ("qh_warm_start", "examples/data/wout_nfp4_QH_warm_start.nc"),
    ("qa_lowres", "examples/data/wout_LandremanPaul2021_QA_lowres.nc"),
    ("qi_final", "examples/data/wout_nfp3_QI_fixed_resolution_final.nc"),
    ("qi_seed", "examples/data/wout_QI_stel_seed_3127.nc"),
    ("lasym_vacuum", "examples/data/wout_basic_non_stellsym_simsopt.nc"),
    ("dshape", "examples/data/wout_DSHAPE.nc"),
    ("cth_finite_beta", "examples/data/wout_cth_like_fixed_bdy.nc"),
    ("li383_finite_beta", "examples/data/wout_li383_low_res.nc"),
    (
        "lasym_finite_beta_single_grid",
        "examples_single_grid/data/wout_basic_non_stellsym_pressure_reference.nc",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.sqrt(np.mean(np.asarray(b, dtype=float) ** 2))), 1.0e-30)
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)) / scale)


@pytest.mark.parametrize(("case_name", "wout_rel"), CASES, ids=[case[0] for case in CASES])
def test_bundled_vmec2000_chipf_is_detected_as_half_mesh(case_name: str, wout_rel: str) -> None:
    """Real bundled wouts should use VMEC2000 half-mesh ``chipf`` convention."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    wout = read_wout(_repo_root() / wout_rel)
    chipf = np.asarray(wout.chipf, dtype=float)
    phipf = np.asarray(wout.phipf, dtype=float)
    iotaf = np.asarray(wout.iotaf, dtype=float)
    iotas = np.asarray(wout.iotas, dtype=float)

    rel_half = _relative_rms(chipf, iotaf * phipf)
    rel_full = _relative_rms(chipf, iotas * phipf)

    assert rel_half < 1.0e-12, case_name
    assert rel_full > 1.0e-3, case_name


@pytest.mark.parametrize(("case_name", "wout_rel"), CASES, ids=[case[0] for case in CASES])
def test_bundled_wout_chips_round_trips_to_vmec2000_chipf(case_name: str, wout_rel: str) -> None:
    """The parity helper should return full-mesh chips whose VMEC average is chipf."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    wout = read_wout(_repo_root() / wout_rel)
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
    assert chips[0] == pytest.approx(0.0, abs=1.0e-14), case_name
    np.testing.assert_allclose(
        np.asarray(half_mesh_avg_from_full_mesh(chips)),
        chipf,
        rtol=1.0e-13,
        atol=1.0e-13,
        err_msg=case_name,
    )
    assert _relative_rms(chips, chipf) > 1.0e-5, case_name


def test_chips_from_wout_chipf_resolves_full_mesh_and_unknown_conventions() -> None:
    """Synthetic branches protect the VMEC add_fluxes full/half mesh convention."""
    pytest.importorskip("jax")

    phipf = np.asarray([2.0, 2.0, 2.0, 2.0])
    chips_full = np.asarray([0.0, 0.6, 1.0, 1.4])
    chipf_half = np.asarray(half_mesh_avg_from_full_mesh(chips_full))
    iotas = chips_full / phipf
    iotaf = chipf_half / phipf

    detected_half = chips_from_wout_chipf(
        chipf=chipf_half,
        phipf=phipf,
        iotaf=iotaf,
        iotas=iotas,
    )
    np.testing.assert_allclose(np.asarray(detected_half), chips_full, rtol=0.0, atol=1.0e-14)

    detected_full = chips_from_wout_chipf(
        chipf=chips_full,
        phipf=phipf,
        iotaf=iotaf,
        iotas=iotas,
    )
    np.testing.assert_allclose(np.asarray(detected_full), chips_full, rtol=0.0, atol=1.0e-14)

    unknown_half = chips_from_wout_chipf(chipf=chipf_half, phipf=phipf, assume_half_if_unknown=True)
    unknown_full = chips_from_wout_chipf(chipf=chips_full, phipf=phipf, assume_half_if_unknown=False)
    np.testing.assert_allclose(np.asarray(unknown_half), chips_full, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(unknown_full), chips_full, rtol=0.0, atol=0.0)

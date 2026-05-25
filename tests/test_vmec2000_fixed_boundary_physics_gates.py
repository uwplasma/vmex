from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.namelist import read_indata
from vmec_jax.wout import read_wout


@dataclass(frozen=True)
class FixedBoundaryWoutCase:
    case: str
    residual_rss_limit: float
    require_volume_scalar: bool = True


FIXED_BOUNDARY_WOUT_CASES = (
    FixedBoundaryWoutCase("circular_tokamak", residual_rss_limit=1.0e-12),
    FixedBoundaryWoutCase("shaped_tokamak_pressure", residual_rss_limit=1.0e-12),
    FixedBoundaryWoutCase("cth_like_fixed_bdy", residual_rss_limit=1.0e-12),
    FixedBoundaryWoutCase("DSHAPE", residual_rss_limit=1.0e-10),
    FixedBoundaryWoutCase("li383_low_res", residual_rss_limit=1.0e-5),
    FixedBoundaryWoutCase(
        "LandremanPaul2021_QA_lowres",
        residual_rss_limit=1.0e-10,
        require_volume_scalar=False,
    ),
    FixedBoundaryWoutCase("nfp4_QH_warm_start", residual_rss_limit=1.0e-10),
    FixedBoundaryWoutCase("nfp3_QI_fixed_resolution_final", residual_rss_limit=1.0e-10),
    FixedBoundaryWoutCase("purely_toroidal_field", residual_rss_limit=1.0e-12),
)


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _case_paths(case: FixedBoundaryWoutCase) -> tuple[Path, Path]:
    data_dir = _data_dir()
    input_path = data_dir / f"input.{case.case}"
    wout_path = data_dir / f"wout_{case.case}.nc"
    if not input_path.exists() or not wout_path.exists():
        pytest.skip(f"Missing bundled fixed-boundary fixture for {case.case}")
    return input_path, wout_path


def _input_coeff(indata, name: str, n: int, m: int) -> float:
    return float(indata.indexed.get(name, {}).get((int(n), int(m)), 0.0))


@pytest.mark.parametrize(
    "case",
    FIXED_BOUNDARY_WOUT_CASES,
    ids=[case.case for case in FIXED_BOUNDARY_WOUT_CASES],
)
def test_vmec2000_fixed_boundary_wout_lcfs_matches_input_boundary(case: FixedBoundaryWoutCase) -> None:
    """Bundled VMEC2000 fixed-boundary WOUTs must preserve the input LCFS."""
    pytest.importorskip("netCDF4")

    input_path, wout_path = _case_paths(case)
    indata = read_indata(input_path)
    wout = read_wout(wout_path)

    assert bool(indata.get_bool("LFREEB", False)) is False
    assert bool(indata.get_bool("LASYM", False)) is False
    assert bool(wout.lasym) is False

    xm = np.asarray(wout.xm, dtype=int)
    xn = np.asarray(wout.xn, dtype=int)
    assert xm.shape == (int(wout.mnmax),)
    assert xn.shape == (int(wout.mnmax),)
    assert np.all(xn % int(wout.nfp) == 0)

    expected_rmnc = np.zeros(int(wout.mnmax), dtype=float)
    expected_zmns = np.zeros(int(wout.mnmax), dtype=float)
    represented_modes: set[tuple[int, int]] = set()
    for idx, (m_value, xn_value) in enumerate(zip(xm, xn, strict=True)):
        n_value = int(xn_value) // int(wout.nfp)
        m_int = int(m_value)
        represented_modes.add((n_value, m_int))
        expected_rmnc[idx] = _input_coeff(indata, "RBC", n_value, m_int)
        expected_zmns[idx] = _input_coeff(indata, "ZBS", n_value, m_int)

    active_input_modes = {
        (int(n), int(m))
        for family in ("RBC", "ZBS")
        for (n, m), value in indata.indexed.get(family, {}).items()
        if abs(float(value)) > 0.0 and int(m) < int(wout.mpol) and abs(int(n)) <= int(wout.ntor)
    }
    assert active_input_modes <= represented_modes

    np.testing.assert_allclose(
        np.asarray(wout.rmnc, dtype=float)[-1],
        expected_rmnc,
        rtol=0.0,
        atol=2.0e-12,
        err_msg=f"{case.case}: LCFS rmnc no longer matches input RBC",
    )
    np.testing.assert_allclose(
        np.asarray(wout.zmns, dtype=float)[-1],
        expected_zmns,
        rtol=0.0,
        atol=2.0e-12,
        err_msg=f"{case.case}: LCFS zmns no longer matches input ZBS",
    )
    np.testing.assert_allclose(np.asarray(wout.rmns, dtype=float), 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(wout.zmnc, dtype=float), 0.0, rtol=0.0, atol=1.0e-14)


@pytest.mark.parametrize(
    "case",
    FIXED_BOUNDARY_WOUT_CASES,
    ids=[case.case for case in FIXED_BOUNDARY_WOUT_CASES],
)
def test_vmec2000_fixed_boundary_wout_convergence_trace_and_profiles(case: FixedBoundaryWoutCase) -> None:
    """Converged fixed-boundary references should preserve VMEC physical scalars."""
    pytest.importorskip("netCDF4")

    input_path, wout_path = _case_paths(case)
    indata = read_indata(input_path)
    wout = read_wout(wout_path)
    ns = int(wout.ns)
    assert ns >= 3

    residual_components = np.asarray([wout.fsqr, wout.fsqz, wout.fsql], dtype=float)
    assert np.isfinite(residual_components).all()
    assert np.all(residual_components >= 0.0)
    assert float(np.linalg.norm(residual_components)) <= case.residual_rss_limit

    fsqt = np.asarray(wout.fsqt, dtype=float).reshape(-1)
    positive_trace = fsqt[fsqt > 0.0]
    assert positive_trace.size > 0
    assert np.isfinite(positive_trace).all()
    assert float(positive_trace[-1]) < 1.0e-2 * float(positive_trace[0])
    assert float(np.linalg.norm(residual_components)) <= 20.0 * float(positive_trace[-1])

    phi = np.asarray(wout.phi, dtype=float)
    assert phi.shape == (ns,)
    assert phi[0] == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(phi[-1], indata.get_float("PHIEDGE", 0.0), rtol=1.0e-12, atol=1.0e-12)
    if abs(float(phi[-1])) > 0.0:
        assert np.all(np.diff(phi) * np.sign(float(phi[-1])) >= -1.0e-13)

    vp = np.asarray(wout.vp, dtype=float)
    pres = np.asarray(wout.pres, dtype=float)
    assert vp.shape == (ns,)
    assert pres.shape == (ns,)
    assert np.isfinite(vp).all()
    assert np.isfinite(pres).all()
    assert float(wout.wb) > 0.0
    assert float(wout.wp) >= 0.0
    np.testing.assert_allclose(
        float(np.sum(vp[1:] * pres[1:]) / float(ns - 1)),
        float(wout.wp),
        rtol=1.0e-11,
        atol=1.0e-13,
        err_msg=f"{case.case}: wp no longer matches half-mesh pressure integral",
    )
    np.testing.assert_allclose(
        float(wout.betatotal),
        float(wout.wp) / float(wout.wb),
        rtol=2.0e-13,
        atol=1.0e-15,
        err_msg=f"{case.case}: betatotal no longer matches wp/wb",
    )

    if case.require_volume_scalar:
        volume_from_vp = 4.0 * np.pi**2 * float(np.sum(vp[1:])) / float(ns - 1)
        np.testing.assert_allclose(
            volume_from_vp,
            float(wout.volume_p),
            rtol=2.0e-6,
            atol=1.0e-12,
            err_msg=f"{case.case}: volume_p no longer matches integrated vp",
        )
        assert float(wout.Aminor_p) > 0.0
        assert float(wout.Rmajor_p) > 0.0
        np.testing.assert_allclose(
            float(wout.Rmajor_p) / float(wout.Aminor_p),
            float(wout.aspect),
            rtol=1.0e-13,
            atol=1.0e-13,
        )

    iotas = np.asarray(wout.iotas, dtype=float)
    assert iotas.shape == (ns,)
    assert np.isfinite(iotas).all()
    assert float(np.max(np.abs(iotas[1:]))) < 3.0
    if float(np.max(np.abs(iotas[1:]))) > 1.0e-8:
        assert float(np.min(iotas[1:])) * float(np.max(iotas[1:])) >= 0.0

    bdotb = np.asarray(wout.bdotb, dtype=float)
    assert bdotb.shape == (ns,)
    assert np.isfinite(bdotb).all()
    assert np.all(bdotb[1:] > 0.0)

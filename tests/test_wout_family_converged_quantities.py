from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.namelist import read_indata
from vmec_jax.wout import read_wout


@dataclass(frozen=True)
class FamilyWoutCase:
    family: str
    input_name: str
    wout_name: str
    ns: int
    mpol: int
    ntor: int
    nfp: int
    edge_iota: float
    mean_iota: float
    phi_edge: float
    wb: float
    wp: float
    fsq_rss_max: float
    edge_nonaxis_norm: float


CASES = (
    FamilyWoutCase(
        family="qi",
        input_name="input.nfp3_QI_fixed_resolution_final",
        wout_name="wout_nfp3_QI_fixed_resolution_final.nc",
        ns=151,
        mpol=6,
        ntor=6,
        nfp=3,
        edge_iota=-1.0132243043122409,
        mean_iota=-1.068058148348451,
        phi_edge=0.03295687308999999,
        wb=0.0007835998624712996,
        wp=0.0,
        fsq_rss_max=1.1e-13,
        edge_nonaxis_norm=0.6734193357360678,
    ),
    FamilyWoutCase(
        family="qi_stel_seed_3127",
        input_name="input.QI_stel_seed_3127",
        wout_name="wout_QI_stel_seed_3127.nc",
        ns=31,
        mpol=5,
        ntor=5,
        nfp=3,
        edge_iota=-0.1128393971420806,
        mean_iota=-0.09974818566649993,
        phi_edge=0.03141592653589792,
        wb=0.002598003916475921,
        wp=0.0,
        fsq_rss_max=1.1e-13,
        edge_nonaxis_norm=0.20909039728129608,
    ),
    FamilyWoutCase(
        family="qh",
        input_name="input.nfp4_QH_warm_start",
        wout_name="wout_nfp4_QH_warm_start.nc",
        ns=35,
        mpol=2,
        ntor=2,
        nfp=4,
        edge_iota=-1.1142302310459877,
        mean_iota=-1.0582056288813784,
        phi_edge=0.04800000000000003,
        wb=0.0037851930088982013,
        wp=0.0,
        fsq_rss_max=1.1e-13,
        edge_nonaxis_norm=0.32329524435599444,
    ),
    FamilyWoutCase(
        family="qa",
        input_name="input.LandremanPaul2021_QA_lowres",
        wout_name="wout_LandremanPaul2021_QA_lowres.nc",
        ns=50,
        mpol=8,
        ntor=8,
        nfp=2,
        edge_iota=0.41584395552394815,
        mean_iota=0.41930464310801124,
        phi_edge=0.08385727554,
        wb=0.007137916586384932,
        wp=0.0,
        fsq_rss_max=3.1e-13,
        edge_nonaxis_norm=0.35767632696877616,
    ),
    FamilyWoutCase(
        family="simple",
        input_name="input.circular_tokamak",
        wout_name="wout_circular_tokamak.nc",
        ns=17,
        mpol=8,
        ntor=0,
        nfp=1,
        edge_iota=0.2703125,
        mean_iota=0.575,
        phi_edge=67.86000000000001,
        wb=172.39494070759545,
        wp=0.0,
        fsq_rss_max=1.3e-14,
        edge_nonaxis_norm=0.0,
    ),
    FamilyWoutCase(
        family="finite_beta",
        input_name="input.shaped_tokamak_pressure",
        wout_name="wout_shaped_tokamak_pressure.nc",
        ns=51,
        mpol=12,
        ntor=0,
        nfp=1,
        edge_iota=0.7035,
        mean_iota=0.875,
        phi_edge=67.86000000000006,
        wb=137.17723177350166,
        wp=0.10191121859319738,
        fsq_rss_max=1.0e-14,
        edge_nonaxis_norm=0.0,
    ),
    FamilyWoutCase(
        family="finite_beta_3d",
        input_name="input.li383_low_res",
        wout_name="wout_li383_low_res.nc",
        ns=16,
        mpol=4,
        ntor=3,
        nfp=3,
        edge_iota=0.6569092765038941,
        mean_iota=0.5544911906253167,
        phi_edge=0.514386,
        wb=0.09601573538802485,
        wp=0.004092301309201012,
        fsq_rss_max=9.0e-7,
        edge_nonaxis_norm=0.3367099768497131,
    ),
    FamilyWoutCase(
        family="cth",
        input_name="input.cth_like_fixed_bdy",
        wout_name="wout_cth_like_fixed_bdy.nc",
        ns=15,
        mpol=5,
        ntor=0,
        nfp=5,
        edge_iota=0.5882703607477844,
        mean_iota=0.9049116971683636,
        phi_edge=-0.035,
        wb=0.0011262897384047997,
        wp=2.641365990298198e-06,
        fsq_rss_max=1.1e-14,
        edge_nonaxis_norm=0.0,
    ),
)


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _vmec_iotaf_from_iotas(iotas: np.ndarray) -> np.ndarray:
    if iotas.size < 3:
        return np.asarray(iotas, dtype=float)
    out = np.zeros_like(iotas, dtype=float)
    out[0] = 1.5 * iotas[1] - 0.5 * iotas[2]
    out[1:-1] = 0.5 * (iotas[1:-1] + iotas[2:])
    out[-1] = 1.5 * iotas[-1] - 0.5 * iotas[-2]
    return out


def _edge_nonaxisymmetric_geometry_norm(wout) -> float:
    nphys = np.asarray(wout.xn, dtype=int) // int(wout.nfp)
    nonaxis = nphys != 0
    rmnc_edge = np.asarray(wout.rmnc, dtype=float)[-1, nonaxis]
    zmns_edge = np.asarray(wout.zmns, dtype=float)[-1, nonaxis]
    return float(np.linalg.norm(rmnc_edge) + np.linalg.norm(zmns_edge))


@pytest.mark.parametrize("case", CASES, ids=[case.family for case in CASES])
def test_bundled_family_wout_converged_scalars_match_manifest(case: FamilyWoutCase) -> None:
    """Required no-executable regression checks on final VMEC2000 wout quantities."""
    pytest.importorskip("netCDF4")

    data_dir = _data_dir()
    assert (data_dir / case.input_name).exists()
    wout = read_wout(data_dir / case.wout_name)

    assert int(wout.ns) == case.ns
    assert int(wout.mpol) == case.mpol
    assert int(wout.ntor) == case.ntor
    assert int(wout.nfp) == case.nfp
    assert bool(wout.lasym) is False

    iotas = np.asarray(wout.iotas, dtype=float)
    scalars = np.asarray(
        [
            iotas[-1],
            float(np.mean(iotas[1:])),
            np.asarray(wout.phi, dtype=float)[-1],
            float(wout.wb),
            float(wout.wp),
            _edge_nonaxisymmetric_geometry_norm(wout),
        ]
    )
    expected = np.asarray(
        [
            case.edge_iota,
            case.mean_iota,
            case.phi_edge,
            case.wb,
            case.wp,
            case.edge_nonaxis_norm,
        ]
    )
    np.testing.assert_allclose(scalars, expected, rtol=2e-13, atol=1e-14)

    fsq_rss = float(np.sqrt(wout.fsqr * wout.fsqr + wout.fsqz * wout.fsqz + wout.fsql * wout.fsql))
    assert fsq_rss <= case.fsq_rss_max


@pytest.mark.parametrize("case", CASES, ids=[case.family for case in CASES])
def test_bundled_family_wout_profiles_are_vmec_consistent(case: FamilyWoutCase) -> None:
    """Check final mesh/profile identities without rerunning VMEC or vmec_jax."""
    pytest.importorskip("netCDF4")

    data_dir = _data_dir()
    indata = read_indata(data_dir / case.input_name)
    assert not bool(indata.get_bool("LRFP", False))

    wout = read_wout(data_dir / case.wout_name)
    iotas = np.asarray(wout.iotas, dtype=float)
    iotaf = np.asarray(wout.iotaf, dtype=float)
    np.testing.assert_allclose(iotaf, _vmec_iotaf_from_iotas(iotas), rtol=1e-14, atol=1e-14)

    hs = 1.0 / float(wout.ns - 1)
    wp_calc = hs * float(np.sum(np.asarray(wout.vp, dtype=float)[1:] * np.asarray(wout.pres, dtype=float)[1:]))
    np.testing.assert_allclose(wp_calc, float(wout.wp), rtol=1e-13, atol=1e-14)

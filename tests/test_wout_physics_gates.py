from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.namelist import read_indata
from vmec_jax.wout import read_wout
from vmec_jax.wout_schema import assert_main_modes_match_wout


@dataclass(frozen=True)
class BundledWoutPhysicsCase:
    case: str
    residual_rss_limit: float
    require_aspect: bool = True


BUNDLED_WOUT_CASES = (
    BundledWoutPhysicsCase("DSHAPE", residual_rss_limit=1.0e-10),
    # This VMEC2000-style fixture intentionally omits populated aspect scalars.
    BundledWoutPhysicsCase("LandremanPaul2021_QA_lowres", residual_rss_limit=1.0e-10, require_aspect=False),
    BundledWoutPhysicsCase("QI_stel_seed_3127", residual_rss_limit=1.0e-10),
    BundledWoutPhysicsCase("basic_non_stellsym_simsopt", residual_rss_limit=5.0e-10),
    BundledWoutPhysicsCase("circular_tokamak", residual_rss_limit=1.0e-12),
    BundledWoutPhysicsCase("cth_like_fixed_bdy", residual_rss_limit=1.0e-12),
    BundledWoutPhysicsCase("li383_low_res", residual_rss_limit=1.0e-5),
    BundledWoutPhysicsCase("nfp3_QI_fixed_resolution_final", residual_rss_limit=1.0e-10),
    BundledWoutPhysicsCase("nfp4_QH_warm_start", residual_rss_limit=1.0e-10),
    BundledWoutPhysicsCase("purely_toroidal_field", residual_rss_limit=1.0e-12),
    BundledWoutPhysicsCase("shaped_tokamak_pressure", residual_rss_limit=1.0e-12),
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


def _assert_shape(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    assert np.asarray(arr).shape == shape, f"{name} shape mismatch"


@pytest.mark.parametrize("case", BUNDLED_WOUT_CASES, ids=[case.case for case in BUNDLED_WOUT_CASES])
def test_bundled_converged_wout_physics_gates(case: BundledWoutPhysicsCase) -> None:
    """Cheap no-executable gates for final VMEC2000 equilibria bundled with the repo."""
    pytest.importorskip("netCDF4")

    data_dir = _data_dir()
    input_path = data_dir / f"input.{case.case}"
    wout_path = data_dir / f"wout_{case.case}.nc"
    if not input_path.exists() or not wout_path.exists():
        pytest.skip(f"Missing bundled input/wout for case={case.case}")

    indata = read_indata(input_path)
    wout = read_wout(wout_path)
    assert_main_modes_match_wout(wout=wout)

    ns = int(wout.ns)
    mnmax = int(wout.mnmax)
    mnmax_nyq = int(wout.mnmax_nyq)
    assert ns >= 3
    assert mnmax > 0
    assert mnmax_nyq > 0
    assert int(wout.signgs) in (-1, 1)

    axis_shape = (int(wout.ntor) + 1,)
    for name in ("raxis_cc", "zaxis_cs", "raxis_cs", "zaxis_cc"):
        axis = np.asarray(getattr(wout, name), dtype=float)
        _assert_shape(name, axis, axis_shape)
        assert np.isfinite(axis).all(), name
    assert float(np.asarray(wout.raxis_cc, dtype=float)[0]) > 0.0

    for name in ("rmnc", "zmns", "lmns", "rmns", "zmnc", "lmnc"):
        _assert_shape(name, getattr(wout, name), (ns, mnmax))
        assert np.isfinite(np.asarray(getattr(wout, name), dtype=float)).all(), name
    for name in ("gmnc", "bmnc", "bsupumnc", "bsupvmnc", "bsubumnc", "bsubvmnc"):
        _assert_shape(name, getattr(wout, name), (ns, mnmax_nyq))
        assert np.isfinite(np.asarray(getattr(wout, name), dtype=float)).all(), name
    if not bool(wout.lasym):
        for name in ("rmns", "zmnc", "lmnc", "raxis_cs", "zaxis_cc"):
            np.testing.assert_allclose(getattr(wout, name), 0.0, atol=0.0)
    else:
        asymmetry = sum(
            float(np.max(np.abs(np.asarray(getattr(wout, name), dtype=float))))
            for name in ("rmns", "zmnc", "lmnc", "raxis_cs", "zaxis_cc")
        )
        assert asymmetry > 0.0

    residual_components = np.asarray([wout.fsqr, wout.fsqz, wout.fsql], dtype=float)
    assert np.isfinite(residual_components).all()
    assert np.all(residual_components >= 0.0)
    residual_rss = float(np.linalg.norm(residual_components))
    assert residual_rss <= case.residual_rss_limit

    iotas = np.asarray(wout.iotas, dtype=float)
    iotaf = np.asarray(wout.iotaf, dtype=float)
    _assert_shape("iotas", iotas, (ns,))
    _assert_shape("iotaf", iotaf, (ns,))
    assert np.isfinite(iotas).all()
    assert np.isfinite(iotaf).all()
    assert iotas[0] == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(iotaf, _vmec_iotaf_from_iotas(iotas), rtol=1.0e-13, atol=1.0e-13)

    physical_iota = iotas[1:]
    assert np.max(np.abs(physical_iota)) < 3.0
    assert np.max(np.abs(np.diff(physical_iota))) < 0.25
    if np.max(np.abs(physical_iota)) > 1.0e-8:
        assert np.min(physical_iota) * np.max(physical_iota) >= 0.0

    phi = np.asarray(wout.phi, dtype=float)
    _assert_shape("phi", phi, (ns,))
    assert np.isfinite(phi).all()
    assert phi[0] == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(phi[-1], indata.get_float("PHIEDGE", 0.0), rtol=1.0e-12, atol=1.0e-12)
    if abs(phi[-1]) > 0.0:
        assert np.all(np.diff(phi) * np.sign(phi[-1]) >= -1.0e-13)

    vp = np.asarray(wout.vp, dtype=float)
    pres = np.asarray(wout.pres, dtype=float)
    equif = np.asarray(wout.equif, dtype=float)
    _assert_shape("vp", vp, (ns,))
    _assert_shape("pres", pres, (ns,))
    _assert_shape("equif", equif, (ns,))
    assert np.isfinite(vp).all()
    assert np.isfinite(pres).all()
    assert np.isfinite(equif).all()
    assert float(wout.wb) > 0.0
    assert float(wout.wp) >= 0.0
    wp_from_profile = float(np.sum(vp[1:] * pres[1:]) / float(ns - 1))
    np.testing.assert_allclose(wp_from_profile, float(wout.wp), rtol=1.0e-11, atol=1.0e-13)

    aspect_scalars = np.asarray([wout.Aminor_p, wout.Rmajor_p, wout.aspect, wout.volume_p], dtype=float)
    assert np.isfinite(aspect_scalars).all()
    if case.require_aspect:
        assert float(wout.Aminor_p) > 0.0
        assert float(wout.Rmajor_p) > 0.0
        assert float(wout.volume_p) > 0.0
        assert 1.0 < float(wout.aspect) < 20.0
        np.testing.assert_allclose(wout.Rmajor_p / wout.Aminor_p, wout.aspect, rtol=1.0e-13, atol=1.0e-13)
    else:
        np.testing.assert_allclose(aspect_scalars, 0.0, atol=0.0)

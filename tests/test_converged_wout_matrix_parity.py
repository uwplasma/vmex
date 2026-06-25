from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.free_boundary import prepare_mgrid_for_config
from vmec_jax.io.wout.schema import assert_main_modes_match_wout
from vmec_jax.namelist import read_indata
from vmec_jax.wout import read_wout


@dataclass(frozen=True)
class ConvergedWoutMatrixCase:
    case: str
    input_path: str
    wout_path: str
    lfreeb: bool
    axisymmetric: bool
    lasym: bool
    multigrid: bool
    finite_beta: bool
    residual_rss_limit: float


CONVERGED_WOUT_MATRIX_CASES = (
    ConvergedWoutMatrixCase(
        case="fixed_axisym_single_cth_like",
        input_path="examples/data/input.cth_like_fixed_bdy",
        wout_path="examples/data/wout_cth_like_fixed_bdy.nc",
        lfreeb=False,
        axisymmetric=True,
        lasym=False,
        multigrid=False,
        finite_beta=True,
        residual_rss_limit=1.0e-12,
    ),
    ConvergedWoutMatrixCase(
        case="fixed_axisym_multigrid_shaped_pressure",
        input_path="examples/data/input.shaped_tokamak_pressure",
        wout_path="examples/data/wout_shaped_tokamak_pressure.nc",
        lfreeb=False,
        axisymmetric=True,
        lasym=False,
        multigrid=True,
        finite_beta=True,
        residual_rss_limit=1.0e-12,
    ),
    ConvergedWoutMatrixCase(
        case="fixed_axisym_lasym_single_updown",
        input_path="examples_single_grid/data/input.up_down_asymmetric_tokamak",
        wout_path="examples_single_grid/data/wout_up_down_asymmetric_tokamak_reference.nc",
        lfreeb=False,
        axisymmetric=True,
        lasym=True,
        multigrid=False,
        finite_beta=False,
        residual_rss_limit=1.0e-10,
    ),
    ConvergedWoutMatrixCase(
        case="fixed_nonaxis_multigrid_qa",
        input_path="examples/data/input.LandremanPaul2021_QA_lowres",
        wout_path="examples/data/wout_LandremanPaul2021_QA_lowres.nc",
        lfreeb=False,
        axisymmetric=False,
        lasym=False,
        multigrid=True,
        finite_beta=False,
        residual_rss_limit=1.0e-12,
    ),
    ConvergedWoutMatrixCase(
        case="fixed_nonaxis_lasym_multigrid_basic_non_stellsym",
        input_path="examples/data/input.basic_non_stellsym_simsopt",
        wout_path="examples/data/wout_basic_non_stellsym_simsopt.nc",
        lfreeb=False,
        axisymmetric=False,
        lasym=True,
        multigrid=True,
        finite_beta=False,
        residual_rss_limit=5.0e-10,
    ),
    ConvergedWoutMatrixCase(
        case="fixed_nonaxis_lasym_single_basic_non_stellsym_pressure",
        input_path="examples_single_grid/data/input.basic_non_stellsym_pressure",
        wout_path="examples_single_grid/data/wout_basic_non_stellsym_pressure_reference.nc",
        lfreeb=False,
        axisymmetric=False,
        lasym=True,
        multigrid=False,
        finite_beta=True,
        residual_rss_limit=2.0e-10,
    ),
    ConvergedWoutMatrixCase(
        case="free_nonaxis_single_cth_like",
        input_path="examples_single_grid/data/input.cth_like_free_bdy",
        wout_path="examples_single_grid/data/wout_cth_like_free_bdy.nc",
        lfreeb=True,
        axisymmetric=False,
        lasym=False,
        multigrid=False,
        finite_beta=True,
        residual_rss_limit=1.0e-12,
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _is_multigrid_ns_array(indata) -> bool:
    ns_array = _as_sequence(indata.get("NS_ARRAY"))
    active = [int(v) for v in ns_array if int(v) > 0]
    return len(active) > 1


def _vmec_iotaf_from_iotas(iotas: np.ndarray) -> np.ndarray:
    if iotas.size < 3:
        return np.asarray(iotas, dtype=float)
    out = np.zeros_like(iotas, dtype=float)
    out[0] = 1.5 * iotas[1] - 0.5 * iotas[2]
    out[1:-1] = 0.5 * (iotas[1:-1] + iotas[2:])
    out[-1] = 1.5 * iotas[-1] - 0.5 * iotas[-2]
    return out


def test_converged_wout_matrix_covers_required_vmec2000_classes() -> None:
    cases = CONVERGED_WOUT_MATRIX_CASES
    assert any(not c.lfreeb for c in cases)
    assert any(c.lfreeb for c in cases)
    assert any(c.axisymmetric for c in cases)
    assert any(not c.axisymmetric for c in cases)
    assert any(c.lasym for c in cases)
    assert any(not c.lasym for c in cases)
    assert any(c.finite_beta for c in cases)
    assert any(not c.finite_beta for c in cases)
    assert any(c.multigrid for c in cases)
    assert any(not c.multigrid for c in cases)
    assert any((not c.lfreeb) and c.axisymmetric and c.lasym for c in cases)
    assert any((not c.lfreeb) and (not c.axisymmetric) and c.lasym for c in cases)
    assert any((not c.lfreeb) and (not c.axisymmetric) and c.lasym and c.finite_beta for c in cases)
    assert any(c.lfreeb and (not c.axisymmetric) for c in cases)


@pytest.mark.parametrize(
    "case",
    CONVERGED_WOUT_MATRIX_CASES,
    ids=[case.case for case in CONVERGED_WOUT_MATRIX_CASES],
)
def test_bundled_converged_wout_matrix_physics_gates(case: ConvergedWoutMatrixCase) -> None:
    """CI-safe gates over representative converged VMEC2000 wout fixtures."""
    pytest.importorskip("netCDF4")

    repo_root = _repo_root()
    input_path = repo_root / case.input_path
    wout_path = repo_root / case.wout_path
    if not input_path.exists() or not wout_path.exists():
        pytest.skip(f"Missing bundled matrix fixture: {case.case}")

    indata = read_indata(input_path)
    cfg, _ = load_config(input_path)
    wout = read_wout(wout_path)
    assert_main_modes_match_wout(wout=wout)

    assert bool(indata.get_bool("LFREEB", False)) is case.lfreeb
    assert bool(cfg.lfreeb) is case.lfreeb
    assert bool(indata.get_bool("LASYM", False)) is case.lasym
    assert bool(int(indata.get_int("NTOR", 0)) == 0) is case.axisymmetric
    assert _is_multigrid_ns_array(indata) is case.multigrid
    if case.lfreeb:
        assert cfg.mgrid_file.upper() != "NONE"
        assert Path(cfg.mgrid_file).exists()
        prepared = prepare_mgrid_for_config(cfg, load_fields=False, strict=True)
        assert prepared is not None
        assert int(prepared.metadata.nfp) == int(wout.nfp)
        assert int(prepared.metadata.nextcur) == len(prepared.extcur)
        assert any(abs(current) > 0.0 for current in prepared.extcur)
        assert int(indata.get_int("NVACSKIP", 0)) > 0
    else:
        assert bool(cfg.lfreeb) is False

    assert int(wout.nfp) == int(indata.get_int("NFP", int(wout.nfp)))
    assert int(wout.ntor) == int(indata.get_int("NTOR", int(wout.ntor)))
    assert bool(wout.lasym) is case.lasym
    assert int(wout.ns) >= 3
    assert int(wout.mnmax) > 0
    assert int(wout.mnmax_nyq) > 0

    residual_components = np.asarray([wout.fsqr, wout.fsqz, wout.fsql], dtype=float)
    assert np.isfinite(residual_components).all()
    assert float(np.linalg.norm(residual_components)) <= case.residual_rss_limit

    phi = np.asarray(wout.phi, dtype=float)
    assert phi.shape == (int(wout.ns),)
    assert np.isfinite(phi).all()
    np.testing.assert_allclose(phi[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(phi[-1], indata.get_float("PHIEDGE", 0.0), rtol=1.0e-12, atol=1.0e-12)

    iotas = np.asarray(wout.iotas, dtype=float)
    iotaf = np.asarray(wout.iotaf, dtype=float)
    assert iotas.shape == (int(wout.ns),)
    assert iotaf.shape == (int(wout.ns),)
    assert np.isfinite(iotas).all()
    assert np.isfinite(iotaf).all()
    np.testing.assert_allclose(iotaf, _vmec_iotaf_from_iotas(iotas), rtol=1.0e-13, atol=1.0e-13)

    for name in ("rmnc", "zmns", "lmns", "gmnc", "bmnc", "bsupumnc", "bsupvmnc", "bsubumnc", "bsubvmnc"):
        arr = np.asarray(getattr(wout, name), dtype=float)
        assert arr.shape[0] == int(wout.ns), name
        assert np.isfinite(arr).all(), name

    asymmetric_norm = float(
        np.linalg.norm(np.asarray(wout.rmns, dtype=float))
        + np.linalg.norm(np.asarray(wout.zmnc, dtype=float))
        + np.linalg.norm(np.asarray(wout.lmnc, dtype=float))
    )
    if case.lasym:
        assert asymmetric_norm > 0.0
    else:
        assert asymmetric_norm == pytest.approx(0.0, abs=1.0e-14)

    asymmetric_nyquist_norm = 0.0
    for name in ("gmns", "bmns", "bsupumns", "bsupvmns", "bsubumns", "bsubvmns"):
        arr = np.asarray(getattr(wout, name), dtype=float)
        assert arr.shape[0] == int(wout.ns), name
        assert np.isfinite(arr).all(), name
        asymmetric_nyquist_norm += float(np.linalg.norm(arr))
    if case.lasym:
        assert asymmetric_nyquist_norm > 0.0
    else:
        assert asymmetric_nyquist_norm == pytest.approx(0.0, abs=1.0e-14)

    assert float(wout.wb) > 0.0
    pressure_norm = float(np.linalg.norm(np.asarray(wout.pres, dtype=float)))
    beta_scalars = np.asarray([wout.betatotal, wout.betapol, wout.betator, wout.betaxis], dtype=float)
    assert np.isfinite(beta_scalars).all()
    np.testing.assert_allclose(wout.betatotal, wout.wp / wout.wb, rtol=2.0e-13, atol=1.0e-15)
    if case.finite_beta:
        assert pressure_norm > 0.0
        assert float(wout.wp) > 0.0
        assert np.all(beta_scalars > 0.0)
    else:
        assert pressure_norm == pytest.approx(0.0, abs=1.0e-14)
        assert float(wout.wp) == pytest.approx(0.0, abs=1.0e-14)
        np.testing.assert_allclose(beta_scalars, 0.0, rtol=0.0, atol=0.0)
    assert np.isfinite([wout.Aminor_p, wout.Rmajor_p, wout.aspect, wout.volume_p]).all()

    mercier_terms = {
        "DMerc": np.asarray(wout.DMerc, dtype=float),
        "Dshear": np.asarray(wout.Dshear, dtype=float),
        "Dwell": np.asarray(wout.Dwell, dtype=float),
        "Dcurr": np.asarray(wout.Dcurr, dtype=float),
        "Dgeod": np.asarray(wout.Dgeod, dtype=float),
    }
    for name, values in mercier_terms.items():
        assert values.shape == (int(wout.ns),), name
        assert np.isfinite(values).all(), name
    np.testing.assert_allclose(
        mercier_terms["DMerc"],
        mercier_terms["Dshear"] + mercier_terms["Dwell"] + mercier_terms["Dcurr"] + mercier_terms["Dgeod"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    for name in ("jdotb", "bdotb", "bdotgradv", "buco", "bvco", "jcuru", "jcurv"):
        values = np.asarray(getattr(wout, name), dtype=float)
        assert values.shape == (int(wout.ns),), name
        assert np.isfinite(values).all(), name
    assert np.all(np.asarray(wout.bdotb, dtype=float)[1:] > 0.0)

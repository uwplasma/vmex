from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.config import load_config
from vmec_jax.fourier import eval_fourier
from vmec_jax.grids import AngleGrid
from vmec_jax.nyquist import nyquist_basis_from_wout
from vmec_jax.static import build_static
from vmec_jax.kernels.bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.wout import read_wout, state_from_wout


@dataclass(frozen=True)
class ContravariantFieldCase:
    case_name: str
    input_rel: str
    wout_rel: str
    ntheta: int
    nzeta: int
    rel_rms_limit: float
    max_abs_limit: float


CONTRAVARIANT_FIELD_CASES = (
    ContravariantFieldCase(
        case_name="finite_beta_axisym",
        input_rel="examples/data/input.shaped_tokamak_pressure",
        wout_rel="examples/data/wout_shaped_tokamak_pressure.nc",
        ntheta=18,
        nzeta=1,
        rel_rms_limit=2.0e-6,
        max_abs_limit=5.0e-6,
    ),
    ContravariantFieldCase(
        case_name="qh_3d",
        input_rel="examples/data/input.nfp4_QH_warm_start",
        wout_rel="examples/data/wout_nfp4_QH_warm_start.nc",
        ntheta=18,
        nzeta=12,
        rel_rms_limit=2.0e-3,
        max_abs_limit=1.0e-2,
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _vmec_reduced_angle_grid(static, *, nfp: int) -> AngleGrid:
    """Grid used by VMEC's reduced theta/zeta synthesis tables."""
    trig = static.trig_vmec
    ntheta1 = int(trig.ntheta1)
    ntheta3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    theta = np.arange(ntheta3, dtype=float) * (2.0 * np.pi / float(ntheta1))
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    return AngleGrid(theta=theta, zeta=zeta, nfp=int(nfp))


def _relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.sqrt(np.mean(np.asarray(b, dtype=float) ** 2))), 1.0e-30)
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)) / scale)


@pytest.mark.parametrize(
    "case",
    CONTRAVARIANT_FIELD_CASES,
    ids=[case.case_name for case in CONTRAVARIANT_FIELD_CASES],
)
def test_bundled_wout_contravariant_field_matches_flux_coordinate_identity(
    case: ContravariantFieldCase,
) -> None:
    """VMEC WOUT ``bsup*`` should close with the flux-coordinate B representation."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    enable_x64(True)

    root = _repo_root()
    cfg, _indata = load_config(str(root / case.input_rel))
    wout = read_wout(root / case.wout_rel)
    cfg = replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
        ntheta=case.ntheta,
        nzeta=case.nzeta,
    )
    static = build_static(cfg)

    reconstructed = vmec_bcovar_half_mesh_from_wout(
        state=state_from_wout(wout),
        static=static,
        wout=wout,
        use_vmec_synthesis=True,
    )
    grid = _vmec_reduced_angle_grid(static, nfp=int(wout.nfp))
    basis_nyq = nyquist_basis_from_wout(wout=wout, grid=grid)

    references = {
        "bsupu": np.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq), dtype=float),
        "bsupv": np.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq), dtype=float),
    }
    actuals = {
        "bsupu": np.asarray(reconstructed.bsupu, dtype=float),
        "bsupv": np.asarray(reconstructed.bsupv, dtype=float),
    }

    for name, actual in actuals.items():
        reference = references[name]
        assert actual.shape == reference.shape, f"{case.case_name}.{name}"
        rel_rms = _relative_rms(actual[1:], reference[1:])
        max_abs = float(np.max(np.abs(actual[1:] - reference[1:])))
        assert rel_rms < case.rel_rms_limit, f"{case.case_name}.{name}: rel_rms={rel_rms:.3e}"
        assert max_abs < case.max_abs_limit, f"{case.case_name}.{name}: max_abs={max_abs:.3e}"

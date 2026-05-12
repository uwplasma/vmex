from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.field import b_cartesian_from_state
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


CASES = (
    ("qh", "input.nfp4_QH_warm_start", "wout_nfp4_QH_warm_start.nc"),
    ("qa", "input.LandremanPaul2021_QA_lowres", "wout_LandremanPaul2021_QA_lowres.nc"),
    ("qi", "input.nfp3_QI_fixed_resolution_final", "wout_nfp3_QI_fixed_resolution_final.nc"),
)


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data"


def _small_static_aligned_to_wout(cfg, wout):
    theta = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    cfg_small = replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
        ntheta=int(theta.size),
        nzeta=int(zeta.size),
    )
    return build_static(cfg_small, grid=AngleGrid(theta=theta, zeta=zeta, nfp=int(wout.nfp)))


def _relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.sqrt(np.mean(np.asarray(b, dtype=float) ** 2))), 1.0e-30)
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)) / scale)


@pytest.mark.parametrize(("case_name", "input_name", "wout_name"), CASES)
def test_bundled_wout_stored_bsup_cartesian_magnitude_matches_bmnc(
    case_name: str,
    input_name: str,
    wout_name: str,
) -> None:
    """Stored VMEC2000 ``bsup*`` fields should reconstruct the same ``|B|`` as ``bmnc``."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    data_dir = _data_dir()
    cfg, _indata = load_config(str(data_dir / input_name))
    wout = read_wout(data_dir / wout_name)
    state = state_from_wout(wout)
    static = _small_static_aligned_to_wout(cfg, wout)

    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // int(wout.nfp)))
    grid_nyq = AngleGrid(theta=np.asarray(static.grid.theta), zeta=np.asarray(static.grid.zeta), nfp=int(wout.nfp))
    basis_nyq = build_helical_basis(modes_nyq, grid_nyq)
    bmag_ref = np.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq), dtype=float)

    surface_indices = (int(wout.ns) // 2, -1)
    rel_errors = []
    for surface_index in surface_indices:
        b_cart = np.asarray(
            b_cartesian_from_state(
                state,
                static,
                wout=wout,
                s_index=surface_index,
                use_wout_bsup=True,
            ),
            dtype=float,
        )
        radial_index = surface_index if surface_index >= 0 else int(wout.ns) + surface_index
        bmag_cart = np.linalg.norm(b_cart, axis=-1)
        assert b_cart.shape == (int(static.grid.ntheta), int(static.grid.nzeta), 3)
        assert np.all(np.isfinite(bmag_cart))
        assert np.all(bmag_ref[radial_index] > 0.0)
        rel_errors.append(_relative_rms(bmag_cart, bmag_ref[radial_index]))

    assert max(rel_errors) < 3.0e-3, f"{case_name}: rel_errors={rel_errors}"

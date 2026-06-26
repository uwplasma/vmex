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
    ("qh", "examples/data/input.nfp4_QH_warm_start", "examples/data/wout_nfp4_QH_warm_start.nc", 3.0e-3),
    (
        "qa",
        "examples/data/input.LandremanPaul2021_QA_lowres",
        "examples/data/wout_LandremanPaul2021_QA_lowres.nc",
        3.0e-3,
    ),
    (
        "qi",
        "examples/data/input.nfp3_QI_fixed_resolution_final",
        "examples/data/wout_nfp3_QI_fixed_resolution_final.nc",
        3.0e-3,
    ),
    (
        "shaped_pressure",
        "examples/data/input.shaped_tokamak_pressure",
        "examples/data/wout_shaped_tokamak_pressure.nc",
        3.0e-3,
    ),
    ("li383", "examples/data/input.li383_low_res", "examples/data/wout_li383_low_res.nc", 5.0e-3),
    (
        "basic_non_stellsym",
        "examples/data/input.basic_non_stellsym_simsopt",
        "examples/data/wout_basic_non_stellsym_simsopt.nc",
        1.2e-2,
    ),
    (
        "basic_non_stellsym_pressure_single_grid",
        "examples/data/single_grid/input.basic_non_stellsym_pressure",
        "examples/data/single_grid/wout_basic_non_stellsym_pressure_reference.nc",
        1.2e-2,
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


@pytest.mark.parametrize(("case_name", "input_name", "wout_name", "max_rel_rms"), CASES)
def test_bundled_wout_stored_bsup_cartesian_magnitude_matches_bmnc(
    case_name: str,
    input_name: str,
    wout_name: str,
    max_rel_rms: float,
) -> None:
    """Stored VMEC2000 ``bsup*`` fields should reconstruct the same ``|B|`` as ``bmnc``.

    This no-solve gate covers stellarator-symmetric, finite-beta, LI383, and
    LASYM=true fixtures.  The LASYM/nonaxisymmetric fixture is intentionally
    retained with its own tolerance because it exercises the sine magnetic
    channels that are absent from stellarator-symmetric wouts.
    """
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    repo_root = _repo_root()
    input_path = repo_root / input_name
    wout_path = repo_root / wout_name
    if not input_path.exists() or not wout_path.exists():
        pytest.skip(f"Missing bundled B-field fixture: {case_name}")

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
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

    assert max(rel_errors) < max_rel_rms, f"{case_name}: rel_errors={rel_errors}"

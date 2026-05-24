from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.external_fields import from_essos_coils
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.vmec2000_exec import find_vmec2000_exec, run_xvmec2000
from vmec_jax.wout import read_wout


ROOT = Path(__file__).resolve().parents[1]
LPQA_INPUT = ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
LPQA_COILS = Path("/Users/rogeriojorge/local/ESSOS_mgrid_pr/examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json")
FINITE_PRESSURE_SCALE = 34.46233666638


def _load_lpqa_essos_coils():
    essos_coils = pytest.importorskip("essos.coils")
    if not LPQA_COILS.exists():
        pytest.skip(f"missing local ESSOS Landreman-Paul QA coils: {LPQA_COILS}")
    coils = essos_coils.Coils_from_json(str(LPQA_COILS))
    if not hasattr(coils, "to_mgrid"):
        pytest.skip("ESSOS Coils.to_mgrid is not available; use ESSOS PR #33 or newer")
    return coils


def _write_lpqa_mgrid(coils, path: Path) -> Path:
    coils.to_mgrid(
        path,
        nr=12,
        nz=12,
        nphi=6,
        rmin=5.0,
        rmax=15.0,
        zmin=-5.0,
        zmax=5.0,
        nfp=int(coils.nfp),
    )
    return path


def _write_freeb_input(
    path: Path,
    *,
    mgrid_file: str | Path,
    niter: int = 2,
    ftol: float = 1.0e-8,
    pressure_scale: float = FINITE_PRESSURE_SCALE,
) -> Path:
    indata = deepcopy(read_indata(LPQA_INPUT))
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": str(mgrid_file),
            "EXTCUR": [1.0],
            "NS_ARRAY": [12],
            "NITER_ARRAY": [int(niter)],
            "FTOL_ARRAY": [float(ftol)],
            "NITER": int(niter),
            "FTOL": float(ftol),
            "MPOL": 4,
            "NTOR": 4,
            "NZETA": 6,
            "NTHETA": 0,
            "NVACSKIP": 6,
            "PRES_SCALE": float(pressure_scale),
            "AM": [1.0, -1.0],
        }
    )
    write_indata(path, indata)
    return path


def _run_vmec_jax_freeb(input_path: Path, *, direct_params=None):
    from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run

    kwargs = {}
    if direct_params is not None:
        kwargs.update(
            {
                "external_field_provider_kind": "direct_coils",
                "external_field_provider_params": direct_params,
            }
        )
    run = run_free_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="parity",
        multigrid_use_input_niter=True,
        verbose=False,
        jit_forces=False,
        **kwargs,
    )
    wout_path = input_path.with_name(f"wout_{input_path.name.removeprefix('input.')}.nc")
    write_wout_from_fixed_boundary_run(wout_path, run)
    return run, read_wout(wout_path)


def _rel_rms(got, ref) -> float:
    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    assert got_arr.shape == ref_arr.shape
    denom = float(np.sqrt(np.mean(ref_arr**2)))
    diff = float(np.sqrt(np.mean((got_arr - ref_arr) ** 2)))
    return diff / denom if denom > 0.0 else diff


def test_essos_direct_coil_free_boundary_matches_generated_mgrid_backend(tmp_path: Path) -> None:
    """The new direct Biot-Savart backend must match the mgrid compatibility backend.

    This is intentionally a short low-resolution equilibrium smoke.  It proves
    that the same ESSOS coil set can drive the `vmec_jax` free-boundary path
    through either an ESSOS-generated mgrid file or direct differentiable coil
    sampling without changing the resulting VMEC state.
    """

    pytest.importorskip("jax")
    coils = _load_lpqa_essos_coils()
    mgrid = _write_lpqa_mgrid(coils, tmp_path / "mgrid_lpqa_from_essos.nc")
    mgrid_input = _write_freeb_input(tmp_path / "input.lpqa_mgrid", mgrid_file=mgrid)
    direct_input = _write_freeb_input(tmp_path / "input.lpqa_direct", mgrid_file="DIRECT_COILS")
    direct_params = from_essos_coils(coils, chunk_size=256)

    _run_mgrid, wout_mgrid = _run_vmec_jax_freeb(mgrid_input)
    _run_direct, wout_direct = _run_vmec_jax_freeb(direct_input, direct_params=direct_params)

    for name in ("rmnc", "zmns", "lmns", "iotas", "iotaf"):
        np.testing.assert_allclose(getattr(wout_direct, name), getattr(wout_mgrid, name), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(wout_direct.aspect, wout_mgrid.aspect, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(wout_direct.wb, wout_mgrid.wb, rtol=1.0e-12, atol=1.0e-12)
    assert float(wout_direct.wp) > 0.0
    np.testing.assert_allclose(wout_direct.wp, wout_mgrid.wp, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.vmec2000
@pytest.mark.xfail(
    reason=(
        "Generated ESSOS-mgrid VMEC2000 free-boundary trace/WOUT parity is not "
        "bounded yet; this optional gate captures the current gap while the "
        "direct-coil provider path is being developed."
    ),
    strict=False,
)
def test_vmec2000_generated_mgrid_free_boundary_matches_vmec_jax_and_direct_coils(tmp_path: Path) -> None:
    """Optional three-way parity gate for generated-mgrid/free-boundary cases.

    Run with `VMEC2000_INTEGRATION=1`.  The intended passing condition is:

    1. VMEC2000 free-boundary from ESSOS-generated mgrid,
    2. `vmec_jax` free-boundary from the same mgrid,
    3. `vmec_jax` free-boundary from direct ESSOS/JAX Biot-Savart coils,

    all produce matching bounded traces/equilibria.
    """

    if os.environ.get("VMEC2000_INTEGRATION", "0") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 executable parity tests")
    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")

    pytest.importorskip("jax")
    coils = _load_lpqa_essos_coils()
    mgrid = _write_lpqa_mgrid(coils, tmp_path / "mgrid_lpqa_from_essos.nc")
    mgrid_input = _write_freeb_input(tmp_path / "input.lpqa_mgrid", mgrid_file=mgrid)
    direct_input = _write_freeb_input(tmp_path / "input.lpqa_direct", mgrid_file="DIRECT_COILS")
    direct_params = from_essos_coils(coils, chunk_size=256)

    vmec2000 = run_xvmec2000(mgrid_input, exec_path=exe, workdir=tmp_path / "vmec2000", timeout_s=90, keep_workdir=True)
    assert vmec2000.stages and vmec2000.stages[-1].rows
    vmec_row = vmec2000.stages[-1].rows[-1]

    run_mgrid, wout_mgrid = _run_vmec_jax_freeb(mgrid_input)
    _run_direct, wout_direct = _run_vmec_jax_freeb(direct_input, direct_params=direct_params)
    diag = run_mgrid.result.diagnostics

    np.testing.assert_allclose(wout_direct.rmnc, wout_mgrid.rmnc, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(wout_direct.zmns, wout_mgrid.zmns, rtol=1.0e-12, atol=1.0e-12)
    assert _rel_rms(wout_direct.iotas, wout_mgrid.iotas) < 1.0e-12

    np.testing.assert_allclose(float(diag["final_fsqr"]), vmec_row.fsqr, rtol=2.0e-2, atol=1.0e-12)
    np.testing.assert_allclose(float(diag["final_fsqz"]), vmec_row.fsqz, rtol=2.0e-2, atol=1.0e-12)
    np.testing.assert_allclose(float(diag["final_fsql"]), vmec_row.fsql, rtol=2.0e-2, atol=1.0e-12)

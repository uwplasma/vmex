from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.free_boundary_validation import free_boundary_response_metrics
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.vmec2000_exec import _patch_indata, find_vmec2000_exec, run_xvmec2000


pytestmark = pytest.mark.vmec2000

VMEC2000_STAGE_TRACE_SINGLE_NS = 13
VMEC2000_STAGE_TRACE_MAX_ITER = 2
VMEC2000_STAGE_TRACE_TIMEOUT_S = 60.0
VMEC2000_STAGE_TRACE_CASES = (
    ("input.circular_tokamak", "1e-3"),
    ("input.basic_non_stellsym_pressure", "2e-3"),
)

VMEC2000_CONVERGED_TIMEOUT_S = 120.0
VMEC2000_CONVERGED_WOUT_CASES = (
    (
        "nfp4_QH_warm_start",
        "input.nfp4_QH_warm_start",
        {
            "NITER": "700",
            "NS_ARRAY": "19",
            "NITER_ARRAY": "700",
            "FTOL_ARRAY": "1e-9",
            "NSTEP": "50",
        },
    ),
    (
        "circular_tokamak",
        "input.circular_tokamak",
        {
            "NITER": "300",
            "NS_ARRAY": "13",
            "NITER_ARRAY": "300",
            "FTOL_ARRAY": "1e-10",
            "NSTEP": "50",
        },
    ),
    (
        "shaped_tokamak_pressure",
        "input.shaped_tokamak_pressure",
        {
            "NITER": "400",
            "NS_ARRAY": "13",
            "NITER_ARRAY": "400",
            "FTOL_ARRAY": "1e-10",
            "NSTEP": "50",
        },
    ),
)
VMEC2000_FREEB_LASYM_TIMEOUT_S = float(os.environ.get("VMEC2000_FREEB_LASYM_TIMEOUT_S", "120.0"))
VMEC2000_DIIID_BETA_TIMEOUT_S = 240.0


def _vmec2000_exec_or_skip() -> Path:
    if os.environ.get("VMEC2000_INTEGRATION") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 executable validation")

    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")
    return exe


def _write_patched_input(src: Path, dst: Path, *, updates: dict[str, str]) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_patch_indata(src.read_text(), updates=updates))
    return dst


def _write_diiid_beta_input(src: Path, dst: Path, *, pressure_scale: float, mgrid_name: str) -> Path:
    indata = read_indata(src)
    base_am = indata.scalars.get("AM", 0.0)
    am_values = base_am if isinstance(base_am, list) else [base_am]
    indata.scalars["AM"] = [float(value) * float(pressure_scale) for value in am_values]
    indata.scalars.update(
        {
            "MGRID_FILE": mgrid_name,
            "NS_ARRAY": [16, 32, 64],
            "NITER_ARRAY": [1000, 2000, 4000],
            "FTOL_ARRAY": [1.0e-8, 1.0e-10, 1.0e-11],
            "NITER": 4000,
            "FTOL": 1.0e-11,
        }
    )
    # Keep the same file VMEC2000-compatible.  VMEC2000 reads staged NS from
    # NS_ARRAY and rejects this DIII-D deck when a standalone NS is emitted.
    indata.scalars.pop("NS", None)
    write_indata(dst, indata)
    return dst


def _field_rel_rms(got, ref, *, radial_skip: int = 0, radial_drop_edge: bool = False) -> float:
    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    assert got_arr.shape == ref_arr.shape
    if got_arr.ndim >= 1 and radial_skip:
        got_arr = got_arr[radial_skip:, ...]
        ref_arr = ref_arr[radial_skip:, ...]
    if got_arr.ndim >= 1 and radial_drop_edge and got_arr.shape[0] > 0:
        got_arr = got_arr[:-1, ...]
        ref_arr = ref_arr[:-1, ...]
    assert got_arr.size > 0
    assert np.isfinite(got_arr).all()
    assert np.isfinite(ref_arr).all()
    diff_rms = float(np.sqrt(np.mean((got_arr - ref_arr) ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref_arr**2)))
    return diff_rms / ref_rms if ref_rms > 0.0 else diff_rms


def _assert_rel_rms(
    name: str,
    got,
    ref,
    *,
    limit: float,
    radial_skip: int = 0,
    radial_drop_edge: bool = False,
) -> None:
    rel_rms = _field_rel_rms(got, ref, radial_skip=radial_skip, radial_drop_edge=radial_drop_edge)
    assert rel_rms < limit, f"{name}: rel_rms={rel_rms:.3e} >= {limit:.3e}"


def _assert_scalar_close(name: str, got: float, ref: float, *, rtol: float, atol: float) -> None:
    np.testing.assert_allclose(
        float(got),
        float(ref),
        rtol=float(rtol),
        atol=float(atol),
        err_msg=f"{name} mismatch",
    )


def test_fast_vmec2000_stage_trace_validation_cases():
    """Optional short executable-backed parity checks for fixed-boundary stages."""

    exe = _vmec2000_exec_or_skip()
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "diagnostics" / "vmec2000_exec_stage_trace_compare.py"

    for input_name, rtol in VMEC2000_STAGE_TRACE_CASES:
        input_path = repo_root / "examples" / "data" / input_name
        if not input_path.exists():
            pytest.skip(f"Missing bundled input: {input_path}")
        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--vmec2000",
            str(exe),
            "--single-ns",
            str(VMEC2000_STAGE_TRACE_SINGLE_NS),
            "--max-iter",
            str(VMEC2000_STAGE_TRACE_MAX_ITER),
            "--vmec-timeout",
            f"{VMEC2000_STAGE_TRACE_TIMEOUT_S:.17g}",
            "--dump-level",
            "lite",
            "--rtol",
            rtol,
            "--atol",
            "1e-10",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True)


def test_vmec2000_free_boundary_lasym_true_reaches_vacuum_solve(tmp_path: Path) -> None:
    """Optional executable-backed guard for the bundled free-boundary LASYM deck."""

    exe = _vmec2000_exec_or_skip()
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "data" / "input.cth_like_free_bdy_lasym_small"
    mgrid_path = repo_root / "examples" / "data" / "mgrid_cth_like_lasym_small.nc"
    if not input_path.exists() or not mgrid_path.exists():
        pytest.skip("Missing bundled free-boundary LASYM fixture")

    result = run_xvmec2000(
        input_path,
        exec_path=exe,
        workdir=tmp_path / "vmec2000_freeb_lasym",
        timeout_s=VMEC2000_FREEB_LASYM_TIMEOUT_S,
        keep_workdir=True,
        indata_updates={
            "MGRID_FILE": f"'{mgrid_path}'",
            "NITER": "120",
            "NITER_ARRAY": "120",
        },
    )

    output = result.stdout + "\n" + result.stderr
    assert "I_TOR MISMATCH" not in output
    assert "VACUUM PRESSURE TURNED ON" in output
    assert result.stages, "VMEC2000 did not emit a parseable threed1 trace"
    assert result.stages[-1].rows[-1].it >= 100
    wout_path = result.workdir / "wout_cth_like_free_bdy_lasym_small.nc"
    assert wout_path.exists(), f"VMEC2000 did not produce {wout_path.name}"


def test_vmec2000_diiid_finite_beta_free_boundary_response_matches_vmec_jax(tmp_path: Path) -> None:
    """Optional executable-backed gate for high-beta DIII-D free-boundary response."""

    exe = _vmec2000_exec_or_skip()
    pytest.importorskip("netCDF4")

    from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
    from vmec_jax.wout import read_wout

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "data" / "input.DIII-D_lasym_false"
    mgrid_path = repo_root / "examples" / "data" / "mgrid_d3d_ef.nc"
    if not input_path.exists() or not mgrid_path.exists():
        pytest.skip("DIII-D input or fetched mgrid_d3d_ef.nc asset is unavailable")

    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    shutil_target = input_dir / mgrid_path.name
    shutil_target.write_bytes(mgrid_path.read_bytes())
    vacuum_input = _write_diiid_beta_input(input_path, input_dir / "input.diiid_b0_mg64", pressure_scale=0.0, mgrid_name=mgrid_path.name)
    beta_input = _write_diiid_beta_input(input_path, input_dir / "input.diiid_b180_mg64", pressure_scale=1.8, mgrid_name=mgrid_path.name)

    vmec2000_wouts: list[Path] = []
    vmec_jax_wouts: list[Path] = []
    for input_case in (vacuum_input, beta_input):
        vmec = run_xvmec2000(
            input_case,
            exec_path=exe,
            workdir=tmp_path / "vmec2000" / input_case.name,
            timeout_s=VMEC2000_DIIID_BETA_TIMEOUT_S,
            keep_workdir=True,
        )
        case = input_case.name.removeprefix("input.")
        wout_vmec = vmec.workdir / f"wout_{case}.nc"
        assert wout_vmec.exists(), f"VMEC2000 did not produce {wout_vmec.name}"
        vmec2000_wouts.append(wout_vmec)

        run = run_free_boundary(
            input_case,
            solver="vmec2000_iter",
            solver_mode="parity",
            multigrid_use_input_niter=True,
            verbose=False,
            jit_forces=False,
        )
        wout_jax = tmp_path / f"wout_{case}_vmec_jax.nc"
        write_wout_from_fixed_boundary_run(wout_jax, run, include_fsq=True)
        vmec_jax_wouts.append(wout_jax)

    vmec2000_response = free_boundary_response_metrics(vmec2000_wouts[0], vmec2000_wouts[1], ntheta=72, nphi=16)
    vmec_jax_response = free_boundary_response_metrics(vmec_jax_wouts[0], vmec_jax_wouts[1], ntheta=72, nphi=16)
    assert vmec2000_response.candidate_beta_percent > 1.0
    assert vmec_jax_response.candidate_beta_percent > 1.0
    assert vmec2000_response.lcfs_max_displacement > 0.05
    assert vmec_jax_response.lcfs_max_displacement > 0.05

    wref = read_wout(vmec2000_wouts[1])
    wjax = read_wout(vmec_jax_wouts[1])
    _assert_scalar_close("finite-beta aspect", wjax.aspect, wref.aspect, rtol=5.0e-4, atol=1.0e-6)
    _assert_rel_rms("finite-beta rmnc", wjax.rmnc, wref.rmnc, limit=1.0e-3)
    _assert_rel_rms("finite-beta zmns", wjax.zmns, wref.zmns, limit=1.0e-3)
    _assert_rel_rms("finite-beta iotaf", wjax.iotaf, wref.iotaf, limit=1.0e-3, radial_skip=1)


@pytest.mark.parametrize(
    "case,input_name,updates",
    VMEC2000_CONVERGED_WOUT_CASES,
)
def test_vmec2000_converged_wout_diagnostics_validation(
    case: str,
    input_name: str,
    updates: dict[str, str],
    tmp_path: Path,
) -> None:
    """Optional executable-backed gate comparing converged end-state diagnostics."""

    exe = _vmec2000_exec_or_skip()
    jax = pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    jax.config.update("jax_disable_jit", False)

    from vmec_jax.driver import run_fixed_boundary, write_wout_from_fixed_boundary_run
    from vmec_jax.wout import read_wout

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "data" / input_name
    if not input_path.exists():
        pytest.skip(f"Missing bundled input: {input_path}")

    patched_input = _write_patched_input(
        input_path,
        tmp_path / "inputs" / input_name,
        updates=updates,
    )

    vmec = run_xvmec2000(
        patched_input,
        exec_path=exe,
        workdir=tmp_path / "vmec2000" / case,
        timeout_s=VMEC2000_CONVERGED_TIMEOUT_S,
        keep_workdir=True,
    )
    wout_vmec_path = vmec.workdir / f"wout_{case}.nc"
    assert wout_vmec_path.exists(), f"VMEC2000 did not produce {wout_vmec_path.name}"

    run = run_fixed_boundary(
        str(patched_input),
        solver="vmec2000_iter",
        solver_mode="parity",
        multigrid_use_input_niter=True,
        verbose=False,
        jit_forces=False,
    )
    wout_jax_path = tmp_path / f"wout_{case}_vmec_jax.nc"
    write_wout_from_fixed_boundary_run(str(wout_jax_path), run)

    wref = read_wout(wout_vmec_path)
    wjax = read_wout(wout_jax_path)

    assert int(wjax.ns) == int(wref.ns)
    assert int(wjax.mpol) == int(wref.mpol)
    assert int(wjax.ntor) == int(wref.ntor)
    assert int(wjax.nfp) == int(wref.nfp)
    assert bool(wjax.lasym) == bool(wref.lasym)

    fsq_ref = float(wref.fsqr + wref.fsqz + wref.fsql)
    fsq_jax = float(wjax.fsqr + wjax.fsqz + wjax.fsql)
    assert np.isfinite([fsq_ref, fsq_jax]).all()
    assert fsq_ref < 1.0e-6, f"{case}: VMEC2000 final fsq={fsq_ref:.3e}"
    assert fsq_jax < 1.0e-6, f"{case}: vmec_jax final fsq={fsq_jax:.3e}"

    for name in ("rmnc", "zmns", "lmns"):
        _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=2.5e-4)
    if bool(wref.lasym):
        # The asymmetric lambda cosine block is gauge-sensitive across the two
        # implementations, so keep this end-state gate on geometry and fields.
        for name in ("rmns", "zmnc"):
            if hasattr(wjax, name) and hasattr(wref, name):
                _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=5.0e-4)
    for name in ("phipf", "chipf", "iotas", "iotaf", "pres", "presf"):
        _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=2.5e-4, radial_skip=1)
    for name in ("gmnc", "bmnc", "bsupumnc", "bsupvmnc", "bsubumnc", "bsubvmnc"):
        _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=2.5e-3, radial_skip=1)
    if bool(wref.lasym):
        for name in ("gmns", "bmns", "bsupumns", "bsupvmns", "bsubumns", "bsubvmns"):
            if hasattr(wjax, name) and hasattr(wref, name):
                _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=5.0e-3, radial_skip=1)

    _assert_scalar_close("wb", wjax.wb, wref.wb, rtol=2.5e-3, atol=1.0e-8)
    _assert_scalar_close("wp", wjax.wp, wref.wp, rtol=2.5e-3, atol=1.0e-8)
    _assert_scalar_close("volume_p", wjax.volume_p, wref.volume_p, rtol=2.5e-3, atol=1.0e-8)

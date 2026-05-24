from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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
VMEC2000_FREEB_LASYM_TIMEOUT_S = 60.0
VMEC2000_FREEB_LASYM_MAX_FSQ_TOTAL = 2.0e-1


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

    from vmec_jax.wout import read_wout

    wout = read_wout(wout_path)
    assert int(wout.ns) == 15
    assert int(wout.mpol) == 5
    assert int(wout.ntor) == 4
    assert int(wout.nfp) == 5
    assert bool(wout.lasym) is True
    fsq_total = float(wout.fsqr + wout.fsqz + wout.fsql)
    assert np.isfinite([fsq_total, wout.wb, wout.wp]).all()
    assert fsq_total < VMEC2000_FREEB_LASYM_MAX_FSQ_TOTAL
    assert float(wout.wb) > 0.0
    assert float(wout.wp) > 0.0


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

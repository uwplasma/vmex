from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.namelist import read_indata
from vmec_jax.vmec2000_exec import _patch_indata, find_vmec2000_exec, run_xvmec2000
from vmec_jax.wout import read_wout


pytestmark = pytest.mark.vmec2000


@dataclass(frozen=True)
class ConvergedParityCase:
    case: str
    input_name: str
    updates: dict[str, str]
    timeout_s: float = 120.0


CONVERGED_PARITY_CASES = (
    ConvergedParityCase(
        case="circular_tokamak",
        input_name="input.circular_tokamak",
        updates={
            "NITER": "300",
            "NS_ARRAY": "13",
            "NITER_ARRAY": "300",
            "FTOL_ARRAY": "1e-10",
            "NSTEP": "50",
        },
    ),
)


def _vmec2000_exec_or_skip() -> Path:
    if os.environ.get("VMEC2000_INTEGRATION", "0") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 executable parity tests")

    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")
    return exe


def _write_patched_input(src: Path, dst: Path, *, updates: dict[str, str]) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_patch_indata(src.read_text(), updates=updates))
    return dst


def _rel_rms(got, ref, *, radial_skip: int = 0) -> float:
    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    assert got_arr.shape == ref_arr.shape
    if got_arr.ndim >= 1 and radial_skip:
        got_arr = got_arr[radial_skip:, ...]
        ref_arr = ref_arr[radial_skip:, ...]
    assert got_arr.size > 0
    assert np.isfinite(got_arr).all()
    assert np.isfinite(ref_arr).all()
    diff_rms = float(np.sqrt(np.mean((got_arr - ref_arr) ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref_arr**2)))
    return diff_rms / ref_rms if ref_rms > 0.0 else diff_rms


def _assert_rel_rms(name: str, got, ref, *, limit: float, radial_skip: int = 0) -> None:
    rel_rms = _rel_rms(got, ref, radial_skip=radial_skip)
    assert rel_rms < limit, f"{name}: rel_rms={rel_rms:.3e} >= {limit:.3e}"


def _vmec_iotaf_from_iotas(iotas: np.ndarray) -> np.ndarray:
    if iotas.size < 3:
        return np.asarray(iotas, dtype=float)
    out = np.zeros_like(iotas, dtype=float)
    out[0] = 1.5 * iotas[1] - 0.5 * iotas[2]
    out[1:-1] = 0.5 * (iotas[1:-1] + iotas[2:])
    out[-1] = 1.5 * iotas[-1] - 0.5 * iotas[-2]
    return out


def _assert_wout_physics_consistent(wout, *, input_path: Path) -> None:
    indata = read_indata(input_path)
    fsq = float(np.linalg.norm(np.asarray([wout.fsqr, wout.fsqz, wout.fsql], dtype=float)))
    assert np.isfinite(fsq)
    assert fsq < 1.0e-6

    iotas = np.asarray(wout.iotas, dtype=float)
    iotaf = np.asarray(wout.iotaf, dtype=float)
    assert iotas.shape == (int(wout.ns),)
    assert iotaf.shape == (int(wout.ns),)
    assert np.isfinite(iotas).all()
    np.testing.assert_allclose(iotaf, _vmec_iotaf_from_iotas(iotas), rtol=1.0e-12, atol=1.0e-12)

    phi = np.asarray(wout.phi, dtype=float)
    np.testing.assert_allclose(phi[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(phi[-1], indata.get_float("PHIEDGE", 0.0), rtol=1.0e-12, atol=1.0e-12)

    assert float(wout.Aminor_p) > 0.0
    assert float(wout.Rmajor_p) > 0.0
    assert float(wout.volume_p) > 0.0
    assert 1.0 < float(wout.aspect) < 20.0
    np.testing.assert_allclose(wout.Rmajor_p / wout.Aminor_p, wout.aspect, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("case", CONVERGED_PARITY_CASES, ids=[case.case for case in CONVERGED_PARITY_CASES])
def test_xvmec2000_converged_wout_matches_vmec_jax(case: ConvergedParityCase, tmp_path: Path) -> None:
    """Optional executable-backed parity over converged wouts, not short finite-step traces."""

    exe = _vmec2000_exec_or_skip()
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    from vmec_jax.driver import run_fixed_boundary, write_wout_from_fixed_boundary_run

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "examples" / "data" / case.input_name
    if not input_path.exists():
        pytest.skip(f"Missing bundled input: {input_path}")

    patched_input = _write_patched_input(
        input_path,
        tmp_path / "inputs" / case.input_name,
        updates=case.updates,
    )

    vmec = run_xvmec2000(
        patched_input,
        exec_path=exe,
        workdir=tmp_path / "vmec2000" / case.case,
        timeout_s=case.timeout_s,
        keep_workdir=True,
    )
    wout_vmec_path = vmec.workdir / f"wout_{case.case}.nc"
    assert wout_vmec_path.exists(), f"VMEC2000 did not produce {wout_vmec_path.name}"

    run = run_fixed_boundary(
        str(patched_input),
        solver="vmec2000_iter",
        solver_mode="parity",
        multigrid_use_input_niter=True,
        verbose=False,
        jit_forces=False,
    )
    wout_jax_path = tmp_path / f"wout_{case.case}_vmec_jax.nc"
    write_wout_from_fixed_boundary_run(str(wout_jax_path), run)

    wref = read_wout(wout_vmec_path)
    wjax = read_wout(wout_jax_path)

    _assert_wout_physics_consistent(wref, input_path=patched_input)
    _assert_wout_physics_consistent(wjax, input_path=patched_input)

    assert int(wjax.ns) == int(wref.ns)
    assert int(wjax.mpol) == int(wref.mpol)
    assert int(wjax.ntor) == int(wref.ntor)
    assert int(wjax.nfp) == int(wref.nfp)
    assert bool(wjax.lasym) == bool(wref.lasym)

    for name in ("rmnc", "zmns", "lmns"):
        _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=1.0e-3)
    for name in ("phipf", "chipf", "iotas", "iotaf", "pres", "presf"):
        _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=1.0e-3, radial_skip=1)
    for name in ("gmnc", "bmnc", "bsupumnc", "bsupvmnc", "bsubumnc", "bsubvmnc"):
        _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=5.0e-3, radial_skip=1)

    np.testing.assert_allclose(wjax.wb, wref.wb, rtol=5.0e-3, atol=1.0e-8)
    np.testing.assert_allclose(wjax.wp, wref.wp, rtol=5.0e-3, atol=1.0e-8)
    np.testing.assert_allclose(wjax.volume_p, wref.volume_p, rtol=5.0e-3, atol=1.0e-8)
    np.testing.assert_allclose(wjax.aspect, wref.aspect, rtol=5.0e-3, atol=1.0e-8)

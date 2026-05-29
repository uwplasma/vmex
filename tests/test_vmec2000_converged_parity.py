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
    input_relpath: str
    updates: dict[str, str]
    lfreeb: bool
    axisymmetric: bool
    lasym: bool
    multigrid: bool
    mgrid_relpath: str | None = None
    nightly: bool = False
    timeout_s: float = 120.0
    require_aspect: bool = True
    xfail_reason: str | None = None
    skip_reason: str | None = None


CONVERGED_PARITY_CASES = (
    ConvergedParityCase(
        case="circular_tokamak",
        input_relpath="examples/data/input.circular_tokamak",
        updates={
            "NITER": "300",
            "NS_ARRAY": "13",
            "NITER_ARRAY": "300",
            "FTOL_ARRAY": "1e-10",
            "NSTEP": "50",
        },
        lfreeb=False,
        axisymmetric=True,
        lasym=False,
        multigrid=False,
    ),
    ConvergedParityCase(
        case="LandremanPaul2021_QA_lowres",
        input_relpath="examples/data/input.LandremanPaul2021_QA_lowres",
        updates={
            "NITER": "1000",
            "NS_ARRAY": "16, 31, 50",
            "NITER_ARRAY": "600, 1000, 1000",
            "FTOL_ARRAY": "1e-10, 1e-10, 1e-10",
            "NSTEP": "50",
        },
        lfreeb=False,
        axisymmetric=False,
        lasym=False,
        multigrid=True,
        nightly=True,
        timeout_s=240.0,
    ),
    ConvergedParityCase(
        case="up_down_asymmetric_tokamak",
        input_relpath="examples_single_grid/data/input.up_down_asymmetric_tokamak",
        updates={
            "NITER": "800",
            "NS_ARRAY": "17",
            "NITER_ARRAY": "800",
            "FTOL_ARRAY": "1e-10",
            "NSTEP": "50",
        },
        lfreeb=False,
        axisymmetric=True,
        lasym=True,
        multigrid=False,
        nightly=True,
        timeout_s=180.0,
        require_aspect=False,
        xfail_reason=(
            "axisymmetric LASYM zero-pressure wout gap: lmns relRMS ~1.8e-2, "
            "bsupumns ~1.1e-2, bsubvmns absolute ~5.7e-4 against near-zero reference"
        ),
    ),
    ConvergedParityCase(
        case="basic_non_stellsym_pressure",
        input_relpath="examples_single_grid/data/input.basic_non_stellsym_pressure",
        updates={
            "NITER": "1200",
            "NS_ARRAY": "25",
            "NITER_ARRAY": "1200",
            "FTOL_ARRAY": "1e-10",
            "NSTEP": "50",
        },
        lfreeb=False,
        axisymmetric=False,
        lasym=True,
        multigrid=False,
        nightly=True,
        timeout_s=240.0,
    ),
    ConvergedParityCase(
        case="nfp4_QH_finite_beta",
        input_relpath="examples/data/input.nfp4_QH_finite_beta",
        updates={},
        lfreeb=False,
        axisymmetric=False,
        lasym=False,
        multigrid=True,
        nightly=True,
        timeout_s=300.0,
    ),
    ConvergedParityCase(
        case="cth_like_free_bdy",
        input_relpath="examples_single_grid/data/input.cth_like_free_bdy",
        updates={
            "NITER": "5000",
            "NITER_ARRAY": "5000",
            "FTOL_ARRAY": "1e-10",
        },
        lfreeb=True,
        axisymmetric=False,
        lasym=False,
        multigrid=False,
        mgrid_relpath="examples_single_grid/data/mgrid_cth_like.nc",
        nightly=True,
        timeout_s=600.0,
        skip_reason=(
            "optional converged free-boundary WOUT parity is not yet a bounded "
            "nightly gate; use the promoted stage-trace free-boundary smoke instead"
        ),
    ),
)

CONVERGED_PARITY_PARAMS = [
    pytest.param(
        case,
        id=case.case,
        marks=tuple(
            mark
            for mark in (
                pytest.mark.xfail(reason=case.xfail_reason, strict=True) if case.xfail_reason else None,
                pytest.mark.skip(reason=case.skip_reason) if case.skip_reason else None,
            )
            if mark is not None
        ),
    )
    for case in CONVERGED_PARITY_CASES
]


def _vmec2000_exec_or_skip() -> Path:
    if os.environ.get("VMEC2000_INTEGRATION", "0") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 executable parity tests")

    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")
    return exe


def _nightly_or_skip(case: ConvergedParityCase) -> None:
    if case.nightly and os.environ.get("VMEC2000_NIGHTLY", "0") != "1":
        pytest.skip("Set VMEC2000_NIGHTLY=1 to run slow converged VMEC2000 parity cases")


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


def _assert_max_abs_diff(name: str, got, ref, *, limit: float, radial_skip: int = 0) -> None:
    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    assert got_arr.shape == ref_arr.shape
    if got_arr.ndim >= 1 and radial_skip:
        got_arr = got_arr[radial_skip:, ...]
        ref_arr = ref_arr[radial_skip:, ...]
    assert got_arr.size > 0
    assert np.isfinite(got_arr).all()
    assert np.isfinite(ref_arr).all()
    max_abs = float(np.max(np.abs(got_arr - ref_arr)))
    assert max_abs < limit, f"{name}: max_abs={max_abs:.3e} >= {limit:.3e}"


def _vmec_iotaf_from_iotas(iotas: np.ndarray) -> np.ndarray:
    if iotas.size < 3:
        return np.asarray(iotas, dtype=float)
    out = np.zeros_like(iotas, dtype=float)
    out[0] = 1.5 * iotas[1] - 0.5 * iotas[2]
    out[1:-1] = 0.5 * (iotas[1:-1] + iotas[2:])
    out[-1] = 1.5 * iotas[-1] - 0.5 * iotas[-2]
    return out


def _assert_wout_physics_consistent(wout, *, input_path: Path, require_aspect: bool = True) -> None:
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

    aspect_scalars = np.asarray([wout.Aminor_p, wout.Rmajor_p, wout.volume_p, wout.aspect], dtype=float)
    assert np.isfinite(aspect_scalars).all()
    if require_aspect or not np.all(aspect_scalars == 0.0):
        assert float(wout.Aminor_p) > 0.0
        assert float(wout.Rmajor_p) > 0.0
        assert float(wout.volume_p) > 0.0
        assert 1.0 < float(wout.aspect) < 20.0
        np.testing.assert_allclose(wout.Rmajor_p / wout.Aminor_p, wout.aspect, rtol=1.0e-12, atol=1.0e-12)


def _assert_glasser_profiles_self_consistent(wout) -> None:
    dmerc = np.asarray(wout.DMerc, dtype=float)
    d_r = np.asarray(wout.D_R, dtype=float)
    correction = np.asarray(wout.glasser_correction, dtype=float)
    valid = np.asarray(wout.glasser_shear_valid, dtype=bool)

    assert dmerc.shape == (int(wout.ns),)
    assert d_r.shape == dmerc.shape
    assert correction.shape == dmerc.shape
    assert valid.shape == dmerc.shape
    assert np.isfinite(dmerc).all()
    assert np.isfinite(d_r).all()
    assert np.isfinite(correction).all()
    np.testing.assert_allclose(d_r, -dmerc + correction, rtol=1.0e-11, atol=1.0e-13)


@pytest.mark.parametrize("case", CONVERGED_PARITY_PARAMS)
def test_xvmec2000_converged_wout_matches_vmec_jax(case: ConvergedParityCase, tmp_path: Path) -> None:
    """Optional executable-backed parity over converged wouts, not short finite-step traces."""

    exe = _vmec2000_exec_or_skip()
    _nightly_or_skip(case)
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    from vmec_jax.driver import run_fixed_boundary, run_free_boundary, write_wout_from_fixed_boundary_run

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / case.input_relpath
    if not input_path.exists():
        pytest.skip(f"Missing bundled input: {input_path}")

    updates = dict(case.updates)
    if case.mgrid_relpath is not None:
        mgrid_path = repo_root / case.mgrid_relpath
        if not mgrid_path.exists():
            pytest.skip(f"Missing bundled mgrid: {mgrid_path}")
        updates["MGRID_FILE"] = f"'{mgrid_path}'"

    patched_input = _write_patched_input(
        input_path,
        tmp_path / "inputs" / input_path.name,
        updates=updates,
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

    run_fn = run_free_boundary if case.lfreeb else run_fixed_boundary
    run = run_fn(
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

    _assert_wout_physics_consistent(wref, input_path=patched_input, require_aspect=case.require_aspect)
    _assert_wout_physics_consistent(wjax, input_path=patched_input, require_aspect=case.require_aspect)

    indata = read_indata(patched_input)
    assert bool(indata.get_bool("LFREEB", False)) is case.lfreeb
    assert bool(indata.get_bool("LASYM", False)) is case.lasym
    assert bool(int(indata.get_int("NTOR", 0)) == 0) is case.axisymmetric

    assert int(wjax.ns) == int(wref.ns)
    assert int(wjax.mpol) == int(wref.mpol)
    assert int(wjax.ntor) == int(wref.ntor)
    assert int(wjax.nfp) == int(wref.nfp)
    assert bool(wjax.lasym) == bool(wref.lasym)

    for name in ("rmnc", "zmns", "lmns"):
        _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=1.0e-3)
    if case.lasym:
        for name in ("rmns", "zmnc"):
            _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=2.0e-3)
    for name in ("phipf", "chipf", "iotas", "iotaf", "pres", "presf"):
        _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=1.0e-3, radial_skip=1)
    for name in ("gmnc", "bmnc", "bsupumnc", "bsupvmnc", "bsubumnc", "bsubvmnc"):
        _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=5.0e-3, radial_skip=1)
    if case.lasym:
        for name in ("gmns", "bmns", "bsupumns", "bsupvmns", "bsubumns", "bsubvmns"):
            _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=1.0e-2, radial_skip=1)

    np.testing.assert_allclose(wjax.wb, wref.wb, rtol=5.0e-3, atol=1.0e-8)
    np.testing.assert_allclose(wjax.wp, wref.wp, rtol=5.0e-3, atol=1.0e-8)
    np.testing.assert_allclose(wjax.volume_p, wref.volume_p, rtol=5.0e-3, atol=1.0e-8)
    np.testing.assert_allclose(wjax.aspect, wref.aspect, rtol=5.0e-3, atol=1.0e-8)

    if float(np.max(np.abs(np.asarray(wref.pres, dtype=float)[1:]))) > 0.0:
        for wout in (wref, wjax):
            _assert_glasser_profiles_self_consistent(wout)
        for name in ("DMerc", "Dshear", "Dwell", "Dgeod"):
            _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=2.0e-2, radial_skip=1)
        _assert_max_abs_diff("Dcurr", wjax.Dcurr, wref.Dcurr, limit=5.0e-6, radial_skip=1)
        _assert_max_abs_diff(
            "glasser_correction",
            wjax.glasser_correction,
            wref.glasser_correction,
            limit=1.0e-5,
            radial_skip=1,
        )
        for name in ("D_R",):
            _assert_rel_rms(name, getattr(wjax, name), getattr(wref, name), limit=5.0e-2, radial_skip=1)

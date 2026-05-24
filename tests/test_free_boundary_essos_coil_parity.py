from __future__ import annotations

from copy import deepcopy
import json
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
FINITE_PRESSURE_SCALE = 34.46233666638
LPQA_COIL_FILE = "ESSOS_biot_savart_LandremanPaulQA.json"


def _candidate_essos_input_dirs() -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("ESSOS_INPUT_DIR"):
        candidates.append(Path(os.environ["ESSOS_INPUT_DIR"]).expanduser())
    candidates.extend(
        [
            ROOT.parent / "ESSOS_mgrid_pr" / "examples" / "input_files",
            ROOT.parent / "ESSOS" / "examples" / "input_files",
            Path.cwd() / "examples" / "input_files",
        ]
    )
    return candidates


def _find_lpqa_coils() -> Path:
    for directory in _candidate_essos_input_dirs():
        path = directory / LPQA_COIL_FILE
        if path.exists():
            return path
    return _candidate_essos_input_dirs()[0] / LPQA_COIL_FILE


LPQA_COILS = _find_lpqa_coils()


def _load_lpqa_essos_coils():
    essos_coils = pytest.importorskip("essos.coils")
    if not LPQA_COILS.exists():
        searched = ", ".join(str(path) for path in _candidate_essos_input_dirs())
        pytest.skip(f"missing ESSOS Landreman-Paul QA coils; set ESSOS_INPUT_DIR. Searched: {searched}")
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


def _rel_rms(got, ref, *, radial_skip: int = 0) -> float:
    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    assert got_arr.shape == ref_arr.shape
    if radial_skip and got_arr.ndim >= 1:
        got_arr = got_arr[radial_skip:, ...]
        ref_arr = ref_arr[radial_skip:, ...]
    assert got_arr.size > 0
    assert np.isfinite(got_arr).all()
    assert np.isfinite(ref_arr).all()
    denom = float(np.sqrt(np.mean(ref_arr**2)))
    diff = float(np.sqrt(np.mean((got_arr - ref_arr) ** 2)))
    return diff / denom if denom > 0.0 else diff


def _assert_rel_rms(name: str, got, ref, *, limit: float, radial_skip: int = 0) -> None:
    rel_rms = _rel_rms(got, ref, radial_skip=radial_skip)
    assert rel_rms < limit, f"{name}: rel_rms={rel_rms:.3e} >= {limit:.3e}"


def _assert_same_wout_layout(got, ref) -> None:
    assert int(got.ns) == int(ref.ns)
    assert int(got.mpol) == int(ref.mpol)
    assert int(got.ntor) == int(ref.ntor)
    assert int(got.nfp) == int(ref.nfp)
    assert bool(got.lasym) == bool(ref.lasym)
    np.testing.assert_array_equal(np.asarray(got.xm, dtype=int), np.asarray(ref.xm, dtype=int))
    np.testing.assert_array_equal(np.asarray(got.xn, dtype=int), np.asarray(ref.xn, dtype=int))


def _assert_vmec_jax_direct_matches_generated_mgrid_wout(wout_direct, wout_mgrid) -> None:
    _assert_same_wout_layout(wout_direct, wout_mgrid)
    for name in ("rmnc", "zmns", "lmns", "iotas", "iotaf"):
        np.testing.assert_allclose(
            getattr(wout_direct, name),
            getattr(wout_mgrid, name),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=f"direct-coil and generated-mgrid vmec_jax WOUT mismatch for {name}",
        )
    for name in ("aspect", "wb", "wp"):
        np.testing.assert_allclose(
            getattr(wout_direct, name),
            getattr(wout_mgrid, name),
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=f"direct-coil and generated-mgrid vmec_jax WOUT mismatch for {name}",
        )
    assert float(wout_direct.wp) > 0.0


def _low_order_mode_mask(wout, *, max_m: int = 2, max_abs_n: int = 2) -> np.ndarray:
    xm = np.asarray(wout.xm, dtype=int)
    xn = np.asarray(wout.xn, dtype=int)
    nfp = max(1, int(wout.nfp))
    n = np.rint(xn / float(nfp)).astype(int)
    return (np.abs(xm) <= max_m) & (np.abs(n) <= max_abs_n)


def _finite_scalar(wout, name: str) -> float | None:
    if not hasattr(wout, name):
        return None
    value = float(getattr(wout, name))
    assert np.isfinite(value), f"{name}: non-finite scalar {value}"
    return value


def _assert_same_sign_and_scale(name: str, got: float, ref: float, *, max_ratio: float) -> None:
    assert np.isfinite([got, ref]).all(), f"{name}: non-finite values got={got}, ref={ref}"
    tiny = 1.0e-14
    if abs(got) <= tiny and abs(ref) <= tiny:
        return
    assert got * ref >= 0.0, f"{name}: sign mismatch got={got:.6e}, ref={ref:.6e}"
    ratio = max(abs(got), tiny) / max(abs(ref), tiny)
    assert (1.0 / max_ratio) <= ratio <= max_ratio, f"{name}: scale ratio={ratio:.3e}"


def _assert_vmec2000_generated_mgrid_wout_matches_vmec_jax(wout_jax, wout_vmec2000) -> None:
    _assert_same_wout_layout(wout_jax, wout_vmec2000)

    np.testing.assert_allclose(wout_jax.aspect, wout_vmec2000.aspect, rtol=1.5e-1, atol=1.0e-8)
    np.testing.assert_allclose(wout_jax.wb, wout_vmec2000.wb, rtol=2.5e-1, atol=1.0e-8)

    for name in ("wp", "betatotal", "betapol", "betator", "betaxis"):
        got = _finite_scalar(wout_jax, name)
        ref = _finite_scalar(wout_vmec2000, name)
        if got is None or ref is None:
            continue
        if abs(got) <= 1.0e-14 and abs(ref) <= 1.0e-14:
            continue
        _assert_same_sign_and_scale(name, got, ref, max_ratio=10.0)

    iotas_jax = np.asarray(wout_jax.iotas, dtype=float)
    iotas_vmec2000 = np.asarray(wout_vmec2000.iotas, dtype=float)
    np.testing.assert_allclose(
        float(np.mean(iotas_jax[1:])),
        float(np.mean(iotas_vmec2000[1:])),
        rtol=2.5e-1,
        atol=1.0e-8,
    )
    _assert_rel_rms("iotas", iotas_jax, iotas_vmec2000, limit=3.5e-1, radial_skip=1)

    low_order = _low_order_mode_mask(wout_vmec2000)
    assert np.any(low_order), "no low-order modes selected for WOUT comparison"
    for name in ("rmnc", "zmns"):
        _assert_rel_rms(
            f"low-order {name}",
            np.asarray(getattr(wout_jax, name))[:, low_order],
            np.asarray(getattr(wout_vmec2000, name))[:, low_order],
            limit=4.0e-1,
            radial_skip=1,
        )


def _fsq_total(wout) -> float:
    fsq = float(np.sum(np.asarray([wout.fsqr, wout.fsqz, wout.fsql], dtype=float)))
    assert np.isfinite(fsq), f"non-finite fsq_total={fsq}"
    return fsq


def _assert_fsq_total_same_scale(wout_jax, wout_vmec2000) -> None:
    fsq_jax = _fsq_total(wout_jax)
    fsq_vmec2000 = _fsq_total(wout_vmec2000)
    if abs(fsq_jax) <= 1.0e-14 and abs(fsq_vmec2000) <= 1.0e-14:
        return
    _assert_same_sign_and_scale("fsq_total", fsq_jax, fsq_vmec2000, max_ratio=10.0)


def _scalar_gap(got_wout, ref_wout, name: str) -> dict[str, float | None]:
    if not (hasattr(got_wout, name) and hasattr(ref_wout, name)):
        return {"got": None, "ref": None, "abs_delta": None, "rel_delta": None}
    got = float(getattr(got_wout, name))
    ref = float(getattr(ref_wout, name))
    abs_delta = abs(got - ref)
    rel_delta = abs_delta / max(abs(ref), 1.0e-300)
    return {"got": got, "ref": ref, "abs_delta": abs_delta, "rel_delta": rel_delta}


def _array_gap(got, ref, *, radial_skip: int = 0, mode_mask: np.ndarray | None = None) -> dict[str, float | int]:
    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    if got_arr.shape != ref_arr.shape:
        return {"shape_mismatch": 1, "got_size": int(got_arr.size), "ref_size": int(ref_arr.size)}
    if radial_skip and got_arr.ndim >= 1:
        got_arr = got_arr[radial_skip:, ...]
        ref_arr = ref_arr[radial_skip:, ...]
    if mode_mask is not None:
        got_arr = got_arr[..., mode_mask]
        ref_arr = ref_arr[..., mode_mask]
    diff = got_arr - ref_arr
    abs_rms = float(np.sqrt(np.mean(diff * diff)))
    ref_rms = float(np.sqrt(np.mean(ref_arr * ref_arr)))
    return {
        "size": int(got_arr.size),
        "abs_rms_delta": abs_rms,
        "rel_rms_delta": abs_rms / max(ref_rms, 1.0e-300),
        "max_abs_delta": float(np.max(np.abs(diff))),
    }


def _wout_gap_report(wout_jax, wout_vmec2000) -> dict[str, object]:
    report: dict[str, object] = {
        "layout": {
            "jax": {
                "ns": int(wout_jax.ns),
                "mpol": int(wout_jax.mpol),
                "ntor": int(wout_jax.ntor),
                "nfp": int(wout_jax.nfp),
                "lasym": bool(wout_jax.lasym),
                "signgs": int(getattr(wout_jax, "signgs", 0)),
            },
            "vmec2000": {
                "ns": int(wout_vmec2000.ns),
                "mpol": int(wout_vmec2000.mpol),
                "ntor": int(wout_vmec2000.ntor),
                "nfp": int(wout_vmec2000.nfp),
                "lasym": bool(wout_vmec2000.lasym),
                "signgs": int(getattr(wout_vmec2000, "signgs", 0)),
            },
        },
        "scalars": {},
        "profiles": {},
        "low_order_modes": {},
    }
    scalar_names = (
        "aspect",
        "Aminor_p",
        "Rmajor_p",
        "volume_p",
        "wb",
        "wp",
        "betatotal",
        "betapol",
        "betator",
        "betaxis",
        "fsqr",
        "fsqz",
        "fsql",
    )
    report["scalars"] = {name: _scalar_gap(wout_jax, wout_vmec2000, name) for name in scalar_names}
    for name in ("iotas", "iotaf", "pres", "presf", "vp", "phipf", "phips", "chipf", "buco", "bvco", "jcuru", "jcurv", "equif"):
        if hasattr(wout_jax, name) and hasattr(wout_vmec2000, name):
            report["profiles"][name] = _array_gap(
                getattr(wout_jax, name),
                getattr(wout_vmec2000, name),
                radial_skip=1,
            )

    try:
        low_order = _low_order_mode_mask(wout_vmec2000)
        for name in ("rmnc", "zmns", "lmns", "bmnc", "gmnc", "bsubumnc", "bsubvmnc", "bsupumnc", "bsupvmnc"):
            if hasattr(wout_jax, name) and hasattr(wout_vmec2000, name):
                report["low_order_modes"][name] = _array_gap(
                    getattr(wout_jax, name),
                    getattr(wout_vmec2000, name),
                    radial_skip=1,
                    mode_mask=low_order,
                )
    except Exception as exc:
        report["low_order_modes_error"] = repr(exc)
    return report


def _write_wout_gap_report(path: Path, *, wout_jax, wout_vmec2000) -> None:
    report = _wout_gap_report(wout_jax, wout_vmec2000)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    worst_scalars = sorted(
        (
            (name, values.get("rel_delta"))
            for name, values in report.get("scalars", {}).items()
            if isinstance(values, dict) and values.get("rel_delta") is not None
        ),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:5]
    print(f"VMEC2000 WOUT gap report: {path}")
    print(f"Worst scalar relative gaps: {worst_scalars}")


def _write_vmec2000_no_wout_report(path: Path, vmec2000) -> None:
    threed_tail: list[str] = []
    if vmec2000.threed1_path is not None and vmec2000.threed1_path.exists():
        threed_tail = vmec2000.threed1_path.read_text(errors="replace").splitlines()[-80:]
    payload = {
        "status": "vmec2000_no_wout",
        "workdir": str(vmec2000.workdir),
        "input_path": str(vmec2000.input_path),
        "runtime_s": float(vmec2000.runtime_s),
        "stdout_tail": vmec2000.stdout.splitlines()[-40:],
        "stderr_tail": vmec2000.stderr.splitlines()[-40:],
        "files": sorted(p.name for p in vmec2000.workdir.iterdir()),
        "threed1_path": None if vmec2000.threed1_path is None else str(vmec2000.threed1_path),
        "threed1_tail": threed_tail,
        "stages": [
            {
                "ns": int(stage.ns),
                "niter": int(stage.niter),
                "ftolv": float(stage.ftolv),
                "row_count": len(stage.rows),
                "last_row": None
                if not stage.rows
                else {
                    "it": int(stage.rows[-1].it),
                    "fsqr": float(stage.rows[-1].fsqr),
                    "fsqz": float(stage.rows[-1].fsqz),
                    "fsql": float(stage.rows[-1].fsql),
                    "fsqr1": float(stage.rows[-1].fsqr1),
                    "fsqz1": float(stage.rows[-1].fsqz1),
                    "fsql1": float(stage.rows[-1].fsql1),
                    "delt0r": stage.rows[-1].delt0r,
                    "r00": stage.rows[-1].r00,
                    "w": stage.rows[-1].w,
                },
            }
            for stage in vmec2000.stages
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"VMEC2000 did not write WOUT; diagnostic report: {path}")


def _vmec2000_wout_path(vmec2000) -> Path:
    case = vmec2000.input_path.name.removeprefix("input.")
    return vmec2000.workdir / f"wout_{case}.nc"


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

    _assert_vmec_jax_direct_matches_generated_mgrid_wout(wout_direct, wout_mgrid)


@pytest.mark.vmec2000
@pytest.mark.xfail(
    reason=(
        "Generated ESSOS-mgrid VMEC2000 free-boundary WOUT parity is not bounded "
        "yet; this optional gate captures the current gap while the direct-coil "
        "provider path is being developed."
    ),
    strict=False,
)
def test_vmec2000_generated_mgrid_free_boundary_matches_vmec_jax_and_direct_coils(tmp_path: Path) -> None:
    """Optional three-way parity gate for generated-mgrid/free-boundary cases.

    Run with `VMEC2000_INTEGRATION=1`.  The intended passing condition is:

    1. VMEC2000 free-boundary from ESSOS-generated mgrid,
    2. `vmec_jax` free-boundary from the same mgrid,
    3. `vmec_jax` free-boundary from direct ESSOS/JAX Biot-Savart coils,

    all produce matching WOUT-level equilibrium quantities.  Per-iteration
    VMEC2000 rows are treated as a printed trace only and are not used as the
    source of truth for accepted final residual components.
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

    _run_mgrid, wout_mgrid = _run_vmec_jax_freeb(mgrid_input)
    _run_direct, wout_direct = _run_vmec_jax_freeb(direct_input, direct_params=direct_params)
    _assert_vmec_jax_direct_matches_generated_mgrid_wout(wout_direct, wout_mgrid)

    vmec2000 = run_xvmec2000(mgrid_input, exec_path=exe, workdir=tmp_path / "vmec2000", timeout_s=90, keep_workdir=True)
    wout_vmec2000_path = _vmec2000_wout_path(vmec2000)
    if not wout_vmec2000_path.exists():
        _write_vmec2000_no_wout_report(tmp_path / "vmec2000_no_wout_report.json", vmec2000)
    assert wout_vmec2000_path.exists(), f"VMEC2000 did not produce {wout_vmec2000_path.name}"
    wout_vmec2000 = read_wout(wout_vmec2000_path)

    _write_wout_gap_report(
        tmp_path / "vmec2000_wout_gap_report.json",
        wout_jax=wout_mgrid,
        wout_vmec2000=wout_vmec2000,
    )
    _assert_vmec2000_generated_mgrid_wout_matches_vmec_jax(wout_mgrid, wout_vmec2000)
    _assert_fsq_total_same_scale(wout_mgrid, wout_vmec2000)

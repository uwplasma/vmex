from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.external_fields import from_essos_coils
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.vmec2000_exec import find_vmec2000_exec, run_xvmec2000
from vmec_jax.wout import read_wout


ROOT = Path(__file__).resolve().parents[1]
LPQA_INPUT = ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"
FINITE_PRESSURE_SCALE = 1000.0
FREE_BOUNDARY_PHIEDGE = -0.025
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


def test_generated_mgrid_diagnostic_boundary_domain_check_flags_outside_surface() -> None:
    from tools.diagnostics.compare_freeb_coils_mgrid_vmec2000 import (
        _boundary_domain_check,
        _wout_boundary_extents,
    )

    wout = SimpleNamespace(
        xm=np.asarray([0, 1]),
        xn=np.asarray([0, 0]),
        rmnc=np.asarray([[0.0, 0.0], [10.0, 2.0]]),
        rmns=np.zeros((2, 2)),
        zmns=np.asarray([[0.0, 0.0], [0.0, 3.0]]),
        zmnc=np.zeros((2, 2)),
        nfp=2,
        lasym=False,
    )

    extents = _wout_boundary_extents(wout, ntheta=64, nphi=8)
    assert extents["available"] is True
    assert extents["rmin"] == pytest.approx(8.0)
    assert extents["rmax"] == pytest.approx(12.0)
    assert extents["zmin"] == pytest.approx(-3.0)
    assert extents["zmax"] == pytest.approx(3.0)
    assert _boundary_domain_check(extents, {"rmin": 7.0, "rmax": 13.0, "zmin": -4.0, "zmax": 4.0})[
        "contained"
    ] is True
    outside = _boundary_domain_check(extents, {"rmin": 9.0, "rmax": 13.0, "zmin": -4.0, "zmax": 4.0})
    assert outside["contained"] is False
    assert outside["margins"]["rmin_margin"] == pytest.approx(-1.0)


def test_vmec2000_sign_probe_updates_flip_phiedge_and_extcur(tmp_path: Path) -> None:
    from tools.diagnostics.compare_freeb_coils_mgrid_vmec2000 import _vmec2000_sign_probe_updates

    mgrid_input = _write_freeb_input(tmp_path / "input.lpqa_mgrid", mgrid_file="mgrid_lpqa_from_essos.nc")

    probes = _vmec2000_sign_probe_updates(mgrid_input)
    by_label = {str(probe["label"]): probe for probe in probes}

    assert set(by_label) == {"flip_phiedge_sign", "flip_extcur_sign", "flip_phiedge_extcur_signs"}
    assert {probe["source"] for probe in probes} == {"sign_convention"}
    assert float(by_label["flip_phiedge_sign"]["updates"]["PHIEDGE"]) == pytest.approx(-FREE_BOUNDARY_PHIEDGE)
    extcur_update = by_label["flip_extcur_sign"]["updates"]["EXTCUR"]
    assert [float(value.strip()) for value in extcur_update.split(",")] == pytest.approx([-1.0])
    combined = by_label["flip_phiedge_extcur_signs"]["updates"]
    assert float(combined["PHIEDGE"]) == pytest.approx(-FREE_BOUNDARY_PHIEDGE)
    assert [float(value.strip()) for value in combined["EXTCUR"].split(",")] == pytest.approx([-1.0])


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
        rmin=0.1,
        rmax=2.5,
        zmin=-1.4,
        zmax=1.4,
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
            "PHIEDGE": FREE_BOUNDARY_PHIEDGE,
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
        free_boundary_activate_fsq=1.0e99,
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
    got_xm_nyq = np.asarray(getattr(got, "xm_nyq", []), dtype=int)
    ref_xm_nyq = np.asarray(getattr(ref, "xm_nyq", []), dtype=int)
    got_xn_nyq = np.asarray(getattr(got, "xn_nyq", []), dtype=int)
    ref_xn_nyq = np.asarray(getattr(ref, "xn_nyq", []), dtype=int)
    if got_xm_nyq.size or ref_xm_nyq.size:
        np.testing.assert_array_equal(got_xm_nyq, ref_xm_nyq)
    if got_xn_nyq.size or ref_xn_nyq.size:
        np.testing.assert_array_equal(got_xn_nyq, ref_xn_nyq)


def _assert_vmec_jax_direct_matches_generated_mgrid_wout(wout_direct, wout_mgrid) -> None:
    _assert_same_wout_layout(wout_direct, wout_mgrid)
    for name in ("rmnc", "zmns", "lmns"):
        _assert_rel_rms(
            f"direct-coil vs generated-mgrid {name}",
            getattr(wout_direct, name),
            getattr(wout_mgrid, name),
            limit=2.0e-3,
        )
    for name in ("iotas", "iotaf"):
        assert np.isfinite(np.asarray(getattr(wout_direct, name), dtype=float)).all()
        assert np.isfinite(np.asarray(getattr(wout_mgrid, name), dtype=float)).all()
    for name in ("aspect", "wb", "wp"):
        np.testing.assert_allclose(
            getattr(wout_direct, name),
            getattr(wout_mgrid, name),
            rtol=2.0e-3,
            atol=1.0e-8,
            err_msg=f"direct-coil and generated-mgrid vmec_jax WOUT mismatch for {name}",
        )
    assert np.isfinite(float(wout_direct.wp))


def _low_order_mode_mask(wout, *, max_m: int = 2, max_abs_n: int = 2) -> np.ndarray:
    xm = np.asarray(wout.xm, dtype=int)
    xn = np.asarray(wout.xn, dtype=int)
    nfp = max(1, int(wout.nfp))
    n = np.rint(xn / float(nfp)).astype(int)
    return (np.abs(xm) <= max_m) & (np.abs(n) <= max_abs_n)


def _low_order_mode_mask_for_array(wout, array, *, max_m: int = 2, max_abs_n: int = 2) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 0:
        raise ValueError("mode arrays must have at least one dimension")
    mode_count = int(arr.shape[-1])
    main_xm = np.asarray(getattr(wout, "xm", []), dtype=int)
    main_xn = np.asarray(getattr(wout, "xn", []), dtype=int)
    nyq_xm = np.asarray(getattr(wout, "xm_nyq", []), dtype=int)
    nyq_xn = np.asarray(getattr(wout, "xn_nyq", []), dtype=int)
    if mode_count == int(main_xm.size):
        xm, xn = main_xm, main_xn
    elif mode_count == int(nyq_xm.size):
        xm, xn = nyq_xm, nyq_xn
    else:
        raise ValueError(
            f"cannot match array mode dimension {mode_count} to main "
            f"({main_xm.size}) or Nyquist ({nyq_xm.size}) WOUT bases"
        )
    nfp = max(1, int(wout.nfp))
    n = np.rint(xn / float(nfp)).astype(int)
    return (np.abs(xm) <= max_m) & (np.abs(n) <= max_abs_n)


def _finite_scalar(wout, name: str) -> float | None:
    if not hasattr(wout, name):
        return None
    value = float(getattr(wout, name))
    assert np.isfinite(value), f"{name}: non-finite scalar {value}"
    return value


def test_low_order_mode_mask_matches_main_and_nyquist_wout_bases() -> None:
    from tools.diagnostics.compare_freeb_coils_mgrid_vmec2000 import (
        _same_layout as diagnostic_same_layout,
        _vmec2000_wout_promotion_quality,
    )

    wout = SimpleNamespace(
        ns=3,
        mpol=3,
        ntor=2,
        nfp=2,
        lasym=False,
        aspect=0.0,
        Aminor_p=0.0,
        Rmajor_p=0.0,
        volume_p=0.0,
        fsqr=11.0,
        fsqz=13.0,
        fsql=1.0e-3,
        xm=np.asarray([0, 1, 2, 3]),
        xn=np.asarray([0, -2, 4, 8]),
        xm_nyq=np.asarray([0, 1, 2, 3, 1, 0]),
        xn_nyq=np.asarray([0, -2, 4, 8, 10, 12]),
    )

    main_mask = _low_order_mode_mask_for_array(wout, np.zeros((3, 4)))
    nyq_mask = _low_order_mode_mask_for_array(wout, np.zeros((3, 6)))

    np.testing.assert_array_equal(main_mask, np.asarray([True, True, True, False]))
    np.testing.assert_array_equal(nyq_mask, np.asarray([True, True, True, False, False, False]))

    same = SimpleNamespace(**vars(wout))
    _assert_same_wout_layout(wout, same)
    assert diagnostic_same_layout(wout, same)

    mismatched_nyq = SimpleNamespace(**vars(wout))
    mismatched_nyq.xn_nyq = np.asarray([0, -2, 4, 8, 10, 14])
    with pytest.raises(AssertionError):
        _assert_same_wout_layout(wout, mismatched_nyq)
    assert not diagnostic_same_layout(wout, mismatched_nyq)

    nonpromotable = _vmec2000_wout_promotion_quality(wout)
    assert nonpromotable["promotable"] is False
    assert "nonpositive_geometry_scalars" in nonpromotable["reasons"]

    promotable = SimpleNamespace(**vars(wout))
    promotable.aspect = 5.0
    promotable.Aminor_p = 1.0
    promotable.Rmajor_p = 5.0
    promotable.volume_p = 100.0
    quality = _vmec2000_wout_promotion_quality(promotable)
    assert quality["promotable"] is True
    assert quality["reasons"] == []


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
        for name in ("rmnc", "zmns", "lmns", "bmnc", "gmnc", "bsubumnc", "bsubvmnc", "bsupumnc", "bsupvmnc"):
            if hasattr(wout_jax, name) and hasattr(wout_vmec2000, name):
                low_order = _low_order_mode_mask_for_array(wout_vmec2000, getattr(wout_vmec2000, name))
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


def _classify_vmec2000_no_wout(vmec2000) -> dict[str, object]:
    from tools.diagnostics.compare_freeb_coils_mgrid_vmec2000 import (
        _classify_vmec2000_result_summary,
        _vmec2000_summary,
    )

    summary = _vmec2000_summary(vmec2000)
    _classify_vmec2000_result_summary(summary, wout_path=_vmec2000_wout_path(vmec2000))
    return summary


def _vmec2000_has_active_vacuum_evidence(vmec2000) -> bool:
    """Return true when VMEC2000 trace rows show non-default vacuum balance."""

    for stage in vmec2000.stages:
        for row in stage.rows:
            if row.delbsq is None or row.fedge is None:
                continue
            default_edge_balance = abs(float(row.delbsq) - 1.0) <= 1.0e-12 and abs(float(row.fedge)) <= 1.0e-14
            if not default_edge_balance:
                return True
    return False


def _write_vmec2000_no_wout_report(path: Path, vmec2000, *, classified_summary: dict[str, object] | None = None) -> None:
    threed_tail: list[str] = []
    if vmec2000.threed1_path is not None and vmec2000.threed1_path.exists():
        threed_tail = vmec2000.threed1_path.read_text(errors="replace").splitlines()[-80:]
    payload = {
        "status": "vmec2000_no_wout",
        "workdir": str(vmec2000.workdir),
        "input_path": str(vmec2000.input_path),
        "returncode": int(getattr(vmec2000, "returncode", 0)),
        "runtime_s": float(vmec2000.runtime_s),
        "stdout_tail": vmec2000.stdout.splitlines()[-40:],
        "stderr_tail": vmec2000.stderr.splitlines()[-40:],
        "files": sorted(p.name for p in vmec2000.workdir.iterdir()),
        "threed1_path": None if vmec2000.threed1_path is None else str(vmec2000.threed1_path),
        "threed1_tail": threed_tail,
        "classified_status": None if classified_summary is None else classified_summary.get("status"),
        "classified_reason": None if classified_summary is None else classified_summary.get("reason"),
        "classified_wout_path": None if classified_summary is None else str(classified_summary.get("wout_path")),
        "underconverged": None if classified_summary is None else classified_summary.get("underconverged"),
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
                    "beta": stage.rows[-1].beta,
                    "avg_m": stage.rows[-1].avg_m,
                    "delbsq": stage.rows[-1].delbsq,
                    "fedge": stage.rows[-1].fedge,
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

    run_mgrid, wout_mgrid = _run_vmec_jax_freeb(mgrid_input)
    run_direct, wout_direct = _run_vmec_jax_freeb(direct_input, direct_params=direct_params)

    _assert_vmec_jax_direct_matches_generated_mgrid_wout(wout_direct, wout_mgrid)
    assert run_mgrid.result.diagnostics["free_boundary"]["vacuum_stub"] is False
    assert run_direct.result.diagnostics["free_boundary"]["vacuum_stub"] is False


@pytest.mark.vmec2000
def test_vmec2000_generated_mgrid_trace_smoke_records_iteration_rows(tmp_path: Path) -> None:
    """Optional VMEC2000 trace gate below the full WOUT-parity xfail.

    This does not promote generated-``mgrid`` WOUT parity.  It proves the
    diagnostic can run VMEC2000 on the same generated grid, records at least one
    VMEC2000 iteration row, confirms VMEC2000 opened the generated grid, parses
    DEL-BSQ/FEDGE edge-balance metadata, and keeps the shared multigrid
    schedule promotable.
    """

    if os.environ.get("VMEC2000_INTEGRATION", "0") != "1":
        pytest.skip("Set VMEC2000_INTEGRATION=1 to run VMEC2000 executable parity tests")
    exe = find_vmec2000_exec()
    if exe is None:
        pytest.skip("xvmec2000 executable not found")
    _load_lpqa_essos_coils()

    from tools.diagnostics.compare_freeb_coils_mgrid_vmec2000 import main as compare_main

    out = tmp_path / "freeb_coils_trace_smoke.json"
    rc = compare_main(
        [
            "--vmec2000-exec",
            str(exe),
            "--ns-array",
            "5,7",
            "--niter-array",
            "2,2",
            "--ftol-array",
            "1e-8,1e-8",
            "--mpol",
            "3",
            "--ntor",
            "2",
            "--mgrid-nphi",
            "4",
            "--nzeta",
            "4",
            "--nvacskip",
            "4",
            "--activate-fsq",
            "1e99",
            "--vmec2000-timeout",
            "120",
            "--out",
            str(out),
            "--workdir",
            str(tmp_path / "freeb_coils_trace_smoke_work"),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["comparisons"]["vmec_jax_direct_vs_generated_mgrid"]["passed"] is True
    assert payload["comparisons"]["vmec_jax_direct_vs_generated_mgrid"]["active_free_boundary"]["both_active"] is True
    assert payload["configuration"]["uses_multigrid_schedule"] is True
    assert payload["configuration"]["mixed_vmec2000_schedule_non_promotable"] is False
    vmec2000 = payload["backends"]["vmec2000_generated_mgrid"]
    assert vmec2000["status"] in {"completed", "no_wout", "more_iter_exit", "nonzero_exit"}
    assert vmec2000["opened_mgrid"] is True
    assert vmec2000["iteration_row_count"] > 0
    assert vmec2000["last_row"] is not None
    assert np.isfinite(float(vmec2000["last_row"]["delbsq"]))
    assert np.isfinite(float(vmec2000["last_row"]["fedge"]))
    if vmec2000["status"] in {"no_wout", "more_iter_exit", "nonzero_exit"}:
        underconverged = vmec2000["underconverged"]
        assert underconverged["classification"] in {
            "reached_niter_without_wout",
            "vmec2000_vacuum_inactive_force_gate",
            "vmec2000_more_iter_exit",
            "vmec2000_runtime_error",
            "vmec2000_nonzero_exit",
            "vmec2000_requested_more_iterations",
            "unknown_no_wout",
        }
        assert np.isfinite(float(underconverged["delbsq_last"]))
        assert np.isfinite(float(underconverged["fedge_last"]))
        assert np.isfinite(float(underconverged["delbsq_over_ftolv"]))
        if underconverged["classification"] == "vmec2000_vacuum_inactive_force_gate":
            assert underconverged["vmec2000_vacuum_activation_blocked"] is True
            assert underconverged["opened_mgrid"] is True
            assert float(underconverged["physical_force_gate_last"]) > float(
                underconverged["physical_force_gate_threshold"]
            )
    else:
        assert payload["summary"]["vmec2000_wout_available"] is True


@pytest.mark.vmec2000
def test_vmec2000_generated_mgrid_free_boundary_matches_vmec_jax_and_direct_coils(tmp_path: Path) -> None:
    """Optional three-way parity gate for generated-mgrid/free-boundary cases.

    Run with `VMEC2000_INTEGRATION=1`.  The intended passing condition is:

    1. VMEC2000 free-boundary from ESSOS-generated mgrid,
    2. `vmec_jax` free-boundary from the same mgrid,
    3. `vmec_jax` free-boundary from direct ESSOS/JAX Biot-Savart coils,

    all produce matching WOUT-level equilibrium quantities.  The only bounded
    xfail is VMEC2000 not promoting the generated-mgrid run to a WOUT; if a
    WOUT exists, the WOUT-level parity assertions are expected to pass.
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
        classified_summary = _classify_vmec2000_no_wout(vmec2000)
        report_path = tmp_path / "vmec2000_no_wout_report.json"
        _write_vmec2000_no_wout_report(report_path, vmec2000, classified_summary=classified_summary)
        xfail_classifications = {
            "reached_niter_without_wout",
            "vmec2000_vacuum_inactive_force_gate",
            "vmec2000_more_iter_exit",
            "vmec2000_requested_more_iterations",
            "unknown_no_wout",
        }
        underconverged = classified_summary.get("underconverged") or {}
        if (
            classified_summary.get("status") in {"no_wout", "more_iter_exit"}
            and isinstance(underconverged, dict)
            and underconverged.get("classification") in xfail_classifications
        ):
            pytest.xfail(
                "Generated ESSOS-mgrid VMEC2000 WOUT promotion blocker: "
                f"status={classified_summary.get('status')} "
                f"classification={underconverged.get('classification')}; report={report_path}"
            )
        pytest.fail(
            "VMEC2000 did not produce a WOUT for a non-promotable reason: "
            f"status={classified_summary.get('status')} reason={classified_summary.get('reason')}; report={report_path}"
        )
    if not _vmec2000_has_active_vacuum_evidence(vmec2000):
        pytest.xfail(
            "Generated ESSOS-mgrid VMEC2000 wrote a WOUT without active-vacuum "
            "trace evidence; this is not promotable free-boundary parity evidence."
        )
    wout_vmec2000 = read_wout(wout_vmec2000_path)

    _write_wout_gap_report(
        tmp_path / "vmec2000_wout_gap_report.json",
        wout_jax=wout_mgrid,
        wout_vmec2000=wout_vmec2000,
    )
    _assert_vmec2000_generated_mgrid_wout_matches_vmec_jax(wout_mgrid, wout_vmec2000)
    _assert_fsq_total_same_scale(wout_mgrid, wout_vmec2000)

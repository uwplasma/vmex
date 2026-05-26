#!/usr/bin/env python3
"""Small direct-coil free-boundary solve benchmark.

The default case uses a synthetic circular coil and a tiny generated
free-boundary input deck.  An optional ESSOS fixture case can be requested with
``--include-essos``; it writes a skipped JSON case when ESSOS or the coil JSON is
not available.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / "results" / "bench_freeb_direct_coil_solve.json"
DEFAULT_WORKDIR = REPO_ROOT / "tmp" / "bench_freeb_direct_coil_solve"
DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
ESSOS_COILS_NAME = "ESSOS_biot_savart_LandremanPaulQA.json"
FINITE_PRESSURE_SCALE = 34.46233666638


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON summary path.")
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="Generated input/output directory.")
    p.add_argument("--max-iter", type=int, default=2, help="Tiny synthetic solve iteration budget.")
    p.add_argument("--warm-repeats", type=int, default=1, help="Warm solve repeats after the cold solve.")
    p.add_argument("--jit-forces", action="store_true", help="Enable JIT force kernels. Off by default for bounded CPU runs.")
    p.add_argument("--activate-fsq", type=float, default=1.0e99, help="Force active direct-coil NESTOR coupling early.")
    p.add_argument("--synthetic-ns", type=int, default=7, help="Synthetic direct-coil radial grid size.")
    p.add_argument("--synthetic-mpol", type=int, default=4, help="Synthetic direct-coil poloidal mode count.")
    p.add_argument("--synthetic-ntor", type=int, default=0, help="Synthetic direct-coil toroidal mode count.")
    p.add_argument("--synthetic-nzeta", type=int, default=2, help="Synthetic direct-coil toroidal grid size.")
    p.add_argument("--synthetic-ntheta", type=int, default=8, help="Synthetic direct-coil poloidal grid size.")
    p.add_argument("--synthetic-nvacskip", type=int, default=1, help="Synthetic direct-coil vacuum refresh cadence.")
    p.add_argument("--synthetic-n-segments", type=int, default=64, help="Synthetic direct-coil segments per base coil.")
    p.add_argument("--synthetic-n-base-coils", type=int, default=1, help="Synthetic direct-coil base coil count.")
    p.add_argument("--synthetic-nfp", type=int, default=1, help="Synthetic direct-coil field periods.")
    p.add_argument("--synthetic-stellsym", action="store_true", help="Expand synthetic coils with stellarator symmetry.")
    p.add_argument("--synthetic-chunk-size", type=int, default=0, help="Synthetic direct-coil point chunk size; 0 disables chunking.")
    p.add_argument("--include-essos", action="store_true", help="Also run an optional small ESSOS fixture case.")
    p.add_argument("--coils-json", type=Path, default=None, help="Optional ESSOS coil JSON.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Base input for optional ESSOS case.")
    p.add_argument("--essos-max-iter", type=int, default=1)
    p.add_argument("--essos-ns", type=int, default=12)
    p.add_argument("--essos-mpol", type=int, default=4)
    p.add_argument("--essos-ntor", type=int, default=4)
    p.add_argument("--essos-nzeta", type=int, default=6)
    p.add_argument("--enable-x64", action=argparse.BooleanOptionalAction, default=True)
    return p


def _backend_info() -> dict[str, Any]:
    from vmec_jax._compat import has_jax, jax, x64_enabled

    info: dict[str, Any] = {"has_jax": bool(has_jax()), "x64_enabled": bool(x64_enabled())}
    if jax is None:
        info.update({"backend": "numpy", "devices": []})
        return info
    try:
        devices = jax.devices()
    except Exception as exc:
        devices = []
        info["devices_error"] = repr(exc)
    info.update(
        {
            "backend": str(jax.default_backend()),
            "devices": [str(device) for device in devices],
            "platforms": sorted({str(getattr(device, "platform", "unknown")) for device in devices}),
        }
    )
    return info


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return _jsonify(arr.item())
    except Exception:
        pass
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(data), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _block_until_ready(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    """Synchronize queued JAX work reachable from a benchmark result."""

    if _depth > 8:
        return value
    if _seen is None:
        _seen = set()
    if isinstance(value, (str, bytes, int, float, bool, Path, type(None))):
        return value
    value_id = id(value)
    if value_id in _seen:
        return value
    _seen.add(value_id)

    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item, _seen=_seen, _depth=_depth + 1)
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            _block_until_ready(item, _seen=_seen, _depth=_depth + 1)
        return value
    if is_dataclass(value):
        for field in fields(value):
            try:
                item = getattr(value, field.name)
            except Exception:
                continue
            _block_until_ready(item, _seen=_seen, _depth=_depth + 1)
        return value
    return value


def _time_once(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    value = fn()
    _block_until_ready(value)
    return float(time.perf_counter() - t0), value


def _summarize_timings(times: list[float]) -> dict[str, float | int | None]:
    if not times:
        return {"repeats": 0, "mean_s": None, "min_s": None, "max_s": None}
    arr = np.asarray(times, dtype=float)
    return {
        "repeats": int(arr.size),
        "mean_s": float(np.mean(arr)),
        "min_s": float(np.min(arr)),
        "max_s": float(np.max(arr)),
    }


def _as_1d_float_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=float)


def _as_1d_int_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=int).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=int)


def _pad_1d(array: np.ndarray, size: int, *, value: float | int) -> np.ndarray:
    if array.size >= size:
        return array[:size]
    return np.pad(array, (0, size - int(array.size)), constant_values=value)


def _summarize_seconds(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "total_s": 0.0, "mean_s": None, "min_s": None, "max_s": None}
    return {
        "count": int(finite.size),
        "total_s": float(np.sum(finite)),
        "mean_s": float(np.mean(finite)),
        "min_s": float(np.min(finite)),
        "max_s": float(np.max(finite)),
    }


def _active_nestor_timing_summary(diag: dict[str, Any]) -> dict[str, Any]:
    sample = _as_1d_float_array(diag.get("freeb_nestor_sample_time_history", []))
    solve = _as_1d_float_array(diag.get("freeb_nestor_solve_time_history", []))
    reused = _as_1d_int_array(diag.get("freeb_nestor_reused_history", []))
    full_update = _as_1d_int_array(diag.get("freeb_full_update_history", []))
    size = max(int(sample.size), int(solve.size), int(reused.size), int(full_update.size))
    if size == 0:
        return {
            "recorded_steps": 0,
            "active_steps": 0,
            "full_update_steps": 0,
            "reused_steps": 0,
            "sampled_steps": 0,
            "sample_time_s": _summarize_seconds(np.zeros((0,), dtype=float)),
            "solve_time_s": _summarize_seconds(np.zeros((0,), dtype=float)),
        }

    sample = _pad_1d(sample, size, value=0.0)
    solve = _pad_1d(solve, size, value=0.0)
    reused = _pad_1d(reused, size, value=0)
    full_update = _pad_1d(full_update, size, value=0)
    active = (full_update != 0) | (reused != 0) | (sample > 0.0) | (solve > 0.0)
    return {
        "recorded_steps": size,
        "active_steps": int(np.count_nonzero(active)),
        "full_update_steps": int(np.count_nonzero(full_update)),
        "reused_steps": int(np.count_nonzero(reused)),
        "sampled_steps": int(np.count_nonzero(sample > 0.0)),
        "sample_time_s": _summarize_seconds(sample[active]),
        "solve_time_s": _summarize_seconds(solve[active]),
    }


def _trial_nestor_timing_summary(diag: dict[str, Any]) -> dict[str, Any]:
    sample = _as_1d_float_array(diag.get("freeb_nestor_trial_sample_time_history", []))
    solve = _as_1d_float_array(diag.get("freeb_nestor_trial_solve_time_history", []))
    reused = _as_1d_int_array(diag.get("freeb_nestor_trial_reused_history", []))
    failed = _as_1d_int_array(diag.get("freeb_nestor_trial_failed_history", []))
    size = max(int(sample.size), int(solve.size), int(reused.size), int(failed.size))
    if size == 0:
        return {
            "recorded_calls": 0,
            "failed_calls": 0,
            "reused_calls": 0,
            "sampled_calls": 0,
            "sample_time_s": _summarize_seconds(np.zeros((0,), dtype=float)),
            "solve_time_s": _summarize_seconds(np.zeros((0,), dtype=float)),
        }

    sample = _pad_1d(sample, size, value=0.0)
    solve = _pad_1d(solve, size, value=0.0)
    reused = _pad_1d(reused, size, value=0)
    failed = _pad_1d(failed, size, value=0)
    return {
        "recorded_calls": size,
        "failed_calls": int(np.count_nonzero(failed)),
        "reused_calls": int(np.count_nonzero(reused)),
        "sampled_calls": int(np.count_nonzero(sample > 0.0)),
        "sample_time_s": _summarize_seconds(sample),
        "solve_time_s": _summarize_seconds(solve),
    }


def _timing_improvement(cold_total_s: Any, warm_total_s: Any) -> dict[str, float | None]:
    try:
        cold = float(cold_total_s)
        warm = float(warm_total_s)
    except Exception:
        return {
            "cold_total_s": None,
            "warm_total_s": None,
            "delta_s": None,
            "speedup": None,
            "reduction_pct": None,
        }
    if not (np.isfinite(cold) and np.isfinite(warm)):
        cold = None
        warm = None
    if cold is None or warm is None:
        return {
            "cold_total_s": cold,
            "warm_total_s": warm,
            "delta_s": None,
            "speedup": None,
            "reduction_pct": None,
        }
    return {
        "cold_total_s": cold,
        "warm_total_s": warm,
        "delta_s": float(cold - warm),
        "speedup": None if warm <= 0.0 else float(cold / warm),
        "reduction_pct": None if cold <= 0.0 else float(100.0 * (cold - warm) / cold),
    }


def _active_nestor_timing_improvement(
    cold_solver_timing: dict[str, Any],
    warm_solver_timing: dict[str, Any],
) -> dict[str, Any]:
    cold = cold_solver_timing.get("active_nestor_timing_summary", {})
    warm = warm_solver_timing.get("active_nestor_timing_summary", {})
    cold_sample = cold.get("sample_time_s", {}) if isinstance(cold, dict) else {}
    warm_sample = warm.get("sample_time_s", {}) if isinstance(warm, dict) else {}
    cold_solve = cold.get("solve_time_s", {}) if isinstance(cold, dict) else {}
    warm_solve = warm.get("solve_time_s", {}) if isinstance(warm, dict) else {}
    return {
        "cold_active_steps": cold.get("active_steps") if isinstance(cold, dict) else None,
        "warm_active_steps": warm.get("active_steps") if isinstance(warm, dict) else None,
        "sample_time_s": _timing_improvement(cold_sample.get("total_s"), warm_sample.get("total_s")),
        "solve_time_s": _timing_improvement(cold_solve.get("total_s"), warm_solve.get("total_s")),
    }


def _trial_nestor_timing_improvement(
    cold_solver_timing: dict[str, Any],
    warm_solver_timing: dict[str, Any],
) -> dict[str, Any]:
    cold = cold_solver_timing.get("trial_nestor_timing_summary", {})
    warm = warm_solver_timing.get("trial_nestor_timing_summary", {})
    cold_sample = cold.get("sample_time_s", {}) if isinstance(cold, dict) else {}
    warm_sample = warm.get("sample_time_s", {}) if isinstance(warm, dict) else {}
    cold_solve = cold.get("solve_time_s", {}) if isinstance(cold, dict) else {}
    warm_solve = warm.get("solve_time_s", {}) if isinstance(warm, dict) else {}
    return {
        "cold_recorded_calls": cold.get("recorded_calls") if isinstance(cold, dict) else None,
        "warm_recorded_calls": warm.get("recorded_calls") if isinstance(warm, dict) else None,
        "sample_time_s": _timing_improvement(cold_sample.get("total_s"), warm_sample.get("total_s")),
        "solve_time_s": _timing_improvement(cold_solve.get("total_s"), warm_solve.get("total_s")),
    }


def _last_float(value: Any) -> float | None:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    val = float(arr[-1])
    return val if np.isfinite(val) else None


def _fsq_summary(run: Any) -> dict[str, Any]:
    result = run.result
    if result is None:
        return {}
    fsqr = _last_float(getattr(result, "fsqr2_history", None))
    fsqz = _last_float(getattr(result, "fsqz2_history", None))
    fsql = _last_float(getattr(result, "fsql2_history", None))
    return {
        "n_iter": int(getattr(result, "n_iter", -1)),
        "fsqr": fsqr,
        "fsqz": fsqz,
        "fsql": fsql,
        "fsq_sum": None if None in (fsqr, fsqz, fsql) else float(fsqr + fsqz + fsql),
    }


def _free_boundary_summary(run: Any) -> dict[str, Any]:
    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    freeb = diag.get("free_boundary", {}) if isinstance(diag, dict) else {}
    out: dict[str, Any] = {}
    if isinstance(freeb, dict):
        out["vacuum_stub"] = bool(freeb.get("vacuum_stub", True))
        out["nestor_model"] = freeb.get("nestor_model")
        out["final_nestor_recompute_attempted"] = bool(freeb.get("final_nestor_recompute_attempted", False))
        out["final_nestor_recompute_failed"] = bool(freeb.get("final_nestor_recompute_failed", False))
        out["final_nestor_sample_time_s"] = float(freeb.get("final_nestor_sample_time_s", 0.0) or 0.0)
        out["final_nestor_solve_time_s"] = float(freeb.get("final_nestor_solve_time_s", 0.0) or 0.0)
        last_diag = freeb.get("last_nestor_diagnostics") or {}
        out["last_provider_kind"] = last_diag.get("provider_kind")
        if isinstance(last_diag, dict) and last_diag:
            out["last_nestor_diagnostics"] = last_diag
    for key in (
        "freeb_full_update_history",
        "freeb_nestor_reused_history",
        "freeb_nestor_solve_time_history",
        "freeb_nestor_sample_time_history",
    ):
        if isinstance(diag, dict) and key in diag:
            out[key] = diag[key]
    return out


def _solver_timing_summary(run: Any) -> dict[str, Any]:
    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    if not isinstance(diag, dict):
        return {}
    out: dict[str, Any] = {}
    if isinstance(diag.get("timing"), dict):
        out["timing"] = diag["timing"]
    for key in (
        "solve_total_s",
        "compute_forces_first_s",
        "compute_forces_total_s",
        "scan_dispatch_s",
        "scan_ready_s",
        "freeb_nestor_solve_time_history",
        "freeb_nestor_sample_time_history",
        "freeb_nestor_reused_history",
        "freeb_full_update_history",
        "freeb_nestor_trial_solve_time_history",
        "freeb_nestor_trial_sample_time_history",
        "freeb_nestor_trial_reused_history",
        "freeb_nestor_trial_failed_history",
    ):
        if key in diag:
            out[key] = diag[key]
    out["active_nestor_timing_summary"] = _active_nestor_timing_summary(diag)
    out["trial_nestor_timing_summary"] = _trial_nestor_timing_summary(diag)
    return out


def _circle_coil_params(
    *,
    n_segments: int = 64,
    n_base_coils: int = 1,
    nfp: int = 1,
    stellsym: bool = False,
    chunk_size: int | None = None,
) -> Any:
    from vmec_jax._compat import jnp
    from vmec_jax.external_fields import CoilFieldParams

    n_base = max(1, int(n_base_coils))
    dofs = jnp.zeros((n_base, 3, 3), dtype=float)
    for idx in range(n_base):
        phase = 2.0 * np.pi * float(idx) / float(n_base)
        radius = 1.8 + (0.04 * ((idx % 3) - 1) if n_base > 1 else 0.0)
        dofs = dofs.at[idx, 0, 2].set(radius)
        dofs = dofs.at[idx, 1, 1].set(radius)
        if n_base > 1:
            dofs = dofs.at[idx, 0, 0].set(0.08 * np.cos(phase))
            dofs = dofs.at[idx, 1, 0].set(0.08 * np.sin(phase))
            dofs = dofs.at[idx, 2, 1].set(0.04 * np.sin(phase))
            dofs = dofs.at[idx, 2, 2].set(0.04 * np.cos(phase))
    return CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.full((n_base,), 3.0e7 / float(n_base), dtype=float),
        n_segments=max(8, int(n_segments)),
        nfp=max(1, int(nfp)),
        stellsym=bool(stellsym),
        chunk_size=chunk_size,
    )


def _write_tiny_direct_input(
    path: Path,
    *,
    max_iter: int,
    ns: int = 7,
    mpol: int = 4,
    ntor: int = 0,
    nzeta: int = 2,
    ntheta: int = 8,
    nfp: int = 1,
    nvacskip: int = 1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ns_i = max(3, int(ns))
    mpol_i = max(2, int(mpol))
    ntor_i = max(0, int(ntor))
    nzeta_i = max(1, int(nzeta))
    ntheta_i = max(4, int(ntheta))
    nfp_i = max(1, int(nfp))
    nvacskip_i = max(1, int(nvacskip))
    path.write_text(
        f"""
&INDATA
  LFREEB = T
  MGRID_FILE = 'DIRECT_COILS'
  EXTCUR = 1.0
  LASYM = F
  NFP = {nfp_i}
  MPOL = {mpol_i}
  NTOR = {ntor_i}
  NS = {ns_i}
  NZETA = {nzeta_i}
  NTHETA = {ntheta_i}
  NS_ARRAY = {ns_i}
  FTOL_ARRAY = 1.0E-8
  NITER_ARRAY = {int(max_iter)}
  NITER = {int(max_iter)}
  FTOL = 1.0E-8
  NSTEP = 20
  NVACSKIP = {nvacskip_i}
  GAMMA = 0.0
  PHIEDGE = 1.0
  CURTOR = 0.0
  SPRES_PED = 1.0
  NCURR = 0
  PRES_SCALE = 1.0E4
  AM = 1.0 -1.0
  AI = 0.4 0.0
  AC = 0.0
  RAXIS = 1.0
  ZAXIS = 0.0
  RBC(0,0) = 1.0  ZBS(0,0) = 0.0
  RBC(0,1) = 0.25 ZBS(0,1) = 0.25
  RBC(0,2) = 0.03 ZBS(0,2) = 0.00
/
""".lstrip()
    )
    return path


def _candidate_essos_dirs() -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("ESSOS_INPUT_DIR"):
        candidates.append(Path(os.environ["ESSOS_INPUT_DIR"]).expanduser())
    candidates.extend(
        [
            REPO_ROOT.parent / "ESSOS_mgrid_pr" / "examples" / "input_files",
            REPO_ROOT.parent / "ESSOS" / "examples" / "input_files",
        ]
    )
    return candidates


def _find_essos_json(requested: Path | None) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"requested --coils-json does not exist: {path}")
        return path
    for directory in _candidate_essos_dirs():
        path = directory / ESSOS_COILS_NAME
        if path.exists():
            return path
    searched = "\n  ".join(str(path) for path in _candidate_essos_dirs())
    raise FileNotFoundError(f"could not find {ESSOS_COILS_NAME}; searched:\n  {searched}")


def _write_essos_input(path: Path, args: argparse.Namespace) -> tuple[Path, Any, dict[str, Any]]:
    from essos.coils import Coils_from_json
    from vmec_jax.external_fields import from_essos_coils
    from vmec_jax.namelist import read_indata, write_indata

    base_input = args.input.expanduser().resolve()
    if not base_input.exists():
        raise FileNotFoundError(f"base free-boundary input does not exist: {base_input}")
    coils_json = _find_essos_json(args.coils_json)
    coils = Coils_from_json(str(coils_json))
    params = from_essos_coils(coils, chunk_size=128)

    indata = deepcopy(read_indata(base_input))
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": "DIRECT_COILS",
            "EXTCUR": [1.0],
            "NS_ARRAY": [int(args.essos_ns)],
            "NITER_ARRAY": [int(args.essos_max_iter)],
            "FTOL_ARRAY": [1.0e-8],
            "NITER": int(args.essos_max_iter),
            "FTOL": 1.0e-8,
            "MPOL": int(args.essos_mpol),
            "NTOR": int(args.essos_ntor),
            "NZETA": int(args.essos_nzeta),
            "NTHETA": 0,
            "NVACSKIP": max(1, int(args.essos_nzeta)),
            "PRES_SCALE": FINITE_PRESSURE_SCALE,
            "AM": [1.0, -1.0],
        }
    )
    write_indata(path, indata)
    return path, params, {"base_input": base_input, "coils_json": coils_json}


def _run_direct_solve(input_path: Path, params: Any, args: argparse.Namespace) -> Any:
    from vmec_jax.driver import run_free_boundary

    return run_free_boundary(
        input_path,
        max_iter=int(args.max_iter),
        multigrid=False,
        verbose=False,
        jit_forces=bool(args.jit_forces),
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
        free_boundary_activate_fsq=float(args.activate_fsq),
    )


def _bench_case(label: str, input_path: Path, params: Any, args: argparse.Namespace) -> dict[str, Any]:
    def run_once() -> Any:
        return _run_direct_solve(input_path, params, args)

    cold_s, cold_run = _time_once(run_once)
    warm_times: list[float] = []
    warm_run = cold_run
    for _ in range(max(0, int(args.warm_repeats))):
        dt, warm_run = _time_once(run_once)
        warm_times.append(dt)
    cold_solver_timing = _solver_timing_summary(cold_run)
    warm_solver_timing = _solver_timing_summary(warm_run)
    return {
        "label": label,
        "status": "completed",
        "input": input_path,
        "cold_or_compile_s": cold_s,
        "warm": _summarize_timings(warm_times),
        "cold_solver_timing": cold_solver_timing,
        "warm_solver_timing": warm_solver_timing,
        "active_nestor_timing_improvement": _active_nestor_timing_improvement(cold_solver_timing, warm_solver_timing),
        "trial_nestor_timing_improvement": _trial_nestor_timing_improvement(cold_solver_timing, warm_solver_timing),
        "fsq": _fsq_summary(warm_run),
        "free_boundary": _free_boundary_summary(warm_run),
    }


def _format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except Exception:
        return "n/a"
    return "n/a" if not np.isfinite(seconds) else f"{seconds:.6f}s"


def _format_speedup(value: Any) -> str:
    try:
        speedup = float(value)
    except Exception:
        return "n/a"
    return "n/a" if not np.isfinite(speedup) else f"{speedup:.2f}x"


def _format_active_nestor_timing(case: dict[str, Any]) -> str:
    improvement = case.get("active_nestor_timing_improvement", {})
    if not isinstance(improvement, dict):
        return ""
    try:
        active_steps = int(improvement.get("cold_active_steps") or 0) + int(
            improvement.get("warm_active_steps") or 0
        )
    except Exception:
        active_steps = 0
    if active_steps <= 0:
        return ""
    sample = improvement.get("sample_time_s", {})
    solve = improvement.get("solve_time_s", {})
    if not isinstance(sample, dict) or not isinstance(solve, dict):
        return ""
    return (
        " active_nestor_sample_total="
        f"{_format_seconds(sample.get('cold_total_s'))}->{_format_seconds(sample.get('warm_total_s'))}"
        f" speedup={_format_speedup(sample.get('speedup'))}"
        " active_nestor_solve_total="
        f"{_format_seconds(solve.get('cold_total_s'))}->{_format_seconds(solve.get('warm_total_s'))}"
    )


def _format_trial_nestor_timing(case: dict[str, Any]) -> str:
    improvement = case.get("trial_nestor_timing_improvement", {})
    if not isinstance(improvement, dict):
        return ""
    try:
        recorded_calls = int(improvement.get("cold_recorded_calls") or 0) + int(
            improvement.get("warm_recorded_calls") or 0
        )
    except Exception:
        recorded_calls = 0
    if recorded_calls <= 0:
        return ""
    sample = improvement.get("sample_time_s", {})
    if not isinstance(sample, dict):
        return ""
    return (
        " trial_nestor_calls="
        f"{improvement.get('cold_recorded_calls')}->{improvement.get('warm_recorded_calls')}"
        " trial_nestor_sample_total="
        f"{_format_seconds(sample.get('cold_total_s'))}->{_format_seconds(sample.get('warm_total_s'))}"
        f" speedup={_format_speedup(sample.get('speedup'))}"
    )


def _format_sampler_diagnostics(case: dict[str, Any]) -> str:
    freeb = case.get("free_boundary", {})
    if not isinstance(freeb, dict):
        return ""
    diag = freeb.get("last_nestor_diagnostics", {})
    if not isinstance(diag, dict) or not diag:
        return ""
    parts: list[str] = []
    if diag.get("sample_points") is not None:
        parts.append(f"sample_points={diag.get('sample_points')}")
    if diag.get("sample_time_s") is not None:
        parts.append(f"final_sample={_format_seconds(diag.get('sample_time_s'))}")
    if diag.get("solve_time_s") is not None:
        parts.append(f"final_solve={_format_seconds(diag.get('solve_time_s'))}")
    if freeb.get("final_nestor_sample_time_s") is not None:
        parts.append(f"final_recompute_sample={_format_seconds(freeb.get('final_nestor_sample_time_s'))}")
    if freeb.get("final_nestor_solve_time_s") is not None:
        parts.append(f"final_recompute_solve={_format_seconds(freeb.get('final_nestor_solve_time_s'))}")
    phase_keys = (
        ("cache", "cache_build_time_s"),
        ("source", "source_time_s"),
        ("bvec", "bvec_time_s"),
        ("matrix", "matrix_time_s"),
        ("linear", "linear_solve_time_s"),
        ("vac_channels", "vacuum_channels_time_s"),
    )
    phase_parts = [
        f"{label}={_format_seconds(diag.get(key))}"
        for label, key in phase_keys
        if diag.get(key) not in (None, 0.0)
    ]
    if phase_parts:
        parts.append("final_solve_breakdown[" + ",".join(phase_parts) + "]")
    if diag.get("physical_matrix_lu_built") is not None:
        parts.append(f"physical_lu={diag.get('physical_matrix_lu_built')}")
    if diag.get("mode_matrix_lu_built") is not None:
        parts.append(f"mode_lu={diag.get('mode_matrix_lu_built')}")
    if diag.get("provider_jit_sampler") is not None:
        parts.append(f"jit_sampler={diag.get('provider_jit_sampler')}")
    if diag.get("provider_chunk_size") is not None:
        parts.append(f"chunk_size={diag.get('provider_chunk_size')}")
    if diag.get("provider_coil_count") is not None:
        parts.append(f"coils={diag.get('provider_coil_count')}")
    if diag.get("provider_segments_per_coil") is not None:
        parts.append(f"segments={diag.get('provider_segments_per_coil')}")
    return "" if not parts else " " + " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.max_iter) < 1:
        raise SystemExit("--max-iter must be >= 1")

    from vmec_jax._compat import enable_x64

    enable_x64(bool(args.enable_x64))
    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "status": "completed",
        "script": str(Path(__file__).resolve()),
        "backend": _backend_info(),
        "parameters": {
            "max_iter": int(args.max_iter),
            "warm_repeats": int(args.warm_repeats),
            "jit_forces": bool(args.jit_forces),
            "activate_fsq": float(args.activate_fsq),
            "synthetic": {
                "ns": int(args.synthetic_ns),
                "mpol": int(args.synthetic_mpol),
                "ntor": int(args.synthetic_ntor),
                "nzeta": int(args.synthetic_nzeta),
                "ntheta": int(args.synthetic_ntheta),
                "nvacskip": int(args.synthetic_nvacskip),
                "n_segments": int(args.synthetic_n_segments),
                "n_base_coils": int(args.synthetic_n_base_coils),
                "nfp": int(args.synthetic_nfp),
                "stellsym": bool(args.synthetic_stellsym),
                "chunk_size": None if int(args.synthetic_chunk_size) <= 0 else int(args.synthetic_chunk_size),
            },
            "env": {
                "VMEC_JAX_FREEB_JIT_COIL_SAMPLER": os.getenv("VMEC_JAX_FREEB_JIT_COIL_SAMPLER"),
                "VMEC_JAX_FREEB_DISABLE_COIL_GEOMETRY_CACHE": os.getenv(
                    "VMEC_JAX_FREEB_DISABLE_COIL_GEOMETRY_CACHE"
                ),
            },
        },
        "cases": [],
    }

    synthetic_input = _write_tiny_direct_input(
        workdir / "input.bench_direct_coil_synthetic",
        max_iter=int(args.max_iter),
        ns=int(args.synthetic_ns),
        mpol=int(args.synthetic_mpol),
        ntor=int(args.synthetic_ntor),
        nzeta=int(args.synthetic_nzeta),
        ntheta=int(args.synthetic_ntheta),
        nfp=int(args.synthetic_nfp),
        nvacskip=int(args.synthetic_nvacskip),
    )
    synthetic_chunk_size = None if int(args.synthetic_chunk_size) <= 0 else int(args.synthetic_chunk_size)
    payload["cases"].append(
        _bench_case(
            "synthetic_direct_coil_solve",
            synthetic_input,
            _circle_coil_params(
                n_segments=int(args.synthetic_n_segments),
                n_base_coils=int(args.synthetic_n_base_coils),
                nfp=int(args.synthetic_nfp),
                stellsym=bool(args.synthetic_stellsym),
                chunk_size=synthetic_chunk_size,
            ),
            args,
        )
    )

    if args.include_essos:
        try:
            essos_input, essos_params, metadata = _write_essos_input(workdir / "input.bench_direct_coil_essos", args)
            essos_args = argparse.Namespace(**vars(args))
            essos_args.max_iter = int(args.essos_max_iter)
            case = _bench_case("essos_direct_coil_solve", essos_input, essos_params, essos_args)
            case["metadata"] = metadata
            payload["cases"].append(case)
        except Exception as exc:
            payload["cases"].append(
                {
                    "label": "essos_direct_coil_solve",
                    "status": "skipped",
                    "reason": "essos_or_free_boundary_fixture_unavailable",
                    "error": repr(exc),
                }
            )
    else:
        payload["cases"].append({"label": "essos_direct_coil_solve", "status": "skipped", "reason": "not_requested"})

    out = args.out.expanduser().resolve()
    _write_json(out, payload)
    print(f"[bench-freeb-direct-coil-solve] wrote {out}")
    for case in payload["cases"]:
        if case["status"] == "completed":
            warm = case["warm"]
            warm_min = "n/a" if warm["min_s"] is None else f"{warm['min_s']:.6f}s"
            print(
                f"[bench-freeb-direct-coil-solve] {case['label']}: "
                f"cold_or_compile={case['cold_or_compile_s']:.6f}s warm_min={warm_min}"
                f"{_format_active_nestor_timing(case)}"
                f"{_format_trial_nestor_timing(case)}"
                f"{_format_sampler_diagnostics(case)}"
            )
        else:
            print(f"[bench-freeb-direct-coil-solve] {case['label']}: skipped ({case.get('reason', 'unknown')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python
"""Compare free-boundary ESSOS coils through direct, mgrid, and VMEC2000 paths.

This diagnostic is intentionally local/optional.  It prefers an ESSOS checkout
with ``Coils.to_mgrid`` support, runs a low-resolution Landreman-Paul QA
free-boundary case through:

1. vmec_jax using an ESSOS-generated mgrid file,
2. vmec_jax using the direct JAX coil provider,
3. VMEC2000 using the same generated mgrid, when ``xvmec2000`` is available.

The output is a JSON report.  Missing optional dependencies are recorded as
skips unless ``--strict`` or the corresponding ``--require-*`` flag is used.

Example:

    python tools/diagnostics/compare_freeb_coils_mgrid_vmec2000.py \
      --out results/freeb_coils_mgrid_vmec2000.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
DEFAULT_OUT = REPO_ROOT / "results" / "freeb_coils_mgrid_vmec2000.json"
DEFAULT_COILS_JSON_NAME = "ESSOS_biot_savart_LandremanPaulQA.json"
DEFAULT_PRESSURE_SCALE = 34.46233666638
TINY = 1.0e-300
VMEC2000_MORE_ITER_RETURNCODE = 2


def _candidate_essos_roots() -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("ESSOS_ROOT"):
        candidates.append(Path(os.environ["ESSOS_ROOT"]).expanduser())
    candidates.extend(
        [
            REPO_ROOT.parent / "ESSOS_mgrid_pr",
            REPO_ROOT.parent / "ESSOS",
        ]
    )
    return candidates


def _candidate_essos_input_dirs(essos_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("ESSOS_INPUT_DIR"):
        candidates.append(Path(os.environ["ESSOS_INPUT_DIR"]).expanduser())
    roots = [essos_root] if essos_root is not None else []
    roots.extend(_candidate_essos_roots())
    for root in roots:
        if root is not None:
            candidates.append(Path(root).expanduser() / "examples" / "input_files")
    candidates.append(Path.cwd() / "examples" / "input_files")
    return candidates


def _find_default_coils_json(essos_root: Path | None = None) -> Path:
    for directory in _candidate_essos_input_dirs(essos_root):
        path = directory / DEFAULT_COILS_JSON_NAME
        if path.exists():
            return path.resolve()
    searched = "\n  ".join(str(p) for p in _candidate_essos_input_dirs(essos_root))
    raise FileNotFoundError(
        f"Could not find {DEFAULT_COILS_JSON_NAME}. Set --coils-json or ESSOS_INPUT_DIR. Searched:\n  {searched}"
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON output path.")
    p.add_argument("--workdir", type=Path, default=None, help="Directory for generated inputs, mgrid, and WOUTs.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Base VMEC input deck.")
    p.add_argument("--coils-json", type=Path, default=None, help="ESSOS coil JSON.")
    p.add_argument(
        "--essos-root",
        type=Path,
        default=None,
        help="ESSOS checkout to put first on sys.path before importing essos.",
    )
    p.add_argument("--pressure-scale", type=float, default=DEFAULT_PRESSURE_SCALE)
    p.add_argument(
        "--phiedge-scale",
        type=float,
        default=1.0,
        help="Scale the base input PHIEDGE. Use -1 to test the opposite VMEC vacuum sign convention.",
    )
    p.add_argument(
        "--extcur-scale",
        type=float,
        default=1.0,
        help="External-current scale written to EXTCUR for generated-mgrid VMEC inputs.",
    )
    p.add_argument("--niter", type=int, default=2)
    p.add_argument("--ftol", type=float, default=1.0e-8)
    p.add_argument("--ns", type=int, default=12)
    p.add_argument("--ns-array", type=str, default=None, help="Comma-separated multigrid NS_ARRAY.")
    p.add_argument("--niter-array", type=str, default=None, help="Comma-separated multigrid NITER_ARRAY.")
    p.add_argument("--ftol-array", type=str, default=None, help="Comma-separated multigrid FTOL_ARRAY.")
    p.add_argument("--mpol", type=int, default=4)
    p.add_argument("--ntor", type=int, default=4)
    p.add_argument(
        "--nzeta",
        type=int,
        default=None,
        help="VMEC NZETA. Defaults to --mgrid-nphi so generated mgrid kp is compatible.",
    )
    p.add_argument("--nvacskip", type=int, default=None)
    p.add_argument("--mgrid-nr", type=int, default=12)
    p.add_argument("--mgrid-nz", type=int, default=12)
    p.add_argument("--mgrid-nphi", type=int, default=6)
    p.add_argument("--mgrid-rmin", type=float, default=5.0)
    p.add_argument("--mgrid-rmax", type=float, default=15.0)
    p.add_argument("--mgrid-zmin", type=float, default=-5.0)
    p.add_argument("--mgrid-zmax", type=float, default=5.0)
    p.add_argument(
        "--mgrid-auto-bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Derive mgrid R/Z bounds from the VMEC boundary and padding. "
            "Use --no-mgrid-auto-bounds to honor explicit --mgrid-rmin/rmax/zmin/zmax."
        ),
    )
    p.add_argument(
        "--mgrid-padding-fraction",
        type=float,
        default=0.30,
        help="Fractional boundary-span padding used by --mgrid-auto-bounds.",
    )
    p.add_argument(
        "--mgrid-min-padding",
        type=float,
        default=0.50,
        help="Minimum absolute R/Z padding used by --mgrid-auto-bounds.",
    )
    p.add_argument("--direct-chunk-size", type=int, default=256)
    p.add_argument("--regularization-epsilon", type=float, default=0.0)
    p.add_argument(
        "--activate-fsq",
        type=float,
        default=None,
        help=(
            "Forwarded to vmec_jax run_free_boundary as free_boundary_activate_fsq. "
            "Use a large value such as 1e99 to force active NESTOR/free-boundary "
            "coupling in short diagnostics."
        ),
    )
    p.add_argument("--jit-forces", action="store_true", help="Enable JIT force kernels for vmec_jax solves.")
    p.add_argument("--skip-vmec2000", action="store_true", help="Do not try to run xvmec2000.")
    p.add_argument("--vmec2000-exec", type=Path, default=None, help="Path to xvmec2000.")
    p.add_argument(
        "--vmec2000-niter",
        type=int,
        default=None,
        help="Override VMEC2000 NITER/NITER_ARRAY only; vmec_jax still uses --niter.",
    )
    p.add_argument("--vmec2000-timeout", type=float, default=90.0)
    p.add_argument(
        "--vmec2000-promotion-probes",
        action="store_true",
        help=(
            "If VMEC2000 does not produce a WOUT, run extra bounded diagnostic "
            "probes with looser output/promote settings and record their status."
        ),
    )
    p.add_argument(
        "--vmec2000-probe-ftols",
        type=str,
        default="1e-2",
        help="Comma-separated FTOL/FTOL_ARRAY values for --vmec2000-promotion-probes.",
    )
    p.add_argument(
        "--vmec2000-probe-max-main-iterations",
        type=str,
        default="2,5",
        help="Comma-separated MAX_MAIN_ITERATIONS values for --vmec2000-promotion-probes.",
    )
    p.add_argument("--require-essos", action="store_true", help="Exit nonzero if ESSOS or Coils.to_mgrid is unavailable.")
    p.add_argument("--require-vmec2000", action="store_true", help="Exit nonzero if VMEC2000 is unavailable or no WOUT is produced.")
    p.add_argument(
        "--fail-on-jax-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero when vmec_jax direct-coil and generated-mgrid WOUTs differ beyond tolerance.",
    )
    p.add_argument(
        "--fail-on-vmec2000-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit nonzero when VMEC2000 WOUT comparison limits are exceeded.",
    )
    p.add_argument("--strict", action="store_true", help="Require all optional paths and fail on all comparison mismatches.")
    p.add_argument("--jax-rtol", type=float, default=1.0e-5)
    p.add_argument("--jax-atol", type=float, default=1.0e-7)
    return p


def _diagnostic_nzeta(args: argparse.Namespace) -> int:
    """Return a VMEC NZETA compatible with the generated mgrid kp."""
    return int(args.mgrid_nphi if args.nzeta is None else args.nzeta)


def _sample_input_boundary_extents(indata: Any, *, ntheta: int = 96, nzeta: int = 96) -> dict[str, float]:
    """Return R/Z extrema from VMEC input Fourier boundary coefficients.

    The generated-mgrid diagnostic must contain the actual plasma boundary.
    Hard-coded small mgrid boxes can otherwise make VMEC2000 fail in the
    vacuum solve for reasons unrelated to the direct-coil/mgrid field provider.
    """

    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / max(1, int(indata.scalars.get("NFP", 1))), int(nzeta), endpoint=False)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    nfp = max(1, int(indata.scalars.get("NFP", 1)))

    r = np.zeros_like(theta_grid, dtype=float)
    z = np.zeros_like(theta_grid, dtype=float)
    for name, sign in (("RBC", "cos"), ("RBS", "sin")):
        for (m, n), value in (indata.indexed.get(name) or {}).items():
            angle = int(m) * theta_grid - int(n) * nfp * zeta_grid
            term = np.cos(angle) if sign == "cos" else np.sin(angle)
            r = r + float(value) * term
    for name, sign in (("ZBC", "cos"), ("ZBS", "sin")):
        for (m, n), value in (indata.indexed.get(name) or {}).items():
            angle = int(m) * theta_grid - int(n) * nfp * zeta_grid
            term = np.cos(angle) if sign == "cos" else np.sin(angle)
            z = z + float(value) * term

    return {
        "boundary_rmin": float(np.min(r)),
        "boundary_rmax": float(np.max(r)),
        "boundary_zmin": float(np.min(z)),
        "boundary_zmax": float(np.max(z)),
    }


def _resolve_mgrid_bounds(indata: Any, args: argparse.Namespace) -> dict[str, float | bool]:
    if not bool(args.mgrid_auto_bounds):
        return {
            "auto_bounds": False,
            "rmin": float(args.mgrid_rmin),
            "rmax": float(args.mgrid_rmax),
            "zmin": float(args.mgrid_zmin),
            "zmax": float(args.mgrid_zmax),
        }

    extents = _sample_input_boundary_extents(indata)
    r_span = max(float(extents["boundary_rmax"] - extents["boundary_rmin"]), 1.0e-12)
    z_span = max(float(extents["boundary_zmax"] - extents["boundary_zmin"]), 1.0e-12)
    r_pad = max(float(args.mgrid_min_padding), float(args.mgrid_padding_fraction) * r_span)
    z_pad = max(float(args.mgrid_min_padding), float(args.mgrid_padding_fraction) * z_span)
    rmin = max(1.0e-6, float(extents["boundary_rmin"]) - r_pad)
    rmax = float(extents["boundary_rmax"]) + r_pad
    zmin = float(extents["boundary_zmin"]) - z_pad
    zmax = float(extents["boundary_zmax"]) + z_pad
    if not (rmin < rmax and zmin < zmax):
        raise ValueError(f"invalid auto mgrid bounds from boundary extents: {extents}")
    return {
        "auto_bounds": True,
        "rmin": rmin,
        "rmax": rmax,
        "zmin": zmin,
        "zmax": zmax,
        "padding_fraction": float(args.mgrid_padding_fraction),
        "min_padding": float(args.mgrid_min_padding),
        **extents,
    }


def _parse_int_array(text: str, *, name: str) -> list[int]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise SystemExit(f"--{name} must contain at least one integer")
    try:
        parsed = [int(value) for value in values]
    except ValueError as exc:
        raise SystemExit(f"--{name} must be a comma-separated integer list") from exc
    if any(value < 1 for value in parsed):
        raise SystemExit(f"--{name} values must be >= 1")
    return parsed


def _parse_float_array(text: str, *, name: str) -> list[float]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise SystemExit(f"--{name} must contain at least one float")
    try:
        parsed = [float(value) for value in values]
    except ValueError as exc:
        raise SystemExit(f"--{name} must be a comma-separated float list") from exc
    if any(value <= 0.0 for value in parsed):
        raise SystemExit(f"--{name} values must be > 0")
    return parsed


def _diagnostic_schedule(args: argparse.Namespace) -> tuple[list[int], list[int], list[float]]:
    """Return the shared VMEC/JAX multigrid schedule for this diagnostic."""
    arrays_requested = any(
        value is not None for value in (args.ns_array, args.niter_array, args.ftol_array)
    )
    if not arrays_requested:
        return [int(args.ns)], [int(args.niter)], [float(args.ftol)]
    if args.ns_array is None or args.niter_array is None or args.ftol_array is None:
        raise SystemExit("--ns-array, --niter-array, and --ftol-array must be provided together")
    ns_array = _parse_int_array(args.ns_array, name="ns-array")
    niter_array = _parse_int_array(args.niter_array, name="niter-array")
    ftol_array = _parse_float_array(args.ftol_array, name="ftol-array")
    lengths = {len(ns_array), len(niter_array), len(ftol_array)}
    if len(lengths) != 1:
        raise SystemExit("--ns-array, --niter-array, and --ftol-array must have equal lengths")
    return ns_array, niter_array, ftol_array


def _format_namelist_array(values: list[int] | list[float]) -> str:
    return ", ".join(
        f"{float(value):.16e}" if isinstance(value, float) else str(int(value))
        for value in values
    )


def _format_namelist_float(value: float) -> str:
    return f"{float(value):.16e}"


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
    if isinstance(value, (bool, str, int)) or value is None:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail_text(path: Path | None, *, lines: int) -> list[str]:
    if path is None or not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-int(lines) :]


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(np.ma.filled(value, np.nan), dtype=float)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _last_float(value: Any) -> float | None:
    try:
        arr = _as_float_array(value).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return _safe_float(arr[-1])


def _rms(value: Any) -> float | None:
    try:
        arr = _as_float_array(value)
    except Exception:
        return None
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return float(np.sqrt(np.mean(arr * arr)))


def _fsq_summary_from_run(run: Any) -> dict[str, Any]:
    result = getattr(run, "result", None)
    if result is None:
        return {"available": False}
    fsqr = _last_float(getattr(result, "fsqr2_history", None))
    fsqz = _last_float(getattr(result, "fsqz2_history", None))
    fsql = _last_float(getattr(result, "fsql2_history", None))
    fsq_sum = None if None in (fsqr, fsqz, fsql) else float(fsqr + fsqz + fsql)
    fsq_norm = None if None in (fsqr, fsqz, fsql) else float(np.sqrt(fsqr * fsqr + fsqz * fsqz + fsql * fsql))
    return {
        "available": True,
        "n_iter": int(getattr(result, "n_iter", -1)),
        "fsqr": fsqr,
        "fsqz": fsqz,
        "fsql": fsql,
        "fsq_sum": fsq_sum,
        "fsq_norm": fsq_norm,
        "w_final": _last_float(getattr(result, "w_history", None)),
    }


def _history_summary(value: Any) -> dict[str, Any]:
    """Return compact scalar diagnostics for optional iteration histories."""

    try:
        arr = _as_float_array(value).reshape(-1)
    except Exception:
        return {"available": False}
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"available": False, "size": int(arr.size)}
    nonzero = finite[np.abs(finite) > 0.0]
    return {
        "available": True,
        "size": int(arr.size),
        "finite_size": int(finite.size),
        "nonzero_size": int(nonzero.size),
        "last": float(finite[-1]),
        "sum": float(np.sum(finite)),
        "max": float(np.max(finite)),
    }


def _free_boundary_summary_from_run(run: Any) -> dict[str, Any]:
    """Extract vmec_jax free-boundary diagnostics relevant to VMEC2000 DEL-BSQ gaps."""

    result = getattr(run, "result", None)
    diag = getattr(result, "diagnostics", {}) if result is not None else {}
    if not isinstance(diag, dict):
        return {"available": False}
    freeb = diag.get("free_boundary", {})
    if not isinstance(freeb, dict) or not freeb:
        return {"available": False}

    last_nestor = freeb.get("last_nestor_diagnostics", {})
    if not isinstance(last_nestor, dict):
        last_nestor = {}
    scalar_keys = (
        "provider_kind",
        "mode",
        "rhs_mode",
        "reused",
        "source_reused",
        "sample_ntheta",
        "sample_nzeta",
        "sample_points",
        "br_rms",
        "bp_rms",
        "bz_rms",
        "bnormal_rms",
        "bnormal_unit_rms",
        "rhs_rms",
        "gsource_rms",
        "bsqvac_rms",
        "bsqvac_mean",
        "bvec_mode_rms",
        "bvec_mode_nonsing_rms",
        "bvec_mode_analytic_rms",
        "sample_time_s",
        "solve_time_s",
        "source_time_s",
        "bvec_time_s",
        "matrix_time_s",
        "linear_solve_time_s",
        "vacuum_channels_time_s",
        "final_nestor_sample_time_s",
        "final_nestor_solve_time_s",
    )
    nestor_scalars = {key: _jsonify(last_nestor[key]) for key in scalar_keys if key in last_nestor}
    if "final_nestor_sample_time_s" in freeb:
        nestor_scalars["final_nestor_sample_time_s"] = _jsonify(freeb.get("final_nestor_sample_time_s"))
    if "final_nestor_solve_time_s" in freeb:
        nestor_scalars["final_nestor_solve_time_s"] = _jsonify(freeb.get("final_nestor_solve_time_s"))

    history_keys = (
        "freeb_nestor_sample_time_history",
        "freeb_nestor_solve_time_history",
        "freeb_nestor_trial_sample_time_history",
        "freeb_nestor_trial_solve_time_history",
        "freeb_full_update_history",
        "freeb_nestor_reused_history",
        "freeb_nestor_trial_reused_history",
        "freeb_nestor_trial_failed_history",
    )
    histories = {
        key: _history_summary(diag.get(key))
        for key in history_keys
        if key in diag
    }

    return {
        "available": True,
        "enabled": bool(freeb.get("enabled", False)),
        "couple_edge": bool(freeb.get("couple_edge", False)),
        "ivac": _jsonify(freeb.get("ivac")),
        "ivacskip": _jsonify(freeb.get("ivacskip")),
        "nvacskip": _jsonify(freeb.get("nvacskip")),
        "nestor_model": str(freeb.get("nestor_model", "")),
        "vacuum_stub": bool(freeb.get("vacuum_stub", True)),
        "final_nestor_recompute_attempted": bool(freeb.get("final_nestor_recompute_attempted", False)),
        "final_nestor_recompute_failed": bool(freeb.get("final_nestor_recompute_failed", False)),
        "last_nestor_diagnostics": nestor_scalars,
        "histories": histories,
    }


def _fsq_total_from_wout(wout: Any) -> float | None:
    values = [_safe_float(getattr(wout, name, None)) for name in ("fsqr", "fsqz", "fsql")]
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _layout_summary(wout: Any) -> dict[str, Any]:
    return {
        "ns": int(wout.ns),
        "mpol": int(wout.mpol),
        "ntor": int(wout.ntor),
        "nfp": int(wout.nfp),
        "lasym": bool(wout.lasym),
        "signgs": int(getattr(wout, "signgs", 0)),
        "mnmax": int(getattr(wout, "mnmax", len(getattr(wout, "xm", [])))),
        "mnmax_nyq": int(getattr(wout, "mnmax_nyq", len(getattr(wout, "xm_nyq", [])))),
    }


def _wout_boundary_extents(wout: Any, *, ntheta: int = 64, nphi: int = 64) -> dict[str, Any]:
    """Return approximate LCFS R/Z extrema from WOUT boundary coefficients."""

    xm = _as_float_array(getattr(wout, "xm", []))
    xn = _as_float_array(getattr(wout, "xn", []))
    rmnc = _as_float_array(getattr(wout, "rmnc", []))
    zmns = _as_float_array(getattr(wout, "zmns", []))
    if xm.size == 0 or xn.size == 0 or rmnc.ndim != 2 or zmns.ndim != 2:
        return {"available": False}
    if rmnc.shape[1] != xm.size or zmns.shape[1] != xm.size:
        return {"available": False, "reason": "mode_shape_mismatch"}
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    phi_period = 2.0 * np.pi / max(1, int(getattr(wout, "nfp", 1)))
    phi = np.linspace(0.0, phi_period, int(nphi), endpoint=False)
    phase = xm[:, None, None] * theta[None, :, None] - xn[:, None, None] * phi[None, None, :]
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    rmnc_edge = rmnc[-1]
    zmns_edge = zmns[-1]
    rmns = _as_float_array(getattr(wout, "rmns", np.zeros_like(rmnc)))
    zmnc = _as_float_array(getattr(wout, "zmnc", np.zeros_like(zmns)))
    if (not bool(getattr(wout, "lasym", False))) or rmns.shape != rmnc.shape:
        rmns_edge = np.zeros_like(rmnc_edge)
    else:
        rmns_edge = rmns[-1]
    if (not bool(getattr(wout, "lasym", False))) or zmnc.shape != zmns.shape:
        zmnc_edge = np.zeros_like(zmns_edge)
    else:
        zmnc_edge = zmnc[-1]
    r = np.einsum("m,mtz->tz", rmnc_edge, cos_phase) + np.einsum("m,mtz->tz", rmns_edge, sin_phase)
    z = np.einsum("m,mtz->tz", zmnc_edge, cos_phase) + np.einsum("m,mtz->tz", zmns_edge, sin_phase)
    return {
        "available": True,
        "ntheta": int(ntheta),
        "nphi": int(nphi),
        "rmin": float(np.min(r)),
        "rmax": float(np.max(r)),
        "zmin": float(np.min(z)),
        "zmax": float(np.max(z)),
    }


def _boundary_domain_check(boundary_extents: dict[str, Any], mgrid_bounds: dict[str, Any]) -> dict[str, Any]:
    """Classify whether a WOUT boundary stays inside generated-mgrid bounds."""

    if not bool(boundary_extents.get("available", False)):
        return {"available": False, "contained": None}
    required = ("rmin", "rmax", "zmin", "zmax")
    if any(key not in mgrid_bounds for key in required):
        return {"available": False, "contained": None, "reason": "missing_mgrid_bounds"}
    margins = {
        "rmin_margin": float(boundary_extents["rmin"]) - float(mgrid_bounds["rmin"]),
        "rmax_margin": float(mgrid_bounds["rmax"]) - float(boundary_extents["rmax"]),
        "zmin_margin": float(boundary_extents["zmin"]) - float(mgrid_bounds["zmin"]),
        "zmax_margin": float(mgrid_bounds["zmax"]) - float(boundary_extents["zmax"]),
    }
    return {
        "available": True,
        "contained": bool(all(value >= 0.0 for value in margins.values())),
        "margins": margins,
    }


def _wout_summary(wout: Any) -> dict[str, Any]:
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
    iotas = _as_float_array(getattr(wout, "iotas", []))
    iotaf = _as_float_array(getattr(wout, "iotaf", []))
    return {
        "path": getattr(wout, "path", None),
        "layout": _layout_summary(wout),
        "scalars": {name: _safe_float(getattr(wout, name, None)) for name in scalar_names},
        "fsq_total": _fsq_total_from_wout(wout),
        "iotas_mean_no_axis": _safe_float(np.mean(iotas[1:])) if iotas.size > 1 else _safe_float(np.mean(iotas)),
        "iotaf_mean": _safe_float(np.mean(iotaf)) if iotaf.size else None,
        "boundary_extents": _wout_boundary_extents(wout),
    }


def _vmec2000_wout_promotion_quality(wout: Any) -> dict[str, Any]:
    """Classify whether a VMEC2000 WOUT is usable as parity evidence.

    Some generated-``mgrid`` sign probes can produce a parseable VMEC2000 WOUT
    even though VMEC2000 has not formed a physically promotable equilibrium
    record: geometry scalars such as aspect, major/minor radius, and volume may
    be zero while low-order Fourier arrays are present.  That WOUT is useful
    diagnostic evidence but must not be counted as external parity evidence.
    """

    geometry_names = ("aspect", "Aminor_p", "Rmajor_p", "volume_p")
    geometry_scalars = {name: _safe_float(getattr(wout, name, None)) for name in geometry_names}
    reasons: list[str] = []
    bad_geometry = [
        name
        for name, value in geometry_scalars.items()
        if value is None or (not np.isfinite(value)) or float(value) <= TINY
    ]
    if bad_geometry:
        reasons.append("nonpositive_geometry_scalars")
    fsq_total = _fsq_total_from_wout(wout)
    if fsq_total is None or (not np.isfinite(float(fsq_total))):
        reasons.append("nonfinite_fsq_total")
    return {
        "promotable": not reasons,
        "reasons": reasons,
        "geometry_scalars": geometry_scalars,
        "fsq_total": fsq_total,
    }


def _scalar_gap(got_wout: Any, ref_wout: Any, name: str, *, rtol: float | None = None, atol: float | None = None) -> dict[str, Any]:
    if not (hasattr(got_wout, name) and hasattr(ref_wout, name)):
        return {"available": False}
    got = _safe_float(getattr(got_wout, name))
    ref = _safe_float(getattr(ref_wout, name))
    if got is None or ref is None:
        return {"available": False, "got": got, "ref": ref}
    abs_delta = abs(got - ref)
    rel_delta = abs_delta / max(abs(ref), TINY)
    out: dict[str, Any] = {
        "available": True,
        "got": got,
        "ref": ref,
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
    }
    if rtol is not None and atol is not None:
        out["within_tolerance"] = bool(np.allclose(got, ref, rtol=float(rtol), atol=float(atol), equal_nan=False))
        out["rtol"] = float(rtol)
        out["atol"] = float(atol)
    return out


def _array_gap(
    got: Any,
    ref: Any,
    *,
    radial_skip: int = 0,
    mode_mask: np.ndarray | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, Any]:
    got_arr = _as_float_array(got)
    ref_arr = _as_float_array(ref)
    if got_arr.shape != ref_arr.shape:
        return {
            "shape_mismatch": True,
            "got_shape": list(got_arr.shape),
            "ref_shape": list(ref_arr.shape),
            "within_tolerance": False if rtol is not None and atol is not None else None,
        }
    if radial_skip and got_arr.ndim >= 1:
        got_arr = got_arr[int(radial_skip) :, ...]
        ref_arr = ref_arr[int(radial_skip) :, ...]
    if mode_mask is not None:
        got_arr = got_arr[..., mode_mask]
        ref_arr = ref_arr[..., mode_mask]
    if got_arr.size == 0:
        return {"size": 0, "within_tolerance": False if rtol is not None and atol is not None else None}
    diff = got_arr - ref_arr
    abs_rms = float(np.sqrt(np.mean(diff * diff)))
    ref_rms = float(np.sqrt(np.mean(ref_arr * ref_arr)))
    out: dict[str, Any] = {
        "size": int(got_arr.size),
        "shape": list(got_arr.shape),
        "abs_rms_delta": abs_rms,
        "rel_rms_delta": abs_rms / max(ref_rms, TINY),
        "max_abs_delta": float(np.max(np.abs(diff))),
    }
    if rtol is not None and atol is not None:
        rms_tolerance = float(atol) + float(rtol) * ref_rms
        out["within_tolerance"] = bool(abs_rms <= rms_tolerance)
        out["rms_tolerance"] = rms_tolerance
        out["rtol"] = float(rtol)
        out["atol"] = float(atol)
    return out


def _same_layout(got_wout: Any, ref_wout: Any) -> bool:
    got = _layout_summary(got_wout)
    ref = _layout_summary(ref_wout)
    if any(got[key] != ref[key] for key in ("ns", "mpol", "ntor", "nfp", "lasym")):
        return False
    try:
        if not np.array_equal(np.asarray(got_wout.xm, dtype=int), np.asarray(ref_wout.xm, dtype=int)):
            return False
        if not np.array_equal(np.asarray(got_wout.xn, dtype=int), np.asarray(ref_wout.xn, dtype=int)):
            return False
        got_xm_nyq = np.asarray(getattr(got_wout, "xm_nyq", []), dtype=int)
        ref_xm_nyq = np.asarray(getattr(ref_wout, "xm_nyq", []), dtype=int)
        got_xn_nyq = np.asarray(getattr(got_wout, "xn_nyq", []), dtype=int)
        ref_xn_nyq = np.asarray(getattr(ref_wout, "xn_nyq", []), dtype=int)
        if got_xm_nyq.size or ref_xm_nyq.size:
            if not np.array_equal(got_xm_nyq, ref_xm_nyq):
                return False
        if got_xn_nyq.size or ref_xn_nyq.size:
            if not np.array_equal(got_xn_nyq, ref_xn_nyq):
                return False
    except Exception:
        return False
    return True


def _low_order_mode_mask(wout: Any, *, max_m: int = 2, max_abs_n: int = 2) -> np.ndarray:
    xm = np.asarray(wout.xm, dtype=int)
    xn = np.asarray(wout.xn, dtype=int)
    nfp = max(1, int(wout.nfp))
    n = np.rint(xn / float(nfp)).astype(int)
    return (np.abs(xm) <= max_m) & (np.abs(n) <= max_abs_n)


def _low_order_mode_mask_for_array(
    wout: Any,
    array: Any,
    *,
    max_m: int = 2,
    max_abs_n: int = 2,
) -> np.ndarray:
    """Return a low-order mask matching an array's mode basis.

    VMEC WOUT files store geometry/lambda arrays on the main ``xm/xn`` basis,
    while magnetic-field arrays use the Nyquist ``xm_nyq/xn_nyq`` basis.  The
    generated-mgrid VMEC2000 diagnostic compares both classes, so the mask must
    be chosen from the array's own last dimension instead of reusing the main
    mode mask for every field.
    """

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


def _jax_backend_comparison(got_wout: Any, ref_wout: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    scalar_names = ("aspect", "wb", "wp")
    array_names = ("rmnc", "zmns", "lmns", "iotas", "iotaf")
    report: dict[str, Any] = {
        "candidate_backend": "vmec_jax_direct_coils",
        "reference_backend": "vmec_jax_generated_mgrid",
        "layout": {
            "candidate": _layout_summary(got_wout),
            "reference": _layout_summary(ref_wout),
            "same": _same_layout(got_wout, ref_wout),
        },
        "scalars": {name: _scalar_gap(got_wout, ref_wout, name, rtol=rtol, atol=atol) for name in scalar_names},
        "arrays": {
            name: _array_gap(getattr(got_wout, name), getattr(ref_wout, name), rtol=rtol, atol=atol)
            for name in array_names
            if hasattr(got_wout, name) and hasattr(ref_wout, name)
        },
    }
    checks = [bool(report["layout"]["same"])]
    checks.extend(bool(values.get("within_tolerance")) for values in report["scalars"].values())
    checks.extend(bool(values.get("within_tolerance")) for values in report["arrays"].values())
    report["passed"] = bool(checks and all(checks))
    return report


def _same_sign_and_scale(got: float | None, ref: float | None, *, max_ratio: float) -> dict[str, Any]:
    if got is None or ref is None:
        return {"available": False, "got": got, "ref": ref, "passed": None}
    tiny = 1.0e-14
    if abs(got) <= tiny and abs(ref) <= tiny:
        return {"available": True, "got": got, "ref": ref, "ratio": 1.0, "passed": True}
    sign_ok = bool(got * ref >= 0.0)
    ratio = max(abs(got), tiny) / max(abs(ref), tiny)
    passed = bool(sign_ok and (1.0 / max_ratio) <= ratio <= max_ratio)
    return {
        "available": True,
        "got": got,
        "ref": ref,
        "same_sign": sign_ok,
        "ratio": ratio,
        "max_ratio": float(max_ratio),
        "passed": passed,
    }


def _relative_limit(gap: dict[str, Any], *, limit: float) -> dict[str, Any]:
    value = gap.get("rel_delta", gap.get("rel_rms_delta"))
    if value is None:
        return {"available": False, "limit": float(limit), "passed": None}
    return {"available": True, "value": float(value), "limit": float(limit), "passed": bool(float(value) <= float(limit))}


def _vmec2000_wout_comparison(candidate_wout: Any, vmec2000_wout: Any, *, candidate_backend: str) -> dict[str, Any]:
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
    profile_names = (
        "iotas",
        "iotaf",
        "pres",
        "presf",
        "vp",
        "phipf",
        "phips",
        "chipf",
        "buco",
        "bvco",
        "jcuru",
        "jcurv",
        "equif",
    )
    report: dict[str, Any] = {
        "candidate_backend": candidate_backend,
        "reference_backend": "vmec2000_generated_mgrid",
        "layout": {
            "candidate": _layout_summary(candidate_wout),
            "reference": _layout_summary(vmec2000_wout),
            "same": _same_layout(candidate_wout, vmec2000_wout),
        },
        "scalars": {name: _scalar_gap(candidate_wout, vmec2000_wout, name) for name in scalar_names},
        "profiles": {},
        "low_order_modes": {},
        "limits": {},
    }
    for name in profile_names:
        if hasattr(candidate_wout, name) and hasattr(vmec2000_wout, name):
            report["profiles"][name] = _array_gap(getattr(candidate_wout, name), getattr(vmec2000_wout, name), radial_skip=1)

    try:
        for name in ("rmnc", "zmns", "lmns", "bmnc", "gmnc", "bsubumnc", "bsubvmnc", "bsupumnc", "bsupvmnc"):
            if hasattr(candidate_wout, name) and hasattr(vmec2000_wout, name):
                low_order = _low_order_mode_mask_for_array(vmec2000_wout, getattr(vmec2000_wout, name))
                report["low_order_modes"][name] = _array_gap(
                    getattr(candidate_wout, name),
                    getattr(vmec2000_wout, name),
                    radial_skip=1,
                    mode_mask=low_order,
                )
        if "rmnc" in report["low_order_modes"]:
            report["low_order_mode_count"] = int(report["low_order_modes"]["rmnc"].get("shape", [0, 0])[-1])
    except Exception as exc:
        report["low_order_modes_error"] = repr(exc)

    report["limits"]["layout_same"] = {"passed": bool(report["layout"]["same"])}
    report["limits"]["aspect_rel_delta"] = _relative_limit(report["scalars"].get("aspect", {}), limit=1.5e-1)
    report["limits"]["wb_rel_delta"] = _relative_limit(report["scalars"].get("wb", {}), limit=2.5e-1)
    report["limits"]["iotas_rel_rms_no_axis"] = _relative_limit(report["profiles"].get("iotas", {}), limit=3.5e-1)
    report["limits"]["low_order_rmnc_rel_rms"] = _relative_limit(report["low_order_modes"].get("rmnc", {}), limit=4.0e-1)
    report["limits"]["low_order_zmns_rel_rms"] = _relative_limit(report["low_order_modes"].get("zmns", {}), limit=4.0e-1)

    for name in ("wp", "betatotal", "betapol", "betator", "betaxis"):
        gap = report["scalars"].get(name, {})
        report["limits"][f"{name}_same_sign_scale"] = _same_sign_and_scale(
            gap.get("got"),
            gap.get("ref"),
            max_ratio=10.0,
        )
    report["limits"]["fsq_total_same_sign_scale"] = _same_sign_and_scale(
        _fsq_total_from_wout(candidate_wout),
        _fsq_total_from_wout(vmec2000_wout),
        max_ratio=10.0,
    )

    limit_results = [
        values.get("passed")
        for values in report["limits"].values()
        if isinstance(values, dict) and values.get("passed") is not None
    ]
    report["passed_current_limits"] = bool(limit_results and all(bool(value) for value in limit_results))
    return report


def _make_freeb_indata(base_indata: Any, *, mgrid_file: str, args: argparse.Namespace) -> Any:
    indata = deepcopy(base_indata)
    nzeta = _diagnostic_nzeta(args)
    ns_array, niter_array, ftol_array = _diagnostic_schedule(args)
    phiedge = float(indata.scalars.get("PHIEDGE", 0.0)) * float(args.phiedge_scale)
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": str(mgrid_file),
            "EXTCUR": [float(args.extcur_scale)],
            "NS_ARRAY": [int(value) for value in ns_array],
            "NITER_ARRAY": [int(value) for value in niter_array],
            "FTOL_ARRAY": [float(value) for value in ftol_array],
            "NITER": int(niter_array[-1]),
            "FTOL": float(ftol_array[-1]),
            "PHIEDGE": phiedge,
            "MPOL": int(args.mpol),
            "NTOR": int(args.ntor),
            "NZETA": int(nzeta),
            "NTHETA": 0,
            "NVACSKIP": int(args.nvacskip) if args.nvacskip is not None else int(nzeta),
            "PRES_SCALE": float(args.pressure_scale),
            "PMASS_TYPE": "power_series",
            "AM": [1.0, -1.0],
        }
    )
    return indata


def _load_essos(args: argparse.Namespace) -> tuple[Any, Path, Path | None, str]:
    requested_root = args.essos_root.expanduser().resolve() if args.essos_root is not None else None
    preferred_root = requested_root
    if preferred_root is None:
        for root in _candidate_essos_roots():
            root = root.expanduser()
            if (root / "essos").exists():
                preferred_root = root.resolve()
                break
    if preferred_root is not None:
        sys.path.insert(0, str(preferred_root))

    try:
        from essos.coils import Coils_from_json
        import essos
    except Exception as exc:
        raise ImportError(
            "Could not import ESSOS. Use --essos-root /path/to/ESSOS_mgrid_pr or set ESSOS_ROOT."
        ) from exc

    if args.coils_json is not None:
        coils_json = args.coils_json.expanduser().resolve()
        if not coils_json.exists():
            raise ValueError(f"explicit --coils-json does not exist: {coils_json}")
    else:
        coils_json = _find_default_coils_json(preferred_root)

    coils = Coils_from_json(str(coils_json))
    if not hasattr(coils, "to_mgrid"):
        module_path = getattr(essos, "__file__", "unknown")
        raise RuntimeError(
            "Imported ESSOS does not provide Coils.to_mgrid. Use ESSOS PR #33 or newer, for example "
            f"--essos-root {REPO_ROOT.parent / 'ESSOS_mgrid_pr'}. Imported essos from {module_path}."
        )
    return coils, coils_json, preferred_root, str(getattr(essos, "__file__", "unknown"))


def _write_generated_mgrid(coils: Any, path: Path, *, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    coils.to_mgrid(
        path,
        nr=int(args.mgrid_nr),
        nz=int(args.mgrid_nz),
        nphi=int(args.mgrid_nphi),
        rmin=float(args.mgrid_rmin),
        rmax=float(args.mgrid_rmax),
        zmin=float(args.mgrid_zmin),
        zmax=float(args.mgrid_zmax),
        nfp=int(coils.nfp),
    )


def _run_vmec_jax_case(
    *,
    input_path: Path,
    wout_path: Path,
    direct_params: Any | None,
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, Any]]:
    from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
    from vmec_jax.wout import read_wout

    kwargs: dict[str, Any] = {}
    if direct_params is not None:
        kwargs.update(
            {
                "external_field_provider_kind": "direct_coils",
                "external_field_provider_params": direct_params,
            }
        )
    t0 = time.perf_counter()
    run = run_free_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="parity",
        multigrid_use_input_niter=True,
        verbose=False,
        jit_forces=bool(args.jit_forces),
        free_boundary_activate_fsq=None if args.activate_fsq is None else float(args.activate_fsq),
        **kwargs,
    )
    runtime_s = float(time.perf_counter() - t0)
    write_wout_from_fixed_boundary_run(wout_path, run)
    wout = read_wout(wout_path)
    return run, wout, {
        "status": "completed",
        "runtime_s": runtime_s,
        "input_path": input_path,
        "wout_path": wout_path,
        "fsq": _fsq_summary_from_run(run),
        "free_boundary": _free_boundary_summary_from_run(run),
        "wout": _wout_summary(wout),
    }


def _vmec2000_wout_path(vmec2000_result: Any) -> Path:
    case = vmec2000_result.input_path.name.removeprefix("input.")
    return vmec2000_result.workdir / f"wout_{case}.nc"


def _vmec2000_summary(vmec2000_result: Any) -> dict[str, Any]:
    from vmec_jax.vmec2000_exec import flatten_threed1, threed1_fsq_total

    rows = flatten_threed1(vmec2000_result.stages)
    fsq_total = threed1_fsq_total(rows)
    last_row = rows[-1] if rows else None
    return {
        "status": "completed",
        "workdir": vmec2000_result.workdir,
        "input_path": vmec2000_result.input_path,
        "returncode": int(getattr(vmec2000_result, "returncode", 0)),
        "opened_mgrid": "Opening vacuum field file:" in vmec2000_result.stdout,
        "runtime_s": float(vmec2000_result.runtime_s),
        "threed1_path": vmec2000_result.threed1_path,
        "threed1_tail": _tail_text(vmec2000_result.threed1_path, lines=80),
        "stdout_tail": vmec2000_result.stdout.splitlines()[-40:],
        "stderr_tail": vmec2000_result.stderr.splitlines()[-40:],
        "files": sorted(p.name for p in vmec2000_result.workdir.iterdir()),
        "stage_count": len(vmec2000_result.stages),
        "iteration_row_count": len(rows),
        "fsq_total_last": _safe_float(fsq_total[-1]) if fsq_total.size else None,
        "last_row": None
        if last_row is None
        else {
            "it": int(last_row.it),
            "fsqr": float(last_row.fsqr),
            "fsqz": float(last_row.fsqz),
            "fsql": float(last_row.fsql),
            "fsqr1": float(last_row.fsqr1),
            "fsqz1": float(last_row.fsqz1),
            "fsql1": float(last_row.fsql1),
            "delt0r": last_row.delt0r,
            "r00": last_row.r00,
            "w": last_row.w,
            "beta": last_row.beta,
            "avg_m": last_row.avg_m,
            "delbsq": last_row.delbsq,
            "fedge": last_row.fedge,
        },
        "stages": [
            {
                "ns": int(stage.ns),
                "niter": int(stage.niter),
                "ftolv": float(stage.ftolv),
                "row_count": len(stage.rows),
            }
            for stage in vmec2000_result.stages
        ],
    }


def _vmec2000_underconverged_details(summary: dict[str, Any]) -> dict[str, Any]:
    stages = summary.get("stages") or []
    last_stage = stages[-1] if stages else {}
    last_row = summary.get("last_row") or {}
    niter = _safe_float(last_stage.get("niter"))
    last_it = _safe_float(last_row.get("it"))
    ftolv = _safe_float(last_stage.get("ftolv"))
    preconditioned_parts = [
        _safe_float(last_row.get(name))
        for name in ("fsqr1", "fsqz1", "fsql1")
    ]
    preconditioned_fsq_total = None
    if all(value is not None for value in preconditioned_parts):
        preconditioned_fsq_total = float(sum(value for value in preconditioned_parts if value is not None))
    physical_fsqr = _safe_float(last_row.get("fsqr"))
    physical_fsqz = _safe_float(last_row.get("fsqz"))
    physical_force_gate = None
    if physical_fsqr is not None and physical_fsqz is not None:
        physical_force_gate = float(physical_fsqr + physical_fsqz)
    vacuum_gate_threshold = 1.0e-3
    delbsq_last = _safe_float(last_row.get("delbsq"))
    fedge_last = _safe_float(last_row.get("fedge"))
    opened_mgrid = bool(summary.get("opened_mgrid", False))
    default_edge_balance = (
        delbsq_last is not None
        and fedge_last is not None
        and abs(delbsq_last - 1.0) <= 1.0e-12
        and abs(fedge_last) <= 1.0e-14
    )
    # In VMEC2000's free-boundary control flow the active vacuum solve is not
    # entered until the physical R/Z force residual is below this gate.  When
    # DEL-BSQ remains at its default value and FEDGE is zero, a generated mgrid
    # run has read the grid but has not yet exercised the active-vacuum path.
    vacuum_activation_blocked = bool(
        opened_mgrid
        and default_edge_balance
        and physical_force_gate is not None
        and physical_force_gate > vacuum_gate_threshold
    )

    stderr_tail = list(summary.get("stderr_tail") or [])
    tails = list(summary.get("stdout_tail") or []) + stderr_tail + list(summary.get("threed1_tail") or [])
    printed_try_increasing_niter = any("Try increasing NITER" in line for line in tails)
    runtime_error_markers = (
        "Fortran runtime error",
        "Error termination",
        "Segmentation fault",
        "SIGSEGV",
        "SIGBUS",
    )
    backtrace_markers = ("Could not print backtrace",)
    runtime_error_lines = [
        line
        for line in stderr_tail
        if any(marker in line for marker in runtime_error_markers)
    ]
    backtrace_lines = [
        line
        for line in stderr_tail
        if any(marker in line for marker in backtrace_markers)
    ]
    returncode = int(summary.get("returncode") or 0)
    nonzero_returncode = returncode != 0
    more_iter_returncode = returncode == VMEC2000_MORE_ITER_RETURNCODE
    reached_niter = bool(niter is not None and last_it is not None and int(last_it) >= int(niter))
    classification = "unknown_no_wout"
    if runtime_error_lines:
        classification = "vmec2000_runtime_error"
    elif vacuum_activation_blocked:
        classification = "vmec2000_vacuum_inactive_force_gate"
    elif more_iter_returncode and (printed_try_increasing_niter or last_it is not None):
        classification = "vmec2000_more_iter_exit"
    elif nonzero_returncode:
        classification = "vmec2000_nonzero_exit"
    elif reached_niter and printed_try_increasing_niter:
        classification = "reached_niter_without_wout"
    elif printed_try_increasing_niter:
        classification = "vmec2000_requested_more_iterations"

    details: dict[str, Any] = {
        "classification": classification,
        "returncode": returncode,
        "nonzero_returncode": nonzero_returncode,
        "more_iter_returncode": more_iter_returncode,
        "runtime_error_detected": bool(runtime_error_lines),
        "runtime_error_tail": runtime_error_lines[-3:],
        "backtrace_detected": bool(backtrace_lines),
        "backtrace_tail": backtrace_lines[-3:],
        "printed_try_increasing_niter": printed_try_increasing_niter,
        "reached_niter": reached_niter,
        "last_it": None if last_it is None else int(last_it),
        "niter": None if niter is None else int(niter),
        "ftolv": ftolv,
        "opened_mgrid": opened_mgrid,
        "physical_fsq_total_last": _safe_float(summary.get("fsq_total_last")),
        "physical_force_gate_last": physical_force_gate,
        "physical_force_gate_threshold": vacuum_gate_threshold,
        "vmec2000_vacuum_activation_blocked": vacuum_activation_blocked,
        "vmec2000_vacuum_active_evidence": bool(opened_mgrid and not default_edge_balance),
        "preconditioned_fsq_total_last": preconditioned_fsq_total,
        "delt0r_last": _safe_float(last_row.get("delt0r")),
        "w_last": _safe_float(last_row.get("w")),
        "beta_last": _safe_float(last_row.get("beta")),
        "avg_m_last": _safe_float(last_row.get("avg_m")),
        "delbsq_last": delbsq_last,
        "fedge_last": fedge_last,
    }
    if preconditioned_fsq_total is not None and ftolv not in (None, 0.0):
        details["preconditioned_fsq_total_over_ftolv"] = float(preconditioned_fsq_total / ftolv)
    delbsq_last = details["delbsq_last"]
    if delbsq_last is not None and ftolv not in (None, 0.0):
        details["delbsq_over_ftolv"] = float(abs(delbsq_last) / ftolv)
    return details


def _vmec2000_nonzero_status(summary: dict[str, Any]) -> tuple[str, str, str]:
    """Classify VMEC2000 nonzero exits without conflating VMEC flags with crashes."""

    details = _vmec2000_underconverged_details(summary)
    if details["classification"] == "vmec2000_vacuum_inactive_force_gate":
        status = "more_iter_exit" if details.get("more_iter_returncode") else "no_wout"
        return (
            status,
            "vmec2000_vacuum_inactive_force_gate",
            "VMEC2000 opened the generated mgrid but did not enter the active "
            "vacuum solve before stopping: DEL-BSQ remained at its default "
            "value, FEDGE stayed zero, and the physical FSQR+FSQZ force was "
            "above VMEC2000's vacuum-activation gate. This is not WOUT-level "
            "generated-mgrid parity evidence; rerun with a better seed/schedule "
            "or promote a fixture that reaches the active-vacuum path.",
        )
    if details["classification"] == "vmec2000_more_iter_exit":
        return (
            "more_iter_exit",
            "vmec2000_more_iterations_required",
            "VMEC2000 exited with more_iter_flag=2 before producing a WOUT. "
            "Inspect underconverged, stdout_tail, threed1_tail, and rerun with a "
            "looser FTOL or larger NITER/MAX_MAIN_ITERATIONS for promotion evidence.",
        )
    if details["classification"] == "vmec2000_runtime_error":
        return (
            "nonzero_exit",
            "vmec2000_runtime_error",
            "VMEC2000 emitted a runtime error before producing a WOUT. Inspect "
            "underconverged.runtime_error_tail, stderr_tail, threed1_tail, and "
            "the generated input/mgrid in the VMEC2000 workdir.",
        )
    return (
        "nonzero_exit",
        "vmec2000_returned_nonzero",
        "VMEC2000 exited nonzero before producing a WOUT. Inspect returncode, "
        "stdout_tail, stderr_tail, threed1_tail, and the workdir files.",
    )


def _classify_vmec2000_result_summary(summary: dict[str, Any], *, wout_path: Path) -> None:
    """Mutate a VMEC2000 summary with status/reason/help based on output state."""

    summary["wout_path"] = wout_path
    if int(summary.get("returncode") or 0) != 0:
        status, reason, help_text = _vmec2000_nonzero_status(summary)
        summary["status"] = status
        summary["reason"] = reason
        summary["underconverged"] = _vmec2000_underconverged_details(summary)
        summary["help"] = help_text
        return
    if not wout_path.exists():
        summary["status"] = "no_wout"
        summary["reason"] = "vmec2000_completed_without_wout"
        summary["underconverged"] = _vmec2000_underconverged_details(summary)
        summary["help"] = (
            "Inspect underconverged, stdout_tail, stderr_tail, threed1_tail, "
            "and the VMEC2000 workdir files in this JSON."
        )
        return
    summary["status"] = "completed"


def _wout_file_diagnostics(path: Path) -> dict[str, Any]:
    """Return compact metadata for a WOUT file that could not be parsed."""

    out: dict[str, Any] = {
        "path": path,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    if not path.exists():
        return out
    try:
        from netCDF4 import Dataset

        with Dataset(path) as ds:
            out["dimensions"] = {str(name): len(value) for name, value in ds.dimensions.items()}
            variables = sorted(str(name) for name in ds.variables.keys())
            out["variables"] = variables
            out["has_mode_table"] = "xm" in variables and "xn" in variables
            if "ier_flag" in ds.variables:
                out["ier_flag"] = int(ds.variables["ier_flag"][()])
    except Exception as exc:
        out["netcdf_error"] = repr(exc)
    return out


def _mark_unreadable_vmec2000_wout(summary: dict[str, Any], *, wout_path: Path, error: Exception) -> None:
    wout_file = _wout_file_diagnostics(wout_path)
    stdout_tail = "\n".join(str(line) for line in summary.get("stdout_tail", []))
    phiedge_wrong_sign = "PHIEDGE HAS WRONG SIGN IN VACUUM SUBROUTINE" in stdout_tail
    summary["status"] = "wout_unreadable"
    summary["reason"] = "vmec2000_phiedge_wrong_sign" if phiedge_wrong_sign else "vmec2000_wout_read_failed"
    summary["wout_read_error"] = repr(error)
    summary["wout_file"] = wout_file
    if phiedge_wrong_sign:
        summary["help"] = (
            "VMEC2000 aborted in the vacuum routine because PHIEDGE has the "
            "wrong sign for this generated-mgrid/current convention. The small "
            "WOUT is an error WOUT without mode tables, not parity evidence. "
            "Try a sign-consistent PHIEDGE/EXTCUR convention before promoting "
            "this optional VMEC2000 row."
        )
    else:
        summary["help"] = (
            "VMEC2000 wrote a WOUT file, but vmec_jax could not parse it. "
            "Inspect wout_file.ier_flag, variables, stdout_tail, threed1_tail, "
            "and the generated input/mgrid in the VMEC2000 workdir."
        )


def _read_vmec2000_wout_for_summary(summary: dict[str, Any], *, wout_path: Path) -> Any | None:
    from vmec_jax.wout import read_wout

    try:
        wout = read_wout(wout_path)
    except Exception as exc:
        _mark_unreadable_vmec2000_wout(summary, wout_path=wout_path, error=exc)
        return None
    summary["wout"] = _wout_summary(wout)
    summary["wout_promotion_quality"] = _vmec2000_wout_promotion_quality(wout)
    return wout


def _vmec2000_probe_updates(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return bounded VMEC2000-only probe patches for WOUT promotion diagnostics."""

    ns_array, niter_array, _ = _diagnostic_schedule(args)
    ftols = _parse_float_array(args.vmec2000_probe_ftols, name="vmec2000-probe-ftols")
    max_main_values = _parse_int_array(
        args.vmec2000_probe_max_main_iterations,
        name="vmec2000-probe-max-main-iterations",
    )

    probes: list[dict[str, Any]] = []
    for ftol in ftols:
        probes.append(
            {
                "label": f"loose_ftol_{ftol:g}",
                "updates": {
                    "FTOL": _format_namelist_float(ftol),
                    "FTOL_ARRAY": _format_namelist_array([float(ftol)] * len(ns_array)),
                },
            }
        )
    probes.append(
        {
            "label": "force_full3d_output",
            "updates": {
                "LFULL3D1OUT": "T",
                "NITER": str(int(niter_array[-1])),
                "NITER_ARRAY": _format_namelist_array(niter_array),
            },
        }
    )
    for max_main in max_main_values:
        probes.append(
            {
                "label": f"max_main_iterations_{max_main}",
                "updates": {
                    "MAX_MAIN_ITERATIONS": str(int(max_main)),
                    "NITER": str(int(niter_array[-1])),
                    "NITER_ARRAY": _format_namelist_array(niter_array),
                },
            }
        )
    return probes


def _run_vmec2000_promotion_probes(
    *,
    mgrid_input: Path,
    mgrid_path: Path,
    workdir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Run optional bounded VMEC2000 probes after a no-WOUT/more-iter result."""

    from vmec_jax.vmec2000_exec import find_vmec2000_exec, run_xvmec2000

    exec_path = args.vmec2000_exec.expanduser().resolve() if args.vmec2000_exec is not None else find_vmec2000_exec()
    if exec_path is None or not exec_path.exists():
        return [
            {
                "label": "promotion_probes",
                "status": "skipped",
                "reason": "vmec2000_exec_not_found",
            }
        ]

    records: list[dict[str, Any]] = []
    for probe in _vmec2000_probe_updates(args):
        label = str(probe["label"])
        probe_workdir = workdir / label
        probe_workdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mgrid_path, probe_workdir / mgrid_path.name)
        try:
            result = run_xvmec2000(
                mgrid_input,
                exec_path=exec_path,
                workdir=probe_workdir,
                timeout_s=float(args.vmec2000_timeout),
                indata_updates={str(key): str(value) for key, value in probe["updates"].items()},
                keep_workdir=True,
            )
        except subprocess.TimeoutExpired as exc:
            records.append(
                {
                    "label": label,
                    "status": "timeout",
                    "reason": "vmec2000_timeout",
                    "updates": probe["updates"],
                    "timeout_s": float(args.vmec2000_timeout),
                    "error": repr(exc),
                }
            )
            continue
        summary = _vmec2000_summary(result)
        summary["label"] = label
        summary["updates"] = probe["updates"]
        summary["exec_path"] = exec_path
        wout_path = _vmec2000_wout_path(result)
        _classify_vmec2000_result_summary(summary, wout_path=wout_path)
        if summary["status"] == "completed":
            _read_vmec2000_wout_for_summary(summary, wout_path=wout_path)
        records.append(summary)
    return records


def _run_vmec2000_case(
    *,
    mgrid_input: Path,
    mgrid_path: Path,
    workdir: Path,
    args: argparse.Namespace,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    from vmec_jax.vmec2000_exec import find_vmec2000_exec, run_xvmec2000
    if args.skip_vmec2000:
        return None, None, {
            "status": "skipped",
            "reason": "skip_vmec2000_requested",
            "help": "Remove --skip-vmec2000 to run xvmec2000 when available.",
        }

    exec_path = args.vmec2000_exec.expanduser().resolve() if args.vmec2000_exec is not None else find_vmec2000_exec()
    if args.vmec2000_exec is not None and (exec_path is None or not exec_path.exists()):
        return None, None, {
            "status": "error",
            "reason": "explicit_vmec2000_exec_not_found",
            "exec_path": args.vmec2000_exec,
            "help": "Fix --vmec2000-exec or omit it to allow auto-discovery/skip behavior.",
        }
    if exec_path is None or not exec_path.exists():
        return None, None, {
            "status": "skipped",
            "reason": "vmec2000_exec_not_found",
            "help": "Set --vmec2000-exec or VMEC2000_EXEC to a valid xvmec2000 executable.",
        }

    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mgrid_path, workdir / mgrid_path.name)
    try:
        indata_updates = None
        if args.vmec2000_niter is not None:
            vmec2000_niter = int(args.vmec2000_niter)
            ns_array, _, ftol_array = _diagnostic_schedule(args)
            niter_override = [vmec2000_niter] * len(ns_array)
            indata_updates = {
                "NITER": str(vmec2000_niter),
                "NITER_ARRAY": _format_namelist_array(niter_override),
                "FTOL": f"{float(ftol_array[-1]):.16e}",
                "FTOL_ARRAY": _format_namelist_array(ftol_array),
            }
        result = run_xvmec2000(
            mgrid_input,
            exec_path=exec_path,
            workdir=workdir,
            timeout_s=float(args.vmec2000_timeout),
            indata_updates=indata_updates,
            keep_workdir=True,
        )
    except subprocess.TimeoutExpired as exc:
        return None, None, {
            "status": "timeout",
            "reason": "vmec2000_timeout",
            "exec_path": exec_path,
            "timeout_s": float(args.vmec2000_timeout),
            "error": repr(exc),
            "help": "Increase --vmec2000-timeout or reduce the diagnostic grid/iteration settings.",
        }

    summary = _vmec2000_summary(result)
    summary["exec_path"] = exec_path
    if args.vmec2000_niter is not None:
        summary["mixed_schedule_non_promotable"] = True
        summary["mixed_schedule_reason"] = "--vmec2000-niter overrides only the VMEC2000 NITER_ARRAY"
        summary["vmec2000_niter_override"] = int(args.vmec2000_niter)
    wout_path = _vmec2000_wout_path(result)
    _classify_vmec2000_result_summary(summary, wout_path=wout_path)
    if summary["status"] in {"more_iter_exit", "nonzero_exit", "no_wout"}:
        return result, None, summary

    wout = _read_vmec2000_wout_for_summary(summary, wout_path=wout_path)
    if wout is None:
        return result, None, summary
    return result, wout, summary


def _base_payload(args: argparse.Namespace, *, out: Path, workdir: Path) -> dict[str, Any]:
    ns_array, niter_array, ftol_array = _diagnostic_schedule(args)
    return {
        "status": "running",
        "script": Path(__file__).resolve(),
        "repo_root": REPO_ROOT,
        "started_at_utc": _now_utc(),
        "out": out,
        "workdir": workdir,
        "base_input": args.input.expanduser().resolve(),
        "configuration": {
            "pressure_scale": float(args.pressure_scale),
            "phiedge_scale": float(args.phiedge_scale),
            "extcur_scale": float(args.extcur_scale),
            "niter": int(niter_array[-1]),
            "ftol": float(ftol_array[-1]),
            "ns": int(ns_array[-1]),
            "ns_array": [int(value) for value in ns_array],
            "niter_array": [int(value) for value in niter_array],
            "ftol_array": [float(value) for value in ftol_array],
            "uses_multigrid_schedule": len(ns_array) > 1,
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "nzeta": int(_diagnostic_nzeta(args)),
            "nvacskip": int(args.nvacskip) if args.nvacskip is not None else int(_diagnostic_nzeta(args)),
            "jit_forces": bool(args.jit_forces),
            "activate_fsq": None if args.activate_fsq is None else float(args.activate_fsq),
            "active_free_boundary_requested": args.activate_fsq is not None,
            "vmec2000_niter": None if args.vmec2000_niter is None else int(args.vmec2000_niter),
            "mixed_vmec2000_schedule_non_promotable": args.vmec2000_niter is not None,
            "jax_rtol": float(args.jax_rtol),
            "jax_atol": float(args.jax_atol),
        },
        "mgrid_generation": {
            "nr": int(args.mgrid_nr),
            "nz": int(args.mgrid_nz),
            "nphi": int(args.mgrid_nphi),
            "auto_bounds_requested": bool(args.mgrid_auto_bounds),
            "rmin": float(args.mgrid_rmin),
            "rmax": float(args.mgrid_rmax),
            "zmin": float(args.mgrid_zmin),
            "zmax": float(args.mgrid_zmax),
        },
        "dependencies": {},
        "inputs": {},
        "backends": {},
        "comparisons": {},
        "errors": [],
        "warnings": [],
    }


def _finish_payload(payload: dict[str, Any], *, hard_errors: list[str], warnings: list[str]) -> None:
    payload["finished_at_utc"] = _now_utc()
    payload["hard_errors"] = hard_errors
    payload["warnings"].extend(warnings)
    if hard_errors:
        payload["status"] = "failed"
    elif warnings:
        payload["status"] = "completed_with_warnings"
    else:
        payload["status"] = "completed"


def _dependency_skip(
    *,
    payload: dict[str, Any],
    out: Path,
    reason: str,
    error: str,
    require: bool,
) -> int:
    payload["status"] = "skipped"
    payload["reason"] = reason
    payload["error"] = error
    payload["help"] = (
        "Use --essos-root /path/to/ESSOS_mgrid_pr or set ESSOS_ROOT/ESSOS_INPUT_DIR. "
        "The ESSOS checkout must provide Coils.to_mgrid."
    )
    payload["finished_at_utc"] = _now_utc()
    _write_json(out, payload)
    print(f"[compare-freeb-coils] skipped: {reason}: {error}", file=sys.stderr)
    print(f"[compare-freeb-coils] wrote {out}", file=sys.stderr)
    return 1 if require else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    nzeta = _diagnostic_nzeta(args)
    if args.strict:
        args.require_essos = True
        args.require_vmec2000 = True
        args.fail_on_jax_mismatch = True
        args.fail_on_vmec2000_mismatch = True
    if args.skip_vmec2000 and args.require_vmec2000:
        raise SystemExit("--skip-vmec2000 conflicts with --require-vmec2000/--strict")
    if int(args.niter) < 1:
        raise SystemExit("--niter must be >= 1")
    if args.vmec2000_niter is not None and int(args.vmec2000_niter) < 1:
        raise SystemExit("--vmec2000-niter must be >= 1")
    _diagnostic_schedule(args)
    if int(args.ns) < 3:
        raise SystemExit("--ns must be >= 3")
    if int(args.mgrid_nphi) % int(nzeta) != 0 and int(nzeta) % int(args.mgrid_nphi) != 0:
        raise SystemExit("--nzeta must be compatible with generated mgrid kp=--mgrid-nphi")

    out = args.out.expanduser().resolve()
    workdir = (args.workdir or (out.parent / f"{out.stem}_work")).expanduser().resolve()
    payload = _base_payload(args, out=out, workdir=workdir)
    hard_errors: list[str] = []
    warnings: list[str] = []

    try:
        coils, coils_json, essos_root, essos_module = _load_essos(args)
    except Exception as exc:
        explicit_path_error = args.coils_json is not None or args.essos_root is not None
        if explicit_path_error:
            payload["status"] = "failed"
            payload["reason"] = "explicit_essos_or_coils_path_invalid"
            payload["error"] = repr(exc)
            payload["finished_at_utc"] = _now_utc()
            _write_json(out, payload)
            print(f"[compare-freeb-coils] failed explicit ESSOS/coils path: {exc!r}", file=sys.stderr)
            return 1
        return _dependency_skip(
            payload=payload,
            out=out,
            reason="essos_or_mgrid_unavailable",
            error=repr(exc),
            require=bool(args.require_essos),
        )

    try:
        from vmec_jax._compat import enable_x64
        from vmec_jax.external_fields import from_essos_coils
        from vmec_jax.namelist import read_indata, write_indata

        enable_x64(True)
    except Exception as exc:
        payload["errors"].append({"stage": "vmec_jax_import", "error": repr(exc)})
        _finish_payload(payload, hard_errors=["vmec_jax_import_failed"], warnings=warnings)
        _write_json(out, payload)
        print(f"[compare-freeb-coils] failed importing vmec_jax helpers: {exc!r}", file=sys.stderr)
        return 1

    workdir.mkdir(parents=True, exist_ok=True)
    base_input = args.input.expanduser().resolve()
    if not base_input.exists():
        payload["errors"].append({"stage": "input", "error": f"missing input deck: {base_input}"})
        _finish_payload(payload, hard_errors=["input_missing"], warnings=warnings)
        _write_json(out, payload)
        return 1

    mgrid_path = workdir / "mgrid_lpqa_from_essos.nc"
    mgrid_input = workdir / "input.lpqa_mgrid"
    direct_input = workdir / "input.lpqa_direct"
    mgrid_wout_path = workdir / "wout_lpqa_mgrid_vmec_jax.nc"
    direct_wout_path = workdir / "wout_lpqa_direct_vmec_jax.nc"

    payload["dependencies"]["essos"] = {
        "status": "available",
        "root": essos_root,
        "module": essos_module,
        "coils_json": coils_json,
        "has_to_mgrid": True,
    }
    payload["coils"] = {
        "nfp": int(coils.nfp),
        "stellsym": bool(coils.stellsym),
        "n_segments": int(coils.n_segments),
        "currents_scale": _safe_float(getattr(coils, "currents_scale", None)),
        "n_base_currents": int(np.asarray(getattr(coils, "dofs_currents", [])).size),
    }

    print(f"[compare-freeb-coils] generating mgrid: {mgrid_path}")
    try:
        base_indata = read_indata(base_input)
        bounds = _resolve_mgrid_bounds(base_indata, args)
        args.mgrid_rmin = float(bounds["rmin"])
        args.mgrid_rmax = float(bounds["rmax"])
        args.mgrid_zmin = float(bounds["zmin"])
        args.mgrid_zmax = float(bounds["zmax"])
        payload["mgrid_generation"].update(bounds)
        _write_generated_mgrid(coils, mgrid_path, args=args)
        write_indata(mgrid_input, _make_freeb_indata(base_indata, mgrid_file=mgrid_path.name, args=args))
        write_indata(direct_input, _make_freeb_indata(base_indata, mgrid_file="DIRECT_COILS", args=args))
        direct_params = from_essos_coils(
            coils,
            regularization_epsilon=float(args.regularization_epsilon),
            chunk_size=int(args.direct_chunk_size) if int(args.direct_chunk_size) > 0 else None,
        )
    except Exception as exc:
        payload["errors"].append({"stage": "setup", "error": repr(exc)})
        _finish_payload(payload, hard_errors=["setup_failed"], warnings=warnings)
        _write_json(out, payload)
        print(f"[compare-freeb-coils] setup failed: {exc!r}", file=sys.stderr)
        return 1

    payload["inputs"] = {
        "mgrid_file": mgrid_path,
        "mgrid_file_size_bytes": mgrid_path.stat().st_size if mgrid_path.exists() else None,
        "vmec_jax_mgrid_input": mgrid_input,
        "vmec_jax_direct_input": direct_input,
    }
    payload["direct_provider"] = {
        "n_base_coils": int(np.asarray(direct_params.base_currents).size),
        "n_segments": int(direct_params.n_segments),
        "nfp": int(direct_params.nfp),
        "stellsym": bool(direct_params.stellsym),
        "base_current_scale": float(direct_params.current_scale),
        "chunk_size": None if direct_params.chunk_size is None else int(direct_params.chunk_size),
        "regularization_epsilon": float(direct_params.regularization_epsilon),
    }

    try:
        print("[compare-freeb-coils] running vmec_jax mgrid backend")
        _run_mgrid, wout_mgrid, payload["backends"]["vmec_jax_mgrid"] = _run_vmec_jax_case(
            input_path=mgrid_input,
            wout_path=mgrid_wout_path,
            direct_params=None,
            args=args,
        )
        print("[compare-freeb-coils] running vmec_jax direct-coil backend")
        _run_direct, wout_direct, payload["backends"]["vmec_jax_direct_coils"] = _run_vmec_jax_case(
            input_path=direct_input,
            wout_path=direct_wout_path,
            direct_params=direct_params,
            args=args,
        )
    except Exception as exc:
        payload["errors"].append({"stage": "vmec_jax_run", "error": repr(exc)})
        _finish_payload(payload, hard_errors=["vmec_jax_run_failed"], warnings=warnings)
        _write_json(out, payload)
        print(f"[compare-freeb-coils] vmec_jax run failed: {exc!r}", file=sys.stderr)
        return 1

    jax_comparison = _jax_backend_comparison(wout_direct, wout_mgrid, rtol=float(args.jax_rtol), atol=float(args.jax_atol))
    payload["comparisons"]["vmec_jax_direct_vs_generated_mgrid"] = jax_comparison
    jax_mgrid_freeb = payload["backends"]["vmec_jax_mgrid"].get("free_boundary", {})
    jax_direct_freeb = payload["backends"]["vmec_jax_direct_coils"].get("free_boundary", {})
    jax_mgrid_active = bool(jax_mgrid_freeb.get("available")) and not bool(jax_mgrid_freeb.get("vacuum_stub", True))
    jax_direct_active = bool(jax_direct_freeb.get("available")) and not bool(jax_direct_freeb.get("vacuum_stub", True))
    payload["comparisons"]["vmec_jax_direct_vs_generated_mgrid"]["active_free_boundary"] = {
        "mgrid_backend": bool(jax_mgrid_active),
        "direct_backend": bool(jax_direct_active),
        "both_active": bool(jax_mgrid_active and jax_direct_active),
        "activate_fsq": None if args.activate_fsq is None else float(args.activate_fsq),
    }
    domain_checks = {
        "vmec_jax_mgrid": _boundary_domain_check(
            payload["backends"]["vmec_jax_mgrid"].get("wout", {}).get("boundary_extents", {}),
            payload["mgrid_generation"],
        ),
        "vmec_jax_direct_coils": _boundary_domain_check(
            payload["backends"]["vmec_jax_direct_coils"].get("wout", {}).get("boundary_extents", {}),
            payload["mgrid_generation"],
        ),
    }
    payload["comparisons"]["vmec_jax_direct_vs_generated_mgrid"]["boundary_vs_mgrid_domain"] = domain_checks
    for backend, check in domain_checks.items():
        if check.get("contained") is False:
            warnings.append(f"{backend}_boundary_outside_generated_mgrid")
    if not (jax_mgrid_active and jax_direct_active):
        warnings.append("vmec_jax_free_boundary_inactive")
    if not bool(jax_comparison["passed"]):
        message = "vmec_jax_direct_vs_generated_mgrid_mismatch"
        warnings.append(message)
        if bool(args.fail_on_jax_mismatch):
            hard_errors.append(message)

    vmec2000_workdir = workdir / "vmec2000_mgrid"
    print("[compare-freeb-coils] checking VMEC2000 mgrid backend")
    vmec2000_result = None
    wout_vmec2000 = None
    try:
        vmec2000_result, wout_vmec2000, vmec2000_summary = _run_vmec2000_case(
            mgrid_input=mgrid_input,
            mgrid_path=mgrid_path,
            workdir=vmec2000_workdir,
            args=args,
        )
    except Exception as exc:
        vmec2000_summary = {
            "status": "error",
            "reason": "vmec2000_run_failed",
            "error": repr(exc),
            "help": "Inspect the generated input and mgrid in the diagnostic workdir.",
        }
    payload["backends"]["vmec2000_generated_mgrid"] = vmec2000_summary

    vmec_status = str(vmec2000_summary.get("status", "unknown"))
    if bool(args.vmec2000_promotion_probes) and vmec_status in {"no_wout", "more_iter_exit", "nonzero_exit", "wout_unreadable"}:
        print("[compare-freeb-coils] running VMEC2000 WOUT-promotion probes")
        payload["backends"]["vmec2000_generated_mgrid"]["promotion_probes"] = _run_vmec2000_promotion_probes(
            mgrid_input=mgrid_input,
            mgrid_path=mgrid_path,
            workdir=vmec2000_workdir / "promotion_probes",
            args=args,
        )
    if vmec_status in ("skipped", "timeout", "no_wout", "more_iter_exit", "nonzero_exit", "wout_unreadable", "error"):
        vmec_reason = str(vmec2000_summary.get("reason") or "")
        warning = vmec_reason if vmec_reason.startswith("vmec2000_") else f"vmec2000_{vmec_status}"
        warnings.append(warning)
        if bool(args.require_vmec2000) or (
            args.vmec2000_exec is not None and vmec_status == "error"
        ):
            hard_errors.append(warning)

    if wout_vmec2000 is not None:
        wout_quality = dict(vmec2000_summary.get("wout_promotion_quality") or {})
        if not bool(wout_quality.get("promotable", False)):
            warnings.append("vmec2000_wout_nonpromotable")
        comp_mgrid = _vmec2000_wout_comparison(wout_mgrid, wout_vmec2000, candidate_backend="vmec_jax_generated_mgrid")
        comp_direct = _vmec2000_wout_comparison(wout_direct, wout_vmec2000, candidate_backend="vmec_jax_direct_coils")
        payload["comparisons"]["vmec_jax_mgrid_vs_vmec2000_mgrid"] = comp_mgrid
        payload["comparisons"]["vmec_jax_direct_vs_vmec2000_mgrid"] = comp_direct
        if not bool(comp_mgrid["passed_current_limits"]):
            warnings.append("vmec_jax_mgrid_vs_vmec2000_mgrid_limits_exceeded")
            if bool(args.fail_on_vmec2000_mismatch):
                hard_errors.append("vmec_jax_mgrid_vs_vmec2000_mgrid_limits_exceeded")
        if not bool(comp_direct["passed_current_limits"]):
            warnings.append("vmec_jax_direct_vs_vmec2000_mgrid_limits_exceeded")
            if bool(args.fail_on_vmec2000_mismatch):
                hard_errors.append("vmec_jax_direct_vs_vmec2000_mgrid_limits_exceeded")

    payload["summary"] = {
        "jax_direct_vs_mgrid_passed": bool(jax_comparison["passed"]),
        "vmec2000_status": vmec_status,
        "vmec2000_wout_available": wout_vmec2000 is not None,
        "vmec2000_wout_promotable": bool(
            dict(vmec2000_summary.get("wout_promotion_quality") or {}).get("promotable", False)
        ),
        "hard_error_count": len(hard_errors),
        "warning_count": len(warnings),
    }
    if vmec2000_result is not None:
        payload["summary"]["vmec2000_workdir"] = vmec2000_result.workdir

    _finish_payload(payload, hard_errors=hard_errors, warnings=warnings)
    _write_json(out, payload)
    print(f"[compare-freeb-coils] wrote {out}")
    if hard_errors:
        print(f"[compare-freeb-coils] failed: {hard_errors}", file=sys.stderr)
        return 1
    if warnings:
        print(f"[compare-freeb-coils] completed with warnings: {warnings}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

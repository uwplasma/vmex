#!/usr/bin/env python
"""Direct-coil free-boundary sensitivity probe.

This diagnostics-only script runs a small finite-pressure free-boundary VMEC
probe using ESSOS Landreman-Paul QA coils through the direct JAX coil provider.
It sweeps external-current scale factors and can add one deterministic coil
geometry perturbation case.  Results are written as JSON for branch debugging;
this is not a stable public benchmark entrypoint.

Example:

    python tools/diagnostics/freeb_direct_provider_sensitivity.py \
      --out results/freeb_direct_provider_sensitivity.json \
      --max-iter 2 --current-scales 0.95 1.0 1.05
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
DEFAULT_PRESSURE_SCALE = 11.48744555546
DEFAULT_CURRENT_SCALES = (0.95, 1.0, 1.05)


def _candidate_essos_input_dirs() -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("ESSOS_INPUT_DIR"):
        candidates.append(Path(os.environ["ESSOS_INPUT_DIR"]).expanduser())
    candidates.extend(
        [
            REPO_ROOT.parent / "ESSOS_mgrid_pr" / "examples" / "input_files",
            REPO_ROOT.parent / "ESSOS" / "examples" / "input_files",
            Path.cwd() / "examples" / "input_files",
        ]
    )
    return candidates


def _find_default_coils_json() -> Path:
    name = "ESSOS_biot_savart_LandremanPaulQA.json"
    for directory in _candidate_essos_input_dirs():
        path = directory / name
        if path.exists():
            return path
    searched = "\n  ".join(str(p) for p in _candidate_essos_input_dirs())
    raise FileNotFoundError(
        f"Could not find {name}. Set ESSOS_INPUT_DIR to an ESSOS examples/input_files directory. Searched:\n  {searched}"
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="JSON output path.")
    p.add_argument("--workdir", type=Path, default=None, help="Directory for the generated temporary input deck.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Base VMEC input deck.")
    p.add_argument("--coils-json", type=Path, default=None, help="ESSOS coil JSON. Defaults to local ESSOS examples.")
    p.add_argument("--current-scales", type=float, nargs="+", default=list(DEFAULT_CURRENT_SCALES))
    p.add_argument("--pressure-scale", type=float, default=DEFAULT_PRESSURE_SCALE)
    p.add_argument(
        "--activate-fsq",
        type=float,
        default=1.0,
        help=(
            "Free-boundary activation threshold for short sensitivity probes. "
            "Use 1e-3 for literal VMEC2000 cadence parity; the default forces "
            "early active vacuum coupling."
        ),
    )
    p.add_argument("--max-iter", type=int, default=4)
    p.add_argument("--niter-array", type=int, nargs="+", default=None)
    p.add_argument("--ns", type=int, default=12)
    p.add_argument("--mpol", type=int, default=4)
    p.add_argument("--ntor", type=int, default=4)
    p.add_argument("--nzeta", type=int, default=8)
    p.add_argument("--ftol", type=float, default=1.0e-8)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--regularization-epsilon", type=float, default=0.0)
    p.add_argument("--jit-forces", action="store_true", help="Enable JIT force kernels. Disabled by default for quick probes.")
    p.add_argument(
        "--geometry-perturb-scale",
        type=float,
        default=0.0,
        help="If nonzero, add one extra case perturbing one coil Fourier coefficient by this amount.",
    )
    p.add_argument("--geometry-perturb-current-scale", type=float, default=1.0)
    p.add_argument("--geometry-perturb-coil", type=int, default=0)
    p.add_argument("--geometry-perturb-axis", choices=("x", "y", "z"), default="x")
    p.add_argument(
        "--geometry-perturb-coeff",
        type=int,
        default=2,
        help="Fourier coefficient index to perturb. ESSOS convention: 0=constant, 1=sin(1), 2=cos(1).",
    )
    return p


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(data), indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def _skip(out: Path, *, reason: str, error: str, args: argparse.Namespace) -> int:
    help_text = (
        "Install/import ESSOS and ensure the coil JSON is available. Typical local usage: "
        "PYTHONPATH=/path/to/ESSOS_mgrid_pr:$PYTHONPATH and optionally "
        "ESSOS_INPUT_DIR=/path/to/ESSOS/examples/input_files."
    )
    payload = {
        "status": "skipped",
        "reason": reason,
        "error": error,
        "help": help_text,
        "requested_out": str(out),
        "args": vars(args),
    }
    _write_json(out, payload)
    print(f"[freeb-direct-provider] skipped: {reason}: {error}", file=sys.stderr)
    print(f"[freeb-direct-provider] wrote {out}", file=sys.stderr)
    return 0


def _make_direct_indata(base_indata: Any, *, args: argparse.Namespace) -> Any:
    indata = deepcopy(base_indata)
    niter_array = [int(v) for v in (args.niter_array if args.niter_array is not None else [args.max_iter])]
    indata.scalars["LFREEB"] = True
    indata.scalars["MGRID_FILE"] = "DIRECT_COILS"
    indata.scalars["EXTCUR"] = [1.0]
    indata.scalars["NS_ARRAY"] = [int(args.ns)]
    indata.scalars["NITER_ARRAY"] = niter_array
    indata.scalars["FTOL_ARRAY"] = [float(args.ftol) for _ in niter_array]
    indata.scalars["NITER"] = int(args.max_iter)
    indata.scalars["FTOL"] = float(args.ftol)
    indata.scalars["MPOL"] = int(args.mpol)
    indata.scalars["NTOR"] = int(args.ntor)
    indata.scalars["NZETA"] = int(args.nzeta)
    indata.scalars["NTHETA"] = 0
    indata.scalars["NVACSKIP"] = max(1, int(args.nzeta))
    indata.scalars["PMASS_TYPE"] = "power_series"
    indata.scalars["AM"] = [1.0, -1.0]
    indata.scalars["PRES_SCALE"] = float(args.pressure_scale)
    return indata


def _last_float(value: Any) -> float | None:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    val = float(arr[-1])
    return val if np.isfinite(val) else None


def _rms(value: Any) -> float | None:
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return None
        out = float(np.sqrt(np.mean(arr * arr)))
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _mean(value: Any) -> float | None:
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return None
        out = float(np.mean(arr))
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _min(value: Any) -> float | None:
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return None
        out = float(np.min(arr))
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _max(value: Any) -> float | None:
    try:
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return None
        out = float(np.max(arr))
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _field_stats(prefix: str, br: Any, bp: Any, bz: Any) -> dict[str, float | None]:
    br_arr = np.asarray(br, dtype=float)
    bp_arr = np.asarray(bp, dtype=float)
    bz_arr = np.asarray(bz, dtype=float)
    bmag = np.sqrt(br_arr * br_arr + bp_arr * bp_arr + bz_arr * bz_arr)
    return {
        f"{prefix}_br_rms": _rms(br_arr),
        f"{prefix}_bp_rms": _rms(bp_arr),
        f"{prefix}_bz_rms": _rms(bz_arr),
        f"{prefix}_bmag_mean": _mean(bmag),
        f"{prefix}_bmag_rms": _rms(bmag),
        f"{prefix}_bmag_max": _max(bmag),
    }


def _sample_direct_external_summary(*, state: Any, static: Any, params: Any) -> dict[str, Any]:
    from vmec_jax.free_boundary import _sample_external_boundary_arrays

    t0 = time.perf_counter()
    try:
        sample = _sample_external_boundary_arrays(
            state=state,
            static=static,
            plascur=0.0,
            external_field_provider_kind="direct_coils",
            external_field_provider_params=params,
        )
        vac = sample.vac_ext
        out: dict[str, Any] = {
            "enabled": True,
            "available": True,
            "provider_kind": "direct_coils",
            "mgrid_path": sample.mgrid_path,
            "shape": list(np.asarray(sample.br).shape),
            "n_samples": int(np.asarray(sample.br).size),
            "R_min": _min(sample.R),
            "R_max": _max(sample.R),
            "Z_min": _min(sample.Z),
            "Z_max": _max(sample.Z),
            "axis_r_mean": _mean(sample.axis_r),
            "axis_z_mean": _mean(sample.axis_z),
            "bu_rms": _rms(vac.bu),
            "bv_rms": _rms(vac.bv),
            "bsupu_rms": _rms(vac.bsupu),
            "bsupv_rms": _rms(vac.bsupv),
            "bsqvac_mean": _mean(vac.bsqvac),
            "bsqvac_max": _max(vac.bsqvac),
            "bnormal_rms": _rms(vac.bnormal),
            "bnormal_unit_rms": _rms(vac.bnormal_unit),
            "det_guv_min": _min(vac.det_guv),
            "det_guv_max": _max(vac.det_guv),
        }
        out.update(_field_stats("direct", sample.br_mgrid, sample.bp_mgrid, sample.bz_mgrid))
        out.update(_field_stats("axis", sample.br_axis, sample.bp_axis, sample.bz_axis))
        out.update(_field_stats("total", sample.br, sample.bp, sample.bz))
    except Exception as exc:
        out = {
            "enabled": True,
            "available": False,
            "provider_kind": "direct_coils",
            "reason": "sample_failed",
            "error": repr(exc),
        }
    out["sample_time_s"] = float(max(0.0, time.perf_counter() - t0))
    return out


def _numeric_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, after_val in after.items():
        if isinstance(after_val, bool) or isinstance(before.get(key), bool):
            continue
        if not isinstance(after_val, (int, float)) or not isinstance(before.get(key), (int, float)):
            continue
        before_val = float(before[key])
        after_val_f = float(after_val)
        if not (np.isfinite(before_val) and np.isfinite(after_val_f)):
            continue
        delta[f"{key}_after_minus_before"] = after_val_f - before_val
    return delta


def _array_delta(after: Any, before: Any) -> dict[str, float] | None:
    try:
        after_arr = np.asarray(after, dtype=float).reshape(-1)
        before_arr = np.asarray(before, dtype=float).reshape(-1)
    except Exception:
        return None
    if after_arr.shape != before_arr.shape or after_arr.size == 0:
        return None
    diff = after_arr - before_arr
    abs_rms = float(np.sqrt(np.mean(diff * diff)))
    before_rms = float(np.sqrt(np.mean(before_arr * before_arr)))
    rel_rms = abs_rms / max(before_rms, 1.0e-300)
    max_abs = float(np.max(np.abs(diff)))
    if not (np.isfinite(abs_rms) and np.isfinite(rel_rms) and np.isfinite(max_abs)):
        return None
    return {
        "absolute_rms_delta": abs_rms,
        "relative_rms_delta": rel_rms,
        "max_abs_delta": max_abs,
    }


def _accepted_state_vector(run: Any) -> np.ndarray | None:
    try:
        from vmec_jax.state import pack_state

        vec = np.asarray(pack_state(run.state), dtype=float).reshape(-1)
    except Exception:
        return None
    if vec.size == 0 or not np.isfinite(vec).all():
        return None
    return vec


def _accepted_state_summary(vec: np.ndarray | None) -> dict[str, Any]:
    if vec is None:
        return {"available": False}
    return {
        "available": True,
        "size": int(vec.size),
        "rms": _rms(vec),
        "max_abs": _max(np.abs(vec)),
    }


def _equilibrium_summary(run: Any) -> dict[str, Any]:
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

    out: dict[str, Any] = {}
    try:
        out["aspect"] = float(equilibrium_aspect_ratio_from_state(state=run.state, static=run.static))
    except Exception as exc:
        out["aspect_error"] = repr(exc)
    try:
        _chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
        )
        iotas_arr = np.asarray(iotas, dtype=float)
        iotaf_arr = np.asarray(iotaf, dtype=float)
        out["mean_iota"] = float(np.nanmean(iotas_arr)) if iotas_arr.size else None
        out["mean_iota_no_axis"] = float(np.nanmean(iotas_arr[1:])) if iotas_arr.size > 1 else out["mean_iota"]
        out["iota_profile"] = iotas_arr
        out["iotaf_profile"] = iotaf_arr
    except Exception as exc:
        out["iota_error"] = repr(exc)
    return out


def _fsq_summary(run: Any) -> dict[str, Any]:
    result = run.result
    if result is None:
        return {}
    fsqr = _last_float(getattr(result, "fsqr2_history", None))
    fsqz = _last_float(getattr(result, "fsqz2_history", None))
    fsql = _last_float(getattr(result, "fsql2_history", None))
    fsq_sum = None if None in (fsqr, fsqz, fsql) else float(fsqr + fsqz + fsql)
    fsq_norm = None if None in (fsqr, fsqz, fsql) else float(np.sqrt(fsqr * fsqr + fsqz * fsqz + fsql * fsql))
    return {
        "n_iter": int(getattr(result, "n_iter", -1)),
        "fsqr": fsqr,
        "fsqz": fsqz,
        "fsql": fsql,
        "fsq_sum": fsq_sum,
        "fsq_norm": fsq_norm,
        "w_final": _last_float(getattr(result, "w_history", None)),
    }


def _free_boundary_diagnostics(run: Any) -> dict[str, Any]:
    result = run.result
    if result is None:
        return {}
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    keys = [
        "free_boundary",
        "free_boundary_external_field",
        "freeb_ivac_history",
        "freeb_ivacskip_history",
        "freeb_full_update_history",
        "freeb_nestor_reused_history",
        "freeb_nestor_solve_time_history",
        "freeb_nestor_sample_time_history",
    ]
    return {key: diagnostics[key] for key in keys if key in diagnostics}


def _scale_current_params(params: Any, scale: float) -> Any:
    return replace(params, current_scale=float(params.current_scale) * float(scale))


def _perturb_geometry_params(params: Any, *, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    axis_index = {"x": 0, "y": 1, "z": 2}[str(args.geometry_perturb_axis)]
    dofs = np.array(params.base_curve_dofs, dtype=float, copy=True)
    coil_index = int(args.geometry_perturb_coil)
    coeff_index = int(args.geometry_perturb_coeff)
    if coil_index < 0 or coil_index >= int(dofs.shape[0]):
        raise ValueError(f"geometry perturb coil index {coil_index} outside [0, {int(dofs.shape[0]) - 1}]")
    if coeff_index < 0 or coeff_index >= int(dofs.shape[2]):
        raise ValueError(f"geometry perturb coeff index {coeff_index} outside [0, {int(dofs.shape[2]) - 1}]")
    before = float(dofs[coil_index, axis_index, coeff_index])
    dofs[coil_index, axis_index, coeff_index] = before + float(args.geometry_perturb_scale)
    metadata = {
        "delta": float(args.geometry_perturb_scale),
        "coil_index": coil_index,
        "axis": str(args.geometry_perturb_axis),
        "axis_index": axis_index,
        "coeff_index": coeff_index,
        "coefficient_before": before,
        "coefficient_after": float(dofs[coil_index, axis_index, coeff_index]),
    }
    return params.with_arrays(base_curve_dofs=dofs), metadata


def _case_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = [
        {
            "label": f"current_scale_{float(scale):.6g}",
            "current_scale": float(scale),
            "geometry_perturbation": None,
        }
        for scale in args.current_scales
    ]
    if float(args.geometry_perturb_scale) != 0.0:
        specs.append(
            {
                "label": (
                    f"geometry_perturb_current_scale_{float(args.geometry_perturb_current_scale):.6g}"
                ),
                "current_scale": float(args.geometry_perturb_current_scale),
                "geometry_perturbation": "requested",
            }
        )
    return specs


def _run_case(
    *,
    input_path: Path,
    initial_run: Any,
    base_params: Any,
    spec: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from vmec_jax.driver import run_free_boundary

    params = _scale_current_params(base_params, float(spec["current_scale"]))
    geometry_perturbation = None
    if spec.get("geometry_perturbation") == "requested":
        params, geometry_perturbation = _perturb_geometry_params(params, args=args)

    before = _sample_direct_external_summary(state=initial_run.state, static=initial_run.static, params=params)
    t0 = time.perf_counter()
    run = run_free_boundary(
        input_path,
        max_iter=int(args.max_iter),
        multigrid=False,
        verbose=False,
        jit_forces=bool(args.jit_forces),
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
        free_boundary_activate_fsq=float(args.activate_fsq),
    )
    wall_s = float(max(0.0, time.perf_counter() - t0))
    after = _sample_direct_external_summary(state=run.state, static=run.static, params=params)
    state_vec = _accepted_state_vector(run)

    return {
        "label": str(spec["label"]),
        "status": "completed",
        "current_scale": float(spec["current_scale"]),
        "provider_current_scale": float(params.current_scale),
        "geometry_perturbation": geometry_perturbation,
        "wall_s": wall_s,
        "fsq": _fsq_summary(run),
        "equilibrium": _equilibrium_summary(run),
        "free_boundary_diagnostics": _free_boundary_diagnostics(run),
        "accepted_state": _accepted_state_summary(state_vec),
        "_accepted_state_vector": state_vec,
        "direct_external_field_before": before,
        "direct_external_field_after": after,
        "direct_external_field_delta": _numeric_delta(after, before),
    }


def _annotate_accepted_state_deltas(payload: dict[str, Any]) -> None:
    completed = [
        run
        for run in payload.get("runs", [])
        if isinstance(run, dict) and run.get("status") == "completed" and run.get("_accepted_state_vector") is not None
    ]
    if not completed:
        payload["accepted_state_reference"] = None
        return

    reference = min(completed, key=lambda run: abs(float(run.get("current_scale", 0.0)) - 1.0))
    reference_vec = reference.get("_accepted_state_vector")
    payload["accepted_state_reference"] = {
        "label": reference.get("label"),
        "current_scale": reference.get("current_scale"),
    }

    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        vec = run.get("_accepted_state_vector")
        if vec is not None and reference_vec is not None:
            delta = _array_delta(vec, reference_vec)
            if delta is not None:
                run["accepted_state_delta_from_reference"] = delta
        run.pop("_accepted_state_vector", None)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = args.out.expanduser().resolve()
    workdir = (args.workdir or (out.parent / f"{out.stem}_work")).expanduser().resolve()

    if int(args.max_iter) < 1:
        raise SystemExit("--max-iter must be >= 1 for this probe")
    if not args.current_scales:
        raise SystemExit("--current-scales must contain at least one value")

    try:
        from essos.coils import Coils_from_json
    except Exception as exc:
        return _skip(out, reason="essos_import_failed", error=repr(exc), args=args)

    try:
        if args.coils_json is not None:
            coils_json = args.coils_json.expanduser().resolve()
            if not coils_json.exists():
                raise FileNotFoundError(f"Requested --coils-json does not exist: {coils_json}")
        else:
            coils_json = _find_default_coils_json()
    except Exception as exc:
        return _skip(out, reason="coils_json_not_found", error=str(exc), args=args)

    try:
        from vmec_jax._compat import enable_x64
        from vmec_jax.driver import run_free_boundary
        from vmec_jax.external_fields import from_essos_coils
        from vmec_jax.namelist import read_indata, write_indata

        enable_x64(True)
        coils = Coils_from_json(str(coils_json))
        base_params = from_essos_coils(
            coils,
            regularization_epsilon=float(args.regularization_epsilon),
            chunk_size=int(args.chunk_size) if args.chunk_size and int(args.chunk_size) > 0 else None,
        )
    except Exception as exc:
        return _skip(out, reason="essos_or_direct_provider_unavailable", error=repr(exc), args=args)

    workdir.mkdir(parents=True, exist_ok=True)
    direct_input = workdir / "input.freeb_direct_provider_sensitivity"
    base_indata = read_indata(args.input.expanduser().resolve())
    direct_indata = _make_direct_indata(base_indata, args=args)
    write_indata(direct_input, direct_indata)

    payload: dict[str, Any] = {
        "status": "running",
        "script": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "base_input": str(args.input.expanduser().resolve()),
        "generated_input": str(direct_input),
        "coils_json": str(coils_json),
        "pressure_scale": float(args.pressure_scale),
        "activate_fsq": float(args.activate_fsq),
        "max_iter": int(args.max_iter),
        "niter_array": [int(v) for v in (args.niter_array if args.niter_array is not None else [args.max_iter])],
        "current_scales": [float(v) for v in args.current_scales],
        "probe_grid": {
            "ns": int(args.ns),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "nzeta": int(args.nzeta),
            "ftol": float(args.ftol),
        },
        "direct_provider": {
            "n_base_coils": int(np.asarray(base_params.base_currents).size),
            "n_segments": int(base_params.n_segments),
            "nfp": int(base_params.nfp),
            "stellsym": bool(base_params.stellsym),
            "base_current_scale": float(base_params.current_scale),
            "chunk_size": None if base_params.chunk_size is None else int(base_params.chunk_size),
            "regularization_epsilon": float(base_params.regularization_epsilon),
        },
        "initial_equilibrium": {},
        "runs": [],
    }

    print(f"[freeb-direct-provider] generated input: {direct_input}")
    print("[freeb-direct-provider] building initial state")
    initial_run = run_free_boundary(
        direct_input,
        use_initial_guess=True,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
    )
    payload["initial_equilibrium"] = _equilibrium_summary(initial_run)

    errors = 0
    for spec in _case_specs(args):
        print(f"[freeb-direct-provider] running {spec['label']}")
        try:
            payload["runs"].append(
                _run_case(
                    input_path=direct_input,
                    initial_run=initial_run,
                    base_params=base_params,
                    spec=spec,
                    args=args,
                )
            )
        except Exception as exc:
            errors += 1
            payload["runs"].append(
                {
                    "label": str(spec.get("label", "unknown")),
                    "status": "error",
                    "current_scale": spec.get("current_scale"),
                    "error": repr(exc),
                }
            )
            print(f"[freeb-direct-provider] case failed {spec.get('label')}: {exc!r}", file=sys.stderr)

    _annotate_accepted_state_deltas(payload)
    payload["status"] = "completed" if errors == 0 else "completed_with_errors"
    payload["error_count"] = int(errors)
    _write_json(out, payload)
    print(f"[freeb-direct-provider] wrote {out}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

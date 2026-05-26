#!/usr/bin/env python3
"""Lightweight external-field provider benchmark.

Default run:

    python tools/benchmarks/bench_external_field_providers.py

The default synthetic direct-coil case is intentionally small and CPU-friendly.
If ESSOS and a Landreman-Paul QA coil JSON are available, an additional
ESSOS-converted direct-coil case is benchmarked; otherwise it is reported as a
clear skip in the JSON output.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

DEFAULT_OUT = REPO_ROOT / "results" / "bench_external_field_providers.json"
ESSOS_COILS_NAME = "ESSOS_biot_savart_LandremanPaulQA.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON summary path.")
    p.add_argument("--points", type=int, default=48, help="Number of cylindrical sample points.")
    p.add_argument("--segments", type=int, default=48, help="Segments per synthetic coil.")
    p.add_argument("--warm-repeats", type=int, default=5, help="Warm timing repeats after cold/compile timing.")
    p.add_argument("--chunk-size", type=int, default=0, help="Optional direct-coil point chunk size; 0 disables.")
    p.add_argument("--coils-json", type=Path, default=None, help="Optional ESSOS coil JSON.")
    p.add_argument("--skip-essos", action="store_true", help="Do not probe optional ESSOS converted coils.")
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


def _block_until_ready(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)
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


def _field_stats(value: Any) -> dict[str, Any]:
    br, bp, bz = (np.asarray(component, dtype=float) for component in value)
    bmag = np.sqrt(br * br + bp * bp + bz * bz)
    return {
        "shape": list(br.shape),
        "br_rms": float(np.sqrt(np.mean(br * br))),
        "bp_rms": float(np.sqrt(np.mean(bp * bp))),
        "bz_rms": float(np.sqrt(np.mean(bz * bz))),
        "bmag_mean": float(np.mean(bmag)),
        "bmag_max": float(np.max(bmag)),
    }


def _sample_points(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = max(1, int(n))
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R = 0.7 + 0.18 * np.cos(theta)
    Z = 0.18 * np.sin(theta)
    phi = np.linspace(0.0, 0.5 * np.pi, n, endpoint=False)
    return R, Z, phi


def _synthetic_params(*, segments: int, chunk_size: int | None) -> Any:
    from vmec_jax._compat import jnp
    from vmec_jax.external_fields import CoilFieldParams

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(1.35)
    dofs = dofs.at[0, 1, 1].set(1.35)
    return CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.0e6], dtype=float),
        n_segments=int(segments),
        nfp=1,
        stellsym=False,
        chunk_size=chunk_size,
    )


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


def _essos_params(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from essos.coils import Coils_from_json
    from vmec_jax.external_fields import from_essos_coils

    coils_json = _find_essos_json(args.coils_json)
    coils = Coils_from_json(str(coils_json))
    params = from_essos_coils(
        coils,
        chunk_size=int(args.chunk_size) if int(args.chunk_size) > 0 else None,
    )
    # Keep the optional case bounded even when the fixture contains high-resolution curves.
    params = replace(params, n_segments=min(int(params.n_segments), int(args.segments)))
    return params, {"coils_json": coils_json}


def _bench_case(
    *,
    label: str,
    provider_kind: str,
    params: Any,
    R: np.ndarray,
    Z: np.ndarray,
    phi: np.ndarray,
    warm_repeats: int,
) -> dict[str, Any]:
    from vmec_jax._compat import has_jax, jax
    from vmec_jax.external_fields import sample_external_field_cylindrical

    def eager_sample() -> Any:
        return sample_external_field_cylindrical(provider_kind, None, params, R, Z, phi)

    compiled_sample = None
    compile_available = False
    compile_error = None
    if has_jax() and jax is not None:
        try:
            compiled_sample = jax.jit(
                lambda trial_params, r, z, p: sample_external_field_cylindrical(
                    provider_kind,
                    None,
                    trial_params,
                    r,
                    z,
                    p,
                )
            )
            compile_available = True
        except Exception as exc:
            compile_error = repr(exc)

    cold_s, cold_value = _time_once(eager_sample if compiled_sample is None else lambda: compiled_sample(params, R, Z, phi))
    warm_times: list[float] = []
    value = cold_value
    for _ in range(max(1, int(warm_repeats))):
        dt, value = _time_once(eager_sample if compiled_sample is None else lambda: compiled_sample(params, R, Z, phi))
        warm_times.append(dt)

    return {
        "label": label,
        "status": "completed",
        "provider_kind": provider_kind,
        "compile_path": "jax.jit" if compile_available else "eager",
        "compile_error": compile_error,
        "cold_or_compile_s": cold_s,
        "warm": _summarize_timings(warm_times),
        "field": _field_stats(value),
    }


def _bench_cached_geometry_case(
    *,
    label: str,
    params: Any,
    R: np.ndarray,
    Z: np.ndarray,
    phi: np.ndarray,
    warm_repeats: int,
) -> dict[str, Any]:
    from vmec_jax._compat import has_jax, jax
    from vmec_jax.external_fields.coils_jax import build_coil_field_geometry, sample_coil_field_cylindrical_from_geometry

    def eager_build() -> Any:
        return build_coil_field_geometry(params)

    compiled_build = None
    compiled_sample = None
    compile_available = False
    compile_error = None
    if has_jax() and jax is not None:
        try:
            compiled_build = jax.jit(build_coil_field_geometry)
            compiled_sample = jax.jit(
                lambda geometry, r, z, p: sample_coil_field_cylindrical_from_geometry(
                    geometry,
                    r,
                    z,
                    p,
                    regularization_epsilon=params.regularization_epsilon,
                    chunk_size=params.chunk_size,
                )
            )
            compile_available = True
        except Exception as exc:
            compile_error = repr(exc)

    cold_build_s, geometry = _time_once(eager_build if compiled_build is None else lambda: compiled_build(params))
    build_warm_times: list[float] = []
    for _ in range(max(1, int(warm_repeats))):
        dt, geometry = _time_once(eager_build if compiled_build is None else lambda: compiled_build(params))
        build_warm_times.append(dt)

    def eager_sample() -> Any:
        return sample_coil_field_cylindrical_from_geometry(
            geometry,
            R,
            Z,
            phi,
            regularization_epsilon=params.regularization_epsilon,
            chunk_size=params.chunk_size,
        )

    cold_field_s, cold_value = _time_once(
        eager_sample if compiled_sample is None else lambda: compiled_sample(geometry, R, Z, phi)
    )
    field_warm_times: list[float] = []
    value = cold_value
    for _ in range(max(1, int(warm_repeats))):
        dt, value = _time_once(eager_sample if compiled_sample is None else lambda: compiled_sample(geometry, R, Z, phi))
        field_warm_times.append(dt)

    return {
        "label": label,
        "status": "completed",
        "provider_kind": "direct_coils_cached_geometry",
        "compile_path": "jax.jit" if compile_available else "eager",
        "compile_error": compile_error,
        "cold_or_compile_s": cold_field_s,
        "warm": _summarize_timings(field_warm_times),
        "geometry_build": {
            "cold_or_compile_s": cold_build_s,
            "warm": _summarize_timings(build_warm_times),
        },
        "field": _field_stats(value),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.points) < 1:
        raise SystemExit("--points must be >= 1")
    if int(args.segments) < 4:
        raise SystemExit("--segments must be >= 4")

    from vmec_jax._compat import enable_x64

    enable_x64(bool(args.enable_x64))
    R, Z, phi = _sample_points(int(args.points))
    chunk_size = int(args.chunk_size) if int(args.chunk_size) > 0 else None

    payload: dict[str, Any] = {
        "status": "completed",
        "script": str(Path(__file__).resolve()),
        "backend": _backend_info(),
        "parameters": {
            "points": int(args.points),
            "segments": int(args.segments),
            "warm_repeats": int(args.warm_repeats),
            "chunk_size": chunk_size,
        },
        "cases": [],
    }

    synthetic = _synthetic_params(segments=int(args.segments), chunk_size=chunk_size)
    payload["cases"].append(
        _bench_case(
            label="synthetic_direct_coils",
            provider_kind="direct_coils",
            params=synthetic,
            R=R,
            Z=Z,
            phi=phi,
            warm_repeats=int(args.warm_repeats),
        )
    )
    payload["cases"].append(
        _bench_cached_geometry_case(
            label="synthetic_direct_coils_cached_geometry",
            params=synthetic,
            R=R,
            Z=Z,
            phi=phi,
            warm_repeats=int(args.warm_repeats),
        )
    )

    if args.skip_essos:
        payload["cases"].append({"label": "essos_direct_coils", "status": "skipped", "reason": "requested"})
        payload["cases"].append({"label": "essos_direct_coils_cached_geometry", "status": "skipped", "reason": "requested"})
    else:
        try:
            essos_params, metadata = _essos_params(args)
            case = _bench_case(
                label="essos_direct_coils",
                provider_kind="direct_coils",
                params=essos_params,
                R=R,
                Z=Z,
                phi=phi,
                warm_repeats=int(args.warm_repeats),
            )
            case["metadata"] = metadata
            payload["cases"].append(case)
            cached_case = _bench_cached_geometry_case(
                label="essos_direct_coils_cached_geometry",
                params=essos_params,
                R=R,
                Z=Z,
                phi=phi,
                warm_repeats=int(args.warm_repeats),
            )
            cached_case["metadata"] = metadata
            payload["cases"].append(cached_case)
        except Exception as exc:
            payload["cases"].append(
                {
                    "label": "essos_direct_coils",
                    "status": "skipped",
                    "reason": "essos_fixture_unavailable",
                    "error": repr(exc),
                }
            )
            payload["cases"].append(
                {
                    "label": "essos_direct_coils_cached_geometry",
                    "status": "skipped",
                    "reason": "essos_fixture_unavailable",
                    "error": repr(exc),
                }
            )

    out = args.out.expanduser().resolve()
    _write_json(out, payload)
    print(f"[bench-external-field-providers] wrote {out}")
    for case in payload["cases"]:
        if case["status"] == "completed":
            warm = case["warm"]
            print(
                f"[bench-external-field-providers] {case['label']}: "
                f"cold_or_compile={case['cold_or_compile_s']:.6f}s warm_min={warm['min_s']:.6f}s"
            )
        else:
            print(
                f"[bench-external-field-providers] {case['label']}: "
                f"skipped ({case.get('reason', 'unknown')})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

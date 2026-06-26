#!/usr/bin/env python3
"""Small coil-gradient benchmarks for the direct free-boundary path.

This keeps the default workload bounded by timing differentiable direct-coil
field objectives and the dense vacuum adjoint solve used by free-boundary
sensitivity tests.  If JAX is unavailable, the script writes a skipped JSON
summary and exits successfully.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / "results" / "bench_freeb_coil_gradient.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON summary path.")
    p.add_argument("--points", type=int, default=24, help="Boundary-like sample points.")
    p.add_argument("--segments", type=int, default=48, help="Segments per synthetic coil.")
    p.add_argument("--matrix-size", type=int, default=24, help="Dense vacuum adjoint matrix size.")
    p.add_argument("--warm-repeats", type=int, default=5)
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


def _sample_points(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, max(1, int(n)), endpoint=False)
    return 0.7 + 0.18 * np.cos(theta), 0.18 * np.sin(theta), np.linspace(0.0, 0.5 * np.pi, theta.size)


def _coil_gradient_case(points: int, segments: int, warm_repeats: int) -> dict[str, Any]:
    from vmec_jax._compat import jax, jnp
    from vmec_jax.external_fields import CoilFieldParams, sample_coil_field_cylindrical

    R, Z, phi = (jnp.asarray(v) for v in _sample_points(points))
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(1.35)
    dofs = dofs.at[0, 1, 1].set(1.35)
    currents = jnp.asarray([2.0e6], dtype=float)

    def objective(curve_dofs: Any, base_currents: Any) -> Any:
        params = CoilFieldParams(
            base_curve_dofs=curve_dofs,
            base_currents=base_currents,
            n_segments=int(segments),
            nfp=1,
            stellsym=False,
        )
        br, bp, bz = sample_coil_field_cylindrical(params, R, Z, phi)
        return jnp.mean(br * br + 0.3 * bp * bp + 1.7 * bz * bz)

    value_and_grad = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    cold_s, cold_value = _time_once(lambda: value_and_grad(dofs, currents))
    warm_times: list[float] = []
    value = cold_value
    for _ in range(max(1, int(warm_repeats))):
        dt, value = _time_once(lambda: value_and_grad(dofs, currents))
        warm_times.append(dt)
    objective_value, grads = value
    grad_dofs, grad_currents = grads
    return {
        "label": "direct_coil_field_value_and_grad",
        "status": "completed",
        "cold_or_compile_s": cold_s,
        "warm": _summarize_timings(warm_times),
        "objective": float(np.asarray(objective_value)),
        "grad_dofs_norm": float(np.linalg.norm(np.asarray(grad_dofs))),
        "grad_currents_norm": float(np.linalg.norm(np.asarray(grad_currents))),
    }


def _dense_vacuum_adjoint_case(matrix_size: int, warm_repeats: int) -> dict[str, Any]:
    from vmec_jax._compat import jax, jnp
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import dense_vacuum_solve_jax

    n = max(2, int(matrix_size))
    idx = jnp.arange(n, dtype=float)
    A = jnp.eye(n, dtype=float) * 3.0
    A = A + 0.05 / (1.0 + jnp.abs(idx[:, None] - idx[None, :]))
    b = jnp.sin(0.3 * idx) + 0.1
    cotangent = jnp.cos(0.2 * idx)

    def objective(rhs: Any) -> Any:
        x = dense_vacuum_solve_jax(A, rhs, symmetric=True)
        return jnp.vdot(cotangent, x)

    grad_fn = jax.jit(jax.value_and_grad(objective))
    cold_s, cold_value = _time_once(lambda: grad_fn(b))
    warm_times: list[float] = []
    value = cold_value
    for _ in range(max(1, int(warm_repeats))):
        dt, value = _time_once(lambda: grad_fn(b))
        warm_times.append(dt)
    objective_value, grad_b = value
    return {
        "label": "dense_vacuum_adjoint_rhs_grad",
        "status": "completed",
        "matrix_size": n,
        "cold_or_compile_s": cold_s,
        "warm": _summarize_timings(warm_times),
        "objective": float(np.asarray(objective_value)),
        "grad_rhs_norm": float(np.linalg.norm(np.asarray(grad_b))),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from vmec_jax._compat import enable_x64, has_jax

    enable_x64(bool(args.enable_x64))
    payload: dict[str, Any] = {
        "status": "completed",
        "script": str(Path(__file__).resolve()),
        "backend": _backend_info(),
        "parameters": {
            "points": int(args.points),
            "segments": int(args.segments),
            "matrix_size": int(args.matrix_size),
            "warm_repeats": int(args.warm_repeats),
        },
        "cases": [],
    }
    if not has_jax():
        payload["status"] = "skipped"
        payload["reason"] = "jax_unavailable"
        _write_json(args.out, payload)
        print(f"[bench-freeb-coil-gradient] skipped: JAX unavailable; wrote {args.out.expanduser().resolve()}")
        return 0

    payload["cases"].append(_coil_gradient_case(int(args.points), int(args.segments), int(args.warm_repeats)))
    payload["cases"].append(_dense_vacuum_adjoint_case(int(args.matrix_size), int(args.warm_repeats)))

    out = args.out.expanduser().resolve()
    _write_json(out, payload)
    print(f"[bench-freeb-coil-gradient] wrote {out}")
    for case in payload["cases"]:
        warm = case["warm"]
        print(
            f"[bench-freeb-coil-gradient] {case['label']}: "
            f"cold_or_compile={case['cold_or_compile_s']:.6f}s warm_min={warm['min_s']:.6f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

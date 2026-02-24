#!/usr/bin/env python3
"""Compare two wout files with optional near-axis exclusion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _open_netcdf(path: Path):
    try:
        import netCDF4  # type: ignore

        ds = netCDF4.Dataset(path, "r")
        return ds, "netcdf4"
    except Exception:
        from scipy.io import netcdf  # type: ignore

        ds = netcdf.netcdf_file(str(path), "r", mmap=False)
        return ds, "scipy"


def _dim_len(ds: Any, name: str) -> int | None:
    if name not in ds.dimensions:
        return None
    dim = ds.dimensions[name]
    try:
        return int(len(dim))
    except TypeError:
        return int(dim)


def _infer_ns(ds: Any) -> int | None:
    for key in ("radius", "ns", "surf", "s"):
        n = _dim_len(ds, key)
        if n is not None:
            return n
    if "iotaf" in ds.variables:
        return int(np.asarray(ds.variables["iotaf"][:]).shape[0])
    return None


def _is_numeric(arr: np.ndarray) -> bool:
    return arr.dtype.kind in ("f", "i", "u")


def _apply_axis_skip(arr: np.ndarray, ns: int | None, skip: int) -> np.ndarray:
    if arr.ndim == 0 or skip <= 0 or ns is None:
        return arr
    out = arr
    # Apply to every radial-like axis, not only axis=0.
    for axis, n in enumerate(arr.shape):
        if n in (ns, ns - 1) and n > skip:
            slicer = [slice(None)] * out.ndim
            slicer[axis] = slice(skip, None)
            out = out[tuple(slicer)]
    return out


@dataclass
class Row:
    name: str
    max_abs: float
    max_rel: float
    mean_rel: float
    passed: bool


def _compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> tuple[float, float, float, bool]:
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1.0e-30)
    rel = diff / denom
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    mean_rel = float(np.nanmean(rel)) if rel.size else 0.0
    passed = bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    return max_abs, max_rel, mean_rel, passed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="First wout file (reference)")
    p.add_argument("--b", required=True, help="Second wout file (candidate)")
    p.add_argument("--axis-skip", type=int, default=6, help="Skip this many near-axis radial points (default: 6)")
    p.add_argument("--rtol", type=float, default=1.0e-4, help="Relative tolerance")
    p.add_argument("--atol", type=float, default=1.0e-12, help="Absolute tolerance")
    p.add_argument("--top", type=int, default=25, help="How many worst variables to print")
    p.add_argument(
        "--focus",
        default="betapol,betator,DMerc,DGeod,jdotb,bsubsmns,fsqr,fsqz,fsql",
        help="Comma-separated variable names to always print",
    )
    args = p.parse_args()

    path_a = Path(args.a).resolve()
    path_b = Path(args.b).resolve()
    ds_a, _ = _open_netcdf(path_a)
    ds_b, _ = _open_netcdf(path_b)

    try:
        ns_a = _infer_ns(ds_a)
        ns_b = _infer_ns(ds_b)
        ns = ns_a if ns_a == ns_b else ns_a

        common = sorted(set(ds_a.variables.keys()).intersection(ds_b.variables.keys()))
        rows: list[Row] = []
        shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

        for name in common:
            try:
                a = np.asarray(ds_a.variables[name][:])
                b = np.asarray(ds_b.variables[name][:])
            except Exception:
                continue
            if not (_is_numeric(a) and _is_numeric(b)):
                continue
            if a.shape != b.shape:
                shape_mismatch.append((name, a.shape, b.shape))
                continue
            aa = _apply_axis_skip(a, ns=ns, skip=int(args.axis_skip))
            bb = _apply_axis_skip(b, ns=ns, skip=int(args.axis_skip))
            if aa.size == 0:
                continue
            max_abs, max_rel, mean_rel, passed = _compare_arrays(aa, bb, rtol=float(args.rtol), atol=float(args.atol))
            rows.append(Row(name=name, max_abs=max_abs, max_rel=max_rel, mean_rel=mean_rel, passed=passed))

        rows_sorted = sorted(rows, key=lambda r: (r.max_rel, r.max_abs), reverse=True)
        failed = [r for r in rows if not r.passed]
        focus = [x.strip() for x in str(args.focus).split(",") if x.strip()]

        print(f"compare_a={path_a}")
        print(f"compare_b={path_b}")
        print(
            "axis_skip={} ns={} rtol={:.3e} atol={:.3e} vars={} failed={} shape_mismatch={}".format(
                int(args.axis_skip),
                int(ns) if ns is not None else -1,
                float(args.rtol),
                float(args.atol),
                len(rows),
                len(failed),
                len(shape_mismatch),
            )
        )
        print("")
        print(f"Top {int(args.top)} by max_rel:")
        for r in rows_sorted[: int(args.top)]:
            status = "PASS" if r.passed else "FAIL"
            print(
                f"{r.name:20s} {status:4s} max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e} mean_rel={r.mean_rel:.3e}"
            )

        print("")
        print("Focus:")
        row_by_name = {r.name: r for r in rows}
        for key in focus:
            r = row_by_name.get(key)
            if r is None:
                print(f"{key:20s} MISSING")
            else:
                status = "PASS" if r.passed else "FAIL"
                print(
                    f"{key:20s} {status:4s} max_abs={r.max_abs:.3e} max_rel={r.max_rel:.3e} mean_rel={r.mean_rel:.3e}"
                )

        if shape_mismatch:
            print("")
            print("Shape mismatches:")
            for name, sa, sb in shape_mismatch[:50]:
                print(f"{name:20s} {sa} vs {sb}")
        return 0 if (len(failed) == 0 and len(shape_mismatch) == 0) else 1
    finally:
        ds_a.close()
        ds_b.close()


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Micro-benchmark for free-boundary WP1 scaffolding.

Measures:
- config + metadata validation,
- full mgrid field load,
- trilinear interpolation throughput on random boundary-like points.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax.free_boundary import interpolate_mgrid_bfield, load_mgrid, prepare_mgrid_for_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input.* with LFREEB=T")
    ap.add_argument("--interp-points", type=int, default=20000, help="Number of interpolation points")
    ap.add_argument("--interp-repeats", type=int, default=5, help="Interpolation repeats")
    args = ap.parse_args()

    inpath = Path(args.input).expanduser().resolve()
    cfg, _ = vj.load_config(str(inpath))
    if not bool(cfg.lfreeb):
        raise SystemExit(f"{inpath} does not enable LFREEB after parsing")

    t0 = time.perf_counter()
    prep = prepare_mgrid_for_config(cfg, load_fields=False, strict=True)
    t1 = time.perf_counter()
    if prep is None:
        raise SystemExit("prepare_mgrid_for_config returned None for LFREEB case")
    meta = prep.metadata if hasattr(prep, "metadata") else prep

    t2 = time.perf_counter()
    data = load_mgrid(meta.path, load_fields=True)
    t3 = time.perf_counter()

    rng = np.random.default_rng(1234)
    n = int(max(1, args.interp_points))
    rr = rng.uniform(meta.rmin, meta.rmax, size=n)
    zz = rng.uniform(meta.zmin, meta.zmax, size=n)
    phi_period = (2.0 * np.pi) / max(1, int(meta.nfp))
    pp = rng.uniform(0.0, phi_period, size=n)
    extcur = getattr(prep, "extcur", tuple(meta.raw_coil_cur))

    t_interp = []
    br_ref = bp_ref = bz_ref = None
    for _ in range(int(max(1, args.interp_repeats))):
        ti0 = time.perf_counter()
        br, bp, bz = interpolate_mgrid_bfield(data, r=rr, z=zz, phi=pp, extcur=extcur)
        ti1 = time.perf_counter()
        t_interp.append(ti1 - ti0)
        br_ref, bp_ref, bz_ref = br, bp, bz

    print(f"input={inpath}")
    print(f"mgrid={meta.path}")
    print(f"prepare_metadata_s={t1 - t0:.6f}")
    print(f"load_fields_s={t3 - t2:.6f}")
    print(f"interp_points={n}")
    print(f"interp_repeats={int(max(1, args.interp_repeats))}")
    print(f"interp_mean_s={float(np.mean(t_interp)):.6f}")
    print(f"interp_min_s={float(np.min(t_interp)):.6f}")
    if br_ref is not None:
        bmag = np.sqrt(br_ref * br_ref + bp_ref * bp_ref + bz_ref * bz_ref)
        print(f"bmag_mean={float(np.mean(bmag)):.6e}")
        print(f"bmag_max={float(np.max(bmag)):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

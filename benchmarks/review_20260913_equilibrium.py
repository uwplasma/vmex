#!/usr/bin/env python
"""Cold, warm and hot-restarted fixed-boundary solves: VMEX against VMEC++.

Feeds the ``equilibrium_timing`` block of ``benchmarks/review_20260913.json``.
Run in an isolated environment with ``vmecpp`` installed; timings are single
runs on a shared machine and are diagnostic, not rankings.

    PYTHONPATH=. python benchmarks/review_20260913_equilibrium.py \
        examples/data/input.LandremanPaul2021_QA_lowres out.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np


def perturb(coefficients, fraction):
    """Scale every boundary coefficient except the largest (the major radius)."""
    values = np.array(coefficients, dtype=float)
    keep = np.unravel_index(np.argmax(np.abs(values)), values.shape)
    scaled = values * (1.0 + fraction)
    scaled[keep] = values[keep]
    return scaled


def timed(call):
    start = time.time()
    result = call()
    return result, time.time() - start


def vmecpp_rows(deck, threads):
    import vmecpp

    base = vmecpp.VmecInput.from_file(deck)
    base.ns_array = np.array([50])
    base.ftol_array = np.array([1e-11])
    base.niter_array = np.array([5000])
    ladder = vmecpp.VmecInput.from_file(deck)
    ladder.ftol_array = np.array([1e-8, 1e-10, 1e-11])
    ladder.niter_array = np.array([2000, 2000, 3000])
    def run(inp, **kw):
        return vmecpp.run(inp, max_threads=threads, verbose=False, **kw)

    _, ladder_s = timed(lambda: run(ladder))
    cold, cold_s = timed(lambda: run(base))
    rows = {"ladder_cold_s": ladder_s, "ns50_cold_s": cold_s, "niter": int(cold.wout.niter)}
    for fraction in (1e-2, 1e-3, 1e-4):
        moved = base.model_copy(deep=True)
        moved.rbc = perturb(base.rbc, fraction)
        moved.zbs = perturb(base.zbs, fraction)
        fresh, fresh_s = timed(lambda: run(moved))
        hot, hot_s = timed(lambda: run(moved, restart_from=cold))
        rows[f"step_{fraction:g}"] = {
            "cold_s": fresh_s, "cold_iterations": int(fresh.wout.niter),
            "hot_s": hot_s, "hot_iterations": int(hot.wout.niter)}
    return rows


def vmex_rows(deck, mode):
    import jax
    import vmex

    inp = vmex.VmecInput.from_file(deck)
    ladder = replace(inp, ftol_array=np.array([1e-8, 1e-10, 1e-11]),
                     niter_array=np.array([2000, 2000, 3000]))
    base = replace(inp, ns_array=np.array([50]), ftol_array=np.array([1e-11]),
                   niter_array=np.array([5000]))
    def solve(i, **kw):
        return vmex.solve(i, mode=mode, device="cpu", **kw)

    _, ladder_cold = timed(lambda: vmex.solve_multigrid(ladder, mode=mode, device="cpu"))
    _, ladder_warm = timed(lambda: vmex.solve_multigrid(ladder, mode=mode, device="cpu"))
    cold, cold_s = timed(lambda: solve(base))
    warm, warm_s = timed(lambda: solve(base))
    rows = {"jax": jax.__version__, "mode": mode, "ladder_cold_s": ladder_cold,
            "ladder_warm_s": ladder_warm, "ns50_cold_s": cold_s, "ns50_warm_s": warm_s,
            "niter": int(warm.iterations)}
    for fraction in (1e-2, 1e-3, 1e-4):
        moved = replace(base, rbc=perturb(base.rbc, fraction), zbs=perturb(base.zbs, fraction))
        fresh, fresh_s = timed(lambda: solve(moved))
        hot, hot_s = timed(lambda: solve(moved, restart_from=warm))
        rows[f"step_{fraction:g}"] = {
            "warm_s": fresh_s, "warm_iterations": int(fresh.iterations),
            "hot_s": hot_s, "hot_iterations": int(hot.iterations)}
    return rows


def main():
    deck, out = sys.argv[1], sys.argv[2]
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    record = {"deck": deck, "threads": threads, "vmecpp": vmecpp_rows(deck, threads),
              "vmex": {mode: vmex_rows(deck, mode) for mode in ("jit", "cli")}}
    json.dump(record, open(out, "w"), indent=1)
    print(json.dumps(record["vmecpp"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Baseline benchmark harness: VMEC2000 vs vmec_jax vs VMEC++.

Runs a fixed suite of input decks (from ``examples/data``) through each
available solver and records wall time, peak RSS, iteration counts, and
convergence status into ``benchmarks/baseline.json``.  This is the single
source for the README performance figure; re-run it after performance work
(overhaul plan, Phases 0 and 10).

Usage:
    python benchmarks/run_baseline.py [--out baseline.json] [--cases a,b,...]
        [--timeout 600] [--skip vmecpp]

Solvers exercised per case:
- ``vmec2000``       — /Users/rogerio/local/STELLOPT/VMEC2000/Release/xvmec2000
- ``vmec_jax_cold``  — fresh ``vmec`` CLI subprocess (includes JAX/XLA setup)
- ``vmec_jax_warm``  — second in-process core solve (structural executable
  cache hot; same ``vmec_jax.core`` route as the CLI)
- ``vmecpp``         — VMEC++ python API, where the case converges cleanly

Every benchmark row runs at ``ns >= RAMP_NS`` (201): decks whose finest
NS_ARRAY stage is below that get a generated variant with the final stage
rewritten to 201 (``grid: ns201``); decks already at or above 201 run as-is
(``grid: input``).  ns=201 is chosen so the *solve* dominates the wall-clock
and the one-time JIT compile is a small fraction of it — a fairer warm
comparison than ns=51, where compile time is a larger share.  FTOL_ARRAY/
NITER_ARRAY are left untouched, so each deck's own convergence semantics
(final ftol, iteration caps) are kept.  Cases listed in MULTIGRID_CASES
additionally run with a generated coarse->fine NS_ARRAY ladder ending at
``max(deck_ns, 201)`` (``grid: multigrid``) to compare multigrid behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "examples" / "data"
XVMEC2000 = Path("/Users/rogerio/local/STELLOPT/VMEC2000/Release/xvmec2000")
VMECJAX_PY = Path.home() / ".venvs/vmecjax/bin/python"
VMECPP_PY = Path.home() / ".venvs/vmecpp/bin/python"

# (case name, aux files to copy alongside the deck)
CASES: dict[str, list[str]] = {
    "solovev": [],
    "DSHAPE": [],
    "circular_tokamak": [],
    "cth_like_fixed_bdy": [],
    "li383_low_res": [],
    "LandremanPaul2021_QA_lowres": [],
    "LandremanPaul2021_QH_reactorScale_lowres": [],
    "nfp4_QH_warm_start": [],
    "NuhrenbergZille_1988_QHS": [],
    # Free boundary with the real mgrid (R15.1: now converges to VMEC2000
    # parity).  Ramped to ns >= 51 like every other row.
    "cth_like_free_bdy": ["mgrid_cth_like.nc"],
    "cth_like_free_bdy_lasym_small": ["mgrid_cth_like_lasym_small.nc"],
}

# Minimum final-stage ns for every benchmark row (see module docstring).
# 201 so the solve dominates the wall-clock and JIT compile is a small share.
RAMP_NS = 201

# Cases that additionally run with a generated coarse->fine NS_ARRAY ladder.
# Ladder ends at max(deck ns, RAMP_NS) so results stay comparable.
MULTIGRID_CASES = {
    "cth_like_fixed_bdy": "15 31 51 101 201",
    "nfp4_QH_warm_start": "17 35 71 101 201",
    "LandremanPaul2021_QA_lowres": None,  # ladder derived from deck ns below
}

TIME_RE = re.compile(r"(\d+\.\d+)\s+real.*?(\d+)\s+maximum resident set size", re.S)

# Active (non-comment) NS_ARRAY assignment; `!`-commented lines don't match.
NS_ARRAY_RE = re.compile(r"^(\s*NS_ARRAY\s*=\s*)([\d,\s]+)", re.I | re.M)


def read_deck_ns(deck: Path) -> int:
    m = NS_ARRAY_RE.search(deck.read_text())
    return int(m.group(2).replace(",", " ").split()[-1]) if m else 0


def make_ramped_deck(deck: Path, dest: Path, min_ns: int = RAMP_NS) -> None:
    """Rewrite the final NS_ARRAY stage to max(deck_ns, min_ns).

    FTOL_ARRAY/NITER_ARRAY are left untouched (same stage count), so the
    deck's own final-ftol and iteration-cap semantics are preserved.
    """

    def repl(m: re.Match) -> str:
        stages = m.group(2).replace(",", " ").split()
        stages[-1] = str(max(int(stages[-1]), min_ns))
        return m.group(1) + " ".join(stages) + "\n "

    dest.write_text(NS_ARRAY_RE.sub(repl, deck.read_text(), count=1))


def make_multigrid_deck(deck: Path, ladder: str | None, dest: Path) -> None:
    """Rewrite NS_ARRAY/FTOL_ARRAY/NITER_ARRAY as a coarse->fine ladder."""
    ns_final = max(read_deck_ns(deck), RAMP_NS)
    if ladder is None:
        third = max(5, ns_final // 4) | 1
        half = max(third + 2, ns_final // 2) | 1
        ladder = f"{third} {half} {ns_final}"
    stages = ladder.split()
    text = deck.read_text()
    text = NS_ARRAY_RE.sub(lambda m: f"{m.group(1)}{' '.join(stages)}\n ", text, count=1)
    text = re.sub(r"FTOL_ARRAY\s*=\s*[\deE.+\-,\s]+", f"FTOL_ARRAY = {' '.join(['1e-8'] * (len(stages) - 1))} 1e-14\n ", text, flags=re.I)
    text = re.sub(r"NITER_ARRAY\s*=\s*[\d,\s]+", f"NITER_ARRAY = {' '.join(['4000'] * len(stages))}\n ", text, flags=re.I)
    dest.write_text(text)


def timed_subprocess(cmd: list[str], cwd: Path, timeout: int) -> dict:
    """Run under /usr/bin/time -l, return wall seconds, peak RSS, output."""
    full = ["/usr/bin/time", "-l"] + cmd
    t0 = time.time()
    try:
        proc = subprocess.run(full, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "wall_s": timeout}
    wall = time.time() - t0
    m = TIME_RE.search(proc.stderr)
    return {
        "ok": proc.returncode == 0,
        "wall_s": round(float(m.group(1)), 3) if m else round(wall, 3),
        "peak_rss_mb": round(int(m.group(2)) / 2**20, 1) if m else None,
        "stdout": proc.stdout[-4000:],
        "stderr": "" if proc.returncode == 0 else proc.stderr[-2000:],
    }


def parse_vmec2000(out: dict) -> dict:
    txt = out.pop("stdout", "")
    out["converged"] = "EXECUTION TERMINATED NORMALLY" in txt
    iters = re.findall(r"^\s*(\d+)\s+[\d.E+-]+\s+[\d.E+-]+", txt, re.M)
    out["iterations"] = int(iters[-1]) if iters else None
    return out


def parse_vmecjax(out: dict) -> dict:
    txt = out.pop("stdout", "")
    out["converged"] = out["ok"] and ("Wrote WOUT" in txt or "wout" in txt.lower())
    iters = re.findall(r"^\s*(\d+)\s+[\d.E+-]+\s+[\d.E+-]+", txt, re.M)
    out["iterations"] = int(iters[-1]) if iters else None
    return out


def run_vmec2000(deck: Path, aux: list[str], timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        shutil.copy(deck, wd)
        for a in aux:
            shutil.copy(DATA / a, wd)
        return parse_vmec2000(timed_subprocess([str(XVMEC2000), deck.name], wd, timeout))


def run_vmecjax_cold(deck: Path, aux: list[str], timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        shutil.copy(deck, wd)
        for a in aux:
            shutil.copy(DATA / a, wd)
        vmec = VMECJAX_PY.parent / "vmec"
        return parse_vmecjax(timed_subprocess([str(vmec), deck.name], wd, timeout))


WARM_SNIPPET = r"""
import json, sys, time, resource
import numpy as np
import vmec_jax  # sets the persistent XLA compilation cache dir
from vmec_jax.core.input import VmecInput
from vmec_jax.core.multigrid import solve_multigrid

def run(path):
    # Same route as the ``vmec`` CLI (core/cli.py): fixed-boundary decks run
    # the full NS_ARRAY ladder; free-boundary decks run the final stage.
    inp = VmecInput.from_file(path)
    if bool(getattr(inp, "lfreeb", False)):
        from vmec_jax.core.freeboundary import solve_free_boundary
        from vmec_jax.core.solver import resolution_from_input

        ns = int(np.atleast_1d(np.asarray(inp.ns_array))[-1])
        return solve_free_boundary(
            inp, resolution=resolution_from_input(inp, ns=ns),
            error_on_no_convergence=False,
        )
    return solve_multigrid(inp)

path = sys.argv[1]
t0 = time.time(); run(path); t1 = time.time()
t2 = time.time(); run(path); t3 = time.time()
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(json.dumps({"cold_inproc_s": round(t1-t0,3), "warm_s": round(t3-t2,3),
                  "peak_rss_mb": round(rss,1)}))
"""


def run_vmecjax_warm(deck: Path, aux: list[str], timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        shutil.copy(deck, wd)
        for a in aux:
            shutil.copy(DATA / a, wd)
        try:
            proc = subprocess.run([str(VMECJAX_PY), "-c", WARM_SNIPPET, deck.name],
                                  cwd=wd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        if proc.returncode != 0:
            return {"ok": False, "error": proc.stderr[-1500:]}
        line = proc.stdout.strip().splitlines()[-1]
        return {"ok": True, "converged": True, **json.loads(line)}


VMECPP_SNIPPET = r"""
import json, sys, time, resource
import vmecpp
inp = vmecpp.VmecInput.from_file(sys.argv[1])
t0 = time.time()
out = vmecpp.run(inp)
wall = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(json.dumps({"wall_s": round(wall,3), "peak_rss_mb": round(rss,1),
                  "iterations": int(out.wout.niter), "converged": True}))
"""


def run_vmecpp(deck: Path, aux: list[str], timeout: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        shutil.copy(deck, wd)
        for a in aux:
            shutil.copy(DATA / a, wd)
        try:
            proc = subprocess.run([str(VMECPP_PY), "-c", VMECPP_SNIPPET, deck.name],
                                  cwd=wd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        if proc.returncode != 0:
            return {"ok": False, "converged": False, "error": proc.stderr[-800:]}
        return {"ok": True, **json.loads(proc.stdout.strip().splitlines()[-1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "benchmarks" / "baseline.json"))
    ap.add_argument("--cases", default=None, help="comma-separated subset")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--skip", default="", help="comma-separated solvers to skip")
    args = ap.parse_args()

    cases = args.cases.split(",") if args.cases else list(CASES)
    skip = set(args.skip.split(",")) if args.skip else set()
    results: dict[str, dict] = {}

    for name in cases:
        aux = CASES.get(name, [])
        src = DATA / f"input.{name}"
        if read_deck_ns(src) >= RAMP_NS:
            variants = [("input", src)]
        else:
            ramped = Path(tempfile.mkdtemp()) / f"input.{name}"
            make_ramped_deck(src, ramped)
            variants = [(f"ns{RAMP_NS}", ramped)]
        if name in MULTIGRID_CASES:
            mg = Path(tempfile.mkdtemp()) / f"input.{name}"
            make_multigrid_deck(src, MULTIGRID_CASES[name], mg)
            variants.append(("multigrid", mg))
        for grid, deck in variants:
            key = f"{name}[{grid}]"
            row: dict[str, dict] = {"ns": read_deck_ns(deck)}
            print(f"=== {key} (ns={row['ns']}) ===", flush=True)
            for solver, fn in [("vmec2000", run_vmec2000),
                               ("vmec_jax_cold", run_vmecjax_cold),
                               ("vmec_jax_warm", run_vmecjax_warm),
                               ("vmecpp", run_vmecpp)]:
                if solver in skip:
                    continue
                if solver in ("vmec_jax_warm", "vmecpp") and name.endswith("free_bdy_lasym_small"):
                    if solver == "vmecpp":
                        row[solver] = {"ok": False, "error": "lasym unsupported"}
                        continue
                r = fn(deck, aux, args.timeout)
                row[solver] = r
                print(f"  {solver:15s} wall={r.get('wall_s', r.get('warm_s'))} ok={r.get('ok')}", flush=True)
            results[key] = row

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage timings of the raw-force block solver on the QA optimization problem.

The problem is ``examples/optimization/QA_optimization.py`` at its first
resolution group (``max_mode=2``, ``mpol=ntor=5``, 31 surfaces, block width
150). Two parts, both on the device JAX selects:

* ``stages``: the compiled pieces of :func:`vmex.core.implicit._raw_block_system`
  at the seed root, each warm-timed with its numeric arrays as arguments --
  colored assembly plus scaling, the normalized SOLVAX factorization of the
  scaled bands, one-vector, matrix and transposed applications, and the complete
  ``_refine_block_factor_core`` executable the Newton refinement, Jacobian
  and adjoint consume.
* ``workflow``: ``residual_and_jac`` and the scalar ``value_and_grad`` of the
  same objective at a fixed sequence of decision vectors, first (compiling)
  pass and a repeated pass, with the implicit solver's effort counters.

    python benchmarks/block_solver.py --output block_solver.json
    python benchmarks/block_solver.py --save-bands /tmp/vmex_qa_bands.npz

``--save-bands`` writes the scaled bands for SOLVAX's
``benchmarks/benchmark_block_factors.py``; keep that file out of Git.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import solvax  # noqa: E402

import vmex  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import implicit as im  # noqa: E402

MAX_MODE, MPOL, ASPECT = 2, 5, 6.0


def qa_problem(smoke: bool):
    inp = vmex.VmecInput.from_file(REPO / "examples" / "data" / "input.minimal_seed_nfp2")
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.05, 0.05
    mpol = 3 if smoke else MPOL
    inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    if smoke:
        inp = replace(inp, ns_array=np.asarray([9]))
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), helicity_m=1, helicity_n=0)
    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, ASPECT, 1.0), (opt.magnetic_well, 0.01, 1.0)]
    return inp, terms


def timed(fn, *args, repeats):
    """Compile time, first call and warm samples of ``jit(fn)`` on resident arrays."""
    jax.block_until_ready(args)
    t0 = time.perf_counter()
    exe = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jax.block_until_ready(exe(*args))
    first_s = time.perf_counter() - t0
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = jax.block_until_ready(exe(*args))
        samples.append(time.perf_counter() - t0)
    memory = exe.memory_analysis()
    return out, {
        "compile_seconds": compile_s, "first_seconds": first_s,
        "median_seconds": float(np.median(samples)), "samples_seconds": samples,
        "temp_bytes": None if memory is None else int(memory.temp_size_in_bytes)}


def stage_timings(inp, repeats, save_bands):
    cfg = im.make_config(inp)
    params = im.params_from_input(inp)
    state, mask = im.solve_implicit_with_aux(params, cfg)
    fields = im._active_state_fields(cfg)
    probe = int(np.ceil(np.sqrt(len(fields) * int(mask.R_cos.shape[1]))))

    def assemble(z, p, frozen, dof_mask):
        system = im._raw_block_system(p, cfg, frozen, dof_mask, fields, probe,
                                      z_star=z, factor=False)
        rs, cs = system.row_scale, system.column_scale
        ns = rs.shape[0]
        prev, nxt = jnp.maximum(jnp.arange(ns) - 1, 0), jnp.minimum(jnp.arange(ns) + 1, ns - 1)
        return (rs[:, :, None] * system.lower * cs[prev][:, None, :],
                rs[:, :, None] * system.diagonal * cs[:, None, :],
                rs[:, :, None] * system.upper * cs[nxt][:, None, :])

    args = (state, params, state, mask)
    report = {}
    bands, report["assembly_and_scaling"] = timed(assemble, *args, repeats=repeats)
    factors, report["factor"] = timed(
        lambda *b: solvax.block_thomas_factor(*b, normalized=True), *bands, repeats=repeats)
    rng = np.random.default_rng(0)
    n, m = bands[1].shape[:2]
    vector = jnp.asarray(rng.standard_normal((n, m)))
    matrix = jnp.asarray(rng.standard_normal((n, m, 8)))
    for label, rhs, transpose in (("apply_vector", vector, False),
                                  ("apply_matrix8", matrix, False),
                                  ("apply_matrix8_transpose", matrix, True)):
        solution, report[label] = timed(
            lambda f, b, t=transpose: solvax.block_thomas_solve(f, b, transpose=t),
            factors, rhs, repeats=repeats)
        if not transpose:
            report[label]["relative_residual"] = float(jnp.max(
                solvax.block_tridiag_relative_residual(*bands, solution, rhs)))
    _, report["refine_block_factor_core"] = timed(
        lambda *a: im._refine_block_factor_core(*a, cfg), *args, repeats=repeats)
    if save_bands is not None:
        np.savez(save_bands, lower=bands[0], diag=bands[1], upper=bands[2])
    return {"n_blocks": int(n), "block_size": int(m), "active_fields": list(fields),
            **report}


def workflow_timings(inp, terms, n_points):
    problem = opt.VmecProblem.from_tuples(inp, terms, max_mode=MAX_MODE)
    scalar = opt.VmecProblem.from_loss(
        inp, lambda s, r: opt.aspect_ratio(s, r) ** 2 + 1e3 * opt.magnetic_well(s, r) ** 2,
        max_mode=MAX_MODE)
    rng = np.random.default_rng(1)
    steps = [np.zeros_like(problem.x0)] + [
        0.01 * problem.scales * rng.standard_normal(problem.x0.shape)
        for _ in range(n_points - 1)]
    out = {"n_dofs": int(problem.x0.size), "n_points": n_points}
    for label, fn, x0, prob in (
            ("residual_and_jac", problem.residual_and_jac, problem.x0, problem),
            ("value_and_grad", scalar.value_and_grad, scalar.x0, scalar)):
        before = dict(im._SOLVE_STATS.get(prob.metadata["config"], {}))
        passes = []
        for _ in range(2):
            times = []
            for step in steps:
                t0 = time.perf_counter()
                value = fn(x0 + step)
                jax.block_until_ready(value)
                times.append(time.perf_counter() - t0)
            passes.append(times)
        after = im._SOLVE_STATS.get(prob.metadata["config"], {})
        out[label] = {"first_pass_seconds": passes[0], "second_pass_seconds": passes[1],
                      "checksum": float(np.sum(np.abs(np.asarray(value[1])))),
                      "counters": {k: None if v is None else float(v) - float(before.get(k) or 0)
                                   for k, v in after.items()}}
    return out


def metadata():
    def sha(path):
        try:
            return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                           text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"vmex_sha": sha(REPO), "solvax_version": solvax.__version__,
            "solvax_sha": sha(Path(solvax.__file__).parent),
            "vmex_version": vmex.__version__, "jax_version": jax.__version__,
            "device": str(jax.devices()[0]), "x64": bool(jax.config.read("jax_enable_x64")),
            "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "load_average": list(os.getloadavg()),
            "xla_flags": os.environ.get("XLA_FLAGS", "")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--skip-workflow", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="mpol=3, 9 surfaces")
    parser.add_argument("--save-bands", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inp, terms = qa_problem(args.smoke)
    result = {"benchmark": "vmex_block_solver", "case": "QA_optimization max_mode=2",
              **metadata(), "stages": stage_timings(inp, args.repeats, args.save_bands)}
    if not args.skip_workflow:
        result["workflow"] = workflow_timings(inp, terms, args.points)
    rendered = json.dumps(result, indent=1) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

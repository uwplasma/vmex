#!/usr/bin/env python
"""Plan B3: the exact block adjoint against the production Krylov adjoint.

At a deck's descent state, for the QA and QI objectives of ``benchmarks/optimization.py``,
the script solves the implicit adjoint two ways: through the transposed raw block
factorization (``_adjoint_block_core``) and with the preconditioned GCROT adjoint
(``_adjoint_gcrot_core``). For each it records the independent adjoint residual, the cost
and the eight max_mode = 1 boundary-gradient entries, plus the objective value and
``mu^T F_raw``, the first-order value error of treating the anchor as a root.

With ``--dense`` (the seed deck is small enough) it also solves both formulations
directly on range(P). That separates the Krylov error of the production adjoint from
the formulation difference between the raw and the preconditioned residual, which
coincide only at an exact root. Seconds are diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

import vmex
from vmex.core import implicit as imp
from vmex.core import optimize as core_optimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
from _provenance import assert_repo_vmex, git_state  # noqa: E402
from newton_finish_arms import build_case, compact_json  # noqa: E402
from optimization import terms  # noqa: E402


def deck_report(case: str, dense: bool) -> dict:
    path, cfg = build_case(case)
    params = imp.params_from_input(cfg.inp, device=cfg.device)
    mask = jax.tree.map(jnp.asarray, imp._MASK_CACHE[imp._mask_cache_key(cfg)])
    P, edge = imp._dof_projector(cfg, mask), imp._edge_mask(cfg)
    state = imp._LAST_SOLVE[cfg][1].state
    z = P(state)
    flat_z, unravel = ravel_pytree(z)
    ntor = cfg.resolution.ntor
    boundary = [(field, m, n) for field in ("rbc", "zbs") for m, n in ((0, 1), (1, -1), (1, 0), (1, 1))]
    residual = {name: imp.residual_fn(cfg, state, mask, formulation=name) for name in ("raw", "preconditioned")}
    f_raw = residual["raw"](z, params)
    report = {"resolution": {"mpol": cfg.resolution.mpol, "ntor": cfg.resolution.ntor, "ns": cfg.resolution.ns},
              "adjoint_tol": cfg.adjoint_tol, "anchor": "descent state",
              "residual_norms": {name: float(imp._tree_norm(f(z, params))) for name, f in residual.items()}}
    if dense:
        projector = np.asarray(jax.vmap(lambda e: ravel_pytree(P(unravel(e)))[0])(jnp.eye(flat_z.size)))
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (projector + projector.T))
        Q = eigenvectors[:, eigenvalues > 0.5]  # P is a symmetric idempotent: its range
        restricted = {name: Q.T @ np.asarray(jax.jit(jax.jacfwd(
            lambda v, f=f: ravel_pytree(f(unravel(v), params))[0]))(flat_z)) @ Q for name, f in residual.items()}
        report["dense"] = {"rank": int(Q.shape[1]),
                           "condition": {name: float(np.linalg.cond(A)) for name, A in restricted.items()}}

    def gradient(multiplier, formulation, direct):
        pullback = jax.vjp(lambda prm: residual[formulation](z, prm), params)[1]
        total = jax.tree.map(jnp.add, pullback(jax.tree.map(jnp.negative, multiplier))[0], direct)
        return np.asarray([float(getattr(total, field)[ntor + n, m]) for field, m, n in boundary])

    def relative(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    for name in ("qa", "qi"):
        traced = [(core_optimize._traceable_term(f), float(t), jnp.asarray(core_optimize._least_squares_weight(w, "cost")))
                  for f, t, w in terms(name)]

        def half_square(x, prm, traced=traced):
            runtime = imp.runtime_from_params(prm, cfg)
            rows = jnp.concatenate([jnp.atleast_1d(w * (jnp.asarray(f(x, runtime)) - t)).ravel() for f, t, w in traced])
            return 0.5 * jnp.vdot(rows, rows)

        b = P(jax.grad(lambda x: half_square(x, params))(state))
        direct = jax.grad(lambda prm: half_square(
            imp._assemble(z, imp.runtime_from_params(prm, cfg), state, P, edge), prm))(params)
        started = time.perf_counter()
        mu, block = imp._adjoint_block_core(params, z, state, mask, b, cfg)
        block_seconds = time.perf_counter() - started
        started = time.perf_counter()
        lam, krylov = imp._adjoint_gcrot_core(params, z, state, mask, b, cfg)
        krylov_seconds = time.perf_counter() - started
        gradients = {"block_raw": gradient(mu, "raw", direct), "gcrot_preconditioned": gradient(lam, "preconditioned", direct)}
        entry = {"value": float(half_square(state, params)),
                 "mu_dot_f_raw": float(sum(jnp.vdot(a, c) for a, c in zip(jax.tree.leaves(mu), jax.tree.leaves(f_raw)))),
                 "block": {"residual_norm": float(block.residual_norm), "tolerance": float(block.tolerance),
                           "accepted": bool(block.converged), "seconds_with_compile": block_seconds},
                 "gcrot": {"iterations": int(krylov.iterations), "residual_norm": float(krylov.residual_norm),
                           "tolerance": float(krylov.tolerance), "accepted": bool(krylov.converged),
                           "seconds_with_compile": krylov_seconds},
                 "gcrot_vs_block_raw": relative(gradients["gcrot_preconditioned"], gradients["block_raw"])}
        if dense:
            rhs = Q.T @ np.asarray(ravel_pytree(b)[0])
            for formulation, A in restricted.items():
                y = np.linalg.solve(A.T, rhs)
                entry[f"dense_{formulation}_adjoint_residual"] = float(np.linalg.norm(A.T @ y - rhs) / np.linalg.norm(rhs))
                gradients[f"dense_{formulation}"] = gradient(unravel(jnp.asarray(Q @ y)), formulation, direct)
            entry.update(
                krylov_error=relative(gradients["gcrot_preconditioned"], gradients["dense_preconditioned"]),
                formulation_difference=relative(gradients["dense_raw"], gradients["dense_preconditioned"]),
                block_vs_dense_raw=relative(gradients["block_raw"], gradients["dense_raw"]))
        entry["gradients"] = {key: value.tolist() for key, value in gradients.items()}
        report[name] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=("seed", "qa_lowres"), default=["seed"])
    parser.add_argument("--dense", action="store_true", help="dense direct adjoints on the seed deck")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_start = os.getloadavg()
    rows = {case: deck_report(case, args.dense and case == "seed") for case in args.cases}
    record = {
        "_provenance": {
            **git_state(REPO), "vmex_module": assert_repo_vmex(vmex.__file__, REPO), "platform": platform.platform(),
            "versions": {"python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__,
                         "vmex": vmex.__version__},
            "threads": os.environ.get("OMP_NUM_THREADS"), "compilation_cache": os.environ.get("VMEX_COMPILATION_CACHE"),
            "load_average": {"start": load_start, "end": os.getloadavg()},
            "regenerate": "PYTHONPATH=. JAX_ENABLE_X64=1 OMP_NUM_THREADS=4 VMEX_COMPILATION_CACHE=disabled "
                          "python benchmarks/adjoint_formulation.py --cases seed qa_lowres --dense --output <record>"},
        "rows": rows}
    text = compact_json(json.loads(json.dumps(record, default=float)))
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()

"""Assemble and factor the exact stationarity KKT matrix at a certified root (R7.6).

H = A.T A + sum_i r_i Hess(r_i) has the span-local pattern of the Gauss-Newton
normal matrix (both couple coefficients that share quadrature points), so it
is recovered exactly from colored Hessian-vector products.  The exact K is
factored once; tangent/adjoint systems are then solved directly.  Shift-invert
Lanczos on the same factor reports the eigenvalues of K nearest zero and the
gauge/feasibility character of their eigenvectors.
Read-only diagnostic: writes one JSON record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse import linalg as sparse_linalg  # noqa: E402
from solvax.compression import column_groups, matrix_from_products  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polish_recovery_sparse as prs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eigenvalues", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    base, _, plan, layout, gauge, scale, c0, _ = prs._load_original_problem(args.state)
    force, _ = prs._linear_problem(base, plan, layout, gauge, scale, c0)
    C = sparse.csr_matrix(prs.native_tangential_gauge_matrix(base, layout, gauge, scale))
    n, q = C.shape[1], C.shape[0]
    r0 = np.asarray(force(c0))
    normal, _, _ = prs._local_normal_system(force, c0, r0, base, plan, layout, gauge, scale)
    gradient = jax.jit(jax.grad(lambda c: 0.5 * jnp.vdot(force(c), force(c))))
    hvp = jax.jit(lambda v: jax.jvp(gradient, (c0,), (v,))[1])
    pattern = sparse.csr_matrix((np.ones(normal.nnz), normal.indices, normal.indptr), shape=normal.shape)
    groups = column_groups(pattern)
    assembly_started = time.perf_counter()
    hessian = matrix_from_products(lambda v: hvp(jnp.asarray(v)), pattern, groups=groups)
    assembly_seconds = time.perf_counter() - assembly_started
    rng = np.random.default_rng(76)
    probe = rng.standard_normal(n)
    hessian_error = float(np.linalg.norm(hessian @ probe - np.asarray(hvp(jnp.asarray(probe)))) / np.linalg.norm(np.asarray(hvp(jnp.asarray(probe)))))
    symmetry = float(sparse.linalg.norm(hessian - hessian.T) / sparse.linalg.norm(hessian))
    residual_term = hessian - normal
    kkt = sparse.bmat([[hessian, C.T], [C, None]], format="csc")
    factor_started = time.perf_counter()
    lu = sparse_linalg.splu(kkt)
    factor_seconds = time.perf_counter() - factor_started
    record = {
        "schema": "vmex-r7-exact-kkt-probe/1",
        "state_sha256": hashlib.sha256(args.state.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coordinates": n,
        "constraints": q,
        "colors": len(groups),
        "hessian_nnz": int(hessian.nnz),
        "hessian_assembly_seconds": assembly_seconds,
        "hessian_product_relative_error": hessian_error,
        "hessian_symmetry_relative": symmetry,
        "residual_term_relative_frobenius": float(sparse.linalg.norm(residual_term) / sparse.linalg.norm(hessian)),
        "factor_seconds": factor_seconds,
        "factor_fill_nnz": int(lu.L.nnz + lu.U.nnz),
        "factor_pivot_min_abs": float(np.min(np.abs(lu.U.diagonal()))),
        "factor_pivot_max_abs": float(np.max(np.abs(lu.U.diagonal()))),
    }
    # Direct solves on random and structured right-hand sides.
    for label, rhs in (
        ("random", rng.standard_normal(n + q)),
        ("gradient_like", np.concatenate((np.asarray(gradient(c0)), np.zeros(q)))),
    ):
        z = lu.solve(rhs)
        record.setdefault("direct_solves", {})[label] = float(np.linalg.norm(kkt @ z - rhs) / np.linalg.norm(rhs))
    # Eigenvalues of K nearest zero (shift-invert on the exact factor).
    operator = sparse_linalg.LinearOperator(kkt.shape, matvec=lu.solve, dtype=float)
    values, vectors = sparse_linalg.eigs(operator, k=args.eigenvalues, which="LM", tol=1e-10)
    order = np.argsort(-np.abs(values))
    near = []
    for index in order:
        mu = values[index]
        vector = np.real(vectors[:, index])
        vector /= np.linalg.norm(vector)
        primal = vector[:n]
        near.append(
            {
                "eigenvalue": float(np.real(1.0 / mu)),
                "primal_fraction": float(np.linalg.norm(primal)),
                "constraint_violation": float(np.linalg.norm(C @ primal) / max(np.linalg.norm(primal), 1e-300)),
                "true_residual": float(np.linalg.norm(kkt @ vector - vector / mu) / np.linalg.norm(vector / mu)),
            }
        )
    record["eigenvalues_nearest_zero"] = near
    record["operator_scale"] = float(sparse.linalg.norm(kkt))
    record["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()

"""Exact implicit derivative of a certified native root (R7.6).

Parameter: a pressure amplitude, p -> (1 + a) p, which leaves the gauge rows,
coordinate metric, and boundary unchanged.  The root is the fixed-chart
stationarity system G(c, nu; a) = [g + C.T nu, C c] = 0 with the exact
Hessian H = A.T A + sum r_i Hess(r_i).  For outputs Q(c):

  tangent:  K y = -G_a,        dQ/da = Q_c . y_c
  adjoint:  K.T lam = [Q_c,0], dQ/da = -lam . G_a

K is symmetric, so both use the same preconditioned GMRES (sparse
Gauss-Newton KKT factor as preconditioner, exact H action as operator).  The
finite-difference control re-solves the root at a = +-h by exact Newton on the
identical chart and compares central differences over several h.
Read-only diagnostic: writes one JSON record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse import linalg as sparse_linalg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polish_recovery_sparse as prs  # noqa: E402

from vmex.core.polish import apply_high_order_correction  # noqa: E402
from vmex.core.strong_force import _RZL  # noqa: E402

OUTPUT_POINTS = {
    "R_axis": (0.0, 0.0, "R"),
    "R_outboard_mid": (0.5, 0.0, "R"),
    "Z_top_mid": (0.5, 0.5 * np.pi, "Z"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restart", type=int, default=60)
    parser.add_argument(
        "--balance-constraints",
        action="store_true",
        help="scale gauge rows by sqrt(mean diag H): same feasible set and primal solution",
    )
    parser.add_argument("--steps", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4, 3e-5])
    args = parser.parse_args()
    started = time.perf_counter()
    base, _, plan, layout, gauge, scale, c0, _ = prs._load_original_problem(args.state)
    if plan.spline_value is None:
        raise ValueError("derivative qualification requires the accurate evaluation mode")
    C = sparse.csr_matrix(prs.native_tangential_gauge_matrix(base, layout, gauge, scale))
    n, q = C.shape[1], C.shape[0]

    def residual(c, a):
        state = replace(base, pressure=base.pressure * (1.0 + a))
        return prs.native_physical_force_residual(c, state, layout, gauge, scale, prs.FORCE_SCALE, prs.VOLUME_SCALE)

    gradient = jax.jit(jax.grad(lambda c, a: 0.5 * jnp.vdot(residual(c, a), residual(c, a))))
    hessian = jax.jit(lambda c, a, v: jax.jvp(lambda x: gradient(x, a), (c,), (v,))[1])
    mixed = jax.jit(lambda c, a: jax.jvp(lambda s: gradient(c, s), (a,), (1.0,))[1])

    def force_at(a):
        return lambda c: residual(c, a)

    r0 = np.asarray(residual(c0, 0.0))
    normal, _, _ = prs._local_normal_system(force_at(0.0), c0, r0, base, plan, layout, gauge, scale)
    frobenius = float(np.sqrt(normal.diagonal().sum()))
    # Row scaling of C changes only the multipliers (nu' = nu / s); the
    # stationarity metric keeps using the original projector below.
    balance = float(np.sqrt(normal.diagonal().mean())) if args.balance_constraints else 1.0
    C_eta = C
    C = C * balance
    kkt_norm = float(np.sqrt(sparse.linalg.norm(normal) ** 2 + 2.0 * sparse.linalg.norm(C) ** 2))
    factor = sparse_linalg.splu(sparse.bmat([[normal, C.T], [C, None]], format="csc"))

    def kkt_solve(c, a, rhs):
        def apply(z):
            top = np.asarray(hessian(c, a, jnp.asarray(z[:n]))) + C.T @ z[n:]
            return np.concatenate((top, C @ z[:n]))

        operator = sparse_linalg.LinearOperator((n + q, n + q), matvec=apply, dtype=float)
        preconditioner = sparse_linalg.LinearOperator((n + q, n + q), matvec=factor.solve, dtype=float)
        count = [0]
        z, info = sparse_linalg.gmres(
            operator, rhs, M=preconditioner, rtol=1e-12, atol=0.0, restart=args.restart, maxiter=20,
            callback=lambda _: count.__setitem__(0, count[0] + 1), callback_type="pr_norm",
        )
        defect = float(np.linalg.norm(apply(z) - rhs))
        true = defect / max(np.linalg.norm(rhs), 1e-300)
        # Normwise backward error (plan section 7.2) with ||K||_F estimated
        # from the assembled Gauss-Newton normal matrix and C.
        backward = defect / (kkt_norm * float(np.linalg.norm(z)) + float(np.linalg.norm(rhs)))
        return z, {
            "gmres_info": int(info),
            "iterations": count[0],
            "true_relative_residual": true,
            "normwise_backward_error": backward,
            "solution_norm": float(np.linalg.norm(z)),
        }

    def eta(c, a):
        g = np.asarray(gradient(c, a))
        pg, _ = prs._sparse_projected_gradient(g, C_eta)
        return float(np.linalg.norm(pg)) / (frobenius * float(np.linalg.norm(np.asarray(residual(c, a)))))

    def resolve(a):
        c = np.asarray(c0, dtype=float)
        history = []
        for _ in range(8):
            value = eta(jnp.asarray(c), a)
            history.append(value)
            if value < 1e-11:
                break
            g = np.asarray(gradient(jnp.asarray(c), a))
            z, _ = kkt_solve(jnp.asarray(c), a, -np.concatenate((g, C @ c)))
            c = c + z[:n]
        return c, history

    def outputs(c):
        state = apply_high_order_correction(base, layout.unpack(scale * c))
        values = {}
        for name, (rho, theta, which) in OUTPUT_POINTS.items():
            R, Z, _ = _RZL(state, jnp.asarray([rho, theta, 0.0]))
            values[name] = R if which == "R" else Z
        return values

    output_grad = {name: jax.grad(lambda c, name=name: outputs(c)[name]) for name in OUTPUT_POINTS}
    root_eta = eta(c0, 0.0)
    Ga = np.concatenate((np.asarray(mixed(c0, 0.0)), np.zeros(q)))
    tangent_started = time.perf_counter()
    y, tangent_solve = kkt_solve(c0, 0.0, -Ga)
    tangent_seconds = time.perf_counter() - tangent_started
    record = {
        "schema": "vmex-r7-implicit-derivative/1",
        "state": str(args.state),
        "state_sha256": hashlib.sha256(args.state.read_bytes()).hexdigest(),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameter": "pressure amplitude a: p -> (1+a) p",
        "root_eta": root_eta,
        "constraint_balance": balance,
        "tangent_solve": {**tangent_solve, "seconds": tangent_seconds},
        "outputs": {},
    }
    resolved = {}
    for h in args.steps:
        plus, plus_history = resolve(h)
        minus, minus_history = resolve(-h)
        resolved[h] = (plus, minus)
        record.setdefault("resolves", {})[str(h)] = {"plus_eta": plus_history, "minus_eta": minus_history}
    for name in OUTPUT_POINTS:
        qc = np.asarray(output_grad[name](c0))
        adjoint_started = time.perf_counter()
        lam, adjoint_solve = kkt_solve(c0, 0.0, np.concatenate((qc, np.zeros(q))))
        tangent_value = float(qc @ y[:n])
        adjoint_value = float(-lam @ Ga)
        fd = {
            str(h): float((outputs(jnp.asarray(p))[name] - outputs(jnp.asarray(m))[name]) / (2 * h))
            for h, (p, m) in resolved.items()
        }
        record["outputs"][name] = {
            "value": float(outputs(c0)[name]),
            "tangent": tangent_value,
            "adjoint": adjoint_value,
            "tangent_adjoint_relative_difference": abs(tangent_value - adjoint_value) / max(abs(tangent_value), 1e-300),
            "central_differences": fd,
            "fd_relative_error": {k: abs(v - tangent_value) / max(abs(tangent_value), 1e-300) for k, v in fd.items()},
            "adjoint_solve": {**adjoint_solve, "seconds": time.perf_counter() - adjoint_started},
        }
    record["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()

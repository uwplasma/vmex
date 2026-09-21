#!/usr/bin/env python
"""Qualify exact-operator reuse of refinement factors in optimization."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
import platform
import resource
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

import vmex
from vmex import optimize as opt
from vmex.core import implicit as imp
from vmex.core.input import VmecInput

from _provenance import assert_repo_vmex, file_sha256, git_state

REPO = Path(__file__).resolve().parents[1]


def build_problem(case: str, *, response_rtol=None):
    """Build a shipped, deterministic optimization problem."""
    if case == "qi":
        from vmex.core.qi import ConstructedQIResidual

        inp = replace(
            VmecInput.from_file(REPO / "examples/data/input.QI_nfp2_initial"),
            ns_array=np.asarray([31]),
            ftol_array=np.asarray([1.0e-12]),
            niter_array=np.asarray([5500]),
        )
        qi = ConstructedQIResidual(
            np.linspace(0.1, 1.0, 6), mboz=12, nboz=12, nphi=61,
            nalpha=18, n_bounce=21,
        )

        def iota_floor(state, runtime):
            return jnp.maximum(0.51 - opt.min_abs_iota(state, runtime), 0.0)

        def mirror_excess(state, runtime):
            return jnp.maximum(opt.mirror_ratio(state, runtime) - 0.2079, 0.0)

        def elongation_excess(state, runtime):
            return jnp.maximum(opt.max_elongation(state, runtime) - 8.0, 0.0)

        terms = [
            (qi, 0.0, 10.0),
            (opt.aspect_ratio, 5.0, 0.005),
            (iota_floor, 0.0, 10.0),
            (mirror_excess, 0.0, 1000.0),
            (elongation_excess, 0.0, 10.0),
        ]
        max_mode = 2
    else:
        inp = VmecInput.from_file(REPO / "examples/data/input.minimal_seed_nfp2")

        def iota_floor(state, runtime):
            return jnp.maximum(0.33 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

        terms = [
            (opt.aspect_ratio, 6.0, 0.01),
            (iota_floor, 0.0, 10.0),
        ]
        max_mode = 1

    inp = replace(inp, delt=0.5).change_resolution(
        mpol=5, ntor=5, ntheta=16, nzeta=14,
    )
    return opt.VmecProblem.from_tuples(
        inp, terms, max_mode=max_mode, use_ess=True,
        forward_max_iterations=5500, jacobian_batch_size=1,
        jacobian_adjoint_tol=response_rtol,
    )



def digest(value):
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else 1024 * value)


def tangents(problem, x):
    base = imp.params_from_input(problem.input_from_x(x))
    rows = []
    for j in range(x.size):
        dx = np.zeros_like(x); dx[j] = 1.0
        shifted = imp.params_from_input(problem.input_from_x(x + dx))
        rows.append(jax.tree.map(jnp.subtract, shifted, base))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *rows)


def candidate_core(problem, rtol):
    cfg = problem.metadata["config"]
    rows_from_state = problem.metadata["jax_residual_from_state"]

    @jax.jit
    def evaluate(params, state, mask, dp, factors):
        project = imp._dof_projector(cfg, mask)
        edge = imp._edge_mask(cfg)
        z = project(state)
        raw = imp.residual_fn(cfg, state, mask, formulation="raw")
        block, row_scale, column_scale = factors

        def precondition(value):
            return imp._block_inverse_apply(
                block, lambda tree: imp._pack_active(cfg, tree),
                lambda matrix: imp._unpack_active(cfg, matrix), project,
                row_scale, column_scale, value)

        def column(direction):
            rhs = jax.tree.map(jnp.negative, jax.jvp(
                lambda prm: raw(z, prm), (params,), (direction,))[1])
            response, solve = imp._adjoint_solve_gcrot(
                lambda value: jax.jvp(lambda q: raw(q, params),
                                      (z,), (value,))[1],
                rhs, cfg, precond=precondition, rtol=rtol,
                max_restarts=cfg.jacobian_adjoint_maxiter, enforce=False)
            applied = jax.jvp(lambda q: raw(q, params),
                              (z,), (response,))[1]
            defect = imp._tree_norm(jax.tree.map(jnp.subtract, rhs, applied))
            rhs_norm = imp._tree_norm(rhs)

            def objective(q, prm):
                runtime = imp.runtime_from_params(prm, cfg)
                physical = imp._assemble(q, runtime, state, project, edge)
                return rows_from_state(physical, runtime)

            objective_column = jax.jvp(
                objective, (z, params), (project(response), direction))[1]
            return objective_column, defect, rhs_norm, solve.iterations

        columns, defects, norms, iterations = jax.lax.map(column, dp)
        return columns.T, defects, norms, iterations

    return evaluate


def compare_fresh(problem, x, candidate, fresh_jacobian):
    fresh = fresh_jacobian(x, "audit")
    residual = np.asarray(problem.residual(x), dtype=np.float64)
    difference = candidate - fresh
    gradient_difference = candidate.T @ residual - fresh.T @ residual
    tiny = np.finfo(float).tiny
    return {
        "jacobian_relative_difference": float(
            np.linalg.norm(difference) / max(np.linalg.norm(fresh), tiny)),
        "jacobian_max_absolute_difference": float(np.max(np.abs(difference))),
        "gradient_relative_difference": float(
            np.linalg.norm(gradient_difference)
            / max(np.linalg.norm(fresh.T @ residual), tiny)),
    }


def seed_gate(term_rows, aspect_tol, iota_tol):
    aspect = term_rows[0]["max_abs"] / np.sqrt(0.01)
    iota = term_rows[1]["max_abs"] / np.sqrt(10.0)
    return {
        "scope": "prospective criterion for this eight-dof seed probe only",
        "thresholds": {"aspect_error": aspect_tol, "iota_violation": iota_tol},
        "observed": {"aspect_error": aspect, "iota_violation": iota},
        "passed": bool(aspect <= aspect_tol and iota <= iota_tol),
    }


def run_arm(case, arm, max_nfev, rtol, audit, initial_x):
    started = time.perf_counter(); rss_start = peak_rss_bytes()
    problem = build_problem(case, response_rtol=rtol)
    cfg = problem.metadata["config"]
    evaluate = candidate_core(problem, rtol)
    mask = imp._fixed_boundary_dof_mask(cfg)
    counts = {
        "reuse": 0, "fallback": 0, "ineligible": 0,
        "refinement_factorizations": 0, "certified_columns": 0,
        "uncertified_columns": 0,
    }
    history = {"residual": [], "jacobian": []}
    factor_cache = {}
    jacobians = {}
    fresh_requests = {"arm": 0, "fallback": 0, "audit": 0}
    original = imp._refine_block_factors

    def observe(config, params, frozen, dof_mask, z):
        factors = original(config, params, frozen, dof_mask, z)
        factor_cache[config] = (imp._params_key(params), factors)
        counts["refinement_factorizations"] += 1
        return factors

    def residual(x):
        value = np.asarray(problem.residual(x), dtype=np.float64)
        history["residual"].append((digest(x), float(value @ value)))
        return value

    def fresh(x, reason):
        fresh_requests[reason] += 1
        return np.asarray(problem.residual_jac(x), dtype=np.float64)

    def jacobian(x):
        fallback = arm == "fresh"
        defects = tolerances = iterations = []
        if not fallback:
            value_at_x = np.asarray(problem.residual(x), dtype=np.float64)
            certificate = problem.metadata["primal_certificate"](x)
            eligible = (np.all(np.isfinite(value_at_x))
                        and certificate is not None
                        and certificate["derivative_admitted"])
            if eligible:
                state, _, status = jax.device_get(
                    problem.metadata["jax_state_runtime_status"](jnp.asarray(x)))
                certificate = problem.metadata["primal_certificate"](x)
                params = imp.params_from_input(problem.input_from_x(x))
                hit = factor_cache.get(cfg)
                eligible = (int(status) == 0 and certificate is not None
                            and certificate["derivative_admitted"]
                            and hit is not None and hit[0] == imp._params_key(params))
            if eligible:
                value, defect, rhs_norm, its = jax.device_get(evaluate(
                    params, state, mask, tangents(problem, np.asarray(x)), hit[1]))
                tolerance = imp._ADJOINT_RESIDUAL_SLACK * rtol * rhs_norm
                fallback = not bool(np.all(np.isfinite(value))
                                    and np.all(defect <= tolerance))
                counts["certified_columns"] += int(np.sum(defect <= tolerance))
                counts["uncertified_columns"] += int(np.sum(defect > tolerance))
                defects = np.asarray(defect, float).tolist()
                tolerances = np.asarray(tolerance, float).tolist()
                iterations = np.asarray(its, int).tolist()
            else:
                counts["ineligible"] += 1
                fallback = True
        if fallback:
            counts["fallback"] += arm == "reuse"
            value = fresh(x, "arm" if arm == "fresh" else "fallback")
        else:
            counts["reuse"] += 1
            value = np.asarray(value, dtype=np.float64)
        singular = np.linalg.svd(value, compute_uv=False)
        row = {
            "x": digest(x), "fallback": bool(fallback),
            "rank": int(np.linalg.matrix_rank(value)),
            "condition": float(singular[0] / singular[-1]) if singular[-1] else None,
            "defects": defects, "tolerances": tolerances,
            "iterations": iterations,
        }
        if audit and not fallback:
            row.update(compare_fresh(problem, x, value, fresh))
        history["jacobian"].append(row)
        jacobians[digest(x)] = value
        return value

    imp._refine_block_factors = observe
    stats_before = dict(imp._SOLVE_STATS.get(cfg) or {})
    try:
        result = scipy.optimize.least_squares(
            residual, problem.x0 if initial_x is None else initial_x,
            jac=jacobian, x_scale=problem.scales, max_nfev=max_nfev)
    finally:
        imp._refine_block_factors = original
    stats_after = dict(imp._SOLVE_STATS.get(cfg) or {})
    # This production counter records completed block-response summaries. On
    # the pinned block lane each evaluation builds one raw block system, but
    # it is not generic factor-kernel instrumentation.
    counts["fresh_response_evaluations_including_audit"] = (
        stats_after.get("jacobians", 0) - stats_before.get("jacobians", 0))
    counts["fresh_jacobian_requests"] = fresh_requests
    term_rows = []
    for name, first, last in problem.metadata["term_slices"]:
        values = np.asarray(result.fun[first:last], dtype=np.float64)
        term_rows.append({"name": name, "norm": float(np.linalg.norm(values)),
                          "max_abs": float(np.max(np.abs(values)))})
    final_jacobian = jacobians.get(digest(result.x))
    checkpoint = {"x": np.asarray(result.x), "residual": np.asarray(result.fun)}
    if final_jacobian is not None:
        checkpoint.update(
            singular_values=np.linalg.svd(final_jacobian, compute_uv=False),
            gradient=final_jacobian.T @ result.fun,
        )
    state, _, state_status = jax.device_get(
        problem.metadata["jax_state_runtime_status"](jnp.asarray(result.x)))
    params = imp.params_from_input(problem.input_from_x(result.x))
    checkpoint["state_status"] = np.asarray(state_status)
    checkpoint.update({
        f"state_{name}": np.asarray(getattr(state, name))
        for name in imp._STATE_FIELDS
    })
    checkpoint.update({
        f"input_{name}": np.asarray(getattr(params, name))
        for name in params.__dataclass_fields__
    })
    return {
        "arm": arm, "rtol": rtol, "max_nfev": max_nfev,
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss_bytes(), "rss_start_bytes": rss_start,
        "counts": counts,
        "result": {
            "status": int(result.status),
            "terminated_successfully": bool(result.status > 0),
            "message": str(result.message), "nfev": int(result.nfev),
            "njev": int(result.njev), "cost": float(result.cost),
            "optimality": float(result.optimality), "x": digest(result.x),
            "residual": digest(result.fun), "terms": term_rows,
        },
        "history": history,
    }, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("seed", "qi"), default="seed")
    parser.add_argument("--arm", choices=("fresh", "reuse"), required=True)
    parser.add_argument("--max-nfev", type=int, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--audit-fresh", action="store_true")
    parser.add_argument("--state-input", type=Path)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--aspect-error-tol", type=float)
    parser.add_argument("--iota-violation-tol", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paired_gate = ((args.aspect_error_tol is None)
                   == (args.iota_violation_tol is None))
    thresholds = (args.aspect_error_tol, args.iota_violation_tol)
    if (args.max_nfev < 1 or not np.isfinite(args.rtol) or args.rtol <= 0
            or not paired_gate
            or (args.aspect_error_tol is not None
                and (args.case != "seed"
                     or not all(np.isfinite(value) and value >= 0
                                for value in thresholds)))):
        parser.error("positive run controls and paired seed thresholds are required")
    assert_repo_vmex(vmex.__file__, REPO)
    initial_x = None
    if args.state_input:
        with np.load(args.state_input) as checkpoint:
            initial_x = np.asarray(checkpoint["x"], dtype=np.float64)
    load_start = os.getloadavg()
    arm, checkpoint = run_arm(
        args.case, args.arm, args.max_nfev, args.rtol,
        args.audit_fresh, initial_x)
    if args.aspect_error_tol is not None:
        arm["accuracy_gate"] = seed_gate(
            arm["result"]["terms"], args.aspect_error_tol,
            args.iota_violation_tol)
    command = " ".join(filter(None, (
        "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
        "VMEX_COMPILATION_CACHE=disabled PYTHONPATH=. python",
        "benchmarks/refinement_factor_reuse.py", f"--case {args.case}",
        f"--arm {args.arm} --max-nfev {args.max_nfev} --rtol {args.rtol}",
        "--audit-fresh" if args.audit_fresh else "",
        "--state-input <state.npz>" if args.state_input else "",
        "--state-output <state.npz>" if args.state_output else "",
        (f"--aspect-error-tol {args.aspect_error_tol} "
         f"--iota-violation-tol {args.iota_violation_tol}"
         if args.aspect_error_tol is not None else ""),
        "--output <output.json>",
    )))
    report = {
        "schema": 1, "case": args.case,
        "command": command,
        "platform": platform.platform(), "jax": jax.__version__,
        "vmex": vmex.__version__, "load_start": load_start,
        "input_sha256": file_sha256(REPO / "examples/data" / (
            "input.QI_nfp2_initial" if args.case == "qi"
            else "input.minimal_seed_nfp2")),
        **git_state(REPO), "optimization": arm,
    }
    if args.state_input:
        report["state_input_sha256"] = file_sha256(args.state_input)
    if args.state_output:
        with args.state_output.open("wb") as stream:
            np.savez_compressed(stream, **checkpoint)
        report["state_output_sha256"] = file_sha256(args.state_output)
    report["load_end"] = os.getloadavg()
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="") if args.output is None else args.output.write_text(text)


if __name__ == "__main__":
    main()

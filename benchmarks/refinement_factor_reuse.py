#!/usr/bin/env python
"""Probe guarded reuse of Newton block factors across optimizer trials.

The experiment records accepted-point/trial pairs from a bounded SciPy
least-squares run. At each trial it compares freshly built block factors with
factors from the preceding accepted point. Both are used only as right
preconditioners for GMRES on the exact current raw-force Jacobian. A stale
preconditioner is admitted only when its first solve reaches VMEX's existing
linear forcing tolerance; otherwise the reported guarded path rebuilds fresh
factors and replays the step.

The JSON report is a reproducible decision record, not a blanket speed claim:
it includes source provenance, load, exact residuals, iteration counts, and the
failed-probe cost. Run with fixed thread settings when comparing timings.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import functools
import json
import os
import platform
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

import vmex
from vmex import optimize as opt
from vmex.core import implicit as imp
from vmex.core.device import commit_to_single_device
from vmex.core.input import VmecInput

from _provenance import assert_repo_vmex, file_sha256, git_state

REPO = Path(__file__).resolve().parents[1]


def build_problem(case: str):
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
    )


def optimizer_pairs(problem, max_nfev: int):
    """Record accepted-point/trial pairs from one bounded optimizer run."""
    calls: list[tuple[str, np.ndarray]] = []

    def residual(x):
        calls.append(("residual", np.asarray(x).copy()))
        return problem.residual(x)

    def jacobian(x):
        calls.append(("jacobian", np.asarray(x).copy()))
        return problem.residual_jac(x)

    scipy.optimize.least_squares(
        residual, problem.x0, jac=jacobian, x_scale=problem.scales,
        max_nfev=max_nfev,
    )
    accepted = {x.tobytes() for kind, x in calls if kind == "jacobian"}
    return [
        (x, calls[index + 1][1], calls[index + 1][1].tobytes() in accepted)
        for index, (kind, x) in enumerate(calls[:-1])
        if kind == "jacobian" and calls[index + 1][0] == "residual"
    ]


def point_data(problem, x, mask):
    """Return the public equilibrium state and implicit parameters at ``x``."""
    problem.residual(x)
    equilibrium = problem.equilibrium_from_x(x)
    cfg = problem.metadata["config"]
    params = imp.params_from_input(problem.input_from_x(x), device=cfg.device)
    return equilibrium.result.state, params, mask


def factors_at(cfg, params, state, mask):
    project = imp._dof_projector(cfg, mask)
    return imp._refine_block_factors(cfg, params, state, mask, project(state))


def finish(cfg, params, state, mask, factors, *, steps: int, z=None):
    """Run exact-operator Newton steps through a supplied right preconditioner."""
    state, params, mask = commit_to_single_device((state, params, mask))
    project = imp._dof_projector(cfg, mask)
    residual = imp.residual_fn(cfg, state, mask)
    z = project(state) if z is None else z
    start = float(imp._tree_norm(residual(z, params)))
    rows = []
    for _ in range(steps):
        before = dict(imp._SOLVE_STATS.get(cfg) or {})
        begun = time.perf_counter()
        z, _, norm, linear = imp._refine_block_step(
            cfg, params, state, mask, z, factors,
        )
        jax.block_until_ready(z)
        after = dict(imp._SOLVE_STATS.get(cfg) or {})
        rows.append({
            "nonlinear_residual": float(norm),
            "linear_relative_residual": float(linear),
            "gmres_iterations": int(
                after.get("refinement_krylov_iterations", 0)
                - before.get("refinement_krylov_iterations", 0)
            ),
            "seconds": time.perf_counter() - begun,
        })
        if not np.isfinite(float(norm)) or float(norm) <= 0.1 * float(cfg.refine_tol):
            break
    best = min([start, *(row["nonlinear_residual"] for row in rows)])
    return {
        "start": start,
        "steps": rows,
        "best": best,
        "last": rows[-1]["nonlinear_residual"] if rows else start,
        "reached_refine_tol": bool(best <= float(cfg.refine_tol)),
        "seconds": sum(row["seconds"] for row in rows),
    }, z


def parameter_tangent(problem, x):
    """First optimization-dof tangent in the public parameter pytree."""
    direction = np.zeros_like(x)
    direction[0] = 1.0
    base = imp.params_from_input(problem.input_from_x(x))
    shifted = imp.params_from_input(problem.input_from_x(x + direction))
    return jax.tree.map(jnp.subtract, shifted, base)


@functools.partial(jax.jit, static_argnames=("cfg", "rtol"))
def response_with_refinement_factors(
    params, state, mask, z_star, factors, tangent, *, cfg, rtol,
):
    """Solve one final-anchor response through refinement factors."""
    project = imp._dof_projector(cfg, mask)
    raw = imp.residual_fn(cfg, state, mask, formulation="raw")
    rhs = jax.tree.map(
        jnp.negative,
        jax.jvp(lambda prm: raw(z_star, prm), (params,), (tangent,))[1],
    )
    block_factors, row_scale, column_scale = factors

    def precondition(value):
        return imp._block_inverse_apply(
            block_factors, lambda tree: imp._pack_active(cfg, tree),
            lambda matrix: imp._unpack_active(cfg, matrix), project,
            row_scale, column_scale, value,
        )

    solution, krylov = imp._adjoint_solve_gcrot(
        lambda value: jax.jvp(
            lambda z: raw(z, params), (z_star,), (value,),
        )[1],
        rhs, cfg, precond=precondition, rtol=rtol,
        max_restarts=cfg.jacobian_adjoint_maxiter, enforce=False,
    )
    return solution, krylov.iterations


@functools.partial(jax.jit, static_argnames=("cfg",))
def response_certificate(params, state, mask, z_star, tangent, solution, *, cfg):
    """True raw-operator response defect at the final anchor."""
    raw = imp.residual_fn(cfg, state, mask, formulation="raw")
    rhs = jax.tree.map(
        jnp.negative,
        jax.jvp(lambda prm: raw(z_star, prm), (params,), (tangent,))[1],
    )
    applied = jax.jvp(
        lambda z: raw(z, params), (z_star,), (solution,),
    )[1]
    return imp._tree_norm(jax.tree.map(jnp.subtract, rhs, applied)), \
        imp._tree_norm(rhs)


def objective_direction(problem, cfg, params, state, mask, z_star, tangent,
                        response):
    """Objective-residual and least-squares directional derivatives."""
    rows_from_state = problem.metadata["jax_residual_from_state"]
    project = imp._dof_projector(cfg, mask)
    edge = imp._edge_mask(cfg)

    def rows(z, prm):
        runtime = imp.runtime_from_params(prm, cfg)
        physical = imp._assemble(z, runtime, state, project, edge)
        return rows_from_state(physical, runtime)

    value, column = jax.jvp(
        rows, (z_star, params), (project(response), tangent),
    )
    return column, jnp.vdot(value, column).real


def final_anchor_response(problem, cfg, params, frozen, mask, z_star, factors, x,
                          response_rtol):
    """Compare reused refinement factors with the fresh response path."""
    project = imp._dof_projector(cfg, mask)
    correction = project(jax.tree.map(
        jnp.subtract, z_star, project(frozen),
    ))
    state = jax.tree.map(jnp.add, frozen, correction)
    z_star = project(state)
    tangent = parameter_tangent(problem, x)

    # Compile each response before timing its warm execution.
    warm_reused, _ = response_with_refinement_factors(
        params, state, mask, z_star, factors, tangent, cfg=cfg,
        rtol=response_rtol,
    )
    jax.block_until_ready(warm_reused)
    begun = time.perf_counter()
    reused, reused_iterations = response_with_refinement_factors(
        params, state, mask, z_star, factors, tangent, cfg=cfg,
        rtol=response_rtol,
    )
    jax.block_until_ready(reused)
    reused_seconds = time.perf_counter() - begun

    tangent_batch = jax.tree.map(lambda value: value[None], tangent)
    fields = imp._active_state_fields(cfg)
    probe = int(np.ceil(np.sqrt(len(fields) * int(mask.R_cos.shape[1]))))
    fresh_response = jax.jit(lambda prm, anchor, dof_mask, tangents:
        imp._implicit_evolved_tangent_multi_rhs(
            prm, cfg, anchor, dof_mask, tangents, active_fields=fields,
            probe_chunk_size=probe, response_chunk_size=1,
            certify_rtol=response_rtol,
            certify_maxiter=cfg.jacobian_adjoint_maxiter,
        ))
    warm_fresh, _ = fresh_response(params, state, mask, tangent_batch)
    jax.block_until_ready(warm_fresh)
    begun = time.perf_counter()
    fresh, fresh_report = fresh_response(params, state, mask, tangent_batch)
    jax.block_until_ready(fresh)
    fresh_seconds = time.perf_counter() - begun
    fresh_single = jax.tree.map(lambda value: value[0], fresh)
    reused_defect, rhs_norm = response_certificate(
        params, state, mask, z_star, tangent, reused, cfg=cfg,
    )
    fresh_defect, _ = response_certificate(
        params, state, mask, z_star, tangent, fresh_single, cfg=cfg,
    )
    tolerance = float(imp._adjoint_acceptance(cfg, rhs_norm, response_rtol))
    agreement = float(imp._tree_norm(jax.tree.map(
        lambda old, new: old - new[0], reused, fresh,
    )))
    fresh_norm = float(imp._tree_norm(fresh_single))
    reused_column, reused_objective = objective_direction(
        problem, cfg, params, state, mask, z_star, tangent, reused,
    )
    fresh_column, fresh_objective = objective_direction(
        problem, cfg, params, state, mask, z_star, tangent, fresh_single,
    )
    column_difference = float(jnp.linalg.norm(reused_column - fresh_column))
    column_norm = float(jnp.linalg.norm(fresh_column))
    objective_difference = abs(float(reused_objective - fresh_objective))
    objective_scale = abs(float(fresh_objective))
    tiny = np.finfo(float).tiny
    raw = imp.residual_fn(cfg, state, mask, formulation="raw")
    return {
        "final_raw_residual": float(imp._tree_norm(raw(z_star, params))),
        "configured_jacobian_rtol": float(cfg.jacobian_adjoint_tol),
        "configured_scalar_adjoint_rtol": float(cfg.adjoint_tol),
        "requested_response_rtol": response_rtol,
        "acceptance_slack": float(imp._ADJOINT_RESIDUAL_SLACK),
        "rhs_norm": float(rhs_norm),
        "comparison": (
            "both paths use the requested response tolerance and are "
            "certified on the identical exact final raw operator"
        ),
        "reused": {
            "residual_norm": float(reused_defect),
            "relative_residual": float(reused_defect) / max(float(rhs_norm), tiny),
            "tolerance": tolerance,
            "converged": bool(float(reused_defect) <= tolerance),
            "iterations": int(reused_iterations),
            "seconds": reused_seconds,
        },
        "fresh": {
            "residual_norm": float(fresh_defect),
            "relative_residual": float(fresh_defect) / max(float(rhs_norm), tiny),
            "tolerance": tolerance,
            "converged": bool(float(fresh_defect) <= tolerance),
            "iterations": int(fresh_report.iterations[0]),
            "seconds": fresh_seconds,
        },
        "solution_difference_norm": agreement,
        "solution_difference_relative": agreement / fresh_norm,
        "objective_residual_direction_difference_norm": column_difference,
        "objective_residual_direction_difference_relative": (
            column_difference / max(column_norm, tiny)
        ),
        "least_squares_directional_derivative": {
            "reused": float(reused_objective),
            "fresh": float(fresh_objective),
            "difference": objective_difference,
            "difference_relative": objective_difference / max(objective_scale, tiny),
        },
    }


def join_finishes(first, second):
    """Combine consecutive step records without losing the original start."""
    rows = [*first["steps"], *second["steps"]]
    best = min([first["start"], *(row["nonlinear_residual"] for row in rows)])
    return {
        "start": first["start"],
        "steps": rows,
        "best": best,
        "last": rows[-1]["nonlinear_residual"] if rows else first["start"],
        "reached_refine_tol": first["reached_refine_tol"] or second[
            "reached_refine_tol"
        ],
        "seconds": first["seconds"] + second["seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("seed", "qi"), default="seed")
    parser.add_argument("--max-nfev", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--response", action="store_true")
    parser.add_argument("--response-rtol", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.max_nfev < 2 or args.pairs < 1 or args.steps < 1
            or (args.response_rtol is not None and args.response_rtol <= 0.0)
            or (args.response_rtol is not None and not args.response)):
        parser.error(
            "max-nfev >= 2, pairs >= 1, steps >= 1, and a positive optional "
            "response-rtol used with --response are required"
        )

    assert_repo_vmex(vmex.__file__, REPO)
    report = {
        "schema": 1,
        "case": args.case,
        "command": (
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "VMEX_COMPILATION_CACHE=disabled PYTHONPATH=. python "
            f"benchmarks/refinement_factor_reuse.py --case {args.case} "
            f"--max-nfev {args.max_nfev} --pairs {args.pairs} "
            f"--steps {args.steps}"
            f"{' --response' if args.response else ''} "
            f"{'--response-rtol ' + str(args.response_rtol) + ' ' if args.response_rtol is not None else ''}"
            "--output <output.json>"
        ),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "vmex": vmex.__version__,
        "load_start": os.getloadavg(),
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "persistent_compilation_cache": os.environ.get(
            "VMEX_COMPILATION_CACHE"
        ),
        "input_sha256": file_sha256(
            REPO / "examples/data" / (
                "input.QI_nfp2_initial" if args.case == "qi"
                else "input.minimal_seed_nfp2"
            )
        ),
        **git_state(REPO),
    }
    problem = build_problem(args.case)
    cfg = problem.metadata["config"]
    mask = imp._fixed_boundary_dof_mask(cfg)
    pairs = optimizer_pairs(problem, args.max_nfev)[: args.pairs]
    if not pairs:
        raise RuntimeError("bounded optimizer run produced no accepted/trial pair")

    rows = []
    for accepted_x, trial_x, trial_accepted in pairs:
        accepted_state, accepted_params, _ = point_data(problem, accepted_x, mask)
        stale = factors_at(cfg, accepted_params, accepted_state, mask)
        # Compile both reusable kernels at the accepted point before timing.
        jax.block_until_ready(stale)
        finish(cfg, accepted_params, accepted_state, mask, stale, steps=1)
        trial_state, trial_params, _ = point_data(problem, trial_x, mask)

        begun = time.perf_counter()
        fresh = factors_at(cfg, trial_params, trial_state, mask)
        jax.block_until_ready(fresh)
        factor_seconds = time.perf_counter() - begun
        fresh_finish, fresh_z = finish(
            cfg, trial_params, trial_state, mask, fresh, steps=args.steps,
        )
        stale_finish, _ = finish(
            cfg, trial_params, trial_state, mask, stale, steps=args.steps,
        )
        guarded_started = time.perf_counter()
        probe, probe_z = finish(
            cfg, trial_params, trial_state, mask, stale, steps=1,
        )
        admitted = (
            probe["steps"][0]["linear_relative_residual"]
            <= imp._REFINE_FORCING
        )
        fallback_factor_seconds = None
        if admitted and args.steps > 1 and not probe["reached_refine_tol"]:
            tail, _ = finish(
                cfg, trial_params, trial_state, mask, stale,
                steps=args.steps - 1, z=probe_z,
            )
            guarded = join_finishes(probe, tail)
        elif admitted:
            guarded = probe
        else:
            fallback_started = time.perf_counter()
            rebuilt = factors_at(cfg, trial_params, trial_state, mask)
            jax.block_until_ready(rebuilt)
            fallback_factor_seconds = time.perf_counter() - fallback_started
            guarded, _ = finish(
                cfg, trial_params, trial_state, mask, rebuilt,
                steps=args.steps,
            )
        guarded["seconds"] = time.perf_counter() - guarded_started
        row = {
            "scaled_step": float(
                np.linalg.norm((trial_x - accepted_x) / problem.scales)
            ),
            "trial_accepted": bool(trial_accepted),
            "fresh_factor_seconds": factor_seconds,
            "fresh": fresh_finish,
            "fresh_total_seconds": factor_seconds + fresh_finish["seconds"],
            "stale": stale_finish,
            "stale_admitted": bool(admitted),
            "guard_probe": probe,
            "fallback_factor_seconds": fallback_factor_seconds,
            "guarded": guarded,
        }
        if (args.response and fresh_finish["reached_refine_tol"]
                and fresh_finish["last"] == fresh_finish["best"]):
            row["final_anchor_response"] = final_anchor_response(
                problem, cfg, trial_params, trial_state, mask, fresh_z, fresh,
                trial_x, (float(cfg.adjoint_tol) if args.response_rtol is None
                          else args.response_rtol),
            )
        rows.append(row)

    report.update(
        forcing_tolerance=imp._REFINE_FORCING,
        policy=(
            "reuse accepted-point factors only as a right preconditioner for "
            "the exact trial operator when the first GMRES solve reaches the "
            "existing forcing tolerance; otherwise rebuild and replay"
        ),
        rows=rows,
        load_end=os.getloadavg(),
        limitation=(
            "Wall times are evidence only when before/after runs use the same "
            "revision, fixed threads, disabled persistent compilation cache, "
            "and comparable load."
        ),
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text)


if __name__ == "__main__":
    main()

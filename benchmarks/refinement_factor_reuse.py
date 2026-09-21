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
        "reached_refine_tol": bool(best <= float(cfg.refine_tol)),
        "seconds": sum(row["seconds"] for row in rows),
    }, z


def join_finishes(first, second):
    """Combine consecutive step records without losing the original start."""
    rows = [*first["steps"], *second["steps"]]
    best = min([first["start"], *(row["nonlinear_residual"] for row in rows)])
    return {
        "start": first["start"],
        "steps": rows,
        "best": best,
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_nfev < 2 or args.pairs < 1 or args.steps < 1:
        parser.error("max-nfev >= 2, pairs >= 1, and steps >= 1 are required")

    assert_repo_vmex(vmex.__file__, REPO)
    report = {
        "schema": 1,
        "case": args.case,
        "command": (
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "VMEX_COMPILATION_CACHE=disabled PYTHONPATH=. python "
            f"benchmarks/refinement_factor_reuse.py --case {args.case} "
            f"--max-nfev {args.max_nfev} --pairs {args.pairs} "
            f"--steps {args.steps} --output <output.json>"
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
        fresh_finish, _ = finish(
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
        rows.append({
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
        })

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

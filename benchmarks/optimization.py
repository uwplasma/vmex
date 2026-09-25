#!/usr/bin/env python
"""Profile one optimizer-neutral VMEX problem in a fresh process.

The selectable QI, QA, QH, QP, and scalar cases exercise the public SciPy and
JAX contracts without committing machine-specific timings to the repository.
Use separate processes when comparing cold compilation policies.

Each phase (build, first derivative, the two JAX contract checks, the
optimizer, the final evaluation) records its wall time, the change in the
implicit configuration's effort counters (``solve_stats``: solve, refinement,
Jacobian and adjoint counts and exclusive host seconds) and the XLA
compilations it ran.  Compile time inside a counted part is already in that
part's seconds, ``compile_seconds_outside_parts`` is the remainder, and
``unattributed_seconds`` is the wall time neither explains.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib import metadata
import json
import logging
import os
import platform
from pathlib import Path
import re
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.optimize

import vmex
from vmex import optimize as opt
from vmex.core import implicit as imp
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual

from _provenance import assert_repo_vmex, file_sha256, git_state


REPO = Path(__file__).resolve().parents[1]
PARTS = ("solve", "refinement", "jacobian", "adjoint")
COMPILE_RECORD = re.compile(
    r"Finished (tracing|jaxpr to MLIR module conversion|XLA compilation of) .* "
    r"in ([0-9.eE+-]+) sec")


def version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class CompileLog(logging.Handler):
    """``jax_log_compiles`` records as ``(end, seconds, is_xla, inside_part)``."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[tuple[float, float, bool, bool]] = []

    def emit(self, record: logging.LogRecord) -> None:
        match = COMPILE_RECORD.search(record.getMessage())
        if match:
            self.records.append((record.created, float(match[2]),
                                 match[1].startswith("XLA"),
                                 bool(imp._OPEN_SECTIONS)))


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Length of a union of intervals: nested traces must not count twice."""
    total, reach = 0.0, -np.inf
    for start, end in sorted(intervals):
        if end > reach:
            total += end - max(start, reach)
            reach = end
    return total


def terms(case: str):
    surfaces = np.linspace(0.1, 1.0, 6)

    def iota_floor(state, runtime):
        return jnp.maximum(0.33 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

    constraints = [(opt.aspect_ratio, 6.0, 0.01), (iota_floor, 0.0, 10.0)]
    if case == "qi":
        return [(QIResidual(surfaces), 0.0, 1.0), *constraints]
    if case in ("qa", "qh", "qp"):
        helicity = {"qa": (1, 0), "qh": (1, -1), "qp": (0, 1)}[case]
        return [(opt.QuasisymmetryRatioResidual(surfaces, *helicity), 0.0, 1.0), *constraints]
    return [
        (opt.aspect_ratio, 6.0, 1.0),
        (opt.mirror_ratio, 0.2, 1.0),
        (opt.magnetic_well, 0.05, 1.0),
        (iota_floor, 0.0, 10.0),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("qi", "qa", "qh", "qp", "scalar"), default="qi")
    parser.add_argument("--nfp", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--max-mode", type=int, default=1)
    parser.add_argument("--derivatives", choices=("implicit", "finite_difference"), default="implicit")
    parser.add_argument("--optimizer", choices=("none", "least_squares", "BFGS", "L-BFGS-B"), default="none")
    parser.add_argument("--nfev", type=int, default=2)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--forward-ftol", type=float)
    parser.add_argument("--forward-max-iterations", type=int)
    parser.add_argument("--max-fsq-ratio", type=float, default=1.0e2)
    args = parser.parse_args()

    compile_log = CompileLog()
    jax_logger = logging.getLogger("jax")
    for handler in jax_logger.handlers:  # compile records must not flood stderr
        handler.setLevel(logging.ERROR)
    jax_logger.addHandler(compile_log)
    jax_logger.setLevel(logging.WARNING)
    jax.config.update("jax_log_compiles", True)
    load_start = os.getloadavg()

    context: dict[str, Any] = {}
    phases: dict[str, dict[str, Any]] = {}

    def counters() -> dict[str, Any]:
        stats = imp._SOLVE_STATS.get(context["config"]) if "config" in context else None
        return {} if stats is None else dict(stats)

    def phase(name: str, function: Callable[[], Any]) -> Any:
        before, first = counters(), len(compile_log.records)
        started = time.perf_counter()
        result = function()
        wall = time.perf_counter() - started
        after, records = counters(), compile_log.records[first:]
        delta = {
            key: None if total is None or before.get(key, 0) is None
            else total - before.get(key, 0)
            for key, total in after.items()
        }
        spans = [(end - seconds, end, inside) for end, seconds, _xla, inside in records]
        outside = union_seconds([(start, end) for start, end, inside in spans if not inside])
        counted = sum(delta[f"{part}_seconds"] or 0.0 for part in PARTS
                      if f"{part}_seconds" in delta)
        phases[name] = {
            "wall_seconds": wall,
            "counters": delta,
            "compiles": sum(xla for _end, _seconds, xla, _inside in records),
            "compile_seconds": union_seconds([(start, end) for start, end, _ in spans]),
            "compile_seconds_outside_parts": outside,
            "unattributed_seconds": wall - counted - outside,
        }
        return result

    path = args.input or REPO / f"examples/data/input.minimal_seed_nfp{min(args.nfp, 4)}"
    inp = VmecInput.from_file(path)
    if args.input is None and inp.nfp != args.nfp:
        inp = replace(inp, nfp=args.nfp)
    mpol = max(args.max_mode + 2, 5)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4,
    )

    def build():
        problem = opt.VmecProblem.from_tuples(
            inp, terms(args.case), max_mode=args.max_mode,
            derivative_method=args.derivatives, workers=args.workers,
            forward_ftol=args.forward_ftol,
            forward_max_iterations=args.forward_max_iterations,
            max_fsq_ratio=args.max_fsq_ratio, use_ess=True,
        )
        context["config"] = problem.metadata.get("config")
        return problem

    problem = phase("build", build)
    value, gradient = phase(
        "first_derivative", lambda: problem.value_and_grad(problem.x0))
    contract = {"host_value": value, "host_gradient_norm": float(np.linalg.norm(gradient))}
    if args.derivatives == "implicit":
        jax_value, jax_gradient = phase("jax_value_and_grad", lambda: jax.device_get(
            problem.jax_value_and_grad(jnp.asarray(problem.x0))
        ))
        contract.update(
            jax_seconds=phases["jax_value_and_grad"]["wall_seconds"],
            value_relative_error=float(abs(jax_value - value) / max(abs(value), 1.0)),
            gradient_relative_error=float(
                np.linalg.norm(jax_gradient - gradient) / max(np.linalg.norm(gradient), 1.0)
            ),
        )
        graph_value, graph_gradient = phase("differentiated_graph", lambda: jax.device_get(
            jax.value_and_grad(problem.jax_fun)(jnp.asarray(problem.x0))
        ))
        contract.update(
            differentiated_graph_seconds=phases["differentiated_graph"]["wall_seconds"],
            differentiated_graph_value_relative_error=float(
                abs(graph_value - value) / max(abs(value), 1.0)
            ),
            differentiated_graph_gradient_relative_error=float(
                np.linalg.norm(graph_gradient - gradient) / max(np.linalg.norm(gradient), 1.0)
            ),
        )

    def optimize():
        if args.optimizer == "least_squares":
            return scipy.optimize.least_squares(
                problem.residual, problem.x0, jac=problem.residual_jac,
                x_scale=problem.scales, max_nfev=args.nfev,
            ).x
        if args.optimizer != "none":
            return scipy.optimize.minimize(
                problem.value_and_grad, problem.x0, jac=True, method=args.optimizer,
                options={"maxiter": args.nfev},
            ).x
        return problem.x0

    x = phase("optimize", optimize)
    evaluation, final_cost = phase("final_evaluation", lambda: (
        problem.evaluate(x, derivatives=False), float(problem.fun(x))))

    wall = sum(item["wall_seconds"] for item in phases.values())
    unattributed = sum(item["unattributed_seconds"] for item in phases.values())
    report = {
        "case": args.case,
        "nfp": int(inp.nfp),
        "max_mode": args.max_mode,
        "dofs": int(problem.x0.size),
        "derivatives": args.derivatives,
        "optimizer": args.optimizer,
        "forward_ftol": problem.metadata["forward_ftol"],
        "forward_max_iterations": problem.metadata["forward_max_iterations"],
        "build_seconds": phases["build"]["wall_seconds"],
        "derivative_seconds": phases["first_derivative"]["wall_seconds"],
        "optimize_seconds": phases["optimize"]["wall_seconds"],
        "initial_cost": value,
        "final_cost": final_cost,
        "contract": contract,
        "diagnostics": dict(evaluation.diagnostics),
        "phases": phases,
        "counters": counters(),
        "compiles": sum(item["compiles"] for item in phases.values()),
        "compile_seconds": sum(item["compile_seconds"] for item in phases.values()),
        "wall_seconds": wall,
        "unattributed_seconds": unattributed,
        "attributed_fraction": 1.0 - unattributed / wall,
        "threads": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")},
        "compilation_cache": os.environ.get("VMEX_COMPILATION_CACHE"),
        "load_average": {"start": load_start, "end": os.getloadavg()},
        "input_sha256": file_sha256(path),
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(), "vmex": vmex.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "jax": jax.__version__, "jaxopt": version("jaxopt"), "optax": version("optax"),
        },
        **git_state(REPO),
        "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

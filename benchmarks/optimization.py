#!/usr/bin/env python
"""Profile one optimizer-neutral VMEX problem in a fresh process.

The selectable QI, QA, QH, QP, and scalar cases exercise the public SciPy and
JAX contracts. Save paired reports with --output; --trace profiles distinct
warm points and adds overhead. Use fresh processes for cold comparisons.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from importlib import metadata
import json
import platform
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.optimize

import vmex
from vmex import optimize as opt
from vmex.core.input import VmecInput
from vmex.core.omnigenity import QIResidual
from vmex.core.qi import ConstructedQIResidual

from _provenance import assert_repo_vmex, file_sha256, git_state


REPO = Path(__file__).resolve().parents[1]


def version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def terms(case: str):
    surfaces = np.linspace(0.1, 1.0, 6)

    def iota_floor(state, runtime):
        return jnp.maximum(0.33 - jnp.abs(opt.mean_iota(state, runtime)), 0.0)

    constraints = [(opt.aspect_ratio, 6.0, 0.01), (iota_floor, 0.0, 10.0)]
    if case in ("qi", "constructed_qi"):
        objective = QIResidual if case == "qi" else ConstructedQIResidual
        return [(objective(surfaces), 0.0, 1.0), *constraints]
    if case in ("qa", "qh", "qp"):
        helicity = {"qa": (1, 0), "qh": (1, -1), "qp": (0, 1)}[case]
        return [(opt.QuasisymmetryRatioResidual(surfaces, *helicity), 0.0, 1.0), *constraints]
    return [
        (opt.aspect_ratio, 6.0, 1.0),
        (opt.mirror_ratio, 0.2, 1.0),
        (opt.magnetic_well, 0.05, 1.0),
        (iota_floor, 0.0, 10.0),
    ]


def fieldline_kernel_report():
    """Dense reference versus separated synthesis, including both AD modes."""
    from vmex.core.bounce import _boozer_field_strength

    modes = np.array([(m, n) for m in range(18) for n in range(-18, 19) if m or n >= 0])
    xm, xn = map(jnp.asarray, modes.T)
    alpha = jnp.linspace(0., 2 * jnp.pi, 27, endpoint=False)
    phi = jnp.linspace(-.2, 2 * jnp.pi - .2, 141)
    rng = np.random.default_rng(81)
    cosine = jnp.asarray(rng.normal(size=(4, len(modes))) / (1 + np.arange(len(modes)))**2)
    args = (cosine, .13 * cosine, jnp.array([.27, .32, .39, .44]))
    tangents = tuple(jnp.asarray(rng.normal(size=x.shape)) * .01 for x in args)

    def dense(c, s, iota):
        theta = alpha[None, :, None] + iota[:, None, None] * phi[None, None, :]
        phase = theta[..., None] * xm - phi[None, None, :, None] * xn
        return (jnp.einsum("sapm,sm->sap", jnp.cos(phase), c)
                + jnp.einsum("sapm,sm->sap", jnp.sin(phase), s))

    def separated(c, s, iota):
        return _boozer_field_strength(c, s, xm, xn, iota, alpha, phi)

    report, values = {}, {}
    for name, fn in (("dense", dense), ("separated", separated)):
        for mode, function in (
            ("value", fn), ("jvp", lambda *x: jax.jvp(fn, x, tangents)),
            ("vjp", jax.grad(lambda *x: jnp.sum(fn(*x)**2), argnums=(0, 1, 2))),
        ):
            started = time.perf_counter()
            executable = jax.jit(function).lower(*args).compile()
            compile_seconds = time.perf_counter() - started
            values[name, mode] = jax.block_until_ready(executable(*args))
            samples = []
            for _ in range(15):
                started = time.perf_counter()
                jax.block_until_ready(executable(*args))
                samples.append(time.perf_counter() - started)
            memory = executable.memory_analysis()
            report[f"{name}_{mode}"] = dict(
                compile_seconds=compile_seconds, samples_seconds=samples,
                median_seconds=float(np.median(samples)),
                temporary_bytes=None if memory is None else memory.temp_size_in_bytes)
    for mode in ("value", "jvp", "vjp"):
        report[f"parity_{mode}"] = [
            dict(max_absolute=float(np.max(np.abs(np.asarray(a) - np.asarray(b)))),
                 relative_l2=float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)))
            for a, b in zip(jax.tree.leaves(values["dense", mode]),
                            jax.tree.leaves(values["separated", mode]))]
    return dict(kernel="fieldline_synthesis", shape=[4, 27, 141, len(modes)],
                dtype=str(cosine.dtype), jax=jax.__version__,
                devices=[str(device) for device in jax.devices()],
                vmex_module=assert_repo_vmex(vmex.__file__, REPO),
                **git_state(REPO), **report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("qi", "constructed_qi", "qa", "qh", "qp", "scalar"), default="qi")
    parser.add_argument("--nfp", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--max-mode", type=int, default=1)
    parser.add_argument("--derivatives", choices=("implicit", "finite_difference"), default="implicit")
    parser.add_argument("--optimizer", choices=("none", "least_squares", "BFGS", "L-BFGS-B"), default="none")
    parser.add_argument("--nfev", type=int, default=2)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--forward-ftol", type=float)
    parser.add_argument("--forward-max-iterations", type=int)
    parser.add_argument("--max-fsq-ratio", type=float, default=1.0e6)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--refine-tol", type=float, default=1.0e-10)
    parser.add_argument("--batch-size", default="1", help="Jacobian columns per batch, or auto")
    parser.add_argument("--warm-evaluations", type=int, default=0)
    parser.add_argument("--keep-input-resolution", action="store_true")
    parser.add_argument("--trace", type=Path, help="Profile warm evaluations (adds timing overhead)")
    parser.add_argument("--output", type=Path, help="JSON report and companion numerical .npz")
    parser.add_argument("--skip-jax-contract", action="store_true", help="Measure only the host optimizer API")
    parser.add_argument("--fieldline-kernel", action="store_true", help="Compare Fourier synthesis with a dense oracle")
    args = parser.parse_args()
    assert_repo_vmex(vmex.__file__, REPO)
    if args.fieldline_kernel:
        device = jax.devices(None if args.device == "auto" else args.device)[0]
        with jax.default_device(device):
            report = fieldline_kernel_report()
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text, end="")
        return
    if args.warm_evaluations < 0:
        parser.error("--warm-evaluations must be non-negative")
    batch_size = "auto" if args.batch_size == "auto" else int(args.batch_size)

    path = args.input or REPO / f"examples/data/input.minimal_seed_nfp{min(args.nfp, 4)}"
    inp = VmecInput.from_file(path)
    if args.input is None and inp.nfp != args.nfp:
        inp = replace(inp, nfp=args.nfp)
    mpol = max(args.max_mode + 2, 5)
    if not args.keep_input_resolution:
        inp = replace(inp, delt=0.5).change_resolution(
            mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4,
        )
    started = time.perf_counter()
    problem = opt.VmecProblem.from_tuples(
        inp, terms(args.case), max_mode=args.max_mode,
        derivative_method=args.derivatives, workers=args.workers,
        forward_ftol=args.forward_ftol,
        forward_max_iterations=args.forward_max_iterations,
        max_fsq_ratio=args.max_fsq_ratio, use_ess=True,
        device=args.device, refine_tol=args.refine_tol,
        jacobian_batch_size=batch_size,
    )
    build_seconds = time.perf_counter() - started
    print(f"Problem built in {build_seconds:.3f}s; evaluating first derivative", file=sys.stderr, flush=True)

    started = time.perf_counter()
    value, gradient = problem.value_and_grad(problem.x0)
    derivative_seconds = time.perf_counter() - started
    print(f"First derivative: {derivative_seconds:.3f}s", file=sys.stderr, flush=True)
    contract = {"host_value": value, "host_gradient_norm": float(np.linalg.norm(gradient))}
    if args.derivatives == "implicit" and not args.skip_jax_contract:
        started = time.perf_counter()
        jax_value, jax_gradient = jax.device_get(
            problem.jax_value_and_grad(jnp.asarray(problem.x0))
        )
        contract.update(
            jax_seconds=time.perf_counter() - started,
            value_relative_error=float(abs(jax_value - value) / max(abs(value), 1.0)),
            gradient_relative_error=float(
                np.linalg.norm(jax_gradient - gradient) / max(np.linalg.norm(gradient), 1.0)
            ),
        )
        started = time.perf_counter()
        graph_value, graph_gradient = jax.device_get(
            jax.value_and_grad(problem.jax_fun)(jnp.asarray(problem.x0))
        )
        contract.update(
            differentiated_graph_seconds=time.perf_counter() - started,
            differentiated_graph_value_relative_error=float(
                abs(graph_value - value) / max(abs(value), 1.0)
            ),
            differentiated_graph_gradient_relative_error=float(
                np.linalg.norm(graph_gradient - gradient) / max(np.linalg.norm(gradient), 1.0)
            ),
        )

    # Distinct points bypass the content-keyed objective cache. Synchronize
    # every measurement, including device callbacks, before stopping its timer.
    warm = []
    samples = {}
    direction = np.linspace(-0.5, 0.5, problem.x0.size)
    context = jax.profiler.trace(str(args.trace)) if args.trace else nullcontext()
    with context:
        for i in range(args.warm_evaluations):
            trial = problem.x0 + (i + 1) * 1e-6 * direction
            started = time.perf_counter()
            residual = np.asarray(jax.block_until_ready(problem.residual(trial)))
            residual_seconds = time.perf_counter() - started
            started = time.perf_counter()
            jacobian = np.asarray(jax.block_until_ready(problem.residual_jac(trial)))
            warm.append(dict(residual_seconds=residual_seconds,
                             jacobian_seconds=time.perf_counter() - started))
            samples.update({f"x_{i}": trial, f"residual_{i}": residual,
                            f"jacobian_{i}": jacobian})

    x = problem.x0
    result = None
    started = time.perf_counter()
    if args.optimizer == "least_squares":
        result = scipy.optimize.least_squares(
            problem.residual, x, jac=problem.residual_jac, x_scale=problem.scales,
            max_nfev=args.nfev,
        )
        x = result.x
    elif args.optimizer != "none":
        result = scipy.optimize.minimize(
            problem.value_and_grad, x, jac=True, method=args.optimizer,
            options={"maxiter": args.nfev},
        )
        x = result.x
    optimize_seconds = time.perf_counter() - started
    print(f"Optimizer: {optimize_seconds:.3f}s", file=sys.stderr, flush=True)
    evaluation = problem.evaluate(x, derivatives=False)

    report = {
        "case": args.case,
        "nfp": int(inp.nfp),
        "max_mode": args.max_mode,
        "dofs": int(problem.x0.size),
        "derivatives": args.derivatives,
        "optimizer": args.optimizer,
        "optimizer_result": None if result is None else {
            "success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "nfev": int(result.nfev),
            "njev": int(result.njev),
        },
        "forward_ftol": problem.metadata["forward_ftol"],
        "forward_max_iterations": problem.metadata["forward_max_iterations"],
        "build_seconds": build_seconds,
        "derivative_seconds": derivative_seconds,
        "optimize_seconds": optimize_seconds,
        "warm_evaluations": warm,
        "refine_tol": args.refine_tol if np.isfinite(args.refine_tol) else "disabled",
        "jacobian_batch_size": batch_size,
        "devices": [str(device) for device in jax.devices()],
        "trace_enabled": args.trace is not None,
        "keep_input_resolution": args.keep_input_resolution,
        "initial_cost": value,
        "final_cost": float(problem.fun(x)),
        "contract": contract,
        "diagnostics": dict(evaluation.diagnostics),
        "input_sha256": file_sha256(path),
        "platform": platform.platform(),
        "versions": {
            "python": platform.python_version(), "vmex": vmex.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "jax": jax.__version__, "jaxlib": version("jaxlib"),
            "solvax": version("solvax"), "booz_xform_jax": version("booz_xform_jax"),
            "jaxopt": version("jaxopt"), "optax": version("optax"),
        },
        **git_state(REPO),
        "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        np.savez(args.output.with_suffix(".npz"), initial_value=value,
                 initial_gradient=gradient, final_x=x, **samples)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

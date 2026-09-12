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
    if case == "collaborator":
        qi = QIResidual(np.array([1, 5, 9, 13]) / 16, mboz=18, nboz=18,
                        oversample=2, nphi=141, nalpha=27, n_levels=16, softness=.02)
        return [
            (opt.aspect_ratio, 7., 1.), (qi, 0., 1.),
            (lambda s, r: jnp.maximum(opt.mirror_ratio(s, r) - .21, 0.), 0., 100.),
            (lambda s, r: jnp.maximum(opt.max_elongation(s, r) - 6., 0.), 0., 100.),
        ]
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


def repeatability(problem, args, path):
    """Diagnostic trajectory, including exact memo checks after failed trials.

    Private cache access is confined to this diagnostic. A cold replay must
    clear refinement as well as solve caches; clearing only the latter can
    pair an old objective with a new raw state at identical parameters.
    """
    from vmex.core import implicit as imp

    cfg = problem.metadata["config"]
    x = problem.x0.copy()
    direction = np.linspace(-.5, .5, x.size)
    direction /= np.linalg.norm(direction)
    rows, arrays = [], {"x0": x, "direction": direction}
    report = dict(case=args.case, device=args.device, residual_only=args.residual_only,
                  refine_tol=args.refine_tol if np.isfinite(args.refine_tol) else "disabled",
                  forward_ftol=float(cfg.ftol), input_sha256=file_sha256(path),
                  steps=args.repeatability_steps, platform=platform.platform(),
                  python=platform.python_version(),
                  source_sha256={str(p): file_sha256(REPO / p) for p in (
                      Path("benchmarks/optimization.py"),
                      *(Path(f"vmex/core/{name}.py") for name in
                        ("optimize", "implicit", "bounce", "omnigenity", "maxj")))},
                  versions={name: version(name) for name in
                            ("jax", "jaxlib", "numpy", "scipy", "solvax", "booz_xform_jax")},
                  **git_state(REPO), rows=rows)

    def evaluate(label, point):
        residual = np.asarray(problem.residual(point))
        row = dict(label=label, residual_cost=float(residual @ residual))
        if not args.residual_only:
            value, gradient = problem.value_and_grad(point)
            row.update(gradient_cost=2 * value, directional_gradient=float(2 * gradient @ direction),
                       derivative_certified=bool(problem.metadata["holder"].get("scalar_certified", False)))
        params = imp.params_from_input(problem.input_from_x(point))
        hit = imp._LAST_SOLVE.get(cfg)
        error = imp._LAST_STATUS_ERROR.get(cfg)
        exact = hit is not None and hit[0] == imp._params_key(params)
        row.update(exact_solve_memo=exact,
                   status_error=None if error is None else type(error).__name__)
        arrays[f"{label}_x"], arrays[f"{label}_residual"] = point.copy(), residual
        if exact and error is None:
            result = hit[1]
            state = np.concatenate([np.asarray(v).ravel() for v in jax.tree.leaves(result.state)])
            arrays[f"{label}_state"] = state
            row.update(converged=bool(result.converged), iterations=int(result.iterations),
                       fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
                       wb=float(result.wb), iota_min=float(np.min(result.iotaf)),
                       iota_max=float(np.max(result.iotaf)),
                       state_relative_change=float(np.linalg.norm(state - arrays["initial_state"])
                                                   / np.linalg.norm(arrays["initial_state"])))
            refined = imp._LAST_REFINED.get(cfg)
            if np.isfinite(args.refine_tol) and refined is not None and refined[0] == hit[0]:
                # Solver diagnostics above describe the raw forward state;
                # the objective consumes this separately refined state.
                evaluated = refined[1]
                flat = np.concatenate([np.asarray(v).ravel() for v in jax.tree.leaves(evaluated)])
                arrays[f"{label}_evaluated_state"] = flat
                mask = imp._fixed_boundary_dof_mask(cfg)
                project = imp._dof_projector(cfg, mask)
                force = imp.residual_fn(cfg, evaluated, mask)(project(evaluated), params)
                row.update(refined_state_relative_correction=float(np.linalg.norm(flat - state)
                                                                    / np.linalg.norm(state)),
                           evaluated_force_norm=float(imp._tree_norm(force)))
        else:
            row["converged"] = False
        rows.append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        print(json.dumps(row), flush=True)

    evaluate("initial", x)
    evaluate("repeat", x)
    for step in args.repeatability_steps:
        for sign, name in ((1, "plus"), (-1, "minus")):
            evaluate(f"{name}_{step}", x + sign * step * direction)
            evaluate(f"return_{name}_{step}", x)
    problem._vg_cache = problem._rj_cache = None
    problem.metadata["holder"]["lin"] = None
    for cache in (imp._LAST_SOLVE, imp._HOT_CACHE, imp._PERTURB_SEED,
                  imp._LAST_REFINED, imp._LAST_REFINEMENT_CORRECTION, imp._LAST_STATUS_ERROR):
        cache.pop(cfg, None)
    evaluate("cold_replay", x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("qi", "constructed_qi", "qa", "qh", "qp", "scalar", "collaborator"), default="qi")
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
    parser.add_argument("--repeatability", action="store_true", help="Replay identical parameters between nearby trials")
    parser.add_argument("--repeatability-steps", nargs="+", type=float, default=[1e-4, 1e-6])
    parser.add_argument("--residual-only", action="store_true", help="Skip derivatives in the repeatability diagnostic")
    args = parser.parse_args()
    assert_repo_vmex(vmex.__file__, REPO)
    if args.repeatability:
        if args.output is None or args.derivatives != "implicit":
            parser.error("--repeatability requires --output and implicit derivatives")
        if any(not np.isfinite(step) or step <= 0 for step in args.repeatability_steps):
            parser.error("repeatability steps must be finite and positive")
        args.output.parent.mkdir(parents=True, exist_ok=True)
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
    if args.case == "collaborator" and args.input is None:
        path = REPO / "benchmarks/review_optimization_20260912.json"
        inp = VmecInput.from_json_text(json.dumps(json.loads(path.read_text())["repeatability_input"]))
    else:
        inp = VmecInput.from_file(path)
    if args.input is None and args.case != "collaborator" and inp.nfp != args.nfp:
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
    if args.repeatability:
        repeatability(problem, args, path)
        return
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

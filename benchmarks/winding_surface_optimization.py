#!/usr/bin/env python
"""Benchmark winding-surface objective revisions without importing the example.

The optimization example executes a full VMEX campaign at module import time, so
this benchmark loads only its pure helper functions from each requested git
revision.  It measures two deliberately separate contexts:

* ``standalone`` evaluates the entropy/PCA scalar, its gradient, and an HVP on
  fixed winding and plasma surfaces.  The unsplit full-matrix SVD is the fidelity
  reference.
* ``optimization_context`` differentiates outer entropy, inverse-distance, and
  mixed observables through the inner L-BFGS-B winding-surface solve.  A patched
  copy of the original full-SVD implementation with ``maxiter=100`` is the
  fidelity reference; the historical implementation itself remains unmodified
  at four iterations.

Every timed executable is compiled in isolation, warmed once, and sampled with
the same deterministic cases and directions.  Compilation and warm execution
are reported separately.  The output contains every timing sample, not only a
best result, and the companion SVGs aggregate by the median across geometry
families with the observed range shown explicitly.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import functools
import gc
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Callable

import jax
import jax.numpy as jnp
from jaxopt import LBFGSB
from jaxopt.implicit_diff import root_jvp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from solvax import gcrot


REPO = Path(__file__).resolve().parents[1]
EXAMPLE = "examples/optimization/QA_winding_optimization.py"
DEFAULT_BASELINE = "origin/ds/working_winding"
DEFAULT_CURRENT = "2aa5d7508da1ab76d21c4ca27c807f2bc2ba5e92"

FUNCTIONS = {
    "_mode_angle",
    "r_surface",
    "z_surface",
    "r_surface_prime_phi",
    "z_surface_prime_phi",
    "r_surface_prime_theta",
    "z_surface_prime_theta",
    "surface_del_phi",
    "surface_del_theta",
    "surface_normal",
    "_ntor_from_coefficients",
    "_symmetric_rc_modes",
    "_symmetric_zs_modes",
    "_active_dof_indices",
    "surface_coefficients_from_dofs",
    "coefficients_to_dofs",
    "mode_weights_from_coefficients",
    "points_normals_normal_lengths",
    "surface_quadrature_weights",
    "reduced_memory_induction_matrix",
    "_periodic_induction_singular_values",
    "_singular_value_objectives",
    "spectral_width",
    "smooth_minimum_distance",
    "smooth_minimum_tangent_radius",
    "enclosed_volume",
    "_calc_objectives",
    "calc_objectives",
    "_periodic_objectives",
    "_combine_winding_objectives",
    "_winding_objective_value",
    "_periodic_winding_objective_value",
    "_active_winding_objective_value",
    "_active_dof_bounds",
    "_solve_winding_surface",
    "_winding_linear_solve",
    "_solve_winding_surface_jvp",
}


@dataclass(frozen=True)
class GeometryCase:
    """Deterministic pair of nested, mildly three-dimensional tori."""

    name: str
    seed: int
    major_radius: float
    plasma_minor_radius: float
    winding_minor_radius: float
    perturbation: float


CASES = {
    case.name: case
    for case in (
        # The scale matches input.minimal_seed_nfp2: R=1 and a=0.1.  The
        # winding minor radius is the analytic one-minor-radius offset used by
        # the example.  The other cases vary clearance and 3-D shaping without
        # changing the algorithm or tuning solver tolerances per case.
        GeometryCase("nominal", 11, 1.0, 0.10, 0.20, 0.001),
        GeometryCase("close", 17, 1.0, 0.10, 0.16, 0.002),
        GeometryCase("shaped", 23, 1.0, 0.10, 0.20, 0.010),
    )
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_at(ref: str) -> tuple[str, str]:
    commit = _git("rev-parse", ref)
    source = _git("show", f"{commit}:{EXAMPLE}")
    return commit, source


def _namespace(source: str, *, resolution: int, reference: bool = False) -> dict:
    if reference:
        source = source.replace(
            "maxiter=4, tol=1e-6",
            "maxiter=WINDING_MAXITER, tol=WINDING_OPTIMALITY_TOL",
        )
    tree = ast.parse(source, filename=EXAMPLE)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in FUNCTIONS
    ]
    namespace = {
        "functools": functools,
        "jax": jax,
        "jnp": jnp,
        "LBFGSB": LBFGSB,
        "root_jvp": root_jvp,
        "gcrot": gcrot,
        "MU0": 4 * np.pi * 1e-7,
        "nfp": 2,
        "WINDING_NPHI": resolution,
        "WINDING_NTHETA": resolution,
        "WINDING_MAXITER": 100,
        "WINDING_OPTIMALITY_TOL": 1e-6,
        "WINDING_LINEAR_RTOL": 1e-8,
        "WINDING_LINEAR_MAX_RESTARTS": 10,
        "COEFFICIENT_STEP_BOUND": 1.0,
        "DENOMINATOR_EPS": 1e-16,
        "SHARPNESS": 300.0,
        "SELF_NEIGHBOR_RADIUS": 2,
        "MINIMUM_DISTANCE": 0.05,
        "DISTANCE_WALL_SCALE": 0.01,
        "MINIMUM_SELF_RADIUS": 0.05,
        "SELF_RADIUS_WALL_SCALE": 0.01,
        "MINIMUM_NORMAL_LENGTH": 1e-6,
        "INVALID_PENALTY_SCALE": 1e12,
        "PCA_WEIGHT": 1.0,
        "SINGULAR_STRENGTH_WEIGHT": 0.0,
        "VOLUME_WEIGHT": 0.02,
        "SPECTRAL_WEIGHT": 0.02,
        "DISTANCE_WALL_WEIGHT": 10.0,
        "SELF_INTERSECTION_WEIGHT": 100.0,
    }
    exec(compile(tree, EXAMPLE, "exec"), namespace)
    if reference:
        # JAXopt's historical default is CG on normal equations.  That is the
        # implementation being benchmarked, but it is not an independent
        # fidelity oracle.  The reference instead materializes the tiny 25x25
        # optimality Jacobian and solves it directly in float64.
        def dense_solve(matvec, rhs):
            matrix = jax.jacfwd(matvec)(jnp.zeros_like(rhs))
            return jnp.linalg.solve(matrix, rhs)

        def dense_root_jvp(*args, **kwargs):
            kwargs["solve"] = dense_solve
            return root_jvp(*args, **kwargs)

        namespace["root_jvp"] = dense_root_jvp
    namespace["_benchmark_maxiter"] = (
        100 if reference or "maxiter=WINDING_MAXITER" in source else 4
    )
    return namespace


def _dofs(case: GeometryCase) -> tuple[jax.Array, jax.Array]:
    """Return winding and plasma coefficient vectors in the example layout."""
    rng = np.random.default_rng(case.seed)

    def torus(minor: float, noise: np.ndarray) -> np.ndarray:
        rc = np.zeros((3, 5))
        zs = np.zeros((3, 5))
        rc[0, 2] = case.major_radius
        rc[1, 2] = minor
        zs[1, 2] = minor
        packed = np.r_[rc.ravel()[2:], zs.ravel()[3:]]
        return packed + case.perturbation * noise

    plasma_noise = rng.normal(size=25)
    winding_noise = rng.normal(size=25)
    # Correlated and independent shaping coexist, as they do for an offset
    # surface after Fourier refitting.
    winding_noise = 0.65 * plasma_noise + 0.35 * winding_noise
    return (
        jnp.asarray(torus(case.winding_minor_radius, winding_noise)),
        jnp.asarray(torus(case.plasma_minor_radius, plasma_noise)),
    )


def _tree_block(value):
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


def _timing_summary(samples: list[float]) -> dict[str, object]:
    values = np.asarray(samples) * 1e3
    q1, q3 = np.percentile(values, [25.0, 75.0])
    mean = float(np.mean(values))
    return {
        "samples_ms": values.tolist(),
        "median_ms": float(np.median(values)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
        "iqr_ms": float(q3 - q1),
        "mean_ms": mean,
        "stdev_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "cv": float(np.std(values, ddof=1) / mean) if len(values) > 1 else 0.0,
    }


def _measure(
    factory: Callable[[], tuple[Callable, tuple]], repeats: int
) -> tuple[dict[str, object], object]:
    """Compile in isolation, warm once, then retain every timing sample."""
    jax.clear_caches()
    gc.collect()
    function, arguments = factory()
    started = time.perf_counter()
    output = _tree_block(function(*arguments))
    compile_and_first = time.perf_counter() - started
    _tree_block(function(*arguments))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = _tree_block(function(*arguments))
        samples.append(time.perf_counter() - started)
    summary = _timing_summary(samples)
    summary["compile_and_first_ms"] = compile_and_first * 1e3
    return summary, output


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.linalg.norm(expected.ravel())
    return float(np.linalg.norm((actual - expected).ravel()) / max(denominator, 1e-300))


def _error(actual, expected) -> dict[str, float]:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    difference = np.abs(actual_array - expected_array)
    return {
        "absolute_max": float(np.max(difference)),
        "absolute_l2": float(np.linalg.norm(difference.ravel())),
        "relative_l2": _relative_error(actual_array, expected_array),
    }


def _full_objectives(namespace: dict, winding, plasma, weights, winding_rc):
    resolution = namespace["WINDING_NPHI"]
    points = namespace["points_normals_normal_lengths"]
    pp, pn, pj, _ = points(plasma, 2, 2, 2, resolution, resolution)
    pw = namespace["surface_quadrature_weights"](pj, resolution, resolution)
    return namespace["calc_objectives"](
        winding,
        pp.reshape((-1, 3)),
        pn.reshape((-1, 3)),
        weights,
        winding_rc,
        pw,
    )


def _variant_objectives(namespace: dict, winding, plasma, weights, winding_rc):
    if "_periodic_objectives" in namespace:
        return namespace["_periodic_objectives"](winding, plasma, weights, winding_rc)
    return _full_objectives(namespace, winding, plasma, weights, winding_rc)


def _standalone_function(namespace: dict, winding, plasma):
    winding_rc = namespace["surface_coefficients_from_dofs"](winding, 2, 2)[0]
    weights = namespace["mode_weights_from_coefficients"](winding_rc)

    def pca(value):
        winding_value, plasma_value = jnp.split(value, 2)
        return _variant_objectives(
            namespace, winding_value, plasma_value, weights, winding_rc
        )[0]

    return pca


def _standalone_row(
    label: str,
    namespace: dict,
    case: GeometryCase,
    resolution: int,
    repeats: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    winding, plasma = _dofs(case)
    value = jnp.concatenate((winding, plasma))
    rng = np.random.default_rng(case.seed + 10_000 + resolution)
    direction = jnp.asarray(rng.normal(size=value.shape))
    direction /= jnp.linalg.norm(direction)
    pca = _standalone_function(namespace, winding, plasma)
    gradient = jax.grad(pca)

    value_timing, objective = _measure(
        lambda: (jax.jit(pca), (value,)), repeats
    )
    gradient_timing, derivative = _measure(
        lambda: (jax.jit(gradient), (value,)), repeats
    )
    hvp_timing, hvp = _measure(
        lambda: (
            jax.jit(lambda argument, tangent: jax.jvp(gradient, (argument,), (tangent,))[1]),
            (value, direction),
        ),
        repeats,
    )
    outputs = {
        "value": np.asarray(objective),
        "gradient": np.asarray(derivative),
        "hvp": np.asarray(hvp),
    }
    row = {
        "case": case.name,
        "resolution": resolution,
        "matrix_order": resolution * resolution,
        "variant": label,
        "timing": {
            "value": value_timing,
            "gradient": gradient_timing,
            "hvp": hvp_timing,
        },
        "outputs": {
            "value": float(objective),
            "gradient_l2": float(jnp.linalg.norm(derivative)),
            "hvp_l2": float(jnp.linalg.norm(hvp)),
        },
    }
    return row, outputs


def _context_functions(namespace: dict, case: GeometryCase):
    base_winding, base_plasma = _dofs(case)
    active_indices = tuple(range(base_winding.size))
    active_array = jnp.asarray(active_indices)

    def geometry(delta):
        plasma = base_plasma + delta
        winding = base_winding + 0.5 * delta
        winding_rc = namespace["surface_coefficients_from_dofs"](winding, 2, 2)[0]
        weights = namespace["mode_weights_from_coefficients"](winding_rc)
        scales = _variant_objectives(
            namespace, winding, plasma, weights, winding_rc
        )
        return winding, plasma, winding_rc, weights, scales

    def solve(delta):
        winding, plasma, winding_rc, weights, scales = geometry(delta)
        initial = winding[active_array]
        if "_periodic_objectives" in namespace:
            return namespace["_solve_winding_surface"](
                initial,
                winding,
                active_indices,
                plasma,
                weights,
                winding_rc,
                scales,
            )
        resolution = namespace["WINDING_NPHI"]
        pp, pn, pj, _ = namespace["points_normals_normal_lengths"](
            plasma, 2, 2, 2, resolution, resolution
        )
        pw = namespace["surface_quadrature_weights"](pj, resolution, resolution)
        return namespace["_solve_winding_surface"](
            initial,
            winding,
            active_indices,
            pp.reshape((-1, 3)),
            pn.reshape((-1, 3)),
            pw,
            weights,
            winding_rc,
            scales,
        )

    def observables(delta):
        winding, plasma, winding_rc, weights, scales = geometry(delta)
        solution = solve(delta)
        optimized = winding.at[active_array].set(solution)
        objectives = _variant_objectives(
            namespace, optimized, plasma, weights, winding_rc
        )
        entropy_normalized = objectives[0] / jax.lax.stop_gradient(scales[0])
        inverse_distance_normalized = (
            jax.lax.stop_gradient(scales[3]) / objectives[3]
        )
        return jnp.stack((entropy_normalized, inverse_distance_normalized))

    def stationarity(delta, solution):
        winding, plasma, winding_rc, weights, scales = geometry(delta)
        initial = winding[active_array]
        bounds = namespace["_active_dof_bounds"](initial)
        if "_periodic_objectives" in namespace:
            args = (bounds, winding, plasma, weights, winding_rc, scales)
        else:
            resolution = namespace["WINDING_NPHI"]
            pp, pn, pj, _ = namespace["points_normals_normal_lengths"](
                plasma, 2, 2, 2, resolution, resolution
            )
            pw = namespace["surface_quadrature_weights"](pj, resolution, resolution)
            args = (
                bounds,
                winding,
                pp.reshape((-1, 3)),
                pn.reshape((-1, 3)),
                pw,
                weights,
                winding_rc,
                scales,
            )
        optimizer = LBFGSB(
            fun=lambda active, full, *rest: namespace[
                "_active_winding_objective_value"
            ](active, full, active_indices, *rest),
            maxiter=namespace["_benchmark_maxiter"],
            tol=namespace.get("WINDING_OPTIMALITY_TOL", 1e-6),
        )
        residual = optimizer.optimality_fun(solution, *args)
        return jnp.max(jnp.abs(residual))

    return solve, observables, stationarity


OBJECTIVE_WEIGHTS = {
    "entropy": jnp.array([1.0, 0.0]),
    "inverse_distance": jnp.array([0.0, 1.0]),
    "mixed": jnp.array([0.5, 0.5]),
}


def _context_row(
    label: str,
    namespace: dict,
    case: GeometryCase,
    resolution: int,
    repeats: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    solve, observables, stationarity = _context_functions(namespace, case)
    delta = jnp.zeros(25)
    scalar_observable = lambda argument, weights: jnp.vdot(  # noqa: E731
        observables(argument), weights
    )
    # The historical custom JVP is intentionally forward-only; benchmarking a
    # reverse transform that cannot run would misrepresent the original VMEX
    # path, which explicitly selected its forward/block-tridiagonal Jacobian.
    # The current implementation's certified custom linear solve is
    # transposable, so it takes the one-row reverse/adjoint path selected by
    # ``implicit_jacobian_method='auto'`` in the real optimization.
    if "_winding_linear_solve" in namespace:
        gradient = jax.grad(scalar_observable, argnums=0)
        differentiation_mode = "reverse"
    else:
        gradient = jax.jacfwd(scalar_observable, argnums=0)
        differentiation_mode = "forward"
    solve_timing, solution = _measure(
        lambda: (jax.jit(solve), (delta,)), repeats
    )
    gradient_timings = {}
    gradients = {}
    # The objective weights are dynamic, so all three scalar objectives share
    # exactly one compiled executable.  Only the first measurement includes
    # compilation; warm samples are retained separately for each objective.
    jax.clear_caches()
    gc.collect()
    compiled_gradient = jax.jit(gradient)
    compile_start = time.perf_counter()
    _tree_block(compiled_gradient(delta, OBJECTIVE_WEIGHTS["mixed"]))
    compile_ms = (time.perf_counter() - compile_start) * 1e3
    for name, weights in OBJECTIVE_WEIGHTS.items():
        _tree_block(compiled_gradient(delta, weights))
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            output = _tree_block(compiled_gradient(delta, weights))
            samples.append(time.perf_counter() - started)
        gradient_timings[name] = _timing_summary(samples)
        gradient_timings[name]["compile_and_first_ms"] = compile_ms
        gradients[name] = np.asarray(output)
    observable_values = np.asarray(_tree_block(jax.jit(observables)(delta)))
    optimality = float(_tree_block(jax.jit(stationarity)(delta, solution)))
    outputs = {
        "observables": observable_values,
        **{f"{name}_gradient": value for name, value in gradients.items()},
    }
    row = {
        "case": case.name,
        "resolution": resolution,
        "matrix_order": resolution * resolution,
        "variant": label,
        "timing": {
            "inner_solve": solve_timing,
            "outer_gradient": gradient_timings,
        },
        "outputs": {
            "entropy_normalized": float(observable_values[0]),
            "inverse_distance_normalized": float(observable_values[1]),
            "stationarity_inf": optimality,
            **{
                f"{name}_gradient_l2": float(np.linalg.norm(value))
                for name, value in gradients.items()
            },
        },
        "differentiation_mode": differentiation_mode,
    }
    return row, outputs


def _with_fidelity(row: dict, outputs: dict, reference: dict) -> dict:
    row = dict(row)
    row["fidelity"] = {
        name: _error(value, reference[name]) for name, value in outputs.items()
    }
    return row


def _parse_variants(values: list[str]) -> dict[str, str]:
    variants = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"variant must be LABEL=REF, got {value!r}")
        label, ref = value.split("=", 1)
        variants[label] = ref
    return variants


def run(args: argparse.Namespace) -> dict:
    jax.config.update("jax_enable_x64", True)
    variants = _parse_variants(args.variant)
    sources = {}
    commits = {}
    for label, ref in variants.items():
        commits[label], sources[label] = _source_at(ref)
    baseline_commit, baseline_source = _source_at(args.baseline_ref)
    commits["full_svd_100_reference"] = baseline_commit

    standalone = []
    for case_name in args.cases:
        case = CASES[case_name]
        for resolution in args.resolutions:
            rows = {}
            outputs = {}
            for label, source in sources.items():
                print(
                    f"standalone case={case.name} resolution={resolution} variant={label}",
                    flush=True,
                )
                namespace = _namespace(source, resolution=resolution)
                rows[label], outputs[label] = _standalone_row(
                    label, namespace, case, resolution, args.repeats
                )
            # The original implementation is the unsplit full-SVD reference
            # for fixed-geometry value/derivative fidelity.
            reference_label = args.standalone_reference
            reference = outputs[reference_label]
            standalone.extend(
                _with_fidelity(rows[label], outputs[label], reference)
                for label in variants
            )

    context = []
    for case_name in args.context_cases:
        case = CASES[case_name]
        for resolution in args.context_resolutions:
            print(
                f"context case={case.name} resolution={resolution} "
                "variant=full_svd_100_reference",
                flush=True,
            )
            reference_namespace = _namespace(
                baseline_source, resolution=resolution, reference=True
            )
            reference_namespace["_benchmark_reference"] = True
            reference_row, reference_outputs = _context_row(
                "full_svd_100_reference",
                reference_namespace,
                case,
                resolution,
                args.context_repeats,
            )
            reference_row = _with_fidelity(
                reference_row, reference_outputs, reference_outputs
            )
            context.append(reference_row)
            for label, source in sources.items():
                print(
                    f"context case={case.name} resolution={resolution} variant={label}",
                    flush=True,
                )
                namespace = _namespace(source, resolution=resolution)
                row, outputs = _context_row(
                    label, namespace, case, resolution, args.context_repeats
                )
                context.append(_with_fidelity(row, outputs, reference_outputs))

    return {
        "_provenance": {
            "measurement_commit": _git("rev-parse", "HEAD"),
            "measurement_dirty": bool(_git("status", "--porcelain")),
            "measurement_date": datetime.now(timezone.utc).isoformat(),
            "source_commits": commits,
            "generator": "benchmarks/winding_surface_optimization.py",
            "command": " ".join(args.command),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "JAX_ENABLE_X64",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VMEX_COMPILATION_CACHE",
                    "XLA_FLAGS",
                )
            },
        },
        "configuration": {
            "repeats": args.repeats,
            "context_repeats": args.context_repeats,
            "cases": [asdict(CASES[name]) for name in args.cases],
            "resolutions": args.resolutions,
            "context_cases": args.context_cases,
            "context_resolutions": args.context_resolutions,
            "field_periods": 2,
            "active_dofs": 25,
            "timing_protocol": "isolated compile; one untimed warmup; all warm samples retained",
            "fidelity_protocol": {
                "standalone": "original unsplit full SVD at identical inputs",
                "optimization_context": (
                    "original full SVD patched only from maxiter=4 to maxiter=100; "
                    "25x25 optimality Jacobian solved directly in float64"
                ),
            },
        },
        "standalone": standalone,
        "optimization_context": context,
    }


COLORS = {
    "original": "#6b7280",
    "current": "#2563eb",
    "new": "#059669",
    "full_svd_100_reference": "#111827",
}


def _labels(record: dict) -> list[str]:
    labels = []
    for row in record["standalone"]:
        if row["variant"] not in labels:
            labels.append(row["variant"])
    return labels


def _aggregate(rows: list[dict], metric: Callable[[dict], float]):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["resolution"]), []).append(metric(row))
    return grouped


def _finish_figure(fig, axes, labels: list[str], title: str) -> None:
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=max(1, len(labels)),
    )
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.86, bottom=0.09, hspace=0.38, wspace=0.28)


def plot_runtime(record: dict, path: Path) -> None:
    labels = _labels(record)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    panels = (
        (axes[0, 0], "Value", lambda row: row["timing"]["value"]["median_ms"]),
        (axes[0, 1], "Gradient", lambda row: row["timing"]["gradient"]["median_ms"]),
        (axes[1, 0], "Hessian-vector product", lambda row: row["timing"]["hvp"]["median_ms"]),
    )
    for axis, title, metric in panels:
        grouped = _aggregate(record["standalone"], metric)
        for label in labels:
            x = sorted({resolution for variant, resolution in grouped if variant == label})
            values = [grouped[label, resolution] for resolution in x]
            center = [statistics.median(item) for item in values]
            lower = [min(item) for item in values]
            upper = [max(item) for item in values]
            color = COLORS.get(label)
            axis.plot(x, center, "o-", label=label, color=color)
            axis.fill_between(x, lower, upper, alpha=0.13, color=color)
        axis.set_title(f"Standalone {title}")
        axis.set_xlabel("toroidal = poloidal resolution")
        axis.set_ylabel("warm median (ms)")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)

    axis = axes[1, 1]
    context_rows = [
        row for row in record["optimization_context"] if row["variant"] in labels
    ]
    grouped = _aggregate(
        context_rows,
        lambda row: row["timing"]["outer_gradient"]["mixed"]["median_ms"],
    )
    for label in labels:
        x = sorted({resolution for variant, resolution in grouped if variant == label})
        values = [grouped[label, resolution] for resolution in x]
        center = [statistics.median(item) for item in values]
        lower = [min(item) for item in values]
        upper = [max(item) for item in values]
        color = COLORS.get(label)
        axis.plot(x, center, "o-", label=label, color=color)
        axis.fill_between(x, lower, upper, alpha=0.13, color=color)
    axis.set_title("Through inner solve: mixed outer gradient")
    axis.set_xlabel("toroidal = poloidal resolution")
    axis.set_ylabel("warm median (ms)")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    _finish_figure(
        fig,
        axes,
        labels,
        "Winding objective runtime scaling (median and case range)",
    )
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_fidelity(record: dict, path: Path) -> None:
    labels = _labels(record)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    panels = (
        (
            axes[0, 0],
            "Standalone objective error",
            record["standalone"],
            lambda row: row["fidelity"]["value"]["absolute_max"],
        ),
        (
            axes[0, 1],
            "Standalone gradient error",
            record["standalone"],
            lambda row: row["fidelity"]["gradient"]["relative_l2"],
        ),
        (
            axes[1, 0],
            "Standalone HVP error",
            record["standalone"],
            lambda row: row["fidelity"]["hvp"]["relative_l2"],
        ),
        (
            axes[1, 1],
            "Nested mixed-gradient error",
            [
                row
                for row in record["optimization_context"]
                if row["variant"] in labels
            ],
            lambda row: row["fidelity"]["mixed_gradient"]["relative_l2"],
        ),
    )
    floor = np.finfo(float).eps
    for axis, title, rows, metric in panels:
        grouped = _aggregate(rows, lambda row: max(metric(row), floor))
        for label in labels:
            x = sorted({resolution for variant, resolution in grouped if variant == label})
            values = [grouped[label, resolution] for resolution in x]
            center = [statistics.median(item) for item in values]
            lower = [min(item) for item in values]
            upper = [max(item) for item in values]
            color = COLORS.get(label)
            axis.plot(x, center, "o-", label=label, color=color)
            axis.fill_between(x, lower, upper, alpha=0.13, color=color)
        axis.set_title(title)
        axis.set_xlabel("toroidal = poloidal resolution")
        axis.set_ylabel("absolute error" if "objective" in title else "relative L2 error")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    _finish_figure(
        fig,
        axes,
        labels,
        "Fidelity against full-SVD references (median and case range)",
    )
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _geomean(values: list[float]) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values)))))


def plot_speedup(record: dict, path: Path) -> None:
    labels = _labels(record)
    original = labels[0]
    comparisons = labels[1:]
    metrics = {
        "value": lambda row: row["timing"]["value"]["median_ms"],
        "gradient": lambda row: row["timing"]["gradient"]["median_ms"],
        "HVP": lambda row: row["timing"]["hvp"]["median_ms"],
    }
    lookup = {
        (row["variant"], row["case"], row["resolution"]): row
        for row in record["standalone"]
    }
    resolutions = sorted({row["resolution"] for row in record["standalone"]})
    fig, axes = plt.subplots(
        1, len(comparisons), figsize=(5.2 * max(1, len(comparisons)), 4.5), squeeze=False,
        constrained_layout=True,
    )
    for axis, comparison in zip(axes[0], comparisons):
        width = 0.23
        positions = np.arange(len(resolutions))
        for index, (metric_name, metric) in enumerate(metrics.items()):
            speedups = []
            for resolution in resolutions:
                samples = []
                for case in record["configuration"]["cases"]:
                    key = (case["name"], resolution)
                    baseline_row = lookup[original, *key]
                    comparison_row = lookup[comparison, *key]
                    samples.append(metric(baseline_row) / metric(comparison_row))
                speedups.append(_geomean(samples))
            axis.bar(
                positions + (index - 1) * width,
                speedups,
                width,
                label=metric_name,
            )
        axis.axhline(1.0, color="#111827", linewidth=1)
        axis.set_xticks(positions, resolutions)
        axis.set_xlabel("toroidal = poloidal resolution")
        axis.set_ylabel(f"{original} / {comparison} speedup")
        axis.set_title(comparison)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Speedup changes with quadrature resolution (geometric mean across cases)")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="LABEL=GIT_REF; repeat in chronological order",
    )
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE)
    parser.add_argument("--standalone-reference", default="original")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--resolutions", nargs="+", type=int, default=[8, 12, 16, 24])
    parser.add_argument(
        "--context-cases", nargs="+", choices=CASES, default=["nominal", "shaped"]
    )
    parser.add_argument(
        "--context-resolutions", nargs="+", type=int, default=[12, 16]
    )
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--context-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path)
    args = parser.parse_args()
    if not args.variant:
        args.variant = [f"original={DEFAULT_BASELINE}", f"current={DEFAULT_CURRENT}"]
    args.command = ["python", "benchmarks/winding_surface_optimization.py", *os.sys.argv[1:]]
    record = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if args.figure_prefix:
        args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
        plot_runtime(record, args.figure_prefix.with_name(args.figure_prefix.name + "_runtime.svg"))
        plot_fidelity(record, args.figure_prefix.with_name(args.figure_prefix.name + "_fidelity.svg"))
        plot_speedup(record, args.figure_prefix.with_name(args.figure_prefix.name + "_speedup.svg"))


if __name__ == "__main__":
    main()

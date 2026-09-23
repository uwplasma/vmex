#!/usr/bin/env python
"""Benchmark winding-surface objective revisions without importing the example.

The optimization example executes a full VMEX campaign at module import time, so
this benchmark loads only its pure helper functions from each requested git
revision.  It measures two deliberately separate contexts:

* ``standalone`` evaluates the entropy/PCA scalar, its gradient, and an HVP on
  fixed winding and plasma surfaces.  The unsplit full-matrix SVD is the fidelity
  reference.  ``standalone_pairwise`` repeats the same protocol for the sum of
  plasma clearance and winding self-approach, retaining the unreduced full pair
  set as its fidelity reference.
* ``optimization_context`` differentiates outer entropy, inverse-distance, and
  mixed observables through the inner L-BFGS-B winding-surface solve.  Patched
  copies of the original full-SVD implementation with ``maxiter=100`` provide
  independent dense references with both frozen and live normalization scales;
  optional five-point finite differences check the total outer derivative.
  The historical implementation itself remains unmodified at four iterations.

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
from jaxopt.linear_solve import solve_lu
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
    "_winding_native_gcrot_solve",
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


def _namespace(
    source: str,
    *,
    resolution: int,
    reference: bool = False,
    live_scales: bool = False,
) -> dict:
    if live_scales:
        marker = "scales = jax.lax.stop_gradient(scales)"
        if marker not in source:
            raise ValueError("live-scale reference patch target was not found")
        source = source.replace(marker, "scales = scales", 1)
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
        "solve_lu": solve_lu,
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
        100
        if reference or "maxiter=WINDING_MAXITER" in source
        else 200
        if "maxiter=200" in source
        else 4
    )
    namespace["_benchmark_reverse"] = (
        "_winding_linear_solve" in namespace
        or "implicit_diff_solve=solve_lu" in source
        or "implicit_diff_solve=_winding_native_gcrot_solve" in source
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


def _standalone_function(
    namespace: dict, winding, plasma, objective: str = "spectral_entropy"
):
    winding_rc = namespace["surface_coefficients_from_dofs"](winding, 2, 2)[0]
    weights = namespace["mode_weights_from_coefficients"](winding_rc)

    def scalar(value):
        winding_value, plasma_value = jnp.split(value, 2)
        objectives = _variant_objectives(
            namespace, winding_value, plasma_value, weights, winding_rc
        )
        if objective == "spectral_entropy":
            return objectives[0]
        if objective == "pairwise_geometry":
            return objectives[3] + objectives[6]
        raise ValueError(f"unknown standalone objective {objective!r}")

    return scalar


def _standalone_row(
    label: str,
    namespace: dict,
    case: GeometryCase,
    resolution: int,
    repeats: int,
    objective_name: str = "spectral_entropy",
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    winding, plasma = _dofs(case)
    value = jnp.concatenate((winding, plasma))
    rng = np.random.default_rng(case.seed + 10_000 + resolution)
    direction = jnp.asarray(rng.normal(size=value.shape))
    direction /= jnp.linalg.norm(direction)
    scalar = _standalone_function(namespace, winding, plasma, objective_name)
    gradient = jax.grad(scalar)

    value_timing, objective_value = _measure(
        lambda: (jax.jit(scalar), (value,)), repeats
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
        "value": np.asarray(objective_value),
        "gradient": np.asarray(derivative),
        "hvp": np.asarray(hvp),
    }
    row = {
        "case": case.name,
        "resolution": resolution,
        "matrix_order": resolution * resolution,
        "variant": label,
        "objective": objective_name,
        "timing": {
            "value": value_timing,
            "gradient": gradient_timing,
            "hvp": hvp_timing,
        },
        "outputs": {
            "value": float(objective_value),
            "gradient_l2": float(jnp.linalg.norm(derivative)),
            "hvp_l2": float(jnp.linalg.norm(hvp)),
        },
    }
    return row, outputs


def _context_functions(namespace: dict, case: GeometryCase):
    base_winding, base_plasma = _dofs(case)
    active_indices = tuple(range(base_winding.size))
    active_array = jnp.asarray(active_indices)
    normalization_winding_rc = namespace["surface_coefficients_from_dofs"](
        base_winding, 2, 2
    )[0]
    normalization_weights = namespace["mode_weights_from_coefficients"](
        normalization_winding_rc
    )
    normalization_scales = jax.tree.map(
        jax.lax.stop_gradient,
        _variant_objectives(
            namespace,
            base_winding,
            base_plasma,
            normalization_weights,
            normalization_winding_rc,
        ),
    )

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
        # Fixed base-point factors put entropy and inverse distance on similar
        # scales without changing the mathematical derivative of the sampled
        # outer map.  Recomputing these factors at every ``delta`` and then
        # applying stop_gradient would make AD inconsistent with finite
        # differences of the values returned by this benchmark.
        entropy_normalized = objectives[0] / normalization_scales[0]
        inverse_distance_normalized = normalization_scales[3] / objectives[3]
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
    directional_steps: tuple[float, ...] = (),
    directional_directions: int = 0,
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
    if namespace.get("_benchmark_reverse", False):
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
    compiled_observables = jax.jit(observables)
    observable_values = np.asarray(_tree_block(compiled_observables(delta)))
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
    directional_fidelity = []
    rng = np.random.default_rng(case.seed + 20_000 + resolution)
    for direction_index in range(directional_directions):
        direction = rng.normal(size=delta.shape)
        direction /= np.linalg.norm(direction)
        direction_array = jnp.asarray(direction)
        ad_directional = {
            name: float(np.vdot(gradient_value, direction))
            for name, gradient_value in gradients.items()
        }
        for step in directional_steps:
            plus_two = np.asarray(_tree_block(compiled_observables(
                2 * step * direction_array
            )))
            plus_one = np.asarray(_tree_block(compiled_observables(
                step * direction_array
            )))
            minus_one = np.asarray(_tree_block(compiled_observables(
                -step * direction_array
            )))
            minus_two = np.asarray(_tree_block(compiled_observables(
                -2 * step * direction_array
            )))
            fd_observables = (
                -plus_two + 8 * plus_one - 8 * minus_one + minus_two
            ) / (12 * step)
            fd_directional = {
                "entropy": float(fd_observables[0]),
                "inverse_distance": float(fd_observables[1]),
                "mixed": float(np.vdot(
                    np.asarray(OBJECTIVE_WEIGHTS["mixed"]), fd_observables
                )),
            }
            directional_fidelity.append({
                "direction": direction_index,
                "step": step,
                "ad": ad_directional,
                "finite_difference": fd_directional,
                "error": {
                    name: {
                        "absolute": abs(ad_directional[name] - value),
                        "relative": abs(ad_directional[name] - value)
                        / max(abs(value), 1e-300),
                    }
                    for name, value in fd_directional.items()
                },
            })
    row["directional_fidelity"] = directional_fidelity
    return row, outputs


def _with_fidelity(row: dict, outputs: dict, reference: dict) -> dict:
    row = dict(row)
    row["fidelity"] = {
        name: _error(value, reference[name]) for name, value in outputs.items()
    }
    return row


def _with_reference_fidelities(
    row: dict,
    outputs: dict,
    references: dict[str, dict[str, np.ndarray]],
    primary: str,
) -> dict:
    """Attach a primary fidelity result and every named reference result."""
    row = _with_fidelity(row, outputs, references[primary])
    row["fidelity_by_reference"] = {
        name: {
            quantity: _error(value, reference[quantity])
            for quantity, value in outputs.items()
        }
        for name, reference in references.items()
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

    standalone_pairwise = []
    for case_name in args.cases:
        case = CASES[case_name]
        for resolution in args.resolutions:
            rows = {}
            outputs = {}
            for label, source in sources.items():
                print(
                    f"standalone-pairwise case={case.name} "
                    f"resolution={resolution} variant={label}",
                    flush=True,
                )
                namespace = _namespace(source, resolution=resolution)
                rows[label], outputs[label] = _standalone_row(
                    label,
                    namespace,
                    case,
                    resolution,
                    args.repeats,
                    objective_name="pairwise_geometry",
                )
            reference = outputs[args.standalone_reference]
            standalone_pairwise.extend(
                _with_fidelity(rows[label], outputs[label], reference)
                for label in variants
            )

    context_labels = args.context_variants or list(sources)
    unknown_context_labels = sorted(set(context_labels) - set(sources))
    if unknown_context_labels:
        raise ValueError(
            "unknown --context-variants labels: "
            + ", ".join(unknown_context_labels)
        )

    context = []
    for case_name in args.context_cases:
        case = CASES[case_name]
        for resolution in args.context_resolutions:
            reference_rows = {}
            reference_outputs = {}
            reference_modes = (
                ("frozen_scale", False),
                ("live_scale", True),
            ) if args.dual_scale_reference else (
                (args.context_reference_scale,
                 args.context_reference_scale == "live_scale"),
            )
            for reference_name, live_scales in reference_modes:
                print(
                    f"context case={case.name} resolution={resolution} "
                    f"variant=full_svd_100_{reference_name}_reference",
                    flush=True,
                )
                reference_namespace = _namespace(
                    baseline_source,
                    resolution=resolution,
                    reference=True,
                    live_scales=live_scales,
                )
                reference_namespace["_benchmark_reference"] = True
                label = f"full_svd_100_{reference_name}_reference"
                reference_rows[reference_name], reference_outputs[reference_name] = (
                    _context_row(
                        label,
                        reference_namespace,
                        case,
                        resolution,
                        args.context_repeats,
                        tuple(args.directional_steps),
                        args.directional_directions,
                    )
                )
            for reference_name, row in reference_rows.items():
                context.append(_with_reference_fidelities(
                    row,
                    reference_outputs[reference_name],
                    reference_outputs,
                    args.context_reference_scale,
                ))
            for label in context_labels:
                source = sources[label]
                print(
                    f"context case={case.name} resolution={resolution} variant={label}",
                    flush=True,
                )
                namespace = _namespace(source, resolution=resolution)
                row, outputs = _context_row(
                    label,
                    namespace,
                    case,
                    resolution,
                    args.context_repeats,
                    tuple(args.directional_steps),
                    args.directional_directions,
                )
                context.append(_with_reference_fidelities(
                    row,
                    outputs,
                    reference_outputs,
                    args.context_reference_scale,
                ))

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
            "context_variants": context_labels,
            "field_periods": 2,
            "active_dofs": 25,
            "timing_protocol": "isolated compile; one untimed warmup; all warm samples retained",
            "fidelity_protocol": {
                "standalone": "original unsplit full SVD at identical inputs",
                "standalone_pairwise": (
                    "original full pair sets at identical inputs"
                ),
                "optimization_context": (
                    "original full SVD patched from maxiter=4 to maxiter=100; "
                    "25x25 optimality Jacobian solved directly in float64; "
                    f"primary normalization-scale derivative mode is "
                    f"{args.context_reference_scale}"
                ),
            },
            "context_reference_scale": args.context_reference_scale,
            "dual_scale_reference": args.dual_scale_reference,
            "directional_steps": args.directional_steps,
            "directional_directions": args.directional_directions,
        },
        "standalone": standalone,
        "standalone_pairwise": standalone_pairwise,
        "optimization_context": context,
    }


COLORS = {
    "original": "#6b7280",
    "current": "#2563eb",
    "new": "#059669",
    "pr366": "#2563eb",
    "pr367": "#d97706",
    "combined": "#059669",
    "live_gcrot": "#059669",
    "native_lu": "#7c3aed",
    "native_gcrot": "#dc2626",
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


def plot_pairwise(record: dict, path: Path) -> None:
    """Plot isolated pairwise runtime and fidelity at every resolution."""
    labels = _labels(record)
    rows = record["standalone_pairwise"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    timing_panels = (
        (axes[0, 0], "Value", lambda row: row["timing"]["value"]["median_ms"]),
        (
            axes[0, 1],
            "Gradient",
            lambda row: row["timing"]["gradient"]["median_ms"],
        ),
        (axes[0, 2], "HVP", lambda row: row["timing"]["hvp"]["median_ms"]),
    )
    fidelity_panels = (
        (
            axes[1, 0],
            "Value error",
            lambda row: row["fidelity"]["value"]["absolute_max"],
        ),
        (
            axes[1, 1],
            "Gradient error",
            lambda row: row["fidelity"]["gradient"]["relative_l2"],
        ),
        (
            axes[1, 2],
            "HVP error",
            lambda row: row["fidelity"]["hvp"]["relative_l2"],
        ),
    )
    for axis, title, metric in timing_panels:
        grouped = _aggregate(rows, metric)
        for label in labels:
            x = sorted({resolution for variant, resolution in grouped if variant == label})
            values = [grouped[label, resolution] for resolution in x]
            color = COLORS.get(label)
            axis.plot(
                x, [statistics.median(item) for item in values], "o-",
                label=label, color=color,
            )
            axis.fill_between(
                x, [min(item) for item in values], [max(item) for item in values],
                alpha=0.13, color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("toroidal = poloidal resolution")
        axis.set_ylabel("warm median (ms)")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)

    floor = np.finfo(float).eps
    for axis, title, metric in fidelity_panels:
        grouped = _aggregate(rows, lambda row: max(metric(row), floor))
        for label in labels:
            x = sorted({resolution for variant, resolution in grouped if variant == label})
            values = [grouped[label, resolution] for resolution in x]
            color = COLORS.get(label)
            axis.plot(
                x, [statistics.median(item) for item in values], "o-",
                label=label, color=color,
            )
            axis.fill_between(
                x, [min(item) for item in values], [max(item) for item in values],
                alpha=0.13, color=color,
            )
        axis.set_title(title)
        axis.set_xlabel("toroidal = poloidal resolution")
        axis.set_ylabel("absolute error" if "Value" in title else "relative L2 error")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    _finish_figure(
        fig,
        axes,
        labels,
        "Pairwise geometry runtime and full-pair fidelity (median and case range)",
    )
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_directional_fidelity(record: dict, path: Path) -> None:
    """Plot AD agreement with five-point finite differences."""
    labels = _labels(record)
    rows = [
        row for row in record["optimization_context"]
        if row["variant"] in labels and row["directional_fidelity"]
    ]
    if not rows:
        return
    step = record["configuration"]["directional_steps"][0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    for axis, objective in zip(axes, OBJECTIVE_WEIGHTS):
        grouped = {}
        for row in rows:
            values = [
                item["error"][objective]["relative"]
                for item in row["directional_fidelity"]
                if item["step"] == step
            ]
            grouped.setdefault(
                (row["variant"], row["resolution"]), []
            ).extend(values)
        for label in labels:
            resolutions = sorted(
                resolution for variant, resolution in grouped
                if variant == label
            )
            samples = [grouped[label, resolution] for resolution in resolutions]
            color = COLORS.get(label)
            axis.plot(
                resolutions,
                [statistics.median(values) for values in samples],
                "o-",
                label=label,
                color=color,
            )
            axis.fill_between(
                resolutions,
                [min(values) for values in samples],
                [max(values) for values in samples],
                alpha=0.13,
                color=color,
            )
        axis.set_title(objective.replace("_", " ").title())
        axis.set_xlabel("toroidal = poloidal resolution")
        axis.set_ylabel("relative directional error")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=len(labels),
    )
    fig.suptitle(
        f"Outer-gradient fidelity against five-point finite differences "
        f"(step={step:g})",
        y=1.12,
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
    parser.add_argument(
        "--context-variants",
        nargs="+",
        help=(
            "variant labels to measure through the inner solve; defaults to "
            "every --variant label"
        ),
    )
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--context-repeats", type=int, default=3)
    parser.add_argument(
        "--context-reference-scale",
        choices=("frozen_scale", "live_scale"),
        default="frozen_scale",
    )
    parser.add_argument("--dual-scale-reference", action="store_true")
    parser.add_argument("--directional-steps", nargs="*", type=float, default=[])
    parser.add_argument("--directional-directions", type=int, default=0)
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
        plot_pairwise(record, args.figure_prefix.with_name(args.figure_prefix.name + "_pairwise.svg"))
        plot_directional_fidelity(
            record,
            args.figure_prefix.with_name(
                args.figure_prefix.name + "_directional.svg"
            ),
        )


if __name__ == "__main__":
    main()

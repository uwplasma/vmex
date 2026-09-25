#!/usr/bin/env python
"""Controlled comparison of winding methods from PRs 301, 303, 304, and 366.

The older pull requests form a stack and predate the current winding example,
so timing their branch heads wholesale would mix algorithm changes with changed
objectives, bounds, active modes, and optimizer wiring.  This script instead
does two things:

* it executes the exact geometry helpers from the historical revisions; and
* it ports each spectral/linear-solve method into the same PR-366 objective
  scaffold for causal standalone and nested comparisons.

All timings retain every sample.  Value, gradient, HVP, root-response, and
nested outer-gradient fidelity are compared with full-matrix/dense references.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Callable

import jax
import jax.numpy as jnp
from jaxopt.linear_solve import solve_normal_cg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from solvax import gcrot

import winding_surface_optimization as common


REPO = Path(__file__).resolve().parents[1]
OLD_EXAMPLE = "examples/optimization/QA_optimization.py"
CURRENT_EXAMPLE = "examples/optimization/QA_winding_optimization.py"
DEFAULT_REFS = {
    "pre_301": "913377b6",
    "pr301": "origin/pr-301",
    "pr303": "origin/pr-303",
    "pr304": "origin/pr-304",
    "pr366": "origin/ds/winding-adjoint-performance",
}

GEOMETRY_FUNCTIONS = {
    "_surface_series",
    "_mode_angle",
    "r_surface",
    "z_surface",
    "r_surface_prime_phi",
    "z_surface_prime_phi",
    "surface_del_phi",
    "r_surface_prime_theta",
    "z_surface_prime_theta",
    "surface_del_theta",
    "surface_normal",
    "_surface_point_and_normal",
    "surface_coefficients_from_dofs",
    "points_normals_normal_lengths",
}

COLORS = {
    "pre_301_loops": "#6b7280",
    "pr301_contraction": "#d97706",
    "pr366_contraction": "#2563eb",
    "fused_geometry_candidate": "#7c3aed",
    "pr301_pr303_full_svd": "#6b7280",
    "pr304_full_matrix_sectors": "#d97706",
    "pr366_one_block_sectors": "#2563eb",
    "normal_cg": "#6b7280",
    "pr303_gcrot": "#d97706",
    "pr366_explicit_certificate": "#2563eb",
    "returned_residual_certificate": "#059669",
    "pr301_port": "#6b7280",
    "pr303_port": "#d97706",
    "pr304_port": "#7c3aed",
    "pr366": "#2563eb",
    "combined": "#059669",
    "full_svd_dense_reference": "#111827",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source(ref: str, path: str) -> tuple[str, str]:
    commit = _git("rev-parse", ref)
    return commit, _git("show", f"{commit}:{path}")


def _selected_source(source: str, names: set[str], filename: str) -> str:
    tree = ast.parse(source, filename=filename)
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    return ast.unparse(tree) + "\n"


def _geometry_namespace(source: str, filename: str) -> tuple[dict, str]:
    selected = _selected_source(source, GEOMETRY_FUNCTIONS, filename)
    namespace = {"jax": jax, "jnp": jnp}
    exec(compile(selected, filename, "exec"), namespace)
    return namespace, hashlib.sha256(selected.encode()).hexdigest()


def _scalar_row(
        label: str, scalar: Callable, value: jax.Array, direction: jax.Array,
        repeats: int, metadata: dict) -> tuple[dict, dict[str, np.ndarray]]:
    gradient = jax.grad(scalar)
    value_timing, objective = common._measure(
        lambda: (jax.jit(scalar), (value,)), repeats)
    gradient_timing, derivative = common._measure(
        lambda: (jax.jit(gradient), (value,)), repeats)
    hvp_timing, hvp = common._measure(
        lambda: (
            jax.jit(lambda argument, tangent: jax.jvp(
                gradient, (argument,), (tangent,))[1]),
            (value, direction),
        ), repeats)
    outputs = {
        "value": np.asarray(objective),
        "gradient": np.asarray(derivative),
        "hvp": np.asarray(hvp),
    }
    row = {
        **metadata,
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


def _attach_fidelity(row: dict, outputs: dict, reference: dict) -> dict:
    row = dict(row)
    row["fidelity"] = {
        key: common._error(value, reference[key])
        for key, value in outputs.items()
    }
    return row


def _geometry_scalar(namespace: dict, resolution: int) -> Callable:
    points = namespace["points_normals_normal_lengths"]

    def scalar(dofs):
        xyz, normals, lengths, raw_normals = points(
            dofs, 2, 2, 2, resolution, resolution)
        # All four returned fields contribute.  Mean scaling keeps the
        # derivative magnitudes comparable as quadrature resolution changes.
        return (jnp.mean(xyz ** 2)
                + 0.3 * jnp.mean(normals ** 2)
                + 0.2 * jnp.mean(lengths)
                + 0.1 * jnp.mean(raw_normals ** 2))

    return scalar


def _full_spectrum(namespace: dict, winding_points, plasma_points,
                   dipole_normals, plasma_normals, winding_weights,
                   plasma_weights, field_periods, tor_num, pol_num):
    del field_periods, tor_num, pol_num
    induction = namespace["reduced_memory_induction_matrix"](
        winding_points, plasma_points, dipole_normals, plasma_normals,
        winding_weights, plasma_weights)
    return jnp.linalg.svd(induction, compute_uv=False)


def _pr304_spectrum(namespace: dict, winding_points, plasma_points,
                    dipole_normals, plasma_normals, winding_weights,
                    plasma_weights, field_periods, tor_num, pol_num):
    induction = namespace["reduced_memory_induction_matrix"](
        winding_points, plasma_points, dipole_normals, plasma_normals,
        winding_weights, plasma_weights)
    if field_periods == 2 and tor_num % 2 == 0:
        half = induction.shape[0] // 2
        a = induction[:half, :half]
        b = induction[:half, half:]
        return jnp.concatenate((
            jnp.linalg.svd(a + b, compute_uv=False),
            jnp.linalg.svd(a - b, compute_uv=False),
        ))
    return jnp.linalg.svd(induction, compute_uv=False)


def _install_spectrum(namespace: dict, method: str) -> None:
    if method == "full":
        namespace["_periodic_induction_singular_values"] = (
            lambda *args: _full_spectrum(namespace, *args))
    elif method == "pr304":
        namespace["_periodic_induction_singular_values"] = (
            lambda *args: _pr304_spectrum(namespace, *args))
    elif method != "pr366":
        raise ValueError(f"unknown spectrum method {method!r}")


def _spectrum_scalar(namespace: dict, resolution: int) -> Callable:
    points = namespace["points_normals_normal_lengths"]
    weights = namespace["surface_quadrature_weights"]
    spectrum = namespace["_periodic_induction_singular_values"]

    def scalar(value):
        winding, plasma = jnp.split(value, 2)
        wp, wn, wj, _ = points(winding, 2, 2, 2, resolution, resolution)
        pp, pn, pj, _ = points(plasma, 2, 2, 2, resolution, resolution)
        singular_values = spectrum(
            wp.reshape((-1, 3)), pp.reshape((-1, 3)),
            wn.reshape((-1, 3)), pn.reshape((-1, 3)),
            weights(wj, resolution, resolution),
            weights(pj, resolution, resolution),
            2, resolution, resolution)
        probabilities = singular_values / jnp.sum(singular_values)
        entropy = -jnp.sum(
            probabilities * jnp.log(jnp.maximum(probabilities, 1e-300)))
        return 1 / jnp.maximum(entropy, 1e-16)

    return scalar


def _pr303_linear_solve(matvec, rhs, *, rtol=1e-8, max_restarts=10):
    _, operator = jax.linearize(matvec, jnp.zeros_like(rhs))

    def solve(action, value):
        result = gcrot(
            action, value, rtol=rtol, max_restarts=max_restarts)
        return jnp.where(result.converged, result.x, jnp.nan)

    return jax.lax.custom_linear_solve(
        operator, rhs, solve=solve, transpose_solve=solve)


def _pr366_linear_solve(matvec, rhs, *, rtol=1e-8, max_restarts=10):
    _, operator = jax.linearize(matvec, jnp.zeros_like(rhs))

    def solve(action, value):
        result = gcrot(
            action, value, rtol=rtol, max_restarts=max_restarts)
        residual = action(result.x) - value
        certified = result.converged & (
            jnp.linalg.norm(residual) <= rtol * jnp.linalg.norm(value))
        return jnp.where(certified, result.x, jnp.nan)

    return jax.lax.custom_linear_solve(
        operator, rhs, solve=solve, transpose_solve=solve)


def _returned_residual_linear_solve(
        matvec, rhs, *, rtol=1e-8, max_restarts=10):
    _, operator = jax.linearize(matvec, jnp.zeros_like(rhs))

    def solve(action, value):
        result = gcrot(
            action, value, rtol=rtol, max_restarts=max_restarts)
        certified = result.converged & (
            result.residual_norm <= rtol * jnp.linalg.norm(value))
        return jnp.where(certified, result.x, jnp.nan)

    return jax.lax.custom_linear_solve(
        operator, rhs, solve=solve, transpose_solve=solve)


def _normal_cg_linear_solve(matvec, rhs):
    return solve_normal_cg(matvec, rhs)


LINEAR_METHODS = {
    "normal_cg": _normal_cg_linear_solve,
    "pr303_gcrot": _pr303_linear_solve,
    "pr366_explicit_certificate": _pr366_linear_solve,
    "returned_residual_certificate": _returned_residual_linear_solve,
}


def _linear_cases() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(304)
    q, _ = np.linalg.qr(rng.normal(size=(25, 25)))
    rhs = rng.normal(size=25)
    cotangent = rng.normal(size=25)
    return {
        "moderate_spd": (
            (q * np.geomspace(1.0, 1e2, 25)) @ q.T, rhs, cotangent),
        "ill_conditioned_spd": (
            (q * np.geomspace(1.0, 1e6, 25)) @ q.T, rhs, cotangent),
        "indefinite": (
            (q * np.r_[
                -np.geomspace(1.0, 1e3, 12),
                np.geomspace(1.0, 1e3, 13),
            ]) @ q.T, rhs, cotangent),
    }


def _linear_row(label: str, solve: Callable, case_name: str,
                matrix: np.ndarray, rhs: np.ndarray, cotangent: np.ndarray,
                repeats: int) -> dict:
    matrix_array = jnp.asarray(matrix)
    rhs_array = jnp.asarray(rhs)
    cotangent_array = jnp.asarray(cotangent)

    def response(value):
        return solve(lambda vector: matrix_array @ vector, value)

    def tangent(value, direction):
        return jax.jvp(response, (value,), (direction,))[1]

    def reverse(value, weight):
        return jax.grad(lambda argument: jnp.vdot(
            response(argument), weight))(value)

    solve_timing, solution = common._measure(
        lambda: (jax.jit(response), (rhs_array,)), repeats)
    jvp_timing, jvp_value = common._measure(
        lambda: (jax.jit(tangent), (rhs_array, cotangent_array)), repeats)
    vjp_timing, vjp_value = common._measure(
        lambda: (jax.jit(reverse), (rhs_array, cotangent_array)), repeats)

    expected = np.linalg.solve(matrix, rhs)
    expected_jvp = np.linalg.solve(matrix, cotangent)
    expected_vjp = np.linalg.solve(matrix.T, cotangent)
    actual = np.asarray(solution)
    return {
        "case": case_name,
        "order": int(matrix.shape[0]),
        "condition_number_2": float(np.linalg.cond(matrix)),
        "variant": label,
        "timing": {
            "solve": solve_timing,
            "jvp": jvp_timing,
            "vjp": vjp_timing,
        },
        "fidelity": {
            "solution": common._error(actual, expected),
            "jvp": common._error(np.asarray(jvp_value), expected_jvp),
            "vjp": common._error(np.asarray(vjp_value), expected_vjp),
            "relative_residual": float(
                np.linalg.norm(matrix @ actual - rhs)
                / max(np.linalg.norm(rhs), 1e-300)),
        },
    }


def _context_namespace(source: str, resolution: int, spectrum_method: str,
                       linear_method: str, reverse: bool,
                       *, dense_reference: bool = False) -> dict:
    namespace = common._namespace(
        source, resolution=resolution, reference=dense_reference)
    _install_spectrum(namespace, spectrum_method)
    if not dense_reference:
        namespace["_winding_linear_solve"] = LINEAR_METHODS[linear_method]
    namespace["_benchmark_reverse"] = reverse
    return namespace


CONTEXT_METHODS = {
    # Every arm uses the current objective, active modes, LBFGSB bounds, and
    # maxiter=100.  Only the reviewed PR methods vary.
    "pr301_port": ("full", "normal_cg", False),
    "pr303_port": ("full", "pr303_gcrot", True),
    "pr304_port": ("pr304", "pr303_gcrot", True),
    "pr366": ("pr366", "pr366_explicit_certificate", True),
    "combined": ("pr366", "returned_residual_certificate", True),
}


def _context_rows(source: str, cases: list[str], resolutions: list[int],
                  repeats: int, directional_steps: tuple[float, ...],
                  directional_directions: int) -> list[dict]:
    rows = []
    for case_name in cases:
        case = common.CASES[case_name]
        for resolution in resolutions:
            print(
                f"context case={case_name} resolution={resolution} "
                "variant=full_svd_dense_reference", flush=True)
            reference_namespace = _context_namespace(
                source, resolution, "full", "pr366_explicit_certificate",
                True, dense_reference=True)
            reference_row, reference_outputs = common._context_row(
                "full_svd_dense_reference", reference_namespace, case,
                resolution, repeats, directional_steps,
                directional_directions)
            rows.append(common._with_fidelity(
                reference_row, reference_outputs, reference_outputs))

            for label, (spectrum, linear, reverse) in CONTEXT_METHODS.items():
                print(
                    f"context case={case_name} resolution={resolution} "
                    f"variant={label}", flush=True)
                namespace = _context_namespace(
                    source, resolution, spectrum, linear, reverse)
                row, outputs = common._context_row(
                    label, namespace, case, resolution, repeats,
                    directional_steps if label in {"pr366", "combined"} else (),
                    directional_directions if label in {"pr366", "combined"} else 0)
                rows.append(common._with_fidelity(
                    row, outputs, reference_outputs))
    return rows


def run(args: argparse.Namespace) -> dict:
    jax.config.update("jax_enable_x64", True)
    refs = dict(DEFAULT_REFS)
    refs.update(item.split("=", 1) for item in args.ref)
    commits = {}
    sources = {}
    for label, ref in refs.items():
        path = OLD_EXAMPLE if label in {
            "pre_301", "pr301", "pr303", "pr304"} else CURRENT_EXAMPLE
        commits[label], sources[label] = _source(ref, path)

    geometry_sources = {}
    geometry_hashes = {}
    for label in ("pre_301", "pr301", "pr303", "pr304", "pr366"):
        path = OLD_EXAMPLE if label != "pr366" else CURRENT_EXAMPLE
        namespace, digest = _geometry_namespace(sources[label], path)
        geometry_sources[label] = namespace
        geometry_hashes[label] = digest
    if args.fused_ref:
        commits["fused_geometry_candidate"], fused_source = _source(
            args.fused_ref, CURRENT_EXAMPLE)
        namespace, digest = _geometry_namespace(
            fused_source, CURRENT_EXAMPLE)
        geometry_sources["fused_geometry_candidate"] = namespace
        geometry_hashes["fused_geometry_candidate"] = digest

    # PRs 301, 303, and 304 contain byte-identical selected geometry helpers.
    if len({geometry_hashes[label] for label in ("pr301", "pr303", "pr304")}) != 1:
        raise RuntimeError("stacked PR geometry helpers unexpectedly differ")
    geometry_variants = {
        "pre_301_loops": geometry_sources["pre_301"],
        "pr301_contraction": geometry_sources["pr301"],
        "pr366_contraction": geometry_sources["pr366"],
    }
    if args.fused_ref:
        geometry_variants["fused_geometry_candidate"] = (
            geometry_sources["fused_geometry_candidate"])

    geometry_rows = []
    for case_name in args.cases:
        case = common.CASES[case_name]
        winding, _ = common._dofs(case)
        for resolution in args.resolutions:
            rng = np.random.default_rng(case.seed + 30_100 + resolution)
            direction = jnp.asarray(rng.normal(size=winding.shape))
            direction /= jnp.linalg.norm(direction)
            measured = {}
            for label, namespace in geometry_variants.items():
                print(
                    f"geometry case={case_name} resolution={resolution} "
                    f"variant={label}", flush=True)
                row, outputs = _scalar_row(
                    label, _geometry_scalar(namespace, resolution), winding,
                    direction, args.repeats, {
                        "case": case_name,
                        "resolution": resolution,
                        "quadrature_points": resolution ** 2,
                    })
                measured[label] = (row, outputs)
            reference = measured["pre_301_loops"][1]
            geometry_rows.extend(
                _attach_fidelity(row, outputs, reference)
                for row, outputs in measured.values())

    _, pr366_source = _source(refs["pr366"], CURRENT_EXAMPLE)
    spectrum_rows = []
    spectrum_variants = {
        "pr301_pr303_full_svd": "full",
        "pr304_full_matrix_sectors": "pr304",
        "pr366_one_block_sectors": "pr366",
    }
    for case_name in args.cases:
        case = common.CASES[case_name]
        winding, plasma = common._dofs(case)
        value = jnp.concatenate((winding, plasma))
        for resolution in args.resolutions:
            rng = np.random.default_rng(case.seed + 30_400 + resolution)
            direction = jnp.asarray(rng.normal(size=value.shape))
            direction /= jnp.linalg.norm(direction)
            measured = {}
            for label, method in spectrum_variants.items():
                print(
                    f"spectrum case={case_name} resolution={resolution} "
                    f"variant={label}", flush=True)
                namespace = common._namespace(
                    pr366_source, resolution=resolution)
                _install_spectrum(namespace, method)
                row, outputs = _scalar_row(
                    label, _spectrum_scalar(namespace, resolution), value,
                    direction, args.repeats, {
                        "case": case_name,
                        "resolution": resolution,
                        "matrix_order": resolution ** 2,
                    })
                measured[label] = (row, outputs)
            reference = measured["pr301_pr303_full_svd"][1]
            spectrum_rows.extend(
                _attach_fidelity(row, outputs, reference)
                for row, outputs in measured.values())

    linear_rows = []
    for case_name, (matrix, rhs, cotangent) in _linear_cases().items():
        for label, solve in LINEAR_METHODS.items():
            print(f"linear case={case_name} variant={label}", flush=True)
            linear_rows.append(_linear_row(
                label, solve, case_name, matrix, rhs, cotangent,
                args.repeats))

    context_rows = _context_rows(
        pr366_source, args.context_cases, args.context_resolutions,
        args.context_repeats, tuple(args.directional_steps),
        args.directional_directions)

    return {
        "_provenance": {
            "measurement_commit": _git("rev-parse", "HEAD"),
            "measurement_dirty": bool(_git("status", "--porcelain")),
            "measurement_date": datetime.now(timezone.utc).isoformat(),
            "source_commits": commits,
            "generator": "benchmarks/winding_surface_pr301_304.py",
            "command": " ".join(args.command),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None,
            "environment": {
                name: os.environ.get(name) for name in (
                    "JAX_ENABLE_X64", "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VMEX_COMPILATION_CACHE", "XLA_FLAGS",
                )
            },
        },
        "review": {
            "stack_order": ["pr301", "pr303", "pr304"],
            "geometry_helper_sha256": geometry_hashes,
            "controlled_port": (
                "Every nested arm uses PR 366's objective, active-mode set, "
                "LBFGSB bounds, maxiter=100, and tolerances; only the reviewed "
                "geometry, spectrum, and root-linear-solve methods vary."
            ),
            "method_scope": {
                "pr301": "vectorized Fourier geometry contractions",
                "pr303": "cached direct-optimality-Jacobian GCROT solve",
                "pr304": "NFP=2 full-matrix A+B/A-B sector SVD",
                "pr366": (
                    "PR301-equivalent contractions, certified direct GCROT, "
                    "and arbitrary-NFP one-block-row sector SVD"
                ),
                "combined": (
                    "PR366 methods with Solvax's returned true-residual norm "
                    "used for the existing GCROT certificate"
                ),
            },
        },
        "configuration": {
            "repeats": args.repeats,
            "context_repeats": args.context_repeats,
            "cases": [common.asdict(common.CASES[name]) for name in args.cases],
            "resolutions": args.resolutions,
            "context_cases": args.context_cases,
            "context_resolutions": args.context_resolutions,
            "directional_steps": args.directional_steps,
            "directional_directions": args.directional_directions,
            "field_periods": 2,
            "active_dofs": 25,
            "timing_protocol": (
                "isolated compile; one untimed warmup; all warm samples retained"
            ),
        },
        "geometry": geometry_rows,
        "spectrum": spectrum_rows,
        "linear_solve": linear_rows,
        "optimization_context": context_rows,
    }


def _aggregate(rows: list[dict], labels: list[str], metric: Callable):
    grouped = {}
    for row in rows:
        if row["variant"] in labels:
            grouped.setdefault(
                (row["variant"], row.get("resolution", row["case"])),
                []).append(metric(row))
    return grouped


def _line_panel(axis, rows: list[dict], labels: list[str], metric: Callable,
                title: str, ylabel: str, *, log=True) -> None:
    grouped = _aggregate(rows, labels, metric)
    x_values = sorted({key[1] for key in grouped})
    for label in labels:
        samples = [grouped[label, x] for x in x_values]
        center = [statistics.median(values) for values in samples]
        lower = [min(values) for values in samples]
        upper = [max(values) for values in samples]
        color = COLORS[label]
        axis.plot(x_values, center, "o-", label=label, color=color)
        axis.fill_between(x_values, lower, upper, alpha=0.13, color=color)
    axis.set_title(title)
    axis.set_xlabel("toroidal = poloidal resolution")
    axis.set_ylabel(ylabel)
    if log:
        axis.set_yscale("log")
    axis.grid(alpha=0.25)


def plot_methods(record: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    geometry_labels = list(dict.fromkeys(
        row["variant"] for row in record["geometry"]))
    spectrum_labels = list(dict.fromkeys(
        row["variant"] for row in record["spectrum"]))
    for column, metric in enumerate(("value", "gradient", "hvp")):
        _line_panel(
            axes[0, column], record["geometry"], geometry_labels,
            lambda row, name=metric: row["timing"][name]["median_ms"],
            f"Geometry {metric}", "warm median (ms)")
        _line_panel(
            axes[1, column], record["spectrum"], spectrum_labels,
            lambda row, name=metric: row["timing"][name]["median_ms"],
            f"Entropy {metric}", "warm median (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Isolated PR-method runtime across resolutions", y=0.995)
    fig.subplots_adjust(top=0.82, hspace=0.33, wspace=0.27)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_fidelity(record: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    floor = np.finfo(float).eps
    geometry_labels = list(dict.fromkeys(
        row["variant"] for row in record["geometry"]))
    spectrum_labels = list(dict.fromkeys(
        row["variant"] for row in record["spectrum"]))
    for column, metric in enumerate(("value", "gradient", "hvp")):
        error_key = "absolute_max" if metric == "value" else "relative_l2"
        _line_panel(
            axes[0, column], record["geometry"], geometry_labels,
            lambda row, name=metric, key=error_key: max(
                row["fidelity"][name][key], floor),
            f"Geometry {metric} error",
            "absolute error" if metric == "value" else "relative L2 error")
        _line_panel(
            axes[1, column], record["spectrum"], spectrum_labels,
            lambda row, name=metric, key=error_key: max(
                row["fidelity"][name][key], floor),
            f"Entropy {metric} error",
            "absolute error" if metric == "value" else "relative L2 error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Value and derivative fidelity against direct references", y=0.995)
    fig.subplots_adjust(top=0.82, hspace=0.33, wspace=0.27)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_linear(record: dict, path: Path) -> None:
    rows = record["linear_solve"]
    labels = list(dict.fromkeys(row["variant"] for row in rows))
    cases = list(dict.fromkeys(row["case"] for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    x = np.arange(len(cases))
    width = 0.19
    for index, label in enumerate(labels):
        selected = {
            row["case"]: row for row in rows if row["variant"] == label}
        axes[0].bar(
            x + (index - 1.5) * width,
            [selected[case]["timing"]["solve"]["median_ms"] for case in cases],
            width, label=label, color=COLORS[label])
        axes[1].bar(
            x + (index - 1.5) * width,
            [max(selected[case]["fidelity"]["relative_residual"],
                 np.finfo(float).eps) for case in cases],
            width, label=label, color=COLORS[label])
        axes[2].bar(
            x + (index - 1.5) * width,
            [max(selected[case]["fidelity"]["vjp"]["relative_l2"],
                 np.finfo(float).eps) for case in cases],
            width, label=label, color=COLORS[label])
    for axis, title, ylabel in zip(
            axes,
            ("Direct solve runtime", "True residual", "Transpose response error"),
            ("warm median (ms)", "relative residual", "relative L2 error")):
        axis.set_xticks(x, [case.replace("_", "\n") for case in cases])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(axis="y", alpha=0.25)
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_context(record: dict, path: Path) -> None:
    rows = [row for row in record["optimization_context"]
            if row["variant"] != "full_svd_dense_reference"]
    labels = list(dict.fromkeys(row["variant"] for row in rows))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    panels = (
        (axes[0, 0], "Inner solve", lambda row: row["timing"]["inner_solve"]["median_ms"], "warm median (ms)"),
        (axes[0, 1], "Entropy outer gradient", lambda row: row["timing"]["outer_gradient"]["entropy"]["median_ms"], "warm median (ms)"),
        (axes[1, 0], "Mixed outer gradient", lambda row: row["timing"]["outer_gradient"]["mixed"]["median_ms"], "warm median (ms)"),
        (axes[1, 1], "Mixed-gradient fidelity", lambda row: max(row["fidelity"]["mixed_gradient"]["relative_l2"], np.finfo(float).eps), "relative L2 error"),
    )
    for axis, title, metric, ylabel in panels:
        _line_panel(axis, rows, labels, metric, title, ylabel)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Controlled methods inside the maxiter=100 optimization", y=0.995)
    fig.subplots_adjust(top=0.84, hspace=0.33, wspace=0.27)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref", action="append", default=[],
        help="override a source as LABEL=GIT_REF")
    parser.add_argument("--fused-ref")
    parser.add_argument(
        "--cases", nargs="+", choices=common.CASES,
        default=list(common.CASES))
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=[8, 12, 16, 24])
    parser.add_argument(
        "--context-cases", nargs="+", choices=common.CASES,
        default=["nominal", "shaped"])
    parser.add_argument(
        "--context-resolutions", nargs="+", type=int, default=[12, 16])
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--context-repeats", type=int, default=3)
    parser.add_argument(
        "--directional-steps", nargs="*", type=float, default=[3e-4, 1e-4])
    parser.add_argument("--directional-directions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path)
    args = parser.parse_args()
    args.command = [
        "python", "benchmarks/winding_surface_pr301_304.py", *os.sys.argv[1:]]
    record = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    if args.figure_prefix:
        args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
        plot_methods(record, args.figure_prefix.with_name(
            args.figure_prefix.name + "_methods.svg"))
        plot_fidelity(record, args.figure_prefix.with_name(
            args.figure_prefix.name + "_fidelity.svg"))
        plot_linear(record, args.figure_prefix.with_name(
            args.figure_prefix.name + "_linear.svg"))
        plot_context(record, args.figure_prefix.with_name(
            args.figure_prefix.name + "_context.svg"))


if __name__ == "__main__":
    main()

"""Where the radial spline knots belong: uniform in ``s`` or uniform in ``rho``.

``benchmarks/residual_vs_resolution.py`` measured the continuum residual of one
converged ``input.DSHAPE`` solve against the number of spline spans, with
breakpoints uniform in the flux label ``s``.  Refinement helped the edge and hurt
the axis: near-axis volume L2 rose monotonically from 1.0e3 to 1.3e4 N m^-3
between four and twelve spans while the edge fell from 1.3e7 to 5.1e2.

The representation is written ``rho^|m| q(s)`` with ``rho = sqrt(s)``
(``radial_basis.evaluate_regularized_mode``), and ``q`` is a clamped B-spline
whose *breakpoints are in* ``s`` (``BSplineBasis.clamped``).  Breakpoints uniform
in ``s`` are therefore sparse in ``rho`` near the axis: the first span of a
four-span basis covers ``s`` in ``[0, 0.25]``, which is the inner half of the
minor radius.  Grading them so they are uniform in ``rho`` -- breakpoints
``linspace(0, 1, spans + 1) ** 2`` in ``s`` -- moves spline freedom inward at an
identical coefficient count, since a clamped degree-``p`` basis on ``k`` spans has
``k + p`` coefficients whatever the breakpoint spacing.

This script lifts one converged solve onto both bases at matched coefficient
counts and certifies each with ``certify_strong_force``, at two quadrature
settings, reporting the near-axis, bulk, edge and total dimensional L2 and both
published normalizations, as ``residual_vs_resolution.py`` does.

Read the region splits with the grid in view.  ``certify_strong_force`` places
Gauss-Legendre nodes per spline span of the state's own basis, so the two
gradings are certified on different grids: the near-axis region ``rho < 0.2`` is
a masked sub-interval of one wide span under uniform-``s`` breakpoints and a
whole span or more under graded ones.  Every row therefore carries its per-region
node counts, the smallest sampled ``rho``, the certificate's own
``radial_refinement_difference``, and the change in each region between the two
quadrature settings.  A region whose value moves under quadrature refinement
reports the grid, not the representation.

This is a measurement of one representation choice on the decks named in the
output, not a claim about any other code or resolution.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import numpy as np

import vmex
from benchmarks._provenance import assert_repo_vmex, git_state
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import certify_strong_force, lift_high_order_state

DECKS = {
    "dshape": "input.DSHAPE",
    "tokamak": "input.shaped_tokamak_pressure_polished",
    "qa": "input.nfp2_QA_smooth_beta",
}

GRADINGS = ("uniform_s", "rho_graded")

# certify_strong_force splits on rho, not s; keep the same edges here so the
# reported node counts describe exactly the regions the report sums over.
NEAR_AXIS_RHO = 0.2
EDGE_RHO = 0.8


def breakpoints(spans: int, grading: str, exponent: float) -> np.ndarray:
    """Spline breakpoints in ``s`` for one grading.

    ``uniform_s`` is the existing convention.  ``rho_graded`` places the same
    number of breakpoints uniformly in ``rho = s ** (1 / exponent)``, which for
    the default ``exponent = 2`` is uniform in the square root of the flux label.
    Both start at 0 and end at 1, so the fixed boundary and the axis limit are
    untouched and only the interior spacing differs.
    """
    uniform = np.linspace(0.0, 1.0, spans + 1)
    if grading == "uniform_s":
        return uniform
    if grading == "rho_graded":
        return uniform**exponent
    raise ValueError(f"unknown grading {grading!r}")


def _certificate(state, *, angular_multiplier, radial_increment):
    """Both published normalizations, the dimensional L2, and where the error sits."""
    report = certify_strong_force(
        state, angular_multiplier=angular_multiplier,
        radial_order_increment=radial_increment)
    window = report.window_normalizations
    whole = report.global_normalizations
    rho = np.asarray(report.radial_nodes, dtype=float)
    return {
        "absolute_l2": float(report.absolute_l2),
        "absolute_linf": float(report.absolute_linf),
        "near_axis_l2": float(report.near_axis_l2),
        "bulk_l2": float(report.bulk_l2),
        "edge_l2": float(report.edge_l2),
        "minimum_signed_jacobian": float(report.minimum_signed_jacobian),
        "radial_refinement_difference": float(report.radial_refinement_difference),
        "boundary_residual": float(report.boundary_residual),
        "pressure_normalized": {
            "whole_volume": float(whole.relative_force_error),
            "window": float(window.relative_force_error),
        },
        "magnetic_normalized": {
            "whole_volume": float(whole.magnetic_relative_force_error),
            "window": float(window.magnetic_relative_force_error),
        },
        "window": {"s_min": float(window.s_min), "s_max": float(window.s_max),
                   "nodes": int(window.node_count)},
        # The certificate's radial grid follows the state's own breakpoints, so
        # the region splits are only comparable across gradings where they are
        # quadrature converged.  These counts say how thin the estimate is.
        "grid": {
            "radial_nodes": int(rho.size),
            "near_axis_nodes": int(np.count_nonzero(rho < NEAR_AXIS_RHO)),
            "bulk_nodes": int(np.count_nonzero((rho >= NEAR_AXIS_RHO) & (rho <= EDGE_RHO))),
            "edge_nodes": int(np.count_nonzero(rho > EDGE_RHO)),
            "minimum_rho": float(rho.min()),
            "maximum_rho": float(rho.max()),
        },
    }


def _solved(deck, ns, mpol, ntor):
    """One converged VMEC-lane solve at this radial mesh, reused by every basis."""
    from vmex.core import implicit
    from vmex.core.input import VmecInput

    inp = VmecInput.from_file(ROOT / "examples/data" / deck)
    if mpol is not None:
        inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=2 * mpol + 6,
                                    nzeta=max(4, 2 * ntor + 2))
    inp = replace(inp, ns_array=np.asarray([ns]), ftol_array=np.asarray([1.0e-12]),
                  niter_array=np.asarray([8000]))
    config = implicit.make_config(inp, ftol=1.0e-12, max_iterations=8000)
    params = implicit.params_from_input(inp)
    started = time.perf_counter()
    state, _mask = implicit.solve_implicit_with_aux(params, config)
    seconds = time.perf_counter() - started
    return state, implicit.runtime_from_params(params, config), seconds


REGIONS = ("absolute_l2", "near_axis_l2", "bulk_l2", "edge_l2")


def _drift(coarse, fine):
    """Relative change of each region between two quadrature settings."""
    return {key: abs(fine[key] - coarse[key]) / max(fine[key], 1.0e-300)
            for key in REGIONS}


def _source_support(basis, s_full):
    """How many VMEC full-mesh samples land in each span, and in the fit at all.

    ``lift_high_order_state`` fits the spline by unregularized least squares on
    the solve's own ``s`` mesh, dropping ``s = 0`` for every ``m > 0`` mode.  A
    span holding no interior sample contributes columns the data cannot
    determine, so the fitted coefficients there are whatever the minimum-norm
    solution supplies rather than a property of the equilibrium.  This is the
    variable that separates a representation result from a fitting artefact, so
    it is recorded beside every certificate.
    """
    edges = np.asarray(basis.breakpoints, dtype=float)
    s_full = np.asarray(s_full, dtype=float)
    interior = s_full[(s_full > 0.0) & (s_full < 1.0)]
    per_span = [int(np.count_nonzero((interior >= lo) & (interior < hi)))
                for lo, hi in zip(edges[:-1], edges[1:], strict=False)]
    positive = s_full[s_full > 1.0e-10]
    return {
        "samples_per_span": per_span,
        "minimum_samples_per_span": int(min(per_span)),
        "empty_spans": int(sum(1 for count in per_span if count == 0)),
        "fit_rows_m_positive": int(positive.size),
        "fit_columns": int(basis.size),
    }


def grading_scan(deck, ns, span_counts, degree, mpol, ntor, quadratures, exponent):
    """Certify both gradings at every span count and every quadrature setting."""
    state, runtime, seconds = _solved(deck, ns, mpol, ntor)
    s_full = np.asarray(runtime.setup.s_full, dtype=float)
    rows = []
    for spans in span_counts:
        sizes = set()
        by_grading = {}
        for grading in GRADINGS:
            basis = BSplineBasis.clamped(breakpoints(spans, grading, exponent),
                                         degree=degree)
            sizes.add(int(basis.size))
            native = lift_high_order_state(state, runtime, radial_basis=basis)
            certificates = {}
            for angular, increment in quadratures:
                certificates[f"angular{angular}_increment{increment}"] = _certificate(
                    native, angular_multiplier=angular, radial_increment=increment)
            coarse, fine = (certificates[f"angular{a}_increment{i}"]
                            for a, i in (quadratures[0], quadratures[-1]))
            by_grading[grading] = {
                "breakpoints_s": [float(v) for v in basis.breakpoints],
                "coefficients_per_mode": int(basis.size),
                "coefficients": int(np.asarray(native.R_cos).size),
                "source_support": _source_support(basis, s_full),
                "certificates": certificates,
                "quadrature_drift": _drift(coarse, fine) if len(quadratures) > 1 else None,
            }
            print(json.dumps({"spans": spans, "grading": grading,
                              "absolute_l2": fine["absolute_l2"],
                              "near_axis_l2": fine["near_axis_l2"]}), flush=True)
        if len(sizes) != 1:
            raise RuntimeError(f"coefficient counts not matched at {spans} spans: {sizes}")
        finest = f"angular{quadratures[-1][0]}_increment{quadratures[-1][1]}"
        uniform = by_grading["uniform_s"]["certificates"][finest]
        graded = by_grading["rho_graded"]["certificates"][finest]
        rows.append({
            "spans": int(spans),
            "degree": int(degree),
            "coefficients_per_mode": sizes.pop(),
            "gradings": by_grading,
            # Below one, the graded basis is the smaller residual.
            "graded_over_uniform": {
                key: graded[key] / max(uniform[key], 1.0e-300) for key in REGIONS},
        })
    return {"ns": int(ns), "solve_seconds": seconds, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deck", choices=sorted(DECKS), default="dshape")
    parser.add_argument("--ns", type=int, default=17,
                        help="radial mesh of the single converged solve")
    parser.add_argument("--spans", default="4,6,8,10",
                        help="spline span counts; both gradings run at each")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--grading-exponent", type=float, default=2.0,
                        help="graded breakpoints are linspace(0,1,k+1) ** exponent")
    parser.add_argument("--quadratures", default="2:2,4:8",
                        help="certificate settings as angular:radial_increment pairs")
    parser.add_argument("--mpol", type=int)
    parser.add_argument("--ntor", type=int)
    parser.add_argument("--budget-seconds", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spans = [int(v) for v in args.spans.split(",") if v]
    try:
        quadratures = [tuple(int(part) for part in pair.split(":"))
                       for pair in args.quadratures.split(",") if pair]
    except ValueError:
        parser.error("quadratures must be angular:radial_increment pairs")
    if not jax.config.x64_enabled:
        parser.error("require float64")
    if not 5 <= args.ns <= 201:
        parser.error("ns must be 5..201")
    if not spans or any(v < 1 or v > 12 for v in spans):
        parser.error("span counts must be 1..12")
    if len(set(spans)) != len(spans):
        parser.error("span counts must be distinct")
    if args.degree not in (3, 5, 7):
        parser.error("degree must be 3, 5 or 7")
    if not 1.0 <= args.grading_exponent <= 4.0:
        parser.error("grading-exponent must be 1.0..4.0")
    if not quadratures or any(len(pair) != 2 for pair in quadratures):
        parser.error("quadratures must be angular:radial_increment pairs")
    if any(not 1 <= a <= 8 or not 0 <= i <= 12 for a, i in quadratures):
        parser.error("angular multipliers must be 1..8 and radial increments 0..12")
    if not 60 <= args.budget_seconds <= 1800:
        parser.error("budget-seconds must be 60..1800")
    if (args.mpol is None) != (args.ntor is None):
        parser.error("set both --mpol and --ntor, or neither")

    def expired(*_):
        raise TimeoutError(f"knot grading scan {args.budget_seconds} s budget")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.budget_seconds)

    provenance = git_state(ROOT)
    provenance.update(
        vmex_module=assert_repo_vmex(vmex.__file__, ROOT),
        python=platform.python_version(), platform=platform.system(),
        versions={name: version(name) for name in ("jax", "jaxlib", "numpy", "scipy", "solvax")},
        device=str(jax.devices()[0]), precision="float64")
    started = time.perf_counter()
    output = {
        "_provenance": provenance,
        "deck": DECKS[args.deck],
        "settings": {k: v for k, v in vars(args).items() if k != "output"},
        "definitions": {
            "uniform_s": "breakpoints linspace(0, 1, spans + 1) in s, the existing convention",
            "rho_graded": "breakpoints linspace(0, 1, spans + 1) ** exponent in s, "
                          "so they are uniform in rho = sqrt(s) at exponent 2",
            "matched": "a clamped degree-p basis on k spans has k + p coefficients per "
                       "mode whatever the spacing, so the two arms are compared at "
                       "identical coefficient counts",
            "absolute_l2": "volume-weighted RMS of |J x B - grad p|, N m^-3",
            "pressure_normalized": "<|F|>/<|grad p|>, Panici et al. 2023",
            "magnetic_normalized": "<|F|>/<|grad(B^2/2 mu0)|>, Thun et al. 2026",
            "graded_over_uniform": "ratio at the finest quadrature; below 1 the graded "
                                   "basis is the smaller residual",
            "quadrature_drift": "relative change of each region between the coarsest and "
                                "finest quadrature setting; a large value means that "
                                "region reports the grid rather than the representation",
            "grid": "the certificate samples per span of the state's own basis, so the "
                    "two gradings are certified on different node sets; the counts say "
                    "how many nodes each region estimate rests on",
            "source_support": "VMEC full-mesh samples available to the least-squares "
                              "lift in each span; an empty span means the fit there is "
                              "determined by the minimum-norm solution, not by the data",
        },
    }
    try:
        output["scan"] = grading_scan(
            DECKS[args.deck], args.ns, spans, args.degree, args.mpol, args.ntor,
            quadratures, args.grading_exponent)
        output["status"] = ("measured at these resolutions on this deck; no trend is "
                            "extrapolated beyond them and no cross-code claim is made")
    except Exception as error:
        output["status"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        output["seconds"] = time.perf_counter() - started
        output["peak_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if platform.system() == "Darwin" else 1024)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

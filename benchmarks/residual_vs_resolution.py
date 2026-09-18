"""Continuum force residual against resolution, in the norms other codes publish.

A small ``FSQR`` says the discrete solve converged; it does not bound the
continuum residual ``J x B - grad p``. This measures that residual on the
independent certificate as resolution rises, along the two axes VMEX can
refine separately:

* the VMEC lane's radial mesh, ``ns``, at fixed Fourier resolution;
* the continuous representation's spline count, at a fixed converged solve.

Both are reported in the two published normalizations so the numbers are
comparable outside this repository: ``<|F|>/<|grad p|>`` of Panici, Conlin,
Dudt, Unalmis and Kolemen, *J. Plasma Phys.* **89** (2023) 955890303, and
``<|F|>/<|grad(B^2/2 mu0)|>`` of Thun, Merlo, Conlin, Panici and Boeckenhoff,
*Nucl. Fusion* **66** (2026), which stays finite in vacuum. The dimensional
volume L2 is carried beside them because no normalization can hide it, and the
near-axis, bulk and edge splits are carried because VMEC's error concentrates
at the axis.

This is a measurement, not a claim about another code: DESC and VMEC++ points
belong on the same axes only when produced under the same contract, which is
recorded per run rather than assumed.
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


def _certificate(state, *, angular_multiplier, radial_increment):
    """Both published normalizations, the dimensional L2, and where the error sits."""
    report = certify_strong_force(
        state, angular_multiplier=angular_multiplier,
        radial_order_increment=radial_increment)
    window = report.window_normalizations
    whole = report.global_normalizations
    return {
        "absolute_l2": float(report.absolute_l2),
        "absolute_linf": float(report.absolute_linf),
        "near_axis_l2": float(report.near_axis_l2),
        "bulk_l2": float(report.bulk_l2),
        "edge_l2": float(report.edge_l2),
        "minimum_signed_jacobian": float(report.minimum_signed_jacobian),
        "radial_refinement_difference": float(report.radial_refinement_difference),
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
    }


def _solved(deck, ns, mpol, ntor):
    """One converged VMEC-lane solve at this radial mesh."""
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
    state, mask = implicit.solve_implicit_with_aux(params, config)
    seconds = time.perf_counter() - started
    return state, mask, implicit.runtime_from_params(params, config), seconds


def radial_scan(deck, mesh, span_counts, degree, mpol, ntor, angular, radial_increment):
    """Certificate against the VMEC lane's radial mesh.

    A lift held fixed while ``ns`` rises measures the basis, not the solve: a
    finer solution carries radial structure a coarse spline basis cannot
    represent, and the certificate then reports that projection error. Every
    mesh point is therefore certified at two lift resolutions, and the pair is
    reported so a reader can see whether the number is lift-limited.
    """
    rows = []
    for ns in mesh:
        state, _, runtime, seconds = _solved(deck, ns, mpol, ntor)
        lifts = {}
        for spans in span_counts:
            basis = BSplineBasis.clamped(np.linspace(0, 1, spans + 1), degree=degree)
            native = lift_high_order_state(state, runtime, radial_basis=basis)
            lifts[str(spans)] = _certificate(
                native, angular_multiplier=angular, radial_increment=radial_increment)
        coarse, fine = (lifts[str(v)] for v in (min(span_counts), max(span_counts)))
        drift = abs(fine["absolute_l2"] - coarse["absolute_l2"]) / max(
            fine["absolute_l2"], 1.0e-300)
        row = {"ns": int(ns), "solve_seconds": seconds, "lift_degree": degree,
               "lifts": lifts, "finest_spans": max(span_counts),
               "lift_convergence_difference": drift,
               "lift_limited": bool(drift > 0.1),
               "absolute_l2": fine["absolute_l2"]}
        rows.append(row)
        print(json.dumps({"axis": "ns", "ns": row["ns"],
                          "absolute_l2": row["absolute_l2"],
                          "lift_limited": row["lift_limited"]}), flush=True)
    return rows


def spline_scan(deck, ns, span_counts, degree, mpol, ntor, angular, radial_increment):
    """Certificate against the continuous representation, at one converged solve."""
    state, _, runtime, seconds = _solved(deck, ns, mpol, ntor)
    rows = []
    for spans in span_counts:
        basis = BSplineBasis.clamped(np.linspace(0, 1, spans + 1), degree=degree)
        native = lift_high_order_state(state, runtime, radial_basis=basis)
        row = {"ns": int(ns), "solve_seconds": seconds, "lift_spans": int(spans),
               "lift_degree": degree,
               "coefficients": int(np.asarray(native.R_cos).size)}
        row.update(_certificate(native, angular_multiplier=angular,
                                radial_increment=radial_increment))
        rows.append(row)
        print(json.dumps({"axis": "spans", **{k: row[k] for k in ("lift_spans", "absolute_l2")}}),
              flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deck", choices=sorted(DECKS), default="dshape")
    parser.add_argument("--mesh", default="9,17,25,33",
                        help="ns values for the radial scan")
    parser.add_argument("--spans", default="2,3,4,6",
                        help="spline span counts for the representation scan")
    parser.add_argument("--lift-spans", type=int, default=4,
                        help="spans held fixed while ns varies")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--mpol", type=int)
    parser.add_argument("--ntor", type=int)
    parser.add_argument("--angular", type=int, default=2)
    parser.add_argument("--radial-increment", type=int, default=2)
    parser.add_argument("--budget-seconds", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mesh = [int(v) for v in args.mesh.split(",") if v]
    spans = [int(v) for v in args.spans.split(",") if v]
    if not jax.config.x64_enabled:
        parser.error("require float64")
    if not mesh or any(n < 5 or n > 201 for n in mesh):
        parser.error("mesh values must be 5..201")
    if not spans or any(v < 1 or v > 12 for v in spans):
        parser.error("span counts must be 1..12")
    if args.degree not in (3, 5, 7):
        parser.error("degree must be 3, 5 or 7")
    if not 60 <= args.budget_seconds <= 21600:
        parser.error("budget-seconds must be 60..21600")
    if (args.mpol is None) != (args.ntor is None):
        parser.error("set both --mpol and --ntor, or neither")

    def expired(*_):
        raise TimeoutError(f"residual scan {args.budget_seconds} s budget")

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
        "normalizations": {
            "pressure_normalized": "<|F|>/<|grad p|>, Panici et al. 2023",
            "magnetic_normalized": "<|F|>/<|grad(B^2/2 mu0)|>, Thun et al. 2026",
            "absolute_l2": "volume-weighted RMS of |J x B - grad p|, N m^-3",
            "lift_limited": "the two lift resolutions disagree by more than 10%, so "
                            "the number reports the basis rather than the solve",
        },
    }
    try:
        output["radial_scan"] = radial_scan(
            DECKS[args.deck], mesh, sorted({args.lift_spans, max(spans)}),
            args.degree, args.mpol, args.ntor, args.angular, args.radial_increment)
        output["spline_scan"] = spline_scan(
            DECKS[args.deck], max(mesh), spans, args.degree, args.mpol, args.ntor,
            args.angular, args.radial_increment)
        output["status"] = ("measured at these resolutions; no cross-code claim is "
                            "made and no trend is extrapolated beyond them")
    except Exception as error:
        output["status"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        output["seconds"] = time.perf_counter() - started
        output["peak_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if platform.system() == "Darwin" else 1024)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

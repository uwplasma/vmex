#!/usr/bin/env python
"""Bounded pilot on a real equilibrium and an explicit kinetic response.

A WOUT alone cannot define ash transport. Supply ``--response module:factory``;
``factory(profile_json)`` must return an object exposing:

* ``transport(equilibrium)``: keyword mapping for ESSOS's SI radial evaluator,
  recomputed for each equilibrium, including fixed-normalization source,
  prescribed Er (or qualified root), volumes and explicit boundary kappa;
* ``verify(candidate, reference, *, seed, stage)``: a dictionary of the hard
  checks named in REQUIRED_GATES below, plus physical metrics/uncertainties;
* optional ``proxy(equilibrium)``: a nonnegative conventional electron-root
  cost for the matched ``--objective proxy`` control;
* optional ``native_transport(state, runtime)`` for the implicit lane, only
  after component and end-to-end directional FD qualification.

The default host FD lane permits an existing full-FP WOUT/array exporter. It
claims no native kinetic gradient. The provider owns DKX physics, SI conversion,
profiles and response validation; this example owns VMEX and the budget. See
``examples/README.md`` for the gate meanings. No manufactured response is
provided as a physical optimization. Keep the same provider/profile/budget for
``ash``, ``qi`` and ``proxy`` comparisons. ``--qualify-only`` evaluates the seed
and a directional FD ladder without optimizing it.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import hashlib
import inspect
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from types import SimpleNamespace
import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.ash import HeliumAshObjective
from vmex.core.qi import ConstructedQIResidual

# Every published acceptance needs independent, energy-weighted alpha evidence.
# The provider must retain unclassified failures/censoring and return False if
# an uncertainty, refinement, source, root or physical tolerance is unresolved.
REQUIRED_GATES = (
    "equilibrium", "kinetic_response", "radial_refinement", "source_matched",
    "device_size_and_field", "qi_and_shape", "bulk_heat_transport",
    "alpha_energy_absolute", "alpha_energy_noninferiority",
    "alpha_independent_holdout", "alpha_numerical_refinement",
    "alpha_all_weight_accounted",
)


def checked_verification(report):
    """Fail closed on missing, false, or non-boolean physical gates."""
    failed = [key for key in REQUIRED_GATES if report.get(key) is not True]
    return {"passed": not failed, "failed": failed, "report": report}


def candidate_status(radial, reference_cost, verification):
    """Distinguish physical feasibility from a demonstrated ash benefit."""
    cost = float(radial.mean_occupation_s)
    if not bool(radial.valid) or not np.isfinite(cost) or cost <= 0:
        return dict(status="rejected_candidate", error="invalid candidate radial response")
    gain = 1.0 - cost / reference_cost
    lower = verification["report"].get("ash_relative_gain_lower95")
    demonstrated = (isinstance(lower, (int, float)) and np.isfinite(lower)
                    and 0 < lower <= gain and gain > 0)
    status = ("rejected_candidate" if not verification["passed"] else
              "accepted" if demonstrated else "feasible_only")
    return dict(status=status, candidate_occupation_s=cost, ash_relative_gain=gain,
                ash_relative_gain_lower95=lower if demonstrated else None)


def load_provider(specification, profile):
    module, function = specification.split(":", 1)
    return getattr(importlib.import_module(module), function)(profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--restart-wout", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--response", required=True, help="importable module:factory")
    parser.add_argument("--objective", choices=("ash", "qi", "proxy"), default="ash")
    parser.add_argument("--derivative-method", choices=("finite_difference", "implicit"),
                        default="finite_difference")
    parser.add_argument("--budget", type=int, default=40,
                        help="total optimizer residual calls, including FD probes")
    parser.add_argument("--directions", type=int, choices=range(8, 21), default=8)
    parser.add_argument("--step-m", type=float, default=0.002)
    parser.add_argument("--box-steps", type=float, default=2.0)
    parser.add_argument("--checkpoint-seed", type=int, default=2718)
    parser.add_argument("--holdout-seed", type=int, default=314159)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--evaluate-wout-only", action="store_true",
                        help="evaluate exact restart WOUT; no equilibrium re-solve or shape gradient")
    parser.add_argument("--output", type=Path, default=Path("outputs/helium_ash"))
    args = parser.parse_args(argv)
    if args.checkpoint_seed == args.holdout_seed:
        parser.error("checkpoint and final alpha holdout seeds must differ")
    if args.budget < args.directions + 1 or not (0 < args.step_m < np.inf):
        parser.error("budget must exceed direction count and step-m must be finite positive")
    if not (0 < args.box_steps < np.inf):
        parser.error("box-steps must be finite positive")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (args.input, args.profile, args.restart_wout) if path is not None}
    versions = {}
    for name in ("vmex", "essos", "dkx", "jax", "solvax"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            # Optional distribution metadata must not hide a physical rejection.
            versions[name] = None
    ledger = dict(input_sha256=hashes,
                  versions=versions,
                  checkpoint_seed=args.checkpoint_seed, holdout_seed=args.holdout_seed,
                  objective=args.objective, budget=args.budget, status="unqualified")

    def save():
        ledger["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "report.json").write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")

    provider = load_provider(args.response, args.profile)
    try:
        provider_source = inspect.getsourcefile(type(provider))
    except TypeError:  # Built-in response objects need not expose Python source.
        provider_source = None
    ledger["response_provider"] = args.response
    if provider_source is not None:
        hashes[str(Path(provider_source).resolve())] = hashlib.sha256(
            Path(provider_source).read_bytes()).hexdigest()
    inp = vj.VmecInput.from_file(args.input)
    if args.evaluate_wout_only:
        if args.restart_wout is None:
            parser.error("--evaluate-wout-only requires --restart-wout")
        reference = SimpleNamespace(wout=vj.read_wout(args.restart_wout))
    else:
        reference = opt.solve_equilibrium(inp, restart_from=args.restart_wout,
                                          raise_on_max_iterations=True)
    try:
        transport = provider.transport(reference)
        seed_ash = HeliumAshObjective(lambda state, rt: transport, 1.0,
                                      "radial coefficient derivatives only")
        radial = seed_ash.evaluate_state(None, None)
        if not bool(radial.valid):
            raise ValueError("seed radial response invalid; inspect D, source, geometry and sink")
        reference_cost = float(radial.mean_occupation_s)
        ledger.update(reference_occupation_s=reference_cost,
                      reference_residence_s=float(radial.mean_residence_s))
        initial_checks = checked_verification(provider.verify(
            reference, reference, seed=args.checkpoint_seed, stage="checkpoint"))
        ledger["checkpoint"] = initial_checks
    except (ValueError, RuntimeError, FloatingPointError) as error:
        ledger.update(status="rejected_seed", error=str(error))
        save()
        print(json.dumps(ledger, indent=2, allow_nan=False))
        return 1
    if not initial_checks["passed"] or args.evaluate_wout_only:
        ledger["status"] = "qualified_seed" if initial_checks["passed"] else "rejected_seed"
        save()
        print(json.dumps(ledger, indent=2, allow_nan=False))
        return 0 if initial_checks["passed"] else 1
    scope = (f"{args.derivative_method}: equilibrium and recomputed kinetic response; "
             "Er/background/source/boundary choices are fixed by the response provider")
    ash = HeliumAshObjective(getattr(provider, "native_transport", lambda state, rt: transport),
                             reference_cost, scope)

    def host_ash(eq):
        response = provider.transport(eq)
        term = replace(ash, transport_response=lambda state, rt: response)
        return term.residuals_state(None, None)

    if args.derivative_method == "implicit" and not hasattr(provider, "native_transport"):
        parser.error("implicit lane requires a qualified native_transport(state, runtime)")
    qi = ConstructedQIResidual((0.25, 0.5, 0.75), mboz=8, nboz=8,
                               nphi=31, nalpha=7, n_bounce=9)
    volume0 = float(opt.volume(reference.state, reference.runtime))
    aspect0 = float(opt.aspect_ratio(reference.state, reference.runtime))
    terms = [(qi, 0.0, 10.0), (opt.aspect_ratio, aspect0, 1.0),
             (lambda state, rt: opt.volume(state, rt) / volume0, 1.0, 10.0),
             (lambda state, rt: jnp.maximum(0.2 - opt.min_abs_iota(state, rt), 0), 0., 10.),
             (lambda state, rt: jnp.maximum(opt.mirror_ratio(state, rt) - 0.25, 0), 0., 10.)]
    if args.objective == "ash":
        terms.append((ash if args.derivative_method == "implicit" else host_ash, 0., 1.))
    if args.objective == "proxy":
        if args.derivative_method != "finite_difference" or not hasattr(provider, "proxy"):
            parser.error("proxy control requires provider.proxy(equilibrium) and host FD")
        terms.append((lambda eq: np.sqrt(provider.proxy(eq)), 0., 1.))
    problem = opt.VmecProblem.from_tuples(
        inp, terms, max_mode=2, vary_major_radius=False,
        derivative_method=args.derivative_method, restart_from=reference,
        progress=True, evaluation_progress=True)
    # Eight m<=1 coefficients first, then selected m<=2 coefficients if requested.
    low = opt.boundary_dof_names(inp, 1, vary_major_radius=False)
    names = low + [name for name in problem.dof_names if name not in low]
    problem = problem.subproblem(active=names[:args.directions])
    x0, step = problem.x0.copy(), args.step_m * problem.scales
    direction = step / np.linalg.norm(step)
    ladder = []
    for h in (1e-3, 3e-4, 1e-4):
        fd = (problem.fun(x0 + h * direction) - problem.fun(x0 - h * direction)) / (2 * h)
        ladder.append({"step_m": h, "directional_fd": float(fd)})
    if not all(np.isfinite(row["directional_fd"]) for row in ladder) or not np.isclose(
            ladder[-1]["directional_fd"], ladder[-2]["directional_fd"], rtol=1e-3, atol=1e-8):
        ledger.update(status="rejected_gradient", error="directional FD ladder has no plateau",
                      gradient_ladder=[{**row, "directional_fd": row["directional_fd"]
                                       if np.isfinite(row["directional_fd"]) else None}
                                      for row in ladder])
        save()
        return 1
    # Implicit claims must agree with the FD plateau before any optimization.
    if args.derivative_method == "implicit":
        ad = float(np.dot(problem.grad(x0), direction))
        if not np.isclose(ad, ladder[-1]["directional_fd"], rtol=1e-3, atol=1e-8):
            raise RuntimeError(f"end-to-end gradient qualification failed: AD={ad}, FD={ladder}")
    ledger.update(derivative_scope=scope, directions=problem.dof_names,
                  gradient_ladder=ladder, qualification_response_calls=6)
    if args.qualify_only:
        ledger["status"] = "qualified_seed"
        save()
        return 0

    class BudgetExhausted(Exception):
        pass

    calls, accepted = 0, np.zeros_like(x0)
    monitor = opt.OptimizationMonitor(problem=problem)

    def residual(y):
        nonlocal calls
        if calls >= args.budget:
            raise BudgetExhausted
        calls += 1
        return problem.residual(x0 + step * y)

    def record(intermediate_result):
        nonlocal accepted
        accepted = intermediate_result.x.copy()
        monitor({"x": x0 + step * accepted, "fun": intermediate_result.fun})

    try:
        jacobian = ("3-point" if args.derivative_method == "finite_difference" else
                    lambda y: problem.residual_jac(x0 + step * y) * step[None, :])
        fitted = least_squares(residual, accepted, jac=jacobian,
            bounds=(-args.box_steps, args.box_steps), callback=record,
            max_nfev=args.budget, ftol=1e-8, xtol=1e-8, gtol=1e-6)
        accepted = fitted.x.copy()
    except BudgetExhausted:
        pass  # Only the latest SciPy-accepted iterate is retained.
    candidate_input = problem.input_from_x(x0 + step * accepted)
    # Independent physical re-solve; never certify the optimizer's cached state.
    candidate = opt.solve_equilibrium(candidate_input, restart_from=reference.wout,
                                      raise_on_max_iterations=True)
    monitor.save(args.output / "accepted_iterations.csv")
    candidate_input.to_indata(args.output / "input.candidate")
    vj.write_wout(args.output / "wout_candidate.nc", candidate.wout)
    ledger["candidate_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.output / "input.candidate", args.output / "wout_candidate.nc")}
    try:
        candidate_radial = replace(ash,
            transport_response=lambda state, rt: provider.transport(candidate)
        ).evaluate_state(None, None)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        ledger.update(status="rejected_candidate", error=str(error), optimizer_response_calls=calls)
        save()
        return 1
    if (not bool(candidate_radial.valid) or
            not np.isfinite(float(candidate_radial.mean_occupation_s)) or
            float(candidate_radial.mean_occupation_s) <= 0):
        ledger.update(status="rejected_candidate", error="invalid candidate radial response",
                      optimizer_response_calls=calls)
        save()
        return 1
    final = checked_verification(provider.verify(
        candidate, reference, seed=args.holdout_seed, stage="independent_holdout"))
    ledger.update(optimizer_response_calls=calls, holdout=final,
                  **candidate_status(candidate_radial, reference_cost, final))
    save()
    print(json.dumps(ledger, indent=2, allow_nan=False))
    return 0 if ledger["status"] in ("accepted", "feasible_only") else 1


if __name__ == "__main__":
    raise SystemExit(main())

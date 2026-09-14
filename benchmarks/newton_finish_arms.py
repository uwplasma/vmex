#!/usr/bin/env python
"""Plan B1: Newton finishes, the block adjoint and a refinement abort, on one deck.

The starts are descent states at fsq 1e-6, 1e-8 and 1e-10 (one cold multigrid
trajectory stopped early, so descent counts are exact differences) and the
deck-tolerance state. Every arm is judged on the certificate the implicit lane
reads, ``|P gc|`` against ``refine_tol``, with work ``W = F + c JVP + f factor``
in warm force-evaluation units and the plan's kill rules on ``fsq = |P gc|^2``:

(a) descent to the deck tolerance, then ``_refined_state``;
(b) Newton on the 1-D preconditioned residual (``newton_direction``, GMRES(50), rtol 1e-3);
(c) Newton on the raw residual, GMRES right-preconditioned by the exact block factorization;
(d) Anderson (window 8) on ``z + dt^2 F(z)``, stopped at the deck fsq;
(e) arm (c) confined to the complement of the ``k`` smallest raw singular modes,
    judged on ``|P gc|`` with their images ``J_p V`` removed;
(f) the adjoint through the transposed block factorization, ``mu = J_raw^-T P gbar``:
    at the root ``dF_p/dp = M dF_raw/dp``, so the 1-D preconditioner ``M`` cancels
    from the parameter pullback;
(g) refinement with a GCROT stagnation abort (< 25 % reduction over 1,000 or 500
    iterations), simulated with one-shot solves cut at the firing cycle and
    compared bit for bit with #320's policy.

At the deck state it also records the smallest raw singular values with their
mode content, the residual change along each near-null mode, per-cycle GCROT
histories of the refinement and adjoint solves (one cycle at a time with the
recycle space carried), the production adjoint, and the share of the QA and QI
boundary gradients carried by the near-null modes. Seconds are diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import functools
import json
import math
import os
import platform
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
import solvax

import vmex
from vmex import optimize as opt
from vmex.core import implicit as imp
from vmex.core import optimize as core_optimize
from vmex.core.input import VmecInput
from vmex.core.multigrid import solve_multigrid
from vmex.core.preconditioner_2d import Prec2DConfig, newton_direction
from vmex.core.solver import evaluate_forces

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import assert_repo_vmex, file_sha256, git_state  # noqa: E402
from optimization import terms  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NEWTON_RTOL = 1e-3
EXACT_KEYS = {"value", "gradient", "gradient_norm"}  # compared with the benchmark rows


def compact_json(value, width=150):
    """Record JSON: 4 significant digits except ``EXACT_KEYS``, lists of flat
    records as columns, and objects or lists of scalars on one line."""
    def prepare(node, exact=False):
        if isinstance(node, float):
            return node if exact or not math.isfinite(node) else float(f"{node:.4g}")
        if isinstance(node, dict):
            return {k: prepare(v, exact or k in EXACT_KEYS) for k, v in sorted(node.items())}
        if not isinstance(node, list):
            return node
        node = [prepare(v, exact) for v in node]
        if (len(node) > 1 and all(isinstance(v, dict) for v in node) and len({tuple(v) for v in node}) == 1
                and not any(isinstance(x, (dict, list)) for v in node for x in v.values())):
            return {k: [v[k] for v in node] for k in node[0]}
        return node

    def leaf(node):
        return not isinstance(node, (dict, list)) or (
            isinstance(node, list) and not any(isinstance(x, (dict, list)) for x in node))

    def render(node, depth):
        text = json.dumps(node)
        members = list(node.values()) if isinstance(node, dict) else node
        if leaf(node) or len(text) + depth <= width or all(map(leaf, members)):
            return text
        pad = " " * (depth + 1)
        items = ([pad + render(v, depth + 1) for v in node] if isinstance(node, list)
                 else [f"{pad}{json.dumps(k)}: {render(v, depth + 1)}" for k, v in node.items()])
        return ("[" if isinstance(node, list) else "{") + "\n" + ",\n".join(items) + (
            "]" if isinstance(node, list) else "}")

    return render(prepare(value), 0) + "\n"


def build_case(case: str):
    if case == "seed":  # the equilibrium of the QA and QI rows of optimization.py
        path = REPO / "examples/data/input.minimal_seed_nfp2"
        inp = replace(VmecInput.from_file(path), delt=0.5).change_resolution(mpol=5, ntor=5, ntheta=16, nzeta=14)
        forward_ftol = None  # the factory's 1e-12
    else:
        path = REPO / "examples/data/input.LandremanPaul2021_QA_lowres"
        inp, forward_ftol = VmecInput.from_file(path), 1e-11  # plan §2: 1e-13 is unreachable
    problem = opt.VmecProblem.from_tuples(inp, terms("qa"), max_mode=1, use_ess=True,
                                          forward_ftol=forward_ftol, forward_max_iterations=5500)
    return path, problem.metadata["config"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("seed", "qa_lowres"), default="seed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="one start, one step per arm: checks the script")
    args = parser.parse_args()
    smoke = args.smoke
    starts_ftol = (1e-8,) if smoke else (1e-6, 1e-8, 1e-10)
    newton_steps = dict(b=1, c=1, e=1) if smoke else dict(b=3, c=8, e=8)
    spectrum_size, deflation_sizes = (4, (2,)) if smoke else (16, (2, 6, 12))
    epsilons = (1e-4,) if smoke else (1e-8, 1e-6, 1e-4, 1e-2)
    history_cycles, objectives = (2, ("qa",)) if smoke else (60, ("qa", "qi"))
    refine_cycles = 2 if smoke else imp._REFINE_MAX_RESTARTS
    load_start = os.getloadavg()

    path, cfg = build_case(args.case)
    params = imp.params_from_input(cfg.inp, device=cfg.device)
    mask = jax.tree.map(jnp.asarray, imp._MASK_CACHE[imp._mask_cache_key(cfg)])
    P, fields, edge = imp._dof_projector(cfg, mask), imp._active_state_fields(cfg), imp._edge_mask(cfg)
    modes = imp._template_runtime(cfg).modes
    pack = lambda tree: imp._pack_active(cfg, tree)  # noqa: E731
    unpack = lambda matrix: imp._unpack_active(cfg, matrix)  # noqa: E731
    norm = lambda tree: float(imp._tree_norm(tree))  # noqa: E731
    flat = lambda tree: ravel_pytree(tree)[0]  # noqa: E731
    add = lambda a, b, scale=1.0: jax.tree.map(lambda x, y: x + scale * y, a, b)  # noqa: E731

    def descent(ftol):
        ftol_array = np.asarray(cfg.inp.ftol_array, dtype=float).copy()
        ftol_array[-1] = ftol
        return solve_multigrid(cfg.inp, ns_array=np.asarray(cfg.inp.ns_array), ftol_array=ftol_array,
                               mode=cfg.mode, lconm1=cfg.lconm1, raise_on_max_iterations=False, use_fft=False)

    deck = imp._LAST_SOLVE[cfg][1]
    runs = {**{str(ftol): descent(ftol) for ftol in starts_ftol}, "deck": deck}
    z_deck = P(deck.state)
    unravel = ravel_pytree(z_deck)[1]

    # ---- compiled pieces -------------------------------------------------------------
    @jax.jit
    def evaluate(z, frozen):
        runtime = imp.runtime_from_params(params, cfg)
        gc, raw, diagnostics = evaluate_forces(imp._assemble(z, runtime, frozen, P, edge), runtime)
        return P(gc), raw.fsqr + raw.fsqz + raw.fsql, diagnostics.jacobian_sign_changed

    def certificate(z, frozen):
        force, fsq, bad = evaluate(z, frozen)
        return norm(force), float(fsq), bool(bad)

    def force_of(frozen, formulation="preconditioned"):
        return imp.residual_fn(cfg, frozen, mask, formulation=formulation)

    @jax.jit
    def residuals(z, frozen):
        return force_of(frozen, "raw")(z, params), force_of(frozen)(z, params)

    @jax.jit
    def jvps(z, frozen, tangent):
        return tuple(jax.jvp(lambda t, f=force_of(frozen, name): f(t, params), (z,), (tangent,))[1]
                     for name in ("raw", "preconditioned"))

    @jax.jit
    def raw_normal(z, frozen, tangent):
        raw = force_of(frozen, "raw")
        _, pullback = jax.vjp(lambda t: raw(t, params), z)
        return P(pullback(jax.jvp(lambda t: raw(t, params), (z,), (P(tangent),))[1])[0])

    @jax.jit
    def block_system(frozen, z):
        system = imp._raw_block_system(params, cfg, frozen, mask, fields, 32, z_star=z)
        return system.factors, system.row_scale, system.column_scale

    def inverse(factors, tree, transpose=False):
        return imp._block_inverse_apply(factors[0], pack, unpack, P, factors[1], factors[2], tree,
                                        transpose=transpose)

    @jax.jit
    def normal_inverse(factors, v):
        return flat(inverse(factors, inverse(factors, P(unravel(v)), transpose=True)))

    def timed(function, *arguments, repeats=5):
        jax.tree.map(np.asarray, function(*arguments))
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            jax.tree.map(np.asarray, function(*arguments))
            samples.append(time.perf_counter() - started)
        return statistics.median(samples)

    # ---- unit costs and (a) today's refinement from the deck state ------------------------
    stats = imp._SOLVE_STATS[cfg]

    def counted(function, *keys):
        before, started = {key: stats[key] for key in keys}, time.perf_counter()
        result = function()
        return result, time.perf_counter() - started, {key: stats[key] - before[key] for key in keys}

    residual_deck = force_of(deck.state)
    jvp_deck = jax.jit(lambda z, v: jax.jvp(lambda t: residual_deck(t, params), (z,), (v,)))
    t_force, t_jvp = timed(evaluate, z_deck, deck.state), timed(jvp_deck, z_deck, z_deck)
    t_factor = timed(block_system, deck.state, z_deck, repeats=3)
    fz = residual_deck(z_deck, params)
    imp._refine_step(cfg, params, deck.state, mask, z_deck, fz)
    _, t_step, delta = counted(lambda: np.asarray(imp._refine_step(cfg, params, deck.state, mask, z_deck, fz)[2]),
                               "refinement_krylov_iterations")
    per_iteration = t_step / max(1, delta["refinement_krylov_iterations"])
    units = {"force_evaluation_seconds": t_force, "jvp_seconds": t_jvp, "block_build_factor_seconds": t_factor,
             "refinement_gcrot_seconds_per_iteration": per_iteration, "c_jvp": t_jvp / t_force,
             "c_refinement_iteration": per_iteration / t_force, "f_factorization": t_factor / t_force}
    refined, seconds, delta = counted(lambda: imp._refined_state(cfg, params, deck.state, mask),
                                      "refinement_steps", "refinement_krylov_iterations")
    refinement = {"seconds": seconds, "steps": delta["refinement_steps"],
                  "gcrot_iterations": delta["refinement_krylov_iterations"],
                  "certificate_before": certificate(z_deck, deck.state)[0],
                  "certificate_after": certificate(P(refined), deck.state)[0],
                  "state_moved": norm(add(P(refined), z_deck, -1.0))}
    refinement["work"] = (1 + 2 * refinement["steps"]
                          + units["c_refinement_iteration"] * refinement["gcrot_iterations"])

    # ---- per-cycle GCROT histories -----------------------------------------------------------
    @functools.partial(jax.jit, static_argnames=("rtol", "transpose", "max_restarts"))
    def gcrot_cycles(x0, recycle, z, frozen, b, rtol, transpose, max_restarts):
        force = force_of(frozen)
        if transpose:
            _, pullback = jax.vjp(lambda t: force(t, params), z)
            operator = lambda v: pullback(v)[0]  # noqa: E731
        else:
            _, operator = jax.linearize(lambda t: force(t, params), z)
        b_flat = flat(b)
        n = int(b_flat.shape[0])
        solution = solvax.gcrot(lambda v: flat(operator(unravel(v))), b_flat, x0=x0,
                                m=min(cfg.adjoint_gcrot_m, n), k=min(cfg.adjoint_gcrot_k, n),
                                rtol=rtol, max_restarts=max_restarts, recycle=recycle)
        return (solution.x, solution.recycle, solution.residual_norm / jnp.linalg.norm(b_flat),
                solution.iterations, solution.converged)

    def history(b, *, rtol, transpose, cycles, z=z_deck):
        x, recycle, rows, total = jnp.zeros_like(flat(b)), None, [], 0
        for _ in range(cycles):
            x, recycle, relative, iterations, converged = gcrot_cycles(
                x, recycle, z, deck.state, b, rtol, transpose, 1)
            total += int(iterations)
            rows.append([total, float(relative)])
            if bool(converged):
                break
        _, _, relative, iterations, converged = gcrot_cycles(
            jnp.zeros_like(flat(b)), None, z, deck.state, b, rtol, transpose, len(rows))
        return {"cycles": rows, "one_shot_same_cycles": {"iterations": int(iterations),
                                                         "relative_residual": float(relative),
                                                         "converged": bool(converged)}}

    refinement["gcrot_history_first_step"] = history(fz, rtol=imp._REFINE_FORCING, transpose=False,
                                                     cycles=refine_cycles)

    # ---- spectrum, mode content and the residual change along each mode ---------------------
    def describe(vector):
        tree = unravel(jnp.asarray(vector))
        energy = np.stack([np.asarray(getattr(tree, name)) ** 2 for name in fields])
        total, ns = energy.sum(), energy.shape[1]
        return {"dominant": [f"{fields[f]}[s={s},m={int(modes.m[k])},n={int(modes.n[k])}]" for f, s, k in (
                    np.unravel_index(i, energy.shape) for i in np.argsort(-energy.ravel())[:3])],
                "field_fraction": {name: float(energy[i].sum() / total) for i, name in enumerate(fields)},
                "fraction_last_rows": {str(r): float(energy[:, ns - r:].sum() / total) for r in (1, 3, 6)}}

    def spectrum(frozen, z, factors, count):
        n = int(flat(z).shape[0])
        values, vectors = eigsh(LinearOperator((n, n), dtype=float, matvec=lambda v: np.asarray(
            normal_inverse(factors, jnp.asarray(v.ravel())))), k=count, which="LA", tol=1e-12)
        order = np.argsort(-values)
        V = vectors[:, order]
        images = [jvps(z, frozen, unravel(jnp.asarray(V[:, i]))) for i in range(count)]
        return (1.0 / np.sqrt(values[order]), V, np.stack([flat(raw) for raw, _ in images], axis=1),
                np.stack([flat(pre) for _, pre in images], axis=1))

    def orthonormal(matrix):
        return jnp.asarray(np.linalg.qr(np.asarray(matrix))[0])

    factors_deck = block_system(deck.state, z_deck)
    sigma_deck, V_deck, JV_deck, JpV_deck = spectrum(deck.state, z_deck, factors_deck, spectrum_size)
    n_state = V_deck.shape[0]
    sigma_max = float(np.sqrt(eigsh(LinearOperator((n_state, n_state), dtype=float, matvec=lambda v: np.asarray(
        flat(raw_normal(z_deck, deck.state, unravel(jnp.asarray(v.ravel())))))), k=1, which="LA", tol=1e-6)[0][0]))
    raw_deck, pre_deck = residuals(z_deck, deck.state)
    gauge = []
    for i in range(min(6, spectrum_size)):
        changes = []
        for epsilon in epsilons:
            raw_e, pre_e = residuals(add(z_deck, unravel(jnp.asarray(V_deck[:, i])), epsilon), deck.state)
            changes.append({"epsilon": epsilon, "raw_change_over_epsilon": norm(add(raw_e, raw_deck, -1.0)) / epsilon,
                            "preconditioned_change_over_epsilon": norm(add(pre_e, pre_deck, -1.0)) / epsilon})
        gauge.append({"sigma": float(sigma_deck[i]), "raw_jacobian_image_norm": float(np.linalg.norm(JV_deck[:, i])),
                      "preconditioned_jacobian_image_norm": float(np.linalg.norm(JpV_deck[:, i])),
                      "residual_change": changes, **describe(V_deck[:, i])})

    def share(images, vector, k):
        return float(np.linalg.norm(np.asarray(orthonormal(images[:, :k])).T @ vector) / np.linalg.norm(vector))

    raw_flat, pre_flat = np.asarray(flat(raw_deck)), np.asarray(flat(pre_deck))
    spectrum_report = {
        "sigma_max": sigma_max, "sigma_smallest": [float(s) for s in sigma_deck],
        "consecutive_ratio": [float(sigma_deck[i + 1] / sigma_deck[i]) for i in range(spectrum_size - 1)],
        "residual_share_on_left_near_null": {str(k): {"raw": share(JV_deck, raw_flat, k),
                                                      "preconditioned": share(JpV_deck, pre_flat, k)}
                                             for k in deflation_sizes},
        "modes": gauge, "edge_lambda_dof_rows_active": bool(np.asarray(mask.L_sin)[-1].any())}
    if args.case == "seed" and not smoke:  # dense cross-check of the Lanczos values
        bands = jax.jit(lambda frozen: (lambda s: (s.lower, s.diagonal, s.upper))(
            imp._raw_block_system(params, cfg, frozen, mask, fields, 32, factor=False)))(deck.state)
        lower, diagonal, upper = (np.asarray(a) for a in bands)
        ns, block, _ = diagonal.shape
        dense = np.zeros((ns * block, ns * block))
        for j in range(ns):
            rows = slice(j * block, (j + 1) * block)
            dense[rows, rows] = diagonal[j]
            if j:
                dense[rows, (j - 1) * block:j * block] = lower[j]
            if j < ns - 1:
                dense[rows, (j + 1) * block:(j + 2) * block] = upper[j]
        active = np.flatnonzero(np.asarray(pack(mask)).reshape(-1) != 0)
        dense_sigma = np.linalg.svd(dense[np.ix_(active, active)], compute_uv=False)
        spectrum_report["dense_svd_cross_check"] = {
            "active_dofs": int(active.size), "sigma_max": float(dense_sigma[0]),
            "sigma_smallest": [float(s) for s in dense_sigma[::-1][:spectrum_size]]}

    # ---- objectives: sensitivity, (f) the block adjoint and the production adjoint -----------
    ntor = cfg.resolution.ntor
    boundary = [(field, m, n) for field in ("rbc", "zbs") for m, n in ((0, 1), (1, -1), (1, 0), (1, 1))]
    zero = jax.tree.map(jnp.zeros_like, params)
    tangents = [replace(zero, **{field: getattr(zero, field).at[ntor + n, m].set(1.0)}) for field, m, n in boundary]

    @jax.jit
    def raw_parameter_jvp(z, frozen, direction):
        return jax.jvp(lambda prm: force_of(frozen, "raw")(z, prm), (params,), (direction,))[1]

    @functools.partial(jax.jit, static_argnames=("formulation",))
    def adjoint_parts(multiplier, b, z, frozen, formulation):
        """Independent adjoint residual and the implicit parameter pullback."""
        force = force_of(frozen, formulation)
        _, pull_state = jax.vjp(lambda t: force(t, params), z)
        defect = imp._tree_norm(add(P(pull_state(multiplier)[0]), b, -1.0)) / imp._tree_norm(b)
        _, pull_param = jax.vjp(lambda prm: force(z, prm), params)
        return pull_param(jax.tree.map(jnp.negative, multiplier))[0], defect

    block_multiplier = jax.jit(lambda factors, b: inverse(factors, b, transpose=True))

    def boundary_entries(tree):
        return np.asarray([float(getattr(tree, field)[ntor + n, m]) for field, m, n in boundary])

    def relative(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(b), 1e-300))

    def objective_pieces(term_list):
        # Resolved as make_problem does: residuals_state for term objects, row scale sqrt(w), cost 0.5 r.r.
        traced = [(core_optimize._traceable_term(f), float(t),
                   jnp.asarray(core_optimize._least_squares_weight(w, "cost"))) for f, t, w in term_list]

        def half_square(x, runtime):
            rows = jnp.concatenate([jnp.atleast_1d(w * (jnp.asarray(f(x, runtime)) - t)).ravel() for f, t, w in traced])
            return 0.5 * jnp.vdot(rows, rows)

        def value(z, prm, frozen):
            runtime = imp.runtime_from_params(prm, cfg)
            return half_square(imp._assemble(z, runtime, frozen, P, edge), runtime)

        return (jax.jit(value),
                jax.jit(lambda z, frozen, v: jax.jvp(lambda s: value(s, params, frozen), (z,), (v,))[1]),
                jax.jit(lambda z, frozen, t: jax.jvp(lambda prm: value(z, prm, frozen), (params,), (t,))[1]),
                jax.jit(lambda x: jax.grad(lambda s: half_square(s, imp.runtime_from_params(params, cfg)))(x)),
                jax.jit(lambda z, frozen: jax.grad(lambda prm: value(z, prm, frozen))(params)))

    def tangent_columns(z, frozen, factors):
        return [inverse(factors, jax.tree.map(jnp.negative, raw_parameter_jvp(z, frozen, t))) for t in tangents]

    def gradient_of(pieces, z, frozen, columns):
        return np.asarray([float(pieces[1](z, frozen, column)) + float(pieces[2](z, frozen, t))
                           for column, t in zip(columns, tangents)])

    def gradient_sensitivity(pieces, frozen, z, factors, V):
        columns = tangent_columns(z, frozen, factors)
        gradient = gradient_of(pieces, z, frozen, columns)
        defects = [norm(add(jvps(z, frozen, column)[0], raw_parameter_jvp(z, frozen, t)))
                   / max(norm(raw_parameter_jvp(z, frozen, t)), 1e-300) for column, t in zip(columns, tangents)]
        g_v = np.asarray([float(pieces[1](z, frozen, unravel(jnp.asarray(V[:, i])))) for i in range(V.shape[1])])
        amplitudes = np.asarray([np.asarray(V).T @ np.asarray(flat(column)) for column in columns])
        report = {"gradient": gradient.tolist(), "gradient_norm": float(np.linalg.norm(gradient)),
                  "block_solve_relative_defect_max": float(max(defects)),
                  "state_derivative_on_near_null": g_v.tolist(),
                  "tangent_amplitude_on_near_null_max": np.abs(amplitudes).max(axis=0).tolist()}
        for k in deflation_sizes:
            report[f"relative_gradient_through_{k}_smallest"] = float(
                np.linalg.norm(amplitudes[:, :k] @ g_v[:k]) / max(np.linalg.norm(gradient), 1e-300))
        return report

    pieces_by_name = {name: objective_pieces(terms(name)) for name in objectives}
    objective_report = {}
    for name, pieces in pieces_by_name.items():
        value_j, grad_param = pieces[0], pieces[4]
        base_value = float(value_j(z_deck, params, deck.state))
        entry = {"value": base_value,
                 "sensitivity_at_deck": gradient_sensitivity(pieces, deck.state, z_deck, factors_deck, V_deck),
                 "value_change_along_near_null": [
                     {str(epsilon): float(value_j(add(z_deck, unravel(jnp.asarray(V_deck[:, i])), epsilon),
                                                  params, deck.state)) - base_value for epsilon in (1e-4, 1e-2)}
                     for i in range(min(6, spectrum_size))]}
        b = P(pieces[3](deck.state))
        entry["adjoint_gcrot_history"] = history(b, rtol=cfg.adjoint_tol, transpose=True, cycles=history_cycles)
        direct = boundary_entries(grad_param(z_deck, deck.state))
        mu = block_multiplier(factors_deck, b)
        implicit_block, defect_block = adjoint_parts(mu, b, z_deck, deck.state, "raw")
        gradient_block = boundary_entries(implicit_block) + direct
        tangent_gradient = entry["sensitivity_at_deck"]["gradient"]
        t_solve = timed(block_multiplier, factors_deck, b, repeats=3)
        t_pullback = timed(adjoint_parts, mu, b, z_deck, deck.state, "raw", repeats=3)
        entry["f_block_adjoint"] = {
            "adjoint_relative_residual_raw": float(defect_block), "gradient": gradient_block.tolist(),
            "relative_difference_to_block_tangent_gradient": relative(gradient_block, tangent_gradient),
            "transposed_solve_seconds": t_solve, "pullback_seconds": t_pullback,
            "work_with_existing_factor": (t_solve + t_pullback) / t_force,
            "work_with_fresh_factor": (t_factor + t_solve + t_pullback) / t_force}
        if not smoke:
            (lam, adjoint), seconds, _ = counted(
                lambda: imp._adjoint_gcrot_core(params, z_deck, deck.state, mask, b, cfg))
            implicit_production, defect_production = adjoint_parts(lam, b, z_deck, deck.state, "preconditioned")
            gradient_production = boundary_entries(implicit_production) + direct
            entry["adjoint_production"] = {
                "seconds": seconds, "iterations": int(adjoint.iterations), "residual_norm": float(adjoint.residual_norm),
                "tolerance": float(adjoint.tolerance), "accepted": bool(adjoint.converged),
                "max_restarts": cfg.adjoint_maxiter, "m": cfg.adjoint_gcrot_m, "k": cfg.adjoint_gcrot_k,
                "rtol": cfg.adjoint_tol, "work": float(adjoint.iterations) * units["c_jvp"],
                "adjoint_relative_residual_preconditioned": float(defect_production),
                "gradient": gradient_production.tolist(),
                "relative_difference_to_block_adjoint": relative(gradient_production, gradient_block),
                "relative_difference_to_block_tangent_gradient": relative(gradient_production, tangent_gradient)}
        objective_report[name] = entry

    # ---- (g) refinement with a stagnation abort ---------------------------------------------
    def stagnation_stop(rows, window, drop):
        for index, (iterations, residual) in enumerate(rows):
            earlier = [r for its, r in rows[:index] if its <= iterations - window]
            if earlier and residual > (1.0 - drop) * earlier[-1]:
                return index + 1
        return None

    def refine_with_policy(abort):
        base = norm(fz)
        z, f_k, residual, best_z, best, steps = z_deck, fz, base, z_deck, base, []
        for step in range(imp._REFINE_MAX_STEPS):
            stop = None
            if abort is not None:
                rows = (refinement["gcrot_history_first_step"]["cycles"] if step == 0 else history(
                    f_k, z=z, rtol=imp._REFINE_FORCING, transpose=False, cycles=refine_cycles)["cycles"])
                stop = stagnation_stop(rows, *abort)
            x, _, linear, iterations, converged = gcrot_cycles(
                jnp.zeros_like(flat(f_k)), None, z, deck.state, f_k, imp._REFINE_FORCING, False, stop or refine_cycles)
            z = add(z, unravel(x), -1.0)
            f_k = residual_deck(z, params)
            previous, residual = residual, norm(f_k)
            steps.append({"cycles": stop or refine_cycles, "aborted_by_rule": stop is not None,
                          "iterations": int(iterations), "converged": bool(converged),
                          "linear_relative_residual": float(linear), "certificate": residual})
            if residual < best:
                best_z, best = z, residual
            if not math.isfinite(residual) or best <= cfg.refine_tol or (not bool(converged) and residual >= previous):
                break
        return (deck.state if best >= base else add(deck.state, P(add(best_z, z_deck, -1.0)))), steps

    def anchor_outputs(state):
        z = P(state)
        columns = tangent_columns(z, state, block_system(state, z))
        outputs = {"state": np.asarray(flat(state)), "residual": np.asarray(flat(evaluate(z, state)[0])),
                   "jacobian": np.stack([np.asarray(flat(column)) for column in columns])}
        for name, pieces in pieces_by_name.items():
            outputs[f"value_{name}"] = np.asarray(pieces[0](z, params, state))
            outputs[f"gradient_{name}"] = gradient_of(pieces, z, state, columns)
        return outputs

    abort_report, reference = {}, None
    for label, abort in (("pr320", None), ("window_1000_drop_25", (1000, 0.25)), ("window_500_drop_25", (500, 0.25))):
        state, steps = refine_with_policy(abort)
        outputs = anchor_outputs(state)
        reference = reference or outputs
        total = sum(s["iterations"] for s in steps)
        abort_report[label] = {
            "steps": steps, "gcrot_iterations": total,
            "iterations_saved_vs_pr320": abort_report["pr320"]["gcrot_iterations"] - total if abort else 0,
            "bit_identical_to_pr320": {key: bool(np.array_equal(outputs[key], reference[key])) for key in outputs}}
    abort_report["pr320"]["state_bit_identical_to_main_refined_state"] = bool(
        np.array_equal(np.asarray(flat(refined)), reference["state"]))

    # ---- Newton arms (b), (c), (e) and the Anderson control (d) ------------------------------
    newton_config = Prec2DConfig(threshold=np.inf, gmres_restart=50, gmres_max_restarts=20, gmres_rtol=NEWTON_RTOL)

    @jax.jit
    def direction_1d(z, frozen):
        force = force_of(frozen)
        rhs = jax.tree.map(jnp.negative, force(z, params))
        step, solution = newton_direction(lambda t: force(t, params), z, rhs, newton_config)
        return step, solution.iterations, solution.residual_norm / imp._tree_norm(rhs)

    @jax.jit
    def direction_block(z, frozen, factors, V, U):
        value, linear = jax.linearize(lambda t: force_of(frozen, "raw")(t, params), z)
        left, right = (lambda w: w - U @ (U.T @ w)), (lambda w: w - V @ (V.T @ w))
        rhs = left(flat(value))
        solution = solvax.gmres(lambda y: left(flat(linear(unravel(right(y))))), -rhs,
                                precond=lambda w: right(flat(inverse(factors, unravel(left(w))))),
                                restart=50, rtol=NEWTON_RTOL, max_restarts=20)
        return unravel(right(solution.x)), solution.iterations, solution.residual_norm / jnp.linalg.norm(rhs)

    @jax.jit
    def projected_certificate(z, frozen, Up):
        force, _, bad = evaluate(z, frozen)
        f = flat(force)
        return jnp.linalg.norm(f - Up @ (Up.T @ f)), jnp.linalg.norm(f), bad

    def newton(frozen, direction, measure, steps_max, factorizations):
        z = P(frozen)
        current, full, _ = measure(z)
        current, full = float(current), float(full)
        initial, steps, jvp_count, evaluations, bad = current, [], 0, 1, False
        for _ in range(steps_max):
            step, matvecs, linear = direction(z)
            jvp_count, evaluations, alpha = jvp_count + int(matvecs), evaluations + 1, 1.0
            while True:
                trial = add(z, step, alpha)
                value, full, sign = measure(trial)
                value, full, sign, evaluations = float(value), float(full), bool(sign), evaluations + 1
                if (math.isfinite(value) and not sign and value < (1.0 - 1e-4 * alpha) * current) or alpha < 1e-3:
                    break
                alpha *= 0.5
            bad |= sign
            steps.append({"matvecs": int(matvecs), "linear_relative_residual": float(linear), "alpha": alpha,
                          "measured": value, "certificate": full, "step_norm": norm(step)})
            z, current = trial, value
            if current <= cfg.refine_tol:
                break
        fsq = [initial ** 2] + [s["measured"] ** 2 for s in steps]
        slow = sum(fsq[k + 1] > fsq[k] / 10.0 for k in range(min(3, len(steps))))
        return z, {"steps": steps, "jvps": jvp_count, "force_evaluations": evaluations,
                   "certified": current <= cfg.refine_tol, "final_measured": current, "final_certificate": full,
                   "jacobian_sign_change": bad, "distance_to_refined": norm(add(z, P(refined), -1.0)),
                   "kills": {"matvecs_over_50_per_step": any(s["matvecs"] > 50 for s in steps),
                             "fsq_fell_under_10x_in_2_of_first_3": slow >= 2, "jacobian_reset": bad},
                   "work": evaluations + units["c_jvp"] * jvp_count + units["f_factorization"] * factorizations}

    window, max_evaluations, alpha_d = 8, 400, float(deck.time_step) ** 2

    @jax.jit
    def anderson_run(flat0, frozen):
        def mapped(v):
            force, fsq, bad = evaluate(unravel(v), frozen)
            return flat(force), fsq, bad

        force0, fsq0, bad0 = mapped(flat0)
        history_x = jnp.zeros((window, flat0.size)).at[0].set(flat0)
        history_r = jnp.zeros((window, flat0.size)).at[0].set(alpha_d * force0)

        def body(carry):
            count, history_x, history_r, _, _, _ = carry
            valid = (jnp.arange(window) < jnp.minimum(count, window))[:, None]
            new = solvax.anderson_mixing(jnp.where(valid, history_x, history_x[0]),
                                         jnp.where(valid, history_r, history_r[0]))
            force, fsq, bad = mapped(new)
            return (count + 1, history_x.at[count % window].set(new),
                    history_r.at[count % window].set(alpha_d * force), fsq, jnp.linalg.norm(force), bad)

        return jax.lax.while_loop(lambda c: (c[0] < max_evaluations) & (c[3] > cfg.ftol) & ~c[5], body,
                                  (jnp.asarray(1), history_x, history_r, fsq0, jnp.linalg.norm(force0), bad0))

    no_modes = jnp.zeros((z_deck.R_cos.size * len(imp._STATE_FIELDS), 0))
    starts = {}
    for label, run in runs.items():
        frozen, z0 = run.state, P(run.state)
        entry = {"descent_iterations_at_stop": int(run.iterations), "certificate": certificate(z0, frozen)[0],
                 "fsq": certificate(z0, frozen)[1]}
        if label != "deck":
            left = int(deck.iterations - run.iterations)
            count, _, _, fsq, force_norm, bad = anderson_run(flat(z0), frozen)
            entry["a_descent_to_deck"] = {"descent_iterations": left, "work_with_refinement": left + refinement["work"],
                                          "final_certificate": refinement["certificate_after"],
                                          "certified": refinement["certificate_after"] <= cfg.refine_tol}
            entry["d_anderson"] = {"force_evaluations": int(count), "fsq": float(fsq),
                                   "final_certificate": float(force_norm), "jacobian_sign_change": bool(bad),
                                   "reached_deck_fsq": float(fsq) <= cfg.ftol,
                                   "certified": float(force_norm) <= cfg.refine_tol,
                                   "evaluation_ratio_vs_descent": left / int(count)}
        plain = lambda z, f=frozen: (lambda r: (imp._tree_norm(r[0]), imp._tree_norm(r[0]), r[2]))(evaluate(z, f))  # noqa: E731
        _, entry["b_newton_1d"] = newton(frozen, lambda z, f=frozen: direction_1d(z, f), plain, newton_steps["b"], 0)
        factors = block_system(frozen, z0)
        _, entry["c_newton_block"] = newton(
            frozen, lambda z, f=frozen, g=factors: direction_block(z, f, g, no_modes, no_modes), plain,
            newton_steps["c"], 1)
        sigma, V, JV, JpV = spectrum(frozen, z0, factors, max(deflation_sizes))
        entry["sigma_smallest"], entry["e_newton_deflated"] = [float(s) for s in sigma], {}
        for k in deflation_sizes:
            Vk, Uk, Upk = jnp.asarray(V[:, :k]), orthonormal(JV[:, :k]), orthonormal(JpV[:, :k])
            z_final, result = newton(
                frozen, lambda z, f=frozen, g=factors, a=Vk, b=Uk: direction_block(z, f, g, a, b),
                lambda z, f=frozen, u=Upk: projected_certificate(z, f, u), newton_steps["e"], 1)
            displacement = np.asarray(flat(add(z_final, z0, -1.0)))
            result["displacement_norm"] = float(np.linalg.norm(displacement))
            result["displacement_on_near_null"] = float(np.linalg.norm(np.asarray(Vk).T @ displacement))
            if label == "deck":
                factors_final = block_system(frozen, z_final)
                result["objectives_at_final"] = {name: {
                    "value_change": float(pieces[0](z_final, params, frozen)) - objective_report[name]["value"],
                    "gradient_relative_change": relative(
                        gradient_sensitivity(pieces, frozen, z_final, factors_final, V_deck)["gradient"],
                        objective_report[name]["sensitivity_at_deck"]["gradient"])}
                    for name, pieces in pieces_by_name.items()}
            entry["e_newton_deflated"][str(k)] = result
        starts[label] = entry
        print(f"start {label} done", file=sys.stderr, flush=True)

    report = {
        "case": args.case, "smoke": smoke, "input_sha256": file_sha256(path),
        "resolution": {"mpol": cfg.resolution.mpol, "ntor": cfg.resolution.ntor, "ns": cfg.resolution.ns,
                       "nfp": cfg.resolution.nfp},
        "deck_ftol": cfg.ftol, "refine_tol": cfg.refine_tol, "descent_iterations_deck": int(deck.iterations),
        "unit_costs": units, "refinement_from_deck": refinement, "refinement_abort": abort_report,
        "raw_jacobian_spectrum_at_deck": spectrum_report, "objectives": objective_report, "starts": starts,
        "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "compilation_cache": os.environ.get("VMEX_COMPILATION_CACHE"),
        "load_average": {"start": load_start, "end": os.getloadavg()}, "platform": platform.platform(),
        "versions": {"python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__,
                     "vmex": vmex.__version__, "solvax": getattr(solvax, "__version__", None)},
        **git_state(REPO), "vmex_module": assert_repo_vmex(vmex.__file__, REPO)}
    if args.output:  # the raw report; the committed record stores it through compact_json
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    print(compact_json(json.loads(json.dumps(report, default=float))))


if __name__ == "__main__":
    main()

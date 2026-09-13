#!/usr/bin/env python
"""Plan B1, first step: can a Newton finish replace the fixed-point refinement?

For one deck, save descent states at ``fsq`` = 1e-6, 1e-8 and 1e-10 (the cold
multigrid trajectory stopped early, so descent counts are exact differences
along one trajectory) and the deck-tolerance state. From each start run:

(a) today: descent iterations to the deck tolerance, then ``_refined_state``;
(b) Newton on the implicit (1-D preconditioned) residual with the
    ``newton_direction`` lane of ``_newton_step``: GMRES(50) at rtol 1e-3,
    at most 20 cycles, backtracking on |F|;
(c) Newton on the raw residual, GMRES right-preconditioned by the exact block
    tridiagonal factorization (``_raw_block_system``) built once per start;
(d) Anderson acceleration (window 8) of ``z + dt^2 F(z)`` in one
    ``lax.while_loop``, stopped at the deck ``fsq``;
(e) arm (c) on the complement of the near-null space: the step is confined
    to the complement of the ``k`` smallest right singular vectors ``V`` of
    the raw Jacobian, the raw residual to the complement of their left
    vectors ``U``, and the line search and certificate use ``|P gc|`` with
    the images ``J_p V`` of those modes removed.

The certificate is ``|P gc|`` (``refine_tol`` = 1e-10). Work is
``W = F + c JVP + f factor`` with ``c`` and ``f`` the warm cost ratios
measured in the same process; the kill rules of plan B1 are evaluated on
``fsq = |F|^2``. At the deck state the script also records the smallest
singular values of the raw Jacobian (Lanczos on ``(J^T J)^-1`` through the
block factorization, cross-checked by a dense SVD on the seed deck) with the
mode content of their vectors; the change of both residuals along each
near-null mode against the step length; the per-cycle relative residual of
the refinement's and the adjoint's GCROT solves (cycles run one at a time
with the recycle space carried, so each warm start adds ``k`` operator
applications that ``iterations`` excludes); the production adjoint GCROT for
the QA and QI objectives; and how much of the QA and QI boundary gradients
flows through the near-null modes. Two further arms:

(f) the adjoint through the transposed block factorization. At the root
    ``dF_p/dp = M dF_raw/dp`` with ``M`` the 1-D preconditioner, so
    ``lambda^T dF_p/dp = (M^T lambda)^T dF_raw/dp`` and the gradient needs only
    ``mu = J_raw^-T P gbar``: one transposed block solve and one parameter
    pullback, with no preconditioner application. The adjoint residual is
    measured by an independent pullback, and the boundary gradient is
    compared with the block-tangent gradient and the production GCROT adjoint;
(g) refinement with a stagnation abort: a GCROT solve stops at the first cycle
    whose relative residual fell less than 25 % over the last 1,000 (or 500)
    iterations. The policies are simulated with one-shot solves cut at that
    cycle, and the returned state, residual, block-tangent Jacobian columns,
    and QA and QI values and gradients are compared bit for bit with #320's
    policy (stop after an unconverged step that does not lower ``|F|``).

Timings are diagnostic under shared load.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import functools
import json
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


def build_case(case: str):
    if case == "seed":  # the equilibrium of the QA and QI rows of optimization.py
        path = REPO / "examples/data/input.minimal_seed_nfp2"
        inp = replace(VmecInput.from_file(path), delt=0.5).change_resolution(
            mpol=5, ntor=5, ntheta=16, nzeta=14)
        forward_ftol = None  # the factory's 1e-12
    else:
        path = REPO / "examples/data/input.LandremanPaul2021_QA_lowres"
        inp = VmecInput.from_file(path)
        forward_ftol = 1e-11  # plan §2: the shipped 1e-13 is unreachable
    problem = opt.VmecProblem.from_tuples(
        inp, terms("qa"), max_mode=1, use_ess=True, forward_ftol=forward_ftol,
        forward_max_iterations=5500)
    return path, problem.metadata["config"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("seed", "qa_lowres"), default="seed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true",
                        help="one start, one step per arm, few modes: checks the script only")
    args = parser.parse_args()
    smoke = args.smoke
    starts_ftol = (1e-8,) if smoke else (1e-6, 1e-8, 1e-10)
    newton_steps = dict(b=1, c=1, e=1) if smoke else dict(b=3, c=8, e=8)
    spectrum_size = 4 if smoke else 16
    deflation_sizes = (2,) if smoke else (2, 6, 12)
    epsilons = (1e-4,) if smoke else (1e-8, 1e-6, 1e-4, 1e-2)
    history_cycles = 2 if smoke else 60
    objectives = ("qa",) if smoke else ("qa", "qi")
    load_start = os.getloadavg()

    path, cfg = build_case(args.case)
    params = imp.params_from_input(cfg.inp, device=cfg.device)
    mask = jax.tree.map(jnp.asarray, imp._MASK_CACHE[imp._mask_cache_key(cfg)])
    P = imp._dof_projector(cfg, mask)
    fields = imp._active_state_fields(cfg)
    edge = imp._edge_mask(cfg)
    modes = imp._template_runtime(cfg).modes
    mode_m, mode_n = np.asarray(modes.m), np.asarray(modes.n)
    pack = lambda tree: imp._pack_active(cfg, tree)  # noqa: E731
    unpack = lambda matrix: imp._unpack_active(cfg, matrix)  # noqa: E731
    norm = lambda tree: float(imp._tree_norm(tree))  # noqa: E731
    flat = lambda tree: ravel_pytree(tree)[0]  # noqa: E731

    def descent(ftol):
        ftol_array = np.asarray(cfg.inp.ftol_array, dtype=float).copy()
        ftol_array[-1] = ftol
        return solve_multigrid(
            cfg.inp, ns_array=np.asarray(cfg.inp.ns_array), ftol_array=ftol_array,
            mode=cfg.mode, lconm1=cfg.lconm1, raise_on_max_iterations=False,
            use_fft=False)

    deck = imp._LAST_SOLVE[cfg][1]
    runs = {ftol: descent(ftol) for ftol in starts_ftol}
    z_deck = P(deck.state)
    unravel = ravel_pytree(z_deck)[1]

    # ---- compiled pieces ---------------------------------------------------------
    @jax.jit
    def evaluate(z, frozen):
        runtime = imp.runtime_from_params(params, cfg)
        x = imp._assemble(z, runtime, frozen, P, edge)
        gc, raw, diagnostics = evaluate_forces(x, runtime)
        return (P(gc), raw.fsqr + raw.fsqz + raw.fsql,
                diagnostics.jacobian_sign_changed)

    def certificate(z, frozen):
        force, fsq, bad = evaluate(z, frozen)
        return norm(force), float(fsq), bool(bad)

    @jax.jit
    def residuals(z, frozen):
        raw = imp.residual_fn(cfg, frozen, mask, formulation="raw")
        return raw(z, params), imp.residual_fn(cfg, frozen, mask)(z, params)

    @jax.jit
    def jvps(z, frozen, tangent):
        raw = imp.residual_fn(cfg, frozen, mask, formulation="raw")
        preconditioned = imp.residual_fn(cfg, frozen, mask)
        return (jax.jvp(lambda t: raw(t, params), (z,), (tangent,))[1],
                jax.jvp(lambda t: preconditioned(t, params), (z,), (tangent,))[1])

    @jax.jit
    def raw_normal(z, frozen, tangent):
        raw = imp.residual_fn(cfg, frozen, mask, formulation="raw")
        value, pullback = jax.vjp(lambda t: raw(t, params), z)
        return P(pullback(jax.jvp(lambda t: raw(t, params), (z,), (P(tangent),))[1])[0])

    @jax.jit
    def block_system(frozen, z):
        system = imp._raw_block_system(params, cfg, frozen, mask, fields, 32, z_star=z)
        return system.factors, system.row_scale, system.column_scale

    def inverse(factors, tree, transpose=False):
        blocks, row_scale, column_scale = factors
        return imp._block_inverse_apply(blocks, pack, unpack, P, row_scale, column_scale,
                                        tree, transpose=transpose)

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

    # ---- warm unit costs ---------------------------------------------------------
    residual_deck = imp.residual_fn(cfg, deck.state, mask)
    jvp_only = jax.jit(lambda z, v: jax.jvp(lambda t: residual_deck(t, params), (z,), (v,)))
    t_force = timed(evaluate, z_deck, deck.state)
    t_jvp = timed(jvp_only, z_deck, z_deck)
    t_factor = timed(block_system, deck.state, z_deck, repeats=3)
    fz = residual_deck(z_deck, params)
    imp._refine_step(cfg, params, deck.state, mask, z_deck, fz)
    before = dict(imp._SOLVE_STATS[cfg])
    started = time.perf_counter()
    np.asarray(imp._refine_step(cfg, params, deck.state, mask, z_deck, fz)[2])
    t_refine_step = time.perf_counter() - started
    refine_iterations = max(1, imp._SOLVE_STATS[cfg]["refinement_krylov_iterations"]
                            - before["refinement_krylov_iterations"])
    units = {
        "force_evaluation_seconds": t_force, "jvp_seconds": t_jvp,
        "block_build_factor_seconds": t_factor,
        "refinement_gcrot_seconds_per_iteration": t_refine_step / refine_iterations,
        "c_jvp": t_jvp / t_force,
        "c_refinement_iteration": t_refine_step / refine_iterations / t_force,
        "f_factorization": t_factor / t_force,
    }

    # ---- (a) today's refinement from the deck state --------------------------------
    before = dict(imp._SOLVE_STATS[cfg])
    started = time.perf_counter()
    refined = imp._refined_state(cfg, params, deck.state, mask)
    after = imp._SOLVE_STATS[cfg]
    refinement = {
        "seconds": time.perf_counter() - started,
        "steps": after["refinement_steps"] - before["refinement_steps"],
        "gcrot_iterations": after["refinement_krylov_iterations"]
        - before["refinement_krylov_iterations"],
        "certificate_before": certificate(z_deck, deck.state)[0],
        "certificate_after": certificate(P(refined), deck.state)[0],
        "state_moved": norm(jax.tree.map(jnp.subtract, P(refined), z_deck)),
    }
    refinement["work"] = (1 + 2 * refinement["steps"]
                          + units["c_refinement_iteration"] * refinement["gcrot_iterations"])

    # ---- per-cycle GCROT histories -------------------------------------------------
    @functools.partial(jax.jit, static_argnames=("rtol", "transpose", "max_restarts"))
    def gcrot_cycles(x0, recycle, z, frozen, b, rtol, transpose, max_restarts):
        force = imp.residual_fn(cfg, frozen, mask)
        if transpose:
            _, pullback = jax.vjp(lambda t: force(t, params), z)
            operator = lambda v: pullback(v)[0]  # noqa: E731
        else:
            _, operator = jax.linearize(lambda t: force(t, params), z)
        b_flat = flat(b)
        n = int(b_flat.shape[0])
        solution = solvax.gcrot(
            lambda v: flat(operator(unravel(v))), b_flat, x0=x0,
            m=min(cfg.adjoint_gcrot_m, n), k=min(cfg.adjoint_gcrot_k, n),
            rtol=rtol, max_restarts=max_restarts, recycle=recycle)
        return (solution.x, solution.recycle,
                solution.residual_norm / jnp.linalg.norm(b_flat),
                solution.iterations, solution.converged)

    def history(b, *, rtol, transpose, cycles, z=None):
        z = z_deck if z is None else z
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
        return {"cycles": rows, "one_shot_same_cycles": {
            "iterations": int(iterations), "relative_residual": float(relative),
            "converged": bool(converged)}}

    refinement["gcrot_history_first_step"] = history(
        fz, rtol=imp._REFINE_FORCING, transpose=False,
        cycles=2 if smoke else imp._REFINE_MAX_RESTARTS)

    # ---- spectrum, vectors and gauge test at a state ---------------------------------
    def describe(vector):
        tree = unravel(jnp.asarray(vector))
        energy = np.stack([np.asarray(getattr(tree, name)) ** 2 for name in fields])
        total = energy.sum()
        top = np.argsort(-energy.ravel())[:3]
        ns = energy.shape[1]
        return {
            "dominant": [
                f"{fields[f]}[s={s},m={int(mode_m[k])},n={int(mode_n[k])}]"
                for f, s, k in (np.unravel_index(i, energy.shape) for i in top)],
            "field_fraction": {name: float(energy[i].sum() / total) for i, name in enumerate(fields)},
            "fraction_last_rows": {str(r): float(energy[:, ns - r:].sum() / total) for r in (1, 3, 6)},
        }

    def spectrum(frozen, z, factors, count):
        n = int(flat(z).shape[0])
        operator = LinearOperator((n, n), dtype=float, matvec=lambda v: np.asarray(
            normal_inverse(factors, jnp.asarray(v.ravel()))))
        values, vectors = eigsh(operator, k=count, which="LA", tol=1e-12)
        order = np.argsort(-values)
        sigma = 1.0 / np.sqrt(values[order])
        V = vectors[:, order]
        left = np.stack([flat(jvps(z, frozen, unravel(jnp.asarray(V[:, i])))[0])
                         for i in range(count)], axis=1)
        pre = np.stack([flat(jvps(z, frozen, unravel(jnp.asarray(V[:, i])))[1])
                        for i in range(count)], axis=1)
        return sigma, V, left, pre

    def orthonormal(matrix):
        return jnp.asarray(np.linalg.qr(np.asarray(matrix))[0])

    factors_deck = block_system(deck.state, z_deck)
    sigma_deck, V_deck, JV_deck, JpV_deck = spectrum(deck.state, z_deck, factors_deck, spectrum_size)
    sigma_max = float(np.sqrt(eigsh(LinearOperator(
        (V_deck.shape[0],) * 2, dtype=float, matvec=lambda v: np.asarray(flat(raw_normal(
            z_deck, deck.state, unravel(jnp.asarray(v.ravel())))))), k=1, which="LA",
        tol=1e-6)[0][0]))
    raw_deck, pre_deck = residuals(z_deck, deck.state)
    gauge = []
    for i in range(min(6, spectrum_size)):
        v = unravel(jnp.asarray(V_deck[:, i]))
        rows = []
        for epsilon in epsilons:
            raw_e, pre_e = residuals(jax.tree.map(lambda a, b: a + epsilon * b, z_deck, v), deck.state)
            rows.append({
                "epsilon": epsilon,
                "raw_change_over_epsilon": norm(jax.tree.map(jnp.subtract, raw_e, raw_deck)) / epsilon,
                "preconditioned_change_over_epsilon":
                    norm(jax.tree.map(jnp.subtract, pre_e, pre_deck)) / epsilon})
        gauge.append({"sigma": float(sigma_deck[i]),
                      "raw_jacobian_image_norm": float(np.linalg.norm(JV_deck[:, i])),
                      "preconditioned_jacobian_image_norm": float(np.linalg.norm(JpV_deck[:, i])),
                      "residual_change": rows, **describe(V_deck[:, i])})
    raw_flat, pre_flat = np.asarray(flat(raw_deck)), np.asarray(flat(pre_deck))
    spectrum_report = {
        "sigma_max": sigma_max,
        "sigma_smallest": [float(s) for s in sigma_deck],
        "consecutive_ratio": [float(sigma_deck[i + 1] / sigma_deck[i]) for i in range(spectrum_size - 1)],
        "residual_share_on_left_near_null": {
            str(k): {"raw": float(np.linalg.norm(np.asarray(orthonormal(JV_deck[:, :k])).T @ raw_flat)
                              / np.linalg.norm(raw_flat)),
                     "preconditioned": float(np.linalg.norm(np.asarray(orthonormal(JpV_deck[:, :k])).T
                                                            @ pre_flat) / np.linalg.norm(pre_flat))}
            for k in deflation_sizes},
        "modes": gauge,
        "edge_lambda_dof_rows_active": bool(np.asarray(mask.L_sin)[-1].any()),
    }
    if args.case == "seed" and not smoke:
        @jax.jit
        def bands(frozen):
            system = imp._raw_block_system(params, cfg, frozen, mask, fields, 32, factor=False)
            return system.lower, system.diagonal, system.upper

        lower, diagonal, upper = (np.asarray(a) for a in bands(deck.state))
        ns, block, _ = diagonal.shape
        dense = np.zeros((ns * block, ns * block))
        for j in range(ns):
            dense[j * block:(j + 1) * block, j * block:(j + 1) * block] = diagonal[j]
            if j:
                dense[j * block:(j + 1) * block, (j - 1) * block:j * block] = lower[j]
            if j < ns - 1:
                dense[j * block:(j + 1) * block, (j + 1) * block:(j + 2) * block] = upper[j]
        active = np.flatnonzero(np.asarray(pack(mask)).reshape(-1) != 0)
        dense_sigma = np.linalg.svd(dense[np.ix_(active, active)], compute_uv=False)
        spectrum_report["dense_svd_cross_check"] = {
            "active_dofs": int(active.size), "sigma_max": float(dense_sigma[0]),
            "sigma_smallest": [float(s) for s in dense_sigma[::-1][:spectrum_size]]}

    # ---- objectives: gradient sensitivity and the production adjoint ------------------
    term_sets = {name: terms(name) for name in objectives}
    boundary = [(field, m, n) for field in ("rbc", "zbs") for m, n in ((0, 1), (1, -1), (1, 0), (1, 1))]
    ntor = cfg.resolution.ntor

    def tangent(field, m, n):
        zero = jax.tree.map(jnp.zeros_like, params)
        return replace(zero, **{field: getattr(zero, field).at[ntor + n, m].set(1.0)})

    tangents = [tangent(*entry) for entry in boundary]

    @jax.jit
    def raw_parameter_jvp(z, frozen, direction):
        raw = imp.residual_fn(cfg, frozen, mask, formulation="raw")
        return jax.jvp(lambda prm: raw(z, prm), (params,), (direction,))[1]

    def objective_pieces(term_list):
        # Resolved as make_problem does: residuals_state for term objects and
        # the default cost semantics (row scale sqrt(w)); cost 0.5 r.r.
        traced = [(core_optimize._traceable_term(f), float(t),
                   jnp.asarray(core_optimize._least_squares_weight(w, "cost")))
                  for f, t, w in term_list]

        def half_square(x, runtime):
            rows = jnp.concatenate([jnp.atleast_1d(w * (jnp.asarray(f(x, runtime)) - t)).ravel()
                                    for f, t, w in traced])
            return 0.5 * jnp.vdot(rows, rows)

        def value(z, prm, frozen):
            runtime = imp.runtime_from_params(prm, cfg)
            return half_square(imp._assemble(z, runtime, frozen, P, edge), runtime)

        value_j = jax.jit(value)
        d_state = jax.jit(lambda z, frozen, v: jax.jvp(lambda s: value(s, params, frozen), (z,), (v,))[1])
        d_param = jax.jit(lambda z, frozen, t: jax.jvp(lambda prm: value(z, prm, frozen), (params,), (t,))[1])
        cotangent = jax.jit(lambda x: jax.grad(
            lambda s: half_square(s, imp.runtime_from_params(params, cfg)))(x))
        grad_param = jax.jit(lambda z, frozen: jax.grad(lambda prm: value(z, prm, frozen))(params))
        return value_j, d_state, d_param, cotangent, grad_param

    @functools.partial(jax.jit, static_argnames=("formulation",))
    def adjoint_parts(multiplier, b, z, frozen, formulation):
        """Independent adjoint residual and the implicit parameter pullback."""
        force = imp.residual_fn(cfg, frozen, mask, formulation=formulation)
        _, pull_state = jax.vjp(lambda t: force(t, params), z)
        defect = imp._tree_norm(jax.tree.map(
            jnp.subtract, P(pull_state(multiplier)[0]), b)) / imp._tree_norm(b)
        _, pull_param = jax.vjp(lambda prm: force(z, prm), params)
        return pull_param(jax.tree.map(jnp.negative, multiplier))[0], defect

    @jax.jit
    def block_multiplier(factors, b):
        return inverse(factors, b, transpose=True)

    def boundary_entries(tree):
        return np.asarray([float(getattr(tree, field)[ntor + n, m]) for field, m, n in boundary])

    def relative(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(b), 1e-300))

    def gradient_sensitivity(pieces, frozen, z, factors, V):
        d_state, d_param = pieces[1], pieces[2]
        gradient, amplitudes, defects = [], [], []
        g_v = np.asarray([float(d_state(z, frozen, unravel(jnp.asarray(V[:, i]))))
                          for i in range(V.shape[1])])
        for direction in tangents:
            rhs = jax.tree.map(jnp.negative, raw_parameter_jvp(z, frozen, direction))
            dz = inverse(factors, rhs)
            defects.append(norm(jax.tree.map(jnp.subtract, jvps(z, frozen, dz)[0], rhs)) / max(norm(rhs), 1e-300))
            gradient.append(float(d_state(z, frozen, dz)) + float(d_param(z, frozen, direction)))
            amplitudes.append(np.asarray(V).T @ np.asarray(flat(dz)))
        gradient, amplitudes = np.asarray(gradient), np.asarray(amplitudes)
        report = {"gradient": gradient.tolist(), "gradient_norm": float(np.linalg.norm(gradient)),
                  "block_solve_relative_defect_max": float(max(defects)),
                  "state_derivative_on_near_null": g_v.tolist(),
                  "tangent_amplitude_on_near_null_max": np.abs(amplitudes).max(axis=0).tolist()}
        for k in deflation_sizes:
            through = amplitudes[:, :k] @ g_v[:k]
            report[f"relative_gradient_through_{k}_smallest"] = float(
                np.linalg.norm(through) / max(np.linalg.norm(gradient), 1e-300))
        return report

    objective_report = {}
    pieces_by_name = {name: objective_pieces(term_list) for name, term_list in term_sets.items()}
    for name, pieces in pieces_by_name.items():
        value_j, cotangent, grad_param = pieces[0], pieces[3], pieces[4]
        entry = {"value": float(value_j(z_deck, params, deck.state)),
                 "sensitivity_at_deck": gradient_sensitivity(pieces, deck.state, z_deck, factors_deck, V_deck),
                 "value_change_along_near_null": [
                     {str(epsilon): float(value_j(jax.tree.map(
                         lambda a, b: a + epsilon * b, z_deck, unravel(jnp.asarray(V_deck[:, i]))),
                         params, deck.state)) - float(value_j(z_deck, params, deck.state))
                      for epsilon in (1e-4, 1e-2)} for i in range(min(6, spectrum_size))]}
        b = P(cotangent(deck.state))
        entry["adjoint_gcrot_history"] = history(
            b, rtol=cfg.adjoint_tol, transpose=True, cycles=history_cycles)
        # (f) the adjoint through the transposed block factorization
        direct = boundary_entries(grad_param(z_deck, deck.state))
        mu = block_multiplier(factors_deck, b)
        implicit_block, defect_block = adjoint_parts(mu, b, z_deck, deck.state, "raw")
        gradient_block = boundary_entries(implicit_block) + direct
        tangent_gradient = entry["sensitivity_at_deck"]["gradient"]
        t_solve = timed(block_multiplier, factors_deck, b, repeats=3)
        t_pullback = timed(adjoint_parts, mu, b, z_deck, deck.state, "raw", repeats=3)
        entry["f_block_adjoint"] = {
            "adjoint_relative_residual_raw": float(defect_block),
            "gradient": gradient_block.tolist(),
            "relative_difference_to_block_tangent_gradient": relative(gradient_block, tangent_gradient),
            "transposed_solve_seconds": t_solve, "pullback_seconds": t_pullback,
            "work_with_existing_factor": (t_solve + t_pullback) / t_force,
            "work_with_fresh_factor": (t_factor + t_solve + t_pullback) / t_force,
        }
        if not smoke:
            started = time.perf_counter()
            lam, stats = imp._adjoint_gcrot_core(params, z_deck, deck.state, mask, b, cfg)
            seconds = time.perf_counter() - started
            implicit_production, defect_production = adjoint_parts(
                lam, b, z_deck, deck.state, "preconditioned")
            gradient_production = boundary_entries(implicit_production) + direct
            entry["adjoint_production"] = {
                "seconds": seconds, "iterations": int(stats.iterations),
                "residual_norm": float(stats.residual_norm), "tolerance": float(stats.tolerance),
                "accepted": bool(stats.converged), "max_restarts": cfg.adjoint_maxiter,
                "m": cfg.adjoint_gcrot_m, "k": cfg.adjoint_gcrot_k, "rtol": cfg.adjoint_tol,
                "work": float(stats.iterations) * units["c_jvp"],
                "adjoint_relative_residual_preconditioned": float(defect_production),
                "gradient": gradient_production.tolist(),
                "relative_difference_to_block_adjoint": relative(gradient_production, gradient_block),
                "relative_difference_to_block_tangent_gradient": relative(gradient_production, tangent_gradient),
            }
        objective_report[name] = entry

    # ---- (g) refinement with a stagnation abort ---------------------------------------
    def stagnation_stop(rows, window, drop):
        """First cycle (1-based) whose residual fell less than ``drop`` over ``window`` iterations."""
        for index, (iterations, relative_residual) in enumerate(rows):
            earlier = [r for its, r in rows[:index] if its <= iterations - window]
            if earlier and relative_residual > (1.0 - drop) * earlier[-1]:
                return index + 1
        return None

    def refine_with_policy(abort):
        cycles_max = 2 if smoke else imp._REFINE_MAX_RESTARTS
        base = norm(fz)
        z, f_k, residual, best_z, best, steps = z_deck, fz, base, z_deck, base, []
        for step in range(imp._REFINE_MAX_STEPS):
            cycles, stop = cycles_max, None
            if abort is not None:
                rows = (refinement["gcrot_history_first_step"]["cycles"] if step == 0 else history(
                    f_k, z=z, rtol=imp._REFINE_FORCING, transpose=False, cycles=cycles_max)["cycles"])
                stop = stagnation_stop(rows, *abort)
                cycles = stop or cycles_max
            x, _, linear, iterations, converged = gcrot_cycles(
                jnp.zeros_like(flat(f_k)), None, z, deck.state, f_k, imp._REFINE_FORCING, False, cycles)
            z = jax.tree.map(jnp.subtract, z, unravel(x))
            f_k = residual_deck(z, params)
            previous, residual = residual, norm(f_k)
            steps.append({"cycles": cycles, "aborted_by_rule": stop is not None,
                          "iterations": int(iterations), "converged": bool(converged),
                          "linear_relative_residual": float(linear), "certificate": residual})
            if not np.isfinite(residual):
                break
            if residual < best:
                best_z, best = z, residual
            if best <= cfg.refine_tol:
                break
            if not bool(converged) and residual >= previous:
                break
        if best >= base:
            return deck.state, steps
        return jax.tree.map(jnp.add, deck.state, P(jax.tree.map(jnp.subtract, best_z, z_deck))), steps

    def anchor_outputs(state):
        z = P(state)
        factors = block_system(state, z)
        columns = [inverse(factors, jax.tree.map(jnp.negative, raw_parameter_jvp(z, state, t)))
                   for t in tangents]
        outputs = {"state": np.asarray(flat(state)), "residual": np.asarray(flat(evaluate(z, state)[0])),
                   "jacobian": np.stack([np.asarray(flat(column)) for column in columns])}
        for name, (value_j, d_state, d_param, _, _) in pieces_by_name.items():
            outputs[f"value_{name}"] = np.asarray(value_j(z, params, state))
            outputs[f"gradient_{name}"] = np.asarray([
                float(d_state(z, state, column)) + float(d_param(z, state, t))
                for column, t in zip(columns, tangents)])
        return outputs

    abort_report, reference = {}, None
    for label, abort in (("pr320", None), ("window_1000_drop_25", (1000, 0.25)),
                         ("window_500_drop_25", (500, 0.25))):
        state, steps = refine_with_policy(abort)
        outputs = anchor_outputs(state)
        reference = outputs if reference is None else reference
        total = sum(s["iterations"] for s in steps)
        abort_report[label] = {
            "steps": steps, "gcrot_iterations": total,
            "iterations_saved_vs_pr320": abort_report["pr320"]["gcrot_iterations"] - total if abort else 0,
            "bit_identical_to_pr320": {key: bool(np.array_equal(outputs[key], reference[key])) for key in outputs}}
    abort_report["pr320"]["state_bit_identical_to_main_refined_state"] = bool(
        np.array_equal(np.asarray(flat(refined)), reference["state"]))

    # ---- Newton arms ----------------------------------------------------------------
    newton_config = Prec2DConfig(threshold=np.inf, gmres_restart=50,
                                 gmres_max_restarts=20, gmres_rtol=NEWTON_RTOL)

    @jax.jit
    def direction_1d(z, frozen):
        force = imp.residual_fn(cfg, frozen, mask)
        rhs = jax.tree.map(jnp.negative, force(z, params))
        step, solution = newton_direction(lambda t: force(t, params), z, rhs, newton_config)
        return step, solution.iterations, solution.residual_norm / imp._tree_norm(rhs)

    @jax.jit
    def direction_block(z, frozen, factors, V, U):
        raw = imp.residual_fn(cfg, frozen, mask, formulation="raw")
        value, linear = jax.linearize(lambda t: raw(t, params), z)
        deflate_left = lambda w: w - U @ (U.T @ w)  # noqa: E731
        deflate_right = lambda w: w - V @ (V.T @ w)  # noqa: E731
        rhs = deflate_left(flat(value))
        solution = solvax.gmres(
            lambda y: deflate_left(flat(linear(unravel(deflate_right(y))))), -rhs,
            precond=lambda w: deflate_right(flat(inverse(factors, unravel(deflate_left(w))))),
            restart=50, rtol=NEWTON_RTOL, max_restarts=20)
        return (unravel(deflate_right(solution.x)), solution.iterations,
                solution.residual_norm / jnp.linalg.norm(rhs))

    @jax.jit
    def projected_certificate(z, frozen, Up):
        force, _, bad = evaluate(z, frozen)
        f = flat(force)
        return jnp.linalg.norm(f - Up @ (Up.T @ f)), jnp.linalg.norm(f), bad

    def newton(frozen, direction, measure, steps_max):
        z = P(frozen)
        current, full, _ = (float(a) if i < 2 else bool(a) for i, a in enumerate(measure(z)))
        initial = current
        steps, jvps_used, evaluations, bad = [], 0, 1, False
        for _ in range(steps_max):
            step, iterations, relative = direction(z)
            jvps_used += int(iterations)
            evaluations += 1
            alpha = 1.0
            while True:
                trial = jax.tree.map(lambda x, d: x + alpha * d, z, step)
                value, full, sign_changed = measure(trial)
                value, full, sign_changed = float(value), float(full), bool(sign_changed)
                evaluations += 1
                if (np.isfinite(value) and not sign_changed
                        and value < (1.0 - 1e-4 * alpha) * current) or alpha < 1e-3:
                    break
                alpha *= 0.5
            bad |= sign_changed
            steps.append({"matvecs": int(iterations), "linear_relative_residual": float(relative),
                          "alpha": alpha, "measured": value, "certificate": full,
                          "step_norm": norm(step)})
            z, current = trial, value
            if current <= cfg.refine_tol:
                break
        fsq = [initial ** 2] + [s["measured"] ** 2 for s in steps]
        slow = sum(fsq[k + 1] > fsq[k] / 10.0 for k in range(min(3, len(steps))))
        return z, {
            "steps": steps, "jvps": jvps_used, "force_evaluations": evaluations,
            "certified": current <= cfg.refine_tol, "final_measured": current,
            "final_certificate": steps[-1]["certificate"] if steps else full,
            "jacobian_sign_change": bad,
            "distance_to_refined": norm(jax.tree.map(jnp.subtract, z, P(refined))),
            "kills": {"matvecs_over_50_per_step": any(s["matvecs"] > 50 for s in steps),
                      "fsq_fell_under_10x_in_2_of_first_3": slow >= 2,
                      "jacobian_reset": bad},
        }

    # ---- (d) Anderson on the damped descent map ----------------------------------------
    window, max_evaluations, alpha_d = 8, 400, float(deck.time_step) ** 2

    def anderson(frozen):
        flat0 = flat(P(frozen))
        n = flat0.size

        @jax.jit
        def run(flat0, frozen):
            def mapped(v):
                force, fsq, bad = evaluate(unravel(v), frozen)
                return flat(force), fsq, bad

            force0, fsq0, bad0 = mapped(flat0)
            history_x = jnp.zeros((window, n)).at[0].set(flat0)
            history_r = jnp.zeros((window, n)).at[0].set(alpha_d * force0)

            def cond(carry):
                count, _, _, fsq, _, bad = carry
                return (count < max_evaluations) & (fsq > cfg.ftol) & ~bad

            def body(carry):
                count, history_x, history_r, _, _, _ = carry
                valid = (jnp.arange(window) < jnp.minimum(count, window))[:, None]
                new = solvax.anderson_mixing(jnp.where(valid, history_x, history_x[0]),
                                             jnp.where(valid, history_r, history_r[0]))
                force, fsq, bad = mapped(new)
                slot = count % window
                return (count + 1, history_x.at[slot].set(new),
                        history_r.at[slot].set(alpha_d * force), fsq, jnp.linalg.norm(force), bad)

            return jax.lax.while_loop(cond, body, (
                jnp.asarray(1), history_x, history_r, fsq0, jnp.linalg.norm(force0), bad0))

        count, _, _, fsq, force_norm, bad = run(flat0, frozen)
        return {"force_evaluations": int(count), "fsq": float(fsq),
                "final_certificate": float(force_norm), "jacobian_sign_change": bool(bad),
                "reached_deck_fsq": float(fsq) <= cfg.ftol,
                "certified": float(force_norm) <= cfg.refine_tol}

    empty = jnp.zeros((z_deck.R_cos.size * len(imp._STATE_FIELDS), 0))
    starts = {}
    for label, run in [*((str(k), v) for k, v in runs.items()), ("deck", deck)]:
        frozen = run.state
        z0 = P(frozen)
        entry = {"descent_iterations_at_stop": int(run.iterations),
                 "certificate": certificate(z0, frozen)[0], "fsq": certificate(z0, frozen)[1]}
        if label != "deck":
            descent_left = int(deck.iterations - run.iterations)
            entry["a_descent_to_deck"] = {
                "descent_iterations": descent_left,
                "work_with_refinement": descent_left + refinement["work"],
                "final_certificate": refinement["certificate_after"],
                "certified": refinement["certificate_after"] <= cfg.refine_tol}
            entry["d_anderson"] = anderson(frozen)
            entry["d_anderson"]["evaluation_ratio_vs_descent"] = (
                descent_left / entry["d_anderson"]["force_evaluations"])
        measure_plain = lambda z, f=frozen: (lambda r: (imp._tree_norm(r[0]), imp._tree_norm(r[0]), r[2]))(evaluate(z, f))  # noqa: E731
        _, entry["b_newton_1d"] = newton(
            frozen, lambda z, f=frozen: direction_1d(z, f), measure_plain, newton_steps["b"])
        factors = block_system(frozen, z0)
        _, entry["c_newton_block"] = newton(
            frozen, lambda z, f=frozen, g=factors: direction_block(z, f, g, empty, empty),
            measure_plain, newton_steps["c"])
        sigma, V, JV, JpV = spectrum(frozen, z0, factors, max(deflation_sizes))
        entry["sigma_smallest"] = [float(s) for s in sigma]
        entry["e_newton_deflated"] = {}
        for k in deflation_sizes:
            Vk, Uk, Upk = jnp.asarray(V[:, :k]), orthonormal(JV[:, :k]), orthonormal(JpV[:, :k])
            z_final, result = newton(
                frozen, lambda z, f=frozen, g=factors, a=Vk, b=Uk: direction_block(z, f, g, a, b),
                lambda z, f=frozen, u=Upk: projected_certificate(z, f, u), newton_steps["e"])
            displacement = np.asarray(flat(jax.tree.map(jnp.subtract, z_final, z0)))
            result["displacement_norm"] = float(np.linalg.norm(displacement))
            result["displacement_on_near_null"] = float(np.linalg.norm(np.asarray(Vk).T @ displacement))
            if label == "deck":
                factors_final = block_system(frozen, z_final)
                result["objectives_at_final"] = {}
                for name, pieces in pieces_by_name.items():
                    sensitivity = gradient_sensitivity(pieces, frozen, z_final, factors_final, V_deck)
                    reference = np.asarray(objective_report[name]["sensitivity_at_deck"]["gradient"])
                    result["objectives_at_final"][name] = {
                        "value_change": float(pieces[0](z_final, params, frozen))
                        - objective_report[name]["value"],
                        "gradient_relative_change": float(
                            np.linalg.norm(np.asarray(sensitivity["gradient"]) - reference)
                            / max(np.linalg.norm(reference), 1e-300))}
            result["work"] = (result["force_evaluations"] + units["c_jvp"] * result["jvps"]
                              + units["f_factorization"])
            entry["e_newton_deflated"][str(k)] = result
        for arm, factorizations in (("b_newton_1d", 0), ("c_newton_block", 1)):
            result = entry[arm]
            result["work"] = (result["force_evaluations"] + units["c_jvp"] * result["jvps"]
                              + units["f_factorization"] * factorizations)
        starts[label] = entry
        print(f"start {label} done", file=sys.stderr, flush=True)

    report = {
        "case": args.case, "smoke": smoke, "input_sha256": file_sha256(path),
        "resolution": {"mpol": cfg.resolution.mpol, "ntor": cfg.resolution.ntor,
                       "ns": cfg.resolution.ns, "nfp": cfg.resolution.nfp},
        "deck_ftol": cfg.ftol, "refine_tol": cfg.refine_tol, "descent_iterations_deck": int(deck.iterations),
        "unit_costs": units, "refinement_from_deck": refinement, "refinement_abort": abort_report,
        "raw_jacobian_spectrum_at_deck": spectrum_report, "objectives": objective_report,
        "starts": starts,
        "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "compilation_cache": os.environ.get("VMEX_COMPILATION_CACHE"),
        "load_average": {"start": load_start, "end": os.getloadavg()},
        "platform": platform.platform(),
        "versions": {"python": platform.python_version(), "jax": jax.__version__,
                     "numpy": np.__version__, "vmex": vmex.__version__,
                     "solvax": getattr(solvax, "__version__", None)},
        **git_state(REPO), "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=float)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

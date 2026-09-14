"""Startup-latency contracts of the staged fixed-point refinement.

Pins the two perf behaviors behind the QA_optimization startup fixes (the
``tests/test_runtime_recompile_keys.py`` idiom — lower-only and counter-based,
no repeated full solves):

- ``_refine_step_core`` is one reusable per-config executable: every per-trial
  quantity (iterate, residual, parameters, frozen anchor, dof mask) is a
  program ARGUMENT, never a baked closure constant, so consecutive optimizer
  trial boundaries share one compiled program.  The previous host-eager step
  re-linearized ``F`` per call and handed ``solvax.gcrot`` a fresh closure, so
  the ``lax.while_loop`` it staged missed the compile cache on every trial —
  one measured ``jit(while)`` recompile per optimizer evaluation;
- the problem-factory seed preflight (``refine=False``) never pays for the
  Newton anchor: refinement runs on the first derivative evaluation instead,
  which memo-hits the preflight's solve, so the work moves behind the
  ``compile_value_and_gradient`` heartbeat without being duplicated.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmex.core import implicit as im
from vmex.core.errors import VmecError
from vmex.core.input import VmecInput

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _small_solovev_setup():
    """Smallest meaningful analytic equilibrium (one fast forward solve)."""
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    inp = inp.change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([5]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]),
    )
    cfg = im.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    return inp, cfg, im.params_from_input(inp)


def test_refine_step_lowering_is_shared_across_trial_points() -> None:
    """Two trial iterates lower to one identical refinement program.

    The lowered text is what the compile cache keys on (one cfg, fixed
    avals), so byte-identical lowerings ARE executable reuse: the second and
    every later optimizer trial run the already-compiled refinement step
    instead of restaging a fresh ``while_loop`` closure.
    """
    _, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    P = im._dof_projector(cfg, mask)
    F = im.residual_fn(cfg, state, mask)
    z0 = P(state)
    fz0 = F(z0, p0)

    # A nearby trial: different values, same structure — exactly what a
    # line-search hands the refinement on consecutive evaluations.
    z1 = jax.tree.map(lambda a: a * (1.0 + 1.0e-6), z0)
    p1 = dataclasses.replace(p0, rbc=p0.rbc * (1.0 + 1.0e-6))
    fz1 = F(z1, p1)

    text_a = im._refine_step_core.lower(z0, fz0, p0, state, mask, cfg=cfg).as_text()
    text_b = im._refine_step_core.lower(z1, fz1, p1, state, mask, cfg=cfg).as_text()
    assert text_a == text_b
    # ... while the trial values genuinely differ.
    assert not np.array_equal(np.asarray(z0.R_cos), np.asarray(z1.R_cos))


def test_preflight_skips_refinement_and_first_derivative_pays_it_once() -> None:
    """``refine=False`` returns the raw solver state without the anchor.

    The subsequent default (``refine=True``) call at the same parameters
    memo-hits the solve and runs the refinement exactly once, so deferring
    the anchor out of problem construction conserves total work.
    """
    _, cfg, p0 = _small_solovev_setup()
    params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), p0)

    calls = {"refine": 0, "solve": 0}
    original_refine = im._refine_fixed_point
    original_solve = im._host_solve

    def counting_refine(*args, **kwargs):
        calls["refine"] += 1
        return original_refine(*args, **kwargs)

    def counting_solve(*args, **kwargs):
        calls["solve"] += 1
        return original_solve(*args, **kwargs)

    im._refine_fixed_point = counting_refine
    im._host_solve = counting_solve
    try:
        raw_state, _ = im._host_solve_and_mask(cfg, params_np, refine=False)
        assert calls == {"refine": 0, "solve": 1}
        hit = im._LAST_SOLVE.get(cfg)
        assert hit is not None
        for raw_leaf, solver_leaf in zip(
            jax.tree.leaves(raw_state), jax.tree.leaves(hit[1].state)
        ):
            np.testing.assert_array_equal(
                np.asarray(raw_leaf), np.asarray(solver_leaf))

        refined_state, _ = im._host_solve_and_mask(cfg, params_np)
        # Memo-hit solve (the counted host call returns the stored result
        # without iterating) and exactly one refinement.
        assert calls == {"refine": 1, "solve": 2}
        stats = im._SOLVE_STATS.get(cfg)
        assert stats is not None and stats["solves"] == 1
        assert stats["refinements"] >= 1 and stats["refinement_seconds"] > 0.0
        assert stats["refinement_krylov_iterations"] >= stats["refinement_steps"] >= 1
    finally:
        im._refine_fixed_point = original_refine
        im._host_solve = original_solve

    # The refined anchor is the memoized one later derivative lanes read.
    memo = im._LAST_REFINED.get(cfg)
    assert memo is not None
    for refined_leaf, memo_leaf in zip(
        jax.tree.leaves(refined_state), jax.tree.leaves(memo[1])
    ):
        np.testing.assert_array_equal(
            np.asarray(refined_leaf), np.asarray(memo_leaf))


def test_adjoint_counters_cover_eager_and_staged_gradients() -> None:
    """Host-eager adjoints are counted; a compiled adjoint makes them unknown."""
    _, cfg, p0 = _small_solovev_setup()

    def volume(params):
        return im.plasma_volume(im.solve_implicit(params, cfg),
                                im.runtime_from_params(params, cfg))

    im._SOLVE_STATS.pop(cfg, None)
    try:
        jax.grad(volume)(p0)
        stats = im._SOLVE_STATS[cfg]
        assert stats["adjoints"] == 1 and stats["adjoint_seconds"] > 0.0
        assert stats["adjoint_krylov_iterations"] == 0  # the exact block adjoint
        previous = bool(jax.config.jax_disable_jit)
        jax.config.update("jax_disable_jit", False)  # the suite default is eager
        try:
            jax.jit(jax.grad(volume))(p0)
        finally:
            jax.config.update("jax_disable_jit", previous)
        assert stats["adjoints"] is None
        assert stats["adjoint_krylov_iterations"] is None
        assert stats["adjoint_seconds"] is None
    finally:
        im._SOLVE_STATS.pop(cfg, None)


def test_rejected_trial_charges_solve_time_but_counts_no_solve(monkeypatch) -> None:
    """A failed trial solve adds host seconds, not a solve or its iterations."""
    _, cfg, p0 = _small_solovev_setup()
    trial = dataclasses.replace(p0, rbc=p0.rbc * (1.0 + 1.0e-3))
    params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), trial)

    def fail(*_args, **_kwargs):
        raise VmecError(message="synthetic failed trial")

    monkeypatch.setattr(im, "solve", fail)
    monkeypatch.setattr(im, "solve_multigrid", fail)
    im._SOLVE_STATS.pop(cfg, None)
    try:
        status = im._host_solve_and_mask_status(cfg, params_np)[2]
        stats = im._SOLVE_STATS[cfg]
        assert int(status) == 1
        assert (stats["solves"], stats["iterations"]) == (0, 0)
        assert stats["solve_seconds"] > 0.0
    finally:
        im._SOLVE_STATS.pop(cfg, None)
        im._LAST_STATUS_ERROR.pop(cfg, None)


def test_step_without_krylov_progress_ends_refinement(monkeypatch) -> None:
    """A step whose solve kept fewer than three digits and did not lower ``|F|`` stops.

    Two trials run compiled on the shared small config, the second from the
    first one's returned state, as the next optimizer evaluation does. Every
    step runs the real staged executable; its host wrapper only reports the
    step as non-improving (the iterate moves, ``|F|`` does not fall). With
    ``_REFINE_MIN_PROGRESS`` at zero every solve counts as fewer than three
    digits, so the first trial stops after one step; at its default the small
    deck's solves converge and the second trial keeps the whole non-monotone
    budget. The branch is decided on the host, so the second trial compiles
    nothing, and neither trial moves the host state.
    """
    _, cfg, p0 = _small_solovev_setup()
    params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), p0)
    state, mask = jax.tree.map(
        jnp.asarray, im._host_solve_and_mask(cfg, params_np, refine=False))
    real_step, default = im._refine_step, im._REFINE_MIN_PROGRESS

    def without_progress(config, params, frozen, dof_mask, z, fz):
        z_new, _, _, linear = real_step(config, params, frozen, dof_mask, z, fz)
        return z_new, fz, im._tree_norm(fz), linear

    monkeypatch.setattr(im, "_refine_step", without_progress)
    lanes = (im._refine_step_core, im._preconditioned_residual_lane)
    steps = lambda: im._SOLVE_STATS.get(cfg, {}).get("refinement_steps", 0)  # noqa: E731
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)  # the suite default is eager
    try:
        start = steps()
        monkeypatch.setattr(im, "_REFINE_MIN_PROGRESS", 0.0)
        refined = im._refined_state(cfg, p0, state, mask)
        assert steps() - start == 1
        compiled = [lane._cache_size() for lane in lanes]
        assert min(compiled) >= 1  # both staged programs ran
        monkeypatch.setattr(im, "_REFINE_MIN_PROGRESS", default)
        again = im._refined_state(cfg, p0, refined, mask)
        assert steps() - start == 1 + im._REFINE_MAX_STEPS
        assert [lane._cache_size() for lane in lanes] == compiled
    finally:
        jax.config.update("jax_disable_jit", previous)
    for result in (refined, again):
        for result_leaf, state_leaf in zip(jax.tree.leaves(result), jax.tree.leaves(state)):
            np.testing.assert_array_equal(np.asarray(result_leaf), np.asarray(state_leaf))

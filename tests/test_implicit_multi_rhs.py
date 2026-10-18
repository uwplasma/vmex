"""Parity tests for the batched reverse-AD helpers (PR #24).

``solve_implicit_with_aux`` and ``implicit_state_pullback_multi_rhs`` add a
vectorized state-cotangent pullback for callers with several objectives sharing
one fixed point (it reuses the residual/projector/VJP setup once and batches the
adjoint solves).  These check it reproduces the scalar ``solve_implicit`` VJP
exactly, so the batched path is a pure efficiency win with identical gradients.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

from vmex.core import implicit as im
from vmex.core.errors import AdjointSolveError
from vmex.core.input import VmecInput

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _solovev_setup():
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    cfg = im.make_config(inp, ftol=1e-13, max_iterations=2000)
    p0 = im.params_from_input(inp)
    return inp, cfg, p0


def _small_solovev_setup():
    """Same analytic equilibrium on the smallest meaningful AD grid.

    Block assembly scales cubically with the modes per radial plane.  The
    forward/transpose/finite-difference identities tested below do not need
    the production-resolution Solovev deck, which remains covered by the
    full gradient and equilibrium parity lanes.
    """
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    inp = inp.change_resolution(
        mpol=3, ntor=0, ntheta=12, nzeta=4,
    )
    inp = dataclasses.replace(
        inp,
        ns_array=np.asarray([5]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([1000]),
    )
    cfg = im.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    return inp, cfg, im.params_from_input(inp)


def _tree_dot(left, right):
    return sum(
        jnp.vdot(a, b).real
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    )


def _relative_difference(got, reference):
    difference = jax.tree.map(jnp.subtract, got, reference)
    return float(jnp.sqrt(_tree_dot(difference, difference) / _tree_dot(reference, reference)))


def _dense_raw_pullback(params, cfg, state, mask, gbar):
    """Parameter cotangent from a dense solve of the raw adjoint on range(P).

    Independent of the block factors: the raw residual's Jacobian is built
    densely, restricted to the projector's range through an eigenbasis (P is
    a symmetric idempotent), and solved directly.  Then ``-mu^T dF_raw/dp``
    plus the direct boundary term, as in the scalar rule.
    """
    frozen = jax.lax.stop_gradient(state)
    project = im._dof_projector(cfg, mask)
    edge = im._edge_mask(cfg)
    raw = im.residual_fn(cfg, frozen, mask, formulation="raw")
    z_star = project(state)
    flat_z, unravel = ravel_pytree(z_star)
    projector = np.asarray(jax.vmap(
        lambda e: ravel_pytree(project(unravel(e)))[0])(jnp.eye(flat_z.size)))
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (projector + projector.T))
    basis = eigenvectors[:, eigenvalues > 0.5]
    jacobian = np.asarray(jax.jacfwd(
        lambda v: ravel_pytree(raw(unravel(v), params))[0])(flat_z))
    rhs = np.asarray(ravel_pytree(project(gbar))[0])
    reduced = basis.T @ jacobian @ basis
    mu = unravel(jnp.asarray(basis @ np.linalg.solve(reduced.T, basis.T @ rhs)))
    implicit = jax.vjp(lambda prm: raw(z_star, prm), params)[1](
        jax.tree.map(jnp.negative, mu))[0]
    direct = jax.vjp(
        lambda prm: im._assemble(
            z_star, im.runtime_from_params(prm, cfg), frozen, project, edge),
        params)[1](gbar)[0]
    return jax.tree.map(jnp.add, implicit, direct)


def test_solve_implicit_with_aux_matches_solve_implicit():
    """The aux helper returns the same converged state as solve_implicit."""
    _, cfg, p0 = _solovev_setup()
    state_aux, mask = im.solve_implicit_with_aux(p0, cfg)
    state_ref = im.solve_implicit(p0, cfg)
    for a, b in zip(jax.tree.leaves(state_aux), jax.tree.leaves(state_ref)):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)
    # the mask is a 0/1 SpectralState of the same structure as the state
    assert jax.tree.structure(mask) == jax.tree.structure(state_ref)
    with pytest.raises(ValueError, match="solver"):
        im.implicit_state_pullback_multi_rhs(
            p0, cfg, state_aux, mask, state_aux, solver="dense"
        )


@pytest.mark.full
def test_multi_rhs_pullback_matches_scalar_vjp():
    """Batched pullbacks match the scalar solve_implicit VJP, formulation by formulation.

    The scalar rule and ``solver="block"`` both solve the raw-force adjoint
    with one block factorization, so they agree to round-off: this checks the
    batching.  The independent raw reference is the dense solve in
    ``test_block_response_forward_transpose_and_fd``.  The default
    ``solver="gcrot"`` solves the preconditioned adjoint, which differs from
    the raw one by the formulation difference where the anchor is not an exact
    root (at most 1.1e-3 on record, https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/adjoint_formulation_20260914.json).
    """
    _, cfg, p0 = _solovev_setup()
    x_star, mask = im.solve_implicit_with_aux(p0, cfg)

    # three distinct state cotangents (same pytree structure as x_star)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    gbars = [jax.tree.map(lambda a, k=k: jax.random.normal(k, a.shape, a.dtype), x_star)
             for k in keys]
    gbar_batch = jax.tree.map(lambda *a: jnp.stack(a), *gbars)

    g_block = im.implicit_state_pullback_multi_rhs(
        p0, cfg, x_star, mask, gbar_batch, solver="block")
    g_gcrot = im.implicit_state_pullback_multi_rhs(p0, cfg, x_star, mask, gbar_batch)

    # scalar reference: the actual solve_implicit custom-VJP, applied per cotangent
    _, pullback = jax.vjp(lambda p: im.solve_implicit(p, cfg), p0)
    for i, gbar in enumerate(gbars):
        g_scalar = pullback(gbar)[0]
        g_block_i = jax.tree.map(lambda a: a[i], g_block)
        for a, b in zip(jax.tree.leaves(g_scalar), jax.tree.leaves(g_block_i)):
            a = np.asarray(a); b = np.asarray(b)
            assert a.shape == b.shape
            if a.size == 0:  # parameter families this deck does not use
                continue
            scale = np.max(np.abs(a)) + 1e-30
            assert np.max(np.abs(a - b)) <= 1e-8 * scale, "block multi-rhs != scalar VJP"
        g_gcrot_i = jax.tree.map(lambda a: a[i], g_gcrot)
        assert _relative_difference(g_gcrot_i, g_scalar) < 1e-3

def test_block_response_forward_transpose_and_fd():
    """One factorization serves tangent and transpose responses accurately."""
    inp, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    zero = jax.tree.map(jnp.zeros_like, p0)
    tangents = (
        dataclasses.replace(
            zero, rbc=zero.rbc.at[inp.ntor, 1].set(1.0)
        ),
        dataclasses.replace(zero, pres_scale=jnp.ones_like(zero.pres_scale)),
    )
    tangent_batch = jax.tree.map(lambda *x: jnp.stack(x), *tangents)
    state_tangent, report = im.implicit_state_tangent_multi_rhs(
        p0, cfg, state, mask, tangent_batch,
        probe_chunk_size=4, response_chunk_size=2,
    )
    assert np.all(np.asarray(report.converged))
    assert np.all(
        np.asarray(report.residual_norm) <= np.asarray(report.tolerance)
    )

    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    cotangents = jax.tree.map(
        lambda value: jnp.stack([
            jax.random.normal(key, value.shape, value.dtype) for key in keys
        ]),
        state,
    )
    pullback = im.implicit_state_pullback_multi_rhs(
        p0, cfg, state, mask, cotangents,
        solver="block", probe_chunk_size=4, response_chunk_size=2,
    )
    lhs = _tree_dot(state_tangent, cotangents)
    rhs = _tree_dot(tangent_batch, pullback)
    np.testing.assert_allclose(lhs, rhs, rtol=2e-8, atol=2e-10)

    # An independent reference in the block pullback's own raw formulation:
    # a dense solve on range(P).  The scalar rule shares the block factors.
    for i in range(2):
        reference = _dense_raw_pullback(
            p0, cfg, state, mask,
            jax.tree.map(lambda value: value[i], cotangents))
        for got, expected in zip(
            jax.tree.leaves(jax.tree.map(lambda value: value[i], pullback)),
            jax.tree.leaves(reference),
        ):
            np.testing.assert_allclose(got, expected, rtol=2e-8, atol=2e-10)

    # The default GCROT pullback solves the preconditioned adjoint.  Where the
    # anchor is not an exact root the formulations differ by O(|F|); the
    # largest difference on record is 1.1e-3, on the QI seed state
    # (https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/adjoint_formulation_20260914.json).
    preconditioned = im.implicit_state_pullback_multi_rhs(
        p0, cfg, state, mask, cotangents
    )
    assert _relative_difference(pullback, preconditioned) < 1e-3

    for i, (tangent, step) in enumerate(zip(tangents, (3e-5, 1e-4))):
        directional = jax.jvp(
            lambda x, p: im.aspect_ratio(x, im.runtime_from_params(p, cfg)),
            (state, p0),
            (jax.tree.map(lambda value: value[i], state_tangent), tangent),
        )[1]
        finite_difference, info = im.frozen_path_directional_fd(
            p0, cfg, im.aspect_ratio, tangent, h=step
        )
        assert max(info["newton_res"]) < 1e-8
        np.testing.assert_allclose(
            directional, finite_difference, rtol=2e-5, atol=2e-8
        )
def test_block_response_refines_columns_the_corrector_leaves_uncertified(monkeypatch):
    """A column the corrector leaves uncertified is refined through the block factors.

    Before refinement, one such column sent the automatic optimizer Jacobian to the
    reverse lane, which maps a pullback over every residual row (655 s instead of
    27 s for 17,716 rows and 48 columns). A stalled corrector is simulated here: it
    returns a perturbed warm start with a non-converged report.
    """
    inp, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    zero = jax.tree.map(jnp.zeros_like, p0)
    tangent_batch = jax.tree.map(lambda *x: jnp.stack(x), dataclasses.replace(
        zero, rbc=zero.rbc.at[inp.ntor, 1].set(1.0)), dataclasses.replace(
        zero, pres_scale=jnp.ones_like(zero.pres_scale)))

    def respond():
        return jax.jit(lambda: im._implicit_evolved_tangent_multi_rhs(
            p0, cfg, state, mask, tangent_batch,
            active_fields=im._active_state_fields(cfg), probe_chunk_size=4,
            response_chunk_size=2, certify_rtol=1e-8, certify_maxiter=1))()

    reference, report = respond()
    assert np.all(np.asarray(report.converged))

    def stalled(A, b, cfg, *, x0=None, max_restarts=None, rtol=None):
        return jax.tree.map(lambda value: 1.001 * value, x0), im.LinearResponseReport(
            residual_norm=jnp.asarray(1.0e30), tolerance=jnp.asarray(0.0),
            iterations=jnp.asarray(1, jnp.int32), converged=jnp.asarray(False))

    monkeypatch.setattr(im, "_adjoint_solve", stalled)
    passes = im._BLOCK_REFINEMENT_PASSES
    monkeypatch.setattr(im, "_BLOCK_REFINEMENT_PASSES", 0)
    _, unrefined = respond()
    assert not np.any(np.asarray(unrefined.converged))
    monkeypatch.setattr(im, "_BLOCK_REFINEMENT_PASSES", passes)
    refined, report = respond()
    assert np.all(np.asarray(report.converged))
    assert _relative_difference(refined, reference) < 1e-8


def test_raw_block_probe_chunking_preserves_exact_factors():
    """Bounded VJP batches assemble the same local block Jacobian."""

    _, cfg, params = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(params, cfg)
    active = im._active_state_fields(cfg)
    scalar = im._raw_block_system(
        params, cfg, state, mask, active, probe_chunk_size=1
    )
    chunked = im._raw_block_system(
        params, cfg, state, mask, active, probe_chunk_size=4
    )
    for name in ("lower", "diagonal", "upper", "row_scale", "column_scale"):
        np.testing.assert_allclose(
            getattr(chunked, name), getattr(scalar, name), rtol=2e-13, atol=2e-13
        )
    with pytest.raises(ValueError, match="probe_chunk_size"):
        im._raw_block_system(
            params, cfg, state, mask, active, probe_chunk_size=0
        )


@pytest.mark.parametrize("deck,ns,ntor", [("solovev", 5, 0)] + [
    pytest.param(*case, marks=pytest.mark.full)  # ~40 s each
    for case in [("solovev", 3, 0),  # every surface its own color
                 ("li383_low_res", 6, 1), ("up_down_asymmetric_tokamak", 4, 0)]
])
def test_colored_bands_match_the_per_row_build(monkeypatch, deck, ns, ntor):
    """Three-colored forward probes give the per-row reverse blocks to round-off."""

    inp = VmecInput.from_file(str(DATA / f"input.{deck}"))
    inp = dataclasses.replace(
        inp.change_resolution(mpol=3, ntor=ntor, ntheta=12, nzeta=4 * ntor + 4),
        ns_array=np.asarray([ns]), ftol_array=np.asarray([1.0e-6]),
        niter_array=np.asarray([500]),
    )
    cfg = im.make_config(inp, ftol=1.0e-6, max_iterations=500)  # any state will do
    params = im.params_from_input(inp)
    state, mask = im.solve_implicit_with_aux(params, cfg)
    active = im._active_state_fields(cfg)
    per_row = im._raw_block_system(
        params, cfg, state, mask, active, probe_chunk_size=4,
        runtime=im.runtime_from_params(params, cfg),
    )
    monkeypatch.setattr(im, "_COLORED_PROBE_BUDGET", 1)  # several probe batches
    colored = im._raw_block_system(params, cfg, state, mask, active, probe_chunk_size=4)
    for name in ("lower", "diagonal", "upper", "row_scale", "column_scale"):
        scale = np.abs(np.asarray(getattr(per_row, name))).max()
        np.testing.assert_allclose(
            getattr(colored, name), getattr(per_row, name), rtol=0, atol=1e-12 * scale
        )


@pytest.mark.full
def test_block_pullback_rejects_unconverged_response():
    """The opt-in transpose path cannot return an uncertified gradient."""
    _, cfg, p0 = _solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    impossible = dataclasses.replace(
        cfg, adjoint_tol=1e-30, adjoint_maxiter=1, adjoint_restart=2
    )
    cotangent = jax.tree.map(lambda value: value[None], state)
    with pytest.raises(AdjointSolveError, match="block-tridiagonal adjoint"):
        im.implicit_state_pullback_multi_rhs(
            p0, impossible, state, mask, cotangent,
            solver="block", probe_chunk_size=4,
        )


@pytest.mark.full
def test_block_response_lasym_parity():
    """All six LASYM state families share the same forward/transpose engine."""
    inp0 = VmecInput.from_file(DATA / "input.basic_non_stellsym_simsopt")
    inp = dataclasses.replace(
        inp0,
        ns_array=np.asarray([11]),
        ftol_array=np.asarray([1e-12]),
        niter_array=np.asarray([4000]),
    )
    cfg = im.make_config(inp, ftol=1e-12, max_iterations=4000)
    params = im.params_from_input(inp)
    state, mask = im.solve_implicit_with_aux(params, cfg)
    assert im._active_state_fields(cfg) == im._STATE_FIELDS

    zero = jax.tree.map(jnp.zeros_like, params)
    tangent = dataclasses.replace(
        zero, rbs=zero.rbs.at[inp.ntor + 1, 1].set(1.0)
    )
    tangent_batch = jax.tree.map(lambda value: value[None], tangent)
    state_tangent, report = im.implicit_state_tangent_multi_rhs(
        params, cfg, state, mask, tangent_batch,
        probe_chunk_size=4, response_chunk_size=1,
    )
    assert bool(np.asarray(report.converged[0]))

    cotangent = jax.tree.map(
        lambda value: jnp.linspace(
            -0.2, 0.3, value.size, dtype=value.dtype
        ).reshape((1,) + value.shape),
        state,
    )
    block = im.implicit_state_pullback_multi_rhs(
        params, cfg, state, mask, cotangent, solver="block",
        probe_chunk_size=4, response_chunk_size=1,
    )
    reference = im.implicit_state_pullback_multi_rhs(
        params, cfg, state, mask, cotangent
    )
    np.testing.assert_allclose(
        _tree_dot(state_tangent, cotangent),
        _tree_dot(tangent_batch, block),
        rtol=2e-6, atol=2e-9,
    )
    for got, expected in zip(
        jax.tree.leaves(block), jax.tree.leaves(reference)
    ):
        np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-9)


def test_jacobian_certification_tolerance_is_separate_from_the_gradient_one():
    """Jacobian columns certify against their own, looser tolerance.

    The two tolerances feed different consumers: a scalar gradient goes to a
    quasi-Newton method that accumulates curvature from it, while a
    least-squares Jacobian only has to point a trust-region step.  Certifying
    columns to the gradient tolerance made the certifier, not the block
    factorization, the cost of an asymmetric Jacobian -- 542 iterations
    against 1e-6 where 1e-4 needs none, for a 3.2e-5 relative change.
    """
    inp = VmecInput.from_file(str(DATA / "input.solovev"))
    cfg = im.make_config(inp, ftol=1.0e-10, max_iterations=1000,
                         adjoint_tol=1.0e-11, jacobian_adjoint_tol=1.0e-4)
    assert cfg.adjoint_tol == 1.0e-11
    assert cfg.jacobian_adjoint_tol == 1.0e-4
    default = im.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    assert default.jacobian_adjoint_tol > default.adjoint_tol
    # The certifier corrects a direct block solve, so its budget is bounded
    # well below the reverse-adjoint one: past the point where the
    # factorization stops preconditioning, more cycles only buy wall time.
    assert default.jacobian_adjoint_maxiter < default.adjoint_maxiter


def test_raw_block_apply_requires_stored_factors():
    """Applying a stored block inverse without factors is a caller error.

    ``_raw_block_system(..., factor=False)`` builds the exact operators but
    keeps no factorization, so the precondition has to be stated rather than
    surfacing later as an attribute error deep inside the solve.
    """
    identity = lambda value: value  # noqa: E731
    system = im._RawBlockSystem(
        factors=None, pack=identity, unpack=identity, project=identity,
        operator=identity, operator_t=identity, band_operator=identity,
        band_operator_t=identity, lower=jnp.zeros((1, 1, 1)),
        diagonal=jnp.zeros((1, 1, 1)), upper=jnp.zeros((1, 1, 1)),
        row_scale=jnp.ones((1, 1)), column_scale=jnp.ones((1, 1)))
    with pytest.raises(ValueError, match="raw block factors"):
        im._raw_block_apply(system, jnp.zeros((1, 1)))


@pytest.mark.full
def test_auto_chunks_match_the_default_pullback():
    """Opt-in "auto" chunk sizes change scheduling only, never values.

    A full extra pullback solve: parity lane c3d sits within ~90 s of its
    45-minute budget, so this equivalence runs in the full tier while the
    cheap resolver/memory-model pins below stay on the fast path.
    """
    _, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    cotangents = jax.tree.map(
        lambda value: jnp.stack([
            jax.random.normal(key, value.shape, value.dtype) for key in keys
        ]),
        state,
    )
    reference = im.implicit_state_pullback_multi_rhs(
        p0, cfg, state, mask, cotangents
    )
    audited = im.implicit_state_pullback_multi_rhs(
        p0, cfg, state, mask, cotangents,
        probe_chunk_size="auto", response_chunk_size="auto",
    )
    for got, expected in zip(
        jax.tree.leaves(audited), jax.tree.leaves(reference)
    ):
        np.testing.assert_allclose(got, expected, rtol=2e-8, atol=2e-10)


def test_measured_chunk_size_follows_the_memory_budget(monkeypatch):
    """``"auto"`` = measured available memory over exact per-column bytes.

    The budget regime is solvax's "largest chunk that fits":
    ``memory_fraction * available // per_column_bytes`` clamped to
    ``[1, dim]``; without a measurement the square-root heuristic bounds
    the chunk instead of guessing a budget.
    """
    monkeypatch.setattr(im, "_measured_memory_bytes", lambda device=None: 1000)
    assert im.measured_chunk_size(64, 100) == 5  # 0.5 * 1000 // 100
    assert im.measured_chunk_size(3, 100) == 3  # clamped to dim
    assert im.measured_chunk_size(64, 10**9) == 1  # never below one column
    monkeypatch.setattr(im, "_measured_memory_bytes", lambda device=None: None)
    assert im.measured_chunk_size(64, 100) == 8  # ceil(sqrt(64)) fallback


def test_chunk_size_resolver_accepts_auto_and_rejects_other_strings():
    resolved = im._resolve_chunk_size(
        "auto", name="probe_chunk_size", dim=17, per_column_bytes=1 << 20)
    assert isinstance(resolved, int) and 1 <= resolved <= 17
    assert im._resolve_chunk_size(
        4, name="probe_chunk_size", dim=17, per_column_bytes=1) == 4
    with pytest.raises(ValueError, match="response_chunk_size"):
        im._resolve_chunk_size(
            "wide", name="response_chunk_size", dim=17, per_column_bytes=1)
    with pytest.raises(ValueError, match="probe_chunk_size"):
        im._resolve_chunk_size(
            0, name="probe_chunk_size", dim=17, per_column_bytes=1)


def test_tree_bytes_is_exact_from_shapes():
    tree = {"a": jnp.zeros((3, 4)), "b": np.zeros(5, dtype=np.float32)}
    assert im._tree_bytes(tree) == 3 * 4 * 8 + 5 * 4


def test_staged_tangent_matches_the_single_program(monkeypatch):
    """The staged tangent solve (stored factors, separate corrector, lazy refinement, padded chunks)
    reproduces the single-program responses, and still refines a column the corrector leaves uncertified."""
    inp, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    zero = jax.tree.map(jnp.zeros_like, p0)
    tangent_batch = jax.tree.map(lambda *x: jnp.stack(x), dataclasses.replace(
        zero, rbc=zero.rbc.at[inp.ntor, 1].set(1.0)), dataclasses.replace(
        zero, pres_scale=jnp.ones_like(zero.pres_scale)), dataclasses.replace(
        zero, zbs=zero.zbs.at[inp.ntor, 1].set(1.0)))

    reference, report = jax.jit(lambda: im._implicit_evolved_tangent_multi_rhs(
        p0, cfg, state, mask, tangent_batch, active_fields=im._active_state_fields(cfg),
        probe_chunk_size=2, response_chunk_size=2, certify_rtol=1e-8, certify_maxiter=1))()
    staged, staged_report = im.staged_tangent_multi_rhs(
        p0, cfg, state, mask, tangent_batch, chunk=2, certify_rtol=1e-8, certify_maxiter=1)
    assert np.all(np.asarray(staged_report.converged)) and np.asarray(staged_report.converged).shape == (3,)
    assert _relative_difference(staged, reference) < 1e-10
    np.testing.assert_allclose(np.asarray(staged_report.residual_norm), np.asarray(report.residual_norm),
                               rtol=1e-6, atol=1e-13)  # round-off-level residuals
    assert np.array_equal(np.asarray(staged_report.iterations), np.asarray(report.iterations))

    def stalled(A, b, cfg, *, x0=None, max_restarts=None, rtol=None):
        return jax.tree.map(lambda value: 1.001 * value, x0), im.LinearResponseReport(
            residual_norm=jnp.asarray(1.0e30), tolerance=jnp.asarray(0.0),
            iterations=jnp.asarray(1, jnp.int32), converged=jnp.asarray(False))

    # Every column fails the corrector's first-residual test: the corrector runs (and stalls), and the
    # refinement certifies the columns.
    check = im._staged_tangent_check

    def failing_check(*args, **kwargs):
        report, accepted = check(*args, **kwargs)
        return report._replace(converged=jnp.zeros_like(report.converged)), jnp.zeros_like(accepted)

    monkeypatch.setattr(im, "_staged_tangent_check", failing_check)
    monkeypatch.setattr(im, "_adjoint_solve", stalled)
    jax.clear_caches()
    refined, refined_report = im.staged_tangent_multi_rhs(
        p0, cfg, state, mask, tangent_batch, chunk=2, certify_rtol=1e-8, certify_maxiter=1)
    jax.clear_caches()
    assert np.all(np.asarray(refined_report.converged))
    assert _relative_difference(refined, reference) < 1e-8


@pytest.mark.parametrize("transpose", [False, True])
def test_scanned_block_thomas_solve_matches_solvax(transpose):
    """The scanned stored-factor solve is solvax's unrolled one, sweep for sweep."""
    from solvax import block_thomas_factor, block_thomas_solve

    rng = np.random.default_rng(0)
    n, m, k = 7, 5, 3
    lower, upper = (jnp.asarray(0.2 * rng.standard_normal((n, m, m))) for _ in range(2))
    diag = jnp.asarray(rng.standard_normal((n, m, m)) + 4.0 * np.eye(m))
    factors = block_thomas_factor(lower, diag, upper)
    rhs = jnp.asarray(rng.standard_normal((n, m, k)))
    got = im._block_thomas_solve_scan(factors, rhs, transpose=transpose)
    want = block_thomas_solve(factors, rhs, transpose=transpose)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-13, atol=1e-13)


def _fresh_small_solovev(max_iterations):
    """The small Solovev deck under a config no other test compiled (``max_iterations`` makes it unique)."""
    inp, _, p0 = _small_solovev_setup()
    cfg = im.make_config(inp, ftol=1.0e-10, max_iterations=max_iterations)
    return inp, cfg, p0


class _CompileLog:
    """Names of the programs JAX compiles (or looks up in its persistent cache) inside the context."""

    def __enter__(self):
        import logging

        self.names, outer = [], self

        class Handler(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                if message.startswith("Compiling "):
                    outer.names.append(message.split()[1])

        self._handler, self._logger = Handler(), logging.getLogger("jax")
        self._level = self._logger.level
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.WARNING)
        self._context = jax.log_compiles(True)
        self._context.__enter__()
        return self

    def __exit__(self, *exc):
        self._context.__exit__(*exc)
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._level)


def test_warm_refinement_compiles_the_refinement_programs_ahead():
    """warm_refinement leaves nothing for the refinement's real calls to compile."""
    from vmex.core.device import commit_to_single_device

    inp, cfg, p0 = _fresh_small_solovev(1013)
    im.warm_refinement(cfg, p0)
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    state, params, mask = commit_to_single_device((state, p0, mask))
    z = im._dof_projector(cfg, mask)(state)
    with _CompileLog() as log:
        im.residual_fn(cfg, state, mask)(z, params)
        factors = im._refine_block_factors(cfg, params, state, mask, z)
        im._refine_block_step(cfg, params, state, mask, z, factors)
    assert not [name for name in log.names if "refine" in name or "residual_lane" in name], log.names


def test_warm_staged_tangent_compiles_the_tangent_programs_ahead():
    """warm_staged_tangent leaves nothing for staged_tangent_multi_rhs to compile."""
    inp, cfg, p0 = _fresh_small_solovev(1017)
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    zero = jax.tree.map(jnp.zeros_like, p0)
    batch = jax.tree.map(lambda *x: jnp.stack(x), dataclasses.replace(
        zero, rbc=zero.rbc.at[inp.ntor, 1].set(1.0)), dataclasses.replace(
        zero, zbs=zero.zbs.at[inp.ntor, 1].set(1.0)), dataclasses.replace(
        zero, pres_scale=jnp.ones_like(zero.pres_scale)))
    im.warm_staged_tangent(cfg, p0, chunk=2, certify_rtol=1e-8, certify_maxiter=1, fallbacks=True)
    with _CompileLog() as log:
        _, report = im.staged_tangent_multi_rhs(p0, cfg, state, mask, batch, chunk=2, certify_rtol=1e-8,
                                                certify_maxiter=1)
    assert np.all(np.asarray(report.converged))
    assert not [name for name in log.names if "staged" in name or "refine" in name], log.names
    # the fallbacks a failing chunk needs are compiled too
    check = im._staged_tangent_check

    def failing_check(*args, **kwargs):
        report, accepted = check(*args, **kwargs)
        return report._replace(converged=jnp.zeros_like(report.converged)), jnp.zeros_like(accepted)

    try:
        im._staged_tangent_check = failing_check
        with _CompileLog() as log:
            im.staged_tangent_multi_rhs(p0, cfg, state, mask, batch, chunk=2, certify_rtol=1e-8, certify_maxiter=1)
    finally:
        im._staged_tangent_check = check
    assert not [name for name in log.names if "staged" in name], log.names


def test_block_state_pullback_with_stored_factors_matches():
    """The pullback through factors from block_factors_at equals the one that assembles its own."""
    inp, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    fields = im._active_state_fields(cfg)
    gbar = jax.tree.map(lambda a: jnp.sin(jnp.arange(a.size, dtype=a.dtype).reshape(a.shape)), state)
    own = im._block_state_pullback(p0, cfg, state, mask, active_fields=fields, probe_chunk_size=2)(gbar)
    stored = im._block_state_pullback(p0, cfg, state, mask, active_fields=fields, probe_chunk_size=2,
                                      factors=im.block_factors_at(cfg, p0, state, mask))(gbar)
    assert bool(stored[1].converged)
    assert _relative_difference(stored[0], own[0]) < 1e-10


def test_seeded_multigrid_solves_run_only_the_final_rung(monkeypatch):
    """A hot-restart trial is seeded at the final radial resolution: the coarse rungs that only build a seed
    are skipped, and the trial reaches the same equilibrium as a cold ladder."""
    inp, _, p0 = _small_solovev_setup()
    inp = dataclasses.replace(inp, ns_array=np.asarray([3, 5]), ftol_array=np.asarray([1e-8, 1e-10]),
                              niter_array=np.asarray([1000]))
    ladders = []
    real = im.solve_multigrid

    def recording(inp, **kwargs):
        ladders.append(list(np.asarray(kwargs["ns_array"])))
        return real(inp, **kwargs)

    monkeypatch.setattr(im, "solve_multigrid", recording)
    cfg = im.make_config(inp, multigrid=True, hot_restart=True, ftol=1e-10, max_iterations=1021)
    im.solve_implicit_with_aux(p0, cfg)
    moved = dataclasses.replace(p0, rbc=p0.rbc.at[inp.ntor, 1].add(1e-3))
    hot, _ = im.solve_implicit_with_aux(moved, cfg)
    cold_cfg = im.make_config(inp, multigrid=True, ftol=1e-10, max_iterations=1023)
    cold, _ = im.solve_implicit_with_aux(moved, cold_cfg)
    assert ladders[0] == [3, 5] and ladders[1] == [5] and ladders[2] == [3, 5]
    assert _relative_difference(hot, cold) < 1e-8


def test_overlapping_compiles_follow_the_cpu_affinity(monkeypatch):
    import os

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)
    assert not im._overlapping_compiles_pay()
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    assert im._overlapping_compiles_pay()
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert not im._overlapping_compiles_pay()


def test_staged_tangent_group_scale_accepts_a_near_null_column():
    """A column whose right-hand side is tiny next to the others' (the pressure direction here: ||b|| 1e-6 against
    3.7 for the boundary one, in the same units) is certified at the Jacobian's absolute accuracy: a response off
    by 1e-10 of the largest right-hand side, which its own relative test rejects (and sent to the GMRES
    corrector), is accepted, while one off by more than that tolerance is still rejected.  With
    ``column_scales`` the staged solve gives the same responses as without."""
    inp, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    zero = jax.tree.map(jnp.zeros_like, p0)
    tangent_batch = jax.tree.map(lambda *x: jnp.stack(x), dataclasses.replace(
        zero, rbc=zero.rbc.at[inp.ntor, 1].set(1.0)), dataclasses.replace(
        zero, pres_scale=jnp.ones_like(zero.pres_scale)))
    params, x_star, dof_mask = im._canonical_arrays(cfg, (p0, state, mask))
    factors = im.block_factors_at(cfg, p0, state, mask)
    piece = im._canonical_arrays(cfg, tangent_batch)
    initial = im._staged_tangent_initial(params, x_star, dof_mask, factors, piece, cfg, 2)
    norms = np.asarray(im._staged_rhs_norms(params, x_star, dof_mask, factors, piece, cfg, 2))
    assert norms[1] < 1e-5 * norms[0]
    rtol = 1e-8

    def verdicts(error, scale):
        # column 1 off by ``error`` x column 0's response: a raw residual of error x ||b_0||
        shifted = jax.tree.map(lambda a: a.at[1].add(error * a[0]), initial)
        _, accepted = im._staged_tangent_check(params, x_star, dof_mask, factors, piece, shifted,
                                               im._canonical_arrays(cfg, jnp.asarray(scale)), cfg, 2, rtol)
        return np.asarray(accepted)

    assert verdicts(1e-10, [0.0, 0.0]).tolist() == [True, False]  # own ||b||: rejected
    assert verdicts(1e-10, [norms[0]] * 2).tolist() == [True, True]  # the group's ||b||: accepted
    assert verdicts(1e-5, [norms[0]] * 2).tolist() == [True, False]  # beyond the group's tolerance: rejected
    own, _ = im.staged_tangent_multi_rhs(p0, cfg, state, mask, tangent_batch, chunk=2, certify_rtol=rtol,
                                         certify_maxiter=5)
    grouped, report = im.staged_tangent_multi_rhs(p0, cfg, state, mask, tangent_batch, chunk=2, certify_rtol=rtol,
                                                  certify_maxiter=5, column_scales=[1.0, 1.0])
    assert np.all(np.asarray(report.converged))
    assert _relative_difference(grouped, own) < 1e-12


def test_staged_tangent_column_scales_put_the_columns_in_one_unit():
    """The certificate scale is the largest right-hand side in the columns' units, converted back to each
    column's: the same boundary direction in metres and in micrometres (scales 1 and 1e6) is the same column in
    those units, so neither is loosened; in raw units the micrometre column would be held to the metre one's."""
    inp, cfg, p0 = _small_solovev_setup()
    state, mask = im.solve_implicit_with_aux(p0, cfg)
    zero = jax.tree.map(jnp.zeros_like, p0)
    tangent_batch = jax.tree.map(lambda *x: jnp.stack(x), dataclasses.replace(
        zero, rbc=zero.rbc.at[inp.ntor, 1].set(1.0)), dataclasses.replace(
        zero, rbc=zero.rbc.at[inp.ntor, 1].set(1e-6)))
    params, x_star, dof_mask = im._canonical_arrays(cfg, (p0, state, mask))
    factors = im.block_factors_at(cfg, p0, state, mask)
    params, x_star, dof_mask, pieces, chunk = im._staged_chunks(cfg, p0, state, mask, tangent_batch, 2)
    norms = np.asarray(im._staged_rhs_norms(params, x_star, dof_mask, factors, pieces[0], cfg, chunk))
    np.testing.assert_allclose(norms[1], 1e-6 * norms[0], rtol=1e-10)
    scaled = np.asarray(im._group_scales(cfg, params, x_star, dof_mask, factors, pieces, chunk, 2, [1.0, 1e6])[0])
    np.testing.assert_allclose(scaled, norms, rtol=1e-10)  # consistent units: each column at its own size
    raw = np.asarray(im._group_scales(cfg, params, x_star, dof_mask, factors, pieces, chunk, 2, [1.0, 1.0])[0])
    np.testing.assert_allclose(raw, [norms[0], norms[0]], rtol=1e-10)
    with pytest.raises(ValueError, match="column_scales"):
        im._group_scales(cfg, params, x_star, dof_mask, factors, pieces, chunk, 2, [1.0, 0.0])

"""Derivative certificates for the coupled free-boundary root."""

from __future__ import annotations

import contextlib
import dataclasses
import gc
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from scipy.sparse.linalg import LinearOperator, gcrotmk

from tests.test_lasym_free_case import lasym_free_field, lasym_free_input
from vmex.core import implicit as im
from vmex.core.input import VmecInput
from vmex.core.mgrid import MgridField, read_mgrid
from vmex.core.solver import SpectralState
from vmex.core.freeboundary_implicit import (
    make_free_boundary_config,
    solve_free_boundary_implicit,
    solve_free_boundary_implicit_status,
)
from vmex.core import freeboundary_implicit as fbi
from vmex.core.errors import (
    AdjointSolveError, VmecConvergenceError, VmecJacobianError)


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


def test_free_boundary_status_callback_turns_trial_error_into_status(monkeypatch):
    """An invalid optimizer trial returns status 1 instead of crossing JAX."""
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([20]),
    )
    field = lasym_free_field()
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-6, max_iterations=20,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
        device="cpu",
    )
    assert cfg.implicit.device.platform == "cpu"

    def fail(*_args, **_kwargs):
        raise VmecJacobianError("invalid trial")

    monkeypatch.setattr(fbi, "_host_solve_and_mask", fail)
    _state, status, fsq, ratio = solve_free_boundary_implicit_status(
        im.params_from_input(inp), field.extcur, cfg)
    assert status == 1
    assert np.isinf(fsq) and np.isinf(ratio)
    assert cfg.resolution.ns == 8

    monkeypatch.setattr(fbi, "_host_solve_and_mask",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("implementation bug")))
    with pytest.raises(RuntimeError, match="implementation bug"):
        fbi._host_solve_and_mask_status(cfg, im.params_from_input(inp), field.extcur)


def test_free_boundary_config_rejects_fixed_boundary_input():
    inp = dataclasses.replace(lasym_free_input(DATA), lfreeb=False)
    with pytest.raises(ValueError, match="LFREEB"):
        make_free_boundary_config(inp, lasym_free_field())


def test_free_boundary_config_validates_adjoint_solver():
    inp, field = lasym_free_input(DATA), lasym_free_field()
    with pytest.raises(ValueError, match="adjoint_solver must be one of"):
        make_free_boundary_config(inp, field, adjoint_solver="dense")
    for name in fbi._ADJOINT_SOLVERS:
        assert make_free_boundary_config(
            inp, field, adjoint_solver=name).adjoint_solver == name


def test_free_boundary_config_validates_adjoint_fail():
    """The adjoint failure policy is opt-in and defaults to raising.

    ``best_effort`` returns a stalled Krylov solution so that one bad trial
    in an optimization is a poor search direction rather than a dead run; a
    typo must not silently select it.
    """
    inp, field = lasym_free_input(DATA), lasym_free_field()
    assert make_free_boundary_config(inp, field).adjoint_fail == "error"
    assert make_free_boundary_config(
        inp, field, adjoint_fail="best_effort").adjoint_fail == "best_effort"
    with pytest.raises(ValueError, match="'error' or 'best_effort'"):
        make_free_boundary_config(inp, field, adjoint_fail="warn")


def test_host_adjoint_refreshes_saved_pullback_at_each_point():
    """Reusing an executable must not reuse a previous root's linearization."""
    def residual(z, p, *_args):
        return jnp.asarray([z[0]**2 + p * z[1], z[0] * z[1] + z[1]**2])

    cfg = SimpleNamespace(
        adjoint_tol=1e-10, adjoint_gcrot_m=2, adjoint_gcrot_k=1,
        adjoint_maxiter=10,
    )
    rhs = jnp.asarray([1., -2.])
    for z, p in [(jnp.asarray([2., 3.]), 0.5), (jnp.asarray([3., 2.]), 1.5)]:
        solved = fbi._host_adjoint(residual, z, p, None, None, None, None, rhs, cfg)
        jacobian = np.array([[2 * z[0], p], [z[1], z[0] + 2 * z[1]]])
        np.testing.assert_allclose(solved, np.linalg.solve(jacobian.T, rhs), rtol=1e-9)


def test_cheaper_transpose_is_accepted_only_on_the_exact_one(monkeypatch):
    """A response adjoint is certified on the coupled operator or not at all.

    The edge-response lane iterates on a cheaper transpose.  A wrong cheaper
    operator must cost a second solve on the exact one, never a wrong answer;
    a right one must be accepted without that second solve.
    """
    def residual(z, p, *_args):
        return jnp.asarray([z[0]**2 + p * z[1], z[0] * z[1] + z[1]**2])

    def mislinearized(z, p, *_args):
        return residual(z, p) + 0.3 * jnp.asarray([z[1]**2, z[0]])

    class Config(SimpleNamespace):
        __hash__ = object.__hash__  # the fallback is counted per configuration

    cfg = Config(
        adjoint_tol=1e-10, adjoint_gcrot_m=2, adjoint_gcrot_k=1,
        adjoint_maxiter=10,
    )
    z, p, rhs = jnp.asarray([2., 3.]), 0.5, jnp.asarray([1., -2.])
    exact = np.linalg.solve(np.array([[4., 0.5], [3., 8.]]).T, rhs)
    warm_starts, solver = [], fbi.gcrotmk

    def counted(*args, **kwargs):
        warm_starts.append(kwargs.get("x0"))
        return solver(*args, **kwargs)

    monkeypatch.setattr(fbi, "gcrotmk", counted)
    for lane, solves in ((mislinearized, 2), (residual, 1)):
        warm_starts.clear()
        solved = fbi._host_adjoint(
            residual, z, p, None, None, None, None, rhs, cfg, certify=True,
            pullback=fbi._prepare_transpose(
                z, p, None, None, None, None, residual=lane))
        np.testing.assert_allclose(solved, exact, rtol=1e-9)
        assert len(warm_starts) == solves
        assert warm_starts[0] is None and all(
            start is not None for start in warm_starts[1:])
    assert im._SOLVE_STATS[cfg]["adjoint_certificate_fallbacks"] == 1


def test_edge_response_models_nestor_and_drives_the_adjoint_lane():
    """The dense edge response, its lane and its gradient, on a small deck.

    The response is an exact linearization wherever it is built, root or not,
    so this needs no converged equilibrium to certify -- only one saved point
    used by both lanes.  Kept out of ``full`` on purpose: these are the paths
    a coverage lane has to execute, and they are cheap at this resolution.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA).change_resolution(
            mpol=3, ntor=0, ntheta=10, nzeta=4),
        ns_array=np.array([5]), ftol_array=np.array([1.0e-6]),
        niter_array=np.array([400]))
    field = lasym_free_field()
    params = im.params_from_input(inp)

    def configure(solver):
        return make_free_boundary_config(
            inp, field, ns=5, ftol=1.0e-6, max_iterations=400,
            adjoint_tol=1.0e-8, adjoint_maxiter=100, adjoint_solver=solver,
            field_from_parameters=lambda current: dataclasses.replace(
                field, extcur=current), device="cpu")

    cfg = configure("edge_response")
    (_state, _status, _fsq, _ratio), saved = fbi._solve_status_fwd(
        params, field.extcur, cfg)
    prm, current, solved, mask, rcon0, zcon0, _ = saved
    frozen = jax.lax.stop_gradient(solved)
    value, jacobian, inputs = fbi._edge_response(
        cfg, prm, current, frozen, rcon0, zcon0)
    assert jacobian.shape == value.shape + inputs.shape
    assert bool(jnp.all(jnp.isfinite(jacobian)))

    icfg = cfg.implicit
    runtime = dataclasses.replace(
        im.runtime_from_params(prm, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=int(icfg.resolution.ns))
    response = (value, jacobian, inputs)
    # At the point it was built on, the model reproduces NESTOR exactly.
    np.testing.assert_allclose(
        np.asarray(fbi._linearized_bsqvac(frozen, runtime, response)),
        np.asarray(cfg.vacuum_program.bsq(
            frozen, runtime, cfg.field_from_parameters(current))),
        rtol=1.0e-12, atol=0.0)

    project = im._dof_projector(icfg, mask)
    z_star = project(solved)
    exact = fbi._projected_residual(cfg, mask)
    modelled = fbi._projected_residual(cfg, mask, response=response)
    tangent = project(jax.tree.map(
        lambda leaf: jnp.asarray(np.random.default_rng(3).standard_normal(
            leaf.shape)) * 1.0e-3, z_star))
    call = lambda lane, z: lane(z, prm, current, frozen, rcon0, zcon0)  # noqa: E731
    products = [jax.jvp(lambda z: call(lane, z), (z_star,), (tangent,))[1]
                for lane in (exact, modelled)]
    assert float(im._tree_norm(jax.tree.map(jnp.subtract, *products))) <= (
        1.0e-10 * float(im._tree_norm(products[0])))

    # The saved response transpose is the transpose of that same lane.
    pullback = fbi._prepare_response_transpose(
        z_star, prm, current, frozen, rcon0, zcon0, mask, response, cfg=cfg)
    cotangent = project(jax.tree.map(
        lambda leaf: jnp.asarray(np.random.default_rng(4).standard_normal(
            leaf.shape)), z_star))
    left = float(_flat(pullback(cotangent)[0]) @ _flat(tangent))
    right = float(_flat(cotangent) @ _flat(products[1]))
    np.testing.assert_allclose(left, right, rtol=1.0e-9, atol=0.0)

    # The lane assembles the same gradient as the certified default, and
    # certifies it on the exact transpose rather than on its own model.
    state_bar = jax.grad(
        lambda state: jnp.mean(state.R_cos[-1] ** 2))(solved)
    gradients = [np.asarray(fbi._solve_bwd_impl(
        configure(name), saved[:6], state_bar)[1])
        for name in ("coupled_gcrot", "edge_response")]
    assert np.max(np.abs(gradients[0])) > 0.0
    np.testing.assert_allclose(gradients[1], gradients[0], rtol=1.0e-6,
                               atol=0.0)
    assert (im._SOLVE_STATS.get(configure("edge_response").implicit) or {}
            ).get("adjoint_certificate_fallbacks", 0) == 0


@contextlib.contextmanager
def monkeypatched_debug():
    """Run the adjoint lanes' diagnostic channel for one block."""
    original = im._adjoint_debug_enabled
    im._adjoint_debug_enabled = lambda: True
    try:
        yield
    finally:
        im._adjoint_debug_enabled = original


def test_edge_basis_pairs_the_constrained_m1_columns():
    """The evolved edge basis is orthonormal and folds the m=1 constraint.

    With ``lconm1`` and ``ntor > 0`` the edge carries redundant m=1
    directions: VMEC evolves one combination of each pair, so the basis must
    span each pair once, not twice, or the reduced edge system inherits a
    null direction.  No equilibrium is solved here -- the basis depends only
    on the configuration and the structural mask.
    """
    inp = VmecInput.from_file(DATA / "input.cth_like_free_bdy_lasym_small")
    data = read_mgrid(DATA / "mgrid_cth_like_lasym_small.nc")
    field = MgridField.from_mgrid_data(
        data, extcur=np.asarray(inp.extcur, dtype=float)[: data.nextcur])
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-6, max_iterations=10, device="cpu",
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current))
    icfg = cfg.implicit
    assert bool(icfg.lconm1) and int(icfg.resolution.ntor) > 0

    mn = int(np.asarray(im._template_runtime(icfg).modes.m).size)
    fields = im._active_state_fields(icfg)
    mask = SpectralState(**{
        name: jnp.ones((int(icfg.resolution.ns), mn))
        for name in im._STATE_FIELDS})
    packed_mask = np.ones(len(fields) * mn)
    basis = fbi._edge_basis(cfg, mask, packed_mask, jnp.float64)

    pairs = len(im._m1_pair_columns(icfg)[0])
    lasym_pairs = pairs * (2 if bool(icfg.resolution.lasym) else 1)
    assert basis.shape == (len(fields) * mn, len(fields) * mn - lasym_pairs)
    np.testing.assert_allclose(
        np.asarray(basis.T @ basis), np.eye(basis.shape[1]),
        rtol=0.0, atol=1.0e-12)


def test_balanced_dense_solver_survives_a_rank_deficient_edge_system():
    """A singular reduced edge system gets a least-squares solve, not a NaN.

    The edge system can inherit redundant m=1 directions on decks the basis
    above cannot fold away, and a plain solve would amplify them.  The exact
    coupled-residual certificate remains authoritative either way, so the
    requirement here is only that the reduced solve stays finite and
    reproduces the right answer on the range.
    """
    singular = np.array([[1.0, 2.0, 3.0],
                         [2.0, 4.0, 6.0],
                         [1.0, 1.0, 1.0]])
    solve_reduced, condition = fbi._balanced_dense_solver(singular)
    assert not (np.isfinite(condition)
                and condition < 1.0 / np.finfo(singular.dtype).eps)
    wanted = np.array([1.0, -2.0, 0.5])
    solution = solve_reduced(singular @ wanted)
    assert np.all(np.isfinite(solution))
    np.testing.assert_allclose(singular @ solution, singular @ wanted,
                               rtol=1.0e-9, atol=1.0e-9)

    well_posed = np.array([[4.0, 1.0], [1.0, 3.0]])
    solve_reduced, condition = fbi._balanced_dense_solver(well_posed)
    assert np.isfinite(condition)
    np.testing.assert_allclose(
        solve_reduced(well_posed @ np.array([2.0, -1.0])),
        np.array([2.0, -1.0]), rtol=1.0e-12, atol=0.0)


def test_schur_lanes_are_reusable_and_leak_nothing_per_gradient():
    """The radial-elimination lanes reuse one executable and strand no arrays.

    Every per-gradient array -- the bulk blocks, the mask, the saved
    transpose -- reaches the jitted helpers as an argument, so ``cfg`` is the
    only static key and one executable serves a whole optimization.  When
    these were closures defined inside the adjoint instead, each backward
    pass was a fresh jit key: the lane recompiled every gradient and JAX's
    trace cache kept that trial's arrays alive as jaxpr constants, a whole
    block system per successful gradient: measured on the free-boundary
    reproducer, 2218 live arrays and 0.289 GiB of device buffers each time.
    What remains here is one array per gradient, bookkeeping rather than a
    block system, so the bound below is a small constant, not equality.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA).change_resolution(
            mpol=3, ntor=0, ntheta=10, nzeta=4),
        ns_array=np.array([5]), ftol_array=np.array([1.0e-6]),
        niter_array=np.array([400]))
    field = lasym_free_field()
    params = im.params_from_input(inp)

    def configure(solver):
        return make_free_boundary_config(
            inp, field, ns=5, ftol=1.0e-6, max_iterations=400,
            adjoint_tol=1.0e-8, adjoint_maxiter=100, adjoint_solver=solver,
            field_from_parameters=lambda current: dataclasses.replace(
                field, extcur=current), device="cpu")

    cfg = configure("boundary_schur")
    (_state, _status, _f, _r), saved = fbi._solve_status_fwd(
        params, field.extcur, cfg)
    state_bar = jax.grad(
        lambda state: jnp.mean(state.R_cos[-1] ** 2))(saved[2])

    live, gradients = [], []
    for index in range(3):
        # A different cotangent each time: a repeated one could be served
        # from a memo without exercising the lane at all.
        scaled = jax.tree.map(lambda leaf, k=index: leaf * (1.0 + 0.1 * k),
                              state_bar)
        gradients.append(np.asarray(
            fbi._solve_bwd_impl(cfg, saved[:6], scaled)[1]))
        gc.collect()
        live.append(len(jax.live_arrays()))
    # One block system is (ns, block, block) plus its scalings, tens of
    # arrays; a bound of two separates bookkeeping from that defect.
    assert live[2] - live[1] <= 2, (
        f"the lane stranded {live[2] - live[1]} arrays in one gradient: {live}")
    assert np.max(np.abs(gradients[0])) > 0.0

    # The diagnostic channel runs too, so its formatting cannot rot unnoticed.
    with monkeypatched_debug():
        elimination = np.asarray(fbi._solve_bwd_impl(
            configure("boundary_schur"), saved[:6], state_bar)[1])
    np.testing.assert_allclose(elimination, gradients[0], rtol=1.0e-6,
                               atol=0.0)

    # A certificate this lane cannot meet must cost a second solve on the
    # exact operator and be counted, never be returned as it stands.
    schur = configure("boundary_schur")
    strict, acceptance = [0], im._adjoint_acceptance

    def first_call_is_impossible(cfg_arg, norm, rtol=None):
        strict[0] += 1
        if strict[0] == 1:
            return 0.0
        return acceptance(cfg_arg, norm, rtol) if rtol is not None else (
            acceptance(cfg_arg, norm))

    before = (im._SOLVE_STATS.get(schur.implicit) or {}).get(
        "adjoint_certificate_fallbacks", 0) or 0
    im._adjoint_acceptance = first_call_is_impossible
    try:
        forced = np.asarray(fbi._solve_bwd_impl(
            schur, saved[:6], state_bar)[1])
    finally:
        im._adjoint_acceptance = acceptance
    after = (im._SOLVE_STATS.get(schur.implicit) or {})[
        "adjoint_certificate_fallbacks"]
    assert after == before + 1
    np.testing.assert_allclose(forced, elimination, rtol=1.0e-6, atol=0.0)


def test_traced_adjoint_linearizes_inside_an_outer_jit(monkeypatch):
    """Under an outer jax.jit the pullback is taken at the root, then staged GCROT."""
    def residual(z, p, field, *_args):
        return jnp.asarray([z[0]**2 + p * z[1] + field, z[0] * z[1] + z[1]**2 - p])

    monkeypatch.setattr(fbi, "_projected_residual", lambda *_args, **_kwargs: residual)
    monkeypatch.setattr(im, "_dof_projector", lambda *_args: (lambda tree: tree))
    cfg = SimpleNamespace(
        implicit=SimpleNamespace(adjoint_tol=1e-12, adjoint_gcrot_m=2,
                                 adjoint_gcrot_k=1, adjoint_maxiter=10),
        adjoint_solver="coupled_gcrot", adjoint_fail="error",
    )
    z, p, field = jnp.asarray([2., 3.]), jnp.asarray(0.5), jnp.asarray(0.25)
    rhs = jnp.asarray([1., -2.])
    params_bar, field_bar = jax.jit(
        lambda bar: fbi._solve_bwd_impl(cfg, (p, field, z, None, None, None), bar))(rhs)
    lam = np.linalg.solve(np.array([[4., 0.5], [3., 8.]]).T, rhs)
    np.testing.assert_allclose(params_bar, -(np.array([3., -1.]) @ lam), rtol=1e-10)
    np.testing.assert_allclose(field_bar, -lam[0], rtol=1e-10)


def test_projected_residual_memo_skips_a_traced_mask():
    """The closure memo keys on mask bytes; a traced mask has none.

    This is the fast guard for the line the stubbed test above never reaches:
    the real :func:`_projected_residual` is called under ``jax.jit`` (enabled
    for this module) with a traced mask, and eagerly with a concrete one.
    """
    cfg = make_free_boundary_config(lasym_free_input(DATA), lasym_free_field())
    mask = {"rows": jnp.zeros((4, 3))}
    eager = fbi._projected_residual(cfg, mask)
    assert fbi._projected_residual(cfg, mask) is eager  # the eager memo hits
    traced = []
    jax.jit(lambda m: traced.append(fbi._projected_residual(cfg, m)) or 0.0)(mask)
    assert len(traced) == 1 and traced[0] is not eager


@pytest.mark.full
def test_jitted_free_boundary_gradient_matches_the_eager_path():
    """A jitted free-boundary objective runs and reproduces the eager gradient.

    The test above stubs :func:`_projected_residual` out, so nothing exercised
    the real closure under a trace.  Its memo keys on the mask bytes, which a
    traced mask does not have, and every jitted free-boundary value-and-gradient
    raised ``TracerArrayConversionError`` there before its first adjoint matvec.

    Both calls start from the same hot state: the root is history-dependent at
    a finite ``ftol`` (see the reproducibility test at the end of this file),
    so a cold first call would compare two roots rather than two lanes.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([16]),
        ftol_array=np.array([1.0e-7]), niter_array=np.array([2500]),
    )
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-7, max_iterations=2500,
        adjoint_tol=1.0e-10, adjoint_maxiter=400,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )

    def objective(current):
        state, _, _, _ = solve_free_boundary_implicit_status(params, current, cfg)
        return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

    current = jnp.asarray(field.extcur)
    objective(current)
    eager_value, eager_gradient = jax.value_and_grad(objective)(current)
    jit_value, jit_gradient = jax.jit(jax.value_and_grad(objective))(current)
    np.testing.assert_allclose(jit_value, eager_value, rtol=1.0e-12)
    # Host and staged GCROT solve the same system to ``adjoint_tol``, not to
    # machine precision, so their gradients agree to that solve tolerance.
    np.testing.assert_allclose(jit_gradient, eager_gradient, rtol=1.0e-6,
                               atol=1.0e-9 * float(jnp.linalg.norm(eager_gradient)))
    # The eager memo still hits: equal mask content returns one closure.
    mask = fbi._FREE_MASK_CACHE[fbi._mask_key(cfg)]
    assert fbi._projected_residual(cfg, mask) is fbi._projected_residual(cfg, mask)


def test_host_adjoint_best_effort_warns_instead_of_raising(monkeypatch):
    """A stalled Krylov solve is a warning under the opt-in policy, not a stop.

    The stall arrives inside the VJP, where an optimizer's own
    ``lax.cond`` on the solve status cannot catch it, so a raise ends the
    whole run over one bad line-search trial.  Under ``best_effort`` the
    inaccurate direction is returned instead and the line search rejects it.
    """
    cfg = SimpleNamespace(adjoint_tol=1.0e-10, adjoint_maxiter=5,
                          adjoint_gcrot_m=2, adjoint_gcrot_k=1)

    def residual(z, params, field, base, rcon, zcon):
        return 2.0 * z

    z_star, rhs = jnp.arange(4.0), jnp.ones(4)
    # A Krylov solve that returns a deliberately wrong answer, so the true
    # residual check that follows it cannot pass.
    monkeypatch.setattr(fbi, "gcrotmk", lambda *args, **kwargs: (np.zeros(4), 0))
    def call(**kwargs):
        return fbi._host_adjoint(
            residual, z_star, None, None, z_star, None, None, rhs, cfg, **kwargs)

    with pytest.raises(AdjointSolveError, match="did not converge"):
        call()
    with pytest.warns(RuntimeWarning, match="best_effort"):
        solution = call(fail="best_effort")
    np.testing.assert_allclose(np.asarray(solution), np.zeros(4))


def test_cold_start_ladders_only_when_one_rung_cannot_converge(monkeypatch):
    """A converged single rung is kept; a stalled one falls back to a ladder."""
    inp = lasym_free_input(DATA)
    field = lasym_free_field()
    asked = []

    def ladder(_inp, *, ns_array, external_field, **_kwargs):
        asked.append(tuple(int(value) for value in ns_array))
        return SimpleNamespace(state="coarse")

    monkeypatch.setattr(
        "vmex.core.multigrid.solve_free_boundary_multigrid", ladder)
    for ns, converged, expected in ((31, True, None), (4, False, None),
                                    (8, False, (4, 8)), (31, False, (15, 31))):
        cfg = make_free_boundary_config(inp, field, ns=ns, ftol=1.0e-6,
                                        max_iterations=20)
        seen = []

        def solve(*, initial_state, _seen=seen, _converged=converged):
            _seen.append(initial_state)
            return SimpleNamespace(
                result=SimpleNamespace(converged=_converged), seed=initial_state)

        stage = fbi._cold_reference(solve, cfg.implicit, inp, field)
        assert seen[0] is None  # the single rung is always tried first
        assert (asked[-1] if asked else None) == expected
        assert stage.seed == ("coarse" if expected is not None else None)


def test_restart_carries_the_reference_continuation_but_not_its_vacuum_cache():
    """The trial's own field rebuilds the vacuum caches; the rest continues."""
    stage = SimpleNamespace(
        continuation_state="state", vacuum="vacuum", rcon0="rcon", zcon0="zcon",
        result=SimpleNamespace(fsqr=1.0, fsqz=2.0, fsql=3.0))
    restart = fbi._continuation(stage)
    assert restart == {
        "initial_state": "state", "vacuum_continuation": "vacuum",
        "constraint_continuation": ("rcon", "zcon"),
        "residual_continuation": (1.0, 2.0, 3.0)}
    assert "reuse_vacuum_cache" not in restart


def test_traced_pullback_says_it_cannot_run_the_host_schur_lane(monkeypatch):
    """boundary_schur is a host lane; under jit the staged solve replaces it."""
    def residual(z, p, field, *_args):
        return jnp.asarray([z[0]**2 + p * z[1] + field, z[0] * z[1] + z[1]**2 - p])

    monkeypatch.setattr(fbi, "_projected_residual", lambda *_a, **_k: residual)
    monkeypatch.setattr(im, "_dof_projector", lambda *_args: (lambda tree: tree))
    cfg = SimpleNamespace(
        implicit=SimpleNamespace(adjoint_tol=1.0e-12, adjoint_gcrot_m=2,
                                 adjoint_gcrot_k=1, adjoint_maxiter=10),
        adjoint_solver="boundary_schur", adjoint_fail="error")
    saved = (jnp.asarray(0.5), jnp.asarray(0.25), jnp.asarray([2., 3.]),
             None, None, None)
    # The warning has to name both consequences: a different solver, and the
    # slower one, so a user who jits does not silently pay for both.
    with pytest.warns(RuntimeWarning, match="not available under jax.jit") as caught:
        jax.jit(lambda bar: fbi._solve_bwd_impl(cfg, saved, bar))(
            jnp.asarray([1., -2.]))
    message = str(caught[0].message)
    assert "staged" in message and "slower" in message


def test_free_boundary_warm_failure_retries_once_from_cold(monkeypatch):
    """A bad cached state is discarded, but implementation errors are not."""
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([20]))
    field = lasym_free_field()
    cfg = make_free_boundary_config(inp, field, ns=8, ftol=1.0e-6,
                                    max_iterations=20)
    runtime = im._template_runtime(cfg.implicit)
    state = im._initial_state(runtime.setup)
    # These are module-level caches keyed on (resolution, lconm1, ncurr), so
    # monkeypatch.setitem restores both entries at teardown; a bare assignment
    # would hand the all-zero mask to the next free-boundary case sharing that
    # key, which reaches the Schur lane as an edge basis with no columns.
    # The cache holds the reference stage whose whole continuation restarts
    # every solve; only its spectral state identifies it to the stub below.
    seed = object()
    reference = SimpleNamespace(
        continuation_state=seed, vacuum=None, rcon0=runtime.rcon0,
        zcon0=runtime.zcon0,
        result=SimpleNamespace(fsqr=0.0, fsqz=0.0, fsql=0.0))
    monkeypatch.setitem(fbi._FREE_HOT_CACHE, cfg, reference)
    monkeypatch.setitem(fbi._FREE_MASK_CACHE, fbi._mask_key(cfg),
                        jax.tree.map(jnp.zeros_like, state))
    calls = []

    def solve(*_args, initial_state=None, **_kwargs):
        calls.append(initial_state)
        if initial_state is seed:
            raise VmecJacobianError("bad warm state")
        result = SimpleNamespace(state=state, fsqr=0.0, fsqz=0.0, fsql=0.0,
                                 converged=True)
        return SimpleNamespace(result=result, continuation_state=state,
                               rcon0=runtime.rcon0, zcon0=runtime.zcon0)

    monkeypatch.setattr(fbi, "_solve_free_boundary_stage", solve)
    # The cold retry builds its start from a coarse rung; that ladder is a real
    # solve, which this stubbed unit test does not need in order to check that
    # the bad reference is dropped.
    monkeypatch.setattr(fbi, "_cold_reference",
                        lambda solve, *_args: solve(initial_state=None))
    solved, *_ = fbi._host_solve_and_mask(cfg, im.params_from_input(inp), field)
    assert calls == [seed, None]
    np.testing.assert_allclose(solved.R_cos, state.R_cos)


def test_failed_trials_do_not_change_a_rebuilt_point(monkeypatch):
    """A cold-rebuilt point stays identical after unrelated failed trials."""
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([20]))
    field = lasym_free_field()
    # The stubbed states are not roots of anything; this is about which
    # stage is returned, so the Newton anchor is off.
    cfg = make_free_boundary_config(inp, field, ns=8, ftol=1.0e-6,
                                    max_iterations=20, refine_tol=np.inf)
    runtime = im._template_runtime(cfg.implicit)
    state = im._initial_state(runtime.setup)
    reference = SimpleNamespace(
        continuation_state=object(), vacuum=None, rcon0=runtime.rcon0,
        zcon0=runtime.zcon0,
        result=SimpleNamespace(fsqr=0.0, fsqz=0.0, fsql=0.0))
    monkeypatch.setitem(fbi._FREE_HOT_CACHE, cfg, reference)
    monkeypatch.setitem(fbi._FREE_MASK_CACHE, fbi._mask_key(cfg),
                        jax.tree.map(jnp.zeros_like, state))
    rebuilt_state = dataclasses.replace(state, R_cos=state.R_cos + 1.0)

    def stage(converged, marker, result_state=state):
        fsq = 0.0 if converged else 1.0e-3
        return SimpleNamespace(
            result=SimpleNamespace(
                state=result_state, fsqr=fsq, fsqz=0.0, fsql=0.0,
                converged=converged, marker=marker),
            continuation_state=result_state, rcon0=runtime.rcon0,
            zcon0=runtime.zcon0)

    # The restart from the reference never reaches ftol.
    monkeypatch.setattr(fbi, "_solve_free_boundary_stage",
                        lambda *_a, **_k: stage(False, "restart"))
    current = jnp.asarray(field.extcur)
    failed_current = current.at[0].add(1.0)
    rebuilds = []

    def rebuild(_solve, _icfg, _inp, trial_field):
        productive = np.array_equal(np.asarray(trial_field), np.asarray(current))
        rebuilds.append(productive)
        return stage(productive, "rebuild" if productive else "failed-rebuild",
                     rebuilt_state if productive else state)

    monkeypatch.setattr(fbi, "_cold_reference", rebuild)
    first, *_rest, first_status, _fsq, _ratio = fbi._host_solve_and_mask_status(
        cfg, im.params_from_input(inp), current)
    assert int(first_status) == 0
    assert rebuilds == [True]
    assert fbi._FREE_LAST_RESULT[cfg].marker == "rebuild"
    assert fbi._FREE_HOT_CACHE[cfg] is reference  # the reference stands

    failed_trials = 8  # the former global budget was exhausted at this point
    for _ in range(failed_trials):
        *_ignored, status, _fsq, _ratio = fbi._host_solve_and_mask_status(
            cfg, im.params_from_input(inp), failed_current)
        assert int(status) == 2
    assert rebuilds == [True] + [False] * failed_trials

    repeated, *_rest, repeated_status, _fsq, _ratio = (
        fbi._host_solve_and_mask_status(
            cfg, im.params_from_input(inp), current))
    assert int(repeated_status) == 0
    np.testing.assert_array_equal(repeated.R_cos, first.R_cos)
    assert fbi._FREE_LAST_RESULT[cfg].marker == "rebuild"


def test_restart_budget_is_the_cold_reference_cost():
    """A restart gets what the cold reference needed, within [cap/10, cap]."""
    icfg = SimpleNamespace(max_iterations=4000)

    def reference(iterations):
        return SimpleNamespace(result=SimpleNamespace(iterations=iterations))

    assert fbi._restart_budget(icfg, reference(943)) == 943
    assert fbi._restart_budget(icfg, reference(51)) == 400
    assert fbi._restart_budget(icfg, reference(9000)) == 4000
    # A reference that carries no count (a stub, or a foreign stage) keeps
    # the whole budget: the old behaviour, never a smaller one by accident.
    assert fbi._restart_budget(icfg, SimpleNamespace(result=SimpleNamespace())) == 4000


def test_stalled_restart_goes_cold_within_the_budget(monkeypatch):
    """A restart that cannot converge is cut at the budget, then solved cold.

    Before, it ran to ``max_iterations``: from halfway to the optimum of the
    finite-beta single-stage deck every trial spent 4000 iterations (19-21 s
    of a 25 s trial) in a restart that never converged before the cold solve
    that certified it.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([1000]))
    field = lasym_free_field()
    cfg = make_free_boundary_config(inp, field, ns=8, ftol=1.0e-6,
                                    max_iterations=1000, refine_tol=np.inf)
    runtime = im._template_runtime(cfg.implicit)
    state = im._initial_state(runtime.setup)
    reference = SimpleNamespace(
        continuation_state=object(), vacuum=None, rcon0=runtime.rcon0,
        zcon0=runtime.zcon0,
        result=SimpleNamespace(fsqr=0.0, fsqz=0.0, fsql=0.0, iterations=300))
    monkeypatch.setitem(fbi._FREE_HOT_CACHE, cfg, reference)
    monkeypatch.setitem(fbi._FREE_MASK_CACHE, fbi._mask_key(cfg),
                        jax.tree.map(jnp.zeros_like, state))
    budgets, colds = [], []

    def stage(converged, marker):
        fsq = 0.0 if converged else 1.0e-3
        return SimpleNamespace(
            result=SimpleNamespace(state=state, fsqr=fsq, fsqz=0.0, fsql=0.0,
                                   converged=converged, marker=marker),
            continuation_state=state, rcon0=runtime.rcon0, zcon0=runtime.zcon0)

    def restart(*_args, max_iterations=None, **_kwargs):
        budgets.append(max_iterations)
        return stage(False, "restart")

    def cold(_solve, _icfg, _inp, _field):
        colds.append(True)
        return stage(True, "cold")

    monkeypatch.setattr(fbi, "_solve_free_boundary_stage", restart)
    monkeypatch.setattr(fbi, "_cold_reference", cold)
    *_, status, _fsq, _ratio = fbi._host_solve_and_mask_status(
        cfg, im.params_from_input(inp), field.extcur)
    assert budgets == [300]      # the reference's cost, not max_iterations
    assert colds == [True]
    assert int(status) == 0
    assert fbi._FREE_LAST_RESULT[cfg].marker == "cold"
    assert fbi._FREE_HOT_CACHE[cfg] is reference  # the reference stands


def test_an_unanchored_solve_is_never_certified(monkeypatch):
    """ftol met but no coupled root: status 3 on trials, an error otherwise."""
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([20]))
    field = lasym_free_field()
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-6, max_iterations=20,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current))
    runtime = im._template_runtime(cfg.implicit)
    state = im._initial_state(runtime.setup)
    reference = SimpleNamespace(
        continuation_state=state, vacuum=None, rcon0=runtime.rcon0,
        zcon0=runtime.zcon0,
        result=SimpleNamespace(fsqr=0.0, fsqz=0.0, fsql=0.0))
    monkeypatch.setitem(fbi._FREE_HOT_CACHE, cfg, reference)
    monkeypatch.setitem(fbi._FREE_MASK_CACHE, fbi._mask_key(cfg),
                        jax.tree.map(jnp.zeros_like, state))
    monkeypatch.setattr(
        fbi, "_solve_free_boundary_stage",
        lambda *_a, **_k: SimpleNamespace(
            result=SimpleNamespace(state=state, fsqr=0.0, fsqz=0.0, fsql=0.0,
                                   converged=True),
            continuation_state=state, rcon0=runtime.rcon0,
            zcon0=runtime.zcon0))
    anchored = []

    def not_contracting(_cfg, _params, _field, host_state, *_args):
        anchored.append(True)
        return host_state, fbi._AnchorReport(
            certified=False, initial_residual=3.2e-6, residual=4.0e-7,
            steps=12)

    monkeypatch.setattr(fbi, "_anchor_root", not_contracting)
    params = im.params_from_input(inp)
    *_, status, _fsq, _ratio = fbi._host_solve_and_mask_status(
        cfg, params, field.extcur)
    assert anchored and int(status) == 3
    with pytest.raises(VmecConvergenceError, match="did not anchor"):
        fbi._host_solve_and_mask(cfg, params, field.extcur)
    # No pullback runs for it: both cotangents are exactly zero.
    cotangents = fbi._solve_status_bwd(
        cfg, (params, field.extcur, state, state, runtime.rcon0,
              runtime.zcon0, np.int32(3)),
        (jax.tree.map(jnp.ones_like, state), None, None, None))
    assert all(not np.any(np.asarray(leaf))
               for leaf in jax.tree.leaves(cotangents))


def test_anchor_linear_solve_is_a_newton_correction_of_the_coupled_residual():
    """The staged anchor solve inverts the exact coupled Jacobian at ``z``.

    Real lanes on a small deck: bulk block factors and NESTOR's dense edge
    response built at ``z``, one compiled GMRES solve, then the correction
    checked against the exact NESTOR tangent of the raw coupled residual.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA).change_resolution(
            mpol=3, ntor=0, ntheta=10, nzeta=4),
        ns_array=np.array([5]), ftol_array=np.array([1.0e-6]),
        niter_array=np.array([400]))
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=5, ftol=1.0e-6, max_iterations=400,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current), device="cpu")
    current = jnp.asarray(field.extcur)
    state, mask, rcon0, zcon0, *_ = fbi._host_solve_and_mask_status(
        cfg, params, current)
    state, mask = (jax.tree.map(jnp.asarray, tree) for tree in (state, mask))
    rcon0, zcon0 = jnp.asarray(rcon0), jnp.asarray(zcon0)
    icfg = cfg.implicit
    project = im._dof_projector(icfg, mask)
    z = project(state)
    lane = (params, current, state, rcon0, zcon0, mask)

    ns = int(icfg.resolution.ns)
    rt = dataclasses.replace(
        im.runtime_from_params(params, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=ns, presf_ns_scale=fbi._presf_ns_scale_traceable(
            params, icfg.inp, ns))
    bsqvac = cfg.vacuum_program.bsq(state, rt, cfg.field_from_parameters(current))
    lower, diagonal, upper, row_scale, column_scale = fbi._frozen_bulk_blocks(
        *lane[:5], mask, z, bsqvac, cfg=cfg,
        probe_chunk_size=fbi._anchor_probe_chunk(cfg, mask))
    factors = fbi._anchor_factor(lower, diagonal, upper, row_scale,
                                 column_scale)
    response = fbi._edge_response(cfg, params, current, state, rcon0, zcon0)
    force = fbi._anchor_raw_residual(z, *lane, cfg=cfg)
    step, iterations, linear = fbi._anchor_linear_solve(
        force, z, *lane[:5], mask, response, factors, row_scale, column_scale,
        cfg=cfg, rtol=1.0e-9)
    assert 0 < int(iterations) <= fbi._ANCHOR_KRYLOV and float(linear) <= 1.0e-9

    _, image = jax.jvp(
        lambda zz: fbi._projected_residual_lane(
            zz, *lane[:5], mask, None, None, cfg=cfg, formulation="raw"),
        (z,), (fbi._unpack_projected(step, mask, cfg=cfg),))
    image = fbi._pack_projected(image, mask, cfg=cfg)
    error = float(jnp.linalg.norm(image + force) / jnp.linalg.norm(force))
    assert error < 1.0e-7, error
    np.testing.assert_allclose(
        float(fbi._anchor_norm(z, *lane, cfg=cfg)),
        float(im._tree_norm(fbi._projected_residual(cfg, mask)(z, *lane[:5]))),
        rtol=1.0e-12)


def _toy_anchor(monkeypatch, *, jacobian_sign=1.0, refine_tol=1.0e-10):
    """A real free-boundary config whose coupled residual is a toy quadratic.

    ``F(p) = d * (p - p*) + gamma * (w . (p - p*))**2 * u`` in the packed
    projected coordinates, its linear solves exact: the anchor's loop,
    damping and bookkeeping run for real against a root whose location is
    known exactly.  ``jacobian_sign=-1`` hands back ascent directions.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-6]), niter_array=np.array([20]))
    field = lasym_free_field()
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-6, max_iterations=20,
        refine_tol=refine_tol,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current))
    cfg = dataclasses.replace(
        cfg, vacuum_program=SimpleNamespace(bsq=lambda *_args: 0.0))
    runtime = im._template_runtime(cfg.implicit)
    state = im._initial_state(runtime.setup)
    mask = jax.tree.map(jnp.ones_like, state)
    packed = np.asarray(fbi._pack_projected(state, mask, cfg=cfg))
    rng = np.random.default_rng(3)
    diagonal = 1.0 + rng.random(packed.shape)
    root = packed + 1.0e-3 * rng.standard_normal(packed.shape)
    w, u = rng.standard_normal(packed.shape), rng.standard_normal(packed.shape)
    w, u = w / np.linalg.norm(w), u / np.linalg.norm(u)
    gamma = 30.0
    calls = {"response": 0, "factor": 0}

    def residual(z):
        offset = np.asarray(fbi._pack_projected(z, mask, cfg=cfg)) - root
        return diagonal * offset + gamma * np.sum(w * offset) ** 2 * u

    def solve(force, z_at, *_args, rtol, **_kwargs):
        # Sherman--Morrison on the rank-one Jacobian of the toy residual.
        offset = np.asarray(fbi._pack_projected(z_at, mask, cfg=cfg)) - root
        scaled_u = 2.0 * gamma * np.sum(w * offset) * u / diagonal
        rhs = np.asarray(force) / diagonal
        solution = rhs - scaled_u * np.sum(w * rhs) / (1.0 + np.sum(w * scaled_u))
        return jnp.asarray(-jacobian_sign * solution), 3, rtol

    def factor(*_args, **_kwargs):
        calls["factor"] += 1
        return (None,) * 5

    def response(*_args, **_kwargs):
        calls["response"] += 1
        return "response"

    monkeypatch.setattr(fbi, "_anchor_norm", lambda z, *_a, **_k: float(
        np.linalg.norm(residual(z))))
    monkeypatch.setattr(fbi, "_anchor_raw_residual",
                        lambda z, *_a, **_k: jnp.asarray(residual(z)))
    monkeypatch.setattr(fbi, "_anchor_linear_solve", solve)
    monkeypatch.setattr(fbi, "_frozen_bulk_blocks", factor)
    monkeypatch.setattr(fbi, "_anchor_factor", lambda *_args: "factors")
    monkeypatch.setattr(fbi, "_edge_response", response)

    def run():
        return fbi._anchor_root(cfg, im.params_from_input(inp), field.extcur,
                                state, mask, runtime.rcon0, runtime.zcon0)

    return run, state, mask, cfg, root, calls


def test_newton_anchor_lands_on_the_root_and_reports_what_it_did(
        monkeypatch, capsys):
    """Damped Newton reaches the known toy root; each exit is reported."""
    run, state, mask, cfg, root, calls = _toy_anchor(monkeypatch)
    monkeypatch.setattr(im, "_adjoint_debug_enabled", lambda: True)
    anchored, report = run()
    assert report.certified and report.residual <= 1.0e-10
    assert report.initial_residual > 1.0e-4 and report.steps >= 2
    assert report.factorizations == 1 and calls["factor"] == 1
    assert calls["response"] == report.steps  # rebuilt at every iterate
    np.testing.assert_allclose(
        np.asarray(fbi._pack_projected(anchored, mask, cfg=cfg)), root,
        rtol=0.0, atol=1.0e-12)
    assert report.displacement > 0.0
    assert "[vmex anchor] step 1" in capsys.readouterr().out

    # A stale factorization is rebuilt before the next step.
    monkeypatch.setattr(fbi, "_ANCHOR_REFACTOR_KRYLOV", 0)
    calls.update(response=0, factor=0)
    _, report = run()
    assert report.certified and report.factorizations == report.steps
    assert calls["factor"] == report.steps


def test_newton_anchor_never_certifies_what_it_cannot_land(
        monkeypatch, capsys):
    """No descent, no budget or no anchor: the host state and the reason."""
    run, state, *_ = _toy_anchor(monkeypatch, jacobian_sign=-1.0)
    monkeypatch.setattr(im, "_adjoint_debug_enabled", lambda: True)
    anchored, report = run()
    assert "no damped step passes" in capsys.readouterr().out
    assert not report.certified and report.residual == report.initial_residual
    assert report.attempts == len(fbi._ANCHOR_FIRST_DAMPING)
    for returned, host in zip(jax.tree.leaves(anchored), jax.tree.leaves(state)):
        np.testing.assert_array_equal(returned, host)

    run, state, *_ = _toy_anchor(monkeypatch)
    monkeypatch.setattr(fbi, "_ANCHOR_MAX_STEPS", 1)
    _, report = run()
    # Every attempt restarts from the host state with a smaller first step.
    assert not report.certified
    assert report.attempts == len(fbi._ANCHOR_FIRST_DAMPING) == report.steps
    assert report.residual < report.initial_residual

    run, state, *_ = _toy_anchor(monkeypatch, refine_tol=np.inf)
    anchored, report = run()
    assert report.certified and anchored is state and report.steps == 0

    run, state, *_ = _toy_anchor(monkeypatch, refine_tol=1.0)
    _, report = run()      # already inside the tolerance: no Newton step
    assert report.certified and report.steps == 0


def test_free_boundary_host_adjoint_rejects_a_false_solver_success(monkeypatch):
    """The true transpose residual, not SciPy's info flag, certifies a gradient."""
    cfg = SimpleNamespace(adjoint_tol=1.0e-10, adjoint_gcrot_m=3,
                          adjoint_gcrot_k=1, adjoint_maxiter=2)
    monkeypatch.setattr(fbi, "gcrotmk", lambda *_args, **_kwargs: (np.zeros(2), 0))
    def residual(z, *_args):
        return z

    with pytest.raises(AdjointSolveError, match="host GCROT"):
        fbi._host_adjoint(residual, jnp.zeros(2), None, None, None, None, None,
                          jnp.ones(2), cfg)

    monkeypatch.setattr(
        fbi, "gcrotmk", lambda *_args, **_kwargs: (np.full(2, np.nan), 0))
    with pytest.raises(AdjointSolveError, match="host GCROT"):
        fbi._host_adjoint(residual, jnp.zeros(2), None, None, None, None, None,
                          jnp.ones(2), cfg)


@pytest.mark.full
def test_free_boundary_current_gradient_matches_resolve_finite_difference():
    """The implicit coil-current response agrees with two independent solves."""
    inp = lasym_free_input(DATA)
    inp = dataclasses.replace(
        inp, ns_array=np.array([16]), ftol_array=np.array([1.0e-7]),
        niter_array=np.array([2500]),
    )
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-7, max_iterations=2500,
        adjoint_tol=1.0e-8, adjoint_maxiter=100,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )

    def objective(current):
        state, _, _, _ = solve_free_boundary_implicit_status(params, current, cfg)
        return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

    current = field.extcur
    strict_state = solve_free_boundary_implicit(params, current, cfg)
    assert bool(jnp.all(jnp.isfinite(strict_state.R_cos)))
    # Exercise the strict public transform as well; the status lane below is
    # the optimizer-facing path and shares this hot forward state.
    _strict_value, _unused_pullback = jax.vjp(
        lambda value: jnp.mean(solve_free_boundary_implicit(
            params, value, cfg).R_cos[-1] ** 2), current)
    derivative = jax.grad(objective)(current)[0]
    # A re-solve finite difference only measures the derivative while both
    # legs stay on the same root; a step large enough to move one of them
    # measures the root change instead, and does so by orders of magnitude.
    # That is the rule both re-solve anchors in this file teach.  Measured
    # here, adjoint against the central difference at three steps:
    #
    #   step   2e-4        2e-5        2e-6
    #   FD    -4.293e-02  -2.306e-01  -2.366e-01   (adjoint -2.270e-01)
    #
    # The 2e-4 leg lands on a different root and collapses the difference by
    # 5x; 2e-5 and 2e-6 agree with the adjoint to 1.6 % and 4.0 %, and 2e-5 is
    # the least noisy of the usable steps.  The adjoint itself is insensitive
    # to all of this: it moves by 0.1 % across solver revisions that move the
    # 2e-4 difference by 5x.
    step = 2.0e-5
    finite_difference = (
        objective(current + step) - objective(current - step)
    ) / (2.0 * step)
    # The host solve uses adaptive vacuum cadence and stops at finite force
    # tolerance; the independent re-solve FD therefore has percent-level
    # noise on this deliberately coarse nightly case.
    np.testing.assert_allclose(
        derivative, finite_difference, rtol=2.0e-2, atol=2.0e-4
    )


@pytest.mark.full
def test_ncsx_free_boundary_current_gradient_matches_resolve_finite_difference():
    """PF5 coil-current response on the NCSX c09r00 family vs cold re-solves.

    Second-family spot certificate for the coil-current adjoint: the
    CTH-like case above is nfp=5 with a generated single-channel field;
    this is the committed nfp=3 NCSX c09r00 mgrid with ten coil groups
    (``tests/test_ncsx_free_boundary_parity.py`` documents the family).
    PF5 carries the largest boundary response of the ten channels
    (|dJ/dI| ~ 3e-8 per A); central-differencing it with a 300 A step
    keeps the FD signal well above the deterministic solver-endpoint
    floor that dominates the weaker channels.  Measured (Apple Silicon
    CPU): adjoint -3.0101e-8 vs central FD -2.9994e-8, relative
    difference 3.6e-3; 280 s wall for the whole certificate cold with
    compilation (the adjoint dominates), 4-5 s per warm-compiled FD
    re-solve.
    """
    inp = dataclasses.replace(
        VmecInput.from_file(DATA / "input.ncsx_c09r00_free_lowres"),
        ns_array=np.array([15]), ftol_array=np.array([1.0e-9]),
        niter_array=np.array([4000]),
    )
    mgrid_path = DATA / "mgrid_ncsx_c09r00_small.nc"
    if not mgrid_path.exists():
        pytest.skip("mgrid_ncsx_c09r00_small.nc not fetched (tools/fetch_assets.py)")
    data = read_mgrid(mgrid_path)
    field = MgridField.from_mgrid_data(
        data, extcur=np.asarray(inp.extcur, dtype=float)[: data.nextcur])
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=15, ftol=1.0e-9, max_iterations=4000,
        adjoint_tol=1.0e-8, adjoint_maxiter=200,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )

    def objective(current):
        state, _, _, _ = solve_free_boundary_implicit_status(params, current, cfg)
        return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

    current = np.asarray(field.extcur, dtype=float)
    pf5 = 7  # EXTCUR(8), the PF5 ring pair at 3.01e4 A
    derivative = float(jax.grad(objective)(jnp.asarray(current))[pf5])
    step = 300.0
    values = []
    for sign in (1.0, -1.0):
        # Independent cold re-solves prevent continuation history from
        # manufacturing agreement with the implicit derivative.
        fbi._FREE_HOT_CACHE.pop(cfg, None)
        perturbed = current.copy()
        perturbed[pf5] += sign * step
        values.append(float(objective(jnp.asarray(perturbed))))
    finite_difference = (values[0] - values[1]) / (2.0 * step)

    assert abs(finite_difference) > 1.0e-9
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-2)


@pytest.mark.full
def test_free_boundary_pressure_gradient_is_certified_at_one_root():
    """The pressure response, certified without re-solving anything.

    The re-solve finite difference this replaced is not noisy on this deck,
    PROVIDED EVERY LEG POPS ``_FREE_HOT_CACHE`` FIRST, as the retired test
    did.  That protocol matters more than the step size and generalises to
    any finite-difference check against this solver: measured on one head
    with both protocols back to back, independent cold legs give 1.565918e-5,
    1.520154e-5 and 1.520142e-5 at steps 1e-2, 1e-3 and 1e-4 -- five digits
    stable, and an eleven-point scan that departs from its own best-fit line
    by nothing measurable -- while legs that continue warm from each other
    give -7.78e-5, +6.83e-4 and -1.03e-3 on the same head, with a departure
    from that line of 75 times the signal.  Write the pop, or measure noise.

    So this test is not retired for noise.  It is retired because of which
    root its legs converge TO.  Both land on genuinely ftol-converged roots,
    but not the same one that a different forward solve reaches: this deck
    stops at J = 0.2752496 in 95 iterations where the previous revision
    stopped at J = 0.2752520 in 467.  Across that change the difference moves
    by 8.8 % while the adjoint moves by 0.2 %, so the ratio of the two is
    1.066 with one forward solve and 0.858 with this one.  The retired test
    allowed ten per cent, and 0.858 is outside it: the test could not be
    re-tuned, because a tolerance calibrated to one root cannot separate a
    wrong adjoint from a different root, and the adjoint here is right.
    (An intermediate revision that always laddered was worse still: its
    coarse rung put neighbouring parameter points on visibly different roots,
    and its finite difference swung four orders of magnitude and changed
    sign.  That arm is gone, and the sign flip that first raised the alarm
    went with it.)

    What is certified instead, both at one saved root:

    1.  ``dF/dp`` in the pressure direction against a central difference *of
        the residual itself*.  Nothing is solved, so the only error is FD
        truncation, and the gate is 1e-6 relative.
    2.  The assembled gradient against forward-adjoint duality: the adjoint
        contraction against ``<dJ/dz, dz>`` with ``(dF/dz) dz = -(dF/dp) dp``
        from an independent forward solve of the same linear system.  The two
        sides differ only by how far each Krylov solve ran, so the gate is
        the configuration's own ``adjoint_tol`` with the same factor of ten
        that ``_adjoint_acceptance`` uses; measured here at 7.8e-8.

    Together these fix the sign, the scale and the ``presf_ns_scale`` term of
    the pressure entry without depending on where the nonlinear solver stops.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-9]), niter_array=np.array([4000]))
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-9, max_iterations=4000,
        adjoint_tol=1.0e-7, adjoint_maxiter=150,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current), device="cpu")

    (_state, status, _fsq, _ratio), saved = fbi._solve_status_fwd(
        params, field.extcur, cfg)
    assert int(status) == 0
    prm, current, solved, mask, rcon0, zcon0, _ = saved
    frozen = jax.lax.stop_gradient(solved)
    project = im._dof_projector(cfg.implicit, mask)
    z_star = project(solved)
    residual = fbi._projected_residual(cfg, mask)
    direction = dataclasses.replace(
        jax.tree.map(jnp.zeros_like, prm),
        am=jnp.zeros_like(prm.am).at[0].set(prm.am[0]))

    def lane(parameters):
        return residual(z_star, parameters, current, frozen, rcon0, zcon0)

    analytic = jax.jvp(lane, (prm,), (direction,))[1]
    assert float(im._tree_norm(analytic)) > 0.0
    step = 1.0e-4
    shifted = [jax.tree.map(lambda a, b, sign=sign: a + sign * step * b,
                            prm, direction) for sign in (1.0, -1.0)]
    finite = jax.tree.map(
        lambda plus, minus: (plus - minus) / (2.0 * step),
        lane(shifted[0]), lane(shifted[1]))
    assert float(im._tree_norm(jax.tree.map(
        jnp.subtract, analytic, finite))) <= 1.0e-6 * float(
            im._tree_norm(analytic))

    state_bar = jax.grad(
        lambda s: jnp.mean(s.R_cos[-1] ** 2 + s.Z_sin[-1] ** 2))(solved)
    params_bar, _ = fbi._solve_bwd_impl(
        cfg, (prm, current, solved, mask, rcon0, zcon0), state_bar)
    adjoint = float(sum(
        jnp.vdot(left, right) for left, right in zip(
            jax.tree.leaves(params_bar), jax.tree.leaves(direction))))
    assert adjoint > 0.0  # more pressure pushes the boundary outward

    rhs_flat = ravel_pytree(project(state_bar))[0]
    forcing, unravel = ravel_pytree(jax.tree.map(jnp.negative, analytic))
    operator = LinearOperator(
        (forcing.size,) * 2,
        matvec=lambda value: np.asarray(ravel_pytree(jax.jvp(
            lambda z: residual(z, prm, current, frozen, rcon0, zcon0),
            (z_star,), (unravel(jnp.asarray(value, forcing.dtype)),))[1])[0]),
        dtype=np.asarray(forcing).dtype)
    tangent, info = gcrotmk(operator, np.asarray(forcing), rtol=1.0e-9,
                            atol=0.0, m=30, k=5, maxiter=300)
    assert info == 0
    forward = float(np.dot(np.asarray(rhs_flat), tangent))
    np.testing.assert_allclose(
        adjoint, forward, rtol=10.0 * cfg.implicit.adjoint_tol, atol=0.0)


@pytest.mark.full
def test_boundary_schur_adjoint_reproduces_the_coupled_gcrot_gradient():
    """Both adjoint solvers invert the same converged plasma-vacuum Jacobian.

    ``coupled_gcrot`` is matrix-free on the full coupled transpose;
    ``boundary_schur`` eliminates the block-tridiagonal bulk exactly and
    solves only NESTOR's edge response (``(I + U.T A.T^-1 E.T U) mu = ...``).
    Agreement between them on one converged root certifies the radial
    elimination itself; the nightly ``full`` test below anchors both against
    an independent re-solve finite difference.

    Both lanes are certified only to ``10 x adjoint_tol x ||rhs||``, so they
    are compared at the percent level, which a wrong Schur complement (rather
    than a differently converged Krylov solve) would miss by far more.
    """
    # The nightly FD anchor retains mpol=12. This cross-solver identity uses
    # the same DIII-D equilibrium at mpol=10, which preserves the 2% agreement
    # while reducing the measured call from 791 s to 128 s.
    base = lasym_free_input(DATA).change_resolution(
        mpol=10, ntor=0, ntheta=30, nzeta=4)
    inp = dataclasses.replace(
        base, ns_array=np.array([8]),
        ftol_array=np.array([1.0e-8]), niter_array=np.array([2500]))
    field = lasym_free_field()
    params = im.params_from_input(inp)

    def configure(solver):
        return make_free_boundary_config(
            inp, field, ns=8, ftol=1.0e-8, max_iterations=2500,
            adjoint_tol=1.0e-5, adjoint_maxiter=100, adjoint_solver=solver,
            schur_probe_chunk_size=4,
            field_from_parameters=lambda current: dataclasses.replace(
                field, extcur=current),
            device="cpu")

    def gradient(cfg):
        def objective(current):
            state, _, _, _ = solve_free_boundary_implicit_status(
                params, current, cfg)
            return jnp.mean(state.R_cos[-1] ** 2 + state.Z_sin[-1] ** 2)

        return np.asarray(jax.grad(objective)(field.extcur))

    coupled_cfg, schur_cfg = configure("coupled_gcrot"), configure("boundary_schur")
    coupled = gradient(coupled_cfg)
    # Re-enter the same root warm: the comparison is about the adjoint, and a
    # second cold ladder would only add solver noise to it.
    fbi._FREE_HOT_CACHE[schur_cfg] = fbi._FREE_HOT_CACHE[coupled_cfg]
    schur = gradient(schur_cfg)

    assert np.all(np.isfinite(coupled)) and np.max(np.abs(coupled)) > 0.0
    np.testing.assert_allclose(schur, coupled, rtol=2.0e-2, atol=1.0e-8)


@pytest.mark.full
def test_edge_response_is_the_exact_linearization_of_nestor():
    """The dense edge response reproduces NESTOR's derivative and the gradient.

    Three facts on one converged LASYM root, each with its own tolerance: the
    response columns against a central difference of the vacuum pressure
    itself (truncation only, 1e-6); the response-linearized coupled lane
    against the exact coupled Jacobian-vector product (round-off, 1e-10); and
    the edge-response gradient against the certified default at the same
    root, where both lanes solve the same operator to the same tolerance.
    """
    base = lasym_free_input(DATA).change_resolution(
        mpol=10, ntor=0, ntheta=30, nzeta=4)
    inp = dataclasses.replace(
        base, ns_array=np.array([8]),
        ftol_array=np.array([1.0e-8]), niter_array=np.array([2500]))
    field = lasym_free_field()
    params = im.params_from_input(inp)

    def configure(solver):
        return make_free_boundary_config(
            inp, field, ns=8, ftol=1.0e-8, max_iterations=2500,
            adjoint_tol=1.0e-9, adjoint_maxiter=100, adjoint_solver=solver,
            field_from_parameters=lambda current: dataclasses.replace(
                field, extcur=current),
            device="cpu")

    coupled_cfg, response_cfg = configure("coupled_gcrot"), configure("edge_response")
    (_, status, _, _), saved = fbi._solve_status_fwd(
        params, field.extcur, response_cfg)
    assert int(status) == 0
    prm, current, state, mask, rcon0, zcon0, _ = saved
    # Both lanes pull the same cotangent back through the same saved root, so
    # solver history cannot stand in for agreement between the two adjoints.
    state_bar = jax.grad(
        lambda s: jnp.mean(s.R_cos[-1] ** 2 + s.Z_sin[-1] ** 2))(state)
    coupled_gradient, response_gradient = (
        np.asarray(fbi._solve_bwd(cfg, saved[:6], state_bar)[1])
        for cfg in (coupled_cfg, response_cfg))
    assert np.max(np.abs(coupled_gradient)) > 0.0
    np.testing.assert_allclose(
        response_gradient, coupled_gradient, rtol=1.0e-6, atol=0.0)

    value, jacobian, inputs = fbi._edge_response(
        response_cfg, prm, current, state, rcon0, zcon0)
    assert jacobian.shape == value.shape + inputs.shape

    icfg = response_cfg.implicit
    rt = dataclasses.replace(
        im.runtime_from_params(prm, icfg), rcon0=rcon0, zcon0=zcon0,
        lfreeb=True, jmax=int(icfg.resolution.ns))
    _, unflatten = ravel_pytree(fbi._vacuum_inputs(state, rt))
    direction = jnp.asarray(
        np.random.default_rng(0).standard_normal(inputs.size))
    direction *= jnp.linalg.norm(inputs) / jnp.linalg.norm(direction)
    step = 1.0e-6

    def pressure(vector):
        return response_cfg.vacuum_program.bsq_edge(
            *unflatten(vector), response_cfg.field_from_parameters(current))

    finite = (pressure(inputs + step * direction)
              - pressure(inputs - step * direction)) / (2.0 * step)
    np.testing.assert_allclose(
        jnp.tensordot(jacobian, direction, axes=1), finite,
        rtol=0.0, atol=1.0e-6 * float(jnp.linalg.norm(finite)))

    project = im._dof_projector(icfg, mask)
    z_star = project(state)
    tangent = project(jax.tree.map(
        lambda leaf: jnp.asarray(np.random.default_rng(1).standard_normal(
            leaf.shape)) * 1.0e-3, z_star))
    products = [
        jax.jvp(lambda z: lane(z, prm, current, state, rcon0, zcon0),
                (z_star,), (tangent,))[1]
        for lane in (fbi._projected_residual(response_cfg, mask),
                     fbi._projected_residual(
                         response_cfg, mask,
                         response=(value, jacobian, inputs)))]
    assert float(im._tree_norm(jax.tree.map(
        jnp.subtract, *products))) <= 1.0e-10 * float(im._tree_norm(products[0]))


@pytest.mark.full
def test_boundary_schur_current_gradient_matches_resolve_finite_difference():
    """The reduced adjoint retains a nontrivial external-field derivative."""
    base = lasym_free_field()
    zeros = np.zeros_like(base.br)
    field = dataclasses.replace(
        base, br=np.concatenate((zeros, base.br)),
        bp=np.concatenate((base.bp, zeros)),
        bz=np.concatenate((base.bz, zeros)), extcur=np.ones(2))
    inp = dataclasses.replace(
        lasym_free_input(DATA), extcur=np.ones(2), ns_array=np.array([8]),
        ftol_array=np.array([1.0e-8]), niter_array=np.array([6000]))
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=8, ftol=1.0e-8, max_iterations=6000,
        adjoint_tol=1.0e-7, adjoint_maxiter=100,
        adjoint_solver="boundary_schur", schur_probe_chunk_size=4,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current), device="cpu")

    def objective(current):
        state, _, _, _ = solve_free_boundary_implicit_status(
            params, current, cfg)
        return jnp.mean(state.R_cos[-1]**2 + state.Z_sin[-1]**2)

    derivative = jax.grad(objective)(field.extcur)[1]
    step = 1.0e-3
    direction = jnp.array([0.0, 1.0])
    values = []
    for sign in (-1.0, 1.0):
        # Independent cold re-solves avoid continuation-history hysteresis in
        # this deliberately coarse nonlinear free-boundary certificate.
        fbi._FREE_HOT_CACHE.pop(cfg, None)
        values.append(objective(field.extcur + sign * step * direction))
    finite_difference = (values[1] - values[0]) / (2.0 * step)
    np.testing.assert_allclose(
        derivative, finite_difference, rtol=5.0e-2, atol=3.0e-4)


def test_presf_ns_scale_is_differentiated_in_the_adjoint_lanes():
    """``presf_ns_scale`` tracks ``am``; the adjoint lanes may not freeze it.

    ``funct3d.f``'s edge force carries ``presf_ns_scale * pressure[-1]``, and
    the ratio ``pmass(1)/pmass(hs*(ns-1.5))`` is a smooth function of ``am``,
    which is a differentiated parameter.  Taking the host float computed from
    the reference input gives the right value at the reference point and no
    derivative at all -- the same failure mode as the frozen lasym ``delta``
    branch, and invisible to any check that compares the traceable lane with
    itself.  This deck is ``power_series`` with a live derivative; a
    ``two_power`` deck has ``p(1) = 0`` and could not show it.
    """
    from vmex.core.freeboundary import (
        _presf_ns_scale, _presf_ns_scale_traceable,
    )

    inp, ns = lasym_free_input(DATA), 9
    assert inp.pmass_type == "power_series"
    params = im.params_from_input(inp)
    host = _presf_ns_scale(inp, ns)
    np.testing.assert_allclose(
        float(_presf_ns_scale_traceable(params, inp, ns)), host,
        rtol=1e-14, atol=0.0)

    am = np.asarray(inp.am, dtype=float)
    active = int(np.max(np.nonzero(am)[0])) + 1
    grad = jax.grad(lambda p: _presf_ns_scale_traceable(p, inp, ns))(params)
    analytic = np.asarray(grad.am, dtype=float)[:active]

    def shifted(k, delta):
        return dataclasses.replace(
            inp, am=np.where(np.arange(am.size) == k, am + delta, am))

    finite = np.empty(active)
    for k in range(active):
        step = 1.0e-6 * max(abs(am[k]), 1.0)
        finite[k] = (_presf_ns_scale(shifted(k, step), ns)
                     - _presf_ns_scale(shifted(k, -step), ns)) / (2.0 * step)

    # The frozen host float reported exactly zero for all of these, so the
    # tolerance only has to separate a live derivative from a missing one.
    # This deck's am coefficients reach 5e7 against a ratio of 0.32, which
    # caps central differences on the host float at a few times 1e-4.
    assert np.max(np.abs(finite)) > 1e-6, f"probe is degenerate: {finite}"
    np.testing.assert_allclose(analytic, finite, rtol=1e-3, atol=0.0)

    # pres_scale cancels in the ratio, so its derivative is genuinely zero.
    assert float(np.asarray(grad.pres_scale)) == 0.0


def _flat(tree):
    """Flatten a state-like pytree to one vector for inner products."""
    return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(tree)])


@pytest.mark.full
def test_free_boundary_gradient_is_certified_factor_by_factor():
    """A certificate whose tolerances come from arithmetic, not solver noise.

    The re-solve certificates in this module central-difference two cold
    forward solves, so their percent-level gates are set by where the solver
    stops rather than by the adjoint, and a disagreement cannot be attributed:
    an FD floor and a genuinely wrong adjoint look the same.  This test
    factors the implicit gradient at a frozen root,

        dJ/dp = -(dF/dp)^T (dF/dz)^-T (dJ/dz),

    and certifies each factor separately, with no cold re-solve anywhere:

    1.  ``dF/dp`` — the AD Jacobian-vector product against a central
        difference *of the residual itself*.  Nothing is solved, so the only
        error is FD truncation and the gate is 1e-6 relative.
    2.  ``(dF/dz)^T`` — the transpose identity ``<v, J u> == <J^T v, u>`` on
        random tangents, to 1e-11 relative.  This is what makes the
        ``jax.vjp`` operator the adjoint of the forward linearization and not
        merely something with the right shape.
    3.  The adjoint linear solve — by its own transpose residual
        ``||J^T lam - rhs|| / ||rhs||``, against the configured tolerance.
    4.  The assembly, by forward-adjoint duality within the one root:
        ``<dJ/dz, dz>`` where ``(dF/dz) dz = -(dF/dp) dp`` from a *forward*
        linear solve, against ``-<lam, (dF/dp) dp>`` from the adjoint.  These
        are the same number computed through opposite sides of the identity,
        so agreeing certifies the whole assembly without a second forward
        solve anywhere.

    Part 4 deliberately does not compare against ``jax.grad``.  Every call to
    the transformed lane re-enters the forward solve, which warm-starts from
    ``_FREE_HOT_CACHE``, so a second call lands on a *different* root -- 4.7e-4
    apart in relative state norm here, which moves ``dJ/dI`` by 1.9e-3.  Both
    roots have exact adjoints; they are just not the same root, and a test that
    compared across them would be measuring root reproducibility while claiming
    to measure the adjoint.  That amplification is worth its own gate, and has
    one below.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([16]),
        ftol_array=np.array([1.0e-7]), niter_array=np.array([2500]),
    )
    field = lasym_free_field()
    params = im.params_from_input(inp)
    adjoint_tol = 1.0e-10
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-7, max_iterations=2500,
        adjoint_tol=adjoint_tol, adjoint_maxiter=400,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )
    current = jnp.asarray(field.extcur)

    # One root for everything.  jax.grad below must differentiate the same
    # solve the factors are built from, so the host solve runs first and the
    # transformed lane picks it up from the hot cache.
    state, mask, rcon0, zcon0 = fbi._host_solve_and_mask(cfg, params, current)
    state = jax.tree.map(jnp.asarray, state)
    mask = jax.tree.map(jnp.asarray, mask)
    rcon0, zcon0 = jnp.asarray(rcon0), jnp.asarray(zcon0)
    project = im._dof_projector(cfg.implicit, mask)
    residual = fbi._projected_residual(cfg, mask)
    frozen = jax.lax.stop_gradient(state)
    z_star = project(state)

    def force_of_current(p):
        return residual(z_star, params, p, frozen, rcon0, zcon0)

    def force_of_state(z):
        return residual(z, params, current, frozen, rcon0, zcon0)

    # 1. dF/dp against a central difference of the residual.  The step is
    # relative to the coil current, and the residual is smooth in it.
    direction = jnp.zeros_like(current).at[0].set(1.0)
    _, jvp_p = jax.jvp(force_of_current, (current,), (direction,))
    step = 1.0e-3 * float(jnp.abs(current[0]))
    fd_p = jax.tree.map(
        lambda plus, minus: (plus - minus) / (2.0 * step),
        force_of_current(current + step * direction),
        force_of_current(current - step * direction),
    )
    ad_vec, fd_vec = _flat(jvp_p), _flat(fd_p)
    scale = float(jnp.linalg.norm(fd_vec))
    assert scale > 0.0
    assert float(jnp.linalg.norm(ad_vec - fd_vec)) / scale < 1.0e-6

    # 2. Transpose identity on the state linearization.
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    leaves, treedef = jax.tree.flatten(z_star)
    def random_like(key):
        parts = jax.random.split(key, len(leaves))
        return jax.tree.unflatten(treedef, [
            jax.random.normal(k, leaf.shape, leaf.dtype)
            for k, leaf in zip(parts, leaves)])
    u, v = project(random_like(keys[0])), project(random_like(keys[1]))
    _, jvp_z = jax.jvp(force_of_state, (z_star,), (u,))
    _, pullback = jax.vjp(force_of_state, z_star)
    jt_v = pullback(v)[0]
    left = float(jnp.dot(_flat(v), _flat(jvp_z)))
    right = float(jnp.dot(_flat(jt_v), _flat(u)))
    assert abs(left) > 0.0
    assert abs(left - right) / abs(left) < 1.0e-11

    # 3. The adjoint solve, by its own transpose residual.
    state_bar = jax.grad(
        lambda z: jnp.mean(z.R_cos[-1] ** 2 + z.Z_sin[-1] ** 2))(state)
    rhs = project(state_bar)
    lam = im._adjoint_solve_gcrot(
        lambda cotangent: pullback(cotangent)[0], rhs, cfg.implicit)[0]
    solve_residual = jax.tree.map(
        jnp.subtract, pullback(lam)[0], rhs)
    assert (float(jnp.linalg.norm(_flat(solve_residual)))
            / float(jnp.linalg.norm(_flat(rhs)))) < 1.0e3 * adjoint_tol

    # 4. Forward-adjoint duality on the same root.
    _, parameter_pullback = jax.vjp(
        lambda p: residual(z_star, params, p, frozen, rcon0, zcon0), current)
    assembled = parameter_pullback(jax.tree.map(jnp.negative, lam))[0]
    assert float(jnp.linalg.norm(assembled)) > 0.0
    adjoint_side = float(jnp.dot(assembled, direction))

    forward_rhs = jax.tree.map(jnp.negative, jvp_p)      # -(dF/dp) dp
    delta_z = im._adjoint_solve_gcrot(
        lambda tangent: jax.jvp(force_of_state, (z_star,), (tangent,))[1],
        forward_rhs, cfg.implicit)[0]
    forward_side = float(jnp.dot(_flat(rhs), _flat(delta_z)))
    assert abs(adjoint_side) > 0.0
    assert abs(forward_side - adjoint_side) / abs(adjoint_side) < 1.0e-6


@pytest.mark.full
def test_free_boundary_root_is_a_function_of_the_parameters():
    """Two entry points and repeated calls return one root, bit for bit.

    They used to not: both entry points solved from whatever state the last
    call left behind, so the same parameters gave different roots -- measured
    4.7e-4 in relative state norm at ``ftol = 1e-7``, which moved ``dJ/dI`` by
    1.9e-3, a 4x amplification, and on the finite-beta free-boundary deck at
    ns = 31 moved the example's objective by 57 %.  An optimizer cannot line
    search through that.  Every solve now restarts from one converged
    reference per configuration instead of from the previous trial, so the
    returned root depends on the parameters alone.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([16]),
        ftol_array=np.array([1.0e-7]), niter_array=np.array([2500]),
    )
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-7, max_iterations=2500,
        adjoint_tol=1.0e-10, adjoint_maxiter=400,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )
    current = jnp.asarray(field.extcur)

    first, first_status, *_ = solve_free_boundary_implicit_status(
        params, current, cfg)
    assert int(first_status) == 0

    # A distant current trial misses the nonlinear acceptance gate after its
    # deterministic cold retry. It must not change the root returned when the
    # accepted point is evaluated again.
    _, failed_status, *_ = solve_free_boundary_implicit_status(
        params, 2.0 * current, cfg)
    assert int(failed_status) == 2

    repeat, *_ = solve_free_boundary_implicit_status(params, current, cfg)
    host_root, *_ = fbi._host_solve_and_mask(cfg, params, current)
    host_root = jax.tree.map(jnp.asarray, host_root)
    scale = float(jnp.linalg.norm(_flat(first)))
    for label, other in (("repeat", repeat), ("host entry point", host_root)):
        gap = float(jnp.linalg.norm(_flat(
            jax.tree.map(jnp.subtract, first, other))))
        assert gap / scale == 0.0, (label, gap / scale)


@pytest.mark.full
def test_free_boundary_trial_state_is_anchored_on_the_coupled_root():
    """Status 0 means a root of the coupled residual, whatever the history.

    VMEC's ``ftol`` bounds a sum of squares and leaves the state along weakly
    damped directions: on this deck the converged solve carries
    ``|F| = 7.2e-06`` and sits 1.7e-02 from the root in coefficient norm.
    The returned state is Newton-anchored to the residual floor instead, and
    the same current returns the same bits after an unrelated nearby trial
    and after a distant one that fails.
    """
    inp = dataclasses.replace(
        lasym_free_input(DATA), ns_array=np.array([16]),
        ftol_array=np.array([1.0e-7]), niter_array=np.array([2500]),
    )
    field = lasym_free_field()
    params = im.params_from_input(inp)
    cfg = make_free_boundary_config(
        inp, field, ns=16, ftol=1.0e-7, max_iterations=2500,
        field_from_parameters=lambda current: dataclasses.replace(
            field, extcur=current),
    )
    current = np.asarray(field.extcur)

    def trial(value):
        state, mask, rcon0, zcon0, status, _fsq, _ratio = (
            fbi._host_solve_and_mask_status(cfg, params, value))
        return (jax.tree.map(jnp.asarray, state), jax.tree.map(jnp.asarray, mask),
                jnp.asarray(rcon0), jnp.asarray(zcon0), int(status))

    state, mask, rcon0, zcon0, status = trial(current)
    assert status == 0
    report = fbi._FREE_LAST_ANCHOR[cfg]
    assert report.certified and report.steps > 0
    project = im._dof_projector(cfg.implicit, mask)
    residual = fbi._projected_residual(cfg, mask)

    def norm(x):
        return float(im._tree_norm(residual(
            project(x), params, current, x, rcon0, zcon0)))

    host = fbi._FREE_LAST_RESULT[cfg].state
    assert norm(host) > 1.0e-7          # where VMEC stopped: off the root
    assert norm(state) <= cfg.implicit.refine_tol * 1.0e-2
    assert report.displacement > 1.0e-3

    # Same parameters, three histories, one answer, bit for bit.
    for other in (1.01 * current, 2.0 * current):
        trial(other)
        again, _mask, _rcon0, _zcon0, again_status = trial(current)
        assert again_status == 0
        for a, b in zip(jax.tree.leaves(again), jax.tree.leaves(state)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

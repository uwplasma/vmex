"""Derivative certificates for the coupled free-boundary root."""

from __future__ import annotations

import dataclasses
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
from vmex.core.freeboundary_implicit import (
    make_free_boundary_config,
    solve_free_boundary_implicit,
    solve_free_boundary_implicit_status,
)
from vmex.core import freeboundary_implicit as fbi
from vmex.core.errors import AdjointSolveError, VmecJacobianError


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
    with pytest.raises(ValueError, match="'boundary_schur' or 'coupled_gcrot'"):
        make_free_boundary_config(inp, field, adjoint_solver="dense")


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
    with pytest.warns(UserWarning, match="cannot run under jax.jit"):
        jax.jit(lambda bar: fbi._solve_bwd_impl(cfg, saved, bar))(
            jnp.asarray([1., -2.]))


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
    step = 2.0e-4
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

    first, *_ = solve_free_boundary_implicit_status(params, current, cfg)
    repeat, *_ = solve_free_boundary_implicit_status(params, current, cfg)
    host_root, *_ = fbi._host_solve_and_mask(cfg, params, current)
    host_root = jax.tree.map(jnp.asarray, host_root)
    scale = float(jnp.linalg.norm(_flat(first)))
    for label, other in (("repeat", repeat), ("host entry point", host_root)):
        gap = float(jnp.linalg.norm(_flat(
            jax.tree.map(jnp.subtract, first, other))))
        assert gap / scale == 0.0, (label, gap / scale)

"""Zero-crash penalty-path tests for ``optimize.least_squares`` (plan Item
I.2): a mid-campaign trial whose equilibrium solve fails must be penalized
(large finite residual, trust region backs off), never crash.  All four
failure lanes are exercised deterministically by making the host solve fail
on chosen calls: the jac=None ``fun`` body, the exception-free implicit
callback status, the finite differentiated penalty, and the final diagnostic
cold re-solve.  Each campaign must complete with a finite cost and no callback
traceback.
"""

from __future__ import annotations

import ast
import contextlib

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from vmex.core import implicit as im  # noqa: E402
from vmex.core import optimize as opt  # noqa: E402
from vmex.core.errors import VmecJacobianError  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")  # full solves: jitted

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
OBJECTIVE = [(opt.aspect_ratio, 4.0, 1.0)]


@pytest.mark.parametrize("size,expected", [(3, (3, 3)), (150, (100, 20))])
def test_refinement_krylov_budget_independent_of_adjoint(monkeypatch, size, expected):
    """An exhausted adjoint budget must not starve its prerequisite primal."""
    seen = []

    def linear_solve(matvec, rhs, **kwargs):
        seen.append((kwargs["m"], kwargs["k"]))
        np.testing.assert_allclose(matvec(rhs), rhs)
        return SimpleNamespace(x=rhs, iterations=0, residual_norm=0.0)

    monkeypatch.setattr(im, "residual_fn", lambda *_: lambda state, params: state)
    monkeypatch.setattr(im, "_solvax_gcrot", linear_solve)
    state = jax.numpy.ones(size)
    cfg = SimpleNamespace(adjoint_gcrot_m=2, adjoint_gcrot_k=1)
    refined, residual, norm, _, _ = im._refine_step_core.__wrapped__(
        state, state, None, None, None, cfg)
    assert seen == [expected]
    np.testing.assert_array_equal(refined, np.zeros(size))
    np.testing.assert_array_equal(residual, np.zeros(size))
    assert float(norm) == 0.0


def _boom() -> VmecJacobianError:
    return VmecJacobianError(
        "INITIAL JACOBIAN CHANGED SIGN!",
        hint="deterministic stand-in for a self-intersecting trial boundary")


def test_status_callback_builds_safe_mask_before_seed_cache(monkeypatch):
    """Even an unprimed failed trial returns a shape-safe zero mask."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp, ftol=1.0e-10, max_iterations=10)
    params = im.params_from_input(inp)
    params_np = jax.tree.map(np.asarray, params)
    saved = dict(im._MASK_CACHE)
    try:
        im._MASK_CACHE.clear()
        monkeypatch.setattr(
            im,
            "_host_solve_and_mask_impl",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(_boom()),
        )
        state, mask, status, fsq, fsq_ratio = im._host_solve_and_mask_status(cfg, params_np)
        assert int(status) == 1
        assert np.isinf(fsq) and np.isinf(fsq_ratio)
        assert all(np.all(value == 0.0) for value in jax.tree.leaves(mask))
        assert jax.tree.structure(state) == jax.tree.structure(mask)
        monkeypatch.setattr(
            im, "_host_solve_and_mask_impl",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bug")),
        )
        with pytest.raises(RuntimeError, match="bug"):
            im._host_solve_and_mask_status(cfg, params_np)
    finally:
        im._MASK_CACHE.clear()
        im._MASK_CACHE.update(saved)


def test_status_callback_exposes_under_converged_fsq(monkeypatch):
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp, ftol=1.0e-14, max_iterations=1, max_fsq_ratio=1.0e-12)
    params = im.params_from_input(inp)
    runtime = im.runtime_from_params(params, cfg)
    state = im._initial_state(runtime.setup)
    mask = jax.tree.map(jax.numpy.zeros_like, state)
    result = SimpleNamespace(converged=False, fsqr=2.0e-10, fsqz=3.0e-10, fsql=0.0)

    def under_converged(config, params_np):
        im._LAST_SOLVE[config] = (im._params_key(params_np), result)
        return jax.tree.map(np.asarray, state), jax.tree.map(np.asarray, mask)

    monkeypatch.setattr(im, "_host_solve_and_mask_impl", under_converged)
    monkeypatch.setattr(im, "_certify_primal", lambda *_: {
        "derivative_certified": False})
    _, _, status, fsq, fsq_ratio = im._host_solve_and_mask_status(
        cfg, jax.tree.map(np.asarray, params)
    )
    assert int(status) == 2
    assert np.isfinite(fsq) and fsq > 0.0
    np.testing.assert_allclose(fsq_ratio, fsq / cfg.ftol)


def test_fd_lane_penalty_path(monkeypatch, capsys):
    """jac=None: a failed trial solve is penalized and the campaign completes."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    real = opt.solve_equilibrium
    calls = {"n": 0, "failed": 0}

    def flaky(trial, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # first FD probe after the (required-good) seed eval
            calls["failed"] += 1
            raise _boom()
        return real(trial, **kwargs)

    monkeypatch.setattr(opt, "solve_equilibrium", flaky)
    res = opt.least_squares(OBJECTIVE, inp, max_mode=1, max_nfev=4,
                            diff_step=1e-4, verbose=1)
    out = capsys.readouterr().out
    assert calls["failed"] == 1
    assert "Cost" in out
    assert "VmecJacobianError" not in out
    assert "Traceback" not in out
    assert np.isfinite(res.cost)
    assert res.failed_trials == 1
    assert isinstance(res.input, VmecInput)


def test_implicit_lane_fun_penalty_path(monkeypatch, capsys):
    """jac='implicit': every failed trial solve penalizes; campaign completes.

    The poison hits new-parameter-key host solves (trial boundaries) only, so
    the seed evaluation, the ``fun(x0)``/``jac(x0)`` memo hits and the final
    diagnostic re-solve stay healthy while every trust-region trial fails —
    the campaign must ride the penalty residual to a clean finish at ``x0``.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    real = im._host_solve
    calls = {"new": 0, "poisoned": 0}

    def flaky(cfg, params):
        hit = im._LAST_SOLVE.get(cfg)
        if hit is None or hit[0] != im._params_key(params):
            calls["new"] += 1
            if calls["new"] >= 2:  # first new key = the x0 seed solve
                calls["poisoned"] += 1
                raise _boom()
        return real(cfg, params)

    monkeypatch.setattr(im, "_host_solve", flaky)
    res = opt.least_squares(OBJECTIVE, inp, max_mode=1, jac="implicit",
                            max_nfev=4, verbose=1)
    out = capsys.readouterr().out
    assert calls["poisoned"] >= 1
    assert "Cost" in out                 # scipy accepted-iteration table
    assert "VmecJacobianError" not in out
    assert "Traceback" not in out        # no exception crossed pure_callback
    assert np.isfinite(res.cost)
    np.testing.assert_allclose(res.x, opt.pack_boundary(inp, 1))  # stayed at x0


def test_minimize_penalty_path(monkeypatch, capsys):
    """A failed scalarized trial gets the smooth consistent penalty pair."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    real = im._host_solve
    calls = {"new": 0, "poisoned": 0}

    def flaky(cfg, params):
        hit = im._LAST_SOLVE.get(cfg)
        if hit is None or hit[0] != im._params_key(params):
            calls["new"] += 1
            if calls["new"] >= 2:
                calls["poisoned"] += 1
                raise _boom()
        return real(cfg, params)

    monkeypatch.setattr(im, "_host_solve", flaky)
    res = opt.minimize(
        OBJECTIVE, inp, max_mode=1, verbose=1,
        options={"maxiter": 2, "maxls": 3})
    out = capsys.readouterr().out
    assert calls["poisoned"] >= 1
    assert "VmecJacobianError" not in out
    assert "Traceback" not in out
    assert np.isfinite(res.cost)
    assert res.monitor is not None


def test_implicit_lane_status_penalty_and_diagnostic_resolve(monkeypatch, capsys):
    """Failed differentiated trials use a penalty; diagnostics re-solve cold.

    The scipy driver evaluates ``jac`` at exactly the accepted iterate
    ``fun`` just solved (a memo-hit host solve), so poisoning memo-hit
    solves after the first ``jac(x0)`` fails later differentiated trials,
    which follow the status-safe penalty branch. ``solve_equilibrium`` fails
    whenever hot-seeded, forcing the final diagnostic's cold-solve fallback.
    """
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    real = im._host_solve
    calls = {"repeat": 0, "poisoned": 0}

    def flaky(cfg, params):
        hit = im._LAST_SOLVE.get(cfg)
        if hit is not None and hit[0] == im._params_key(params):
            calls["repeat"] += 1
            # repeats 1-2: fun(x0), jac(x0); later repeats are the
            # differentiated evaluations of accepted steps -> fail those.
            if calls["repeat"] >= 3:
                calls["poisoned"] += 1
                raise _boom()
        return real(cfg, params)

    real_solve_eq = opt.solve_equilibrium
    seeded = {"n": 0}

    def flaky_solve_eq(trial, *, initial_state=None, **kwargs):
        if initial_state is not None:  # the hot-seeded diagnostic re-solve
            seeded["n"] += 1
            raise _boom()
        return real_solve_eq(trial, **kwargs)

    monkeypatch.setattr(im, "_host_solve", flaky)
    monkeypatch.setattr(opt, "solve_equilibrium", flaky_solve_eq)
    res = opt.least_squares(OBJECTIVE, inp, max_mode=1, jac="implicit",
                            max_nfev=4, verbose=1)
    out = capsys.readouterr().out
    assert calls["poisoned"] >= 1
    assert "VmecJacobianError" not in out
    assert "Traceback" not in out
    assert seeded["n"] == 1
    assert np.isfinite(res.cost)
    assert res.equilibrium is not None  # cold-solve fallback delivered it
    assert res.equilibrium.result.converged


@pytest.mark.parametrize("norm,fsq,geometry,accepted", [
    (1e-12, 1e-15, True, True),
    (1e-7, 1e-15, True, False),
    (1e-12, 1.0, True, False),
    (1e-12, 1e-15, False, False),
    (np.nan, 1e-15, True, False),
    (1e-12, np.inf, True, False),
    (1e-12, -1.0, True, False),
])
def test_actual_primal_certificate_independent_of_refinement_budget(
    monkeypatch, norm, fsq, geometry, accepted
):
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp, ftol=1e-12, primal_tol=1e-10,
                         refine_tol=np.inf, max_fsq_ratio=10.0)
    monkeypatch.setattr(im, "_primal_measurements",
                        lambda *_: (norm, fsq, geometry))
    certificate = im._certify_primal(cfg, None, np.zeros(1), np.ones(1))
    assert certificate["derivative_certified"] is accepted
    assert certificate["primal_tol"] == 1e-10


@pytest.mark.parametrize("tolerance", [0, -1, np.inf, np.nan])
def test_primal_tolerance_must_remain_finite(tolerance):
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    with pytest.raises(ValueError, match="primal_tol"):
        im.make_config(inp, primal_tol=tolerance)


def test_plain_pullback_rejects_nonroot_even_with_exact_adjoint(monkeypatch):
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp, refine_tol=np.inf)
    params = jax.numpy.ones(1)
    monkeypatch.setattr(im, "_primal_measurements", lambda *_: (
        jax.numpy.asarray(1e-5), jax.numpy.asarray(1e-15),
        jax.numpy.asarray(True)))
    monkeypatch.setattr(im, "_solve_implicit_bwd_impl",
                        lambda *_: (jax.numpy.ones(1),))
    with pytest.raises(RuntimeError, match="certified primal"):
        im._solve_implicit_bwd(cfg, (params, params, params), params)
    traced = jax.jit(lambda p: im._solve_implicit_bwd(
        cfg, (p, p, p), p)[0])(params)
    assert np.isnan(np.asarray(traced)).all()


def test_plain_pullback_preserves_certified_gradient(monkeypatch):
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp)
    params = jax.numpy.ones(1)
    monkeypatch.setattr(im, "_primal_measurements", lambda *_: (
        jax.numpy.asarray(1e-12), jax.numpy.asarray(1e-15),
        jax.numpy.asarray(True)))
    monkeypatch.setattr(im, "_solve_implicit_bwd_impl",
                        lambda *_: (2 * jax.numpy.ones(1),))
    gradient, = im._solve_implicit_bwd(cfg, (params, params, params), params)
    np.testing.assert_array_equal(gradient, [2.0])


def test_primal_measurements_evaluate_returned_assembled_state(monkeypatch):
    cfg = SimpleNamespace()
    monkeypatch.setattr(im, "_dof_projector", lambda *_: lambda x: x)
    monkeypatch.setattr(im, "_edge_mask", lambda *_: None)
    monkeypatch.setattr(im, "runtime_from_params", lambda *_: None)
    monkeypatch.setattr(im, "_assemble", lambda z, *_: z + 2.0)

    def forces(state, _runtime):
        # A fresh force measurement must see assembled returned coefficients.
        np.testing.assert_array_equal(state, [5.0])
        return state, SimpleNamespace(fsqr=1.0, fsqz=2.0, fsql=3.0), SimpleNamespace(
            jacobian_sign_changed=jax.numpy.asarray(False))

    monkeypatch.setattr(im, "evaluate_forces", forces)
    norm, fsq, valid = im._primal_measurements.__wrapped__(
        jax.numpy.asarray([3.0]), None, None, cfg)
    assert float(norm) == 5.0
    assert float(fsq) == 6.0
    assert bool(valid)


@pytest.mark.parametrize("host_converged,actual_norm,expected", [
    (True, 1e-4, 2), (False, 1e-12, 0),
])
def test_status_uses_actual_primal_not_historical_stop(
    monkeypatch, host_converged, actual_norm, expected
):
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp, ftol=1e-12, refine_tol=np.inf)
    params = im.params_from_input(inp)
    state = np.ones(1)
    result = SimpleNamespace(converged=host_converged,
                             fsqr=1e-3, fsqz=0., fsql=0.)

    def solved(config, params_np):
        im._LAST_SOLVE[config] = (im._params_key(params_np), result)
        return state, state

    monkeypatch.setattr(im, "_host_solve_and_mask_impl", solved)
    monkeypatch.setattr(im, "_primal_measurements",
                        lambda *_: (actual_norm, 1e-15, True))
    _, _, status, fsq, _ = im._host_solve_and_mask_status(cfg, params)
    assert int(status) == expected
    assert float(fsq) == 1e-3  # provenance is preserved, not mislabelled
    key, state_key, certificate = im._LAST_PRIMAL_CERTIFICATE[cfg]
    assert key == im._params_key(params)
    assert state_key == im._primal_state_key(state)
    assert certificate["primal_residual_norm"] == actual_norm


@pytest.mark.parametrize("matching", [True, False])
def test_materialization_uses_only_matching_refined_coefficients(matching):
    from dataclasses import dataclass

    @dataclass
    class Result:
        state: object
        fsqr: float

    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp)
    params = im.params_from_input(inp)
    result = Result(state=np.array([1.0]), fsqr=1e-5)
    refined = np.array([2.0])
    key = im._params_key(params) if matching else b"other parameters"
    im._LAST_REFINED[cfg] = (key, refined)
    materialized = opt._result_at_implicit_anchor(result, cfg, params)
    np.testing.assert_array_equal(materialized.state, refined if matching else result.state)
    assert materialized.fsqr == result.fsqr
    np.testing.assert_array_equal(result.state, [1.0])


def test_real_primal_measurements_match_linearized_residual():
    """One small force pass, without a nonlinear solve or an adjoint."""
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp, ns=5, max_iterations=1)
    params = im.params_from_input(inp)
    runtime = im.runtime_from_params(params, cfg)
    state = im._initial_state(runtime.setup)
    mask = im._fixed_boundary_dof_mask(cfg)
    norm, fsq, valid = im._primal_measurements(state, params, mask, cfg)
    project = im._dof_projector(cfg, mask)
    residual = im.residual_fn(cfg, state, mask)(project(state), params)
    np.testing.assert_allclose(norm, im._tree_norm(residual), rtol=1e-12, atol=1e-15)
    assert np.isfinite(float(fsq))
    assert bool(valid)


def test_certified_eager_primal_preserves_typed_linear_failure(monkeypatch):
    from vmex.core.errors import AdjointSolveError

    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp)
    params = jax.numpy.ones(1)
    monkeypatch.setattr(im, "_primal_measurements", lambda *_: (
        jax.numpy.asarray(1e-12), jax.numpy.asarray(1e-15),
        jax.numpy.asarray(True)))
    failure = AdjointSolveError(message="deliberate linear failure")

    def failed_adjoint(_cfg, residuals, _gbar):
        # The eager branch must not silently stage the previous host policy.
        assert not isinstance(residuals[0], jax.core.Tracer)
        raise failure

    monkeypatch.setattr(im, "_solve_implicit_bwd_impl", failed_adjoint)
    with pytest.raises(AdjointSolveError) as caught:
        im._solve_implicit_bwd(cfg, (params, params, params), params)
    assert caught.value is failure


@pytest.mark.parametrize("lane", ["tangent", "pullback"])
def test_batched_response_requires_its_actual_primal(monkeypatch, lane):
    inp = VmecInput.from_file(DATA_DIR / "input.solovev")
    cfg = im.make_config(inp)
    state = jax.numpy.ones(1)
    directions = jax.numpy.asarray([[1.0], [2.0]])
    monkeypatch.setattr(im, "_active_state_fields", lambda *_: ())
    monkeypatch.setattr(im, "_resolved_chunk_sizes", lambda *_: (1, 1))
    monkeypatch.setattr(im, "_dof_projector", lambda *_: lambda x: x)
    monkeypatch.setattr(im, "_edge_mask", lambda *_: None)
    monkeypatch.setattr(im, "runtime_from_params", lambda *_: None)
    monkeypatch.setattr(im, "_assemble", lambda z, *_: z)
    residual = [1e-12]
    monkeypatch.setattr(im, "_primal_measurements", lambda *_: (
        jax.numpy.asarray(residual[0]), jax.numpy.asarray(1e-15),
        jax.numpy.asarray(True)))
    report = im.LinearResponseReport(
        residual_norm=jax.numpy.zeros(2), tolerance=jax.numpy.ones(2),
        iterations=jax.numpy.ones(2, dtype=int), converged=jax.numpy.ones(2, dtype=bool))
    monkeypatch.setattr(im, "_implicit_evolved_tangent_multi_rhs",
                        lambda *_args, **_kwargs: (3 * directions, report))
    monkeypatch.setattr(im, "_implicit_state_pullback_multi_rhs_impl",
                        lambda *_args, **_kwargs: 3 * directions)

    def response(params):
        if lane == "tangent":
            values, status = im.implicit_state_tangent_multi_rhs(
                params, cfg, state, state, directions)
            return values, status.converged
        values = im.implicit_state_pullback_multi_rhs(
            params, cfg, state, state, directions)
        return values, jax.numpy.all(jax.numpy.isfinite(values))

    values, accepted = response(state)
    np.testing.assert_array_equal(values, 3 * directions)
    assert np.asarray(accepted).all()
    residual[0] = 1e-3
    values, accepted = jax.jit(response)(state)
    assert np.isnan(np.asarray(values)).all()
    assert not np.asarray(accepted).any()


@pytest.fixture
def host_response():
    # Exercise the real nested host guards with controlled callback outcomes;
    # equilibrium, field and linear-solver accuracy have independent tests.
    path = Path(opt.__file__)
    owner = next(node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "_least_squares_implicit")
    nodes = [node for node in owner.body if isinstance(node, ast.FunctionDef)
             and node.name in ("certified_trial", "jac_fn", "jacobian_host", "value_and_grad")]
    cfg = type("Config", (), {"ftol": 1e-12, "max_fsq_ratio": 1e6})()
    x = np.array([1.])
    def key(value):
        return np.asarray(value, dtype=float).tobytes()
    cache = SimpleNamespace(_LAST_STATUS_ERROR={}, _LAST_SOLVE={},
        _LAST_PRIMAL_CERTIFICATE={}, _LAST_REFINED={}, _params_key=key, _primal_state_key=key,
        _timed=lambda *args: contextlib.nullcontext())
    control = dict(eligible=True, event=None, phase="reverse", status=0, linear=0, stashes=0)
    holder = dict(nres=2, lin=None, failed_trials=0, derivative_fallbacks=0,
                  last_jac=None, last_jac_key=None)

    def status(value):
        control["status"] += 1
        state = np.array([2.])
        cache._LAST_SOLVE[cfg] = (key(value), SimpleNamespace(
            converged=True, fsqr=1e-15, fsqz=0., fsql=0.))
        cache._LAST_REFINED[cfg] = (key(value), state)
        cache._LAST_PRIMAL_CERTIFICATE[cfg] = (
            key(value), key(state), {"derivative_certified": control["eligible"]})
        return np.array([2.])

    def linear(value, phase):
        control["linear"] += 1
        if phase == "block" and control["phase"] == "gmres":
            raise ValueError("primary failed")
        if control["event"] == "failure":
            raise ValueError("fresh derivative failed")
        if phase == control["phase"]:
            if control["event"] == "remove":
                cache._LAST_PRIMAL_CERTIFICATE.pop(cfg, None)
            elif control["event"] == "change":
                state = np.array([8.])
                cache._LAST_REFINED[cfg] = (key(value), state)
                cache._LAST_PRIMAL_CERTIFICATE[cfg] = (
                    key(value), key(state), {"derivative_certified": True})
        matrix = np.ones((1, 1))
        return matrix if phase == "reverse" else (matrix, matrix, np.zeros(4))

    def stash(*args):
        control["stashes"] += 1

    namespace = dict(np=np, cfg=cfg, x0=x, imp=cache, jax=jax,
        params_of=lambda value: value, _place=lambda value: value,
        FunctionProblem=SimpleNamespace(_key=key), fun=status, rows_jit=status,
        holder=holder, jac_solver="reverse", traceable_scalar=None, warm_start="perturbation",
        reverse_jit=lambda value: linear(value, "reverse"),
        reverse_gradient_jit=lambda value, rows: linear(value, "reverse").T @ rows,
        jac_jit=lambda value: linear(value, "block"),
        gmres_jit=lambda value: linear(value, "gmres"),
        _record_linear_response=lambda *args: None,
        _select_host_jacobian=lambda matrix, summary, **kwargs: (matrix, False),
        _stash_linearization=stash, AdjointSolveError=opt.AdjointSolveError,
        failure_jacobian=lambda value: np.zeros((1, 1)),
        failure_value_and_gradient=lambda value: (99., np.zeros_like(value)))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    status(x)
    return namespace, cache, cfg, control, holder, key


def test_host_primal_admission_refresh_and_same_state_fallback(host_response):
    ns, cache, cfg, control, holder, key = host_response
    x = ns["x0"]
    control["eligible"] = False
    ns["fun"](x)
    calls = control["status"]
    np.testing.assert_array_equal(ns["jac_fn"](x), [[0.]])
    assert control["linear"] == 0 and control["status"] == calls
    control["eligible"] = True
    cache._LAST_PRIMAL_CERTIFICATE.pop(cfg)
    np.testing.assert_array_equal(ns["jac_fn"](x), [[1.]])
    assert control["status"] == calls + 1
    for missing in (False, True):
        if missing:
            cache._LAST_REFINED.pop(cfg)
        else:
            cache._LAST_REFINED[cfg] = (key(x), np.array([7.]))
        calls = control["status"]
        np.testing.assert_array_equal(ns["jac_fn"](x), [[1.]])
        assert control["status"] == calls + 1
    control["event"] = "failure"
    np.testing.assert_array_equal(ns["jac_fn"](x), [[1.]])
    state = np.array([9.])
    cache._LAST_REFINED[cfg] = (key(x), state)
    cache._LAST_PRIMAL_CERTIFICATE[cfg] = (key(x), key(state), {"derivative_certified": True})
    with pytest.raises(ValueError, match="fresh derivative failed"):
        ns["jac_fn"](x)
    control.update(event=None, eligible=True)
    np.testing.assert_array_equal(ns["jac_fn"](x + .1), [[1.]])
    assert holder["last_jac_key"] == (key(x + .1), cache._LAST_PRIMAL_CERTIFICATE[cfg][1])
    value, gradient = ns["value_and_grad"](x + .1)
    assert value == 2.
    np.testing.assert_array_equal(gradient, [2.])
    control["eligible"] = False
    cache._LAST_PRIMAL_CERTIFICATE.pop(cfg)
    calls = control["linear"]
    np.testing.assert_array_equal(ns["jac_fn"](x + .1), [[0.]])
    assert control["linear"] == calls


@pytest.mark.parametrize("lane", ["reverse", "block", "gmres"])
@pytest.mark.parametrize("event", ["remove", "change"])
def test_host_response_drift_preserves_cache_predictor_and_scalar_pair(host_response, lane, event):
    ns, cache, cfg, control, holder, key = host_response
    control.update(phase=lane, event=event)
    ns["jac_solver"] = "reverse" if lane == "reverse" else "block"
    with pytest.raises(opt.AdjointSolveError, match="primal changed"):
        ns["jac_fn"](ns["x0"])
    assert holder["last_jac"] is None and control["stashes"] == 0
    value, gradient = ns["value_and_grad"](ns["x0"])
    assert value == 99.
    np.testing.assert_array_equal(gradient, [0.])
    assert holder["last_jac"] is None and control["stashes"] == 0


@pytest.mark.parametrize("drop_certificate", [False, True])
def test_host_scalar_preserves_unrelated_derivative_error(host_response, drop_certificate):
    ns, cache, cfg, control, holder, key = host_response
    def fail(value):
        if drop_certificate:
            cache._LAST_PRIMAL_CERTIFICATE.pop(cfg)
        raise RuntimeError("unrelated derivative failure")
    ns["jac_fn"] = fail
    ns["reverse_gradient_jit"] = lambda value, rows: fail(value)
    with pytest.raises(RuntimeError, match="unrelated derivative failure"):
        ns["value_and_grad"](ns["x0"])

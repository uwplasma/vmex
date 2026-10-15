"""Persistent discrete forward-family regressions. Authored by rogeriojorge."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmex.core import implicit as im
from vmex.core.errors import VmecConvergenceError
from vmex.core.input import VmecInput
from vmex.core.optimize import VmecProblem, solve_equilibrium

jax.config.update("jax_enable_x64", True)


def _loss(state, runtime):
    # Current-constrained iota is sensitive to the released m=1 family.
    return im.iota_edge(state, runtime)


def _build(seed, **kwargs):
    return VmecProblem.from_loss(
        seed.inp, _loss, max_mode=1, use_ess=False, frozen_reference=seed,
        refine_tol=1e-11, adjoint_tol=1e-9, max_fsq_ratio=1.0, **kwargs)


@pytest.fixture(scope="module")
def family():
    inp = VmecInput.from_file(str(
        Path(__file__).resolve().parents[1] / "examples/data/input.li383_low_res"))
    inp = dataclasses.replace(inp, ftol_array=np.array([1e-13]),
                              niter_array=np.array([6000]))
    seed = solve_equilibrium(inp)
    assert seed.result.converged
    return seed, _build(seed)


def _norm(tree):
    return float(im._tree_norm(tree))


def _complement(cfg, state):
    p = im._dof_projector(cfg, cfg.frozen_reference.mask)
    return jax.tree.map(lambda a, b, e: (a - b) * (1.0 - e),
                        state, p(state), im._edge_mask(cfg))


def test_public_gradient_at_reference_and_displaced_point(family):
    _, problem = family
    cfg = problem.metadata["config"]
    rng = np.random.default_rng(20261015)
    direction = rng.normal(size=problem.x0.size)
    direction /= np.linalg.norm(direction)
    offset = rng.normal(size=direction.size)
    offset *= 2e-4 / np.linalg.norm(offset)
    for x in (problem.x0, problem.x0 + offset):
        value, gradient = problem.value_and_grad(x)
        ad = float(gradient @ direction)
        assert np.isfinite(value) and abs(ad) > 1e-3
        for h in (1e-4, 3e-5, 1e-5):
            fd = (problem.fun(x + h * direction) - problem.fun(x - h * direction)) / (2 * h)
            relative = abs(fd - ad) / max(abs(fd), abs(ad))
            print(f"offset={np.linalg.norm(x-problem.x0):.2e} h={h:.0e} "
                  f"ad={ad:.10e} fd={fd:.10e} rel={relative:.3e}")
            assert relative < 1e-4
        eq = problem.equilibrium_from_x(x)
        assert problem.fun(x) == pytest.approx(float(_loss(eq.state, eq.runtime)), abs=1e-12)
        evaluation = problem.evaluate(x, derivatives=False)
        assert evaluation.success
        assert evaluation.diagnostics["frozen_reference_residual"] <= cfg.refine_tol
        frozen = cfg.frozen_reference
        drift = _norm(jax.tree.map(jnp.subtract,
            _complement(cfg, eq.state), _complement(cfg, frozen.state)))
        assert drift < 1e-14
        params = im.params_from_input(problem.input_from_x(x), device=cfg.device)
        assert im._check_frozen_root(cfg, params, eq.state, frozen.state, frozen.mask) <= cfg.refine_tol
        np.testing.assert_allclose(eq.result.iotaf, im.iota_profile(eq.state, eq.runtime), atol=1e-14)
        np.testing.assert_allclose(eq.result.wb, im.mhd_energy(eq.state, eq.runtime)[0], atol=1e-14)


def test_trial_order_and_fresh_problem_keep_same_reference(family):
    seed, problem = family
    direction = np.zeros_like(problem.x0)
    direction[0] = 1.0
    target = problem.x0 + 1e-4 * direction
    before = problem.fun(target)
    state_before = problem.equilibrium_from_x(target).state
    problem.fun(problem.x0 - 3e-4 * direction)
    after = problem.fun(target)
    assert after == pytest.approx(before, abs=1e-11)
    assert _norm(jax.tree.map(jnp.subtract,
        state_before, problem.equilibrium_from_x(target).state)) < 1e-9
    fresh = _build(seed)
    assert fresh.metadata["config"] is problem.metadata["config"]
    assert fresh.fun(target) == pytest.approx(before, abs=1e-11)


def test_reference_cache_isolation_and_snapshot(family):
    seed, problem = family
    cfg = problem.metadata["config"]
    frozen = cfg.frozen_reference
    snapshot = jax.tree.map(lambda a: np.array(a, copy=True), frozen.state)
    # Deliberately move an antisymmetric m=1 component, which the evolved
    # projector cannot absorb. Refine this new explicit family at capture.
    pos, neg = im._m1_pair_columns(cfg)
    z = np.array(seed.state.Z_sin, copy=True)
    z[2, pos[0]] += 1e-5
    z[2, neg[0]] -= 1e-5
    mutable = dataclasses.replace(seed.state, Z_sin=z)
    reference = dataclasses.replace(seed, state=mutable)
    base = im._canonical_config(dataclasses.replace(cfg, frozen_reference=None))
    other = im._with_frozen_reference(base, reference)
    assert other is not cfg
    assert _norm(jax.tree.map(jnp.subtract,
        _complement(other, other.frozen_reference.state),
        _complement(cfg, frozen.state))) > 1e-6
    saved = np.array(other.frozen_reference.state.Z_sin)
    z[:] = 0.0
    np.testing.assert_array_equal(other.frozen_reference.state.Z_sin, saved)
    params = im.params_from_input(seed.inp, device=cfg.device)
    im._host_solve_and_mask(other, params)
    assert im._LAST_REFINED[other] is not im._LAST_REFINED[cfg]
    for before, after in zip(jax.tree.leaves(snapshot), jax.tree.leaves(frozen.state), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("mismatch", ["resolution", "constraint", "runtime", "orientation", "modes", "static", "unconverged", "nonfinite", "shape", "free_boundary"])
def test_reference_rejects_incompatible_equilibrium(family, mismatch):
    seed, problem = family
    cfg = dataclasses.replace(problem.metadata["config"], frozen_reference=None)
    ref = seed
    if mismatch == "resolution":
        cfg = dataclasses.replace(cfg, resolution=dataclasses.replace(cfg.resolution, ns=17))
    elif mismatch == "constraint":
        cfg = dataclasses.replace(cfg, lconm1=not cfg.lconm1)
    elif mismatch == "runtime":
        ref = dataclasses.replace(seed, runtime=dataclasses.replace(seed.runtime, lforbal=True))
    elif mismatch == "orientation":
        ref = dataclasses.replace(seed, runtime=dataclasses.replace(seed.runtime,
            setup=dataclasses.replace(seed.runtime.setup, signgs=-seed.runtime.setup.signgs)))
    elif mismatch == "shape":
        ref = dataclasses.replace(seed, state=dataclasses.replace(seed.state, R_cos=seed.state.R_cos[:, :-1]))
    elif mismatch == "free_boundary":
        cfg = dataclasses.replace(cfg, inp=dataclasses.replace(seed.inp, lfreeb=True, mgrid_file="unused.mgrid"))
    elif mismatch == "modes":
        ref = dataclasses.replace(seed, result=dataclasses.replace(seed.result, xm=seed.result.xm + 1))
    elif mismatch == "static":
        cfg = dataclasses.replace(cfg, inp=dataclasses.replace(seed.inp, tcon0=seed.inp.tcon0 + 1))
    elif mismatch == "unconverged":
        ref = dataclasses.replace(seed, result=dataclasses.replace(seed.result, converged=False))
    elif mismatch == "nonfinite":
        ref = dataclasses.replace(seed, state=dataclasses.replace(seed.state, Z_sin=seed.state.Z_sin.at[1, 0].set(jnp.nan)))
    with pytest.raises(ValueError, match="frozen_reference"):
        im._with_frozen_reference(cfg, ref)


def test_frozen_family_requires_scalar_implicit_finite_refinement(family):
    seed, _ = family
    for kwargs in ({"derivative_method": "finite_difference"}, {"refine_tol": np.inf}):
        with pytest.raises(ValueError, match="frozen_reference"):
            VmecProblem.from_loss(seed.inp, _loss, frozen_reference=seed, **kwargs)
    with pytest.raises(ValueError, match="scalar loss"):
        VmecProblem.from_tuples(seed.inp, [(_loss, 0., 1.)], frozen_reference=seed)


def test_failed_newton_is_rejected_even_if_host_converged(family, monkeypatch):
    seed, problem = family
    cfg = problem.metadata["config"]
    x = problem.x0.copy()
    x[0] += 7e-5
    params = im.params_from_input(problem.input_from_x(x), device=cfg.device)
    # The host predictor converges, but returning it unchanged after projection
    # leaves a measurable fixed-family residual. Legacy best-effort behavior
    # must not admit it as a certified family root.
    monkeypatch.setattr(im, "_refined_state", lambda _c, _p, state, _m, **_kw: state)
    im._LAST_REFINED.pop(cfg, None)
    state, mask, status, _, _ = im._host_solve_and_mask_status(cfg, params)
    assert im._LAST_SOLVE[cfg][1].converged
    assert int(status) == 1
    assert isinstance(im._LAST_STATUS_ERROR[cfg], VmecConvergenceError)
    with pytest.raises(RuntimeError, match="usable VMEC equilibrium"):
        problem.equilibrium_from_x(x)
    # Ensure failure is exposed as the scalar wall through the public route.
    value, gradient = problem.value_and_grad(x)
    assert value > 1.0 and np.all(np.isfinite(gradient))
    assert problem.fun(x) == pytest.approx(value)
    with pytest.raises(VmecConvergenceError, match="frozen_reference residual"):
        im.solve_implicit(params, cfg)


def test_nonfinite_fixed_family_residual_is_rejected(family, monkeypatch):
    seed, problem = family
    cfg = problem.metadata["config"]
    frozen = cfg.frozen_reference
    params = im.params_from_input(seed.inp, device=cfg.device)
    monkeypatch.setattr(im, "residual_fn", lambda *_a, **_kw: lambda *_:
                        jax.tree.map(lambda a: jnp.full_like(a, jnp.nan), frozen.state))
    with pytest.raises(VmecConvergenceError, match="frozen_reference residual nan"):
        im._check_frozen_root(cfg, params, frozen.state, frozen.state, frozen.mask)


def test_reference_requires_equilibrium_descriptor(family):
    _, problem = family
    cfg = dataclasses.replace(problem.metadata["config"], frozen_reference=None)
    with pytest.raises(TypeError, match="converged Equilibrium"):
        im._with_frozen_reference(cfg, object())


def test_forward_plot_state_uses_same_family(family):
    _, problem = family
    state, runtime = problem.metadata["host_state_runtime"](problem.x0)
    assert float(_loss(state, runtime)) == pytest.approx(problem.fun(problem.x0), abs=1e-12)
    assert problem.evaluate(problem.x0, derivatives=False).success

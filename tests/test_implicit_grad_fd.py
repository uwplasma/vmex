"""Implicit-gradient validations split from ``test_implicit_grad.py``.

They live in their own module only so the pull-request lanes stay inside
their time cap; helpers, fixtures, and the FD cache are shared with the
parent module, whose docstring explains the FD step choices.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from tests import test_implicit_grad as _base
from tests.test_implicit_grad import CASES, DATA_DIR, _fd
from vmex.core import implicit as im
from vmex.core.input import VmecInput
from vmex.core.statephysics import mean_iota

# Shared fixtures register here by name.  Assignment rather than import keeps
# the test parameters that request them from shadowing an imported name.
_release_jax_caches_in_full_matrix = _base._release_jax_caches_in_full_matrix
solovev = _base.solovev


# ---------------------------------------------------------------------------
# 5. one 3D case: li383, d(wb)/d(boundary coefficient) vs FD
# ---------------------------------------------------------------------------


def test_li383_boundary_gradient_vs_fd():
    name = "li383_low_res"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name])
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)

    grad = jax.grad(
        lambda p: im.run(inp, p, ftol=cfg.ftol,
                         max_iterations=cfg.max_iterations).wb)(p0)
    ad = float(np.asarray(grad.rbc)[ntor, 1])
    fd = _fd(name, inp, cfg, p0, "rbc", (ntor, 1), 4e-4)["wb"]
    rel = abs(ad / fd - 1.0)
    print(f"\n[{name}] d(wb)/d(RBC(0,1)) h=4e-4: AD={ad:+.10e} FD={fd:+.10e} "
          f"rel={rel:.2e} (FD noise floor ~3e-5)")
    assert rel <= 2e-4


# ---------------------------------------------------------------------------
# 6. iteration-policy independence + memory sanity (informational)
# ---------------------------------------------------------------------------


def test_gradient_independent_of_iteration_policy(solovev):
    """Only the fixed point defines the derivative; RSS prints are the
    O(1)-memory sanity (backward = a few residual linearizations, no tape)."""
    import resource

    name, inp, cfg, p0, _, _, _ = solovev
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # noqa: E731

    grads = {}
    r0 = rss()
    for cap in (500, 5000):
        grads[cap] = jax.grad(
            lambda p: im.run(inp, p, ftol=cfg.ftol, max_iterations=cap).wb)(p0)
        print(f"\n[{name}] grad(wb) with max_iterations={cap}: "
              f"peak RSS delta so far = {rss() - r0:.0f} MB")

    a = np.asarray(grads[500].rbc)
    b = np.asarray(grads[5000].rbc)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-14)
    assert np.all(np.isfinite(a))


# ---------------------------------------------------------------------------
# 7. solver-sensitive metric (iota_edge, ncurr=1): the implicit adjoint equals
#    the FROZEN-PATH FD.
# ---------------------------------------------------------------------------


def test_iota_edge_gradient_vs_frozen_path_fd():
    """``d(iota_edge)/d(boundary)`` on the 3D ``ncurr=1`` case: the implicit
    adjoint must equal the frozen-path central FD to solver accuracy.  With the
    m=1 family released, a naive re-solve FD was far off (``h=1e-4``):

        RBC(n=-1,m=1):  adjoint -0.77343,  frozen-path FD -0.77343,  naive FD +0.04543  (sign flip)
        RBC(n=+1,m=1):  adjoint -1.26567,  frozen-path FD -1.26567,  naive FD -2.14135  (69% off)
    """
    name = "li383_low_res"
    inp = VmecInput.from_file(str(DATA_DIR / f"input.{name}"))
    cfg = im.make_config(inp, **CASES[name])
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)
    assert int(inp.ncurr) == 1  # derived-iota case where the effect is largest

    grad = jax.grad(
        lambda p: im.run(inp, p, ftol=cfg.ftol,
                         max_iterations=cfg.max_iterations).iota_edge)(p0)
    zero = jax.tree.map(jnp.zeros_like, p0)
    print(f"\n[{name}] d(iota_edge)/d(RBC): implicit adjoint vs frozen-path FD")
    for n in (ntor - 1, ntor + 1):          # m=1, n=-1 (naive-FD sign flip) and n=+1
        ad = float(np.asarray(grad.rbc)[n, 1])
        tangent = dataclasses.replace(zero, rbc=zero.rbc.at[n, 1].set(1.0))
        fd, info = im.frozen_path_directional_fd(
            p0, cfg, im.iota_edge, tangent, h=1e-4)
        res = max(info["newton_res"])
        rel = abs(ad / fd - 1.0) if fd else abs(ad - fd)
        print(f"  RBC(n={n - ntor:+d},m=1): AD={ad:+.9e}  frozen-FD={fd:+.9e}  "
              f"rel={rel:.2e}  (Newton res {res:.0e})")
        assert res < 1e-8, f"n={n - ntor}: frozen solve not converged (res {res:.1e})"
        assert rel <= 3e-4, (
            f"RBC(n={n - ntor},1): adjoint {ad:.5e} vs frozen-path FD {fd:.5e} "
            f"(rel {rel:.2e}) — the implicit gradient must match the frozen path")


def test_li383_resolve_fd_matches_the_implicit_derivative():
    """Cold re-solves 1 % away from the config's first solve match the implicit
    derivative.  The m=1 constrained family has no equation at the root; the
    released family left mean iota 1-3 % off and an interior ``Z_sin``
    objective up to 140x.  Implicit solves hold it on the input's guess."""
    from vmex.core.residuals import hold_m1_family

    inp = VmecInput.from_file(str(DATA_DIR / "input.li383_low_res"))
    cfg = im.make_config(inp, **CASES["li383_low_res"])
    p0 = im.params_from_input(inp)
    ntor, js = int(inp.ntor), cfg.resolution.ns // 2
    zero = jax.tree.map(jnp.zeros_like, p0)
    unit = lambda f, n: dataclasses.replace(zero, **{f: getattr(zero, f).at[ntor + n, 1].set(1.0)})  # noqa: E731
    metrics = {"mean iota": lambda x, p: mean_iota(x, im.runtime_from_params(p, cfg)),
               "interior Z_sin": lambda x, p: jnp.mean(x.Z_sin[js] ** 2)}
    im.solve_implicit(p0, cfg)
    p1 = jax.tree.map(lambda a, d: a + 0.01 * d, p0, unit("rbc", 1))
    x1, rt1 = im.solve_implicit(p1, cfg), im.runtime_from_params(p1, cfg)
    held = hold_m1_family(x1, im._initial_state(rt1.setup), rt1.modes, lasym=False)
    np.testing.assert_allclose(np.asarray(held.Z_sin), np.asarray(x1.Z_sin), rtol=0, atol=1e-15)
    grads = {k: jax.grad(lambda p, f=f: f(im.solve_implicit(p, cfg), p))(p1) for k, f in metrics.items()}
    h = 5e-4
    for field, n in (("rbc", 1), ("zbs", -1)):
        t, legs = unit(field, n), []
        for sign in (1.0, -1.0):
            for cache in (im._HOT_CACHE, im._LAST_REFINED, im._LAST_SOLVE, im._LAST_REFINEMENT_CORRECTION):
                cache.clear()
            p = jax.tree.map(lambda a, d, s=sign: a + s * h * d, p1, t)
            legs.append({k: float(f(im.solve_implicit(p, cfg), p)) for k, f in metrics.items()})
        for k in metrics:
            implicit = float(sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(grads[k]), jax.tree.leaves(t))))
            resolve = (legs[0][k] - legs[1][k]) / (2 * h)
            print(f"\nd({k})/d({field}(n={n:+d},m=1)): implicit {implicit:+.6e} re-solve {resolve:+.6e}")
            assert implicit == pytest.approx(resolve, rel=1e-3)


# ---------------------------------------------------------------------------
# 8. typed errors through pure_callback (zero-crash policy)
# ---------------------------------------------------------------------------


def test_typed_error_through_pure_callback():
    """A failing host solve raises the SHORT typed exception, not callback
    noise: the ``_HOST_ERROR`` relay stashes the typed exception in the host
    callback and re-raises it at the ``pure_callback`` site with ``from
    None`` (previously a ~3.7 KB raw ``JaxRuntimeError``)."""
    from vmex.core.errors import VmecConvergenceError

    inp = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    with pytest.raises(VmecConvergenceError) as excinfo:
        im.run(inp, ftol=1e-14, max_iterations=3)
    exc = excinfo.value
    assert len(str(exc)) < 200, f"typed message must stay short: {len(str(exc))} chars"
    assert "MORE ITERATIONS REQUIRED" in str(exc)
    assert exc.__cause__ is None and exc.__suppress_context__  # noise killed
    assert exc.ftol == 1e-14  # diagnostics preserved

    # the custom-vjp forward rule call site relays the typed exception too
    p0 = im.params_from_input(inp)
    with pytest.raises(VmecConvergenceError):
        jax.grad(lambda p: im.run(inp, p, ftol=1e-14, max_iterations=3).wb)(p0)


# ---------------------------------------------------------------------------
# 9. multigrid implicit gradient vs frozen-path FD
# ---------------------------------------------------------------------------


def test_multigrid_gradient_vs_frozen_path_fd():
    """``im.run(multigrid=True)`` through a genuine ns 5 -> 11 solovev ladder:
    the adjoint must match the frozen-path central FD of ``wb`` along
    ``RBC(0,1)`` (only the final fixed point defines the derivative)."""
    inp0 = VmecInput.from_file(str(DATA_DIR / "input.solovev"))
    inp = dataclasses.replace(
        inp0,
        ns_array=np.array([5, 11]),
        ftol_array=np.array([1e-10, 1e-14]),
        niter_array=np.array([1000, 2000]),
    )
    cfg = im.make_config(inp, multigrid=True, ftol=1e-14)
    assert cfg.multigrid and int(cfg.resolution.ns) == 11
    p0 = im.params_from_input(inp)
    ntor = int(inp.ntor)

    grad = jax.grad(
        lambda p: im.run(inp, p, multigrid=True, ftol=1e-14).wb)(p0)
    ad = float(np.asarray(grad.rbc)[ntor, 1])

    zero = jax.tree.map(jnp.zeros_like, p0)
    tangent = dataclasses.replace(zero, rbc=zero.rbc.at[ntor, 1].set(1.0))
    fd, info = im.frozen_path_directional_fd(p0, cfg, lambda s, rt:
                                             im.mhd_energy(s, rt)[0],
                                             tangent, h=3e-5)
    res = max(info["newton_res"])
    rel = abs(ad / fd - 1.0)
    print(f"\n[solovev multigrid ns=5->11] d(wb)/d(RBC(0,1)) h=3e-5: "
          f"AD={ad:+.10e}  frozen-FD={fd:+.10e}  rel={rel:.2e} "
          f"(Newton res {res:.0e})")
    assert res < 1e-8, f"frozen solve not converged (res {res:.1e})"
    assert rel <= 1e-6

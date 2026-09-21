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

# Shared fixtures register here by name.  Assignment rather than import keeps
# the test parameters that request them from shadowing an imported name.
_jit_enabled = _base._jit_enabled
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


def test_solovev_directional_taylor_with_independent_resolves(solovev):
    """A physical observable has a second-order one-sided Taylor remainder."""
    name, inp, cfg, p0, _, _, _ = solovev
    ntor = int(inp.ntor)
    gradient = jax.grad(
        lambda p: im.run(
            inp, p, ftol=cfg.ftol, max_iterations=cfg.max_iterations
        ).wb
    )(p0)
    directional = float(np.asarray(gradient.rbc)[ntor, 1])
    base = _base._outputs(name, inp, cfg, p0)["wb"]
    remainders = []
    for step in (1.2e-4, 6.0e-5, 3.0e-5):
        perturbed = _base._perturb(p0, "rbc", (ntor, 1), step)
        value = _base._outputs(name, inp, cfg, perturbed)["wb"]
        remainders.append(abs(value - base - step * directional))
    orders = np.log2(np.asarray(remainders[:-1]) / remainders[1:])
    assert np.all(np.isfinite(remainders))
    assert np.min(orders) > 1.7


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
#    the FROZEN-PATH FD; a naive full re-solve FD is NOT a valid reference.
# ---------------------------------------------------------------------------


def test_iota_edge_gradient_vs_frozen_path_fd():
    """``d(iota_edge)/d(boundary)`` on the 3D ``ncurr=1`` case: the implicit
    adjoint must equal the frozen-path central FD to solver accuracy.  A naive
    re-solve FD is deliberately NOT the reference (measured, ``h=1e-4``):

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


# ---------------------------------------------------------------------------
# 8. typed errors through pure_callback (zero-crash policy, plan Item I.1)
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
# 9. multigrid implicit gradient vs frozen-path FD (plan Item I.4)
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

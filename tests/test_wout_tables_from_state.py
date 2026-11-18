"""``wout_tables_from_state``: traceable wout tables vs the solver's wout, and their gradient."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vmex.core import implicit as im  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402
from vmex.core.optimize import solve_equilibrium  # noqa: E402
from vmex.core.statephysics import wout_tables_from_state  # noqa: E402

DECK = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.minimal_seed_nfp2"
# Measured relative differences (max |diff| / max |wout|); the jxbforce band limit is not applied.
RTOL = {"bmnc": 1e-12, "gmnc": 1e-12, "rmnc": 1e-12, "zmns": 1e-12, "bsupumnc": 1e-12,
        "bsupvmnc": 1e-12, "bsubvmnc": 1e-5, "bsubumnc": 2e-3, "bsubsmns": 2e-3}


def _qa_input() -> VmecInput:
    inp = VmecInput.from_file(DECK)
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.05, 0.05
    return dataclasses.replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
        mpol=5, ntor=5, ntheta=16, nzeta=14)


def test_tables_match_wout_and_gradient_is_finite():
    inp = _qa_input()
    eq = solve_equilibrium(inp)
    tables, wout = wout_tables_from_state(eq.state, eq.runtime), eq.wout
    for name, rtol in RTOL.items():
        got, ref = np.asarray(tables[name]), np.asarray(getattr(wout, name))
        assert got.shape == ref.shape, name
        assert np.abs(got - ref).max() <= rtol * np.abs(ref).max(), name
    for name in ("xm", "xn", "xm_nyq", "xn_nyq"):
        np.testing.assert_array_equal(tables[name], np.asarray(getattr(wout, name)))

    cfg = im.make_config(inp)
    p0 = im.params_from_input(inp)

    def loss(p):
        t = wout_tables_from_state(im.solve_implicit(p, cfg), im.runtime_from_params(p, cfg))
        return sum(jnp.sum(t[k] ** 2) for k in RTOL)

    grads = jax.tree.leaves(jax.grad(loss)(p0))
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grads)
    assert any(np.abs(np.asarray(g)).max() > 0 for g in grads)

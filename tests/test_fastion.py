"""Boozer-form fast-ion proxies: adapter contract and the gradient through the implicit solve."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

pytest.importorskip("neo_jax.fastion")

from neo_jax.fastion import FastIonBudget  # noqa: E402

from vmex.core import fastion  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
from vmex.core import optimize as opt  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
FAST = FastIonBudget(n_lines=8, n_transits=2, nodes_per_period=16, n_panel=16, n_level=24, n_well=24, n_dense=48)


@pytest.fixture(scope="module")
def li383():
    eq = opt.solve_equilibrium(VmecInput.from_file(DATA / "input.li383_low_res"))
    assert eq.result.converged
    return eq


def test_adapter_values_ordering_and_bounds(li383):
    """Tokamak: both proxies vanish; 3D: 0 <= Gamma_alpha <= f_trapped, Gamma_c > 0."""
    tok = opt.solve_equilibrium(VmecInput.from_file(DATA / "input.circular_tokamak"))
    out = fastion.fast_ion_from_state(tok.state, tok.runtime, (0.5,), budget=FAST)
    assert float(out["gamma_c"][0]) < 1e-6 and float(out["gamma_alpha"][0]) < 1e-6
    out = fastion.fast_ion_from_state(li383.state, li383.runtime, (0.4, 0.6), budget=FAST)
    assert np.all(np.asarray(out["gamma_c"]) > 1e-3)
    assert np.all(np.asarray(out["gamma_alpha"]) <= np.asarray(out["f_trapped"]) + 1e-9)


@pytest.mark.parametrize("proxy, shift, offset, steps",
                         [("gamma_c", 0.0, 2, (1e-4, 3e-5)), ("gamma_alpha", 0.01, 1, (3e-4, 1e-4))])
def test_gradient_through_the_solve_matches_converged_finite_differences(proxy, shift, offset, steps):
    """Reverse mode vs central differences of the re-solved value on a precise QA (ns = 31).

    Gamma_c, ``rbc[ntor + 1, 2]``: -1.4508 against FD -1.4714 (step 1e-4) and -1.4592 (3e-5); coefficients
    (ntor, 1) and (ntor, 2) agree to 1e-3 relative.  ``rbc[ntor + 1, 1]`` (AD 61.2, FD 66.8 at 3e-5 and 16 at
    1e-4) varies on a scale of 1e-4 and is not asserted: at the exact optimum both proxies are proportional to
    the quasisymmetry error and have a cusp.  Gamma_alpha is taken 0.01 away in ``rbc[ntor + 1, 1]`` and tested
    there: AD 5.2418 against FD 5.2412 (3e-4) and 5.2405 (1e-4).
    """
    inp = VmecInput.from_file(DATA / "input.nfp2_QA")
    params = im.params_from_input(inp, device=None)
    params = dataclasses.replace(params, rbc=jnp.asarray(params.rbc).at[(int(inp.ntor) + 1, 1)].add(shift))
    index = (int(inp.ntor) + 1, offset)
    budget = FastIonBudget(n_lines=16, n_transits=3, nodes_per_period=16, n_panel=32, n_level=32, n_dense=64)

    def objective(p):
        sol = im.run(inp, p, ns=31, ftol=1e-14, max_iterations=20000, adjoint_tol=1e-13, device=None)
        return fastion.fast_ion_from_state(sol.state, sol.runtime, (0.5,), budget=budget)[proxy][0]

    ad = float(np.asarray(jax.grad(objective)(params).rbc)[index])

    def central(step):
        v = [float(objective(dataclasses.replace(params, rbc=jnp.asarray(params.rbc).at[index].add(s * step))))
             for s in (-1.0, 1.0)]
        return (v[1] - v[0]) / (2 * step)

    fd = [central(step) for step in steps]
    assert abs(fd[0] / fd[1] - 1) < 0.03
    assert abs(ad / fd[1] - 1) < 0.03


def test_class_contract_and_validation(li383):
    term = fastion.GammaC([0.4, 0.6], weights=[1.0, 4.0], budget=FAST)
    rows = np.asarray(term(li383))
    assert rows.shape == (2,) and np.all(rows > 0)
    assert float(term.total(li383)) == pytest.approx(rows @ rows)
    assert fastion.GammaAlpha([0.5], budget=FAST).key == "gamma_alpha"
    with pytest.raises(ValueError, match="increasing"):
        fastion.GammaC([0.7, 0.3])
    with pytest.raises(ValueError, match="weights"):
        fastion.GammaC([0.3, 0.7], weights=[1.0])
    with pytest.raises(ValueError, match="strictly inside"):
        fastion._rows(np.linspace(0, 1, 10), [0.0])

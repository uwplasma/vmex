"""Forward derivative of the Boozer spectrum through the implicit equilibrium solve.

``solve_implicit`` carries a custom VJP, so ``jax.jvp`` cannot enter it. The
forward tangent of the solved state is the implicit-function-theorem response
``dz = -(dF/dz)^{-1} (dF/dp) t`` (``implicit_state_tangent_multi_rhs``), and one
``jax.jvp`` carries ``(dz, t)`` through the Boozer transform of Sanchez et al.
(2000) in ``boozer_spectrum_state(geometry=True)``. The oracle is the central
difference of re-solved states; its best step from 1e-3 to 1e-6 must agree to
1e-6 of the largest harmonic of each family, on the first, a middle and the last
half-mesh surface.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from vmex.core import implicit as im
from vmex.core.input import VmecInput
from vmex.core.omnigenity import boozer_spectrum_state

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
NS = 5
DECKS = {
    "symmetric": ("input.LandremanPaul2021_QA_lowres", 1),
    "pressure": ("input.li383_low_res", 1),
    "asymmetric": ("input.up_down_asymmetric_tokamak", 0),
}


@pytest.mark.full  # 20 to 35 s per deck on a laptop
@pytest.mark.parametrize("deck", ["symmetric", "pressure", "asymmetric"])
def test_boozer_tangent_matches_resolved_central_differences(deck):
    name, ntor = DECKS[deck]
    inp = VmecInput.from_file(str(DATA_DIR / name))
    inp = dataclasses.replace(
        inp.change_resolution(mpol=3, ntor=ntor, ntheta=12, nzeta=4 * ntor + 4),
        ns_array=np.asarray([NS]),
        ftol_array=np.asarray([1e-14]),
        niter_array=np.asarray([20000]),
    )
    cfg = im.make_config(inp, ftol=1e-14, max_iterations=20000)
    p0 = im.params_from_input(inp)
    zero = jax.tree.map(jnp.zeros_like, p0)
    direction = dataclasses.replace(zero, rbc=zero.rbc.at[ntor, 1].set(1.0))  # boundary R(m=1, n=0)
    surfaces = [(j - 0.5) / (NS - 1) for j in (1, NS // 2, NS - 1)]
    families = ["bmnc", "rmnc", "zmns", "pmns", "gmnc"]
    families += ["bmns", "rmns", "zmnc", "pmnc", "gmns"] if inp.lasym else []

    def boozer(state, params):
        out = boozer_spectrum_state(
            state,
            im.runtime_from_params(params, cfg),
            surfaces=surfaces,
            mboz=4,
            nboz=4 if ntor else 0,
            oversample=1,
            geometry=True,
        )
        return {key: out[key + "_b"] for key in families}

    state, mask = im.solve_implicit_with_aux(p0, cfg)
    dz, report = im.implicit_state_tangent_multi_rhs(p0, cfg, state, mask, jax.tree.map(lambda x: x[None], direction))
    assert np.all(np.asarray(report.converged))
    _, tangent = jax.jvp(boozer, (state, p0), (jax.tree.map(lambda x: x[0], dz), direction))

    best = {key: np.inf for key in families}
    for step in (1e-3, 1e-4, 1e-5, 1e-6):
        plus, minus = (
            boozer(im.solve_implicit(p, cfg), p)
            for p in (jax.tree.map(lambda a, b, s=sign: a + s * step * b, p0, direction) for sign in (1.0, -1.0))
        )
        for key in families:
            ad = np.asarray(tangent[key])
            fd = (np.asarray(plus[key]) - np.asarray(minus[key])) / (2.0 * step)
            best[key] = np.minimum(best[key], np.max(np.abs(ad - fd), axis=1) / np.max(np.abs(ad), axis=1))
    assert max(float(np.max(error)) for error in best.values()) < 1e-6, best

"""R1 convergence protection: from a circular-torus seed, the QS optimization
building blocks reach a real, non-trivial residual reduction (not just the
``VMEC_JAX_EXAMPLES_CI`` smoke budget of ``test_examples.py``).

These are ``full``-marked (nightly only, ``RUN_FULL=1``): each runs *real*
implicit-gradient continuation (``jac="implicit"`` + ESS, the exact path the
``examples/optimization`` scripts use) and asserts the achieved
``QuasisymmetryRatioResidual.total`` bound.  QA runs two stages to its precise
bound (< 1e-3); QH/QP run a single stage (they are compile-bound at higher
``max_mode`` -- ~13 min per stage -- so the nightly test guards the
single-stage bound, and the deeper precision is recorded in each docstring).

Measured on office (2x A4000 / origin/main f45a6491), implicit ``max_mode=1``:
QA (with the helical seed kick) 2.04e-01 -> 9.82e-03; QH 6.91e-01 -> 1.40e-01;
QP 4.46e-01 -> descends.  Bounds below carry margin over those.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")

from vmec_jax.core.input import VmecInput
from vmec_jax.core import optimize as opt

DATA = __import__("pathlib").Path(__file__).resolve().parents[2] / "examples" / "data"


def _nfp2_seed(kick: float = 0.0) -> VmecInput:
    inp = VmecInput.from_file(DATA / "input.minimal_seed_nfp2")
    if kick:
        rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
        rbc[inp.ntor + 1, 1] += kick
        zbs[inp.ntor + 1, 1] += kick
        inp = dataclasses.replace(inp, rbc=rbc, zbs=zbs)
    return inp


def _qh_seed() -> VmecInput:
    ntor = mpol = 6  # one harmonic above max_mode 5
    a = 0.125
    rbc = np.zeros((2 * ntor + 1, mpol))
    zbs = np.zeros((2 * ntor + 1, mpol))
    rbc[ntor, 0] = 1.0
    rbc[ntor, 1] = a
    zbs[ntor, 1] = a
    return VmecInput(nfp=4, mpol=mpol, ntor=ntor, rbc=rbc, zbs=zbs,
                     phiedge=np.pi * a ** 2, lasym=False, lfreeb=False,
                     mgrid_file="NONE", ncurr=1, curtor=0.0, pres_scale=0.0,
                     ns_array=[35], ftol_array=[1e-13], niter_array=[3000], delt=0.9)


@pytest.mark.full
def test_qa_reaches_precise():
    """QA (nfp2, helicity (1,0)) reaches *precise* QS via implicit continuation.

    Measured (office A4000): 2.043e-01 -> 9.82e-03 (max_mode=1) -> 1.70e-04
    (max_mode=2).  This two-stage nightly run protects the headline precise-QA
    claim; the bound (< 1e-3) carries margin over the measured 1.7e-4.
    """
    inp = _nfp2_seed(kick=0.01)  # helical kick breaks the axisymmetric saddle
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, 0)
    seed = float(qs.total(opt.solve_equilibrium(inp)))
    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0), (opt.mean_iota, 0.42, 10.0)]
    r = opt.least_squares(terms, inp, max_mode=(1, 2), jac="implicit", use_ess=True,
                          max_nfev=80, ftol=1e-9, xtol=1e-10)
    final = float(qs.total(r.equilibrium))
    assert final < 1e-3, f"QA QS {seed:.3e} -> {final:.3e} (expected precise < 1e-3)"
    assert abs(float(opt.aspect_ratio(r.equilibrium.state, r.equilibrium.runtime)) - 6.0) < 0.05


@pytest.mark.full
def test_qh_implicit_converges():
    """QH (nfp4, helicity (1,-1)) descends from the axisymmetric seed via implicit.

    Note: implicit escapes the exact-axisymmetric saddle where finite
    differences stall (the QS residual is even in the symmetry-breaking
    harmonic, so the FD gradient vanishes) — no seed kick needed.
    """
    inp = _qh_seed()
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, -1)
    seed = float(qs.total(opt.solve_equilibrium(inp)))
    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 8.0, 1.0)]
    r = opt.least_squares(terms, inp, max_mode=1, jac="implicit", use_ess=True,
                          max_nfev=60, ftol=1e-9, xtol=1e-10)
    final = float(qs.total(r.equilibrium))
    assert final < 0.30, f"QH QS {seed:.3e} -> {final:.3e} (expected < 0.30)"


@pytest.mark.full
def test_qp_implicit_descends():
    """QP (nfp2, helicity (0,1)) descends via implicit (basin-limited target)."""
    inp = _nfp2_seed()
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 0, 1)
    seed = float(qs.total(opt.solve_equilibrium(inp)))

    def iota_shortfall(state, rt):
        import jax.numpy as jnp
        return jnp.maximum(0.15 - jnp.abs(opt.mean_iota(state, rt)), 0.0)

    terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, 6.0, 1.0),
             (iota_shortfall, 0.0, 100.0), (opt.mirror_ratio, 0.20, 10.0)]
    r = opt.least_squares(terms, inp, max_mode=1, jac="implicit", use_ess=True,
                          max_nfev=60, ftol=1e-9, xtol=1e-10)
    final = float(qs.total(r.equilibrium))
    assert final < 0.9 * seed, f"QP QS {seed:.3e} -> {final:.3e} (expected descent)"

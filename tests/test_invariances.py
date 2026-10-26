"""Exact-identity property tests on the ``cth_like_fixed_bdy`` deck (~15 s).

Each transformation of the input maps the VMEC equilibrium onto a known
image of itself, so the converged scalars (and the iteration count) must agree
with the untransformed run to solver round-off:

- mirror image ``Z -> -Z`` (flips the coordinate handedness, ``signgs``);
- ``nfp=5`` vs ``nfp=1`` for an axisymmetric deck run through the 3-D path;
- uniform length rescaling ``R, Z -> L R, L Z`` with ``phiedge -> L**2 phiedge``
  and ``curtor -> L curtor`` (B unchanged): ``V ~ L**3``, ``wb ~ L**3``;
- coarse-to-fine continuation vs the direct solve at the same final ``ns``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import vmex as vj  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.fixture(scope="module")
def base():
    inp = vj.VmecInput.from_file(DATA / "input.cth_like_fixed_bdy")
    return inp, _solve(inp)


def _solve(inp):
    r = vj.solve_multigrid(inp, verbose=False)
    w = vj.wout_from_state(
        inp=inp, state=r.state, fsqr=float(r.fsqr), fsqz=float(r.fsqz),
        fsql=float(r.fsql), niter=int(r.iterations), converged=bool(r.converged))
    return dict(wb=float(w.wb), vol=float(w.volume_p), aspect=float(w.aspect),
                iota=np.asarray(w.iotaf), iters=int(r.iterations))


def _same_shape(got, ref, iters=True, rtol=1e-11):
    for k in ("wb", "vol", "aspect"):
        assert got[k] == pytest.approx(ref[k], rel=rtol), k
    np.testing.assert_allclose(got["iota"], ref["iota"], rtol=1e-9)
    if iters:
        assert got["iters"] == ref["iters"]


def test_mirror_image_z_flip(base):
    inp, ref = base
    flip = replace(inp, zbs=-inp.zbs, zaxis_s=-inp.zaxis_s)
    _same_shape(_solve(flip), ref)


def test_nfp1_equals_whole_torus(base):
    """Axisymmetric deck padded to ntor=1: nfp=5 and nfp=1 are the same torus."""
    inp, ref = base
    pad = lambda a: np.vstack([np.zeros_like(a), a, np.zeros_like(a)])  # noqa: E731
    ext = lambda a: np.r_[a, 0.0]  # noqa: E731
    d3 = replace(
        inp, ntor=1, rbc=pad(inp.rbc), zbs=pad(inp.zbs), rbs=pad(inp.rbs),
        zbc=pad(inp.zbc), raxis_c=ext(inp.raxis_c), zaxis_s=ext(inp.zaxis_s),
        raxis_s=ext(inp.raxis_s), zaxis_c=ext(inp.zaxis_c))
    for nfp in (inp.nfp, 1):
        _same_shape(_solve(replace(d3, nfp=nfp)), ref)


def test_length_rescaling(base):
    inp, ref = base
    s = 2.0
    big = replace(
        inp, rbc=s * inp.rbc, zbs=s * inp.zbs, raxis_c=s * inp.raxis_c,
        zaxis_s=s * inp.zaxis_s, phiedge=s**2 * inp.phiedge,
        curtor=s * inp.curtor)
    got = _solve(big)
    assert got["wb"] == pytest.approx(s**3 * ref["wb"], rel=1e-11)
    assert got["vol"] == pytest.approx(s**3 * ref["vol"], rel=1e-11)
    assert got["aspect"] == pytest.approx(ref["aspect"], rel=1e-11)
    np.testing.assert_allclose(got["iota"], ref["iota"], rtol=1e-9)
    assert got["iters"] == ref["iters"]


def test_continuation_matches_direct_solve(base):
    """A coarse-to-fine ladder converges to the direct ns=15 equilibrium."""
    inp, ref = base
    lad = replace(inp, ns_array=[7, 15], ftol_array=[1e-10, 1e-14],
                  niter_array=[2000, 2000])
    got = _solve(lad)
    assert got["wb"] == pytest.approx(ref["wb"], rel=1e-9)
    assert got["vol"] == pytest.approx(ref["vol"], rel=1e-9)
    np.testing.assert_allclose(got["iota"], ref["iota"], atol=1e-5)

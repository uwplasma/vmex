from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax import preconditioner_1d as p1d
from vmec_jax.vmec_tomnsp import TomnspsRZL


def _cfg(*, mpol=2, ntor=1, ntheta=4, nzeta=2, nfp=1, lasym=False, lthreed=False):
    return SimpleNamespace(mpol=mpol, ntor=ntor, ntheta=ntheta, nzeta=nzeta, nfp=nfp, lasym=lasym, lthreed=lthreed)


def _bc(*, ns=3, ntheta=3, nzeta=2):
    shape = (ns, ntheta, nzeta)
    radial = np.arange(ns, dtype=float)[:, None, None]
    theta = np.arange(ntheta, dtype=float)[None, :, None]
    zeta = np.arange(nzeta, dtype=float)[None, None, :]
    return SimpleNamespace(
        guu=2.0 + 0.2 * radial + 0.03 * theta,
        guv=0.4 + 0.1 * radial + 0.02 * zeta,
        gvv=3.0 + 0.3 * radial + 0.01 * theta,
        jac=SimpleNamespace(
            sqrtg=1.0 + 0.05 * radial + 0.02 * theta + 0.01 * zeta,
            r12=1.0 + 0.1 * radial + 0.01 * theta + np.zeros(shape),
            tau=0.2 + 0.02 * radial + np.zeros(shape),
            zs=0.3 + 0.01 * radial + np.zeros(shape),
            zu12=0.4 + 0.01 * theta + np.zeros(shape),
            rs=0.5 + 0.01 * radial + np.zeros(shape),
            ru12=0.6 + 0.01 * theta + np.zeros(shape),
        ),
        bsq=1.0 + 0.1 * radial + np.zeros(shape),
        bsupv=0.5 + 0.03 * radial + np.zeros(shape),
        lamscale=1.25,
    )


def _k(*, ns=3, ntheta=3, nzeta=2):
    shape = (ns, ntheta, nzeta)
    return SimpleNamespace(
        pzu_even=np.full(shape, 0.4),
        pzu_odd=np.full(shape, 0.5),
        pz1_odd=np.full(shape, 0.6),
        pru_even=np.full(shape, 0.7),
        pru_odd=np.full(shape, 0.8),
        pr1_odd=np.full(shape, 0.9),
    )


def test_sqrt_weight_and_lambda_short_mesh_branches():
    full0, half0 = p1d._sqrt_profiles_from_ns(0)
    full1, half1 = p1d._sqrt_profiles_from_ns(1)
    sm0, sp0 = p1d._sm_sp_from_s(np.asarray([0.0]))
    np.testing.assert_allclose(full0, [])
    np.testing.assert_allclose(half0, [])
    np.testing.assert_allclose(full1, [0.0])
    np.testing.assert_allclose(half1, [])
    np.testing.assert_allclose(sm0, [])
    np.testing.assert_allclose(sp0, [])

    np.testing.assert_allclose(p1d.wint_from_config(cfg=_cfg(lasym=True)), np.full((4,), 1.0 / 8.0))

    cfg = _cfg(mpol=2, ntor=1, lthreed=True)
    out, fac, debug = p1d.lambda_preconditioner(
        bc=_bc(ns=1),
        trig=SimpleNamespace(r0scale=1.0),
        s=np.asarray([0.0]),
        cfg=cfg,
        return_faclam=True,
        return_debug=True,
    )
    assert out.shape == (1, 2, 2)
    np.testing.assert_allclose(out, 0.0)
    np.testing.assert_allclose(fac, 0.0)
    for value in debug.values():
        np.testing.assert_allclose(value, [0.0])


def test_preconditioner_matrices_and_apply_edge_paths():
    cfg = _cfg(mpol=2, ntor=1, lthreed=False, lasym=False)
    s = np.linspace(0.0, 1.0, 3)
    bc = _bc(ns=3)
    k = _k(ns=3)

    empty = p1d._compute_preconditioning_matrix(
        xs=np.zeros((0, 1, 1)),
        xu12=np.zeros((0, 1, 1)),
        xu_e=np.zeros((1, 1, 1)),
        xu_o=np.zeros((1, 1, 1)),
        x1_o=np.zeros((1, 1, 1)),
        r12=np.zeros((0, 1, 1)),
        total_pressure=np.zeros((0, 1, 1)),
        tau=np.zeros((0, 1, 1)),
        bsupv=np.zeros((0, 1, 1)),
        sqrtg=np.zeros((0, 1, 1)),
        w_int=np.ones((1,)),
        sqrt_sh=np.zeros((0,)),
        sm=np.zeros((0,)),
        sp=np.zeros((0,)),
        delta_s=1.0,
    )
    assert [arr.shape for arr in empty] == [(0, 2), (0, 2), (0, 2), (0, 2), (0,)]

    mats, jmin, jmax = p1d.rz_preconditioner_matrices(bc=bc, k=k, trig=None, s=s, cfg=cfg, jmax_override=2)
    assert jmax == 2
    assert jmin.shape == (2, 2)
    for key in ("ar", "br", "dr", "az", "bz", "dz"):
        assert mats[key].shape == (2, 2, 2)

    rhs = np.asarray([1.0, 2.0, 3.0])
    unchanged = p1d._tridiagonal_solve(np.ones(3), np.ones(3), np.ones(3), rhs, 2, 2)
    np.testing.assert_allclose(unchanged, rhs)
    solved = p1d._tridiagonal_solve(np.ones(3), np.zeros(3), np.ones(3), rhs, 0, 3)
    assert np.all(np.isfinite(solved))

    base = np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2) + 1.0
    frzl = TomnspsRZL(frcc=base, frss=None, fzsc=-base, fzcs=None, flsc=np.zeros_like(base), flcs=None)
    out = p1d.rz_preconditioner_apply(frzl_in=frzl, mats=mats, jmax=jmax, cfg=cfg)
    assert out.frcc.shape == base.shape
    assert out.fzsc.shape == base.shape

    cfg_skip = _cfg(lthreed=True, lasym=False)
    assert p1d.rz_preconditioner(frzl_in=frzl, bc=bc, k=k, trig=None, s=s, cfg=cfg_skip) is frzl
    assert p1d.rz_preconditioner_apply(frzl_in=frzl, mats=mats, jmax=jmax, cfg=cfg_skip) is frzl

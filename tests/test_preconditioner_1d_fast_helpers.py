from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax import preconditioner_1d as p1d
from vmec_jax.kernels.tomnsp import TomnspsRZL


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


def test_preconditioner_cached_apply_clamps_jmax_and_zeros_nonzero_m_axis() -> None:
    cfg = _cfg(mpol=2, ntor=0, lthreed=False, lasym=False)
    shape = (3, 2, 1)
    mats = {
        "ar": np.zeros(shape),
        "br": np.zeros(shape),
        "dr": np.full(shape, 2.0),
        "az": np.zeros(shape),
        "bz": np.zeros(shape),
        "dz": np.full(shape, 4.0),
    }
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
    frzl = TomnspsRZL(frcc=base, frss=None, fzsc=-base, fzcs=None, flsc=np.zeros_like(base), flcs=None)

    no_active_surfaces = p1d.rz_preconditioner_apply(frzl_in=frzl, mats=mats, jmax=-5, cfg=cfg)
    np.testing.assert_allclose(no_active_surfaces.frcc, base)
    np.testing.assert_allclose(no_active_surfaces.fzsc, -base)

    full = p1d.rz_preconditioner_apply(frzl_in=frzl, mats=mats, jmax=99, cfg=cfg)
    expected_r = base.copy()
    expected_z = -base.copy()
    expected_r[:, 0, :] *= 0.5
    expected_z[:, 0, :] *= 0.25
    expected_r[0, 1, :] = 0.0
    expected_z[0, 1, :] = 0.0
    expected_r[1:, 1, :] *= 0.5
    expected_z[1:, 1, :] *= 0.25

    np.testing.assert_allclose(full.frcc, expected_r)
    np.testing.assert_allclose(full.fzsc, expected_z)
    np.testing.assert_allclose(full.frcc[0, 1, :], 0.0)
    np.testing.assert_allclose(full.fzsc[0, 1, :], 0.0)


def test_preconditioner_full_apply_and_faclam_debug_branch():
    cfg = _cfg(mpol=2, ntor=1, lthreed=True, lasym=False)
    s = np.linspace(0.0, 1.0, 3)
    bc = _bc(ns=3)
    k = _k(ns=3)

    lam, faclam, debug = p1d.lambda_preconditioner(
        bc=bc,
        trig=SimpleNamespace(r0scale=1.0),
        s=s,
        cfg=cfg,
        return_faclam=True,
        return_debug=True,
    )

    assert lam.shape == (3, 2, 2)
    np.testing.assert_allclose(faclam, lam)
    assert set(debug) == {"blam_pre", "clam_pre", "dlam_pre", "blam_post", "clam_post", "dlam_post"}
    assert np.count_nonzero(np.asarray(debug["dlam_pre"])) > 0

    base = np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2) + 1.0
    frzl = TomnspsRZL(frcc=base, frss=None, fzsc=-base, fzcs=None, flsc=np.zeros_like(base), flcs=None)
    out = p1d.rz_preconditioner(frzl_in=frzl, bc=bc, k=k, trig=None, s=s, cfg=_cfg(mpol=2, ntor=1))

    assert out is not frzl
    assert out.frcc.shape == base.shape
    assert out.fzsc.shape == base.shape
    assert np.all(np.isfinite(out.frcc))
    assert np.all(np.isfinite(out.fzsc))


def test_preconditioning_matrix_rejects_incomplete_radial_inputs():
    kwargs = dict(
        xs=np.ones((2, 1, 1)),
        xu12=np.ones((2, 1, 1)),
        xu_e=np.ones((3, 1, 1)),
        xu_o=np.ones((3, 1, 1)),
        x1_o=np.ones((3, 1, 1)),
        r12=np.ones((2, 1, 1)),
        total_pressure=np.ones((2, 1, 1)),
        tau=np.ones((2, 1, 1)),
        bsupv=np.ones((2, 1, 1)),
        sqrtg=np.ones((2, 1, 1)),
        w_int=np.ones((1,)),
        sqrt_sh=np.ones((2,)),
        sm=np.ones((2,)),
        sp=np.ones((2,)),
        delta_s=1.0,
    )

    cases = [
        ("xu_e", "xu_e must have ns_half\\+1 entries"),
        ("xu_o", "xu_o must have ns_half\\+1 entries"),
        ("x1_o", "x1_o must have ns_half\\+1 entries"),
        ("sqrt_sh", "sqrt_sh must have ns_half entries"),
        ("sm", "sm/sp must have ns_half entries"),
        ("sp", "sm/sp must have ns_half entries"),
    ]
    for key, message in cases:
        bad = dict(kwargs)
        bad[key] = np.ones((1, 1, 1)) if key.startswith("x") else np.ones((1,))
        with np.testing.assert_raises_regex(ValueError, message):
            p1d._compute_preconditioning_matrix(**bad)

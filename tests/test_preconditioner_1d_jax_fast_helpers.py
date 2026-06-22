from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest


Cfg = namedtuple("Cfg", "mpol ntor ntheta nzeta nfp lasym lthreed")


def _cfg(*, mpol=3, ntor=1, ntheta=6, nzeta=4, nfp=1, lasym=False, lthreed=True):
    return Cfg(
        mpol=mpol,
        ntor=ntor,
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=nfp,
        lasym=lasym,
        lthreed=lthreed,
    )


def _lambda_bc(*, ns: int, ntheta_eff: int, nzeta: int):
    shape = (ns, ntheta_eff, nzeta)
    radial = np.arange(ns, dtype=float)[:, None, None]
    theta = np.arange(ntheta_eff, dtype=float)[None, :, None]
    zeta = np.arange(nzeta, dtype=float)[None, None, :]
    return SimpleNamespace(
        guu=2.0 + 0.2 * radial + 0.03 * theta,
        guv=0.4 + 0.1 * radial + 0.02 * zeta,
        gvv=3.0 + 0.3 * radial + 0.01 * theta,
        jac=SimpleNamespace(sqrtg=1.0 + 0.05 * radial + 0.02 * theta + 0.01 * zeta),
        lamscale=1.5,
    )


def _manual_mats(*, ns=4, mpol=3, nrange=2):
    from vmec_jax import preconditioner_1d_jax as p1d

    shape = (ns, mpol, nrange)
    ar = np.full(shape, -0.08)
    br = np.full(shape, -0.05)
    dr = np.full(shape, 3.0)
    az = np.full(shape, -0.07)
    bz = np.full(shape, -0.04)
    dz = np.full(shape, 2.6)
    cr, ir = p1d._tridi_precompute_coeffs(ar, dr, br)
    cz, iz = p1d._tridi_precompute_coeffs(az, dz, bz)
    dlr_t, dr_t, dur_t = p1d._tridi_pretranspose_for_lax(ar, dr, br)
    dlz_t, dz_t, duz_t = p1d._tridi_pretranspose_for_lax(az, dz, bz)
    return {
        "ar": ar,
        "br": br,
        "dr": dr,
        "az": az,
        "bz": bz,
        "dz": dz,
        "cr": cr,
        "ir": ir,
        "cz": cz,
        "iz": iz,
        "dlr_t": dlr_t,
        "dr_t": dr_t,
        "dur_t": dur_t,
        "dlz_t": dlz_t,
        "dz_t": dz_t,
        "duz_t": duz_t,
    }


def _frzl(*, ns=4, mpol=3, nrange=2, include_optional=True):
    from vmec_jax.vmec_tomnsp import TomnspsRZL

    base = (np.arange(ns * mpol * nrange, dtype=float).reshape(ns, mpol, nrange) + 1.0) / 10.0
    zeros = np.zeros_like(base)
    if not include_optional:
        return TomnspsRZL(frcc=base, frss=None, fzsc=-base, fzcs=None, flsc=zeros, flcs=None)
    return TomnspsRZL(
        frcc=base,
        frss=base + 0.2,
        fzsc=-base,
        fzcs=base - 0.3,
        flsc=zeros,
        flcs=zeros + 0.1,
        frsc=base + 0.4,
        frcs=base + 0.6,
        fzcc=-base - 0.5,
        fzss=base + 0.8,
        flcc=zeros + 0.2,
        flss=zeros + 0.3,
    )


def test_env_profiles_lambda_preconditioner_and_cache_helpers(monkeypatch):
    pytest.importorskip("jax")

    from vmec_jax import preconditioner_1d_jax as p1d

    monkeypatch.setattr(p1d, "_ENV_USE_PRECOMPUTED", None)
    monkeypatch.setattr(p1d, "_ENV_USE_LAX_TRIDI", None)
    monkeypatch.setattr(p1d, "_ENV_RZ_MATRIX_FULL_JIT", None)
    monkeypatch.setenv("VMEC_JAX_TRIDI_PRECOMPUTE", "yes")
    monkeypatch.setenv("VMEC_JAX_TRIDI_SOLVE", "force")
    assert p1d._get_env_tridi_flags() == (True, True)
    assert p1d._rz_matrix_full_jit_enabled() is True

    monkeypatch.setattr(p1d, "_ENV_USE_PRECOMPUTED", None)
    monkeypatch.setattr(p1d, "_ENV_USE_LAX_TRIDI", None)
    monkeypatch.setattr(p1d, "_ENV_RZ_MATRIX_FULL_JIT", None)
    monkeypatch.setenv("VMEC_JAX_TRIDI_PRECOMPUTE", "0")
    monkeypatch.setenv("VMEC_JAX_TRIDI_SOLVE", "no")
    monkeypatch.setenv("VMEC_JAX_RZ_MATRIX_FULL_JIT", "0")
    assert p1d._get_env_tridi_flags() == (False, False)
    assert p1d._rz_matrix_full_jit_enabled() is False

    monkeypatch.setenv("VMEC_JAX_PRECOND_CACHE_LIMIT", "bad")
    assert p1d._lambda_precond_cache_limit() == 16
    monkeypatch.setenv("VMEC_JAX_PRECOND_CACHE_LIMIT", "0")
    assert p1d._lambda_precond_cache_limit() == 1

    monkeypatch.setenv("VMEC_JAX_PRECOND_CACHE_LIMIT", "2")
    p1d.clear_preconditioner_jit_caches()
    p1d._lambda_precond_cache_put(("a",), "A")
    p1d._lambda_precond_cache_put(("b",), "B")
    assert p1d._lambda_precond_cache_get(("a",)) == "A"
    p1d._lambda_precond_cache_put(("c",), "C")
    assert list(p1d._LAMBDA_PRECOND_JIT_CACHE) == [("a",), ("c",)]
    p1d.clear_preconditioner_jit_caches()
    assert not p1d._LAMBDA_PRECOND_JIT_CACHE

    full0, half0 = p1d._sqrt_profiles_from_ns(0, dtype=float)
    full1, half1 = p1d._sqrt_profiles_from_ns(1, dtype=float)
    sm, sp = p1d._sm_sp_from_profiles(np.array([0.0]), np.array([]))
    np.testing.assert_allclose(np.asarray(full0), [])
    np.testing.assert_allclose(np.asarray(half0), [])
    np.testing.assert_allclose(np.asarray(full1), [0.0])
    np.testing.assert_allclose(np.asarray(half1), [])
    np.testing.assert_allclose(np.asarray(sm), [])
    np.testing.assert_allclose(np.asarray(sp), [])

    np.testing.assert_allclose(np.asarray(p1d._wint_from_config(cfg=_cfg(ntheta=1, lasym=True), dtype=float)), [])
    w_lasym = np.asarray(p1d._wint_from_config(cfg=_cfg(ntheta=4, nzeta=2, lasym=True), dtype=float))
    np.testing.assert_allclose(w_lasym, np.full((4,), 1.0 / 8.0))


def test_lambda_preconditioner_debug_faclam_and_short_mesh():
    pytest.importorskip("jax")

    from vmec_jax.preconditioner_1d_jax import lambda_preconditioner

    cfg = _cfg(mpol=2, ntor=1, ntheta=4, nzeta=2, lthreed=True)
    out, fac, debug = lambda_preconditioner(
        bc=_lambda_bc(ns=1, ntheta_eff=3, nzeta=2),
        trig=SimpleNamespace(r0scale=1.2),
        s=np.array([0.0]),
        cfg=cfg,
        return_faclam=True,
        return_debug=True,
    )
    assert np.asarray(out).shape == (1, 2, 2)
    np.testing.assert_allclose(np.asarray(out), 0.0)
    np.testing.assert_allclose(np.asarray(fac), 0.0)
    for value in debug.values():
        np.testing.assert_allclose(np.asarray(value), [0.0])

    lam, fac, debug = lambda_preconditioner(
        bc=_lambda_bc(ns=3, ntheta_eff=3, nzeta=2),
        trig=SimpleNamespace(r0scale=1.2),
        s=np.linspace(0.0, 1.0, 3),
        cfg=cfg,
        damping_factor=1.7,
        return_faclam=True,
        return_debug=True,
    )
    np.testing.assert_allclose(np.asarray(lam), np.asarray(fac))
    assert float(np.asarray(debug["dlam_pre"])[1]) > 0.0
    assert np.asarray(lam)[0, 0, 0] != 0.0
    np.testing.assert_allclose(np.asarray(lam)[0, 1:, :], 0.0)

    lam_only, fac_only = lambda_preconditioner(
        bc=_lambda_bc(ns=3, ntheta_eff=3, nzeta=2),
        trig=SimpleNamespace(r0scale=1.2),
        s=np.linspace(0.0, 1.0, 3),
        cfg=cfg,
        return_faclam=True,
    )
    np.testing.assert_allclose(np.asarray(lam_only), np.asarray(fac_only))


def test_compute_assemble_reassemble_and_lax_tridi_branches():
    pytest.importorskip("jax")

    from vmec_jax import preconditioner_1d_jax as p1d

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
    assert [np.asarray(x).shape for x in empty] == [(0, 2), (0, 2), (0, 2), (0, 2), (0,)]

    ns_half = 2
    shape_h = (ns_half, 3, 2)
    shape_f = (ns_half + 1, 3, 2)
    arm, ard, brm, brd, cxd = p1d._compute_preconditioning_matrix(
        xs=np.full(shape_h, 0.3),
        xu12=np.full(shape_h, 0.4),
        xu_e=np.full(shape_f, 0.5),
        xu_o=np.full(shape_f, 0.2),
        x1_o=np.full(shape_f, 0.1),
        r12=np.full(shape_h, 1.1),
        total_pressure=np.full(shape_h, 2.0),
        tau=np.ones(shape_h),
        bsupv=np.full(shape_h, 0.6),
        sqrtg=np.full(shape_h, 1.3),
        w_int=np.array([0.2, 0.3, 0.5]),
        sqrt_sh=np.array([0.4, 0.8]),
        sm=np.array([0.9, 0.8]),
        sp=np.array([0.9, 1.1]),
        delta_s=0.5,
        ns_full=4,
    )
    assert np.asarray(arm).shape == (2, 2)
    assert np.asarray(ard).shape == (3, 2)
    assert np.asarray(cxd).shape == (3,)

    mats, jmin, jmax = p1d._assemble_rz_preconditioner_matrices_impl(
        arm=np.array([[0.2, 0.1], [0.3, 0.15]]),
        ard=np.full((4, 2), 3.0),
        brm=np.array([[0.1, 0.03], [0.2, 0.04]]),
        brd=np.full((4, 2), 0.2),
        azm=np.array([[0.25, 0.08], [0.35, 0.12]]),
        azd=np.full((4, 2), 2.8),
        bzm=np.array([[0.09, 0.02], [0.11, 0.03]]),
        bzd=np.full((4, 2), 0.15),
        cxd=np.full((4,), 0.05),
        delta_s=0.25,
        cfg=_cfg(mpol=3, ntor=1, nfp=2),
        jmax_override=4,
        use_precomputed=True,
        use_lax_tridi=True,
    )
    assert jmax == 4
    assert np.asarray(jmin).shape == (3, 2)
    for key in ("cr", "ir", "cz", "iz", "dlr_t", "dr_t", "dur_t", "dlz_t", "dz_t", "duz_t"):
        assert key in mats

    with pytest.raises(KeyError, match="Missing cached preconditioner coefficients"):
        p1d.rz_preconditioner_matrices_reassemble(mats={"arm_parity": arm}, cfg=_cfg())

    rhs = (np.arange(3 * 2, dtype=float).reshape(3, 2) + 1.0)[..., None]
    direct = p1d._tridi_solve_batched_jmin0(
        np.full((3, 2), -0.1),
        np.full((3, 2), 2.0),
        np.full((3, 2), -0.2),
        rhs,
    )
    lax = p1d._tridi_solve_batched_jmin0(
        np.full((3, 2), -0.1),
        np.full((3, 2), 2.0),
        np.full((3, 2), -0.2),
        rhs,
        use_lax_tridi=True,
    )
    np.testing.assert_allclose(np.asarray(lax), np.asarray(direct), rtol=1e-6, atol=1e-6)


def test_rz_preconditioner_numpy_jax_precomputed_and_lax_agree():
    pytest.importorskip("jax")

    from vmec_jax.preconditioner_1d_jax import rz_preconditioner_apply, rz_preconditioner_apply_numpy

    cfg = _cfg(mpol=3, ntor=1, lthreed=True, lasym=True)
    mats = _manual_mats()
    frzl = _frzl(include_optional=True)

    direct = rz_preconditioner_apply(
        frzl_in=frzl,
        mats=mats,
        jmax=4,
        cfg=cfg,
        use_precomputed=False,
        use_lax_tridi=False,
    )
    precomputed = rz_preconditioner_apply(
        frzl_in=frzl,
        mats=mats,
        jmax=4,
        cfg=cfg,
        use_precomputed=True,
        use_lax_tridi=False,
    )
    lax = rz_preconditioner_apply(
        frzl_in=frzl,
        mats=mats,
        jmax=4,
        cfg=cfg,
        use_precomputed=False,
        use_lax_tridi=True,
    )
    numpy_out = rz_preconditioner_apply_numpy(
        frzl_in=frzl,
        mats=mats,
        jmax=4,
        cfg=cfg,
        use_precomputed=True,
    )

    for attr in ("frcc", "frss", "fzsc", "fzcs", "frsc", "frcs", "fzcc", "fzss", "flsc", "flcs", "flcc", "flss"):
        expected = np.asarray(getattr(direct, attr))
        np.testing.assert_allclose(np.asarray(getattr(lax, attr)), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(getattr(precomputed, attr)), expected, rtol=1e-3, atol=5e-4)
        numpy_value = np.asarray(getattr(numpy_out, attr))
        if attr.startswith(("fr", "fz")):
            original = np.asarray(getattr(frzl, attr))
            np.testing.assert_allclose(numpy_value[:, 0, :], expected[:, 0, :], rtol=1e-3, atol=5e-4)
            np.testing.assert_allclose(numpy_value[1:, 1:, :], expected[1:, 1:, :], rtol=1e-3, atol=5e-4)
            np.testing.assert_allclose(numpy_value[0, 1:, :], original[0, 1:, :], rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_allclose(numpy_value, expected, rtol=1e-6, atol=1e-6)

    sparse_mats = {key: value for key, value in mats.items() if key in {"ar", "br", "dr", "az", "bz", "dz"}}
    minimal = rz_preconditioner_apply_numpy(
        frzl_in=_frzl(include_optional=False),
        mats=sparse_mats,
        jmax=0,
        cfg=_cfg(mpol=3, ntor=1, lthreed=False, lasym=False),
        use_precomputed=True,
    )
    assert minimal.frss is None
    assert minimal.fzcs is None


def test_high_mode_non_lasym_precomputed_tridi_matches_direct_apply():
    pytest.importorskip("jax")

    from vmec_jax.preconditioner_1d_jax import rz_preconditioner_apply

    cfg = _cfg(mpol=5, ntor=4, lthreed=True, lasym=False)
    mats = _manual_mats(ns=5, mpol=5, nrange=5)
    frzl = _frzl(ns=5, mpol=5, nrange=5, include_optional=False)

    direct = rz_preconditioner_apply(
        frzl_in=frzl,
        mats=mats,
        jmax=5,
        cfg=cfg,
        use_precomputed=False,
        use_lax_tridi=False,
    )
    precomputed = rz_preconditioner_apply(
        frzl_in=frzl,
        mats=mats,
        jmax=5,
        cfg=cfg,
        use_precomputed=True,
        use_lax_tridi=False,
    )

    for attr in ("frcc", "fzsc", "flsc"):
        np.testing.assert_allclose(
            np.asarray(getattr(precomputed, attr)),
            np.asarray(getattr(direct, attr)),
            rtol=1e-3,
            atol=5e-4,
        )
    assert precomputed.frss is None
    assert precomputed.fzcs is None


def test_rz_preconditioner_apply_jit_reuses_inactive_placeholders(monkeypatch):
    from vmec_jax import preconditioner_1d_jax as p1d
    from vmec_jax.vmec_tomnsp import TomnspsRZL

    mats = _manual_mats()
    frzl = _frzl(include_optional=False)
    seen = {}

    def fake_make_apply(**_kwargs):
        def fake_apply(
            frcc,
            fzsc,
            frss,
            fzcs,
            frsc,
            frcs,
            fzcc,
            fzss,
            *_args,
        ):
            seen.update(
                frcc=frcc,
                fzsc=fzsc,
                frss=frss,
                fzcs=fzcs,
                frsc=frsc,
                frcs=frcs,
                fzcc=fzcc,
                fzss=fzss,
            )
            return TomnspsRZL(frcc=frcc, frss=None, fzsc=fzsc, fzcs=None, flsc=frzl.flsc, flcs=None)

        return fake_apply

    monkeypatch.setattr(p1d, "_make_rz_preconditioner_apply_jit", fake_make_apply)
    out = p1d.rz_preconditioner_apply_jit(
        frzl_in=frzl,
        mats=mats,
        jmax=4,
        cfg=_cfg(mpol=3, ntor=1, lthreed=False, lasym=False),
    )

    assert out.frss is None
    assert out.fzcs is None
    assert seen["frss"] is seen["frcc"]
    assert seen["frsc"] is seen["frcc"]
    assert seen["frcs"] is seen["frcc"]
    assert seen["fzcs"] is seen["fzsc"]
    assert seen["fzcc"] is seen["fzsc"]
    assert seen["fzss"] is seen["fzsc"]


def test_numpy_tridiagonal_edge_cases():
    from vmec_jax import preconditioner_1d_jax as p1d

    empty = np.zeros((0, 2))
    np.testing.assert_allclose(p1d._tridi_fwd_np(empty, empty, empty), empty)
    one = np.array([[2.0, 3.0]])
    np.testing.assert_allclose(p1d._tridi_bwd_np(one, one), one)
    np.testing.assert_allclose(p1d._tridi_solve_np(empty, empty, empty, empty), empty)

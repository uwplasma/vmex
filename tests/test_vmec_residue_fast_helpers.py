from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import jax, jnp
from vmec_jax.solvers.fixed_boundary.preconditioning.operators import metric_surface_precond_from_bcovar_jax
from vmec_jax.vmec_residue import (
    VmecForceNormsDynamic,
    VmecForceNorms,
    _PWINT_CACHE,
    _SCALXC_CACHE,
    _WINT_CACHE,
    _pwint_cache_key,
    _scalxc_cache_key,
    _wint_cache_key,
    vmec_apply_m1_constraints,
    vmec_apply_scalxc_to_tomnsps,
    vmec_force_norms_from_bcovar_dynamic,
    vmec_force_norms_scales_from_bcovar_dynamic,
    vmec_fsq_from_tomnsps,
    vmec_fsq_from_tomnsps_dynamic,
    vmec_fsq_sums_from_tomnsps,
    vmec_gcx2_from_tomnsps,
    vmec_gcx2_from_tomnsps_np,
    vmec_pwint_from_trig,
    vmec_rz_decompose_signed,
    vmec_rz_norm_from_state,
    vmec_scalxc_from_s,
    vmec_wint_from_trig,
    vmec_zero_m1_zforce,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_trig_tables


def _constant_tomnsps(*, ns: int = 3, mpol: int = 3, ntor1: int = 1) -> TomnspsRZL:
    shape = (ns, mpol, ntor1)

    def block(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=float)

    return TomnspsRZL(
        frcc=block(1.0),
        frss=block(4.0),
        fzsc=block(2.0),
        fzcs=block(5.0),
        flsc=block(3.0),
        flcs=block(6.0),
        frsc=block(7.0),
        frcs=block(10.0),
        fzcc=block(8.0),
        fzss=block(11.0),
        flcc=block(9.0),
        flss=block(12.0),
    )


def test_m1_constraint_rotation_and_zforce_zeroing_are_local_to_m1_z_blocks():
    frzl = _constant_tomnsps()
    constrained = vmec_apply_m1_constraints(frzl=frzl, lconm1=True)
    osqrt2 = 1.0 / np.sqrt(2.0)

    np.testing.assert_allclose(np.asarray(constrained.frss)[:, 1, :], (4.0 + 5.0) * osqrt2)
    np.testing.assert_allclose(np.asarray(constrained.fzcs)[:, 1, :], (4.0 - 5.0) * osqrt2)
    np.testing.assert_allclose(np.asarray(constrained.frsc)[:, 1, :], (7.0 + 8.0) * osqrt2)
    np.testing.assert_allclose(np.asarray(constrained.fzcc)[:, 1, :], (7.0 - 8.0) * osqrt2)

    np.testing.assert_allclose(np.asarray(constrained.frss)[:, [0, 2], :], 4.0)
    np.testing.assert_allclose(np.asarray(constrained.fzcs)[:, [0, 2], :], 5.0)
    np.testing.assert_allclose(np.asarray(constrained.fzsc), 2.0)

    zeroed = vmec_zero_m1_zforce(frzl=frzl, enabled=True)
    np.testing.assert_allclose(np.asarray(zeroed.fzcs)[:, 1, :], 0.0)
    np.testing.assert_allclose(np.asarray(zeroed.fzcc)[:, 1, :], 0.0)
    np.testing.assert_allclose(np.asarray(zeroed.fzsc), 2.0)
    np.testing.assert_allclose(np.asarray(zeroed.fzcs)[:, [0, 2], :], 5.0)


def test_scalxc_and_fsq_sums_apply_odd_m_scaling_and_edge_policy():
    s = np.array([0.0, 0.25, 1.0])
    scalxc = vmec_scalxc_from_s(s=s, mpol=3)
    np.testing.assert_allclose(np.asarray(scalxc), [[1.0, 2.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])

    frzl = _constant_tomnsps()
    scaled = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)
    np.testing.assert_allclose(np.asarray(scaled.frcc)[:, :, 0], np.asarray(scalxc) * 1.0)
    np.testing.assert_allclose(np.asarray(scaled.fzsc)[:, :, 0], np.asarray(scalxc) * 2.0)

    sums = vmec_fsq_sums_from_tomnsps(
        frzl=frzl,
        apply_m1_constraints=False,
        apply_scalxc=True,
        include_edge=False,
        s=s,
    )
    assert sums.gcr2 == (1.0**2 + 4.0**2 + 7.0**2 + 10.0**2) * 12.0
    assert sums.gcz2 == (2.0**2 + 5.0**2 + 8.0**2 + 11.0**2) * 12.0
    assert sums.gcl2 == (3.0**2 + 6.0**2 + 9.0**2 + 12.0**2) * 15.0

    gcx2_jax = vmec_gcx2_from_tomnsps(
        frzl=frzl,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=False,
    )
    gcx2_np = vmec_gcx2_from_tomnsps_np(frzl=frzl, include_edge=False)
    np.testing.assert_allclose(np.asarray(gcx2_jax), np.asarray(gcx2_np))

    norms = VmecForceNorms(fnorm=2.0, fnormL=3.0, r1=5.0)
    dynamic_norms = VmecForceNormsDynamic(
        fnorm=2.0,
        fnormL=3.0,
        r1=5.0,
        r2=0.0,
        volume=0.0,
        wb=0.0,
        wp=0.0,
        vp=np.zeros(3),
    )
    fsq = vmec_fsq_from_tomnsps(
        frzl=frzl,
        norms=norms,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=False,
    )
    fsq_dynamic = vmec_fsq_from_tomnsps_dynamic(
        frzl=frzl,
        norms=dynamic_norms,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=False,
    )
    np.testing.assert_allclose([fsq.fsqr, fsq.fsqz, fsq.fsql], np.asarray([fsq_dynamic.fsqr, fsq_dynamic.fsqz, fsq_dynamic.fsql]))


def test_weight_and_scalxc_cache_keys_are_shape_and_grid_specific():
    trig = vmec_trig_tables(ntheta=8, nzeta=4, nfp=2, mmax=3, nmax=2, lasym=False, cache=False)
    _WINT_CACHE.clear()
    _PWINT_CACHE.clear()
    _SCALXC_CACHE.clear()

    wint1 = vmec_wint_from_trig(trig, nzeta=4)
    wint2 = vmec_wint_from_trig(trig, nzeta=4)
    assert wint1 is wint2
    assert _wint_cache_key(trig, nzeta=4) in _WINT_CACHE
    assert _wint_cache_key(trig, nzeta=3) not in _WINT_CACHE

    pwint1 = vmec_pwint_from_trig(trig, ns=3, nzeta=4)
    pwint2 = vmec_pwint_from_trig(trig, ns=3, nzeta=4)
    assert pwint1 is pwint2
    assert _pwint_cache_key(trig, ns=3, nzeta=4) in _PWINT_CACHE
    np.testing.assert_allclose(np.asarray(pwint1[0]), 0.0)
    np.testing.assert_allclose(np.asarray(pwint1[1]), np.asarray(wint1))
    with pytest.raises(ValueError, match="ns must be >= 1"):
        vmec_pwint_from_trig(trig, ns=0, nzeta=4)

    s = np.asarray([0.0, 0.25, 1.0])
    scalxc1 = vmec_scalxc_from_s(s=s, mpol=4)
    scalxc2 = vmec_scalxc_from_s(s=s, mpol=4)
    np.testing.assert_allclose(np.asarray(scalxc1), np.asarray(scalxc2))
    assert _scalxc_cache_key(s=s, mpol=4) in _SCALXC_CACHE
    assert vmec_scalxc_from_s(s=np.asarray([]), mpol=3).shape == (0, 3)
    assert vmec_scalxc_from_s(s=np.asarray([0.0, 1.0]), mpol=0).shape == (2, 0)


def test_dynamic_force_norm_guards_and_tree_roundtrip():
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=False, cache=False)

    short_bc = SimpleNamespace(jac=SimpleNamespace(sqrtg=np.ones((1, 2, 3))))
    norms = vmec_force_norms_from_bcovar_dynamic(bc=short_bc, trig=trig, s=np.asarray([0.0]), signgs=1)
    assert np.isnan(float(norms.fnorm))
    assert np.asarray(norms.vp).shape == (1,)

    bad_bc = SimpleNamespace(jac=SimpleNamespace(sqrtg=np.ones((2, 3))))
    with pytest.raises(ValueError, match="sqrtg"):
        vmec_force_norms_from_bcovar_dynamic(bc=bad_bc, trig=trig, s=np.asarray([0.0, 1.0]), signgs=1)

    shape = (3, int(trig.ntheta3), 3)
    bc = SimpleNamespace(
        jac=SimpleNamespace(sqrtg=np.ones(shape), r12=np.ones(shape)),
        bsupu=np.full(shape, 0.4),
        bsupv=np.full(shape, 0.6),
        bsubu=np.full(shape, 0.5),
        bsubv=np.full(shape, 0.25),
        bsq=np.full(shape, 2.0),
        guu=np.full(shape, 1.5),
        lamscale=2.0,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(bc=bc, trig=trig, s=np.linspace(0.0, 1.0, 3), signgs=1)
    leaves, aux = norms.tree_flatten()
    rebuilt = VmecForceNormsDynamic.tree_unflatten(aux, leaves)
    np.testing.assert_allclose(np.asarray(rebuilt.vp), np.asarray(norms.vp))
    assert np.isfinite(float(norms.fnorm))
    assert np.isfinite(float(norms.fnormL))
    assert float(norms.volume) > 0.0


def test_dynamic_force_norms_scales_fused_matches_separate_helpers():
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=False, cache=False)
    shape = (3, int(trig.ntheta3), 3)
    bc = SimpleNamespace(
        jac=SimpleNamespace(
            sqrtg=jnp.ones(shape),
            r12=jnp.linspace(1.0, 1.2, np.prod(shape)).reshape(shape),
        ),
        bsupu=jnp.full(shape, 0.4),
        bsupv=jnp.full(shape, 0.6),
        bsubu=jnp.linspace(0.5, 0.8, np.prod(shape)).reshape(shape),
        bsubv=jnp.full(shape, 0.25),
        bsq=jnp.full(shape, 2.0),
        guu=jnp.linspace(1.2, 1.7, np.prod(shape)).reshape(shape),
        lamscale=2.0,
    )
    s = np.linspace(0.0, 1.0, 3)

    fused = vmec_force_norms_scales_from_bcovar_dynamic(bc=bc, trig=trig, s=s, signgs=1)
    norms = vmec_force_norms_from_bcovar_dynamic(bc=bc, trig=trig, s=s, signgs=1)
    rz_scale, l_scale = metric_surface_precond_from_bcovar_jax(bc=bc, trig=trig)

    np.testing.assert_allclose(np.asarray(fused.norms.fnorm), np.asarray(norms.fnorm), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(fused.norms.fnormL), np.asarray(norms.fnormL), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(fused.rz_scale), np.asarray(rz_scale), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(fused.l_scale), np.asarray(l_scale), rtol=1e-13)

    def scalar(scale):
        scaled_bc = SimpleNamespace(**{**bc.__dict__, "guu": scale * bc.guu})
        out = vmec_force_norms_scales_from_bcovar_dynamic(bc=scaled_bc, trig=trig, s=s, signgs=1)
        return jnp.sum(out.rz_scale) + out.norms.fnorm

    grad = jax.grad(scalar)(jnp.asarray(1.0))
    assert np.isfinite(float(grad))


def test_rz_decompose_signed_and_norm_options(load_case_circular_tokamak):
    _cfg, _indata, static, _boundary, state = load_case_circular_tokamak

    rcc, rss, zsc, zcs = vmec_rz_decompose_signed(
        state,
        static,
        apply_scalxc=False,
        apply_basis_norm=False,
    )
    assert rcc.shape[0] == static.cfg.ns
    assert rss.shape == rcc.shape
    assert zsc.shape == rcc.shape
    assert zcs.shape == rcc.shape

    norm_full = vmec_rz_norm_from_state(
        state=state,
        static=static,
        apply_scalxc=False,
        apply_basis_norm=False,
    )
    manual_full = np.sum(np.asarray(zsc) ** 2)
    m_idx = np.arange(rcc.shape[1])[None, :, None]
    n_idx = np.arange(rcc.shape[2])[None, None, :]
    include_rcc = (m_idx > 0) | (n_idx > 0)
    manual_full += np.sum(np.where(include_rcc, np.asarray(rcc) ** 2, 0.0))
    manual_full += np.sum(np.asarray(rss) ** 2) + np.sum(np.asarray(zcs) ** 2)
    np.testing.assert_allclose(float(norm_full), manual_full, rtol=1e-12, atol=1e-12)

    norm_slice = vmec_rz_norm_from_state(
        state=state,
        static=static,
        apply_scalxc=False,
        apply_basis_norm=False,
        ns_min=1,
        ns_max=3,
    )
    assert 0.0 <= float(norm_slice) <= float(norm_full)

    scaled = vmec_rz_decompose_signed(
        state,
        static,
        apply_scalxc=True,
        apply_basis_norm=True,
        s=static.s,
    )
    for arr in scaled:
        assert np.all(np.isfinite(np.asarray(arr)))

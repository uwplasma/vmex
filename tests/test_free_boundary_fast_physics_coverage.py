from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import vmec_jax.free_boundary as freeb
from vmec_jax.config import VMECConfig
from vmec_jax.free_boundary import (
    ExternalBoundarySample,
    NestorPoissonCache,
    NestorRuntimeState,
    NestorVmecLikeCache,
    _build_vmec_mode_basis,
    _vmec_nonsingular_gsource_from_bexni,
    _vmec_nonsingular_terms_from_bexni,
    nestor_external_only_step,
    vacuum_boundary_fields_from_cylindrical,
)


def _static(*, ntheta: int, nzeta: int, mpol: int = 1, ntor: int = 0, lasym: bool = False) -> SimpleNamespace:
    cfg = VMECConfig(
        mpol=mpol,
        ntor=ntor,
        ns=3,
        nfp=1,
        lasym=lasym,
        lthreed=True,
        lconm1=True,
        ntheta=ntheta,
        nzeta=nzeta,
    )
    return SimpleNamespace(cfg=cfg, trig_vmec=None, signgs=-1)


def _analytic_sample(*, ntheta: int = 4, nzeta: int = 3, field_scale: float = 1.0) -> ExternalBoundarySample:
    theta = (2.0 * np.pi / float(ntheta)) * np.arange(ntheta)
    zeta = (2.0 * np.pi / float(nzeta)) * np.arange(nzeta)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")

    R = 2.0 + 0.20 * np.cos(th) + 0.03 * np.cos(ze)
    Z = 0.20 * np.sin(th) + 0.02 * np.sin(ze)
    Ru = -0.20 * np.sin(th)
    Zu = 0.20 * np.cos(th)
    Rv = -0.03 * np.sin(ze)
    Zv = 0.02 * np.cos(ze)
    br = field_scale * (0.35 + 0.07 * np.cos(th) + 0.02 * np.sin(ze))
    bp = 0.12 + 0.03 * np.cos(ze)
    bz = field_scale * (0.25 * np.sin(th) - 0.04 * np.cos(ze))
    phi = np.broadcast_to(zeta[None, :], R.shape)
    vac = vacuum_boundary_fields_from_cylindrical(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    zeros = np.zeros_like(R)

    return ExternalBoundarySample(
        mgrid_path="synthetic:no-netcdf",
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        br=br,
        bp=bp,
        bz=bz,
        br_mgrid=br,
        bp_mgrid=bp,
        bz_mgrid=bz,
        br_axis=zeros.copy(),
        bp_axis=zeros.copy(),
        bz_axis=zeros.copy(),
        axis_r=2.0 + 0.03 * np.cos(zeta),
        axis_z=0.02 * np.sin(zeta),
        vac_ext=vac,
        ruu=-0.20 * np.cos(th),
        ruv=zeros.copy(),
        rvv=-0.03 * np.cos(ze),
        zuu=-0.20 * np.sin(th),
        zuv=zeros.copy(),
        zvv=-0.02 * np.sin(ze),
    )


def test_spectral_ivacskip_reuses_poisson_cache_but_refreshes_rhs(monkeypatch) -> None:
    sample1 = _analytic_sample(ntheta=4, nzeta=3, field_scale=1.0)
    sample2 = _analytic_sample(ntheta=4, nzeta=3, field_scale=1.8)
    samples = [sample1, sample2]
    build_calls = 0
    build_poisson_cache = freeb._build_poisson_cache

    def sample_external_boundary_arrays(**_kwargs):
        return samples.pop(0)

    def counted_build_poisson_cache(*, ntheta: int, nzeta: int) -> NestorPoissonCache:
        nonlocal build_calls
        build_calls += 1
        return build_poisson_cache(ntheta=ntheta, nzeta=nzeta)

    monkeypatch.setattr(freeb, "_sample_external_boundary_arrays", sample_external_boundary_arrays)
    monkeypatch.setattr(freeb, "_build_poisson_cache", counted_build_poisson_cache)
    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "fast")
    monkeypatch.setenv("VMEC_JAX_FREEB_RHS_MODE", "bnormal_unit")
    monkeypatch.setenv("VMEC_JAX_FREEB_REUSE_RHS_UPDATE", "1")

    result_full, runtime_full = nestor_external_only_step(
        state=object(),
        static=_static(ntheta=4, nzeta=3),
        ivac=1,
        iter_idx=1,
    )
    result_reuse, runtime_reuse = nestor_external_only_step(
        state=object(),
        static=_static(ntheta=4, nzeta=3),
        ivac=1,
        ivacskip=1,
        iter_idx=2,
        runtime=runtime_full,
    )

    assert result_full.reused is False
    assert result_reuse.reused is True
    assert build_calls == 1
    assert runtime_reuse.operator_cache is runtime_full.operator_cache
    assert runtime_reuse.update_count == 1
    assert runtime_reuse.reuse_count == 1
    assert np.isfinite(result_reuse.vac_total.bsqvac).all()
    assert float(np.max(result_reuse.vac_total.bsqvac)) > 0.0
    assert not np.allclose(result_full.phi, result_reuse.phi)


def test_legacy_reuse_hold_path_preserves_cached_bsqvac_without_sampling(monkeypatch) -> None:
    runtime = NestorRuntimeState(
        operator_cache=NestorPoissonCache(ntheta=2, nzeta=2, lam=np.ones((2, 2))),
        phi=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        bsqvac=np.asarray([[0.5, 0.6], [0.7, 0.8]]),
        mode="spectral_poisson_external_only",
        update_count=5,
        reuse_count=7,
        source_cache_iter=11,
        gsource_cached=np.arange(4.0),
        source_sym_cached=np.arange(4.0) + 10.0,
        bvec_nonsing_cached=np.arange(4.0) + 20.0,
    )

    def fail_if_sampled(**_kwargs):
        raise AssertionError("legacy hold reuse must not resample the boundary")

    monkeypatch.setattr(freeb, "_sample_external_boundary_arrays", fail_if_sampled)
    monkeypatch.setenv("VMEC_JAX_FREEB_REUSE_RHS_UPDATE", "0")

    result, runtime_next = nestor_external_only_step(
        state=object(),
        static=object(),
        ivac=2,
        runtime=runtime,
    )

    assert result.reused is True
    assert result.sample_time_s == 0.0
    assert result.solve_time_s == 0.0
    np.testing.assert_allclose(result.phi, runtime.phi)
    np.testing.assert_allclose(result.vac_total.bsqvac, runtime.bsqvac)
    assert runtime_next.update_count == runtime.update_count
    assert runtime_next.reuse_count == runtime.reuse_count + 1
    np.testing.assert_allclose(runtime_next.gsource_cached, runtime.gsource_cached)
    np.testing.assert_allclose(runtime_next.source_sym_cached, runtime.source_sym_cached)
    np.testing.assert_allclose(runtime_next.bvec_nonsing_cached, runtime.bvec_nonsing_cached)


def test_dense_mode_reuse_keeps_operator_and_cached_source_vectors(monkeypatch) -> None:
    sample1 = _analytic_sample(ntheta=3, nzeta=2, field_scale=1.0)
    sample2 = _analytic_sample(ntheta=3, nzeta=2, field_scale=2.0)
    samples = [sample1, sample2]

    monkeypatch.setattr(freeb, "_sample_external_boundary_arrays", lambda **_kwargs: samples.pop(0))
    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "dense")
    monkeypatch.setenv("VMEC_JAX_FREEB_DENSE_SOLVE_MODE", "mode")
    monkeypatch.setenv("VMEC_JAX_FREEB_USE_GREENF_SOURCE", "no")
    monkeypatch.setenv("VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC", "0")

    static = _static(ntheta=3, nzeta=2, mpol=1, ntor=0)
    result_full, runtime_full = nestor_external_only_step(
        state=object(),
        static=static,
        ivac=1,
        iter_idx=20,
    )
    result_reuse, runtime_reuse = nestor_external_only_step(
        state=object(),
        static=static,
        ivac=2,
        ivacskip=1,
        iter_idx=21,
        runtime=runtime_full,
    )

    assert result_full.model == "vmec2000_like_dense_integral"
    assert result_reuse.model == "vmec2000_like_dense_integral"
    assert result_reuse.reused is True
    assert isinstance(runtime_full.operator_cache, NestorVmecLikeCache)
    assert runtime_reuse.operator_cache is runtime_full.operator_cache
    assert runtime_reuse.update_count == 1
    assert runtime_reuse.reuse_count == 1
    assert runtime_reuse.source_cache_iter == 20
    np.testing.assert_allclose(runtime_reuse.gsource_cached, runtime_full.gsource_cached)
    np.testing.assert_allclose(runtime_reuse.source_sym_cached, runtime_full.source_sym_cached)
    np.testing.assert_allclose(runtime_reuse.bvec_nonsing_cached, runtime_full.bvec_nonsing_cached)
    assert np.isfinite(result_reuse.vac_total.bsqvac).all()
    assert float(np.max(result_reuse.vac_total.bsqvac)) > 0.0


def test_nonsingular_source_identity_matches_full_terms_on_tiny_surface() -> None:
    sample = _analytic_sample(ntheta=2, nzeta=2, field_scale=1.0)
    basis = _build_vmec_mode_basis(
        ntheta=sample.R.shape[0],
        nzeta=sample.R.shape[1],
        nfp=1,
        mf=1,
        nf=1,
        lasym=False,
        wint=np.full(sample.R.shape, 1.0 / float(sample.R.size)),
    )
    bexni = np.linspace(-0.2, 0.4, sample.R.size)

    gsource_direct = _vmec_nonsingular_gsource_from_bexni(
        sample=sample,
        basis=basis,
        bexni=bexni,
        signgs=-1,
        nvper=1,
    )
    gsource_terms, grpmn = _vmec_nonsingular_terms_from_bexni(
        sample=sample,
        basis=basis,
        bexni=bexni,
        signgs=-1,
        nvper=1,
    )
    zero_gsource, zero_grpmn = _vmec_nonsingular_terms_from_bexni(
        sample=sample,
        basis=basis,
        bexni=np.zeros_like(bexni),
        signgs=-1,
        nvper=1,
    )

    np.testing.assert_allclose(gsource_terms, gsource_direct, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(zero_gsource, 0.0, atol=1.0e-14)
    assert grpmn.shape == (int(basis["mnpd2"]), int(basis["nuv3"]))
    assert zero_grpmn.shape == grpmn.shape
    assert np.isfinite(gsource_terms).all()
    assert np.isfinite(grpmn).all()
    assert np.isfinite(zero_grpmn).all()

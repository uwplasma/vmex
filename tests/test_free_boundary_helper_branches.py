from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.config import VMECConfig
from vmec_jax.free_boundary import (
    ExternalBoundarySample,
    NestorVmecLikeCache,
    VacuumBoundaryFields,
    _FREEB_BOUNDARY_SETUP_CACHE,
    _FREEB_HOST_PHASE_CACHE,
    _FREEB_WINT_CACHE,
    _axis_current_field_simple,
    _axis_current_field_vmec_filament,
    _build_vmec_like_cache,
    _build_vmec_mode_basis,
    _decode_char_rows,
    _decode_char_scalar,
    _dense_lu_factor,
    _ensure_vmec_nonsingular_kernel_tables,
    _freeb_boundary_sample_setup,
    _freeb_boundary_trig,
    _freeb_host_phase_stack,
    _solve_vmec_like_dense,
    _solve_vmec_like_mode_from_gsource,
    _spectral_second_derivatives_2d,
    _vacuum_channels_from_sample_potvac,
    _vmec_boundary_wint,
    _vmec_mode_matrix_from_grpmn,
    _vmec_nonsingular_gsource_from_bexni,
    _vmec_nonsingular_terms_from_bexni,
    _vmec_source_from_gsource,
)


def _cfg(*, ntheta: int = 4, nzeta: int = 3, nfp: int = 2, lasym: bool = False) -> VMECConfig:
    return VMECConfig(
        mpol=2,
        ntor=1,
        ns=3,
        nfp=nfp,
        lasym=lasym,
        lthreed=True,
        lconm1=True,
        ntheta=ntheta,
        nzeta=nzeta,
    )


def _manual_trig(*, ntheta: int = 3, nzeta: int = 2, mmax: int = 1, nmax: int = 1) -> SimpleNamespace:
    theta = (2.0 * np.pi / float(ntheta)) * np.arange(ntheta)
    zeta = (2.0 * np.pi / float(nzeta)) * np.arange(nzeta)
    m = np.arange(mmax + 1)
    n = np.arange(nmax + 1)
    phase_u = theta[:, None] * m[None, :]
    phase_v = zeta[:, None] * n[None, :]
    return SimpleNamespace(
        cosmu=np.cos(phase_u),
        sinmu=np.sin(phase_u),
        cosmum=m[None, :] * np.cos(phase_u),
        sinmum=-m[None, :] * np.sin(phase_u),
        cosnv=np.cos(phase_v),
        sinnv=np.sin(phase_v),
        cosnvn=-n[None, :] * np.cos(phase_v),
        sinnvn=-n[None, :] * np.sin(phase_v),
    )


def _zero_vacuum(shape: tuple[int, int]) -> VacuumBoundaryFields:
    zeros = np.zeros(shape, dtype=float)
    ones = np.ones(shape, dtype=float)
    return VacuumBoundaryFields(
        bu=zeros.copy(),
        bv=zeros.copy(),
        bsupu=zeros.copy(),
        bsupv=zeros.copy(),
        bsqvac=zeros.copy(),
        bnormal=zeros.copy(),
        bnormal_unit=zeros.copy(),
        g_uu=ones.copy(),
        g_uv=zeros.copy(),
        g_vv=ones.copy(),
        det_guv=ones.copy(),
    )


def _analytic_sample(*, ntheta: int = 4, nzeta: int = 3) -> ExternalBoundarySample:
    theta = (2.0 * np.pi / float(ntheta)) * np.arange(ntheta)
    zeta = (2.0 * np.pi / float(nzeta)) * np.arange(nzeta)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")

    R = 2.0 + 0.10 * np.cos(th) + 0.03 * np.cos(ze)
    Z = 0.10 * np.sin(th) + 0.02 * np.sin(ze)
    Ru = -0.10 * np.sin(th)
    Zu = 0.10 * np.cos(th)
    Rv = -0.03 * np.sin(ze)
    Zv = 0.02 * np.cos(ze)
    ruu = -0.10 * np.cos(th)
    ruv = np.zeros_like(R)
    rvv = -0.03 * np.cos(ze)
    zuu = -0.10 * np.sin(th)
    zuv = np.zeros_like(R)
    zvv = -0.02 * np.sin(ze)
    zeros = np.zeros_like(R)

    return ExternalBoundarySample(
        mgrid_path="synthetic.nc",
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=zeros.copy(),
        br=zeros.copy(),
        bp=zeros.copy(),
        bz=zeros.copy(),
        br_mgrid=zeros.copy(),
        bp_mgrid=zeros.copy(),
        bz_mgrid=zeros.copy(),
        br_axis=zeros.copy(),
        bp_axis=zeros.copy(),
        bz_axis=zeros.copy(),
        axis_r=np.ones(nzeta),
        axis_z=np.zeros(nzeta),
        vac_ext=_zero_vacuum((ntheta, nzeta)),
        ruu=ruu,
        ruv=ruv,
        rvv=rvv,
        zuu=zuu,
        zuv=zuv,
        zvv=zvv,
    )


def test_host_phase_stack_rebuilds_bad_cache_entry_and_rejects_unknown_derivative() -> None:
    modes = SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, -1]))
    trig = _manual_trig()
    key = (id(modes), id(trig), ("base",))
    old = _FREEB_HOST_PHASE_CACHE.get(key)
    _FREEB_HOST_PHASE_CACHE[key] = np.zeros((1, 1, 1, 1))
    try:
        phase = _freeb_host_phase_stack(modes=modes, trig=trig, derivs=("base",))
        assert phase.shape == (1, 4, 3, 2)
        assert _FREEB_HOST_PHASE_CACHE[key].shape == phase.shape

        with pytest.raises(ValueError, match="Unknown deriv"):
            _freeb_host_phase_stack(modes=modes, trig=trig, derivs=("bad",))
    finally:
        if old is None:
            _FREEB_HOST_PHASE_CACHE.pop(key, None)
        else:
            _FREEB_HOST_PHASE_CACHE[key] = old


def test_boundary_setup_trig_and_wint_caches_cover_fallback_paths() -> None:
    cfg = _cfg()
    modes = SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, -1]))
    static = SimpleNamespace(cfg=cfg, modes=modes, trig_vmec=None, m_is_even=np.asarray([True, False]))

    setup = _freeb_boundary_sample_setup(static=static, sample_nzeta=2)
    cached_setup = _freeb_boundary_sample_setup(static=static, sample_nzeta=2)
    assert cached_setup is setup
    assert setup.second_facs.shape == (3, 1, 2)
    assert setup.phi_grid.shape == (np.asarray(setup.trig.cosmu).shape[0], 2)

    trig = _freeb_boundary_trig(cfg=cfg, nzeta=2)
    assert _freeb_boundary_trig(cfg=cfg, nzeta=2) is trig

    bad_static = SimpleNamespace(trig_vmec=object())
    fallback_w = _vmec_boundary_wint(static=bad_static, ntheta=2, nzeta=3)
    assert _vmec_boundary_wint(static=bad_static, ntheta=2, nzeta=3) is fallback_w
    np.testing.assert_allclose(fallback_w, np.full((2, 3), 1.0 / 6.0))

    basis = _build_vmec_mode_basis(
        ntheta=2,
        nzeta=3,
        nfp=1,
        mf=0,
        nf=0,
        lasym=True,
        wint=np.asarray([99.0]),
    )
    np.testing.assert_allclose(basis["wint"], np.full(6, 1.0 / 6.0))

    _FREEB_BOUNDARY_SETUP_CACHE.pop((id(static), -1, 2), None)
    _FREEB_WINT_CACHE.pop((id(static), setup.phi_grid.shape[0], 2), None)
    _FREEB_WINT_CACHE.pop((id(bad_static), 2, 3), None)


def test_axis_current_helpers_nonzero_and_degenerate_filament_paths() -> None:
    R = np.full((2, 4), 2.0)
    Z = np.zeros((2, 4))
    phi = np.broadcast_to((2.0 * np.pi / 4.0) * np.arange(4), R.shape)
    axis_r = np.full(4, 0.5)
    axis_z = np.asarray([0.0, 0.1, 0.0, -0.1])

    simple = _axis_current_field_simple(R=R, Z=Z, phi=phi, axis_r=axis_r, axis_z=axis_z, nfp=1, plascur=1.0)
    filament = _axis_current_field_vmec_filament(R=R, Z=Z, axis_r=axis_r, axis_z=axis_z, nfp=1, plascur=1.0)
    for got in (*simple, *filament):
        assert got.shape == R.shape
        assert np.isfinite(got).all()
    assert max(float(np.max(np.abs(got))) for got in simple) > 0.0
    assert max(float(np.max(np.abs(got))) for got in filament) > 0.0

    degenerate = _axis_current_field_vmec_filament(
        R=R[:, :3],
        Z=Z[:, :3],
        axis_r=np.zeros(3),
        axis_z=np.zeros(3),
        nfp=1,
        plascur=1.0,
    )
    for got in degenerate:
        np.testing.assert_allclose(got, 0.0)


def test_spectral_second_derivatives_match_single_fourier_mode() -> None:
    nu, nv = 8, 8
    u = (2.0 * np.pi / float(nu)) * np.arange(nu)
    v = (2.0 * np.pi / float(nv)) * np.arange(nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    field = np.cos(uu + 2.0 * vv)

    duu, duv, dvv = _spectral_second_derivatives_2d(field)

    np.testing.assert_allclose(duu, -field, atol=1.0e-12)
    np.testing.assert_allclose(duv, -2.0 * field, atol=1.0e-12)
    np.testing.assert_allclose(dvv, -4.0 * field, atol=1.0e-12)


def test_vmec_source_mode_matrix_and_mode_solve_helper_branches() -> None:
    ntheta, nzeta = 3, 2
    wint = np.full((ntheta, nzeta), 1.0 / float(ntheta * nzeta))
    basis_sym = _build_vmec_mode_basis(ntheta=ntheta, nzeta=nzeta, nfp=1, mf=1, nf=1, lasym=False, wint=wint)
    gsource_full = np.arange(int(basis_sym["nuv_full"]), dtype=float)

    src = _vmec_source_from_gsource(gsource=gsource_full, basis=basis_sym)
    imirr_full = np.asarray(basis_sym["imirr_full"], dtype=np.int64)
    expected = 0.5 * float(basis_sym["onp"]) * (gsource_full[: int(basis_sym["nuv3"])] - gsource_full[imirr_full[: int(basis_sym["nuv3"])]])
    np.testing.assert_allclose(src, expected)

    with pytest.raises(ValueError, match="invalid_grpmn_shape"):
        _vmec_mode_matrix_from_grpmn(grpmn=np.zeros((0, int(basis_sym["nuv3"]))), basis=basis_sym)

    basis_asym = _build_vmec_mode_basis(ntheta=ntheta, nzeta=nzeta, nfp=1, mf=1, nf=1, lasym=True, wint=wint)
    mnpd = int(basis_asym["mnpd"])
    with pytest.raises(ValueError, match="invalid_grpmn_shape_lasym"):
        _vmec_mode_matrix_from_grpmn(grpmn=np.zeros((mnpd, int(basis_asym["nuv3"]))), basis=basis_asym)

    matrix = _vmec_mode_matrix_from_grpmn(grpmn=np.zeros((2 * mnpd, int(basis_asym["nuv3"]))), basis=basis_asym)
    assert matrix.shape == (2 * mnpd, 2 * mnpd)
    assert matrix[mnpd + int(basis_asym["mn0"]), mnpd + int(basis_asym["mn0"])] > matrix[int(basis_asym["mn0"]), int(basis_asym["mn0"])]

    mode_matrix = 2.0 * np.eye(2 * mnpd)
    rhs_mode = np.linspace(0.0, 1.0, 2 * mnpd)
    cache = NestorVmecLikeCache(
        ntheta=ntheta,
        nzeta=nzeta,
        matrix=np.eye(ntheta * nzeta),
        rhs_scale=np.ones(ntheta * nzeta),
        mode_basis=basis_asym,
        mode_matrix=mode_matrix,
    )
    phi, potvac, rhs_eff = _solve_vmec_like_mode_from_gsource(cache=cache, gsource=np.ones(ntheta * nzeta), rhs_mode=rhs_mode)
    assert phi.shape == (ntheta, nzeta)
    assert float(np.mean(phi)) == pytest.approx(0.0, abs=1.0e-14)
    np.testing.assert_allclose(potvac, 0.5 * rhs_mode)
    np.testing.assert_allclose(rhs_eff, rhs_mode)

    missing_cache = NestorVmecLikeCache(ntheta=1, nzeta=1, matrix=np.eye(1), rhs_scale=np.ones(1))
    with pytest.raises(ValueError, match="missing_mode_cache"):
        _solve_vmec_like_mode_from_gsource(cache=missing_cache, gsource=np.ones(1))


def test_dense_cache_solve_and_direct_vmec_like_cache_are_tiny_and_finite() -> None:
    matrix = np.asarray([[2.0, 0.0], [0.0, 2.0]])
    rhs = np.asarray([[4.0, 5.0]])
    cache = NestorVmecLikeCache(ntheta=1, nzeta=2, matrix=matrix, rhs_scale=np.asarray([1.0, 2.0]))
    phi = _solve_vmec_like_dense(rhs, cache)
    np.testing.assert_allclose(phi, np.asarray([[-1.5, 1.5]]))

    lu_fac = _dense_lu_factor(matrix)
    solved = _solve_vmec_like_dense(rhs, NestorVmecLikeCache(ntheta=1, nzeta=2, matrix=matrix, rhs_scale=np.ones(2), matrix_lu=lu_fac))
    assert solved.shape == (1, 2)

    sample = _analytic_sample()
    direct = _build_vmec_like_cache(
        sample,
        alpha=0.25,
        dist_eps=0.05,
        rhs_floor=1.0e-8,
        diag_coeff=1.0,
        row_sum_zero=True,
        singular_diag_scale=0.2,
        nfp=1,
        mf=1,
        nf=1,
        lasym=True,
        wint_vmec=np.full(sample.R.shape, 1.0 / float(sample.R.size)),
    )
    assert direct.matrix.shape == (sample.R.size, sample.R.size)
    assert direct.mode_matrix is not None
    assert direct.mode_matrix.shape == (2 * int(direct.mode_basis["mnpd"]), 2 * int(direct.mode_basis["mnpd"]))


def test_nonsingular_kernel_helpers_use_cache_and_tiny_surface_terms() -> None:
    sample = _analytic_sample()
    basis = _build_vmec_mode_basis(
        ntheta=sample.R.shape[0],
        nzeta=sample.R.shape[1],
        nfp=1,
        mf=1,
        nf=1,
        lasym=True,
        wint=np.full(sample.R.shape, 1.0 / float(sample.R.size)),
    )

    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=1)
    assert _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=1) is tables
    assert _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2) is not tables

    gsource = _vmec_nonsingular_gsource_from_bexni(
        sample=sample,
        basis=basis,
        bexni=np.ones(5),
        signgs=1,
        nvper=1,
    )
    assert gsource.shape == (int(basis["nuv_full"]),)
    assert np.isfinite(gsource).all()

    gstore, grpmn = _vmec_nonsingular_terms_from_bexni(
        sample=sample,
        basis=basis,
        bexni=np.linspace(1.0, 2.0, sample.R.size),
        signgs=-1,
        nvper=1,
    )
    assert gstore.shape == (int(basis["nuv_full"]),)
    assert grpmn.shape == (int(basis["mnpd2"]), int(basis["nuv3"]))
    assert np.isfinite(gstore).all()
    assert np.isfinite(grpmn).all()
    assert float(np.max(np.abs(grpmn))) > 0.0


def test_lasym_potvac_cosine_derivatives_and_decode_scalar_edges() -> None:
    sample = _analytic_sample(ntheta=4, nzeta=3)
    basis = _build_vmec_mode_basis(
        ntheta=4,
        nzeta=3,
        nfp=1,
        mf=1,
        nf=1,
        lasym=True,
        wint=np.full((4, 3), 1.0 / 12.0),
    )
    mnpd = int(basis["mnpd"])
    potvac = np.zeros(2 * mnpd)
    m1_idx = int(np.where(np.asarray(basis["xmpot"]) == 1)[0][0])
    potvac[mnpd + m1_idx] = 0.5

    changed = _vacuum_channels_from_sample_potvac(sample=sample, basis=basis, potvac=potvac)
    assert changed.bu.shape == sample.R.shape
    assert changed.bv.shape == sample.R.shape
    assert float(np.max(np.abs(changed.bu - sample.vac_ext.bu))) > 0.0

    assert _decode_char_scalar(np.asarray(" scalar ", dtype="U8")) == "scalar"
    assert _decode_char_rows(np.asarray([b"r", b"1", b" "], dtype="S1")) == ("r1",)

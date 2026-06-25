from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.free_boundary as freeb
from vmec_jax.config import FreeBoundaryConfig, VMECConfig
from vmec_jax.free_boundary import (
    ExternalBoundarySample,
    MGridData,
    MGridMetadata,
    NestorPoissonCache,
    NestorRuntimeState,
    NestorVmecLikeCache,
    VacuumBoundaryFields,
    _MGRID_FIELD_CACHE,
    _FREEB_HOST_PHASE_CACHE,
    _build_vmec_like_cache,
    _build_vmec_mode_basis,
    _decode_char_scalar,
    _dense_lu_factor,
    _dense_lu_solve,
    _freeb_host_phase_stack,
    _maybe_dump_scalpot_jax,
    _sample_external_boundary_arrays,
    _axis_current_field_vmec_filament,
    _vmec_analytic_bvec_from_geometry,
    _vmec_analytic_terms_from_geometry,
    _vmec_nonsingular_gsource_from_bexni,
    _vmec_nonsingular_terms_from_bexni,
    _vmec_source_from_gsource,
    interpolate_mgrid_bfield,
    nestor_external_only_step,
    prepare_mgrid_for_config,
    sample_external_vacuum_diagnostics,
)


def _vacuum(shape: tuple[int, int]) -> VacuumBoundaryFields:
    zeros = np.zeros(shape, dtype=float)
    ones = np.ones(shape, dtype=float)
    return VacuumBoundaryFields(
        bu=ones.copy(),
        bv=0.5 * ones,
        bsupu=ones.copy(),
        bsupv=0.5 * ones,
        bsqvac=0.625 * ones,
        bnormal=0.25 * ones,
        bnormal_unit=0.125 * ones,
        g_uu=ones.copy(),
        g_uv=zeros.copy(),
        g_vv=ones.copy(),
        det_guv=ones.copy(),
    )


def _sample(*, ntheta: int = 3, nzeta: int = 2, include_second: bool = True) -> ExternalBoundarySample:
    theta = (2.0 * np.pi / float(ntheta)) * np.arange(ntheta)
    zeta = (2.0 * np.pi / float(nzeta)) * np.arange(nzeta)
    th, ze = np.meshgrid(theta, zeta, indexing="ij")

    R = 2.0 + 0.12 * np.cos(th) + 0.04 * np.cos(ze)
    Z = 0.13 * np.sin(th) + 0.03 * np.sin(ze)
    Ru = -0.12 * np.sin(th)
    Zu = 0.13 * np.cos(th)
    Rv = -0.04 * np.sin(ze)
    Zv = 0.03 * np.cos(ze)
    zeros = np.zeros_like(R)
    second = {
        "ruu": -0.12 * np.cos(th),
        "ruv": zeros.copy(),
        "rvv": -0.04 * np.cos(ze),
        "zuu": -0.13 * np.sin(th),
        "zuv": zeros.copy(),
        "zvv": -0.03 * np.sin(ze),
    }
    if not include_second:
        second = dict.fromkeys(second, None)

    return ExternalBoundarySample(
        mgrid_path="synthetic.nc",
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=np.broadcast_to(zeta[None, :], R.shape),
        br=0.1 * np.ones_like(R),
        bp=0.2 * np.ones_like(R),
        bz=-0.3 * np.ones_like(R),
        br_mgrid=0.1 * np.ones_like(R),
        bp_mgrid=0.2 * np.ones_like(R),
        bz_mgrid=-0.3 * np.ones_like(R),
        br_axis=zeros.copy(),
        bp_axis=zeros.copy(),
        bz_axis=zeros.copy(),
        axis_r=np.full(nzeta, 1.1),
        axis_z=np.linspace(-0.1, 0.1, nzeta),
        vac_ext=_vacuum(R.shape),
        **second,
    )


def _mgrid(*, path: str = "cached_mgrid.nc", nextcur: int = 1, kp: int = 2) -> MGridData:
    meta = MGridMetadata(
        path=path,
        ir=2,
        jz=2,
        kp=kp,
        nfp=1,
        nextcur=nextcur,
        rmin=0.0,
        rmax=3.0,
        zmin=-1.0,
        zmax=1.0,
        mgrid_mode="S",
        coil_groups=tuple(f"coil_{i}" for i in range(nextcur)),
        raw_coil_cur=tuple(1.0 + i for i in range(nextcur)),
    )
    shape = (nextcur, kp, 2, 2)
    values = np.arange(max(1, int(np.prod(shape))), dtype=float).reshape(shape) if nextcur else np.zeros(shape)
    return MGridData(metadata=meta, br=values + 0.5, bp=values + 1.5, bz=values - 1.5)


def test_sample_external_boundary_arrays_cached_mgrid_axis_override_and_diagnostics(monkeypatch):
    data = _mgrid()
    _MGRID_FIELD_CACHE[data.metadata.path] = data
    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=2,
        nfp=1,
        lasym=False,
        lthreed=True,
        lconm1=False,
        ntheta=4,
        nzeta=2,
    )
    static = SimpleNamespace(
        cfg=cfg,
        modes=SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 0])),
        trig_vmec=None,
        m_is_even=np.asarray([True, False]),
        mgrid_metadata=data.metadata,
        free_boundary_extcur=(0.25,),
        signgs=-1,
    )
    state = SimpleNamespace(
        Rcos=np.asarray([[1.0, 0.0], [2.0, 0.15]]),
        Rsin=np.zeros((2, 2)),
        Zcos=np.zeros((2, 2)),
        Zsin=np.asarray([[0.0, 0.0], [0.0, 0.20]]),
    )

    monkeypatch.setenv("VMEC_JAX_FREEB_DISABLE_M1_CONVERSION", "1")
    monkeypatch.setenv("VMEC_JAX_FREEB_AXIS_MODE", "override_only")
    monkeypatch.setenv("VMEC_JAX_FREEB_AXIS_FIELD_MODE", "simple")
    monkeypatch.setenv("VMEC_JAX_DUMP_SCALPOT", "1")

    sample = _sample_external_boundary_arrays(
        state=state,
        static=static,
        axis_override=(np.asarray([1.0, 1.1]), np.asarray([0.0, 0.05])),
        plascur=0.2,
    )

    assert sample.R.shape == (3, 2)
    assert sample.axis_r_full is not None
    np.testing.assert_allclose(sample.axis_r, [1.0, 1.1])
    assert np.isfinite(sample.vac_ext.bsqvac).all()
    assert float(np.max(np.abs(sample.br_mgrid))) > 0.0

    monkeypatch.setattr(freeb, "_sample_external_boundary_arrays", lambda **_kwargs: sample)
    diagnostics = sample_external_vacuum_diagnostics(state=state, static=static, plascur=0.2)
    assert diagnostics["enabled"] is True
    assert diagnostics["available"] is True
    assert diagnostics["n_samples"] == int(sample.R.size)
    assert diagnostics["bmag_max"] >= diagnostics["bmag_mean"]

    _MGRID_FIELD_CACHE.pop(data.metadata.path, None)


def test_prepare_mgrid_for_config_validates_and_normalizes(monkeypatch):
    meta = replace(_mgrid(nextcur=3, kp=8).metadata, nfp=2)
    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=3,
        nfp=2,
        lasym=False,
        lthreed=True,
        lconm1=True,
        ntheta=4,
        nzeta=2,
        free_boundary=FreeBoundaryConfig(
            enabled=True,
            mgrid_file="dummy.nc",
            extcur=(5.0,),
            nvacskip=1,
        ),
    )

    monkeypatch.setattr(freeb, "load_mgrid", lambda path, *, load_fields=True: meta)
    prepared = prepare_mgrid_for_config(cfg, load_fields=False)
    assert prepared is not None
    assert prepared.extcur == (5.0, 0.0, 0.0)
    assert prepared.metadata.kp == 4

    data = _mgrid(nextcur=3, kp=8)
    data = replace(data, metadata=replace(data.metadata, nfp=2))
    monkeypatch.setattr(freeb, "load_mgrid", lambda path, *, load_fields=True: data)
    loaded = prepare_mgrid_for_config(cfg, load_fields=True)
    assert isinstance(loaded, MGridData)
    assert loaded.metadata.kp == 4

    monkeypatch.setattr(freeb, "load_mgrid", lambda path, *, load_fields=True: replace(meta, nfp=1))
    with pytest.raises(ValueError, match="does not match"):
        prepare_mgrid_for_config(cfg, load_fields=False)

    monkeypatch.setattr(freeb, "load_mgrid", lambda path, *, load_fields=True: replace(meta, kp=3))
    with pytest.raises(ValueError, match="divisible"):
        prepare_mgrid_for_config(cfg, load_fields=False)

    disabled = replace(cfg, free_boundary=FreeBoundaryConfig(enabled=False))
    assert prepare_mgrid_for_config(disabled, strict=False) is None


def test_interpolation_zero_current_groups_and_decode_scalar_edges():
    data = _mgrid(nextcur=0, kp=1)
    br, bp, bz = interpolate_mgrid_bfield(data, r=np.asarray([1.5]), z=0.0, phi=0.0)
    np.testing.assert_allclose(br, [0.0])
    np.testing.assert_allclose(bp, [0.0])
    np.testing.assert_allclose(bz, [0.0])

    assert _decode_char_scalar(np.asarray(b"mode_s  ", dtype="S8")) == "mode_s"
    assert _decode_char_scalar(np.asarray([1, 2, 3])) == "[1 2 3]"


def test_stale_phase_cache_lu_and_axis_filament_edge_branches(monkeypatch):
    import vmec_jax.solvers.free_boundary.jax_nestor_operator as jax_nestor_operator

    modes = SimpleNamespace(n=np.asarray([0]))
    trig = SimpleNamespace(cosnv=np.ones((1, 1)), cosmu=np.ones((1, 1)))
    key = (id(modes), id(trig), ("base",))
    _FREEB_HOST_PHASE_CACHE[key] = np.zeros((1, 1, 1, 1))
    try:
        with pytest.raises(AttributeError):
            _freeb_host_phase_stack(modes=modes, trig=trig, derivs=("base",))
    finally:
        _FREEB_HOST_PHASE_CACHE.pop(key, None)

    matrix = np.asarray([[2.0, 0.0], [0.0, 3.0]])
    rhs = np.asarray([4.0, 9.0])
    monkeypatch.setattr(jax_nestor_operator, "_SCIPY_LU_FACTOR", None)
    assert _dense_lu_factor(matrix) is None
    monkeypatch.setattr(
        jax_nestor_operator,
        "_SCIPY_LU_SOLVE",
        lambda lu_fac, rhs_arr: (_ for _ in ()).throw(RuntimeError("bad lu")),
    )
    np.testing.assert_allclose(_dense_lu_solve(("bad",), matrix, rhs), [2.0, 3.0])

    R = np.full((2, 2), 2.0)
    Z = np.zeros((2, 2))
    filament = _axis_current_field_vmec_filament(
        R=R,
        Z=Z,
        axis_r=np.asarray([0.5, 0.5]),
        axis_z=np.asarray([0.0, 0.1]),
        nfp=1,
        plascur=1.0,
    )
    assert all(arr.shape == R.shape for arr in filament)
    assert any(float(np.max(np.abs(arr))) > 0.0 for arr in filament)


def test_sample_external_boundary_arrays_axis_fallback_paths(monkeypatch):
    data = _mgrid(path="fallback_mgrid.nc")
    _MGRID_FIELD_CACHE[data.metadata.path] = data
    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=2,
        nfp=1,
        lasym=False,
        lthreed=True,
        lconm1=False,
        ntheta=4,
        nzeta=2,
    )
    static = SimpleNamespace(
        cfg=cfg,
        modes=SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 0])),
        trig_vmec=None,
        m_is_even=np.asarray([True, False]),
        mgrid_metadata=data.metadata,
        free_boundary_extcur=(0.25,),
        signgs=-1,
    )
    state = SimpleNamespace(
        Rcos=np.asarray([[1.0, 0.0], [2.0, 0.15]]),
        Rsin=np.zeros((2, 2)),
        Zcos=np.zeros((2, 2)),
        Zsin=np.asarray([[0.0, 0.0], [0.0, 0.20]]),
    )

    empty_path_static = SimpleNamespace(**{**static.__dict__, "mgrid_metadata": replace(data.metadata, path="")})
    with pytest.raises(ValueError, match="missing_mgrid_path"):
        _sample_external_boundary_arrays(state=state, static=empty_path_static)

    import vmec_jax.vmec_parity as parity_module

    monkeypatch.setattr(parity_module, "signed_maps_from_modes", lambda modes: (_ for _ in ()).throw(RuntimeError("no maps")))
    monkeypatch.setenv("VMEC_JAX_FREEB_DISABLE_M1_CONVERSION", "1")
    monkeypatch.setenv("VMEC_JAX_FREEB_AXIS_FROM_PARITY", "0")
    monkeypatch.setattr(
        freeb,
        "_axis_current_field_vmec_filament",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("filament failed")),
    )
    direct = _sample_external_boundary_arrays(state=state, static=static, plascur=0.1)
    assert direct.R.shape == (3, 2)
    assert np.isfinite(direct.vac_ext.bsqvac).all()

    monkeypatch.setenv("VMEC_JAX_FREEB_AXIS_MODE", "parity_only")
    monkeypatch.setenv("VMEC_JAX_FREEB_AXIS_FROM_PARITY", "1")
    monkeypatch.setenv("VMEC_JAX_FREEB_AXIS_FIELD_MODE", "simple")
    parity = _sample_external_boundary_arrays(state=state, static=static, plascur=0.1)
    assert parity.axis_r.shape == (2,)
    assert np.isfinite(parity.br_axis).all()

    _MGRID_FIELD_CACHE.pop(data.metadata.path, None)


def test_nonsingular_and_analytic_vmec_helpers_reduced_symmetric_branches():
    sample = _sample(ntheta=3, nzeta=2)
    np.testing.assert_array_equal(
        _vmec_nonsingular_gsource_from_bexni(
            sample=sample,
            basis={"nu_full": 0},
            bexni=np.asarray([]),
            signgs=1,
            nvper=1,
        ),
        np.zeros((0,), dtype=float),
    )
    empty_gstore, empty_grpmn = _vmec_nonsingular_terms_from_bexni(
        sample=sample,
        basis={"nu_full": 0},
        bexni=np.asarray([]),
        signgs=1,
        nvper=1,
    )
    assert empty_gstore.shape == (0,)
    assert empty_grpmn.shape == (0, 0)

    wint = np.full(sample.R.shape, 1.0 / float(sample.R.size))
    basis_sym = _build_vmec_mode_basis(
        ntheta=sample.R.shape[0],
        nzeta=sample.R.shape[1],
        nfp=1,
        mf=1,
        nf=1,
        lasym=False,
        wint=wint,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        gsource = _vmec_nonsingular_gsource_from_bexni(
            sample=sample,
            basis=basis_sym,
            bexni=np.asarray([1.0, -0.5]),
            signgs=-1,
            nvper=2,
        )
    assert gsource.shape == (int(basis_sym["nuv_full"]),)
    assert np.any(np.isfinite(gsource))

    bad_second = replace(
        sample,
        ruu=np.zeros((1, 1)),
        ruv=np.zeros((1, 1)),
        rvv=np.zeros((1, 1)),
        zuu=np.zeros((1, 1)),
        zuv=np.zeros((1, 1)),
        zvv=np.zeros((1, 1)),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        gstore, grpmn = _vmec_nonsingular_terms_from_bexni(
            sample=bad_second,
            basis=basis_sym,
            bexni=np.asarray([0.25]),
            signgs=1,
            nvper=2,
        )
    assert gstore.shape == (int(basis_sym["nuv_full"]),)
    assert grpmn.shape == (int(basis_sym["mnpd2"]), int(basis_sym["nuv3"]))
    assert np.any(np.isfinite(gstore))
    assert np.any(np.isfinite(grpmn))

    basis_asym = _build_vmec_mode_basis(
        ntheta=sample.R.shape[0],
        nzeta=sample.R.shape[1],
        nfp=1,
        mf=1,
        nf=1,
        lasym=True,
        wint=wint,
    )
    bvec_asym, grpmn_asym = _vmec_analytic_terms_from_geometry(
        sample=sample,
        basis=basis_asym,
        bexni=np.asarray([0.5]),
        signgs=-1,
    )
    assert bvec_asym.shape == (2 * int(basis_asym["mnpd"]),)
    assert grpmn_asym.shape == (2 * int(basis_asym["mnpd"]), int(basis_asym["nuv3"]))
    assert np.isfinite(bvec_asym).all()
    assert np.isfinite(grpmn_asym).all()

    sample_no_second = _sample(ntheta=3, nzeta=2, include_second=False)
    bvec_sym = _vmec_analytic_bvec_from_geometry(
        sample=sample_no_second,
        basis=basis_sym,
        bexni=np.ones(sample_no_second.R.size),
        signgs=1,
    )
    assert bvec_sym.shape == (int(basis_sym["mnpd"]),)
    assert np.isfinite(bvec_sym).all()


def test_vmec_like_cache_source_fallbacks_and_scalpot_dump(tmp_path, monkeypatch):
    sample = _sample(ntheta=2, nzeta=2)
    zero_weight_sample = replace(sample, phi=np.asarray(0.0), vac_ext=replace(sample.vac_ext, det_guv=np.zeros_like(sample.R)))
    cache = _build_vmec_like_cache(
        zero_weight_sample,
        alpha=0.1,
        dist_eps=0.05,
        rhs_floor=1.0e-6,
        diag_coeff=0.5,
        row_sum_zero=False,
        singular_diag_scale=0.0,
        nfp=1,
        mf=1,
        nf=0,
        lasym=False,
    )
    assert cache.matrix.shape == (sample.R.size, sample.R.size)
    np.testing.assert_allclose(cache.rhs_scale, np.full(sample.R.size, 1.0 / sample.R.size))

    basis_asym = _build_vmec_mode_basis(ntheta=2, nzeta=2, nfp=1, mf=1, nf=1, lasym=True, wint=np.full((2, 2), 0.25))
    short_src = _vmec_source_from_gsource(gsource=np.asarray([3.0]), basis=basis_asym)
    np.testing.assert_allclose(short_src, [3.0])

    monkeypatch.setenv("VMEC_JAX_DUMP_SCALPOT", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "4")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    dense_cache = NestorVmecLikeCache(
        ntheta=2,
        nzeta=2,
        matrix=np.eye(4),
        rhs_scale=np.ones(4),
    )
    dump_sample = replace(
        sample,
        axis_r_full=np.asarray([1.0, 1.1]),
        axis_z_full=np.asarray([0.0, 0.1]),
        axis_r_parity=np.asarray([0.9, 1.0]),
        axis_z_parity=np.asarray([0.1, 0.2]),
    )
    rhs = np.arange(4.0).reshape(2, 2)
    phi = np.zeros((2, 2), dtype=float)

    _maybe_dump_scalpot_jax(
        iter_idx=None,
        ivac=0,
        reused=False,
        mode="dense",
        rhs=rhs,
        phi=phi,
        vac=sample.vac_ext,
        cache=dense_cache,
        sample=dump_sample,
        mf=1,
        nf=1,
        nfp=1,
        lasym=True,
    )
    assert not (tmp_path / "scalpot_jax_iter0.npz").exists()

    _maybe_dump_scalpot_jax(
        iter_idx=4,
        ivac=1,
        reused=False,
        mode="vmec2000_like_dense_integral",
        rhs=rhs,
        phi=phi,
        vac=sample.vac_ext,
        cache=dense_cache,
        sample=dump_sample,
        mf=1,
        nf=1,
        nfp=1,
        lasym=True,
        wint_vmec=np.full((2, 2), 0.25),
        gsource_vmec=np.ones(4),
        potvac=np.arange(12.0),
        bvec_mode=np.arange(12.0),
        bvec_mode_nonsing=np.arange(12.0) + 1.0,
        bvec_mode_analytic=np.arange(12.0) + 2.0,
        source_cache_iter=7,
        matrix_override_applied=True,
        amatrix_mode_pre=np.eye(12),
        amatrix_mode_from_grpmn=2.0 * np.eye(12),
        grpmn_nonsing=np.ones((12, 4)),
        grpmn_analytic=2.0 * np.ones((12, 4)),
        grpmn_total=3.0 * np.ones((12, 4)),
        plascur=1.25,
    )
    with np.load(tmp_path / "scalpot_jax_iter4.npz") as dump:
        assert str(dump["cache_kind"]) == "dense"
        assert dump["axis_r_full"].shape == (2,)
        assert dump["bvec_mode_cos"].shape == (6,)
        assert dump["gsource_kernel"].shape == (2, 2)
        assert int(dump["matrix_override_applied"]) == 1

    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "5")
    _maybe_dump_scalpot_jax(
        iter_idx=5,
        ivac=2,
        reused=True,
        mode="spectral_poisson_external_only",
        rhs=rhs,
        phi=phi,
        vac=sample.vac_ext,
        cache=NestorPoissonCache(ntheta=2, nzeta=2, lam=np.ones((2, 2))),
        sample=sample,
        mf=0,
        nf=0,
        nfp=1,
        lasym=False,
    )
    with np.load(tmp_path / "scalpot_jax_iter5.npz") as dump:
        assert str(dump["cache_kind"]) == "spectral"
        assert dump["lam"].shape == (2, 2)

    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "6")
    _maybe_dump_scalpot_jax(
        iter_idx=6,
        ivac=3,
        reused=False,
        mode="dense_no_mode_basis",
        rhs=rhs,
        phi=phi,
        vac=sample.vac_ext,
        cache=dense_cache,
        sample=sample,
        mf=1,
        nf=0,
        nfp=1,
        lasym=False,
    )
    with np.load(tmp_path / "scalpot_jax_iter6.npz") as dump:
        assert str(dump["cache_kind"]) == "dense"
        assert dump["bvec_mode_sin"].shape == (2,)
        assert dump["amatrix_mode"].shape == (2, 2)

    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "7")
    _maybe_dump_scalpot_jax(
        iter_idx=7,
        ivac=4,
        reused=False,
        mode="unknown_cache",
        rhs=rhs,
        phi=phi,
        vac=sample.vac_ext,
        cache=object(),
        sample=sample,
        mf=0,
        nf=0,
        nfp=1,
        lasym=False,
    )
    with np.load(tmp_path / "scalpot_jax_iter7.npz") as dump:
        assert str(dump["cache_kind"]) == "unknown"


def test_nestor_external_only_step_reuse_spectral_and_dense_fallback(monkeypatch):
    runtime = NestorRuntimeState(
        operator_cache=NestorPoissonCache(ntheta=2, nzeta=2, lam=np.ones((2, 2))),
        phi=np.ones((2, 2)),
        bsqvac=2.0 * np.ones((2, 2)),
        mode="spectral_poisson_external_only",
        update_count=2,
        reuse_count=3,
        source_cache_iter=9,
        gsource_cached=np.ones(4),
        source_sym_cached=2.0 * np.ones(4),
        bvec_nonsing_cached=3.0 * np.ones(4),
    )
    monkeypatch.setenv("VMEC_JAX_FREEB_REUSE_RHS_UPDATE", "0")
    result, runtime_next = nestor_external_only_step(state=object(), static=object(), ivac=0, runtime=runtime)
    assert result.reused is True
    assert runtime_next.reuse_count == 4
    np.testing.assert_allclose(runtime_next.gsource_cached, np.ones(4))

    monkeypatch.delenv("VMEC_JAX_FREEB_REUSE_RHS_UPDATE", raising=False)
    sample = _sample(ntheta=2, nzeta=2)
    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lthreed=True,
        lconm1=True,
        ntheta=2,
        nzeta=2,
    )
    static = SimpleNamespace(cfg=cfg, trig_vmec=None, signgs=-1)
    monkeypatch.setattr(freeb, "_sample_external_boundary_arrays", lambda **_kwargs: sample)
    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "fast")
    monkeypatch.setenv("VMEC_JAX_FREEB_RHS_MODE", "bnormal")
    result_update, runtime_update = nestor_external_only_step(state=object(), static=static, ivac=1, iter_idx=10)
    assert result_update.reused is False
    assert result_update.model == "spectral_poisson_external_only"
    assert runtime_update.update_count == 1

    result_reuse, runtime_reuse = nestor_external_only_step(
        state=object(),
        static=static,
        ivac=0,
        ivacskip=1,
        runtime=runtime_update,
        iter_idx=11,
    )
    assert result_reuse.reused is True
    assert runtime_reuse.reuse_count == 1

    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "dense")
    monkeypatch.setattr(freeb, "_build_vmec_like_cache", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    result_fallback, _ = nestor_external_only_step(state=object(), static=static, ivac=1, iter_idx=12)
    assert result_fallback.model == "spectral_poisson_external_only_fallback:dense_failed"


def test_nestor_external_only_step_dense_grid_and_mode_success(monkeypatch):
    sample = _sample(ntheta=2, nzeta=2)
    cfg = VMECConfig(
        mpol=1,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lthreed=True,
        lconm1=True,
        ntheta=2,
        nzeta=2,
    )
    static = SimpleNamespace(cfg=cfg, trig_vmec=None, signgs=-1)
    monkeypatch.setattr(freeb, "_sample_external_boundary_arrays", lambda **_kwargs: sample)
    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "dense")
    monkeypatch.setenv("VMEC_JAX_FREEB_USE_GREENF_SOURCE", "no")

    monkeypatch.setenv("VMEC_JAX_FREEB_DENSE_SOLVE_MODE", "grid")
    grid_result, grid_runtime = nestor_external_only_step(state=object(), static=static, ivac=1, iter_idx=20)
    assert grid_result.model == "vmec2000_like_dense_integral"
    assert isinstance(grid_runtime.operator_cache, NestorVmecLikeCache)
    assert grid_result.phi.shape == sample.R.shape

    monkeypatch.setenv("VMEC_JAX_FREEB_DENSE_SOLVE_MODE", "mode")
    monkeypatch.setenv("VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC", "0")
    mode_result, mode_runtime = nestor_external_only_step(state=object(), static=static, ivac=1, iter_idx=21)
    assert mode_result.model == "vmec2000_like_dense_integral"
    assert isinstance(mode_runtime.bvec_nonsing_cached, np.ndarray)
    assert mode_runtime.source_cache_iter == 21

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.config import FreeBoundaryConfig, VMECConfig
from vmec_jax.free_boundary import (
    ExternalBoundarySample,
    MGridData,
    MGridMetadata,
    VacuumBoundaryFields,
    _as_float_env,
    _as_int_env,
    _axis_current_field_simple,
    _axis_current_field_vmec_filament,
    _base_nestor_mode,
    _broadcast_xyz,
    _build_vmec_mode_basis,
    _decode_char_rows,
    _decode_char_scalar,
    _dense_lu_solve,
    _freeb_use_greenf_source,
    _is_dense_mode,
    _is_spectral_mode,
    _normalize_extcur,
    _parse_iter_list_env,
    _select_nestor_mode,
    _solve_periodic_poisson_fft,
    _spectral_grad,
    _vacuum_channels_from_sample_phi,
    _vacuum_channels_from_sample_potvac,
    _vmec_realspace_synthesis_multi_host,
    boundary_metric_from_rz,
    contravariant_boundary_field_from_covariant,
    covariant_boundary_field_from_cylindrical,
    initial_free_boundary_state,
    interpolate_mgrid_bfield,
    load_mgrid,
    sample_external_vacuum_diagnostics,
    validate_free_boundary_config,
    vacuum_boundary_fields_from_cylindrical,
)


def _cfg(*, enabled: bool = True, mgrid_file: str = "mgrid.nc", nvacskip: int = 3) -> VMECConfig:
    return VMECConfig(
        mpol=3,
        ntor=1,
        ns=7,
        nfp=2,
        lasym=False,
        lthreed=True,
        lconm1=True,
        ntheta=8,
        nzeta=4,
        free_boundary=FreeBoundaryConfig(
            enabled=enabled,
            mgrid_file=mgrid_file,
            extcur=(1.5,),
            nvacskip=nvacskip,
        ),
    )


def _mgrid_data(*, raw_cur: tuple[float, ...] = (2.0, -1.0), kp: int = 2) -> MGridData:
    meta = MGridMetadata(
        path="/tmp/mgrid_synthetic.nc",
        ir=2,
        jz=2,
        kp=kp,
        nfp=1,
        nextcur=2,
        rmin=1.0,
        rmax=3.0,
        zmin=-1.0,
        zmax=1.0,
        mgrid_mode="S",
        coil_groups=("a", "b"),
        raw_coil_cur=raw_cur,
    )
    shape = (2, kp, 2, 2)
    br = np.zeros(shape, dtype=float)
    bp = np.zeros(shape, dtype=float)
    bz = np.zeros(shape, dtype=float)
    for coil in range(2):
        for k in range(kp):
            for j in range(2):
                for i in range(2):
                    base = 100.0 * coil + 10.0 * k + 2.0 * j + i
                    br[coil, k, j, i] = base
                    bp[coil, k, j, i] = base + 1000.0
                    bz[coil, k, j, i] = base - 1000.0
    return MGridData(metadata=meta, br=br, bp=bp, bz=bz)


def test_free_boundary_config_validation_and_runtime_state_policy():
    validate_free_boundary_config(_cfg(enabled=False, mgrid_file="", nvacskip=0), strict=True)
    validate_free_boundary_config(_cfg(enabled=True, mgrid_file="NONE"), strict=False)
    with pytest.raises(ValueError, match="MGRID_FILE"):
        validate_free_boundary_config(_cfg(enabled=True, mgrid_file="NONE"), strict=True)
    with pytest.raises(ValueError, match="nvacskip"):
        validate_free_boundary_config(_cfg(enabled=True, mgrid_file="mgrid.nc", nvacskip=0), strict=True)

    state = initial_free_boundary_state(_cfg(nvacskip=5))
    assert (state.ivac, state.ivacskip, state.nvacskip, state.nvskip0) == (0, 0, 5, 5)
    assert initial_free_boundary_state(_cfg(nvacskip=0)).nvskip0 == 1

    assert _normalize_extcur((1.0,), 3) == (1.0, 0.0, 0.0)
    assert _normalize_extcur((1.0, 2.0, 3.0), 2) == (1.0, 2.0)
    assert _normalize_extcur((1.0,), 0) == ()


def test_mgrid_interpolation_clamps_periodizes_and_validates_shapes():
    data = _mgrid_data()
    br, bp, bz = interpolate_mgrid_bfield(
        data,
        r=np.array([[2.0]]),
        z=np.array([[0.0]]),
        phi=np.array([[0.5 * np.pi]]),
    )
    # Midpoint in R/Z and halfway between the two toroidal planes.
    k0_midpoint = 0.5 * (0.5 * (0.0 + 1.0) + 0.5 * (2.0 + 3.0))
    k1_midpoint = 0.5 * (0.5 * (10.0 + 11.0) + 0.5 * (12.0 + 13.0))
    expected_br_coil0 = 0.5 * k0_midpoint + 0.5 * k1_midpoint
    expected_br_coil1 = expected_br_coil0 + 100.0
    expected_br = 2.0 * expected_br_coil0 - expected_br_coil1
    assert br.shape == (1, 1)
    assert float(br[0, 0]) == pytest.approx(expected_br)
    assert float(bp[0, 0]) == pytest.approx(expected_br + 1000.0)
    assert float(bz[0, 0]) == pytest.approx(expected_br - 1000.0)

    periodic, _, _ = interpolate_mgrid_bfield(data, r=99.0, z=-99.0, phi=2.0 * np.pi)
    clamped, _, _ = interpolate_mgrid_bfield(data, r=3.0, z=-1.0, phi=0.0)
    np.testing.assert_allclose(periodic, clamped)

    data_kv = _mgrid_data(kp=6)
    vmec_kv, _, _ = interpolate_mgrid_bfield(
        data_kv,
        r=np.array([[1.0, 1.0, 1.0]]),
        z=np.array([[-1.0, -1.0, -1.0]]),
        phi=np.zeros((1, 3)),
        use_vmec_kv=True,
    )
    np.testing.assert_allclose(vmec_kv, np.array([[-100.0, -80.0, -60.0]]))
    with pytest.raises(ValueError, match="must be divisible"):
        interpolate_mgrid_bfield(
            data,
            r=np.array([[1.0, 1.0, 1.0]]),
            z=np.array([[-1.0, -1.0, -1.0]]),
            phi=np.zeros((1, 3)),
            use_vmec_kv=True,
        )
    with pytest.raises(ValueError, match="explicit zeta axis"):
        interpolate_mgrid_bfield(data, r=1.0, z=0.0, phi=0.0, use_vmec_kv=True)

    bad_meta = replace(data.metadata, ir=1)
    with pytest.raises(ValueError, match="dimensions too small"):
        interpolate_mgrid_bfield(replace(data, metadata=bad_meta), r=1.0, z=0.0, phi=0.0)

    bad_shape = replace(data, br=data.br[:, :, :, :1])
    with pytest.raises(ValueError, match="field shape"):
        interpolate_mgrid_bfield(bad_shape, r=1.0, z=0.0, phi=0.0)


def test_mgrid_interpolation_uses_unit_current_when_raw_currents_are_absent():
    meta = MGridMetadata(
        path="dummy.nc",
        ir=2,
        jz=2,
        kp=1,
        nfp=1,
        nextcur=1,
        rmin=0.0,
        rmax=1.0,
        zmin=0.0,
        zmax=1.0,
        mgrid_mode="S",
        coil_groups=("A",),
        raw_coil_cur=(),
    )
    data = MGridData(
        metadata=meta,
        br=np.full((1, 1, 2, 2), 4.0),
        bp=np.full((1, 1, 2, 2), -2.0),
        bz=np.full((1, 1, 2, 2), 0.5),
    )

    br, bp, bz = interpolate_mgrid_bfield(data, r=0.5, z=0.5, phi=0.0)

    assert float(br) == pytest.approx(4.0)
    assert float(bp) == pytest.approx(-2.0)
    assert float(bz) == pytest.approx(0.5)

    bad_bounds = replace(data, metadata=replace(meta, rmax=meta.rmin))
    with pytest.raises(ValueError, match="Invalid mgrid bounds"):
        interpolate_mgrid_bfield(bad_bounds, r=0.5, z=0.5, phi=0.0)


def test_mgrid_interpolation_explicit_empty_extcur_zeroes_raw_current_field():
    data = _mgrid_data(raw_cur=(2.0, -1.0))

    br_zero, bp_zero, bz_zero = interpolate_mgrid_bfield(data, r=2.0, z=0.0, phi=0.0, extcur=())
    np.testing.assert_allclose(br_zero, 0.0)
    np.testing.assert_allclose(bp_zero, 0.0)
    np.testing.assert_allclose(bz_zero, 0.0)

    br_first, bp_first, bz_first = interpolate_mgrid_bfield(data, r=2.0, z=0.0, phi=0.0, extcur=(3.0,))
    assert float(abs(br_first)) > 0.0
    assert float(abs(bp_first)) > 0.0
    assert float(abs(bz_first)) > 0.0


def test_boundary_metric_field_projection_and_degenerate_determinant_floor():
    R = np.array([[2.0]])
    Ru = np.array([[1.0]])
    Zu = np.array([[0.0]])
    Rv = np.array([[0.0]])
    Zv = np.array([[0.0]])
    br = np.array([[3.0]])
    bp = np.array([[5.0]])
    bz = np.array([[7.0]])

    g_uu, g_uv, g_vv, det = boundary_metric_from_rz(R=R, Ru=Ru, Zu=Zu, Rv=Rv, Zv=Zv)
    np.testing.assert_allclose(g_uu, [[1.0]])
    np.testing.assert_allclose(g_uv, [[0.0]])
    np.testing.assert_allclose(g_vv, [[4.0]])
    np.testing.assert_allclose(det, [[4.0]])

    bu, bv = covariant_boundary_field_from_cylindrical(br=br, bp=bp, bz=bz, R=R, Ru=Ru, Zu=Zu, Rv=Rv, Zv=Zv)
    np.testing.assert_allclose(bu, [[3.0]])
    np.testing.assert_allclose(bv, [[10.0]])

    bsupu, bsupv, det2 = contravariant_boundary_field_from_covariant(bu=bu, bv=bv, g_uu=g_uu, g_uv=g_uv, g_vv=g_vv)
    np.testing.assert_allclose(bsupu, [[3.0]])
    np.testing.assert_allclose(bsupv, [[2.5]])
    np.testing.assert_allclose(det2, [[4.0]])

    vac = vacuum_boundary_fields_from_cylindrical(br=br, bp=bp, bz=bz, R=R, Ru=Ru, Zu=Zu, Rv=Rv, Zv=Zv)
    np.testing.assert_allclose(vac.bsqvac, [[17.0]])
    np.testing.assert_allclose(vac.bnormal, [[14.0]])
    np.testing.assert_allclose(vac.bnormal_unit, [[7.0]])

    bsupu_floor, bsupv_floor, det_floor = contravariant_boundary_field_from_covariant(
        bu=np.array([[1.0]]),
        bv=np.array([[2.0]]),
        g_uu=np.zeros((1, 1)),
        g_uv=np.zeros((1, 1)),
        g_vv=np.zeros((1, 1)),
        det_floor=1.0e-6,
    )
    np.testing.assert_allclose(det_floor, [[0.0]])
    assert np.isfinite(bsupu_floor).all()
    assert np.isfinite(bsupv_floor).all()


def test_host_realspace_synthesis_constant_mode_and_shape_validation():
    modes = SimpleNamespace(m=np.asarray([0]), n=np.asarray([0]))
    trig = SimpleNamespace(
        cosmu=np.ones((3, 1)),
        sinmu=np.zeros((3, 1)),
        cosmum=np.zeros((3, 1)),
        sinmum=np.zeros((3, 1)),
        cosnv=np.ones((2, 1)),
        sinnv=np.zeros((2, 1)),
        cosnvn=np.zeros((2, 1)),
        sinnvn=np.zeros((2, 1)),
    )

    base, dtheta, dzeta = _vmec_realspace_synthesis_multi_host(
        coeff_cos=np.asarray([[2.0]]),
        coeff_sin=np.asarray([[5.0]]),
        modes=modes,
        trig=trig,
        derivs=("base", "dtheta", "dzeta"),
    )
    base_cached, dtheta_cached, dzeta_cached = _vmec_realspace_synthesis_multi_host(
        coeff_cos=np.asarray([[2.0]]),
        coeff_sin=np.asarray([[5.0]]),
        modes=modes,
        trig=trig,
        derivs=("base", "dtheta", "dzeta"),
    )

    assert base.shape == (1, 3, 2)
    np.testing.assert_allclose(base, 2.0)
    np.testing.assert_allclose(dtheta, 0.0)
    np.testing.assert_allclose(dzeta, 0.0)
    np.testing.assert_allclose(base_cached, base)
    np.testing.assert_allclose(dtheta_cached, dtheta)
    np.testing.assert_allclose(dzeta_cached, dzeta)

    with pytest.raises(ValueError, match="same shape"):
        _vmec_realspace_synthesis_multi_host(
            coeff_cos=np.asarray([[1.0]]),
            coeff_sin=np.asarray([[1.0, 2.0]]),
            modes=modes,
            trig=trig,
        )


def test_axis_current_helpers_zero_current_and_validation_paths():
    R = np.full((2, 2), 2.0)
    Z = np.zeros((2, 2))
    phi = np.zeros((2, 2))
    axis_r = np.ones(2)
    axis_z = np.zeros(2)

    for arr in _axis_current_field_simple(R=R, Z=Z, phi=phi, axis_r=axis_r, axis_z=axis_z, nfp=1, plascur=0.0):
        np.testing.assert_allclose(arr, 0.0)
    for arr in _axis_current_field_vmec_filament(R=R, Z=Z, axis_r=axis_r, axis_z=axis_z, nfp=1, plascur=0.0):
        np.testing.assert_allclose(arr, 0.0)

    with pytest.raises(ValueError, match="R/Z/phi"):
        _axis_current_field_simple(R=R[0], Z=Z, phi=phi, axis_r=axis_r, axis_z=axis_z, nfp=1, plascur=1.0)
    with pytest.raises(ValueError, match="axis arrays"):
        _axis_current_field_simple(R=R, Z=Z, phi=phi, axis_r=axis_r[:1], axis_z=axis_z, nfp=1, plascur=1.0)
    with pytest.raises(ValueError, match="R/Z"):
        _axis_current_field_vmec_filament(R=R[0], Z=Z, axis_r=axis_r, axis_z=axis_z, nfp=1, plascur=1.0)
    with pytest.raises(ValueError, match="axis arrays"):
        _axis_current_field_vmec_filament(R=R, Z=Z, axis_r=axis_r[:1], axis_z=axis_z, nfp=1, plascur=1.0)


def test_env_parsing_mode_selection_and_greenf_policy(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_FREEB_USE_GREENF_SOURCE", raising=False)
    assert _freeb_use_greenf_source(0) is True
    monkeypatch.setenv("VMEC_JAX_FREEB_USE_GREENF_SOURCE", "no")
    assert _freeb_use_greenf_source(3) is False
    monkeypatch.setenv("VMEC_JAX_FREEB_USE_GREENF_SOURCE", "yes")
    assert _freeb_use_greenf_source(3) is True

    monkeypatch.setenv("VMEC_JAX_INT_TEST", "bad")
    monkeypatch.setenv("VMEC_JAX_FLOAT_TEST", "bad")
    assert _as_int_env("VMEC_JAX_INT_TEST", 7) == 7
    assert _as_float_env("VMEC_JAX_FLOAT_TEST", 2.5) == 2.5
    monkeypatch.setenv("VMEC_JAX_ITER_TEST", "1, bad; 3,,4")
    assert _parse_iter_list_env("VMEC_JAX_ITER_TEST") == {1, 3, 4}

    assert _base_nestor_mode("vmec2000_like_dense_integral_fallback:x") == "vmec2000_like_dense_integral"
    assert _is_dense_mode("vmec2000_like_dense_integral_fallback:x") is True
    assert _is_spectral_mode("spectral_poisson_external_only_fallback:x") is True

    monkeypatch.setenv("VMEC_JAX_FREEB_VMEC_LIKE_MAX_POINTS", "4")
    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "dense")
    assert _select_nestor_mode(ntheta=2, nzeta=2) == ("vmec2000_like_dense_integral", "forced_vmec_like")
    mode, reason = _select_nestor_mode(ntheta=3, nzeta=2)
    assert mode == "spectral_poisson_external_only"
    assert reason == "fallback_max_points:6>4"

    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "fast")
    assert _select_nestor_mode(ntheta=100, nzeta=100) == ("spectral_poisson_external_only", "forced_fast")
    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "auto")
    assert _select_nestor_mode(ntheta=2, nzeta=2) == ("vmec2000_like_dense_integral", "auto_vmec_like")


def test_vacuum_channel_helpers_apply_scalar_potential_modes():
    shape = (4, 4)
    zeros = np.zeros(shape)
    ones = np.ones(shape)
    vac_ext = VacuumBoundaryFields(
        bu=ones.copy(),
        bv=2.0 * ones,
        bsupu=ones.copy(),
        bsupv=2.0 * ones,
        bsqvac=2.5 * ones,
        bnormal=zeros.copy(),
        bnormal_unit=zeros.copy(),
        g_uu=ones.copy(),
        g_uv=zeros.copy(),
        g_vv=ones.copy(),
        det_guv=ones.copy(),
    )
    sample = ExternalBoundarySample(
        mgrid_path="dummy.nc",
        R=ones.copy(),
        Z=zeros.copy(),
        Ru=ones.copy(),
        Zu=zeros.copy(),
        Rv=zeros.copy(),
        Zv=zeros.copy(),
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
        axis_r=np.ones(shape[1]),
        axis_z=np.zeros(shape[1]),
        vac_ext=vac_ext,
    )

    unchanged = _vacuum_channels_from_sample_phi(sample, np.full(shape, 3.0))
    np.testing.assert_allclose(unchanged.bu, vac_ext.bu)
    np.testing.assert_allclose(unchanged.bv, vac_ext.bv)

    basis = _build_vmec_mode_basis(
        ntheta=shape[0],
        nzeta=shape[1],
        nfp=1,
        mf=1,
        nf=0,
        lasym=True,
        wint=np.full(shape, 1.0 / float(np.prod(shape))),
    )
    mnpd = int(basis["mnpd"])
    potvac = np.zeros(2 * mnpd)
    m1_idx = int(np.where(np.asarray(basis["xmpot"]) == 1)[0][0])
    potvac[m1_idx] = 0.25
    changed = _vacuum_channels_from_sample_potvac(sample=sample, basis=basis, potvac=potvac)

    assert changed.bu.shape == shape
    assert changed.bv.shape == shape
    assert np.max(np.abs(changed.bu - vac_ext.bu)) > 0.0
    np.testing.assert_allclose(changed.bnormal, vac_ext.bnormal)

    with pytest.raises(ValueError, match="potvac_too_small"):
        _vacuum_channels_from_sample_potvac(sample=sample, basis=basis, potvac=np.zeros(mnpd - 1))


def test_small_numeric_and_decode_helpers():
    rr, zz, pp = _broadcast_xyz(np.array([1.0, 2.0]), 0.5, np.array([[0.0], [1.0]]))
    assert rr.shape == (2, 2)
    assert zz.shape == (2, 2)
    assert pp.shape == (2, 2)

    assert _decode_char_scalar(np.frombuffer(b"AB ", dtype="S1")) == "AB"
    assert _decode_char_scalar(np.array(["x", "y", " "], dtype="U1")) == "xy"
    assert _decode_char_rows(np.array([[b"a", b" "], [b"b", b"c"]], dtype="S1")) == ("a", "bc")
    assert _decode_char_rows(np.array(5)) == ()

    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([9.0, 8.0])
    np.testing.assert_allclose(_dense_lu_solve(None, matrix, rhs), np.linalg.solve(matrix, rhs))

    from vmec_jax.free_boundary import _build_poisson_cache

    cache = _build_poisson_cache(ntheta=4, nzeta=4)
    rhs_grid = np.arange(16, dtype=float).reshape(4, 4)
    phi = _solve_periodic_poisson_fft(rhs_grid, cache)
    assert float(np.mean(phi)) == pytest.approx(0.0, abs=1.0e-14)
    du, dv = _spectral_grad(np.ones((4, 4)))
    np.testing.assert_allclose(du, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(dv, 0.0, atol=1.0e-14)


def test_sample_external_vacuum_diagnostics_reports_sampling_failures():
    out = sample_external_vacuum_diagnostics(
        state=SimpleNamespace(),
        static=SimpleNamespace(mgrid_metadata=None),
    )

    assert out["enabled"] is False
    assert out["available"] is False
    assert out["vacuum_stub"] is True
    assert out["reason"] == "sample_failed"
    assert "missing_mgrid_metadata" in out["error"]
    assert float(out["sample_time_s"]) >= 0.0


def test_load_mgrid_reports_missing_metadata_and_bad_field_shape(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    missing = tmp_path / "mgrid_missing.nc"
    with netCDF4.Dataset(str(missing), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createVariable("ir", "i4", ()).assignValue(2)
    with pytest.raises(KeyError, match="Missing mgrid variable: jz"):
        load_mgrid(missing, load_fields=False)

    bad_shape = tmp_path / "mgrid_bad_shape.nc"
    with netCDF4.Dataset(str(bad_shape), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("rad", 2)
        ds.createDimension("zee", 2)
        ds.createDimension("phi", 2)
        ds.createDimension("bad_phi", 3)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 2),
            ("nfp", 1),
            ("nextcur", 1),
        ):
            ds.createVariable(name, "i4", ()).assignValue(value)
        for name, value in (("rmin", 0.0), ("rmax", 1.0), ("zmin", 0.0), ("zmax", 1.0)):
            ds.createVariable(name, "f8", ()).assignValue(value)
        ds.createVariable("br_001", "f8", ("bad_phi", "zee", "rad"))[:] = 0.0

    with pytest.raises(ValueError, match="br_001 shape"):
        load_mgrid(bad_shape, load_fields=True)

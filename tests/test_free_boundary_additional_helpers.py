from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vmec_jax.config import FreeBoundaryConfig, VMECConfig
from vmec_jax.free_boundary import (
    MGridData,
    MGridMetadata,
    _as_float_env,
    _as_int_env,
    _base_nestor_mode,
    _broadcast_xyz,
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
    boundary_metric_from_rz,
    contravariant_boundary_field_from_covariant,
    covariant_boundary_field_from_cylindrical,
    initial_free_boundary_state,
    interpolate_mgrid_bfield,
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

    vmec_kv, _, _ = interpolate_mgrid_bfield(
        data,
        r=np.array([[1.0, 1.0, 1.0]]),
        z=np.array([[-1.0, -1.0, -1.0]]),
        phi=np.zeros((1, 3)),
        use_vmec_kv=True,
    )
    np.testing.assert_allclose(vmec_kv, np.array([[-100.0, -90.0, -90.0]]))
    with pytest.raises(ValueError, match="explicit zeta axis"):
        interpolate_mgrid_bfield(data, r=1.0, z=0.0, phi=0.0, use_vmec_kv=True)

    bad_meta = replace(data.metadata, ir=1)
    with pytest.raises(ValueError, match="dimensions too small"):
        interpolate_mgrid_bfield(replace(data, metadata=bad_meta), r=1.0, z=0.0, phi=0.0)

    bad_shape = replace(data, br=data.br[:, :, :, :1])
    with pytest.raises(ValueError, match="field shape"):
        interpolate_mgrid_bfield(bad_shape, r=1.0, z=0.0, phi=0.0)


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

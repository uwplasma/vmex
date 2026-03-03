from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import config_from_indata
from vmec_jax.free_boundary import (
    MGridData,
    MGridMetadata,
    PreparedMGrid,
    boundary_metric_from_rz,
    covariant_boundary_field_from_cylindrical,
    contravariant_boundary_field_from_covariant,
    interpolate_mgrid_bfield,
    load_mgrid,
    prepare_mgrid_for_config,
    vacuum_boundary_fields_from_cylindrical,
)
from vmec_jax.namelist import read_indata
from vmec_jax.static import build_static
from vmec_jax.driver import run_fixed_boundary


def test_free_boundary_config_vmec2000_defaults(tmp_path: Path):
    txt = """
&INDATA
  NFP = 3
  MPOL = 5
  NTOR = 2
  NS = 11
  LASYM = F
  LFREEB = T
  MGRID_FILE = 'NONE'
  NVACSKIP = 0
/
"""
    p = tmp_path / "input.fb_defaults"
    p.write_text(txt)
    indata = read_indata(p)
    cfg = config_from_indata(indata)

    # VMEC read_indata/readin behavior.
    assert cfg.lfreeb is False
    assert cfg.mgrid_file.upper() == "NONE"
    assert cfg.nvacskip == 3
    assert cfg.extcur == ()

    static = build_static(cfg)
    assert static.free_boundary_state0 is None


def test_free_boundary_config_extcur_indexed(tmp_path: Path):
    txt = """
&INDATA
  NFP = 2
  MPOL = 5
  NTOR = 1
  NS = 9
  LFREEB = T
  MGRID_FILE = 'mgrid_test.nc'
  NVACSKIP = -5
  EXTCUR(1) = 12.0
  EXTCUR(3) = -4.5
/
"""
    p = tmp_path / "input.fb_extcur"
    p.write_text(txt)
    indata = read_indata(p)
    cfg = config_from_indata(indata)

    assert cfg.lfreeb is True
    assert cfg.mgrid_file == "mgrid_test.nc"
    # NVACSKIP<=0 falls back to NFP.
    assert cfg.nvacskip == 2
    assert cfg.extcur == (12.0, 0.0, -4.5)

    static = build_static(cfg)
    assert static.free_boundary_state0 is not None
    assert static.free_boundary_state0.nvacskip == 2


def test_mgrid_loader_skeleton(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    p = tmp_path / "mgrid_test.nc"
    with netCDF4.Dataset(str(p), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 8)
        ds.createDimension("external_coil_groups", 2)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 2)
        ds.createDimension("rad", 3)
        ds.createDimension("zee", 4)
        ds.createDimension("phi", 2)

        for name, value in (
            ("ir", 3),
            ("jz", 4),
            ("kp", 2),
            ("nfp", 5),
            ("nextcur", 2),
        ):
            v = ds.createVariable(name, "i4", ())
            v.assignValue(value)
        for name, value in (
            ("rmin", 1.0),
            ("rmax", 2.0),
            ("zmin", -0.3),
            ("zmax", 0.7),
        ):
            v = ds.createVariable(name, "f8", ())
            v.assignValue(value)

        mgrid_mode = ds.createVariable("mgrid_mode", "S1", ("dim_00001",))
        mgrid_mode[:] = np.asarray(list("S"), dtype="S1")
        coil_group = ds.createVariable("coil_group", "S1", ("external_coil_groups", "stringsize"))
        names = np.full((2, 8), b" ", dtype="S1")
        names[0, :5] = np.asarray(list("coilA"), dtype="S1")
        names[1, :5] = np.asarray(list("coilB"), dtype="S1")
        coil_group[:] = names
        raw = ds.createVariable("raw_coil_cur", "f8", ("external_coils",))
        raw[:] = np.asarray([10.0, -7.0], dtype=np.float64)

        for i in (1, 2):
            for prefix in ("br", "bp", "bz"):
                v = ds.createVariable(f"{prefix}_{i:03d}", "f8", ("phi", "zee", "rad"))
                v[:] = (100 * i) + np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)

    meta = load_mgrid(p, load_fields=False)
    assert isinstance(meta, MGridMetadata)
    assert meta.ir == 3
    assert meta.jz == 4
    assert meta.kp == 2
    assert meta.nfp == 5
    assert meta.nextcur == 2
    assert meta.coil_groups == ("coilA", "coilB")
    assert meta.raw_coil_cur == (10.0, -7.0)

    data = load_mgrid(p, load_fields=True)
    assert isinstance(data, MGridData)
    assert data.br.shape == (2, 2, 4, 3)
    assert data.bp.shape == (2, 2, 4, 3)
    assert data.bz.shape == (2, 2, 4, 3)
    # Check one value round-trips correctly.
    assert float(data.br[1, 1, 3, 2]) == pytest.approx(200 + (1 * 12 + 3 * 3 + 2))


def test_prepare_mgrid_for_config_validates_and_normalizes_extcur(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_test.nc"
    with netCDF4.Dataset(str(mg), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 8)
        ds.createDimension("external_coil_groups", 2)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 2)
        ds.createDimension("rad", 2)
        ds.createDimension("zee", 3)
        ds.createDimension("phi", 6)
        for name, value in (
            ("ir", 2),
            ("jz", 3),
            ("kp", 6),
            ("nfp", 3),
            ("nextcur", 2),
        ):
            v = ds.createVariable(name, "i4", ())
            v.assignValue(value)
        for name, value in (("rmin", 1.0), ("rmax", 2.0), ("zmin", -1.0), ("zmax", 1.0)):
            v = ds.createVariable(name, "f8", ())
            v.assignValue(value)

    txt = f"""
&INDATA
  NFP = 3
  MPOL = 5
  NTOR = 1
  NS = 9
  NZETA = 3
  LFREEB = T
  MGRID_FILE = '{mg}'
  EXTCUR(1) = 5.0
/
"""
    p = tmp_path / "input.fb_prepare"
    p.write_text(txt)
    cfg = config_from_indata(read_indata(p))
    prepared = prepare_mgrid_for_config(cfg, load_fields=False, strict=True)
    assert isinstance(prepared, PreparedMGrid)
    assert prepared.metadata.nextcur == 2
    assert prepared.extcur == (5.0, 0.0)


def test_prepare_mgrid_for_config_rejects_nfp_mismatch(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_bad_nfp.nc"
    with netCDF4.Dataset(str(mg), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 4)
        ds.createDimension("external_coil_groups", 1)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 1)
        ds.createDimension("rad", 2)
        ds.createDimension("zee", 2)
        ds.createDimension("phi", 4)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 4),
            ("nfp", 7),
            ("nextcur", 1),
        ):
            v = ds.createVariable(name, "i4", ())
            v.assignValue(value)
        for name, value in (("rmin", 1.0), ("rmax", 2.0), ("zmin", -1.0), ("zmax", 1.0)):
            v = ds.createVariable(name, "f8", ())
            v.assignValue(value)

    txt = f"""
&INDATA
  NFP = 2
  MPOL = 5
  NTOR = 1
  NS = 9
  NZETA = 2
  LFREEB = T
  MGRID_FILE = '{mg}'
/
"""
    p = tmp_path / "input.fb_bad_nfp"
    p.write_text(txt)
    cfg = config_from_indata(read_indata(p))
    with pytest.raises(ValueError, match="MGRID nfp"):
        prepare_mgrid_for_config(cfg, load_fields=False, strict=True)


def test_prepare_mgrid_for_config_rejects_kp_nzeta_mismatch(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_bad_kp.nc"
    with netCDF4.Dataset(str(mg), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 4)
        ds.createDimension("external_coil_groups", 1)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 1)
        ds.createDimension("rad", 2)
        ds.createDimension("zee", 2)
        ds.createDimension("phi", 5)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 5),
            ("nfp", 2),
            ("nextcur", 1),
        ):
            v = ds.createVariable(name, "i4", ())
            v.assignValue(value)
        for name, value in (("rmin", 1.0), ("rmax", 2.0), ("zmin", -1.0), ("zmax", 1.0)):
            v = ds.createVariable(name, "f8", ())
            v.assignValue(value)

    txt = f"""
&INDATA
  NFP = 2
  MPOL = 5
  NTOR = 1
  NS = 9
  NZETA = 3
  LFREEB = T
  MGRID_FILE = '{mg}'
/
"""
    p = tmp_path / "input.fb_bad_kp"
    p.write_text(txt)
    cfg = config_from_indata(read_indata(p))
    with pytest.raises(ValueError, match="kp="):
        prepare_mgrid_for_config(cfg, load_fields=False, strict=True)


def test_run_fixed_boundary_initial_guess_carries_mgrid_metadata(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_ok.nc"
    with netCDF4.Dataset(str(mg), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 8)
        ds.createDimension("external_coil_groups", 2)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 2)
        ds.createDimension("rad", 2)
        ds.createDimension("zee", 2)
        ds.createDimension("phi", 4)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 4),
            ("nfp", 1),
            ("nextcur", 2),
        ):
            v = ds.createVariable(name, "i4", ())
            v.assignValue(value)
        for name, value in (("rmin", 1.0), ("rmax", 2.0), ("zmin", -1.0), ("zmax", 1.0)):
            v = ds.createVariable(name, "f8", ())
            v.assignValue(value)

    inpath = tmp_path / "input.fb_guess"
    inpath.write_text(
        f"""
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 9
  NZETA = 2
  NTHETA = 8
  LASYM = F
  LFREEB = T
  MGRID_FILE = '{mg}'
  EXTCUR(1) = 3.5
  RBC(0,0) = 6.0
  ZBS(1,0) = 2.0
/
"""
    )

    run = run_fixed_boundary(
        inpath,
        use_initial_guess=True,
        verbose=False,
    )
    assert run.cfg.lfreeb is True
    assert run.static.mgrid_metadata is not None
    assert int(run.static.mgrid_metadata.kp) == 4
    assert run.static.free_boundary_extcur == (3.5, 0.0)


def test_run_fixed_boundary_freeb_diagnostics_stub(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_diag.nc"
    with netCDF4.Dataset(str(mg), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 8)
        ds.createDimension("external_coil_groups", 1)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 1)
        ds.createDimension("rad", 2)
        ds.createDimension("zee", 2)
        ds.createDimension("phi", 4)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 4),
            ("nfp", 1),
            ("nextcur", 1),
        ):
            v = ds.createVariable(name, "i4", ())
            v.assignValue(value)
        for name, value in (("rmin", 1.0), ("rmax", 2.0), ("zmin", -1.0), ("zmax", 1.0)):
            v = ds.createVariable(name, "f8", ())
            v.assignValue(value)

    inpath = tmp_path / "input.fb_diag"
    inpath.write_text(
        f"""
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 9
  NZETA = 2
  NTHETA = 8
  LASYM = F
  LFREEB = T
  MGRID_FILE = '{mg}'
  NVACSKIP = 2
  RBC(0,0) = 6.0
  ZBS(1,0) = 2.0
/
"""
    )

    run = run_fixed_boundary(
        inpath,
        solver="vmec2000_iter",
        max_iter=1,
        multigrid=False,
        verbose=False,
    )
    assert run.result is not None
    fb = run.result.diagnostics.get("free_boundary")
    assert isinstance(fb, dict)
    assert bool(fb.get("enabled", False)) is True
    assert int(fb.get("nvacskip", 0)) == 2
    assert bool(fb.get("vacuum_stub", False)) is True
    fbext = run.result.diagnostics.get("free_boundary_external_field")
    assert isinstance(fbext, dict)
    assert bool(fbext.get("vacuum_stub", False)) is True
    assert "sample_time_s" in fbext


def test_run_fixed_boundary_freeb_external_sampling_can_be_disabled(tmp_path: Path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_diag_off.nc"
    with netCDF4.Dataset(str(mg), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 8)
        ds.createDimension("external_coil_groups", 1)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 1)
        ds.createDimension("rad", 2)
        ds.createDimension("zee", 2)
        ds.createDimension("phi", 4)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 4),
            ("nfp", 1),
            ("nextcur", 1),
        ):
            v = ds.createVariable(name, "i4", ())
            v.assignValue(value)
        for name, value in (("rmin", 1.0), ("rmax", 2.0), ("zmin", -1.0), ("zmax", 1.0)):
            v = ds.createVariable(name, "f8", ())
            v.assignValue(value)

    inpath = tmp_path / "input.fb_diag_off"
    inpath.write_text(
        f"""
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 9
  NZETA = 2
  NTHETA = 8
  LASYM = F
  LFREEB = T
  MGRID_FILE = '{mg}'
  NVACSKIP = 2
  RBC(0,0) = 6.0
  ZBS(1,0) = 2.0
/
"""
    )

    monkeypatch.setenv("VMEC_JAX_FREEB_SAMPLE_EXTERNAL", "0")
    run = run_fixed_boundary(
        inpath,
        solver="vmec2000_iter",
        max_iter=1,
        multigrid=False,
        verbose=False,
    )
    fbext = run.result.diagnostics.get("free_boundary_external_field")
    assert isinstance(fbext, dict)
    assert bool(fbext.get("vacuum_stub", False)) is True
    assert bool(fbext.get("enabled", True)) is False
    assert bool(fbext.get("available", True)) is False
    assert fbext.get("reason") == "disabled_by_env"


def test_interpolate_mgrid_bfield_trilinear_linear_field():
    meta = MGridMetadata(
        path="dummy.nc",
        ir=3,
        jz=3,
        kp=4,
        nfp=1,
        nextcur=2,
        rmin=0.0,
        rmax=2.0,
        zmin=-1.0,
        zmax=1.0,
        mgrid_mode="S",
        coil_groups=("A", "B"),
        raw_coil_cur=(1.0, 1.0),
    )
    # Build a linear field in index-space mapped to physical coordinates.
    r_nodes = np.linspace(meta.rmin, meta.rmax, meta.ir)
    z_nodes = np.linspace(meta.zmin, meta.zmax, meta.jz)
    phi_nodes = np.arange(meta.kp, dtype=float) * (2.0 * np.pi / meta.kp)
    br = np.zeros((meta.nextcur, meta.kp, meta.jz, meta.ir), dtype=float)
    bp = np.zeros_like(br)
    bz = np.zeros_like(br)
    for ig in range(meta.nextcur):
        scale = float(ig + 1)
        for k, phi in enumerate(phi_nodes):
            for j, zv in enumerate(z_nodes):
                for i, rv in enumerate(r_nodes):
                    # Linear in (r,z,phi): trilinear interpolation should be exact.
                    val = scale * (rv + 2.0 * zv + 0.1 * phi)
                    br[ig, k, j, i] = val
                    bp[ig, k, j, i] = 2.0 * val
                    bz[ig, k, j, i] = -val
    data = MGridData(metadata=meta, br=br, bp=bp, bz=bz)
    extcur = (2.0, -0.5)

    rq = np.array([0.25, 1.5])
    zq = np.array([-0.25, 0.75])
    phiq = np.array([0.5, 2.0 * np.pi + 0.2])  # second point checks periodic wrap
    br_q, bp_q, bz_q = interpolate_mgrid_bfield(data, r=rq, z=zq, phi=phiq, extcur=extcur)

    # Expected from analytic linear field with same extcur weighting.
    coeff = extcur[0] * 1.0 + extcur[1] * 2.0
    expected = coeff * (rq + 2.0 * zq + 0.1 * np.mod(phiq, 2.0 * np.pi))
    np.testing.assert_allclose(br_q, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bp_q, 2.0 * expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bz_q, -expected, rtol=1e-12, atol=1e-12)


def test_boundary_vacuum_projection_toroidal_field():
    ntheta = 16
    nzeta = 8
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)[:, None]
    _zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)[None, :]
    r0 = 10.0
    a = 1.2
    bphi0 = 2.5

    R = r0 + a * np.cos(theta)
    Ru = -a * np.sin(theta) * np.ones_like(_zeta)
    Zu = a * np.cos(theta) * np.ones_like(_zeta)
    Rv = np.zeros((ntheta, nzeta))
    Zv = np.zeros((ntheta, nzeta))
    br = np.zeros((ntheta, nzeta))
    bp = np.full((ntheta, nzeta), bphi0)
    bz = np.zeros((ntheta, nzeta))

    g_uu, g_uv, g_vv, det = boundary_metric_from_rz(R=R, Ru=Ru, Zu=Zu, Rv=Rv, Zv=Zv)
    np.testing.assert_allclose(g_uu, a * a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g_uv, 0.0, rtol=1e-12, atol=1e-12)
    R2 = np.broadcast_to(R * R, g_vv.shape)
    np.testing.assert_allclose(g_vv, R2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(det, (a * a) * R2, rtol=1e-12, atol=1e-12)

    bu, bv = covariant_boundary_field_from_cylindrical(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    Rb = np.broadcast_to(R, bv.shape)
    np.testing.assert_allclose(bu, 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bv, Rb * bphi0, rtol=1e-12, atol=1e-12)

    bsupu, bsupv, _ = contravariant_boundary_field_from_covariant(
        bu=bu,
        bv=bv,
        g_uu=g_uu,
        g_uv=g_uv,
        g_vv=g_vv,
    )
    np.testing.assert_allclose(bsupu, 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bsupv, bphi0 / Rb, rtol=1e-12, atol=1e-12)

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
    np.testing.assert_allclose(vac.bsqvac, bphi0 * bphi0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vac.bnormal, 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vac.bnormal_unit, 0.0, rtol=1e-12, atol=1e-12)


def test_run_fixed_boundary_freeb_diagnostics_include_wp2_channels(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_diag_wp2.nc"
    with netCDF4.Dataset(str(mg), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 8)
        ds.createDimension("external_coil_groups", 1)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 1)
        ds.createDimension("rad", 2)
        ds.createDimension("zee", 2)
        ds.createDimension("phi", 4)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 4),
            ("nfp", 1),
            ("nextcur", 1),
        ):
            ds.createVariable(name, "i4", ()).assignValue(value)
        for name, value in (("rmin", 1.0), ("rmax", 12.0), ("zmin", -3.0), ("zmax", 3.0)):
            ds.createVariable(name, "f8", ()).assignValue(value)
        # Uniform toroidal field in the single coil group.
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = 1.0
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0

    inpath = tmp_path / "input.fb_diag_wp2"
    inpath.write_text(
        f"""
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 9
  NZETA = 2
  NTHETA = 8
  LASYM = F
  LFREEB = T
  MGRID_FILE = '{mg}'
  NVACSKIP = 2
  RBC(0,0) = 6.0
  ZBS(1,0) = 2.0
/
"""
    )

    run = run_fixed_boundary(
        inpath,
        solver="vmec2000_iter",
        max_iter=1,
        multigrid=False,
        verbose=False,
    )
    fbext = run.result.diagnostics.get("free_boundary_external_field")
    assert isinstance(fbext, dict)
    for key in (
        "bu_rms",
        "bv_rms",
        "bsupu_rms",
        "bsupv_rms",
        "bsqvac_mean",
        "bsqvac_max",
        "bnormal_rms",
        "bnormal_unit_rms",
        "det_guv_min",
        "det_guv_max",
    ):
        assert key in fbext
    assert bool(fbext.get("available", False)) is True

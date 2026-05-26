from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import config_from_indata
from vmec_jax.free_boundary import (
    MGridData,
    MGridMetadata,
    PreparedMGrid,
    _axis_current_field_vmec_filament,
    _build_vmec_mode_basis,
    _freeb_use_greenf_source,
    _vmec_bvec_from_gsource,
    boundary_metric_from_rz,
    covariant_boundary_field_from_cylindrical,
    contravariant_boundary_field_from_covariant,
    interpolate_mgrid_bfield,
    load_mgrid,
    nestor_external_only_step,
    prepare_mgrid_for_config,
    sample_external_vacuum_diagnostics,
    vacuum_boundary_fields_from_cylindrical,
)
from vmec_jax.namelist import read_indata
from vmec_jax.solve import (
    _free_boundary_iter_controls_vmec,
    _free_boundary_prev_rz_fsq_next,
    _free_boundary_should_damp_constraint_baseline,
    _free_boundary_turnon_resets_iter1_immediately,
)
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


def test_free_boundary_iter_controls_vmec_threshold_and_skip_logic():
    ivac = -1
    nvacskip = 9
    nvskip0 = 9

    # iter2<=1 does not advance vacuum activation.
    ivac, ivacskip, nvacskip = _free_boundary_iter_controls_vmec(
        iter2=1,
        iter1=1,
        ivac=ivac,
        nvacskip=nvacskip,
        nvskip0=nvskip0,
        fsq_rz_prev=1.0e-1,
    )
    assert ivac == -1
    assert ivacskip == 0
    assert nvacskip == 9

    # VMEC funct3d increments ivac by +1 whenever residual is below threshold.
    # Starting from ivac=-1, the first low-residual step moves to ivac=0.
    ivac, ivacskip, nvacskip = _free_boundary_iter_controls_vmec(
        iter2=2,
        iter1=1,
        ivac=ivac,
        nvacskip=nvacskip,
        nvskip0=nvskip0,
        fsq_rz_prev=1.0e-4,
    )
    assert ivac == 0
    assert ivacskip == 0

    # Next low-residual step reaches ivac=1 (vacuum turn-on point).
    ivac, ivacskip, nvacskip = _free_boundary_iter_controls_vmec(
        iter2=3,
        iter1=1,
        ivac=ivac,
        nvacskip=nvacskip,
        nvskip0=nvskip0,
        fsq_rz_prev=1.0e-4,
    )
    assert ivac == 1
    assert ivacskip == 0

    # ivac<=2 forces full updates.
    ivac, ivacskip, nvacskip = _free_boundary_iter_controls_vmec(
        iter2=4,
        iter1=1,
        ivac=ivac,
        nvacskip=nvacskip,
        nvskip0=nvskip0,
        fsq_rz_prev=1.0e-4,
    )
    assert ivac == 2
    assert ivacskip == 0

    ivac, ivacskip, nvacskip = _free_boundary_iter_controls_vmec(
        iter2=5,
        iter1=1,
        ivac=ivac,
        nvacskip=nvacskip,
        nvskip0=nvskip0,
        fsq_rz_prev=1.0e-4,
    )
    assert ivac == 3
    assert ivacskip == (5 - 1) % nvacskip

    # After ivac>2, reuse cadence can become nonzero.
    ivac, ivacskip, nvacskip = _free_boundary_iter_controls_vmec(
        iter2=6,
        iter1=1,
        ivac=ivac,
        nvacskip=nvacskip,
        nvskip0=nvskip0,
        fsq_rz_prev=1.0e-4,
    )
    assert ivac == 4
    assert ivacskip == (6 - 1) % nvacskip

    # Solver path initializes ivac at 0, so the first low-residual iteration
    # should immediately reach the turn-on state ivac==1.
    ivac2, ivs2, _ = _free_boundary_iter_controls_vmec(
        iter2=2,
        iter1=1,
        ivac=0,
        nvacskip=9,
        nvskip0=9,
        fsq_rz_prev=1.0e-4,
    )
    assert ivac2 == 1
    assert ivs2 == 0


def test_free_boundary_iter_controls_vmec_updates_nvacskip_on_full_step():
    ivac, ivacskip, nvacskip = _free_boundary_iter_controls_vmec(
        iter2=10,
        iter1=1,
        ivac=5,
        nvacskip=9,
        nvskip0=9,
        fsq_rz_prev=1.0e-12,
    )
    assert ivac == 6
    assert ivacskip == 0
    # fsq=1e-12 -> max(1e-1,1e11*fsq)=1e-1 -> nvacskip target=10
    assert nvacskip == 10


def test_free_boundary_prev_rz_fsq_next_preserves_pre_turnon_value():
    assert _free_boundary_prev_rz_fsq_next(
        prev_fsq_before=8.9e-4,
        fsq_rz_curr=2.5e-1,
        turnon_restart=True,
        preserve_turnon_restart=True,
    ) == pytest.approx(8.9e-4)
    assert _free_boundary_prev_rz_fsq_next(
        prev_fsq_before=8.9e-4,
        fsq_rz_curr=2.5e-1,
        turnon_restart=False,
        preserve_turnon_restart=True,
        ) == pytest.approx(2.5e-1)
    assert _free_boundary_prev_rz_fsq_next(
        prev_fsq_before=8.9e-4,
        fsq_rz_curr=2.5e-1,
        turnon_restart=True,
        preserve_turnon_restart=False,
    ) == pytest.approx(2.5e-1)


def test_free_boundary_constraint_baseline_damping_skips_turnon_step():
    assert _free_boundary_should_damp_constraint_baseline(
        freeb_ivac=1,
        freeb_turnon_iter=True,
        lthreed=True,
    ) is False
    assert _free_boundary_should_damp_constraint_baseline(
        freeb_ivac=2,
        freeb_turnon_iter=False,
        lthreed=True,
    ) is True
    assert _free_boundary_should_damp_constraint_baseline(
        freeb_ivac=1,
        freeb_turnon_iter=True,
        lthreed=False,
    ) is True
    assert _free_boundary_should_damp_constraint_baseline(
        freeb_ivac=-1,
        freeb_turnon_iter=False,
        lthreed=True,
    ) is False


def test_free_boundary_turnon_iter1_reset_policy_depends_on_lasym_and_topology():
    assert _free_boundary_turnon_resets_iter1_immediately(lthreed=False, lasym=False) is True
    assert _free_boundary_turnon_resets_iter1_immediately(lthreed=False, lasym=True) is True
    assert _free_boundary_turnon_resets_iter1_immediately(lthreed=True, lasym=False) is True
    assert _free_boundary_turnon_resets_iter1_immediately(lthreed=True, lasym=True) is False


def test_vmec_mode_basis_and_bvec_skip_modes():
    ntheta, nzeta = 5, 7
    wint = np.full((ntheta, nzeta), 1.0 / float(ntheta * nzeta))
    basis = _build_vmec_mode_basis(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=2,
        mf=2,
        nf=1,
        lasym=False,
        wint=wint,
    )
    # VMEC precal convention: reduced nu3 grid uses full nu spacing.
    assert int(basis["nu_full"]) == 2 * (ntheta - 1)
    assert int(basis["nuv3"]) == ntheta * nzeta
    assert int(basis["nuv_full"]) == int(basis["nu_full"]) * nzeta
    assert np.asarray(basis["imirr_full"]).shape == (int(basis["nuv_full"]),)
    gsource = np.ones((ntheta, nzeta), dtype=float)
    bvec = _vmec_bvec_from_gsource(gsource=gsource, basis=basis)
    assert bvec.shape == (basis["mnpd"],)
    # VMEC fouri.f: (m==0 and n<0) entries are skipped/cycled.
    skip = np.logical_and(np.asarray(basis["xmpot"]) == 0, np.asarray(basis["n_raw"]) < 0)
    assert np.allclose(np.asarray(bvec)[skip], 0.0)
    # Constant source anti-symmetrizes to zero in non-LASYM path.
    assert np.linalg.norm(np.asarray(bvec)) < 1.0e-12

    basis_asym = _build_vmec_mode_basis(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=2,
        mf=2,
        nf=1,
        lasym=True,
        wint=wint,
    )
    bvec_asym = _vmec_bvec_from_gsource(gsource=gsource, basis=basis_asym)
    mnpd = int(basis_asym["mnpd"])
    assert bvec_asym.shape == (2 * mnpd,)
    assert np.allclose(np.asarray(bvec_asym[:mnpd])[skip], 0.0)
    assert np.allclose(np.asarray(bvec_asym[mnpd:])[skip], 0.0)


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
        ds.createDimension("phi", 8)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 8),
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
        ds.createDimension("phi", 8)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 8),
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
    # With NTOR=0, nzeta is collapsed to 1 (axisym optimisation), so the
    # vacuum sampling uses 2*nzeta = 2 toroidal planes (kp_effective = 2).
    assert int(run.static.mgrid_metadata.kp) == 2
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


def test_run_fixed_boundary_resolves_relative_mgrid_from_input_dir(tmp_path: Path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    input_dir = tmp_path / "case"
    input_dir.mkdir()
    other_dir = tmp_path / "outside"
    other_dir.mkdir()

    mg = input_dir / "mgrid_rel.nc"
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

    inpath = input_dir / "input.fb_relpath"
    inpath.write_text(
        """
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 9
  NZETA = 2
  NTHETA = 8
  LASYM = F
  LFREEB = T
  MGRID_FILE = 'mgrid_rel.nc'
  EXTCUR(1) = 2.5
  RBC(0,0) = 6.0
  ZBS(1,0) = 2.0
/
"""
    )

    monkeypatch.chdir(other_dir)
    run = run_fixed_boundary(
        inpath,
        use_initial_guess=True,
        verbose=False,
    )

    assert run.cfg.lfreeb is True
    assert run.cfg.mgrid_file == str(mg.resolve())
    assert run.static.mgrid_metadata is not None
    # With NTOR=0, nzeta is collapsed to 1 (axisym optimisation), so the
    # vacuum sampling uses 2*nzeta = 2 toroidal planes (kp_effective = 2).
    assert int(run.static.mgrid_metadata.kp) == 2
    assert run.static.free_boundary_extcur == (2.5,)


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


def test_interpolate_mgrid_bfield_vmec_kv_subsamples_divisible_planes():
    meta = MGridMetadata(
        path="dummy.nc",
        ir=2,
        jz=2,
        kp=8,
        nfp=1,
        nextcur=1,
        rmin=0.0,
        rmax=1.0,
        zmin=0.0,
        zmax=1.0,
        mgrid_mode="S",
        coil_groups=("A",),
        raw_coil_cur=(1.0,),
    )
    # Encode toroidal plane index directly in the field values.
    br = np.zeros((1, meta.kp, meta.jz, meta.ir), dtype=float)
    bp = np.zeros_like(br)
    bz = np.zeros_like(br)
    for k in range(meta.kp):
        br[0, k, :, :] = float(k)
        bp[0, k, :, :] = 10.0 * float(k)
        bz[0, k, :, :] = -float(k)
    data = MGridData(metadata=meta, br=br, bp=bp, bz=bz)

    # VMEC becoil samples the mgrid planes matching the VMEC zeta grid.
    # For kp=8 and nzeta=4, this should sample planes k=0,2,4,6.
    r = np.full((2, 4), 0.5)
    z = np.full((2, 4), 0.5)
    phi = np.zeros((2, 4))
    br_q, bp_q, bz_q = interpolate_mgrid_bfield(
        data,
        r=r,
        z=z,
        phi=phi,
        use_vmec_kv=True,
    )
    expected_k = np.array([0.0, 2.0, 4.0, 6.0], dtype=float)
    expected = np.broadcast_to(expected_k[None, :], r.shape)
    np.testing.assert_allclose(br_q, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(bp_q, 10.0 * expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(bz_q, -expected, rtol=0.0, atol=1e-14)


def test_interpolate_mgrid_bfield_allows_single_toroidal_plane():
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
        raw_coil_cur=(1.0,),
    )
    br = np.full((1, 1, 2, 2), 3.0, dtype=float)
    bp = np.full((1, 1, 2, 2), -2.0, dtype=float)
    bz = np.full((1, 1, 2, 2), 7.5, dtype=float)
    data = MGridData(metadata=meta, br=br, bp=bp, bz=bz)
    r = np.full((2, 4), 0.5)
    z = np.full((2, 4), 0.5)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False).reshape(2, 4)

    br_q, bp_q, bz_q = interpolate_mgrid_bfield(
        data,
        r=r,
        z=z,
        phi=phi,
        use_vmec_kv=True,
    )
    np.testing.assert_allclose(br_q, 3.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(bp_q, -2.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(bz_q, 7.5, rtol=0.0, atol=1e-14)


def test_axis_current_vmec_filament_nonzero_for_nzeta1():
    # VMEC precal/tolicu uses nvper=64 when nv=1 (axisymmetric vacuum). Ensure
    # the JAX filament path remains non-degenerate for nzeta=1.
    ntheta = 8
    nzeta = 1
    R = np.full((ntheta, nzeta), 2.0, dtype=float)
    Z = np.linspace(-0.25, 0.25, ntheta, dtype=float).reshape(ntheta, nzeta)
    axis_r = np.asarray([1.0], dtype=float)
    axis_z = np.asarray([0.0], dtype=float)
    br, bp, bz = _axis_current_field_vmec_filament(
        R=R,
        Z=Z,
        axis_r=axis_r,
        axis_z=axis_z,
        nfp=1,
        plascur=0.3,
    )
    total_rms = float(np.sqrt(np.mean(br * br + bp * bp + bz * bz)))
    assert total_rms > 0.0


def test_freeb_use_greenf_source_default_and_env_override(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_FREEB_USE_GREENF_SOURCE", raising=False)
    assert _freeb_use_greenf_source(0) is True
    assert _freeb_use_greenf_source(3) is True

    monkeypatch.setenv("VMEC_JAX_FREEB_USE_GREENF_SOURCE", "0")
    assert _freeb_use_greenf_source(0) is False
    assert _freeb_use_greenf_source(3) is False

    monkeypatch.setenv("VMEC_JAX_FREEB_USE_GREENF_SOURCE", "1")
    assert _freeb_use_greenf_source(0) is True


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
    # VMEC vacuum convention stores bsqvac = 0.5*|B|^2.
    np.testing.assert_allclose(vac.bsqvac, 0.5 * bphi0 * bphi0, rtol=1e-12, atol=1e-12)
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


def test_nestor_external_only_step_reuse(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_nestor_reuse.nc"
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
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = 1.0
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0

    inpath = tmp_path / "input.fb_nestor_reuse"
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
    state = run.result.state
    static = run.static

    step1, rt1 = nestor_external_only_step(state=state, static=static, ivac=1, runtime=None)
    assert step1.reused is False
    assert step1.vac_total.bsqvac.shape == step1.phi.shape
    assert int(rt1.update_count) == 1
    assert int(rt1.reuse_count) == 0

    step2, rt2 = nestor_external_only_step(state=state, static=static, ivac=2, runtime=rt1)
    assert step2.reused is True
    assert int(rt2.update_count) == 1
    assert int(rt2.reuse_count) == 1
    assert float(step2.sample_time_s) >= 0.0
    assert float(step2.solve_time_s) >= 0.0
    assert step2.vac_total.bsqvac.shape == step1.vac_total.bsqvac.shape
    assert np.max(np.abs(np.asarray(step2.vac_total.det_guv))) > 0.0


def test_nestor_reuse_legacy_hold_path(tmp_path: Path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_nestor_hold.nc"
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
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = 1.0
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0

    inpath = tmp_path / "input.fb_nestor_hold"
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
    state = run.result.state
    static = run.static

    step1, rt1 = nestor_external_only_step(state=state, static=static, ivac=1, runtime=None)
    monkeypatch.setenv("VMEC_JAX_FREEB_REUSE_RHS_UPDATE", "0")
    step2, rt2 = nestor_external_only_step(state=state, static=static, ivac=2, runtime=rt1)
    assert bool(step2.reused) is True
    assert float(step2.sample_time_s) == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(np.asarray(step2.phi), np.asarray(step1.phi), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(step2.vac_total.bsqvac), np.asarray(step1.vac_total.bsqvac), rtol=0.0, atol=0.0)
    assert int(rt2.update_count) == int(rt1.update_count)
    assert int(rt2.reuse_count) == int(rt1.reuse_count) + 1


def test_nestor_vmec2000_like_mode_and_fallback(tmp_path: Path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_nestor_mode.nc"
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
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = 1.0
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0

    inpath = tmp_path / "input.fb_nestor_mode"
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
    state = run.result.state
    static = run.static

    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "vmec2000_like")
    monkeypatch.setenv("VMEC_JAX_FREEB_VMEC_LIKE_MAX_POINTS", "4096")
    step_dense, rt_dense = nestor_external_only_step(state=state, static=static, ivac=1, runtime=None)
    assert str(step_dense.model).startswith("vmec2000_like_dense_integral")
    assert str(rt_dense.mode).startswith("vmec2000_like_dense_integral")

    monkeypatch.setenv("VMEC_JAX_FREEB_VMEC_LIKE_MAX_POINTS", "4")
    step_fb, rt_fb = nestor_external_only_step(state=state, static=static, ivac=1, runtime=None)
    assert str(step_fb.model).startswith("spectral_poisson_external_only_fallback")
    assert str(rt_fb.mode).startswith("spectral_poisson_external_only_fallback")


def test_freeb_axis_current_sampling_changes_boundary_field(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_axis_current.nc"
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
        # Keep mgrid contribution zero so the axis-current effect is isolated.
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0

    inpath = tmp_path / "input.fb_axis_current"
    inpath.write_text(
        f"""
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 9
  NZETA = 4
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
    state = run.result.state
    static = run.static

    d0 = sample_external_vacuum_diagnostics(state=state, static=static, plascur=0.0)
    d1 = sample_external_vacuum_diagnostics(state=state, static=static, plascur=0.2)

    assert float(d0.get("br_axis_rms", 0.0)) == pytest.approx(0.0, abs=1e-14)
    assert float(d0.get("bp_axis_rms", 0.0)) == pytest.approx(0.0, abs=1e-14)
    assert float(d0.get("bz_axis_rms", 0.0)) == pytest.approx(0.0, abs=1e-14)
    axis_rms_sum = (
        float(d1.get("br_axis_rms", 0.0))
        + float(d1.get("bp_axis_rms", 0.0))
        + float(d1.get("bz_axis_rms", 0.0))
    )
    assert axis_rms_sum > 0.0


def test_axisymmetric_freeb_sampling_collapses_toroidal_grid(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_axisym_nv1.nc"
    with netCDF4.Dataset(str(mg), mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("stringsize", 8)
        ds.createDimension("external_coil_groups", 1)
        ds.createDimension("dim_00001", 1)
        ds.createDimension("external_coils", 1)
        ds.createDimension("rad", 2)
        ds.createDimension("zee", 2)
        ds.createDimension("phi", 8)
        for name, value in (
            ("ir", 2),
            ("jz", 2),
            ("kp", 8),
            ("nfp", 1),
            ("nextcur", 1),
        ):
            ds.createVariable(name, "i4", ()).assignValue(value)
        for name, value in (("rmin", 1.0), ("rmax", 12.0), ("zmin", -3.0), ("zmax", 3.0)):
            ds.createVariable(name, "f8", ()).assignValue(value)
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0

    inpath = tmp_path / "input.fb_axisym_nv1"
    inpath.write_text(
        f"""
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 9
  NZETA = 8
  NTHETA = 10
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
    d = sample_external_vacuum_diagnostics(state=run.result.state, static=run.static, plascur=0.0)

    assert bool(d.get("available", False)) is True
    expected_ntheta3 = int(np.asarray(run.static.trig_vmec.cosmu).shape[0])
    assert int(d.get("n_samples", -1)) == expected_ntheta3


@pytest.mark.full
def test_freeb_turnon_restart_sets_iter1_and_reuse_step(tmp_path: Path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_turnon.nc"
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
        for name, value in (("rmin", 1.0), ("rmax", 10.0), ("zmin", -3.0), ("zmax", 3.0)):
            ds.createVariable(name, "f8", ()).assignValue(value)
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0

    inpath = tmp_path / "input.fb_turnon"
    inpath.write_text(
        f"""
&INDATA
  NFP = 1
  MPOL = 5
  NTOR = 0
  NS = 9
  NZETA = 4
  NTHETA = 8
  LASYM = F
  LFREEB = T
  MGRID_FILE = '{mg}'
  NVACSKIP = 3
  RBC(0,0) = 6.0
  ZBS(1,0) = 2.0
/
"""
    )

    monkeypatch.setenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "1.0e8")
    monkeypatch.setenv("VMEC_JAX_FREEB_NESTOR_MODE", "vmec2000_like")
    monkeypatch.setenv("VMEC_JAX_FREEB_VMEC_LIKE_MAX_POINTS", "1000000")

    run = run_fixed_boundary(
        inpath,
        solver="vmec2000_iter",
        max_iter=6,
        multigrid=False,
        verbose=False,
    )
    diag = run.result.diagnostics
    ivac = np.asarray(diag["freeb_ivac_history"], dtype=int)
    ivacskip = np.asarray(diag["freeb_ivacskip_history"], dtype=int)
    reused = np.asarray(diag["freeb_nestor_reused_history"], dtype=int)

    turnon_idx = np.where(ivac == 1)[0]
    assert turnon_idx.size >= 1
    k = int(turnon_idx[0])
    # Turn-on iteration itself must be a full update.
    assert int(ivacskip[k]) == 0
    assert int(reused[k]) == 0
    # VMEC eqsolve promotes ivac==1 -> 2 after the turn-on iteration, so the
    # next step can already enter reuse cadence.
    if (k + 1) < ivac.size:
        assert int(ivac[k + 1]) >= 2
    # Ensure reuse activates once ivac advances beyond 2.
    post_turnon = np.where((ivac > 2) & (ivacskip > 0))[0]
    if post_turnon.size:
        assert np.any(reused[post_turnon] == 1)


def test_run_fixed_boundary_freeb_edge_coupling_diag(tmp_path: Path):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_edge_coupling.nc"
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
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = 1.0
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0

    inpath = tmp_path / "input.fb_edge_coupling"
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
        max_iter=2,
        multigrid=False,
        verbose=False,
    )
    diag = run.result.diagnostics
    fb = diag.get("free_boundary", {})
    assert bool(fb.get("enabled", False)) is True
    assert bool(fb.get("couple_edge", False)) is True
    assert "nestor_model" in fb
    inc = np.asarray(diag.get("include_edge_history", np.zeros((0,), dtype=int)))
    reused = np.asarray(diag.get("freeb_nestor_reused_history", np.zeros((0,), dtype=int)))
    stime = np.asarray(diag.get("freeb_nestor_solve_time_history", np.zeros((0,), dtype=float)))
    assert inc.size >= 1
    assert reused.size >= 1
    assert stime.size >= 1
    assert np.all((inc == 0) | (inc == 1))
    assert np.all((reused == 0) | (reused == 1))


def test_run_fixed_boundary_freeb_edge_coupling_can_be_disabled(tmp_path: Path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4", reason="netCDF4 required for mgrid loader test")

    mg = tmp_path / "mgrid_fb_edge_off.nc"
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
        ds.createVariable("br_001", "f8", ("phi", "zee", "rad"))[:] = 0.0
        ds.createVariable("bp_001", "f8", ("phi", "zee", "rad"))[:] = 1.0
        ds.createVariable("bz_001", "f8", ("phi", "zee", "rad"))[:] = 0.0

    inpath = tmp_path / "input.fb_edge_off"
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

    monkeypatch.setenv("VMEC_JAX_FREEB_COUPLE_EDGE", "0")
    run = run_fixed_boundary(
        inpath,
        solver="vmec2000_iter",
        max_iter=2,
        multigrid=False,
        verbose=False,
    )
    fb = run.result.diagnostics.get("free_boundary", {})
    assert bool(fb.get("enabled", False)) is True
    assert bool(fb.get("couple_edge", True)) is False

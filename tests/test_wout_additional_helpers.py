from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.integrals import cumrect_s_halfmesh
from vmec_jax.namelist import InData
from vmec_jax.kernels.lforbal import MU0 as LFORBAL_MU0
from vmec_jax.wout import (
    MU0,
    WoutData,
    _compute_equif_wout,
    _jxbforce_getbsubs_coeffs_lasym_false,
    _jxbforce_getbsubs_coeffs_lasym_true,
    equilibrium_iota_profiles_from_state,
    read_wout,
    write_wout,
)


def _synthetic_wout(path: Path) -> WoutData:
    ns = 3
    mnmax = 2
    mnmax_nyq = 2
    profile = np.asarray([0.0, 1.0, 2.0])
    main = np.arange(ns * mnmax, dtype=float).reshape(ns, mnmax)
    nyq = np.arange(ns * mnmax_nyq, dtype=float).reshape(ns, mnmax_nyq)

    return WoutData(
        path=path,
        ns=ns,
        mpol=2,
        ntor=0,
        nfp=1,
        lasym=False,
        signgs=1,
        mnmax=mnmax,
        mpol_nyq=1,
        ntor_nyq=0,
        mnmax_nyq=mnmax_nyq,
        xm=np.asarray([0, 1]),
        xn=np.asarray([0, 0]),
        xm_nyq=np.asarray([0, 1]),
        xn_nyq=np.asarray([0, 0]),
        rmnc=main + 1.0,
        rmns=np.zeros_like(main),
        zmnc=np.zeros_like(main),
        zmns=main + 2.0,
        lmnc=np.zeros_like(main),
        lmns=main + 3.0,
        phipf=profile + 1.0,
        chipf=profile + 2.0,
        phips=np.asarray([0.0, 1.5, 2.5]),
        iotaf=profile + 0.25,
        iotas=profile + 0.5,
        gmnc=nyq + 1.0,
        gmns=nyq + 2.0,
        bsupumnc=nyq + 3.0,
        bsupumns=nyq + 4.0,
        bsupvmnc=nyq + 5.0,
        bsupvmns=nyq + 6.0,
        bsubumnc=nyq + 7.0,
        bsubumns=nyq + 8.0,
        bsubvmnc=nyq + 9.0,
        bsubvmns=nyq + 10.0,
        bsubsmns=nyq + 11.0,
        bsubsmnc=nyq + 12.0,
        bmnc=nyq + 13.0,
        bmns=nyq + 14.0,
        wb=1.25,
        volume_p=2.5,
        gamma=0.0,
        wp=3.5,
        vp=profile + 3.0,
        pres=MU0 * np.asarray([0.0, 4.0, 8.0]),
        presf=MU0 * np.asarray([1.0, 5.0, 9.0]),
        fsqr=0.1,
        fsqz=0.2,
        fsql=0.3,
        fsqt=np.asarray([0.4, 0.5]),
        equif=profile + 4.0,
        phi=profile + 5.0,
        buco=profile + 6.0,
        bvco=profile + 7.0,
        jcuru=profile + 8.0,
        jcurv=profile + 9.0,
        raxis_cc=np.asarray([10.0]),
        zaxis_cs=np.asarray([11.0]),
        raxis_cs=np.asarray([12.0]),
        zaxis_cc=np.asarray([13.0]),
        Aminor_p=14.0,
        Rmajor_p=15.0,
        aspect=16.0,
        betatotal=17.0,
        betapol=18.0,
        betator=19.0,
        betaxis=20.0,
        ctor=21.0,
        DMerc=profile + 10.0,
        Dshear=profile + 11.0,
        Dwell=profile + 12.0,
        Dcurr=profile + 13.0,
        Dgeod=profile + 14.0,
        D_R=profile + 14.5,
        H=profile + 14.75,
        glasser_correction=profile + 14.875,
        glasser_shear_valid=np.asarray([False, True, True]),
        jdotb=profile + 15.0,
        bdotb=profile + 16.0,
        bdotgradv=profile + 17.0,
        ac=np.asarray([]),
        ac_aux_s=np.asarray([]),
        ac_aux_f=np.asarray([]),
        pcurr_type="power_series",
        piota_type="akima_spline",
    )


def test_write_wout_roundtrips_synthetic_profiles_and_default_aux_arrays(tmp_path: Path) -> None:
    wout = _synthetic_wout(tmp_path / "synthetic_source.nc")
    out_path = tmp_path / "wout_synthetic.nc"

    write_wout(out_path, wout)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_wout(out_path, wout)

    loaded = read_wout(out_path)

    assert loaded.ns == 3
    assert loaded.mpol == 2
    assert loaded.ntor == 0
    assert loaded.pcurr_type == "power_series"
    assert loaded.piota_type == "akima_spline"
    np.testing.assert_allclose(loaded.pres, wout.pres)
    np.testing.assert_allclose(loaded.presf, wout.presf)
    np.testing.assert_allclose(loaded.rmnc, wout.rmnc)
    np.testing.assert_allclose(loaded.bmns, wout.bmns)
    np.testing.assert_allclose(loaded.phi, wout.phi)
    np.testing.assert_allclose(loaded.D_R, wout.D_R)
    np.testing.assert_allclose(loaded.H, wout.H)
    np.testing.assert_allclose(loaded.glasser_correction, wout.glasser_correction)
    np.testing.assert_array_equal(loaded.glasser_shear_valid, wout.glasser_shear_valid)
    np.testing.assert_allclose(loaded.ac, np.zeros((21,)))
    np.testing.assert_allclose(loaded.ac_aux_s, -np.ones((1,)))
    np.testing.assert_allclose(loaded.ac_aux_f, np.zeros((1,)))


def test_read_wout_applies_optional_defaults_and_phi_fallback(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "wout_minimal_optional_defaults.nc"
    ns = 3
    mnmax = 2
    mnmax_nyq = 2

    def write_var(ds, name: str, dims: tuple[str, ...], data, dtype: str = "f8") -> None:
        var = ds.createVariable(name, dtype, dims)
        var[...] = np.asarray(data)

    with netcdf4.Dataset(path, mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("radius", ns)
        ds.createDimension("mn_mode", mnmax)
        ds.createDimension("mn_mode_nyq", mnmax_nyq)

        write_var(ds, "ns", (), ns, "i4")
        write_var(ds, "mpol", (), 2, "i4")
        write_var(ds, "ntor", (), 0, "i4")
        write_var(ds, "nfp", (), 1, "i4")
        write_var(ds, "signgs", (), -1, "i4")
        write_var(ds, "lasym__logical__", (), 0, "i4")
        write_var(ds, "xm", ("mn_mode",), [0, 1])
        write_var(ds, "xn", ("mn_mode",), [0, 0])
        write_var(ds, "xm_nyq", ("mn_mode_nyq",), [0, 1])
        write_var(ds, "xn_nyq", ("mn_mode_nyq",), [0, 0])

        coeff = np.arange(ns * mnmax, dtype=float).reshape(ns, mnmax)
        write_var(ds, "rmnc", ("radius", "mn_mode"), coeff + 1.0)
        write_var(ds, "zmns", ("radius", "mn_mode"), coeff + 2.0)
        write_var(ds, "lmns", ("radius", "mn_mode"), coeff + 3.0)
        phipf = np.asarray([2.0, 4.0, 6.0])
        write_var(ds, "phipf", ("radius",), phipf)
        write_var(ds, "chipf", ("radius",), [1.0, 3.0, 5.0])
        write_var(ds, "phips", ("radius",), [0.0, 4.0, 6.0])

        nyq = np.arange(ns * mnmax_nyq, dtype=float).reshape(ns, mnmax_nyq)
        write_var(ds, "gmnc", ("radius", "mn_mode_nyq"), nyq + 1.0)
        write_var(ds, "bsupumnc", ("radius", "mn_mode_nyq"), nyq + 2.0)
        write_var(ds, "bsupvmnc", ("radius", "mn_mode_nyq"), nyq + 3.0)
        write_var(ds, "wb", (), 1.5)
        write_var(ds, "volume_p", (), 2.5)
        write_var(ds, "DMerc", ("radius",), [0.0, 0.30, 0.0])
        write_var(ds, "DShear", ("radius",), [0.0, 0.16, 0.0])
        write_var(ds, "DWell", ("radius",), [0.0, 0.0, 0.0])
        write_var(ds, "DCurr", ("radius",), [0.0, -0.25, 0.0])
        write_var(ds, "DGeod", ("radius",), [0.0, 0.0, 0.0])

    wout = read_wout(path)
    expected_phi = np.asarray(cumrect_s_halfmesh(phipf, np.linspace(0.0, 1.0, ns)))

    assert wout.mnmax == mnmax
    assert wout.mnmax_nyq == mnmax_nyq
    assert wout.mpol_nyq == 1
    assert wout.ntor_nyq == 0
    assert wout.lasym is False
    assert wout.signgs == -1
    np.testing.assert_allclose(wout.phi, expected_phi)
    np.testing.assert_allclose(wout.rmns, np.zeros((ns, mnmax)))
    np.testing.assert_allclose(wout.gmns, np.zeros((ns, mnmax_nyq)))
    np.testing.assert_allclose(wout.bmnc, np.zeros((ns, mnmax_nyq)))
    np.testing.assert_allclose(wout.pres, np.zeros((ns,)))
    np.testing.assert_allclose(wout.H, [0.0, 0.25, 0.0])
    np.testing.assert_allclose(wout.glasser_correction, [0.0, (0.25 - 0.32) ** 2 / 0.64, 0.0])
    np.testing.assert_allclose(wout.D_R, [0.0, -0.30 + (0.25 - 0.32) ** 2 / 0.64, 0.0])
    np.testing.assert_array_equal(wout.glasser_shear_valid, [False, True, False])
    assert wout.fsqt.shape == (0,)
    np.testing.assert_allclose(wout.ac_aux_s, -np.ones((101,)))


def test_read_wout_rejects_bad_scalars_and_handles_single_surface_phi_fallback(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")

    def write_var(ds, name: str, dims: tuple[str, ...], data, dtype: str = "f8") -> None:
        var = ds.createVariable(name, dtype, dims)
        var[...] = np.asarray(data)

    bad_path = tmp_path / "wout_bad_scalars.nc"
    with netcdf4.Dataset(bad_path, mode="w", format="NETCDF3_CLASSIC") as ds:
        write_var(ds, "ns", (), 0, "i4")
        write_var(ds, "mpol", (), 1, "i4")
        write_var(ds, "ntor", (), 0, "i4")
        write_var(ds, "nfp", (), 1, "i4")
        write_var(ds, "signgs", (), 1, "i4")
        write_var(ds, "lasym__logical__", (), 0, "i4")

    with pytest.raises(ValueError, match="Incomplete or masked wout scalar metadata"):
        read_wout(bad_path)

    single_path = tmp_path / "wout_single_surface.nc"
    with netcdf4.Dataset(single_path, mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("radius", 1)
        ds.createDimension("mn_mode", 1)
        ds.createDimension("mn_mode_nyq", 1)

        write_var(ds, "ns", (), 1, "i4")
        write_var(ds, "mpol", (), 1, "i4")
        write_var(ds, "ntor", (), 0, "i4")
        write_var(ds, "nfp", (), 1, "i4")
        write_var(ds, "signgs", (), 1, "i4")
        write_var(ds, "lasym__logical__", (), 0, "i4")
        write_var(ds, "xm", ("mn_mode",), [0.0])
        write_var(ds, "xn", ("mn_mode",), [0.0])
        write_var(ds, "xm_nyq", ("mn_mode_nyq",), [0.0])
        write_var(ds, "xn_nyq", ("mn_mode_nyq",), [0.0])
        write_var(ds, "rmnc", ("radius", "mn_mode"), [[1.0]])
        write_var(ds, "zmns", ("radius", "mn_mode"), [[0.0]])
        write_var(ds, "lmns", ("radius", "mn_mode"), [[0.0]])
        write_var(ds, "phipf", ("radius",), [2.0])
        write_var(ds, "chipf", ("radius",), [0.0])
        write_var(ds, "phips", ("radius",), [0.0])
        write_var(ds, "gmnc", ("radius", "mn_mode_nyq"), [[1.0]])
        write_var(ds, "bsupumnc", ("radius", "mn_mode_nyq"), [[0.0]])
        write_var(ds, "bsupvmnc", ("radius", "mn_mode_nyq"), [[0.0]])
        write_var(ds, "wb", (), 0.0)
        write_var(ds, "volume_p", (), 0.0)

    wout = read_wout(single_path)

    assert wout.ns == 1
    np.testing.assert_allclose(wout.phi, [0.0])
    np.testing.assert_allclose(wout.rmns, np.zeros((1, 1)))
    np.testing.assert_allclose(wout.bmnc, np.zeros((1, 1)))


def test_compute_equif_wout_matches_weighted_currents_and_endpoint_rules() -> None:
    trig = SimpleNamespace(
        cosmui3=np.full((2, 1), 0.5),
        mscale=np.asarray([1.0]),
        cosnv=np.zeros((2, 1)),
    )
    ns = 4
    s = np.linspace(0.0, 1.0, ns)
    bsubu_levels = np.asarray([0.0, 1.0, 2.0, 4.0])
    bsubv_levels = np.asarray([0.0, 3.0, 6.0, 10.0])
    bsubu = np.broadcast_to(bsubu_levels[:, None, None], (ns, 2, 2)).copy()
    bsubv = np.broadcast_to(bsubv_levels[:, None, None], (ns, 2, 2)).copy()
    pres = np.asarray([0.0, 3.0, 1.0, 0.0])
    vp = np.asarray([0.0, 5.0, 7.0, 9.0])
    phipf = np.asarray([1.0, 2.0, 3.0, 4.0])
    chipf = np.asarray([0.5, 1.5, 2.5, 3.5])

    buco, bvco, jcuru, jcurv, equif = _compute_equif_wout(
        bsubu=bsubu,
        bsubv=bsubv,
        pres=pres,
        vp=vp,
        phipf=phipf,
        chipf=chipf,
        signgs=-1,
        trig=trig,
        s=s,
    )

    expected_buco = 2.0 * bsubu_levels
    expected_bvco = 2.0 * bsubv_levels
    hs = 1.0 / float(ns - 1)
    ohs = 1.0 / hs
    expected_jcuru = np.zeros((ns,))
    expected_jcurv = np.zeros((ns,))
    expected_vpphi = np.zeros((ns,))
    expected_presgrad = np.zeros((ns,))
    for js in range(1, ns - 1):
        expected_jcurv[js] = -ohs * (expected_buco[js + 1] - expected_buco[js])
        expected_jcuru[js] = ohs * (expected_bvco[js + 1] - expected_bvco[js])
        expected_vpphi[js] = 0.5 * (vp[js + 1] + vp[js])
        expected_presgrad[js] = (pres[js + 1] - pres[js]) * ohs

    expected_equif = np.zeros((ns,))
    for js in range(1, ns - 1):
        denom = (
            abs(expected_jcurv[js] * chipf[js])
            + abs(expected_jcuru[js] * phipf[js])
            + abs(expected_presgrad[js] * expected_vpphi[js])
        )
        raw = ((-phipf[js] * expected_jcuru[js] + chipf[js] * expected_jcurv[js]) / expected_vpphi[js]) + (
            expected_presgrad[js]
        )
        expected_equif[js] = raw * expected_vpphi[js] / denom

    for arr in (expected_equif, expected_jcuru, expected_jcurv):
        arr[0] = 2.0 * arr[1] - arr[2]
        arr[-1] = 2.0 * arr[-2] - arr[-3]

    np.testing.assert_allclose(buco, expected_buco)
    np.testing.assert_allclose(bvco, expected_bvco)
    np.testing.assert_allclose(jcuru, expected_jcuru / LFORBAL_MU0)
    np.testing.assert_allclose(jcurv, expected_jcurv / LFORBAL_MU0)
    np.testing.assert_allclose(equif, expected_equif)

    short = _compute_equif_wout(
        bsubu=bsubu[:2],
        bsubv=bsubv[:2],
        pres=pres[:2],
        vp=vp[:2],
        phipf=phipf[:2],
        chipf=chipf[:2],
        signgs=1,
        trig=trig,
        s=s[:2],
    )
    for profile in short:
        np.testing.assert_allclose(profile, np.zeros((2,)))


def test_compute_equif_wout_leaves_zero_denominator_surfaces_finite() -> None:
    trig = SimpleNamespace(
        cosmui3=np.ones((2, 1)),
        mscale=np.asarray([1.0]),
        cosnv=np.zeros((1, 1)),
    )
    ns = 4
    zeros = np.zeros((ns, 2, 1), dtype=float)

    buco, bvco, jcuru, jcurv, equif = _compute_equif_wout(
        bsubu=zeros,
        bsubv=zeros,
        pres=np.zeros((ns,), dtype=float),
        vp=np.ones((ns,), dtype=float),
        phipf=np.ones((ns,), dtype=float),
        chipf=np.ones((ns,), dtype=float),
        signgs=1,
        trig=trig,
        s=np.linspace(0.0, 1.0, ns),
    )

    for profile in (buco, bvco, jcuru, jcurv, equif):
        np.testing.assert_allclose(profile, np.zeros((ns,)))


def test_jxbforce_getbsubs_coefficients_cover_symmetric_and_asymmetric_collocation() -> None:
    theta = np.asarray([0.0, 0.5 * np.pi])
    zeta = np.asarray([0.0, np.pi])
    trig = SimpleNamespace(
        ntheta2=2,
        cosmu=np.stack([np.cos(m * theta) for m in range(2)], axis=1),
        sinmu=np.stack([np.sin(m * theta) for m in range(2)], axis=1),
        cosnv=np.stack([np.cos(n * zeta) for n in range(2)], axis=1),
        sinnv=np.stack([np.sin(n * zeta) for n in range(2)], axis=1),
    )
    frho = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    bsupu = np.full_like(frho, 0.7)
    bsupv = np.full_like(frho, 1.3)

    coeff = _jxbforce_getbsubs_coeffs_lasym_false(frho=frho, bsupu=bsupu, bsupv=bsupv, trig=trig, nfp=1)

    assert coeff is not None
    assert coeff.shape == (2, 3)
    assert np.all(np.isfinite(coeff))
    np.testing.assert_allclose(coeff[0, 1], 2.692307692307692, rtol=1.0e-12)
    assert _jxbforce_getbsubs_coeffs_lasym_false(frho=frho[:1], bsupu=bsupu, bsupv=bsupv, trig=trig, nfp=1) is None

    theta_lasym = np.linspace(0.0, np.pi, 3, endpoint=False)
    zeta_lasym = np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
    trig_lasym = SimpleNamespace(
        ntheta2=3,
        ntheta3=3,
        cosmu=np.stack([np.cos(m * theta_lasym) for m in range(3)], axis=1),
        sinmu=np.stack([np.sin(m * theta_lasym) for m in range(3)], axis=1),
        cosnv=np.stack([np.cos(n * zeta_lasym) for n in range(2)], axis=1),
        sinnv=np.stack([np.sin(n * zeta_lasym) for n in range(2)], axis=1),
    )
    frho_lasym = np.arange(1.0, 10.0).reshape(3, 3)
    bsupu_lasym = np.full_like(frho_lasym, 0.7)
    bsupv_lasym = np.full_like(frho_lasym, 1.3)

    coeff_lasym = _jxbforce_getbsubs_coeffs_lasym_true(
        frho=frho_lasym,
        bsupu=bsupu_lasym,
        bsupv=bsupv_lasym,
        trig=trig_lasym,
        nfp=1,
    )

    assert coeff_lasym is not None
    assert coeff_lasym.shape == (3, 3, 2)
    assert np.all(np.isfinite(coeff_lasym))
    assert np.linalg.norm(coeff_lasym) > 0.0
    assert _jxbforce_getbsubs_coeffs_lasym_true(
        frho=frho_lasym,
        bsupu=bsupu_lasym[:2],
        bsupv=bsupv_lasym,
        trig=trig_lasym,
        nfp=1,
    ) is None


def test_equilibrium_iota_profiles_iota_driven_branch_uses_prescribed_half_mesh_profile() -> None:
    static = SimpleNamespace(s=jnp.linspace(0.0, 1.0, 4))
    indata = InData(
        scalars={
            "NCURR": 0,
            "PHIEDGE": float(2.0 * np.pi),
            "PIOTA_TYPE": "power_series",
            "AI": [0.5, 0.25],
        },
        indexed={},
    )

    chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=None,
        static=static,
        indata=indata,
        signgs=1,
    )

    s_half = np.asarray([0.0, 1.0 / 6.0, 0.5, 5.0 / 6.0])
    expected_iotas = 0.5 + 0.25 * s_half
    expected_iotas[0] = 0.0
    expected_iotaf = np.asarray(
        [
            1.5 * expected_iotas[1] - 0.5 * expected_iotas[2],
            0.5 * (expected_iotas[1] + expected_iotas[2]),
            0.5 * (expected_iotas[2] + expected_iotas[3]),
            1.5 * expected_iotas[3] - 0.5 * expected_iotas[2],
        ]
    )

    np.testing.assert_allclose(np.asarray(iotas), expected_iotas)
    np.testing.assert_allclose(np.asarray(chips), expected_iotas)
    np.testing.assert_allclose(np.asarray(iotaf), expected_iotaf)

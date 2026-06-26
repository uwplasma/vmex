from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.boundary as boundary_module
import vmec_jax.energy as energy_module
import vmec_jax.profiles as profiles_module
import vmec_jax.kernels.bcovar as bcovar_module
import vmec_jax.kernels.residue as residue_module
import vmec_jax.wout as wout_module
from vmec_jax.config import VMECConfig
from vmec_jax.modes import vmec_mode_table
from vmec_jax.namelist import InData
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.wout import (
    MU0,
    WoutData,
    _icurv_full_mesh_from_indata,
    state_from_wout,
    wout_minimal_from_fixed_boundary,
    write_wout,
)


def _tiny_wout(path: Path, *, ns: int = 3, m_modes: tuple[int, ...] = (0, 1)) -> WoutData:
    K = len(m_modes)
    mn_nyq = max(1, K)
    mpol = max(m_modes) + 1
    profile = np.linspace(0.0, 1.0, ns)
    main = np.arange(ns * K, dtype=float).reshape(ns, K) if K else np.zeros((ns, 0))
    nyq = np.arange(ns * mn_nyq, dtype=float).reshape(ns, mn_nyq)
    zeros_main = np.zeros_like(main)
    zeros_nyq = np.zeros_like(nyq)
    xm = np.asarray(m_modes, dtype=int)
    xn = np.zeros((K,), dtype=int)
    xm_nyq = np.arange(mn_nyq, dtype=int)
    xn_nyq = np.zeros((mn_nyq,), dtype=int)

    return WoutData(
        path=path,
        ns=ns,
        mpol=mpol,
        ntor=0,
        nfp=1,
        lasym=False,
        signgs=1,
        mnmax=K,
        mpol_nyq=int(np.max(xm_nyq)) if xm_nyq.size else 0,
        ntor_nyq=0,
        mnmax_nyq=mn_nyq,
        xm=xm,
        xn=xn,
        xm_nyq=xm_nyq,
        xn_nyq=xn_nyq,
        rmnc=main + 1.0,
        rmns=zeros_main.copy(),
        zmnc=zeros_main.copy(),
        zmns=main + 0.5,
        lmnc=0.1 * (main + 1.0),
        lmns=0.2 * (main + 1.0),
        phipf=profile + 1.0,
        chipf=profile + 0.5,
        phips=profile.copy(),
        iotaf=profile + 0.1,
        iotas=profile + 0.2,
        gmnc=nyq + 1.0,
        gmns=zeros_nyq.copy(),
        bsupumnc=nyq + 2.0,
        bsupumns=zeros_nyq.copy(),
        bsupvmnc=nyq + 3.0,
        bsupvmns=zeros_nyq.copy(),
        bsubumnc=nyq + 4.0,
        bsubumns=zeros_nyq.copy(),
        bsubvmnc=nyq + 5.0,
        bsubvmns=zeros_nyq.copy(),
        bsubsmns=nyq + 6.0,
        bsubsmnc=zeros_nyq.copy(),
        bmnc=nyq + 7.0,
        bmns=zeros_nyq.copy(),
        wb=1.0,
        volume_p=2.0,
        gamma=0.0,
        wp=0.5,
        vp=profile + 1.0,
        pres=MU0 * (profile + 2.0),
        presf=MU0 * (profile + 3.0),
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        fsqt=np.asarray([0.0]),
        equif=np.zeros((ns,), dtype=float),
        phi=profile.copy(),
        buco=np.zeros((ns,), dtype=float),
        bvco=np.zeros((ns,), dtype=float),
        jcuru=np.zeros((ns,), dtype=float),
        jcurv=np.zeros((ns,), dtype=float),
        raxis_cc=np.asarray([1.0]),
        zaxis_cs=np.asarray([0.0]),
        raxis_cs=np.asarray([0.0]),
        zaxis_cc=np.asarray([0.0]),
        Aminor_p=0.0,
        Rmajor_p=0.0,
        aspect=0.0,
        betatotal=0.0,
        betapol=0.0,
        betator=0.0,
        betaxis=0.0,
        ctor=0.0,
        DMerc=np.zeros((ns,), dtype=float),
        Dshear=np.zeros((ns,), dtype=float),
        Dwell=np.zeros((ns,), dtype=float),
        Dcurr=np.zeros((ns,), dtype=float),
        Dgeod=np.zeros((ns,), dtype=float),
        jdotb=np.zeros((ns,), dtype=float),
        bdotb=np.zeros((ns,), dtype=float),
        bdotgradv=np.zeros((ns,), dtype=float),
        ac=np.asarray([1.0]),
        ac_aux_s=np.asarray([-1.0]),
        ac_aux_f=np.asarray([0.0]),
        pcurr_type="power_series",
        piota_type="power_series",
    )


def test_current_profile_single_surface_and_bad_profile_shape(monkeypatch):
    def fake_eval_profiles(indata, s):
        s_arr = np.asarray(s)
        if s_arr.size == 1:
            return {"current": np.asarray([1.0])}
        return {"current": np.asarray([1.0])}

    monkeypatch.setattr(profiles_module, "eval_profiles", fake_eval_profiles)
    indata = InData(scalars={"NCURR": 1, "CURTOR": 1.0}, indexed={})

    one = _icurv_full_mesh_from_indata(indata=indata, s_full=np.asarray([0.0]), signgs=1)
    np.testing.assert_allclose(np.asarray(one), [0.0])

    bad_shape = _icurv_full_mesh_from_indata(indata=indata, s_full=np.asarray([0.0, 0.5, 1.0]), signgs=1)
    np.testing.assert_allclose(np.asarray(bad_shape), [0.0, 0.0, 0.0])


def test_state_from_wout_single_surface_and_zero_lamscale_branches(tmp_path):
    single = _tiny_wout(tmp_path / "single.nc", ns=1, m_modes=(0,))
    single = WoutData(**{**single.__dict__, "phipf": np.asarray([0.0]), "phips": np.asarray([0.0])})
    state_single = state_from_wout(single)
    assert state_single.layout.ns == 1
    assert state_single.Rcos.shape == (1, 1)

    zero_lamscale = _tiny_wout(tmp_path / "zero_lamscale.nc", ns=3, m_modes=(0, 1))
    zero_lamscale = WoutData(
        **{
            **zero_lamscale.__dict__,
            "phipf": np.zeros((3,), dtype=float),
            "phips": np.zeros((3,), dtype=float),
        }
    )
    state_zero = state_from_wout(zero_lamscale)
    assert state_zero.layout.ns == 3
    assert state_zero.Lsin.shape == (3, 2)


def test_write_wout_logs_set_fill_off_failure(tmp_path, monkeypatch, capsys):
    netcdf4 = pytest.importorskip("netCDF4")
    real_dataset = netcdf4.Dataset

    class DatasetWrapper:
        def __init__(self, *args, **kwargs):
            self._dataset = real_dataset(*args, **kwargs)

        def __enter__(self):
            self._dataset.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._dataset.__exit__(exc_type, exc, tb)

        def __getattr__(self, name):
            return getattr(self._dataset, name)

        def set_fill_off(self):
            raise RuntimeError("fill toggle failed")

    monkeypatch.setattr(netcdf4, "Dataset", DatasetWrapper)
    monkeypatch.setenv("VMEC_JAX_MERCIER_LOG", "1")

    out = tmp_path / "wout_fill_fallback.nc"
    write_wout(out, _tiny_wout(tmp_path / "source.nc", ns=2, m_modes=(0,)))

    assert out.exists()
    assert "fill toggle failed" in capsys.readouterr().out


def test_nonconverged_wout_roundtrip_preserves_fields_and_status(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    source = _tiny_wout(tmp_path / "source.nc", ns=3, m_modes=(0, 1))
    source = WoutData(
        **{
            **source.__dict__,
            "fsqr": 0.1,
            "fsqz": 0.2,
            "fsql": 0.3,
            "betatotal": 4.0,
            "betapol": 5.0,
            "betator": 6.0,
            "ier_flag": 1,
            "vmec_jax_converged": False,
            "vmec_jax_status": "nonconverged",
        }
    )

    out = tmp_path / "wout_nonconverged.nc"
    write_wout(out, source)
    reread = wout_module.read_wout(out)

    assert reread.ier_flag == 1
    assert reread.vmec_jax_converged is False
    assert reread.vmec_jax_status == "nonconverged"
    assert reread.fsqr == pytest.approx(0.1)
    assert reread.fsqz == pytest.approx(0.2)
    assert reread.fsql == pytest.approx(0.3)
    assert reread.betatotal == pytest.approx(4.0)
    assert reread.betapol == pytest.approx(5.0)
    assert reread.betator == pytest.approx(6.0)

    with netcdf4.Dataset(out) as ds:
        assert int(ds.variables["ier_flag"][:]) == 1
        assert int(ds.variables["vmec_jax_converged__logical__"][:]) == 0
        assert "vmec_jax_status" in ds.variables


def test_wout_minimal_light_nonconverged_dumps_and_defaults(tmp_path, monkeypatch, capsys):
    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=True,
        ntheta=4,
        nzeta=2,
    )
    modes = vmec_mode_table(cfg.mpol, cfg.ntor)
    layout = StateLayout(ns=cfg.ns, K=modes.K, lasym=False)
    radial = np.linspace(0.0, 1.0, cfg.ns)[:, None]
    state = VMECState(
        layout=layout,
        Rcos=np.concatenate([1.0 + radial, 0.1 + 0.05 * radial], axis=1),
        Rsin=np.zeros((cfg.ns, modes.K)),
        Zcos=np.zeros((cfg.ns, modes.K)),
        Zsin=np.concatenate([np.zeros_like(radial), 0.2 + 0.05 * radial], axis=1),
        Lcos=np.zeros((cfg.ns, modes.K)),
        Lsin=np.zeros((cfg.ns, modes.K)),
    )
    static = SimpleNamespace(cfg=cfg, modes=modes, s=np.linspace(0.0, 1.0, cfg.ns))
    indata = InData(scalars={"LRFP": True, "GAMMA": 0.0, "AC": 2.5}, indexed={})

    monkeypatch.setenv("VMEC_JAX_WOUT_LIGHT", "1")
    monkeypatch.setenv("VMEC_JAX_WOUT_TIMING", "1")
    monkeypatch.setenv("VMEC_JAX_WOUT_FORCE_IEQUI1", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_TAG", "wave2")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSUB_PARITY", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSUBH", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSUB_SOURCES", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_WROUT_MODES", "1")

    def fake_geom(*, state, modes, trig):
        shape = (cfg.ns, int(np.asarray(trig.cosmu).shape[0]), int(np.asarray(trig.cosnv).shape[0]))
        base = np.ones(shape, dtype=float)
        return {
            "R": 2.0 * base,
            "Zu": 0.5 * base,
            "pr1_even": base,
            "pr1_odd": 0.1 * base,
            "pz1_even": 0.2 * base,
            "pz1_odd": 0.3 * base,
            "pru_even": 0.4 * base,
            "pru_odd": 0.5 * base,
            "pzu_even": 0.6 * base,
            "pzu_odd": 0.7 * base,
            "prv_even": 0.8 * base,
            "prv_odd": 0.9 * base,
            "pzv_even": base,
            "pzv_odd": 1.1 * base,
        }

    def fake_bcovar(**kwargs):
        trig = wout_module.vmec_trig_tables(
            ntheta=cfg.ntheta,
            nzeta=cfg.nzeta,
            nfp=cfg.nfp,
            mmax=cfg.mpol - 1,
            nmax=cfg.ntor,
            lasym=cfg.lasym,
        )
        shape = (cfg.ns, int(np.asarray(trig.cosmu).shape[0]), int(np.asarray(trig.cosnv).shape[0]))
        base = np.arange(np.prod(shape), dtype=float).reshape(shape) / 100.0
        jac = SimpleNamespace(
            sqrtg=1.0 + base,
            r12=1.0 + base,
            ru12=0.1 + base,
            zu12=0.2 + base,
            rs=0.3 + base,
            zs=0.4 + base,
        )
        return SimpleNamespace(
            bsupu=0.2 + base,
            bsupv=0.3 + base,
            bsubu=0.4 + base,
            bsubv=0.5 + base,
            bsubu_e=0.6 + base,
            bsubv_e=0.7 + base,
            bsubu_e_scaled=0.8 + base,
            bsubv_e_scaled=0.9 + base,
            bsubu_preblend=1.0 + base,
            bsubv_preblend=1.1 + base,
            bsubu_parity_even=1.2 + base,
            bsubu_parity_odd=1.3 + base,
            bsubv_parity_even=1.4 + base,
            bsubv_parity_odd=1.5 + base,
            bsq=2.0 + base,
            jac=jac,
        )

    monkeypatch.setattr(wout_module, "_vmec_realspace_geom_light_from_state", fake_geom)
    monkeypatch.setattr(wout_module, "_compute_aspectratio", lambda **kwargs: (1.0, 2.0, 2.0, 3.0, None))
    monkeypatch.setattr(wout_module, "_compute_bsubs_half_mesh", lambda **kwargs: np.zeros_like(kwargs["bsupu"]))
    monkeypatch.setattr(
        energy_module,
        "flux_profiles_from_indata",
        lambda indata, s, signgs: SimpleNamespace(
            chipf=np.asarray([0.2, 0.4, 0.6]),
            phips=np.asarray([0.0, 0.5, 1.0]),
            phipf=np.asarray([1.0, 1.5, 2.0]),
        ),
    )
    monkeypatch.setattr(
        profiles_module,
        "eval_profiles",
        lambda indata, s: {
            "pressure": np.linspace(0.0, 1.0, np.asarray(s).size),
            "iota": np.linspace(0.1, 0.3, np.asarray(s).size),
        },
    )
    monkeypatch.setattr(boundary_module, "boundary_from_indata", lambda indata, modes: SimpleNamespace(R_cos=np.asarray([1.5, 0.0])))
    monkeypatch.setattr(bcovar_module, "vmec_bcovar_half_mesh_from_wout", fake_bcovar)
    monkeypatch.setattr(
        residue_module,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **kwargs: SimpleNamespace(
            vp=np.asarray([1.0, 1.1, 1.2]),
            wb=np.asarray(2.0),
            wp=np.asarray(0.5),
            volume=np.asarray(0.25),
        ),
    )

    wout = wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_light.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=0.1,
        fsqz=0.2,
        fsql=0.3,
        fsqt=np.asarray([0.4, 0.5]),
        converged=False,
    )

    np.testing.assert_allclose(wout.buco, 0.0)
    np.testing.assert_allclose(wout.equif, 0.0)
    assert wout.Aminor_p == pytest.approx(1.0)
    assert wout.Rmajor_p == pytest.approx(2.0)
    assert wout.aspect == pytest.approx(2.0)
    assert wout.volume_p == pytest.approx(3.0)
    assert wout.betatotal == pytest.approx(0.25)
    assert wout.betapol == 0.0
    assert wout.betator == 0.0
    assert wout.ier_flag == 1
    assert wout.vmec_jax_converged is False
    assert wout.vmec_jax_status == "nonconverged"
    assert wout.ac[0] == pytest.approx(2.5)
    assert wout.pcurr_type == "power_series"
    assert wout.piota_type == "power_series"
    assert (tmp_path / "bsub_parity_dump.npz").exists()
    assert (tmp_path / "bsubh_wout.npz").exists()
    assert (tmp_path / "bsub_sources_wave2.npz").exists()
    assert (tmp_path / "wrout_modes_jax.dat").exists()
    assert "[vmec_jax wout timing]" in capsys.readouterr().out

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.boundary as boundary_module
import vmec_jax.driver as driver
import vmec_jax.energy as energy_module
import vmec_jax.field as field_module
import vmec_jax.profiles as profiles_module
import vmec_jax.kernels.bcovar as bcovar_module
import vmec_jax.kernels.forces as forces_module
import vmec_jax.kernels.residue as residue_module
import vmec_jax.wout as wout_module
from vmec_jax.config import VMECConfig
from vmec_jax.energy import FluxProfiles
from vmec_jax.modes import vmec_mode_table
from vmec_jax.namelist import InData
from vmec_jax.solve import SolveVmecResidualResult
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.wout import MU0, WoutData, state_from_wout, wout_minimal_from_fixed_boundary


def _tiny_wout(path: Path, *, ns: int = 3, m_modes: tuple[int, ...] = (0, 1, 2)) -> WoutData:
    k = len(m_modes)
    profile = np.linspace(0.0, 1.0, ns)
    main = np.arange(ns * k, dtype=float).reshape(ns, k) + 1.0
    zeros_main = np.zeros_like(main)
    return WoutData(
        path=path,
        ns=ns,
        mpol=max(m_modes) + 1,
        ntor=0,
        nfp=1,
        lasym=False,
        signgs=1,
        mnmax=k,
        mpol_nyq=max(m_modes),
        ntor_nyq=0,
        mnmax_nyq=k,
        xm=np.asarray(m_modes, dtype=int),
        xn=np.zeros((k,), dtype=int),
        xm_nyq=np.asarray(m_modes, dtype=int),
        xn_nyq=np.zeros((k,), dtype=int),
        rmnc=main,
        rmns=zeros_main.copy(),
        zmnc=zeros_main.copy(),
        zmns=0.5 * main,
        lmnc=0.1 * main,
        lmns=0.2 * main,
        phipf=np.ones(ns) * (2.0 * np.pi),
        chipf=np.zeros(ns),
        phips=profile.copy(),
        iotaf=np.zeros(ns),
        iotas=np.zeros(ns),
        gmnc=zeros_main.copy(),
        gmns=zeros_main.copy(),
        bsupumnc=zeros_main.copy(),
        bsupumns=zeros_main.copy(),
        bsupvmnc=zeros_main.copy(),
        bsupvmns=zeros_main.copy(),
        bsubumnc=zeros_main.copy(),
        bsubumns=zeros_main.copy(),
        bsubvmnc=zeros_main.copy(),
        bsubvmns=zeros_main.copy(),
        bsubsmns=zeros_main.copy(),
        bsubsmnc=zeros_main.copy(),
        bmnc=zeros_main.copy(),
        bmns=zeros_main.copy(),
        wb=0.0,
        volume_p=0.0,
        gamma=0.0,
        wp=0.0,
        vp=np.ones(ns),
        pres=MU0 * np.zeros(ns),
        presf=MU0 * np.zeros(ns),
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        fsqt=np.zeros(0),
        equif=np.zeros(ns),
        phi=np.zeros(ns),
        buco=np.zeros(ns),
        bvco=np.zeros(ns),
        jcuru=np.zeros(ns),
        jcurv=np.zeros(ns),
        raxis_cc=np.zeros(1),
        zaxis_cs=np.zeros(1),
        raxis_cs=np.zeros(1),
        zaxis_cc=np.zeros(1),
        Aminor_p=0.0,
        Rmajor_p=0.0,
        aspect=0.0,
        betatotal=0.0,
        betapol=0.0,
        betator=0.0,
        betaxis=0.0,
        ctor=0.0,
        DMerc=np.zeros(ns),
        Dshear=np.zeros(ns),
        Dwell=np.zeros(ns),
        Dcurr=np.zeros(ns),
        Dgeod=np.zeros(ns),
        jdotb=np.zeros(ns),
        bdotb=np.zeros(ns),
        bdotgradv=np.zeros(ns),
        ac=np.zeros(0),
        ac_aux_s=np.zeros(0),
        ac_aux_f=np.zeros(0),
        pcurr_type="",
        piota_type="",
    )


def test_state_from_wout_lambda_rejects_bad_shapes_and_handles_missing_m(monkeypatch, tmp_path: Path) -> None:
    base = _tiny_wout(tmp_path / "wout.nc", ns=4, m_modes=(0, 1))

    with pytest.raises(ValueError, match="Expected lam_wout with shape"):
        state_from_wout(replace(base, lmns=np.zeros((base.ns + 1, base.mnmax))))

    with pytest.raises(ValueError, match="Expected phipf with shape"):
        state_from_wout(replace(base, phipf=np.ones(base.ns + 1)))

    monkeypatch.setattr(wout_module, "assert_main_modes_match_wout", lambda **_kwargs: None)
    gapped = replace(base, mpol=2, xm=np.asarray([0, 2], dtype=int))

    state = state_from_wout(gapped)

    assert state.layout.ns == 4
    assert state.Lsin.shape == (4, 2)
    assert np.all(np.isfinite(state.Lsin))


def test_state_from_wout_recovers_internal_lambda_for_half_mesh_parity_branches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    base = _tiny_wout(tmp_path / "wout_lambda.nc", ns=4, m_modes=(0, 1, 2, 3))
    m_modes = np.asarray(base.xm, dtype=int)
    ns = int(base.ns)
    phipf_internal = np.asarray([1.0, 1.25, 1.5, 2.0])
    lamscale = 2.5

    internal_lmns = np.asarray(
        [
            [0.20, -0.10, 0.00, 0.00],
            [0.25, -0.125, 0.12, -0.04],
            [0.32, 0.02, 0.18, 0.03],
            [0.45, 0.07, 0.24, 0.08],
        ],
        dtype=float,
    )
    internal_lmnc = -0.4 * internal_lmns + np.asarray([0.01, 0.02, 0.03, 0.04])[None, :]
    internal_lmnc[1, m_modes == 0] = internal_lmnc[0, m_modes == 0] * phipf_internal[1] / phipf_internal[0]
    internal_lmnc[1, m_modes == 1] = internal_lmnc[0, m_modes == 1] * phipf_internal[1] / phipf_internal[0]
    internal_lmnc[0, m_modes > 1] = 0.0

    def wout_lambda_from_internal(internal: np.ndarray) -> np.ndarray:
        s = np.linspace(0.0, 1.0, ns, dtype=float)
        hs = float(s[1] - s[0])
        sqrts_f = np.zeros((ns + 1,), dtype=float)
        shalf_f = np.zeros((ns + 1,), dtype=float)
        for i in range(1, ns + 1):
            sqrts_f[i] = np.sqrt(max(hs * float(i - 1), 0.0))
            shalf_f[i] = np.sqrt(hs * abs(float(i) - 1.5))
        sqrts_f[ns] = 1.0

        sm_f = np.zeros((ns + 1,), dtype=float)
        sp_f = np.zeros((ns + 1,), dtype=float)
        for i in range(2, ns + 1):
            sm_f[i] = shalf_f[i] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
            if i < ns:
                sp_f[i] = shalf_f[i + 1] / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
            else:
                sp_f[i] = 1.0 / sqrts_f[i] if sqrts_f[i] != 0.0 else 0.0
        sp_f[1] = sm_f[2]

        pre_wrout = np.asarray(internal, dtype=float) * lamscale / phipf_internal[:, None]
        out = np.zeros_like(pre_wrout)
        for col, mval in enumerate(m_modes):
            if int(mval) == 1:
                out[1, col] = 0.5 * pre_wrout[0, col] * (sm_f[2] + sp_f[1])
            elif int(mval) == 0:
                out[1, col] = pre_wrout[0, col]
            elif (int(mval) % 2) == 0:
                out[1, col] = 0.5 * pre_wrout[1, col]
            else:
                out[1, col] = 0.5 * pre_wrout[1, col] * sm_f[2]

            for js in range(3, ns + 1):
                if (int(mval) % 2) == 0:
                    out[js - 1, col] = 0.5 * (pre_wrout[js - 1, col] + pre_wrout[js - 2, col])
                else:
                    out[js - 1, col] = 0.5 * (
                        pre_wrout[js - 1, col] * sm_f[js] + sp_f[js - 1] * pre_wrout[js - 2, col]
                    )
        return out

    monkeypatch.setattr(field_module, "lamscale_from_phips", lambda _phips, _s: lamscale)
    wout = replace(
        base,
        signgs=-1,
        phipf=-(2.0 * np.pi) * phipf_internal,
        lmns=wout_lambda_from_internal(internal_lmns),
        lmnc=wout_lambda_from_internal(internal_lmnc),
    )

    state = state_from_wout(wout)

    mode_scale = np.where(m_modes == 0, 1.0, 1.0 / np.sqrt(2.0))[None, :]
    np.testing.assert_allclose(state.Lsin, internal_lmns * mode_scale, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(state.Lcos, internal_lmnc * mode_scale, rtol=1.0e-13, atol=1.0e-13)


def _state_static_indata(*, ns: int = 3, ac_value=(), lasym: bool = False) -> tuple[VMECState, SimpleNamespace, InData]:
    cfg = VMECConfig(mpol=2, ntor=0, ns=ns, nfp=1, lasym=lasym, lthreed=False, lconm1=True, ntheta=4, nzeta=2)
    modes = vmec_mode_table(cfg.mpol, cfg.ntor)
    layout = StateLayout(ns=ns, K=modes.K, lasym=lasym)
    radial = np.linspace(0.0, 1.0, ns)[:, None]
    zeros = np.zeros((ns, modes.K))
    state = VMECState(
        layout=layout,
        Rcos=np.concatenate([1.0 + radial, 0.2 + radial], axis=1),
        Rsin=np.concatenate([0.05 + radial, 0.06 + radial], axis=1) if lasym else zeros.copy(),
        Zcos=np.concatenate([0.07 + radial, 0.08 + radial], axis=1) if lasym else zeros.copy(),
        Zsin=np.concatenate([np.zeros_like(radial), 0.3 + radial], axis=1),
        Lcos=np.concatenate([0.1 + radial, 0.2 + radial], axis=1),
        Lsin=np.concatenate([0.3 + radial, 0.4 + radial], axis=1),
    )
    static = SimpleNamespace(cfg=cfg, modes=modes, s=np.linspace(0.0, 1.0, ns))
    indata = InData(scalars={"GAMMA": 0.0, "AC": ac_value, "FTOL": 1.0e-12}, indexed={})
    return state, static, indata


def _patch_wout_minimal_dependencies(monkeypatch, *, cfg: VMECConfig, bc_scale: float = 1.0):
    trig = wout_module.vmec_trig_tables(
        ntheta=cfg.ntheta,
        nzeta=cfg.nzeta,
        nfp=cfg.nfp,
        mmax=cfg.mpol - 1,
        nmax=cfg.ntor,
        lasym=cfg.lasym,
        cache=False,
    )
    shape = (cfg.ns, int(np.asarray(trig.cosmu).shape[0]), int(np.asarray(trig.cosnv).shape[0]))
    base = bc_scale + np.arange(np.prod(shape), dtype=float).reshape(shape) / 100.0
    jac = SimpleNamespace(
        sqrtg=1.0 + base,
        r12=1.0 + base,
        ru12=0.1 + base,
        zu12=0.2 + base,
        rs=0.3 + base,
        zs=0.4 + base,
    )
    bc = SimpleNamespace(
        bsupu=0.2 + base,
        bsupv=0.3 + base,
        bsubu=0.4 + base,
        bsubv=0.5 + base,
        bsubu_e=1.4 + base,
        bsubv_e=1.5 + base,
        bsubu_e_scaled=2.4 + base,
        bsubv_e_scaled=2.5 + base,
        bsubu_preblend=3.4 + base,
        bsubv_preblend=3.5 + base,
        bsubu_parity_even=4.4 + base,
        bsubu_parity_odd=4.5 + base,
        bsubv_parity_even=4.6 + base,
        bsubv_parity_odd=4.7 + base,
        bsq=6.0 + base,
        jac=jac,
    )
    geom = {"R": np.ones(shape) * 2.0, "Zu": np.ones(shape) * 0.5}

    monkeypatch.setattr(wout_module, "vmec_realspace_geom_from_state", lambda **_kwargs: geom)
    monkeypatch.setattr(wout_module, "_vmec_realspace_geom_light_from_state", lambda **_kwargs: geom)
    monkeypatch.setattr(wout_module, "_compute_aspectratio", lambda **_kwargs: (1.0, 2.0, 2.0, 3.0, 4.0))
    monkeypatch.setattr(wout_module, "_compute_bsubs_half_mesh", lambda **_kwargs: np.ones(shape) * 0.25)
    monkeypatch.setattr(
        energy_module,
        "flux_profiles_from_indata",
        lambda _indata, s, signgs: SimpleNamespace(
            chipf=np.linspace(0.1, 0.2, len(s)),
            phips=np.linspace(0.0, 1.0, len(s)),
            phipf=np.ones(len(s)),
        ),
    )
    monkeypatch.setattr(
        profiles_module,
        "eval_profiles",
        lambda _indata, s: {
            "pressure": np.linspace(0.0, 0.1, len(s)),
            "iota": np.linspace(0.0, 0.2, len(s)),
        },
    )
    monkeypatch.setattr(boundary_module, "boundary_from_indata", lambda _indata, _modes: SimpleNamespace(R_cos=np.asarray([1.0, 0.0])))
    monkeypatch.setattr(bcovar_module, "vmec_bcovar_half_mesh_from_wout", lambda **_kwargs: bc)
    k_force = SimpleNamespace(
        bc=bc,
        pr1_even=np.zeros(shape),
        pr1_odd=np.zeros(shape),
        pz1_even=np.zeros(shape),
        pz1_odd=np.zeros(shape),
        pru_even=np.zeros(shape),
        pru_odd=np.zeros(shape),
        pzu_even=np.zeros(shape),
        pzu_odd=np.zeros(shape),
        prv_even=np.zeros(shape),
        prv_odd=np.zeros(shape),
        pzv_even=np.zeros(shape),
        pzv_odd=np.zeros(shape),
    )
    monkeypatch.setattr(forces_module, "vmec_forces_rz_from_wout", lambda **_kwargs: k_force)
    monkeypatch.setattr(
        residue_module,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **_kwargs: SimpleNamespace(vp=np.ones(cfg.ns), wb=np.asarray(2.0), wp=np.asarray(0.5), volume=np.asarray(0.25)),
    )
    monkeypatch.setattr(wout_module, "_compute_mercier", lambda **_kwargs: (np.zeros(cfg.ns),) * 8)
    return bc, shape


@pytest.mark.parametrize(
    ("env_name", "env_value", "expected_key"),
    [
        ("VMEC_JAX_MERCIER_BSUB_SOURCE", "bsubu_e_scaled", "bsubu_e_scaled"),
        ("VMEC_JAX_MERCIER_USE_BSUBE", "1", "bsubu_e"),
        ("VMEC_JAX_BSUB_FILTER_USE_BC_PARITY", "1", "bsubu_parity_even"),
    ],
)
def test_wout_minimal_bsub_source_filter_branches(monkeypatch, tmp_path: Path, env_name: str, env_value: str, expected_key: str) -> None:
    state, static, indata = _state_static_indata(ns=3, ac_value=object())
    bc, _shape = _patch_wout_minimal_dependencies(monkeypatch, cfg=static.cfg)
    seen: dict[str, np.ndarray] = {}

    def fake_parity_filter(**kwargs):
        seen["bsubu_even"] = np.asarray(kwargs["bsubu_even"])
        return np.asarray(kwargs["bsubu_even"]), np.asarray(kwargs["bsubv_even"])

    monkeypatch.setenv(env_name, env_value)
    monkeypatch.setenv("VMEC_JAX_DUMP_BSUB_PARITY_INPUTS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_TAG", "wave10")
    monkeypatch.setattr(wout_module, "_filter_bsubuv_jxbforce_parity", fake_parity_filter)

    out = wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        converged=True,
    )

    np.testing.assert_allclose(seen["bsubu_even"], getattr(bc, expected_key))
    assert out.ac.shape[0] == 21
    assert (tmp_path / "bsub_parity_inputs_wave10.npz").exists()


def test_wout_minimal_equif_correction_and_raw_filter_branch(monkeypatch, tmp_path: Path) -> None:
    state, static, indata = _state_static_indata(ns=3)
    bc, _shape = _patch_wout_minimal_dependencies(monkeypatch, cfg=static.cfg)
    seen: dict[str, np.ndarray] = {}

    def fake_equif_correction(**kwargs):
        seen["equif_bsubv"] = np.asarray(kwargs["bsubv"])
        return np.asarray(kwargs["bsubv"]) + 9.0

    def fake_loop_filter(**kwargs):
        seen["raw_filter_bsubv"] = np.asarray(kwargs["bsubv"])
        return np.asarray(kwargs["bsubu"]), np.asarray(kwargs["bsubv"])

    monkeypatch.setenv("VMEC_JAX_DISABLE_BSUBV_EQUI_CORR", "0")
    monkeypatch.setenv("VMEC_JAX_MERCIER_FILTER_FROM_RAW", "1")
    monkeypatch.setattr(wout_module, "_apply_bsubv_equif_correction", fake_equif_correction)
    monkeypatch.setattr(wout_module, "_filter_bsubuv_jxbforce_loop", fake_loop_filter)

    wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_raw.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        converged=True,
    )

    np.testing.assert_allclose(seen["equif_bsubv"], bc.bsubv)
    np.testing.assert_allclose(seen["raw_filter_bsubv"], bc.bsubv + 9.0)


def test_wout_minimal_lasym_loop_and_presym_dump_branch(monkeypatch, tmp_path: Path) -> None:
    state, static, indata = _state_static_indata(ns=3, lasym=True)
    _bc, _shape = _patch_wout_minimal_dependencies(monkeypatch, cfg=static.cfg)
    monkeypatch.setenv("VMEC_JAX_SKIP_BSUB_FILTER", "1")
    monkeypatch.setenv("VMEC_JAX_WROUT_LASYM_LOOP", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSUB_PRE_SYM", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_TAG", "lasym")

    out = wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_lasym.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        converged=True,
    )

    assert out.lasym is True
    assert out.gmns.shape == out.gmnc.shape
    assert out.bsubumns.shape == out.bsubumnc.shape
    dump = tmp_path / "bsub_pre_sym_jax_lasym.dat"
    assert dump.exists()
    text = dump.read_text(encoding="utf-8")
    assert "columns: js lt lz bsubu bsubv bsupu bsupv bsubs" in text
    assert "ntheta3=" in text


def test_wout_minimal_lasym_bsubvmns_uses_corrected_asymmetric_source_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    state, static, indata = _state_static_indata(ns=3, lasym=True)
    bc, shape = _patch_wout_minimal_dependencies(monkeypatch, cfg=static.cfg)
    trig = wout_module.vmec_trig_tables(
        ntheta=static.cfg.ntheta,
        nzeta=static.cfg.nzeta,
        nfp=static.cfg.nfp,
        mmax=static.cfg.mpol - 1,
        nmax=static.cfg.ntor,
        lasym=True,
        cache=False,
    )
    corrected_bsubv = np.asarray(bc.bsubv, dtype=float) + 100.0 + np.arange(np.prod(shape), dtype=float).reshape(shape)
    raw_bsubv_sym, raw_bsubv_asym = wout_module._vmec_symoutput_split(
        f=np.asarray(bc.bsubv, dtype=float),
        trig=trig,
    )
    _, corrected_bsubv_asym = wout_module._vmec_symoutput_split(f=corrected_bsubv, trig=trig)
    cos_inputs: list[np.ndarray] = []
    sin_inputs: list[np.ndarray] = []

    def fake_equif_correction(**kwargs):
        np.testing.assert_allclose(np.asarray(kwargs["bsubv"]), bc.bsubv)
        np.testing.assert_allclose(np.asarray(kwargs["bsubv_e"]), bc.bsubv_e)
        return corrected_bsubv

    def fake_cos_coeffs(*, f, modes, trig):
        del trig
        cos_inputs.append(np.asarray(f, dtype=float).copy())
        return np.full((shape[0], modes.K), float(len(cos_inputs)), dtype=float)

    def fake_sin_coeffs(*, f, modes, trig):
        del trig
        sin_inputs.append(np.asarray(f, dtype=float).copy())
        return np.full((shape[0], modes.K), -float(len(sin_inputs)), dtype=float)

    monkeypatch.setenv("VMEC_JAX_SKIP_BSUB_FILTER", "1")
    monkeypatch.setattr(wout_module, "_apply_bsubv_equif_correction", fake_equif_correction)
    monkeypatch.setattr(wout_module, "_vmec_wrout_nyquist_cos_coeffs", fake_cos_coeffs)
    monkeypatch.setattr(wout_module, "_vmec_wrout_nyquist_sin_coeffs", fake_sin_coeffs)

    out = wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_lasym_bsubv_source.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        converged=True,
    )

    assert out.lasym is True
    assert len(cos_inputs) >= 4
    assert len(sin_inputs) >= 5
    np.testing.assert_allclose(cos_inputs[3], raw_bsubv_sym)
    np.testing.assert_allclose(sin_inputs[4], corrected_bsubv_asym)
    assert not np.allclose(sin_inputs[4], raw_bsubv_asym)


def test_wout_minimal_force_bss_and_lambda_zero_or_single_surface_branches(monkeypatch, tmp_path: Path) -> None:
    state, static, indata = _state_static_indata(ns=1)
    bc, shape = _patch_wout_minimal_dependencies(monkeypatch, cfg=static.cfg)
    k_force = SimpleNamespace(
        bc=bc,
        pr1_even=np.zeros(shape),
        pr1_odd=np.zeros(shape),
        pz1_even=np.zeros(shape),
        pz1_odd=np.zeros(shape),
        pru_even=np.zeros(shape),
        pru_odd=np.zeros(shape),
        pzu_even=np.zeros(shape),
        pzu_odd=np.zeros(shape),
        prv_even=np.zeros(shape),
        prv_odd=np.zeros(shape),
        pzv_even=np.zeros(shape),
        pzv_odd=np.zeros(shape),
        crmn_e=np.ones(shape) * 7.0,
        czmn_e=np.ones(shape) * 8.0,
        bzmn_e=np.ones(shape),
        brmn_e=np.ones(shape),
        azmn_e=np.ones(shape),
        armn_e=np.ones(shape),
    )

    monkeypatch.setenv("VMEC_JAX_WOUT_FORCE_BSS", "1")
    monkeypatch.setenv("VMEC_JAX_WROUT_LOOP", "1")
    monkeypatch.setattr(forces_module, "vmec_forces_rz_from_wout", lambda **_kwargs: k_force)
    monkeypatch.setattr(field_module, "lamscale_from_phips", lambda _phips, _s: 0.0)

    out_zero_scale = wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_force.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        converged=True,
    )

    np.testing.assert_allclose(out_zero_scale.lmns, 0.0)
    np.testing.assert_allclose(out_zero_scale.lmnc, 0.0)

    monkeypatch.setattr(field_module, "lamscale_from_phips", lambda _phips, _s: 1.0)
    out_single_surface = wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_single.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        converged=True,
    )

    np.testing.assert_allclose(out_single_surface.lmns, 0.0)
    np.testing.assert_allclose(out_single_surface.lmnc, 0.0)


def test_wout_minimal_rejects_mismatched_layout_and_lambda_shape(monkeypatch, tmp_path: Path) -> None:
    state, static, indata = _state_static_indata(ns=3)
    _patch_wout_minimal_dependencies(monkeypatch, cfg=static.cfg)

    bad_layout = replace(state, layout=StateLayout(ns=3, K=99, lasym=False))
    with pytest.raises(ValueError, match="state mode count"):
        wout_minimal_from_fixed_boundary(
            path=tmp_path / "bad_layout.nc",
            state=bad_layout,
            static=static,
            indata=indata,
            signgs=1,
            fsqr=0.0,
            fsqz=0.0,
            fsql=0.0,
        )

    bad_lambda = replace(state, Lsin=np.zeros((4, state.layout.K)))
    with pytest.raises(ValueError, match="Expected lam_full with shape"):
        wout_minimal_from_fixed_boundary(
            path=tmp_path / "bad_lambda.nc",
            state=bad_lambda,
            static=static,
            indata=indata,
            signgs=1,
            fsqr=0.0,
            fsqz=0.0,
            fsql=0.0,
        )


def _write_input(tmp_path: Path, body: str = "") -> Path:
    path = tmp_path / "input.wave10"
    path.write_text("&INDATA\n" + body + "/\n")
    return path


def _driver_state(ns: int):
    arr = np.zeros((int(ns), 1), dtype=float)
    return SimpleNamespace(layout=SimpleNamespace(ns=int(ns)), Rcos=arr, Rsin=arr, Zcos=arr, Zsin=arr, Lcos=arr, Lsin=arr)


def _patch_light_driver(monkeypatch, *, ns: int = 3) -> None:
    cfg = VMECConfig(mpol=2, ntor=0, ns=ns, nfp=1, lasym=False, lthreed=False, lconm1=True, ntheta=4, nzeta=1)
    indata = InData(scalars={"NITER": 2, "FTOL": 1.0e-12, "DELT": 0.125, "PHIEDGE": 1.0}, indexed={})

    def fake_static(cfg_in, **_kwargs):
        return SimpleNamespace(
            cfg=cfg_in,
            modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0]), K=1),
            s=np.linspace(0.0, 1.0, int(cfg_in.ns)),
        )

    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "0")
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "cpu")
    monkeypatch.setattr(driver, "load_config", lambda _path: (cfg, indata))
    monkeypatch.setattr(driver, "validate_free_boundary_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "prepare_mgrid_for_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "build_static", fake_static)
    monkeypatch.setattr(driver, "boundary_from_indata", lambda *args, **kwargs: object())
    monkeypatch.setattr(driver, "initial_guess_from_boundary", lambda static, *_args, **_kwargs: _driver_state(static.cfg.ns))
    monkeypatch.setattr(
        driver,
        "flux_profiles_from_indata",
        lambda _indata, s, *, signgs: FluxProfiles(
            phipf=np.ones_like(np.asarray(s, dtype=float)),
            chipf=np.zeros_like(np.asarray(s, dtype=float)),
            phips=np.ones_like(np.asarray(s, dtype=float)),
            signgs=int(signgs),
            lamscale=np.asarray(1.0),
        ),
    )
    monkeypatch.setattr(driver, "eval_profiles", lambda _indata, s: {"pressure": np.zeros_like(np.asarray(s))})
    monkeypatch.setattr(driver, "_final_flux_profiles_from_state", lambda **kwargs: (kwargs["flux_local"], kwargs["prof_local"]))


def _driver_result(state, *, max_iter: int, fsq: float, converged: bool) -> SolveVmecResidualResult:
    return SolveVmecResidualResult(
        state=state,
        n_iter=max(0, int(max_iter) - 1),
        w_history=np.asarray([float(fsq)], dtype=float),
        fsqr2_history=np.asarray([float(fsq)], dtype=float),
        fsqz2_history=np.asarray([0.0], dtype=float),
        fsql2_history=np.asarray([0.0], dtype=float),
        grad_rms_history=np.asarray([], dtype=float),
        step_history=np.asarray([], dtype=float),
        diagnostics={
            "converged": bool(converged),
            "final_fsqr": float(fsq),
            "final_fsqz": 0.0,
            "final_fsql": 0.0,
            "resume_state": {"time_step": 0.25},
        },
    )


def test_driver_cli_finisher_strict_result_without_converged_flag_skips_attempts(monkeypatch, tmp_path: Path) -> None:
    _patch_light_driver(monkeypatch)
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append(kwargs)
        result = _driver_result(state, max_iter=kwargs["max_iter"], fsq=0.0, converged=False)
        diagnostics = dict(result.diagnostics)
        diagnostics.pop("converged", None)
        diagnostics["ftol"] = 1.0e-12
        return replace(result, diagnostics=diagnostics)

    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        _write_input(tmp_path),
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=2,
        multigrid=False,
        verbose=False,
        cli_fixed_boundary_mode=True,
        jit_forces=False,
        finish_policy="converge",
    )

    assert len(calls) == 1
    assert run.result.diagnostics["converged"] is True
    np.testing.assert_array_equal(run.result.diagnostics["cli_fixed_boundary_finish_budgets"], np.zeros(0, dtype=int))


def test_driver_cli_finisher_budget_cap_exhausts_before_parity_attempt(monkeypatch, tmp_path: Path) -> None:
    _patch_light_driver(monkeypatch)
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append(kwargs)
        fsq = 10.0 / float(len(calls))
        return _driver_result(state, max_iter=kwargs["max_iter"], fsq=fsq, converged=False)

    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        _write_input(tmp_path),
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=1,
        multigrid=False,
        verbose=False,
        cli_fixed_boundary_mode=True,
        jit_forces=False,
        finish_policy="converge",
    )

    diag = run.result.diagnostics
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["accelerated", "accelerated"]
    assert diag["cli_fixed_boundary_finish_budget_cap"] == 2
    assert diag["cli_fixed_boundary_finish_budget_exhausted"] is True


def test_driver_budget_and_stage_switch_remaining_branches() -> None:
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=8, ns_stages=[]) == 8
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=3, ns_stages=[4, 4, 4]) == [3, 0, 1]
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=0.0,
            best_total_fsq=1.0,
            target_total_fsq=0.5,
            chunk_iters=2,
            remaining_budget=5,
        )
        == "nondecreasing_total_fsq"
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=1.0,
            best_total_fsq=0.0,
            target_total_fsq=0.5,
            chunk_iters=2,
            remaining_budget=5,
        )
        is None
    )

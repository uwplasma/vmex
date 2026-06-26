from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.boundary as boundary_module
import vmec_jax.energy as energy_module
import vmec_jax.profiles as profiles_module
import vmec_jax.kernels.bcovar as bcovar_module
import vmec_jax.kernels.forces as forces_module
import vmec_jax.kernels.residue as residue_module
import vmec_jax.wout as wout_module
from vmec_jax.config import VMECConfig
from vmec_jax.modes import nyquist_mode_table_from_grid
from vmec_jax.modes import vmec_mode_table
from vmec_jax.namelist import InData
from vmec_jax.state import StateLayout
from vmec_jax.state import VMECState


def _case_objects():
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
    s = np.linspace(0.0, 1.0, cfg.ns)
    radial = s[:, None]
    layout = StateLayout(ns=cfg.ns, K=modes.K, lasym=False)
    state = VMECState(
        layout=layout,
        Rcos=np.concatenate([1.0 + 0.1 * radial, 0.15 + 0.02 * radial], axis=1),
        Rsin=np.zeros((cfg.ns, modes.K)),
        Zcos=np.zeros((cfg.ns, modes.K)),
        Zsin=np.concatenate([np.zeros_like(radial), 0.25 + 0.03 * radial], axis=1),
        Lcos=np.zeros((cfg.ns, modes.K)),
        Lsin=0.01 * np.ones((cfg.ns, modes.K)),
    )
    static = SimpleNamespace(cfg=cfg, modes=modes, s=s)
    indata = InData(scalars={"GAMMA": 0.0, "LBSUBS": False}, indexed={})
    return cfg, modes, state, static, indata


def _trig_for_cfg(cfg: VMECConfig):
    nyq_modes = nyquist_mode_table_from_grid(
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
    )
    mmax_nyq = int(np.max(nyq_modes.m)) if int(nyq_modes.K) > 0 else 0
    nmax_nyq = int(np.max(np.abs(nyq_modes.n))) if int(nyq_modes.K) > 0 else 0
    return wout_module.vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        mmax=max(int(cfg.mpol) - 1, mmax_nyq),
        nmax=max(int(cfg.ntor), nmax_nyq),
        lasym=bool(cfg.lasym),
    )


def _fake_bcovar(shape: tuple[int, int, int]):
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) / 100.0
    jac = SimpleNamespace(
        sqrtg=1.0 + base,
        r12=1.2 + base,
        ru12=0.2 + base,
        zu12=0.3 + base,
        rs=0.4 + base,
        zs=0.5 + base,
    )
    return SimpleNamespace(
        bsupu=0.10 + base,
        bsupv=0.20 + base,
        bsubu=0.30 + base,
        bsubv=0.40 + base,
        bsubu_e=1.30 + base,
        bsubv_e=1.40 + base,
        bsubu_e_scaled=2.30 + base,
        bsubv_e_scaled=2.40 + base,
        bsubu_parity_even=3.30 + base,
        bsubu_parity_odd=3.40 + base,
        bsubv_parity_even=3.50 + base,
        bsubv_parity_odd=3.60 + base,
        bsq=5.0 + base,
        jac=jac,
    )


def _install_fast_wout_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    force_bss: bool = False,
    beta: tuple[float, float, float, float] = (1.25, 2.5, 3.75, 5.0),
    mercier_impl=None,
):
    cfg, _modes, _state, _static, _indata = _case_objects()
    trig = _trig_for_cfg(cfg)
    reduced_shape = (cfg.ns, int(np.asarray(trig.cosmu).shape[0]), int(np.asarray(trig.cosnv).shape[0]))
    full_shape = (cfg.ns, int(trig.ntheta1), reduced_shape[2])
    bc = _fake_bcovar(reduced_shape)
    calls: dict[str, object] = {"bc": bc, "bsubs_calls": [], "mercier_calls": []}

    monkeypatch.delenv("VMEC_JAX_WOUT_LIGHT", raising=False)
    monkeypatch.setenv("VMEC_JAX_SKIP_BSUB_FILTER", "1")
    monkeypatch.setattr(
        energy_module,
        "flux_profiles_from_indata",
        lambda indata, s, signgs: SimpleNamespace(
            chipf=np.asarray([0.2, 0.3, 0.4]),
            phips=np.asarray([0.0, 0.5, 1.0]),
            phipf=np.asarray([0.0, 1.0, 1.5]),
        ),
    )
    monkeypatch.setattr(
        profiles_module,
        "eval_profiles",
        lambda indata, s: {
            "pressure": np.asarray([0.0, 0.6, 0.9]),
            "iota": np.asarray([0.0, 0.2, 0.25]),
        },
    )
    monkeypatch.setattr(
        boundary_module,
        "boundary_from_indata",
        lambda indata, modes: SimpleNamespace(R_cos=np.asarray([1.7, 0.0])),
    )
    monkeypatch.setattr(
        residue_module,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **kwargs: SimpleNamespace(
            vp=np.asarray([1.0, 1.25, 1.5]),
            wb=np.asarray(4.0),
            wp=np.asarray(1.0),
            volume=np.asarray(0.5),
        ),
    )
    monkeypatch.setattr(
        wout_module,
        "vmec_realspace_geom_from_state",
        lambda *, state, modes, trig: {
            "R": np.ones(reduced_shape, dtype=float),
            "Zu": 0.5 * np.ones(reduced_shape, dtype=float),
        },
    )
    monkeypatch.setattr(wout_module, "_compute_aspectratio", lambda **kwargs: (0.5, 1.5, 3.0, 2.0, 1.0))
    monkeypatch.setattr(
        wout_module,
        "_compute_equif_wout",
        lambda **kwargs: tuple(np.zeros((cfg.ns,), dtype=float) for _ in range(5)),
    )
    monkeypatch.setattr(wout_module, "_compute_eqfor_betaxis", lambda **kwargs: beta[3])
    monkeypatch.setattr(wout_module, "_compute_eqfor_beta", lambda **kwargs: beta)

    def fake_bsubs(**kwargs):
        calls["bsubs_calls"].append(kwargs)
        return np.zeros_like(np.asarray(kwargs["bsupu"], dtype=float))

    def fake_mercier(**kwargs):
        calls["mercier_calls"].append(kwargs)
        if mercier_impl is not None:
            return mercier_impl(**kwargs)
        ns = cfg.ns
        return tuple(np.full((ns,), float(i), dtype=float) for i in range(8))

    monkeypatch.setattr(wout_module, "_compute_bsubs_half_mesh", fake_bsubs)
    monkeypatch.setattr(wout_module, "_compute_mercier", fake_mercier)

    if force_bss:
        monkeypatch.delenv("VMEC_JAX_WOUT_FAST_BCOVAR", raising=False)
        monkeypatch.setenv("VMEC_JAX_WOUT_FORCE_BSS", "1")
        force_base = np.arange(np.prod(full_shape), dtype=float).reshape(full_shape)
        k_force = SimpleNamespace(
            bc=bc,
            pr1_even=np.zeros(full_shape, dtype=float),
            pr1_odd=np.zeros(full_shape, dtype=float),
            pz1_even=np.zeros(full_shape, dtype=float),
            pz1_odd=np.zeros(full_shape, dtype=float),
            pru_even=np.zeros(full_shape, dtype=float),
            pru_odd=np.zeros(full_shape, dtype=float),
            pzu_even=np.zeros(full_shape, dtype=float),
            pzu_odd=np.zeros(full_shape, dtype=float),
            prv_even=np.zeros(full_shape, dtype=float),
            prv_odd=np.zeros(full_shape, dtype=float),
            pzv_even=np.zeros(full_shape, dtype=float),
            pzv_odd=np.zeros(full_shape, dtype=float),
            crmn_e=10.0 + force_base,
            czmn_e=20.0 + force_base,
            bzmn_e=30.0 + force_base,
            brmn_e=40.0 + force_base,
            azmn_e=50.0 + force_base,
            armn_e=60.0 + force_base,
        )
        calls["k_force"] = k_force
        monkeypatch.setattr(forces_module, "vmec_forces_rz_from_wout", lambda **kwargs: k_force)
    else:
        monkeypatch.setenv("VMEC_JAX_WOUT_FAST_BCOVAR", "1")
        monkeypatch.delenv("VMEC_JAX_WOUT_FORCE_BSS", raising=False)
        monkeypatch.setattr(bcovar_module, "vmec_bcovar_half_mesh_from_wout", lambda **kwargs: bc)

    return calls


def _fake_force_payload(bc, shape: tuple[int, int, int]):
    zeros = np.zeros(shape, dtype=float)
    return SimpleNamespace(
        bc=bc,
        pr1_even=zeros,
        pr1_odd=zeros,
        pz1_even=zeros,
        pz1_odd=zeros,
        pru_even=zeros,
        pru_odd=zeros,
        pzu_even=zeros,
        pzu_odd=zeros,
        prv_even=zeros,
        prv_odd=zeros,
        pzv_even=zeros,
        pzv_odd=zeros,
    )


def _run_wout_minimal(monkeypatch: pytest.MonkeyPatch, tmp_path, **stub_kwargs):
    _cfg, _modes, state, static, indata = _case_objects()
    converged = stub_kwargs.pop("converged", True)
    calls = _install_fast_wout_stubs(monkeypatch, **stub_kwargs)
    out = wout_module.wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_branch.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=0.01,
        fsqz=0.02,
        fsql=0.03,
        fsqt=np.asarray([0.04]),
        converged=converged,
    )
    return out, calls


def test_reuse_final_bcovar_env_uses_payload_even_on_fast_path(tmp_path, monkeypatch):
    cfg, _modes, state, static, indata = _case_objects()
    trig = _trig_for_cfg(cfg)
    full_shape = (cfg.ns, int(trig.ntheta1), int(np.asarray(trig.cosnv).shape[0]))
    calls = _install_fast_wout_stubs(monkeypatch)
    payload = _fake_force_payload(calls["bc"], full_shape)

    bcovar_calls = []

    def unexpected_bcovar(**kwargs):
        bcovar_calls.append(kwargs)
        raise AssertionError("fast bcovar recompute should be skipped")

    monkeypatch.setenv("VMEC_JAX_WOUT_FAST_BCOVAR", "1")
    monkeypatch.setenv("VMEC_JAX_WOUT_REUSE_FINAL_BCOVAR", "1")
    monkeypatch.setattr(bcovar_module, "vmec_bcovar_half_mesh_from_wout", unexpected_bcovar)

    wout = wout_module.wout_minimal_from_fixed_boundary(
        path=tmp_path / "wout_reuse_payload.nc",
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        fsqr=0.01,
        fsqz=0.02,
        fsql=0.03,
        force_payload_override=payload,
    )

    assert bcovar_calls == []
    assert np.asarray(wout.bmnc).shape[0] == cfg.ns


def test_enable_bsubs_env_overrides_namelist_for_mercier(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_ENABLE_BSUBS_CORR", "1")

    _wout, calls = _run_wout_minimal(monkeypatch, tmp_path)

    mercier_call = calls["mercier_calls"][0]
    assert mercier_call["lbsubs"] is True


def test_mercier_bsub_source_env_uses_scaled_bcovar_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_MERCIER_BSUB_SOURCE", "bsubu_e_scaled")

    _wout, calls = _run_wout_minimal(monkeypatch, tmp_path)

    bc = calls["bc"]
    mercier_call = calls["mercier_calls"][0]
    np.testing.assert_allclose(mercier_call["bsubu"], bc.bsubu_e_scaled)
    np.testing.assert_allclose(mercier_call["bsubv"], bc.bsubv_e_scaled)


def test_mercier_failure_is_swallowed_unless_strict(tmp_path, monkeypatch):
    def raise_mercier(**kwargs):
        raise RuntimeError("synthetic mercier failure")

    wout, _calls = _run_wout_minimal(monkeypatch, tmp_path, mercier_impl=raise_mercier)
    np.testing.assert_allclose(wout.DMerc, 0.0)
    np.testing.assert_allclose(wout.jdotb, 0.0)

    monkeypatch.setenv("VMEC_JAX_STRICT_WOUT_DIAGNOSTICS", "1")
    with pytest.raises(RuntimeError, match="synthetic mercier failure"):
        _run_wout_minimal(monkeypatch, tmp_path, mercier_impl=raise_mercier)


def test_nonconverged_beta_retention_and_legacy_zero_env(tmp_path, monkeypatch):
    beta = (1.25, 2.5, 3.75, 5.0)

    wout_default, _calls = _run_wout_minimal(monkeypatch, tmp_path, beta=beta, converged=False)
    assert wout_default.betapol == pytest.approx(beta[0])
    assert wout_default.betator == pytest.approx(beta[1])
    assert wout_default.betatotal == pytest.approx(beta[2])
    assert wout_default.betaxis == pytest.approx(beta[3])
    assert wout_default.Aminor_p == pytest.approx(0.5)
    assert wout_default.Rmajor_p == pytest.approx(1.5)
    assert wout_default.aspect == pytest.approx(3.0)
    assert wout_default.volume_p == pytest.approx(2.0)
    assert wout_default.vmec_jax_converged is False

    monkeypatch.setenv("VMEC_JAX_WOUT_ZERO_NONCONVERGED_BETA", "1")
    wout_legacy, _calls = _run_wout_minimal(monkeypatch, tmp_path, beta=beta, converged=False)
    assert wout_legacy.betatotal == 0.0
    assert wout_legacy.betapol == 0.0
    assert wout_legacy.betator == 0.0
    assert wout_legacy.betaxis == pytest.approx(beta[3])


def test_forced_bss_uses_symforced_force_kernel_arrays(tmp_path, monkeypatch):
    _wout, calls = _run_wout_minimal(monkeypatch, tmp_path, force_bss=True)

    cfg, _modes, _state, _static, _indata = _case_objects()
    trig = _trig_for_cfg(cfg)
    k_force = calls["k_force"]
    bsubs_call = calls["bsubs_calls"][0]

    expected_bsupu = wout_module._vmec_symforce_apply(f=k_force.crmn_e, trig=trig, kind="crs")
    expected_bsupv = wout_module._vmec_symforce_apply(f=k_force.czmn_e, trig=trig, kind="czs")
    expected_rs = wout_module._vmec_symforce_apply(f=k_force.bzmn_e, trig=trig, kind="bzs")
    expected_zs = wout_module._vmec_symforce_apply(f=k_force.brmn_e, trig=trig, kind="brs")
    expected_ru12 = wout_module._vmec_symforce_apply(f=k_force.azmn_e, trig=trig, kind="azs")
    expected_zu12 = wout_module._vmec_symforce_apply(f=k_force.armn_e, trig=trig, kind="ars")

    np.testing.assert_allclose(bsubs_call["bsupu"], expected_bsupu)
    np.testing.assert_allclose(bsubs_call["bsupv"], expected_bsupv)
    np.testing.assert_allclose(bsubs_call["force_rs"], expected_rs)
    np.testing.assert_allclose(bsubs_call["force_zs"], expected_zs)
    np.testing.assert_allclose(bsubs_call["force_ru12"], expected_ru12)
    np.testing.assert_allclose(bsubs_call["force_zu12"], expected_zu12)


def test_bss_scalxc_env_and_single_surface_guard(monkeypatch):
    s = np.asarray([0.0, 0.25, 1.0])
    arr = np.ones((3, 1, 1), dtype=float)

    monkeypatch.delenv("VMEC_JAX_BSS_UNDO_SCALXC", raising=False)
    unchanged = wout_module._undo_bss_scalxc_if_enabled(s, arr)[0]
    np.testing.assert_allclose(unchanged, arr)

    monkeypatch.setenv("VMEC_JAX_BSS_UNDO_SCALXC", "1")
    scaled = wout_module._undo_bss_scalxc_if_enabled(s, arr)[0]
    np.testing.assert_allclose(scaled[:, 0, 0], [0.5, 0.5, 1.0])

    modes = vmec_mode_table(1, 0)
    trig = wout_module.vmec_trig_tables(ntheta=4, nzeta=2, nfp=1, mmax=0, nmax=0, lasym=False)
    shape = (1, int(np.asarray(trig.cosmu).shape[0]), int(np.asarray(trig.cosnv).shape[0]))
    state = VMECState(
        layout=StateLayout(ns=1, K=modes.K, lasym=False),
        Rcos=np.ones((1, modes.K), dtype=float),
        Rsin=np.zeros((1, modes.K), dtype=float),
        Zcos=np.zeros((1, modes.K), dtype=float),
        Zsin=np.zeros((1, modes.K), dtype=float),
        Lcos=np.zeros((1, modes.K), dtype=float),
        Lsin=np.zeros((1, modes.K), dtype=float),
    )

    bsubs = wout_module._compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=np.asarray([0.0]),
        lconm1=True,
        lthreed=False,
        lasym=False,
        bsupu=np.ones(shape, dtype=float),
        bsupv=np.ones(shape, dtype=float),
        trig=trig,
        geom={},
    )
    np.testing.assert_allclose(bsubs, 0.0)


def test_bsubs_correction_early_guards_return_inputs():
    trig = wout_module.vmec_trig_tables(ntheta=4, nzeta=2, nfp=1, mmax=1, nmax=0, lasym=False)
    shape = (2, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))
    bsubs = np.arange(np.prod(shape), dtype=float).reshape(shape)
    bsubsu = bsubs + 10.0
    bsubsv = bsubs + 20.0

    out = wout_module._jxbforce_apply_bsubs_correction_lasym_false(
        bsubu=bsubs,
        bsubv=bsubs,
        bsubs=bsubs,
        bsubsu=bsubsu,
        bsubsv=bsubsv,
        bsupu=bsubs,
        bsupv=bsubs,
        sqrtg=np.ones(shape, dtype=float),
        pres=np.ones((shape[0],), dtype=float),
        vp=np.ones((shape[0],), dtype=float),
        hs=0.5,
        signgs=1.0,
        trig=trig,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    np.testing.assert_allclose(out[0], bsubs)
    np.testing.assert_allclose(out[1], bsubsu)
    np.testing.assert_allclose(out[2], bsubsv)

    shape3 = (3, shape[1], shape[2])
    zeros = np.zeros(shape3, dtype=float)
    out_zero_hs = wout_module._jxbforce_apply_bsubs_correction_lasym_true(
        bsubu=zeros,
        bsubv=zeros,
        bsubs=zeros,
        bsubsu=zeros + 1.0,
        bsubsv=zeros + 2.0,
        bsupu=zeros,
        bsupv=zeros,
        sqrtg=np.ones(shape3, dtype=float),
        pres=np.ones((shape3[0],), dtype=float),
        vp=np.ones((shape3[0],), dtype=float),
        hs=0.0,
        signgs=1.0,
        trig=trig,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    np.testing.assert_allclose(out_zero_hs[0], zeros)
    np.testing.assert_allclose(out_zero_hs[1], zeros + 1.0)
    np.testing.assert_allclose(out_zero_hs[2], zeros + 2.0)

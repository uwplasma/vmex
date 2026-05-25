from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.vmec_forces as vf
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.config import VMECConfig
from vmec_jax.fourier import eval_fourier
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.namelist import InData
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import (
    VmecRZForceKernels,
    _constraint_kernels_from_state,
    _vmec_force_profile_enabled,
    _vmec_force_profile_log,
    rz_residual_coeffs_from_kernels,
    vmec_forces_rz_from_wout,
    vmec_forces_rz_from_wout_reference_fields,
)


def _k_index(modes, m: int, n: int) -> int:
    for k, (mm, nn) in enumerate(zip(modes.m, modes.n)):
        if int(mm) == int(m) and int(nn) == int(n):
            return k
    raise KeyError((m, n))


def _zero_state(static) -> VMECState:
    ns = int(static.cfg.ns)
    K = int(static.modes.K)
    layout = StateLayout(ns=ns, K=K, lasym=bool(static.cfg.lasym))
    zeros = np.zeros((ns, K), dtype=float)
    return VMECState(
        layout=layout,
        Rcos=zeros,
        Rsin=zeros,
        Zcos=zeros,
        Zsin=zeros,
        Lcos=zeros,
        Lsin=zeros,
    )


def _full_mesh_shape(static) -> tuple[int, int, int]:
    return (int(static.cfg.ns), int(static.trig_vmec.ntheta3), int(static.cfg.nzeta))


def _synthetic_bcovar(static):
    shape = _full_mesh_shape(static)

    def arr(value: float):
        return np.full(shape, value, dtype=float)

    bc = SimpleNamespace(
        lu_e=arr(0.2),
        lv_e=arr(0.3),
        gij_b_uu=arr(0.4),
        gij_b_uv=arr(0.05),
        gij_b_vv=arr(0.6),
        jac=SimpleNamespace(
            ru12=arr(1.1),
            zu12=arr(1.2),
            rs=arr(0.7),
            zs=arr(0.8),
            r12=arr(1.0),
            sqrtg=arr(2.0),
            tau=arr(3.0),
        ),
        bsq=arr(0.9),
        bsupu=arr(0.1),
        bsupv=arr(0.2),
        bsubu=arr(0.3),
        bsubv=arr(0.4),
    )
    parity = SimpleNamespace(
        pr1_even=arr(4.0),
        pr1_odd=arr(0.5),
        pz1_even=arr(0.0),
        pz1_odd=arr(0.0),
        pru_even=arr(1.0),
        pru_odd=arr(0.25),
        pzu_even=arr(2.0),
        pzu_odd=arr(0.5),
        prv_even=arr(0.0),
        prv_odd=arr(0.0),
        pzv_even=arr(0.0),
        pzv_odd=arr(0.0),
        lu_odd=arr(0.0),
        lv_odd=arr(0.0),
    )
    return bc, parity


def test_force_profile_log_respects_env(monkeypatch, capsys) -> None:
    monkeypatch.delenv("VMEC_JAX_PROFILE_FORCE", raising=False)
    assert _vmec_force_profile_enabled() is False
    _vmec_force_profile_log("hidden", extra=1)
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("VMEC_JAX_PROFILE_FORCE", "yes")
    assert _vmec_force_profile_enabled() is True
    _vmec_force_profile_log("visible", start=time.perf_counter(), extra=2)
    out = capsys.readouterr().out
    assert "[vmec_jax force]" in out
    assert "'stage': 'visible'" in out
    assert "'extra': 2" in out
    assert "elapsed_s" in out


def test_constraint_zero_branch_shape_mismatch_and_debug_dumps(monkeypatch, tmp_path) -> None:
    cfg = VMECConfig(
        mpol=2,
        ntor=1,
        ns=3,
        nfp=1,
        lasym=False,
        lconm1=False,
        lthreed=True,
        ntheta=6,
        nzeta=4,
    )
    static = build_static(cfg)
    state = _zero_state(static)
    shape = _full_mesh_shape(static)
    ones = np.ones(shape, dtype=float)
    bc = SimpleNamespace(
        jac=SimpleNamespace(
            r12=ones,
            sqrtg=2.0 * ones,
            tau=3.0 * ones,
            ru12=4.0 * ones,
            zu12=5.0 * ones,
        ),
        bsq=6.0 * ones,
        bsupu=7.0 * ones,
        bsupv=8.0 * ones,
        bsubu=9.0 * ones,
        bsubv=10.0 * ones,
    )
    wout = SimpleNamespace(ntor=cfg.ntor, mpol=cfg.mpol, signgs=-1, lasym=cfg.lasym)

    zero = _constraint_kernels_from_state(
        state=state,
        static=static,
        wout=wout,
        bc=bc,
        pru_0=ones,
        pru_1=0.1 * ones,
        pzu_0=2.0 * ones,
        pzu_1=0.2 * ones,
        constraint_tcon0=None,
        trig=static.trig_vmec,
    )
    np.testing.assert_allclose(np.asarray(zero.gcon), np.zeros(shape))
    np.testing.assert_allclose(np.asarray(zero.tcon), np.zeros((cfg.ns,)))

    with pytest.raises(ValueError, match="rcon0_override shape mismatch"):
        _constraint_kernels_from_state(
            state=state,
            static=static,
            wout=wout,
            bc=bc,
            pru_0=ones,
            pru_1=0.1 * ones,
            pzu_0=2.0 * ones,
            pzu_1=0.2 * ones,
            constraint_tcon0=0.0,
            tcon_override=np.zeros((cfg.ns,), dtype=float),
            rcon0_override=np.zeros((cfg.ns, shape[1]), dtype=float),
            zcon0_override=np.zeros(shape, dtype=float),
            trig=static.trig_vmec,
        )

    monkeypatch.setenv("VMEC_JAX_DUMP_CONSTRAINTS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_BCOVAR", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "2-3")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    dumped = _constraint_kernels_from_state(
        state=state,
        static=static,
        wout=wout,
        bc=bc,
        pru_0=ones,
        pru_1=0.1 * ones,
        pzu_0=2.0 * ones,
        pzu_1=0.2 * ones,
        constraint_tcon0=0.0,
        tcon_override=np.zeros((cfg.ns,), dtype=float),
        trig=static.trig_vmec,
        iter_idx=2,
    )
    assert np.asarray(dumped.gcon).shape == shape
    assert (tmp_path / "constraints_raw_iter2.npz").exists()
    assert (tmp_path / "bcovar_raw_iter2.npz").exists()
    with np.load(tmp_path / "constraints_raw_iter2.npz") as data:
        assert {"gcon", "ztemp", "tcon", "rcon", "zcon"}.issubset(data.files)
    with np.load(tmp_path / "bcovar_raw_iter2.npz") as data:
        assert {"sqrtg", "bsupu", "bsubv", "overg", "wint"}.issubset(data.files)


def test_freeb_edge_coupling_synthetic_pressure_scale_and_dump(monkeypatch, tmp_path) -> None:
    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lconm1=False,
        lthreed=False,
        ntheta=6,
        nzeta=1,
    )
    static = build_static(cfg)
    state = _zero_state(static)
    wout = SimpleNamespace(
        phips=np.ones((cfg.ns,), dtype=float),
        pres=np.asarray([0.1, 0.2, 0.75], dtype=float),
        nfp=1,
        mpol=cfg.mpol,
        ntor=cfg.ntor,
        signgs=-1,
        lasym=False,
    )

    def fake_bcovar(*args, **kwargs):
        return _synthetic_bcovar(static)

    monkeypatch.setattr(vf, "vmec_bcovar_half_mesh_from_wout", fake_bcovar)
    monkeypatch.setenv("VMEC_JAX_FREEB_RBSQ_SCALE", "0.5")
    monkeypatch.setenv("VMEC_JAX_DUMP_FREEB_COUPLING", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))

    base = vmec_forces_rz_from_wout(state=state, static=static, wout=wout, constraint_tcon0=0.0)
    vac_edge = np.full((int(static.trig_vmec.ntheta3), cfg.nzeta), 0.25, dtype=float)
    vac_full = np.zeros(_full_mesh_shape(static), dtype=float)
    vac_full[-1] = vac_edge

    edge = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout,
        constraint_tcon0=0.0,
        freeb_bsqvac_half=vac_edge,
        freeb_pres_scale=2.0,
        iter_idx=7,
    )
    full = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout,
        constraint_tcon0=0.0,
        freeb_bsqvac_half=vac_full,
        freeb_pres_scale=2.0,
    )

    np.testing.assert_allclose(np.asarray(edge.armn_e), np.asarray(full.armn_e))
    np.testing.assert_allclose(np.asarray(edge.azmn_o), np.asarray(full.azmn_o))

    gcon_edge = 0.25 + 2.0 * 0.75
    rbsq_edge = gcon_edge * (4.0 + 0.5) * 2.0 * 0.5
    np.testing.assert_allclose(np.asarray(edge.armn_e - base.armn_e)[-1], (2.0 + 0.5) * rbsq_edge)
    np.testing.assert_allclose(np.asarray(edge.azmn_e - base.azmn_e)[-1], -(1.0 + 0.25) * rbsq_edge)
    dump_path = tmp_path / "freeb_coupling_iter7.npz"
    assert dump_path.exists()
    with np.load(dump_path) as dump:
        assert "plasma_bsq_edge" in dump.files
        assert "plasma_bsq_edge_extrap" in dump.files
        assert "dbsq_edge_proxy" in dump.files
        np.testing.assert_allclose(dump["dbsq_edge_proxy"], np.abs(dump["gcon_edge"] - dump["plasma_bsq_edge_extrap"]))

    with pytest.raises(ValueError, match="freeb_bsqvac_half shape mismatch"):
        vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout,
            constraint_tcon0=0.0,
            freeb_bsqvac_half=np.zeros((2, 2, 2, 2), dtype=float),
        )


def test_reference_fields_with_synthetic_wout_builds_half_mesh_and_lambda_kernels() -> None:
    cfg = VMECConfig(
        mpol=2,
        ntor=1,
        ns=3,
        nfp=1,
        lasym=False,
        lconm1=False,
        lthreed=True,
        ntheta=6,
        nzeta=4,
    )
    static = build_static(cfg)
    K = int(static.modes.K)
    Rcos = np.zeros((K,), dtype=float)
    Rsin = np.zeros((K,), dtype=float)
    Zcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)
    Rcos[_k_index(static.modes, 0, 0)] = 10.0
    Rcos[_k_index(static.modes, 1, 0)] = 1.5
    Zsin[_k_index(static.modes, 1, 0)] = 0.8
    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)
    state = initial_guess_from_boundary(
        static,
        boundary,
        InData(scalars={"RAXIS_CC": [10.0, 0.0], "ZAXIS_CS": [0.0, 0.0]}, indexed={}),
        infer_axis_if_missing=False,
    )

    k00 = _k_index(static.modes, 0, 0)

    def nyq_coeff(value: float):
        coeff = np.zeros((cfg.ns, K), dtype=float)
        coeff[:, k00] = value
        return coeff

    wout = SimpleNamespace(
        nfp=1,
        mpol=cfg.mpol,
        ntor=cfg.ntor,
        lasym=cfg.lasym,
        signgs=-1,
        xm_nyq=np.asarray(static.modes.m),
        xn_nyq=np.asarray(static.modes.n),
        gmnc=nyq_coeff(2.0),
        gmns=nyq_coeff(0.0),
        bsupumnc=nyq_coeff(0.2),
        bsupumns=nyq_coeff(0.0),
        bsupvmnc=nyq_coeff(0.3),
        bsupvmns=nyq_coeff(0.0),
        bsubumnc=nyq_coeff(0.4),
        bsubumns=nyq_coeff(0.0),
        bsubvmnc=nyq_coeff(0.5),
        bsubvmns=nyq_coeff(0.0),
        bmnc=nyq_coeff(1.2),
        bmns=nyq_coeff(0.0),
        phips=np.ones((cfg.ns,), dtype=float),
        pres=np.linspace(0.1, 0.2, cfg.ns),
    )

    kernels = vmec_forces_rz_from_wout_reference_fields(
        state=state,
        static=static,
        wout=wout,
        constraint_tcon0=0.0,
    )

    basis_shape = (cfg.ns, cfg.ntheta, cfg.nzeta)
    assert np.asarray(kernels.armn_e).shape == basis_shape
    assert np.all(np.isfinite(np.asarray(kernels.armn_e)))
    np.testing.assert_allclose(np.asarray(kernels.gcon), np.zeros(basis_shape))
    np.testing.assert_allclose(np.asarray(kernels.bc.bsubu_e)[0], 0.4)
    np.testing.assert_allclose(np.asarray(kernels.bc.bsubu_e)[-1], 0.2)
    np.testing.assert_allclose(np.asarray(kernels.bc.clmn_even)[0], 0.0)
    np.testing.assert_allclose(np.asarray(kernels.bc.blmn_odd)[0], 0.0)


def test_rz_residual_coeffs_from_synthetic_kernel_derivatives() -> None:
    cfg = VMECConfig(
        mpol=2,
        ntor=1,
        ns=2,
        nfp=1,
        lasym=False,
        lconm1=False,
        lthreed=True,
        ntheta=8,
        nzeta=4,
    )
    static = build_static(cfg)
    coeff_zero = np.zeros((cfg.ns, int(static.modes.K)), dtype=float)
    field_zero = eval_fourier(coeff_zero, coeff_zero, static.basis)

    brmn_odd_sin = coeff_zero.copy()
    brmn_odd_sin[:, _k_index(static.modes, 1, 0)] = 2.0
    crmn_odd_cos = coeff_zero.copy()
    crmn_odd_cos[:, _k_index(static.modes, 1, 1)] = 3.0

    kernels = VmecRZForceKernels(
        armn_e=field_zero,
        armn_o=field_zero,
        brmn_e=field_zero,
        brmn_o=eval_fourier(coeff_zero, brmn_odd_sin, static.basis),
        crmn_e=field_zero,
        crmn_o=eval_fourier(crmn_odd_cos, coeff_zero, static.basis),
        azmn_e=field_zero,
        azmn_o=field_zero,
        bzmn_e=field_zero,
        bzmn_o=field_zero,
        czmn_e=field_zero,
        czmn_o=field_zero,
        bc=SimpleNamespace(),
        arcon_e=field_zero,
        arcon_o=field_zero,
        azcon_e=field_zero,
        azcon_o=field_zero,
        gcon=field_zero,
        pr1_even=field_zero,
        pr1_odd=field_zero,
        pz1_even=field_zero,
        pz1_odd=field_zero,
        pru_even=field_zero,
        pru_odd=field_zero,
        pzu_even=field_zero,
        pzu_odd=field_zero,
        prv_even=field_zero,
        prv_odd=field_zero,
        pzv_even=field_zero,
        pzv_odd=field_zero,
    )

    coeffs = rz_residual_coeffs_from_kernels(kernels, static=static)
    expected_gcr_cos = coeff_zero.copy()
    expected_gcr_sin = coeff_zero.copy()
    expected_gcr_cos[:, _k_index(static.modes, 1, 0)] = -2.0
    expected_gcr_sin[:, _k_index(static.modes, 1, 1)] = 3.0

    np.testing.assert_allclose(np.asarray(coeffs.gcr_cos), expected_gcr_cos, atol=1e-12)
    np.testing.assert_allclose(np.asarray(coeffs.gcr_sin), expected_gcr_sin, atol=1e-12)
    np.testing.assert_allclose(np.asarray(coeffs.gcz_cos), coeff_zero, atol=1e-12)
    np.testing.assert_allclose(np.asarray(coeffs.gcz_sin), coeff_zero, atol=1e-12)

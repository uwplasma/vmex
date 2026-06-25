from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.preconditioner_1d as p1d
import vmec_jax.vmec_bcovar as vb
import vmec_jax.vmec_forces as vf
from vmec_jax.config import VMECConfig
from vmec_jax.static import build_static
from vmec_jax.vmec_forces import VmecRZForceKernels
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_trig_tables


def _cfg(*, ns=4, mpol=2, ntor=0, ntheta=8, nzeta=1, lasym=False, lthreed=False):
    return VMECConfig(
        ns=ns,
        mpol=mpol,
        ntor=ntor,
        nfp=1,
        lasym=lasym,
        lthreed=lthreed,
        lconm1=False,
        ntheta=ntheta,
        nzeta=nzeta,
    )


def _constant_kernel(shape, value=1.0):
    return np.full(shape, value, dtype=float)


def _kernel(shape, *, bc=None) -> VmecRZForceKernels:
    arrays = [_constant_kernel(shape, idx + 1.0) for idx in range(12)]
    z = np.zeros(shape, dtype=float)
    return VmecRZForceKernels(
        armn_e=arrays[0],
        armn_o=arrays[1],
        brmn_e=arrays[2],
        brmn_o=arrays[3],
        crmn_e=arrays[4],
        crmn_o=arrays[5],
        azmn_e=arrays[6],
        azmn_o=arrays[7],
        bzmn_e=arrays[8],
        bzmn_o=arrays[9],
        czmn_e=arrays[10],
        czmn_o=arrays[11],
        bc=SimpleNamespace() if bc is None else bc,
        arcon_e=z,
        arcon_o=z,
        azcon_e=z,
        azcon_o=z,
        gcon=z,
        pr1_even=z,
        pr1_odd=z,
        pz1_even=z,
        pz1_odd=z,
        pru_even=z,
        pru_odd=z,
        pzu_even=z,
        pzu_odd=z,
        prv_even=z,
        prv_odd=z,
        pzv_even=z,
        pzv_odd=z,
    )


def _circular_axisymmetric_case(*, ns=3):
    cfg = _cfg(ns=ns, mpol=2, ntor=0, ntheta=8, nzeta=1)
    static = build_static(cfg)
    modes = static.modes
    idx_m0 = int(np.flatnonzero((modes.m == 0) & (modes.n == 0))[0])
    idx_m1 = int(np.flatnonzero((modes.m == 1) & (modes.n == 0))[0])
    zeros = np.zeros((ns, int(modes.K)), dtype=float)
    rcos = zeros.copy()
    zsin = zeros.copy()
    rcos[:, idx_m0] = 2.0
    rcos[:, idx_m1] = np.sqrt(np.asarray(static.s)) / np.sqrt(2.0)
    zsin[:, idx_m1] = np.sqrt(np.asarray(static.s)) / np.sqrt(2.0)
    state = SimpleNamespace(Rcos=rcos, Rsin=zeros, Zcos=zeros, Zsin=zsin, Lcos=zeros, Lsin=zeros)
    wout = SimpleNamespace(
        phipf_internal=np.ones(ns),
        chipf_internal=np.zeros(ns),
        chips_eff=np.linspace(0.0, 0.2, ns),
        phipf=np.ones(ns),
        phips=np.r_[0.0, np.ones(ns - 1)],
        signgs=1,
        nfp=1,
        mpol=2,
        ntor=0,
        lasym=False,
        ncurr=0,
        lcurrent=False,
        icurv=np.zeros(ns),
        pres=np.zeros(ns),
    )
    return static, state, wout


def _preconditioner_bc(*, ns=3, ntheta=3, nzeta=2):
    shape = (ns, ntheta, nzeta)
    radial = np.arange(ns, dtype=float)[:, None, None]
    theta = np.arange(ntheta, dtype=float)[None, :, None]
    return SimpleNamespace(
        jac=SimpleNamespace(
            r12=1.0 + 0.1 * radial + np.zeros(shape),
            tau=0.2 + np.zeros(shape),
            zs=0.3 + np.zeros(shape),
            zu12=0.4 + np.zeros(shape),
            rs=0.5 + np.zeros(shape),
            ru12=0.6 + np.zeros(shape),
            sqrtg=1.0 + 0.05 * radial + 0.01 * theta + np.zeros(shape),
        ),
        bsq=1.0 + 0.1 * radial + np.zeros(shape),
        bsupv=0.2 + np.zeros(shape),
    )


def _preconditioner_k(*, ns=3, ntheta=3, nzeta=2):
    shape = (ns, ntheta, nzeta)
    return SimpleNamespace(
        pzu_even=np.full(shape, 0.4),
        pzu_odd=np.full(shape, 0.5),
        pz1_odd=np.full(shape, 0.6),
        pru_even=np.full(shape, 0.7),
        pru_odd=np.full(shape, 0.8),
        pr1_odd=np.full(shape, 0.9),
    )


def test_production_force_path_uses_compact_bcovar_payload():
    static, state, wout = _circular_axisymmetric_case(ns=4)

    kernels = vf.vmec_forces_rz_from_wout(state=state, static=static, wout=wout)

    assert isinstance(kernels.bc, vb.VmecForceBcovarPayload)
    for attr in (
        "jac",
        "guu",
        "bsupu",
        "bsupv",
        "bsubu",
        "bsubv",
        "clmn_even",
        "blmn_even",
        "bsq",
        "gij_b_uu",
        "gij_b_uv",
        "gij_b_vv",
        "lu_e",
        "lv_e",
        "lamscale",
    ):
        assert hasattr(kernels.bc, attr)
    assert not hasattr(kernels.bc, "bsubu_parity_even")


def test_force_scope_fallbacks_and_kernel_pytree_roundtrip(monkeypatch):
    class RaisingContext:
        def __enter__(self):
            raise RuntimeError("scope unavailable")

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_jax = SimpleNamespace(
        named_scope=lambda name: RaisingContext(),
        profiler=SimpleNamespace(TraceAnnotation=lambda name: RaisingContext()),
    )
    monkeypatch.setattr(vf, "has_jax", lambda: True)
    monkeypatch.setattr(vf, "jax", fake_jax)

    with vf._named_scope("covered"):
        scoped = True
    with vf._trace("covered"):
        traced = True
    assert scoped is True
    assert traced is True

    k = _kernel((2, 2, 1))
    children, aux = k.tree_flatten()
    rebuilt = VmecRZForceKernels.tree_unflatten(aux, children)
    np.testing.assert_allclose(np.asarray(rebuilt.armn_e), np.asarray(k.armn_e))
    assert rebuilt.bc is k.bc


def test_lasym_internal_residual_uses_sym_and_asym_transforms(monkeypatch):
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=1, nmax=1, lasym=True)
    shape = (2, int(trig.ntheta3), 3)
    coeff_shape = (2, 2, 2)
    sym_out = TomnspsRZL(
        frcc=np.full(coeff_shape, 1.0),
        frss=np.full(coeff_shape, 2.0),
        fzsc=np.full(coeff_shape, 3.0),
        fzcs=np.full(coeff_shape, 4.0),
        flsc=np.full(coeff_shape, 5.0),
        flcs=np.full(coeff_shape, 6.0),
    )
    asym_out = TomnspsRZL(
        frcc=np.zeros(coeff_shape),
        frss=None,
        fzsc=np.zeros(coeff_shape),
        fzcs=None,
        flsc=np.zeros(coeff_shape),
        flcs=None,
        frsc=np.full(coeff_shape, 7.0),
        frcs=np.full(coeff_shape, 8.0),
        fzcc=np.full(coeff_shape, 9.0),
        fzss=np.full(coeff_shape, 10.0),
        flcc=np.full(coeff_shape, 11.0),
        flss=np.full(coeff_shape, 12.0),
    )
    calls: list[tuple[str, bool]] = []

    def fake_tomnsps_rzl(**kwargs):
        calls.append(("sym", bool(kwargs["lasym"])))
        assert kwargs["armn_even"].shape == shape
        return sym_out

    def fake_tomnspa_rzl(**kwargs):
        calls.append(("asym", bool(kwargs["lasym"])))
        assert kwargs["armn_even"].shape == shape
        return asym_out

    monkeypatch.setattr(vf, "tomnsps_rzl", fake_tomnsps_rzl)
    monkeypatch.setattr(vf, "tomnspa_rzl", fake_tomnspa_rzl)

    out = vf.vmec_residual_internal_from_kernels(
        _kernel(shape),
        cfg_ntheta=8,
        cfg_nzeta=3,
        wout=SimpleNamespace(nfp=1, mpol=2, ntor=1, lasym=True),
        trig=trig,
        include_edge=True,
    )

    assert calls == [("sym", True), ("asym", True)]
    np.testing.assert_allclose(np.asarray(out.frcc), 1.0)
    np.testing.assert_allclose(np.asarray(out.flss), 12.0)


def test_lasym_symforce_split_matches_vmec_reflection_rules(monkeypatch):
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=1, nmax=1, lasym=True)
    ns = 1
    shape = (ns, int(trig.ntheta3), 3)
    coeff_shape = (ns, 2, 2)
    zeros_out = TomnspsRZL(
        frcc=np.zeros(coeff_shape),
        frss=np.zeros(coeff_shape),
        fzsc=np.zeros(coeff_shape),
        fzcs=np.zeros(coeff_shape),
        flsc=np.zeros(coeff_shape),
        flcs=np.zeros(coeff_shape),
    )
    asym_out = TomnspsRZL(
        frcc=np.zeros(coeff_shape),
        frss=None,
        fzsc=np.zeros(coeff_shape),
        fzcs=None,
        flsc=np.zeros(coeff_shape),
        flcs=None,
        frsc=np.zeros(coeff_shape),
        frcs=np.zeros(coeff_shape),
        fzcc=np.zeros(coeff_shape),
        fzss=np.zeros(coeff_shape),
        flcc=np.zeros(coeff_shape),
        flss=np.zeros(coeff_shape),
    )

    armn = np.arange(np.prod(shape), dtype=float).reshape(shape)
    brmn = 100.0 + armn
    k = _kernel(shape)
    k = VmecRZForceKernels(
        armn_e=armn,
        armn_o=k.armn_o,
        brmn_e=brmn,
        brmn_o=k.brmn_o,
        crmn_e=k.crmn_e,
        crmn_o=k.crmn_o,
        azmn_e=k.azmn_e,
        azmn_o=k.azmn_o,
        bzmn_e=k.bzmn_e,
        bzmn_o=k.bzmn_o,
        czmn_e=k.czmn_e,
        czmn_o=k.czmn_o,
        bc=k.bc,
        arcon_e=k.arcon_e,
        arcon_o=k.arcon_o,
        azcon_e=k.azcon_e,
        azcon_o=k.azcon_o,
        gcon=k.gcon,
        pr1_even=k.pr1_even,
        pr1_odd=k.pr1_odd,
        pz1_even=k.pz1_even,
        pz1_odd=k.pz1_odd,
        pru_even=k.pru_even,
        pru_odd=k.pru_odd,
        pzu_even=k.pzu_even,
        pzu_odd=k.pzu_odd,
        prv_even=k.prv_even,
        prv_odd=k.prv_odd,
        pzv_even=k.pzv_even,
        pzv_odd=k.pzv_odd,
    )
    captured: dict[str, dict[str, np.ndarray]] = {}

    def fake_tomnsps_rzl(**kwargs):
        captured["sym"] = {name: np.asarray(kwargs[name]) for name in ("armn_even", "brmn_even")}
        return zeros_out

    def fake_tomnspa_rzl(**kwargs):
        captured["asym"] = {name: np.asarray(kwargs[name]) for name in ("armn_even", "brmn_even")}
        return asym_out

    monkeypatch.setattr(vf, "tomnsps_rzl", fake_tomnsps_rzl)
    monkeypatch.setattr(vf, "tomnspa_rzl", fake_tomnspa_rzl)

    vf.vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=8,
        cfg_nzeta=3,
        wout=SimpleNamespace(nfp=1, mpol=2, ntor=1, lasym=True),
        trig=trig,
    )

    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    i0 = np.arange(nt2)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (shape[2] - np.arange(shape[2])) % shape[2]
    armn_ref = armn[:, ir0, :][:, :, kk]
    brmn_ref = brmn[:, ir0, :][:, :, kk]

    expected_armn_sym = armn.copy()
    expected_armn_sym[:, :nt2, :] = 0.5 * (armn[:, :nt2, :] + armn_ref)
    expected_armn_asym = np.zeros_like(armn)
    expected_armn_asym[:, :nt2, :] = 0.5 * (armn[:, :nt2, :] - armn_ref)
    expected_brmn_sym = brmn.copy()
    expected_brmn_sym[:, :nt2, :] = 0.5 * (brmn[:, :nt2, :] - brmn_ref)
    expected_brmn_asym = np.zeros_like(brmn)
    expected_brmn_asym[:, :nt2, :] = 0.5 * (brmn[:, :nt2, :] + brmn_ref)

    np.testing.assert_allclose(captured["sym"]["armn_even"], expected_armn_sym)
    np.testing.assert_allclose(captured["asym"]["armn_even"], expected_armn_asym)
    np.testing.assert_allclose(captured["sym"]["brmn_even"], expected_brmn_sym)
    np.testing.assert_allclose(captured["asym"]["brmn_even"], expected_brmn_asym)


def test_constraint_zcon_override_shape_mismatch():
    cfg = _cfg(ns=3, mpol=2, ntor=1, ntheta=6, nzeta=3, lthreed=True)
    static = build_static(cfg)
    shape = (cfg.ns, int(static.trig_vmec.ntheta3), cfg.nzeta)
    zeros = np.zeros((cfg.ns, int(static.modes.K)), dtype=float)
    state = SimpleNamespace(Rcos=zeros, Rsin=zeros, Zcos=zeros, Zsin=zeros)
    ones = np.ones(shape, dtype=float)
    bc = SimpleNamespace(
        jac=SimpleNamespace(r12=ones, sqrtg=ones, ru12=ones, zu12=ones),
        bsq=ones,
        bsupu=ones,
        bsupv=ones,
        bsubu=ones,
        bsubv=ones,
    )

    with pytest.raises(ValueError, match="zcon0_override shape mismatch"):
        vf._constraint_kernels_from_state(
            state=state,
            static=static,
            wout=SimpleNamespace(ntor=cfg.ntor, mpol=cfg.mpol, nfp=1, signgs=1, lasym=False),
            bc=bc,
            pru_0=ones,
            pru_1=zeros[:, :1, None] + ones,
            pzu_0=ones,
            pzu_1=ones,
            constraint_tcon0=0.0,
            tcon_override=np.zeros(cfg.ns),
            rcon0_override=np.zeros(shape),
            zcon0_override=np.zeros((cfg.ns, shape[1])),
            trig=static.trig_vmec,
        )


def test_bcovar_cached_fluxes_and_freeb_shape_errors():
    static, state, wout = _circular_axisymmetric_case(ns=3)
    bc, aux = vb.vmec_bcovar_half_mesh_from_wout(state=state, static=static, wout=wout, return_parity_aux=True)
    assert np.asarray(bc.bsupu).shape == np.asarray(aux.pr1_even).shape
    np.testing.assert_allclose(np.asarray(bc.bsupu[0]), 0.0)

    with pytest.raises(ValueError, match="freeb_bsqvac_edge must have shape"):
        vb.vmec_bcovar_half_mesh_from_wout(
            state=state,
            static=static,
            wout=wout,
            freeb_bsqvac_edge=np.zeros((1, 1, 1)),
        )

    with pytest.raises(ValueError, match="freeb_bsqvac_edge shape mismatch"):
        vb.vmec_bcovar_half_mesh_from_wout(
            state=state,
            static=static,
            wout=wout,
            freeb_bsqvac_edge=np.zeros((2, 2)),
        )


def test_preconditioner_validation_and_free_boundary_edge_conditioning():
    xs = np.zeros((1, 2, 1))
    ok_full = np.zeros((2, 2, 1))
    with pytest.raises(ValueError, match="xu_e must have ns_half\\+1 entries"):
        p1d._compute_preconditioning_matrix(
            xs=xs,
            xu12=xs,
            xu_e=ok_full[:1],
            xu_o=ok_full,
            x1_o=ok_full,
            r12=xs,
            total_pressure=xs,
            tau=xs,
            bsupv=xs,
            sqrtg=np.ones_like(xs),
            w_int=np.ones(2),
            sqrt_sh=np.ones(1),
            sm=np.ones(1),
            sp=np.ones(1),
            delta_s=1.0,
        )

    cfg = SimpleNamespace(mpol=3, ntor=1, ntheta=4, nzeta=2, nfp=1, lasym=False, lthreed=False)
    s = np.linspace(0.0, 1.0, 3)
    mats_edge, _, jmax_edge = p1d.rz_preconditioner_matrices(
        bc=_preconditioner_bc(ns=3),
        k=_preconditioner_k(ns=3),
        trig=None,
        s=s,
        cfg=cfg,
        jmax_override=3,
    )
    mats_inner, _, jmax_inner = p1d.rz_preconditioner_matrices(
        bc=_preconditioner_bc(ns=3),
        k=_preconditioner_k(ns=3),
        trig=None,
        s=s,
        cfg=cfg,
        jmax_override=2,
    )

    assert jmax_edge == 3
    assert jmax_inner == 2
    assert np.any(mats_edge["dr"][-1, 2:, :] != 0.0)
    assert not np.allclose(mats_edge["dz"][-1, 0, 0], mats_inner["dz"][-1, 0, 0])

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax import finite_beta
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.vmec_residue import (
    vmec_force_norms_from_bcovar,
    vmec_force_norms_from_bcovar_dynamic,
    vmec_fsq_sums_from_tomnsps,
    vmec_rz_decompose_signed,
    vmec_rz_norm_from_state,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_trig_tables
from vmec_jax.modes import vmec_mode_table


def _state_for_modes(ns: int, modes, *, lasym: bool) -> VMECState:
    layout = StateLayout(ns=ns, K=modes.K, lasym=lasym)
    base = np.arange(ns * modes.K, dtype=float).reshape(ns, modes.K) / 10.0
    return VMECState(
        layout=layout,
        Rcos=1.0 + base,
        Rsin=0.3 + base if lasym else np.zeros_like(base),
        Zcos=0.4 + base if lasym else np.zeros_like(base),
        Zsin=0.2 + base,
        Lcos=np.zeros_like(base),
        Lsin=np.zeros_like(base),
    )


def test_residue_rz_norm_and_decompose_lasym_m1_scaled_branches() -> None:
    modes = vmec_mode_table(mpol=3, ntor=1)
    state = _state_for_modes(ns=3, modes=modes, lasym=True)
    static = SimpleNamespace(
        modes=modes,
        s=np.asarray([0.0, 0.5, 1.0]),
        cfg=SimpleNamespace(lthreed=True, lconm1=True, lasym=True),
    )

    rcc, rss, zsc, zcs = vmec_rz_decompose_signed(
        state,
        static,
        apply_scalxc=True,
        apply_basis_norm=True,
    )
    rz_norm = vmec_rz_norm_from_state(
        state=state,
        static=static,
        apply_scalxc=True,
        apply_basis_norm=True,
        ns_min=1,
        ns_max=3,
    )

    assert rcc.shape[0] == 3
    assert rss.shape == rcc.shape
    assert zsc.shape == rcc.shape
    assert zcs.shape == rcc.shape
    assert float(np.asarray(rz_norm)) > 0.0


def test_force_norms_from_bcovar_static_and_dynamic_edge_branches() -> None:
    trig = vmec_trig_tables(ntheta=6, nzeta=2, nfp=1, mmax=2, nmax=1, lasym=False, cache=False)
    shape = (3, int(trig.ntheta3), 2)
    ones = np.ones(shape, dtype=float)
    bc = SimpleNamespace(
        jac=SimpleNamespace(sqrtg=2.0 * ones, r12=0.5 * ones),
        guu=1.5 * ones,
        bsupu=0.4 * ones,
        bsupv=0.2 * ones,
        bsubu=0.7 * ones,
        bsubv=0.3 * ones,
        bsq=1.0 * ones,
        lamscale=2.0,
    )
    wout = SimpleNamespace(vp=np.asarray([0.0, 1.0, 1.5]), wb=2.0, wp=0.5)

    short = vmec_force_norms_from_bcovar(bc=bc, trig=trig, wout=wout, s=np.asarray([0.0]))
    assert np.isnan(short.fnorm)

    static_norms = vmec_force_norms_from_bcovar(
        bc=bc,
        trig=trig,
        wout=wout,
        s=np.asarray([0.0, 0.5, 1.0]),
    )
    dynamic_norms = vmec_force_norms_from_bcovar_dynamic(
        bc=bc,
        trig=trig,
        s=np.asarray([0.0, 0.5, 1.0]),
        signgs=1,
    )
    assert np.isfinite(static_norms.fnorm)
    assert np.isfinite(static_norms.fnormL)
    assert np.isfinite(np.asarray(dynamic_norms.fnorm))
    assert np.asarray(dynamic_norms.vp).shape == (3,)

    bad_bc = SimpleNamespace(**{**bc.__dict__, "jac": SimpleNamespace(sqrtg=np.ones((3, 2)), r12=ones)})
    with pytest.raises(ValueError, match="sqrtg must be"):
        vmec_force_norms_from_bcovar_dynamic(bc=bad_bc, trig=trig, s=np.asarray([0.0, 0.5, 1.0]), signgs=1)


def test_fsq_sums_records_all_optional_lasym_blocks_with_scaling() -> None:
    shape = (1, 2, 1)
    base = np.ones(shape, dtype=float)
    frzl = TomnspsRZL(
        frcc=base,
        frss=2.0 * base,
        fzsc=3.0 * base,
        fzcs=4.0 * base,
        flsc=5.0 * base,
        flcs=6.0 * base,
        frsc=7.0 * base,
        frcs=8.0 * base,
        fzcc=9.0 * base,
        fzss=10.0 * base,
        flcc=11.0 * base,
        flss=12.0 * base,
    )

    sums = vmec_fsq_sums_from_tomnsps(
        frzl=frzl,
        lconm1=True,
        apply_m1_constraints=True,
        apply_scalxc=True,
        include_edge=False,
        s=np.asarray([0.0]),
    )

    assert set(sums.gcr2_blocks) == {"frcc", "frss", "frsc", "frcs"}
    assert set(sums.gcz2_blocks) == {"fzsc", "fzcs", "fzcc", "fzss"}
    assert set(sums.gcl2_blocks) == {"flsc", "flcs", "flcc", "flss"}
    assert sums.gcr2 > 0.0
    assert sums.gcz2 > 0.0
    assert sums.gcl2 > 0.0


def test_redl_geometry_and_mismatch_state_paths_are_composable(monkeypatch) -> None:
    trig = vmec_trig_tables(ntheta=6, nzeta=2, nfp=2, mmax=1, nmax=1, lasym=False, cache=False)
    s = np.asarray([0.0, 0.5, 1.0])
    static = SimpleNamespace(
        s=s,
        trig_vmec=None,
        cfg=SimpleNamespace(ntheta=6, nzeta=2, nfp=2, mpol=2, ntor=1, lasym=False),
    )
    state = SimpleNamespace(Rcos=np.ones((3, 1)))
    indata = SimpleNamespace(get_float=lambda key, default=0.0: 2.0 if key == "PHIEDGE" else default)
    shape = (3, int(trig.ntheta3), 2)
    ones = np.ones(shape, dtype=float)
    wout_like = SimpleNamespace(iotas=np.asarray([0.0, 0.45, 0.5]))
    pres = np.asarray([0.0, 0.5, 1.0])
    bc = SimpleNamespace(
        bsubu=0.2 * ones,
        bsubv=0.4 * ones,
        bsq=4.0 * ones,
        jac=SimpleNamespace(sqrtg=ones),
    )

    monkeypatch.setattr(finite_beta, "_wout_like_for_state", lambda **_kwargs: (wout_like, pres))
    monkeypatch.setattr(finite_beta, "vmec_bcovar_half_mesh_from_wout", lambda **_kwargs: bc)

    captured: dict[str, np.ndarray] = {}

    def fake_trapped_fraction_from_modb_sqrtg(**kwargs):
        captured["modB"] = np.asarray(kwargs["modB"], dtype=float)
        captured["sqrtg"] = np.asarray(kwargs["sqrtg"], dtype=float)
        captured["n_lambda"] = np.asarray(kwargs["n_lambda"], dtype=int)
        return {
            "fsa_1overB": np.asarray([1.0, 1.1, 1.2]),
            "epsilon": np.asarray([0.1, 0.2, 0.3]),
            "f_t": np.asarray([0.4, 0.5, 0.6]),
        }

    monkeypatch.setattr(
        finite_beta,
        "trapped_fraction_from_modb_sqrtg",
        fake_trapped_fraction_from_modb_sqrtg,
    )

    geom = finite_beta.redl_bootstrap_geometry_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        surfaces=(0.4, 0.9),
        n_lambda=4,
    )
    np.testing.assert_array_equal(np.asarray(geom["indices"]), [1, 2])
    np.testing.assert_allclose(np.asarray(geom["iota"]), [0.45, 0.5])
    assert geom["nfp"] == 2
    np.testing.assert_allclose(captured["modB"], np.sqrt(2.0 * (bc.bsq - pres[:, None, None])))
    np.testing.assert_allclose(captured["sqrtg"], ones)
    assert int(captured["n_lambda"]) == 4

    monkeypatch.setattr(finite_beta, "redl_bootstrap_geometry_from_state", lambda **_kwargs: geom)
    monkeypatch.setattr(
        finite_beta,
        "redl_bootstrap_jdotb",
        lambda **_kwargs: (np.asarray([2.0, 3.0]), {"source": "fake-redl"}),
    )
    monkeypatch.setattr(
        finite_beta,
        "mercier_terms_from_state",
        lambda **_kwargs: {"jdotb": np.asarray([0.0, 1.0, 5.0])},
    )
    monkeypatch.setattr(
        finite_beta,
        "redl_bootstrap_mismatch_from_profiles",
        lambda *, jdotB_vmec, jdotB_redl: np.asarray(jdotB_vmec) - np.asarray(jdotB_redl),
    )

    mismatch = finite_beta.redl_bootstrap_mismatch_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=1,
        helicity_n=1,
        ne_coeffs=(1.0,),
        Te_coeffs=(1.0,),
    )
    np.testing.assert_allclose(np.asarray(mismatch["residuals1d"]), [-1.0, 2.0])
    np.testing.assert_allclose(np.asarray(mismatch["total"]), 5.0)
    assert mismatch["redl"] == {"source": "fake-redl"}


def test_finite_beta_edge_branches_for_small_meshes_and_invalid_shapes() -> None:
    np.testing.assert_array_equal(np.asarray(finite_beta._surface_indices([0.0, 1.0], None)), [])
    np.testing.assert_array_equal(np.asarray(finite_beta._surface_indices([0.0, 0.5, 1.0], None)), [1])

    tiny_shape = (2, 2, 1)
    zeros = np.zeros(tiny_shape, dtype=float)
    gpp = finite_beta.mercier_gpp_from_realspace_geometry(
        s=np.asarray([0.0, 1.0]),
        phips=np.ones(2),
        sqrtg=np.ones(tiny_shape),
        R_even=zeros,
        R_odd=zeros,
        Ru_even=zeros,
        Ru_odd=zeros,
        Zu_even=zeros,
        Zu_odd=zeros,
        Rv_even=zeros,
        Rv_odd=zeros,
        Zv_even=zeros,
        Zv_odd=zeros,
        signgs=1,
    )
    np.testing.assert_allclose(np.asarray(gpp), zeros)

    trig = vmec_trig_tables(ntheta=6, nzeta=2, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    with pytest.raises(ValueError, match="Expected f"):
        finite_beta._mercier_symoutput_split_jax(f=np.ones((2, 3)), trig=trig)
    with pytest.raises(ValueError, match="smaller"):
        finite_beta._mercier_symoutput_split_jax(f=np.ones((2, 1, 2)), trig=trig)
    with pytest.raises(ValueError, match="shape mismatch"):
        finite_beta._mercier_extend_parity_to_full_jax(
            par0=np.ones((2, int(trig.ntheta2), 2)),
            par1=np.ones((2, int(trig.ntheta2) - 1, 2)),
            trig=trig,
        )
    full_no_tail = finite_beta._mercier_extend_parity_to_full_jax(
        par0=np.ones((2, int(trig.ntheta2), 2)),
        par1=np.zeros((2, int(trig.ntheta2), 2)),
        trig=SimpleNamespace(ntheta1=int(trig.ntheta1), ntheta3=int(trig.ntheta2)),
    )
    assert np.asarray(full_no_tail).shape == (2, int(trig.ntheta2), 2)
    no_reflection_tail = finite_beta._mercier_extend_parity_to_full_jax(
        par0=np.ones((2, int(trig.ntheta2), 2)),
        par1=np.zeros((2, int(trig.ntheta2), 2)),
        trig=SimpleNamespace(ntheta1=int(trig.ntheta2) - 1, ntheta3=int(trig.ntheta2) + 1),
    )
    assert np.asarray(no_reflection_tail).shape == (2, int(trig.ntheta2) + 1, 2)

    with pytest.raises(ValueError, match="bsubs grid"):
        finite_beta.mercier_bsubs_derivatives_lasym_false(
            bsubs=np.ones((2, int(trig.ntheta2) - 1, 2)),
            trig=trig,
            mmax_force=1,
            nmax_force=1,
        )
    neg_false = finite_beta.mercier_bsubs_derivatives_lasym_false(
        bsubs=np.ones((2, int(trig.ntheta2), 2)),
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
    )
    np.testing.assert_allclose(np.asarray(neg_false["bsubsu"]), 0.0)
    nyquist_false = finite_beta.mercier_bsubs_derivatives_lasym_false(
        bsubs=np.ones((2, int(trig.ntheta2), 2)),
        trig=trig,
        mmax_force=2,
        nmax_force=1,
    )
    assert np.asarray(nyquist_false["bsubsu"]).shape == (2, int(trig.ntheta2), 2)

    with pytest.raises(ValueError, match="LASYM bsubs"):
        finite_beta.mercier_bsubs_derivatives_lasym_true(
            bsubs=np.ones((2, int(trig.ntheta3) - 1, 2)),
            trig=trig,
            mmax_force=1,
            nmax_force=1,
        )
    neg_true = finite_beta.mercier_bsubs_derivatives_lasym_true(
        bsubs=np.ones((2, int(trig.ntheta3), 2)),
        trig=trig,
        mmax_force=1,
        nmax_force=-1,
    )
    np.testing.assert_allclose(np.asarray(neg_true["bsubsv"]), 0.0)
    nyquist_true = finite_beta.mercier_bsubs_derivatives_lasym_true(
        bsubs=np.ones((2, int(trig.ntheta3), 2)),
        trig=trig,
        mmax_force=2,
        nmax_force=1,
    )
    assert np.asarray(nyquist_true["bsubsv"]).shape == (2, int(trig.ntheta3), 2)

    one_surface = (1, 2, 1)
    one = np.ones(one_surface)
    zeta = finite_beta.mercier_zeta_half_mesh_from_realspace_geometry(
        s=np.asarray([0.0]),
        Rv_even=one,
        Rv_odd=one,
        Zv_even=one,
        Zv_odd=one,
    )
    bss = finite_beta.mercier_bss_half_mesh_geometry_from_realspace(
        s=np.asarray([0.0]),
        rs=one,
        zs=one,
        R_odd=one,
        Z_odd=one,
        Rv_even=one,
        Rv_odd=one,
        Zv_even=one,
        Zv_odd=one,
    )
    bdotk = finite_beta.mercier_bdotk_from_covariant_derivatives(
        bsubu=one,
        bsubv=one,
        bsubsu=one,
        bsubsv=one,
        s=np.asarray([0.0]),
    )
    np.testing.assert_allclose(np.asarray(zeta["rv12"]), 0.0)
    np.testing.assert_allclose(np.asarray(bss["rs12"]), 0.0)
    np.testing.assert_allclose(np.asarray(bdotk["bdotk"]), 0.0)

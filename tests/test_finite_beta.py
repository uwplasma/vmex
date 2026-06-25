from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vmec_jax as vj
from vmec_jax import finite_beta
from vmec_jax._compat import jnp
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.static import build_static
from vmec_jax.state import VMECState
from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.vmec_residue import vmec_force_norms_from_bcovar_dynamic
from vmec_jax.wout import _compute_mercier, _vmec_wint_from_trig_jax


pytestmark = pytest.mark.full


def test_finite_beta_scalars_are_finite_on_bundled_qi_input():
    path = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.nfp4_QI_finite_beta"
    cfg, indata = load_config(str(path))
    static = build_static(cfg)
    boundary = boundary_from_indata(indata, static.modes)
    state = initial_guess_from_boundary(static, boundary, indata)
    geom = eval_geom(state, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))

    scalars = vj.finite_beta_scalars_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )

    for key in ("aspect", "mean_iota", "volavgB", "betatotal", "wb", "wp", "volume"):
        assert np.isfinite(float(np.asarray(scalars[key])))
    assert float(np.asarray(scalars["betatotal"])) > 0.0


def test_mercier_terms_are_finite_on_bundled_qi_input():
    path = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.nfp4_QI_finite_beta"
    cfg, indata = load_config(str(path))
    static = build_static(cfg)
    boundary = boundary_from_indata(indata, static.modes)
    state = initial_guess_from_boundary(static, boundary, indata)
    geom = eval_geom(state, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))

    terms = vj.mercier_terms_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )

    for key in ("DMerc", "Dshear", "Dcurr", "Dwell", "Dgeod", "D_R", "H", "glasser_correction", "torcur", "vp"):
        arr = np.asarray(terms[key])
        assert arr.shape == np.asarray(static.s).shape
        assert np.all(np.isfinite(arr))
    assert np.asarray(terms["glasser_shear_valid"]).shape == np.asarray(static.s).shape


def test_mercier_terms_from_state_matches_wout_mercier_path_on_bundled_qi_input():
    path = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.nfp4_QI_finite_beta"
    cfg, indata = load_config(str(path))
    static = build_static(cfg)
    boundary = boundary_from_indata(indata, static.modes)
    state = initial_guess_from_boundary(static, boundary, indata)
    geom_eval = eval_geom(state, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom_eval.sqrtg), axis_index=1))

    actual = vj.mercier_terms_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    wout_like, pres = finite_beta._wout_like_for_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        use_wout_bsup=False,
        use_wout_bsub_for_lambda=False,
        use_wout_bmag_for_bsq=False,
        use_vmec_synthesis=True,
        trig=static.trig_vmec,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(
        bc=bc,
        trig=static.trig_vmec,
        s=static.s,
        signgs=signgs,
    )
    geom = {
        "R": np.asarray(geom_eval.R),
        "Z": np.asarray(geom_eval.Z),
        "Ru": np.asarray(geom_eval.Rt),
        "Zu": np.asarray(geom_eval.Zt),
        "Rv": np.asarray(geom_eval.Rp),
        "Zv": np.asarray(geom_eval.Zp),
    }
    expected = _compute_mercier(
        state=state,
        geom_modes=static.modes,
        s=np.asarray(static.s),
        lconm1=bool(static.cfg.lconm1),
        lthreed=bool(static.cfg.lthreed),
        lasym=bool(static.cfg.lasym),
        nfp=int(static.cfg.nfp),
        lbsubs=False,
        mmax_force=max(int(static.cfg.mpol) - 1, 0),
        nmax_force=int(static.cfg.ntor),
        pres=np.asarray(pres),
        vp=np.asarray(norms.vp),
        phips=np.asarray(wout_like.phips),
        iotas=np.asarray(wout_like.iotas),
        bsq=np.asarray(bc.bsq),
        sqrtg=np.asarray(bc.jac.sqrtg),
        bsubu=np.asarray(bc.bsubu),
        bsubv=np.asarray(bc.bsubv),
        bsupu=np.asarray(bc.bsupu),
        bsupv=np.asarray(bc.bsupv),
        trig=static.trig_vmec,
        geom=geom,
        jac_half=bc.jac,
        signgs=signgs,
    )

    for key, reference in zip(
        ("DMerc", "Dshear", "Dcurr", "Dwell", "Dgeod", "jdotb", "bdotb", "bdotgradv"),
        expected,
    ):
        np.testing.assert_allclose(np.asarray(actual[key]), reference, rtol=1e-11, atol=1e-10)

    wint = np.asarray(_vmec_wint_from_trig_jax(static.trig_vmec))
    expected_torcur = np.zeros_like(np.asarray(static.s, dtype=float))
    expected_torcur[1:] = float(signgs) * 2.0 * np.pi * np.sum(np.asarray(bc.bsubu)[1:] * wint[None, :, :], axis=(1, 2))
    expected_ip = np.asarray(
        finite_beta.mercier_terms_from_profile_integrals(
            s=np.asarray(static.s),
            phips=np.asarray(wout_like.phips),
            iotas=np.asarray(wout_like.iotas),
            vp=np.asarray(norms.vp),
            pres=np.asarray(pres),
            torcur=expected_torcur,
            tpp=np.asarray(actual["tpp"]),
            tbb=np.asarray(actual["tbb"]),
            tjb=np.asarray(actual["tjb"]),
            tjj=np.asarray(actual["tjj"]),
            signgs=signgs,
        )["ip"]
    )
    np.testing.assert_allclose(np.asarray(actual["torcur"]), expected_torcur, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(actual["ip"]), expected_ip, rtol=1e-12, atol=1e-12)


def test_mercier_terms_from_state_dmerc_and_dr_ad_match_fd_on_bundled_qi_input():
    """Physics-shaped AD/FD gate for finite-beta Mercier and Glasser profiles."""

    jax = pytest.importorskip("jax")

    path = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.nfp4_QI_finite_beta"
    cfg, indata = load_config(str(path))
    static = build_static(cfg)
    boundary = boundary_from_indata(indata, static.modes)
    state0 = initial_guess_from_boundary(static, boundary, indata)
    geom = eval_geom(state0, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))

    mode_m = np.asarray(static.modes.m)
    mode_n = np.asarray(static.modes.n)
    idx10 = int(np.flatnonzero((mode_m == 1) & (mode_n == 0))[0])
    radial_shape = jnp.asarray(static.s, dtype=jnp.float64)
    radial_shape = radial_shape / jnp.maximum(radial_shape[-1], jnp.asarray(1.0e-12, dtype=jnp.float64))

    Rcos0 = jnp.asarray(state0.Rcos, dtype=jnp.float64)
    Rsin0 = jnp.asarray(state0.Rsin, dtype=jnp.float64)
    Zcos0 = jnp.asarray(state0.Zcos, dtype=jnp.float64)
    Zsin0 = jnp.asarray(state0.Zsin, dtype=jnp.float64)
    Lcos0 = jnp.asarray(state0.Lcos, dtype=jnp.float64)
    Lsin0 = jnp.asarray(state0.Lsin, dtype=jnp.float64)

    def state_for_alpha(alpha):
        return VMECState(
            layout=state0.layout,
            Rcos=Rcos0.at[:, idx10].add(alpha * radial_shape),
            Rsin=Rsin0,
            Zcos=Zcos0,
            Zsin=Zsin0,
            Lcos=Lcos0,
            Lsin=Lsin0,
        )

    def objective(alpha, key: str):
        terms = vj.mercier_terms_from_state(
            state=state_for_alpha(alpha),
            static=static,
            indata=indata,
            signgs=signgs,
        )
        values = jnp.asarray(terms[key], dtype=jnp.float64)
        return jnp.sum(values[1:-1])

    alpha0 = jnp.asarray(0.0, dtype=jnp.float64)
    eps = jnp.asarray(1.0e-5, dtype=jnp.float64)
    for key in ("DMerc", "D_R"):
        grad_ad = jax.grad(lambda alpha: objective(alpha, key))(alpha0)
        grad_fd = (objective(alpha0 + eps, key) - objective(alpha0 - eps, key)) / (2.0 * eps)
        assert np.isfinite(float(np.asarray(grad_ad)))
        assert np.isfinite(float(np.asarray(grad_fd)))
        np.testing.assert_allclose(np.asarray(grad_ad), np.asarray(grad_fd), rtol=2.0e-7, atol=1.0e-6)

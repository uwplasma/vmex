from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax import finite_beta
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.finite_beta import FiniteBetaTargets
from vmec_jax.modes import ModeTable
from vmec_jax.namelist import InData
from vmec_jax.vmec_realspace import vmec_realspace_synthesis_multi
from vmec_jax.vmec_tomnsp import vmec_trig_tables


def test_s_half_from_static_matches_vmec_half_mesh_convention():
    static = SimpleNamespace(s=jnp.asarray([0.0, 0.25, 1.0]))
    np.testing.assert_allclose(np.asarray(finite_beta._s_half_from_static(static)), [0.0, 0.125, 0.625])

    single = SimpleNamespace(s=jnp.asarray([0.0]))
    np.testing.assert_allclose(np.asarray(finite_beta._s_half_from_static(single)), [0.0])


def test_wout_like_for_state_builds_profile_and_flux_fields(monkeypatch):
    modes = ModeTable(m=np.array([0, 1]), n=np.array([0, 0]))
    static = SimpleNamespace(
        s=jnp.asarray([0.0, 0.5, 1.0]),
        modes=modes,
        cfg=SimpleNamespace(nfp=2, mpol=2, ntor=0, lasym=False),
    )
    indata = InData(scalars={"NCURR": 1, "GAMMA": 0.0, "LRFP": False}, indexed={}, source_path=None)

    monkeypatch.setattr(
        finite_beta,
        "flux_profiles_from_indata",
        lambda *_args, **_kwargs: SimpleNamespace(
            phipf=jnp.asarray([4.0, 5.0, 6.0]),
            phips=jnp.asarray([9.0, 2.0, 3.0]),
            chipf=jnp.asarray([1.0, 2.0, 3.0]),
        ),
    )
    monkeypatch.setattr(
        finite_beta,
        "eval_profiles",
        lambda *_args, **_kwargs: {"pressure": jnp.asarray([7.0, 8.0, 9.0])},
    )
    monkeypatch.setattr(
        finite_beta,
        "equilibrium_iota_profiles_from_state",
        lambda **_kwargs: (
            jnp.asarray([0.0, 1.0, 3.0]),
            jnp.asarray([0.0, 0.1, 0.2]),
            jnp.asarray([0.0, 0.15, 0.25]),
        ),
    )
    monkeypatch.setattr(finite_beta, "_chipf_from_chips", lambda chips: jnp.asarray(chips) + 10.0)
    monkeypatch.setattr(
        "vmec_jax.boundary.boundary_from_indata",
        lambda *_args, **_kwargs: BoundaryCoeffs(
            R_cos=np.array([1.5, 0.0]),
            R_sin=np.zeros(2),
            Z_cos=np.zeros(2),
            Z_sin=np.zeros(2),
        ),
    )
    monkeypatch.setattr(
        finite_beta,
        "_mass_half_mesh_from_indata",
        lambda **_kwargs: jnp.asarray([0.0, 0.2, 0.3]),
    )
    monkeypatch.setattr(
        finite_beta,
        "_icurv_full_mesh_from_indata",
        lambda **_kwargs: jnp.asarray([0.0, 0.4, 0.5]),
    )

    wout_like, pres = finite_beta._wout_like_for_state(
        state=object(),
        static=static,
        indata=indata,
        signgs=-1,
    )

    np.testing.assert_allclose(np.asarray(wout_like.phips), [0.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(wout_like.phipf), [4.0, 5.0, 6.0])
    np.testing.assert_allclose(np.asarray(wout_like.chipf), [10.0, 11.0, 13.0])
    np.testing.assert_allclose(np.asarray(wout_like.mass), [0.0, 0.2, 0.3])
    np.testing.assert_allclose(np.asarray(wout_like.icurv), [0.0, 0.4, 0.5])
    np.testing.assert_allclose(np.asarray(pres), [0.0, 8.0, 9.0])
    assert wout_like.signgs == -1
    assert wout_like.nfp == 2
    assert wout_like.lcurrent


def test_finite_beta_scalars_from_state_uses_iota_and_energy_diagnostics(monkeypatch):
    static = SimpleNamespace(s=jnp.asarray([0.0, 0.5, 1.0]), trig_vmec=object())

    monkeypatch.setattr(
        finite_beta,
        "equilibrium_aspect_ratio_from_state",
        lambda **_kwargs: jnp.asarray(6.25),
    )
    monkeypatch.setattr(
        finite_beta,
        "equilibrium_iota_profiles_from_state",
        lambda **_kwargs: (
            jnp.asarray([0.0, 1.0, 3.0]),
            jnp.asarray([0.0, 0.2, -0.4]),
            jnp.asarray([0.0, 0.3, -0.5]),
        ),
    )
    monkeypatch.setattr(
        finite_beta,
        "_wout_like_for_state",
        lambda **_kwargs: (SimpleNamespace(), jnp.asarray([0.0, 1.0, 2.0])),
    )
    monkeypatch.setattr(
        finite_beta,
        "vmec_bcovar_half_mesh_from_wout",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        finite_beta,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **_kwargs: SimpleNamespace(
            wb=jnp.asarray(8.0),
            wp=jnp.asarray(2.0),
            volume=jnp.asarray(4.0),
        ),
    )

    scalars = finite_beta.finite_beta_scalars_from_state(
        state=object(),
        static=static,
        indata=object(),
        signgs=1,
    )

    assert float(scalars["aspect"]) == 6.25
    np.testing.assert_allclose(np.asarray(scalars["iotas"]), [0.0, 0.2, -0.4])
    np.testing.assert_allclose(np.asarray(scalars["iotaf"]), [0.0, 0.3, -0.5])
    assert float(scalars["mean_iota"]) == 0.4
    assert float(scalars["min_iota"]) == 0.3
    assert float(scalars["max_iota"]) == 0.5
    assert float(scalars["betatotal"]) == 0.25
    assert float(scalars["volavgB"]) == 2.0
    assert float(scalars["wb"]) == 8.0
    assert float(scalars["wp"]) == 2.0
    assert float(scalars["volume"]) == 4.0


def test_finite_beta_scalars_from_state_gradient_matches_finite_difference(monkeypatch):
    pytest.importorskip("jax")
    import jax

    static = SimpleNamespace(s=jnp.asarray([0.0, 0.25, 0.75, 1.0]), trig_vmec=object())

    def _state(scale):
        return SimpleNamespace(scale=jnp.asarray(scale, dtype=jnp.float64))

    def fake_aspect(*, state, **_kwargs):
        x = jnp.asarray(state.scale, dtype=jnp.float64)
        return 5.0 + 0.25 * x + 0.03 * x**2

    def fake_iota_profiles(*, state, **_kwargs):
        x = jnp.asarray(state.scale, dtype=jnp.float64)
        chips = jnp.asarray([0.0, 0.1 + 0.01 * x, 0.2 + 0.02 * x, 0.3 + 0.03 * x], dtype=jnp.float64)
        iotas = jnp.asarray([0.0, 0.30 + 0.04 * x, 0.50 + 0.05 * x, 0.70 + 0.06 * x], dtype=jnp.float64)
        iotaf = jnp.asarray([0.0, 0.40 + 0.02 * x, 0.60 + 0.03 * x, 0.80 + 0.04 * x], dtype=jnp.float64)
        return chips, iotas, iotaf

    def fake_bcovar(*, state, **_kwargs):
        return SimpleNamespace(scale=jnp.asarray(state.scale, dtype=jnp.float64))

    def fake_norms(*, bc, **_kwargs):
        x = jnp.asarray(bc.scale, dtype=jnp.float64)
        return SimpleNamespace(
            wb=2.0 + 0.4 * x**2,
            wp=0.3 + 0.05 * x + 0.02 * x**2,
            volume=1.5 + 0.1 * x,
            vp=jnp.asarray([0.0, 1.0 + 0.1 * x, 0.9 + 0.05 * x, 0.8 + 0.02 * x], dtype=jnp.float64),
        )

    monkeypatch.setattr(finite_beta, "equilibrium_aspect_ratio_from_state", fake_aspect)
    monkeypatch.setattr(finite_beta, "equilibrium_iota_profiles_from_state", fake_iota_profiles)
    monkeypatch.setattr(finite_beta, "_wout_like_for_state", lambda **_kwargs: (SimpleNamespace(), jnp.zeros(4)))
    monkeypatch.setattr(finite_beta, "vmec_bcovar_half_mesh_from_wout", fake_bcovar)
    monkeypatch.setattr(finite_beta, "vmec_force_norms_from_bcovar_dynamic", fake_norms)

    def objective(scale):
        scalars = finite_beta.finite_beta_scalars_from_state(
            state=_state(scale),
            static=static,
            indata=object(),
            signgs=1,
        )
        return (
            scalars["aspect"]
            + scalars["mean_iota"]
            + scalars["min_iota"]
            + scalars["max_iota"]
            + 0.7 * scalars["volavgB"]
            + 1.3 * scalars["betatotal"]
        )

    x0 = 1.2
    eps = 1.0e-6
    ad_grad = jax.grad(lambda x: objective(x))(jnp.asarray(x0, dtype=jnp.float64))
    fd_grad = (float(objective(x0 + eps)) - float(objective(x0 - eps))) / (2.0 * eps)

    np.testing.assert_allclose(np.asarray(ad_grad), fd_grad, rtol=2.0e-6, atol=2.0e-8)


def test_magnetic_well_from_vp_uses_vmec_endpoint_extrapolation():
    vp = jnp.asarray([0.0, 2.0, 1.5, 1.0], dtype=jnp.float64)

    well = finite_beta.magnetic_well_from_vp(vp)

    np.testing.assert_allclose(np.asarray(well), 2.0 / 3.0)
    np.testing.assert_allclose(np.asarray(finite_beta.magnetic_well_from_vp([1.0, 2.0])), 0.0)
    np.testing.assert_allclose(np.asarray(finite_beta.magnetic_well_from_vp([0.0, 1.0, 3.0])), 0.0)


def test_magnetic_well_from_vp_is_differentiable():
    import jax

    def objective(scale):
        vp = jnp.asarray([0.0, 1.0 + scale, 0.9, 0.8], dtype=jnp.float64)
        return finite_beta.magnetic_well_from_vp(vp)

    value, grad = jax.value_and_grad(objective)(jnp.asarray(0.2, dtype=jnp.float64))

    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_magnetic_well_from_state_uses_finite_beta_vp(monkeypatch):
    captured = {}

    def _fake_scalars_from_state(**kwargs):
        captured.update(kwargs)
        return {"vp": jnp.asarray([0.0, 2.0, 1.5, 1.0], dtype=jnp.float64)}

    monkeypatch.setattr(finite_beta, "finite_beta_scalars_from_state", _fake_scalars_from_state)

    well = finite_beta.magnetic_well_from_state(
        state="state",
        static="static",
        indata="indata",
        signgs=-1,
    )

    assert captured == {"state": "state", "static": "static", "indata": "indata", "signgs": -1}
    np.testing.assert_allclose(np.asarray(well), 2.0 / 3.0)


def test_magnetic_well_helpers_are_public_exports():
    import vmec_jax as vj
    from vmec_jax import api

    np.testing.assert_allclose(np.asarray(vj.magnetic_well_from_vp([0.0, 2.0, 1.5, 1.0])), 2.0 / 3.0)
    assert vj.magnetic_well_from_state is finite_beta.magnetic_well_from_state
    assert api.magnetic_well_from_state is finite_beta.magnetic_well_from_state


def test_finite_beta_global_residuals_apply_one_sided_constraints(monkeypatch):
    def _fake_scalars_from_state(**_kwargs):
        return {
            "aspect": jnp.asarray(7.0),
            "min_iota": jnp.asarray(0.30),
            "mean_iota": jnp.asarray(0.35),
            "max_iota": jnp.asarray(0.80),
            "volavgB": jnp.asarray(2.50),
            "betatotal": jnp.asarray(0.04),
        }

    monkeypatch.setattr(finite_beta, "finite_beta_scalars_from_state", _fake_scalars_from_state)
    targets = FiniteBetaTargets(
        aspect_ratio=6.0,
        min_iota=0.41,
        min_average_iota=0.45,
        max_iota=0.70,
        volavgB=2.0,
        beta_total=0.05,
        aspect_weight=2.0,
        iota_weight=3.0,
        max_iota_weight=4.0,
        volavgB_weight=5.0,
        beta_weight=6.0,
    )

    residuals = finite_beta.finite_beta_global_residuals_from_state(
        state=None,
        static=None,
        indata=None,
        signgs=1,
        targets=targets,
    )

    np.testing.assert_allclose(
        np.asarray(residuals),
        [2.0, -0.33, -0.30, 0.40, 2.50, -0.06],
        rtol=1e-12,
        atol=1e-12,
    )


def test_finite_beta_global_residuals_are_zero_for_satisfied_one_sided_constraints(monkeypatch):
    def _fake_scalars_from_state(**_kwargs):
        return {
            "aspect": jnp.asarray(5.5),
            "min_iota": jnp.asarray(0.42),
            "mean_iota": jnp.asarray(0.46),
            "max_iota": jnp.asarray(0.65),
            "volavgB": jnp.asarray(2.0),
            "betatotal": jnp.asarray(0.05),
        }

    monkeypatch.setattr(finite_beta, "finite_beta_scalars_from_state", _fake_scalars_from_state)
    targets = FiniteBetaTargets(
        aspect_ratio=6.0,
        min_iota=0.41,
        min_average_iota=0.45,
        max_iota=0.70,
        volavgB=2.0,
        beta_total=0.05,
    )

    residuals = finite_beta.finite_beta_global_residuals_from_state(
        state=None,
        static=None,
        indata=None,
        signgs=1,
        targets=targets,
    )

    np.testing.assert_allclose(np.asarray(residuals), np.zeros(6))


def _make_fake_mercier_state_inputs(scale=1.0):
    trig = vmec_trig_tables(ntheta=6, nzeta=3, nfp=1, mmax=0, nmax=0, lasym=False, cache=False)
    s = jnp.linspace(0.0, 1.0, 4)
    shape = (4, int(trig.ntheta2), 3)
    cfg = SimpleNamespace(
        ntheta=6,
        nzeta=3,
        nfp=1,
        mpol=1,
        ntor=0,
        lasym=False,
        lconm1=False,
        lthreed=False,
    )
    static = SimpleNamespace(s=s, trig_vmec=trig, cfg=cfg, modes=ModeTable(m=np.array([0]), n=np.array([0])))
    zeros = jnp.zeros((4, 1), dtype=jnp.float64)
    state = SimpleNamespace(
        Rcos=jnp.asarray([[scale], [1.0], [1.0], [1.0]], dtype=jnp.float64),
        Rsin=zeros,
        Zcos=zeros,
        Zsin=zeros,
    )
    return state, static, shape


def _patch_fake_mercier_state_dependencies(monkeypatch, shape):
    one = jnp.ones(shape, dtype=jnp.float64)
    s = jnp.linspace(0.0, 1.0, shape[0])

    monkeypatch.setattr(
        finite_beta,
        "_wout_like_for_state",
        lambda **_kwargs: (
            SimpleNamespace(
                phips=jnp.asarray([0.0, 1.0, 1.0, 1.0], dtype=jnp.float64),
                iotas=jnp.asarray([0.0, 0.1, 0.2, 0.3], dtype=jnp.float64),
            ),
            jnp.asarray([0.0, 0.1, 0.1, 0.1], dtype=jnp.float64),
        ),
    )
    monkeypatch.setattr(
        finite_beta,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **_kwargs: SimpleNamespace(vp=jnp.asarray([0.0, 1.0, 1.1, 1.2], dtype=jnp.float64)),
    )

    def fake_bcovar(*, state, **_kwargs):
        scale = jnp.asarray(state.Rcos[0, 0], dtype=jnp.float64)
        return SimpleNamespace(
            bsupu=0.2 * one,
            bsupv=0.3 * one,
            bsubu=scale * (0.4 * one + 0.05 * s[:, None, None]),
            bsubv=0.5 * one + 0.03 * s[:, None, None],
            bsq=1.0 + 0.2 * one,
            jac=SimpleNamespace(
                sqrtg=one,
                rs=0.11 * one,
                zs=0.12 * one,
                ru12=0.2 * one,
                zu12=0.3 * one,
            ),
        )

    monkeypatch.setattr(finite_beta, "vmec_bcovar_half_mesh_from_wout", fake_bcovar)
    monkeypatch.setattr(
        finite_beta,
        "mercier_realspace_geometry_channels_from_state",
        lambda **_kwargs: {
            "R_even": one,
            "R_odd": jnp.zeros_like(one),
            "Z_even": 0.1 * one,
            "Z_odd": jnp.zeros_like(one),
            "Ru_even": 0.2 * one,
            "Ru_odd": jnp.zeros_like(one),
            "Zu_even": 0.3 * one,
            "Zu_odd": jnp.zeros_like(one),
            "Rv_even": 0.1 * one,
            "Rv_odd": jnp.zeros_like(one),
            "Zv_even": 0.2 * one,
            "Zv_odd": jnp.zeros_like(one),
        },
    )


def test_mercier_terms_from_state_composes_stellarator_symmetric_channels(monkeypatch):
    state, static, shape = _make_fake_mercier_state_inputs()
    _patch_fake_mercier_state_dependencies(monkeypatch, shape)

    terms = finite_beta.mercier_terms_from_state(
        state=state,
        static=static,
        indata=object(),
        signgs=1,
        include_channels=True,
    )

    for key in (
        "DMerc",
        "Dshear",
        "Dcurr",
        "Dwell",
        "Dgeod",
        "D_R",
        "H",
        "glasser_correction",
        "tpp",
        "tbb",
        "tjb",
        "tjj",
        "jdotb",
        "bdotb",
        "bdotgradv",
        "torcur",
        "ip",
    ):
        assert terms[key].shape == (4,)
        assert np.all(np.isfinite(np.asarray(terms[key])))
    assert terms["glasser_shear_valid"].shape == (4,)
    for key in ("gpp", "bsubs_half", "bsubs_full", "bsubsu", "bsubsv", "itheta", "izeta", "bdotk", "bdotk_merc", "sqrtg"):
        assert terms[key].shape == shape
        assert np.all(np.isfinite(np.asarray(terms[key])))


def test_mercier_terms_from_state_is_differentiable(monkeypatch):
    import jax

    _state, static, shape = _make_fake_mercier_state_inputs()
    _patch_fake_mercier_state_dependencies(monkeypatch, shape)

    def objective(scale):
        zeros = jnp.zeros((4, 1), dtype=jnp.float64)
        state = SimpleNamespace(
            Rcos=jnp.asarray([[scale], [1.0], [1.0], [1.0]], dtype=jnp.float64),
            Rsin=zeros,
            Zcos=zeros,
            Zsin=zeros,
        )
        terms = finite_beta.mercier_terms_from_state(
            state=state,
            static=static,
            indata=object(),
            signgs=1,
        )
        return jnp.sum(terms["DMerc"][1:-1]) + 0.1 * jnp.sum(terms["D_R"][1:-1])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_mercier_terms_from_state_composes_lasym_channels(monkeypatch):
    state, static, _shape = _make_fake_mercier_state_inputs()
    trig = vmec_trig_tables(ntheta=6, nzeta=3, nfp=1, mmax=0, nmax=0, lasym=True, cache=False)
    shape = (4, int(trig.ntheta3), 3)
    static.trig_vmec = trig
    static.cfg.lasym = True
    _patch_fake_mercier_state_dependencies(monkeypatch, shape)

    terms = finite_beta.mercier_terms_from_state(
        state=state,
        static=static,
        indata=object(),
        signgs=1,
        include_channels=True,
    )

    for key in (
        "DMerc",
        "Dshear",
        "Dcurr",
        "Dwell",
        "Dgeod",
        "D_R",
        "H",
        "glasser_correction",
        "tpp",
        "tbb",
        "tjb",
        "tjj",
        "jdotb",
        "bdotb",
        "bdotgradv",
        "torcur",
        "ip",
    ):
        assert terms[key].shape == (4,)
        assert np.all(np.isfinite(np.asarray(terms[key])))
    assert terms["glasser_shear_valid"].shape == (4,)
    for key in ("gpp", "bsubs_half", "bsubs_full", "bsubsu", "bsubsv", "itheta", "izeta", "bdotk", "bdotk_merc", "sqrtg"):
        assert terms[key].shape == shape
        assert np.all(np.isfinite(np.asarray(terms[key])))


def _mercier_terms_numpy_reference(
    *,
    s,
    phips,
    iotas,
    vp,
    pres,
    torcur,
    tpp,
    tbb,
    tjb,
    tjj,
    jdotb=None,
    bdotb=None,
    signgs=1,
):
    s = np.asarray(s, dtype=float)
    phips = np.asarray(phips, dtype=float)
    iotas = np.asarray(iotas, dtype=float)
    vp = np.asarray(vp, dtype=float)
    pres = np.asarray(pres, dtype=float)
    torcur = np.asarray(torcur, dtype=float)
    tpp = np.asarray(tpp, dtype=float)
    tbb = np.asarray(tbb, dtype=float)
    tjb = np.asarray(tjb, dtype=float)
    tjj = np.asarray(tjj, dtype=float)
    ns = s.shape[0]
    out = {
        key: np.zeros(ns)
        for key in (
            "DMerc",
            "Dshear",
            "Dcurr",
            "Dwell",
            "Dgeod",
            "D_R",
            "H",
            "glasser_correction",
            "shear",
            "vpp",
            "presp",
            "ip",
        )
    }
    out["glasser_shear_valid"] = np.zeros(ns, dtype=bool)
    if ns < 3:
        return out
    if jdotb is not None:
        jdotb = np.asarray(jdotb, dtype=float)
    if bdotb is not None:
        bdotb = np.asarray(bdotb, dtype=float)
    sign_jac = 1.0 if signgs >= 0 else -1.0
    phip_real = (2.0 * np.pi) * phips * sign_jac
    vp_real = np.zeros_like(phip_real)
    np.divide(sign_jac * (2.0 * np.pi) ** 2 * vp, phip_real, out=vp_real, where=phip_real != 0.0)
    vp_real[0] = 0.0
    hs = 1.0 / float(ns - 1)
    for i in range(1, ns - 1):
        phip_full = 0.5 * (phip_real[i + 1] + phip_real[i])
        denom = 0.0 if phip_full == 0.0 else 1.0 / (hs * phip_full)
        shear = (iotas[i + 1] - iotas[i]) * denom
        vpp = (vp_real[i + 1] - vp_real[i]) * denom
        presp = (pres[i + 1] - pres[i]) * denom
        ip = (torcur[i + 1] - torcur[i]) * denom
        dshear = 0.25 * shear * shear
        dcurr = -shear * (tjb[i] - ip * tbb[i])
        dwell = presp * (vpp - presp * tpp[i]) * tbb[i]
        dgeod = tjb[i] * tjb[i] - tbb[i] * tjj[i]
        out["Dshear"][i] = dshear
        out["Dcurr"][i] = dcurr
        out["Dwell"][i] = dwell
        out["Dgeod"][i] = dgeod
        out["DMerc"][i] = dshear + dcurr + dwell + dgeod
        out["shear"][i] = shear
        out["vpp"][i] = vpp
        out["presp"][i] = presp
        out["ip"][i] = ip
        if jdotb is not None and bdotb is not None and bdotb[i] != 0.0:
            h_term = shear * (tjb[i] - (jdotb[i] / bdotb[i]) * tbb[i])
        else:
            h_term = -dcurr
        out["H"][i] = h_term
        if shear != 0.0:
            out["glasser_shear_valid"][i] = True
            out["glasser_correction"][i] = (h_term - 0.5 * shear * shear) ** 2 / (shear * shear)
            out["D_R"][i] = -out["DMerc"][i] + out["glasser_correction"][i]
    return out


def test_mercier_terms_from_profile_integrals_matches_vmec_algebra():
    import vmec_jax as vj

    data = dict(
        s=np.linspace(0.0, 1.0, 5),
        phips=np.array([0.0, 0.8, 0.9, 1.0, 1.1]),
        iotas=np.array([0.0, 0.35, 0.40, 0.43, 0.45]),
        vp=np.array([0.0, 1.2, 1.4, 1.55, 1.7]),
        pres=np.array([0.0, 0.03, 0.022, 0.012, 0.0]),
        torcur=np.array([0.0, 0.4, 0.5, 0.58, 0.63]),
        tpp=np.array([0.0, 2.0, 2.1, 2.2, 0.0]),
        tbb=np.array([0.0, 0.7, 0.72, 0.74, 0.0]),
        tjb=np.array([0.0, 0.05, 0.04, 0.03, 0.0]),
        tjj=np.array([0.0, 0.08, 0.07, 0.06, 0.0]),
        jdotb=np.array([0.0, 0.12, 0.10, 0.08, 0.0]),
        bdotb=np.array([0.0, 0.90, 0.92, 0.95, 0.0]),
        signgs=1,
    )

    actual = finite_beta.mercier_terms_from_profile_integrals(**data)
    public_actual = vj.mercier_terms_from_profile_integrals(**data)
    expected = _mercier_terms_numpy_reference(**data)

    for key in (
        "DMerc",
        "Dshear",
        "Dcurr",
        "Dwell",
        "Dgeod",
        "D_R",
        "H",
        "glasser_correction",
        "shear",
        "vpp",
        "presp",
        "ip",
    ):
        np.testing.assert_allclose(np.asarray(actual[key]), expected[key], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(np.asarray(public_actual[key]), expected[key], rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(np.asarray(actual["glasser_shear_valid"]), expected["glasser_shear_valid"])
    np.testing.assert_array_equal(np.asarray(public_actual["glasser_shear_valid"]), expected["glasser_shear_valid"])


def test_glasser_resistive_interchange_matches_landreman_jorge_relation():
    import jax
    import vmec_jax as vj

    dmerc = jnp.asarray([0.0, 0.3, -0.1, 0.0], dtype=jnp.float64)
    shear = jnp.asarray([0.0, 0.8, 1.2, 0.0], dtype=jnp.float64)
    h_term = jnp.asarray([0.0, 0.25, 0.4, 0.0], dtype=jnp.float64)
    result = vj.glasser_resistive_interchange_from_mercier_terms(DMerc=dmerc, shear=shear, H=h_term)
    expected = -np.asarray(dmerc) + (np.asarray(h_term) - 0.5 * np.asarray(shear) ** 2) ** 2 / np.where(
        np.asarray(shear) != 0.0,
        np.asarray(shear) ** 2,
        1.0,
    )
    expected[[0, -1]] = 0.0
    np.testing.assert_allclose(np.asarray(result["D_R"]), expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(np.asarray(result["glasser_shear_valid"]), [False, True, True, False])

    vacuum = vj.glasser_resistive_interchange_from_mercier_terms(
        DMerc=0.25 * shear * shear,
        shear=shear,
        H=jnp.zeros_like(shear),
    )
    np.testing.assert_allclose(np.asarray(vacuum["D_R"]), 0.0, atol=1e-13)

    def objective(h_scale):
        terms = vj.glasser_resistive_interchange_from_mercier_terms(
            DMerc=dmerc,
            shear=shear,
            H=h_scale * h_term,
            shear_epsilon=1.0e-12,
        )
        return jnp.sum(terms["D_R"][1:-1])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0, dtype=jnp.float64))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_mercier_terms_from_profile_integrals_are_differentiable():
    import jax

    s = jnp.linspace(0.0, 1.0, 5)
    phips = jnp.asarray([0.0, 0.8, 0.9, 1.0, 1.1])
    iotas = jnp.asarray([0.0, 0.35, 0.40, 0.43, 0.45])
    vp = jnp.asarray([0.0, 1.2, 1.4, 1.55, 1.7])
    torcur = jnp.asarray([0.0, 0.4, 0.5, 0.58, 0.63])
    tpp = jnp.asarray([0.0, 2.0, 2.1, 2.2, 0.0])
    tbb = jnp.asarray([0.0, 0.7, 0.72, 0.74, 0.0])
    tjb = jnp.asarray([0.0, 0.05, 0.04, 0.03, 0.0])
    tjj = jnp.asarray([0.0, 0.08, 0.07, 0.06, 0.0])

    def objective(pressure_scale):
        pres = pressure_scale * jnp.asarray([0.0, 0.03, 0.022, 0.012, 0.0])
        terms = finite_beta.mercier_terms_from_profile_integrals(
            s=s,
            phips=phips,
            iotas=iotas,
            vp=vp,
            pres=pres,
            torcur=torcur,
            tpp=tpp,
            tbb=tbb,
            tjb=tjb,
            tjj=tjj,
        )
        return jnp.sum(terms["DMerc"][1:-1])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_mercier_short_mesh_helpers_return_zero_profiles():
    s = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    zeros_1d = jnp.zeros_like(s)
    terms = finite_beta.mercier_terms_from_profile_integrals(
        s=s,
        phips=zeros_1d,
        iotas=zeros_1d,
        vp=zeros_1d,
        pres=zeros_1d,
        torcur=zeros_1d,
        tpp=zeros_1d,
        tbb=zeros_1d,
        tjb=zeros_1d,
        tjj=zeros_1d,
    )
    for key in ("DMerc", "Dshear", "Dcurr", "Dwell", "Dgeod", "D_R", "H", "glasser_correction", "shear", "vpp", "presp", "ip"):
        np.testing.assert_allclose(np.asarray(terms[key]), 0.0)
    np.testing.assert_array_equal(np.asarray(terms["glasser_shear_valid"]), [False, False])

    shape = (2, 2, 2)
    zeros_3d = jnp.zeros(shape, dtype=jnp.float64)
    integrals = finite_beta.mercier_surface_integrals_from_realspace(
        phips=zeros_1d,
        sqrtg=zeros_3d,
        b2=zeros_3d,
        gpp=zeros_3d,
        bdotk_merc=zeros_3d,
        wint=jnp.ones(shape[1:], dtype=jnp.float64),
    )
    for key in ("tpp", "tbb", "tjb", "tjj"):
        np.testing.assert_allclose(np.asarray(integrals[key]), 0.0)

    profiles = finite_beta.jxbforce_profiles_from_realspace(
        phips=zeros_1d,
        sqrtg=zeros_3d,
        bsq=zeros_3d,
        pres=zeros_1d,
        vp=zeros_1d,
        bdotk=zeros_3d,
        wint=jnp.ones(shape[1:], dtype=jnp.float64),
    )
    for key in ("jdotb", "bdotb", "bdotgradv"):
        np.testing.assert_allclose(np.asarray(profiles[key]), 0.0)


def _mercier_integrals_numpy_reference(*, phips, sqrtg, b2, gpp, bdotk_merc, wint, signgs=1):
    phips = np.asarray(phips, dtype=float)
    sqrtg = np.asarray(sqrtg, dtype=float)
    b2 = np.asarray(b2, dtype=float)
    gpp = np.asarray(gpp, dtype=float)
    bdotk_merc = np.asarray(bdotk_merc, dtype=float)
    wint = np.asarray(wint, dtype=float)
    ns = phips.shape[0]
    out = {key: np.zeros(ns) for key in ("tpp", "tbb", "tjb", "tjj")}
    sign_jac = 1.0 if signgs >= 0 else -1.0
    phip_real = (2.0 * np.pi) * phips * sign_jac
    for i in range(1, ns - 1):
        phip_full = 0.5 * (phip_real[i + 1] + phip_real[i])
        gsqrt_raw = 0.5 * (sqrtg[i] + sqrtg[i + 1])
        gsqrt_full = np.zeros_like(gsqrt_raw) if phip_full == 0.0 else gsqrt_raw / phip_full
        b2i = 0.5 * (b2[i + 1] + b2[i])
        b2_safe = np.where(b2i != 0.0, b2i, 1.0)
        bdotj_norm = np.zeros_like(gsqrt_raw)
        np.divide(bdotk_merc[i], gsqrt_raw, out=bdotj_norm, where=gsqrt_raw != 0.0)
        jdotb = bdotj_norm * gpp[i] * gsqrt_full
        norm = (2.0 * np.pi) ** 2
        out["tpp"][i] = np.sum((gsqrt_full / b2_safe) * wint) * norm
        out["tbb"][i] = np.sum((b2i * gsqrt_full * gpp[i]) * wint) * norm
        out["tjb"][i] = np.sum(jdotb * wint) * norm
        out["tjj"][i] = np.sum((jdotb * bdotj_norm / b2_safe) * wint) * norm
    return out


def test_mercier_surface_integrals_from_realspace_match_vmec_reduction():
    rng = np.random.default_rng(1234)
    data = dict(
        phips=np.array([0.0, 0.8, 0.9, 1.0, 1.1]),
        sqrtg=1.0 + 0.2 * rng.random((5, 3, 4)),
        b2=2.0 + 0.1 * rng.random((5, 3, 4)),
        gpp=0.5 + 0.1 * rng.random((5, 3, 4)),
        bdotk_merc=0.02 * rng.random((5, 3, 4)),
        wint=np.full((3, 4), 1.0 / 12.0),
        signgs=1,
    )

    actual = finite_beta.mercier_surface_integrals_from_realspace(**data)
    expected = _mercier_integrals_numpy_reference(**data)

    for key in ("tpp", "tbb", "tjb", "tjj"):
        np.testing.assert_allclose(np.asarray(actual[key]), expected[key], rtol=1e-13, atol=1e-13)


def test_mercier_surface_integrals_from_realspace_are_differentiable():
    import jax

    phips = jnp.asarray([0.0, 0.8, 0.9, 1.0, 1.1])
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 3, endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi, 4, endpoint=False)
    grid = 1.0 + 0.1 * jnp.cos(theta)[None, :, None] + 0.05 * jnp.sin(zeta)[None, None, :]
    sqrtg = jnp.tile(grid, (5, 1, 1))
    gpp = 0.5 * jnp.ones_like(sqrtg)
    bdotk_merc = 0.02 * jnp.ones_like(sqrtg)
    wint = jnp.full((3, 4), 1.0 / 12.0)

    def objective(scale):
        b2 = scale * (2.0 + 0.1 * grid)
        b2 = jnp.tile(b2, (5, 1, 1))
        integrals = finite_beta.mercier_surface_integrals_from_realspace(
            phips=phips,
            sqrtg=sqrtg,
            b2=b2,
            gpp=gpp,
            bdotk_merc=bdotk_merc,
            wint=wint,
        )
        return jnp.sum(integrals["tpp"][1:-1] + integrals["tbb"][1:-1])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def _jxbforce_profiles_numpy_reference(*, phips, sqrtg, bsq, pres, vp, bdotk, wint, sigma_an=None, signgs=1):
    phips = np.asarray(phips, dtype=float)
    sqrtg = np.asarray(sqrtg, dtype=float)
    bsq = np.asarray(bsq, dtype=float)
    pres = np.asarray(pres, dtype=float)
    vp = np.asarray(vp, dtype=float)
    bdotk = np.asarray(bdotk, dtype=float)
    wint = np.asarray(wint, dtype=float)
    sigma_an = np.ones_like(sqrtg) if sigma_an is None else np.asarray(sigma_an, dtype=float)
    ns = phips.shape[0]
    out = {key: np.zeros(ns) for key in ("jdotb", "bdotb", "bdotgradv")}
    if ns < 3:
        return out
    dnorm1 = float((2.0 * np.pi) ** 2)
    sign_jac = 1.0 if signgs >= 0 else -1.0
    for js in range(1, ns - 1):
        denom = vp[js + 1] + vp[js]
        if denom == 0.0:
            continue
        tjnorm = 2.0 / denom / dnorm1 * sign_jac
        sqgb2 = sqrtg[js + 1] * (bsq[js + 1] - pres[js + 1]) + sqrtg[js] * (bsq[js] - pres[js])
        out["jdotb"][js] = dnorm1 * tjnorm * np.sum((bdotk[js] / sigma_an[js]) * wint)
        out["bdotb"][js] = dnorm1 * tjnorm * np.sum((sqgb2 / sigma_an[js]) * wint)
        out["bdotgradv"][js] = 0.5 * dnorm1 * tjnorm * (phips[js] + phips[js + 1])
    out["jdotb"][0] = 2.0 * out["jdotb"][1] - out["jdotb"][2]
    out["jdotb"][-1] = 2.0 * out["jdotb"][-2] - out["jdotb"][-3]
    out["bdotb"][0] = 2.0 * out["bdotb"][2] - out["bdotb"][1]
    out["bdotb"][-1] = 2.0 * out["bdotb"][-2] - out["bdotb"][-3]
    out["bdotgradv"][0] = 2.0 * out["bdotgradv"][1] - out["bdotgradv"][2]
    out["bdotgradv"][-1] = 2.0 * out["bdotgradv"][-2] - out["bdotgradv"][-3]
    return out


def test_jxbforce_profiles_from_realspace_match_vmec_reduction():
    rng = np.random.default_rng(5678)
    shape = (5, 4, 3)
    data = dict(
        phips=np.array([0.0, 0.7, 0.8, 0.9, 1.0]),
        sqrtg=1.0 + 0.2 * rng.random(shape),
        bsq=1.5 + 0.1 * rng.random(shape),
        pres=np.array([0.0, 0.04, 0.03, 0.02, 0.0]),
        vp=np.array([0.0, 1.0, 1.1, 1.2, 1.3]),
        bdotk=0.05 * rng.random(shape),
        wint=np.full(shape[1:], 1.0 / np.prod(shape[1:])),
        sigma_an=0.9 + 0.2 * rng.random(shape),
        signgs=-1,
    )

    actual = finite_beta.jxbforce_profiles_from_realspace(**data)
    expected = _jxbforce_profiles_numpy_reference(**data)

    for key in ("jdotb", "bdotb", "bdotgradv"):
        np.testing.assert_allclose(np.asarray(actual[key]), expected[key], rtol=1e-13, atol=1e-13)


def test_jxbforce_profiles_from_realspace_are_differentiable():
    import jax

    phips = jnp.asarray([0.0, 0.7, 0.8, 0.9, 1.0])
    sqrtg = jnp.ones((5, 3, 4))
    bsq_base = 1.5 + 0.1 * jnp.arange(60, dtype=jnp.float64).reshape((5, 3, 4)) / 60.0
    pres = jnp.asarray([0.0, 0.04, 0.03, 0.02, 0.0])
    vp = jnp.asarray([0.0, 1.0, 1.1, 1.2, 1.3])
    bdotk = 0.05 * jnp.ones((5, 3, 4))
    wint = jnp.full((3, 4), 1.0 / 12.0)

    def objective(scale):
        profiles = finite_beta.jxbforce_profiles_from_realspace(
            phips=phips,
            sqrtg=sqrtg,
            bsq=scale * bsq_base,
            pres=pres,
            vp=vp,
            bdotk=bdotk,
            wint=wint,
        )
        return jnp.sum(profiles["bdotb"][1:-1])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_redl_polynomial_profile_and_trapped_fraction_helpers():
    s = jnp.asarray([0.25, 0.5, 0.75], dtype=jnp.float64)
    values, derivs = finite_beta.polynomial_profile_and_derivative([1.0, 2.0, 3.0], s)
    np.testing.assert_allclose(np.asarray(values), 1.0 + 2.0 * np.asarray(s) + 3.0 * np.asarray(s) ** 2)
    np.testing.assert_allclose(np.asarray(derivs), 2.0 + 6.0 * np.asarray(s))

    modB = 2.0 * jnp.ones((3, 4, 5), dtype=jnp.float64)
    sqrtg = 7.0 * jnp.ones_like(modB)
    trapped = finite_beta.trapped_fraction_from_modb_sqrtg(modB=modB, sqrtg=sqrtg)
    np.testing.assert_allclose(np.asarray(trapped["Bmin"]), 2.0)
    np.testing.assert_allclose(np.asarray(trapped["Bmax"]), 2.0)
    np.testing.assert_allclose(np.asarray(trapped["epsilon"]), 0.0)
    np.testing.assert_allclose(np.asarray(trapped["f_t"]), 0.0, atol=1.0e-12)


def test_trapped_fraction_rejects_invalid_shapes():
    modB = jnp.ones((2, 3), dtype=jnp.float64)
    sqrtg = jnp.ones_like(modB)
    with pytest.raises(ValueError, match="shape"):
        finite_beta.trapped_fraction_from_modb_sqrtg(modB=modB, sqrtg=sqrtg)

    modB = jnp.ones((2, 3, 4), dtype=jnp.float64)
    sqrtg = jnp.ones((2, 3, 5), dtype=jnp.float64)
    with pytest.raises(ValueError, match="shape mismatch"):
        finite_beta.trapped_fraction_from_modb_sqrtg(modB=modB, sqrtg=sqrtg)


def test_redl_bootstrap_mismatch_normalization_and_grad():
    pytest.importorskip("jax")
    import jax

    vmec = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float64)
    redl = jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float64)
    residuals = finite_beta.redl_bootstrap_mismatch_from_profiles(jdotB_vmec=vmec, jdotB_redl=redl)
    np.testing.assert_allclose(np.sum(np.asarray(residuals) ** 2), 1.0)

    s = jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float64)

    def objective(scale):
        jdotB, details = finite_beta.redl_bootstrap_jdotb(
            s=s,
            G=jnp.asarray([2.0, 2.1, 2.2]),
            R=jnp.asarray([1.7, 1.8, 1.9]),
            iota=jnp.asarray([0.42, 0.44, 0.46]),
            epsilon=jnp.asarray([0.10, 0.11, 0.12]),
            f_t=jnp.asarray([0.20, 0.22, 0.24]),
            psi_edge=-1.0,
            nfp=2,
            helicity_n=0,
            ne_coeffs=scale * jnp.asarray([3.0e20, 0.0, -2.0e20]),
            Te_coeffs=jnp.asarray([8.0e3, -5.0e3]),
            Zeff_coeffs=1.0,
        )
        assert set(("L31", "L32", "alpha", "jdotB")).issubset(details)
        return jnp.sum(jdotB * jdotB)

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0, dtype=jnp.float64))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))


@pytest.mark.simsopt
@pytest.mark.skipif(
    os.environ.get("RUN_SIMSOPT_VALIDATION") != "1",
    reason="Set RUN_SIMSOPT_VALIDATION=1 to run optional SIMSOPT validation",
)
def test_redl_bootstrap_formula_matches_simsopt_when_available():
    try:
        simsopt_bootstrap = pytest.importorskip("simsopt.mhd.bootstrap")
        simsopt_profiles = pytest.importorskip("simsopt.mhd.profiles")
    except RuntimeError as exc:
        pytest.skip(f"SIMSOPT bootstrap import failed: {exc}")

    s = np.asarray([0.2, 0.5, 0.8])
    ne_coeffs = np.asarray([3.0e20, 0.0, -2.0e20])
    te_coeffs = np.asarray([8.0e3, -5.0e3])
    kwargs = {
        "helicity_n": 0,
        "s": s,
        "G": np.asarray([2.0, 2.1, 2.2]),
        "R": np.asarray([1.7, 1.8, 1.9]),
        "iota": np.asarray([0.42, 0.44, 0.46]),
        "epsilon": np.asarray([0.10, 0.11, 0.12]),
        "f_t": np.asarray([0.20, 0.22, 0.24]),
        "psi_edge": -1.0,
        "nfp": 2,
    }
    expected, _details = simsopt_bootstrap.j_dot_B_Redl(
        simsopt_profiles.ProfilePolynomial(ne_coeffs),
        simsopt_profiles.ProfilePolynomial(te_coeffs),
        simsopt_profiles.ProfilePolynomial(te_coeffs),
        1.0,
        **kwargs,
    )
    actual, actual_details = finite_beta.redl_bootstrap_jdotb(
        ne_coeffs=ne_coeffs,
        Te_coeffs=te_coeffs,
        Ti_coeffs=te_coeffs,
        Zeff_coeffs=1.0,
        **{key: jnp.asarray(value) if isinstance(value, np.ndarray) else value for key, value in kwargs.items()},
    )

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-13, atol=1.0e-8)
    np.testing.assert_allclose(np.asarray(actual_details["L31"]), np.asarray(_details.L31), rtol=1.0e-13, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(actual_details["L32"]), np.asarray(_details.L32), rtol=1.0e-13, atol=1.0e-14)


def _mercier_gpp_numpy_reference(
    *,
    s,
    phips,
    sqrtg,
    R_even,
    R_odd,
    Ru_even,
    Ru_odd,
    Zu_even,
    Zu_odd,
    Rv_even,
    Rv_odd,
    Zv_even,
    Zv_odd,
    signgs=1,
):
    s = np.asarray(s, dtype=float)
    phips = np.asarray(phips, dtype=float)
    sqrtg = np.asarray(sqrtg, dtype=float)
    out = np.zeros_like(sqrtg)
    sign_jac = 1.0 if signgs >= 0 else -1.0
    phip_real = (2.0 * np.pi) * phips * sign_jac
    for i in range(1, s.shape[0] - 1):
        phip_full = 0.5 * (phip_real[i + 1] + phip_real[i])
        gsqrt_raw = 0.5 * (sqrtg[i] + sqrtg[i + 1])
        gsqrt_full = np.zeros_like(gsqrt_raw) if phip_full == 0.0 else gsqrt_raw / phip_full
        sqs = np.sqrt(s[i])
        r1f = R_even[i] + sqs * R_odd[i]
        rtf = Ru_even[i] + sqs * Ru_odd[i]
        ztf = Zu_even[i] + sqs * Zu_odd[i]
        rzf = Rv_even[i] + sqs * Rv_odd[i]
        zzf = Zv_even[i] + sqs * Zv_odd[i]
        gtt = rtf * rtf + ztf * ztf
        denom = gtt * r1f * r1f + (rtf * zzf - rzf * ztf) ** 2
        out[i] = np.where(denom != 0.0, (gsqrt_full * gsqrt_full) / denom, 0.0)
    return out


def test_mercier_gpp_from_realspace_geometry_matches_vmec_formula():
    rng = np.random.default_rng(4321)
    shape = (5, 3, 4)
    data = dict(
        s=np.linspace(0.0, 1.0, shape[0]),
        phips=np.array([0.0, 0.8, 0.9, 1.0, 1.1]),
        sqrtg=1.0 + 0.2 * rng.random(shape),
        R_even=1.2 + 0.1 * rng.random(shape),
        R_odd=0.05 * rng.random(shape),
        Ru_even=0.2 + 0.1 * rng.random(shape),
        Ru_odd=0.04 * rng.random(shape),
        Zu_even=0.25 + 0.1 * rng.random(shape),
        Zu_odd=0.04 * rng.random(shape),
        Rv_even=0.1 + 0.05 * rng.random(shape),
        Rv_odd=0.02 * rng.random(shape),
        Zv_even=0.1 + 0.05 * rng.random(shape),
        Zv_odd=0.02 * rng.random(shape),
        signgs=1,
    )

    actual = finite_beta.mercier_gpp_from_realspace_geometry(**data)
    expected = _mercier_gpp_numpy_reference(**data)

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-13, atol=1e-13)


def test_mercier_gpp_from_realspace_geometry_is_differentiable():
    import jax

    shape = (5, 3, 4)
    s = jnp.linspace(0.0, 1.0, shape[0])
    phips = jnp.asarray([0.0, 0.8, 0.9, 1.0, 1.1])
    sqrtg = jnp.ones(shape, dtype=jnp.float64)
    ones = jnp.ones(shape, dtype=jnp.float64)

    def objective(scale):
        gpp = finite_beta.mercier_gpp_from_realspace_geometry(
            s=s,
            phips=phips,
            sqrtg=sqrtg,
            R_even=scale * (1.2 * ones),
            R_odd=0.05 * ones,
            Ru_even=0.2 * ones,
            Ru_odd=0.03 * ones,
            Zu_even=0.25 * ones,
            Zu_odd=0.02 * ones,
            Rv_even=0.1 * ones,
            Rv_odd=0.01 * ones,
            Zv_even=0.15 * ones,
            Zv_odd=0.01 * ones,
        )
        return jnp.sum(gpp[1:-1])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_mercier_realspace_geometry_channels_from_state_matches_synthesis():
    modes = ModeTable(m=np.array([0, 1, 2]), n=np.array([0, 0, 1]))
    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=2, mmax=2, nmax=1, lasym=False, cache=False)
    s = jnp.linspace(0.0, 1.0, 4)
    state = SimpleNamespace(
        Rcos=jnp.asarray(
            [
                [1.0, 0.2, 0.0],
                [1.1, 0.2, 0.03],
                [1.2, 0.25, 0.04],
                [1.3, 0.3, 0.05],
            ],
            dtype=jnp.float64,
        ),
        Rsin=jnp.asarray(
            [
                [0.0, 0.02, 0.0],
                [0.0, 0.02, 0.01],
                [0.0, 0.03, 0.015],
                [0.0, 0.04, 0.02],
            ],
            dtype=jnp.float64,
        ),
        Zcos=jnp.asarray(
            [
                [0.0, 0.015, 0.0],
                [0.0, 0.015, 0.008],
                [0.0, 0.02, 0.010],
                [0.0, 0.025, 0.012],
            ],
            dtype=jnp.float64,
        ),
        Zsin=jnp.asarray(
            [
                [0.0, 0.18, 0.0],
                [0.0, 0.18, 0.02],
                [0.0, 0.22, 0.03],
                [0.0, 0.26, 0.04],
            ],
            dtype=jnp.float64,
        ),
    )

    actual = finite_beta.mercier_realspace_geometry_channels_from_state(
        state=state,
        modes=modes,
        trig=trig,
        s=s,
        lconm1=False,
        lthreed=True,
        lasym=False,
        apply_scalxc=True,
    )

    mask_even = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float64)
    mask_odd = 1.0 - mask_even
    coeff_cos_stack = jnp.stack([state.Rcos, state.Zcos], axis=0)
    coeff_sin_stack = jnp.stack([state.Rsin, state.Zsin], axis=0)
    mask_stack = jnp.stack([mask_even, mask_odd], axis=0)
    coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
    coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
    stack, stack_t, stack_p = vmec_realspace_synthesis_multi(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
        derivs=("base", "dtheta", "dzeta"),
    )

    expected = {
        "R_even": stack[0, 0],
        "R_odd": stack[1, 0],
        "Z_even": stack[0, 1],
        "Z_odd": stack[1, 1],
        "Ru_even": stack_t[0, 0],
        "Ru_odd": stack_t[1, 0],
        "Zu_even": stack_t[0, 1],
        "Zu_odd": stack_t[1, 1],
        "Rv_even": stack_p[0, 0],
        "Rv_odd": stack_p[1, 0],
        "Zv_even": stack_p[0, 1],
        "Zv_odd": stack_p[1, 1],
    }
    for key, value in expected.items():
        np.testing.assert_allclose(np.asarray(actual[key]), np.asarray(value), rtol=1e-13, atol=1e-13)


def test_mercier_realspace_geometry_channels_from_state_lasym_uses_phase_split():
    modes = ModeTable(m=np.array([0, 1]), n=np.array([0, 1]))
    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=2, mmax=1, nmax=1, lasym=True, cache=False)
    s = jnp.linspace(0.0, 1.0, 4)
    state = SimpleNamespace(
        Rcos=jnp.asarray([[1.0, 0.2], [1.1, 0.2], [1.2, 0.25], [1.3, 0.3]], dtype=jnp.float64),
        Rsin=jnp.asarray([[0.0, 0.02], [0.0, 0.02], [0.0, 0.03], [0.0, 0.04]], dtype=jnp.float64),
        Zcos=jnp.asarray([[0.0, 0.015], [0.0, 0.015], [0.0, 0.02], [0.0, 0.025]], dtype=jnp.float64),
        Zsin=jnp.asarray([[0.0, 0.18], [0.0, 0.18], [0.0, 0.22], [0.0, 0.26]], dtype=jnp.float64),
    )

    actual = finite_beta.mercier_realspace_geometry_channels_from_state(
        state=state,
        modes=modes,
        trig=trig,
        s=s,
        lconm1=False,
        lthreed=True,
        lasym=True,
        apply_scalxc=True,
        phase_split=True,
    )

    coeff_cos_stack = jnp.stack([state.Rcos, state.Zcos], axis=0)
    coeff_sin_stack = jnp.stack([state.Rsin, state.Zsin], axis=0)
    zeros = jnp.zeros_like(coeff_cos_stack)
    stack, stack_t, stack_p = vmec_realspace_synthesis_multi(
        coeff_cos=jnp.stack([coeff_cos_stack, zeros], axis=0),
        coeff_sin=jnp.stack([zeros, coeff_sin_stack], axis=0),
        modes=modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=False,
        s=s,
        derivs=("base", "dtheta", "dzeta"),
    )

    expected = {
        "R_even": stack[0, 0],
        "R_odd": stack[1, 0],
        "Z_even": stack[0, 1],
        "Z_odd": stack[1, 1],
        "Ru_even": stack_t[0, 0],
        "Ru_odd": stack_t[1, 0],
        "Zu_even": stack_t[0, 1],
        "Zu_odd": stack_t[1, 1],
        "Rv_even": stack_p[0, 0],
        "Rv_odd": stack_p[1, 0],
        "Zv_even": stack_p[0, 1],
        "Zv_odd": stack_p[1, 1],
    }
    for key, value in expected.items():
        np.testing.assert_allclose(np.asarray(actual[key]), np.asarray(value), rtol=1e-13, atol=1e-13)


def test_mercier_realspace_geometry_channels_from_state_is_differentiable():
    import jax

    modes = ModeTable(m=np.array([0, 1]), n=np.array([0, 0]))
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)
    s = jnp.linspace(0.0, 1.0, 4)
    zeros = jnp.zeros((4, 2), dtype=jnp.float64)

    def objective(scale):
        state = SimpleNamespace(
            Rcos=jnp.stack([jnp.ones(4), scale * jnp.asarray([0.0, 0.2, 0.3, 0.4])], axis=1),
            Rsin=zeros,
            Zcos=zeros,
            Zsin=jnp.stack([jnp.zeros(4), jnp.asarray([0.0, 0.1, 0.2, 0.3])], axis=1),
        )
        channels = finite_beta.mercier_realspace_geometry_channels_from_state(
            state=state,
            modes=modes,
            trig=trig,
            s=s,
            lconm1=False,
            apply_scalxc=True,
        )
        return jnp.sum(channels["R_odd"] * channels["Zu_odd"])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def _mercier_bsubs_derivatives_lasym_false_numpy_reference(
    *,
    bsubs,
    trig,
    mmax_force,
    nmax_force,
):
    bsubs = np.asarray(bsubs, dtype=float)
    ns, _ntheta, nzeta = bsubs.shape
    nt2 = int(trig.ntheta2)
    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        zeros = np.zeros((ns, nt2, nzeta), dtype=float)
        return {"bsubsu": zeros, "bsubsv": zeros}

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosmum = np.asarray(trig.cosmum, dtype=float)[:nt2, : mmax + 1]
    sinmum = np.asarray(trig.sinmum, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=float)[:, : nmax + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=float)[:, : nmax + 1]

    dmult = np.full((mmax + 1, nmax + 1), 1.0 / float(trig.r0scale) ** 2)
    mnyq = np.asarray(trig.cosmui).shape[1] - 1
    nnyq = np.asarray(trig.cosnv).shape[1] - 1
    if mnyq > 0 and mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax:
        dmult[:, nnyq] *= 0.5

    bsubs_nt2 = bsubs[:, :nt2, :]
    f_theta_sin = np.einsum("sik,im->smk", bsubs_nt2, sinmui, optimize=True)
    f_theta_cos = np.einsum("sik,im->smk", bsubs_nt2, cosmui, optimize=True)
    bsubsmn1 = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True) * dmult[None, :, :]
    bsubsmn2 = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True) * dmult[None, :, :]

    tmp_su_1 = np.einsum("smn,im->sin", bsubsmn1, cosmum, optimize=True)
    tmp_su_2 = np.einsum("smn,im->sin", bsubsmn2, sinmum, optimize=True)
    bsubsu = np.einsum("sin,kn->sik", tmp_su_1, cosnv, optimize=True) + np.einsum(
        "sin,kn->sik", tmp_su_2, sinnv, optimize=True
    )

    tmp_sv_1 = np.einsum("smn,im->sin", bsubsmn1, sinmu, optimize=True)
    tmp_sv_2 = np.einsum("smn,im->sin", bsubsmn2, cosmu, optimize=True)
    bsubsv = np.einsum("sin,kn->sik", tmp_sv_1, sinnvn, optimize=True) + np.einsum(
        "sin,kn->sik", tmp_sv_2, cosnvn, optimize=True
    )
    return {"bsubsu": bsubsu, "bsubsv": bsubsv}


def _symoutput_split_numpy(*, f, trig, reversed_sym=False):
    f = np.asarray(f, dtype=float)
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    nzeta = int(f.shape[2])
    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    f_half = f[:, :nt2, :]
    f_ref = f[:, ir0, :][:, :, kk]
    if reversed_sym:
        return 0.5 * (f_half - f_ref), 0.5 * (f_half + f_ref)
    return 0.5 * (f_half + f_ref), 0.5 * (f_half - f_ref)


def _extend_parity_to_full_numpy(*, par0, par1, trig):
    par0 = np.asarray(par0, dtype=float)
    par1 = np.asarray(par1, dtype=float)
    ns, nt2, nzeta = par0.shape
    nt1 = int(trig.ntheta1)
    nt3 = int(getattr(trig, "ntheta3", nt2))
    full = np.zeros((ns, nt3, nzeta), dtype=float)
    full[:, :nt2, :] = par0 + par1
    if nt3 == nt2:
        return full
    i0 = np.arange(nt2, dtype=int)
    ir0 = np.where(i0 == 0, 0, nt1 - i0)
    kk = (nzeta - np.arange(nzeta, dtype=int)) % nzeta
    mask = ir0 >= nt2
    if np.any(mask):
        full[:, ir0[mask], :] = par0[:, mask, :][:, :, kk] - par1[:, mask, :][:, :, kk]
    return full


def test_mercier_symoutput_split_separates_vmec_reflection_parity():
    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=2, mmax=2, nmax=2, lasym=True, cache=False)
    nt2 = int(trig.ntheta2)
    nt3 = int(trig.ntheta3)
    nzeta = 5
    theta = 2.0 * np.pi * np.arange(nt3, dtype=float) / float(trig.ntheta1)
    zeta = 2.0 * np.pi * np.arange(nzeta, dtype=float) / float(nzeta)
    radial = np.asarray([1.0, 1.3], dtype=float)[:, None, None]
    cos_theta = np.cos(theta)[None, :, None]
    sin_theta = np.sin(theta)[None, :, None]
    cos_zeta = np.cos(zeta)[None, None, :]
    sin_zeta = np.sin(zeta)[None, None, :]

    sym_full = radial * (1.0 + 0.20 * cos_theta + 0.05 * cos_zeta + 0.03 * cos_theta * cos_zeta)
    asym_full = radial * (0.11 * sin_theta + 0.07 * sin_zeta + 0.04 * sin_theta * cos_zeta)
    full = sym_full + asym_full

    sym, asym = finite_beta._mercier_symoutput_split_jax(f=full, trig=trig)
    reversed_sym, reversed_asym = finite_beta._mercier_symoutput_split_jax(
        f=full,
        trig=trig,
        reversed_sym=True,
    )

    np.testing.assert_allclose(np.asarray(sym), sym_full[:, :nt2, :], rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(asym), asym_full[:, :nt2, :], rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(reversed_sym), asym_full[:, :nt2, :], rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(reversed_asym), sym_full[:, :nt2, :], rtol=1.0e-13, atol=1.0e-13)


def _mercier_bsubs_derivatives_lasym_true_numpy_reference(
    *,
    bsubs,
    trig,
    mmax_force,
    nmax_force,
):
    bsubs = np.asarray(bsubs, dtype=float)
    ns, _ntheta, nzeta = bsubs.shape
    nt2 = int(trig.ntheta2)
    nt3 = int(getattr(trig, "ntheta3", nt2))
    mmax = int(mmax_force)
    nmax = int(nmax_force)
    if mmax < 0 or nmax < 0:
        zeros = np.zeros((ns, nt3, nzeta), dtype=float)
        return {"bsubsu": zeros, "bsubsv": zeros}

    cosmui = np.asarray(trig.cosmui, dtype=float)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=float)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=float)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=float)[:nt2, : mmax + 1]
    cosmum = np.asarray(trig.cosmum, dtype=float)[:nt2, : mmax + 1]
    sinmum = np.asarray(trig.sinmum, dtype=float)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=float)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=float)[:, : nmax + 1]
    cosnvn = np.asarray(trig.cosnvn, dtype=float)[:, : nmax + 1]
    sinnvn = np.asarray(trig.sinnvn, dtype=float)[:, : nmax + 1]

    dmult = np.full((mmax + 1, nmax + 1), 1.0 / float(trig.r0scale) ** 2)
    mnyq = np.asarray(trig.cosmui).shape[1] - 1
    nnyq = np.asarray(trig.cosnv).shape[1] - 1
    if mnyq > 0 and mnyq <= mmax:
        dmult[mnyq, :] *= 0.5
    if nnyq > 0 and nnyq <= nmax:
        dmult[:, nnyq] *= 0.5

    bsubs_sym, bsubs_asym = _symoutput_split_numpy(f=bsubs, trig=trig, reversed_sym=True)

    f_theta_sin = np.einsum("sik,im->smk", bsubs_sym[:, :nt2, :], sinmui, optimize=True)
    f_theta_cos = np.einsum("sik,im->smk", bsubs_sym[:, :nt2, :], cosmui, optimize=True)
    bsubsmn1 = np.einsum("smk,kn->smn", f_theta_sin, cosnv, optimize=True) * dmult[None, :, :]
    bsubsmn2 = np.einsum("smk,kn->smn", f_theta_cos, sinnv, optimize=True) * dmult[None, :, :]

    tmp_su_1 = np.einsum("smn,im->sin", bsubsmn1, cosmum, optimize=True)
    tmp_su_2 = np.einsum("smn,im->sin", bsubsmn2, sinmum, optimize=True)
    bsubsu_s = np.einsum("sin,kn->sik", tmp_su_1, cosnv, optimize=True) + np.einsum(
        "sin,kn->sik", tmp_su_2, sinnv, optimize=True
    )

    tmp_sv_1 = np.einsum("smn,im->sin", bsubsmn1, sinmu, optimize=True)
    tmp_sv_2 = np.einsum("smn,im->sin", bsubsmn2, cosmu, optimize=True)
    bsubsv_s = np.einsum("sin,kn->sik", tmp_sv_1, sinnvn, optimize=True) + np.einsum(
        "sin,kn->sik", tmp_sv_2, cosnvn, optimize=True
    )

    f_theta_cos_a = np.einsum("sik,im->smk", bsubs_asym[:, :nt2, :], cosmui, optimize=True)
    f_theta_sin_a = np.einsum("sik,im->smk", bsubs_asym[:, :nt2, :], sinmui, optimize=True)
    bsubsmn3 = np.einsum("smk,kn->smn", f_theta_cos_a, cosnv, optimize=True) * dmult[None, :, :]
    bsubsmn4 = np.einsum("smk,kn->smn", f_theta_sin_a, sinnv, optimize=True) * dmult[None, :, :]

    tmp_su_3 = np.einsum("smn,im->sin", bsubsmn3, sinmum, optimize=True)
    tmp_su_4 = np.einsum("smn,im->sin", bsubsmn4, cosmum, optimize=True)
    bsubsu_a = np.einsum("sin,kn->sik", tmp_su_3, cosnv, optimize=True) + np.einsum(
        "sin,kn->sik", tmp_su_4, sinnv, optimize=True
    )

    tmp_sv_3 = np.einsum("smn,im->sin", bsubsmn3, cosmu, optimize=True)
    tmp_sv_4 = np.einsum("smn,im->sin", bsubsmn4, sinmu, optimize=True)
    bsubsv_a = np.einsum("sin,kn->sik", tmp_sv_3, sinnvn, optimize=True) + np.einsum(
        "sin,kn->sik", tmp_sv_4, cosnvn, optimize=True
    )

    return {
        "bsubsu": _extend_parity_to_full_numpy(par0=bsubsu_s, par1=bsubsu_a, trig=trig),
        "bsubsv": _extend_parity_to_full_numpy(par0=bsubsv_s, par1=bsubsv_a, trig=trig),
    }


def test_mercier_bsubs_derivatives_lasym_false_matches_vmec_transform():
    rng = np.random.default_rng(8765)
    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=3, mmax=4, nmax=3, lasym=False, cache=False)
    bsubs = rng.normal(size=(4, int(trig.ntheta2), 5))

    actual = finite_beta.mercier_bsubs_derivatives_lasym_false(
        bsubs=bsubs,
        trig=trig,
        mmax_force=2,
        nmax_force=2,
    )
    expected = _mercier_bsubs_derivatives_lasym_false_numpy_reference(
        bsubs=bsubs,
        trig=trig,
        mmax_force=2,
        nmax_force=2,
    )

    for key in ("bsubsu", "bsubsv"):
        np.testing.assert_allclose(np.asarray(actual[key]), expected[key], rtol=1e-13, atol=1e-13)


def test_mercier_bsubs_derivatives_lasym_false_is_differentiable():
    import jax

    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=3, mmax=4, nmax=3, lasym=False, cache=False)
    base = jnp.linspace(0.1, 1.0, 4 * int(trig.ntheta2) * 5, dtype=jnp.float64).reshape(
        (4, int(trig.ntheta2), 5)
    )

    def objective(scale):
        channels = finite_beta.mercier_bsubs_derivatives_lasym_false(
            bsubs=scale * base,
            trig=trig,
            mmax_force=2,
            nmax_force=2,
        )
        return jnp.sum(channels["bsubsu"] ** 2) + jnp.sum(channels["bsubsv"] ** 2)

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_mercier_bsubs_derivatives_lasym_true_matches_vmec_transform():
    rng = np.random.default_rng(24601)
    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=3, mmax=4, nmax=3, lasym=True, cache=False)
    bsubs = rng.normal(size=(4, int(trig.ntheta3), 5))

    actual = finite_beta.mercier_bsubs_derivatives_lasym_true(
        bsubs=bsubs,
        trig=trig,
        mmax_force=2,
        nmax_force=2,
    )
    expected = _mercier_bsubs_derivatives_lasym_true_numpy_reference(
        bsubs=bsubs,
        trig=trig,
        mmax_force=2,
        nmax_force=2,
    )

    for key in ("bsubsu", "bsubsv"):
        np.testing.assert_allclose(np.asarray(actual[key]), expected[key], rtol=1e-13, atol=1e-13)


def test_mercier_bsubs_derivatives_lasym_true_is_differentiable():
    import jax

    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=3, mmax=4, nmax=3, lasym=True, cache=False)
    base = jnp.linspace(0.1, 1.0, 4 * int(trig.ntheta3) * 5, dtype=jnp.float64).reshape(
        (4, int(trig.ntheta3), 5)
    )

    def objective(scale):
        channels = finite_beta.mercier_bsubs_derivatives_lasym_true(
            bsubs=scale * base,
            trig=trig,
            mmax_force=2,
            nmax_force=2,
        )
        return jnp.sum(channels["bsubsu"] ** 2) + jnp.sum(channels["bsubsv"] ** 2)

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_mercier_bsubs_half_mesh_from_geometry_matches_bss_formula():
    rng = np.random.default_rng(2468)
    shape = (4, 3, 2)
    data = {
        name: rng.normal(size=shape)
        for name in ("bsupu", "bsupv", "rs12", "zs12", "ru12", "zu12", "rv12", "zv12")
    }

    actual = finite_beta.mercier_bsubs_half_mesh_from_geometry(**data)
    g_su = data["rs12"] * data["ru12"] + data["zs12"] * data["zu12"]
    g_sv = data["rs12"] * data["rv12"] + data["zs12"] * data["zv12"]
    expected = data["bsupu"] * g_su + data["bsupv"] * g_sv

    np.testing.assert_allclose(np.asarray(actual["g_su"]), g_su, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(actual["g_sv"]), g_sv, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(actual["bsubs"]), expected, rtol=1e-13, atol=1e-13)


def test_mercier_bsubs_half_mesh_from_geometry_is_differentiable():
    import jax

    shape = (4, 3, 2)
    base = jnp.ones(shape, dtype=jnp.float64)

    def objective(scale):
        channels = finite_beta.mercier_bsubs_half_mesh_from_geometry(
            bsupu=scale * base,
            bsupv=0.5 * base,
            rs12=0.2 * base,
            zs12=0.3 * base,
            ru12=0.4 * base,
            zu12=0.5 * base,
            rv12=0.6 * base,
            zv12=0.7 * base,
        )
        return jnp.sum(channels["bsubs"])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_mercier_zeta_half_mesh_from_realspace_geometry_matches_vmec_formula():
    rng = np.random.default_rng(1357)
    shape = (5, 3, 2)
    data = dict(
        s=np.linspace(0.0, 1.0, shape[0]),
        Rv_even=rng.normal(size=shape),
        Rv_odd=rng.normal(size=shape),
        Zv_even=rng.normal(size=shape),
        Zv_odd=rng.normal(size=shape),
    )

    actual = finite_beta.mercier_zeta_half_mesh_from_realspace_geometry(**data)
    sh = np.sqrt(0.5 * (data["s"][1:] + data["s"][:-1]))[:, None, None]
    rv_inner = 0.5 * (data["Rv_even"][1:] + data["Rv_even"][:-1] + sh * (data["Rv_odd"][1:] + data["Rv_odd"][:-1]))
    zv_inner = 0.5 * (data["Zv_even"][1:] + data["Zv_even"][:-1] + sh * (data["Zv_odd"][1:] + data["Zv_odd"][:-1]))
    rv_expected = np.zeros(shape)
    zv_expected = np.zeros(shape)
    rv_expected[1:] = rv_inner
    zv_expected[1:] = zv_inner
    rv_expected[0] = rv_inner[0]
    zv_expected[0] = zv_inner[0]

    np.testing.assert_allclose(np.asarray(actual["rv12"]), rv_expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(actual["zv12"]), zv_expected, rtol=1e-13, atol=1e-13)


def test_mercier_bss_half_mesh_geometry_from_realspace_matches_vmec_formula():
    rng = np.random.default_rng(97531)
    shape = (5, 3, 2)
    data = dict(
        s=np.linspace(0.0, 1.0, shape[0]),
        rs=rng.normal(size=shape),
        zs=rng.normal(size=shape),
        R_odd=rng.normal(size=shape),
        Z_odd=rng.normal(size=shape),
        Rv_even=rng.normal(size=shape),
        Rv_odd=rng.normal(size=shape),
        Zv_even=rng.normal(size=shape),
        Zv_odd=rng.normal(size=shape),
    )

    actual = finite_beta.mercier_bss_half_mesh_geometry_from_realspace(**data)
    sh = np.sqrt(0.5 * (data["s"][1:] + data["s"][:-1]))[:, None, None]
    rs_expected = np.zeros(shape)
    zs_expected = np.zeros(shape)
    rs_expected[1:] = data["rs"][1:] + 0.25 * (data["R_odd"][1:] + data["R_odd"][:-1]) / sh
    zs_expected[1:] = data["zs"][1:] + 0.25 * (data["Z_odd"][1:] + data["Z_odd"][:-1]) / sh
    rs_expected[0] = rs_expected[1]
    zs_expected[0] = zs_expected[1]

    zeta_expected = finite_beta.mercier_zeta_half_mesh_from_realspace_geometry(
        s=data["s"],
        Rv_even=data["Rv_even"],
        Rv_odd=data["Rv_odd"],
        Zv_even=data["Zv_even"],
        Zv_odd=data["Zv_odd"],
    )

    np.testing.assert_allclose(np.asarray(actual["rs12"]), rs_expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(actual["zs12"]), zs_expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(actual["rv12"]), np.asarray(zeta_expected["rv12"]), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(actual["zv12"]), np.asarray(zeta_expected["zv12"]), rtol=1e-13, atol=1e-13)


def test_mercier_zeta_half_mesh_from_realspace_geometry_is_differentiable():
    import jax

    shape = (5, 3, 2)
    s = jnp.linspace(0.0, 1.0, shape[0])
    base = jnp.ones(shape, dtype=jnp.float64)

    def objective(scale):
        channels = finite_beta.mercier_zeta_half_mesh_from_realspace_geometry(
            s=s,
            Rv_even=scale * base,
            Rv_odd=0.2 * base,
            Zv_even=0.3 * base,
            Zv_odd=0.4 * base,
        )
        return jnp.sum(channels["rv12"] + channels["zv12"])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def test_mercier_bsubs_full_mesh_from_half_mesh_matches_jxbforce_convention():
    bsubs_half = np.arange(5 * 2 * 3, dtype=float).reshape((5, 2, 3))
    expected = bsubs_half.copy()
    expected[1:-1] = 0.5 * (bsubs_half[1:-1] + bsubs_half[2:])
    expected[0] = 0.0

    actual = finite_beta.mercier_bsubs_full_mesh_from_half_mesh(bsubs_half=bsubs_half)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-13, atol=1e-13)


def test_mercier_bsubs_full_mesh_from_half_mesh_is_differentiable():
    import jax

    bsubs_half = jnp.arange(5 * 2 * 3, dtype=jnp.float64).reshape((5, 2, 3))

    def objective(scale):
        bsubs_full = finite_beta.mercier_bsubs_full_mesh_from_half_mesh(
            bsubs_half=scale * bsubs_half
        )
        return jnp.sum(bsubs_full[1:])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0


def _mercier_bdotk_numpy_reference(*, bsubu, bsubv, bsubsu, bsubsv, s):
    bsubu = np.asarray(bsubu, dtype=float)
    bsubv = np.asarray(bsubv, dtype=float)
    bsubsu = np.asarray(bsubsu, dtype=float)
    bsubsv = np.asarray(bsubsv, dtype=float)
    s = np.asarray(s, dtype=float)
    ns = s.shape[0]
    out = {key: np.zeros_like(bsubu) for key in ("itheta", "izeta", "bdotk", "bdotk_merc")}
    if ns < 3:
        return out
    hs = 1.0 / float(ns - 1)
    ohs = 1.0 / hs
    out["itheta"][1:-1] = bsubsv[1:-1] - ohs * (bsubv[2:] - bsubv[1:-1])
    out["izeta"][1:-1] = -bsubsu[1:-1] + ohs * (bsubu[2:] - bsubu[1:-1])
    out["izeta"][0] = 2.0 * out["izeta"][1] - out["izeta"][2]
    out["izeta"][-1] = 2.0 * out["izeta"][-2] - out["izeta"][-3]
    out["itheta"] = out["itheta"] / finite_beta.MU0
    out["izeta"] = out["izeta"] / finite_beta.MU0
    bsubu1 = 0.5 * (bsubu[2:] + bsubu[1:-1])
    bsubv1 = 0.5 * (bsubv[2:] + bsubv[1:-1])
    out["bdotk"][1:-1] = out["itheta"][1:-1] * bsubu1 + out["izeta"][1:-1] * bsubv1
    out["bdotk_merc"] = finite_beta.MU0 * out["bdotk"]
    return out


def test_mercier_bdotk_from_covariant_derivatives_matches_vmec_block():
    rng = np.random.default_rng(5678)
    data = dict(
        bsubu=0.1 + rng.random((5, 3, 4)),
        bsubv=0.2 + rng.random((5, 3, 4)),
        bsubsu=0.03 * rng.random((5, 3, 4)),
        bsubsv=0.04 * rng.random((5, 3, 4)),
        s=np.linspace(0.0, 1.0, 5),
    )

    actual = finite_beta.mercier_bdotk_from_covariant_derivatives(**data)
    expected = _mercier_bdotk_numpy_reference(**data)

    for key in ("itheta", "izeta", "bdotk", "bdotk_merc"):
        np.testing.assert_allclose(np.asarray(actual[key]), expected[key], rtol=1e-13, atol=1e-13)


def test_mercier_bdotk_from_covariant_derivatives_is_differentiable():
    import jax

    s = jnp.linspace(0.0, 1.0, 5)
    base = jnp.ones((5, 3, 4), dtype=jnp.float64)
    bsubu = 0.2 * base
    bsubv = 0.3 * base
    bsubsu = 0.01 * base
    bsubsv = 0.02 * base

    def objective(scale):
        channels = finite_beta.mercier_bdotk_from_covariant_derivatives(
            bsubu=scale * bsubu,
            bsubv=bsubv,
            bsubsu=bsubsu,
            bsubsv=bsubsv,
            s=s,
        )
        return jnp.sum(channels["bdotk_merc"][1:-1])

    value, grad = jax.value_and_grad(objective)(jnp.asarray(1.0))
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
    assert abs(float(np.asarray(grad))) > 0.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax import finite_beta
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.finite_beta import FiniteBetaTargets
from vmec_jax.modes import ModeTable
from vmec_jax.namelist import InData


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


def _mercier_terms_numpy_reference(*, s, phips, iotas, vp, pres, torcur, tpp, tbb, tjb, tjj, signgs=1):
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
    out = {key: np.zeros(ns) for key in ("DMerc", "Dshear", "Dcurr", "Dwell", "Dgeod", "shear", "vpp", "presp", "ip")}
    if ns < 3:
        return out
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
        signgs=1,
    )

    actual = finite_beta.mercier_terms_from_profile_integrals(**data)
    public_actual = vj.mercier_terms_from_profile_integrals(**data)
    expected = _mercier_terms_numpy_reference(**data)

    for key in ("DMerc", "Dshear", "Dcurr", "Dwell", "Dgeod", "shear", "vpp", "presp", "ip"):
        np.testing.assert_allclose(np.asarray(actual[key]), expected[key], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(np.asarray(public_actual[key]), expected[key], rtol=1e-13, atol=1e-13)


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

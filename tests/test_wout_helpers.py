from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.modes import ModeTable
from vmec_jax.namelist import InData
from vmec_jax.wout import (
    _apply_nyquist_half_weight,
    _apply_bsubv_equif_correction,
    _bool_from_nc,
    _bsubuv_parity_from_bcovar,
    _bsubuv_parity_from_coeffs,
    _bsubuv_parity_from_realspace_jxbforce,
    _bss_scalxc_undo_factor,
    _bss_should_undo_scalxc,
    _chipf_from_chips,
    _compute_aspectratio,
    _compute_ctor_from_buco,
    _compute_eqfor_beta,
    _compute_eqfor_betaxis,
    _filter_bsubuv_jxbforce,
    _filter_bsubuv_jxbforce_lasym_loop,
    _filter_bsubuv_jxbforce_loop,
    _filter_bsubuv_jxbforce_parity,
    _filter_bsubuv_jxbforce_parity_loop,
    _jxbforce_apply_bsubs_correction_lasym_false,
    _jxbforce_apply_bsubs_correction_lasym_true,
    _jxbforce_bsubsu_bsubsv_loop,
    _jxbforce_filter_with_bsubs_derivs_loop,
    _jxbforce_getbsubs_coeffs_lasym_false,
    _jxbforce_getbsubs_coeffs_lasym_true,
    _lambda_full_from_wout_half_mesh,
    _lambda_half_mesh_weights,
    _lambda_wout_from_full_mesh,
    _vmec_jxbforce_cos_coeffs,
    _vmec_jxbforce_sin_coeffs,
    _jxbforce_nyquist_limits,
    _glasser_from_wout_mercier_terms,
    _nc_scalar,
    _pshalf_from_s,
    _safe_divide,
    _undo_bss_scalxc_if_enabled,
    _vmec_symforce_antisym,
    _vmec_symforce_apply,
    _vmec_symoutput_expand,
    _vmec_symoutput_split,
    _vmec_wint_from_trig,
    _vmec_wint_from_trig_jax,
    _vmec_wrout_lasym_bsubuv_output_scale,
    _vmec_wrout_nyquist_cos_coeffs,
    _vmec_wrout_nyquist_lasym_loop,
    _vmec_wrout_nyquist_sin_coeffs,
    _vmec_wrout_nyquist_sin_coeffs_loop,
    _vmec_wrout_nyquist_synthesis,
    assert_main_modes_match_wout,
    _glasser_profiles_from_wout_data,
    _wout_current_profile_metadata_from_indata,
)
from vmec_jax.mercier import glasser_resistive_interchange_from_mercier_terms
from vmec_jax.io.wout import flux as wout_flux_helpers
from vmec_jax.io.wout.diagnostics import (
    glasser_from_wout_mercier_terms,
    glasser_profiles_from_wout_data,
    glasser_profiles_from_wout_variables,
)
from vmec_jax.io.wout.flux import wout_current_profile_metadata_from_indata
from vmec_jax.vmec_tomnsp import vmec_trig_tables


def _reference_mode_coeffs(*, f, modes, trig, kind: str, wrout: bool) -> np.ndarray:
    f = np.asarray(f, dtype=float)[:, : int(trig.ntheta2), :]
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    cosmui = np.asarray(trig.cosmui, dtype=float)[: int(trig.ntheta2), :]
    sinmui = np.asarray(trig.sinmui, dtype=float)[: int(trig.ntheta2), :]
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)
    if wrout:
        cosmui = cosmui.copy()
        cosnv = cosnv.copy()
        if cosmui.shape[1] > 1:
            cosmui[:, -1] *= 0.5
        if cosnv.shape[1] > 1:
            cosnv[:, -1] *= 0.5
        mscale = np.asarray(trig.mscale, dtype=float)
        nscale = np.asarray(trig.nscale, dtype=float)

    out = np.zeros((f.shape[0], m_arr.size), dtype=float)
    for js in range(f.shape[0]):
        for idx, (m, n) in enumerate(zip(m_arr, n_arr)):
            n_abs = abs(int(n))
            sgn = -1.0 if n < 0 else 1.0
            if wrout:
                dmult = 0.5 * mscale[m] * nscale[n_abs] / float(trig.r0scale) ** 2
                if m == 0 or n == 0:
                    dmult *= 2.0
            else:
                dmult = 1.0 / float(trig.r0scale) ** 2
                if m == cosmui.shape[1] - 1 and m > 0:
                    dmult *= 0.5
                if n_abs == cosnv.shape[1] - 1 and n_abs != 0:
                    dmult *= 0.5
            acc = 0.0
            for j in range(f.shape[1]):
                for k in range(f.shape[2]):
                    if kind == "cos":
                        basis = cosmui[j, m] * cosnv[k, n_abs] + sgn * sinmui[j, m] * sinnv[k, n_abs]
                    elif kind == "sin":
                        basis = sinmui[j, m] * cosnv[k, n_abs] - sgn * cosmui[j, m] * sinnv[k, n_abs]
                    else:
                        raise ValueError(kind)
                    acc += dmult * basis * f[js, j, k]
            out[js, idx] = acc
    return out


def test_scalar_and_boolean_netcdf_helpers_handle_masked_and_bad_inputs():
    assert _bool_from_nc(np.asarray([1])) is True
    assert _bool_from_nc(np.ma.masked_all((1,), dtype=int)) is False
    assert _nc_scalar(np.asarray([3.25])) == 3.25
    assert _nc_scalar(np.asarray([3.25]), as_int=True) == 3
    assert _nc_scalar(object(), default=4.5) == 4.5
    assert _nc_scalar(object(), default=4.5, as_int=True) == 4


def test_mesh_weight_and_safe_divide_helpers():
    np.testing.assert_allclose(_pshalf_from_s(np.asarray([0.0])), [0.0])
    np.testing.assert_allclose(
        _pshalf_from_s(np.asarray([0.0, 0.25, 1.0])),
        np.sqrt([0.125, 0.125, 0.625]),
    )
    np.testing.assert_allclose(_safe_divide(np.asarray([2.0, 3.0]), np.asarray([0.0, 6.0])), [2.0, 0.5])

    trig = SimpleNamespace(cosmui3=np.asarray([[2.0], [4.0]]), mscale=np.asarray([2.0]), cosnv=np.zeros((3, 1)))
    np.testing.assert_allclose(_vmec_wint_from_trig(trig), np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    with pytest.raises(ValueError, match="cosmui3"):
        _vmec_wint_from_trig(SimpleNamespace(cosmui3=np.asarray([1.0]), mscale=np.asarray([1.0]), cosnv=np.zeros((1, 1))))
    with pytest.raises(ValueError, match="non-empty"):
        _vmec_wint_from_trig(SimpleNamespace(cosmui3=np.asarray([[1.0]]), mscale=np.asarray([]), cosnv=np.zeros((1, 1))))

    np.testing.assert_allclose(
        np.asarray(_vmec_wint_from_trig_jax(trig)),
        np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    )
    with pytest.raises(ValueError, match="cosmui3"):
        _vmec_wint_from_trig_jax(
            SimpleNamespace(cosmui3=np.asarray([1.0]), mscale=np.asarray([1.0]), cosnv=np.zeros((1, 1)))
        )
    with pytest.raises(ValueError, match="non-empty"):
        _vmec_wint_from_trig_jax(
            SimpleNamespace(cosmui3=np.asarray([[1.0]]), mscale=np.asarray([]), cosnv=np.zeros((1, 1)))
        )


def test_wout_glasser_fallback_matches_current_term_reconstruction_and_private_alias():
    dmerc = np.asarray([0.0, 0.12, -0.03, 0.07, 0.0], dtype=float)
    dshear = np.asarray([0.0, 0.04, 0.00, 0.09, 0.0], dtype=float)
    dcurr = np.asarray([0.0, -0.05, 0.20, 0.03, 0.0], dtype=float)

    d_r, h_term, correction, valid = glasser_from_wout_mercier_terms(
        DMerc=dmerc,
        Dshear=dshear,
        Dcurr=dcurr,
    )
    alias = _glasser_from_wout_mercier_terms(DMerc=dmerc, Dshear=dshear, Dcurr=dcurr)

    shear2 = np.maximum(4.0 * dshear, 0.0)
    expected_h = -dcurr
    expected_valid = shear2 > 0.0
    expected_correction = np.zeros_like(dmerc)
    expected_correction[expected_valid] = (
        (expected_h[expected_valid] - 0.5 * shear2[expected_valid]) ** 2 / shear2[expected_valid]
    )
    expected_d_r = np.zeros_like(dmerc)
    expected_d_r[expected_valid] = -dmerc[expected_valid] + expected_correction[expected_valid]

    np.testing.assert_allclose(d_r, expected_d_r, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(h_term, expected_h, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(correction, expected_correction, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_array_equal(valid, expected_valid)
    for actual, expected in zip(alias, (d_r, h_term, correction, valid), strict=True):
        np.testing.assert_array_equal(actual, expected)

    public_terms = glasser_resistive_interchange_from_mercier_terms(
        DMerc=dmerc,
        shear=np.sqrt(shear2),
        H=expected_h,
    )
    np.testing.assert_allclose(np.asarray(public_terms["D_R"]), d_r, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(public_terms["H"]), h_term, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(public_terms["glasser_correction"]),
        correction,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_array_equal(np.asarray(public_terms["glasser_shear_valid"]), valid)


def test_wout_glasser_profile_reader_uses_persisted_or_fallback_variables():
    dmerc = np.asarray([0.0, 0.12, -0.03, 0.07, 0.0], dtype=float)
    dshear = np.asarray([0.0, 0.04, 0.00, 0.09, 0.0], dtype=float)
    dcurr = np.asarray([0.0, -0.05, 0.20, 0.03, 0.0], dtype=float)
    fallback = glasser_from_wout_mercier_terms(DMerc=dmerc, Dshear=dshear, Dcurr=dcurr)

    missing = glasser_profiles_from_wout_variables({}, DMerc=dmerc, Dshear=dshear, Dcurr=dcurr)
    np.testing.assert_allclose(missing.D_R, fallback[0])
    np.testing.assert_allclose(missing.H, fallback[1])
    np.testing.assert_allclose(missing.correction, fallback[2])
    np.testing.assert_array_equal(missing.shear_valid, fallback[3])

    variables = {
        "D_R": np.asarray([1, 2, 3, 4, 5], dtype=float),
        "HGlasser": np.asarray([2, 3, 4, 5, 6], dtype=float),
        "GlasserCorrection": np.asarray([3, 4, 5, 6, 7], dtype=float),
        "GlasserShearValid": np.asarray([1, 0, 1, 0, 1], dtype=float),
    }
    persisted = glasser_profiles_from_wout_variables(variables, DMerc=dmerc, Dshear=dshear, Dcurr=dcurr)
    np.testing.assert_allclose(persisted.D_R, variables["D_R"])
    np.testing.assert_allclose(persisted.H, variables["HGlasser"])
    np.testing.assert_allclose(persisted.correction, variables["GlasserCorrection"])
    np.testing.assert_array_equal(persisted.shear_valid, np.asarray([True, False, True, False, True]))

    legacy_h = glasser_profiles_from_wout_variables({"H": np.arange(5.0)}, DMerc=dmerc, Dshear=dshear, Dcurr=dcurr)
    np.testing.assert_allclose(legacy_h.H, np.arange(5.0))


def test_wout_glasser_profile_writer_bundle_uses_data_or_zero_defaults():
    wout = SimpleNamespace(
        D_R=np.asarray([0.0, 1.0, 2.0]),
        H=np.asarray([3.0, 4.0, 5.0]),
        glasser_correction=np.asarray([6.0, 7.0, 8.0]),
        glasser_shear_valid=np.asarray([True, False, True]),
    )
    profiles = glasser_profiles_from_wout_data(wout, 3)
    alias = _glasser_profiles_from_wout_data(wout, 3)

    np.testing.assert_allclose(profiles.D_R, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(profiles.H, [3.0, 4.0, 5.0])
    np.testing.assert_allclose(profiles.correction, [6.0, 7.0, 8.0])
    np.testing.assert_array_equal(profiles.shear_valid, [True, False, True])
    for actual, expected in zip(alias, profiles, strict=True):
        np.testing.assert_array_equal(actual, expected)

    defaults = glasser_profiles_from_wout_data(SimpleNamespace(), 4)
    np.testing.assert_allclose(defaults.D_R, np.zeros(4))
    np.testing.assert_allclose(defaults.H, np.zeros(4))
    np.testing.assert_allclose(defaults.correction, np.zeros(4))
    np.testing.assert_array_equal(defaults.shear_valid, np.zeros(4, dtype=bool))


def test_wout_current_profile_metadata_from_indata_preserves_vmecplot_defaults():
    defaults = wout_current_profile_metadata_from_indata(InData(scalars={}, indexed={}))
    np.testing.assert_allclose(defaults.ac, np.zeros(21))
    np.testing.assert_allclose(defaults.ac_aux_s, -np.ones(101))
    np.testing.assert_allclose(defaults.ac_aux_f, np.zeros(101))
    assert defaults.pcurr_type == "power_series"
    assert defaults.piota_type == "power_series"

    scalar_ac = wout_current_profile_metadata_from_indata(InData(scalars={"AC": 2.5}, indexed={}))
    assert scalar_ac.ac.shape == (21,)
    assert scalar_ac.ac[0] == pytest.approx(2.5)
    np.testing.assert_allclose(scalar_ac.ac[1:], 0.0)

    long_ac_values = [float(i) for i in range(25)]
    custom = InData(
        scalars={
            "AC": long_ac_values,
            "PCURR_TYPE": "cubic_spline_i",
            "PIOTA_TYPE": "akima_spline",
        },
        indexed={},
    )
    metadata = wout_current_profile_metadata_from_indata(custom, ndfmax=5)
    alias = _wout_current_profile_metadata_from_indata(custom, ndfmax=5)

    assert metadata.ac.shape == (25,)
    np.testing.assert_allclose(metadata.ac, long_ac_values)
    np.testing.assert_allclose(metadata.ac_aux_s, -np.ones(5))
    np.testing.assert_allclose(metadata.ac_aux_f, np.zeros(5))
    assert metadata.pcurr_type == "cubic_spline_i"
    assert metadata.piota_type == "akima_spline"
    for actual, expected in zip(alias, metadata, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_lambda_wout_half_mesh_roundtrip_covers_m_parity_branches():
    s = np.linspace(0.0, 1.0, 5)
    m_modes = np.asarray([0, 1, 2, 3], dtype=int)
    phipf_internal = np.asarray([1.0, 1.2, 1.4, 1.7, 2.0])
    lamscale = 2.5
    lam_full = np.asarray(
        [
            [0.25 / 1.2, -0.125 / 1.2, 0.00, 0.00],
            [0.25, -0.125, 0.12, -0.04],
            [0.32, 0.02, 0.18, 0.03],
            [0.45, 0.07, 0.24, 0.08],
            [0.51, 0.11, 0.31, 0.13],
        ],
        dtype=float,
    )

    lam_wout = _lambda_wout_from_full_mesh(
        lam_full=lam_full,
        m_modes=m_modes,
        s=s,
        phipf_internal=phipf_internal,
        lamscale=lamscale,
    )
    np.testing.assert_allclose(
        wout_flux_helpers.lambda_wout_from_full_mesh(
            lam_full=lam_full,
            m_modes=m_modes,
            s=s,
            phipf_internal=phipf_internal,
            lamscale=lamscale,
        ),
        lam_wout,
    )

    assert lam_wout.shape == lam_full.shape
    np.testing.assert_allclose(lam_wout[0], 0.0)
    recovered = _lambda_full_from_wout_half_mesh(
        lam_wout=lam_wout,
        m_modes=m_modes,
        s=s,
        phipf_internal=phipf_internal,
        lamscale=lamscale,
    )
    np.testing.assert_allclose(
        wout_flux_helpers.lambda_full_from_wout_half_mesh(
            lam_wout=lam_wout,
            m_modes=m_modes,
            s=s,
            phipf_internal=phipf_internal,
            lamscale=lamscale,
        ),
        recovered,
    )

    # Modes m>1 have no recoverable axis lambda after VMEC's half-mesh write
    # convention, but all physical half/full surfaces must roundtrip.
    expected = lam_full.copy()
    expected[0, m_modes > 1] = 0.0
    np.testing.assert_allclose(recovered, expected, rtol=1.0e-13, atol=1.0e-13)


def test_lambda_wout_half_mesh_validation_and_degenerate_scaling():
    s = np.linspace(0.0, 1.0, 3)
    lam = np.ones((3, 2))
    m_modes = np.asarray([0, 1])
    phipf_internal = np.asarray([0.0, 1.0, 2.0])

    zero_scale = _lambda_wout_from_full_mesh(
        lam_full=lam,
        m_modes=m_modes,
        s=s,
        phipf_internal=phipf_internal,
        lamscale=0.0,
    )
    np.testing.assert_allclose(zero_scale, 0.0)

    with pytest.raises(ValueError, match="lam_full"):
        _lambda_wout_from_full_mesh(
            lam_full=np.ones((4, 2)),
            m_modes=m_modes,
            s=s,
            phipf_internal=phipf_internal,
            lamscale=1.0,
        )
    with pytest.raises(ValueError, match="m_modes"):
        _lambda_wout_from_full_mesh(
            lam_full=lam,
            m_modes=np.asarray([0]),
            s=s,
            phipf_internal=phipf_internal,
            lamscale=1.0,
        )
    with pytest.raises(ValueError, match="phipf"):
        _lambda_wout_from_full_mesh(
            lam_full=lam,
            m_modes=m_modes,
            s=s,
            phipf_internal=np.ones(4),
            lamscale=1.0,
        )
    with pytest.raises(ValueError, match="lam_wout"):
        _lambda_full_from_wout_half_mesh(
            lam_wout=np.ones((4, 2)),
            m_modes=m_modes,
            s=s,
            phipf_internal=phipf_internal,
            lamscale=1.0,
        )
    with pytest.raises(ValueError, match="m_modes"):
        _lambda_full_from_wout_half_mesh(
            lam_wout=lam,
            m_modes=np.asarray([0]),
            s=s,
            phipf_internal=phipf_internal,
            lamscale=1.0,
        )
    with pytest.raises(ValueError, match="phipf"):
        _lambda_full_from_wout_half_mesh(
            lam_wout=lam,
            m_modes=m_modes,
            s=s,
            phipf_internal=np.ones(4),
            lamscale=1.0,
        )

    single = _lambda_wout_from_full_mesh(
        lam_full=np.asarray([[3.0, 4.0]]),
        m_modes=m_modes,
        s=np.asarray([0.0]),
        phipf_internal=np.asarray([1.0]),
        lamscale=1.0,
    )
    np.testing.assert_allclose(single, 0.0)
    recovered_single = _lambda_full_from_wout_half_mesh(
        lam_wout=np.asarray([[3.0, 4.0]]),
        m_modes=m_modes,
        s=np.asarray([0.0]),
        phipf_internal=np.asarray([1.0]),
        lamscale=1.0,
    )
    np.testing.assert_allclose(recovered_single, [[3.0, 4.0]])

    sm_f, sp_f = _lambda_half_mesh_weights(np.asarray([0.0]))
    np.testing.assert_allclose(sm_f, [0.0, 0.0])
    np.testing.assert_allclose(sp_f, [0.0, 0.0])


def test_bss_scalxc_undo_helpers(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_BSS_UNDO_SCALXC", raising=False)
    assert _bss_should_undo_scalxc() is False

    s = np.asarray([0.0, 0.25, 1.0])
    factor = _bss_scalxc_undo_factor(s)
    np.testing.assert_allclose(factor[:, 0, 0], [0.5, 0.5, 1.0])
    assert _bss_scalxc_undo_factor(np.asarray([])).shape == (0, 1, 1)
    np.testing.assert_allclose(_bss_scalxc_undo_factor(np.asarray([0.04]))[:, 0, 0], [1.0])

    a = np.ones((3, 2, 1))
    b = 2.0 * np.ones((3, 2, 1))
    out_a, out_b = _undo_bss_scalxc_if_enabled(s, a, b)
    assert out_a is a
    assert out_b is b

    monkeypatch.setenv("VMEC_JAX_BSS_UNDO_SCALXC", "1")
    assert _bss_should_undo_scalxc() is True
    out_a, out_b = _undo_bss_scalxc_if_enabled(s, a, b)
    np.testing.assert_allclose(out_a[:, 0, 0], [0.5, 0.5, 1.0])
    np.testing.assert_allclose(out_b[:, 0, 0], [1.0, 1.0, 2.0])


def test_eqfor_beta_and_aspectratio_helpers_are_finite_and_guard_shapes():
    ns = 4
    shape = (ns, 2, 3)
    pres = np.asarray([0.0, 0.2, 0.1, 0.05])
    vp = np.asarray([0.0, 1.0, 1.1, 1.2])
    bsq = np.full(shape, 3.0)
    r12 = np.full(shape, 2.0)
    bsupv = np.full(shape, 0.5)
    sqrtg = np.ones(shape)
    wint = np.full((2, 3), 0.25)

    betas = _compute_eqfor_beta(
        pres=pres,
        vp=vp,
        bsq=bsq,
        r12=r12,
        bsupv=bsupv,
        sqrtg=sqrtg,
        wint=wint,
        signgs=1,
    )
    assert len(betas) == 4
    assert np.all(np.isfinite(betas))
    assert _compute_eqfor_beta(
        pres=pres[:2],
        vp=vp[:2],
        bsq=bsq[:2],
        r12=r12[:2],
        bsupv=bsupv[:2],
        sqrtg=sqrtg[:2],
        wint=wint,
        signgs=1,
    ) == (0.0, 0.0, 0.0, 0.0)
    assert _compute_eqfor_betaxis(pres=pres[:2], vp=vp[:2], bsq=bsq[:2], sqrtg=sqrtg[:2], wint=wint, signgs=1) == 0.0
    assert np.isfinite(_compute_eqfor_betaxis(pres=pres, vp=vp, bsq=bsq, sqrtg=sqrtg, wint=wint, signgs=1))

    R = np.ones(shape)
    R[-1] = np.asarray([[2.0, 2.2, 2.4], [2.1, 2.3, 2.5]])
    Zu = np.ones(shape)
    aminor, rmajor, aspect, volume, area = _compute_aspectratio(R=R, Zu=Zu, wint=wint)
    assert aminor > 0.0
    assert rmajor > 0.0
    assert aspect > 0.0
    assert volume > 0.0
    assert area > 0.0
    with pytest.raises(ValueError, match="shape"):
        _compute_aspectratio(R=np.ones((2, 2)), Zu=Zu, wint=wint)
    with pytest.raises(ValueError, match="wint shape"):
        _compute_aspectratio(R=R, Zu=Zu, wint=np.ones((2, 2)))
    assert _compute_aspectratio(R=R, Zu=np.zeros_like(Zu), wint=wint) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_bsubv_equif_correction_and_ctor_branches(monkeypatch):
    import vmec_jax.wout as wout_module

    bsubv_short = np.ones((2, 2, 1))
    np.testing.assert_allclose(
        _apply_bsubv_equif_correction(bsubv=bsubv_short, bsubv_e=bsubv_short, trig=SimpleNamespace()),
        bsubv_short,
    )

    bsubv = np.arange(3 * 2 * 1, dtype=float).reshape(3, 2, 1)
    bsubv_e = np.ones_like(bsubv) * 2.0
    pwint = np.ones_like(bsubv) * 0.5
    monkeypatch.setattr(wout_module, "vmec_pwint_from_trig", lambda _trig, *, ns, nzeta: pwint)

    corrected = _apply_bsubv_equif_correction(bsubv=bsubv, bsubv_e=bsubv_e, trig=SimpleNamespace())

    assert corrected.shape == bsubv.shape
    np.testing.assert_allclose(np.sum(corrected[1] * pwint[1]), np.sum(bsubv[1] * pwint[1]))
    np.testing.assert_allclose(np.sum(corrected[2] * pwint[2]), np.sum(bsubv[2] * pwint[2]))

    monkeypatch.setattr(wout_module, "vmec_pwint_from_trig", lambda _trig, *, ns, nzeta: np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="pwint shape mismatch"):
        _apply_bsubv_equif_correction(bsubv=bsubv, bsubv_e=bsubv_e, trig=SimpleNamespace())

    assert _compute_ctor_from_buco(buco=np.asarray([3.0]), signgs=1, indata=InData(scalars={}, indexed={})) == 0.0

    fixed = InData(scalars={"LFREEB": False}, indexed={})
    np.testing.assert_allclose(
        _compute_ctor_from_buco(buco=np.asarray([1.0, 3.0]), signgs=-1, indata=fixed),
        -2.0 * np.pi * (1.5 * 3.0 - 0.5 * 1.0) / wout_module.MU0,
    )

    free_prec = InData(scalars={"LFREEB": True, "ICTRL_PREC2D": 2, "LHESS_EXACT": False}, indexed={})
    np.testing.assert_allclose(
        _compute_ctor_from_buco(buco=np.asarray([1.0, 3.0]), signgs=1, indata=free_prec),
        2.0 * np.pi * 3.0 / wout_module.MU0,
    )

    free_exact_no_prec = InData(scalars={"LFREEB": True, "ICTRL_PREC2D": 0, "LHESS_EXACT": True}, indexed={})
    np.testing.assert_allclose(
        _compute_ctor_from_buco(buco=np.asarray([1.0, 3.0]), signgs=1, indata=free_exact_no_prec),
        2.0 * np.pi * (1.5 * 3.0 - 0.5 * 1.0) / wout_module.MU0,
    )


def test_nyquist_limits_and_half_weighting():
    trig = SimpleNamespace(ntheta2=5, cosnv=np.zeros((6, 4)))
    assert _jxbforce_nyquist_limits(trig) == (4, 3)

    coeff_cos = np.ones((2, 4))
    coeff_sin = np.ones((2, 4)) * 2.0
    modes = ModeTable(m=np.asarray([0, 1, 2, 1]), n=np.asarray([0, 0, 1, -2]))
    cos_out, sin_out = _apply_nyquist_half_weight(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=modes, trig=trig)
    np.testing.assert_allclose(cos_out[:, [0, 1]], 1.0)
    np.testing.assert_allclose(sin_out[:, [0, 1]], 2.0)
    np.testing.assert_allclose(cos_out[:, [2, 3]], 0.5)
    np.testing.assert_allclose(sin_out[:, [2, 3]], 1.0)
    assert not np.shares_memory(cos_out, coeff_cos)
    with pytest.raises(ValueError, match="shape"):
        _apply_nyquist_half_weight(coeff_cos=np.ones(3), coeff_sin=coeff_sin, modes=modes, trig=trig)


def test_symmetry_split_apply_antisym_and_expand():
    trig = SimpleNamespace(ntheta2=3, ntheta1=4, ntheta3=5)
    f = np.arange(1 * 5 * 4, dtype=float).reshape(1, 5, 4)

    sym, asym = _vmec_symoutput_split(f=f, trig=trig)
    sym_rev, asym_rev = _vmec_symoutput_split(f=f, trig=trig, reversed_sym=True)
    np.testing.assert_allclose(sym + asym, f[:, :3, :])
    np.testing.assert_allclose(sym_rev, asym)
    np.testing.assert_allclose(asym_rev, sym)

    applied_sym = _vmec_symforce_apply(f=f, trig=trig, kind="ars")
    applied_antisym = _vmec_symforce_apply(f=f, trig=trig, kind="brs")
    np.testing.assert_allclose(applied_sym[:, :3, :], sym)
    np.testing.assert_allclose(applied_antisym[:, :3, :], asym)
    with pytest.raises(ValueError, match="unknown kind"):
        _vmec_symforce_apply(f=f, trig=trig, kind="bad")

    base = np.full_like(f, -1.0)
    anti = _vmec_symforce_antisym(f=f, trig=trig, kind="ars", base=base)
    np.testing.assert_allclose(anti[:, :3, :], asym)
    np.testing.assert_allclose(anti[:, 3:, :], -1.0)
    with pytest.raises(ValueError, match="base shape"):
        _vmec_symforce_antisym(f=f, trig=trig, kind="ars", base=np.zeros((1, 3, 4)))

    expanded = _vmec_symoutput_expand(sym=sym, asym=asym, trig=trig)
    assert expanded.shape == f.shape
    np.testing.assert_allclose(expanded[:, :3, :], f[:, :3, :])
    with pytest.raises(ValueError, match="sym/asym shape"):
        _vmec_symoutput_expand(sym=sym, asym=asym[:, :2, :], trig=trig)


def test_wrout_nyquist_coefficients_match_reference_loops():
    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=2, mmax=3, nmax=2, lasym=False, cache=False)
    modes = ModeTable(m=np.asarray([0, 1, 2, 3, 1]), n=np.asarray([0, 0, 1, -2, -1]))
    f = np.linspace(-0.7, 1.1, 2 * int(trig.ntheta2) * 5).reshape(2, int(trig.ntheta2), 5)

    cos_coeff = _vmec_wrout_nyquist_cos_coeffs(f=f, modes=modes, trig=trig)
    sin_coeff = _vmec_wrout_nyquist_sin_coeffs(f=f, modes=modes, trig=trig)

    np.testing.assert_allclose(
        cos_coeff,
        _reference_mode_coeffs(f=f, modes=modes, trig=trig, kind="cos", wrout=True),
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        sin_coeff,
        _reference_mode_coeffs(f=f, modes=modes, trig=trig, kind="sin", wrout=True),
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        sin_coeff,
        _vmec_wrout_nyquist_sin_coeffs_loop(f=f, modes=modes, trig=trig),
        rtol=2e-14,
        atol=2e-14,
    )

    empty = ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int))
    assert _vmec_wrout_nyquist_cos_coeffs(f=f, modes=empty, trig=trig).shape == (2, 0)
    assert _vmec_wrout_nyquist_sin_coeffs(f=f, modes=empty, trig=trig).shape == (2, 0)
    assert _vmec_wrout_nyquist_sin_coeffs_loop(f=f, modes=empty, trig=trig).shape == (2, 0)
    with pytest.raises(ValueError, match="shape"):
        _vmec_wrout_nyquist_cos_coeffs(f=f[0], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="smaller"):
        _vmec_wrout_nyquist_sin_coeffs(f=f[:, :2, :], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="mode limits"):
        _vmec_wrout_nyquist_sin_coeffs(
            f=f,
            modes=ModeTable(m=np.asarray([4]), n=np.asarray([0])),
            trig=trig,
        )


def test_wrout_nyquist_lasym_half_domain_unit_modes_normalize_to_one():
    trig = vmec_trig_tables(ntheta=8, nzeta=7, nfp=1, mmax=3, nmax=2, lasym=True, cache=False)
    modes = ModeTable(m=np.asarray([0, 1]), n=np.asarray([0, 1]))
    theta = 2.0 * np.pi * np.arange(int(trig.ntheta3)) / float(trig.ntheta1)
    zeta = 2.0 * np.pi * np.arange(7) / 7.0

    constant = np.ones((1, int(trig.ntheta3), 7), dtype=float)
    cos_mode = np.cos(theta[:, None] - zeta[None, :])[None, :, :]
    sin_mode = np.sin(theta[:, None] - zeta[None, :])[None, :, :]

    constant_sym, constant_asym = _vmec_symoutput_split(f=constant, trig=trig)
    cos_sym, cos_asym = _vmec_symoutput_split(f=cos_mode, trig=trig)
    sin_sym, sin_asym = _vmec_symoutput_split(f=sin_mode, trig=trig)

    constant_coeff = _vmec_wrout_nyquist_cos_coeffs(f=constant_sym, modes=modes, trig=trig)
    cos_coeff = _vmec_wrout_nyquist_cos_coeffs(f=cos_sym, modes=modes, trig=trig)
    sin_coeff = _vmec_wrout_nyquist_sin_coeffs(f=sin_asym, modes=modes, trig=trig)

    np.testing.assert_allclose(constant_coeff[0], [1.0, 0.0], rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(cos_coeff[0], [0.0, 1.0], rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(sin_coeff[0], [0.0, 1.0], rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(cos_asym, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(sin_sym, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(constant_asym, 0.0, atol=2.0e-14)

    def _with_axis(field):
        return np.concatenate([np.zeros_like(field), field], axis=0)

    loop_cos = _vmec_wrout_nyquist_lasym_loop(
        bsq=_with_axis(cos_sym),
        gsqrt=_with_axis(cos_sym),
        bsubu=_with_axis(cos_sym),
        bsubv=_with_axis(cos_sym),
        bsubs=np.zeros((2, int(trig.ntheta2), 7), dtype=float),
        bsupu=_with_axis(cos_sym),
        bsupv=_with_axis(cos_sym),
        modes=modes,
        trig=trig,
    )
    for name in ("gmnc", "bmnc", "bsupumnc", "bsupvmnc"):
        np.testing.assert_allclose(loop_cos[name][1], [0.0, 1.0], rtol=2.0e-14, atol=2.0e-14)

    loop_sin = _vmec_wrout_nyquist_lasym_loop(
        bsq=_with_axis(sin_asym),
        gsqrt=_with_axis(sin_asym),
        bsubu=_with_axis(sin_asym),
        bsubv=_with_axis(sin_asym),
        bsubs=np.zeros((2, int(trig.ntheta2), 7), dtype=float),
        bsupu=_with_axis(sin_asym),
        bsupv=_with_axis(sin_asym),
        modes=modes,
        trig=trig,
    )
    for name in ("gmns", "bmns", "bsupumns", "bsupvmns"):
        np.testing.assert_allclose(loop_sin[name][1], [0.0, 1.0], rtol=2.0e-14, atol=2.0e-14)

    scaled = _vmec_wrout_lasym_bsubuv_output_scale(
        bsubumnc=loop_cos["bsubumnc"],
        bsubvmnc=loop_cos["bsubvmnc"],
        bsubumns=loop_sin["bsubumns"],
        bsubvmns=loop_sin["bsubvmns"],
    )
    for coeff in scaled:
        np.testing.assert_allclose(coeff[1], [0.0, 2.0], rtol=2.0e-14, atol=2.0e-14)


def test_jxbforce_coefficients_match_reference_loops():
    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=2, mmax=3, nmax=2, lasym=False, cache=False)
    modes = ModeTable(m=np.asarray([0, 1, 2, 3, 1]), n=np.asarray([0, 0, 1, -2, -1]))
    f = np.cos(np.linspace(0.0, 2.0, 2 * int(trig.ntheta2) * 5)).reshape(2, int(trig.ntheta2), 5)

    np.testing.assert_allclose(
        _vmec_jxbforce_cos_coeffs(f=f, modes=modes, trig=trig),
        _reference_mode_coeffs(f=f, modes=modes, trig=trig, kind="cos", wrout=False),
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        _vmec_jxbforce_sin_coeffs(f=f, modes=modes, trig=trig),
        _reference_mode_coeffs(f=f, modes=modes, trig=trig, kind="sin", wrout=False),
        rtol=2e-14,
        atol=2e-14,
    )

    empty = ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int))
    assert _vmec_jxbforce_cos_coeffs(f=f, modes=empty, trig=trig).shape == (2, 0)
    assert _vmec_jxbforce_sin_coeffs(f=f, modes=empty, trig=trig).shape == (2, 0)
    with pytest.raises(ValueError, match="shape"):
        _vmec_jxbforce_cos_coeffs(f=f[0], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="smaller"):
        _vmec_jxbforce_sin_coeffs(f=f[:, :2, :], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="mode limits"):
        _vmec_jxbforce_cos_coeffs(
            f=f,
            modes=ModeTable(m=np.asarray([4]), n=np.asarray([0])),
            trig=trig,
        )


def test_wrout_nyquist_synthesis_and_chipf_edges():
    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=2, mmax=3, nmax=2, lasym=False, cache=False)
    modes = ModeTable(m=np.asarray([0, 1, 3, 1]), n=np.asarray([0, 0, -2, -1]))
    coeff_c = np.asarray([[1.0, -0.5, 0.25, 0.75], [0.1, 0.2, -0.3, 0.4]])
    coeff_s = np.asarray([[0.0, 0.2, -0.4, 0.6], [0.3, -0.1, 0.5, -0.7]])

    synth = _vmec_wrout_nyquist_synthesis(coeff_c=coeff_c, coeff_s=coeff_s, modes=modes, trig=trig)
    assert synth.shape == (2, int(trig.ntheta2), 5)

    m = modes.m
    n = modes.n
    n_abs = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)
    dmult = 0.5 * np.asarray(trig.mscale)[m] * np.asarray(trig.nscale)[n_abs] / float(trig.r0scale) ** 2
    dmult = np.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    raw_c = coeff_c / dmult[None, :]
    raw_s = coeff_s / dmult[None, :]
    expected = np.zeros_like(synth)
    for js in range(coeff_c.shape[0]):
        for j in range(int(trig.ntheta2)):
            for k in range(5):
                for idx, (mi, ni) in enumerate(zip(m, n)):
                    n1 = abs(int(ni))
                    sign = -1.0 if ni < 0 else 1.0
                    cos_basis = trig.cosmu[j, mi] * trig.cosnv[k, n1] + sign * trig.sinmu[j, mi] * trig.sinnv[k, n1]
                    sin_basis = trig.sinmu[j, mi] * trig.cosnv[k, n1] - sign * trig.cosmu[j, mi] * trig.sinnv[k, n1]
                    expected[js, j, k] += raw_c[js, idx] * cos_basis + raw_s[js, idx] * sin_basis
    np.testing.assert_allclose(synth, expected)

    empty = ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int))
    assert _vmec_wrout_nyquist_synthesis(coeff_c=np.zeros((2, 0)), coeff_s=np.zeros((2, 0)), modes=empty, trig=trig).shape == (
        2,
        0,
        0,
    )
    with pytest.raises(ValueError, match="shape"):
        _vmec_wrout_nyquist_synthesis(coeff_c=coeff_c[0], coeff_s=coeff_s, modes=modes, trig=trig)
    with pytest.raises(ValueError, match="mode limits"):
        _vmec_wrout_nyquist_synthesis(
            coeff_c=np.ones((1, 1)),
            coeff_s=np.ones((1, 1)),
            modes=ModeTable(m=np.asarray([4]), n=np.asarray([0])),
            trig=trig,
        )

    np.testing.assert_allclose(_chipf_from_chips(np.asarray([2.0])), [2.0])
    np.testing.assert_allclose(_chipf_from_chips(np.asarray([2.0, 4.0])), [4.0, 5.0])
    np.testing.assert_allclose(_chipf_from_chips(np.asarray([0.0, 2.0, 4.0, 8.0])), [1.0, 3.0, 6.0, 10.0])
    np.testing.assert_allclose(np.asarray(_chipf_from_chips(jnp.asarray([0.0, 2.0, 4.0, 8.0]))), [1.0, 3.0, 6.0, 10.0])


def test_schema_scalar_fallbacks_and_xm_mode_mismatch_branch():
    class BadArray:
        def __array__(self, dtype=None, copy=None):
            raise TypeError("not array-like")

        def __bool__(self):
            return True

    assert _bool_from_nc(BadArray()) is True
    assert _nc_scalar(np.asarray(["not-int"]), default=6, as_int=True) == 6
    assert _nc_scalar(np.asarray(["not-float"]), default=2.5) == 2.5

    bad_xm = SimpleNamespace(
        path="wout_bad_xm.nc",
        mpol=2,
        ntor=0,
        nfp=1,
        xm=np.asarray([0, 2]),
        xn=np.asarray([0, 0]),
    )
    with pytest.raises(ValueError, match="xm ordering"):
        assert_main_modes_match_wout(wout=bad_xm)


def test_bsubuv_parity_splits_coeffs_realspace_and_bcovar_scaling():
    trig = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=2, nmax=0, lasym=False, cache=False)
    modes = ModeTable(m=np.asarray([0, 1, 2]), n=np.asarray([0, 0, 0]))
    coeff_c = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    coeff_s = np.asarray([[0.0, 0.25, 0.5], [0.75, 1.0, 1.25]])

    bsubu_even, bsubu_odd, bsubv_even, bsubv_odd = _bsubuv_parity_from_coeffs(
        bsubumnc=coeff_c,
        bsubumns=coeff_s,
        bsubvmnc=2.0 * coeff_c,
        bsubvmns=-coeff_s,
        modes=modes,
        trig=trig,
    )

    full_u = _vmec_wrout_nyquist_synthesis(coeff_c=coeff_c, coeff_s=coeff_s, modes=modes, trig=trig)
    full_v = _vmec_wrout_nyquist_synthesis(coeff_c=2.0 * coeff_c, coeff_s=-coeff_s, modes=modes, trig=trig)
    np.testing.assert_allclose(bsubu_even + bsubu_odd, full_u, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(bsubv_even + bsubv_odd, full_v, rtol=2.0e-14, atol=2.0e-14)
    assert float(np.linalg.norm(bsubu_even)) > 0.0
    assert float(np.linalg.norm(bsubu_odd)) > 0.0

    constant_u = np.ones((2, int(trig.ntheta2), 1)) * 3.0
    constant_v = 2.0 * constant_u
    real_even_u, real_odd_u, real_even_v, real_odd_v = _bsubuv_parity_from_realspace_jxbforce(
        bsubu=constant_u,
        bsubv=constant_v,
        trig=trig,
    )
    np.testing.assert_allclose(real_even_u + real_odd_u, constant_u, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(real_even_v + real_odd_v, constant_v, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(real_odd_u, 0.0, atol=2.0e-14)

    with pytest.raises(ValueError, match="shape mismatch"):
        _bsubuv_parity_from_realspace_jxbforce(bsubu=constant_u, bsubv=constant_v[:, :1, :], trig=trig)
    with pytest.raises(ValueError, match="shape"):
        _bsubuv_parity_from_realspace_jxbforce(bsubu=constant_u[0], bsubv=constant_v[0], trig=trig)
    with pytest.raises(ValueError, match="smaller"):
        _bsubuv_parity_from_realspace_jxbforce(bsubu=constant_u[:, :1, :], bsubv=constant_v[:, :1, :], trig=trig)

    s = np.asarray([0.0, 0.25, 1.0])
    bsub_even = np.ones((3, 2, 1))
    _, odd_sqrt, _, odd_v_sqrt = _bsubuv_parity_from_bcovar(
        bsubu_even=bsub_even,
        bsubv_even=2.0 * bsub_even,
        s=s,
        iequi=0,
    )
    np.testing.assert_allclose(odd_sqrt[:, 0, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(odd_v_sqrt[:, 0, 0], [0.0, 1.0, 2.0])

    _, odd_half, _, _ = _bsubuv_parity_from_bcovar(
        bsubu_even=bsub_even,
        bsubv_even=2.0 * bsub_even,
        s=s,
        iequi=1,
    )
    np.testing.assert_allclose(odd_half[:, 0, 0], np.sqrt([0.125, 0.125, 0.625]))


def test_jxbforce_bsub_filters_match_loop_paths_and_guards(monkeypatch):
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=False, cache=False)
    ns = 3
    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    s = np.linspace(0.0, 1.0, ns)
    bsubu = np.linspace(-0.4, 0.7, ns * (nt2 + 1) * nzeta).reshape(ns, nt2 + 1, nzeta)
    bsubv = np.cos(bsubu)

    with pytest.raises(ValueError, match="smaller"):
        _filter_bsubuv_jxbforce(bsubu=bsubu[:, : nt2 - 1], bsubv=bsubv[:, : nt2 - 1], trig=trig, nfp=1, mmax_force=1, nmax_force=1)

    neg_u, neg_v = _filter_bsubuv_jxbforce(bsubu=bsubu, bsubv=bsubv, trig=trig, nfp=1, mmax_force=-1, nmax_force=1)
    np.testing.assert_allclose(neg_u, bsubu[:, :nt2, :])
    np.testing.assert_allclose(neg_v, bsubv[:, :nt2, :])

    vec_u, vec_v = _filter_bsubuv_jxbforce(
        bsubu=bsubu,
        bsubv=bsubv,
        trig=trig,
        nfp=1,
        mmax_force=1,
        nmax_force=1,
        s=s,
    )
    loop_u, loop_v = _filter_bsubuv_jxbforce_loop(
        bsubu=bsubu,
        bsubv=bsubv,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=s,
    )
    np.testing.assert_allclose(vec_u, loop_u, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(vec_v, loop_v, rtol=2.0e-14, atol=2.0e-14)

    full_u, full_v = _filter_bsubuv_jxbforce(
        bsubu=bsubu,
        bsubv=bsubv,
        trig=trig,
        nfp=1,
        mmax_force=2,
        nmax_force=1,
        s=s,
    )
    np.testing.assert_allclose(full_u, bsubu[:, :nt2, :])
    np.testing.assert_allclose(full_v, bsubv[:, :nt2, :])

    monkeypatch.setenv("VMEC_JAX_BSUB_FILTER_LOOP", "1")
    env_u, env_v = _filter_bsubuv_jxbforce(
        bsubu=bsubu,
        bsubv=bsubv,
        trig=trig,
        nfp=1,
        mmax_force=1,
        nmax_force=1,
        s=s,
    )
    np.testing.assert_allclose(env_u, loop_u, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(env_v, loop_v, rtol=2.0e-14, atol=2.0e-14)
    monkeypatch.delenv("VMEC_JAX_BSUB_FILTER_LOOP", raising=False)

    even_u = bsubu[:, :nt2, :]
    odd_u = 0.5 * even_u
    even_v = bsubv[:, :nt2, :]
    odd_v = 0.25 * even_v
    with pytest.raises(ValueError, match="shape mismatch"):
        _filter_bsubuv_jxbforce_parity(
            bsubu_even=even_u,
            bsubu_odd=odd_u[:, :1, :],
            bsubv_even=even_v,
            bsubv_odd=odd_v,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
        )
    with pytest.raises(ValueError, match="smaller"):
        _filter_bsubuv_jxbforce_parity(
            bsubu_even=even_u[:, : nt2 - 1],
            bsubu_odd=odd_u[:, : nt2 - 1],
            bsubv_even=even_v[:, : nt2 - 1],
            bsubv_odd=odd_v[:, : nt2 - 1],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
        )

    neg_pu, neg_pv = _filter_bsubuv_jxbforce_parity(
        bsubu_even=even_u,
        bsubu_odd=odd_u,
        bsubv_even=even_v,
        bsubv_odd=odd_v,
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
    )
    np.testing.assert_allclose(neg_pu, even_u)
    np.testing.assert_allclose(neg_pv, even_v)

    parity_u, parity_v = _filter_bsubuv_jxbforce_parity(
        bsubu_even=even_u,
        bsubu_odd=odd_u,
        bsubv_even=even_v,
        bsubv_odd=odd_v,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=s,
    )
    parity_loop_u, parity_loop_v = _filter_bsubuv_jxbforce_parity_loop(
        bsubu_even=even_u,
        bsubu_odd=odd_u,
        bsubv_even=even_v,
        bsubv_odd=odd_v,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=s,
    )
    np.testing.assert_allclose(parity_u, parity_loop_u, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(parity_v, parity_loop_v, rtol=2.0e-14, atol=2.0e-14)

    monkeypatch.setenv("VMEC_JAX_BSUB_FILTER_LOOP", "1")
    env_parity_u, env_parity_v = _filter_bsubuv_jxbforce_parity(
        bsubu_even=even_u,
        bsubu_odd=odd_u,
        bsubv_even=even_v,
        bsubv_odd=odd_v,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=s,
    )
    np.testing.assert_allclose(env_parity_u, parity_loop_u, rtol=2.0e-14, atol=2.0e-14)
    np.testing.assert_allclose(env_parity_v, parity_loop_v, rtol=2.0e-14, atol=2.0e-14)


def test_jxbforce_bsubs_derivative_filters_and_guards():
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=False, cache=False)
    ns = 3
    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    s = np.linspace(0.0, 1.0, ns)
    base = np.linspace(-0.2, 0.6, ns * nt2 * nzeta).reshape(ns, nt2, nzeta)
    bsubs = np.sin(base)
    even_u = base
    odd_u = 0.5 * base
    even_v = np.cos(base)
    odd_v = 0.25 * even_v

    outputs = _jxbforce_filter_with_bsubs_derivs_loop(
        bsubs=bsubs,
        bsubu_even=even_u,
        bsubu_odd=odd_u,
        bsubv_even=even_v,
        bsubv_odd=odd_v,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=s,
    )
    assert [arr.shape for arr in outputs] == [(ns, nt2, nzeta)] * 4
    assert all(np.all(np.isfinite(arr)) for arr in outputs)

    neg = _jxbforce_filter_with_bsubs_derivs_loop(
        bsubs=bsubs,
        bsubu_even=even_u,
        bsubu_odd=odd_u,
        bsubv_even=even_v,
        bsubv_odd=odd_v,
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
        s=s,
    )
    for arr in neg:
        np.testing.assert_allclose(arr, 0.0)

    with pytest.raises(ValueError, match="shape mismatch"):
        _jxbforce_filter_with_bsubs_derivs_loop(
            bsubs=bsubs,
            bsubu_even=even_u,
            bsubu_odd=odd_u[:, :1, :],
            bsubv_even=even_v,
            bsubv_odd=odd_v,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    with pytest.raises(ValueError, match="bsubs and parity"):
        _jxbforce_filter_with_bsubs_derivs_loop(
            bsubs=bsubs[:, :1, :],
            bsubu_even=even_u,
            bsubu_odd=odd_u,
            bsubv_even=even_v,
            bsubv_odd=odd_v,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    with pytest.raises(ValueError, match="smaller"):
        _jxbforce_filter_with_bsubs_derivs_loop(
            bsubs=bsubs[:, : nt2 - 1, :],
            bsubu_even=even_u[:, : nt2 - 1, :],
            bsubu_odd=odd_u[:, : nt2 - 1, :],
            bsubv_even=even_v[:, : nt2 - 1, :],
            bsubv_odd=odd_v[:, : nt2 - 1, :],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )

    bsubsu, bsubsv = _jxbforce_bsubsu_bsubsv_loop(bsubs=bsubs, trig=trig, mmax_force=1, nmax_force=1)
    assert bsubsu.shape == (ns, nt2, nzeta)
    assert bsubsv.shape == (ns, nt2, nzeta)
    assert np.all(np.isfinite(bsubsu))
    assert np.all(np.isfinite(bsubsv))

    zeros_u, zeros_v = _jxbforce_bsubsu_bsubsv_loop(bsubs=bsubs, trig=trig, mmax_force=-1, nmax_force=1)
    np.testing.assert_allclose(zeros_u, 0.0)
    np.testing.assert_allclose(zeros_v, 0.0)
    with pytest.raises(ValueError, match="smaller"):
        _jxbforce_bsubsu_bsubsv_loop(bsubs=bsubs[:, : nt2 - 1, :], trig=trig, mmax_force=1, nmax_force=1)


def test_lasym_filter_getbsubs_and_correction_fast_paths(monkeypatch):
    import vmec_jax.wout as wout_module

    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    ns = 2
    nt3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    s = np.asarray([0.0, 1.0])
    bsubu = np.linspace(-0.2, 0.8, ns * nt3 * nzeta).reshape(ns, nt3, nzeta)
    bsubv = np.cos(bsubu)

    with pytest.raises(ValueError, match="shape mismatch"):
        _filter_bsubuv_jxbforce_lasym_loop(
            bsubu=bsubu,
            bsubv=bsubv[:, :1, :],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
        )
    with pytest.raises(ValueError, match="parity channel"):
        _filter_bsubuv_jxbforce_lasym_loop(
            bsubu=bsubu,
            bsubv=bsubv,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            bsubu_even=bsubu[:, :1, :],
            bsubu_odd=0.5 * bsubu,
            bsubv_even=bsubv,
            bsubv_odd=0.25 * bsubv,
        )
    with pytest.raises(ValueError, match="smaller"):
        _filter_bsubuv_jxbforce_lasym_loop(
            bsubu=bsubu[:, : int(trig.ntheta3) - 1, :],
            bsubv=bsubv[:, : int(trig.ntheta3) - 1, :],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
        )

    neg_u, neg_v = _filter_bsubuv_jxbforce_lasym_loop(
        bsubu=bsubu,
        bsubv=bsubv,
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
    )
    np.testing.assert_allclose(neg_u, bsubu)
    np.testing.assert_allclose(neg_v, bsubv)

    filtered_u, filtered_v = _filter_bsubuv_jxbforce_lasym_loop(
        bsubu=bsubu,
        bsubv=bsubv,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=s,
        bsubu_even=bsubu,
        bsubu_odd=0.5 * bsubu,
        bsubv_even=bsubv,
        bsubv_odd=0.25 * bsubv,
    )
    assert filtered_u.shape == bsubu.shape
    assert filtered_v.shape == bsubv.shape
    assert np.all(np.isfinite(filtered_u))
    assert np.all(np.isfinite(filtered_v))

    trig_false = vmec_trig_tables(ntheta=4, nzeta=4, nfp=1, mmax=2, nmax=2, lasym=False, cache=False)
    nt2 = int(trig_false.ntheta2)
    nzeta_false = int(np.asarray(trig_false.cosnv).shape[0])
    grid = np.arange(nt2 * nzeta_false, dtype=float).reshape(nt2, nzeta_false)
    coeff_false = _jxbforce_getbsubs_coeffs_lasym_false(
        frho=0.1 + np.sin(grid),
        bsupu=1.0 + 0.2 * np.cos(grid),
        bsupv=0.7 + 0.1 * np.sin(2.0 * grid),
        trig=trig_false,
        nfp=1,
    )
    assert coeff_false is not None
    assert coeff_false.shape == (nt2, 2 * (nzeta_false // 2) + 1)
    assert _jxbforce_getbsubs_coeffs_lasym_false(
        frho=grid[:1],
        bsupu=np.ones_like(grid),
        bsupv=np.ones_like(grid),
        trig=trig_false,
        nfp=1,
    ) is None

    trig_true = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=2, nmax=0, lasym=True, cache=False)
    nt3_true = int(trig_true.ntheta3)
    grid_true = np.arange(nt3_true, dtype=float).reshape(nt3_true, 1)
    coeff_true = _jxbforce_getbsubs_coeffs_lasym_true(
        frho=0.1 + np.sin(grid_true),
        bsupu=1.0 + 0.2 * np.cos(grid_true),
        bsupv=0.7 + 0.1 * np.sin(2.0 * grid_true),
        trig=trig_true,
        nfp=1,
    )
    assert coeff_true is not None
    assert coeff_true.shape == (int(trig_true.ntheta2), 1, 2)
    assert _jxbforce_getbsubs_coeffs_lasym_true(
        frho=grid_true[:1],
        bsupu=np.ones_like(grid_true),
        bsupv=np.ones_like(grid_true),
        trig=trig_true,
        nfp=1,
    ) is None

    ns_corr = 3
    shape_false = (ns_corr, nt2, nzeta_false)
    ones_false = np.ones(shape_false)
    monkeypatch.setattr(wout_module, "_jxbforce_getbsubs_coeffs_lasym_false", lambda **_: np.ones_like(coeff_false) * 0.05)
    corr_false = _jxbforce_apply_bsubs_correction_lasym_false(
        bsubu=ones_false,
        bsubv=2.0 * ones_false,
        bsubs=np.zeros(shape_false),
        bsubsu=np.zeros(shape_false),
        bsubsv=np.zeros(shape_false),
        bsupu=0.3 * ones_false,
        bsupv=0.2 * ones_false,
        sqrtg=ones_false,
        pres=np.asarray([0.0, 0.1, 0.2]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        hs=0.5,
        signgs=1.0,
        trig=trig_false,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    assert [arr.shape for arr in corr_false] == [shape_false] * 3
    assert float(np.linalg.norm(corr_false[0][1])) > 0.0

    shape_true = (ns_corr, int(trig_true.ntheta2), 1)
    ones_true = np.ones(shape_true)
    monkeypatch.setattr(wout_module, "_jxbforce_getbsubs_coeffs_lasym_true", lambda **_: np.ones_like(coeff_true) * 0.05)
    corr_true = _jxbforce_apply_bsubs_correction_lasym_true(
        bsubu=ones_true,
        bsubv=2.0 * ones_true,
        bsubs=np.zeros(shape_true),
        bsubsu=np.zeros(shape_true),
        bsubsv=np.zeros(shape_true),
        bsupu=0.3 * ones_true,
        bsupv=0.2 * ones_true,
        sqrtg=ones_true,
        pres=np.asarray([0.0, 0.1, 0.2]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        hs=0.5,
        signgs=1.0,
        trig=trig_true,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    assert [arr.shape for arr in corr_true] == [(ns_corr, int(trig_true.ntheta3), 1)] * 3
    assert float(np.linalg.norm(corr_true[0][1])) > 0.0

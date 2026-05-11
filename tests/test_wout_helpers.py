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
    _chipf_from_chips,
    _compute_aspectratio,
    _compute_ctor_from_buco,
    _compute_eqfor_beta,
    _compute_eqfor_betaxis,
    _vmec_jxbforce_cos_coeffs,
    _vmec_jxbforce_sin_coeffs,
    _jxbforce_nyquist_limits,
    _nc_scalar,
    _pshalf_from_s,
    _safe_divide,
    _vmec_symforce_antisym,
    _vmec_symforce_apply,
    _vmec_symoutput_expand,
    _vmec_symoutput_split,
    _vmec_wint_from_trig,
    _vmec_wint_from_trig_jax,
    _vmec_wrout_nyquist_cos_coeffs,
    _vmec_wrout_nyquist_sin_coeffs,
    _vmec_wrout_nyquist_sin_coeffs_loop,
    _vmec_wrout_nyquist_synthesis,
)
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

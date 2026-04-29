from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.modes import ModeTable
from vmec_jax.wout import (
    _apply_nyquist_half_weight,
    _bool_from_nc,
    _compute_aspectratio,
    _compute_eqfor_beta,
    _compute_eqfor_betaxis,
    _jxbforce_nyquist_limits,
    _nc_scalar,
    _pshalf_from_s,
    _safe_divide,
    _vmec_symforce_antisym,
    _vmec_symforce_apply,
    _vmec_symoutput_expand,
    _vmec_symoutput_split,
    _vmec_wint_from_trig,
)


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

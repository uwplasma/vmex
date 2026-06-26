from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.profiles as profiles_module
import vmec_jax.wout as wout
from vmec_jax.modes import ModeTable
from vmec_jax.namelist import InData
from vmec_jax.kernels.tomnsp import vmec_trig_tables


def _grid(shape: tuple[int, ...], offset: float = 0.0, scale: float = 0.01) -> np.ndarray:
    return offset + scale * np.arange(int(np.prod(shape)), dtype=float).reshape(shape)


class _Var:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, _key):
        return self.value


def _indata(**scalars) -> InData:
    return InData(scalars=scalars, indexed={})


def test_scalar_weight_profile_and_beta_helpers_cover_errors_and_edges(monkeypatch) -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=3, nmax=2, lasym=False, cache=False)
    wint = wout._vmec_wint_from_trig(trig)
    assert wint.shape == (int(trig.ntheta3), int(np.asarray(trig.cosnv).shape[0]))
    np.testing.assert_allclose(np.asarray(wout._vmec_wint_from_trig_jax(trig)), wint)

    bad_ndim = SimpleNamespace(cosmui3=np.ones((2, 2, 1)), mscale=np.ones(1), cosnv=np.ones((1, 1)))
    with pytest.raises(ValueError, match="cosmui3"):
        wout._vmec_wint_from_trig(bad_ndim)
    with pytest.raises(ValueError, match="cosmui3"):
        wout._vmec_wint_from_trig_jax(bad_ndim)

    empty_scale = SimpleNamespace(cosmui3=np.ones((2, 1)), mscale=np.asarray([]), cosnv=np.ones((1, 1)))
    with pytest.raises(ValueError, match="non-empty"):
        wout._vmec_wint_from_trig(empty_scale)
    with pytest.raises(ValueError, match="non-empty"):
        wout._vmec_wint_from_trig_jax(empty_scale)

    np.testing.assert_allclose(wout._pshalf_from_s(np.asarray([0.25])), np.asarray([0.5]))
    np.testing.assert_allclose(
        wout._pshalf_from_s(np.asarray([0.0, 0.25, 1.0])),
        np.sqrt(np.asarray([0.125, 0.125, 0.625])),
    )

    monkeypatch.delenv("VMEC_JAX_BSS_UNDO_SCALXC", raising=False)
    arr = np.ones((3, 1, 1))
    assert not wout._bss_should_undo_scalxc()
    np.testing.assert_allclose(wout._undo_bss_scalxc_if_enabled(np.asarray([0.0, 0.25, 1.0]), arr)[0], arr)
    monkeypatch.setenv("VMEC_JAX_BSS_UNDO_SCALXC", "1")
    assert wout._bss_should_undo_scalxc()
    factor = wout._bss_scalxc_undo_factor(np.asarray([0.0, 0.25, 1.0]))
    np.testing.assert_allclose(factor[:, 0, 0], np.asarray([0.5, 0.5, 1.0]))
    np.testing.assert_allclose(
        wout._undo_bss_scalxc_if_enabled(np.asarray([0.0, 0.25, 1.0]), arr)[0][:, 0, 0],
        np.asarray([0.5, 0.5, 1.0]),
    )
    np.testing.assert_allclose(wout._bss_scalxc_undo_factor(np.asarray([0.0]))[:, 0, 0], np.asarray([1.0]))

    np.testing.assert_allclose(
        wout._safe_divide(np.asarray([2.0, 3.0]), np.asarray([0.0, 1.5])),
        np.asarray([2.0, 2.0]),
    )

    shape = (3, *wint.shape)
    assert wout._compute_eqfor_beta(
        pres=np.zeros(2),
        vp=np.ones(2),
        bsq=np.ones((2, *wint.shape)),
        r12=np.ones((2, *wint.shape)),
        bsupv=np.ones((2, *wint.shape)),
        sqrtg=np.ones((2, *wint.shape)),
        wint=wint,
        signgs=1,
    ) == (0.0, 0.0, 0.0, 0.0)
    beta = wout._compute_eqfor_beta(
        pres=np.asarray([0.0, 0.2, 0.1]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        bsq=2.0 + _grid(shape),
        r12=1.0 + _grid(shape, scale=0.02),
        bsupv=0.4 + _grid(shape, scale=0.01),
        sqrtg=np.ones(shape),
        wint=wint,
        signgs=1,
    )
    assert len(beta) == 4
    assert np.all(np.isfinite(beta))
    assert wout._compute_eqfor_betaxis(
        pres=np.zeros(2),
        vp=np.ones(2),
        bsq=np.ones((2, *wint.shape)),
        sqrtg=np.ones((2, *wint.shape)),
        wint=wint,
        signgs=1,
    ) == 0.0
    assert wout._compute_eqfor_betaxis(
        pres=np.zeros(3),
        vp=np.ones(3),
        bsq=np.zeros(shape),
        sqrtg=np.ones(shape),
        wint=wint,
        signgs=1,
    ) == 0.0

    with pytest.raises(ValueError, match="R/Zu"):
        wout._compute_aspectratio(R=np.ones(wint.shape), Zu=np.ones((1, *wint.shape)), wint=wint)
    with pytest.raises(ValueError, match="wint shape"):
        wout._compute_aspectratio(R=np.ones((1, *wint.shape)), Zu=np.ones((1, *wint.shape)), wint=np.ones((1, 1)))
    assert wout._compute_aspectratio(
        R=np.ones((1, *wint.shape)),
        Zu=np.zeros((1, *wint.shape)),
        wint=wint,
    )[:3] == (0.0, 0.0, 0.0)
    aspect = wout._compute_aspectratio(
        R=2.0 + _grid((2, *wint.shape), scale=0.02),
        Zu=1.0 + _grid((2, *wint.shape), scale=0.03),
        wint=wint,
    )
    assert np.all(np.isfinite(aspect))

    zero_equif = wout._compute_equif_wout(
        bsubu=np.ones((2, *wint.shape)),
        bsubv=np.ones((2, *wint.shape)),
        pres=np.zeros(2),
        vp=np.ones(2),
        phipf=np.ones(2),
        chipf=np.ones(2),
        signgs=1,
        trig=trig,
        s=np.asarray([0.0, 1.0]),
    )
    assert all(arr.shape == (2,) and not np.any(arr) for arr in zero_equif)
    equif = wout._compute_equif_wout(
        bsubu=0.2 + _grid((4, *wint.shape), scale=0.04),
        bsubv=-0.1 + _grid((4, *wint.shape), scale=0.03),
        pres=np.asarray([0.0, 0.3, 0.2, 0.05]),
        vp=np.asarray([1.0, 1.1, 1.3, 1.6]),
        phipf=np.asarray([0.0, 0.4, 0.5, 0.6]),
        chipf=np.asarray([0.0, 0.2, 0.25, 0.3]),
        signgs=-1,
        trig=trig,
        s=np.linspace(0.0, 1.0, 4),
    )
    assert all(arr.shape == (4,) for arr in equif)
    assert all(np.all(np.isfinite(arr)) for arr in equif)


def test_bsubv_correction_ctor_metadata_and_phi_profile_branches() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=3, nmax=2, lasym=False, cache=False)
    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    short = np.ones((2, nt2, nzeta))
    np.testing.assert_allclose(wout._apply_bsubv_equif_correction(bsubv=short, bsubv_e=short, trig=trig), short)

    shape = (4, nt2, nzeta)
    corrected = wout._apply_bsubv_equif_correction(
        bsubv=0.2 + _grid(shape, scale=0.03),
        bsubv_e=-0.1 + _grid(shape, scale=0.02),
        trig=trig,
    )
    assert corrected.shape == shape
    assert np.all(np.isfinite(corrected))

    with pytest.raises(ValueError, match="pwint shape"):
        wout._apply_bsubv_equif_correction(
            bsubv=np.ones((3, nt2 + 1, nzeta)),
            bsubv_e=np.ones((3, nt2 + 1, nzeta)),
            trig=trig,
        )

    assert wout._compute_ctor_from_buco(buco=np.asarray([1.0]), signgs=1, indata=_indata()) == 0.0
    buco = np.asarray([0.1, 0.4, 0.9])
    base = wout._compute_ctor_from_buco(
        buco=buco,
        signgs=1,
        indata=_indata(LFREEB=False, ICTRL_PREC2D=9, LHESS_EXACT=False),
    )
    lctor = wout._compute_ctor_from_buco(
        buco=buco,
        signgs=-1,
        indata=_indata(LFREEB=True, ICTRL_PREC2D=2, LHESS_EXACT=False),
    )
    lctor_exact = wout._compute_ctor_from_buco(
        buco=buco,
        signgs=1,
        indata=_indata(LFREEB=True, ICTRL_PREC2D=1, LHESS_EXACT=True),
    )
    no_lctor_exact = wout._compute_ctor_from_buco(
        buco=buco,
        signgs=1,
        indata=_indata(LFREEB=True, ICTRL_PREC2D=0, LHESS_EXACT=True),
    )
    assert np.isfinite(base)
    assert np.isfinite(lctor)
    assert np.isfinite(lctor_exact)
    assert np.isfinite(no_lctor_exact)
    assert base != lctor

    variables = {
        "ns": _Var(3),
        "mpol": _Var(2),
        "ntor": _Var(1),
        "nfp": _Var(5),
    }
    assert wout._read_wout_scalar_metadata(variables, path="synthetic.nc") == (3, 2, 1, 5, False, 1)
    variables["lasym__logical__"] = _Var(1)
    variables["signgs"] = _Var(-1)
    assert wout._read_wout_scalar_metadata(variables, path="synthetic.nc") == (3, 2, 1, 5, True, -1)
    variables["ns"] = _Var(0)
    with pytest.raises(ValueError, match="Incomplete"):
        wout._read_wout_scalar_metadata(variables, path="synthetic.nc")

    phi = np.asarray([0.0, 0.2, 0.5])
    np.testing.assert_allclose(wout._wout_phi_profile_from_variables({"phi": _Var(phi)}, ns=3, phipf=np.ones(3)), phi)
    synthesized = wout._wout_phi_profile_from_variables({}, ns=3, phipf=np.asarray([0.0, 2.0, 4.0]))
    assert synthesized.shape == (3,)
    assert np.all(np.isfinite(synthesized))


def test_filter_and_projection_helpers_cover_vectorized_error_identity_and_negative_paths(monkeypatch) -> None:
    monkeypatch.delenv("VMEC_JAX_BSUB_FILTER_LOOP", raising=False)
    trig = vmec_trig_tables(ntheta=4, nzeta=4, nfp=1, mmax=3, nmax=2, lasym=False, cache=False)
    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    shape = (2, nt2, nzeta)
    base = 0.3 + _grid(shape, scale=0.05)
    other = np.cos(base)
    s = np.asarray([0.0, 1.0])

    modes = ModeTable(m=np.asarray([0, 1, 2], dtype=int), n=np.asarray([0, 1, -1], dtype=int))
    coeffs = np.arange(6, dtype=float).reshape(2, 3) + 1.0
    parity = wout._bsubuv_parity_from_coeffs(
        bsubumnc=coeffs,
        bsubumns=0.1 * coeffs,
        bsubvmnc=0.2 * coeffs,
        bsubvmns=0.3 * coeffs,
        modes=modes,
        trig=trig,
    )
    assert [arr.shape for arr in parity] == [shape] * 4

    with pytest.raises(ValueError, match="shape mismatch"):
        wout._bsubuv_parity_from_realspace_jxbforce(bsubu=base, bsubv=other[:, :, :1], trig=trig)
    with pytest.raises(ValueError, match="Expected"):
        wout._bsubuv_parity_from_realspace_jxbforce(bsubu=base[0], bsubv=other[0], trig=trig)
    with pytest.raises(ValueError, match="smaller"):
        wout._bsubuv_parity_from_realspace_jxbforce(bsubu=base[:, : nt2 - 1], bsubv=other[:, : nt2 - 1], trig=trig)
    realspace_parity = wout._bsubuv_parity_from_realspace_jxbforce(bsubu=base, bsubv=other, trig=trig)
    assert [arr.shape for arr in realspace_parity] == [shape] * 4

    even, odd_sqrt, _, _ = wout._bsubuv_parity_from_bcovar(bsubu_even=base, bsubv_even=other, s=s, iequi=0)
    _, odd_half, _, _ = wout._bsubuv_parity_from_bcovar(bsubu_even=base, bsubv_even=other, s=s, iequi=1)
    np.testing.assert_allclose(even, base)
    assert not np.allclose(odd_sqrt[0], odd_half[0])

    with pytest.raises(ValueError, match="smaller"):
        wout._filter_bsubuv_jxbforce(
            bsubu=base[:, : nt2 - 1],
            bsubv=other[:, : nt2 - 1],
            trig=trig,
            nfp=1,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    neg_u, neg_v = wout._filter_bsubuv_jxbforce(
        bsubu=base,
        bsubv=other,
        trig=trig,
        nfp=1,
        mmax_force=-1,
        nmax_force=1,
        s=s,
    )
    np.testing.assert_allclose(neg_u, base)
    np.testing.assert_allclose(neg_v, other)
    full_u, full_v = wout._filter_bsubuv_jxbforce(
        bsubu=base,
        bsubv=other,
        trig=trig,
        nfp=1,
        mmax_force=99,
        nmax_force=99,
        s=s,
    )
    np.testing.assert_allclose(full_u, base)
    np.testing.assert_allclose(full_v, other)
    filt_u, filt_v = wout._filter_bsubuv_jxbforce(
        bsubu=base,
        bsubv=other,
        trig=trig,
        nfp=1,
        mmax_force=1,
        nmax_force=1,
        s=s,
    )
    assert filt_u.shape == shape
    assert filt_v.shape == shape

    with pytest.raises(ValueError, match="shape mismatch"):
        wout._filter_bsubuv_jxbforce_parity(
            bsubu_even=base,
            bsubu_odd=base[:, :, :1],
            bsubv_even=other,
            bsubv_odd=other,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    with pytest.raises(ValueError, match="smaller"):
        wout._filter_bsubuv_jxbforce_parity(
            bsubu_even=base[:, : nt2 - 1],
            bsubu_odd=base[:, : nt2 - 1],
            bsubv_even=other[:, : nt2 - 1],
            bsubv_odd=other[:, : nt2 - 1],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    parity_neg = wout._filter_bsubuv_jxbforce_parity(
        bsubu_even=base,
        bsubu_odd=0.5 * base,
        bsubv_even=other,
        bsubv_odd=0.25 * other,
        trig=trig,
        mmax_force=1,
        nmax_force=-1,
        s=s,
    )
    np.testing.assert_allclose(parity_neg[0], base)
    parity_filt = wout._filter_bsubuv_jxbforce_parity(
        bsubu_even=base,
        bsubu_odd=0.5 * base,
        bsubv_even=other,
        bsubv_odd=0.25 * other,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=s,
    )
    assert [arr.shape for arr in parity_filt] == [shape] * 2

    with pytest.raises(ValueError, match="shape mismatch"):
        wout._jxbforce_filter_with_bsubs_derivs_loop(
            bsubs=base,
            bsubu_even=base,
            bsubu_odd=base[:, :, :1],
            bsubv_even=other,
            bsubv_odd=other,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    with pytest.raises(ValueError, match="bsubs and parity"):
        wout._jxbforce_filter_with_bsubs_derivs_loop(
            bsubs=base[:, :1],
            bsubu_even=base,
            bsubu_odd=base,
            bsubv_even=other,
            bsubv_odd=other,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    coupled_neg = wout._jxbforce_filter_with_bsubs_derivs_loop(
        bsubs=base,
        bsubu_even=base,
        bsubu_odd=base,
        bsubv_even=other,
        bsubv_odd=other,
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
        s=s,
    )
    assert all(arr.shape == shape and not np.any(arr) for arr in coupled_neg)

    with pytest.raises(ValueError, match="smaller"):
        wout._filter_bsubuv_jxbforce_loop(
            bsubu=base[:, : nt2 - 1],
            bsubv=other[:, : nt2 - 1],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    loop_neg = wout._filter_bsubuv_jxbforce_loop(
        bsubu=base,
        bsubv=other,
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
        s=s,
    )
    np.testing.assert_allclose(loop_neg[0], base)
    loop_full = wout._filter_bsubuv_jxbforce_loop(
        bsubu=base,
        bsubv=other,
        trig=trig,
        mmax_force=99,
        nmax_force=99,
        s=s,
    )
    np.testing.assert_allclose(loop_full[0], base)

    with pytest.raises(ValueError, match="smaller"):
        wout._jxbforce_bsubsu_bsubsv_loop(bsubs=base[:, : nt2 - 1], trig=trig, mmax_force=1, nmax_force=1)
    deriv_neg = wout._jxbforce_bsubsu_bsubsv_loop(bsubs=base, trig=trig, mmax_force=1, nmax_force=-1)
    assert all(arr.shape == shape and not np.any(arr) for arr in deriv_neg)


def test_lasym_filter_nyquist_and_symmetry_helpers_cover_shape_branches() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=3, nmax=2, lasym=True, cache=False)
    nt2 = int(trig.ntheta2)
    nt3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    shape = (2, nt3, nzeta)
    base = 0.1 + _grid(shape, scale=0.04)
    other = np.sin(base)
    s = np.asarray([0.0, 1.0])

    with pytest.raises(ValueError, match="shape mismatch"):
        wout._filter_bsubuv_jxbforce_lasym_loop(
            bsubu=base,
            bsubv=other[:, :, :1],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    with pytest.raises(ValueError, match="parity channel"):
        wout._filter_bsubuv_jxbforce_lasym_loop(
            bsubu=base,
            bsubv=other,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
            bsubu_even=base[:, :, :1],
            bsubu_odd=base,
            bsubv_even=other,
            bsubv_odd=other,
        )
    with pytest.raises(ValueError, match="smaller"):
        wout._filter_bsubuv_jxbforce_lasym_loop(
            bsubu=base[:, : nt3 - 1],
            bsubv=other[:, : nt3 - 1],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s,
        )
    neg_u, neg_v = wout._filter_bsubuv_jxbforce_lasym_loop(
        bsubu=base,
        bsubv=other,
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
        s=s,
    )
    np.testing.assert_allclose(neg_u, base)
    np.testing.assert_allclose(neg_v, other)
    lasym_filtered = wout._filter_bsubuv_jxbforce_lasym_loop(
        bsubu=base,
        bsubv=other,
        trig=trig,
        mmax_force=1,
        nmax_force=1,
        s=s,
        bsubu_even=base,
        bsubu_odd=0.5 * base,
        bsubv_even=other,
        bsubv_odd=0.25 * other,
    )
    assert [arr.shape for arr in lasym_filtered] == [shape] * 2

    coeff = np.ones((2, 3))
    with pytest.raises(ValueError, match="Expected coeff"):
        wout._apply_nyquist_half_weight(coeff_cos=coeff[0], coeff_sin=coeff, modes=ModeTable(np.arange(3), np.zeros(3)), trig=trig)
    empty_weight = wout._apply_nyquist_half_weight(
        coeff_cos=np.ones((2, 0)),
        coeff_sin=np.ones((2, 0)),
        modes=ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int)),
        trig=trig,
    )
    assert empty_weight[0].shape == (2, 0)
    weighted_cos, weighted_sin = wout._apply_nyquist_half_weight(
        coeff_cos=coeff,
        coeff_sin=2.0 * coeff,
        modes=ModeTable(m=np.asarray([0, 1, 2], dtype=int), n=np.asarray([0, 1, 0], dtype=int)),
        trig=trig,
    )
    np.testing.assert_allclose(weighted_cos[:, 1:], 0.5)
    np.testing.assert_allclose(weighted_sin[:, 1:], 1.0)

    modes = ModeTable(m=np.asarray([0, 1, 2], dtype=int), n=np.asarray([0, 1, -1], dtype=int))
    reduced = base[:, :nt2, :]
    assert wout._vmec_wrout_nyquist_cos_coeffs(
        f=reduced,
        modes=ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int)),
        trig=trig,
    ).shape == (2, 0)
    with pytest.raises(ValueError, match="Expected f"):
        wout._vmec_wrout_nyquist_cos_coeffs(f=reduced[0], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="smaller"):
        wout._vmec_wrout_nyquist_sin_coeffs(f=reduced[:, : nt2 - 1], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="do not cover"):
        wout._vmec_wrout_nyquist_cos_coeffs(
            f=reduced,
            modes=ModeTable(m=np.asarray([99], dtype=int), n=np.asarray([0], dtype=int)),
            trig=trig,
        )
    cos_coeff = wout._vmec_wrout_nyquist_cos_coeffs(f=reduced, modes=modes, trig=trig)
    sin_coeff = wout._vmec_wrout_nyquist_sin_coeffs(f=reduced, modes=modes, trig=trig)
    assert cos_coeff.shape == (2, 3)
    assert sin_coeff.shape == (2, 3)

    with pytest.raises(ValueError, match="Expected coeff"):
        wout._vmec_wrout_nyquist_synthesis(coeff_c=cos_coeff[0], coeff_s=sin_coeff, modes=modes, trig=trig)
    assert wout._vmec_wrout_nyquist_synthesis(
        coeff_c=np.ones((2, 0)),
        coeff_s=np.ones((2, 0)),
        modes=ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int)),
        trig=trig,
    ).shape == (2, 0, 0)
    synthesized = wout._vmec_wrout_nyquist_synthesis(coeff_c=cos_coeff, coeff_s=sin_coeff, modes=modes, trig=trig)
    assert synthesized.shape == (2, nt2, nzeta)
    loop_sin = wout._vmec_wrout_nyquist_sin_coeffs_loop(f=reduced, modes=modes, trig=trig)
    assert loop_sin.shape == (2, 3)

    assert wout._vmec_jxbforce_cos_coeffs(
        f=reduced,
        modes=ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int)),
        trig=trig,
    ).shape == (2, 0)
    with pytest.raises(ValueError, match="Expected f"):
        wout._vmec_jxbforce_sin_coeffs(f=reduced[0], modes=modes, trig=trig)
    jxb_cos = wout._vmec_jxbforce_cos_coeffs(f=reduced, modes=modes, trig=trig)
    jxb_sin = wout._vmec_jxbforce_sin_coeffs(f=reduced, modes=modes, trig=trig)
    assert jxb_cos.shape == (2, 3)
    assert jxb_sin.shape == (2, 3)

    scaled = wout._vmec_wrout_lasym_bsubuv_output_scale(
        bsubumnc=cos_coeff,
        bsubvmnc=sin_coeff,
        bsubumns=jxb_cos,
        bsubvmns=jxb_sin,
    )
    np.testing.assert_allclose(scaled[0], 2.0 * cos_coeff)

    split_sym, split_asym = wout._vmec_symoutput_split(f=base, trig=trig)
    rev_sym, rev_asym = wout._vmec_symoutput_split(f=base, trig=trig, reversed_sym=True)
    assert split_sym.shape == (2, nt2, nzeta)
    np.testing.assert_allclose(rev_sym, split_asym)
    np.testing.assert_allclose(rev_asym, split_sym)
    with pytest.raises(ValueError, match="Expected f"):
        wout._vmec_symoutput_split(f=base[0], trig=trig)
    with pytest.raises(ValueError, match="smaller"):
        wout._vmec_symforce_apply(f=base[:, : nt2 - 1], trig=trig, kind="ars")

    zero_zeta = np.ones((2, nt2, 0))
    assert wout._vmec_symoutput_split(f=zero_zeta, trig=trig)[0].shape == zero_zeta.shape
    assert wout._vmec_symforce_apply(f=zero_zeta, trig=trig, kind="ars").shape == zero_zeta.shape
    assert wout._vmec_symforce_antisym(f=zero_zeta, trig=trig, kind="ars").shape == zero_zeta.shape

    sym_applied = wout._vmec_symforce_apply(f=base, trig=trig, kind="ars")
    antisym_applied = wout._vmec_symforce_apply(f=base, trig=trig, kind="brs")
    assert sym_applied.shape == base.shape
    assert antisym_applied.shape == base.shape
    with pytest.raises(ValueError, match="unknown kind"):
        wout._vmec_symforce_apply(f=base, trig=trig, kind="bad")

    antisym = wout._vmec_symforce_antisym(f=base, trig=trig, kind="ars", base=2.0 * base)
    antisym_other = wout._vmec_symforce_antisym(f=base, trig=trig, kind="brs")
    assert antisym.shape == base.shape
    assert antisym_other.shape == base.shape
    with pytest.raises(ValueError, match="base shape"):
        wout._vmec_symforce_antisym(f=base, trig=trig, kind="ars", base=base[:, :, :1])
    with pytest.raises(ValueError, match="unknown kind"):
        wout._vmec_symforce_antisym(f=base, trig=trig, kind="bad")

    with pytest.raises(ValueError, match="Expected sym"):
        wout._vmec_symoutput_expand(sym=reduced[0], asym=None, trig=trig)
    with pytest.raises(ValueError, match="sym/asym"):
        wout._vmec_symoutput_expand(sym=reduced, asym=reduced[:, :, :1], trig=trig)
    expanded = wout._vmec_symoutput_expand(sym=split_sym, asym=split_asym, trig=trig)
    assert expanded.shape == base.shape
    short_trig = SimpleNamespace(ntheta1=nt2, ntheta2=nt2, ntheta3=nt2)
    short_expanded = wout._vmec_symoutput_expand(sym=reduced, asym=None, trig=short_trig)
    assert short_expanded.shape == reduced.shape


def test_bsubs_correction_positive_paths_chipf_and_current_profile_branches(monkeypatch) -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=3, nmax=0, lasym=False, cache=False)
    nt2 = int(trig.ntheta2)
    shape = (3, nt2, 1)
    ones = np.ones(shape)
    bsubs = 0.2 + _grid(shape, scale=0.03)

    monkeypatch.setattr(
        wout,
        "_jxbforce_getbsubs_coeffs_lasym_false",
        lambda **_: np.ones((nt2, 1), dtype=float),
    )
    corrected = wout._jxbforce_apply_bsubs_correction_lasym_false(
        bsubu=0.4 * ones,
        bsubv=0.5 * ones,
        bsubs=bsubs.copy(),
        bsubsu=np.zeros(shape),
        bsubsv=np.zeros(shape),
        bsupu=0.3 * ones,
        bsupv=0.2 * ones,
        sqrtg=ones,
        pres=np.asarray([0.0, 0.1, 0.3]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        hs=0.5,
        signgs=1.0,
        trig=trig,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    assert [arr.shape for arr in corrected] == [shape] * 3
    assert not np.allclose(corrected[0], bsubs)

    trig_lasym = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=3, nmax=0, lasym=True, cache=False)
    nt3 = int(trig_lasym.ntheta3)
    lasym_shape = (3, nt3, 1)
    lasym_ones = np.ones(lasym_shape)
    monkeypatch.setattr(
        wout,
        "_jxbforce_getbsubs_coeffs_lasym_true",
        lambda **_: np.ones((int(trig_lasym.ntheta2), 1, 2), dtype=float),
    )
    corrected_lasym = wout._jxbforce_apply_bsubs_correction_lasym_true(
        bsubu=0.4 * lasym_ones,
        bsubv=0.5 * lasym_ones,
        bsubs=0.2 + _grid(lasym_shape, scale=0.03),
        bsubsu=np.zeros(lasym_shape),
        bsubsv=np.zeros(lasym_shape),
        bsupu=0.3 * lasym_ones,
        bsupv=0.2 * lasym_ones,
        sqrtg=lasym_ones,
        pres=np.asarray([0.0, 0.1, 0.3]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        hs=0.5,
        signgs=1.0,
        trig=trig_lasym,
        nfp=1,
        sum_w=lambda arr: float(np.sum(arr)),
    )
    assert [arr.shape for arr in corrected_lasym] == [lasym_shape] * 3
    assert np.any(corrected_lasym[0])

    np.testing.assert_allclose(wout._chipf_from_chips(np.asarray([2.0])), np.asarray([2.0]))
    np.testing.assert_allclose(wout._chipf_from_chips(np.asarray([1.0, 3.0])), np.asarray([3.0, 4.0]))
    np.testing.assert_allclose(wout._chipf_from_chips(np.asarray([0.0, 2.0, 4.0])), np.asarray([1.0, 3.0, 5.0]))
    np.testing.assert_allclose(
        np.asarray(wout._chipf_from_chips(wout.jnp.asarray([0.0, 2.0, 4.0]))),
        np.asarray([1.0, 3.0, 5.0]),
    )

    s_full = np.asarray([0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        np.asarray(wout._icurv_full_mesh_from_indata(indata=_indata(NCURR=0), s_full=s_full, signgs=1)),
        np.zeros_like(s_full),
    )
    np.testing.assert_allclose(
        np.asarray(wout._icurv_full_mesh_from_indata(indata=_indata(NCURR=1, CURTOR=0.0), s_full=s_full, signgs=1)),
        np.zeros_like(s_full),
    )

    def fake_eval_profiles(_indata_obj, s):
        arr = np.asarray(s, dtype=float)
        if arr.shape == (1,) and np.isclose(arr[0], 1.0):
            return {"current": np.asarray([2.0])}
        return {"current": np.linspace(0.0, 2.0, arr.size)}

    monkeypatch.setattr(profiles_module, "eval_profiles", fake_eval_profiles)
    icurv = np.asarray(
        wout._icurv_full_mesh_from_indata(indata=_indata(NCURR=1, CURTOR=5.0), s_full=s_full, signgs=-1)
    )
    assert icurv.shape == s_full.shape
    assert icurv[0] == 0.0
    assert np.any(icurv[1:] != 0.0)

    def fake_bad_profiles(_indata_obj, s):
        arr = np.asarray(s, dtype=float)
        if arr.shape == (1,) and np.isclose(arr[0], 1.0):
            return {"current": np.asarray([0.0])}
        return {"current": np.asarray([1.0])}

    monkeypatch.setattr(profiles_module, "eval_profiles", fake_bad_profiles)
    bad = np.asarray(wout._icurv_full_mesh_from_indata(indata=_indata(NCURR=1, CURTOR=5.0), s_full=s_full, signgs=1))
    np.testing.assert_allclose(bad, np.zeros_like(s_full))

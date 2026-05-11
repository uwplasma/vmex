from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.vmec_forces import (
    VmecRZResidualCoeffs,
    _avg_forward_half,
    _avg_forward_half_to_int,
    _diff_forward_half,
    _diff_forward_half_noavg,
    _parse_iter_list,
    _pshalf_from_s,
    _select_parity_coeffs,
    _sum_forward_half,
    _with_axis_zero,
    rz_residual_scalars_like_vmec,
)


def test_force_helper_radial_stencils_match_vmec_forward_rules() -> None:
    a = np.asarray([2.0, 5.0, 11.0])
    b = np.asarray([1.0, 3.0, 7.0])

    np.testing.assert_allclose(np.asarray(_with_axis_zero(a)), [0.0, 5.0, 11.0])
    np.testing.assert_allclose(np.asarray(_with_axis_zero(np.asarray([]))), [])
    np.testing.assert_allclose(np.asarray(_avg_forward_half_to_int(a)), [3.5, 8.0, 5.5])
    np.testing.assert_allclose(np.asarray(_sum_forward_half(a)), [7.0, 16.0, 11.0])
    np.testing.assert_allclose(np.asarray(_diff_forward_half(a, b)), [5.0, 11.0, -7.5])
    np.testing.assert_allclose(np.asarray(_diff_forward_half_noavg(a)), [3.0, 6.0, -11.0])
    np.testing.assert_allclose(np.asarray(_avg_forward_half(a)), [3.5, 8.0, 5.5])

    one = np.asarray([4.0])
    np.testing.assert_allclose(np.asarray(_avg_forward_half_to_int(one)), one)
    np.testing.assert_allclose(np.asarray(_sum_forward_half(one)), one)
    np.testing.assert_allclose(np.asarray(_diff_forward_half(one, one)), one)
    np.testing.assert_allclose(np.asarray(_diff_forward_half_noavg(one)), one)
    np.testing.assert_allclose(np.asarray(_avg_forward_half(one)), one)


def test_force_helper_pshalf_and_iter_list_edges() -> None:
    s = np.asarray([0.0, 0.25, 1.0])
    np.testing.assert_allclose(np.asarray(_pshalf_from_s(s)), np.sqrt([0.125, 0.125, 0.625]))
    np.testing.assert_allclose(np.asarray(_pshalf_from_s(np.asarray([-1.0]))), [0.0])

    assert _parse_iter_list("") is None
    assert _parse_iter_list(" 1, 3-5, 9 , bad, 8-6 ") == {1, 3, 4, 5, 6, 7, 8, 9}
    assert _parse_iter_list("bad, nope") is None


def test_force_parity_selection_uses_even_m_channel() -> None:
    coeff_even = np.asarray([[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0]])
    coeff_odd = -coeff_even
    m = np.asarray([0, 1, 2, 3])
    selected = np.asarray(_select_parity_coeffs(coeff_even=coeff_even, coeff_odd=coeff_odd, m=m))

    expected = coeff_even.copy()
    expected[:, [1, 3]] = coeff_odd[:, [1, 3]]
    np.testing.assert_allclose(selected, expected)


def test_rz_residual_scalars_follow_documented_normalization() -> None:
    coeffs = VmecRZResidualCoeffs(
        gcr_cos=np.asarray([[100.0, 100.0], [1.0, 2.0], [3.0, 4.0]]),
        gcr_sin=np.asarray([[100.0, 100.0], [0.5, 1.5], [2.5, 3.5]]),
        gcz_cos=np.asarray([[100.0, 100.0], [2.0, 3.0], [4.0, 5.0]]),
        gcz_sin=np.asarray([[100.0, 100.0], [1.0, 2.0], [3.0, 4.0]]),
    )
    bc = SimpleNamespace(
        jac=SimpleNamespace(r12=np.asarray([[[9.0]], [[2.0]], [[3.0]]])),
        gij_b_uu=np.asarray([[[99.0]], [[5.0]], [[7.0]]]),
    )
    wout = SimpleNamespace(volume_p=8.0 * np.pi**2, wb=6.0, wp=2.0)
    s = np.asarray([0.0, 0.5, 1.0])

    scalars = rz_residual_scalars_like_vmec(coeffs, bc=bc, wout=wout, s=s)
    vol_norm = wout.volume_p / (4.0 * np.pi**2)
    r2 = max(wout.wb, wout.wp) / vol_norm
    avg_guu_r2 = np.mean(np.asarray([5.0 * 2.0**2, 7.0 * 3.0**2]))
    gnorm = 0.25 / (avg_guu_r2 * r2 * r2)
    expected_r = gnorm * np.sum(coeffs.gcr_cos[1:] ** 2 + coeffs.gcr_sin[1:] ** 2)
    expected_z = gnorm * np.sum(coeffs.gcz_cos[1:] ** 2 + coeffs.gcz_sin[1:] ** 2)
    assert scalars.fsqr_like == expected_r
    assert scalars.fsqz_like == expected_z

    short = rz_residual_scalars_like_vmec(coeffs, bc=bc, wout=wout, s=np.asarray([0.0]))
    assert np.isnan(short.fsqr_like)
    assert np.isnan(short.fsqz_like)

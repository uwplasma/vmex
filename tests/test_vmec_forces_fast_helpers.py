from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import vmec_jax.kernels.forces as vf
import vmec_jax.kernels.bcovar as vb
from vmec_jax.kernels.forces import (
    VmecRZResidualCoeffs,
    _add_edge_row,
    _avg_forward_half,
    _avg_forward_half_to_int_or_zero,
    _avg_forward_half_to_int,
    _diff_forward_half,
    _diff_forward_half_noavg,
    _odd_force_radial_updates,
    _parse_iter_list,
    _pshalf_from_s,
    _scale_lambda_full_mesh,
    _scale_lambda_full_mesh_zero_axis,
    _select_parity_coeffs,
    _sum_forward_half,
    _with_axis_zero,
    rz_residual_coeffs_from_kernels,
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


def test_force_refactor_helpers_cover_edge_and_single_surface_branches() -> None:
    empty = np.zeros((0, 2), dtype=float)
    one = np.asarray([[2.0, -3.0]], dtype=float)
    three = np.arange(6.0, dtype=float).reshape(3, 2)

    np.testing.assert_allclose(np.asarray(_add_edge_row(empty, np.asarray([1.0, 2.0]))), empty)
    np.testing.assert_allclose(np.asarray(_add_edge_row(three, np.asarray([10.0, 20.0]))), [[0, 1], [2, 3], [14, 25]])

    np.testing.assert_allclose(np.asarray(_avg_forward_half_to_int_or_zero(one)), np.zeros_like(one))
    np.testing.assert_allclose(np.asarray(_avg_forward_half_to_int_or_zero(three)), [[1, 2], [3, 4], [2, 2.5]])

    np.testing.assert_allclose(np.asarray(_scale_lambda_full_mesh(one, 7.0)), one)
    np.testing.assert_allclose(np.asarray(_scale_lambda_full_mesh(three, 2.0)), [[0, 1], [-4, -6], [-8, -10]])
    np.testing.assert_allclose(np.asarray(_scale_lambda_full_mesh_zero_axis(one, 7.0)), np.zeros_like(one))
    np.testing.assert_allclose(
        np.asarray(_scale_lambda_full_mesh_zero_axis(three, 2.0)),
        [[0, 0], [-4, -6], [-8, -10]],
    )

    armn_o = np.asarray([[1.0], [3.0], [6.0]])
    azmn_o = np.asarray([[2.0], [5.0], [9.0]])
    brmn_o = np.asarray([[4.0], [8.0], [12.0]])
    bzmn_o = np.asarray([[10.0], [14.0], [20.0]])
    lu_o = np.asarray([[1.0], [2.0], [4.0]])
    pzu_0 = np.asarray([[0.5], [0.25], [0.125]])
    pru_0 = np.asarray([[0.2], [0.3], [0.4]])
    bsqr_s = np.asarray([[2.0], [4.0], [8.0]])
    lv_es = np.asarray([[1.0], [3.0], [5.0]])

    empty_out = _odd_force_radial_updates(
        armn_o=empty,
        azmn_o=empty,
        brmn_o=empty,
        bzmn_o=empty,
        lu_o=empty,
        pzu_0=empty,
        pru_0=empty,
        bsqr_s=empty,
        lv_es=empty,
    )
    assert all(out_i is in_i for out_i, in_i in zip(empty_out, (empty, empty, empty, empty, empty), strict=True))

    out = _odd_force_radial_updates(
        armn_o=armn_o,
        azmn_o=azmn_o,
        brmn_o=brmn_o,
        bzmn_o=bzmn_o,
        lu_o=lu_o,
        pzu_0=pzu_0,
        pru_0=pru_0,
        bsqr_s=bsqr_s,
        lv_es=lv_es,
    )
    np.testing.assert_allclose(np.asarray(out[0]), [[3.0], [6.0], [-4.5]])
    np.testing.assert_allclose(np.asarray(out[1]), [[3.4], [5.2], [-5.8]])
    np.testing.assert_allclose(np.asarray(out[2]), [[6.0], [10.0], [6.0]])
    np.testing.assert_allclose(np.asarray(out[3]), [[12.0], [17.0], [10.0]])
    np.testing.assert_allclose(np.asarray(out[4]), [[3.0], [6.0], [4.0]])

    single_out = _odd_force_radial_updates(
        armn_o=armn_o[:1],
        azmn_o=azmn_o[:1],
        brmn_o=brmn_o[:1],
        bzmn_o=bzmn_o[:1],
        lu_o=lu_o[:1],
        pzu_0=pzu_0[:1],
        pru_0=pru_0[:1],
        bsqr_s=bsqr_s[:1],
        lv_es=lv_es[:1],
    )
    np.testing.assert_allclose(np.asarray(single_out[0]), [[-1.5]])
    np.testing.assert_allclose(np.asarray(single_out[4]), lu_o[:1])


def test_bcovar_refactor_radial_helpers_cover_empty_and_single_surface_branches() -> None:
    empty = np.zeros((0, 2), dtype=float)
    one = np.asarray([[5.0, -1.0]], dtype=float)
    three = np.arange(6.0, dtype=float).reshape(3, 2)

    np.testing.assert_allclose(np.asarray(vb._replace_axis_row(empty, np.asarray([9.0, 9.0]))), empty)
    np.testing.assert_allclose(np.asarray(vb._replace_axis_row(three, np.asarray([9.0, 8.0]))), [[9, 8], [2, 3], [4, 5]])
    np.testing.assert_allclose(np.asarray(vb._replace_edge_row(empty, np.asarray([9.0, 9.0]))), empty)
    np.testing.assert_allclose(np.asarray(vb._replace_edge_row(three, np.asarray([9.0, 8.0]))), [[0, 1], [2, 3], [9, 8]])
    np.testing.assert_allclose(np.asarray(vb._with_axis_zero(empty)), empty)
    np.testing.assert_allclose(np.asarray(vb._with_axis_zero(three)), [[0, 0], [2, 3], [4, 5]])
    np.testing.assert_allclose(np.asarray(vb._prepend_axis_zero(three[1:], three)), [[0, 0], [2, 3], [4, 5]])
    np.testing.assert_allclose(np.asarray(vb._avg_forward_half_to_int_or_zero(one)), np.zeros_like(one))
    np.testing.assert_allclose(np.asarray(vb._avg_forward_half_to_int_or_zero(three)), [[1, 2], [3, 4], [2, 2.5]])
    np.testing.assert_allclose(np.asarray(vb._scale_lambda_full_mesh(one, 3.0)), one)
    np.testing.assert_allclose(np.asarray(vb._scale_lambda_full_mesh(three, 3.0)), [[0, 1], [-6, -9], [-12, -15]])


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


def test_rz_residual_coeffs_use_parity_and_helical_derivatives(monkeypatch) -> None:
    ns = 2
    k_modes = 3

    def coeff(start: float) -> np.ndarray:
        return start + np.arange(ns * k_modes, dtype=float).reshape(ns, k_modes)

    projected = iter(
        [
            (coeff(10.0), coeff(20.0)),  # armn even
            (coeff(30.0), coeff(40.0)),  # armn odd
            (coeff(50.0), coeff(60.0)),  # brmn even
            (coeff(70.0), coeff(80.0)),  # brmn odd
            (coeff(90.0), coeff(100.0)),  # crmn even
            (coeff(110.0), coeff(120.0)),  # crmn odd
            (coeff(130.0), coeff(140.0)),  # azmn even
            (coeff(150.0), coeff(160.0)),  # azmn odd
            (coeff(170.0), coeff(180.0)),  # bzmn even
            (coeff(190.0), coeff(200.0)),  # bzmn odd
            (coeff(210.0), coeff(220.0)),  # czmn even
            (coeff(230.0), coeff(240.0)),  # czmn odd
        ]
    )

    def fake_project_to_modes(_field, _basis):
        return next(projected)

    monkeypatch.setattr(vf, "project_to_modes", fake_project_to_modes)
    shape = (ns, 1, 1)
    zeros = np.zeros(shape)
    kernels = SimpleNamespace(
        armn_e=zeros,
        armn_o=zeros,
        brmn_e=zeros,
        brmn_o=zeros,
        crmn_e=zeros,
        crmn_o=zeros,
        azmn_e=zeros,
        azmn_o=zeros,
        bzmn_e=zeros,
        bzmn_o=zeros,
        czmn_e=zeros,
        czmn_o=zeros,
    )
    static = SimpleNamespace(
        modes=SimpleNamespace(m=np.asarray([0, 1, 2]), n=np.asarray([0, 1, -1])),
        grid=SimpleNamespace(nfp=2),
        basis=object(),
    )

    out = rz_residual_coeffs_from_kernels(kernels, static=static)

    m = static.modes.m[None, :]
    n_phys = (static.modes.n * static.grid.nfp)[None, :]

    def select(even, odd):
        return np.where((static.modes.m % 2)[None, :] == 0, even, odd)

    aR_c, aR_s = select(coeff(10.0), coeff(30.0)), select(coeff(20.0), coeff(40.0))
    bR_c, bR_s = select(coeff(50.0), coeff(70.0)), select(coeff(60.0), coeff(80.0))
    cR_c, cR_s = select(coeff(90.0), coeff(110.0)), select(coeff(100.0), coeff(120.0))
    aZ_c, aZ_s = select(coeff(130.0), coeff(150.0)), select(coeff(140.0), coeff(160.0))
    bZ_c, bZ_s = select(coeff(170.0), coeff(190.0)), select(coeff(180.0), coeff(200.0))
    cZ_c, cZ_s = select(coeff(210.0), coeff(230.0)), select(coeff(220.0), coeff(240.0))

    np.testing.assert_allclose(np.asarray(out.gcr_cos), aR_c - m * bR_s - n_phys * cR_s)
    np.testing.assert_allclose(np.asarray(out.gcr_sin), aR_s + m * bR_c + n_phys * cR_c)
    np.testing.assert_allclose(np.asarray(out.gcz_cos), aZ_c - m * bZ_s - n_phys * cZ_s)
    np.testing.assert_allclose(np.asarray(out.gcz_sin), aZ_s + m * bZ_c + n_phys * cZ_c)


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

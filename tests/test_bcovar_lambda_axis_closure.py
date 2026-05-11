from __future__ import annotations

import numpy as np

from vmec_jax.vmec_bcovar import (
    _apply_vmec_lambda_axis_closure,
    _half_mesh_from_even_odd,
    _metric_cross_even_odd,
    _metric_even_odd,
    _pshalf_from_s,
)


def test_pshalf_and_half_mesh_staggering_match_vmec_rules():
    s = np.array([0.0, 0.25, 1.0])
    pshalf = np.asarray(_pshalf_from_s(s))
    expected_pshalf = np.sqrt([0.125, 0.125, 0.625])
    np.testing.assert_allclose(pshalf, expected_pshalf)

    even = np.array([2.0, 4.0, 8.0])[:, None, None]
    odd = np.array([10.0, 20.0, 40.0])[:, None, None]
    half = np.asarray(_half_mesh_from_even_odd(even, odd, s=s))[:, 0, 0]

    expected_first = 0.5 * (4.0 + 2.0 + expected_pshalf[1] * (20.0 + 10.0))
    expected_second = 0.5 * (8.0 + 4.0 + expected_pshalf[2] * (40.0 + 20.0))
    np.testing.assert_allclose(half, [expected_first, expected_first, expected_second])

    one_surface_even = np.array([7.0])[:, None, None]
    one_surface_odd = np.array([11.0])[:, None, None]
    np.testing.assert_allclose(
        np.asarray(_half_mesh_from_even_odd(one_surface_even, one_surface_odd, s=np.array([0.5]))),
        one_surface_even,
    )
    np.testing.assert_allclose(np.asarray(_pshalf_from_s(np.array([-0.25]))), [0.0])


def test_metric_even_odd_decompositions_match_explicit_products():
    s = np.array([0.0, 0.25])
    a0 = np.array([[[2.0]], [[3.0]]])
    a1 = np.array([[[5.0]], [[7.0]]])
    b0 = np.array([[[11.0]], [[13.0]]])
    b1 = np.array([[[17.0]], [[19.0]]])

    even, odd = _metric_even_odd(a0=a0, a1=a1, b0=b0, b1=b1, s=s)
    expected_even = a0 * a0 + b0 * b0 + s[:, None, None] * (a1 * a1 + b1 * b1)
    expected_odd = 2.0 * (a0 * a1 + b0 * b1)
    np.testing.assert_allclose(np.asarray(even), expected_even)
    np.testing.assert_allclose(np.asarray(odd), expected_odd)

    cross_even, cross_odd = _metric_cross_even_odd(a0=a0, a1=a1, b0=b0, b1=b1, s=s)
    np.testing.assert_allclose(np.asarray(cross_even), a0 * b0 + s[:, None, None] * (a1 * b1))
    np.testing.assert_allclose(np.asarray(cross_odd), a0 * b1 + a1 * b0)


def test_lambda_axis_closure_copies_m0_npos_modes():
    lsin = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
        ],
        dtype=float,
    )
    m_modes = np.array([0, 0, 1, 0], dtype=int)
    n_modes = np.array([0, 1, 1, 2], dtype=int)

    out = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            lthreed=True,
            ntor=2,
        )
    )

    np.testing.assert_allclose(out[0, 0], lsin[0, 0])  # n=0 unchanged
    np.testing.assert_allclose(out[0, 1], lsin[1, 1])  # m=0,n>0 copied
    np.testing.assert_allclose(out[0, 2], lsin[0, 2])  # m!=0 unchanged
    np.testing.assert_allclose(out[0, 3], lsin[1, 3])  # m=0,n>0 copied
    np.testing.assert_allclose(out[1:], lsin[1:])  # interior unchanged


def test_lambda_axis_closure_disabled_for_axisymmetric_or_ntor_zero():
    lsin = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=float)
    m_modes = np.array([0, 0], dtype=int)
    n_modes = np.array([0, 1], dtype=int)

    out_axis = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            lthreed=False,
            ntor=1,
        )
    )
    out_ntor0 = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            lthreed=True,
            ntor=0,
        )
    )
    np.testing.assert_allclose(out_axis, lsin)
    np.testing.assert_allclose(out_ntor0, lsin)


def test_lambda_axis_closure_noop_when_ns_one():
    lsin = np.array([[1.0, 2.0, 3.0]], dtype=float)
    m_modes = np.array([0, 0, 1], dtype=int)
    n_modes = np.array([0, 2, 1], dtype=int)

    out = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            lthreed=True,
            ntor=2,
        )
    )
    np.testing.assert_allclose(out, lsin)

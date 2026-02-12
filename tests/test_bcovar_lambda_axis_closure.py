from __future__ import annotations

import numpy as np

from vmec_jax.vmec_bcovar import _apply_vmec_lambda_axis_closure


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

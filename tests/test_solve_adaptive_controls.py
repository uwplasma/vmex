from __future__ import annotations

import numpy as np

from vmec_jax.solve import (
    _resolve_cg_tol,
    _resolve_grad_tol,
    _resolve_lbfgs_curvature_tol,
    _resolve_lm_damping,
)


def test_resolve_grad_tol_scales_with_initial_gradient():
    tol_small = _resolve_grad_tol(None, grad_rms0=1.0e-6, dtype=np.float64)
    tol_large = _resolve_grad_tol(None, grad_rms0=1.0, dtype=np.float64)

    assert tol_small > 0.0
    assert tol_large > tol_small
    assert np.isclose(tol_large / tol_small, 1.0e6, rtol=1.0e-12)


def test_resolve_lbfgs_curvature_tol_tracks_vector_scale():
    s1 = np.array([1.0, 2.0, 3.0])
    y1 = np.array([2.0, 4.0, 6.0])
    s2 = 10.0 * s1
    y2 = 10.0 * y1

    tol1 = _resolve_lbfgs_curvature_tol(s1, y1)
    tol2 = _resolve_lbfgs_curvature_tol(s2, y2)

    assert tol1 > 0.0
    assert np.isclose(tol2 / tol1, 100.0, rtol=1.0e-12)


def test_resolve_cg_tol_tightens_with_progress():
    tol0 = _resolve_cg_tol(None, current_obj=1.0, initial_obj=1.0, target_obj=1.0e-12, dtype=np.float64)
    tol1 = _resolve_cg_tol(None, current_obj=1.0e-6, initial_obj=1.0, target_obj=1.0e-12, dtype=np.float64)

    assert 0.0 < tol1 < tol0 < 1.0


def test_resolve_lm_damping_uses_curvature_scale():
    d1 = _resolve_lm_damping(None, curvature_scale=1.0, dtype=np.float64)
    d2 = _resolve_lm_damping(None, curvature_scale=1.0e6, dtype=np.float64)

    assert d1 > 0.0
    assert d2 > d1

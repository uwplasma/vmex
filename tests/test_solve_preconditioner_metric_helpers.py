from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.solve_preconditioner_helpers import (
    metric_surface_precond_from_bcovar_jax,
    metric_surface_precond_from_bcovar_np,
    metric_surface_precond_scales_jax,
    metric_surface_precond_scales_np,
)


def _bcovar_payload(dtype=np.float64):
    guu = np.array(
        [
            [[2.0, 1.0], [1.5, 0.5]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0e-12, 1.0e-12], [1.0e-12, 1.0e-12]],
        ],
        dtype=dtype,
    )
    return SimpleNamespace(
        guu=guu,
        jac=SimpleNamespace(r12=np.ones_like(guu)),
        bsubu=np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0e12, 1.0e12], [1.0e12, 1.0e12]],
            ],
            dtype=dtype,
        ),
        bsubv=np.zeros_like(guu),
    )


def _wint_from_trig(_trig, *, nzeta: int):
    assert nzeta == 2
    return np.array([[1.0, 2.0], [3.0, 4.0]])


def test_metric_surface_precond_from_bcovar_np_matches_scale_kernel():
    bc = _bcovar_payload()
    trig = SimpleNamespace(name="trig")

    rz, lam = metric_surface_precond_from_bcovar_np(
        bc=bc,
        trig=trig,
        wint_from_trig_func=_wint_from_trig,
    )
    expected_rz, expected_lam = metric_surface_precond_scales_np(
        guu=bc.guu,
        r12=bc.jac.r12,
        bsubu=bc.bsubu,
        bsubv=bc.bsubv,
        w_ang=_wint_from_trig(trig, nzeta=2),
    )

    np.testing.assert_allclose(rz, expected_rz)
    np.testing.assert_allclose(lam, expected_lam)


def test_metric_surface_precond_from_bcovar_jax_matches_scale_kernel():
    bc_np = _bcovar_payload()
    bc = SimpleNamespace(
        guu=jnp.asarray(bc_np.guu),
        jac=SimpleNamespace(r12=jnp.asarray(bc_np.jac.r12)),
        bsubu=jnp.asarray(bc_np.bsubu),
        bsubv=jnp.asarray(bc_np.bsubv),
    )
    trig = SimpleNamespace(name="trig")

    rz, lam = metric_surface_precond_from_bcovar_jax(
        bc=bc,
        trig=trig,
        wint_from_trig_func=_wint_from_trig,
    )
    expected_rz, expected_lam = metric_surface_precond_scales_jax(
        guu=bc.guu,
        r12=bc.jac.r12,
        bsubu=bc.bsubu,
        bsubv=bc.bsubv,
        w_ang=_wint_from_trig(trig, nzeta=2),
    )

    np.testing.assert_allclose(np.asarray(rz), np.asarray(expected_rz))
    np.testing.assert_allclose(np.asarray(lam), np.asarray(expected_lam))


def test_metric_surface_precond_from_bcovar_np_allows_injected_scale_kernel():
    bc = _bcovar_payload(dtype=np.float32)
    captured = {}

    def scale_kernel(**kwargs):
        captured.update(kwargs)
        return np.array([1.0]), np.array([2.0])

    rz, lam = metric_surface_precond_from_bcovar_np(
        bc=bc,
        trig=SimpleNamespace(),
        wint_from_trig_func=_wint_from_trig,
        scales_func=scale_kernel,
    )

    np.testing.assert_allclose(rz, [1.0])
    np.testing.assert_allclose(lam, [2.0])
    assert captured["guu"].dtype == np.float32
    assert captured["w_ang"].dtype == np.float32


def test_metric_surface_precond_from_bcovar_jax_allows_injected_scale_kernel():
    bc = _bcovar_payload(dtype=np.float32)
    captured = {}

    def scale_kernel(**kwargs):
        captured.update(kwargs)
        return jnp.asarray([3.0]), jnp.asarray([4.0])

    rz, lam = metric_surface_precond_from_bcovar_jax(
        bc=SimpleNamespace(
            guu=jnp.asarray(bc.guu),
            jac=SimpleNamespace(r12=jnp.asarray(bc.jac.r12)),
            bsubu=jnp.asarray(bc.bsubu),
            bsubv=jnp.asarray(bc.bsubv),
        ),
        trig=SimpleNamespace(),
        wint_from_trig_func=_wint_from_trig,
        scales_func=scale_kernel,
    )

    np.testing.assert_allclose(np.asarray(rz), [3.0])
    np.testing.assert_allclose(np.asarray(lam), [4.0])
    assert captured["guu"].dtype == jnp.float32
    assert captured["w_ang"].dtype == jnp.float32

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.solve_preconditioner_helpers import (
    PreconditionerCacheDecision,
    metric_surface_precond_from_bcovar_jax,
    metric_surface_precond_from_bcovar_np,
    metric_surface_precond_scales_jax,
    metric_surface_precond_scales_np,
    resolve_preconditioner_cache_decision,
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


def _precond_decision(**overrides):
    defaults = dict(
        precond_traced=False,
        vmec2000_cache_valid=True,
        need_bcovar_update=False,
        precond_cache_seeded_from_bcovar_update=False,
        need_lam_prec=False,
        need_lamcal=False,
        cache_prec_lam_prec=object(),
        cache_prec_rz_mats=object(),
        cache_prec_rz_jmax=4,
        precond_expected_jmax=4,
        can_reassemble_func=lambda _mats: True,
    )
    defaults.update(overrides)
    return resolve_preconditioner_cache_decision(**defaults)


def test_resolve_preconditioner_cache_decision_allows_clean_cache_hit() -> None:
    got = _precond_decision()

    assert isinstance(got, PreconditionerCacheDecision)
    assert got.need_prec_reassemble is False
    assert got.can_reuse_bcovar_seeded_precond is False
    assert got.need_prec_refresh is False


def test_resolve_preconditioner_cache_decision_refreshes_traced_or_missing_cache() -> None:
    assert _precond_decision(precond_traced=True).need_prec_refresh is True
    assert _precond_decision(cache_prec_lam_prec=None).need_prec_refresh is True
    assert _precond_decision(cache_prec_rz_mats=None).need_prec_refresh is True
    assert _precond_decision(cache_prec_rz_jmax=None).need_prec_refresh is True
    assert _precond_decision(vmec2000_cache_valid=False).need_prec_refresh is True


def test_resolve_preconditioner_cache_decision_reuses_seeded_bcovar_update_cache() -> None:
    got = _precond_decision(
        need_bcovar_update=True,
        precond_cache_seeded_from_bcovar_update=True,
    )

    assert got.can_reuse_bcovar_seeded_precond is True
    assert got.need_prec_refresh is False

    blocked_by_debug_dump = _precond_decision(
        need_bcovar_update=True,
        precond_cache_seeded_from_bcovar_update=True,
        need_lam_prec=True,
    )
    assert blocked_by_debug_dump.can_reuse_bcovar_seeded_precond is False
    assert blocked_by_debug_dump.need_prec_refresh is True


def test_resolve_preconditioner_cache_decision_distinguishes_reassemble_from_refresh() -> None:
    reassemble = _precond_decision(cache_prec_rz_jmax=3, precond_expected_jmax=4)

    assert reassemble.need_prec_reassemble is True
    assert reassemble.need_prec_refresh is False

    refresh = _precond_decision(
        cache_prec_rz_jmax=3,
        precond_expected_jmax=4,
        can_reassemble_func=lambda _mats: False,
    )
    assert refresh.need_prec_reassemble is False
    assert refresh.need_prec_refresh is True

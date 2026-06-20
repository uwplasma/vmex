from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.solvers.fixed_boundary.preconditioning.operators import (
    LambdaPreconditionerOutputs,
    PreconditionerCacheSnapshot,
    PreconditionerCacheDecision,
    PreconditionerCacheState,
    PreconditionerCacheUpdate,
    empty_preconditioner_cache_snapshot,
    lambda_preconditioner_outputs,
    metric_surface_precond_from_bcovar_jax,
    metric_surface_precond_from_bcovar_np,
    metric_surface_precond_scales_jax,
    metric_surface_precond_scales_np,
    resolve_preconditioner_cache_decision,
    update_preconditioner_cache,
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


def test_lambda_preconditioner_outputs_requests_only_needed_payloads() -> None:
    calls = []

    def fake_lambda_preconditioner(_bc, *, return_faclam=False, return_debug=False):
        calls.append((bool(return_faclam), bool(return_debug)))
        if return_faclam and return_debug:
            return "lam", "faclam", "debug"
        if return_debug:
            return "lam", "debug"
        if return_faclam:
            return "lam", "faclam"
        return "lam"

    cases = [
        (False, False, LambdaPreconditionerOutputs("lam", None, None), (False, False)),
        (True, False, LambdaPreconditionerOutputs("lam", "faclam", None), (True, False)),
        (False, True, LambdaPreconditionerOutputs("lam", None, "debug"), (False, True)),
        (True, True, LambdaPreconditionerOutputs("lam", "faclam", "debug"), (True, True)),
    ]
    for need_lam_prec, need_lamcal, expected, expected_call in cases:
        got = lambda_preconditioner_outputs(
            object(),
            need_lam_prec=need_lam_prec,
            need_lamcal=need_lamcal,
            lambda_preconditioner_func=fake_lambda_preconditioner,
        )
        assert got == expected
        assert calls[-1] == expected_call


def test_empty_preconditioner_cache_snapshot_matches_iteration_tuple_order() -> None:
    got = empty_preconditioner_cache_snapshot()

    assert isinstance(got, PreconditionerCacheSnapshot)
    assert tuple(got) == (False,) + (None,) * 12
    assert got.valid is False
    assert got.prec_rz_jmax is None
    assert got.prec_lam_debug is None


def test_preconditioner_cache_state_round_trips_legacy_resume_payload() -> None:
    cache = PreconditionerCacheState()
    cache.update_from_resume_state(
        {
            "vmec2000_cache_valid": 1,
            "cache_precond_diag": ("diag",),
            "cache_tcon": "tcon",
            "cache_norms": "norms",
            "cache_rz_scale": "rz-scale",
            "cache_l_scale": "l-scale",
            "cache_rz_norm": 2.0,
            "cache_f_norm1": 0.5,
            "cache_prec_rz_mats": "mats",
            "cache_prec_rz_jmax": 7,
            "cache_prec_lam_prec": "lam",
            "cache_prec_faclam": "faclam",
            "cache_prec_lam_debug": "debug",
        }
    )

    payload = cache.legacy_resume_payload()

    assert payload["vmec2000_cache_valid"] is True
    assert payload["cache_precond_diag"] == ("diag",)
    assert payload["cache_tcon"] == "tcon"
    assert payload["cache_norms"] == "norms"
    assert payload["cache_rz_scale"] == "rz-scale"
    assert payload["cache_l_scale"] == "l-scale"
    assert payload["cache_rz_norm"] == 2.0
    assert payload["cache_f_norm1"] == 0.5
    assert payload["cache_prec_rz_mats"] == "mats"
    assert payload["cache_prec_rz_jmax"] == 7
    assert payload["cache_prec_lam_prec"] == "lam"
    assert payload["cache_prec_faclam"] == "faclam"
    assert payload["cache_prec_lam_debug"] == "debug"

    cache.clear()
    cleared = cache.legacy_resume_payload()
    assert cleared["vmec2000_cache_valid"] is False
    assert all(cleared[key] is None for key in cleared if key != "vmec2000_cache_valid")


def _cache_update_inputs(**overrides):
    calls = []

    def lambda_preconditioner(_bc, *, return_faclam=False, return_debug=False):
        calls.append(("lambda", bool(return_faclam), bool(return_debug)))
        if return_faclam and return_debug:
            return "lam-new", "faclam-new", "debug-new"
        if return_debug:
            return "lam-new", "debug-new"
        if return_faclam:
            return "lam-new", "faclam-new"
        return "lam-new"

    def matrices(**kwargs):
        calls.append(("matrices", kwargs["jmax_override"], kwargs["use_precomputed"], kwargs["use_lax_tridi"]))
        return "mats-new", 0, 7

    def reassemble(**kwargs):
        calls.append(("reassemble", kwargs["jmax_override"]))
        return "mats-reassembled", 0, 9

    defaults = dict(
        bc="bc",
        k=SimpleNamespace(bc="bc"),
        cfg=SimpleNamespace(name="cfg"),
        precond_traced=False,
        vmec2000_cache_valid=True,
        need_bcovar_update=False,
        precond_cache_seeded_from_bcovar_update=False,
        need_lam_prec=False,
        need_lamcal=False,
        cache_prec_lam_prec="lam-cache",
        cache_prec_faclam="faclam-cache",
        cache_prec_lam_debug="debug-cache",
        cache_prec_rz_mats="mats-cache",
        cache_prec_rz_jmax=4,
        precond_expected_jmax=4,
        precond_jmax_override=11,
        preconditioner_use_precomputed_tridi=True,
        preconditioner_use_lax_tridi=False,
        lambda_preconditioner_func=lambda_preconditioner,
        rz_preconditioner_matrices_func=matrices,
        rz_preconditioner_matrices_reassemble_func=reassemble,
        can_reassemble_func=lambda _mats: True,
    )
    defaults.update(overrides)
    return defaults, calls


def test_update_preconditioner_cache_reuses_clean_cache_hit() -> None:
    kwargs, calls = _cache_update_inputs(need_lam_prec=True, need_lamcal=True)

    got = update_preconditioner_cache(**kwargs)

    assert isinstance(got, PreconditionerCacheUpdate)
    assert got.decision.need_prec_refresh is False
    assert got.decision.need_prec_reassemble is False
    assert got.lam_prec == "lam-cache"
    assert got.faclam_dump == "faclam-cache"
    assert got.lam_debug == "debug-cache"
    assert got.mats == "mats-cache"
    assert got.jmax == 4
    assert calls == []


def test_update_preconditioner_cache_refreshes_missing_cache_and_updates_debug_payloads() -> None:
    kwargs, calls = _cache_update_inputs(
        cache_prec_lam_prec=None,
        need_lam_prec=True,
        need_lamcal=True,
    )

    got = update_preconditioner_cache(**kwargs)

    assert got.decision.need_prec_refresh is True
    assert got.lam_prec == "lam-new"
    assert got.faclam_dump == "faclam-new"
    assert got.lam_debug == "debug-new"
    assert got.mats == "mats-new"
    assert got.cache_prec_rz_jmax == 7
    assert calls == [("lambda", True, True), ("matrices", 11, True, False)]


def test_update_preconditioner_cache_reassembles_jmax_mismatch_without_refresh() -> None:
    kwargs, calls = _cache_update_inputs(cache_prec_rz_jmax=3, precond_expected_jmax=9)

    got = update_preconditioner_cache(**kwargs)

    assert got.decision.need_prec_refresh is False
    assert got.decision.need_prec_reassemble is True
    assert got.lam_prec == "lam-cache"
    assert got.faclam_dump is None
    assert got.lam_debug is None
    assert got.mats == "mats-reassembled"
    assert got.cache_prec_rz_jmax == 9
    assert calls == [("reassemble", 11)]

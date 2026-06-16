from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.scan.payload import (
    ScanForceBlocks,
    ScanForcePayload,
    ScanStepFields,
    build_current_preconditioned_scan_payload,
    build_initial_preconditioner_cache,
    build_restart_preconditioned_scan_payload,
    build_scan_force_payload,
    build_scan_step_fields,
    current_scan_payload,
    mask_scan_restart_force_payload,
    restart_scan_payload,
    select_scan_force_payload,
    select_scan_step_fields,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL


def _frzl(value: float = 1.0, *, lasym: bool = False) -> TomnspsRZL:
    shape = (3, 2, 2)
    base = np.full(shape, float(value))
    kwargs = {}
    if lasym:
        kwargs.update(
            frsc=base + 6.0,
            frcs=base + 7.0,
            fzcc=base + 8.0,
            fzss=base + 9.0,
            flcc=base + 10.0,
            flss=base + 11.0,
        )
    return TomnspsRZL(
        frcc=base,
        frss=None,
        fzsc=base + 2.0,
        fzcs=None,
        flsc=base + 4.0,
        flcs=None,
        **kwargs,
    )


def _payload(tag: float, *, cache_valid=True) -> ScanForcePayload:
    block = np.asarray([float(tag), float(tag) + 1.0])
    blocks = ScanForceBlocks(*(block + idx for idx in range(12)))
    return ScanForcePayload(
        blocks=blocks,
        fsqr=tag + 20.0,
        fsqz=tag + 21.0,
        fsql=tag + 22.0,
        fsqr1=tag + 23.0,
        fsqz1=tag + 24.0,
        fsql1=tag + 25.0,
        cache_precond_diag=(tag,),
        cache_tcon=tag + 26.0,
        cache_norms=SimpleNamespace(tag=tag),
        cache_rz_scale=tag + 27.0,
        cache_l_scale=tag + 28.0,
        cache_rz_norm=tag + 29.0,
        cache_f_norm1=tag + 30.0,
        cache_rz_mats=tag + 31.0,
        cache_lam_prec=tag + 32.0,
        cache_valid=cache_valid,
    )


def _fake_cond(pred, true_fun, false_fun, operand):
    return true_fun(operand) if bool(np.asarray(pred)) else false_fun(operand)


def _build_payload_kwargs(*, frzl_rz: TomnspsRZL, apply_lambda_update_scale: bool = True) -> dict:
    return {
        "frzl_rz": frzl_rz,
        "cache_lam_prec": np.asarray(2.0),
        "w_mode_mn": np.full((2, 2), 3.0),
        "lambda_update_scale_j": np.asarray(5.0),
        "apply_lambda_update_scale": apply_lambda_update_scale,
        "fsqr": 1.0,
        "fsqz": 2.0,
        "fsql": 3.0,
        "f_norm1": 0.5,
        "delta_s": 0.25,
        "s": np.linspace(0.0, 1.0, 3),
        "lconm1": True,
        "cache_precond_diag": ("diag",),
        "cache_tcon": "tcon",
        "cache_norms": "norms",
        "cache_rz_scale": "rzs",
        "cache_l_scale": "ls",
        "cache_rz_norm": 4.0,
        "cache_f_norm1": 0.5,
        "cache_rz_mats": "mats",
        "cache_valid": True,
    }


def test_mask_scan_restart_force_payload_preserves_or_zeros_and_invalidates_cache():
    blocks = (np.asarray([1.0, 2.0]), np.asarray([[3.0, 4.0]]))

    kept, valid = mask_scan_restart_force_payload(
        force_blocks=blocks,
        cache_valid=True,
        do_restart=False,
    )
    for actual, expected in zip(kept, blocks):
        np.testing.assert_allclose(np.asarray(actual), expected)
    assert bool(np.asarray(valid))

    masked, valid = mask_scan_restart_force_payload(
        force_blocks=blocks,
        cache_valid=True,
        do_restart=True,
    )
    for actual, expected in zip(masked, blocks):
        np.testing.assert_allclose(np.asarray(actual), np.zeros_like(expected))
    assert not bool(np.asarray(valid))


def test_build_scan_force_payload_scales_lambda_blocks_and_zero_fills_symmetric_optionals():
    payload = build_scan_force_payload(
        frzl_rz=_frzl(1.0, lasym=False),
        cache_lam_prec=np.asarray(2.0),
        w_mode_mn=np.full((2, 2), 3.0),
        lambda_update_scale_j=np.asarray(5.0),
        apply_lambda_update_scale=True,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        f_norm1=0.5,
        delta_s=0.25,
        s=np.linspace(0.0, 1.0, 3),
        lconm1=True,
        cache_precond_diag=("diag",),
        cache_tcon="tcon",
        cache_norms="norms",
        cache_rz_scale="rzs",
        cache_l_scale="ls",
        cache_rz_norm=4.0,
        cache_f_norm1=0.5,
        cache_rz_mats="mats",
        cache_valid=True,
    )

    np.testing.assert_allclose(np.asarray(payload.blocks.frcc), np.full((3, 2, 2), 3.0))
    np.testing.assert_allclose(np.asarray(payload.blocks.flsc), np.full((3, 2, 2), 150.0))
    np.testing.assert_allclose(np.asarray(payload.blocks.frsc), 0.0)
    np.testing.assert_allclose(np.asarray(payload.blocks.flcc), 0.0)
    assert bool(np.asarray(payload.cache_valid))
    np.testing.assert_allclose(np.asarray(payload.fsql1), 200.0)


def test_build_scan_force_payload_uses_lasym_optional_blocks_and_current_optional_metric_source():
    optional_source = _frzl(1.0, lasym=True)
    optional_source = TomnspsRZL(
        frcc=optional_source.frcc,
        frss=optional_source.frss,
        fzsc=optional_source.fzsc,
        fzcs=optional_source.fzcs,
        flsc=optional_source.flsc,
        flcs=optional_source.flcs,
        frsc=optional_source.frsc,
        frcs=optional_source.frcs,
        fzcc=optional_source.fzcc,
        fzss=optional_source.fzss,
        flcc=np.full((3, 2, 2), 100.0),
        flss=np.full((3, 2, 2), 200.0),
    )

    payload = build_scan_force_payload(
        frzl_rz=_frzl(1.0, lasym=True),
        cache_lam_prec=np.asarray(2.0),
        w_mode_mn=np.full((2, 2), 3.0),
        lambda_update_scale_j=np.asarray(5.0),
        apply_lambda_update_scale=True,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        f_norm1=1.0,
        delta_s=0.25,
        s=np.linspace(0.0, 1.0, 3),
        lconm1=True,
        cache_precond_diag=("diag",),
        cache_tcon="tcon",
        cache_norms="norms",
        cache_rz_scale="rzs",
        cache_l_scale="ls",
        cache_rz_norm=4.0,
        cache_f_norm1=1.0,
        cache_rz_mats="mats",
        cache_valid=True,
        lambda_fsq1_optional_source=optional_source,
    )

    np.testing.assert_allclose(np.asarray(payload.blocks.frsc), np.full((3, 2, 2), 21.0))
    np.testing.assert_allclose(np.asarray(payload.blocks.flcc), np.full((3, 2, 2), 330.0))
    expected_fsql1 = (
        np.sum(np.full((2, 2, 2), 10.0) ** 2)
        + np.sum(np.full((2, 2, 2), 100.0) ** 2)
        + np.sum(np.full((2, 2, 2), 200.0) ** 2)
    ) * 0.25
    np.testing.assert_allclose(np.asarray(payload.fsql1), expected_fsql1)


def test_scan_payload_wrappers_include_nonzero_flcs_in_lambda_metric():
    base = np.full((3, 2, 2), 1.0)
    frzl = TomnspsRZL(
        frcc=base,
        frss=None,
        fzsc=base + 2.0,
        fzcs=None,
        flsc=base + 4.0,
        flcs=base + 5.0,
    )
    kwargs = _build_payload_kwargs(frzl_rz=frzl, apply_lambda_update_scale=False)

    current = current_scan_payload(**kwargs)
    restart = restart_scan_payload(**kwargs)

    np.testing.assert_allclose(np.asarray(current.blocks.flcs), np.full((3, 2, 2), 36.0))
    expected_fsql1 = (np.sum(np.full((2, 2, 2), 10.0) ** 2) + np.sum(np.full((2, 2, 2), 12.0) ** 2)) * 0.25
    np.testing.assert_allclose(np.asarray(current.fsql1), expected_fsql1)
    np.testing.assert_allclose(np.asarray(restart.fsql1), expected_fsql1)


def test_build_initial_preconditioner_cache_builds_fresh_cache(monkeypatch):
    import vmec_jax.preconditioner_1d_jax as precond_module

    matrix_calls = []
    monkeypatch.setattr(
        precond_module,
        "rz_preconditioner_matrices",
        lambda **kwargs: matrix_calls.append(kwargs) or ("fresh-mats", 0, 99),
    )

    norms = SimpleNamespace(tag="norms")
    cache = build_initial_preconditioner_cache(
        state_init="state",
        k=SimpleNamespace(bc="bc", tcon=np.asarray([0.25])),
        norms=norms,
        rz_scale="rz-scale",
        l_scale="l-scale",
        constraint_tcon0=0.0,
        zero_precond_diag=("zero-diag",),
        zero_tcon=np.asarray([0.0]),
        trig="trig",
        s=np.linspace(0.0, 1.0, 5),
        cfg=SimpleNamespace(),
        dtype=np.float64,
        scan_use_precomputed=True,
        scan_use_lax_tridi=False,
        lambda_preconditioner_func=lambda bc: np.asarray(2.0) if bc == "bc" else pytest.fail("wrong bc"),
        rz_norm_func=lambda state: np.asarray(4.0) if state == "state" else pytest.fail("wrong state"),
        resume_state=None,
    )

    assert cache.precond_diag == ("zero-diag",)
    assert cache.norms is norms
    assert cache.rz_scale == "rz-scale"
    assert cache.l_scale == "l-scale"
    assert cache.rz_mats == "fresh-mats"
    assert cache.jmax == 4
    assert bool(np.asarray(cache.valid))
    np.testing.assert_allclose(np.asarray(cache.f_norm1), 0.25)
    np.testing.assert_allclose(np.asarray(cache.lam_prec), 2.0)
    assert matrix_calls[0]["bc"] == "bc"
    assert matrix_calls[0]["use_precomputed"] is True
    assert matrix_calls[0]["use_lax_tridi"] is False


def test_build_initial_preconditioner_cache_overlays_resume_cache(monkeypatch):
    import vmec_jax.preconditioner_1d_jax as precond_module

    monkeypatch.setattr(
        precond_module,
        "rz_preconditioner_matrices",
        lambda **_kwargs: ("fresh-mats", 0, 3),
    )

    cache = build_initial_preconditioner_cache(
        state_init="state",
        k=SimpleNamespace(bc="bc", tcon=np.asarray([0.25])),
        norms="fresh-norms",
        rz_scale="fresh-rz",
        l_scale="fresh-l",
        constraint_tcon0=0.0,
        zero_precond_diag=("zero-diag",),
        zero_tcon=np.asarray([0.0]),
        trig="trig",
        s=np.linspace(0.0, 1.0, 4),
        cfg=SimpleNamespace(),
        dtype=np.float64,
        scan_use_precomputed=False,
        scan_use_lax_tridi=True,
        lambda_preconditioner_func=lambda _bc: np.asarray(2.0),
        rz_norm_func=lambda _state: np.asarray(4.0),
        resume_state={
            "vmec2000_cache_valid": False,
            "cache_precond_diag": ("resume-diag",),
            "cache_tcon": "resume-tcon",
            "cache_norms": "resume-norms",
            "cache_rz_scale": "resume-rz",
            "cache_l_scale": "resume-l",
            "cache_rz_norm": "8.0",
            "cache_f_norm1": "0.125",
            "cache_prec_rz_mats": "resume-mats",
            "cache_prec_lam_prec": "resume-lam",
        },
    )

    assert cache.precond_diag == ("resume-diag",)
    assert cache.tcon == "resume-tcon"
    assert cache.norms == "resume-norms"
    assert cache.rz_scale == "resume-rz"
    assert cache.l_scale == "resume-l"
    assert cache.rz_mats == "resume-mats"
    assert cache.lam_prec == "resume-lam"
    assert not bool(np.asarray(cache.valid))
    assert cache.jmax == 3
    np.testing.assert_allclose(np.asarray(cache.rz_norm), 8.0)
    np.testing.assert_allclose(np.asarray(cache.f_norm1), 0.125)


def test_build_current_preconditioned_scan_payload_keeps_valid_cache(monkeypatch):
    import vmec_jax.preconditioner_1d_jax as precond_module

    monkeypatch.setattr(
        precond_module,
        "rz_preconditioner_apply",
        lambda *, frzl_in, mats, jmax, cfg, use_precomputed, use_lax_tridi: frzl_in,
    )
    carry = SimpleNamespace(
        state="state",
        cache_precond_diag=("cached-diag",),
        cache_tcon=np.asarray([0.1]),
        cache_norms="cached-norms",
        cache_rz_scale="cached-rz-scale",
        cache_l_scale="cached-l-scale",
        cache_rz_norm=np.asarray(4.0),
        cache_f_norm1=np.asarray(0.25),
        cache_prec_lam_prec=np.asarray(2.0),
        cache_prec_rz_mats="cached-mats",
        cache_valid=True,
    )
    payload = build_current_preconditioned_scan_payload(
        need_bcovar_update=False,
        carry_adv=carry,
        k=SimpleNamespace(bc=object()),
        frzl=_frzl(1.0),
        norms_used=SimpleNamespace(),
        rz_scale="new-rz-scale",
        l_scale="new-l-scale",
        constraint_tcon0=0.0,
        zero_precond_diag=("zero",),
        zero_tcon=np.asarray([0.0]),
        trig=object(),
        s=np.linspace(0.0, 1.0, 3),
        cfg=SimpleNamespace(lconm1=True),
        dtype=float,
        scan_use_precomputed=False,
        scan_use_lax_tridi=False,
        lambda_preconditioner_func=lambda _bc: pytest.fail("valid cache should not refresh lambda preconditioner"),
        rz_norm_func=lambda state: np.asarray(99.0) if state == "state" else pytest.fail("wrong state"),
        scale_m1_precond_rhs_func=lambda frzl, _mats: frzl,
        w_mode_mn=np.full((2, 2), 3.0),
        lambda_update_scale_j=np.asarray(1.0),
        apply_lambda_update_scale=False,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delta_s=0.25,
        jmax0=2,
        cond=_fake_cond,
    )

    assert payload.cache_precond_diag == ("cached-diag",)
    assert payload.cache_norms == "cached-norms"
    assert payload.cache_rz_mats == "cached-mats"
    assert bool(np.asarray(payload.cache_valid))
    np.testing.assert_allclose(np.asarray(payload.cache_f_norm1), 0.25)


def test_build_current_preconditioned_scan_payload_refreshes_cache(monkeypatch):
    import vmec_jax.preconditioner_1d_jax as precond_module

    matrix_calls = []
    monkeypatch.setattr(
        precond_module,
        "rz_preconditioner_matrices",
        lambda **kwargs: matrix_calls.append(kwargs) or ("fresh-mats", 0, 2),
    )
    monkeypatch.setattr(
        precond_module,
        "rz_preconditioner_apply",
        lambda *, frzl_in, mats, jmax, cfg, use_precomputed, use_lax_tridi: frzl_in,
    )
    carry = SimpleNamespace(
        state="state",
        cache_precond_diag=("old",),
        cache_tcon=np.asarray([0.1]),
        cache_norms="old-norms",
        cache_rz_scale="old-rz-scale",
        cache_l_scale="old-l-scale",
        cache_rz_norm=np.asarray(9.0),
        cache_f_norm1=np.asarray(1.0 / 9.0),
        cache_prec_lam_prec=np.asarray(1.0),
        cache_prec_rz_mats="old-mats",
        cache_valid=False,
    )
    norms = SimpleNamespace(tag="fresh")
    payload = build_current_preconditioned_scan_payload(
        need_bcovar_update=True,
        carry_adv=carry,
        k=SimpleNamespace(bc="bc"),
        frzl=_frzl(1.0),
        norms_used=norms,
        rz_scale="fresh-rz-scale",
        l_scale="fresh-l-scale",
        constraint_tcon0=0.0,
        zero_precond_diag=("zero",),
        zero_tcon=np.asarray([0.0]),
        trig="trig",
        s=np.linspace(0.0, 1.0, 3),
        cfg=SimpleNamespace(lconm1=True),
        dtype=float,
        scan_use_precomputed=True,
        scan_use_lax_tridi=True,
        lambda_preconditioner_func=lambda bc: np.asarray(2.0) if bc == "bc" else pytest.fail("wrong bc"),
        rz_norm_func=lambda state: np.asarray(4.0) if state == "state" else pytest.fail("wrong state"),
        scale_m1_precond_rhs_func=lambda frzl, _mats: frzl,
        w_mode_mn=np.full((2, 2), 3.0),
        lambda_update_scale_j=np.asarray(1.0),
        apply_lambda_update_scale=False,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delta_s=0.25,
        jmax0=2,
        cond=_fake_cond,
    )

    assert payload.cache_precond_diag == ("zero",)
    assert payload.cache_norms is norms
    assert payload.cache_rz_scale == "fresh-rz-scale"
    assert payload.cache_l_scale == "fresh-l-scale"
    assert payload.cache_rz_mats == "fresh-mats"
    assert bool(np.asarray(payload.cache_valid))
    np.testing.assert_allclose(np.asarray(payload.cache_f_norm1), 0.25)
    assert matrix_calls[0]["bc"] == "bc"
    assert matrix_calls[0]["use_precomputed"] is True
    assert matrix_calls[0]["use_lax_tridi"] is True


def test_build_restart_preconditioned_scan_payload_recomputes_restart_forces(monkeypatch):
    import vmec_jax.preconditioner_1d_jax as precond_module

    monkeypatch.setattr(
        precond_module,
        "rz_preconditioner_matrices",
        lambda *, bc, k, trig, s, cfg, use_precomputed, use_lax_tridi: ("restart-mats", 1, 2),
    )
    monkeypatch.setattr(
        precond_module,
        "rz_preconditioner_apply",
        lambda *, frzl_in, mats, jmax, cfg, use_precomputed, use_lax_tridi: frzl_in,
    )

    def compute_forces_scan(state, **kwargs):
        assert state == "state-post"
        assert kwargs["include_edge"] is False
        return (
            SimpleNamespace(bc=object(), tcon=np.asarray([0.0])),
            _frzl(1.0),
            np.asarray(10.0),
            np.asarray(20.0),
            np.asarray(30.0),
            "restart-rz-scale",
            "restart-l-scale",
            SimpleNamespace(r1=np.asarray(2.0), fnorm=np.asarray(3.0), fnormL=np.asarray(5.0)),
        )

    payload = build_restart_preconditioned_scan_payload(
        state_post="state-post",
        compute_forces_scan_func=compute_forces_scan,
        trace_context=nullcontext,
        zero_m1=np.asarray(0.0),
        zero_precond_diag=("zero",),
        zero_tcon=np.asarray([0.0]),
        constraint_active_false=False,
        constraint_tcon0=0.0,
        trig=SimpleNamespace(),
        s=np.linspace(0.0, 1.0, 3),
        cfg=SimpleNamespace(lconm1=True),
        dtype=np.float64,
        scan_use_precomputed=False,
        scan_use_lax_tridi=False,
        lambda_preconditioner_func=lambda _bc: np.asarray(2.0),
        rz_norm_func=lambda _state: np.asarray(4.0),
        scale_m1_precond_rhs_func=lambda frzl, _mats: frzl,
        w_mode_mn=np.ones((2, 2)),
        lambda_update_scale_j=np.asarray(1.0),
        apply_lambda_update_scale=False,
        delta_s=np.asarray(0.25),
        jmax0=2,
    )

    np.testing.assert_allclose(np.asarray(payload.fsqr), 60.0)
    np.testing.assert_allclose(np.asarray(payload.fsqz), 120.0)
    np.testing.assert_allclose(np.asarray(payload.fsql), 150.0)
    np.testing.assert_allclose(np.asarray(payload.cache_f_norm1), 0.25)
    assert payload.cache_rz_mats == "restart-mats"
    assert bool(np.asarray(payload.cache_valid))


def test_select_scan_force_payload_restart_and_no_restart_paths():
    current = _payload(1.0, cache_valid=True)
    restart = _payload(10.0, cache_valid=True)

    selected = select_scan_force_payload(
        do_restart=True,
        use_restart_payload=True,
        restart_payload_fn=lambda _: restart,
        current_payload_fn=lambda _: current,
        cond=_fake_cond,
    )
    np.testing.assert_allclose(np.asarray(selected.blocks.frcc), np.asarray(restart.blocks.frcc))
    assert bool(np.asarray(selected.cache_valid))

    selected = select_scan_force_payload(
        do_restart=True,
        use_restart_payload=False,
        restart_payload_fn=lambda _: restart,
        current_payload_fn=lambda _: current,
        cond=_fake_cond,
    )
    np.testing.assert_allclose(np.asarray(selected.blocks.frcc), 0.0)
    assert not bool(np.asarray(selected.cache_valid))

    selected = select_scan_force_payload(
        do_restart=False,
        use_restart_payload=False,
        restart_payload_fn=lambda _: restart,
        current_payload_fn=lambda _: current,
        cond=_fake_cond,
    )
    np.testing.assert_allclose(np.asarray(selected.blocks.frcc), np.asarray(current.blocks.frcc))
    assert bool(np.asarray(selected.cache_valid))


def test_select_scan_step_fields_matches_accept_reject_semantics():
    accepted = ScanStepFields(*(f"accepted-{idx}" for idx in range(15)))
    rejected = ScanStepFields(*(f"rejected-{idx}" for idx in range(15)))

    fields = select_scan_step_fields(
        vmec2000_control=True,
        do_restart=True,
        accept_step_fn=lambda _: accepted,
        reject_step_fn=lambda _: rejected,
        cond=_fake_cond,
    )
    assert fields.state == "accepted-0"

    fields = select_scan_step_fields(
        vmec2000_control=False,
        do_restart=True,
        accept_step_fn=lambda _: accepted,
        reject_step_fn=lambda _: rejected,
        cond=_fake_cond,
    )
    assert fields.state == "rejected-0"

    fields = select_scan_step_fields(
        vmec2000_control=False,
        do_restart=False,
        accept_step_fn=lambda _: accepted,
        reject_step_fn=lambda _: rejected,
        cond=_fake_cond,
    )
    assert fields.state == "accepted-0"


def test_build_scan_step_fields_applies_vmec_accept_update_and_nonvmec_reject():
    payload = _payload(1.0)
    state = SimpleNamespace(
        layout="layout",
        Rcos=np.asarray([10.0, 20.0]),
        Rsin=np.asarray([1.0, 2.0]),
        Zcos=np.asarray([3.0, 4.0]),
        Zsin=np.asarray([30.0, 40.0]),
        Lcos=np.asarray([5.0, 6.0]),
        Lsin=np.asarray([50.0, 60.0]),
    )
    zeros = tuple(np.zeros(2) for _ in range(12))

    def add_modes(a, b):
        return np.asarray(a) + np.asarray(b)

    fields = build_scan_step_fields(
        payload=payload,
        state_post=state,
        velocity_blocks_post=zeros,
        inv_tau_post=np.asarray([0.1, 0.2]),
        fsq_prev_post=np.asarray(1.0),
        fsq1=np.asarray(1.0),
        time_step_post=np.asarray(0.5),
        iter2=np.asarray(2),
        iter1_post=np.asarray(1),
        k_ndamp=2,
        dtype=np.float64,
        flip_sign=np.asarray(1.0),
        lasym=True,
        static=SimpleNamespace(),
        edge_Rcos=None,
        edge_Rsin=None,
        edge_Zcos=None,
        edge_Zsin=None,
        free_boundary_enabled=False,
        idx00=0,
        mn_cos_to_signed_physical=add_modes,
        mn_sin_to_signed_physical=add_modes,
        mn_sin_to_signed_physical_lambda=add_modes,
        mn_cos_to_signed_physical_lambda=add_modes,
        enforce_fixed_boundary_and_axis=lambda state_arg, *_args, **_kwargs: state_arg,
        apply_vmec_lambda_axis_rules=lambda state_arg: state_arg,
        vmec2000_control=True,
        do_restart=True,
        cond=_fake_cond,
    )
    assert fields.state is not state
    assert bool(np.any(np.asarray(fields.state.Rcos) != np.asarray(state.Rcos)))
    assert bool(np.any(np.asarray(fields.state.Rsin) != np.asarray(state.Rsin)))
    assert np.asarray(fields.fsq_prev) == pytest.approx(1.0)

    rejected = build_scan_step_fields(
        payload=payload,
        state_post=state,
        velocity_blocks_post=zeros,
        inv_tau_post=np.asarray([0.1, 0.2]),
        fsq_prev_post=np.asarray(7.0),
        fsq1=np.asarray(1.0),
        time_step_post=np.asarray(0.5),
        iter2=np.asarray(2),
        iter1_post=np.asarray(1),
        k_ndamp=2,
        dtype=np.float64,
        flip_sign=np.asarray(1.0),
        lasym=True,
        static=SimpleNamespace(),
        edge_Rcos=None,
        edge_Rsin=None,
        edge_Zcos=None,
        edge_Zsin=None,
        free_boundary_enabled=False,
        idx00=0,
        mn_cos_to_signed_physical=add_modes,
        mn_sin_to_signed_physical=add_modes,
        mn_sin_to_signed_physical_lambda=add_modes,
        mn_cos_to_signed_physical_lambda=add_modes,
        enforce_fixed_boundary_and_axis=lambda state_arg, *_args, **_kwargs: state_arg,
        apply_vmec_lambda_axis_rules=lambda state_arg: state_arg,
        vmec2000_control=False,
        do_restart=True,
        cond=_fake_cond,
    )
    assert rejected.state is state
    for actual, expected in zip(tuple(rejected)[1:13], zeros, strict=True):
        np.testing.assert_allclose(np.asarray(actual), expected)
    assert np.asarray(rejected.fsq_prev) == pytest.approx(7.0)

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solvers.fixed_boundary.scan.payload import (
    ScanForceBlocks,
    ScanForcePayload,
    ScanStepFields,
    build_scan_force_payload,
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

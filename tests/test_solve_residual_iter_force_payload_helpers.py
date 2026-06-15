from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solvers.fixed_boundary.residual.force_payload import (
    ResidualForceMetricPayload,
    ResidualForcePayloadResult,
    force_z_channel_square_sums,
    maybe_debug_force_z_channel_square_sums,
    metric_force_payload_after_edge_policy,
    residual_force_payload_after_m1_scalxc_with_scan_debug,
    residual_force_payload_from_kernels,
    residual_force_gcx2_after_edge_policy,
    residual_force_z_nan_guard,
    resolve_residual_force_mask_pack,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL


def _frzl(*, edge_z_nan: bool = False) -> TomnspsRZL:
    shape = (3, 2, 1)

    def block(offset: float) -> np.ndarray:
        return np.arange(np.prod(shape), dtype=float).reshape(shape) + offset

    fzsc = block(20.0)
    if edge_z_nan:
        fzsc = fzsc.copy()
        fzsc[-1, 0, 0] = np.nan
    return TomnspsRZL(
        frcc=block(1.0),
        frss=block(10.0),
        fzsc=fzsc,
        fzcs=block(30.0),
        flsc=block(40.0),
        flcs=block(50.0),
        frsc=block(60.0),
        frcs=block(70.0),
        fzcc=block(80.0),
        fzss=block(90.0),
        flcc=block(100.0),
        flss=block(110.0),
    )


def test_force_z_channel_square_sums_handles_asymmetric_and_symmetric_only_payloads() -> None:
    frzl = _frzl()

    fzsc2, fzcs2 = force_z_channel_square_sums(frzl)

    np.testing.assert_allclose(np.asarray(fzsc2), np.sum(frzl.fzsc * frzl.fzsc))
    np.testing.assert_allclose(np.asarray(fzcs2), np.sum(frzl.fzcs * frzl.fzcs))

    symmetric = TomnspsRZL(
        frcc=frzl.frcc,
        frss=None,
        fzsc=frzl.fzsc,
        fzcs=None,
        flsc=frzl.flsc,
        flcs=None,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    )
    _, fzcs2_symmetric = force_z_channel_square_sums(symmetric)
    assert float(np.asarray(fzcs2_symmetric)) == 0.0


def test_maybe_debug_force_z_channel_square_sums_uses_injected_debug_printer() -> None:
    calls = []

    class DebugPrinter:
        @staticmethod
        def print(message, **kwargs):
            calls.append((message, kwargs))

    frzl = _frzl()
    maybe_debug_force_z_channel_square_sums(
        frzl,
        enabled=False,
        message="ignored {fzsc} {fzcs}",
        debug_module=DebugPrinter(),
    )
    assert calls == []

    maybe_debug_force_z_channel_square_sums(
        frzl,
        enabled=True,
        message="debug {fzsc} {fzcs}",
        debug_module=DebugPrinter(),
    )
    assert calls[0][0] == "debug {fzsc} {fzcs}"
    np.testing.assert_allclose(np.asarray(calls[0][1]["fzsc"]), np.sum(frzl.fzsc * frzl.fzsc))
    np.testing.assert_allclose(np.asarray(calls[0][1]["fzcs"]), np.sum(frzl.fzcs * frzl.fzcs))


def test_residual_force_payload_after_m1_scalxc_debug_wrapper_selects_fast_and_debug_paths() -> None:
    frzl = _frzl()
    final_calls = []

    def final_func(payload, **kwargs):
        final_calls.append(kwargs)
        return payload

    got = residual_force_payload_after_m1_scalxc_with_scan_debug(
        frzl,
        s=np.asarray([0.0, 0.5, 1.0]),
        apply_m1_constraints=True,
        lconm1=True,
        zero_m1=True,
        scan_debug_force_enabled=False,
        final_func=final_func,
    )
    assert got is frzl
    assert final_calls[0]["apply_m1_constraints"] is True

    debug_calls = []

    class DebugPrinter:
        @staticmethod
        def print(message, **kwargs):
            debug_calls.append((message, kwargs))

    after_m1 = _frzl()
    after_zero = _frzl()
    after_scalxc = _frzl()

    def stages_func(_payload, **_kwargs):
        return SimpleNamespace(after_m1=after_m1, after_zero_m1=after_zero, after_scalxc=after_scalxc)

    got = residual_force_payload_after_m1_scalxc_with_scan_debug(
        frzl,
        s=np.asarray([0.0, 0.5, 1.0]),
        apply_m1_constraints=True,
        lconm1=True,
        zero_m1=True,
        scan_debug_force_enabled=True,
        debug_module=DebugPrinter(),
        stages_func=stages_func,
    )
    assert got is after_scalxc
    assert [call[0] for call in debug_calls] == [
        "[scan-debug-m1] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
        "[scan-debug-zero] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
        "[scan-debug-scalxc] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
    ]


def test_resolve_residual_force_mask_pack_defaults_to_metric_edge_policy() -> None:
    static = SimpleNamespace(tomnsps_masks="interior", tomnsps_masks_edge="edge")

    include_edge_residual, mask = resolve_residual_force_mask_pack(
        static,
        include_edge=True,
        include_edge_residual=None,
    )
    assert include_edge_residual is True
    assert mask == "edge"

    include_edge_residual, mask = resolve_residual_force_mask_pack(
        static,
        include_edge=False,
        include_edge_residual=None,
    )
    assert include_edge_residual is False
    assert mask == "interior"


def test_resolve_residual_force_mask_pack_honors_explicit_residual_edge_policy() -> None:
    static = SimpleNamespace(tomnsps_masks="interior", tomnsps_masks_edge="edge")

    include_edge_residual, mask = resolve_residual_force_mask_pack(
        static,
        include_edge=False,
        include_edge_residual=True,
    )

    assert include_edge_residual is True
    assert mask == "edge"


def test_resolve_residual_force_mask_pack_handles_missing_precomputed_masks() -> None:
    include_edge_residual, mask = resolve_residual_force_mask_pack(
        SimpleNamespace(),
        include_edge=True,
        include_edge_residual=False,
    )

    assert include_edge_residual is False
    assert mask is None


def test_metric_force_payload_after_edge_policy_keeps_full_payload_when_requested() -> None:
    frzl = _frzl()

    got = metric_force_payload_after_edge_policy(frzl, include_edge=True)

    assert got is frzl


def test_metric_force_payload_after_edge_policy_masks_only_metric_edge_payload() -> None:
    frzl = _frzl()

    got = metric_force_payload_after_edge_policy(frzl, include_edge=False)

    for name in ("frcc", "frss", "fzsc", "fzcs", "frsc", "frcs", "fzcc", "fzss"):
        np.testing.assert_allclose(np.asarray(getattr(got, name))[-1], 0.0)
        np.testing.assert_allclose(np.asarray(getattr(got, name))[:-1], np.asarray(getattr(frzl, name))[:-1])
    for name in ("flsc", "flcs", "flcc", "flss"):
        np.testing.assert_allclose(np.asarray(getattr(got, name)), np.asarray(getattr(frzl, name)))


def test_residual_force_z_nan_guard_returns_zero_for_finite_payload() -> None:
    got = residual_force_z_nan_guard(_frzl())

    assert float(np.asarray(got)) == 0.0


def test_residual_force_z_nan_guard_preserves_nan_payload() -> None:
    got = residual_force_z_nan_guard(_frzl(edge_z_nan=True))

    assert np.isnan(np.asarray(got))


def test_residual_force_gcx2_after_edge_policy_applies_edge_and_nan_guard() -> None:
    frzl = _frzl(edge_z_nan=True)

    got = residual_force_gcx2_after_edge_policy(
        frzl,
        include_edge=False,
        lconm1=True,
        s=np.asarray([0.0, 0.5, 1.0]),
    )

    assert isinstance(got, ResidualForceMetricPayload)
    assert np.isnan(np.asarray(got.gcz2))
    np.testing.assert_allclose(np.asarray(got.frzl_metric.fzsc)[-1, 1, 0], 0.0)

    finite = residual_force_gcx2_after_edge_policy(
        _frzl(edge_z_nan=False),
        include_edge=False,
        lconm1=True,
        s=np.asarray([0.0, 0.5, 1.0]),
    )
    # R/Z norms exclude the edge row, lambda norms retain all rows.
    frzl_no_edge = metric_force_payload_after_edge_policy(_frzl(edge_z_nan=False), include_edge=False)
    expected_r = 0.0
    expected_z = 0.0
    for name in ("frcc", "frss", "frsc", "frcs"):
        expected_r += float(np.sum(np.asarray(getattr(frzl_no_edge, name))[:-1] ** 2))
    for name in ("fzsc", "fzcs", "fzcc", "fzss"):
        expected_z += float(np.sum(np.asarray(getattr(frzl_no_edge, name))[:-1] ** 2))
    expected_l = 0.0
    for name in ("flsc", "flcs", "flcc", "flss"):
        expected_l += float(np.sum(np.asarray(getattr(frzl_no_edge, name)) ** 2))

    np.testing.assert_allclose(np.asarray(finite.gcr2), expected_r)
    np.testing.assert_allclose(np.asarray(finite.gcz2), expected_z)
    np.testing.assert_allclose(np.asarray(finite.gcl2), expected_l)


def test_residual_force_payload_from_kernels_routes_masks_callbacks_and_hlo_dump() -> None:
    frzl = _frzl()
    static = SimpleNamespace(cfg=SimpleNamespace(ntheta=8, nzeta=6), tomnsps_masks="interior", tomnsps_masks_edge="edge")
    residual_calls = []
    postprocess_calls = []
    metric_calls = []
    raw_callbacks = []
    gc_callbacks = []
    hlo_calls = []

    def residual_func(kernels, **kwargs):
        residual_calls.append((kernels, kwargs))
        return frzl

    def postprocess_func(payload, **kwargs):
        postprocess_calls.append((payload, kwargs))
        return payload

    def metric_func(payload, **kwargs):
        metric_calls.append((payload, kwargs))
        return ResidualForceMetricPayload(frzl_metric=payload, gcr2=1.0, gcz2=2.0, gcl2=3.0)

    def hlo_dump_func(**kwargs):
        hlo_calls.append(kwargs)
        # Exercise the nested HLO residual function with the same fake kernels.
        kwargs["fn"](*kwargs["args"])

    got = residual_force_payload_from_kernels(
        kernels="kernels",
        static=static,
        wout="wout",
        trig="trig",
        apply_lforbal=True,
        include_edge=False,
        include_edge_residual=True,
        apply_m1_constraints=True,
        lconm1=False,
        zero_m1="zero",
        s=np.asarray([0.0, 0.5, 1.0]),
        scan_debug_force_enabled=False,
        dump_hlo_force_tomnsps=True,
        hlo_dump_func=hlo_dump_func,
        raw_tomnsps_callback=raw_callbacks.append,
        gc_callback=gc_callbacks.append,
        residual_func=residual_func,
        postprocess_func=postprocess_func,
        metric_func=metric_func,
    )

    assert isinstance(got, ResidualForcePayloadResult)
    assert got.include_edge_residual is True
    assert got.mask_pack == "edge"
    assert got.frzl_raw is frzl
    assert got.frzl_full is frzl
    assert got.metric_payload.gcr2 == 1.0
    assert raw_callbacks == [frzl]
    assert gc_callbacks == [frzl]
    assert len(hlo_calls) == 1
    assert residual_calls[0][1]["cfg_ntheta"] == 8
    assert residual_calls[0][1]["cfg_nzeta"] == 6
    assert residual_calls[0][1]["include_edge"] is True
    assert residual_calls[0][1]["masks"] == "edge"
    assert residual_calls[1][1]["include_edge"] is True
    assert postprocess_calls[0][1]["apply_m1_constraints"] is True
    assert postprocess_calls[0][1]["lconm1"] is False
    assert postprocess_calls[0][1]["zero_m1"] == "zero"
    assert metric_calls[0][1]["include_edge"] is False
    assert metric_calls[0][1]["lconm1"] is False


def test_residual_force_payload_from_kernels_skips_optional_callbacks() -> None:
    frzl = _frzl()
    static = SimpleNamespace(cfg=SimpleNamespace(ntheta=8, nzeta=6))
    residual_calls = []

    def residual_func(kernels, **kwargs):
        residual_calls.append((kernels, kwargs))
        return frzl

    got = residual_force_payload_from_kernels(
        kernels="kernels",
        static=static,
        wout=None,
        trig=None,
        apply_lforbal=False,
        include_edge=True,
        include_edge_residual=None,
        apply_m1_constraints=False,
        lconm1=True,
        zero_m1=False,
        s=np.asarray([0.0, 0.5, 1.0]),
        scan_debug_force_enabled=False,
        dump_hlo_force_tomnsps=False,
        residual_func=residual_func,
        postprocess_func=lambda payload, **_kwargs: payload,
        metric_func=lambda payload, **_kwargs: ResidualForceMetricPayload(
            frzl_metric=payload,
            gcr2=0.0,
            gcz2=0.0,
            gcl2=0.0,
        ),
    )

    assert got.include_edge_residual is True
    assert got.mask_pack is None
    assert len(residual_calls) == 1
    assert residual_calls[0][0] == "kernels"
    assert residual_calls[0][1]["include_edge"] is True
    assert residual_calls[0][1]["masks"] is None

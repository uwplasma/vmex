from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solve_residual_iter_force_payload_helpers import (
    ResidualForceMetricPayload,
    metric_force_payload_after_edge_policy,
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

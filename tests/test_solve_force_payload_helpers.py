from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import vmec_jax.solve as solve
from vmec_jax.solve_force_payload_helpers import (
    ForceBlocks,
    ResidualForcePayloadStages,
    normalize_force_blocks,
    preconditioner_output_blocks_np,
    residual_force_payload_after_m1_scalxc,
    residual_force_payload_m1_scalxc_stages,
    zero_edge_rz_force_block,
    zero_edge_rz_force_blocks,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL


def _blocks() -> TomnspsRZL:
    shape = (3, 3, 1)

    def block(offset: float) -> np.ndarray:
        return np.arange(np.prod(shape), dtype=float).reshape(shape) + offset

    return TomnspsRZL(
        frcc=block(1.0),
        frss=block(10.0),
        fzsc=block(20.0),
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


def test_solve_reexports_payload_helpers() -> None:
    assert solve._ForceBlocks is ForceBlocks
    assert solve._zero_edge_rz_force_block is zero_edge_rz_force_block
    assert solve._zero_edge_rz_force_blocks is zero_edge_rz_force_blocks
    assert solve._preconditioner_output_blocks_np is preconditioner_output_blocks_np
    assert solve._residual_force_payload_m1_scalxc_stages is residual_force_payload_m1_scalxc_stages


def test_residual_force_payload_applies_m1_zeroing_and_scalxc_in_order() -> None:
    frzl = _blocks()
    s = np.asarray([0.0, 0.25, 1.0])

    got = residual_force_payload_after_m1_scalxc(
        frzl,
        s=s,
        apply_m1_constraints=True,
        lconm1=True,
        zero_m1=True,
    )

    scalxc = np.asarray([2.0, 2.0, 1.0])[:, None]
    osqrt2 = 1.0 / np.sqrt(2.0)
    np.testing.assert_allclose(np.asarray(got.frss)[:, 1, :], (frzl.frss[:, 1, :] + frzl.fzcs[:, 1, :]) * osqrt2 * scalxc)
    np.testing.assert_allclose(np.asarray(got.fzcs)[:, 1, :], 0.0)
    np.testing.assert_allclose(np.asarray(got.frsc)[:, 1, :], (frzl.frsc[:, 1, :] + frzl.fzcc[:, 1, :]) * osqrt2 * scalxc)
    np.testing.assert_allclose(np.asarray(got.fzcc)[:, 1, :], 0.0)
    np.testing.assert_allclose(np.asarray(got.fzsc)[:, 1, :], frzl.fzsc[:, 1, :] * scalxc)
    np.testing.assert_allclose(np.asarray(got.flsc)[:, 1, :], frzl.flsc[:, 1, :] * scalxc)

    np.testing.assert_allclose(np.asarray(got.frss)[:, 0, :], frzl.frss[:, 0, :])
    np.testing.assert_allclose(np.asarray(got.fzcs)[:, 2, :], frzl.fzcs[:, 2, :])


def test_residual_force_payload_stages_expose_intermediate_debug_payloads() -> None:
    frzl = _blocks()
    s = np.asarray([0.0, 0.25, 1.0])

    stages = residual_force_payload_m1_scalxc_stages(
        frzl,
        s=s,
        apply_m1_constraints=True,
        lconm1=True,
        zero_m1=True,
    )

    assert isinstance(stages, ResidualForcePayloadStages)
    osqrt2 = 1.0 / np.sqrt(2.0)
    scalxc = np.asarray([2.0, 2.0, 1.0])[:, None]
    np.testing.assert_allclose(
        np.asarray(stages.after_m1.frss)[:, 1, :],
        (frzl.frss[:, 1, :] + frzl.fzcs[:, 1, :]) * osqrt2,
    )
    np.testing.assert_allclose(
        np.asarray(stages.after_m1.fzcs)[:, 1, :],
        (frzl.frss[:, 1, :] - frzl.fzcs[:, 1, :]) * osqrt2,
    )
    np.testing.assert_allclose(np.asarray(stages.after_zero_m1.fzcs)[:, 1, :], 0.0)
    np.testing.assert_allclose(
        np.asarray(stages.after_scalxc.frss)[:, 1, :],
        (frzl.frss[:, 1, :] + frzl.fzcs[:, 1, :]) * osqrt2 * scalxc,
    )
    final_payload = residual_force_payload_after_m1_scalxc(
        frzl,
        s=s,
        apply_m1_constraints=True,
        lconm1=True,
        zero_m1=True,
    )
    np.testing.assert_allclose(np.asarray(stages.after_scalxc.frss), np.asarray(final_payload.frss))


def test_residual_force_payload_can_skip_m1_transform_but_still_applies_scalxc() -> None:
    frzl = _blocks()

    got = residual_force_payload_after_m1_scalxc(
        frzl,
        s=np.asarray([0.0, 0.25, 1.0]),
        apply_m1_constraints=False,
        lconm1=True,
        zero_m1=False,
    )

    np.testing.assert_allclose(np.asarray(got.frss)[:, 1, :], frzl.frss[:, 1, :] * np.asarray([2.0, 2.0, 1.0])[:, None])
    np.testing.assert_allclose(np.asarray(got.fzcs)[:, 1, :], frzl.fzcs[:, 1, :] * np.asarray([2.0, 2.0, 1.0])[:, None])


def test_zero_edge_rz_force_blocks_masks_only_rz_payload() -> None:
    frzl = _blocks()

    got = zero_edge_rz_force_blocks(frzl)

    for name in ("frcc", "frss", "fzsc", "fzcs", "frsc", "frcs", "fzcc", "fzss"):
        np.testing.assert_allclose(np.asarray(getattr(got, name))[-1], 0.0)
        np.testing.assert_allclose(np.asarray(getattr(got, name))[:-1], np.asarray(getattr(frzl, name))[:-1])
    for name in ("flsc", "flcs", "flcc", "flss"):
        np.testing.assert_allclose(np.asarray(getattr(got, name)), np.asarray(getattr(frzl, name)))

    one_row = np.ones((1, 3, 1))
    assert zero_edge_rz_force_block(one_row) is one_row


def test_zero_edge_rz_force_block_handles_none_and_device_array_path() -> None:
    assert zero_edge_rz_force_block(None) is None

    one_row = np.ones((1, 2))
    short = zero_edge_rz_force_block(one_row, preserve_numpy=False)
    np.testing.assert_allclose(np.asarray(short), one_row)

    block = np.arange(6.0).reshape(3, 2)
    masked = zero_edge_rz_force_block(block, preserve_numpy=False)
    np.testing.assert_allclose(np.asarray(masked)[:-1], block[:-1])
    np.testing.assert_allclose(np.asarray(masked)[-1], 0.0)


def test_normalize_force_blocks_preserves_numpy_blocks_and_optional_none() -> None:
    frzl = _blocks()
    frzl = TomnspsRZL(
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

    got = normalize_force_blocks(frzl)

    assert got.frcc is frzl.frcc
    assert got.frss is None
    assert got.flcs is None


def test_normalize_force_blocks_falls_back_for_non_dataclass_payload() -> None:
    base = np.asarray([[1.0, np.nan]])
    frzl = SimpleNamespace(
        frcc=base,
        frss=None,
        fzsc=base + 1.0,
        fzcs=None,
        flsc=base + 2.0,
        flcs=None,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    )

    got = normalize_force_blocks(frzl)

    assert isinstance(got, TomnspsRZL)
    np.testing.assert_allclose(np.asarray(got.frcc), base)
    assert got.frss is None


def test_preconditioner_output_payload_scales_only_lambda_blocks() -> None:
    base = np.arange(8.0).reshape(2, 2, 2) + 1.0
    lam = np.linspace(1.0, 2.0, 8).reshape(2, 2, 2)
    frzl_rz = SimpleNamespace(
        frcc=base,
        frss=None,
        fzsc=base + 20.0,
        fzcs=base + 30.0,
        flsc=base + 40.0,
        flcs=None,
        frsc=base + 60.0,
        frcs=None,
        fzcc=base + 80.0,
        fzss=None,
        flcc=base + 100.0,
        flss=None,
    )

    got = preconditioner_output_blocks_np(frzl_rz=frzl_rz, lam_prec=lam)

    assert isinstance(got, ForceBlocks)
    assert got.frss is None
    assert got.flcs is None
    assert got.frcs is None
    np.testing.assert_allclose(got.frcc, base)
    np.testing.assert_allclose(got.fzcc, base + 80.0)
    np.testing.assert_allclose(got.flsc, (base + 40.0) * lam)
    np.testing.assert_allclose(got.flcc, (base + 100.0) * lam)

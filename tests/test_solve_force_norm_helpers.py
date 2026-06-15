from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.solvers.fixed_boundary.residual.force_norms import (
    mode_weight_force_blocks_jax,
    residual_fsq_from_norms,
)
from vmec_jax.solvers.fixed_boundary.residual.payload_blocks import ForceBlocks


def test_residual_fsq_from_norms_matches_vmec_scalar_products() -> None:
    norms = SimpleNamespace(r1=2.0, fnorm=3.0, fnormL=5.0)

    fsqr, fsqz, fsql = residual_fsq_from_norms(
        norms,
        gcr2=np.asarray(7.0),
        gcz2=np.asarray(11.0),
        gcl2=np.asarray(13.0),
    )

    assert float(np.asarray(fsqr)) == 42.0
    assert float(np.asarray(fsqz)) == 66.0
    assert float(np.asarray(fsql)) == 65.0


def test_residual_fsq_from_norms_preserves_jax_array_inputs_and_solve_alias() -> None:
    import vmec_jax.solve as solve

    norms = SimpleNamespace(
        r1=jnp.asarray(0.5),
        fnorm=jnp.asarray(4.0),
        fnormL=jnp.asarray(8.0),
    )

    got = residual_fsq_from_norms(
        norms,
        gcr2=jnp.asarray(1.5),
        gcz2=jnp.asarray(2.5),
        gcl2=jnp.asarray(3.5),
    )
    alias = solve._residual_fsq_from_norms(
        norms,
        gcr2=jnp.asarray(1.5),
        gcz2=jnp.asarray(2.5),
        gcl2=jnp.asarray(3.5),
    )

    for actual, expected in zip(got, (3.0, 5.0, 28.0)):
        np.testing.assert_allclose(np.asarray(actual), expected)
    for actual, expected in zip(alias, got):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))


def test_mode_weight_force_blocks_jax_scales_all_channels_and_zero_fills_optional_blocks() -> None:
    import vmec_jax.solve as solve

    base = np.arange(8.0).reshape(2, 2, 2) + 1.0
    weights = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    blocks = ForceBlocks(
        frcc=base,
        frss=None,
        fzsc=base + 10.0,
        fzcs=None,
        flsc=base + 20.0,
        flcs=base + 30.0,
        frsc=base + 40.0,
        frcs=base + 50.0,
        fzcc=base + 60.0,
        fzss=base + 70.0,
        flcc=base + 80.0,
        flss=base + 90.0,
    )

    got = mode_weight_force_blocks_jax(blocks, w_mode_mn=weights)
    alias = solve._mode_weight_force_blocks_jax(blocks, w_mode_mn=weights)

    expected_weight = weights[None, :, :]
    np.testing.assert_allclose(np.asarray(got.frcc), base * expected_weight)
    np.testing.assert_allclose(np.asarray(got.frss), np.zeros_like(base))
    np.testing.assert_allclose(np.asarray(got.fzsc), (base + 10.0) * expected_weight)
    np.testing.assert_allclose(np.asarray(got.fzcs), np.zeros_like(base))
    np.testing.assert_allclose(np.asarray(got.flcs), (base + 30.0) * expected_weight)
    np.testing.assert_allclose(np.asarray(got.frsc), (base + 40.0) * expected_weight)
    np.testing.assert_allclose(np.asarray(got.flss), (base + 90.0) * expected_weight)
    for actual, expected in zip(alias, got):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))

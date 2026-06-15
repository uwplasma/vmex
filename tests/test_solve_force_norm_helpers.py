from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.solve_force_norm_helpers import residual_fsq_from_norms


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

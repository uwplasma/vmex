from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.solvers.fixed_boundary.optimization.residual_objective import (
    assemble_residual_objective_terms,
    prepare_residual_objective_blocks,
    residual_objective_vector,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL


def _block(value: float, *, ns: int = 3) -> np.ndarray:
    return np.full((ns, 2, 1), float(value), dtype=float)


def _minimal_frzl(**overrides) -> TomnspsRZL:
    values = {
        "frcc": _block(1.0),
        "frss": None,
        "fzsc": _block(2.0),
        "fzcs": None,
        "flsc": _block(3.0),
        "flcs": None,
    }
    values.update(overrides)
    return TomnspsRZL(**values)


def test_objective_terms_apply_norm_weights_and_edge_policy():
    frzl = _minimal_frzl(frss=_block(4.0), flcs=_block(5.0))
    norms = SimpleNamespace(r1=jnp.asarray(2.0), fnorm=jnp.asarray(3.0), fnormL=jnp.asarray(7.0))

    terms = assemble_residual_objective_terms(
        frzl=frzl,
        norms=norms,
        s=jnp.asarray([0.0, 0.5, 1.0]),
        w_rz=0.5,
        w_l=0.25,
        zero_m1_zforce=False,
        lconm1=True,
        apply_m1_constraints=False,
        zero_m1_after_m1_constraints=False,
        include_edge=False,
        apply_scalxc=False,
        zero_edge_rz_blocks=False,
        objective_scale=2.0,
    )

    assert float(terms.gcr2) == 2 * 2 * (1.0**2 + 4.0**2)
    assert float(terms.gcz2) == 2 * 2 * (2.0**2)
    assert float(terms.gcl2) == 3 * 2 * (3.0**2 + 5.0**2)
    assert float(terms.fsqr2) == float(terms.gcr2) * 6.0
    assert float(terms.fsqz2) == float(terms.gcz2) * 6.0
    assert float(terms.fsql2) == float(terms.gcl2) * 7.0
    expected_w = 2.0 * (0.5 * (float(terms.fsqr2) + float(terms.fsqz2)) + 0.25 * float(terms.fsql2))
    assert float(terms.w) == expected_w


def test_prepare_blocks_preserves_existing_lbfgs_and_gn_m1_zero_ordering():
    frss = _block(0.0)
    fzcs = _block(0.0)
    frss[:, 1, :] = 4.0
    fzcs[:, 1, :] = 2.0
    frzl = _minimal_frzl(frss=frss, fzcs=fzcs)

    lbfgs_order = prepare_residual_objective_blocks(
        frzl=frzl,
        s=jnp.asarray([0.0, 0.5, 1.0]),
        zero_m1_zforce=True,
        lconm1=True,
        apply_m1_constraints=True,
        zero_m1_after_m1_constraints=False,
        apply_scalxc=False,
        zero_edge_rz_blocks=False,
    )
    gn_order = prepare_residual_objective_blocks(
        frzl=frzl,
        s=jnp.asarray([0.0, 0.5, 1.0]),
        zero_m1_zforce=True,
        lconm1=True,
        apply_m1_constraints=True,
        zero_m1_after_m1_constraints=True,
        apply_scalxc=False,
        zero_edge_rz_blocks=False,
    )

    np.testing.assert_allclose(np.asarray(lbfgs_order.fzcs)[:, 1, :], 4.0 / np.sqrt(2.0))
    np.testing.assert_allclose(np.asarray(gn_order.fzcs)[:, 1, :], 0.0)
    np.testing.assert_allclose(np.asarray(lbfgs_order.frss)[:, 1, :], 4.0 / np.sqrt(2.0))
    np.testing.assert_allclose(np.asarray(gn_order.frss)[:, 1, :], 6.0 / np.sqrt(2.0))


def test_residual_objective_vector_stacks_optional_blocks_in_gn_order():
    frzl = TomnspsRZL(
        frcc=jnp.asarray([[[1.0]]]),
        frss=jnp.asarray([[[4.0]]]),
        fzsc=jnp.asarray([[[2.0]]]),
        fzcs=jnp.asarray([[[5.0]]]),
        flsc=jnp.asarray([[[3.0]]]),
        flcs=jnp.asarray([[[6.0]]]),
        frsc=jnp.asarray([[[7.0]]]),
        fzcc=jnp.asarray([[[8.0]]]),
        flcc=jnp.asarray([[[9.0]]]),
        frcs=jnp.asarray([[[10.0]]]),
        fzss=jnp.asarray([[[11.0]]]),
        flss=jnp.asarray([[[12.0]]]),
    )
    norms = SimpleNamespace(r1=jnp.asarray(4.0), fnorm=jnp.asarray(9.0), fnormL=jnp.asarray(16.0))

    vec = residual_objective_vector(frzl=frzl, norms=norms, w_rz=4.0, w_l=0.25)

    np.testing.assert_allclose(
        np.asarray(vec),
        np.asarray([12.0, 24.0, 6.0, 48.0, 60.0, 12.0, 84.0, 96.0, 18.0, 120.0, 132.0, 24.0]),
    )

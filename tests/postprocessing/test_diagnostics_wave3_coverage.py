from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.diagnostics import (
    _signed_to_mn_cos,
    _signed_to_mn_sin,
    _vmec_basis_norm,
    print_jacobian_stats,
    print_summary,
    slice_excluding_axis,
    summarize_array,
    summarize_many,
    vmec_internal_mn_from_state,
    vmec_xc_from_mn_blocks,
)
from vmec_jax.modes import ModeTable


def _modes() -> ModeTable:
    return ModeTable(
        m=np.asarray([0, 0, 1, 1, 1, 2], dtype=int),
        n=np.asarray([0, 1, -1, 1, 0, 0], dtype=int),
    )


def _state(ns: int = 2) -> SimpleNamespace:
    coeffs = np.arange(ns * _modes().K, dtype=float).reshape(ns, _modes().K) + 1.0
    return SimpleNamespace(
        Rcos=coeffs,
        Rsin=coeffs + 10.0,
        Zcos=coeffs + 20.0,
        Zsin=coeffs + 30.0,
        Lcos=coeffs + 40.0,
        Lsin=coeffs + 50.0,
    )


def _cfg(*, lasym: bool, lthreed: bool, lconm1: bool = True) -> SimpleNamespace:
    return SimpleNamespace(mpol=3, ntor=1, lasym=lasym, lthreed=lthreed, lconm1=lconm1)


def test_summary_printing_and_slice_helpers_cover_empty_nonfinite_and_counts(capsys) -> None:
    empty = summarize_array("empty", np.asarray([], dtype=float))
    assert empty.shape == (0,)
    assert np.isnan(empty.min)
    assert np.all(np.isnan(empty.q))

    ints = summarize_array("ints", np.asarray([[0, -2], [3, 0]], dtype=np.int64))
    assert ints.dtype == "int64"
    assert ints.n_nan == 0
    assert ints.n_inf == 0
    assert ints.n_zero == 2
    assert ints.n_neg == 1

    with pytest.warns(RuntimeWarning):
        nonfinite = summarize_array("nonfinite", np.asarray([np.nan, np.inf, -np.inf]))
    assert nonfinite.n_nan == 1
    assert nonfinite.n_inf == 2
    assert np.isnan(nonfinite.mean)

    mixed = summarize_array("mixed", np.asarray([np.nan, np.inf, -1.0, 0.0, 3.0]))
    assert mixed.min == -1.0
    assert mixed.max == 3.0
    assert mixed.n_nan == 1
    assert mixed.n_inf == 1
    assert mixed.n_zero == 1
    assert mixed.n_neg == 1

    print_summary(mixed, indent="> ")
    print_summary(empty)
    summarize_many([("a", [1.0, 2.0]), ("b", np.asarray([0.0]))], indent="--")
    print_jacobian_stats(np.asarray([-2.0, 0.0, 3.0]), indent="  ")
    captured = capsys.readouterr().out
    assert "> mixed:" in captured
    assert "counts: nan=1 inf=1 zero=1 neg=1" in captured
    assert "--a:" in captured
    assert "|sqrtg|" in captured

    scalar = slice_excluding_axis(3.0)
    assert scalar.shape == ()
    np.testing.assert_array_equal(slice_excluding_axis(np.asarray([4.0])), np.asarray([4.0]))
    np.testing.assert_array_equal(
        slice_excluding_axis(np.arange(6).reshape(2, 3), axis_dim=0),
        np.asarray([[3, 4, 5]]),
    )
    np.testing.assert_array_equal(
        slice_excluding_axis(np.arange(6).reshape(2, 3), axis_dim=1),
        np.asarray([[1, 2], [4, 5]]),
    )


def test_signed_mode_conversions_and_internal_blocks_cover_lasym_constraints_and_scaling() -> None:
    modes = _modes()
    coeffs = np.arange(12, dtype=float).reshape(2, 6) + 1.0

    rcc, rss = _signed_to_mn_cos(coeffs, modes=modes, mpol=3, ntor=1)
    np.testing.assert_allclose(rcc[:, 1, 1], coeffs[:, 3] + coeffs[:, 2])
    np.testing.assert_allclose(rss[:, 1, 1], coeffs[:, 3] - coeffs[:, 2])
    np.testing.assert_allclose(rss[:, 0, 1], 0.0)
    np.testing.assert_allclose(rss[:, 1, 0], 0.0)
    np.testing.assert_allclose(rcc[:, 2, 1], 0.0)

    zsc, zcs = _signed_to_mn_sin(coeffs, modes=modes, mpol=3, ntor=1)
    np.testing.assert_allclose(zsc[:, 1, 1], coeffs[:, 3] + coeffs[:, 2])
    np.testing.assert_allclose(zcs[:, 1, 1], coeffs[:, 2] - coeffs[:, 3])
    np.testing.assert_allclose(zcs[:, 1, 0], 0.0)
    np.testing.assert_allclose(zsc[:, 0, 1], 0.0)

    np.testing.assert_allclose(_vmec_basis_norm(mpol=1, ntor=0), np.ones((1, 1)))
    norm = _vmec_basis_norm(mpol=3, ntor=1)
    assert norm[0, 0] == 1.0
    assert np.isclose(norm[1, 0], 1.0 / np.sqrt(2.0))
    assert np.isclose(norm[1, 1], 0.5)

    static_lasym = SimpleNamespace(cfg=_cfg(lasym=True, lthreed=True), modes=modes)
    blocks = vmec_internal_mn_from_state(
        _state(),
        static_lasym,
        apply_basis_norm=False,
        apply_m1_constraint=True,
    )
    assert set(blocks) == {"rcc", "rss", "zsc", "zcs", "lsc", "lcs", "rsc", "rcs", "zcc", "zss", "lcc", "lss"}
    assert all(value.shape == (2, 3, 2) for value in blocks.values())
    assert not np.allclose(blocks["rss"][:, 1, :], blocks["zcs"][:, 1, :])

    static_sym = SimpleNamespace(cfg=_cfg(lasym=False, lthreed=False, lconm1=False), modes=modes)
    sym_blocks = vmec_internal_mn_from_state(
        _state(),
        static_sym,
        apply_basis_norm=True,
        apply_m1_constraint=True,
    )
    assert set(sym_blocks) == {"rcc", "rss", "zsc", "zcs", "lsc", "lcs"}
    np.testing.assert_allclose(sym_blocks["rcc"][:, 1, 1], rcc[:, 1, 1] * 0.5)


def test_xc_packing_covers_all_geometry_orderings_and_zero_filled_optional_blocks() -> None:
    ns = 2
    mpol = 2
    ntor = 1
    mnsize = mpol * (ntor + 1)
    mns = ns * mnsize

    def block(offset: float) -> np.ndarray:
        return offset + np.arange(mns, dtype=float).reshape(ns, mpol, ntor + 1)

    base_kwargs = dict(
        rcc=block(0.0),
        rss=block(10.0),
        zsc=block(20.0),
        zcs=block(30.0),
        lsc=block(40.0),
        lcs=block(50.0),
    )

    axisym = vmec_xc_from_mn_blocks(
        **base_kwargs,
        cfg=SimpleNamespace(mpol=mpol, ntor=ntor, lthreed=False, lasym=False),
    )
    assert axisym.shape == (3 * mns,)
    np.testing.assert_allclose(axisym[:mns], block(0.0).reshape(ns, mnsize).T.reshape(-1))

    axisym_lasym = vmec_xc_from_mn_blocks(
        **base_kwargs,
        rsc=None,
        rcs=np.asarray([]),
        zcc=block(60.0),
        lcc=block(70.0),
        cfg=SimpleNamespace(mpol=mpol, ntor=ntor, lthreed=False, lasym=True),
    )
    assert axisym_lasym.shape == (6 * mns,)
    np.testing.assert_allclose(axisym_lasym[mns : 2 * mns], 0.0)

    threed = vmec_xc_from_mn_blocks(
        **base_kwargs,
        cfg=SimpleNamespace(mpol=mpol, ntor=ntor, lthreed=True, lasym=False),
    )
    assert threed.shape == (6 * mns,)

    full = vmec_xc_from_mn_blocks(
        **base_kwargs,
        rsc=None,
        rcs=np.asarray([]),
        zcc=block(60.0),
        zss=block(70.0),
        lcc=block(80.0),
        lss=block(90.0),
        cfg=SimpleNamespace(mpol=mpol, ntor=ntor, lthreed=True, lasym=True),
    )
    assert full.shape == (12 * mns,)
    np.testing.assert_allclose(full[2 * mns : 4 * mns], 0.0)

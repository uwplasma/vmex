from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import vmec_jax.solve as solve
from vmec_jax.solve_residual_iter_geometry_helpers import (
    _m1_internal_to_physical_pair,
    _mn_sin_to_signed_physical_batch,
    _rz_norm_np,
)


def test_solve_reexports_extracted_geometry_helpers() -> None:
    assert solve._m1_internal_to_physical_pair is _m1_internal_to_physical_pair
    assert solve._mn_sin_to_signed_physical_batch is _mn_sin_to_signed_physical_batch
    assert solve._rz_norm_np is _rz_norm_np


def test_m1_internal_to_physical_pair_converts_only_m1_row_and_handles_none() -> None:
    rss = np.arange(12.0).reshape(2, 3, 2)
    zcs = 100.0 + np.arange(12.0).reshape(2, 3, 2)

    got_rss, got_zcs = _m1_internal_to_physical_pair(rss, zcs, use_m1_pair_convert=True)

    want_rss = rss.copy()
    want_zcs = zcs.copy()
    want_rss[:, 1, :] = rss[:, 1, :] + zcs[:, 1, :]
    want_zcs[:, 1, :] = rss[:, 1, :] - zcs[:, 1, :]
    np.testing.assert_allclose(np.asarray(got_rss), want_rss)
    np.testing.assert_allclose(np.asarray(got_zcs), want_zcs)

    passthrough_rss, passthrough_zcs = _m1_internal_to_physical_pair(rss, None, use_m1_pair_convert=False)
    np.testing.assert_allclose(np.asarray(passthrough_rss), rss)
    np.testing.assert_allclose(np.asarray(passthrough_zcs), np.zeros_like(rss))

    filled_rss, filled_zcs = _m1_internal_to_physical_pair(None, zcs, use_m1_pair_convert=False)
    np.testing.assert_allclose(np.asarray(filled_rss), np.zeros_like(zcs))
    np.testing.assert_allclose(np.asarray(filled_zcs), zcs)

    assert _m1_internal_to_physical_pair(None, None, use_m1_pair_convert=True) == (None, None)


def test_mn_sin_to_signed_physical_batch_scales_inputs_and_fills_absent_cs() -> None:
    sc = np.arange(12.0).reshape(2, 3, 2) + 2.0
    cs = np.arange(12.0).reshape(2, 3, 2) + 20.0
    scalxc_mn = np.asarray([[[1.0], [2.0], [4.0]], [[0.5], [5.0], [10.0]]])
    seen = {}

    def mapper(sc_scaled, cs_scaled):
        seen["sc"] = np.asarray(sc_scaled)
        seen["cs"] = np.asarray(cs_scaled)
        return sc_scaled + 2.0 * cs_scaled

    got = _mn_sin_to_signed_physical_batch(sc, cs, scalxc_mn=scalxc_mn, mn_sin_to_signed_batch=mapper)

    np.testing.assert_allclose(seen["sc"], sc / scalxc_mn)
    np.testing.assert_allclose(seen["cs"], cs / scalxc_mn)
    np.testing.assert_allclose(np.asarray(got), (sc / scalxc_mn) + 2.0 * (cs / scalxc_mn))

    got_no_cs = _mn_sin_to_signed_physical_batch(sc, None, scalxc_mn=scalxc_mn, mn_sin_to_signed_batch=mapper)
    np.testing.assert_allclose(seen["cs"], np.zeros_like(sc))
    np.testing.assert_allclose(np.asarray(got_no_cs), sc / scalxc_mn)


def test_rz_norm_np_matches_signed_mode_reference_and_excludes_axis() -> None:
    state = SimpleNamespace(
        Rcos=np.asarray(
            [
                [100.0, 200.0, 300.0, 400.0, 500.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        ),
        Zsin=np.asarray(
            [
                [600.0, 700.0, 800.0, 900.0, 1000.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
            ]
        ),
        Rsin=np.asarray(
            [
                [1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
                [26.0, 27.0, 28.0, 29.0, 30.0],
            ]
        ),
        Zcos=np.asarray(
            [
                [1600.0, 1700.0, 1800.0, 1900.0, 2000.0],
                [31.0, 32.0, 33.0, 34.0, 35.0],
                [36.0, 37.0, 38.0, 39.0, 40.0],
            ]
        ),
    )
    params = dict(
        kp_idx_np=np.asarray([0, 1, 3]),
        kn_idx_np=np.asarray([-1, 2, 4]),
        has_kn_np=np.asarray([False, True, True]),
        m_idx_np=np.asarray([0, 1, 0]),
        n_idx_np=np.asarray([0, 1, 1]),
        include_rcc_np=np.asarray([False, True, True]),
    )

    rcc = np.asarray([[5.0, 9.0], [15.0, 19.0]])
    zsc = np.asarray([[11.0, 25.0, 0.0], [16.0, 35.0, 0.0]])
    rss = np.asarray([[-1.0], [-1.0]])
    zcs = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    stellsym_expected = float(np.sum(zsc * zsc) + np.sum(rcc * rcc) + np.sum(rss * rss) + np.sum(zcs * zcs))

    assert _rz_norm_np(state, **params, lthreed=True, lasym=False) == stellsym_expected
    assert _rz_norm_np(state, **params, lthreed=False, lasym=False) == float(np.sum(zsc * zsc) + np.sum(rcc * rcc))

    rsc = np.asarray([[21.0, 45.0, 49.0], [26.0, 55.0, 59.0]])
    rcs = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    zcc = np.asarray([[31.0, 65.0, 69.0], [36.0, 75.0, 79.0]])
    zss = np.asarray([[-1.0], [-1.0]])
    lasym_expected = stellsym_expected + float(np.sum(rsc * rsc) + np.sum(rcs * rcs) + np.sum(zcc * zcc) + np.sum(zss * zss))

    assert _rz_norm_np(state, **params, lthreed=True, lasym=True) == lasym_expected

    state_axis_changed = SimpleNamespace(
        Rcos=np.asarray(state.Rcos).copy(),
        Zsin=np.asarray(state.Zsin).copy(),
        Rsin=np.asarray(state.Rsin).copy(),
        Zcos=np.asarray(state.Zcos).copy(),
    )
    for name in ("Rcos", "Zsin", "Rsin", "Zcos"):
        getattr(state_axis_changed, name)[0, :] *= -9.0
    assert _rz_norm_np(state_axis_changed, **params, lthreed=True, lasym=True) == lasym_expected

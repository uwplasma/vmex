from __future__ import annotations

from pathlib import Path

import numpy as np

from vmec_jax.config import load_config
from vmec_jax.solve import (
    _enforce_field_rows,
    _enforce_fixed_boundary_and_axis,
    _replace_mode_slice,
    _scale_mode_slice,
    _zero_coeff_column,
)
from vmec_jax.state import StateLayout, VMECState, zeros_state
from vmec_jax.static import build_static


def test_zero_coeff_column_matches_masking():
    arr = np.arange(12, dtype=float).reshape(3, 4)
    idx = 2
    got = np.asarray(_zero_coeff_column(arr, idx=idx))
    want = np.asarray(arr).copy()
    want[:, idx] = 0.0
    np.testing.assert_allclose(got, want)


def test_replace_and_scale_mode_slice_match_reference():
    arr = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    repl = np.full((2, 3), 7.0)
    got_repl = np.asarray(_replace_mode_slice(arr, mode_idx=1, replacement=repl))
    want_repl = np.asarray(arr).copy()
    want_repl[:, 1, :] = repl
    np.testing.assert_allclose(got_repl, want_repl)

    scale = np.asarray([2.0, 3.0], dtype=float)
    got_scale = np.asarray(_scale_mode_slice(arr, mode_idx=1, scale=scale))
    want_scale = np.asarray(arr).copy()
    want_scale[:, 1, :] *= scale[:, None]
    np.testing.assert_allclose(got_scale, want_scale)


def test_enforce_field_rows_matches_legacy_axis_and_edge():
    arr = np.arange(5 * 4, dtype=float).reshape(5, 4)
    mask = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float)
    edge = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=float)

    got = np.asarray(_enforce_field_rows(arr, axis_mask=mask, edge_row=edge))

    want = np.concatenate([arr[:-1, :], edge[None, :]], axis=0)
    want = np.concatenate([want[:1, :] * mask[None, :], want[1:, :]], axis=0)
    np.testing.assert_allclose(got, want)


def test_enforce_field_rows_matches_legacy_single_row():
    arr = np.arange(4, dtype=float).reshape(1, 4)
    mask = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float)
    edge = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=float)

    got = np.asarray(_enforce_field_rows(arr, axis_mask=mask, edge_row=edge))

    want = np.concatenate([arr[:-1, :], edge[None, :]], axis=0)
    want = np.concatenate([want[:1, :] * mask[None, :], want[1:, :]], axis=0)
    np.testing.assert_allclose(got, want)


def test_enforce_fixed_boundary_and_axis_matches_component_reference():
    root = Path(__file__).resolve().parents[1]
    cfg, _ = load_config(str(root / "examples/data/input.circular_tokamak"))
    static = build_static(cfg)
    layout = StateLayout(ns=cfg.ns, K=static.modes.m.size, lasym=cfg.lasym)
    state0 = zeros_state(layout)
    rng = np.random.default_rng(0)
    state = VMECState(
        layout=layout,
        Rcos=rng.standard_normal(state0.Rcos.shape),
        Rsin=rng.standard_normal(state0.Rsin.shape),
        Zcos=rng.standard_normal(state0.Zcos.shape),
        Zsin=rng.standard_normal(state0.Zsin.shape),
        Lcos=rng.standard_normal(state0.Lcos.shape),
        Lsin=rng.standard_normal(state0.Lsin.shape),
    )
    idx00 = int(np.where((np.asarray(static.modes.m) == 0) & (np.asarray(static.modes.n) == 0))[0][0])
    edge_Rcos = rng.standard_normal((layout.K,))
    edge_Rsin = rng.standard_normal((layout.K,))
    edge_Zcos = rng.standard_normal((layout.K,))
    edge_Zsin = rng.standard_normal((layout.K,))

    got = _enforce_fixed_boundary_and_axis(
        state,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_axis=True,
        enforce_edge=True,
        enforce_lambda_axis=True,
        idx00=idx00,
    )

    mask_m0 = np.asarray(static.m_is_m0, dtype=float)
    Rcos = np.asarray(state.Rcos)
    Rsin = np.asarray(state.Rsin)
    Zcos = np.asarray(state.Zcos)
    Zsin = np.asarray(state.Zsin)
    Lcos = np.asarray(state.Lcos)
    Lsin = np.asarray(state.Lsin)

    Rcos = np.concatenate([Rcos[:-1, :], edge_Rcos[None, :]], axis=0)
    Rsin = np.concatenate([Rsin[:-1, :], edge_Rsin[None, :]], axis=0)
    Zcos = np.concatenate([Zcos[:-1, :], edge_Zcos[None, :]], axis=0)
    Zsin = np.concatenate([Zsin[:-1, :], edge_Zsin[None, :]], axis=0)
    Rcos = np.concatenate([Rcos[:1, :] * mask_m0[None, :], Rcos[1:, :]], axis=0)
    Rsin = np.concatenate([Rsin[:1, :] * mask_m0[None, :], Rsin[1:, :]], axis=0)
    Zcos = np.concatenate([Zcos[:1, :] * mask_m0[None, :], Zcos[1:, :]], axis=0)
    Zsin = np.concatenate([Zsin[:1, :] * mask_m0[None, :], Zsin[1:, :]], axis=0)
    Lcos = np.concatenate([np.zeros_like(Lcos[:1, :]), Lcos[1:, :]], axis=0)
    Lsin = np.concatenate([np.zeros_like(Lsin[:1, :]), Lsin[1:, :]], axis=0)
    Lcos[:, idx00] = 0.0
    Lsin[:, idx00] = 0.0

    np.testing.assert_allclose(np.asarray(got.Rcos), Rcos)
    np.testing.assert_allclose(np.asarray(got.Rsin), Rsin)
    np.testing.assert_allclose(np.asarray(got.Zcos), Zcos)
    np.testing.assert_allclose(np.asarray(got.Zsin), Zsin)
    np.testing.assert_allclose(np.asarray(got.Lcos), Lcos)
    np.testing.assert_allclose(np.asarray(got.Lsin), Lsin)

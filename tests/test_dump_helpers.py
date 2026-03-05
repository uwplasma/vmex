from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_lam_prec_dump_axisym_lasym_uses_vmec_ntmax(tmp_path, monkeypatch):
    from vmec_jax.solve import _maybe_dump_lam_prec

    static = SimpleNamespace(cfg=SimpleNamespace(ns=4, mpol=3, ntor=0, lthreed=False, lasym=True))
    lam_prec = np.arange(12, dtype=float).reshape(4, 3, 1)

    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "7")

    _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=lam_prec, static=static, iter_idx=7)

    data = np.load(tmp_path / "lam_prec_ns4_iter7.npz")
    pfaclam = np.asarray(data["pfaclam"])
    faclam = np.asarray(data["faclam"])
    expected = np.transpose(lam_prec, (0, 2, 1))

    assert pfaclam.shape == (4, 1, 3, 2)
    assert faclam.shape == (4, 1, 3, 2)
    assert np.allclose(pfaclam[:, :, :, 0], expected)
    assert np.allclose(faclam[:, :, :, 0], expected)
    assert np.allclose(pfaclam[:, :, 1:, 1], expected[:, :, 1:])
    assert np.allclose(faclam[:, :, 1:, 1], expected[:, :, 1:])
    assert np.allclose(pfaclam[:, 0, 0, 1], 0.0)
    assert np.allclose(faclam[:, 0, 0, 1], 0.0)


def test_precond_mats_dump_writes_matrix_channels(tmp_path, monkeypatch):
    from vmec_jax.solve import _maybe_dump_precond_mats

    static = SimpleNamespace(cfg=SimpleNamespace(ns=5, mpol=4, ntor=1, lthreed=True, lasym=True))
    mats = {
        "ar": np.arange(40, dtype=float).reshape(5, 4, 2),
        "br": np.arange(40, 80, dtype=float).reshape(5, 4, 2),
        "dr": np.arange(80, 120, dtype=float).reshape(5, 4, 2),
        "az": np.arange(120, 160, dtype=float).reshape(5, 4, 2),
        "bz": np.arange(160, 200, dtype=float).reshape(5, 4, 2),
        "dz": np.arange(200, 240, dtype=float).reshape(5, 4, 2),
    }

    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND_MATS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "11")

    _maybe_dump_precond_mats(mats=mats, static=static, iter_idx=11, jmax=5, used_cache=False)

    data = np.load(tmp_path / "precond_mats_ns5_iter11.npz")

    assert int(data["jmax"]) == 5
    assert bool(data["used_cache"]) is False
    assert bool(data["lthreed"]) is True
    assert bool(data["lasym"]) is True
    for key, arr in mats.items():
        assert np.array_equal(np.asarray(data[key]), arr)


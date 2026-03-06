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


def test_gmetric_dump_writes_half_mesh_metric(tmp_path, monkeypatch):
    from vmec_jax.solve import _maybe_dump_gmetric

    static = SimpleNamespace(cfg=SimpleNamespace(ns=3))
    bc = SimpleNamespace(
        guu=np.arange(12, dtype=float).reshape(3, 2, 2),
        guv=np.arange(12, 24, dtype=float).reshape(3, 2, 2),
        gvv=np.arange(24, 36, dtype=float).reshape(3, 2, 2),
        jac=SimpleNamespace(r12=np.ones((3, 2, 2), dtype=float) * 2.0),
    )

    monkeypatch.setenv("VMEC_JAX_DUMP_GMETRIC", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "13")

    _maybe_dump_gmetric(bc=bc, static=static, iter_idx=13)

    path = tmp_path / "gmetric_iter13.dat"
    lines = path.read_text().splitlines()

    assert lines[:5] == [
        "# bcovar metric dump (half mesh)",
        "ns=3",
        "ntheta3=2",
        "nzeta=2",
        "columns: js lt lz pguu pguv pgvv",
    ]
    assert lines[5].split() == ["1", "1", "1", "0.0000000000000000e+00", "0.0000000000000000e+00", "0.0000000000000000e+00"]
    assert lines[6].split() == ["2", "1", "1", "4.0000000000000000e+00", "1.6000000000000000e+01", "2.4000000000000000e+01"]
    assert lines[-1].split() == ["3", "2", "2", "1.1000000000000000e+01", "2.3000000000000000e+01", "3.1000000000000000e+01"]


def test_vmec_scale_m1_factors_prefer_parity_diagonals():
    from vmec_jax.solve import _vmec_scale_m1_factors_from_mats

    mats = {
        "dr": np.array(
            [
                [[-2.0], [-20.0]],
                [[-3.0], [-99.0]],
                [[-4.0], [-40.0]],
            ],
            dtype=float,
        ),
        "dz": np.array(
            [
                [[-6.0], [-60.0]],
                [[-7.0], [-88.0]],
                [[-8.0], [-80.0]],
            ],
            dtype=float,
        ),
        "ard_parity": np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float),
        "brd_parity": np.array([[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]], dtype=float),
        "azd_parity": np.array([[7.0, 70.0], [8.0, 80.0], [9.0, 90.0]], dtype=float),
        "bzd_parity": np.array([[10.0, 100.0], [11.0, 110.0], [12.0, 120.0]], dtype=float),
    }

    fac_r, fac_z = _vmec_scale_m1_factors_from_mats(mats)

    sr = np.array([50.0, 70.0, 90.0])
    sz = np.array([170.0, 190.0, 210.0])
    denom = sr + sz
    assert np.allclose(fac_r, sr / denom)
    assert np.allclose(fac_z, sz / denom)


def test_vmec_scale_m1_factors_fall_back_to_expanded_diagonals():
    from vmec_jax.solve import _vmec_scale_m1_factors_from_mats

    mats = {
        "dr": np.array(
            [
                [[-2.0], [-20.0]],
                [[-3.0], [-30.0]],
            ],
            dtype=float,
        ),
        "dz": np.array(
            [
                [[-6.0], [-60.0]],
                [[-7.0], [-70.0]],
            ],
            dtype=float,
        ),
    }

    fac_r, fac_z = _vmec_scale_m1_factors_from_mats(mats)

    assert np.allclose(fac_r, np.array([0.25, 0.3]))
    assert np.allclose(fac_z, np.array([0.75, 0.7]))


def test_precond_inputs_dump_writes_hidden_kernel_channels(tmp_path, monkeypatch):
    from vmec_jax.solve import _maybe_dump_precond_inputs

    static = SimpleNamespace(cfg=SimpleNamespace(ns=3))
    bc = SimpleNamespace(
        bsq=np.arange(12, dtype=float).reshape(3, 2, 2),
        jac=SimpleNamespace(
            r12=np.arange(12, 24, dtype=float).reshape(3, 2, 2),
            sqrtg=np.arange(24, 36, dtype=float).reshape(3, 2, 2),
            ru12=np.arange(36, 48, dtype=float).reshape(3, 2, 2),
            zu12=np.arange(48, 60, dtype=float).reshape(3, 2, 2),
            tau=np.arange(60, 72, dtype=float).reshape(3, 2, 2),
            rs=np.arange(72, 84, dtype=float).reshape(3, 2, 2),
            zs=np.arange(84, 96, dtype=float).reshape(3, 2, 2),
        ),
    )
    trig = SimpleNamespace(wint3_precond=np.ones((1, 2, 2), dtype=float) * 0.25)
    kernels = SimpleNamespace(
        pru_even=np.arange(96, 108, dtype=float).reshape(3, 2, 2),
        pru_odd=np.arange(108, 120, dtype=float).reshape(3, 2, 2),
        pzu_even=np.arange(120, 132, dtype=float).reshape(3, 2, 2),
        pzu_odd=np.arange(132, 144, dtype=float).reshape(3, 2, 2),
        pr1_odd=np.arange(144, 156, dtype=float).reshape(3, 2, 2),
        pz1_odd=np.arange(156, 168, dtype=float).reshape(3, 2, 2),
    )

    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "17")

    _maybe_dump_precond_inputs(bc=bc, trig=trig, static=static, iter_idx=17, kernels=kernels)

    path = tmp_path / "precond_hidden_iter17.npz"
    data = np.load(path)

    expected = {
        "tau": bc.jac.tau,
        "rs": bc.jac.rs,
        "zs": bc.jac.zs,
        "pru_even": kernels.pru_even,
        "pru_odd": kernels.pru_odd,
        "pzu_even": kernels.pzu_even,
        "pzu_odd": kernels.pzu_odd,
        "pr1_odd": kernels.pr1_odd,
        "pz1_odd": kernels.pz1_odd,
    }
    for key, arr in expected.items():
        assert np.array_equal(np.asarray(data[key]), arr)

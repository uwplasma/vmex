from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.modes import ModeTable
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.vmec_tomnsp import vmec_trig_tables
from vmec_jax.wout import _compute_bsubs_half_mesh, _vmec_wrout_nyquist_lasym_loop


def _state_and_modes(ns: int = 3) -> tuple[VMECState, ModeTable]:
    modes = ModeTable(m=np.asarray([0, 1], dtype=int), n=np.asarray([0, 0], dtype=int))
    layout = StateLayout(ns=ns, K=modes.K, lasym=False)
    radial = np.linspace(0.0, 1.0, ns)[:, None]
    Rcos = np.concatenate([1.0 + 0.2 * radial, 0.15 + 0.05 * radial], axis=1)
    Zsin = np.concatenate([0.2 + 0.1 * radial, 0.3 + 0.07 * radial], axis=1)
    zeros = np.zeros_like(Rcos)
    state = VMECState(layout=layout, Rcos=Rcos, Rsin=zeros.copy(), Zcos=zeros.copy(), Zsin=Zsin, Lcos=zeros, Lsin=zeros)
    return state, modes


def _grid_field(shape: tuple[int, int, int], offset: float, scale: float = 0.01) -> np.ndarray:
    return offset + scale * np.arange(int(np.prod(shape)), dtype=float).reshape(shape)


def test_compute_bsubs_half_mesh_covers_parity_force_jacobian_and_dump_paths(monkeypatch, tmp_path) -> None:
    state, modes = _state_and_modes()
    trig = vmec_trig_tables(ntheta=4, nzeta=2, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)
    s = np.asarray([0.0, 0.25, 1.0])
    shape = (s.size, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))
    bsupu = _grid_field(shape, 0.4)
    bsupv = _grid_field(shape, -0.2)

    parity_geom = {
        "pr1_even": _grid_field(shape, 1.0),
        "pr1_odd": _grid_field(shape, 0.2),
        "pz1_even": _grid_field(shape, -0.3),
        "pz1_odd": _grid_field(shape, 0.4),
        "pru_even": _grid_field(shape, 0.5),
        "pru_odd": _grid_field(shape, -0.1),
        "pzu_even": _grid_field(shape, 0.7),
        "pzu_odd": _grid_field(shape, 0.3),
        "prv_even": _grid_field(shape, -0.5),
        "prv_odd": _grid_field(shape, 0.6),
        "pzv_even": _grid_field(shape, 0.8),
        "pzv_odd": _grid_field(shape, -0.4),
    }

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_TAG", "tiny")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSS_INPUTS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_BSS_TERMS", "1")
    parity_out = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom=parity_geom,
    )
    assert parity_out.shape == shape
    assert np.all(np.isfinite(parity_out))
    assert (tmp_path / "bss_inputs_jax_tiny.dat").exists()
    with np.load(tmp_path / "bss_terms_jax_tiny.npz") as dump:
        assert dump["bsubs"].shape == shape
        assert dump["gsu"].shape == shape

    monkeypatch.delenv("VMEC_JAX_DUMP_BSS_INPUTS", raising=False)
    monkeypatch.delenv("VMEC_JAX_DUMP_BSS_TERMS", raising=False)

    force_out = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom={},
        force_rs=_grid_field(shape, 0.11),
        force_zs=_grid_field(shape, 0.13),
        force_ru12=_grid_field(shape, 0.17),
        force_zu12=_grid_field(shape, 0.19),
    )
    assert force_out.shape == shape
    assert float(np.linalg.norm(force_out)) > 0.0

    jac = SimpleNamespace(
        ru12=_grid_field(shape, 0.21),
        zu12=_grid_field(shape, 0.23),
        rs=_grid_field(shape, 0.29),
        zs=_grid_field(shape, 0.31),
    )
    jac_out = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom={},
        jac_half=jac,
    )
    assert jac_out.shape == shape
    assert float(np.linalg.norm(jac_out)) > 0.0

    fallback_out = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=modes,
        s=s,
        lconm1=False,
        lthreed=False,
        lasym=False,
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom={},
    )
    assert fallback_out.shape == shape
    assert np.all(np.isfinite(fallback_out))


def test_vmec_wrout_nyquist_lasym_loop_covers_symmetric_asymmetric_channels() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    modes = ModeTable(m=np.asarray([0, 1, 1, 2], dtype=int), n=np.asarray([0, -1, 1, 0], dtype=int))
    ns = 3
    shape = (ns, int(trig.ntheta2), int(np.asarray(trig.cosnv).shape[0]))
    base = _grid_field(shape, -0.4, scale=0.03)

    coeffs = _vmec_wrout_nyquist_lasym_loop(
        bsq=1.5 + base,
        gsqrt=2.0 + base,
        bsubu=0.2 + base,
        bsubv=-0.1 + base,
        bsubs=0.3 + base,
        bsupu=0.4 + base,
        bsupv=0.5 + base,
        modes=modes,
        trig=trig,
    )

    expected_keys = {
        "gmnc",
        "bmnc",
        "bsubumnc",
        "bsubvmnc",
        "bsubsmns",
        "bsupumnc",
        "bsupvmnc",
        "gmns",
        "bmns",
        "bsubumns",
        "bsubvmns",
        "bsubsmnc",
        "bsupumns",
        "bsupvmns",
    }
    assert set(coeffs) == expected_keys
    for value in coeffs.values():
        assert value.shape == (ns, modes.K)
        assert np.all(np.isfinite(value))
    np.testing.assert_allclose(coeffs["gmnc"][0], 0.0)
    np.testing.assert_allclose(coeffs["gmns"][0], 0.0)
    np.testing.assert_allclose(coeffs["bsubsmns"][0], 2.0 * coeffs["bsubsmns"][1] - coeffs["bsubsmns"][2])
    np.testing.assert_allclose(coeffs["bsubsmnc"][0], 2.0 * coeffs["bsubsmnc"][1] - coeffs["bsubsmnc"][2])

    empty = _vmec_wrout_nyquist_lasym_loop(
        bsq=1.5 + base,
        gsqrt=2.0 + base,
        bsubu=0.2 + base,
        bsubv=-0.1 + base,
        bsubs=0.3 + base,
        bsupu=0.4 + base,
        bsupv=0.5 + base,
        modes=ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int)),
        trig=trig,
    )
    assert all(value.shape == (ns, 0) for value in empty.values())

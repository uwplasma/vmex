from __future__ import annotations

from pathlib import Path

import numpy as np

from vmec_jax.config import load_config
from vmec_jax.fourier import build_helical_basis, eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable, vmec_mode_table
from vmec_jax.static import build_static
from vmec_jax.kernels.parity import (
    _mn_cos_to_signed,
    _mn_cos_to_signed_cached,
    _mn_cos_to_signed_host,
    _mn_index_maps,
    _mn_sin_to_signed,
    _mn_sin_to_signed_cached,
    _mn_sin_to_signed_host,
    _signed_to_mn_cos,
    _signed_to_mn_cos_cached,
    _signed_to_mn_cos_host,
    _signed_to_mn_sin,
    _signed_to_mn_sin_cached,
    _signed_to_mn_sin_host,
    split_rzl_even_odd_lasym,
    signed_maps_from_modes,
    vmec_m1_internal_to_physical_signed,
    vmec_m1_internal_to_physical_signed_host,
    vmec_m1_physical_to_internal_signed,
)


def _rng_arrays(*, ns: int, ncoeff: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((ns, ncoeff)),
        rng.standard_normal((ns, ncoeff)),
        rng.standard_normal((ns, ncoeff)),
        rng.standard_normal((ns, ncoeff)),
    )


def test_host_signed_to_mn_matches_jax_cached():
    root = Path(__file__).resolve().parents[1]
    cfg, _ = load_config(str(root / "examples/data/input.basic_non_stellsym_pressure"))
    static = build_static(cfg)
    maps = signed_maps_from_modes(static.modes)
    coeffs = np.random.default_rng(1).standard_normal((cfg.ns, static.modes.m.size))

    rcc_j, rss_j = _signed_to_mn_cos_cached(coeffs, maps=maps)
    rcc_h, rss_h = _signed_to_mn_cos_host(coeffs, maps=maps)
    np.testing.assert_allclose(np.asarray(rcc_h), np.asarray(rcc_j), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rss_h), np.asarray(rss_j), rtol=1e-12, atol=1e-12)

    sc_j, cs_j = _signed_to_mn_sin_cached(coeffs, maps=maps)
    sc_h, cs_h = _signed_to_mn_sin_host(coeffs, maps=maps)
    np.testing.assert_allclose(np.asarray(sc_h), np.asarray(sc_j), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cs_h), np.asarray(cs_j), rtol=1e-12, atol=1e-12)


def test_host_m1_internal_to_physical_matches_jax_lasym_false():
    root = Path(__file__).resolve().parents[1]
    cfg, _ = load_config(str(root / "examples/data/input.LandremanPaul2021_QA_lowres"))
    static = build_static(cfg)
    arrays = _rng_arrays(ns=cfg.ns, ncoeff=static.modes.m.size, seed=2)

    got = vmec_m1_internal_to_physical_signed(
        Rcos=arrays[0],
        Zsin=arrays[1],
        Rsin=arrays[2],
        Zcos=arrays[3],
        modes=static.modes,
        lthreed=bool(cfg.lthreed),
        lasym=bool(cfg.lasym),
        lconm1=bool(cfg.lconm1),
    )
    host = vmec_m1_internal_to_physical_signed_host(
        Rcos=arrays[0],
        Zsin=arrays[1],
        Rsin=arrays[2],
        Zcos=arrays[3],
        modes=static.modes,
        lthreed=bool(cfg.lthreed),
        lasym=bool(cfg.lasym),
        lconm1=bool(cfg.lconm1),
    )
    for a, b in zip(host, got):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


def test_host_m1_internal_to_physical_matches_jax_lasym_true():
    root = Path(__file__).resolve().parents[1]
    cfg, _ = load_config(str(root / "examples/data/input.basic_non_stellsym_pressure"))
    static = build_static(cfg)
    arrays = _rng_arrays(ns=cfg.ns, ncoeff=static.modes.m.size, seed=3)

    got = vmec_m1_internal_to_physical_signed(
        Rcos=arrays[0],
        Zsin=arrays[1],
        Rsin=arrays[2],
        Zcos=arrays[3],
        modes=static.modes,
        lthreed=bool(cfg.lthreed),
        lasym=bool(cfg.lasym),
        lconm1=bool(cfg.lconm1),
    )
    host = vmec_m1_internal_to_physical_signed_host(
        Rcos=arrays[0],
        Zsin=arrays[1],
        Rsin=arrays[2],
        Zcos=arrays[3],
        modes=static.modes,
        lthreed=bool(cfg.lthreed),
        lasym=bool(cfg.lasym),
        lconm1=bool(cfg.lconm1),
    )
    for a, b in zip(host, got):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


def test_lasym_realspace_split_recombines_cosine_and_sine_sectors():
    modes = ModeTable(m=np.array([0, 1, 2]), n=np.array([0, 1, -1]))
    grid = AngleGrid(
        theta=np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False),
        zeta=np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False),
        nfp=2,
    )
    basis = build_helical_basis(modes, grid, cache=False)
    state = type(
        "State",
        (),
        {
            "Rcos": np.array([1.0, 0.2, -0.1]),
            "Zcos": np.array([0.5, -0.3, 0.4]),
            "Lcos": np.array([0.0, 0.1, 0.2]),
            "Rsin": np.array([0.0, 0.7, -0.2]),
            "Zsin": np.array([0.0, -0.4, 0.3]),
            "Lsin": np.array([0.0, 0.2, -0.5]),
        },
    )()

    split = split_rzl_even_odd_lasym(state, basis)

    full_r = eval_fourier(state.Rcos, state.Rsin, basis, coeffs_internal=True)
    full_rt = eval_fourier_dtheta(state.Rcos, state.Rsin, basis, coeffs_internal=True)
    full_rp = eval_fourier_dzeta_phys(state.Rcos, state.Rsin, basis, coeffs_internal=True)
    np.testing.assert_allclose(np.asarray(split.R_even + split.R_odd), np.asarray(full_r), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(split.Rt_even + split.Rt_odd), np.asarray(full_rt), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(split.Rp_even + split.Rp_odd), np.asarray(full_rp), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(split.Z_even + split.Z_odd),
        np.asarray(eval_fourier(state.Zcos, state.Zsin, basis, coeffs_internal=True)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(split.L_even + split.L_odd),
        np.asarray(eval_fourier(state.Lcos, state.Lsin, basis, coeffs_internal=True)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_signed_mode_uncached_and_cached_round_trips_with_missing_negative_modes():
    modes = ModeTable(
        m=np.array([0, 0, 1, 1, 1, 2]),
        n=np.array([0, 1, -1, 0, 1, 1]),
    )
    _mpol, _ntor, idx_pos, idx_neg = _mn_index_maps(modes)
    maps = signed_maps_from_modes(modes)

    coeffs = np.arange(12.0).reshape(2, 6) + 1.0
    coeffs[:, 1] = 0.0  # m=0 sine-like n>0 content is constrained away by VMEC.

    rcc, rss = _signed_to_mn_cos_cached(coeffs, maps=maps)
    rcc_raw, rss_raw = _signed_to_mn_cos_cached(coeffs.astype(np.float32), maps=maps)
    rcc_uncached, rss_uncached = _signed_to_mn_cos(coeffs, idx_pos, idx_neg)
    rcc_direct, rss_direct = _signed_to_mn_cos_host(coeffs, maps=maps)
    np.testing.assert_allclose(np.asarray(rcc), np.asarray(rcc_uncached), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rss), np.asarray(rss_uncached), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rcc), np.asarray(rcc_direct), rtol=1e-12, atol=1e-12)
    assert np.asarray(rcc_raw).dtype == np.float32

    sc, cs = _signed_to_mn_sin_cached(coeffs, maps=maps)
    sc_direct, cs_direct = _signed_to_mn_sin(coeffs, idx_pos, idx_neg)
    sc_host, cs_host = _signed_to_mn_sin_host(coeffs, maps=maps)
    np.testing.assert_allclose(np.asarray(sc), np.asarray(sc_direct), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cs), np.asarray(cs_direct), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sc), np.asarray(sc_host), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cs), np.asarray(cs_host), rtol=1e-12, atol=1e-12)

    cos_back = _mn_cos_to_signed(rcc, rss, idx_pos, idx_neg, ncoeff=coeffs.shape[1])
    cos_back_cached = _mn_cos_to_signed_cached(rcc, rss, maps=maps, ncoeff=coeffs.shape[1])
    cos_back_host = _mn_cos_to_signed_host(np.asarray(rcc), np.asarray(rss), maps=maps, ncoeff=coeffs.shape[1])
    sin_back = _mn_sin_to_signed(sc, cs, idx_pos, idx_neg, ncoeff=coeffs.shape[1])
    sin_back_cached = _mn_sin_to_signed_cached(sc, cs, maps=maps, ncoeff=coeffs.shape[1])
    sin_back_host = _mn_sin_to_signed_host(np.asarray(sc), np.asarray(cs), maps=maps, ncoeff=coeffs.shape[1])

    np.testing.assert_allclose(np.asarray(cos_back), coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cos_back_cached), coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cos_back_host[:, 1:], coeffs[:, 1:], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sin_back), coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sin_back_cached), coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sin_back_host[:, 1:], coeffs[:, 1:], rtol=1e-12, atol=1e-12)

    empty_modes = ModeTable(m=np.array([], dtype=int), n=np.array([], dtype=int))
    assert _mn_index_maps(empty_modes)[0:2] == (0, 0)
    empty_maps = signed_maps_from_modes(empty_modes)
    empty_coeffs = np.zeros((2, 0))
    assert np.asarray(_signed_to_mn_cos(empty_coeffs, np.zeros((0, 0)), np.zeros((0, 0)))[0]).shape == (2, 0, 0)
    assert np.asarray(_signed_to_mn_sin(empty_coeffs, np.zeros((0, 0)), np.zeros((0, 0)))[0]).shape == (2, 0, 0)
    assert np.asarray(_signed_to_mn_cos_cached(empty_coeffs, maps=empty_maps)[0]).shape == (2, 0, 0)
    assert np.asarray(_signed_to_mn_sin_cached(empty_coeffs, maps=empty_maps)[0]).shape == (2, 0, 0)
    assert _signed_to_mn_cos_host(empty_coeffs, maps=empty_maps)[0].shape == (2, 0, 0)
    assert _signed_to_mn_sin_host(empty_coeffs, maps=empty_maps)[0].shape == (2, 0, 0)
    assert _mn_cos_to_signed(np.zeros((2, 0, 0)), np.zeros((2, 0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), 0).shape == (
        2,
        0,
    )
    assert _mn_sin_to_signed(np.zeros((2, 0, 0)), np.zeros((2, 0, 0)), np.zeros((0, 0)), np.zeros((0, 0)), 0).shape == (
        2,
        0,
    )

    nonempty_maps = signed_maps_from_modes(vmec_mode_table(2, 1))
    rcc0 = np.zeros((2, nonempty_maps.mpol, nonempty_maps.nrange))
    rss0 = np.zeros_like(rcc0)
    assert np.asarray(_mn_cos_to_signed_cached(rcc0, rss0, maps=nonempty_maps, ncoeff=0)).shape == (2, 0)
    assert _mn_cos_to_signed_host(rcc0, rss0, maps=nonempty_maps, ncoeff=0).shape == (2, 0)
    assert np.asarray(_mn_sin_to_signed_cached(rcc0, rss0, maps=nonempty_maps, ncoeff=0)).shape == (2, 0)
    assert _mn_sin_to_signed_host(rcc0, rss0, maps=nonempty_maps, ncoeff=0).shape == (2, 0)


def test_m1_physical_internal_signed_conversion_is_invertible_and_has_early_returns():
    modes = vmec_mode_table(3, 1)
    arrays = _rng_arrays(ns=4, ncoeff=modes.m.size, seed=4)

    internal = vmec_m1_physical_to_internal_signed(
        Rcos=arrays[0],
        Zsin=arrays[1],
        Rsin=arrays[2],
        Zcos=arrays[3],
        modes=modes,
        lthreed=True,
        lasym=True,
        lconm1=True,
    )
    physical = vmec_m1_internal_to_physical_signed(
        Rcos=internal[0],
        Zsin=internal[1],
        Rsin=internal[2],
        Zcos=internal[3],
        modes=modes,
        lthreed=True,
        lasym=True,
        lconm1=True,
    )
    for expected, got in zip(arrays, physical):
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-12, atol=1e-12)

    unchanged = vmec_m1_physical_to_internal_signed(
        Rcos=arrays[0],
        Zsin=arrays[1],
        Rsin=arrays[2],
        Zcos=arrays[3],
        modes=modes,
        lthreed=False,
        lasym=False,
        lconm1=True,
    )
    for expected, got in zip(arrays, unchanged):
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-12, atol=1e-12)

    m0_modes = vmec_mode_table(1, 0)
    small = _rng_arrays(ns=2, ncoeff=m0_modes.m.size, seed=5)
    early_jax = vmec_m1_internal_to_physical_signed(
        Rcos=small[0],
        Zsin=small[1],
        Rsin=small[2],
        Zcos=small[3],
        modes=m0_modes,
        lthreed=True,
        lasym=False,
        lconm1=True,
    )
    early_host = vmec_m1_internal_to_physical_signed_host(
        Rcos=small[0],
        Zsin=small[1],
        Rsin=small[2],
        Zcos=small[3],
        modes=m0_modes,
        lthreed=True,
        lasym=False,
        lconm1=True,
    )
    early_internal = vmec_m1_physical_to_internal_signed(
        Rcos=small[0],
        Zsin=small[1],
        Rsin=small[2],
        Zcos=small[3],
        modes=m0_modes,
        lthreed=True,
        lasym=False,
        lconm1=True,
    )
    for expected, got_jax, got_host, got_internal in zip(small, early_jax, early_host, early_internal):
        np.testing.assert_allclose(np.asarray(got_jax), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(got_host), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(got_internal), expected, rtol=1e-12, atol=1e-12)

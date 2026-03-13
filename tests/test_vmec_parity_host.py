from __future__ import annotations

from pathlib import Path

import numpy as np

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.vmec_parity import (
    _signed_to_mn_cos_cached,
    _signed_to_mn_cos_host,
    _signed_to_mn_sin_cached,
    _signed_to_mn_sin_host,
    signed_maps_from_modes,
    vmec_m1_internal_to_physical_signed,
    vmec_m1_internal_to_physical_signed_host,
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

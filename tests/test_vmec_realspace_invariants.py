from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import enable_x64
from vmec_jax.modes import vmec_mode_table
from vmec_jax.kernels.realspace import (
    vmec_realspace_analysis,
    vmec_realspace_geom_from_state,
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
    vmec_realspace_synthesis_dzeta_phys,
    vmec_realspace_synthesis_multi,
)
from vmec_jax.kernels.tomnsp import vmec_trig_tables


def _deterministic_coefficients(ns: int, nmodes: int) -> tuple[np.ndarray, np.ndarray]:
    surfaces = np.arange(ns, dtype=float)[:, None]
    modes = np.arange(nmodes, dtype=float)[None, :]
    signs = np.where((np.arange(nmodes) % 2) == 0, 1.0, -1.0)[None, :]

    coeff_cos = signs * (0.03 + 0.005 * surfaces + 0.002 * modes)
    coeff_sin = -signs * (0.01 + 0.003 * surfaces + 0.001 * modes)
    coeff_sin[:, 0] = 0.0
    return coeff_cos, coeff_sin


def _odd_m_scalxc_for_default_s(ns: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, ns)
    sqrts = np.sqrt(np.maximum(s, 0.0))
    sqrts[-1] = 1.0
    sq2 = sqrts[1] if ns >= 2 else 1.0
    return 1.0 / np.maximum(sqrts, sq2)


def test_lasym_full_grid_analysis_roundtrips_both_parities():
    enable_x64()

    ns = 3
    mpol = 4
    ntor = 2
    nfp = 3
    modes = vmec_mode_table(mpol, ntor)
    trig = vmec_trig_tables(
        ntheta=16,
        nzeta=9,
        nfp=nfp,
        mmax=mpol - 1,
        nmax=ntor,
        lasym=True,
        dtype=np.float64,
        cache=False,
    )
    coeff_cos, coeff_sin = _deterministic_coefficients(ns, modes.K)

    realspace = vmec_realspace_synthesis(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
    )
    coeff_cos_back, coeff_sin_back = vmec_realspace_analysis(
        f=realspace,
        modes=modes,
        trig=trig,
        parity="both",
    )

    np.testing.assert_allclose(np.asarray(coeff_cos_back), coeff_cos, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(coeff_sin_back), coeff_sin, rtol=1e-13, atol=1e-13)


def test_helical_mixed_partials_commute_on_physical_zeta_derivative():
    enable_x64()

    ns = 2
    mpol = 4
    ntor = 2
    nfp = 2
    modes = vmec_mode_table(mpol, ntor)
    trig = vmec_trig_tables(
        ntheta=14,
        nzeta=7,
        nfp=nfp,
        mmax=mpol - 1,
        nmax=ntor,
        lasym=True,
        dtype=np.float64,
        cache=False,
    )
    coeff_cos, coeff_sin = _deterministic_coefficients(ns, modes.K)

    _, dtheta_multi, dzeta_multi = vmec_realspace_synthesis_multi(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
        derivs=("base", "dtheta", "dzeta"),
    )
    np.testing.assert_allclose(
        np.asarray(dtheta_multi),
        np.asarray(
            vmec_realspace_synthesis_dtheta(
                coeff_cos=coeff_cos,
                coeff_sin=coeff_sin,
                modes=modes,
                trig=trig,
            )
        ),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(dzeta_multi),
        np.asarray(
            vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=coeff_cos,
                coeff_sin=coeff_sin,
                modes=modes,
                trig=trig,
            )
        ),
        rtol=1e-13,
        atol=1e-13,
    )

    m = modes.m[None, :]
    n_phys = (modes.n * nfp)[None, :]
    coeff_cos_theta = m * coeff_sin
    coeff_sin_theta = -m * coeff_cos
    coeff_cos_zeta = -n_phys * coeff_sin
    coeff_sin_zeta = n_phys * coeff_cos

    dzeta_dtheta = vmec_realspace_synthesis_dzeta_phys(
        coeff_cos=coeff_cos_theta,
        coeff_sin=coeff_sin_theta,
        modes=modes,
        trig=trig,
    )
    dtheta_dzeta = vmec_realspace_synthesis_dtheta(
        coeff_cos=coeff_cos_zeta,
        coeff_sin=coeff_sin_zeta,
        modes=modes,
        trig=trig,
    )

    np.testing.assert_allclose(np.asarray(dzeta_dtheta), np.asarray(dtheta_dzeta), rtol=1e-13, atol=1e-13)


def test_single_surface_multiderivative_fallback_matches_stacked_scalxc_path():
    enable_x64()

    ns = 1
    mpol = 3
    ntor = 1
    nfp = 2
    modes = vmec_mode_table(mpol, ntor)
    trig = vmec_trig_tables(
        ntheta=12,
        nzeta=5,
        nfp=nfp,
        mmax=mpol - 1,
        nmax=ntor,
        lasym=True,
        dtype=np.float64,
        cache=False,
    )
    coeff_cos, coeff_sin = _deterministic_coefficients(ns, modes.K)

    stacked = vmec_realspace_synthesis_multi(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
        derivs=("base", "dtheta", "dzeta"),
        apply_scalxc=True,
        use_stacked_dot=True,
    )
    fallback = vmec_realspace_synthesis_multi(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
        derivs=("base", "dtheta", "dzeta"),
        apply_scalxc=True,
        use_stacked_dot=False,
    )

    for actual, expected in zip(fallback, stacked, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-13, atol=1e-13)

    base, dtheta, dzeta = stacked
    assert np.max(np.abs(np.asarray(dtheta))) > 0.0
    assert np.max(np.abs(np.asarray(dzeta))) > 0.0
    assert np.asarray(base).shape == (ns, trig.ntheta3, trig.cosnv.shape[0])


def test_geom_from_state_preserves_circular_axisymmetric_identities():
    enable_x64()

    ns = 4
    mpol = 2
    ntor = 0
    modes = vmec_mode_table(mpol, ntor)
    trig = vmec_trig_tables(
        ntheta=10,
        nzeta=1,
        nfp=1,
        mmax=mpol - 1,
        nmax=ntor,
        lasym=False,
        dtype=np.float64,
        cache=False,
    )

    idx_m0 = int(np.flatnonzero((modes.m == 0) & (modes.n == 0))[0])
    idx_m1 = int(np.flatnonzero((modes.m == 1) & (modes.n == 0))[0])
    radius = np.sqrt(np.linspace(0.0, 1.0, ns))
    major_radius = 3.0
    scale_m0 = float(trig.mscale[0] * trig.nscale[0])
    scale_m1 = float(trig.mscale[1] * trig.nscale[0])
    odd_scalxc = _odd_m_scalxc_for_default_s(ns)

    Rcos = np.zeros((ns, modes.K), dtype=float)
    Rsin = np.zeros_like(Rcos)
    Zcos = np.zeros_like(Rcos)
    Zsin = np.zeros_like(Rcos)
    Rcos[:, idx_m0] = major_radius / scale_m0
    Rcos[:, idx_m1] = radius / (odd_scalxc * scale_m1)
    Zsin[:, idx_m1] = radius / (odd_scalxc * scale_m1)
    state = SimpleNamespace(Rcos=Rcos, Rsin=Rsin, Zcos=Zcos, Zsin=Zsin)

    geom = vmec_realspace_geom_from_state(state=state, modes=modes, trig=trig)

    theta = (2.0 * np.pi) * np.arange(trig.ntheta3, dtype=float) / float(trig.ntheta1)
    cos_theta = np.cos(theta)[None, :, None]
    sin_theta = np.sin(theta)[None, :, None]
    radius_grid = radius[:, None, None]

    np.testing.assert_allclose(np.asarray(geom["R"]), major_radius + radius_grid * cos_theta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom["Z"]), radius_grid * sin_theta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom["Ru"]), -radius_grid * sin_theta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom["Zu"]), radius_grid * cos_theta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom["Rv"]), 0.0, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(geom["Zv"]), 0.0, rtol=0.0, atol=1e-13)
    assert geom["Lu"] is None
    assert geom["Lv"] is None

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.wout as wout
from vmec_jax.modes import ModeTable
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.kernels.tomnsp import vmec_trig_tables


def _state(ns: int = 3, *, lasym: bool = False) -> VMECState:
    modes = ModeTable(m=np.asarray([0, 1], dtype=int), n=np.asarray([0, 0], dtype=int))
    layout = StateLayout(ns=ns, K=2, lasym=lasym)
    radial = np.linspace(0.0, 1.0, ns)[:, None]
    rcos = np.concatenate([2.0 + 0.0 * radial, 0.15 * radial], axis=1)
    zsin = np.concatenate([0.0 * radial, 0.25 * radial], axis=1)
    rsin = np.zeros((ns, 2))
    zcos = np.zeros((ns, 2))
    if lasym:
        rsin[:, 1] = 0.02 * np.linspace(0.0, 1.0, ns)
        zcos[:, 1] = -0.03 * np.linspace(0.0, 1.0, ns)
    return VMECState(
        layout=layout,
        Rcos=rcos,
        Rsin=rsin,
        Zcos=zcos,
        Zsin=zsin,
        Lcos=np.zeros((ns, 2)),
        Lsin=np.zeros((ns, 2)),
    )


def test_equilibrium_aspect_ratio_builds_default_trig_and_handles_zero_cross_section():
    static = SimpleNamespace(
        cfg=SimpleNamespace(ntheta=6, nzeta=1, nfp=1, mpol=1, ntor=0, lasym=False),
        modes=ModeTable(m=np.asarray([0, 1], dtype=int), n=np.asarray([0, 0], dtype=int)),
    )

    aspect = float(np.asarray(wout.equilibrium_aspect_ratio_from_state(state=_state(), static=static)))
    assert np.isfinite(aspect)
    assert aspect > 0.0

    zero_area = _state()
    zero_area = VMECState(
        layout=zero_area.layout,
        Rcos=zero_area.Rcos,
        Rsin=zero_area.Rsin,
        Zcos=zero_area.Zcos,
        Zsin=np.zeros_like(zero_area.Zsin),
        Lcos=zero_area.Lcos,
        Lsin=zero_area.Lsin,
    )
    assert float(np.asarray(wout.equilibrium_aspect_ratio_from_state(state=zero_area, static=static))) == 0.0


def test_realspace_geom_light_infers_lasym_from_asymmetric_coefficients():
    state = _state(lasym=True)
    modes = ModeTable(m=np.asarray([0, 1], dtype=int), n=np.asarray([0, 0], dtype=int))
    trig = vmec_trig_tables(ntheta=6, nzeta=1, nfp=1, mmax=1, nmax=0, lasym=True, cache=False)

    geom = wout._vmec_realspace_geom_light_from_state(state=state, modes=modes, trig=trig, lasym=None)

    assert set(geom) == {"R", "Z", "Zu"}
    assert all(np.asarray(arr).shape == (1, int(trig.ntheta3), 1) for arr in geom.values())
    assert np.all(np.isfinite(np.asarray(geom["R"])))


def test_symforce_output_paths_cover_reversed_and_empty_zeta_cases():
    trig = vmec_trig_tables(ntheta=6, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=False, cache=False)
    ntheta_full = max(int(trig.ntheta3), int(trig.ntheta1) + 1)
    full = np.arange(ntheta_full * 3, dtype=float).reshape(1, ntheta_full, 3)
    sym, asym = wout._vmec_symoutput_split(f=full, trig=trig, reversed_sym=True)
    expanded = wout._vmec_symoutput_expand(sym=sym, asym=asym, trig=trig)

    np.testing.assert_allclose(expanded[:, : int(trig.ntheta2), :], full[:, : int(trig.ntheta2), :])
    brs = wout._vmec_symforce_apply(f=full, trig=trig, kind="brs")
    anti = wout._vmec_symforce_antisym(f=full, trig=trig, kind="brs", base=np.ones_like(full))
    assert brs.shape == full.shape
    assert anti.shape == full.shape
    assert not np.allclose(brs[:, : int(trig.ntheta2), :], full[:, : int(trig.ntheta2), :])

    empty = np.zeros((1, ntheta_full, 0))
    empty_sym, empty_asym = wout._vmec_symoutput_split(f=empty, trig=trig)
    assert empty_sym.shape == (1, int(trig.ntheta2), 0)
    assert empty_asym.shape == (1, int(trig.ntheta2), 0)
    np.testing.assert_allclose(wout._vmec_symforce_apply(f=empty, trig=trig, kind="ars"), empty)
    np.testing.assert_allclose(wout._vmec_symforce_antisym(f=empty, trig=trig, kind="ars"), empty)

    with pytest.raises(ValueError, match="base shape mismatch"):
        wout._vmec_symforce_antisym(f=full, trig=trig, kind="ars", base=np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="sym/asym shape mismatch"):
        wout._vmec_symoutput_expand(sym=sym, asym=asym[:, :, :1], trig=trig)


def test_nyquist_analysis_and_synthesis_handle_empty_modes_and_table_limits():
    trig = vmec_trig_tables(ntheta=6, nzeta=5, nfp=1, mmax=2, nmax=2, lasym=False, cache=False)
    field = 0.2 + 0.01 * np.arange(2 * int(trig.ntheta2) * 5, dtype=float).reshape(2, int(trig.ntheta2), 5)
    modes = ModeTable(m=np.asarray([0, 1, 2], dtype=int), n=np.asarray([0, -1, 2], dtype=int))

    cos_jxb = wout._vmec_jxbforce_cos_coeffs(f=field, modes=modes, trig=trig)
    sin_jxb = wout._vmec_jxbforce_sin_coeffs(f=field, modes=modes, trig=trig)
    sin_vec = wout._vmec_wrout_nyquist_sin_coeffs(f=field, modes=modes, trig=trig)
    sin_loop = wout._vmec_wrout_nyquist_sin_coeffs_loop(f=field, modes=modes, trig=trig)
    synthesized = wout._vmec_wrout_nyquist_synthesis(coeff_c=cos_jxb, coeff_s=sin_jxb, modes=modes, trig=trig)

    assert cos_jxb.shape == (2, 3)
    assert sin_jxb.shape == (2, 3)
    np.testing.assert_allclose(sin_vec, sin_loop, rtol=1e-12, atol=1e-12)
    assert synthesized.shape == field.shape
    assert np.all(np.isfinite(synthesized))

    empty_modes = ModeTable(m=np.asarray([], dtype=int), n=np.asarray([], dtype=int))
    assert wout._vmec_jxbforce_cos_coeffs(f=field, modes=empty_modes, trig=trig).shape == (2, 0)
    assert wout._vmec_jxbforce_sin_coeffs(f=field, modes=empty_modes, trig=trig).shape == (2, 0)
    assert wout._vmec_wrout_nyquist_sin_coeffs(f=field, modes=empty_modes, trig=trig).shape == (2, 0)
    assert wout._vmec_wrout_nyquist_sin_coeffs_loop(f=field, modes=empty_modes, trig=trig).shape == (2, 0)
    assert (
        wout._vmec_wrout_nyquist_synthesis(
            coeff_c=np.zeros((2, 0)), coeff_s=np.zeros((2, 0)), modes=empty_modes, trig=trig
        ).shape
        == (2, 0, 0)
    )

    too_high = ModeTable(m=np.asarray([3], dtype=int), n=np.asarray([0], dtype=int))
    with pytest.raises(ValueError, match="Trig tables do not cover"):
        wout._vmec_wrout_nyquist_synthesis(coeff_c=np.ones((2, 1)), coeff_s=np.ones((2, 1)), modes=too_high, trig=trig)
    scaled = wout._vmec_wrout_lasym_bsubuv_output_scale(
        bsubumnc=np.asarray([[1.0]]),
        bsubvmnc=np.asarray([[2.0]]),
        bsubumns=np.asarray([[3.0]]),
        bsubvmns=np.asarray([[4.0]]),
    )
    assert [float(arr[0, 0]) for arr in scaled] == [2.0, 4.0, 6.0, 8.0]

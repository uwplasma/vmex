from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.wout as wout
from vmec_jax.modes import ModeTable
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.kernels.tomnsp import vmec_trig_tables


def _grid(shape: tuple[int, ...], offset: float = 0.0, scale: float = 0.01) -> np.ndarray:
    return offset + scale * np.arange(int(np.prod(shape)), dtype=float).reshape(shape)


def test_filter_and_fourier_helpers_cover_remaining_error_and_identity_edges() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=5, nfp=1, mmax=2, nmax=2, lasym=False, cache=False)
    nt2 = int(trig.ntheta2)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    shape = (1, nt2, nzeta)
    base = 0.2 + _grid(shape, scale=0.04)
    other = np.cos(base)
    s_single = np.asarray([0.25])

    filtered = wout._filter_bsubuv_jxbforce(
        bsubu=base,
        bsubv=other,
        trig=trig,
        nfp=1,
        mmax_force=1,
        nmax_force=1,
        s=s_single,
    )
    assert [arr.shape for arr in filtered] == [shape, shape]
    assert all(np.all(np.isfinite(arr)) for arr in filtered)

    with pytest.raises(ValueError, match="shape mismatch"):
        wout._filter_bsubuv_jxbforce_parity_loop(
            bsubu_even=base,
            bsubu_odd=base[:, :, :1],
            bsubv_even=other,
            bsubv_odd=other,
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s_single,
        )
    with pytest.raises(ValueError, match="smaller"):
        wout._filter_bsubuv_jxbforce_parity_loop(
            bsubu_even=base[:, : nt2 - 1],
            bsubu_odd=base[:, : nt2 - 1],
            bsubv_even=other[:, : nt2 - 1],
            bsubv_odd=other[:, : nt2 - 1],
            trig=trig,
            mmax_force=1,
            nmax_force=1,
            s=s_single,
        )
    neg_u, neg_v = wout._filter_bsubuv_jxbforce_parity_loop(
        bsubu_even=base,
        bsubu_odd=0.5 * base,
        bsubv_even=other,
        bsubv_odd=0.25 * other,
        trig=trig,
        mmax_force=-1,
        nmax_force=1,
        s=s_single,
    )
    np.testing.assert_allclose(neg_u, base)
    np.testing.assert_allclose(neg_v, other)

    coeff = np.ones((2, 1))
    unweighted = wout._apply_nyquist_half_weight(
        coeff_cos=coeff,
        coeff_sin=2.0 * coeff,
        modes=ModeTable(m=np.asarray([0], dtype=int), n=np.asarray([0], dtype=int)),
        trig=trig,
    )
    np.testing.assert_allclose(unweighted[0], coeff)
    np.testing.assert_allclose(unweighted[1], 2.0 * coeff)

    modes = ModeTable(m=np.asarray([0, 1], dtype=int), n=np.asarray([0, 1], dtype=int))
    with pytest.raises(ValueError, match="smaller"):
        wout._vmec_wrout_nyquist_cos_coeffs(f=base[:, : nt2 - 1], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="Expected f"):
        wout._vmec_wrout_nyquist_sin_coeffs(f=base[0], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="Expected f"):
        wout._vmec_symforce_apply(f=base[0], trig=trig, kind="ars")
    with pytest.raises(ValueError, match="Expected f"):
        wout._vmec_symforce_antisym(f=base[0], trig=trig, kind="ars")
    with pytest.raises(ValueError, match="smaller"):
        wout._vmec_symoutput_split(f=base[:, : nt2 - 1], trig=trig)
    with pytest.raises(ValueError, match="smaller"):
        wout._vmec_symforce_antisym(f=base[:, : nt2 - 1], trig=trig, kind="ars")

    expanded = wout._vmec_symoutput_expand(
        sym=np.ones((1, 2, 1)),
        asym=None,
        trig=SimpleNamespace(ntheta1=3, ntheta3=1),
    )
    assert expanded.shape == (1, 2, 1)

    trig_lasym = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    lasym_shape = (2, int(trig_lasym.ntheta2), int(np.asarray(trig_lasym.cosnv).shape[0]))
    lasym_base = _grid(lasym_shape, offset=0.1, scale=0.02)
    with pytest.raises(ValueError, match="Expected bsq"):
        wout._vmec_wrout_nyquist_lasym_loop(
            bsq=lasym_base[0],
            gsqrt=lasym_base,
            bsubu=lasym_base,
            bsubv=lasym_base,
            bsubs=lasym_base,
            bsupu=lasym_base,
            bsupv=lasym_base,
            modes=modes,
            trig=trig_lasym,
        )
    with pytest.raises(ValueError, match="reduced theta"):
        wout._vmec_wrout_nyquist_lasym_loop(
            bsq=np.ones((2, int(trig_lasym.ntheta2) + 1, lasym_shape[2])),
            gsqrt=lasym_base,
            bsubu=lasym_base,
            bsubv=lasym_base,
            bsubs=lasym_base,
            bsupu=lasym_base,
            bsupv=lasym_base,
            modes=modes,
            trig=trig_lasym,
        )

    with pytest.raises(ValueError, match="Expected f"):
        wout._vmec_wrout_nyquist_sin_coeffs_loop(f=base[0], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="smaller"):
        wout._vmec_wrout_nyquist_sin_coeffs_loop(f=base[:, : nt2 - 1], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="do not cover"):
        wout._vmec_wrout_nyquist_sin_coeffs_loop(
            f=base,
            modes=ModeTable(m=np.asarray([99], dtype=int), n=np.asarray([0], dtype=int)),
            trig=trig,
        )
    with pytest.raises(ValueError, match="smaller"):
        wout._vmec_jxbforce_cos_coeffs(f=base[:, : nt2 - 1], modes=modes, trig=trig)
    with pytest.raises(ValueError, match="do not cover"):
        wout._vmec_jxbforce_sin_coeffs(
            f=base,
            modes=ModeTable(m=np.asarray([99], dtype=int), n=np.asarray([0], dtype=int)),
            trig=trig,
        )

    np.testing.assert_allclose(np.asarray(wout._chipf_from_chips(wout.jnp.asarray([2.0]))), [2.0])


def test_getbsubs_coefficients_cover_unavailable_and_linalg_failure_paths(monkeypatch) -> None:
    trig_false = vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=False, cache=False)
    nt2 = int(trig_false.ntheta2)
    nzeta = int(np.asarray(trig_false.cosnv).shape[0])
    frho = np.ones((nt2, nzeta))

    assert (
        wout._jxbforce_getbsubs_coeffs_lasym_false(
            frho=frho,
            bsupu=np.ones((nt2, nzeta + 1)),
            bsupv=np.ones((nt2, nzeta)),
            trig=trig_false,
            nfp=1,
        )
        is None
    )
    empty_false = SimpleNamespace(ntheta2=0, cosnv=np.zeros((0, 1)))
    assert (
        wout._jxbforce_getbsubs_coeffs_lasym_false(
            frho=np.zeros((0, 0)),
            bsupu=np.zeros((0, 0)),
            bsupv=np.zeros((0, 0)),
            trig=empty_false,
            nfp=1,
        )
        is None
    )

    trig_true = vmec_trig_tables(ntheta=4, nzeta=5, nfp=1, mmax=2, nmax=2, lasym=True, cache=False)
    nt3 = int(trig_true.ntheta3)
    nzeta_true = int(np.asarray(trig_true.cosnv).shape[0])
    grid_true = np.arange(nt3 * nzeta_true, dtype=float).reshape(nt3, nzeta_true)

    assert (
        wout._jxbforce_getbsubs_coeffs_lasym_true(
            frho=0.1 + np.sin(grid_true),
            bsupu=np.ones((nt3, nzeta_true + 1)),
            bsupv=0.7 + 0.1 * np.sin(grid_true),
            trig=trig_true,
            nfp=1,
        )
        is None
    )
    assert (
        wout._jxbforce_getbsubs_coeffs_lasym_true(
            frho=0.1 + np.sin(grid_true),
            bsupu=1.0 + 0.2 * np.cos(grid_true),
            bsupv=0.7 + 0.1 * np.sin(2.0 * grid_true),
            trig=trig_true,
            nfp=1,
        )
        is None
    )

    empty_true = SimpleNamespace(ntheta3=0, ntheta2=0, cosnv=np.zeros((0, 1)))
    assert (
        wout._jxbforce_getbsubs_coeffs_lasym_true(
            frho=np.zeros((0, 0)),
            bsupu=np.zeros((0, 0)),
            bsupv=np.zeros((0, 0)),
            trig=empty_true,
            nfp=1,
        )
        is None
    )

    valid_true = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=2, nmax=0, lasym=True, cache=False)
    valid_grid = np.arange(int(valid_true.ntheta3), dtype=float).reshape(int(valid_true.ntheta3), 1)

    def raise_linalg(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(wout.np.linalg, "solve", raise_linalg)
    monkeypatch.setattr(wout.np.linalg, "lstsq", raise_linalg)

    assert (
        wout._jxbforce_getbsubs_coeffs_lasym_true(
            frho=0.1 + np.sin(valid_grid),
            bsupu=1.0 + 0.2 * np.cos(valid_grid),
            bsupv=0.7 + 0.1 * np.sin(2.0 * valid_grid),
            trig=valid_true,
            nfp=1,
        )
        is None
    )


def test_compute_mercier_exact_sum_symmetrizes_full_grid_inputs_and_stays_finite(monkeypatch) -> None:
    ns = 3
    modes = ModeTable(m=np.asarray([0, 1], dtype=int), n=np.asarray([0, 0], dtype=int))
    layout = StateLayout(ns=ns, K=2, lasym=False)
    radial = np.linspace(0.0, 1.0, ns)[:, None]
    state = VMECState(
        layout=layout,
        Rcos=np.concatenate([2.0 + 0.0 * radial, 0.2 + 0.0 * radial], axis=1),
        Rsin=np.zeros((ns, 2)),
        Zcos=np.zeros((ns, 2)),
        Zsin=np.concatenate([0.0 * radial, 0.3 + 0.0 * radial], axis=1),
        Lcos=np.zeros((ns, 2)),
        Lsin=np.zeros((ns, 2)),
    )
    trig = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)
    nt2 = int(trig.ntheta2)
    nt3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    half_shape = (ns, nt2, nzeta)
    full_shape = (ns, nt3, nzeta)
    geom = {
        "R": 2.0 * np.ones(full_shape),
        "Z": 0.2 * np.ones(full_shape),
        "Ru": 0.1 * np.ones(full_shape),
        "Zu": 0.3 * np.ones(full_shape),
        "Rv": np.zeros(full_shape),
        "Zv": np.zeros(full_shape),
    }

    monkeypatch.setenv("VMEC_JAX_MERCIER_EXACT_SUM", "1")
    monkeypatch.setattr(wout, "_compute_bsubs_half_mesh", lambda **_kwargs: np.zeros(half_shape))

    mercier = wout._compute_mercier(
        state=state,
        geom_modes=modes,
        s=np.asarray([0.0, 0.5, 1.0]),
        lconm1=False,
        lthreed=False,
        lasym=False,
        nfp=1,
        lbsubs=False,
        mmax_force=1,
        nmax_force=0,
        pres=np.asarray([0.0, 0.1, 0.0]),
        vp=np.asarray([1.0, 1.2, 1.4]),
        phips=np.asarray([0.0, 0.8, 1.0]),
        iotas=np.asarray([0.0, 0.2, 0.3]),
        bsq=2.0 * np.ones(half_shape),
        sqrtg=np.ones(half_shape),
        bsubu=np.ones(full_shape),
        bsubv=0.5 * np.ones(full_shape),
        bsupu=0.1 * np.ones(half_shape),
        bsupv=0.2 * np.ones(half_shape),
        trig=trig,
        geom=geom,
        jac_half=SimpleNamespace(),
        signgs=1,
    )

    assert len(mercier) == 8
    assert all(arr.shape == (ns,) for arr in mercier)
    assert all(np.all(np.isfinite(arr)) for arr in mercier)
    assert np.any(np.abs(mercier[0][1:-1]) > 0.0)

    short = wout._compute_mercier(
        state=state,
        geom_modes=modes,
        s=np.asarray([0.0, 1.0]),
        lconm1=False,
        lthreed=False,
        lasym=False,
        nfp=1,
        lbsubs=False,
        mmax_force=0,
        nmax_force=0,
        pres=np.asarray([0.0, 0.0]),
        vp=np.asarray([1.0, 1.0]),
        phips=np.asarray([0.0, 1.0]),
        iotas=np.asarray([0.0, 0.0]),
        bsq=np.ones((2, nt2, nzeta)),
        sqrtg=np.ones((2, nt2, nzeta)),
        bsubu=np.ones((2, nt2, nzeta)),
        bsubv=np.ones((2, nt2, nzeta)),
        bsupu=np.ones((2, nt2, nzeta)),
        bsupv=np.ones((2, nt2, nzeta)),
        trig=trig,
        geom=geom,
    )
    assert len(short) == 8
    for profile in short:
        np.testing.assert_allclose(profile, np.zeros((2,)))

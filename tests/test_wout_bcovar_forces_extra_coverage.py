from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import vmec_jax.kernels.forces as vf
from vmec_jax.config import VMECConfig
from vmec_jax.namelist import InData
from vmec_jax.static import build_static
from vmec_jax.kernels.bcovar import (
    _apply_vmec_lambda_axis_closure,
    _half_mesh_from_even_odd,
    _metric_cross_even_odd,
    _metric_even_odd,
    _pshalf_from_s,
    vmec_bcovar_half_mesh_from_wout,
)
from vmec_jax.wout import MU0, _compute_equif_wout, _vmec_symoutput_split


def _mode_index(modes, *, m: int, n: int) -> int:
    hits = np.flatnonzero((np.asarray(modes.m) == m) & (np.asarray(modes.n) == n))
    if hits.size == 0:
        raise KeyError((m, n))
    return int(hits[0])


def _zeros_state(static):
    shape = (int(static.cfg.ns), int(static.modes.K))
    zeros = np.zeros(shape, dtype=float)
    return SimpleNamespace(
        Rcos=zeros.copy(),
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zeros.copy(),
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )


def _filled(shape, value: float) -> np.ndarray:
    return np.full(shape, value, dtype=float)


def test_wout_equif_zero_when_pressure_gradient_balances_toroidal_current() -> None:
    trig = SimpleNamespace(
        cosmui3=np.full((2, 1), 0.25, dtype=float),
        mscale=np.asarray([1.0], dtype=float),
        cosnv=np.zeros((2, 1), dtype=float),
    )
    ns = 4
    s = np.linspace(0.0, 1.0, ns)
    bsubu = np.zeros((ns, 2, 2), dtype=float)
    bsubv_levels = np.asarray([0.0, 0.0, 1.0, 2.0], dtype=float)
    bsubv = np.broadcast_to(bsubv_levels[:, None, None], bsubu.shape).copy()
    pres = np.asarray([0.0, 0.0, -1.0, -2.0], dtype=float)

    buco, bvco, jcuru, jcurv, equif = _compute_equif_wout(
        bsubu=bsubu,
        bsubv=bsubv,
        pres=pres,
        vp=np.ones(ns, dtype=float),
        phipf=np.ones(ns, dtype=float),
        chipf=np.zeros(ns, dtype=float),
        signgs=1,
        trig=trig,
        s=s,
    )

    np.testing.assert_allclose(buco, np.zeros(ns))
    np.testing.assert_allclose(bvco, bsubv_levels)
    np.testing.assert_allclose(jcurv, np.zeros(ns))
    np.testing.assert_allclose(jcuru[1:], -3.0 / MU0)
    np.testing.assert_allclose(equif, np.zeros(ns), atol=1.0e-14)


def test_wout_lasym_symoutput_split_preserves_symmetric_and_asymmetric_channels() -> None:
    trig = vf.vmec_trig_tables(ntheta=6, nzeta=3, nfp=1, mmax=1, nmax=1, lasym=True, cache=False)
    nt2 = int(trig.ntheta2)
    nt1 = int(trig.ntheta1)
    nt3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])

    def reflected_channel(*, sign: int) -> np.ndarray:
        out = np.zeros((2, nt3, nzeta), dtype=float)
        seen: set[tuple[int, int]] = set()
        for i in range(nt2):
            ir = 0 if i == 0 else nt1 - i
            for kz in range(nzeta):
                kzr = 0 if kz == 0 else nzeta - kz
                key = tuple(sorted(((i, kz), (ir, kzr)))[0])
                if key in seen:
                    continue
                seen.add(key)
                value = np.asarray([1.0 + i + 0.1 * kz, -0.5 - 0.2 * i + 0.05 * kz])
                if sign < 0 and (i, kz) == (ir, kzr):
                    value = np.zeros_like(value)
                out[:, i, kz] = value
                out[:, ir, kzr] = sign * value
        return out

    symmetric = reflected_channel(sign=1)
    antisymmetric = reflected_channel(sign=-1)

    sym, asym = _vmec_symoutput_split(f=symmetric + antisymmetric, trig=trig)
    rev_sym, rev_asym = _vmec_symoutput_split(f=symmetric + antisymmetric, trig=trig, reversed_sym=True)

    np.testing.assert_allclose(sym, symmetric[:, :nt2, :])
    np.testing.assert_allclose(asym, antisymmetric[:, :nt2, :])
    np.testing.assert_allclose(rev_sym, antisymmetric[:, :nt2, :])
    np.testing.assert_allclose(rev_asym, symmetric[:, :nt2, :])


def test_bcovar_lasym_channels_recombine_to_covariant_fields() -> None:
    cfg = VMECConfig(
        ns=3,
        mpol=2,
        ntor=1,
        nfp=1,
        lasym=True,
        lthreed=True,
        lconm1=True,
        ntheta=6,
        nzeta=3,
    )
    static = build_static(cfg)
    state = _zeros_state(static)
    s = np.asarray(static.s)
    state.Rcos[:, _mode_index(static.modes, m=0, n=0)] = 2.0
    state.Rcos[:, _mode_index(static.modes, m=1, n=0)] = 0.25 * np.sqrt(s)
    state.Zsin[:, _mode_index(static.modes, m=1, n=0)] = 0.25 * np.sqrt(s)
    state.Rsin[:, _mode_index(static.modes, m=1, n=0)] = 0.03 * s
    state.Zcos[:, _mode_index(static.modes, m=1, n=0)] = -0.02 * s

    wout = SimpleNamespace(
        phipf=np.ones(cfg.ns),
        phips=np.r_[0.0, np.ones(cfg.ns - 1)],
        chipf=np.zeros(cfg.ns),
        iotaf=np.zeros(cfg.ns),
        iotas=np.zeros(cfg.ns),
        signgs=1,
        nfp=1,
        mpol=cfg.mpol,
        ntor=cfg.ntor,
        lasym=True,
        flux_is_internal=True,
        ncurr=0,
        lcurrent=False,
        icurv=np.zeros(cfg.ns),
        pres=np.zeros(cfg.ns),
    )

    bc, aux = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout,
        use_vmec_synthesis=True,
        return_parity_aux=True,
    )

    expected_shape = (cfg.ns, int(static.trig_vmec.ntheta3), cfg.nzeta)
    assert np.asarray(bc.bsupu).shape == expected_shape
    assert np.asarray(aux.pr1_even).shape == expected_shape
    assert np.all(np.isfinite(np.asarray(bc.bsq)))
    assert float(np.linalg.norm(np.asarray(aux.pr1_odd))) > 0.0

    pshalf = np.asarray(_pshalf_from_s(static.s))[:, None, None]
    np.testing.assert_allclose(
        np.asarray(bc.bsubu),
        np.asarray(bc.bsubu_parity_even) + pshalf * np.asarray(bc.bsubu_parity_odd),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(bc.bsubv),
        np.asarray(bc.bsubv_parity_even) + pshalf * np.asarray(bc.bsubv_parity_odd),
        atol=1.0e-12,
    )


def test_bcovar_even_odd_metric_reconstructs_physical_half_mesh() -> None:
    s = np.asarray([0.0, 0.25, 1.0])
    a0 = np.asarray([[[1.0]], [[1.2]], [[1.5]]])
    a1 = np.asarray([[[0.3]], [[-0.1]], [[0.2]]])
    b0 = np.asarray([[[0.4]], [[0.7]], [[0.9]]])
    b1 = np.asarray([[[-0.2]], [[0.5]], [[0.1]]])

    even, odd = _metric_even_odd(a0=a0, a1=a1, b0=b0, b1=b1, s=s)
    full = (a0 + np.sqrt(s)[:, None, None] * a1) ** 2 + (b0 + np.sqrt(s)[:, None, None] * b1) ** 2
    np.testing.assert_allclose(np.asarray(even) + np.sqrt(s)[:, None, None] * np.asarray(odd), full)

    cross_even, cross_odd = _metric_cross_even_odd(a0=a0, a1=a1, b0=b0, b1=b1, s=s)
    cross_full = (a0 + np.sqrt(s)[:, None, None] * a1) * (b0 + np.sqrt(s)[:, None, None] * b1)
    np.testing.assert_allclose(np.asarray(cross_even) + np.sqrt(s)[:, None, None] * np.asarray(cross_odd), cross_full)

    half = np.asarray(_half_mesh_from_even_odd(even, odd, s=s))
    pshalf = np.asarray(_pshalf_from_s(s))[:, None, None]
    expected_inner = 0.5 * (even[1:] + even[:-1] + pshalf[1:] * (odd[1:] + odd[:-1]))
    np.testing.assert_allclose(half, np.concatenate([expected_inner[:1], expected_inner], axis=0))

    singleton_even = np.asarray([[[3.0]]])
    np.testing.assert_allclose(
        np.asarray(_half_mesh_from_even_odd(singleton_even, np.asarray([[[9.0]]]), s=np.asarray([0.0]))),
        singleton_even,
    )


def test_bcovar_lambda_axis_closure_copies_only_three_dimensional_m0_modes() -> None:
    lsin = np.asarray(
        [
            [10.0, 20.0, 30.0, 40.0],
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
        ]
    )
    m_modes = np.asarray([0, 0, 1, 0])
    n_modes = np.asarray([0, 1, 1, -1])

    closed = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            lthreed=True,
            ntor=1,
        )
    )
    expected = lsin.copy()
    expected[0, 1] = lsin[1, 1]
    np.testing.assert_allclose(closed, expected)

    mask_closed = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            axis_copy_mask=np.asarray([False, False, False, True]),
            lthreed=True,
            ntor=1,
        )
    )
    expected_mask = lsin.copy()
    expected_mask[0, 3] = lsin[1, 3]
    np.testing.assert_allclose(mask_closed, expected_mask)
    np.testing.assert_allclose(
        np.asarray(
            _apply_vmec_lambda_axis_closure(
                Lsin=lsin,
                m_modes=m_modes,
                n_modes=n_modes,
                lthreed=False,
                ntor=1,
            )
        ),
        lsin,
    )
    np.testing.assert_allclose(
        np.asarray(
            _apply_vmec_lambda_axis_closure(
                Lsin=lsin,
                m_modes=m_modes,
                n_modes=n_modes,
                lthreed=True,
                ntor=0,
            )
        ),
        lsin,
    )


def test_forces_indata_profile_fill_passes_flux_and_pressure_coefficients(monkeypatch) -> None:
    cfg = VMECConfig(
        ns=3,
        mpol=2,
        ntor=0,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=False,
        ntheta=6,
        nzeta=1,
    )
    static = build_static(cfg)
    state = _zeros_state(static)
    shape = (cfg.ns, int(static.trig_vmec.ntheta3), cfg.nzeta)
    captured: dict[str, np.ndarray] = {}

    def fake_bcovar_half_mesh_from_wout(*, wout, pres=None, **kwargs):
        del kwargs
        captured["phipf"] = np.asarray(wout.phipf, dtype=float)
        captured["phips"] = np.asarray(wout.phips, dtype=float)
        captured["chipf"] = np.asarray(wout.chipf, dtype=float)
        captured["pres"] = np.asarray(pres, dtype=float)
        bc = SimpleNamespace(
            lu_e=_filled(shape, 0.2),
            lv_e=_filled(shape, 0.1),
            gij_b_uu=_filled(shape, 0.3),
            gij_b_uv=_filled(shape, 0.0),
            gij_b_vv=_filled(shape, 0.4),
            jac=SimpleNamespace(
                ru12=_filled(shape, 1.0),
                zu12=_filled(shape, 2.0),
                rs=_filled(shape, 0.5),
                zs=_filled(shape, 0.25),
                r12=_filled(shape, 1.0),
                sqrtg=_filled(shape, 2.0),
                tau=_filled(shape, 3.0),
            ),
            bsq=_filled(shape, 0.7),
            bsupu=_filled(shape, 0.1),
            bsupv=_filled(shape, 0.2),
            bsubu=_filled(shape, 0.3),
            bsubv=_filled(shape, 0.4),
        )
        parity = SimpleNamespace(
            pr1_even=_filled(shape, 1.0),
            pr1_odd=_filled(shape, 0.1),
            pz1_even=_filled(shape, 0.2),
            pz1_odd=_filled(shape, 0.05),
            pru_even=_filled(shape, 0.3),
            pru_odd=_filled(shape, 0.02),
            pzu_even=_filled(shape, 0.4),
            pzu_odd=_filled(shape, 0.03),
            prv_even=_filled(shape, 0.0),
            prv_odd=_filled(shape, 0.0),
            pzv_even=_filled(shape, 0.0),
            pzv_odd=_filled(shape, 0.0),
            lu_odd=_filled(shape, 0.0),
            lv_odd=_filled(shape, 0.0),
        )
        return bc, parity

    monkeypatch.setattr(vf, "vmec_bcovar_half_mesh_from_wout", fake_bcovar_half_mesh_from_wout)
    indata = InData(
        scalars={
            "PHIEDGE": float(2.0 * np.pi),
            "NCURR": 0,
            "PIOTA_TYPE": "power_series",
            "AI": [0.25],
            "PMASS_TYPE": "power_series",
            "AM": [5.0],
        },
        indexed={},
    )

    kernels = vf.vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=SimpleNamespace(nfp=1, mpol=cfg.mpol, ntor=cfg.ntor, signgs=-1, lasym=False),
        indata=indata,
        constraint_tcon0=0.0,
    )

    np.testing.assert_allclose(captured["phipf"], -np.ones(cfg.ns))
    np.testing.assert_allclose(captured["phips"], [0.0, -1.0, -1.0])
    np.testing.assert_allclose(captured["chipf"], -0.25 * np.ones(cfg.ns))
    np.testing.assert_allclose(captured["pres"], MU0 * 5.0 * np.ones(cfg.ns), rtol=1.0e-14)
    assert np.asarray(kernels.armn_e).shape == shape
    np.testing.assert_allclose(np.asarray(kernels.gcon), np.zeros(shape))
    pshalf = np.asarray(vf._pshalf_from_s(static.s))[:, None, None]
    expected_crmn = np.broadcast_to(0.1 * pshalf, shape).copy()
    expected_crmn[0] = 0.0
    np.testing.assert_allclose(np.asarray(kernels.crmn_e), expected_crmn)
    np.testing.assert_allclose(np.asarray(kernels.czmn_e)[1:], 0.2)

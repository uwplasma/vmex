from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.config import VMECConfig
from vmec_jax.coords import Coords, eval_coords
from vmec_jax.init_guess import (
    _axis_array,
    _flip_boundary_theta,
    _read_axis_coeffs,
    _recompute_axis_from_state_vmec,
    initial_guess_from_boundary,
)
from vmec_jax.namelist import InData
from vmec_jax.plotting import (
    bmag_from_state_physical,
    plot_bmag_contours,
    plot_objective_history,
    plot_wout,
    surface_rz_from_state,
)
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.static import build_static


def _k_index(static, m: int, n: int) -> int:
    for k, (mm, nn) in enumerate(zip(static.modes.m, static.modes.n)):
        if int(mm) == m and int(nn) == n:
            return k
    raise KeyError((m, n))


def _cfg(*, mpol=2, ntor=1, ns=3, nfp=1, lasym=False, lthreed=True, ntheta=8, nzeta=4) -> VMECConfig:
    return VMECConfig(
        mpol=mpol,
        ntor=ntor,
        ns=ns,
        nfp=nfp,
        lasym=lasym,
        lthreed=lthreed,
        lconm1=False,
        ntheta=ntheta,
        nzeta=nzeta,
    )


def _boundary(static, *, r0: float = 10.0) -> BoundaryCoeffs:
    K = int(static.modes.K)
    Rcos = np.zeros(K)
    Rsin = np.zeros(K)
    Zcos = np.zeros(K)
    Zsin = np.zeros(K)
    Rcos[_k_index(static, 0, 0)] = r0
    Rcos[_k_index(static, 1, 0)] = 1.0
    Zsin[_k_index(static, 1, 0)] = 0.7
    if int(static.cfg.ntor) >= 1:
        Rcos[_k_index(static, 0, 1)] = 0.25
        Rsin[_k_index(static, 0, 1)] = 0.05
        Zcos[_k_index(static, 0, 1)] = -0.04
        Zsin[_k_index(static, 0, 1)] = 0.03
    return BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)


def _state(static) -> VMECState:
    ns = int(static.cfg.ns)
    K = int(static.modes.K)
    layout = StateLayout(ns=ns, K=K, lasym=bool(static.cfg.lasym))
    Rcos = np.zeros((ns, K))
    Rsin = np.zeros_like(Rcos)
    Zcos = np.zeros_like(Rcos)
    Zsin = np.zeros_like(Rcos)
    Lcos = np.zeros_like(Rcos)
    Lsin = np.zeros_like(Rcos)
    Rcos[:, _k_index(static, 0, 0)] = 2.0
    Rcos[:, _k_index(static, 1, 0)] = np.linspace(0.0, 0.3, ns)
    Zsin[:, _k_index(static, 1, 0)] = np.linspace(0.0, 0.2, ns)
    if int(static.cfg.ntor) >= 1:
        Rsin[:, _k_index(static, 0, 1)] = 0.02
        Zcos[:, _k_index(static, 0, 1)] = -0.03
    return VMECState(layout, Rcos, Rsin, Zcos, Zsin, Lcos, Lsin)


def _wout(*, ns: int = 2, nfp: int = 1, lasym: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        ns=ns,
        nfp=nfp,
        lasym=lasym,
        xm=np.asarray([0.0, 1.0]),
        xn=np.asarray([0.0, 0.0]),
        xm_nyq=np.asarray([0.0, 1.0]),
        xn_nyq=np.asarray([0.0, 0.0]),
        rmnc=np.asarray([[1.4, 0.1], [1.5, 0.2]])[:ns],
        rmns=np.asarray([[0.0, 0.0], [0.0, 0.0]])[:ns],
        zmns=np.asarray([[0.0, 0.1], [0.0, 0.2]])[:ns],
        zmnc=np.asarray([[0.0, 0.0], [0.0, 0.0]])[:ns],
        bmnc=np.asarray([[2.0, 0.0], [2.1, 0.0]])[:ns],
        bmns=np.asarray([[0.0, 0.0], [0.0, 0.0]])[:ns],
    )


def test_axis_read_defaults_partial_axes_and_lrecompute_branch(monkeypatch) -> None:
    static = build_static(_cfg())
    boundary = _boundary(static)

    legacy = InData(scalars={"RAXIS": [9.5, 0.1], "ZAXIS": 0.2}, indexed={})
    assert set(_read_axis_coeffs(legacy)) == {"RAXIS_CC", "ZAXIS_CS"}
    np.testing.assert_allclose(_axis_array([1.0], 2, dtype=float), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(_axis_array([1.0, 2.0, 3.0], 1, dtype=float), [1.0, 2.0])
    assert _axis_array(None, 1, dtype=float) is None

    monkeypatch.setenv("VMEC_JAX_INIT_GUESS_JAX", "0")
    indata = InData(scalars={"RAXIS_CC": [9.5, 0.15], "LRECOMPUTE": True}, indexed={})
    state = initial_guess_from_boundary(static, boundary, indata, infer_axis_if_missing=True)

    assert state.Rcos.shape == (static.cfg.ns, static.modes.K)
    assert np.isfinite(np.asarray(state.Rcos)).all()
    assert abs(float(np.asarray(state.Rcos)[0, _k_index(static, 0, 0)]) - 9.5) > 1.0e-6


def test_missing_axis_inference_disabled_keeps_explicit_zero_axis_and_flip_partner_noop() -> None:
    static = build_static(_cfg(lasym=True))
    boundary = _boundary(static)
    zero_axis = InData(
        scalars={"RAXIS_CC": [0.0, 0.0], "RAXIS_CS": [0.0, 0.0], "ZAXIS_CC": [0.0, 0.0], "ZAXIS_CS": [0.0, 0.0]},
        indexed={},
    )

    state = initial_guess_from_boundary(static, boundary, zero_axis, infer_axis_if_missing=False)
    np.testing.assert_allclose(np.asarray(state.Rcos)[0, _k_index(static, 0, 0)], 0.0, atol=1e-14)

    truncated = BoundaryCoeffs(
        R_cos=np.asarray(boundary.R_cos).copy(),
        R_sin=np.asarray(boundary.R_sin).copy(),
        Z_cos=np.asarray(boundary.Z_cos).copy(),
        Z_sin=np.asarray(boundary.Z_sin).copy(),
    )
    flipped = _flip_boundary_theta(static, truncated)
    np.testing.assert_allclose(np.asarray(flipped.R_cos)[_k_index(static, 0, 1)], boundary.R_cos[_k_index(static, 0, 1)])


def test_axis_recompute_from_state_error_and_even_nzeta_nyquist_branch() -> None:
    static = build_static(_cfg(ntor=2, nzeta=4, ntheta=6))
    trig = static.trig_vmec
    shape = (static.cfg.ns, int(trig.ntheta3), static.cfg.nzeta)
    theta = np.linspace(0.0, np.pi, shape[1])[:, None]
    base_r = 10.0 + np.cos(theta) + np.zeros((shape[1], shape[2]))
    base_z = 0.5 * np.sin(theta) + np.zeros((shape[1], shape[2]))
    pr1_even = np.repeat(base_r[None, :, :], static.cfg.ns, axis=0)
    pz1_even = np.repeat(base_z[None, :, :], static.cfg.ns, axis=0)
    zeros = np.zeros(shape)
    pru_even = np.repeat((-np.sin(theta) + np.zeros((shape[1], shape[2])))[None, :, :], static.cfg.ns, axis=0)
    pzu_even = np.repeat((0.5 * np.cos(theta) + np.zeros((shape[1], shape[2])))[None, :, :], static.cfg.ns, axis=0)

    with pytest.raises(ValueError, match="Unexpected pr1_even shape"):
        _recompute_axis_from_state_vmec(
            static,
            pr1_even=zeros[:, 0],
            pr1_odd=zeros,
            pz1_even=zeros,
            pz1_odd=zeros,
            pru_even=zeros,
            pru_odd=zeros,
            pzu_even=zeros,
            pzu_odd=zeros,
            signgs=-1,
            trig=trig,
        )

    raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = _recompute_axis_from_state_vmec(
        static,
        pr1_even=pr1_even,
        pr1_odd=zeros,
        pz1_even=pz1_even,
        pz1_odd=zeros,
        pru_even=pru_even,
        pru_odd=zeros,
        pzu_even=pzu_even,
        pzu_odd=zeros,
        signgs=-1,
        n_grid=3,
        trig=trig,
    )

    assert raxis_cc.shape == (static.cfg.ntor + 1,)
    assert raxis_cs.shape == zaxis_cc.shape == zaxis_cs.shape == raxis_cc.shape
    assert np.isfinite(raxis_cc).all()


def test_coords_and_state_surface_branches_for_lasym_and_lthreed() -> None:
    static = build_static(_cfg(lasym=True))
    state = _state(static)

    coords = eval_coords(state, static.basis)
    children, aux = coords.tree_flatten()
    rebuilt = Coords.tree_unflatten(aux, children)
    np.testing.assert_allclose(np.asarray(rebuilt.R), np.asarray(coords.R))

    theta = np.asarray([0.0, np.pi / 2.0])
    zeta = np.asarray([0.0, np.pi / 3.0])
    R, Z = surface_rz_from_state(state, static.modes, theta=theta, zeta=zeta, s_index=1, nfp=static.cfg.nfp)
    assert R.shape == Z.shape == (2, 2)
    assert np.isfinite(R).all()
    assert np.isfinite(Z).all()


def test_plotting_contours_history_wrappers_and_error_branches(monkeypatch, tmp_path) -> None:
    contour_path = plot_bmag_contours(_wout(nfp=2), _wout(nfp=2), outdir=tmp_path)
    assert contour_path == tmp_path / "bmag_surface.png"
    assert contour_path.exists()

    history_path = tmp_path / "history.json"
    history_path.write_text(
        '{"history":[{"objective":4.0,"qs_objective":2.0,"aspect":5.0,"iota":0.2},'
        '{"objective":1.0,"aspect":4.0,"iota":0.3}],'
        '"target_aspect":4.0,"target_iota":0.25,"iota_abs_min":0.1,"stage_boundaries":[0],"nfev":2}',
        encoding="utf-8",
    )
    out_history = plot_objective_history(history_path, outdir=tmp_path / "hist")
    assert out_history.exists()

    static = build_static(_cfg(ntor=0, lthreed=False, nzeta=1))
    state = _state(static)
    with pytest.raises(ValueError, match="indata must be provided"):
        bmag_from_state_physical(state, static, theta=np.asarray([0.0]), phi=np.asarray([0.0]), s_index=1, signgs=1)


def test_plot_wout_name_defaults_and_show_branch(monkeypatch, tmp_path) -> None:
    import matplotlib.figure
    import matplotlib.pyplot as plt

    tiny = _wout(ns=2, nfp=1)
    tiny.ntor = 0
    tiny.phi = np.asarray([0.0, 1.0])
    tiny.iotas = np.asarray([0.1, 0.2])
    tiny.iotaf = np.asarray([0.1, 0.2])
    tiny.pres = np.asarray([1.0, 0.5])
    tiny.presf = np.asarray([1.0, 0.5])
    tiny.buco = np.asarray([0.0, 0.0])
    tiny.bvco = np.asarray([0.0, 0.0])
    tiny.jcuru = np.asarray([0.0, 0.0])
    tiny.jcurv = np.asarray([0.0, 0.0])
    tiny.DMerc = np.asarray([0.0, 0.0])
    tiny.raxis_cc = np.asarray([1.5])
    tiny.raxis_cs = np.asarray([0.0])
    tiny.zaxis_cs = np.asarray([0.0])
    tiny.zaxis_cc = np.asarray([0.0])
    tiny.bmnc = np.asarray(tiny.bmnc, dtype=float)
    tiny.bmnc[:, 1] = 0.05

    monkeypatch.setattr("vmec_jax.wout.read_wout", lambda _path: tiny)
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda _fig, path, *args, **kwargs: Path(path).write_bytes(b"plot"),
    )
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    results = plot_wout(tmp_path / "wout_tiny.nc", outdir=tmp_path / "plots", show=True)
    assert results["vmec_params"].name == "tiny_VMECparams.pdf"
    assert all(path.exists() for path in results.values())
    assert show_calls == [True]

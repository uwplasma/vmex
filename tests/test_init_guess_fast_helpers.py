from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.config import VMECConfig
from vmec_jax.init_guess import (
    _any_value_is_traced,
    _apply_m1_constraint,
    _axis_parity_from_state_lasym,
    _boundary_cross_section_areas,
    _boundary_is_traced,
    _flip_boundary_theta,
    _guess_axis_from_boundary,
    _recompute_axis_from_boundary,
    _undo_m1_constraint_for_recompute,
    _vmec_lflip_from_boundary,
    _vmec_lflip_from_boundary_jax,
    extract_axis_override_from_state,
    initial_guess_from_boundary,
)
from vmec_jax.namelist import InData
from vmec_jax.static import build_static


def _k_index(modes, m: int, n: int) -> int:
    for k, (mm, nn) in enumerate(zip(modes.m, modes.n)):
        if int(mm) == int(m) and int(nn) == int(n):
            return k
    raise KeyError((m, n))


def _ellipse_boundary(static, *, r0: float = 10.0, r_axis_n1: float = 0.4) -> BoundaryCoeffs:
    K = int(static.modes.K)
    Rcos = np.zeros((K,), dtype=float)
    Rsin = np.zeros((K,), dtype=float)
    Zcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)
    Rcos[_k_index(static.modes, 0, 0)] = r0
    if int(static.cfg.ntor) >= 1:
        Rcos[_k_index(static.modes, 0, 1)] = r_axis_n1
    Rcos[_k_index(static.modes, 1, 0)] = 2.0
    Zsin[_k_index(static.modes, 1, 0)] = 1.0
    return BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)


def test_boundary_axis_guess_area_and_trace_helpers() -> None:
    cfg = VMECConfig(
        mpol=2,
        ntor=1,
        ns=3,
        nfp=1,
        lasym=False,
        lconm1=False,
        lthreed=True,
        ntheta=8,
        nzeta=4,
    )
    static = build_static(cfg)
    boundary = _ellipse_boundary(static)

    raxis_cc, zaxis_cs = _guess_axis_from_boundary(static, boundary)
    np.testing.assert_allclose(np.asarray(raxis_cc), [10.0, 0.4], atol=1e-12)
    np.testing.assert_allclose(np.asarray(zaxis_cs), [0.0, 0.0], atol=1e-12)

    area = np.asarray(_boundary_cross_section_areas(static, boundary))
    assert np.all(area > 0.0)
    flipped_z = BoundaryCoeffs(
        R_cos=boundary.R_cos,
        R_sin=boundary.R_sin,
        Z_cos=boundary.Z_cos,
        Z_sin=-np.asarray(boundary.Z_sin),
    )
    np.testing.assert_allclose(np.asarray(_boundary_cross_section_areas(static, flipped_z)), -area)

    assert _boundary_is_traced(boundary) is False
    assert _any_value_is_traced(boundary.R_cos, 1.0) is False


def test_lflip_jax_and_m1_constraint_guards_return_identity() -> None:
    cfg = VMECConfig(
        mpol=1,
        ntor=0,
        ns=3,
        nfp=1,
        lasym=False,
        lconm1=True,
        lthreed=False,
        ntheta=6,
        nzeta=1,
    )
    static = build_static(cfg)
    boundary = BoundaryCoeffs(
        R_cos=np.ones((static.modes.K,), dtype=float),
        R_sin=np.zeros((static.modes.K,), dtype=float),
        Z_cos=np.zeros((static.modes.K,), dtype=float),
        Z_sin=np.zeros((static.modes.K,), dtype=float),
    )

    assert bool(np.asarray(_vmec_lflip_from_boundary_jax(static, boundary))) is False
    assert _apply_m1_constraint(static, boundary) is boundary
    assert _undo_m1_constraint_for_recompute(static, boundary) is boundary

    unconstrained_cfg = VMECConfig(
        mpol=2,
        ntor=1,
        ns=3,
        nfp=1,
        lasym=True,
        lconm1=False,
        lthreed=True,
        ntheta=6,
        nzeta=4,
    )
    unconstrained_static = build_static(unconstrained_cfg)
    unconstrained_boundary = _ellipse_boundary(unconstrained_static)
    assert _apply_m1_constraint(unconstrained_static, unconstrained_boundary) is unconstrained_boundary
    assert _undo_m1_constraint_for_recompute(unconstrained_static, unconstrained_boundary) is unconstrained_boundary


def test_recompute_axis_from_boundary_tiny_symmetric_ellipse() -> None:
    cfg = VMECConfig(
        mpol=2,
        ntor=1,
        ns=3,
        nfp=1,
        lasym=False,
        lconm1=False,
        lthreed=True,
        ntheta=6,
        nzeta=4,
    )
    static = build_static(cfg)
    boundary = _ellipse_boundary(static, r_axis_n1=0.2)

    raxis_cc, zaxis_cs = _recompute_axis_from_boundary(
        static,
        boundary,
        raxis_cc=np.asarray([9.8, 0.1]),
        zaxis_cs=np.asarray([0.0, 0.0]),
        signgs=-1,
        n_grid=3,
    )

    np.testing.assert_allclose(raxis_cc, [10.0, 0.2], atol=1e-12)
    np.testing.assert_allclose(zaxis_cs, [0.0, 0.0], atol=1e-12)


def test_axis_override_preserves_lasym_projection_and_parity_channels() -> None:
    cfg = VMECConfig(
        mpol=2,
        ntor=1,
        ns=3,
        nfp=1,
        lasym=True,
        lconm1=False,
        lthreed=True,
        ntheta=8,
        nzeta=4,
    )
    static = build_static(cfg)
    K = int(static.modes.K)
    Rcos = np.zeros((K,), dtype=float)
    Rsin = np.zeros((K,), dtype=float)
    Zcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)
    k00 = _k_index(static.modes, 0, 0)
    k01 = _k_index(static.modes, 0, 1)
    k10 = _k_index(static.modes, 1, 0)
    Rcos[k00] = 8.0
    Rcos[k01] = 1.0
    Rsin[k01] = 2.0
    Zcos[k01] = 3.0
    Zsin[k01] = 4.0
    Rsin[k10] = 0.6
    Zcos[k10] = -0.5
    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)
    indata = InData(scalars={"RAXIS_CC": [99.0, 99.0], "ZAXIS_CS": [99.0, 99.0]}, indexed={})
    axis_override = {
        "raxis_cc": np.asarray([5.0, 0.25]),
        "raxis_cs": np.asarray([0.0, 0.35]),
        "zaxis_cc": np.asarray([0.0, -0.45]),
        "zaxis_cs": np.asarray([0.0, 0.55]),
    }

    state = initial_guess_from_boundary(
        static,
        boundary,
        indata,
        axis_override=axis_override,
        infer_axis_if_missing=True,
        vmec_project=True,
    )

    np.testing.assert_allclose(np.asarray(state.Rcos)[:, k00], [5.0, 6.5, 8.0], atol=1e-12)
    assert np.asarray(state.Rsin)[0, k01] == pytest.approx(0.35, abs=1e-12)
    assert np.asarray(state.Zcos)[0, k01] == pytest.approx(-0.45, abs=1e-12)
    assert np.asarray(state.Zsin)[0, k01] == pytest.approx(0.55, abs=1e-12)

    parity = _axis_parity_from_state_lasym(state=state, static=static, trig=static.trig_vmec)
    expected_shape = (cfg.ns, int(static.trig_vmec.ntheta3), cfg.nzeta)
    assert parity["pr1_even"].shape == expected_shape
    assert parity["pr1_odd"].shape == expected_shape
    assert np.max(np.abs(parity["pr1_odd"])) > 0.0
    assert np.max(np.abs(parity["pz1_even"])) > 0.0

    parity_with_default_trig = _axis_parity_from_state_lasym(state=state, static=static)
    np.testing.assert_allclose(parity_with_default_trig["pr1_even"], parity["pr1_even"])


def test_numpy_lflip_and_axis_override_fallback_indexing() -> None:
    cfg = VMECConfig(
        mpol=2,
        ntor=1,
        ns=3,
        nfp=1,
        lasym=False,
        lconm1=False,
        lthreed=True,
        ntheta=8,
        nzeta=4,
    )
    static = build_static(cfg)
    boundary = _ellipse_boundary(static)
    k10 = _k_index(static.modes, 1, 0)
    k11 = _k_index(static.modes, 1, 1)
    k1m1 = _k_index(static.modes, 1, -1)

    decisive = BoundaryCoeffs(
        R_cos=np.asarray(boundary.R_cos),
        R_sin=np.asarray(boundary.R_sin),
        Z_cos=np.asarray(boundary.Z_cos),
        Z_sin=-np.asarray(boundary.Z_sin),
    )
    assert _vmec_lflip_from_boundary(static, decisive) is True

    ambiguous = BoundaryCoeffs(
        R_cos=np.asarray(boundary.R_cos).copy(),
        R_sin=np.asarray(boundary.R_sin),
        Z_cos=np.asarray(boundary.Z_cos),
        Z_sin=np.zeros_like(boundary.Z_sin),
    )
    assert _vmec_lflip_from_boundary(static, ambiguous) is None

    one_sided = BoundaryCoeffs(
        R_cos=np.arange(static.modes.K, dtype=float),
        R_sin=10.0 + np.arange(static.modes.K, dtype=float),
        Z_cos=20.0 + np.arange(static.modes.K, dtype=float),
        Z_sin=30.0 + np.arange(static.modes.K, dtype=float),
    )
    flipped = _flip_boundary_theta(static, one_sided)
    assert flipped.R_cos[k11] == pytest.approx(-one_sided.R_cos[k1m1])
    assert flipped.R_sin[k11] == pytest.approx(one_sided.R_sin[k1m1])
    assert flipped.Z_cos[k10] == pytest.approx(-one_sided.Z_cos[k10])
    assert flipped.Z_sin[k10] == pytest.approx(one_sided.Z_sin[k10])

    state = initial_guess_from_boundary(
        static,
        boundary,
        InData(scalars={"RAXIS_CC": [9.0, 0.5], "ZAXIS_CS": [0.0, 0.25]}, indexed={}),
        infer_axis_if_missing=False,
        vmec_project=True,
    )
    # Exercise the fallback path that builds m0_n_index when a custom static-like
    # object does not cache it.
    static_no_index = SimpleNamespace(cfg=static.cfg, modes=static.modes)
    axis = extract_axis_override_from_state(state, static_no_index)

    m0_indices = [_k_index(static.modes, 0, 0), _k_index(static.modes, 0, 1)]
    np.testing.assert_allclose(np.asarray(axis["raxis_cc"]), np.asarray(state.Rcos)[0, m0_indices])
    np.testing.assert_allclose(np.asarray(axis["zaxis_cs"]), np.asarray(state.Zsin)[0, m0_indices])

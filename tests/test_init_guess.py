import numpy as np
import pytest

from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.config import VMECConfig
from vmec_jax.init_guess import (
    _axis_array,
    _blend_axis_m0_full,
    _flip_boundary_theta,
    _flip_boundary_theta_arrays,
    _read_axis_coeffs,
    _recompute_axis_from_state_vmec,
    _recompute_axis_from_state_vmec_jax,
    _vmec_lflip_from_boundary,
    extract_axis_override_from_state,
    initial_guess_from_boundary,
)
from vmec_jax.namelist import InData
from vmec_jax.static import build_static


def _k_index(modes, m, n):
    for k, (mm, nn) in enumerate(zip(modes.m, modes.n)):
        if int(mm) == int(m) and int(nn) == int(n):
            return k
    raise KeyError((m, n))


def test_initial_guess_scaling_and_axis_blend():
    cfg = VMECConfig(mpol=3, ntor=2, ns=5, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=8, nzeta=6)
    static = build_static(cfg)
    K = static.modes.K

    Rcos = np.zeros((K,), dtype=float)
    Rsin = np.zeros((K,), dtype=float)
    Zcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)

    k00 = _k_index(static.modes, 0, 0)
    k10 = _k_index(static.modes, 1, 0)
    k20 = _k_index(static.modes, 2, 0)

    Rcos[k00] = 10.0
    Rcos[k10] = 2.0
    Rcos[k20] = 3.0
    Rsin[k00] = 4.0

    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)
    indata = InData(scalars={"RAXIS_CC": [5.0], "ZAXIS_CS": [0.0]}, indexed={})

    st0 = initial_guess_from_boundary(static, boundary, indata)
    s = np.asarray(static.s)
    rho = np.sqrt(s)

    # m>0 scaling uses rho**m with VMEC internal mode scaling (mscale=nscale=sqrt(2) for m>0, n!=0).
    mscale = np.sqrt(2.0)
    assert st0.Rcos[1, k10] == pytest.approx(rho[1] * 2.0 / mscale)
    assert st0.Rcos[1, k20] == pytest.approx((rho[1] ** 2) * 3.0 / mscale)

    # m=0 Rcos blends between axis and boundary
    assert st0.Rcos[0, k00] == pytest.approx(5.0)
    assert st0.Rcos[-1, k00] == pytest.approx(10.0)
    assert st0.Rcos[2, k00] == pytest.approx(7.5)

    # m=0 Rsin is suppressed for lasym=False (stellarator symmetry).
    assert st0.Rsin[0, k00] == pytest.approx(0.0)
    assert st0.Rsin[-1, k00] == pytest.approx(0.0)


def test_axis_aliases_padding_and_truncation():
    indata = InData(
        scalars={
            "RAXIS": [1.0, 2.0, 3.0],
            "RAXIS_CS": 0.25,
            "ZAXIS": [4.0],
        },
        indexed={},
    )

    axis = _read_axis_coeffs(indata)

    assert axis["RAXIS_CC"] == [1.0, 2.0, 3.0]
    assert axis["RAXIS_CS"] == 0.25
    assert axis["ZAXIS_CS"] == [4.0]
    np.testing.assert_allclose(_axis_array(axis["RAXIS_CC"], 1, dtype=float), [1.0, 2.0])
    np.testing.assert_allclose(_axis_array(axis["ZAXIS_CS"], 2, dtype=float), [4.0, 0.0, 0.0])
    assert _axis_array(None, 2, dtype=float) is None


def test_vmec_lflip_and_theta_flip_arrays_match_reference():
    cfg = VMECConfig(mpol=3, ntor=1, ns=4, nfp=1, lasym=True, lconm1=True, lthreed=True, ntheta=8, nzeta=4)
    static = build_static(cfg)
    K = static.modes.K
    values = np.arange(1, K + 1, dtype=float)
    boundary = BoundaryCoeffs(
        R_cos=values.copy(),
        R_sin=10.0 + values,
        Z_cos=20.0 + values,
        Z_sin=-(30.0 + values),
    )

    assert _vmec_lflip_from_boundary(static, boundary) is True

    flipped = _flip_boundary_theta(static, boundary)
    got = _flip_boundary_theta_arrays(
        static,
        np.asarray(boundary.R_cos),
        np.asarray(boundary.R_sin),
        np.asarray(boundary.Z_cos),
        np.asarray(boundary.Z_sin),
    )

    for actual, expected in zip(got, (flipped.R_cos, flipped.R_sin, flipped.Z_cos, flipped.Z_sin), strict=True):
        np.testing.assert_allclose(np.asarray(actual), expected)

    zero_z = BoundaryCoeffs(R_cos=boundary.R_cos, R_sin=boundary.R_sin, Z_cos=boundary.Z_cos, Z_sin=np.zeros(K))
    assert _vmec_lflip_from_boundary(static, zero_z) is None


def test_blend_axis_m0_full_no_valid_m0_index_is_noop():
    cfg = VMECConfig(mpol=2, ntor=1, ns=3, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=6, nzeta=4)
    static = build_static(cfg)
    object.__setattr__(static, "m0_n_index", np.full((cfg.ntor + 1,), -1, dtype=int))
    shape = (cfg.ns, static.modes.K)
    Rcos = np.arange(np.prod(shape), dtype=float).reshape(shape)
    Rsin = Rcos + 10.0
    Zcos = Rcos + 20.0
    Zsin = Rcos + 30.0

    out = _blend_axis_m0_full(
        static=static,
        s=np.asarray(static.s),
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Rcos_b=Rcos[-1:],
        Rsin_b=Rsin[-1:],
        Zcos_b=Zcos[-1:],
        Zsin_b=Zsin[-1:],
        raxis_cc=np.zeros(cfg.ntor + 1),
        raxis_cs=np.zeros(cfg.ntor + 1),
        zaxis_cc=np.zeros(cfg.ntor + 1),
        zaxis_cs=np.zeros(cfg.ntor + 1),
    )

    for actual, expected in zip(out, (Rcos, Rsin, Zcos, Zsin), strict=True):
        np.testing.assert_allclose(np.asarray(actual), expected)


def _axis_recompute_arrays(static):
    ns = int(static.cfg.ns)
    ntheta = int(static.trig_vmec.ntheta3)
    nzeta = int(static.cfg.nzeta)
    s = np.arange(ns, dtype=float)[:, None, None]
    theta = np.linspace(0.0, 1.0, ntheta, dtype=float)[None, :, None]
    zeta = np.linspace(0.0, 1.0, nzeta, dtype=float)[None, None, :]
    pr1_even = 1.0 + 0.05 * s + 0.03 * theta + 0.02 * zeta
    pr1_odd = 0.01 * (1.0 + theta)
    pz1_even = 0.02 * zeta + 0.01 * s
    pz1_odd = 0.04 * (theta - 0.5)
    pru_even = 0.5 + 0.02 * s + 0.03 * theta
    pru_odd = 0.01 + 0.02 * zeta
    pzu_even = 0.4 + 0.01 * theta + 0.02 * zeta
    pzu_odd = 0.02 + 0.01 * s
    shape = (ns, ntheta, nzeta)
    return {
        "pr1_even": pr1_even,
        "pr1_odd": np.broadcast_to(pr1_odd, shape).copy(),
        "pz1_even": np.broadcast_to(pz1_even, shape).copy(),
        "pz1_odd": np.broadcast_to(pz1_odd, shape).copy(),
        "pru_even": np.broadcast_to(pru_even, shape).copy(),
        "pru_odd": np.broadcast_to(pru_odd, shape).copy(),
        "pzu_even": np.broadcast_to(pzu_even, shape).copy(),
        "pzu_odd": np.broadcast_to(pzu_odd, shape).copy(),
    }


@pytest.mark.parametrize("lasym", [False, True])
def test_axis_recompute_numpy_and_jax_paths_match(lasym):
    cfg = VMECConfig(mpol=2, ntor=1, ns=3, nfp=1, lasym=lasym, lconm1=True, lthreed=True, ntheta=6, nzeta=4)
    static = build_static(cfg)
    arrays = _axis_recompute_arrays(static)

    expected = _recompute_axis_from_state_vmec(static, **arrays, signgs=-1, n_grid=4, trig=static.trig_vmec)
    actual = _recompute_axis_from_state_vmec_jax(static, **arrays, signgs=-1, n_grid=4, trig=static.trig_vmec)

    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(got), want, atol=1e-12)


def test_axis_recompute_validates_mesh_shapes():
    cfg = VMECConfig(mpol=2, ntor=1, ns=3, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=6, nzeta=4)
    static = build_static(cfg)
    arrays = _axis_recompute_arrays(static)

    with pytest.raises(ValueError, match="Unexpected pr1_even shape"):
        _recompute_axis_from_state_vmec(
            static,
            **{**arrays, "pr1_even": np.zeros((cfg.ns, static.trig_vmec.ntheta3))},
            signgs=-1,
            trig=static.trig_vmec,
        )

    with pytest.raises(ValueError, match="zeta size"):
        _recompute_axis_from_state_vmec(
            static,
            **{**arrays, "pr1_even": np.zeros((cfg.ns, static.trig_vmec.ntheta3, cfg.nzeta + 1))},
            signgs=-1,
            trig=static.trig_vmec,
        )

    ns1_cfg = VMECConfig(mpol=2, ntor=1, ns=1, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=6, nzeta=4)
    ns1_static = build_static(ns1_cfg)
    shape = (1, int(ns1_static.trig_vmec.ntheta3), int(ns1_cfg.nzeta))
    zeros = {key: np.zeros(shape) for key in arrays}
    with pytest.raises(ValueError, match="ns >= 2"):
        _recompute_axis_from_state_vmec(ns1_static, **zeros, signgs=-1, trig=ns1_static.trig_vmec)


def test_initial_guess_can_freeze_zero_axis_and_extract_override():
    cfg = VMECConfig(mpol=2, ntor=1, ns=4, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=6, nzeta=4)
    static = build_static(cfg)
    K = static.modes.K
    k00 = _k_index(static.modes, 0, 0)
    Rcos = np.zeros((K,), dtype=float)
    Rcos[k00] = 8.0
    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=np.zeros(K), Z_cos=np.zeros(K), Z_sin=np.zeros(K))
    indata = InData(scalars={"RAXIS_CC": [0.0, 0.0], "ZAXIS_CS": [0.0, 0.0]}, indexed={})

    state = initial_guess_from_boundary(static, boundary, indata, infer_axis_if_missing=False)

    assert state.Rcos[0, k00] == pytest.approx(0.0)
    assert state.Rcos[-1, k00] == pytest.approx(8.0)

    override = extract_axis_override_from_state(state, static)
    np.testing.assert_allclose(np.asarray(override["raxis_cc"]), [0.0, 0.0])
    assert set(override) == {"raxis_cc", "raxis_cs", "zaxis_cc", "zaxis_cs"}

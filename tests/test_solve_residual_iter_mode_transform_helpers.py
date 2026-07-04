from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.modes import vmec_mode_table
from vmec_jax.solvers.fixed_boundary.residual.mode_transform import (
    build_mode_transform_context,
    build_mode_transform_host_projection,
    mn_cos_to_signed_host_projected,
    mn_sin_to_signed_host_projected,
    mode_diag_weights_mn,
    mode_diag_weights_mn_np,
    vmec_scalxc_from_s_np,
)
from vmec_jax.kernels.residue import vmec_scalxc_from_s
from vmec_jax.kernels.parity import (
    _mn_cos_to_signed_host,
    _mn_sin_to_signed_host,
    signed_maps_from_modes,
)


def test_host_projection_matches_vmec_parity_host_transforms() -> None:
    modes = vmec_mode_table(4, 3)
    maps = signed_maps_from_modes(modes)
    projection = build_mode_transform_host_projection(maps, ncoeff=modes.m.size)
    rng = np.random.default_rng(1234)
    cc = rng.standard_normal((3, maps.mpol, maps.nrange))
    ss = rng.standard_normal((3, maps.mpol, maps.nrange))
    sc = rng.standard_normal((3, maps.mpol, maps.nrange))
    cs = rng.standard_normal((3, maps.mpol, maps.nrange))

    np.testing.assert_allclose(
        mn_cos_to_signed_host_projected(cc, ss, projection),
        _mn_cos_to_signed_host(cc, ss, maps=maps, ncoeff=modes.m.size),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        mn_cos_to_signed_host_projected(cc, None, projection),
        _mn_cos_to_signed_host(cc, np.zeros_like(cc), maps=maps, ncoeff=modes.m.size),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        mn_sin_to_signed_host_projected(sc, cs, projection),
        _mn_sin_to_signed_host(sc, cs, maps=maps, ncoeff=modes.m.size),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        mn_sin_to_signed_host_projected(sc, None, projection),
        _mn_sin_to_signed_host(sc, np.zeros_like(sc), maps=maps, ncoeff=modes.m.size),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_host_projection_handles_zero_coefficients() -> None:
    maps = signed_maps_from_modes(vmec_mode_table(2, 1))
    projection = build_mode_transform_host_projection(maps, ncoeff=0)
    assert projection.ncoeff == 0
    assert projection.n_half == 0
    assert projection.A_cos is None
    assert projection.A_sin is None
    cc = np.zeros((2, maps.mpol, maps.nrange))
    np.testing.assert_allclose(mn_cos_to_signed_host_projected(cc, None, projection), np.zeros((2, 0)))
    np.testing.assert_allclose(mn_sin_to_signed_host_projected(cc, None, projection), np.zeros((2, 0)))


def test_numpy_scalxc_and_mode_diag_weights_match_jax_helpers() -> None:
    s = np.asarray([0.0, 0.25, 1.0])
    scalxc = vmec_scalxc_from_s_np(s, mpol=5, dtype=np.float64)
    expected = np.asarray(
        [
            [1.0, 2.0, 1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(scalxc, expected)
    assert vmec_scalxc_from_s_np(np.asarray([]), mpol=3, dtype=np.float64).shape == (0, 3)
    assert vmec_scalxc_from_s_np(s, mpol=0, dtype=np.float64).shape == (3, 0)

    weights_np = mode_diag_weights_mn_np(
        mpol=4,
        nrange=3,
        nfp=2.0,
        mode_diag_exponent=0.75,
        dtype=np.float64,
    )
    weights_jax = mode_diag_weights_mn(
        mpol=4,
        nrange=3,
        nfp=2.0,
        mode_diag_exponent=0.75,
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(weights_np, np.asarray(weights_jax), rtol=1.0e-15, atol=1.0e-15)


def _static(*, mpol: int = 3, ntor: int = 2, nfp: int = 2, lasym: bool = True):
    modes = vmec_mode_table(mpol, ntor)
    return SimpleNamespace(
        cfg=SimpleNamespace(mpol=mpol, ntor=ntor, nfp=nfp, lthreed=True, lasym=lasym),
        modes=modes,
        signed_maps=None,
        mn_idx_m=None,
        mn_idx_n=None,
        mn_idx_kp=None,
        mn_idx_kn=None,
        mn_has_kn=None,
        m_is_m0=None,
    )


def _state(ns: int, k: int):
    base = np.arange(ns * k, dtype=float).reshape(ns, k) + 1.0
    return SimpleNamespace(
        Rcos=base,
        Rsin=base + 10.0,
        Zcos=base + 20.0,
        Zsin=base + 30.0,
        Lcos=base + 40.0,
        Lsin=base + 50.0,
    )


def test_mode_transform_context_builds_host_transforms_and_weights() -> None:
    static = _static(mpol=3, ntor=2, nfp=2, lasym=True)
    static.signed_maps = signed_maps_from_modes(static.modes)
    static.mode_transform_host_projection = build_mode_transform_host_projection(
        static.signed_maps,
        ncoeff=static.modes.m.size,
    )
    state0 = _state(ns=3, k=static.modes.m.size)
    context = build_mode_transform_context(
        static=static,
        state0=state0,
        s=np.asarray([0.0, 0.25, 1.0]),
        host_update_assembly=True,
        setup_host_enforce=False,
        divide_by_scalxc_for_update=True,
        mode_diag_exponent=0.5,
        tree_has_tracer=lambda _value: False,
        vmec_scalxc_from_s=vmec_scalxc_from_s,
    )

    assert context.mpol == 3
    assert context.ntor == 2
    assert context.nrange == 3
    assert context.ncoeff == static.modes.m.size
    assert context.host_projection is static.mode_transform_host_projection
    assert context.scalxc_mn_np is not None
    assert context.w_mode_mn_np is not None
    np.testing.assert_allclose(context.m0_mask, np.asarray(static.modes.m) == 0)

    rng = np.random.default_rng(42)
    sc = rng.standard_normal((3, context.mpol, context.nrange))
    cs = rng.standard_normal((3, context.mpol, context.nrange))
    direct = context.mn_sin_to_signed_physical(sc, cs)
    batched = context.mn_sin_to_signed_physical_batch(sc[None, ...], cs[None, ...])[0]
    np.testing.assert_allclose(np.asarray(batched), np.asarray(direct), rtol=1.0e-13, atol=1.0e-13)


def test_mode_transform_context_zero_exponent_uses_unit_weights() -> None:
    static = _static(mpol=4, ntor=3, nfp=2, lasym=False)
    state0 = _state(ns=3, k=static.modes.m.size)
    context = build_mode_transform_context(
        static=static,
        state0=state0,
        s=np.asarray([0.0, 0.25, 1.0]),
        host_update_assembly=False,
        setup_host_enforce=False,
        divide_by_scalxc_for_update=False,
        mode_diag_exponent=0.0,
        tree_has_tracer=lambda _value: False,
        vmec_scalxc_from_s=vmec_scalxc_from_s,
    )

    np.testing.assert_allclose(np.asarray(context.w_mode_mn), np.ones((4, 4)))
    assert context.w_mode_mn_np is None


def test_mode_transform_context_rz_norm_jax_matches_numpy_with_static_indices() -> None:
    static = _static(mpol=3, ntor=2, nfp=2, lasym=True)
    state0 = _state(ns=4, k=static.modes.m.size)
    context = build_mode_transform_context(
        static=static,
        state0=state0,
        s=np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
        host_update_assembly=False,
        setup_host_enforce=False,
        divide_by_scalxc_for_update=True,
        mode_diag_exponent=1.0,
        tree_has_tracer=lambda _value: False,
        vmec_scalxc_from_s=vmec_scalxc_from_s,
    )

    np.testing.assert_allclose(
        np.asarray(context.rz_norm(state0)),
        context.rz_norm_np(state0),
        rtol=1.0e-13,
        atol=1.0e-13,
    )

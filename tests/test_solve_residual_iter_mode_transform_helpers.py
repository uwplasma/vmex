from __future__ import annotations

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.modes import vmec_mode_table
from vmec_jax.solvers.fixed_boundary.residual.mode_transform import (
    build_mode_transform_host_projection,
    mn_cos_to_signed_host_projected,
    mn_sin_to_signed_host_projected,
    mode_diag_weights_mn,
    mode_diag_weights_mn_np,
    vmec_scalxc_from_s_np,
)
from vmec_jax.vmec_parity import (
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

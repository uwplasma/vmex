from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import (
    CoilFieldParams,
    MGridFieldParams,
    interpolate_mgrid_bfield_jax,
    sample_coil_field_cylindrical,
    sample_external_field_cylindrical,
    sample_mgrid_field_cylindrical,
)
from vmec_jax.external_fields.base import broadcast_cylindrical_coordinates
from vmec_jax.free_boundary import MGridData, MGridMetadata, interpolate_mgrid_bfield


def _affine_mgrid_params():
    from vmec_jax._compat import jnp

    rmin, rmax = 1.0, 2.0
    zmin, zmax = -0.8, 0.7
    nfp = 1
    nextcur, kp, jz, ir = 2, 6, 5, 4
    r_grid = jnp.linspace(rmin, rmax, ir)
    z_grid = jnp.linspace(zmin, zmax, jz)
    phi_grid = jnp.arange(kp, dtype=float) * ((2.0 * jnp.pi / nfp) / kp)

    coeffs = {
        "br": (
            jnp.asarray([1.2, -0.4]),
            jnp.asarray([0.7, 0.5]),
            jnp.asarray([0.3, -0.2]),
            jnp.asarray([0.1, 0.4]),
        ),
        "bphi": (
            jnp.asarray([-0.6, 0.2]),
            jnp.asarray([1.1, -0.3]),
            jnp.asarray([0.05, 0.4]),
            jnp.asarray([-0.2, 0.3]),
        ),
        "bz": (
            jnp.asarray([0.9, 0.8]),
            jnp.asarray([-0.2, 0.6]),
            jnp.asarray([0.15, 0.1]),
            jnp.asarray([0.0, -0.1]),
        ),
    }

    def build(component):
        a, b, c, d = coeffs[component]
        return (
            a[:, None, None, None] * r_grid[None, None, None, :]
            + b[:, None, None, None] * z_grid[None, None, :, None]
            + c[:, None, None, None] * phi_grid[None, :, None, None]
            + d[:, None, None, None]
        )

    params = MGridFieldParams(
        br=build("br"),
        bphi=build("bphi"),
        bz=build("bz"),
        extcur=jnp.asarray([0.8, -1.3]),
        rmin=rmin,
        rmax=rmax,
        zmin=zmin,
        zmax=zmax,
        nfp=nfp,
    )
    return params, coeffs


def _expected_affine(coeffs, component, extcur, r, z, phi):
    a, b, c, d = coeffs[component]
    a = np.asarray(a)[:, None, None]
    b = np.asarray(b)[:, None, None]
    c = np.asarray(c)[:, None, None]
    d = np.asarray(d)[:, None, None]
    per_current = a * np.asarray(r)[None, ...] + b * np.asarray(z)[None, ...] + c * np.asarray(phi)[None, ...] + d
    return np.sum(np.asarray(extcur)[:, None, None] * per_current, axis=0)


def _off_axis_coil_params():
    from vmec_jax._compat import jnp

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    # A single off-axis circular filament.  The displacement makes the field
    # vary with toroidal angle, so exact-node mgrid parity checks phi handling.
    dofs = dofs.at[0, 0, 0].set(1.65)
    dofs = dofs.at[0, 0, 2].set(0.22)
    dofs = dofs.at[0, 1, 1].set(0.22)
    dofs = dofs.at[0, 2, 0].set(0.08)
    return CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.1e5]),
        n_segments=96,
        nfp=1,
        stellsym=False,
    )


def _mgrid_from_direct_coil_nodes(coil_params):
    from vmec_jax._compat import jnp

    rmin, rmax = 0.72, 1.18
    zmin, zmax = -0.24, 0.24
    nfp = 1
    kp, jz, ir = 7, 5, 6
    r_grid = jnp.linspace(rmin, rmax, ir)
    z_grid = jnp.linspace(zmin, zmax, jz)
    phi_grid = jnp.arange(kp, dtype=float) * ((2.0 * jnp.pi / nfp) / kp)
    phi_mesh, z_mesh, r_mesh = jnp.meshgrid(phi_grid, z_grid, r_grid, indexing="ij")
    br, bphi, bz = sample_coil_field_cylindrical(coil_params, r_mesh, z_mesh, phi_mesh)
    params = MGridFieldParams(
        br=br[None, ...],
        bphi=bphi[None, ...],
        bz=bz[None, ...],
        extcur=jnp.asarray([1.0]),
        rmin=rmin,
        rmax=rmax,
        zmin=zmin,
        zmax=zmax,
        nfp=nfp,
    )
    return params, r_grid, z_grid, phi_grid


def test_mgrid_jax_affine_values_match_exact_and_legacy_interpolator():
    enable_x64(True)
    params, coeffs = _affine_mgrid_params()
    R = np.asarray([[1.12, 1.44], [1.61, 1.83]])
    Z = np.asarray([[-0.52, -0.21], [0.05, 0.31]])
    phi = np.asarray([[0.21, 0.54], [0.73, 1.02]])

    actual = sample_mgrid_field_cylindrical(params, R, Z, phi)
    dispatch = sample_external_field_cylindrical("mgrid", None, params, R, Z, phi)
    expected = (
        _expected_affine(coeffs, "br", params.extcur, R, Z, phi),
        _expected_affine(coeffs, "bphi", params.extcur, R, Z, phi),
        _expected_affine(coeffs, "bz", params.extcur, R, Z, phi),
    )

    for got, got_dispatch, want in zip(actual, dispatch, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=2.0e-14, atol=1.0e-13)
        np.testing.assert_allclose(got_dispatch, got, rtol=0.0, atol=0.0)

    legacy = interpolate_mgrid_bfield(
        MGridData(
            metadata=MGridMetadata(
                path="synthetic",
                ir=int(params.br.shape[3]),
                jz=int(params.br.shape[2]),
                kp=int(params.br.shape[1]),
                nfp=params.nfp,
                nextcur=int(params.br.shape[0]),
                rmin=params.rmin,
                rmax=params.rmax,
                zmin=params.zmin,
                zmax=params.zmax,
                mgrid_mode="S",
                coil_groups=(),
                raw_coil_cur=(),
            ),
            br=np.asarray(params.br),
            bp=np.asarray(params.bphi),
            bz=np.asarray(params.bz),
        ),
        r=R,
        z=Z,
        phi=phi,
        extcur=tuple(np.asarray(params.extcur)),
    )
    for got, want in zip(actual, legacy, strict=True):
        np.testing.assert_allclose(got, want, rtol=2.0e-14, atol=1.0e-13)


def test_mgrid_jax_generated_from_direct_coils_matches_biot_savart_at_grid_nodes():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    coil_params = _off_axis_coil_params()
    mgrid_params, r_grid, z_grid, phi_grid = _mgrid_from_direct_coil_nodes(coil_params)
    r_idx = np.asarray([[0, 2, 5], [1, 3, 4]])
    z_idx = np.asarray([[0, 2, 4], [1, 3, 2]])
    phi_idx = np.asarray([[0, 1, 3], [6, 4, 2]])
    R = jnp.asarray(np.asarray(r_grid)[r_idx])
    Z = jnp.asarray(np.asarray(z_grid)[z_idx])
    phi = jnp.asarray(np.asarray(phi_grid)[phi_idx])

    direct = sample_coil_field_cylindrical(coil_params, R, Z, phi)
    from_mgrid = sample_mgrid_field_cylindrical(mgrid_params, R, Z, phi)
    from_wrapped_mgrid = sample_mgrid_field_cylindrical(mgrid_params, R, Z, phi + 2.0 * jnp.pi)

    for got, got_wrapped, want in zip(from_mgrid, from_wrapped_mgrid, direct, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=2.0e-11, atol=2.0e-14)
        np.testing.assert_allclose(np.asarray(got_wrapped), np.asarray(want), rtol=2.0e-11, atol=2.0e-14)


def test_mgrid_jax_extcur_scales_generated_direct_coil_field_linearly():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    coil_params = _off_axis_coil_params()
    mgrid_params, r_grid, z_grid, phi_grid = _mgrid_from_direct_coil_nodes(coil_params)
    scale = 2.75
    scaled_mgrid = mgrid_params.with_arrays(extcur=jnp.asarray([scale]))
    R = jnp.asarray([[float(r_grid[1]), float(r_grid[3])]])
    Z = jnp.asarray([[float(z_grid[1]), float(z_grid[3])]])
    phi = jnp.asarray([[float(phi_grid[2]), float(phi_grid[5])]])

    direct = sample_coil_field_cylindrical(coil_params, R, Z, phi)
    scaled = sample_mgrid_field_cylindrical(scaled_mgrid, R, Z, phi)

    for got, want in zip(scaled, direct, strict=True):
        np.testing.assert_allclose(np.asarray(got), scale * np.asarray(want), rtol=2.0e-11, atol=2.0e-14)


def test_mgrid_jax_params_are_pytree_leaves_and_with_arrays_preserves_metadata():
    pytest.importorskip("jax")
    from vmec_jax._compat import tree_util

    params, _coeffs = _affine_mgrid_params()

    children, treedef = tree_util.tree_flatten(params)
    rebuilt = tree_util.tree_unflatten(treedef, children)

    assert isinstance(rebuilt, MGridFieldParams)
    assert rebuilt.rmin == params.rmin
    assert rebuilt.rmax == params.rmax
    assert rebuilt.zmin == params.zmin
    assert rebuilt.zmax == params.zmax
    assert rebuilt.nfp == params.nfp
    assert rebuilt.use_vmec_kv is params.use_vmec_kv
    for got, want in zip(children, (params.br, params.bphi, params.bz, params.extcur), strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0.0, atol=0.0)

    updated = params.with_arrays(extcur=np.asarray([2.0, -3.0]))
    assert updated.rmin == params.rmin
    np.testing.assert_allclose(np.asarray(updated.br), np.asarray(params.br), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(updated.extcur), [2.0, -3.0], rtol=0.0, atol=0.0)


def test_external_field_dispatch_rejects_unknown_provider_and_broadcasts_coordinates():
    from vmec_jax._compat import jnp

    R, Z, phi = broadcast_cylindrical_coordinates(1.0, jnp.asarray([0.0, 0.1]), 0.25)
    assert R.shape == (2,)
    np.testing.assert_allclose(np.asarray(R), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(Z), [0.0, 0.1])
    np.testing.assert_allclose(np.asarray(phi), [0.25, 0.25])

    with pytest.raises(ValueError, match=r"Unknown external-field provider"):
        sample_external_field_cylindrical("not-a-provider", None, None, R, Z, phi)


def test_mgrid_jax_rejects_invalid_field_shapes_and_current_lengths():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    params, _coeffs = _affine_mgrid_params()
    with pytest.raises(ValueError, match=r"shape"):
        interpolate_mgrid_bfield_jax(
            jnp.zeros((2, 3, 4)),
            params.bphi,
            params.bz,
            extcur=params.extcur,
            r=1.1,
            z=0.0,
            phi=0.0,
            rmin=params.rmin,
            rmax=params.rmax,
            zmin=params.zmin,
            zmax=params.zmax,
        )
    with pytest.raises(ValueError, match=r"identical"):
        interpolate_mgrid_bfield_jax(
            params.br,
            jnp.zeros((2, 5, 5, 4)),
            params.bz,
            extcur=params.extcur,
            r=1.1,
            z=0.0,
            phi=0.0,
            rmin=params.rmin,
            rmax=params.rmax,
            zmin=params.zmin,
            zmax=params.zmax,
        )
    with pytest.raises(ValueError, match=r"extcur length"):
        interpolate_mgrid_bfield_jax(
            params.br,
            params.bphi,
            params.bz,
            extcur=jnp.asarray([1.0]),
            r=1.1,
            z=0.0,
            phi=0.0,
            rmin=params.rmin,
            rmax=params.rmax,
            zmin=params.zmin,
            zmax=params.zmax,
        )
    with pytest.raises(ValueError, match=r"too small"):
        interpolate_mgrid_bfield_jax(
            jnp.zeros((1, 1, 1, 2)),
            jnp.zeros((1, 1, 1, 2)),
            jnp.zeros((1, 1, 1, 2)),
            extcur=jnp.asarray([1.0]),
            r=0.5,
            z=0.0,
            phi=0.0,
            rmin=0.0,
            rmax=1.0,
            zmin=0.0,
            zmax=1.0,
        )


def test_mgrid_jax_vmec_kv_subsamples_file_planes_like_legacy_interpolator():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    kp = 8
    br = np.zeros((1, kp, 2, 2), dtype=float)
    bphi = np.zeros_like(br)
    bz = np.zeros_like(br)
    for k in range(kp):
        br[0, k, :, :] = float(k)
        bphi[0, k, :, :] = 10.0 * float(k)
        bz[0, k, :, :] = -float(k)
    r = np.full((2, 4), 0.5)
    z = np.full((2, 4), 0.5)
    phi = np.zeros((2, 4))

    actual = interpolate_mgrid_bfield_jax(
        jnp.asarray(br),
        jnp.asarray(bphi),
        jnp.asarray(bz),
        extcur=jnp.asarray([1.0]),
        r=jnp.asarray(r),
        z=jnp.asarray(z),
        phi=jnp.asarray(phi),
        rmin=0.0,
        rmax=1.0,
        zmin=0.0,
        zmax=1.0,
        nfp=1,
        use_vmec_kv=True,
    )
    legacy = interpolate_mgrid_bfield(
        MGridData(
            metadata=MGridMetadata(
                path="synthetic",
                ir=2,
                jz=2,
                kp=kp,
                nfp=1,
                nextcur=1,
                rmin=0.0,
                rmax=1.0,
                zmin=0.0,
                zmax=1.0,
                mgrid_mode="S",
                coil_groups=(),
                raw_coil_cur=(),
            ),
            br=br,
            bp=bphi,
            bz=bz,
        ),
        r=r,
        z=z,
        phi=phi,
        extcur=(1.0,),
        use_vmec_kv=True,
    )
    expected = np.broadcast_to(np.asarray([0.0, 2.0, 4.0, 6.0])[None, :], r.shape)
    np.testing.assert_allclose(np.asarray(actual[0]), expected, rtol=0.0, atol=1e-14)
    for got, want in zip(actual, legacy, strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=0.0, atol=1e-14)


def test_mgrid_jax_vmec_kv_validation_and_single_plane_case():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    fields = jnp.ones((1, 1, 2, 2))
    actual = interpolate_mgrid_bfield_jax(
        fields,
        2.0 * fields,
        -fields,
        extcur=jnp.asarray([3.0]),
        r=jnp.full((2, 3), 0.25),
        z=jnp.full((2, 3), 0.75),
        phi=jnp.zeros((2, 3)),
        rmin=0.0,
        rmax=1.0,
        zmin=0.0,
        zmax=1.0,
        use_vmec_kv=True,
    )
    np.testing.assert_allclose(np.asarray(actual[0]), 3.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(actual[1]), 6.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(actual[2]), -3.0, rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match=r"explicit zeta axis"):
        interpolate_mgrid_bfield_jax(
            fields,
            fields,
            fields,
            extcur=jnp.asarray([1.0]),
            r=0.25,
            z=0.75,
            phi=0.0,
            rmin=0.0,
            rmax=1.0,
            zmin=0.0,
            zmax=1.0,
            use_vmec_kv=True,
        )

    bad = jnp.ones((1, 5, 2, 2))
    with pytest.raises(ValueError, match=r"must be divisible"):
        interpolate_mgrid_bfield_jax(
            bad,
            bad,
            bad,
            extcur=jnp.asarray([1.0]),
            r=jnp.full((1, 3), 0.25),
            z=jnp.full((1, 3), 0.75),
            phi=jnp.zeros((1, 3)),
            rmin=0.0,
            rmax=1.0,
            zmin=0.0,
            zmax=1.0,
            use_vmec_kv=True,
        )


def test_mgrid_jax_gradient_wrt_extcur_matches_per_current_values():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    params, coeffs = _affine_mgrid_params()
    R = jnp.asarray([1.22, 1.48])
    Z = jnp.asarray([-0.42, -0.11])
    phi = jnp.asarray([0.25, 0.65])

    def objective(extcur):
        trial = params.with_arrays(extcur=extcur)
        br, _bphi, _bz = sample_mgrid_field_cylindrical(trial, R, Z, phi)
        return jnp.sum(br)

    grad_extcur = np.asarray(jax.grad(objective)(params.extcur))
    a, b, c, d = coeffs["br"]
    expected = np.asarray(jnp.sum(a[:, None] * R[None, :] + b[:, None] * Z[None, :] + c[:, None] * phi[None, :] + d[:, None], axis=1))

    np.testing.assert_allclose(grad_extcur, expected, rtol=2.0e-14, atol=1.0e-13)


def test_mgrid_jax_gradient_wrt_field_value_matches_trilinear_weight():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)
    params, _coeffs = _affine_mgrid_params()
    R = 1.31
    Z = -0.37
    phi = 0.41

    def objective(br_values):
        trial = params.with_arrays(br=br_values)
        br, _bphi, _bz = sample_mgrid_field_cylindrical(trial, R, Z, phi)
        return br

    grad_br = np.asarray(jax.grad(objective)(params.br))

    ir = int(params.br.shape[3])
    jz = int(params.br.shape[2])
    kp = int(params.br.shape[1])
    fr = (R - params.rmin) * ((ir - 1) / (params.rmax - params.rmin))
    fz = (Z - params.zmin) * ((jz - 1) / (params.zmax - params.zmin))
    fk = phi * (kp / (2.0 * np.pi / params.nfp))
    i0 = int(np.floor(fr))
    j0 = int(np.floor(fz))
    k0 = int(np.floor(fk))
    expected_weight = float(params.extcur[0]) * (1.0 - (fr - i0)) * (1.0 - (fz - j0)) * (1.0 - (fk - k0))

    np.testing.assert_allclose(grad_br[0, k0, j0, i0], expected_weight, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.sum(grad_br), np.sum(np.asarray(params.extcur)), rtol=1.0e-13, atol=1.0e-13)


def test_mgrid_jax_coordinate_derivative_matches_affine_coefficients():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    params, coeffs = _affine_mgrid_params()

    def objective(coords):
        R, Z, phi = coords
        br, _bphi, _bz = interpolate_mgrid_bfield_jax(
            params.br,
            params.bphi,
            params.bz,
            extcur=params.extcur,
            r=R,
            z=Z,
            phi=phi,
            rmin=params.rmin,
            rmax=params.rmax,
            zmin=params.zmin,
            zmax=params.zmax,
            nfp=params.nfp,
        )
        return br

    coords = jnp.asarray([1.37, -0.29, 0.52])
    exact = np.asarray(jax.grad(objective)(coords))
    a, b, c, _d = coeffs["br"]
    expected = np.asarray(
        [
            jnp.sum(params.extcur * a),
            jnp.sum(params.extcur * b),
            jnp.sum(params.extcur * c),
        ]
    )

    np.testing.assert_allclose(exact, expected, rtol=3.0e-13, atol=1.0e-13)

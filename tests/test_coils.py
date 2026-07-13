"""Tests for ``vmec_jax.core.coils``: ESSOS cross-validation + properties.

Covers:

- ``CoilSet.from_essos`` + field parity vs ``essos.fields.BiotSavart`` at
  rtol 1e-10,
- ``to_mgrid_data`` + ``write_mgrid`` vs ESSOS ``coils_to_mgrid`` (PR#33):
  per-group B arrays at rtol 1e-8, cross-readable netCDF files,
- grad of ``sum |B|^2`` w.r.t. currents and Fourier dofs (finite/nonzero) and
  jit equivalence.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmec_jax.core.coils import (  # noqa: E402
    CoilSet,
    b_cyl,
    biot_savart,
    circular_loop_on_axis_bz,
    coil_geometry,
    field_on_cylindrical_grid,
    planar_ellipse_coils,
    square_mirror_coils,
    tokamak_coils,
    two_coil_on_axis_bz,
    to_mgrid_data,
)
from vmec_jax.core.mgrid import MgridField, read_mgrid, write_mgrid  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _random_coilset(seed: int = 7) -> CoilSet:
    """Random Fourier coils: 3 base coils, order 4, nfp=2, stellsym."""

    rng = np.random.default_rng(seed)
    n_coils, order = 3, 4
    dofs = np.zeros((n_coils, 3, 2 * order + 1))
    # A sane toroidal-field-like base so points stay off the centerlines...
    angles = (np.arange(n_coils) + 0.5) * 2.0 * np.pi / (2 * 2 * n_coils)
    dofs[:, 0, 0] = 3.0 * np.cos(angles)
    dofs[:, 1, 0] = 3.0 * np.sin(angles)
    dofs[:, 0, 2] = 1.0 * np.cos(angles)
    dofs[:, 1, 2] = 1.0 * np.sin(angles)
    dofs[:, 2, 1] = -1.0
    # ...plus random higher harmonics.
    dofs += 0.05 * rng.standard_normal(dofs.shape)
    currents = 1.0e5 * (1.0 + 0.3 * rng.standard_normal(n_coils))
    return CoilSet(
        base_curve_dofs=jnp.asarray(dofs),
        base_currents=jnp.asarray(currents),
        n_segments=48,
        nfp=2,
        stellsym=True,
        current_scale=1.0,
    )


def _random_points(n: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.uniform(1.5, 4.5, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = rng.uniform(-1.5, 1.5, size=n)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


@pytest.fixture(scope="module")
def coilset() -> CoilSet:
    return _random_coilset()


def _essos_coils():
    """Small ESSOS coil set built from ESSOS primitives (skips if missing)."""

    essos_coils_mod = pytest.importorskip("essos.coils")
    curves = essos_coils_mod.CreateEquallySpacedCurves(
        n_curves=2, order=3, R=1.5, r=0.5, n_segments=40, nfp=2, stellsym=True
    )
    currents = jnp.array([1.0e5, 1.1e5])
    return essos_coils_mod.Coils(curves, currents)


# ---------------------------------------------------------------------------
# Consistency properties
# ---------------------------------------------------------------------------


def test_b_cyl_matches_cartesian_rotation(coilset: CoilSet) -> None:
    """b_cyl equals the Cartesian Biot-Savart field rotated to (R, phi, Z)."""
    rng = np.random.default_rng(3)
    r = jnp.asarray(rng.uniform(1.5, 4.5, size=100))
    phi = jnp.asarray(rng.uniform(0.0, 2.0 * np.pi, size=100))
    z = jnp.asarray(rng.uniform(-1.5, 1.5, size=100))
    br, bp, bz = b_cyl(coilset, r, phi, z)
    points = jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=-1)
    b_xyz = biot_savart(coilset, points)
    assert np.all(np.isfinite(np.asarray(b_xyz)))
    br_ref = b_xyz[:, 0] * jnp.cos(phi) + b_xyz[:, 1] * jnp.sin(phi)
    bp_ref = -b_xyz[:, 0] * jnp.sin(phi) + b_xyz[:, 1] * jnp.cos(phi)
    scale = float(jnp.max(jnp.abs(b_xyz)))
    for new, ref in ((br, br_ref), (bp, bp_ref), (bz, b_xyz[:, 2])):
        np.testing.assert_allclose(np.asarray(new), np.asarray(ref), rtol=1e-12, atol=1e-15 * scale)


def test_biot_savart_chunked_matches_unchunked(coilset: CoilSet) -> None:
    points = jnp.asarray(_random_points(n=37, seed=5))
    b_full = np.asarray(biot_savart(coilset, points))
    from dataclasses import replace

    b_chunk = np.asarray(biot_savart(replace(coilset, chunk_size=8), points))
    np.testing.assert_allclose(b_chunk, b_full, rtol=1e-12, atol=1e-15 * np.max(np.abs(b_full)))


def test_two_coil_on_axis_formula_matches_discretized_biot_savart() -> None:
    radius, separation, current = 0.8, 1.6, 2.5e5
    dofs = np.zeros((2, 3, 3))
    dofs[:, 0, 2] = radius  # x = radius*cos(2*pi*t)
    dofs[:, 1, 1] = radius  # y = radius*sin(2*pi*t)
    dofs[:, 2, 0] = np.array([-0.5 * separation, 0.5 * separation])
    coils = CoilSet(
        base_curve_dofs=jnp.asarray(dofs),
        base_currents=jnp.asarray([current, current]),
        n_segments=512,
    )
    z = jnp.linspace(-0.7, 0.7, 15)
    points = jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z], axis=-1)
    direct = biot_savart(coils, points)
    analytic = two_coil_on_axis_bz(
        z,
        coil_radius=radius,
        separation=separation,
        current=current,
    )
    np.testing.assert_allclose(direct[:, :2], 0.0, atol=2.0e-16)
    np.testing.assert_allclose(direct[:, 2], analytic, rtol=2.0e-12, atol=2.0e-15)
    np.testing.assert_allclose(
        circular_loop_on_axis_bz(0.0, loop_radius=radius, current=current),
        4.0e-7 * np.pi * current / (2.0 * radius),
        rtol=2.0e-14,
    )
    assert np.isfinite(
        float(
            jax.grad(lambda a: two_coil_on_axis_bz(0.0, coil_radius=a, separation=separation, current=current))(radius)
        )
    )


def test_planar_ellipse_constructor_preserves_geometry_and_orientation() -> None:
    coils = planar_ellipse_coils(
        [[1.0, 2.0, 3.0]],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        semi_major=0.6,
        semi_minor=0.3,
        currents=2.0e5,
        n_segments=64,
    )
    dofs = np.asarray(coils.base_curve_dofs[0])
    np.testing.assert_allclose(dofs[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.linalg.norm(dofs[:, 2]), 0.6)
    np.testing.assert_allclose(np.linalg.norm(dofs[:, 1]), 0.3)
    normal = np.cross(dofs[:, 2], dofs[:, 1])
    np.testing.assert_allclose(normal / np.linalg.norm(normal), [0.0, 1.0, 0.0])


def test_square_mirror_coils_builds_four_ordered_sides() -> None:
    coils = square_mirror_coils(n_per_side=4, side_length=3.0, n_segments=32)
    centers = np.asarray(coils.base_curve_dofs[:, :, 0])
    assert coils.n_base_coils == 16
    np.testing.assert_allclose(centers[:, 2], 0.0)
    assert np.all(np.sum(np.isclose(np.abs(centers[:, :2]), 1.5), axis=1) == 1)
    field = biot_savart(coils, jnp.asarray([[0.0, 0.0, 0.0], [0.2, 0.1, 0.3]]))
    assert np.all(np.isfinite(np.asarray(field)))


# ---------------------------------------------------------------------------
# A/B vs ESSOS BiotSavart
# ---------------------------------------------------------------------------


def test_from_essos_matches_essos_biot_savart() -> None:
    fields_mod = pytest.importorskip("essos.fields")
    coils = _essos_coils()
    cs = CoilSet.from_essos(coils)
    assert cs.n_base_coils == 2
    assert cs.nfp == 2
    assert cs.stellsym is True
    assert cs.n_coils == 8

    rng = np.random.default_rng(11)
    n = 100
    r = rng.uniform(0.8, 2.2, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = rng.uniform(-0.4, 0.4, size=n)
    points = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)

    b_new = np.asarray(biot_savart(cs, jnp.asarray(points)))
    field = fields_mod.BiotSavart(coils)
    b_essos = np.asarray(jax.vmap(field.B)(jnp.asarray(points)))
    scale = np.max(np.abs(b_essos))
    np.testing.assert_allclose(b_new, b_essos, rtol=1e-10, atol=1e-12 * scale)


def test_from_essos_rejects_non_essos_objects() -> None:
    with pytest.raises(ImportError, match="essos.coils.Coils"):
        CoilSet.from_essos(object())


# ---------------------------------------------------------------------------
# mgrid export vs ESSOS coils_to_mgrid (PR#33)
# ---------------------------------------------------------------------------


def test_to_mgrid_matches_essos_coils_to_mgrid(tmp_path) -> None:
    essos_mgrid_mod = pytest.importorskip("essos.mgrid")
    coils = _essos_coils()
    cs = CoilSet.from_essos(coils)

    grid = dict(rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5)
    ir, jz, kp = 8, 6, 4

    essos_file = tmp_path / "mgrid_essos.nc"
    essos_grid = essos_mgrid_mod.coils_to_mgrid(coils, essos_file, nr=ir, nz=jz, nphi=kp, **grid)

    # ESSOS lumps all coils into one raw group (mode "N", raw_coil_cur = 1).
    data = to_mgrid_data(cs, ir=ir, jz=jz, kp=kp, mgrid_mode="N", single_group=True, **grid)
    assert data.nextcur == 1
    assert data.mgrid_mode == "N"
    assert data.raw_coil_cur == (1.0,)
    assert (data.ir, data.jz, data.kp, data.nfp) == (ir, jz, kp, 2)

    scale = max(np.max(np.abs(np.asarray(a[0]))) for a in (essos_grid.br_arr, essos_grid.bp_arr, essos_grid.bz_arr))
    for ours, theirs in ((data.br, essos_grid.br_arr), (data.bp, essos_grid.bp_arr), (data.bz, essos_grid.bz_arr)):
        assert ours.shape == (1, kp, jz, ir)
        np.testing.assert_allclose(ours[0], np.asarray(theirs[0]), rtol=1e-8, atol=1e-10 * scale)

    # write_mgrid file vs the ESSOS-written file: same per-group tables.
    ours_file = tmp_path / "mgrid_vmecjax.nc"
    write_mgrid(ours_file, data)
    back_ours = read_mgrid(ours_file)
    back_essos = read_mgrid(essos_file)
    assert back_ours.nextcur == back_essos.nextcur == 1
    assert (back_ours.ir, back_ours.jz, back_ours.kp, back_ours.nfp) == (
        back_essos.ir,
        back_essos.jz,
        back_essos.kp,
        back_essos.nfp,
    )
    for name in ("rmin", "rmax", "zmin", "zmax"):
        assert getattr(back_ours, name) == getattr(back_essos, name)
    for name in ("br", "bp", "bz"):
        np.testing.assert_allclose(getattr(back_ours, name), getattr(back_essos, name), rtol=1e-8, atol=1e-10 * scale)

    # ESSOS' own reader accepts our file.
    theirs_read = essos_mgrid_mod.MGrid.from_file(ours_file)
    np.testing.assert_allclose(np.asarray(theirs_read.br_arr[0]), data.br[0], rtol=1e-12)


def test_per_group_mgrid_modes_consistent() -> None:
    """Sum over 'S'-mode groups weighted by raw_coil_cur == raw total field."""

    cs = _random_coilset(seed=19)
    grid = dict(rmin=2.0, rmax=4.0, zmin=-1.0, zmax=1.0)
    ir, jz, kp = 6, 5, 4

    data_s = to_mgrid_data(cs, ir=ir, jz=jz, kp=kp, mgrid_mode="S", **grid)
    assert data_s.nextcur == cs.n_base_coils
    np.testing.assert_allclose(data_s.raw_coil_cur, np.asarray(cs.base_currents), rtol=1e-15)

    br_raw, bp_raw, bz_raw = field_on_cylindrical_grid(cs, ir=ir, jz=jz, kp=kp, **grid)
    cur = np.asarray(data_s.raw_coil_cur)[:, None, None, None]
    for scaled, raw in ((data_s.br, br_raw), (data_s.bp, bp_raw), (data_s.bz, bz_raw)):
        total_from_s = np.sum(cur * scaled, axis=0)
        total_raw = np.sum(np.asarray(raw), axis=0)
        np.testing.assert_allclose(total_from_s, total_raw, rtol=1e-12, atol=1e-15 * np.max(np.abs(total_raw)))

    # Raw per-group field must equal the total field of the full coil set.
    br1, bp1, bz1 = field_on_cylindrical_grid(cs, ir=ir, jz=jz, kp=kp, single_group=True, **grid)
    total_br = np.sum(np.asarray(br_raw), axis=0)
    np.testing.assert_allclose(total_br, np.asarray(br1[0]), rtol=1e-12, atol=1e-15 * np.max(np.abs(total_br)))


def test_mgrid_off_grid_error_converges_for_smooth_loop_field() -> None:
    """Trilinear mgrid error decreases under refinement away from coils."""

    coils = planar_ellipse_coils(
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0]],
        semi_major=1.0,
        semi_minor=1.0,
        currents=1.0e5,
        n_segments=128,
    )
    r = np.array([0.13, 0.21, 0.34, 0.47, 0.59])
    phi = np.array([0.1, 0.7, 1.4, 2.1, 3.0])
    z = np.array([0.16, 0.23, 0.31, 0.42, 0.54])
    direct = np.stack([np.asarray(x) for x in coils.b_cyl(r, phi, z)], axis=-1)
    scale = np.linalg.norm(direct)

    errors = []
    for n in (9, 17, 33):
        data = to_mgrid_data(
            coils, 0.0, 0.7, 0.1, 0.6, ir=n, jz=n, kp=4, mgrid_mode="S"
        )
        field = MgridField.from_mgrid_data(data)
        sampled = np.stack([np.asarray(x) for x in field.b_cyl(r, phi, z)], axis=-1)
        errors.append(np.linalg.norm(sampled - direct) / scale)

    np.testing.assert_allclose(errors, [1.1613121e-3, 2.1271369e-4, 3.3976697e-5], rtol=2e-5)
    assert errors[1] < errors[0] / 4.0
    assert errors[2] < errors[1] / 4.0


def test_tokamak_coils_build_tf_and_pf_geometry() -> None:
    coils = tokamak_coils(
        major_radius=2.0,
        tf_coil_radius=0.8,
        tf_current=3.0e5,
        pf_coils=[[1.2, 1.0, -2.0e5], [1.2, -1.0, -2.0e5]],
        n_tf_coils=8,
        n_segments=48,
    )
    assert coils.n_base_coils == 10
    np.testing.assert_allclose(np.asarray(coils.base_currents[:8]), 3.0e5)
    np.testing.assert_allclose(np.asarray(coils.base_currents[8:]), -2.0e5)
    gamma, _gamma_dash, _currents = coil_geometry(coils)
    centers = np.mean(np.asarray(gamma), axis=1)
    np.testing.assert_allclose(np.linalg.norm(centers[:8, :2], axis=1), 2.0, atol=1e-14)
    np.testing.assert_allclose(centers[8:, 2], [1.0, -1.0], atol=1e-14)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"n_tf_coils": 3}, "at least 4"),
        ({"pf_coils": [[1.0, 0.0]]}, "shape"),
        ({"pf_coils": [[-1.0, 0.0, 1.0]]}, "radii"),
    ],
)
def test_tokamak_coils_validates_inputs(changes, match) -> None:
    args = dict(
        major_radius=2.0,
        tf_coil_radius=0.8,
        tf_current=3.0e5,
        pf_coils=[],
        n_tf_coils=8,
    )
    args.update(changes)
    with pytest.raises(ValueError, match=match):
        tokamak_coils(**args)


# ---------------------------------------------------------------------------
# Differentiability + jit
# ---------------------------------------------------------------------------


def test_grad_wrt_currents_and_dofs(coilset: CoilSet) -> None:
    points = jnp.asarray(_random_points(n=20, seed=9))

    def loss(currents: jnp.ndarray, dofs: jnp.ndarray) -> jnp.ndarray:
        cs = coilset.with_arrays(base_curve_dofs=dofs, base_currents=currents)
        b = biot_savart(cs, points)
        return jnp.sum(b * b)

    g_cur = jax.grad(loss, argnums=0)(coilset.base_currents, coilset.base_curve_dofs)
    g_dofs = jax.grad(loss, argnums=1)(coilset.base_currents, coilset.base_curve_dofs)
    assert np.all(np.isfinite(np.asarray(g_cur)))
    assert np.all(np.isfinite(np.asarray(g_dofs)))
    assert np.all(np.abs(np.asarray(g_cur)) > 0.0)
    # One specific Fourier dof (coil 0, x component, cos(2*pi*t) coefficient).
    assert abs(float(g_dofs[0, 0, 2])) > 0.0

    # Finite-difference cross-check on the currents gradient.
    eps = 1.0
    e0 = jnp.zeros_like(coilset.base_currents).at[0].set(eps)
    fd = (
        float(loss(coilset.base_currents + e0, coilset.base_curve_dofs))
        - float(loss(coilset.base_currents - e0, coilset.base_curve_dofs))
    ) / (2.0 * eps)
    np.testing.assert_allclose(float(g_cur[0]), fd, rtol=1e-6)


def test_jit_equivalence(coilset: CoilSet) -> None:
    points = jnp.asarray(_random_points(n=30, seed=13))
    b_eager = np.asarray(biot_savart(coilset, points))
    b_jit = np.asarray(jax.jit(biot_savart)(coilset, points))
    np.testing.assert_allclose(b_jit, b_eager, rtol=1e-12, atol=1e-15 * np.max(np.abs(b_eager)))

    r, phi, z = jnp.asarray([3.1, 2.7]), jnp.asarray([0.3, 4.0]), jnp.asarray([0.2, -0.4])
    eager = b_cyl(coilset, r, phi, z)
    jitted = jax.jit(b_cyl)(coilset, r, phi, z)
    for a, b in zip(jitted, eager):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)

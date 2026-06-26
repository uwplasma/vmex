from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import tree_util
from vmec_jax.kernels.tomnsp import (
    TomnspsRZL,
    _GRID_CACHE,
    _TOMNSPS_MASK_CACHE,
    _TRIG_CACHE,
    tomnspa_rzl,
    tomnsps_masks,
    tomnsps_rzl,
    vmec_angle_grid,
    vmec_theta_sizes,
    vmec_trig_tables,
)


def test_vmec_theta_sizes_match_read_indata_logic():
    ntheta1, ntheta2, ntheta3 = vmec_theta_sizes(22, lasym=False)
    assert ntheta1 == 22
    assert ntheta2 == 12
    assert ntheta3 == 12


def test_vmec_angle_grid_half_interval_includes_pi_when_symmetric():
    g = vmec_angle_grid(ntheta=22, nzeta=5, nfp=3, lasym=False)
    assert g.theta.size == 12
    assert np.isclose(g.theta[0], 0.0)
    assert np.isclose(g.theta[-1], np.pi)
    assert g.zeta.size == 5
    assert np.isclose(g.zeta[0], 0.0)
    assert np.isclose(g.zeta[-1], 2.0 * np.pi * (4.0 / 5.0))


def test_vmec_trig_tables_include_nfp_in_derivative_tables():
    t = vmec_trig_tables(ntheta=22, nzeta=8, nfp=3, mmax=4, nmax=4, lasym=False)
    # For n=1: cosnvn = (n*nfp)*cosnv, sinnvn = -(n*nfp)*sinnv.
    n = 1
    assert np.allclose(np.asarray(t.cosnvn)[:, n], (n * 3) * np.asarray(t.cosnv)[:, n])
    assert np.allclose(np.asarray(t.sinnvn)[:, n], -(n * 3) * np.asarray(t.sinnv)[:, n])


def test_vmec_trig_tables_cosmui3_matches_fixaray_behavior():
    # lasym=False: ntheta3==ntheta2, so cosmui3 is the same as cosmui (endpoint half-weights).
    t = vmec_trig_tables(ntheta=10, nzeta=7, nfp=3, mmax=6, nmax=4, lasym=False)
    assert t.ntheta2 == t.ntheta3
    assert np.allclose(np.asarray(t.cosmui3), np.asarray(t.cosmui))


def test_vmec_trig_tables_dnorm_and_dnorm3_match_fixaray():
    # dnorm always uses the reduced interval [0, pi].
    t = vmec_trig_tables(ntheta=22, nzeta=9, nfp=3, mmax=4, nmax=4, lasym=False)
    assert np.isclose(t.dnorm, 1.0 / (9 * (t.ntheta2 - 1)))
    assert np.isclose(t.dnorm3, t.dnorm)


def test_grid_and_trig_cache_keys_are_parameter_specific():
    _GRID_CACHE.clear()
    _TRIG_CACHE.clear()

    grid1 = vmec_angle_grid(ntheta=8, nzeta=0, nfp=2, lasym=False)
    grid2 = vmec_angle_grid(ntheta=8, nzeta=1, nfp=2, lasym=False)
    grid3 = vmec_angle_grid(ntheta=8, nzeta=1, nfp=3, lasym=False)
    assert grid1 is grid2
    assert grid3 is not grid1
    assert grid1.zeta.size == 1
    with pytest.raises(ValueError, match="Invalid theta sizes"):
        vmec_angle_grid(ntheta=0, nzeta=1, nfp=1, lasym=False)
    with pytest.raises(ValueError, match="nfp must be positive"):
        vmec_angle_grid(ntheta=8, nzeta=1, nfp=0, lasym=False)

    trig1 = vmec_trig_tables(ntheta=8, nzeta=3, nfp=2, mmax=2, nmax=1, lasym=True, dtype=np.float64)
    trig2 = vmec_trig_tables(ntheta=8, nzeta=3, nfp=2, mmax=2, nmax=1, lasym=True, dtype=np.float64)
    trig3 = vmec_trig_tables(ntheta=8, nzeta=3, nfp=2, mmax=2, nmax=1, lasym=True, dtype=np.float32)
    assert trig1 is trig2
    assert trig3 is not trig1
    assert np.isclose(trig1.dnorm, 1.0 / (3 * trig1.ntheta3))
    with pytest.raises(ValueError, match="Invalid theta sizes"):
        vmec_trig_tables(ntheta=0, nzeta=3, nfp=1, mmax=1, nmax=1, lasym=False)
    with pytest.raises(ValueError, match="nfp must be positive"):
        vmec_trig_tables(ntheta=8, nzeta=3, nfp=0, mmax=1, nmax=1, lasym=False)
    with pytest.raises(ValueError, match="mmax/nmax must be nonnegative"):
        vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=-1, nmax=1, lasym=False)


def test_trig_tables_and_tomnsps_are_pytree_roundtrippable():
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=2, mmax=2, nmax=1, lasym=True, cache=False)
    leaves, treedef = tree_util.tree_flatten(trig)
    rebuilt = tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.ntheta3 == trig.ntheta3
    np.testing.assert_allclose(np.asarray(rebuilt.cosnvn), np.asarray(trig.cosnvn))

    shape = (3, 2, 2)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape)
    frzl = TomnspsRZL(
        frcc=base,
        frss=base + 1,
        fzsc=base + 2,
        fzcs=base + 3,
        flsc=base + 4,
        flcs=base + 5,
        frsc=base + 6,
        frcs=base + 7,
        fzcc=base + 8,
        fzss=base + 9,
        flcc=base + 10,
        flss=base + 11,
    )
    leaves, treedef = tree_util.tree_flatten(frzl)
    rebuilt = tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(rebuilt.flss), np.asarray(frzl.flss))


def test_tomnsps_masks_validate_and_cache_by_dtype_and_edge_policy():
    _TOMNSPS_MASK_CACHE.clear()
    with pytest.raises(ValueError, match="ns must be positive"):
        tomnsps_masks(ns=0, mpol=1, include_edge=False)
    with pytest.raises(ValueError, match="mpol must be positive"):
        tomnsps_masks(ns=1, mpol=0, include_edge=False)

    no_edge64 = tomnsps_masks(ns=3, mpol=2, include_edge=False, dtype=np.float64)
    no_edge64_again = tomnsps_masks(ns=3, mpol=2, include_edge=False, dtype=np.float64)
    edge64 = tomnsps_masks(ns=3, mpol=2, include_edge=True, dtype=np.float64)
    no_edge32 = tomnsps_masks(ns=3, mpol=2, include_edge=False, dtype=np.float32)
    assert no_edge64_again is no_edge64
    assert edge64 is not no_edge64
    assert no_edge32 is not no_edge64
    np.testing.assert_array_equal(np.asarray(no_edge64.mask_rz[:, :, 0]), [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    np.testing.assert_array_equal(np.asarray(edge64.mask_rz[:, :, 0])[-1], [1.0, 1.0])


def _tiny_transform_inputs(shape):
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10.0
    return {
        "armn_even": base + 0.1,
        "armn_odd": base + 0.2,
        "brmn_even": base + 0.3,
        "brmn_odd": base + 0.4,
        "crmn_even": base + 0.5,
        "crmn_odd": base + 0.6,
        "azmn_even": base + 0.7,
        "azmn_odd": base + 0.8,
        "bzmn_even": base + 0.9,
        "bzmn_odd": base + 1.0,
        "czmn_even": base + 1.1,
        "czmn_odd": base + 1.2,
    }


def test_tomnsps_transform_guards_masks_and_optional_blocks():
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    args = _tiny_transform_inputs((3, trig.ntheta3, 3))

    with pytest.raises(ValueError, match="mpol must be positive"):
        tomnsps_rzl(**args, mpol=0, ntor=1, nfp=1, lasym=True, trig=trig)
    with pytest.raises(ValueError, match="ntor must be nonnegative"):
        tomnsps_rzl(**args, mpol=2, ntor=-1, nfp=1, lasym=True, trig=trig)
    bad_trig = vmec_trig_tables(ntheta=10, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    with pytest.raises(ValueError, match="Input grid does not match trig tables"):
        tomnsps_rzl(**args, mpol=2, ntor=1, nfp=1, lasym=True, trig=bad_trig)

    masks = tomnsps_masks(ns=3, mpol=2, include_edge=False)
    out = tomnsps_rzl(**args, mpol=2, ntor=1, nfp=1, lasym=True, trig=trig, masks=masks)
    assert out.frcc.shape == (3, 2, 2)
    assert out.frss is not None
    np.testing.assert_allclose(np.asarray(out.frcc[-1]), 0.0)
    np.testing.assert_allclose(np.asarray(out.fzsc[-1]), 0.0)
    np.testing.assert_allclose(np.asarray(out.flsc[0]), 0.0)

    axisym = tomnsps_rzl(**args, mpol=2, ntor=0, nfp=1, lasym=True, trig=trig)
    assert axisym.frss is None
    assert axisym.fzcs is None
    assert axisym.flcs is None


def test_tomnspa_transform_guards_and_asymmetric_blocks(monkeypatch):
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    args = _tiny_transform_inputs((3, trig.ntheta3, 3))
    with pytest.raises(ValueError, match="mpol must be positive"):
        tomnspa_rzl(**args, mpol=0, ntor=1, nfp=1, lasym=True, trig=trig)
    with pytest.raises(ValueError, match="ntor must be nonnegative"):
        tomnspa_rzl(**args, mpol=2, ntor=-1, nfp=1, lasym=True, trig=trig)

    monkeypatch.setenv("VMEC_JAX_TOMNSPA_LAM_SCALE", "1.25")
    out = tomnspa_rzl(
        **args,
        mpol=2,
        ntor=1,
        nfp=1,
        lasym=True,
        trig=trig,
        masks=tomnsps_masks(ns=3, mpol=2, include_edge=False),
    )
    np.testing.assert_allclose(np.asarray(out.frcc), 0.0)
    assert out.frsc is not None
    assert out.frcs is not None
    assert out.fzcc is not None
    assert out.fzss is not None
    assert out.flcc is not None
    assert out.flss is not None
    np.testing.assert_allclose(np.asarray(out.frsc[-1]), 0.0)
    np.testing.assert_allclose(np.asarray(out.flcc[0]), 0.0)

    axisym = tomnspa_rzl(**args, mpol=2, ntor=0, nfp=1, lasym=True, trig=trig)
    assert axisym.frcs is None
    assert axisym.fzss is None
    assert axisym.flss is None

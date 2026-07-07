from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import tree_util
import vmec_jax.coords as coords_mod
import vmec_jax.fourier as fourier_mod
from vmec_jax.coords import Coords
from vmec_jax.fourier import (
    HelicalBasis,
    _HELICAL_BASIS_CACHE,
    build_helical_basis,
    eval_fourier,
    eval_fourier_dtheta,
    eval_fourier_dzeta_phys,
    project_to_modes,
)
from vmec_jax.geom import Geom, _cross, _dot
from vmec_jax.grids import AngleGrid
from vmec_jax.modes import ModeTable


def test_helical_basis_cache_pytree_and_phase_stack_fallbacks():
    modes = ModeTable(m=np.asarray([0, 1]), n=np.asarray([0, -1]))
    grid = AngleGrid(theta=np.asarray([0.0, np.pi / 2]), zeta=np.asarray([0.0, np.pi]), nfp=3)
    _HELICAL_BASIS_CACHE.clear()

    basis = build_helical_basis(modes, grid)
    cached = build_helical_basis(modes, grid)
    uncached = build_helical_basis(modes, grid, cache=False)

    assert cached is basis
    assert uncached is not basis
    assert basis.phase_stack.shape[0] == 2 * modes.K

    leaves, treedef = tree_util.tree_flatten(basis)
    rebuilt = tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(rebuilt.cos_phase), np.asarray(basis.cos_phase))
    assert rebuilt.nfp == 3

    no_stack = HelicalBasis(
        cos_phase=basis.cos_phase,
        sin_phase=basis.sin_phase,
        phase_stack=None,
        m=basis.m,
        n=basis.n,
        nfp=basis.nfp,
    )
    c = np.asarray([[2.0, 0.5]])
    s = np.asarray([[0.0, -0.25]])

    np.testing.assert_allclose(
        np.asarray(eval_fourier(c, s, no_stack)),
        np.asarray(eval_fourier(c, s, basis)),
    )
    np.testing.assert_allclose(
        np.asarray(eval_fourier_dtheta(c, s, no_stack)),
        np.asarray(eval_fourier_dtheta(c, s, basis)),
    )
    np.testing.assert_allclose(
        np.asarray(eval_fourier_dzeta_phys(c, s, no_stack)),
        np.asarray(eval_fourier_dzeta_phys(c, s, basis)),
    )

    internal = np.asarray(eval_fourier(c, s, basis, coeffs_internal=True))
    physical = np.asarray(eval_fourier(c, s, basis, coeffs_internal=False))
    assert not np.allclose(internal, physical)


def test_pytree_registration_helpers_tolerate_duplicate_registration(monkeypatch):
    class Dummy:
        pass

    def duplicate_registration(cls):
        raise ValueError("Duplicate custom PyTreeDef type registration")

    monkeypatch.setattr(fourier_mod, "_register_pytree_node_class", duplicate_registration, raising=False)
    monkeypatch.setattr(coords_mod, "_register_pytree_node_class", duplicate_registration, raising=False)
    assert fourier_mod.register_pytree_node_class(Dummy) is Dummy
    assert coords_mod.register_pytree_node_class(Dummy) is Dummy

    def different_registration_error(cls):
        raise ValueError("different registration error")

    monkeypatch.setattr(fourier_mod, "_register_pytree_node_class", different_registration_error, raising=False)
    with pytest.raises(ValueError, match="different registration error"):
        fourier_mod.register_pytree_node_class(Dummy)
    monkeypatch.setattr(coords_mod, "_register_pytree_node_class", different_registration_error, raising=False)
    with pytest.raises(ValueError, match="different registration error"):
        coords_mod.register_pytree_node_class(Dummy)


def test_project_to_modes_normalization_and_raw_inner_products():
    modes = ModeTable(m=np.asarray([0, 1]), n=np.asarray([0, 0]))
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    grid = AngleGrid(theta=theta, zeta=np.asarray([0.0]), nfp=1)
    basis = build_helical_basis(modes, grid, cache=False)
    field = np.asarray(eval_fourier(np.asarray([3.0, 2.0]), np.asarray([0.0, -1.0]), basis))

    raw_cos, raw_sin = project_to_modes(field, basis, normalize=False)
    coeff_cos, coeff_sin = project_to_modes(field, basis, normalize=True)

    assert float(np.asarray(raw_cos)[0]) == np.sum(field)
    np.testing.assert_allclose(np.asarray(coeff_cos), [3.0, 2.0], atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(coeff_sin), [0.0, -1.0], atol=1.0e-14)

    # The normalize path should skip the m=n=0 special case if metadata is absent.
    basis_no_meta = type("BasisNoMeta", (), {})()
    basis_no_meta.cos_phase = basis.cos_phase
    basis_no_meta.sin_phase = basis.sin_phase
    coeff_cos_no_meta, coeff_sin_no_meta = project_to_modes(field, basis_no_meta, normalize=True)
    np.testing.assert_allclose(np.asarray(coeff_cos_no_meta), [6.0, 2.0], atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(coeff_sin_no_meta), [0.0, -1.0], atol=1.0e-14)


def test_coords_and_geom_pytree_aliases_and_vector_helpers():
    base = np.arange(4.0).reshape(2, 2)
    coords = Coords(
        R=base,
        Z=base + 1,
        L=base + 2,
        R_theta=base + 3,
        Z_theta=base + 4,
        L_theta=base + 5,
        R_phi=base + 6,
        Z_phi=base + 7,
        L_phi=base + 8,
    )
    leaves, treedef = tree_util.tree_flatten(coords)
    rebuilt_coords = tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(rebuilt_coords.L_phi), base + 8)

    geom = Geom(
        R=base,
        Z=base + 1,
        L=base + 2,
        Rs=base + 3,
        Zs=base + 4,
        Ls=base + 5,
        Rt=base + 6,
        Zt=base + 7,
        Lt=base + 8,
        Rp=base + 9,
        Zp=base + 10,
        Lp=base + 11,
        sqrtg=base + 12,
        g_ss=base + 13,
        g_st=base + 14,
        g_sp=base + 15,
        g_tt=base + 16,
        g_tp=base + 17,
        g_pp=base + 18,
    )
    leaves, treedef = tree_util.tree_flatten(geom)
    rebuilt_geom = tree_util.tree_unflatten(treedef, leaves)

    np.testing.assert_allclose(np.asarray(rebuilt_geom.R_s), base + 3)
    np.testing.assert_allclose(np.asarray(rebuilt_geom.Z_theta), base + 7)
    np.testing.assert_allclose(np.asarray(rebuilt_geom.L_phi), base + 11)

    a = np.asarray([[1.0, 0.0, 0.0]])
    b = np.asarray([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(np.asarray(_cross(a, b)), [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(np.asarray(_dot(a, b)), [0.0])


def test_eval_geom_metrics_are_finite_for_torus_boundary():
    """Metric/Jacobian evaluation should run and produce finite geometry."""
    pytest.importorskip("jax")

    from vmec_jax.boundary import BoundaryCoeffs
    from vmec_jax.config import VMECConfig
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.namelist import InData
    from vmec_jax.static import build_static

    cfg = VMECConfig(ns=7, mpol=3, ntor=0, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=12, nzeta=3)
    static = build_static(cfg)
    K = int(static.modes.K)
    Rcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)
    k00 = int(np.where((np.asarray(static.modes.m) == 0) & (np.asarray(static.modes.n) == 0))[0][0])
    k10 = int(np.where((np.asarray(static.modes.m) == 1) & (np.asarray(static.modes.n) == 0))[0][0])
    Rcos[k00] = 3.0
    Rcos[k10] = 1.0
    Zsin[k10] = 0.6
    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=np.zeros_like(Rcos), Z_cos=np.zeros_like(Rcos), Z_sin=Zsin)
    indata = InData(scalars={"RAXIS_CC": [3.0], "ZAXIS_CS": [0.0]}, indexed={})
    st0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=False)

    sqrtg = np.asarray(eval_geom(st0, static).sqrtg)
    assert np.all(np.isfinite(sqrtg))
    assert np.max(np.abs(sqrtg)) > 1e-6

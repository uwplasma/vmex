from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from vmec_jax.namelist import read_indata
from vmec_jax.config import VMECConfig
from vmec_jax.coords import eval_coords
from vmec_jax.modes import ModeTable, vmec_mode_table
from vmec_jax.grids import make_angle_grid
from vmec_jax.boundary import (
    BoundaryCoeffs,
    _boundary_cache_key,
    _get_indexed,
    boundary_aspect_ratio,
    boundary_aspect_ratio_from_static,
    boundary_apply_vmec_m1_constraint,
    boundary_from_indata,
    boundary_from_input_convention,
    boundary_input_from_indata,
    boundary_undo_vmec_m1_constraint,
)
from vmec_jax.namelist import InData
from vmec_jax.fourier import build_helical_basis, eval_fourier
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.static import build_static


def test_circular_boundary(tmp_path: Path):
    # R = R0 + a cos(theta), Z = a sin(theta)
    R0 = 6.0
    a = 2.0
    txt = f"""
&INDATA
  NFP = 1
  MPOL = 6
  NTOR = 0
  NTHETA = 32
  NZETA = 1
  RBC(0,0) = {R0}
  RBC(0,1) = {a}
  ZBS(0,1) = {a}
/
"""
    p = tmp_path / "input.test"
    p.write_text(txt)

    indata = read_indata(p)
    modes = vmec_mode_table(mpol=6, ntor=0)
    grid = make_angle_grid(ntheta=32, nzeta=1, nfp=1)
    bdy = boundary_from_indata(indata, modes)
    basis = build_helical_basis(modes, grid)

    R = np.asarray(eval_fourier(bdy.R_cos, bdy.R_sin, basis))
    Z = np.asarray(eval_fourier(bdy.Z_cos, bdy.Z_sin, basis))

    theta = grid.theta
    assert np.allclose(R[:, 0], R0 + a * np.cos(theta), atol=1e-12)
    assert np.allclose(Z[:, 0], a * np.sin(theta), atol=1e-12)

    aspect = float(boundary_aspect_ratio(bdy, basis))
    aspect_from_static = float(
        boundary_aspect_ratio_from_static(bdy, SimpleNamespace(modes=modes, grid=grid))
    )
    np.testing.assert_allclose(aspect, R0 / a, rtol=1.0e-2, atol=0.0)
    np.testing.assert_allclose(aspect_from_static, aspect, rtol=0.0, atol=1.0e-12)


def test_boundary_matches_initial_guess_state_surface():
    cfg = VMECConfig(ns=7, mpol=3, ntor=0, nfp=1, lasym=False, lconm1=True, lthreed=True, ntheta=12, nzeta=3)
    static = build_static(cfg)
    K = int(static.modes.K)

    Rcos = np.zeros((K,), dtype=float)
    Rsin = np.zeros((K,), dtype=float)
    Zcos = np.zeros((K,), dtype=float)
    Zsin = np.zeros((K,), dtype=float)

    k00 = int(np.where((np.asarray(static.modes.m) == 0) & (np.asarray(static.modes.n) == 0))[0][0])
    k10 = int(np.where((np.asarray(static.modes.m) == 1) & (np.asarray(static.modes.n) == 0))[0][0])
    Rcos[k00] = 3.0
    Rcos[k10] = 1.0
    Zsin[k10] = 0.6

    bdy = BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)
    indata = InData(scalars={"RAXIS_CC": [3.0], "ZAXIS_CS": [0.0]}, indexed={})
    state0 = initial_guess_from_boundary(static, bdy, indata, vmec_project=False)

    coords = eval_coords(state0, static.basis)
    R = np.asarray(coords.R)
    Z = np.asarray(coords.Z)

    Rb = np.asarray(eval_fourier(bdy.R_cos, bdy.R_sin, static.basis))
    Zb = np.asarray(eval_fourier(bdy.Z_cos, bdy.Z_sin, static.basis))

    # The outer radial surface is initialized directly from the VMEC boundary coefficients.
    assert np.max(np.abs(R[-1] - Rb)) < 1e-12
    assert np.max(np.abs(Z[-1] - Zb)) < 1e-12


def test_boundary_input_conversion_matches_solver_boundary(tmp_path: Path):
    txt = """
&INDATA
  NFP = 4
  LASYM = F
  MPOL = 3
  NTOR = 1
  NTHETA = 16
  NZETA = 14
  RBC(0,0) = 1.0
  RBC(1,0) = 0.13
  RBC(-1,1) = -0.07
  RBC(0,1) = 0.16
  RBC(1,1) = -0.01
  ZBS(1,0) = 0.12
  ZBS(-1,1) = 0.10
  ZBS(0,1) = 0.17
  ZBS(1,1) = -0.02
/
"""
    p = tmp_path / "input.test3d"
    p.write_text(txt)

    indata = read_indata(p)
    modes = vmec_mode_table(mpol=3, ntor=1)
    raw = boundary_input_from_indata(indata, modes)
    converted = boundary_from_input_convention(raw, modes, lasym=False, apply_m1_constraint=False)
    direct = boundary_from_indata(indata, modes, apply_m1_constraint=False)

    np.testing.assert_allclose(np.asarray(converted.R_cos), np.asarray(direct.R_cos))
    np.testing.assert_allclose(np.asarray(converted.R_sin), np.asarray(direct.R_sin))
    np.testing.assert_allclose(np.asarray(converted.Z_cos), np.asarray(direct.Z_cos))
    np.testing.assert_allclose(np.asarray(converted.Z_sin), np.asarray(direct.Z_sin))


def test_boundary_input_conversion_is_jittable(tmp_path: Path):
    txt = """
&INDATA
  NFP = 4
  LASYM = F
  MPOL = 3
  NTOR = 1
  NTHETA = 16
  NZETA = 14
  RBC(0,0) = 1.0
  RBC(1,0) = 0.13
  RBC(-1,1) = -0.07
  RBC(0,1) = 0.16
  RBC(1,1) = -0.01
  ZBS(1,0) = 0.12
  ZBS(-1,1) = 0.10
  ZBS(0,1) = 0.17
  ZBS(1,1) = -0.02
/
"""
    p = tmp_path / "input.test3d"
    p.write_text(txt)

    indata = read_indata(p)
    modes = vmec_mode_table(mpol=3, ntor=1)
    raw = boundary_input_from_indata(indata, modes)
    raw_jax = type(raw)(
        R_cos=jax.numpy.asarray(raw.R_cos),
        R_sin=jax.numpy.asarray(raw.R_sin),
        Z_cos=jax.numpy.asarray(raw.Z_cos),
        Z_sin=jax.numpy.asarray(raw.Z_sin),
    )
    fn = jax.jit(
        lambda b: boundary_from_input_convention(
            b,
            modes,
            lasym=False,
            apply_m1_constraint=False,
        ).R_cos
    )
    out = np.asarray(fn(raw_jax))
    expected = np.asarray(boundary_from_indata(indata, modes, apply_m1_constraint=False).R_cos)
    np.testing.assert_allclose(out, expected)


def test_boundary_indexed_cache_key_and_negative_mode_branches(tmp_path: Path):
    indata = InData(
        scalars={"LASYM": False, "LCONM1": True},
        indexed={"RBC": {(0, 0): 1.0, (1, 0): "skip", (2, 0): True, (-1, 1): 0.2}},
    )
    assert _get_indexed(indata, "RBC") == {(0, 0): 1.0, (-1, 1): 0.2}

    modes = ModeTable(m=np.asarray([-1, 0, 1]), n=np.asarray([1, 0, 1]))
    key = _boundary_cache_key(
        indata,
        modes,
        apply_m1_constraint=True,
        rbc={(0, 0): 1.0},
        rbs={},
        zbc={},
        zbs={(1, 1): 0.3},
    )
    assert key[0] is None
    assert key[1][0] == ((0, 0, 1.0),)

    missing = tmp_path / "missing.input"
    indata.source_path = str(missing)
    key_missing_path = _boundary_cache_key(indata, modes, apply_m1_constraint=False, rbc={}, rbs={}, zbc={}, zbs={})
    assert key_missing_path[0] is None

    bdy = BoundaryCoeffs(
        R_cos=np.asarray([9.0, 1.0, 2.0]),
        R_sin=np.asarray([9.0, 0.0, 3.0]),
        Z_cos=np.asarray([9.0, 0.0, 4.0]),
        Z_sin=np.asarray([9.0, 0.0, 5.0]),
    )
    converted = boundary_from_input_convention(bdy, modes, lasym=True, apply_m1_constraint=False)
    np.testing.assert_allclose(np.asarray(converted.R_cos)[0], 0.0)
    np.testing.assert_allclose(np.asarray(converted.Z_sin)[0], 0.0)


def test_boundary_m1_constraint_roundtrip_lasym_three_dimensional():
    modes = vmec_mode_table(mpol=3, ntor=1)
    size = len(modes.m)
    raw = BoundaryCoeffs(
        R_cos=np.linspace(1.0, 2.0, size),
        R_sin=np.concatenate([[0.0], np.linspace(0.1, 0.2, size - 1)]),
        Z_cos=np.concatenate([[0.0], np.linspace(-0.2, 0.3, size - 1)]),
        Z_sin=np.concatenate([[0.0], np.linspace(0.4, 0.9, size - 1)]),
    )
    constrained = boundary_apply_vmec_m1_constraint(raw, modes, lthreed=True, lasym=True)
    restored = boundary_undo_vmec_m1_constraint(constrained, modes, lthreed=True, lasym=True)
    np.testing.assert_allclose(np.asarray(restored.R_cos), raw.R_cos)
    np.testing.assert_allclose(np.asarray(restored.R_sin), raw.R_sin)
    np.testing.assert_allclose(np.asarray(restored.Z_cos), raw.Z_cos)
    np.testing.assert_allclose(np.asarray(restored.Z_sin), raw.Z_sin)


def test_boundary_m1_constraint_undo_jax_path_matches_numpy():
    modes = vmec_mode_table(mpol=3, ntor=1)
    size = len(modes.m)
    raw = BoundaryCoeffs(
        R_cos=np.linspace(1.0, 2.0, size),
        R_sin=np.concatenate([[0.0], np.linspace(0.1, 0.2, size - 1)]),
        Z_cos=np.concatenate([[0.0], np.linspace(-0.2, 0.3, size - 1)]),
        Z_sin=np.concatenate([[0.0], np.linspace(0.4, 0.9, size - 1)]),
    )
    constrained = boundary_apply_vmec_m1_constraint(raw, modes, lthreed=True, lasym=True)
    expected = boundary_undo_vmec_m1_constraint(constrained, modes, lthreed=True, lasym=True)

    @jax.jit
    def undo_jax(r_cos, r_sin, z_cos, z_sin):
        out = boundary_undo_vmec_m1_constraint(
            BoundaryCoeffs(R_cos=r_cos, R_sin=r_sin, Z_cos=z_cos, Z_sin=z_sin),
            modes,
            lthreed=True,
            lasym=True,
        )
        return out.R_cos, out.R_sin, out.Z_cos, out.Z_sin

    actual = undo_jax(
        jax.numpy.asarray(constrained.R_cos),
        jax.numpy.asarray(constrained.R_sin),
        jax.numpy.asarray(constrained.Z_cos),
        jax.numpy.asarray(constrained.Z_sin),
    )

    np.testing.assert_allclose(np.asarray(actual[0]), np.asarray(expected.R_cos))
    np.testing.assert_allclose(np.asarray(actual[1]), np.asarray(expected.R_sin))
    np.testing.assert_allclose(np.asarray(actual[2]), np.asarray(expected.Z_cos))
    np.testing.assert_allclose(np.asarray(actual[3]), np.asarray(expected.Z_sin))

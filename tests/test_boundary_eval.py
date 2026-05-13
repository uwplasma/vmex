from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from vmec_jax.namelist import read_indata
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

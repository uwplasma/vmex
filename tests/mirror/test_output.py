"""Mirror-native output and CLI plotting tests."""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import matplotlib.image as mpimg
import numpy as np
import pytest

pytest.importorskip("netCDF4")

from vmec_jax.core import cli
from vmec_jax.mirror import (
    MoutData,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    mirror_energy,
    mout_from_result,
    read_mout,
    write_mout,
)


def _sample_mout() -> MoutData:
    ns, ntheta, nxi = 3, 5, 7
    s = np.linspace(0.0, 1.0, ns)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    z = np.linspace(-1.0, 1.0, nxi)
    tt, zz = np.meshgrid(theta, z, indexing="ij")
    boundary = 0.25 * (1.0 + 0.15 * zz**2 + 0.04 * np.cos(2.0 * tt))
    radius_scale = np.broadcast_to(boundary, (ns, ntheta, nxi)).copy()
    mod_b = np.broadcast_to(0.8 + 0.35 * zz**2, (ns, ntheta, nxi)).copy()
    b_theta = 0.04
    b_xyz = np.empty((ns, ntheta, nxi, 3))
    b_xyz[..., 0] = -b_theta * np.sin(theta)[None, :, None]
    b_xyz[..., 1] = b_theta * np.cos(theta)[None, :, None]
    b_xyz[..., 2] = 1.0
    pressure = 2.0e4 * (1.0 - s[:, None, None]) * np.ones((1, ntheta, nxi))
    angle = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    coils = np.stack([
        np.column_stack([0.55 * np.cos(angle), 0.55 * np.sin(angle), np.full_like(angle, end)])
        for end in (-1.15, 1.15)
    ])
    return MoutData(
        s=s,
        theta=theta,
        xi=z,
        z=z,
        boundary_radius=boundary,
        radius_scale=radius_scale,
        lambda_stream=np.zeros_like(radius_scale),
        mod_b=mod_b,
        b_xyz=b_xyz,
        p_perpendicular=pressure,
        p_parallel=pressure,
        history=np.column_stack([np.arange(6), np.geomspace(1.0e-3, 1.0e-12, 6)]),
        coil_xyz=coils,
        ftol=1.0e-12,
        iterations=5,
        converged=True,
        mass_scale=1.0,
        variational_max=8.0e-13,
        normal_stress_rms=2.0e-10,
        b_normal_rms=3.0e-11,
        closure="isotropic",
    )


def test_mout_roundtrip(tmp_path) -> None:
    path = write_mout(tmp_path / "mout_sample.nc", _sample_mout())
    loaded = read_mout(path)
    assert loaded.schema == "vmec_jax.mirror.mout/1"
    assert loaded.converged
    assert loaded.closure == "isotropic"
    np.testing.assert_allclose(loaded.boundary_radius, _sample_mout().boundary_radius)
    np.testing.assert_allclose(loaded.b_xyz, _sample_mout().b_xyz)


def test_mout_from_solved_result_contract() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.25, grid)
    state = MirrorState.from_boundary(boundary, grid)
    shape = grid.shape
    result = SimpleNamespace(
        boundary=boundary,
        plasma_state=state,
        plasma_b_squared=np.ones(shape),
        perpendicular_pressure=np.zeros(shape),
        history=np.asarray([[0.0, 1.0e-13]]),
        iterations=1,
        converged=True,
        mass_scale=1.0,
        variational_max=1.0e-13,
        interface=SimpleNamespace(normal_stress_rms=2.0e-13, vacuum_b_normal_rms=3.0e-13),
        message="converged",
    )
    data = mout_from_result(result, grid, config, axial_flux_derivative=0.02)
    assert data.b_xyz.shape == (*shape, 3)
    assert np.all(np.isfinite(data.b_xyz))
    assert np.all(np.isnan(data.p_parallel))


def test_mout_accepts_fixed_boundary_result() -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=3, nxi=5))
    grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.25, grid)
    state = MirrorState.from_boundary(boundary, grid)
    energy = mirror_energy(
        state,
        grid,
        axial_flux_derivative=0.02,
        mass_profile=np.asarray([2.0e-4, 1.0e-4, 0.0]),
    )
    result = SimpleNamespace(
        state=state,
        energy=energy,
        variational=SimpleNamespace(maximum=1.0e-13),
        history=np.asarray([[0.0, energy.total, 0.0, 0.0, 1.0e-13, 2.0e-3]]),
        iterations=0,
        converged=True,
        message="converged",
    )
    data = mout_from_result(
        result,
        grid,
        config,
        boundary=boundary,
        axial_flux_derivative=0.02,
        closure="isotropic",
    )

    assert data.boundary_radius.shape == (1, grid.nxi)
    np.testing.assert_allclose(data.p_perpendicular, data.p_parallel)
    assert np.all(np.isfinite(data.mod_b))
    np.testing.assert_allclose(data.history, [[0.0, 1.0e-13]])


def test_cli_plots_mout_without_toroidal_dispatch(tmp_path) -> None:
    path = write_mout(tmp_path / "mout_sample.nc", _sample_mout())
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = cli.main(["--plot", str(path), "--outdir", str(tmp_path), "--quiet"])
    assert code == 0
    expected = {
        "sample_summary.png",
        "sample_cross_sections.png",
        "sample_modB.png",
        "sample_3d.png",
    }
    assert expected == {item.name for item in tmp_path.glob("sample_*.png")}
    for name in expected:
        pixels = mpimg.imread(tmp_path / name)
        assert pixels.shape[0] > 200 and pixels.shape[1] > 300
        assert float(np.std(pixels)) > 0.03

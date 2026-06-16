from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    is_mirror_output,
    load_mirror_output,
    mirror_axisym_slice_to_csv,
    mirror_output_to_npz,
    run_mirror_fixed_boundary,
    write_mirror_output,
)
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.io.schema import MOUT_ALGORITHM, MOUT_SCHEMA_VERSION

pytestmark = pytest.mark.mirror


def _small_result():
    config = MirrorConfig(MirrorResolution(ns=7, ntheta=1, nxi=11, mpol=0), z_min=-1.0, z_max=1.0)
    boundary = MirrorBoundary.polynomial_radius(r0=0.28, a2=0.08)
    return run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.012),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.polynomial([0.2, -0.1], gamma=2.0),
        options=MirrorSolveOptions(
            maxiter=2,
            step_size=1.0e-4,
            tolerance=1.0e-12,
            mu0=1.0,
            pressure_continuation=(0.0, 1.0),
        ),
    )


def _small_3d_result():
    config = MirrorConfig(MirrorResolution(ns=5, ntheta=9, nxi=9, mpol=3), z_min=-1.0, z_max=1.0)
    boundary = MirrorBoundary.cosine_modulated_radius(r0=0.28, a2=0.06, epsilon=0.04, theta_mode=2)
    return run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=2, tolerance=1.0e-10, mu0=1.0),
    )


def test_mirror_output_roundtrip_preserves_schema_and_arrays(tmp_path):
    result = _small_result()
    path = tmp_path / "mout_roundtrip.nc"

    written = write_mirror_output(path, result)
    loaded = load_mirror_output(written)

    assert written == path
    assert is_mirror_output(path)
    assert loaded.attributes["geometry_type"] == "mirror"
    assert loaded.attributes["mirror_schema_version"] == MOUT_SCHEMA_VERSION
    assert loaded.attributes["algorithm"] == MOUT_ALGORITHM
    assert loaded.geometry.r.shape == (result.grid.ns, result.grid.ntheta, result.grid.nxi)
    assert loaded.field.bmag.shape == loaded.geometry.r.shape
    assert loaded.geometry.boundary_r.shape == (result.grid.ntheta, result.grid.nxi)
    assert np.allclose(loaded.s, result.grid.s_full)
    assert np.allclose(loaded.xi, result.grid.xi)
    assert np.allclose(loaded.geometry.r[:, 0, :], result.state.a * result.grid.rho_full[:, None])
    assert np.allclose(loaded.geometry.boundary_r[0], result.boundary.radius_on_grid(result.grid))
    assert np.isclose(loaded.diagnostics.energy_total, loaded.history.energy_total[-1])
    assert np.isclose(loaded.diagnostics.residual_norm, loaded.history.residual_norm[-1])
    assert np.isclose(loaded.diagnostics.fsq, loaded.history.fsq[-1])
    assert np.isclose(loaded.diagnostics.normalized_force, loaded.history.normalized_force[-1])
    assert loaded.diagnostics.active_force_dof == loaded.history.active_force_dof[-1]
    assert loaded.diagnostics.fsq >= 0.0
    assert loaded.diagnostics.normalized_force >= 0.0
    assert loaded.diagnostics.active_force_dof > 0
    assert np.all(loaded.history.accepted)

    with pytest.raises(FileExistsError):
        write_mirror_output(path, result)


def test_mirror_output_roundtrip_preserves_nonaxisymmetric_arrays(tmp_path):
    pytest.importorskip("scipy.optimize")
    result = _small_3d_result()
    path = tmp_path / "mout_nonaxisymmetric.nc"

    written = write_mirror_output(path, result)
    loaded = load_mirror_output(written)

    assert loaded.geometry.r.shape == (result.grid.ns, result.grid.ntheta, result.grid.nxi)
    assert loaded.field.bmag.shape == loaded.geometry.r.shape
    assert loaded.geometry.boundary_r.shape == (result.grid.ntheta, result.grid.nxi)
    assert np.allclose(loaded.geometry.boundary_r, result.boundary.radius_on_grid_3d(result.grid))
    assert np.max(np.ptp(loaded.geometry.boundary_r, axis=0)) > 0.0
    assert np.isclose(loaded.diagnostics.energy_total, loaded.history.energy_total[-1])
    assert np.isclose(loaded.diagnostics.fsq, loaded.history.fsq[-1])
    assert loaded.diagnostics.normalized_force >= 0.0
    assert loaded.diagnostics.min_sqrtg > 0.0


def test_mirror_output_exports_npz_and_axisym_csv(tmp_path):
    result = _small_result()
    mout = write_mirror_output(tmp_path / "mout_exports.nc", result)
    output = load_mirror_output(mout)

    npz_path = mirror_output_to_npz(output, tmp_path / "mirror_arrays.npz")
    csv_path = mirror_axisym_slice_to_csv(output, tmp_path / "mirror_slice.csv")

    with np.load(npz_path) as data:
        assert set(data.files) >= {"s", "xi", "r", "sqrtg", "bmag", "pressure"}
        assert data["bmag"].shape == output.field.bmag.shape
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    assert table.shape[0] == output.ns * output.nxi
    assert np.all(table[:, 3] >= 0.0)


def test_mirror_output_detector_uses_schema_for_nonstandard_name(tmp_path):
    result = _small_result()
    path = write_mirror_output(tmp_path / "case.nc", result)
    assert is_mirror_output(path)
    assert not is_mirror_output(tmp_path / "not_netcdf.txt")

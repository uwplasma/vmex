from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import CoilFieldParams, sample_coil_field_cylindrical
from vmec_jax.free_boundary import ExternalBoundarySample, sample_free_boundary_external_field
from vmec_jax.namelist import read_indata, write_indata


ROOT = Path(__file__).resolve().parents[1]


def _circle_coil_params() -> CoilFieldParams:
    from vmec_jax._compat import jnp

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(1.4)
    dofs = dofs.at[0, 1, 1].set(1.4)
    return CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.0]),
        n_segments=96,
    )


def _simple_boundary(ntheta: int = 6, nzeta: int = 4):
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0.0, 0.5 * np.pi, nzeta, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    major = 0.5
    minor = 0.15
    R = major + minor * np.cos(tt)
    Z = minor * np.sin(tt)
    Ru = -minor * np.sin(tt)
    Zu = minor * np.cos(tt)
    Rv = np.zeros_like(R)
    Zv = np.zeros_like(Z)
    return R, Z, Ru, Zu, Rv, Zv, pp


def test_sample_free_boundary_external_field_from_direct_coils_matches_provider_components():
    enable_x64(True)
    params = _circle_coil_params()
    R, Z, Ru, Zu, Rv, Zv, phi = _simple_boundary()

    sample = sample_free_boundary_external_field(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        provider_kind="direct_coils",
        provider_params=params,
        label="direct_coils_test",
    )
    expected = sample_coil_field_cylindrical(params, R, Z, phi)

    assert isinstance(sample, ExternalBoundarySample)
    assert sample.mgrid_path == "direct_coils_test"
    np.testing.assert_allclose(sample.br, expected[0], rtol=1.0e-13, atol=1.0e-17)
    np.testing.assert_allclose(sample.bp, expected[1], rtol=1.0e-13, atol=1.0e-17)
    np.testing.assert_allclose(sample.bz, expected[2], rtol=1.0e-13, atol=1.0e-17)
    np.testing.assert_allclose(sample.br_axis, np.zeros_like(R), atol=0.0)
    assert sample.vac_ext.bu.shape == R.shape
    assert sample.vac_ext.bv.shape == R.shape
    assert sample.vac_ext.bnormal.shape == R.shape
    assert np.all(np.isfinite(sample.vac_ext.bsqvac))


def test_sample_free_boundary_external_field_adds_axis_field_separately():
    enable_x64(True)
    params = _circle_coil_params()
    R, Z, Ru, Zu, Rv, Zv, phi = _simple_boundary(ntheta=5, nzeta=3)
    axis = (0.1 * np.ones_like(R), -0.2 * np.ones_like(R), 0.3 * np.ones_like(R))

    sample = sample_free_boundary_external_field(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        provider_kind="direct_coils",
        provider_params=params,
        axis_field=axis,
        axis_r=np.linspace(0.4, 0.5, R.shape[-1]),
        axis_z=np.linspace(-0.1, 0.1, R.shape[-1]),
    )
    external = sample_coil_field_cylindrical(params, R, Z, phi)

    np.testing.assert_allclose(sample.br, np.asarray(external[0]) + axis[0], rtol=1.0e-13, atol=1.0e-17)
    np.testing.assert_allclose(sample.bp, np.asarray(external[1]) + axis[1], rtol=1.0e-13, atol=1.0e-17)
    np.testing.assert_allclose(sample.bz, np.asarray(external[2]) + axis[2], rtol=1.0e-13, atol=1.0e-17)
    np.testing.assert_allclose(sample.br_mgrid, external[0], rtol=1.0e-13, atol=1.0e-17)
    np.testing.assert_allclose(sample.br_axis, axis[0], rtol=0.0, atol=0.0)
    assert sample.axis_r.shape == (R.shape[-1],)
    assert sample.axis_z.shape == (R.shape[-1],)


def test_run_free_boundary_accepts_direct_coil_provider_without_mgrid_file(tmp_path):
    enable_x64(True)
    from vmec_jax.driver import run_free_boundary

    indata = deepcopy(read_indata(ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"))
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": "DIRECT_COILS",
            "EXTCUR": [1.0],
            "NS_ARRAY": [12],
            "NITER_ARRAY": [1],
            "FTOL_ARRAY": [1.0e-8],
            "NITER": 1,
            "FTOL": 1.0e-8,
            "MPOL": 3,
            "NTOR": 2,
            "NZETA": 4,
            "NTHETA": 0,
            "NVACSKIP": 4,
            "PRES_SCALE": 0.0,
            "AM": [1.0, -1.0],
        }
    )
    input_path = tmp_path / "input.direct_coil_smoke"
    write_indata(input_path, indata)

    run = run_free_boundary(
        input_path,
        max_iter=1,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=_circle_coil_params(),
    )

    diag = run.result.diagnostics
    assert diag["free_boundary_external_field"]["provider_kind"] == "direct_coils"
    assert diag["free_boundary_external_field"]["reason"] == "direct_provider_runtime_path"
    assert np.isfinite(float(diag["final_fsqr"]))
    assert np.isfinite(float(diag["final_fsqz"]))
    assert np.isfinite(float(diag["final_fsql"]))

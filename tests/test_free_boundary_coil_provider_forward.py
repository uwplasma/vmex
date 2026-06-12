from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import (
    CoilFieldParams,
    MGridFieldParams,
    build_coil_field_geometry,
    sample_coil_field_cylindrical,
)
from vmec_jax.free_boundary import ExternalBoundarySample, sample_free_boundary_external_field
from vmec_jax.namelist import read_indata, write_indata


ROOT = Path(__file__).resolve().parents[1]
LPQA_UNIT_INPUT = ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"
LPQA_UNIT_FREE_BOUNDARY_PHIEDGE = -0.025


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


def _off_axis_coil_params() -> CoilFieldParams:
    from vmec_jax._compat import jnp

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    # Off-axis displacement makes the field toroidally varying, so this checks
    # that the mgrid and direct-coil provider paths use the same phi convention.
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


def _mgrid_from_direct_coil_nodes(coil_params: CoilFieldParams):
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
    return (
        MGridFieldParams(
            br=br[None, ...],
            bphi=bphi[None, ...],
            bz=bz[None, ...],
            extcur=jnp.asarray([1.0]),
            rmin=rmin,
            rmax=rmax,
            zmin=zmin,
            zmax=zmax,
            nfp=nfp,
        ),
        r_grid,
        z_grid,
        phi_grid,
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


def test_generated_mgrid_boundary_projection_matches_direct_coil_provider_at_nodes():
    enable_x64(True)
    from vmec_jax._compat import jnp

    coil_params = _off_axis_coil_params()
    mgrid_params, r_grid, z_grid, phi_grid = _mgrid_from_direct_coil_nodes(coil_params)
    r_idx = np.asarray([[0, 2, 5], [1, 3, 4]])
    z_idx = np.asarray([[0, 2, 4], [1, 3, 2]])
    phi_idx = np.asarray([[0, 1, 3], [6, 4, 2]])
    R = jnp.asarray(np.asarray(r_grid)[r_idx])
    Z = jnp.asarray(np.asarray(z_grid)[z_idx])
    phi = jnp.asarray(np.asarray(phi_grid)[phi_idx])
    Ru = jnp.asarray([[0.03, -0.02, 0.01], [0.04, -0.01, 0.02]])
    Zu = jnp.asarray([[0.11, 0.09, 0.07], [0.08, 0.06, 0.10]])
    Rv = jnp.asarray([[0.02, 0.01, -0.03], [0.01, -0.02, 0.04]])
    Zv = jnp.asarray([[0.04, -0.03, 0.02], [-0.01, 0.02, 0.03]])

    direct = sample_free_boundary_external_field(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        provider_kind="direct_coils",
        provider_params=coil_params,
        label="direct_node_projection",
    )
    generated_mgrid = sample_free_boundary_external_field(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        provider_kind="mgrid",
        provider_params=mgrid_params,
        label="generated_mgrid_node_projection",
    )

    for name in ("br", "bp", "bz", "br_mgrid", "bp_mgrid", "bz_mgrid"):
        np.testing.assert_allclose(
            np.asarray(getattr(generated_mgrid, name)),
            np.asarray(getattr(direct, name)),
            rtol=2.0e-11,
            atol=2.0e-14,
            err_msg=f"generated mgrid projection changed {name}",
        )
    for name in ("bnormal", "bnormal_unit", "bu", "bv", "bsqvac"):
        np.testing.assert_allclose(
            np.asarray(getattr(generated_mgrid.vac_ext, name)),
            np.asarray(getattr(direct.vac_ext, name)),
            rtol=2.0e-11,
            atol=2.0e-14,
            err_msg=f"generated mgrid projection changed vac_ext.{name}",
        )


def test_generated_mgrid_boundary_projection_tracks_direct_coil_provider_off_grid():
    enable_x64(True)
    from vmec_jax._compat import jnp

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    # Keep the coil well outside the sampled boundary. The generated mgrid path
    # is trilinear, so this gate checks interpolation convention and bounded
    # projection error instead of exact node equality.
    dofs = dofs.at[0, 0, 0].set(3.0)
    dofs = dofs.at[0, 0, 2].set(0.18)
    dofs = dofs.at[0, 1, 1].set(0.18)
    dofs = dofs.at[0, 2, 0].set(0.08)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.1e5]),
        n_segments=96,
        nfp=1,
        stellsym=False,
    )

    rmin, rmax = 0.72, 1.18
    zmin, zmax = -0.24, 0.24
    kp, jz, ir = 24, 13, 14
    r_grid = jnp.linspace(rmin, rmax, ir)
    z_grid = jnp.linspace(zmin, zmax, jz)
    phi_grid = jnp.arange(kp, dtype=float) * (2.0 * jnp.pi / kp)
    phi_mesh, z_mesh, r_mesh = jnp.meshgrid(phi_grid, z_grid, r_grid, indexing="ij")
    br, bphi, bz = sample_coil_field_cylindrical(coil_params, r_mesh, z_mesh, phi_mesh)
    mgrid_params = MGridFieldParams(
        br=br[None, ...],
        bphi=bphi[None, ...],
        bz=bz[None, ...],
        extcur=jnp.asarray([1.0]),
        rmin=rmin,
        rmax=rmax,
        zmin=zmin,
        zmax=zmax,
        nfp=1,
    )

    R = jnp.asarray([[0.76, 0.93, 1.11], [0.81, 1.02, 1.15]])
    Z = jnp.asarray([[-0.19, -0.07, 0.17], [-0.12, 0.04, 0.21]])
    phi = jnp.asarray([[0.17, 1.05, 2.41], [3.18, 4.02, 5.36]])
    Ru = jnp.asarray([[0.03, -0.02, 0.01], [0.04, -0.01, 0.02]])
    Zu = jnp.asarray([[0.11, 0.09, 0.07], [0.08, 0.06, 0.10]])
    Rv = jnp.asarray([[0.02, 0.01, -0.03], [0.01, -0.02, 0.04]])
    Zv = jnp.asarray([[0.04, -0.03, 0.02], [-0.01, 0.02, 0.03]])

    direct = sample_free_boundary_external_field(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        provider_kind="direct_coils",
        provider_params=coil_params,
        label="direct_off_grid_projection",
    )
    generated_mgrid = sample_free_boundary_external_field(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        provider_kind="mgrid",
        provider_params=mgrid_params,
        label="generated_mgrid_off_grid_projection",
    )

    def assert_max_rel(name, got, want, limit=5.0e-2):
        got_arr = np.asarray(got, dtype=float)
        want_arr = np.asarray(want, dtype=float)
        denom = max(float(np.max(np.abs(want_arr))), 1.0e-30)
        rel = float(np.max(np.abs(got_arr - want_arr))) / denom
        assert rel < limit, f"{name}: max_rel={rel:.3e} >= {limit:.3e}"

    for name in ("br", "bp", "bz", "br_mgrid", "bp_mgrid", "bz_mgrid"):
        assert_max_rel(name, getattr(generated_mgrid, name), getattr(direct, name))
    for name in ("bnormal", "bnormal_unit", "bu", "bv", "bsqvac"):
        assert_max_rel(f"vac_ext.{name}", getattr(generated_mgrid.vac_ext, name), getattr(direct.vac_ext, name))

    for sample in (direct, generated_mgrid):
        field_norm = 0.0
        for name in ("br", "bp", "bz"):
            arr = np.asarray(getattr(sample, name), dtype=float)
            assert np.isfinite(arr).all()
            field_norm = max(field_norm, float(np.max(np.abs(arr))))
        assert field_norm > 0.0
        assert np.isfinite(np.asarray(sample.vac_ext.bsqvac, dtype=float)).all()
        assert float(np.max(np.asarray(sample.vac_ext.bsqvac, dtype=float))) > 0.0
        assert np.isfinite(np.asarray(sample.vac_ext.det_guv, dtype=float)).all()
        assert float(np.min(np.asarray(sample.vac_ext.det_guv, dtype=float))) > 0.0


def test_direct_coil_provider_cached_geometry_and_alias_match_uncached_projection():
    enable_x64(True)
    from vmec_jax._compat import jnp

    params = _off_axis_coil_params()
    R, Z, Ru, Zu, Rv, Zv, phi = _simple_boundary(ntheta=5, nzeta=4)
    R = jnp.asarray(R + 0.25)
    Z = jnp.asarray(Z)
    Ru = jnp.asarray(Ru)
    Zu = jnp.asarray(Zu)
    Rv = jnp.asarray(Rv + 0.01)
    Zv = jnp.asarray(Zv)
    phi = jnp.asarray(phi + 0.13)

    uncached = sample_free_boundary_external_field(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        provider_kind="direct_coils",
        provider_params=params,
        label="uncached_direct",
    )
    cached = sample_free_boundary_external_field(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        provider_kind="direct_coils",
        provider_static={
            "coil_geometry": build_coil_field_geometry(params),
            "regularization_epsilon": params.regularization_epsilon,
            "chunk_size": params.chunk_size,
            "cache_scope": "test_host_forward_only",
            "jit_sampler": False,
        },
        provider_params=params,
        label="cached_direct",
    )
    alias = sample_free_boundary_external_field(
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=phi,
        provider_kind="coils",
        provider_params=params,
        label="alias_direct",
    )

    expected = sample_coil_field_cylindrical(params, R, Z, phi)
    assert cached.mgrid_path == "cached_direct"
    for name, want in zip(("br", "bp", "bz"), expected, strict=True):
        np.testing.assert_allclose(np.asarray(getattr(uncached, name)), np.asarray(want), rtol=2.0e-13, atol=2.0e-14)
    for got_sample in (cached, alias):
        for name in ("br", "bp", "bz", "br_mgrid", "bp_mgrid", "bz_mgrid"):
            np.testing.assert_allclose(
                np.asarray(getattr(got_sample, name)),
                np.asarray(getattr(uncached, name)),
                rtol=2.0e-13,
                atol=2.0e-14,
                err_msg=f"{got_sample.mgrid_path} changed direct-coil provider {name}",
            )
        for name in ("bnormal", "bnormal_unit", "bu", "bv", "bsqvac"):
            np.testing.assert_allclose(
                np.asarray(getattr(got_sample.vac_ext, name)),
                np.asarray(getattr(uncached.vac_ext, name)),
                rtol=2.0e-13,
                atol=2.0e-14,
                err_msg=f"{got_sample.mgrid_path} changed direct-coil projection vac_ext.{name}",
            )


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

    indata = deepcopy(read_indata(LPQA_UNIT_INPUT))
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
            "PHIEDGE": LPQA_UNIT_FREE_BOUNDARY_PHIEDGE,
            "MPOL": 3,
            "NTOR": 2,
            "NZETA": 4,
            "NTHETA": 0,
            "NVACSKIP": 4,
            "PRES_SCALE": 1.0,
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


def test_run_free_boundary_direct_coil_geometry_cache_matches_uncached_path(tmp_path, monkeypatch):
    enable_x64(True)
    from vmec_jax.driver import run_free_boundary

    indata = deepcopy(read_indata(LPQA_UNIT_INPUT))
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
            "PHIEDGE": LPQA_UNIT_FREE_BOUNDARY_PHIEDGE,
            "MPOL": 3,
            "NTOR": 2,
            "NZETA": 4,
            "NTHETA": 0,
            "NVACSKIP": 4,
            "PRES_SCALE": 1.0,
            "AM": [1.0, -1.0],
        }
    )
    input_path = tmp_path / "input.direct_coil_cache_parity"
    write_indata(input_path, indata)
    params = _circle_coil_params()

    monkeypatch.delenv("VMEC_JAX_FREEB_DISABLE_COIL_GEOMETRY_CACHE", raising=False)
    cached = run_free_boundary(
        input_path,
        max_iter=1,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )
    monkeypatch.setenv("VMEC_JAX_FREEB_DISABLE_COIL_GEOMETRY_CACHE", "1")
    uncached = run_free_boundary(
        input_path,
        max_iter=1,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )

    for name in ("Rcos", "Zsin", "Rsin", "Zcos", "Lcos", "Lsin"):
        np.testing.assert_allclose(
            np.asarray(getattr(cached.state, name)),
            np.asarray(getattr(uncached.state, name)),
            rtol=1.0e-13,
            atol=1.0e-13,
            err_msg=f"cached direct-coil provider changed {name}",
        )
    for key in ("final_fsqr", "final_fsqz", "final_fsql"):
        assert cached.result.diagnostics[key] == pytest.approx(uncached.result.diagnostics[key], rel=1.0e-13, abs=1.0e-13)


def test_run_free_boundary_host_setup_enforce_matches_default_path(tmp_path, monkeypatch):
    enable_x64(True)
    from vmec_jax.driver import run_free_boundary

    indata = deepcopy(read_indata(LPQA_UNIT_INPUT))
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
            "PHIEDGE": LPQA_UNIT_FREE_BOUNDARY_PHIEDGE,
            "MPOL": 3,
            "NTOR": 2,
            "NZETA": 4,
            "NTHETA": 0,
            "NVACSKIP": 4,
            "PRES_SCALE": 1.0,
            "AM": [1.0, -1.0],
        }
    )
    input_path = tmp_path / "input.direct_coil_host_setup"
    write_indata(input_path, indata)
    params = _circle_coil_params()

    monkeypatch.setenv("VMEC_JAX_HOST_SETUP_ENFORCE", "0")
    default = run_free_boundary(
        input_path,
        max_iter=1,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )
    monkeypatch.setenv("VMEC_JAX_HOST_SETUP_ENFORCE", "1")
    host_setup = run_free_boundary(
        input_path,
        max_iter=1,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )

    for name in ("Rcos", "Zsin", "Rsin", "Zcos", "Lcos", "Lsin"):
        np.testing.assert_allclose(
            np.asarray(getattr(host_setup.state, name)),
            np.asarray(getattr(default.state, name)),
            rtol=1.0e-13,
            atol=1.0e-13,
            err_msg=f"host setup enforcement changed {name}",
        )
    for key in ("final_fsqr", "final_fsqz", "final_fsql"):
        assert host_setup.result.diagnostics[key] == pytest.approx(
            default.result.diagnostics[key],
            rel=1.0e-13,
            abs=1.0e-13,
        )

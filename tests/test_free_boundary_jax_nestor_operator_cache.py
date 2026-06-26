from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import enable_x64, has_jax
import vmec_jax.free_boundary as freeb
from vmec_jax.free_boundary import ExternalBoundarySample, VacuumBoundaryFields


pytestmark = pytest.mark.skipif(not has_jax(), reason="JAX NESTOR operator cache tests require JAX")


def _sample(shape: tuple[int, ...] = (2, 3), *, with_second_derivs: bool = True) -> ExternalBoundarySample:
    arr = np.ones(shape)
    vac = VacuumBoundaryFields(
        bu=arr,
        bv=arr,
        bsupu=arr,
        bsupv=arr,
        bsqvac=arr,
        bnormal=arr,
        bnormal_unit=arr,
        g_uu=arr,
        g_uv=arr,
        g_vv=arr,
        det_guv=arr,
    )
    second = arr if with_second_derivs else None
    return ExternalBoundarySample(
        mgrid_path="synthetic",
        R=arr,
        Z=2.0 * arr,
        Ru=3.0 * arr,
        Zu=4.0 * arr,
        Rv=5.0 * arr,
        Zv=6.0 * arr,
        phi=np.zeros(shape),
        br=arr,
        bp=arr,
        bz=arr,
        br_mgrid=arr,
        bp_mgrid=arr,
        bz_mgrid=arr,
        br_axis=np.zeros(shape),
        bp_axis=np.zeros(shape),
        bz_axis=np.zeros(shape),
        axis_r=np.zeros(shape[-1] if len(shape) >= 1 else 1),
        axis_z=np.zeros(shape[-1] if len(shape) >= 1 else 1),
        vac_ext=vac,
        ruu=second,
        ruv=second,
        rvv=second,
        zuu=second,
        zuv=second,
        zvv=second,
    )


def test_jax_nestor_operator_guard_reports_shape_and_basis_contracts():
    enable_x64(True)
    sample = _sample()
    basis = {"nuv3": sample.R.size, "nuv_full": sample.R.size, "nu_full": sample.R.shape[0], "lasym": False}

    assert freeb._jax_nestor_operator_guard(sample=sample, basis=None) == (False, "missing_mode_basis")
    assert freeb._jax_nestor_operator_guard(sample=_sample((2, 3, 1)), basis=basis) == (False, "sample_R_not_2d")
    assert freeb._jax_nestor_operator_guard(sample=sample, basis={**basis, "nuv3": sample.R.size + 1}) == (
        False,
        "requires_active_vmec_grid_points",
    )
    assert freeb._jax_nestor_operator_guard(sample=sample, basis={**basis, "lasym": True, "nuv_full": sample.R.size + 1}) == (
        False,
        "requires_lasym_full_vmec_grid_points",
    )
    assert freeb._jax_nestor_operator_guard(sample=sample, basis={**basis, "nu_full": sample.R.shape[0] - 1}) == (
        False,
        "active_grid_exceeds_full_grid",
    )

    bad_Z = _sample()
    object.__setattr__(bad_Z, "Z", np.ones((1, 3)))
    assert freeb._jax_nestor_operator_guard(sample=bad_Z, basis=basis) == (False, "Z_shape_mismatch")

    missing_second = _sample(with_second_derivs=False)
    assert freeb._jax_nestor_operator_guard(sample=missing_second, basis=basis) == (False, "missing_ruu")
    bad_second = _sample()
    object.__setattr__(bad_second, "ruu", np.ones((1, 3)))
    assert freeb._jax_nestor_operator_guard(sample=bad_second, basis=basis) == (False, "ruu_shape_mismatch")

    assert freeb._jax_nestor_operator_guard(sample=sample, basis=basis) == (True, "enabled")


def test_jitted_jax_nestor_operator_cache_batches_dense_operator(monkeypatch):
    from vmec_jax._compat import jax, jnp
    import vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives as adjoint

    enable_x64(True)
    jax.config.update("jax_disable_jit", False)
    freeb._FREEB_JAX_NESTOR_OPERATOR_FN_CACHE.clear()

    def fake_dense_vmec_nestor_mode_solve_jax(
        *,
        R,
        Z,
        Ru,
        Zu,
        Rv,
        Zv,
        ruu,
        ruv,
        rvv,
        zuu,
        zuv,
        zvv,
        bexni,
        basis,
        tables,
        signgs,
        nvper,
        include_analytic,
        symmetric,
    ):
        del Ru, Zu, Rv, Zv, ruu, ruv, rvv, zuu, zuv, zvv, basis, tables
        scale = jnp.asarray(float(signgs + nvper + int(include_analytic) + int(symmetric)), dtype=R.dtype)
        return {"phi": R + Z + bexni + scale}

    monkeypatch.setattr(adjoint, "dense_vmec_nestor_mode_solve_jax", fake_dense_vmec_nestor_mode_solve_jax)

    try:
        args = tuple(jnp.ones((2, 2)) * (idx + 1.0) for idx in range(13))
        basis = {"xm": np.asarray([0, 1]), "xn": np.asarray([0, 1])}
        tables = {"kernel": np.asarray([[1.0, 0.5], [0.25, 0.125]])}
        compiled, reused = freeb._jitted_jax_nestor_operator(
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
            include_analytic=True,
            symmetric=False,
            example_args=args,
        )
        assert compiled is not None
        assert not reused
        out = compiled(*args)
        np.testing.assert_allclose(np.asarray(out["phi"]), np.asarray(args[0] + args[1] + args[-1] + 4.0))

        cached, reused = freeb._jitted_jax_nestor_operator(
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
            include_analytic=True,
            symmetric=False,
            example_args=args,
        )
        assert reused
        assert cached is compiled

        freeb._FREEB_JAX_NESTOR_OPERATOR_FN_CACHE.clear()
        for idx in range(32):
            freeb._FREEB_JAX_NESTOR_OPERATOR_FN_CACHE[(idx,)] = compiled
        fresh, reused = freeb._jitted_jax_nestor_operator(
            basis=basis,
            tables={"kernel": np.asarray([[2.0]])},
            signgs=-1,
            nvper=1,
            include_analytic=False,
            symmetric=True,
        )
        assert fresh is not None
        assert not reused
        assert len(freeb._FREEB_JAX_NESTOR_OPERATOR_FN_CACHE) == 1
    finally:
        freeb._FREEB_JAX_NESTOR_OPERATOR_FN_CACHE.clear()
        jax.config.update("jax_disable_jit", True)

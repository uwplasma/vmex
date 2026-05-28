from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import CoilFieldParams, sample_coil_field_cylindrical
from vmec_jax.free_boundary import (
    _build_vmec_mode_basis,
    _ensure_vmec_nonsingular_kernel_tables,
    _vmec_analytic_terms_from_geometry,
    _vmec_bvec_from_gsource,
    _vmec_mode_matrix_from_grpmn,
    _vmec_nonsingular_terms_from_bexni,
    _vmec_source_from_gsource,
    ExternalBoundarySample,
    VacuumBoundaryFields,
    vacuum_boundary_fields_from_cylindrical,
)
from vmec_jax.free_boundary_adjoint import (
    dense_fixed_point_solve_jax,
    dense_mode_vacuum_solve_jax,
    dense_nonlinear_solve_jax,
    dense_vmec_nestor_mode_solve_jax,
    dense_vacuum_residual,
    dense_vacuum_solve_jax,
    direct_coil_projected_mode_fixed_point_objective_jax,
    mode_matrix_from_grpmn_jax,
    mode_rhs_from_gsource_jax,
    vacuum_boundary_fields_from_cylindrical_jax,
    vmec_analytic_terms_from_geometry_jax,
    vmec_nonsingular_terms_from_bexni_jax,
    vmec_source_from_gsource_jax,
)


def _well_conditioned_matrix():
    from vmec_jax._compat import jnp

    A = jnp.asarray(
        [
            [3.0, 0.2, -0.1],
            [0.4, 2.5, 0.3],
            [-0.2, 0.1, 2.2],
        ]
    )
    b = jnp.asarray([1.0, -0.4, 0.7])
    return A, b


def test_dense_vacuum_solve_matches_jnp_linalg_solve():
    from vmec_jax._compat import jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()

    actual = dense_vacuum_solve_jax(A, b)
    expected = jnp.linalg.solve(A, b)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(dense_vacuum_residual(A, actual, b), np.zeros_like(np.asarray(b)), atol=1.0e-14)


def test_dense_vacuum_vjp_wrt_b_matches_transpose_solve():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()
    cotangent = jnp.asarray([0.3, -0.2, 0.5])

    def objective(rhs):
        x = dense_vacuum_solve_jax(A, rhs)
        return jnp.vdot(cotangent, x)

    grad_b = jax.grad(objective)(b)
    expected = jnp.linalg.solve(A.T, cotangent)

    np.testing.assert_allclose(grad_b, expected, rtol=1.0e-13, atol=1.0e-13)


def test_dense_vacuum_gradient_wrt_rhs_parameter_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()
    direction = jnp.asarray([0.2, -0.1, 0.4])
    cotangent = jnp.asarray([0.3, -0.2, 0.5])

    def objective(scale):
        x = dense_vacuum_solve_jax(A, b + scale * direction)
        return jnp.vdot(cotangent, x)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-9, atol=1.0e-11)


def test_dense_vacuum_gradient_wrt_matrix_parameter_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()
    dA = jnp.asarray(
        [
            [0.0, 0.2, 0.0],
            [-0.1, 0.0, 0.3],
            [0.0, 0.1, 0.0],
        ]
    )
    cotangent = jnp.asarray([0.3, -0.2, 0.5])

    def objective(scale):
        x = dense_vacuum_solve_jax(A + scale * dA, b)
        return jnp.vdot(cotangent, x)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-9, atol=1.0e-11)


def test_dense_vacuum_symmetric_mode_uses_symmetric_transpose_solve():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A = jnp.asarray([[3.0, 0.2], [0.2, 2.0]])
    b = jnp.asarray([0.7, -0.1])
    cotangent = jnp.asarray([0.4, 0.5])

    def objective(rhs):
        return jnp.vdot(cotangent, dense_vacuum_solve_jax(A, rhs, symmetric=True))

    grad_b = jax.grad(objective)(b)
    expected = jnp.linalg.solve(A, cotangent)

    np.testing.assert_allclose(grad_b, expected, rtol=1.0e-13, atol=1.0e-13)


def _nonlinear_residual(x, params):
    from vmec_jax._compat import jnp

    rhs = params["rhs"]
    return jnp.asarray(
        [
            x[0] + 0.2 * x[0] ** 3 + 0.1 * x[1] - rhs[0],
            x[1] + 0.1 * x[0] ** 2 + 0.3 * jnp.sin(x[1]) - rhs[1],
        ]
    )


def test_dense_nonlinear_solve_drives_residual_to_zero():
    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    params = {"rhs": jnp.asarray([0.4, -0.2], dtype=float)}
    root = dense_nonlinear_solve_jax(
        _nonlinear_residual,
        jnp.asarray([0.0, 0.0], dtype=float),
        params,
        max_iter=12,
    )

    residual = _nonlinear_residual(root, params)
    np.testing.assert_allclose(residual, np.zeros(2), rtol=0.0, atol=1.0e-12)


def test_dense_nonlinear_implicit_adjoint_matches_finite_difference_for_rhs():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    rhs0 = jnp.asarray([0.4, -0.2], dtype=float)
    direction = jnp.asarray([0.3, -0.5], dtype=float)
    weights = jnp.asarray([1.2, 0.7], dtype=float)

    def objective(scale):
        root = dense_nonlinear_solve_jax(
            _nonlinear_residual,
            jnp.asarray([0.0, 0.0], dtype=float),
            {"rhs": rhs0 + scale * direction},
            max_iter=12,
        )
        return 0.5 * jnp.vdot(weights * root, root)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=3.0e-8, atol=1.0e-10)


def test_dense_fixed_point_implicit_adjoint_matches_finite_difference_for_rhs():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    rhs0 = jnp.asarray([0.25, -0.18], dtype=float)
    direction = jnp.asarray([0.2, -0.4], dtype=float)
    weights = jnp.asarray([0.8, 1.1], dtype=float)

    def update(state, params):
        rhs = params["rhs"]
        return jnp.asarray(
            [
                0.12 + 0.18 * jnp.tanh(state[0]) + 0.07 * state[1] + rhs[0],
                -0.09 + 0.05 * state[0] ** 2 + 0.16 * jnp.sin(state[1]) + rhs[1],
            ]
        )

    def objective(scale):
        root = dense_fixed_point_solve_jax(
            update,
            jnp.asarray([0.0, 0.0], dtype=float),
            {"rhs": rhs0 + scale * direction},
            max_iter=12,
        )
        return 0.5 * jnp.vdot(weights * root, root)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=5.0e-8, atol=1.0e-10)


def _mode_basis_for_rhs_tests(*, lasym: bool = False):
    ntheta, nzeta = 4, 5
    wint = np.full((ntheta, nzeta), 1.0 / float(ntheta * nzeta))
    return _build_vmec_mode_basis(
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=2,
        mf=2,
        nf=1,
        lasym=lasym,
        wint=wint,
    )


def _mode_rhs_from_basis(gsource, basis):
    return mode_rhs_from_gsource_jax(
        gsource,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"],
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        onp=float(basis["onp"]),
        lasym=bool(basis["lasym"]),
        nuv3=int(basis["nuv3"]),
        nuv_full=int(basis["nuv_full"]),
        imirr=basis["imirr"],
        imirr_full=basis["imirr_full"],
    )


def _mode_matrix_from_basis(grpmn, basis):
    return mode_matrix_from_grpmn_jax(
        grpmn,
        sin_basis=basis["sinmni"],
        cos_basis=basis["cosmni"],
        xmpot=basis["xmpot"],
        n_raw=basis["n_raw"],
        lasym=bool(basis["lasym"]),
        mn0=int(basis["mn0"]),
    )


def _nonsingular_boundary_sample(*, radius_shift: float = 0.0, lasym: bool = True):
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    nzeta = 5
    ntheta = int(basis["nuv3"]) // nzeta
    theta = np.asarray(basis["theta"], dtype=float).reshape(ntheta, nzeta)
    zeta = np.asarray(basis["zeta"], dtype=float).reshape(ntheta, nzeta)
    R = 1.25 + 0.04 * radius_shift + 0.05 * np.cos(theta) + 0.02 * np.cos(theta - zeta)
    Z = 0.18 * np.sin(theta) + 0.03 * np.sin(theta + zeta)
    Ru = -0.05 * np.sin(theta) - 0.02 * np.sin(theta - zeta)
    Zu = 0.18 * np.cos(theta) + 0.03 * np.cos(theta + zeta)
    Rv = 0.02 * np.sin(theta - zeta)
    Zv = 0.03 * np.cos(theta + zeta)
    ruu = -0.05 * np.cos(theta) - 0.02 * np.cos(theta - zeta)
    ruv = 0.02 * np.cos(theta - zeta)
    rvv = -0.02 * np.cos(theta - zeta)
    zuu = -0.18 * np.sin(theta) - 0.03 * np.sin(theta + zeta)
    zuv = -0.03 * np.sin(theta + zeta)
    zvv = -0.03 * np.sin(theta + zeta)
    zeros = np.zeros_like(R)
    ones = np.ones_like(R)
    vac = VacuumBoundaryFields(
        bu=zeros,
        bv=zeros,
        bsupu=zeros,
        bsupv=zeros,
        bsqvac=zeros,
        bnormal=zeros,
        bnormal_unit=zeros,
        g_uu=ones,
        g_uv=zeros,
        g_vv=ones,
        det_guv=ones,
    )
    sample = ExternalBoundarySample(
        mgrid_path="synthetic",
        R=R,
        Z=Z,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
        phi=zeta / float(basis["nfp"]),
        br=zeros,
        bp=zeros,
        bz=zeros,
        br_mgrid=zeros,
        bp_mgrid=zeros,
        bz_mgrid=zeros,
        br_axis=zeros,
        bp_axis=zeros,
        bz_axis=zeros,
        axis_r=np.zeros((1,), dtype=float),
        axis_z=np.zeros((1,), dtype=float),
        vac_ext=vac,
        ruu=ruu,
        ruv=ruv,
        rvv=rvv,
        zuu=zuu,
        zuv=zuv,
        zvv=zvv,
    )
    return basis, sample


def _jax_nonsingular_terms(sample, basis, bexni):
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)
    return vmec_nonsingular_terms_from_bexni_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bexni,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
    )


def _jax_analytic_terms(sample, basis, bexni):
    return vmec_analytic_terms_from_geometry_jax(
        R=sample.R,
        Ru=sample.Ru,
        Rv=sample.Rv,
        Zu=sample.Zu,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bexni,
        basis=basis,
        signgs=1,
    )


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_source_and_mode_rhs_match_numpy_reference(lasym):
    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    gsource = np.linspace(-0.8, 1.3, int(basis["nuv_full"]), dtype=float)

    actual_source = vmec_source_from_gsource_jax(
        gsource,
        onp=float(basis["onp"]),
        lasym=bool(basis["lasym"]),
        nuv3=int(basis["nuv3"]),
        nuv_full=int(basis["nuv_full"]),
        imirr=basis["imirr"],
        imirr_full=basis["imirr_full"],
    )
    expected_source = np.asarray(_vmec_source_from_gsource(gsource=gsource, basis=basis))
    np.testing.assert_allclose(actual_source, expected_source, rtol=1.0e-13, atol=1.0e-13)

    actual_rhs = _mode_rhs_from_basis(gsource, basis)
    expected_rhs = _vmec_bvec_from_gsource(gsource=gsource, basis=basis)
    np.testing.assert_allclose(actual_rhs, expected_rhs, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_mode_rhs_gradient_wrt_gsource_matches_finite_difference(lasym):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    gsource = jnp.asarray(np.linspace(-0.8, 1.3, int(basis["nuv_full"]), dtype=float))
    direction = jnp.asarray(np.cos(np.arange(int(basis["nuv_full"]), dtype=float)))
    rhs0 = _mode_rhs_from_basis(gsource, basis)
    weights = jnp.asarray(np.linspace(0.3, 1.1, int(rhs0.shape[0]), dtype=float))

    def objective(scale):
        rhs = _mode_rhs_from_basis(gsource + scale * direction, basis)
        return jnp.vdot(weights, rhs)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)
    np.testing.assert_allclose(exact, fd, rtol=3.0e-9, atol=1.0e-11)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_mode_matrix_matches_numpy_reference(lasym):
    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    mnpd2 = int(basis["mnpd2"])
    nuv3 = int(basis["nuv3"])
    rows = np.arange(mnpd2, dtype=float)[:, None]
    cols = np.arange(nuv3, dtype=float)[None, :]
    grpmn = 0.04 * np.sin(0.3 + 0.2 * rows + 0.1 * cols) + 0.02 * np.cos(0.4 * rows - 0.3 * cols)

    actual = _mode_matrix_from_basis(grpmn, basis)
    expected = _vmec_mode_matrix_from_grpmn(grpmn=grpmn, basis=basis)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_mode_matrix_gradient_wrt_grpmn_matches_finite_difference(lasym):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    mnpd2 = int(basis["mnpd2"])
    nuv3 = int(basis["nuv3"])
    rows = np.arange(mnpd2, dtype=float)[:, None]
    cols = np.arange(nuv3, dtype=float)[None, :]
    grpmn = jnp.asarray(
        0.03 * np.sin(0.2 + 0.15 * rows + 0.1 * cols)
        + 0.01 * np.cos(0.25 * rows - 0.35 * cols),
        dtype=float,
    )
    direction = jnp.asarray(np.cos(0.4 * rows + 0.2 * cols), dtype=float)
    weights = jnp.asarray(np.sin(0.1 + np.arange(int(_mode_matrix_from_basis(grpmn, basis).size))), dtype=float)
    weights = jnp.reshape(weights, _mode_matrix_from_basis(grpmn, basis).shape)

    def objective(scale):
        matrix = _mode_matrix_from_basis(grpmn + scale * direction, basis)
        return jnp.vdot(weights, matrix)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=3.0e-9, atol=1.0e-11)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_source_matrix_solve_chain_gradients_match_finite_difference(lasym):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis = _mode_basis_for_rhs_tests(lasym=lasym)
    gsource = jnp.asarray(np.linspace(-0.2, 0.35, int(basis["nuv_full"]), dtype=float))
    g_direction = jnp.asarray(np.sin(np.arange(int(basis["nuv_full"]), dtype=float) + 0.2))
    mnpd2 = int(basis["mnpd2"])
    nuv3 = int(basis["nuv3"])
    rows = np.arange(mnpd2, dtype=float)[:, None]
    cols = np.arange(nuv3, dtype=float)[None, :]
    grpmn = jnp.asarray(0.01 * np.sin(0.4 + 0.2 * rows + 0.15 * cols), dtype=float)
    grpmn_direction = jnp.asarray(0.02 * np.cos(0.2 * rows - 0.1 * cols), dtype=float)
    phi_weights = jnp.asarray(np.cos(0.3 + np.arange(nuv3, dtype=float)), dtype=float)

    def response(source_scale, matrix_scale):
        rhs = _mode_rhs_from_basis(gsource + source_scale * g_direction, basis)
        matrix = _mode_matrix_from_basis(grpmn + matrix_scale * grpmn_direction, basis)
        out = dense_mode_vacuum_solve_jax(
            matrix,
            rhs,
            basis["sinmni"],
            basis["cosmni"] if lasym else None,
        )
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_matrix = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_matrix = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=5.0e-8, atol=1.0e-10)
    np.testing.assert_allclose(exact_matrix, fd_matrix, rtol=5.0e-8, atol=1.0e-10)


def test_jax_vmec_nonsingular_green_terms_match_numpy_reference():
    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    bexni = np.linspace(-0.18, 0.24, int(basis["nuv3"]), dtype=float)

    actual_gsource, actual_grpmn = _jax_nonsingular_terms(sample, basis, bexni)
    expected_gsource, expected_grpmn = _vmec_nonsingular_terms_from_bexni(
        sample=sample,
        basis=basis,
        bexni=bexni,
        signgs=1,
        nvper=2,
    )

    np.testing.assert_allclose(actual_gsource, expected_gsource, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(actual_grpmn, expected_grpmn, rtol=2.0e-12, atol=2.0e-12)


def test_jax_vmec_nonsingular_green_solve_chain_gradients_match_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    base_bex = jnp.asarray(np.linspace(-0.18, 0.24, int(basis["nuv3"]), dtype=float))
    bex_direction = jnp.asarray(np.cos(0.2 + np.arange(int(basis["nuv3"]), dtype=float)))
    phi_weights = jnp.asarray(np.sin(0.1 + np.arange(int(basis["nuv3"]), dtype=float)), dtype=float)

    def response(source_scale, geometry_scale):
        tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)
        gsource, grpmn = vmec_nonsingular_terms_from_bexni_jax(
            R=jnp.asarray(sample.R) + 0.04 * geometry_scale,
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=base_bex + source_scale * bex_direction,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
        rhs = _mode_rhs_from_basis(gsource, basis)
        matrix = _mode_matrix_from_basis(grpmn, basis)
        out = dense_mode_vacuum_solve_jax(matrix, rhs, basis["sinmni"], basis["cosmni"])
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_geometry = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_geometry = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=7.0e-7, atol=1.0e-10)
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=7.0e-7, atol=1.0e-10)


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_analytic_terms_match_numpy_reference(lasym):
    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    if not lasym:
        basis = _mode_basis_for_rhs_tests(lasym=False)
    bexni = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)

    actual_bvec, actual_grpmn = _jax_analytic_terms(sample, basis, bexni)
    expected_bvec, expected_grpmn = _vmec_analytic_terms_from_geometry(
        sample=sample,
        basis=basis,
        bexni=bexni,
        signgs=1,
    )

    np.testing.assert_allclose(actual_bvec, expected_bvec, rtol=4.0e-12, atol=4.0e-12)
    np.testing.assert_allclose(actual_grpmn, expected_grpmn, rtol=4.0e-12, atol=4.0e-12)


def test_jax_vmec_analytic_terms_validate_geometry_basis_and_source_shapes():
    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    bexni = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)

    with pytest.raises(ValueError, match="R must be a 2D"):
        vmec_analytic_terms_from_geometry_jax(
            R=np.ravel(sample.R),
            Ru=sample.Ru,
            Rv=sample.Rv,
            Zu=sample.Zu,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            signgs=1,
        )
    with pytest.raises(ValueError, match="Ru must match R shape"):
        vmec_analytic_terms_from_geometry_jax(
            R=sample.R,
            Ru=sample.Ru[:, :-1],
            Rv=sample.Rv,
            Zu=sample.Zu,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            signgs=1,
        )
    bad_basis = dict(basis)
    bad_basis["theta"] = np.asarray(basis["theta"])[:-1]
    with pytest.raises(ValueError, match="basis theta/zeta"):
        _jax_analytic_terms(sample, bad_basis, bexni)
    with pytest.raises(ValueError, match="bexni"):
        _jax_analytic_terms(sample, basis, bexni[:2])


@pytest.mark.parametrize("lasym", [False, True])
def test_jax_vmec_analytic_mode_solve_chain_gradients_match_finite_difference(lasym):
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    if not lasym:
        basis = _mode_basis_for_rhs_tests(lasym=False)
    base_bex = jnp.asarray(np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float))
    bex_direction = jnp.asarray(np.sin(0.17 + np.arange(int(basis["nuv3"]), dtype=float)))
    phi_weights = jnp.asarray(np.cos(0.19 + np.arange(int(basis["nuv3"]), dtype=float)), dtype=float)

    def response(source_scale, geometry_scale):
        bvec, grpmn = vmec_analytic_terms_from_geometry_jax(
            R=jnp.asarray(sample.R) + 0.02 * geometry_scale,
            Ru=sample.Ru,
            Rv=sample.Rv,
            Zu=sample.Zu,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=base_bex + source_scale * bex_direction,
            basis=basis,
            signgs=1,
        )
        matrix = _mode_matrix_from_basis(grpmn, basis)
        out = dense_mode_vacuum_solve_jax(
            matrix,
            bvec,
            basis["sinmni"],
            basis["cosmni"] if lasym else None,
        )
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_geometry = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_geometry = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=2.0e-6, atol=1.0e-10)
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=2.0e-6, atol=1.0e-10)


def test_jax_vmec_combined_analytic_nonsingular_solve_chain_gradients_match_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    base_bex = jnp.asarray(np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float))
    bex_direction = jnp.asarray(np.cos(0.23 + np.arange(int(basis["nuv3"]), dtype=float)))
    phi_weights = jnp.asarray(np.sin(0.27 + np.arange(int(basis["nuv3"]), dtype=float)), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)

    def response(source_scale, geometry_scale):
        R = jnp.asarray(sample.R) + 0.015 * geometry_scale
        bex = base_bex + source_scale * bex_direction
        gsource_nonsing, grpmn_nonsing = vmec_nonsingular_terms_from_bexni_jax(
            R=R,
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bex,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
        bvec_analytic, grpmn_analytic = vmec_analytic_terms_from_geometry_jax(
            R=R,
            Ru=sample.Ru,
            Rv=sample.Rv,
            Zu=sample.Zu,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bex,
            basis=basis,
            signgs=1,
        )
        rhs = _mode_rhs_from_basis(gsource_nonsing, basis) + bvec_analytic
        matrix = _mode_matrix_from_basis(grpmn_nonsing + grpmn_analytic, basis)
        out = dense_mode_vacuum_solve_jax(matrix, rhs, basis["sinmni"], basis["cosmni"])
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_geometry = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_geometry = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=3.0e-6, atol=1.0e-10)
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=3.0e-6, atol=1.0e-10)


def test_dense_vmec_nestor_mode_solve_matches_manual_combined_operator():
    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    bex = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)

    actual = dense_vmec_nestor_mode_solve_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bex,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
    )
    gsource_nonsing, grpmn_nonsing = _jax_nonsingular_terms(sample, basis, bex)
    bvec_analytic, grpmn_analytic = _jax_analytic_terms(sample, basis, bex)
    rhs = _mode_rhs_from_basis(gsource_nonsing, basis) + bvec_analytic
    matrix = _mode_matrix_from_basis(grpmn_nonsing + grpmn_analytic, basis)
    expected = dense_mode_vacuum_solve_jax(matrix, rhs, basis["sinmni"], basis["cosmni"])

    np.testing.assert_allclose(actual["rhs_mode"], rhs, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(actual["mode_matrix"], matrix, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(actual["mode_coeffs"], expected["mode_coeffs"], rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(actual["phi_flat"], expected["phi_flat"], rtol=1.0e-13, atol=1.0e-13)


def test_dense_vmec_nestor_mode_solve_matches_host_reduced_symmetric_grid():
    """Reduced stellarator-symmetric samples should match the host full-grid reconstruction path."""

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample(lasym=False)
    bex = np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)

    actual = dense_vmec_nestor_mode_solve_jax(
        R=sample.R,
        Z=sample.Z,
        Ru=sample.Ru,
        Zu=sample.Zu,
        Rv=sample.Rv,
        Zv=sample.Zv,
        ruu=sample.ruu,
        ruv=sample.ruv,
        rvv=sample.rvv,
        zuu=sample.zuu,
        zuv=sample.zuv,
        zvv=sample.zvv,
        bexni=bex,
        basis=basis,
        tables=tables,
        signgs=1,
        nvper=2,
    )
    gsource_nonsing, grpmn_nonsing = _vmec_nonsingular_terms_from_bexni(
        sample=sample,
        basis=basis,
        bexni=bex,
        signgs=1,
        nvper=2,
    )
    bvec_analytic, grpmn_analytic = _vmec_analytic_terms_from_geometry(
        sample=sample,
        basis=basis,
        bexni=bex,
        signgs=1,
    )
    rhs = _vmec_bvec_from_gsource(gsource=gsource_nonsing, basis=basis) + bvec_analytic
    matrix = _vmec_mode_matrix_from_grpmn(grpmn=grpmn_nonsing + grpmn_analytic, basis=basis)
    expected = dense_mode_vacuum_solve_jax(matrix, rhs, basis["sinmni"])

    np.testing.assert_allclose(actual["gsource_nonsing"], gsource_nonsing, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["grpmn"], grpmn_nonsing + grpmn_analytic, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["rhs_mode"], rhs, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["mode_matrix"], matrix, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["mode_coeffs"], expected["mode_coeffs"], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual["phi_flat"], expected["phi_flat"], rtol=1.0e-12, atol=1.0e-12)


def test_dense_vmec_nestor_mode_solve_gradients_match_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    basis, sample = _nonsingular_boundary_sample()
    base_bex = jnp.asarray(np.linspace(-0.11, 0.29, int(basis["nuv3"]), dtype=float))
    bex_direction = jnp.asarray(np.cos(0.23 + np.arange(int(basis["nuv3"]), dtype=float)))
    phi_weights = jnp.asarray(np.sin(0.27 + np.arange(int(basis["nuv3"]), dtype=float)), dtype=float)
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)

    def response(source_scale, geometry_scale):
        out = dense_vmec_nestor_mode_solve_jax(
            R=jnp.asarray(sample.R) + 0.015 * geometry_scale,
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=base_bex + source_scale * bex_direction,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.1 * jnp.vdot(
            phi_weights,
            out["phi_flat"],
        )

    exact_source = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    exact_geometry = jax.grad(lambda scale: response(0.0, scale))(0.0)
    eps = 1.0e-6
    fd_source = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    fd_geometry = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact_source, fd_source, rtol=3.0e-6, atol=1.0e-10)
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=3.0e-6, atol=1.0e-10)


def test_free_boundary_adjoint_operator_validation_errors_are_explicit():
    """Guard the public validation contract of the JAX NESTOR operator blocks."""

    basis, sample = _nonsingular_boundary_sample()
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=2)
    bexni = np.linspace(-0.18, 0.24, int(basis["nuv3"]), dtype=float)

    with pytest.raises(ValueError, match="square dense matrix"):
        dense_vacuum_solve_jax(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="leading dimension"):
        dense_vacuum_solve_jax(np.eye(2), np.ones(3))
    with pytest.raises(ValueError, match="requires imirr"):
        vmec_source_from_gsource_jax(np.ones(4), onp=1.0, lasym=False)

    with pytest.raises(ValueError, match="sin_basis must be a 2D array"):
        mode_rhs_from_gsource_jax(np.ones(3), sin_basis=np.ones(3), xmpot=np.arange(3), n_raw=np.arange(3), onp=1.0, lasym=True)
    with pytest.raises(ValueError, match="cos_basis is required"):
        mode_rhs_from_gsource_jax(
            np.ones(3),
            sin_basis=np.ones((3, 2)),
            xmpot=np.arange(2),
            n_raw=np.arange(2),
            onp=1.0,
            lasym=True,
        )
    with pytest.raises(ValueError, match="cos_basis must match"):
        mode_rhs_from_gsource_jax(
            np.ones(3),
            sin_basis=np.ones((3, 2)),
            cos_basis=np.ones((3, 1)),
            xmpot=np.arange(2),
            n_raw=np.arange(2),
            onp=1.0,
            lasym=True,
        )

    with pytest.raises(ValueError, match="grpmn must be a 2D array"):
        mode_matrix_from_grpmn_jax(np.ones(4), sin_basis=np.ones((3, 2)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=False)
    with pytest.raises(ValueError, match="sin_basis must be a 2D array"):
        mode_matrix_from_grpmn_jax(np.ones((2, 3)), sin_basis=np.ones(2), xmpot=np.arange(2), n_raw=np.arange(2), lasym=False)
    with pytest.raises(ValueError, match="invalid_grpmn_shape"):
        mode_matrix_from_grpmn_jax(np.ones((1, 3)), sin_basis=np.ones((3, 2)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=False)
    with pytest.raises(ValueError, match="invalid_grpmn_shape_lasym"):
        mode_matrix_from_grpmn_jax(np.ones((2, 3)), sin_basis=np.ones((3, 2)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=True, cos_basis=np.ones((3, 2)))
    with pytest.raises(ValueError, match="cos_basis is required"):
        mode_matrix_from_grpmn_jax(np.ones((4, 3)), sin_basis=np.ones((3, 2)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=True)
    with pytest.raises(ValueError, match="cos_basis must match"):
        mode_matrix_from_grpmn_jax(np.ones((4, 3)), sin_basis=np.ones((3, 2)), cos_basis=np.ones((3, 1)), xmpot=np.arange(2), n_raw=np.arange(2), lasym=True)

    with pytest.raises(ValueError, match="R must be a 2D"):
        vmec_nonsingular_terms_from_bexni_jax(
            R=np.ones(3),
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
    with pytest.raises(ValueError, match="Z must match R shape"):
        vmec_nonsingular_terms_from_bexni_jax(
            R=sample.R,
            Z=sample.Z[:, :-1],
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            tables=tables,
            signgs=1,
            nvper=2,
        )
    bad_basis = dict(basis)
    bad_basis["nu_full"] = int(basis["nu_full"]) + 1
    with pytest.raises(ValueError, match="nu_full"):
        _jax_nonsingular_terms(sample, bad_basis, bexni)
    bad_tables = dict(tables)
    bad_tables["cosui"] = np.empty((1, 0), dtype=float)
    with pytest.raises(ValueError, match="table shape"):
        vmec_nonsingular_terms_from_bexni_jax(
            R=sample.R,
            Z=sample.Z,
            Ru=sample.Ru,
            Zu=sample.Zu,
            Rv=sample.Rv,
            Zv=sample.Zv,
            ruu=sample.ruu,
            ruv=sample.ruv,
            rvv=sample.rvv,
            zuu=sample.zuu,
            zuv=sample.zuv,
            zvv=sample.zvv,
            bexni=bexni,
            basis=basis,
            tables=bad_tables,
            signgs=1,
            nvper=2,
        )
    with pytest.raises(ValueError, match="bexni"):
        _jax_nonsingular_terms(sample, basis, bexni[:2])

    with pytest.raises(ValueError, match="sin_basis must be a 2D array"):
        dense_mode_vacuum_solve_jax(np.eye(2), np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="must match sin_basis columns"):
        dense_mode_vacuum_solve_jax(np.eye(2), np.ones(2), np.ones((3, 3)))
    with pytest.raises(ValueError, match="cos_basis must match"):
        dense_mode_vacuum_solve_jax(np.eye(4), np.ones(4), np.ones((3, 2)), np.ones((3, 1)))
    with pytest.raises(ValueError, match="2 \\* sin_basis columns"):
        dense_mode_vacuum_solve_jax(np.eye(3), np.ones(3), np.ones((3, 2)), np.ones((3, 2)))


def _mode_vacuum_inputs(*, lasym: bool = False):
    from vmec_jax._compat import jnp

    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [
            [3.0, 0.2, -0.1],
            [0.4, 2.6, 0.3],
            [-0.2, 0.1, 2.4],
        ],
        dtype=float,
    )
    rhs = jnp.asarray([0.5, -0.2, 0.4], dtype=float)
    if not lasym:
        return mode_matrix, rhs, sin_basis, None

    cos_basis = jnp.asarray(
        [
            [0.5, -0.3, 0.1],
            [-0.2, 0.4, -0.6],
            [0.3, 0.2, 0.7],
            [-0.1, -0.5, 0.2],
        ],
        dtype=float,
    )
    top = jnp.concatenate([mode_matrix + 0.8 * jnp.eye(3), 0.1 * jnp.eye(3)], axis=1)
    bottom = jnp.concatenate([-0.05 * jnp.eye(3), mode_matrix + 1.1 * jnp.eye(3)], axis=1)
    return jnp.concatenate([top, bottom], axis=0), jnp.concatenate([rhs, -0.3 * rhs]), sin_basis, cos_basis


def test_dense_mode_vacuum_solve_reconstructs_grid_potential():
    from vmec_jax._compat import jnp

    enable_x64(True)
    A, rhs, sin_basis, _cos_basis = _mode_vacuum_inputs()

    actual = dense_mode_vacuum_solve_jax(A, rhs, sin_basis)
    coeffs = jnp.linalg.solve(A, rhs)

    np.testing.assert_allclose(actual["mode_coeffs"], coeffs, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(actual["phi_flat"], sin_basis @ coeffs, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(actual["residual"], np.zeros_like(np.asarray(rhs)), atol=1.0e-14)


def test_dense_mode_vacuum_solve_reconstructs_lasym_grid_potential():
    from vmec_jax._compat import jnp

    enable_x64(True)
    A, rhs, sin_basis, cos_basis = _mode_vacuum_inputs(lasym=True)

    actual = dense_mode_vacuum_solve_jax(A, rhs, sin_basis, cos_basis)
    coeffs = jnp.linalg.solve(A, rhs)
    nmodes = sin_basis.shape[1]

    np.testing.assert_allclose(actual["mode_coeffs"], coeffs, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(
        actual["phi_flat"],
        sin_basis @ coeffs[:nmodes] + cos_basis @ coeffs[nmodes:],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_dense_mode_vacuum_gradient_wrt_rhs_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, rhs, sin_basis, _cos_basis = _mode_vacuum_inputs()
    direction = jnp.asarray([0.2, -0.3, 0.1], dtype=float)
    weights = jnp.asarray([0.7, -0.2, 0.4, 0.1], dtype=float)

    def objective(scale):
        response = dense_mode_vacuum_solve_jax(A, rhs + scale * direction, sin_basis)
        return jnp.vdot(weights, response["phi_flat"])

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-9, atol=1.0e-11)


def test_dense_mode_vacuum_gradient_wrt_matrix_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, rhs, sin_basis, _cos_basis = _mode_vacuum_inputs()
    dA = jnp.asarray(
        [
            [0.0, 0.1, -0.2],
            [0.05, 0.0, 0.1],
            [-0.1, 0.2, 0.0],
        ],
        dtype=float,
    )
    weights = jnp.asarray([0.7, -0.2, 0.4, 0.1], dtype=float)

    def objective(scale):
        response = dense_mode_vacuum_solve_jax(A + scale * dA, rhs, sin_basis)
        return jnp.vdot(weights, response["phi_flat"])

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=3.0e-9, atol=1.0e-11)


def _toy_coil_vacuum_response(*, current_scale: float = 0.0, radius_shift: float = 0.0):
    """Small direct-coil -> vacuum-linear-solve chain for adjoint checks."""

    from vmec_jax._compat import jnp

    radius = 1.15 + 0.02 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([3.0e7 * (1.0 + 0.01 * current_scale)], dtype=float),
        n_segments=96,
        regularization_epsilon=1.0e-9,
    )
    R = jnp.asarray([0.24, 0.37, 0.51], dtype=float)
    Z = jnp.asarray([0.11, -0.17, 0.23], dtype=float)
    phi = jnp.asarray([0.0, 0.4, 0.9], dtype=float)
    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    rhs = jnp.stack(
        (
            br[0] + 0.3 * bphi[1],
            bz[1] - 0.2 * br[2],
            bphi[2] + 0.5 * bz[0],
        )
    )
    A = jnp.asarray(
        [
            [2.7, 0.2, -0.1],
            [0.1, 2.2, 0.3],
            [-0.2, 0.4, 2.5],
        ],
        dtype=float,
    )
    x = dense_vacuum_solve_jax(A, rhs)
    return 0.5 * jnp.vdot(x, x) + 0.1 * jnp.vdot(rhs, rhs)


def _toy_coil_nonlinear_response(*, current_scale: float = 0.0, radius_shift: float = 0.0):
    """Small direct-coil -> nonlinear implicit-root chain for phase-2 checks."""

    from vmec_jax._compat import jnp

    radius = 1.25 + 0.03 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([6.0e6 * (1.0 + 0.02 * current_scale)], dtype=float),
        n_segments=96,
        regularization_epsilon=1.0e-9,
    )
    R = jnp.asarray([0.32, 0.47, 0.61], dtype=float)
    Z = jnp.asarray([0.09, -0.16, 0.19], dtype=float)
    phi = jnp.asarray([0.0, 0.35, 0.8], dtype=float)
    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    rhs = 0.04 * jnp.stack(
        (
            br[0] + 0.2 * bphi[1] - 0.1 * bz[2],
            bz[1] - 0.3 * br[2] + 0.1 * bphi[0],
        )
    )

    def residual(x, residual_params):
        local_rhs = residual_params["rhs"]
        return jnp.asarray(
            [
                x[0] + 0.1 * x[0] ** 3 + 0.08 * x[1] - local_rhs[0],
                0.9 * x[1] + 0.05 * x[0] ** 2 + 0.15 * jnp.sin(x[1]) - local_rhs[1],
            ]
        )

    root = dense_nonlinear_solve_jax(
        residual,
        jnp.asarray([0.0, 0.0], dtype=float),
        {"rhs": rhs},
        max_iter=14,
    )
    weights = jnp.asarray([0.9, 1.3], dtype=float)
    return 0.5 * jnp.vdot(weights * root, root) + 0.02 * jnp.vdot(rhs, rhs)


def _toy_coil_free_boundary_fixed_point_response(
    *,
    current_scale: float = 0.0,
    radius_shift: float = 0.0,
):
    """Direct coils -> state-dependent boundary -> vacuum response fixed point."""

    from vmec_jax._compat import jnp

    radius = 1.28 + 0.02 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([4.5e6 * (1.0 + 0.015 * current_scale)], dtype=float),
        n_segments=64,
        regularization_epsilon=1.0e-9,
    )
    theta_shape = (2, 2)
    phi = jnp.asarray([[0.0, 0.4], [0.8, 1.2]], dtype=float)
    Ru = jnp.asarray([[0.02, -0.03], [0.04, -0.02]], dtype=float)
    Zu = jnp.asarray([[0.19, 0.21], [0.18, 0.20]], dtype=float)
    Rv = jnp.asarray([[0.03, 0.01], [-0.02, 0.04]], dtype=float)
    Zv = jnp.asarray([[0.02, -0.01], [0.03, -0.02]], dtype=float)
    weights = jnp.asarray([[0.4, -0.2], [0.6, -0.3]], dtype=float)
    A = jnp.asarray([[2.6, 0.18], [-0.12, 2.4]], dtype=float)

    def update(state, params):
        R = jnp.asarray([[0.72, 0.81], [0.86, 0.76]], dtype=float) + 0.04 * state[0]
        Z = jnp.asarray([[0.11, -0.12], [0.17, -0.15]], dtype=float) + 0.03 * state[1]
        br, bphi, bz = sample_coil_field_cylindrical(params["coil"], R, Z, phi)
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bphi,
            bz=bz,
            R=R,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
        )
        rhs = 0.03 * jnp.asarray(
            [
                jnp.mean(vac["bnormal_unit"] * weights),
                jnp.mean((vac["bsqvac"] - jnp.mean(vac["bsqvac"])) * weights),
            ]
        )
        response = dense_vacuum_solve_jax(A, rhs)
        update_state = jnp.asarray([0.03, -0.02], dtype=float) + 0.45 * jnp.tanh(response)
        if update_state.shape != (2,):  # pragma: no cover - defensive shape guard.
            raise AssertionError(f"unexpected update shape {update_state.shape} for boundary grid {theta_shape}")
        return update_state

    root = dense_fixed_point_solve_jax(
        update,
        jnp.asarray([0.0, 0.0], dtype=float),
        {"coil": coil_params},
        max_iter=12,
    )
    final = update(root, {"coil": coil_params})
    return 0.5 * jnp.vdot(jnp.asarray([1.1, 0.9], dtype=float) * root, root) + 0.1 * jnp.vdot(final, final)


def _toy_coil_projected_mode_fixed_point_response(
    *,
    current_scale: float = 0.0,
    radius_shift: float = 0.0,
):
    """Direct coils -> moving boundary -> mode vacuum response -> fixed point."""

    from vmec_jax._compat import jnp

    radius = 1.34 + 0.025 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([5.5e6 * (1.0 + 0.012 * current_scale)], dtype=float),
        n_segments=64,
        regularization_epsilon=1.0e-9,
    )
    phi = jnp.asarray([[0.05, 0.45], [0.85, 1.25]], dtype=float)
    Ru_base = jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float)
    Zu_base = jnp.asarray([[0.20, 0.22], [0.19, 0.21]], dtype=float)
    Rv_base = jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float)
    Zv_base = jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [
            [3.2, 0.12, -0.06],
            [0.16, 2.7, 0.21],
            [-0.09, 0.24, 2.9],
        ],
        dtype=float,
    )
    mode_to_state = jnp.asarray(
        [
            [0.09, -0.04, 0.03],
            [-0.02, 0.08, 0.05],
            [0.04, 0.03, -0.07],
        ],
        dtype=float,
    )

    def boundary_from_state(state):
        base_R = jnp.asarray([[0.74, 0.83], [0.89, 0.78]], dtype=float)
        base_Z = jnp.asarray([[0.10, -0.13], [0.18, -0.16]], dtype=float)
        shape = jnp.asarray([[0.25, -0.15], [0.35, -0.20]], dtype=float)
        R = base_R + 0.035 * state[0] + 0.018 * state[2] * shape
        Z = base_Z + 0.028 * state[1] - 0.012 * state[2] * shape
        return {
            "R": R,
            "Z": Z,
            "phi": phi,
            "Ru": Ru_base + 0.01 * state[2],
            "Zu": Zu_base - 0.008 * state[2],
            "Rv": Rv_base,
            "Zv": Zv_base,
        }

    def update_from_response(_state, response, _vac, _boundary, _params):
        mode_coeffs = jnp.asarray(response["mode_coeffs"])
        pressure_like = jnp.asarray([0.02, -0.015, 0.01], dtype=float)
        return pressure_like + 0.22 * jnp.tanh(mode_to_state @ mode_coeffs)

    solved = direct_coil_projected_mode_fixed_point_objective_jax(
        coil_params,
        jnp.asarray([0.0, 0.0, 0.0], dtype=float),
        boundary_from_state=boundary_from_state,
        update_from_response=update_from_response,
        mode_matrix=mode_matrix,
        sin_basis=sin_basis,
        xmpot=jnp.asarray([0, 1, 1]),
        n_raw=jnp.asarray([0, 0, 1]),
        imirr=jnp.asarray([1, 0, 3, 2]),
        nuv3=4,
        nuv_full=4,
        max_iter=14,
        state_weights=jnp.asarray([1.0, 0.8, 1.2], dtype=float),
        update_weights=0.08,
        mode_weights=0.02,
        fixed_point_residual_weight=10.0,
    )
    return solved["objective"]


def test_dense_vacuum_adjoint_chain_wrt_coil_current_matches_finite_difference():
    """Validate a direct-coil field feeding an implicit vacuum solve."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda scale: _toy_coil_vacuum_response(current_scale=scale))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_vacuum_response(current_scale=eps)
        - _toy_coil_vacuum_response(current_scale=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=2.0e-6, atol=1.0e-10)


def test_dense_vacuum_adjoint_chain_wrt_coil_geometry_matches_finite_difference():
    """Validate the same chain for a Fourier curve coefficient perturbation."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda shift: _toy_coil_vacuum_response(radius_shift=shift))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_vacuum_response(radius_shift=eps)
        - _toy_coil_vacuum_response(radius_shift=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=2.0e-6, atol=1.0e-10)


def test_dense_nonlinear_adjoint_chain_wrt_coil_current_matches_finite_difference():
    """Validate direct-coil controls through a nonlinear implicit root."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda scale: _toy_coil_nonlinear_response(current_scale=scale))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_nonlinear_response(current_scale=eps)
        - _toy_coil_nonlinear_response(current_scale=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=5.0e-6, atol=1.0e-10)


def test_dense_nonlinear_adjoint_chain_wrt_coil_geometry_matches_finite_difference():
    """Validate a coil Fourier perturbation through a nonlinear implicit root."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda shift: _toy_coil_nonlinear_response(radius_shift=shift))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_nonlinear_response(radius_shift=eps)
        - _toy_coil_nonlinear_response(radius_shift=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=5.0e-6, atol=1.0e-10)


def test_dense_fixed_point_direct_coil_loop_wrt_current_matches_finite_difference():
    """Validate a miniature complete free-boundary fixed-point coil loop."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda scale: _toy_coil_free_boundary_fixed_point_response(current_scale=scale))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_free_boundary_fixed_point_response(current_scale=eps)
        - _toy_coil_free_boundary_fixed_point_response(current_scale=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-9
    np.testing.assert_allclose(exact, fd, rtol=8.0e-6, atol=1.0e-10)


def test_dense_fixed_point_direct_coil_loop_wrt_geometry_matches_finite_difference():
    """Validate the fixed-point loop for one coil Fourier geometry coefficient."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda shift: _toy_coil_free_boundary_fixed_point_response(radius_shift=shift))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_free_boundary_fixed_point_response(radius_shift=eps)
        - _toy_coil_free_boundary_fixed_point_response(radius_shift=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-9
    np.testing.assert_allclose(exact, fd, rtol=8.0e-6, atol=1.0e-10)


def test_dense_fixed_point_projected_mode_loop_wrt_current_matches_finite_difference():
    """Validate moving-boundary direct-coil fixed point through mode response."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda scale: _toy_coil_projected_mode_fixed_point_response(current_scale=scale))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_mode_fixed_point_response(current_scale=eps)
        - _toy_coil_projected_mode_fixed_point_response(current_scale=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-9
    np.testing.assert_allclose(exact, fd, rtol=1.0e-5, atol=1.0e-10)


def test_dense_fixed_point_projected_mode_loop_wrt_geometry_matches_finite_difference():
    """Validate moving-boundary fixed point for one coil Fourier coefficient."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda shift: _toy_coil_projected_mode_fixed_point_response(radius_shift=shift))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_mode_fixed_point_response(radius_shift=eps)
        - _toy_coil_projected_mode_fixed_point_response(radius_shift=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-9
    np.testing.assert_allclose(exact, fd, rtol=1.0e-5, atol=1.0e-10)


def test_projected_mode_fixed_point_objective_exposes_components():
    """Check the scalar objective wrapper returns usable diagnostics."""

    from vmec_jax._compat import jnp

    enable_x64(True)
    value = _toy_coil_projected_mode_fixed_point_response()
    assert np.isfinite(float(value))
    assert float(value) > 0.0

    radius = 1.34
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([5.5e6], dtype=float),
        n_segments=32,
        regularization_epsilon=1.0e-9,
    )
    phi = jnp.asarray([[0.05, 0.45], [0.85, 1.25]], dtype=float)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )

    def boundary_from_state(state):
        return {
            "R": jnp.asarray([[0.74, 0.83], [0.89, 0.78]], dtype=float) + 0.02 * state[0],
            "Z": jnp.asarray([[0.10, -0.13], [0.18, -0.16]], dtype=float) + 0.02 * state[1],
            "phi": phi,
            "Ru": jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float),
            "Zu": jnp.asarray([[0.20, 0.22], [0.19, 0.21]], dtype=float),
            "Rv": jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float),
            "Zv": jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float),
        }

    def update_from_response(_state, response, _vac, _boundary, _params):
        return 0.1 * jnp.tanh(jnp.asarray(response["mode_coeffs"])[:2])

    solved = direct_coil_projected_mode_fixed_point_objective_jax(
        coil_params,
        jnp.asarray([0.0, 0.0], dtype=float),
        boundary_from_state=boundary_from_state,
        update_from_response=update_from_response,
        mode_matrix=jnp.asarray([[3.2, 0.12, -0.06], [0.16, 2.7, 0.21], [-0.09, 0.24, 2.9]]),
        sin_basis=sin_basis,
        xmpot=jnp.asarray([0, 1, 1]),
        n_raw=jnp.asarray([0, 0, 1]),
        imirr=jnp.asarray([1, 0, 3, 2]),
        nuv3=4,
        nuv_full=4,
        max_iter=10,
        state_weights=1.0,
        mode_weights=0.01,
    )

    assert {"state", "mode", "fixed_point_residual"}.issubset(solved["objective_components"])
    np.testing.assert_allclose(solved["fixed_point_residual"], np.zeros(2), atol=1.0e-11)
    assert float(solved["objective"]) >= float(solved["objective_components"]["state"])


def test_projected_mode_fixed_point_objective_value_and_grad_wrt_coil_pytree():
    """Validate the optimizer-facing scalar objective has coil pytree gradients."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    radius = 1.34
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    coil_params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([5.5e6], dtype=float),
        n_segments=32,
        regularization_epsilon=1.0e-9,
    )
    phi = jnp.asarray([[0.05, 0.45], [0.85, 1.25]], dtype=float)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [[3.2, 0.12, -0.06], [0.16, 2.7, 0.21], [-0.09, 0.24, 2.9]],
        dtype=float,
    )
    mode_to_state = jnp.asarray([[0.08, -0.04, 0.02], [-0.03, 0.06, 0.05]], dtype=float)

    def boundary_from_state(state):
        shape = jnp.asarray([[0.25, -0.15], [0.35, -0.20]], dtype=float)
        return {
            "R": jnp.asarray([[0.74, 0.83], [0.89, 0.78]], dtype=float)
            + 0.02 * state[0]
            + 0.01 * state[1] * shape,
            "Z": jnp.asarray([[0.10, -0.13], [0.18, -0.16]], dtype=float)
            + 0.025 * state[1],
            "phi": phi,
            "Ru": jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float),
            "Zu": jnp.asarray([[0.20, 0.22], [0.19, 0.21]], dtype=float),
            "Rv": jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float),
            "Zv": jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float),
        }

    def update_from_response(_state, response, _vac, _boundary, _params):
        return jnp.asarray([0.01, -0.015], dtype=float) + 0.12 * jnp.tanh(
            mode_to_state @ jnp.asarray(response["mode_coeffs"])
        )

    def objective(params):
        solved = direct_coil_projected_mode_fixed_point_objective_jax(
            params,
            jnp.asarray([0.0, 0.0], dtype=float),
            boundary_from_state=boundary_from_state,
            update_from_response=update_from_response,
            mode_matrix=mode_matrix,
            sin_basis=sin_basis,
            xmpot=jnp.asarray([0, 1, 1]),
            n_raw=jnp.asarray([0, 0, 1]),
            imirr=jnp.asarray([1, 0, 3, 2]),
            nuv3=4,
            nuv_full=4,
            max_iter=10,
            state_weights=jnp.asarray([1.0, 0.7], dtype=float),
            mode_weights=0.02,
            rhs_mode_weights=0.01,
            bnormal_weight=0.005,
            fixed_point_residual_weight=10.0,
        )
        return solved["objective"]

    value, grad_params = jax.value_and_grad(objective)(coil_params)

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad_params.base_currents)))
    assert np.all(np.isfinite(np.asarray(grad_params.base_curve_dofs)))
    assert float(jnp.linalg.norm(grad_params.base_currents)) > 1.0e-18
    assert float(jnp.linalg.norm(grad_params.base_curve_dofs)) > 1.0e-10


def _boundary_projection_inputs():
    from vmec_jax._compat import jnp

    br = jnp.asarray([[0.11, -0.07], [0.05, 0.09]], dtype=float)
    bp = jnp.asarray([[0.31, 0.22], [-0.18, 0.14]], dtype=float)
    bz = jnp.asarray([[-0.12, 0.08], [0.16, -0.05]], dtype=float)
    R = jnp.asarray([[1.2, 1.1], [0.9, 1.05]], dtype=float)
    Ru = jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float)
    Zu = jnp.asarray([[0.25, 0.23], [0.21, 0.24]], dtype=float)
    Rv = jnp.asarray([[0.07, 0.02], [-0.05, 0.04]], dtype=float)
    Zv = jnp.asarray([[0.01, -0.03], [0.06, -0.02]], dtype=float)
    return br, bp, bz, R, Ru, Zu, Rv, Zv


def test_jax_boundary_projection_matches_numpy_reference():
    enable_x64(True)
    br, bp, bz, R, Ru, Zu, Rv, Zv = _boundary_projection_inputs()

    actual = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    expected = vacuum_boundary_fields_from_cylindrical(
        br=np.asarray(br),
        bp=np.asarray(bp),
        bz=np.asarray(bz),
        R=np.asarray(R),
        Ru=np.asarray(Ru),
        Zu=np.asarray(Zu),
        Rv=np.asarray(Rv),
        Zv=np.asarray(Zv),
    )

    for key in ("bu", "bv", "bsupu", "bsupv", "bsqvac", "bnormal", "bnormal_unit", "det_guv"):
        np.testing.assert_allclose(actual[key], getattr(expected, key), rtol=1.0e-13, atol=1.0e-13)


def test_jax_boundary_projection_gradient_wrt_field_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    br, bp, bz, R, Ru, Zu, Rv, Zv = _boundary_projection_inputs()
    weights = jnp.asarray([[0.4, -0.2], [0.7, -0.5]], dtype=float)
    direction = jnp.asarray([[0.3, -0.1], [0.2, 0.5]], dtype=float)

    def objective(scale):
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br + scale * direction,
            bp=bp,
            bz=bz,
            R=R,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
        )
        return jnp.sum(weights * vac["bsqvac"]) + 0.2 * jnp.sum(vac["bnormal_unit"] ** 2)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=5.0e-8, atol=1.0e-10)


def test_jax_boundary_projection_gradient_wrt_geometry_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    br, bp, bz, R, Ru, Zu, Rv, Zv = _boundary_projection_inputs()
    weights = jnp.asarray([[0.4, -0.2], [0.7, -0.5]], dtype=float)
    direction = jnp.asarray([[0.1, 0.2], [-0.3, 0.4]], dtype=float)

    def objective(scale):
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bp,
            bz=bz,
            R=R + scale * direction,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
        )
        return jnp.sum(weights * vac["bsqvac"]) + 0.2 * jnp.sum(vac["bnormal"] ** 2)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=5.0e-8, atol=1.0e-10)


def _toy_coil_projected_vacuum_response(*, current_scale: float = 0.0, radius_shift: float = 0.0):
    """Direct coils -> boundary projection -> implicit vacuum solve."""

    from vmec_jax._compat import jnp

    radius = 1.45 + 0.03 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.5e7 * (1.0 + 0.02 * current_scale)], dtype=float),
        n_segments=128,
        regularization_epsilon=1.0e-9,
    )
    R = jnp.asarray([[0.78, 0.86], [0.92, 0.81]], dtype=float)
    Z = jnp.asarray([[0.16, -0.13], [0.22, -0.19]], dtype=float)
    phi = jnp.asarray([[0.05, 0.45], [0.9, 1.25]], dtype=float)
    Ru = jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float)
    Zu = jnp.asarray([[0.22, 0.24], [0.21, 0.23]], dtype=float)
    Rv = jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float)
    Zv = jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float)

    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    vac = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bphi,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    weights = jnp.asarray([[0.4, -0.2], [0.7, -0.5]], dtype=float)
    rhs = jnp.stack(
        (
            jnp.mean(vac["bsqvac"]),
            jnp.mean(vac["bnormal_unit"] * weights),
            jnp.mean((vac["bu"] - 0.3 * vac["bv"]) * weights),
        )
    )
    A = jnp.asarray(
        [
            [2.8, 0.15, -0.08],
            [0.2, 2.4, 0.25],
            [-0.1, 0.3, 2.6],
        ],
        dtype=float,
    )
    x = dense_vacuum_solve_jax(A, rhs)
    return 0.5 * jnp.vdot(x, x) + 0.05 * jnp.mean(vac["bnormal"] ** 2)


def _toy_coil_projected_mode_vacuum_response(*, current_scale: float = 0.0, radius_shift: float = 0.0):
    """Direct coils -> projection -> mode-space vacuum solve."""

    from vmec_jax._compat import jnp

    radius = 1.45 + 0.03 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.5e7 * (1.0 + 0.02 * current_scale)], dtype=float),
        n_segments=128,
        regularization_epsilon=1.0e-9,
    )
    R = jnp.asarray([[0.78, 0.86], [0.92, 0.81]], dtype=float)
    Z = jnp.asarray([[0.16, -0.13], [0.22, -0.19]], dtype=float)
    phi = jnp.asarray([[0.05, 0.45], [0.9, 1.25]], dtype=float)
    Ru = jnp.asarray([[0.03, -0.04], [0.02, 0.05]], dtype=float)
    Zu = jnp.asarray([[0.22, 0.24], [0.21, 0.23]], dtype=float)
    Rv = jnp.asarray([[0.04, 0.01], [-0.03, 0.05]], dtype=float)
    Zv = jnp.asarray([[0.02, -0.03], [0.06, -0.01]], dtype=float)
    sin_basis = jnp.asarray(
        [
            [0.0, 0.2, -0.3],
            [0.4, -0.1, 0.5],
            [-0.2, 0.6, 0.1],
            [0.7, 0.3, -0.4],
        ],
        dtype=float,
    )
    mode_matrix = jnp.asarray(
        [
            [3.1, 0.15, -0.08],
            [0.2, 2.5, 0.25],
            [-0.1, 0.3, 2.7],
        ],
        dtype=float,
    )

    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    vac = vacuum_boundary_fields_from_cylindrical_jax(
        br=br,
        bp=bphi,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    rhs_mode = mode_rhs_from_gsource_jax(
        vac["bnormal"],
        sin_basis=sin_basis,
        xmpot=jnp.asarray([0, 1, 1]),
        n_raw=jnp.asarray([0, 0, 1]),
        onp=1.0,
        lasym=False,
        imirr=jnp.asarray([1, 0, 3, 2]),
        nuv3=4,
        nuv_full=4,
    )
    response = dense_mode_vacuum_solve_jax(mode_matrix, rhs_mode, sin_basis)
    weights = jnp.asarray([0.7, -0.2, 0.4, 0.1], dtype=float)
    return 0.5 * jnp.vdot(response["mode_coeffs"], response["mode_coeffs"]) + 0.1 * jnp.vdot(
        weights,
        response["phi_flat"],
    )


def test_dense_vacuum_adjoint_chain_through_projection_wrt_current_matches_finite_difference():
    """Validate the next rung in the coil-to-vacuum adjoint chain."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda scale: _toy_coil_projected_vacuum_response(current_scale=scale))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_vacuum_response(current_scale=eps)
        - _toy_coil_projected_vacuum_response(current_scale=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=3.0e-6, atol=1.0e-10)


def test_dense_vacuum_adjoint_chain_through_projection_wrt_geometry_matches_finite_difference():
    """Validate projected vacuum sensitivity to one coil Fourier coefficient."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda shift: _toy_coil_projected_vacuum_response(radius_shift=shift))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_vacuum_response(radius_shift=eps)
        - _toy_coil_projected_vacuum_response(radius_shift=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=3.0e-6, atol=1.0e-10)


def test_dense_mode_vacuum_chain_through_projection_wrt_current_matches_finite_difference():
    """Validate the mode-space scaffold in a direct-coil projected chain."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda scale: _toy_coil_projected_mode_vacuum_response(current_scale=scale))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_mode_vacuum_response(current_scale=eps)
        - _toy_coil_projected_mode_vacuum_response(current_scale=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=4.0e-6, atol=1.0e-10)


def test_dense_mode_vacuum_chain_through_projection_wrt_geometry_matches_finite_difference():
    """Validate the mode-space scaffold for a coil Fourier perturbation."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda shift: _toy_coil_projected_mode_vacuum_response(radius_shift=shift))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_projected_mode_vacuum_response(radius_shift=eps)
        - _toy_coil_projected_mode_vacuum_response(radius_shift=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=4.0e-6, atol=1.0e-10)

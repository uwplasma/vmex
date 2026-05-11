from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.field import (
    TWOPI,
    b2_from_bsup,
    b_cartesian_from_bsup,
    bsub_from_bsup,
    bsup_from_sqrtg_lambda,
    full_mesh_from_half_mesh_avg,
    half_mesh_avg_from_full_mesh,
)


def test_half_mesh_average_roundtrips_full_mesh_profile():
    full = np.asarray([0.0, 1.25, 0.7, -0.2, 0.4], dtype=float)

    half = np.asarray(half_mesh_avg_from_full_mesh(full))
    roundtrip = np.asarray(full_mesh_from_half_mesh_avg(half))

    np.testing.assert_allclose(half[0], 1.5 * full[1] - 0.5 * full[2], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(half[1:-1], 0.5 * (full[1:-1] + full[2:]), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(half[-1], 1.5 * full[-1] - 0.5 * full[-2], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(roundtrip, full, rtol=0.0, atol=1.0e-14)


def test_bsup_from_sqrtg_lambda_applies_vmec_flux_and_lambda_signs():
    sqrtg = np.asarray(
        [
            [[2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0]],
        ],
        dtype=float,
    )
    lam_u = np.asarray(
        [
            [[0.10, -0.20], [0.30, -0.40]],
            [[0.15, -0.25], [0.35, -0.45]],
        ],
        dtype=float,
    )
    lam_v = np.asarray(
        [
            [[-0.50, 0.20], [-0.10, 0.40]],
            [[0.45, -0.35], [0.25, -0.15]],
        ],
        dtype=float,
    )
    phip_internal = np.asarray([1.5, -2.0], dtype=float)
    chip_internal = np.asarray([-0.25, 0.75], dtype=float)
    signgs = -1
    lamscale = 0.4

    bsupu_public, bsupv_public = bsup_from_sqrtg_lambda(
        sqrtg=sqrtg,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=phip_internal * TWOPI * signgs,
        chipf=chip_internal * TWOPI * signgs,
        signgs=signgs,
        lamscale=lamscale,
        flux_is_internal=False,
    )
    bsupu_internal, bsupv_internal = bsup_from_sqrtg_lambda(
        sqrtg=sqrtg,
        lam_u=lam_u,
        lam_v=lam_v,
        phipf=phip_internal,
        chipf=chip_internal,
        signgs=signgs,
        lamscale=lamscale,
        flux_is_internal=True,
    )

    denom = signgs * sqrtg
    expected_bsupu = -((chip_internal[:, None, None] - lamscale * lam_v) / denom)
    expected_bsupv = -((phip_internal[:, None, None] + lamscale * lam_u) / denom)

    np.testing.assert_allclose(np.asarray(bsupu_public), expected_bsupu, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bsupv_public), expected_bsupv, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bsupu_public), np.asarray(bsupu_internal), rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bsupv_public), np.asarray(bsupv_internal), rtol=1.0e-13, atol=1.0e-13)


def test_cartesian_field_norm_matches_metric_contraction_for_circular_geometry():
    theta = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    ones = np.ones((2, theta.size, zeta.size), dtype=float)
    minor_radius = np.asarray([0.35, 0.70], dtype=float)[:, None, None]
    cos_theta = np.cos(theta)[None, :, None]
    sin_theta = np.sin(theta)[None, :, None]
    sin_zeta = np.sin(zeta)[None, None, :]
    major_radius = 2.5

    r_cyl = (major_radius + minor_radius * cos_theta) * ones
    rt = (-minor_radius * sin_theta) * ones
    zt = (minor_radius * cos_theta) * ones
    rp = np.zeros_like(r_cyl)
    zp = np.zeros_like(r_cyl)
    geom = SimpleNamespace(
        R=r_cyl,
        Rt=rt,
        Zt=zt,
        Rp=rp,
        Zp=zp,
        g_tt=(minor_radius * minor_radius) * ones,
        g_tp=np.zeros_like(r_cyl),
        g_pp=r_cyl * r_cyl,
    )
    bsupu = 0.2 + 0.03 * cos_theta + 0.01 * sin_zeta
    bsupv = -0.4 + 0.05 * cos_theta - 0.02 * sin_zeta

    bsubu, bsubv = bsub_from_bsup(geom, bsupu, bsupv)
    b2_metric = b2_from_bsup(geom, bsupu, bsupv)
    bcart = b_cartesian_from_bsup(geom, bsupu, bsupv, zeta=zeta, nfp=3)
    b2_cartesian = np.sum(np.asarray(bcart) ** 2, axis=-1)

    np.testing.assert_allclose(np.asarray(bsubu), geom.g_tt * bsupu, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bsubv), geom.g_pp * bsupv, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(b2_cartesian, np.asarray(b2_metric), rtol=1.0e-13, atol=1.0e-13)

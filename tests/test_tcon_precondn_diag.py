from __future__ import annotations

import numpy as np
import pytest


def _reference_preconditioning_matrix(
    *,
    xs,
    xu12,
    xu_e,
    xu_o,
    x1_o,
    r12,
    total_pressure,
    sqrtg,
    bsupv,
    w_int,
    sqrt_sh,
    sm,
    sp,
    delta_s,
    ns_full,
):
    xs = np.asarray(xs, dtype=float)
    xu12 = np.asarray(xu12, dtype=float)
    xu_e = np.asarray(xu_e, dtype=float)
    xu_o = np.asarray(xu_o, dtype=float)
    x1_o = np.asarray(x1_o, dtype=float)
    r12 = np.asarray(r12, dtype=float)
    total_pressure = np.asarray(total_pressure, dtype=float)
    sqrtg = np.asarray(sqrtg, dtype=float)
    bsupv = np.asarray(bsupv, dtype=float)
    w_int = np.asarray(w_int, dtype=float)
    sqrt_sh = np.asarray(sqrt_sh, dtype=float)
    sm = np.asarray(sm, dtype=float)
    sp = np.asarray(sp, dtype=float)
    ns_half = int(xs.shape[0])
    ntheta = int(xs.shape[1])
    nzeta = int(xs.shape[2])
    ax = np.zeros((ns_half, 4), dtype=float)
    bx = np.zeros((ns_half, 3), dtype=float)
    cx = np.zeros((ns_half,), dtype=float)
    pfactor = -4.0
    for jh in range(ns_half):
        sh = sqrt_sh[jh] if sqrt_sh[jh] != 0.0 else 1.0
        for kl in range(ntheta * nzeta):
            l = kl % ntheta
            k = kl // ntheta
            p_tau = pfactor * r12[jh, l, k] * r12[jh, l, k] * total_pressure[jh, l, k] / sqrtg[jh, l, k] * w_int[l]
            t1a = xu12[jh, l, k] / delta_s
            t2a = 0.25 * (xu_e[jh + 1, l, k] / sh + xu_o[jh + 1, l, k]) / sh
            t3a = 0.25 * (xu_e[jh, l, k] / sh + xu_o[jh, l, k]) / sh
            ax[jh, 0] += p_tau * t1a * t1a
            ax[jh, 1] += p_tau * (t1a + t2a) * (-t1a + t3a)
            ax[jh, 2] += p_tau * (t1a + t2a) * (t1a + t2a)
            ax[jh, 3] += p_tau * (-t1a + t3a) * (-t1a + t3a)
            t1b = 0.5 * (xs[jh, l, k] + 0.5 / sh * x1_o[jh + 1, l, k])
            t2b = 0.5 * (xs[jh, l, k] + 0.5 / sh * x1_o[jh, l, k])
            bx[jh, 0] += p_tau * t1b * t2b
            bx[jh, 1] += p_tau * t1b * t1b
            bx[jh, 2] += p_tau * t2b * t2b
            cx[jh] += 0.25 * pfactor * (bsupv[jh, l, k] ** 2) * sqrtg[jh, l, k] * w_int[l]

    axm = np.stack([-ax[:, 0], ax[:, 1] * sm * sp], axis=1)
    bxm = np.stack([bx[:, 0], bx[:, 0] * sm * sp], axis=1)
    z = np.zeros((1,), dtype=float)
    axd = np.stack(
        [
            np.concatenate([z, ax[:, 0]], axis=0)[:ns_full] + np.concatenate([ax[:, 0], z], axis=0)[:ns_full],
            np.concatenate([z, ax[:, 2] * (sm * sm)], axis=0)[:ns_full]
            + np.concatenate([ax[:, 3] * (sp * sp), z], axis=0)[:ns_full],
        ],
        axis=1,
    )
    bxd = np.stack(
        [
            np.concatenate([z, bx[:, 1]], axis=0)[:ns_full] + np.concatenate([bx[:, 2], z], axis=0)[:ns_full],
            np.concatenate([z, bx[:, 1] * (sm * sm)], axis=0)[:ns_full]
            + np.concatenate([bx[:, 2] * (sp * sp), z], axis=0)[:ns_full],
        ],
        axis=1,
    )
    cxd = np.concatenate([z, cx], axis=0)[:ns_full] + np.concatenate([cx, z], axis=0)[:ns_full]
    return axm, axd, bxm, bxd, cxd


def _synthetic_precond_inputs():
    xs = np.array([[[0.9], [1.1]], [[1.2], [1.4]]], dtype=float)
    xu12 = np.array([[[0.2], [0.3]], [[0.4], [0.5]]], dtype=float)
    xu_e = np.array([[[0.6], [0.7]], [[0.8], [0.9]], [[1.0], [1.1]]], dtype=float)
    xu_o = np.array([[[0.15], [0.2]], [[0.25], [0.3]], [[0.35], [0.4]]], dtype=float)
    x1_o = np.array([[[0.05], [0.07]], [[0.09], [0.11]], [[0.13], [0.15]]], dtype=float)
    r12 = np.array([[[1.8], [1.6]], [[1.4], [1.2]]], dtype=float)
    total_pressure = np.array([[[2.0], [2.2]], [[2.4], [2.6]]], dtype=float)
    sqrtg = np.array([[[0.5], [0.7]], [[0.8], [1.1]]], dtype=float)
    tau = np.full_like(sqrtg, 9.0)
    bsupv = np.array([[[0.4], [0.45]], [[0.5], [0.55]]], dtype=float)
    w_int = np.array([0.3, 0.7], dtype=float)
    sqrt_sh = np.array([0.2, 0.6], dtype=float)
    sm = np.array([0.4, 0.8], dtype=float)
    sp = np.array([0.4, 1.2], dtype=float)
    delta_s = 0.25
    ns_full = 3
    return {
        "xs": xs,
        "xu12": xu12,
        "xu_e": xu_e,
        "xu_o": xu_o,
        "x1_o": x1_o,
        "r12": r12,
        "total_pressure": total_pressure,
        "tau": tau,
        "bsupv": bsupv,
        "sqrtg": sqrtg,
        "w_int": w_int,
        "sqrt_sh": sqrt_sh,
        "sm": sm,
        "sp": sp,
        "delta_s": delta_s,
        "ns_full": ns_full,
    }


def test_numpy_precond_matrix_uses_sqrtg_not_tau():
    from vmec_jax.preconditioner_1d import _compute_preconditioning_matrix

    inputs = _synthetic_precond_inputs()
    out = _compute_preconditioning_matrix(**inputs)
    ref = _reference_preconditioning_matrix(**{k: v for k, v in inputs.items() if k != "tau"})

    for got, want in zip(out, ref, strict=True):
        assert np.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_jax_precond_matrix_uses_sqrtg_not_tau():
    pytest.importorskip("jax")

    from vmec_jax.preconditioner_1d_jax import _compute_preconditioning_matrix

    inputs = _synthetic_precond_inputs()
    out = _compute_preconditioning_matrix(**inputs)
    ref = _reference_preconditioning_matrix(**{k: v for k, v in inputs.items() if k != "tau"})

    for got, want in zip(out, ref, strict=True):
        assert np.allclose(np.asarray(got), want, rtol=1e-12, atol=1e-12)


def test_tcon_from_bcovar_precondn_diag_matches_reference_formula():
    pytest.importorskip("jax")

    from vmec_jax._compat import jnp
    from vmec_jax.vmec_constraints import tcon_from_bcovar_precondn_diag
    from vmec_jax.vmec_tomnsp import vmec_trig_tables

    # Small synthetic problem with well-conditioned norms.
    ns = 6
    ntheta = 8
    nzeta = 4
    nfp = 1
    mmax = 4
    nmax = 2
    lasym = False

    s = jnp.linspace(0.0, 1.0, ns)
    trig = vmec_trig_tables(ntheta=ntheta, nzeta=nzeta, nfp=nfp, mmax=mmax, nmax=nmax, lasym=lasym, dtype=jnp.float64)

    # Shapes (ns, ntheta3, nzeta).
    bsq = jnp.ones((ns, int(trig.ntheta3), nzeta), dtype=jnp.float64) * 2.0
    r12 = jnp.ones_like(bsq) * 1.3
    sqrtg = jnp.ones_like(bsq) * 0.9
    ru12 = jnp.ones_like(bsq) * 0.7
    zu12 = jnp.ones_like(bsq) * 0.4

    # ru0/zu0 should vary on angles so norms are nontrivial.
    th = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, int(trig.ntheta3), endpoint=False))
    ze = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False))
    ru0 = (1.0 + 0.1 * jnp.sin(th)[:, None] + 0.05 * jnp.cos(ze)[None, :])[None, :, :] * jnp.ones((ns, 1, 1))
    zu0 = (0.9 + 0.08 * jnp.cos(th)[:, None] + 0.03 * jnp.sin(ze)[None, :])[None, :, :] * jnp.ones((ns, 1, 1))

    tcon0 = 0.3
    out = tcon_from_bcovar_precondn_diag(
        tcon0=tcon0,
        trig=trig,
        s=s,
        signgs=1,
        lasym=lasym,
        bsq=bsq,
        r12=r12,
        sqrtg=sqrtg,
        ru12=ru12,
        zu12=zu12,
        ru0=ru0,
        zu0=zu0,
    )

    out_np = np.asarray(out)
    assert out_np.shape == (ns,)
    assert np.isfinite(out_np).all()

    # Reference computation (numpy), mirroring bcovar.f scaling and the reduced precondn diagonal.
    hs = float(np.asarray(s[1] - s[0]))
    ohs = 1.0 / hs
    pfactor = -4.0 * float(trig.r0scale) ** 2
    wint = np.asarray(trig.cosmui3[:, 0]) / float(np.asarray(trig.mscale[0]))
    wint3 = wint[None, :, None] * np.ones((1, 1, nzeta))

    ptau = (pfactor * (np.asarray(r12) ** 2) * np.asarray(bsq) * wint3) / np.asarray(sqrtg)
    ax_r = np.sum(ptau * ((np.asarray(zu12) * ohs) ** 2), axis=(1, 2))
    ax_z = np.sum(ptau * ((np.asarray(ru12) * ohs) ** 2), axis=(1, 2))
    ax_r[0] = 0.0
    ax_z[0] = 0.0
    ard1 = ax_r + np.concatenate([ax_r[1:], np.zeros((1,))])
    azd1 = ax_z + np.concatenate([ax_z[1:], np.zeros((1,))])

    arnorm = np.sum((np.asarray(ru0) ** 2) * wint3, axis=(1, 2))
    aznorm = np.sum((np.asarray(zu0) ** 2) * wint3, axis=(1, 2))

    tcon0_clamped = min(abs(tcon0), 1.0)
    ns_f = float(ns)
    tcon_mul = tcon0_clamped * (1.0 + ns_f * (1.0 / 60.0 + ns_f / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (float(trig.r0scale) ** 2)) ** 2)

    ref = np.zeros((ns,), dtype=float)
    # VMEC sets `tcon(1)=tcon0` (clamped) before overwriting the interior.
    ref[0] = tcon0_clamped
    for js in range(1, ns - 1):
        ref[js] = min(abs(ard1[js]) / arnorm[js], abs(azd1[js]) / aznorm[js]) * (tcon_mul * (32.0 * hs) ** 2)
    ref[-1] = 0.5 * ref[-2]

    assert np.allclose(out_np, ref, rtol=2e-12, atol=2e-12)

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.kernels.residue import (
    VmecForceNorms,
    VmecForceNormsDynamic,
    vmec_force_norms_from_bcovar,
    vmec_force_norms_from_bcovar_dynamic,
    vmec_fsq_from_tomnsps,
    vmec_fsq_from_tomnsps_dynamic,
    vmec_scalxc_from_s,
    vmec_wint_from_trig,
)
from vmec_jax.kernels.tomnsp import TomnspsRZL, vmec_trig_tables


def test_dynamic_force_norms_match_vmec_scalar_path_for_synthetic_bcovar() -> None:
    """No-wout force normalization should preserve VMEC's scalar gate conventions."""
    trig = vmec_trig_tables(ntheta=8, nzeta=5, nfp=2, mmax=2, nmax=1, lasym=True, cache=False)
    ns = 4
    shape = (ns, int(trig.ntheta3), 5)
    js = np.arange(ns, dtype=float)[:, None, None]
    lt = np.arange(shape[1], dtype=float)[None, :, None]
    lz = np.arange(shape[2], dtype=float)[None, None, :]

    jac_phys = 1.0 + 0.17 * js + 0.03 * lt + 0.05 * lz
    bsupu = 0.35 + 0.04 * js + 0.01 * lt
    bsupv = 0.22 + 0.03 * js + 0.02 * lz
    bsubu = 0.80 + 0.02 * js + 0.03 * lt + 0.01 * lz
    bsubv = 0.55 + 0.01 * js + 0.02 * lt + 0.04 * lz
    b2 = bsupu * bsubu + bsupv * bsubv
    pressure = np.asarray([0.0, 1.8, 1.5, 1.2])
    r12 = 0.90 + 0.02 * js + 0.01 * lt
    guu = 1.40 + 0.04 * js + 0.02 * lt + 0.03 * lz
    lamscale = 1.7
    signgs = -1

    bc = SimpleNamespace(
        jac=SimpleNamespace(sqrtg=-jac_phys, r12=r12),
        guu=guu,
        bsupu=bsupu,
        bsupv=bsupv,
        bsubu=bsubu,
        bsubv=bsubv,
        bsq=0.5 * b2 + pressure[:, None, None],
        lamscale=lamscale,
    )
    s = np.linspace(0.0, 1.0, ns)

    dynamic = vmec_force_norms_from_bcovar_dynamic(bc=bc, trig=trig, s=s, signgs=signgs)

    w_ang = np.asarray(vmec_wint_from_trig(trig, nzeta=shape[2]), dtype=float)
    axis_mask = (np.arange(ns) > 0).astype(float)[:, None, None]
    hs = float(s[1] - s[0])

    vp_expected = np.sum(w_ang[None, :, :] * jac_phys * axis_mask, axis=(1, 2))
    volume_expected = hs * float(np.sum(vp_expected[1:]))
    wblocal = np.sum(w_ang[None, :, :] * jac_phys * (0.5 * b2) * axis_mask, axis=(1, 2))
    wb_expected = hs * abs(float(np.sum(wblocal[1:])))
    wp_expected = hs * float(np.sum(vp_expected[1:] * pressure[1:]))
    assert wp_expected > wb_expected

    r2_expected = max(wb_expected, wp_expected) / volume_expected
    denom_f = float(np.sum(guu * (r12 * r12) * w_ang[None, :, :] * axis_mask))
    fnorm_expected = 1.0 / (denom_f * r2_expected * r2_expected)
    denom_l = float(np.sum((bsubu * bsubu + bsubv * bsubv) * w_ang[None, :, :] * axis_mask))
    fnorm_l_expected = 1.0 / (denom_l * lamscale * lamscale)
    r1_expected = 1.0 / (2.0 * float(trig.r0scale)) ** 2

    np.testing.assert_allclose(np.asarray(dynamic.vp), vp_expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(float(dynamic.volume), volume_expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(float(dynamic.wb), wb_expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(float(dynamic.wp), wp_expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(float(dynamic.r2), r2_expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(float(dynamic.fnorm), fnorm_expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(float(dynamic.fnormL), fnorm_l_expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(float(dynamic.r1), r1_expected, rtol=1.0e-13, atol=1.0e-13)

    # The dynamic no-wout path must stay algebraically identical to the legacy
    # wout-scalar path when the same VMEC scalar profiles are supplied.
    static = vmec_force_norms_from_bcovar(
        bc=bc,
        trig=trig,
        wout=SimpleNamespace(vp=vp_expected, wb=wb_expected, wp=wp_expected),
        s=s,
    )
    np.testing.assert_allclose(static.fnorm, float(dynamic.fnorm), rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(static.fnormL, float(dynamic.fnormL), rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(static.r1, float(dynamic.r1), rtol=1.0e-13, atol=1.0e-13)


def test_fsq_scalar_gates_apply_vmec_scaling_lasym_blocks_and_edge_policy() -> None:
    """Protect the getfsq policy used by required parity checks."""
    ns = 4
    mpol = 3
    ntor1 = 2
    shape = (ns, mpol, ntor1)

    def block(offset: float) -> np.ndarray:
        return (np.arange(np.prod(shape), dtype=float).reshape(shape) + offset) / 10.0

    frzl = TomnspsRZL(
        frcc=block(1.0),
        frss=block(10.0),
        fzsc=block(20.0),
        fzcs=block(30.0),
        flsc=block(40.0),
        flcs=block(50.0),
        frsc=block(60.0),
        frcs=block(70.0),
        fzcc=block(80.0),
        fzss=block(90.0),
        flcc=block(100.0),
        flss=block(110.0),
    )
    s = np.asarray([0.0, 1.0 / 9.0, 4.0 / 9.0, 1.0])
    scalxc = np.asarray(vmec_scalxc_from_s(s=s, mpol=mpol), dtype=float)[:, :, None]

    def scaled_sum(blocks: tuple[np.ndarray | None, ...], radial_slice: slice) -> float:
        return float(sum(np.sum((np.asarray(arr) * scalxc)[radial_slice] ** 2) for arr in blocks if arr is not None))

    r_blocks = (frzl.frcc, frzl.frss, frzl.frsc, frzl.frcs)
    z_blocks = (frzl.fzsc, frzl.fzcs, frzl.fzcc, frzl.fzss)
    l_blocks = (frzl.flsc, frzl.flcs, frzl.flcc, frzl.flss)

    gcr2_no_edge = scaled_sum(r_blocks, slice(None, ns - 1))
    gcz2_no_edge = scaled_sum(z_blocks, slice(None, ns - 1))
    gcr2_with_edge = scaled_sum(r_blocks, slice(None))
    gcz2_with_edge = scaled_sum(z_blocks, slice(None))
    gcl2_all = scaled_sum(l_blocks, slice(None))
    assert gcr2_with_edge > gcr2_no_edge
    assert gcz2_with_edge > gcz2_no_edge

    static_norms = VmecForceNorms(fnorm=0.25, fnormL=0.5, r1=4.0)
    dynamic_norms = VmecForceNormsDynamic(
        fnorm=static_norms.fnorm,
        fnormL=static_norms.fnormL,
        r1=static_norms.r1,
        r2=0.0,
        volume=0.0,
        wb=0.0,
        wp=0.0,
        vp=np.zeros(ns),
    )

    fsq = vmec_fsq_from_tomnsps(
        frzl=frzl,
        norms=static_norms,
        lconm1=False,
        apply_m1_constraints=False,
        include_edge=False,
        apply_scalxc=True,
        s=s,
    )
    fsq_dynamic = vmec_fsq_from_tomnsps_dynamic(
        frzl=frzl,
        norms=dynamic_norms,
        lconm1=False,
        apply_m1_constraints=False,
        include_edge=False,
        apply_scalxc=True,
        s=s,
    )
    expected_no_edge = np.asarray(
        [
            static_norms.r1 * static_norms.fnorm * gcr2_no_edge,
            static_norms.r1 * static_norms.fnorm * gcz2_no_edge,
            static_norms.fnormL * gcl2_all,
        ]
    )
    np.testing.assert_allclose([fsq.fsqr, fsq.fsqz, fsq.fsql], expected_no_edge, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(
        [float(fsq_dynamic.fsqr), float(fsq_dynamic.fsqz), float(fsq_dynamic.fsql)],
        expected_no_edge,
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    fsq_with_edge = vmec_fsq_from_tomnsps(
        frzl=frzl,
        norms=static_norms,
        lconm1=False,
        apply_m1_constraints=False,
        include_edge=True,
        apply_scalxc=True,
        s=s,
    )
    expected_with_edge = np.asarray(
        [
            static_norms.r1 * static_norms.fnorm * gcr2_with_edge,
            static_norms.r1 * static_norms.fnorm * gcz2_with_edge,
            static_norms.fnormL * gcl2_all,
        ]
    )
    np.testing.assert_allclose(
        [fsq_with_edge.fsqr, fsq_with_edge.fsqz, fsq_with_edge.fsql],
        expected_with_edge,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(fsq_with_edge.fsql, fsq.fsql, rtol=1.0e-13, atol=1.0e-13)

from __future__ import annotations

from dataclasses import replace

import numpy as np

import vmec_jax.vmec_tomnsp as vt
from vmec_jax.vmec_tomnsp import tomnspa_rzl, tomnsps_masks, tomnsps_rzl, vmec_trig_tables


def _transform_inputs(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    base = np.linspace(-0.7, 0.9, int(np.prod(shape)), dtype=float).reshape(shape)
    names = (
        "armn_even",
        "armn_odd",
        "brmn_even",
        "brmn_odd",
        "crmn_even",
        "crmn_odd",
        "azmn_even",
        "azmn_odd",
        "bzmn_even",
        "bzmn_odd",
        "czmn_even",
        "czmn_odd",
    )
    return {name: base + 0.125 * i for i, name in enumerate(names)}


def test_tomnsps_fft_fused_and_split_paths_are_consistent(monkeypatch) -> None:
    trig = vmec_trig_tables(ntheta=6, nzeta=4, nfp=2, mmax=2, nmax=1, lasym=False, cache=False)
    args = _transform_inputs((4, int(trig.ntheta3), int(np.asarray(trig.cosnv).shape[0])))

    monkeypatch.setattr(vt, "_TOMNSPS_FFT_ENV", "1")
    vt._TOMNSPS_FFT_CACHE.clear()
    monkeypatch.setenv("VMEC_JAX_TOMNSPS_FFT_FUSED", "1")
    fused = tomnsps_rzl(**args, mpol=2, ntor=1, nfp=2, lasym=False, trig=trig)

    monkeypatch.setenv("VMEC_JAX_TOMNSPS_FFT_FUSED", "0")
    split = tomnsps_rzl(**args, mpol=2, ntor=1, nfp=2, lasym=False, trig=trig)

    for name in ("frcc", "fzsc", "flsc", "frss", "fzcs", "flcs"):
        np.testing.assert_allclose(
            np.asarray(getattr(fused, name)),
            np.asarray(getattr(split, name)),
            rtol=2.0e-14,
            atol=2.0e-14,
        )

    axisym = tomnsps_rzl(**args, mpol=2, ntor=0, nfp=2, lasym=False, trig=trig)
    assert axisym.frss is None
    assert axisym.fzcs is None
    assert axisym.flcs is None


def test_tomnsps_unfused_dft_fallback_tables_host_masks_and_deterministic_reductions(monkeypatch) -> None:
    trig = vmec_trig_tables(ntheta=6, nzeta=4, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    trig_without_caches = replace(
        trig,
        cosmui_nt2=None,
        sinmui_nt2=None,
        cosmumi_nt2=None,
        sinmumi_nt2=None,
        basis_theta_cs_nt2=None,
        basis_zeta_cs=None,
    )
    args = _transform_inputs((4, int(trig.ntheta3), int(np.asarray(trig.cosnv).shape[0])))
    masks = replace(
        tomnsps_masks(ns=4, mpol=2, include_edge=True, dtype=np.float64, cache=False),
        mask_even_j=None,
        mask_rz_j=None,
        mask_l_j=None,
        xmpq1_j=None,
    )

    monkeypatch.setattr(vt, "_TOMNSPS_FFT_ENV", "0")
    vt._TOMNSPS_FFT_CACHE.clear()
    monkeypatch.setattr(vt, "_TOMNSPS_THETA_FUSED", False)
    monkeypatch.setattr(vt, "_TOMNSPS_ZETA_FUSED", False)

    out = tomnsps_rzl(
        **args,
        mpol=2,
        ntor=1,
        nfp=1,
        lasym=True,
        trig=trig_without_caches,
        include_edge=True,
        masks=masks,
    )
    assert out.frcc.shape == (4, 2, 2)
    assert out.frss is not None
    assert np.all(np.isfinite(np.asarray(out.frcc)))

    monkeypatch.setattr(vt, "_DETERMINISTIC_REDUCE", True)
    arr_theta = np.arange(2 * 1 * 3 * 4 * 2, dtype=float).reshape(2, 1, 3, 4, 2)
    mat_theta = np.linspace(-0.2, 0.4, 4 * 3, dtype=float).reshape(4, 3)
    det_theta = np.asarray(vt._theta_contract(arr_theta, mat_theta))
    ref_theta = np.einsum("apsik,im->apsmk", arr_theta, mat_theta)
    np.testing.assert_allclose(det_theta, ref_theta)

    arr_zeta = np.arange(1 * 3 * 2 * 4, dtype=float).reshape(1, 3, 2, 4)
    mat_zeta = np.linspace(0.1, 0.7, 4 * 3, dtype=float).reshape(4, 3)
    det_zeta = np.asarray(vt._zeta_contract(arr_zeta, mat_zeta))
    ref_zeta = np.einsum("psmk,kn->psmn", arr_zeta, mat_zeta)
    np.testing.assert_allclose(det_zeta, ref_zeta)

    def precision_rejecting_einsum(expr, *operands, precision=None):
        if precision is not None:
            raise TypeError("precision unsupported")
        return np.einsum(expr, *operands)

    monkeypatch.setattr(vt, "_JNP_EINSUM", precision_rejecting_einsum)
    fallback = vt._einsum("ij,jk->ik", np.ones((1, 1)), np.ones((1, 1)))
    np.testing.assert_allclose(np.asarray(fallback), [[1.0]])


def test_tomnspa_unfused_and_fused_zeta_paths_with_scale_envs(monkeypatch) -> None:
    trig = vmec_trig_tables(ntheta=6, nzeta=4, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    trig_without_caches = replace(
        trig,
        cosmui_nt2=None,
        sinmui_nt2=None,
        cosmumi_nt2=None,
        sinmumi_nt2=None,
        basis_theta_cs_nt2=None,
        basis_theta_mu_nt2=None,
        basis_zeta_all=None,
    )
    args = _transform_inputs((4, int(trig.ntheta3), int(np.asarray(trig.cosnv).shape[0])))
    masks = replace(
        tomnsps_masks(ns=4, mpol=2, include_edge=False, dtype=np.float64, cache=False),
        mask_rz_j=None,
        mask_l_j=None,
        xmpq1_j=None,
    )

    monkeypatch.setattr(vt, "_TOMNSPS_THETA_FUSED", False)
    monkeypatch.setattr(vt, "_TOMNSPA_ZETA_FUSED", False)
    monkeypatch.setenv("VMEC_JAX_TOMNSPA_LAM_SCALE", "not-a-number")
    unfused = tomnspa_rzl(
        **args,
        mpol=2,
        ntor=1,
        nfp=1,
        lasym=True,
        trig=trig_without_caches,
        masks=masks,
    )
    assert unfused.frsc.shape == (4, 2, 2)
    assert unfused.frcs is not None
    np.testing.assert_allclose(np.asarray(unfused.frsc[-1]), 0.0)

    monkeypatch.setattr(vt, "_TOMNSPS_THETA_FUSED", True)
    monkeypatch.setattr(vt, "_TOMNSPA_ZETA_FUSED", True)
    monkeypatch.setenv("VMEC_JAX_TOMNSPA_LAM_SCALE", "yes")
    fused = tomnspa_rzl(
        **args,
        mpol=2,
        ntor=1,
        nfp=1,
        lasym=True,
        trig=trig_without_caches,
        masks=masks,
    )
    assert fused.frsc.shape == unfused.frsc.shape
    assert fused.flcc is not None
    assert fused.flss is not None
    assert np.all(np.isfinite(np.asarray(fused.flcc)))

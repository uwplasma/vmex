from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import vmec_jax.kernels.tomnsp as vt
from vmec_jax.kernels.tomnsp import tomnspa_rzl, tomnsps_masks, tomnsps_rzl, vmec_trig_tables


def test_tomnsps_fft_policy_override_is_scoped(monkeypatch) -> None:
    monkeypatch.setattr(vt, "_TOMNSPS_FFT_ENV", "")
    vt._TOMNSPS_FFT_CACHE[:] = [False]

    with vt.tomnsps_fft_policy_override(True):
        assert vt._TOMNSPS_FFT_ENV == "1"
        assert vt._TOMNSPS_FFT_CACHE == []
        assert vt._get_tomnsps_fft() is True

    assert vt._TOMNSPS_FFT_ENV == ""
    assert vt._TOMNSPS_FFT_CACHE == [False]

    monkeypatch.setattr(vt, "_TOMNSPS_FFT_ENV", "0")
    vt._TOMNSPS_FFT_CACHE.clear()
    with vt.tomnsps_fft_policy_override(True):
        assert vt._TOMNSPS_FFT_ENV == "0"
        assert vt._get_tomnsps_fft() is False


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


def test_tomnsp_pytree_roundtrips_preserve_cached_tables_and_outputs() -> None:
    trig = vmec_trig_tables(ntheta=6, nzeta=3, nfp=2, mmax=2, nmax=1, lasym=True, cache=False)

    children, aux = trig.tree_flatten()
    restored = type(trig).tree_unflatten(aux, children)

    assert restored.ntheta1 == trig.ntheta1
    assert restored.ntheta2 == trig.ntheta2
    assert restored.ntheta3 == trig.ntheta3
    assert restored.dnorm == trig.dnorm
    np.testing.assert_allclose(np.asarray(restored.cosmu), np.asarray(trig.cosmu))
    np.testing.assert_allclose(np.asarray(restored.basis_zeta_all), np.asarray(trig.basis_zeta_all))

    rzl = vt.TomnspsRZL(
        frcc=np.ones((1, 1, 1)),
        frss=None,
        fzsc=np.ones((1, 1, 1)) * 2.0,
        fzcs=None,
        flsc=np.ones((1, 1, 1)) * 3.0,
        flcs=None,
    )
    rzl_children, rzl_aux = rzl.tree_flatten()
    rzl_restored = vt.TomnspsRZL.tree_unflatten(rzl_aux, rzl_children)
    np.testing.assert_allclose(rzl_restored.frcc, rzl.frcc)
    assert rzl_restored.frss is None


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


def test_tomnsp_validation_cache_and_zero_size_helper_branches(monkeypatch) -> None:
    with pytest.raises(ValueError, match="Invalid theta sizes"):
        vmec_trig_tables(ntheta=0, nzeta=1, nfp=1, mmax=0, nmax=0, lasym=False, cache=False)
    with pytest.raises(ValueError, match="nzeta must be positive"):
        vmec_trig_tables(ntheta=4, nzeta=0, nfp=1, mmax=0, nmax=0, lasym=False, cache=False)
    with pytest.raises(ValueError, match="nfp must be positive"):
        vmec_trig_tables(ntheta=4, nzeta=1, nfp=0, mmax=0, nmax=0, lasym=False, cache=False)
    with pytest.raises(ValueError, match="mmax/nmax must be nonnegative"):
        vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=-1, nmax=0, lasym=False, cache=False)

    first = vmec_trig_tables(ntheta=4, nzeta=2, nfp=1, mmax=0, nmax=0, lasym=True, cache=True)
    second = vmec_trig_tables(ntheta=4, nzeta=2, nfp=1, mmax=0, nmax=0, lasym=True, cache=True)
    assert second is first
    assert vt.vmec_angle_grid(ntheta=4, nzeta=0, nfp=1, lasym=False, cache=False).zeta.shape == (1,)
    assert vt.vmec_angle_grid(ntheta=4, nzeta=2, nfp=1, lasym=True, cache=False).theta.shape == (4,)
    cached_grid = vt.vmec_angle_grid(ntheta=4, nzeta=2, nfp=1, lasym=False, cache=True)
    assert vt.vmec_angle_grid(ntheta=4, nzeta=2, nfp=1, lasym=False, cache=True) is cached_grid
    with pytest.raises(ValueError, match="Invalid theta sizes"):
        vt.vmec_angle_grid(ntheta=0, nzeta=2, nfp=1, lasym=False, cache=False)
    with pytest.raises(ValueError, match="nfp must be positive"):
        vt.vmec_angle_grid(ntheta=4, nzeta=2, nfp=0, lasym=False, cache=False)

    assert vt._theta_transform_fft(np.ones((1, 1, 0, 2)), mpol=3, dnorm=1.0, mscale=np.ones(3), want_sin=False).shape == (
        1,
        1,
        3,
        2,
    )
    assert vt._zeta_transform_fft(np.ones((2, 0)), ntor=2, nscale=np.ones(3), want_sin=True).shape == (2, 3)

    monkeypatch.setattr(vt, "has_jax", lambda: False)
    assert vt._cache_allowed() is True
    np.testing.assert_allclose(vt._einsum("ij,jk->ik", np.ones((1, 1)), np.ones((1, 1))), [[1.0]])
    arr_theta = np.arange(1 * 1 * 2 * 3 * 1, dtype=float).reshape(1, 1, 2, 3, 1)
    mat_theta = np.linspace(0.1, 0.3, 3 * 2, dtype=float).reshape(3, 2)
    np.testing.assert_allclose(vt._theta_contract(arr_theta, mat_theta), np.einsum("apsik,im->apsmk", arr_theta, mat_theta))
    arr_zeta = np.arange(1 * 2 * 2 * 3, dtype=float).reshape(1, 2, 2, 3)
    mat_zeta = np.linspace(0.2, 0.8, 3 * 2, dtype=float).reshape(3, 2)
    np.testing.assert_allclose(vt._zeta_contract(arr_zeta, mat_zeta), np.einsum("psmk,kn->psmn", arr_zeta, mat_zeta))
    np.testing.assert_array_equal(vt._mparity_mask(4, dtype=np.float64), [1.0, 0.0, 1.0, 0.0])
    assert vt._mparity_mask(4, dtype=np.float64) is vt._mparity_mask(4, dtype=np.float64)

    fake_jax = ModuleType("jax")
    fake_jax.core = SimpleNamespace(trace_ctx=SimpleNamespace(is_top_level=lambda: True))
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setattr(vt, "has_jax", lambda: True)
    assert vt._cache_allowed() is True

    fake_jax.core = SimpleNamespace(trace_ctx=SimpleNamespace(is_top_level=lambda: (_ for _ in ()).throw(RuntimeError("trace"))))
    assert vt._cache_allowed() is False
    fake_jax.core = SimpleNamespace(trace_ctx=SimpleNamespace(is_top_level=lambda: True))

    with pytest.raises(ValueError, match="ns must be positive"):
        tomnsps_masks(ns=0, mpol=1, include_edge=True, cache=False)
    with pytest.raises(ValueError, match="mpol must be positive"):
        tomnsps_masks(ns=1, mpol=0, include_edge=True, cache=False)
    masks = tomnsps_masks(ns=3, mpol=2, include_edge=True, dtype=np.float64, cache=True)
    assert tomnsps_masks(ns=3, mpol=2, include_edge=True, dtype=np.float64, cache=True) is masks


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


def test_tomnsp_transform_error_paths_and_mask_fallbacks() -> None:
    trig = vmec_trig_tables(ntheta=6, nzeta=4, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    args = _transform_inputs((3, int(trig.ntheta3), int(np.asarray(trig.cosnv).shape[0])))

    with pytest.raises(ValueError, match="mpol must be positive"):
        tomnsps_rzl(**args, mpol=0, ntor=1, nfp=1, lasym=True, trig=trig)
    with pytest.raises(ValueError, match="ntor must be nonnegative"):
        tomnsps_rzl(**args, mpol=2, ntor=-1, nfp=1, lasym=True, trig=trig)
    with pytest.raises(ValueError, match="mpol must be positive"):
        tomnspa_rzl(**args, mpol=0, ntor=1, nfp=1, lasym=True, trig=trig)
    with pytest.raises(ValueError, match="ntor must be nonnegative"):
        tomnspa_rzl(**args, mpol=2, ntor=-1, nfp=1, lasym=True, trig=trig)

    bad_grid = replace(trig, ntheta3=int(trig.ntheta3) + 1)
    with pytest.raises(ValueError, match="Input grid does not match trig tables"):
        tomnsps_rzl(**args, mpol=2, ntor=1, nfp=1, lasym=True, trig=bad_grid)
    with pytest.raises(ValueError, match="Input grid does not match trig tables"):
        tomnspa_rzl(**args, mpol=2, ntor=1, nfp=1, lasym=True, trig=bad_grid)

    mismatched_masks = SimpleNamespace(ns=99, mpol=2, include_edge=False)
    out = tomnsps_rzl(**args, mpol=2, ntor=1, nfp=1, lasym=True, trig=trig, masks=mismatched_masks)
    out_asym = tomnspa_rzl(**args, mpol=2, ntor=1, nfp=1, lasym=True, trig=trig, masks=mismatched_masks)
    assert out.frcc.shape == (3, 2, 2)
    assert out_asym.frsc.shape == (3, 2, 2)


def test_tomnsp_axisymmetric_dft_paths_keep_non_threed_outputs_none(monkeypatch) -> None:
    trig = vmec_trig_tables(ntheta=6, nzeta=4, nfp=1, mmax=2, nmax=1, lasym=True, cache=False)
    args = _transform_inputs((3, int(trig.ntheta3), int(np.asarray(trig.cosnv).shape[0])))

    monkeypatch.setattr(vt, "_TOMNSPS_FFT_ENV", "0")
    vt._TOMNSPS_FFT_CACHE.clear()
    monkeypatch.setattr(vt, "_TOMNSPS_ZETA_FUSED", False)
    monkeypatch.setattr(vt, "_TOMNSPA_ZETA_FUSED", False)

    out = tomnsps_rzl(**args, mpol=2, ntor=0, nfp=1, lasym=True, trig=trig)
    asym = tomnspa_rzl(**args, mpol=2, ntor=0, nfp=1, lasym=True, trig=trig)

    assert out.frss is None
    assert out.fzcs is None
    assert out.flcs is None
    assert asym.frcs is None
    assert asym.fzss is None
    assert asym.flss is None

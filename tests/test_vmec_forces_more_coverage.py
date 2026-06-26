from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.kernels.forces as vf
import vmec_jax.kernels.lforbal as vl
from vmec_jax.kernels.forces import VmecRZForceKernels, vmec_residual_internal_from_kernels
from vmec_jax.kernels.tomnsp import TomnspsRZL, vmec_trig_tables


def _trig():
    return vmec_trig_tables(ntheta=4, nzeta=2, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)


def _kernels(shape: tuple[int, int, int]) -> VmecRZForceKernels:
    base = np.arange(np.prod(shape), dtype=float).reshape(shape)
    zeros = np.zeros(shape, dtype=float)
    return VmecRZForceKernels(
        armn_e=base + 1.0,
        armn_o=base + 2.0,
        brmn_e=zeros,
        brmn_o=zeros,
        crmn_e=zeros,
        crmn_o=zeros,
        azmn_e=base + 3.0,
        azmn_o=base + 4.0,
        bzmn_e=base + 5.0,
        bzmn_o=base + 6.0,
        czmn_e=zeros,
        czmn_o=zeros,
        bc=SimpleNamespace(),
        arcon_e=zeros,
        arcon_o=zeros,
        azcon_e=zeros,
        azcon_o=zeros,
        gcon=zeros,
        pr1_even=zeros,
        pr1_odd=zeros,
        pz1_even=zeros,
        pz1_odd=zeros,
        pru_even=zeros + 0.25,
        pru_odd=zeros + 0.5,
        pzu_even=zeros + 0.75,
        pzu_odd=zeros + 1.0,
        prv_even=zeros,
        prv_odd=zeros,
        pzv_even=zeros,
        pzv_odd=zeros,
    )


def _tomnsps_output(ns: int = 3) -> TomnspsRZL:
    block = np.arange(ns * 2, dtype=float).reshape(ns, 2, 1)
    return TomnspsRZL(
        frcc=block,
        frss=-block,
        fzsc=100.0 + block,
        fzcs=200.0 + block,
        flsc=300.0 + block,
        flcs=400.0 + block,
    )


def test_residual_internal_scan_debug_force_prints_kernel_norms(monkeypatch) -> None:
    jax = pytest.importorskip("jax")
    trig = _trig()
    kernels = _kernels((3, int(trig.ntheta3), int(trig.cosnv.shape[0])))
    wout = SimpleNamespace(nfp=1, mpol=2, ntor=0, lasym=False)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_print(message: str, **kwargs) -> None:
        calls.append((message, kwargs))

    monkeypatch.setenv("VMEC_JAX_SCAN_DEBUG_FORCE", "1")
    monkeypatch.setattr(jax.debug, "print", fake_print)
    monkeypatch.setattr(vf, "tomnsps_rzl", lambda **_: _tomnsps_output())

    out = vmec_residual_internal_from_kernels(
        kernels,
        cfg_ntheta=4,
        cfg_nzeta=2,
        wout=wout,
        trig=trig,
    )

    assert out.frcc.shape == (3, 2, 1)
    assert calls
    message, kwargs = calls[0]
    assert message.startswith("[tomnsps-debug]")
    np.testing.assert_allclose(np.asarray(kwargs["aze"]), np.sum(kernels.azmn_e * kernels.azmn_e))
    np.testing.assert_allclose(np.asarray(kwargs["bze"]), np.sum(kernels.bzmn_e * kernels.bzmn_e))
    np.testing.assert_allclose(np.asarray(kwargs["azo"]), np.sum(kernels.azmn_o * kernels.azmn_o))
    np.testing.assert_allclose(np.asarray(kwargs["bzo"]), np.sum(kernels.bzmn_o * kernels.bzmn_o))


def test_residual_internal_apply_lforbal_uses_state_factors_and_replaces_rz_blocks(monkeypatch) -> None:
    trig = _trig()
    kernels = _kernels((3, int(trig.ntheta3), int(trig.cosnv.shape[0])))
    wout = SimpleNamespace(nfp=1, mpol=2, ntor=0, lasym=False)
    factor_sentinel = object()
    factor_calls: list[dict[str, object]] = []
    apply_calls: list[dict[str, object]] = []

    def fake_factors_from_state(**kwargs):
        factor_calls.append(kwargs)
        return factor_sentinel

    def fake_apply_lforbal_to_tomnsps(**kwargs):
        apply_calls.append(kwargs)
        return kwargs["frcc"] + 10.0, kwargs["fzsc"] - 20.0

    monkeypatch.setattr(vf, "tomnsps_rzl", lambda **_: _tomnsps_output())
    monkeypatch.setattr(vl, "lforbal_factors_from_state", fake_factors_from_state)
    monkeypatch.setattr(vl, "apply_lforbal_to_tomnsps", fake_apply_lforbal_to_tomnsps)

    out = vmec_residual_internal_from_kernels(
        kernels,
        cfg_ntheta=4,
        cfg_nzeta=2,
        wout=wout,
        trig=trig,
        apply_lforbal=True,
    )

    original = _tomnsps_output()
    np.testing.assert_allclose(np.asarray(out.frcc), np.asarray(original.frcc) + 10.0)
    np.testing.assert_allclose(np.asarray(out.fzsc), np.asarray(original.fzsc) - 20.0)
    np.testing.assert_allclose(np.asarray(out.frss), np.asarray(original.frss))
    np.testing.assert_allclose(np.asarray(out.fzcs), np.asarray(original.fzcs))
    np.testing.assert_allclose(np.asarray(out.flsc), np.asarray(original.flsc))
    np.testing.assert_allclose(np.asarray(out.flcs), np.asarray(original.flcs))

    assert factor_calls
    factor_kwargs = factor_calls[0]
    assert factor_kwargs["bc"] is kernels.bc
    assert factor_kwargs["trig"] is trig
    assert factor_kwargs["wout"] is wout
    np.testing.assert_allclose(np.asarray(factor_kwargs["s"]), [0.0, 0.5, 1.0])
    assert factor_kwargs["pru_even"] is kernels.pru_even
    assert factor_kwargs["pru_odd"] is kernels.pru_odd
    assert factor_kwargs["pzu_even"] is kernels.pzu_even
    assert factor_kwargs["pzu_odd"] is kernels.pzu_odd
    assert factor_kwargs["pr1_odd"] is kernels.pr1_odd
    assert factor_kwargs["pz1_odd"] is kernels.pz1_odd

    assert apply_calls
    apply_kwargs = apply_calls[0]
    np.testing.assert_allclose(np.asarray(apply_kwargs["frcc"]), np.asarray(original.frcc))
    np.testing.assert_allclose(np.asarray(apply_kwargs["fzsc"]), np.asarray(original.fzsc))
    assert apply_kwargs["factors"] is factor_sentinel
    assert apply_kwargs["trig"] is trig

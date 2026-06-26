from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.solve as solve_module
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.solve import (
    _enforce_field_rows,
    _enforce_fixed_boundary_and_axis,
    _preconditioner_apply_payload_fused,
    _preconditioner_output_payload_jit,
    _preconditioner_output_scaling_jit,
    _replace_mode_slice,
    _scale_mode_slice,
    _zero_coeff_column,
)
from vmec_jax.solvers.fixed_boundary.residual.ptau import state_tau_minmax_from_vmec_state
from vmec_jax.state import StateLayout, VMECState, zeros_state
from vmec_jax.static import build_static
from vmec_jax.kernels.tomnsp import TomnspsRZL, vmec_angle_grid


def test_zero_coeff_column_matches_masking():
    arr = np.arange(12, dtype=float).reshape(3, 4)
    idx = 2
    got = np.asarray(_zero_coeff_column(arr, idx=idx))
    want = np.asarray(arr).copy()
    want[:, idx] = 0.0
    np.testing.assert_allclose(got, want)


def test_replace_and_scale_mode_slice_match_reference():
    arr = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    repl = np.full((2, 3), 7.0)
    got_repl = np.asarray(_replace_mode_slice(arr, mode_idx=1, replacement=repl))
    want_repl = np.asarray(arr).copy()
    want_repl[:, 1, :] = repl
    np.testing.assert_allclose(got_repl, want_repl)

    scale = np.asarray([2.0, 3.0], dtype=float)
    got_scale = np.asarray(_scale_mode_slice(arr, mode_idx=1, scale=scale))
    want_scale = np.asarray(arr).copy()
    want_scale[:, 1, :] *= scale[:, None]
    np.testing.assert_allclose(got_scale, want_scale)


def test_enforce_field_rows_matches_legacy_axis_and_edge():
    arr = np.arange(5 * 4, dtype=float).reshape(5, 4)
    mask = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float)
    edge = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=float)

    got = np.asarray(_enforce_field_rows(arr, axis_mask=mask, edge_row=edge))

    want = np.concatenate([arr[:-1, :], edge[None, :]], axis=0)
    want = np.concatenate([want[:1, :] * mask[None, :], want[1:, :]], axis=0)
    np.testing.assert_allclose(got, want)


def test_enforce_field_rows_matches_legacy_single_row():
    arr = np.arange(4, dtype=float).reshape(1, 4)
    mask = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float)
    edge = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=float)

    got = np.asarray(_enforce_field_rows(arr, axis_mask=mask, edge_row=edge))

    want = np.concatenate([arr[:-1, :], edge[None, :]], axis=0)
    want = np.concatenate([want[:1, :] * mask[None, :], want[1:, :]], axis=0)
    np.testing.assert_allclose(got, want)


def test_enforce_fixed_boundary_and_axis_matches_component_reference():
    root = Path(__file__).resolve().parents[1]
    cfg, _ = load_config(str(root / "examples/data/input.circular_tokamak"))
    static = build_static(cfg)
    layout = StateLayout(ns=cfg.ns, K=static.modes.m.size, lasym=cfg.lasym)
    state0 = zeros_state(layout)
    rng = np.random.default_rng(0)
    state = VMECState(
        layout=layout,
        Rcos=rng.standard_normal(state0.Rcos.shape),
        Rsin=rng.standard_normal(state0.Rsin.shape),
        Zcos=rng.standard_normal(state0.Zcos.shape),
        Zsin=rng.standard_normal(state0.Zsin.shape),
        Lcos=rng.standard_normal(state0.Lcos.shape),
        Lsin=rng.standard_normal(state0.Lsin.shape),
    )
    idx00 = int(np.where((np.asarray(static.modes.m) == 0) & (np.asarray(static.modes.n) == 0))[0][0])
    edge_Rcos = rng.standard_normal((layout.K,))
    edge_Rsin = rng.standard_normal((layout.K,))
    edge_Zcos = rng.standard_normal((layout.K,))
    edge_Zsin = rng.standard_normal((layout.K,))

    got = _enforce_fixed_boundary_and_axis(
        state,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_axis=True,
        enforce_edge=True,
        enforce_lambda_axis=True,
        idx00=idx00,
    )

    mask_m0 = np.asarray(static.m_is_m0, dtype=float)
    Rcos = np.asarray(state.Rcos)
    Rsin = np.asarray(state.Rsin)
    Zcos = np.asarray(state.Zcos)
    Zsin = np.asarray(state.Zsin)
    Lcos = np.asarray(state.Lcos)
    Lsin = np.asarray(state.Lsin)

    Rcos = np.concatenate([Rcos[:-1, :], edge_Rcos[None, :]], axis=0)
    Rsin = np.concatenate([Rsin[:-1, :], edge_Rsin[None, :]], axis=0)
    Zcos = np.concatenate([Zcos[:-1, :], edge_Zcos[None, :]], axis=0)
    Zsin = np.concatenate([Zsin[:-1, :], edge_Zsin[None, :]], axis=0)
    Rcos = np.concatenate([Rcos[:1, :] * mask_m0[None, :], Rcos[1:, :]], axis=0)
    Rsin = np.concatenate([Rsin[:1, :] * mask_m0[None, :], Rsin[1:, :]], axis=0)
    Zcos = np.concatenate([Zcos[:1, :] * mask_m0[None, :], Zcos[1:, :]], axis=0)
    Zsin = np.concatenate([Zsin[:1, :] * mask_m0[None, :], Zsin[1:, :]], axis=0)
    Lcos = np.concatenate([np.zeros_like(Lcos[:1, :]), Lcos[1:, :]], axis=0)
    Lsin = np.concatenate([np.zeros_like(Lsin[:1, :]), Lsin[1:, :]], axis=0)
    Lcos[:, idx00] = 0.0
    Lsin[:, idx00] = 0.0

    np.testing.assert_allclose(np.asarray(got.Rcos), Rcos)
    np.testing.assert_allclose(np.asarray(got.Rsin), Rsin)
    np.testing.assert_allclose(np.asarray(got.Zcos), Zcos)
    np.testing.assert_allclose(np.asarray(got.Zsin), Zsin)
    np.testing.assert_allclose(np.asarray(got.Lcos), Lcos)
    np.testing.assert_allclose(np.asarray(got.Lsin), Lsin)


def test_preconditioner_output_scaling_jit_matches_reference():
    pytest.importorskip("jax")

    shape = (4, 3, 2)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
    frzl_rz = TomnspsRZL(
        frcc=base,
        frss=None,
        fzsc=base + 2.0,
        fzcs=base + 3.0,
        flsc=base + 4.0,
        flcs=None,
        frsc=base + 5.0,
        frcs=None,
        fzcc=None,
        fzss=base + 6.0,
        flcc=base + 7.0,
        flss=None,
    )
    lam_prec = np.linspace(0.5, 1.25, shape[0])[:, None, None]
    w_mode_mn = np.array([[1.0, 0.5], [0.25, 0.125], [0.1, 0.05]])
    lambda_scale = np.asarray(1.75)

    scale_outputs = _preconditioner_output_scaling_jit(apply_lambda_update_scale=True)
    pre, upd = scale_outputs(frzl_rz, lam_prec, w_mode_mn, lambda_scale)
    pre = tuple(None if x is None else np.asarray(x) for x in pre)
    upd = tuple(np.asarray(x) for x in upd)

    frcc = np.asarray(frzl_rz.frcc)
    frss = frzl_rz.frss
    fzsc = np.asarray(frzl_rz.fzsc)
    fzcs = np.asarray(frzl_rz.fzcs)
    flsc = np.asarray(frzl_rz.flsc) * lam_prec
    flcs = None
    frsc = np.asarray(frzl_rz.frsc)
    frcs = np.zeros_like(frcc)
    fzcc = np.zeros_like(fzsc)
    fzss = np.asarray(frzl_rz.fzss)
    flcc = np.asarray(frzl_rz.flcc) * lam_prec
    flss = np.zeros_like(flsc)
    w = w_mode_mn[None, :, :]

    want_pre = (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss)
    want_upd = (
        frcc * w,
        np.zeros_like(frcc) * w,
        fzsc * w,
        fzcs * w,
        flsc * w * lambda_scale,
        np.zeros_like(flsc) * w * lambda_scale,
        frsc * w,
        frcs * w,
        fzcc * w,
        fzss * w,
        flcc * w * lambda_scale,
        flss * w * lambda_scale,
    )

    for got_arr, want_arr in zip(pre, want_pre):
        if want_arr is None:
            assert got_arr is None
        else:
            np.testing.assert_allclose(got_arr, want_arr)
    for got_arr, want_arr in zip(upd, want_upd):
        np.testing.assert_allclose(got_arr, want_arr)


def test_preconditioner_output_payload_jit_matches_scaling_and_fsq1_reference():
    pytest.importorskip("jax")

    shape = (3, 2, 2)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10.0 + 0.5
    frzl_rz = TomnspsRZL(
        frcc=base,
        frss=base + 1.0,
        fzsc=base + 2.0,
        fzcs=base + 3.0,
        flsc=base + 4.0,
        flcs=base + 5.0,
        frsc=base + 6.0,
        frcs=base + 7.0,
        fzcc=base + 8.0,
        fzss=base + 9.0,
        flcc=base + 10.0,
        flss=base + 11.0,
    )
    lam_prec = np.linspace(0.25, 0.75, shape[0])[:, None, None]
    w_mode_mn = np.array([[1.0, 0.5], [0.25, 0.125]])
    lambda_scale = np.asarray(3.0)
    f_norm1 = np.asarray(2.0)
    delta_s = np.asarray(0.125)
    s = np.linspace(0.0, 1.0, shape[0])

    payload = _preconditioner_output_payload_jit(
        apply_lambda_update_scale=True,
        vmec2000_control=True,
        lconm1=True,
    )
    pre, upd, diag = payload(frzl_rz, lam_prec, w_mode_mn, lambda_scale, f_norm1, delta_s, s)

    scaler = _preconditioner_output_scaling_jit(apply_lambda_update_scale=True)
    pre_ref, upd_ref = scaler(frzl_rz, lam_prec, w_mode_mn, lambda_scale)
    for got, want in zip(pre, pre_ref, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want))
    for got, want in zip(upd, upd_ref, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want))

    gcr2 = np.sum(np.asarray(pre_ref[0]) ** 2) + np.sum(np.asarray(pre_ref[1]) ** 2)
    gcr2 = gcr2 + np.sum(np.asarray(pre_ref[6]) ** 2) + np.sum(np.asarray(pre_ref[7]) ** 2)
    gcz2 = np.sum(np.asarray(pre_ref[2]) ** 2) + np.sum(np.asarray(pre_ref[3]) ** 2)
    gcz2 = gcz2 + np.sum(np.asarray(pre_ref[8]) ** 2) + np.sum(np.asarray(pre_ref[9]) ** 2)
    gcl2_full = (
        np.sum(np.asarray(pre_ref[4])[1:] ** 2)
        + np.sum(np.asarray(pre_ref[5])[1:] ** 2)
        + np.sum(np.asarray(pre_ref[10])[1:] ** 2)
        + np.sum(np.asarray(pre_ref[11])[1:] ** 2)
    )
    np.testing.assert_allclose(np.asarray(diag[3]), gcr2 * f_norm1)
    np.testing.assert_allclose(np.asarray(diag[4]), gcz2 * f_norm1)
    np.testing.assert_allclose(np.asarray(diag[5]), gcl2_full * delta_s)


def test_preconditioner_apply_payload_fused_can_return_ptau_control_payload():
    pytest.importorskip("jax")

    shape = (3, 2, 1)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10.0 + 1.0
    frzl = TomnspsRZL(
        frcc=base,
        frss=None,
        fzsc=base + 0.25,
        fzcs=None,
        flsc=base + 0.5,
        flcs=None,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    )
    mats = {
        "ar": np.zeros(shape),
        "br": np.zeros(shape),
        "dr": np.ones(shape),
        "az": np.zeros(shape),
        "bz": np.zeros(shape),
        "dz": np.ones(shape),
    }
    pbase = np.arange(np.prod(shape), dtype=float).reshape(shape) / 7.0 + 0.3
    ptau_arrays = tuple(pbase + offset for offset in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
    pshalf = np.asarray([0.0, 0.5, 1.0])
    ohs = np.asarray(2.0)

    pre, upd, diag, control = _preconditioner_apply_payload_fused(
        frzl_in=frzl,
        mats=mats,
        jmax=shape[0],
        cfg=SimpleNamespace(lthreed=False, lasym=False),
        lam_prec=np.ones(shape),
        w_mode_mn=np.ones(shape[1:]),
        lambda_update_scale_j=np.asarray(1.0),
        f_norm1=np.asarray(1.5),
        delta_s=np.asarray(0.25),
        s=np.linspace(0.0, 1.0, shape[0]),
        use_precomputed=False,
        use_lax_tridi=False,
        apply_lambda_update_scale=False,
        vmec2000_control=True,
        lconm1=True,
        include_control_ptau=True,
        control_ptau_arrays=ptau_arrays,
        control_ptau_pshalf=pshalf,
        control_ptau_ohs=ohs,
    )

    expected_ptau = solve_module._ptau_compute_jit(*ptau_arrays, pshalf, ohs)
    np.testing.assert_allclose(np.asarray(control[0]), np.asarray(diag[6]))
    np.testing.assert_allclose(np.asarray(control[1]), np.asarray(expected_ptau[0]))
    np.testing.assert_allclose(np.asarray(control[2]), np.asarray(expected_ptau[1]))
    assert pre[0].shape == shape
    assert upd[0].shape == shape


def test_state_tau_minmax_from_vmec_state_uses_host_patch_path():
    calls: list[str] = []

    class PatchContext:
        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")

    def jacobian_from_state(**kwargs):
        calls.append("jac")
        assert kwargs["lconm1"] is True
        return SimpleNamespace(tau=np.asarray([100.0, -2.0, 3.0]))

    min_tau, max_tau = state_tau_minmax_from_vmec_state(
        state=object(),
        modes=object(),
        trig=object(),
        s=object(),
        lconm1=True,
        lthreed=True,
        mask_even=None,
        mask_odd=None,
        host_update_assembly=True,
        tree_has_tracer=lambda _value: False,
        jacobian_from_state=jacobian_from_state,
        device_get_floats=lambda *args: pytest.fail("host path should not device-get"),
        jnp_module=np,
        numpy_patch_context=PatchContext,
    )

    assert calls == ["enter", "jac", "exit"]
    assert min_tau == pytest.approx(-2.0)
    assert max_tau == pytest.approx(3.0)


def test_state_tau_minmax_from_vmec_state_uses_device_get_path():
    def jacobian_from_state(**_kwargs):
        return SimpleNamespace(tau=np.asarray([100.0, -4.0, 5.0]))

    def device_get_floats(*args):
        return tuple(float(np.asarray(arg)) for arg in args)

    min_tau, max_tau = state_tau_minmax_from_vmec_state(
        state=object(),
        modes=object(),
        trig=object(),
        s=object(),
        lconm1=True,
        lthreed=True,
        mask_even=None,
        mask_odd=None,
        host_update_assembly=True,
        tree_has_tracer=lambda _value: True,
        jacobian_from_state=jacobian_from_state,
        device_get_floats=device_get_floats,
        jnp_module=np,
        numpy_patch_context=lambda: pytest.fail("device path should not patch numpy"),
    )

    assert min_tau == pytest.approx(-4.0)
    assert max_tau == pytest.approx(5.0)


def test_preconditioner_output_scaling_gate_is_gpu_only_without_gpu(monkeypatch):
    pytest.importorskip("jax")

    root = Path(__file__).resolve().parents[1]
    cfg, indata = load_config(str(root / "examples/data/input.circular_tokamak"))
    grid = vmec_angle_grid(ntheta=8, nzeta=4, nfp=cfg.nfp, lasym=cfg.lasym)
    static = build_static(cfg, grid=grid)
    boundary = boundary_from_indata(indata, static.modes)
    state0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=False)
    solve_kwargs = dict(
        indata=indata,
        signgs=1,
        ftol=float(indata.get_float("FTOL", 1.0e-13)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 5e-3)),
        include_constraint_force=True,
        apply_m1_constraints=True,
        precond_radial_alpha=0.5,
        precond_lambda_alpha=0.5,
        mode_diag_exponent=0.0,
        auto_flip_force=False,
        divide_by_scalxc_for_update=False,
        lambda_update_scale=1.75,
        enforce_vmec_lambda_axis=True,
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        reference_mode=False,
        use_restart_triggers=True,
        vmecpp_restart=False,
        use_direct_fallback=False,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces=False,
        use_scan=False,
        host_update_assembly=False,
    )

    def fail_if_fused(*, apply_lambda_update_scale):
        raise AssertionError("CPU backend must not use fused preconditioner output scaling")

    monkeypatch.setattr(solve_module.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(solve_module, "_preconditioner_output_scaling_jit", fail_if_fused)
    cpu_res = solve_module.solve_fixed_boundary_residual_iter(state0, static, **solve_kwargs)

    calls = []
    original_apply = _preconditioner_apply_payload_fused

    def count_fused_apply(*args, **kwargs):
        calls.append(bool(kwargs["apply_lambda_update_scale"]))
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(solve_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(solve_module, "_preconditioner_apply_payload_fused", count_fused_apply)
    gpu_res = solve_module.solve_fixed_boundary_residual_iter(state0, static, **solve_kwargs)

    assert calls == [True]
    np.testing.assert_allclose(np.asarray(gpu_res.w_history), np.asarray(cpu_res.w_history), rtol=1e-12, atol=1e-12)
    for field in ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin"):
        np.testing.assert_allclose(
            np.asarray(getattr(gpu_res.state, field)),
            np.asarray(getattr(cpu_res.state, field)),
            rtol=1e-11,
            atol=1e-11,
        )

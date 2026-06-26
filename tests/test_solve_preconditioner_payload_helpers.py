from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.preconditioning.operators import PreconditionerCacheState
from vmec_jax.solvers.fixed_boundary.residual import preconditioner_payload as payload_mod
from vmec_jax.solvers.fixed_boundary.residual.preconditioner_payload import (
    apply_vmec2000_preconditioner_runtime,
    host_preconditioned_residual_scalar_channels,
    jax_preconditioned_residual_scalar_channels,
    materialize_accepted_control_payload,
    refresh_preconditioner_cache_state_runtime,
    seed_preconditioner_cache_from_bcovar_update,
)
from vmec_jax.kernels.tomnsp import TomnspsRZL


class _FakeJax:
    calls = []

    @classmethod
    def device_get(cls, value):
        cls.calls.append(value)
        return value


def _tiny_frzl(value: float = 1.0) -> TomnspsRZL:
    arr = np.full((2, 1, 1), value, dtype=float)
    return TomnspsRZL(
        frcc=arr,
        frss=None,
        fzsc=2.0 * arr,
        fzcs=None,
        flsc=3.0 * arr,
        flcs=None,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    )


def _apply_runtime_kwargs(**overrides):
    frzl = _tiny_frzl()
    kwargs = dict(
        frzl=frzl,
        k="kernel",
        state="state",
        iter2=2,
        cfg=type("Cfg", (), {"lthreed": True, "lasym": False})(),
        s=np.asarray([0.0, 1.0]),
        delta_s=np.asarray(0.5),
        w_mode_mn=np.asarray([1.0]),
        lambda_update_scale=1.0,
        lambda_update_scale_j=np.asarray(1.0),
        lconm1=True,
        vmec2000_control=True,
        vmec2000_cache_valid=True,
        need_bcovar_update=False,
        cache_rz_norm=2.0,
        cache_f_norm1=0.5,
        host_update_assembly=False,
        use_fused_precond_output_scaling=False,
        scale_m1_rhs=True,
        adjoint_trace=False,
        adjoint_trace_mode="minimal",
        accepted_control_ptau_arrays=None,
        ptau_pshalf_jax=None,
        ptau_ohs_jax=None,
        preconditioner_use_precomputed_tridi=True,
        preconditioner_use_lax_tridi=False,
        timing_detail_enabled=False,
        timing_stats={"precond_apply": 0.0},
        perf_counter=iter([1.0, 1.25]).__next__,
        block_until_ready=lambda _x: None,
        refresh_preconditioner_cache_func=lambda _k, *, iter2: (
            np.asarray([2.0]),
            {"mats": True},
            1,
            True,
            False,
            True,
        ),
        scale_m1_precond_rhs_func=lambda frzl_in, _mats: frzl_in,
        rz_preconditioner_apply_func=lambda *, frzl_in, **_kwargs: frzl_in,
        rz_norm_func=lambda _state: np.asarray(2.0),
    )
    kwargs.update(overrides)
    return kwargs


def _host_from_payload(payload, *, device_get_floats):
    if payload == "unused":
        return 0.0, None, False
    fsq1, min_tau, max_tau = device_get_floats(*payload)
    return fsq1, (min_tau, max_tau), True


def test_host_preconditioned_residual_scalar_channels_uses_vmec2000_full_lambda_norm() -> None:
    out = host_preconditioned_residual_scalar_channels(
        gcr2_p=4.0,
        gcz2_p=8.0,
        gcl2_p=99.0,
        frzl_pre="frzl",
        frzl_pre_host="frzl-host",
        vmec2000_control=True,
        vmec2000_cache_valid=False,
        need_bcovar_update=True,
        cache_rz_norm=None,
        cache_f_norm1=None,
        state="state",
        delta_s=0.25,
        numpy_module=__import__("numpy"),
        rz_norm_np=lambda _state: 2.0,
        lambda_preconditioned_full_norm=lambda frzl, *, use_jax: 5.0 if frzl == "frzl-host" and not use_jax else 0.0,
        finite_float_or_zero=lambda value: 0.0 if value != value else float(value),
    )

    assert out.rz_norm == 2.0
    assert out.f_norm1 == 0.5
    assert out.fsqr1 == 2.0
    assert out.fsqz1 == 4.0
    assert out.fsql1 == 1.25
    assert out.fsq1 == 7.25


def test_jax_preconditioned_residual_scalar_channels_uses_cached_norm_and_safe_sum() -> None:
    import numpy as np

    out = jax_preconditioned_residual_scalar_channels(
        gcr2_p=np.asarray(4.0),
        gcz2_p=np.asarray(8.0),
        gcl2_p=np.asarray(9.0),
        frzl_pre="frzl",
        vmec2000_control=False,
        vmec2000_cache_valid=True,
        need_bcovar_update=False,
        cache_rz_norm=4.0,
        cache_f_norm1=0.25,
        state="state",
        delta_s=np.asarray(0.5),
        jnp_module=np,
        cached_or_current_f_norm1_jax=lambda **_kwargs: (np.asarray(4.0), np.asarray(0.25)),
        rz_norm_func=lambda _state: np.asarray(-1.0),
        lambda_preconditioned_full_norm=lambda *_args, **_kwargs: np.asarray(-1.0),
    )

    assert float(out.rz_norm) == 4.0
    assert float(out.f_norm1) == 0.25
    assert float(out.fsqr1_safe) == 1.0
    assert float(out.fsqz1_safe) == 2.0
    assert float(out.fsql1_safe) == 4.5
    assert float(out.fsq1) == 7.5


def test_residual_preconditioner_operators_use_numpy_lambda_for_host_path(monkeypatch) -> None:
    import vmec_jax.preconditioner_1d as p1d
    import vmec_jax.preconditioner_1d_jax as p1d_jax

    calls = []

    def fake_lambda_preconditioner(**kwargs):
        calls.append(kwargs)
        return ("numpy-lambda", kwargs["return_faclam"], kwargs["return_debug"])

    monkeypatch.setattr(p1d, "lambda_preconditioner", fake_lambda_preconditioner)
    monkeypatch.setattr(
        p1d_jax,
        "lambda_preconditioner_cached",
        lambda **_kwargs: pytest.fail("host path should not call JAX lambda preconditioner"),
    )

    ops = payload_mod.residual_preconditioner_operators(
        trig=SimpleNamespace(r0scale=1.25),
        s=np.linspace(0.0, 1.0, 3),
        cfg=SimpleNamespace(mpol=2, ntor=1, ntheta=4, nzeta=2, nfp=1, lasym=False, lthreed=True),
        use_numpy_preconditioner_apply=True,
        tree_has_tracer_func=lambda _value: False,
        radial_tridi_smooth_dirichlet_func=lambda a, **_kwargs: a,
        jnp_module=np,
    )

    got = ops.lambda_preconditioner("bc", return_faclam=True, return_debug=True)

    assert got == ("numpy-lambda", True, True)
    assert calls[0]["bc"] == "bc"
    assert calls[0]["r0scale"] == pytest.approx(1.25)
    assert calls[0]["return_faclam"] is True
    assert calls[0]["return_debug"] is True


def test_residual_preconditioner_operators_use_numpy_rz_seed_for_3d_host_path(monkeypatch) -> None:
    calls = []

    def fake_numpy_host(**kwargs):
        calls.append(("numpy-host", kwargs))
        return "numpy-rz", "numpy-jmin", 4

    def fake_jax(**kwargs):
        calls.append(("jax", kwargs))
        return "jax-rz", "jax-jmin", 4

    import vmec_jax.preconditioner_1d_jax as p1d_jax

    monkeypatch.setattr(p1d_jax, "rz_preconditioner_matrices_numpy_host", fake_numpy_host)
    monkeypatch.setattr(p1d_jax, "rz_preconditioner_matrices", fake_jax)

    ops = payload_mod.residual_preconditioner_operators(
        trig="trig",
        s=np.linspace(0.0, 1.0, 3),
        cfg=SimpleNamespace(mpol=3, ntor=1, ntheta=6, nzeta=4, nfp=2, lasym=False, lthreed=True),
        use_numpy_preconditioner_apply=True,
        tree_has_tracer_func=lambda _value: False,
        radial_tridi_smooth_dirichlet_func=lambda a, **_kwargs: a,
        jnp_module=np,
    )

    got = ops.rz_preconditioner_matrices(
        bc="bc",
        k="k",
        jmax_override=3,
        use_precomputed=False,
        use_lax_tridi=False,
    )

    assert got == ("numpy-rz", "numpy-jmin", 4)
    assert calls[0][0] == "numpy-host"
    assert calls[0][1]["bc"] == "bc"
    assert calls[0][1]["k"] == "k"
    assert calls[0][1]["use_precomputed"] is False
    assert calls[0][1]["use_lax_tridi"] is False


def test_residual_preconditioner_operators_keep_jax_rz_seed_for_traced_path(monkeypatch) -> None:
    calls = []

    def fake_numpy_host(**kwargs):
        calls.append(("numpy-host", kwargs))
        return "numpy-rz", "numpy-jmin", 4

    def fake_jax(**kwargs):
        calls.append(("jax", kwargs))
        return "jax-rz", "jax-jmin", 4

    import vmec_jax.preconditioner_1d_jax as p1d_jax

    monkeypatch.setattr(p1d_jax, "rz_preconditioner_matrices_numpy_host", fake_numpy_host)
    monkeypatch.setattr(p1d_jax, "rz_preconditioner_matrices", fake_jax)

    ops = payload_mod.residual_preconditioner_operators(
        trig="trig",
        s=np.linspace(0.0, 1.0, 3),
        cfg=SimpleNamespace(mpol=3, ntor=1, ntheta=6, nzeta=4, nfp=2, lasym=False, lthreed=True),
        use_numpy_preconditioner_apply=True,
        tree_has_tracer_func=lambda _value: True,
        radial_tridi_smooth_dirichlet_func=lambda a, **_kwargs: a,
        jnp_module=np,
    )

    got = ops.rz_preconditioner_matrices(bc="bc", k="k")

    assert got == ("jax-rz", "jax-jmin", 4)
    assert calls[0][0] == "jax"


def test_seed_preconditioner_cache_from_bcovar_update_builds_vmec2000_seed() -> None:
    stats = {
        "precond_refresh_seed": 0.0,
        "precond_refresh": 0.0,
        "preconditioner": 0.0,
        "precond_refresh_calls": 0,
    }
    cache = PreconditionerCacheState()
    matrix_calls = []

    out = seed_preconditioner_cache_from_bcovar_update(
        cache=cache,
        k=SimpleNamespace(bc="bc", tcon=np.asarray([0.25])),
        state=SimpleNamespace(Rcos=np.asarray([1.0])),
        trig="trig",
        s=np.linspace(0.0, 1.0, 3),
        cfg=SimpleNamespace(lasym=False),
        norms_used="norms",
        rz_scale="rz-scale",
        l_scale="l-scale",
        constraint_tcon0=0.0,
        zero_tcon=np.asarray([0.0, 0.0, 0.0]),
        host_update_assembly=True,
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([10.0, 10.05, 10.15, 10.20, 10.35, 10.50]).__next__,
        tree_has_tracer=lambda _value: False,
        rz_norm_np=lambda _state: 4.0,
        rz_norm_func=lambda _state: pytest.fail("host path should use rz_norm_np"),
        lambda_preconditioner_func=lambda bc: np.asarray(2.0) if bc == "bc" else pytest.fail("wrong bc"),
        rz_preconditioner_matrices_func=lambda **kwargs: matrix_calls.append(kwargs) or ("mats", 0, 2),
        precond_jmax_override=None,
        preconditioner_use_precomputed_tridi=True,
        preconditioner_use_lax_tridi=False,
        jnp_module=np,
    )

    assert out.cache_update_trace is True
    assert out.seeded_from_bcovar_update is True
    assert out.seed_time_in_residual_metrics == pytest.approx(0.50)
    assert cache.valid is True
    assert cache.precond_diag is None
    np.testing.assert_allclose(cache.tcon, np.zeros(3))
    assert cache.norms == "norms"
    assert cache.rz_scale == "rz-scale"
    assert cache.l_scale == "l-scale"
    assert cache.rz_norm == 4.0
    assert cache.f_norm1 == 0.25
    np.testing.assert_allclose(cache.prec_lam_prec, 2.0)
    assert cache.prec_rz_mats == "mats"
    assert cache.prec_rz_jmax == 2
    assert matrix_calls[0]["use_precomputed"] is True
    assert matrix_calls[0]["use_lax_tridi"] is False
    assert stats["precond_refresh_seed"] == pytest.approx(0.50)
    assert stats["precond_refresh_seed_lambda"] == pytest.approx(0.10)
    assert stats["precond_refresh_seed_rz_matrices"] == pytest.approx(0.15)
    assert stats["precond_refresh_calls"] == 1


def test_seed_preconditioner_cache_from_bcovar_update_lasym_skips_1d_seed() -> None:
    cache = PreconditionerCacheState()

    out = seed_preconditioner_cache_from_bcovar_update(
        cache=cache,
        k=SimpleNamespace(bc="bc", tcon=np.asarray([0.25])),
        state=SimpleNamespace(Rcos=np.asarray([1.0])),
        trig="trig",
        s=np.linspace(0.0, 1.0, 3),
        cfg=SimpleNamespace(lasym=True),
        norms_used="norms",
        rz_scale="rz-scale",
        l_scale="l-scale",
        constraint_tcon0=0.0,
        zero_tcon=np.asarray([0.0, 0.0, 0.0]),
        host_update_assembly=False,
        timing_enabled=False,
        timing_stats={},
        perf_counter=lambda: 0.0,
        tree_has_tracer=lambda _value: False,
        rz_norm_np=lambda _state: pytest.fail("device path should use rz_norm_func"),
        rz_norm_func=lambda _state: np.asarray(5.0),
        lambda_preconditioner_func=lambda _bc: pytest.fail("lasym path should not seed 1D lambda preconditioner"),
        rz_preconditioner_matrices_func=lambda **_kwargs: pytest.fail("lasym path should not seed 1D R/Z matrices"),
        precond_jmax_override=None,
        preconditioner_use_precomputed_tridi=None,
        preconditioner_use_lax_tridi=None,
        jnp_module=np,
    )

    assert out.cache_update_trace is False
    assert out.seeded_from_bcovar_update is False
    assert out.seed_time_in_residual_metrics == 0.0
    assert cache.valid is True
    assert cache.prec_lam_prec is None
    assert cache.prec_rz_mats is None
    np.testing.assert_allclose(cache.f_norm1, 0.2)


def test_refresh_preconditioner_cache_state_runtime_updates_cache_fields() -> None:
    cache = PreconditionerCacheState()
    cache.valid = True
    cache.prec_lam_prec = "old-lam"
    cache.prec_faclam = "old-fac"
    cache.prec_lam_debug = "old-debug"
    cache.prec_rz_mats = "old-mats"
    cache.prec_rz_jmax = 2
    decision = SimpleNamespace(
        need_prec_refresh=False,
        can_reuse_bcovar_seeded_precond=False,
        need_prec_reassemble=False,
    )

    def update_preconditioner_cache_func(**kwargs):
        assert kwargs["vmec2000_cache_valid"] is True
        assert kwargs["cache_prec_lam_prec"] == "old-lam"
        return SimpleNamespace(
            decision=decision,
            lam_prec="new-lam",
            mats={"new": True},
            jmax=4,
            faclam_dump=None,
            lam_debug=None,
            cache_prec_lam_prec="cache-lam",
            cache_prec_faclam="cache-fac",
            cache_prec_lam_debug="cache-debug",
            cache_prec_rz_mats={"cache": True},
            cache_prec_rz_jmax=4,
        )

    out = refresh_preconditioner_cache_state_runtime(
        SimpleNamespace(bc="bc"),
        cache=cache,
        iter2=3,
        cfg=SimpleNamespace(),
        static=SimpleNamespace(),
        env_dump_lam="0",
        env_dump_lamcal="0",
        timing_enabled=False,
        timing_stats={},
        perf_counter=lambda: 0.0,
        block_until_ready=None,
        tree_has_tracer=lambda _value: False,
        update_preconditioner_cache_func=update_preconditioner_cache_func,
        can_reassemble_func=lambda **_kwargs: False,
        lambda_preconditioner_func=lambda **_kwargs: "unused",
        rz_preconditioner_matrices_func=lambda **_kwargs: "unused",
        maybe_dump_lam_prec=lambda **_kwargs: None,
        maybe_dump_precond_mats=lambda **_kwargs: None,
        maybe_dump_lamcal=lambda **_kwargs: None,
        need_bcovar_update=False,
        precond_cache_seeded_from_bcovar_update=False,
        precond_expected_jmax=4,
        precond_jmax_override=None,
        preconditioner_use_precomputed_tridi=True,
        preconditioner_use_lax_tridi=False,
    )

    assert out == ("new-lam", {"new": True}, 4, False, False, False)
    assert cache.prec_lam_prec == "cache-lam"
    assert cache.prec_faclam == "cache-fac"
    assert cache.prec_lam_debug == "cache-debug"
    assert cache.prec_rz_mats == {"cache": True}
    assert cache.prec_rz_jmax == 4


def test_materialize_accepted_control_payload_uses_existing_payload() -> None:
    stats = {"iteration_control_fsq1_payload_get": 0.0, "iteration_control_fsq1_direct_get": 0.0}

    out = materialize_accepted_control_payload(
        accepted_control_ptau_payload=(1.25, -0.5, 2.0),
        use_control_payload=True,
        fsq1_j=9.0,
        k="kernel",
        ptau_pshalf_jax="pshalf",
        ptau_ohs_jax="ohs",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([10.0, 10.25]).__next__,
        jax_module=_FakeJax,
        device_get_floats=lambda *vals: tuple(float(v) for v in vals),
        accepted_control_ptau_host_from_payload=_host_from_payload,
        scan_math_kernel_arrays_from_k=lambda _k: (_k,),
        accepted_control_payload_jit=lambda: None,
    )

    assert out.fsq1 == 1.25
    assert out.accepted_control_ptau_host == (-0.5, 2.0)
    assert out.control_payload_used is True
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.25)
    assert stats["iteration_control_fsq1_direct_get"] == 0.0


def test_materialize_accepted_control_payload_builds_payload_when_requested() -> None:
    stats = {"iteration_control_fsq1_payload_get": 0.0, "iteration_control_fsq1_direct_get": 0.0}
    payload_calls = []

    def payload_fn(fsq1_j, *args):
        payload_calls.append((fsq1_j, args))
        return (2.5, -1.0, 3.0)

    out = materialize_accepted_control_payload(
        accepted_control_ptau_payload=None,
        use_control_payload=True,
        fsq1_j=2.5,
        k="kernel",
        ptau_pshalf_jax="pshalf",
        ptau_ohs_jax="ohs",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([20.0, 20.5]).__next__,
        jax_module=_FakeJax,
        device_get_floats=lambda *vals: tuple(float(v) for v in vals),
        accepted_control_ptau_host_from_payload=_host_from_payload,
        scan_math_kernel_arrays_from_k=lambda _k: ("ptau",),
        accepted_control_payload_jit=lambda: payload_fn,
    )

    assert payload_calls == [(2.5, ("ptau", "pshalf", "ohs"))]
    assert out.fsq1 == 2.5
    assert out.accepted_control_ptau_host == (-1.0, 3.0)
    assert out.control_payload_used is True
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.5)
    assert stats["iteration_control_fsq1_direct_get"] == 0.0


def test_materialize_accepted_control_payload_falls_back_to_direct_device_get() -> None:
    _FakeJax.calls = []
    stats = {"iteration_control_fsq1_payload_get": 0.0, "iteration_control_fsq1_direct_get": 0.0}

    out = materialize_accepted_control_payload(
        accepted_control_ptau_payload="unused",
        use_control_payload=True,
        fsq1_j=7.0,
        k="kernel",
        ptau_pshalf_jax="pshalf",
        ptau_ohs_jax="ohs",
        timing_enabled=True,
        timing_stats=stats,
        perf_counter=iter([30.0, 30.1, 40.0, 40.2]).__next__,
        jax_module=_FakeJax,
        device_get_floats=lambda *vals: tuple(float(v) for v in vals),
        accepted_control_ptau_host_from_payload=_host_from_payload,
        scan_math_kernel_arrays_from_k=lambda _k: None,
        accepted_control_payload_jit=lambda: None,
    )

    assert out.fsq1 == 7.0
    assert out.accepted_control_ptau_host is None
    assert out.control_payload_used is False
    assert _FakeJax.calls == [7.0]
    assert stats["iteration_control_fsq1_payload_get"] == pytest.approx(0.1)
    assert stats["iteration_control_fsq1_direct_get"] == pytest.approx(0.2)


def test_apply_vmec2000_preconditioner_runtime_plain_path_keeps_raw_apply() -> None:
    out = apply_vmec2000_preconditioner_runtime(**_apply_runtime_kwargs())

    assert out.cache_update_trace is True
    assert out.frzl_lam_pre is not None
    assert out.update_blocks is None
    assert out.outputs_scaled is False
    assert out.fsq1_ready is False
    np.testing.assert_allclose(np.asarray(out.blocks.flsc), 6.0 * np.ones((2, 1, 1)))


def test_apply_vmec2000_preconditioner_runtime_output_payload_path(monkeypatch: pytest.MonkeyPatch) -> None:
    pre_blocks = tuple(np.asarray(float(i)) for i in range(12))
    update_blocks = tuple(np.asarray(10.0 + float(i)) for i in range(12))
    diag = tuple(np.asarray(v) for v in (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.6))

    monkeypatch.setattr(
        payload_mod,
        "_preconditioner_output_payload_jit",
        lambda **_kwargs: lambda *_args: (pre_blocks, update_blocks, diag),
    )
    out = apply_vmec2000_preconditioner_runtime(
        **_apply_runtime_kwargs(use_fused_precond_output_scaling=True)
    )

    assert out.outputs_scaled is True
    assert out.fsq1_ready is True
    assert out.gcr2_p == pytest.approx(1.0)
    assert out.fsq1_safe == pytest.approx(0.6)
    assert out.blocks.frcc == pytest.approx(0.0)
    assert out.update_blocks.flss == pytest.approx(21.0)


def test_apply_vmec2000_preconditioner_runtime_fused_full_trace_materializes_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_calls = []
    pre_blocks = tuple(np.asarray(float(i)) for i in range(12))
    update_blocks = tuple(np.asarray(20.0 + float(i)) for i in range(12))
    diag = tuple(np.asarray(v) for v in (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.6))

    monkeypatch.setattr(
        payload_mod,
        "_preconditioner_apply_payload_fused",
        lambda **_kwargs: (pre_blocks, update_blocks, diag, ("ptau",)),
    )

    def rz_apply(*, frzl_in, **kwargs):
        apply_calls.append(kwargs)
        return frzl_in

    out = apply_vmec2000_preconditioner_runtime(
        **_apply_runtime_kwargs(
            use_fused_precond_output_scaling=True,
            adjoint_trace=True,
            adjoint_trace_mode="full",
            refresh_preconditioner_cache_func=lambda _k, *, iter2: (
                np.asarray([2.0]),
                {"mats": True},
                1,
                False,
                False,
                False,
            ),
            rz_preconditioner_apply_func=rz_apply,
        )
    )

    assert out.outputs_scaled is True
    assert out.accepted_control_ptau_payload == ("ptau",)
    assert out.frzl_rz is not None
    assert out.frzl_lam_pre is None
    assert len(apply_calls) == 1

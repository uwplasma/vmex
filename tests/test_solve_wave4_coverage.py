from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax import solve as solve_mod
from vmec_jax._compat import has_jax, jnp
from vmec_jax.config import VMECConfig
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.static import build_static
from vmec_jax.kernels.tomnsp import TomnspsRZL


pytestmark = pytest.mark.skipif(not has_jax(), reason="solve helpers require JAX array updates")


def _small_static(*, ns: int = 5):
    cfg = VMECConfig(
        ns=ns,
        mpol=3,
        ntor=1,
        nfp=2,
        lasym=False,
        lthreed=True,
        lconm1=True,
        ntheta=8,
        nzeta=4,
    )
    return build_static(cfg)


def _state(static, *, start: float = 1.0) -> VMECState:
    layout = StateLayout(ns=int(static.cfg.ns), K=int(static.modes.K), lasym=bool(static.cfg.lasym))
    base = np.arange(layout.ns * layout.K, dtype=float).reshape(layout.ns, layout.K) + start
    return VMECState(
        layout=layout,
        Rcos=base,
        Rsin=base + 100.0,
        Zcos=base + 200.0,
        Zsin=base + 300.0,
        Lcos=base + 400.0,
        Lsin=base + 500.0,
    )


def _solver_common_kwargs():
    return {
        "phipf": np.ones(3),
        "chipf": np.ones(3),
        "signgs": 1,
        "lamscale": 1.0,
        "verbose": False,
    }


def _frzl(*, ns: int = 3, mpol: int = 2, nrange: int = 2) -> TomnspsRZL:
    shape = (ns, mpol, nrange)
    base = np.arange(np.prod(shape), dtype=float).reshape(shape)
    return TomnspsRZL(
        frcc=base + 1.0,
        frss=base + 2.0,
        fzsc=base + 3.0,
        fzcs=base + 4.0,
        flsc=base + 5.0,
        flcs=base + 6.0,
        frsc=base + 7.0,
        frcs=base + 8.0,
        fzcc=base + 9.0,
        fzss=base + 10.0,
        flcc=base + 11.0,
        flss=base + 12.0,
    )


def test_jit_cache_helpers_cover_no_jax_and_cached_paths(monkeypatch):
    fake_static = SimpleNamespace(cfg=SimpleNamespace(ns=3, mpol=2, ntor=1, lasym=False))

    monkeypatch.setattr(solve_mod, "has_jax", lambda: False)
    assert solve_mod._strict_update_step_jit(
        fake_static,
        limit_update_rms=False,
        need_update_rms=False,
        divide_by_scalxc_for_update=False,
    ) is None
    assert solve_mod._preconditioner_output_scaling_jit(apply_lambda_update_scale=False) is None

    monkeypatch.setattr(solve_mod, "has_jax", lambda: True)
    solve_mod._STRICT_UPDATE_STEP_JIT_CACHE.clear()
    solve_mod._PRECOND_OUTPUT_SCALE_JIT_CACHE.clear()

    first_step = solve_mod._strict_update_step_jit(
        fake_static,
        limit_update_rms=True,
        need_update_rms=True,
        divide_by_scalxc_for_update=False,
    )
    second_step = solve_mod._strict_update_step_jit(
        fake_static,
        limit_update_rms=True,
        need_update_rms=True,
        divide_by_scalxc_for_update=False,
    )
    assert second_step is first_step
    no_rms_step = solve_mod._strict_update_step_jit(
        fake_static,
        limit_update_rms=False,
        need_update_rms=False,
        divide_by_scalxc_for_update=False,
    )
    yes_rms_step = solve_mod._strict_update_step_jit(
        fake_static,
        limit_update_rms=False,
        need_update_rms=True,
        divide_by_scalxc_for_update=False,
    )
    assert yes_rms_step is not no_rms_step

    first_scale = solve_mod._preconditioner_output_scaling_jit(apply_lambda_update_scale=True)
    second_scale = solve_mod._preconditioner_output_scaling_jit(apply_lambda_update_scale=True)
    assert second_scale is first_scale


def test_zero_edge_rz_force_block_jax_path_preserves_short_mesh_and_zeros_lcfs():
    short = jnp.ones((1, 2))
    short_out = solve_mod._zero_edge_rz_force_block(short, preserve_numpy=False)
    np.testing.assert_allclose(np.asarray(short_out), np.ones((1, 2)))

    arr = jnp.arange(6.0).reshape(3, 2)
    out = solve_mod._zero_edge_rz_force_block(arr, preserve_numpy=False)
    np.testing.assert_allclose(np.asarray(out)[:-1], np.asarray(arr)[:-1])
    np.testing.assert_allclose(np.asarray(out)[-1], 0.0)


def test_vmec2000_print_cadence_and_legacy_dump_path_without_directory(monkeypatch):
    assert solve_mod._should_print_vmec2000_row(
        iter_idx=4,
        max_iter=10,
        nstep_screen=2,
        verbose=True,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
    )
    assert not solve_mod._should_print_vmec2000_row(
        iter_idx=4,
        max_iter=10,
        nstep_screen=2,
        verbose=False,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
    )

    monkeypatch.setenv("VMEC_JAX_DUMP_TIMECONTROL", "1")
    monkeypatch.delenv("VMEC_JAX_DUMP_DIR", raising=False)
    assert solve_mod._legacy_dump_record_path(
        enable_env="VMEC_JAX_DUMP_TIMECONTROL",
        filename="time_control.log",
    ) is None


def test_legacy_dump_write_failures_are_nonfatal(monkeypatch):
    static = _small_static(ns=3)
    state = _state(static)
    zeros = np.zeros_like(np.asarray(state.Rcos))

    class BadPath:
        def open(self, *args, **kwargs):
            raise OSError("blocked")

    monkeypatch.setattr(solve_mod, "_legacy_dump_record_path", lambda **kwargs: BadPath())

    solve_mod._maybe_dump_time_control_record(
        iter_idx=1,
        fsq=1.0,
        fsq0=2.0,
        res0=3.0,
        res1=4.0,
        time_step=0.5,
    )
    solve_mod._dump_time_control_trace_record(
        stage="probe",
        iter2=2,
        iter1=1,
        fsq=1.0,
        fsq0=2.0,
        res0=3.0,
        res1=4.0,
        time_step=0.5,
        irst=1,
    )
    solve_mod._maybe_dump_checkpoint_record(iter_idx=2, fsq=1.0, fsq0=2.0, res0=3.0, res1=4.0)
    solve_mod._dump_freeb_control_trace_record(
        iter2=2,
        iter1=1,
        ivac=1,
        ivacskip=0,
        nvacskip=3,
        fsq_rz_prev=0.25,
        cached=True,
    )
    solve_mod._dump_freeb_axis_trace_record(iter2=2, axis_r=np.array([1.0]), axis_z=np.array([2.0]))
    solve_mod._maybe_dump_evolve_trace_record(
        static=static,
        iter2=2,
        iter1=1,
        stage="accepted",
        fsq1_val=1.0,
        fsq_prev_val=2.0,
        time_step_val=0.5,
        dtau_val=0.25,
        b1_val=0.1,
        fac_val=0.2,
        state_val=state,
        vRcc_val=zeros,
        vRss_val=zeros,
        vZsc_val=zeros,
        vZcs_val=zeros,
        vLsc_val=zeros,
        vLcs_val=zeros,
        frcc_val=zeros,
        frss_val=zeros,
        fzsc_val=zeros,
        fzcs_val=zeros,
        flsc_val=zeros,
        flcs_val=zeros,
    )


def test_radial_tridi_and_half_mesh_helpers_cover_errors_and_singletons():
    arr3 = jnp.arange(12.0).reshape(3, 2, 2)
    with pytest.raises(ValueError, match="ndim>=2"):
        solve_mod._radial_tridi_smooth_dirichlet(arr3, alpha=0.25, allow_3d=False)

    out3 = solve_mod._radial_tridi_smooth_dirichlet(arr3, alpha=0.25, allow_3d=True)
    assert np.asarray(out3).shape == (3, 2, 2)
    np.testing.assert_allclose(np.asarray(out3)[0], np.asarray(arr3)[0])
    np.testing.assert_allclose(np.asarray(out3)[-1], np.asarray(arr3)[-1])

    np.testing.assert_allclose(solve_mod._pshalf_from_s_np(np.array([-4.0])), np.array([0.0]))
    np.testing.assert_allclose(
        np.asarray(solve_mod._pshalf_from_s_jax(np.array([-4.0]), jnp.float64)),
        np.array([0.0]),
    )
    np.testing.assert_allclose(np.asarray(solve_mod._half_mesh_from_full_mesh(jnp.array([2.0]))), np.array([2.0]))


def test_jacobian_terms_dump_resizes_short_half_mesh(tmp_path, monkeypatch):
    shape = (3, 1, 1)
    base = np.ones(shape)
    names = (
        "pr1_even",
        "pr1_odd",
        "pz1_even",
        "pz1_odd",
        "pru_even",
        "pru_odd",
        "pzu_even",
        "pzu_odd",
        "prv_even",
        "prv_odd",
        "pzv_even",
        "pzv_odd",
    )
    kernels = SimpleNamespace(**{name: base + idx for idx, name in enumerate(names)})

    monkeypatch.setenv("VMEC_JAX_DUMP_JACOBIAN_TERMS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "3")

    solve_mod._maybe_dump_jacobian_terms_record(k=kernels, s=np.array([0.0, 1.0]), iter_idx=3)

    out = tmp_path / "jacobian_terms_iter3.dat"
    assert out.exists()
    assert "jacobian term dump" in out.read_text(encoding="utf-8")


def test_host_restart_decision_reference_and_vmecpp_control_branches():
    invalid_stage = solve_mod._host_restart_decision(
        iter2=1,
        iter1=1,
        fsqr=1.0,
        fsqz=0.0,
        fsql=0.0,
        fsq1=1.0,
        fsq_prev=2.0,
        res0=1.0,
        bad_growth_streak=0,
        pre_restart_reason="none",
        reference_mode=False,
        vmec2000_control=False,
        bad_jacobian=False,
        stage_prev_fsq=object(),
        stage_transition_factor=2.0,
        lmove_axis=False,
        vmecpp_restart=False,
        k_preconditioner_update_interval=8,
    )
    assert invalid_stage.pre_restart_reason == "none"

    reference_bad_jacobian = solve_mod._host_restart_decision(
        iter2=2,
        iter1=1,
        fsqr=101.0,
        fsqz=0.0,
        fsql=0.0,
        fsq1=1.0,
        fsq_prev=1.0,
        res0=1.0,
        bad_growth_streak=0,
        pre_restart_reason="none",
        reference_mode=True,
        vmec2000_control=False,
        bad_jacobian=False,
        stage_prev_fsq=None,
        stage_transition_factor=2.0,
        lmove_axis=False,
        vmecpp_restart=False,
        k_preconditioner_update_interval=100,
    )
    assert reference_bad_jacobian.pre_restart_reason == "bad_jacobian"

    vmecpp_bad_progress = solve_mod._host_restart_decision(
        iter2=11,
        iter1=0,
        fsqr=0.02,
        fsqz=0.0,
        fsql=0.0,
        fsq1=0.02,
        fsq_prev=0.01,
        res0=0.01,
        bad_growth_streak=0,
        pre_restart_reason="none",
        reference_mode=False,
        vmec2000_control=True,
        bad_jacobian=False,
        stage_prev_fsq=None,
        stage_transition_factor=2.0,
        lmove_axis=False,
        vmecpp_restart=True,
        k_preconditioner_update_interval=5,
    )
    assert vmecpp_bad_progress.vmecpp_bad_progress
    assert vmecpp_bad_progress.pre_restart_reason == "bad_progress_vmecpp"


def test_dump_helpers_honor_iter_filters_before_payload_materialization(tmp_path, monkeypatch):
    static = _small_static(ns=3)
    state = _state(static)
    zeros = np.zeros_like(np.asarray(state.Rcos))

    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VMEC_JAX_DUMP_ITER", "99")

    monkeypatch.setenv("VMEC_JAX_DUMP_TOMNSPS", "1")
    solve_mod._maybe_dump_tomnsps(frzl=SimpleNamespace(), static=static, iter_idx=1)

    monkeypatch.setenv("VMEC_JAX_DUMP_FORCE_KERNELS", "1")
    solve_mod._maybe_dump_force_kernels(k=SimpleNamespace(), static=static, iter_idx=1)

    monkeypatch.setenv("VMEC_JAX_DUMP_GCX2", "1")
    solve_mod._maybe_dump_gcx2(gcr2=1.0, gcz2=2.0, gcl2=3.0, iter_idx=1, include_edge=False, ns=3)

    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND_MATS", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND_MATS_ITER", "99")
    solve_mod._maybe_dump_precond_mats(mats={}, static=static, iter_idx=1, jmax=2)

    monkeypatch.setenv("VMEC_JAX_DUMP_LAM", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_LAM_ITER", "99")
    solve_mod._maybe_dump_lam_prec(lam_prec=np.zeros((1,)), faclam=None, static=static, iter_idx=1)
    solve_mod._maybe_dump_lam_fsql1(fsql1_pre=1.0, fsql1_post=2.0, static=static, iter_idx=1)

    monkeypatch.setenv("VMEC_JAX_DUMP_LAMCAL", "1")
    solve_mod._maybe_dump_lamcal(lam_debug={}, static=static, iter_idx=1)

    monkeypatch.delenv("VMEC_JAX_DUMP_LAM_ITER", raising=False)
    solve_mod._maybe_dump_lam_gcl(
        frzl_pre=SimpleNamespace(),
        frzl_post=SimpleNamespace(),
        static=static,
        iter_idx=1,
        delta_s=1.0,
    )

    monkeypatch.setenv("VMEC_JAX_DUMP_BSUBE", "1")
    solve_mod._maybe_dump_bsube(bc=SimpleNamespace(), static=static, iter_idx=1)

    monkeypatch.setenv("VMEC_JAX_DUMP_BSUBE_TERMS", "1")
    solve_mod._maybe_dump_bsube_terms(bc=SimpleNamespace(), static=static, iter_idx=1)

    monkeypatch.setenv("VMEC_JAX_DUMP_BSUBH", "1")
    solve_mod._maybe_dump_bsubh(bc=SimpleNamespace(), static=static, iter_idx=1)

    monkeypatch.setenv("VMEC_JAX_DUMP_LULV", "1")
    solve_mod._maybe_dump_lulv(bc=SimpleNamespace(), static=static, iter_idx=1)

    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND", "1")
    solve_mod._maybe_dump_precond_inputs(bc=SimpleNamespace(), trig=None, static=static, iter_idx=1)

    monkeypatch.setenv("VMEC_JAX_DUMP_GMETRIC", "1")
    solve_mod._maybe_dump_gmetric(bc=SimpleNamespace(), static=static, iter_idx=1)

    monkeypatch.setenv("VMEC_JAX_DUMP_XC", "1")
    solve_mod._maybe_dump_xc(
        state=state,
        vRcc=zeros,
        vRss=zeros,
        vZsc=zeros,
        vZcs=zeros,
        vLsc=zeros,
        vLcs=zeros,
        static=static,
        iter_idx=1,
    )

    assert not any(tmp_path.iterdir())


def test_gc_stage_filter_defaults_invalid_stage_to_precond(monkeypatch, tmp_path):
    static = _small_static(ns=3)
    monkeypatch.setenv("VMEC_JAX_DUMP_GC", "1")
    monkeypatch.setenv("VMEC_JAX_DUMP_GC_DIR", str(tmp_path))
    monkeypatch.delenv("VMEC_JAX_DUMP_GC_ITER", raising=False)
    monkeypatch.setenv("VMEC_JAX_DUMP_GC_STAGE", "not-a-stage")

    solve_mod._maybe_dump_gc(frzl=SimpleNamespace(), static=static, iter_idx=1, label="raw")

    assert not any(tmp_path.iterdir())


def test_preconditioner_and_field_slice_pure_branches():
    frzl = _frzl(ns=2, mpol=2, nrange=1)
    empty_mats = {
        "dr": np.zeros((0, 2, 1)),
        "dz": np.zeros((0, 2, 1)),
    }

    assert solve_mod._scale_m1_precond_rhs_from_mats(
        frzl,
        empty_mats,
        lconm1=True,
        mpol=2,
        host_update_assembly=True,
    ) is frzl
    assert solve_mod._scale_m1_precond_rhs_from_mats(
        frzl,
        empty_mats,
        lconm1=True,
        mpol=2,
        host_update_assembly=False,
    ) is frzl

    arr = jnp.arange(12.0).reshape(2, 3, 2)
    repl = jnp.full((2, 2), -1.0)
    replaced = solve_mod._replace_mode_slice(arr, mode_idx=0, replacement=repl)
    np.testing.assert_allclose(np.asarray(replaced)[:, 0, :], -1.0)
    np.testing.assert_allclose(np.asarray(replaced)[:, 1:, :], np.asarray(arr)[:, 1:, :])
    assert solve_mod._replace_mode_slice_np(None, mode_idx=0, replacement=repl) is None

    empty_rows = solve_mod._enforce_field_rows(jnp.zeros((0, 3)), edge_row=np.ones(3))
    assert np.asarray(empty_rows).shape == (0, 3)
    edge_only = solve_mod._enforce_field_rows(jnp.arange(6.0).reshape(2, 3), edge_row=-np.ones(3))
    np.testing.assert_allclose(np.asarray(edge_only)[0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(edge_only)[-1], -1.0)


def test_precond_inputs_and_gmetric_error_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("VMEC_JAX_DUMP_DIR", str(tmp_path))
    monkeypatch.delenv("VMEC_JAX_DUMP_ITER", raising=False)

    arr = np.ones((2, 1, 1))
    bc = SimpleNamespace(
        jac=SimpleNamespace(r12=arr, sqrtg=arr, ru12=arr, zu12=arr),
        bsq=arr,
    )
    trig = SimpleNamespace(wint3_precond=None)

    monkeypatch.setenv("VMEC_JAX_DUMP_PRECOND", "1")
    solve_mod._maybe_dump_precond_inputs(
        bc=bc,
        trig=trig,
        static=SimpleNamespace(cfg=SimpleNamespace(ns=2)),
        iter_idx=2,
        kernels=SimpleNamespace(pru_even=arr),
    )
    assert (tmp_path / "precond_inputs_iter2.dat").exists()
    assert not (tmp_path / "precond_hidden_iter2.npz").exists()

    monkeypatch.setenv("VMEC_JAX_DUMP_GMETRIC", "1")
    solve_mod._maybe_dump_gmetric(bc=SimpleNamespace(), static=SimpleNamespace(cfg=SimpleNamespace(ns=2)), iter_idx=3)
    solve_mod._maybe_dump_gmetric(
        bc=SimpleNamespace(guu=np.ones((2, 1)), guv=np.ones((2, 1)), gvv=np.ones((2, 1))),
        static=SimpleNamespace(cfg=SimpleNamespace(ns=2)),
        iter_idx=4,
    )
    solve_mod._maybe_dump_gmetric(
        bc=SimpleNamespace(guu=arr, guv=arr, gvv=arr, jac=SimpleNamespace()),
        static=SimpleNamespace(cfg=SimpleNamespace(ns=2)),
        iter_idx=5,
    )
    assert (tmp_path / "gmetric_iter5.dat").exists()


def test_current_profile_mismatch_falls_back_to_zero(monkeypatch):
    import vmec_jax.profiles as profiles

    def fake_eval_profiles(_indata, s):
        if int(np.asarray(s).shape[0]) == 1:
            return {"current": jnp.asarray([1.0])}
        return {"current": jnp.asarray([1.0])}

    monkeypatch.setattr(profiles, "eval_profiles", fake_eval_profiles)
    indata = SimpleNamespace(
        get_int=lambda name, default=0: 1 if name == "NCURR" else default,
        get_float=lambda name, default=0.0: 1.0 if name == "CURTOR" else default,
    )

    out = solve_mod._icurv_full_mesh_from_indata(indata=indata, s_full=jnp.linspace(0.0, 1.0, 3), signgs=1)
    np.testing.assert_allclose(np.asarray(out), 0.0)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda state, static: solve_mod.solve_fixed_boundary_gd(
                state,
                static,
                **_solver_common_kwargs(),
                max_iter=0,
            ),
            "max_iter",
        ),
        (
            lambda state, static: solve_mod.solve_fixed_boundary_gd(
                state,
                static,
                **_solver_common_kwargs(),
                max_backtracks=-1,
            ),
            "max_backtracks",
        ),
        (
            lambda state, static: solve_mod.solve_fixed_boundary_gd(
                state,
                static,
                **_solver_common_kwargs(),
                bt_factor=1.0,
            ),
            "bt_factor",
        ),
        (
            lambda state, static: solve_mod.solve_fixed_boundary_lbfgs(
                state,
                static,
                **_solver_common_kwargs(),
                max_iter=0,
            ),
            "max_iter",
        ),
        (
            lambda state, static: solve_mod.solve_fixed_boundary_lbfgs(
                state,
                static,
                **_solver_common_kwargs(),
                max_backtracks=-1,
            ),
            "max_backtracks",
        ),
        (
            lambda state, static: solve_mod.solve_fixed_boundary_lbfgs(
                state,
                static,
                **_solver_common_kwargs(),
                bt_factor=1.0,
            ),
            "bt_factor",
        ),
        (
            lambda state, static: solve_mod.solve_fixed_boundary_lbfgs(
                state,
                static,
                **_solver_common_kwargs(),
                gamma=1.0,
            ),
            "gamma=1",
        ),
        (
            lambda state, static: solve_mod.solve_fixed_boundary_lbfgs_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                history_size=0,
            ),
            "history_size",
        ),
        (
            lambda state, static: solve_mod.solve_fixed_boundary_lbfgs_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                max_iter=0,
            ),
            "max_iter",
        ),
        (
            lambda state, static: solve_mod.solve_fixed_boundary_lbfgs_vmec_residual(
                state,
                static,
                indata=SimpleNamespace(),
                signgs=1,
                bt_factor=1.0,
            ),
            "bt_factor",
        ),
    ],
)
def test_additional_solver_entry_validation_branches_short_circuit(call, match):
    static = _small_static(ns=3)
    state = _state(static)

    with pytest.raises(ValueError, match=match):
        call(state, static)

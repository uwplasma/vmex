from __future__ import annotations

import pytest

from vmec_jax.solvers.fixed_boundary.residual.config import (
    HEAVY_DUMP_ENVS,
    bad_jacobian_tau_tolerance,
    indata_has_profile_setup_work,
    legacy_dump_enabled,
    parse_bad_jacobian_config,
    resolve_axis_reset_config,
    resolve_chunked_scan_config,
    resolve_debug_print_config,
    resolve_dump_history_config,
    resolve_host_profile_setup,
    resolve_host_residual_metric_config,
    resolve_nstep_screen,
    resolve_setup_host_enforce,
    should_probe_bad_jacobian_state,
)


def test_bad_jacobian_mode_falls_back_to_ptau_and_clamps_relative_tolerance():
    cfg = parse_bad_jacobian_config(
        {
            "VMEC_JAX_BADJAC_MODE": "bogus",
            "VMEC_JAX_DUMP_PTAU_STATE": "yes",
            "VMEC_JAX_BADJAC_STATE_PROBE": "true",
            "VMEC_JAX_PTAU_TOL": "not-a-float",
            "VMEC_JAX_PTAU_TOL_REL": "-1.0",
        }
    )

    assert cfg.mode == "ptau"
    assert cfg.use_state is False
    assert cfg.dump_ptau_state is True
    assert cfg.state_probe is True
    assert cfg.ptau_tol == 0.0
    assert cfg.ptau_tol_rel == 0.0


def test_bad_jacobian_state_mode_and_tolerance_policy():
    cfg = parse_bad_jacobian_config(
        {
            "VMEC_JAX_BADJAC_MODE": " STATE ",
            "VMEC_JAX_PTAU_TOL": "-2.0e-4",
            "VMEC_JAX_PTAU_TOL_REL": "1.0e-3",
        }
    )

    assert cfg.mode == "state"
    assert cfg.use_state is True
    assert cfg.ptau_tol == pytest.approx(-2.0e-4)
    assert cfg.ptau_tol_rel == pytest.approx(1.0e-3)
    assert bad_jacobian_tau_tolerance(ptau_tol=cfg.ptau_tol, ptau_tol_rel=cfg.ptau_tol_rel, tau_scale=0.5) == pytest.approx(
        5.0e-4
    )
    assert bad_jacobian_tau_tolerance(ptau_tol=cfg.ptau_tol, ptau_tol_rel=0.0, tau_scale=0.5) == pytest.approx(
        2.0e-4
    )


def test_bad_jacobian_invalid_relative_tolerance_falls_back_to_zero():
    cfg = parse_bad_jacobian_config({"VMEC_JAX_PTAU_TOL": "2.5e-6", "VMEC_JAX_PTAU_TOL_REL": "not-a-float"})

    assert cfg.ptau_tol == pytest.approx(2.5e-6)
    assert cfg.ptau_tol_rel == 0.0


def test_bad_jacobian_invalid_initial_probe_iters_uses_default_and_clamps_negative():
    invalid = parse_bad_jacobian_config({"VMEC_JAX_BADJAC_INITIAL_STATE_PROBE_ITERS": "not-an-int"})
    assert invalid.initial_state_probe_iters == 2

    negative = parse_bad_jacobian_config({"VMEC_JAX_BADJAC_INITIAL_STATE_PROBE_ITERS": "-5"})
    assert negative.initial_state_probe_iters == 0

    assert should_probe_bad_jacobian_state(state_probe=True, initial_state_probe_iters=2, iter_idx=2)
    assert not should_probe_bad_jacobian_state(state_probe=True, initial_state_probe_iters=2, iter_idx=3)


def test_heavy_dump_flags_disable_jit_and_force_full_history():
    env = {HEAVY_DUMP_ENVS[0]: "1"}

    cfg = resolve_dump_history_config(env=env, jit_forces=True, light_history=True)

    assert cfg.dumps_enabled is True
    assert cfg.dump_any is True
    assert cfg.jit_forces is False
    assert cfg.light_history is False
    assert cfg.track_history is True
    assert cfg.disabled_jit_for_dumps is True


def test_light_dump_flags_force_full_history_without_disabling_jit():
    cfg = resolve_dump_history_config(
        env={"VMEC_JAX_DUMP_SCALARS": "1"},
        jit_forces=True,
        light_history=True,
    )

    assert cfg.dumps_enabled is False
    assert cfg.dump_any is True
    assert cfg.jit_forces is True
    assert cfg.light_history is False
    assert cfg.track_history is True
    assert cfg.disabled_jit_for_dumps is False


def test_legacy_dump_flag_parsing_matches_unstripped_solve_behavior():
    assert legacy_dump_enabled("false") is True
    assert legacy_dump_enabled(" 0 ") is True
    assert legacy_dump_enabled("0") is False
    assert legacy_dump_enabled("") is False


def test_chunked_scan_and_fallback_disable_for_differentiated_scan():
    cfg = resolve_chunked_scan_config(
        use_scan=True,
        state_has_tracer=True,
        scan_fallback_enabled=True,
        chunked_env="1",
    )

    assert cfg.differentiating_scan is True
    assert cfg.force_chunked_scan is False
    assert cfg.scan_fallback_enabled is False


def test_chunked_scan_disabled_when_not_using_scan_but_fallback_preserved():
    cfg = resolve_chunked_scan_config(
        use_scan=False,
        state_has_tracer=False,
        scan_fallback_enabled=True,
        chunked_env="1",
    )

    assert cfg.differentiating_scan is False
    assert cfg.force_chunked_scan is False
    assert cfg.scan_fallback_enabled is True


def test_host_residual_metric_policy_auto_and_explicit_flags():
    cpu_auto = resolve_host_residual_metric_config(
        backend_name="cpu",
        fsq1_norms_env="auto",
        residual_metrics_env="auto",
    )
    assert cpu_auto.fsq1_norms_on_accelerator is False
    assert cpu_auto.residual_metrics_on_accelerator is False

    gpu_auto = resolve_host_residual_metric_config(
        backend_name="gpu",
        fsq1_norms_env="auto",
        residual_metrics_env="auto",
    )
    assert gpu_auto.fsq1_norms_on_accelerator is True
    assert gpu_auto.residual_metrics_on_accelerator is False

    explicit = resolve_host_residual_metric_config(
        backend_name="gpu",
        fsq1_norms_env=" off ",
        residual_metrics_env=" yes ",
    )
    assert explicit.fsq1_norms_on_accelerator is False
    assert explicit.residual_metrics_on_accelerator is True


def test_host_profile_setup_policy_auto_and_explicit_flags():
    assert resolve_host_profile_setup(backend_name="cpu", profile_setup_env="auto") is False
    assert (
        resolve_host_profile_setup(
            backend_name="cpu",
            profile_setup_env="auto",
            profile_setup_has_work=True,
        )
        is True
    )
    assert resolve_host_profile_setup(backend_name="gpu", profile_setup_env="auto") is True
    assert resolve_host_profile_setup(backend_name="gpu", profile_setup_env=" off ") is False
    assert resolve_host_profile_setup(backend_name="cpu", profile_setup_env=" on ") is True


class _ProfileDeck:
    def __init__(self, values):
        self.values = dict(values)

    def get(self, key, default=None):
        return self.values.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self.values.get(key, default))

    def get_float(self, key, default=0.0):
        return float(self.values.get(key, default))

    def get_int(self, key, default=0):
        return int(self.values.get(key, default))


def test_indata_profile_setup_work_detects_only_real_profile_work():
    assert indata_has_profile_setup_work(None) is False
    assert indata_has_profile_setup_work(_ProfileDeck({"PRES_SCALE": 1.0, "NCURR": 1, "CURTOR": 0.0})) is False
    assert indata_has_profile_setup_work(_ProfileDeck({"PRES_SCALE": 1.0, "AM": [0.0, 2.0]})) is True
    assert indata_has_profile_setup_work(_ProfileDeck({"PRES_SCALE": 0.0, "AM": [0.0, 2.0]})) is False
    assert indata_has_profile_setup_work(_ProfileDeck({"AI": [0.0, 0.4]})) is True
    assert indata_has_profile_setup_work(
        _ProfileDeck({"NCURR": 1, "CURTOR": -2.0, "AC": [0.0, 3.0]})
    ) is True
    assert indata_has_profile_setup_work(_ProfileDeck({"APHI": [1.0, 0.1]})) is True
    assert indata_has_profile_setup_work(_ProfileDeck({"LRFP": True})) is True


def test_axis_reset_config_preserves_legacy_env_policy():
    default = resolve_axis_reset_config(
        force_axis_reset_env=None,
        axis_reset_always_3d_env=None,
        axis_reset_fsq_min_env=None,
    )
    assert default.force_axis_reset is False
    assert default.axis_reset_always_3d is False
    assert default.axis_reset_fsq_min == pytest.approx(1.0)

    explicit = resolve_axis_reset_config(
        force_axis_reset_env=" yes ",
        axis_reset_always_3d_env="off",
        axis_reset_fsq_min_env="2.5",
    )
    assert explicit.force_axis_reset is True
    assert explicit.axis_reset_always_3d is True
    assert explicit.axis_reset_fsq_min == pytest.approx(2.5)

    invalid = resolve_axis_reset_config(
        force_axis_reset_env="0",
        axis_reset_always_3d_env="false",
        axis_reset_fsq_min_env="not-a-float",
    )
    assert invalid.force_axis_reset is False
    assert invalid.axis_reset_always_3d is False
    assert invalid.axis_reset_fsq_min == 0.0

    negative = resolve_axis_reset_config(
        force_axis_reset_env="",
        axis_reset_always_3d_env="",
        axis_reset_fsq_min_env="-2.0",
    )
    assert negative.axis_reset_fsq_min == 0.0


def test_setup_host_enforce_policy_preserves_tracing_and_backend_rules():
    assert (
        resolve_setup_host_enforce(
            setup_host_enforce_env="0",
            host_update_assembly=False,
            use_scan=False,
            state_has_tracer=False,
            backend_name="gpu",
        )
        is False
    )
    assert (
        resolve_setup_host_enforce(
            setup_host_enforce_env="force",
            host_update_assembly=True,
            use_scan=True,
            state_has_tracer=False,
            backend_name="cpu",
        )
        is True
    )
    assert (
        resolve_setup_host_enforce(
            setup_host_enforce_env="1",
            host_update_assembly=False,
            use_scan=False,
            state_has_tracer=True,
            backend_name="gpu",
        )
        is False
    )
    assert (
        resolve_setup_host_enforce(
            setup_host_enforce_env="auto",
            host_update_assembly=False,
            use_scan=False,
            state_has_tracer=False,
            backend_name="gpu",
        )
        is True
    )
    assert (
        resolve_setup_host_enforce(
            setup_host_enforce_env="auto",
            host_update_assembly=False,
            use_scan=False,
            state_has_tracer=False,
            backend_name="cpu",
        )
        is False
    )
    assert (
        resolve_setup_host_enforce(
            setup_host_enforce_env="auto",
            host_update_assembly=True,
            use_scan=False,
            state_has_tracer=False,
            backend_name="gpu",
        )
        is False
    )


def test_nstep_override_parsing_and_clamping():
    assert resolve_nstep_screen(indata_nstep=50, override_env="7") == 7
    assert resolve_nstep_screen(indata_nstep=50, override_env="bad") == 50
    assert resolve_nstep_screen(indata_nstep=50, override_env="0") == 50
    assert resolve_nstep_screen(indata_nstep=-3, override_env="") == 1
    assert resolve_nstep_screen(indata_nstep=50, override_env="-2") == 1


def test_debug_print_mode_fallback_and_ordering():
    invalid = resolve_debug_print_config(
        print_env="1",
        mode_env="not-a-mode",
        ordered_env="yes",
    )
    assert invalid.print_live is True
    assert invalid.mode == "debug_print"
    assert invalid.ordered is True

    no_io = resolve_debug_print_config(
        print_env="false",
        mode_env="io_callback",
        ordered_env="0",
        io_callback_available=False,
    )
    assert no_io.print_live is False
    assert no_io.mode == "debug_print"
    assert no_io.ordered is False

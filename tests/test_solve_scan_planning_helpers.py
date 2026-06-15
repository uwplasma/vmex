import pytest

from vmec_jax.solve_residual_iter_policy import vmec2000_scan_options_from_env
from vmec_jax.solvers.fixed_boundary.scan.planning import (
    SCAN_TIMING_COUNT_KEYS,
    SCAN_TIMING_KEYS,
    apply_state_only_scan_options,
    build_scan_timing_report,
    build_vmec2000_scan_cache_key,
    new_scan_timing_stats,
    normalize_scan_print_mode,
    resolve_scan_iteration_plan,
    resolve_scan_preflight_iters,
    resolve_scan_run_flags,
    scan_chunk_settings,
    scan_jit_forces_enabled,
    scan_jit_preflight_enabled,
    scan_timing_enabled,
    validate_vmec2000_scan_guards,
)


def _scan_options(**overrides):
    params = dict(
        verbose=True,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
        light_history=False,
        scan_minimal_default=False,
        dump_any=False,
        fsq_total_target=None,
        backend_name="gpu",
        force_chunked_scan_run=False,
        scan_print_env="1",
        scan_print_mode_env="debug_callback",
        scan_print_ordered_env="0",
        scan_print_chunked_env="1",
        scan_light_env="0",
        scan_minimal_env="",
        scan_core_env="",
        scan_trace_env="0",
        abort_scan_env="0",
        scan_precompute_env="",
        tridi_precompute_env="",
        scan_lax_env="",
        tridi_solve_env="",
        scan_restart_payload_env="",
    )
    params.update(overrides)
    return vmec2000_scan_options_from_env(**params)


def _cache_key(**overrides):
    params = dict(
        static_key=("static", (2, 3)),
        wout_key=("wout", "float64"),
        edge_signature_key=((2,), "float64"),
        tomnsps_policy_key=("tomnsps", "auto"),
        max_iter_tail=9,
        preflight_iters=1,
        iter_offset0=0,
        step_size=0.2,
        initial_flip_sign=-1.0,
        lambda_update_scale=0.5,
        ftol=1.0e-12,
        nstep_screen=25,
        use_restart_triggers=True,
        vmecpp_restart=False,
        scan_use_precomputed=False,
        scan_use_lax_tridi=False,
        scan_use_restart_payload=False,
        stage_prev_fsq=None,
        stage_transition_factor=50.0,
        stage_transition_scale=0.5,
        jit_forces_scan=True,
        state_only_scan=False,
        scan_light=False,
        scan_minimal=True,
        scan_fallback_iters=20,
        scan_fallback_accept_frac=0.5,
        scan_fallback_fsq_factor=50.0,
        scan_fallback_badjac_limit=10,
        scan_fallback_fsq_abs=1.0e-2,
    )
    params.update(overrides)
    return build_vmec2000_scan_cache_key(**params)


@pytest.mark.parametrize("value", ["", "0", "false", "no", " FALSE "])
def test_scan_timing_disabled_tokens(value):
    assert not scan_timing_enabled(value)


@pytest.mark.parametrize("value", ["1", "yes", "true", "debug"])
def test_scan_timing_enabled_tokens(value):
    assert scan_timing_enabled(value)


def test_timing_report_math_excludes_dispatch_breakdown_from_leaf_total():
    stats = new_scan_timing_stats()
    assert tuple(stats) == SCAN_TIMING_KEYS + SCAN_TIMING_COUNT_KEYS
    stats["scan_setup_s"] = 1.0
    stats["scan_device_dispatch_s"] = 0.25
    stats["scan_device_ready_s"] = 0.75
    stats["scan_device_run_s"] = 1.0
    stats["scan_runner_cache_miss_device_run_s"] = 0.6
    stats["scan_runner_cache_miss_ready_s"] = 0.2
    stats["scan_runner_cache_build_s"] = 0.1
    stats["scan_postprocess_s"] = 2.0
    stats["scan_runner_cache_hit_count"] = 3
    stats["scan_runner_cache_miss_category_iteration_budget_count"] = 2

    report = build_scan_timing_report(iterations=7, stats=stats, scan_total_s=5.9)

    assert report["iterations"] == 7
    assert report["scan_total_s"] == 5.9
    assert report["scan_unattributed_s"] == pytest.approx(1.0)
    assert report["scan_device_dispatch_s"] == pytest.approx(0.25)
    assert report["scan_device_ready_s"] == pytest.approx(0.75)
    assert report["scan_runner_cache_hit_count"] == 3
    assert report["scan_runner_cache_miss_count"] == 0
    assert report["scan_runner_cache_miss_category_iteration_budget_count"] == 2
    assert report["scan_cold_cache_miss_s"] == pytest.approx(0.6)
    assert report["scan_cold_cache_miss_ready_s"] == pytest.approx(0.2)
    assert report["scan_cache_build_wrapper_s"] == pytest.approx(0.1)
    assert build_scan_timing_report(iterations=7, stats=stats, scan_total_s=3.0)["scan_unattributed_s"] == 0.0


def test_run_flags_disable_fallback_for_state_only_and_chunking_for_traced_scan():
    normal = resolve_scan_run_flags(
        state_only=False,
        scan_differentiated=False,
        scan_fallback_enabled=True,
        force_chunked_scan=True,
    )
    assert not normal.state_only_scan
    assert normal.scan_fallback_enabled_run
    assert normal.force_chunked_scan_run

    flags = resolve_scan_run_flags(
        state_only=True,
        scan_differentiated=False,
        scan_fallback_enabled=True,
        force_chunked_scan=True,
    )
    assert flags.state_only_scan
    assert not flags.scan_fallback_enabled_run
    assert flags.force_chunked_scan_run

    traced = resolve_scan_run_flags(
        state_only=False,
        scan_differentiated=True,
        scan_fallback_enabled=True,
        force_chunked_scan=True,
    )
    assert not traced.scan_fallback_enabled_run
    assert not traced.force_chunked_scan_run


def test_state_only_overrides_light_minimal_and_print_options():
    options = _scan_options(
        light_history=True,
        scan_minimal_default=False,
        scan_print_chunked_env="1",
    )
    assert options.scan_light
    assert not options.scan_minimal
    assert options.scan_collect_scalars
    assert options.scan_collect_print
    assert options.chunked_print

    state_only = apply_state_only_scan_options(options, state_only_scan=True)

    assert not state_only.scan_light
    assert state_only.scan_minimal
    assert not state_only.scan_collect_scalars
    assert not state_only.scan_collect_print
    assert not state_only.print_in_scan
    assert not state_only.chunked_print
    assert apply_state_only_scan_options(options, state_only_scan=False) is options


def test_scan_options_env_branches_for_minimal_light_and_restart_payload():
    quiet = _scan_options(verbose=False, scan_minimal_default=None, backend_name="cpu")
    assert quiet.scan_minimal
    assert not quiet.print_in_scan
    assert quiet.scan_use_restart_payload

    dumped = _scan_options(dump_any=True, scan_light_env="1", scan_minimal_env="1")
    assert not dumped.scan_light
    assert not dumped.scan_minimal

    forced_payload_off = _scan_options(backend_name="cpu", scan_restart_payload_env="0")
    assert not forced_payload_off.scan_use_restart_payload


def test_scan_options_explicit_backend_and_print_branches():
    default_cpu_lax = _scan_options(backend_name="cpu", scan_lax_env="", tridi_solve_env="")
    default_gpu_lax = _scan_options(backend_name="gpu", scan_lax_env="", tridi_solve_env="")
    assert not default_cpu_lax.scan_use_lax_tridi
    assert not default_gpu_lax.scan_use_lax_tridi
    assert default_cpu_lax.scan_use_precomputed
    assert default_gpu_lax.scan_use_precomputed

    precompute = _scan_options(scan_precompute_env="yes", tridi_precompute_env="0")
    assert precompute.scan_use_precomputed

    explicit_precompute_off = _scan_options(backend_name="gpu", scan_precompute_env="0", tridi_precompute_env="")
    assert not explicit_precompute_off.scan_use_precomputed

    tridi_precompute = _scan_options(scan_precompute_env="", tridi_precompute_env="1")
    assert tridi_precompute.scan_use_precomputed

    explicit_lax_off = _scan_options(scan_lax_env="0", tridi_solve_env="lax")
    assert not explicit_lax_off.scan_use_lax_tridi

    tridi_lax = _scan_options(scan_lax_env="", tridi_solve_env="force")
    assert tridi_lax.scan_use_lax_tridi

    forced_payload_on = _scan_options(backend_name="gpu", scan_restart_payload_env="yes")
    assert forced_payload_on.scan_use_restart_payload

    forced_chunked = _scan_options(force_chunked_scan_run=True, scan_print_chunked_env="0")
    assert forced_chunked.chunked_print
    assert not forced_chunked.print_in_scan

    invalid_print_mode = _scan_options(scan_print_mode_env="bogus", scan_print_chunked_env="0")
    assert invalid_print_mode.scan_print_mode == "debug_print"


def test_scan_jit_forces_and_preflight_env_branches():
    assert scan_jit_forces_enabled(env_value=None, jit_forces=True)
    assert not scan_jit_forces_enabled(env_value="0", jit_forces=True)
    assert scan_jit_forces_enabled(env_value="yes", jit_forces=False)
    assert scan_jit_preflight_enabled(env_value=None, backend_name="gpu", scan_differentiated=False)
    assert not scan_jit_preflight_enabled(env_value=None, backend_name="cpu", scan_differentiated=False)
    assert not scan_jit_preflight_enabled(env_value=None, backend_name="cuda", scan_differentiated=True)
    assert not scan_jit_preflight_enabled(env_value="0", backend_name="gpu", scan_differentiated=False)
    assert scan_jit_preflight_enabled(env_value="yes", backend_name="cpu", scan_differentiated=True)

    assert (
        resolve_scan_preflight_iters(
            jit_forces_scan=True,
            vmec2000_control=True,
            max_iter=5,
            axis_reset_repeat=False,
            preflight_env=None,
        ).preflight_iters
        == 0
    )
    assert (
        resolve_scan_preflight_iters(
            jit_forces_scan=False,
            vmec2000_control=True,
            max_iter=5,
            axis_reset_repeat=False,
            preflight_env=None,
        ).preflight_iters
        == 1
    )
    assert (
        resolve_scan_preflight_iters(
            jit_forces_scan=True,
            vmec2000_control=True,
            max_iter=5,
            axis_reset_repeat=False,
            preflight_env="not-an-int",
        ).preflight_iters
        == 1
    )
    assert (
        resolve_scan_preflight_iters(
            jit_forces_scan=True,
            vmec2000_control=False,
            max_iter=5,
            axis_reset_repeat=True,
            preflight_env="0",
        ).preflight_iters
        == 1
    )
    explicit = resolve_scan_preflight_iters(
        jit_forces_scan=True,
        vmec2000_control=True,
        max_iter=5,
        axis_reset_repeat=False,
        preflight_env="3",
    )
    assert explicit.preflight_default == "0"
    assert explicit.preflight_iters == 3


def test_scan_iteration_plan_defaults_invalid_env_and_clamps_preflight():
    non_control = resolve_scan_iteration_plan(
        max_iter=3,
        preflight_iters=20,
        vmec2000_control=False,
        extra_iters_env=None,
    )
    assert non_control.extra_iters == 10
    assert non_control.max_iter_scan == 13
    assert non_control.preflight_iters == 13
    assert non_control.max_iter_tail == 0

    invalid = resolve_scan_iteration_plan(
        max_iter=3,
        preflight_iters=2,
        vmec2000_control=True,
        extra_iters_env="bad",
    )
    assert invalid.extra_iters == 0
    assert invalid.max_iter_scan == 3
    assert invalid.max_iter_tail == 1

    zero = resolve_scan_iteration_plan(
        max_iter=0,
        preflight_iters=2,
        vmec2000_control=True,
        extra_iters_env="-5",
    )
    assert zero.extra_iters == 0
    assert zero.preflight_iters == 0

    explicit = resolve_scan_iteration_plan(
        max_iter=4,
        preflight_iters=1,
        vmec2000_control=True,
        extra_iters_env="6",
    )
    assert explicit.extra_iters == 6
    assert explicit.max_iter_scan == 10
    assert explicit.max_iter_tail == 9


def test_scan_chunk_settings_match_quiet_and_printing_modes():
    assert scan_chunk_settings(
        max_iter_scan=40,
        nstep_screen=5,
        need_print=False,
        lthreed=True,
        backend_name="cpu",
        chunk_size_env="",
    ) == (40, True)
    assert scan_chunk_settings(
        max_iter_scan=40,
        nstep_screen=5,
        need_print=True,
        lthreed=True,
        backend_name="gpu",
        chunk_size_env="",
    ) == (5, False)
    assert scan_chunk_settings(
        max_iter_scan=40,
        nstep_screen=5,
        need_print=False,
        lthreed=True,
        backend_name="gpu",
        chunk_size_env="",
    ) == (40, True)
    assert scan_chunk_settings(
        max_iter_scan=40,
        nstep_screen=5,
        need_print=False,
        lthreed=True,
        backend_name="gpu",
        chunk_size_env="bad",
    ) == (5, True)
    assert scan_chunk_settings(
        max_iter_scan=40,
        nstep_screen=5,
        need_print=False,
        lthreed=True,
        backend_name="gpu",
        chunk_size_env="0",
    ) == (1, True)
    assert scan_chunk_settings(
        max_iter_scan=1500,
        nstep_screen=5,
        need_print=False,
        lthreed=True,
        backend_name="gpu",
        chunk_size_env="",
        spectral_mode_count=8,
    ) == (512, False)
    assert scan_chunk_settings(
        max_iter_scan=1500,
        nstep_screen=5,
        need_print=False,
        lthreed=True,
        backend_name="gpu",
        chunk_size_env="",
        spectral_mode_count=50,
    ) == (512, False)


def test_scan_cache_key_is_stable_and_tracks_behavioral_toggles():
    base = _cache_key()
    equivalent = _cache_key(
        max_iter_tail=9.0,
        preflight_iters=True,
        stage_prev_fsq=None,
        scan_fallback_badjac_limit=10.0,
    )
    assert equivalent == base
    assert _cache_key(scan_light=True) != base
    assert _cache_key(stage_prev_fsq=3) != base
    assert _cache_key(tomnsps_policy_key=("tomnsps", "fft")) != base
    assert _cache_key(scan_use_precomputed=True) != base
    assert _cache_key(scan_use_lax_tridi=True) != base
    assert _cache_key(stage_prev_fsq=3)[18] == 3.0


def test_scan_print_mode_normalization_and_invalid_guard_errors():
    assert normalize_scan_print_mode(scan_print_mode="io_callback", io_callback_available=True) == "io_callback"
    assert normalize_scan_print_mode(scan_print_mode="io_callback", io_callback_available=False) == "debug_print"
    assert normalize_scan_print_mode(scan_print_mode="unknown", io_callback_available=True) == "debug_print"

    validate_vmec2000_scan_guards(
        backtracking=False,
        limit_dt_from_force=False,
        limit_update_rms=False,
        use_direct_fallback=False,
        reference_mode=False,
        strict_update=True,
        auto_flip_force=False,
    )
    with pytest.raises(ValueError, match="backtracking=False"):
        validate_vmec2000_scan_guards(
            backtracking=True,
            limit_dt_from_force=False,
            limit_update_rms=False,
            use_direct_fallback=False,
            reference_mode=False,
            strict_update=True,
            auto_flip_force=False,
        )
    with pytest.raises(ValueError, match="strict_update=True"):
        validate_vmec2000_scan_guards(
            backtracking=False,
            limit_dt_from_force=False,
            limit_update_rms=False,
            use_direct_fallback=False,
            reference_mode=False,
            strict_update=False,
            auto_flip_force=False,
        )
    with pytest.raises(ValueError, match="auto_flip_force=True"):
        validate_vmec2000_scan_guards(
            backtracking=False,
            limit_dt_from_force=False,
            limit_update_rms=False,
            use_direct_fallback=False,
            reference_mode=False,
            strict_update=True,
            auto_flip_force=True,
        )
    with pytest.raises(ValueError, match="limit_dt_from_force=False"):
        validate_vmec2000_scan_guards(
            backtracking=False,
            limit_dt_from_force=True,
            limit_update_rms=False,
            use_direct_fallback=False,
            reference_mode=False,
            strict_update=True,
            auto_flip_force=False,
        )

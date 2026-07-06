from collections import OrderedDict, namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax._compat import jnp
from vmec_jax.solvers.fixed_boundary.residual.policy import vmec2000_scan_options_from_env
from vmec_jax.solvers.fixed_boundary.residual.accelerated_scan import _accelerated_scan_cache_key
from vmec_jax.solvers.fixed_boundary.scan.controller import _select_initial_rz_norm_func
from vmec_jax.solvers.fixed_boundary.scan.planning import (
    SCAN_TIMING_COUNT_KEYS,
    SCAN_TIMING_KEYS,
    apply_state_only_scan_options,
    build_scan_timing_report,
    build_vmec2000_scan_cache_key,
    default_vmec2000_controller_constants,
    new_scan_timing_stats,
    normalize_scan_print_mode,
    resolve_scan_iteration_plan,
    resolve_scan_iteration_runtime_plan,
    resolve_scan_preflight_iters,
    resolve_scan_run_flags,
    resolve_vmec2000_scan_setup,
    scan_chunk_settings,
    scan_jit_forces_enabled,
    scan_jit_preflight_enabled,
    scan_timing_enabled,
    validate_vmec2000_scan_guards,
)
from vmec_jax.solvers.fixed_boundary.scan.runtime import (
    get_or_build_scan_runner,
    maybe_explicit_compile_scan_runner,
    maybe_record_scan_runner_hlo_summary,
    maybe_record_scan_runner_arg_summary,
    record_scan_history_summary,
    record_scan_runner_arg_summary,
    resolve_scan_runtime_hooks,
    resolve_scan_runtime_hooks_from_env,
    run_scan_preflight_step,
    run_nonchunked_scan,
    scan_arg_summary_enabled,
    scan_explicit_compile_enabled,
    scan_hlo_summary_enabled,
    scan_trace_context_or_null,
    summarize_scan_runner_hlo_text,
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
        fsq_total_target=None,
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


def test_accelerated_scan_cache_key_excludes_dynamic_scalar_controls():
    base = dict(
        static_key=("static", 16, 4, 4),
        wout_key=("wout", "float64"),
        edge_signature_key=("edge", (4,)),
        max_iter=50,
        state_only=False,
        has_fsq_total_target=False,
        precond_radial_alpha=0.5,
        precond_lambda_alpha=0.25,
        apply_m1_constraints=True,
        jit_forces=True,
    )

    key1 = _accelerated_scan_cache_key(**base)
    key2 = _accelerated_scan_cache_key(**base)
    key3 = _accelerated_scan_cache_key(**{**base, "has_fsq_total_target": True})
    key4 = _accelerated_scan_cache_key(**{**base, "state_only": True})

    assert key1 == key2
    assert key1 != key3
    assert key1 != key4
    assert "scan_v2" in key1


def test_accelerated_scan_cache_key_excludes_dynamic_edge_values():
    base = dict(
        static_key=("static", 16, 4, 4),
        wout_key=("wout", "float64"),
        edge_signature_key=(((4,), "float64"),),
        max_iter=50,
        state_only=False,
        has_fsq_total_target=False,
        precond_radial_alpha=0.5,
        precond_lambda_alpha=0.25,
        apply_m1_constraints=True,
        jit_forces=True,
    )

    same_shape_new_boundary = {**base, "edge_signature_key": (((4,), "float64"),)}
    changed_shape = {**base, "edge_signature_key": (((5,), "float64"),)}

    assert _accelerated_scan_cache_key(**base) == _accelerated_scan_cache_key(**same_shape_new_boundary)
    assert _accelerated_scan_cache_key(**base) != _accelerated_scan_cache_key(**changed_shape)


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
    stats["scan_runner_explicit_lower_s"] = 0.3
    stats["scan_runner_explicit_compile_s"] = 0.4
    stats["scan_runner_explicit_compile_count"] = 1
    stats["scan_runner_explicit_compile_miss_count"] = 1
    stats["scan_runner_explicit_hlo_line_count"] = 20
    stats["scan_runner_explicit_hlo_instruction_count"] = 10
    stats["scan_runner_explicit_hlo_op_add_count"] = 2
    stats["scan_runner_arg_leaf_count"] = 5
    stats["scan_runner_arg_array_leaf_count"] = 3
    stats["scan_runner_arg_scalar_leaf_count"] = 2
    stats["scan_runner_arg_array_nbytes"] = 96
    stats["scan_runner_arg_path_arg0_state_leaf_count"] = 4
    stats["scan_runner_arg_path_arg0_state_array_leaf_count"] = 4
    stats["scan_runner_arg_path_arg0_state_array_nbytes"] = 80
    stats["scan_postprocess_s"] = 2.0
    stats["scan_runner_cache_hit_count"] = 3
    stats["scan_runner_cache_miss_category_iteration_budget_count"] = 2

    report = build_scan_timing_report(iterations=7, stats=stats, scan_total_s=5.9)

    assert report["iterations"] == 7
    assert report["scan_total_s"] == 5.9
    assert report["scan_unattributed_s"] == pytest.approx(0.3)
    assert report["scan_device_dispatch_s"] == pytest.approx(0.25)
    assert report["scan_device_ready_s"] == pytest.approx(0.75)
    assert report["scan_runner_cache_hit_count"] == 3
    assert report["scan_runner_cache_miss_count"] == 0
    assert report["scan_runner_cache_miss_category_iteration_budget_count"] == 2
    assert report["scan_runner_explicit_lower_s"] == pytest.approx(0.3)
    assert report["scan_runner_explicit_compile_s"] == pytest.approx(0.4)
    assert report["scan_runner_explicit_compile_count"] == 1
    assert report["scan_runner_explicit_compile_miss_count"] == 1
    assert report["scan_runner_explicit_hlo_line_count"] == 20
    assert report["scan_runner_explicit_hlo_instruction_count"] == 10
    assert report["scan_runner_explicit_hlo_op_add_count"] == 2
    assert report["scan_runner_arg_leaf_count"] == 5
    assert report["scan_runner_arg_array_leaf_count"] == 3
    assert report["scan_runner_arg_scalar_leaf_count"] == 2
    assert report["scan_runner_arg_array_nbytes"] == 96
    assert report["scan_runner_arg_path_arg0_state_leaf_count"] == 4
    assert report["scan_runner_arg_path_arg0_state_array_leaf_count"] == 4
    assert report["scan_runner_arg_path_arg0_state_array_nbytes"] == 80
    assert report["scan_cold_cache_miss_s"] == pytest.approx(0.6)
    assert report["scan_cold_cache_miss_ready_s"] == pytest.approx(0.2)
    assert report["scan_cache_build_wrapper_s"] == pytest.approx(0.1)
    assert build_scan_timing_report(iterations=7, stats=stats, scan_total_s=3.0)["scan_unattributed_s"] == 0.0


def test_default_vmec2000_controller_constants_match_legacy_values():
    constants = default_vmec2000_controller_constants()

    assert constants.preconditioner_update_interval == 25
    assert constants.restart_badjac_factor == pytest.approx(0.9)
    assert constants.restart_badprog_factor == pytest.approx(1.03)
    assert constants.vmec2000_fact == pytest.approx(1.0e4)
    assert constants.ndamp == 10


def test_scan_runtime_hooks_disable_optional_callbacks_for_quiet_defaults():
    hooks = resolve_scan_runtime_hooks(
        dump_timecontrol_env="0",
        dump_dir_env="",
        print_in_scan=False,
        scan_print_mode="not-a-mode",
        scan_trace=False,
    )

    assert not hooks.dump_timecontrol_scan
    assert hooks.timecontrol_callback is None
    assert hooks.timecontrol_path is None
    assert hooks.io_callback is None
    assert not hooks.print_in_scan
    assert hooks.scan_print_mode == "debug_print"
    assert not hooks.scan_trace
    with scan_trace_context_or_null(hooks, "scan/test") as value:
        assert value is None


def test_scan_runtime_hooks_disable_timecontrol_without_dump_dir(tmp_path):
    hooks = resolve_scan_runtime_hooks(
        dump_timecontrol_env="1",
        dump_dir_env="",
        print_in_scan=False,
        scan_print_mode="debug_print",
        scan_trace=False,
    )

    assert not hooks.dump_timecontrol_scan
    assert hooks.timecontrol_path is None

    hooks_with_path = resolve_scan_runtime_hooks(
        dump_timecontrol_env="1",
        dump_dir_env=str(tmp_path),
        print_in_scan=False,
        scan_print_mode="debug_print",
        scan_trace=False,
    )
    if hooks_with_path.timecontrol_callback is not None:
        assert hooks_with_path.dump_timecontrol_scan
        assert hooks_with_path.timecontrol_path == tmp_path / "time_control_trace.log"


def test_scan_runtime_hooks_from_env_matches_direct_resolver(tmp_path):
    hooks = resolve_scan_runtime_hooks_from_env(
        {
            "VMEC_JAX_DUMP_TIMECONTROL": "1",
            "VMEC_JAX_DUMP_DIR": str(tmp_path),
        },
        print_in_scan=False,
        scan_print_mode="debug_print",
        scan_trace=False,
    )

    direct = resolve_scan_runtime_hooks(
        dump_timecontrol_env="1",
        dump_dir_env=str(tmp_path),
        print_in_scan=False,
        scan_print_mode="debug_print",
        scan_trace=False,
    )

    assert hooks.dump_timecontrol_scan == direct.dump_timecontrol_scan
    assert hooks.timecontrol_path == direct.timecontrol_path
    assert hooks.print_in_scan == direct.print_in_scan
    assert hooks.scan_print_mode == direct.scan_print_mode


def test_get_or_build_scan_runner_records_miss_hit_and_bypass_paths():
    cache = OrderedDict()
    stats = {
        "scan_runner_cache_lookup_s": 0.0,
        "scan_runner_cache_build_s": 0.0,
        "scan_runner_cache_hit_count": 0,
        "scan_runner_cache_miss_count": 0,
        "scan_runner_cache_bypass_count": 0,
    }
    miss_records = []
    times = iter([10.0, 10.25, 11.0, 11.5, 12.0, 12.1])

    def jit_func(func):
        return ("jit", func)

    def cache_get(cache_obj, key):
        return cache_obj.get(key)

    def cache_put(cache_obj, key, value, *, env_name, default):
        assert env_name == "VMEC_JAX_SCAN_RUNNER_CACHE_SIZE"
        assert default == 32
        cache_obj[key] = value
        return value

    def record_miss(stats_obj, *, requested_key, existing_keys):
        miss_records.append((requested_key, existing_keys))
        stats_obj["scan_runner_cache_miss_category_test_count"] = (
            int(stats_obj.get("scan_runner_cache_miss_category_test_count", 0)) + 1
        )

    runner_miss, status_miss = get_or_build_scan_runner(
        "run",
        cache=cache,
        key=("case", 1),
        differentiating_scan=False,
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        jit_func=jit_func,
        cache_get=cache_get,
        cache_put=cache_put,
        record_miss_categories=record_miss,
        perf_counter=lambda: next(times),
    )
    runner_hit, status_hit = get_or_build_scan_runner(
        "run",
        cache=cache,
        key=("case", 1),
        differentiating_scan=False,
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        jit_func=jit_func,
        cache_get=cache_get,
        cache_put=cache_put,
        record_miss_categories=record_miss,
        perf_counter=lambda: next(times),
    )
    runner_bypass, status_bypass = get_or_build_scan_runner(
        "run2",
        cache=cache,
        key=("case", 2),
        differentiating_scan=True,
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        jit_func=jit_func,
        cache_get=lambda *_args: pytest.fail("differentiating path must bypass cache lookup"),
        cache_put=lambda *_args, **_kwargs: pytest.fail("differentiating path must bypass cache put"),
        record_miss_categories=record_miss,
        perf_counter=lambda: next(times),
    )

    assert status_miss == "miss"
    assert runner_miss == ("jit", "run")
    assert status_hit == "hit"
    assert runner_hit is runner_miss
    assert status_bypass == "bypass"
    assert runner_bypass == ("jit", "run2")
    assert stats["scan_runner_cache_lookup_s"] > 0.0
    assert stats["scan_runner_cache_build_s"] > 0.0
    assert stats["scan_runner_cache_miss_count"] == 1
    assert stats["scan_runner_cache_hit_count"] == 1
    assert stats["scan_runner_cache_bypass_count"] == 1
    assert stats["scan_runner_cache_miss_category_test_count"] == 1


def test_scan_explicit_compile_enabled_tokens():
    assert not scan_explicit_compile_enabled("")
    assert not scan_explicit_compile_enabled("off")
    assert scan_explicit_compile_enabled("1")
    assert scan_explicit_compile_enabled("diagnostic")


def test_scan_hlo_summary_enabled_tokens():
    assert not scan_hlo_summary_enabled("")
    assert not scan_hlo_summary_enabled("no")
    assert scan_hlo_summary_enabled("1")
    assert scan_hlo_summary_enabled("summary")


def test_summarize_scan_runner_hlo_text_counts_lines_instructions_and_ops():
    text = """
HloModule jit_scan
ENTRY main {
  %a = f64[2]{0} parameter(0)
  %b = f64[2]{0} parameter(1)
  %c = f64[2]{0} add(%a, %b)
  ROOT %d = f64[2]{0} multiply(%c, %b)
}
"""

    summary = summarize_scan_runner_hlo_text(text)

    assert summary["scan_runner_explicit_hlo_line_count"] == 7
    assert summary["scan_runner_explicit_hlo_instruction_count"] == 4
    assert summary["scan_runner_explicit_hlo_op_parameter_count"] == 2
    assert summary["scan_runner_explicit_hlo_op_add_count"] == 1
    assert summary["scan_runner_explicit_hlo_op_multiply_count"] == 1


def test_maybe_record_scan_runner_hlo_summary_records_or_reports_failure():
    class Hlo:
        def as_hlo_text(self):
            return "%a = f64[] parameter(0)\nROOT %b = f64[] negate(%a)\n"

    class Lowered:
        def compiler_ir(self, *, dialect):
            assert dialect == "hlo"
            return Hlo()

    stats = {}
    maybe_record_scan_runner_hlo_summary(
        Lowered(),
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        env_value="1",
    )
    assert stats["scan_runner_explicit_hlo_line_count"] == 2
    assert stats["scan_runner_explicit_hlo_instruction_count"] == 2
    assert stats["scan_runner_explicit_hlo_op_parameter_count"] == 1
    assert stats["scan_runner_explicit_hlo_op_negate_count"] == 1

    class BadLowered:
        def compiler_ir(self, *, dialect):
            raise RuntimeError("no hlo")

    maybe_record_scan_runner_hlo_summary(
        BadLowered(),
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        env_value="1",
    )
    assert stats["scan_runner_explicit_hlo_failure_count"] == 1


def test_record_scan_runner_arg_summary_counts_leaves_and_array_bytes():
    pair = namedtuple("Pair", "left right")
    arr = jnp.ones((2, 3), dtype=jnp.float64)
    small = jnp.zeros((1,), dtype=jnp.float64)
    stats = {}

    record_scan_runner_arg_summary(
        (pair(left=arr, right={"scalar": 3}), [small]),
        scan_timing_enabled=True,
        scan_timing_stats=stats,
    )

    assert stats["scan_runner_arg_leaf_count"] == 3
    assert stats["scan_runner_arg_array_leaf_count"] == 2
    assert stats["scan_runner_arg_scalar_leaf_count"] == 1
    assert stats["scan_runner_arg_array_nbytes"] == arr.nbytes + small.nbytes
    assert stats["scan_runner_arg_path_arg0_left_leaf_count"] == 1
    assert stats["scan_runner_arg_path_arg0_left_array_leaf_count"] == 1
    assert stats["scan_runner_arg_path_arg0_left_array_nbytes"] == arr.nbytes
    assert stats["scan_runner_arg_path_arg0_right_leaf_count"] == 1
    assert stats["scan_runner_arg_path_arg0_right_array_leaf_count"] == 0
    assert stats["scan_runner_arg_path_arg0_right_array_nbytes"] == 0
    assert stats["scan_runner_arg_path_arg1_0_leaf_count"] == 1
    assert stats["scan_runner_arg_path_arg1_0_array_nbytes"] == small.nbytes
    assert stats["scan_runner_arg_category_other_leaf_count"] == 2
    assert stats["scan_runner_arg_category_other_array_leaf_count"] == 1
    assert stats["scan_runner_arg_category_other_array_nbytes"] == arr.nbytes
    assert stats["scan_runner_arg_category_iteration_input_leaf_count"] == 1
    assert stats["scan_runner_arg_category_iteration_input_array_leaf_count"] == 1
    assert stats["scan_runner_arg_category_iteration_input_array_nbytes"] == small.nbytes


def test_record_scan_runner_arg_summary_reports_compact_rz_preconditioner_carry():
    """Preconditioner carry diagnostics should flag apply-only R/Z matrices."""

    carry = namedtuple("Carry", "cache_prec_rz_mats")
    mats = {
        "ar": jnp.ones((2, 2), dtype=jnp.float64),
        "br": jnp.ones((2, 2), dtype=jnp.float64),
        "dr": jnp.ones((2, 2), dtype=jnp.float64),
        "az": jnp.ones((2, 2), dtype=jnp.float64),
        "bz": jnp.ones((2, 2), dtype=jnp.float64),
        "dz": jnp.ones((2, 2), dtype=jnp.float64),
        "m1_fac_r": jnp.ones((2,), dtype=jnp.float64),
        "m1_fac_z": jnp.ones((2,), dtype=jnp.float64),
    }
    stats = {}

    record_scan_runner_arg_summary(
        (carry(cache_prec_rz_mats=mats), jnp.arange(3)),
        scan_timing_enabled=True,
        scan_timing_stats=stats,
    )

    assert stats["scan_runner_arg_preconditioner_rz_mats_key_count"] == 8
    assert stats["scan_runner_arg_preconditioner_rz_mats_unexpected_key_count"] == 0
    assert stats["scan_runner_arg_preconditioner_rz_mats_missing_mandatory_key_count"] == 0
    assert stats["scan_runner_arg_preconditioner_rz_mats_compact_ok_count"] == 1


def test_record_scan_runner_arg_summary_flags_reassembly_rz_preconditioner_carry():
    """Reassembly coefficients in the scan carry should be visible as regressions."""

    carry = namedtuple("Carry", "cache_prec_rz_mats")
    mats = {
        "ar": jnp.ones((2, 2), dtype=jnp.float64),
        "br": jnp.ones((2, 2), dtype=jnp.float64),
        "dr": jnp.ones((2, 2), dtype=jnp.float64),
        "az": jnp.ones((2, 2), dtype=jnp.float64),
        "bz": jnp.ones((2, 2), dtype=jnp.float64),
        "dz": jnp.ones((2, 2), dtype=jnp.float64),
        "m1_fac_r": jnp.ones((2,), dtype=jnp.float64),
        "arm_parity": jnp.ones((2, 2), dtype=jnp.float64),
    }
    stats = {}

    record_scan_runner_arg_summary(
        (carry(cache_prec_rz_mats=mats), jnp.arange(3)),
        scan_timing_enabled=True,
        scan_timing_stats=stats,
    )

    assert stats["scan_runner_arg_preconditioner_rz_mats_key_count"] == 8
    assert stats["scan_runner_arg_preconditioner_rz_mats_unexpected_key_count"] == 1
    assert stats["scan_runner_arg_preconditioner_rz_mats_missing_mandatory_key_count"] == 1
    assert stats["scan_runner_arg_preconditioner_rz_mats_compact_ok_count"] == 0


def test_record_scan_history_summary_counts_materialized_history_bytes():
    """Scan history diagnostics should separate output rows from carry inputs."""

    hist = (
        jnp.ones((4,), dtype=jnp.float64),
        2 * jnp.ones((4,), dtype=jnp.float64),
        {"accepted": jnp.ones((4,), dtype=bool), "label": "light"},
    )
    stats = {}

    record_scan_history_summary(hist, scan_timing_enabled=True, scan_timing_stats=stats)

    assert stats["scan_history_none"] == 0
    assert stats["scan_history_leaf_count"] == 4
    assert stats["scan_history_array_leaf_count"] == 3
    assert stats["scan_history_scalar_leaf_count"] == 1
    assert stats["scan_history_array_nbytes"] == (
        hist[0].nbytes + hist[1].nbytes + hist[2]["accepted"].nbytes
    )


def test_record_scan_history_summary_reports_state_only_history():
    """State-only scans should report that no history tree was materialized."""

    stats = {}

    record_scan_history_summary(None, scan_timing_enabled=True, scan_timing_stats=stats)

    assert stats["scan_history_none"] == 1
    assert stats["scan_history_leaf_count"] == 0
    assert stats["scan_history_array_leaf_count"] == 0
    assert stats["scan_history_array_nbytes"] == 0


def test_scan_arg_summary_is_independent_from_explicit_compile():
    stats = {}
    arr = jnp.ones((4,), dtype=jnp.float64)

    assert not scan_arg_summary_enabled("0")
    assert scan_arg_summary_enabled("1")

    maybe_record_scan_runner_arg_summary(
        (arr,),
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        env_value="0",
    )
    assert "scan_runner_arg_leaf_count" not in stats

    maybe_record_scan_runner_arg_summary(
        (arr,),
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        env_value="1",
    )
    assert stats["scan_runner_arg_leaf_count"] == 1
    assert stats["scan_runner_arg_array_leaf_count"] == 1
    assert stats["scan_runner_arg_array_nbytes"] == arr.nbytes


def test_maybe_explicit_compile_scan_runner_records_success_and_returns_compiled():
    events = []
    stats = {
        "scan_runner_explicit_lower_s": 0.0,
        "scan_runner_explicit_compile_s": 0.0,
        "scan_runner_explicit_compile_count": 0,
        "scan_runner_explicit_compile_failure_count": 0,
    }
    times = iter([1.0, 1.25, 2.0, 2.75])

    class Lowered:
        def compile(self):
            events.append("compile")

            def compiled(*args):
                events.append(("compiled", args))
                return "compiled-result"

            return compiled

    class Runner:
        def lower(self, *args):
            events.append(("lower", args))
            return Lowered()

        def __call__(self, *args):
            events.append(("runner", args))
            return "runner-result"

    compiled = maybe_explicit_compile_scan_runner(
        Runner(),
        ("carry", "it"),
        cache_status="miss",
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        perf_counter=lambda: next(times),
        env_value="1",
    )

    assert compiled("carry", "it") == "compiled-result"
    assert events == [("lower", ("carry", "it")), "compile", ("compiled", ("carry", "it"))]
    assert stats["scan_runner_explicit_lower_s"] == pytest.approx(0.25)
    assert stats["scan_runner_explicit_compile_s"] == pytest.approx(0.75)
    assert stats["scan_runner_explicit_compile_count"] == 1
    assert stats["scan_runner_explicit_compile_miss_count"] == 1


def test_maybe_explicit_compile_scan_runner_can_record_args_without_compiling():
    events = []
    stats = {}
    arr = jnp.ones((2,), dtype=jnp.float64)

    class Runner:
        def lower(self, *_args):
            events.append("lower")
            raise AssertionError("lower should not run")

        def __call__(self, *args):
            events.append(("runner", args))
            return "runner-result"

    runner = Runner()
    returned = maybe_explicit_compile_scan_runner(
        runner,
        (arr,),
        cache_status="hit",
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        perf_counter=lambda: 1.0,
        env_value="0",
        arg_summary_env_value="1",
    )

    assert returned is runner
    assert returned(arr) == "runner-result"
    assert events == [("runner", (arr,))]
    assert stats["scan_runner_arg_leaf_count"] == 1
    assert stats["scan_runner_arg_array_leaf_count"] == 1
    assert "scan_runner_explicit_compile_count" not in stats


def test_maybe_explicit_compile_scan_runner_is_noop_or_safe_on_failure():
    stats = {"scan_runner_explicit_compile_failure_count": 0}

    class Runner:
        def lower(self, *_args):
            raise RuntimeError("cannot lower")

        def __call__(self, *args):
            return ("runner", args)

    runner = Runner()
    assert (
        maybe_explicit_compile_scan_runner(
            runner,
            ("carry",),
            cache_status="miss",
            scan_timing_enabled=True,
            scan_timing_stats=stats,
            perf_counter=lambda: 1.0,
            env_value="0",
        )
        is runner
    )
    assert (
        maybe_explicit_compile_scan_runner(
            runner,
            ("carry",),
            cache_status="miss",
            scan_timing_enabled=True,
            scan_timing_stats=stats,
            perf_counter=lambda: 1.0,
            env_value="1",
        )
        is runner
    )
    assert stats["scan_runner_explicit_compile_failure_count"] == 1


def test_select_initial_rz_norm_func_uses_host_for_nontraced_state() -> None:
    state = SimpleNamespace(kind="state")
    calls = {"host": 0, "jax": 0}

    def host_norm(arg):
        assert arg is state
        calls["host"] += 1
        return 7.0

    def jax_norm(_arg):
        calls["jax"] += 1
        return jnp.asarray(9.0)

    selected = _select_initial_rz_norm_func(
        state_init=state,
        rz_norm_func=jax_norm,
        rz_norm_np_func=host_norm,
        tree_has_tracer=lambda _value: False,
        dtype=jnp.float64,
    )

    assert float(selected(state)) == pytest.approx(7.0)
    assert calls == {"host": 1, "jax": 0}


def test_select_initial_rz_norm_func_keeps_jax_for_traced_state() -> None:
    state = SimpleNamespace(kind="state")

    def jax_norm(_arg):
        return jnp.asarray(11.0)

    selected = _select_initial_rz_norm_func(
        state_init=state,
        rz_norm_func=jax_norm,
        rz_norm_np_func=lambda _arg: 7.0,
        tree_has_tracer=lambda _value: True,
        dtype=jnp.float64,
    )

    assert selected is jax_norm
    assert float(selected(state)) == pytest.approx(11.0)


def test_select_initial_rz_norm_func_falls_back_to_jax_on_host_failure() -> None:
    state = SimpleNamespace(kind="state")

    def host_norm(_arg):
        raise RuntimeError("synthetic host norm failure")

    def jax_norm(_arg):
        return jnp.asarray(13.0)

    selected = _select_initial_rz_norm_func(
        state_init=state,
        rz_norm_func=jax_norm,
        rz_norm_np_func=host_norm,
        tree_has_tracer=lambda _value: False,
        dtype=jnp.float64,
    )

    assert float(selected(state)) == pytest.approx(13.0)


def test_run_scan_preflight_step_handles_nojit_and_offset_with_timing():
    Carry = namedtuple("Carry", "iter_offset value")
    times = iter([1.0, 1.4])
    stats = {"scan_preflight_s": 0.0}
    block_calls = []

    class Jnp:
        int32 = "int32"

        @staticmethod
        def asarray(value, dtype=None):
            return ("arr", value, dtype)

    class Jax:
        class tree_util:
            @staticmethod
            def tree_map(fn, value):
                return tuple(fn(item) for item in value)

        @staticmethod
        def disable_jit():
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return None

            return Context()

    def scan_step(carry, it):
        assert carry.iter_offset == ("arr", 7, "int32")
        return carry._replace(value="advanced"), ("h0", it)

    result = run_scan_preflight_step(
        Carry(iter_offset=None, value="initial"),
        iter_offset_preflight=7,
        jit_preflight=False,
        get_scan_runner=lambda _seq_len: pytest.fail("nojit preflight should not request runner"),
        scan_step=scan_step,
        scan_timing_enabled=True,
        scan_timing_stats=stats,
        block_scan_value=lambda value: block_calls.append(value) or value,
        perf_counter=lambda: next(times),
        jnp_module=Jnp,
        jax_module=Jax,
    )

    assert result.carry.value == "advanced"
    assert result.history_row == ("h0", ("arr", 0, "int32"))
    assert block_calls == [(result.carry, result.history_row)]
    assert stats["scan_preflight_s"] == pytest.approx(0.4)


def test_run_scan_preflight_step_handles_jitted_sequence_history():
    Carry = namedtuple("Carry", "iter_offset value")

    class Jnp:
        int32 = "int32"

        @staticmethod
        def asarray(value, dtype=None):
            return tuple(value) if isinstance(value, list) else ("arr", value, dtype)

    class Jax:
        class tree_util:
            @staticmethod
            def tree_map(fn, value):
                return tuple(fn(item) for item in value)

        @staticmethod
        def disable_jit():
            raise AssertionError("jit preflight should not disable jit")

    def get_scan_runner(seq_len):
        assert seq_len == 1

        def runner(carry, it_seq):
            assert it_seq == (0,)
            return carry._replace(value="jitted"), (("fsqr0",), ("fsqz0",), ("fsql0",))

        return runner, "miss"

    result = run_scan_preflight_step(
        Carry(iter_offset=None, value="initial"),
        iter_offset_preflight=None,
        jit_preflight=True,
        get_scan_runner=get_scan_runner,
        scan_step=lambda *_args: pytest.fail("jit preflight should use runner"),
        scan_timing_enabled=False,
        scan_timing_stats={},
        block_scan_value=lambda value: pytest.fail("timing disabled should not block"),
        perf_counter=lambda: pytest.fail("timing disabled should not request time"),
        jnp_module=Jnp,
        jax_module=Jax,
    )

    assert result.carry.value == "jitted"
    assert result.history_row == ("fsqr0", "fsqz0", "fsql0")


def test_run_nonchunked_scan_passes_compact_iteration_sequence_and_scalar_controls():
    calls = []

    class Runtime:
        @staticmethod
        def ready(*_args, **_kwargs):
            pytest.fail("timing disabled should not synchronize device values")

    def get_scan_runner(seq_len):
        assert seq_len == 5

        def runner(carry, it_seq, ftol, target):
            calls.append((it_seq, ftol, target))
            return carry + ("advanced",), (
                jnp.ones((5,), dtype=jnp.float64),
                2 * jnp.ones((5,), dtype=jnp.float64),
                3 * jnp.ones((5,), dtype=jnp.float64),
            )

        return runner, "miss"

    result = run_nonchunked_scan(
        ("carry",),
        max_iter_scan=5,
        max_iter_tail=5,
        preflight_iters=0,
        iter_offset_preflight=0,
        axis_reset_repeat=False,
        iter_offset0=0,
        get_scan_runner=get_scan_runner,
        scan_step=lambda *_args: pytest.fail("no preflight should not call scan_step directly"),
        runtime_scan_args=(jnp.asarray(1.0e-8), jnp.asarray(jnp.nan)),
        scan_jit_preflight_enabled_func=lambda **_kwargs: pytest.fail("no preflight should not probe jit policy"),
        scan_jit_preflight_env=None,
        backend_name="cpu",
        scan_differentiated=False,
        scan_collect_print=False,
        scan_timing_enabled=False,
        scan_timing_stats={},
        scan_device_runtime=Runtime(),
        perf_counter=lambda: pytest.fail("timing disabled should not request time"),
        state_only_scan=False,
        scan_fallback_enabled_run=False,
        scan_fallback_iters=0,
        jnp_module=jnp,
        jax_module=SimpleNamespace(tree_util=None),
    )

    assert result.carry_final == ("carry", "advanced")
    assert len(calls) == 1
    it_seq, ftol, target = calls[0]
    assert tuple(np.asarray(it_seq)) == (0, 1, 2, 3, 4)
    assert np.shape(ftol) == ()
    assert np.shape(target) == ()
    assert float(ftol) == pytest.approx(1.0e-8)
    assert np.isnan(float(target))


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


def test_vmec2000_scan_setup_resolves_state_only_and_preconditioner_overrides():
    setup = resolve_vmec2000_scan_setup(
        env={
            "VMEC_JAX_SCAN_LIGHT": "1",
            "VMEC_JAX_SCAN_PRINT": "1",
            "VMEC_JAX_SCAN_PRECOND_PRECOMPUTE": "",
            "VMEC_JAX_SCAN_PRECOND_LAXTRIDI": "",
            "VMEC_JAX_TRIDI_PRECOMPUTE": "0",
            "VMEC_JAX_TRIDI_SOLVE": "0",
        },
        state_only=True,
        scan_differentiated=False,
        scan_fallback_enabled=True,
        force_chunked_scan=True,
        indata_nstep=4,
        preconditioner_use_precomputed_tridi=True,
        preconditioner_use_lax_tridi=True,
        verbose=True,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
        light_history=False,
        scan_minimal_default=False,
        dump_any=False,
        fsq_total_target=None,
        backend_name="gpu",
    )

    assert setup.state_only_scan
    assert not setup.scan_fallback_enabled_run
    assert setup.force_chunked_scan_run
    assert setup.nstep_screen == 4
    assert setup.options.scan_minimal
    assert not setup.options.scan_collect_scalars
    assert not setup.options.print_in_scan
    assert setup.options.scan_use_precomputed
    assert setup.options.scan_use_lax_tridi


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
    assert not default_cpu_lax.scan_use_precomputed
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
    assert scan_jit_preflight_enabled(env_value=None, backend_name="cpu", scan_differentiated=False)
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
    assert base[0] == "vmec2000_scan_v10"
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
    assert _cache_key(stage_prev_fsq=3)[18] is True
    assert _cache_key(stage_prev_fsq=3) == _cache_key(stage_prev_fsq=4)
    assert _cache_key(stage_transition_factor=20.0) == base
    assert _cache_key(stage_transition_scale=0.25) == base
    state_only = _cache_key(
        state_only_scan=True,
        scan_light=False,
        scan_minimal=True,
        scan_fallback_iters=7,
        scan_fallback_badjac_limit=2,
    )
    state_only_tuned = _cache_key(
        state_only_scan=True,
        scan_light=False,
        scan_minimal=True,
        scan_fallback_iters=99,
        scan_fallback_badjac_limit=50,
        nstep_screen=200,
    )
    assert state_only == state_only_tuned
    assert state_only == _cache_key(state_only_scan=True, scan_light=True, scan_minimal=False)
    assert state_only[-2:] == (0, 0)
    assert state_only[12] == 0
    assert state_only[20:23] == (True, False, True)


def test_scan_iteration_runtime_plan_resolves_offsets_and_cache_key():
    plan = resolve_scan_iteration_runtime_plan(
        env={
            "VMEC_JAX_SCAN_PREFLIGHT": "2",
            "VMEC_JAX_SCAN_EXTRA_ITERS": "3",
            "VMEC_JAX_TOMNSPS_FFT": "yes",
            "VMEC_JAX_TOMNSPS_FFT_FUSED": "0",
            "VMEC_JAX_TOMNSPS_THETA_FUSED": "1",
            "VMEC_JAX_TOMNSPS_ZETA_FUSED": "1",
        },
        jit_forces_scan=False,
        vmec2000_control=True,
        max_iter=10,
        axis_reset_repeat=True,
        iter_offset0=4,
        static_key=("static",),
        wout_key=("wout",),
        edge_signature_key=("edge",),
        step_size=0.2,
        initial_flip_sign=-1.0,
        lambda_update_scale=0.5,
        ftol=1.0e-12,
        fsq_total_target=None,
        nstep_screen=25,
        use_restart_triggers=True,
        vmecpp_restart=False,
        scan_use_precomputed=True,
        scan_use_lax_tridi=True,
        scan_use_restart_payload=True,
        stage_prev_fsq=None,
        stage_transition_factor=1.0,
        stage_transition_scale=2.0,
        state_only_scan=False,
        scan_light=True,
        scan_minimal=False,
        scan_fallback_iters=7,
        scan_fallback_accept_frac=0.5,
        scan_fallback_fsq_factor=3.0,
        scan_fallback_badjac_limit=2,
        scan_fallback_fsq_abs=1.0e-4,
    )

    assert plan.preflight_iters == 2
    assert plan.max_iter_scan == 13
    assert plan.max_iter_tail == 11
    assert plan.iter_offset_preflight == 0
    assert plan.iter_offset0 == -1
    assert plan.axis_reset_repeated
    assert plan.scan_cache_key[1:4] == (("static",), ("wout",), ("edge",))
    assert plan.scan_cache_key[4] == ("yes", "0", "1", "1")
    assert plan.scan_cache_key[5:8] == (11, 2, -1)
    assert plan.scan_cache_key[15:18] == (True, True, True)


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

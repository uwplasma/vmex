import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.performance import (
    accumulate_scan_device_ready_timing,
    exact_parameter_cache_key,
    exact_parameter_cache_key_fingerprint,
    explain_scan_cache_key_delta,
    replay_timing_breakdown,
    scan_cache_miss_category_counts,
    scan_cache_key_delta_summary,
    scan_cache_key_field_names,
)
from vmec_jax.solvers.fixed_boundary.scan.planning import build_vmec2000_scan_cache_key


def _scan_cache_key(**overrides):
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


def test_exact_parameter_cache_key_flattens_to_float64_bytes():
    matrix_params = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    vector_params = [1.0, 2.0, 3.0, 4.0]

    assert exact_parameter_cache_key(matrix_params) == exact_parameter_cache_key(vector_params)
    assert exact_parameter_cache_key([1.0, 2.0]) != exact_parameter_cache_key([1.0, 2.0 + 1.0e-12])

    fingerprint = exact_parameter_cache_key_fingerprint(matrix_params)
    assert fingerprint["dtype"] == "float64"
    assert fingerprint["n_params"] == 4
    assert fingerprint["byte_length"] == 4 * np.dtype(np.float64).itemsize
    assert fingerprint["cache_key"] == exact_parameter_cache_key(vector_params)


def test_scan_cache_key_delta_labels_behavioral_toggles():
    base = _scan_cache_key()
    changed = _scan_cache_key(
        max_iter_tail=10,
        stage_prev_fsq=3.0,
        scan_use_precomputed=True,
        scan_use_lax_tridi=True,
        scan_light=True,
        scan_minimal=False,
    )

    deltas = explain_scan_cache_key_delta(base, changed)

    assert [(delta.index, delta.field) for delta in deltas] == [
        (5, "max_iter_tail"),
        (15, "scan_use_precomputed"),
        (16, "scan_use_lax_tridi"),
        (18, "stage_prev_fsq"),
        (23, "scan_light"),
        (24, "scan_minimal"),
    ]
    assert deltas[3].before is None
    assert deltas[3].after == 3.0


def test_scan_cache_key_delta_summary_groups_cache_miss_causes():
    base = _scan_cache_key()
    changed = _scan_cache_key(
        ftol=1.0e-10,
        max_iter_tail=12,
        scan_use_precomputed=True,
        scan_fallback_iters=40,
        stage_transition_scale=0.25,
    )

    summary = scan_cache_key_delta_summary(base, changed)

    assert summary["changed"] is True
    assert summary["n_changed"] == 5
    assert summary["fields"] == (
        "max_iter_tail",
        "ftol",
        "scan_use_precomputed",
        "stage_transition_scale",
        "scan_fallback_iters",
    )
    assert summary["categories"] == (
        "iteration_budget",
        "tolerance",
        "scan_policy",
        "stage_transition",
        "fallback_policy",
    )
    assert summary["category_fields"]["iteration_budget"] == ("max_iter_tail",)
    assert summary["category_fields"]["tolerance"] == ("ftol",)
    assert summary["category_fields"]["scan_policy"] == ("scan_use_precomputed",)
    assert summary["category_fields"]["stage_transition"] == ("stage_transition_scale",)
    assert summary["category_fields"]["fallback_policy"] == ("scan_fallback_iters",)

    unchanged = scan_cache_key_delta_summary(base, base)
    assert unchanged == {
        "changed": False,
        "n_changed": 0,
        "fields": (),
        "categories": (),
        "category_fields": {},
    }


def test_scan_cache_miss_category_counts_identifies_nearest_cached_key():
    base = _scan_cache_key(max_iter_tail=9, ftol=1.0e-12)
    unrelated = ("scan_v1", ("static",), ("wout",), ("edge",), 3, 0.1, -1.0, 1.0, 0.5, 0.5, True, True)
    requested = _scan_cache_key(max_iter_tail=15, ftol=1.0e-10)

    assert scan_cache_key_field_names(base + (7,))[-1] == "seq_len"
    assert scan_cache_key_field_names(unrelated)[0] == "schema"
    assert scan_cache_key_field_names(("custom_schema", "a", "b")) == ("field_0", "field_1", "field_2")
    assert scan_cache_miss_category_counts(requested, []) == {"cold_empty": 1}
    assert scan_cache_miss_category_counts(requested, [unrelated]) == {"schema": 1}
    assert scan_cache_miss_category_counts(base, [base]) == {"unknown": 1}

    counts = scan_cache_miss_category_counts(requested, [unrelated, base])

    assert counts == {"iteration_budget": 1, "tolerance": 1}


def test_replay_timing_breakdown_prefers_total_and_falls_back_to_split():
    profile = {
        "jacobian_tape_replay": {"count": 2, "wall_time_s": 3.5},
        "jacobian_tape_replay_dispatch": {"count": 2, "wall_time_s": 0.4},
        "jacobian_tape_replay_ready": {"count": 2, "wall_time_s": 2.6},
    }

    breakdown = replay_timing_breakdown(profile, prefix="jacobian")

    assert breakdown == {
        "total_s": 3.5,
        "dispatch_s": 0.4,
        "ready_s": 2.6,
        "split_total_s": 3.0,
        "count": 2,
    }

    split_only = {
        "state_tangent_tape_replay_dispatch": {"count": 1, "wall_time_s": 0.25},
        "state_tangent_tape_replay_ready": {"count": 1, "wall_time_s": 0.75},
    }
    assert replay_timing_breakdown(split_only, prefix="state_tangent")["total_s"] == pytest.approx(1.0)


def test_accumulate_scan_device_ready_timing_is_safe_for_missing_start():
    stats: dict[str, float] = {}

    assert not accumulate_scan_device_ready_timing(
        stats,
        start=None,
        dispatch_done=11.0,
        ready_done=13.0,
    )
    assert stats == {}

    assert accumulate_scan_device_ready_timing(
        stats,
        start=10.0,
        dispatch_done=11.5,
        ready_done=14.0,
    )
    assert stats["scan_device_dispatch_s"] == pytest.approx(1.5)
    assert stats["scan_device_ready_s"] == pytest.approx(2.5)
    assert stats["scan_device_run_s"] == pytest.approx(4.0)

    assert accumulate_scan_device_ready_timing(
        stats,
        start=20.0,
        dispatch_done=20.25,
        ready_done=21.0,
    )
    assert stats["scan_device_dispatch_s"] == pytest.approx(1.75)
    assert stats["scan_device_ready_s"] == pytest.approx(3.25)
    assert stats["scan_device_run_s"] == pytest.approx(5.0)

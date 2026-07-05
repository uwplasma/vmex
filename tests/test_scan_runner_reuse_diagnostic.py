"""Unit tests for the fixed-boundary scan-runner reuse diagnostic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.diagnostics import check_scan_runner_reuse as reuse


def test_replace_ftol_prefers_ftol_array_and_preserves_comment() -> None:
    """Changing FTOL_ARRAY should leave the rest of the input deck untouched."""

    text = "  FTOL_ARRAY  = 1e-13  ! convergence\n  FTOL = 1e-8\n"
    updated = reuse._replace_ftol(text, 1.0e-9)

    assert "FTOL_ARRAY  = 1.0000000000000001e-09  ! convergence" in updated
    assert "  FTOL = 1e-8" in updated


def test_replace_ftol_falls_back_to_scalar_ftol() -> None:
    """Input decks without FTOL_ARRAY still support scalar FTOL replacement."""

    updated = reuse._replace_ftol("&INDATA\n  FTOL = 1e-8\n/\n", 1.0e-10)

    assert "FTOL = 1e-10" not in updated
    assert "FTOL = 1.0000000000000000e-10" in updated


def test_replace_ftol_requires_a_tolerance_line() -> None:
    """A clear error is better than silently generating an unchanged deck."""

    with pytest.raises(ValueError, match="FTOL_ARRAY or FTOL"):
        reuse._replace_ftol("&INDATA\n  NITER = 3\n/\n", 1.0e-9)


def test_env_override_helpers_restore_existing_and_missing_values(monkeypatch) -> None:
    """Fallback-threshold diagnostics must not leak process environment changes."""

    monkeypatch.setenv("VMEC_JAX_SCAN_FALLBACK_FSQ_FACTOR", "50")
    monkeypatch.delenv("VMEC_JAX_SCAN_FALLBACK_IMPROVE", raising=False)

    previous = reuse._set_env_overrides(
        {
            "VMEC_JAX_SCAN_FALLBACK_FSQ_FACTOR": "17",
            "VMEC_JAX_SCAN_FALLBACK_IMPROVE": "0.03",
        }
    )

    assert previous == {
        "VMEC_JAX_SCAN_FALLBACK_FSQ_FACTOR": "50",
        "VMEC_JAX_SCAN_FALLBACK_IMPROVE": None,
    }
    assert reuse.os.environ["VMEC_JAX_SCAN_FALLBACK_FSQ_FACTOR"] == "17"
    assert reuse.os.environ["VMEC_JAX_SCAN_FALLBACK_IMPROVE"] == "0.03"

    reuse._restore_env(previous)

    assert reuse.os.environ["VMEC_JAX_SCAN_FALLBACK_FSQ_FACTOR"] == "50"
    assert "VMEC_JAX_SCAN_FALLBACK_IMPROVE" not in reuse.os.environ


def test_stage_prev_fsq_cache_key_probe_keeps_numeric_value_dynamic() -> None:
    """Continuation-stage residual values should not create one runner per value."""

    probe = reuse._stage_prev_fsq_cache_key_probe()

    assert probe["passed"] is True
    assert probe["schema"] == "vmec2000_scan_v9"
    assert probe["inactive_to_active_delta"]["fields"] == ("has_stage_prev_fsq",)
    assert probe["active_value_delta"]["changed"] is False


def test_cache_counts_missing_timing_defaults_to_zero() -> None:
    """Diagnostic reports should be stable even if timing is absent."""

    result = SimpleNamespace(diagnostics={})

    assert reuse._cache_counts(result) == {"hit": 0, "miss": 0, "bypass": 0}


def test_cache_counts_extracts_scan_runner_fields() -> None:
    """Only the scan-runner hit/miss/bypass counters are surfaced."""

    result = SimpleNamespace(
        diagnostics={
            "timing": {
                "scan_runner_cache_hit_count": 2,
                "scan_runner_cache_miss_count": 1,
                "scan_runner_cache_bypass_count": 0,
                "scan_total_s": 3.4,
            }
        }
    )

    assert reuse._cache_counts(result) == {"hit": 2, "miss": 1, "bypass": 0}


def test_scan_timing_snapshot_extracts_hit_miss_dispatch_fields() -> None:
    """Reuse diagnostics should expose dispatch/ready splits for cache-status analysis."""

    result = SimpleNamespace(
        diagnostics={
            "timing": {
                "scan_total_s": 3.0,
                "scan_initial_compute_forces_s": 0.5,
                "scan_run_setup_s": 0.7,
                "scan_device_run_s": 2.0,
                "scan_device_dispatch_s": 1.99,
                "scan_device_ready_s": 0.01,
                "scan_runner_cache_lookup_s": "0.001",
                "scan_runner_cache_build_s": 0.002,
                "scan_runner_cache_hit_dispatch_s": 0.0,
                "scan_runner_cache_hit_ready_s": 0.0,
                "scan_runner_cache_miss_dispatch_s": 1.99,
                "scan_runner_cache_miss_ready_s": 0.01,
            }
        }
    )

    timing = reuse._scan_timing_snapshot(result)

    assert timing["scan_total_s"] == 3.0
    assert timing["scan_runner_cache_lookup_s"] == 0.001
    assert timing["scan_runner_cache_miss_dispatch_s"] == 1.99
    assert timing["scan_runner_cache_miss_ready_s"] == 0.01


def test_scan_timing_snapshot_missing_or_bad_values_default_to_zero() -> None:
    """Malformed profile timing should not break the reuse diagnostic."""

    result = SimpleNamespace(diagnostics={"timing": {"scan_total_s": object()}})

    timing = reuse._scan_timing_snapshot(result)

    assert timing["scan_total_s"] == 0.0
    assert timing["scan_device_dispatch_s"] == 0.0

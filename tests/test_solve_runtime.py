from __future__ import annotations

import pytest

import vmec_jax._solve_runtime as runtime


def test_scan_chunk_settings_invalid_env_uses_screen_floor(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_SCAN_CHUNK_SIZE", "not-an-int")

    chunk_size, cap_to_remaining = runtime._scan_chunk_settings(
        max_iter_scan=20,
        nstep_screen=0,
        need_print=False,
        lthreed=True,
        backend_name="gpu",
    )

    assert chunk_size == 1
    assert cap_to_remaining is True


def test_scan_chunk_settings_printing_uses_screen_budget(monkeypatch):
    monkeypatch.delenv("VMEC_JAX_SCAN_CHUNK_SIZE", raising=False)

    chunk_size, cap_to_remaining = runtime._scan_chunk_settings(
        max_iter_scan=100,
        nstep_screen=7,
        need_print=True,
        lthreed=False,
        backend_name="cpu",
    )

    assert chunk_size == 7
    assert cap_to_remaining is False


@pytest.mark.parametrize(
    ("scan_core_env", "scan_minimal", "fsq_total_target", "expected"),
    [
        ("", True, None, False),
        ("", False, 1.0, False),
        ("", True, 1.0, True),
        ("false", True, 1.0, False),
        ("yes", False, None, True),
    ],
)
def test_default_scan_core_policy(scan_core_env, scan_minimal, fsq_total_target, expected):
    assert (
        runtime._default_scan_core(
            scan_core_env=scan_core_env,
            scan_minimal=scan_minimal,
            fsq_total_target=fsq_total_target,
        )
        is expected
    )


def test_scan_fallback_policy_false_override_and_parse_defaults():
    policy = runtime._scan_fallback_policy(
        backend_name="cpu",
        enabled_env=" false ",
        iters_env="0",
        badjac_limit_env="bad",
        fsq_abs_env="bad",
        accept_frac_env="-0.25",
        fsq_factor_env="bad",
        improve_env="1.0",
    )

    assert policy.enabled is False
    assert policy.iters == 1
    assert policy.badjac_limit == 10
    assert policy.fsq_abs == 1.0e-2
    assert policy.accept_frac == 0.0
    assert policy.fsq_factor == 50.0
    assert policy.improve == 0.9


def test_residual_convergence_flags_use_exact_thresholds_and_total_target():
    assert runtime._residual_convergence_flags(
        fsqr=1.0,
        fsqz=1.0,
        fsql=1.0,
        ftol=1.0,
        fsq_total_target=2.0,
    ) == (True, False, True)
    assert runtime._residual_convergence_flags(
        fsqr=2.0,
        fsqz=0.5,
        fsql=0.5,
        ftol=1.0,
        fsq_total_target=3.0,
    ) == (False, True, True)


def test_runtime_env_enabled_treats_none_as_nonempty_string():
    assert runtime._runtime_env_enabled(None) is True


def test_hash_array_bytes_reports_opaque_unarrayable_values():
    class Unarrayable:
        def __array__(self, dtype=None):
            raise TypeError("cannot materialize")

    assert runtime._hash_array_bytes(Unarrayable()) == "opaque:Unarrayable"

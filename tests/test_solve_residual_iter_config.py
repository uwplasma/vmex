from __future__ import annotations

import pytest

from vmec_jax.solve_residual_iter_config import (
    HEAVY_DUMP_ENVS,
    bad_jacobian_tau_tolerance,
    legacy_dump_enabled,
    parse_bad_jacobian_config,
    resolve_chunked_scan_config,
    resolve_debug_print_config,
    resolve_dump_history_config,
    resolve_nstep_screen,
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

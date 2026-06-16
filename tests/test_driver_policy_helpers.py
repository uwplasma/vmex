from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import os
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
from vmec_jax.drivers import flux as driver_flux
from vmec_jax.drivers import io as driver_io
from vmec_jax.drivers import runtime as driver_runtime
from vmec_jax.drivers import solve as driver_solve
from vmec_jax.drivers.policy import (
    dynamic_scan_probe_settings,
    resolve_driver_signgs,
    resolve_driver_step_size,
    resolve_stage_jit_settings,
    resolve_vmec2000_jit_forces_policy,
)


class _Input:
    def __init__(self, **values):
        self.values = dict(values)

    def get(self, key, default=None):
        return self.values.get(key, default)

    def get_bool(self, key, default=False):
        return bool(self.values.get(key, default))

    def get_float(self, key, default=0.0):
        return float(self.values.get(key, default))

    def get_int(self, key, default=0):
        return int(self.values.get(key, default))


@pytest.mark.parametrize(
    ("solver_mode", "performance_mode", "expected"),
    [
        (None, True, "default"),
        (None, False, "parity"),
        (" DEFAULT ", False, "default"),
        ("Fast", False, "default"),
        ("SAFE", True, "parity"),
        (" reference ", True, "parity"),
        (" PERF ", False, "accelerated"),
        ("accelerated", True, "accelerated"),
    ],
)
def test_normalize_solver_mode_handles_aliases_case_and_defaults(solver_mode, performance_mode, expected):
    assert driver._normalize_solver_mode(solver_mode=solver_mode, performance_mode=performance_mode) == expected


def test_normalize_solver_mode_reports_valid_modes():
    with pytest.raises(ValueError, match="Expected one of: accelerated, default, parity"):
        driver._normalize_solver_mode(solver_mode="not-a-mode", performance_mode=False)


def test_dynamic_scan_probe_settings_helper_uses_backend_and_env_dict():
    env = {"VMEC_JAX_DYNAMIC_SCAN_ITERS": "7", "VMEC_JAX_DYNAMIC_SCAN_TIMED": "on"}

    assert dynamic_scan_probe_settings(
        5,
        backend_name_func=lambda: "gpu",
        getenv=lambda key, default="": env.get(key, default),
    ) == (4, True, "gpu")

    env = {"VMEC_JAX_DYNAMIC_SCAN_ITERS": "bad", "VMEC_JAX_DYNAMIC_SCAN_TIMED": "off"}
    assert dynamic_scan_probe_settings(
        10,
        backend_name_func=lambda: "gpu",
        getenv=lambda key, default="": env.get(key, default),
    ) == (3, False, "gpu")

    assert dynamic_scan_probe_settings(
        50,
        backend_name_func=lambda: "cpu",
        getenv=lambda _key, default="": default,
    ) == (10, True, "cpu")


def test_resolve_stage_jit_settings_preserves_scan_and_warmup_policy():
    env = {}
    settings = resolve_stage_jit_settings(
        jit_forces_base=True,
        scan_mode=True,
        solver="vmec2000_iter",
        performance_mode=False,
        jit_precompile=None,
        getenv=lambda key, default=None: env.get(key, default),
    )
    assert settings.jit_forces_eff is False
    assert settings.jit_precompile_eff is False
    assert settings.jit_warmup_iters == 0
    assert settings.jit_precompile_noscan is True
    assert settings.jit_warmup_noscan == 0

    env = {"VMEC_JAX_SCAN_JIT_FORCES": "1", "VMEC_JAX_JIT_PRECOMPILE": "0"}
    settings = resolve_stage_jit_settings(
        jit_forces_base=True,
        scan_mode=True,
        solver="vmec2000_iter",
        performance_mode=False,
        jit_precompile=None,
        getenv=lambda key, default=None: env.get(key, default),
    )
    assert settings.jit_forces_eff is True
    assert settings.jit_precompile_eff is False
    assert settings.jit_warmup_iters == 0
    assert settings.jit_precompile_noscan is False
    assert settings.jit_warmup_noscan == 2

    env = {"VMEC_JAX_JIT_WARMUP_ITERS": "bad"}
    settings = resolve_stage_jit_settings(
        jit_forces_base=True,
        scan_mode=False,
        solver="vmec2000_iter",
        performance_mode=True,
        jit_precompile=False,
        getenv=lambda key, default=None: env.get(key, default),
    )
    assert settings.jit_forces_eff is True
    assert settings.jit_precompile_eff is False
    assert settings.jit_warmup_iters == 2
    assert settings.jit_precompile_noscan is False
    assert settings.jit_warmup_noscan == 2


def test_resolve_driver_signgs_preserves_vmec2000_parity_and_sanitizes_input():
    assert resolve_driver_signgs(solver_lower="vmec2000_iter", indata=_Input(SIGNGS=1)) == -1
    assert resolve_driver_signgs(solver_lower="VMEC2000_SCAN", indata=_Input(SIGNGS=1)) == -1
    assert resolve_driver_signgs(solver_lower="vmec_lbfgs", indata=_Input(SIGNGS=1)) == 1
    assert resolve_driver_signgs(solver_lower="vmec_lbfgs", indata=_Input(SIGNGS=7)) == -1


def test_resolve_vmec2000_jit_forces_policy_honors_solver_and_env_overrides():
    assert resolve_vmec2000_jit_forces_policy(
        solver_lower="gd",
        jit_forces="auto",
        getenv=lambda _key, default="": default,
    ) == "auto"
    assert resolve_vmec2000_jit_forces_policy(
        solver_lower="vmec2000_iter",
        jit_forces="auto",
        getenv=lambda _key, default="": default,
    ) is True
    assert resolve_vmec2000_jit_forces_policy(
        solver_lower="vmec2000_iter",
        jit_forces=False,
        getenv=lambda key, default="": {"VMEC_JAX_VMEC2000_FORCE_JIT": "1"}.get(key, default),
    ) is True
    assert resolve_vmec2000_jit_forces_policy(
        solver_lower="vmec2000_iter",
        jit_forces=True,
        getenv=lambda key, default="": {"VMEC_JAX_VMEC2000_FORCE_NOJIT": "yes"}.get(key, default),
    ) is False


def test_resolve_driver_step_size_uses_explicit_vmec_and_generic_defaults():
    sentinel = object()
    assert resolve_driver_step_size(
        step_size=0.125,
        step_size_sentinel=sentinel,
        solver_lower="vmec2000_iter",
        indata=_Input(DELT=0.9),
    ) == pytest.approx(0.125)
    assert resolve_driver_step_size(
        step_size=sentinel,
        step_size_sentinel=sentinel,
        solver_lower="vmec2000_iter",
        indata=_Input(DELT=0.9),
    ) == pytest.approx(0.9)
    assert resolve_driver_step_size(
        step_size=None,
        step_size_sentinel=sentinel,
        solver_lower="gd",
        indata=_Input(DELT=0.9),
    ) == pytest.approx(5.0e-3)


def test_profiles_from_static_prefers_host_default_and_uses_vmec_half_mesh():
    static = SimpleNamespace(cfg=SimpleNamespace(ns=4), s=np.asarray([0.0, 0.25, 0.75, 1.0]))
    calls = []

    def host_default(indata, s, signgs):
        calls.append(("host", tuple(s), signgs))
        return "host-flux"

    def fallback(_indata, _s, _signgs):
        pytest.fail("host-default profile path should be used")

    def eval_profiles(_indata, s_half):
        calls.append(("eval", tuple(s_half), None))
        return {"pressure": np.asarray(s_half) + 1.0}

    flux, profiles, pressure = driver_flux.profiles_from_static(
        indata=_Input(),
        static_in=static,
        signgs=-1,
        flux_profiles_from_indata_host_default_func=host_default,
        flux_profiles_from_indata_func=fallback,
        eval_profiles_func=eval_profiles,
    )

    assert flux == "host-flux"
    assert profiles["pressure"].tolist() == pytest.approx([1.0, 1.125, 1.5, 1.875])
    np.testing.assert_allclose(pressure, [1.0, 1.125, 1.5, 1.875])
    assert calls == [
        ("host", (0.0, 0.25, 0.75, 1.0), -1),
        ("eval", (0.0, 0.125, 0.5, 0.875), None),
    ]


def test_profiles_from_static_falls_back_and_defaults_missing_pressure():
    static = SimpleNamespace(cfg=SimpleNamespace(ns=1), s=np.asarray([0.0]))

    flux, profiles, pressure = driver_flux.profiles_from_static(
        indata=_Input(),
        static_in=static,
        signgs=1,
        flux_profiles_from_indata_host_default_func=lambda *_args, **_kwargs: None,
        flux_profiles_from_indata_func=lambda _indata, s, signgs: ("fallback", tuple(s), signgs),
        eval_profiles_func=lambda _indata, s_half: {"iota": np.asarray(s_half)},
    )

    assert flux == ("fallback", (0.0,), 1)
    np.testing.assert_allclose(profiles["iota"], [0.0])
    np.testing.assert_allclose(pressure, [0.0])


def test_initial_guess_with_optional_nojit_uses_standard_path_by_default():
    calls = []

    def initial_guess(static, boundary, indata, *, vmec_project, infer_axis_if_missing):
        calls.append((static, boundary, indata, vmec_project, infer_axis_if_missing))
        return "state"

    state = driver_solve.initial_guess_with_optional_nojit(
        "static",
        "boundary",
        "indata",
        vmec_project=True,
        infer_axis_if_missing=False,
        performance_mode=False,
        initial_guess_from_boundary_func=initial_guess,
        default_backend_name_func=lambda: "cpu",
        getenv=lambda _key, default="": default,
    )

    assert state == "state"
    assert calls == [("static", "boundary", "indata", True, False)]


def test_initial_guess_with_optional_nojit_uses_cpu_numpy_patch_when_safe():
    calls = []
    patch_events = []

    class Patch:
        def __enter__(self):
            patch_events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            patch_events.append("exit")

    def initial_guess(*_args, **_kwargs):
        calls.append(tuple(patch_events))
        return "numpy-state"

    state = driver_solve.initial_guess_with_optional_nojit(
        "static",
        "boundary",
        "indata",
        vmec_project=False,
        infer_axis_if_missing=True,
        performance_mode=True,
        initial_guess_from_boundary_func=initial_guess,
        default_backend_name_func=lambda: "cpu",
        getenv=lambda _key, default="": default,
        contains_jax_tracer_func=lambda _boundary: False,
        numpy_module_patch_func=Patch,
    )

    assert state == "numpy-state"
    assert calls == [("enter",)]
    assert patch_events == ["enter", "exit"]


def test_initial_guess_with_optional_nojit_skips_numpy_patch_for_tracers_or_env_disable():
    patch_calls = []

    def initial_guess(*_args, **_kwargs):
        return "state"

    state_traced = driver_solve.initial_guess_with_optional_nojit(
        "static",
        "boundary",
        "indata",
        vmec_project=False,
        infer_axis_if_missing=True,
        performance_mode=True,
        initial_guess_from_boundary_func=initial_guess,
        default_backend_name_func=lambda: "cpu",
        getenv=lambda _key, default="": default,
        contains_jax_tracer_func=lambda _boundary: True,
        numpy_module_patch_func=lambda: patch_calls.append("patch"),
    )
    state_env = driver_solve.initial_guess_with_optional_nojit(
        "static",
        "boundary",
        "indata",
        vmec_project=False,
        infer_axis_if_missing=True,
        performance_mode=True,
        initial_guess_from_boundary_func=initial_guess,
        default_backend_name_func=lambda: "cpu",
        getenv=lambda key, default="": {"VMEC_JAX_CPU_NUMPY_INIT_GUESS": "0"}.get(key, default),
        contains_jax_tracer_func=lambda _boundary: False,
        numpy_module_patch_func=lambda: patch_calls.append("patch"),
    )

    assert state_traced == "state"
    assert state_env == "state"
    assert patch_calls == []


def test_driver_io_helpers_print_concise_and_vmec_style_banners() -> None:
    lines: list[str] = []

    def collect(message="", **_kwargs):
        lines.append(str(message))

    cfg = SimpleNamespace(ns=7, mpol=3, ntor=1, nfp=2)
    driver_io.print_fixed_boundary_intro(
        input_path="input.case",
        cfg=cfg,
        solver="gd",
        use_initial_guess=False,
        max_iter=5,
        step_size=0.25,
        history_size=4,
        print_func=collect,
    )
    assert lines == [
        "[vmec_jax] fixed-boundary run (gd solve)",
        "[vmec_jax] input=input.case",
        "[vmec_jax] ns=7 mpol=3 ntor=1 nfp=2",
        "[vmec_jax] max_iter=5 step_size=0.25 history_size=4",
    ]

    lines.clear()
    driver_io.print_vmec2000_run_header(
        input_path="input.case",
        version="test-version",
        now=datetime(2026, 6, 16, 1, 2, 3),
        print_func=collect,
    )
    assert any("PROCESSING INPUT.CASE" in line for line in lines)
    assert any("THIS IS PARVMEC (PARALLEL VMEC), VERSION test-version" in line for line in lines)
    assert any("DATE = Jun 16,2026  TIME = 01:02:03" in line for line in lines)

    lines.clear()
    result = SimpleNamespace(n_iter=10, diagnostics={"converged": False, "ijacob": 3})
    driver_io.print_vmec2000_run_summary(
        input_path="input.case",
        result=result,
        niter_stage=10,
        total_time=1.25,
        print_func=collect,
    )
    assert " Try increasing NITER or PRE_NITER if the preconditioner is on." in lines
    assert " EXECUTION FINISHED WITHOUT REQUESTED CONVERGENCE" in lines
    assert any("FILE : case" in line for line in lines)
    assert any("NUMBER OF JACOBIAN RESETS =    3" in line for line in lines)
    assert any("TOTAL COMPUTATIONAL TIME (SEC)             1.25" in line for line in lines)


@pytest.mark.parametrize(
    ("backend", "indata", "expected"),
    [
        ("gpu", _Input(LFREEB=True, NS_ARRAY=[5, 9]), ("default", True)),
        ("gpu", _Input(NS_ARRAY=[5, 9]), ("parity", False)),
        ("gpu", _Input(NS_ARRAY=[9]), ("accelerated", True)),
        ("cpu", _Input(LASYM=True), ("accelerated", True)),
        ("cpu", _Input(NCURR=1, NS_ARRAY=[5, 9], NITER_ARRAY=[10, 20]), ("accelerated", True)),
        ("cpu", _Input(NCURR=1, NS_ARRAY=[5, 9]), ("parity", False)),
        ("cpu", _Input(), ("default", True)),
    ],
)
def test_default_non_autodiff_solver_policy_for_backend_uses_input_structure(backend, indata, expected):
    assert driver._default_non_autodiff_solver_policy_for_backend(indata, backend) == expected


def test_resolve_initial_fixed_boundary_policy_preserves_interactive_cpu_cli_defaults():
    policy = driver._resolve_initial_fixed_boundary_policy(
        requested_solver_device="auto",
        policy_backend="cpu",
        indata=_Input(),
        cfg=SimpleNamespace(lfreeb=False),
        solver="vmec2000_iter",
        solver_mode=None,
        performance_mode=True,
        use_scan=None,
        verbose=True,
        grid=None,
        cli_fixed_boundary_mode=False,
        auto_cli_fixed_boundary_mode=True,
    )

    assert policy.solver_mode_explicit is False
    assert policy.solver_mode_eff == "default"
    assert policy.performance_mode is True
    assert policy.accelerated_mode is False
    assert policy.use_scan is False
    assert policy.cli_fixed_boundary_mode is True


def test_resolve_initial_fixed_boundary_policy_honors_explicit_solver_mode_and_scan():
    policy = driver._resolve_initial_fixed_boundary_policy(
        requested_solver_device="gpu",
        policy_backend="gpu",
        indata=_Input(NS_ARRAY=[5, 9]),
        cfg=SimpleNamespace(lfreeb=False),
        solver="vmec2000_iter",
        solver_mode="safe",
        performance_mode=True,
        use_scan=None,
        verbose=True,
        grid=None,
        cli_fixed_boundary_mode=False,
        auto_cli_fixed_boundary_mode=True,
    )

    assert policy.solver_mode_explicit is True
    assert policy.solver_mode_eff == "parity"
    assert policy.performance_mode is False
    assert policy.accelerated_mode is False
    assert policy.use_scan is True
    assert policy.cli_fixed_boundary_mode is False


def test_resolve_axis_infer_missing_policy_matches_parity_performance_and_env():
    assert driver._resolve_axis_infer_missing_policy(
        solver_lower="vmec_lbfgs",
        performance_mode=False,
    ) is True
    assert driver._resolve_axis_infer_missing_policy(
        solver_lower="vmec2000_iter",
        performance_mode=False,
        getenv=lambda _key, default="": default,
    ) is False
    assert driver._resolve_axis_infer_missing_policy(
        solver_lower="vmec2000_iter",
        performance_mode=True,
        getenv=lambda _key, default="": default,
    ) is True
    assert driver._resolve_axis_infer_missing_policy(
        solver_lower="vmec2000_iter",
        performance_mode=False,
        getenv=lambda key, default="": {"VMEC_JAX_ENABLE_AXIS_INFER": "yes"}.get(key, default),
    ) is True
    assert driver._resolve_axis_infer_missing_policy(
        solver_lower="vmec2000_iter",
        performance_mode=True,
        getenv=lambda key, default="": {"VMEC_JAX_DISABLE_AXIS_INFER": "1"}.get(key, default),
    ) is False


@dataclass(frozen=True)
class _RestartCfg:
    ns: int


def test_resolve_restart_context_loads_wout_state_and_copies_resume_state(tmp_path):
    restart_state = SimpleNamespace(layout=SimpleNamespace(ns=5), marker="restart")
    resume = {"iter_offset": 7}
    calls: list[object] = []

    def read_wout_func(path):
        calls.append(path)
        return "wout"

    context = driver_runtime.resolve_restart_context(
        cfg=_RestartCfg(ns=3),
        restart_state=None,
        restart_wout_path=tmp_path / "wout_case.nc",
        restart_solver_state=resume,
        ns_override=5,
        read_wout_func=read_wout_func,
        state_from_wout_func=lambda wout: restart_state if wout == "wout" else None,
        replace_func=replace,
    )

    assert calls == [tmp_path / "wout_case.nc"]
    assert context.cfg.ns == 5
    assert context.restart_state is restart_state
    assert context.restart_wout == "wout"
    assert context.restart_solver_state["iter_offset"] == 7
    assert context.restart_solver_state["state_checkpoint"] is restart_state
    assert "state_checkpoint" not in resume


def test_resolve_restart_context_rejects_mismatched_ns_override():
    restart_state = SimpleNamespace(layout=SimpleNamespace(ns=5))
    with pytest.raises(ValueError, match="restart_state ns=5 does not match ns_override=7"):
        driver_runtime.resolve_restart_context(
            cfg=_RestartCfg(ns=3),
            restart_state=restart_state,
            restart_wout_path=None,
            restart_solver_state=None,
            ns_override=7,
            read_wout_func=lambda _path: None,
            state_from_wout_func=lambda _wout: None,
            replace_func=replace,
        )


def test_resolve_external_field_provider_context_keeps_mgrid_path_legacy() -> None:
    context = driver_runtime.resolve_external_field_provider_context(
        external_field_provider_kind="mgrid",
        external_field_provider_static={"existing": True},
        external_field_provider_params=object(),
    )

    assert context.direct_external_provider is False
    assert context.provider_kind == "mgrid"
    assert context.provider_static == {"existing": True}


def test_resolve_external_field_provider_context_caches_direct_coil_geometry() -> None:
    params = SimpleNamespace(regularization_epsilon=1.25, chunk_size=7)
    calls = []

    def build_geometry(got):
        calls.append(got)
        return "geometry"

    context = driver_runtime.resolve_external_field_provider_context(
        external_field_provider_kind="Coils",
        external_field_provider_static={"existing": True},
        external_field_provider_params=params,
        getenv=lambda key, default="": {"VMEC_JAX_FREEB_JIT_COIL_SAMPLER": "0"}.get(key, default),
        build_coil_field_geometry_func=build_geometry,
    )

    assert context.direct_external_provider is True
    assert context.provider_kind == "coils"
    assert context.provider_static["existing"] is True
    assert context.provider_static["coil_geometry"] == "geometry"
    assert context.provider_static["regularization_epsilon"] == 1.25
    assert context.provider_static["chunk_size"] == 7
    assert context.provider_static["cache_scope"] == "host_forward_only"
    assert context.provider_static["jit_sampler"] is False
    assert calls == [params]


def test_resolve_external_field_provider_context_respects_existing_or_disabled_cache() -> None:
    params = SimpleNamespace()
    existing = {"coil_geometry": object()}

    context_existing = driver_runtime.resolve_external_field_provider_context(
        external_field_provider_kind="direct_coils",
        external_field_provider_static=existing,
        external_field_provider_params=params,
        build_coil_field_geometry_func=lambda _params: pytest.fail("should not rebuild existing geometry"),
    )
    assert context_existing.provider_static is existing

    context_disabled = driver_runtime.resolve_external_field_provider_context(
        external_field_provider_kind="direct_coils",
        external_field_provider_static=None,
        external_field_provider_params=params,
        getenv=lambda key, default="": {"VMEC_JAX_FREEB_DISABLE_COIL_GEOMETRY_CACHE": "yes"}.get(key, default),
        build_coil_field_geometry_func=lambda _params: pytest.fail("cache disabled"),
    )
    assert context_disabled.provider_static is None


def test_resolve_external_field_provider_context_falls_back_on_custom_provider_error() -> None:
    original_static = {"custom": object()}

    context = driver_runtime.resolve_external_field_provider_context(
        external_field_provider_kind="direct_coils",
        external_field_provider_static=original_static,
        external_field_provider_params=object(),
        build_coil_field_geometry_func=lambda _params: (_ for _ in ()).throw(TypeError("custom")),
    )

    assert context.direct_external_provider is True
    assert context.provider_static is original_static


def test_resolve_jit_forces_auto_policy_preserves_explicit_flags():
    static = SimpleNamespace(
        modes=SimpleNamespace(m=np.arange(3)),
        cfg=SimpleNamespace(ns=3, ntheta=4, nzeta=5),
    )

    assert driver._resolve_jit_forces_auto_policy(False, static, niter_i=100) is False
    assert driver._resolve_jit_forces_auto_policy(True, static, niter_i=0) is True
    assert driver._resolve_jit_forces_auto_policy("false", static, niter_i=0) is True


def test_resolve_jit_forces_auto_policy_uses_work_and_iteration_thresholds():
    small = SimpleNamespace(
        modes=SimpleNamespace(m=np.arange(2)),
        cfg=SimpleNamespace(ns=3, ntheta=4, nzeta=5),
    )
    large = SimpleNamespace(
        modes=SimpleNamespace(m=np.arange(2_000)),
        cfg=SimpleNamespace(ns=50, ntheta=50, nzeta=1),
    )

    assert driver._resolve_jit_forces_auto_policy("auto", small, niter_i=4) is False
    assert driver._resolve_jit_forces_auto_policy(" auto ", small, niter_i=5) is True
    assert driver._resolve_jit_forces_auto_policy("AUTO", large, niter_i=1) is True
    assert driver._resolve_jit_forces_auto_policy("auto", object(), niter_i=1) is True


def test_host_update_default_and_scan_policy_handle_fallback_branches(monkeypatch):
    cfg = SimpleNamespace(ns=2, mpol=2, ntor=1, lasym=False)
    monkeypatch.setenv("VMEC_JAX_HOST_UPDATE_CPU_WORK_LIMIT", "not-an-int")

    assert driver._host_update_assembly_driver_default(
        cfg=cfg,
        performance_mode=True,
        backend="cpu",
        use_scan=False,
    ) is True

    assert driver._default_use_scan_for_backend(_Input(), "unknown-backend", "default") is False


def test_precomputed_tridi_policy_tolerates_bad_mode_counts():
    class BadModeCfg:
        lasym = False
        ntor = 16

        @property
        def mpol(self):
            raise RuntimeError("mode metadata unavailable")

    assert driver._default_preconditioner_use_precomputed_tridi(
        cfg=BadModeCfg(),
        backend="gpu",
        performance_mode=True,
        use_scan=False,
    ) is None


@pytest.mark.parametrize("solver_device", [None, "", " none ", "AUTO", "default"])
def test_resolve_fixed_boundary_solver_device_inherits_default_for_auto_values(solver_device):
    assert (
        driver._resolve_fixed_boundary_solver_device_name(
            solver_device=solver_device,
            backend="gpu",
            cfg=object(),
            indata=object(),
            solver_lower="vmec2000_iter",
            cli_fixed_boundary_mode=True,
            accelerated_mode=True,
            ns_list_input=[5, 9],
            niter_list_input=[10, 20],
            restart_state_present=True,
            restart_solver_state_present=True,
        )
        is None
    )


@pytest.mark.parametrize(("solver_device", "expected"), [(" cpu ", "cpu"), ("GPU", "gpu"), ("tpu", "tpu")])
def test_resolve_fixed_boundary_solver_device_preserves_explicit_names(solver_device, expected):
    assert (
        driver._resolve_fixed_boundary_solver_device_name(
            solver_device=solver_device,
            backend="cpu",
            cfg=object(),
            indata=object(),
            solver_lower="vmec2000_iter",
            cli_fixed_boundary_mode=False,
            accelerated_mode=False,
            ns_list_input=None,
            niter_list_input=None,
            restart_state_present=False,
            restart_solver_state_present=False,
        )
        == expected
    )


def test_dynamic_scan_probe_settings_clamps_single_iteration_probe(monkeypatch):
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "cpu")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "50")
    monkeypatch.setenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "off")

    pre_iters, timed_probe, backend = driver._dynamic_scan_probe_settings(1)

    assert pre_iters == 1
    assert timed_probe is False
    assert backend == "cpu"


def test_example_paths_reports_missing_wout_as_none(tmp_path):
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.synthetic"
    input_path.write_text("&INDATA\n/\n")

    found_input, found_wout = driver.example_paths("synthetic", root=tmp_path)

    assert found_input == input_path
    assert found_wout is None


def test_requested_final_ftol_prefers_last_valid_list_value_and_clamps():
    fallback = _Input(FTOL=4.0e-7)
    negative_fallback = _Input(FTOL=-1.0)

    assert driver._requested_final_ftol(indata=fallback, ftol_list_input=("1e-4", 2.0e-5)) == 2.0e-5
    assert driver._requested_final_ftol(indata=fallback, ftol_list_input=[1.0e-4, -2.0]) == 0.0
    assert driver._requested_final_ftol(indata=fallback, ftol_list_input=object()) == 4.0e-7
    assert driver._requested_final_ftol(indata=negative_fallback, ftol_list_input=[]) == 0.0


def test_accelerated_total_target_clamps_negative_ftol():
    assert driver._accelerated_fsq_total_target_from_ftol(-1.0e-8) == 0.0
    assert driver._accelerated_fsq_total_target_from_ftol(2.0e-8) == pytest.approx(6.0e-8)


def test_allocate_integer_budget_clamps_inputs_and_distributes_remainder():
    assert driver._allocate_integer_budget(total=-3, weights=[1, 2, 3]) == [0, 0, 0]
    assert driver._allocate_integer_budget(total=6, weights=[-5, 1, 2]) == [0, 2, 4]
    assert driver._allocate_integer_budget(total=5, weights=[1, 2]) == [2, 3]
    assert driver._allocate_integer_budget(total=5, weights=[0, 0, 0]) == [0, 0, 5]


def test_as_list_like_tolerates_broken_numpy_type_check(monkeypatch):
    monkeypatch.setattr(driver.np, "ndarray", object())

    assert driver._as_list_like(object()) is None


def test_accelerated_cli_budget_helpers_scale_total_and_weight_stages():
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=100, ns_stages=[9, 36]) == 50
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=0, ns_stages=[4]) == 1
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=10, ns_stages=[4, 7, 10]) == [5, 3, 2]
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=0, ns_stages=[]) == [1]


def test_distribute_stage_iters_matches_vmec_budget_edges():
    assert driver._distribute_stage_iters(iters=0, nstep=3) == [0]
    assert driver._distribute_stage_iters(iters=2, nstep=5) == [2]
    assert driver._distribute_stage_iters(iters=7, nstep=3) == [3, 2, 2]
    assert driver._distribute_stage_iters(iters=5, nstep=1) == [5]


def test_resume_state_sanitizers_drop_unsafe_payloads_and_clamp_step():
    resume_state = {
        "time_step": 0.25,
        "inv_tau": [1.0, 2.0],
        "iter_offset": 17,
        "flip_sign": -1,
        "vmec2000_cache_valid": True,
        "cached_arrays": object(),
    }

    cross_grid = driver._sanitize_resume_state_for_grid_change(resume_state, step_size=0.1)
    same_grid = driver._sanitize_resume_state_for_same_grid(resume_state, step_size=0.1)

    assert cross_grid["time_step"] == pytest.approx(0.1)
    assert cross_grid["inv_tau"] == [pytest.approx(1.5)] * 10
    assert cross_grid["iter_offset"] == 0
    assert cross_grid["flip_sign"] == -1.0
    assert cross_grid["vmec2000_cache_valid"] is False
    assert "cached_arrays" not in cross_grid

    assert same_grid["time_step"] == pytest.approx(0.1)
    assert same_grid["inv_tau"] == [1.0, 2.0]
    assert same_grid["iter_offset"] == 17
    assert same_grid["vmec2000_cache_valid"] is False
    assert "cached_arrays" not in same_grid
    assert driver._sanitize_resume_state_for_grid_change({}, step_size=0.1) is None
    assert driver._sanitize_resume_state_for_same_grid(None, step_size=0.1) is None


def test_cat_result_history_concatenates_in_order_and_skips_missing_payloads():
    results = [
        SimpleNamespace(trace=np.asarray([1.0, 2.0])),
        SimpleNamespace(trace=None),
        SimpleNamespace(trace=np.asarray([3.0])),
        SimpleNamespace(),
    ]

    np.testing.assert_allclose(driver._cat_result_history(results, "trace"), [1.0, 2.0, 3.0])


def test_cat_result_history_returns_empty_float_array_when_no_parts_exist():
    history = driver._cat_result_history([SimpleNamespace(trace=None), SimpleNamespace()], "trace")

    assert history.shape == (0,)
    assert history.dtype == np.dtype(float)


def _stage_result(label, *, n_iter, w_history, diagnostics=None):
    zeros = np.zeros((0,), dtype=float)
    return driver.SolveVmecResidualResult(
        state=label,
        n_iter=int(n_iter),
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(w_history, dtype=float) + 10.0,
        fsqz2_history=np.asarray(w_history, dtype=float) + 20.0,
        fsql2_history=np.asarray(w_history, dtype=float) + 30.0,
        grad_rms_history=zeros,
        step_history=zeros,
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )


def test_merge_stage_chunk_results_concatenates_histories_and_diagnostics():
    first = _stage_result(
        "first",
        n_iter=1,
        w_history=[1.0, 0.5],
        diagnostics={
            "step_status_history": np.asarray([1.0]),
            "first_only": True,
            "timing": {
                "solve_total_s": 2.0,
                "iteration_loop_s": 1.5,
                "compute_forces_calls": 2,
                "iterations": 2,
            },
        },
    )
    second = _stage_result(
        "second",
        n_iter=2,
        w_history=[0.25, 0.125, 0.0625],
        diagnostics={
            "step_status_history": np.asarray([2.0, 3.0]),
            "time_step_history": np.asarray([0.1]),
            "timing": {
                "scan_total_s": 3.0,
                "iteration_loop_s": 2.5,
                "compute_forces_calls": 3,
                "iterations": 3,
            },
        },
    )

    merged = driver._merge_stage_chunk_results([first, second], mode_i="accelerated")

    assert merged.state == "second"
    assert merged.n_iter == 4
    np.testing.assert_allclose(merged.w_history, [1.0, 0.5, 0.25, 0.125, 0.0625])
    np.testing.assert_allclose(merged.diagnostics["step_status_history"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(merged.diagnostics["time_step_history"], [0.1])
    np.testing.assert_array_equal(merged.diagnostics["accelerated_stage_chunk_iters"], [2, 3])
    assert merged.diagnostics["accelerated_stage_chunk_count"] == 2
    assert merged.diagnostics["accelerated_stage_chunked"] is True
    assert merged.diagnostics["accelerated_stage_effective_mode"] == "accelerated"
    assert merged.diagnostics["timing"]["solve_total_s"] == pytest.approx(5.0)
    assert merged.diagnostics["timing"]["iteration_loop_s"] == pytest.approx(4.0)
    assert merged.diagnostics["timing"]["compute_forces_calls"] == 5
    assert merged.diagnostics["timing"]["iterations"] == 5
    assert merged.diagnostics["timing"]["solve_total_per_iter_s"] == pytest.approx(1.0)
    np.testing.assert_allclose(merged.diagnostics["timing"]["chunk_solve_total_s"], [2.0, 3.0])


def test_merge_stage_chunk_results_marks_single_chunk_without_concatenation():
    result = _stage_result("single", n_iter=0, w_history=[1.0], diagnostics={"kept": "yes"})

    merged = driver._merge_stage_chunk_results([result], mode_i="parity")

    assert merged.state == "single"
    assert merged.diagnostics["kept"] == "yes"
    assert merged.diagnostics["accelerated_stage_chunked"] is False
    assert merged.diagnostics["accelerated_stage_effective_mode"] == "parity"


def test_direct_free_boundary_lax_tridi_default_stays_delegated(monkeypatch):
    cfg = SimpleNamespace(lfreeb=True)

    monkeypatch.delenv("VMEC_JAX_TRIDI_SOLVE", raising=False)
    assert (
        driver._default_preconditioner_use_lax_tridi(
            cfg=cfg,
            backend="cpu",
            performance_mode=True,
            use_scan=False,
            direct_external_provider=True,
        )
        is None
    )

    monkeypatch.setenv("VMEC_JAX_TRIDI_SOLVE", "1")
    assert (
        driver._default_preconditioner_use_lax_tridi(
            cfg=cfg,
            backend="gpu",
            performance_mode=True,
            use_scan=True,
            direct_external_provider=True,
        )
        is None
    )


def test_copy_final_force_payload_handles_unsettable_result_object():
    source = SimpleNamespace(_final_force_payload={"payload": 1})

    assert driver._copy_final_force_payload(3, source) == 3


def test_stage_switch_reason_from_progress_reports_only_actionable_misses():
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=10.0,
            best_total_fsq=9.0,
            target_total_fsq=1.0,
            chunk_iters=2,
            remaining_budget=0,
        )
        is None
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=10.0,
            best_total_fsq=np.inf,
            target_total_fsq=1.0,
            chunk_iters=2,
            remaining_budget=10,
        )
        == "nonfinite_total_fsq"
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=10.0,
            best_total_fsq=10.0,
            target_total_fsq=1.0,
            chunk_iters=2,
            remaining_budget=10,
        )
        == "nondecreasing_total_fsq"
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=10.0,
            best_total_fsq=0.5,
            target_total_fsq=1.0,
            chunk_iters=2,
            remaining_budget=10,
        )
        is None
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=0.0,
            best_total_fsq=-1.0,
            target_total_fsq=0.0,
            chunk_iters=2,
            remaining_budget=10,
        )
        is None
    )
    assert (
        driver._stage_switch_reason_from_progress(
            start_total_fsq=100.0,
            best_total_fsq=10.0,
            target_total_fsq=1.0,
            chunk_iters=1,
            remaining_budget=2,
        )
        is None
    )
    assert driver._stage_switch_reason_from_progress(
        start_total_fsq=100.0,
        best_total_fsq=90.0,
        target_total_fsq=1.0,
        chunk_iters=1,
        remaining_budget=3,
    ).startswith("projected_budget_miss:")


def test_vmec_history_comparison_helpers_cover_mismatch_and_tolerance():
    lhs = SimpleNamespace(
        w_history=np.asarray([1.0, 2.0]),
        fsqr2_history=np.asarray([1.0]),
        fsqz2_history=np.asarray([2.0]),
        fsql2_history=np.asarray([3.0]),
    )
    rhs = SimpleNamespace(
        w_history=np.asarray([1.0, 2.0 + 1.0e-7]),
        fsqr2_history=np.asarray([1.0]),
        fsqz2_history=np.asarray([2.0]),
        fsql2_history=np.asarray([3.0]),
    )
    wrong_shape = SimpleNamespace(
        w_history=np.asarray([1.0]),
        fsqr2_history=np.asarray([1.0]),
        fsqz2_history=np.asarray([2.0]),
        fsql2_history=np.asarray([3.0]),
    )

    assert driver._vmec_history_relerr(np.asarray([1.0]), np.asarray([[1.0]])) == np.inf
    assert driver._vmec_histories_match(lhs, rhs, rtol=1.0e-5, atol=0.0) is True
    assert driver._vmec_histories_match(lhs, rhs, rtol=1.0e-10, atol=0.0) is False
    assert driver._vmec_histories_match(lhs, wrong_shape, rtol=1.0, atol=1.0) is False


def test_result_final_residuals_prefers_explicit_diagnostics_over_histories():
    result = SimpleNamespace(
        diagnostics={"final_fsqr": "1.0", "final_fsqz": 2, "final_fsql": np.asarray(3.0)},
        fsqr2_history=np.asarray([10.0]),
        fsqz2_history=np.asarray([20.0]),
        fsql2_history=np.asarray([30.0]),
    )

    assert driver._result_final_residuals(result) == (1.0, 2.0, 3.0)


def test_result_final_residuals_falls_back_when_explicit_diagnostics_are_incomplete():
    result = SimpleNamespace(
        diagnostics={"final_fsqr": 1.0, "final_fsqz": None, "final_fsql": 3.0},
        fsqr2_history=np.asarray([[1.0, 4.0]]),
        fsqz2_history=np.asarray([[2.0, 5.0]]),
        fsql2_history=np.asarray([[3.0, 6.0]]),
    )

    assert driver._result_final_residuals(result) == (4.0, 5.0, 6.0)


def test_result_final_residuals_uses_flattened_diagnostic_histories():
    result = SimpleNamespace(
        diagnostics={
            "fsqr_full": np.asarray([[1.0, 2.0]]),
            "fsqz_full": np.asarray([[3.0, 4.0]]),
            "fsql_full": np.asarray([[5.0, 6.0]]),
        },
    )

    assert driver._result_final_residuals(result) == (2.0, 4.0, 6.0)


def test_result_final_residuals_ignores_unparseable_explicit_values_and_broken_histories():
    class BrokenHistoryResult:
        diagnostics = {
            "final_fsqr": "not-a-float",
            "final_fsqz": 2.0,
            "final_fsql": 3.0,
            "fsqr_full": [],
            "fsqz_full": [],
            "fsql_full": [],
        }

        @property
        def fsqr2_history(self):
            raise RuntimeError("history not materialized")

        @property
        def fsqz2_history(self):
            raise RuntimeError("history not materialized")

        @property
        def fsql2_history(self):
            raise RuntimeError("history not materialized")

    assert driver._result_final_residuals(BrokenHistoryResult()) is None


def test_result_final_fsq_prefers_weight_history_then_residual_sum():
    weighted = SimpleNamespace(
        diagnostics={"final_fsqr": 1.0, "final_fsqz": 2.0, "final_fsql": 3.0},
        w_history=np.asarray([99.0, 0.25]),
    )
    residual_only = SimpleNamespace(diagnostics={"final_fsqr": 1.0, "final_fsqz": 2.0, "final_fsql": 3.0})

    assert driver._result_final_fsq(weighted) == 0.25
    assert driver._result_final_fsq(residual_only) == 6.0
    assert np.isinf(driver._result_final_fsq(None))


def test_result_meets_requested_ftol_respects_strict_diagnostic_flag():
    result = SimpleNamespace(
        diagnostics={"converged_strict": False, "final_fsqr": 0.0, "final_fsqz": 0.0, "final_fsql": 0.0},
    )

    assert driver._result_meets_requested_ftol(result, ftol=1.0) is False


def test_result_meets_requested_ftol_uses_legacy_converged_only_without_ftol_metadata():
    result = SimpleNamespace(
        diagnostics={"converged": False, "final_fsqr": 0.0, "final_fsqz": 0.0, "final_fsql": 0.0},
    )

    assert driver._result_meets_requested_ftol(result, ftol=1.0) is False


def test_result_meets_requested_ftol_clamps_negative_requested_target():
    zero = SimpleNamespace(
        diagnostics={"requested_ftol": -1.0, "final_fsqr": 0.0, "final_fsqz": 0.0, "final_fsql": 0.0},
    )
    nonzero = SimpleNamespace(
        diagnostics={"requested_ftol": -1.0, "final_fsqr": 1.0e-16, "final_fsqz": 0.0, "final_fsql": 0.0},
    )

    assert driver._result_meets_requested_ftol(zero, ftol=-1.0) is True
    assert driver._result_meets_requested_ftol(nonzero, ftol=-1.0) is False


def test_result_hits_total_target_clamps_negative_target_and_handles_none():
    zero = SimpleNamespace(w_history=np.asarray([0.0]), diagnostics={})
    nonzero = SimpleNamespace(w_history=np.asarray([1.0e-12]), diagnostics={})

    assert driver._result_hits_total_target(zero, fsq_total_target=-1.0) is True
    assert driver._result_hits_total_target(nonzero, fsq_total_target=-1.0) is False
    assert driver._result_hits_total_target(zero, fsq_total_target=None) is False


def test_copy_final_force_payload_preserves_solver_payload_when_present():
    source = _stage_result("source", n_iter=1, w_history=[1.0])
    result = _stage_result("result", n_iter=1, w_history=[2.0])
    payload = {"fsqr": 1.0}
    object.__setattr__(source, "_final_force_payload", payload)

    out = driver._copy_final_force_payload(result, source)

    assert out is result
    assert getattr(out, "_final_force_payload") is payload


def test_wout_from_fixed_boundary_run_samples_fsqt_and_falls_back_to_residual_recompute(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured = {}

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(path=kwargs["path"], marker="wout")

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: (7.0, 8.0, 9.0))
    monkeypatch.delenv("VMEC_JAX_WOUT_FAST_BCOVAR", raising=False)

    result = SimpleNamespace(
        diagnostics={"converged": True},
        fsqr2_history=np.asarray([1.0, 2.0, 3.0]),
        fsqz2_history=np.asarray([4.0, 5.0, 6.0]),
    )
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=result,
        flux=None,
        profiles={},
        signgs=-1,
    )

    out = driver.wout_from_fixed_boundary_run(
        run,
        include_fsq=True,
        path=tmp_path / "wout_synthetic.nc",
        fast_bcovar=True,
    )

    assert out.marker == "wout"
    assert captured["fsqr"] == 7.0
    assert captured["fsqz"] == 8.0
    assert captured["fsql"] == 9.0
    assert captured["converged"] is True
    np.testing.assert_allclose(captured["fsqt"][:3], [5.0, 7.0, 9.0])
    assert captured["fsqt"].shape == (100,)
    assert os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR") is None


def test_wout_from_fixed_boundary_run_include_fsq_false_restores_existing_fast_env(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured = {}

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(path=kwargs["path"])

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: pytest.fail("unexpected recompute"))
    monkeypatch.setenv("VMEC_JAX_WOUT_FAST_BCOVAR", "original")
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=None,
        flux=None,
        profiles={},
        signgs=1,
    )

    driver.wout_from_fixed_boundary_run(
        run,
        include_fsq=False,
        path=tmp_path / "wout_no_fsq.nc",
        fast_bcovar=False,
    )

    assert captured["fsqr"] == 0.0
    assert captured["fsqz"] == 0.0
    assert captured["fsql"] == 0.0
    assert captured["fsqt"] is None
    assert captured["converged"] is None
    assert os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] == "original"


def test_wout_from_fixed_boundary_run_parity_mode_uses_legacy_bcovar_by_default(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured_env = []

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured_env.append(os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR"))
        return SimpleNamespace(path=kwargs["path"])

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: pytest.fail("unexpected recompute"))
    monkeypatch.delenv("VMEC_JAX_WOUT_FAST_BCOVAR", raising=False)
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=SimpleNamespace(diagnostics={"solver_mode": "parity"}),
        flux=None,
        profiles={},
        signgs=1,
    )

    driver.wout_from_fixed_boundary_run(run, include_fsq=False, path=tmp_path / "wout_parity.nc")

    assert captured_env == ["0"]
    assert os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR") is None


def test_wout_from_fixed_boundary_run_explicit_fast_bcovar_overrides_parity_mode(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured_env = []

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured_env.append(os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR"))
        return SimpleNamespace(path=kwargs["path"])

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: pytest.fail("unexpected recompute"))
    monkeypatch.delenv("VMEC_JAX_WOUT_FAST_BCOVAR", raising=False)
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=SimpleNamespace(diagnostics={"solver_mode": "parity"}),
        flux=None,
        profiles={},
        signgs=1,
    )

    driver.wout_from_fixed_boundary_run(
        run,
        include_fsq=False,
        path=tmp_path / "wout_parity_fast.nc",
        fast_bcovar=True,
    )

    assert captured_env == ["1"]
    assert os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR") is None


def test_wout_from_fixed_boundary_run_uses_complete_result_histories_without_recompute(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    captured = {}

    def _fake_wout_minimal_from_fixed_boundary(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(path=kwargs["path"], marker="packed")

    monkeypatch.setattr(wout_module, "wout_minimal_from_fixed_boundary", _fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(driver, "residual_scalars_from_state", lambda **_kwargs: pytest.fail("unexpected recompute"))

    result = SimpleNamespace(
        diagnostics={"converged": False},
        fsqr2_history=np.asarray([10.0, 1.0]),
        fsqz2_history=np.asarray([20.0, 2.0]),
        fsql2_history=np.asarray([30.0, 3.0]),
    )
    run = driver.FixedBoundaryRun(
        cfg=object(),
        indata=object(),
        static=object(),
        state=object(),
        result=result,
        flux=None,
        profiles={},
        signgs=1,
    )

    out = driver.wout_from_fixed_boundary_run(run, include_fsq=True, path=tmp_path / "wout_hist.nc")

    assert out.marker == "packed"
    assert captured["fsqr"] == 1.0
    assert captured["fsqz"] == 2.0
    assert captured["fsql"] == 3.0
    assert captured["converged"] is False
    np.testing.assert_allclose(captured["fsqt"][:2], [30.0, 3.0])
    np.testing.assert_allclose(captured["fsqt"][2:], np.zeros(98))

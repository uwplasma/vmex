"""Pure configuration helpers for the residual-iteration solve."""

from __future__ import annotations

from typing import Mapping, NamedTuple


FALSE_TOKENS = ("", "0", "false", "no")

LIGHT_DUMP_ENVS = (
    "VMEC_JAX_DUMP_SCALARS",
    "VMEC_JAX_DUMP_GCX2",
    "VMEC_JAX_DUMP_FSQ1",
    "VMEC_JAX_DUMP_TIMECONTROL",
    "VMEC_JAX_DUMP_CHECKPOINT",
)

HEAVY_DUMP_ENVS = (
    "VMEC_JAX_DUMP_TOMNSPS",
    "VMEC_JAX_DUMP_TOMNSPS_KERNELS",
    "VMEC_JAX_DUMP_FORCE_KERNELS",
    "VMEC_JAX_DUMP_GC",
    "VMEC_JAX_DUMP_BSUBE",
    "VMEC_JAX_DUMP_BSUBE_TERMS",
    "VMEC_JAX_DUMP_LULV",
    "VMEC_JAX_DUMP_XC",
    "VMEC_JAX_DUMP_LAM",
    "VMEC_JAX_DUMP_LAMCAL",
    "VMEC_JAX_DUMP_LAM_FSQL1",
    "VMEC_JAX_DUMP_LAM_GCL",
)


class BadJacobianConfig(NamedTuple):
    """Environment-derived policy for bad-Jacobian detection and probing."""

    mode: str
    use_state: bool
    dump_ptau_state: bool
    state_probe: bool
    initial_state_probe_iters: int
    ptau_tol: float
    ptau_tol_rel: float


class DumpHistoryConfig(NamedTuple):
    """Environment-derived policy for optional debug dumps and history tracking."""

    dumps_enabled: bool
    dump_any: bool
    jit_forces: bool
    light_history: bool
    track_history: bool
    disabled_jit_for_dumps: bool


class ChunkedScanConfig(NamedTuple):
    """Resolved scan chunking and fallback policy for residual iteration."""

    force_chunked_scan: bool
    scan_fallback_enabled: bool
    differentiating_scan: bool


class HostResidualMetricConfig(NamedTuple):
    """Policy for where residual scalar metrics are reduced."""

    fsq1_norms_on_accelerator: bool
    residual_metrics_on_accelerator: bool


class AxisResetConfig(NamedTuple):
    """Initial magnetic-axis reset thresholds and toggles."""

    force_axis_reset: bool
    axis_reset_always_3d: bool
    axis_reset_fsq_min: float


class DebugPrintConfig(NamedTuple):
    """Resolved VMEC iteration print mode for host and scan loops."""

    print_live: bool
    mode: str
    ordered: bool


def _env_value(env: Mapping[str, str | None], name: str, default: str = "") -> str:
    value = env.get(name, default)
    return default if value is None else str(value)


def env_flag_enabled(value: str | None) -> bool:
    """Parse modern boolean-ish env flags using stripped/lower false tokens."""
    return str("" if value is None else value).strip().lower() not in FALSE_TOKENS


def env_flag_enabled_with_off(value: str | None) -> bool:
    """Parse env flags that also treat ``off`` as false."""
    return str("" if value is None else value).strip().lower() not in (*FALSE_TOKENS, "off")


def legacy_dump_enabled(value: str | None) -> bool:
    """Match solve.py's legacy dump enablement exactly: no stripping or lowercasing."""
    value_s = "" if value is None else str(value)
    return value_s not in ("", "0")


def parse_bad_jacobian_config(env: Mapping[str, str | None]) -> BadJacobianConfig:
    """Parse bad-Jacobian mode/probe/tolerance env policy."""
    mode = _env_value(env, "VMEC_JAX_BADJAC_MODE", "ptau").strip().lower()
    if mode not in ("ptau", "state"):
        mode = "ptau"

    try:
        ptau_tol = float(_env_value(env, "VMEC_JAX_PTAU_TOL", "").strip() or 0.0)
    except Exception:
        ptau_tol = 0.0
    try:
        ptau_tol_rel = float(_env_value(env, "VMEC_JAX_PTAU_TOL_REL", "1.0e-6").strip() or 0.0)
    except Exception:
        ptau_tol_rel = 0.0
    if ptau_tol_rel < 0.0:
        ptau_tol_rel = 0.0
    try:
        initial_state_probe_iters = int(_env_value(env, "VMEC_JAX_BADJAC_INITIAL_STATE_PROBE_ITERS", "2").strip() or 0)
    except Exception:
        initial_state_probe_iters = 2
    if initial_state_probe_iters < 0:
        initial_state_probe_iters = 0

    return BadJacobianConfig(
        mode=mode,
        use_state=mode == "state",
        dump_ptau_state=env_flag_enabled(_env_value(env, "VMEC_JAX_DUMP_PTAU_STATE", "0")),
        state_probe=env_flag_enabled(_env_value(env, "VMEC_JAX_BADJAC_STATE_PROBE", "0")),
        initial_state_probe_iters=int(initial_state_probe_iters),
        ptau_tol=ptau_tol,
        ptau_tol_rel=ptau_tol_rel,
    )


def bad_jacobian_tau_tolerance(*, ptau_tol: float, ptau_tol_rel: float, tau_scale: float) -> float:
    """Return the host bad-Jacobian tau tolerance used by ptau/state checks."""
    if float(ptau_tol_rel) > 0.0:
        return max(abs(float(ptau_tol)), float(ptau_tol_rel) * float(tau_scale))
    return max(abs(float(ptau_tol)), 0.0)


def should_probe_bad_jacobian_state(
    *,
    state_probe: bool,
    initial_state_probe_iters: int,
    iter_idx: int,
) -> bool:
    """Return whether to run the optional expensive state-Jacobian safety probe.

    The production bad-Jacobian check uses VMEC's cheap ``ptau`` sign change
    proxy.  The full state-Jacobian probe is a diagnostic/parity knob because it
    adds host/device work in the first few iterations, especially on GPU.
    """

    return bool(state_probe) and int(initial_state_probe_iters) > 0 and int(iter_idx) <= int(initial_state_probe_iters)


def resolve_dump_history_config(
    *,
    env: Mapping[str, str | None],
    jit_forces: bool,
    light_history: bool,
    heavy_dump_envs: tuple[str, ...] = HEAVY_DUMP_ENVS,
    light_dump_envs: tuple[str, ...] = LIGHT_DUMP_ENVS,
) -> DumpHistoryConfig:
    """Resolve debug-dump effects on JIT and residual-history collection."""
    dumps_enabled = any(legacy_dump_enabled(_env_value(env, name, "")) for name in heavy_dump_envs)
    dump_any = dumps_enabled or any(legacy_dump_enabled(_env_value(env, name, "")) for name in light_dump_envs)
    disabled_jit = bool(dumps_enabled and jit_forces)
    jit_forces_resolved = False if disabled_jit else bool(jit_forces)
    light_history_resolved = False if dump_any else bool(light_history)
    return DumpHistoryConfig(
        dumps_enabled=bool(dumps_enabled),
        dump_any=bool(dump_any),
        jit_forces=bool(jit_forces_resolved),
        light_history=bool(light_history_resolved),
        track_history=not bool(light_history_resolved),
        disabled_jit_for_dumps=bool(disabled_jit),
    )


def resolve_chunked_scan_config(
    *,
    use_scan: bool,
    state_has_tracer: bool,
    scan_fallback_enabled: bool,
    chunked_env: str | None,
) -> ChunkedScanConfig:
    """Resolve chunked scan and fallback availability for traced/differentiated solves."""
    force_chunked_scan = env_flag_enabled("1" if chunked_env is None else chunked_env)
    differentiating_scan = bool(use_scan) and bool(state_has_tracer)
    if force_chunked_scan and (not bool(use_scan)):
        force_chunked_scan = False
    if differentiating_scan:
        force_chunked_scan = False
        scan_fallback_enabled = False
    return ChunkedScanConfig(
        force_chunked_scan=bool(force_chunked_scan),
        scan_fallback_enabled=bool(scan_fallback_enabled),
        differentiating_scan=bool(differentiating_scan),
    )


def resolve_host_residual_metric_config(
    *,
    backend_name: str,
    fsq1_norms_env: str | None,
    residual_metrics_env: str | None,
) -> HostResidualMetricConfig:
    """Resolve host metric collection policy for accelerated residual solves."""

    fsq1_norms_value = str("auto" if fsq1_norms_env is None else fsq1_norms_env).strip().lower()
    if fsq1_norms_value == "auto":
        fsq1_norms_on_accelerator = str(backend_name).strip().lower() != "cpu"
    else:
        fsq1_norms_on_accelerator = env_flag_enabled_with_off(fsq1_norms_value)

    residual_metrics_value = str("auto" if residual_metrics_env is None else residual_metrics_env).strip().lower()
    if residual_metrics_value == "auto":
        residual_metrics_on_accelerator = False
    else:
        residual_metrics_on_accelerator = env_flag_enabled_with_off(residual_metrics_value)

    return HostResidualMetricConfig(
        fsq1_norms_on_accelerator=bool(fsq1_norms_on_accelerator),
        residual_metrics_on_accelerator=bool(residual_metrics_on_accelerator),
    )


def resolve_host_profile_setup(*, backend_name: str, profile_setup_env: str | None) -> bool:
    """Resolve host-side flux-profile setup policy for residual solves."""

    value = str("auto" if profile_setup_env is None else profile_setup_env).strip().lower()
    if value == "auto":
        return str(backend_name).strip().lower() != "cpu"
    return env_flag_enabled_with_off(value)


def resolve_axis_reset_config(
    *,
    force_axis_reset_env: str | None,
    axis_reset_always_3d_env: str | None,
    axis_reset_fsq_min_env: str | None,
) -> AxisResetConfig:
    """Resolve VMEC-style initial-axis reset environment policy."""

    try:
        fsq_min_value = str("1.0" if axis_reset_fsq_min_env is None else axis_reset_fsq_min_env).strip()
        axis_reset_fsq_min = float(fsq_min_value) if fsq_min_value else 0.0
    except Exception:
        axis_reset_fsq_min = 0.0
    if axis_reset_fsq_min < 0.0:
        axis_reset_fsq_min = 0.0

    return AxisResetConfig(
        force_axis_reset=env_flag_enabled("0" if force_axis_reset_env is None else force_axis_reset_env),
        axis_reset_always_3d=env_flag_enabled("0" if axis_reset_always_3d_env is None else axis_reset_always_3d_env),
        axis_reset_fsq_min=float(axis_reset_fsq_min),
    )


def resolve_setup_host_enforce(
    *,
    setup_host_enforce_env: str | None,
    host_update_assembly: bool,
    use_scan: bool,
    state_has_tracer: bool,
    backend_name: str,
) -> bool:
    """Resolve host-side setup row/gauge enforcement policy."""

    value = str("auto" if setup_host_enforce_env is None else setup_host_enforce_env).strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return False
    if value in ("1", "true", "yes", "on", "force"):
        return not bool(state_has_tracer)
    return (
        (not bool(host_update_assembly))
        and (not bool(use_scan))
        and (not bool(state_has_tracer))
        and (str(backend_name).strip().lower() != "cpu")
    )


def resolve_nstep_screen(*, indata_nstep: int, override_env: str | None) -> int:
    """Resolve VMEC screen cadence with the legacy NSTEP override semantics."""
    nstep_screen = int(indata_nstep)
    nstep_override = "" if override_env is None else str(override_env).strip()
    if nstep_override not in ("", "0"):
        try:
            nstep_screen = int(nstep_override)
        except Exception:
            pass
    if nstep_screen < 1:
        nstep_screen = 1
    return int(nstep_screen)


def normalize_debug_print_mode(mode_env: str | None) -> str:
    """Normalize JAX debug printing mode, falling back to debug_print."""
    mode = str("" if mode_env is None else mode_env).strip().lower()
    if mode not in ("debug_print", "debug_callback", "io_callback"):
        mode = "debug_print"
    return mode


def resolve_debug_print_config(
    *,
    print_env: str | None,
    mode_env: str | None,
    ordered_env: str | None,
    io_callback_available: bool = True,
) -> DebugPrintConfig:
    """Resolve print liveness/mode/ordering without importing JAX."""
    mode = normalize_debug_print_mode(mode_env)
    if mode == "io_callback" and not bool(io_callback_available):
        mode = "debug_print"
    return DebugPrintConfig(
        print_live=env_flag_enabled(print_env),
        mode=mode,
        ordered=env_flag_enabled(ordered_env),
    )

"""Setup-policy helpers for the VMEC residual-iteration solve."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

__all__ = [
    "FreeBoundarySetupPolicy",
    "ResidualCacheKeys",
    "build_residual_profile_setup",
    "build_residual_ptau_bindings",
    "build_residual_static_grid_setup",
    "build_residual_cache_keys",
    "free_boundary_pressure_edge_scale",
    "grid_matches_vmec_static_grid",
    "resolve_free_boundary_setup_policy",
]


_FALSE_STRINGS = ("", "0", "false", "no", "off")


@dataclass(frozen=True)
class FreeBoundarySetupPolicy:
    """Resolved free-boundary and strict-update setup decisions."""

    free_boundary_enabled: bool
    free_boundary_provider_kind: str
    direct_free_boundary_provider: bool
    freeb_nvacskip: int
    freeb_nvskip0: int
    freeb_couple_edge: bool
    use_scan: bool
    freeb_sample_external: bool
    jit_strict_update_enabled: bool
    update_work: int
    cpu_work_limit: int


@dataclass(frozen=True)
class ResidualCacheKeys:
    """Cache keys for fixed-boundary residual force and scan runners."""

    static_key: tuple[Any, ...]
    wout_key: tuple[Any, ...]
    edge_signature_key: Any
    edge_value_key: Any


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() not in _FALSE_STRINGS


def grid_matches_vmec_static_grid(current_grid: Any, vmec_grid: Any) -> bool:
    """Return whether an existing static grid already matches VMEC's grid."""

    try:
        theta_curr = np.asarray(current_grid.theta)
        zeta_curr = np.asarray(current_grid.zeta)
        return bool(
            int(current_grid.nfp) == int(vmec_grid.nfp)
            and theta_curr.shape == np.asarray(vmec_grid.theta).shape
            and zeta_curr.shape == np.asarray(vmec_grid.zeta).shape
            and np.allclose(theta_curr, np.asarray(vmec_grid.theta))
            and np.allclose(zeta_curr, np.asarray(vmec_grid.zeta))
        )
    except Exception:
        return False


def build_residual_static_grid_setup(
    *,
    static: Any,
    build_static_func: Any,
    vmec_angle_grid_func: Any,
) -> tuple[Any, Any]:
    """Rebuild static data on VMEC's internal force grid when needed."""

    cfg = static.cfg
    grid_vmec = vmec_angle_grid_func(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    if not grid_matches_vmec_static_grid(static.grid, grid_vmec):
        static = build_static_func(
            cfg,
            grid=grid_vmec,
            mgrid_metadata=getattr(static, "mgrid_metadata", None),
            free_boundary_extcur=getattr(static, "free_boundary_extcur", None),
        )
    return static, static.cfg


def build_residual_profile_setup(
    *,
    indata: Any,
    static: Any,
    s: Any,
    signgs: int,
    idx00: int,
    state0: Any,
    state0_has_tracer: bool,
    host_update_assembly: bool,
    host_profile_setup: bool,
    build_wout_like_profiles_func: Any,
    resolve_residual_trig_func: Any,
    vmec_trig_tables_func: Any,
    tree_has_tracer_func: Any,
    jnp_module: Any,
    setup_phase_timings: dict[str, float] | None = None,
    timing_enabled: bool = False,
    perf_counter_func: Any = None,
) -> tuple[Any, Any]:
    """Build profile data and VMEC-grid trig tables for the residual loop."""

    def _timer_start() -> float | None:
        if not bool(timing_enabled) or perf_counter_func is None:
            return None
        return float(perf_counter_func())

    def _record_timing(key: str, start: float | None) -> None:
        if start is None or setup_phase_timings is None or perf_counter_func is None:
            return
        setup_phase_timings[key] = float(setup_phase_timings.get(key, 0.0)) + (
            float(perf_counter_func()) - float(start)
        )

    profile_numpy_patch = None
    if bool(host_update_assembly) or bool(host_profile_setup):
        try:
            from vmec_jax.vmec_numpy_forces import _numpy_module_patch as profile_numpy_patch
        except Exception:
            profile_numpy_patch = None
    if bool(state0_has_tracer):
        profile_numpy_patch = None

    with profile_numpy_patch() if profile_numpy_patch is not None else nullcontext():
        _t_profile_data = _timer_start()
        s_profile = s
        if profile_numpy_patch is not None:
            from vmec_jax.vmec_numpy_forces import _wrap as _np_wrap

            s_profile = _np_wrap(np.asarray(s))
        profile_setup = build_wout_like_profiles_func(
            indata=indata,
            static=static,
            s_profile=s_profile,
            signgs=signgs,
            idx00=idx00,
            prefer_host_default_profiles=not bool(state0_has_tracer),
            s_profile_has_tracer=tree_has_tracer_func(s_profile),
        )
        _record_timing("setup_profile_data", _t_profile_data)

    _t_trig_tables = _timer_start()
    trig = resolve_residual_trig_func(
        state0=state0,
        static=static,
        wout_like=profile_setup.wout_like,
        vmec_trig_tables_func=vmec_trig_tables_func,
        jnp_module=jnp_module,
    )
    _record_timing("setup_trig_tables", _t_trig_tables)
    return profile_setup.wout_like, trig


def build_residual_ptau_bindings(
    *,
    s: Any,
    has_jax_value: bool,
    s_has_tracer: bool,
    pshalf_from_s_np_func: Any,
    pshalf_from_s_jax_func: Any,
    build_context_func: Any,
    compute_jit_func: Any,
    ptau_minmax_host_helper: Any,
    ptau_minmax_helper: Any,
    scan_ptau_minmax_host_func: Any,
    scan_ptau_minmax_jax_func: Any,
    accepted_control_ptau_arrays_helper: Any,
    scan_kernel_arrays_from_k_func: Any,
    has_jax_func: Any,
) -> tuple[Any, Any, Any, Any]:
    """Bind ptau min/max helpers once during residual-solve setup."""

    context = build_context_func(
        s,
        has_jax=bool(has_jax_value),
        s_has_tracer=bool(s_has_tracer),
        pshalf_from_s_np=pshalf_from_s_np_func,
        pshalf_from_s_jax=pshalf_from_s_jax_func,
    )
    minmax_from_k_host = partial(
        ptau_minmax_host_helper,
        ptau_context=context,
        compute_jit=compute_jit_func,
        ptau_minmax_host_func=scan_ptau_minmax_host_func,
    )
    minmax = partial(
        ptau_minmax_helper,
        ptau_context=context,
        has_jax_func=has_jax_func,
        compute_jit=compute_jit_func,
        pshalf_from_s_jax=pshalf_from_s_jax_func,
        ptau_minmax_host_func=scan_ptau_minmax_host_func,
        ptau_minmax_jax_func=scan_ptau_minmax_jax_func,
    )
    accepted_control_arrays = partial(
        accepted_control_ptau_arrays_helper,
        kernel_arrays_from_k=scan_kernel_arrays_from_k_func,
    )
    return context, minmax_from_k_host, minmax, accepted_control_arrays


def resolve_free_boundary_setup_policy(
    cfg: Any,
    *,
    external_field_provider_kind: str | None,
    use_scan: bool,
    freeb_couple_env: str,
    freeb_sample_env: str,
    jit_strict_update_env: str,
    backend_name: str,
    host_update_assembly: bool,
    cpu_work_limit_env: str,
) -> FreeBoundarySetupPolicy:
    """Resolve host setup decisions for free-boundary residual iterations."""

    free_boundary_enabled = bool(getattr(cfg, "lfreeb", False))
    provider_kind = "" if external_field_provider_kind is None else str(external_field_provider_kind).strip().lower()
    direct_provider = provider_kind in ("direct_coils", "coils", "coil")
    freeb_nvacskip = max(1, int(getattr(cfg, "nvacskip", int(getattr(cfg, "nfp", 1)))))
    freeb_nvskip0 = max(1, freeb_nvacskip)
    freeb_couple_edge = bool(free_boundary_enabled) and _truthy_env(freeb_couple_env)
    resolved_use_scan = bool(use_scan)
    if free_boundary_enabled and resolved_use_scan:
        # Free-boundary coupling is currently wired through the VMEC2000
        # control path, including ivacskip-driven reuse.
        resolved_use_scan = False

    sample_external = _truthy_env(freeb_sample_env)
    strict_env = str(jit_strict_update_env or "").strip().lower()
    jit_strict_update_enabled = strict_env not in _FALSE_STRINGS
    update_work = 0
    try:
        cpu_work_limit = int(str(cpu_work_limit_env).strip())
    except Exception:
        cpu_work_limit = 1000
    if strict_env == "auto":
        nrange = int(getattr(cfg, "ntor", 0)) + 1
        if bool(getattr(cfg, "lasym", False)):
            nrange = 2 * int(getattr(cfg, "ntor", 0)) + 1
        update_work = int(getattr(cfg, "ns", 0)) * int(getattr(cfg, "mpol", 0)) * int(nrange)
        backend = str(backend_name or "").strip().lower()
        jit_strict_update_enabled = (backend != "cpu") or (
            backend == "cpu" and (not bool(host_update_assembly)) and update_work >= cpu_work_limit
        )

    return FreeBoundarySetupPolicy(
        free_boundary_enabled=bool(free_boundary_enabled),
        free_boundary_provider_kind=provider_kind,
        direct_free_boundary_provider=bool(direct_provider),
        freeb_nvacskip=int(freeb_nvacskip),
        freeb_nvskip0=int(freeb_nvskip0),
        freeb_couple_edge=bool(freeb_couple_edge),
        use_scan=bool(resolved_use_scan),
        freeb_sample_external=bool(sample_external),
        jit_strict_update_enabled=bool(jit_strict_update_enabled),
        update_work=int(update_work),
        cpu_work_limit=int(cpu_work_limit),
    )


def free_boundary_pressure_edge_scale(
    *,
    free_boundary_enabled: bool,
    indata: Any,
    s: Any,
    eval_profiles_func: Any | None = None,
) -> float | None:
    """Return VMEC's edge pressure scaling for free-boundary coupling.

    VMEC samples pressure on the last half-mesh point for the coupled
    free-boundary force, while the external-field interface often needs the
    boundary value.  This helper returns the ratio ``pressure(s=1) /
    pressure(s_edge_half)`` when the input deck and mesh are available.
    """

    s_arr = np.asarray(s, dtype=float)
    if not bool(free_boundary_enabled) or indata is None or int(s_arr.shape[0]) < 2:
        return None
    try:
        if eval_profiles_func is None:
            from vmec_jax.profiles import eval_profiles as eval_profiles_func

        hs_f = float(s_arr[1] - s_arr[0])
        sedge = hs_f * (float(int(s_arr.shape[0])) - 1.5)
        prof_edge = eval_profiles_func(indata, np.asarray([sedge], dtype=float))
        prof_one = eval_profiles_func(indata, np.asarray([1.0], dtype=float))
        p_edge = float(np.asarray(prof_edge.get("pressure", np.asarray([0.0], dtype=float))).reshape(-1)[0])
        p_one = float(np.asarray(prof_one.get("pressure", np.asarray([0.0], dtype=float))).reshape(-1)[0])
        return float(p_one / p_edge) if p_edge != 0.0 else 0.0
    except Exception:
        return None


def build_residual_cache_keys(
    *,
    static: Any,
    wout_like: Any,
    edge_Rcos: Any,
    edge_Rsin: Any,
    edge_Zcos: Any,
    edge_Zsin: Any,
    constraint_tcon0: float | None,
    hash_array_bytes_func,
    edge_signature_key_func,
    edge_value_key_func,
) -> ResidualCacheKeys:
    """Build stable cache keys for residual force kernels and scan runners."""

    static_key = (
        int(static.cfg.mpol),
        int(static.cfg.ntor),
        int(static.cfg.ntheta),
        int(static.cfg.nzeta),
        int(static.cfg.nfp),
        int(static.cfg.ns),
        bool(static.cfg.lasym),
        hash_array_bytes_func(static.modes.m),
        hash_array_bytes_func(static.modes.n),
        hash_array_bytes_func(static.grid.theta),
        hash_array_bytes_func(static.grid.zeta),
    )
    wout_key = (
        int(wout_like.nfp),
        int(wout_like.mpol),
        int(wout_like.ntor),
        bool(wout_like.lasym),
        int(wout_like.signgs),
        hash_array_bytes_func(wout_like.phipf),
        hash_array_bytes_func(wout_like.phips),
        hash_array_bytes_func(wout_like.chipf),
        hash_array_bytes_func(wout_like.pres),
        hash_array_bytes_func(wout_like.icurv) if getattr(wout_like, "icurv", None) is not None else None,
        float(constraint_tcon0) if constraint_tcon0 is not None else None,
    )
    return ResidualCacheKeys(
        static_key=static_key,
        wout_key=wout_key,
        edge_signature_key=edge_signature_key_func(edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin),
        edge_value_key=edge_value_key_func(edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin),
    )

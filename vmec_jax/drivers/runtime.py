"""Runtime setup helpers for CLI/API driver entry points."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RestartContext:
    """Normalized restart inputs for a fixed-boundary run."""

    cfg: Any
    restart_state: Any | None
    restart_wout: Any | None
    restart_solver_state: Any | None


@dataclass(frozen=True)
class ExternalFieldProviderContext:
    """Normalized direct-external-field provider setup for a driver run."""

    direct_external_provider: bool
    provider_kind: str
    provider_static: Any


def maybe_enable_compilation_cache(
    *,
    accelerator_requested: bool = False,
    default_compilation_cache_dir: Callable[[], str | None],
    path_cls: Callable[[Any], Any] = Path,
) -> None:
    """Enable JAX's persistent compilation cache when policy/env allow it."""

    if os.getenv("VMEC_JAX_COMPILATION_CACHE", "").strip().lower() in ("0", "false", "no", "off"):
        return
    if os.getenv("VMEC_JAX_DISABLE_COMPILATION_CACHE", "") not in ("", "0"):
        return

    cache_dir = default_compilation_cache_dir()
    if (
        not cache_dir
        and bool(accelerator_requested)
        and "JAX_COMPILATION_CACHE_DIR" not in os.environ
        and "VMEC_JAX_COMPILATION_CACHE_DIR" not in os.environ
        and "VMEC_JAX_COMPILATION_CACHE" not in os.environ
    ):
        # solver_device="gpu" is a vmec_jax runtime request, not a JAX env var,
        # so _compat cannot see it during package import. Enable the same
        # machine-scoped cache policy here before the first solve compilation.
        os.environ["VMEC_JAX_COMPILATION_CACHE"] = "1"
        try:
            cache_dir = default_compilation_cache_dir()
        finally:
            os.environ.pop("VMEC_JAX_COMPILATION_CACHE", None)
    if str(cache_dir).strip().lower() in ("disabled", "0", "false", "no", "off"):
        return
    if not cache_dir:
        return
    try:
        import jax
        from jax.experimental.compilation_cache import compilation_cache

        cache_path = path_cls(cache_dir)
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fall back to /tmp when the home cache is not writable.
            try:
                cache_path = path_cls("/tmp/vmec_jax/jax_compilation_cache")
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                return
        cache_dir = str(cache_path)
        compilation_cache.set_cache_dir(cache_dir)
        try:
            jax.config.update("jax_enable_compilation_cache", True)
        except Exception:
            pass
        try:
            jax.config.update("jax_compilation_cache_dir", cache_dir)
        except Exception:
            pass
        try:
            min_compile = os.getenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "0")
            jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile))
            min_entry = os.getenv("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry))
            xla_caches = os.getenv("VMEC_JAX_PERSISTENT_CACHE_XLA_CACHES", "").strip()
            if not xla_caches and bool(accelerator_requested):
                xla_caches = "xla_gpu_per_fusion_autotune_cache_dir"
            if xla_caches.lower() not in ("", "none", "0", "false", "no", "off"):
                jax.config.update("jax_persistent_cache_enable_xla_caches", xla_caches)
            max_size = os.getenv("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", "")
            if max_size:
                jax.config.update("jax_compilation_cache_max_size", int(max_size))
            explain = os.getenv("VMEC_JAX_EXPLAIN_CACHE_MISSES", "")
            if explain.strip().lower() not in ("", "0", "false", "no"):
                jax.config.update("jax_explain_cache_misses", True)
        except Exception:
            pass
    except Exception:
        return


def resolve_restart_context(
    *,
    cfg,
    restart_state,
    restart_wout_path,
    restart_solver_state,
    ns_override,
    read_wout_func: Callable[[Any], Any],
    state_from_wout_func: Callable[[Any], Any],
    replace_func: Callable[..., Any],
    path_cls: Callable[[Any], Any] = Path,
) -> RestartContext:
    """Resolve restart WOUT/state inputs and enforce grid consistency.

    The driver historically exposed ``read_wout`` and ``state_from_wout`` as
    module-level monkeypatch points. Accepting those callables keeps that
    compatibility while moving the mechanical restart normalization out of the
    long public workflow.
    """

    restart_state_eff = restart_state
    restart_wout = None
    if restart_wout_path is not None:
        restart_wout = read_wout_func(path_cls(restart_wout_path))
        restart_state_eff = state_from_wout_func(restart_wout)

    cfg_eff = cfg
    restart_solver_state_eff = restart_solver_state
    if restart_state_eff is not None:
        restart_ns = int(restart_state_eff.layout.ns)
        if ns_override is not None and int(ns_override) != restart_ns:
            raise ValueError(f"restart_state ns={restart_ns} does not match ns_override={ns_override}")
        cfg_eff = replace_func(cfg_eff, ns=int(restart_ns))
        if restart_solver_state_eff is not None:
            # Ensure resume checkpoints align with the provided restart state.
            try:
                restart_solver_state_eff = dict(restart_solver_state_eff)
                restart_solver_state_eff["state_checkpoint"] = restart_state_eff
            except Exception:
                pass
    elif ns_override is not None:
        cfg_eff = replace_func(cfg_eff, ns=int(ns_override))

    return RestartContext(
        cfg=cfg_eff,
        restart_state=restart_state_eff,
        restart_wout=restart_wout,
        restart_solver_state=restart_solver_state_eff,
    )


def resolve_external_field_provider_context(
    *,
    external_field_provider_kind: str | None,
    external_field_provider_static: Any,
    external_field_provider_params: Any,
    getenv: Callable[[str, str], str] = os.getenv,
    build_coil_field_geometry_func: Callable[[Any], Any] | None = None,
) -> ExternalFieldProviderContext:
    """Resolve direct-provider mode and optional host-side coil geometry cache."""

    provider_kind = "" if external_field_provider_kind is None else str(external_field_provider_kind).strip().lower()
    direct_external_provider = external_field_provider_kind is not None and provider_kind not in (
        "",
        "mgrid",
        "legacy_mgrid",
    )
    provider_static_eff = external_field_provider_static
    should_cache_coil_geometry = (
        bool(direct_external_provider)
        and provider_kind in ("direct_coils", "coils", "coil")
        and external_field_provider_params is not None
        and (
            provider_static_eff is None
            or (isinstance(provider_static_eff, dict) and "coil_geometry" not in provider_static_eff)
        )
        and getenv("VMEC_JAX_FREEB_DISABLE_COIL_GEOMETRY_CACHE", "").strip().lower()
        not in ("1", "true", "yes", "on")
    )
    if should_cache_coil_geometry:
        try:
            if build_coil_field_geometry_func is None:
                from ..external_fields import build_coil_field_geometry as build_coil_field_geometry_func

            jit_sampler_env = getenv("VMEC_JAX_FREEB_JIT_COIL_SAMPLER", "1").strip().lower()
            static_base = {} if provider_static_eff is None else dict(provider_static_eff)
            provider_static_eff = {
                **static_base,
                "coil_geometry": build_coil_field_geometry_func(external_field_provider_params),
                "regularization_epsilon": getattr(external_field_provider_params, "regularization_epsilon", 0.0),
                "chunk_size": getattr(external_field_provider_params, "chunk_size", None),
                "cache_scope": "host_forward_only",
                "jit_sampler": jit_sampler_env not in ("", "0", "false", "no"),
            }
        except Exception:
            # Preserve custom provider-like objects that are incompatible with
            # the built-in coil geometry cache.
            provider_static_eff = external_field_provider_static

    return ExternalFieldProviderContext(
        direct_external_provider=bool(direct_external_provider),
        provider_kind=provider_kind,
        provider_static=provider_static_eff,
    )


__all__ = [
    "ExternalFieldProviderContext",
    "RestartContext",
    "maybe_enable_compilation_cache",
    "resolve_external_field_provider_context",
    "resolve_restart_context",
]

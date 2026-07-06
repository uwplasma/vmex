"""Replay and tangent-cache policy helpers for fixed-boundary exact callbacks."""

from __future__ import annotations

import os


def _env_flag(name: str) -> bool | None:
    """Return a boolean environment override without mutating optimizer state."""

    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _positive_int_env(name: str, default: int) -> int:
    """Return a positive integer environment setting or a safe default."""

    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def optimizer_backend_name(solver_device_name: str | None) -> str:
    """Return the active optimizer backend name without changing device policy."""

    backend = str(solver_device_name or "").strip().lower()
    if backend:
        return backend
    try:
        from ..._compat import jax as _jax

        return str(_jax.default_backend()).strip().lower() if _jax is not None else "cpu"
    except Exception:
        return "cpu"


def _backend_is_accelerator(backend: str) -> bool:
    """Return true for JAX accelerator backends that favor coarse replay buckets."""

    normalized = str(backend).strip().lower()
    return normalized in {"gpu", "cuda", "rocm"} or normalized.startswith(("gpu:", "cuda:", "rocm:"))


def dynamic_replay_bucket_for_backend(backend: str) -> int:
    """Return the dynamic replay bucket used for optimizer provenance.

    The replay implementation defaults to coarser buckets on accelerators and
    smaller buckets on CPU.  This helper mirrors that policy while respecting an
    explicit optimizer device selection, so diagnostics remain meaningful even
    when the outer Python process default backend differs from the callback
    backend.
    """

    default = 128 if _backend_is_accelerator(backend) else 32
    return _positive_int_env("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", default)


def dynamic_replay_mode_from_env() -> str:
    """Return the configured dynamic replay linearization strategy."""

    mode = os.getenv("VMEC_JAX_DYNAMIC_REPLAY_MODE", "basepoint").strip().lower()
    if mode in ("whole_scan", "scan", "full_scan"):
        return "whole_scan"
    return "basepoint"


def _optimizer_lasym(optimizer) -> bool:
    """Return LASYM status for real optimizers and lightweight test doubles."""

    static = getattr(optimizer, "_static", None)
    return bool(getattr(getattr(static, "cfg", None), "lasym", False))


def _optimizer_has_stellarator_asymmetry(optimizer) -> bool:
    """Return whether replay policy should use LASYM/asymmetric safeguards.

    The production optimizer exposes ``_has_stellarator_asymmetric_configuration``.
    Some unit tests intentionally use minimal objects to exercise metadata paths
    without constructing a full VMEC static state; for those, fall back to the
    parameter-spec kinds and static ``cfg.lasym`` flag.
    """

    checker = getattr(optimizer, "_has_stellarator_asymmetric_configuration", None)
    if callable(checker):
        try:
            return bool(checker())
        except AttributeError:
            pass
    specs = getattr(optimizer, "_specs", ())
    if any(str(getattr(spec, "kind", "")).lower() in ("rs", "zc") for spec in specs):
        return True
    return _optimizer_lasym(optimizer)


def lasym_replay_column_chunk(optimizer, n_params: int) -> int | None:
    """Replay-column chunk heuristic for dense exact Jacobians."""

    env_override = os.environ.get("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK")
    if env_override is not None:
        from ...discrete_adjoint import _replay_column_chunk_override

        handled, requested = _replay_column_chunk_override(env_override)
        if handled:
            return requested
    if os.environ.get("VMEC_JAX_REPLAY_COLUMN_CHUNK") is not None:
        return None
    backend_name = None
    if optimizer._solver_device_name is not None:
        backend_name = str(optimizer._solver_device_name).lower()
    else:
        try:
            from ..._compat import jax as _jax

            backend_name = str(_jax.default_backend()).lower()
        except Exception:
            backend_name = None
    if backend_name in ("gpu", "cuda", "rocm"):
        if int(n_params) < 24:
            return None
        if _optimizer_lasym(optimizer):
            # LASYM doubles the boundary columns and remains more memory
            # sensitive on GPU; keep the older conservative replay chunks.
            return 8
        if int(n_params) <= 64:
            return int(n_params)
        if int(n_params) <= 128:
            return max(24, int(n_params) // 2)
        return 64
    if backend_name == "tpu":
        return None
    if not _optimizer_lasym(optimizer):
        return None
    if int(n_params) >= 64:
        return 8
    if int(n_params) >= 32:
        return 4
    return None


def precompute_linear_operator_initial_tangents_enabled(optimizer, n_params: int) -> bool:
    """Whether matrix-free operators should cache initial-state tangent columns."""

    if int(n_params) <= 0:
        return False
    flag = os.getenv("VMEC_JAX_OPT_LINEAR_OPERATOR_INITIAL_TANGENTS")
    if flag is not None:
        return flag.strip().lower() not in ("", "0", "false", "no", "off")
    backend = optimizer_backend_name(getattr(optimizer, "_solver_device_name", None))
    if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
        return False
    if _optimizer_has_stellarator_asymmetry(optimizer):
        return False
    min_dofs = int(os.getenv("VMEC_JAX_OPT_LINEAR_OPERATOR_INITIAL_TANGENT_MIN_DOFS", "64"))
    max_dofs = int(os.getenv("VMEC_JAX_OPT_LINEAR_OPERATOR_INITIAL_TANGENT_MAX_DOFS", "128"))
    return min_dofs <= int(n_params) <= max_dofs


def scalar_gradient_initial_tangents_enabled(optimizer, n_params: int) -> bool:
    """Whether scalar-adjoint gradients should project cached initial tangents."""

    if int(n_params) <= 0:
        return False
    flag = os.getenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENTS")
    if flag is not None:
        return flag.strip().lower() not in ("", "0", "false", "no", "off")
    backend = optimizer_backend_name(getattr(optimizer, "_solver_device_name", None))
    if backend not in ("cpu", "gpu", "cuda", "rocm", "tpu", "metal"):
        return False
    if _optimizer_has_stellarator_asymmetry(optimizer):
        return False
    min_dofs = int(os.getenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENT_MIN_DOFS", "24"))
    default_max_dofs = "128" if backend == "cpu" else "256"
    max_dofs = int(os.getenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENT_MAX_DOFS", default_max_dofs))
    return min_dofs <= int(n_params) <= max_dofs


def projected_replay_residuals_enabled(optimizer, n_params: int | None = None) -> bool:
    """Whether dense Jacobians should project replayed tangents without an intermediate sync."""

    flag = os.getenv("VMEC_JAX_OPT_PROJECTED_REPLAY_RESIDUALS")
    if flag is not None:
        return flag.strip().lower() in ("1", "true", "yes", "on")
    solver_device_name = getattr(optimizer, "_solver_device_name", None)
    if solver_device_name is None:
        try:
            from ..._compat import jax as _jax

            backend = str(_jax.default_backend()).strip().lower() if _jax is not None else ""
        except Exception:
            return False
    else:
        backend = optimizer_backend_name(solver_device_name)
    if backend not in ("cpu", "gpu", "cuda", "rocm"):
        return False
    if n_params is None:
        return False
    static = getattr(optimizer, "_static", None)
    if bool(getattr(getattr(static, "cfg", None), "lasym", False)):
        return False
    # Projected replay avoids materializing full state tangent columns on the
    # host before residual projection.  Recent QA/QH/QP budget probes show this
    # pays off on CPU as well as accelerators once the exact callback has a
    # multi-column dense Jacobian.
    min_params = 8 if backend == "cpu" else 48
    return int(n_params) >= min_params


def fused_projected_replay_enabled() -> bool:
    """Whether projected replay should fuse replay and residual projection when possible."""

    flag = os.getenv("VMEC_JAX_OPT_FUSED_PROJECTED_REPLAY", "").strip().lower()
    if flag:
        return flag in ("1", "true", "yes", "on")
    return False


def chunked_projected_replay_projection_enabled(
    optimizer,
    column_chunk: int | None,
    n_params: int,
) -> bool:
    """Whether to project residual tangents immediately after each replay chunk."""

    if column_chunk is None:
        return False
    if int(n_params) <= int(column_chunk):
        return False
    flag = os.getenv("VMEC_JAX_OPT_CHUNKED_PROJECTED_REPLAY_PROJECTION", "").strip().lower()
    if flag:
        return flag in ("1", "true", "yes", "on")
    backend = optimizer_backend_name(getattr(optimizer, "_solver_device_name", None))
    if backend not in ("gpu", "cuda", "rocm"):
        return False
    if _optimizer_lasym(optimizer):
        return False
    return True


def exact_replay_policy_metadata(optimizer, n_params: int | None = None) -> dict[str, object]:
    """Summarize exact-callback replay policy choices without changing them.

    The exact callback has several mathematically equivalent replay routes:
    dense tangent replay, projected replay, chunked projection, scalar-adjoint
    gradients, and accelerator-oriented JVP-only tapes.  This diagnostic record
    exposes the chosen route and its controlling backend/shape conditions so
    optimization histories and benchmark summaries can classify performance
    regressions without storing large Jacobians or mutating profile counters.
    """

    n_params_int = None if n_params is None else int(n_params)
    backend = optimizer_backend_name(getattr(optimizer, "_solver_device_name", None))
    static = getattr(optimizer, "_static", None)
    lasym = _optimizer_lasym(optimizer)
    accelerator = backend in ("gpu", "cuda", "rocm", "tpu", "metal")
    gpu_like = backend in ("gpu", "cuda", "rocm", "tpu", "metal")

    jvp_only_override = _env_flag("VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE")
    basepoint_override = _env_flag("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES")
    column_chunk = None
    projected = False
    chunked_projection = False
    scalar_initial_tangents = False
    linear_operator_initial_tangents = False
    if n_params_int is not None:
        column_chunk = lasym_replay_column_chunk(optimizer, n_params_int)
        projected = projected_replay_residuals_enabled(optimizer, n_params_int)
        chunked_projection = chunked_projected_replay_projection_enabled(
            optimizer,
            column_chunk,
            n_params_int,
        )
        scalar_initial_tangents = scalar_gradient_initial_tangents_enabled(optimizer, n_params_int)
        linear_operator_initial_tangents = precompute_linear_operator_initial_tangents_enabled(
            optimizer,
            n_params_int,
        )

    return {
        "backend": backend,
        "n_parameters": n_params_int,
        "lasym": lasym,
        "projected_replay": bool(projected),
        "projected_replay_reason": "enabled" if projected else "disabled_or_below_threshold",
        "fused_projected_replay": fused_projected_replay_enabled(),
        "column_chunk": None if column_chunk is None else int(column_chunk),
        "chunked_projected_replay_projection": bool(chunked_projection),
        "dynamic_replay_mode": dynamic_replay_mode_from_env(),
        "dynamic_replay_bucket": dynamic_replay_bucket_for_backend(backend),
        "scalar_gradient_initial_tangents": bool(scalar_initial_tangents),
        "linear_operator_initial_tangents": bool(linear_operator_initial_tangents),
        "jvp_only_exact_tape": bool(gpu_like if jvp_only_override is None else jvp_only_override),
        "jvp_only_basepoint_carries": bool(gpu_like if basepoint_override is None else basepoint_override),
        "accelerator_backend": bool(accelerator),
    }

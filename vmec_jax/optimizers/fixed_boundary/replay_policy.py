"""Replay and tangent-cache policy helpers for fixed-boundary exact callbacks."""

from __future__ import annotations

import os


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
        if bool(getattr(optimizer._static.cfg, "lasym", False)):
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
    if not bool(getattr(optimizer._static.cfg, "lasym", False)):
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
    if optimizer._has_stellarator_asymmetric_configuration():
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
    if optimizer._has_stellarator_asymmetric_configuration():
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
    if backend not in ("gpu", "cuda", "rocm"):
        return False
    if n_params is None:
        return False
    static = getattr(optimizer, "_static", None)
    if bool(getattr(getattr(static, "cfg", None), "lasym", False)):
        return False
    # Projected replay only pays off for larger non-LASYM dense Jacobians on
    # accelerator backends.  Current CPU profile shards are neutral to slightly
    # faster on the simpler standard replay path, so leave CPU opt-in through
    # VMEC_JAX_OPT_PROJECTED_REPLAY_RESIDUALS instead of widening the default.
    return int(n_params) >= 48


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
    if bool(getattr(getattr(optimizer._static, "cfg", None), "lasym", False)):
        return False
    return True

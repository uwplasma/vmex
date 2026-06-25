"""Mode-continuation stage policy helpers for fixed-boundary optimization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryModeLimits:
    """Boundary-parameter mode limits for one optimization stage."""

    mode: int
    max_m: int | None = None
    max_n: int | None = None
    label: str | None = None


def qs_stage_modes(
    *,
    max_mode: int,
    use_mode_continuation: bool,
    continuation_nfev: int,
) -> list[int]:
    """Repeated mode-continuation sequence used by the QA/QH/QP examples."""

    if bool(use_mode_continuation) and int(max_mode) > 1 and int(continuation_nfev) > 0:
        modes: list[int] = []
        for mode in range(1, int(max_mode) + 1):
            modes.extend([mode] * (2 if mode == 1 else 3))
        return modes
    return [int(max_mode)]


def repeated_stage_modes(
    *,
    max_mode: int,
    use_mode_continuation: bool,
    continuation_nfev: int,
    repeats: int = 5,
) -> list[int]:
    """Same-mode repeated continuation used by the QI example."""

    del continuation_nfev
    if bool(use_mode_continuation) and int(max_mode) > 1:
        return [int(max_mode)] * max(1, int(repeats))
    return [int(max_mode)]


def qs_stage_budget(
    *,
    stage_mode: int,
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
) -> int:
    """Outer residual/Jacobian budget for one fixed-boundary stage."""

    if int(max_nfev) <= 0:
        raise ValueError("max_nfev must be a positive integer for outer optimization stages.")
    if int(stage_mode) == int(max_mode):
        return int(max_nfev)
    return int(continuation_nfev) if int(continuation_nfev) > 0 else int(max_nfev)


def normalize_boundary_mode_limits(stage_mode) -> BoundaryModeLimits:
    """Normalize an int/tuple/dict stage descriptor into mode limits."""

    if isinstance(stage_mode, BoundaryModeLimits):
        return stage_mode
    if isinstance(stage_mode, dict):
        max_m = stage_mode.get("max_m")
        max_n = stage_mode.get("max_n")
        mode_raw = stage_mode.get("mode", stage_mode.get("max_mode"))
        if mode_raw is None:
            finite_limits = [value for value in (max_m, max_n) if value is not None]
            if not finite_limits:
                raise ValueError("Boundary mode-limit dictionaries require mode/max_mode or max_m/max_n.")
            mode_raw = max(int(value) for value in finite_limits)
        return BoundaryModeLimits(
            mode=int(mode_raw),
            max_m=None if max_m is None else int(max_m),
            max_n=None if max_n is None else int(max_n),
            label=stage_mode.get("label"),
        )
    if isinstance(stage_mode, (tuple, list)):
        if len(stage_mode) == 2:
            max_m, max_n = (None if value is None else int(value) for value in stage_mode)
            finite_limits = [value for value in (max_m, max_n) if value is not None]
            if not finite_limits:
                raise ValueError("At least one of max_m or max_n must be finite.")
            return BoundaryModeLimits(mode=max(finite_limits), max_m=max_m, max_n=max_n)
        if len(stage_mode) == 3:
            mode, max_m, max_n = stage_mode
            return BoundaryModeLimits(
                mode=int(mode),
                max_m=None if max_m is None else int(max_m),
                max_n=None if max_n is None else int(max_n),
            )
        raise ValueError("Boundary stage tuples must be (max_m, max_n) or (mode, max_m, max_n).")
    return BoundaryModeLimits(mode=int(stage_mode))


def describe_boundary_mode_limits(stage_mode) -> str:
    """Return a compact label for a boundary-mode stage descriptor."""

    limits = normalize_boundary_mode_limits(stage_mode)
    max_m = limits.mode if limits.max_m is None else limits.max_m
    max_n = limits.mode if limits.max_n is None else limits.max_n
    base = f"mode{limits.mode:02d}_m{int(max_m):02d}_n{int(max_n):02d}"
    return f"{base}_{limits.label}" if limits.label else base


__all__ = [
    "BoundaryModeLimits",
    "describe_boundary_mode_limits",
    "normalize_boundary_mode_limits",
    "qs_stage_budget",
    "qs_stage_modes",
    "repeated_stage_modes",
]

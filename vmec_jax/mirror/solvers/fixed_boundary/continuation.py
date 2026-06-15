"""Continuation helpers for fixed-boundary mirror solves."""

from __future__ import annotations

from .diagnostics import ensure_finite_pressure_scale
from ...core.profiles import PressureProfile


def scaled_pressure_profile(pressure: PressureProfile, scale: float) -> PressureProfile:
    """Return a pressure profile with coefficients scaled by ``scale``."""
    scale = ensure_finite_pressure_scale(scale)
    return PressureProfile(coefficients=pressure.coefficients * scale, gamma=pressure.gamma)


def pressure_stage_profiles(
    pressure: PressureProfile,
    stages,
) -> tuple[tuple[int, float, PressureProfile], ...]:
    """Return indexed pressure-continuation stages."""
    if stages is None:
        stages = (1.0,)
    normalized = tuple(float(stage) for stage in stages)
    if not normalized:
        raise ValueError("pressure continuation must contain at least one stage")
    return tuple(
        (idx, ensure_finite_pressure_scale(scale), scaled_pressure_profile(pressure, scale))
        for idx, scale in enumerate(normalized)
    )

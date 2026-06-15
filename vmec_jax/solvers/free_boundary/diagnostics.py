"""Free-boundary diagnostic adapters used by VMEC solve paths."""

from __future__ import annotations

from typing import Any

from ...state import VMECState


def sample_free_boundary_external_field(*, state: VMECState, static) -> dict[str, Any]:
    """Sample external-field boundary diagnostics for a solve/static pair."""

    from ...free_boundary import sample_external_vacuum_diagnostics

    plascur = float(getattr(static, "free_boundary_plascur", 0.0) or 0.0)
    return sample_external_vacuum_diagnostics(
        state=state,
        static=static,
        plascur=plascur,
    )

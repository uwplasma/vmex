"""Setup-policy helpers for the VMEC residual-iteration solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FreeBoundarySetupPolicy",
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

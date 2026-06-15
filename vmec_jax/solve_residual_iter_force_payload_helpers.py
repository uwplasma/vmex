"""Force-payload postprocessing seams for residual iteration."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from ._compat import jnp
from .solve_force_payload_helpers import zero_edge_rz_force_blocks
from .vmec_residue import vmec_gcx2_from_tomnsps
from .vmec_tomnsp import TomnspsRZL

__all__ = [
    "ResidualForceMetricPayload",
    "metric_force_payload_after_edge_policy",
    "residual_force_gcx2_after_edge_policy",
    "residual_force_z_nan_guard",
    "resolve_residual_force_mask_pack",
]


class ResidualForceMetricPayload(NamedTuple):
    """Metric payload and scalar VMEC force norms after edge policy."""

    frzl_metric: TomnspsRZL
    gcr2: Any
    gcz2: Any
    gcl2: Any


def resolve_residual_force_mask_pack(
    static: Any,
    *,
    include_edge: bool,
    include_edge_residual: bool | None,
) -> tuple[bool, Any | None]:
    """Resolve residual-edge policy and the matching precomputed TOMNSP mask."""

    include_edge_residual_resolved = bool(include_edge if include_edge_residual is None else include_edge_residual)
    mask_pack = None
    if getattr(static, "tomnsps_masks", None) is not None:
        mask_pack = (
            getattr(static, "tomnsps_masks_edge")
            if bool(include_edge_residual_resolved)
            else getattr(static, "tomnsps_masks")
        )
    return include_edge_residual_resolved, mask_pack


def metric_force_payload_after_edge_policy(
    frzl: TomnspsRZL,
    *,
    include_edge: bool,
    zero_edge_rz_force_blocks_func: Callable[..., TomnspsRZL] = zero_edge_rz_force_blocks,
) -> TomnspsRZL:
    """Return the force payload used for R/Z metric scalars.

    The solver keeps the full residual payload for preconditioning and free-boundary
    parity, but optionally removes the LCFS contribution before forming physical
    R/Z force norms.  This helper isolates that policy from the iteration loop.
    """

    if bool(include_edge):
        return frzl
    return zero_edge_rz_force_blocks_func(frzl)


def residual_force_z_nan_guard(frzl: TomnspsRZL):
    """Return a scalar that preserves NaNs from Z-force channels.

    VMEC's scalar norm path can mask the edge contribution before reducing the
    arrays.  This guard keeps a NaN from the full Z-force payload visible in the
    final ``gcz2`` scalar while adding exactly zero for finite payloads.
    """

    z_force_dummy = jnp.sum(jnp.asarray(frzl.fzsc))
    if frzl.fzcs is not None:
        z_force_dummy = z_force_dummy + jnp.sum(jnp.asarray(frzl.fzcs))
    return jnp.where(
        jnp.isnan(z_force_dummy),
        z_force_dummy,
        jnp.asarray(0.0, dtype=jnp.asarray(z_force_dummy).dtype),
    )


def residual_force_gcx2_after_edge_policy(
    frzl: TomnspsRZL,
    *,
    include_edge: bool,
    lconm1: bool,
    s: Any,
    zero_edge_rz_force_blocks_func: Callable[..., TomnspsRZL] = zero_edge_rz_force_blocks,
    gcx2_func: Callable[..., tuple[Any, Any, Any]] = vmec_gcx2_from_tomnsps,
) -> ResidualForceMetricPayload:
    """Return force-norm scalars after solver edge masking and NaN policy."""

    frzl_metric = metric_force_payload_after_edge_policy(
        frzl,
        include_edge=bool(include_edge),
        zero_edge_rz_force_blocks_func=zero_edge_rz_force_blocks_func,
    )
    gcr2, gcz2, gcl2 = gcx2_func(
        frzl=frzl_metric,
        lconm1=bool(lconm1),
        apply_m1_constraints=False,
        include_edge=bool(include_edge),
        apply_scalxc=False,
        s=s,
    )
    return ResidualForceMetricPayload(
        frzl_metric=frzl_metric,
        gcr2=gcr2,
        gcz2=gcz2 + residual_force_z_nan_guard(frzl),
        gcl2=gcl2,
    )

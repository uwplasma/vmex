"""Pure assembly helpers for VMEC residual-objective blocks."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ...._compat import jnp
from ....kernels.residue import (
    vmec_apply_m1_constraints,
    vmec_apply_scalxc_to_tomnsps,
    vmec_zero_m1_zforce,
)
from ....kernels.tomnsp import TomnspsRZL


R_RESIDUAL_BLOCKS = ("frcc", "frss", "frsc", "frcs")
Z_RESIDUAL_BLOCKS = ("fzsc", "fzcs", "fzcc", "fzss")
L_RESIDUAL_BLOCKS = ("flsc", "flcs", "flcc", "flss")
VECTOR_RESIDUAL_BLOCKS = (
    ("frcc", "rz"),
    ("fzsc", "rz"),
    ("flsc", "l"),
    ("frss", "rz"),
    ("fzcs", "rz"),
    ("flcs", "l"),
    ("frsc", "rz"),
    ("fzcc", "rz"),
    ("flcc", "l"),
    ("frcs", "rz"),
    ("fzss", "rz"),
    ("flss", "l"),
)


class ResidualObjectiveTerms(NamedTuple):
    """Represent ResidualObjectiveTerms data for fixed-boundary VMEC solve and implicit differentiation."""
    frzl: TomnspsRZL
    norms: Any
    gcr2: Any
    gcz2: Any
    gcl2: Any
    fsqr2: Any
    fsqz2: Any
    fsql2: Any
    w: Any


def zero_edge_rz_force_block(a: Any, *, preserve_numpy: bool = True) -> Any:
    """Zero the LCFS row in an R/Z force block, leaving lambda blocks untouched."""
    if a is None:
        return None
    if preserve_numpy and isinstance(a, np.ndarray):
        if a.shape[0] < 2:
            return a
        out = a.copy()
        out[-1] = np.zeros_like(a[-1])
        return out
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    return a.at[-1].set(jnp.zeros_like(a[-1]))


def zero_edge_rz_force_blocks(frzl: TomnspsRZL, *, preserve_numpy: bool = True) -> TomnspsRZL:
    """Zero LCFS rows for every R/Z block in a ``TomnspsRZL`` container."""
    return TomnspsRZL(
        frcc=zero_edge_rz_force_block(frzl.frcc, preserve_numpy=preserve_numpy),
        frss=zero_edge_rz_force_block(frzl.frss, preserve_numpy=preserve_numpy),
        fzsc=zero_edge_rz_force_block(frzl.fzsc, preserve_numpy=preserve_numpy),
        fzcs=zero_edge_rz_force_block(frzl.fzcs, preserve_numpy=preserve_numpy),
        flsc=frzl.flsc,
        flcs=frzl.flcs,
        frsc=zero_edge_rz_force_block(getattr(frzl, "frsc", None), preserve_numpy=preserve_numpy),
        frcs=zero_edge_rz_force_block(getattr(frzl, "frcs", None), preserve_numpy=preserve_numpy),
        fzcc=zero_edge_rz_force_block(getattr(frzl, "fzcc", None), preserve_numpy=preserve_numpy),
        fzss=zero_edge_rz_force_block(getattr(frzl, "fzss", None), preserve_numpy=preserve_numpy),
        flcc=getattr(frzl, "flcc", None),
        flss=getattr(frzl, "flss", None),
    )


def prepare_residual_objective_blocks(
    *,
    frzl: TomnspsRZL,
    s: Any,
    zero_m1_zforce: Any,
    lconm1: bool,
    apply_m1_constraints: bool,
    zero_m1_after_m1_constraints: bool,
    apply_scalxc: bool = True,
    zero_edge_rz_blocks: bool = False,
) -> TomnspsRZL:
    """Apply residual-objective block transforms without computing physics kernels."""
    out = frzl
    if not bool(zero_m1_after_m1_constraints):
        out = vmec_zero_m1_zforce(frzl=out, enabled=zero_m1_zforce)
    if bool(apply_m1_constraints):
        out = vmec_apply_m1_constraints(frzl=out, lconm1=bool(lconm1))
    if bool(zero_m1_after_m1_constraints):
        out = vmec_zero_m1_zforce(frzl=out, enabled=zero_m1_zforce)
    if bool(apply_scalxc):
        out = vmec_apply_scalxc_to_tomnsps(frzl=out, s=s)
    if bool(zero_edge_rz_blocks):
        out = zero_edge_rz_force_blocks(out, preserve_numpy=False)
    return out


def _sum_square_blocks(frzl: TomnspsRZL, names: tuple[str, ...], radial_stop: int | None) -> Any:
    total = None
    for name in names:
        block = getattr(frzl, name, None)
        if block is None:
            continue
        arr = jnp.asarray(block)
        if radial_stop is not None:
            arr = arr[:radial_stop]
        value = jnp.sum(arr * arr)
        total = value if total is None else total + value
    if total is None:
        return jnp.asarray(0.0, dtype=jnp.asarray(frzl.frcc).dtype)
    return total


def residual_objective_block_sums(
    *,
    frzl: TomnspsRZL,
    include_edge: bool = False,
) -> tuple[Any, Any, Any]:
    """Return ``(gcr2, gcz2, gcl2)`` from already-prepared residual blocks."""
    ns = int(jnp.asarray(frzl.frcc).shape[0])
    radial_stop = None if bool(include_edge) or ns <= 1 else ns - 1
    gcr2 = _sum_square_blocks(frzl, R_RESIDUAL_BLOCKS, radial_stop)
    gcz2 = _sum_square_blocks(frzl, Z_RESIDUAL_BLOCKS, radial_stop)
    gcl2 = _sum_square_blocks(frzl, L_RESIDUAL_BLOCKS, None)
    return gcr2, gcz2, gcl2


def assemble_residual_objective_terms(
    *,
    frzl: TomnspsRZL,
    norms: Any,
    s: Any,
    w_rz: float,
    w_l: float,
    zero_m1_zforce: Any,
    lconm1: bool,
    apply_m1_constraints: bool,
    zero_m1_after_m1_constraints: bool,
    include_edge: bool = False,
    apply_scalxc: bool = True,
    zero_edge_rz_blocks: bool = False,
    objective_scale: float | None = None,
) -> ResidualObjectiveTerms:
    """Assemble transformed blocks, VMEC-normalized terms, and scalar objective."""
    frzl_prepared = prepare_residual_objective_blocks(
        frzl=frzl,
        s=s,
        zero_m1_zforce=zero_m1_zforce,
        lconm1=bool(lconm1),
        apply_m1_constraints=bool(apply_m1_constraints),
        zero_m1_after_m1_constraints=bool(zero_m1_after_m1_constraints),
        apply_scalxc=bool(apply_scalxc),
        zero_edge_rz_blocks=bool(zero_edge_rz_blocks),
    )
    gcr2, gcz2, gcl2 = residual_objective_block_sums(frzl=frzl_prepared, include_edge=include_edge)
    fsqr2 = norms.r1 * norms.fnorm * gcr2
    fsqz2 = norms.r1 * norms.fnorm * gcz2
    fsql2 = norms.fnormL * gcl2
    w = (w_rz * (fsqr2 + fsqz2)) + (w_l * fsql2)
    if objective_scale is not None:
        w = jnp.asarray(float(objective_scale), dtype=jnp.asarray(w).dtype) * w
    return ResidualObjectiveTerms(
        frzl=frzl_prepared,
        norms=norms,
        gcr2=gcr2,
        gcz2=gcz2,
        gcl2=gcl2,
        fsqr2=fsqr2,
        fsqz2=fsqz2,
        fsql2=fsql2,
        w=w,
    )


def residual_objective_vector(
    *,
    frzl: TomnspsRZL,
    norms: Any,
    w_rz: float,
    w_l: float,
) -> Any:
    """Stack weighted residual blocks for the Gauss-Newton least-squares system."""
    dtype = jnp.asarray(frzl.frcc).dtype
    scale_rz = jnp.sqrt(jnp.asarray(w_rz, dtype=dtype)) * jnp.sqrt(jnp.asarray(norms.r1 * norms.fnorm, dtype=dtype))
    scale_l = jnp.sqrt(jnp.asarray(w_l, dtype=dtype)) * jnp.sqrt(jnp.asarray(norms.fnormL, dtype=dtype))
    parts = []
    for name, group in VECTOR_RESIDUAL_BLOCKS:
        block = getattr(frzl, name, None)
        if block is None:
            continue
        scale = scale_l if group == "l" else scale_rz
        parts.append(jnp.ravel(scale * jnp.asarray(block)))
    return jnp.concatenate(parts, axis=0)

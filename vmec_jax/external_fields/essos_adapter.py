"""Optional adapter from ESSOS coils to vmec_jax direct-coil params."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jnp

from .coils_jax import CoilFieldParams


def from_essos_coils(coils: Any, regularization_epsilon: float = 0.0, chunk_size: int | None = None) -> CoilFieldParams:
    """Convert an ESSOS ``Coils`` object into ``CoilFieldParams``.

    ESSOS is intentionally not imported at module import time.  The adapter
    works with objects exposing the ESSOS ``Coils`` attributes:
    ``dofs_curves``, ``dofs_currents``, ``currents_scale``, ``n_segments``,
    ``nfp``, and ``stellsym``.

    Raises
    ------
    ImportError
        If the supplied object does not expose the expected ESSOS attributes.
    """

    required = ("dofs_curves", "dofs_currents", "currents_scale", "n_segments", "nfp", "stellsym")
    missing = [name for name in required if not hasattr(coils, name)]
    if missing:
        raise ImportError(
            "Cannot convert ESSOS coils: object is missing "
            f"{', '.join(missing)}. Install/import ESSOS and pass an essos.coils.Coils instance."
        )

    base_curve_dofs = jnp.asarray(coils.dofs_curves)
    base_currents = jnp.asarray(coils.dofs_currents)
    if base_curve_dofs.ndim != 3 or base_curve_dofs.shape[1] != 3 or base_curve_dofs.shape[2] % 2 != 1:
        raise ValueError("ESSOS dofs_curves must have shape (n_base_coils, 3, 2 * order + 1)")
    if base_currents.ndim != 1:
        raise ValueError("ESSOS dofs_currents must have shape (n_base_coils,)")
    if base_currents.shape[0] != base_curve_dofs.shape[0]:
        raise ValueError(
            "ESSOS dofs_currents length must match dofs_curves n_base_coils: "
            f"{base_currents.shape[0]} != {base_curve_dofs.shape[0]}"
        )

    normalized_chunk_size = None if chunk_size is None else int(chunk_size)
    if normalized_chunk_size is not None and normalized_chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {normalized_chunk_size}")

    return CoilFieldParams(
        base_curve_dofs=base_curve_dofs,
        base_currents=base_currents,
        n_segments=int(coils.n_segments),
        nfp=int(coils.nfp),
        stellsym=bool(coils.stellsym),
        current_scale=float(coils.currents_scale),
        regularization_epsilon=float(regularization_epsilon),
        chunk_size=normalized_chunk_size,
    )

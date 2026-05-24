"""Provider dispatch for external free-boundary magnetic fields.

This module deliberately keeps the public internal API function-first:
provider parameters are pytrees and sampling is a pure function.  The legacy
``mgrid`` path remains the VMEC2000-compatibility backend, while ``direct_coils``
is the differentiable single-stage optimization backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vmec_jax._compat import jnp


@dataclass(frozen=True)
class ExternalFieldSample:
    """Cylindrical magnetic-field components sampled on a VMEC boundary grid."""

    br: Any
    bphi: Any
    bz: Any


@dataclass(frozen=True)
class ExternalFieldProviderConfig:
    """Static metadata for an external-field provider.

    Parameters
    ----------
    kind:
        Provider name.  Supported values in phase 1 are ``"direct_coils"`` and
        aliases ``"coils"`` / ``"coil"``.  The ``"mgrid"`` slot is reserved for
        the JAX mgrid interpolation backend.
    static:
        Provider-specific static metadata.  This should not contain
        differentiable arrays.
    """

    kind: str
    static: Any = None


def sample_external_field_cylindrical(
    provider_kind: str,
    provider_static: Any,
    provider_params: Any,
    R: Any,
    Z: Any,
    phi: Any,
) -> tuple[Any, Any, Any]:
    """Sample an external magnetic field in cylindrical components.

    Parameters
    ----------
    provider_kind:
        External-field backend name.
    provider_static:
        Non-differentiable provider metadata.  The direct-coil provider does
        not require this argument in phase 1.  Host-side free-boundary drivers
        may pass a prebuilt ``coil_geometry`` cache here for forward
        benchmarking; callers that need gradients with respect to changing
        coil parameters should leave this unset so geometry is rebuilt from
        ``provider_params`` inside the transformed function.
    provider_params:
        Pytree containing differentiable provider parameters.
    R, Z, phi:
        Cylindrical evaluation coordinates.  Arrays should be broadcastable to
        a common boundary-grid shape, typically ``(ntheta, nzeta)``.

    Returns
    -------
    br, bphi, bz:
        Cylindrical magnetic-field components with the broadcasted input shape.
    """

    kind = str(provider_kind).lower()
    if kind in ("direct_coils", "coils", "coil"):
        if isinstance(provider_static, dict) and "coil_geometry" in provider_static:
            from .coils_jax import sample_coil_field_cylindrical_from_geometry

            return sample_coil_field_cylindrical_from_geometry(
                provider_static["coil_geometry"],
                R,
                Z,
                phi,
                regularization_epsilon=float(
                    provider_static.get(
                        "regularization_epsilon",
                        getattr(provider_params, "regularization_epsilon", 0.0),
                    )
                ),
                chunk_size=provider_static.get("chunk_size", getattr(provider_params, "chunk_size", None)),
            )

        from .coils_jax import sample_coil_field_cylindrical

        return sample_coil_field_cylindrical(provider_params, R, Z, phi)
    if kind == "mgrid":
        try:
            from .mgrid_jax import sample_mgrid_field_cylindrical
        except ImportError as exc:  # pragma: no cover - mgrid_jax lands in WP4.
            raise NotImplementedError("JAX mgrid provider is planned but not implemented yet.") from exc

        return sample_mgrid_field_cylindrical(provider_params, R, Z, phi)
    raise ValueError(f"Unknown external-field provider kind: {provider_kind!r}")


def broadcast_cylindrical_coordinates(R: Any, Z: Any, phi: Any) -> tuple[Any, Any, Any]:
    """Return broadcasted cylindrical coordinates as JAX arrays."""

    return jnp.broadcast_arrays(jnp.asarray(R), jnp.asarray(Z), jnp.asarray(phi))

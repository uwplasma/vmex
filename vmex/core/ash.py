"""Source-weighted trace-helium occupation on supplied transport coefficients.

The radial discretization and checked SOLVAX solves belong to ``essos.ash``.
This module only adapts their physical cost to VMEX's objective interface.
Core residence with a prescribed sink is not a pumped-exhaust prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class HeliumAshObjective:
    """A positive occupation cost, with a fixed reference in seconds.

    ``transport_response(state, runtime)`` returns keyword arguments for
    ``essos.ash.evaluate_helium_residence``: physical radial edges, cell
    volumes, face volume derivatives, helium D/V, source, boundary kappa,
    and optional cell cost. Geometry, source normalization, and Er derivatives
    are included only if the callback computes them from traced inputs.
    Record those choices in the required ``gradient_scope`` description.

    The callback must recompute the physical response for each design; a
    frozen coefficient table does not provide a kinetic shape derivative.
    File readers and host kinetic preparation belong outside a claimed JAX
    gradient. A prescribed-background/Er pilot is legitimate when declared.
    Neither this class nor an implicit VMEX solve qualifies kinetic gradients
    or an ambipolar branch automatically.

    ``total_state`` returns mean source-weighted occupation / reference.
    ``residuals_state`` returns its square root: with target 0 and weight 1,
    VMEX's least-squares cost is half that positive ratio. Zero residence is
    not asserted to be physically attainable. Invalid radial responses yield
    NaN, never a favorable finite penalty. Equilibrium, kinetic convergence,
    bulk transport and independent alpha holdouts are separate hard gates.
    """

    transport_response: Callable[[Any, Any], Mapping[str, Any]]
    reference_occupation_s: float
    gradient_scope: str

    name = "helium occupation"

    def __post_init__(self) -> None:
        if not callable(self.transport_response):
            raise TypeError("transport_response must be callable")
        reference = float(self.reference_occupation_s)
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference_occupation_s must be finite and positive")
        if not isinstance(self.gradient_scope, str) or not self.gradient_scope.strip():
            raise ValueError("gradient_scope must explicitly describe the derivative scope")
        object.__setattr__(self, "reference_occupation_s", reference)

    def evaluate_state(self, state: Any, runtime: Any) -> Any:
        """Return the ESSOS radial result, including conservation diagnostics."""
        try:
            from essos.ash import evaluate_helium_residence
        except ImportError as error:
            raise ImportError(
                "HeliumAshObjective requires ESSOS with essos.ash; install the "
                "helium-residence feature dependency before using this objective"
            ) from error
        return evaluate_helium_residence(**self.transport_response(state, runtime))

    def total_state(self, state: Any, runtime: Any) -> Any:
        """Dimensionless positive cost; invalid inputs remain nonfinite."""
        result = self.evaluate_state(state, runtime)
        occupation = jnp.asarray(result.mean_occupation_s)
        if occupation.shape != ():
            raise ValueError("mean_occupation_s must be scalar")
        valid = result.valid & jnp.isfinite(occupation) & (occupation > 0)
        return jnp.where(valid, occupation / self.reference_occupation_s, jnp.nan)

    def residuals_state(self, state: Any, runtime: Any) -> Any:
        """One least-squares row whose square equals ``total_state``."""
        return jnp.atleast_1d(jnp.sqrt(self.total_state(state, runtime)))

    __call__ = residuals_state

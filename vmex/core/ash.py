"""Adapt ESSOS's source-weighted helium occupation to VMEX least squares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class HeliumAshObjective:
    """Mean source-weighted occupation divided by a fixed positive reference.

    ``transport_response(state, runtime)`` supplies SI keyword arguments to
    ``essos.ash.evaluate_helium_residence``: geometry/volumes, D/V, source,
    boundary sink and optional cell cost. Recompute them for each design.
    ``gradient_scope`` must state traced dependencies and fixed inputs;
    frozen tables and host files do not provide kinetic shape derivatives
    or qualify an ambipolar branch.

    ``residuals_state`` is the square root of ``total_state``: target 0 and
    weight 1 give a least-squares cost of half the normalized occupation.
    Invalid responses yield NaN. Equilibrium/kinetic convergence, matched
    source, bulk transport and independent alpha holdouts require separate
    qualification. A prescribed boundary sink predicts core residence, not
    pumped exhaust; zero residence is not a claimed attainable target.
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

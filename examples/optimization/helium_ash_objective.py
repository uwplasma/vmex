"""Bounded diffusion-only objective example; manufactured slab, not a design study.

Requires ESSOS's optional ``essos.ash`` API. The geometry, source and edge
sink stay fixed; no kinetic shape derivative or pumped exhaust is modeled.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from scipy.optimize import least_squares

from vmex import optimize as opt

jax.config.update("jax_enable_x64", True)


def transport_response(state, _runtime):
    """Uniform births in a unit slab with a reflecting axis and absorbing edge."""
    edges = jnp.linspace(0., 1., 41)
    return dict(radial_edges_m=edges, cell_volume_m3=jnp.diff(edges),
                volume_derivative_m2=jnp.ones_like(edges),
                diffusion_m2_s=jnp.full_like(edges, state[0]),
                convection_m_s=jnp.zeros_like(edges), source_m3_s=jnp.ones(40),
                boundary_kappa_m_s=jnp.inf)


def main():
    """Minimize residence over a declared diffusion interval in SI units."""
    seed = [0.5]
    initial = opt.HeliumAshObjective(transport_response, 1., "diffusion only; fixed geometry/source/sink")
    reference = float(initial.evaluate_state(seed, None).mean_occupation_s)
    objective = opt.HeliumAshObjective(transport_response, reference, initial.gradient_scope)
    residual = jax.jit(lambda x: objective(x, None))
    result = least_squares(residual, seed, bounds=([.2], [2.]))
    cost = float(objective.total_state(result.x, None))
    print(f"D = {result.x[0]:.6g} m^2/s; normalized occupation = {cost:.6g}")
    return result, cost


if __name__ == "__main__":
    main()

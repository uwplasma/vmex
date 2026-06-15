from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    run_mirror_fixed_boundary,
)
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.state import MirrorStateAxisym
from vmec_jax.mirror.kernels.energy import magnetic_energy_axisym
from vmec_jax.mirror.kernels.fields import evaluate_axisym_field
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry

pytestmark = pytest.mark.mirror


def test_cylinder_noop_solver_energy_is_resolution_stable():
    radius = 0.32
    length = 1.1
    b0 = 1.4
    expected_energies = []
    computed_energies = []
    for ns, nxi in [(7, 9), (9, 13), (11, 17)]:
        config = MirrorConfig(MirrorResolution(ns=ns, ntheta=1, nxi=nxi, mpol=0), z_min=-length, z_max=length)
        boundary = MirrorBoundary.constant_radius(radius)
        psi = PsiPrimeProfile.constant(0.5 * radius**2 * b0)
        result = run_mirror_fixed_boundary(
            config,
            boundary,
            psi_prime=psi,
            i_prime=IPrimeProfile.zero(),
            pressure=PressureProfile.zero(),
            options=MirrorSolveOptions(maxiter=0, mu0=1.0),
        )
        state = MirrorStateAxisym.from_boundary(result.grid, boundary)
        geometry = evaluate_axisym_geometry(state, result.grid)
        field = evaluate_axisym_field(state, result.grid, geometry, psi_prime=psi, i_prime=IPrimeProfile.zero())
        computed_energies.append(magnetic_energy_axisym(field, geometry, result.grid, mu0=1.0))
        expected_energies.append(b0**2 * geometry.volume / 2.0)
        assert np.isclose(result.final_trace.energy_total, computed_energies[-1], rtol=2.0e-13)

    assert np.allclose(computed_energies, expected_energies, rtol=2.0e-13, atol=2.0e-13)
    assert np.max(computed_energies) - np.min(computed_energies) < 2.0e-13


def test_short_solve_reduces_residual_more_at_higher_iteration_budget():
    config = MirrorConfig(MirrorResolution(ns=7, ntheta=1, nxi=13, mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    boundary = MirrorBoundary.constant_radius(0.3)
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    initial_state = MirrorStateAxisym(a=base.a * (1.0 + 0.02 * s * (1.0 - s) * (1.0 - xi**2)), lam=base.lam)
    kwargs = dict(
        config=config,
        boundary=boundary,
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        initial_state=initial_state,
    )

    one_step = run_mirror_fixed_boundary(
        **kwargs,
        options=MirrorSolveOptions(maxiter=1, step_size=0.05, tolerance=1.0e-12, mu0=1.0),
    )
    ten_steps = run_mirror_fixed_boundary(
        **kwargs,
        options=MirrorSolveOptions(maxiter=10, step_size=0.05, tolerance=1.0e-12, mu0=1.0),
    )

    assert ten_steps.final_trace.energy_total <= one_step.final_trace.energy_total
    assert ten_steps.final_trace.residual_norm <= one_step.final_trace.residual_norm

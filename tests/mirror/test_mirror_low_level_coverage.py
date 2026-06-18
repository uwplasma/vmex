from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    plot_mirror_output,
    run_mirror_fixed_boundary,
    write_mirror_output,
)
from vmec_jax.mirror.core.basis import ChebyshevLobattoBasis, ThetaFourierBasis
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.grids import make_mirror_grid
from vmec_jax.mirror.core.state import MirrorState3D, MirrorStateAxisym
from vmec_jax.mirror.io.mout import load_mirror_output
from vmec_jax.mirror.kernels.energy import magnetic_energy_axisym, pressure_energy_axisym
from vmec_jax.mirror.kernels.fields import (
    MirrorContravariantFluxes,
    contravariant_fluxes_from_lambda,
    divergence_free_numerator,
)
from vmec_jax.mirror.kernels.fourier import (
    evaluate_real_fourier,
    fourier_second_derivative,
    real_fourier_modes,
    theta_nodes,
    theta_weights,
)
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry
from vmec_jax.mirror.kernels.residuals import field_diagnostics
from vmec_jax.mirror.plotting.export import _plot_name
from vmec_jax.mirror.solvers.fixed_boundary.continuation import pressure_stage_profiles
from vmec_jax.mirror.solvers.fixed_boundary.optimizers import (
    OptimizerOptions,
    _residual_linear_maxiter_policy_key,
    _residual_linear_solver_key,
    _residual_preconditioner_key,
    _sanitize_scale,
    _scaling_key,
    _validate_smoothing_alpha,
    axisym_residual_linear_maxiter,
    pack_axisym_reduced_state,
    reduced_axisym_energy_and_gradient,
    unpack_axisym_reduced_state,
)
from vmec_jax.mirror.validation.coils import (
    circular_loop_field_rz,
    circular_loop_on_axis_bz,
    mirror_boundary_from_on_axis_bz,
    on_axis_mirror_ratio,
    two_coil_field_rz,
    two_coil_on_axis_bz,
)

pytestmark = pytest.mark.mirror


def test_basis_convenience_methods_and_fourier_guards():
    axial = ChebyshevLobattoBasis.from_num_nodes(5)
    values = axial.nodes**2

    assert axial.num_nodes == 5
    assert axial.integrate(np.ones_like(axial.nodes)) == pytest.approx(2.0)
    assert np.allclose(axial.differentiate(values), 2.0 * axial.nodes, atol=1.0e-12)
    assert axial.interpolation_matrix(axial.nodes).shape == (5, 5)
    assert np.allclose(axial.interpolate(values, axial.nodes), values)
    assert axial.filter(values, cutoff=1).shape == values.shape

    theta_basis = ThetaFourierBasis.from_resolution(7, mpol=2)
    cos_coeffs = np.array([1.0, 0.0, 0.5])
    assert theta_basis.ntheta == 7
    assert theta_basis.mpol == 2
    assert theta_basis.evaluate(cos_coeffs).shape == (7,)
    assert theta_basis.evaluate_derivative(cos_coeffs).shape == (7,)
    assert theta_basis.differentiate(np.sin(theta_basis.theta)).shape == (7,)

    second = fourier_second_derivative(np.cos(theta_basis.theta), axis=0)
    assert np.allclose(second, -np.cos(theta_basis.theta), atol=1.0e-12)

    with pytest.raises(ValueError):
        theta_nodes(0)
    with pytest.raises(ValueError):
        theta_weights(0)
    with pytest.raises(ValueError):
        real_fourier_modes(-1)
    with pytest.raises(ValueError):
        evaluate_real_fourier(theta_basis.theta, np.ones(2), sin_coeffs=np.ones(3))


def test_boundary_grid_profile_and_state_guards():
    with pytest.raises(ValueError):
        make_mirror_grid(ns=1)
    with pytest.raises(ValueError):
        make_mirror_grid(ns=3, nxi=1)
    with pytest.raises(ValueError):
        make_mirror_grid(ns=3, z_min=1.0, z_max=1.0)

    grid = make_mirror_grid(ns=4, ntheta=5, nxi=5, mpol=2)
    boundary = MirrorBoundary.tabulated_radius(np.array([-1.0, 0.0, 1.0]), np.array([0.4, 0.3, 0.4]))
    assert np.all(boundary.radius_on_grid(grid) > 0.0)
    assert boundary.radius_on_grid_3d(grid).shape == (grid.ntheta, grid.nxi)

    nonaxis = MirrorBoundary.cosine_modulated_radius(r0=0.4, epsilon=0.1, theta_mode=2)
    assert not nonaxis.is_axisymmetric
    assert nonaxis.radius_on_grid_3d(grid).shape == (grid.ntheta, grid.nxi)
    with pytest.raises(ValueError):
        nonaxis.radius(grid.xi)
    with pytest.raises(ValueError):
        nonaxis.radius_on_grid(grid)
    with pytest.raises(ValueError):
        MirrorBoundary.constant_radius(0.0)
    with pytest.raises(ValueError):
        MirrorBoundary.polynomial_radius(r0=-1.0)
    with pytest.raises(ValueError):
        MirrorBoundary.tabulated_radius([0.0], [1.0])
    with pytest.raises(ValueError):
        MirrorBoundary.tabulated_radius([0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        MirrorBoundary.tabulated_radius([0.0, 1.0], [1.0, -1.0])
    with pytest.raises(ValueError):
        MirrorBoundary.cosine_modulated_radius(r0=0.3, epsilon=0.1, theta_mode=0)
    with pytest.raises(ValueError):
        MirrorBoundary(kind="unknown").radius(grid.xi)

    with pytest.raises(ValueError):
        PsiPrimeProfile.polynomial([])
    with pytest.raises(ValueError):
        IPrimeProfile.polynomial(np.ones((1, 1)))
    with pytest.raises(ValueError):
        PressureProfile.polynomial([])

    axis_state = MirrorStateAxisym.from_boundary(grid, boundary, project=False)
    flat, aux = axis_state.tree_flatten()
    assert MirrorStateAxisym.tree_unflatten(aux, flat).shape == axis_state.shape
    with pytest.raises(ValueError):
        MirrorStateAxisym(a=np.ones(3), lam=np.ones(3))
    with pytest.raises(ValueError):
        MirrorStateAxisym(a=np.ones((2, 2)), lam=np.ones((2, 3)))

    state3d = MirrorState3D.from_boundary(grid, nonaxis, project=False)
    flat3d, aux3d = state3d.tree_flatten()
    assert MirrorState3D.tree_unflatten(aux3d, flat3d).shape == state3d.shape
    with pytest.raises(ValueError):
        MirrorState3D(a=np.ones((2, 2)), lam=np.ones((2, 2)))
    with pytest.raises(ValueError):
        MirrorState3D(a=np.ones((2, 2, 2)), lam=np.ones((2, 2, 3)))


def test_low_level_field_energy_residual_and_optimizer_guards():
    grid = make_mirror_grid(ns=5, ntheta=1, nxi=7)
    boundary = MirrorBoundary.constant_radius(0.3)
    state = MirrorStateAxisym.from_boundary(grid, boundary)
    geometry = evaluate_axisym_geometry(state, grid)
    field = type("Field", (), {"b2": np.ones_like(geometry.sqrtg), "bmag": np.array([0.0, 1.0])})()

    with pytest.raises(ValueError):
        magnetic_energy_axisym(field, geometry, grid, mu0=0.0)
    with pytest.raises(ValueError):
        pressure_energy_axisym(PressureProfile.constant(1.0, gamma=1.0), geometry, grid)

    diagnostics = field_diagnostics(field, grid)
    assert diagnostics.mirror_ratio == float("inf")
    fluxes = MirrorContravariantFluxes(jb_theta=np.zeros_like(state.a), jb_xi=np.ones_like(state.a))
    assert field_diagnostics(field, grid, fluxes=fluxes).max_divergence_numerator >= 0.0
    with pytest.raises(ValueError):
        divergence_free_numerator(MirrorContravariantFluxes(np.zeros((2, 2)), np.zeros((2, 3))), grid)
    with pytest.raises(ValueError):
        contravariant_fluxes_from_lambda(np.zeros((2, 2, 2, 2)), grid, psi_prime=PsiPrimeProfile.constant(0.1))
    with pytest.raises(ValueError):
        contravariant_fluxes_from_lambda(
            np.zeros((grid.ns + 1, grid.nxi)), grid, psi_prime=PsiPrimeProfile.constant(0.1)
        )

    with pytest.raises(ValueError):
        _scaling_key("bad")
    with pytest.raises(ValueError):
        _residual_preconditioner_key("bad")
    with pytest.raises(ValueError):
        _residual_linear_maxiter_policy_key("bad")
    with pytest.raises(ValueError):
        _residual_linear_solver_key("bad")
    assert _residual_linear_solver_key("lsqr") == "lsqr"
    assert _residual_linear_solver_key("block_dense_lstsq") == "block_dense_lstsq"
    assert _residual_linear_solver_key("block_lsmr") == "block_lsmr"
    assert _residual_linear_solver_key("split_lsmr") == "block_lsmr"
    with pytest.raises(ValueError):
        _sanitize_scale([1.0, 2.0], expected_size=3)
    with pytest.raises(ValueError):
        _validate_smoothing_alpha(-1.0, name="alpha")
    with pytest.raises(ValueError):
        axisym_residual_linear_maxiter(
            OptimizerOptions(residual_linear_adaptive_factor=0.0),
            grid,
            vector_size=10,
            residual_norm=1.0,
        )
    assert (
        axisym_residual_linear_maxiter(
            OptimizerOptions(residual_linear_maxiter=7, residual_linear_maxiter_policy="adaptive"),
            grid,
            vector_size=10,
            residual_norm=float("nan"),
        )
        == 10
    )
    assert (
        axisym_residual_linear_maxiter(
            OptimizerOptions(residual_linear_maxiter=7, residual_linear_maxiter_policy="adaptive", tolerance=1.0),
            grid,
            vector_size=10,
            residual_norm=0.5,
        )
        == 7
    )


def test_reduced_state_sources_and_continuation_guards():
    grid = make_mirror_grid(ns=5, ntheta=1, nxi=7)
    boundary = MirrorBoundary.constant_radius(0.3)
    state = MirrorStateAxisym.from_boundary(grid, boundary)
    vector = pack_axisym_reduced_state(state, grid, boundary)

    with pytest.raises(ValueError):
        unpack_axisym_reduced_state(vector[:-1], grid, boundary)
    with pytest.raises(ValueError):
        reduced_axisym_energy_and_gradient(
            vector,
            grid,
            boundary,
            psi_prime=PsiPrimeProfile.constant(0.01),
            i_prime=IPrimeProfile.zero(),
            pressure=PressureProfile.zero(),
            source_a=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError):
        reduced_axisym_energy_and_gradient(
            vector,
            grid,
            boundary,
            psi_prime=PsiPrimeProfile.constant(0.01),
            i_prime=IPrimeProfile.zero(),
            pressure=PressureProfile.zero(),
            source_lam=np.zeros((2, 2)),
        )

    stages = pressure_stage_profiles(PressureProfile.constant(1.0), None)
    assert len(stages) == 1
    with pytest.raises(ValueError):
        pressure_stage_profiles(PressureProfile.zero(), ())


def test_coil_validation_guards_and_boundary_builder():
    z = np.linspace(-1.0, 1.0, 7)
    bz = two_coil_on_axis_bz(z, coil_radius_m=0.5, separation_m=1.2, current_a=10.0)
    boundary = mirror_boundary_from_on_axis_bz(0.01, z, bz)
    assert boundary.is_axisymmetric
    assert two_coil_field_rz(
        np.array([0.0, 0.1]), np.array([0.0, 0.2]), coil_radius_m=0.5, separation_m=1.2, current_a=10.0
    ).br.shape == (2,)
    assert circular_loop_field_rz(
        np.array([0.0, 0.1]), np.array([0.0, 0.2]), loop_radius_m=0.5, current_a=10.0
    ).bz.shape == (2,)
    assert on_axis_mirror_ratio(np.array([1.0, 2.0])) == pytest.approx(2.0)

    with pytest.raises(ValueError):
        circular_loop_on_axis_bz(z, loop_radius_m=0.0, current_a=1.0)
    with pytest.raises(ValueError):
        circular_loop_field_rz(0.1, 0.0, loop_radius_m=-1.0, current_a=1.0)
    with pytest.raises(ValueError):
        two_coil_on_axis_bz(z, coil_radius_m=0.5, separation_m=0.0, current_a=1.0)
    with pytest.raises(ValueError):
        two_coil_field_rz(0.1, 0.0, coil_radius_m=0.5, separation_m=0.0, current_a=1.0)
    with pytest.raises(ValueError):
        on_axis_mirror_ratio(np.array([1.0]))
    with pytest.raises(ValueError):
        on_axis_mirror_ratio(np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        mirror_boundary_from_on_axis_bz(0.01, np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        mirror_boundary_from_on_axis_bz(0.01, z, np.ones(z.size + 1))
    with pytest.raises(ValueError):
        mirror_boundary_from_on_axis_bz(0.01, z[::-1], np.ones_like(z))
    with pytest.raises(ValueError):
        mirror_boundary_from_on_axis_bz(0.0, z, np.ones_like(z))


def test_plot_name_for_in_memory_output_and_show(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    config = MirrorConfig(MirrorResolution(ns=5, ntheta=1, nxi=7, mpol=0), z_min=-1.0, z_max=1.0)
    result = run_mirror_fixed_boundary(
        config,
        MirrorBoundary.constant_radius(0.3),
        psi_prime=PsiPrimeProfile.constant(0.01),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(maxiter=1, step_size=1.0e-4, tolerance=1.0e-12, mu0=1.0),
    )
    output = load_mirror_output(write_mirror_output(tmp_path / "mout_case.nc", result))
    output_without_path = replace(output, path=None)
    assert _plot_name(output_without_path, None) == "mirror"
    assert _plot_name(output, None) == "case"
    assert _plot_name(output, "named") == "named"

    calls = []
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))
    paths = plot_mirror_output(output_without_path, outdir=tmp_path / "figures", show=True)
    assert calls == ["show"]
    assert paths["surfaces_rz"].exists()

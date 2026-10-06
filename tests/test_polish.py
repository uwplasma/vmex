"""Force-balance polish driver: scope, configuration, and WOUT export."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from vmex.core import implicit, solver
from vmex.core.errors import StrongForceCertificationError, VmecInputError
from vmex.core.input import VmecInput
from vmex.core import polish
from vmex.core.polish import (
    PolishConfig,
    polish_legacy_solution,
    polished_wout_input,
    polished_wout_ns,
    polished_wout_state,
    sample_high_order_state,
)
from vmex.core.strong_force import append_high_order_state_modes, certify_strong_force, lift_high_order_state

pytestmark = pytest.mark.usefixtures("_module_jit_enabled")

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _small_solovev_input():
    inp = VmecInput.from_file(DATA / "input.solovev").change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    return dataclasses.replace(
        inp, ns_array=np.asarray([5]), ftol_array=np.asarray([1.0e-10]), niter_array=np.asarray([1000])
    )


@pytest.fixture(scope="module")
def small_lift():
    inp = _small_solovev_input()
    config = implicit.make_config(inp, ftol=1.0e-10, max_iterations=1000)
    params = implicit.params_from_input(inp)
    state, _ = implicit.solve_implicit_with_aux(params, config)
    runtime = implicit.runtime_from_params(params, config)
    return lift_high_order_state(state, runtime, degree=3), state


def test_sample_high_order_state_inverts_the_lift_on_any_mesh(small_lift):
    native, state = small_lift
    inp = _small_solovev_input()
    for ns in (5, 11):
        runtime = solver.prepare_runtime(inp, solver.resolution_from_input(inp, ns=ns))
        sampled = sample_high_order_state(native, runtime)
        assert np.shape(np.asarray(sampled.R_cos)) == (ns, native.m.size)
        relift = lift_high_order_state(sampled, runtime, radial_basis=native.radial_basis, degree=3)
        for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
            np.testing.assert_allclose(np.asarray(getattr(relift, name)), np.asarray(getattr(native, name)),
                                       rtol=0.0, atol=1.0e-11)
    # The lift pins the edge coefficients, so the solve-mesh sample keeps the boundary.
    runtime = solver.prepare_runtime(inp, solver.resolution_from_input(inp, ns=5))
    sampled = sample_high_order_state(native, runtime)
    np.testing.assert_allclose(np.asarray(sampled.R_cos[-1]), np.asarray(state.R_cos[-1]), rtol=0.0, atol=1.0e-13)


def test_sample_high_order_state_requires_matching_mode_tables(small_lift):
    native, _ = small_lift
    other = VmecInput.from_file(DATA / "input.solovev").change_resolution(mpol=4, ntor=0, ntheta=12, nzeta=4)
    runtime = solver.prepare_runtime(other, solver.resolution_from_input(other, ns=5))
    with pytest.raises(ValueError, match="mode tables"):
        sample_high_order_state(native, runtime)


def test_polished_wout_mesh_and_modes_carry_the_native_state():
    def native(size, breakpoints=None, m_max=3):
        spans = size - 3 if breakpoints is None else len(breakpoints) - 1
        breaks = np.linspace(0.0, 1.0, spans + 1) if breakpoints is None else np.asarray(breakpoints)
        return SimpleNamespace(radial_basis=SimpleNamespace(size=size, breakpoints=breaks), m=np.arange(m_max + 1))

    # Four samples per default reconstruction span, never coarser than the solve.
    assert polished_wout_ns(native(17), solve_ns=31) == 129
    assert polished_wout_ns(native(17), solve_ns=201) == 201
    # Two samples per coefficient, and two uniform-s samples in the narrowest span.
    assert polished_wout_ns(native(90), solve_ns=31) == 181
    assert polished_wout_ns(native(20, [0.0, 0.01, 0.02, 0.5, 1.0]), solve_ns=31) == 201
    # MPOL grows only when padded modes exceed it.
    deck = _small_solovev_input()
    assert polished_wout_input(native(17, m_max=2), deck) is deck
    assert int(polished_wout_input(native(17, m_max=8), deck).mpol) == 9


def test_polish_scope_and_configuration():
    deck = dataclasses.replace(_small_solovev_input(), ncurr=1)
    # Outside the scope: AUTO leaves the solve unpolished, an explicit request refuses.
    assert polish_legacy_solution(deck, None, None, auto=True) is None
    with pytest.raises(VmecInputError, match="axisymmetric"):
        polish_legacy_solution(deck, None, None)
    for updates, message in (({"fail_policy": "maybe"}, "fail_policy"), ({"degree": 4}, "degree"),
                             ({"force_tolerance": 0.0}, "positive")):
        with pytest.raises(ValueError, match=message):
            PolishConfig(**updates)


def test_polish_driver_reports_fails_and_exports_with_a_stub_solver(monkeypatch, small_lift):
    """The driver, the solver hook, and the export around a stubbed polish (the
    real polish runs in the full-polish-gn lane)."""

    native, state = small_lift
    inp = _small_solovev_input()
    window = SimpleNamespace(volume_average_force=2.0, magnetic_relative_force_error=1.0e-3, s_min=0.0, s_max=1.0)
    certificate = SimpleNamespace(absolute_l2=1.0, normalized_l2=0.5, minimum_signed_jacobian=1.0,
                                  window_normalizations=window)
    stub = polish.NativePolishResult(native, 1.0, 1.0e-3, False, 1, 2, 1, 0.1)
    monkeypatch.setattr(polish, "certify_strong_force", lambda _state: certificate)
    monkeypatch.setattr(polish, "force_error_measures", lambda *_: ())
    monkeypatch.setattr(polish, "physical_scales", lambda _state: (1.0, 1.0))
    monkeypatch.setattr(polish, "polish_native", lambda *_, **__: stub)
    resolution = solver.resolution_from_input(inp, ns=5)
    lines = []
    fallback = polish_legacy_solution(inp, resolution, state, verbose=True,
                                      config=PolishConfig(degree=3, fail_policy="return_unpolished"),
                                      emit=lambda *parts, **_: lines.append(" ".join(map(str, parts))))
    report = fallback.polish_report
    assert not report.converged and report.nonlinear_iterations == 3 and report.normalization_window == (0.0, 1.0)
    assert "native polish: |F|_rms" in "".join(lines) and "FAILED" in "".join(lines)
    with pytest.raises(StrongForceCertificationError, match="did not converge"):
        polish_legacy_solution(inp, resolution, state, config=PolishConfig(degree=3))
    exported = polished_wout_state(fallback.native_equilibrium, inp, solve_ns=5)
    assert np.shape(np.asarray(exported.R_cos))[0] == polished_wout_ns(fallback.native_equilibrium, solve_ns=5)
    # The solver hook: an out-of-scope AUTO keeps the result, a polish replaces it.
    base = SimpleNamespace(state=state)
    monkeypatch.setattr(solver, "replace", lambda result, **fields: {**vars(result), **fields})
    for returned, check in ((None, lambda out: out is base), (fallback, lambda out: out["polish_report"] is report)):
        monkeypatch.setattr(polish, "polish_legacy_solution", lambda *_, _r=returned, **__: _r)
        assert check(solver._polish_solve_result(inp, resolution, base, polish="auto", polish_config=None,
                                                 lconm1=True, verbose=True, emit=lambda *_, **__: None))


def test_polish_building_blocks_reject_malformed_requests(small_lift):
    native, _ = small_lift
    with pytest.raises(ValueError, match="radial_order"):
        polish.make_variational_plan(native, radial_order=1)
    with pytest.raises(ValueError, match="angular grid"):
        polish.make_variational_plan(native, ntheta=0)
    with pytest.raises(ValueError, match="force coordinates"):
        polish._chart(native, 1.0, 1.0).force(np.zeros(3))
    with pytest.raises(ValueError, match="nonzero length"):
        append_high_order_state_modes(native, [], [])
    with pytest.raises(ValueError, match="angular shifts"):
        certify_strong_force(native, theta_shift=1.0)


def test_public_solver_rejects_unknown_polish_mode_before_solving():
    inp = VmecInput.from_file(DATA / "input.solovev")
    with pytest.raises(ValueError, match="False, True, or 'auto'"):
        solver.solve(inp, polish="unknown")


def test_public_solver_resolves_polish_keywords_only():
    """Directives live in run_options; the solver sees only its keywords."""
    inp = VmecInput.from_file(DATA / "input.solovev")
    assert solver._resolve_force_balance_polish(inp, None, None) is False
    assert solver._resolve_force_balance_polish(inp, True, None) is True
    assert solver._resolve_force_balance_polish(inp, "auto", None) == "auto"
    with pytest.raises(ValueError, match="either polish or polish_force_balance"):
        solver._resolve_force_balance_polish(inp, False, True)


def _current_torus(orientation=-1):
    """Manufactured nonaxisymmetric geometry, independent of an equilibrium solve."""
    from vmex.core.radial_basis import BSplineBasis
    from vmex.core.strong_force import HighOrderEquilibriumState

    basis = BSplineBasis.clamped([0.0, 0.5, 1.0], degree=3)
    R = np.zeros((4, basis.size))
    Z = np.zeros_like(R)
    L = np.zeros_like(R)
    R[0], R[1], R[2], R[3] = 10.0, 1.0, 0.03, 0.02
    Z[1], Z[2], Z[3] = -orientation, 0.02, -0.01
    L[3] = 0.005
    greville = np.array([np.mean(basis.knots[i+1:i+4]) for i in range(basis.size)])
    pressure = np.linalg.solve(np.asarray(basis.basis_matrix(greville)), 5000 * (1 - greville))
    state = HighOrderEquilibriumState(
        radial_basis=basis, m=np.array([0, 1, 0, 1]), n=np.array([0, 0, 1, 1]), nfp=2,
        R_cos=jnp.asarray(R), R_sin=jnp.zeros_like(R), Z_cos=jnp.zeros_like(Z), Z_sin=jnp.asarray(Z),
        L_cos=jnp.zeros_like(L), L_sin=jnp.asarray(L), phipf=jnp.full(basis.size, 0.5),
        chipf=jnp.zeros(basis.size), pressure=jnp.asarray(pressure), jacobian_sign=orientation,
        boundary_R_cos=jnp.asarray(R[:, -1]), boundary_R_sin=jnp.zeros(4),
        boundary_Z_cos=jnp.zeros(4), boundary_Z_sin=jnp.asarray(Z[:, -1]))
    inp = dataclasses.replace(_small_solovev_input().change_resolution(mpol=3, ntor=1), ncurr=1, nfp=2,
                              curtor=-2e4, ac=np.array([1.0, -0.2]))
    return state, inp


@pytest.mark.parametrize("orientation", [-1, 1])
def test_functional_current_ampere_radial_derivative_and_independent_curl(orientation):
    """The angular circulation, radial finite difference and AD curl are separate oracles."""
    from vmex.core.profiles import MU0, current
    from vmex.core.strong_force import evaluate_strong_force

    state, inp = _current_torus(orientation)
    plan = polish.make_variational_plan(state, radial_order=3, ntheta=12, nzeta=10)
    chi, derivative = polish.prescribed_current_flux_jets(state, plan, inp)
    force = polish.evaluate_prescribed_current_force(state, plan, inp)
    fields = polish.evaluate_variational_fields(state, plan)
    enclosed = orientation * 2 * np.pi / MU0 * jnp.mean(
        jnp.sum(force.B * fields.dposition_dtheta, axis=-1), axis=(1, 2))
    expected = inp.curtor * current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, plan.rho**2)
    expected /= current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, 1.0)
    np.testing.assert_allclose(enclosed, expected, rtol=2e-13, atol=1e-8)
    zero = polish.evaluate_prescribed_current_force(
        state, plan, dataclasses.replace(inp, curtor=0.0, ac=np.zeros(2)))
    np.testing.assert_allclose(jnp.mean(jnp.sum(zero.B * fields.dposition_dtheta, axis=-1),
                                       axis=(1, 2)), 0.0, atol=1e-14)
    # A single moving radius retabulates radial/geometry jets independently.
    def at_rho(rho):
        m = np.abs(state.m)
        return dataclasses.replace(plan, rho=jnp.asarray([rho]),
            profile_basis=state.radial_basis.basis_matrix(jnp.array([rho**2])),
            profile_derivative=2 * rho * state.radial_basis.basis_matrix(jnp.array([rho**2]), derivative=1),
            **polish._stable_radial_tables(state.radial_basis, np.array([rho**2]), np.array([rho]), m))
    rho, h = float(plan.rho[2]), 1e-5
    difference = (polish.prescribed_current_flux_jets(state, at_rho(rho+h), inp)[0]
                  - polish.prescribed_current_flux_jets(state, at_rho(rho-h), inp)[0]) / (2*h)
    np.testing.assert_allclose(difference, derivative[2:3], rtol=1e-7, atol=1e-9)
    # Local Hermite interpolation carries exactly chi and chi_rho at each
    # oracle radius; no global chipf fit enters the functional-current result.
    for i in (0, 3, 5):
        matrix = np.stack((np.asarray(plan.profile_basis[i]), np.asarray(plan.profile_derivative[i])))
        fitted = np.linalg.lstsq(matrix, np.array([chi[i], derivative[i]]), rcond=None)[0]
        oracle = evaluate_strong_force(dataclasses.replace(state, chipf=jnp.asarray(fitted)),
                                      plan.rho[i], plan.theta[2], plan.zeta[3])
        for name in ("B", "J", "force"):
            np.testing.assert_allclose(getattr(force, name)[i, 2, 3], getattr(oracle, name),
                                       rtol=2e-9, atol=1e-6)


def test_current_mode_blocks_and_geometry_derivatives():
    state, inp = _current_torus()
    plan = polish.make_variational_plan(state, radial_order=3, ntheta=8, nzeta=6)
    current_plan = polish.make_variational_plan(state, radial_order=3, ntheta=12, nzeta=10)
    layout = polish.make_native_correction_layout(state)
    scale = 1e-3 * polish.native_coordinate_scales(state, layout, plan)
    jets = jax.jit(lambda geometry: polish.prescribed_current_flux_jets(geometry, current_plan, inp))
    def residual(x):
        geometry = polish.apply_high_order_correction(state, layout.unpack(scale * x))
        samples = polish._current_force_samples(geometry, plan, jets(geometry))
        return polish._weighted_force(samples, plan, state.jacobian_sign, 1e6, 200.0).reshape(-1)
    zero = jnp.zeros(layout.size)
    r, J = jax.linearize(residual, zero)
    JT = jax.linear_transpose(J, zero)
    rng = np.random.default_rng(42)
    direction = jnp.asarray(rng.normal(size=layout.size))
    direction /= jnp.linalg.norm(direction)
    fd = (residual(1e-4 * direction) - residual(-1e-4 * direction)) / 2e-4
    np.testing.assert_allclose(fd, J(direction), rtol=2e-5, atol=1e-9)
    u = jnp.asarray(rng.normal(size=r.shape))
    np.testing.assert_allclose(jnp.vdot(u, J(direction)), jnp.vdot(JT(u)[0], direction), rtol=1e-11)
    indices, blocks = polish._current_mode_blocks(state, plan, layout, scale, jets, 1e6, 200.0)
    for rows, block in zip(indices, blocks):
        columns = rows[rows < layout.size]
        vector = rng.normal(size=columns.size)
        tangent = jnp.zeros(layout.size).at[columns].set(vector)
        expected = np.asarray(JT(J(tangent))[0])[columns]
        np.testing.assert_allclose(block[:columns.size, :columns.size] @ vector,
                                   expected, rtol=2e-10, atol=1e-12)


def test_current_polish_reduces_force_preserves_constraints_and_retains_scope():
    state, inp = _current_torus()
    checkpoints = []
    result = polish.polish_prescribed_current(state, inp, max_iterations=3, radial_order=3,
        ntheta=8, nzeta=6, current_ntheta=12, current_nzeta=10, callback=checkpoints.append)
    assert 0 < result.iterations <= 3 and result.final_cost < result.initial_cost
    assert len(checkpoints) == result.iterations
    for name in ("pressure", "phipf", "chipf"):
        np.testing.assert_array_equal(getattr(result.geometry, name), getattr(state, name))
    for name in ("R_cos", "R_sin", "Z_cos", "Z_sin"):
        np.testing.assert_array_equal(getattr(result.geometry, name)[:, -1], getattr(state, name)[:, -1])
    heldout = polish.make_variational_plan(result.geometry, radial_order=4, ntheta=16, nzeta=12)
    samples = result.samples(heldout)
    assert np.isfinite(samples.force).all() and np.min(state.jacobian_sign * samples.sqrt_g) > 0
    assert not polish.native_polish_supported(inp)
    with pytest.raises(VmecInputError, match="axisymmetric"):
        polish.polish_legacy_solution(inp, None, None)
    for updates in ({"ncurr": 0}, {"lasym": True}, {"lfreeb": True, "mgrid_file": "mgrid.nc"}, {"gamma": 1.0}):
        with pytest.raises(VmecInputError, match="NCURR=1"):
            polish.polish_prescribed_current(state, dataclasses.replace(inp, **updates))
    for options in ({"max_iterations": 0}, {"damping": 0}, {"damping": np.nan}):
        with pytest.raises(ValueError, match="positive"):
            polish.polish_prescribed_current(state, inp, **options)
    with pytest.raises(ValueError, match="current profile"):
        polish.polish_prescribed_current(state, dataclasses.replace(inp, ac=np.zeros(2)))
    inverted = dataclasses.replace(state, jacobian_sign=1)
    with pytest.raises(ValueError, match="Jacobian"):
        polish.polish_prescribed_current(inverted, inp)

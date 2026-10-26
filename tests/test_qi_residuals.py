"""Analytic and differentiation checks for quasi-isodynamic residuals."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

import jax
import jax.numpy as jnp

from vmex.core import qi
from vmex.core import implicit as im
from vmex.core.input import VmecInput
def _boozer(perturbation=0.0):
    return {
        "bmnc_b": jnp.array([[1.0, 0.2, perturbation]]),
        "xm_b": jnp.array([0.0, 0.0, 1.0]),
        "xn_b": jnp.array([0.0, 2.0, 0.0]),
        "iota_b": jnp.array([0.4]),
        "G_b": jnp.array([2.0]),
        "I_b": jnp.array([0.0]),
        "nfp": 2,
        "s_b": jnp.array([0.5]),
    }


def _action_residual(perturbation):
    booz = _boozer(perturbation)
    return qi.j_invariant_qi_residual_from_boozer(
        bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
        iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
        nfp=booz["nfp"], pitch=[1.0], nalpha=9, num_periods=4,
        points_per_period=64, max_wells=6)


def test_action_residual_zero_and_non_qi_ordering():
    exact = _action_residual(0.0)
    perturbed = _action_residual(0.06)
    assert exact["valid_pitch"][0, 0]
    assert float(exact["total"]) < 1.0e-24
    assert float(perturbed["total"]) > 1.0e-5
    asymmetric = qi.j_invariant_qi_residual_from_boozer(
        bmnc_b=_boozer()["bmnc_b"], bmns_b=[[0.0, 0.0, 0.04]],
        xm_b=_boozer()["xm_b"], xn_b=_boozer()["xn_b"],
        iota_b=_boozer()["iota_b"], G_b=_boozer()["G_b"],
        I_b=_boozer()["I_b"], nfp=2, pitch=[1.0], nalpha=9,
        points_per_period=64, num_periods=4, max_wells=6)
    assert np.isfinite(float(asymmetric["total"]))

    constant = qi.j_invariant_qi_residual_from_boozer(
        bmnc_b=[[1.0]], xm_b=[0.0], xn_b=[0.0], iota_b=[0.4],
        G_b=[2.0], I_b=[0.0], nfp=2, pitch=[1.0])
    assert not constant["valid_pitch"][0, 0]
    assert np.isnan(float(constant["total"]))

    booz = _boozer(0.06)
    totals = [
        float(qi.j_invariant_qi_residual_from_boozer(
            bmnc_b=booz["bmnc_b"], xm_b=booz["xm_b"], xn_b=booz["xn_b"],
            iota_b=booz["iota_b"], G_b=booz["G_b"], I_b=booz["I_b"],
            nfp=booz["nfp"], pitch=[1.0], nalpha=9, num_periods=4,
            points_per_period=points, max_wells=6)["total"])
        for points in (32, 64, 128)
    ]
    assert abs(totals[1] - totals[2]) < abs(totals[0] - totals[2])


def test_action_residual_pools_distinct_alpha_independent_well_families():
    # B(phi) = 2 + 0.4 cos(2 phi) + 0.15 sin(phi) has two distinct trapped
    # wells per period and no alpha dependence (all Fourier modes have m=0).
    out = qi.j_invariant_qi_residual_from_boozer(
        bmnc_b=[[2.0, 0.4, 0.0]], bmns_b=[[0.0, 0.0, 0.15]],
        xm_b=[0.0, 0.0, 0.0], xn_b=[0.0, -2.0, -1.0],
        iota_b=[0.4], G_b=[2.0], I_b=[0.0], nfp=1,
        pitch=[0.5], nalpha=3, num_periods=1,
        points_per_period=128, max_wells=2, quadrature_order=32)

    action = np.asarray(out["action"])[0, :, 0, :]
    assert np.asarray(out["valid_pitch"])[0, 0]
    assert np.asarray(out["usable_mask"])[0, :, 0, :].all()
    np.testing.assert_allclose(action - action[0], 0.0, atol=1.0e-12)
    assert abs(action[0, 0] - action[0, 1]) > 0.5
    expected_pooled_variance = ((action[0, 0] - action[0, 1]) / action[0].sum()) ** 2
    assert float(out["total"]) == pytest.approx(expected_pooled_variance)
    assert float(out["total"]) > 0.1


def _two_family_options(amplitude=0.0):
    return dict(
        bmnc_b=jnp.array([[2.0, 0.4, 0.0, amplitude]]),
        bmns_b=[[0.0, 0.0, 0.15, 0.0]],
        xm_b=[0.0, 0.0, 0.0, 1.0], xn_b=[0.0, -2.0, -1.0, 0.0],
        iota_b=[0.0], G_b=[2.0], I_b=[0.0], nfp=1,
        pitch=[0.5], num_periods=1, points_per_period=1024,
        max_wells=2, quadrature_order=128)


def test_local_alpha_derivative_vanishes_for_distinct_invariant_families():
    out = qi.j_invariant_alpha_derivative_from_boozer(
        alpha=[0.2, 1.1], **_two_family_options())
    assert np.asarray(out["usable_mask"]).all()
    action = np.asarray(out["action"])[0, 0, 0]
    assert abs(action[0] - action[1]) > 0.5
    np.testing.assert_array_equal(out["action_alpha_derivative"], 0.0)
    for alpha in ([], [[0.0, 0.1]]):
        with pytest.raises(ValueError, match="alpha"):
            qi.j_invariant_alpha_derivative_from_boozer(
                alpha=alpha, **_two_family_options())


def test_local_alpha_derivative_matches_moving_root_oracle_and_parameter_ad():
    # Independent continuous-field quadrature moves both roots with alpha.
    # It includes the Boozer line element 2/B as well as the speed factor.
    def reference_actions(alpha):
        def field(phi):
            return 2.0 + 0.4 * np.cos(2 * phi) + 0.15 * np.sin(phi) + 0.06 * np.cos(alpha)

        roots = [brentq(lambda phi: field(phi) - 2.0, lo, hi)
                 for lo, hi in zip(np.arange(4) * np.pi / 2,
                                   np.arange(1, 5) * np.pi / 2)]
        values = []
        for left, right in (roots[:2], roots[2:]):
            midpoint, width = (left + right) / 2, (right - left) / 2

            def integrand(angle):
                b = field(midpoint + width * np.sin(angle))
                return 4 * np.sqrt(max(1 - 0.5 * b, 0.0)) / b * width * np.cos(angle)

            values.append(quad(integrand, -np.pi / 2, np.pi / 2,
                               epsabs=1e-12, epsrel=1e-12)[0])
        return np.array(values)

    alpha = 0.7
    step = 1.0e-5
    reference = (reference_actions(alpha + step) - reference_actions(alpha - step)) / (2 * step)
    for reference_step in (1.0e-4, 1.0e-6):
        estimate = (reference_actions(alpha + reference_step)
                    - reference_actions(alpha - reference_step)) / (2 * reference_step)
        np.testing.assert_allclose(estimate, reference, rtol=1e-7)

    def evaluate(amplitude):
        return qi.j_invariant_alpha_derivative_from_boozer(
            alpha=alpha, **_two_family_options(amplitude))

    out = jax.jit(evaluate)(0.06)
    np.testing.assert_allclose(out["action"][0, 0, 0], reference_actions(alpha), rtol=3e-5)
    np.testing.assert_allclose(out["action_alpha_derivative"][0, 0, 0], reference, rtol=5e-4)
    assert np.all(np.abs(reference) > 0.05)

    def objective(amplitude):
        return jnp.sum(evaluate(amplitude)["action_alpha_derivative"] ** 2)

    derivative = jax.grad(objective)(0.06)
    # Larger steps can cross the piecewise-linear interpolation cells.
    for parameter_step in (1.0e-5, 1.0e-6):
        finite_difference = (objective(0.06 + parameter_step)
                             - objective(0.06 - parameter_step)) / (2 * parameter_step)
        np.testing.assert_allclose(derivative, finite_difference, rtol=5e-4)


@pytest.mark.parametrize("status,pitch,max_wells", [
    ("absent", 0.3, 2), ("marginal", 1 / 1.6, 2),
    ("merged", 1 / 2.4, 2), ("overflow", 0.5, 1),
])
def test_local_alpha_derivative_masks_invalid_topology(status, pitch, max_wells):
    options = _two_family_options()
    options.update(pitch=[pitch], max_wells=max_wells,
                   bmns_b=[[0.0, 0.0, 0.0, 0.0]])
    out = qi.j_invariant_alpha_derivative_from_boozer(alpha=0.0, **options)
    assert np.asarray(out[status]).all()
    assert np.isnan(np.asarray(out["action_alpha_derivative"])).all()


def test_local_alpha_derivative_retains_complete_wells_in_a_cut_trace():
    out = qi.j_invariant_alpha_derivative_from_boozer(
        alpha=[0.0], bmnc_b=[[1.0, -0.2]], xm_b=[0.0, 0.0],
        xn_b=[0.0, 1.0], iota_b=[0.0], G_b=[1.0], I_b=[0.0],
        nfp=1, pitch=[1.0], num_periods=2, points_per_period=32, max_wells=3)
    assert np.asarray(out["truncated"]).all()
    usable = np.asarray(out["usable_mask"])
    assert usable.sum() == 1
    derivative = np.asarray(out["action_alpha_derivative"])
    np.testing.assert_array_equal(derivative[usable], 0.0)
    assert np.isnan(derivative[~usable]).all()


def test_action_residual_ad_matches_finite_difference():
    def objective(amplitude):
        return _action_residual(amplitude)["total"]

    amplitude = jnp.asarray(0.06, dtype=jnp.float64)
    compiled = jax.jit(objective)(amplitude)
    derivative = jax.grad(objective)(amplitude)
    step = 1.0e-5
    finite_difference = (
        objective(amplitude + step) - objective(amplitude - step)) / (2.0 * step)
    assert compiled == pytest.approx(objective(amplitude))
    np.testing.assert_allclose(derivative, finite_difference, rtol=2.0e-4)


def test_objective_terms_share_the_composable_interface(monkeypatch):
    booz = _boozer(0.04)
    monkeypatch.setattr(qi, "boozer_spectrum_state", lambda *args, **kwargs: booz)
    eq = SimpleNamespace(state=object(), runtime=object())

    action = qi.JInvariantQIResidual(
        [0.5], [1.0], nalpha=7, points_per_period=64,
        num_periods=4, max_wells=6)
    action_rows = action(eq)
    assert np.all(np.isfinite(np.asarray(action_rows)))
    assert float(action.total(eq)) == pytest.approx(float(jnp.sum(action_rows**2)))

    constructed = qi.ConstructedQIResidual(
        [0.5], nphi=41, nalpha=7, n_bounce=7)
    constructed_rows = constructed.residuals(eq)
    assert np.all(np.isfinite(np.asarray(constructed_rows)))
    assert float(constructed.total(eq)) == pytest.approx(
        float(jnp.sum(constructed_rows**2)))


def test_objective_input_validation():
    with pytest.raises(ValueError, match="positive finite"):
        qi.JInvariantQIResidual([0.5], [0.0])
    with pytest.raises(ValueError, match="weights"):
        qi.JInvariantQIResidual([0.3, 0.7], [1.0], weights=[1.0])
    with pytest.raises(ValueError, match="weights"):
        qi.ConstructedQIResidual([0.3, 0.7], weights=[1.0])
    with pytest.raises(ValueError, match="nalpha"):
        qi.j_invariant_qi_residual_from_boozer(
            bmnc_b=[[1.0]], xm_b=[0.0], xn_b=[0.0], iota_b=[0.4],
            G_b=[2.0], I_b=[0.0], nfp=2, pitch=[1.0], nalpha=1)
    with pytest.raises(ValueError, match="bmnc_b"):
        qi.j_invariant_qi_residual_from_boozer(
            bmnc_b=[1.0], xm_b=[0.0], xn_b=[0.0], iota_b=[0.4],
            G_b=[2.0], I_b=[0.0], nfp=2, pitch=[1.0])
    with pytest.raises(ValueError, match="weights"):
        qi.j_invariant_qi_residual_from_boozer(
            bmnc_b=[[1.0], [1.0]], xm_b=[0.0], xn_b=[0.0],
            iota_b=[0.4, 0.4], G_b=[2.0, 2.0], I_b=[0.0, 0.0],
            nfp=2, pitch=[1.0], weights=[1.0])


@pytest.mark.full
def test_implicit_boundary_gradient_matches_reconverged_fd():
    inp = VmecInput.from_file(
        Path(__file__).resolve().parents[1] / "examples/data/input.nfp1_QI")
    params = im.params_from_input(inp, device=None)
    term = qi.JInvariantQIResidual(
        [0.5], [1.0 / 0.95], mboz=6, nboz=6, nalpha=5,
        points_per_period=32, num_periods=3, max_wells=6,
        quadrature_order=32)

    def objective(p):
        solution = im.run(
            inp, p, ns=15, ftol=1.0e-12, max_iterations=12000,
            adjoint_tol=1.0e-12, device=None)
        return term.total_state(solution.state, solution.runtime)

    index = (int(inp.ntor), 1)
    implicit = float(np.asarray(jax.grad(objective)(params).rbc)[index])
    step = 3.0e-4
    values = [
        objective(dataclasses.replace(
            params, rbc=params.rbc.at[index].add(sign * step)))
        for sign in (-1.0, 1.0)
    ]
    finite_difference = float((values[1] - values[0]) / (2.0 * step))
    relative = abs(implicit - finite_difference) / max(
        abs(implicit), abs(finite_difference))
    assert relative < 3.0e-3

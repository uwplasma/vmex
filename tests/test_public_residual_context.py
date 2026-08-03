from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from vmex.core.input import VmecInput
from vmex.core import optimize
from vmex.core.omnigenity import QIResidual


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
pytestmark = pytest.mark.usefixtures("_module_jit_enabled")


@pytest.fixture(scope="module")
def legacy_qi_equilibrium():
    inp = VmecInput.from_file(DATA / "input.solovev")
    context = optimize.ImplicitResidualContext(
        inp,
        [(optimize.aspect_ratio, 1.0, 1.0)],
        max_mode=1,
        jac_solver="reverse",
        warm_start="state",
    )
    equilibrium = context.solution(context.x0)
    residual = optimize.LegacyQIResidual(
        [0.5],
        mboz=4,
        nboz=3,
        oversample=1,
        nphi=17,
        nalpha=5,
        n_bounce=7,
    )
    return inp, equilibrium, residual


def test_public_scalar_residual_context():
    inp = VmecInput.from_file(DATA / "input.solovev")
    context = optimize.ImplicitResidualContext(
        inp,
        [(optimize.aspect_ratio, 1.0, 1.0)],
        max_mode=1,
        jac_solver="reverse",
        warm_start="state",
        solve_kwargs={"mode": "cli", "lconm1": True, "use_fft": False},
    )
    residual, jacobian, equilibrium = context.residual_jacobian(context.x0)
    assert residual.shape == (1,)
    assert jacobian.shape == (1, context.x0.size)
    assert np.all(np.isfinite(jacobian))
    assert equilibrium.result.converged
    assert context.term_slices == (slice(0, 1),)


def test_public_context_jacobian_matches_central_difference():
    inp = VmecInput.from_file(DATA / "input.solovev")
    context = optimize.ImplicitResidualContext(
        inp,
        [(optimize.aspect_ratio, 1.0, 1.0)],
        max_mode=1,
        jac_solver="reverse",
        warm_start="state",
    )
    x = context.x0
    jacobian = context.jacobian(x)
    step = 3.0e-5
    for column in range(x.size):
        direction = np.zeros_like(x)
        direction[column] = step
        finite_difference = (
            context.residuals(x + direction) - context.residuals(x - direction)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            jacobian[:, column], finite_difference, rtol=2.0e-4, atol=2.0e-6
        )


def test_mean_iota_and_mirror_directional_derivatives():
    inp = VmecInput.from_file(DATA / "input.nfp2_QA")
    context = optimize.ImplicitResidualContext(
        inp,
        [
            (optimize.mean_iota, 0.0, 1.0),
            (optimize.mirror_ratio, 0.0, 1.0),
        ],
        max_mode=1,
        jac_solver="block",
        warm_start="state",
    )
    x = context.x0
    direction = np.linspace(-0.4, 0.7, x.size)
    direction /= np.linalg.norm(direction)
    analytic = context.jacobian(x) @ direction
    step = 2.0e-5
    finite_difference = (
        context.residuals(x + step * direction)
        - context.residuals(x - step * direction)
    ) / (2.0 * step)

    assert np.all(np.isfinite(analytic))
    assert np.all(np.abs(analytic) > 1.0e-8)
    # This converged, non-axisymmetric fixture measures about 2.0% relative
    # error for mean iota and 0.13% for mirror ratio against re-solved FD.
    np.testing.assert_allclose(analytic, finite_difference, rtol=3.0e-2, atol=1.0e-8)


def test_public_context_qs_residual_directional_derivative():
    inp = VmecInput.from_file(DATA / "input.solovev")
    qs = optimize.QuasisymmetryRatioResidual([0.5], 1, -1)
    context = optimize.ImplicitResidualContext(
        inp,
        [(qs, 0.0, 1.0)],
        max_mode=1,
        jac_solver="block",
        warm_start="state",
    )
    x = context.x0
    direction = np.linspace(0.6, -0.8, x.size)
    direction /= np.linalg.norm(direction)
    analytic = context.jacobian(x) @ direction
    step = 2.0e-5
    finite_difference = (
        context.residuals(x + step * direction)
        - context.residuals(x - step * direction)
    ) / (2.0 * step)

    assert analytic.shape == finite_difference.shape
    assert np.all(np.isfinite(analytic))
    assert np.linalg.norm(analytic) > 1.0e-8
    relative_error = (
        np.linalg.norm(analytic - finite_difference)
        / np.linalg.norm(finite_difference)
    )
    assert relative_error < 3.0e-2


def test_public_context_explicit_parameter_subset_and_order():
    inp = VmecInput.from_file(DATA / "input.solovev")
    full = optimize.ImplicitResidualContext(
        inp,
        [(optimize.aspect_ratio, 1.0, 1.0)],
        max_mode=2,
        jac_solver="block",
        warm_start="state",
    )
    indices = np.asarray([3, 0])
    reduced = optimize.ImplicitResidualContext(
        inp,
        [(optimize.aspect_ratio, 1.0, 1.0)],
        max_mode=2,
        parameter_indices=indices,
        jac_solver="block",
        warm_start="state",
    )
    reduced_reverse = optimize.ImplicitResidualContext(
        inp,
        [(optimize.aspect_ratio, 1.0, 1.0)],
        max_mode=2,
        parameter_indices=indices,
        jac_solver="reverse",
        warm_start="state",
    )
    np.testing.assert_allclose(reduced.x0, full.x0[indices])
    np.testing.assert_allclose(reduced.residuals(reduced.x0), full.residuals(full.x0))
    np.testing.assert_allclose(
        reduced.jacobian(reduced.x0), full.jacobian(full.x0)[:, indices]
    )
    np.testing.assert_allclose(
        reduced_reverse.jacobian(reduced_reverse.x0),
        full.jacobian(full.x0)[:, indices],
    )
    assert reduced.solution(reduced.x0).result.converged


def test_legacy_qi_options_are_python_scalars():
    residual = optimize.LegacyQIResidual(
        [0.25, 0.5], nphi=np.int64(31), nalpha=np.int64(7), n_bounce=np.int64(9)
    )
    assert residual.options["nphi"] == 31
    assert isinstance(residual.options["nphi"], int)


def test_legacy_qi_traceable_and_wout_routes_agree(legacy_qi_equilibrium):
    _, equilibrium, residual = legacy_qi_equilibrium
    traceable = residual.compute_state(equilibrium.state, equilibrium.runtime)
    from_wout = optimize.quasi_isodynamic_residual_from_wout(
        equilibrium.wout,
        surfaces=residual.surfaces,
        mboz=residual.mboz,
        nboz=residual.nboz,
        weights=residual.weights,
        **residual.options,
    )
    traceable_rows = np.asarray(traceable["residuals1d"])
    wout_rows = np.asarray(from_wout["residuals1d"])
    assert traceable_rows.shape == wout_rows.shape

    # The state route uses the solver's half-mesh field tables whereas the
    # WOUT route transforms reconstructed output tables. Their discretizations
    # are not bitwise identical; this fixture measures ~2.3% relative row-norm
    # disagreement and ~0.6% total disagreement at the selected low resolution.
    relative_row_error = (
        np.linalg.norm(traceable_rows - wout_rows) / np.linalg.norm(wout_rows)
    )
    assert relative_row_error < 3.0e-2
    np.testing.assert_allclose(
        float(traceable["total"]), float(from_wout["total"]), rtol=1.0e-2
    )


def test_legacy_qi_eager_jit_parity_and_state_gradient(legacy_qi_equilibrium):
    _, equilibrium, residual = legacy_qi_equilibrium

    def rows(state):
        return residual.residuals_state(state, equilibrium.runtime)

    eager = rows(equilibrium.state)
    compiled = jax.jit(rows)(equilibrium.state)
    np.testing.assert_allclose(
        np.asarray(compiled), np.asarray(eager), rtol=1.0e-12, atol=1.0e-15
    )

    gradient = jax.grad(
        lambda state: residual.total_state(state, equilibrium.runtime)
    )(equilibrium.state)
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(gradient)]
    assert all(np.all(np.isfinite(leaf)) for leaf in leaves)
    assert np.sqrt(sum(float(np.sum(leaf * leaf)) for leaf in leaves)) > 1.0e-10


def test_legacy_qi_composes_with_implicit_least_squares(legacy_qi_equilibrium):
    inp, _, residual = legacy_qi_equilibrium
    result = optimize.least_squares(
        [(residual, 0.0, 1.0)],
        inp,
        max_mode=1,
        jac="implicit",
        jac_solver="block",
        warm_start="state",
        max_nfev=1,
    )
    assert result.nfev == 1
    assert result.fun.shape == (240,)
    assert result.jac.shape == (240, 2)
    assert np.all(np.isfinite(result.jac))
    assert np.linalg.norm(result.jac) > 1.0e-10


@pytest.mark.parametrize(
    "residual",
    [
        QIResidual([0.5], mboz=4, nboz=3, oversample=1),
        optimize.LegacyQIResidual([0.5], mboz=4, nboz=3, oversample=1),
    ],
)
def test_traceable_qi_definitions_share_clear_lasym_rejection(residual):
    runtime = SimpleNamespace(setup=SimpleNamespace(lasym=True))
    with pytest.raises(NotImplementedError, match="stellarator-symmetric.*lasym = False"):
        residual.compute_state(None, runtime)


@pytest.mark.parametrize(
    "control",
    ["aligned_profile", "weighted_shuffle", "shuffle_profile_nphi_out"],
)
def test_legacy_qi_rejects_removed_controls(control):
    with pytest.raises(TypeError, match=control):
        optimize.LegacyQIResidual([0.5], **{control: True})


def test_maximum_elongation_directional_derivative():
    inp = VmecInput.from_file(DATA / "input.solovev")
    rbc = np.array(inp.rbc, copy=True)
    zbs = np.array(inp.zbs, copy=True)
    rbc[inp.ntor, 1] = 0.35
    zbs[inp.ntor, 1] = 0.20
    inp = replace(inp, rbc=rbc, zbs=zbs)
    context = optimize.ImplicitResidualContext(
        inp,
        [(optimize.maximum_elongation, 1.0, 1.0)],
        max_mode=1,
        jac_solver="reverse",
        warm_start="state",
    )
    x = context.x0
    direction = np.linspace(-0.5, 0.75, x.size)
    direction /= np.linalg.norm(direction)
    analytic = (context.jacobian(x) @ direction).item()
    step = 2.0e-5
    finite_difference = (
        (context.residuals(x + step * direction)
         - context.residuals(x - step * direction))
        / (2.0 * step)
    ).item()
    np.testing.assert_allclose(analytic, finite_difference, rtol=5.0e-4, atol=2.0e-6)

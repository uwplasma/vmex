from dataclasses import replace
from pathlib import Path

import numpy as np

from vmex.core.input import VmecInput
from vmex.core import optimize


DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


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


def test_public_context_explicit_parameter_subset_and_order():
    inp = VmecInput.from_file(DATA / "input.solovev")
    full = optimize.ImplicitResidualContext(
        inp,
        [(optimize.aspect_ratio, 1.0, 1.0)],
        max_mode=1,
        jac_solver="reverse",
        warm_start="state",
    )
    indices = np.asarray([1, 0])
    reduced = optimize.ImplicitResidualContext(
        inp,
        [(optimize.aspect_ratio, 1.0, 1.0)],
        max_mode=1,
        parameter_indices=indices,
        jac_solver="reverse",
        warm_start="state",
    )
    np.testing.assert_allclose(reduced.x0, full.x0[indices])
    np.testing.assert_allclose(reduced.residuals(reduced.x0), full.residuals(full.x0))
    np.testing.assert_allclose(
        reduced.jacobian(reduced.x0), full.jacobian(full.x0)[:, indices]
    )
    assert reduced.solution(reduced.x0).result.converged


def test_legacy_qi_options_are_python_scalars():
    residual = optimize.LegacyQIResidual(
        [0.25, 0.5], nphi=np.int64(31), nalpha=np.int64(7), n_bounce=np.int64(9)
    )
    assert residual.options["nphi"] == 31
    assert isinstance(residual.options["nphi"], int)


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

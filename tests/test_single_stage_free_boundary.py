"""Coil-shape gradients through a free-boundary equilibrium (single-stage optimization).

The Landreman-Paul precise-QA coils hold a vacuum free boundary at 5 x 5
modes, ns = 16.  The adjoint derivative of the aspect ratio with respect to
every Fourier coefficient of the coils, along one random direction, must match
a central difference of two independent solves, and the converged boundary's
pressure jump (VMEC's DEL-BSQ) must be small.
"""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmex as vj
from vmex import optimize as opt
from vmex.core import implicit as im

pytest.importorskip("essos")
from essos.coils import Coils  # noqa: E402
from essos.fields import BiotSavart  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
NS, FTOL, MODES = 16, 1e-13, 5


def test_coil_gradient_matches_resolve_finite_difference_and_boundary_balances():
    coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
    inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
    k = inp.ntor - MODES
    window = (slice(k, k + 2 * MODES + 1), slice(None, MODES))
    inp = replace(
        inp, lfreeb=True, mgrid_file="direct field", mpol=MODES, ntor=MODES, nzeta=16,
        rbc=inp.rbc[window], zbs=inp.zbs[window], rbs=inp.rbs[window], zbc=inp.zbc[window],
        raxis_c=inp.raxis_c[:MODES + 1], zaxis_s=inp.zaxis_s[:MODES + 1],
        raxis_s=inp.raxis_s[:MODES + 1], zaxis_c=inp.zaxis_c[:MODES + 1],
        phiedge=-0.025, ns_array=[NS], niter_array=[4000], ftol_array=[FTOL])
    x0 = jnp.asarray(coils.dofs)

    def field(u):
        return BiotSavart(coils.with_dofs(x0 + u))

    params = im.params_from_input(inp)
    cfg = vj.make_free_boundary_config(
        inp, field(jnp.zeros_like(x0)), ns=NS, ftol=FTOL, max_iterations=4000,
        adjoint_tol=1e-10, field_from_parameters=field)
    context = im.runtime_from_params(params, cfg.implicit)

    def aspect(u):
        state, status, _, _ = vj.solve_free_boundary_implicit_status(params, u, cfg)
        return opt.aspect_ratio(state, context), status

    (value, status), gradient = jax.value_and_grad(aspect, has_aux=True)(jnp.zeros_like(x0))
    assert int(status) == 0 and abs(float(value) - 6.0) < 1e-2
    direction = jnp.asarray(np.random.default_rng(0).standard_normal(x0.size))
    direction /= jnp.linalg.norm(direction)
    # Smaller steps measure the solver's round-off, not the derivative.
    step = 1e-4
    (plus, _), (minus, _) = aspect(step * direction), aspect(-step * direction)
    np.testing.assert_allclose(
        float(gradient @ direction), (float(plus) - float(minus)) / (2 * step), rtol=5e-3)

    result = vj.solve_free_boundary_multigrid(
        inp, external_field=field(jnp.zeros_like(x0)), raise_on_max_iterations=False)
    assert result.converged and 0.0 < result.vacuum.delbsq < 1e-2

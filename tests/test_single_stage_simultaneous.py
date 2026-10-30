"""True simultaneous plasma-boundary + coil single-stage joint gradient:
``jax.value_and_grad`` of a combined objective threads through the implicit
adjoint (boundary dofs) AND virtual casing (coil dofs) at once.  Checked:
``surface_field_data_from_state`` rebuilds the surface field traceably
(bit-exact vs the wout path), and a frozen ``PrecisionPlan`` + the NaN-safe
Laplace kernel gradient keep the boundary gradient finite.  Skipped
without ``virtual_casing_jax``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import vmex as vj  # noqa: E402
from vmex.core import virtual_casing as FBD  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
from vmex.core.mgrid import MgridField, read_mgrid  # noqa: E402
from vmex.core.wout import wout_from_state  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
INPUT = DATA / "input.cth_like_free_bdy"
MGRID = DATA / "mgrid_cth_like.nc"
NP = NT = 16
SOLVE = dict(ftol=1e-10, max_iterations=1500)
pytestmark = pytest.mark.skipif(
    not FBD._HAVE_VCJ,
    reason="requires virtual_casing_jax",
)


def test_joint_trial_rejection_is_costly_differentiable_and_cached():
    """An uncertified equilibrium cannot beat the seed or evaluate coil costs."""
    from vmex.core.single_stage import SingleStageProblem
    from vmex.core.monitoring import OptimizationMonitor

    # Exercise the host optimizer contract without compiling an equilibrium.
    joint = SingleStageProblem.__new__(SingleStageProblem)
    joint.problem = SimpleNamespace(metadata={"term_slices": [("plasma", 0, 1)]})
    joint.monitor = OptimizationMonitor()
    joint.counts, joint._cached = {"trials": 0, "rejected": 0}, {}
    joint.rejected_trial_cost = 10.0
    coil_calls = []

    def plasma_vg(u):
        accepted = float(u[0]) < 0.5
        return ((2.0, (np.array([2.0 if accepted else 0.0]), np.array([0.0]))),
                np.zeros(1))

    def coil_vg(u):
        coil_calls.append(np.asarray(u))
        return ((2.0, np.array([2.0, 0.0, 0.0, 0.0])), np.zeros(1))

    joint._plasma_vg, joint._coil_vg = plasma_vg, coil_vg
    seed, gradient = joint.value_and_grad(np.zeros(1))
    assert seed == 1.0
    gradient[:] = 123  # SciPy must not be able to mutate the cached gradient.
    assert joint.value_and_grad(np.zeros(1))[1][0] == 0
    assert joint.counts["trials"] == 1
    value, gradient = joint.value_and_grad(np.ones(1))
    assert value > seed
    step = 1e-5
    difference = (joint.value_and_grad(np.array([1 + step]))[0]
                  - joint.value_and_grad(np.array([1 - step]))[0]) / (2 * step)
    np.testing.assert_allclose(gradient[0], difference, rtol=1e-9)
    assert joint.counts["rejected"] == 3
    assert len(coil_calls) == 1


@pytest.mark.parametrize("failure", [None, "fixed solve", "free solve", "fixed NaN", "held infinity"])
def test_joint_design_verdict_requires_finite_metrics_and_both_solves(failure, capsys):
    """A good fitted boundary cannot hide a failed or nonfinite held-boundary check."""
    from vmex.core.single_stage import design_target_report

    metrics = {"min |iota|": 0.5, "aspect": 5.0, "B.n/B RMS": 0.001,
               "coil-surface distance": 0.2, "coil-coil distance": 0.2,
               "maximum curvature": 5.0, "beta": 0.005, "verification solve converged": True}
    free = {"converged": True, "delbsq": 0.001, "sampled_boundary_distance_m": 0.001,
            "min_abs_iota": 0.5, "aspect": 5.0, "coil_surface_distance_m": 0.2, "beta": 0.005}
    if failure == "fixed solve":
        metrics["verification solve converged"] = False
    elif failure == "free solve":
        free = {"converged": False}  # Failed solves need no physical metrics.
    elif failure == "fixed NaN":
        metrics["B.n/B RMS"] = np.nan
    elif failure == "held infinity":
        free["min_abs_iota"] = np.inf
    verdict = design_target_report(
        metrics, free, iota_floor=0.42, aspect_limit=6, normal_field_limit=0.01,
        clearance=0.15, coil_distance=0.17, curvature=7, beta_bounds=(0.0045, 0.0055),
        delbsq_limit=0.02, boundary_distance_limit=0.02)
    assert verdict["met"] == (failure is None)
    assert bool(verdict["unmet"]) == (failure is not None)
    assert "held DEL-BSQ <=" in verdict["targets"]
    output = capsys.readouterr().out
    assert output.count("All stated targets met.") == (failure is None)
    assert output.count("did NOT meet its stated targets") == (failure is not None)


@pytest.fixture(scope="module")
def cth():
    if not MGRID.exists():
        pytest.skip("mgrid_cth_like.nc not fetched (tools/fetch_assets.py)")
    inp = vj.VmecInput.from_file(str(INPUT))
    res = vj.solve_multigrid(inp, verbose=False)
    assert res.converged
    base = MgridField.from_mgrid_data(read_mgrid(MGRID), extcur=jnp.asarray([4700.0, 1000.0]))
    return inp, res, base


def _mf(base, extcur):
    return MgridField(br=base.br, bp=base.bp, bz=base.bz, extcur=extcur,
                      rmin=base.rmin, rmax=base.rmax, zmin=base.zmin,
                      zmax=base.zmax, nfp=base.nfp)


@pytest.mark.full
def test_surface_field_from_state_matches_wout(cth):
    """The traceable state->surface bridge reproduces the wout path bit-for-bit."""
    inp, res, _ = cth
    wout = wout_from_state(inp=inp, state=res.state, fsqr=float(res.fsqr),
                           fsqz=float(res.fsqz), fsql=float(res.fsql),
                           niter=int(res.iterations), converged=True)
    sd_w = FBD.surface_field_data_from_wout(wout, nphi=20, ntheta=20)
    sd_s = FBD.surface_field_data_from_state(inp, res.state, nphi=20, ntheta=20)
    for name in ("gamma", "B_total", "normal", "area_vector"):
        a = np.asarray(getattr(sd_w, name))
        b = np.asarray(getattr(sd_s, name))
        assert np.max(np.abs(a - b)) <= 1e-9 * (np.max(np.abs(a)) + 1e-30), name


def _joint(inp, base, plan, target):
    def objective(params, extcur):
        sol = im.run(inp, params, **SOLVE)
        sd = FBD.surface_field_data_from_state(inp, sol.state, nphi=NP, ntheta=NT)
        prob = FBD.FreeBoundaryDiffProblem.from_surface_data(sd, digits=4, precision=plan)
        return prob.bnormal_objective(_mf(base, extcur)) + (sol.iota_edge - target) ** 2
    return objective


def _setup(cth):
    inp, _, base = cth
    p0 = im.params_from_input(inp)
    sol0 = im.run(inp, p0, **SOLVE)
    sd0 = FBD.surface_field_data_from_state(inp, sol0.state, nphi=NP, ntheta=NT)
    plan = FBD.plan_vc_precision(sd0, digits=4)
    obj = _joint(inp, base, plan, float(sol0.iota_edge) + 0.03)
    return inp, base, p0, obj


@pytest.mark.full
def test_joint_gradient_is_finite(cth):
    """value_and_grad wrt BOTH boundary params and coil currents is finite.

    This is the regression guard for the virtual_casing_jax NaN-safe Laplace
    kernel gradient: before the ``_safe_rinv`` double-``where`` fix the boundary
    block came back all-NaN while the value was correct.
    """
    inp, base, p0, obj = _setup(cth)
    val, (gp, ge) = jax.value_and_grad(obj, argnums=(0, 1))(p0, base.extcur)
    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.asarray(gp.rbc))), "boundary gradient not finite"
    assert np.all(np.isfinite(np.asarray(ge))), "coil gradient not finite"
    # both blocks must actually be exercised (non-trivial)
    assert np.linalg.norm(np.asarray(gp.rbc)) > 0.0
    assert np.linalg.norm(np.asarray(ge)) > 0.0


@pytest.mark.full
def test_joint_gradient_matches_finite_difference(cth):
    """AD vs central FD for one coil current and one boundary coefficient."""
    inp, base, p0, obj = _setup(cth)
    _, (gp, ge) = jax.value_and_grad(obj, argnums=(0, 1))(p0, base.extcur)
    f = lambda p, e: float(obj(p, e))  # noqa: E731
    ntor = int(inp.ntor)

    # coil current (no equilibrium re-solve -> clean FD)
    h = 1.0
    fd_e = (f(p0, base.extcur.at[0].add(h)) - f(p0, base.extcur.at[0].add(-h))) / (2 * h)
    assert abs(float(np.asarray(ge)[0]) / fd_e - 1.0) < 5e-3

    # boundary coefficient (threads the adjoint AND the moving VC surface); the
    # FD re-solves the equilibrium each step to finite ftol, so its floor is ~1%.
    h = 3e-4
    pp = dataclasses.replace(p0, rbc=p0.rbc.at[ntor, 1].add(h))
    pm = dataclasses.replace(p0, rbc=p0.rbc.at[ntor, 1].add(-h))
    fd_r = (f(pp, base.extcur) - f(pm, base.extcur)) / (2 * h)
    assert abs(float(np.asarray(gp.rbc)[ntor, 1]) / fd_r - 1.0) < 3e-2

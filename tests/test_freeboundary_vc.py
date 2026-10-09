"""Free boundary from the three virtual-casing interface conditions.

:mod:`vmex.core.freeboundary_vc` solves for the boundary on which ``B.n``,
the pressure jump and the sheet current vanish. Lanes: input validation; the
residual rows' definitions on a converged finite-beta state; and (``full``)
the vacuum and finite-beta Landreman--Paul QA cases with their bundled ESSOS
coils, where the solve agrees with NESTOR and is independent of its start.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import freeboundary_vc as fvc  # noqa: E402
from vmex.core import virtual_casing as vc  # noqa: E402
from vmex.core.errors import TrialRejected  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
needs_vc = pytest.mark.skipif(not vc.have_virtual_casing_jax(), reason="requires virtual_casing_jax")


def _deck(name, mpol, ns):
    inp = vj.VmecInput.from_file(DATA / name).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=16)
    return replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)", ns_array=np.array([ns]),
                   niter_array=np.array([5000]), ftol_array=np.array([1e-10]), delt=0.5)


@dataclass(frozen=True)
class _FilamentField:
    """Biot-Savart field of filament coils as a pytree (arrays as leaves, so the model compiles once)."""

    gamma: Any  # (coils, points, 3)
    gamma_dash: Any
    currents: Any  # (coils,)

    def b_cyl(self, r, phi, z):
        xyz = jnp.stack(jnp.broadcast_arrays(r * jnp.cos(phi), r * jnp.sin(phi), z), axis=-1)
        d = xyz[..., None, None, :] - self.gamma
        dB = jnp.cross(self.gamma_dash, d) * (jnp.sum(d * d, axis=-1) ** -1.5)[..., None]
        B = 1e-7 * jnp.mean(jnp.sum(dB * self.currents[:, None, None], axis=-3), axis=-2)
        cos, sin = jnp.cos(phi), jnp.sin(phi)
        return cos * B[..., 0] + sin * B[..., 1], -sin * B[..., 0] + cos * B[..., 1], B[..., 2]


jax.tree_util.register_dataclass(_FilamentField, data_fields=["gamma", "gamma_dash", "currents"], meta_fields=[])


def _coil_field(name):
    """The bundled ESSOS coils as an exact filament field (no mgrid tabulation)."""
    pytest.importorskip("essos")
    from essos.coils import Coils

    coils = Coils.from_json(str(DATA / name))
    return _FilamentField(jnp.asarray(coils.gamma), jnp.asarray(coils.gamma_dash), jnp.asarray(coils.currents))


def test_nonzero_edge_pressure_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), pmass_type="power_series",
                  am=np.array([1.0e3, -0.5e3] + [0.0] * 19))
    with pytest.raises(ValueError, match="edge pressure"):
        fvc.solve_free_boundary_three_term(inp, external_field=lambda xyz: xyz)


def test_asymmetric_boundary_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lasym=True)
    with pytest.raises(NotImplementedError, match="lasym"):
        fvc.solve_free_boundary_three_term(inp, external_field=lambda xyz: xyz)


def test_unknown_boundary_condition_is_rejected():
    with pytest.raises(ValueError, match="boundary_condition"):
        vj.solve_free_boundary_multigrid(_deck("input.LandremanPaul2021_QA_lowres", 3, 11),
                                         external_field=lambda xyz: xyz, boundary_condition="sheet")


@needs_vc
@pytest.mark.usefixtures("_module_jit_enabled")
def test_rows_are_the_three_interface_conditions():
    """Normal row = B_out.n/|B|, sheet rows tangent, and together they are the whole jump."""
    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 11), lfreeb=False)
    eq = opt.solve_equilibrium(inp)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    rows, weights = fvc.boundary_residual(inp, eq.solution, field, runtime=eq.solver_context, nphi=16, ntheta=16)
    data = vc.surface_field_data_from_state(inp, eq.solution, runtime=eq.solver_context, nphi=16, ntheta=16)
    iface = vc.PlasmaVacuumInterface.from_surface_data(data)
    B_in = np.asarray(data.B_total)
    B = np.linalg.norm(B_in, axis=0)
    jump = (np.asarray(iface.total_B_out(field)) - B_in) / B
    normal = np.asarray(iface.normal)
    np.testing.assert_allclose(rows[0], np.asarray(iface.bnormal_residual(field)) / B, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(rows[1], np.asarray(iface.pressure_balance_residual(field)) / (2 * B**2),
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(np.sum(np.asarray(rows[2:]) * normal, axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(rows[0] ** 2 + np.sum(np.asarray(rows[2:]) ** 2, axis=0), np.sum(jump**2, axis=0),
                               rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=1e-12)
    summary = fvc.summarize_boundary_residual(rows, weights)
    assert 0 < summary.normal < 0.05 and 0 < summary.sheet_current < 0.05


def _lcfs_distance(w, w0, nphi=5, ntheta=361):
    """Max over sampled planes of the distance from ``w``'s LCFS to ``w0``'s, in metres."""
    theta, worst = np.linspace(0, 2 * np.pi, ntheta), 0.0
    for phi in np.linspace(0, np.pi / int(w.nfp), nphi):
        def rz(x):
            angle = np.outer(theta, np.asarray(x.xm)) - np.asarray(x.xn) * phi
            return np.cos(angle) @ np.asarray(x.rmnc)[-1], np.sin(angle) @ np.asarray(x.zmns)[-1]
        (R, Z), (R0, Z0) = rz(w), rz(w0)
        worst = max(worst, float(np.max(np.min(np.hypot(R[:, None] - R0[None], Z[:, None] - Z0[None]), axis=1))))
    return worst


def _residual(inp, eq, field, n=48):
    return fvc.summarize_boundary_residual(*fvc.boundary_residual(
        inp, eq.solution, field, runtime=eq.solver_context, nphi=n, ntheta=n))


def _nestor(inp, field):
    """NESTOR free boundary as an Equilibrium (fixed-boundary solve on its LCFS) and its deck."""
    free = vj.solve_free_boundary_multigrid(inp, external_field=field, report_boundary_residual=True)
    wout = vj.wout_from_state(inp=inp, state=free.state, fsqr=float(free.fsqr), fsqz=float(free.fsqz),
                              fsql=float(free.fsql), niter=int(free.iterations), converged=bool(free.converged),
                              vacuum_output=free.vacuum)
    rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
    for m, n, r, z in zip(np.asarray(wout.xm, int), np.asarray(wout.xn, int) // int(wout.nfp),
                          np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1]):
        rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    deck = replace(inp, rbc=rbc, zbs=zbs, lfreeb=False, raxis_c=np.asarray(wout.raxis_cc)[:inp.ntor + 1],
                   zaxis_s=np.asarray(wout.zaxis_cs)[:inp.ntor + 1])
    return opt.solve_equilibrium(deck, initial_state=free.state), deck, free.boundary_residual


@needs_vc
@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_vacuum_free_boundary_agrees_with_nestor():
    """In vacuum the free boundary is a flux surface of the coil field alone: both methods find it.

    VMEC + NESTOR itself lands ~1.6 mm from the exact free boundary of an
    essentially exact field (VMEC2000 1.3 mm), so the methods are compared at
    that level; from either start the interface conditions end below NESTOR's.
    At mpol = 4 the least-squares minimum is also flat to ~3 mm, so the
    surfaces are compared within the sum of the two floors, and more sharply
    through the residuals and the rotational transform.
    """
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 25), phiedge=-0.025, curtor=0.0,
                  am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    nestor, nestor_deck, reported = _nestor(inp, field)
    before = _residual(inp, nestor, field)
    for name in ("normal", "pressure", "sheet_current"):   # the gate on the NESTOR result
        np.testing.assert_allclose(getattr(reported, name), getattr(before, name), rtol=0.25)
    # The net-current row's reference: outside the plasma the toroidal circulation is the coils' alone, so a
    # free boundary's edge R B_phi equals the coil current that _coil_net_current integrates on the axis.
    np.testing.assert_allclose(abs(fvc._coil_net_current(field, nestor.wout)), abs(float(nestor.wout.rbtor)),
                               rtol=1e-3)
    fits = [fvc.solve_free_boundary_three_term(inp, external_field=field, initial_boundary=start)
            for start in (None, nestor_deck)]
    for fit in fits:
        after = _residual(inp, fit.equilibrium, field)
        assert after.normal < before.normal and after.sheet_current < before.sheet_current
    walls = [fit.equilibrium.wout for fit in fits]
    for wall in walls:   # budget: NESTOR's own floor (~1.6 mm) plus the flat least-squares minimum (~3 mm)
        assert _lcfs_distance(wall, nestor.wout) < 5e-3
    np.testing.assert_allclose(np.asarray(walls[0].iotaf)[[0, -1]], np.asarray(nestor.wout.iotaf)[[0, -1]], atol=3e-3)
    # The model's Jacobian (implicit state tangents, field as an argument) is upstream's implicit Jacobian of
    # the same rows at the same state (the seed's: two independent re-solves of a boundary differ at the
    # floor); and a warm restart at an unchanged field stays where it is.
    fit, model = fits[0], fits[0].model
    problem = opt.make_problem(model.fixed, objective_terms=[(lambda state, runtime: model.rows_at(
        state, runtime, field), 0.0, 1.0)], weight_semantics="residual", max_mode=model.max_mode,
        vary_major_radius=True, jacobian_batch_size=8, device=jax.devices()[0])
    J = model.linearize(model.x0, (*model.seed, model.params0, True), field)[0]
    J_upstream = np.asarray(problem.residual_jac(model.x0))
    assert np.linalg.norm(J - J_upstream) < 1e-9 * np.linalg.norm(J_upstream)
    again = fvc.solve_free_boundary_three_term(inp, external_field=field, previous=fit)
    assert again.njev == 0 and again.cost <= fit.cost * (1 + 1e-6)
    assert _lcfs_distance(again.equilibrium.wout, walls[0]) < 1e-4


@needs_vc
@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_finite_beta_free_boundary_removes_the_sheet_current():
    """LP QA at 0.5% beta with its fitted coils, through the solver's public option."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    design = _residual(inp, opt.solve_equilibrium(replace(inp, lfreeb=False)), field)
    result = vj.solve_free_boundary_multigrid(inp, external_field=field, boundary_condition="three_term")
    assert result.converged and result.vacuum is None
    res = result.boundary_residual
    for name in ("normal", "pressure", "sheet_current"):
        assert getattr(res, name) < 0.25 * getattr(design, name), name
    assert res.sheet_current < 1.5e-3


def test_unknown_problem_boundary_condition_is_rejected():
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    with pytest.raises(ValueError, match="boundary_condition"):
        FreeBoundaryProblem.from_loss(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lambda s, r, x: 0.0,
                                      np.zeros(1), field_from_parameters=lambda x: None, boundary_condition="sheet")


def test_problem_boundary_condition_dispatches_to_three_term(monkeypatch):
    """``boundary_condition="three_term"`` hands the loss, maps, quantities and options to the three-term problem."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    calls = []
    monkeypatch.setattr(fvc.ThreeTermFreeBoundaryProblem, "from_loss",
                        classmethod(lambda cls, *args, **kwargs: calls.append((args, kwargs)) or "problem"))
    inp, loss, field = _deck("input.LandremanPaul2021_QA_lowres", 3, 11), (lambda s, r, x: 0.0), (lambda x: None)
    got = FreeBoundaryProblem.from_loss(inp, loss, np.zeros(1), field_from_parameters=field, quantities=(len,),
                                        boundary_condition="three_term", three_term_options=dict(nphi=16))
    (args, kwargs), = calls
    assert got == "problem" and args[:2] == (inp, loss) and kwargs["field_from_parameters"] is field
    assert kwargs["quantities"] == (len,) and kwargs["nphi"] == 16


@needs_vc
@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_three_term_problem_gradients_match_resolved_differences():
    """Design gradients by the implicit function theorem of the boundary fit, against re-solved central
    differences, along a plasma coordinate (PHIEDGE) and a field coordinate (the coil currents)."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 13), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[1])),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        quantities=(opt.major_radius,),
        parameter_quantities=(lambda state, runtime, x: opt.aspect_ratio(state, runtime) * (1 + x[1]),),
        boundary_condition="three_term",
        three_term_options=dict(nphi=24, ntheta=24, quadrature=(4 * 2 * 24, 48), boundary_ftol=1e-10,
                                boundary_max_nfev=80))
    assert isinstance(problem, fvc.ThreeTermFreeBoundaryProblem)
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    jac = problem.constraint_jac(x0)
    assert problem.boundary_residual(x0).sheet_current < 1e-2
    for k, h in ((0, 2e-3), (1, 2e-3)):
        e = h * np.eye(2)[k]
        values = [np.r_[problem.fun(x0 + s * e), problem.constraint_values(x0 + s * e)] for s in (1, -1)]
        fd = (values[0] - values[1]) / (2 * h)
        np.testing.assert_allclose(np.r_[grad[k], jac[:, k]], fd, rtol=2e-2, atol=1e-8 * max(1.0, abs(value)))
    problem.accept_x(x0 + 2e-3 * np.eye(2)[1])
    assert problem.accepted_step == 1
    problem.boundary_max_residual = 1e-12  # no fit gets there: the trial is no solution
    with pytest.raises(TrialRejected):
        problem.fun(x0 + 1e-3 * np.eye(2)[0])


def _lp_beta0p5_profiles(inp):
    """Landreman--Buller--Drevlak kinetic profiles whose 2 e n T is the LP QA 0.5% deck's pressure."""
    from vmex.core import bootstrap as bs

    n0 = 1.5e20
    T0 = float(inp.am[0]) / (2 * bs.ELEMENTARY_CHARGE * n0)
    return bs.KineticProfiles(ne_coeffs=n0 * np.array([1, 0, 0, 0, 0, -1.0]), Te_coeffs=T0 * np.array([1, -1.0]),
                              Ti_coeffs=T0 * np.array([1, -1.0]))


def test_bootstrap_current_profile_layout():
    """A knot per half-mesh surface, I'(0) = 0 and three unknown knots inside s_1, CURTOR its integral."""
    from vmex.core.profiles import current

    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 25), lfreeb=False)
    from vmex.core.bootstrap import HalfMeshCurrent

    block = HalfMeshCurrent(inp, _lp_beta0p5_profiles(inp), 0)
    s = (np.arange(1, 25) - 0.5) / 24
    np.testing.assert_allclose(block.knots[4:-1], s)
    values = np.asarray(block.values(block.x0))
    assert values[0] == 0.0 and block.deck.pcurr_type == "line_segment_ip"
    np.testing.assert_array_equal(block.deck.ac_aux_s, block.knots)
    np.testing.assert_allclose(block.deck.ac_aux_f, values)
    assert block.x0.size == 3 + 24
    assert block.deck.curtor == pytest.approx(float(current("line_segment_ip", inp.ac, block.knots, values, 1.0)))
    # the deck's own enclosed current away from the axis, where only the first cell differs
    deck_I = lambda d, x: float(d.curtor) * np.asarray(current(d.pcurr_type, d.ac, d.ac_aux_s, d.ac_aux_f, x)) / float(  # noqa: E731
        current(d.pcurr_type, d.ac, d.ac_aux_s, d.ac_aux_f, 1.0))
    outer = s[s > 0.2]
    np.testing.assert_allclose(deck_I(block.deck, outer), deck_I(inp, outer), rtol=2e-2)


@needs_vc
@pytest.mark.full
@pytest.mark.usefixtures("_module_jit_enabled")
def test_bootstrap_current_solved_with_the_free_boundary():
    """LP QA at 0.5% beta: boundary and Redl current in one solve, with exact columns for both."""
    from vmex.core import bootstrap as bs

    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    profiles = _lp_beta0p5_profiles(inp)
    fit = fvc.solve_free_boundary_three_term(inp, external_field=field, bootstrap=profiles, jacobian_ftol=None)
    model = fit.model
    state, mask, params, _ = fit.aux
    assert fit.x.size == model.n_boundary + 3 + 24
    assert model.bootstrap_residual(state, params) < 1e-2
    assert fit.boundary_residual.sheet_current < 2e-3
    # independently of the inversion: VMEC's <J.B> (finite-difference identity) is Redl's away from the ends
    runtime = model._im.runtime_from_params(params, model.cfg)
    hm = bs._half_mesh_fields(state, runtime)
    s = np.asarray(hm.s_half)
    jr = np.asarray(bs.j_dot_B_redl(profiles, bs._geometry_from_half(hm, hm.s_half, n_lambda=bs.N_LAMBDA), 0)[0])
    jv = np.asarray(bs._jv_from_half(hm, hm.s_half))
    inner = (s > 0.1) & (s < 0.9)
    assert np.max(np.abs(jv - jr)[inner]) < 0.05 * np.max(np.abs(jr))
    # Jacobian columns of a boundary and two current coordinates against central differences of re-solved rows
    # (small steps: near the axis Redl's rows are strongly nonlinear in the boundary, through f_t ~ eps^(1/2))
    J = model.linearize(fit.x, fit.aux, field)[0]
    for k in (0, model.n_boundary + 3, model.n_boundary + 15):
        h = 1e-5 * model.x_scale[k]
        rows = [model.evaluate(fit.x + sign * h * np.eye(fit.x.size)[k], params, field, tight=True)[0]
                for sign in (1, -1)]
        fd = (rows[0] - rows[1]) / (2 * h)
        assert np.linalg.norm(J[:, k] - fd) < 1e-4 * np.linalg.norm(fd), k


@pytest.mark.usefixtures("_module_jit_enabled")
def test_fixed_boundary_bootstrap_model_matches_picard():
    """fixed_boundary=True: the Gauss-Newton current at a fixed boundary is the Picard loop's, with exact columns."""
    from vmex.core import bootstrap as bs

    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 25), lfreeb=False,
                  ftol_array=np.array([1e-13]), niter_array=np.array([20000]))
    profiles = _lp_beta0p5_profiles(inp)
    model = fvc.ThreeTermFreeBoundaryModel(inp, bootstrap=profiles, fixed_boundary=True)
    assert model.n_boundary == 0 and model.x0.size == 3 + 24  # three knots inside s_1, then the half mesh
    out = model.solve_boundary(model.params0, None, ftol=1e-10)
    state, _, params, _ = out["aux"]
    assert model.bootstrap_residual(state, params) < 1e-4
    picard = bs.self_consistent_bootstrap(inp, profiles, 0, n_iter=40, tol=1e-6, relax=0.5, profile="half_mesh")
    np.testing.assert_allclose(np.asarray(params.ac_aux_f), picard.input.ac_aux_f,
                               rtol=0, atol=2e-4 * np.max(np.abs(picard.input.ac_aux_f)))
    J = model.linearize(out["x"], out["aux"], None)[0]
    k, h = 5, 1e-5 * model.x_scale[5]
    rows = [model.evaluate(out["x"] + sign * h * np.eye(out["x"].size)[k], params, None, tight=True)[0]
            for sign in (1, -1)]
    fd = (rows[0] - rows[1]) / (2 * h)
    assert np.linalg.norm(J[:, k] - fd) < 1e-5 * np.linalg.norm(fd)

"""Free boundary from the three virtual-casing interface conditions.

:mod:`vmex.core.freeboundary_vc` solves for the boundary on which ``B.n``,
the pressure jump and the sheet current vanish. PR lane (mpol 3, ns 11, 16 x 16
grids): input validation; the rows' definitions; the vacuum solve from two
starts with its rows, Jacobian and warm restart against finite differences;
design gradients against re-solved differences; the fixed-boundary bootstrap
current; Newton on stand-in rows. Full lane (``full``, mpol 4, ns 25): the
Landreman--Paul QA cases with their bundled ESSOS coils -- vacuum against
NESTOR, finite beta and the bootstrap current with the free boundary -- and the
bootstrap current against the Picard loop.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
import types
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
from vmex.core.errors import TrialRejected, VmecError  # noqa: E402

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


def _stand_in(n, rows, jac, *, vacuum=True, repeat=None):
    """A model whose rows and Jacobian are ``rows(z, tight)`` and ``jac(z)`` (no equilibrium): for Newton's logic.

    ``repeat`` gives the rows of a second solve of the same point (default: the same rows, no noise)."""
    model = fvc.ThreeTermsFreeBoundaryModel.__new__(fvc.ThreeTermsFreeBoundaryModel)
    model.vacuum, model.fixed_boundary = vacuum, False
    model.x0, model.x_scale, model.n_boundary, model.R0 = np.zeros(n), np.ones(n), n, 1.0
    model.boundary_modes = [(1, 0)] * (n - 1) + [(0, 0)]
    model.max_mode = 1  # the lowest band: no band continuation below it
    model.cfg = model.cfg_trial = None
    model.evaluate = lambda z, params, field, tight=False: (rows(z, True), (z, None, None, True))
    model.linearize = lambda z, aux, field, extra=None: (jac(z), None, None, (z, None, None, True))
    model.rows_at = lambda aux, field: rows(aux[0], True)
    model.repeat = (lambda z, params, field, aux: rows(z, True)) if repeat is None else repeat
    return model


def test_newton_convergence_test_is_a_step_in_metres():
    """Converged when the Newton step is within round-off of the coordinates (sqrt(eps) R0) or within the step of
    the measured row noise; the step's size is the RMS normal displacement; a singular Jacobian has no step."""
    eps = np.finfo(float).eps
    assert fvc.displacement_rms([3.0, 4.0], [(0, 0), (1, 0)]) == pytest.approx(np.sqrt(9.0 + 8.0))
    assert fvc.newton_step_converged(0.9 * np.sqrt(eps) * 2.0, 0.0, 2.0)
    assert not fvc.newton_step_converged(1.1 * np.sqrt(eps) * 2.0, 0.0, 2.0)
    assert fvc.newton_step_converged(1e-6, 2e-6, 2.0) and not fvc.newton_step_converged(3e-6, 2e-6, 2.0)
    assert fvc._newton_step(np.array([[1.0, 1.0], [1.0, 1.0 + 1e-17]]), np.ones(2)) is None
    assert fvc._newton_step(np.array([[np.nan]]), np.ones(1)) is None
    np.testing.assert_allclose(fvc._newton_step(np.array([[2.0, 0.0], [0.0, 1e-9]]), np.array([2.0, 1e-9])),
                               [-1.0, -1.0])


def test_normal_displacement_map_grid_is_refined_until_converged(monkeypatch):
    """The displacement map's grid doubles until the map converges (to sqrt(eps)); a fixed grid is taken as given;
    a map that does not converge in its grid is an error."""
    inp = _deck("input.LandremanPaul2021_QA_lowres", 3, 11)
    M, modes, _ = fvc._normal_displacement_map(inp, 2)
    fine, _, _ = fvc._normal_displacement_map(inp, 2, ntheta=256, nzeta=256)
    np.testing.assert_allclose(M, fine, rtol=0, atol=1e-7 * np.abs(fine).max())
    calls = []
    monkeypatch.setattr(fvc, "_normal_displacement_map_on", lambda *args: (
        calls.append(1) or (np.full((2, 2), (-1.0) ** len(calls)), modes, None)))
    with pytest.raises(ValueError, match="does not converge"):
        fvc._normal_displacement_map(inp, 2)


def test_nonzero_edge_pressure_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), pmass_type="power_series",
                  am=np.array([1.0e3, -0.5e3] + [0.0] * 19))
    with pytest.raises(ValueError, match="edge pressure"):
        fvc.solve_free_boundary_three_terms(inp, external_field=lambda xyz: xyz)
    # the model too (and so the single-stage problem built on it): its rows have no sheet current either
    with pytest.raises(ValueError, match="edge pressure"):
        fvc.ThreeTermsFreeBoundaryModel(inp)


def test_asymmetric_boundary_is_rejected():
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lasym=True)
    with pytest.raises(NotImplementedError, match="lasym"):
        fvc.solve_free_boundary_three_terms(inp, external_field=lambda xyz: xyz)


def test_unknown_boundary_condition_is_rejected():
    with pytest.raises(ValueError, match="boundary_condition"):
        vj.solve_free_boundary_multigrid(_deck("input.LandremanPaul2021_QA_lowres", 3, 11),
                                         external_field=lambda xyz: xyz, boundary_condition="sheet")


@needs_vc
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
    assert summary.normal_max == pytest.approx(np.max(np.abs(rows[0]))) and summary.normal_max >= summary.normal
    assert "pressure balance" in str(summary) and "sheet current" in str(summary)


def test_vacuum_boundary_residual_is_b_dot_n_alone():
    """In vacuum the only interface condition is B.n = 0 (the field outside is the coils'): one row,
    B_coil . n / |B_in| on the boundary, reported as its area-weighted RMS and maximum, no pressure or sheet terms."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), phiedge=-0.025, curtor=0.0, am=np.zeros(21),
                  lfreeb=False)
    eq = opt.solve_equilibrium(inp)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    rows, weights = fvc.boundary_residual(inp, eq.solution, field, runtime=eq.solver_context, nphi=16, ntheta=16)
    data = vc.surface_field_data_from_state(inp, eq.solution, runtime=eq.solver_context, nphi=16, ntheta=16)
    B_coil = np.asarray(vc.external_B_cartesian(field, jnp.asarray(data.gamma), jnp.asarray(data.phi)))
    expected = np.sum(B_coil * np.asarray(data.normal), axis=0) / np.linalg.norm(np.asarray(data.B_total), axis=0)
    assert rows.shape[0] == 1
    np.testing.assert_allclose(rows[0], expected, rtol=1e-10, atol=1e-14)
    summary = fvc.summarize_boundary_residual(rows, weights)
    assert summary.pressure is None and summary.sheet_current is None
    np.testing.assert_allclose(summary.normal, np.sqrt(np.sum(np.asarray(weights) * expected**2)), rtol=1e-12)
    assert summary.normal_max == pytest.approx(np.max(np.abs(expected)))
    assert str(summary).startswith("B.n/|B| RMS") and "pressure" not in str(summary)


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
    """The three conditions on ``eq``'s boundary, with the fixed singular quadrature (4 nfp n, 2 n) that the
    model's docstring recommends (planning it to 4 digits can exhaust a GPU on a finite-beta surface)."""
    data = vc.surface_field_data_from_state(inp, eq.solution, runtime=eq.solver_context, nphi=n, ntheta=n)
    precision = vc.plan_vc_precision(data, quad_nt=4 * int(inp.nfp) * n, quad_np=2 * n)
    return fvc.summarize_boundary_residual(*fvc.boundary_residual(
        inp, eq.solution, field, runtime=eq.solver_context, nphi=n, ntheta=n, precision=precision))


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
def test_vacuum_free_boundary_agrees_with_nestor():
    """In vacuum the free boundary is a flux surface of the coil field alone: both methods find it.

    The answer depends on the deck: its boundary is the reference of the poloidal labelling (module
    docstring), not only the start.  Here the deck is the one VMEC + NESTOR leaves; from it and from the
    original LP deck's boundary the solve lands on one boundary, with the interface conditions below NESTOR's.
    VMEC + NESTOR itself lands ~1.6 mm from the exact free boundary of an essentially exact field (VMEC2000
    1.3 mm), so the surfaces are compared at that level, and more sharply through the residuals and the
    rotational transform.  At mpol = 4 the labelling matters: with the original LP deck as the deck (its
    labelling truncated to 4 modes) B.n ends at 9.7e-4, again from either start; with NESTOR's at 5.4e-4
    (NESTOR's own: 5.9e-4).
    """
    lp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 25), phiedge=-0.025, curtor=0.0,
                 am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    nestor, nestor_deck, reported = _nestor(lp, field)
    inp = replace(lp, rbc=nestor_deck.rbc, zbs=nestor_deck.zbs, raxis_c=nestor_deck.raxis_c,
                  zaxis_s=nestor_deck.zaxis_s)
    before = _residual(inp, nestor, field)
    for name in ("normal", "normal_max"):   # the gate on the NESTOR result (vacuum: B.n alone)
        np.testing.assert_allclose(getattr(reported, name), getattr(before, name), rtol=0.25)
    # The net-current row's reference: outside the plasma the toroidal circulation is the coils' alone, so a
    # free boundary's edge R B_phi equals the coil current that _coil_net_current integrates on the axis.
    np.testing.assert_allclose(abs(fvc._coil_net_current(field, nestor.wout)), abs(float(nestor.wout.rbtor)),
                               rtol=1e-3)
    fits = [fvc.solve_free_boundary_three_terms(inp, external_field=field, initial_boundary=start)
            for start in (None, lp)]
    np.testing.assert_allclose(fits[1].x, fits[0].x, rtol=0, atol=1e-9)
    for fit in fits:
        after = _residual(inp, fit.equilibrium, field)
        assert fit.success and after.normal < before.normal and after.normal_max < before.normal_max
    walls = [fit.equilibrium.wout for fit in fits]
    for wall in walls:   # budget: NESTOR's own floor (~1.6 mm)
        assert _lcfs_distance(wall, nestor.wout) < 2e-3
    np.testing.assert_allclose(np.asarray(walls[0].iotaf)[[0, -1]], np.asarray(nestor.wout.iotaf)[[0, -1]], atol=3e-3)
    # With the LP deck itself as the deck (its labelling, truncated to 4 modes, as the reference) the answer is
    # another one (B.n 9.7e-4), but again the same from either start.
    lp_fits = [fvc.solve_free_boundary_three_terms(lp, external_field=field, initial_boundary=start)
               for start in (None, nestor_deck)]
    np.testing.assert_allclose(lp_fits[1].x, lp_fits[0].x, rtol=0, atol=1e-9)
    assert all(fit.success for fit in lp_fits)
    # a warm restart at an unchanged field stays where it is.
    fit = fits[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = fvc.solve_free_boundary_three_terms(inp, external_field=field, previous=fit)
    assert not any("did not lower B.n" in str(w.message) for w in caught)
    assert again.njev == 0 and again.cost <= fit.cost
    assert _lcfs_distance(again.equilibrium.wout, walls[0]) < 1e-4


_FINITE_BETA_LEAK = pytest.mark.xfail(strict=True, reason=(
    "known limitation: the finite-beta rows act on the boundary's modes (two below the interior's), and B.n "
    "outside them is not controlled (the solve warns); LP QA 0.5 % beta with its bundled coils at mpol 4 ends "
    "at B.n 6.0e-3 (deck 6.0e-3), at mpol 6 1.15e-3 (deck 1.17e-3). At mpol 4 most of it (4.3e-3) lies outside "
    "even the interior's modes: the mpol-4 truncation of the deck, whose B.n at 12 modes is 1.2e-3 (a 12-mode "
    "re-solve of the mpol-4 deck reads 4.8e-3); see docs/explanation/validation.md"))


@needs_vc
@pytest.mark.full
@_FINITE_BETA_LEAK
def test_finite_beta_free_boundary_removes_the_sheet_current():
    """LP QA at 0.5% beta with its fitted coils, through the solver's public option."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    design = _residual(inp, opt.solve_equilibrium(replace(inp, lfreeb=False)), field)
    result = vj.solve_free_boundary_multigrid(inp, external_field=field, boundary_condition="three_terms",
                                              three_terms_options=dict(quadrature=(4 * int(inp.nfp) * 48, 96)))
    assert result.converged and result.vacuum is None
    res = result.boundary_residual
    for name in ("normal", "pressure", "sheet_current"):
        assert getattr(res, name) < 0.25 * getattr(design, name), name
    assert res.sheet_current < 1.5e-3


@needs_vc
def test_vacuum_free_boundary_is_the_least_squares_boundary_from_any_start(monkeypatch):
    """Vacuum, LP QA at mpol 3: the rows are the h-gradient of the mean square of B_coil . n (exact Hessian as
    the Jacobian), Newton reaches them from the deck and from another start, and both land on one boundary."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    start = replace(inp, rbc=np.asarray(inp.rbc) * 1.01, zbs=np.asarray(inp.zbs) * 0.99)
    fits = [fvc.solve_free_boundary_three_terms(inp, external_field=field, nphi=16, ntheta=16, initial_boundary=b)
            for b in (None, start)]
    model = fits[0].model
    for fit in fits:
        assert fit.success and fit.cost < 1e-18
        assert fit.boundary_residual.normal < fit.initial_boundary_residual.normal
    # one boundary, to the solver's own step tolerance (newton_step_converged: sqrt(eps) R0 of normal displacement)
    gap = fvc.displacement_rms(fits[1].x - fits[0].x, model.boundary_modes)
    assert fvc.newton_step_converged(gap, 0.0, model.R0), gap
    for fit in fits:  # the reported equilibrium is the rows' state, re-solved tight on its own rung
        np.testing.assert_allclose(np.asarray(fit.equilibrium.solution.R_cos), np.asarray(fit.aux[0].R_cos),
                                   rtol=0, atol=1e-6)
    calls, solve = [], opt.solve_equilibrium  # ... seeded as a restart (the deck's lower rungs skipped)
    monkeypatch.setattr(opt, "solve_equilibrium", lambda *a, **k: (calls.append(k), solve(*a, **k))[1])
    fvc.solve_free_boundary_three_terms(inp, external_field=field, previous=fits[0])
    assert calls[-1]["restart_from"] is not None and "initial_state" not in calls[-1]
    monkeypatch.setattr(opt, "solve_equilibrium", solve)
    # the rows at the deck: R0 d/dh of the mean square of B.n / B0, against central differences of it
    rows = np.asarray(model._rows(model.seed[0], model.params0, field))
    nfp, ntor = int(inp.nfp), int(inp.ntor)
    n, m = np.meshgrid(np.arange(-ntor, ntor + 1) * nfp, np.arange(inp.mpol), indexing="ij")
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi / nfp, 16, endpoint=False),
                             np.linspace(0, 2 * np.pi, 16, endpoint=False), indexing="ij")

    def mean_square(x):
        p = model.with_boundary(model.params0, jnp.asarray(x))
        return float(fvc._mean_square_normal_field(jnp.ravel(p.rbc), jnp.ravel(p.zbs), jnp.asarray(m.ravel(), float),
                                                   jnp.asarray(n.ravel(), float), phi, theta, jnp.asarray(phi[:, 0]),
                                                   field))

    R0, eye = float(np.asarray(inp.rbc)[ntor, 0]), np.eye(model.x0.size)
    for k in (0, 3):
        fd = (mean_square(1e-6 * eye[k]) - mean_square(-1e-6 * eye[k])) / 2e-6 * R0 / (2 * model.B0**2)
        np.testing.assert_allclose(rows[k], fd, rtol=1e-5)
    # the Jacobian (exact Hessian through the implicit state tangents) against re-solved central differences
    J = model.linearize(fits[0].x, fits[0].aux, field)[0]
    k, h = 2, 1e-6
    fd = [model.evaluate(fits[0].x + s * h * eye[k], model.params0, field, tight=True)[0] for s in (1, -1)]
    np.testing.assert_allclose(J[:, k], (fd[0] - fd[1]) / (2 * h), rtol=1e-4, atol=1e-6 * np.abs(J[:, k]).max())
    # the rows' repeatability: the answer solved again from the deck's state, and the deck solved again cold, agree
    # to the rows' noise (dimensionless rows; measured ~1e-13 at the answer)
    for x, aux in ((fits[0].x, fits[0].aux), (model.x0, (model.seed[0], model.seed[1], model.params0, True))):
        rows, again = model.rows_at(aux, field), model.repeat(x, model.params0, field, aux)
        np.testing.assert_allclose(again, rows, rtol=0, atol=1e-10)
    with warnings.catch_warnings(record=True) as caught:  # a warm restart is measured against the deck too
        warnings.simplefilter("always")
        again = fvc.solve_free_boundary_three_terms(inp, external_field=field, previous=fits[0])
    assert again.njev == 0 and again.success
    assert not any("did not lower B.n" in str(w.message) for w in caught)
    assert 0.0 < fits[0].normal_outside_modes <= fits[0].boundary_residual.normal
    # a solve that does not lower B.n says so, with the part outside the boundary's modes
    monkeypatch.setattr(model, "boundary_residual", lambda *args: fvc.BoundaryResidual(1.0, 1.0))
    with pytest.warns(RuntimeWarning, match="outside the boundary's"):
        fvc.solve_free_boundary_three_terms(inp, external_field=field, previous=fits[0])


def test_square_galerkin_boundary_is_solved_by_newton_from_a_stale_jacobian():
    """Square rows (the Galerkin free boundary): Newton reaches the root to round-off, reusing a stale Jacobian
    while it converges fast and refreshing it otherwise; the root does not depend on the start."""

    def rows(x, tight=True):
        return np.array([x[0] ** 2 + x[1] - 3.0, x[0] - x[1] ** 3 + 7.0])

    def jac(x):
        return np.array([[2.0 * x[0], 1.0], [1.0, -3.0 * x[1] ** 2]])

    model = _stand_in(2, rows, jac)
    root = np.array([1.0, 2.0])
    for start, jacobian in ((np.array([1.3, 1.7]), None), (np.array([0.8, 2.2]), jac(np.array([1.5, 1.5]))),
                            (np.array([0.9, 2.1]), -jac(np.array([1.0, 2.0])))):  # a wrong one: refreshed
        out = model.solve_boundary(None, None, x0=start, jacobian=jacobian)
        assert out["converged"]
        np.testing.assert_allclose(out["x"], root, atol=1e-12)
        assert out["nfev"] <= 12


def test_newton_stops_at_the_noise_of_the_rows():
    """Rows with evaluation noise (as virtual-casing rows have): below round-off of the coordinates Newton stops;
    where the noise is larger and a fresh step no longer descends, the boundary is solved again from a second seed,
    and the step is converged if it is within the step of that difference (the rows' noise), a stall otherwise."""

    def draw(x):  # a new draw per point
        return np.random.default_rng(np.frombuffer(np.asarray(x, dtype=float).tobytes(), dtype=np.uint32)).normal(size=2)

    def exact(x):
        return np.array([x[0] ** 2 + x[1] - 3.0, x[0] - x[1] ** 3 + 7.0])

    def jac(x):
        return np.array([[2.0 * x[0], 1.0], [1.0, -3.0 * x[1] ** 2]])

    for noise in (1e-11, 1e-6):
        model = _stand_in(2, lambda x, tight=True, noise=noise: exact(x) + noise * draw(x), jac,
                          repeat=lambda x, params, field, aux, noise=noise: exact(x) - 10.0 * noise * draw(x))
        for jacobian in (None, jac(np.array([1.0, 2.0]))):  # fresh Jacobians; a reused one at the root
            start = np.array([1.3, 1.7]) if jacobian is None else np.array([1.0, 2.0])
            out = model.solve_boundary(None, None, x0=start, jacobian=jacobian)
            assert out["converged"]
            np.testing.assert_allclose(out["x"], [1.0, 2.0], atol=1e-9 if noise < 1e-8 else 1e-5)
            assert out["njev"] <= 6
            # the noise is measured only where it limits the step, then kept for the model's later solves
            assert (out["noise"] is None) == (noise < 1e-8)
    # a repeat that lands on the same rows sees no noise; the rows' deviation from the Jacobian's linear model,
    # which does not fall with the step's length, does, and the step is judged against it
    model = _stand_in(2, lambda x, tight=True: exact(x) + 1e-6 * draw(x), jac)
    out = model.solve_boundary(None, None, x0=np.array([1.3, 1.7]))
    assert out["converged"] and out["noise"] > 0.0
    np.testing.assert_allclose(out["x"], [1.0, 2.0], atol=1e-5)


def test_unknown_problem_boundary_condition_is_rejected():
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    with pytest.raises(ValueError, match="boundary_condition"):
        FreeBoundaryProblem.from_loss(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), lambda s, r, x: 0.0,
                                      np.zeros(1), field_from_parameters=lambda x: None, boundary_condition="sheet")


def test_problem_boundary_condition_dispatches_to_three_terms(monkeypatch):
    """``boundary_condition="three_terms"`` hands the loss, maps, quantities and options to the three-terms problem."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    calls = []
    monkeypatch.setattr(fvc.ThreeTermsFreeBoundaryProblem, "from_loss",
                        classmethod(lambda cls, *args, **kwargs: calls.append((args, kwargs)) or "problem"))
    inp, loss, field = _deck("input.LandremanPaul2021_QA_lowres", 3, 11), (lambda s, r, x: 0.0), (lambda x: None)
    got = FreeBoundaryProblem.from_loss(inp, loss, np.zeros(1), field_from_parameters=field, quantities=(len,),
                                        boundary_condition="three_terms", three_terms_options=dict(nphi=16))
    (args, kwargs), = calls
    assert got == "problem" and args[:2] == (inp, loss) and kwargs["field_from_parameters"] is field
    assert kwargs["quantities"] == (len,) and kwargs["nphi"] == 16


def test_problem_method_selects_the_free_boundary(monkeypatch):
    """``method=`` names the free boundary in one word: Galerkin and least squares are the three-terms problems
    (their keywords in ``three_terms_options``), NESTOR the VMEC one; a contradicting ``boundary_condition`` or
    ``three_terms_options`` method is refused, as is an unknown method."""
    from vmex.core import freeboundary_lsq as fls
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    calls = []
    for cls in (fvc.ThreeTermsFreeBoundaryProblem, fls.LeastSquaresFreeBoundaryProblem):
        monkeypatch.setattr(cls, "from_loss", classmethod(lambda cls, *args, **kwargs: calls.append(cls) or cls))
    inp, loss, field = _deck("input.LandremanPaul2021_QA_lowres", 3, 11), (lambda s, r, x: 0.0), (lambda x: None)

    def build(deck=inp, **kwargs):
        return FreeBoundaryProblem.from_loss(deck, loss, np.zeros(1), field_from_parameters=field, **kwargs)

    assert build(method="galerkin", three_terms_options=dict(nphi=16)) is fvc.ThreeTermsFreeBoundaryProblem
    assert build(method="least_squares", boundary_condition="three_terms") is fls.LeastSquaresFreeBoundaryProblem
    with pytest.raises(ValueError, match="input must enable free-boundary"):  # NESTOR's own checks: its route
        build(replace(inp, lfreeb=False), method="nestor")
    with pytest.raises(ValueError, match="solves boundary_condition='three_terms'"):
        build(method="galerkin", boundary_condition="nestor")
    with pytest.raises(ValueError, match="disagrees"):
        build(method="galerkin", three_terms_options=dict(method="least_squares"))
    with pytest.raises(ValueError, match="method must be"):
        build(method="vmec")
    assert calls == [fvc.ThreeTermsFreeBoundaryProblem, fls.LeastSquaresFreeBoundaryProblem]


@needs_vc
def test_three_terms_problem_seed_starts_at_boundary_start(monkeypatch):
    """``boundary_start`` (a restart's saved boundary) is where the seed's solve starts on the deck: the start is
    the deck displacement of that surface, and from the deck's own free boundary the solve starts near the root
    and lands on the same one as the cold solve."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    starts, solve = [], fvc.solve_band_continuation

    def recording(model, params, field, *, x0=None, **kwargs):
        starts.append(x0)
        return solve(model, params, field, x0=x0, **kwargs)

    monkeypatch.setattr(fvc, "solve_band_continuation", recording)

    def build(boundary_start=None):
        return FreeBoundaryProblem.from_loss(
            inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(1),
            field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[0])),
            boundary_condition="three_terms", three_terms_options=dict(nphi=16, ntheta=16, quadrature=(4 * 2 * 16, 32),
                                                       boundary_max_nfev=80, boundary_start=boundary_start))

    cold = build()
    seed = cold._record(np.zeros(1))  # the seed's free boundary, solved by from_loss
    boundary = seed["aux"][2]  # its rbc, zbs on the deck
    warm = build(types.SimpleNamespace(rbc=np.asarray(boundary.rbc), zbs=np.asarray(boundary.zbs)))
    assert starts[0] is None and starts[1] is not None and np.any(starts[1])
    nb = cold.model.n_boundary
    # displacement_of measures along the surface normal, the coordinates along the deck's radial lines: the start
    # is near the root, not on it
    assert np.linalg.norm(starts[1][:nb] - seed["x"][:nb]) < 0.5 * np.linalg.norm(seed["x"][:nb])
    np.testing.assert_allclose(warm.fun(np.zeros(1)), cold.fun(np.zeros(1)), rtol=1e-8)
    # the saved coordinates themselves (boundary_coordinates, the run's own deck): the start is the root
    exact = build(cold.boundary_coordinates(np.zeros(1)))
    np.testing.assert_array_equal(starts[2], seed["x"])
    assert exact._record(np.zeros(1))["nfev"] <= warm._record(np.zeros(1))["nfev"]
    np.testing.assert_allclose(exact.fun(np.zeros(1)), cold.fun(np.zeros(1)), rtol=1e-10)


@needs_vc
def test_three_terms_problem_gradients_match_resolved_differences():
    """Design gradients by the implicit function theorem of the boundary fit, against re-solved central
    differences, along a plasma coordinate (PHIEDGE) and a field coordinate (the coil currents)."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2),
        field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[1])),
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        quantities=(opt.major_radius,),
        parameter_quantities=(lambda state, runtime, x: opt.aspect_ratio(state, runtime) * (1 + x[1]),),
        boundary_condition="three_terms",
        three_terms_options=dict(nphi=16, ntheta=16, quadrature=(4 * 2 * 16, 32),
                                boundary_max_nfev=80))
    assert isinstance(problem, fvc.ThreeTermsFreeBoundaryProblem)
    x0 = problem.x0
    value, grad = problem.value_and_grad(x0)
    jac = problem.constraint_jac(x0)
    assert problem.boundary_residual(x0).normal < 1e-2 and problem.boundary_residual(x0).sheet_current is None
    for k, h in ((0, 2e-3), (1, 2e-3)):
        e = h * np.eye(2)[k]
        values = [np.r_[problem.fun(x0 + s * e), problem.constraint_values(x0 + s * e)] for s in (1, -1)]
        fd = (values[0] - values[1]) / (2 * h)
        np.testing.assert_allclose(np.r_[grad[k], jac[:, k]], fd, rtol=2e-2, atol=1e-8 * max(1.0, abs(value)))
    problem.accept_x(x0 + 2e-3 * np.eye(2)[1])
    assert problem.accepted_step == 1
    # a boundary fit that did not converge is a failed trial, never a value (whatever its residual)
    model = problem.model
    solve, newton = model.solve_boundary, model._newton_boundary
    model.solve_boundary = lambda *args, **kwargs: dict(solve(*args, **kwargs), converged=False)
    model._newton_boundary = lambda *args, **kwargs: dict(newton(*args, **kwargs), converged=False)
    rereferenced = model.rereferenced

    def no_deck(b):
        raise VmecError("no other deck")

    model.rereferenced = no_deck
    failed = problem.stats["failed"]
    with pytest.raises(TrialRejected):
        problem.fun(x0 + 1.5e-3 * np.eye(2)[0])
    assert problem.stats["failed"] == failed + 1
    # warm starts that need damping hand over to the cold solve from the deck, which finds the boundary
    routes = []

    def warm_damped(*args, **kwargs):
        if kwargs.get("jacobian") is None:  # the cold solve's own Newton
            return newton(*args, **kwargs)
        routes.append("warm")
        return dict(newton(*args, **kwargs), converged=False, damped=True)

    def cold(*args, **kwargs):
        routes.append("cold" if kwargs.get("jacobian") is None else "polish")
        return solve(*args, **kwargs)

    model._newton_boundary, model.solve_boundary = warm_damped, cold
    problem.fun(x0 + 2e-3 * np.eye(2)[0])
    assert routes == ["warm", "warm", "cold"]
    model._newton_boundary = newton
    # the trial's fit converges, the gradient's polish of it does not: no gradient either
    model.solve_boundary = lambda *args, **kwargs: dict(solve(*args, **kwargs), converged=False)
    with pytest.raises(TrialRejected, match="polish"):
        problem.value_and_grad(x0 + 2.5e-3 * np.eye(2)[0])
    model.solve_boundary = solve
    # a design the deck reaches from no start: its trial is solved on the anchor's boundary as a new deck, and
    # the new deck's values and gradients are the free boundary's (the same up to the theta labelling)
    xt = x0 + 1e-3 * np.eye(2)[1]
    before = np.r_[problem.fun(xt), problem.constraint_values(xt)]
    problem._cache.pop(xt.tobytes())
    model.rereferenced = rereferenced
    model.solve_boundary = lambda *args, **kwargs: dict(solve(*args, **kwargs), converged=False)
    model._newton_boundary = lambda *args, **kwargs: dict(newton(*args, **kwargs), converged=False)
    value, grad = problem.value_and_grad(xt)
    assert problem.model is not model and problem.stats["rereferences"] == 1
    assert [r["step"] for r in problem.references] == [0, 1] and problem.references[1]["deck"] is problem.model.fixed
    # the deck's band of displacements is another truncation of the boundary: the same free boundary up to it
    # (mpol 3: 1e-5, the major radius 11 um)
    np.testing.assert_allclose(np.r_[value, problem.constraint_values(xt)], before, rtol=5e-5)
    jac = problem.constraint_jac(xt)
    for k, h in ((0, 2e-3), (1, 2e-3)):
        e = h * np.eye(2)[k]
        values = [np.r_[problem.fun(xt + s * e), problem.constraint_values(xt + s * e)] for s in (1, -1)]
        fd = (values[0] - values[1]) / (2 * h)
        np.testing.assert_allclose(np.r_[grad[k], jac[:, k]], fd, rtol=2e-2, atol=1e-8 * max(1.0, abs(value)))
    assert problem.stats["rereferences"] == 1  # the new deck reaches its neighbours


@needs_vc
@pytest.mark.parametrize("method", ["galerkin", "least_squares"])
def test_normal_outside_modes_quantity_value_and_gradient(method):
    """``NORMAL_OUTSIDE_MODES`` as a parameter quantity: its value is the model's ``normal_outside_modes`` of the
    trial and its design gradient (direct field term plus the moving boundary) matches re-solved differences."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    field = lambda x: replace(coils, currents=coils.currents * (1 + x[1]))  # noqa: E731
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(2), field_from_parameters=field,
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        parameter_quantities=(fvc.NORMAL_OUTSIDE_MODES,), boundary_condition="three_terms", method=method,
        three_terms_options=dict(nphi=16, ntheta=16, quadrature=(4 * 2 * 16, 32), boundary_max_nfev=80))
    x0 = problem.x0
    value = problem.constraint_values(x0)
    sol = problem._record(x0)
    np.testing.assert_allclose(value, [problem.model.normal_outside_modes(sol["state"], sol["params"], field(x0))],
                               rtol=1e-10)
    assert 0 < value[0] < 1e-2
    jac = problem.constraint_jac(x0)
    # h = 4e-3: below ~2e-3 a least-squares trial stops at its Gauss-Newton prediction (the step left is under the
    # rows' noise), so its differences give the Gauss-Newton derivative (5 % off here); Galerkin agrees at any h
    for k, h in ((0, 4e-3), (1, 4e-3)):
        e = h * np.eye(2)[k]
        fd = (problem.constraint_values(x0 + e) - problem.constraint_values(x0 - e)) / (2 * h)
        np.testing.assert_allclose(jac[:, k], fd, rtol=3e-2, atol=1e-9)
    with pytest.raises(TypeError):
        fvc.NORMAL_OUTSIDE_MODES(None, None, None)


def test_three_terms_problem_plasma_map_gradient_at_moved_point():
    """A plasma map that scales its input (PHIEDGE * (1 + x)) is differentiated at the deck's parameters: the
    design gradient at a moved point x = 0.05 matches re-solved central differences (differentiating the map at
    the trial's own, already scaled, parameters gave exactly 1.05 times the differences)."""
    from vmex.core.freeboundary_problem import FreeBoundaryProblem

    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    problem = FreeBoundaryProblem.from_loss(
        inp, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(1),
        field_from_parameters=lambda x: coils,
        plasma_from_parameters=lambda params, x: replace(params, phiedge=params.phiedge * (1 + x[0])),
        quantities=(opt.major_radius,), boundary_condition="three_terms",
        three_terms_options=dict(nphi=16, ntheta=16, quadrature=(4 * 2 * 16, 32), boundary_max_nfev=80))
    x1, h = np.array([0.05]), 2e-3
    value, grad = problem.value_and_grad(x1)
    jac = problem.constraint_jac(x1)
    values = [np.r_[problem.fun(x1 + s * h), problem.constraint_values(x1 + s * h)] for s in (1, -1)]
    fd = (values[0] - values[1]) / (2 * h)
    np.testing.assert_allclose(np.r_[grad[0], jac[:, 0]], fd, rtol=2e-2, atol=1e-8 * max(1.0, abs(value)))


def _lp_beta0p5_profiles(inp):
    """Landreman--Buller--Drevlak kinetic profiles whose 2 e n T is the LP QA 0.5% deck's pressure."""
    from vmex.core import bootstrap as bs

    n0 = 1.5e20
    T0 = float(inp.am[0]) / (2 * bs.ELEMENTARY_CHARGE * n0)
    return bs.KineticProfiles(ne_coeffs=n0 * np.array([1, 0, 0, 0, 0, -1.0]), Te_coeffs=T0 * np.array([1, -1.0]),
                              Ti_coeffs=T0 * np.array([1, -1.0]))


def test_bootstrap_current_profile_layout():
    """A knot per half-mesh surface and one on the axis, the first segment with the deck's I(s_1), CURTOR the
    profile's integral."""
    from vmex.core.profiles import current

    ns = 25  # the deck's current is resolved to 2 % by the line segments from here
    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, ns), lfreeb=False)
    from vmex.core.bootstrap import HalfMeshCurrent

    block = HalfMeshCurrent(inp, _lp_beta0p5_profiles(inp), 0)
    s = (np.arange(1, ns) - 0.5) / (ns - 1)
    np.testing.assert_allclose(block.knots[1:-1], s)
    values = np.asarray(block.values(block.x0))
    assert block.knots[0] == 0.0 and block.deck.pcurr_type == "line_segment_ip"
    np.testing.assert_array_equal(block.deck.ac_aux_s, block.knots)
    np.testing.assert_allclose(block.deck.ac_aux_f, values)
    assert block.x0.size == ns
    assert block.deck.curtor == pytest.approx(float(current("line_segment_ip", inp.ac, block.knots, values, 1.0)))
    # the deck's own enclosed current away from the axis, where only the first cell differs
    deck_I = lambda d, x: float(d.curtor) * np.asarray(current(d.pcurr_type, d.ac, d.ac_aux_s, d.ac_aux_f, x)) / float(  # noqa: E731
        current(d.pcurr_type, d.ac, d.ac_aux_s, d.ac_aux_f, 1.0))
    outer = s[s > 0.2]
    np.testing.assert_allclose(deck_I(block.deck, outer), deck_I(inp, outer), rtol=2e-2)
    assert deck_I(block.deck, s[0]) == pytest.approx(deck_I(inp, s[0]), rel=1e-3)  # the first segment's integral


def test_bootstrap_rows_are_the_current_errors_field_over_b0():
    """The rows are sqrt(w_j) (I' - I'_Redl) / unit with w_j = d CURTOR / d v_j and unit = L B0 / mu0: their
    squared norm is the integral over s of the error's poloidal field over B0, at any ns; no weight to tune."""
    import jax
    from vmex.core.bootstrap import HalfMeshCurrent
    from vmex.core.profiles import MU0

    for ns in (13, 25):
        inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, ns), lfreeb=False)
        block = HalfMeshCurrent(inp, _lp_beta0p5_profiles(inp), 0)
        curtor = jax.grad(lambda v: block.current(block.values(v), 1.0))(jnp.asarray(block.x0))
        np.testing.assert_allclose(block.weights, curtor, rtol=1e-12, atol=1e-15)
        assert block.weights.sum() == pytest.approx(1.0) and np.all(block.weights > 0)
        # the edge value continues the last two knots' line: a linear I' is exact at s = 1
        assert float(block.values(1.0 - block.knots[1:-1])[-1]) == pytest.approx(0.0, abs=1e-14)
        with pytest.raises(ValueError, match="unit is not set"):
            block.x_scale
        with pytest.raises(ValueError, match="unit is not set"):
            block.rows(None, None, None)
        with pytest.raises(ValueError, match="positive and finite"):
            block.set_unit(0.0, 1.0)
        block.set_unit(0.5, 2.0)
        assert block.unit == pytest.approx(1.0 / MU0)
        np.testing.assert_allclose(block.x_scale, 1.0 / MU0)
        error = 1e5 * np.ones(block.x0.size)  # a uniform error of 1e5 A per unit s
        block.mismatch = lambda *_: (jnp.asarray(error), None)
        rows = np.asarray(block.rows(None, None, None))
        assert np.sum(rows**2) == pytest.approx((1e5 * MU0) ** 2, rel=1e-12)


def test_bootstrap_current_has_a_knot_per_surface_beyond_ndatafmax(tmp_path):
    """ns = 101 (the interior the finite-beta interface terms need): 102 knots, more than VMEC2000's 101; the
    profile keeps them all and only an INDATA namelist of it is refit to 101."""
    from vmex.core.bootstrap import HalfMeshCurrent

    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 101), lfreeb=False)
    block = HalfMeshCurrent(inp, _lp_beta0p5_profiles(inp), 0)
    assert block.knots.size == 101 + 1 and block.x0.size == 101
    np.testing.assert_array_equal(block.deck.ac_aux_s, block.knots)
    with pytest.warns(RuntimeWarning, match="refit to VMEC2000's 101"):
        reread = vj.VmecInput.from_file(block.deck.to_indata(tmp_path / "input.bootstrap"))
    assert reread.ac_aux_s.size == 101 and reread.ac_aux_s[-1] == 1.0


@needs_vc
@pytest.mark.full
@_FINITE_BETA_LEAK
def test_bootstrap_current_solved_with_the_free_boundary():
    """LP QA at 0.5% beta: boundary and Redl current in one solve, with exact columns for both."""
    from vmex.core import bootstrap as bs

    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 4, 25)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    profiles = _lp_beta0p5_profiles(inp)
    fit = fvc.solve_free_boundary_three_terms(inp, external_field=field, bootstrap=profiles, bootstrap_helicity=0,
                                              quadrature=(4 * int(inp.nfp) * 48, 96))
    model = fit.model
    state, mask, params, _ = fit.aux
    assert fit.x.size == model.n_boundary + 25
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
    for k in (0, model.n_boundary + 1, model.n_boundary + 13):
        h = 1e-5 * model.x_scale[k]
        rows = [model.evaluate(fit.x + sign * h * np.eye(fit.x.size)[k], params, field, tight=True)[0]
                for sign in (1, -1)]
        fd = (rows[0] - rows[1]) / (2 * h)
        assert np.linalg.norm(J[:, k] - fd) < 1e-4 * np.linalg.norm(fd), k


def _fixed_boundary_bootstrap():
    """LP QA 0.5 % at a fixed boundary: the Gauss-Newton bootstrap current (``fixed_boundary=True``) solved."""
    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 11), lfreeb=False,
                  ftol_array=np.array([1e-13]), niter_array=np.array([20000]))
    profiles = _lp_beta0p5_profiles(inp)
    model = fvc.ThreeTermsFreeBoundaryModel(inp, bootstrap=profiles, bootstrap_helicity=0, fixed_boundary=True)
    return inp, profiles, model, model.solve_boundary(model.params0, None)


def test_fixed_boundary_bootstrap_model_has_exact_columns():
    """fixed_boundary=True: the Gauss-Newton current at a fixed boundary is Redl's, with exact Jacobian columns."""
    inp, profiles, model, out = _fixed_boundary_bootstrap()
    assert model.n_boundary == 0 and model.x0.size == 1 + 10  # the axis knot, then the half mesh
    state, _, params, _ = out["aux"]
    assert model.bootstrap_residual(state, params) < 1e-4
    J = model.linearize(out["x"], out["aux"], None)[0]
    k, h = 3, 1e-5 * model.x_scale[3]
    rows = [model.evaluate(out["x"] + sign * h * np.eye(out["x"].size)[k], params, None, tight=True)[0]
            for sign in (1, -1)]
    fd = (rows[0] - rows[1]) / (2 * h)
    assert np.linalg.norm(J[:, k] - fd) < 1e-5 * np.linalg.norm(fd)


@pytest.mark.full
def test_fixed_boundary_bootstrap_model_at_ns_101_has_exact_knot_columns():
    """ns = 101 with the bootstrap current in the solve (101 knot unknowns, beyond VMEC2000's ndatafmax): Redl's
    current, and Jacobian columns of the axis, a mid-radius and the last knot against re-solved differences."""
    inp = replace(_deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 101), lfreeb=False,
                  ftol_array=np.array([1e-13]), niter_array=np.array([40000]))
    model = fvc.ThreeTermsFreeBoundaryModel(inp, bootstrap=_lp_beta0p5_profiles(inp), bootstrap_helicity=0,
                                            fixed_boundary=True)
    out = model.solve_boundary(model.params0, None)
    assert out["converged"] and model.x0.size == 101
    state, _, params, _ = out["aux"]
    assert model.bootstrap_residual(state, params) < 1e-4
    J = model.linearize(out["x"], out["aux"], None)[0]
    for k in (0, 51, model.x0.size - 1):
        h = 1e-5 * model.x_scale[k]
        rows = [model.evaluate(out["x"] + sign * h * np.eye(out["x"].size)[k], params, None, tight=True)[0]
                for sign in (1, -1)]
        fd = (rows[0] - rows[1]) / (2 * h)
        assert np.linalg.norm(J[:, k] - fd) < 1e-5 * np.linalg.norm(fd), k


@pytest.mark.full
def test_fixed_boundary_bootstrap_model_matches_picard():
    """The Gauss-Newton current at a fixed boundary is the one the Picard loop converges to."""
    from vmex.core import bootstrap as bs

    inp, profiles, model, out = _fixed_boundary_bootstrap()
    params = out["aux"][2]
    picard = bs.self_consistent_bootstrap(inp, profiles, 0, n_iter=40, tol=1e-6, relax=0.5, profile="half_mesh")
    np.testing.assert_allclose(np.asarray(params.ac_aux_f), picard.input.ac_aux_f,
                               rtol=0, atol=2e-4 * np.max(np.abs(picard.input.ac_aux_f)))


def test_newton_solves_the_trials_tight_once_their_tolerance_is_the_noise():
    """Trials at a looser force tolerance (``trial_ftol``) carry noise.  Convergence is tested on rows solved to the
    deck's tolerance only, and when a fresh Newton step no longer descends on the trials they are solved tight."""

    def rows(x, tight):
        exact = np.array([x[0] ** 2 + x[1] - 3.0, x[0] - x[1] ** 3 + 7.0])
        noise = np.random.default_rng(np.frombuffer(np.asarray(x, dtype=float).tobytes(), dtype=np.uint32)).normal(size=2)  # a new draw per point
        return exact if tight else exact + 1e-6 * noise

    def jac(x):
        return np.array([[2.0 * x[0], 1.0], [1.0, -3.0 * x[1] ** 2]])

    model = _stand_in(2, rows, jac)
    model.cfg, model.cfg_trial = "deck", "trial"
    model.evaluate = lambda z, params, field, tight=False: (rows(z, tight), (z, None, None, tight))
    out = model.solve_boundary(None, None, x0=np.array([1.3, 1.7]))
    assert out["converged"] and out["aux"][3]
    np.testing.assert_allclose(out["x"], [1.0, 2.0], atol=1e-12)
    # loose rows that are already within round-off of a root are re-solved tight before convergence is declared
    loose = []
    model.evaluate = lambda z, params, field, tight=False: (loose.append(not tight) or rows(z, True) + (
        0.0 if tight else 1e-15), (z, None, None, tight))
    out = model.solve_boundary(None, None, x0=np.array([1.0, 2.0]), jacobian=jac(np.array([1.0, 2.0])))
    assert out["converged"] and out["aux"][3] and out["njev"] == 0 and loose[0] and not loose[-1]
    # trial noise above the rows near the root: no fresh step descends on loose trials, so they are solved tight
    model.evaluate = lambda z, params, field, tight=False: (rows(z, True) + (0.0 if tight else 1e-3), (
        z, None, None, tight))
    out = model.solve_boundary(None, None, x0=np.array([1.0 + 1e-5, 2.0]))
    assert out["converged"] and out["aux"][3]
    np.testing.assert_allclose(out["x"], [1.0, 2.0], atol=1e-12)


def test_newton_backs_off_a_start_without_equilibrium_and_reports_a_stall(capsys):
    """A start with no certified equilibrium backs off toward the deck; rows without a root stall at their
    stationary point (a singular Jacobian there), reported as not converged (and printed with ``verbose``)."""

    def rows(x, tight=True):
        return np.array([x[0] ** 2 + 1.0])  # no root: the cost is stationary at x = 0

    model = _stand_in(1, rows, lambda z: np.array([[2.0 * z[0]]]))
    model.evaluate = lambda z, params, field, tight=False: (
        None if abs(z[0]) > 1.0 else (rows(z), (z, None, None, True)))
    out = model.solve_boundary(None, None, x0=np.array([1.2]), verbose=1)  # 1.2 fails, 0.9 = 0.75 x 1.2 is solved
    assert not out["converged"] and abs(out["x"][0]) < 0.9
    assert "Newton stalled" in capsys.readouterr().out
    # a singular Jacobian (given, then fresh) has no Newton step: refreshed, then a stall
    out = model.solve_boundary(None, None, x0=np.array([0.0]), jacobian=np.zeros((1, 1)), verbose=1)
    assert not out["converged"] and out["njev"] <= 2 and "singular Jacobian" in capsys.readouterr().out
    model.evaluate = lambda z, params, field, tight=False: None
    with pytest.raises(RuntimeError, match="no certified equilibrium"):
        model.solve_boundary(None, None, x0=np.array([1.2]))


def test_fixed_boundary_model_needs_the_bootstrap_current():
    with pytest.raises(ValueError, match="bootstrap"):
        fvc.ThreeTermsFreeBoundaryModel(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), fixed_boundary=True)


def test_coil_net_current_is_the_circulation_around_the_axis():
    """G = (1/2 pi) times the circulation of the field around the magnetic axis: R B_phi for a 1/R field."""
    from types import SimpleNamespace

    @dataclass(frozen=True)
    class Toroidal:
        b0r0: float

        def b_cyl(self, r, phi, z):
            return 0.0 * r, self.b0r0 / r, 0.0 * r

    wout = SimpleNamespace(nfp=2, raxis_cc=np.array([1.0, 0.05]), zaxis_cs=np.array([0.0, 0.03]))
    assert fvc._coil_net_current(Toroidal(1.7), wout) == pytest.approx(1.7, rel=1e-10)


@needs_vc
def test_finite_beta_rows_test_b_dot_n_against_the_decks_response():
    """Finite beta (on band 1 here, for speed; the default is the interior's band): the rows are R0 T0^T r then the
    net-current row, with r = sqrt(w) B_out . n / |B_in| (coils + virtual casing) on the grid and T0 the response
    of the deck's r to each boundary coordinate (here three columns, by central differences of re-solved
    equilibria; on a GPU also the guard against miscompiled column maps).  The rows'
    Jacobian (state-only rows: all through the implicit state responses) and their derivative along the coil
    currents (through r and T0) match central differences."""
    inp = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 11)
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    nfp, n = int(inp.nfp), 16
    model = fvc.ThreeTermsFreeBoundaryModel(inp, nphi=n, ntheta=n, quadrature=(4 * nfp * n, 2 * n), max_mode=1)
    assert not model.vacuum and model.max_mode == 1 and model.top_band == max(inp.mpol - 1, inp.ntor)
    state, mask = model.seed
    x0, nb, step = np.asarray(model.x0), model.n_boundary, 1e-6

    def residual(state, params, f):
        runtime = model._im.runtime_from_params(params, model.cfg)
        data = vc.surface_field_data_from_state(model.fixed, state, runtime=runtime, nphi=n, ntheta=n)
        B_out = np.asarray(vc.PlasmaVacuumInterface.from_surface_data(
            data, p_edge=0.0, precision=model.precision).total_B_out(f))
        B_in, area = np.asarray(data.B_total), np.linalg.norm(np.asarray(data.area_vector), axis=0)
        return (np.sqrt(area / area.sum()) * np.sum(B_out * np.asarray(data.normal), 0)
                / np.linalg.norm(B_in, axis=0)).ravel()

    with pytest.raises(ValueError, match="deck's response"):  # a bare field has no test functions at finite beta
        model.rows(state, model.params0, field)
    rows = np.asarray(model._rows(state, model.params0, model.attach(field)))
    J = np.asarray(model.linearize(x0, (state, mask, model.params0, True), field)[0])
    r0 = residual(state, model.params0, field)
    R0 = float(np.asarray(model.fixed.rbc)[int(inp.ntor), 0])
    for k in (0, nb // 2, nb - 1):  # a low and a middle mode, and (0, 0) (no test function, a Jacobian column)
        pair = []
        for sign in (1.0, -1.0):
            params = model.with_boundary(model.params0, jnp.asarray(x0 + sign * step * np.eye(x0.size)[k]))
            solved = model.solve(params, seed=state)[0]
            pair.append((residual(solved, params, field), np.asarray(model._rows(solved, params, model.attach(field)))))
        T0_k = (pair[0][0] - pair[1][0]) / (2 * step)  # the deck response's column k
        if k < nb - 1:
            np.testing.assert_allclose(rows[k], R0 * T0_k @ r0, rtol=1e-5)
        central = (pair[0][1] - pair[1][1]) / (2 * step)
        np.testing.assert_allclose(J[:, k], central, rtol=1e-5, atol=1e-6 * np.abs(J).max())

    def with_currents(scale):
        return replace(field, currents=field.currents * scale)

    tangent = replace(jax.tree.map(jnp.zeros_like, field), currents=field.currents)
    along = np.asarray(jax.jvp(lambda f: model.rows(state, model.params0, model.attach(f)), (field,), (tangent,))[1])
    central = (np.asarray(model._rows(state, model.params0, model.attach(with_currents(1 + step))))
               - np.asarray(model._rows(state, model.params0, model.attach(with_currents(1 - step))))) / (2 * step)
    np.testing.assert_allclose(along, central, rtol=1e-6, atol=1e-9 * np.abs(along).max())


def test_newton_ends_not_converged_at_a_minimum_that_is_no_root():
    """``z^3 - 2 z + 2`` from ``z = 0``: Newton cycles (0 -> 1 -> 0) and ``|rows|^2`` has a local minimum at
    ``z = sqrt(2/3)`` where the row is 0.91, not zero.  The damped Newton steps end there as a stall
    (``converged=False``, the row not zero): a minimum of ``|rows|^2`` is never returned as the answer.
    From inside the basin of the root (``z = -1.6``, a warm Jacobian) the same solve converges to the root."""

    def rows(z):
        return np.array([z[0] ** 3 - 2.0 * z[0] + 2.0])

    def jac(z):
        return np.array([[3.0 * z[0] ** 2 - 2.0]])

    model = _stand_in(1, lambda z, tight=True: rows(z), jac, vacuum=False)
    out = model.solve_boundary(None, None, x0=np.array([0.0]), verbose=1)
    assert not out["converged"] and abs(float(out["rows"][0])) > 0.5 and out["band"] is None
    warm = model.solve_boundary(None, None, x0=np.array([-1.6]), jacobian=jac(np.array([-1.6])))
    assert warm["converged"]
    np.testing.assert_allclose(warm["x"][0], -1.7692923542386314, atol=1e-7)  # the root, to the step tolerance


class _Program:
    """Stand-in jitted program: records its AOT lowering, or fails it."""

    def __init__(self, fail=False):
        self.fail, self.lowered = fail, []

    def lower(self, *args):
        if self.fail:
            raise RuntimeError("no program")
        self.lowered.append(args)
        return self

    def compile(self):
        return self


class _Cotangents(_Program):
    """Stand-in cotangent program: ``rows`` rows of a (state, params) cotangent pair."""

    def __init__(self, rows, fail=False):
        super().__init__(fail)
        self.rows = rows

    def __call__(self, state, params, x):
        return jnp.zeros((self.rows, 3)), jnp.zeros((self.rows, 2))


class _Implicit:
    """Stand-in for :mod:`vmex.core.implicit` as the warm-up threads use it."""

    def __init__(self, fail=False, cpus=2):
        self.fail, self.calls, self.cpus = fail, [], cpus

    def _overlapping_compiles_pay(self):
        return self.cpus > 1

    def _canonical_arrays(self, cfg, tree):
        return tree

    def block_factors_at(self, cfg, params, state, mask):
        self.calls.append("factors")
        return "factors"

    def warm_refinement(self, cfg, params):
        if self.fail:
            raise RuntimeError("no refinement")
        self.calls.append("refinement")

    def warm_staged_tangent(self, cfg, params, *, chunk, certify_rtol, certify_maxiter=None, fallbacks=False):
        if self.fail:
            raise RuntimeError("no tangent")
        self.calls.append(("tangent", chunk))


@pytest.mark.parametrize("fail", [False, True])
def test_gradient_programs_compile_from_placeholders_on_background_threads(fail):
    """The problem starts two warm-up threads (response fallbacks; pullback and direct Jacobian) from placeholder
    arguments; a failure in either is swallowed.  The pullback's programs (each function's cotangents and the shared
    adjoint, on the stacked rows) compile in parallel."""
    implicit = _Implicit(fail)
    implicit._placeholder_arguments = lambda cfg, params: (jnp.ones(3), jnp.ones(3), params)
    implicit._zeros_like_structure = lambda cfg, shapes: jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), shapes)
    implicit._refine_block_factor_core = lambda z, params, frozen, mask, cfg: (z * params, frozen + mask)
    model = type("Model", (), {"_im": implicit, "params0": jnp.ones(3), "chunk": 2})()
    model.cfg = type("Cfg", (), {"adjoint_tol": 1e-11, "adjoint_maxiter": 300})()
    cotangents = (_Cotangents(1, fail), _Cotangents(5, fail))
    problem = type("Problem", (), {"model": model, "_cotangents": cotangents, "_adjoint": _Program(fail),
                                   "_direct": _Program(fail)})()
    threads = fvc.ThreeTermsFreeBoundaryProblem._warm_gradient_programs(problem, np.ones(4))
    for thread in threads:
        thread.join()
    if not fail:
        assert ("tangent", 2) in implicit.calls
        for program in cotangents:
            (state, params, x), = program.lowered
            assert x.shape == (4,)
        (state, mask, params, state_bars, params_bars, factors), = problem._adjoint.lowered
        assert state_bars.shape == (6, 3) and params_bars.shape == (6, 2) and len(factors) == 2
        assert len(problem._direct.lowered) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_model_warm_up_threads_swallow_their_failures(fail):
    """The refinement and tangent warm-ups run on daemon threads; a failing one leaves nothing behind."""
    import threading

    implicit = _Implicit(fail)
    model = type("Model", (), {"_im": implicit, "params0": "p0", "chunk": 2})()
    model.cfg = type("Cfg", (), {"adjoint_tol": 1e-11})()
    fvc.ThreeTermsFreeBoundaryModel._warm_programs(model)
    model._warm.join()
    for thread in threading.enumerate():
        if thread.name in ("vmex-refine-warm", "vmex-tangent-warm"):
            thread.join()
    if not fail:
        assert "refinement" in implicit.calls and ("tangent", 2) in implicit.calls


def test_try_hands_back_the_exception():
    assert fvc._try(lambda a: a + 1, 1) == 2
    error = fvc._try(lambda: 1 / 0)
    assert isinstance(error, ZeroDivisionError)


def test_geometry_spectra_are_the_field_spectra_geometry():
    """The model's radial-line and axis inputs: _state_field_spectra's geometry coefficients without its fields,
    and the wout axis rows."""
    inp = vj.VmecInput.from_file(str(DATA / "input.solovev")).change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = replace(inp, ns_array=np.asarray([5]), ftol_array=np.asarray([1e-10]), niter_array=np.asarray([1000]))
    eq = opt.solve_equilibrium(inp)
    state, runtime = eq.solution, eq.solver_context
    full = vc._state_field_spectra(inp, state, runtime)
    xm, xn, rmnc, zmns = vc._state_geometry_spectra(state, runtime.resolution)
    for got, key in ((xm, "xm"), (xn, "xn"), (rmnc, "rmnc"), (zmns, "zmns")):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(full[key]))
    n_axis = int(runtime.resolution.ntor) + 1
    np.testing.assert_allclose(np.asarray(rmnc)[0, :n_axis], np.asarray(eq.wout.raxis_cc), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(np.asarray(zmns)[0, :n_axis], np.asarray(eq.wout.zaxis_cs), rtol=1e-13, atol=1e-15)


def test_compiled_surface_data_is_the_eager_construction():
    """The model's seed surface data, built as one program, equals the eager construction."""
    from vmex.core import implicit as im

    inp = vj.VmecInput.from_file(str(DATA / "input.solovev")).change_resolution(mpol=3, ntor=0, ntheta=12, nzeta=4)
    inp = replace(inp, ns_array=np.asarray([5]), ftol_array=np.asarray([1e-10]), niter_array=np.asarray([1000]))
    cfg = im.make_config(inp, ftol=1e-10, max_iterations=1000)
    params = im.params_from_input(inp)
    state, _ = im.solve_implicit_with_aux(params, cfg)
    got = fvc._surface_data(inp, cfg, state, params, 6, 8)
    want = vc.surface_field_data_from_state(inp, state, runtime=im.runtime_from_params(params, cfg), nphi=6,
                                            ntheta=8)
    for name in ("gamma", "B_total", "normal", "area_vector", "theta", "phi"):
        np.testing.assert_allclose(np.asarray(getattr(got, name)), np.asarray(getattr(want, name)),
                                   rtol=1e-12, atol=1e-14)
    assert (got.nfp, got.stellsym, got.signgs, got.source_convention) == \
        (want.nfp, want.stellsym, want.signgs, want.source_convention)


def test_warm_ups_stay_off_on_a_single_cpu():
    """A process confined to one CPU starts no warm-up thread: there is nothing to overlap."""
    implicit = _Implicit(cpus=1)
    model = type("Model", (), {"_im": implicit, "params0": "p0", "chunk": 2})()
    model.cfg = type("Cfg", (), {"adjoint_tol": 1e-11, "adjoint_maxiter": 300})()
    fvc.ThreeTermsFreeBoundaryModel._warm_programs(model)
    assert model._warm is None
    problem = type("Problem", (), {"model": model, "_pullback": _Program(), "_direct": _Program()})()
    assert fvc.ThreeTermsFreeBoundaryProblem._warm_gradient_programs(problem, np.ones(2)) == (None, None)
    assert implicit.calls == []


def test_uncertified_state_responses_are_an_error():
    """A response column that misses its certificate stops the linearization instead of entering the Jacobian."""
    from vmex.core.implicit import LinearResponseReport

    def report(converged):
        n = len(converged)
        return LinearResponseReport(residual_norm=np.full(n, 2.0), tolerance=np.ones(n),
                                    iterations=np.zeros(n, int), converged=np.asarray(converged))

    assert fvc._certified("dz", report([True, True])) == "dz"
    with pytest.raises(RuntimeError, match="1 of 2 state responses"):
        fvc._certified("dz", report([True, False]))


def _beta2p5_deck(mpol):
    """The LP QA 2.5 % deck at ``mpol = ntor`` with its own radial ladder (a production-size free boundary)."""
    inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_beta2p5_bootstrap")
    inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=max(64, 4 * mpol + 8), nzeta=max(64, 4 * mpol + 8))
    return replace(inp, lfreeb=True, mgrid_file="essos_coils(direct)")


@needs_vc
@pytest.mark.full  # minutes per case on a CPU runner; the production case is the GPU guard of the miscompile
@pytest.mark.parametrize("case", [
    "small",
    # the size at which XLA:GPU miscompiled a bare lax.map over the push's columns (columns 55-84 % off)
    "production",
])
def test_finite_beta_jacobian_columns_match_resolved_differences(case):
    """Finite beta: every Jacobian column pushed by the model is the central difference of tightly re-solved
    rows."""
    if case == "small":
        inp, n = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 11), 16
        field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    else:
        inp, n = _beta2p5_deck(10), 48
        field = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta2p5_bootstrap.json")
    nfp = int(inp.nfp)
    model = fvc.ThreeTermsFreeBoundaryModel(inp, nphi=n, ntheta=n, quadrature=(4 * nfp * n, 2 * n))
    x0 = np.zeros(model.x0.size)
    _, aux = model.evaluate(x0, model.params0, field, tight=True)
    J = np.asarray(model.linearize(x0, aux, field)[0])
    for k in (0, model.n_boundary - 1):
        h = 1e-5
        rows = []
        for sign in (1, -1):
            params = model.with_boundary(model.params0, jnp.asarray(x0 + sign * h * np.eye(x0.size)[k]))
            state, _ = model.solve(params, seed=aux[0], tight=True)
            rows.append(np.asarray(model._rows(state, params, model.attach(field))))
        fd = (rows[0] - rows[1]) / (2 * h)
        assert np.linalg.norm(J[:, k] - fd) < 1e-5 * np.linalg.norm(fd), k


def test_vacuum_displacement_includes_the_next_poloidal_harmonic():
    """Vacuum unknowns: the boundary modes, then m = mpol on every toroidal mode (it moves the top coefficients
    through the poloidal variation of the transport direction), then (0, 0); the columns stay independent."""
    inp = _deck("input.LandremanPaul2021_QA_lowres", 4, 11)
    top = max(inp.mpol - 1, inp.ntor)
    M, modes, scale = fvc._normal_displacement_map(inp, top, extend=True)
    M0, modes0, _ = fvc._normal_displacement_map(inp, top)
    extra = [(inp.mpol, n) for n in range(-inp.ntor, inp.ntor + 1)]
    assert modes == modes0[:-1] + extra + [(0, 0)] and M.shape == (M0.shape[0], M0.shape[1] + len(extra))
    assert scale.size == len(modes)
    s = np.linalg.svd(M, compute_uv=False)
    assert s[-1] > 1e-6 * s[0]  # an independent direction per unknown
    np.testing.assert_allclose(M[:, :M0.shape[1] - 1], M0[:, :-1], atol=1e-7 * np.abs(M0).max())  # each map to its grid's rtol


def test_a_plasma_below_the_fields_round_off_is_vacuum():
    """A current or pressure whose field is below the round-off of the deck's toroidal field cannot change B in
    double precision, so it is no plasma: a WOUT restart's CURTOR 1.86e-12 A (G2, LP QA 12 x 12) selected the
    finite-beta formulation (221 instead of 313 boundary coordinates, QS 3.19e-7 instead of 5.07e-7)."""
    lp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 11), curtor=0.0, pres_scale=0.0)
    assert fvc._plasma_below_roundoff(lp)
    assert fvc._plasma_below_roundoff(replace(lp, curtor=1.8634724839594607e-12))
    assert fvc._plasma_below_roundoff(replace(lp, curtor=-1.8634724839594607e-12))
    assert not fvc._plasma_below_roundoff(replace(lp, curtor=1e-3))  # a milliampere is a plasma current
    p = replace(lp, pres_scale=1.0, am=np.r_[1.0, -1.0, np.zeros(19)])  # 1 Pa on the axis: a plasma pressure
    assert fvc._edge_pressure(p)[1] > 0.0 and not fvc._plasma_below_roundoff(p)
    tiny = replace(p, pres_scale=1e-40)
    assert fvc._edge_pressure(tiny)[1] > 0.0 and fvc._plasma_below_roundoff(tiny)
    assert not fvc._plasma_below_roundoff(replace(lp, zbs=np.zeros_like(lp.zbs)))  # no cross section: no field scale


@pytest.mark.full
def test_round_off_current_keeps_the_vacuum_formulation():
    """CURTOR 0 and a round-off CURTOR give the same model: the vacuum rows and the same boundary unknowns."""
    lp = replace(_deck("input.LandremanPaul2021_QA_lowres", 4, 11), phiedge=-0.025, curtor=0.0,
                 am=np.zeros(21), pres_scale=0.0)
    zero = fvc.ThreeTermsFreeBoundaryModel(lp)
    roundoff = fvc.ThreeTermsFreeBoundaryModel(replace(lp, curtor=1.8634724839594607e-12))
    assert zero.vacuum and roundoff.vacuum
    assert (zero.max_mode, zero.n_boundary, zero.boundary_modes) == (
        roundoff.max_mode, roundoff.n_boundary, roundoff.boundary_modes)
    assert not fvc.ThreeTermsFreeBoundaryModel(replace(lp, curtor=1e3)).vacuum


@needs_vc
@pytest.mark.full  # about 6 min per draw on a CPU runner
@pytest.mark.parametrize("seed", [0, 1])
def test_finite_beta_design_gradients_for_random_profiles(seed):
    """Design gradients of the finite-beta free boundary for random pressure and current profiles.

    LP QA at 0.5 % beta with its fitted coils (mpol 3, ns 11, 16 x 16 grids). The pressure is
    ``p0 (1 - s)(1 + a1 s + a2 s^2)`` (zero at the edge) and the current a power series in ``I'(s)``, with
    coefficients drawn from ``seed``. The design moves the coil currents, PRES_SCALE, AM[1], AC[0], AC[1] and
    CURTOR. Implicit-function gradients of the aspect ratio, iota at the edge and the major radius are compared
    with central differences of re-solved free boundaries of the same problem. The GPU sweep behind this test
    (8 x 8, QA/QH/QI decks, peaked, broad and edge-current profiles, the Redl current in the solve) agrees to
    1e-4 or better at its best step, limited by the re-solves' noise.
    """
    from vmex.core.freeboundary_problem import FreeBoundaryProblem
    from vmex.core.implicit import iota_edge

    rng = np.random.default_rng(seed)
    deck = _deck("input.LandremanPaul2021_QA_beta0p5_bootstrap", 3, 11)
    a1, a2 = rng.uniform(-0.5, 0.5, 2)
    am = np.zeros_like(np.asarray(deck.am, dtype=float))
    am[:4] = float(deck.am[0]) * np.convolve([1.0, -1.0], [1.0, a1, a2])  # p(1) = 0
    ac = np.zeros_like(np.asarray(deck.ac, dtype=float))
    ac[:3] = rng.uniform(-1.0, 1.0, 3) + np.array([1.5, 0.0, 0.0])  # I'(0) > 0: a monotonic enclosed current
    deck = replace(deck, am=am, pcurr_type="power_series", ac=ac, ac_aux_s=None, ac_aux_f=None,
                   curtor=float(deck.curtor) * rng.uniform(0.5, 1.0))
    coils = _coil_field("ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json")
    plasma = (("pres_scale", None), ("am", 1), ("ac", 0), ("ac", 1), ("curtor", None))
    scale = [abs(float(getattr(deck, f) if i is None else np.asarray(getattr(deck, f))[i])) for f, i in plasma]

    def plasma_at(params, x):
        changes = {}
        for k, ((name, i), s) in enumerate(zip(plasma, scale)):
            value = changes.get(name, getattr(params, name))
            changes[name] = value + x[1 + k] * s if i is None else value.at[i].add(x[1 + k] * s)
        return replace(params, **changes)

    n = 1 + len(plasma)
    problem = FreeBoundaryProblem.from_loss(
        deck, lambda state, runtime, x: opt.aspect_ratio(state, runtime), np.zeros(n),
        field_from_parameters=lambda x: replace(coils, currents=coils.currents * (1 + x[0])),
        plasma_from_parameters=plasma_at, quantities=(iota_edge, opt.major_radius),
        boundary_condition="three_terms",
        three_terms_options=dict(nphi=16, ntheta=16, quadrature=(4 * 2 * 16, 32), boundary_max_nfev=80))
    x0 = problem.x0
    _, grad = problem.value_and_grad(x0)
    exact = np.vstack([grad, problem.constraint_jac(x0)])
    h = 3e-3
    for k in range(n):
        e = h * np.eye(n)[k]
        values = [np.r_[problem.fun(x0 + s * e), problem.constraint_values(x0 + s * e)] for s in (1, -1)]
        fd = (values[0] - values[1]) / (2 * h)
        floor = 1e-3 * np.max(np.abs(exact), axis=1)  # a component far below its row's size is not resolved
        np.testing.assert_array_less(np.abs(exact[:, k] - fd), 2e-2 * np.maximum(np.abs(fd), floor) + 1e-12,
                                     err_msg=f"design coordinate {k} (seed {seed})")


def test_deck_response_cache_round_trip_and_keys(tmp_path, monkeypatch):
    """The deck-response cache: a stored entry comes back exactly; the key changes with any input (array bytes,
    dataclass fields, containers) and a damaged or missing entry is a miss; with the compilation cache off there
    is no cache at all."""
    monkeypatch.setenv("VMEX_COMPILATION_CACHE", "1")
    monkeypatch.setenv("VMEX_COMPILATION_CACHE_DIR", str(tmp_path))
    rng = np.random.default_rng(1)
    value = (tuple(jnp.asarray(rng.standard_normal((3, 4))) for _ in range(4)),
             tuple(jnp.asarray(rng.standard_normal((2, 3, 4))) for _ in range(4)))
    inp = _deck("input.LandremanPaul2021_QA_lowres", 3, 11)
    key = fvc._digest("deck-response", inp, 48, value)
    assert fvc._deck_cache_load(key) is None
    fvc._deck_cache_store(key, value)
    back = fvc._deck_cache_load(key)
    for a, b in zip(jax.tree.leaves(back), jax.tree.leaves(value)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    bumped = np.asarray(inp.rbc).copy()
    bumped[0, 1] += 1e-15
    assert fvc._digest("deck-response", replace(inp, rbc=bumped), 48, value) != key
    assert fvc._digest("deck-response", replace(inp, phiedge=inp.phiedge * (1 + 1e-15)), 48, value) != key
    assert fvc._digest("deck-response", inp, 64, value) != key
    assert fvc._digest("deck-response", inp, 48, value) == key and len(fvc._source_digest()) == 64
    fvc._deck_cache_path(key).write_bytes(b"damaged")
    assert fvc._deck_cache_load(key) is None
    fvc._deck_cache_store(key, value[0])  # not a deck response (4 leaves): not stored
    assert fvc._deck_cache_load(key) is None
    calls = []
    fresh = fvc._digest("another deck")
    first = fvc._deck_cached(fresh, lambda: calls.append(1) or value)
    again = fvc._deck_cached(fresh, lambda: calls.append(1) or value)
    assert calls == [1] and len(jax.tree.leaves(first)) == len(jax.tree.leaves(again)) == 8
    # every kind of input enters the key: mappings, registered pytrees, plain objects, anything else by repr
    assert fvc._digest({"a": 1}) != fvc._digest({"a": 2}) and fvc._digest({"a": 1, "b": 2}) == fvc._digest({"b": 2, "a": 1})
    from vmex.core.implicit import params_from_input
    params = params_from_input(inp)
    assert fvc._digest(params) != fvc._digest(replace(params, phiedge=params.phiedge + 1.0))
    from types import SimpleNamespace
    assert fvc._digest(SimpleNamespace(a=1)) != fvc._digest(SimpleNamespace(a=2))
    assert fvc._digest(slice(1, 2)) != fvc._digest(slice(1, 3))

    class Pair:  # a registered pytree that is no dataclass
        def __init__(self, a, b):
            self.a, self.b = a, b

    jax.tree_util.register_pytree_node(Pair, lambda p: ((p.a, p.b), None), lambda _, c: Pair(*c))
    assert fvc._digest(Pair(jnp.ones(2), 1.0)) != fvc._digest(Pair(jnp.ones(2), 2.0))
    monkeypatch.setenv("VMEX_COMPILATION_CACHE", "0")
    assert fvc._deck_cache_path(key) is None and fvc._deck_cache_load(key) is None
    fvc._deck_cache_store(key, value)  # no cache: a no-op


@needs_vc
def test_band_continuation_is_the_direct_solve_and_builds_the_band_up_where_it_stalls(monkeypatch):
    """The band continuation: where Newton on the model's band converges it is that solve; where it stalls
    (forced here), the band is built up from band 1, each band from the previous band's boundary, to the same
    root; a band that stalls on the way ends it as a stall (not converged) that reports the last converged band."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    model = fvc.ThreeTermsFreeBoundaryModel(inp, nphi=16, ntheta=16)
    target = model.max_mode
    assert target == model.top_band > 2
    direct_model, direct = fvc.solve_band_continuation(model, model.params0, field)
    assert direct_model is model and direct["converged"] and direct["band"] == direct["target"] == target
    assert [b[:2] for b in direct["bands"]] == [(target, True)]

    newton = fvc.ThreeTermsFreeBoundaryModel._newton_boundary
    calls = []

    def stalling(bands):
        def patched(self, params, field, *, x0=None, **kwargs):
            calls.append(self.max_mode)
            if self.max_mode in bands and (self.max_mode != target or calls.count(target) == 1):
                return dict(converged=False, nfev=1, njev=0, x=np.asarray(self.x0 if x0 is None else x0),
                            jacobian=None)
            return newton(self, params, field, x0=x0, **kwargs)
        return patched

    monkeypatch.setattr(fvc.ThreeTermsFreeBoundaryModel, "_newton_boundary", stalling({target}))
    built_model, built = fvc.solve_band_continuation(model, model.params0, field, x0=0.5 * direct["x"])
    assert built["converged"] and built["band"] == built_model.max_mode == target and built_model is model
    assert [b[:2] for b in built["bands"]] == [(target, False)] + [(b, True) for b in range(1, target + 1)]
    for a, b in zip((built["aux"][2].rbc, built["aux"][2].zbs), (direct["aux"][2].rbc, direct["aux"][2].zbs)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-9)

    calls.clear()
    monkeypatch.setattr(fvc.ThreeTermsFreeBoundaryModel, "_newton_boundary", stalling({target, 2}))
    lower_model, lower = fvc.solve_band_continuation(model, model.params0, field)
    assert not lower["converged"] and lower["band"] == lower_model.max_mode == 1 and lower["target"] == target
    assert [b[:2] for b in lower["bands"]] == [(target, False), (1, True), (2, False)]
    calls.clear()
    stalled = model.solve_boundary(model.params0, field)  # this band's Newton result, reported as a stall
    assert not stalled["converged"] and stalled["band"] == 1 and stalled["x"].size == model.x0.size
    calls.clear()
    trial = model.solve_boundary(model.params0, field, continuation=False)  # a trial: Newton alone, no band 1
    assert not trial["converged"] and calls == [target]


def test_band_continuation_newton_damps_and_trials_do_not(monkeypatch):
    """Seeds and re-solves (the band continuation) let Newton backtrack freely on every band: ill-conditioned rows
    may need shortened steps inside the basin.  Optimizer trials (Newton alone) keep the fail-fast rule."""
    inp = replace(_deck("input.LandremanPaul2021_QA_lowres", 3, 11), phiedge=-0.025, curtor=0.0, am=np.zeros(21))
    field = _coil_field("ESSOS_biot_savart_LandremanPaulQA.json")
    model = fvc.ThreeTermsFreeBoundaryModel(inp, nphi=16, ntheta=16)
    seen = []

    def stall(self, params, field, *, x0=None, undamped=False, **kwargs):
        seen.append((self.max_mode, undamped))
        return dict(converged=False, nfev=1, njev=0, x=np.asarray(self.x0 if x0 is None else x0), jacobian=None)

    monkeypatch.setattr(fvc.ThreeTermsFreeBoundaryModel, "_newton_boundary", stall)
    fvc.solve_band_continuation(model, model.params0, field)
    assert seen and not any(undamped for _, undamped in seen)
    seen.clear()
    model.solve_boundary(model.params0, field, continuation=False)
    assert seen == [(model.max_mode, True)]


def test_rereference_failures_keep_the_deck():
    """A deck that cannot be re-referenced, or a new deck on which the trial does not converge, leaves the problem
    on its deck: the trial fails and nothing of the new model remains."""
    fixed = fvc.ThreeTermsFreeBoundaryModel.__new__(fvc.ThreeTermsFreeBoundaryModel)
    fixed.fixed_boundary = True
    with pytest.raises(ValueError, match="no boundary to re-reference"):
        fixed.rereferenced(np.zeros(1))

    problem = fvc.ThreeTermsFreeBoundaryProblem.__new__(fvc.ThreeTermsFreeBoundaryProblem)
    deck = type("Deck", (), {})()
    problem.__dict__.update({name: object() for name in problem._BOUND}, model=deck, boundary_max_nfev=5,
                            field=lambda x: None, stats=dict(boundary_evaluations=0), references=[])
    bound = dict((name, problem.__dict__[name]) for name in problem._BOUND)

    def no_deck(b):
        raise VmecError("no equilibrium on that boundary")

    deck.rereferenced = no_deck
    assert problem._rereference(np.zeros(2), dict(b=np.zeros(3))) is None
    other = type("Other", (), {"solve_boundary": lambda self, *a, **k: dict(converged=False, nfev=4)})()
    deck.rereferenced = lambda b: other

    def bind(model, x):
        problem.model, problem._params_at = model, lambda x: None

    problem._bind = bind
    assert problem._rereference(np.zeros(2), dict(b=np.zeros(3))) is None
    assert all(problem.__dict__[name] is bound[name] for name in problem._BOUND) and problem.model is deck
    assert problem.stats["boundary_evaluations"] == 4 and problem.references == []

    def raises(*args, **kwargs):
        raise VmecError("no certified equilibrium")

    other.solve_boundary = raises  # an uncertified equilibrium on the new deck: the same failure
    assert problem._rereference(np.zeros(2), dict(b=np.zeros(3))) is None and problem.model is deck

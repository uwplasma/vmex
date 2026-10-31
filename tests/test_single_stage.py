"""Single-stage orchestration contracts without optional solvers or field codes."""
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import sys

import numpy as np
import pytest
import jax.numpy as jnp

from vmex.core import single_stage as ss


@dataclass
class Deck:
    nfp: int = 4
    phiedge: float = .23
    pres_scale: float = 30000.
    curtor: float = 17.
    ncurr: int = 1
    lasym: bool = False
    lfreeb: bool = False
    mgrid_file: str = "NONE"
    rbc: object = field(default_factory=lambda: np.ones((3, 2)))
    zbs: object = field(default_factory=lambda: np.zeros((3, 2)))
    am: object = field(default_factory=lambda: np.array([1., -1.]))
    ac: object = field(default_factory=lambda: np.array([0., 1.]))
    ns_array: object = field(default_factory=lambda: np.array([17]))
    ftol_array: object = field(default_factory=lambda: np.array([1e-9]))
    niter_array: object = field(default_factory=lambda: np.array([1000]))

    def to_indata(self, filename):
        Path(filename).write_text("saved input")


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Replace numerical boundaries, retaining real orchestration and distances."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ss, "jnp", np)
    events = []
    monkeypatch.setattr(ss.jax, "clear_caches", lambda: events.append("clear caches"))
    monkeypatch.setattr(ss.gc, "collect", lambda: None)

    class Coils:
        dofs_curves = np.array([1., 2., 3.])
        dofs_currents = np.array([4., 5.])
        gamma = np.array([[[0., 0., .5], [1., 0., .5]],
                          [[0., 1., .5], [1., 1., .5]]])
        curvature = np.array([[2., 3.], [4., 5.]])

        def with_dofs(self, dofs):
            result = Coils()
            result.updated_dofs = np.asarray(dofs)
            return result

        def to_json(self, path):
            Path(path).write_text("saved coils")

        def to_vtk(self, path):
            events.append(("coil vtk", path, self))

    class Surface:
        gamma = np.array([[[0., 0., 0.], [1., 0., 0.]]])

        def to_vtk(self, path, **kwargs):
            events.append(("surface vtk", path, kwargs["field"].coils))

        @classmethod
        def from_wout_file(cls, path, **kwargs):
            events.append(("held surface", path, kwargs))
            result = cls()
            result.gamma = result.gamma + [0., 0., .01]
            return result

    def make_surface(rbc, zbs, nfp, **kwargs):
        events.append(("surface", np.asarray(rbc), np.asarray(zbs), nfp, kwargs))
        return Surface()

    modules = {name: ModuleType(name) for name in ("essos", "essos.fields", "essos.surfaces")}
    modules["essos.fields"].BiotSavart = lambda coils: NS(coils=coils)
    modules["essos.surfaces"].SurfaceRZFourier = Surface
    modules["essos.surfaces"].surfacerzfourier_from_boundary = make_surface
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    deck, seed, fine = Deck(), NS(solution="seed state"), NS(
        solution="fine state", solver_context="fine context", state="fine state",
        runtime="fine context", result=NS(converged=True), wout=NS(nfp=4))
    problem = NS(x0=np.array([.1, .2]), scales=np.array([1., 2.]))
    problem.input_from_x = lambda x: deck
    problem.equilibrium_from_x = lambda x: seed
    problem.boundary_from_x = lambda x: (np.asarray(x), -np.asarray(x))
    final_coils = Coils()
    monitor = NS(save=lambda path: events.append(("history", path)),
                 plot=lambda path, **kw: events.append(("history plot", path, kw)),
                 movie_surface_coils=lambda path, *a, **kw: events.append(("movie", path, a, kw)))
    joint = NS(problem=problem, x0=np.arange(5.), scales=np.full(5, .1), n_boundary=2,
               coils=Coils(), vc_digits=3, nphi=17, ntheta=16, monitor=monitor,
               objects_from_x=lambda x: events.append(("objects", np.asarray(x).copy()))
               or (Surface(), final_coils))
    monkeypatch.setattr(ss.vc, "surface_field_data_from_state",
                        lambda *a, **kw: events.append(("field data", a, kw)) or "data")
    monkeypatch.setattr(ss.vc, "plan_vc_precision", lambda *a, **kw: "precision")
    monkeypatch.setattr(ss, "total_normal_field", lambda *a, **kw: (
        np.array([[.01, .02]]), np.array([[.25, .75]])))
    for name, value in (("min_abs_iota", .45), ("aspect_ratio", 6.), ("volume_average_beta", .005)):
        monkeypatch.setattr(ss.opt, name, lambda *a, value=value: value)
    return NS(events=events, Coils=Coils, Surface=Surface, deck=deck, seed=seed,
              fine=fine, joint=joint, final_coils=final_coils)


@pytest.mark.parametrize("outcome", ["converged", "not converged", "no vacuum", "vmec error"])
def test_verification_preserves_physics_and_independently_checks_the_held_boundary(harness, monkeypatch, outcome):
    h = harness
    solves = []

    def fixed(inp, **kwargs):
        solves.append(("fixed", inp, kwargs))
        return h.fine

    def free(inp, **kwargs):
        solves.append(("free", inp, kwargs))
        if outcome == "vmec error":
            raise ss.VmecError("independent free solve failed")
        return NS(converged=np.array([True, outcome != "not converged"]), state="free state",
                  fsqr=1e-13, fsqz=2e-13, fsql=3e-13, iterations=14,
                  vacuum=None if outcome == "no vacuum" else NS(delbsq=.001))

    monkeypatch.setattr(ss.opt, "solve_equilibrium", fixed)
    monkeypatch.setattr(ss.vj, "solve_free_boundary_multigrid", free)
    monkeypatch.setattr(ss.vj, "wout_from_state", lambda **kw: NS(
        aspect=6.1, betatotal=.0049, iotas=np.array([0., -.44, -.46])))
    monkeypatch.setattr(ss.vj, "write_wout", lambda path, wout: path)
    equilibrium, surface, metrics, held = ss.verify_design(
        h.joint, np.ones(5), prefix="test", ns=31, ftol=1e-11,
        free_ns=(16, 41), free_ftol=1e-10, verbose=False)
    assert equilibrium is h.fine
    assert isinstance(surface, h.Surface)
    assert [row[0] for row in solves] == ["fixed", "free"]
    for _, inp, _ in solves:
        assert (inp.phiedge, inp.pres_scale, inp.curtor, inp.ncurr) == (.23, 30000., 17., 1)
        np.testing.assert_array_equal(inp.am, h.deck.am)
        np.testing.assert_array_equal(inp.ac, h.deck.ac)
    np.testing.assert_array_equal(solves[0][1].ns_array, [31])
    assert solves[0][2] == {"initial_state": "seed state", "verbose": False}
    np.testing.assert_array_equal(solves[1][1].ns_array, [16, 41])
    np.testing.assert_array_equal(solves[1][1].ftol_array, [1e-10, 1e-10])
    assert solves[1][1].lfreeb and solves[1][1].mgrid_file == "direct ESSOS field"
    assert "initial_state" not in solves[1][2]  # A cold independent check.
    assert solves[1][2]["external_field"].coils is h.final_coils
    assert solves[1][2]["raise_on_max_iterations"] is False
    assert Path("input.test_optimized").is_file()
    assert Path("coils_test_optimized.json").is_file()
    objects = next(row[1] for row in h.events if isinstance(row, tuple) and row[0] == "objects")
    np.testing.assert_array_equal(objects, h.joint.x0 + h.joint.scales)
    assert metrics["B.n/B RMS"] == pytest.approx(np.sqrt(.25*.01**2 + .75*.02**2))
    assert metrics["coil-coil distance"] == pytest.approx(1.)
    assert metrics["coil-surface distance"] == pytest.approx(.5)
    assert held["converged"] == (outcome == "converged")
    assert held["metrics_provisional"] == (outcome != "converged")
    if outcome == "vmec error":
        assert held["error"] == "independent free solve failed"
        assert not any(event[0] == "held surface" for event in h.events if isinstance(event, tuple))
    else:
        assert held["sampled_boundary_distance_m"] == pytest.approx(.01)
        assert held["coil_surface_distance_m"] == pytest.approx(.49)
        assert held["min_abs_iota"] == pytest.approx(.44)
        assert held["delbsq"] == (None if outcome == "no vacuum" else .001)


@pytest.mark.parametrize("make_movie", [False, True])
def test_saved_artifacts_pair_initial_and_final_geometry_and_replay_the_monitor(harness, monkeypatch, make_movie, capsys):
    h = harness
    monkeypatch.setattr(ss.vj, "write_wout", lambda path, wout: h.events.append(("wout", path, wout)) or path)
    monkeypatch.setattr(ss.vj, "plot_optimization_objects", lambda *a: h.events.append(("geometry plot", a)))
    monkeypatch.setattr(ss.vj, "plot_wout", lambda *a: {"summary": "summary.png"})
    monkeypatch.setattr(ss, "coil_field", lambda coils: ("field", coils))
    surface = h.Surface()
    ss.save_outputs(h.joint, np.ones(5), h.fine, surface,
                    prefix="test", make_movie=make_movie, surface_color="B.n/B")
    vtk = [row for row in h.events if isinstance(row, tuple) and row[0] == "surface vtk"]
    assert [(row[1], row[2]) for row in vtk] == [
        ("surface_test_initial", h.joint.coils), ("surface_test_optimized", h.final_coils)]
    initial = next(row for row in h.events if isinstance(row, tuple) and row[0] == "surface")
    np.testing.assert_array_equal(initial[1], h.joint.problem.x0)
    np.testing.assert_array_equal(initial[2], -h.joint.problem.x0)
    options = next(row[1] for row in h.events if isinstance(row, tuple) and row[0] == "objects")
    np.testing.assert_array_equal(options, h.joint.x0 + h.joint.scales)
    geometry = next(row for row in h.events if isinstance(row, tuple) and row[0] == "geometry plot")[1]
    assert geometry[1][2] is h.joint.coils and geometry[2] == ("Optimized", surface, h.final_coils)
    assert ("history", "test_objectives.csv") in h.events
    assert any(row[:2] == ("wout", "wout_test_optimized.nc") for row in h.events if isinstance(row, tuple))
    movies = [row for row in h.events if isinstance(row, tuple) and row[0] == "movie"]
    assert bool(movies) == make_movie
    if make_movie:
        options = movies[0][3]
        assert options["plasma_problem"] is h.joint.problem
        assert options["surface_color"] == "B.n/B"
        assert options["external_field"]((surface, h.final_coils)) == ("field", h.final_coils)
        assert (options["nphi"], options["ntheta"]) == (17, 16)
    assert "Wrote summary.png" in capsys.readouterr().out


def test_initialization_freezes_one_seed_plan_and_preserves_currents(harness, monkeypatch):
    h = harness
    calls = []
    h.seed.solver_context = "seed context"
    # Execute the two objective closures without an adjoint or JAX compilation.
    monkeypatch.setattr(ss.jax, "jit", lambda function: function)
    monkeypatch.setattr(ss.jax, "value_and_grad", lambda function, **kw: (
        lambda u: (function(u), np.zeros_like(u))))
    monkeypatch.setattr(ss.opt, "OptimizationMonitor", lambda problem: "monitor")
    h.joint.problem.jax_objective_from_state = lambda x, limits, **kw: (
        calls.append((x.copy(), kw, limits("state", "context"))) or "plasma result")
    joint = ss.SingleStageProblem(h.joint.problem, h.joint.coils,
        lambda coils, surface: np.array([1., 2., 3., 4.]), nphi=17, ntheta=16,
        iota_floor=.4, aspect_limit=7., normal_field_limit=.03)
    np.testing.assert_array_equal(joint.x0, [.1, .2, 1., 2., 3.])
    np.testing.assert_allclose(joint.scales, [.02, .04, .05, .05, .05])
    assert joint.seed_equilibrium is h.seed and joint.monitor == "monitor"
    assert joint.nfp == 4 and joint.counts == {"trials": 0, "rejected": 0}
    data_calls = [row for row in h.events if isinstance(row, tuple) and row[0] == "field data"]
    assert len(data_calls) == 1 and data_calls[0][2]["runtime"] == "seed context"
    _, coils = joint.objects_from_x(joint.x0 + joint.scales)
    np.testing.assert_allclose(coils.updated_dofs, [1.05, 2.05, 3.05, 4., 5.])
    assert joint._plasma_vg(np.zeros(5))[0] == "plasma result"
    assert calls[0][1] == {"n_extra_terms": 1, "return_status": True}
    assert calls[0][2] == 0.
    (cost, terms), _ = joint._coil_vg(np.zeros(5))
    assert cost == 10. and terms.tolist() == [1., 2., 3., 4.]


def test_physical_flux_uses_the_full_phi_zero_section_and_positive_cartesian_phi(harness, monkeypatch):
    inp = Deck(rbc=np.array([[.1, 0.], [1., .2], [.1, 0.]]),
               zbs=np.array([[0., 0.], [0., .2], [0., 0.]]))
    # Uniform +Y field: physical +phi flux is B*pi*a^2, regardless of center.
    monkeypatch.setattr(ss, "coil_field", lambda coils: lambda points: np.broadcast_to([0., 3., 0.], points.shape))
    assert ss.toroidal_flux(inp, None, nr=8, ntheta=64) == pytest.approx(3*np.pi*.2**2)
    inp.lasym = True
    with pytest.raises(ValueError, match="stellarator symmetry"):
        ss.toroidal_flux(inp, None)


def test_batched_coil_field_preserves_point_shape_and_normal_rms_uses_area_weights(harness, monkeypatch):
    monkeypatch.setattr(sys.modules["essos.fields"], "BiotSavart", lambda coils: NS(
        B=lambda point: np.asarray(point) + [0., 0., 1.]))
    monkeypatch.setattr(ss.jax, "vmap", lambda function: lambda points: np.stack([
        function(point) for point in points]))
    points = np.arange(12.).reshape(2, 2, 3)
    np.testing.assert_array_equal(ss.coil_field(harness.Coils())(points), points + [0., 0., 1.])
    surface = harness.Surface()
    surface.unitnormal = np.array([[[0., 0., 1.], [1., 0., 0.]]])
    surface.area_element = np.array([[1., 3.]])
    assert ss.normal_field_rms(harness.Coils(), surface) == pytest.approx(np.sqrt(.625))


def test_total_normal_field_uses_equilibrium_strength_and_the_frozen_precision(monkeypatch):
    monkeypatch.setattr(ss, "jnp", np)
    data = NS(B_total=np.array([[[0., 0.]], [[0., 0.]], [[3., 4.]]]))
    field, weights, calls = object(), np.array([[.25, .75]]), []

    def residual(external_field):
        assert external_field is field
        return np.array([[.3, .8]])

    interface = NS(weights=weights, bnormal_residual=residual)
    monkeypatch.setattr(ss, "coil_field", lambda coils: field)
    monkeypatch.setattr(ss.vc, "surface_field_data_from_state", lambda *a, **kw: data)
    monkeypatch.setattr(ss.vc, "PlasmaVacuumInterface", NS(from_surface_data=lambda *a, **kw: (
        calls.append(kw) or interface)))
    normal, actual_weights = ss.total_normal_field("input", "coils", "state", "context", 1, 2,
                                                   "caller precision", digits=5)
    np.testing.assert_allclose(normal, [[.1, .2]])
    assert actual_weights is weights
    assert calls == [{"digits": 5, "precision": "caller precision"}]


def test_common_current_scale_fits_external_target_with_area_weights():
    from vmex.core.single_stage import common_current_scale

    coil = jnp.asarray([[[1., 0.]], [[0., 2.]], [[0., 0.]]])
    plasma = jnp.asarray([[[.5, .3]], [[.2, -.4]], [[0., 0.]]])
    external = jnp.asarray([[[2., 0.]], [[0., 3.]], [[0., 0.]]])
    weights = jnp.asarray([[.8, .2]])
    total = external + plasma
    factor = float(common_current_scale(coil, total - plasma, weights))
    # Weighted optimum differs from both the unweighted and total-field fits.
    assert factor == pytest.approx(1.75)
    assert factor != pytest.approx(float(common_current_scale(coil, total, weights)))
    def loss(value):
        return float(jnp.sum(weights * jnp.sum((value * coil - external)**2, axis=0)))
    assert loss(factor) < loss(factor - .1)
    assert loss(factor) < loss(factor + .1)


def test_common_current_scale_selects_physical_orientation_and_rejects_no_field():
    from vmex.core.single_stage import common_current_scale

    target = jnp.asarray([[[1., 2.]], [[.3, .4]], [[.5, .6]]])
    weights = jnp.asarray([[.25, .75]])
    assert float(common_current_scale(-target / 3, target, weights)) == pytest.approx(-3.)
    assert not np.isfinite(float(common_current_scale(jnp.zeros_like(target), target, weights)))


def _geometry_objectives(monkeypatch):
    module = ModuleType("essos.objective_functions")
    module.loss_coil_separation = lambda *a, **kw: 2.
    module.loss_coil_surface_distance = lambda *a, **kw: 3.
    monkeypatch.setitem(sys.modules, module.__name__, module)
    # These tests concern geometric normalization, not AD or ESSOS integration.
    monkeypatch.setattr(ss.jax, "vmap", lambda function: lambda *xs: np.asarray([
        function(*row) for row in zip(*xs)]))


class _GeometryCoils:
    def __init__(self, gamma, lengths, curvature):
        self.gamma, self.length, self.curvature = map(np.asarray, (gamma, lengths, curvature))
        self.dofs_currents = np.ones(len(lengths))

    def __len__(self):
        return len(self.gamma)




def test_legacy_coil_self_costs_count_only_unique_coils(harness, monkeypatch):
    _geometry_objectives(monkeypatch)
    coils = _GeometryCoils(np.concatenate((harness.Coils.gamma, harness.Coils.gamma + 10)),
                           [1., 2., 100., 200.], [[2., 3.], [4., 5.], [100., 100.], [200., 200.]])
    coils.dofs_currents = np.ones(2)
    penalty = ss.CoilPenalty(1.5, 4., 1.5, .7)
    before = penalty(coils, harness.Surface())
    coils.length[2:] *= 100
    coils.curvature[2:] *= 100
    np.testing.assert_array_equal(penalty(coils, harness.Surface()), before)
    coils.length[0] = 3.
    assert penalty(coils, harness.Surface())[0] > before[0]

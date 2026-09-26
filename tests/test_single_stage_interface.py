"""Matched entry points, file restarts and real coil fits without plasma solves."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/single-stage-benchmarks"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def entries():
    return (load("single_stage_optimization_scalar"),
            load("free_boundary_single_stage_optimization_scalar"))


@pytest.mark.parametrize("flags", [[], ["--maxiter", "7"],
    ["--input", "input.custom", "--wout", "source.nc", "--initial-coils", "initial.json",
     "--resolution", "4", "3", "21", "--grid", "24", "20", "--coil-fit-maxiter", "3",
     "--device", "cpu", "--no-plots", "--no-movie", "--ftol", "1e-14", "--accepted-steps", "2"],
    ["--coils", "fitted.json", "--plots", "--movie"]])
def test_matching_options_and_defaults(entries, flags):
    fixed, free = (vars(entry.parse_args(flags)) for entry in entries)
    fixed.pop("output")
    free.pop("output")
    assert fixed == free
    for key in ("input", "wout", "coils", "initial_coils"):
        assert fixed[key] is None or fixed[key].is_absolute()


@pytest.mark.parametrize("flags", [["--wout", "alone.nc"],
    ["--coils", "a.json", "--initial-coils", "b.json"], ["--coil-fit-maxiter", "0"],
    ["--resolution", "0", "2", "11"], ["--ftol", "nan"]])
def test_matching_validation(entries, flags):
    for entry in entries:
        with pytest.raises(SystemExit):
            entry.parse_args(flags)


@pytest.mark.parametrize('flag', ['--seed', '--qualification'])
def test_saved_start_uses_its_coils_and_configuration(entries, tmp_path, flag):
    _, free = entries
    report = tmp_path/'start.json'
    report.write_text(json.dumps({'configuration': dict(input=str(free.INPUT), wout=None,
        resolution=[8, 8, 51], grid=[64, 64], ftol=1e-15, device='cpu')}))
    args = free.parse_args([flag, str(report)])
    assert args.coils is None  # default fitted coils cannot override the bundle
    assert args.resolution == [8, 8, 51] and args.device == 'cpu'
    assert free.parse_args([flag, str(report), '--coils', 'explicit.json']).coils.name == 'explicit.json'
    with pytest.raises(SystemExit):
        free.parse_args([flag, str(report), '--initial-coils', 'new.json'])
    with pytest.raises(SystemExit):
        free.parse_args(['--seed', str(report), '--qualification', str(report)])


def test_seed_manifest_authenticates_case_and_artifacts(entries, tmp_path):
    import hashlib
    _, free = entries
    paths = {name: tmp_path/(name+'.data') for name in ('coils', 'checkpoint')}
    for name, path in paths.items():
        path.write_text(name)
    contract = {'physics': 'vacuum', 'code': 'current'}
    manifest = dict(schema='vmex.single-stage-seed/v1', accepted_step=0, contract=contract,
        artifacts={name: dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                   for name, path in paths.items()})
    file = tmp_path/'seed.json'
    file.write_text(json.dumps(manifest))
    report, assets = free.common.read_seed(file, contract=contract)
    assert assets == paths and 'passed' not in report
    with pytest.raises(ValueError, match='differs from current'):
        free.common.read_seed(file, contract={'physics': 'changed'})
    paths['checkpoint'].write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        free.common.read_seed(file, contract=contract)
    manifest['artifacts']['checkpoint']['file'] = '../checkpoint'
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='beside the manifest'):
        free.common.read_seed(file, contract=contract)


def test_dry_runs_and_explicit_input_defaults(entries, tmp_path, capsys, monkeypatch):
    for entry in entries:
        output = tmp_path / Path(entry.__file__).stem
        assert entry.main(["--dry-run", "--output", str(output)]) == 0
        config = json.loads(capsys.readouterr().out)
        assert config["optimizer"] == "SLSQP"
        assert config["arguments"]["resolution"] == [8, 8, 51]
        assert config["arguments"]["ftol"] == 1e-15
        assert not output.exists()
        custom = entry.parse_args(["--input", "input.custom"])
        assert custom.resolution is None and custom.grid is None
        monkeypatch.setattr(entry, "COILS", Path("default.json"))
        assert entry.parse_args(["--initial-coils", "initial.json"]).coils is None


def test_fixed_run_restores_working_directory_on_failure(entries, tmp_path, monkeypatch):
    fixed, _ = entries
    old = Path.cwd()
    output = tmp_path / "failed"
    # Restore environment changes made by main after this test.
    import os
    for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH", "JAX_ENABLE_X64",
                "JAX_PLATFORMS", "VMEX_COMPILATION_CACHE", "JAX_ENABLE_COMPILATION_CACHE", "MPLBACKEND"):
        monkeypatch.setenv(key, os.environ.get(key, ""))

    def fail(args):
        assert Path.cwd() == output
        assert args.input == old / "input.custom"
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(fixed, "build_problem", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        fixed.main(["--device", "cpu", "--input", "input.custom", "--output", str(output)])
    assert Path.cwd() == old


@pytest.mark.parametrize("mode", ["generated", "initial-coils", "coils"])
def test_fixed_file_workflow_real_coil_fit(entries, tmp_path, monkeypatch, mode):
    """Run the fixed setup/fit, replacing only the equilibrium/optimizer entry."""
    pytest.importorskip("essos")
    import jax.numpy as jnp
    import vmex as vj
    from vmex import optimize as opt
    import inspect
    from vmex.core.optimize import make_problem
    from essos.coils import Coils

    fixed, free = entries
    for entry in entries:
        for name, value in dict(N_COILS=1, COIL_ORDER=1, N_SEGMENTS=8, NPHI=8, NTHETA=8).items():
            monkeypatch.setattr(entry, name, value)
    monkeypatch.delenv("VMEX_EXAMPLES_CI", raising=False)
    output = tmp_path / "fixed"
    output.mkdir()
    flags = ["--input", str(EXAMPLES.parent / "single_stage_support/data/input.rotating_ellipse"), "--device", "cpu",
             "--output", str(output), "--coil-fit-maxiter", "2"]
    inp, _ = fixed.common.load_input(fixed.parse_args(flags))
    coil_file = tmp_path / "coils.json"
    seed_coils = fixed.common.initial_coils(inp, None, parameters=vars(fixed))
    if mode != "generated":
        # Preserve user currents, including ones that differ from the default.
        seed_coils = Coils(seed_coils.curves, jnp.array([314159.]))
        seed_coils.to_json(str(coil_file))
        flags.extend([f"--{mode}", str(coil_file)])
    seed = object()
    captured = []
    real_loader = fixed.common.load_input
    monkeypatch.setattr(fixed.common, "load_input", lambda args, **kw: (real_loader(args)[0], seed))
    def boundary(x):
        return (jnp.asarray(inp.rbc).at[inp.ntor, 1].add(x[0]),
                jnp.asarray(inp.zbs).at[inp.ntor, 1].add(x[1]))

    fake_problem = SimpleNamespace(x0=np.zeros(2), scales=np.ones(2), dof_names=("r", "z"),
                                   metadata={}, boundary_from_x=boundary)
    fake_problem.with_accepted_state = lambda: fake_problem

    def plasma(inp, terms, **kw):
        inspect.signature(make_problem).bind(inp, **kw)
        captured.append(kw)
        return fake_problem

    monkeypatch.setattr(opt.VmecProblem, "from_loss", plasma)
    monkeypatch.setattr(opt.VmecProblem, "from_tuples", plasma)

    class FitFinished(Exception):
        pass

    def stop_at_optimizer(x, **kw):
        np.testing.assert_array_equal(x[:2], [0., 0.])  # Stage two freezes the boundary.
        raise FitFinished

    monkeypatch.setattr(vj.FunctionProblem, "from_functions", stop_at_optimizer)
    monkeypatch.chdir(output)
    with pytest.raises(FitFinished):
        fixed.run(fixed.parse_args(flags))
    assert len(captured) == 2 and all(call["restart_from"] is seed for call in captured)
    fitted = Coils.from_json(str(output / "coils.stage2.json"))
    np.testing.assert_array_equal(fitted.dofs_currents_raw, seed_coils.dofs_currents_raw)
    report = json.loads((output / "stage_two.json").read_text())
    assert report["reused"] == (mode == "coils")
    assert report["objective"] <= report["initial_objective"] + 1e-10
    if mode == "coils":
        np.testing.assert_array_equal(fitted.curves.dofs, seed_coils.curves.dofs)
        assert report["iterations"] == 0

    # The free entry must produce the same stage-two geometry from the same files.
    free_output = tmp_path / "free"
    free_output.mkdir()
    monkeypatch.setattr(opt.FreeBoundaryProblem, "from_loss", lambda *a, **kw:
        SimpleNamespace(accepted_step=0, enable_root_polishing=lambda **kw: None))
    args = free.parse_args(flags)
    args.output = free_output
    stage = free.build_problem(args)
    # Scalar sums and squared residual vectors differ in floating-point reduction order.
    np.testing.assert_allclose(stage.coils.curves.dofs, fitted.curves.dofs, rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(stage.coils.dofs_currents_raw, fitted.dofs_currents_raw)


@pytest.mark.parametrize("folder", ["single-stage-benchmarks", "coil-constraints-benchmarks"])
def test_production_and_qualification_are_separate(folder, tmp_path, monkeypatch, capsys):
    """Both fixed entries use one builder, and qualification never optimizes."""
    import os
    import sys

    directory = EXAMPLES.parent / folder
    monkeypatch.syspath_prepend(str(directory))
    for name in ("parameters", "single_stage_optimization_scalar", "_scalar_constraints", "_coil_constraints"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("single_stage_optimization_scalar",
        directory / "single_stage_optimization_scalar.py")
    entry = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, entry)
    spec.loader.exec_module(entry)
    from single_stage_support import verification

    assert entry.fixed is load("single_stage_optimization_scalar").fixed
    assert entry.main(["--dry-run"]) == 0
    assert verification.main(entry, ["--dry-run"]) == 0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        entry.parse_args(["--check-gradients"])

    for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH", "JAX_ENABLE_X64",
                "JAX_PLATFORMS", "VMEX_COMPILATION_CACHE", "JAX_ENABLE_COMPILATION_CACHE", "MPLBACKEND"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    calls, stage = [], object()
    def build(args):
        assert Path.cwd() == args.output
        (args.output / "provenance.json").write_text("{}")
        calls.append("build")
        return stage
    def check(actual, output):
        assert actual is stage
        calls.append("derivatives")
        return {"passed": True}
    monkeypatch.setattr(entry, "build_problem", build)
    monkeypatch.setattr(entry, "run_optimizer", lambda actual, args: calls.append("optimize") or object())
    monkeypatch.setattr(entry, "verify_endpoint", lambda *a: calls.append("endpoint") or 0)
    monkeypatch.setattr(verification, "verify_problem", check)
    cwd = Path.cwd()
    assert entry.main(["--output", str(tmp_path / "production")]) == 0
    assert calls == ["build", "optimize", "endpoint"] and Path.cwd() == cwd
    calls.clear()
    assert verification.main(entry, ["--output", str(tmp_path / "qualification")]) == 0
    assert calls == ["build", "derivatives"] and Path.cwd() == cwd


def test_fixed_qualification_rejects_wrong_and_nonfinite_rows(tmp_path):
    from single_stage_support.verification import check_direction
    matrix = np.array([[2., -.5], [1., 3.], [0., 0.]])
    point, direction = np.zeros(2), np.array([.3, -.7])
    analytic = matrix @ direction
    assert check_direction(lambda x: matrix @ x, analytic, point, direction, output=tmp_path)["passed"]
    assert not check_direction(lambda x: matrix @ x, 2*analytic, point, direction, output=tmp_path)["passed"]
    assert not check_direction(lambda x: np.full(3, np.nan), analytic, point, direction, output=tmp_path)["passed"]


def test_coil_free_contract_tracks_bounds_and_uses_shared_input(monkeypatch, tmp_path):
    pytest.importorskip("essos")
    import sys
    directory = EXAMPLES.parent / "coil-constraints-benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    for name in ("parameters", "_coil_constraints", "_coil_resolution"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("coil_free_case",
        directory / "free_boundary_single_stage_optimization_scalar.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    args = entry.parse_args(["--device", "cpu", "--output", str(tmp_path)])
    configured = []
    entry.configure_solver(SimpleNamespace(enable_matrix_free=lambda **options: configured.append(options)), args)
    assert configured[0]['refresh_horizon'] == 10
    assert configured[0]['refresh_max_steps'] == args.accepted_steps
    original = entry.qualification_contract(args)
    monkeypatch.setattr(entry.P, "ACCEPTED_STEPS", 200)
    monkeypatch.setattr(entry, "ACCEPTED_STEPS", 200)
    assert entry.qualification_contract(args) == original
    monkeypatch.setattr(entry.P, "MSC_LIMIT", 4.)
    assert entry.qualification_contract(args) != original

    def load_input(actual, *, restore_state):
        assert actual is args and restore_state
        raise RuntimeError("shared input reached")
    monkeypatch.setattr(entry.common, "load_input", load_input)
    with pytest.raises(RuntimeError, match="shared input reached"):
        entry.build_problem(args)


def test_constraint_settings_do_not_leak_between_cases(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES.parent))
    from single_stage_support.constraints import PhysicalConstraints

    first = PhysicalConstraints(IOTA_FLOOR=.19, RADIUS_TARGET=1.)
    values = np.array([.2, 1.])
    before = np.asarray(first.inequalities(values))
    second = PhysicalConstraints(IOTA_FLOOR=.3, RADIUS_TARGET=1.2)
    assert float(second.inequalities(values)[0]) < 0
    np.testing.assert_array_equal(first.inequalities(values), before)


def test_shared_implementation_changes_invalidate_qualification(entries, monkeypatch, tmp_path):
    pytest.importorskip("essos")
    _, entry = entries
    args = entry.parse_args(["--device", "cpu"])
    shared = tmp_path / "free.py"
    shared.write_bytes(Path(entry.free.__file__).read_bytes())
    monkeypatch.setattr(entry.free, "__file__", str(shared))
    original = entry.qualification_contract(args)
    shared.write_text(shared.read_text() + "\nSHARED_IMPLEMENTATION_REVISION = 2\n")
    assert entry.qualification_contract(args) != original


def test_shared_free_builder_preserves_hard_constraint_objective(monkeypatch, tmp_path):
    """Exercise the actual builder with analytic plasma quantities and real coils."""
    pytest.importorskip("essos")
    import jax
    import jax.numpy as jnp
    import sys
    from vmex import optimize as opt

    directory = EXAMPLES.parent / "coil-constraints-benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    for name in ("parameters", "_coil_constraints"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("hard_coil_entry",
        directory / "free_boundary_single_stage_optimization_scalar.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    args = entry.parse_args(["--device", "cpu", "--output", str(tmp_path)])
    inp, _ = entry.common.load_input(args)
    seed = object()
    monkeypatch.setattr(entry.common, "load_input", lambda *a, **k: (inp, seed))
    monkeypatch.setattr(opt, "QuasisymmetryRatioResidual", lambda *a:
        SimpleNamespace(residuals_state=lambda state, runtime: state[:2]))
    monkeypatch.setattr(opt, "aspect_ratio", lambda state, runtime: state[2])
    monkeypatch.setattr(opt, "min_abs_iota", lambda state, runtime: state[3])
    # The objective still constructs the moving surface before its empty coil
    # penalty; only the equilibrium-to-surface conversion is synthetic here.
    monkeypatch.setattr(opt, "boundary_from_state", lambda *a: (inp.rbc, inp.zbs, None, None))
    captured = {}
    problem = SimpleNamespace(accepted_step=0, accepted=SimpleNamespace(root_residual_norm=1e-13),
        solver_info={}, enable_root_polishing=lambda **kw: captured.update(polishing=kw))
    def from_loss(inp, loss, **kwargs):
        captured.update(loss=loss, **kwargs)
        return problem
    monkeypatch.setattr(opt.FreeBoundaryProblem, "from_loss", from_loss)
    stage = entry.build_problem(args)
    assert captured["restart_from"] is seed
    assert len(captured["coil_quantities"]) == 1
    state = jnp.array([2., .3, 5.2, .18])
    value, derivative = jax.value_and_grad(lambda s: captured["loss"](s, None, stage.coils))(state)
    np.testing.assert_allclose(value, .5*(4.+.09+.04+.001), rtol=1e-13)
    np.testing.assert_allclose(derivative, [2., .3, .2, -.1], rtol=1e-12, atol=1e-13)
    assert stage.coil_costs(stage.coils, None).size == 0
    values = np.array([.1905, .991, .201])
    np.testing.assert_allclose(stage.inequalities(values), [0., 0., 1.8, 0.], atol=1e-13)
    h = 1e-6
    fd = np.column_stack([(stage.inequalities(values+h*d)-stage.inequalities(values-h*d))/(2*h)
                          for d in np.eye(3)])
    np.testing.assert_allclose(fd, stage.constraint_transform, rtol=1e-9, atol=1e-9)

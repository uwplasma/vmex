"""Run the documented examples at reduced, deterministic CI budgets.

Every test sets ``VMEX_EXAMPLES_CI=1`` and requires a clean exit, physics
progress where applicable, and the documented output artifacts.  Most scripts
read the variable to shrink their work; a few have no smoke path and run as
shipped, among them ``vmex_get_B_gradB.py`` and ``vmex_get_B_outside_plasma.py``,
whose cost is XLA compilation that a coarser equilibrium does not reduce.
Commented optional objective terms are exercised by their physics/AD unit tests;
this module keeps their example wiring explicit.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("netCDF4")

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
DATA_DIR = EXAMPLES / "data"

_COST_RE = re.compile(r"^\s*\d+\s+\d+\s+([0-9.eE+-]+)", re.MULTILINE)
_SCALAR_COST_RE = re.compile(
    r"optimizer scalar cost:\s*([0-9.eE+-]+)\s*->\s*([0-9.eE+-]+)")

ESSOS_COIL_EXAMPLES = (
    EXAMPLES / "take_free_boundary_gradients.py",
    EXAMPLES / "vmex_fixed_free_boundary_comparison.py",
    EXAMPLES / "vmex_get_B_outside_plasma.py",
    EXAMPLES / "vmex_fieldline_tracing_vacuum.py",
    EXAMPLES / "vmex_fieldline_tracing_finite_beta.py",
    EXAMPLES / "optimization" / "single_stage_optimization.py",
    EXAMPLES / "optimization" / "single_stage_optimization_augmented_lagrangian.py",
    EXAMPLES / "optimization" / "single_stage_optimization_least_squares.py",
    EXAMPLES / "optimization" / "single_stage_optimization_finite_beta.py",
    EXAMPLES / "optimization" / "single_stage_free_boundary_optimization.py",
    EXAMPLES / "optimization" / "single_stage_free_boundary_optimization_finite_beta.py",
)


def test_released_essos_reads_bundled_coil_fixtures() -> None:
    """The compact coil files retain the schema supported by ESSOS 0.16."""
    pytest.importorskip("essos")
    from essos.coils import Coils

    if hasattr(Coils, "from_json"):
        load = Coils.from_json
    else:
        from essos.coils import Coils_from_json

        load = Coils_from_json
    for path in sorted(DATA_DIR.glob("ESSOS_biot_savart_*.json")):
        coils = load(str(path))
        assert np.all(np.isfinite(np.asarray(coils.gamma)))
        assert np.all(np.isfinite(np.asarray(coils.currents)))


def test_coil_examples_need_only_the_pinned_essos_release() -> None:
    """Every coil example runs on the ESSOS the ``coils`` extra installs.

    These examples used to raise with a ``pip install essos @ git+...`` line
    because the API they need (uwplasma/ESSOS#58) was unreleased. ESSOS 0.17
    carries it, the extra pins that floor, and the instruction is gone -- so
    what has to stay true is that the floor and the API agree.
    """
    import tomllib
    from importlib.metadata import version

    from packaging.version import Version

    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    coils = pyproject["project"]["optional-dependencies"]["coils"]
    assert coils == ["essos>=0.17"], coils

    for script in ESSOS_COIL_EXAMPLES:
        text = script.read_text()
        assert "git+https://github.com/uwplasma/ESSOS" not in text, script.name

    essos = pytest.importorskip("essos")
    assert Version(version("essos")) >= Version("0.17"), version("essos")
    from essos.coils import Coils
    from essos.dynamics import LevelsetStoppingCriterion, trace_field_lines
    from essos.objective_functions import (
        loss_coil_separation,
        loss_coil_surface_distance,
    )
    from essos.surfaces import surfacerzfourier_from_boundary

    del (essos, LevelsetStoppingCriterion, trace_field_lines,
         loss_coil_separation, loss_coil_surface_distance,
         surfacerzfourier_from_boundary)
    for name in ("from_json", "with_dofs", "dof_names"):
        assert hasattr(Coils, name), name


#: Shipped examples that no test RUNS, each with the reason it is exempt.
#: The QH, QI and QP entries drive the same code as a tested QA sibling on a
#: different symmetry class, so the driver is covered and only the deck is not.
#: Keeping the list explicit is what makes the gap reviewable: a new example
#: that nothing runs fails the guard below until it is either run or listed
#: here on purpose.  ``EXECUTED_EXAMPLES`` is the other half of the partition;
#: between them they must name every shipped example exactly once.
UNTESTED_EXAMPLES = {
    "examples/mirror/pleiades_mirror_reference.py": "needs an external Pleiades checkout",
    "examples/mirror/stellarator_mirror_hybrid.py": "mirror hybrid, covered by tests/mirror",
    "examples/optimization/QH_optimization_finite_beta_scalar.py": "QA sibling is tested",
    "examples/optimization/QH_optimization_scalar.py": "QA sibling is tested",
    "examples/optimization/QI_optimization_finite_beta_scalar.py": "QA sibling is tested",
    "examples/optimization/QI_optimization_scalar.py": "QA sibling is tested",
    "examples/optimization/QP_optimization_finite_beta_scalar.py": "QA sibling is tested",
    "examples/optimization/QP_optimization_scalar.py": "QA sibling is tested",
    "examples/optimization/QP_optimization_scipy.py": "QA sibling is tested",
    "examples/optimization/QA_maxJ_continuation.py": "covered by the source test above; a run is the QI sibling's",
    "examples/optimization/QA_optimization_DMerc_vacuum.py": "covered by the source test above; its certificate lane has no smoke path",
    "examples/optimization/QA_optimization_global.py": "covered by the source test above; basin hopping has no meaningful smoke budget",
    "examples/optimization/stellarator_asymmetry/QA_optimization_finite_beta.py": "asymmetric variants share the symmetric drivers",
    "examples/optimization/stellarator_asymmetry/QH_optimization_finite_beta.py": "asymmetric variants share the symmetric drivers",
    "examples/optimization/stellarator_asymmetry/QI_optimization_finite_beta.py": "asymmetric variants share the symmetric drivers",
    "examples/optimization/stellarator_asymmetry/QP_optimization_finite_beta.py": "asymmetric variants share the symmetric drivers",
    "examples/plot_optimized_families.py": "plots families produced by the tested optimization examples",
}


#: Shipped examples that a test in this module actually executes.  This is a
#: declaration, not a search: ``_run_example`` refuses to run a script that is
#: not listed, and the guard below requires this set and ``UNTESTED_EXAMPLES``
#: to partition the shipped examples exactly.
#:
#: The previous guard searched the test sources for each example's *basename*,
#: which silently accepted any example whose name another file shares.  That
#: masked all four ``optimization/stellarator_asymmetry`` vacuum scripts behind
#: their symmetric namesakes: none of them had ever run, and
#: ``QH_optimization.py`` exited non-zero when it finally did.
EXECUTED_EXAMPLES = {
    "examples/epsilon_effective.py",
    "examples/finite_beta_scan.py",
    "examples/fixed_boundary_run.py",
    "examples/force_balance_polishing.py",
    "examples/free_boundary_beta_scan.py",
    "examples/free_boundary_essos_coils.py",
    "examples/free_boundary_mgrid.py",
    "examples/free_boundary_phiedge.py",
    "examples/hot_restart_scan.py",
    "examples/mirror/mirror_fixed_boundary_nonaxisymmetric.py",
    "examples/mirror/mirror_free_boundary_beta_scan.py",
    "examples/mirror/qi_mirror_hybrid_fourier_vs_bspline.py",
    "examples/optimization/QA_optimization.py",
    "examples/optimization/QA_optimization_ballooning.py",
    "examples/optimization/QA_optimization_bootstrap.py",
    "examples/optimization/QA_optimization_finite_beta.py",
    "examples/optimization/QA_optimization_finite_beta_scalar.py",
    "examples/optimization/QA_optimization_scalar.py",
    "examples/optimization/QA_optimization_scipy.py",
    "examples/optimization/QH_optimization.py",
    "examples/optimization/QH_optimization_bootstrap.py",
    "examples/optimization/QI_maxJ_continuation.py",
    "examples/optimization/QI_optimization.py",
    "examples/optimization/QI_optimization_bootstrap.py",
    "examples/optimization/QI_optimization_jaxopt.py",
    "examples/optimization/QI_optimization_optax.py",
    "examples/optimization/QI_optimization_scipy.py",
    "examples/optimization/QP_optimization.py",
    "examples/optimization/omnigenity_epsilon_gammac_maxj.py",
    "examples/optimization/single_stage_free_boundary_optimization.py",
    "examples/optimization/single_stage_free_boundary_optimization_finite_beta.py",
    "examples/optimization/single_stage_optimization.py",
    "examples/optimization/single_stage_optimization_augmented_lagrangian.py",
    "examples/optimization/single_stage_optimization_finite_beta.py",
    "examples/optimization/single_stage_optimization_least_squares.py",
    "examples/optimization/stellarator_asymmetry/QA_optimization.py",
    "examples/optimization/stellarator_asymmetry/QH_optimization.py",
    "examples/optimization/stellarator_asymmetry/QI_optimization.py",
    "examples/optimization/stellarator_asymmetry/QP_optimization.py",
    "examples/parallel_ensemble_scan.py",
    "examples/plot_and_boozer.py",
    "examples/profiles_power_and_spline.py",
    "examples/run_from_json.py",
    "examples/take_fixed_boundary_gradients.py",
    "examples/take_free_boundary_gradients.py",
    "examples/vmex_essos_workflow.py",
    "examples/vmex_fieldline_tracing_finite_beta.py",
    "examples/vmex_fieldline_tracing_vacuum.py",
    "examples/vmex_fixed_free_boundary_comparison.py",
    "examples/vmex_get_B_gradB.py",
    "examples/vmex_get_B_outside_plasma.py",
}


def _shipped_examples() -> list[Path]:
    return sorted(p for p in EXAMPLES.rglob("*.py") if not p.name.startswith("_"))


def test_every_example_parses() -> None:
    """Every shipped example compiles.

    Cheap, and it catches the class of damage an edit across many examples can
    do -- a removed import guard, a stranded ``except`` -- without running any
    of them.
    """
    for script in _shipped_examples():
        try:
            ast.parse(script.read_text(), filename=str(script))
        except SyntaxError as error:  # pragma: no cover - the failure is the point
            raise AssertionError(f"{script.relative_to(REPO)}: {error}") from error


def _smoke_block_clobbers(tree: ast.Module) -> list[str]:
    """Values a ``ci_smoke`` block sets that a later statement discards.

    Two shapes count.  ``X = replace(X, kw=...)`` inside the block followed by
    an unconditional ``X = replace(X, kw=...)`` re-supplying the same keyword,
    and a smoke-assigned name rebound later by an expression that never reads
    it.  A later ``replace`` naming *other* keywords carries the smoke value
    forward and is correct, so it is not reported.
    """
    def assigned(node):
        names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                names += [e.id for e in target.elts if isinstance(e, ast.Name)]
        return names

    def replace_keywords(node):
        value = node.value
        if not isinstance(value, ast.Call):
            return set()
        function = value.func
        name = (function.attr if isinstance(function, ast.Attribute)
                else getattr(function, "id", ""))
        if name != "replace":
            return set()
        return {k.arg for k in value.keywords if k.arg}

    smoke_names: dict[str, int] = {}
    smoke_keywords: dict[tuple[str, str], int] = {}
    problems: list[str] = []
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id in ("ci_smoke", "CI", "ci")):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assign):
                    keywords = replace_keywords(inner)
                    for name in assigned(inner):
                        smoke_names.setdefault(name, inner.lineno)
                        for keyword in keywords:
                            smoke_keywords.setdefault((name, keyword), inner.lineno)
            continue
        if not isinstance(node, ast.Assign):
            continue
        keywords = replace_keywords(node)
        for name in assigned(node):
            for keyword in keywords:
                if (name, keyword) in smoke_keywords:
                    problems.append(
                        f"line {smoke_keywords[(name, keyword)]} sets {name}.{keyword} "
                        f"under the smoke switch; line {node.lineno} supplies it again")
            reads = any(isinstance(n, ast.Name) and n.id == name
                        and isinstance(n.ctx, ast.Load)
                        for n in ast.walk(node.value))
            if name in smoke_names and not reads:
                problems.append(
                    f"line {smoke_names[name]} sets {name} under the smoke switch; "
                    f"line {node.lineno} rebuilds it without reading it")
    return problems


def test_smoke_budgets_are_not_silently_discarded() -> None:
    """A ``ci_smoke`` block must not be undone by a later unconditional line.

    ``stellarator_asymmetry/QA_optimization.py`` and ``QH_optimization.py``
    both reduced the radial grid under the switch and then overwrote
    ``ns_array``/``ftol_array``/``niter_array`` three statements later, so the
    smoke budget never applied.  Every trial hit the iteration cap, its
    residual came back non-finite, and least squares stopped at iteration 0 on
    the 1e6 sentinel while reporting ``gtol`` success -- QH then exited 1 in
    ``monitor.plot``.  The failure is invisible in review and silent at run
    time, so it is worth a parser.
    """
    problems = {}
    for script in _shipped_examples():
        found = _smoke_block_clobbers(ast.parse(script.read_text()))
        if found:
            problems[str(script.relative_to(REPO))] = found
    assert not problems, problems


def test_every_example_is_run_or_listed_as_untested() -> None:
    """Every shipped example is either run by a test or listed as exempt.

    The two sets must partition the shipped examples.  Matching on basenames
    is what let four never-run examples pass as covered, so this compares
    whole repository-relative paths and requires the sets to be disjoint.
    """
    shipped = {str(script.relative_to(REPO)) for script in _shipped_examples()}
    listed = set(UNTESTED_EXAMPLES)
    assert not (EXECUTED_EXAMPLES & listed), {
        "both run and listed as untested": sorted(EXECUTED_EXAMPLES & listed)}
    assert shipped == EXECUTED_EXAMPLES | listed, {
        "shipped but neither run nor listed": sorted(shipped - EXECUTED_EXAMPLES - listed),
        "named but not shipped": sorted((EXECUTED_EXAMPLES | listed) - shipped),
    }


def _run_example(script: Path, cwd: Path, timeout: int = 2400,
                 args: tuple[str, ...] = (), **extra_env: str) -> str:
    if script.is_relative_to(EXAMPLES):
        # Keeps EXECUTED_EXAMPLES honest from the other side: a new test that
        # runs an example fails here until the example is declared.
        assert str(script.relative_to(REPO)) in EXECUTED_EXAMPLES, (
            f"{script.relative_to(REPO)} is run by a test but is not listed in "
            "EXECUTED_EXAMPLES")
    env = dict(os.environ, VMEX_EXAMPLES_CI="1", **extra_env)
    env.pop("JAX_DISABLE_JIT", None)
    proc = subprocess.run(
        [sys.executable, str(script), *args], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"{script.name} failed (rc={proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-4000:]}")
    return proc.stdout


def _assert_cost_decreased(stdout: str, name: str) -> None:
    costs = [float(c) for c in _COST_RE.findall(stdout)]
    if len(costs) < 2:
        scalar_costs = _SCALAR_COST_RE.search(stdout)
        costs = [] if scalar_costs is None else [
            float(scalar_costs.group(1)), float(scalar_costs.group(2))]
    assert len(costs) >= 2, f"{name}: expected optimizer cost evidence, got {costs}"
    assert min(costs) < costs[0], (
        f"{name}: optimizer cost did not decrease: first {costs[0]:.6e}, "
        f"best {min(costs):.6e}")


def test_fixed_boundary_run(tmp_path):
    out = _run_example(EXAMPLES / "fixed_boundary_run.py", tmp_path, timeout=900)
    assert "converged = True" in out
    outdir = tmp_path / "output_fixed_boundary_run"
    assert (outdir / "wout_li383_low_res.nc").exists()
    assert (outdir / "li383_low_res_summary.png").exists()


def test_plot_and_boozer(tmp_path):
    out = _run_example(EXAMPLES / "plot_and_boozer.py", tmp_path, timeout=900)
    assert "converged = True" in out
    outdir = tmp_path / "output_plot_and_boozer"
    assert (outdir / "wout_li383_low_res.nc").exists()
    # every plot_wout figure kind is written unconditionally
    for suffix in (
        "summary", "surfaces", "modB", "stability", "boundary3d",
    ):
        assert (outdir / f"li383_low_res_{suffix}.png").exists()


def test_profiles_power_and_spline(tmp_path):
    out = _run_example(EXAMPLES / "profiles_power_and_spline.py", tmp_path, timeout=900)
    # both profile representations converge to the same equilibrium
    assert out.count("converged=True") == 2
    match = re.search(r"\|d aspect\| = ([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(1)) < 1e-3


def test_run_from_json(tmp_path):
    out = _run_example(EXAMPLES / "run_from_json.py", tmp_path, timeout=900)
    match = re.search(r"\|diff\|=([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(1)) < 1e-6
    assert (tmp_path / "output_run_from_json" / "circular_tokamak.json").exists()
    assert (tmp_path / "output_run_from_json" / "wout_circular_tokamak.nc").exists()


def test_epsilon_effective_example_bounds_the_neo_controls() -> None:
    """The example must stay responsive and must not chase a rational surface.

    ``max_rational_field_periods=0`` asks NEO_JAX for an unlimited exact
    rational correction, which on a near-rational surface does not finish in
    an example's budget; the script documents the bound and keeps it finite.
    """
    source = (EXAMPLES / "epsilon_effective.py").read_text()
    assert "epsilon_effective_from_wout" in source
    assert "max_rational_field_periods=100000" in source
    assert "input.LandremanPaul2021_QA_lowres" in source
    # a radial profile, not a single surface: the trend is the result
    assert "SURFACES = np.linspace(0.15, 0.95, 5)" in source
    assert "surfaces=SURFACES" in source and "config=NEO_CONFIG" in source


# NEO_JAX is the optional ``neoclassical`` extra, which no CI lane installs
# today, so this skips everywhere in CI; it is the on-demand check for the
# example (``pip install vmex[neoclassical]``, then RUN_FULL=1).
@pytest.mark.full  # ~3.5 min: one solve, then a Boozer transform per surface
def test_epsilon_effective_example(tmp_path):
    pytest.importorskip("neo_jax")
    out = _run_example(EXAMPLES / "epsilon_effective.py", tmp_path, timeout=1800)
    values = re.search(r"epsilon_eff\^\(3/2\) = \[(.+?)\]", out, re.S)
    assert values is not None, out
    profile = np.array([float(v) for v in values.group(1).split()])
    assert profile.size == 5 and np.all(np.isfinite(profile))
    # a QA ripple is small but nonzero, and rises toward the boundary
    assert np.all(profile > 0) and np.all(profile < 1e-2)
    assert profile[-1] > profile[0], f"eps_eff should rise outward: {profile}"
    assert (tmp_path / "epsilon_effective.png").stat().st_size > 10_000


def test_force_balance_polishing_example_refuses_an_uncertified_export() -> None:
    """The polish is only evidence if the example stops when it fails.

    ``solve_file`` returns a polished state whenever the deck asks for one;
    the certificate is the ``polish_report``. Exporting the dense-mesh WOUT
    without checking ``converged`` would ship an uncertified equilibrium as
    if it were polished, so keep both guards explicit.
    """
    deck = EXAMPLES / "data" / "input.shaped_tokamak_pressure_polished"
    assert "POLISH_FORCE_BALANCE = .TRUE." in deck.read_text()
    source = (EXAMPLES / "force_balance_polishing.py").read_text()
    assert "result.polished_state is None or result.polish_report is None" in source
    assert "if not report.converged:" in source
    for field in ("initial_normalized_l2", "final_normalized_l2",
                  "nonlinear_iterations", "termination_reason"):
        assert field in source


@pytest.mark.full  # nightly: ordinary solve + strong-force polish + 12 figures (~2 min)
def test_force_balance_polishing_example(tmp_path):
    out = _run_example(EXAMPLES / "force_balance_polishing.py", tmp_path, timeout=1200)
    assert "POLISH CERTIFIED" in out
    assert "independent strong-force certificate over s in [0.10, 0.99]:" in out
    for label in ("eps_F volume L2 (<= 2 by construction)", "<|F|> [N m^-3]",
                  "<|F|>/<|grad(B^2/2mu0)|>"):
        certificate = re.search(
            re.escape(label) + r"\s+([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
        assert certificate is not None, out
        initial, final = (float(g) for g in certificate.groups())
        assert np.isfinite([initial, final]).all() and 0 <= final < initial, (
            f"the polish must lower the reported {label}: {out}")
    outdir = tmp_path / "output_force_balance_polishing"
    for name in ("wout_shaped_tokamak_before_polish.nc",
                 "wout_shaped_tokamak_pressure_polished.nc"):
        assert (outdir / name).exists()
    # both stages plot, so the before/after comparison the docstring promises
    # is actually produced
    for stage, stem in (("before", "shaped_tokamak_before_polish"),
                        ("after", "shaped_tokamak_pressure_polished")):
        assert (outdir / stage / f"{stem}_summary.png").stat().st_size > 10_000
    # the fair comparison: both files on one mesh, certified the same way; the
    # near-axis error is where the polish gain lives
    assert "both WOUT files on ns = 129, read back and certified the same way" in out
    near_axis = re.search(r"rho < 0\.2\s+\[N m\^-3\]\s+([0-9.eE+-]+) => ([0-9.eE+-]+)", out)
    assert near_axis is not None, out
    initial, final = (float(g) for g in near_axis.groups())
    assert final < 0.1 * initial, out
    assert (outdir / "polish_before_after.webp").stat().st_size > 10_000


def test_hot_restart_scan(tmp_path):
    out = _run_example(EXAMPLES / "hot_restart_scan.py", tmp_path, timeout=900)
    base = re.search(r"cold base solve:\s*(\d+) iters", out)
    warm = [int(m) for m in re.findall(r"^\s*[0-9.]+\s+(\d+)\s+[0-9.]+\s+warm", out, re.M)]
    assert base is not None and int(base.group(1)) > 10, "base should need many iters"
    assert len(warm) == 5 and max(warm) <= 5, f"warm restarts should be cheap: {warm}"


def test_parallel_ensemble_scan(tmp_path):
    out = _run_example(EXAMPLES / "parallel_ensemble_scan.py", tmp_path, timeout=900)
    # the correctness contract: threaded ensemble is bit-identical to serial
    assert "max|state diff| vs serial = 0.0e+00" in out
    assert "iterations identical: True" in out
    # the strong-scaling table printed at least one worker row
    assert re.search(r"^\s*\d+\s+[0-9.]+\s+[0-9.]+x", out, re.M) is not None


@pytest.mark.full  # nightly: free-bdy NESTOR solve ~10s; parity already covered in shard-a
def test_free_boundary_mgrid(tmp_path):
    out = _run_example(EXAMPLES / "free_boundary_mgrid.py", tmp_path, timeout=900)
    assert "converged = True" in out
    assert (tmp_path / "output_free_boundary_mgrid" / "wout_cth_like_free_bdy.nc").exists()


@pytest.mark.full  # nightly: ~1 min (2 adjoint grads + 4 FD solves, subprocess cold-start)
def test_take_fixed_boundary_gradients(tmp_path):
    """Both implicit-adjoint gradients agree with central finite differences.

    Unlike the free boundary, a fixed-boundary re-solve follows the same path,
    so the difference quotient is a usable reference for the boundary
    coefficient and for phiedge alike.
    """
    out = _run_example(EXAMPLES / "take_fixed_boundary_gradients.py", tmp_path,
                       timeout=900)
    rels = [float(m) for m in re.findall(r"rel=([0-9.eE+-]+)", out)]
    assert len(rels) == 2, f"expected two AD-vs-FD checks, got {rels}"
    assert max(rels) < 1e-4, f"adjoint gradient disagrees with FD: rel={rels}"


@pytest.mark.full  # one free solve and both adjoint solvers
def test_take_free_boundary_gradients(tmp_path):
    pytest.importorskip("essos")
    out = _run_example(EXAMPLES / "take_free_boundary_gradients.py", tmp_path, timeout=900)
    # The certificate is the two independent adjoint solvers agreeing.  Both
    # linearize the Newton-anchored coupled root, so they agree to 1.1e-11 at
    # the smoke settings and 3.5e-10 at the shipped ones; before the anchor
    # the smoke settings (ftol 1e-7) left them 1.6e-02 apart.
    match = re.search(r"they differ by ([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(1)) < 1.0e-8


@pytest.mark.full  # independent finite-beta free solve, inner fixed solve, and VC field
def test_fixed_free_boundary_comparison(tmp_path):
    pytest.importorskip("essos")
    pytest.importorskip("virtual_casing_jax")
    out = _run_example(
        EXAMPLES / "vmex_fixed_free_boundary_comparison.py", tmp_path,
        timeout=900)
    field = re.search(r"Outer-region pointwise error: median=([0-9.eE+-]+)", out)
    surfaces = re.search(r"Common-surface RMS errors \[m\] = \[([^]]+)\]", out)
    # The 2.625%-beta production case gives a measurable outer-region
    # difference while retaining a converged common inner half-volume.
    assert field is not None and 3.0e-2 < float(field.group(1)) < 2.5e-1
    assert surfaces is not None and np.max(np.fromstring(surfaces.group(1), sep=" ")) < 1.5e-2
    assert (tmp_path / "vmex_fixed_free_boundary_comparison.png").stat().st_size > 10_000


def test_free_boundary_single_stage_examples_show_explicit_optimizer_contract():
    """The examples expose the solve, AD, and SciPy directly, from a checked seed."""
    fixed = (EXAMPLES / "optimization" / "single_stage_optimization.py").read_text()
    for name in ("single_stage_free_boundary_optimization.py",
                 "single_stage_free_boundary_optimization_finite_beta.py"):
        text = (EXAMPLES / "optimization" / name).read_text()
        assert "solve_free_boundary_implicit_status" in text
        assert "jax.value_and_grad" in text
        assert "FunctionProblem.from_functions" in text
        assert "minimize(free_problem.value_and_grad" in text
        assert "pack_boundary" not in text
        assert "mgrid file" in text
        # Coils fitted to the fixed-boundary seed, then checked by a free solve,
        # rather than assumed to confine it.
        assert "coil_fit = minimize(" in text and "toroidal_flux" in text
        assert "do not hold a converged free boundary" in text
        # The same problem as the fixed-boundary example, value for value.
        for name_ in ("NFP", "SEED_MINOR_RADIUS", "SEED_ELLIPSE", "IOTA_FLOOR", "ASPECT_LIMIT",
                      "COIL_SURFACE_DISTANCE_LIMIT", "COIL_DISTANCE_LIMIT", "CURVATURE_LIMIT",
                      "IOTA_CONSTRAINT", "ASPECT_CONSTRAINT", "NORMAL_FIELD_CONSTRAINT",
                      "CURVATURE_OBJECTIVE_LIMIT", "COIL_DISTANCE_CONSTRAINT",
                      "COIL_SURFACE_DISTANCE_CONSTRAINT", "N_COILS", "COIL_ORDER",
                      "COIL_MAJOR_RADIUS", "COIL_MINOR_RADIUS", "COIL_CURRENT", "N_SEGMENTS",
                      "LENGTH_TARGET", "LENGTH_WEIGHT", "CURVATURE_WEIGHT",
                      "COIL_DISTANCE_WEIGHT", "COIL_SURFACE_DISTANCE_WEIGHT",
                      "CONSTRAINT_WEIGHT", "PARAMETER_BOUND", "COIL_FIT_MAXITER"):
            line = re.search(rf"^{name_} = .*$", fixed, re.M)
            assert line is not None and re.search(
                rf"^{re.escape(line.group(0))}", text, re.M), (name, name_)


def test_global_optimization_example_exposes_optimizer_contract():
    """The global example keeps SciPy, exact gradients, and local polish visible."""
    text = (EXAMPLES / "optimization" / "QA_optimization_global.py").read_text()
    assert "basinhopping(value_and_gradient" in text
    assert '"method": "L-BFGS-B"' in text
    assert "least_squares(problem.residual" in text
    assert "ess_alpha=ESS_ALPHA" in text


def test_qa_optimization_keeps_explicit_least_squares_lane():
    """The canonical QA example keeps its residual/Jacobian tutorial."""
    text = (EXAMPLES / "optimization" / "QA_optimization.py").read_text()
    assert "VmecProblem.from_tuples" in text
    assert "compile_residual_and_jacobian" in text
    assert "least_squares(" in text
    assert "VmecProblem.from_loss" not in text
    assert "ess_alpha=ESS_ALPHA" in text


@pytest.mark.parametrize("case", ["QA", "QH", "QP", "QI"])
@pytest.mark.parametrize("finite_beta", [False, True])
def test_scalar_optimization_examples_expose_one_adjoint_lane(case, finite_beta):
    """All eight scalar examples keep the one-adjoint lane visible in the file.

    They used to import ``run_scalar_stage`` from ``_scalar_driver.py``, so the
    part that did the optimizing lived in a second file.  It is inlined now:
    each script is readable end to end, and these assertions are what keep the
    lane explicit rather than drifting back behind a helper.
    """
    suffix = "_finite_beta_scalar" if finite_beta else "_scalar"
    text = (EXAMPLES / "optimization" / f"{case}_optimization{suffix}.py").read_text()
    assert "_scalar_driver" not in text
    assert "objective_terms" in text
    assert "POLISH_FORCE_BALANCE = False" in text
    # one stem drives input., wout_ and the monitor files
    assert f'OUTPUT_NAME = "{case}_' in text
    assert 'to_indata(f"input.{OUTPUT_NAME}")' in text
    assert 'write_wout(f"wout_{OUTPUT_NAME}.nc"' in text
    # the scalar lane itself: one aggregate loss, differentiated once per step
    assert "VmecProblem.from_loss" in text
    assert "residuals_from_tuples" in text
    assert "0.5 * jnp.vdot(rows, rows)" in text
    assert "compile_value_and_gradient" in text
    assert 'method="L-BFGS-B"' in text
    if finite_beta:
        _assert_beta_from_toroidal_flux(text)
        assert "opt.mercier_stability_residual" in text
        assert "opt.glasser_stability_residual" in text
    else:
        assert "TARGET_BETA" not in text


def _assert_beta_from_toroidal_flux(text: str) -> None:
    """Beta is set once through PHIEDGE, never by recalibrating the pressure.

    A pressure calibrated at the seed's own aspect ratio contradicts the
    aspect-ratio target (at fixed flux beta scales as the aspect ratio to the
    power -4), and a per-stage recalibration loop hides that by moving the
    pressure under the optimizer.
    """
    assert "TARGET_BETA" in text
    assert "PRES_SCALE = " in text
    assert "closed_form_phiedge = np.pi * minor_radius**2" in text
    assert "phiedge=closed_form_phiedge * np.sqrt(" in text
    assert "input_minor_radius(" in text
    assert "pres_scale=inp.pres_scale" not in text  # no pressure rescaling
    assert "calibration =" not in text and "def calibrate" not in text
    assert "(opt.volume_average_beta, TARGET_BETA, " in text


def test_qa_finite_beta_least_squares_example_sets_beta_through_phiedge():
    text = (EXAMPLES / "optimization" / "QA_optimization_finite_beta.py").read_text()
    _assert_beta_from_toroidal_flux(text)
    assert "opt.mercier_stability_residual" in text
    assert "opt.glasser_stability_residual" in text
    assert "VmecProblem.from_tuples" in text
    assert "least_squares(" in text


@pytest.mark.parametrize("case", ["QA", "QH", "QP", "QI"])
@pytest.mark.parametrize("suffix", ["", "_finite_beta"])
def test_stellarator_asymmetry_examples_expose_all_boundary_families(case, suffix):
    """Keep all eight LASYM examples explicit without eight cold compiles in CI."""
    source = (EXAMPLES / "optimization" / "stellarator_asymmetry"
              / f"{case}_optimization{suffix}.py").read_text()
    assert "lasym=True" in source
    assert "rbs[inp.ntor + 1, 1]" in source and "zbc[inp.ntor + 1, 1]" in source
    assert "asymmetric boundary norm" in source
    assert "ess_alpha=ESS_ALPHA" in source
    if suffix:
        _assert_beta_from_toroidal_flux(source)


@pytest.mark.full  # nightly: four LASYM stages, twice the dofs of a symmetric one
@pytest.mark.parametrize("case", ["QA", "QH", "QI", "QP"])
def test_stellarator_asymmetry_vacuum_examples_run(case, tmp_path):
    """The four vacuum LASYM examples converge and descend.

    Until the coverage guard was made path-aware these had never run: each was
    masked by the symmetric example of the same basename.  ``QH`` exited 1 when
    it first did, because the smoke block's radial grid was overwritten by the
    unconditional ``replace`` below it, so every trial hit the iteration cap and
    returned the non-finite-residual sentinel.
    """
    if case == "QI":
        pytest.importorskip("booz_xform_jax")
    script = (EXAMPLES / "optimization" / "stellarator_asymmetry"
              / f"{case}_optimization.py")
    out = _run_example(script, tmp_path, timeout=1800)
    _assert_cost_decreased(out, f"LASYM {case}")
    # The point of the LASYM lane: the boundary must leave the symmetric subspace.
    norm = re.search(r"asymmetric boundary norm = ([0-9.eE+-]+)", out)
    assert norm is not None and float(norm.group(1)) > 0.0, out[-2000:]
    assert (tmp_path / f"input.{case}_LASYM_optimized").exists()
    assert (tmp_path / f"wout_{case}_LASYM_optimized.nc").exists()
    assert (tmp_path / f"{case}_LASYM_optimized_objectives.png").exists()


def test_qa_maxj_example_states_its_physical_scope():
    text = (EXAMPLES / "optimization" / "QA_maxJ_continuation.py").read_text()
    assert "maximum-J is incompatible with quasisymmetry near the magnetic axis" in text
    assert "input.minimal_seed_nfp" in text
    assert "opt.magnetic_well" in text


def test_combined_confinement_example_states_the_surrogate_policy():
    """Optimize GammaCSmooth, report hard values, promise no zero losses."""
    text = (EXAMPLES / "optimization" / "omnigenity_epsilon_gammac_maxj.py").read_text()
    assert "optimize the surrogate, report the hard values" in text
    assert "GammaCSmooth(" in text and "GammaC(" in text
    assert "does not promise" in text
    assert "normalized by its seed value" in text
    assert "VmecProblem.from_loss" in text  # one aggregate scalar adjoint
    assert "epsilon_eff unavailable" in text  # optional NEO_JAX states its absence


@pytest.mark.full  # one direct-coil free solve, coupled adjoint, and output solve (~2 min)
def test_vacuum_free_boundary_single_stage_optimization(tmp_path):
    pytest.importorskip("essos")
    out = _run_example(
        EXAMPLES / "optimization" / "single_stage_free_boundary_optimization.py",
        tmp_path, timeout=600)
    assert "no boundary variables and no mgrid file" in out
    assert re.search(r"\[seed\] QA total = [0-9.eE+-]+, .*free boundary within", out)
    assert re.search(r"\[final\] QA total = ([0-9.eE+-]+)", out)
    # Smoke mode exits 0 even when a target is missed, so the report must say so.
    assert re.search(r"Minimum \|iota\| = [0-9.]+ \(target >= [0-9.]+\)", out)
    summary = json.loads(
        (tmp_path / "single_stage_free_boundary_optimization_summary.json").read_text())
    assert summary["met"] == (not summary["unmet"])
    assert summary["met"] or "did NOT meet its stated targets" in out
    for name in ("wout_single_stage_free_boundary_optimized.nc",
                 "single_stage_free_boundary_optimization.png",
                 "single_stage_free_boundary_objectives.png"):
        assert (tmp_path / name).stat().st_size > 0


@pytest.mark.full
@pytest.mark.weekly  # the same derivative at 0.5% beta
def test_finite_beta_free_boundary_single_stage_optimization(tmp_path):
    pytest.importorskip("essos")
    out = _run_example(
        EXAMPLES / "optimization" /
        "single_stage_free_boundary_optimization_finite_beta.py",
        tmp_path, timeout=900)
    assert "True NESTOR free boundary + ESSOS at beta 0.50%" in out
    assert re.search(r"\[final\] QA total = [0-9.eE+-]+, beta = ([0-9.eE+-]+)", out)
    assert re.search(r"Volume-average beta = [0-9.]+% \(target 0.50%", out)
    summary = json.loads((tmp_path / (
        "single_stage_free_boundary_optimization_finite_beta_summary.json")).read_text())
    assert summary["met"] == (not summary["unmet"])
    assert summary["met"] or "did NOT meet its stated targets" in out
    for name in ("wout_single_stage_free_boundary_finite_beta_optimized.nc",
                 "single_stage_free_boundary_finite_beta_optimization.png",
                 "single_stage_free_boundary_finite_beta_objectives.png"):
        assert (tmp_path / name).stat().st_size > 0


@pytest.mark.full  # nightly: one NESTOR solve per pressure point (~40s)
def test_free_boundary_beta_scan(tmp_path):
    out = _run_example(EXAMPLES / "free_boundary_beta_scan.py", tmp_path, timeout=1200)
    betas = [float(b) for _, b in re.findall(
        r"^\s*([0-9.]+)\s+([0-9.eE+-]+)\s+[0-9.]+\s+\d+\s*$", out, re.M)]
    assert len(betas) == 3 and betas[-1] > 1e-2, f"beta should reach finite values: {betas}"


@pytest.mark.full
def test_mirror_fixed_boundary_nonaxisymmetric_example(tmp_path):
    import json
    _run_example(
        EXAMPLES / "mirror" / "mirror_fixed_boundary_nonaxisymmetric.py",
        tmp_path,
        timeout=1200,
    )
    outdir = tmp_path / "results" / "mirror_fixed_boundary_nonaxisymmetric"
    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["rotating_ellipse"]["status"] == "supported"
    assert summary["rotating_ellipse"]["variational_max"] < 1.0e-12
    assert summary["rotating_ellipse"]["strong_force_normalized_rms"] < 5.0e-2
    assert summary["rotating_ellipse"]["boundary_gradient_relative_error"] < 1.0e-4
    assert summary["rotating_ellipse"]["adjoint_relative_residual"] < 1.0e-8
    assert summary["straight_field_line"]["status"].startswith("paraxial")
    assert summary["straight_field_line"]["variational_max"] < 1.0e-12
    assert summary["straight_field_line"]["final_linear_residual"] < 1.0e-8
    assert summary["straight_field_line"]["linear_iterations"] < 1000
    # Paraxial benchmark: the unconstrained bulk force is clean and gated,
    # while the expected cut boundary layer dominates the all-volume and
    # end-collar norms (device-normalized all-volume above 0.1).
    assert summary["straight_field_line"]["strong_force_bulk_rms"] < 5.0e-2
    assert (
        summary["straight_field_line"]["strong_force_end_collar_rms"]
        > summary["straight_field_line"]["strong_force_bulk_rms"]
    )
    assert summary["straight_field_line"]["strong_force_device_normalized_rms"] > 0.1
    assert summary["straight_field_line"]["axial_flux_derivative_min"] > 4.49e-4
    for case in summary:
        for suffix in ("3d", "cross_sections", "modB", "summary"):
            assert (outdir / f"{case}_{suffix}.png").stat().st_size > 10_000


@pytest.mark.full
def test_qi_mirror_hybrid_example(tmp_path):
    """The QI-mirror hybrid script runs end to end on its smoke budget.

    tests/mirror covers the library calls; this covers the script, which
    once imported a private helper after it was made public and crashed
    before its first solve.
    """
    import json
    _run_example(EXAMPLES / "mirror" / "qi_mirror_hybrid_fourier_vs_bspline.py",
                 tmp_path, timeout=900)
    outdir = tmp_path / "results" / "qi_mirror_hybrid"
    summary = json.loads((outdir / "summary.json").read_text())
    assert len(summary["cut_phi"]) == 4
    assert summary["splice_closure"] < 1.0e-12
    assert summary["hybrid_divergence_rms"] < 1.0e-10
    assert (outdir / "qi_mirror_hybrid.webp").stat().st_size > 10_000


@pytest.mark.full
def test_mirror_free_boundary_beta_scan_example(tmp_path):
    import json
    pytest.importorskip("essos")
    _run_example(EXAMPLES / "mirror" / "mirror_free_boundary_beta_scan.py", tmp_path, timeout=2400)
    outdir = tmp_path / "results" / "mirror_free_boundary_beta_scan"
    summary = json.loads((outdir / "beta_scan_summary.json").read_text())
    assert [row["requested_beta"] for row in summary] == [0.0, 0.10, 0.25, 0.50]
    assert [row["supported_lane"] for row in summary] == [True, True, False, False]
    assert summary[-1]["center_radius"] > summary[0]["center_radius"]
    assert summary[-1]["center_axis_field"] < summary[0]["center_axis_field"]
    for beta in ("000p0", "010p0", "050p0"):
        for suffix in ("3d", "cross_sections", "modB", "summary"):
            assert (outdir / f"mirror_beta_{beta}pct_{suffix}.png").stat().st_size > 10_000


@pytest.mark.full  # nightly: free-bdy NESTOR solve with direct-coil Biot-Savart (~90s)
def test_free_boundary_essos_coils(tmp_path):
    # The example needs only ``essos.coils`` (loading) + ``essos.fields.BiotSavart``
    # (tabulation) — present in every released ESSOS >= 0.16, with an in-example
    # fallback to the legacy ``Coils_from_json`` loader.  Under VMEX_EXAMPLES_CI=1
    # the script solves a single coarse beta point (ns=16), keeping this bounded.
    pytest.importorskip("essos.coils")
    pytest.importorskip("essos.fields")
    out = _run_example(EXAMPLES / "free_boundary_essos_coils.py", tmp_path, timeout=900)
    # table rows: nominal%  PRES_SCALE  actual-beta%  iters  fsq  aspect  axis-R
    rows = re.findall(r"^\s*([0-9.]+)%\s+([0-9.]+)\s+([0-9.]+)%\s+\d+\s+([0-9.eE+-]+)",
                      out, re.M)
    assert len(rows) == 1, f"CI mode should solve exactly one beta point:\n{out}"
    nominal, _pres_scale, actual, fsq = (float(x) for x in rows[0])
    assert abs(actual - nominal) <= 0.15, (
        f"actual betatotal {actual}% not calibrated to nominal {nominal}%")
    assert fsq < 1e-7, f"free-boundary point should converge, fsq={fsq}"


@pytest.mark.full  # nightly: PHIEDGE root solve on the ESSOS QA coils
def test_free_boundary_phiedge(tmp_path):
    pytest.importorskip("essos.coils")
    pytest.importorskip("essos.fields")
    out = _run_example(EXAMPLES / "free_boundary_phiedge.py", tmp_path, timeout=900)
    error = float(re.search(r"relative error ([0-9.eE+-]+)", out).group(1))
    assert error <= 2e-3, out  # cold re-solve at the returned PHIEDGE; RTOL = 1e-3 in CI mode


@pytest.mark.full  # nightly: fixed + free-boundary solve either side of the seam (~100s)
def test_vmex_essos_workflow(tmp_path):
    # Both interop seams in one script: vj.essos_vmec_field (equilibrium ->
    # essos.fields.Vmec) and vj.MgridField.from_coils (ESSOS coils -> external
    # field).  Released-ESSOS surface only, so this runs against any ESSOS.
    pytest.importorskip("essos.coils")
    pytest.importorskip("essos.dynamics")
    pytest.importorskip("essos.fields")
    out = _run_example(EXAMPLES / "vmex_essos_workflow.py", tmp_path, timeout=900)
    assert out.count("converged = True") == 2, out

    # The wout tables cross unchanged: ESSOS' to_xyz rebuilds the LCFS vmex
    # wrote, so this is a machine-precision identity, not a tolerance.
    transfers = [float(v) for v in re.findall(r"LCFS transfer error ([0-9.eE+-]+) m", out)]
    assert len(transfers) == 2 and max(transfers) < 1e-12, out

    # iota measured by ESSOS from a field-line trace against the iota vmex
    # computed from force balance: an independent check of the same handoff.
    traced = re.findall(
        r"iota traced ([0-9.eE+-]+) .* vs wout ([0-9.eE+-]+)", out)
    assert len(traced) == 2, out
    for got, expected in traced:
        assert abs(float(got) / float(expected) - 1.0) < 1e-3, out

    # Coming back the other way, the tabulated coil field must reproduce
    # direct Biot-Savart on the surface NESTOR evaluates it on.
    tabulation = re.search(r"on the LCFS: ([0-9.eE+-]+) median", out)
    assert tabulation is not None and float(tabulation.group(1)) < 1e-3, out


def test_finite_beta_scan(tmp_path):
    out = _run_example(EXAMPLES / "finite_beta_scan.py", tmp_path, timeout=900)
    # rows: pres_scale  beta_tot  R_axis  Shafranov  minDMerc
    rows = re.findall(r"^\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+[0-9.]+\s+([+-][0-9.]+)\s",
                      out, re.M)
    betas = [float(b) for _, b, _ in rows]
    shafr = [float(s) for _, _, s in rows]
    assert len(betas) == 3, f"expected 3 pressure points, got {rows}"
    assert betas[-1] > betas[0] and betas[-1] > 5e-3, "beta should rise into finite-beta"
    assert shafr[-1] > shafr[0], "magnetic axis should shift outward (Shafranov)"


@pytest.mark.parametrize("case", [
    "QA",  # PR smoke: proves the QS optimization pipeline end-to-end
    pytest.param("QH", marks=pytest.mark.full),  # nightly (subprocess cold-start heavy)
    pytest.param("QP", marks=pytest.mark.full),
])
def test_qs_optimization_examples(case, tmp_path):
    script = EXAMPLES / "optimization" / f"{case}_optimization.py"
    out = _run_example(script, tmp_path)
    _assert_cost_decreased(out, case)
    assert (tmp_path / f"input.{case}_optimized").exists()
    assert (tmp_path / f"wout_{case}_optimized.nc").exists()
    assert (tmp_path / f"{case}_optimized_summary.png").exists()
    match = re.search(r"\[final\] QS total = ([0-9.eE+-]+)", out)
    assert match is not None and np.isfinite(float(match.group(1)))


def test_qa_scalar_optimization_example(tmp_path):
    """The vacuum scalar driver descends and writes distinct outputs."""
    script = EXAMPLES / "optimization" / "QA_optimization_scalar.py"
    out = _run_example(script, tmp_path)
    match = re.search(r"scalar cost: ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(2)) < float(match.group(1))
    assert "[final] QS total" in out
    assert (tmp_path / "input.QA_scalar_optimized").exists()
    assert (tmp_path / "wout_QA_scalar_optimized.nc").exists()
    assert (tmp_path / "QA_scalar_optimized_summary.png").exists()


@pytest.mark.full
def test_qa_finite_beta_scalar_optimization_example(tmp_path):
    """The finite-beta scalar driver includes pressure and stability rows."""
    script = EXAMPLES / "optimization" / "QA_optimization_finite_beta_scalar.py"
    out = _run_example(script, tmp_path)
    match = re.search(r"scalar cost: ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(2)) < float(match.group(1))
    beta = re.search(r"\[final\].*beta = ([0-9.]+)%", out)
    assert beta is not None and 0.5 < float(beta.group(1)) < 2.0
    assert (tmp_path / "input.QA_finite_beta_scalar_optimized").exists()
    assert (tmp_path / "wout_QA_finite_beta_scalar_optimized.nc").exists()
    assert (tmp_path / "QA_finite_beta_scalar_optimized_summary.png").exists()


@pytest.mark.full
def test_qa_finite_beta_optimization_example(tmp_path):
    """The finite-beta least-squares driver starts at the target beta."""
    script = EXAMPLES / "optimization" / "QA_optimization_finite_beta.py"
    out = _run_example(script, tmp_path)
    _assert_cost_decreased(out, "QA finite beta")
    corrected = re.search(r"corrected [0-9.]+ Wb gives beta ([0-9.]+)%", out)
    assert corrected is not None and abs(float(corrected.group(1)) - 1.0) < 0.01
    beta = re.search(r"\[final\].*beta = ([0-9.]+)%", out)
    assert beta is not None and 0.5 < float(beta.group(1)) < 2.0
    assert (tmp_path / "input.QA_finite_beta_optimized").exists()
    assert (tmp_path / "wout_QA_finite_beta_optimized.nc").exists()
    assert (tmp_path / "QA_finite_beta_optimized_summary.png").exists()


@pytest.mark.full  # nightly: shared Boozer + bounce-action Jacobian is cold-compile heavy
def test_qi_maxj_continuation_example(tmp_path):
    """Reduced-budget QI+maximum-J continuation smoke test."""
    script = EXAMPLES / "optimization" / "QI_maxJ_continuation.py"
    # 814-832 s on a hosted runner sharing its cores with a second worker
    # (weekly examples lane, 2026-09-19), so 900 s timed out on 2026-09-23.
    out = _run_example(script, tmp_path, timeout=1800)
    _assert_cost_decreased(out, "QI-maxJ")
    seed = re.search(r"\[seed\] QI = ([0-9.eE+-]+)", out)
    final = re.search(r"\[final\] QI = ([0-9.eE+-]+)", out)
    assert seed is not None and final is not None
    assert np.isfinite(float(final.group(1))) and float(final.group(1)) <= 1.05 * float(seed.group(1))
    assert re.search(r"J-invariance = ([0-9.eE+-]+), maximum-J = ([0-9.eE+-]+)", out)
    displacement = re.search(r"normalized boundary displacement = ([0-9.eE+-]+)", out)
    maxj_fraction = re.search(r"maximum-J fraction = ([0-9.]+)%", out)
    assert displacement is not None and float(displacement.group(1)) > 1.0e-3
    assert maxj_fraction is not None and np.isfinite(float(maxj_fraction.group(1)))
    assert (tmp_path / "input.QI_maxJ_optimized").exists()
    assert (tmp_path / "wout_QI_maxJ_optimized.nc").exists()
    assert (tmp_path / "QI_maxJ_optimized_summary.png").stat().st_size > 10_000


@pytest.mark.full  # nightly: Gamma_c surrogate + Boozer action adjoint is cold-compile heavy
def test_combined_confinement_objective_example(tmp_path):
    """Reduced-budget epsilon/Gamma_c/maximum-J combined-objective smoke test."""
    script = EXAMPLES / "optimization" / "omnigenity_epsilon_gammac_maxj.py"
    out = _run_example(script, tmp_path, timeout=1800)
    _assert_cost_decreased(out, "eps-gammac-maxJ")
    hard = re.search(r"hard Gamma_c mean = ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    assert hard is not None
    assert all(np.isfinite(float(v)) and float(v) > 0.0 for v in hard.groups())
    assert re.search(
        r"maximum-J residual = ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    # the hard ripple line appears when NEO_JAX is installed; its absence
    # must be stated, never silently skipped
    assert (re.search(r"epsilon_eff\^\(3/2\) mean = ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
            or "epsilon_eff unavailable" in out)
    assert (tmp_path / "input.QA_eps_gammac_maxJ").exists()
    assert (tmp_path / "wout_QA_eps_gammac_maxJ.nc").exists()
    assert (tmp_path / "QA_eps_gammac_maxJ_summary.png").stat().st_size > 10_000


@pytest.mark.full  # nightly: QI mode ladder + Boozer, subprocess cold-start heavy
def test_qi_optimization_example(tmp_path):
    pytest.importorskip("booz_xform_jax")
    script = EXAMPLES / "optimization" / "QI_optimization.py"
    out = _run_example(script, tmp_path)
    _assert_cost_decreased(out, "QI")
    stage = re.search(r"\[QI mode \d+\] constructed QI = ([0-9.eE+-]+)", out)
    final = re.search(r"\[final\] constructed QI = ([0-9.eE+-]+)", out)
    assert stage is not None and final is not None
    # the finer re-solve the example ends on must not undo the optimization
    assert np.isfinite(float(final.group(1)))
    assert float(final.group(1)) <= 1.05 * float(stage.group(1))
    # the headline line carries the optimized total and its independent
    # recomputation (equal grids under the CI budget, finer otherwise)
    totals = re.search(r"QI total ([0-9.eE+-]+); independent fine-grid "
                       r"validation ([0-9.eE+-]+)", out)
    assert totals is not None
    assert all(np.isfinite(float(value)) for value in totals.groups())
    assert (tmp_path / "wout_QI_optimized.nc").exists()


@pytest.mark.full  # nightly: every residual evaluation is a finite-beta solve
def test_qa_ballooning_optimization_example(tmp_path):
    """Reduced-budget QA + infinite-n ballooning optimization smoke test."""
    script = EXAMPLES / "optimization" / "QA_optimization_ballooning.py"
    out = _run_example(script, tmp_path, timeout=1800)
    _assert_cost_decreased(out, "QA-ballooning")
    seed = re.search(r"max lambda = ([0-9.eE+-]+) \(unstable\)", out)
    final = re.search(r"max lambda ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    assert seed is not None and final is not None
    # The seed must be the case the objective is for: ballooning-unstable while
    # Mercier says nothing is wrong.  Otherwise the example proves nothing.
    assert float(seed.group(1)) > 0.0
    assert re.search(r"min DMerc = \+[0-9.eE+-]+ \(Mercier-stable\)", out)
    assert float(final.group(2)) < float(final.group(1))
    assert (tmp_path / "input.QA_ballooning_optimized").exists()
    assert (tmp_path / "wout_QA_ballooning_optimized.nc").exists()
    assert (tmp_path / "QA_ballooning_optimized_stability.png").stat().st_size > 10_000


def test_ballooning_example_scans_the_ballooning_parameter():
    """The example must not optimize a bound taken at a single zeta0.

    ``lambda`` peaks at a configuration-dependent ``zeta0``; on this seed the
    single-point default reports 3.27e-3 where the scan reports 4.42e-3, so a
    ``zeta0 = 0`` objective would drive the wrong quantity to zero.
    """
    source = (EXAMPLES / "optimization" / "QA_optimization_ballooning.py").read_text()
    assert "ZETA0S = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5)" in source
    assert "zeta0s=ZETA0S" in source


def test_vacuum_qs_examples_expose_trial_pressure_terms():
    """Vacuum QS examples expose the tested trial-pressure stability terms."""
    for name in ("QA_optimization_DMerc_vacuum.py", "QH_optimization.py"):
        source = (EXAMPLES / "optimization" / name).read_text()
        assert "USE_TRIAL_STABILITY" in source
        assert "(trial_dmerc, 0.0, stability_weights)" in source
        assert "(trial_dr, 0.0, stability_weights)" in source
        assert "weights rise smoothly toward the edge" in source


# The self-consistent-bootstrap examples reproduce arXiv:2205.02914 against the
# Zenodo dataset, which is a large local-only archive (not in CI) — skip when
# absent.  Nightly-gated: each runs a multi-iteration Picard loop of solves.
_ZENODO_2205 = Path(os.environ.get(
    "VMEX_ZENODO_2205_02914",
    str(Path.home() / "local" /
        "20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current")))


@pytest.mark.full
@pytest.mark.skipif(not _ZENODO_2205.is_dir(),
                    reason="arXiv:2205.02914 Zenodo dataset not present")
@pytest.mark.parametrize("case", ["QA", "QH"])
def test_bootstrap_selfconsistent_examples(case, tmp_path):
    script = REPO / "benchmarks" / f"{case}_bootstrap_selfconsistent.py"
    # The guard above accepts the dataset at its default location, but the
    # script only reads the environment variable, so pass the resolved path.
    out = _run_example(script, tmp_path, timeout=1200,
                       VMEX_ZENODO_2205_02914=str(_ZENODO_2205))
    m = re.search(r"final f_boot = ([0-9.eE+-]+)", out)
    assert m is not None and float(m.group(1)) < 5e-2, f"{case} f_boot: {out[-400:]}"
    assert (tmp_path / f"output_{case}_bootstrap_selfconsistent"
            / f"wout_{case}_bootstrap_selfconsistent.nc").exists()


@pytest.mark.full  # nightly: Picard seed + one exact finite-beta optimization stage
@pytest.mark.parametrize(("case", "figure_of_merit"),
                         [("QA", "QS"), ("QH", "QS"), ("QI", "constructed QI")])
def test_bootstrap_optimization_examples(case, figure_of_merit, tmp_path):
    if case == "QI":
        pytest.importorskip("booz_xform_jax")
    script = EXAMPLES / "optimization" / f"{case}_optimization_bootstrap.py"
    out = _run_example(script, tmp_path, timeout=1800)
    _assert_cost_decreased(out, f"{case}-bootstrap")
    assert "self-consistent seed" in out and f"[final] {figure_of_merit}" in out
    match = re.search(r"\[final\].*beta = ([0-9.]+)%", out)
    assert match is not None and 1.0 < float(match.group(1)) < 4.0
    assert (tmp_path / f"input.{case}_bootstrap_optimized").exists()
    assert (tmp_path / f"wout_{case}_bootstrap_optimized.nc").exists()
    assert (tmp_path / f"{case}_bootstrap_current.png").exists()


@pytest.mark.full  # nightly: optional optimizer interoperability, cold JAX compilation
# each wout name carries the script's own METHOD constant
@pytest.mark.parametrize(("script_name", "dependency", "output"), [
    ("QA_optimization_scipy.py", None, "wout_QA_scipy_L-BFGS-B.nc"),
    ("QI_optimization_scipy.py", None, "wout_QI_scipy_L-BFGS-B.nc"),
    ("QI_optimization_jaxopt.py", "jaxopt", "wout_QI_jaxopt_LM.nc"),
    ("QI_optimization_optax.py", "optax", "wout_QI_optax_adam.nc"),
])
def test_scalar_optimizer_examples(script_name, dependency, output, tmp_path):
    if dependency is not None:
        pytest.importorskip(dependency)
    out = _run_example(EXAMPLES / "optimization" / script_name, tmp_path, timeout=1800)
    assert "final cost" in out
    assert (tmp_path / output).exists()


@pytest.mark.full  # nightly: the same physics with the limits as constraints
def test_fixed_boundary_single_stage_augmented_lagrangian(tmp_path):
    """The companion carries the three limits on multipliers, over a stage loop.

    Which of the three reaches the targets at full budget is recorded in the
    examples README, not asserted here, because smoke mode reaches none of them.
    """
    pytest.importorskip("essos")
    out = _run_example(
        EXAMPLES / "optimization" / "single_stage_optimization_augmented_lagrangian.py",
        tmp_path, timeout=1800)
    match = re.search(r"Objective: ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(2)) < float(match.group(1))
    # The stage loop is the machinery this file exists to show.
    assert re.search(r"\[stage 1\] \d+ L-BFGS-B iterations, \d+ trials, violation = ", out)
    for diagnostic in ("B.n/B: area-weighted RMS", "Minimum coil-surface distance",
                       "Minimum coil-coil distance", "Maximum curvature", "Coil lengths",
                       "Minimum |iota| = ", "Aspect ratio = "):
        assert diagnostic in out
    summary = json.loads(
        (tmp_path / "single_stage_augmented_lagrangian_summary.json").read_text())
    assert summary["smoke"] and summary["trials"] >= 1 and "stages" in summary
    assert summary["met"] == (not summary["unmet"])
    for name in ("wout_single_stage_augmented_lagrangian_optimized.nc",
                 "single_stage_augmented_lagrangian_objectives.png",
                 "coils_single_stage_augmented_lagrangian_optimized.vtu"):
        assert (tmp_path / name).exists()


@pytest.mark.full  # nightly: the joint Gauss-Newton form of the same problem
def test_fixed_boundary_single_stage_least_squares(tmp_path):
    """Residual vector and Jacobian rather than a scalar and a gradient."""
    pytest.importorskip("essos")
    out = _run_example(
        EXAMPLES / "optimization" / "single_stage_optimization_least_squares.py",
        tmp_path, timeout=2400)
    match = re.search(r"Objective: ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(2)) < float(match.group(1))
    assert re.search(r"\[solve\] \d+ residual evaluations, \d+ trials, status ", out)
    summary = json.loads(
        (tmp_path / "single_stage_least_squares_summary.json").read_text())
    assert summary["smoke"] and summary["residual_evaluations"] >= 1
    assert summary["met"] == (not summary["unmet"])
    for name in ("wout_single_stage_least_squares_optimized.nc",
                 "single_stage_least_squares_objectives.png",
                 "coils_single_stage_least_squares_optimized.vtu"):
        assert (tmp_path / name).exists()


@pytest.mark.full  # nightly: exact VMEX+ESSOS reverse-mode graph and ParaView output
def test_fixed_boundary_single_stage_optimization(tmp_path):
    pytest.importorskip("essos")
    out = _run_example(
        EXAMPLES / "optimization" / "single_stage_optimization.py", tmp_path, timeout=1800)
    # Smoke mode caps the budget, so only the fall of the objective is asserted.
    match = re.search(r"Objective: ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    assert match is not None and float(match.group(2)) < float(match.group(1))
    # One bounded solve and no stage loop: the simplicity this file is named for.
    assert re.search(r"\[solve\] \d+ L-BFGS-B iterations, \d+ trials, status ", out)
    assert "[stage 1]" not in out
    for diagnostic in ("B.n/B: area-weighted RMS", "Minimum coil-surface distance",
                       "Minimum coil-coil distance", "Maximum curvature", "Coil lengths",
                       "Minimum |iota| = ", "Aspect ratio = "):
        assert diagnostic in out
    normal = re.search(r"B\.n/B: area-weighted RMS = ([0-9.]+)%, max = ([0-9.]+)%", out)
    assert normal is not None and all(np.isfinite(float(value)) for value in normal.groups())
    summary = json.loads((tmp_path / "single_stage_optimization_summary.json").read_text())
    assert summary["smoke"] and summary["trials"] >= 1
    assert summary["met"] == (not summary["unmet"])
    verdict = "All stated targets met." if summary["met"] else "did NOT meet its stated targets"
    assert verdict in out
    for name in ("wout_single_stage_optimized.nc", "single_stage_objectives.png",
                 "surface_single_stage_initial.vts", "coils_single_stage_initial.vtu",
                 "surface_single_stage_optimized.vts", "coils_single_stage_optimized.vtu"):
        assert (tmp_path / name).exists()
    surface_vtk = (tmp_path / "surface_single_stage_optimized.vts").read_bytes()
    assert b'Name="B_BiotSavart"' in surface_vtk
    assert b'Name="B_dot_n_over_B"' in surface_vtk


@pytest.mark.full  # nightly: one finite-beta VMEX + VCJ + ESSOS graph
def test_finite_beta_single_stage_optimization(tmp_path, monkeypatch):
    pytest.importorskip("essos")
    pytest.importorskip("virtual_casing_jax")
    # Keep this independent compile from filling a shared developer cache.
    monkeypatch.setenv("VMEX_COMPILATION_CACHE", "disabled")
    out = _run_example(
        EXAMPLES / "optimization" / "single_stage_optimization_finite_beta.py",
        tmp_path, timeout=1800)
    for diagnostic in ("[final] QA total", "(B_coils + B_plasma).n/B: area-weighted RMS",
                       "Volume-average beta = ", "Coil lengths", "Maximum curvature",
                       "Minimum |iota| = ", "Aspect ratio = "):
        assert diagnostic in out
    summary = json.loads(
        (tmp_path / "single_stage_optimization_finite_beta_summary.json").read_text())
    assert summary["met"] == (not summary["unmet"])
    assert summary["met"] or "did NOT meet its stated targets" in out
    for name in ("wout_single_stage_finite_beta_optimized.nc",
                 "single_stage_finite_beta_objectives.png",
                 "surface_single_stage_finite_beta_initial.vts",
                 "coils_single_stage_finite_beta_initial.vtu",
                 "surface_single_stage_finite_beta_optimized.vts",
                 "coils_single_stage_finite_beta_optimized.vtu"):
        assert (tmp_path / name).exists()


def test_field_query_examples_cover_inside_outside_and_vjps() -> None:
    """Keep the two runnable API examples explicit without another slow solve."""
    interior = (EXAMPLES / "vmex_get_B_gradB.py").read_text()
    exterior = (EXAMPLES / "vmex_get_B_outside_plasma.py").read_text()
    for source in (interior, exterior):
        for call in ("set_points_xyz", "set_points_flux", ".B()", ".absB()", ".gradB()", ".B_vjp(",
                     ".gradB_vjp(", ".gradgradB_vjp(", ".gradgradgradB_vjp("):
            assert call in source
        assert "VmecProblem.from_input" in source and "SimpleNamespace" not in source
    assert "get_points_flux" in interior
    assert "uses_virtual_casing" in exterior and "exterior_field" in exterior
    assert "ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json" in exterior


# Wall time here is XLA compilation of a few large graphs: the order-3 forward
# derivative plus one parameter adjoint per order. It is set by that structure,
# not by resolution or by the number of evaluated points, so a coarser smoke
# equilibrium does not move it. Taking those derivatives in flux coordinates at
# the root, rather than through the inverse map, cut it roughly in half: cold,
# on two cores, 2.8 min (interior) and 3.8 min (exterior), from 5.5 and 5.9.
# This lane runs with ``-n 2``, so a four-core runner gives each test that share.
@pytest.mark.full  # nightly: two bounded high-order field/VJP compilations
@pytest.mark.parametrize(("script", "message"), [
    ("vmex_get_B_gradB.py", "gradgradgradB VJP shapes"),
    ("vmex_get_B_outside_plasma.py", "uses virtual casing = True"),
])
def test_field_query_examples_run(script, message, tmp_path):
    if "outside" in script:
        pytest.importorskip("essos")
        pytest.importorskip("virtual_casing_jax")
    out = _run_example(EXAMPLES / script, tmp_path, timeout=600)
    assert message in out and "dof_names =" in out
    # A clean exit is not a field: a stalled inversion used to print NaN here.
    for label in (r"B \[T\]", r"\|B\| \[T\]", "largest VJP entries"):
        line = re.search(rf"^{label} = (.*)$", out, flags=re.MULTILINE)
        assert line is not None, label
        values = [float(v) for v in re.findall(r"[-+]?(?:\d[\d.]*(?:e[-+]?\d+)?|nan|inf)", line.group(1))]
        assert values and np.all(np.isfinite(values)) and np.any(np.abs(values) > 0.0), line.group(0)


def test_fieldline_example_uses_vmex_virtual_casing_and_actual_essos_coils() -> None:
    """Keep the integration path explicit without adding a slow solve to PR CI."""
    vacuum = (EXAMPLES / "vmex_fieldline_tracing_vacuum.py").read_text()
    finite = (EXAMPLES / "vmex_fieldline_tracing_finite_beta.py").read_text()
    for source in (vacuum, finite):
        for contract in ("BiotSavart", "field_in_flux_coordinates", "trace_field_lines",
                         "poincare_plot", "True boundary B.n/B"):
            assert contract in source
    assert 'plasma="vacuum"' in vacuum and "exterior_field" in vacuum
    assert "VmecExtender" in finite and "with_graded_quadrature" in finite
    assert "ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json" in finite


def test_single_stage_examples_use_general_surface_output_and_movie_colors() -> None:
    vacuum = (EXAMPLES / "optimization" / "single_stage_optimization.py").read_text()
    finite = (EXAMPLES / "optimization" / "single_stage_optimization_finite_beta.py").read_text()
    assert "from pyevtk" not in finite
    assert "surface_initial.to_vtk" in finite
    for source in (vacuum, finite):
        assert "MOVIE_SURFACE_COLOR" in source and "surface_color=" in source


def test_single_stage_examples_enforce_targets_and_fail_loudly() -> None:
    """Targets are constraints checked at the end; a missed one is a non-zero exit."""
    fixed = (EXAMPLES / "optimization" / "single_stage_optimization.py").read_text()
    auglag = (EXAMPLES / "optimization"
              / "single_stage_optimization_augmented_lagrangian.py").read_text()
    squares = (EXAMPLES / "optimization"
               / "single_stage_optimization_least_squares.py").read_text()
    free = (EXAMPLES / "optimization" / "single_stage_free_boundary_optimization.py").read_text()
    finite = (EXAMPLES / "optimization" / "single_stage_optimization_finite_beta.py").read_text()
    free_finite = (EXAMPLES / "optimization"
                   / "single_stage_free_boundary_optimization_finite_beta.py").read_text()
    assert "def hinge(" in fixed and 'method="L-BFGS-B"' in fixed
    assert "def augmented_lagrangian(" in auglag and 'method="L-BFGS-B"' in auglag
    for constraint in ("IOTA_CONSTRAINT", "ASPECT_CONSTRAINT", "NORMAL_FIELD_CONSTRAINT"):
        assert constraint in fixed and constraint in auglag and constraint in squares
    # One equilibrium solve and one scalar adjoint per trial in the two scalar
    # lanes: no residual Jacobian and no second residual callback.
    for source in (fixed, auglag):
        assert "jax_objective_from_state" in source
        assert "jax_value_and_grad" not in source and "jax_residual" not in source
    # The least-squares lane is the one that does ask for a Jacobian.
    assert "residual_and_jac" in squares and "least_squares(" in squares
    # The free-boundary pullback is host-eager; everything after the solve is jitted.
    assert "@jax.jit\ndef accepted_terms(" in free
    # Finite beta at 0.5% with a simple pressure and no current; no bootstrap.
    for source in (finite, free_finite):
        assert "TARGET_BETA = 0.005" in source and "PRESSURE_PROFILE = [1.0, -1.0]" in source
        assert "ncurr=1, curtor=0.0, ac=np.zeros(21)" in source
        for bootstrap in ("vmex.core.bootstrap", "RedlBootstrapMismatch", "KineticProfiles"):
            assert bootstrap not in source
    for source in (fixed, auglag, squares, free, finite, free_finite):
        assert "did NOT meet its stated targets" in source
        assert "if unmet and not ci_smoke:\n    raise SystemExit(1)" in source
        assert "_summary.json" in source


# Cold, on two cores, the finite-beta script takes 1.4 min; most of it is the
# virtual-casing and tracing compilations, and this lane runs with ``-n 2``.
# Nothing here changed, so this budget stays where it is: 300 s is the value
# that timed out on a runner.
@pytest.mark.full  # nightly: two bounded ESSOS tracing integrations
@pytest.mark.parametrize(("script", "message", "output"), [
    ("vmex_fieldline_tracing_vacuum.py", "VMEX exterior API outside", "vmex_fieldline_tracing_vacuum.png"),
    ("vmex_fieldline_tracing_finite_beta.py", "VMEX coil + virtual-casing field outside",
     "vmex_fieldline_tracing_finite_beta.png"),
])
def test_vmex_fieldline_tracing_examples(script, message, output, tmp_path):
    pytest.importorskip("essos")
    pytest.importorskip("virtual_casing_jax")
    out = _run_example(EXAMPLES / script, tmp_path, timeout=600)
    assert message in out
    if "finite_beta" in script:
        alignment = re.search(r"Boundary field alignment = ([0-9.eE+-]+)", out)
        assert alignment is not None and float(alignment.group(1)) > 0.9
        bounded = re.search(r"Exterior trace QA: (\d+)/(\d+) lines remained", out)
        assert bounded is not None and int(bounded.group(1)) > 0
    assert (tmp_path / output).stat().st_size > 10_000

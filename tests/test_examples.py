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
    EXAMPLES / "optimization" / "QA_optimization_alpha_losses.py",
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
    assert coils == ["essos>=0.22.0"], coils

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
    "examples/prescribed_current_polishing.py",
    "examples/free_boundary_beta_scan.py",
    "examples/free_boundary_essos_coils.py",
    "examples/free_boundary_mgrid.py",
    "examples/free_boundary_phiedge.py",
    "examples/free_boundary_position_control.py",
    "examples/free_boundary_relax_constraint.py",
    "examples/hot_restart_scan.py",
    "examples/mirror/mirror_fixed_boundary_axisymmetric.py",
    "examples/mirror/mirror_fixed_boundary_nonaxisymmetric.py",
    "examples/mirror/mirror_free_boundary_beta_scan.py",
    "examples/mirror/qi_mirror_hybrid_fourier_vs_bspline.py",
    "examples/optimization/QA_optimization.py",
    "examples/optimization/QA_optimization_alpha_losses.py",
    "examples/optimization/QA_optimization_ballooning.py",
    "examples/optimization/QA_optimization_bootstrap.py",
    "examples/optimization/QA_optimization_low_bootstrap.py",
    "examples/optimization/QA_optimization_finite_beta.py",
    "examples/optimization/QA_optimization_finite_beta_scalar.py",
    "examples/optimization/QA_optimization_scalar.py",
    "examples/optimization/QA_optimization_scipy.py",
    "examples/optimization/QA_optimization_turbulence_linear.py",
    "examples/optimization/QA_optimization_turbulence_nonlinear.py",
    "examples/optimization/QA_optimization_turbulence_quasilinear.py",
    "examples/optimization/QH_optimization.py",
    "examples/optimization/QH_optimization_bootstrap.py",
    "examples/optimization/QI_maxJ_continuation.py",
    "examples/optimization/QI_optimization.py",
    "examples/optimization/QI_optimization_bootstrap.py",
    "examples/optimization/QI_optimization_bootstrap_dkx.py",
    "examples/optimization/QI_optimization_jaxopt.py",
    "examples/optimization/QI_optimization_optax.py",
    "examples/optimization/QI_optimization_scipy.py",
    "examples/optimization/QP_optimization.py",
    "examples/optimization/pwO_optimization.py",
    "examples/optimization/pwO_optimization_finite_beta.py",
    "examples/optimization/single_stage_pwO_optimization.py",
    "examples/optimization/omnigenity_epsilon_gammac_maxj.py",
    "examples/optimization/optimize_bootstrap_current.py",
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
    "examples/vmex_essos_tracing_to_wall.py",
    "examples/vmex_mrx_comparison.py",
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
    source = (EXAMPLES / "force_balance_polishing.py").read_text()
    assert "polish=True" in source
    assert "result.polished_state is None or result.polish_report is None" in source
    assert "if not report.converged:" in source
    for field in ("initial_normalized_l2", "final_normalized_l2",
                  "nonlinear_iterations", "termination_reason"):
        assert field in source


@pytest.mark.full  # nightly: discrete solve + prescribed-current polish
def test_prescribed_current_polishing_example(tmp_path):
    out = _run_example(EXAMPLES / "prescribed_current_polishing.py", tmp_path, timeout=600)
    assert "Forward result only" in out
    for label in ("axis", "bulk", "edge"):
        match = re.search(label + r" force/pressure-gradient RMS: ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
        assert match and np.isfinite([float(v) for v in match.groups()]).all(), out


@pytest.mark.full  # nightly: ordinary solve + native strong-force polish (~3 min)
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
                 "wout_shaped_tokamak_pressure.nc"):
        assert (outdir / name).exists()
    # the fair comparison: both files on one mesh, certified the same way; the
    # near-axis error is where the polish gain lives
    assert re.search(r"both WOUT files on ns = \d+, read back and certified the same way", out), out
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
    assert 'method="L-BFGS-B"' in text and '"method": short_lbfgsb' in text
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
    # polishing is axisymmetric-only; these non-axisymmetric scripts must not offer it
    assert "POLISH_FORCE_BALANCE" not in text
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
    if case == "QI":  # NEO on the asymmetric Boozer spectrum, or the stated reason
        assert "optimized epsilon_eff^(3/2)" in out or "epsilon_eff skipped" in out


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


def test_mirror_fixed_boundary_axisymmetric_example(tmp_path):
    """The concise driver example reproduces the exact vacuum mirror."""
    import json
    _run_example(EXAMPLES / "mirror" / "mirror_fixed_boundary_axisymmetric.py", tmp_path, timeout=600)
    outdir = tmp_path / "results" / "mirror_fixed_boundary_axisymmetric"
    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["converged"] and summary["variational_max"] <= 1.0e-12
    assert summary["mirror_ratio_relative_error"] < 2.0e-3  # measured 1.0e-3
    assert summary["axis_field_max_relative_error"] < 2.0e-3  # measured 8.0e-4


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


@pytest.mark.full  # nightly: QA least squares, then ESSOS alpha tracing (smoke pass)
def test_qa_alpha_losses_optimization(tmp_path):
    pytest.importorskip("essos")
    out = _run_example(EXAMPLES / "optimization" / "QA_optimization_alpha_losses.py",
                       tmp_path, timeout=1200)
    losses = dict(re.findall(r"^\s*(seed|QA stage|alpha stage): \d+ alphas for [0-9.]+ ms lose "
                             r"([0-9.]+) %", out, re.M))
    assert set(losses) == {"seed", "QA stage", "alpha stage"}, out
    assert all(0.0 <= float(value) <= 100.0 for value in losses.values())
    for name in ("wout_QA_alpha_losses_optimized.nc", "QA_alpha_losses_optimized_losses.png",
                 "QA_alpha_losses_optimized_trace.png"):
        assert (tmp_path / name).exists(), name


@pytest.mark.full  # nightly: coarse free-boundary solve plus 1,000 relaxation iterations (~1.5 min)
def test_free_boundary_relax_constraint(tmp_path):
    pytest.importorskip("essos.fields")
    out = _run_example(EXAMPLES / "free_boundary_relax_constraint.py", tmp_path, timeout=1500)
    rows = dict(re.findall(r"^\s*(converged|relaxed)\s+\S+\s+([0-9.eE+-]+)", out, re.M))
    assert set(rows) == {"converged", "relaxed"}, out
    assert float(rows["relaxed"]) < 0.5 * float(rows["converged"]), out  # 3x at ns = 12


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


def test_free_boundary_position_control(tmp_path):
    # Five vertical-field offsets, each without and with control (~1 min).
    out = _run_example(EXAMPLES / "free_boundary_position_control.py", tmp_path, timeout=600)
    rows = re.findall(r"^\s*(-?[0-9.]+) \|\s+(\w+)\s+\d+\s+\S+\s+\S+ \|"
                      r"\s+(\w+)\s+\d+\s+\S+\s+(-?[0-9.]+)\s+(-?[0-9.]+)$", out, re.M)
    assert len(rows) == 5, out
    for offset, free, held, shift, bz in rows:
        assert (free == "NO") == (float(offset) == -40.0), out   # only the deficit defeats it
        assert held == "yes" and abs(float(shift)) < 1.0, out      # axis within 1 cm
        assert abs(float(bz) + float(offset)) < 2.0, out           # recovers the offset
    assert (tmp_path / "output_position_control" / "mgrid_d3d_vertical_deficit.nc").exists()


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


@pytest.mark.full
@pytest.mark.parametrize("family", ["canonical", "mapped"])
@pytest.mark.parametrize("finite_beta", [False, True])
def test_pwo_stage1_example(tmp_path, family, finite_beta):
    suffix = "_finite_beta" if finite_beta else ""
    _run_example(EXAMPLES / "optimization" / f"pwO_optimization{suffix}.py", tmp_path,
                 VMEX_PWO_FAMILY=family)
    stem = f"pwO{suffix}_optimized"
    summary = json.loads((tmp_path / f"{stem}_summary.json").read_text())
    assert summary["family"] == family
    assert all(summary["checks"].values()), summary["unmet"]
    assert summary["actual_s"] == [0.5]
    assert summary["metrics"]["fine"] < summary["metrics"]["initial_fine"]
    assert all(np.all(np.isfinite(value)) for value in summary["metrics"].values())
    assert not summary["research_validation_complete"]
    assert (tmp_path / f"wout_{stem}.nc").exists()


@pytest.mark.parametrize("theta_shift,zeta_shift", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("nfp,iota,t1,t2", [(2, -0.42, 0.3, -3.0),
                                            (5, -0.9, 0.441421356, -3.141421356)])
def test_pwo_canonical_seed_phase_matches_periodic_field(nfp, iota, t1, t2, theta_shift, zeta_shift):
    import runpy
    from vmex.core.pwo import canonical_pwo_field

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    theta = 2 * np.pi * np.arange(24)[None, :, None] / 24
    zeta = 2 * np.pi * np.arange(24)[None, None, :] / (24 * nfp)
    params = dict(iota=iota, nfp=nfp, bmin=1.0, bmax=2.0, t1=t1, t2=t2)
    original = canonical_pwo_field(theta, zeta, zeta_center=0.0, **params)
    selected_theta, selected_zeta = theta_shift * np.pi, zeta_shift * np.pi / nfp
    shifted = canonical_pwo_field(theta, zeta, theta_center=selected_theta,
                                  zeta_center=selected_zeta, **params)["B_target"]
    rolled = np.roll(original["B_target"], (theta_shift * 12, zeta_shift * 12), axis=(-2, -1))
    np.testing.assert_allclose(rolled, shifted, atol=2e-14, rtol=2e-14)
    if nfp == 5:
        assert np.min(original["corner_margin"]) < 0  # valid patch crosses the centered cell
    theta_phase, zeta_phase, costs = example["canonical_seed_phase"](shifted, np.asarray(original["B_target"]), nfp)
    assert (theta_phase, zeta_phase) == pytest.approx((selected_theta, selected_zeta))
    assert costs[2 * theta_shift + zeta_shift] == pytest.approx(0.0, abs=1e-25)
    with pytest.raises(ValueError, match="even angular grids"):
        example["canonical_seed_phase"](shifted[..., :-1], np.asarray(original["B_target"])[..., :-1], nfp)

    target = example["make_target"]([0.5], family="canonical", bmin=1.0, bmax=2.0,
                                     b_ref=1.5, options={}, grid=39, iota_sign=-1.0,
                                     theta_center=theta_phase, zeta_center=zeta_phase)
    score = target.kernel(bmnc_b=np.array([[1.5]]), xm_b=np.array([0]), xn_b=np.array([0]),
                          iota_b=np.array([iota]), nfp=nfp, **target.options)
    expected = canonical_pwo_field(score["theta"][None, :, None], score["zeta"][None, None, :],
                                   **{**params, "t1": 0.3, "t2": -3.0,
                                      "theta_center": theta_phase, "zeta_center": zeta_phase})
    np.testing.assert_allclose(score["B_target"], expected["B_target"], atol=2e-14, rtol=2e-14)


def test_pwo_canonical_calibration_holds_broad_finite_p_mean_and_mirror():
    import runpy
    from vmex.core.pwo import canonical_pwo_field

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    theta = 2 * np.pi * np.arange(96)[:, None] / 96
    zeta = 2 * np.pi * np.arange(96)[None, :] / (2 * 96)
    shape = dict(iota=-0.03774573741535906, nfp=2, t1=0.3,
                 t2=-0.07549147483071812, p=2, theta_center=0.0, zeta_center=0.0)
    unit = np.asarray(canonical_pwo_field(theta, zeta, bmin=1.0, bmax=2.0, **shape)["B_target"]) - 1
    assert unit.min() > 1e-9  # Finite-p tails violate the former zero-minimum assumption.
    mean, mirror = 0.92, 0.29
    bmin, bmax = example["canonical_target_bounds"](unit, mean, mirror)
    assert 0 < bmin < bmax
    calibrated = np.asarray(canonical_pwo_field(theta, zeta, bmin=bmin, bmax=bmax, **shape)["B_target"])
    assert calibrated.mean() == pytest.approx(mean, rel=1e-13)
    assert np.ptp(calibrated) / (calibrated.max() + calibrated.min()) == pytest.approx(mirror, rel=1e-13)


def test_pwo_qp_gate_and_penalty_controls():
    import runpy

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    failures = example["qp_gate_failures"]
    metrics = dict(qs_total=0.01, aspect=6.0, min_abs_iota=0.41, mirror=0.1, elongation=2.0)
    assert failures(metrics) == []
    for name, value, gate in (("qs_total", np.nan, "QP objective finite"),
                              ("aspect", 6.3, "aspect target"),
                              ("min_abs_iota", 0.39, "iota floor"),
                              ("mirror", 0.04, "mirror interval"),
                              ("mirror", 0.350018, "mirror interval"),
                              ("elongation", 8.1, "elongation limit")):
        assert gate in failures({**metrics, name: value})

    hold = example["hold_ratio"]
    assert float(hold(1.02, 1.0, 0.03, "interval")) == 0.0
    assert float(hold(1.02, 1.0, 0.03, "equality")) == pytest.approx(2 / 3)
    assert float(hold(0.96, 1.0, 0.03, "interval")) == pytest.approx(1 / 3)
    example["run"].__globals__["PENALTY"] = "invalid"
    with pytest.raises(ValueError, match="equality\\|interval"):
        example["run"]()


def test_pwo_qp_gate_preserves_checkpoints_before_target_calibration(tmp_path, monkeypatch):
    """A failed independent re-solve cannot silently become the pwO reference."""
    import runpy
    from types import SimpleNamespace

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    run = example["run"]
    run.__globals__.update(SEED="near_circular", QS_INPUT=None, QS_MAX_MODES=[1], QS_MAXITER=[1],
                           FAMILY="canonical", METHOD="scalar", PENALTY="equality", FINAL_NS=32)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(run.__globals__, "restart_state", lambda wout, deck: None)

    def solve(deck, **kwargs):
        return SimpleNamespace(state=deck, runtime=None,
                               wout=SimpleNamespace(ns=deck.ns_array[-1], Aminor_p=0.1, volavgB=1.0))

    def write_wout(path, deck):
        Path(path).write_text(str(int(deck.ns)))

    opt = example["opt"]
    monkeypatch.setattr(opt, "solve_equilibrium", solve)
    monkeypatch.setattr(example["vj"], "write_wout", write_wout)
    monkeypatch.setattr(opt, "QuasisymmetryRatioResidual", lambda *args, **kwargs:
                        SimpleNamespace(total_state=lambda state, runtime: 0.01))
    for name, value in (("aspect_ratio", 6.0), ("min_abs_iota", 0.3),
                        ("mirror_ratio", 0.1), ("max_elongation", 2.0)):
        monkeypatch.setattr(opt, name, lambda state, runtime, value=value: value)

    class Problem:
        x0 = np.zeros(1)
        scales = np.ones(1)

        def __init__(self, deck):
            self.deck = deck

        def subproblem(self, **kwargs):
            return self

        def value_and_grad(self, x):
            return float(x @ x), 2 * x

        def input_from_x(self, x):
            from dataclasses import replace
            rbc = self.deck.rbc.copy()
            rbc[self.deck.ntor - 1, 1] = -0.02
            return replace(self.deck, rbc=rbc)

        def equilibrium_from_x(self, x):
            return solve(self.input_from_x(x))

        def x_from_input(self, deck):
            return self.x0.copy()

    monkeypatch.setattr(opt.VmecProblem, "from_loss", lambda deck, *args, **kwargs: Problem(deck))
    with pytest.raises(RuntimeError, match="QP preconditioning failed iota floor"):
        run()
    raw = example["vj"].VmecInput.from_file(tmp_path / "input.pwO_optimized_near_circular")
    trained = example["vj"].VmecInput.from_file(tmp_path / "input.pwO_optimized_qs")
    assert raw.rbc[raw.ntor - 1, 1] == -0.005
    assert trained.rbc[trained.ntor - 1, 1] == -0.02
    assert (tmp_path / "wout_pwO_optimized_near_circular.nc").read_text() == "16"
    assert (tmp_path / "wout_pwO_optimized_qs_refined.nc").read_text() == "32"
    summary = json.loads((tmp_path / "pwO_optimized_qs_summary.json").read_text())
    assert summary["metrics"]["min_abs_iota"] == 0.3
    assert summary["status"] == "failed" and summary["unmet"] == ["iota floor"]
    assert not (tmp_path / "input.pwO_optimized").exists()


@pytest.mark.parametrize("family", ["canonical", "mapped"])
@pytest.mark.parametrize("penalty, expected", [("equality", 31.25), ("interval", 11.25)])
@pytest.mark.parametrize("edge_mirror, margin_cost", [(0.1, 0.0), (0.35, 5 / 9)])
def test_pwo_mirror_controls_penalize_collapse_and_limit_approach(
        family, penalty, expected, edge_mirror, margin_cost, monkeypatch):
    """Zero fit residual cannot hide local collapse or a weak edge-mirror margin."""
    import runpy
    from types import SimpleNamespace

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    run = example["run"]
    run.__globals__.update(SEED="qi", QS_INPUT=None, FAMILY=family, METHOD="scalar", PENALTY=penalty)
    opt = example["opt"]
    wout = SimpleNamespace(aspect=6.0, Aminor_p=0.1, volavgB=1.0)
    monkeypatch.setattr(opt, "solve_equilibrium", lambda deck, **kwargs:
                        SimpleNamespace(state=deck, runtime=None, wout=wout))
    for name, value in (("aspect_ratio", 6.0), ("min_abs_iota", 0.42), ("mean_iota", 0.42),
                        ("mirror_ratio", edge_mirror), ("max_elongation", 2.0)):
        monkeypatch.setattr(opt, name, lambda state, runtime, value=value: value)
    shape = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    trial = False

    class Target:
        def compute_state(self, state, runtime):
            mirror = 0.075 if trial else 0.1
            B = 1 + mirror * (2 * shape - 1)
            return dict(B=B, B_mapped=B, B_target=1 + shape,
                        mirror=np.array([mirror]), residuals1d=np.zeros(1))

    monkeypatch.setitem(run.__globals__, "CanonicalPwOResidual", lambda *args, **kwargs: Target())
    monkeypatch.setitem(run.__globals__, "make_target", lambda *args, **kwargs: Target())

    class EvaluatedRows(Exception):
        pass

    def inspect_loss(deck, loss, **kwargs):
        nonlocal trial
        trial = True
        assert float(loss(deck, None)) == pytest.approx(expected + margin_cost)
        raise EvaluatedRows

    monkeypatch.setattr(opt.VmecProblem, "from_loss", inspect_loss)
    with pytest.raises(EvaluatedRows):
        run()


@pytest.mark.full
def test_pwo_single_stage_example(tmp_path):
    pytest.importorskip("essos")
    _run_example(EXAMPLES / "optimization" / "single_stage_pwO_optimization.py", tmp_path)
    summary = json.loads((tmp_path / "single_stage_pwO_summary.json").read_text())
    assert summary["final"] < summary["initial"]
    assert summary["geometric_proxies_only"] and not summary["research_validation_complete"]
    assert all(np.all(np.isfinite(value)) for value in summary["metrics"].values())
    assert (tmp_path / "coils_single_stage_pwO.json").exists()


@pytest.mark.full  # nightly: every residual evaluation is a finite-beta solve
def test_qa_ballooning_optimization_example(tmp_path):
    """Reduced-budget QA + infinite-n ballooning optimization smoke test."""
    script = EXAMPLES / "optimization" / "QA_optimization_ballooning.py"
    out = _run_example(script, tmp_path, timeout=1800)
    _assert_cost_decreased(out, "QA-ballooning")
    final = re.search(r"max lambda ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    mercier = re.search(r"min PHIEDGE\^2 DMerc ([0-9.eE+-]+) -> ([0-9.eE+-]+)", out)
    beta = re.search(r"beta ([0-9.]+)% -> ([0-9.]+)%", out)
    assert final is not None and mercier is not None and beta is not None
    # The seed must be the case the objective is for: ballooning-unstable while
    # Mercier says nothing is wrong.  Otherwise the example proves nothing.
    assert float(final.group(1)) > 0.0
    assert float(mercier.group(1)) > 0.0
    # Ballooning improves without buying it with Mercier or beta.
    assert float(final.group(2)) < float(final.group(1))
    assert float(mercier.group(2)) > 0.0
    assert abs(float(beta.group(2)) - float(beta.group(1))) < 0.1
    assert (tmp_path / "input.QA_ballooning_optimized").exists()
    assert (tmp_path / "wout_QA_ballooning_optimized.nc").exists()
    assert (tmp_path / "QA_ballooning_optimized_stability.png").stat().st_size > 10_000


@pytest.mark.full  # nightly: every residual evaluation runs a GKX solve
@pytest.mark.parametrize("kind", ["linear", "quasilinear", "nonlinear"])
def test_qa_turbulence_optimization_examples(tmp_path, kind):
    """QA + one GKX turbulence tuple: linear, quasilinear and nonlinear."""
    pytest.importorskip("gkx")
    script = EXAMPLES / "optimization" / f"QA_optimization_turbulence_{kind}.py"
    out = _run_example(script, tmp_path, timeout=2400)
    _assert_cost_decreased(out, f"QA-turbulence-{kind}")
    assert "stage wall time" in out
    assert (tmp_path / f"wout_QA_turbulence_{kind}_optimized.nc").exists()


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


@pytest.mark.full  # nightly: Picard seed + one exact finite-beta stage, low bootstrap
@pytest.mark.parametrize(("script_name", "output", "dependency"), [
    ("QA_optimization_low_bootstrap.py", "QA_low_bootstrap_optimized", None),
    ("optimize_bootstrap_current.py", "bootstrap_current_optimized", "booz_xform_jax"),
    ("QI_optimization_bootstrap_dkx.py", "QI_bootstrap_dkx_optimized", "dkx"),
])
def test_low_bootstrap_optimization_examples(script_name, output, dependency, tmp_path):
    if dependency == "dkx":
        pytest.importorskip("booz_xform_jax")
    if dependency is not None:
        pytest.importorskip(dependency)
    out = _run_example(EXAMPLES / "optimization" / script_name, tmp_path, timeout=1800)
    costs = re.search(r"cost: initial ([0-9.eE+-]+) -> final ([0-9.eE+-]+)", out)
    assert costs is not None and float(costs.group(2)) < float(costs.group(1)), out[-2000:]
    assert "self-consistent seed" in out and "[final]" in out
    assert (tmp_path / f"wout_{output}.nc").exists()
    assert (tmp_path / f"{output}_current.png").exists()
    if script_name == "optimize_bootstrap_current.py":
        assert "trend toward QI:" in out
        assert (tmp_path / f"{output}_boozer_modB.png").exists()
    if dependency == "dkx":
        assert "Redl" in out and "DKX" in out


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


# Smoke on one office core, mostly XLA compilation of two traces.
@pytest.mark.full
def test_vmex_essos_tracing_to_wall_example(tmp_path):
    pytest.importorskip("essos")
    out = _run_example(EXAMPLES / "vmex_essos_tracing_to_wall.py", tmp_path, timeout=900)
    assert "max |dB|/|B|" in out and "strike the wall" in out
    assert re.search(r"stay inside +\d+ of 8", out)
    assert (tmp_path / "vmex_essos_tracing_to_wall.webp").stat().st_size > 10_000


def test_vmex_mrx_comparison_example(tmp_path):
    """The MRX comparison exits cleanly without MRX installed.

    No CI lane installs ``mrx``, so there the script prints the install
    command and stops before solving (seconds).  Where MRX is present the smoke
    pass solves a 9/17-surface case, takes one Newton step on a coarse mesh
    and draws the figure.
    """
    out = _run_example(EXAMPLES / "vmex_mrx_comparison.py", tmp_path, timeout=1200)
    if "MRX is not installed" in out:
        assert "pip install mrx" in out
        return
    assert "VMEX: iota axis/edge" in out
    assert "force residual" in out and "|B| difference" in out
    figure = tmp_path / "output_vmex_mrx_comparison" / "vmex_mrx_comparison.png"
    assert figure.stat().st_size > 10_000


def _pwo_component_score(scale=1.):
    families = []
    for index, (mean, variance) in enumerate(((7., .4), (2., .5), (3., .6))):
        endpoints = [[0., 0.], [1., 5.]] if index == 0 else [[1., 5.], [2., 6.]]
        left = 0. if index < 2 else 2.
        families.append(dict(mean_action_m=mean*scale, variance_m2=variance*scale**2,
            endpoints=endpoints, rows=[dict(u=u, weight=1/3, left=left, right=left+1.,
                J=(mean+offset*np.sqrt(1.5*variance))*scale)
                for u, offset in zip((.25, .5, .75), (-1., 0., 1.))]))
    return dict(families=families, per_pitch=[dict(level_T=.8, value=10.*scale**2),
                                           dict(level_T=.9, value=scale**2)])


def test_pwo_component_uses_separate_plateaus_and_fixed_raw_terms():
    import runpy
    terms = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))['component_terms']
    score = _pwo_component_score()
    assert terms(score, [0, 1, 2]) == pytest.approx([4., .4, .5, .6])
    assert terms(_pwo_component_score(2.), (0, 2, 1)) == pytest.approx([16., 1.6, 2.4, 2.])
    # Symmetric daughters may share a mean; the parent retains its own plateau.
    score['families'][2]['mean_action_m'] = 2.
    assert terms(score, [0, 1, 2])[0] == 9.


def test_pwo_family_gradient_avoids_scalar_variance_fd_sign_reversal():
    import runpy
    linearize = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))['family_gradient_rows']
    h = 1e-4
    def score(t):
        amplitude = 1e-3+t-2e5*t*t
        actions = 5.+amplitude*np.array([-1., 0., 1.])
        weights = np.array([.25, .5, .25]); mean = weights@actions
        family = dict(mean_action_m=mean, variance_m2=weights@((actions-mean)**2),
                      rows=[dict(J=J, weight=w) for J, w in zip(actions, weights)])
        return dict(families=[family], per_pitch=[dict(level_T=.8)])
    baseline, minus, plus = score(0.), score(-h), score(h)
    entry = dict(charts=[dict(level_T=.8, weight=1.)], length=.5)
    scalar_fd = (plus['families'][0]['variance_m2']-minus['families'][0]['variance_m2'])/(2*h*.5**2)
    chain_fd = (np.array(linearize(plus, baseline, entry))-
                linearize(minus, baseline, entry))/(2*h)
    assert scalar_fd < 0
    assert chain_fd == pytest.approx([1e-3/.5**2], rel=1e-9)


def test_pwo_family_gradient_keeps_pitch_weights_length_and_component_units():
    import runpy
    linearize = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))['family_gradient_rows']
    def score(scale):
        result = _pwo_component_score(scale)
        result['families'].append(result['families'][1])
        return result
    entry = dict(charts=[dict(level_T=level, weight=w)
                        for level, w in zip((.8, .8, .8, .9), (1., 2., 3., 999.))],
                 length=2., component=[0, 1, 2])
    baseline = score(1.); h = 1e-4
    derivative = (np.array(linearize(score(1+h), baseline, entry))-
                  linearize(score(1-h), baseline, entry))/(2*h)
    # Pitch averages divide by length squared; the four component terms remain m².
    assert derivative == pytest.approx([(1*.8+2*1.+3*1.2)/(6*4), 1./4, 8., .8, 1., 1.2])


@pytest.mark.parametrize('component', [None, [], [0, 1], [0, 1, 2, 3], [0, 1, 1],
                                       [True, 1, 2], [0., 1, 2], [-1, 1, 2], [0, 1, 3]])
def test_pwo_component_rejects_invalid_selector(component):
    import runpy
    terms = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))['component_terms']
    with pytest.raises(ValueError, match='three distinct'):
        terms(_pwo_component_score(), component)


@pytest.mark.parametrize('problem', ['event', 'daughter_event', 'shape', 'nodes', 'overlap',
                                    'crossed_order', 'nonfinite_node', 'action', 'width', 'mean', 'variance'])
def test_pwo_component_rejects_disconnected_or_invalid_families(problem):
    import runpy
    terms = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))['component_terms']
    score = _pwo_component_score(); parent, first, second = score['families']
    if problem == 'event':
        parent['endpoints'][1][0] += .1
    if problem == 'daughter_event':
        second['endpoints'][1][1] += .1
    if problem == 'shape':
        first['endpoints'] = [[1., 5.]]
    if problem == 'nodes':
        second['rows'][0]['u'] += .1
    if problem == 'overlap':
        second['rows'][0]['left'] = .5
    if problem == 'crossed_order':
        second['rows'][0].update(left=-2., right=-1.)
    if problem == 'nonfinite_node':
        second['rows'][0]['u'] = np.nan
    if problem == 'action':
        parent['rows'][0]['J'] = 0.
    if problem == 'width':
        parent['rows'][0]['right'] = -1.
    if problem == 'mean':
        parent['mean_action_m'] = np.nan
    if problem == 'variance':
        first['variance_m2'] = -1.
    with pytest.raises(ValueError):
        terms(score, [0, 1, 2])


@pytest.mark.parametrize('reject', [None, 'selected_pitch', 'other_pitch', 'other_surface', 'heldout', 'level'])
def test_pwo_component_step_retains_unselected_and_heldout_checks(tmp_path, monkeypatch, reject):
    import runpy
    from types import SimpleNamespace as NS
    example = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))
    run = example['run_physical_step']; env = run.__globals__
    monkeypatch.chdir(tmp_path); monkeypatch.setitem(env, 'ci_smoke', True)
    names = ['RBC(1,1)', 'ZBS(0,1)', 'ZBS(-1,1)', 'RBC(0,2)', 'RBC(0,1)', 'ZBS(1,1)', 'ZBS(2,1)']
    inp = NS(nfp=1, phiedge=.1, ntor=0, rbc=np.array([[1.]]), to_indata=lambda p: Path(p).touch())
    def equilibrium(x):
        return NS(state=x, runtime=None, result=NS(converged=True), wout=NS(aspect=6., Aminor_p=.2, volavgB=1.))
    problem = NS(names=names, x0=np.zeros(7), equilibrium_from_x=equilibrium, input_from_x=lambda x: inp)
    monkeypatch.setattr(example['opt'], 'VmecProblem', NS(from_loss=lambda *a, **k: problem))
    for name, value in [('min_abs_iota', .5), ('mean_iota', .5), ('mirror_ratio', .2), ('max_elongation', 2.)]:
        monkeypatch.setattr(example['opt'], name, lambda *a, v=value: v)
    spectrum = dict(bmnc_b=[[1., .2]], bmns_b=[[0., 0.]], xm_b=[0, 0], xn_b=[0, 1], nfp=1,
                    iota_b=[0.], I_b=[0.])
    monkeypatch.setitem(env, 'boozer_spectrum_state', lambda x, runtime, surfaces, **k:
                        dict(spectrum, s_b=surfaces, G_b=[1-float(np.sum(x))]))
    monkeypatch.setattr(example['vj'], 'write_wout', lambda p, w: Path(p).touch())
    class Families:
        def __init__(self, reference, charts, length):
            self.surface, self.heldout = reference['s_b'][0], charts[0]['heldout']
            self.initial = _pwo_component_score()
        def __call__(self, sp):
            q = sp['G_b'][0]; score = _pwo_component_score(q)
            score['families'].append(score['families'][1])
            if self.surface != .5:
                score['per_pitch'] = [dict(level_T=.8, value=q**2)]
            if reject == 'selected_pitch' and self.surface == .5:
                score['per_pitch'][0]['value'] = 10./q**2
            if ((reject == 'other_pitch' and self.surface == .5) or
                    (reject == 'other_surface' and self.surface == .25) or
                    (reject == 'heldout' and self.heldout)):
                score['per_pitch'][-1]['value'] = 1./q**2
            return score
    monkeypatch.setitem(env, 'PwOActionFamilies', Families)
    groups = [[dict(reference=dict(s_b=[s]), charts=[dict(level_T=level, weight=1., heldout=heldout)
                                                   for level in (.8, .8, .8, .9)],
                    length=1., **({'component': [0, 1, 2]} if s == .5 else {}))
               for s in (.5, .25, .75)] for heldout in (False, True)]
    if reject == 'level':
        groups[0][0]['charts'] = [dict(level_T=level, heldout=False) for level in (.8, .8, .9)]
    result = run(inp, None, lambda inp, eq: (inp, eq), output_name='component', seed_route='qi',
                 family_charts=groups, step_lengths=[.001])
    assert result['accepted'] == (reject is None)
    if reject == 'level':
        assert 'one reflecting level' in result['rejection'] and result['search_evaluations'] == 0
        return
    assert result['action_objective'] == 'connected_component'
    assert result['physical_reference']['family_charts'] == groups
    if reject is None:
        scored = result['trial_scores'][0][0]
        assert len(result['physical_reference']['normalization']) == 8
        assert len(scored['component_terms_m2']) == 4
    else:
        assert ('No common descent' in result.get('rejection', '') or
                'descent checks' in result['trials'][0]['rejection'])


def test_pwo_physical_rejection_preserves_accepted_checkpoint(tmp_path, monkeypatch):
    import runpy
    from types import SimpleNamespace

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    run = example["run_physical_step"]
    monkeypatch.chdir(tmp_path)
    for name in ("input.accepted", "wout_accepted.nc", "accepted_summary.json"):
        (tmp_path / name).write_text("previous accepted result")
    def fail(*args, **kwargs):
        raise RuntimeError("Unconverged equilibrium")
    monkeypatch.setattr(example["opt"], "VmecProblem", SimpleNamespace(from_loss=fail))
    monkeypatch.setitem(run.__globals__, "ci_smoke", False)
    with pytest.raises(SystemExit, match="1"):
        inp = SimpleNamespace(nfp=2, phiedge=.1, ntor=0, rbc=np.array([[1.]]))
        run(inp, None, None, output_name="accepted", seed_route="qi")
    result = json.loads((tmp_path / "accepted_rejected_summary.json").read_text())
    assert not result["accepted"] and result["rejection"] == "Unconverged equilibrium"
    assert result["search_evaluations"] == result["refinement_evaluations"] == 0
    for name in ("input.accepted", "wout_accepted.nc", "accepted_summary.json"):
        assert (tmp_path / name).read_text() == "previous accepted result"


@pytest.mark.parametrize("change", [{"nfp": 3}, {"phiedge": .2}, {"R00": 2.}])
def test_pwo_physical_restart_rejects_changed_invariants(tmp_path, monkeypatch, change):
    import runpy
    from types import SimpleNamespace

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    run = example["run_physical_step"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(run.__globals__, "ci_smoke", False)
    inp = SimpleNamespace(nfp=2, phiedge=.1, ntor=0, rbc=np.array([[1.]]))
    reference = dict(accepted=True, physical_reference=dict(identity=dict(nfp=2, phiedge=.1, R00=1.)))
    reference["physical_reference"]["identity"].update(change)
    with pytest.raises(SystemExit, match="1"):
        run(inp, None, None, output_name="restart", seed_route="qi", reference=reference)
    result = json.loads((tmp_path / "restart_summary.json").read_text())
    assert "same NFP, flux and R00" in result["rejection"]
    assert result["search_evaluations"] == result["refinement_evaluations"] == 0


@pytest.mark.parametrize("steps", [(), (0.,), (np.nan,), (.01, .1), (.3, .1, .03, .003)])
def test_pwo_physical_rejects_invalid_step_lengths(tmp_path, monkeypatch, steps):
    import runpy

    run = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))["run_physical_step"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(run.__globals__, "ci_smoke", False)
    with pytest.raises(SystemExit, match="1"):
        run(None, None, None, output_name="invalid", seed_route="qi", step_lengths=steps)
    result = json.loads((tmp_path / "invalid_summary.json").read_text())
    assert "step lengths" in result["rejection"]
    assert result["search_evaluations"] == result["refinement_evaluations"] == 0


@pytest.mark.parametrize('charts,restart', [([], False), ([[{}]*3], False), ([[{}]*2]*2, False), ([], True)])
def test_pwo_physical_rejects_invalid_family_configuration(tmp_path, monkeypatch, charts, restart):
    import runpy
    from types import SimpleNamespace
    run = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))['run_physical_step']
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(run.__globals__, 'ci_smoke', False)
    inp = SimpleNamespace(nfp=2, phiedge=.1, ntor=0, rbc=np.array([[1.]]))
    reference = dict(accepted=True, physical_reference=dict(identity=dict(nfp=2, phiedge=.1, R00=1.))) if restart else None
    with pytest.raises(SystemExit):
        run(inp, None, None, output_name='invalid', seed_route='qi', family_charts=charts, reference=reference)
    result = json.loads((tmp_path / 'invalid_summary.json').read_text())
    assert ('reuse its saved family charts' if restart else 'two sampling groups on three surfaces') in result['rejection']
    assert result['search_evaluations'] == 0


@pytest.mark.parametrize('reject_validation', [False, True])
def test_pwo_physical_family_step_checks_heldout_charts(tmp_path, monkeypatch, reject_validation):
    """Exercise search/acceptance with an analytic field and an equilibrium test double."""
    import runpy
    from types import SimpleNamespace as NS
    example = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))
    run = example['run_physical_step']; env = run.__globals__
    monkeypatch.chdir(tmp_path); monkeypatch.setitem(env, 'ci_smoke', True)
    names = ['RBC(1,1)', 'ZBS(0,1)', 'ZBS(-1,1)', 'RBC(0,2)', 'RBC(0,1)', 'ZBS(1,1)', 'ZBS(2,1)']
    inp = NS(nfp=1, phiedge=.1, ntor=0, rbc=np.array([[1.]]), to_indata=lambda p: Path(p).touch())
    def equilibrium(x):
        return NS(state=x, runtime=None, result=NS(converged=True),
                  wout=NS(aspect=6., Aminor_p=.2, volavgB=1.))
    problem = NS(names=names, x0=np.zeros(7), equilibrium_from_x=equilibrium, input_from_x=lambda x: inp)
    monkeypatch.setattr(example['opt'], 'VmecProblem', NS(from_loss=lambda *a, **k: problem))
    for name, value in [('min_abs_iota', .5), ('mean_iota', .5), ('mirror_ratio', .2), ('max_elongation', 2.)]:
        monkeypatch.setattr(example['opt'], name, lambda *a, v=value: v)
    spectrum = dict(bmnc_b=[[1., .2, .02]], bmns_b=[[0.]*3], xm_b=[0, 0, 1], xn_b=[0, 1, 0],
                    nfp=1, iota_b=[0.], G_b=[1.], I_b=[0.])
    monkeypatch.setitem(env, 'boozer_spectrum_state', lambda x, runtime, surfaces, **k:
                        dict(spectrum, s_b=surfaces, G_b=[1-float(np.sum(x))]))
    monkeypatch.setattr(example['vj'], 'write_wout', lambda p, w: Path(p).touch())
    groups = []
    for level in (.81, .805):
        lower = np.arccos((level-.8)/.02); nodes, weights = np.polynomial.legendre.leggauss(4); rows = []
        for u, weight in zip((nodes+1)/2, weights/2):
            alpha = lower+u*(2*np.pi-2*lower)
            width = np.arccos((1+.02*np.cos(alpha)-level)/.2)
            rows.append(dict(u=u, weight=weight, left=np.pi-width, right=np.pi+width))
        chart = dict(level_T=level, weight=1., endpoints=[[lower, np.pi], [2*np.pi-lower, np.pi]], rows=rows)
        groups.append([dict(reference=dict(spectrum, s_b=[s]), charts=[chart], length=.2) for s in (.5, .25, .75)])
    if reject_validation:
        original = env['PwOActionFamilies']
        def scorer(**kwargs):
            objective = original(**kwargs)
            def score(sp):
                result = objective(sp)
                if kwargs['charts'][0]['level_T'] == .805:
                    result['per_pitch'][0]['value'] = 1/result['per_pitch'][0]['value']
                return result
            return score
        monkeypatch.setitem(env, 'PwOActionFamilies', scorer)
    result = run(inp, None, lambda inp, eq: (inp, eq), output_name='family', seed_route='qi',
                 family_charts=groups, step_lengths=[.001])
    assert result['accepted'] != reject_validation
    assert result['action_objective'] == 'family_variance'
    assert result['parameters'] == names
    assert result['physical_reference']['family_charts'] == groups
    assert (tmp_path / 'wout_family.nc').exists() != reject_validation
    if reject_validation:
        assert 'descent checks' in result['trials'][0]['rejection']
    else:
        assert all(a['value'] < b['value'] for before, after in zip(result['refined_before'], result['refined_after'])
                   for b, a in zip(before, after))
        continued = run(inp, None, lambda inp, eq: (inp, eq), output_name='continued', seed_route='qi',
                        reference=result, step_lengths=[.001])
        assert continued['accepted']
        assert continued['physical_reference'] == result['physical_reference']


def test_pwo_physical_policy_dispatches_before_seed_preparation(tmp_path, monkeypatch):
    """A saved low-iota physical policy bypasses QP and surrogate preparation."""
    import runpy

    example = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))
    run = example['run']; env = run.__globals__
    env.update(METHOD='physical', SEED='near_circular', QS_INPUT='unused-qp-input')
    policy = dict(input='saved-low-iota-input', restart='frozen-seed-wout',
                  fixed_B_T=1., training=[], validation=[])
    path = tmp_path / 'policy.json'; path.write_text(json.dumps(policy))
    monkeypatch.setenv('VMEX_PWO_FAMILY_CHARTS', str(path))
    monkeypatch.setattr(example['vj'].VmecInput, 'from_file',
                        lambda *a, **k: pytest.fail('legacy seed preparation ran'))
    monkeypatch.setattr(example['opt'], 'QuasisymmetryRatioResidual',
                        lambda *a, **k: pytest.fail('QP preconditioning ran'))
    calls = []
    monkeypatch.setitem(env, 'run_centered_variance_step',
                        lambda p, **k: calls.append((p, k)) or {'accepted': True})
    assert run(output_name='low-iota') == {'accepted': True}
    assert calls == [(policy, {'output_name': 'low-iota'})]
    for kwargs in (dict(beta_targets=(0., .01)), dict(physical_steps=[.001])):
        with pytest.raises(ValueError):
            run(**kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize('smoke', [False, True])
def test_pwo_physical_policy_dispatch_reports_failed_screen(tmp_path, monkeypatch, smoke):
    import runpy

    run = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))['run']
    env = run.__globals__
    env.update(METHOD='physical', ci_smoke=smoke)
    path = tmp_path / 'policy.json'; path.write_text('{}')
    monkeypatch.setenv('VMEX_PWO_FAMILY_CHARTS', str(path))
    rejected = dict(accepted=False, rejection='unresolved finite gain')
    monkeypatch.setitem(env, 'run_centered_variance_step', lambda *a, **k: rejected)
    if smoke:
        assert run() == rejected
    else:
        with pytest.raises(SystemExit, match='1'):
            run()


@pytest.mark.parametrize('charts', [None, [[{'legacy': True}]*3]*2])
def test_pwo_physical_list_and_default_keep_legacy_dispatch(tmp_path, monkeypatch, charts):
    import runpy
    from types import SimpleNamespace as NS

    example = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))
    run = example['run']; env = run.__globals__
    env.update(METHOD='physical', SEED='qi', QS_INPUT=None)
    monkeypatch.delenv('VMEX_PWO_REFERENCE', raising=False)
    monkeypatch.delenv('VMEX_PWO_FAMILY_CHARTS', raising=False)
    if charts is not None:
        path = tmp_path / 'charts.json'; path.write_text(json.dumps(charts))
        monkeypatch.setenv('VMEX_PWO_FAMILY_CHARTS', str(path))
    monkeypatch.setattr(example['opt'], 'solve_equilibrium', lambda *a, **k: NS())
    monkeypatch.setitem(env, 'run_centered_variance_step',
                        lambda *a, **k: pytest.fail('legacy dispatch changed'))
    calls = []
    monkeypatch.setitem(env, 'run_physical_step',
                        lambda *a, **k: calls.append(k) or {'accepted': True})
    assert run(physical_steps=[.001]) == {'accepted': True}
    assert calls[0]['family_charts'] == charts
    assert calls[0]['step_lengths'] == [.001]
    assert calls[0]['seed_route'] == 'qi'


def _pwo_centered_case(tmp_path, monkeypatch, mode='descent'):
    """Analytic family actions with a distinct field-amplitude direction."""
    import copy
    import runpy
    from dataclasses import replace
    from types import SimpleNamespace as NS

    example = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))
    run = example['run_centered_variance_step']; env = run.__globals__
    monkeypatch.chdir(tmp_path)
    deck = example['vj'].VmecInput.from_file(DATA_DIR / 'input.minimal_seed_nfp2')
    opt = example['opt']; x0 = opt.pack_boundary(deck, 1, vary_major_radius=False)
    input_path = tmp_path / 'input.seed'; deck.to_indata(input_path)
    seed_path = tmp_path / 'wout_seed.nc'; seed_path.touch()
    restarts, solves, normalized, reads = [], [], [], []
    frozen_source = object()

    def read_wout(source):
        assert Path(source) == seed_path
        reads.append(source)
        return frozen_source

    def restart(source, inp):
        assert source is frozen_source
        token = np.zeros(1); restarts.append(token)
        return token

    def solve(inp, *, initial_state, raise_on_max_iterations):
        assert raise_on_max_iterations and initial_state is restarts[-1]
        assert initial_state[0] == 0.  # Any reuse of a previous solve is visible.
        initial_state[0] = len(restarts)
        y = (opt.pack_boundary(inp, 1, vary_major_radius=False)-x0)/.02
        assert inp.nfp == deck.nfp and inp.phiedge == deck.phiedge
        assert inp.rbc[inp.ntor, 0] == deck.rbc[deck.ntor, 0]
        assert inp.mpol == deck.mpol and inp.ntor == deck.ntor
        np.testing.assert_array_equal(inp.ns_array, deck.ns_array)
        tolerance = float(inp.ftol_array[-1]); amplitude = 2.*(1+.2*y[0])
        if mode == 'amplitude':
            amplitude = 2.
        solves.append((y.copy(), tolerance))
        result = NS(converged=mode != 'unconverged', fsqr=tolerance/2,
                    fsqz=tolerance/3, fsql=tolerance/4)
        if mode == 'force' and np.linalg.norm(y) > .001:
            result.fsqr = 2*tolerance
        if mode == 'nonfinite_field':
            amplitude = np.nan
        stored_input = replace(inp, ftol_array=np.asarray(inp.ftol_array)*10) if mode == 'stored_ftol' else inp
        return NS(inp=stored_input, state=NS(y=y, amplitude=amplitude, tolerance=tolerance),
                  runtime=None, result=result, wout=NS(volavgB=amplitude))

    def spectrum(state, runtime, surfaces, **options):
        assert surfaces == [.5]
        amplitude = state.amplitude
        shift = .1 if ((mode == 'tolerance' and state.tolerance != deck.ftol_array[-1]) or
                       (mode == 'band' and options['mboz'] == 20)) else 0.
        # All four dimensionful arrays require the same fixed-B normalization.
        return dict(bmnc_b=amplitude*np.array([[1., .2]]),
                    bmns_b=amplitude*np.array([[0., .1]]), xm_b=np.array([0, 1]),
                    xn_b=np.array([0, 0]), iota_b=np.array([.02]),
                    G_b=np.array([amplitude*(1+state.y[1]+shift)]),
                    I_b=np.array([amplitude*(.4 if mode == 'crossed' and
                        state.tolerance != deck.ftol_array[-1] and options['mboz'] == 10 else .3)]), nfp=2,
                    s_b=np.array([.6] if mode == 'surface' else surfaces))

    class Families:
        def __init__(self, reference, charts, length):
            if isinstance(reference['nfp'], (bool, np.bool_)) or not isinstance(reference['nfp'], (int, np.integer)):
                raise ValueError('nfp must be an integer scalar')
            self.reference, self.charts, self.length = reference, charts, length
            self.weights = np.array([c['weight'] for c in charts], dtype=float)
            self.weights /= self.weights.sum()

        def __call__(self, sp):
            assert isinstance(sp['nfp'], (int, np.integer)) and not isinstance(sp['nfp'], (bool, np.bool_))
            np.testing.assert_allclose(sp['bmnc_b'], [[1., .2]], atol=1e-14)
            np.testing.assert_allclose(sp['bmns_b'], [[0., .1]], atol=1e-14)
            assert any(np.allclose(sp['I_b'], [value], atol=1e-14)
                       for value in ((.3, .4) if mode == 'crossed' else (.3,)))
            np.testing.assert_array_equal(sp['iota_b'], [.02])
            t = float(sp['G_b'][0])-1.; normalized.append(t)
            curvature = {'half': 50., 'no_descent': 100.}.get(mode, 0.)
            factor = 1-t+curvature*t*t
            if mode == 'validation' and self.charts[0].get('heldout'):
                factor = 1+t
            if mode == 'nodes' and self.charts[0].get('heldout'):
                factor *= 1.2
            if mode == 'crossed' and not self.charts[0].get('heldout') and sp['I_b'][0] > .35:
                factor *= 1.2
            families = []
            for index, chart in enumerate(self.charts):
                weights = np.array([r['weight'] for r in chart['rows']])
                offsets = np.arange(len(weights), dtype=float)
                offsets -= weights@offsets
                local = 1+t if mode == 'canonical' or (mode == 'parent' and index == 0) else factor
                actions = 10.+100*t+(index+1)*local*offsets
                slopes = (1+t if mode == 'slope' or (mode == 'family_slope' and index == 0)
                          else factor)*offsets
                mean = weights@actions
                family_rows = [dict(u=r['u'], weight=r['weight'], J=J, dJ=dJ, left=.2+.3*index)
                               for r, J, dJ in zip(chart['rows'], actions, slopes)]
                if mode == 'quadrature':
                    family_rows[0]['weight'] += .01
                families.append(dict(mean_action_m=float(mean),
                    alpha_domain=[.2, .2+(.5+index)*(1+.4*t)],
                    variance_m2=float(weights@((actions-mean)**2)),
                    raw_Jalpha2=float(weights@(slopes**2)), rows=family_rows))
            value = sum(w*f['variance_m2'] for w, f in zip(self.weights, families))/self.length**2
            raw_q = sum(w*sum(r['weight']*r['dJ']**2 for r in f['rows'])
                        for w, f in zip(self.weights, families))
            if mode == 'bad_variance':
                value *= 2
            return dict(value=float(value), raw_Jalpha2=float(raw_q), families=families)

    reference = dict(nfp=2, s_b=[.5], xm_b=[0, 1], xn_b=[0, 0])
    charts = [dict(id=f'family-{i}', level_T=.8, weight=w, endpoints=[[0., 0.], [1., 1.]],
                   rows=[dict(u=u, weight=q) for u, q in zip((.2, .5, .8), (.2, .3, .5))])
              for i, w in enumerate((1., 3.))]
    training = dict(reference=reference, charts=charts, length=2.)
    validation = copy.deepcopy(training)
    for chart in validation['charts']:
        chart['heldout'] = True
    policy = dict(input=str(input_path), restart=str(seed_path), fixed_B_T=1.,
                  training=training, validation=validation, parent_families=['family-0'])
    monkeypatch.setitem(env, 'restart_state', restart)
    monkeypatch.setitem(env, 'boozer_spectrum_state', spectrum)
    monkeypatch.setitem(env, 'PwOActionFamilies', Families)
    monkeypatch.setattr(opt, 'solve_equilibrium', solve)
    monkeypatch.setattr(example['vj'], 'read_wout', read_wout)
    monkeypatch.setattr(example['vj'], 'write_wout', lambda path, wout: Path(path).touch())
    return run, policy, NS(restarts=restarts, solves=solves, normalized=normalized, reads=reads, deck=deck)


@pytest.mark.parametrize('reference_files', [False, True])
def test_pwo_centered_family_step_uses_fixed_weights_and_independent_restarts(tmp_path, monkeypatch, reference_files):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch)
    if reference_files:
        for group in ('training', 'validation'):
            reference = tmp_path / f'{group}.npz'
            np.savez(reference, **policy[group]['reference'])
            policy[group]['reference'] = str(reference)
    result = run(policy, output_name='centered')
    assert result['accepted'] and not result['pwO_qualified']
    assert not result['research_validation_complete']
    assert len(result['parameters']) == 8
    assert len(calls.reads) == 1
    assert len(calls.restarts) == len({id(s) for s in calls.restarts}) == len(calls.solves)
    assert result['MHD_solves'] == len(calls.solves)
    first = calls.solves[:17]
    np.testing.assert_array_equal(first[0][0], np.zeros(8))
    for j in range(8):
        np.testing.assert_allclose(first[1+2*j][0], -1e-4*np.eye(8)[j], atol=1e-14)
        np.testing.assert_allclose(first[2+2*j][0], 1e-4*np.eye(8)[j], atol=1e-14)
    residual = np.asarray(result['model']['centered_action_residual'])
    centered = np.array([0., 1., 2.])-1.3
    expected = np.r_[np.sqrt(.25*np.array([.2, .3, .5]))*centered/2,
                     np.sqrt(.75*np.array([.2, .3, .5]))*centered]
    np.testing.assert_allclose(residual, expected, atol=1e-14)
    jacobian = np.asarray(result['model']['centered_action_jacobian'])
    np.testing.assert_allclose(jacobian[:, 1], -expected, atol=2e-11)
    np.testing.assert_allclose(np.delete(jacobian, 1, axis=1), 0., atol=2e-11)
    step = np.asarray(result['model']['proposal'])
    assert np.linalg.norm(step) <= policy.get('trust_radius', .03)*(1+1e-12)
    assert abs(np.asarray(result['model']['amplitude_gradient'])@step) < 1e-12
    assert result['cases']['center']['normalization_factor'] == .5
    assert calls.normalized and (tmp_path / 'wout_centered.nc').exists()
    saved = json.loads((tmp_path / 'centered_summary.json').read_text())
    assert saved['accepted'] and saved['model'] == result['model']


@pytest.mark.parametrize('mode,guarded,protected,accepted,fraction', [
    ('half', True, True, True, .5), ('no_descent', True, True, False, None),
    ('validation', True, True, False, None), ('slope', True, True, False, None),
    ('slope', False, True, True, 1.), ('parent', True, True, False, None),
    ('parent', True, False, True, 1.),
])
def test_pwo_centered_family_step_requires_actual_guarded_progress(
        tmp_path, monkeypatch, mode, guarded, protected, accepted, fraction):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch, mode)
    policy['guarded'] = guarded
    if not protected:
        policy['parent_families'] = []
    result = run(policy, output_name='progress')
    assert result['accepted'] == accepted
    assert result.get('accepted_fraction') == fraction
    assert (tmp_path / 'wout_progress.nc').exists() == accepted
    if mode in ('half', 'no_descent'):
        assert result['model']['predicted_V'] < np.sum(np.square(result['model']['centered_action_residual']))
        assert len(result['trials']) == 2 and not result['trials'][0]['accepted']
    assert all(np.linalg.norm(y) <= .03*(1+1e-10) for y, tolerance in calls.solves)


@pytest.mark.parametrize('mode,message', [
    ('unconverged', 'actual stored FTOL'), ('force', 'actual stored FTOL'),
    ('stored_ftol', 'actual stored FTOL'),
    ('nonfinite_field', 'volume-average field'), ('quadrature', 'quadrature identity'),
    ('bad_variance', 'aggregate variance'), ('surface', 'surface or field period'),
    ('amplitude', 'amplitude derivative'),
])
def test_pwo_centered_family_step_rejects_invalid_numerical_evidence(tmp_path, monkeypatch, mode, message):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch, mode)
    result = run(policy, output_name='invalid-evidence')
    assert not result['accepted']
    errors = [result.get('rejection', '')]+[t.get('rejection', '') for t in result['trials']]
    assert any(message in error for error in errors)
    assert not (tmp_path / 'wout_invalid-evidence.nc').exists()


@pytest.mark.parametrize('mode,source', [('nodes', 'quadrature_nodes'), ('band', 'spectral_band'),
                                      ('tolerance', 'equilibrium_tolerance'), ('crossed', 'spectral_band')])
def test_pwo_centered_family_step_does_not_accept_unresolved_finite_gain(tmp_path, monkeypatch, mode, source):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch, mode)
    result = run(policy, output_name='uncertainty')
    assert not result['accepted'] and not result['finite_screen_pass']
    for trial in result['trials']:
        assessment = trial['assessment']['V']
        assert assessment['gain'] > 0  # The actual action improves on the finest check.
        assert assessment['classification'] == 'unresolved_observed_changes'
        assert any(change[source] > 0 for change in assessment['state_changes'])
        assert not trial['checks']['V']
    assert len(result['trials']) == 2


@pytest.mark.parametrize('change', ['field', 'nonfinite_field', 'half_step', 'trust_radius',
    'scales', 'ftol_check', 'surface', 'id', 'duplicate', 'level', 'weights', 'length',
    'validation_surface', 'nfp', 'float_nfp', 'fractional_nfp', 'bool_nfp',
    'parent', 'family_guard', 'guarded', 'empty_id', 'endpoints'])
def test_pwo_centered_family_policy_rejects_inconsistent_fixed_data(tmp_path, monkeypatch, change):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch)
    if change in ('field', 'nonfinite_field'):
        policy['fixed_B_T'] = 0. if change == 'field' else np.nan
    elif change in ('half_step', 'trust_radius'):
        policy[change] = 0.
    elif change == 'scales':
        policy['scales'] = [np.nan]*8
    elif change == 'ftol_check':
        policy['ftol_check'] = float(calls.deck.ftol_array[-1])
    elif change == 'surface':
        policy['training']['reference'].pop('s_b')
    elif change == 'id':
        policy['validation']['charts'][0]['id'] = 'changed'
    elif change == 'duplicate':
        for group in ('training', 'validation'):
            policy[group]['charts'][1]['id'] = 'family-0'
    elif change == 'level':
        policy['validation']['charts'][0]['level_T'] += .01
    elif change == 'endpoints':
        policy['validation']['charts'][0]['endpoints'][1][0] += .01
    elif change == 'weights':
        policy['validation']['charts'][0]['weight'] = 2.
    elif change == 'length':
        policy['validation']['length'] = 3.
    elif change == 'validation_surface':
        policy['validation']['reference']['s_b'] = [.6]
    elif change == 'nfp':
        policy['validation']['reference']['nfp'] = 3
    elif change in ('float_nfp', 'fractional_nfp', 'bool_nfp'):
        policy['training']['reference']['nfp'] = dict(float_nfp=2., fractional_nfp=2.1, bool_nfp=True)[change]
    elif change == 'parent':
        policy['parent_families'] = ['absent-family']
    elif change == 'family_guard':
        policy['family_guards'] = ['absent-family']
    elif change == 'guarded':
        policy['guarded'] = 'false'
    elif change == 'empty_id':
        for group in ('training', 'validation'):
            policy[group]['charts'][1]['id'] = ''
    with pytest.raises(ValueError):
        run(policy, output_name='bad-policy')
    assert not calls.solves


@pytest.mark.parametrize('step', [[.001]*7, [[.001]*4]*2, [np.nan]+[0.]*7,
                                 [np.inf]+[0.]*7, [0.]*8, [.031]+[0.]*7])
def test_pwo_centered_supplied_step_rejects_invalid_displacement_before_solves(tmp_path, monkeypatch, step):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch)
    policy['proposed_step'] = step
    with pytest.raises(ValueError, match='proposed_step'):
        run(policy, output_name='invalid-step')
    assert not calls.solves and not calls.reads


@pytest.mark.parametrize('mode,guard_family,accepted', [
    ('descent', False, True), ('half', False, False),
    ('family_slope', False, True), ('family_slope', True, False),
])
def test_pwo_centered_supplied_step_has_one_independent_screen_without_fd(
        tmp_path, monkeypatch, mode, guard_family, accepted):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch, mode)
    step = np.zeros(8); step[1] = .03
    policy['proposed_step'] = step.tolist()
    if guard_family:
        policy['family_guards'] = ['family-0']
    def no_optimizer(*args, **kwargs):
        raise AssertionError('Caller-supplied step must skip optimization')
    monkeypatch.setitem(run.__globals__, 'minimize', no_optimizer)
    result = run(policy, output_name='fixed-step')
    assert result['accepted'] == accepted
    assert result['proposal_source'] == 'caller_supplied'
    assert result['model']['derivative_validated_by_driver'] is False
    np.testing.assert_array_equal(result['model']['proposal'], step)
    assert result['MHD_solves'] == len(calls.solves) == 4
    assert len(calls.restarts) == len({id(s) for s in calls.restarts}) == 4
    assert set(result['cases']) == {'center', 'check-center', 'full', 'check-full'}
    assert len(result['trials']) == 1 and result['trials'][0]['fraction'] == 1.
    for (y, tolerance), expected in zip(calls.solves, (np.zeros(8), np.zeros(8), step, step)):
        np.testing.assert_allclose(y, expected, atol=1e-14)
    if guard_family:
        trial = result['trials'][0]
        assert trial['checks']['Q']  # Aggregate improvement does not hide a family regression.
        assert not trial['checks']['family-0_Q']
        assert trial['assessment']['family-0_Q']['classification'] == 'increase_beyond_observed_changes'


def test_pwo_centered_family_slope_guard_applies_to_optimizer_model(tmp_path, monkeypatch):
    from types import SimpleNamespace

    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch, 'family_slope')
    policy['family_guards'] = ['family-0']
    def infeasible_proposal(*args, **kwargs):
        z = np.zeros(7); z[0] = 1.
        margins = kwargs['constraints'][0]['fun'](z)
        assert margins[-1] < 0  # The family Q guard rejects the aggregate-descent direction.
        return SimpleNamespace(x=z, success=True, status=0, message='forced proposal', nit=1)
    monkeypatch.setitem(run.__globals__, 'minimize', infeasible_proposal)
    result = run(policy, output_name='guard-model')
    assert not result['accepted'] and 'No feasible descending' in result['rejection']
    assert result['MHD_solves'] == len(calls.solves) == 17 and not result['trials']


@pytest.mark.parametrize('canonical', [False, True])
def test_pwo_policy_dispatch_preserves_explicit_objective(tmp_path, monkeypatch, canonical):
    import runpy

    run = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))['run']
    env = run.__globals__; env.update(METHOD='physical', SEED='near_circular')
    policy = {'objective': 'canonical_slope' if canonical else 'centered_variance'}
    path = tmp_path/'policy.json'; path.write_text(json.dumps(policy))
    monkeypatch.setenv('VMEX_PWO_FAMILY_CHARTS', str(path))
    calls = []
    monkeypatch.setitem(env, 'run_centered_variance_step',
                        lambda p, **k: calls.append(p) or {'accepted': True})
    assert run()['accepted'] and calls == [policy]


def test_pwo_canonical_rows_use_live_unequal_spans_and_declared_pitch_mass():
    """Direct integration oracle exposes fixed-weight and /occupied-measure bugs."""
    import copy
    import runpy
    from types import SimpleNamespace as NS

    example = runpy.run_path(str(EXAMPLES/'optimization'/'pwO_optimization.py'))
    rows = example['canonical_slope_rows']; weights = example['canonical_pitch_weights']
    charts = [dict(level_T=level, weight=999., rows=[dict(u=.25, weight=.4), dict(u=.75, weight=.6)])
              for level in (.8, .8, 1.1)]
    scorer = NS(charts=charts, reference={'nfp': 2}, length=2.)
    score = {'families': [dict(alpha_domain=[a, b], rows=[
        dict(u=.25, weight=.4, dJ=d, left=.2), dict(u=.75, weight=.6, dJ=2*d, left=.3)])
        for (a, b), d in zip(((0., .2), (.4, 1.5), (2., 2.7)), (2., 3., 5.))]}
    policy = dict(pitch_weights=[dict(level_T=.8, weight=.3), dict(level_T=1.1, weight=.7)])
    pitches = weights(policy, charts)
    expected = np.concatenate([np.sqrt(pitch*span*np.array([.4, .6])/(2*np.pi))*np.array([d, 2*d])/2
        for pitch, span, d in ((.3, .2, 2.), (.3, 1.1, 3.), (.7, .7, 5.))])
    q = rows(score, scorer, pitches)
    np.testing.assert_allclose(q, expected, rtol=2e-15)
    moved = copy.deepcopy(score); moved['families'][1]['alpha_domain'][1] += .6
    expected_M = q@q + .3*.6/(2*np.pi)*(.4*3**2+.6*6**2)/4
    actual = rows(moved, scorer, pitches)
    assert actual@actual == pytest.approx(expected_M)
    assert not np.isclose(actual@actual, q@q)
    moved['families'][0]['rows'][0]['left'] = np.pi
    with pytest.raises(ValueError, match='ownership'):
        rows(moved, scorer, pitches)
    moved = copy.deepcopy(score); moved['families'][0]['alpha_domain'] = [1., .5]
    with pytest.raises(ValueError, match='span'):
        rows(moved, scorer, pitches)
    moved = copy.deepcopy(score); moved['families'][0]['rows'][0]['weight'] = .5
    with pytest.raises(ValueError, match='quadrature'):
        rows(moved, scorer, pitches)
    moved = copy.deepcopy(score); moved['families'][0]['rows'][0]['dJ'] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        rows(moved, scorer, pitches)


@pytest.mark.parametrize('records', [None, [], {}, [{'level_T': .8}],
    [{'level_T': .8, 'weight': 0.}], [{'level_T': .8, 'weight': np.nan}],
    [{'level_T': .8, 'weight': .4}], [{'level_T': .9, 'weight': 1.}],
    [{'level_T': .8, 'weight': .5}, {'level_T': .8, 'weight': .5}],
    [{'level_T': .8, 'weight': 1., 'extra': 0}]])
def test_pwo_canonical_policy_rejects_invalid_pitch_masses_before_solves(tmp_path, monkeypatch, records):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch)
    policy.update(objective='canonical_slope', pitch_weights=records)
    with pytest.raises(ValueError, match='pitch_weights'):
        run(policy)
    assert not calls.solves


def test_pwo_canonical_policy_rejects_unknown_objective(tmp_path, monkeypatch):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch)
    policy['objective'] = 'unknown'
    with pytest.raises(ValueError, match='objective'):
        run(policy)
    assert not calls.solves


def test_pwo_canonical_native_gn_and_finite_screen_optimize_live_slope(tmp_path, monkeypatch):
    """Slope falls while centered variance rises; native GN must follow slope."""
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch, 'canonical')
    policy.update(objective='canonical_slope', pitch_weights=[dict(level_T=.8, weight=1.)], guarded=False)
    result = run(policy, output_name='canonical')
    assert result['accepted'] and result['method'] == 'canonical_family_slope'
    assert result['objective'] == 'canonical_slope' and not result['pwO_qualified']
    model = result['model']; q0 = np.asarray(model['objective_residual'])
    expected = np.r_[np.sqrt(.5/(2*np.pi)*np.array([.2, .3, .5])),
                     np.sqrt(1.5/(2*np.pi)*np.array([.2, .3, .5]))]*(np.tile([0., 1., 2.], 2)-1.3)/2
    np.testing.assert_allclose(q0, expected, atol=1e-14)
    np.testing.assert_allclose(np.asarray(model['objective_jacobian'])[:, 1], -.8*q0, atol=2e-10)
    assert model['proposal'][1] > 0 and model['predicted_objective'] < model['center_objective']
    assert model['predicted_V'] > model['center_V']
    trial = result['trials'][0]
    assert trial['checks'] == {'M': True}
    assert trial['assessment']['M']['classification'] == 'decrease_beyond_observed_changes'
    assert trial['assessment']['V']['change_percent'] > 0
    assert result['MHD_solves'] == len(calls.solves)


@pytest.mark.parametrize('mode,source', [('nodes', 'quadrature_nodes'), ('band', 'spectral_band'),
                                      ('tolerance', 'equilibrium_tolerance')])
def test_pwo_canonical_supplied_step_requires_resolved_m(tmp_path, monkeypatch, mode, source):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch, mode)
    step = np.zeros(8); step[1] = .03
    policy.update(objective='canonical_slope', pitch_weights=[dict(level_T=.8, weight=1.)],
                  proposed_step=step.tolist(), guarded=False)
    result = run(policy, output_name='canonical-unresolved')
    assert not result['accepted'] and len(result['trials']) == 1
    assessment = result['trials'][0]['assessment']['M']
    assert assessment['gain'] > 0 and assessment['classification'] == 'unresolved_observed_changes'
    assert any(change[source] > 0 for change in assessment['state_changes'])
    assert not result['trials'][0]['checks']['M']


@pytest.mark.parametrize('key,value', [('periodic', True), ('seam_roots', [1., 2.])])
def test_pwo_family_policy_requires_matching_periodic_metadata(tmp_path, monkeypatch, key, value):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch)
    policy['validation']['charts'][0][key] = value
    with pytest.raises(ValueError, match='periodic seams'):
        run(policy)
    assert not calls.solves


def _pwo_balanced_case(tmp_path, monkeypatch, *, nonlinear=False, zero_pitch=False):
    """Three coupled pitches: one shape direction trades off the middle pitch."""
    import copy

    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch)
    env = run.__globals__; original_spectrum = env['boozer_spectrum_state']
    original_families = env['PwOActionFamilies']
    def spectrum(state, *args, **kwargs):
        sp = original_spectrum(state, *args, **kwargs)
        sp['I_b'] = np.array([state.amplitude*(.3+state.y[2])])
        return sp
    class Families(original_families):
        def __call__(self, sp):
            result = super().__call__(dict(sp, I_b=np.array([.3])))
            t, u = float(sp['G_b'][0])-1., float(sp['I_b'][0])-.3
            factors = [1-2*t, 1+.2*t-.25*u, 1-2*t]
            if nonlinear:
                factors[1] += 10*(t*t+u*u)
            if zero_pitch:
                factors[1] = 0.
            for family, factor in zip(result['families'], factors, strict=True):
                weights = np.array([r['weight'] for r in family['rows']])
                offsets = np.arange(len(weights)); offsets = offsets-weights@offsets
                for row, offset in zip(family['rows'], offsets, strict=True):
                    row['dJ'] = factor*offset
                family['raw_Jalpha2'] = float(weights@(factor*offsets)**2)
            result['raw_Jalpha2'] = float(self.weights@np.array([f['raw_Jalpha2'] for f in result['families']]))
            return result
    levels = [.8, .9, 1.]
    for key in ('training', 'validation'):
        source = policy[key]['charts'][0]
        policy[key]['charts'] = [dict(copy.deepcopy(source), id=f'coupled-{i}', level_T=level, weight=1.)
                                 for i, level in enumerate(levels)]
    # Deliberately differ from chart order; finite pitch labels must stay stable.
    policy.update(objective='canonical_slope', step_strategy='maximin_relative', guarded=False,
                  pitch_weights=[dict(level_T=level, weight=1/3) for level in [.9, 1., .8]],
                  parent_families=[], family_guards=[])
    monkeypatch.setitem(env, 'boozer_spectrum_state', spectrum)
    monkeypatch.setitem(env, 'PwOActionFamilies', Families)
    return run, policy, calls


def test_pwo_three_pitch_fit_balances_a_coupled_tradeoff():
    import runpy

    fit = runpy.run_path(str(EXAMPLES/'optimization'/'pwO_optimization.py'))['fit_action_step']
    q = np.ones(3); matrix = np.array([[-.4, 0.], [.04, -.05], [-.4, 0.]])
    def constraints(z):
        return np.array([1-z@z])
    def jac(z):
        return np.array([-2*z])
    aggregate = fit(q, matrix, constraints, jac)
    balanced = fit(q, matrix, constraints, jac, pitch_blocks=[np.array([i]) for i in range(3)])
    assert aggregate.success and balanced.success
    assert (q+matrix@aggregate.x)[1]**2 > q[1]**2
    after = q+matrix@balanced.x[:-1]
    assert np.all(after**2 < q**2) and balanced.x[-1] > 0
    np.testing.assert_allclose(1-after**2, balanced.x[-1], atol=1e-9)
    with pytest.raises(ValueError, match='positive baseline M'):
        fit(np.array([1., 0., 1.]), matrix, constraints, jac,
            pitch_blocks=[np.array([i]) for i in range(3)])


@pytest.mark.parametrize('objective,strategy', [('canonical_slope', 'unknown'),
    ('canonical_slope', None), ('centered_variance', 'maximin_relative')])
def test_pwo_step_strategy_rejects_invalid_policy_before_solves(tmp_path, monkeypatch, objective, strategy):
    run, policy, calls = _pwo_centered_case(tmp_path, monkeypatch)
    policy.update(objective=objective, step_strategy=strategy)
    with pytest.raises(ValueError, match='step_strategy'):
        run(policy)
    assert not calls.solves


@pytest.mark.parametrize('supplied', [False, True])
def test_pwo_maximin_rejects_zero_pitch_before_fd_or_finite_trial(tmp_path, monkeypatch, supplied):
    run, policy, calls = _pwo_balanced_case(tmp_path, monkeypatch, zero_pitch=True)
    if supplied:
        policy['proposed_step'] = [0., .003, .029, 0., 0., 0., 0., 0.]
    result = run(policy, output_name='zero-pitch')
    assert not result['accepted'] and 'positive baseline M' in result['rejection']
    assert len(calls.solves) == 1


def test_pwo_native_maximin_uses_existing_fd_and_stable_pitch_mapping(tmp_path, monkeypatch):
    run, policy, calls = _pwo_balanced_case(tmp_path, monkeypatch)
    result = run(policy, output_name='balanced-native')
    assert result['accepted'] and result['step_strategy'] == 'maximin_relative'
    assert result['MHD_solves'] == len(calls.solves) == 20
    assert result['model']['maximin_t'] > 0
    assert min(result['model']['predicted_pitch_relative_decreases']) > 0
    assert result['balanced_pitches_T'] == [.9, 1., .8]
    center = result['cases']['center']['training']
    assert [r['level_T'] for r in center['canonical_M_per_pitch']] == [.9, 1., .8]
    expected = np.array([1.5, 2.5, .5])*.61/(8*np.pi)
    np.testing.assert_allclose([r['M'] for r in center['canonical_M_per_pitch']], expected, atol=1e-14)
    assert sum(r['M']/3 for r in center['canonical_M_per_pitch']) == pytest.approx(center['canonical_M'])
    trial = result['trials'][0]
    assert all(trial['checks'][f'pitch_{i}_M'] for i in range(3))
    assert all(trial['assessment'][f'pitch_{i}_M']['change_percent'] < 0 for i in range(3))


@pytest.mark.parametrize('nonlinear,strategy,accepted', [(False, 'maximin_relative', True),
    (True, 'aggregate', True), (True, 'maximin_relative', False)])
def test_pwo_supplied_step_balances_actual_pitches_not_only_aggregate(
        tmp_path, monkeypatch, nonlinear, strategy, accepted):
    run, policy, calls = _pwo_balanced_case(tmp_path, monkeypatch, nonlinear=nonlinear)
    policy.update(step_strategy=strategy, proposed_step=[0., .003, .029, 0., 0., 0., 0., 0.])
    monkeypatch.setitem(run.__globals__, 'minimize',
                        lambda *args, **kwargs: pytest.fail('Supplied step refitted'))
    result = run(policy, output_name='balanced-supplied')
    assert result['accepted'] == accepted and result['MHD_solves'] == len(calls.solves) == 4
    assert result['proposal_source'] == 'caller_supplied' and len(result['trials']) == 1
    trial = result['trials'][0]
    assert trial['checks']['M']  # Aggregate improves even when the middle pitch regresses.
    if strategy == 'maximin_relative':
        assert trial['checks']['pitch_0_M'] == (not nonlinear)  # Declared first pitch is .9.
        if nonlinear:
            assert trial['assessment']['pitch_0_M']['classification'] == 'increase_beyond_observed_changes'
    else:
        assert trial['checks'] == {'M': True}


def _pwo_confinement_case(tmp_path, monkeypatch, mode='valid', lasym=False):
    """Real INDATA and Fourier updates; analytic external MHD/Boozer/NEO services."""
    import runpy
    from dataclasses import replace
    from types import SimpleNamespace as NS
    import scipy.optimize
    example = runpy.run_path(str(EXAMPLES / 'optimization' / 'pwO_optimization.py'))
    run = example['run_confinement']; env = run.__globals__; opt = example['opt']
    monkeypatch.chdir(tmp_path)
    seed = example['vj'].VmecInput.from_file(DATA_DIR / 'input.minimal_seed_nfp2').change_resolution(mpol=5, ntor=4)
    seed = replace(seed, lasym=lasym, ncurr=1, curtor=321., pres_scale=123.,
                   am=np.array([1., -.4, .1]), ac=np.array([1., .2]), ai=np.array([.4, .03]))
    arrays = {name: np.array(getattr(seed, name), copy=True) for name in ['rbc', 'zbs', 'raxis_c', 'zaxis_s']}
    arrays['rbc'][seed.ntor+1, 4] = .0027; arrays['zbs'][seed.ntor-2, 3] = .0043
    arrays['raxis_c'][1] = .03; arrays['zaxis_s'][2] = .02
    seed = replace(seed, **arrays)
    input_path = tmp_path / 'input.external'; seed.to_indata(input_path)
    seed = example['vj'].VmecInput.from_file(input_path)
    parameters = ['RBC(-1,3)', 'ZBS(2,4)']+(['RBS(1,3)', 'ZBC(-2,4)'] if lasym else [])
    names = opt.boundary_dof_names(seed, 4); selected = [names.index(n) for n in parameters]
    x0 = opt.pack_boundary(seed, 4); calls = NS(decks=[], restarts=[], restart_sources=[], writes=[], local=[], global_=0, census=[])
    source = object(); monkeypatch.setattr(example['vj'], 'read_wout', lambda _: source)

    def restart(wout, deck):
        assert wout is source or any(name.endswith('_seed_check.nc') and saved is wout for name, saved in calls.writes)
        calls.restart_sources.append(wout)
        token = [False]; calls.restarts.append(token); return token

    def solve(deck, *, initial_state, raise_on_max_iterations):
        assert raise_on_max_iterations and initial_state is calls.restarts[-1] and not initial_state[0]
        initial_state[0] = True; calls.decks.append(deck)
        x = (opt.pack_boundary(deck, 4)[selected]-x0[selected])/.0003
        ftol = float(seed.ftol_array[-1]); bad = mode == 'force' and np.any(x)
        result = NS(converged=True, fsqr=ftol*(2 if bad else .5), fsqz=ftol/3, fsql=ftol/4)
        return NS(inp=deck, result=result, wout=NS(x=float(x[0]), aspect=6., Aminor_p=.2, volavgB=1., lasym=lasym, refined=deck.mpol>seed.mpol))

    def write(filename, wout):
        calls.writes.append((str(filename), wout)); Path(filename).touch()
    monkeypatch.setitem(env, 'restart_state', restart)
    monkeypatch.setattr(opt, 'solve_equilibrium', solve)
    monkeypatch.setattr(example['vj'], 'write_wout', write)

    class Boozer:
        def __init__(self, **kw):
            self.band = kw['mboz']; self.asym = lasym
        def read_wout_data(self, wout):
            self.wout = wout; self.nfp = seed.nfp
            self.s_in = np.array([.25]) if mode == 'collapse' else np.array([.25, .75])
            if wout.refined and mode == 'surface_roundoff':
                self.s_in = np.nextafter(self.s_in, np.inf)
            if wout.refined and mode == 'surface_shift' and wout.x:
                self.s_in = self.s_in+1e-4
            self.iota = np.full(len(self.s_in), .6)
        def run(self, *, jit):
            assert jit is True
            n = len(self.compute_surfs); marker = 1+.01*self.wout.x
            self.s_b = self.s_in[self.compute_surfs]
            self.xm_b, self.xn_b = np.array([0, 1]), np.array([0, seed.nfp])
            self.bmnc_b = np.tile([[marker], [.2]], (1, n))
            self.rmnc_b = np.tile([[1.], [.1]], (1, n))
            self.zmns_b = self.numns_b = np.zeros((2, n))
            self.Boozer_G, self.Boozer_I = np.ones(n), np.zeros(n)
            self.rmns_b = self.zmnc_b = self.numnc_b = self.bmns_b = np.zeros((2, n))

    class Family:
        def __init__(self, reference, charts, length):
            self.reference = reference; self.charts = charts; self.group = charts[0]['group']
            self.initial = {'families': [dict(mean_action_m=1.1)]}
        def __call__(self, spectrum):
            np.testing.assert_array_equal(spectrum['s_b'], self.reference['s_b'])
            changed = abs(spectrum['bmnc_b'][0, 0]-1) > 1e-9
            if changed and mode == self.group+'_family':
                raise ValueError('Lost '+self.group+' family')
            return {'families': [dict(rows=[{'J': v} for v in [1., 1.1, 1.2]], alpha_domain=[0., 2.])]}

    def census(**kw):
        assert kw['check_roots'] and kw['match_critical_census']
        assert kw['points_per_period'] == 256 and kw['max_periods'] == 64
        np.testing.assert_allclose(kw['alphas'], (np.arange(8)+.5)*2*np.pi/8)
        token = {'census_id': len(calls.census)}
        calls.census.append(dict(arguments=kw, result=token))
        if mode == 'census' and kw['reference'] is not None:
            raise ValueError('Independent critical census changed')
        return token
    monkeypatch.setitem(env, 'physical_pwo_action_from_boozer', census)

    def neo(booz, *, config, **kw):
        assert kw['jax_surface_scan'] is False
        changed = abs(booz['bmnc_b'][0, 0]-1) > 1e-9; out = []
        for i, s in enumerate(booz['s_b']):
            value = [1e-4, 1e-6][i]*(.8 if changed else 1.)
            if changed and mode == 'tradeoff' and i == 1:
                value = 1.1e-6
            if changed and mode == 'holdout_ripple' and config.npart > 80:
                value *= 2
            if changed and mode == 'resolution_gain' and config.npart > 80:
                value = [1e-4, 1e-6][i]*.95
            d = dict(approximation_used=changed and mode == 'neo_approx',
                     rational_correction_valid=not(changed and mode == 'neo_rational'), final_n=1000,
                     rational_correction_applied=False)
            out.append(NS(epsilon_effective=np.nan if changed and mode == 'neo_nan' else value,
                diagnostics=d, s=s+.01 if changed and mode == 'neo_surface' else s,
                iota=np.nan if changed and mode == 'neo_iota' else .6,
                b_ref=np.nan if changed and mode == 'neo_reference' else 1.2, r_ref=1.))
        return out[:1] if changed and mode == 'neo_count' else out
    monkeypatch.setitem(sys.modules, 'booz_xform_jax', NS(Booz_xform=Boozer))
    monkeypatch.setitem(sys.modules, 'neo_jax', NS(NeoConfig=lambda **kw: NS(**kw), run_neo=neo))
    monkeypatch.setitem(env, 'PwOActionFamilies', Family)

    def local(fun, x, **kw):
        calls.local.append(kw); assert kw['method'] == 'Nelder-Mead'
        assert fun(np.zeros(len(x))) == 1.  # Seed is cached, never solved again by the optimizer.
        if mode == 'budget':
            for point in kw['options']['initial_simplex']:
                fun(point)
        else:
            point = np.full(len(x), -.1); point[0] = .2; fun(point); fun(point)
        kw['callback'](point)
        return NS(success=True, message='Simplex tolerances satisfied', nit=1, nfev=2)
    def global_(fun, bounds, **kw):
        assert fun(np.zeros(len(bounds))) == 1.
        for j in range(100):
            calls.global_ += 1; point = np.zeros(len(bounds)); point[0] = .001*(j+1); fun(point)
    monkeypatch.setitem(env, 'minimize', local)
    monkeypatch.setattr(scipy.optimize, 'differential_evolution', global_)
    def charts(group):
        return [dict(reference={'nfp': seed.nfp, 's_b': [s]},
                     charts=[{'group': group, 'level_T': 1.1}], length=1.) for s in [.25, .75]]
    policy = dict(optimizer='derivative_free', input=str(input_path), restart='external.nc', parameters=parameters, scales=.0003,
        surfaces=[.25, .75], training=charts('training'), validation=charts('validation'),
        global_steps=1 if mode == 'budget' else 0, global_calls=2, local_calls=len(parameters)+1)
    return run, policy, seed, calls, selected, x0


@pytest.mark.parametrize('lasym', [False, True])
def test_pwo_confinement_preserves_external_input_and_independent_restarts(tmp_path, monkeypatch, lasym):
    run, policy, seed, calls, selected, x0 = _pwo_confinement_case(tmp_path, monkeypatch, lasym=lasym)
    for group in ['training', 'validation']:
        path = tmp_path/(group+'.npz'); np.savez(path, **policy[group][0]['reference'])
        policy[group][0]['reference'] = str(path)
    result = run(policy, output_name='direct')
    assert result['accepted'] and not result['pwO_qualified']
    from vmex import optimize as opt
    for deck in calls.decks:
        for name in ['phiedge', 'curtor', 'ncurr', 'pres_scale', 'pmass_type', 'pcurr_type', 'piota_type']:
            assert getattr(deck, name) == getattr(seed, name)
        for name in ['am', 'ac', 'ai', 'aphi', 'am_aux_s', 'am_aux_f', 'ac_aux_s', 'ac_aux_f', 'ai_aux_s', 'ai_aux_f']:
            np.testing.assert_array_equal(getattr(deck, name), getattr(seed, name))
        for name in ['raxis_c', 'zaxis_s', 'raxis_s', 'zaxis_c']:
            np.testing.assert_array_equal(getattr(deck, name)[:seed.ntor+1], getattr(seed, name))
        values = opt.pack_boundary(deck, 4); mask = np.ones(len(x0), bool); mask[selected] = False
        np.testing.assert_array_equal(values[mask], x0[mask])
        assert deck.rbc[deck.ntor, 0] == seed.rbc[seed.ntor, 0]
    assert len(calls.restarts) == len(calls.decks) and len({id(t) for t in calls.restarts}) == len(calls.restarts)
    assert len(result['cases']) == 1  # Seed and repeated optimizer point are cached.
    assert len(calls.decks) == 4  # Coarse/refined baseline and coarse/refined best only.
    assert all(w is calls.restart_sources[0] for w in calls.restart_sources[:3])
    assert calls.restart_sources[-1] is next(w for name, w in calls.writes if name.endswith('_seed_check.nc'))
    assert np.all(np.asarray(result['best_training']['x']) != 0)  # All selected m3/m4/LASYM modes moved.
    assert (tmp_path/'input.direct').exists() and (tmp_path/'wout_direct.nc').exists()
    assert len(calls.census) == 4
    for i in range(2):
        assert calls.census[i]['arguments']['reference'] is None
        assert calls.census[i+2]['arguments']['reference'] is calls.census[i]['result']
    assert (tmp_path/'input.direct_seed_check').exists() and (tmp_path/'input.direct_candidate_check').exists()
    assert calls.decks[-1].mpol > seed.mpol  # Accepted checkpoint follows independent refined solve.
    assert calls.local[0]['options']['initial_simplex'].shape == (len(selected)+1, len(selected))


@pytest.mark.parametrize('mode', ['force', 'neo_nan', 'neo_iota', 'neo_surface', 'neo_count', 'neo_approx',
                                  'neo_rational', 'neo_reference', 'training_family', 'validation_family', 'holdout_ripple', 'resolution_gain', 'census'])
def test_pwo_confinement_rejects_invalid_training_or_independent_holdout(tmp_path, monkeypatch, mode):
    run, policy, _, calls, _, _ = _pwo_confinement_case(tmp_path, monkeypatch, mode)
    result = run(policy, output_name='rejected')
    assert not result['accepted']
    assert not (tmp_path/'input.rejected').exists() and not (tmp_path/'wout_rejected.nc').exists()
    holdout = mode in ['validation_family', 'holdout_ripple', 'resolution_gain', 'census']
    assert (tmp_path/'input.rejected_candidate').exists() == holdout
    assert any('seed' in name for name, _ in calls.writes)
    if mode == 'census':
        assert 'critical census changed' in result['validation_rejection']
        assert calls.census[-1]['arguments']['reference'] is calls.census[0]['result']
        assert (tmp_path/'input.rejected_candidate_check').exists()
    if not holdout:
        assert not any(row['valid'] for row in result['cases'])


@pytest.mark.parametrize('mode,message', [('collapse', 'collapsed'), ('ftol', 'force tolerance')])
def test_pwo_confinement_rejects_surface_collapse_before_optimization(tmp_path, monkeypatch, mode, message):
    from dataclasses import replace
    run, policy, seed, calls, _, _ = _pwo_confinement_case(tmp_path, monkeypatch, mode)
    if mode == 'ftol':
        monkeypatch.setattr(type(seed), 'from_file', lambda _: replace(seed, ftol_array=np.array([0.])))
    with pytest.raises(ValueError, match=message):
        run(policy, output_name='collapse')
    assert not calls.local and not (tmp_path/'input.collapse').exists()


def test_pwo_confinement_limits_actual_global_and_local_work(tmp_path, monkeypatch):
    run, policy, _, calls, _, _ = _pwo_confinement_case(tmp_path, monkeypatch, 'budget')
    result = run(policy, output_name='budget')
    assert calls.global_ == policy['global_calls']+1  # Extra request stops before another solve.
    assert len(result['cases']) <= policy['global_calls']+policy['local_calls']
    assert calls.local[0]['options']['maxfev'] == policy['local_calls']
    policy['local_calls'] = len(policy['parameters'])
    with pytest.raises(ValueError, match='D\\+1'):
        run(policy, output_name='bad_budget')


def test_pwo_confinement_uses_common_normalization_for_surface_tradeoff(tmp_path, monkeypatch):
    run, policy, _, _, _, _ = _pwo_confinement_case(tmp_path, monkeypatch, 'tradeoff')
    result = run(policy, output_name='tradeoff')
    assert result['accepted']
    assert result['best_training']['objective'] == pytest.approx((.8e-4+1.1e-6)/(1e-4+1e-6))


@pytest.mark.parametrize('mode,accepted', [('surface_roundoff', True), ('surface_shift', False)])
def test_pwo_confinement_distinguishes_surface_roundoff_from_physical_shift(tmp_path, monkeypatch, mode, accepted):
    run, policy, _, _, _, _ = _pwo_confinement_case(tmp_path, monkeypatch, mode)
    result = run(policy, output_name='surface')
    assert result['accepted'] is accepted
    if accepted:
        assert result['checked_baseline']['actual_s'] == result['baseline']['actual_s']
        assert result['validation']['actual_s'] == result['baseline']['actual_s']
    else:
        assert 'surfaces changed' in result['validation_rejection']
        assert not (tmp_path/'input.surface').exists()


@pytest.mark.parametrize('updates', [dict(parameters=['RBC(0,0)']), dict(scales=0.),
    dict(optimizer='unknown'), dict(max_mode=0),
    dict(optimizer_ftol=0.), dict(optimizer_ftol=np.nan),
    dict(constraints={}), dict(constraints={'aspect': [6., 5.], 'iota_min': .41}),
    dict(constraints={'aspect': [5., 6.], 'iota_min': np.nan}),
    dict(optimizer='adjoint', constraints={'aspect': [5., 6.], 'iota_min': .41}),
    dict(surfaces=[.75, .25]), dict(neo={'rational_surface_policy': 'approximate'}),
    dict(training=[]), dict(training=[{'reference': {'nfp': 99, 's_b': [.25]}}]),
    dict(family_rtol=-1), dict(check_resolution=[2, 1, 1, 2, 2]), dict(free=True)])
def test_pwo_confinement_rejects_invalid_policy_before_solves(tmp_path, monkeypatch, updates):
    from dataclasses import replace
    run, policy, seed, calls, _, _ = _pwo_confinement_case(tmp_path, monkeypatch)
    if updates.get('free'):
        monkeypatch.setattr(type(seed), 'from_file', lambda _: replace(seed, lfreeb=True, mgrid_file='external_grid.nc'))
    policy.update({k: v for k, v in updates.items() if k != 'free'})
    with pytest.raises(ValueError):
        run(policy, output_name='invalid')
    assert not calls.decks


@pytest.mark.parametrize('failure', [None, 'axis', 'fine_axis', 'aspect', 'jacobian', 'sine_jacobian', 'nonfinite'])
def test_pwo_confinement_absolute_constraints_check_full_profile_and_refinement(tmp_path, monkeypatch, failure):
    run, policy, _, _, _, _ = _pwo_confinement_case(tmp_path, monkeypatch, lasym=failure == 'sine_jacobian')
    policy['constraints'] = {'aspect': [5.9, 6.1], 'iota_min': .41}
    opt = run.__globals__['opt']; original = opt.solve_equilibrium
    def solve(*args, **kwargs):
        eq = original(*args, **kwargs); w = eq.wout; changed = bool(w.x)
        w.iotaf = np.array([.6, .6]); w.signgs = 1; w.xm_nyq = np.array([0, 1]); w.xn_nyq = np.zeros(2)
        w.gmnc = np.tile([1., 0.], (2, 1)); w.gmns = np.zeros((2, 2))
        if changed:
            w.Aminor_p *= 1.1  # Absolute constraints replace the seed-relative size bound.
            if failure == 'axis' or (failure == 'fine_axis' and w.refined):
                w.iotaf[0] = .40  # Selected Boozer surfaces still have iota=.6.
            if failure == 'aspect':
                w.aspect = 6.2
            if failure == 'jacobian':
                w.gmnc *= -1
            if failure == 'sine_jacobian':
                w.gmns[:, 1] = 2.
            if failure == 'nonfinite':
                w.iotaf[0] = np.nan
        return eq
    monkeypatch.setattr(opt, 'solve_equilibrium', solve)
    result = run(policy, output_name='absolute')
    assert result['accepted'] == (failure is None)
    if failure is None:
        assert result['validation']['iota_min'] == .6
        assert result['validation']['metrics'][1] == pytest.approx(.22)


@pytest.mark.parametrize('bad', [False, True])
def test_pwo_confinement_policy_dispatch(tmp_path, monkeypatch, bad):
    run, _, _, _, _, _ = _pwo_confinement_case(tmp_path, monkeypatch)
    env = run.__globals__; path = tmp_path/'policy.json'; path.write_text('{}')
    monkeypatch.setitem(env, 'METHOD', 'confinement'); monkeypatch.setenv('VMEX_PWO_POLICY', str(path))
    monkeypatch.setitem(env, 'run_confinement', lambda p, **kw: {'accepted': True})
    if bad:
        with pytest.raises(ValueError, match='preserves supplied profiles'):
            env['run'](beta_targets=(0., .01))
    else:
        assert env['run']() == {'accepted': True}


@pytest.mark.parametrize('disagreement', [None, 'ripple', 'actions', 'candidate_actions', 'candidate_branch', 'no_charts'])
def test_pwo_confinement_scalar_loss_has_live_physics_and_penalty_gradients(tmp_path, monkeypatch, disagreement):
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace as NS
    from vmex.core import neoclassical, statephysics
    run, _, _, _, _, _ = _pwo_confinement_case(tmp_path, monkeypatch)
    env = run.__globals__

    def spectrum(state, runtime, **kwargs):
        assert kwargs['geometry'] is True and kwargs['oversample'] == 1
        return dict(bmnc_b=jnp.array([[1+.01*state[0], .2*(1+.2*state[1])]]),
                    bmns_b=jnp.zeros((1, 2)), xm_b=np.array([0, 1]), xn_b=np.array([0, 2]),
                    iota_b=jnp.array([.6+.01*state[2]]), G_b=jnp.array([1.]), I_b=jnp.array([0.]),
                    s_b=jnp.array([.25]), nfp=2)

    class Family:
        reference = {'s_b': [.25]}
        initial = {'families': [dict(mean_action_m=1.1)]}
        def __call__(self, sp):
            x = float((sp['bmnc_b'][0, 0]-1)/.01)
            actions = np.array([1., 1.1, 1.2])+x*np.array([-.02, .01, .05])
            return {'families': [dict(rows=[{'J': v} for v in actions], alpha_domain=[0., 2*(1-.8*x)])]}
        def jax_components(self, sp, **kwargs):
            x = (sp['bmnc_b'][0, 0]-1)/.01
            actions = jnp.array([1., 1.1, 1.2])+x*jnp.array([-.02, .01, .05])
            return {'families': [dict(actions=actions+(1 if disagreement == 'actions' else .01*x if disagreement == 'candidate_actions' else 0),
                action_range_m=jnp.max(actions)-jnp.min(actions),
                alpha_domain=jnp.array([0., 2*(1-.8*x)+(.01*x if disagreement == 'candidate_branch' else 0)]))]}

    monkeypatch.setitem(env, 'boozer_spectrum_state', spectrum)
    def ripple(sp, **kwargs):
        return sp['s_b'], sp['bmnc_b'][:, 0]**2+.3*sp['iota_b']+.02*sp['G_b']
    monkeypatch.setattr(neoclassical, 'epsilon_effective_from_spectrum', ripple)
    monkeypatch.setattr(statephysics, '_aspect_scalars',
        lambda st, rt: (.2*(1+.05*st[2]), 0., 6*(1+.04*st[0]), 1.))
    monkeypatch.setattr(statephysics, '_field_chain',
        lambda st, rt: (NS(wb=(1+.08*st[1])**2/(2*(2*jnp.pi)**2)),))
    baseline = dict(eps32=[1.2+(1 if disagreement == 'ripple' else 0)],
        metrics=[6., .2, 1., 1., .2, .6], family_ranges=[.2/1.1], family_spans=[2.])
    if disagreement == 'no_charts':
        baseline.update(family_ranges=[], family_spans=[])
    arguments = (NS(state=jnp.zeros(3), runtime=None), [.25], 8,
                 [] if disagreement == 'no_charts' else [Family()], baseline, np.array([1.]), None, 0.)
    if disagreement in ['ripple', 'actions']:
        with pytest.raises(ValueError, match='disagrees' if disagreement == 'ripple' else 'quadrature'):
            env['confinement_loss'](*arguments)
        return
    loss = env['confinement_loss'](*arguments)
    if disagreement and disagreement != 'no_charts':
        with pytest.raises(ValueError, match='branch or quadrature'):
            loss.validate(jnp.array([.8, .9, .7]), None)
        return
    value_and_gradient = jax.jit(jax.value_and_grad(lambda st: loss(st, None)))
    np.testing.assert_allclose(value_and_gradient(jnp.zeros(3))[0], 1., atol=1e-13)
    loss.validate(jnp.zeros(3), None)
    point, direction = jnp.array([.8, .9, .7]), jnp.array([.2, -.3, .4])
    value, gradient = value_and_gradient(point)
    assert value > float(ripple(spectrum(point, None, geometry=True, oversample=1))[1][0]/1.2)
    assert np.all(np.isfinite(gradient)) and np.all(np.abs(gradient) > 1e-5)
    for step in [1e-4, 5e-5]:
        fd = (loss(point+step*direction, None)-loss(point-step*direction, None))/(2*step)
        np.testing.assert_allclose(gradient@direction, fd, rtol=2e-6, atol=1e-8)


def _pwo_confinement_adjoint_case(tmp_path, monkeypatch, mode='valid', all_modes=False):
    import jax
    import jax.numpy as jnp
    from dataclasses import replace
    from types import SimpleNamespace as NS
    from vmex.core import implicit as im
    run, policy, seed, calls, _, _ = _pwo_confinement_case(tmp_path, monkeypatch, mode)
    env, opt = run.__globals__, run.__globals__['opt']
    policy.pop('optimizer')  # The default is the scalar adjoint.
    policy.update(global_steps=1, scales=.0003 if all_modes else [.0002, .0005], local_calls=2)
    if all_modes:
        policy.pop('parameters'); policy['max_mode'] = 8
    def loss_factory(*args):
        assert args[-2].reverse_mode is True
        def loss(state, runtime):
            return jnp.sum(state)
        loss.validate = lambda state, runtime: None
        return loss
    monkeypatch.setitem(env, 'confinement_loss', loss_factory)

    def factory(inp, loss, **kwargs):
        names = opt.boundary_dof_names(inp, kwargs['max_mode'])
        origin = opt.pack_boundary(inp, kwargs['max_mode'])
        calls.problem = NS(inp=inp, kwargs=kwargs, names=names, origins=[], gradients=[])
        def subproblem(*, active):
            ids = np.array([names.index(n) for n in active]); x0 = origin[ids].copy()
            scales = np.broadcast_to(policy['scales'], x0.shape)
            coefficients = jnp.linspace(.2, .4, len(ids))
            def fun(x):
                return 1+jnp.vdot((x-jnp.asarray(x0))/jnp.asarray(scales), coefficients)
            pair = jax.jit(jax.value_and_grad(fun))
            def value_and_grad(x):
                value, gradient = pair(x)
                calls.problem.origins.append(np.array(x)); calls.problem.gradients.append(np.array(gradient))
                factor = np.nan if mode == 'nonfinite' or (mode == 'trial_nonfinite' and np.any(x != x0)) else 2 if mode == 'gradient' else 1
                return float(value), np.asarray(gradient)*factor
            def equilibrium(x):
                full = origin.copy(); full[ids] = x
                deck = opt.unpack_boundary(inp, full, kwargs['max_mode'])
                eq = opt.solve_equilibrium(deck,
                    initial_state=env['restart_state'](env['vj'].read_wout('external.nc'), deck),
                    raise_on_max_iterations=True)
                eq.state, eq.runtime = np.asarray(x), None
                return eq
            cfg = NS(device=None, inp=inp, max_mode=kwargs['max_mode'], ids=ids, scales=scales, coefficients=coefficients)
            return NS(x0=x0, value_and_grad=value_and_grad, equilibrium_from_x=equilibrium, metadata={'config': cfg})
        return NS(subproblem=subproblem)
    monkeypatch.setattr(opt.VmecProblem, 'from_loss', factory)
    def frozen(params, cfg, loss, delta, *, h, newton_steps, newton_rtol):
        assert newton_steps == 20 and newton_rtol == 1e-13
        deck = replace(cfg.inp, **{key: np.asarray(getattr(delta, key)) for key in ('rbc', 'zbs', 'rbs', 'zbc')})
        dx = opt.pack_boundary(deck, cfg.max_mode)[cfg.ids]
        return float((dx/cfg.scales)@cfg.coefficients), {'newton_res': [1e-8 if mode == 'residual' else 1e-14]*2}
    monkeypatch.setattr(im, 'frozen_path_directional_fd', frozen)
    def local(pair, zero, **kwargs):
        calls.local.append(kwargs)
        assert kwargs['jac'] is True and kwargs['method'] == 'L-BFGS-B'
        point = np.full(len(zero), -.1); point[0] = .2
        value, gradient = pair(point)
        np.testing.assert_allclose(gradient, np.linspace(.2, .4, len(zero)), rtol=1e-13)
        assert np.isfinite(value)
        kwargs['callback'](point)
        return NS(success=True, message='Relative function decrease satisfied', nit=1, nfev=1)
    monkeypatch.setitem(env, 'minimize', local)
    return run, policy, seed, calls


@pytest.mark.parametrize('all_modes', [False, True])
def test_pwo_confinement_adjoint_scales_selected_coefficients_and_opens_mode8(tmp_path, monkeypatch, all_modes):
    run, policy, seed, calls = _pwo_confinement_adjoint_case(tmp_path, monkeypatch, all_modes=all_modes)
    result = run(policy, output_name='adjoint')
    assert result['accepted'] and result['optimizer'] == 'adjoint' and calls.global_ == 0
    assert len(calls.local) == 1 and result['gradient_check']['finite_differences']
    np.testing.assert_allclose(result['gradient_check']['finite_differences'], result['gradient_check']['adjoint'], rtol=1e-10)
    assert calls.problem.kwargs['restart_from'].inp.mpol == calls.problem.inp.mpol
    assert calls.problem.kwargs['vary_major_radius'] is False
    assert calls.problem.inp.rbc[calls.problem.inp.ntor, 0] == seed.rbc[seed.ntor, 0]
    if all_modes:
        assert calls.problem.inp.mpol == 9 and calls.problem.inp.ntor == 8
        assert len(result['parameters']) == len(calls.problem.names) == 288
        assert 'RBC(8,8)' in result['parameters'] and 'RBC(0,0)' not in result['parameters']
    else:
        assert result['parameters'] == policy['parameters']
        displacement = calls.problem.origins[-1]-calls.problem.origins[0]
        np.testing.assert_allclose(displacement, np.asarray(policy['scales'])*[.2, -.1], atol=1e-17)


@pytest.mark.parametrize('mode', ['training_family', 'validation_family', 'census', 'neo_rational', 'force', 'holdout_ripple'])
def test_transport_only_search_retains_physics_checks_without_family_requirements(tmp_path, monkeypatch, mode):
    run, policy, _, calls = _pwo_confinement_adjoint_case(tmp_path, monkeypatch, mode)
    policy.pop('training'); policy.pop('validation')
    result = run(policy, output_name='transport')
    assert result['accepted'] == (mode in ['training_family', 'validation_family', 'census'])
    assert not result['family_guards'] and not result['pwO_qualified'] and not calls.census
    policy['training'] = []
    with pytest.raises(ValueError, match='Supply both'):
        run(policy)


@pytest.mark.parametrize('mode', ['valid', 'neo_rational', 'force', 'holdout_ripple'])
def test_transport_proposals_use_independent_acceptance_without_gradients(tmp_path, monkeypatch, mode):
    run, policy, _, calls, _, _ = _pwo_confinement_case(tmp_path, monkeypatch, mode)
    policy.pop('training'); policy.pop('validation')
    policy.update(optimizer='proposals', proposals=[[.2, -.1], [.1, 0.]])
    result = run(policy, output_name='proposals')
    assert result['accepted'] == (mode == 'valid')
    assert not calls.local and not calls.global_ and 'gradient_check' not in result
    assert len(result['cases']) == 2


@pytest.mark.parametrize('proposals', [[], [[0.]], [[0., np.nan]], [[1.1, 0.]], [[0., 0.]]*4])
def test_transport_proposals_reject_invalid_steps_before_solving(tmp_path, monkeypatch, proposals):
    run, policy, _, calls, _, _ = _pwo_confinement_case(tmp_path, monkeypatch)
    policy.update(optimizer='proposals', proposals=proposals, local_calls=3)
    with pytest.raises(ValueError, match='proposals must'):
        run(policy)
    assert not calls.decks


@pytest.mark.parametrize('mode', ['gradient', 'residual', 'nonfinite', 'trial_nonfinite'])
def test_pwo_confinement_adjoint_gradient_failure_stops_before_promotion(tmp_path, monkeypatch, mode):
    run, policy, _, calls = _pwo_confinement_adjoint_case(tmp_path, monkeypatch, mode)
    with pytest.raises(RuntimeError, match='directional check' if mode in ('gradient', 'residual') else 'Nonfinite'):
        run(policy, output_name='bad_gradient')
    assert len(calls.local) == (1 if mode == 'trial_nonfinite' else 0) and calls.global_ == 0
    assert not (tmp_path/'input.bad_gradient').exists()
    if mode != 'nonfinite':
        assert json.loads((tmp_path/'bad_gradient_summary.json').read_text())['gradient_check']


def test_pwo_confinement_adjoint_budget_stops_before_another_gradient(tmp_path, monkeypatch):
    run, policy, _, calls = _pwo_confinement_adjoint_case(tmp_path, monkeypatch)
    policy['local_calls'] = 1
    def search(pair, zero, **kwargs):
        pair(zero+.1)
        pair(zero+.2)
        pytest.fail('Candidate budget did not stop the search')
    monkeypatch.setitem(run.__globals__, 'minimize', search)
    result = run(policy, output_name='bounded')
    assert len(result['cases']) == 1 and len(calls.problem.gradients) == 2
    assert not result['termination']['success']
    assert result['termination']['message'] == 'Evaluation budget reached'


@pytest.mark.parametrize('adjoint', [False, True])
def test_pwo_confinement_optimizer_tolerance_and_iteration_history(tmp_path, monkeypatch, adjoint):
    case = _pwo_confinement_adjoint_case if adjoint else _pwo_confinement_case
    run, policy, _, calls, *_ = case(tmp_path, monkeypatch)
    policy['optimizer_ftol'] = .002
    result = run(policy, output_name='history')
    assert calls.local[0]['options']['ftol' if adjoint else 'fatol'] == .002
    assert result['optimizer_ftol'] == .002 and result['termination']['nit'] == 1
    assert result['termination']['success'] and result['accepted']
    assert len(result['iterations']) == 1
    assert result['iterations'][0]['best_objective'] == result['best_training']['objective']


@pytest.mark.parametrize('mode', ['force', 'training_family', 'neo_rational', 'validation_family', 'census'])
def test_pwo_confinement_adjoint_never_promotes_failed_independent_guards(tmp_path, monkeypatch, mode):
    run, policy, _, calls = _pwo_confinement_adjoint_case(tmp_path, monkeypatch, mode)
    result = run(policy, output_name='adjoint_rejected')
    assert not result['accepted'] and len(calls.local) == 1
    assert not (tmp_path/'input.adjoint_rejected').exists()
    assert not (tmp_path/'wout_adjoint_rejected.nc').exists()
    if mode in ['force', 'training_family', 'neo_rational']:
        assert not any(row['valid'] for row in result['cases'])
    else:
        assert result['validation_rejection']


def test_pwo_mode8_override_prepares_288_boundary_variables(monkeypatch):
    import runpy
    from vmex import optimize as opt

    monkeypatch.setenv("VMEX_EXAMPLES_CI", "0")
    monkeypatch.setenv("VMEX_PWO_METHOD", "scalar")
    monkeypatch.setenv("VMEX_PWO_SEED", "qi")
    monkeypatch.delenv("VMEX_PWO_QS_INPUT", raising=False)
    monkeypatch.setenv("VMEX_PWO_MAX_MODE", "8")
    monkeypatch.setenv("VMEX_PWO_MAXITER", "20")
    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    assert example["MAX_MODES"] == [8] and example["MAXITER"] == [20]
    decks = []

    def capture(deck, **kwargs):
        decks.append(deck)
        raise RuntimeError("captured prepared input")

    monkeypatch.setattr(opt, "solve_equilibrium", capture)
    with pytest.raises(RuntimeError, match="captured prepared input"):
        example["run"]()
    deck = decks[0]
    assert deck.mpol >= 9 and deck.ntor >= 8
    assert deck.ntheta >= 32 and deck.nzeta >= 32
    assert len(opt.pack_boundary(deck, 8, vary_major_radius=False)) == 288
    assert example["FINAL_MPOL"] > example["MPOL"]
    assert example["PWO_OPTIONS"]["mboz"] >= 16 and example["PWO_OPTIONS"]["nboz"] >= 16
    assert example["GRID"] >= 48
    assert example["VALIDATION_OPTIONS"]["mboz"] >= 2 * example["PWO_OPTIONS"]["mboz"]
    assert example["VALIDATION_GRID"] >= 2 * example["GRID"] + 1
    monkeypatch.setenv("VMEX_EXAMPLES_CI", "1")
    smoke = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    assert smoke["MAX_MODES"] == [1] and smoke["MAXITER"] == [1]
    assert smoke["PWO_OPTIONS"]["mboz"] == 4 and smoke["GRID"] == 12
    assert smoke["VALIDATION_GRID"] == 17


@pytest.mark.parametrize("name", ["VMEX_PWO_MAX_MODE", "VMEX_PWO_MAXITER"])
def test_pwo_budget_overrides_reject_nonpositive_values(monkeypatch, name):
    import runpy

    monkeypatch.setenv("VMEX_EXAMPLES_CI", "0")
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match=name + " must be positive"):
        runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))


def test_pwo_scalar_checkpoint_keeps_best_accepted_physical_vector(tmp_path, monkeypatch):
    import runpy
    from types import SimpleNamespace

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    callback = example["scalar_checkpoint_callback"]
    saved, monitored = [], []
    monkeypatch.setitem(callback.__globals__, "save_seed_checkpoint",
                        lambda deck, eq, output, label: saved.append((deck, eq, label)))
    stage = SimpleNamespace(input_from_x=lambda x: x.copy(), equilibrium_from_x=lambda x: x.copy())
    output = str(tmp_path / "fit")
    record = callback(stage, np.array([1., 2.]), np.array([.1, .2]), monitored.append, output)
    record(SimpleNamespace(x=np.array([.5, -.5]), fun=3.))
    record(SimpleNamespace(x=np.array([.8, -.8]), fun=4.))
    record(SimpleNamespace(x=np.array([.2, -.2]), fun=np.nan))
    record(SimpleNamespace(x=np.array([.1, -.1]), fun=2.))
    assert len(saved) == 2 and len(monitored) == 4
    np.testing.assert_allclose(saved[-1][0], [1.01, 1.98])
    np.testing.assert_array_equal(saved[-1][0], saved[-1][1])
    assert saved[-1][2] == "best"
    metadata = json.loads(Path(output + "_best.json").read_text())
    assert metadata["cost"] == 2.
    np.testing.assert_allclose(metadata["x"], [1.01, 1.98])


def test_pwo_explicit_target_keeps_tesla_shape_and_allows_new_mirror(monkeypatch):
    import runpy
    import vmex.core.pwo as pwo

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    parameters = dict(bmin=.85, bmax=1.15, target_iota=-.42, t1=.3, t2=-2.)
    target = example["make_target"]([.5], family="canonical", bmin=.1, bmax=9., b_ref=1.,
                                    options={}, grid=24, iota_sign=-1., target_parameters=parameters)
    spectrum = dict(bmnc_b=np.array([[1., .15]]), bmns_b=np.zeros((1, 2)),
                    xm_b=np.array([0, 1]), xn_b=np.array([0, 0]), iota_b=np.array([-.43]), nfp=2)
    monkeypatch.setattr(pwo, "boozer_spectrum_state", lambda *a, **k: dict(spectrum))
    first = target.compute_state(None, None)
    spectrum["iota_b"] = np.array([-.41])
    second = target.compute_state(None, None)
    np.testing.assert_array_equal(first["B_target"], second["B_target"])
    assert first["iota_b"][0] != second["iota_b"][0]
    assert target.options["bmin"] == .85 and target.options["bmax"] == 1.15
    assert first["mirror"][0] == pytest.approx(.15)
    out = {**second, "residuals1d": np.zeros(1)}
    fixed = example["target_physics_rows"](out, [1.], [.33], parameters)
    legacy = example["target_physics_rows"](out, [1.], [.33])
    np.testing.assert_allclose(fixed, 0., atol=1e-12)
    assert np.linalg.norm(legacy) > 1.
    out["iota_b"] = np.array([-.39])
    assert example["target_physics_rows"](out, [1.], [.33], parameters)[-1] > 0
    out["mirror"] = np.array([.01])
    assert example["target_physics_rows"](out, [1.], [.33], parameters)[-2] > 0
    with pytest.raises(ValueError, match="signed iota"):
        example["make_target"]([.5], family="canonical", bmin=1., bmax=2., b_ref=1.,
                               options={}, grid=24, iota_sign=1., target_parameters=parameters)
    singular = example["make_target"]([.5], family="canonical", bmin=1., bmax=2., b_ref=1.,
                                      options={}, grid=24, iota_sign=-1.,
                                      target_parameters={**parameters, "t1": 1., "t2": 1.})
    with pytest.raises(ValueError, match="canonical target needs"):
        singular.compute_state(None, None)


@pytest.mark.parametrize("change", [{"extra": 1}, {"bmin": 0}, {"bmax": .8},
                                    {"target_iota": 0}, {"t1": np.nan}, {"t2": [2]}])
def test_pwo_explicit_target_rejects_invalid_configuration(change):
    import runpy

    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    parameters = dict(bmin=.85, bmax=1.15, target_iota=-.42, t1=.3, t2=-2.)
    with pytest.raises(ValueError, match="Explicit target"):
        example["validate_target_parameters"]({**parameters, **change})


def test_pwo_explicit_target_skips_seed_mirror_calibration(tmp_path, monkeypatch):
    import runpy
    from types import SimpleNamespace as NS

    parameters = dict(bmin=.85, bmax=1.15, target_iota=-.42, t1=.3, t2=-2.)
    path = tmp_path / "target.json"; path.write_text(json.dumps(parameters))
    monkeypatch.setenv("VMEX_PWO_TARGET", str(path))
    example = runpy.run_path(str(EXAMPLES / "optimization" / "pwO_optimization.py"))
    run = example["run"]; env = run.__globals__
    env.update(SEED="qi", QS_INPUT=None, METHOD="scalar", FAMILY="canonical")
    monkeypatch.setitem(env, "canonical_target_bounds", lambda *a: pytest.fail("seed mirror calibration ran"))
    eq = NS(state=None, runtime=None, wout=NS(aspect=6., Aminor_p=.1, volavgB=1.))
    monkeypatch.setattr(example["opt"], "solve_equilibrium", lambda *a, **k: eq)
    monkeypatch.setattr(example["opt"], "mean_iota", lambda *a: -.42)
    monkeypatch.setattr(example["CanonicalPwOResidual"], "compute_state", lambda *a: dict(
        B=np.ones((1, 24, 24)), B_target=np.ones((1, 24, 24)), mirror=np.array([.33])))

    def captured_factory(*args, **kwargs):
        raise RuntimeError("reached scalar factory")

    monkeypatch.setattr(example["opt"].VmecProblem, "from_loss", captured_factory)
    with pytest.raises(RuntimeError, match="reached scalar factory"):
        run()
    env["FAMILY"] = "mapped"
    with pytest.raises(ValueError, match="requires canonical"):
        run()
    env.update(FAMILY="canonical", METHOD="physical")
    with pytest.raises(ValueError, match="requires canonical"):
        run()

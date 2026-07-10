"""Smoke tests for the plan.md §10 examples at reduced budgets.

Each example script reads ``VMEC_JAX_EXAMPLES_CI=1`` (parameters-at-top
hook) and shrinks its continuation schedule / trial budget; the tests run
the scripts as subprocesses in a temp cwd and assert that

- the script exits cleanly,
- the reported least-squares cost decreases from its first evaluation,
- the promised outputs (optimized input deck, wout file, figures) exist.

The "commented-out but CI-tested" extra objective terms of the optimization
examples (magnetic well, DMerc floor, L_grad_B floor) are exercised
*uncommented* in ``test_extra_terms_work_uncommented`` — exactly the
expressions the example comments show, one finite-difference least-squares
iteration each, plus the traceable magnetic-well term through
``jac="implicit"``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("netCDF4")

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
DATA_DIR = EXAMPLES / "data"

_COST_RE = re.compile(r"\[least_squares\] cost = ([0-9.eE+-]+)")


def _run_example(script: Path, cwd: Path, timeout: int = 2400) -> str:
    env = dict(os.environ, VMEC_JAX_EXAMPLES_CI="1")
    env.pop("JAX_DISABLE_JIT", None)
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"{script.name} failed (rc={proc.returncode})\n"
        f"--- stdout tail ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-4000:]}")
    return proc.stdout


def _assert_cost_decreased(stdout: str, name: str) -> None:
    costs = [float(c) for c in _COST_RE.findall(stdout)]
    assert len(costs) >= 2, f"{name}: expected verbose cost lines, got {costs}"
    assert min(costs) < costs[0], (
        f"{name}: least-squares cost did not decrease: first {costs[0]:.6e}, "
        f"best {min(costs):.6e}")


def test_fixed_boundary_run(tmp_path):
    out = _run_example(EXAMPLES / "fixed_boundary_run.py", tmp_path, timeout=900)
    assert "converged = True" in out
    outdir = tmp_path / "output_fixed_boundary_run"
    assert (outdir / "wout_li383_low_res.nc").exists()
    assert (outdir / "li383_low_res_summary.png").exists()


@pytest.mark.parametrize("case", [
    "QA",  # PR smoke: proves the QS optimization pipeline end-to-end
    pytest.param("QH", marks=pytest.mark.full),  # nightly (subprocess cold-start heavy)
    pytest.param("QP", marks=pytest.mark.full),
])
def test_qs_optimization_examples(case, tmp_path):
    script = EXAMPLES / "optimization" / f"{case}_optimization.py"
    out = _run_example(script, tmp_path)
    _assert_cost_decreased(out, case)
    outdir = tmp_path / f"output_{case}_optimization"
    assert (outdir / f"input.{case}_optimized").exists()
    assert (outdir / f"wout_{case}_optimized.nc").exists()
    assert (outdir / f"{case}_optimized_summary.png").exists()
    # the final printout carries the achieved QS total (docstring claim hook)
    match = re.search(r"QS total: seed ([0-9.eE+-]+) -> final ([0-9.eE+-]+)", out)
    assert match is not None and np.isfinite(float(match.group(2)))


@pytest.mark.full  # nightly: QP-basin + QI stages + Boozer, subprocess cold-start heavy
def test_qi_optimization_example(tmp_path):
    pytest.importorskip("booz_xform_jax")
    script = EXAMPLES / "optimization" / "QI_optimization.py"
    out = _run_example(script, tmp_path)
    _assert_cost_decreased(out, "QI")
    match = re.search(r"QI total: seed ([0-9.eE+-]+) -> final ([0-9.eE+-]+)", out)
    assert match is not None
    seed, final = float(match.group(1)), float(match.group(2))
    assert np.isfinite(final) and final <= seed * 1.05
    outdir = tmp_path / "output_QI_optimization"
    assert (outdir / "wout_QI_optimized.nc").exists()


def test_extra_terms_work_uncommented():
    """The commented-out example terms run as objectives when uncommented."""
    import jax

    from vmec_jax.core.input import VmecInput
    from vmec_jax.core import optimize as opt

    was_disabled = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)  # conftest disables jit globally
    try:
        inp = VmecInput.from_file(DATA_DIR / "input.minimal_seed_nfp2")
        # exactly the expressions the example comments show:
        extra_terms = [
            (opt.magnetic_well, 0.05, 1.0),
            (lambda eq: np.minimum(opt.d_merc(eq)[2:-1], 0.0), 0.0, 100.0),
            (lambda eq: max(1.0 / opt.l_grad_b(eq) - 1.0 / 0.35, 0.0), 0.0, 1.0),
        ]
        terms = [(opt.aspect_ratio, 5.0, 1.0)] + extra_terms
        result = opt.least_squares(terms, inp, max_mode=1, jac=None,
                                   use_ess=True, max_nfev=2)
        assert np.all(np.isfinite(result.fun))
        # the traceable extra term also differentiates through jac="implicit"
        result2 = opt.least_squares(
            [(opt.aspect_ratio, 5.0, 1.0), (opt.magnetic_well, 0.05, 1.0)],
            inp, max_mode=1, jac="implicit", use_ess=True, max_nfev=2)
        assert np.all(np.isfinite(result2.fun))
        # host wout-engine terms are rejected with a clear error in implicit mode
        with pytest.raises(ValueError, match="implicit-differentiable"):
            opt.least_squares([(opt.d_merc, 0.0, 1.0)], inp, max_mode=1,
                              jac="implicit", max_nfev=1)
    finally:
        jax.config.update("jax_disable_jit", was_disabled)

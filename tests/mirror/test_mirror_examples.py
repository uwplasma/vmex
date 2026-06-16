from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from vmec_jax.mirror import load_mirror_output
from vmec_jax.mirror.plotting.bfield import mirror_boundary_field_line_data

pytestmark = pytest.mark.mirror


def _load_run_case(script_name: str):
    script = Path("examples/mirror") / script_name
    return runpy.run_path(str(script))["run_case"]


def test_mirror_examples_write_readable_outputs_without_plots(tmp_path):
    cases = [
        ("fixed_cylinder.py", {"maxiter": 1, "write_plots": False}),
        ("fixed_flared_tube.py", {"maxiter": 1, "write_plots": False}),
        ("wham_vacuum_boundary.py", {"midplane_radius": 0.25, "maxiter": 1, "write_plots": False}),
        ("nonaxisymmetric_boundary.py", {"epsilon": 0.03, "maxiter": 1, "write_plots": False}),
    ]
    for script_name, kwargs in cases:
        run_case = _load_run_case(script_name)
        mout = run_case(tmp_path / script_name.removesuffix(".py"), **kwargs)
        output = load_mirror_output(mout)
        assert output.attributes["geometry_type"] == "mirror"
        assert output.diagnostics.min_sqrtg > 0.0
        assert output.diagnostics.mirror_ratio >= 1.0


def test_nonaxisymmetric_example_runs_as_standalone_script(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror/nonaxisymmetric_boundary.py",
            "--outdir",
            str(tmp_path / "standalone"),
            "--maxiter",
            "0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mout = Path(completed.stdout.strip())
    output = load_mirror_output(mout)
    assert output.ntheta > 1
    assert output.diagnostics.min_sqrtg > 0.0


def test_root_two_coil_axisym_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_two_coil_axisym.py",
            "--outdir",
            str(tmp_path / "two_coil"),
            "--maxiter",
            "0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mout = Path(completed.stdout.strip())
    output = load_mirror_output(mout)
    metrics = (mout.parent / "two_coil_axisym_metrics.json").read_text()
    assert output.ntheta == 1
    assert output.diagnostics.min_sqrtg > 0.0
    assert output.diagnostics.fsq >= 0.0
    assert output.diagnostics.active_force_dof > 0
    assert "axis_bz_relative_linf" in metrics
    assert "off_axis_br_relative_linf" in metrics


def test_root_finite_current_pitch_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_finite_current_pitch.py",
            "--outdir",
            str(tmp_path / "finite_current_pitch"),
            "--maxiter",
            "0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mout = Path(completed.stdout.strip())
    output = load_mirror_output(mout)
    metrics = json.loads((mout.parent / "finite_current_pitch_metrics.json").read_text())
    lines = mirror_boundary_field_line_data(output, num_lines=2)
    theta_advance = lines.theta[:, -1] - lines.theta[:, 0]
    assert output.ntheta == 1
    assert output.diagnostics.min_sqrtg > 0.0
    assert output.diagnostics.active_force_dof > 0
    assert output.profiles.i_prime[0] > 0.0
    assert np.min(theta_advance) > 1.0
    assert metrics["field_line_theta_advance_mean"] > 1.0

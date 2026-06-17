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


def test_root_fixed_boundary_solve_diagnostic_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_fixed_boundary_solve_diagnostic.py",
            "--outdir",
            str(tmp_path / "solve_diagnostic"),
            "--ns-array",
            "7",
            "--nxi",
            "13",
            "--maxiter",
            "2",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    rows = json.loads(path.read_text())
    assert len(rows) == 1
    assert rows[0]["ns"] == 7
    assert rows[0]["optimizer"] == "lbfgs"
    assert rows[0]["residual_linear_maxiter"] == 16
    assert rows[0]["residual_linear_maxiter_policy"] == "adaptive"
    assert rows[0]["residual_preconditioner"] == "radial_xi_tridi"
    assert rows[0]["residual_xi_alpha"] == pytest.approx(0.2)
    assert rows[0]["optimizer_nit"] <= 2
    assert "optimizer_rejection_reason" in rows[0]
    assert "optimizer_candidate_min_sqrtg" in rows[0]
    assert rows[0]["final_residual_norm"] >= 0.0
    assert Path(rows[0]["mout"]).exists()


def test_root_manufactured_fixed_boundary_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_manufactured_fixed_boundary.py",
            "--outdir",
            str(tmp_path / "manufactured"),
            "--ns",
            "5",
            "--nxi",
            "9",
            "--maxiter",
            "20",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert metrics["optimizer_success"]
    assert metrics["reached_projected_gtol"]
    assert metrics["final_residual_norm"] < 1.0e-12
    assert metrics["final_exact_error_norm"] < 1.0e-10


def test_root_solver_comparison_example_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_solver_comparison.py",
            "--outdir",
            str(tmp_path / "solver_comparison"),
            "--cases",
            "cylinder,two_coil,manufactured",
            "--maxiter-gd",
            "1",
            "--maxiter-lbfgs",
            "2",
            "--maxiter-newton",
            "2",
            "--two-coil-ns",
            "5",
            "--two-coil-nxi",
            "9",
            "--residual-linear-maxiter",
            "12",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    row_keys = {(row["case"], row["optimizer"], row["solver_scope"]) for row in metrics["rows"]}
    assert ("cylinder", "gradient_descent", "production_fixed_boundary") in row_keys
    assert ("cylinder", "lbfgs", "production_fixed_boundary") in row_keys
    assert ("cylinder", "residual_newton", "production_fixed_boundary") in row_keys
    assert ("two_coil", "residual_newton", "production_fixed_boundary") in row_keys
    assert ("manufactured", "residual_newton", "manufactured_source_validation") in row_keys
    assert len(metrics["histories"]) == len(metrics["rows"])
    assert all(row["final_residual_norm"] >= 0.0 for row in metrics["rows"])
    production_rows = [row for row in metrics["rows"] if row["solver_scope"] == "production_fixed_boundary"]
    assert all(row["residual_preconditioner"] == "radial_xi_tridi" for row in production_rows)
    assert all(row["residual_linear_maxiter_policy"] == "adaptive" for row in production_rows)


def test_root_residual_newton_convergence_grid_runs_without_plots(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_residual_newton_convergence_grid.py",
            "--outdir",
            str(tmp_path / "convergence_grid"),
            "--ns-array",
            "5",
            "--nxi-array",
            "9",
            "--maxiter-array",
            "2",
            "--residual-linear-maxiter-array",
            "8",
            "--preconditioners",
            "radial_xi_tridi",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert len(metrics["rows"]) == 1
    assert len(metrics["histories"]) == 1
    row = metrics["rows"][0]
    assert row["ns"] == 5
    assert row["nxi"] == 9
    assert row["optimizer"] == "residual_newton"
    assert row["residual_preconditioner"] == "radial_xi_tridi"
    assert row["residual_linear_maxiter"] == 8
    assert row["residual_linear_maxiter_policy"] == "fixed"
    assert row["residual_linear_maxiter_effective_max"] == 8
    assert row["final_residual_norm"] >= 0.0
    assert row["component_norm"] == pytest.approx(row["final_residual_norm"])
    assert row["component_active_dof"] > 0
    assert row["residual_a_norm"] >= 0.0
    assert row["residual_lam_norm"] >= 0.0
    assert row["residual_a_cap_adjacent_norm"] >= 0.0
    assert row["residual_lam_interior_xi_norm"] >= 0.0
    assert metrics["histories"][0]["row_id"] == row["row_id"]


def test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/mirror_residual_newton_convergence_grid.py",
            "--outdir",
            str(tmp_path / "finite_current_convergence_grid"),
            "--ns-array",
            "5",
            "--nxi-array",
            "9",
            "--maxiter-array",
            "1",
            "--residual-linear-maxiter-array",
            "8",
            "--residual-linear-maxiter-policy",
            "adaptive",
            "--i-prime",
            "0.01",
            "--preconditioners",
            "radial_xi_lambda_xi_tridi",
            "--residual-xi-alpha",
            "1.0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    path = Path(completed.stdout.strip())
    metrics = json.loads(path.read_text())
    assert metrics["case_label"] == "finite_current_two_coil"
    assert metrics["i_prime_value"] == pytest.approx(0.01)
    row = metrics["rows"][0]
    assert row["finite_current"]
    assert row["i_prime_value"] == pytest.approx(0.01)
    assert row["residual_preconditioner"] == "radial_xi_lambda_xi_tridi"
    assert row["residual_xi_alpha"] == pytest.approx(1.0)
    assert row["twist_proxy_i_prime_over_psi_prime"] > 0.0
    assert row["residual_lam_norm"] > 0.0
    assert row["residual_lam_fraction"] > 0.0
    assert row["residual_linear_maxiter_policy"] == "adaptive"
    assert row["residual_linear_maxiter_effective_max"] >= 8

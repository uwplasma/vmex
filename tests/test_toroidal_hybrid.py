from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from importlib import import_module

import numpy as np

from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.toroidal_hybrid import (
    evaluate_toroidal_hybrid_indata_boundary,
    sample_toroidal_stellarator_mirror_hybrid_boundary,
    toroidal_stellarator_mirror_hybrid_indata,
    toroidal_stellarator_mirror_hybrid_metrics,
)


def test_toroidal_hybrid_boundary_is_stellarator_symmetric_and_corner_localized():
    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(ntheta=32, nzeta=32)
    metrics = toroidal_stellarator_mirror_hybrid_metrics(samples)

    assert metrics["min_R"] > 0.0
    assert metrics["stellsym_R_error"] < 1.0e-13
    assert metrics["stellsym_Z_error"] < 1.0e-13
    assert metrics["corner_weight_max"] == 1.0
    assert metrics["side_weight_max"] == 1.0

    side_cols = [0, samples.zeta.size // 2]
    corner_cols = [samples.zeta.size // 4, (3 * samples.zeta.size) // 4]
    side_m2 = np.mean(np.abs(samples.R[:, side_cols] - np.mean(samples.R[:, side_cols], axis=0)))
    corner_m2 = np.mean(np.abs(samples.R[:, corner_cols] - np.mean(samples.R[:, corner_cols], axis=0)))
    assert corner_m2 > 0.5 * side_m2


def test_toroidal_hybrid_indata_roundtrips_and_reconstructs_samples(tmp_path: Path):
    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(ntheta=32, nzeta=32)
    indata = toroidal_stellarator_mirror_hybrid_indata(nfp=2, mpol=5, ntor=4, ntheta_fit=32, nzeta_fit=32)

    input_path = tmp_path / "input.hybrid"
    write_indata(input_path, indata)
    read_back = read_indata(input_path)
    reconstructed = evaluate_toroidal_hybrid_indata_boundary(read_back, ntheta=32, nzeta=32)

    np.testing.assert_allclose(reconstructed.R, samples.R, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(reconstructed.Z, samples.Z, rtol=0.0, atol=1.0e-12)
    assert read_back.get_int("NFP") == 2
    assert read_back.get_int("MPOL") == 5
    assert read_back.get_int("NTOR") == 4
    assert "RBS" not in read_back.indexed
    assert "ZBC" not in read_back.indexed


def test_toroidal_hybrid_example_runs_without_plots(tmp_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{Path.cwd()}{os.pathsep}{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid.py",
            "--outdir",
            str(tmp_path / "hybrid"),
            "--ntheta-fit",
            "32",
            "--nzeta-fit",
            "32",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    metrics_path = Path(completed.stdout.strip())
    metrics = json.loads(metrics_path.read_text())
    assert Path(metrics["input"]).exists()
    assert metrics["figures"] == {}
    assert metrics["stellsym_R_error"] < 1.0e-13
    assert metrics["stellsym_Z_error"] < 1.0e-13
    assert metrics["rbc_count"] > 3
    assert metrics["zbs_count"] > 3


def test_toroidal_hybrid_convergence_example_runs_without_solve(tmp_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{Path.cwd()}{os.pathsep}{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid_convergence.py",
            "--outdir",
            str(tmp_path / "hybrid_convergence"),
            "--ns-array",
            "7,9",
            "--mode-pairs",
            "5:4",
            "--ntheta-fit",
            "32",
            "--nzeta-fit",
            "32",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    summary_path = Path(completed.stdout.strip())
    summary = json.loads(summary_path.read_text())
    assert len(summary["rows"]) == 2
    assert Path(summary["csv"]).exists()
    assert summary["figures"] == {}
    assert all(not row["ran_solve"] for row in summary["rows"])
    assert all(row["fsq_history"] == [] for row in summary["rows"])
    assert all(row["max_boundary_fit_error"] < 1.0e-12 for row in summary["rows"])
    assert [row["ns"] for row in summary["rows"]] == [7, 9]


def test_toroidal_hybrid_convergence_history_summary_uses_iteration_labels():
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_convergence")
    summary = module._summarize_fsq_history(
        np.asarray([3.0, 2.0, 5.0]),
        iterations=np.asarray([1, 7, 11]),
    )

    assert summary["initial_fsq"] == 3.0
    assert summary["best_fsq"] == 2.0
    assert summary["best_iter"] == 7
    assert summary["fsq_reduction"] == 1.5
    assert summary["final_fsq"] == 5.0

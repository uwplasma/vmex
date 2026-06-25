from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from importlib import import_module

import numpy as np
import pytest

from vmec_jax.namelist import read_indata, write_indata
import vmec_jax.toroidal_hybrid as toroidal_hybrid
from vmec_jax.toroidal_hybrid import (
    ToroidalHybridBoundarySamples,
    evaluate_toroidal_hybrid_indata_boundary,
    recommended_square_axis_nzeta,
    sample_square_axis_stellarator_mirror_hybrid_boundary,
    sample_toroidal_stellarator_mirror_hybrid_boundary,
    square_axis_stellarator_mirror_hybrid_indata,
    toroidal_hybrid_cross_section_anisotropy,
    toroidal_hybrid_cross_section_orientation,
    toroidal_stellarator_mirror_hybrid_indata,
    toroidal_stellarator_mirror_hybrid_metrics,
)


def _assert_nonblank_image(path: str, image_module) -> None:
    pixels = image_module.imread(path)
    assert pixels.size > 0
    assert float(np.std(pixels)) > 1.0e-4


def test_toroidal_hybrid_boundary_is_stellarator_symmetric_and_corner_localized():
    import vmec_jax as vj
    import vmec_jax.api as public_api

    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(ntheta=32, nzeta=32)
    metrics = toroidal_stellarator_mirror_hybrid_metrics(samples)

    assert metrics["min_R"] > 0.0
    assert metrics["stellsym_R_error"] < 1.0e-13
    assert metrics["stellsym_Z_error"] < 1.0e-13
    assert metrics["corner_weight_max"] == 1.0
    assert metrics["side_weight_max"] == 1.0
    assert metrics["side_orientation_span"] < 1.0e-12
    assert metrics["orientation_valid_fraction"] > 0.8
    assert metrics["valid_side_orientation_span"] < 1.0e-12
    assert metrics["valid_corner_orientation_span"] > 0.05
    assert metrics["side_corner_weight_overlap_max"] <= 0.25 + 1.0e-14

    side_cols = [0, samples.zeta.size // 2]
    corner_cols = [samples.zeta.size // 4, (3 * samples.zeta.size) // 4]
    side_m2 = np.mean(np.abs(samples.R[:, side_cols] - np.mean(samples.R[:, side_cols], axis=0)))
    corner_m2 = np.mean(np.abs(samples.R[:, corner_cols] - np.mean(samples.R[:, corner_cols], axis=0)))
    assert corner_m2 > 0.5 * side_m2

    orientation = toroidal_hybrid_cross_section_orientation(samples)
    anisotropy = toroidal_hybrid_cross_section_anisotropy(samples)
    assert orientation.shape == samples.zeta.shape
    assert anisotropy.shape == samples.zeta.shape
    assert np.min(anisotropy) >= 0.0
    assert np.max(anisotropy) == pytest.approx(metrics["cross_section_anisotropy_max"])
    assert np.ptp(orientation) == pytest.approx(metrics["cross_section_orientation_span"])
    assert vj.toroidal_hybrid_cross_section_anisotropy is toroidal_hybrid_cross_section_anisotropy
    assert vj.toroidal_hybrid_cross_section_orientation is toroidal_hybrid_cross_section_orientation
    assert public_api.toroidal_hybrid_cross_section_anisotropy is toroidal_hybrid_cross_section_anisotropy
    assert public_api.toroidal_hybrid_cross_section_orientation is toroidal_hybrid_cross_section_orientation


def test_square_axis_toroidal_hybrid_boundary_and_indata_are_public():
    import vmec_jax as vj
    import vmec_jax.api as public_api

    samples = sample_square_axis_stellarator_mirror_hybrid_boundary(ntheta=32, nzeta=64)
    metrics = toroidal_stellarator_mirror_hybrid_metrics(samples)
    assert metrics["min_R"] > 0.0
    assert metrics["stellsym_R_error"] < 1.0e-13
    assert metrics["stellsym_Z_error"] < 1.0e-13
    assert metrics["side_weight_max"] == pytest.approx(1.0)
    assert metrics["corner_weight_max"] == pytest.approx(1.0)
    assert np.ptp(np.mean(samples.R, axis=0)) > 0.3

    indata = square_axis_stellarator_mirror_hybrid_indata(mpol=4, ntor=8, ntheta_fit=32, nzeta_fit=64)
    assert indata.get_int("NFP") == 1
    assert indata.get_int("MPOL") == 4
    assert indata.get_int("NTOR") == 8
    assert "RBS" not in indata.indexed
    assert "ZBC" not in indata.indexed
    assert vj.sample_square_axis_stellarator_mirror_hybrid_boundary is sample_square_axis_stellarator_mirror_hybrid_boundary
    assert vj.square_axis_stellarator_mirror_hybrid_indata is square_axis_stellarator_mirror_hybrid_indata
    assert public_api.square_axis_stellarator_mirror_hybrid_indata is square_axis_stellarator_mirror_hybrid_indata
    assert public_api.recommended_square_axis_nzeta is recommended_square_axis_nzeta


def test_square_axis_recommended_nzeta_and_example_guard(tmp_path: Path):
    assert recommended_square_axis_nzeta(12) == 32
    assert recommended_square_axis_nzeta(23) == 56
    with pytest.raises(ValueError, match="ntor must be nonnegative"):
        recommended_square_axis_nzeta(-1)

    module = import_module("examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary")
    config = module.ExampleConfig(
        outdir=tmp_path / "underresolved",
        betas_percent=(0.0,),
        ntor=12,
        nzeta=16,
        write_plots=False,
        beta_continuation_restart=False,
    )
    with pytest.raises(ValueError, match="NZETA=16 is underresolved for NTOR=12"):
        module.run_example(config)


def test_square_axis_spline_option_reduces_low_mode_projection_error():
    kwargs = {
        "ntheta": 64,
        "nzeta": 128,
        "axis_square_power": 3.0,
        "axis_spline_corner_radius_factor": 1.14,
        "minor_radius": 0.03,
        "side_elongation": 0.08,
        "side_minor_modulation": 0.08,
        "corner_ellipticity": 0.04,
        "corner_amplitude": 0.004,
        "corner_rotation": 0.30,
    }
    errors = {}
    for axis_kind in ("superellipse", "spline"):
        samples = sample_square_axis_stellarator_mirror_hybrid_boundary(axis_kind=axis_kind, **kwargs)
        indata = square_axis_stellarator_mirror_hybrid_indata(
            mpol=6,
            ntor=12,
            ntheta_fit=64,
            nzeta_fit=128,
            axis_kind=axis_kind,
            **{key: value for key, value in kwargs.items() if key not in {"ntheta", "nzeta"}},
        )
        reconstructed = evaluate_toroidal_hybrid_indata_boundary(indata, ntheta=64, nzeta=128)
        errors[axis_kind] = max(
            float(np.max(np.abs(reconstructed.R - samples.R))),
            float(np.max(np.abs(reconstructed.Z - samples.Z))),
        )

    assert errors["spline"] < errors["superellipse"]
    assert errors["spline"] < 2.0e-4


def test_toroidal_hybrid_localization_powers_sharpen_side_and_corner_regions():
    base = sample_toroidal_stellarator_mirror_hybrid_boundary(ntheta=32, nzeta=32)
    sharp = sample_toroidal_stellarator_mirror_hybrid_boundary(
        ntheta=32,
        nzeta=32,
        side_power=3.0,
        corner_power=3.0,
    )
    metrics = toroidal_stellarator_mirror_hybrid_metrics(sharp)

    assert metrics["stellsym_R_error"] < 1.0e-13
    assert metrics["stellsym_Z_error"] < 1.0e-13
    assert sharp.side_weight.max() == pytest.approx(1.0)
    assert sharp.corner_weight.max() == pytest.approx(1.0)
    assert sharp.side_weight.mean() < base.side_weight.mean()
    assert sharp.corner_weight.mean() < base.corner_weight.mean()


def test_toroidal_hybrid_indata_roundtrips_and_reconstructs_samples(tmp_path: Path):
    sample_kwargs = {
        "side_minor_modulation": 0.16,
        "side_elongation": 0.35,
        "side_power": 2.0,
        "corner_amplitude": 0.025,
        "corner_ellipticity": 0.22,
        "corner_rotation": 0.42,
        "corner_power": 2.0,
    }
    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(ntheta=64, nzeta=64, **sample_kwargs)
    indata = toroidal_stellarator_mirror_hybrid_indata(
        nfp=2,
        mpol=5,
        ntor=20,
        ntheta_fit=64,
        nzeta_fit=64,
        **sample_kwargs,
    )

    input_path = tmp_path / "input.hybrid"
    write_indata(input_path, indata)
    read_back = read_indata(input_path)
    reconstructed = evaluate_toroidal_hybrid_indata_boundary(read_back, ntheta=64, nzeta=64)

    np.testing.assert_allclose(reconstructed.R, samples.R, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(reconstructed.Z, samples.Z, rtol=0.0, atol=1.0e-12)
    assert read_back.get_int("NFP") == 2
    assert read_back.get_int("MPOL") == 5
    assert read_back.get_int("NTOR") == 20
    assert "RBS" not in read_back.indexed
    assert "ZBC" not in read_back.indexed


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ntheta": 7}, "at least 8"),
        ({"minor_radius": 0.0}, "major_radius"),
        ({"major_radius": 0.1, "minor_radius": 0.2}, "major_radius"),
        ({"corner_helicity": -1}, "nonnegative"),
        ({"corner_ellipticity": -0.1}, "corner_ellipticity"),
        ({"corner_ellipticity": 1.0}, "corner_ellipticity"),
        ({"corner_rotation": np.inf}, "corner_rotation"),
        (
            {
                "major_radius": 0.3,
                "minor_radius": 0.29,
                "axis_oval": -0.2,
                "side_minor_modulation": 0.3,
                "corner_amplitude": 0.1,
            },
            "nonpositive cylindrical R",
        ),
        ({"side_power": 0.0}, "side_power"),
        ({"corner_power": np.inf}, "corner_power"),
    ],
)
def test_toroidal_hybrid_boundary_rejects_invalid_geometry(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sample_toroidal_stellarator_mirror_hybrid_boundary(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"nfp": 0}, "nfp"),
        ({"mpol": 2}, "mpol"),
        ({"ntor": 2}, "ntor"),
        ({"ntor": 3, "corner_helicity": 2}, "ntor"),
    ],
)
def test_toroidal_hybrid_indata_rejects_invalid_mode_extent(kwargs, match):
    with pytest.raises(ValueError, match=match):
        toroidal_stellarator_mirror_hybrid_indata(**kwargs)


def test_toroidal_hybrid_indata_rejects_non_stellarator_symmetric_samples(monkeypatch):
    def asymmetric_samples(*, ntheta, nzeta, **_kwargs):
        theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
        zeta = np.linspace(0.0, 2.0 * np.pi, int(nzeta), endpoint=False)
        theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")
        return ToroidalHybridBoundarySamples(
            theta=theta,
            zeta=zeta,
            R=1.2 + 0.05 * np.cos(theta2) + 0.01 * np.sin(theta2 + zeta2),
            Z=0.12 * np.sin(theta2),
            side_weight=np.cos(zeta2) ** 2,
            corner_weight=np.sin(zeta2) ** 2,
        )

    monkeypatch.setattr(
        toroidal_hybrid,
        "sample_toroidal_stellarator_mirror_hybrid_boundary",
        asymmetric_samples,
    )

    with pytest.raises(ValueError, match="not stellarator symmetric"):
        toroidal_hybrid.toroidal_stellarator_mirror_hybrid_indata(
            mpol=4,
            ntor=4,
            ntheta_fit=32,
            nzeta_fit=32,
            coeff_tol=1.0e-14,
        )


def test_toroidal_hybrid_example_runs_without_plots(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid.py",
            "--outdir",
            str(tmp_path / "hybrid"),
            "--ntheta-fit",
            "64",
            "--nzeta-fit",
            "64",
            "--ntor",
            "10",
            "--side-minor-modulation",
            "0.16",
            "--side-elongation",
            "0.35",
            "--side-power",
            "2.0",
            "--corner-amplitude",
            "0.025",
            "--corner-ellipticity",
            "0.22",
            "--corner-rotation",
            "0.42",
            "--corner-power",
            "2.0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics_path = Path(completed.stdout.strip())
    metrics = json.loads(metrics_path.read_text())
    assert Path(metrics["input"]).exists()
    assert metrics["figures"] == {}
    assert metrics["hybrid_fixture_kind"] == "toroidal_stellarator_mirror_hybrid"
    assert metrics["final_hybrid_target_kind"] == "toroidal_stellarator_mirror_hybrid"
    assert metrics["production_hybrid_claim"] is False
    assert metrics["stellsym_R_error"] < 1.0e-13
    assert metrics["stellsym_Z_error"] < 1.0e-13
    assert metrics["rbc_count"] > 3
    assert metrics["zbs_count"] > 3
    assert metrics["sample_parameters"]["side_minor_modulation"] == 0.16
    assert metrics["sample_parameters"]["side_elongation"] == 0.35
    assert metrics["ntor"] == 10
    assert metrics["sample_parameters"]["side_power"] == 2.0
    assert metrics["sample_parameters"]["corner_amplitude"] == 0.025
    assert metrics["sample_parameters"]["corner_ellipticity"] == 0.22
    assert metrics["sample_parameters"]["corner_rotation"] == 0.42
    assert metrics["sample_parameters"]["corner_power"] == 2.0


def test_toroidal_hybrid_example_writes_nonblank_plots(tmp_path: Path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid.py",
            "--outdir",
            str(tmp_path / "hybrid_plots"),
            "--ntheta-fit",
            "64",
            "--nzeta-fit",
            "64",
            "--ntor",
            "10",
            "--side-minor-modulation",
            "0.16",
            "--side-elongation",
            "0.35",
            "--side-power",
            "2.0",
            "--corner-amplitude",
            "0.025",
            "--corner-ellipticity",
            "0.22",
            "--corner-rotation",
            "0.42",
            "--corner-power",
            "2.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(Path(completed.stdout.strip()).read_text())
    assert set(metrics["figures"]) == {"lcfs_3d", "top_view", "cross_sections", "region_orientation"}
    for path in metrics["figures"].values():
        _assert_nonblank_image(path, image)


def test_square_coil_hybrid_free_boundary_example_runs_without_plots(tmp_path: Path):
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary")
    metrics_path = module.run_example(
        module.ExampleConfig(
            outdir=tmp_path / "square_coils",
            betas_percent=(0.0,),
            n_coils_per_side=1,
            coil_segments=24,
            mpol=3,
            ntor=4,
            ns=5,
            ns_array=(5,),
            nzeta=8,
            max_iter=2,
            ftol=1.0e-6,
            niter_array=(2,),
            ftol_array=(1.0e-6,),
            use_multigrid_schedule=False,
            enforce_recommended_nzeta=False,
            field_line_count=1,
            field_line_steps=20,
            field_line_turns=0.2,
            write_plots=False,
            jit_forces=False,
        )
    )

    metrics = json.loads(metrics_path.read_text())
    rows = metrics["rows"]
    assert metrics["metrics_schema"] == "toroidal_stellarator_mirror_hybrid_square_coils_free_boundary_solve"
    assert metrics["workflow_status"] == "actual_vmec_jax_free_boundary_beta_scan"
    assert metrics["actual_free_boundary_solve"] is True
    assert metrics["production_free_boundary_claim"] is False
    assert metrics["hybrid_fixture_kind"] == "square_axis_toroidal_stellarator_mirror_hybrid"
    assert metrics["coil_count"] == 4
    assert metrics["n_coils_per_side"] == 1
    assert metrics["betas_percent"] == [0.0]
    assert metrics["figures"] == {}
    assert Path(metrics["coils_json"]).exists()
    assert Path(metrics["summary_csv"]).exists()
    assert len(rows) == 1
    assert Path(rows[0]["input"]).exists()
    assert Path(rows[0]["wout"]).exists()
    assert np.isfinite(float(rows[0]["final_fsqr"]))
    assert np.isfinite(float(rows[0]["final_fsqz"]))
    assert np.isfinite(float(rows[0]["final_fsql"]))
    assert rows[0]["free_boundary_bnormal_rms"] is not None
    assert rows[0]["virtual_casing_status"] == "computed" or rows[0]["virtual_casing_status"].startswith(
        ("skipped_", "failed:")
    )
    assert rows[0]["mean_iota"] is not None
    with Path(metrics["summary_csv"]).open(newline="") as file_obj:
        csv_rows = list(csv.DictReader(file_obj))
    assert len(csv_rows) == len(rows)
    assert csv_rows[0]["beta_percent"] == "0.0"
    assert "virtual_casing_status" in csv_rows[0]


def test_square_coil_hybrid_free_boundary_example_writes_nonblank_plots(tmp_path: Path):
    image = pytest.importorskip("matplotlib.image")
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary")
    metrics_path = module.run_example(
        module.ExampleConfig(
            outdir=tmp_path / "square_coils_plots",
            betas_percent=(0.0, 10.0),
            n_coils_per_side=1,
            coil_segments=24,
            mpol=3,
            ntor=4,
            ns=5,
            ns_array=(5,),
            nzeta=8,
            max_iter=1,
            ftol=1.0e-6,
            niter_array=(1,),
            ftol_array=(1.0e-6,),
            use_multigrid_schedule=False,
            enforce_recommended_nzeta=False,
            field_line_count=1,
            field_line_steps=24,
            field_line_turns=0.25,
            write_plots=True,
            jit_forces=False,
        )
    )

    metrics = json.loads(metrics_path.read_text())
    assert set(metrics["figures"]) == {
        "geometry_3d",
        "top_view",
        "cross_sections",
        "boundary_bmag",
        "beta_response",
        "convergence_iota",
    }
    assert len(metrics["rows"]) == 2
    for path in metrics["figures"].values():
        _assert_nonblank_image(path, image)


def test_toroidal_hybrid_convergence_example_runs_without_solve(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid_convergence.py",
            "--outdir",
            str(tmp_path / "hybrid_convergence"),
            "--ns-array",
            "7,9",
            "--mode-pairs",
            "5:20",
            "--ntheta-fit",
            "64",
            "--nzeta-fit",
            "64",
            "--side-power",
            "2.0",
            "--corner-power",
            "2.0",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary_path = Path(completed.stdout.strip())
    summary = json.loads(summary_path.read_text())
    assert len(summary["rows"]) == 2
    assert Path(summary["csv"]).exists()
    assert summary["figures"] == {}
    assert summary["resolution_preset"] == "manual"
    assert summary["target_resolution_ladder"] is False
    assert summary["target_resolution_promotion_claim"] is False
    assert all(not row["ran_solve"] for row in summary["rows"])
    assert all(row["resolution_preset"] == "manual" for row in summary["rows"])
    assert all(row["target_resolution_ladder"] is False for row in summary["rows"])
    assert all(row["target_resolution_promotion_claim"] is False for row in summary["rows"])
    assert all(row["cli_finish"] is True for row in summary["rows"])
    assert all(row["nstep"] == 25 for row in summary["rows"])
    assert all(row["full_solver_diagnostics"] is False for row in summary["rows"])
    assert all(row["diagnostic_step_history_size"] == 0 for row in summary["rows"])
    assert all(row["diagnostic_initial_axis_reset_attempted"] is None for row in summary["rows"])
    assert all(row["diagnostic_initial_axis_reset_reset"] is None for row in summary["rows"])
    assert all(row["initialization_policy"] == "vmec_jax_default_input_boundary" for row in summary["rows"])
    assert all(
        row["vmec_jax_axis_initialization_policy"] == "boundary_inferred_missing_axis" for row in summary["rows"]
    )
    assert all(row["vmec2000_initialization_policy"] == "vmec2000_default_input_boundary" for row in summary["rows"])
    assert all(row["direct_initial_residual_requested"] is True for row in summary["rows"])
    assert all(row["direct_initial_residual_source"] is None for row in summary["rows"])
    assert all(row["direct_initial_fsq"] is None for row in summary["rows"])
    assert all(row["direct_initial_fsq_ratio_vmec2000"] is None for row in summary["rows"])
    assert all(row["initial_fsq_ratio_direct_initial"] is None for row in summary["rows"])
    assert all(row["vmec2000_initial_fsq_ratio_direct_initial"] is None for row in summary["rows"])
    assert all(row["initial_residual_source"] is None for row in summary["rows"])
    assert all(row["vmec2000_initial_residual_source"] is None for row in summary["rows"])
    assert all(row["initial_fsq_ratio_vmec2000"] is None for row in summary["rows"])
    assert all(row["fsq_history"] == [] for row in summary["rows"])
    assert all(row["max_boundary_fit_error"] < 1.0e-12 for row in summary["rows"])
    assert all(row["max_orientation_fit_error"] < 1.0e-12 for row in summary["rows"])
    assert all(0.8 < row["orientation_fit_valid_fraction"] <= 1.0 for row in summary["rows"])
    assert all(row["side_orientation_span"] < 1.0e-12 for row in summary["rows"])
    assert all(row["valid_side_orientation_span"] < 1.0e-12 for row in summary["rows"])
    assert all(row["valid_corner_orientation_span"] > 0.05 for row in summary["rows"])
    assert all(row["fitted_side_orientation_span"] < 1.0e-12 for row in summary["rows"])
    assert all(row["fitted_valid_side_orientation_span"] < 1.0e-12 for row in summary["rows"])
    assert all(row["fitted_valid_corner_orientation_span"] > 0.05 for row in summary["rows"])
    assert all(row["cross_section_anisotropy_max"] > 0.0 for row in summary["rows"])
    assert all(row["fitted_cross_section_anisotropy_max"] > 0.0 for row in summary["rows"])
    assert [row["ns"] for row in summary["rows"]] == [7, 9]
    assert all(row["ntor"] == 20 for row in summary["rows"])
    assert summary["shape_cases"][0]["sample_parameters"]["side_power"] == 2.0
    assert summary["shape_cases"][0]["sample_parameters"]["corner_power"] == 2.0
    with Path(summary["csv"]).open(newline="") as file_obj:
        csv_row = next(csv.DictReader(file_obj))
    assert csv_row["initialization_policy"] == "vmec_jax_default_input_boundary"
    assert csv_row["resolution_preset"] == "manual"
    assert csv_row["target_resolution_ladder"] == "False"
    assert csv_row["target_resolution_promotion_claim"] == "False"
    assert csv_row["cli_finish"] == "True"
    assert csv_row["nstep"] == "25"
    assert csv_row["full_solver_diagnostics"] == "False"
    assert csv_row["diagnostic_step_history_size"] == "0"
    assert csv_row["diagnostic_initial_axis_reset_attempted"] == ""
    assert csv_row["diagnostic_initial_axis_reset_reset"] == ""
    assert csv_row["vmec_jax_axis_initialization_policy"] == "boundary_inferred_missing_axis"
    assert csv_row["vmec2000_initialization_policy"] == "vmec2000_default_input_boundary"
    assert csv_row["direct_initial_residual_requested"] == "True"
    assert csv_row["direct_initial_residual_source"] == ""
    assert csv_row["direct_initial_fsq_ratio_vmec2000"] == ""
    assert csv_row["initial_fsq_ratio_direct_initial"] == ""
    assert csv_row["vmec2000_initial_fsq_ratio_direct_initial"] == ""
    assert csv_row["initial_residual_source"] == ""
    assert csv_row["vmec2000_initial_residual_source"] == ""
    assert csv_row["initial_fsq_ratio_vmec2000"] == ""
    assert float(csv_row["max_orientation_fit_error"]) < 1.0e-12
    assert 0.8 < float(csv_row["orientation_fit_valid_fraction"]) <= 1.0
    assert float(csv_row["fitted_valid_corner_orientation_span"]) > 0.05


def test_toroidal_hybrid_convergence_example_target_preset_without_solve(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid_convergence.py",
            "--outdir",
            str(tmp_path / "hybrid_target_preset"),
            "--resolution-preset",
            "target",
            "--ntheta-fit",
            "64",
            "--nzeta-fit",
            "64",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(Path(completed.stdout.strip()).read_text())
    rows = summary["rows"]
    assert summary["resolution_preset"] == "target"
    assert summary["target_resolution_ladder"] is True
    assert summary["target_resolution_promotion_claim"] is False
    assert len(rows) == 6
    assert sorted({row["ns"] for row in rows}) == [7, 9, 15]
    assert sorted({(row["mpol"], row["ntor"]) for row in rows}) == [(5, 20), (6, 24)]
    assert all(row["resolution_preset"] == "target" for row in rows)
    assert all(row["target_resolution_ladder"] is True for row in rows)
    assert all(row["target_resolution_promotion_claim"] is False for row in rows)
    assert all(row["ran_solve"] is False for row in rows)
    assert all(row["max_boundary_fit_error"] < 1.0e-12 for row in rows)
    with Path(summary["csv"]).open(newline="") as file_obj:
        csv_rows = list(csv.DictReader(file_obj))
    assert len(csv_rows) == len(rows)
    assert {row["resolution_preset"] for row in csv_rows} == {"target"}
    assert {row["target_resolution_ladder"] for row in csv_rows} == {"True"}
    assert {row["target_resolution_promotion_claim"] for row in csv_rows} == {"False"}


def test_toroidal_hybrid_convergence_example_filters_target_preset_cases(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid_convergence.py",
            "--outdir",
            str(tmp_path / "hybrid_target_filtered"),
            "--resolution-preset",
            "target",
            "--case-filter",
            "*ns015*",
            "--ntheta-fit",
            "64",
            "--nzeta-fit",
            "64",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(Path(completed.stdout.strip()).read_text())
    rows = summary["rows"]
    assert summary["resolution_preset"] == "target"
    assert summary["case_filters"] == ["*ns015*"]
    assert len(rows) == 2
    assert {row["ns"] for row in rows} == {15}
    assert sorted((row["mpol"], row["ntor"]) for row in rows) == [(5, 20), (6, 24)]
    assert all("ns015" in row["case"] for row in rows)


def test_toroidal_hybrid_convergence_example_aggregates_chunk_jsons(tmp_path: Path):
    chunk_a = tmp_path / "chunk_a.json"
    chunk_b = tmp_path / "chunk_b.json"
    chunk_a.write_text(
        json.dumps(
            {
                "resolution_preset": "target",
                "target_resolution_ladder": True,
                "case_filters": ["*ns007*", "*ns009*"],
                "rows": [
                    {
                        "case": "ns009_mpol05_ntor20",
                        "shape_case": "custom",
                        "resolution_preset": "target",
                        "target_resolution_ladder": True,
                        "target_resolution_promotion_claim": False,
                        "ns": 9,
                        "mpol": 5,
                        "ntor": 20,
                        "ran_solve": True,
                        "ran_vmec2000": True,
                        "converged_by_total_fsq": True,
                        "vmec2000_returncode": 0,
                        "requested_ftol": 0.01,
                        "direct_initial_fsq_ratio_vmec2000": 1.006,
                        "best_fsq": 0.04,
                        "best_fsqr": 0.004,
                        "best_fsqz": 0.02,
                        "best_fsql": 0.01,
                        "final_fsq": 0.05,
                        "final_fsqr": 0.005,
                        "final_fsqz": 0.02,
                        "final_fsql": 0.015,
                        "vmec2000_final_fsq": 0.009,
                        "vmec2000_final_fsqr": 0.003,
                        "vmec2000_final_fsqz": 0.004,
                        "vmec2000_final_fsql": 0.002,
                        "max_boundary_fit_error": 1.0e-14,
                    },
                    {
                        "case": "ns007_mpol05_ntor20",
                        "shape_case": "custom",
                        "resolution_preset": "target",
                        "target_resolution_ladder": True,
                        "target_resolution_promotion_claim": False,
                        "ns": 7,
                        "mpol": 5,
                        "ntor": 20,
                        "ran_solve": True,
                        "ran_vmec2000": True,
                        "converged_by_total_fsq": False,
                        "vmec2000_returncode": 1,
                        "requested_ftol": 0.01,
                        "direct_initial_fsq_ratio_vmec2000": 9.0,
                        "best_fsq": 9.0,
                        "best_fsqr": 2.0,
                        "best_fsqz": 3.0,
                        "best_fsql": 4.0,
                        "final_fsq": 9.0,
                        "final_fsqr": 2.0,
                        "final_fsqz": 3.0,
                        "final_fsql": 4.0,
                        "vmec2000_final_fsq": 9.0,
                        "max_boundary_fit_error": 1.0e-14,
                    },
                ],
            }
        )
        + "\n"
    )
    chunk_b.write_text(
        json.dumps(
            {
                "resolution_preset": "target",
                "target_resolution_ladder": True,
                "case_filters": ["*ns007*", "*ns015*"],
                "rows": [
                    {
                        "case": "ns007_mpol05_ntor20",
                        "shape_case": "custom",
                        "resolution_preset": "target",
                        "target_resolution_ladder": True,
                        "target_resolution_promotion_claim": False,
                        "ns": 7,
                        "mpol": 5,
                        "ntor": 20,
                        "ran_solve": True,
                        "ran_vmec2000": True,
                        "converged_by_total_fsq": True,
                        "vmec2000_returncode": 0,
                        "requested_ftol": 0.01,
                        "direct_initial_fsq_ratio_vmec2000": 1.002,
                        "best_fsq": 0.03,
                        "best_fsqr": 0.004,
                        "best_fsqz": 0.003,
                        "best_fsql": 0.002,
                        "final_fsq": 0.04,
                        "final_fsqr": 0.005,
                        "final_fsqz": 0.004,
                        "final_fsql": 0.003,
                        "vmec2000_final_fsq": 0.008,
                        "vmec2000_final_fsqr": 0.002,
                        "vmec2000_final_fsqz": 0.003,
                        "vmec2000_final_fsql": 0.004,
                        "max_boundary_fit_error": 1.0e-14,
                    },
                    {
                        "case": "ns015_mpol06_ntor24",
                        "shape_case": "custom",
                        "resolution_preset": "target",
                        "target_resolution_ladder": True,
                        "target_resolution_promotion_claim": False,
                        "ns": 15,
                        "mpol": 6,
                        "ntor": 24,
                        "ran_solve": True,
                        "ran_vmec2000": True,
                        "converged_by_total_fsq": False,
                        "vmec2000_returncode": 0,
                        "requested_ftol": 0.01,
                        "direct_initial_fsq_ratio_vmec2000": 1.004,
                        "best_fsq": 0.06,
                        "best_fsqr": 0.03,
                        "best_fsqz": 0.01,
                        "best_fsql": 0.02,
                        "final_fsq": 0.12,
                        "final_fsqr": 0.03,
                        "final_fsqz": 0.02,
                        "final_fsql": 0.04,
                        "vmec2000_final_fsq": 0.022,
                        "vmec2000_final_fsqr": 0.005,
                        "vmec2000_final_fsqz": 0.006,
                        "vmec2000_final_fsql": 0.007,
                        "max_boundary_fit_error": 1.0e-14,
                    },
                ],
            }
        )
        + "\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid_convergence.py",
            "--outdir",
            str(tmp_path / "aggregate"),
            "--aggregate-json",
            str(chunk_a),
            str(chunk_b),
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(Path(completed.stdout.strip()).read_text())
    rows = summary["rows"]
    assert summary["aggregate_schema"] == "toroidal_stellarator_mirror_hybrid_convergence_aggregate.v1"
    assert summary["case_count"] == 3
    assert summary["duplicate_cases_replaced"] == ["ns007_mpol05_ntor20"]
    assert [row["case"] for row in rows] == [
        "ns007_mpol05_ntor20",
        "ns009_mpol05_ntor20",
        "ns015_mpol06_ntor24",
    ]
    assert rows[0]["vmec2000_returncode"] == 0
    assert Path(rows[0]["aggregate_source_json"]) == chunk_b.resolve()
    metrics = summary["aggregate_metrics"]
    assert metrics["row_count"] == 3
    assert metrics["ran_solve_rows"] == 3
    assert metrics["vmec2000_returncode_zero_rows"] == 3
    assert metrics["vmec_jax_total_fsq_converged_rows"] == 2
    assert metrics["vmec_jax_strict_component_known_rows"] == 3
    assert metrics["vmec_jax_strict_component_pass_rows"] == 1
    assert metrics["vmec_jax_strict_component_blocker_counts"] == {"fsql": 1, "fsqz": 1}
    assert metrics["vmec_jax_final_max_component_over_ftol_min"] == pytest.approx(0.5)
    assert metrics["vmec_jax_final_max_component_over_ftol_max"] == pytest.approx(4.0)
    assert metrics["vmec2000_final_max_component_over_ftol_max"] == pytest.approx(0.7)
    assert metrics["direct_initial_fsq_ratio_vmec2000_min"] == pytest.approx(1.002)
    assert metrics["direct_initial_fsq_ratio_vmec2000_max"] == pytest.approx(1.006)
    assert metrics["best_fsq_min"] == pytest.approx(0.03)
    assert metrics["final_fsq_max"] == pytest.approx(0.12)
    assert rows[0]["strict_component_pass"] is True
    assert rows[0]["strict_component_bottleneck"] is None
    assert rows[1]["strict_component_bottleneck"] == "fsqz"
    assert rows[2]["strict_component_bottleneck"] == "fsql"
    assert Path(summary["csv"]).exists()
    with Path(summary["csv"]).open(newline="") as file_obj:
        csv_rows = list(csv.DictReader(file_obj))
    assert [row["case"] for row in csv_rows] == [row["case"] for row in rows]
    assert Path(csv_rows[0]["aggregate_source_json"]) == chunk_b.resolve()
    assert csv_rows[2]["final_max_component_name"] == "fsql"
    assert float(csv_rows[2]["final_max_component_over_ftol"]) == pytest.approx(4.0)


def test_toroidal_hybrid_convergence_example_writes_nonblank_no_solve_plots(tmp_path: Path):
    image = pytest.importorskip("matplotlib.image")
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid_convergence.py",
            "--outdir",
            str(tmp_path / "hybrid_convergence_plots"),
            "--ns-array",
            "7,9",
            "--mode-pairs",
            "5:20",
            "--ntheta-fit",
            "64",
            "--nzeta-fit",
            "64",
            "--side-power",
            "2.0",
            "--corner-power",
            "2.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(Path(completed.stdout.strip()).read_text())
    assert set(summary["figures"]) == {"convergence", "orientation"}
    for path in summary["figures"].values():
        _assert_nonblank_image(path, image)


def test_toroidal_hybrid_convergence_example_scans_shape_cases_without_solve(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            "examples/toroidal_stellarator_mirror_hybrid_convergence.py",
            "--outdir",
            str(tmp_path / "hybrid_shape_scan"),
            "--ns-array",
            "7",
            "--mode-pairs",
            "5:20",
            "--ntheta-fit",
            "64",
            "--nzeta-fit",
            "64",
            "--shape-cases",
            "default,sharp",
            "--no-plots",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary_path = Path(completed.stdout.strip())
    summary = json.loads(summary_path.read_text())
    rows = summary["rows"]
    assert [case["name"] for case in summary["shape_cases"]] == ["default", "sharp"]
    assert [row["shape_case"] for row in rows] == ["default", "sharp"]
    assert [row["case"] for row in rows] == ["default_ns007_mpol05_ntor20", "sharp_ns007_mpol05_ntor20"]
    assert rows[0]["side_power"] == 1.0
    assert rows[1]["side_power"] == 2.0
    assert rows[1]["corner_ellipticity"] == 0.22
    assert rows[1]["corner_rotation"] == 0.42
    assert rows[1]["corner_power"] == 2.0
    assert all(row["max_boundary_fit_error"] < 1.0e-12 for row in rows)
    assert all(row["max_orientation_fit_error"] < 1.0e-12 for row in rows)
    assert all(0.8 < row["orientation_fit_valid_fraction"] <= 1.0 for row in rows)
    assert all(row["fitted_side_orientation_span"] < 1.0e-12 for row in rows)
    assert all(row["fitted_valid_corner_orientation_span"] > 0.05 for row in rows)


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
    assert module._parse_shape_cases("default, sharp") == ["default", "sharp"]
    assert module._parse_case_filters("a*, b") == ("a*", "b")
    assert module._parse_path_args(["a,b", "c"]) == [Path("a"), Path("b"), Path("c")]
    assert module._case_matches_filters("sharp_ns015_mpol05_ntor20", ("*ns015*",))
    assert not module._case_matches_filters("sharp_ns009_mpol05_ntor20", ("*ns015*",))
    np.testing.assert_array_equal(
        module._row_history_iterations({"iter_history": [3, 5, 9]}, 3),
        np.asarray([3, 5, 9]),
    )
    np.testing.assert_array_equal(
        module._row_history_iterations({"iter_history": [3]}, 3),
        np.asarray([1, 2, 3]),
    )
    diag_fields = module._solver_diagnostic_fields(
        {
            "light_history": False,
            "resume_state_mode": "minimal",
            "multigrid_stage_modes": np.asarray(["accelerated"], dtype=object),
            "multigrid_niter_stages": np.asarray([8], dtype=int),
            "multigrid_stage_offsets": np.asarray([0], dtype=int),
            "iter2_history": np.asarray([1, 2], dtype=int),
            "step_status_history": np.asarray(["accepted", "accepted"], dtype=object),
            "restart_reason_history": np.asarray(["none", "bad_progress"], dtype=object),
            "dt_eff_history": np.asarray([0.1, 0.05], dtype=float),
            "update_rms_history": np.asarray([1.0e-3, 5.0e-4], dtype=float),
            "w_try_ratio_history": np.asarray([0.8, 1.2], dtype=float),
            "bcovar_update_history": np.asarray([1, 0], dtype=int),
            "initial_axis_reset_attempted": True,
            "initial_axis_reset_reset": True,
            "initial_axis_reset_bad_jacobian": True,
            "initial_axis_reset_force_reset": False,
            "initial_axis_reset_fsq": 7.0,
            "initial_axis_reset_ptau_min": -1.0,
            "initial_axis_reset_ptau_max": 2.0,
            "initial_axis_reset_state_tau_min": -0.5,
            "initial_axis_reset_state_tau_max": 1.5,
            "initial_axis_reset_error": None,
        },
        fallback_size=2,
    )
    assert diag_fields["diagnostic_light_history"] is False
    assert diag_fields["diagnostic_resume_state_mode"] == "minimal"
    assert diag_fields["diagnostic_stage_modes"] == ["accelerated"]
    assert diag_fields["diagnostic_step_iter_history"] == [1, 2]
    assert diag_fields["diagnostic_step_history_size"] == 2
    assert diag_fields["diagnostic_time_step_history_size"] == 0
    assert diag_fields["diagnostic_step_status_counts"] == {"accepted": 2}
    assert diag_fields["diagnostic_restart_reason_counts"] == {"none": 1, "bad_progress": 1}
    assert diag_fields["diagnostic_bcovar_updates"] == 1
    assert diag_fields["diagnostic_initial_bcovar_update"] is True
    assert diag_fields["diagnostic_final_dt_eff"] == 0.05
    assert diag_fields["diagnostic_max_update_rms"] == 1.0e-3
    assert diag_fields["diagnostic_final_update_rms"] == 5.0e-4
    assert diag_fields["diagnostic_initial_axis_reset_attempted"] is True
    assert diag_fields["diagnostic_initial_axis_reset_reset"] is True
    assert diag_fields["diagnostic_initial_axis_reset_bad_jacobian"] is True
    assert diag_fields["diagnostic_initial_axis_reset_force_reset"] is False
    assert diag_fields["diagnostic_initial_axis_reset_fsq"] == 7.0
    assert diag_fields["diagnostic_initial_axis_reset_ptau_min"] == -1.0
    assert diag_fields["diagnostic_initial_axis_reset_ptau_max"] == 2.0
    assert diag_fields["diagnostic_initial_axis_reset_state_tau_min"] == -0.5
    assert diag_fields["diagnostic_initial_axis_reset_state_tau_max"] == 1.5
    assert diag_fields["diagnostic_initial_axis_reset_error"] is None
    scan_diag_fields = module._solver_diagnostic_fields(
        {
            "light_history": False,
            "resume_state_mode": "minimal",
            "scan_path": "vmec2000",
            "scan_minimal": False,
            "scan_use_precomputed": True,
            "scan_use_lax_tridi": False,
            "time_step_history": np.asarray([0.2, 0.1, 0.05], dtype=float),
        },
        fallback_size=3,
    )
    assert scan_diag_fields["diagnostic_light_history"] is False
    assert scan_diag_fields["diagnostic_scan_path"] == "vmec2000"
    assert scan_diag_fields["diagnostic_scan_minimal"] is False
    assert scan_diag_fields["diagnostic_scan_light"] is False
    assert scan_diag_fields["diagnostic_scan_use_precomputed"] is True
    assert scan_diag_fields["diagnostic_scan_use_lax_tridi"] is False
    assert scan_diag_fields["diagnostic_step_history_size"] == 3
    assert scan_diag_fields["diagnostic_step_iter_history"] == [1, 2, 3]
    assert scan_diag_fields["diagnostic_time_step_history"] == [0.2, 0.1, 0.05]
    assert scan_diag_fields["diagnostic_time_step_history_size"] == 3
    assert scan_diag_fields["diagnostic_initial_time_step"] == 0.2
    assert scan_diag_fields["diagnostic_final_time_step"] == 0.05
    assert scan_diag_fields["diagnostic_min_time_step"] == 0.05
    assert scan_diag_fields["diagnostic_max_time_step"] == 0.2
    cli_finish_fields = module._cli_finish_diagnostic_fields(
        {
            "cli_fixed_boundary_mode": True,
            "cli_fixed_boundary_initial_policy": "boundary_inferred_missing_axis",
            "cli_fixed_boundary_finish_budgets": np.asarray([20, 40], dtype=int),
            "cli_fixed_boundary_finish_fsq": np.asarray([1.0e-4, 2.0e-8], dtype=float),
            "cli_fixed_boundary_finish_converged": np.asarray([False, True], dtype=bool),
            "cli_fixed_boundary_finish_modes": np.asarray(["accelerated", "parity"], dtype=object),
            "cli_fixed_boundary_finish_budget_cap": 80,
            "cli_fixed_boundary_finish_budget_exhausted": False,
            "cli_fixed_boundary_full_parity_fallback": True,
            "cli_fixed_boundary_partial_parity_fallback": False,
            "cli_fixed_boundary_staged_followup_used": True,
        }
    )
    assert cli_finish_fields["cli_fixed_boundary_mode"] is True
    assert cli_finish_fields["cli_fixed_boundary_initial_policy"] == "boundary_inferred_missing_axis"
    assert cli_finish_fields["cli_fixed_boundary_finish_attempts"] == 2
    assert cli_finish_fields["cli_fixed_boundary_finish_budgets"] == [20, 40]
    assert cli_finish_fields["cli_fixed_boundary_finish_fsq"] == [1.0e-4, 2.0e-8]
    assert cli_finish_fields["cli_fixed_boundary_finish_converged"] == [False, True]
    assert cli_finish_fields["cli_fixed_boundary_finish_modes"] == ["accelerated", "parity"]
    assert cli_finish_fields["cli_fixed_boundary_finish_best_fsq"] == 2.0e-8
    assert cli_finish_fields["cli_fixed_boundary_finish_budget_cap"] == 80
    assert cli_finish_fields["cli_fixed_boundary_finish_budget_exhausted"] is False
    assert cli_finish_fields["cli_fixed_boundary_full_parity_fallback"] is True
    assert cli_finish_fields["cli_fixed_boundary_partial_parity_fallback"] is False
    assert cli_finish_fields["cli_fixed_boundary_staged_followup_used"] is True
    assert module._csv_cell({"accepted": 2}) == '{"accepted": 2}'
    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(ntheta=32, nzeta=32)
    orientation_fit = module._orientation_fit_diagnostics(samples, samples)
    assert orientation_fit["max_orientation_fit_error"] == pytest.approx(0.0)
    assert orientation_fit["orientation_fit_valid_fraction"] > 0.8
    with pytest.raises(ValueError, match="unknown shape"):
        module._parse_shape_cases("unknown")


def test_toroidal_hybrid_fsq_history_plot_handles_offscale_direct_initial(tmp_path: Path):
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_convergence")
    path = module._write_fsq_history_plot(
        [
            {
                "case": "offscale",
                "direct_initial_fsq": 1.0e11,
                "fsq_history": [1.0e-1, 1.0e-2],
                "iter_history": [1, 2],
                "vmec2000_fsq_history": [1.0e-1, 1.0e-3],
                "vmec2000_iter_history": [1, 2],
            }
        ],
        outdir=tmp_path,
    )

    assert path is not None
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_toroidal_hybrid_axis_initialization_policy_tracks_solver_mode_and_env(monkeypatch):
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_convergence")

    monkeypatch.delenv("VMEC_JAX_ENABLE_AXIS_INFER", raising=False)
    monkeypatch.delenv("VMEC_JAX_DISABLE_AXIS_INFER", raising=False)
    assert module._vmec_jax_axis_initialization_policy("parity") == "raw_input_axis_or_zero"
    assert module._vmec_jax_axis_initialization_policy("accelerated") == "boundary_inferred_missing_axis"

    monkeypatch.setenv("VMEC_JAX_ENABLE_AXIS_INFER", "1")
    assert module._vmec_jax_axis_initialization_policy("parity") == "boundary_inferred_missing_axis"

    monkeypatch.setenv("VMEC_JAX_DISABLE_AXIS_INFER", "1")
    assert module._vmec_jax_axis_initialization_policy("accelerated") == "raw_input_axis_or_zero"


def test_toroidal_hybrid_initial_residual_comparison_ratios():
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_convergence")
    row = {
        "direct_initial_fsq": 3.0,
        "direct_initial_fsqr": 2.0,
        "direct_initial_fsqz": 6.0,
        "direct_initial_fsql": None,
        "initial_fsq": 2.0,
        "vmec2000_initial_fsq": 4.0,
        "vmec2000_initial_fsqr": 2.0,
        "vmec2000_initial_fsqz": 0.0,
        "vmec2000_initial_fsql": 5.0,
        "initial_fsqr": 1.0,
        "initial_fsqz": 3.0,
        "initial_fsql": None,
    }

    module._attach_initial_residual_comparison(row)

    assert row["direct_initial_fsq_ratio_vmec2000"] == 0.75
    assert row["direct_initial_fsqr_ratio_vmec2000"] == 1.0
    assert row["direct_initial_fsqz_ratio_vmec2000"] is None
    assert row["direct_initial_fsql_ratio_vmec2000"] is None
    assert row["initial_fsq_ratio_direct_initial"] == pytest.approx(2.0 / 3.0)
    assert row["vmec2000_initial_fsq_ratio_direct_initial"] == pytest.approx(4.0 / 3.0)
    assert row["initial_fsq_ratio_vmec2000"] == 0.5
    assert row["initial_fsqr_ratio_vmec2000"] == 0.5
    assert row["initial_fsqz_ratio_vmec2000"] is None
    assert row["initial_fsql_ratio_vmec2000"] is None


def test_toroidal_hybrid_direct_initial_residual_helper(monkeypatch, tmp_path: Path):
    module = import_module("examples.toroidal_stellarator_mirror_hybrid_convergence")
    calls = {}

    class DummyWout:
        fsqr = 1.0
        fsqz = 2.0
        fsql = 3.0

    def fake_run_fixed_boundary(path, **kwargs):
        calls["path"] = Path(path)
        calls["run_kwargs"] = kwargs
        return object()

    def fake_wout_from_fixed_boundary_run(run, **kwargs):
        calls["wout_run"] = run
        calls["wout_kwargs"] = kwargs
        return DummyWout()

    monkeypatch.setattr(module.vj, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(module.vj, "wout_from_fixed_boundary_run", fake_wout_from_fixed_boundary_run)

    fields = module._compute_direct_initial_residual(
        tmp_path / "input.case",
        solver_mode="parity",
        use_scan=False,
    )

    assert calls["path"] == tmp_path / "input.case"
    assert calls["run_kwargs"]["use_initial_guess"] is True
    assert calls["run_kwargs"]["solver"] == "vmec2000_iter"
    assert calls["run_kwargs"]["solver_mode"] == "parity"
    assert calls["run_kwargs"]["use_scan"] is False
    assert calls["wout_kwargs"] == {"include_fsq": True, "fast_bcovar": False}
    assert fields["direct_initial_residual_source"] == "vmec_jax_initial_guess_residual_scalars"
    assert fields["direct_initial_axis_initialization_policy"] == "raw_input_axis_or_zero"
    assert fields["direct_initial_fsq"] == 6.0
    assert fields["direct_initial_fsqr"] == 1.0
    assert fields["direct_initial_fsqz"] == 2.0
    assert fields["direct_initial_fsql"] == 3.0

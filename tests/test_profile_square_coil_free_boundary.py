from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from tools.diagnostics import profile_square_coil_free_boundary as profile


def test_square_coil_profile_residual_payload_keeps_solver_mode_and_history_tails():
    diagnostics = {
        "solver_mode": "parity",
        "use_scan": True,
        "performance_mode": False,
        "converged": False,
        "converged_strict": False,
        "requested_ftol": 1.0e-12,
        "final_fsqr": 1.0e-5,
        "final_fsqz": 2.0e-5,
        "final_fsql": 3.0e-6,
        "bad_resets": 0,
        "ijacob": 1,
        "free_boundary": {
            "nestor_model": "vmec2000_like_dense_integral",
            "couple_edge": True,
            "activate_fsq": 1.0e-3,
            "ivac": 3,
            "ivacskip": 0,
            "nvacskip": 2,
            "anderson_pressure_enabled": True,
            "last_nestor_diagnostics": {
                "bnormal_rms": 4.0e-4,
                "bsqvac_rms": 1.5e-2,
            },
        },
        "freeb_ivac_history": np.array([1, 2, 3]),
        "freeb_ivacskip_history": np.array([0, 1, 0]),
        "freeb_full_update_history": np.array([1, 0, 1]),
        "freeb_nestor_reused_history": np.array([0, 1, 0]),
        "freeb_nestor_bnormal_rms_history": np.array([1.0e-3, 7.0e-4, 4.0e-4]),
        "freeb_anderson_pressure_applied_history": np.array([0, 0, 1]),
        "freeb_anderson_pressure_theta_history": np.array([np.nan, np.nan, 0.4]),
        "freeb_anderson_pressure_residual_norm_history": np.array([1.0, 0.8, 0.6]),
        "include_edge_history": np.array([0, 1, 1]),
        "bad_jacobian_history": np.array([0, 0, 0]),
        "time_step_history": np.array([0.05, 0.05, 0.04]),
        "dt_eff_history": np.array([0.05, 0.05, 0.04]),
        "update_rms_history": np.array([1.0e-2, 4.0e-3, 2.0e-3]),
    }
    result = SimpleNamespace(
        n_iter=3,
        diagnostics=diagnostics,
        w_history=np.array([1.0, 0.5, 0.25, 0.125]),
        fsqr2_history=np.array([1.0e-3, 1.0e-4, 1.0e-5]),
        fsqz2_history=np.array([2.0e-3, 2.0e-4, 2.0e-5]),
        fsql2_history=np.array([3.0e-4, 3.0e-5, 3.0e-6]),
    )
    run = SimpleNamespace(result=result)

    payload = profile._final_residuals(run)

    assert payload["solver_mode"] == "parity"
    assert payload["use_scan"] is True
    assert payload["free_boundary_active"] is True
    assert payload["free_boundary_anderson_pressure_enabled"] is True
    assert payload["free_boundary_anderson_pressure_last_applied"] == pytest.approx(1.0)
    assert payload["free_boundary_anderson_pressure_last_theta"] == pytest.approx(0.4)
    assert payload["final_fsq_component_sum"] == pytest.approx(3.3e-5)
    assert payload["history"]["fsq_component_sum_tail"] == pytest.approx([0.0033, 0.00033, 3.3e-5])
    assert payload["history"]["fsq_component_sum_stats"]["min"] == pytest.approx(3.3e-5)
    assert payload["history"]["fsq_component_sum_tail_projection"]["per_iter_factor"] == pytest.approx(0.1)
    assert payload["history"]["fsq_component_sum_tail_projection"][
        "estimated_additional_iterations_to_target"
    ]["1e-12"] == 8
    assert payload["history"]["freeb_full_update_stats"]["sum"] == pytest.approx(2.0)
    assert payload["history"]["freeb_anderson_pressure_applied_stats"]["sum"] == pytest.approx(1.0)
    assert payload["history"]["freeb_anderson_pressure_theta_tail"][-1] == pytest.approx(0.4)
    assert payload["history"]["bad_jacobian_stats"]["nonzero_count"] == 0
    assert payload["history"]["time_step_stats"]["last"] == pytest.approx(0.04)
    assert payload["history"]["update_rms_stats"]["max"] == pytest.approx(1.0e-2)
    assert payload["history"]["freeb_ivac_tail"] == [1, 2, 3]
    assert payload["history"]["include_edge_tail"] == [0, 1, 1]


def test_square_coil_profile_tail_decay_projection_estimates_remaining_iterations():
    values = np.asarray([1.0e-6, 3.0e-7, 9.0e-8, 2.7e-8])

    projection = profile._tail_decay_projection(values, length=4, targets=(1.0e-8, 1.0e-12))

    assert projection["window"] == 4
    assert projection["monotone_decrease_fraction"] == pytest.approx(1.0)
    assert projection["per_iter_factor"] == pytest.approx(0.3)
    assert projection["estimated_additional_iterations_to_target"]["1e-08"] == 1
    assert projection["estimated_additional_iterations_to_target"]["1e-12"] == 9


def test_square_coil_profile_partial_vmec2000_payload_reads_timeout_rows(tmp_path: Path):
    workdir = tmp_path / "vmec2000_mgrid"
    workdir.mkdir()
    (workdir / "threed1.case").write_text(
        "\n".join(
            [
                " NS =    9 NO. FOURIER MODES =  113 FTOLV =  1.000E-08 NITER =   5000",
                " ITER    FSQR      FSQZ      FSQL      fsqr      fsqz      fsql      DELT    RAX(v=0)      WMHD      BETA      <M>   DEL-BSQ   FEDGE",
                "    1   4.00E-03  2.00E-03  1.00E-03  1.00E-04  2.00E-04  3.00E-04  5.00E-02  1.50E+00  3.00E-01  0.000E+00  1.0 1.00E-02 2.00E-03",
                " 'Plasma Boundary exceeded Vacuum Grid Size         '",
                "  200   4.00E-06  2.00E-06  1.00E-06  1.00E-08  2.00E-08  3.00E-08  5.00E-02  1.50E+00  3.00E-01  0.000E+00  1.0 1.00E-02 2.00E-06",
            ]
        )
        + "\n"
    )

    payload = profile._partial_vmec2000_payload(workdir)

    assert payload["iteration_row_count"] == 2
    assert payload["last_row"]["it"] == 200
    assert payload["last_row"]["total"] == pytest.approx(7.0e-6)
    assert payload["last_row"]["max_component"] == pytest.approx(4.0e-6)
    assert payload["min_total"] == pytest.approx(7.0e-6)
    assert payload["final_max_component"] == pytest.approx(4.0e-6)
    assert payload["strict_components_met"] is False
    assert payload["stage_summaries"][0]["strict_components_met"] is False
    assert payload["vacuum_grid_exceeded_count"] == 1


def test_square_coil_profile_writes_partial_vmec2000_sidecar(tmp_path: Path):
    workdir = tmp_path / "vmec2000_mgrid"
    workdir.mkdir()
    (workdir / "threed1.case").write_text(
        "\n".join(
            [
                " NS =    9 NO. FOURIER MODES =  113 FTOLV =  1.000E-12 NITER =   5000",
                " ITER    FSQR      FSQZ      FSQL      fsqr      fsqz      fsql      DELT",
                "  200   4.00E-11  2.00E-11  1.00E-11  1.00E-12  2.00E-12  3.00E-12  2.00E-02",
            ]
        )
        + "\n"
    )

    path = profile._write_partial_vmec2000_payload(outdir=tmp_path, workdir=workdir)

    data = json.loads(path.read_text())
    assert path.name == "_partial_vmec2000_payload.json"
    assert data["iteration_row_count"] == 1
    assert data["updated_unix_s"] > 0.0
    assert data["final_max_component"] == pytest.approx(4.0e-11)
    assert data["strict_components_met"] is False


def test_square_coil_profile_provider_parity_payload_reports_field_and_bnormal_error(monkeypatch, tmp_path: Path):
    vac_direct = SimpleNamespace(
        bnormal=np.array([[2.0, 4.0], [6.0, 8.0]]),
        bnormal_unit=np.array([[0.2, 0.4], [0.6, 0.8]]),
        bu=np.array([[1.0, 2.0], [3.0, 4.0]]),
        bv=np.array([[1.5, 2.5], [3.5, 4.5]]),
        bsqvac=np.array([[10.0, 11.0], [12.0, 13.0]]),
    )
    vac_mgrid = SimpleNamespace(
        bnormal=vac_direct.bnormal + 0.5,
        bnormal_unit=vac_direct.bnormal_unit + 0.05,
        bu=vac_direct.bu + 0.1,
        bv=vac_direct.bv + 0.1,
        bsqvac=vac_direct.bsqvac + 0.2,
    )
    direct_sample = SimpleNamespace(
        R=np.array([[1.0, 1.1], [1.2, 1.3]]),
        Z=np.array([[-0.1, 0.0], [0.1, 0.2]]),
        br_mgrid=np.ones((2, 2)),
        bp_mgrid=2.0 * np.ones((2, 2)),
        bz_mgrid=3.0 * np.ones((2, 2)),
        vac_ext=vac_direct,
    )
    mgrid_sample = SimpleNamespace(
        R=direct_sample.R,
        Z=direct_sample.Z,
        br_mgrid=direct_sample.br_mgrid + 0.01,
        bp_mgrid=direct_sample.bp_mgrid + 0.02,
        bz_mgrid=direct_sample.bz_mgrid + 0.03,
        vac_ext=vac_mgrid,
    )

    monkeypatch.setattr(
        profile,
        "run_free_boundary",
        lambda *args, **kwargs: SimpleNamespace(state=object(), static=object()),
    )
    monkeypatch.setattr(profile, "build_coil_field_geometry", lambda _params: "geometry")

    def fake_sample(**kwargs):
        return direct_sample if kwargs.get("external_field_provider_kind") == "direct_coils" else mgrid_sample

    monkeypatch.setattr(profile, "_sample_external_boundary_arrays", fake_sample)
    config = SimpleNamespace(nzeta=2)
    coil_params = SimpleNamespace(regularization_epsilon=0.0, chunk_size=None)

    payload = profile._provider_parity_payload(
        mgrid_input=tmp_path / "input.square_mgrid",
        coil_params=coil_params,
        config=config,
        bounds={"rmin": 0.8, "rmax": 1.5, "zmin": -0.3, "zmax": 0.4},
        mgrid_nphi=4,
    )

    assert payload["status"] == "completed"
    assert payload["mgrid_kp_divisible_by_nzeta"] is True
    assert payload["domain"]["contained"] is True
    assert payload["components"]["br_mgrid"]["diff_rms"] == pytest.approx(0.01)
    assert payload["field_vector"]["diff_rms"] == pytest.approx(np.sqrt(0.01**2 + 0.02**2 + 0.03**2))
    assert payload["vacuum_channels"]["bnormal"]["diff_rms"] == pytest.approx(0.5)


def test_square_coil_profile_rejects_mgrid_nphi_not_multiple_of_nzeta(tmp_path: Path):
    with pytest.raises(ValueError, match="mgrid-nphi=.*incompatible.*nzeta"):
        profile.main(
            [
                "--outdir",
                str(tmp_path / "profile"),
                "--nzeta",
                "4",
                "--mgrid-nphi",
                "6",
                "--skip-direct",
                "--skip-mgrid",
                "--skip-provider-parity",
            ]
        )


def test_square_coil_profile_records_boundary_projection_payload(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(profile, "write_mgrid_from_coils", lambda *args, **kwargs: None)

    outdir = tmp_path / "profile"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "3",
            "--ntor",
            "4",
            "--ns",
            "5",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--skip-direct",
            "--skip-mgrid",
            "--skip-provider-parity",
        ]
    )

    report = outdir / "square_coil_free_boundary_backend_profile.json"
    data = json.loads(report.read_text())
    projection = data["boundary_projection"]
    assert projection["mpol"] == 3
    assert projection["ntor"] == 4
    assert projection["recommended_nzeta"] == 16
    assert projection["mode_count"] > 0
    assert np.isfinite(float(projection["max_abs_error"]))


def test_square_coil_profile_defaults_nzeta_to_square_axis_recommendation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(profile, "write_mgrid_from_coils", lambda *args, **kwargs: None)

    outdir = tmp_path / "profile_auto_nzeta"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "5",
            "--ntor",
            "12",
            "--ns",
            "5",
            "--max-iter",
            "2",
            "--skip-direct",
            "--skip-mgrid",
            "--skip-provider-parity",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert data["configuration"]["nzeta"] == profile.recommended_square_axis_nzeta(12)
    assert data["configuration"]["nzeta_auto"] is True
    assert data["configuration"]["nzeta_underrecommended"] is False


def test_square_coil_profile_records_direct_coil_chunk_size(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(profile, "write_mgrid_from_coils", lambda *args, **kwargs: None)

    outdir = tmp_path / "profile_jit_sampler"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "3",
            "--ntor",
            "4",
            "--ns",
            "5",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--coil-chunk-size",
            "0",
            "--skip-direct",
            "--skip-mgrid",
            "--skip-provider-parity",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert data["configuration"]["coil_chunk_size"] is None
    assert data["mgrid"]["created"] is False
    assert data["mgrid"]["write_chunk_size"] is None


def test_square_coil_profile_direct_only_skips_mgrid_write(monkeypatch, tmp_path: Path):
    write_calls = []

    def fake_write_mgrid(*args, **kwargs):
        write_calls.append((args, kwargs))

    monkeypatch.setattr(profile, "write_mgrid_from_coils", fake_write_mgrid)

    outdir = tmp_path / "profile_direct_only"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "3",
            "--ntor",
            "4",
            "--ns",
            "5",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--coil-chunk-size",
            "0",
            "--skip-direct",
            "--skip-mgrid",
            "--skip-provider-parity",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert write_calls == []
    assert data["mgrid"]["created"] is False


def test_square_coil_profile_chunks_mgrid_when_direct_sampler_is_jit(monkeypatch, tmp_path: Path):
    captured = {}
    monkeypatch.setattr(profile, "_run_jax_backend", lambda **kwargs: {"status": "skipped_test"})

    def fake_write_mgrid(*args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(profile, "write_mgrid_from_coils", fake_write_mgrid)

    outdir = tmp_path / "profile_mgrid_chunked"
    profile.main(
        [
            "--outdir",
            str(outdir),
            "--mpol",
            "3",
            "--ntor",
            "4",
            "--ns",
            "5",
            "--nzeta",
            "16",
            "--max-iter",
            "2",
            "--coil-chunk-size",
            "0",
            "--skip-direct",
            "--skip-provider-parity",
        ]
    )

    data = json.loads((outdir / "square_coil_free_boundary_backend_profile.json").read_text())
    assert captured["kwargs"]["chunk_size"] == 512
    assert data["mgrid"]["created"] is True
    assert data["mgrid"]["write_chunk_size"] == 512

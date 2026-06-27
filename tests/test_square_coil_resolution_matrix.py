from __future__ import annotations

import json
from pathlib import Path

from tools.diagnostics import square_coil_resolution_matrix as matrix
from vmec_jax.toroidal_hybrid import recommended_square_axis_ntheta, recommended_square_axis_nzeta


def test_square_coil_resolution_matrix_classifies_small_decks(tmp_path: Path):
    args = matrix._parser().parse_args(
        [
            "--decks",
            "3:4:16,3:4:8",
            "--target-error",
            "none",
            "--outdir-root",
            str(tmp_path),
        ]
    )

    rows = matrix.build_rows(args)

    assert len(rows) == 2
    assert rows[0]["mpol"] == 3
    assert rows[0]["ntor"] == 4
    assert rows[0]["ntheta"] == recommended_square_axis_ntheta(3)
    assert rows[0]["nzeta"] == 16
    assert rows[0]["status"] == "diagnostic_gate_disabled"
    assert rows[0]["reasons"] == ["projection_gate_disabled"]
    assert rows[0]["recommended_nzeta"] == recommended_square_axis_nzeta(4)
    assert rows[0]["strict_schedule_status"] == "strict_ready"
    assert rows[0]["edge_control_projection"] == "stellarator"
    assert rows[0]["strict_solver_native_spline_status"] == "edge_only_bridge"
    assert rows[0]["strict_solver_native_spline_edge_controls"] is True
    assert rows[0]["strict_full_native_spline_state_required"] is True
    assert rows[1]["status"] == "diagnostic_underresolved"
    assert "nzeta_below_square_axis_recommendation" in rows[1]["reasons"]


def test_square_coil_resolution_matrix_auto_nzeta_and_commands(tmp_path: Path):
    args = matrix._parser().parse_args(
        [
            "--decks",
            "5:28:auto,5:28:64:96",
            "--target-error",
            "5e-12",
            "--outdir-root",
            str(tmp_path),
            "--print-preflight-commands",
            "--print-scale-commands",
            "--print-vmec2000-commands",
            "--print-jax-commands",
            "--include-control-map",
            "--edge-control-ridge",
            "1e-10",
            "--edge-control-trust-radius",
            "1e-7",
            "--vmec2000-exec",
            "/opt/xvmec",
        ]
    )

    rows = matrix.build_rows(args)

    assert rows[0]["nzeta"] == max(64, recommended_square_axis_nzeta(28))
    assert rows[0]["ntheta"] == recommended_square_axis_ntheta(5)
    assert rows[0]["mgrid_nphi"] == rows[0]["nzeta"]
    assert rows[0]["status"] == "production_ready"
    assert rows[0]["control_map_status"] == "available"
    assert rows[0]["control_map_square_count"] == 3
    assert rows[0]["control_map_square_condition"] is not None
    assert rows[0]["control_map_stellarator_count"] == 9
    assert rows[0]["control_map_stellarator_condition"] is not None
    assert "--resolution-diagnostics-only" in rows[0]["preflight_command"]
    assert "--scale-diagnostics-only" in rows[0]["scale_command"]
    assert "--ntheta 64" in rows[0]["scale_command"]
    assert "--run-vmec2000" in rows[0]["vmec2000_command"]
    assert "--vmec2000-exec /opt/xvmec" in rows[0]["vmec2000_command"]
    assert "--freeb-edge-control-projection stellarator" in rows[0]["jax_command"]
    assert "--freeb-edge-control-update-mode native_coordinate" in rows[0]["jax_command"]
    assert "--freeb-edge-control-native-force-metric least_squares" in rows[0]["jax_command"]
    assert "--freeb-edge-control-ridge 1e-10" in rows[0]["jax_command"]
    assert "--freeb-edge-control-trust-radius 1e-07" in rows[0]["jax_command"]
    assert "--strict-backtracking-accept-ratio 1" in rows[0]["jax_command"]
    assert "square_coil_resolution_mpol5_ntor28_nzeta64" in rows[0]["preflight_command"]
    assert rows[1]["status"] == "diagnostic_underresolved"
    assert "mgrid_nphi_not_multiple_of_nzeta" in rows[1]["reasons"]


def test_square_coil_resolution_matrix_reports_effective_auto_bumped_deck(tmp_path: Path):
    args = matrix._parser().parse_args(
        [
            "--decks",
            "5:28:32",
            "--target-error",
            "5e-12",
            "--outdir-root",
            str(tmp_path),
            "--print-preflight-commands",
        ]
    )

    rows = matrix.build_rows(args)

    assert len(rows) == 1
    row = rows[0]
    assert row["requested_mpol"] == 5
    assert row["requested_ntor"] == 28
    assert row["requested_nzeta"] == 32
    assert row["requested_status"] == "diagnostic_underresolved"
    assert "nzeta_below_square_axis_recommendation" in row["requested_reasons"]
    assert row["mpol"] == 5
    assert row["ntor"] == 28
    assert row["nzeta"] == recommended_square_axis_nzeta(28)
    assert row["mgrid_nphi"] == row["nzeta"]
    assert row["nzeta_auto_bumped_to_recommended"] is True
    assert row["status"] == "production_ready"
    assert "--nzeta 64" in row["preflight_command"]


def test_square_coil_resolution_matrix_main_json(capsys, tmp_path: Path):
    rc = matrix.main(
        [
            "--decks",
            "3:4:16",
            "--target-error",
            "none",
            "--format",
            "json",
            "--outdir-root",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["mpol"] == 3
    assert payload[0]["ntor"] == 4
    assert payload[0]["status"] == "diagnostic_gate_disabled"

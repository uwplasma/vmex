from __future__ import annotations

from pathlib import Path

import pytest

from tools.diagnostics import square_coil_followup_commands as followup
from vmec_jax.toroidal_hybrid import recommended_square_axis_ntheta, recommended_square_axis_nzeta


def test_square_coil_followup_commands_emit_strict_vmec2000_scan(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.015,0.02",
            "--vmec2000-exec",
            "/opt/xvmec",
        ]
    )

    commands = followup.build_commands(args)

    assert len(commands) == 2
    first = commands[0]
    assert first[:2] == ["python3", "tools/diagnostics/profile_square_coil_free_boundary.py"]
    assert first[first.index("--ftol") + 1] == "1e-12"
    assert first[first.index("--ftol-array") + 1] == "1e-8,1e-10,1e-12"
    assert first[first.index("--niter-array") + 1] == "8000,16000,32000"
    assert first[first.index("--max-iter") + 1] == "32000"
    assert first[first.index("--axis-kind") + 1] == "control_spline"
    assert first[first.index("--axis-spline-control-count") + 1] == "16"
    assert first[first.index("--delt") + 1] == "0.015"
    assert first[first.index("--vmec2000-exec") + 1] == "/opt/xvmec"
    assert "--skip-direct" in first
    assert "--skip-mgrid" in first
    assert "--skip-provider-parity" in first
    assert "--accepted-provider-parity" not in first
    assert "--run-vmec2000" in first
    assert "--max-boundary-projection-error" in first
    assert "5e-12" in first

    outdir = Path(first[first.index("--outdir") + 1])
    assert outdir.parent == tmp_path
    assert "delt0p015" in outdir.name
    assert "niter32k" in outdir.name
    assert "control_spline" in outdir.name
    assert "axisctrl16" in outdir.name
    assert "vmec2000" in outdir.name


def test_square_coil_followup_commands_emit_accepted_provider_parity_scan(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "provider-parity",
        ]
    )

    command = followup.build_commands(args)[0]

    assert "--accepted-provider-parity" in command
    assert "--return-best-scored-state" in command
    assert "--skip-direct" not in command
    assert "--skip-mgrid" not in command
    assert "--skip-provider-parity" not in command
    assert "--run-vmec2000" not in command
    assert command[command.index("--coil-chunk-size") + 1] == "512"
    outdir = Path(command[command.index("--outdir") + 1])
    assert "provider_parity" in outdir.name


def test_square_coil_followup_commands_emit_projected_delta_for_non_polish_edge_projection(
    tmp_path: Path,
):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "provider-parity",
            "--freeb-edge-control-projection",
            "square",
        ]
    )

    command = followup.build_commands(args)[0]

    assert command[command.index("--freeb-edge-control-projection") + 1] == "square"
    assert command[command.index("--freeb-edge-control-update-mode") + 1] == "projected_delta"
    outdir = Path(command[command.index("--outdir") + 1])
    assert "edge_square_projected_delta" in outdir.name


def test_square_coil_followup_commands_emit_resolution_preflight(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "resolution-preflight",
            "--ntor",
            "31",
        ]
    )

    command = followup.build_commands(args)[0]

    expected_nzeta = max(64, recommended_square_axis_nzeta(31))
    assert "--resolution-diagnostics-only" in command
    assert command[command.index("--nzeta") + 1] == str(expected_nzeta)
    assert command[command.index("--mgrid-nphi") + 1] == str(expected_nzeta)
    assert "--run-vmec2000" not in command
    assert "--accepted-provider-parity" not in command
    assert "--jit-forces" not in command
    outdir = Path(command[command.index("--outdir") + 1])
    assert "resolution_preflight" in outdir.name


def test_square_coil_followup_commands_emit_full_backend_scan(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "full-backend",
            "--vmec2000-exec",
            "/opt/xvmec",
        ]
    )

    command = followup.build_commands(args)[0]

    assert "--accepted-provider-parity" in command
    assert "--run-vmec2000" in command
    assert command[command.index("--vmec2000-exec") + 1] == "/opt/xvmec"
    outdir = Path(command[command.index("--outdir") + 1])
    assert "full_backend" in outdir.name


def test_square_coil_followup_commands_emit_direct_gpu_speed_probe(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "direct-gpu",
            "--freeb-anderson-pressure",
        ]
    )

    command = followup.build_commands(args)[0]

    assert "--skip-mgrid" in command
    assert "--skip-provider-parity" in command
    assert "--jit-forces" in command
    assert "--jit-direct-sampler" in command
    assert "--verbose-solver" in command
    assert "--freeb-anderson-pressure" in command
    assert "--accepted-provider-parity" not in command
    assert "--run-vmec2000" not in command
    assert command.count("--coil-chunk-size") == 1
    assert command[command.index("--coil-chunk-size") + 1] == "0"
    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu" in outdir.name


def test_square_coil_followup_commands_emit_direct_gpu_jax_nestor_probe(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "direct-gpu-jax-nestor",
            "--no-freeb-jax-nestor-jit-operator",
            "--freeb-include-edge",
            "--freeb-dense-solve-mode",
            "grid",
            "--no-freeb-experimental-fouri-matrix",
            "--freeb-add-analytic-bvec",
        ]
    )

    command = followup.build_commands(args)[0]

    assert "--skip-mgrid" in command
    assert "--skip-provider-parity" in command
    assert "--jit-forces" in command
    assert "--jit-direct-sampler" in command
    assert "--verbose-solver" in command
    assert "--freeb-jax-nestor-operator" in command
    assert "--no-freeb-jax-nestor-jit-operator" in command
    assert "--freeb-include-edge" in command
    assert command[command.index("--freeb-dense-solve-mode") + 1] == "grid"
    assert "--no-freeb-experimental-fouri-matrix" in command
    assert "--freeb-add-analytic-bvec" in command
    assert "--run-vmec2000" not in command
    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu_jax_nestor" in outdir.name


def test_square_coil_followup_commands_emit_direct_gpu_edge_polish(tmp_path: Path):
    seed = tmp_path / "seed_wout.nc"
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.015",
            "--profile-kind",
            "direct-gpu-edge-polish",
            "--jax-initial-restart-wout",
            str(seed),
        ]
    )

    command = followup.build_commands(args)[0]

    assert "--skip-mgrid" in command
    assert "--skip-provider-parity" in command
    assert "--jit-forces" in command
    assert "--jit-direct-sampler" in command
    assert "--return-best-scored-state" in command
    assert "--freeb-anderson-pressure" in command
    assert command[command.index("--freeb-edge-control-projection") + 1] == "square"
    assert command[command.index("--freeb-edge-control-rcond") + 1] == "1e-12"
    assert command[command.index("--freeb-edge-control-ridge") + 1] == "0"
    assert command[command.index("--freeb-edge-control-update-mode") + 1] == "coordinate"
    assert "--freeb-edge-control-trust-radius" not in command
    assert command[command.index("--jax-hot-restart-count") + 1] == "2"
    assert command[command.index("--jax-hot-restart-iters") + 1] == "32000"
    assert command[command.index("--jax-hot-restart-policy") + 1] == "freeb"
    assert command[command.index("--jax-initial-restart-wout") + 1] == str(seed)
    assert "--freeb-jax-nestor-operator" not in command
    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu_edge_polish" in outdir.name
    assert "edge_square_coordinate" in outdir.name


def test_square_coil_followup_commands_emit_scaled_edge_polish_probe(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "direct-gpu-edge-polish",
            "--mpol",
            "5",
            "--ntor",
            "28",
            "--ntheta",
            "64",
            "--nzeta",
            "64",
            "--phiedge",
            "-0.0025078856391256765",
            "--niter-array",
            "1000,2000,8000",
            "--jax-hot-restart-count",
            "1",
            "--jax-hot-restart-iters",
            "8000",
            "--freeb-edge-control-ridge",
            "1e-10",
            "--freeb-edge-control-trust-radius",
            "1e-7",
        ]
    )

    command = followup.build_commands(args)[0]

    assert command[command.index("--mpol") + 1] == "5"
    assert command[command.index("--ntor") + 1] == "28"
    assert command[command.index("--ntheta") + 1] == "64"
    assert command[command.index("--nzeta") + 1] == "64"
    assert command[command.index("--phiedge") + 1] == "-0.002507885639125676"
    assert command[command.index("--freeb-edge-control-projection") + 1] == "square"
    assert command[command.index("--freeb-edge-control-ridge") + 1] == "1e-10"
    assert command[command.index("--freeb-edge-control-trust-radius") + 1] == "1e-07"
    assert command[command.index("--freeb-edge-control-update-mode") + 1] == "coordinate"
    assert command[command.index("--jax-hot-restart-count") + 1] == "1"
    assert command[command.index("--jax-hot-restart-iters") + 1] == "8000"
    assert "--freeb-anderson-pressure" in command
    assert "--skip-mgrid" in command
    assert "--run-vmec2000" not in command

    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu_edge_polish" in outdir.name
    assert "ntheta64" in outdir.name
    assert "nzeta64" in outdir.name
    assert "phiedgem0p00250789" in outdir.name
    assert "edge_square_coordinate" in outdir.name
    assert "ridge1em10" in outdir.name
    assert "trust1em07" in outdir.name


def test_square_coil_followup_commands_emit_stellarator_edge_polish(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "direct-gpu-edge-stellarator-polish",
        ]
    )

    command = followup.build_commands(args)[0]

    assert command[command.index("--freeb-edge-control-projection") + 1] == "stellarator"
    assert command[command.index("--freeb-edge-control-update-mode") + 1] == "coordinate"
    assert command[command.index("--axis-spline-control-count") + 1] == "16"
    assert "--freeb-jax-nestor-operator" not in command
    assert "--freeb-anderson-pressure" in command
    assert "--skip-mgrid" in command
    assert "--run-vmec2000" not in command
    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu_edge_stellarator_polish" in outdir.name
    assert "edge_stellarator_coordinate" in outdir.name


def test_square_coil_followup_commands_emit_stellarator_native_edge_polish(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "direct-gpu-edge-stellarator-native-polish",
        ]
    )

    command = followup.build_commands(args)[0]

    assert command[command.index("--freeb-edge-control-projection") + 1] == "stellarator"
    assert command[command.index("--freeb-edge-control-update-mode") + 1] == "native_coordinate"
    assert "--freeb-anderson-pressure" in command
    assert "--jit-forces" in command
    assert "--skip-mgrid" in command
    assert "--run-vmec2000" not in command
    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu_edge_stellarator_native_polish" in outdir.name
    assert "edge_stellarator_native_coordinate" in outdir.name


def test_square_coil_followup_commands_emit_strict_backtracking_native_polish(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "direct-gpu-edge-stellarator-native-polish",
            "--strict-backtracking",
        ]
    )

    command = followup.build_commands(args)[0]

    assert "--strict-backtracking" in command
    assert command[command.index("--freeb-edge-control-projection") + 1] == "stellarator"
    assert command[command.index("--freeb-edge-control-update-mode") + 1] == "native_coordinate"
    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu_edge_stellarator_native_polish" in outdir.name
    assert "edge_stellarator_native_coordinate" in outdir.name
    assert outdir.name.endswith("_strictbt")


def test_square_coil_followup_commands_reject_underrecommended_ntheta(tmp_path: Path):
    recommended = recommended_square_axis_ntheta(5)
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "direct-gpu-edge-polish",
            "--mpol",
            "5",
            "--ntheta",
            str(recommended - 8),
        ]
    )

    with pytest.raises(ValueError, match="below the square-axis recommendation"):
        followup.build_commands(args)


def test_square_coil_followup_commands_emit_direct_gpu_edge_jax_nestor_polish(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.015",
            "--profile-kind",
            "direct-gpu-edge-jax-nestor-polish",
            "--freeb-edge-control-projection",
            "stellarator",
            "--jax-hot-restart-count",
            "1",
            "--jax-hot-restart-iters",
            "1234",
            "--jax-hot-restart-always",
        ]
    )

    command = followup.build_commands(args)[0]

    assert command[command.index("--freeb-edge-control-projection") + 1] == "stellarator"
    assert command[command.index("--freeb-edge-control-update-mode") + 1] == "coordinate"
    assert "--freeb-jax-nestor-operator" in command
    assert command[command.index("--jax-hot-restart-count") + 1] == "1"
    assert command[command.index("--jax-hot-restart-iters") + 1] == "1234"
    assert "--jax-hot-restart-always" in command
    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu_edge_jax_nestor_polish" in outdir.name
    assert "edge_stellarator_coordinate" in outdir.name


def test_square_coil_followup_commands_emit_stellarator_edge_jax_nestor_polish(
    tmp_path: Path,
):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.015",
            "--profile-kind",
            "direct-gpu-edge-stellarator-jax-nestor-polish",
        ]
    )

    command = followup.build_commands(args)[0]

    assert command[command.index("--freeb-edge-control-projection") + 1] == "stellarator"
    assert command[command.index("--freeb-edge-control-update-mode") + 1] == "coordinate"
    assert "--freeb-jax-nestor-operator" in command
    assert "--freeb-anderson-pressure" in command
    assert "--skip-mgrid" in command
    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu_edge_stellarator_jax_nestor_polish" in outdir.name
    assert "edge_stellarator_coordinate" in outdir.name


def test_square_coil_followup_commands_emit_native_spline_control_prototype(tmp_path: Path):
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.02",
            "--profile-kind",
            "native-spline-control-prototype",
        ]
    )

    command = followup.build_commands(args)[0]

    assert "--native-spline-control-prototype" in command
    assert "--resolution-diagnostics-only" in command
    assert command[command.index("--freeb-edge-control-projection") + 1] == "stellarator"
    assert command[command.index("--freeb-edge-control-update-mode") + 1] == "coordinate"
    assert "--run-vmec2000" not in command
    assert "--jit-forces" not in command
    outdir = Path(command[command.index("--outdir") + 1])
    assert "native_spline_control_prototype" in outdir.name
    assert "edge_stellarator_coordinate" in outdir.name


def test_square_coil_followup_commands_default_nzeta_tracks_ntor(tmp_path: Path):
    ntor = 31
    args = followup._parser().parse_args(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.025",
            "--ntor",
            str(ntor),
            "--niter-array",
            "7,11,12345",
        ]
    )

    command = followup.build_commands(args)[0]

    expected_nzeta = max(64, recommended_square_axis_nzeta(ntor))
    assert command[command.index("--nzeta") + 1] == str(expected_nzeta)
    assert command[command.index("--mgrid-nphi") + 1] == str(expected_nzeta)
    assert command[command.index("--max-iter") + 1] == "12345"
    outdir = Path(command[command.index("--outdir") + 1])
    assert f"ntor{ntor}" in outdir.name
    assert f"nzeta{expected_nzeta}" in outdir.name
    assert "niter12345" in outdir.name


def test_square_coil_followup_commands_main_prints_shell_lines(capsys, tmp_path: Path):
    rc = followup.main(
        [
            "--outdir-root",
            str(tmp_path),
            "--delt-values",
            "0.015",
            "--python",
            "python",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("python tools/diagnostics/profile_square_coil_free_boundary.py ")
    assert "--run-vmec2000" in out
    assert "--ftol 1e-12" in out

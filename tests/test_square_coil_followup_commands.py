from __future__ import annotations

from pathlib import Path

from tools.diagnostics import square_coil_followup_commands as followup
from vmec_jax.toroidal_hybrid import recommended_square_axis_nzeta


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
    assert "--freeb-anderson-pressure" in command
    assert "--accepted-provider-parity" not in command
    assert "--run-vmec2000" not in command
    assert command.count("--coil-chunk-size") == 1
    assert command[command.index("--coil-chunk-size") + 1] == "0"
    outdir = Path(command[command.index("--outdir") + 1])
    assert "direct_gpu" in outdir.name


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

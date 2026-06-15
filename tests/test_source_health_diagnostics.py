from __future__ import annotations

from pathlib import Path

from tools.diagnostics import source_health


def test_source_health_collects_and_sorts_python_files(tmp_path: Path) -> None:
    small = tmp_path / "small.py"
    large = tmp_path / "large.py"
    ignored = tmp_path / "notes.txt"
    small.write_text("x = 1\n", encoding="utf-8")
    large.write_text("x = 1\n" * 3, encoding="utf-8")
    ignored.write_text("not python\n" * 10, encoding="utf-8")

    stats = source_health.collect_source_stats([tmp_path])

    assert [(item.path.name, item.lines) for item in stats] == [("large.py", 3), ("small.py", 1)]


def test_source_health_report_marks_warning_threshold(tmp_path: Path) -> None:
    source = tmp_path / "large.py"
    source.write_text("x = 1\n" * 3, encoding="utf-8")

    report = source_health.format_source_health_report(
        [source_health.SourceFileStat(path=source, lines=3)],
        top=1,
        warn_lines=2,
    )

    assert "WARN" in report
    assert "large.py" in report
    assert "3" in report


def test_source_health_fail_lines_is_opt_in(tmp_path: Path, capsys) -> None:
    source = tmp_path / "large.py"
    source.write_text("x = 1\n" * 3, encoding="utf-8")

    assert source_health.main([str(tmp_path), "--fail-lines", "0"]) == 0
    assert source_health.main([str(tmp_path), "--fail-lines", "3"]) == 1
    assert "large.py" in capsys.readouterr().out


def test_source_health_root_namespace_counts_helper_prefixes(tmp_path: Path) -> None:
    package = tmp_path / "vmec_jax"
    package.mkdir()
    (package / "solve_scan.py").write_text("", encoding="utf-8")
    (package / "driver_api.py").write_text("", encoding="utf-8")
    (package / "field.py").write_text("", encoding="utf-8")
    (package / "notes.txt").write_text("", encoding="utf-8")

    stat = source_health.collect_root_namespace_stat(package)

    assert stat.python_files == 3
    assert [path.name for path in stat.helper_prefix_files] == ["driver_api.py", "solve_scan.py"]
    report = source_health.format_root_namespace_report(stat, max_helper_prefix_files=2)
    assert "helper-prefix files: 2" in report
    assert "solve_scan.py" in report


def test_source_health_root_helper_prefix_gate_is_baseline_aware(tmp_path: Path) -> None:
    package = tmp_path / "vmec_jax"
    package.mkdir()
    (package / "solve_scan.py").write_text("", encoding="utf-8")
    (package / "solve_policy.py").write_text("", encoding="utf-8")
    source = tmp_path / "small.py"
    source.write_text("x = 1\n", encoding="utf-8")

    common_args = [
        str(tmp_path),
        "--root-namespace",
        str(package),
        "--root-helper-prefix",
        "solve_",
    ]
    assert source_health.main([*common_args, "--max-root-helper-prefix-files", "2"]) == 0
    assert source_health.main([*common_args, "--max-root-helper-prefix-files", "1"]) == 1

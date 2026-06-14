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

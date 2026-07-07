from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.diagnostics.qi.qi_landscape_scan import (
    METRICS,
    ScanAxis,
    choose_default_dofs,
    parse_dofs,
    plot_report,
    resolve_input_path,
    scan_landscape_records,
)
from vmec_jax.optimization import BoundaryParamSpec


def _spec(name: str, index: int) -> BoundaryParamSpec:
    return BoundaryParamSpec(name=name, kind="rc", index=index, m=index, n=index)


def test_parse_dofs_and_resolve_optimization_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage_01_mode03"
    stage_dir.mkdir(parents=True)
    final_input = run_dir / "input.final"
    stage_input = stage_dir / "input.final"
    final_input.write_text("&INDATA\n/")
    stage_input.write_text("&INDATA\n/")

    assert parse_dofs("rc11,zs11") == ("rc11", "zs11")
    assert resolve_input_path(run_dir) == final_input
    with pytest.raises(argparse.ArgumentTypeError, match="one or two"):
        parse_dofs("rc11,zs11,rc22")


def test_choose_default_dofs_prefers_largest_non_axis_base_coefficients() -> None:
    specs = [_spec("rc00", 0), _spec("rc11", 1), _spec("zs11", 2), _spec("rc22", 3)]
    optimizer = SimpleNamespace(_base_params_vector=lambda: np.asarray([10.0, 0.1, -0.4, 0.2]))
    stage = SimpleNamespace(specs=specs, optimizer=optimizer)

    assert choose_default_dofs(stage, count=2) == ("zs11", "rc22")


def test_scan_landscape_records_injects_boundary_increments_and_metrics() -> None:
    specs = [_spec("rc11", 0), _spec("zs11", 1)]
    seen = []

    def evaluate(params):
        seen.append(np.asarray(params, dtype=float))
        return {
            "qi_smooth_total": params[0] ** 2 + params[1] ** 2,
            "qi_mirror_ratio_max": 0.2 + params[0],
            "qi_max_elongation": 7.0 + params[1],
            "aspect": 5.0,
            "mean_iota": -0.45,
        }

    records = scan_landscape_records(
        axes=[
            ScanAxis("rc11", (-0.1, 0.1)),
            ScanAxis("zs11", (-0.2, 0.2)),
        ],
        specs=specs,
        evaluate=evaluate,
    )

    assert len(records) == 4
    np.testing.assert_allclose(seen[0], [-0.1, -0.2])
    np.testing.assert_allclose(seen[-1], [0.1, 0.2])
    assert records[0]["deltas"] == {"rc11": -0.1, "zs11": -0.2}
    assert records[0]["metrics"]["qi_smooth_total"] == pytest.approx(0.05)


def test_scan_landscape_plot_uses_contour_lines_for_2d(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    contour_calls = []
    original_contour = Axes.contour

    def spy_contour(self, *args, **kwargs):
        contour_calls.append((args, kwargs))
        return original_contour(self, *args, **kwargs)

    def fail_contourf(self, *args, **kwargs):
        raise AssertionError("qi_landscape_scan must not use contourf")

    monkeypatch.setattr(Axes, "contour", spy_contour)
    monkeypatch.setattr(Axes, "contourf", fail_contourf)

    axis0 = [-0.1, 0.0, 0.1]
    axis1 = [-0.2, 0.0, 0.2]
    records = []
    for x in axis0:
        for y in axis1:
            metrics = {
                key: float((idx + 1) + x + 2.0 * y + 0.1 * idx * x * y)
                for idx, (key, _label) in enumerate(METRICS)
            }
            records.append({"deltas": {"rc11": x, "zs11": y}, "metrics": metrics})
    report = {
        "dimension": 2,
        "dofs": ["rc11", "zs11"],
        "axes": [{"dof": "rc11", "values": axis0}, {"dof": "zs11", "values": axis1}],
        "records": records,
    }

    out = tmp_path / "landscape.png"
    plot_report(report, out)

    assert out.exists()
    assert len(contour_calls) == len(METRICS)

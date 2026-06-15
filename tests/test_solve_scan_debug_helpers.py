from pathlib import Path

import numpy as np
import pytest

from vmec_jax.solve_diagnostics_io import _format_vmec2000_iter_row
from vmec_jax.solvers.fixed_boundary.scan.debug import (
    _append_timecontrol_scan_trace_row,
    _axis_guess_lines,
    _emit_vmec2000_iter_row,
    _emit_scan_prints,
    _print_axis_guess,
    _print_vmec2000_row,
    _record_scan_device_ready,
    _timecontrol_scan_stage_name,
)


def test_print_vmec2000_row_respects_controls_and_formats_lasym(capsys):
    printed = _print_vmec2000_row(
        iter_idx=4,
        fsqr=1.2e-3,
        fsqz=2.3e-4,
        fsql=3.4e-5,
        delt0r=4.5e-2,
        r00=1.23456,
        z00=-0.125,
        w_mhd=6.7,
        lasym=True,
        verbose=False,
    )
    assert not printed
    assert capsys.readouterr().out == ""

    printed = _print_vmec2000_row(
        iter_idx=4,
        fsqr=1.2e-3,
        fsqz=2.3e-4,
        fsql=3.4e-5,
        delt0r=4.5e-2,
        r00=1.23456,
        z00=-0.125,
        w_mhd=6.7,
        lasym=True,
    )

    assert printed
    assert capsys.readouterr().out == (
        _format_vmec2000_iter_row(
            iter_idx=4,
            fsqr=1.2e-3,
            fsqz=2.3e-4,
            fsql=3.4e-5,
            delt0r=4.5e-2,
            r00=1.23456,
            z00=-0.125,
            w_mhd=6.7,
            lasym=True,
        )
        + "\n"
    )


class _FakeJaxDebug:
    def __init__(self):
        self.print_calls = []
        self.callback_calls = []

    def print(self, fmt, **kwargs):
        self.print_calls.append((fmt, kwargs))

    def callback(self, callback, *args, **kwargs):
        self.callback_calls.append((callback, args, kwargs))
        callback(*args)


def test_emit_vmec2000_iter_row_uses_plain_print_fallback_and_controls():
    rows = []

    assert not _emit_vmec2000_iter_row(
        iter_idx=1,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=0.1,
        r00=1.2,
        w_mhd=4.0,
        lasym=False,
        verbose=False,
        print_row=lambda **kwargs: rows.append(kwargs),
    )
    assert rows == []

    assert _emit_vmec2000_iter_row(
        iter_idx=2,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=0.1,
        r00=1.2,
        w_mhd=4.0,
        lasym=True,
        z00=None,
        print_row=lambda **kwargs: rows.append(kwargs),
    )
    assert rows[0]["lasym"] is True
    assert np.isnan(rows[0]["z00"])


@pytest.mark.parametrize("lasym", [False, True])
def test_emit_vmec2000_iter_row_uses_jax_debug_print(lasym):
    debug = _FakeJaxDebug()

    assert _emit_vmec2000_iter_row(
        iter_idx=3,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=0.1,
        r00=1.2,
        z00=0.25,
        w_mhd=4.0,
        lasym=lasym,
        scan_print_mode="debug_print",
        scan_print_ordered=True,
        jax_debug=debug,
    )
    assert len(debug.print_calls) == 1
    fmt, kwargs = debug.print_calls[0]
    assert "z00" in fmt if lasym else "z00" not in fmt
    assert kwargs["ordered"] is True
    assert kwargs["i"] == 3


@pytest.mark.parametrize("lasym", [False, True])
def test_emit_vmec2000_iter_row_uses_jax_debug_callback(lasym):
    debug = _FakeJaxDebug()
    rows = []

    assert _emit_vmec2000_iter_row(
        iter_idx=4,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=0.1,
        r00=1.2,
        z00=-0.25,
        w_mhd=4.0,
        lasym=lasym,
        scan_print_mode="debug_callback",
        scan_print_ordered=True,
        jax_debug=debug,
        print_row=lambda **kwargs: rows.append(kwargs),
    )
    assert len(debug.callback_calls) == 1
    assert debug.callback_calls[0][2]["ordered"] is True
    assert rows[0]["lasym"] is lasym
    assert rows[0]["iter_idx"] == 4
    if lasym:
        assert rows[0]["z00"] == pytest.approx(-0.25)
    else:
        assert "z00" not in rows[0]


@pytest.mark.parametrize("lasym", [False, True])
def test_emit_vmec2000_iter_row_uses_io_callback(lasym):
    debug = _FakeJaxDebug()
    rows = []
    io_calls = []

    def fake_io_callback(callback, result_shape_dtypes, *args, **kwargs):
        io_calls.append((result_shape_dtypes, args, kwargs))
        return callback(*args)

    assert _emit_vmec2000_iter_row(
        iter_idx=5,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=0.1,
        r00=1.2,
        z00=0.75,
        w_mhd=4.0,
        lasym=lasym,
        scan_print_mode="io_callback",
        jax_debug=debug,
        io_callback=fake_io_callback,
        print_row=lambda **kwargs: rows.append(kwargs),
    )
    assert len(io_calls) == 1
    assert rows[0]["lasym"] is lasym
    if lasym:
        assert rows[0]["z00"] == pytest.approx(0.75)
    else:
        assert "z00" not in rows[0]


def test_emit_vmec2000_iter_row_missing_io_callback_returns_false():
    assert not _emit_vmec2000_iter_row(
        iter_idx=6,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=0.1,
        r00=1.2,
        w_mhd=4.0,
        lasym=False,
        scan_print_mode="io_callback",
        jax_debug=_FakeJaxDebug(),
        io_callback=None,
    )


def test_print_axis_guess_matches_vmec_block_and_swallows_bad_inputs(capsys):
    raxis = np.asarray([1.0, -2.5e-3])
    zaxis = np.asarray([[0.0, 3.25]])

    assert _axis_guess_lines(raxis, zaxis) == (
        "  ---- Improved AXIS Guess ----",
        "      RAXIS_CC =    1   -0.0025",
        "      ZAXIS_CS =    0   3.25",
        "  -----------------------------",
    )
    assert _print_axis_guess(raxis, zaxis)
    assert capsys.readouterr().out.splitlines() == list(_axis_guess_lines(raxis, zaxis))

    class BadArray:
        def __array__(self, dtype=None):
            raise RuntimeError("nope")

    assert not _print_axis_guess(BadArray(), zaxis)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("stage_id", "expected"),
    [(0, "init"), (1, "pre"), (2, "checkpoint"), (3, "restart"), (99, "pre")],
)
def test_timecontrol_scan_stage_name(stage_id, expected):
    assert _timecontrol_scan_stage_name(stage_id) == expected


def test_append_timecontrol_scan_trace_row_writes_and_swallows_failures(tmp_path):
    path = tmp_path / "trace.log"

    assert _append_timecontrol_scan_trace_row(
        path,
        stage_id=2,
        iter2=7,
        iter1=3,
        fsq=1.25,
        fsq0=2.5,
        res0=-1.0,
        res1=4.0,
        time_step=0.125,
        irst=2,
    )
    text = path.read_text(encoding="utf-8")
    assert "checkpoint" in text
    assert "1.2500000000000000e+00" in text
    assert "1.2500000000000000e-01" in text

    assert not _append_timecontrol_scan_trace_row(
        tmp_path / "missing" / "trace.log",
        stage_id=1,
        iter2=1,
        iter1=1,
        fsq=1.0,
        fsq0=1.0,
        res0=1.0,
        res1=1.0,
        time_step=1.0,
        irst=0,
    )


def test_emit_scan_prints_light_history_uses_cadence_or_convergence():
    hist_np = (
        np.asarray([1.0, 0.20, 0.01]),
        np.asarray([1.0, 0.20, 0.01]),
        np.asarray([1.0, 0.20, 0.01]),
        np.asarray([True, True, True]),
        np.asarray([1.23456, 2.34567, 3.45678]),
        np.asarray([-1.23456, -2.34567, -3.45678]),
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([False, False, False]),
    )
    rows = []

    converged = _emit_scan_prints(
        hist_np=hist_np,
        it_start=10,
        max_iter_local=15,
        scan_minimal=False,
        scan_light=True,
        ftol=0.05,
        fsq_total_target=None,
        iter_offset0=2,
        should_print=lambda iter_idx, max_iter: iter_idx == 13 and max_iter == 15,
        print_row=lambda **kwargs: rows.append(kwargs),
    )

    assert converged
    assert [row["iter_idx"] for row in rows] == [13, 15]
    assert rows[0]["r00"] == pytest.approx(1.235)
    assert rows[0]["z00"] == pytest.approx(-1.235)


def test_emit_scan_prints_full_history_total_target_and_limits_max_iter():
    base = [
        np.asarray([0.6, 0.4, 0.2]),
        np.asarray([0.6, 0.4, 0.2]),
        np.asarray([0.6, 0.4, 0.2]),
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([True, True, True]),
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([4.0, 5.0, 6.0]),
        np.asarray([7.0, 8.0, 9.0]),
        np.asarray([0.1, 0.2, 0.3]),
    ]
    hist_np = tuple(base + [np.asarray([0, 0, 0]) for _ in range(14)])
    rows = []

    converged = _emit_scan_prints(
        hist_np=hist_np,
        it_start=0,
        max_iter_local=2,
        scan_minimal=False,
        scan_light=False,
        ftol=0.01,
        fsq_total_target=1.25,
        iter_offset0=0,
        should_print=lambda iter_idx, max_iter: False,
        print_row=lambda **kwargs: rows.append(kwargs),
    )

    assert converged
    assert [row["iter_idx"] for row in rows] == [2]
    assert rows[0]["w_mhd"] == 8.0


def test_emit_scan_prints_minimal_skips_rows():
    rows = []

    assert not _emit_scan_prints(
        hist_np=(np.asarray([0.0]),),
        it_start=0,
        max_iter_local=1,
        scan_minimal=True,
        scan_light=True,
        ftol=1.0,
        fsq_total_target=None,
        iter_offset0=0,
        should_print=lambda *_: True,
        print_row=lambda **kwargs: rows.append(kwargs),
    )
    assert rows == []


def test_record_scan_device_ready_accumulates_breakdown():
    stats = {
        "scan_device_dispatch_s": 1.0,
        "scan_device_ready_s": 2.0,
        "scan_device_run_s": 3.0,
    }

    assert not _record_scan_device_ready(start=None, dispatch_done=11.0, ready_done=13.0, stats=stats)
    assert stats == {
        "scan_device_dispatch_s": 1.0,
        "scan_device_ready_s": 2.0,
        "scan_device_run_s": 3.0,
    }

    assert _record_scan_device_ready(start=10.0, dispatch_done=11.5, ready_done=14.0, stats=stats)
    assert stats == {
        "scan_device_dispatch_s": 2.5,
        "scan_device_ready_s": 4.5,
        "scan_device_run_s": 7.0,
    }


def test_append_timecontrol_accepts_string_path(tmp_path):
    path = Path(tmp_path / "string_path.log")

    assert _append_timecontrol_scan_trace_row(
        str(path),
        stage_id=0,
        iter2=1,
        iter1=2,
        fsq=3.0,
        fsq0=4.0,
        res0=5.0,
        res1=6.0,
        time_step=7.0,
        irst=8,
    )
    assert "init" in path.read_text(encoding="utf-8")

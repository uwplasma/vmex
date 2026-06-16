from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.solvers.fixed_boundary.diagnostics.io import _format_vmec2000_iter_row
from vmec_jax.solvers.fixed_boundary.scan.debug import (
    _append_timecontrol_scan_trace_row,
    _axis_guess_lines,
    dump_vmec2000_scan_ptau_rows,
    emit_vmec2000_post_scan_rows,
    maybe_debug_scan_force_first_iter,
    maybe_debug_scan_state_iter,
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


def test_emit_vmec2000_post_scan_rows_replays_selected_rows_with_vmec_rounding():
    rows = []
    histories = SimpleNamespace(
        r00=np.asarray([1.23456, 2.34567, 3.45678]),
        z00=np.asarray([-0.12345, -0.23456, -0.34567]),
        w_mhd=np.asarray([10.0, 20.0, 30.0]),
        dt=np.asarray([0.1, 0.2, 0.3]),
    )

    printed = emit_vmec2000_post_scan_rows(
        enabled=True,
        scan_histories=histories,
        fsqr_full=np.asarray([1.0, 2.0, 3.0]),
        fsqz_full=np.asarray([4.0, 5.0, 6.0]),
        fsql_full=np.asarray([7.0, 8.0, 9.0]),
        conv_idx_print=2,
        max_iter=3,
        should_print=lambda iter_idx, last_iter: iter_idx == last_iter,
        print_row=lambda **kwargs: rows.append(kwargs),
    )

    assert printed == 1
    assert rows == [
        {
            "iter_idx": 2,
            "fsqr": 2.0,
            "fsqz": 5.0,
            "fsql": 8.0,
            "delt0r": 0.2,
            "r00": float(f"{2.34567:.3E}"),
            "w_mhd": 20.0,
            "z00": float(f"{-0.23456:.3E}"),
        }
    ]
    assert (
        emit_vmec2000_post_scan_rows(
            enabled=False,
            scan_histories=histories,
            fsqr_full=np.asarray([1.0]),
            fsqz_full=np.asarray([1.0]),
            fsql_full=np.asarray([1.0]),
            conv_idx_print=0,
            max_iter=1,
            should_print=lambda *_args: True,
            print_row=lambda **_kwargs: pytest.fail("disabled replay should not print"),
        )
        == 0
    )


def test_dump_vmec2000_scan_ptau_rows_replays_scan_diagnostics():
    calls = []
    histories = SimpleNamespace(
        ptau_min=np.asarray([-1.0, -2.0]),
        ptau_max=np.asarray([1.0, 2.0]),
        tau_min_state=np.asarray([np.nan, -0.25]),
        tau_max_state=np.asarray([0.5, np.nan]),
        badjac_ptau=np.asarray([1, 0]),
        badjac_state=np.asarray([0, 1]),
        bad_jac=np.asarray([True, False]),
    )

    dumped = dump_vmec2000_scan_ptau_rows(
        enabled=True,
        scan_histories=histories,
        conv_idx_print=0,
        max_iter=2,
        iter_offset0=5,
        badjac_mode="state",
        dump_ptau=lambda **kwargs: calls.append(kwargs) or True,
    )

    assert dumped == 2
    assert calls[0]["iter_idx"] == 6
    assert calls[0]["tau_min_state"] is None
    assert calls[0]["tau_max_state"] == pytest.approx(0.5)
    assert calls[0]["badjac_ptau"] is True
    assert calls[0]["badjac_state"] is False
    assert calls[0]["badjac_used"] is True
    assert calls[0]["mode"] == "state"
    assert calls[0]["label"] == "scan"
    assert calls[1]["iter_idx"] == 7
    assert calls[1]["tau_min_state"] == pytest.approx(-0.25)
    assert calls[1]["tau_max_state"] is None
    assert calls[1]["badjac_used"] is False
    assert (
        dump_vmec2000_scan_ptau_rows(
            enabled=False,
            scan_histories=histories,
            conv_idx_print=0,
            max_iter=2,
            iter_offset0=0,
            badjac_mode="state",
            dump_ptau=lambda **_kwargs: pytest.fail("disabled dump should not run"),
        )
        == 0
    )


def test_maybe_debug_scan_force_first_iter_emits_first_iteration_payload():
    calls = []
    frzl = SimpleNamespace(
        fzsc=np.asarray([[[1.0, 2.0]]]),
        fzcs=np.asarray([[[3.0, 4.0], [5.0, 6.0]]]),
    )
    state = SimpleNamespace(Rcos=np.asarray([1.0, 2.0]), Zsin=np.asarray([3.0]))
    norms = SimpleNamespace(fnorm=7.0, r1=8.0)

    emitted = maybe_debug_scan_force_first_iter(
        enabled=True,
        iter2=1,
        frzl=frzl,
        carry_state=state,
        use_cached_precond=True,
        need_bcovar_update=False,
        norms_used=norms,
        gcr2=0.1,
        gcz2=0.2,
        fsqr=0.3,
        fsqz=0.4,
        jnp_module=np,
        cond=lambda pred, true, false, operand=None: true(operand) if bool(pred) else false(operand),
        debug_print=lambda fmt, **kwargs: calls.append((fmt, kwargs)),
    )

    assert emitted
    assert len(calls) == 1
    assert calls[0][0].startswith("[scan-debug]")
    assert calls[0][1]["fzsc2"] == pytest.approx(5.0)
    assert calls[0][1]["fzcsm1"] == pytest.approx(61.0)
    assert calls[0][1]["rcsum"] == pytest.approx(3.0)
    assert not maybe_debug_scan_force_first_iter(
        enabled=False,
        iter2=1,
        frzl=frzl,
        carry_state=state,
        use_cached_precond=True,
        need_bcovar_update=False,
        norms_used=norms,
        gcr2=0.1,
        gcz2=0.2,
        fsqr=0.3,
        fsqz=0.4,
        jnp_module=np,
        cond=lambda *_args, **_kwargs: pytest.fail("disabled debug should not enter cond"),
        debug_print=lambda **_kwargs: pytest.fail("disabled debug should not print"),
    )


def test_maybe_debug_scan_state_iter_emits_requested_iteration_payload():
    calls = []
    state = SimpleNamespace(Rcos=np.asarray([1.0]), Zsin=np.asarray([2.0]), Lsin=np.asarray([3.0]))
    checkpoint = SimpleNamespace(Rcos=np.asarray([4.0]), Zsin=np.asarray([5.0]), Lsin=np.asarray([6.0]))
    carry = SimpleNamespace(state=state, state_checkpoint=checkpoint)
    norms = SimpleNamespace(fnorm=2.0, r1=3.0, fnormL=4.0)

    emitted = maybe_debug_scan_state_iter(
        scan_debug_iter=7,
        iter2=7,
        carry_adv=carry,
        use_cached_precond=False,
        need_bcovar_update=True,
        norms_used=norms,
        gcr2=0.5,
        gcz2=0.25,
        gcl2=0.125,
        jnp_module=np,
        cond=lambda pred, true, false, operand=None: true(operand) if bool(pred) else false(operand),
        debug_print=lambda fmt, **kwargs: calls.append((fmt, kwargs)),
    )

    assert emitted
    assert len(calls) == 1
    assert calls[0][0].startswith("[scan-state]")
    assert calls[0][1]["fsqr"] == pytest.approx(3.0)
    assert calls[0][1]["fsqz"] == pytest.approx(1.5)
    assert calls[0][1]["fsql"] == pytest.approx(0.5)
    assert not maybe_debug_scan_state_iter(
        scan_debug_iter=0,
        iter2=7,
        carry_adv=carry,
        use_cached_precond=False,
        need_bcovar_update=True,
        norms_used=norms,
        gcr2=0.5,
        gcz2=0.25,
        gcl2=0.125,
        jnp_module=np,
        cond=lambda *_args, **_kwargs: pytest.fail("disabled debug should not enter cond"),
        debug_print=lambda **_kwargs: pytest.fail("disabled debug should not print"),
    )


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

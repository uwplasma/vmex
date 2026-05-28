from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.solve_scan_debug_helpers import _emit_scan_prints, _emit_vmec2000_iter_row
from vmec_jax.solve_scan_math_helpers import _ptau_minmax_from_k_host


def _kernel(ns: int = 3) -> SimpleNamespace:
    zeros = np.zeros((ns, 1, 1))
    pz1_even = np.asarray([0.0, 1.0, 3.0], dtype=float)[:ns, None, None]
    return SimpleNamespace(
        pru_even=np.full((ns, 1, 1), 2.0),
        pru_odd=zeros,
        pzu_even=zeros,
        pzu_odd=zeros,
        pr1_even=zeros,
        pr1_odd=zeros,
        pz1_even=pz1_even,
        pz1_odd=zeros,
    )


def test_emit_vmec2000_iter_row_covers_disabled_live_and_lasym_missing_callback() -> None:
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
        print_live=False,
        print_row=lambda **kwargs: rows.append(kwargs),
    )
    assert rows == []

    assert not _emit_vmec2000_iter_row(
        iter_idx=2,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=0.1,
        r00=1.2,
        z00=0.3,
        w_mhd=4.0,
        lasym=True,
        scan_print_mode="io_callback",
        jax_debug=object(),
        io_callback=None,
        print_row=lambda **kwargs: rows.append(kwargs),
    )
    assert rows == []


def test_emit_scan_prints_covers_iteration_limit_break_and_no_convergence() -> None:
    light_hist = (
        np.asarray([10.0, 9.0]),
        np.asarray([8.0, 7.0]),
        np.asarray([6.0, 5.0]),
        np.asarray([False, False]),
        np.asarray([1.0, 1.1]),
        np.asarray([0.0, 0.1]),
        np.asarray([2.0, 2.1]),
        np.asarray([0.9, 0.8]),
        np.asarray([False, False]),
    )
    rows = []

    assert not _emit_scan_prints(
        hist_np=light_hist,
        it_start=1,
        max_iter_local=1,
        scan_minimal=False,
        scan_light=True,
        ftol=1.0e-12,
        fsq_total_target=None,
        iter_offset0=0,
        should_print=lambda _iter, _max_iter: True,
        print_row=lambda **kwargs: rows.append(kwargs),
    )
    assert rows == []

    assert not _emit_scan_prints(
        hist_np=light_hist,
        it_start=0,
        max_iter_local=3,
        scan_minimal=False,
        scan_light=True,
        ftol=1.0e-12,
        fsq_total_target=None,
        iter_offset0=0,
        should_print=lambda _iter, _max_iter: False,
        print_row=lambda **kwargs: rows.append(kwargs),
    )
    assert rows == []


def test_ptau_host_compute_jit_exception_path() -> None:
    assert _ptau_minmax_from_k_host(
        _kernel(),
        pshalf=np.asarray([1.0, 1.0, 1.0]),
        ohs=2.0,
        compute_jit=lambda *_args: (_ for _ in ()).throw(RuntimeError("bad jit")),
    ) == (None, None)

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.diagnostics import converged_wout_parity_benchmark as bench


def test_field_mode_hotspots_reports_largest_relative_mode_errors() -> None:
    ref = SimpleNamespace(
        xm=np.array([0, 1, 2]),
        xn=np.array([0, 0, 0]),
        xm_nyq=np.array([0, 1, 2, 3]),
        xn_nyq=np.array([0, 0, 0, 0]),
        lmns=np.array(
            [
                [0.0, 10.0, 5.0],
                [0.0, 10.0, 5.0],
            ]
        ),
    )
    got = SimpleNamespace(
        xm=ref.xm,
        xn=ref.xn,
        xm_nyq=ref.xm_nyq,
        xn_nyq=ref.xn_nyq,
        lmns=np.array(
            [
                [0.0, 11.0, 5.1],
                [0.0, 11.0, 5.1],
            ]
        ),
    )

    rows = bench._field_mode_hotspots(got, ref, "lmns", top_n=2)

    assert [row["m"] for row in rows] == [1, 2]
    assert rows[0]["rel_rms"] == pytest.approx(0.1)
    assert rows[0]["diff_rms"] == pytest.approx(1.0)
    assert rows[1]["rel_rms"] == pytest.approx(0.02)


def test_field_mode_hotspots_uses_nyquist_mode_arrays_when_needed() -> None:
    ref = SimpleNamespace(
        xm=np.array([0, 1]),
        xn=np.array([0, 0]),
        xm_nyq=np.array([0, 1, 2, 3]),
        xn_nyq=np.array([0, 0, 0, 0]),
        bsubvmns=np.ones((3, 4)),
    )
    got = SimpleNamespace(
        xm=ref.xm,
        xn=ref.xn,
        xm_nyq=ref.xm_nyq,
        xn_nyq=ref.xn_nyq,
        bsubvmns=np.array(
            [
                [1.0, 1.0, 1.0, 3.0],
                [1.0, 1.0, 1.0, 3.0],
                [1.0, 1.0, 1.0, 3.0],
            ]
        ),
    )

    rows = bench._field_mode_hotspots(got, ref, "bsubvmns", radial_skip=1, top_n=1)

    assert rows == [
        {
            "index": 3,
            "m": 3,
            "n": 0,
            "rel_rms": 2.0,
            "diff_rms": 2.0,
            "ref_rms": 1.0,
            "max_abs_diff": 2.0,
        }
    ]

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.diagnostics.parity import converged_wout_parity_benchmark as bench


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


def test_reference_state_roundtrip_rel_rms_rebuilds_from_input(monkeypatch) -> None:
    import vmec_jax.config as config_mod
    import vmec_jax.namelist as namelist_mod
    import vmec_jax.static as static_mod

    ref = SimpleNamespace(
        signgs=-1,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        fsqt=np.array([4.0, 5.0]),
    )
    indata = object()
    cfg = object()
    static = object()
    state = object()
    rebuilt = object()

    monkeypatch.setattr(namelist_mod, "read_indata", lambda path: indata)
    monkeypatch.setattr(config_mod, "config_from_indata", lambda arg: cfg)
    monkeypatch.setattr(static_mod, "build_static", lambda arg: static)
    monkeypatch.setattr(bench, "state_from_wout", lambda arg: state)

    def fake_wout_minimal_from_fixed_boundary(**kwargs):
        assert kwargs["path"] == "reference_state_roundtrip"
        assert kwargs["state"] is state
        assert kwargs["static"] is static
        assert kwargs["indata"] is indata
        assert kwargs["signgs"] == -1
        assert kwargs["fsqr"] == 1.0
        assert kwargs["fsqz"] == 2.0
        assert kwargs["fsql"] == 3.0
        np.testing.assert_allclose(kwargs["fsqt"], [4.0, 5.0])
        assert kwargs["converged"] is True
        return rebuilt

    monkeypatch.setattr(bench, "wout_minimal_from_fixed_boundary", fake_wout_minimal_from_fixed_boundary)
    monkeypatch.setattr(
        bench,
        "_compare_wouts",
        lambda got, ref_arg, *, lasym: {"lmns": 0.0} if got is rebuilt and ref_arg is ref and lasym else {},
    )

    assert bench._reference_state_roundtrip_rel_rms(ref, Path("input.test"), lasym=True) == {"lmns": 0.0}

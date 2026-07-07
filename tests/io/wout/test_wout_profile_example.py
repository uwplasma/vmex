from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import load_python_module


ROOT = Path(__file__).resolve().parents[3]


def _load_example_module():
    script = ROOT / "examples" / "diagnostics" / "load_save_wout_profiles.py"
    return load_python_module(script, name="load_save_wout_profiles_test", register=False)


def _toy_wout(ns: int = 4):
    return SimpleNamespace(
        ns=ns,
        mpol=3,
        ntor=2,
        nfp=2,
        lasym=False,
        aspect=5.0,
        volume_p=12.0,
        betatotal=0.01,
        fsqr=1.0e-12,
        fsqz=2.0e-12,
        fsql=3.0e-12,
        iotaf=np.linspace(0.3, 0.6, ns),
    )


def test_wout_profile_example_reuses_existing_wout(monkeypatch, tmp_path: Path) -> None:
    mod = _load_example_module()
    wout_path = tmp_path / "wout.nc"
    wout_path.write_text("placeholder")
    expected = _toy_wout()

    monkeypatch.setattr(mod, "WOUT_FILE", wout_path)
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "read_wout", lambda path: expected if path == wout_path else None)
    monkeypatch.setattr(mod.vj, "run_fixed_boundary", lambda *args, **kwargs: pytest.fail("should reuse wout"))

    assert mod.load_or_create_wout() is expected


def test_wout_profile_example_bmag_profile_and_summary(monkeypatch, capsys) -> None:
    mod = _load_example_module()
    wout = _toy_wout(ns=3)

    def fake_bmag_grid(_wout, *, s_index, ntheta, nzeta):
        assert ntheta == 32
        assert nzeta == 32
        bmag = np.full((ntheta, nzeta), 1.0 + float(s_index))
        return np.arange(ntheta), np.arange(nzeta), bmag

    monkeypatch.setattr(mod.vj, "vmecplot2_bmag_grid", fake_bmag_grid)

    s, b_mean, b_max = mod.radial_bmag_profile(wout)
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0])
    assert np.isnan(b_mean[0])
    assert np.isnan(b_max[0])
    np.testing.assert_allclose(b_mean[1:], [2.0, 3.0])
    np.testing.assert_allclose(b_max[1:], [2.0, 3.0])

    mod.print_wout_summary(wout)
    output = capsys.readouterr().out
    assert "Wout summary:" in output
    assert "Radial profiles:" in output
    assert "iota_f" in output
    assert "<|B|>" in output
    assert "magnetic-axis row" in output


def test_wout_profile_example_main_saves_roundtrip(monkeypatch, tmp_path: Path) -> None:
    mod = _load_example_module()
    wout = _toy_wout(ns=2)
    writes = []

    monkeypatch.setattr(mod, "ROUNDTRIP_FILE", tmp_path / "wout_roundtrip.nc")
    monkeypatch.setattr(mod, "load_or_create_wout", lambda: wout)
    monkeypatch.setattr(mod, "write_wout", lambda path, obj, *, overwrite: writes.append((path, obj, overwrite)))
    monkeypatch.setattr(mod, "print_wout_summary", lambda obj: writes.append(("printed", obj, None)))

    mod.main()

    assert writes[0] == (tmp_path / "wout_roundtrip.nc", wout, True)
    assert writes[1] == ("printed", wout, None)

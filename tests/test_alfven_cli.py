"""``vmex --alfven``: default harmonic selection, the dependency message, and an end-to-end smoke run (FAX)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from vmex.core import alfven_run
from vmex.core.cli import main

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


@pytest.mark.parametrize("nfp, iota, n, m_max", [
    (1, [0.3, 0.5], [-1, 0, 1, 2], 7),  # tokamak: iota < 1 adds n = +-1; m_max = 3 max(1, 0.5 * 2) + 4
    (5, [1.1, 1.3], [0, 5, 10], 12),  # iota > 1: no n = +-1; the cap of 12
    (2, [0.8, 0.9], [-1, 0, 1, 2, 4], 12),
])
def test_default_harmonic_selection(nfp, iota, n, m_max):
    assert alfven_run.default_harmonics(nfp, np.array(iota)) == (n, m_max)


def test_alfven_flag_runs_after_a_solve_and_reports_missing_fax(tmp_path, monkeypatch):
    """The solve path hands its WOUT and the options to the driver; an ImportError becomes the dependency message."""
    from vmex import _compat
    from vmex.core import cli
    from vmex.core.errors import VmecInputError

    calls = []
    monkeypatch.setattr(alfven_run, "run_alfven", lambda wout, outdir, **kw: calls.append((wout, kw)))
    assert main(["--test", "--alfven", "--alfven-density", "5e19", "--alfven-ion-mass", "1", "--outdir", str(tmp_path),
                 "--quiet"]) == 0
    assert calls[0][0].name.startswith("wout_") and calls[0][1]["density"] == 5e19 and calls[0][1]["ion_mass"] == 1.0

    monkeypatch.undo()
    monkeypatch.setattr(_compat.importlib_metadata, "version", lambda name: (_ for _ in ()).throw(
        _compat.importlib_metadata.PackageNotFoundError(name)))
    args = cli.build_parser().parse_args([str(calls[0][0]), "--alfven"])
    with pytest.raises(VmecInputError, match="OPTIONAL DEPENDENCY") as err:
        cli._run_alfven(calls[0][0], args, tmp_path, emit=print, quiet=True)
    assert "alfvex>=" in str(err.value.hint)


def test_alfven_cli_smoke(tmp_path, capsys, monkeypatch):
    fax = pytest.importorskip("fax")
    if not hasattr(fax, "summary"):
        pytest.skip("this FAX predates fax.summary")
    import vmex
    from vmex import _compat
    from vmex.core import optimize as opt
    from vmex.core.input import VmecInput

    monkeypatch.setitem(_compat.OPTIONAL_MINIMUMS, "alfvex", "0")  # a source checkout of FAX before its release
    inp = VmecInput.from_file(DATA_DIR / "input.shaped_tokamak_pressure")
    inp = dataclasses.replace(inp, ns_array=np.array([8]), ftol_array=np.array([1e-12]), niter_array=np.array([2000]))
    wout = vmex.write_wout(tmp_path / "wout_smoke.nc", opt.solve_equilibrium(inp).wout)
    case = tmp_path / "alfven.toml"
    case.write_text('[geometry]\nmboz = 4\nnboz = 4\nsurfaces = [0.2, 0.4, 0.6, 0.8]\n[modes]\nn = [1]\nm_max = 3\nm_min = 1\n'
                    '[run]\nntheta = 16\nnzeta = 8\ncount = 3\n')
    argv = [str(wout), "--alfven", "--outdir", str(tmp_path), "--alfven-config", str(case)]
    assert main(argv) == 0
    assert "FAX shear-Alfven continuum" in capsys.readouterr().out
    out = json.loads((tmp_path / "smoke_alfven.json").read_text())
    assert out["density_m3"] == 1e19 and out["ion_mass_amu"] == 2.0 and out["density_kg_m3"] == pytest.approx(2 * 1.6605e-8, rel=1e-3)
    assert out["mode"] is None or (out["mode"]["converged"] and np.isfinite(out["mode"]["frequency_khz"]))
    assert out["gaps"] and (out["mode"] or out["mode_reason"])
    assert (tmp_path / "smoke_alfven.png").stat().st_size > 5_000
    case.write_text("[bogus]\nx = 1\n")
    assert main(argv) != 0
    assert "unknown table or key in [bogus]" in capsys.readouterr().out

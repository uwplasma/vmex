"""``vmex --alfven``: the dependency message and end-to-end smoke runs (FAX)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from vmex.core import alfven_run
from vmex.core.cli import main

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


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
    if not hasattr(fax, "mode_families"):
        pytest.skip("this FAX predates the local-gap fax.summary")
    import vmex
    from vmex import _compat
    from vmex.core import optimize as opt
    from vmex.core.input import VmecInput

    monkeypatch.setitem(_compat.OPTIONAL_MINIMUMS, "alfvex", "0")  # a source checkout of FAX before its release
    inp = VmecInput.from_file(DATA_DIR / "input.shaped_tokamak_pressure")
    inp = dataclasses.replace(inp, ns_array=np.array([8]), ftol_array=np.array([1e-12]), niter_array=np.array([2000]))
    wout = vmex.write_wout(tmp_path / "wout_smoke.nc", opt.solve_equilibrium(inp).wout)
    case = tmp_path / "alfven.toml"
    case.write_text('[geometry]\nmboz = 4\nnboz = 4\nsurfaces = [0.2, 0.4, 0.6, 0.8]\n[modes]\nn = [1]\nm_max = 4\nm_min = 0\n'
                    '[plasma]\ndensity_profile = [1.0, -0.5]\n[run]\nntheta = 16\nnzeta = 8\ncount = 3\n')
    argv = [str(wout), "--alfven", "--outdir", str(tmp_path), "--alfven-config", str(case)]
    assert main(argv) == 0
    assert "FAX shear-Alfven continuum" in capsys.readouterr().out
    out = json.loads((tmp_path / "smoke_alfven.json").read_text())
    n_i, rho = np.array(out["density_m3"]), np.array(out["density_kg_m3"])
    assert n_i[0] < 1e19 and np.all(np.diff(n_i) < 0) and out["ion_mass_amu"] == 2.0 and out["sound"]  # 1 - s / 2, finite pressure
    assert rho == pytest.approx(n_i * 2 * 1.6605e-27, rel=1e-3) and out["alfven_khz"] > 0
    assert out["mode"] is None or (out["mode"]["converged"] and np.isfinite(out["mode"]["frequency_khz"]))
    assert out["mode"] or out["mode_reason"]
    assert (tmp_path / "smoke_alfven.png").stat().st_size > 5_000
    case.write_text('[geometry]\nmboz = 4\nnboz = 4\n[modes]\nm_max = 4\n[run]\ncount = 2\n')
    assert main(argv) == 0  # Boozer on the half-mesh surfaces themselves; fax.alfven_harmonics picks n = 1, 2
    assert {n for _, n in json.loads((tmp_path / "smoke_alfven.json").read_text())["harmonics"]} == {1, 2}
    assert len(json.loads((tmp_path / "smoke_alfven.json").read_text())["density_m3"]) == 7  # ns = 8
    case.write_text("[bogus]\nx = 1\n")
    assert main(argv) != 0
    assert "unknown table or key in [bogus]" in capsys.readouterr().out

"""``vmex --turbulence``: helper identities (no gkx) and a tiny end-to-end smoke run (gkx)."""

from __future__ import annotations

import dataclasses
import json
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import numpy as np
import pytest

from vmex.core import gk_run
from vmex.core.cli import main

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"


def test_saturated_mean_is_window_mean_with_batch_error():
    t = np.linspace(0.0, 100.0, 401)
    q = np.where(t < 50.0, 0.0, 2.0 + 0.1 * np.sin(t))
    mean, err, t0 = gk_run.saturated_mean(t, q)
    assert t0 == 50.0
    assert mean == pytest.approx(q[t >= 50.0].mean())
    assert 0.0 < err < 0.1
    assert gk_run.saturated_mean(t, q, t_start=80.0)[2] == 80.0


def test_ky_scan_points_sit_on_the_gkx_grid(tmp_path):
    cfg = gk_run.TurbulenceSettings(ky_min=0.1, ky_max=1.0, nky=8)
    ky = gk_run.ky_grid(cfg)
    assert ky[0] == pytest.approx(0.1) and ky[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(ky / 0.1, np.rint(ky / 0.1), atol=1e-12)
    deck = tomllib.loads(gk_run.write_deck(tmp_path / "d.toml", tmp_path / "wout_x.nc", cfg,
                                           nonlinear=False).read_text())
    assert deck["grid"]["y0"] == pytest.approx(10.0)
    assert (deck["grid"]["Ny"] - 1) // 3 >= 10  # highest scan ky is a resolved mode
    assert deck["geometry"]["torflux"] == 0.5 and deck["physics"]["adiabatic_electrons"] is True
    kinetic = dataclasses.replace(cfg, kinetic_electrons=True)
    deck = tomllib.loads(gk_run.write_deck(tmp_path / "k.toml", tmp_path / "wout_x.nc", kinetic,
                                           nonlinear=True).read_text())
    assert [s["name"] for s in deck["species"]] == ["ion", "electron"]
    assert deck["physics"]["nonlinear"] is True and deck["physics"]["adiabatic_electrons"] is False
GEOM = {"q": 1.4, "s_hat": 0.8, "gds21": np.array([0.5, 0.0]), "gds22": np.array([2.0, 1.0])}


def test_defaults_deck_and_settings_block(tmp_path):
    cfg = gk_run.TurbulenceSettings()
    assert (cfg.nx, cfg.ny, cfg.nz, cfg.npol, cfg.nl, cfg.nm, cfg.t_max, cfg.t_sat_min) == (
        96, 96, 24, 3.0, 4, 8, 800.0, 300.0)
    out = tmp_path / "x.out.nc"
    deck = tomllib.loads(gk_run.write_deck(tmp_path / "d.toml", tmp_path / "wout_x.nc", cfg,
                                           nonlinear=True).read_text())
    assert deck["grid"]["Nz"] == deck["grid"]["ntheta"] == 72 and deck["geometry"]["npol"] == 3.0
    assert deck["grid"]["boundary"] == "fix aspect" and deck["time"]["saturation_t_min"] == 300.0
    old = tomllib.loads(gk_run.write_deck(tmp_path / "o.toml", tmp_path / "wout_x.nc", cfg, nonlinear=True,
                                          saturation=False).read_text())
    assert "saturation_t_min" not in old["time"]
    text = "\n".join(gk_run.settings_block(Path("wout_x.nc"), cfg, backend="GKX test, cpu", out=out, geom=GEOM,
                                           rel_sem=0.1, t_sat_min=300.0))
    # jtwist = round(2 * 0.8 * 0.5 / 2) = 0 -> 1; the box follows GKX's rule x0 = y0 jtwist / factor
    for needle in ("s = 0.5 (rho = 0.707)", "alpha = 0", "q = 1.400", "s_hat = 0.800", "a/LT = 3", "a/Ln = 1",
                   "96 x 96 x 72", "(Nl, Nm) = (4, 8)", "y0 = 10", "jtwist = 1 (GKX default rule)",
                   "t_max = 800", "SEM <= 10%", "t = 300", "adiabatic", "nu_i = 0.01", "D = 0.05", "float32",
                   "uncertainty", "seed", str(out)):
        assert needle in text, needle


def test_kinetic_electron_deck_follows_gkx_example(tmp_path):
    cfg = gk_run.TurbulenceSettings(kinetic_electrons=True)
    assert (cfg.nx, cfg.ny, cfg.nz) == (16, 16, 16)
    for nonlinear in (False, True):
        deck = tomllib.loads(gk_run.write_deck(tmp_path / "k.toml", tmp_path / "wout_x.nc", cfg,
                                               nonlinear=nonlinear).read_text())
        assert ("cfl" in deck["time"]) is (not nonlinear)  # linear time route: 0.2 of the CFL step
        ion, ele = deck["species"]
        assert (ion["Nl"], ion["Nm"], ele["Nl"], ele["Nm"]) == (4, 8, 4, 48)
        assert deck["physics"]["electrostatic"] is False and deck["physics"]["beta"] == 1e-4
        assert deck["terms"]["apar"] == 1.0 and deck["physics"]["use_apar"] is True


def test_cli_exposes_only_the_essential_flags(tmp_path, monkeypatch):
    from vmex.core import cli

    seen = []
    monkeypatch.setattr(gk_run, "run_turbulence", lambda wout, outdir, cfg, emit: seen.append(cfg))
    for flags in ([], ["--turbulence-kinetic-electrons", "--turbulence-tmax", "50", "--turbulence-s", "0.3"]):
        args = cli.build_parser().parse_args(["w.nc", "--turbulence", *flags])
        cli._run_turbulence(tmp_path / "w.nc", args, tmp_path, emit=print, quiet=True)
    assert seen[0] == gk_run.TurbulenceSettings()
    assert seen[1].kinetic_electrons and seen[1].t_max == 50.0 and seen[1].s == 0.3
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["w.nc", "--turbulence", "--turbulence-grid", "8", "8", "16"])


def test_turbulence_smoke(tmp_path, capsys):
    pytest.importorskip("gkx", minversion="2.6.0")
    import vmex
    from vmex.core import optimize as opt
    from vmex.core.input import VmecInput

    inp = VmecInput.from_file(DATA_DIR / "input.shaped_tokamak_pressure")
    inp = dataclasses.replace(inp, ns_array=np.array([13]), ftol_array=np.array([1e-12]),
                              niter_array=np.array([2000]))
    wout = vmex.write_wout(tmp_path / "wout_smoke.nc", opt.solve_equilibrium(inp).wout)
    cfg = gk_run.TurbulenceSettings(ky_min=0.2, ky_max=0.2, nky=1, nx=8, ny=8, nz=16, nl=2, nm=2, t_max=2.0)
    gk_run.run_turbulence(wout, tmp_path, cfg)
    out = capsys.readouterr().out
    for needle in ("Linear ky scan", "Turbulence summary", "flux tube", "jtwist", "uncertainty"):
        assert needle in out, needle
    assert (tmp_path / "smoke_turbulence.out.nc").stat().st_size > 0
    summary = json.loads((tmp_path / "smoke_turbulence.json").read_text())
    assert summary["ky"] == pytest.approx([0.2])
    assert np.all(np.isfinite(summary["gamma"])) and np.all(np.isfinite(summary["heat_flux"]))
    for suffix in ("", "_geometry", "_fluxes"):
        assert (tmp_path / f"smoke_turbulence{suffix}.png").stat().st_size > 10_000


def test_turbulence_flag_runs_after_a_solve_and_reports_missing_gkx(tmp_path, monkeypatch):
    """The solve path hands its WOUT to the driver; an ImportError becomes the dependency message."""
    from vmex.core import cli
    from vmex.core.errors import VmecInputError

    calls = []
    monkeypatch.setattr(gk_run, "run_turbulence", lambda wout, outdir, cfg, emit: calls.append((wout, cfg)))
    assert main(["--test", "--turbulence", "--turbulence-s", "0.3", "--outdir", str(tmp_path), "--quiet"]) == 0
    assert calls and calls[0][0].name.startswith("wout_") and calls[0][1].s == 0.3

    def missing(*args, **kwargs):
        raise ImportError("vmex --turbulence needs gkx>=2.6.0")

    monkeypatch.setattr(gk_run, "run_turbulence", missing)
    args = cli.build_parser().parse_args([str(calls[0][0]), "--turbulence"])
    with pytest.raises(VmecInputError, match="OPTIONAL DEPENDENCY"):
        cli._run_turbulence(calls[0][0], args, tmp_path, emit=print, quiet=True)

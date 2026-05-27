from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vmec_jax import cli
from vmec_jax.booz import (
    BoozConfig,
    _as_first,
    _case_from_vmec_or_wout,
    _surface_indices_from_values,
    _truthy,
    parse_booz_surfaces,
    read_booz_config,
    resolve_boozmn_path,
    run_booz_xform,
)
from vmec_jax.namelist import InData
from vmec_jax.plotting import plot_boozmn


def test_booz_config_parser_reads_separate_namelist(tmp_path: Path) -> None:
    path = tmp_path / "input.case"
    path.write_text(
        """
&INDATA
  NITER = 1
/

&BOOZ_XFORM_JAX
  LBOOZ = T
  MBOOZ = 17
  NBOOZ = 19
  BOOZ_SURFACES = 0.25, 0.5, 1.0
  JIT_BOOZ = F
/
""".strip()
    )

    cfg = read_booz_config(path)

    assert cfg == BoozConfig(enabled=True, mbooz=17, nbooz=19, surfaces=(0.25, 0.5, 1.0), jit=False)
    plain = tmp_path / "input.plain"
    plain.write_text("&INDATA\n/\n")
    assert read_booz_config(plain).enabled is False


def test_booz_private_parsers_cover_scalar_truthiness_and_errors(tmp_path: Path) -> None:
    assert _truthy(True) is True
    assert _truthy(0) is False
    assert _truthy(2.0) is True
    assert _truthy("off", default=True) is False
    assert _truthy("ON") is True
    assert _truthy("unknown", default=True) is True
    assert _as_first([], default="fallback") == "fallback"
    assert _as_first(["first", "second"]) == "first"
    assert _case_from_vmec_or_wout(Path("input_case")) == "case"
    assert _case_from_vmec_or_wout(Path("custom.ext")) == "custom"

    unterminated = tmp_path / "input.bad"
    unterminated.write_text("&BOOZ_XFORM_JAX\n LBOOZ = T\n")
    with pytest.raises(ValueError, match="No terminating"):
        read_booz_config(unterminated)


def test_tracked_example_inputs_carry_disabled_boozer_defaults() -> None:
    roots = [Path("examples/data"), Path("vmec_jax/data")]
    input_paths = sorted(path for root in roots for path in root.glob("input.*"))
    assert input_paths
    for path in input_paths:
        cfg = read_booz_config(path)
        assert cfg.enabled is False, path
        assert cfg.mbooz == 32, path
        assert cfg.nbooz == 32, path
        assert cfg.surfaces is None, path


def test_parse_booz_surfaces_accepts_all_strings_and_indices() -> None:
    assert parse_booz_surfaces(None) is None
    assert parse_booz_surfaces("all") is None
    assert parse_booz_surfaces("*") is None
    assert parse_booz_surfaces("0.1, 0.5 1.0") == (0.1, 0.5, 1.0)
    assert parse_booz_surfaces([0, 4, 8]) == (0.0, 4.0, 8.0)


def test_resolve_boozmn_path_uses_vmec_case_conventions(tmp_path: Path) -> None:
    assert resolve_boozmn_path(source_path=tmp_path / "wout_case.nc") == tmp_path / "boozmn_case.nc"
    assert resolve_boozmn_path(source_path=tmp_path / "input.case") == tmp_path / "boozmn_case.nc"
    assert (
        resolve_boozmn_path(source_path=tmp_path / "wout_case.nc", outdir=tmp_path / "out")
        == tmp_path / "out" / "boozmn_case.nc"
    )
    explicit = tmp_path / "custom.nc"
    assert resolve_boozmn_path(source_path=tmp_path / "wout_case.nc", output=explicit) == explicit


def test_run_booz_xform_writes_boozmn_with_requested_resolution_and_surfaces(monkeypatch, tmp_path: Path) -> None:
    wout = tmp_path / "wout_case.nc"
    wout.write_text("placeholder")
    calls: dict[str, object] = {}

    class FakeBooz:
        def __init__(self, *, verbose: int | bool, mboz: int, nboz: int):
            calls["init"] = (verbose, mboz, nboz)
            self.ns_in = 5
            self.s_in = np.linspace(0.0, 1.0, 5)
            self.compute_surfs = None

        def read_wout(self, filename: str, *, flux: bool):
            calls["read_wout"] = (Path(filename), flux)

        def run(self, *, jit: bool):
            calls["run"] = (jit, list(self.compute_surfs))

        def write_boozmn(self, filename: str):
            calls["write"] = Path(filename)
            Path(filename).write_text("booz")

    fake_module = ModuleType("booz_xform_jax")
    fake_module.Booz_xform = FakeBooz
    monkeypatch.setitem(sys.modules, "booz_xform_jax", fake_module)

    out = run_booz_xform(
        wout,
        outdir=tmp_path / "out",
        mbooz=8,
        nbooz=9,
        surfaces=(0.0, 0.51, 1.0),
        jit=True,
        verbose=False,
    )

    assert out == (tmp_path / "out" / "boozmn_case.nc").resolve()
    assert out.read_text() == "booz"
    assert calls["init"] == (0, 8, 9)
    assert calls["read_wout"] == (wout.resolve(), False)
    assert calls["run"] == (True, [0, 2, 4])
    assert calls["write"] == out


def test_booz_surface_indices_accept_integer_indices_and_validate_bounds(tmp_path: Path) -> None:
    bx = SimpleNamespace(ns_in=4, s_in=np.asarray([0.1, 0.3]))

    assert _surface_indices_from_values(bx, None) is None
    assert _surface_indices_from_values(bx, (0.0, 0.5, 1.0)) == [0, 1, 3]
    assert _surface_indices_from_values(SimpleNamespace(ns_in=4, s_in=np.linspace(0.0, 1.0, 4)), (2, 3)) == [2, 3]

    with pytest.raises(ValueError, match="before reading"):
        _surface_indices_from_values(SimpleNamespace(ns_in=0), (0.5,))
    with pytest.raises(ValueError, match="outside"):
        _surface_indices_from_values(SimpleNamespace(ns_in=2, s_in=np.linspace(0.0, 1.0, 2)), (3,))

    with pytest.raises(FileNotFoundError, match="WOUT file not found"):
        run_booz_xform(tmp_path / "missing_wout.nc")


def test_cli_booz_and_plot_dispatch_for_wout_and_boozmn(monkeypatch, tmp_path: Path) -> None:
    wout = tmp_path / "wout_case.nc"
    wout.write_text("wout")
    generic_wout = tmp_path / "equilibrium.nc"
    generic_wout.write_text("wout")
    boozmn = tmp_path / "boozmn_case.nc"
    boozmn.write_text("booz")
    calls: list[tuple] = []

    monkeypatch.setattr(cli, "_plot_wout_file", lambda path, outdir: calls.append(("plot_wout", path, outdir)))
    monkeypatch.setattr(cli, "_plot_boozmn_file", lambda path, outdir: calls.append(("plot_booz", path, outdir)))
    monkeypatch.setattr(
        cli,
        "_run_booz_for_wout",
        lambda path, *, source_input_path, args, plot, outdir: calls.append(("run_booz", path, source_input_path, plot, outdir))
        or boozmn.resolve(),
    )

    assert cli.main(["--booz", "--plot", str(wout), "--outdir", str(tmp_path / "plots")]) == 0
    assert calls == [
        ("plot_wout", wout.resolve(), (tmp_path / "plots").resolve()),
        ("run_booz", wout.resolve(), None, True, (tmp_path / "plots").resolve()),
    ]

    calls.clear()
    assert cli.main(["--booz", str(generic_wout), "--outdir", str(tmp_path / "plots")]) == 0
    assert calls == [
        ("run_booz", generic_wout.resolve(), None, False, (tmp_path / "plots").resolve()),
    ]

    calls.clear()
    assert cli.main(["--plot", str(boozmn), "--outdir", str(tmp_path / "plots")]) == 0
    assert calls == [("plot_booz", boozmn.resolve(), (tmp_path / "plots").resolve())]


def test_cli_booz_config_for_path_merges_input_and_cli_options(tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text(
        """
&INDATA
/
&BOOZ_XFORM_JAX
  LBOOZ = T
  MBOOZ = 12
  NBOOZ = 13
  BOOZ_SURFACES = 0.25, 1.0
  JIT_BOOZ = F
/
""".strip()
    )
    parser = cli.build_parser()

    args = parser.parse_args([str(input_path)])
    assert cli._booz_config_for_path(input_path, args) == BoozConfig(
        enabled=True, mbooz=12, nbooz=13, surfaces=(0.25, 1.0), jit=False
    )

    args = parser.parse_args([str(input_path), "--booz", "--mbooz", "20", "--booz-surfaces", "all", "--jit-booz"])
    assert cli._booz_config_for_path(input_path, args) == BoozConfig(
        enabled=True, mbooz=20, nbooz=13, surfaces=None, jit=True
    )


def test_cli_run_booz_for_wout_uses_config_output_and_optional_plot(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text(
        """
&INDATA
/
&BOOZ_XFORM_JAX
  LBOOZ = T
  MBOOZ = 10
  NBOOZ = 11
  BOOZ_SURFACES = 0.25, 1.0
  JIT_BOOZ = T
/
""".strip()
    )
    wout = tmp_path / "wout_case.nc"
    wout.write_text("wout")
    boozmn = tmp_path / "chosen_boozmn.nc"
    calls: dict[str, object] = {}
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            str(wout),
            "--booz",
            "--booz-output",
            str(boozmn),
            "--quiet",
        ]
    )

    def fake_run_booz_xform(path, **kwargs):
        calls["run"] = (Path(path), kwargs)
        Path(kwargs["output_path"]).write_text("booz")
        return Path(kwargs["output_path"])

    monkeypatch.setattr("vmec_jax.booz.run_booz_xform", fake_run_booz_xform)
    monkeypatch.setattr(cli, "_plot_boozmn_file", lambda path, outdir: calls.setdefault("plot", (path, outdir)))

    out = cli._run_booz_for_wout(
        wout,
        source_input_path=input_path,
        args=args,
        plot=True,
        outdir=tmp_path / "plots",
    )

    assert out == boozmn.resolve()
    assert calls["run"] == (
        wout,
        {
            "output_path": boozmn.resolve(),
            "mbooz": 10,
            "nbooz": 11,
            "surfaces": (0.25, 1.0),
            "jit": True,
            "verbose": False,
        },
    )
    assert calls["plot"] == (boozmn.resolve(), tmp_path / "plots")


def test_cli_input_booz_plot_runs_after_wout(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text(
        """
&INDATA
  NITER = 1
/
&BOOZ_XFORM_JAX
  LBOOZ = F
  MBOOZ = 12
  NBOOZ = 13
  BOOZ_SURFACES = 'all'
/
""".strip()
    )
    indata = InData(scalars={"NITER": 1}, indexed={})
    calls: dict[str, object] = {}

    monkeypatch.setattr(cli, "read_indata", lambda path: indata)
    monkeypatch.setattr(cli, "default_non_autodiff_solver_policy", lambda _indata: ("default", True))
    monkeypatch.setattr(cli, "_default_use_scan_for_backend", lambda _indata, _backend, _mode: False)
    monkeypatch.setattr(
        cli,
        "run_fixed_boundary",
        lambda path, **kwargs: calls.setdefault("run", (Path(path), kwargs)) or SimpleNamespace(state=SimpleNamespace(Rcos=0.0)),
    )

    def fake_write_wout(path: Path, run, *, include_fsq: bool):
        calls["wout"] = (path, include_fsq)
        path.write_text("wout")

    def fake_run_booz(path: Path, *, source_input_path: Path | None, args, plot: bool, outdir: Path):
        calls["booz"] = (path, source_input_path, args.mbooz, args.nbooz, plot, outdir)
        return outdir / "boozmn_case.nc"

    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", fake_write_wout)
    monkeypatch.setattr(cli, "_plot_wout_file", lambda path, outdir: calls.setdefault("plot_wout", (path, outdir)))
    monkeypatch.setattr(cli, "_run_booz_for_wout", fake_run_booz)

    assert cli.main([str(input_path), "--booz", "--plot", "--quiet", "--outdir", str(tmp_path / "out")]) == 0

    assert calls["run"][0] == input_path.resolve()
    assert calls["wout"] == ((tmp_path / "out" / "wout_case.nc").resolve(), True)
    assert calls["plot_wout"] == ((tmp_path / "out" / "wout_case.nc").resolve(), (tmp_path / "out").resolve())
    assert calls["booz"] == (
        (tmp_path / "out" / "wout_case.nc").resolve(),
        input_path.resolve(),
        None,
        None,
        True,
        (tmp_path / "out").resolve(),
    )


def test_real_booz_xform_and_plotting_smoke(tmp_path: Path) -> None:
    pytest.importorskip("booz_xform_jax")
    wout = Path("examples/data/wout_nfp4_QH_warm_start.nc")
    if not wout.exists():
        pytest.skip("optional WOUT fixture is not present")

    boozmn = run_booz_xform(wout, outdir=tmp_path, mbooz=4, nbooz=4, surfaces=(1.0,), verbose=False)
    assert boozmn.exists()
    assert boozmn.stat().st_size > 0

    plots = plot_boozmn(boozmn, outdir=tmp_path / "figures")
    assert set(plots) == {"bmag_contours", "mode_families", "lcfs_spectrum"}
    for path in plots.values():
        assert path.exists()
        assert path.stat().st_size > 0

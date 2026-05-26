from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.cli as cli
import vmec_jax.driver as driver
from vmec_jax.config import FreeBoundaryConfig, VMECConfig
from vmec_jax.namelist import InData
from vmec_jax.solve import SolveVmecResidualResult


def _cfg(*, lfreeb: bool = False, ns: int = 3) -> VMECConfig:
    return VMECConfig(
        mpol=2,
        ntor=0,
        ns=ns,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=True,
        ntheta=4,
        nzeta=1,
        free_boundary=FreeBoundaryConfig(
            enabled=lfreeb,
            mgrid_file="mgrid.synthetic.nc" if lfreeb else "NONE",
            extcur=(1.25,) if lfreeb else (),
            nvacskip=1,
        ),
    )


def _indata(**scalars) -> InData:
    base = {
        "NITER": 2,
        "FTOL": 1.0e-6,
        "DELT": 0.125,
        "SIGNGS": 1,
    }
    base.update(scalars)
    return InData(scalars=base, indexed={})


def _state_for_ns(ns: int, *, label: str) -> SimpleNamespace:
    return SimpleNamespace(
        layout=SimpleNamespace(ns=int(ns)),
        label=label,
        Rcos=np.asarray([float(ns)]),
    )


def _install_fast_run_fakes(monkeypatch, *, cfg: VMECConfig, indata: InData):
    calls: dict[str, list] = {
        "build_static": [],
        "prepare_mgrid": [],
        "validate_free_boundary": [],
        "solve_residual_iter": [],
        "initial_guess": [],
        "final_flux": [],
    }

    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "0")
    monkeypatch.setattr(driver, "load_config", lambda path: (cfg, indata))
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "cpu")
    monkeypatch.setattr(
        driver,
        "validate_free_boundary_config",
        lambda cfg_in, *, strict: calls["validate_free_boundary"].append((cfg_in, strict)),
    )
    monkeypatch.setattr(
        driver,
        "prepare_mgrid_for_config",
        lambda cfg_in, *, load_fields, strict: calls["prepare_mgrid"].append((cfg_in, load_fields, strict)) or None,
    )

    def fake_build_static(cfg_in, **kwargs):
        calls["build_static"].append((cfg_in, kwargs))
        return SimpleNamespace(
            cfg=cfg_in,
            s=np.linspace(0.0, 1.0, int(cfg_in.ns)),
            modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0])),
        )

    def fake_initial_guess(static, boundary, indata_arg, **kwargs):
        calls["initial_guess"].append((static, boundary, indata_arg, kwargs))
        return _state_for_ns(static.cfg.ns, label="initial")

    def fake_solve_residual_iter(state, static, **kwargs):
        calls["solve_residual_iter"].append((state, static, kwargs))
        return SolveVmecResidualResult(
            state=_state_for_ns(static.cfg.ns, label="solved"),
            n_iter=max(0, int(kwargs["max_iter"]) - 1),
            w_history=np.asarray([1.0, 1.0e-10]),
            fsqr2_history=np.asarray([1.0, 1.0e-10]),
            fsqz2_history=np.asarray([1.0, 1.0e-10]),
            fsql2_history=np.asarray([1.0, 1.0e-10]),
            grad_rms_history=np.asarray([0.0]),
            step_history=np.asarray([float(kwargs["step_size"])]),
            diagnostics={
                "converged": True,
                "use_scan": bool(kwargs["use_scan"]),
                "vmec2000_scan": bool(kwargs["use_scan"]),
                "resume_state": {"time_step": float(kwargs["step_size"])},
            },
        )

    def fake_final_flux_profiles_from_state(**kwargs):
        calls["final_flux"].append(kwargs)
        prof = dict(kwargs["prof_local"])
        prof["finalized"] = True
        return kwargs["flux_local"], prof

    monkeypatch.setattr(driver, "build_static", fake_build_static)
    monkeypatch.setattr(driver, "boundary_from_indata", lambda _indata, modes: SimpleNamespace(modes=modes))
    monkeypatch.setattr(
        driver,
        "flux_profiles_from_indata",
        lambda _indata, s, *, signgs: SimpleNamespace(
            phipf=np.ones_like(np.asarray(s), dtype=float),
            chipf=np.zeros_like(np.asarray(s), dtype=float),
            lamscale=np.ones_like(np.asarray(s), dtype=float),
        ),
    )
    monkeypatch.setattr(
        driver,
        "eval_profiles",
        lambda _indata, s: {"pressure": np.zeros_like(np.asarray(s), dtype=float)},
    )
    monkeypatch.setattr(driver, "initial_guess_from_boundary", fake_initial_guess)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solve_residual_iter)
    monkeypatch.setattr(driver, "_final_flux_profiles_from_state", fake_final_flux_profiles_from_state)
    return calls


@pytest.mark.parametrize("lfreeb, expected_scan", [(False, True), (True, False)])
def test_run_fixed_boundary_dispatches_fixed_and_free_static_branches(monkeypatch, tmp_path: Path, lfreeb: bool, expected_scan: bool) -> None:
    cfg = _cfg(lfreeb=lfreeb)
    indata = _indata(LFREEB=lfreeb, MGRID_FILE=cfg.mgrid_file)
    calls = _install_fast_run_fakes(monkeypatch, cfg=cfg, indata=indata)

    run = driver.run_fixed_boundary(
        tmp_path / "input.synthetic",
        solver_mode="accelerated",
        max_iter=2,
        verbose=False,
        grid=object(),
        jit_forces=False,
        jit_precompile=False,
        _auto_cli_fixed_boundary_mode=False,
    )

    assert run.cfg is cfg
    assert run.state.label == "solved"
    assert run.profiles["finalized"] is True
    if lfreeb:
        assert calls["validate_free_boundary"] == [(cfg, True)]
        assert calls["prepare_mgrid"] == [(cfg, False, True)]
    else:
        assert calls["validate_free_boundary"] == []
        assert calls["prepare_mgrid"] == []
    build_kwargs = calls["build_static"][0][1]
    if lfreeb:
        assert "mgrid_metadata" in build_kwargs
        assert "free_boundary_extcur" in build_kwargs
    else:
        assert "mgrid_metadata" not in build_kwargs
        assert "free_boundary_extcur" not in build_kwargs
    solve_kwargs = calls["solve_residual_iter"][0][2]
    assert solve_kwargs["signgs"] == -1
    assert solve_kwargs["use_scan"] is expected_scan


def test_run_fixed_boundary_selects_default_accelerated_policy_and_explicit_parity(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg()
    indata = _indata()
    calls = _install_fast_run_fakes(monkeypatch, cfg=cfg, indata=indata)
    monkeypatch.setattr(driver, "_default_non_autodiff_solver_policy_for_backend", lambda _indata, _backend: ("accelerated", True))
    monkeypatch.setattr(driver, "_default_use_scan_for_backend", lambda _indata, _backend, _solver_mode: True)

    accelerated = driver.run_fixed_boundary(
        tmp_path / "input.synthetic",
        max_iter=2,
        verbose=False,
        grid=object(),
        jit_forces=False,
        jit_precompile=False,
        _auto_cli_fixed_boundary_mode=False,
    )

    assert accelerated.result.diagnostics["solver_mode"] == "accelerated"
    assert accelerated.result.diagnostics["accelerated_mode"] is True
    assert accelerated.result.diagnostics["accelerated_scan"] is True
    assert calls["solve_residual_iter"][-1][2]["use_scan"] is True

    parity = driver.run_fixed_boundary(
        tmp_path / "input.synthetic",
        solver_mode="parity",
        max_iter=2,
        verbose=False,
        grid=object(),
        jit_forces=False,
        jit_precompile=False,
        _auto_cli_fixed_boundary_mode=False,
    )

    assert parity.result.diagnostics["solver_mode"] == "parity"
    assert parity.result.diagnostics["accelerated_mode"] is False
    assert parity.result.diagnostics["accelerated_scan"] is False
    assert calls["solve_residual_iter"][-1][2]["use_scan"] is False
    assert calls["solve_residual_iter"][-1][2]["host_update_assembly"] is False


def test_direct_coil_free_boundary_exposes_limited_updates(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(lfreeb=True)
    indata = _indata(LFREEB=True, MGRID_FILE="DIRECT_COILS")
    calls = _install_fast_run_fakes(monkeypatch, cfg=cfg, indata=indata)

    driver.run_free_boundary(
        tmp_path / "input.direct",
        solver_mode="parity",
        max_iter=2,
        verbose=False,
        grid=object(),
        jit_forces=False,
        jit_precompile=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_static={"coil_geometry": object()},
        external_field_provider_params=object(),
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls["solve_residual_iter"][-1][2]["limit_update_rms"] is False

    driver.run_free_boundary(
        tmp_path / "input.direct",
        solver_mode="parity",
        max_iter=2,
        verbose=False,
        grid=object(),
        jit_forces=False,
        jit_precompile=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_static={"coil_geometry": object()},
        external_field_provider_params=object(),
        limit_update_rms=True,
        _auto_cli_fixed_boundary_mode=False,
    )

    assert calls["solve_residual_iter"][-1][2]["limit_update_rms"] is True


def test_run_free_boundary_rejects_fixed_input_and_delegates_free_input(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")

    monkeypatch.setattr(driver, "load_config", lambda _path: (_cfg(lfreeb=False), _indata()))
    with pytest.raises(ValueError, match="not a free-boundary case"):
        driver.run_free_boundary(input_path, verbose=False)

    captured = {}
    free_cfg = _cfg(lfreeb=True)
    monkeypatch.setattr(driver, "load_config", lambda _path: (free_cfg, _indata(LFREEB=True, MGRID_FILE=free_cfg.mgrid_file)))

    def fake_run_fixed_boundary(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return SimpleNamespace(kind="free-run")

    monkeypatch.setattr(driver, "run_fixed_boundary", fake_run_fixed_boundary)

    out = driver.run_free_boundary(input_path, verbose=False, max_iter=1)

    assert out.kind == "free-run"
    assert captured["path"] == input_path
    assert captured["kwargs"] == {"verbose": False, "max_iter": 1}


def test_cli_run_writes_wout_and_plot_mode_skips_solver_and_wout(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n  NITER = 4\n/\n")
    output_path = tmp_path / "nested" / "custom_wout.nc"
    calls: list[tuple] = []

    monkeypatch.setattr(cli, "read_indata", lambda _path: _indata(NITER=4))
    monkeypatch.setattr(cli, "default_non_autodiff_solver_policy", lambda _indata_arg: ("default", True))
    monkeypatch.setattr(cli, "_default_use_scan_for_backend", lambda _indata_arg, _backend, _mode: False)

    def fake_run_fixed_boundary(path, **kwargs):
        calls.append(("run", Path(path), kwargs))
        return SimpleNamespace(state=SimpleNamespace(Rcos=np.asarray([0.0])))

    def fake_write_wout(path, run, *, include_fsq):
        calls.append(("write", Path(path), run, include_fsq, Path(path).parent.exists()))
        Path(path).write_text("synthetic wout")

    monkeypatch.setattr(cli, "run_fixed_boundary", fake_run_fixed_boundary)
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", fake_write_wout)

    assert cli.main([str(input_path), "--output", str(output_path), "--quiet"]) == 0
    assert calls[0][0] == "run"
    assert calls[0][2]["cli_fixed_boundary_mode"] is True
    assert calls[1][0] == "write"
    assert calls[1][1] == output_path.resolve()
    assert calls[1][3] is True
    assert calls[1][4] is True
    assert output_path.read_text() == "synthetic wout"

    wout_path = tmp_path / "wout_plot.nc"
    wout_path.write_text("placeholder")
    plot_calls = []
    monkeypatch.setitem(
        sys.modules,
        "vmec_jax.plotting",
        SimpleNamespace(plot_wout=lambda path, *, outdir: plot_calls.append((path, outdir))),
    )
    monkeypatch.setattr(cli, "run_fixed_boundary", lambda *args, **kwargs: pytest.fail("plot mode should not solve"))
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: pytest.fail("plot mode should not write wout"))

    assert cli.main(["--plot", str(wout_path), "--outdir", str(tmp_path / "plots")]) == 0
    assert plot_calls == [(wout_path.resolve(), (tmp_path / "plots").resolve())]


def test_cli_errors_before_solver_for_conflicting_modes_and_missing_plot(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "input.case"
    input_path.write_text("&INDATA\n/\n")
    monkeypatch.setattr(cli, "run_fixed_boundary", lambda *args, **kwargs: pytest.fail("error path should not solve"))
    monkeypatch.setattr(cli, "write_wout_from_fixed_boundary_run", lambda *args, **kwargs: pytest.fail("error path should not write wout"))

    with pytest.raises(SystemExit) as conflicting_modes:
        cli.main([str(input_path), "--parity", "--fast"])
    assert conflicting_modes.value.code == 2

    with pytest.raises(SystemExit) as missing_plot:
        cli.main(["--plot", str(tmp_path / "missing_wout.nc")])
    assert missing_plot.value.code == 2

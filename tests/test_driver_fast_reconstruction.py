from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.energy import FluxProfiles
from vmec_jax.solve import SolveVmecResidualResult


class _Input:
    def __init__(self, **values):
        self.values = dict(values)

    def get_int(self, key, default=0):
        return int(self.values.get(key, default))

    def get_float(self, key, default=0.0):
        return float(self.values.get(key, default))

    def get_bool(self, key, default=False):
        return bool(self.values.get(key, default))


def test_solve_fixed_boundary_from_boundary_wires_solver_inputs(monkeypatch):
    state0 = object()
    solved_state = object()
    boundary = object()
    static = object()
    flux = SimpleNamespace(phipf=np.asarray([1.0]), chipf=np.asarray([2.0]), lamscale=3.0)
    pressure = np.asarray([4.0])
    captured = {}

    monkeypatch.setattr(
        driver,
        "initial_guess_from_boundary",
        lambda static_arg, boundary_arg, indata_arg, *, vmec_project: (
            captured.update(
                {
                    "initial_static": static_arg,
                    "initial_boundary": boundary_arg,
                    "initial_indata": indata_arg,
                    "vmec_project": vmec_project,
                }
            )
            or state0
        ),
    )

    def fake_solve(state_arg, static_arg, **kwargs):
        captured["solve_state"] = state_arg
        captured["solve_static"] = static_arg
        captured["solve_kwargs"] = dict(kwargs)
        return SimpleNamespace(state=solved_state)

    monkeypatch.setattr(driver, "solve_fixed_boundary_gd", fake_solve)
    indata = _Input(GAMMA=2.5)

    out = driver.solve_fixed_boundary_from_boundary(
        boundary=boundary,
        static=static,
        indata=indata,
        flux=flux,
        pressure=pressure,
        signgs=-1,
        max_iter=7,
        step_size=0.125,
        jacobian_penalty=9.0,
        jit_grad=True,
        differentiable=False,
        stop_grad_in_update=False,
        verbose=True,
        vmec_project=False,
    )

    assert out is solved_state
    assert captured["initial_static"] is static
    assert captured["initial_boundary"] is boundary
    assert captured["initial_indata"] is indata
    assert captured["vmec_project"] is False
    kwargs = captured["solve_kwargs"]
    assert captured["solve_state"] is state0
    assert captured["solve_static"] is static
    assert kwargs["phipf"] is flux.phipf
    assert kwargs["chipf"] is flux.chipf
    assert kwargs["signgs"] == -1
    assert kwargs["pressure"] is pressure
    assert kwargs["gamma"] == pytest.approx(2.5)
    assert kwargs["max_iter"] == 7
    assert kwargs["step_size"] == pytest.approx(0.125)
    assert kwargs["jacobian_penalty"] == pytest.approx(9.0)
    assert kwargs["jit_grad"] is True
    assert kwargs["differentiable"] is False
    assert kwargs["stop_grad_in_update"] is False
    assert kwargs["verbose"] is True


def test_residual_scalars_from_state_runs_kernel_pipeline(monkeypatch):
    import vmec_jax.kernels.forces as forces
    import vmec_jax.kernels.residue as residue
    import vmec_jax.kernels.tomnsp as tomnsp

    calls = {}
    state = object()
    indata = object()
    static = SimpleNamespace(
        cfg=SimpleNamespace(nfp=2, mpol=3, ntor=1, lasym=True, ntheta=5, nzeta=7, lconm1=False),
        s=np.asarray([0.0, 0.5, 1.0]),
    )

    def fake_trig_tables(**kwargs):
        calls["trig_kwargs"] = kwargs
        return "trig"

    monkeypatch.setattr(tomnsp, "vmec_trig_tables", fake_trig_tables)

    def fake_forces(**kwargs):
        calls["forces_kwargs"] = kwargs
        return SimpleNamespace(bc="bc")

    def fake_internal(kernels, **kwargs):
        calls["internal_kernels"] = kernels
        calls["internal_kwargs"] = kwargs
        return SimpleNamespace(
            frcc="frcc",
            frss="frss",
            fzsc="fzsc",
            fzcs="fzcs",
            flsc="flsc",
            flcs="flcs",
            frsc="frsc",
            frcs="frcs",
            fzcc="fzcc",
            fzss="fzss",
            flcc="flcc",
            flss="flss",
        )

    def fake_norms(**kwargs):
        calls["norm_kwargs"] = kwargs
        return "norms"

    def fake_fsq(**kwargs):
        calls["fsq_kwargs"] = kwargs
        return SimpleNamespace(fsqr=1.25, fsqz=2.5, fsql=3.75)

    monkeypatch.setattr(forces, "vmec_forces_rz_from_wout", fake_forces)
    monkeypatch.setattr(forces, "vmec_residual_internal_from_kernels", fake_internal)
    monkeypatch.setattr(residue, "vmec_force_norms_from_bcovar_dynamic", fake_norms)
    monkeypatch.setattr(residue, "vmec_fsq_from_tomnsps_dynamic", fake_fsq)

    scalars = driver.residual_scalars_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=-1,
        use_vmec_synthesis=False,
    )

    assert scalars == (1.25, 2.5, 3.75)
    assert calls["trig_kwargs"] == {
        "ntheta": 5,
        "nzeta": 7,
        "nfp": 2,
        "mmax": 2,
        "nmax": 1,
        "lasym": True,
    }
    assert calls["forces_kwargs"]["state"] is state
    assert calls["forces_kwargs"]["static"] is static
    assert calls["forces_kwargs"]["indata"] is indata
    assert calls["forces_kwargs"]["use_vmec_synthesis"] is False
    assert calls["forces_kwargs"]["wout"].signgs == -1
    assert calls["internal_kwargs"]["cfg_ntheta"] == 5
    assert calls["internal_kwargs"]["cfg_nzeta"] == 7
    assert calls["norm_kwargs"] == {"bc": "bc", "trig": "trig", "s": static.s, "signgs": -1}
    assert calls["fsq_kwargs"]["norms"] == "norms"
    assert calls["fsq_kwargs"]["lconm1"] is False


def test_final_flux_profiles_from_state_recomputes_current_driven_iota(monkeypatch):
    import vmec_jax.kernels.bcovar as bcovar
    import vmec_jax.kernels.residue as residue
    import vmec_jax.kernels.tomnsp as tomnsp
    import vmec_jax.wout as wout_module

    calls = {}
    ns = 3
    static = SimpleNamespace(
        s=np.asarray([0.0, 0.5, 1.0]),
        modes=SimpleNamespace(m=np.asarray([0, 1]), n=np.asarray([0, 0])),
        cfg=SimpleNamespace(nfp=1, mpol=2, ntor=0, lasym=False),
        grid=SimpleNamespace(theta=np.asarray([0.0, 1.0]), zeta=np.asarray([0.0, 1.0])),
        trig_vmec=None,
    )
    state = SimpleNamespace(
        Rcos=np.zeros((ns, 1)),
        Rsin=np.zeros((ns, 1)),
        Zcos=np.zeros((ns, 1)),
        Zsin=np.zeros((ns, 1)),
        Lcos=np.zeros((ns, 1)),
        Lsin=np.zeros((ns, 1)),
    )
    flux_in = FluxProfiles(
        phipf=np.asarray([0.0, 1.0, 2.0]),
        chipf=np.asarray([0.0, 5.0, 7.0]),
        phips=np.asarray([0.0, 2.0, 4.0]),
        signgs=-1,
        lamscale=np.asarray(9.0),
    )
    pressure = np.asarray([100.0, 3.0, 4.0])

    monkeypatch.setattr(
        driver,
        "boundary_from_indata",
        lambda _indata, _modes: BoundaryCoeffs(
            R_cos=np.asarray([2.0, 0.0]),
            R_sin=np.zeros(2),
            Z_cos=np.zeros(2),
            Z_sin=np.zeros(2),
        ),
    )
    def fake_trig_tables(**kwargs):
        calls["trig"] = kwargs
        return "trig"

    monkeypatch.setattr(tomnsp, "vmec_trig_tables", fake_trig_tables)
    monkeypatch.setattr(wout_module, "_icurv_full_mesh_from_indata", lambda **_kwargs: np.asarray([0.0, 4.0, 8.0]))
    monkeypatch.setattr(wout_module, "_chipf_from_chips", lambda chips: np.asarray(chips) + 10.0)
    monkeypatch.setattr(driver, "_iotaf_from_iotas", lambda iotas, *, lrfp: np.asarray(iotas) + (1.0 if lrfp else 0.5))
    monkeypatch.setattr(residue, "vmec_pwint_from_trig", lambda _trig, *, ns, nzeta: np.ones((ns, 2, nzeta)))

    def fake_bcovar(**kwargs):
        calls["bcovar_kwargs"] = kwargs
        shape = (ns, 2, 2)
        return SimpleNamespace(
            jac=SimpleNamespace(sqrtg=np.ones(shape)),
            guu=np.ones(shape),
            guv=np.zeros(shape),
            bsupu=np.zeros(shape),
            bsupv=np.zeros(shape),
        )

    monkeypatch.setattr(bcovar, "vmec_bcovar_half_mesh_from_wout", fake_bcovar)

    flux_out, prof_out = driver._final_flux_profiles_from_state(
        indata=_Input(NCURR=1, GAMMA=2.0, LRFP=False),
        static_in=static,
        state=state,
        signgs=-1,
        flux_local=flux_in,
        prof_local={"pressure": pressure, "keep": "value"},
        pressure_local=pressure,
    )

    assert calls["trig"]["ntheta"] == 2
    assert calls["trig"]["nzeta"] == 2
    assert calls["bcovar_kwargs"]["use_vmec_synthesis"] is True
    np.testing.assert_allclose(np.asarray(prof_out["iota"]), [0.0, 0.5, 0.5])
    np.testing.assert_allclose(np.asarray(prof_out["iotaf"]), [0.5, 1.0, 1.0])
    np.testing.assert_allclose(np.asarray(flux_out.chipf), [10.0, 11.0, 12.0])
    np.testing.assert_allclose(np.asarray(flux_out.phips), [0.0, 2.0, 4.0])
    assert prof_out["keep"] == "value"
    assert flux_out.signgs == -1


def test_final_flux_profiles_from_state_fast_exit_branches(monkeypatch):
    flux = object()
    prof = {"iota": np.asarray([1.0])}
    state = SimpleNamespace(Rcos=np.zeros((2, 1)))
    static = SimpleNamespace(s=np.asarray([0.0, 0.5, 1.0]))

    assert (
        driver._final_flux_profiles_from_state(
            indata=_Input(NCURR=0),
            static_in=static,
            state=state,
            signgs=1,
            flux_local=flux,
            prof_local=prof,
            pressure_local=np.asarray([0.0, 1.0, 2.0]),
        )
        == (flux, prof)
    )

    monkeypatch.setenv("VMEC_JAX_DISABLE_WOUT_NCURR_RECOMPUTE", "1")
    assert (
        driver._final_flux_profiles_from_state(
            indata=_Input(NCURR=1),
            static_in=static,
            state=state,
            signgs=1,
            flux_local=flux,
            prof_local=prof,
            pressure_local=np.asarray([0.0, 1.0, 2.0]),
        )
        == (flux, prof)
    )
    monkeypatch.delenv("VMEC_JAX_DISABLE_WOUT_NCURR_RECOMPUTE")
    assert (
        driver._final_flux_profiles_from_state(
            indata=_Input(NCURR=1),
            static_in=static,
            state=state,
            signgs=1,
            flux_local=flux,
            prof_local=prof,
            pressure_local=np.asarray([0.0, 1.0, 2.0]),
        )
        == (flux, prof)
    )


def test_final_flux_profiles_from_state_handles_no_jax_and_empty_axis(monkeypatch):
    import vmec_jax.kernels.bcovar as bcovar
    import vmec_jax.kernels.residue as residue
    import vmec_jax.wout as wout_module

    class _BadArray:
        def __array__(self, dtype=None):
            del dtype
            raise TypeError("synthetic conversion failure")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jax":
            raise RuntimeError("jax unavailable")
        return original_import(name, *args, **kwargs)

    ns = 0
    static = SimpleNamespace(
        s=np.asarray([], dtype=float),
        modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0])),
        cfg=SimpleNamespace(nfp=1, mpol=1, ntor=0, lasym=False),
        grid=SimpleNamespace(theta=np.asarray([0.0]), zeta=np.asarray([0.0])),
        trig_vmec=object(),
    )
    state = SimpleNamespace(
        Rcos=_BadArray(),
        Rsin=np.zeros((ns, 1)),
        Zcos=np.zeros((ns, 1)),
        Zsin=np.zeros((ns, 1)),
        Lcos=np.zeros((ns, 1)),
        Lsin=np.zeros((ns, 1)),
    )
    flux = FluxProfiles(
        phipf=np.asarray([], dtype=float),
        chipf=np.asarray([], dtype=float),
        phips=np.asarray([], dtype=float),
        signgs=1,
        lamscale=np.asarray(1.0),
    )

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(driver, "boundary_from_indata", lambda *_args, **_kwargs: SimpleNamespace(R_cos=np.asarray([1.0])))
    monkeypatch.setattr(wout_module, "_icurv_full_mesh_from_indata", lambda **_kwargs: np.asarray([], dtype=float))
    monkeypatch.setattr(wout_module, "_chipf_from_chips", lambda chips: np.asarray(chips, dtype=float))
    monkeypatch.setattr(driver, "_iotaf_from_iotas", lambda iotas, *, lrfp: np.asarray(iotas, dtype=float))
    monkeypatch.setattr(residue, "vmec_pwint_from_trig", lambda *_args, **_kwargs: np.zeros((ns, 1, 1)))
    monkeypatch.setattr(
        bcovar,
        "vmec_bcovar_half_mesh_from_wout",
        lambda **_kwargs: SimpleNamespace(
            jac=SimpleNamespace(sqrtg=np.zeros((ns, 1, 1))),
            guu=np.zeros((ns, 1, 1)),
            guv=np.zeros((ns, 1, 1)),
            bsupu=np.zeros((ns, 1, 1)),
            bsupv=np.zeros((ns, 1, 1)),
        ),
    )

    flux_out, prof_out = driver._final_flux_profiles_from_state(
        indata=_Input(NCURR=1),
        static_in=static,
        state=state,
        signgs=1,
        flux_local=flux,
        prof_local={"pressure": np.asarray([], dtype=float)},
        pressure_local=np.asarray([], dtype=float),
    )

    assert flux_out.chipf.size == 0
    assert prof_out["iota"].size == 0
    assert prof_out["iotaf"].size == 0


def test_write_wout_from_fixed_boundary_run_creates_parent_and_delegates(monkeypatch, tmp_path):
    import vmec_jax.wout as wout_module

    run = object()
    wout = SimpleNamespace(marker="wout")
    captured = {}

    def fake_from_run(run_arg, **kwargs):
        captured["from_run"] = run_arg
        captured["kwargs"] = kwargs
        return wout

    def fake_write(path, wout_arg, *, overwrite):
        captured["write_path"] = Path(path)
        captured["wout"] = wout_arg
        captured["overwrite"] = overwrite

    monkeypatch.setattr(driver, "wout_from_fixed_boundary_run", fake_from_run)
    monkeypatch.setattr(wout_module, "write_wout", fake_write)

    out_path = tmp_path / "nested" / "wout_test.nc"
    assert driver.write_wout_from_fixed_boundary_run(out_path, run, include_fsq=False, fast_bcovar=True) is wout
    assert out_path.parent.exists()
    assert captured["from_run"] is run
    assert captured["kwargs"] == {"include_fsq": False, "path": out_path, "fast_bcovar": True}
    assert captured["write_path"] == out_path
    assert captured["wout"] is wout
    assert captured["overwrite"] is True


def test_load_example_and_wrappers_use_lightweight_dependencies(monkeypatch, tmp_path):
    cfg = SimpleNamespace(ns=5)
    indata = object()
    static = object()
    wout = object()
    state = object()
    calls = {}
    data_dir = tmp_path / "examples" / "data"
    data_dir.mkdir(parents=True)
    input_path = data_dir / "input.case"
    wout_path = data_dir / "wout_case_reference.nc"
    input_path.write_text("&INDATA\n/\n")
    wout_path.write_text("not really netcdf")

    def fake_load_config(path):
        calls["load_config_path"] = path
        return cfg, indata

    def fake_prepare_mgrid_for_config(cfg_arg, **kwargs):
        calls["mgrid"] = (cfg_arg, kwargs)
        return None

    def fake_build_static(cfg_arg, **kwargs):
        calls["static_kwargs"] = kwargs
        return static

    def fake_read_wout(path):
        calls["read_wout_path"] = Path(path)
        return wout

    def fake_state_from_wout(wout_arg):
        calls["state_wout"] = wout_arg
        return state

    monkeypatch.setattr(driver, "load_config", fake_load_config)
    monkeypatch.setattr(driver, "prepare_mgrid_for_config", fake_prepare_mgrid_for_config)
    monkeypatch.setattr(driver, "build_static", fake_build_static)
    monkeypatch.setattr(driver, "read_wout", fake_read_wout)
    monkeypatch.setattr(driver, "state_from_wout", fake_state_from_wout)

    loaded = driver.load_example("case", root=tmp_path, with_wout=True, grid="grid")

    assert loaded.input_path == input_path
    assert loaded.wout_path == wout_path
    assert loaded.cfg is cfg
    assert loaded.indata is indata
    assert loaded.static is static
    assert loaded.wout is wout
    assert loaded.state is state
    assert calls["load_config_path"] == str(input_path)
    assert "mgrid" not in calls
    assert calls["static_kwargs"] == {"grid": "grid", "mgrid_metadata": None, "free_boundary_extcur": None}
    assert calls["read_wout_path"] == wout_path
    assert calls["state_wout"] is wout

    assert driver.load_input(input_path) == (cfg, indata)
    assert driver.load_wout(wout_path) is wout

    no_wout = driver.load_example("case", root=tmp_path, with_wout=False)
    assert no_wout.wout is None
    assert no_wout.state is None


def test_result_with_diag_returns_copy_without_mutating_original():
    result = SolveVmecResidualResult(
        state="state",
        n_iter=2,
        w_history=np.asarray([1.0, 0.5]),
        fsqr2_history=np.asarray([2.0]),
        fsqz2_history=np.asarray([3.0]),
        fsql2_history=np.asarray([4.0]),
        grad_rms_history=np.asarray([5.0]),
        step_history=np.asarray([6.0]),
        diagnostics={"kept": True},
    )

    updated = driver._result_with_diag(result, added=7)

    assert updated is not result
    assert updated.state == "state"
    assert updated.n_iter == 2
    assert updated.diagnostics == {"kept": True, "added": 7}
    assert result.diagnostics == {"kept": True}
    np.testing.assert_allclose(updated.w_history, [1.0, 0.5])


def test_finalize_fixed_boundary_solver_run_attaches_public_diagnostics(monkeypatch):
    result = SolveVmecResidualResult(
        state="solved-state",
        n_iter=3,
        w_history=np.asarray([2.0, 1.0]),
        fsqr2_history=np.asarray([0.2]),
        fsqz2_history=np.asarray([0.3]),
        fsql2_history=np.asarray([0.4]),
        grad_rms_history=np.asarray([0.5]),
        step_history=np.asarray([0.6]),
        diagnostics={"existing": "diag"},
    )
    calls: dict[str, object] = {}

    def fake_summary(result_arg, *, solver, verbose):
        calls["summary"] = (result_arg, solver, verbose)

    def fake_finalize_flux_profiles_for_run(**kwargs):
        calls["flux_finalize"] = kwargs
        return "static-out", "flux-out", {"iota": np.asarray([0.0, 0.4])}

    def fake_finish(run_arg, *, initial_policy, enabled):
        calls["finish"] = (run_arg, initial_policy, enabled)
        return run_arg

    monkeypatch.setattr(driver._driver_solve_helpers, "maybe_print_optimizer_summary", fake_summary)
    monkeypatch.setattr(driver._driver_flux_helpers, "finalize_flux_profiles_for_run", fake_finalize_flux_profiles_for_run)

    run = driver._finalize_fixed_boundary_solver_run(
        cfg="cfg",
        indata="indata",
        static="static-in",
        result=result,
        signgs=-1,
        flux_local="flux-local",
        profiles_local={"pressure": np.asarray([1.0])},
        pressure_local=np.asarray([1.0]),
        static_profile_cache="cache",
        solver="vmec2000_iter",
        verbose=True,
        finish_policy_eff="converge",
        cli_fixed_boundary_finish_enabled=True,
        multigrid=True,
        ns_stages=[5, 9],
        maybe_finish_cli_fixed_boundary_run=fake_finish,
    )

    assert calls["summary"] == (result, "vmec2000_iter", True)
    assert calls["flux_finalize"]["final_flux_profiles_from_state_func"] is driver._final_flux_profiles_from_state
    assert run.cfg == "cfg"
    assert run.indata == "indata"
    assert run.static == "static-out"
    assert run.state == "solved-state"
    assert run.flux == "flux-out"
    assert run.profiles["iota"].tolist() == [0.0, 0.4]
    assert run.signgs == -1
    assert run.result is not result
    assert run.result.diagnostics["existing"] == "diag"
    assert run.result.diagnostics["fixed_boundary_finish_policy"] == "converge"
    assert run.result.diagnostics["cli_fixed_boundary_finish_enabled"] is True
    assert calls["finish"][0] is run
    assert calls["finish"][1:] == ("multigrid", True)

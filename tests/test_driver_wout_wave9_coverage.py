from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
import vmec_jax.solve as solve_module
import vmec_jax.wout as wout
from vmec_jax.energy import FluxProfiles
from vmec_jax.solve import SolveVmecResidualResult
from vmec_jax.wout import MU0, WoutData


class _RaisingModes:
    @property
    def m(self):
        raise RuntimeError("synthetic mode table failure")

    n = np.asarray([0], dtype=int)


def _write_input(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "input.wave9"
    path.write_text("&INDATA\n" + body + "/\n")
    return path


def _fake_state(ns: int = 3):
    arr = np.zeros((int(ns), 1), dtype=float)
    return SimpleNamespace(
        layout=SimpleNamespace(ns=int(ns)),
        Rcos=arr,
        Rsin=arr,
        Zcos=arr,
        Zsin=arr,
        Lcos=arr,
        Lsin=arr,
    )


def _patch_light_driver(monkeypatch, *, raising_modes: bool = False) -> None:
    def fake_static(cfg, **_kwargs):
        modes = _RaisingModes() if raising_modes else SimpleNamespace(m=np.asarray([0]), n=np.asarray([0]), K=1)
        return SimpleNamespace(
            cfg=cfg,
            modes=modes,
            s=np.linspace(0.0, 1.0, int(cfg.ns)),
            grid=SimpleNamespace(theta=np.asarray([0.0]), zeta=np.asarray([0.0]), ntheta=1, nzeta=1),
            trig_vmec=None,
        )

    monkeypatch.setenv("VMEC_JAX_DISABLE_JIT_INIT", "1")
    monkeypatch.setenv("VMEC_JAX_COMPILATION_CACHE", "0")
    monkeypatch.setattr(driver, "_default_backend_name", lambda: "cpu")
    monkeypatch.setattr(driver, "validate_free_boundary_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "prepare_mgrid_for_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(driver, "build_static", fake_static)
    monkeypatch.setattr(driver, "boundary_from_indata", lambda *args, **kwargs: object())
    monkeypatch.setattr(driver, "initial_guess_from_boundary", lambda static, *_args, **_kwargs: _fake_state(static.cfg.ns))
    monkeypatch.setattr(driver, "interp_vmec_state", lambda *_args, ns_new, **_kwargs: _fake_state(ns_new))
    monkeypatch.setattr(
        driver,
        "flux_profiles_from_indata",
        lambda _indata, s, *, signgs: FluxProfiles(
            phipf=np.ones_like(np.asarray(s, dtype=float)),
            chipf=np.zeros_like(np.asarray(s, dtype=float)),
            phips=np.ones_like(np.asarray(s, dtype=float)),
            signgs=int(signgs),
            lamscale=np.asarray(1.0),
        ),
    )
    monkeypatch.setattr(driver, "eval_profiles", lambda _indata, s: {"pressure": np.zeros_like(np.asarray(s))})


def _solve_result(state, *, max_iter: int, fsq: float, converged: bool, diagnostics: dict | None = None):
    diag = {
        "converged": bool(converged),
        "final_fsqr": float(fsq),
        "final_fsqz": 0.0,
        "final_fsql": 0.0,
        "resume_state": {"time_step": 0.25, "inv_tau": [1.0], "iter_offset": 2},
    }
    if diagnostics:
        diag.update(diagnostics)
    return SolveVmecResidualResult(
        state=state,
        n_iter=max(0, int(max_iter) - 1),
        w_history=np.asarray([float(fsq)], dtype=float),
        fsqr2_history=np.asarray([float(fsq)], dtype=float),
        fsqz2_history=np.asarray([0.0], dtype=float),
        fsql2_history=np.asarray([0.0], dtype=float),
        grad_rms_history=np.asarray([], dtype=float),
        step_history=np.asarray([], dtype=float),
        diagnostics=diag,
    )


def test_driver_budget_helpers_cover_small_remainder_and_stage_edges() -> None:
    assert driver._allocate_integer_budget(total=2, weights=[1, 1, 10]) == [0, 0, 2]
    assert driver._accelerated_cli_budgeted_total_iters(total_budget=9, ns_stages=[12, 6]) == 9
    assert driver._accelerated_cli_budgeted_stage_iters(total_budget=3, ns_stages=[8, 8, 8]) == [3, 0, 1]
    assert driver._distribute_stage_iters(iters=3, nstep=5) == [3]
    assert driver._distribute_stage_iters(iters=7, nstep=3) == [3, 2, 2]


def test_driver_jit_auto_exception_path_returns_true(monkeypatch, tmp_path: Path) -> None:
    input_path = _write_input(
        tmp_path,
        """
  LFREEB = F
  NFP = 1
  MPOL = 2
  NTOR = 0
  NS = 3
  NITER = 1
  FTOL = 1e-9
  PHIEDGE = 1.0
  RBC(0,0) = 1.0
  ZBS(1,0) = 0.1
""",
    )
    calls: list[dict[str, object]] = []

    def fake_solver(state, static, **kwargs):
        calls.append({"jit_forces": bool(kwargs["jit_forces"]), "ns": int(static.cfg.ns)})
        return _solve_result(state, max_iter=kwargs["max_iter"], fsq=1.0e-12, converged=True)

    _patch_light_driver(monkeypatch, raising_modes=True)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    driver.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="parity",
        max_iter=1,
        multigrid=False,
        verbose=False,
        jit_forces="auto",
    )

    assert calls == [{"jit_forces": True, "ns": 3}]


def test_cli_finisher_sanitizes_minimal_resume_and_resolves_finish_jit(monkeypatch, tmp_path: Path) -> None:
    input_path = _write_input(
        tmp_path,
        """
  LFREEB = F
  NFP = 1
  MPOL = 2
  NTOR = 0
  NS = 3
  NITER = 2
  FTOL = 1e-12
  PHIEDGE = 1.0
  RBC(0,0) = 1.0
  ZBS(1,0) = 0.1
""",
    )
    calls: list[dict[str, object]] = []

    def fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "jit_forces": bool(kwargs["jit_forces"]),
                "resume_state_mode": kwargs.get("resume_state_mode"),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        # Initial accelerated solve misses strict FTOL. The first finish attempt
        # improves but keeps an invalid flip_sign so the sanitizer drops it.
        fsq = 1.0e-6 if idx == 0 else 1.0e-13
        return _solve_result(
            state,
            max_iter=kwargs["max_iter"],
            fsq=fsq,
            converged=idx > 0,
            diagnostics={
                "resume_state": {
                    "time_step": "0.125",
                    "inv_tau": np.asarray([4.0, 5.0]),
                    "iter_offset": "7",
                    "vmec2000_cache_valid": 1,
                    "flip_sign": object(),
                    "cache_precond_diag": np.ones(2),
                }
            },
        )

    _patch_light_driver(monkeypatch, raising_modes=False)
    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=2,
        multigrid=False,
        verbose=False,
        cli_fixed_boundary_mode=True,
        jit_forces="definitely-on",
        finish_policy="converge",
    )

    assert [call["jit_forces"] for call in calls] == [True, True]
    assert [call["resume_state_mode"] for call in calls] == ["minimal", "minimal"]
    diag = run.result.diagnostics
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["accelerated"]
    assert diag["resume_state"] == {
        "time_step": 0.125,
        "inv_tau": [4.0, 5.0],
        "iter_offset": 7,
        "vmec2000_cache_valid": True,
    }


def _minimal_wout(path: Path, *, ns: int = 3) -> WoutData:
    mn = 1
    profile = np.arange(ns, dtype=float)
    main = profile[:, None]
    return WoutData(
        path=path,
        ns=ns,
        mpol=1,
        ntor=0,
        nfp=1,
        lasym=False,
        signgs=1,
        mnmax=mn,
        mpol_nyq=0,
        ntor_nyq=0,
        mnmax_nyq=mn,
        xm=np.asarray([0]),
        xn=np.asarray([0]),
        xm_nyq=np.asarray([0]),
        xn_nyq=np.asarray([0]),
        rmnc=1.0 + main,
        rmns=np.zeros_like(main),
        zmnc=np.zeros_like(main),
        zmns=0.1 + main,
        lmnc=2.0 + main,
        lmns=3.0 + main,
        phipf=np.ones(ns),
        chipf=np.zeros(ns),
        phips=np.zeros(ns),
        iotaf=np.zeros(ns),
        iotas=np.zeros(ns),
        gmnc=np.zeros_like(main),
        gmns=np.zeros_like(main),
        bsupumnc=np.zeros_like(main),
        bsupumns=np.zeros_like(main),
        bsupvmnc=np.zeros_like(main),
        bsupvmns=np.zeros_like(main),
        bsubumnc=np.zeros_like(main),
        bsubumns=np.zeros_like(main),
        bsubvmnc=np.zeros_like(main),
        bsubvmns=np.zeros_like(main),
        bsubsmns=np.zeros_like(main),
        bsubsmnc=np.zeros_like(main),
        bmnc=np.zeros_like(main),
        bmns=np.zeros_like(main),
        wb=0.0,
        volume_p=0.0,
        gamma=0.0,
        wp=0.0,
        vp=np.ones(ns),
        pres=MU0 * np.zeros(ns),
        presf=MU0 * np.zeros(ns),
        fsqr=0.0,
        fsqz=0.0,
        fsql=0.0,
        fsqt=np.zeros(0),
        equif=np.zeros(ns),
        phi=np.zeros(ns),
        buco=np.zeros(ns),
        bvco=np.zeros(ns),
        jcuru=np.zeros(ns),
        jcurv=np.zeros(ns),
        raxis_cc=np.zeros(1),
        zaxis_cs=np.zeros(1),
        raxis_cs=np.zeros(1),
        zaxis_cc=np.zeros(1),
        Aminor_p=0.0,
        Rmajor_p=0.0,
        aspect=0.0,
        betatotal=0.0,
        betapol=0.0,
        betator=0.0,
        betaxis=0.0,
        ctor=0.0,
        DMerc=np.zeros(ns),
        Dshear=np.zeros(ns),
        Dwell=np.zeros(ns),
        Dcurr=np.zeros(ns),
        Dgeod=np.zeros(ns),
        jdotb=np.zeros(ns),
        bdotb=np.zeros(ns),
        bdotgradv=np.zeros(ns),
        ac=np.zeros(0),
        ac_aux_s=np.zeros(0),
        ac_aux_f=np.zeros(0),
        pcurr_type="",
        piota_type="",
    )


def test_wout_single_surface_lambda_state_roundtrip_uses_raw_lambda_copy(tmp_path: Path) -> None:
    src = _minimal_wout(tmp_path / "single.nc", ns=1)
    src = src.__class__(
        **{
            **src.__dict__,
            "signgs": 0,
            "lmnc": np.asarray([[11.0]]),
            "lmns": np.asarray([[-7.0]]),
            "phipf": np.asarray([4.0]),
        }
    )

    state = wout.state_from_wout(src)

    np.testing.assert_allclose(state.Lcos, [[11.0]])
    np.testing.assert_allclose(state.Lsin, [[-7.0]])
    np.testing.assert_allclose(state.Rcos, src.rmnc)


def test_wout_read_rejects_fully_masked_mode_table(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "wout_masked_modes.nc"

    with netcdf4.Dataset(path, mode="w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("radius", 1)
        ds.createDimension("mn_mode", 1)
        ds.createDimension("mn_mode_nyq", 1)
        ds.createVariable("ns", "i4", ())[:] = 1
        ds.createVariable("mpol", "i4", ())[:] = 1
        ds.createVariable("ntor", "i4", ())[:] = 0
        ds.createVariable("nfp", "i4", ())[:] = 1
        xm = ds.createVariable("xm", "f8", ("mn_mode",), fill_value=-999.0)
        xm[:] = np.ma.masked_all((1,), dtype=float)
        for name in ("xn", "xm_nyq", "xn_nyq"):
            ds.createVariable(name, "f8", ("mn_mode" if name == "xn" else "mn_mode_nyq",))[:] = [0.0]

    with pytest.raises(ValueError, match=r"masked wout mode metadata \(xm\)"):
        wout.read_wout(path)


def test_wout_write_propagates_fill_off_when_debug_raise_enabled(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")
    monkeypatch.setenv("VMEC_JAX_MERCIER_RAISE", "1")

    class DatasetBoom:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def set_fill_off(self):
            raise RuntimeError("fill disabled failure")

    monkeypatch.setattr(wout.netCDF4 if hasattr(wout, "netCDF4") else __import__("netCDF4"), "Dataset", DatasetBoom)

    with pytest.raises(RuntimeError, match="fill disabled failure"):
        wout.write_wout(tmp_path / "wout_boom.nc", _minimal_wout(tmp_path / "src.nc"), overwrite=True)


def test_bsubs_coeff_helpers_return_none_for_shape_and_singular_failures(monkeypatch) -> None:
    trig_false = wout.vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=2, nmax=1, lasym=False, cache=False)
    nt2 = int(trig_false.ntheta2)
    nzeta = int(np.asarray(trig_false.cosnv).shape[0])
    assert (
        wout._jxbforce_getbsubs_coeffs_lasym_false(
            frho=np.zeros((nt2 + 1, nzeta)),
            bsupu=np.zeros((nt2, nzeta)),
            bsupv=np.zeros((nt2, nzeta)),
            trig=trig_false,
            nfp=1,
        )
        is None
    )

    trig_true = wout.vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=2, nmax=0, lasym=True, cache=False)
    nt3 = int(trig_true.ntheta3)
    assert (
        wout._jxbforce_getbsubs_coeffs_lasym_true(
            frho=np.zeros((nt3, 1)),
            bsupu=np.zeros((nt3 + 1, 1)),
            bsupv=np.zeros((nt3, 1)),
            trig=trig_true,
            nfp=1,
        )
        is None
    )

    def fail_lstsq(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced lstsq failure")

    monkeypatch.setattr(wout.np.linalg, "solve", lambda *_args, **_kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError("forced solve failure")))
    monkeypatch.setattr(wout.np.linalg, "lstsq", fail_lstsq)

    assert (
        wout._jxbforce_getbsubs_coeffs_lasym_false(
            frho=np.ones((nt2, nzeta)),
            bsupu=np.ones((nt2, nzeta)),
            bsupv=np.ones((nt2, nzeta)),
            trig=trig_false,
            nfp=1,
        )
        is None
    )

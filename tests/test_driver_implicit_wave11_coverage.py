from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
import vmec_jax.implicit as implicit
import vmec_jax.solve as solve_module
from vmec_jax._compat import has_jax
from vmec_jax.energy import FluxProfiles
from vmec_jax.solve import SolveVmecResidualResult
from vmec_jax.state import StateLayout, VMECState


def _write_input(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "input.wave11"
    path.write_text("&INDATA\n" + body + "/\n")
    return path


def _fake_state(ns: int = 3, K: int = 1) -> SimpleNamespace:
    arr = np.zeros((int(ns), int(K)), dtype=float)
    return SimpleNamespace(
        layout=SimpleNamespace(ns=int(ns), K=int(K), size=6 * int(ns) * int(K)),
        Rcos=arr,
        Rsin=arr,
        Zcos=arr,
        Zsin=arr,
        Lcos=arr,
        Lsin=arr,
    )


def _patch_light_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_static(cfg, **_kwargs):
        return SimpleNamespace(
            cfg=cfg,
            modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0]), K=1),
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
    monkeypatch.setattr(
        driver, "initial_guess_from_boundary", lambda static, *_args, **_kwargs: _fake_state(static.cfg.ns)
    )
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


def _solve_result(state, *, fsq: float, converged: bool, n_iter: int = 0, diagnostics: dict | None = None):
    diag = {
        "converged": bool(converged),
        "ftol": 1.0e-12,
        "final_fsqr": float(fsq),
        "final_fsqz": 0.0,
        "final_fsql": 0.0,
        "resume_state": {"time_step": 0.25, "iter_offset": 1},
    }
    if diagnostics:
        diag.update(diagnostics)
    return SolveVmecResidualResult(
        state=state,
        n_iter=int(n_iter),
        w_history=np.asarray([float(fsq)], dtype=float),
        fsqr2_history=np.asarray([float(fsq)], dtype=float),
        fsqz2_history=np.asarray([0.0], dtype=float),
        fsql2_history=np.asarray([0.0], dtype=float),
        grad_rms_history=np.asarray([], dtype=float),
        step_history=np.asarray([], dtype=float),
        diagnostics=diag,
    )


def _basic_input(tmp_path: Path, *, niter: int = 2) -> Path:
    return _write_input(
        tmp_path,
        f"""
  LFREEB = F
  NFP = 1
  MPOL = 2
  NTOR = 0
  NS = 3
  NITER = {int(niter)}
  FTOL = 1e-12
  PHIEDGE = 1.0
  RBC(0,0) = 1.0
  ZBS(1,0) = 0.1
""",
    )


def test_cli_finisher_marks_strict_residual_converged_even_if_solver_flag_is_false(monkeypatch, tmp_path: Path) -> None:
    _patch_light_driver(monkeypatch)
    input_path = _basic_input(tmp_path, niter=2)
    calls = []

    def fake_solver(state, static, **kwargs):
        calls.append(kwargs)
        return _solve_result(state, fsq=1.0e-14, converged=False)

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
    )

    assert len(calls) == 1
    diag = run.result.diagnostics
    assert diag["converged"] is True
    assert diag["converged_strict"] is True
    assert diag["cli_fixed_boundary_full_parity_fallback"] is False
    assert np.asarray(diag["cli_fixed_boundary_finish_budgets"]).size == 0


def test_cli_finisher_records_budget_cap_exhaustion_after_accelerated_and_parity_attempts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_light_driver(monkeypatch)
    input_path = _basic_input(tmp_path, niter=1)
    fsq_by_call = [1.0e-4, 1.0e-4, 1.0e-4]
    calls: list[dict[str, object]] = []

    def fake_solver(state, static, **kwargs):
        idx = len(calls)
        calls.append(
            {
                "max_iter": int(kwargs["max_iter"]),
                "resume_state_mode": kwargs.get("resume_state_mode"),
                "fsq_total_target": kwargs.get("fsq_total_target"),
                "use_scan": bool(kwargs["use_scan"]),
            }
        )
        return _solve_result(state, fsq=fsq_by_call[min(idx, len(fsq_by_call) - 1)], converged=False)

    monkeypatch.setattr(driver, "solve_fixed_boundary_residual_iter", fake_solver)
    monkeypatch.setattr(solve_module, "solve_fixed_boundary_residual_iter", fake_solver)

    run = driver.run_fixed_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="accelerated",
        max_iter=1,
        multigrid=False,
        verbose=False,
        cli_fixed_boundary_mode=True,
        use_scan=False,
        finish_policy="converge",
    )

    assert [call["max_iter"] for call in calls] == [1, 1, 1]
    assert calls[1]["resume_state_mode"] == "minimal"
    assert calls[1]["fsq_total_target"] == pytest.approx(3.0e-12)
    assert calls[2]["resume_state_mode"] == "full"
    assert calls[2]["fsq_total_target"] is None
    diag = run.result.diagnostics
    np.testing.assert_array_equal(diag["cli_fixed_boundary_finish_budgets"], [1, 1])
    assert np.asarray(diag["cli_fixed_boundary_finish_modes"]).tolist() == ["accelerated", "parity"]
    assert diag["cli_fixed_boundary_finish_budget_cap"] == 2
    assert diag["cli_fixed_boundary_finish_budget_exhausted"] is True
    assert diag["converged"] is False


@pytest.mark.skipif(not has_jax(), reason="implicit wrappers require JAX")
def test_implicit_reexported_residual_adjoint_wrappers_delegate_and_preserve_options(monkeypatch) -> None:
    from vmec_jax._compat import jnp

    calls = {}

    def fake_lineax_impl(matvec, b, *, x0, tol, max_iter, lineax_module, jax_module):
        calls["lineax"] = (
            matvec(jnp.asarray([2.0])),
            np.asarray(b),
            np.asarray(x0),
            tol,
            max_iter,
            lineax_module,
            jax_module,
        )
        return jnp.asarray([3.0]), True, {"route": "fake"}

    def fake_columns_impl(linear_map, *, input_size, output_size, dtype, chunk_size):
        calls["columns"] = (linear_map(jnp.asarray([4.0])), input_size, output_size, dtype, chunk_size)
        return jnp.ones((int(output_size), int(input_size)), dtype=dtype)

    sentinel_lx = object()
    sentinel_jax = object()
    monkeypatch.setattr(implicit, "_lineax_bicgstab_solve_impl", fake_lineax_impl)
    monkeypatch.setattr(implicit, "_linear_map_jacobian_columns_impl", fake_columns_impl)
    monkeypatch.setattr(implicit, "lx", sentinel_lx)
    monkeypatch.setattr(implicit, "jax", sentinel_jax)

    value, success, stats = implicit._lineax_bicgstab_solve(
        lambda x: x + 1.0,
        jnp.asarray([1.0]),
        x0=jnp.asarray([0.5]),
        tol=1.0e-5,
        max_iter=7,
    )
    jac = implicit._linear_map_jacobian_columns(
        lambda x: x * 2.0,
        input_size=2,
        output_size=3,
        dtype=jnp.float32,
        chunk_size=4,
    )

    np.testing.assert_allclose(np.asarray(value), [3.0])
    assert success is True
    assert stats == {"route": "fake"}
    np.testing.assert_allclose(np.asarray(calls["lineax"][0]), [3.0])
    assert calls["lineax"][3:5] == (1.0e-5, 7)
    assert calls["lineax"][5] is sentinel_lx
    assert calls["lineax"][6] is sentinel_jax
    np.testing.assert_allclose(np.asarray(jac), np.ones((3, 2), dtype=np.float32))
    np.testing.assert_allclose(np.asarray(calls["columns"][0]), [8.0])
    assert calls["columns"][1:4] == (2, 3, jnp.float32)
    assert calls["columns"][4] == 4


@pytest.mark.skipif(not has_jax(), reason="implicit fixed-boundary wrapper requires JAX")
def test_implicit_fixed_boundary_lbfgs_zero_iteration_falls_back_to_gd(monkeypatch) -> None:
    from vmec_jax._compat import jnp

    layout = StateLayout(ns=2, K=1, lasym=False)
    arr = jnp.zeros((2, 1))
    state = VMECState(layout=layout, Rcos=arr, Rsin=arr, Zcos=arr, Zsin=arr, Lcos=arr, Lsin=arr)
    static = SimpleNamespace(
        cfg=SimpleNamespace(nfp=1, ntheta=1, nzeta=1),
        modes=SimpleNamespace(m=np.asarray([0]), n=np.asarray([0])),
        s=jnp.asarray([0.0, 1.0]),
        grid=SimpleNamespace(theta=jnp.asarray([0.0]), zeta=jnp.asarray([0.0])),
    )
    calls = []

    def fake_lbfgs(state0, _static, **kwargs):
        calls.append(("lbfgs", kwargs))
        return SimpleNamespace(state=state0, n_iter=0, grad_rms_history=np.asarray([1.0]), diagnostics={})

    def fake_gd(state0, _static, **kwargs):
        calls.append(("gd", kwargs))
        updated = VMECState(
            layout=state0.layout,
            Rcos=jnp.ones_like(state0.Rcos),
            Rsin=state0.Rsin,
            Zcos=state0.Zcos,
            Zsin=state0.Zsin,
            Lcos=state0.Lcos,
            Lsin=state0.Lsin,
        )
        return SimpleNamespace(
            state=updated, n_iter=3, grad_rms_history=np.asarray([1.0e-9]), diagnostics={"grad_tol": 1.0e-6}
        )

    monkeypatch.setattr(implicit, "solve_fixed_boundary_lbfgs", fake_lbfgs)
    monkeypatch.setattr(implicit, "solve_fixed_boundary_gd", fake_gd)

    out = implicit.solve_fixed_boundary_state_implicit(
        state,
        static,
        phipf=jnp.ones(2),
        chipf=jnp.zeros(2),
        signgs=1,
        lamscale=jnp.asarray(1.0),
        pressure=jnp.zeros(2),
        solver="lbfgs",
        max_iter=5,
        step_size=0.5,
    )

    assert [name for name, _kwargs in calls] == ["lbfgs", "gd"]
    assert calls[1][1]["max_iter"] == 50
    assert calls[1][1]["step_size"] == pytest.approx(0.1)
    np.testing.assert_allclose(np.asarray(out.Rcos), 1.0)

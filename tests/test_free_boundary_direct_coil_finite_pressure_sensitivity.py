from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

from vmec_jax._compat import enable_x64, has_jax
from vmec_jax.external_fields import CoilFieldParams, from_essos_coils
from vmec_jax.free_boundary import nestor_external_only_step
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.profiles import eval_profiles
from vmec_jax.state import pack_state


ROOT = Path(__file__).resolve().parents[1]
LPQA_INPUT = ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
FINITE_PRESSURE_SCALE = 34.46233666638
LPQA_COIL_FILE = "ESSOS_biot_savart_LandremanPaulQA.json"


def _candidate_essos_input_dirs() -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("ESSOS_INPUT_DIR"):
        candidates.append(Path(os.environ["ESSOS_INPUT_DIR"]).expanduser())
    candidates.extend(
        [
            ROOT.parent / "ESSOS_mgrid_pr" / "examples" / "input_files",
            ROOT.parent / "ESSOS" / "examples" / "input_files",
            Path.cwd() / "examples" / "input_files",
        ]
    )
    return candidates


def _find_lpqa_coils() -> Path:
    for directory in _candidate_essos_input_dirs():
        path = directory / LPQA_COIL_FILE
        if path.exists():
            return path
    return _candidate_essos_input_dirs()[0] / LPQA_COIL_FILE


LPQA_COILS = _find_lpqa_coils()


pytestmark = pytest.mark.skipif(not has_jax(), reason="direct-coil finite-pressure sensitivity tests require JAX")


def _circle_coil_params(*, current: float = 3.0e7, radius: float = 1.8, n_segments: int = 96) -> CoilFieldParams:
    from vmec_jax._compat import jnp

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    return CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([current], dtype=float),
        n_segments=int(n_segments),
        nfp=1,
        stellsym=False,
    )


def _write_tiny_direct_freeb_input(
    path: Path,
    *,
    lasym: bool = False,
    niter: int = 4,
    mpol: int = 4,
    ntheta: int = 8,
) -> Path:
    lasym_flag = "T" if bool(lasym) else "F"
    path.write_text(
        f"""
&INDATA
  LFREEB = T
  MGRID_FILE = 'DIRECT_COILS'
  EXTCUR = 1.0
  LASYM = {lasym_flag}
  NFP = 1
  MPOL = {int(mpol)}
  NTOR = 0
  NS = 7
  NZETA = 2
  NTHETA = {int(ntheta)}
  NS_ARRAY = 7
  FTOL_ARRAY = 1.0E-8
  NITER_ARRAY = {int(niter)}
  NITER = {int(niter)}
  FTOL = 1.0E-8
  NSTEP = 20
  NVACSKIP = 1
  GAMMA = 0.0
  PHIEDGE = 1.0
  CURTOR = 0.0
  SPRES_PED = 1.0
  NCURR = 0
  PRES_SCALE = 1.0E4
  AM = 1.0 -1.0
  AI = 0.4 0.0
  AC = 0.0
  RAXIS = 1.0
  ZAXIS = 0.0
  RBC(0,0) = 1.0  ZBS(0,0) = 0.0
  RBC(0,1) = 0.25 ZBS(0,1) = 0.25
  RBC(0,2) = 0.03 ZBS(0,2) = 0.00
/
""".lstrip()
    )
    return path


def _write_lpqa_direct_freeb_input(path: Path, *, niter: int = 3) -> Path:
    indata = deepcopy(read_indata(LPQA_INPUT))
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": "DIRECT_COILS",
            "EXTCUR": [1.0],
            "NS_ARRAY": [12],
            "NITER_ARRAY": [int(niter)],
            "FTOL_ARRAY": [1.0e-8],
            "NITER": int(niter),
            "FTOL": 1.0e-8,
            "MPOL": 4,
            "NTOR": 4,
            "NZETA": 6,
            "NTHETA": 0,
            "NVACSKIP": 1,
            "PRES_SCALE": FINITE_PRESSURE_SCALE,
            "AM": [1.0, -1.0],
        }
    )
    write_indata(path, indata)
    return path


def _run_direct_initial_guess(input_path: Path, params: CoilFieldParams):
    from vmec_jax.driver import run_free_boundary

    return run_free_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )


def _run_direct_solve(input_path: Path, params: CoilFieldParams):
    from vmec_jax.driver import run_free_boundary

    return run_free_boundary(
        input_path,
        solver="vmec2000_iter",
        solver_mode="parity",
        multigrid_use_input_niter=True,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )


def _relative_rms_delta(a, b) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    assert a_arr.shape == b_arr.shape
    delta = float(np.sqrt(np.mean((b_arr - a_arr) ** 2)))
    scale = max(float(np.sqrt(np.mean(a_arr * a_arr))), 1.0e-300)
    return delta / scale


def _pressure_profile(run) -> np.ndarray:
    prof = eval_profiles(run.indata, run.static.s)
    return np.asarray(prof.get("pressure", np.zeros_like(np.asarray(run.static.s))), dtype=float)


def _active_free_boundary(run) -> bool:
    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    freeb = diag.get("free_boundary", {}) if isinstance(diag, dict) else {}
    if not isinstance(freeb, dict):
        return False
    if bool(freeb.get("vacuum_stub", True)):
        return False
    full_updates = np.asarray(diag.get("freeb_full_update_history", []), dtype=int)
    return bool(full_updates.size and np.any(full_updates > 0))


def test_active_direct_coil_provider_is_sensitive_in_finite_pressure_context(tmp_path: Path) -> None:
    """Active NESTOR sampling should change when direct-coil parameters change."""

    enable_x64(True)
    base_params = _circle_coil_params()
    perturbed_params = _circle_coil_params(current=3.3e7)
    input_path = _write_tiny_direct_freeb_input(tmp_path / "input.direct_provider_pressure")
    run = _run_direct_initial_guess(input_path, base_params)

    pressure = _pressure_profile(run)
    assert np.max(pressure) > 0.0

    base, _ = nestor_external_only_step(
        state=run.state,
        static=run.static,
        ivac=1,
        ivacskip=0,
        iter_idx=1,
        runtime=None,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
    )
    perturbed, _ = nestor_external_only_step(
        state=run.state,
        static=run.static,
        ivac=1,
        ivacskip=0,
        iter_idx=1,
        runtime=None,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=perturbed_params,
    )

    assert base.diagnostics is not None
    assert base.diagnostics["provider_kind"] == "direct_coils"
    assert not bool(base.reused)
    assert np.isfinite(np.asarray(base.vac_total.bsqvac)).all()
    assert _relative_rms_delta(base.vac_total.bsqvac, perturbed.vac_total.bsqvac) > 1.0e-3


def test_direct_coil_reuse_refreshes_source_when_provider_changes(tmp_path: Path) -> None:
    """Direct providers must not reuse stale VMEC source vectors across coil changes."""

    enable_x64(True)
    base_params = _circle_coil_params()
    perturbed_params = _circle_coil_params(current=3.3e7)
    input_path = _write_tiny_direct_freeb_input(tmp_path / "input.direct_provider_reuse_source")
    run = _run_direct_initial_guess(input_path, base_params)

    full, runtime = nestor_external_only_step(
        state=run.state,
        static=run.static,
        ivac=1,
        ivacskip=0,
        iter_idx=1,
        runtime=None,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
    )
    reuse, _ = nestor_external_only_step(
        state=run.state,
        static=run.static,
        ivac=2,
        ivacskip=1,
        iter_idx=2,
        runtime=runtime,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=perturbed_params,
    )

    assert full.diagnostics is not None
    assert reuse.diagnostics is not None
    assert reuse.reused
    assert reuse.diagnostics["provider_kind"] == "direct_coils"
    assert reuse.diagnostics["source_reused"] is False
    assert reuse.diagnostics["gsource_rms"] > full.diagnostics["gsource_rms"] * 1.05
    assert _relative_rms_delta(full.vac_total.bsqvac, reuse.vac_total.bsqvac) > 1.0e-3


def test_direct_coil_dense_nestor_output_is_independent_of_nonsingular_ip_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense direct-coil NESTOR output should not depend on the source-assembly chunk size."""

    enable_x64(True)
    params = _circle_coil_params()
    input_path = _write_tiny_direct_freeb_input(tmp_path / "input.direct_provider_chunk_invariance")
    run = _run_direct_initial_guess(input_path, params)

    for key, value in {
        "VMEC_JAX_FREEB_NESTOR_MODE": "dense",
        "VMEC_JAX_FREEB_DENSE_SOLVE_MODE": "mode",
        "VMEC_JAX_FREEB_USE_GREENF_SOURCE": "yes",
        "VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX": "1",
    }.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setenv("VMEC_JAX_FREEB_NONSINGULAR_IP_CHUNK", "1")
    scalar, _ = nestor_external_only_step(
        state=run.state,
        static=run.static,
        ivac=1,
        ivacskip=0,
        iter_idx=1,
        runtime=None,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )
    monkeypatch.setenv("VMEC_JAX_FREEB_NONSINGULAR_IP_CHUNK", "5")
    chunked, _ = nestor_external_only_step(
        state=run.state,
        static=run.static,
        ivac=1,
        ivacskip=0,
        iter_idx=1,
        runtime=None,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )

    for result in (scalar, chunked):
        assert result.model == "vmec2000_like_dense_integral"
        assert result.diagnostics is not None
        assert result.diagnostics["provider_kind"] == "direct_coils"
        assert result.diagnostics["source_reused"] is False
        assert np.isfinite(np.asarray(result.phi)).all()
        assert np.isfinite(np.asarray(result.vac_total.bsqvac)).all()

    np.testing.assert_allclose(chunked.phi, scalar.phi, rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(chunked.vac_total.bsqvac, scalar.vac_total.bsqvac, rtol=1.0e-11, atol=1.0e-12)
    for key in ("gsource_rms", "bvec_mode_nonsing_rms", "bvec_mode_rms"):
        np.testing.assert_allclose(chunked.diagnostics[key], scalar.diagnostics[key], rtol=1.0e-11, atol=1.0e-12)


def test_forced_activation_reports_direct_coil_nestor_diagnostics(tmp_path: Path) -> None:
    """Explicit activation should expose active direct-coil NESTOR diagnostics."""

    enable_x64(True)
    from vmec_jax.driver import run_free_boundary

    params = _circle_coil_params(current=3.0e7)
    input_path = _write_tiny_direct_freeb_input(tmp_path / "input.direct_provider_forced_active")
    run = run_free_boundary(
        input_path,
        max_iter=4,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
        free_boundary_activate_fsq=1.0e99,
    )

    freeb = run.result.diagnostics["free_boundary"]
    assert freeb["vacuum_stub"] is False
    assert freeb["activate_fsq"] == 1.0e99
    assert freeb["nestor_model"].startswith("vmec2000_like_dense_integral")
    assert freeb["final_nestor_recompute_attempted"] is True
    assert freeb["final_nestor_recompute_failed"] is False
    assert freeb["final_nestor_sample_time_s"] > 0.0
    assert freeb["final_nestor_solve_time_s"] > 0.0
    nestor_diag = freeb["last_nestor_diagnostics"]
    assert nestor_diag["provider_kind"] == "direct_coils"
    assert nestor_diag["bnormal_rms"] > 0.0
    assert nestor_diag["bsqvac_rms"] > 0.0
    trial_samples = np.asarray(run.result.diagnostics["freeb_nestor_trial_sample_time_history"], dtype=float)
    trial_failed = np.asarray(run.result.diagnostics["freeb_nestor_trial_failed_history"], dtype=int)
    assert trial_samples.ndim == 1
    assert trial_failed.shape == trial_samples.shape
    assert np.all(trial_samples >= 0.0)
    assert np.count_nonzero(trial_failed) == 0


def test_direct_coil_trial_nestor_timing_records_solver_trial_calls(tmp_path: Path) -> None:
    """Solver-level trial scoring should record rejected NESTOR sample timings."""

    enable_x64(True)
    from vmec_jax.driver import run_free_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    params = _circle_coil_params(current=3.0e7)
    input_path = _write_tiny_direct_freeb_input(tmp_path / "input.direct_trial_timing")
    init = run_free_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )
    result = solve_fixed_boundary_residual_iter(
        init.state,
        init.static,
        indata=init.indata,
        signgs=init.signgs,
        max_iter=4,
        ftol=1.0e-8,
        vmec2000_control=True,
        auto_flip_force=False,
        use_direct_fallback=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces=False,
        use_scan=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
        free_boundary_activate_fsq=1.0e99,
    )

    trial_samples = np.asarray(result.diagnostics["freeb_nestor_trial_sample_time_history"], dtype=float)
    trial_failed = np.asarray(result.diagnostics["freeb_nestor_trial_failed_history"], dtype=int)
    assert trial_samples.size >= 1
    assert trial_failed.shape == trial_samples.shape
    assert np.all(trial_samples > 0.0)
    assert np.count_nonzero(trial_failed) == 0


def test_direct_coil_current_only_objective_fd_slope_is_stable(tmp_path: Path) -> None:
    """Central finite-difference slopes should be stable for a current-only direct-coil objective."""

    enable_x64(True)
    from examples.optimization.free_boundary_QS_coil_optimization import (
        apply_coil_variables,
        run_direct_free_boundary,
        summarize_run,
    )

    input_path = _write_tiny_direct_freeb_input(tmp_path / "input.direct_current_fd_slope")
    base_params = _circle_coil_params(current=3.0e7)
    variables = [("current", (0,))]

    def objective(x: float) -> float:
        params = apply_coil_variables(
            base_params,
            np.asarray([x], dtype=float),
            variables=variables,
            current_step=0.02,
            dof_step=0.0,
        )
        run, wall_s = run_direct_free_boundary(
            input_path,
            params,
            vmec_max_iter=4,
            activate_fsq=1.0e99,
        )
        summary = summarize_run(
            run,
            params,
            objective=np.nan,
            wall_s=wall_s,
            target_aspect=6.0,
            target_iota=0.4,
        )
        assert summary["free_boundary_vacuum_stub"] is False
        assert summary["free_boundary_nestor_model"].startswith("vmec2000_like_dense_integral")
        assert summary["free_boundary_bnormal_rms"] > 0.0
        assert summary["free_boundary_bsqvac_rms"] > 0.0
        return float(summary["free_boundary_bnormal_rms"])

    slopes = []
    for eps in (0.25, 0.125):
        forward = objective(eps)
        backward = objective(-eps)
        slopes.append((forward - backward) / (2.0 * eps))

    slopes = np.asarray(slopes, dtype=float)
    assert np.all(np.isfinite(slopes))
    assert np.min(np.abs(slopes)) > 1.0e-7
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=5.0e-6, atol=1.0e-12)


def test_direct_coil_geometry_dof_accepted_state_fd_slope_is_stable(tmp_path: Path) -> None:
    """Boundary-normal vacuum response should vary smoothly with a coil geometry DOF."""

    enable_x64(True)
    from examples.optimization.free_boundary_QS_coil_optimization import (
        apply_coil_variables,
        run_direct_free_boundary,
        summarize_run,
    )

    input_path = _write_tiny_direct_freeb_input(tmp_path / "input.direct_geometry_fd_slope")
    base_params = _circle_coil_params(current=3.0e7)
    variables = [("fourier_dof", (0, 0, 2))]

    def objective(x: float) -> float:
        params = apply_coil_variables(
            base_params,
            np.asarray([x], dtype=float),
            variables=variables,
            current_step=0.0,
            dof_step=1.0e-2,
        )
        run, wall_s = run_direct_free_boundary(
            input_path,
            params,
            vmec_max_iter=4,
            activate_fsq=1.0e99,
        )
        summary = summarize_run(
            run,
            params,
            objective=np.nan,
            wall_s=wall_s,
            target_aspect=6.0,
            target_iota=0.4,
        )
        assert summary["free_boundary_vacuum_stub"] is False
        assert summary["free_boundary_nestor_model"].startswith("vmec2000_like_dense_integral")
        assert summary["free_boundary_bnormal_rms"] > 0.0
        assert summary["free_boundary_bsqvac_rms"] > 0.0
        return float(summary["free_boundary_bnormal_rms"])

    slopes = []
    for eps in (0.25, 0.125):
        forward = objective(eps)
        backward = objective(-eps)
        slopes.append((forward - backward) / (2.0 * eps))

    slopes = np.asarray(slopes, dtype=float)
    assert np.all(np.isfinite(slopes))
    assert np.min(np.abs(slopes)) > 1.0e-7
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=1.0e-4, atol=1.0e-12)


def test_jax_nestor_operator_complete_solve_fd_slopes_for_current_and_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in JAX NESTOR complete solves should have finite FD response to coil variables."""

    enable_x64(True)
    from vmec_jax._compat import jnp
    from vmec_jax.driver import run_free_boundary

    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_symmetric_jax_nestor_fd",
        lasym=False,
        niter=2,
        mpol=3,
        ntheta=6,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=64)
    monkeypatch.setenv("VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR", "1")

    def metric(params: CoilFieldParams) -> float:
        run = run_free_boundary(
            input_path,
            max_iter=2,
            multigrid=False,
            verbose=False,
            jit_forces=False,
            external_field_provider_kind="direct_coils",
            external_field_provider_params=params,
            free_boundary_activate_fsq=1.0e99,
        )
        freeb = run.result.diagnostics["free_boundary"]
        assert freeb["vacuum_stub"] is False
        assert freeb["final_nestor_recompute_failed"] is False
        nestor = freeb["last_nestor_diagnostics"]
        assert nestor["jax_nestor_operator_applied"] is True
        assert nestor["jax_nestor_operator_reason"] == "applied"
        assert nestor["provider_kind"] == "direct_coils"
        return float(nestor["bnormal_rms"])

    eps = 0.25
    current_forward = base_params.with_arrays(
        base_currents=jnp.asarray(base_params.base_currents) * (1.0 + 0.02 * eps)
    )
    current_backward = base_params.with_arrays(
        base_currents=jnp.asarray(base_params.base_currents) * (1.0 - 0.02 * eps)
    )
    geometry_forward = base_params.with_arrays(
        base_curve_dofs=jnp.asarray(base_params.base_curve_dofs).at[0, 0, 2].add(1.0e-2 * eps)
    )
    geometry_backward = base_params.with_arrays(
        base_curve_dofs=jnp.asarray(base_params.base_curve_dofs).at[0, 0, 2].add(-1.0e-2 * eps)
    )

    # This is still an outer solve finite-response guard.  The driver path
    # materializes host NumPy state/diagnostics between iterations, so it should
    # not be treated as full-loop AD validation.
    current_slope = (metric(current_forward) - metric(current_backward)) / (2.0 * eps)
    geometry_slope = (metric(geometry_forward) - metric(geometry_backward)) / (2.0 * eps)

    slopes = np.asarray([current_slope, geometry_slope], dtype=float)
    assert np.all(np.isfinite(slopes))
    assert np.min(np.abs(slopes)) > 1.0e-16


def test_jax_nestor_operator_accepted_solve_ad_matches_central_fd_for_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected promotion gate for accepted-solve direct-coil AD-vs-FD.

    The fixed-boundary JAX NESTOR operator is differentiable, and the accepted
    solve has finite FD response.  The current blocker is the full
    ``run_free_boundary`` path: under ``jax.grad`` tracing the accepted NESTOR
    diagnostics are dropped before a differentiable accepted-state scalar can
    be compared with finite differences.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp
    from vmec_jax.driver import run_free_boundary

    enable_x64(True)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_accepted_ad_fd",
        lasym=False,
        niter=2,
        mpol=3,
        ntheta=6,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=64)
    monkeypatch.setenv("VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR", "1")
    monkeypatch.setenv("VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR", "0")

    def params_for(scale):
        return base_params.with_arrays(
            base_currents=jnp.asarray(base_params.base_currents) * (1.0 + 0.02 * scale)
        )

    def accepted_bnormal_metric(scale):
        run = run_free_boundary(
            input_path,
            max_iter=2,
            multigrid=False,
            verbose=False,
            jit_forces=False,
            external_field_provider_kind="direct_coils",
            external_field_provider_params=params_for(scale),
            free_boundary_activate_fsq=1.0e99,
        )
        freeb = run.result.diagnostics["free_boundary"]
        assert freeb["vacuum_stub"] is False, (
            "full-loop-ad-missing-accepted-nestor-diagnostics: traced accepted "
            "solve stayed on the vacuum-stub path before exposing a differentiable metric"
        )
        assert freeb["final_nestor_recompute_failed"] is False, (
            "full-loop-ad-missing-accepted-nestor-diagnostics: traced accepted "
            "solve failed final NESTOR recompute before exposing a differentiable metric"
        )
        nestor = freeb["last_nestor_diagnostics"]
        assert "bnormal_rms" in nestor, (
            "full-loop-ad-missing-accepted-nestor-diagnostics: traced accepted "
            "solve dropped last_nestor_diagnostics before exposing a differentiable metric"
        )
        assert nestor["provider_kind"] == "direct_coils"
        assert nestor["jax_nestor_operator_reason"] == "applied"
        return jnp.asarray(nestor["bnormal_rms"])

    eps = 0.25
    fd_current = (accepted_bnormal_metric(eps) - accepted_bnormal_metric(-eps)) / (2.0 * eps)
    assert np.isfinite(np.asarray(fd_current, dtype=float))
    assert abs(float(np.asarray(fd_current))) > 1.0e-16

    try:
        exact_current = jax.grad(accepted_bnormal_metric)(0.0)
    except AssertionError as exc:
        if "full-loop-ad-missing-accepted-nestor-diagnostics" in str(exc):
            pytest.xfail(
                "accepted-solve AD-vs-FD is blocked because jax.grad(run_free_boundary) "
                "does not expose accepted NESTOR diagnostics as differentiable data"
            )
        raise

    assert np.isfinite(np.asarray(exact_current, dtype=float))
    assert abs(float(np.asarray(exact_current))) > 1.0e-16
    np.testing.assert_allclose(exact_current, fd_current, rtol=1.0e-3, atol=1.0e-12)


def test_jax_nestor_operator_fixed_boundary_ad_matches_central_fd_for_coil_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate the JAX operator chain on a fixed boundary from the tiny free-boundary case.

    This promotes the gradient lane one rung beyond finite/nonzero solve
    response without claiming differentiation through the VMEC iteration loop:
    direct coils -> boundary projection -> VMEC/NESTOR source/matrix assembly
    -> dense mode solve is checked against central FD while the plasma boundary
    is held fixed.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp
    from vmec_jax.driver import run_free_boundary
    from vmec_jax.external_fields import sample_coil_field_cylindrical
    from vmec_jax.free_boundary import (
        _build_vmec_mode_basis,
        _ensure_vmec_nonsingular_kernel_tables,
        _sample_external_boundary_arrays,
        _vmec_boundary_wint,
    )
    from vmec_jax.free_boundary_adjoint import (
        dense_vmec_nestor_mode_solve_jax,
        vacuum_boundary_fields_from_cylindrical_jax,
    )

    enable_x64(True)
    for key, value in {
        "VMEC_JAX_FREEB_NESTOR_MODE": "dense",
        "VMEC_JAX_FREEB_DENSE_SOLVE_MODE": "mode",
        "VMEC_JAX_FREEB_USE_GREENF_SOURCE": "1",
        "VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX": "1",
        "VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC": "1",
        "VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR": "1",
    }.items():
        monkeypatch.setenv(key, value)

    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_jax_nestor_ad_fd",
        lasym=False,
        niter=2,
        mpol=3,
        ntheta=6,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=64)
    init = run_free_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
    )
    result, _runtime = nestor_external_only_step(
        state=init.state,
        static=init.static,
        ivac=1,
        ivacskip=0,
        iter_idx=1,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
    )
    assert result.diagnostics is not None
    assert result.diagnostics["jax_nestor_operator_applied"] is True
    assert result.diagnostics["jax_nestor_operator_reason"] == "applied"
    assert result.diagnostics["bnormal_rms"] > 0.0

    sample = _sample_external_boundary_arrays(
        state=init.state,
        static=init.static,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
    )
    wint = _vmec_boundary_wint(static=init.static, ntheta=sample.R.shape[0], nzeta=sample.R.shape[1])
    basis = _build_vmec_mode_basis(
        ntheta=sample.R.shape[0],
        nzeta=sample.R.shape[1],
        nfp=int(init.static.cfg.nfp),
        mf=int(init.static.cfg.mpol) + 1,
        nf=int(init.static.cfg.ntor),
        lasym=bool(init.static.cfg.lasym),
        wint=wint,
    )
    nvper = 64 if int(sample.R.shape[1]) == 1 else max(1, int(init.static.cfg.nfp))
    tables = _ensure_vmec_nonsingular_kernel_tables(basis=basis, nv=sample.R.shape[1], nvper=nvper)

    R = jnp.asarray(sample.R)
    Z = jnp.asarray(sample.Z)
    phi = jnp.asarray(sample.phi)
    Ru = jnp.asarray(sample.Ru)
    Zu = jnp.asarray(sample.Zu)
    Rv = jnp.asarray(sample.Rv)
    Zv = jnp.asarray(sample.Zv)
    ruu = jnp.asarray(sample.ruu)
    ruv = jnp.asarray(sample.ruv)
    rvv = jnp.asarray(sample.rvv)
    zuu = jnp.asarray(sample.zuu)
    zuv = jnp.asarray(sample.zuv)
    zvv = jnp.asarray(sample.zvv)
    wint_jax = jnp.asarray(wint)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)

    def params_for(current_scale, geometry_scale):
        return base_params.with_arrays(
            base_curve_dofs=base_dofs.at[0, 0, 2].add(1.0e-2 * geometry_scale),
            base_currents=base_currents * (1.0 + 0.02 * current_scale),
        )

    def response(current_scale, geometry_scale):
        params = params_for(current_scale, geometry_scale)
        br, bp, bz = sample_coil_field_cylindrical(params, R, Z, phi)
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bp,
            bz=bz,
            R=R,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
        )
        bexni = -vac["bnormal"] * wint_jax * ((2.0 * jnp.pi) ** 2)
        out = dense_vmec_nestor_mode_solve_jax(
            R=R,
            Z=Z,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            ruu=ruu,
            ruv=ruv,
            rvv=rvv,
            zuu=zuu,
            zuv=zuv,
            zvv=zvv,
            bexni=jnp.ravel(bexni),
            basis=basis,
            tables=tables,
            signgs=int(init.signgs),
            nvper=nvper,
            include_analytic=True,
        )
        return 0.5 * jnp.vdot(out["mode_coeffs"], out["mode_coeffs"]) + 0.05 * jnp.vdot(
            out["phi_flat"],
            out["phi_flat"],
        )

    eps = 1.0e-4
    exact_current = jax.grad(lambda scale: response(scale, 0.0))(0.0)
    fd_current = (response(eps, 0.0) - response(-eps, 0.0)) / (2.0 * eps)
    exact_geometry = jax.grad(lambda scale: response(0.0, scale))(0.0)
    fd_geometry = (response(0.0, eps) - response(0.0, -eps)) / (2.0 * eps)

    derivs = np.asarray([exact_current, fd_current, exact_geometry, fd_geometry], dtype=float)
    assert np.all(np.isfinite(derivs))
    assert np.min(np.abs(derivs)) > 1.0e-8
    np.testing.assert_allclose(exact_current, fd_current, rtol=2.0e-6, atol=1.0e-8)
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=2.0e-6, atol=1.0e-8)


@pytest.mark.full
def test_essos_full_solve_state_is_sensitive_to_direct_coil_current_at_finite_pressure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional full-solve guard for accepted-state sensitivity to ESSOS coil parameters."""

    if os.environ.get("RUN_FULL", "") != "1":
        pytest.skip("Set RUN_FULL=1 to run the optional ESSOS full-solve sensitivity test")
    essos_coils = pytest.importorskip("essos.coils")
    if not LPQA_COILS.exists():
        pytest.skip(f"missing local ESSOS Landreman-Paul QA coils: {LPQA_COILS}")

    from vmec_jax._compat import jnp

    enable_x64(True)
    coils = essos_coils.Coils_from_json(str(LPQA_COILS))
    base_params = from_essos_coils(coils, chunk_size=256)
    scaled_params = base_params.with_arrays(base_currents=jnp.asarray(base_params.base_currents) * 100.0)
    input_path = _write_lpqa_direct_freeb_input(tmp_path / "input.lpqa_direct_finite_pressure")

    monkeypatch.setenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "1.0e99")
    base_run = _run_direct_solve(input_path, base_params)
    scaled_run = _run_direct_solve(input_path, scaled_params)

    assert np.max(_pressure_profile(base_run)) > 0.0
    if not (_active_free_boundary(base_run) and _active_free_boundary(scaled_run)):
        pytest.xfail(
            "Optional direct-coil finite-pressure full solve did not enter active "
            "free-boundary vacuum coupling within the gated short budget."
        )

    state_delta = _relative_rms_delta(pack_state(base_run.state), pack_state(scaled_run.state))
    assert state_delta > 1.0e-9

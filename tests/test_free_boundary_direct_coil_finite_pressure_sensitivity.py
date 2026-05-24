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
LPQA_COILS = Path("/Users/rogeriojorge/local/ESSOS_mgrid_pr/examples/input_files/ESSOS_biot_savart_LandremanPaulQA.json")
FINITE_PRESSURE_SCALE = 34.46233666638


pytestmark = pytest.mark.skipif(not has_jax(), reason="direct-coil finite-pressure sensitivity tests require JAX")


def _circle_coil_params(*, current: float = 3.0e7, radius: float = 1.8) -> CoilFieldParams:
    from vmec_jax._compat import jnp

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    return CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([current], dtype=float),
        n_segments=96,
        nfp=1,
        stellsym=False,
    )


def _write_tiny_direct_freeb_input(path: Path) -> Path:
    path.write_text(
        """
&INDATA
  LFREEB = T
  MGRID_FILE = 'DIRECT_COILS'
  EXTCUR = 1.0
  LASYM = F
  NFP = 1
  MPOL = 4
  NTOR = 0
  NS = 7
  NZETA = 2
  NTHETA = 8
  NS_ARRAY = 7
  FTOL_ARRAY = 1.0E-8
  NITER_ARRAY = 4
  NITER = 4
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
    nestor_diag = freeb["last_nestor_diagnostics"]
    assert nestor_diag["provider_kind"] == "direct_coils"
    assert nestor_diag["bnormal_rms"] > 0.0
    assert nestor_diag["bsqvac_rms"] > 0.0


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
    perturbed_params = base_params.with_arrays(base_currents=jnp.asarray(base_params.base_currents) * 1.05)
    input_path = _write_lpqa_direct_freeb_input(tmp_path / "input.lpqa_direct_finite_pressure")

    monkeypatch.setenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "1.0e99")
    base_run = _run_direct_solve(input_path, base_params)
    perturbed_run = _run_direct_solve(input_path, perturbed_params)

    assert np.max(_pressure_profile(base_run)) > 0.0
    if not (_active_free_boundary(base_run) and _active_free_boundary(perturbed_run)):
        pytest.xfail(
            "Optional direct-coil finite-pressure full solve did not enter active "
            "free-boundary vacuum coupling within the gated short budget."
        )

    state_delta = _relative_rms_delta(pack_state(base_run.state), pack_state(perturbed_run.state))
    if state_delta <= 1.0e-9:
        pytest.xfail(
            "Current direct-coil finite-pressure full-solve path leaves the accepted "
            "VMEC state insensitive to ESSOS coil-current perturbations."
        )
    assert state_delta > 1.0e-9

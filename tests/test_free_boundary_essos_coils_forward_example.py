from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.bootstrap_current import BootstrapCurrentIteration, BootstrapCurrentResult

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "free_boundary_essos_coils_forward.py"
BETA_SCAN_SCRIPT_PATH = ROOT / "examples" / "free_boundary_essos_coils_beta_scan.py"


def _load_forward_module():
    spec = importlib.util.spec_from_file_location("free_boundary_essos_coils_forward_example", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_beta_scan_module():
    spec = importlib.util.spec_from_file_location("free_boundary_essos_coils_beta_scan", BETA_SCAN_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_forward_direct_coil_example_imports_without_essos_import_side_effects():
    module = _load_forward_module()

    assert module.DEFAULT_OUTDIR.name == "free_boundary_essos_coils_forward"
    assert callable(module.main)
    assert callable(module._summarize_run)


def test_beta_scan_pressure_continuation_helpers_convert_wout_boundary(tmp_path):
    module = _load_beta_scan_module()
    base_indata = module.read_indata(ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres")
    base_indata.scalars["MPOL"] = 5
    base_indata.scalars["NTOR"] = 5
    wout = SimpleNamespace(
        nfp=2,
        lasym=False,
        xm=np.array([0, 1, 1, 6]),
        xn=np.array([0.0, -2.0, 2.0, 0.0]),
        rmnc=np.array([[1.2, 0.1, 0.2, 9.0]]),
        zmns=np.array([[0.0, -0.01, 0.02, 9.0]]),
        raxis_cc=np.array([1.1, 0.01]),
        zaxis_cs=np.array([0.0, 0.02]),
    )

    warmed = module.continue_indata_from_wout_boundary(base_indata, wout)

    assert module._vmec_input_n_from_wout_xn(4.0, nfp=2) == 2
    assert warmed.indexed["RBC"][(0, 0)] == 1.2
    assert warmed.indexed["RBC"][(-1, 1)] == 0.1
    assert warmed.indexed["RBC"][(1, 1)] == 0.2
    assert (0, 6) not in warmed.indexed["RBC"]
    assert warmed.indexed["ZBS"][(-1, 1)] == -0.01
    assert warmed.scalars["RAXIS_CC"] == [1.1, 0.01]
    assert warmed.scalars["ZAXIS_CS"] == [0.0, 0.02]

    wout_path = tmp_path / "wout_ok.nc"
    wout_path.write_text("placeholder")
    ok_summary = {"fsqr": 1.0e-8, "fsqz": 2.0e-8, "fsql": 3.0e-8, "wout": str(wout_path)}
    bad_summary = {"fsqr": 1.0e-3, "fsqz": 0.0, "fsql": 0.0, "wout": str(wout_path)}
    assert module._summary_is_promotable_for_pressure_continuation(ok_summary, max_fsq=1.0e-6)
    assert not module._summary_is_promotable_for_pressure_continuation(bad_summary, max_fsq=1.0e-6)


def test_beta_scan_summary_checkpoint_preserves_partial_runs(tmp_path):
    module = _load_beta_scan_module()
    args = SimpleNamespace(
        coil_current_scale=1.0,
        phiedge=-0.025,
        pressure_scale_for_one_percent_beta=1000.0,
        pressure_continuation=True,
        pressure_continuation_max_fsq=1.0e-6,
        disable_direct_coil_source_reuse=False,
        direct_coil_trial_resample=False,
        direct_coil_limit_update_rms=False,
        ns=12,
        max_iter=20,
        ftol=1.0e-8,
    )
    summary_path = tmp_path / "summary.json"
    runs = [
        {
            "backend": "direct",
            "nominal_beta_percent": 0.0,
            "fsqr": 1.0e-12,
            "fsqz": 2.0e-12,
            "fsql": 3.0e-12,
        }
    ]

    module._write_summary_checkpoint(
        summary_path,
        coils_json=tmp_path / "coils.json",
        mgrid_file=tmp_path / "mgrid.nc",
        args=args,
        scale_summary={"coil_r_mean": 1.0},
        ns_array=[16, 101],
        niter_array=[600, 3000],
        ftol_array=[1.0e-8, 1.0e-12],
        summaries=runs,
        complete=False,
    )
    partial = json.loads(summary_path.read_text())
    assert partial["complete"] is False
    assert partial["runs"] == runs
    assert partial["ns_array"] == [16, 101]

    module._write_summary_checkpoint(
        summary_path,
        coils_json=tmp_path / "coils.json",
        mgrid_file=tmp_path / "mgrid.nc",
        args=args,
        scale_summary={"coil_r_mean": 1.0},
        ns_array=[16, 101],
        niter_array=[600, 3000],
        ftol_array=[1.0e-8, 1.0e-12],
        summaries=runs,
        complete=True,
    )
    complete = json.loads(summary_path.read_text())
    assert complete["complete"] is True


def test_beta_scan_resume_existing_case_uses_wout_path(monkeypatch, tmp_path):
    module = _load_beta_scan_module()
    assert (
        module._resume_existing_case(
            output_dir=tmp_path,
            backend="direct",
            beta_percent=0.5,
            pressure_scale_for_one_percent_beta=1000.0,
        )
        is None
    )

    wout_path = tmp_path / "wout_direct_beta_0.500.nc"
    wout_path.write_text("placeholder")

    def fake_summarize_existing_wout(path, *, backend, beta_percent):
        return {
            "wout": str(path),
            "backend": backend,
            "nominal_beta_percent": float(beta_percent),
            "fsqr": 1.0e-12,
            "fsqz": 2.0e-12,
            "fsql": 3.0e-12,
        }

    monkeypatch.setattr(module, "summarize_existing_wout", fake_summarize_existing_wout)
    summary = module._resume_existing_case(
        output_dir=tmp_path,
        backend="direct",
        beta_percent=0.5,
        pressure_scale_for_one_percent_beta=1200.0,
    )
    assert summary["wout"] == str(wout_path)
    assert summary["pressure_scale"] == 600.0


def test_beta_scan_bootstrap_current_preconditioner_updates_indata(monkeypatch, tmp_path):
    module = _load_beta_scan_module()
    base = module.read_indata(ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres")
    base.scalars["PHIEDGE"] = -0.025
    base.scalars["MGRID_FILE"] = "mgrid_fixture.nc"

    def fake_fixed_point(indata, *, options, solve_fn, ne_coeffs, Te_coeffs, Ti_coeffs, Zeff_coeffs):
        assert options.helicity_n == 0
        assert options.n_current == 5
        assert options.max_current_update_norm == pytest.approx(0.3)
        assert np.asarray(ne_coeffs).size > 0
        assert np.asarray(Te_coeffs).size > 0
        # Exercise the solve callback enough to verify mgrid path rewriting.
        run = solve_fn(indata)
        assert Path(run.indata.scalars["MGRID_FILE"]).is_absolute()
        updated = module.deepcopy(indata)
        updated.scalars["CURTOR"] = 123.0
        updated.scalars["PCURR_TYPE"] = "cubic_spline_ip"
        updated.scalars["AC"] = [1.0]
        updated.scalars["AC_AUX_S"] = [0.0, 1.0]
        updated.scalars["AC_AUX_F"] = [2.0, 2.0]
        return BootstrapCurrentResult(
            indata=updated,
            history=(
                BootstrapCurrentIteration(
                    iteration=1,
                    mismatch_norm=0.5,
                    current_update_norm=0.25,
                    curtor=123.0,
                    ac_aux_s=(0.0, 1.0),
                    ac_aux_f=(2.0, 2.0),
                ),
            ),
            converged=True,
            reason="test",
            last_run=run,
            last_diagnostics={},
        )

    def fake_run_free_boundary(input_path, **kwargs):
        assert kwargs["max_iter"] == 3
        staged = module.read_indata(input_path)
        assert staged.scalars["NS_ARRAY"] == [8, 16]
        assert staged.scalars["NITER_ARRAY"] == [2, 3]
        assert staged.scalars["FTOL_ARRAY"] == [1.0e-6, 1.0e-8]
        assert staged.scalars["NITER"] == 3
        assert staged.scalars["FTOL"] == pytest.approx(1.0e-8)
        return SimpleNamespace(indata=staged, signgs=1)

    monkeypatch.setattr(module, "bootstrap_current_fixed_point", fake_fixed_point)
    monkeypatch.setattr(module, "run_free_boundary", fake_run_free_boundary)

    updated, summary = module.apply_bootstrap_current_fixed_point_preconditioner(
        base,
        backend="mgrid",
        beta_percent=1.0,
        output_dir=tmp_path,
        label="case",
        mgrid_file=tmp_path / "mgrid_fixture.nc",
        pressure_profile="standard",
        helicity_n=0,
        surfaces=(0.2, 0.8),
        n_current=5,
        max_fixed_point_iter=1,
        damping=0.5,
        max_current_update_norm=0.3,
        current_tol=1.0e-2,
        mismatch_tol=1.0e-2,
        vmec_max_iter=3,
        activate_fsq=1.0e99,
        bootstrap_ns_array=(8, 16),
        bootstrap_niter_array=(2, 3),
        bootstrap_ftol_array=(1.0e-6, 1.0e-8),
    )

    assert updated.scalars["CURTOR"] == 123.0
    assert summary["enabled"] is True
    assert summary["converged"] is True
    assert summary["iterations"] == 1
    assert summary["max_current_update_norm"] == pytest.approx(0.3)
    assert summary["final_mismatch_norm"] == 0.5
    assert summary["last_evaluated_mismatch_norm"] == 0.5
    assert summary["returned_mismatch_norm"] == 0.5
    assert summary["returned_current"]["curtor"] == 123.0
    assert summary["returned_current"]["ac_aux_f"] == [2.0, 2.0]
    assert summary["bootstrap_ns_array"] == [8, 16]
    assert summary["bootstrap_niter_array"] == [2, 3]
    assert summary["bootstrap_ftol_array"] == [1.0e-6, 1.0e-8]
    assert Path(summary["history_json"]).exists()
    assert Path(summary["final_input"]).exists()


def test_beta_scan_bootstrap_current_summary_distinguishes_returned_best(monkeypatch, tmp_path):
    module = _load_beta_scan_module()
    base = module.read_indata(ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres")
    base.scalars["PHIEDGE"] = -0.025
    best = module.deepcopy(base)
    best.scalars["CURTOR"] = 111.0
    best.scalars["PCURR_TYPE"] = "cubic_spline_ip"
    best.scalars["AC"] = [1.0]
    best.scalars["AC_AUX_S"] = [0.0, 1.0]
    best.scalars["AC_AUX_F"] = [1.0, 1.0]

    def fake_fixed_point(*_args, **_kwargs):
        return BootstrapCurrentResult(
            indata=best,
            history=(
                BootstrapCurrentIteration(
                    iteration=1,
                    mismatch_norm=0.4,
                    current_update_norm=0.2,
                    curtor=111.0,
                    ac_aux_s=(0.0, 1.0),
                    ac_aux_f=(1.0, 1.0),
                ),
                BootstrapCurrentIteration(
                    iteration=2,
                    mismatch_norm=0.9,
                    current_update_norm=0.7,
                    curtor=222.0,
                    ac_aux_s=(0.0, 1.0),
                    ac_aux_f=(2.0, 2.0),
                ),
            ),
            converged=False,
            reason="max_fixed_point_iter",
            returned_best_evaluated=True,
            best_evaluated_iteration=1,
            best_evaluated_mismatch_norm=0.4,
        )

    monkeypatch.setattr(module, "bootstrap_current_fixed_point", fake_fixed_point)
    updated, summary = module.apply_bootstrap_current_fixed_point_preconditioner(
        base,
        backend="direct",
        beta_percent=1.0,
        output_dir=tmp_path,
        label="case",
        mgrid_file=tmp_path / "mgrid.nc",
        pressure_profile="standard",
        helicity_n=0,
        surfaces=(0.2, 0.8),
        n_current=5,
        max_fixed_point_iter=2,
        damping=0.5,
        current_tol=1.0e-2,
        mismatch_tol=1.0e-2,
        vmec_max_iter=3,
        activate_fsq=1.0e99,
        return_best_evaluated_on_max_iter=True,
    )

    assert updated is best
    assert summary["returned_best_evaluated"] is True
    assert summary["best_evaluated_iteration"] == 1
    assert summary["last_evaluated_mismatch_norm"] == 0.9
    assert summary["last_proposed_current_update_norm"] == 0.7
    assert summary["last_proposed_curtor"] == 222.0
    assert summary["returned_mismatch_norm"] == 0.4
    assert summary["returned_current"]["curtor"] == 111.0
    assert summary["returned_current"]["ac_aux_f"] == [1.0, 1.0]
    assert summary["final_mismatch_norm"] == 0.9


def test_beta_scan_bootstrap_current_preconditioner_skip_and_validation(tmp_path):
    module = _load_beta_scan_module()
    base = module.read_indata(ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres")
    skipped, summary = module.apply_bootstrap_current_fixed_point_preconditioner(
        base,
        backend="direct",
        beta_percent=0.0,
        output_dir=tmp_path,
        label="zero",
        mgrid_file=tmp_path / "mgrid.nc",
        pressure_profile="standard",
        helicity_n=0,
        surfaces=(0.2, 0.8),
        n_current=4,
        max_fixed_point_iter=1,
        damping=0.5,
        current_tol=1.0e-2,
        mismatch_tol=1.0e-2,
        vmec_max_iter=2,
        activate_fsq=None,
    )
    assert skipped is base
    assert summary["skipped"] == "zero_beta"

    with pytest.raises(ValueError, match="pressure-profile standard"):
        module.apply_bootstrap_current_fixed_point_preconditioner(
            base,
            backend="direct",
            beta_percent=1.0,
            output_dir=tmp_path,
            label="bad",
            mgrid_file=tmp_path / "mgrid.nc",
            pressure_profile="linear-scale",
            helicity_n=0,
            surfaces=(0.2, 0.8),
            n_current=4,
            max_fixed_point_iter=1,
            damping=0.5,
            current_tol=1.0e-2,
            mismatch_tol=1.0e-2,
            vmec_max_iter=2,
            activate_fsq=None,
        )


@pytest.mark.skipif(
    os.environ.get("RUN_FREEB_BOOTSTRAP_BETA_SCAN", "").strip().lower() not in {"1", "true", "yes"},
    reason="set RUN_FREEB_BOOTSTRAP_BETA_SCAN=1 to run the optional ESSOS/direct-coil bootstrap beta scan",
)
def test_beta_scan_bootstrap_current_direct_coil_active_smoke(tmp_path):
    """Run a tiny real ESSOS/direct-coil finite-beta scan with Redl feedback.

    This is an optional integration gate because it imports ESSOS assets and
    launches real free-boundary solves.  It verifies the code path that the
    mocked tests cannot: finite-pressure direct coils, active NESTOR coupling,
    and persisted bootstrap-current stage diagnostics.
    """

    pytest.importorskip("essos.coils")
    module = _load_beta_scan_module()
    try:
        coils_json = module.find_essos_landreman_paul_qa_coils()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    rc = module.main(
        [
            "--outdir",
            str(tmp_path),
            "--input",
            str(ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"),
            "--coils-json",
            str(coils_json),
            "--skip-mgrid-runs",
            "--betas",
            "0",
            "1",
            "--pressure-profile",
            "standard",
            "--bootstrap-current-fixed-point",
            "--bootstrap-helicity-n",
            "0",
            "--bootstrap-max-fixed-point-iter",
            "1",
            "--bootstrap-n-current",
            "8",
            "--bootstrap-surfaces",
            "0.25 0.50 0.75",
            "--bootstrap-vmec-max-iter",
            "2",
            "--bootstrap-damping",
            "0.5",
            "--bootstrap-max-current-update-norm",
            "0.1",
            "--max-iter",
            "2",
            "--ns",
            "8",
            "--mpol",
            "3",
            "--ntor",
            "3",
            "--mgrid-nr",
            "8",
            "--mgrid-nz",
            "8",
            "--mgrid-nphi",
            "4",
            "--activate-fsq",
            "1e99",
            "--allow-scale-mismatch",
        ]
    )

    assert rc == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete"] is True
    assert len(summary["runs"]) == 2
    finite_beta = next(run for run in summary["runs"] if run["nominal_beta_percent"] == 1.0)
    assert finite_beta["backend"] == "direct"
    assert finite_beta["free_boundary_vacuum_stub"] is False
    assert finite_beta["free_boundary_nestor_model"] == "vmec2000_like_dense_integral"
    assert finite_beta["free_boundary_bnormal_rms_history"]["count"] >= 1
    assert finite_beta["bootstrap_current"]["iterations"] == 1
    assert finite_beta["bootstrap_current"]["reason"] == "max_fixed_point_iter"
    assert finite_beta["bootstrap_current"]["max_current_update_norm"] == pytest.approx(0.1)
    assert finite_beta["bootstrap_current"]["any_current_update_limited"] is True
    assert finite_beta["bootstrap_current"]["final_effective_damping"] < 0.5
    assert Path(finite_beta["bootstrap_current"]["history_json"]).exists()
    assert Path(finite_beta["bootstrap_current"]["final_input"]).exists()

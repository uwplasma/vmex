from __future__ import annotations

from copy import deepcopy
import json
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
LPQA_INPUT = ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"
FINITE_PRESSURE_SCALE = 1000.0
FREE_BOUNDARY_PHIEDGE = -0.025
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
            "PHIEDGE": FREE_BOUNDARY_PHIEDGE,
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


@pytest.mark.py311_coverage_only
def test_direct_coil_trace_fingerprint_detects_control_branch_changes() -> None:
    from vmec_jax._compat import jnp
    from vmec_jax.free_boundary_adjoint import (
        direct_coil_accepted_trace_array_controls_jax,
        direct_coil_accepted_trace_branch_metadata,
        direct_coil_accepted_trace_fingerprint_delta_summary,
        direct_coil_accepted_trace_preconditioner_controls_jax,
        direct_coil_accepted_trace_preconditioner_policy_segments,
        direct_coil_accepted_trace_scalar_controls_jax,
        direct_coil_accepted_trace_step_controls_jax,
        direct_coil_accepted_trace_step_policy_segment_summary,
        direct_coil_accepted_trace_step_policy_segments,
        direct_coil_accepted_trace_fingerprint,
        direct_coil_accepted_trace_fingerprint_delta,
        direct_coil_same_branch_replay_gate_report,
        free_boundary_adjoint_trace_replay_diagnostics,
    )

    z = np.arange(6.0).reshape(2, 3)
    trace0 = {
        "dt_eff": np.asarray(0.5),
        "b1": np.asarray(0.125),
        "fac": np.asarray(0.9),
        "force_scale": np.asarray(1.0),
        "max_update_rms_pre": np.asarray(0.25),
        "lambda_update_scale": np.asarray([1.0, 0.5]),
        "limit_update_rms": np.asarray(1.0),
        "flip_sign": False,
        "divide_by_scalxc_for_update": True,
        "preconditioner_use_precomputed_tridi": False,
        "preconditioner_use_lax_tridi": True,
        "precond_jmax": 2,
        "precond_mats": {"ar": z + 6.0, "br": z + 7.0},
        "lam_prec": np.asarray([1.0, 2.0, 3.0]),
        "w_mode_mn": np.ones((2, 3)),
        "vRcc_before": z,
        "vRss_before": z + 1.0,
        "vZsc_before": z + 2.0,
        "vZcs_before": z + 3.0,
        "vLsc_before": z + 4.0,
        "vLcs_before": z + 5.0,
        "freeb_bsqvac_half": np.ones((2, 3)),
        "freeb_nestor_trace": {"gsource": np.ones(2), "bsqvac": np.ones(2)},
        "state_pre": np.ones(4),
        "state_post": np.ones(4) * 2.0,
    }
    trace1 = {
        **trace0,
        "dt_eff": np.asarray(0.25),
        "freeb_bsqvac_half": np.ones((2, 3)) * 3.0,
    }
    trace2 = {
        **trace0,
        "dt_eff": np.asarray(0.125),
        "freeb_bsqvac_half": np.ones((2, 3)) * 4.0,
    }
    scalar_controls = direct_coil_accepted_trace_scalar_controls_jax([trace0, trace1])
    assert np.allclose(np.asarray(scalar_controls["dt_eff"]), np.asarray([0.5, 0.25]))
    assert np.allclose(np.asarray(scalar_controls["lambda_update_scale"]), np.asarray([[1.0, 0.5], [1.0, 0.5]]))
    assert np.array_equal(np.asarray(scalar_controls["flip_sign"]), np.asarray([False, False]))
    assert np.array_equal(np.asarray(scalar_controls["preconditioner_use_lax_tridi"]), np.asarray([True, True]))

    bad_scalar_shape = dict(trace1)
    bad_scalar_shape["lambda_update_scale"] = np.asarray([1.0, 0.5, 0.25])
    with pytest.raises(ValueError, match="lambda_update_scale"):
        direct_coil_accepted_trace_scalar_controls_jax([trace0, bad_scalar_shape])
    with pytest.raises(ValueError, match="accepted trace"):
        direct_coil_accepted_trace_scalar_controls_jax([])

    array_controls = direct_coil_accepted_trace_array_controls_jax([trace0, trace1])
    assert np.asarray(array_controls["vRcc_before"]).shape == (2, 2, 3)
    np.testing.assert_allclose(np.asarray(array_controls["vZsc_before"][0]), z + 2.0)

    bad_array_shape = dict(trace1)
    bad_array_shape["vRcc_before"] = np.ones((3, 3))
    with pytest.raises(ValueError, match="vRcc_before"):
        direct_coil_accepted_trace_array_controls_jax([trace0, bad_array_shape])

    optional_lasym = {**trace0, "vRsc_before": z}
    with pytest.raises(ValueError, match="vRsc_before"):
        direct_coil_accepted_trace_array_controls_jax([optional_lasym, trace1])

    preconditioner_controls = direct_coil_accepted_trace_preconditioner_controls_jax([trace0, trace1])
    assert np.asarray(preconditioner_controls["precond_mats"]["ar"]).shape == (2, 2, 3)
    np.testing.assert_allclose(np.asarray(preconditioner_controls["lam_prec"][0]), np.asarray([1.0, 2.0, 3.0]))
    same_policy_segments = direct_coil_accepted_trace_preconditioner_policy_segments([trace0, trace1])
    assert [(segment["start"], segment["stop"], segment["n_steps"]) for segment in same_policy_segments] == [(0, 2, 2)]
    assert same_policy_segments[0]["signature"][1] == 1
    assert same_policy_segments[0]["signature"][2] == 2
    with pytest.raises(ValueError, match="accepted trace"):
        direct_coil_accepted_trace_preconditioner_controls_jax([])

    with pytest.raises(KeyError, match="br_axis"):
        direct_coil_accepted_trace_step_controls_jax([trace0, trace1])
    with pytest.raises(ValueError, match="accepted trace"):
        direct_coil_accepted_trace_step_controls_jax([])
    mixed_force_trace = {**trace0, "force_state_pre": {"r": np.ones(2)}}
    mixed_step_controls = direct_coil_accepted_trace_step_controls_jax(
        [mixed_force_trace, trace1],
        include_nestor_axes=False,
    )
    assert "state_pre" in mixed_step_controls
    assert "force_state_pre" not in mixed_step_controls
    bad_optional_shape = {**trace1, "force_state_pre": {"r": np.ones(3)}}
    with pytest.raises(ValueError, match="force_state_pre"):
        direct_coil_accepted_trace_step_controls_jax(
            [mixed_force_trace, bad_optional_shape],
            include_nestor_axes=False,
        )
    both_optional_trace0 = {
        **trace0,
        "force_state_pre": {"r": np.ones(2)},
        "freeb_pres_scale": np.asarray([1.0, 2.0]),
        "constraint_rcon0": np.asarray([0.0, 1.0]),
    }
    both_optional_trace1 = {
        **trace1,
        "force_state_pre": {"r": np.ones(2) * 3.0},
        "freeb_pres_scale": np.asarray([3.0, 4.0]),
        "constraint_rcon0": np.asarray([2.0, 3.0]),
    }
    optional_step_controls = direct_coil_accepted_trace_step_controls_jax(
        [both_optional_trace0, both_optional_trace1],
        include_nestor_axes=False,
    )
    assert np.asarray(optional_step_controls["force_state_pre"]["r"]).shape == (2, 2)
    np.testing.assert_allclose(np.asarray(optional_step_controls["freeb_pres_scale"][1]), np.asarray([3.0, 4.0]))
    np.testing.assert_allclose(np.asarray(optional_step_controls["constraint_rcon0"][0]), np.asarray([0.0, 1.0]))
    axis_trace0 = deepcopy(trace0)
    axis_trace1 = deepcopy(trace1)
    for trace, offset in ((axis_trace0, 0.0), (axis_trace1, 10.0)):
        trace["freeb_nestor_trace"] = {
            **trace["freeb_nestor_trace"],
            "br_axis": np.ones((2, 3)) + offset,
            "bp_axis": np.ones((2, 3)) * 2.0 + offset,
            "bz_axis": np.ones((2, 3)) * 3.0 + offset,
        }
    axis_controls = direct_coil_accepted_trace_step_controls_jax([axis_trace0, axis_trace1])
    assert np.asarray(axis_controls["freeb_nestor_axes"]["br_axis"]).shape == (2, 2, 3)
    np.testing.assert_allclose(np.asarray(axis_controls["freeb_nestor_axes"]["bz_axis"][1]), np.ones((2, 3)) * 13.0)
    changed_static_trace = {**axis_trace1, "include_edge_residual": True}
    step_policy_segments = direct_coil_accepted_trace_step_policy_segments([axis_trace0, changed_static_trace])
    assert [(segment["start"], segment["stop"], segment["n_steps"]) for segment in step_policy_segments] == [
        (0, 1, 1),
        (1, 2, 1),
    ]
    step_policy_summary = direct_coil_accepted_trace_step_policy_segment_summary(
        [axis_trace0, changed_static_trace],
        accept_mask=np.asarray([True, False]),
        done_mask=np.asarray([False, False]),
    )
    assert step_policy_summary[0]["accepted_steps"] == 1
    assert step_policy_summary[1]["rejected_steps"] == 1

    branch_metadata = direct_coil_accepted_trace_branch_metadata([trace0, trace1])
    assert branch_metadata["n_steps"] == 2
    assert branch_metadata["n_free_boundary_replay_steps"] == 2
    assert branch_metadata["fingerprint"]["n_freeb_steps"] == 2
    assert np.array_equal(np.asarray(branch_metadata["accepted_mask"]), np.asarray([True, True]))
    assert np.array_equal(np.asarray(branch_metadata["done_mask"]), np.asarray([False, True]))
    assert np.array_equal(np.asarray(branch_metadata["reset_to_trace_pre"]), np.asarray([False, False]))
    assert np.array_equal(np.asarray(branch_metadata["active_free_boundary_mask"]), np.asarray([True, True]))
    assert branch_metadata["preconditioner_policy_segment_summary"][0]["free_boundary_replay_steps"] == 2
    branch_metadata_json = direct_coil_accepted_trace_branch_metadata(
        [trace0, trace1],
        accept_mask=np.asarray([True, False]),
        done_mask=np.asarray([False, False]),
        json_safe=True,
    )
    json.dumps(branch_metadata_json, allow_nan=False)
    assert branch_metadata_json["accepted_mask"] == [True, False]
    assert branch_metadata_json["active_free_boundary_mask"] == [True, False]
    assert branch_metadata_json["preconditioner_policy_segment_summary"][0]["rejected_steps"] == 1
    padded_diagnostics = free_boundary_adjoint_trace_replay_diagnostics(
        {"adjoint_step_trace": [trace0, trace1, trace2]},
        accept_mask=np.asarray([True, True, False]),
        done_mask=np.asarray([False, True, False]),
    )
    assert padded_diagnostics["differentiates_adaptive_controller"] is False
    assert padded_diagnostics["n_steps"] == 3
    assert padded_diagnostics["branch_fingerprint"]["n_steps"] == 3
    assert np.array_equal(np.asarray(padded_diagnostics["masks"]["active"]), np.asarray([True, True, False]))
    assert np.array_equal(np.asarray(padded_diagnostics["masks"]["accepted"]), np.asarray([True, True, False]))
    assert np.array_equal(np.asarray(padded_diagnostics["masks"]["rejected"]), np.asarray([False, False, False]))
    assert np.array_equal(np.asarray(padded_diagnostics["masks"]["done"]), np.asarray([False, True, True]))
    assert padded_diagnostics["replay_diagnostics"]["preconditioner_policy_n_segments"] == 1
    assert padded_diagnostics["replay_diagnostics"]["scalar_controls_stackable"] is True
    assert padded_diagnostics["replay_diagnostics"]["array_controls_stackable"] is True
    assert padded_diagnostics["replay_diagnostics"]["preconditioner_controls_stackable"] is True
    padded_json = free_boundary_adjoint_trace_replay_diagnostics(
        {"diagnostics": {"adjoint_step_trace": [trace0, trace1, trace2]}},
        accept_mask=np.asarray([True, True, False]),
        done_mask=np.asarray([False, True, False]),
        json_safe=True,
    )
    json.dumps(padded_json, allow_nan=False)
    assert padded_json["masks"]["done"] == [False, True, True]
    with pytest.raises(RuntimeError, match="adjoint_trace=True"):
        free_boundary_adjoint_trace_replay_diagnostics({"diagnostics": {}})
    synthetic_fingerprint = direct_coil_accepted_trace_fingerprint([trace0, trace1])
    synthetic_report = {
        "branch_compatibility": {
            "same_branch": True,
            "same_accepted_trace_branch": True,
            "same_residual_branch": True,
            "base_fingerprint": synthetic_fingerprint,
            "plus_fingerprint": synthetic_fingerprint,
            "minus_fingerprint": synthetic_fingerprint,
        },
        "trace_replay_diagnostics": {
            "base": free_boundary_adjoint_trace_replay_diagnostics([trace0, trace1]),
            "plus": free_boundary_adjoint_trace_replay_diagnostics([trace0, trace1]),
            "minus": free_boundary_adjoint_trace_replay_diagnostics([trace0, trace1]),
        },
    }
    synthetic_gate = direct_coil_same_branch_replay_gate_report(synthetic_report)
    assert synthetic_gate["passed"], synthetic_gate
    json.dumps(direct_coil_same_branch_replay_gate_report(synthetic_report, json_safe=True), allow_nan=False)
    bad_synthetic_report = deepcopy(synthetic_report)
    bad_synthetic_report["trace_replay_diagnostics"]["plus"]["differentiates_adaptive_controller"] = True
    bad_synthetic_gate = direct_coil_same_branch_replay_gate_report(bad_synthetic_report)
    assert not bad_synthetic_gate["passed"]
    assert any("adaptive-controller" in error for error in bad_synthetic_gate["errors"])
    mismatch_synthetic_report = deepcopy(synthetic_report)
    mismatch_synthetic_report["branch_compatibility"].pop("base_fingerprint")
    mismatch_synthetic_report["trace_replay_diagnostics"]["plus"] = {
        "differentiates_adaptive_controller": False,
        "n_steps": 99,
        "branch_fingerprint": {"n_steps": 98, "n_freeb_steps": 97, "freeb_sizes": np.asarray([99])},
        "masks": {
            "active": np.asarray([True]),
            "accepted": np.asarray([False]),
            "rejected": np.asarray([True]),
            "done": np.asarray([False]),
            "has_active_freeb_replay": np.asarray([False]),
        },
        "replay_diagnostics": {
            "scalar_controls_stackable": False,
            "array_controls_stackable": False,
            "preconditioner_policy_n_segments": 0,
        },
    }
    mismatch_synthetic_report["trace_replay_diagnostics"]["minus"] = "missing"
    mismatch_gate = direct_coil_same_branch_replay_gate_report(mismatch_synthetic_report)
    assert not mismatch_gate["passed"]
    assert any("base: missing branch fingerprint" in error for error in mismatch_gate["errors"])
    assert any("plus: n_steps mismatch" in error for error in mismatch_gate["errors"])
    assert any("plus: fingerprint n_steps mismatch" in error for error in mismatch_gate["errors"])
    assert any("plus: fingerprint n_freeb_steps mismatch" in error for error in mismatch_gate["errors"])
    assert any("plus: freeb_sizes mismatch" in error for error in mismatch_gate["errors"])
    assert any("plus: mask 'active' has shape" in error for error in mismatch_gate["errors"])
    assert any("plus: no accepted active free-boundary replay slots" in error for error in mismatch_gate["errors"])
    assert any("plus: scalar controls are not stackable" in error for error in mismatch_gate["errors"])
    assert any("plus: array controls are not stackable" in error for error in mismatch_gate["errors"])
    assert any("plus: no preconditioner policy segments" in error for error in mismatch_gate["errors"])
    assert any("minus: missing replay diagnostics" in error for error in mismatch_gate["errors"])

    from vmec_jax.free_boundary_adjoint import (
        _pytree_batched_directional_vdot_jax,
        direct_coil_accepted_trace_controller_custom_vjp_scalars_jax,
        direct_coil_adaptive_full_loop_same_branch_gate_report,
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax,
        direct_coil_same_branch_physical_scalar_gate_report,
        direct_coil_same_branch_controller_scalar_custom_vjp_report,
        direct_coil_same_branch_controller_scalars_custom_vjp_report,
    )

    physical_synthetic_report = deepcopy(synthetic_report)
    physical_synthetic_report["base"] = {"traces": (trace0, trace1)}
    physical_synthetic_report["plus"] = {"traces": (trace0, trace1)}
    physical_synthetic_report["minus"] = {"traces": (trace0, trace1)}
    physical_synthetic_report["objective_values"] = {
        "aspect": {"base": 5.0, "plus": 5.01, "minus": 4.99, "central_fd_directional": 100.0},
        "accepted_bnormal_rms": {
            "base": 0.2,
            "plus": 0.21,
            "minus": 0.19,
            "central_fd_directional": 100.0,
        },
    }
    physical_scalars_report = {
        "same_branch": True,
        "replay_option_flags": {"use_stacked_step_controls": True},
        "scalar_keys": ("aspect", "accepted_bnormal_rms"),
        "scalar_reports": {
            "aspect": {
                "passed": True,
                "same_branch": True,
                "exact_directional": 100.0,
                "abs_error": 0.0,
                "rel_error": 0.0,
                "base_abs_delta": 1.0e-6,
            },
            "accepted_bnormal_rms": {
                "passed": True,
                "same_branch": True,
                "exact_directional": 100.0,
                "abs_error": 0.0,
                "rel_error": 0.0,
                "base_abs_delta": 1.0e-6,
            },
        },
    }
    physical_gate = direct_coil_same_branch_physical_scalar_gate_report(
        physical_synthetic_report,
        physical_scalars_report,
    )
    assert physical_gate["passed"], physical_gate
    assert physical_gate["scalar_keys"] == ("aspect", "accepted_bnormal_rms")
    assert physical_gate["differentiates_adaptive_controller"] is False
    assert physical_gate["same_accepted_trace_branch"] is True
    assert physical_gate["same_residual_branch"] is True
    adaptive_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        physical_synthetic_report,
        physical_scalars_report,
    )
    assert adaptive_gate["passed"], adaptive_gate
    assert adaptive_gate["contract"] == "same-branch adaptive full-loop seam report"
    assert adaptive_gate["differentiates_adaptive_controller"] is False
    assert adaptive_gate["differentiates_run_free_boundary"] is False
    assert adaptive_gate["same_stacked_step_policy_branch"] is True
    assert adaptive_gate["used_stacked_step_controls"] is True
    unstacked_allowed_scalars_report = deepcopy(physical_scalars_report)
    unstacked_allowed_scalars_report["replay_option_flags"] = {"use_stacked_step_controls": False}
    unstacked_allowed_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        physical_synthetic_report,
        unstacked_allowed_scalars_report,
        require_stacked_step_controls=False,
    )
    assert unstacked_allowed_gate["passed"], unstacked_allowed_gate
    assert unstacked_allowed_gate["requires_stacked_step_controls"] is False
    assert unstacked_allowed_gate["used_stacked_step_controls"] is False
    json.dumps(
        direct_coil_adaptive_full_loop_same_branch_gate_report(
            physical_synthetic_report,
            physical_scalars_report,
            json_safe=True,
        ),
        allow_nan=False,
    )
    json.dumps(
        direct_coil_same_branch_physical_scalar_gate_report(
            physical_synthetic_report,
            physical_scalars_report,
            json_safe=True,
        ),
        allow_nan=False,
    )
    bad_physical_scalars_report = deepcopy(physical_scalars_report)
    bad_physical_scalars_report["same_branch"] = False
    bad_physical_scalars_report["scalar_reports"]["aspect"]["passed"] = False
    bad_physical_scalars_report["scalar_reports"]["aspect"]["exact_directional"] = np.nan
    bad_physical_scalars_report["scalar_reports"]["missing_objective"] = {
        "passed": True,
        "same_branch": True,
        "exact_directional": 1.0,
        "abs_error": 0.0,
        "rel_error": 0.0,
        "base_abs_delta": 0.0,
    }
    bad_physical_report = deepcopy(physical_synthetic_report)
    bad_physical_report["branch_compatibility"]["same_branch"] = False
    bad_physical_report["branch_compatibility"]["same_accepted_trace_branch"] = False
    bad_physical_report["branch_compatibility"]["same_residual_branch"] = False
    bad_physical_report["objective_values"]["accepted_bnormal_rms"]["central_fd_directional"] = np.nan
    changed_policy_trace = {**trace1, "include_edge_residual": True}
    bad_physical_report["base"] = {"traces": ()}
    bad_physical_report["plus"] = {"traces": (trace0, changed_policy_trace)}
    bad_physical_report.pop("minus")
    bad_physical_scalars_report["replay_option_flags"] = {"use_stacked_step_controls": False}
    bad_physical_gate = direct_coil_same_branch_physical_scalar_gate_report(
        bad_physical_report,
        bad_physical_scalars_report,
        scalar_keys=("aspect", "accepted_bnormal_rms", "missing", "missing_objective"),
    )
    assert not bad_physical_gate["passed"]
    assert any("replay gate failed" in error for error in bad_physical_gate["errors"])
    assert any("not same-branch" in error for error in bad_physical_gate["errors"])
    assert any("accepted-trace branch fingerprint changed" in error for error in bad_physical_gate["errors"])
    assert any("residual-controller branch fingerprint changed" in error for error in bad_physical_gate["errors"])
    assert any("missing scalar report" in error for error in bad_physical_gate["errors"])
    assert any("missing complete-solve objective values" in error for error in bad_physical_gate["errors"])
    assert any("non-finite complete-solve FD" in error for error in bad_physical_gate["errors"])
    assert any("non-finite custom-VJP" in error for error in bad_physical_gate["errors"])
    bad_adaptive_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        bad_physical_report,
        bad_physical_scalars_report,
        scalar_keys=("aspect", "accepted_bnormal_rms", "missing", "missing_objective"),
    )
    assert not bad_adaptive_gate["passed"]
    assert any("stacked step-control replay was not used" in error for error in bad_adaptive_gate["errors"])
    assert any("stacked step-policy branch changed" in error for error in bad_adaptive_gate["errors"])
    assert any("base: no accepted step-policy segments" in error for error in bad_adaptive_gate["errors"])
    assert any("minus: missing complete-solve payload" in error for error in bad_adaptive_gate["errors"])
    assert any("physical scalar gate:" in error for error in bad_adaptive_gate["errors"])

    with pytest.raises(ValueError, match="input_path and params"):
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
            scalar_fn=lambda payload: {"objective": 0.0},
            replay_scalar_fn=lambda replay, payload: 0.0,
        )
    with pytest.raises(ValueError, match="params must be supplied"):
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
            complete_payload={"traces": (), "init": object()},
            scalar_fn=lambda payload: {"objective": 0.0},
            replay_scalar_fn=lambda replay, payload: 0.0,
        )
    with pytest.raises(ValueError, match="no accepted traces"):
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
            params={},
            complete_payload={"traces": (), "init": object()},
            scalar_fn=lambda payload: {"objective": 0.0},
            replay_scalar_fn=lambda replay, payload: 0.0,
        )
    with pytest.raises(RuntimeError, match="no active free-boundary trace"):
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
            params={},
            complete_payload={"traces": ({"freeb_bsqvac_half": None},), "init": object()},
            scalar_fn=lambda payload: {"objective": 0.0},
            replay_scalar_fn=lambda replay, payload: 0.0,
        )
    with pytest.raises(ValueError, match="initialization result"):
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
            params={},
            complete_payload={"traces": ({"freeb_bsqvac_half": np.ones(1)},)},
            scalar_fn=lambda payload: {"objective": 0.0},
            replay_scalar_fn=lambda replay, payload: 0.0,
        )

    jacobian_tree = {
        "a": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "b": jnp.asarray([[0.5], [-1.0]]),
    }
    direction_tree = {"a": jnp.asarray([10.0, -2.0]), "b": jnp.asarray([4.0])}
    contracted = _pytree_batched_directional_vdot_jax(jacobian_tree, direction_tree, 2)
    np.testing.assert_allclose(np.asarray(contracted), np.asarray([8.0, 18.0]))
    np.testing.assert_allclose(
        np.asarray(_pytree_batched_directional_vdot_jax({}, {}, 3)),
        np.zeros(3),
    )

    with pytest.raises(ValueError, match="replay_scalar_fns"):
        direct_coil_same_branch_controller_scalars_custom_vjp_report(
            {"objective_values": {}},
            base_params={},
            direction={},
            replay_scalar_fns={},
        )
    with pytest.raises(ValueError, match="scalar_fns"):
        direct_coil_accepted_trace_controller_custom_vjp_scalars_jax(
            {},
            None,
            scalar_fns=(),
        )
    with pytest.raises(KeyError, match="not present"):
        direct_coil_same_branch_controller_scalar_custom_vjp_report(
            {"objective_values": {}},
            base_params={},
            direction={},
            scalar_key="missing",
            replay_scalar_fn=lambda _replay, _payload: 0.0,
        )
    with pytest.raises(KeyError, match="not present"):
        direct_coil_same_branch_controller_scalars_custom_vjp_report(
            {"objective_values": {"known": {"base": 0.0, "central_fd_directional": 0.0}}},
            base_params={},
            direction={},
            replay_scalar_fns={"missing": lambda _replay, _payload: 0.0},
        )
    with pytest.raises(ValueError, match="accepted traces"):
        direct_coil_same_branch_controller_scalars_custom_vjp_report(
            {
                "objective_values": {"known": {"base": 0.0, "central_fd_directional": 0.0}},
                "base": {"traces": ()},
            },
            base_params={},
            direction={},
            replay_scalar_fns={"known": lambda _replay, _payload: 0.0},
        )

    bad_preconditioner_shape = dict(trace1)
    bad_preconditioner_shape["precond_mats"] = {"ar": np.ones((3, 3)), "br": z + 7.0}
    with pytest.raises(ValueError, match="precond_mats"):
        direct_coil_accepted_trace_preconditioner_controls_jax([trace0, bad_preconditioner_shape])

    fingerprint = direct_coil_accepted_trace_fingerprint([trace0, trace1])
    assert fingerprint["n_steps"] == 2
    assert fingerprint["n_freeb_steps"] == 2
    assert np.array_equal(fingerprint["freeb_sizes"], np.asarray([6, 6]))

    same = direct_coil_accepted_trace_fingerprint_delta([trace0, trace1], [trace0, trace1])
    assert same["compatible"]
    same_json = direct_coil_accepted_trace_fingerprint_delta_summary([trace0, trace1], [trace0, trace1])
    json.dumps(same_json, allow_nan=False)
    assert same_json["compatible"]
    assert same_json["reference"]["precond_jmax"] == [2, 2]

    field_only_change = dict(trace0)
    field_only_change["freeb_bsqvac_half"] = np.ones((2, 3)) * 99.0
    same_branch = direct_coil_accepted_trace_fingerprint_delta(
        [trace0, trace1],
        [field_only_change, trace1],
    )
    assert same_branch["compatible"]

    control_change = dict(trace0)
    control_change["fac"] = np.asarray(0.7)
    different_branch = direct_coil_accepted_trace_fingerprint_delta(
        [trace0, trace1],
        [control_change, trace1],
    )
    assert not different_branch["compatible"]
    assert "scalars.fac" in different_branch["changed_fields"]

    b1_change = dict(trace0)
    b1_change["b1"] = np.asarray(0.25)
    different_b1 = direct_coil_accepted_trace_fingerprint_delta(
        [trace0, trace1],
        [b1_change, trace1],
    )
    assert not different_b1["compatible"]
    assert "scalars.b1" in different_b1["changed_fields"]

    preconditioner_policy_change = dict(trace0)
    preconditioner_policy_change["preconditioner_use_lax_tridi"] = False
    policy_segments = direct_coil_accepted_trace_preconditioner_policy_segments(
        [trace0, preconditioner_policy_change, trace1]
    )
    assert [(segment["start"], segment["stop"], segment["n_steps"]) for segment in policy_segments] == [
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 1),
    ]
    different_preconditioner_policy = direct_coil_accepted_trace_fingerprint_delta(
        [trace0, trace1],
        [preconditioner_policy_change, trace1],
    )
    assert not different_preconditioner_policy["compatible"]
    assert "flags.preconditioner_use_lax_tridi" in different_preconditioner_policy["changed_fields"]

    preconditioner_jmax_change = dict(trace0)
    preconditioner_jmax_change["precond_jmax"] = 3
    different_preconditioner_jmax = direct_coil_accepted_trace_fingerprint_delta(
        [trace0, trace1],
        [preconditioner_jmax_change, trace1],
    )
    assert not different_preconditioner_jmax["compatible"]
    assert "precond_jmax" in different_preconditioner_jmax["changed_fields"]

    preconditioner_shape_change = dict(trace0)
    preconditioner_shape_change["precond_mats"] = {"ar": np.ones((3, 3)), "br": z + 7.0}
    different_preconditioner_shape = direct_coil_accepted_trace_fingerprint_delta(
        [trace0, trace1],
        [preconditioner_shape_change, trace1],
    )
    assert not different_preconditioner_shape["compatible"]
    assert "precond_mats_shapes" in different_preconditioner_shape["changed_fields"]

    size_change = dict(trace0)
    size_change["freeb_bsqvac_half"] = np.ones((3, 3))
    different_shape = direct_coil_accepted_trace_fingerprint_delta(
        [trace0, trace1],
        [size_change, trace1],
    )
    assert not different_shape["compatible"]
    assert "freeb_sizes" in different_shape["changed_fields"]


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


def _final_residuals(run) -> np.ndarray:
    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    return np.asarray([diag.get("final_fsqr"), diag.get("final_fsqz"), diag.get("final_fsql")], dtype=float)


def _trace_scalar_value(value: object) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _assert_full_solve_wout_sanity(run, wout_path: Path) -> None:
    from vmec_jax.driver import load_wout, write_wout_from_fixed_boundary_run

    pressure = _pressure_profile(run)
    residuals = _final_residuals(run)
    assert np.max(pressure) > 0.0
    assert np.all(np.isfinite(residuals))
    assert np.all(residuals >= 0.0)

    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True, fast_bcovar=True)
    assert wout_path.exists()
    wout = load_wout(wout_path)
    wout_residuals = np.asarray([wout.fsqr, wout.fsqz, wout.fsql], dtype=float)
    assert np.all(np.isfinite(wout_residuals))
    assert np.all(wout_residuals >= 0.0)
    assert np.max(np.asarray(wout.presf, dtype=float)) > 0.0
    for field in ("rmnc", "zmns", "lmns", "phipf", "iotaf"):
        values = np.asarray(getattr(wout, field), dtype=float)
        assert values.size > 0
        assert np.all(np.isfinite(values)), f"non-finite WOUT field {field}"


def _first_nonzero_current_index(params: CoilFieldParams) -> int:
    currents = np.asarray(params.base_currents, dtype=float).reshape(-1)
    nonzero = np.flatnonzero(np.abs(currents) > 0.0)
    if nonzero.size == 0:
        pytest.skip("ESSOS coil file does not expose a nonzero current DOF")
    return int(nonzero[0])


def _first_fourier_geometry_index(params: CoilFieldParams) -> tuple[int, int, int]:
    dofs = np.asarray(params.base_curve_dofs, dtype=float)
    if dofs.ndim != 3 or dofs.shape[2] <= 1:
        pytest.skip("ESSOS coil file does not expose Fourier geometry DOFs")
    nonconstant = np.argwhere(np.abs(dofs[:, :, 1:]) > 0.0)
    if nonconstant.size == 0:
        return 0, 0, 1
    i, xyz, shifted_mode = (int(v) for v in nonconstant[0])
    return i, xyz, shifted_mode + 1


def _state_central_difference_rms(plus_run, minus_run, *, step: float) -> float:
    plus = np.asarray(pack_state(plus_run.state), dtype=float)
    minus = np.asarray(pack_state(minus_run.state), dtype=float)
    assert plus.shape == minus.shape
    diff = (plus - minus) / (2.0 * float(step))
    assert np.all(np.isfinite(diff))
    return float(np.sqrt(np.mean(diff * diff)))


def test_active_direct_coil_provider_is_sensitive_in_finite_pressure_context(tmp_path: Path) -> None:
    """Active NESTOR sampling should scale consistently with direct-coil current."""

    enable_x64(True)
    base_current = 3.0e7
    perturbed_current = 3.3e7
    current_ratio = perturbed_current / base_current
    base_params = _circle_coil_params(current=base_current)
    perturbed_params = _circle_coil_params(current=perturbed_current)
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
    for key in ("bnormal_rms", "bnormal_unit_rms", "rhs_rms", "gsource_rms"):
        np.testing.assert_allclose(
            perturbed.diagnostics[key],
            current_ratio * base.diagnostics[key],
            rtol=1.0e-12,
            atol=1.0e-12,
            err_msg=f"{key} should scale linearly with direct-coil current",
        )
    np.testing.assert_allclose(
        perturbed.phi,
        current_ratio * base.phi,
        rtol=1.0e-11,
        atol=1.0e-12,
        err_msg="NESTOR potential should scale linearly with direct-coil current",
    )
    np.testing.assert_allclose(
        perturbed.vac_total.bsqvac,
        current_ratio**2 * base.vac_total.bsqvac,
        rtol=1.0e-11,
        atol=1.0e-12,
        err_msg="vacuum B^2 should scale quadratically with direct-coil current",
    )
    np.testing.assert_allclose(
        perturbed.diagnostics["bsqvac_rms"],
        current_ratio**2 * base.diagnostics["bsqvac_rms"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
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


def test_forced_active_direct_coil_finite_pressure_solve_has_physics_diagnostics(
    tmp_path: Path,
) -> None:
    """A tiny active direct-coil finite-pressure solve exposes active NESTOR diagnostics."""

    enable_x64(True)
    from vmec_jax.driver import run_free_boundary
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

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

    diag = run.result.diagnostics
    freeb = diag["free_boundary"]
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

    pressure = _pressure_profile(run)
    aspect = float(np.asarray(equilibrium_aspect_ratio_from_state(state=run.state, static=run.static)))
    _chips, iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=run.state,
        static=run.static,
        indata=run.indata,
        signgs=int(run.signgs),
    )
    iotas = np.asarray(iotas, dtype=float)
    iotaf = np.asarray(iotaf, dtype=float)
    residuals = np.asarray([diag["final_fsqr"], diag["final_fsqz"], diag["final_fsql"]], dtype=float)

    assert np.max(pressure) > 0.0
    assert np.all(np.isfinite(residuals))
    assert np.all(residuals >= 0.0)
    assert np.isfinite(aspect)
    assert aspect > 1.0
    assert iotas.size > 0
    assert np.all(np.isfinite(iotas))
    assert np.all(np.isfinite(iotaf))


def test_active_direct_coil_adjoint_trace_records_vacuum_forcing_and_pressure_scale(tmp_path: Path) -> None:
    """Accepted active free-boundary steps must carry vacuum forcing into replay traces."""

    enable_x64(True)
    from vmec_jax.driver import run_free_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    params = _circle_coil_params(current=3.0e7, n_segments=32)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_provider_adjoint_trace_forcing",
        niter=2,
        mpol=3,
        ntheta=6,
    )
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
        max_iter=2,
        ftol=1.0e-8,
        vmec2000_control=True,
        auto_flip_force=False,
        use_direct_fallback=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces=False,
        adjoint_trace=True,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
        free_boundary_activate_fsq=1.0e99,
    )

    freeb = result.diagnostics["free_boundary"]
    assert freeb["vacuum_stub"] is False
    traces = result.diagnostics.get("adjoint_step_trace", [])
    active_traces = [trace for trace in traces if trace.get("freeb_bsqvac_half") is not None]
    assert active_traces
    for trace in active_traces:
        vac = np.asarray(trace["freeb_bsqvac_half"], dtype=float)
        assert vac.ndim == 2
        assert vac.size > 0
        assert np.all(np.isfinite(vac))
        assert float(np.linalg.norm(vac)) > 0.0
        assert trace["freeb_pres_scale"] is not None
        assert np.isfinite(float(trace["freeb_pres_scale"]))


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


def test_direct_coil_complete_solve_proxy_objective_fd_response_for_current_and_geometry(tmp_path: Path) -> None:
    """The phase-1 coil-only proxy objective should respond smoothly to coil controls."""

    enable_x64(True)
    from examples.optimization.free_boundary_QS_coil_optimization import (
        apply_coil_variables,
        objective_from_summary,
        objective_terms_from_summary,
        run_direct_free_boundary,
        summarize_run,
    )

    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_proxy_objective_fd",
        niter=2,
        mpol=3,
        ntheta=6,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=32)
    variables = [("current", (0,)), ("fourier_dof", (0, 0, 2))]
    residual_weight = 1.0
    aspect_weight = 0.02
    iota_weight = 0.02

    def objective(x_current: float, x_geometry: float) -> float:
        params = apply_coil_variables(
            base_params,
            np.asarray([x_current, x_geometry], dtype=float),
            variables=variables,
            current_step=0.02,
            dof_step=1.0e-2,
        )
        run, wall_s = run_direct_free_boundary(
            input_path,
            params,
            vmec_max_iter=2,
            activate_fsq=1.0e99,
            jit_forces=False,
        )
        summary = summarize_run(
            run,
            params,
            objective=np.nan,
            wall_s=wall_s,
            target_aspect=6.0,
            target_iota=0.4,
        )
        terms = objective_terms_from_summary(
            summary,
            residual_weight=residual_weight,
            aspect_weight=aspect_weight,
            iota_weight=iota_weight,
        )
        assert summary["free_boundary_vacuum_stub"] is False
        assert summary["free_boundary_nestor_model"].startswith("vmec2000_like_dense_integral")
        assert not terms["missing_unweighted_terms"]
        assert terms["residual"]["contribution"] > 0.0
        return objective_from_summary(
            summary,
            residual_weight=residual_weight,
            aspect_weight=aspect_weight,
            iota_weight=iota_weight,
        )

    eps = 0.2
    base = objective(0.0, 0.0)
    current_slope = (objective(eps, 0.0) - objective(-eps, 0.0)) / (2.0 * eps)
    geometry_slope = (objective(0.0, eps) - objective(0.0, -eps)) / (2.0 * eps)

    values = np.asarray([base, current_slope, geometry_slope], dtype=float)
    assert np.all(np.isfinite(values))
    assert base > 0.0
    assert abs(float(current_slope)) > 1.0e-12
    assert abs(float(geometry_slope)) > 1.0e-12


@pytest.mark.py311_coverage_only
def test_jax_nestor_operator_complete_solve_fd_slope_for_mixed_coil_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in JAX NESTOR complete solves should respond to mixed coil variables."""

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
    base_currents = jnp.asarray(base_params.base_currents)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)

    def params_for(scale: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_currents=base_currents * (1.0 + 0.02 * float(scale)),
            base_curve_dofs=base_dofs.at[0, 0, 2].add(1.0e-2 * float(scale)),
        )

    # This is still an outer solve finite-response guard.  The driver path
    # materializes host NumPy state/diagnostics between iterations, so it should
    # not be treated as full-loop AD validation.
    mixed_slope = (metric(params_for(eps)) - metric(params_for(-eps))) / (2.0 * eps)

    assert np.isfinite(float(mixed_slope))
    assert abs(float(mixed_slope)) > 1.0e-16


def _set_same_branch_custom_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "VMEC_JAX_FREEB_NESTOR_MODE": "dense",
        "VMEC_JAX_FREEB_DENSE_SOLVE_MODE": "mode",
        "VMEC_JAX_FREEB_USE_GREENF_SOURCE": "1",
        "VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX": "1",
        "VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC": "1",
        "VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR": "1",
        "VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR": "0",
    }.items():
        monkeypatch.setenv(key, value)


def _assert_direct_coil_same_branch_custom_vjp_matches_complete_fd(
    *,
    input_path: Path,
    base_params: CoilFieldParams,
    direction: CoilFieldParams,
    params_for,
    check_controller: bool = True,
    check_segmented_controller: bool = True,
    check_aspect_scalar: bool = True,
    check_boundary_moment_scalar: bool = False,
    check_accepted_bnormal_rms_scalar: bool = False,
    check_accepted_bsqvac_rms_scalar: bool = False,
    check_production_branch_local_scalar: bool = False,
) -> None:
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp
    from vmec_jax.free_boundary_adjoint import (
        direct_coil_accepted_trace_controller_custom_vjp_objective_jax,
        direct_coil_adaptive_full_loop_same_branch_gate_report,
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax,
        free_boundary_boundary_geometry_jax,
        direct_coil_same_branch_physical_scalar_gate_report,
        direct_coil_same_branch_controller_scalar_custom_vjp_report,
        direct_coil_same_branch_controller_scalars_custom_vjp_report,
        direct_coil_same_branch_complete_solve_fd_report,
        direct_coil_same_branch_replay_gate_report,
        direct_coil_fixed_trace_custom_vjp_objective_jax,
    )
    from vmec_jax.state import pack_state
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    def state_norm_objective(state) -> float:
        packed = np.asarray(pack_state(state), dtype=float)
        return float(0.5 * np.vdot(packed, packed))

    def lcfs_boundary_moment(state, static):
        geometry = free_boundary_boundary_geometry_jax(state, static)
        R = jnp.asarray(geometry["R"])
        Z = jnp.asarray(geometry["Z"])
        return jnp.mean((R - 1.0) * (R - 1.0) + Z * Z)

    def accepted_bsqvac_rms_from_payload(payload) -> float:
        values = [
            float(np.sqrt(np.mean(np.square(np.asarray(trace["freeb_bsqvac_half"], dtype=float)))))
            for trace in payload["traces"]
            if trace.get("freeb_bsqvac_half") is not None
        ]
        if not values:
            return 0.0
        return float(np.mean(values))

    def accepted_bnormal_rms_from_payload(payload) -> float:
        values = [
            float(np.sqrt(np.mean(np.square(np.asarray(trace["freeb_nestor_trace"]["bnormal"], dtype=float)))))
            for trace in payload["traces"]
            if trace.get("freeb_bsqvac_half") is not None
            and isinstance(trace.get("freeb_nestor_trace"), dict)
            and trace["freeb_nestor_trace"].get("bnormal") is not None
        ]
        if not values:
            return 0.0
        return float(np.mean(values))

    def accepted_bnormal_rms_from_replay(replay) -> object:
        accepted = jnp.asarray(replay["history"]["accepted"], dtype=jnp.asarray(replay["history"]["bnormal_rms"]).dtype)
        active = jnp.asarray(
            replay["controls"]["has_active_freeb_replay"],
            dtype=jnp.asarray(replay["history"]["bnormal_rms"]).dtype,
        )
        weights = accepted * active
        denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=weights.dtype))
        return jnp.sum(weights * jnp.asarray(replay["history"]["bnormal_rms"])) / denom

    def accepted_bsqvac_rms_from_replay(replay) -> object:
        accepted = jnp.asarray(replay["history"]["accepted"], dtype=jnp.asarray(replay["history"]["bsqvac_rms"]).dtype)
        active = jnp.asarray(
            replay["controls"]["has_active_freeb_replay"],
            dtype=jnp.asarray(replay["history"]["bsqvac_rms"]).dtype,
        )
        weights = accepted * active
        denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=weights.dtype))
        return jnp.sum(weights * jnp.asarray(replay["history"]["bsqvac_rms"])) / denom

    eps = 1.0e-4
    complete_report = direct_coil_same_branch_complete_solve_fd_report(
        input_path,
        base_params,
        params_for=params_for,
        objective_fn=lambda payload: {
            "objective": state_norm_objective(payload["result"].state),
            "aspect": float(
                np.asarray(
                    equilibrium_aspect_ratio_from_state(
                        state=payload["result"].state,
                        static=payload["init"].static,
                    )
                )
            ),
            "lcfs_boundary_moment": float(np.asarray(lcfs_boundary_moment(payload["result"].state, payload["init"].static))),
            "accepted_bnormal_rms": accepted_bnormal_rms_from_payload(payload),
            "accepted_bsqvac_rms": accepted_bsqvac_rms_from_payload(payload),
        },
        eps=eps,
        solve_kwargs={
            "max_iter": 2,
            "ftol": 1.0e-8,
            "vmec2000_control": True,
            "auto_flip_force": False,
            "use_direct_fallback": True,
            "verbose": False,
            "verbose_vmec2000_table": False,
            "jit_forces": False,
            "use_scan": False,
            "host_update_assembly": False,
            "adjoint_trace": True,
            "adjoint_trace_mode": "full",
            "external_field_provider_kind": "direct_coils",
            "free_boundary_activate_fsq": 1.0e99,
        },
        fingerprint_rtol=1.0e-6,
        fingerprint_atol=1.0e-9,
    )
    base_init = complete_report["base"]["init"]
    base_result = complete_report["base"]["result"]
    base_traces = complete_report["base"]["traces"]
    plus_result = complete_report["plus"]["result"]
    minus_result = complete_report["minus"]["result"]
    plus_branch = complete_report["branch_compatibility"]["plus"]
    minus_branch = complete_report["branch_compatibility"]["minus"]
    base_fingerprint = complete_report["branch_compatibility"]["base_fingerprint"]
    plus_fingerprint = complete_report["branch_compatibility"]["plus_fingerprint"]
    minus_fingerprint = complete_report["branch_compatibility"]["minus_fingerprint"]
    base_residual_fingerprint = complete_report["branch_compatibility"]["base_residual_fingerprint"]
    plus_residual_fingerprint = complete_report["branch_compatibility"]["plus_residual_fingerprint"]
    minus_residual_fingerprint = complete_report["branch_compatibility"]["minus_residual_fingerprint"]
    assert complete_report["branch_compatibility"]["same_branch"] is True
    assert complete_report["branch_compatibility"]["same_accepted_trace_branch"] is True
    assert complete_report["branch_compatibility"]["same_residual_branch"] is True
    assert plus_branch["compatible"], plus_branch["changed_fields"]
    assert minus_branch["compatible"], minus_branch["changed_fields"]
    assert base_fingerprint["n_steps"] == plus_fingerprint["n_steps"] == minus_fingerprint["n_steps"]
    assert base_residual_fingerprint == plus_residual_fingerprint == minus_residual_fingerprint
    assert base_fingerprint["n_freeb_steps"] > 0
    assert plus_fingerprint["n_freeb_steps"] == base_fingerprint["n_freeb_steps"]
    assert minus_fingerprint["n_freeb_steps"] == base_fingerprint["n_freeb_steps"]
    np.testing.assert_array_equal(plus_fingerprint["freeb_sizes"], base_fingerprint["freeb_sizes"])
    np.testing.assert_array_equal(minus_fingerprint["freeb_sizes"], base_fingerprint["freeb_sizes"])
    expected_fingerprints = {
        "base": base_fingerprint,
        "plus": plus_fingerprint,
        "minus": minus_fingerprint,
    }
    trace_replay_diagnostics = complete_report["trace_replay_diagnostics"]
    assert set(trace_replay_diagnostics) == set(expected_fingerprints)
    for label, replay_diagnostics in trace_replay_diagnostics.items():
        fingerprint = expected_fingerprints[label]
        assert replay_diagnostics["contract"] == "fixed accepted-trace replay diagnostics only"
        assert replay_diagnostics["differentiates_adaptive_controller"] is False
        assert replay_diagnostics["n_steps"] == fingerprint["n_steps"]
        assert replay_diagnostics["branch_fingerprint"]["n_steps"] == fingerprint["n_steps"]
        assert replay_diagnostics["branch_fingerprint"]["n_freeb_steps"] == fingerprint["n_freeb_steps"]
        np.testing.assert_array_equal(replay_diagnostics["branch_fingerprint"]["freeb_sizes"], fingerprint["freeb_sizes"])
        for mask_key in ("active", "accepted", "rejected", "done", "has_active_freeb_replay"):
            assert np.asarray(replay_diagnostics["masks"][mask_key], dtype=bool).shape == (fingerprint["n_steps"],)
        assert bool(np.any(np.asarray(replay_diagnostics["masks"]["accepted"], dtype=bool)))
        assert bool(np.any(np.asarray(replay_diagnostics["masks"]["has_active_freeb_replay"], dtype=bool)))
        replay_payload = replay_diagnostics["replay_diagnostics"]
        assert replay_payload["scalar_controls_stackable"] is True
        assert replay_payload["array_controls_stackable"] is True
        assert replay_payload["preconditioner_policy_n_segments"] >= 1
        assert sum(
            segment["free_boundary_replay_steps"]
            for segment in replay_payload["preconditioner_policy_segment_summary"]
        ) == fingerprint["n_freeb_steps"]
        if not replay_payload["preconditioner_controls_stackable"]:
            assert "preconditioner_controls" in replay_payload["errors"]
    replay_gate = direct_coil_same_branch_replay_gate_report(complete_report)
    assert replay_gate["passed"], replay_gate
    assert replay_gate["contract"] == "same-branch accepted-trace replay gate"
    assert replay_gate["same_branch"] is True
    assert replay_gate["differentiates_adaptive_controller"] is False
    json.dumps(direct_coil_same_branch_replay_gate_report(complete_report, json_safe=True), allow_nan=False)

    assert complete_report["primary_objective"] == "objective"
    assert set(complete_report["objective_values"]) == {
        "objective",
        "aspect",
        "lcfs_boundary_moment",
        "accepted_bnormal_rms",
        "accepted_bsqvac_rms",
    }
    complete_fd = float(complete_report["values"]["central_fd_directional"])

    def custom_objective(params: CoilFieldParams):
        return direct_coil_fixed_trace_custom_vjp_objective_jax(
            params,
            base_traces[0]["state_pre"],
            static=base_init.static,
            traces=base_traces,
            signgs=int(base_init.signgs),
            state_weight=1.0,
            bsqvac_weight=0.0,
            force_weight=0.0,
            enforce_edge=False,
        )

    grad = jax.grad(custom_objective)(base_params)
    exact = sum(
        jnp.vdot(grad_leaf, direction_leaf)
        for grad_leaf, direction_leaf in zip(
            jax.tree_util.tree_leaves(grad),
            jax.tree_util.tree_leaves(direction),
            strict=True,
        )
    )
    base_complete = state_norm_objective(base_result.state)
    base_fixed_trace = float(np.asarray(custom_objective(base_params)))
    assert abs(base_fixed_trace - base_complete) < 2.0e-3
    assert np.isfinite(float(np.asarray(exact)))
    assert np.isfinite(float(complete_fd))
    np.testing.assert_allclose(exact, complete_fd, rtol=2.0e-3, atol=1.0e-8)

    controller_exact = None
    if check_controller:
        def controller_custom_objective(params: CoilFieldParams):
            return direct_coil_accepted_trace_controller_custom_vjp_objective_jax(
                params,
                base_traces[0]["state_pre"],
                static=base_init.static,
                traces=base_traces,
                signgs=int(base_init.signgs),
                state_weight=1.0,
                bsqvac_weight=0.0,
                force_weight=0.0,
                enforce_edge=False,
            )

        controller_grad = jax.grad(controller_custom_objective)(base_params)
        controller_exact = sum(
            jnp.vdot(grad_leaf, direction_leaf)
            for grad_leaf, direction_leaf in zip(
                jax.tree_util.tree_leaves(controller_grad),
                jax.tree_util.tree_leaves(direction),
                strict=True,
            )
        )
        base_controller_trace = float(np.asarray(controller_custom_objective(base_params)))
        assert abs(base_controller_trace - base_complete) < 2.0e-3
        np.testing.assert_allclose(controller_exact, complete_fd, rtol=2.0e-3, atol=1.0e-8)
        np.testing.assert_allclose(controller_exact, exact, rtol=2.0e-3, atol=1.0e-8)

    if check_segmented_controller:
        def segmented_controller_custom_objective(params: CoilFieldParams):
            return direct_coil_accepted_trace_controller_custom_vjp_objective_jax(
                params,
                base_traces[0]["state_pre"],
                static=base_init.static,
                traces=base_traces,
                signgs=int(base_init.signgs),
                state_weight=1.0,
                bsqvac_weight=0.0,
                force_weight=0.0,
                enforce_edge=False,
                use_preconditioner_policy_segments=True,
            )

        segmented_controller_grad = jax.grad(segmented_controller_custom_objective)(base_params)
        segmented_controller_exact = sum(
            jnp.vdot(grad_leaf, direction_leaf)
            for grad_leaf, direction_leaf in zip(
                jax.tree_util.tree_leaves(segmented_controller_grad),
                jax.tree_util.tree_leaves(direction),
                strict=True,
            )
        )
        base_segmented_controller_trace = float(np.asarray(segmented_controller_custom_objective(base_params)))
        assert abs(base_segmented_controller_trace - base_complete) < 2.0e-3
        np.testing.assert_allclose(segmented_controller_exact, complete_fd, rtol=2.0e-3, atol=1.0e-8)
        if controller_exact is not None:
            np.testing.assert_allclose(segmented_controller_exact, controller_exact, rtol=2.0e-3, atol=1.0e-8)
        else:
            np.testing.assert_allclose(segmented_controller_exact, exact, rtol=2.0e-3, atol=1.0e-8)

    def aspect_objective_from_state(state) -> float:
        return float(np.asarray(equilibrium_aspect_ratio_from_state(state=state, static=base_init.static)))

    complete_aspect_fd = (
        aspect_objective_from_state(plus_result.state) - aspect_objective_from_state(minus_result.state)
    ) / (2.0 * eps)
    np.testing.assert_allclose(
        complete_report["objective_values"]["aspect"]["central_fd_directional"],
        complete_aspect_fd,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    if check_aspect_scalar:
        replay_scalar_fns = {
            "aspect": lambda replay, payload: equilibrium_aspect_ratio_from_state(
                state=replay["state"],
                static=payload["init"].static,
            ),
        }
        rtol_by_key = {"aspect": 5.0e-3}
        atol_by_key = {"aspect": 5.0e-8}
        if check_boundary_moment_scalar:
            replay_scalar_fns["lcfs_boundary_moment"] = lambda replay, payload: lcfs_boundary_moment(
                replay["state"],
                payload["init"].static,
            )
            rtol_by_key["lcfs_boundary_moment"] = 5.0e-3
            atol_by_key["lcfs_boundary_moment"] = 5.0e-8
        if check_accepted_bsqvac_rms_scalar:
            replay_scalar_fns["accepted_bsqvac_rms"] = lambda replay, _payload: accepted_bsqvac_rms_from_replay(replay)
            rtol_by_key["accepted_bsqvac_rms"] = 1.0e-2
            atol_by_key["accepted_bsqvac_rms"] = 1.0e-8
        if check_accepted_bnormal_rms_scalar:
            replay_scalar_fns["accepted_bnormal_rms"] = lambda replay, _payload: accepted_bnormal_rms_from_replay(replay)
            rtol_by_key["accepted_bnormal_rms"] = 1.0e-2
            atol_by_key["accepted_bnormal_rms"] = 1.0e-8
        scalars_report = direct_coil_same_branch_controller_scalars_custom_vjp_report(
            complete_report,
            base_params,
            direction,
            replay_scalar_fns=replay_scalar_fns,
            replay_kwargs={"use_stacked_step_controls": True},
            eps=eps,
            rtol=rtol_by_key,
            atol=atol_by_key,
            compute_frozen_fd=False,
        )
        assert scalars_report["passed"], scalars_report
        assert scalars_report["replay_option_flags"]["use_stacked_step_controls"] is True
        physical_scalar_gate = direct_coil_same_branch_physical_scalar_gate_report(
            complete_report,
            scalars_report,
            scalar_keys=tuple(replay_scalar_fns),
        )
        assert physical_scalar_gate["passed"], physical_scalar_gate
        assert physical_scalar_gate["contract"] == "same-branch complete-solve physical-scalar AD-vs-FD gate"
        assert physical_scalar_gate["same_branch"] is True
        assert physical_scalar_gate["differentiates_adaptive_controller"] is False
        assert physical_scalar_gate["scalar_keys"] == tuple(replay_scalar_fns)
        assert physical_scalar_gate["replay_gate"]["passed"] is True
        adaptive_full_loop_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
            complete_report,
            scalars_report,
            scalar_keys=tuple(replay_scalar_fns),
        )
        assert adaptive_full_loop_gate["passed"], adaptive_full_loop_gate
        assert adaptive_full_loop_gate["contract"] == "same-branch adaptive full-loop seam report"
        assert adaptive_full_loop_gate["differentiates_adaptive_controller"] is False
        assert adaptive_full_loop_gate["differentiates_run_free_boundary"] is False
        assert adaptive_full_loop_gate["same_branch"] is True
        assert adaptive_full_loop_gate["same_accepted_trace_branch"] is True
        assert adaptive_full_loop_gate["same_residual_branch"] is True
        assert adaptive_full_loop_gate["same_stacked_step_policy_branch"] is True
        assert adaptive_full_loop_gate["used_stacked_step_controls"] is True
        json.dumps(
            direct_coil_adaptive_full_loop_same_branch_gate_report(
                complete_report,
                scalars_report,
                scalar_keys=tuple(replay_scalar_fns),
                json_safe=True,
            ),
            allow_nan=False,
        )
        json.dumps(
            direct_coil_same_branch_physical_scalar_gate_report(
                complete_report,
                scalars_report,
                scalar_keys=tuple(replay_scalar_fns),
                json_safe=True,
            ),
            allow_nan=False,
        )
        aspect_report = scalars_report["scalar_reports"]["aspect"]
        assert aspect_report["passed"], aspect_report
        assert aspect_report["same_branch"] is True
        assert aspect_report["replay_gate"]["passed"] is True
        assert aspect_report["base_abs_delta"] < 2.0e-3
        np.testing.assert_allclose(
            aspect_report["complete_fd_directional"],
            complete_aspect_fd,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        if check_production_branch_local_scalar:
            production_branch_local = direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
                params=base_params,
                complete_payload=complete_report["base"],
                scalar_key="aspect",
                scalar_fn=lambda payload: {
                    "aspect": aspect_objective_from_state(payload["result"].state),
                },
                replay_scalar_fn=lambda replay, payload: equilibrium_aspect_ratio_from_state(
                    state=replay["state"],
                    static=payload["init"].static,
                ),
                replay_kwargs={"use_stacked_step_controls": True},
            )
            production_branch_exact = sum(
                jnp.vdot(grad_leaf, direction_leaf)
                for grad_leaf, direction_leaf in zip(
                    jax.tree_util.tree_leaves(production_branch_local["grad"]),
                    jax.tree_util.tree_leaves(direction),
                    strict=True,
                )
            )
            assert production_branch_local["uses_production_forward"] is True
            assert production_branch_local["differentiates_adaptive_controller"] is False
            assert production_branch_local["differentiates_run_free_boundary"] is False
            assert production_branch_local["differentiates_fixed_accepted_branch"] is True
            assert production_branch_local["trace_replay_diagnostics"]["differentiates_adaptive_controller"] is False
            assert production_branch_local["replay_option_flags"]["use_stacked_step_controls"] is True
            assert production_branch_local["base_abs_delta"] < 2.0e-3
            np.testing.assert_allclose(
                production_branch_exact,
                complete_aspect_fd,
                rtol=5.0e-3,
                atol=5.0e-8,
            )
        if check_boundary_moment_scalar:
            moment_report = scalars_report["scalar_reports"]["lcfs_boundary_moment"]
            assert moment_report["passed"], moment_report
            assert moment_report["same_branch"] is True
            assert moment_report["replay_gate"]["passed"] is True
            assert moment_report["base_abs_delta"] < 2.0e-3
        if check_accepted_bsqvac_rms_scalar:
            bsqvac_values = complete_report["objective_values"]["accepted_bsqvac_rms"]
            assert bsqvac_values["base"] > 0.0
            assert bsqvac_values["plus"] > bsqvac_values["minus"]
            assert bsqvac_values["central_fd_directional"] > 0.0
            bsqvac_report = scalars_report["scalar_reports"]["accepted_bsqvac_rms"]
            assert bsqvac_report["passed"], bsqvac_report
            assert bsqvac_report["same_branch"] is True
            assert bsqvac_report["replay_gate"]["passed"] is True
            assert bsqvac_report["base_abs_delta"] < 2.0e-3
        if check_accepted_bnormal_rms_scalar:
            bnormal_values = complete_report["objective_values"]["accepted_bnormal_rms"]
            assert bnormal_values["base"] > 0.0
            assert bnormal_values["plus"] > bnormal_values["minus"]
            assert bnormal_values["central_fd_directional"] > 0.0
            bnormal_report = scalars_report["scalar_reports"]["accepted_bnormal_rms"]
            assert bnormal_report["passed"], bnormal_report
            assert bnormal_report["same_branch"] is True
            assert bnormal_report["replay_gate"]["passed"] is True
            assert bnormal_report["base_abs_delta"] < 2.0e-3
    elif check_boundary_moment_scalar:
        moment_report = direct_coil_same_branch_controller_scalar_custom_vjp_report(
            complete_report,
            base_params,
            direction,
            scalar_key="lcfs_boundary_moment",
            replay_scalar_fn=lambda replay, payload: lcfs_boundary_moment(
                replay["state"],
                payload["init"].static,
            ),
            eps=eps,
            rtol=5.0e-3,
            atol=5.0e-8,
            compute_frozen_fd=False,
        )
        assert moment_report["passed"], moment_report
        assert moment_report["same_branch"] is True
        assert moment_report["replay_gate"]["passed"] is True
        assert moment_report["base_abs_delta"] < 2.0e-3
    elif check_accepted_bsqvac_rms_scalar:
        bsqvac_values = complete_report["objective_values"]["accepted_bsqvac_rms"]
        assert bsqvac_values["base"] > 0.0
        assert bsqvac_values["plus"] > bsqvac_values["minus"]
        assert bsqvac_values["central_fd_directional"] > 0.0
        bsqvac_report = direct_coil_same_branch_controller_scalar_custom_vjp_report(
            complete_report,
            base_params,
            direction,
            scalar_key="accepted_bsqvac_rms",
            replay_scalar_fn=lambda replay, _payload: accepted_bsqvac_rms_from_replay(replay),
            eps=eps,
            rtol=1.0e-2,
            atol=1.0e-8,
            compute_frozen_fd=False,
        )
        assert bsqvac_report["passed"], bsqvac_report
        assert bsqvac_report["same_branch"] is True
        assert bsqvac_report["replay_gate"]["passed"] is True
        assert bsqvac_report["base_abs_delta"] < 2.0e-3
    elif check_accepted_bnormal_rms_scalar:
        bnormal_values = complete_report["objective_values"]["accepted_bnormal_rms"]
        assert bnormal_values["base"] > 0.0
        assert bnormal_values["plus"] > bnormal_values["minus"]
        assert bnormal_values["central_fd_directional"] > 0.0
        bnormal_report = direct_coil_same_branch_controller_scalar_custom_vjp_report(
            complete_report,
            base_params,
            direction,
            scalar_key="accepted_bnormal_rms",
            replay_scalar_fn=lambda replay, _payload: accepted_bnormal_rms_from_replay(replay),
            eps=eps,
            rtol=1.0e-2,
            atol=1.0e-8,
            compute_frozen_fd=False,
        )
        assert bnormal_report["passed"], bnormal_report
        assert bnormal_report["same_branch"] is True
        assert bnormal_report["replay_gate"]["passed"] is True
        assert bnormal_report["base_abs_delta"] < 2.0e-3


@pytest.mark.py311_coverage_only
def test_direct_coil_current_only_same_branch_custom_vjp_matches_complete_solve_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current-only custom VJP matches complete-solve FD on one accepted branch."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    _set_same_branch_custom_vjp_env(monkeypatch)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_current_only_same_branch_custom_vjp",
        lasym=False,
        niter=1,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=24)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    direction = base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(base_dofs),
        base_currents=base_currents * 0.02,
    )

    def params_for(scale: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_curve_dofs=base_dofs,
            base_currents=base_currents * (1.0 + 0.02 * float(scale)),
        )

    _assert_direct_coil_same_branch_custom_vjp_matches_complete_fd(
        input_path=input_path,
        base_params=base_params,
        direction=direction,
        params_for=params_for,
        check_controller=False,
        check_segmented_controller=False,
        check_aspect_scalar=True,
        check_boundary_moment_scalar=False,
        check_accepted_bnormal_rms_scalar=True,
        check_accepted_bsqvac_rms_scalar=True,
        check_production_branch_local_scalar=True,
    )


@pytest.mark.py311_coverage_only
def test_direct_coil_fourier_only_same_branch_custom_vjp_matches_complete_solve_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fourier-coefficient custom VJP matches complete-solve FD on one branch."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    _set_same_branch_custom_vjp_env(monkeypatch)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_fourier_only_same_branch_custom_vjp",
        lasym=False,
        niter=2,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=24)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    dof_index = (0, 0, 2)
    dof_step = 5.0e-3
    direction = base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(base_dofs).at[dof_index].set(dof_step),
        base_currents=jnp.zeros_like(base_currents),
    )

    def params_for(scale: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_curve_dofs=base_dofs.at[dof_index].add(dof_step * float(scale)),
            base_currents=base_currents,
        )

    _assert_direct_coil_same_branch_custom_vjp_matches_complete_fd(
        input_path=input_path,
        base_params=base_params,
        direction=direction,
        params_for=params_for,
        check_controller=False,
        check_segmented_controller=False,
        check_aspect_scalar=True,
        check_boundary_moment_scalar=True,
    )


@pytest.mark.py311_coverage_only
def test_direct_coil_lasym_fixed_trace_custom_vjp_matches_complete_solve_fd_on_same_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LASYM fixed-trace custom VJP matches complete-solve FD on one branch.

    Current-only and Fourier-only stellsym controls are covered by the
    preceding same-branch gates.  This retained mixed-direction case keeps the
    asymmetric branch represented without duplicating the stellsym solve triplet.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    _set_same_branch_custom_vjp_env(monkeypatch)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_same_branch_custom_vjp",
        lasym=True,
        niter=2,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=24)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    direction = base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(base_dofs).at[0, 0, 2].set(5.0e-3),
        base_currents=base_currents * 0.02,
    )

    def params_for(scale: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_curve_dofs=base_dofs.at[0, 0, 2].add(5.0e-3 * float(scale)),
            base_currents=base_currents * (1.0 + 0.02 * float(scale)),
        )

    _assert_direct_coil_same_branch_custom_vjp_matches_complete_fd(
        input_path=input_path,
        base_params=base_params,
        direction=direction,
        params_for=params_for,
        check_controller=False,
        check_segmented_controller=False,
        check_aspect_scalar=True,
        check_boundary_moment_scalar=True,
    )


@pytest.mark.py311_coverage_only
def test_jax_nestor_operator_accepted_solve_ad_matches_central_fd_for_current_and_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted-boundary direct-coil replay AD matches central FD for coil controls.

    This validates the first promoted accepted-output rung: run the nonlinear
    VMEC free-boundary solve once, freeze its accepted plasma boundary, then
    replay the final direct-coil NESTOR normal-field metric through a pure JAX
    path.  It does not claim differentiation through the VMEC iteration loop.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp
    from vmec_jax.driver import run_free_boundary
    from vmec_jax.free_boundary import _sample_external_boundary_arrays
    from vmec_jax.free_boundary_adjoint import direct_coil_boundary_bnormal_rms_jax

    enable_x64(True)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_accepted_ad_fd",
        lasym=False,
        niter=2,
        mpol=3,
        ntheta=6,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=64)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    monkeypatch.setenv("VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR", "1")
    monkeypatch.setenv("VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR", "0")

    def params_for(current_scale, geometry_scale):
        return base_params.with_arrays(
            base_curve_dofs=base_dofs.at[0, 0, 2].add(1.0e-2 * geometry_scale),
            base_currents=base_currents * (1.0 + 0.02 * current_scale),
        )

    run = run_free_boundary(
        input_path,
        max_iter=2,
        multigrid=False,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
        free_boundary_activate_fsq=1.0e99,
    )
    freeb = run.result.diagnostics["free_boundary"]
    assert freeb["vacuum_stub"] is False
    assert freeb["final_nestor_recompute_failed"] is False
    nestor = freeb["last_nestor_diagnostics"]
    assert nestor["provider_kind"] == "direct_coils"
    assert nestor["jax_nestor_operator_reason"] == "applied"
    assert "bnormal_rms" in nestor

    plascur = float(freeb.get("plascur", 0.0))
    sample_coils_only = _sample_external_boundary_arrays(
        state=run.state,
        static=run.static,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
    )
    sample = _sample_external_boundary_arrays(
        state=run.state,
        static=run.static,
        plascur=plascur,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
    )
    R = jnp.asarray(sample.R)
    Z = jnp.asarray(sample.Z)
    phi = jnp.asarray(sample.phi)
    Ru = jnp.asarray(sample.Ru)
    Zu = jnp.asarray(sample.Zu)
    Rv = jnp.asarray(sample.Rv)
    Zv = jnp.asarray(sample.Zv)
    br_axis = jnp.asarray(sample.br - sample_coils_only.br)
    bp_axis = jnp.asarray(sample.bp - sample_coils_only.bp)
    bz_axis = jnp.asarray(sample.bz - sample_coils_only.bz)

    def accepted_bnormal_metric(current_scale, geometry_scale):
        return direct_coil_boundary_bnormal_rms_jax(
            params_for(current_scale, geometry_scale),
            R=R,
            Z=Z,
            phi=phi,
            Ru=Ru,
            Zu=Zu,
            Rv=Rv,
            Zv=Zv,
            br_add=br_axis,
            bp_add=bp_axis,
            bz_add=bz_axis,
        )

    # The final diagnostic may be produced by a separate host recompute context.
    # This gate promotes the frozen accepted-boundary AD-vs-FD check below, so
    # only require the base replay and diagnostic to be finite physical scalars.
    base_metric = accepted_bnormal_metric(0.0, 0.0)
    assert np.isfinite(np.asarray(base_metric, dtype=float))
    assert np.isfinite(float(nestor["bnormal_rms"]))
    assert float(np.asarray(base_metric)) > 0.0
    assert float(nestor["bnormal_rms"]) > 0.0

    eps = 0.25
    fd_current = (accepted_bnormal_metric(eps, 0.0) - accepted_bnormal_metric(-eps, 0.0)) / (2.0 * eps)
    assert np.isfinite(np.asarray(fd_current, dtype=float))
    assert abs(float(np.asarray(fd_current))) > 1.0e-16

    exact_current = jax.grad(lambda scale: accepted_bnormal_metric(scale, 0.0))(0.0)
    assert np.isfinite(np.asarray(exact_current, dtype=float))
    assert abs(float(np.asarray(exact_current))) > 1.0e-16
    np.testing.assert_allclose(exact_current, fd_current, rtol=1.0e-3, atol=1.0e-12)

    fd_geometry = (accepted_bnormal_metric(0.0, eps) - accepted_bnormal_metric(0.0, -eps)) / (2.0 * eps)
    assert np.isfinite(np.asarray(fd_geometry, dtype=float))
    assert abs(float(np.asarray(fd_geometry))) > 1.0e-16

    exact_geometry = jax.grad(lambda scale: accepted_bnormal_metric(0.0, scale))(0.0)
    assert np.isfinite(np.asarray(exact_geometry, dtype=float))
    assert abs(float(np.asarray(exact_geometry))) > 1.0e-16
    np.testing.assert_allclose(exact_geometry, fd_geometry, rtol=1.0e-3, atol=1.0e-12)

    mixed_derivs = []
    for current_direction, geometry_direction in ((1.0, 0.5), (1.0, -0.5)):
        def mixed_metric(scale):
            return accepted_bnormal_metric(
                current_direction * scale,
                geometry_direction * scale,
            )

        fd_mixed = (mixed_metric(eps) - mixed_metric(-eps)) / (2.0 * eps)
        exact_mixed = jax.grad(mixed_metric)(0.0)
        mixed_derivs.extend([exact_mixed, fd_mixed])
        np.testing.assert_allclose(exact_mixed, fd_mixed, rtol=1.0e-3, atol=1.0e-12)

    mixed_derivs = np.asarray(mixed_derivs, dtype=float)
    assert np.all(np.isfinite(mixed_derivs))
    assert np.max(np.abs(mixed_derivs)) > 1.0e-16


@pytest.mark.py311_coverage_only
def test_direct_coil_accepted_update_replay_ad_matches_fd_for_coil_pytree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay one accepted free-boundary update with coil-derived vacuum forcing.

    This is a phase-2 production-trace rung: the nonlinear solve supplies the
    accepted step constants, while the differentiable path recomputes the
    direct-coil NESTOR mode solve and feeds its ``bsqvac`` into the strict VMEC
    update. It validates that accepted-update replay is differentiable with
    respect to a mixed current/geometry ``CoilFieldParams`` direction, without
    claiming a full custom VJP for the host-controlled nonlinear loop.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp
    from vmec_jax.discrete_adjoint import (
        preconditioned_force_channels_from_rz_output,
        strict_update_accepted_step,
        strict_update_one_step_from_trace,
        strict_update_one_step_from_state,
    )
    from vmec_jax.driver import run_free_boundary
    from vmec_jax.free_boundary import _sample_external_boundary_arrays
    from vmec_jax.free_boundary_adjoint import (
        direct_coil_accepted_trace_controller_replay_objective_jax,
        direct_coil_accepted_trace_fingerprint,
        direct_coil_accepted_trace_fingerprint_delta,
        direct_coil_accepted_trace_step_controls_jax,
        direct_coil_accepted_trace_step_policy_segments,
        direct_coil_accepted_trace_replay_objective_jax,
        direct_coil_boundary_bsqvac_jax,
        direct_coil_boundary_bsqvac_from_trace_jax,
        direct_coil_boundary_replay_context,
        free_boundary_boundary_geometry_jax,
        pytree_directional_derivative_check_jax,
        vacuum_boundary_fields_from_mode_coeffs_jax,
    )
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state, unpack_state
    from vmec_jax.vmec_tomnsp import TomnspsRZL

    enable_x64(True)
    for key, value in {
        "VMEC_JAX_FREEB_NESTOR_MODE": "dense",
        "VMEC_JAX_FREEB_DENSE_SOLVE_MODE": "mode",
        "VMEC_JAX_FREEB_USE_GREENF_SOURCE": "1",
        "VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX": "1",
        "VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC": "1",
        "VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR": "1",
        "VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR": "0",
    }.items():
        monkeypatch.setenv(key, value)

    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_accepted_update_ad_fd",
        lasym=False,
        niter=3,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=16)
    init = run_free_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
    )
    result = solve_fixed_boundary_residual_iter(
        init.state,
        init.static,
        indata=init.indata,
        signgs=init.signgs,
        max_iter=3,
        ftol=1.0e-8,
        vmec2000_control=True,
        auto_flip_force=False,
        use_direct_fallback=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces=False,
        use_scan=False,
        host_update_assembly=False,
        adjoint_trace=True,
        adjoint_trace_mode="full",
        external_field_provider_kind="direct_coils",
        external_field_provider_params=base_params,
        free_boundary_activate_fsq=1.0e99,
    )
    traces = result.diagnostics.get("adjoint_step_trace", [])
    active_traces = [trace for trace in traces if trace.get("freeb_bsqvac_half") is not None]
    assert len(active_traces) >= 2
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    direction = base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(base_dofs).at[0, 0, 2].set(1.0e-2),
        base_currents=base_currents * 0.02,
    )

    selected = None
    for include_analytic in (True, False):
        for candidate_idx, candidate_trace in enumerate(active_traces):
            sample = _sample_external_boundary_arrays(
                state=candidate_trace["state_pre"],
                static=init.static,
                plascur=float(
                    candidate_trace.get("freeb_plascur_for_bsqvac", candidate_trace.get("freeb_plascur", 0.0))
                ),
                external_field_provider_kind="direct_coils",
                external_field_provider_params=base_params,
            )
            context = direct_coil_boundary_replay_context(init.static, {"R": sample.R})
            candidate_nestor_trace = candidate_trace.get("freeb_nestor_trace")
            if not isinstance(candidate_nestor_trace, dict):
                continue

            def _candidate_replay_from_coils(params: CoilFieldParams):
                return direct_coil_boundary_bsqvac_jax(
                    params,
                    R=jnp.asarray(sample.R),
                    Z=jnp.asarray(sample.Z),
                    phi=jnp.asarray(sample.phi),
                    Ru=jnp.asarray(sample.Ru),
                    Zu=jnp.asarray(sample.Zu),
                    Rv=jnp.asarray(sample.Rv),
                    Zv=jnp.asarray(sample.Zv),
                    ruu=jnp.asarray(sample.ruu),
                    ruv=jnp.asarray(sample.ruv),
                    rvv=jnp.asarray(sample.rvv),
                    zuu=jnp.asarray(sample.zuu),
                    zuv=jnp.asarray(sample.zuv),
                    zvv=jnp.asarray(sample.zvv),
                    basis=context["basis"],
                    tables=context["tables"],
                    signgs=int(init.signgs),
                    nvper=context["nvper"],
                    br_add=jnp.asarray(candidate_nestor_trace["br_axis"]),
                    bp_add=jnp.asarray(candidate_nestor_trace["bp_axis"]),
                    bz_add=jnp.asarray(candidate_nestor_trace["bz_axis"]),
                    wint=jnp.asarray(context["wint"]),
                    include_analytic=bool(include_analytic),
                )

            candidate_replay0 = _candidate_replay_from_coils(base_params)
            candidate_bsqvac0 = candidate_replay0["bsqvac"]
            candidate_bsqvac0_np = np.asarray(candidate_bsqvac0, dtype=float)
            if np.all(np.isfinite(candidate_bsqvac0_np)) and float(np.linalg.norm(candidate_bsqvac0_np)) > 0.0:
                selected = (
                    candidate_idx,
                    candidate_trace,
                    candidate_nestor_trace,
                    context["basis"],
                    _candidate_replay_from_coils,
                    candidate_replay0,
                    candidate_bsqvac0,
                    bool(include_analytic),
                )
                break
        if selected is not None:
            break

    assert selected is not None, "No finite accepted direct-coil replay trace found"
    selected_idx, trace, nestor_trace, basis, replay_from_coils, replay0, bsqvac0, analytic_replay = selected
    assert selected_idx + 1 < len(active_traces), "Selected trace must have a following trace for two-step replay"
    trace1 = active_traces[selected_idx + 1]

    def bsqvac_from_coils(params: CoilFieldParams):
        return replay_from_coils(params)["bsqvac"]

    assert bsqvac0.shape == np.asarray(trace["freeb_bsqvac_half"]).shape
    assert np.all(np.isfinite(np.asarray(bsqvac0, dtype=float)))
    assert float(np.linalg.norm(np.asarray(bsqvac0, dtype=float))) > 0.0
    np.testing.assert_allclose(
        np.asarray(nestor_trace["bsqvac"]),
        np.asarray(trace["freeb_bsqvac_half"]),
        rtol=0.0,
        atol=0.0,
    )
    trace_channels = vacuum_boundary_fields_from_mode_coeffs_jax(
        nestor_trace["potvac"],
        basis=basis,
        bu_ext=nestor_trace["bu_ext"],
        bv_ext=nestor_trace["bv_ext"],
        g_uu=nestor_trace["g_uu"],
        g_uv=nestor_trace["g_uv"],
        g_vv=nestor_trace["g_vv"],
    )
    np.testing.assert_allclose(
        np.asarray(trace_channels["bsqvac"]),
        np.asarray(nestor_trace["bsqvac"]),
        rtol=1.0e-13,
        atol=1.0e-12,
    )
    if analytic_replay:
        np.testing.assert_allclose(
            np.asarray(replay0["mode_solution"]["mode_coeffs"]),
            np.asarray(nestor_trace["potvac"]),
            rtol=1.0e-13,
            atol=1.0e-12,
        )
        bsqvac_delta = np.asarray(bsqvac0, dtype=float) - np.asarray(trace["freeb_bsqvac_half"], dtype=float)
        bsqvac_rel = np.linalg.norm(bsqvac_delta) / max(
            1.0,
            np.linalg.norm(np.asarray(trace["freeb_bsqvac_half"], dtype=float)),
        )
        assert bsqvac_rel < 1.0e-13

    # The accepted trace must be exactly replayable once the force channels have
    # been computed. This protects accepted-output correctness separately from
    # the harder coil -> NESTOR -> force reconstruction path below.
    traced_rz_force = TomnspsRZL(
        frcc=trace["frzl_rz_frcc"],
        frss=trace["frzl_rz_frss"],
        fzsc=trace["frzl_rz_fzsc"],
        fzcs=trace["frzl_rz_fzcs"],
        flsc=trace["frzl_rz_flsc"],
        flcs=trace["frzl_rz_flcs"],
        frsc=trace["frzl_rz_frsc"],
        frcs=trace["frzl_rz_frcs"],
        fzcc=trace["frzl_rz_fzcc"],
        fzss=trace["frzl_rz_fzss"],
        flcc=trace["frzl_rz_flcc"],
        flss=trace["frzl_rz_flss"],
    )
    traced_force = preconditioned_force_channels_from_rz_output(
        frzl_rz=traced_rz_force,
        lam_prec=trace["lam_prec"],
        w_mode_mn=trace["w_mode_mn"],
        lambda_update_scale=trace["lambda_update_scale"],
    )
    for key in ("frcc_u", "frss_u", "fzsc_u", "fzcs_u", "flsc_u", "flcs_u"):
        np.testing.assert_allclose(np.asarray(traced_force[key]), np.asarray(trace[key]), rtol=0.0, atol=0.0)
    exact_step = strict_update_accepted_step(
        trace["state_pre"],
        init.static,
        dt_eff=trace["dt_eff"],
        b1=trace["b1"],
        fac=trace["fac"],
        force_scale=trace["force_scale"],
        flip_sign=trace["flip_sign"],
        vRcc_before=trace["vRcc_before"],
        vRss_before=trace["vRss_before"],
        vZsc_before=trace["vZsc_before"],
        vZcs_before=trace["vZcs_before"],
        vLsc_before=trace["vLsc_before"],
        vLcs_before=trace["vLcs_before"],
        frcc_u=trace["frcc_u"],
        frss_u=trace["frss_u"],
        fzsc_u=trace["fzsc_u"],
        fzcs_u=trace["fzcs_u"],
        flsc_u=trace["flsc_u"],
        flcs_u=trace["flcs_u"],
        max_update_rms=trace["max_update_rms_pre"],
        limit_update_rms=trace["limit_update_rms"],
        divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
        enforce_edge=False,
    )
    np.testing.assert_allclose(
        np.asarray(pack_state(exact_step["state_post"])),
        np.asarray(pack_state(trace["state_post"])),
        rtol=0.0,
        atol=0.0,
    )

    def objective(params: CoilFieldParams):
        out = strict_update_one_step_from_state(
            trace["state_pre"],
            init.static,
            wout_like=trace["wout_like"],
            trig=trace["trig"],
            apply_lforbal=trace["apply_lforbal"],
            include_edge_residual=trace["include_edge_residual"],
            apply_m1_constraints=trace["apply_m1_constraints"],
            zero_m1=trace["zero_m1"],
            mats=trace["precond_mats"],
            jmax=trace["precond_jmax"],
            lam_prec=trace["lam_prec"],
            w_mode_mn=trace["w_mode_mn"],
            lambda_update_scale=trace["lambda_update_scale"],
            dt_eff=trace["dt_eff"],
            b1=trace["b1"],
            fac=trace["fac"],
            force_scale=trace["force_scale"],
            flip_sign=trace["flip_sign"],
            vRcc_before=trace["vRcc_before"],
            vRss_before=trace["vRss_before"],
            vZsc_before=trace["vZsc_before"],
            vZcs_before=trace["vZcs_before"],
            vLsc_before=trace["vLsc_before"],
            vLcs_before=trace["vLcs_before"],
            max_update_rms=trace["max_update_rms_pre"],
            limit_update_rms=trace["limit_update_rms"],
            divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
            preconditioner_use_precomputed_tridi=trace["preconditioner_use_precomputed_tridi"],
            preconditioner_use_lax_tridi=trace["preconditioner_use_lax_tridi"],
            freeb_bsqvac_half=bsqvac_from_coils(params),
            freeb_pres_scale=trace["freeb_pres_scale"],
            constraint_rcon0=trace.get("constraint_rcon0"),
            constraint_zcon0=trace.get("constraint_zcon0"),
            constraint_tcon0=trace.get("constraint_tcon0"),
            constraint_precond_diag=trace.get("constraint_precond_diag"),
            constraint_tcon=trace.get("constraint_tcon"),
            constraint_precond_active=trace.get("constraint_precond_active"),
            constraint_tcon_active=trace.get("constraint_tcon_active"),
            enforce_edge=False,
        )
        state_post = jnp.asarray(pack_state(out["step"]["state_post"]))
        force_norm = jnp.asarray(out["force"]["frcc_u"])
        return 0.5 * jnp.vdot(state_post, state_post) + 1.0e-3 * jnp.vdot(force_norm, force_norm)

    check = pytree_directional_derivative_check_jax(objective, base_params, direction, eps=1.0e-3)
    exact = float(np.asarray(check["exact_directional"]))
    fd = float(np.asarray(check["fd_directional"]))
    assert np.isfinite(exact)
    assert np.isfinite(fd)
    assert abs(exact) > 1.0e-16
    assert abs(fd) > 1.0e-16
    np.testing.assert_allclose(exact, fd, rtol=3.0e-3, atol=1.0e-10)

    flat0 = jnp.asarray(pack_state(trace["state_pre"]))

    def state_replay_objective(flat_state):
        state = unpack_state(flat_state, trace["state_pre"].layout)
        geometry = free_boundary_boundary_geometry_jax(state, init.static)
        state_context = direct_coil_boundary_replay_context(init.static, geometry)
        state_replay = direct_coil_boundary_bsqvac_from_trace_jax(
            base_params,
            geometry,
            trace,
            basis=state_context["basis"],
            tables=state_context["tables"],
            signgs=int(init.signgs),
            nvper=state_context["nvper"],
            wint=jnp.asarray(state_context["wint"]),
            include_analytic=True,
        )
        state_bsqvac = jnp.asarray(state_replay["bsqvac"])
        return 0.5 * jnp.vdot(state_bsqvac, state_bsqvac)

    eps_state = 1.0e-4
    candidate_indices = np.unique(
        np.asarray(
            [
                0,
                flat0.size // 7,
                flat0.size // 5,
                flat0.size // 3,
                flat0.size // 2,
                (2 * flat0.size) // 3,
                flat0.size - 1,
            ],
            dtype=int,
        )
    )
    directional_pairs: list[tuple[object, object]] = []
    for idx in candidate_indices:
        state_direction = jnp.zeros_like(flat0).at[int(idx)].set(1.0)
        _, exact_state = jax.jvp(state_replay_objective, (flat0,), (state_direction,))
        fd_state = (
            state_replay_objective(flat0 + eps_state * state_direction)
            - state_replay_objective(flat0 - eps_state * state_direction)
        ) / (2.0 * eps_state)
        if np.isfinite(float(exact_state)) and np.isfinite(float(fd_state)) and abs(float(fd_state)) > 1.0e-16:
            directional_pairs.append((exact_state, fd_state))
            break
    assert directional_pairs, "No finite nonzero state direction found for accepted-boundary replay"
    exact_state, fd_state = directional_pairs[0]
    np.testing.assert_allclose(exact_state, fd_state, rtol=2.0e-4, atol=1.0e-10)

    trace0 = trace
    fingerprint = direct_coil_accepted_trace_fingerprint([trace0, trace1])
    assert fingerprint["n_steps"] == 2
    assert fingerprint["n_freeb_steps"] == 2
    assert np.all(fingerprint["freeb_sizes"] > 0)
    same_branch = direct_coil_accepted_trace_fingerprint_delta([trace0, trace1], [trace0, trace1])
    assert same_branch["compatible"]
    changed_trace0 = dict(trace0)
    changed_trace0["dt_eff"] = _trace_scalar_value(trace0["dt_eff"]) * 1.05
    changed_branch = direct_coil_accepted_trace_fingerprint_delta([trace0, trace1], [changed_trace0, trace1])
    assert not changed_branch["compatible"]
    assert "scalars.dt_eff" in changed_branch["changed_fields"]
    truncated_branch = direct_coil_accepted_trace_fingerprint_delta([trace0, trace1], [trace0])
    assert not truncated_branch["compatible"]
    assert "n_steps" in truncated_branch["changed_fields"]

    first_step = strict_update_accepted_step(
        trace0["state_pre"],
        init.static,
        dt_eff=trace0["dt_eff"],
        b1=trace0["b1"],
        fac=trace0["fac"],
        force_scale=trace0["force_scale"],
        flip_sign=trace0["flip_sign"],
        vRcc_before=trace0["vRcc_before"],
        vRss_before=trace0["vRss_before"],
        vZsc_before=trace0["vZsc_before"],
        vZcs_before=trace0["vZcs_before"],
        vLsc_before=trace0["vLsc_before"],
        vLcs_before=trace0["vLcs_before"],
        frcc_u=trace0["frcc_u"],
        frss_u=trace0["frss_u"],
        fzsc_u=trace0["fzsc_u"],
        fzcs_u=trace0["fzcs_u"],
        flsc_u=trace0["flsc_u"],
        flcs_u=trace0["flcs_u"],
        max_update_rms=trace0["max_update_rms_pre"],
        limit_update_rms=trace0["limit_update_rms"],
        divide_by_scalxc_for_update=trace0["divide_by_scalxc_for_update"],
        enforce_edge=False,
    )
    replayed_state1 = first_step["state_post"]
    np.testing.assert_allclose(
        np.asarray(pack_state(replayed_state1)),
        np.asarray(pack_state(trace0["state_post"])),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(pack_state(replayed_state1)),
        np.asarray(pack_state(trace1["state_pre"])),
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    geometry = free_boundary_boundary_geometry_jax(replayed_state1, init.static)
    context = direct_coil_boundary_replay_context(init.static, geometry)
    nestor_trace1 = trace1.get("freeb_nestor_trace")
    assert isinstance(nestor_trace1, dict)
    replay1 = direct_coil_boundary_bsqvac_from_trace_jax(
        base_params,
        geometry,
        trace1,
        basis=context["basis"],
        tables=context["tables"],
        signgs=int(init.signgs),
        nvper=context["nvper"],
        wint=jnp.asarray(context["wint"]),
        include_analytic=True,
    )
    np.testing.assert_allclose(
        np.asarray(replay1["bsqvac"]),
        np.asarray(trace1["freeb_bsqvac_half"]),
        rtol=2.0e-12,
        atol=1.0e-10,
    )
    second_step = strict_update_one_step_from_trace(
        replayed_state1,
        init.static,
        trace1,
        freeb_bsqvac_half=replay1["bsqvac"],
        enforce_edge=False,
    )
    np.testing.assert_allclose(
        np.asarray(pack_state(second_step["step"]["state_post"])),
        np.asarray(pack_state(trace1["state_post"])),
        rtol=1.0e-10,
        atol=1.0e-11,
    )

    replay = direct_coil_accepted_trace_replay_objective_jax(
        base_params,
        trace0["state_pre"],
        static=init.static,
        traces=[trace0, trace1],
        signgs=int(init.signgs),
        state_weight=1.0,
        bsqvac_weight=1.0e-12,
        force_weight=0.0,
        enforce_edge=False,
    )
    assert {"state", "bsqvac", "force"}.issubset(replay["objective_components"])
    assert np.isfinite(float(replay["objective"]))
    controller_replay = direct_coil_accepted_trace_controller_replay_objective_jax(
        base_params,
        trace0["state_pre"],
        static=init.static,
        traces=[trace0, trace1],
        signgs=int(init.signgs),
        state_weight=1.0,
        bsqvac_weight=1.0e-12,
        force_weight=0.0,
        enforce_edge=False,
    )
    np.testing.assert_array_equal(np.asarray(controller_replay["history"]["accepted"]), np.asarray([True, True]))
    np.testing.assert_array_equal(np.asarray(controller_replay["history"]["rejected"]), np.asarray([False, False]))
    np.testing.assert_array_equal(
        np.asarray(controller_replay["controls"]["step_index"]),
        np.asarray([0, 1]),
    )
    np.testing.assert_array_equal(
        np.asarray(controller_replay["controls"]["reset_to_trace_pre"]),
        np.asarray([False, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(controller_replay["controls"]["has_active_freeb_replay"]),
        np.asarray([True, True]),
    )
    np.testing.assert_allclose(
        np.asarray(controller_replay["controls"]["step_scalars"]["dt_eff"]),
        np.asarray([_trace_scalar_value(trace0["dt_eff"]), _trace_scalar_value(trace1["dt_eff"])]),
    )
    assert "flip_sign" in controller_replay["controls"]["step_scalars"]
    assert "limit_update_rms" in controller_replay["controls"]["step_scalars"]
    assert "divide_by_scalxc_for_update" in controller_replay["controls"]["step_scalars"]
    assert "preconditioner_use_lax_tridi" not in controller_replay["controls"]["step_scalars"]
    assert "preconditioner_use_precomputed_tridi" not in controller_replay["controls"]["step_scalars"]
    np.testing.assert_allclose(
        np.asarray(controller_replay["scalar_controls"]["fac"]),
        np.asarray([_trace_scalar_value(trace0["fac"]), _trace_scalar_value(trace1["fac"])]),
    )
    assert "limit_update_rms" in controller_replay["scalar_controls"]
    assert "preconditioner_use_lax_tridi" in controller_replay["scalar_controls"]
    assert controller_replay["preconditioner_controls_segment_stacked"] == ()
    np.testing.assert_allclose(
        np.asarray(controller_replay["controls"]["step_arrays"]["vRcc_before"][0]),
        np.asarray(trace0["vRcc_before"]),
    )
    assert np.asarray(controller_replay["array_controls"]["vLcs_before"]).shape[0] == 2
    np.testing.assert_allclose(
        np.asarray(controller_replay["controls"]["step_preconditioner"]["lam_prec"][0]),
        np.asarray(trace0["lam_prec"]),
    )
    assert controller_replay["preconditioner_controls_stacked"]
    assert np.asarray(controller_replay["preconditioner_controls"]["w_mode_mn"]).shape[0] == 2
    assert controller_replay["preconditioner_policy_n_segments"] == 1
    assert controller_replay["step_policy_n_segments"] >= 1
    assert "state_pre" in direct_coil_accepted_trace_step_controls_jax([trace0, trace1])
    step_segments = direct_coil_accepted_trace_step_policy_segments([trace0, trace1])
    assert step_segments[0]["start"] == 0
    assert step_segments[-1]["stop"] == 2
    assert sum(int(segment["n_steps"]) for segment in step_segments) == 2
    assert [
        (segment["start"], segment["stop"], segment["n_steps"])
        for segment in controller_replay["preconditioner_policy_segments"]
    ] == [(0, 2, 2)]
    assert [
        (
            segment["start"],
            segment["stop"],
            segment["accepted_steps"],
            segment["rejected_steps"],
            segment["free_boundary_replay_steps"],
            segment["state_resets"],
        )
        for segment in controller_replay["preconditioner_policy_segment_summary"]
    ] == [(0, 2, 2, 0, 2, 0)]
    np.testing.assert_allclose(
        np.asarray(controller_replay["objective"]),
        np.asarray(replay["objective"]),
        rtol=2.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(pack_state(controller_replay["state"])),
        np.asarray(pack_state(replay["state"])),
        rtol=5.0e-12,
        atol=5.0e-12,
    )
    segmented_controller_replay = direct_coil_accepted_trace_controller_replay_objective_jax(
        base_params,
        trace0["state_pre"],
        static=init.static,
        traces=[trace0, trace1],
        signgs=int(init.signgs),
        state_weight=1.0,
        bsqvac_weight=1.0e-12,
        force_weight=0.0,
        enforce_edge=False,
        use_preconditioner_policy_segments=True,
    )
    assert segmented_controller_replay["used_preconditioner_policy_segments"]
    assert segmented_controller_replay["preconditioner_controls_segment_stacked"] == (True,)
    np.testing.assert_allclose(
        np.asarray(segmented_controller_replay["objective"]),
        np.asarray(controller_replay["objective"]),
        rtol=2.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(pack_state(segmented_controller_replay["state"])),
        np.asarray(pack_state(controller_replay["state"])),
        rtol=5.0e-12,
        atol=5.0e-12,
    )
    stacked_controller_replay = direct_coil_accepted_trace_controller_replay_objective_jax(
        base_params,
        trace0["state_pre"],
        static=init.static,
        traces=[trace0, trace1],
        signgs=int(init.signgs),
        state_weight=1.0,
        bsqvac_weight=1.0e-12,
        force_weight=0.0,
        enforce_edge=False,
        use_stacked_step_controls=True,
    )
    assert stacked_controller_replay["used_stacked_step_controls"]
    assert stacked_controller_replay["step_policy_n_segments"] == len(step_segments)
    assert stacked_controller_replay["preconditioner_controls_segment_stacked"] == (True,) * len(step_segments)
    np.testing.assert_allclose(
        np.asarray(stacked_controller_replay["objective"]),
        np.asarray(controller_replay["objective"]),
        rtol=2.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(pack_state(stacked_controller_replay["state"])),
        np.asarray(pack_state(controller_replay["state"])),
        rtol=5.0e-12,
        atol=5.0e-12,
    )
    for key in ("active", "accepted", "rejected", "done", "state_reset"):
        np.testing.assert_array_equal(
            np.asarray(segmented_controller_replay["history"][key]),
            np.asarray(controller_replay["history"][key]),
        )
        np.testing.assert_array_equal(
            np.asarray(stacked_controller_replay["history"][key]),
            np.asarray(controller_replay["history"][key]),
        )
    static_changed_trace = dict(trace1)
    static_changed_trace["include_edge_residual"] = not bool(trace1["include_edge_residual"])
    assert [
        (segment["start"], segment["stop"], segment["n_steps"])
        for segment in direct_coil_accepted_trace_step_policy_segments([trace0, static_changed_trace])
    ] == [(0, 1, 1), (1, 2, 1)]
    padded_bad_trace = dict(trace1)
    padded_bad_trace["dt_eff"] = _trace_scalar_value(trace1["dt_eff"]) * 10.0
    padded_bad_trace["force_scale"] = _trace_scalar_value(trace1["force_scale"]) * 10.0
    padded_controller_replay = direct_coil_accepted_trace_controller_replay_objective_jax(
        base_params,
        trace0["state_pre"],
        static=init.static,
        traces=[trace0, trace1, padded_bad_trace],
        signgs=int(init.signgs),
        state_weight=1.0,
        bsqvac_weight=1.0e-12,
        force_weight=0.0,
        enforce_edge=False,
        accept_mask=np.asarray([True, True, False]),
        done_mask=np.asarray([False, True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(padded_controller_replay["history"]["active"]),
        np.asarray([True, True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(padded_controller_replay["history"]["accepted"]),
        np.asarray([True, True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(padded_controller_replay["controls"]["reset_to_trace_pre"]),
        np.asarray([False, False, True]),
    )
    assert padded_controller_replay["preconditioner_policy_n_segments"] == 1
    assert [
        (segment["start"], segment["stop"], segment["n_steps"])
        for segment in padded_controller_replay["preconditioner_policy_segments"]
    ] == [(0, 3, 3)]
    assert [
        (
            segment["start"],
            segment["stop"],
            segment["accepted_steps"],
            segment["rejected_steps"],
            segment["done_markers"],
            segment["free_boundary_replay_steps"],
            segment["state_resets"],
        )
        for segment in padded_controller_replay["preconditioner_policy_segment_summary"]
    ] == [(0, 3, 2, 1, 1, 3, 1)]
    np.testing.assert_allclose(
        np.asarray(padded_controller_replay["objective"]),
        np.asarray(controller_replay["objective"]),
        rtol=2.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("lasym", [False, True], ids=["stellsym", "lasym"])
def test_jax_free_boundary_boundary_geometry_matches_host_sampler(
    tmp_path: Path,
    lasym: bool,
) -> None:
    """The phase-2 JAX boundary sampler must match production host geometry."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp
    from vmec_jax.driver import run_free_boundary
    from vmec_jax.free_boundary import _sample_external_boundary_arrays
    from vmec_jax.free_boundary_adjoint import free_boundary_boundary_geometry_jax
    from vmec_jax.state import pack_state, unpack_state

    enable_x64(True)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / f"input.direct_boundary_geometry_{int(lasym)}",
        lasym=lasym,
        niter=1,
        mpol=3,
        ntheta=6,
    )
    params = _circle_coil_params(current=3.0e7, n_segments=64)
    run = run_free_boundary(
        input_path,
        use_initial_guess=True,
        verbose=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )
    host = _sample_external_boundary_arrays(
        state=run.state,
        static=run.static,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
    )
    geom = free_boundary_boundary_geometry_jax(run.state, run.static)
    for key in ("R", "Z", "phi", "Ru", "Zu", "Rv", "Zv", "ruu", "ruv", "rvv", "zuu", "zuv", "zvv"):
        np.testing.assert_allclose(
            np.asarray(geom[key]),
            np.asarray(getattr(host, key)),
            rtol=1.0e-13,
            atol=1.0e-13,
            err_msg=f"JAX boundary geometry mismatch for {key}",
        )

    def geometry_objective(flat_state):
        state = unpack_state(flat_state, run.state.layout)
        g = free_boundary_boundary_geometry_jax(state, run.static)
        return jnp.mean(g["R"] * g["R"] + g["Z"] * g["Z"])

    grad = jax.grad(geometry_objective)(jnp.asarray(pack_state(run.state)))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


@pytest.mark.parametrize("lasym", [False, True], ids=["stellsym", "lasym"])
@pytest.mark.py311_coverage_only
def test_jax_nestor_operator_fixed_boundary_ad_matches_central_fd_for_coil_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lasym: bool,
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
    from vmec_jax.free_boundary import _sample_external_boundary_arrays
    from vmec_jax.free_boundary_adjoint import (
        dense_vmec_nestor_mode_solve_jax,
        direct_coil_boundary_replay_context,
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
        lasym=lasym,
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
    context = direct_coil_boundary_replay_context(init.static, {"R": sample.R})
    basis = context["basis"]
    tables = context["tables"]
    nvper = context["nvper"]

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
    wint_jax = jnp.asarray(context["wint"])
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
def test_essos_full_solve_state_central_fd_response_to_current_and_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional full-solve finite-difference guard for ESSOS coil controls.

    This samples complete solves at +/- perturbations around one current and
    one Fourier curve coefficient. It is a finite-response guard, not a
    full-loop AD validation.
    """

    if os.environ.get("RUN_FULL", "") != "1":
        pytest.skip("Set RUN_FULL=1 to run the optional ESSOS full-solve sensitivity test")
    if not LPQA_COILS.exists():
        pytest.skip(f"missing local ESSOS Landreman-Paul QA coils: {LPQA_COILS}")
    pytest.importorskip("netCDF4")
    essos_coils = pytest.importorskip("essos.coils")

    from vmec_jax._compat import jnp

    enable_x64(True)
    coils = essos_coils.Coils_from_json(str(LPQA_COILS))
    base_params = from_essos_coils(coils, chunk_size=256)
    input_path = _write_lpqa_direct_freeb_input(tmp_path / "input.lpqa_direct_finite_pressure")

    current_index = _first_nonzero_current_index(base_params)
    geometry_index = _first_fourier_geometry_index(base_params)
    base_currents = jnp.asarray(base_params.base_currents)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    current_step = 0.05
    geometry_step = max(2.5e-3, 1.0e-3 * abs(float(np.asarray(base_dofs[geometry_index]))))

    def current_params(sign: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_currents=base_currents.at[current_index].multiply(1.0 + float(sign) * current_step)
        )

    def geometry_params(sign: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_curve_dofs=base_dofs.at[geometry_index].add(float(sign) * geometry_step)
        )

    def solve(label: str, params: CoilFieldParams):
        run = _run_direct_solve(input_path, params)
        _assert_full_solve_wout_sanity(run, tmp_path / f"wout_{label}.nc")
        return run

    monkeypatch.setenv("VMEC_JAX_FREEB_ACTIVATE_FSQ", "1.0e99")
    current_minus = solve("current_minus", current_params(-1.0))
    current_plus = solve("current_plus", current_params(1.0))
    geometry_minus = solve("geometry_minus", geometry_params(-1.0))
    geometry_plus = solve("geometry_plus", geometry_params(1.0))

    runs = {
        "current_minus": current_minus,
        "current_plus": current_plus,
        "geometry_minus": geometry_minus,
        "geometry_plus": geometry_plus,
    }
    inactive = [label for label, run in runs.items() if not _active_free_boundary(run)]
    if inactive:
        pytest.xfail(
            "Optional direct-coil finite-pressure full solve did not enter active "
            f"free-boundary vacuum coupling within the gated short budget: {inactive}"
        )

    current_response = _relative_rms_delta(pack_state(current_minus.state), pack_state(current_plus.state))
    geometry_response = _relative_rms_delta(pack_state(geometry_minus.state), pack_state(geometry_plus.state))
    current_fd_rms = _state_central_difference_rms(current_plus, current_minus, step=current_step)
    geometry_fd_rms = _state_central_difference_rms(geometry_plus, geometry_minus, step=geometry_step)

    assert current_response > 1.0e-12
    assert geometry_response > 1.0e-12
    assert current_fd_rms > 0.0
    assert geometry_fd_rms > 0.0

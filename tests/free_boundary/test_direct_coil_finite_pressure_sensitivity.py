from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import (
    circular_coil_params as _shared_circular_coil_params,
    direct_free_boundary_initial_guess as _run_direct_initial_guess,
    direct_free_boundary_solve as _run_direct_solve,
    direct_nestor_step as _direct_nestor_step,
    tiny_direct_freeb_input as _write_tiny_direct_freeb_input,
)
from vmec_jax._compat import enable_x64, has_jax
from vmec_jax.external_fields import CoilFieldParams, from_essos_coils
from vmec_jax.free_boundary import nestor_external_only_step
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.profiles import eval_profiles
from vmec_jax.state import pack_state


ROOT = Path(__file__).resolve().parents[2]
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


def _assert_errors_contain(report: dict, *snippets: str) -> None:
    errors = report["errors"]
    for snippet in snippets:
        assert any(snippet in error for error in errors)


def _assert_branch_local_replay_contract(
    report: dict,
    *,
    rejected_slots: int = 0,
    graph_metadata: bool = True,
) -> None:
    assert report["uses_production_forward"] is True
    assert report["differentiates_adaptive_controller"] is False
    assert report["differentiates_run_free_boundary"] is False
    assert report["differentiates_fixed_accepted_branch"] is True
    assert report["production_values_source"] == "precomputed"
    assert report["replay_payload_source"] == "user"
    assert report["includes_payload"] is False
    assert report["payload"] is None
    assert report["trace_replay_diagnostics"]["differentiates_adaptive_controller"] is False
    assert report["replay_option_flags"]["use_stacked_step_controls"] is True
    assert report["replay_option_flags"]["use_accepted_only_fast_path"] is (rejected_slots == 0)
    assert report["controller_slot_summary"]["accepted_slots"] >= 1
    assert report["controller_slot_summary"]["rejected_slots"] == rejected_slots
    if graph_metadata:
        metadata = report["replay_graph_metadata"]
        assert metadata["differentiates_adaptive_controller"] is False
        assert metadata["n_steps"] >= 1
        assert metadata["active_free_boundary_replay_steps"] >= 1
        assert metadata["step_policy_n_segments"] >= 1


def _assert_nonnegative_timings(report: dict, *keys: str) -> None:
    for key in keys:
        assert report["timings"][key] >= 0.0


def _circle_coil_params(*, current: float = 3.0e7, radius: float = 1.8, n_segments: int = 96) -> CoilFieldParams:
    return _shared_circular_coil_params(current=current, radius=radius, n_segments=n_segments)


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


def _run_forced_active_direct_solve(input_path: Path, params: CoilFieldParams, *, max_iter: int, **overrides):
    """Run a tiny direct-coil solve with free-boundary updates active from iteration one."""

    from vmec_jax.driver import run_free_boundary

    kwargs = {
        "max_iter": int(max_iter),
        "multigrid": False,
        "verbose": False,
        "jit_forces": False,
        "external_field_provider_kind": "direct_coils",
        "external_field_provider_params": params,
        "free_boundary_activate_fsq": 1.0e99,
    }
    kwargs.update(overrides)
    return run_free_boundary(input_path, **kwargs)


def _solve_direct_residual_iter(init, params: CoilFieldParams, *, max_iter: int, **overrides):
    """Advance a tiny direct-coil state through the VMEC2000-style residual loop."""

    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    kwargs = {
        "max_iter": int(max_iter),
        "ftol": 1.0e-8,
        "vmec2000_control": True,
        "auto_flip_force": False,
        "use_direct_fallback": True,
        "verbose": False,
        "verbose_vmec2000_table": False,
        "jit_forces": False,
        "external_field_provider_kind": "direct_coils",
        "external_field_provider_params": params,
        "free_boundary_activate_fsq": 1.0e99,
    }
    kwargs.update(overrides)
    return solve_fixed_boundary_residual_iter(
        init.state,
        init.static,
        indata=init.indata,
        signgs=init.signgs,
        **kwargs,
    )


def _synthetic_direct_coil_trace(
    z: np.ndarray,
    *,
    dt_eff: float = 0.5,
    bsqvac_scale: float = 1.0,
    axis_offset: float | None = None,
    **updates,
) -> dict:
    nestor_trace = {"gsource": np.ones(2), "bsqvac": np.ones(2)}
    if axis_offset is not None:
        nestor_trace |= {
            "br_axis": np.ones((2, 3)) + axis_offset,
            "bp_axis": np.ones((2, 3)) * 2.0 + axis_offset,
            "bz_axis": np.ones((2, 3)) * 3.0 + axis_offset,
    }
    trace = {
        "dt_eff": np.asarray(dt_eff),
        "b1": np.asarray(0.125), "fac": np.asarray(0.9),
        "force_scale": np.asarray(1.0), "max_update_rms_pre": np.asarray(0.25),
        "lambda_update_scale": np.asarray([1.0, 0.5]),
        "limit_update_rms": np.asarray(1.0),
        "flip_sign": False, "divide_by_scalxc_for_update": True,
        "preconditioner_use_precomputed_tridi": False, "preconditioner_use_lax_tridi": True,
        "precond_jmax": 2,
        "precond_mats": {"ar": z + 6.0, "br": z + 7.0},
        "lam_prec": np.asarray([1.0, 2.0, 3.0]),
        "w_mode_mn": np.ones((2, 3)),
        "vRcc_before": z, "vRss_before": z + 1.0,
        "vZsc_before": z + 2.0, "vZcs_before": z + 3.0,
        "vLsc_before": z + 4.0, "vLcs_before": z + 5.0,
        "freeb_bsqvac_half": np.ones((2, 3)) * bsqvac_scale,
        "freeb_nestor_trace": nestor_trace,
        "state_pre": np.ones(4), "state_post": np.ones(4) * 2.0,
    }
    trace.update(updates)
    return trace


def _synthetic_same_branch_replay_report(trace0: dict, trace1: dict) -> dict:
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_accepted_trace_fingerprint,
        free_boundary_adjoint_trace_replay_diagnostics,
    )

    traces = [trace0, trace1]
    fingerprint = direct_coil_accepted_trace_fingerprint(traces)
    return {
        "branch_compatibility": {
            "same_branch": True,
            "same_accepted_trace_branch": True,
            "same_residual_branch": True,
            "base_fingerprint": fingerprint,
            "plus_fingerprint": fingerprint,
            "minus_fingerprint": fingerprint,
        },
        "trace_replay_diagnostics": {
            label: free_boundary_adjoint_trace_replay_diagnostics(traces)
            for label in ("base", "plus", "minus")
        },
    }


def _synthetic_physical_scalar_inputs(synthetic_report: dict, branch_metadata: dict, trace0: dict, trace1: dict) -> tuple[dict, dict]:
    physical_report = deepcopy(synthetic_report)
    physical_report.update({label: {"traces": (trace0, trace1)} for label in ("base", "plus", "minus")})
    physical_report["objective_values"] = {
        "aspect": {"base": 5.0, "plus": 5.01, "minus": 4.99, "central_fd_directional": 100.0},
        "accepted_bnormal_rms": {
            "base": 0.2, "plus": 0.21, "minus": 0.19, "central_fd_directional": 100.0,
        },
    }
    scalar_report = {
        "passed": True, "same_branch": True, "uses_production_forward": True,
        "differentiates_adaptive_controller": False, "differentiates_run_free_boundary": False,
        "differentiates_fixed_accepted_branch": True,
        "replay_option_flags": {"use_stacked_step_controls": True},
        "replay_branch_metadata": branch_metadata,
        "scalar_keys": ("aspect", "accepted_bnormal_rms"),
        "scalar_reports": {
            key: {
                "passed": True, "same_branch": True, "exact_directional": 100.0,
                "abs_error": 0.0, "rel_error": 0.0, "base_abs_delta": 1.0e-6,
            }
            for key in ("aspect", "accepted_bnormal_rms")
        },
    }
    return physical_report, scalar_report


def _assert_synthetic_physical_and_adaptive_gate_reports(
    *,
    physical_synthetic_report: dict,
    physical_scalars_report: dict,
    trace0: dict,
    trace1: dict,
) -> None:
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_adaptive_full_loop_same_branch_gate_report,
        direct_coil_same_branch_physical_scalar_gate_report,
    )

    physical_gate = direct_coil_same_branch_physical_scalar_gate_report(
        physical_synthetic_report,
        physical_scalars_report,
    )
    assert physical_gate["passed"], physical_gate
    assert physical_gate["scalar_keys"] == ("aspect", "accepted_bnormal_rms")
    assert physical_gate["controller_slot_summary"]["accepted_slots"] == 2
    assert physical_gate["controller_slot_summary"]["rejected_slots"] == 0
    assert physical_gate["differentiates_adaptive_controller"] is False
    assert physical_gate["same_accepted_trace_branch"] is True
    assert physical_gate["same_residual_branch"] is True

    adaptive_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        physical_synthetic_report,
        physical_scalars_report,
    )
    assert adaptive_gate["passed"], adaptive_gate
    assert adaptive_gate["contract"] == "same-branch adaptive full-loop seam report"
    assert adaptive_gate["ad_vs_fd_gate"] == "complete-loop central FD vs branch-local stacked replay custom VJP"
    assert adaptive_gate["adaptive_loop_scope"] == "fingerprint-gated branch-local accepted/rejected replay slots"
    assert adaptive_gate["differentiates_adaptive_controller"] is False
    assert adaptive_gate["differentiates_run_free_boundary"] is False
    assert adaptive_gate["same_stacked_step_policy_branch"] is True
    assert adaptive_gate["used_stacked_step_controls"] is True
    assert adaptive_gate["controller_slot_summary"]["accepted_slots"] == 2
    assert adaptive_gate["controller_slot_summary"]["rejected_slots"] == 0
    assert adaptive_gate["controller_slot_summary"]["fixed_rejected_controller_slot_present"] is False

    missing_rejected_slot_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        physical_synthetic_report,
        physical_scalars_report,
        require_fixed_rejected_controller_slot=True,
    )
    assert not missing_rejected_slot_gate["passed"]
    _assert_errors_contain(missing_rejected_slot_gate, "fixed rejected controller slot", "accepted-only fast path")
    missing_complete_loop_rejected_slot_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        physical_synthetic_report,
        physical_scalars_report,
        require_complete_loop_rejected_controller_slot=True,
    )
    assert not missing_complete_loop_rejected_slot_gate["passed"]
    assert missing_complete_loop_rejected_slot_gate["requires_complete_loop_rejected_controller_slot"] is True
    assert missing_complete_loop_rejected_slot_gate["complete_loop_rejected_controller_slot_present"] is False
    _assert_errors_contain(missing_complete_loop_rejected_slot_gate, "complete-loop branch fingerprints")
    missing_status_rejected_slot_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        physical_synthetic_report,
        physical_scalars_report,
        require_status_derived_rejected_controller_slot=True,
    )
    assert not missing_status_rejected_slot_gate["passed"]
    _assert_errors_contain(missing_status_rejected_slot_gate, "trace step_status")

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
    _assert_errors_contain(
        bad_physical_gate,
        "replay gate failed",
        "not same-branch",
        "accepted-trace branch fingerprint changed",
        "residual-controller branch fingerprint changed",
        "missing scalar report",
        "missing complete-solve objective values",
        "non-finite complete-solve FD",
        "non-finite custom-VJP",
    )
    bad_adaptive_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        bad_physical_report,
        bad_physical_scalars_report,
        scalar_keys=("aspect", "accepted_bnormal_rms", "missing", "missing_objective"),
    )
    assert not bad_adaptive_gate["passed"]
    _assert_errors_contain(
        bad_adaptive_gate,
        "stacked step-control replay was not used",
        "stacked step-policy branch changed",
        "base: no accepted step-policy segments",
        "minus: missing complete-solve payload",
        "physical scalar gate:",
    )


def _assert_direct_coil_trace_control_array_contracts(trace0: dict, trace1: dict, z: np.ndarray) -> None:
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_accepted_trace_array_controls_jax,
        direct_coil_accepted_trace_preconditioner_controls_jax,
        direct_coil_accepted_trace_preconditioner_policy_segments,
        direct_coil_accepted_trace_scalar_controls_jax,
        direct_coil_accepted_trace_step_controls_jax,
    )

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


def _assert_direct_coil_trace_replay_graph_contracts(trace0: dict, trace1: dict, z: np.ndarray) -> tuple[dict, dict]:
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_accepted_trace_controller_controls_jax,
        direct_coil_accepted_trace_replay_graph_metadata,
        direct_coil_accepted_trace_step_controls_jax,
        direct_coil_accepted_trace_step_policy_segment_summary,
        direct_coil_accepted_trace_step_policy_segments,
        _accepted_trace_effective_controller_masks,
        _accepted_trace_segment_is_unconditionally_accepted,
        _direct_coil_trace_boundary_shape,
    )

    axis_trace0 = _synthetic_direct_coil_trace(z, axis_offset=0.0)
    axis_trace1 = _synthetic_direct_coil_trace(z, dt_eff=0.25, bsqvac_scale=3.0, axis_offset=10.0)
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
    mixed_segment_controls = direct_coil_accepted_trace_controller_controls_jax(
        [axis_trace0, changed_static_trace],
        accept_mask=np.asarray([True, False]),
        done_mask=np.asarray([False, False]),
    )
    mixed_segment_masks = _accepted_trace_effective_controller_masks(mixed_segment_controls)
    assert _accepted_trace_segment_is_unconditionally_accepted(mixed_segment_masks, start=0, stop=1)
    assert not _accepted_trace_segment_is_unconditionally_accepted(mixed_segment_masks, start=1, stop=2)
    assert _direct_coil_trace_boundary_shape(axis_trace0) == (2, 3)
    bsqvac_only_trace = deepcopy(trace0)
    bsqvac_only_trace["freeb_nestor_trace"] = {}
    assert _direct_coil_trace_boundary_shape(bsqvac_only_trace) == (2, 3)
    no_shape_trace = deepcopy(trace0)
    no_shape_trace["freeb_bsqvac_half"] = None
    no_shape_trace["freeb_nestor_trace"] = {}
    assert _direct_coil_trace_boundary_shape(no_shape_trace) is None
    replay_graph = direct_coil_accepted_trace_replay_graph_metadata(
        [axis_trace0, changed_static_trace],
        static=SimpleNamespace(cfg=SimpleNamespace(nfp=3, mpol=4, ntor=2, lasym=True)),
        accept_mask=np.asarray([True, False]),
        done_mask=np.asarray([False, False]),
        sample_nzeta=3,
        include_analytic=False,
        use_stacked_step_controls=False,
        use_accepted_only_fast_path=False,
        json_safe=True,
    )
    expected_replay_graph = {
        "contract": "fixed accepted-branch replay graph metadata",
        "n_steps": 2, "accepted_steps": 1, "rejected_steps": 1, "done_markers": 0, "state_resets": 0,
        "free_boundary_trace_steps": 2, "active_free_boundary_replay_steps": 1,
        "step_policy_n_segments": 2, "preconditioner_policy_n_segments": 1,
        "boundary_shapes": [[2, 3]], "bsqvac_half_shapes": [[2, 3]], "nestor_axis_shapes": [[2, 3]],
        "inferred_boundary_shape": [2, 3], "sample_nzeta": 3,
        "nfp": 3, "mpol": 4, "ntor": 2, "lasym": True, "nvper": 3,
        "include_analytic": False, "use_stacked_step_controls": False, "use_accepted_only_fast_path": False,
    }
    for key, value in expected_replay_graph.items():
        assert replay_graph[key] == value
    assert "signature_repr" not in replay_graph["step_policy_segment_summary"][0]
    assert "signature_repr" not in replay_graph["preconditioner_policy_segment_summary"][0]
    json.dumps(replay_graph, allow_nan=False)
    with pytest.raises(ValueError, match="accepted trace"):
        direct_coil_accepted_trace_replay_graph_metadata([])
    return axis_trace0, changed_static_trace


def _assert_trace_fingerprint_status_and_replay_gate_contracts(trace0: dict, trace1: dict, trace2: dict):
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_accepted_trace_branch_metadata,
        direct_coil_accepted_trace_controller_replay_plan,
        direct_coil_accepted_trace_controller_slot_fingerprint,
        direct_coil_accepted_trace_controller_slot_summary,
        direct_coil_accepted_trace_controller_controls_jax,
        direct_coil_accepted_trace_status_masks,
        direct_coil_same_branch_replay_gate_report,
        free_boundary_adjoint_trace_replay_diagnostics,
        _accepted_trace_segment_is_unconditionally_accepted,
    )

    branch_metadata = direct_coil_accepted_trace_branch_metadata([trace0, trace1])
    branch_slot_summary = direct_coil_accepted_trace_controller_slot_summary(branch_metadata)
    assert branch_metadata["n_steps"] == 2
    assert branch_metadata["n_free_boundary_replay_steps"] == 2
    assert branch_slot_summary == {"n_steps": 2, "active_slots": 2, "accepted_slots": 2, "rejected_slots": 0, "done_markers": 1, "active_free_boundary_slots": 2, "accepted_free_boundary_slots": 2, "fixed_rejected_controller_slot_present": False}
    assert branch_metadata["fingerprint"]["n_freeb_steps"] == 2
    for key, expected in (
        ("accepted_mask", [True, True]),
        ("done_mask", [False, True]),
        ("reset_to_trace_pre", [False, False]),
        ("active_free_boundary_mask", [True, True]),
    ):
        assert np.array_equal(np.asarray(branch_metadata[key]), np.asarray(expected))
    assert branch_metadata["preconditioner_policy_segment_summary"][0]["free_boundary_replay_steps"] == 2
    assert _accepted_trace_segment_is_unconditionally_accepted(branch_metadata["masks"], start=0, stop=2)
    status_rejected_trace = {**trace1, "step_status": "rejected"}
    status_masks = direct_coil_accepted_trace_status_masks([trace0, status_rejected_trace])
    assert status_masks["step_status"] == ("accepted", "rejected")
    assert np.array_equal(status_masks["accept_mask"], np.asarray([True, False]))
    assert status_masks["status_acceptance_source"] == "trace_step_status"
    status_controls = direct_coil_accepted_trace_controller_controls_jax([trace0, status_rejected_trace])
    assert np.array_equal(np.asarray(status_controls["accept"]), np.asarray([True, False]))
    status_metadata = direct_coil_accepted_trace_branch_metadata([trace0, status_rejected_trace])
    status_slot_summary = direct_coil_accepted_trace_controller_slot_summary(status_metadata)
    status_slot_fingerprint = direct_coil_accepted_trace_controller_slot_fingerprint(status_metadata)
    assert status_metadata["step_status"] == ("accepted", "rejected")
    for key, expected in (("accepted_mask", [True, False]), ("rejected_mask", [False, True])):
        assert np.array_equal(np.asarray(status_metadata[key]), np.asarray(expected))
    assert status_metadata["preconditioner_policy_segment_summary"][0]["rejected_steps"] == 1
    assert status_slot_summary["accepted_slots"] == 1
    assert status_slot_summary["rejected_slots"] == 1
    assert status_slot_summary["fixed_rejected_controller_slot_present"] is True
    assert status_slot_fingerprint["accepted_mask"] == [True, False]
    assert status_slot_fingerprint["rejected_mask"] == [False, True]
    assert status_slot_fingerprint["step_status"] == ["accepted", "rejected"]
    assert status_slot_fingerprint["status_acceptance_source"] == "trace_step_status"
    assert status_slot_fingerprint["summary"]["rejected_slots"] == 1
    override_status_metadata = direct_coil_accepted_trace_branch_metadata(
        [trace0, status_rejected_trace],
        accept_mask=np.asarray([True, True]),
    )
    for key, expected in (("accepted_mask", [True, True]), ("rejected_mask", [False, False])):
        assert np.array_equal(np.asarray(override_status_metadata[key]), np.asarray(expected))
    override_plan = direct_coil_accepted_trace_controller_replay_plan(
        [trace0, status_rejected_trace],
        static=SimpleNamespace(cfg=SimpleNamespace(nfp=1, mpol=2, ntor=0, lasym=False)),
        accept_mask=np.asarray([True, False]),
        use_stacked_step_controls=False,
        use_accepted_only_fast_path=False,
    )
    assert override_plan["status_masks"]["step_status"] == ("accepted", "rejected")
    assert np.array_equal(np.asarray(override_plan["controls"]["accept"]), np.asarray([True, False]))
    branch_metadata_json = direct_coil_accepted_trace_branch_metadata([trace0, trace1], accept_mask=np.asarray([True, False]), done_mask=np.asarray([False, False]), json_safe=True)
    rejected_slot_summary = direct_coil_accepted_trace_controller_slot_summary(branch_metadata_json)
    json.dumps(branch_metadata_json, allow_nan=False)
    assert branch_metadata_json["accepted_mask"] == [True, False]
    assert branch_metadata_json["active_free_boundary_mask"] == [True, False]
    assert branch_metadata_json["preconditioner_policy_segment_summary"][0]["rejected_steps"] == 1
    assert rejected_slot_summary["accepted_slots"] == 1
    assert rejected_slot_summary["rejected_slots"] == 1
    assert rejected_slot_summary["fixed_rejected_controller_slot_present"] is True
    padded_diagnostics = free_boundary_adjoint_trace_replay_diagnostics(
        {"adjoint_step_trace": [trace0, trace1, trace2]},
        accept_mask=np.asarray([True, True, False]),
        done_mask=np.asarray([False, True, False]),
    )
    assert padded_diagnostics["differentiates_adaptive_controller"] is False
    assert padded_diagnostics["n_steps"] == 3
    assert padded_diagnostics["branch_fingerprint"]["n_steps"] == 3
    for key, expected in (
        ("active", [True, True, False]),
        ("accepted", [True, True, False]),
        ("rejected", [False, False, False]),
        ("done", [False, True, True]),
    ):
        assert np.array_equal(np.asarray(padded_diagnostics["masks"][key]), np.asarray(expected))
    assert padded_diagnostics["replay_diagnostics"]["preconditioner_policy_n_segments"] == 1
    assert padded_diagnostics["replay_diagnostics"]["scalar_controls_stackable"] is True
    assert padded_diagnostics["replay_diagnostics"]["array_controls_stackable"] is True
    assert padded_diagnostics["replay_diagnostics"]["preconditioner_controls_stackable"] is True
    padded_json = free_boundary_adjoint_trace_replay_diagnostics({"diagnostics": {"adjoint_step_trace": [trace0, trace1, trace2]}}, accept_mask=np.asarray([True, True, False]), done_mask=np.asarray([False, True, False]), json_safe=True)
    json.dumps(padded_json, allow_nan=False)
    assert padded_json["masks"]["done"] == [False, True, True]
    with pytest.raises(RuntimeError, match="adjoint_trace=True"):
        free_boundary_adjoint_trace_replay_diagnostics({"diagnostics": {}})
    synthetic_report = _synthetic_same_branch_replay_report(trace0, trace1)
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
    _assert_errors_contain(mismatch_gate, "base: missing branch fingerprint", "plus: n_steps mismatch", "plus: fingerprint n_steps mismatch", "plus: fingerprint n_freeb_steps mismatch", "plus: freeb_sizes mismatch", "plus: mask 'active' has shape", "plus: no accepted active free-boundary replay slots", "plus: scalar controls are not stackable", "plus: array controls are not stackable", "plus: no preconditioner policy segments", "minus: missing replay diagnostics")
    return synthetic_report, branch_metadata


@pytest.mark.py311_coverage_only
def test_direct_coil_trace_fingerprint_detects_control_branch_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    from vmec_jax._compat import jax, jnp

    z = np.arange(6.0).reshape(2, 3)
    trace0 = _synthetic_direct_coil_trace(z)
    trace1 = _synthetic_direct_coil_trace(z, dt_eff=0.25, bsqvac_scale=3.0)
    trace2 = _synthetic_direct_coil_trace(z, dt_eff=0.125, bsqvac_scale=4.0)
    _assert_direct_coil_trace_control_array_contracts(trace0, trace1, z)
    axis_trace0, changed_static_trace = _assert_direct_coil_trace_replay_graph_contracts(trace0, trace1, z)
    synthetic_report, branch_metadata = _assert_trace_fingerprint_status_and_replay_gate_contracts(
        trace0,
        trace1,
        trace2,
    )

    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        _block_until_ready_for_timing,
        _pytree_batched_directional_vdot_jax,
        direct_coil_accepted_trace_fingerprint,
        direct_coil_accepted_trace_fingerprint_delta,
        direct_coil_accepted_trace_fingerprint_delta_summary,
        direct_coil_accepted_trace_preconditioner_controls_jax,
        direct_coil_accepted_trace_preconditioner_policy_segments,
        direct_coil_accepted_trace_controller_custom_vjp_scalars_jax,
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax,
        direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax,
        direct_coil_same_branch_controller_scalar_custom_vjp_report,
        direct_coil_same_branch_controller_scalars_custom_vjp_report,
        _accepted_step_policy_signature_for_complete_payload,
        _accepted_step_policy_summary_for_complete_payload,
        _pytree_pullback_basis_jax,
        _pytree_unstack_leading_axis_jax,
    )

    physical_synthetic_report, physical_scalars_report = _synthetic_physical_scalar_inputs(
        synthetic_report, branch_metadata, trace0, trace1
    )
    _assert_synthetic_physical_and_adaptive_gate_reports(
        physical_synthetic_report=physical_synthetic_report,
        physical_scalars_report=physical_scalars_report,
        trace0=trace0,
        trace1=trace1,
    )

    def scalar_call(**kwargs):
        return direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
            scalar_fn=lambda payload: {"objective": 0.0},
            replay_scalar_fn=lambda replay, payload: 0.0,
            **kwargs,
        )

    def scalars_call(**kwargs):
        return direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
            scalar_fn=lambda payload: {"objective": 0.0},
            replay_scalar_fns={"objective": lambda replay, payload: 0.0},
            **kwargs,
        )

    with pytest.raises(ValueError, match="input_path and params"):
        scalar_call()
    with pytest.raises(ValueError, match="replay_scalar_fns"):
        direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
            params={},
            complete_payload={"traces": (), "init": object()},
            scalar_fn=lambda payload: {"objective": 0.0},
            replay_scalar_fns={},
        )
    with pytest.raises(ValueError, match="input_path and params"):
        scalars_call()
    for call, error_type, match, kwargs in (
        (scalar_call, ValueError, "params must be supplied", {"complete_payload": {"traces": (), "init": object()}}),
        (scalar_call, ValueError, "no accepted traces", {"params": {}, "complete_payload": {"traces": (), "init": object()}}),
        (scalars_call, ValueError, "no accepted traces", {"params": {}, "complete_payload": {"traces": (), "init": object()}}),
        (
            scalar_call,
            RuntimeError,
            "no active free-boundary trace",
            {"params": {}, "complete_payload": {"traces": ({"freeb_bsqvac_half": None},), "init": object()}},
        ),
        (
            scalars_call,
            RuntimeError,
            "no active free-boundary trace",
            {"params": {}, "complete_payload": {"traces": ({"freeb_bsqvac_half": None},), "init": object()}},
        ),
        (
            scalar_call,
            ValueError,
            "initialization result",
            {"params": {}, "complete_payload": {"traces": ({"freeb_bsqvac_half": np.ones(1)},)}},
        ),
        (
            scalars_call,
            ValueError,
            "initialization result",
            {"params": {}, "complete_payload": {"traces": ({"freeb_bsqvac_half": np.ones(1)},)}},
        ),
    ):
        with pytest.raises(error_type, match=match):
            call(**kwargs)
    invalid_mode_payload = {
        "params": {},
        "init": SimpleNamespace(static=None, signgs=1),
        "traces": (
            {
                "freeb_bsqvac_half": np.ones(1),
                "freeb_nestor_trace": {"active": True},
                "state_pre": object(),
            },
        ),
    }
    for call in (scalar_call, scalars_call):
        with pytest.raises(ValueError, match="replay_ad_mode"):
            call(complete_payload=invalid_mode_payload, replay_ad_mode="invalid")
    with pytest.raises(ValueError, match="direction_params"):
        scalars_call(
            params={},
            direction_params={},
            complete_payload=invalid_mode_payload,
            replay_ad_mode="custom_vjp",
        )

    import vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives as freeb_adj

    replay_options_seen: list[dict[str, object]] = []

    def fake_direct_coil_replay(coil_params, _state_pre, **_replay_options):
        replay_options_seen.append(dict(_replay_options))
        if isinstance(coil_params, CoilFieldParams):
            x = coil_params.base_currents[0]
        else:
            x = coil_params["x"]
        return {"linear": 2.0 * x, "quadratic": x * x}

    monkeypatch.setattr(
        freeb_adj,
        "direct_coil_accepted_trace_controller_replay_objective_jax",
        fake_direct_coil_replay,
    )
    synthetic_jvp_payload = {
        "params": {"x": jnp.asarray(2.0)},
        "init": SimpleNamespace(
            static=SimpleNamespace(cfg=SimpleNamespace(nfp=1, mpol=2, ntor=0, lasym=False)),
            signgs=1,
        ),
        "traces": (axis_trace0,),
    }
    synthetic_jvp_report = direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
        complete_payload=synthetic_jvp_payload,
        direction_params={"x": jnp.asarray(0.25)},
        scalar_fn=lambda payload: {"linear": 4.0, "quadratic": 4.0},
        production_values={"linear": 4.0, "quadratic": 4.0},
        replay_scalar_fns={
            "linear": lambda replay, payload: replay["linear"],
            "quadratic": lambda replay, payload: replay["quadratic"],
        },
        include_trace_replay_diagnostics=False,
        include_replay_graph_metadata=False,
    )
    assert synthetic_jvp_report["derivative_mode"] == "directional_jvp"
    assert synthetic_jvp_report["jacobian"] is None
    assert synthetic_jvp_report["grads"] == {}
    assert synthetic_jvp_report["production_values_source"] == "precomputed"
    assert synthetic_jvp_report["trace_replay_diagnostics"]["omitted"] is True
    assert synthetic_jvp_report["replay_option_flags"]["replay_ad_mode"] == "direct"
    assert synthetic_jvp_report["replay_option_flags"]["directional_jvp_fast_path"] == "none"
    assert synthetic_jvp_report["replay_option_flags"]["directional_uses_fixed_coil_geometry"] is False
    assert synthetic_jvp_report["includes_replay_graph_metadata"] is False
    assert synthetic_jvp_report["replay_graph_metadata"]["omitted"] is True
    assert synthetic_jvp_report["replay_graph_metadata"]["differentiates_adaptive_controller"] is False
    np.testing.assert_allclose(
        np.asarray(synthetic_jvp_report["replay_values"]),
        np.asarray([4.0, 4.0]),
    )
    np.testing.assert_allclose(
        np.asarray(synthetic_jvp_report["directional_derivatives"]["linear"]),
        np.asarray(0.5),
    )
    np.testing.assert_allclose(
        np.asarray(synthetic_jvp_report["directional_derivatives"]["quadratic"]),
        np.asarray(1.0),
    )
    assert synthetic_jvp_report["timings"]["replay_jvp_wall_s"] >= 0.0
    assert synthetic_jvp_report["timings"]["replay_vjp_wall_s"] == 0.0
    assert synthetic_jvp_report["timings"]["replay_pullbacks_wall_s"] == 0.0
    assert synthetic_jvp_report["timings"]["replay_graph_metadata_wall_s"] >= 0.0
    assert synthetic_jvp_report["timings"]["jacobian_stack_ready_s"] == 0.0

    current_base_params = _circle_coil_params(current=3.0, n_segments=8)
    current_direction = current_base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(current_base_params.base_curve_dofs),
        base_currents=jnp.asarray([0.25]),
    )
    current_only_report = direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
        params=current_base_params,
        complete_payload={**synthetic_jvp_payload, "params": current_base_params},
        direction_params=current_direction,
        scalar_fn=lambda payload: {"linear": 6.0, "quadratic": 9.0},
        production_values={"linear": 6.0, "quadratic": 9.0},
        replay_scalar_fns={
            "linear": lambda replay, payload: replay["linear"],
            "quadratic": lambda replay, payload: replay["quadratic"],
        },
        include_trace_replay_diagnostics=False,
        include_replay_graph_metadata=False,
    )
    assert current_only_report["derivative_mode"] == "directional_jvp"
    assert current_only_report["replay_option_flags"]["directional_jvp_fast_path"] == "current_only"
    assert current_only_report["replay_option_flags"]["directional_uses_fixed_coil_geometry"] is True
    assert replay_options_seen[-1]["coil_geometry"] is not None
    np.testing.assert_allclose(
        np.asarray(current_only_report["replay_values"]),
        np.asarray([6.0, 9.0]),
    )
    np.testing.assert_allclose(
        np.asarray(current_only_report["directional_derivatives"]["linear"]),
        np.asarray(0.5),
    )
    np.testing.assert_allclose(
        np.asarray(current_only_report["directional_derivatives"]["quadratic"]),
        np.asarray(1.5),
    )

    ready_tree = _block_until_ready_for_timing({"value": jnp.asarray([1.0, 2.0])})
    np.testing.assert_allclose(np.asarray(ready_tree["value"]), np.asarray([1.0, 2.0]))

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
    step_signature_empty = _accepted_step_policy_signature_for_complete_payload({})
    assert step_signature_empty == ()
    step_summary_empty = _accepted_step_policy_summary_for_complete_payload({})
    assert step_summary_empty == {"n_segments": 0, "segments": ()}
    step_signature = _accepted_step_policy_signature_for_complete_payload(
        {"traces": (axis_trace0, changed_static_trace)}
    )
    assert len(step_signature) == 2
    assert step_signature[0][0:3] == (0, 1, 1)
    step_summary = _accepted_step_policy_summary_for_complete_payload(
        {"traces": (axis_trace0, changed_static_trace)}
    )
    assert step_summary["n_segments"] == 2
    assert step_summary["segments"][1] == {"start": 1, "stop": 2, "n_steps": 1}

    def vector_objective(params):
        x = params["x"]
        return jnp.asarray([x * x, 3.0 * x])

    _, toy_pullback = jax.vjp(vector_objective, {"x": jnp.asarray(2.0)})
    batched_pullback = _pytree_pullback_basis_jax(toy_pullback, jnp.eye(2))
    np.testing.assert_allclose(np.asarray(batched_pullback["x"]), np.asarray([4.0, 3.0]))
    unstacked_pullback = _pytree_unstack_leading_axis_jax(batched_pullback, 2)
    np.testing.assert_allclose(np.asarray(unstacked_pullback[0]["x"]), np.asarray(4.0))
    np.testing.assert_allclose(np.asarray(unstacked_pullback[1]["x"]), np.asarray(3.0))

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
        direct_coil_same_branch_controller_scalar_custom_vjp_report(
            {
                "objective_values": {"known": {"base": 0.0, "central_fd_directional": 0.0}},
                "base": {"traces": ()},
            },
            base_params={},
            direction={},
            scalar_key="known",
            replay_scalar_fn=lambda _replay, _payload: 0.0,
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
    with pytest.raises(ValueError, match="replay traces"):
        direct_coil_same_branch_controller_scalars_custom_vjp_report(
            {
                "branch_compatibility": {"same_branch": True},
                "trace_replay_diagnostics": {},
                "objective_values": {"known": {"base": 0.0, "central_fd_directional": 0.0}},
                "base": {
                    "init": SimpleNamespace(static=None, signgs=1),
                    "traces": (trace0,),
                },
            },
            base_params={},
            direction={},
            replay_scalar_fns={"known": lambda _replay, _payload: 0.0},
            replay_kwargs={"traces": ()},
        )

    bad_preconditioner_shape = dict(trace1)
    bad_preconditioner_shape["precond_mats"] = {"ar": np.ones((3, 3)), "br": z + 7.0}
    with pytest.raises(ValueError, match="precond_mats"):
        direct_coil_accepted_trace_preconditioner_controls_jax([trace0, bad_preconditioner_shape])

    fingerprint = direct_coil_accepted_trace_fingerprint([trace0, trace1])
    assert fingerprint["n_steps"] == 2
    assert fingerprint["n_freeb_steps"] == 2
    assert np.array_equal(fingerprint["freeb_sizes"], np.asarray([6, 6]))
    assert fingerprint["step_status"] == ("accepted", "accepted")
    np.testing.assert_array_equal(fingerprint["accept_mask"], np.asarray([1, 1]))
    np.testing.assert_array_equal(fingerprint["done_mask"], np.asarray([0, 1]))
    empty_fingerprint = direct_coil_accepted_trace_fingerprint([])
    assert empty_fingerprint["step_status"] == ()
    np.testing.assert_array_equal(empty_fingerprint["accept_mask"], np.asarray([], dtype=int))
    np.testing.assert_array_equal(empty_fingerprint["done_mask"], np.asarray([], dtype=int))

    same = direct_coil_accepted_trace_fingerprint_delta([trace0, trace1], [trace0, trace1])
    assert same["compatible"]
    same_json = direct_coil_accepted_trace_fingerprint_delta_summary([trace0, trace1], [trace0, trace1])
    json.dumps(same_json, allow_nan=False)
    assert same_json["compatible"]
    assert same_json["reference"]["precond_jmax"] == [2, 2]

    status_change = dict(trace1)
    status_change["step_status"] = "restart_bad_progress"
    status_json = direct_coil_accepted_trace_fingerprint_delta_summary(
        [trace0, trace1],
        [trace0, status_change],
    )
    json.dumps(status_json, allow_nan=False)
    assert status_json["candidate"]["step_status"] == ["accepted", "restart_bad_progress"]
    assert status_json["candidate"]["accept_mask"] == [1, 0]

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

    for trace_index, updates, compatible, changed_fields in (
        (0, {"freeb_bsqvac_half": np.ones((2, 3)) * 99.0}, True, ()),
        (1, {"step_status": "restart_bad_progress"}, False, ("step_status", "accept_mask")),
        (0, {"fac": np.asarray(0.7)}, False, ("scalars.fac",)),
        (0, {"b1": np.asarray(0.25)}, False, ("scalars.b1",)),
        (
            0,
            {"preconditioner_use_lax_tridi": False},
            False,
            ("flags.preconditioner_use_lax_tridi",),
        ),
        (0, {"precond_jmax": 3}, False, ("precond_jmax",)),
        (0, {"precond_mats": {"ar": np.ones((3, 3)), "br": z + 7.0}}, False, ("precond_mats_shapes",)),
        (0, {"freeb_bsqvac_half": np.ones((3, 3))}, False, ("freeb_sizes",)),
    ):
        candidate = [dict(trace0), dict(trace1)]
        candidate[trace_index].update(updates)
        delta = direct_coil_accepted_trace_fingerprint_delta([trace0, trace1], candidate)
        assert delta["compatible"] is compatible
        for field in changed_fields:
            assert field in delta["changed_fields"]


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


_STRICT_ACCEPTED_STEP_TRACE_KEYS = (
    "dt_eff", "b1", "fac", "force_scale", "flip_sign",
    "vRcc_before", "vRss_before", "vZsc_before", "vZcs_before", "vLsc_before", "vLcs_before",
    "frcc_u", "frss_u", "fzsc_u", "fzcs_u", "flsc_u", "flcs_u",
    "limit_update_rms", "divide_by_scalxc_for_update",
)


def _strict_accepted_step_from_trace(trace_payload, static, *, state_pre=None):
    from vmec_jax.discrete_adjoint import strict_update_accepted_step

    kwargs = {key: trace_payload[key] for key in _STRICT_ACCEPTED_STEP_TRACE_KEYS}
    kwargs["max_update_rms"] = trace_payload["max_update_rms_pre"]
    return strict_update_accepted_step(
        trace_payload["state_pre"] if state_pre is None else state_pre,
        static,
        **kwargs,
        enforce_edge=False,
    )


def _lcfs_boundary_moment_from_state(state, static):
    from vmec_jax._compat import jnp
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import free_boundary_boundary_geometry_jax

    geometry = free_boundary_boundary_geometry_jax(state, static)
    R = jnp.asarray(geometry["R"])
    Z = jnp.asarray(geometry["Z"])
    return jnp.mean((R - 1.0) * (R - 1.0) + Z * Z)


def _qs_total_from_state(state, static, indata, signgs):
    from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state

    qs = quasisymmetry_ratio_residual_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
        surfaces=[0.5],
        helicity_m=1,
        helicity_n=0,
        ntheta=7,
        nphi=8,
    )
    return qs["total"]


def _accepted_trace_rms_from_payload(payload, *, trace_key: str | None = None, nestor_key: str | None = None) -> float:
    values = []
    for trace in payload["traces"]:
        if trace.get("freeb_bsqvac_half") is None:
            continue
        if nestor_key is None:
            if trace_key is None or trace.get(trace_key) is None:
                continue
            value = trace[trace_key]
        else:
            nestor_trace = trace.get("freeb_nestor_trace")
            if not isinstance(nestor_trace, dict) or nestor_trace.get(nestor_key) is None:
                continue
            value = nestor_trace[nestor_key]
        values.append(float(np.sqrt(np.mean(np.square(np.asarray(value, dtype=float))))))
    return float(np.mean(values)) if values else 0.0


def _accepted_history_rms_from_replay(replay, history_key: str) -> object:
    from vmec_jax._compat import jnp

    history = jnp.asarray(replay["history"][history_key])
    accepted = jnp.asarray(replay["history"]["accepted"], dtype=history.dtype)
    active = jnp.asarray(replay["controls"]["has_active_freeb_replay"], dtype=history.dtype)
    weights = accepted * active
    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=weights.dtype))
    return jnp.sum(weights * history) / denom


def _assert_accepted_vacuum_scalar_fd(values: dict[str, float], *, require_positive_slope: bool) -> None:
    assert values["base"] > 0.0
    assert np.isfinite(float(values["plus"]))
    assert np.isfinite(float(values["minus"]))
    assert np.isfinite(float(values["central_fd_directional"]))
    if require_positive_slope:
        assert values["plus"] > values["minus"]
        assert values["central_fd_directional"] > 0.0
    else:
        assert abs(float(values["central_fd_directional"])) > 1.0e-12


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


def _assert_direct_coil_bnormal_fd_slope_stable(
    tmp_path: Path,
    *,
    input_name: str,
    variables: list[tuple[str, tuple[int, ...]]],
    current_step: float,
    dof_step: float,
    rtol: float,
) -> None:
    from examples.optimization.free_boundary_QS_coil_optimization import (
        apply_coil_variables,
        run_direct_free_boundary,
        summarize_run,
    )

    enable_x64(True)
    input_path = _write_tiny_direct_freeb_input(tmp_path / input_name)
    base_params = _circle_coil_params(current=3.0e7)

    def objective(x: float) -> float:
        params = apply_coil_variables(
            base_params,
            np.asarray([x], dtype=float),
            variables=variables,
            current_step=current_step,
            dof_step=dof_step,
        )
        run, wall_s = run_direct_free_boundary(input_path, params, vmec_max_iter=4, activate_fsq=1.0e99)
        summary = summarize_run(run, params, objective=np.nan, wall_s=wall_s, target_aspect=6.0, target_iota=0.4)
        assert summary["free_boundary_vacuum_stub"] is False
        assert summary["free_boundary_nestor_model"].startswith("vmec2000_like_dense_integral")
        assert summary["free_boundary_bnormal_rms"] > 0.0
        assert summary["free_boundary_bsqvac_rms"] > 0.0
        return float(summary["free_boundary_bnormal_rms"])

    slopes = np.asarray([(objective(eps) - objective(-eps)) / (2.0 * eps) for eps in (0.25, 0.125)], dtype=float)
    assert np.all(np.isfinite(slopes))
    assert np.min(np.abs(slopes)) > 1.0e-7
    np.testing.assert_allclose(slopes[0], slopes[1], rtol=rtol, atol=1.0e-12)


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

    base, _ = _direct_nestor_step(run, base_params)
    perturbed, _ = _direct_nestor_step(run, perturbed_params)

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

    full, runtime = _direct_nestor_step(run, base_params)
    reuse, _ = _direct_nestor_step(run, perturbed_params, ivac=2, ivacskip=1, iter_idx=2, runtime=runtime)

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
    scalar, _ = _direct_nestor_step(run, params)
    monkeypatch.setenv("VMEC_JAX_FREEB_NONSINGULAR_IP_CHUNK", "5")
    chunked, _ = _direct_nestor_step(run, params)

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
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

    params = _circle_coil_params(current=3.0e7)
    input_path = _write_tiny_direct_freeb_input(tmp_path / "input.direct_provider_forced_active")
    run = _run_forced_active_direct_solve(
        input_path,
        params,
        max_iter=4,
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

    params = _circle_coil_params(current=3.0e7, n_segments=32)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_provider_adjoint_trace_forcing",
        niter=2,
        mpol=3,
        ntheta=6,
    )
    init = _run_direct_initial_guess(input_path, params)
    result = _solve_direct_residual_iter(
        init,
        params,
        max_iter=2,
        adjoint_trace=True,
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


def test_full_adjoint_trace_records_raw_preconditioner_on_fused_payload_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full replay traces must include raw preconditioned forces on fused backends."""

    enable_x64(True)
    import vmec_jax.solve as solve_mod

    params = _circle_coil_params(current=3.0e7, n_segments=24)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_provider_fused_full_trace",
        niter=2,
        mpol=3,
        ntheta=6,
    )
    init = _run_direct_initial_guess(input_path, params)

    monkeypatch.setattr(solve_mod.jax, "default_backend", lambda: "gpu")
    result = _solve_direct_residual_iter(
        init,
        params,
        max_iter=2,
        use_scan=False,
        host_update_assembly=False,
        adjoint_trace=True,
        adjoint_trace_mode="full",
    )

    traces = result.diagnostics.get("adjoint_step_trace", [])
    assert traces
    for trace in traces:
        raw_precond = np.asarray(trace["frzl_rz_frcc"], dtype=float)
        update_precond = np.asarray(trace["frcc_u"], dtype=float)
        assert raw_precond.shape == update_precond.shape
        assert np.all(np.isfinite(raw_precond))
        assert np.linalg.norm(raw_precond) > 0.0


def test_direct_coil_trial_nestor_timing_records_solver_trial_calls(tmp_path: Path) -> None:
    """Solver-level trial scoring should record rejected NESTOR sample timings."""

    enable_x64(True)

    params = _circle_coil_params(current=3.0e7)
    input_path = _write_tiny_direct_freeb_input(tmp_path / "input.direct_trial_timing")
    init = _run_direct_initial_guess(input_path, params)
    result = _solve_direct_residual_iter(
        init,
        params,
        max_iter=4,
        use_scan=False,
    )

    trial_samples = np.asarray(result.diagnostics["freeb_nestor_trial_sample_time_history"], dtype=float)
    trial_failed = np.asarray(result.diagnostics["freeb_nestor_trial_failed_history"], dtype=int)
    assert trial_samples.size >= 1
    assert trial_failed.shape == trial_samples.shape
    assert np.all(trial_samples > 0.0)
    assert np.count_nonzero(trial_failed) == 0


def test_direct_coil_current_only_objective_fd_slope_is_stable(tmp_path: Path) -> None:
    """Central finite-difference slopes should be stable for a current-only direct-coil objective."""

    _assert_direct_coil_bnormal_fd_slope_stable(
        tmp_path,
        input_name="input.direct_current_fd_slope",
        variables=[("current", (0,))],
        current_step=0.02,
        dof_step=0.0,
        rtol=5.0e-6,
    )


def test_direct_coil_geometry_dof_accepted_state_fd_slope_is_stable(tmp_path: Path) -> None:
    """Boundary-normal vacuum response should vary smoothly with a coil geometry DOF."""

    _assert_direct_coil_bnormal_fd_slope_stable(
        tmp_path,
        input_name="input.direct_geometry_fd_slope",
        variables=[("fourier_dof", (0, 0, 2))],
        current_step=0.0,
        dof_step=1.0e-2,
        rtol=1.0e-4,
    )


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
        run = _run_forced_active_direct_solve(
            input_path,
            params,
            max_iter=2,
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


def _assert_native_rejected_slot_branch(
    complete_report: dict,
    *,
    expected_status: tuple[str, ...] = ("momentum", "momentum", "restart_bad_jacobian"),
    accept_mask: tuple[int, ...] = (1, 1, 0),
    done_mask: tuple[int, ...] = (0, 0, 1),
) -> None:
    branch = complete_report["branch_compatibility"]
    assert branch["same_branch"], branch
    for label in ("base", "plus", "minus"):
        fingerprint = branch[f"{label}_fingerprint"]
        assert fingerprint["step_status"] == expected_status
        np.testing.assert_array_equal(np.asarray(fingerprint["accept_mask"], dtype=int), accept_mask)
        np.testing.assert_array_equal(np.asarray(fingerprint["done_mask"], dtype=int), done_mask)


def _complete_report_base_values(complete_report: dict) -> dict[str, object]:
    return {
        key: values["base"]
        for key, values in complete_report["objective_values"].items()
    }


def _assert_native_rejected_branch_local_contract(
    branch_local: dict,
    *,
    expected_accept_mask: tuple[bool, ...] = (True, True, False),
) -> None:
    assert branch_local["uses_production_forward"] is True
    assert branch_local["differentiates_adaptive_controller"] is False
    assert branch_local["differentiates_run_free_boundary"] is False
    assert branch_local["differentiates_fixed_accepted_branch"] is True
    assert branch_local["controller_slot_summary"]["rejected_slots"] == 1
    assert branch_local["replay_branch_metadata"]["status_acceptance_source"] == "trace_step_status"
    np.testing.assert_array_equal(
        np.asarray(branch_local["replay_branch_metadata"]["status_masks"]["accept_mask"], dtype=bool),
        expected_accept_mask,
    )


def _native_rejected_adaptive_gate_report(
    complete_report: dict,
    scalars_report: dict,
    *,
    scalar_keys: tuple[str, ...],
):
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import direct_coil_adaptive_full_loop_same_branch_gate_report

    return direct_coil_adaptive_full_loop_same_branch_gate_report(
        complete_report,
        scalars_report,
        scalar_keys=scalar_keys,
        require_complete_loop_rejected_controller_slot=True,
        require_fixed_rejected_controller_slot=True,
        require_status_derived_rejected_controller_slot=True,
    )


def _assert_native_rejected_adaptive_gate(
    complete_report: dict,
    scalars_report: dict,
    *,
    scalar_keys: tuple[str, ...],
):
    adaptive_gate = _native_rejected_adaptive_gate_report(
        complete_report,
        scalars_report,
        scalar_keys=scalar_keys,
    )
    assert adaptive_gate["passed"], adaptive_gate
    assert adaptive_gate["fingerprint_gated"] is True
    assert adaptive_gate["same_branch"] is True
    assert adaptive_gate["complete_loop_rejected_controller_slot_present"] is True
    assert adaptive_gate["fixed_rejected_controller_slot_present"] is True
    assert adaptive_gate["status_derived_rejected_controller_slot_present"] is True
    assert adaptive_gate["differentiates_adaptive_controller"] is False
    assert adaptive_gate["differentiates_run_free_boundary"] is False


def _aspect_state_norm_qs_scalar_map(payload):
    from vmec_jax._compat import jnp
    from vmec_jax.state import pack_state
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    state = payload["result"].state
    packed = pack_state(state)
    return {
        "aspect": equilibrium_aspect_ratio_from_state(state=state, static=payload["init"].static),
        "state_norm": 0.5 * jnp.vdot(packed, packed),
        "qs_total": _qs_total_from_state(state, payload["init"].static, payload["init"].indata, payload["init"].signgs),
    }


def _aspect_state_norm_qs_replay_scalar_fns():
    from vmec_jax._compat import jnp
    from vmec_jax.state import pack_state
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    return {
        "aspect": lambda replay, payload: equilibrium_aspect_ratio_from_state(
            state=replay["state"],
            static=payload["init"].static,
        ),
        "state_norm": lambda replay, _payload: 0.5 * jnp.vdot(
            pack_state(replay["state"]),
            pack_state(replay["state"]),
        ),
        "qs_total": lambda replay, payload: _qs_total_from_state(
            replay["state"],
            payload["init"].static,
            payload["init"].indata,
            payload["init"].signgs,
        ),
    }


def _aspect_qs_boundary_scalar_map(payload):
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    state = payload["result"].state
    return {
        "aspect": equilibrium_aspect_ratio_from_state(state=state, static=payload["init"].static),
        "qs_total": _qs_total_from_state(state, payload["init"].static, payload["init"].indata, payload["init"].signgs),
        "lcfs_boundary_moment": _lcfs_boundary_moment_from_state(state, payload["init"].static),
    }


def _aspect_qs_boundary_replay_scalar_fns():
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    return {
        "aspect": lambda replay, payload: equilibrium_aspect_ratio_from_state(
            state=replay["state"],
            static=payload["init"].static,
        ),
        "qs_total": lambda replay, payload: _qs_total_from_state(
            replay["state"],
            payload["init"].static,
            payload["init"].indata,
            payload["init"].signgs,
        ),
        "lcfs_boundary_moment": lambda replay, payload: _lcfs_boundary_moment_from_state(
            replay["state"],
            payload["init"].static,
        ),
    }


def _native_rejected_slot_scalars_report(
    *,
    input_path: Path,
    base_params: CoilFieldParams,
    direction: CoilFieldParams,
    params_for,
    scalar_map,
    replay_scalar_fns: dict,
    scalar_keys: tuple[str, ...],
    rtol: dict,
    atol: dict,
    base_value_atol: dict,
    replay_kwargs_extra: dict | None = None,
    solve_kwargs_extra: dict | None = None,
) -> dict:
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_branch_local_scalars_report_from_complete_fd,
        direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax,
        direct_coil_same_branch_complete_solve_fd_report,
    )

    solve_kwargs = {
        "max_iter": 3,
        "step_size": 0.9,
        "ftol": 1.0e-12,
        "use_restart_triggers": True,
        "vmecpp_restart": True,
        "free_boundary_activate_fsq": 1.0e99,
    }
    if solve_kwargs_extra:
        solve_kwargs.update(solve_kwargs_extra)
    complete_report = direct_coil_same_branch_complete_solve_fd_report(
        input_path,
        base_params,
        params_for=params_for,
        objective_fn=scalar_map,
        eps=0.25,
        solve_kwargs=solve_kwargs,
        fingerprint_rtol=1.0e-6,
        fingerprint_atol=1.0e-9,
    )
    _assert_native_rejected_slot_branch(complete_report)
    replay_kwargs = {"use_stacked_step_controls": True, "use_accepted_only_fast_path": False}
    if replay_kwargs_extra:
        replay_kwargs.update(replay_kwargs_extra)
    branch_local = direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
        params=base_params,
        direction_params=direction,
        complete_payload=complete_report["base"],
        scalar_keys=scalar_keys,
        production_values=_complete_report_base_values(complete_report),
        replay_payload={"init": complete_report["base"]["init"]},
        scalar_fn=scalar_map,
        replay_scalar_fns=replay_scalar_fns,
        replay_kwargs=replay_kwargs,
        include_payload=False,
        include_replay_graph_metadata=False,
    )
    _assert_native_rejected_branch_local_contract(branch_local)
    scalars_report = direct_coil_branch_local_scalars_report_from_complete_fd(
        complete_report,
        branch_local,
        scalar_keys=scalar_keys,
        rtol=rtol,
        atol=atol,
        base_value_atol=base_value_atol,
    )
    assert scalars_report["passed"], scalars_report
    _assert_native_rejected_adaptive_gate(complete_report, scalars_report, scalar_keys=scalar_keys)
    return {"complete_report": complete_report, "branch_local": branch_local, "scalars_report": scalars_report}


def _assert_same_branch_physical_and_adaptive_scalar_gates(
    complete_report: dict,
    scalars_report: dict,
    *,
    scalar_keys: tuple[str, ...],
    base_fingerprint: dict,
):
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_adaptive_full_loop_same_branch_gate_report,
        direct_coil_same_branch_physical_scalar_gate_report,
    )

    physical_scalar_gate = direct_coil_same_branch_physical_scalar_gate_report(
        complete_report, scalars_report, scalar_keys=scalar_keys
    )
    assert physical_scalar_gate["passed"], physical_scalar_gate
    assert physical_scalar_gate["scalar_keys"] == scalar_keys
    assert physical_scalar_gate["controller_slot_summary"]["accepted_slots"] >= 1
    assert physical_scalar_gate["controller_slot_summary"]["rejected_slots"] == 0
    assert physical_scalar_gate["replay_gate"]["passed"] is True
    for key, expected in (
        ("contract", "same-branch complete-solve physical-scalar AD-vs-FD gate"),
        ("same_branch", True),
        ("differentiates_adaptive_controller", False),
    ):
        assert physical_scalar_gate[key] == expected
    adaptive_full_loop_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        complete_report, scalars_report, scalar_keys=scalar_keys
    )
    assert adaptive_full_loop_gate["passed"], adaptive_full_loop_gate
    assert set(adaptive_full_loop_gate["branch_fingerprints"]) == {"base", "plus", "minus"}
    assert set(adaptive_full_loop_gate["residual_branch_fingerprints"]) == {"base", "plus", "minus"}
    assert adaptive_full_loop_gate["branch_fingerprints"]["base"]["n_steps"] == base_fingerprint["n_steps"]
    assert adaptive_full_loop_gate["branch_fingerprints"]["base"]["n_freeb_steps"] == base_fingerprint["n_freeb_steps"]
    assert adaptive_full_loop_gate["controller_slot_summary"]["accepted_slots"] >= 1
    assert adaptive_full_loop_gate["controller_slot_summary"]["rejected_slots"] == 0
    for key, expected in (
        ("contract", "same-branch adaptive full-loop seam report"),
        ("ad_vs_fd_gate", "complete-loop central FD vs branch-local stacked replay custom VJP"),
        ("adaptive_loop_scope", "fingerprint-gated branch-local accepted/rejected replay slots"),
        ("differentiates_adaptive_controller", False),
        ("differentiates_run_free_boundary", False),
        ("fingerprint_gated", True),
        ("same_branch", True),
        ("same_accepted_trace_branch", True),
        ("same_residual_branch", True),
        ("same_full_loop_branch_fingerprint", True),
        ("same_residual_branch_fingerprint", True),
        ("same_stacked_step_policy_branch", True),
        ("used_stacked_step_controls", True),
        ("requires_fixed_rejected_controller_slot", False),
        ("fixed_rejected_controller_slot_present", False),
    ):
        assert adaptive_full_loop_gate[key] == expected
    for report_fn in (
        direct_coil_adaptive_full_loop_same_branch_gate_report,
        direct_coil_same_branch_physical_scalar_gate_report,
    ):
        json.dumps(report_fn(complete_report, scalars_report, scalar_keys=scalar_keys, json_safe=True), allow_nan=False)


def _assert_complete_report_replay_contract(complete_report: dict, replay_gate_report_fn) -> dict[str, dict]:
    branch = complete_report["branch_compatibility"]
    base_fingerprint = branch["base_fingerprint"]
    plus_fingerprint = branch["plus_fingerprint"]
    minus_fingerprint = branch["minus_fingerprint"]
    assert branch["same_branch"] is True
    assert branch["same_accepted_trace_branch"] is True
    assert branch["same_residual_branch"] is True
    assert branch["plus"]["compatible"], branch["plus"]["changed_fields"]
    assert branch["minus"]["compatible"], branch["minus"]["changed_fields"]
    assert base_fingerprint["n_steps"] == plus_fingerprint["n_steps"] == minus_fingerprint["n_steps"]
    assert branch["base_residual_fingerprint"] == branch["plus_residual_fingerprint"] == branch["minus_residual_fingerprint"]
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
    replay_gate = replay_gate_report_fn(complete_report)
    assert replay_gate["passed"], replay_gate
    assert replay_gate["contract"] == "same-branch accepted-trace replay gate"
    assert replay_gate["same_branch"] is True
    assert replay_gate["differentiates_adaptive_controller"] is False
    json.dumps(replay_gate_report_fn(complete_report, json_safe=True), allow_nan=False)
    return expected_fingerprints


def _state_norm_from_state(state) -> float:
    packed = np.asarray(pack_state(state), dtype=float)
    return float(0.5 * np.vdot(packed, packed))


def _axis_R_from_state(state, static):
    from vmec_jax._compat import jnp

    idx = np.where((np.asarray(static.modes.m) == 0) & (np.asarray(static.modes.n) == 0))[0]
    idx00 = int(idx[0]) if idx.size else 0
    return jnp.asarray(state.Rcos)[0, idx00]


def _same_branch_complete_scalar_values(payload: dict, *, include_qs_total: bool = False) -> dict:
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    state = payload["result"].state
    init = payload["init"]
    values = {
        "objective": _state_norm_from_state(state),
        "aspect": float(np.asarray(equilibrium_aspect_ratio_from_state(state=state, static=init.static))),
        "lcfs_boundary_moment": float(np.asarray(_lcfs_boundary_moment_from_state(state, init.static))),
        "axis_R": float(np.asarray(_axis_R_from_state(state, init.static))),
        "accepted_bnormal_rms": _accepted_trace_rms_from_payload(payload, nestor_key="bnormal"),
        "accepted_bsqvac_rms": _accepted_trace_rms_from_payload(payload, trace_key="freeb_bsqvac_half"),
    }
    if include_qs_total:
        values["qs_total"] = float(np.asarray(_qs_total_from_state(state, init.static, init.indata, init.signgs)))
    return values


def _same_branch_replay_scalar_config(
    *,
    include_aspect: bool,
    include_axis_R: bool,
    include_boundary_moment: bool,
    include_qs_total: bool,
    include_accepted_bsqvac_rms: bool,
    include_accepted_bnormal_rms: bool,
) -> tuple[dict, dict[str, float], dict[str, float]]:
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    replay_scalar_fns: dict = {}
    rtol_by_key: dict[str, float] = {}
    atol_by_key: dict[str, float] = {}
    if include_aspect:
        replay_scalar_fns["aspect"] = lambda replay, payload: equilibrium_aspect_ratio_from_state(
            state=replay["state"],
            static=payload["init"].static,
        )
        rtol_by_key["aspect"] = 5.0e-3
        atol_by_key["aspect"] = 5.0e-8
    if include_axis_R:
        replay_scalar_fns["axis_R"] = lambda replay, payload: _axis_R_from_state(
            replay["state"],
            payload["init"].static,
        )
        rtol_by_key["axis_R"] = 5.0e-3
        atol_by_key["axis_R"] = 5.0e-8
    if include_boundary_moment:
        replay_scalar_fns["lcfs_boundary_moment"] = lambda replay, payload: _lcfs_boundary_moment_from_state(
            replay["state"],
            payload["init"].static,
        )
        rtol_by_key["lcfs_boundary_moment"] = 5.0e-3
        atol_by_key["lcfs_boundary_moment"] = 5.0e-8
    if include_qs_total:
        replay_scalar_fns["qs_total"] = lambda replay, payload: _qs_total_from_state(
            replay["state"],
            payload["init"].static,
            payload["init"].indata,
            payload["init"].signgs,
        )
        rtol_by_key["qs_total"] = 2.0e-2
        atol_by_key["qs_total"] = 1.0e-8
    if include_accepted_bsqvac_rms:
        replay_scalar_fns["accepted_bsqvac_rms"] = lambda replay, _payload: _accepted_history_rms_from_replay(
            replay, "bsqvac_rms"
        )
        rtol_by_key["accepted_bsqvac_rms"] = 1.0e-2
        atol_by_key["accepted_bsqvac_rms"] = 1.0e-8
    if include_accepted_bnormal_rms:
        replay_scalar_fns["accepted_bnormal_rms"] = lambda replay, _payload: _accepted_history_rms_from_replay(
            replay, "bnormal_rms"
        )
        rtol_by_key["accepted_bnormal_rms"] = 1.0e-2
        atol_by_key["accepted_bnormal_rms"] = 1.0e-8
    return replay_scalar_fns, rtol_by_key, atol_by_key


def _assert_scalar_report_payload_ok(scalar_report: dict, *, max_base_delta: float = 2.0e-3) -> dict:
    assert scalar_report["passed"], scalar_report
    assert scalar_report["same_branch"] is True
    assert scalar_report["replay_gate"]["passed"] is True
    assert scalar_report["base_abs_delta"] < max_base_delta
    return scalar_report


def _assert_same_branch_scalar_report_ok(scalars_report: dict, scalar_key: str, *, max_base_delta: float = 2.0e-3) -> dict:
    return _assert_scalar_report_payload_ok(
        scalars_report["scalar_reports"][scalar_key],
        max_base_delta=max_base_delta,
    )


def _assert_single_controller_scalar_report_ok(
    *,
    complete_report: dict,
    base_params: CoilFieldParams,
    direction: CoilFieldParams,
    scalar_key: str,
    replay_scalar_fn,
    eps: float,
    rtol: float,
    atol: float,
) -> dict:
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import direct_coil_same_branch_controller_scalar_custom_vjp_report

    scalar_report = direct_coil_same_branch_controller_scalar_custom_vjp_report(
        complete_report,
        base_params,
        direction,
        scalar_key=scalar_key,
        replay_scalar_fn=replay_scalar_fn,
        eps=eps,
        rtol=rtol,
        atol=atol,
        compute_frozen_fd=False,
    )
    return _assert_scalar_report_payload_ok(scalar_report)


def _padded_rejected_trace(base_traces: tuple | list) -> tuple:
    rejected_trace = deepcopy(base_traces[-1])
    rejected_trace["step_status"] = "rejected"
    return tuple(base_traces) + (rejected_trace,)


def _assert_rejected_slot_metadata(metadata: dict, *, base_traces: tuple | list) -> None:
    expected_active_freeb_steps = sum(
        1
        for trace in base_traces
        if trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
    )
    assert int(np.count_nonzero(np.asarray(metadata["rejected_mask"], dtype=bool))) == 1
    assert int(np.count_nonzero(np.asarray(metadata["accepted_mask"], dtype=bool))) == len(base_traces)
    assert int(metadata["n_steps"]) == len(base_traces) + 1
    assert int(metadata["n_free_boundary_replay_steps"]) == expected_active_freeb_steps
    assert metadata["status_acceptance_source"] == "trace_step_status"
    assert tuple(metadata["status_masks"]["step_status"])[-1] == "rejected"
    assert bool(np.asarray(metadata["status_masks"]["accept_mask"], dtype=bool)[-1]) is False


def _assert_fixed_rejected_slot_gate(
    *,
    complete_report: dict,
    rejected_scalars_report: dict,
    scalar_keys: tuple[str, ...],
) -> dict:
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import direct_coil_adaptive_full_loop_same_branch_gate_report

    rejected_slot_gate = direct_coil_adaptive_full_loop_same_branch_gate_report(
        complete_report,
        rejected_scalars_report,
        scalar_keys=scalar_keys,
        require_fixed_rejected_controller_slot=True,
        require_status_derived_rejected_controller_slot=True,
    )
    assert rejected_slot_gate["passed"], rejected_slot_gate
    assert rejected_slot_gate["same_branch"] is True
    assert rejected_slot_gate["differentiates_adaptive_controller"] is False
    assert rejected_slot_gate["differentiates_run_free_boundary"] is False
    assert rejected_slot_gate["same_stacked_step_policy_branch"] is True
    assert rejected_slot_gate["requires_fixed_rejected_controller_slot"] is True
    assert rejected_slot_gate["requires_status_derived_rejected_controller_slot"] is True
    assert rejected_slot_gate["fixed_rejected_controller_slot_present"] is True
    assert rejected_slot_gate["status_derived_rejected_controller_slot_present"] is True
    assert rejected_slot_gate["status_acceptance_source"] == "trace_step_status"
    assert rejected_slot_gate["fixed_rejected_controller_slots"] == 1
    assert rejected_slot_gate["controller_slot_summary"]["accepted_slots"] >= 1
    assert rejected_slot_gate["controller_slot_summary"]["rejected_slots"] == 1
    assert rejected_slot_gate["controller_slot_summary"]["fixed_rejected_controller_slot_present"] is True
    assert rejected_slot_gate["controller_slot_fingerprint"]["rejected_mask"][-1] is True
    assert rejected_slot_gate["controller_slot_fingerprint"]["status_acceptance_source"] == "trace_step_status"
    assert rejected_slot_gate["controller_slot_fingerprint"]["summary"]["rejected_slots"] == 1
    assert rejected_slot_gate["used_stacked_step_controls"] is True
    assert rejected_slot_gate["replay_option_flags"]["use_accepted_only_fast_path"] is False
    json.dumps(
        direct_coil_adaptive_full_loop_same_branch_gate_report(
            complete_report,
            rejected_scalars_report,
            scalar_keys=scalar_keys,
            require_fixed_rejected_controller_slot=True,
            require_status_derived_rejected_controller_slot=True,
            json_safe=True,
        ),
        allow_nan=False,
    )
    return rejected_slot_gate


def _production_same_branch_scalar_values(payload: dict) -> dict:
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    state = payload["result"].state
    init = payload["init"]
    return {
        "aspect": equilibrium_aspect_ratio_from_state(state=state, static=init.static),
        "axis_R": _axis_R_from_state(state, init.static),
        "lcfs_boundary_moment": _lcfs_boundary_moment_from_state(state, init.static),
        "qs_total": _qs_total_from_state(state, init.static, init.indata, init.signgs),
        "accepted_bnormal_rms": _accepted_trace_rms_from_payload(payload, nestor_key="bnormal"),
        "accepted_bsqvac_rms": _accepted_trace_rms_from_payload(payload, trace_key="freeb_bsqvac_half"),
    }


def _production_scalar_rtol(scalar_keys: tuple[str, ...]) -> dict[str, float]:
    return {
        key: (
            2.0e-2
            if key == "qs_total"
            else 1.0e-2
            if key in {"accepted_bnormal_rms", "accepted_bsqvac_rms"}
            else 5.0e-3
        )
        for key in scalar_keys
    }


def _select_finite_direct_coil_replay_trace(init, base_params: CoilFieldParams, active_traces: list[dict]) -> SimpleNamespace:
    from vmec_jax._compat import jnp
    from vmec_jax.free_boundary import _sample_external_boundary_arrays
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_boundary_bsqvac_jax,
        direct_coil_boundary_replay_context,
    )

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
            nestor_trace = candidate_trace.get("freeb_nestor_trace")
            if not isinstance(nestor_trace, dict):
                continue

            def replay_from_coils(
                params: CoilFieldParams,
                *,
                sample=sample,
                context=context,
                nestor_trace=nestor_trace,
                include_analytic=bool(include_analytic),
            ):
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
                    br_add=jnp.asarray(nestor_trace["br_axis"]),
                    bp_add=jnp.asarray(nestor_trace["bp_axis"]),
                    bz_add=jnp.asarray(nestor_trace["bz_axis"]),
                    wint=jnp.asarray(context["wint"]),
                    include_analytic=include_analytic,
                )

            replay0 = replay_from_coils(base_params)
            bsqvac0 = replay0["bsqvac"]
            bsqvac0_np = np.asarray(bsqvac0, dtype=float)
            if np.all(np.isfinite(bsqvac0_np)) and float(np.linalg.norm(bsqvac0_np)) > 0.0:
                return SimpleNamespace(
                    index=candidate_idx,
                    trace=candidate_trace,
                    nestor_trace=nestor_trace,
                    basis=context["basis"],
                    replay_from_coils=replay_from_coils,
                    replay0=replay0,
                    bsqvac0=bsqvac0,
                    analytic_replay=bool(include_analytic),
                )
    raise AssertionError("No finite accepted direct-coil replay trace found")


def _assert_selected_direct_coil_replay_matches_trace(selected: SimpleNamespace) -> None:
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import vacuum_boundary_fields_from_mode_coeffs_jax

    trace = selected.trace
    nestor_trace = selected.nestor_trace
    bsqvac0 = selected.bsqvac0
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
        basis=selected.basis,
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
    if selected.analytic_replay:
        np.testing.assert_allclose(
            np.asarray(selected.replay0["mode_solution"]["mode_coeffs"]),
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


def _assert_trace_force_channels_and_step_replay(trace: dict, static):
    from vmec_jax.discrete_adjoint import preconditioned_force_channels_from_rz_output
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import accepted_trace_effective_state_pre
    from vmec_jax.state import pack_state
    from vmec_jax.kernels.tomnsp import TomnspsRZL

    traced_rz_force = TomnspsRZL(
        **{
            name: trace[f"frzl_rz_{name}"]
            for name in (
                "frcc", "frss", "fzsc", "fzcs", "flsc", "flcs",
                "frsc", "frcs", "fzcc", "fzss", "flcc", "flss",
            )
        }
    )
    traced_force = preconditioned_force_channels_from_rz_output(
        frzl_rz=traced_rz_force,
        lam_prec=trace["lam_prec"],
        w_mode_mn=trace["w_mode_mn"],
        lambda_update_scale=trace["lambda_update_scale"],
    )
    for key in ("frcc_u", "frss_u", "fzsc_u", "fzcs_u", "flsc_u", "flcs_u"):
        np.testing.assert_allclose(np.asarray(traced_force[key]), np.asarray(trace[key]), rtol=0.0, atol=0.0)

    effective_state_pre = accepted_trace_effective_state_pre(trace)
    exact_step = _strict_accepted_step_from_trace(trace, static, state_pre=effective_state_pre)
    np.testing.assert_allclose(
        np.asarray(pack_state(exact_step["state_post"])),
        np.asarray(pack_state(trace["state_post"])),
        rtol=0.0,
        atol=0.0,
    )
    return effective_state_pre


def _assert_strict_update_coil_directional_derivative_matches_fd(
    *,
    init,
    trace: dict,
    effective_state_pre,
    base_params: CoilFieldParams,
    direction: CoilFieldParams,
    bsqvac_from_coils,
) -> None:
    from vmec_jax._compat import jnp
    from vmec_jax.discrete_adjoint import strict_update_one_step_from_trace
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import pytree_directional_derivative_check_jax
    from vmec_jax.state import pack_state

    def objective(params: CoilFieldParams):
        out = strict_update_one_step_from_trace(
            effective_state_pre,
            init.static,
            trace,
            freeb_bsqvac_half=bsqvac_from_coils(params),
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


def _assert_direct_coil_same_branch_custom_vjp_matches_complete_fd(
    *,
    input_path: Path,
    base_params: CoilFieldParams,
    direction: CoilFieldParams,
    params_for,
    check_controller: bool = True,
    check_segmented_controller: bool = True,
    check_aspect_scalar: bool = True,
    check_axis_R_scalar: bool = False,
    check_boundary_moment_scalar: bool = False,
    check_qs_total_scalar: bool = False,
    check_accepted_bnormal_rms_scalar: bool = False,
    check_accepted_bsqvac_rms_scalar: bool = False,
    require_positive_accepted_vacuum_scalar_slope: bool = True,
    check_production_branch_local_scalar: bool = False,
    check_fixed_rejected_controller_mask_gate: bool = False,
) -> None:
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_accepted_trace_controller_custom_vjp_objective_jax,
        direct_coil_branch_local_scalars_report_from_complete_fd,
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax,
        direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax,
        direct_coil_same_branch_controller_scalars_custom_vjp_report,
        direct_coil_same_branch_complete_solve_fd_report,
        direct_coil_same_branch_replay_gate_report,
        direct_coil_fixed_trace_custom_vjp_objective_jax,
    )
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    eps = 1.0e-4
    complete_report = direct_coil_same_branch_complete_solve_fd_report(
        input_path,
        base_params,
        params_for=params_for,
        objective_fn=lambda payload: _same_branch_complete_scalar_values(
            payload,
            include_qs_total=check_qs_total_scalar,
        ),
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
    expected_fingerprints = _assert_complete_report_replay_contract(
        complete_report,
        direct_coil_same_branch_replay_gate_report,
    )
    base_fingerprint = expected_fingerprints["base"]

    assert complete_report["primary_objective"] == "objective"
    expected_objective_value_keys = {
        "objective",
        "aspect",
        "axis_R",
        "lcfs_boundary_moment",
        "accepted_bnormal_rms",
        "accepted_bsqvac_rms",
    }
    if check_qs_total_scalar:
        expected_objective_value_keys.add("qs_total")
    assert set(complete_report["objective_values"]) == expected_objective_value_keys
    complete_fd = float(complete_report["values"]["central_fd_directional"])

    def directional_grad(objective):
        grad = jax.grad(objective)(base_params)
        return sum(
            jnp.vdot(grad_leaf, direction_leaf)
            for grad_leaf, direction_leaf in zip(
                jax.tree_util.tree_leaves(grad),
                jax.tree_util.tree_leaves(direction),
                strict=True,
            )
        )

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

    exact = directional_grad(custom_objective)
    base_complete = _state_norm_from_state(base_result.state)
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

        controller_exact = directional_grad(controller_custom_objective)
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

        segmented_controller_exact = directional_grad(segmented_controller_custom_objective)
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
        replay_scalar_fns, rtol_by_key, atol_by_key = _same_branch_replay_scalar_config(
            include_aspect=True,
            include_axis_R=check_axis_R_scalar,
            include_boundary_moment=check_boundary_moment_scalar,
            include_qs_total=check_qs_total_scalar,
            include_accepted_bsqvac_rms=check_accepted_bsqvac_rms_scalar,
            include_accepted_bnormal_rms=check_accepted_bnormal_rms_scalar,
        )
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
        assert scalars_report["replay_option_flags"]["use_accepted_only_fast_path"] is True
        _assert_same_branch_physical_and_adaptive_scalar_gates(
            complete_report,
            scalars_report,
            scalar_keys=tuple(replay_scalar_fns),
            base_fingerprint=base_fingerprint,
        )
        if check_fixed_rejected_controller_mask_gate:
            padded_traces = _padded_rejected_trace(base_traces)
            rejected_slot_scalars_report = direct_coil_same_branch_controller_scalars_custom_vjp_report(
                complete_report,
                base_params,
                direction,
                replay_scalar_fns=replay_scalar_fns,
                replay_kwargs={
                    "traces": padded_traces,
                    "use_stacked_step_controls": True,
                    "use_accepted_only_fast_path": False,
                },
                eps=eps,
                rtol=rtol_by_key,
                atol=atol_by_key,
                compute_frozen_fd=False,
            )
            assert rejected_slot_scalars_report["passed"], rejected_slot_scalars_report
            assert rejected_slot_scalars_report["replay_option_flags"]["use_stacked_step_controls"] is True
            assert rejected_slot_scalars_report["replay_option_flags"]["use_accepted_only_fast_path"] is False
            _assert_rejected_slot_metadata(rejected_slot_scalars_report["replay_branch_metadata"], base_traces=base_traces)
            rejected_slot_gate = _assert_fixed_rejected_slot_gate(
                complete_report=complete_report,
                rejected_scalars_report=rejected_slot_scalars_report,
                scalar_keys=tuple(replay_scalar_fns),
            )
            assert rejected_slot_gate["controller_slot_summary"]["accepted_slots"] == len(base_traces)
        aspect_report = _assert_same_branch_scalar_report_ok(scalars_report, "aspect")
        np.testing.assert_allclose(
            aspect_report["complete_fd_directional"],
            complete_aspect_fd,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        if check_production_branch_local_scalar:
            complete_base_values = _complete_report_base_values(complete_report)
            production_branch_local = direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
                params=base_params,
                complete_payload=complete_report["base"],
                scalar_key="aspect",
                production_values={"aspect": complete_base_values["aspect"]},
                replay_payload={"init": complete_report["base"]["init"]},
                scalar_fn=lambda payload: {
                    "aspect": aspect_objective_from_state(payload["result"].state),
                },
                replay_scalar_fn=lambda replay, payload: equilibrium_aspect_ratio_from_state(
                    state=replay["state"],
                    static=payload["init"].static,
                ),
                replay_kwargs={"use_stacked_step_controls": True},
                include_payload=False,
            )
            production_branch_exact = sum(
                jnp.vdot(grad_leaf, direction_leaf)
                for grad_leaf, direction_leaf in zip(
                    jax.tree_util.tree_leaves(production_branch_local["grad"]),
                    jax.tree_util.tree_leaves(direction),
                    strict=True,
                )
            )
            _assert_branch_local_replay_contract(production_branch_local)
            _assert_nonnegative_timings(
                production_branch_local,
                "production_scalar_eval_wall_s",
                "replay_value_and_grad_dispatch_s",
                "replay_value_and_grad_ready_s",
                "replay_value_and_grad_wall_s",
                "replay_graph_metadata_wall_s",
                "total_wall_s",
            )
            assert production_branch_local["base_abs_delta"] < 2.0e-3
            np.testing.assert_allclose(
                production_branch_exact,
                complete_aspect_fd,
                rtol=5.0e-3,
                atol=5.0e-8,
            )

            vector_scalar_keys = tuple(replay_scalar_fns)
            production_branch_local_scalars = (
                direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
                    params=base_params,
                    direction_params=direction,
                    complete_payload=complete_report["base"],
                    scalar_keys=vector_scalar_keys,
                    production_values={key: complete_base_values[key] for key in vector_scalar_keys},
                    replay_payload={"init": complete_report["base"]["init"]},
                    scalar_fn=_production_same_branch_scalar_values,
                    replay_scalar_fns=replay_scalar_fns,
                    replay_kwargs={"use_stacked_step_controls": True},
                    include_payload=False,
                )
            )
            _assert_branch_local_replay_contract(production_branch_local_scalars)
            assert production_branch_local_scalars["derivative_mode"] == "directional_jvp"
            assert production_branch_local_scalars["scalar_keys"] == vector_scalar_keys
            _assert_nonnegative_timings(
                production_branch_local_scalars,
                "production_scalar_eval_wall_s",
                "replay_jvp_wall_s",
                "replay_graph_metadata_wall_s",
                "jacobian_stack_ready_s",
                "total_wall_s",
            )
            assert production_branch_local_scalars["timings"]["replay_vjp_wall_s"] == 0.0
            assert production_branch_local_scalars["timings"]["replay_pullbacks_wall_s"] == 0.0
            assert production_branch_local_scalars["base_abs_delta"]["aspect"] < 2.0e-3
            assert production_branch_local_scalars["max_base_abs_delta"] < 2.0e-3
            np.testing.assert_allclose(
                production_branch_local_scalars["values"]["aspect"],
                production_branch_local["value"],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                production_branch_local_scalars["replay_value_map"]["aspect"],
                production_branch_local["replay_value"],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            for key in vector_scalar_keys:
                production_branch_vector_exact = production_branch_local_scalars["directional_derivatives"][key]
                complete_directional = complete_report["objective_values"][key]["central_fd_directional"]
                np.testing.assert_allclose(
                    production_branch_vector_exact,
                    complete_directional,
                    rtol=_production_scalar_rtol(vector_scalar_keys)[key],
                    atol=5.0e-8,
                )
            production_rtol = _production_scalar_rtol(vector_scalar_keys)
            production_scalars_report = direct_coil_branch_local_scalars_report_from_complete_fd(
                complete_report,
                production_branch_local_scalars,
                scalar_keys=vector_scalar_keys,
                rtol=production_rtol,
                atol={key: 5.0e-8 for key in vector_scalar_keys},
                base_value_atol={key: 2.0e-3 for key in vector_scalar_keys},
            )
            assert production_scalars_report["passed"], production_scalars_report
            assert production_scalars_report["uses_production_forward"] is True
            assert production_scalars_report["differentiates_adaptive_controller"] is False
            assert production_scalars_report["differentiates_run_free_boundary"] is False
            assert production_scalars_report["differentiates_fixed_accepted_branch"] is True
            _assert_same_branch_physical_and_adaptive_scalar_gates(
                complete_report,
                production_scalars_report,
                scalar_keys=vector_scalar_keys,
                base_fingerprint=base_fingerprint,
            )
            for key in vector_scalar_keys:
                scalar_report = production_scalars_report["scalar_reports"][key]
                assert scalar_report["passed"], scalar_report
                assert np.isfinite(scalar_report["exact_directional"])
                assert np.isfinite(scalar_report["complete_fd_directional"])
                assert scalar_report["base_abs_delta"] < 2.0e-3
            assert direct_coil_branch_local_scalars_report_from_complete_fd(
                complete_report,
                production_branch_local_scalars,
                scalar_keys=vector_scalar_keys,
                rtol=production_rtol,
                atol={key: 5.0e-8 for key in vector_scalar_keys},
                base_value_atol={key: 2.0e-3 for key in vector_scalar_keys},
                json_safe=True,
            )["passed"]
            if check_fixed_rejected_controller_mask_gate:
                padded_traces = _padded_rejected_trace(base_traces)
                production_rejected_scalars = (
                    direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
                        params=base_params,
                        direction_params=direction,
                        complete_payload=complete_report["base"],
                        scalar_keys=vector_scalar_keys,
                        production_values={key: complete_base_values[key] for key in vector_scalar_keys},
                        replay_payload={"init": complete_report["base"]["init"]},
                        scalar_fn=_production_same_branch_scalar_values,
                        replay_scalar_fns=replay_scalar_fns,
                        replay_kwargs={
                            "traces": padded_traces,
                            "use_stacked_step_controls": True,
                            "use_accepted_only_fast_path": False,
                        },
                        include_payload=False,
                        include_replay_graph_metadata=False,
                    )
                )
                _assert_branch_local_replay_contract(
                    production_rejected_scalars,
                    rejected_slots=1,
                    graph_metadata=False,
                )
                _assert_rejected_slot_metadata(production_rejected_scalars["replay_branch_metadata"], base_traces=base_traces)
                production_rejected_report = direct_coil_branch_local_scalars_report_from_complete_fd(
                    complete_report,
                    production_rejected_scalars,
                    scalar_keys=vector_scalar_keys,
                    rtol=production_rtol,
                    atol={key: 5.0e-8 for key in vector_scalar_keys},
                    base_value_atol={key: 2.0e-3 for key in vector_scalar_keys},
                )
                assert production_rejected_report["passed"], production_rejected_report
                production_rejected_gate = _assert_fixed_rejected_slot_gate(
                    complete_report=complete_report,
                    rejected_scalars_report=production_rejected_report,
                    scalar_keys=vector_scalar_keys,
                )
                assert production_rejected_gate["fingerprint_gated"] is True
        if check_qs_total_scalar:
            _assert_same_branch_scalar_report_ok(scalars_report, "qs_total")
        if check_axis_R_scalar:
            _assert_same_branch_scalar_report_ok(scalars_report, "axis_R")
        if check_boundary_moment_scalar:
            _assert_same_branch_scalar_report_ok(scalars_report, "lcfs_boundary_moment")
        if check_accepted_bsqvac_rms_scalar:
            bsqvac_values = complete_report["objective_values"]["accepted_bsqvac_rms"]
            _assert_accepted_vacuum_scalar_fd(bsqvac_values, require_positive_slope=require_positive_accepted_vacuum_scalar_slope)
            _assert_same_branch_scalar_report_ok(scalars_report, "accepted_bsqvac_rms")
        if check_accepted_bnormal_rms_scalar:
            bnormal_values = complete_report["objective_values"]["accepted_bnormal_rms"]
            _assert_accepted_vacuum_scalar_fd(bnormal_values, require_positive_slope=require_positive_accepted_vacuum_scalar_slope)
            _assert_same_branch_scalar_report_ok(scalars_report, "accepted_bnormal_rms")
    elif check_boundary_moment_scalar:
        _assert_single_controller_scalar_report_ok(
            complete_report=complete_report,
            base_params=base_params,
            direction=direction,
            scalar_key="lcfs_boundary_moment",
            replay_scalar_fn=lambda replay, payload: _lcfs_boundary_moment_from_state(
                replay["state"],
                payload["init"].static,
            ),
            eps=eps,
            rtol=5.0e-3,
            atol=5.0e-8,
        )
    elif check_accepted_bsqvac_rms_scalar:
        bsqvac_values = complete_report["objective_values"]["accepted_bsqvac_rms"]
        _assert_accepted_vacuum_scalar_fd(bsqvac_values, require_positive_slope=require_positive_accepted_vacuum_scalar_slope)
        _assert_single_controller_scalar_report_ok(
            complete_report=complete_report,
            base_params=base_params,
            direction=direction,
            scalar_key="accepted_bsqvac_rms",
            replay_scalar_fn=lambda replay, _payload: _accepted_history_rms_from_replay(replay, "bsqvac_rms"),
            eps=eps,
            rtol=1.0e-2,
            atol=1.0e-8,
        )
    elif check_accepted_bnormal_rms_scalar:
        bnormal_values = complete_report["objective_values"]["accepted_bnormal_rms"]
        _assert_accepted_vacuum_scalar_fd(bnormal_values, require_positive_slope=require_positive_accepted_vacuum_scalar_slope)
        _assert_single_controller_scalar_report_ok(
            complete_report=complete_report,
            base_params=base_params,
            direction=direction,
            scalar_key="accepted_bnormal_rms",
            replay_scalar_fn=lambda replay, _payload: _accepted_history_rms_from_replay(replay, "bnormal_rms"),
            eps=eps,
            rtol=1.0e-2,
            atol=1.0e-8,
        )


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
        check_axis_R_scalar=True,
        check_boundary_moment_scalar=False,
        check_qs_total_scalar=True,
        check_accepted_bnormal_rms_scalar=True,
        check_accepted_bsqvac_rms_scalar=True,
        check_production_branch_local_scalar=True,
        check_fixed_rejected_controller_mask_gate=True,
    )


@pytest.mark.py311_coverage_only
def test_direct_coil_native_rejected_slot_same_branch_jvp_matches_complete_solve_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native restart/rejected slots can be validated under an unchanged branch."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    _set_same_branch_custom_vjp_env(monkeypatch)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_native_rejected_slot",
        lasym=False,
        niter=3,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e8, n_segments=24)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    direction = base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(base_dofs),
        base_currents=base_currents * 0.002,
    )

    def params_for(scale: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_curve_dofs=base_dofs,
            base_currents=base_currents * (1.0 + 0.002 * float(scale)),
        )

    scalar_keys = ("aspect", "state_norm", "qs_total")
    reports = _native_rejected_slot_scalars_report(
        input_path=input_path,
        base_params=base_params,
        direction=direction,
        params_for=params_for,
        scalar_map=_aspect_state_norm_qs_scalar_map,
        replay_scalar_fns=_aspect_state_norm_qs_replay_scalar_fns(),
        scalar_keys=scalar_keys,
        rtol={"aspect": 5.0e-3, "state_norm": 5.0e-3, "qs_total": 2.0e-2},
        atol={"aspect": 5.0e-8, "state_norm": 5.0e-8, "qs_total": 1.0e-8},
        base_value_atol={"aspect": 2.0e-3, "state_norm": 2.0e-3, "qs_total": 2.0e-3},
    )
    complete_report = reports["complete_report"]
    scalars_report = reports["scalars_report"]

    changed_branch_report = deepcopy(complete_report)
    changed_branch_report["branch_compatibility"]["same_branch"] = False
    changed_branch_report["branch_compatibility"]["plus_fingerprint"] = deepcopy(
        changed_branch_report["branch_compatibility"]["plus_fingerprint"]
    )
    changed_branch_report["branch_compatibility"]["plus_fingerprint"]["step_status"] = (
        "momentum",
        "restart_bad_jacobian",
        "restart_bad_jacobian",
    )
    changed_branch_gate = _native_rejected_adaptive_gate_report(
        changed_branch_report,
        scalars_report,
        scalar_keys=scalar_keys,
    )
    assert changed_branch_gate["passed"] is False
    assert changed_branch_gate["same_branch"] is False
    assert changed_branch_gate["same_full_loop_branch_fingerprint"] is False
    assert any("branch fingerprints" in error for error in changed_branch_gate["errors"])


@pytest.mark.py311_coverage_only
def test_direct_coil_native_rejected_slot_betatotal_jvp_matches_complete_solve_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finite-beta scalar JVP is branch-local and fingerprint-gated."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp
    from vmec_jax.finite_beta import finite_beta_scalars_from_state

    enable_x64(True)
    _set_same_branch_custom_vjp_env(monkeypatch)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_native_rejected_slot_betatotal",
        lasym=False,
        niter=3,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e8, n_segments=24)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    current_fraction = 0.002
    direction = base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(base_dofs),
        base_currents=base_currents * current_fraction,
    )

    def params_for(scale: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_curve_dofs=base_dofs,
            base_currents=base_currents * (1.0 + current_fraction * float(scale)),
        )

    def betatotal_from_state(state, payload):
        return finite_beta_scalars_from_state(
            state=state,
            static=payload["init"].static,
            indata=payload["init"].indata,
            signgs=int(payload["init"].signgs),
        )["betatotal"]

    def scalar_map(payload):
        return {"betatotal": betatotal_from_state(payload["result"].state, payload)}

    scalar_keys = ("betatotal",)
    reports = _native_rejected_slot_scalars_report(
        input_path=input_path,
        base_params=base_params,
        direction=direction,
        params_for=params_for,
        scalar_map=scalar_map,
        replay_scalar_fns={
            "betatotal": lambda replay, payload: betatotal_from_state(replay["state"], payload),
        },
        scalar_keys=scalar_keys,
        rtol={"betatotal": 1.0e-2},
        atol={"betatotal": 1.0e-8},
        base_value_atol={"betatotal": 2.0e-3},
    )
    complete_report = reports["complete_report"]
    scalars_report = reports["scalars_report"]

    changed_branch_report = deepcopy(complete_report)
    changed_branch_report["branch_compatibility"]["same_branch"] = False
    changed_branch_gate = _native_rejected_adaptive_gate_report(
        changed_branch_report,
        scalars_report,
        scalar_keys=scalar_keys,
    )
    assert changed_branch_gate["passed"] is False
    assert changed_branch_gate["same_full_loop_branch_fingerprint"] is False


@pytest.mark.py311_coverage_only
def test_direct_coil_native_rejected_slot_geometry_jvp_matches_complete_solve_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Geometry-only rejected-slot JVP is valid under an unchanged branch."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    _set_same_branch_custom_vjp_env(monkeypatch)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_native_rejected_slot_geometry",
        lasym=False,
        niter=3,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e8, n_segments=24)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    dof_index = (0, 0, 2)
    dof_step = 1.0e-3
    direction = base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(base_dofs).at[dof_index].set(dof_step),
        base_currents=jnp.zeros_like(base_currents),
    )

    def params_for(scale: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_curve_dofs=base_dofs.at[dof_index].add(dof_step * float(scale)),
            base_currents=base_currents,
        )

    scalar_keys = ("aspect", "state_norm", "qs_total")
    reports = _native_rejected_slot_scalars_report(
        input_path=input_path,
        base_params=base_params,
        direction=direction,
        params_for=params_for,
        scalar_map=_aspect_state_norm_qs_scalar_map,
        replay_scalar_fns=_aspect_state_norm_qs_replay_scalar_fns(),
        scalar_keys=scalar_keys,
        rtol={"aspect": 5.0e-3, "state_norm": 5.0e-3, "qs_total": 2.0e-2},
        atol={"aspect": 5.0e-8, "state_norm": 5.0e-8, "qs_total": 1.0e-8},
        base_value_atol={"aspect": 2.0e-3, "state_norm": 2.0e-3, "qs_total": 2.0e-3},
    )
    branch_local = reports["branch_local"]
    assert branch_local["uses_production_forward"] is True
    assert branch_local["derivative_mode"] == "directional_jvp"
    assert branch_local["replay_option_flags"]["directional_jvp_fast_path"] == "none"
    assert branch_local["replay_option_flags"]["directional_uses_fixed_coil_geometry"] is False


@pytest.mark.py311_coverage_only
def test_direct_coil_native_rejected_slot_mixed_state_only_branch_trace_jvp_matches_complete_solve_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production-style mixed current/geometry JVP matches FD on one branch."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp

    enable_x64(True)
    _set_same_branch_custom_vjp_env(monkeypatch)
    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_native_rejected_slot_mixed_state_only",
        lasym=False,
        niter=3,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e8, n_segments=24)
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    dof_index = (0, 0, 2)
    dof_step = 1.0e-3
    current_fraction = 0.002
    direction = base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(base_dofs).at[dof_index].set(dof_step),
        base_currents=base_currents * current_fraction,
    )

    def params_for(scale: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_curve_dofs=base_dofs.at[dof_index].add(dof_step * float(scale)),
            base_currents=base_currents * (1.0 + current_fraction * float(scale)),
        )

    scalar_keys = ("aspect", "qs_total", "lcfs_boundary_moment")
    reports = _native_rejected_slot_scalars_report(
        input_path=input_path,
        base_params=base_params,
        direction=direction,
        params_for=params_for,
        scalar_map=_aspect_qs_boundary_scalar_map,
        replay_scalar_fns=_aspect_qs_boundary_replay_scalar_fns(),
        scalar_keys=scalar_keys,
        rtol={"aspect": 5.0e-3, "qs_total": 2.0e-2, "lcfs_boundary_moment": 5.0e-3},
        atol={"aspect": 5.0e-8, "qs_total": 1.0e-8, "lcfs_boundary_moment": 5.0e-8},
        base_value_atol={"aspect": 2.0e-3, "qs_total": 2.0e-3, "lcfs_boundary_moment": 2.0e-3},
        replay_kwargs_extra={"state_only_replay": True},
        solve_kwargs_extra={"adjoint_trace_mode": "branch"},
    )
    branch_local = reports["branch_local"]
    assert branch_local["uses_production_forward"] is True
    assert branch_local["derivative_mode"] == "directional_jvp"
    assert branch_local["replay_option_flags"]["state_only_replay"] is True
    assert branch_local["replay_option_flags"]["directional_jvp_fast_path"] == "none"
    assert branch_local["replay_option_flags"]["directional_uses_fixed_coil_geometry"] is False


@pytest.mark.py311_coverage_only
def test_direct_coil_branch_trace_mode_keeps_replay_controls_without_raw_force_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lean branch traces keep replay controls but omit full raw-force payloads."""

    pytest.importorskip("jax")
    enable_x64(True)
    _set_same_branch_custom_vjp_env(monkeypatch)

    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_accepted_trace_preconditioner_controls_jax,
        direct_coil_accepted_trace_preconditioner_policy_segments,
        direct_coil_accepted_trace_replay_graph_metadata,
        direct_coil_accepted_trace_step_controls_jax,
        direct_coil_complete_solve_trace,
        free_boundary_adjoint_trace_replay_diagnostics,
    )

    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_branch_trace_mode",
        lasym=False,
        niter=1,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=24)

    payload = direct_coil_complete_solve_trace(
        input_path,
        base_params,
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
            "adjoint_trace_mode": "branch",
            "external_field_provider_kind": "direct_coils",
            "free_boundary_activate_fsq": 1.0e99,
        },
    )

    traces = tuple(payload["traces"])
    assert traces
    trace = traces[0]
    for key in (
        "state_pre",
        "state_post",
        "dt_eff",
        "b1",
        "fac",
        "force_scale",
        "lam_prec",
        "precond_mats",
        "w_mode_mn",
        "freeb_bsqvac_half",
        "freeb_nestor_trace",
    ):
        assert key in trace
    for omitted_key in ("frzl_frcc", "frzl_rz_frcc", "frcc_u", "vRcc_after"):
        assert omitted_key not in trace

    step_controls = direct_coil_accepted_trace_step_controls_jax(traces)
    preconditioner_segments = direct_coil_accepted_trace_preconditioner_policy_segments(traces)
    segment = preconditioner_segments[0]
    preconditioner_controls = direct_coil_accepted_trace_preconditioner_controls_jax(
        traces[int(segment["start"]) : int(segment["stop"])]
    )
    assert "state_pre" in step_controls
    assert "precond_mats" in preconditioner_controls
    metadata = direct_coil_accepted_trace_replay_graph_metadata(
        traces,
        static=payload["init"].static,
        use_stacked_step_controls=True,
        json_safe=True,
    )
    assert metadata["active_free_boundary_replay_steps"] >= 1
    diagnostics = free_boundary_adjoint_trace_replay_diagnostics(traces)
    assert diagnostics["differentiates_adaptive_controller"] is False
    assert diagnostics["replay_diagnostics"]["scalar_controls_stackable"] is True
    assert diagnostics["replay_diagnostics"]["array_controls_stackable"] is True
    assert diagnostics["replay_diagnostics"]["preconditioner_policy_n_segments"] >= 1


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
        check_accepted_bnormal_rms_scalar=True,
        check_accepted_bsqvac_rms_scalar=True,
        require_positive_accepted_vacuum_scalar_slope=False,
        check_production_branch_local_scalar=True,
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
    from vmec_jax.free_boundary import _sample_external_boundary_arrays
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import direct_coil_boundary_bnormal_rms_jax

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

    run = _run_forced_active_direct_solve(
        input_path,
        base_params,
        max_iter=2,
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
    from vmec_jax.discrete_adjoint import strict_update_one_step_from_trace
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
        direct_coil_accepted_trace_controller_replay_objective_jax,
        direct_coil_accepted_trace_controller_replay_plan,
        accepted_trace_effective_state_pre,
        direct_coil_accepted_trace_fingerprint,
        direct_coil_accepted_trace_fingerprint_delta,
        direct_coil_accepted_trace_step_controls_jax,
        direct_coil_accepted_trace_step_policy_segments,
        direct_coil_accepted_trace_replay_objective_jax,
        direct_coil_boundary_bsqvac_from_trace_jax,
        direct_coil_boundary_replay_context,
        free_boundary_boundary_geometry_jax,
    )
    from vmec_jax.state import pack_state, unpack_state

    enable_x64(True)
    _set_same_branch_custom_vjp_env(monkeypatch)

    input_path = _write_tiny_direct_freeb_input(
        tmp_path / "input.direct_accepted_update_ad_fd",
        lasym=False,
        niter=3,
        mpol=3,
        ntheta=4,
    )
    base_params = _circle_coil_params(current=3.0e7, n_segments=16)
    init = _run_direct_initial_guess(input_path, base_params)
    result = _solve_direct_residual_iter(
        init,
        base_params,
        max_iter=3,
        use_scan=False,
        host_update_assembly=False,
        adjoint_trace=True,
        adjoint_trace_mode="full",
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

    selected = _select_finite_direct_coil_replay_trace(init, base_params, active_traces)
    assert selected.index + 1 < len(active_traces), "Selected trace must have a following trace for two-step replay"
    trace = selected.trace
    trace1 = active_traces[selected.index + 1]
    replay_from_coils = selected.replay_from_coils

    def bsqvac_from_coils(params: CoilFieldParams):
        return replay_from_coils(params)["bsqvac"]

    _assert_selected_direct_coil_replay_matches_trace(selected)

    # The accepted trace must be exactly replayable once the force channels have
    # been computed. This protects accepted-output correctness separately from
    # the harder coil -> NESTOR -> force reconstruction path below.
    effective_state_pre = _assert_trace_force_channels_and_step_replay(trace, init.static)
    _assert_strict_update_coil_directional_derivative_matches_fd(
        init=init,
        trace=trace,
        effective_state_pre=effective_state_pre,
        base_params=base_params,
        direction=direction,
        bsqvac_from_coils=bsqvac_from_coils,
    )

    flat0 = jnp.asarray(pack_state(effective_state_pre))

    def state_replay_objective(flat_state):
        state = unpack_state(flat_state, effective_state_pre.layout)
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
            [0, flat0.size // 7, flat0.size // 5, flat0.size // 3, flat0.size // 2, (2 * flat0.size) // 3, flat0.size - 1],
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

    first_step = _strict_accepted_step_from_trace(trace0, init.static, state_pre=accepted_trace_effective_state_pre(trace0))
    replayed_state1 = first_step["state_post"]
    np.testing.assert_allclose(
        np.asarray(pack_state(replayed_state1)),
        np.asarray(pack_state(trace0["state_post"])),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(pack_state(replayed_state1)),
        np.asarray(pack_state(accepted_trace_effective_state_pre(trace1))),
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

    traces01 = [trace0, trace1]

    def accepted_replay_objective(**kwargs):
        return direct_coil_accepted_trace_replay_objective_jax(
            base_params,
            accepted_trace_effective_state_pre(trace0),
            static=init.static,
            traces=traces01,
            signgs=int(init.signgs),
            state_weight=1.0,
            bsqvac_weight=1.0e-12,
            force_weight=0.0,
            enforce_edge=False,
            **kwargs,
        )

    def controller_replay_objective(
        *,
        params: CoilFieldParams = base_params,
        start_state: Any | None = None,
        traces: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        state_weight = kwargs.pop("state_weight", 1.0)
        bsqvac_weight = kwargs.pop("bsqvac_weight", 1.0e-12)
        force_weight = kwargs.pop("force_weight", 0.0)
        enforce_edge = kwargs.pop("enforce_edge", False)
        return direct_coil_accepted_trace_controller_replay_objective_jax(
            params,
            accepted_trace_effective_state_pre(trace0) if start_state is None else start_state,
            static=init.static,
            traces=traces01 if traces is None else traces,
            signgs=int(init.signgs),
            state_weight=state_weight,
            bsqvac_weight=bsqvac_weight,
            force_weight=force_weight,
            enforce_edge=enforce_edge,
            **kwargs,
        )

    def assert_objective_close(left, right, *, rtol: float = 2.0e-12, atol: float = 1.0e-12) -> None:
        np.testing.assert_allclose(np.asarray(left["objective"]), np.asarray(right["objective"]), rtol=rtol, atol=atol)

    def assert_state_close(left, right, *, rtol: float = 5.0e-12, atol: float = 5.0e-12) -> None:
        np.testing.assert_allclose(np.asarray(pack_state(left["state"])), np.asarray(pack_state(right["state"])), rtol=rtol, atol=atol)

    def assert_array_equal(container, name, expected) -> None:
        np.testing.assert_array_equal(np.asarray(container[name]), np.asarray(expected))

    def assert_history_close(left, right, key, *, rtol: float = 5.0e-12, atol: float = 5.0e-12) -> None:
        np.testing.assert_allclose(np.asarray(left["history"][key]), np.asarray(right["history"][key]), rtol=rtol, atol=atol)

    def assert_history_equal(left, right, key) -> None:
        np.testing.assert_array_equal(np.asarray(left["history"][key]), np.asarray(right["history"][key]))

    replay = accepted_replay_objective()
    assert {"state", "bsqvac", "force"}.issubset(replay["objective_components"])
    assert np.isfinite(float(replay["objective"]))
    controller_replay = controller_replay_objective()
    fallback_controller_replay = controller_replay_objective(use_accepted_only_fast_path=False)
    assert controller_replay["used_accepted_only_fast_path"]
    assert controller_replay["accepted_only_fast_path_segments"] == (True,)
    assert not fallback_controller_replay["used_accepted_only_fast_path"]
    assert fallback_controller_replay["accepted_only_fast_path_segments"] == (False,)
    assert_objective_close(controller_replay, fallback_controller_replay)
    assert_state_close(controller_replay, fallback_controller_replay)
    for key in ("active", "accepted", "rejected", "done", "state_reset", "force", "bsqvac"):
        assert_history_close(controller_replay, fallback_controller_replay, key)
    for container, name, expected in (
        (controller_replay["history"], "accepted", [True, True]),
        (controller_replay["history"], "rejected", [False, False]),
        (controller_replay["controls"], "step_index", [0, 1]),
        (controller_replay["controls"], "reset_to_trace_pre", [False, False]),
        (controller_replay["controls"], "has_active_freeb_replay", [True, True]),
    ):
        assert_array_equal(container, name, expected)
    np.testing.assert_allclose(
        np.asarray(controller_replay["controls"]["step_scalars"]["dt_eff"]),
        np.asarray([_trace_scalar_value(trace0["dt_eff"]), _trace_scalar_value(trace1["dt_eff"])]),
    )
    for key in ("flip_sign", "limit_update_rms", "divide_by_scalxc_for_update"):
        assert key in controller_replay["controls"]["step_scalars"]
    for key in ("preconditioner_use_lax_tridi", "preconditioner_use_precomputed_tridi"):
        assert key not in controller_replay["controls"]["step_scalars"]
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
    assert_objective_close(controller_replay, replay)
    assert_state_close(controller_replay, replay)
    segmented_controller_replay = controller_replay_objective(use_preconditioner_policy_segments=True)
    segmented_fallback_replay = controller_replay_objective(
        use_preconditioner_policy_segments=True,
        use_accepted_only_fast_path=False,
    )
    assert segmented_controller_replay["used_preconditioner_policy_segments"]
    assert segmented_controller_replay["preconditioner_controls_segment_stacked"] == (True,)
    assert segmented_controller_replay["used_accepted_only_fast_path"]
    assert segmented_controller_replay["accepted_only_fast_path_segments"] == (True,)
    assert segmented_fallback_replay["accepted_only_fast_path_segments"] == (False,)
    assert_objective_close(segmented_controller_replay, controller_replay)
    assert_state_close(segmented_controller_replay, controller_replay)
    assert_objective_close(segmented_controller_replay, segmented_fallback_replay)
    stacked_controller_replay = controller_replay_objective(use_stacked_step_controls=True)
    stacked_fallback_replay = controller_replay_objective(
        use_stacked_step_controls=True,
        use_accepted_only_fast_path=False,
    )
    assert stacked_controller_replay["used_stacked_step_controls"]
    assert stacked_controller_replay["step_policy_n_segments"] == len(step_segments)
    assert stacked_controller_replay["preconditioner_controls_segment_stacked"] == (True,) * len(step_segments)
    assert stacked_controller_replay["used_accepted_only_fast_path"]
    assert stacked_controller_replay["accepted_only_fast_path_segments"] == (True,) * len(step_segments)
    assert stacked_fallback_replay["accepted_only_fast_path_segments"] == (False,) * len(step_segments)
    assert_objective_close(stacked_controller_replay, controller_replay)
    stacked_plan = direct_coil_accepted_trace_controller_replay_plan(
        [trace0, trace1],
        static=init.static,
        use_stacked_step_controls=True,
    )
    assert stacked_plan["segment_source"] == "step_policy"
    assert stacked_plan["accepted_only_fast_path_segments"] == (True,) * len(step_segments)
    stacked_plan_replay = controller_replay_objective(
        use_stacked_step_controls=True,
        replay_plan=stacked_plan,
    )
    assert stacked_plan_replay["used_stacked_step_controls"]
    assert_objective_close(stacked_plan_replay, stacked_controller_replay)
    stacked_state_only_replay = controller_replay_objective(
        state_weight=0.0,
        bsqvac_weight=0.0,
        force_weight=0.0,
        use_stacked_step_controls=True,
        replay_plan=stacked_plan,
        state_only_replay=True,
    )
    assert stacked_state_only_replay["used_stacked_step_controls"]
    assert stacked_state_only_replay["used_state_only_replay"] is True
    assert stacked_state_only_replay["history"] == {}
    assert_state_close(stacked_state_only_replay, stacked_controller_replay)
    frozen_vacuum_field_replay = controller_replay_objective(
        state_weight=0.0,
        bsqvac_weight=0.0,
        force_weight=0.0,
        use_stacked_step_controls=True,
        replay_plan=stacked_plan,
        state_only_replay=True,
        freeze_vacuum_field=True,
    )
    frozen_bsqvac_replay = controller_replay_objective(
        state_weight=0.0,
        bsqvac_weight=0.0,
        force_weight=0.0,
        use_stacked_step_controls=True,
        replay_plan=stacked_plan,
        state_only_replay=True,
        freeze_freeb_bsqvac=True,
    )
    assert frozen_vacuum_field_replay["used_state_only_replay"] is True
    assert frozen_bsqvac_replay["used_state_only_replay"] is True
    assert np.isfinite(float(jnp.linalg.norm(pack_state(frozen_vacuum_field_replay["state"]))))
    assert np.isfinite(float(jnp.linalg.norm(pack_state(frozen_bsqvac_replay["state"]))))

    def full_replay_state_norm(params: CoilFieldParams):
        replay = controller_replay_objective(
            params=params,
            state_weight=0.0,
            bsqvac_weight=0.0,
            force_weight=0.0,
            use_stacked_step_controls=True,
            replay_plan=stacked_plan,
            include_replay_aux=False,
        )
        return jnp.linalg.norm(pack_state(replay["state"]))

    def state_only_replay_state_norm(params: CoilFieldParams):
        replay = controller_replay_objective(
            params=params,
            state_weight=0.0,
            bsqvac_weight=0.0,
            force_weight=0.0,
            use_stacked_step_controls=True,
            replay_plan=stacked_plan,
            include_replay_aux=False,
            state_only_replay=True,
        )
        return jnp.linalg.norm(pack_state(replay["state"]))

    full_value, full_jvp = jax.jvp(full_replay_state_norm, (base_params,), (direction,))
    state_only_value, state_only_jvp = jax.jvp(state_only_replay_state_norm, (base_params,), (direction,))
    np.testing.assert_allclose(state_only_value, full_value, rtol=5.0e-12, atol=5.0e-12)
    np.testing.assert_allclose(state_only_jvp, full_jvp, rtol=5.0e-12, atol=5.0e-12)
    assert_state_close(stacked_controller_replay, controller_replay)
    assert_objective_close(stacked_controller_replay, stacked_fallback_replay)
    for key in ("active", "accepted", "rejected", "done", "state_reset"):
        assert_history_equal(segmented_controller_replay, controller_replay, key)
        assert_history_equal(stacked_controller_replay, controller_replay, key)

    axis_reference_trace = deepcopy(trace0)
    axis_changed_trace = deepcopy(trace0)
    axis_changed_nestor = deepcopy(trace0["freeb_nestor_trace"])
    axis_changed_nestor["br_axis"] = np.asarray(axis_changed_nestor["br_axis"], dtype=float) + 2.0e-4
    axis_changed_nestor["bp_axis"] = np.asarray(axis_changed_nestor["bp_axis"], dtype=float) - 1.0e-4
    axis_changed_nestor["bz_axis"] = np.asarray(axis_changed_nestor["bz_axis"], dtype=float) + 3.0e-4
    axis_changed_trace["freeb_nestor_trace"] = axis_changed_nestor
    assert [
        (segment["start"], segment["stop"], segment["n_steps"])
        for segment in direct_coil_accepted_trace_step_policy_segments([axis_reference_trace, axis_changed_trace])
    ] == [(0, 2, 2)]
    axis_reference_replay = controller_replay_objective(
        start_state=axis_reference_trace["state_pre"],
        traces=[axis_reference_trace, axis_reference_trace],
    )
    axis_changed_controller_replay = controller_replay_objective(
        start_state=axis_reference_trace["state_pre"],
        traces=[axis_reference_trace, axis_changed_trace],
    )
    axis_changed_stacked_replay = controller_replay_objective(
        start_state=axis_reference_trace["state_pre"],
        traces=[axis_reference_trace, axis_changed_trace],
        use_stacked_step_controls=True,
    )
    assert axis_changed_stacked_replay["used_stacked_step_controls"]
    assert axis_changed_stacked_replay["step_policy_n_segments"] == 1
    assert abs(
        float(
            np.asarray(axis_changed_controller_replay["history"]["bsqvac_rms"][-1])
            - np.asarray(axis_reference_replay["history"]["bsqvac_rms"][-1])
        )
    ) > 1.0e-9
    for key in ("bsqvac_rms", "bnormal_rms"):
        assert_history_close(axis_changed_stacked_replay, axis_changed_controller_replay, key)
    assert_objective_close(axis_changed_stacked_replay, axis_changed_controller_replay, rtol=5.0e-12, atol=5.0e-12)
    assert_state_close(axis_changed_stacked_replay, axis_changed_controller_replay)

    static_changed_trace = dict(trace1)
    static_changed_trace["include_edge_residual"] = not bool(trace1["include_edge_residual"])
    assert [
        (segment["start"], segment["stop"], segment["n_steps"])
        for segment in direct_coil_accepted_trace_step_policy_segments([trace0, static_changed_trace])
    ] == [(0, 1, 1), (1, 2, 1)]
    padded_bad_trace = dict(trace1)
    padded_bad_trace["dt_eff"] = _trace_scalar_value(trace1["dt_eff"]) * 10.0
    padded_bad_trace["force_scale"] = _trace_scalar_value(trace1["force_scale"]) * 10.0
    padded_controller_replay = controller_replay_objective(
        traces=[trace0, trace1, padded_bad_trace],
        accept_mask=np.asarray([True, True, False]),
        done_mask=np.asarray([False, True, False]),
    )
    assert not padded_controller_replay["used_accepted_only_fast_path"]
    assert padded_controller_replay["accepted_only_fast_path_segments"] == (False,)
    assert_array_equal(padded_controller_replay["history"], "active", [True, True, False])
    assert_array_equal(padded_controller_replay["history"], "accepted", [True, True, False])
    assert_array_equal(padded_controller_replay["controls"], "reset_to_trace_pre", [False, False, True])
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
    assert_objective_close(padded_controller_replay, controller_replay)


@pytest.mark.py311_coverage_only
def test_direct_coil_vacuum_field_override_replay_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the diagnostic frozen-vacuum replay branch with synthetic arrays.

    The frozen-vacuum path is diagnostic-only, but it is useful for performance
    attribution.  This test keeps it honest without launching a VMEC solve: the
    direct-coil/NESTOR source arrays are supplied from a trace, while the dense
    solve and field reconstruction are patched to a small deterministic map.
    """

    pytest.importorskip("jax")
    from vmec_jax._compat import jnp
    import vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives as freeb_adj

    shape = (2, 2)
    base_trace = {
        "bnormal": jnp.asarray([[1.0, -2.0], [3.0, -4.0]]),
        "g_uu": jnp.ones(shape),
        "g_uv": 2.0 * jnp.ones(shape),
        "g_vv": 3.0 * jnp.ones(shape),
    }
    trace_with_tangents = {
        "freeb_nestor_trace": {
            **base_trace,
            "bexu_ext": 4.0 * jnp.ones(shape),
            "bexv_ext": -5.0 * jnp.ones(shape),
        }
    }
    override = freeb_adj._direct_coil_trace_vacuum_field_override(trace_with_tangents)
    np.testing.assert_allclose(np.asarray(override["bu"]), 4.0)
    np.testing.assert_allclose(np.asarray(override["bv"]), -5.0)

    trace_with_legacy_names = {**base_trace, "bu": 6.0 * jnp.ones(shape), "bv": 7.0 * jnp.ones(shape)}
    legacy_override = freeb_adj._direct_coil_trace_vacuum_field_override(trace_with_legacy_names)
    np.testing.assert_allclose(np.asarray(legacy_override["bu"]), 6.0)
    np.testing.assert_allclose(np.asarray(legacy_override["bv"]), 7.0)

    zero_tangent_override = freeb_adj._direct_coil_trace_vacuum_field_override(base_trace)
    np.testing.assert_allclose(np.asarray(zero_tangent_override["bu"]), 0.0)
    np.testing.assert_allclose(np.asarray(zero_tangent_override["bv"]), 0.0)

    with pytest.raises(ValueError, match="NESTOR trace"):
        freeb_adj._direct_coil_trace_vacuum_field_override({"freeb_nestor_trace": object()})
    with pytest.raises(ValueError, match="missing vacuum-field override"):
        freeb_adj._direct_coil_trace_vacuum_field_override({"bnormal": jnp.ones(shape)})

    calls: list[dict[str, object]] = []

    def fake_dense_vmec_nestor_mode_solve_jax(**kwargs):
        calls.append(
            {
                "bexni": kwargs["bexni"],
                "include_analytic": kwargs["include_analytic"],
                "include_phi_flat": kwargs["include_phi_flat"],
                "include_residual": kwargs["include_residual"],
                "solve_mode": kwargs["solve_mode"],
            }
        )
        return {"mode_coeffs": 0.0 * kwargs["bexni"]}

    def fake_vacuum_boundary_fields_from_mode_coeffs_jax(_mode_coeffs, *, bu_ext, bv_ext, g_uu, g_uv, g_vv, basis):
        del basis
        return {
            "bsqvac": jnp.asarray(bu_ext)
            + 2.0 * jnp.asarray(bv_ext)
            + jnp.asarray(g_uu)
            + jnp.asarray(g_uv)
            + jnp.asarray(g_vv)
        }

    monkeypatch.setattr(freeb_adj, "dense_vmec_nestor_mode_solve_jax", fake_dense_vmec_nestor_mode_solve_jax)
    monkeypatch.setattr(
        freeb_adj,
        "vacuum_boundary_fields_from_mode_coeffs_jax",
        fake_vacuum_boundary_fields_from_mode_coeffs_jax,
    )

    grid = jnp.ones(shape)
    grid_kwargs = dict.fromkeys(("R", "Z", "phi", "Ru", "Zu", "Rv", "Zv", "ruu", "ruv", "rvv", "zuu", "zuv", "zvv"), grid)
    out = freeb_adj.direct_coil_boundary_bsqvac_jax(
        params=None,
        **grid_kwargs,
        basis={},
        tables={},
        signgs=1,
        nvper=1,
        wint=0.5 * grid,
        include_analytic=False,
        include_diagnostics=True,
        vac_override=override,
    )
    expected_bexni = -override["bnormal"] * 0.5 * ((2.0 * np.pi) ** 2)
    np.testing.assert_allclose(np.asarray(calls[-1]["bexni"]).reshape(shape), np.asarray(expected_bexni))
    assert calls[-1]["include_analytic"] is False
    assert calls[-1]["include_phi_flat"] is True
    assert calls[-1]["include_residual"] is True
    assert calls[-1]["solve_mode"] == "dense"
    np.testing.assert_allclose(np.asarray(out["bsqvac"]), np.asarray(override["bu"] + 2.0 * override["bv"] + 6.0))
    assert {"channels", "mode_solution", "vac", "bexni"}.issubset(out)

    out_no_diag = freeb_adj.direct_coil_boundary_bsqvac_jax(
        params=None,
        **grid_kwargs,
        basis={},
        tables={},
        signgs=1,
        nvper=1,
        include_diagnostics=False,
        include_mode_diagnostics=False,
        vac_override=zero_tangent_override,
        nestor_solve_mode="matrix_free",
    )
    assert set(out_no_diag) == {"bsqvac"}
    assert calls[-1]["include_phi_flat"] is False
    assert calls[-1]["include_residual"] is False
    assert calls[-1]["solve_mode"] == "matrix_free"


@pytest.mark.parametrize("lasym", [False, True], ids=["stellsym", "lasym"])
def test_jax_free_boundary_boundary_geometry_matches_host_sampler(
    tmp_path: Path,
    lasym: bool,
) -> None:
    """The phase-2 JAX boundary sampler must match production host geometry."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp
    from vmec_jax.free_boundary import _sample_external_boundary_arrays
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import free_boundary_boundary_geometry_jax
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
    run = _run_direct_initial_guess(input_path, params)
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
    from vmec_jax.external_fields import sample_coil_field_cylindrical
    from vmec_jax.free_boundary import _sample_external_boundary_arrays
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
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
    init = _run_direct_initial_guess(input_path, base_params)
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

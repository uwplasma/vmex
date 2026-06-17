"""Output, checkpoint, and reporting helpers for fixed-boundary workflows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .parameterization import boundary_param_names


def print_qs_problem_summary(
    *,
    method: str,
    max_nfev: int,
    use_mode_continuation: bool,
    use_ess: bool,
    ess_alpha: float,
    objectives: Sequence,
    specs: Sequence,
    x_scale: np.ndarray,
    optimizer,
    params0,
) -> None:
    """Print the problem summary used by the standalone examples."""

    print(f"Parameter space ({len(specs)} DOFs): {boundary_param_names(specs)}")
    print("Objectives:")
    for term in objectives:
        print(f"  - {term.name}: target={term.target}, weight={term.weight}")
    if use_ess:
        print(f"ESS scales (alpha={ess_alpha}): min={x_scale.min():.3f}  max={x_scale.max():.3f}")
    else:
        print("ESS disabled - uniform scales.")
    print(f"Aspect ratio (initial):        {optimizer.aspect_ratio(params0):.6f}")
    print(f"Field objective (initial):     {optimizer.quasisymmetry_objective(params0):.6e}")
    print(f"Running {method} (max_nfev={max_nfev}, continuation={use_mode_continuation}) ...")


def print_qs_final_summary(
    result: dict,
    *,
    target_iota: float | None = None,
    iota_abs_min: float | None = None,
) -> None:
    """Print the final scalar diagnostics from an optimization result."""

    hist = result.get("_history_dump", {})
    print(f"\nTermination: {result['message']}")
    print(f"Aspect ratio (final):          {float(hist.get('aspect_final', float('nan'))):.6f}")
    if "iota_final" in hist:
        if target_iota is not None:
            target = f"  target={target_iota:.6f}"
        elif iota_abs_min is not None:
            target = f"  min |iota|={iota_abs_min:.6f}"
        else:
            target = ""
        print(f"Mean iota (final):             {float(hist['iota_final']):.6f}{target}")
    print(f"Field objective (final):       {float(hist.get('qs_final', float('nan'))):.6e}")
    print(f"Total objective (final):       {float(hist.get('objective_final', float('nan'))):.6e}")
    obj0 = hist.get("objective_initial")
    objf = hist.get("objective_final")
    if obj0 is not None and float(obj0) > 0.0 and objf is not None:
        print(f"Objective reduction:           {100.0 * (1.0 - float(objf) / float(obj0)):.1f}%")


def save_qs_stage_artifacts(
    *,
    stage_dir: Path,
    optimizer,
    params_initial,
    params_final,
    result,
    save_inputs: bool = True,
    save_wouts: bool = False,
    save_rerun_wouts: bool = False,
    run_fixed_boundary_func,
    write_wout_from_fixed_boundary_run_func,
) -> None:
    """Save stage input files and optionally wout files."""

    stage_dir.mkdir(parents=True, exist_ok=True)
    if save_inputs:
        optimizer.save_input(stage_dir / "input.initial", params_initial)
        optimizer.save_input(stage_dir / "input.final", params_final)
    if save_wouts:
        optimizer.save_wout(stage_dir / "wout_initial.nc", params_initial, state=result.get("_state_initial"))
        optimizer.save_wout(stage_dir / "wout_final.nc", params_final, state=result.get("_state_final"))
    else:
        remove_stale(stage_dir / "wout_initial.nc")
        remove_stale(stage_dir / "wout_final.nc")
    if save_rerun_wouts:
        rerun = run_fixed_boundary_func(str(stage_dir / "input.initial"), verbose=False)
        write_wout_from_fixed_boundary_run_func(str(stage_dir / "wout_initial_rerun.nc"), rerun)
        rerun = run_fixed_boundary_func(str(stage_dir / "input.final"), verbose=False)
        write_wout_from_fixed_boundary_run_func(str(stage_dir / "wout_final_rerun.nc"), rerun)
    else:
        remove_stale(stage_dir / "wout_initial_rerun.nc")
        remove_stale(stage_dir / "wout_final_rerun.nc")


def result_objective_final(result) -> float:
    """Return the final scalar objective recorded for a stage result."""

    try:
        history = result.get("_history_dump", {})
        if "objective_final" in history:
            return float(history["objective_final"])
        if "objective" in result:
            return float(result["objective"])
        if "cost" in result:
            return 2.0 * float(result["cost"])
    except Exception:
        return math.inf
    return math.inf


def select_nonworsening_stage_record(
    attempted_record,
    accepted_stage_records,
    *,
    stage_label: str,
):
    """Reject continuation stages whose final objective worsens the handoff."""

    if not accepted_stage_records:
        return attempted_record
    previous_record = accepted_stage_records[-1]
    if int(attempted_record[0]) != int(previous_record[0]):
        return attempted_record
    previous_objective = result_objective_final(previous_record[3])
    attempted_objective = result_objective_final(attempted_record[3])
    if not math.isfinite(previous_objective):
        return attempted_record
    tolerance = max(1.0e-12, 1.0e-10 * max(1.0, abs(previous_objective)))
    if math.isfinite(attempted_objective) and attempted_objective <= previous_objective + tolerance:
        return attempted_record

    history = attempted_record[3].setdefault("_history_dump", {})
    history["stage_rejected_nonworsening"] = True
    history["stage_rejected_reference_objective"] = float(previous_objective)
    history["stage_rejected_attempt_objective"] = float(attempted_objective)
    history["stage_rejected_reason"] = "continuation stage worsened final objective"
    print(
        f"Stage {stage_label} rejected by non-worsening continuation guard: "
        f"{attempted_objective:.6e} > {previous_objective:.6e}."
    )
    return previous_record


def write_qi_workflow_stage_checkpoint(
    *,
    output_dir: Path,
    stage_dir: Path,
    stage_index: int,
    stage_limits,
    result: dict,
    completed_stage_modes,
    requested_stage_modes,
    normalize_boundary_mode_limits_func,
    describe_boundary_mode_limits_func,
    stage_mode_checkpoint_descriptor_func,
) -> Path:
    """Write a resumable checkpoint after one QI continuation stage."""

    output_dir = Path(output_dir)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    limits = normalize_boundary_mode_limits_func(stage_limits)
    history = dict(result.get("_history_dump", {}))
    diagnostics = {
        "partial": True,
        "objective_final": history.get("objective_final"),
        "qs_final": history.get("qs_final"),
        "aspect": history.get("aspect_final"),
        "mean_iota": history.get("iota_final"),
        "nfev": history.get("nfev"),
        "njev": history.get("njev"),
        "total_wall_time_s": history.get("total_wall_time_s"),
        "stage_checkpoint_source": "optimization_workflow",
    }
    checkpoint = {
        "partial": True,
        "role": "mode_continuation",
        "stage": int(stage_index),
        "name": describe_boundary_mode_limits_func(limits),
        "stage_modes": [stage_mode_checkpoint_descriptor_func(limits)],
        "completed_stage_modes": [int(mode) for mode in completed_stage_modes],
        "requested_stage_modes": [stage_mode_checkpoint_descriptor_func(mode) for mode in requested_stage_modes],
        "stage_output_dir": str(stage_dir),
        "initial_input_path": str(stage_dir / "input.initial"),
        "final_input_path": str(stage_dir / "input.final"),
        "initial_wout_path": str(stage_dir / "wout_initial.nc"),
        "final_wout_path": str(stage_dir / "wout_final.nc"),
        "input_path": str(stage_dir / "input.final"),
        "wout_path": str(stage_dir / "wout_final.nc"),
        "history_path": str(stage_dir / "history.json"),
        "diagnostics_path": str(stage_dir / "diagnostics.json"),
        "history": history,
        "diagnostics": diagnostics,
    }
    history_path = stage_dir / "history.json"
    diagnostics_path = stage_dir / "diagnostics.json"
    checkpoint_path = stage_dir / "qi_stage_checkpoint.json"
    root_checkpoint_path = output_dir / "stage_checkpoint.json"
    write_json_atomic(history_path, history)
    write_json_atomic(diagnostics_path, diagnostics)
    write_json_atomic(checkpoint_path, checkpoint)
    write_json_atomic(root_checkpoint_path, checkpoint)
    return checkpoint_path


def save_qs_final_outputs(
    *,
    output_dir: Path,
    stage_records,
    final_optimizer,
    final_result: dict,
    label: str,
    target_aspect: float | None = None,
    target_iota: float | None = None,
    iota_abs_min: float | None = None,
    save_rerun_wouts: bool = False,
    annotate_final_history_func,
    run_fixed_boundary_func,
    write_wout_from_fixed_boundary_run_func,
) -> None:
    """Save initial/final inputs, wouts, and history."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _initial_mode, initial_optimizer, initial_params0, initial_result = stage_records[0]
    initial_optimizer.save_input(output_dir / "input.initial", initial_params0)
    initial_optimizer.save_wout(
        output_dir / "wout_initial.nc",
        initial_params0,
        state=initial_result.get("_state_initial"),
    )
    if save_rerun_wouts:
        rerun = run_fixed_boundary_func(str(output_dir / "input.initial"), verbose=False)
        write_wout_from_fixed_boundary_run_func(str(output_dir / "wout_initial_rerun.nc"), rerun)
    else:
        remove_stale(output_dir / "wout_initial_rerun.nc")

    final_optimizer.save_input(output_dir / "input.final", final_result["x"])
    final_optimizer.save_wout(
        output_dir / "wout_final.nc",
        final_result["x"],
        state=final_result.get("_state_final"),
    )
    if save_rerun_wouts:
        rerun = run_fixed_boundary_func(str(output_dir / "input.final"), verbose=False)
        write_wout_from_fixed_boundary_run_func(str(output_dir / "wout_final_rerun.nc"), rerun)
    else:
        remove_stale(output_dir / "wout_final_rerun.nc")

    annotate_final_history_func(
        final_result,
        label=label,
        target_aspect=target_aspect,
        target_iota=target_iota,
        iota_abs_min=iota_abs_min,
    )
    final_optimizer.save_history(output_dir / "history.json", final_result)


def annotate_qs_final_history(
    final_result: dict,
    *,
    label: str,
    target_aspect: float | None = None,
    target_iota: float | None = None,
    iota_abs_min: float | None = None,
) -> None:
    """Attach final optimization metadata without writing artifacts."""

    history = final_result["_history_dump"]
    history["label"] = label
    if target_aspect is not None:
        history["target_aspect"] = float(target_aspect)
    if target_iota is not None:
        history["target_iota"] = float(target_iota)
    if iota_abs_min is not None:
        history["iota_abs_min"] = float(iota_abs_min)


def combine_qs_stage_histories(
    *,
    label: str,
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
    stage_modes,
    stage_records,
    normalize_boundary_mode_limits_func,
    qs_stage_budget_func,
) -> dict | None:
    """Merge per-stage histories into one optimization history."""

    if len(stage_records) <= 1:
        return None
    normalized_stage_modes = [normalize_boundary_mode_limits_func(stage_mode) for stage_mode in stage_modes]

    combined_entries = []
    stage_boundaries = []
    wall_offset = 0.0
    nfev_total = 0
    njev_total = 0
    for idx, (_mode, _optimizer, _params0, result) in enumerate(stage_records):
        stage_hist = result["_history_dump"]
        entries = stage_hist["history"] if idx == 0 else stage_hist["history"][1:]
        for entry in entries:
            entry_copy = dict(entry)
            entry_copy["wall_time_s"] = float(entry_copy["wall_time_s"]) + wall_offset
            combined_entries.append(entry_copy)
        wall_offset = float(combined_entries[-1]["wall_time_s"]) if combined_entries else wall_offset
        stage_boundaries.append(len(combined_entries) - 1)
        nfev_total += int(stage_hist["nfev"])
        njev_total += int(stage_hist["njev"])

    final_hist = stage_records[-1][3]["_history_dump"]
    first_hist = stage_records[0][3]["_history_dump"]
    out = dict(final_hist)
    out.update(
        {
            "label": label,
            "max_nfev": int(
                sum(
                    qs_stage_budget_func(
                        stage_mode=int(mode.mode),
                        max_mode=int(max_mode),
                        max_nfev=int(max_nfev),
                        continuation_nfev=int(continuation_nfev),
                    )
                    for mode in normalized_stage_modes
                )
            ),
            "total_wall_time_s": float(wall_offset),
            "nfev": int(nfev_total),
            "njev": int(njev_total),
            "objective_initial": float(first_hist["objective_initial"]),
            "objective_final": float(final_hist["objective_final"]),
            "qs_initial": float(first_hist["qs_initial"]),
            "qs_final": float(final_hist["qs_final"]),
            "aspect_initial": float(first_hist["aspect_initial"]),
            "aspect_final": float(final_hist["aspect_final"]),
            "history": combined_entries,
            "stage_boundaries": stage_boundaries,
            "stage_mode_descriptors": [
                {
                    "mode": int(mode.mode),
                    "max_m": None if mode.max_m is None else int(mode.max_m),
                    "max_n": None if mode.max_n is None else int(mode.max_n),
                    "label": mode.label,
                }
                for mode in normalized_stage_modes
            ],
        }
    )
    if combined_entries and "iota" in combined_entries[0] and "iota" in combined_entries[-1]:
        out["iota_initial"] = float(combined_entries[0]["iota"])
        out["iota_final"] = float(combined_entries[-1]["iota"])
    return out


def remove_stale(path: Path) -> None:
    """Remove a stale optional artifact if it exists."""

    try:
        path.unlink()
    except FileNotFoundError:
        pass


def stage_mode_checkpoint_descriptor(stage_mode, *, normalize_boundary_mode_limits_func) -> dict[str, object]:
    """Return the JSON-safe descriptor for one continuation stage mode."""

    limits = normalize_boundary_mode_limits_func(stage_mode)
    return {
        "mode": int(limits.mode),
        "max_m": None if limits.max_m is None else int(limits.max_m),
        "max_n": None if limits.max_n is None else int(limits.max_n),
        "label": limits.label,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    """Atomically write JSON with NumPy/JAX-safe conversion."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(json_safe(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def json_safe(value):
    """Convert common array/scalar containers to JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def slice_boozer_surfaces(booz: dict, surface_index: int) -> dict:
    """Return a single-surface Boozer-output view for QI objectives."""

    bmnc = booz.get("bmnc_b")
    if bmnc is None:
        raise ValueError("Boozer output must include bmnc_b to slice surfaces.")
    nsurf = int(np.asarray(bmnc).shape[0])
    index = int(surface_index)
    if index < 0:
        index += nsurf
    if index < 0 or index >= nsurf:
        raise ValueError(f"surface_index {surface_index} is outside the Boozer surface range 0..{nsurf - 1}.")
    out = dict(booz)
    for key in ("bmnc_b", "bmns_b", "iota_b", "s_b"):
        value = out.get(key)
        if value is not None:
            out[key] = value[index : index + 1]
    return out

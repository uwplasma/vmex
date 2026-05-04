#!/usr/bin/env python
"""Small teaching helpers for fixed-boundary QA/QH/QP examples.

The example scripts intentionally keep the user-facing flow close to SIMSOPT:

1. choose the VMEC input and resolution,
2. choose the free boundary modes,
3. build a list of objective terms,
4. choose the optimizer,
5. run, save input/wout/history, and plot.

To add an objective, append an :class:`ObjectiveTerm` to the ``OBJECTIVES`` list
in the script.  The callback receives a :class:`StageContext` and a VMEC state
and returns either a scalar or a vector; the optimizer minimizes
``weight * (value - target)`` in least-squares form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64, jnp
from vmec_jax.quasi_isodynamic import lgradb_penalty_from_state
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


enable_x64(True)


@dataclass(frozen=True)
class StageContext:
    """Objects needed by objective callbacks for one mode-continuation stage."""

    static: object
    indata: object
    boundary_input: object
    specs: Sequence[vj.BoundaryParamSpec]
    signgs: int
    flux: object
    pressure: object


@dataclass(frozen=True)
class ObjectiveTerm:
    """One weighted least-squares objective block.

    Parameters
    ----------
    name:
        Short label used in printed summaries.
    evaluate:
        ``evaluate(ctx, state)`` returning a scalar or vector JAX value.
    target:
        Scalar or vector target subtracted from the evaluated value.
    weight:
        Multiplicative residual weight.
    total:
        Optional scalar diagnostic for the field-quality objective.  When
        provided, histories use it for ``qs_objective`` / ``qs_final``.
    track_iota:
        If true, the optimizer records mean-iota history for plotting.
    """

    name: str
    evaluate: Callable[[StageContext, object], object]
    target: float | np.ndarray = 0.0
    weight: float = 1.0
    total: Callable[[StageContext, object], object] | None = None
    track_iota: bool = False

    def residual(self, ctx: StageContext, state) -> object:
        value = _as_vector(self.evaluate(ctx, state))
        target = jnp.asarray(self.target, dtype=jnp.float64)
        if int(target.ndim) == 0:
            target = jnp.full_like(value, target)
        else:
            target = jnp.ravel(target)
        return float(self.weight) * (value - target)


def aspect_objective(target: float, weight: float = 1.0) -> ObjectiveTerm:
    """Aspect-ratio objective."""

    def _evaluate(ctx: StageContext, state):
        return equilibrium_aspect_ratio_from_state(state=state, static=ctx.static)

    return ObjectiveTerm("aspect", _evaluate, target=target, weight=weight)


def mean_iota_objective(target: float, weight: float = 1.0) -> ObjectiveTerm:
    """Mean full-mesh rotational-transform objective."""

    return ObjectiveTerm(
        "iota",
        lambda ctx, state: mean_iota(ctx, state),
        target=target,
        weight=weight,
        track_iota=True,
    )


def abs_mean_iota_floor_objective(
    target: float,
    weight: float = 1.0,
    *,
    softness: float = 1.0e-3,
) -> ObjectiveTerm:
    """Smooth lower-bound penalty enforcing ``abs(mean_iota) >= target``."""

    def _evaluate(ctx: StageContext, state):
        return vj.smooth_min_abs_iota_residual(
            mean_iota(ctx, state),
            float(target),
            softness=float(softness),
        )

    return ObjectiveTerm(
        "abs_iota_floor",
        _evaluate,
        target=0.0,
        weight=weight,
        track_iota=True,
    )


def quasisymmetry_objective(
    *,
    helicity_m: int,
    helicity_n: int,
    surfaces,
    weight: float = 1.0,
) -> ObjectiveTerm:
    """Quasisymmetry residual objective for QA, QH, or QP."""

    def _qs(ctx: StageContext, state):
        return quasisymmetry_ratio_residual_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            flux_local=ctx.flux,
            prof_local={"pressure": ctx.pressure},
            pressure_local=ctx.pressure,
            surfaces=surfaces,
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
        )

    return ObjectiveTerm(
        "qs",
        lambda ctx, state: _qs(ctx, state)["residuals1d"],
        target=0.0,
        weight=weight,
        total=lambda ctx, state: float(weight) ** 2 * _qs(ctx, state)["total"],
    )


def lgradb_objective(
    *,
    threshold: float,
    weight: float = 1.0,
    s_index: int = -1,
    ntheta: int = 9,
    nphi: int = 7,
    smooth_penalty: float = 0.0,
) -> ObjectiveTerm:
    """Minimum ``L_grad_B`` objective used by the omnigenity examples.

    This objective penalizes locations where the local magnetic-field scale
    length is below ``threshold``.  It is a differentiable JAX objective, so it
    can be appended to the same least-squares list as quasisymmetry, aspect
    ratio, or iota terms.
    """

    def _lgradb(ctx: StageContext, state):
        return lgradb_penalty_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            flux_local=ctx.flux,
            threshold=float(threshold),
            s_index=int(s_index),
            ntheta=int(ntheta),
            nphi=int(nphi),
            smooth_penalty=float(smooth_penalty),
        )

    return ObjectiveTerm(
        "LgradB",
        lambda ctx, state: _lgradb(ctx, state)["residuals1d"],
        target=0.0,
        weight=weight,
        total=lambda ctx, state: float(weight) ** 2 * _lgradb(ctx, state)["total"],
    )


def mean_iota(ctx: StageContext, state):
    """Mean rotational transform on full-mesh surfaces, excluding the axis."""

    _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=ctx.static,
        indata=ctx.indata,
        signgs=ctx.signgs,
    )
    iotas = jnp.asarray(iotas, dtype=jnp.float64)
    return jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else jnp.mean(iotas[1:])


def _as_vector(value):
    arr = jnp.asarray(value, dtype=jnp.float64)
    return arr.reshape((1,)) if int(arr.ndim) == 0 else jnp.ravel(arr)


def objectives_track_iota(objectives: Sequence[ObjectiveTerm], target_iota: float | None = None) -> bool:
    """Return true when optimization history should record mean iota."""

    return target_iota is not None or any(term.track_iota for term in objectives)


def qs_stage_modes(
    *,
    max_mode: int,
    use_mode_continuation: bool,
    continuation_nfev: int,
) -> list[int]:
    """Mode-continuation sequence for a user-facing fixed-boundary script.

    The omnigenity reference examples repeatedly optimize at a given active
    mode before increasing the boundary space.  We keep two mode-1 seed passes
    and then repeat the higher modes three times, so max-mode 3 gives
    ``[1, 1, 2, 2, 2, 3, 3, 3]``.
    """

    if bool(use_mode_continuation) and int(max_mode) > 1 and int(continuation_nfev) > 0:
        modes: list[int] = []
        for mode in range(1, int(max_mode) + 1):
            modes.extend([mode] * (2 if mode == 1 else 3))
        return modes
    return [int(max_mode)]


def qs_stage_budget(
    *,
    stage_mode: int,
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
) -> int:
    """Outer residual/Jacobian budget for one fixed-boundary stage."""

    return int(max_nfev) if int(stage_mode) == int(max_mode) else int(continuation_nfev)


def print_qs_problem_summary(
    *,
    method: str,
    max_nfev: int,
    use_mode_continuation: bool,
    use_ess: bool,
    ess_alpha: float,
    objectives: Sequence[ObjectiveTerm],
    specs: Sequence[vj.BoundaryParamSpec],
    x_scale: np.ndarray,
    optimizer,
    params0,
) -> None:
    """Print the problem summary used by the standalone examples."""

    print(f"Parameter space ({len(specs)} DOFs): {vj.boundary_param_names(specs)}")
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
        _remove_stale(stage_dir / "wout_initial.nc")
        _remove_stale(stage_dir / "wout_final.nc")
    if save_rerun_wouts:
        rerun = vj.run_fixed_boundary(str(stage_dir / "input.initial"), verbose=False)
        vj.write_wout_from_fixed_boundary_run(str(stage_dir / "wout_initial_rerun.nc"), rerun)
        rerun = vj.run_fixed_boundary(str(stage_dir / "input.final"), verbose=False)
        vj.write_wout_from_fixed_boundary_run(str(stage_dir / "wout_final_rerun.nc"), rerun)
    else:
        _remove_stale(stage_dir / "wout_initial_rerun.nc")
        _remove_stale(stage_dir / "wout_final_rerun.nc")


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
    plot: bool = True,
    save_rerun_wouts: bool = False,
) -> None:
    """Save initial/final inputs, wouts, history, and plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _initial_mode, initial_optimizer, initial_params0, initial_result = stage_records[0]
    initial_optimizer.save_input(output_dir / "input.initial", initial_params0)
    initial_optimizer.save_wout(
        output_dir / "wout_initial.nc",
        initial_params0,
        state=initial_result.get("_state_initial"),
    )
    if save_rerun_wouts:
        rerun = vj.run_fixed_boundary(str(output_dir / "input.initial"), verbose=False)
        vj.write_wout_from_fixed_boundary_run(str(output_dir / "wout_initial_rerun.nc"), rerun)
    else:
        _remove_stale(output_dir / "wout_initial_rerun.nc")

    final_optimizer.save_input(output_dir / "input.final", final_result["x"])
    final_optimizer.save_wout(
        output_dir / "wout_final.nc",
        final_result["x"],
        state=final_result.get("_state_final"),
    )
    if save_rerun_wouts:
        rerun = vj.run_fixed_boundary(str(output_dir / "input.final"), verbose=False)
        vj.write_wout_from_fixed_boundary_run(str(output_dir / "wout_final_rerun.nc"), rerun)
    else:
        _remove_stale(output_dir / "wout_final_rerun.nc")

    history = final_result["_history_dump"]
    history["label"] = label
    if target_aspect is not None:
        history["target_aspect"] = float(target_aspect)
    if target_iota is not None:
        history["target_iota"] = float(target_iota)
    if iota_abs_min is not None:
        history["iota_abs_min"] = float(iota_abs_min)
    final_optimizer.save_history(output_dir / "history.json", final_result)

    if plot:
        print("\nGenerating plots ...")
        vj.plot_qh_optimization(
            output_dir / "wout_initial.nc",
            output_dir / "wout_final.nc",
            output_dir / "history.json",
            outdir=output_dir,
        )
        print(f"Done. Results saved to {output_dir}/")


def combine_qs_stage_histories(
    *,
    label: str,
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
    stage_modes,
    stage_records,
) -> dict | None:
    """Merge per-stage histories into one optimization history."""

    if len(stage_records) <= 1:
        return None

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
                    int(max_nfev) if int(mode) == int(max_mode) else int(continuation_nfev)
                    for mode in stage_modes
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
        }
    )
    if combined_entries and "iota" in combined_entries[0] and "iota" in combined_entries[-1]:
        out["iota_initial"] = float(combined_entries[0]["iota"])
        out["iota_final"] = float(combined_entries[-1]["iota"])
    return out


def _remove_stale(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

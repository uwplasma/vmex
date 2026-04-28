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
from vmec_jax.config import config_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.optimization import rebuild_indata_with_resolution
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


@dataclass(frozen=True)
class FixedBoundaryQSConfig:
    """Run controls shared by the fixed-boundary QA/QH/QP examples."""

    input_file: Path
    output_dir: Path
    max_mode: int
    max_nfev: int
    vmec_mpol: int = 5
    vmec_ntor: int = 5
    continuation_nfev: int = 10
    use_mode_continuation: bool = True
    use_ess: bool = False
    ess_alpha: float = 2.5
    method: str = "scipy"
    scipy_tr_solver: str | None = "lsmr"
    scipy_lsmr_maxiter: int | None = None
    ftol: float = 1.0e-3
    gtol: float = 1.0e-3
    xtol: float = 1.0e-3
    inner_max_iter: int = 0
    inner_ftol: float = 0.0
    trial_max_iter: int = 300
    trial_ftol: float = 1.0e-10
    solver_device: str | None = None
    target_aspect: float | None = None
    target_iota: float | None = None
    label: str = "Optimisation"
    save_stage_inputs: bool = True
    save_stage_wouts: bool = False
    save_rerun_wouts: bool = False
    plot: bool = True


@dataclass(frozen=True)
class _StageBundle:
    mode: int
    ctx: StageContext
    optimizer: vj.FixedBoundaryExactOptimizer
    x_scale: np.ndarray
    iota_fn: Callable | None


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


def abs_mean_iota_floor_objective(target: float, weight: float = 1.0) -> ObjectiveTerm:
    """Lower-bound penalty enforcing ``abs(mean_iota) >= target``."""

    def _evaluate(ctx: StageContext, state):
        return jnp.minimum(jnp.abs(mean_iota(ctx, state)) - float(target), 0.0)

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


def run_qs_optimization(
    config: FixedBoundaryQSConfig,
    objectives: Sequence[ObjectiveTerm],
) -> dict:
    """Run a fixed-boundary exact optimization from a compact objective list."""

    if not objectives:
        raise ValueError("At least one objective term is required.")

    print(f"Loading {config.input_file.name} ...")
    cfg, indata = vj.load_config(str(config.input_file))
    indata = rebuild_indata_with_resolution(
        indata,
        mpol=int(config.vmec_mpol),
        ntor=int(config.vmec_ntor),
    )
    cfg = config_from_indata(indata)

    stage_modes = (
        list(range(1, int(config.max_mode) + 1))
        if (
            bool(config.use_mode_continuation)
            and int(config.max_mode) > 1
            and int(config.continuation_nfev) > 0
        )
        else [int(config.max_mode)]
    )

    stage_records = []
    params_stage = None
    prev_specs = None

    for stage_mode in stage_modes:
        stage = _build_stage(config, cfg, indata, stage_mode, objectives)
        params0 = (
            np.zeros(len(stage.ctx.specs), dtype=float)
            if params_stage is None
            else vj.lift_boundary_params(prev_specs, params_stage, stage.ctx.specs)
        )
        nfev = int(config.max_nfev) if stage_mode == int(config.max_mode) else int(config.continuation_nfev)

        if stage_mode == int(config.max_mode):
            _print_problem_summary(config, objectives, stage, params0)
        else:
            print(f"Stage {stage_mode} -> {stage_mode + 1} continuation seed (budget={nfev}) ...")

        result = stage.optimizer.run(
            params0,
            method=config.method,
            max_nfev=nfev,
            ftol=config.ftol,
            gtol=config.gtol,
            xtol=config.xtol,
            x_scale=stage.x_scale,
            verbose=1 if stage_mode == int(config.max_mode) else 0,
            iota_fn=stage.iota_fn,
            target_iota=config.target_iota,
            target_aspect=config.target_aspect,
            scipy_tr_solver=config.scipy_tr_solver,
            scipy_lsmr_maxiter=config.scipy_lsmr_maxiter,
        )
        _save_stage_artifacts(
            config,
            config.output_dir / f"stage_{stage_mode:02d}",
            stage.optimizer,
            params0,
            result["x"],
            result,
        )
        stage_records.append((stage_mode, stage, params0, result))
        prev_specs = stage.ctx.specs
        params_stage = result["x"]

    final_mode, final_stage, _final_params0, final_result = stage_records[-1]
    del final_mode
    history = _combined_history(config, stage_modes, stage_records)
    if history is not None:
        final_result["_history_dump"] = history

    _print_final_summary(config, final_result)
    _save_final_outputs(config, stage_records, final_stage, final_result)
    return final_result


def _build_stage(
    config: FixedBoundaryQSConfig,
    cfg,
    indata,
    max_mode: int,
    objectives: Sequence[ObjectiveTerm],
) -> _StageBundle:
    static = vj.build_static(cfg)
    boundary = vj.boundary_from_indata(indata, static.modes, apply_m1_constraint=False)
    stage_indata, static, boundary = vj.extend_boundary_for_max_mode(
        indata,
        static,
        boundary,
        int(max_mode),
    )
    boundary_input = vj.boundary_input_from_indata(stage_indata, static.modes)
    specs = vj.boundary_param_specs(
        boundary_input,
        static.modes,
        max_mode=int(max_mode),
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    guess = initial_guess_from_boundary(static, boundary, stage_indata, vmec_project=True)
    geom = eval_geom(guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    flux = vj.flux_profiles_from_indata(stage_indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))
    ctx = StageContext(
        static=static,
        indata=stage_indata,
        boundary_input=boundary_input,
        specs=specs,
        signgs=signgs,
        flux=flux,
        pressure=pressure,
    )

    def residuals_from_state(state):
        return jnp.concatenate([term.residual(ctx, state) for term in objectives])

    totals = [term.total for term in objectives if term.total is not None]
    residuals_from_state._n_non_qs = 0
    residuals_from_state._qs_total_from_state = (
        lambda state: float(sum(float(total(ctx, state)) for total in totals))
        if totals
        else lambda state: 0.0
    )

    optimizer = vj.FixedBoundaryExactOptimizer(
        static,
        stage_indata,
        boundary,
        specs,
        residuals_from_state,
        boundary_input=boundary_input,
        inner_max_iter=config.inner_max_iter,
        inner_ftol=config.inner_ftol,
        trial_max_iter=config.trial_max_iter,
        trial_ftol=config.trial_ftol,
        solver_device=config.solver_device,
    )
    x_scale = (
        vj.create_x_scale(specs, alpha=float(config.ess_alpha))
        if config.use_ess
        else np.ones(len(specs), dtype=float)
    )
    iota_fn = (lambda state: float(mean_iota(ctx, state))) if _tracks_iota(objectives, config) else None
    return _StageBundle(
        mode=int(max_mode),
        ctx=ctx,
        optimizer=optimizer,
        x_scale=np.asarray(x_scale, dtype=float),
        iota_fn=iota_fn,
    )


def _as_vector(value):
    arr = jnp.asarray(value, dtype=jnp.float64)
    return arr.reshape((1,)) if int(arr.ndim) == 0 else jnp.ravel(arr)


def _tracks_iota(objectives: Sequence[ObjectiveTerm], config: FixedBoundaryQSConfig) -> bool:
    return config.target_iota is not None or any(term.track_iota for term in objectives)


def _print_problem_summary(config, objectives, stage: _StageBundle, params0) -> None:
    print(f"Parameter space ({len(stage.ctx.specs)} DOFs): {vj.boundary_param_names(stage.ctx.specs)}")
    print("Objectives:")
    for term in objectives:
        print(f"  - {term.name}: target={term.target}, weight={term.weight}")
    if config.use_ess:
        print(
            f"ESS scales (alpha={config.ess_alpha}): "
            f"min={stage.x_scale.min():.3f}  max={stage.x_scale.max():.3f}"
        )
    else:
        print("ESS disabled - uniform scales.")
    print(f"Aspect ratio (initial):        {stage.optimizer.aspect_ratio(params0):.6f}")
    print(f"Field objective (initial):     {stage.optimizer.quasisymmetry_objective(params0):.6e}")
    print(
        f"Running {config.method} "
        f"(max_nfev={config.max_nfev}, continuation={config.use_mode_continuation}) ..."
    )


def _print_final_summary(config: FixedBoundaryQSConfig, result: dict) -> None:
    hist = result.get("_history_dump", {})
    print(f"\nTermination: {result['message']}")
    print(f"Aspect ratio (final):          {float(hist.get('aspect_final', float('nan'))):.6f}")
    if "iota_final" in hist:
        target = "" if config.target_iota is None else f"  target={config.target_iota:.6f}"
        print(f"Mean iota (final):             {float(hist['iota_final']):.6f}{target}")
    print(f"Field objective (final):       {float(hist.get('qs_final', float('nan'))):.6e}")
    print(f"Total objective (final):       {float(hist.get('objective_final', float('nan'))):.6e}")
    obj0 = hist.get("objective_initial")
    objf = hist.get("objective_final")
    if obj0 is not None and float(obj0) > 0.0 and objf is not None:
        print(f"Objective reduction:           {100.0 * (1.0 - float(objf) / float(obj0)):.1f}%")


def _save_stage_artifacts(
    config: FixedBoundaryQSConfig,
    stage_dir: Path,
    opt,
    params_initial,
    params_final,
    result,
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    if config.save_stage_inputs:
        opt.save_input(stage_dir / "input.initial", params_initial)
        opt.save_input(stage_dir / "input.final", params_final)
    if config.save_stage_wouts:
        opt.save_wout(stage_dir / "wout_initial.nc", params_initial, state=result.get("_state_initial"))
        opt.save_wout(stage_dir / "wout_final.nc", params_final, state=result.get("_state_final"))
    else:
        _remove_stale(stage_dir / "wout_initial.nc")
        _remove_stale(stage_dir / "wout_final.nc")
    if config.save_rerun_wouts:
        rerun = vj.run_fixed_boundary(str(stage_dir / "input.initial"), verbose=False)
        vj.write_wout_from_fixed_boundary_run(str(stage_dir / "wout_initial_rerun.nc"), rerun)
        rerun = vj.run_fixed_boundary(str(stage_dir / "input.final"), verbose=False)
        vj.write_wout_from_fixed_boundary_run(str(stage_dir / "wout_final_rerun.nc"), rerun)
    else:
        _remove_stale(stage_dir / "wout_initial_rerun.nc")
        _remove_stale(stage_dir / "wout_final_rerun.nc")


def _save_final_outputs(
    config: FixedBoundaryQSConfig,
    stage_records,
    final_stage: _StageBundle,
    final_result: dict,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _initial_mode, initial_stage, initial_params0, initial_result = stage_records[0]
    initial_stage.optimizer.save_input(config.output_dir / "input.initial", initial_params0)
    initial_stage.optimizer.save_wout(
        config.output_dir / "wout_initial.nc",
        initial_params0,
        state=initial_result.get("_state_initial"),
    )
    if config.save_rerun_wouts:
        rerun = vj.run_fixed_boundary(str(config.output_dir / "input.initial"), verbose=False)
        vj.write_wout_from_fixed_boundary_run(str(config.output_dir / "wout_initial_rerun.nc"), rerun)
    else:
        _remove_stale(config.output_dir / "wout_initial_rerun.nc")

    final_stage.optimizer.save_input(config.output_dir / "input.final", final_result["x"])
    final_stage.optimizer.save_wout(
        config.output_dir / "wout_final.nc",
        final_result["x"],
        state=final_result.get("_state_final"),
    )
    if config.save_rerun_wouts:
        rerun = vj.run_fixed_boundary(str(config.output_dir / "input.final"), verbose=False)
        vj.write_wout_from_fixed_boundary_run(str(config.output_dir / "wout_final_rerun.nc"), rerun)
    else:
        _remove_stale(config.output_dir / "wout_final_rerun.nc")

    history = final_result["_history_dump"]
    history["label"] = config.label
    if config.target_aspect is not None:
        history["target_aspect"] = float(config.target_aspect)
    if config.target_iota is not None:
        history["target_iota"] = float(config.target_iota)
    final_stage.optimizer.save_history(config.output_dir / "history.json", final_result)

    if config.plot:
        print("\nGenerating plots ...")
        vj.plot_qh_optimization(
            config.output_dir / "wout_initial.nc",
            config.output_dir / "wout_final.nc",
            config.output_dir / "history.json",
            outdir=config.output_dir,
        )
        print(f"Done. Results saved to {config.output_dir}/")


def _combined_history(config: FixedBoundaryQSConfig, stage_modes, stage_records) -> dict | None:
    if len(stage_records) <= 1:
        return None

    combined_entries = []
    stage_boundaries = []
    wall_offset = 0.0
    nfev_total = 0
    njev_total = 0
    for idx, (_mode, _stage, _params0, result) in enumerate(stage_records):
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
            "label": config.label,
            "max_nfev": int(
                sum(
                    int(config.max_nfev) if mode == int(config.max_mode) else int(config.continuation_nfev)
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

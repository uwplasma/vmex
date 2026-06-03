#!/usr/bin/env python
"""Small utilities for finite-beta stage-one optimization examples.

The finite-beta example scripts intentionally use
``FixedBoundaryExactOptimizer`` directly instead of ``least_squares_solve``:
each stage builds local finite-pressure/current residual closures before the
shared workflow-object abstraction is useful.  The scripts still keep the
SIMSOPT-like flow visible: load the input deck, build VMEC static data, choose
boundary DOFs, define ``residuals_from_state``, run, and save outputs.  This
module only contains lightweight diagnostics and file I/O helpers so the
scripts do not hide the scientific setup behind a config object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import vmec_jax as vj
from vmec_jax._compat import jnp
from vmec_jax.wout import equilibrium_iota_profiles_from_state


@dataclass(frozen=True)
class FiniteBetaStageSummary:
    """Serializable diagnostics for one finite-beta optimizer stage."""

    mode: int
    objective_initial: float | None
    objective_final: float | None
    qs_initial: float | None
    qs_final: float | None
    aspect_initial: float | None
    aspect_final: float | None
    iota_initial: float | None
    iota_final: float | None
    nfev: int | None
    njev: int | None
    nit: int | None
    max_nfev: int | None
    success: bool | None
    status: int | None
    message: str | None
    method: str | None
    solver_device: str | None
    total_wall_time_s: float | None
    selected_best_exact_point: bool | None
    n_params: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class FiniteBetaStageRecord:
    """Named view of the explicit finite-beta stage tuple.

    The examples still instantiate and run ``FixedBoundaryExactOptimizer``
    directly; this adapter only gives tests and users stable attribute names.
    """

    mode: int
    optimizer: Any
    params_initial: np.ndarray
    result: dict

    @property
    def params_final(self) -> np.ndarray:
        """Optimized boundary parameter vector for this stage."""

        return np.asarray(self.result["x"], dtype=float)

    @property
    def history(self) -> dict:
        """Optimizer history dictionary written by ``save_history``."""

        return self.result.get("_history_dump", {})

    @property
    def summary(self) -> FiniteBetaStageSummary:
        """Small structured summary for this stage."""

        return finite_beta_stage_summary(self)


@dataclass(frozen=True)
class FiniteBetaStage1Result:
    """Structured view of a finite-beta stage-one run.

    This deliberately wraps raw optimizer objects and result dictionaries
    instead of hiding them behind a higher-level solve API.
    """

    stage_records: tuple[Any, ...]

    @property
    def stages(self) -> tuple[FiniteBetaStageRecord, ...]:
        """Named stage records in continuation order."""

        return tuple(_as_stage_record(record) for record in self.stage_records)

    @property
    def initial_stage(self) -> FiniteBetaStageRecord:
        """First finite-beta stage."""

        return self.stages[0]

    @property
    def final_stage(self) -> FiniteBetaStageRecord:
        """Last finite-beta stage."""

        return self.stages[-1]

    @property
    def final_optimizer(self):
        """Raw ``FixedBoundaryExactOptimizer`` for the last stage."""

        return self.final_stage.optimizer

    @property
    def final_result(self) -> dict:
        """Raw optimizer result dictionary for the last stage."""

        return self.final_stage.result

    @property
    def final_params(self) -> np.ndarray:
        """Optimized boundary parameter vector for the last stage."""

        return self.final_stage.params_final

    @property
    def stage_summaries(self) -> tuple[FiniteBetaStageSummary, ...]:
        """Structured summaries for each mode-continuation stage."""

        return tuple(stage.summary for stage in self.stages)

    @property
    def final_summary(self) -> FiniteBetaStageSummary:
        """Structured summary for the final stage."""

        return self.stage_summaries[-1]

    @property
    def summary(self) -> dict[str, Any]:
        """JSON-friendly run summary."""

        return {
            "stages": [stage.to_dict() for stage in self.stage_summaries],
            "final": self.final_summary.to_dict(),
        }


def pressure_profile(indata, static):
    """Pressure profile on the VMEC full radial mesh."""

    prof = vj.eval_profiles(indata, jnp.asarray(static.s))
    return jnp.asarray(prof.get("pressure", jnp.zeros_like(jnp.asarray(static.s))))


def finite_beta_profile_bundle(beta_percent: float):
    """Standard density/temperature/pressure profiles for finite-beta examples.

    This is the vmec_jax counterpart of the SIMSOPT stage-one setup used in
    ``single_stage_optimization_finite_beta``: density and temperature are
    polynomial profiles, pressure is ``e * (ne*Te + ni*Ti)``, and the same
    coefficients are passed to the Redl bootstrap-current residual.
    """

    return vj.standard_finite_beta_profiles(float(beta_percent))


def apply_finite_beta_pressure_profile(indata, *, beta_percent: float):
    """Return ``indata`` with the standard finite-beta pressure profile."""

    bundle = finite_beta_profile_bundle(beta_percent)
    return vj.with_pressure_profile(indata, bundle.pressure_pa, pres_scale=1.0)


def mean_abs_iota(state, *, static, indata, signgs: int):
    """Mean absolute full-mesh rotational transform, excluding the axis."""

    _chips, _iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    iotaf = jnp.asarray(iotaf, dtype=jnp.float64)
    return jnp.mean(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0)


def finite_beta_stage_modes(
    *,
    max_mode: int,
    use_mode_continuation: bool,
    continuation_nfev: int,
) -> list[int]:
    """Mode-continuation sequence for finite-beta examples."""

    if bool(use_mode_continuation) and int(max_mode) > 1 and int(continuation_nfev) > 0:
        return list(range(1, int(max_mode) + 1))
    return [int(max_mode)]


def finite_beta_stage_budget(
    *,
    stage_mode: int,
    max_mode: int,
    max_nfev: int,
    continuation_nfev: int,
) -> int:
    """Outer residual/Jacobian budget for one finite-beta stage."""

    return int(max_nfev) if int(stage_mode) == int(max_mode) else int(continuation_nfev)


def finite_beta_stage_record(
    *,
    mode: int,
    optimizer,
    params_initial,
    result: dict,
) -> FiniteBetaStageRecord:
    """Build a named stage record without changing the explicit workflow."""

    return FiniteBetaStageRecord(
        mode=int(mode),
        optimizer=optimizer,
        params_initial=np.asarray(params_initial, dtype=float),
        result=result,
    )


def finite_beta_stage_summary(stage_record) -> FiniteBetaStageSummary:
    """Extract a structured summary from a finite-beta stage record."""

    stage = _as_stage_record(stage_record)
    result = stage.result
    hist = result.get("_history_dump", {})
    return FiniteBetaStageSummary(
        mode=int(stage.mode),
        objective_initial=_float_or_none(hist.get("objective_initial")),
        objective_final=_float_or_none(hist.get("objective_final", result.get("objective"))),
        qs_initial=_float_or_none(hist.get("qs_initial")),
        qs_final=_float_or_none(hist.get("qs_final")),
        aspect_initial=_float_or_none(hist.get("aspect_initial")),
        aspect_final=_float_or_none(hist.get("aspect_final")),
        iota_initial=_float_or_none(hist.get("iota_initial")),
        iota_final=_float_or_none(hist.get("iota_final", _last_history_value(hist, "iota"))),
        nfev=_int_or_none(hist.get("nfev", result.get("nfev"))),
        njev=_int_or_none(hist.get("njev", result.get("njev"))),
        nit=_int_or_none(hist.get("nit", result.get("nit"))),
        max_nfev=_int_or_none(hist.get("max_nfev")),
        success=_bool_or_none(hist.get("success", result.get("success"))),
        status=_int_or_none(result.get("status")),
        message=_str_or_none(hist.get("message", result.get("message"))),
        method=_str_or_none(hist.get("method")),
        solver_device=_str_or_none(hist.get("solver_device")),
        total_wall_time_s=_float_or_none(hist.get("total_wall_time_s")),
        selected_best_exact_point=_bool_or_none(hist.get("selected_best_exact_point")),
        n_params=int(stage.params_initial.size),
    )


def finite_beta_stage1_result(stage_records) -> FiniteBetaStage1Result:
    """Return a structured view of finite-beta stage records."""

    records = tuple(stage_records)
    if not records:
        raise ValueError("finite-beta stage-one result requires at least one stage record")
    return FiniteBetaStage1Result(stage_records=records)


def save_stage_artifacts(
    *,
    stage_dir: Path,
    optimizer,
    params_initial,
    params_final,
    result,
    save_inputs: bool = True,
    save_wouts: bool = False,
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
    optimizer.save_history(stage_dir / "history.json", result)


def save_final_outputs(
    *,
    output_dir: Path,
    stage_records,
    final_optimizer,
    final_result: dict,
) -> None:
    """Save initial/final inputs, wouts, and history."""

    output_dir.mkdir(parents=True, exist_ok=True)
    initial_stage = _as_stage_record(stage_records[0])
    initial_optimizer = initial_stage.optimizer
    initial_params0 = initial_stage.params_initial
    initial_result = initial_stage.result
    initial_optimizer.save_input(output_dir / "input.initial", initial_params0)
    initial_optimizer.save_wout(
        output_dir / "wout_initial.nc",
        initial_params0,
        state=initial_result.get("_state_initial"),
    )
    final_optimizer.save_input(output_dir / "input.final", final_result["x"])
    final_optimizer.save_wout(
        output_dir / "wout_final.nc",
        final_result["x"],
        state=final_result.get("_state_final"),
    )
    final_optimizer.save_history(output_dir / "history.json", final_result)


def print_final_summary(
    result: dict | FiniteBetaStage1Result | FiniteBetaStageSummary,
) -> None:
    """Print finite-beta optimization diagnostics."""

    if isinstance(result, FiniteBetaStage1Result):
        summary = result.final_summary
    elif isinstance(result, FiniteBetaStageSummary):
        summary = result
    else:
        summary = finite_beta_stage_summary((0, None, np.zeros(0, dtype=float), result))

    print(f"\nTermination: {summary.message}")
    if summary.objective_final is not None:
        print(f"Final objective: {summary.objective_final:.6e}")
    if summary.aspect_final is not None:
        print(f"Final aspect:    {summary.aspect_final:.6f}")
    if summary.iota_final is not None:
        print(f"Final |iota|:    {summary.iota_final:.6f}")


def _as_stage_record(record) -> FiniteBetaStageRecord:
    if isinstance(record, FiniteBetaStageRecord):
        return record
    if isinstance(record, tuple) and len(record) == 4:
        mode, optimizer, params_initial, result = record
        return finite_beta_stage_record(
            mode=mode,
            optimizer=optimizer,
            params_initial=params_initial,
            result=result,
        )
    if all(hasattr(record, name) for name in ("mode", "optimizer", "params_initial", "result")):
        return finite_beta_stage_record(
            mode=record.mode,
            optimizer=record.optimizer,
            params_initial=record.params_initial,
            result=record.result,
        )
    raise TypeError(
        "stage record must be (mode, optimizer, params_initial, result) or expose matching attributes"
    )


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(np.asarray(value))
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _str_or_none(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _last_history_value(history: dict, key: str):
    entries = history.get("history") or ()
    if not entries:
        return None
    return entries[-1].get(key)


def _remove_stale(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

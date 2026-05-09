#!/usr/bin/env python
"""Small utilities for finite-beta stage-one optimization examples.

The finite-beta example scripts keep the optimization workflow visible:
load the input deck, build VMEC static data, choose boundary DOFs, define
``residuals_from_state``, instantiate ``FixedBoundaryExactOptimizer``, run, and
save outputs.  This module only contains lightweight diagnostics and file I/O
helpers so the scripts do not hide the scientific setup behind a config object.
"""

from __future__ import annotations

from pathlib import Path

import vmec_jax as vj
from vmec_jax._compat import jnp
from vmec_jax.wout import equilibrium_iota_profiles_from_state


def pressure_profile(indata, static):
    """Pressure profile on the VMEC full radial mesh."""

    prof = vj.eval_profiles(indata, jnp.asarray(static.s))
    return jnp.asarray(prof.get("pressure", jnp.zeros_like(jnp.asarray(static.s))))


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
    _initial_mode, initial_optimizer, initial_params0, initial_result = stage_records[0]
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


def print_final_summary(result: dict) -> None:
    """Print finite-beta optimization diagnostics."""

    hist = result["_history_dump"]
    print(f"\nTermination: {result['message']}")
    print(f"Final objective: {hist['objective_final']:.6e}")
    print(f"Final aspect:    {hist['aspect_final']:.6f}")
    if hist["history"] and "iota" in hist["history"][-1]:
        print(f"Final |iota|:    {hist['history'][-1]['iota']:.6f}")


def _remove_stale(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

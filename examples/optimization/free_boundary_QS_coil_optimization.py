#!/usr/bin/env python
"""Phase-1 smoke scaffold for free-boundary coil-only optimization.

This is intentionally *not* a production QS optimization.  The only optimizer
variables are direct-coil currents and selected direct-coil Fourier dofs.  The
plasma boundary coefficients from the VMEC input deck are never included in the
optimization vector.

The phase-1 objective is deliberately cheap:

* VMEC residual from a tiny direct-coil free-boundary solve,
* aspect-ratio target,
* mean-iota target.

Phase-2 note: replace or augment this proxy with Boozer/QS residuals once that
path is cheap enough for this single-stage free-boundary coil loop and complete
full-loop finite-difference checks pass.

Run a minimal smoke from the repository root:

    python examples/optimization/free_boundary_QS_coil_optimization.py --smoke

If ESSOS assets are not available, the default ESSOS provider exits with code
77 and a helpful message.  For a provider-only development smoke that does not
need ESSOS, pass ``--provider circle``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import jax, jnp
from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import CoilFieldParams, from_essos_coils
from vmec_jax.external_fields.coils_jax import coil_current_norm, coil_lengths
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.robust_coils import (
    CoilPerturbationSample,
    aggregate_risk,
    perturb_coil_params,
    sample_coil_perturbations,
)
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


SKIP_EXIT_CODE = 77
DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "free_boundary_QS_coil_optimization"
DEFAULT_ESSOS_COIL_JSON = "ESSOS_biot_savart_LandremanPaulQA.json"


class SkipExample(RuntimeError):
    """Raised when optional external assets needed by the example are absent."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")


def candidate_essos_input_dirs() -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("ESSOS_INPUT_DIR"):
        candidates.append(Path(os.environ["ESSOS_INPUT_DIR"]).expanduser())
    candidates.extend(
        [
            REPO_ROOT.parent / "ESSOS" / "examples" / "input_files",
            REPO_ROOT.parent / "ESSOS_mgrid_pr" / "examples" / "input_files",
        ]
    )
    return candidates


def find_essos_coil_json() -> Path:
    for directory in candidate_essos_input_dirs():
        path = directory / DEFAULT_ESSOS_COIL_JSON
        if path.exists():
            return path
    searched = "\n  ".join(str(path) for path in candidate_essos_input_dirs())
    raise SkipExample(
        f"Missing ESSOS coil asset {DEFAULT_ESSOS_COIL_JSON}. Set ESSOS_INPUT_DIR "
        f"to an ESSOS examples/input_files directory. Searched:\n  {searched}"
    )


def load_essos_provider(coils_json: Path | None, *, chunk_size: int, current_scale: float) -> tuple[CoilFieldParams, dict[str, Any]]:
    try:
        from essos.coils import Coils_from_json
    except Exception as exc:  # pragma: no cover - depends on optional ESSOS.
        raise SkipExample(
            "ESSOS is not importable. Install ESSOS or set PYTHONPATH to an ESSOS checkout, "
            "or use --provider circle for a synthetic direct-coil smoke."
        ) from exc

    resolved = coils_json if coils_json is not None else find_essos_coil_json()
    if not resolved.exists():
        raise SkipExample(f"ESSOS coil JSON does not exist: {resolved}")
    coils = Coils_from_json(str(resolved))
    params = from_essos_coils(coils, chunk_size=chunk_size)
    params = replace(params, current_scale=float(params.current_scale) * float(current_scale))
    metadata = {
        "provider": "essos",
        "coils_json": resolved,
        "n_base_coils": int(np.asarray(params.base_currents).size),
        "n_segments": int(params.n_segments),
        "nfp": int(params.nfp),
        "stellsym": bool(params.stellsym),
        "current_scale_multiplier": float(current_scale),
    }
    return params, metadata


def make_circle_provider(*, current_scale: float) -> tuple[CoilFieldParams, dict[str, Any]]:
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(1.4)
    dofs = dofs.at[0, 1, 1].set(1.4)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.0]),
        n_segments=96,
        current_scale=float(current_scale),
    )
    return params, {"provider": "circle", "current_scale_multiplier": float(current_scale)}


def make_free_boundary_indata(
    input_path: Path,
    output_path: Path,
    *,
    vmec_max_iter: int,
    ftol: float,
    ns: int,
    mpol: int,
    ntor: int,
    nzeta: int,
    pressure_scale: float,
) -> Path:
    indata = deepcopy(read_indata(input_path))
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": "DIRECT_COILS",
            "EXTCUR": [1.0],
            "NS_ARRAY": [int(ns)],
            "NITER_ARRAY": [int(vmec_max_iter)],
            "FTOL_ARRAY": [float(ftol)],
            "NITER": int(vmec_max_iter),
            "FTOL": float(ftol),
            "MPOL": int(mpol),
            "NTOR": int(ntor),
            "NZETA": int(nzeta),
            "NTHETA": 0,
            "NVACSKIP": max(1, int(nzeta)),
            "PRES_SCALE": float(pressure_scale),
            "AM": [1.0, -1.0],
        }
    )
    write_indata(output_path, indata)
    return output_path


def select_coil_variables(
    params: CoilFieldParams,
    *,
    max_current_vars: int,
    max_fourier_vars: int,
) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...]]]]:
    base_currents = np.asarray(params.base_currents, dtype=float)
    base_dofs = np.asarray(params.base_curve_dofs, dtype=float)
    variables: list[tuple[str, tuple[int, ...]]] = []

    for i in range(min(int(max_current_vars), base_currents.size)):
        variables.append(("current", (i,)))

    if max_fourier_vars > 0:
        nonzero_dofs = np.argwhere(np.abs(base_dofs) > 0.0)
        dof_indices = nonzero_dofs[: int(max_fourier_vars)]
        for index in dof_indices:
            variables.append(("fourier_dof", tuple(int(i) for i in index)))

    return np.zeros(len(variables), dtype=float), variables


def apply_coil_variables(
    base_params: CoilFieldParams,
    x: np.ndarray,
    variables: list[tuple[str, tuple[int, ...]]],
    *,
    current_step: float,
    dof_step: float,
) -> CoilFieldParams:
    currents = np.asarray(base_params.base_currents, dtype=float).copy()
    dofs = np.asarray(base_params.base_curve_dofs, dtype=float).copy()

    for value, (kind, index) in zip(np.asarray(x, dtype=float), variables, strict=True):
        if kind == "current":
            i = index[0]
            currents[i] *= 1.0 + float(current_step) * float(value)
        elif kind == "fourier_dof":
            dofs[index] += float(dof_step) * float(value)
        else:  # pragma: no cover - defensive programming for future variable kinds.
            raise ValueError(f"unknown coil variable kind {kind!r}")

    return base_params.with_arrays(base_curve_dofs=jnp.asarray(dofs), base_currents=jnp.asarray(currents))


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(np.asarray(value))
    except Exception:
        return None
    return result if np.isfinite(result) else None


def array_history(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        return [float(v) for v in np.asarray(value, dtype=float).reshape(-1)]
    except Exception:
        return []


def run_direct_free_boundary(
    input_path: Path,
    params: CoilFieldParams,
    *,
    vmec_max_iter: int,
    activate_fsq: float,
) -> tuple[Any, float]:
    start = time.perf_counter()
    run = run_free_boundary(
        input_path,
        max_iter=int(vmec_max_iter),
        multigrid=False,
        verbose=False,
        jit_forces=False,
        external_field_provider_kind="direct_coils",
        external_field_provider_params=params,
        free_boundary_activate_fsq=float(activate_fsq),
    )
    return run, time.perf_counter() - start


def summarize_run(
    run: Any,
    params: CoilFieldParams,
    *,
    objective: float,
    wall_s: float,
    target_aspect: float,
    target_iota: float,
) -> dict[str, Any]:
    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    fsqr = float_or_none(diag.get("final_fsqr"))
    fsqz = float_or_none(diag.get("final_fsqz"))
    fsql = float_or_none(diag.get("final_fsql"))
    residual_proxy = sum(value for value in (fsqr, fsqz, fsql) if value is not None)
    result = run.result

    aspect = None
    mean_iota = None
    try:
        aspect = float(np.asarray(equilibrium_aspect_ratio_from_state(state=run.state, static=run.static)))
    except Exception:
        pass
    try:
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
        )
        iota_arr = np.asarray(iotas, dtype=float)
        mean_iota = float(np.nanmean(iota_arr[1:] if iota_arr.size > 1 else iota_arr))
    except Exception:
        pass

    return {
        "objective": float(objective),
        "wall_s": float(wall_s),
        "vmec_n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        "fsqr": fsqr,
        "fsqz": fsqz,
        "fsql": fsql,
        "residual_proxy": float(residual_proxy),
        "aspect": aspect,
        "target_aspect": float(target_aspect),
        "mean_iota": mean_iota,
        "target_iota": float(target_iota),
        "coil_current_norm": float(np.asarray(coil_current_norm(params))),
        "mean_coil_length": float(np.mean(np.asarray(coil_lengths(params), dtype=float))),
        "vmec_history": {
            "w": array_history(getattr(result, "w_history", None)),
            "fsqr2": array_history(getattr(result, "fsqr2_history", None)),
            "fsqz2": array_history(getattr(result, "fsqz2_history", None)),
            "fsql2": array_history(getattr(result, "fsql2_history", None)),
        },
    }


def objective_from_summary(
    summary: dict[str, Any],
    *,
    residual_weight: float,
    aspect_weight: float,
    iota_weight: float,
) -> float:
    residual = float(summary.get("residual_proxy") or 0.0)
    aspect = summary.get("aspect")
    mean_iota = summary.get("mean_iota")
    aspect_penalty = 0.0 if aspect is None else (float(aspect) - float(summary["target_aspect"])) ** 2
    iota_penalty = 0.0 if mean_iota is None else (float(mean_iota) - float(summary["target_iota"])) ** 2
    return float(residual_weight) * residual + float(aspect_weight) * aspect_penalty + float(iota_weight) * iota_penalty


def robust_risk_method(method: str) -> str:
    if method == "smooth":
        return "smooth_max"
    return method


def robust_sample_at(samples: CoilPerturbationSample, index: int) -> CoilPerturbationSample:
    return CoilPerturbationSample(
        current_factors=samples.current_factors[index],
        displacement_xyz=samples.displacement_xyz[index],
        toroidal_phase=samples.toroidal_phase[index],
        centerline_dof_delta=samples.centerline_dof_delta[index],
    )


def optimize_coils(args: argparse.Namespace) -> dict[str, Any]:
    if args.provider == "essos":
        base_params, provider_metadata = load_essos_provider(
            args.coils_json,
            chunk_size=int(args.chunk_size),
            current_scale=float(args.current_scale),
        )
    elif args.provider == "circle":
        base_params, provider_metadata = make_circle_provider(current_scale=float(args.current_scale))
    else:
        raise ValueError(f"unknown provider {args.provider!r}")

    x0, variables = select_coil_variables(
        base_params,
        max_current_vars=int(args.max_current_vars),
        max_fourier_vars=int(args.max_fourier_vars),
    )
    if not variables:
        raise ValueError("No coil optimization variables selected. Increase --max-current-vars or --max-fourier-vars.")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = make_free_boundary_indata(
        args.input,
        outdir / "input.direct_coil_phase1_smoke",
        vmec_max_iter=int(args.vmec_max_iter),
        ftol=float(args.ftol),
        ns=int(args.ns),
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        nzeta=int(args.nzeta),
        pressure_scale=float(args.pressure_scale),
    )

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    robust_samples: CoilPerturbationSample | None = None
    robust_options: dict[str, Any] | None = None
    if int(args.robust_samples) < 0:
        raise ValueError("--robust-samples must be non-negative.")
    if int(args.robust_samples) > 0:
        if jax is None:  # pragma: no cover - JAX is a declared dependency in CI.
            raise RuntimeError("JAX is required for --robust-samples.")
        robust_samples = sample_coil_perturbations(
            jax.random.PRNGKey(int(args.robust_seed)),
            base_params,
            int(args.robust_samples),
            current_sigma=float(args.robust_current_sigma),
            displacement_sigma=float(args.robust_displacement_sigma),
            toroidal_phase_sigma=float(args.robust_toroidal_phase_sigma),
            centerline_sigma=float(args.robust_centerline_sigma),
            centerline_include_constant=bool(args.robust_centerline_include_constant),
        )
        robust_options = {
            "samples": int(args.robust_samples),
            "scenario_count_including_nominal": int(args.robust_samples) + 1,
            "risk": str(args.robust_risk),
            "aggregate_risk_method": robust_risk_method(str(args.robust_risk)),
            "risk_std_weight": float(args.robust_std_weight),
            "risk_temperature": float(args.robust_temperature),
            "seed": int(args.robust_seed),
            "current_sigma": float(args.robust_current_sigma),
            "displacement_sigma": float(args.robust_displacement_sigma),
            "toroidal_phase_sigma": float(args.robust_toroidal_phase_sigma),
            "centerline_sigma": float(args.robust_centerline_sigma),
            "centerline_include_constant": bool(args.robust_centerline_include_constant),
        }

    def evaluate(x: np.ndarray) -> float:
        nonlocal best
        eval_id = len(history)
        params = apply_coil_variables(
            base_params,
            x,
            variables,
            current_step=float(args.current_step),
            dof_step=float(args.dof_step),
        )

        if robust_samples is not None:
            scenario_entries: list[dict[str, Any]] = []
            scenario_objectives: list[float] = []
            nominal_run: Any | None = None
            for scenario_id in range(int(args.robust_samples) + 1):
                scenario_name = "nominal" if scenario_id == 0 else f"perturbation_{scenario_id - 1}"
                scenario_params = (
                    params
                    if scenario_id == 0
                    else perturb_coil_params(params, robust_sample_at(robust_samples, scenario_id - 1))
                )
                try:
                    run, wall_s = run_direct_free_boundary(
                        input_path,
                        scenario_params,
                        vmec_max_iter=int(args.vmec_max_iter),
                        activate_fsq=float(args.activate_fsq),
                    )
                    provisional = summarize_run(
                        run,
                        scenario_params,
                        objective=np.nan,
                        wall_s=wall_s,
                        target_aspect=float(args.target_aspect),
                        target_iota=float(args.target_iota),
                    )
                    scenario_objective = objective_from_summary(
                        provisional,
                        residual_weight=float(args.residual_weight),
                        aspect_weight=float(args.aspect_weight),
                        iota_weight=float(args.iota_weight),
                    )
                    provisional["objective"] = scenario_objective
                    if scenario_id == 0:
                        nominal_run = run
                    scenario_entries.append(
                        {
                            "scenario": scenario_name,
                            "perturbed": scenario_id > 0,
                            "summary": provisional,
                        }
                    )
                except Exception as exc:
                    scenario_objective = float(args.failure_objective)
                    scenario_entries.append(
                        {
                            "scenario": scenario_name,
                            "perturbed": scenario_id > 0,
                            "error": f"{type(exc).__name__}: {exc}",
                            "summary": {"objective": scenario_objective},
                        }
                    )
                    print(
                        f"eval={eval_id:03d} scenario={scenario_name} failed with "
                        f"{scenario_entries[-1]['error']}; returning {scenario_objective:.3e}",
                        flush=True,
                    )
                scenario_objectives.append(float(scenario_objective))

            objective = float(
                np.asarray(
                    aggregate_risk(
                        jnp.asarray(scenario_objectives, dtype=float),
                        robust_risk_method(str(args.robust_risk)),
                        std_weight=float(args.robust_std_weight),
                        temperature=float(args.robust_temperature),
                    )
                )
            )
            nominal_summary = dict(scenario_entries[0]["summary"])
            nominal_summary.update(
                {
                    "objective": objective,
                    "nominal_objective": float(scenario_objectives[0]),
                    "scenario_objectives": scenario_objectives,
                    "robust_samples": int(args.robust_samples),
                    "robust_risk": str(args.robust_risk),
                }
            )
            entry = {
                "eval": eval_id,
                "x": np.asarray(x, dtype=float).tolist(),
                "variables": [{"kind": kind, "index": index} for kind, index in variables],
                "summary": nominal_summary,
                "scenarios": scenario_entries,
            }
            if best is None or objective < float(best["summary"]["objective"]):
                best = entry
                if nominal_run is not None:
                    write_wout_from_fixed_boundary_run(outdir / "wout_best_direct_coil_phase1.nc", nominal_run, include_fsq=True)
            print(
                f"eval={eval_id:03d} objective={objective:.6e} "
                f"nominal={scenario_objectives[0]:.6e} robust_samples={int(args.robust_samples)} "
                f"risk={args.robust_risk} wall_s={sum(float(s['summary'].get('wall_s') or 0.0) for s in scenario_entries):.2f}",
                flush=True,
            )
            history.append(entry)
            write_json(outdir / "history.json", history)
            return objective

        try:
            run, wall_s = run_direct_free_boundary(
                input_path,
                params,
                vmec_max_iter=int(args.vmec_max_iter),
                activate_fsq=float(args.activate_fsq),
            )
            provisional = summarize_run(
                run,
                params,
                objective=np.nan,
                wall_s=wall_s,
                target_aspect=float(args.target_aspect),
                target_iota=float(args.target_iota),
            )
            objective = objective_from_summary(
                provisional,
                residual_weight=float(args.residual_weight),
                aspect_weight=float(args.aspect_weight),
                iota_weight=float(args.iota_weight),
            )
            provisional["objective"] = objective
            entry = {
                "eval": eval_id,
                "x": np.asarray(x, dtype=float).tolist(),
                "variables": [{"kind": kind, "index": index} for kind, index in variables],
                "summary": provisional,
            }
            if best is None or objective < float(best["summary"]["objective"]):
                best = entry
                write_wout_from_fixed_boundary_run(outdir / "wout_best_direct_coil_phase1.nc", run, include_fsq=True)
            print(
                f"eval={eval_id:03d} objective={objective:.6e} "
                f"residual={provisional['residual_proxy']:.3e} aspect={provisional['aspect']} "
                f"mean_iota={provisional['mean_iota']} wall_s={wall_s:.2f}",
                flush=True,
            )
        except Exception as exc:
            objective = float(args.failure_objective)
            entry = {
                "eval": eval_id,
                "x": np.asarray(x, dtype=float).tolist(),
                "variables": [{"kind": kind, "index": index} for kind, index in variables],
                "error": f"{type(exc).__name__}: {exc}",
                "summary": {"objective": objective},
            }
            print(f"eval={eval_id:03d} failed with {entry['error']}; returning {objective:.3e}", flush=True)
        history.append(entry)
        write_json(outdir / "history.json", history)
        return objective

    from scipy.optimize import minimize

    optimizer_result = minimize(
        evaluate,
        x0,
        method="Powell",
        options={
            "maxiter": int(args.max_iter),
            "maxfev": int(args.max_evals),
            "xtol": float(args.xtol),
            "ftol": float(args.optimizer_ftol),
            "disp": False,
        },
    )

    summary = {
        "phase": "phase-1-smoke",
        "scope": "coil-only direct-coil free-boundary scaffold",
        "plasma_boundary_optimized": False,
        "optimized_variables": [{"kind": kind, "index": index} for kind, index in variables],
        "objective_model": {
            "description": "Cheap residual/aspect/iota proxy; Boozer/QS residual is intentionally deferred.",
            "phase2_note": "Add Boozer/QS objective after full-loop gradients are validated.",
            "residual_weight": float(args.residual_weight),
            "aspect_weight": float(args.aspect_weight),
            "iota_weight": float(args.iota_weight),
        },
        "provider": provider_metadata,
        "input": input_path,
        "outdir": outdir,
        "optimizer": {
            "method": "Powell",
            "success": bool(optimizer_result.success),
            "message": str(optimizer_result.message),
            "nfev": int(optimizer_result.nfev),
            "nit": int(optimizer_result.nit),
            "fun": float(optimizer_result.fun),
            "x": np.asarray(optimizer_result.x, dtype=float),
        },
        "best": best,
        "history_json": outdir / "history.json",
        "best_wout": outdir / "wout_best_direct_coil_phase1.nc",
    }
    if robust_options is not None:
        summary["robust_objective"] = robust_options
    write_json(outdir / "summary.json", summary)
    print(f"Wrote {outdir / 'history.json'}")
    print(f"Wrote {outdir / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use tiny defaults for a fast phase-1 smoke.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--provider", choices=("essos", "circle"), default="essos")
    parser.add_argument("--coils-json", type=Path, default=None)
    parser.add_argument("--current-scale", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=None, help="Outer Powell optimizer iterations.")
    parser.add_argument("--max-evals", type=int, default=None, help="Maximum objective evaluations.")
    parser.add_argument("--vmec-max-iter", type=int, default=None, help="Inner free-boundary VMEC iterations.")
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--optimizer-ftol", type=float, default=1.0e-4)
    parser.add_argument("--xtol", type=float, default=1.0e-4)
    parser.add_argument("--ns", type=int, default=None)
    parser.add_argument("--mpol", type=int, default=None)
    parser.add_argument("--ntor", type=int, default=None)
    parser.add_argument("--nzeta", type=int, default=None)
    parser.add_argument("--pressure-scale", type=float, default=0.0)
    parser.add_argument("--activate-fsq", type=float, default=1.0)
    parser.add_argument("--max-current-vars", type=int, default=1)
    parser.add_argument("--max-fourier-vars", type=int, default=1)
    parser.add_argument("--current-step", type=float, default=0.02)
    parser.add_argument("--dof-step", type=float, default=1.0e-3)
    parser.add_argument("--target-aspect", type=float, default=6.0)
    parser.add_argument("--target-iota", type=float, default=0.4)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--aspect-weight", type=float, default=1.0e-2)
    parser.add_argument("--iota-weight", type=float, default=1.0)
    parser.add_argument("--failure-objective", type=float, default=1.0e30)
    parser.add_argument("--robust-samples", type=int, default=0, help="Number of perturbed coil scenarios to add to the nominal objective.")
    parser.add_argument("--robust-risk", choices=("mean", "mean_plus_std", "smooth"), default="mean")
    parser.add_argument("--robust-std-weight", type=float, default=1.0)
    parser.add_argument("--robust-temperature", type=float, default=1.0e-3, help="Temperature for --robust-risk smooth.")
    parser.add_argument("--robust-seed", type=int, default=20240524)
    parser.add_argument("--robust-current-sigma", type=float, default=1.0e-2, help="Fractional current perturbation sigma.")
    parser.add_argument("--robust-displacement-sigma", type=float, default=0.0, help="Rigid Cartesian displacement sigma.")
    parser.add_argument("--robust-toroidal-phase-sigma", type=float, default=0.0, help="Toroidal phase rotation sigma in radians.")
    parser.add_argument("--robust-centerline-sigma", type=float, default=0.0, help="Fourier centerline coefficient perturbation sigma.")
    parser.add_argument("--robust-centerline-include-constant", action="store_true")
    return parser


def apply_smoke_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        args.max_iter = 1 if args.max_iter is None else args.max_iter
        args.max_evals = 3 if args.max_evals is None else args.max_evals
        args.vmec_max_iter = 1 if args.vmec_max_iter is None else args.vmec_max_iter
        args.ns = 12 if args.ns is None else args.ns
        args.mpol = 3 if args.mpol is None else args.mpol
        args.ntor = 2 if args.ntor is None else args.ntor
        args.nzeta = 4 if args.nzeta is None else args.nzeta
        return args

    args.max_iter = 4 if args.max_iter is None else args.max_iter
    args.max_evals = 12 if args.max_evals is None else args.max_evals
    args.vmec_max_iter = 3 if args.vmec_max_iter is None else args.vmec_max_iter
    args.ns = 12 if args.ns is None else args.ns
    args.mpol = 4 if args.mpol is None else args.mpol
    args.ntor = 4 if args.ntor is None else args.ntor
    args.nzeta = 6 if args.nzeta is None else args.nzeta
    return args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = apply_smoke_defaults(parser.parse_args(argv))
    try:
        optimize_coils(args)
    except SkipExample as exc:
        print(f"SKIP: {exc}", file=sys.stderr)
        return SKIP_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

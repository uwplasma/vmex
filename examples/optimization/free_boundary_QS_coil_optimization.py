#!/usr/bin/env python
"""Direct-coil free-boundary coil-only quasisymmetry optimization.

This is intentionally *not* a production QS optimization.  The only optimizer
variables are direct-coil currents and selected direct-coil Fourier dofs.  The
plasma boundary coefficients from the VMEC input deck are never included in the
optimization vector.

The validation-scale objective is deliberately transparent:

* VMEC residual from a tiny direct-coil free-boundary solve,
* VMEC-state quasisymmetry-ratio residual,
* aspect-ratio target,
* mean-iota target.

The QS residual is evaluated from the accepted VMEC state, not from a
full coil-to-Boozer exact adjoint.  The optional same-branch report writes the
current complete-solve finite-difference and fixed-accepted-branch derivative
evidence without claiming differentiation through adaptive host branch
selection.

Run a minimal smoke from the repository root:

    python examples/optimization/free_boundary_QS_coil_optimization.py --smoke --provider circle

Add a same-branch derivative artifact using the validated branch-local vector
JVP report:

    python examples/optimization/free_boundary_QS_coil_optimization.py --smoke --provider circle --write-same-branch-report

Preview the generated input, selected coil variables, objective weights, and
baseline coil diagnostics without running VMEC:

    python examples/optimization/free_boundary_QS_coil_optimization.py --smoke --dry-run --provider circle

For the optional ESSOS provider, set the ESSOS checkout and input directory:

    export ESSOS_ROOT=/path/to/ESSOS_mgrid_pr
    export ESSOS_INPUT_DIR=$ESSOS_ROOT/examples/input_files
    PYTHONPATH=$ESSOS_ROOT:$PYTHONPATH python examples/optimization/free_boundary_QS_coil_optimization.py --smoke --provider essos

If ESSOS assets are not available, the ESSOS provider exits with code 77 and a
helpful message.
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
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import jax, jnp
from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import CoilFieldParams, from_essos_coils
from vmec_jax.external_fields.coils_jax import coil_current_norm, coil_lengths
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.profiles import pressure_profile_to_vmec_am, standard_finite_beta_profiles
from vmec_jax.quasisymmetry import quasisymmetry_angle_cache_from_static, quasisymmetry_ratio_residual_from_state
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state


SKIP_EXIT_CODE = 77
DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "free_boundary_QS_coil_optimization"
DEFAULT_ESSOS_COIL_JSON = "ESSOS_biot_savart_LandremanPaulQA.json"
DEFAULT_FREE_BOUNDARY_PHIEDGE = -0.025
DEFAULT_SAME_BRANCH_VECTOR_KEYS = ("aspect", "qs_total")
SUPPORTED_SAME_BRANCH_VECTOR_KEYS = (
    "state_norm",
    "aspect",
    "qs_total",
    "lcfs_boundary_moment",
    "accepted_bnormal_rms",
)
STATE_ONLY_SAME_BRANCH_KEYS = (
    "state_norm",
    "aspect",
    "qs_total",
    "lcfs_boundary_moment",
)
SINGLE_STAGE_LIMITATIONS = [
    "The QS term is a VMEC-state quasisymmetry-ratio residual, not a Boozer-space exact-adjoint objective.",
    "Production full-loop direct-coil free-boundary adjoints are not promoted yet.",
    "ESSOS and VMEC2000 generated-mgrid comparisons remain optional external-asset diagnostics.",
]


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


def make_circle_provider(*, current_scale: float, chunk_size: int | None = None) -> tuple[CoilFieldParams, dict[str, Any]]:
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(1.4)
    dofs = dofs.at[0, 1, 1].set(1.4)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([2.0]),
        n_segments=96,
        current_scale=float(current_scale),
        chunk_size=None if chunk_size is None else int(chunk_size),
    )
    return params, {
        "provider": "circle",
        "current_scale_multiplier": float(current_scale),
        "chunk_size": None if chunk_size is None else int(chunk_size),
    }


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
    beta_percent: float,
    pressure_profile: str,
    pressure_scale: float,
    phiedge: float,
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
            "PHIEDGE": float(phiedge),
            "MPOL": int(mpol),
            "NTOR": int(ntor),
            "NZETA": int(nzeta),
            "NTHETA": 0,
            "NVACSKIP": max(1, int(nzeta)),
        }
    )
    pressure_profile = str(pressure_profile).strip().lower()
    if pressure_profile == "standard":
        profiles = standard_finite_beta_profiles(float(beta_percent))
        am, pres_scale = pressure_profile_to_vmec_am(profiles.pressure_pa, pres_scale=1.0)
        indata.scalars["PMASS_TYPE"] = "power_series"
        indata.scalars["PRES_SCALE"] = pres_scale
        indata.scalars["AM"] = am
    elif pressure_profile in {"linear", "linear-scale", "legacy"}:
        indata.scalars["PMASS_TYPE"] = "power_series"
        indata.scalars["PRES_SCALE"] = float(pressure_scale)
        indata.scalars["AM"] = [1.0, -1.0]
    else:
        raise ValueError("pressure_profile must be 'standard' or 'linear-scale'")
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


def coil_diagnostics(params: CoilFieldParams) -> dict[str, Any]:
    lengths = np.asarray(coil_lengths(params), dtype=float).reshape(-1)
    currents = np.asarray(params.base_currents, dtype=float).reshape(-1)
    dofs = np.asarray(params.base_curve_dofs, dtype=float)
    return {
        "n_base_coils": int(currents.size),
        "n_segments": int(params.n_segments),
        "nfp": int(params.nfp),
        "stellsym": bool(params.stellsym),
        "current_scale": float(params.current_scale),
        "current_min": float(np.min(currents)) if currents.size else None,
        "current_max": float(np.max(currents)) if currents.size else None,
        "coil_current_norm": float(np.asarray(coil_current_norm(params))),
        "mean_coil_length": float(np.mean(lengths)) if lengths.size else None,
        "min_coil_length": float(np.min(lengths)) if lengths.size else None,
        "max_coil_length": float(np.max(lengths)) if lengths.size else None,
        "base_curve_dofs_shape": [int(v) for v in dofs.shape],
        "nonzero_base_curve_dofs": int(np.count_nonzero(np.abs(dofs) > 0.0)),
    }


def variable_records(
    variables: list[tuple[str, tuple[int, ...]]],
    base_params: CoilFieldParams,
    *,
    current_step: float,
    dof_step: float,
) -> list[dict[str, Any]]:
    currents = np.asarray(base_params.base_currents, dtype=float)
    dofs = np.asarray(base_params.base_curve_dofs, dtype=float)
    records: list[dict[str, Any]] = []
    for kind, index in variables:
        record: dict[str, Any] = {"kind": kind, "index": index}
        if kind == "current":
            i = index[0]
            record.update(
                {
                    "base_value": float(currents[i]),
                    "parameterization": "multiplicative",
                    "unit_x_delta": float(currents[i]) * float(current_step),
                    "current_step_fraction": float(current_step),
                }
            )
        elif kind == "fourier_dof":
            record.update(
                {
                    "base_value": float(dofs[index]),
                    "parameterization": "additive",
                    "unit_x_delta": float(dof_step),
                    "dof_step": float(dof_step),
                }
            )
        else:  # pragma: no cover - defensive programming for future variable kinds.
            record["parameterization"] = "unknown"
        records.append(record)
    return records


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


def parse_float_list(text: str) -> list[float]:
    """Parse comma/space-separated floats from a small CLI option."""

    cleaned = str(text).replace(",", " ")
    values = [float(part) for part in cleaned.split() if part]
    if not values:
        raise ValueError("expected at least one floating-point value")
    return values


def parse_same_branch_vector_keys(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse branch-local vector report scalar keys from a small CLI option."""

    if value is None:
        keys = DEFAULT_SAME_BRANCH_VECTOR_KEYS
    elif isinstance(value, str):
        keys = tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
    else:
        keys = tuple(str(part).strip() for part in value if str(part).strip())
    if not keys:
        raise ValueError("expected at least one same-branch vector scalar key")
    unsupported = tuple(key for key in keys if key not in SUPPORTED_SAME_BRANCH_VECTOR_KEYS)
    if unsupported:
        supported = ", ".join(SUPPORTED_SAME_BRANCH_VECTOR_KEYS)
        raise ValueError(f"Unsupported same-branch vector scalar key(s) {unsupported}; supported keys: {supported}")
    return keys


def run_direct_free_boundary(
    input_path: Path,
    params: CoilFieldParams,
    *,
    vmec_max_iter: int,
    activate_fsq: float,
    jit_forces: bool = True,
) -> tuple[Any, float]:
    start = time.perf_counter()
    run = run_free_boundary(
        input_path,
        max_iter=int(vmec_max_iter),
        multigrid=False,
        verbose=False,
        jit_forces=bool(jit_forces),
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
    helicity_m: int = 1,
    helicity_n: int = 0,
    qs_surfaces: list[float] | None = None,
    qs_ntheta: int = 31,
    qs_nphi: int = 32,
) -> dict[str, Any]:
    qs_surfaces = [0.25, 0.5, 0.75] if qs_surfaces is None else qs_surfaces
    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    freeb = diag.get("free_boundary", {}) if isinstance(diag, dict) else {}
    nestor = freeb.get("last_nestor_diagnostics", {}) if isinstance(freeb, dict) else {}
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
    qs_total = None
    try:
        qs = quasisymmetry_ratio_residual_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
            surfaces=qs_surfaces,
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
            ntheta=int(qs_ntheta),
            nphi=int(qs_nphi),
        )
        qs_total = float(np.asarray(qs["total"]))
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
        "qs_total": qs_total,
        "qs_helicity_m": int(helicity_m),
        "qs_helicity_n": int(helicity_n),
        "qs_surfaces": [float(value) for value in qs_surfaces],
        "qs_ntheta": int(qs_ntheta),
        "qs_nphi": int(qs_nphi),
        "coil_current_norm": float(np.asarray(coil_current_norm(params))),
        "mean_coil_length": float(np.mean(np.asarray(coil_lengths(params), dtype=float))),
        "free_boundary_vacuum_stub": freeb.get("vacuum_stub") if isinstance(freeb, dict) else None,
        "free_boundary_nestor_model": freeb.get("nestor_model") if isinstance(freeb, dict) else None,
        "free_boundary_bnormal_rms": nestor.get("bnormal_rms") if isinstance(nestor, dict) else None,
        "free_boundary_bsqvac_rms": nestor.get("bsqvac_rms") if isinstance(nestor, dict) else None,
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
    qs_weight: float = 0.0,
) -> float:
    return float(
        objective_terms_from_summary(
            summary,
            residual_weight=residual_weight,
            qs_weight=qs_weight,
            aspect_weight=aspect_weight,
            iota_weight=iota_weight,
        )["total"]
    )


def objective_terms_from_summary(
    summary: dict[str, Any],
    *,
    residual_weight: float,
    aspect_weight: float,
    iota_weight: float,
    qs_weight: float = 0.0,
) -> dict[str, Any]:
    residual = float(summary.get("residual_proxy") or 0.0)
    qs_total = summary.get("qs_total")
    aspect = summary.get("aspect")
    mean_iota = summary.get("mean_iota")
    aspect_error = None if aspect is None else float(aspect) - float(summary["target_aspect"])
    iota_error = None if mean_iota is None else float(mean_iota) - float(summary["target_iota"])
    qs_penalty = 0.0 if qs_total is None else float(qs_total)
    aspect_penalty = 0.0 if aspect_error is None else aspect_error**2
    iota_penalty = 0.0 if iota_error is None else iota_error**2
    residual_term = float(residual_weight) * residual
    qs_term = float(qs_weight) * qs_penalty
    aspect_term = float(aspect_weight) * aspect_penalty
    iota_term = float(iota_weight) * iota_penalty
    missing_terms = []
    if qs_total is None and float(qs_weight) != 0.0:
        missing_terms.append("qs_total")
    if aspect is None and float(aspect_weight) != 0.0:
        missing_terms.append("aspect")
    if mean_iota is None and float(iota_weight) != 0.0:
        missing_terms.append("mean_iota")
    return {
        "total": float(residual_term + qs_term + aspect_term + iota_term),
        "residual": {
            "value": residual,
            "weight": float(residual_weight),
            "contribution": float(residual_term),
        },
        "quasisymmetry": {
            "value": None if qs_total is None else float(qs_total),
            "target": 0.0,
            "weight": float(qs_weight),
            "contribution": float(qs_term),
            "helicity_m": int(summary.get("qs_helicity_m", 1)),
            "helicity_n": int(summary.get("qs_helicity_n", 0)),
            "surfaces": [float(value) for value in summary.get("qs_surfaces", [])],
        },
        "aspect": {
            "value": None if aspect is None else float(aspect),
            "target": float(summary["target_aspect"]),
            "error": aspect_error,
            "squared_error": float(aspect_penalty),
            "weight": float(aspect_weight),
            "contribution": float(aspect_term),
        },
        "mean_iota": {
            "value": None if mean_iota is None else float(mean_iota),
            "target": float(summary["target_iota"]),
            "error": iota_error,
            "squared_error": float(iota_penalty),
            "weight": float(iota_weight),
            "contribution": float(iota_term),
        },
        "missing_unweighted_terms": missing_terms,
    }


def same_branch_direction_from_variables(variables: list[tuple[str, tuple[int, ...]]]) -> np.ndarray:
    """Return a mixed current/Fourier validation direction in optimizer space."""
    direction = np.zeros(len(variables), dtype=float)
    current_index = next((i for i, (kind, _index) in enumerate(variables) if kind == "current"), None)
    fourier_index = next((i for i, (kind, _index) in enumerate(variables) if kind == "fourier_dof"), None)
    if current_index is not None:
        direction[current_index] = 1.0
    if fourier_index is not None:
        direction[fourier_index] = 1.0
    if not np.any(direction):
        raise ValueError("same-branch validation needs at least one selected coil variable")
    return direction


def coil_param_direction_from_variables(
    base_params: CoilFieldParams,
    x_direction: np.ndarray,
    variables: list[tuple[str, tuple[int, ...]]],
    *,
    current_step: float,
    dof_step: float,
) -> CoilFieldParams:
    """Return the direct-coil parameter tangent for one optimizer direction."""

    currents = np.zeros_like(np.asarray(base_params.base_currents, dtype=float))
    dofs = np.zeros_like(np.asarray(base_params.base_curve_dofs, dtype=float))
    for value, (kind, index) in zip(np.asarray(x_direction, dtype=float), variables, strict=True):
        if value == 0.0:
            continue
        if kind == "current":
            i = index[0]
            currents[i] += float(value) * float(current_step) * float(np.asarray(base_params.base_currents)[i])
        elif kind == "fourier_dof":
            dofs[index] += float(value) * float(dof_step)
        else:  # pragma: no cover - defensive programming for future variable kinds.
            raise ValueError(f"unknown coil variable kind {kind!r}")
    return base_params.with_arrays(base_curve_dofs=jnp.asarray(dofs), base_currents=jnp.asarray(currents))


def _vector_jacobian_directional(jacobian: Any, direction: Any, n_outputs: int) -> np.ndarray:
    """Contract a row-stacked pytree Jacobian with one pytree direction."""

    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda jac_leaf, direction_leaf: jnp.sum(
                jnp.reshape(jnp.asarray(jac_leaf), (int(n_outputs), -1))
                * jnp.reshape(jnp.asarray(direction_leaf), (1, -1)),
                axis=1,
            ),
            jacobian,
            direction,
        )
    )
    if not leaves:
        return np.zeros(int(n_outputs), dtype=float)
    total = leaves[0]
    for leaf in leaves[1:]:
        total = total + leaf
    return np.asarray(total, dtype=float)


def _pytree_directional_vdot(gradient: Any, direction: Any) -> float:
    """Contract one pytree gradient with one pytree direction."""

    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(
            lambda grad_leaf, direction_leaf: jnp.sum(jnp.asarray(grad_leaf) * jnp.asarray(direction_leaf)),
            gradient,
            direction,
        )
    )
    if not leaves:
        return 0.0
    total = leaves[0]
    for leaf in leaves[1:]:
        total = total + leaf
    return float(np.asarray(total, dtype=float))


def same_branch_report_anchor_params(
    base_params: CoilFieldParams,
    best: dict[str, Any] | None,
    variables: list[tuple[str, tuple[int, ...]]],
    args: argparse.Namespace,
) -> tuple[CoilFieldParams, str]:
    """Return the coil point used by the opt-in branch-local derivative report."""

    anchor = str(getattr(args, "same_branch_report_anchor", "best")).strip().lower()
    if anchor not in {"initial", "best"}:
        raise ValueError("--same-branch-report-anchor must be one of initial, best")
    if anchor == "initial":
        return base_params, "initial"
    if best is None or "x" not in best:
        return base_params, "initial_no_best_available"
    return (
        apply_coil_variables(
            base_params,
            np.asarray(best["x"], dtype=float),
            variables,
            current_step=float(args.current_step),
            dof_step=float(args.dof_step),
        ),
        "best",
    )


def same_branch_derivative_proposal_from_report(
    report: dict[str, Any],
    objective_model: dict[str, Any],
    best: dict[str, Any] | None,
    *,
    step_size: float,
) -> dict[str, Any]:
    """Return one conservative derivative-assisted proposal from a report.

    The proposal uses the validated fixed-accepted-branch directional JVP only
    to choose a one-dimensional trial direction.  A normal complete VMEC solve
    must still evaluate and accept or reject the returned ``trial_x``.
    """

    if best is None or "x" not in best:
        return {"available": False, "reason": "no best point is available"}
    vector = report.get("branch_local_vector_jacobian", {})
    if not bool(vector.get("available", False)):
        return {"available": False, "reason": str(vector.get("reason", "branch-local vector report unavailable"))}
    if bool(vector.get("differentiates_adaptive_controller", True)):
        return {"available": False, "reason": "branch-local vector report claims adaptive-controller differentiation"}
    if not bool(vector.get("differentiates_fixed_accepted_branch", False)):
        return {"available": False, "reason": "branch-local vector report does not differentiate a fixed accepted branch"}

    scalars = vector.get("scalars", {})
    contributions: dict[str, dict[str, float]] = {}
    directional = 0.0

    if "qs_total" in scalars:
        scalar = scalars["qs_total"]
        deriv = float(scalar["exact_directional"])
        contribution = float(objective_model.get("qs_weight", 0.0)) * deriv
        contributions["qs_total"] = {"exact_directional": deriv, "contribution": contribution}
        directional += contribution

    if "aspect" in scalars:
        scalar = scalars["aspect"]
        value = float(scalar["value"])
        deriv = float(scalar["exact_directional"])
        target = float(objective_model.get("target_aspect", value))
        contribution = 2.0 * float(objective_model.get("aspect_weight", 0.0)) * (value - target) * deriv
        contributions["aspect"] = {
            "value": value,
            "target": target,
            "exact_directional": deriv,
            "contribution": contribution,
        }
        directional += contribution

    if not contributions:
        return {"available": False, "reason": "no report scalars map to the objective terms"}
    if not np.isfinite(directional):
        return {"available": False, "reason": "non-finite directional derivative"}
    if directional == 0.0:
        return {"available": False, "reason": "zero directional derivative"}

    direction_x = np.asarray(report.get("direction_x", []), dtype=float)
    x_best = np.asarray(best["x"], dtype=float)
    if direction_x.shape != x_best.shape:
        return {
            "available": False,
            "reason": f"direction_x shape {direction_x.shape} does not match best x shape {x_best.shape}",
        }
    alpha = -float(step_size) * float(np.sign(directional))
    trial_x = x_best + alpha * direction_x
    return {
        "available": True,
        "scope": "fixed accepted-branch directional proposal; complete solve decides acceptance",
        "differentiates_adaptive_controller": False,
        "directional_derivative": float(directional),
        "contributions": contributions,
        "alpha": float(alpha),
        "step_size": float(step_size),
        "direction_x": direction_x.tolist(),
        "base_x": x_best.tolist(),
        "trial_x": trial_x.tolist(),
    }


def write_same_branch_validation_report(
    *,
    input_path: Path,
    base_params: CoilFieldParams,
    variables: list[tuple[str, tuple[int, ...]]],
    args: argparse.Namespace,
    outdir: Path,
    report_anchor: str = "initial",
) -> Path:
    """Write an optional same-branch complete-solve FD report for this example."""
    from vmec_jax.free_boundary_adjoint import (
        direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax,
        direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax,
        direct_coil_same_branch_complete_solve_fd_report,
        free_boundary_boundary_geometry_jax,
    )
    from vmec_jax.state import pack_state

    direction_x = same_branch_direction_from_variables(variables)
    direction_params = coil_param_direction_from_variables(
        base_params,
        direction_x,
        variables,
        current_step=float(args.current_step),
        dof_step=float(args.dof_step),
    )
    qs_surfaces = parse_float_list(str(args.qs_surfaces))
    qs_angle_cache_by_key: dict[tuple[int, ...], dict[str, object]] = {}

    def qs_angle_cache_for_static(static: Any) -> dict[str, object]:
        cfg = static.cfg
        key = (
            int(cfg.nfp),
            int(cfg.mpol),
            int(cfg.ntor),
            int(cfg.ntheta),
            int(cfg.nzeta),
            int(args.qs_ntheta),
            int(args.qs_nphi),
        )
        if key not in qs_angle_cache_by_key:
            qs_angle_cache_by_key[key] = quasisymmetry_angle_cache_from_static(
                static,
                ntheta=int(args.qs_ntheta),
                nphi=int(args.qs_nphi),
            )
        return qs_angle_cache_by_key[key]

    def lcfs_boundary_moment(state: Any, static: Any) -> Any:
        geometry = free_boundary_boundary_geometry_jax(state, static)
        r = jnp.asarray(geometry["R"])
        z = jnp.asarray(geometry["Z"])
        return jnp.mean((r - 1.0) * (r - 1.0) + z * z)

    def accepted_bnormal_rms_from_payload(payload: dict[str, Any]) -> float:
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

    def accepted_bnormal_rms_from_replay(replay: dict[str, Any]) -> Any:
        bnormal = jnp.asarray(replay["history"]["bnormal_rms"])
        accepted = jnp.asarray(replay["history"]["accepted"], dtype=bnormal.dtype)
        active = jnp.asarray(replay["controls"]["has_active_freeb_replay"], dtype=bnormal.dtype)
        weights = accepted * active
        denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=bnormal.dtype))
        return jnp.sum(weights * bnormal) / denom

    def qs_total_from_state(state: Any, static: Any, indata: Any, signgs: int) -> Any:
        qs = quasisymmetry_ratio_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=int(signgs),
            surfaces=qs_surfaces,
            helicity_m=int(args.helicity_m),
            helicity_n=int(args.helicity_n),
            ntheta=int(args.qs_ntheta),
            nphi=int(args.qs_nphi),
            angle_cache=qs_angle_cache_for_static(static),
        )
        return qs["total"]

    def params_for(scale: float) -> CoilFieldParams:
        return apply_coil_variables(
            base_params,
            direction_x * float(scale),
            variables,
            current_step=float(args.current_step),
            dof_step=float(args.dof_step),
        )

    def objective_fn(payload: dict[str, Any]) -> dict[str, float]:
        run_like = SimpleNamespace(
            result=payload["result"],
            state=payload["result"].state,
            static=payload["init"].static,
            indata=payload["init"].indata,
            signgs=payload["init"].signgs,
        )
        summary = summarize_run(
            run_like,
            payload["params"],
            objective=np.nan,
            wall_s=np.nan,
            target_aspect=float(args.target_aspect),
            target_iota=float(args.target_iota),
            helicity_m=int(args.helicity_m),
            helicity_n=int(args.helicity_n),
            qs_surfaces=qs_surfaces,
            qs_ntheta=int(args.qs_ntheta),
            qs_nphi=int(args.qs_nphi),
        )
        total = objective_from_summary(
            summary,
            residual_weight=float(args.residual_weight),
            qs_weight=float(args.qs_weight),
            aspect_weight=float(args.aspect_weight),
            iota_weight=float(args.iota_weight),
        )
        return {
            "objective": total,
            "state_norm": float(np.linalg.norm(np.asarray(pack_state(payload["result"].state), dtype=float))),
            "residual_proxy": float(summary.get("residual_proxy") or 0.0),
            "qs_total": float(summary["qs_total"]) if summary.get("qs_total") is not None else np.nan,
            "aspect": float(summary["aspect"]) if summary.get("aspect") is not None else np.nan,
            "lcfs_boundary_moment": float(np.asarray(lcfs_boundary_moment(payload["result"].state, payload["init"].static))),
            "mean_iota": float(summary["mean_iota"]) if summary.get("mean_iota") is not None else np.nan,
            "accepted_bnormal_rms": accepted_bnormal_rms_from_payload(payload),
            "bnormal_rms": float(summary["free_boundary_bnormal_rms"])
            if summary.get("free_boundary_bnormal_rms") is not None
            else np.nan,
        }

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    report = direct_coil_same_branch_complete_solve_fd_report(
        input_path,
        base_params,
        params_for=params_for,
        objective_fn=objective_fn,
        eps=float(args.same_branch_report_eps),
        solve_kwargs={
            "max_iter": int(args.same_branch_report_max_iter or args.vmec_max_iter),
            "ftol": float(args.ftol),
            "vmec2000_control": True,
            "auto_flip_force": False,
            "use_direct_fallback": True,
            "verbose": False,
            "verbose_vmec2000_table": False,
            "jit_forces": bool(args.jit_forces),
            "use_scan": False,
            "host_update_assembly": False,
            "adjoint_trace": True,
            "adjoint_trace_mode": "branch",
            "external_field_provider_kind": "direct_coils",
            "free_boundary_activate_fsq": float(args.activate_fsq),
        },
    )
    timings["complete_solve_fd_wall_s"] = float(time.perf_counter() - t0)
    compact_report = {
        "phase": "phase-2-same-branch-complete-solve-fd",
        "scope": "coil-only proxy-objective validation; not arbitrary adaptive-branch differentiation",
        "input": str(input_path),
        "report_anchor": str(report_anchor),
        "eps": float(args.same_branch_report_eps),
        "direction_x": direction_x.tolist(),
        "direction_variables": [
            variable_manifest
            for active, variable_manifest in zip(
                direction_x != 0.0,
                variable_records(
                    variables,
                    base_params,
                    current_step=float(args.current_step),
                    dof_step=float(args.dof_step),
                ),
                strict=True,
            )
            if bool(active)
        ],
        "branch_compatibility": {
            "same_branch": bool(report["branch_compatibility"]["same_branch"]),
            "plus_changed_fields": list(report["branch_compatibility"]["plus"]["changed_fields"]),
            "minus_changed_fields": list(report["branch_compatibility"]["minus"]["changed_fields"]),
            "plus_max_abs_scalar_delta": float(report["branch_compatibility"]["plus"]["max_abs_scalar_delta"]),
            "minus_max_abs_scalar_delta": float(report["branch_compatibility"]["minus"]["max_abs_scalar_delta"]),
            "plus_max_rel_scalar_delta": float(report["branch_compatibility"]["plus"]["max_rel_scalar_delta"]),
            "minus_max_rel_scalar_delta": float(report["branch_compatibility"]["minus"]["max_rel_scalar_delta"]),
        },
        "values": report["values"],
        "objective_values": report["objective_values"],
        "primary_objective": report["primary_objective"],
    }
    mode = str(getattr(args, "same_branch_report_mode", "none")).strip().lower()
    ad_mode = str(getattr(args, "same_branch_report_ad_mode", "direct")).strip().lower()
    same_branch = bool(report["branch_compatibility"]["same_branch"])
    vector_keys = parse_same_branch_vector_keys(getattr(args, "same_branch_report_vector_keys", None))
    branch_local_scalar: dict[str, Any] = {
        "available": False,
        "scope": "fixed accepted branch only; does not differentiate adaptive host branch selection",
        "mode": mode,
        "replay_ad_mode": ad_mode,
        "same_branch": same_branch,
        "reason": "not requested" if mode != "scalar" else "branch fingerprint is not same-branch compatible",
    }
    branch_local_vector: dict[str, Any] = {
        "available": False,
        "scope": "fixed accepted branch only; does not differentiate adaptive host branch selection",
        "mode": mode,
        "replay_ad_mode": ad_mode,
        "same_branch": same_branch,
        "scalar_keys": list(vector_keys),
        "reason": "not requested" if mode != "vector" else "branch fingerprint is not same-branch compatible",
    }
    if mode not in {"none", "scalar", "vector"}:
        raise ValueError("--same-branch-report-mode must be one of none, scalar, vector")
    if ad_mode not in {"direct", "custom_vjp"}:
        raise ValueError("--same-branch-report-ad-mode must be one of direct, custom_vjp")
    report_base_values = {
        str(key): float(values["base"])
        for key, values in report["objective_values"].items()
        if isinstance(values, dict) and "base" in values
    }
    replay_payload = {"init": report["base"]["init"]} if isinstance(report.get("base"), dict) and "init" in report["base"] else None
    scalar_value_fns = {
        "state_norm": lambda payload: float(np.linalg.norm(np.asarray(pack_state(payload["result"].state), dtype=float))),
        "aspect": lambda payload: float(
            np.asarray(
                equilibrium_aspect_ratio_from_state(
                    state=payload["result"].state,
                    static=payload["init"].static,
                )
            )
        ),
        "qs_total": lambda payload: float(
            np.asarray(
                qs_total_from_state(
                    payload["result"].state,
                    payload["init"].static,
                    payload["init"].indata,
                    payload["init"].signgs,
                )
            )
        ),
        "lcfs_boundary_moment": lambda payload: float(
            np.asarray(lcfs_boundary_moment(payload["result"].state, payload["init"].static))
        ),
        "accepted_bnormal_rms": accepted_bnormal_rms_from_payload,
    }
    scalar_replay_fns = {
        "state_norm": lambda replay, _payload: jnp.linalg.norm(pack_state(replay["state"])),
        "aspect": lambda replay, payload: equilibrium_aspect_ratio_from_state(
            state=replay["state"],
            static=payload["init"].static,
        ),
        "qs_total": lambda replay, payload: qs_total_from_state(
            replay["state"],
            payload["init"].static,
            payload["init"].indata,
            payload["init"].signgs,
        ),
        "lcfs_boundary_moment": lambda replay, payload: lcfs_boundary_moment(
            replay["state"],
            payload["init"].static,
        ),
        "accepted_bnormal_rms": lambda replay, _payload: accepted_bnormal_rms_from_replay(replay),
    }
    scalar_key = str(getattr(args, "same_branch_report_scalar_key", "qs_total"))
    scalar_uses_state_only_replay = scalar_key in STATE_ONLY_SAME_BRANCH_KEYS
    vector_uses_state_only_replay = all(key in STATE_ONLY_SAME_BRANCH_KEYS for key in vector_keys)
    replay_kwargs = {
        "use_stacked_step_controls": True,
        "use_accepted_only_fast_path": True,
        "jit_preconditioner_apply": not bool(getattr(args, "same_branch_report_disable_jit_preconditioner", False)),
        "include_analytic": not bool(getattr(args, "same_branch_report_disable_analytic", False)),
        "include_mode_diagnostics": False,
        "nestor_solve_mode": str(getattr(args, "same_branch_report_nestor_solve_mode", "dense")),
        "nestor_operator_solver": str(getattr(args, "same_branch_report_nestor_operator_solver", "gmres")),
        "nestor_operator_tol": float(getattr(args, "same_branch_report_nestor_operator_tol", 1.0e-11)),
        "nestor_operator_atol": float(getattr(args, "same_branch_report_nestor_operator_atol", 1.0e-13)),
        "nestor_operator_maxiter": getattr(args, "same_branch_report_nestor_operator_maxiter", None),
        "nestor_operator_restart": getattr(args, "same_branch_report_nestor_operator_restart", None),
        "freeze_vacuum_field": bool(getattr(args, "same_branch_report_freeze_vacuum_field", False)),
        "freeze_freeb_bsqvac": bool(getattr(args, "same_branch_report_freeze_bsqvac", False)),
    }
    if same_branch and mode == "scalar" and "base" in report and scalar_key in report["objective_values"]:
        t0 = time.perf_counter()
        scalar = direct_coil_run_free_boundary_branch_local_scalar_value_and_grad_jax(
            params=base_params,
            complete_payload=report["base"],
            scalar_key=scalar_key,
            production_values={scalar_key: report_base_values[scalar_key]},
            replay_payload=replay_payload,
            scalar_fn=lambda payload: {scalar_key: scalar_value_fns[scalar_key](payload)},
            replay_scalar_fn=lambda replay, payload: scalar_replay_fns[scalar_key](replay, payload),
            replay_kwargs={**replay_kwargs, "state_only_replay": scalar_uses_state_only_replay},
            replay_ad_mode=ad_mode,
            include_trace_replay_diagnostics=False,
            include_payload=False,
            include_replay_graph_metadata=False,
        )
        timings["branch_local_scalar_wall_s"] = float(time.perf_counter() - t0)
        scalar_timings = {str(key): float(value) for key, value in scalar.get("timings", {}).items()}
        for key, value in scalar_timings.items():
            timings[f"branch_local_scalar_{key}"] = value
        exact_directional = _pytree_directional_vdot(scalar["grad"], direction_params)
        branch_local_scalar = {
            "available": True,
            "scope": "fixed accepted branch only; does not differentiate adaptive host branch selection",
            "mode": mode,
            "uses_production_forward": bool(scalar["uses_production_forward"]),
            "differentiates_adaptive_controller": bool(scalar["differentiates_adaptive_controller"]),
            "differentiates_run_free_boundary": bool(scalar["differentiates_run_free_boundary"]),
            "differentiates_fixed_accepted_branch": bool(scalar["differentiates_fixed_accepted_branch"]),
            "replay_ad_mode": str(scalar["replay_ad_mode"]),
            "scalar_key": str(scalar["scalar_key"]),
            "production_values_source": str(scalar.get("production_values_source", "unknown")),
            "replay_payload_source": str(scalar.get("replay_payload_source", "unknown")),
            "includes_payload": bool(scalar.get("includes_payload", True)),
            "includes_replay_graph_metadata": bool(scalar.get("includes_replay_graph_metadata", True)),
            "state_only_replay": bool(scalar_uses_state_only_replay),
            "replay_option_flags": scalar["replay_option_flags"],
            "replay_graph_metadata": scalar.get("replay_graph_metadata", {}),
            "value": float(scalar["value"]),
            "replay_value": float(np.asarray(scalar["replay_value"], dtype=float)),
            "base_abs_delta": float(scalar["base_abs_delta"]),
            "exact_directional": float(exact_directional),
            "complete_fd_directional": float(report["objective_values"][scalar_key]["central_fd_directional"]),
            "abs_error": float(abs(exact_directional - report["objective_values"][scalar_key]["central_fd_directional"])),
            "timings": scalar_timings,
        }
    missing_vector_keys = tuple(key for key in vector_keys if key not in report["objective_values"])
    if mode == "vector" and missing_vector_keys:
        branch_local_vector["reason"] = f"missing complete-solve objective value(s): {missing_vector_keys}"
    if same_branch and mode == "vector" and "base" in report and not missing_vector_keys:
        scalar_keys = vector_keys
        t0 = time.perf_counter()
        vector = direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
            params=base_params,
            direction_params=direction_params if ad_mode == "direct" else None,
            complete_payload=report["base"],
            scalar_keys=scalar_keys,
            production_values={key: report_base_values[key] for key in scalar_keys},
            replay_payload=replay_payload,
            scalar_fn=lambda payload: {
                "state_norm": float(np.linalg.norm(np.asarray(pack_state(payload["result"].state), dtype=float))),
                "aspect": float(
                    np.asarray(
                        equilibrium_aspect_ratio_from_state(
                            state=payload["result"].state,
                            static=payload["init"].static,
                        )
                    )
                ),
                "qs_total": float(
                    np.asarray(
                        qs_total_from_state(
                            payload["result"].state,
                            payload["init"].static,
                            payload["init"].indata,
                            payload["init"].signgs,
                        )
                    )
                ),
                "lcfs_boundary_moment": float(
                    np.asarray(lcfs_boundary_moment(payload["result"].state, payload["init"].static))
                ),
                "accepted_bnormal_rms": accepted_bnormal_rms_from_payload(payload),
            },
            replay_scalar_fns={
                "state_norm": lambda replay, _payload: jnp.linalg.norm(pack_state(replay["state"])),
                "aspect": lambda replay, payload: equilibrium_aspect_ratio_from_state(
                    state=replay["state"],
                    static=payload["init"].static,
                ),
                "qs_total": lambda replay, payload: qs_total_from_state(
                    replay["state"],
                    payload["init"].static,
                    payload["init"].indata,
                    payload["init"].signgs,
                ),
                "lcfs_boundary_moment": lambda replay, payload: lcfs_boundary_moment(
                    replay["state"],
                    payload["init"].static,
                ),
                "accepted_bnormal_rms": lambda replay, _payload: accepted_bnormal_rms_from_replay(replay),
            },
            replay_kwargs={**replay_kwargs, "state_only_replay": vector_uses_state_only_replay},
            replay_ad_mode=ad_mode,
            include_trace_replay_diagnostics=False,
            include_payload=False,
            include_replay_graph_metadata=False,
        )
        timings["branch_local_vector_wall_s"] = float(time.perf_counter() - t0)
        vector_timings = {str(key): float(value) for key, value in vector.get("timings", {}).items()}
        for key, value in vector_timings.items():
            timings[f"branch_local_vector_{key}"] = value
        if vector.get("directional_derivatives") is None:
            directionals = _vector_jacobian_directional(vector["jacobian"], direction_params, len(scalar_keys))
        else:
            directionals = [
                float(np.asarray(vector["directional_derivatives"][key], dtype=float))
                for key in scalar_keys
            ]
        branch_local_vector = {
            "available": True,
            "scope": "fixed accepted branch only; does not differentiate adaptive host branch selection",
            "uses_production_forward": bool(vector["uses_production_forward"]),
            "differentiates_adaptive_controller": bool(vector["differentiates_adaptive_controller"]),
            "differentiates_run_free_boundary": bool(vector["differentiates_run_free_boundary"]),
            "differentiates_fixed_accepted_branch": bool(vector["differentiates_fixed_accepted_branch"]),
            "replay_ad_mode": str(vector["replay_ad_mode"]),
            "derivative_mode": str(vector.get("derivative_mode", "full_jacobian_vjp")),
            "scalar_keys": list(scalar_keys),
            "production_values_source": str(vector.get("production_values_source", "unknown")),
            "replay_payload_source": str(vector.get("replay_payload_source", "unknown")),
            "includes_payload": bool(vector.get("includes_payload", True)),
            "includes_replay_graph_metadata": bool(vector.get("includes_replay_graph_metadata", True)),
            "state_only_replay": bool(vector_uses_state_only_replay),
            "replay_option_flags": vector["replay_option_flags"],
            "replay_graph_metadata": vector.get("replay_graph_metadata", {}),
            "max_base_abs_delta": float(vector["max_base_abs_delta"]),
            "timings": vector_timings,
            "scalars": {
                key: {
                    "value": float(vector["values"][key]),
                    "replay_value": float(np.asarray(vector["replay_value_map"][key], dtype=float)),
                    "base_abs_delta": float(vector["base_abs_delta"][key]),
                    "exact_directional": float(directionals[index]),
                    "complete_fd_directional": float(report["objective_values"][key]["central_fd_directional"]),
                    "abs_error": float(abs(directionals[index] - report["objective_values"][key]["central_fd_directional"])),
                }
                for index, key in enumerate(scalar_keys)
            },
        }
    compact_report["branch_local_scalar_gradient"] = branch_local_scalar
    compact_report["branch_local_vector_jacobian"] = branch_local_vector
    compact_report["timings"] = timings
    path = outdir / "same_branch_complete_solve_report.json"
    write_json(path, compact_report)
    return path


def optimize_coils(args: argparse.Namespace) -> dict[str, Any]:
    if args.provider == "essos":
        base_params, provider_metadata = load_essos_provider(
            args.coils_json,
            chunk_size=256 if args.chunk_size is None else int(args.chunk_size),
            current_scale=float(args.current_scale),
        )
    elif args.provider == "circle":
        base_params, provider_metadata = make_circle_provider(
            current_scale=float(args.current_scale),
            chunk_size=None if args.chunk_size is None else int(args.chunk_size),
        )
    else:
        raise ValueError(f"unknown provider {args.provider!r}")

    x0, variables = select_coil_variables(
        base_params,
        max_current_vars=int(args.max_current_vars),
        max_fourier_vars=int(args.max_fourier_vars),
    )
    if not variables:
        raise ValueError("No coil optimization variables selected. Increase --max-current-vars or --max-fourier-vars.")
    variable_manifest = variable_records(
        variables,
        base_params,
        current_step=float(args.current_step),
        dof_step=float(args.dof_step),
    )

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = make_free_boundary_indata(
        args.input,
        outdir / "input.direct_coil_qs",
        vmec_max_iter=int(args.vmec_max_iter),
        ftol=float(args.ftol),
        ns=int(args.ns),
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        nzeta=int(args.nzeta),
        beta_percent=float(args.beta),
        pressure_profile=str(args.pressure_profile),
        pressure_scale=float(args.pressure_scale),
        phiedge=float(args.phiedge),
    )

    objective_model = {
        "description": "Deterministic direct-coil free-boundary objective with VMEC residual, QS, aspect, and iota terms.",
        "qs_note": (
            "The QS term is evaluated from the accepted VMEC state. Full coil-to-Boozer/QS exact "
            "gradients through adaptive free-boundary branch selection remain a separate promotion gate."
        ),
        "helicity_m": int(args.helicity_m),
        "helicity_n": int(args.helicity_n),
        "qs_surfaces": parse_float_list(str(args.qs_surfaces)),
        "qs_ntheta": int(args.qs_ntheta),
        "qs_nphi": int(args.qs_nphi),
        "target_aspect": float(args.target_aspect),
        "target_iota": float(args.target_iota),
        "residual_weight": float(args.residual_weight),
        "qs_weight": float(args.qs_weight),
        "aspect_weight": float(args.aspect_weight),
        "iota_weight": float(args.iota_weight),
        "failure_objective": float(args.failure_objective),
    }
    vmec_config = {
        "input_template": args.input,
        "generated_input": input_path,
        "external_field_provider_kind": "direct_coils",
        "mgrid_file": "DIRECT_COILS",
        "uses_generated_mgrid": False,
        "vmec_max_iter": int(args.vmec_max_iter),
        "ftol": float(args.ftol),
        "ns": int(args.ns),
        "mpol": int(args.mpol),
        "ntor": int(args.ntor),
        "nzeta": int(args.nzeta),
        "beta_percent": float(args.beta),
        "pressure_profile": str(args.pressure_profile),
        "pressure_scale": float(args.pressure_scale),
        "phiedge": float(args.phiedge),
        "activate_fsq": float(args.activate_fsq),
        "jit_forces": bool(args.jit_forces),
    }
    optimizer_config = {
        "method": "Powell",
        "max_iter": int(args.max_iter),
        "max_evals": int(args.max_evals),
        "xtol": float(args.xtol),
        "ftol": float(args.optimizer_ftol),
    }
    same_branch_report_config = {
        "enabled": bool(args.write_same_branch_report),
        "mode": str(args.same_branch_report_mode),
        "ad_mode": str(args.same_branch_report_ad_mode),
        "vector_keys": list(parse_same_branch_vector_keys(getattr(args, "same_branch_report_vector_keys", None))),
        "default_derivative_detail": (
            "direct vector JVP for several physical scalars"
            if str(args.same_branch_report_mode) == "vector" and str(args.same_branch_report_ad_mode) == "direct"
            else "user-selected report mode"
        ),
        "contract": (
            "production-forward values plus fixed accepted-branch replay derivatives; "
            "does not differentiate adaptive host branch selection"
        ),
        "eps": float(args.same_branch_report_eps),
        "max_iter": int(args.same_branch_report_max_iter or args.vmec_max_iter),
        "anchor": str(getattr(args, "same_branch_report_anchor", "best")),
        "diagnostic_disable_analytic": bool(getattr(args, "same_branch_report_disable_analytic", False)),
        "diagnostic_freeze_vacuum_field": bool(getattr(args, "same_branch_report_freeze_vacuum_field", False)),
        "diagnostic_freeze_bsqvac": bool(getattr(args, "same_branch_report_freeze_bsqvac", False)),
        "nestor_solve_mode": str(getattr(args, "same_branch_report_nestor_solve_mode", "dense")),
        "nestor_operator_solver": str(getattr(args, "same_branch_report_nestor_operator_solver", "gmres")),
        "nestor_operator_tol": float(getattr(args, "same_branch_report_nestor_operator_tol", 1.0e-11)),
        "nestor_operator_atol": float(getattr(args, "same_branch_report_nestor_operator_atol", 1.0e-13)),
        "nestor_operator_maxiter": getattr(args, "same_branch_report_nestor_operator_maxiter", None),
        "nestor_operator_restart": getattr(args, "same_branch_report_nestor_operator_restart", None),
    }
    same_branch_derivative_proposal_config = {
        "enabled": bool(args.same_branch_derivative_proposal),
        "requires_same_branch_report": True,
        "scope": (
            "one fixed-accepted-branch directional proposal followed by a "
            "normal complete-solve objective evaluation"
        ),
        "step_size": float(args.same_branch_proposal_step),
        "differentiates_adaptive_controller": False,
    }
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    if bool(args.dry_run):
        summary = {
            "phase": "single-stage-direct-coil-validation",
            "scope": "deterministic coil-only direct-coil free-boundary QS optimization example",
            "dry_run": True,
            "plasma_boundary_optimized": False,
            "single_stage_limitations": SINGLE_STAGE_LIMITATIONS,
            "optimized_variables": variable_manifest,
            "objective_model": objective_model,
            "provider": provider_metadata,
            "baseline_coils": coil_diagnostics(base_params),
            "vmec_config": vmec_config,
            "optimizer_config": optimizer_config,
            "same_branch_report_config": same_branch_report_config,
            "same_branch_derivative_proposal_config": same_branch_derivative_proposal_config,
            "input": input_path,
            "outdir": outdir,
            "history_json": outdir / "history.json",
            "best_wout": outdir / "wout_best_direct_coil_qs.nc",
        }
        write_json(outdir / "summary.json", summary)
        print(f"Dry run: wrote {outdir / 'summary.json'} without running VMEC or the optimizer.")
        return summary

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

        try:
            run, wall_s = run_direct_free_boundary(
                input_path,
                params,
                vmec_max_iter=int(args.vmec_max_iter),
                activate_fsq=float(args.activate_fsq),
                jit_forces=bool(args.jit_forces),
            )
            provisional = summarize_run(
                run,
                params,
                objective=np.nan,
                wall_s=wall_s,
                target_aspect=float(args.target_aspect),
                target_iota=float(args.target_iota),
                helicity_m=int(args.helicity_m),
                helicity_n=int(args.helicity_n),
                qs_surfaces=parse_float_list(str(args.qs_surfaces)),
                qs_ntheta=int(args.qs_ntheta),
                qs_nphi=int(args.qs_nphi),
            )
            objective_terms = objective_terms_from_summary(
                provisional,
                residual_weight=float(args.residual_weight),
                qs_weight=float(args.qs_weight),
                aspect_weight=float(args.aspect_weight),
                iota_weight=float(args.iota_weight),
            )
            objective = float(objective_terms["total"])
            provisional["objective"] = objective
            provisional["objective_terms"] = objective_terms
            entry = {
                "eval": eval_id,
                "x": np.asarray(x, dtype=float).tolist(),
                "variables": variable_manifest,
                "coil_diagnostics": coil_diagnostics(params),
                "summary": provisional,
            }
            if best is None or objective < float(best["summary"]["objective"]):
                best = entry
                write_wout_from_fixed_boundary_run(outdir / "wout_best_direct_coil_qs.nc", run, include_fsq=True)
            print(
                f"eval={eval_id:03d} objective={objective:.6e} "
                f"residual={provisional['residual_proxy']:.3e} qs={provisional['qs_total']} "
                f"aspect={provisional['aspect']} "
                f"mean_iota={provisional['mean_iota']} "
                f"residual_term={objective_terms['residual']['contribution']:.3e} "
                f"qs_term={objective_terms['quasisymmetry']['contribution']:.3e} "
                f"aspect_term={objective_terms['aspect']['contribution']:.3e} "
                f"iota_term={objective_terms['mean_iota']['contribution']:.3e} wall_s={wall_s:.2f}",
                flush=True,
            )
        except Exception as exc:
            objective = float(args.failure_objective)
            entry = {
                "eval": eval_id,
                "x": np.asarray(x, dtype=float).tolist(),
                "variables": variable_manifest,
                "coil_diagnostics": coil_diagnostics(params),
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
        "phase": "single-stage-direct-coil-validation",
        "scope": "deterministic coil-only direct-coil free-boundary QS optimization example",
        "dry_run": False,
        "plasma_boundary_optimized": False,
        "single_stage_limitations": SINGLE_STAGE_LIMITATIONS,
        "optimized_variables": variable_manifest,
        "objective_model": objective_model,
        "provider": provider_metadata,
        "baseline_coils": coil_diagnostics(base_params),
        "vmec_config": vmec_config,
        "optimizer_config": optimizer_config,
        "same_branch_report_config": same_branch_report_config,
        "same_branch_derivative_proposal_config": same_branch_derivative_proposal_config,
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
        "best_wout": outdir / "wout_best_direct_coil_qs.nc",
    }
    if bool(args.write_same_branch_report):
        report_best_before_derivative_proposal = best
        report_params, report_anchor = same_branch_report_anchor_params(
            base_params,
            best,
            variables,
            args,
        )
        report_path = write_same_branch_validation_report(
            input_path=input_path,
            base_params=report_params,
            variables=variables,
            args=args,
            outdir=outdir,
            report_anchor=report_anchor,
        )
        summary["same_branch_complete_solve_report"] = report_path
        summary["same_branch_complete_solve_report_anchor"] = report_anchor
        summary["same_branch_complete_solve_report_final_best_status"] = {
            "report_generated_before_derivative_proposal": bool(args.same_branch_derivative_proposal),
            "final_best_changed_after_report": False,
            "report_matches_final_best": True,
        }
        if bool(args.same_branch_derivative_proposal):
            report_data = json.loads(report_path.read_text())
            proposal = same_branch_derivative_proposal_from_report(
                report_data,
                objective_model,
                best,
                step_size=float(args.same_branch_proposal_step),
            )
            if proposal.get("available"):
                previous_best = best
                trial_objective = evaluate(np.asarray(proposal["trial_x"], dtype=float))
                trial_entry = history[-1]
                proposal["trial_eval"] = int(trial_entry["eval"])
                proposal["trial_objective"] = float(trial_objective)
                proposal["previous_best_objective"] = (
                    None if previous_best is None else float(previous_best["summary"]["objective"])
                )
                proposal["accepted_by_complete_solve"] = bool(best is trial_entry)
                final_best_changed_after_report = bool(best is trial_entry)
                summary["same_branch_complete_solve_report_final_best_status"] = {
                    "report_generated_before_derivative_proposal": True,
                    "final_best_changed_after_report": final_best_changed_after_report,
                    "report_matches_final_best": not final_best_changed_after_report,
                    "note": (
                        "The same-branch report is the derivative evidence used to form the trial. "
                        "If the normal complete solve accepts that trial, rerun the report at the "
                        "new best point for final-point derivative evidence."
                    ),
                }
            elif report_best_before_derivative_proposal is not best:
                summary["same_branch_complete_solve_report_final_best_status"] = {
                    "report_generated_before_derivative_proposal": True,
                    "final_best_changed_after_report": True,
                    "report_matches_final_best": False,
                    "note": "The final best point changed after the report was written.",
                }
            summary["same_branch_derivative_proposal"] = proposal
        else:
            summary["same_branch_derivative_proposal"] = {
                "available": False,
                "reason": "not requested",
            }
    elif bool(args.same_branch_derivative_proposal):
        summary["same_branch_derivative_proposal"] = {
            "available": False,
            "reason": "--same-branch-derivative-proposal requires --write-same-branch-report",
        }
    summary["best"] = best
    write_json(outdir / "summary.json", summary)
    print(f"Wrote {outdir / 'history.json'}")
    print(f"Wrote {outdir / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Use tiny defaults for a fast direct-coil QS smoke.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configuration and coil/variable diagnostics without running VMEC or the optimizer.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--provider", choices=("essos", "circle"), default="essos")
    parser.add_argument("--coils-json", type=Path, default=None)
    parser.add_argument("--current-scale", type=float, default=1.0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Direct-coil field point chunk size. By default, synthetic circle runs are "
            "unchunked and ESSOS runs use 256. Use 0 to disable chunking explicitly."
        ),
    )
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
    parser.add_argument("--beta", type=float, default=0.0, help="Nominal beta percent for --pressure-profile standard.")
    parser.add_argument(
        "--pressure-profile",
        choices=("standard", "linear-scale"),
        default="standard",
        help=(
            "Pressure-profile model. 'standard' uses e*(ne*Te+ni*Ti) with "
            "Landreman-style beta scaling. 'linear-scale' uses the legacy "
            "PRES_SCALE*(1-s) profile."
        ),
    )
    parser.add_argument("--pressure-scale", type=float, default=0.0, help="Legacy PRES_SCALE for --pressure-profile linear-scale.")
    parser.add_argument(
        "--phiedge",
        type=float,
        default=DEFAULT_FREE_BOUNDARY_PHIEDGE,
        help="PHIEDGE override matching the unit-scale ESSOS LP-QA coil/input fixture.",
    )
    parser.add_argument("--activate-fsq", type=float, default=1.0e99)
    parser.add_argument(
        "--jit-forces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use JIT force kernels; --no-jit-forces is a parity/debug escape hatch.",
    )
    parser.add_argument("--max-current-vars", type=int, default=1)
    parser.add_argument("--max-fourier-vars", type=int, default=1)
    parser.add_argument("--current-step", type=float, default=0.02)
    parser.add_argument("--dof-step", type=float, default=1.0e-3)
    parser.add_argument("--target-aspect", type=float, default=6.0)
    parser.add_argument("--target-iota", type=float, default=0.4)
    parser.add_argument(
        "--helicity-m",
        type=int,
        default=1,
        help="QS helicity m for the VMEC-state quasisymmetry-ratio residual; QH uses 1.",
    )
    parser.add_argument(
        "--helicity-n",
        type=int,
        default=0,
        help="QS helicity n for the VMEC-state quasisymmetry-ratio residual; QA uses 0, QH typically uses -1.",
    )
    parser.add_argument(
        "--qs-surfaces",
        default="0.25,0.5,0.75",
        help="Comma/space-separated normalized toroidal-flux surfaces for the QS residual.",
    )
    parser.add_argument("--qs-ntheta", type=int, default=31, help="Angular theta grid for the QS residual.")
    parser.add_argument("--qs-nphi", type=int, default=32, help="Angular phi grid for the QS residual.")
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--qs-weight", type=float, default=1.0)
    parser.add_argument("--aspect-weight", type=float, default=1.0e-2)
    parser.add_argument("--iota-weight", type=float, default=1.0)
    parser.add_argument("--failure-objective", type=float, default=1.0e30)
    parser.add_argument(
        "--write-same-branch-report",
        action="store_true",
        help="After the optimization, write an opt-in same-branch complete-solve FD validation report.",
    )
    parser.add_argument(
        "--same-branch-report-anchor",
        choices=("best", "initial"),
        default="best",
        help=(
            "Coil point for --write-same-branch-report. The default validates "
            "the best optimized coil point; 'initial' preserves the older "
            "initial-coil diagnostic."
        ),
    )
    parser.add_argument("--same-branch-report-eps", type=float, default=1.0e-4)
    parser.add_argument(
        "--same-branch-report-mode",
        choices=("scalar", "vector", "none"),
        default="vector",
        help=(
            "Derivative detail for --write-same-branch-report. 'vector' is the default "
            "validated production-report path: in direct mode it reports JVP "
            "directional derivatives for several physical scalars without materializing "
            "the full Jacobian. 'scalar' validates one branch-local physical-scalar "
            "gradient. 'none' writes only complete-solve FD diagnostics."
        ),
    )
    parser.add_argument(
        "--same-branch-report-scalar-key",
        choices=("state_norm", "aspect", "qs_total", "lcfs_boundary_moment", "accepted_bnormal_rms"),
        default="qs_total",
        help=(
            "Physical scalar validated by --same-branch-report-mode scalar. "
            "Use 'state_norm' as a non-physics replay-graph timing probe, "
            "'aspect' for a cheap physical scalar, or 'qs_total' for the QS-relevant scalar."
        ),
    )
    parser.add_argument(
        "--same-branch-report-vector-keys",
        default=",".join(DEFAULT_SAME_BRANCH_VECTOR_KEYS),
        help=(
            "Comma/space-separated physical scalars for --same-branch-report-mode vector. "
            f"Supported: {', '.join(SUPPORTED_SAME_BRANCH_VECTOR_KEYS)}. "
            "Use state_norm as a non-physics replay-graph timing probe. "
            "Use all supported keys for broader validation, or the default smaller "
            "aspect,qs_total set for lower cold JVP graph cost. Final-state-only "
            f"keys ({', '.join(STATE_ONLY_SAME_BRANCH_KEYS)}) use a compact replay "
            "that omits accepted-history RMS arrays; accepted_bnormal_rms keeps "
            "the full-history path."
        ),
    )
    parser.add_argument(
        "--same-branch-report-ad-mode",
        choices=("direct", "custom_vjp"),
        default="direct",
        help=(
            "Accepted-branch AD path for scalar/vector derivative reports. "
            "'direct' differentiates the fixed replay directly and is faster; "
            "'custom_vjp' exercises the explicit custom-VJP wrapper."
        ),
    )
    parser.add_argument(
        "--same-branch-report-disable-jit-preconditioner",
        action="store_true",
        help=(
            "Diagnostic only: use the non-JIT radial preconditioner apply inside "
            "branch-local accepted replay to isolate cold JVP graph construction."
        ),
    )
    parser.add_argument(
        "--same-branch-report-disable-analytic",
        action="store_true",
        help=(
            "Diagnostic only: omit analytic NESTOR terms from branch-local accepted replay "
            "to isolate graph construction cost. This changes the replay operator and is "
            "not a promoted physics-validation path."
        ),
    )
    parser.add_argument(
        "--same-branch-report-freeze-bsqvac",
        action="store_true",
        help=(
            "Diagnostic only: reuse accepted-trace bsqvac instead of differentiably recomputing "
            "the direct-coil/NESTOR vacuum response. This isolates strict VMEC update graph cost "
            "and is not a promoted physics-validation path."
        ),
    )
    parser.add_argument(
        "--same-branch-report-freeze-vacuum-field",
        action="store_true",
        help=(
            "Diagnostic only: reuse accepted-trace vacuum-field projection arrays while still "
            "running JAX NESTOR/source assembly. This isolates Biot-Savart/projection graph cost "
            "from NESTOR graph cost and is not a promoted physics-validation path."
        ),
    )
    parser.add_argument(
        "--same-branch-report-nestor-solve-mode",
        choices=("dense", "matrix_free", "operator", "operator_gmres", "gmres", "bicgstab"),
        default="dense",
        help=(
            "NESTOR/source solve used inside the fixed accepted-branch replay. "
            "The default dense path is the promoted validation path; matrix_free/gmres/bicgstab "
            "exercise the opt-in matrix-free response seam for profiling."
        ),
    )
    parser.add_argument(
        "--same-branch-report-nestor-operator-solver",
        choices=("gmres", "bicgstab"),
        default="gmres",
        help="Krylov solver for --same-branch-report-nestor-solve-mode matrix_free/operator.",
    )
    parser.add_argument(
        "--same-branch-report-nestor-operator-tol",
        type=float,
        default=1.0e-11,
        help="Relative tolerance for the matrix-free NESTOR/source Krylov solve.",
    )
    parser.add_argument(
        "--same-branch-report-nestor-operator-atol",
        type=float,
        default=1.0e-13,
        help="Absolute tolerance for the matrix-free NESTOR/source Krylov solve.",
    )
    parser.add_argument(
        "--same-branch-report-nestor-operator-maxiter",
        type=int,
        default=None,
        help="Optional maximum Krylov iterations for the matrix-free NESTOR/source solve.",
    )
    parser.add_argument(
        "--same-branch-report-nestor-operator-restart",
        type=int,
        default=None,
        help="Optional GMRES restart length for the matrix-free NESTOR/source solve.",
    )
    parser.add_argument(
        "--same-branch-report-max-iter",
        type=int,
        default=None,
        help="Inner iterations for --write-same-branch-report; defaults to --vmec-max-iter.",
    )
    parser.add_argument(
        "--same-branch-derivative-proposal",
        action="store_true",
        help=(
            "Opt-in only: after Powell, use the same-branch vector/JVP report "
            "to propose one directional coil step, then evaluate that trial "
            "with the normal complete-solve objective. This does not "
            "differentiate adaptive host branch selection."
        ),
    )
    parser.add_argument(
        "--same-branch-proposal-step",
        type=float,
        default=0.05,
        help="Optimizer-coordinate step length for --same-branch-derivative-proposal.",
    )
    return parser


def apply_smoke_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        if args.chunk_size is None:
            args.chunk_size = None if args.provider == "circle" else 256
        elif int(args.chunk_size) <= 0:
            args.chunk_size = None
        args.max_iter = 1 if args.max_iter is None else args.max_iter
        args.max_evals = 3 if args.max_evals is None else args.max_evals
        # VMEC free-boundary cadence turns on the vacuum/NESTOR path only after
        # the first iteration, so the smoke run needs at least two inner steps.
        args.vmec_max_iter = 2 if args.vmec_max_iter is None else args.vmec_max_iter
        args.ns = 12 if args.ns is None else args.ns
        args.mpol = 3 if args.mpol is None else args.mpol
        args.ntor = 2 if args.ntor is None else args.ntor
        args.nzeta = 4 if args.nzeta is None else args.nzeta
        return args

    if args.chunk_size is None:
        args.chunk_size = None if args.provider == "circle" else 256
    elif int(args.chunk_size) <= 0:
        args.chunk_size = None
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

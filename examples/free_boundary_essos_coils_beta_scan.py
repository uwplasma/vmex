#!/usr/bin/env python
"""Free-boundary beta scan from ESSOS Landreman-Paul QA coils.

This example demonstrates both free-boundary external-field backends:

1. ESSOS coils -> mgrid file -> vmec_jax free-boundary solve.
2. ESSOS coils -> vmec_jax direct JAX Biot-Savart provider -> free-boundary solve.

The direct-coil provider path avoids writing an mgrid file.  Its field sampling
is differentiable with respect to coil Fourier coefficients and currents.  The
full production NESTOR/free-boundary adjoint is still phase-2 work, so this
example should be treated as a forward research lane plus provider-gradient
foundation rather than a publication claim for full-solve exact adjoints.

Run from the repository root:

    export ESSOS_ROOT=/path/to/ESSOS_mgrid_pr
    export ESSOS_INPUT_DIR=$ESSOS_ROOT/examples/input_files
    PYTHONPATH=.:$ESSOS_ROOT:$PYTHONPATH python examples/free_boundary_essos_coils_beta_scan.py

Use smaller settings for a quick smoke run:

    export ESSOS_ROOT=/path/to/ESSOS_mgrid_pr
    export ESSOS_INPUT_DIR=$ESSOS_ROOT/examples/input_files
    PYTHONPATH=.:$ESSOS_ROOT:$PYTHONPATH python examples/free_boundary_essos_coils_beta_scan.py --betas 0 1 --max-iter 2 --mgrid-nr 8 --mgrid-nz 8 --mgrid-nphi 4 --activate-fsq 1e99

By default, finite pressure is built from the standard density/temperature
profiles used by the SIMSOPT finite-beta/bootstrap examples,
``p = e * (ne*Te + ni*Ti)``.  Use ``--pressure-profile linear-scale`` only for
legacy plumbing probes that need the older ``PRES_SCALE*(1-s)`` profile.

The default ESSOS LP-QA coil JSON is unit-scale.  Use
``examples/data/input.LandremanPaul2021_QA_lowres`` with unit-scale mgrid
bounds, or pass ``--allow-scale-mismatch`` when deliberately testing a scaled
plasma/coil pair.  High-resolution promotion runs should use staged radial
continuation, for example ``--ns-array 16,31,51,101 --ftol-array
1e-8,1e-10,1e-11,1e-12``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import from_essos_coils
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.profiles import pressure_profile_to_vmec_am, standard_finite_beta_profiles
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state, read_wout

DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"
DEFAULT_RESULTS = REPO_ROOT / "results" / "free_boundary_essos_coils_beta_scan"
DEFAULT_NOMINAL_BETA_PERCENT = (0.0, 1.0, 2.0)

# Legacy linear pressure scale retained for regression/debug runs. The default
# pressure model below uses density/temperature-derived finite-beta profiles.
PRESSURE_SCALE_FOR_ONE_PERCENT_BETA = 1000.0
DEFAULT_FREE_BOUNDARY_PHIEDGE = -0.025
DEFAULT_PRESSURE_PROFILE = "standard"


def _candidate_essos_input_dirs() -> list[Path]:
    candidates = []
    user_env = None
    import os

    if os.getenv("ESSOS_INPUT_DIR"):
        user_env = Path(os.environ["ESSOS_INPUT_DIR"]).expanduser()
    if user_env is not None:
        candidates.append(user_env)
    candidates.extend(
        [
            REPO_ROOT.parent / "ESSOS_mgrid_pr" / "examples" / "input_files",
            REPO_ROOT.parent / "ESSOS" / "examples" / "input_files",
            Path.cwd() / "examples" / "input_files",
        ]
    )
    return candidates


def find_essos_landreman_paul_qa_coils() -> Path:
    """Find the ESSOS Landreman-Paul QA coil JSON in local example assets."""

    name = "ESSOS_biot_savart_LandremanPaulQA.json"
    for directory in _candidate_essos_input_dirs():
        path = directory / name
        if path.exists():
            return path
    searched = "\n  ".join(str(p) for p in _candidate_essos_input_dirs())
    raise FileNotFoundError(
        f"Could not find {name}. Set ESSOS_INPUT_DIR to the ESSOS examples/input_files directory. Searched:\n  {searched}"
    )


def make_free_boundary_indata(
    base_indata,
    *,
    beta_percent: float,
    mgrid_file: str,
    niter: int,
    ftol: float,
    ns: int,
    mpol: int,
    ntor: int,
    nzeta: int,
    ns_array: list[int] | None = None,
    niter_array: list[int] | None = None,
    ftol_array: list[float] | None = None,
    pressure_scale_for_one_percent_beta: float = PRESSURE_SCALE_FOR_ONE_PERCENT_BETA,
    pressure_profile: str = DEFAULT_PRESSURE_PROFILE,
    phiedge: float | None = None,
) -> Any:
    """Create a small free-boundary input deck for one nominal beta."""

    indata = deepcopy(base_indata)
    ns_values = [int(ns)] if ns_array is None else [int(value) for value in ns_array]
    niter_values = [int(niter)] if niter_array is None else [int(value) for value in niter_array]
    ftol_values = [float(ftol)] if ftol_array is None else [float(value) for value in ftol_array]
    if not (len(ns_values) == len(niter_values) == len(ftol_values)):
        raise ValueError(
            "ns_array, niter_array, and ftol_array must have the same length: "
            f"{len(ns_values)}, {len(niter_values)}, {len(ftol_values)}"
        )
    indata.scalars["LFREEB"] = True
    indata.scalars["MGRID_FILE"] = str(mgrid_file)
    indata.scalars["EXTCUR"] = [1.0]
    indata.scalars["NS_ARRAY"] = ns_values
    indata.scalars["NITER_ARRAY"] = niter_values
    indata.scalars["FTOL_ARRAY"] = ftol_values
    indata.scalars["NITER"] = int(niter_values[-1])
    indata.scalars["FTOL"] = float(ftol_values[-1])
    indata.scalars["MPOL"] = int(mpol)
    indata.scalars["NTOR"] = int(ntor)
    indata.scalars["NZETA"] = int(nzeta)
    indata.scalars["NTHETA"] = 0
    indata.scalars["NVACSKIP"] = max(1, int(nzeta))
    pressure_profile = str(pressure_profile).strip().lower()
    if pressure_profile == "standard":
        profiles = standard_finite_beta_profiles(float(beta_percent))
        am, pres_scale = pressure_profile_to_vmec_am(profiles.pressure_pa, pres_scale=1.0)
        indata.scalars["PMASS_TYPE"] = "power_series"
        indata.scalars["AM"] = am
        indata.scalars["PRES_SCALE"] = pres_scale
    elif pressure_profile in {"linear", "linear-scale", "legacy"}:
        indata.scalars["PMASS_TYPE"] = "power_series"
        # p(s) = PRES_SCALE * (1 - s), retained for legacy sensitivity probes.
        indata.scalars["AM"] = [1.0, -1.0]
        indata.scalars["PRES_SCALE"] = float(pressure_scale_for_one_percent_beta) * float(beta_percent)
    else:
        raise ValueError("pressure_profile must be 'standard' or 'linear-scale'")
    if phiedge is not None:
        indata.scalars["PHIEDGE"] = float(phiedge)
    return indata


def _parse_number_list(value: str | None, *, cast):
    if value is None:
        return None
    items = [item.strip() for item in str(value).replace(",", " ").split() if item.strip()]
    return [cast(item) for item in items]


def _coil_plasma_scale_summary(coils, indata) -> dict[str, float]:
    """Return simple geometry/field scales that catch mismatched fixtures."""

    gamma = np.asarray(coils.gamma, dtype=float)
    coil_r = np.sqrt(gamma[..., 0] ** 2 + gamma[..., 1] ** 2)
    coeffs = indata.indexed.get("RBC", {})
    r0 = float(coeffs.get((0, 0), coeffs.get((0, 0, 0), 0.0)) or 0.0)
    r_modes = [abs(float(value)) for key, value in coeffs.items() if key != (0, 0)]
    plasma_r_span = float(sum(r_modes)) if r_modes else 0.0
    return {
        "coil_r_min": float(np.nanmin(coil_r)),
        "coil_r_max": float(np.nanmax(coil_r)),
        "coil_r_mean": float(np.nanmean(coil_r)),
        "plasma_r0": r0,
        "plasma_r_span_estimate": plasma_r_span,
        "coil_to_plasma_major_radius_ratio": float(np.nanmean(coil_r) / r0) if r0 else float("nan"),
    }


def _vmec_input_n_from_wout_xn(xn_value: float, *, nfp: int) -> int:
    """Convert VMEC WOUT ``xn`` convention back to namelist ``n``."""

    if nfp > 0:
        scaled = float(xn_value) / float(nfp)
        rounded = int(round(scaled))
        if abs(scaled - rounded) < 1.0e-8:
            return rounded
    return int(round(float(xn_value)))


def continue_indata_from_wout_boundary(base_indata, wout) -> Any:
    """Warm-start a free-boundary input from the accepted WOUT LCFS and axis.

    VMEC free-boundary pressure ramps are substantially more robust when the
    next pressure point starts from the previously accepted free-boundary LCFS
    rather than from the original vacuum-boundary guess.  This helper updates
    only geometry/axis fields; pressure, mgrid/direct-provider settings, and
    run controls remain owned by :func:`make_free_boundary_indata`.
    """

    indata = deepcopy(base_indata)
    nfp = max(1, int(getattr(wout, "nfp", indata.get_int("NFP", 1))))
    mpol = int(indata.get_int("MPOL", getattr(wout, "mpol", 0)))
    ntor = int(indata.get_int("NTOR", getattr(wout, "ntor", 0)))
    lasym = bool(indata.get_bool("LASYM", getattr(wout, "lasym", False)))

    boundary_maps: dict[str, dict[tuple[int, int], float]] = {"RBC": {}, "ZBS": {}}
    if lasym:
        boundary_maps["RBS"] = {}
        boundary_maps["ZBC"] = {}

    xm = np.asarray(wout.xm, dtype=int)
    xn = np.asarray(wout.xn, dtype=float)
    rmnc = np.asarray(wout.rmnc, dtype=float)[-1]
    zmns = np.asarray(wout.zmns, dtype=float)[-1]
    rmns = np.asarray(getattr(wout, "rmns", np.zeros_like(wout.rmnc)), dtype=float)[-1]
    zmnc = np.asarray(getattr(wout, "zmnc", np.zeros_like(wout.zmns)), dtype=float)[-1]
    for k, (m_i, xn_i) in enumerate(zip(xm, xn, strict=True)):
        m = int(m_i)
        n = _vmec_input_n_from_wout_xn(float(xn_i), nfp=nfp)
        if m < 0 or m > mpol or abs(n) > ntor:
            continue
        boundary_maps["RBC"][(n, m)] = float(rmnc[k])
        boundary_maps["ZBS"][(n, m)] = float(zmns[k])
        if lasym:
            boundary_maps["RBS"][(n, m)] = float(rmns[k])
            boundary_maps["ZBC"][(n, m)] = float(zmnc[k])

    for name in ("RBC", "ZBS", "RBS", "ZBC"):
        if name in boundary_maps:
            indata.indexed[name] = boundary_maps[name]
        else:
            indata.indexed.pop(name, None)

    if hasattr(wout, "raxis_cc"):
        indata.scalars["RAXIS_CC"] = [float(x) for x in np.asarray(wout.raxis_cc, dtype=float)]
    if hasattr(wout, "zaxis_cs"):
        indata.scalars["ZAXIS_CS"] = [float(x) for x in np.asarray(wout.zaxis_cs, dtype=float)]
    if lasym and hasattr(wout, "raxis_cs"):
        indata.scalars["RAXIS_CS"] = [float(x) for x in np.asarray(wout.raxis_cs, dtype=float)]
    if lasym and hasattr(wout, "zaxis_cc"):
        indata.scalars["ZAXIS_CC"] = [float(x) for x in np.asarray(wout.zaxis_cc, dtype=float)]
    return indata


def _summary_is_promotable_for_pressure_continuation(summary: dict[str, Any], *, max_fsq: float) -> bool:
    values = [summary.get("fsqr"), summary.get("fsqz"), summary.get("fsql")]
    try:
        fsq_total = float(sum(float(v) for v in values if v is not None))
    except Exception:
        return False
    return bool(np.isfinite(fsq_total) and fsq_total <= float(max_fsq) and Path(str(summary.get("wout", ""))).exists())


def summarize_run(run, wout_path: Path, *, backend: str, beta_percent: float, wall_s: float) -> dict[str, Any]:
    """Collect lightweight scalar diagnostics for the JSON summary."""

    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}

    def _history_head_tail(name: str, *, n: int = 6) -> dict[str, Any] | None:
        arr = diag.get(name)
        if arr is None:
            return None
        try:
            values = np.asarray(arr)
            if values.size == 0:
                return {"count": 0, "head": [], "tail": []}
            flat = values.reshape(-1)
            return {
                "count": int(flat.size),
                "head": [float(x) for x in flat[:n]],
                "tail": [float(x) for x in flat[-n:]],
            }
        except Exception:
            return None

    summary: dict[str, Any] = {
        "backend": backend,
        "nominal_beta_percent": float(beta_percent),
        "wall_s": float(wall_s),
        "wout": str(wout_path),
        "n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        "fsqr": None,
        "fsqz": None,
        "fsql": None,
        "aspect": None,
        "mean_iota": None,
        "pressure_scale": float(PRESSURE_SCALE_FOR_ONE_PERCENT_BETA) * float(beta_percent),
        "max_pressure": None,
        "wp": None,
        "wb": None,
        "beta_proxy": None,
        "beta_proxy_percent": None,
        "free_boundary_ivac": None,
        "free_boundary_nestor_model": None,
        "free_boundary_vacuum_stub": None,
        "free_boundary_activate_fsq": None,
        "free_boundary_bnormal_rms": None,
        "free_boundary_bsqvac_rms": None,
        "free_boundary_gsource_rms": None,
        "free_boundary_bnormal_rms_history": _history_head_tail("freeb_nestor_bnormal_rms_history"),
        "free_boundary_bsqvac_rms_history": _history_head_tail("freeb_nestor_bsqvac_rms_history"),
        "free_boundary_gsource_rms_history": _history_head_tail("freeb_nestor_gsource_rms_history"),
        "free_boundary_source_reused_history": _history_head_tail("freeb_nestor_source_reused_history"),
    }
    for key in ("final_fsqr", "final_fsqz", "final_fsql"):
        val = diag.get(key)
        if val is not None:
            summary[key.replace("final_", "")] = float(val)
    freeb_diag = diag.get("free_boundary")
    if isinstance(freeb_diag, dict):
        summary["free_boundary_ivac"] = freeb_diag.get("ivac")
        summary["free_boundary_nestor_model"] = freeb_diag.get("nestor_model")
        summary["free_boundary_vacuum_stub"] = freeb_diag.get("vacuum_stub")
        summary["free_boundary_activate_fsq"] = freeb_diag.get("activate_fsq")
        nestor_diag = freeb_diag.get("last_nestor_diagnostics")
        if isinstance(nestor_diag, dict):
            summary["free_boundary_bnormal_rms"] = nestor_diag.get("bnormal_rms")
            summary["free_boundary_bsqvac_rms"] = nestor_diag.get("bsqvac_rms")
            summary["free_boundary_gsource_rms"] = nestor_diag.get("gsource_rms")
    try:
        summary["aspect"] = float(equilibrium_aspect_ratio_from_state(state=run.state, static=run.static))
    except Exception:
        pass
    try:
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
        )
        summary["mean_iota"] = float(np.nanmean(np.asarray(iotas, dtype=float)))
    except Exception:
        pass
    try:
        wout = read_wout(wout_path)
        for name in ("fsqr", "fsqz", "fsql", "aspect"):
            if hasattr(wout, name):
                value = float(getattr(wout, name))
                if np.isfinite(value):
                    summary[name] = value
        summary["max_pressure"] = float(np.nanmax(np.asarray(wout.presf, dtype=float)))
        summary["wp"] = float(wout.wp)
        summary["wb"] = float(wout.wb)
        if float(wout.wb) != 0.0:
            summary["beta_proxy"] = float(wout.wp) / float(wout.wb)
            summary["beta_proxy_percent"] = 100.0 * float(wout.wp) / float(wout.wb)
    except Exception:
        pass
    return summary


def summarize_existing_wout(wout_path: Path, *, backend: str, beta_percent: float) -> dict[str, Any]:
    """Collect summary diagnostics from a pre-existing WOUT file.

    This supports resuming long pressure-continuation scans after an interrupt.
    The resumed entry has no in-process free-boundary history because only the
    accepted equilibrium was persisted.
    """

    wout = read_wout(wout_path)
    iotas = np.asarray(getattr(wout, "iotas", getattr(wout, "iotaf", [])), dtype=float)
    wb = float(getattr(wout, "wb", 0.0))
    wp = float(getattr(wout, "wp", 0.0))
    summary: dict[str, Any] = {
        "backend": backend,
        "nominal_beta_percent": float(beta_percent),
        "wall_s": 0.0,
        "wout": str(wout_path),
        "n_iter": None,
        "fsqr": float(getattr(wout, "fsqr", np.nan)),
        "fsqz": float(getattr(wout, "fsqz", np.nan)),
        "fsql": float(getattr(wout, "fsql", np.nan)),
        "aspect": float(getattr(wout, "aspect", np.nan)),
        "mean_iota": float(np.nanmean(iotas)) if iotas.size else None,
        "pressure_scale": float(PRESSURE_SCALE_FOR_ONE_PERCENT_BETA) * float(beta_percent),
        "max_pressure": float(np.nanmax(np.asarray(getattr(wout, "presf", [np.nan]), dtype=float))),
        "wp": wp,
        "wb": wb,
        "beta_proxy": wp / wb if wb != 0.0 else None,
        "beta_proxy_percent": 100.0 * wp / wb if wb != 0.0 else None,
        "free_boundary_ivac": None,
        "free_boundary_nestor_model": None,
        "free_boundary_vacuum_stub": None,
        "free_boundary_activate_fsq": None,
        "free_boundary_bnormal_rms": None,
        "free_boundary_bsqvac_rms": None,
        "free_boundary_gsource_rms": None,
        "free_boundary_bnormal_rms_history": None,
        "free_boundary_bsqvac_rms_history": None,
        "free_boundary_gsource_rms_history": None,
        "free_boundary_source_reused_history": None,
        "resumed_from_existing_wout": True,
    }
    return summary


def _case_wout_path(output_dir: Path, *, backend: str, beta_percent: float) -> Path:
    return output_dir / f"wout_{backend}_beta_{float(beta_percent):.3f}.nc"


def _nominal_pressure_scale(pressure_scale_for_one_percent_beta: float, beta_percent: float) -> float:
    """Return the historical pressure-scale diagnostic used in scan summaries.

    The standard finite-beta profile writes VMEC ``PRES_SCALE=1`` because the
    pressure magnitude is already encoded in the profile coefficients.  The
    scan summary keeps this nominal linear scale as a stable comparison field
    for existing post-processing and resume tests.
    """

    return float(pressure_scale_for_one_percent_beta) * float(beta_percent)


def _vmec_pres_scale(pressure_scale_for_one_percent_beta: float, beta_percent: float, pressure_profile: str) -> float:
    """Return the actual VMEC ``PRES_SCALE`` written for a pressure model."""

    if str(pressure_profile).strip().lower() == "standard":
        return 1.0
    return _nominal_pressure_scale(pressure_scale_for_one_percent_beta, beta_percent)


def _summary_payload(
    *,
    coils_json: Path,
    mgrid_file: Path,
    args,
    scale_summary: dict[str, float],
    ns_array: list[int] | None,
    niter_array: list[int] | None,
    ftol_array: list[float] | None,
    summaries: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    """Build the beta-scan JSON payload.

    The same payload is written after each completed case and again at normal
    exit.  Long high-resolution pressure scans can therefore be interrupted
    without losing metrics for already accepted beta points.
    """

    return {
        "complete": bool(complete),
        "coils_json": str(coils_json),
        "mgrid": str(mgrid_file),
        "coil_current_scale": float(args.coil_current_scale),
        "phiedge_override": None if args.phiedge is None else float(args.phiedge),
        "pressure_scale_for_one_percent_beta": float(args.pressure_scale_for_one_percent_beta),
        "pressure_profile": str(getattr(args, "pressure_profile", DEFAULT_PRESSURE_PROFILE)),
        "pressure_continuation": bool(args.pressure_continuation),
        "pressure_continuation_max_fsq": float(args.pressure_continuation_max_fsq),
        "direct_coil_source_reuse": not bool(args.disable_direct_coil_source_reuse),
        "direct_coil_trial_resample": bool(args.direct_coil_trial_resample),
        "direct_coil_limit_update_rms": bool(args.direct_coil_limit_update_rms),
        "coil_plasma_scale_summary": scale_summary,
        "ns_array": ns_array or [int(args.ns)],
        "niter_array": niter_array or [int(args.max_iter)],
        "ftol_array": ftol_array or [float(args.ftol)],
        "runs": summaries,
    }


def _write_summary_checkpoint(
    summary_path: Path,
    *,
    coils_json: Path,
    mgrid_file: Path,
    args,
    scale_summary: dict[str, float],
    ns_array: list[int] | None,
    niter_array: list[int] | None,
    ftol_array: list[float] | None,
    summaries: list[dict[str, Any]],
    complete: bool,
) -> None:
    summary_path.write_text(
        json.dumps(
            _summary_payload(
                coils_json=coils_json,
                mgrid_file=mgrid_file,
                args=args,
                scale_summary=scale_summary,
                ns_array=ns_array,
                niter_array=niter_array,
                ftol_array=ftol_array,
                summaries=summaries,
                complete=complete,
            ),
            indent=2,
        )
    )


def _resume_existing_case(
    *,
    output_dir: Path,
    backend: str,
    beta_percent: float,
    pressure_scale_for_one_percent_beta: float,
    pressure_profile: str = DEFAULT_PRESSURE_PROFILE,
) -> dict[str, Any] | None:
    """Return diagnostics for an existing case WOUT, if present."""

    wout_path = _case_wout_path(output_dir, backend=backend, beta_percent=beta_percent)
    if not wout_path.exists():
        return None
    summary = summarize_existing_wout(wout_path, backend=backend, beta_percent=beta_percent)
    summary["pressure_profile"] = str(pressure_profile)
    summary["pressure_scale"] = _nominal_pressure_scale(pressure_scale_for_one_percent_beta, beta_percent)
    summary["vmec_pres_scale"] = _vmec_pres_scale(pressure_scale_for_one_percent_beta, beta_percent, pressure_profile)
    return summary


def run_one_case(
    *,
    backend: str,
    input_path: Path,
    output_dir: Path,
    beta_percent: float,
    pressure_scale_for_one_percent_beta: float,
    max_iter: int,
    activate_fsq: float | None,
    pressure_profile: str = DEFAULT_PRESSURE_PROFILE,
    direct_coil_params=None,
    direct_coil_source_reuse: bool = True,
    direct_coil_trial_resample: bool = False,
    direct_coil_limit_update_rms: bool = False,
) -> dict[str, Any]:
    """Run one mgrid or direct-coil free-boundary case."""

    t0 = time.perf_counter()
    if backend == "mgrid":
        run = run_free_boundary(
            input_path,
            max_iter=int(max_iter),
            multigrid=False,
            verbose=False,
            jit_forces=False,
            free_boundary_activate_fsq=activate_fsq,
        )
    elif backend == "direct":
        run = run_free_boundary(
            input_path,
            max_iter=int(max_iter),
            multigrid=False,
            verbose=False,
            jit_forces=False,
            external_field_provider_kind="direct_coils",
            external_field_provider_static={
                "allow_source_reuse": bool(direct_coil_source_reuse),
                "resample_trial_bsqvac": bool(direct_coil_trial_resample),
            },
            external_field_provider_params=direct_coil_params,
            free_boundary_activate_fsq=activate_fsq,
            limit_update_rms=bool(direct_coil_limit_update_rms),
        )
    else:
        raise ValueError(f"unknown backend {backend!r}")
    wall_s = time.perf_counter() - t0
    wout_path = _case_wout_path(output_dir, backend=backend, beta_percent=beta_percent)
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    summary = summarize_run(run, wout_path, backend=backend, beta_percent=beta_percent, wall_s=wall_s)
    summary["pressure_profile"] = str(pressure_profile)
    summary["pressure_scale"] = _nominal_pressure_scale(pressure_scale_for_one_percent_beta, beta_percent)
    summary["vmec_pres_scale"] = _vmec_pres_scale(pressure_scale_for_one_percent_beta, beta_percent, pressure_profile)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--coils-json", type=Path, default=None)
    parser.add_argument("--betas", type=float, nargs="*", default=list(DEFAULT_NOMINAL_BETA_PERCENT))
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--ftol", type=float, default=1.0e-8)
    parser.add_argument("--ns", type=int, default=12)
    parser.add_argument(
        "--ns-array",
        type=str,
        default=None,
        help="Optional comma/space-separated multigrid NS_ARRAY, e.g. '16,31,51,101'.",
    )
    parser.add_argument(
        "--niter-array",
        type=str,
        default=None,
        help="Optional comma/space-separated NITER_ARRAY matching --ns-array.",
    )
    parser.add_argument(
        "--ftol-array",
        type=str,
        default=None,
        help="Optional comma/space-separated FTOL_ARRAY matching --ns-array.",
    )
    parser.add_argument("--mpol", type=int, default=5)
    parser.add_argument("--ntor", type=int, default=5)
    parser.add_argument("--mgrid-nr", type=int, default=32)
    parser.add_argument("--mgrid-nz", type=int, default=32)
    parser.add_argument("--mgrid-nphi", type=int, default=16)
    parser.add_argument("--mgrid-rmin", type=float, default=0.1)
    parser.add_argument("--mgrid-rmax", type=float, default=2.5)
    parser.add_argument("--mgrid-zmin", type=float, default=-1.4)
    parser.add_argument("--mgrid-zmax", type=float, default=1.4)
    parser.add_argument(
        "--pressure-scale-for-one-percent-beta",
        type=float,
        default=PRESSURE_SCALE_FOR_ONE_PERCENT_BETA,
        help=(
            "Legacy scale factor for --pressure-profile linear-scale. "
            "The default standard profile derives pressure from density and "
            "temperature profiles, so this value is ignored unless the legacy "
            "linear profile is requested."
        ),
    )
    parser.add_argument(
        "--pressure-profile",
        choices=("standard", "linear-scale"),
        default=DEFAULT_PRESSURE_PROFILE,
        help=(
            "Pressure-profile model. 'standard' uses e*(ne*Te+ni*Ti) with "
            "Landreman-style beta scaling; 'linear-scale' preserves the old "
            "PRES_SCALE*(1-s) plumbing probe."
        ),
    )
    parser.add_argument(
        "--pressure-continuation",
        action="store_true",
        help=(
            "Warm-start each finite-pressure point from the previously converged "
            "free-boundary LCFS for the same backend. This is the recommended "
            "promotion path for LP-QA finite-pressure scans because direct "
            "pressure jumps can leave the free-boundary iteration outside the "
            "convergent basin."
        ),
    )
    parser.add_argument(
        "--pressure-continuation-max-fsq",
        type=float,
        default=1.0e-6,
        help=(
            "Maximum fsqr+fsqz+fsql allowed before a run is accepted as the "
            "seed for the next --pressure-continuation step."
        ),
    )
    parser.add_argument(
        "--phiedge",
        type=float,
        default=DEFAULT_FREE_BOUNDARY_PHIEDGE,
        help=(
            "Optional PHIEDGE override. This is useful when matching the VMEC "
            "toroidal-flux scale and sign to a direct-coil validation fixture."
        ),
    )
    parser.add_argument(
        "--coil-current-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply ESSOS coil currents before generating both the mgrid and "
            "the direct-coil provider. The default preserves the fixture exactly; "
            "larger values are useful for finite-pressure sensitivity probes."
        ),
    )
    parser.add_argument(
        "--activate-fsq",
        type=float,
        default=1.0e99,
        help=(
            "Free-boundary activation threshold for this finite-pressure research example. "
            "Use 1e-3 for literal VMEC2000 cadence parity; the default forces immediate "
            "vacuum coupling so direct-coil and mgrid backends are exercised in short runs."
        ),
    )
    parser.add_argument("--skip-mgrid-runs", action="store_true")
    parser.add_argument("--skip-direct-runs", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Skip any beta/backend case whose WOUT already exists in --outdir. "
            "With --pressure-continuation, the existing WOUT is also promoted "
            "as the seed for the next pressure point if its residual satisfies "
            "--pressure-continuation-max-fsq."
        ),
    )
    parser.add_argument(
        "--disable-direct-coil-source-reuse",
        action="store_true",
        help=(
            "Disable VMEC-style cached-source reuse for direct-coil runs. "
            "The default keeps reuse enabled because the coils are fixed during "
            "each equilibrium solve; lower-level tests still cover the no-stale-"
            "source path when provider parameters change between calls."
        ),
    )
    parser.add_argument(
        "--direct-coil-trial-resample",
        action="store_true",
        help=(
            "Recompute the direct-coil vacuum field on trial/rejected boundaries. "
            "This is useful for phase-2 exact-control experiments. The default "
            "keeps VMEC-style accepted-state vacuum during trial scoring, which "
            "is more robust for finite-pressure LP-QA continuation."
        ),
    )
    parser.add_argument(
        "--direct-coil-limit-update-rms",
        action="store_true",
        help=(
            "Enable the guarded VMEC coefficient-update limiter for direct-coil "
            "runs. This is a phase-2 diagnostics control; it is off by default "
            "so provider parity tests preserve the baseline VMEC control path."
        ),
    )
    parser.add_argument(
        "--allow-scale-mismatch",
        action="store_true",
        help="Disable the coil/plasma scale sanity warning for non-default research fixtures.",
    )
    args = parser.parse_args(argv)
    ns_array = _parse_number_list(args.ns_array, cast=int)
    niter_array = _parse_number_list(args.niter_array, cast=int)
    ftol_array = _parse_number_list(args.ftol_array, cast=float)
    if ns_array is not None:
        if niter_array is None:
            niter_array = [int(args.max_iter)] * len(ns_array)
        if ftol_array is None:
            ftol_array = [float(args.ftol)] * len(ns_array)

    try:
        from essos.coils import Coils_from_json
    except Exception as exc:
        raise ImportError(
            "This example requires ESSOS with Coils_from_json. Install the ESSOS mgrid branch or set PYTHONPATH to it."
        ) from exc

    coils_json = args.coils_json or find_essos_landreman_paul_qa_coils()
    coils = Coils_from_json(str(coils_json))
    if not args.skip_mgrid_runs and not hasattr(coils, "to_mgrid"):
        raise AttributeError(
            "ESSOS Coils.to_mgrid is not available. Use the ESSOS PR branch that adds mgrid generation from coils."
        )
    if float(args.coil_current_scale) != 1.0:
        coils.currents_scale = float(coils.currents_scale) * float(args.coil_current_scale)
    direct_params = from_essos_coils(coils, chunk_size=256)

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    mgrid_file = outdir / "mgrid_landreman_paul_qa_from_essos.nc"
    if not args.skip_mgrid_runs:
        print(f"Writing mgrid from ESSOS coils: {mgrid_file}")
        coils.to_mgrid(
            mgrid_file,
            nr=args.mgrid_nr,
            nphi=args.mgrid_nphi,
            nz=args.mgrid_nz,
            rmin=args.mgrid_rmin,
            rmax=args.mgrid_rmax,
            zmin=args.mgrid_zmin,
            zmax=args.mgrid_zmax,
            nfp=int(coils.nfp),
        )

    base_indata = read_indata(args.input)
    scale_summary = _coil_plasma_scale_summary(coils, base_indata)
    ratio = scale_summary["coil_to_plasma_major_radius_ratio"]
    if not args.allow_scale_mismatch and np.isfinite(ratio) and not (0.5 <= ratio <= 2.0):
        print(
            "WARNING: ESSOS coil/plasma scale mismatch. "
            f"coil <R> / plasma RBC(0,0) = {ratio:.3g}. "
            "The default ESSOS LP-QA coils are unit-scale; use "
            "examples/data/input.LandremanPaul2021_QA_lowres or pass "
            "--allow-scale-mismatch for deliberate research scans.",
            flush=True,
        )
    summaries = []
    summary_path = outdir / "summary.json"
    continuation_bases: dict[str, Any] = {}
    continuation_has_promoted_seed: dict[str, bool] = {}
    if args.pressure_continuation:
        continuation_template = deepcopy(base_indata)
        continuation_template.scalars["MPOL"] = int(args.mpol)
        continuation_template.scalars["NTOR"] = int(args.ntor)
        continuation_bases = {
            "mgrid": deepcopy(continuation_template),
            "direct": deepcopy(continuation_template),
        }
        continuation_has_promoted_seed = {"mgrid": False, "direct": False}
    for beta_percent in args.betas:
        beta_tag = f"{float(beta_percent):.3f}".replace(".", "p")
        if not args.skip_mgrid_runs:
            mgrid_base = continuation_bases.get("mgrid", base_indata)
            mgrid_indata = make_free_boundary_indata(
                mgrid_base,
                beta_percent=beta_percent,
                mgrid_file=mgrid_file.name,
                niter=args.max_iter,
                ftol=args.ftol,
                ns=args.ns,
                mpol=args.mpol,
                ntor=args.ntor,
                nzeta=args.mgrid_nphi,
                ns_array=ns_array,
                niter_array=niter_array,
                ftol_array=ftol_array,
                pressure_scale_for_one_percent_beta=args.pressure_scale_for_one_percent_beta,
                pressure_profile=args.pressure_profile,
                phiedge=args.phiedge,
            )
            input_mgrid = outdir / f"input.lpqa_mgrid_beta_{beta_tag}"
            summary = (
                _resume_existing_case(
                    output_dir=outdir,
                    backend="mgrid",
                    beta_percent=beta_percent,
                    pressure_scale_for_one_percent_beta=args.pressure_scale_for_one_percent_beta,
                    pressure_profile=args.pressure_profile,
                )
                if args.resume_existing
                else None
            )
            if summary is not None:
                print(f"Resuming mgrid beta={beta_percent:.3f}% from existing WOUT: {summary['wout']}")
                summary["pressure_continuation_seeded_from_previous"] = bool(
                    continuation_has_promoted_seed.get("mgrid", False)
                )
                summary["pressure_continuation_promoted_seed"] = False
                if args.pressure_continuation and _summary_is_promotable_for_pressure_continuation(
                    summary, max_fsq=args.pressure_continuation_max_fsq
                ):
                    continuation_bases["mgrid"] = continue_indata_from_wout_boundary(
                        mgrid_base, read_wout(summary["wout"])
                    )
                    continuation_has_promoted_seed["mgrid"] = True
                    summary["pressure_continuation_promoted_seed"] = True
                summaries.append(summary)
                _write_summary_checkpoint(
                    summary_path,
                    coils_json=coils_json,
                    mgrid_file=mgrid_file,
                    args=args,
                    scale_summary=scale_summary,
                    ns_array=ns_array,
                    niter_array=niter_array,
                    ftol_array=ftol_array,
                    summaries=summaries,
                    complete=False,
                )
            else:
                write_indata(input_mgrid, mgrid_indata)
                print(f"Running mgrid beta={beta_percent:.3f}%: {input_mgrid}")
                summary = run_one_case(
                    backend="mgrid",
                    input_path=input_mgrid,
                    output_dir=outdir,
                    beta_percent=beta_percent,
                    pressure_scale_for_one_percent_beta=args.pressure_scale_for_one_percent_beta,
                    pressure_profile=args.pressure_profile,
                    max_iter=args.max_iter,
                    activate_fsq=args.activate_fsq,
                )
                summary["pressure_continuation_seeded_from_previous"] = bool(
                    continuation_has_promoted_seed.get("mgrid", False)
                )
                summary["pressure_continuation_promoted_seed"] = False
                if args.pressure_continuation and _summary_is_promotable_for_pressure_continuation(
                    summary, max_fsq=args.pressure_continuation_max_fsq
                ):
                    continuation_bases["mgrid"] = continue_indata_from_wout_boundary(
                        mgrid_base, read_wout(summary["wout"])
                    )
                    continuation_has_promoted_seed["mgrid"] = True
                    summary["pressure_continuation_promoted_seed"] = True
                summaries.append(summary)
                _write_summary_checkpoint(
                    summary_path,
                    coils_json=coils_json,
                    mgrid_file=mgrid_file,
                    args=args,
                    scale_summary=scale_summary,
                    ns_array=ns_array,
                    niter_array=niter_array,
                    ftol_array=ftol_array,
                    summaries=summaries,
                    complete=False,
                )

        if not args.skip_direct_runs:
            direct_base = continuation_bases.get("direct", base_indata)
            direct_indata = make_free_boundary_indata(
                direct_base,
                beta_percent=beta_percent,
                mgrid_file="DIRECT_COILS",
                niter=args.max_iter,
                ftol=args.ftol,
                ns=args.ns,
                mpol=args.mpol,
                ntor=args.ntor,
                nzeta=args.mgrid_nphi,
                ns_array=ns_array,
                niter_array=niter_array,
                ftol_array=ftol_array,
                pressure_scale_for_one_percent_beta=args.pressure_scale_for_one_percent_beta,
                pressure_profile=args.pressure_profile,
                phiedge=args.phiedge,
            )
            input_direct = outdir / f"input.lpqa_direct_beta_{beta_tag}"
            summary = (
                _resume_existing_case(
                    output_dir=outdir,
                    backend="direct",
                    beta_percent=beta_percent,
                    pressure_scale_for_one_percent_beta=args.pressure_scale_for_one_percent_beta,
                    pressure_profile=args.pressure_profile,
                )
                if args.resume_existing
                else None
            )
            if summary is not None:
                print(f"Resuming direct-coil beta={beta_percent:.3f}% from existing WOUT: {summary['wout']}")
                summary["pressure_continuation_seeded_from_previous"] = bool(
                    continuation_has_promoted_seed.get("direct", False)
                )
                summary["pressure_continuation_promoted_seed"] = False
                if args.pressure_continuation and _summary_is_promotable_for_pressure_continuation(
                    summary, max_fsq=args.pressure_continuation_max_fsq
                ):
                    continuation_bases["direct"] = continue_indata_from_wout_boundary(
                        direct_base, read_wout(summary["wout"])
                    )
                    continuation_has_promoted_seed["direct"] = True
                    summary["pressure_continuation_promoted_seed"] = True
                summaries.append(summary)
                _write_summary_checkpoint(
                    summary_path,
                    coils_json=coils_json,
                    mgrid_file=mgrid_file,
                    args=args,
                    scale_summary=scale_summary,
                    ns_array=ns_array,
                    niter_array=niter_array,
                    ftol_array=ftol_array,
                    summaries=summaries,
                    complete=False,
                )
            else:
                write_indata(input_direct, direct_indata)
                print(f"Running direct-coil beta={beta_percent:.3f}%: {input_direct}")
                summary = run_one_case(
                    backend="direct",
                    input_path=input_direct,
                    output_dir=outdir,
                    beta_percent=beta_percent,
                    pressure_scale_for_one_percent_beta=args.pressure_scale_for_one_percent_beta,
                    pressure_profile=args.pressure_profile,
                    max_iter=args.max_iter,
                    activate_fsq=args.activate_fsq,
                    direct_coil_params=direct_params,
                    direct_coil_source_reuse=not args.disable_direct_coil_source_reuse,
                    direct_coil_trial_resample=args.direct_coil_trial_resample,
                    direct_coil_limit_update_rms=args.direct_coil_limit_update_rms,
                )
                summary["pressure_continuation_seeded_from_previous"] = bool(
                    continuation_has_promoted_seed.get("direct", False)
                )
                summary["pressure_continuation_promoted_seed"] = False
                if args.pressure_continuation and _summary_is_promotable_for_pressure_continuation(
                    summary, max_fsq=args.pressure_continuation_max_fsq
                ):
                    continuation_bases["direct"] = continue_indata_from_wout_boundary(
                        direct_base, read_wout(summary["wout"])
                    )
                    continuation_has_promoted_seed["direct"] = True
                    summary["pressure_continuation_promoted_seed"] = True
                summaries.append(summary)
                _write_summary_checkpoint(
                    summary_path,
                    coils_json=coils_json,
                    mgrid_file=mgrid_file,
                    args=args,
                    scale_summary=scale_summary,
                    ns_array=ns_array,
                    niter_array=niter_array,
                    ftol_array=ftol_array,
                    summaries=summaries,
                    complete=False,
                )

    _write_summary_checkpoint(
        summary_path,
        coils_json=coils_json,
        mgrid_file=mgrid_file,
        args=args,
        scale_summary=scale_summary,
        ns_array=ns_array,
        niter_array=niter_array,
        ftol_array=ftol_array,
        summaries=summaries,
        complete=True,
    )
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

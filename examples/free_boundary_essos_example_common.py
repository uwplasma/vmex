"""Small helper functions for ESSOS free-boundary examples.

The example scripts keep the scientific workflow visible.  This module only
handles repeated boilerplate: locating the ESSOS Landreman-Paul QA coil file,
choosing a magnetic-grid box that contains the VMEC boundary, preparing a tiny
free-boundary input deck, and writing compact JSON summaries.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import coil_current_norm, coil_lengths
from vmec_jax.namelist import InData, write_indata
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state, read_wout


DEFAULT_INPUT = REPO_ROOT / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"
DEFAULT_COILS_JSON_NAME = "ESSOS_biot_savart_LandremanPaulQA.json"
DEFAULT_PRESSURE_SCALE = 34.46233666638


def free_boundary_workflow_metadata(*, direct_coils: bool) -> dict[str, Any]:
    """Return user-facing metadata that distinguishes the two example flows."""

    if direct_coils:
        return {
            "flow": "direct_essos_coils_no_mgrid",
            "field_backend": "direct_coils",
            "workflow_steps": [
                "load ESSOS coils",
                "convert coils to CoilFieldParams",
                "write VMEC input with MGRID_FILE='DIRECT_COILS'",
                "run vmec_jax with direct JAX Biot-Savart sampling",
            ],
            "python_provider_required": True,
            "uses_mgrid_file": False,
            "mgrid_compatibility_example": str(REPO_ROOT / "examples" / "free_boundary_essos_mgrid_forward.py"),
            "vmec_input_replay": (
                "MGRID_FILE='DIRECT_COILS' is a vmec_jax Python-provider tag. "
                "Replay this input through this example or run_free_boundary with CoilFieldParams, "
                "not as a bare mgrid/VMEC2000 input."
            ),
        }
    return {
        "flow": "essos_generated_mgrid_compatibility",
        "field_backend": "mgrid",
        "workflow_steps": [
            "load ESSOS coils",
            "sample coils onto a VMEC-compatible mgrid NetCDF",
            "write VMEC input that references the generated mgrid",
            "run vmec_jax with the mgrid compatibility backend",
        ],
        "python_provider_required": False,
        "uses_mgrid_file": True,
        "direct_coil_example": str(REPO_ROOT / "examples" / "free_boundary_essos_direct_forward.py"),
        "vmec_input_replay": (
            "The generated input is replayable by mgrid-compatible tooling when the referenced "
            "mgrid NetCDF is available at the MGRID_FILE path."
        ),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def candidate_essos_input_dirs() -> list[Path]:
    candidates: list[Path] = []
    if os.getenv("ESSOS_INPUT_DIR"):
        candidates.append(Path(os.environ["ESSOS_INPUT_DIR"]).expanduser())
    if os.getenv("ESSOS_ROOT"):
        candidates.append(Path(os.environ["ESSOS_ROOT"]).expanduser() / "examples" / "input_files")
    candidates.extend(
        [
            REPO_ROOT.parent / "ESSOS_mgrid_pr" / "examples" / "input_files",
            REPO_ROOT.parent / "ESSOS" / "examples" / "input_files",
        ]
    )
    return candidates


def find_essos_landreman_paul_qa_coils() -> Path:
    """Return the ESSOS Landreman-Paul QA coil JSON path."""

    for directory in candidate_essos_input_dirs():
        path = directory / DEFAULT_COILS_JSON_NAME
        if path.exists():
            return path
    searched = "\n  ".join(str(path) for path in candidate_essos_input_dirs())
    raise FileNotFoundError(
        f"Could not find {DEFAULT_COILS_JSON_NAME}. Set ESSOS_INPUT_DIR to the ESSOS examples/input_files "
        f"directory. Searched:\n  {searched}"
    )


def load_essos_coils(coils_json: Path | None = None) -> Any:
    """Load ESSOS coils, importing ESSOS only when the example runs."""

    try:
        from essos.coils import Coils_from_json
    except Exception as exc:
        raise ImportError("Install ESSOS or put an ESSOS checkout on PYTHONPATH to run this example.") from exc

    return Coils_from_json(str(coils_json or find_essos_landreman_paul_qa_coils()))


def sample_input_boundary_extents(indata: InData, *, ntheta: int = 96, nzeta: int = 96) -> dict[str, float]:
    """Sample the VMEC input boundary and return cylindrical R/Z extrema."""

    nfp = max(1, int(indata.scalars.get("NFP", 1)))
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / nfp, int(nzeta), endpoint=False)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")

    r = np.zeros_like(theta_grid, dtype=float)
    z = np.zeros_like(theta_grid, dtype=float)
    for name, trig in (("RBC", np.cos), ("RBS", np.sin)):
        for (m, n), value in (indata.indexed.get(name) or {}).items():
            r = r + float(value) * trig(int(m) * theta_grid - int(n) * nfp * zeta_grid)
    for name, trig in (("ZBC", np.cos), ("ZBS", np.sin)):
        for (m, n), value in (indata.indexed.get(name) or {}).items():
            z = z + float(value) * trig(int(m) * theta_grid - int(n) * nfp * zeta_grid)

    return {
        "boundary_rmin": float(np.min(r)),
        "boundary_rmax": float(np.max(r)),
        "boundary_zmin": float(np.min(z)),
        "boundary_zmax": float(np.max(z)),
    }


def mgrid_bounds_from_indata(
    indata: InData,
    *,
    padding_fraction: float = 0.30,
    min_padding: float = 0.50,
) -> dict[str, float]:
    """Choose mgrid bounds that contain the input boundary plus padding."""

    extents = sample_input_boundary_extents(indata)
    r_span = max(extents["boundary_rmax"] - extents["boundary_rmin"], 1.0e-12)
    z_span = max(extents["boundary_zmax"] - extents["boundary_zmin"], 1.0e-12)
    r_pad = max(float(min_padding), float(padding_fraction) * r_span)
    z_pad = max(float(min_padding), float(padding_fraction) * z_span)
    return {
        "rmin": max(1.0e-6, extents["boundary_rmin"] - r_pad),
        "rmax": extents["boundary_rmax"] + r_pad,
        "zmin": extents["boundary_zmin"] - z_pad,
        "zmax": extents["boundary_zmax"] + z_pad,
        **extents,
    }


def make_lpqa_free_boundary_indata(
    base_indata: InData,
    *,
    mgrid_file: str,
    ns: int = 7,
    max_iter: int = 2,
    ftol: float = 1.0e-8,
    mpol: int = 3,
    ntor: int = 2,
    nzeta: int = 8,
    nvacskip: int | None = None,
    pressure_scale: float = DEFAULT_PRESSURE_SCALE,
    phiedge_scale: float = 1.0,
    extcur_scale: float = 1.0,
) -> InData:
    """Return a low-resolution finite-pressure free-boundary LP-QA input."""

    indata = deepcopy(base_indata)
    phiedge = float(indata.scalars.get("PHIEDGE", 0.0)) * float(phiedge_scale)
    indata.scalars.update(
        {
            "LFREEB": True,
            "MGRID_FILE": str(mgrid_file),
            "EXTCUR": [float(extcur_scale)],
            "NS_ARRAY": [int(ns)],
            "NITER_ARRAY": [int(max_iter)],
            "FTOL_ARRAY": [float(ftol)],
            "NITER": int(max_iter),
            "FTOL": float(ftol),
            "MPOL": int(mpol),
            "NTOR": int(ntor),
            "NZETA": int(nzeta),
            "NTHETA": 0,
            "NVACSKIP": int(nvacskip if nvacskip is not None else nzeta),
            "PHIEDGE": phiedge,
            "PRES_SCALE": float(pressure_scale),
            "PMASS_TYPE": "power_series",
            "AM": [1.0, -1.0],
        }
    )
    return indata


def run_one_free_boundary_solve(
    *,
    input_path: Path,
    wout_path: Path,
    max_iter: int,
    jit_forces: bool,
    activate_fsq: float,
    external_field_provider_params: Any | None = None,
) -> tuple[Any | None, float]:
    """Run one free-boundary solve and write a WOUT."""

    kwargs: dict[str, Any] = {}
    if external_field_provider_params is not None:
        kwargs = {
            "external_field_provider_kind": "direct_coils",
            "external_field_provider_params": external_field_provider_params,
        }

    t0 = time.perf_counter()
    run = run_free_boundary(
        input_path,
        max_iter=int(max_iter),
        multigrid=False,
        verbose=False,
        jit_forces=bool(jit_forces),
        free_boundary_activate_fsq=float(activate_fsq),
        **kwargs,
    )
    wall_s = time.perf_counter() - t0
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    return run, wall_s


def summarize_free_boundary_run(
    *,
    backend: str,
    input_path: Path,
    wout_path: Path | None,
    wall_s: float,
    run: Any | None,
    coil_params: Any | None = None,
    mgrid_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Return a compact summary for examples and tests."""

    direct_coils = coil_params is not None and mgrid_path is None
    workflow = free_boundary_workflow_metadata(direct_coils=direct_coils)
    summary: dict[str, Any] = {
        "backend": str(backend),
        "flow": workflow["flow"],
        "workflow": workflow,
        "dry_run": bool(dry_run),
        "input": input_path,
        "wout": wout_path,
        "mgrid": mgrid_path,
        "external_field_provider_kind": "direct_coils" if direct_coils else "mgrid",
        "mgrid_file": "DIRECT_COILS" if direct_coils else mgrid_path,
        "uses_generated_mgrid": mgrid_path is not None,
        "wall_s": float(wall_s),
        "surface_dofs_optimized": False,
        "fsqr": None,
        "fsqz": None,
        "fsql": None,
        "aspect": None,
        "mean_iota": None,
        "free_boundary_vacuum_stub": None,
        "free_boundary_nestor_model": None,
        "free_boundary_bnormal_rms": None,
        "free_boundary_bsqvac_rms": None,
    }
    if coil_params is not None:
        summary["coil_current_norm"] = float(np.asarray(coil_current_norm(coil_params)))
        summary["coil_length_mean"] = float(np.mean(np.asarray(coil_lengths(coil_params), dtype=float)))
    if run is not None:
        diag = getattr(run.result, "diagnostics", {}) if getattr(run, "result", None) is not None else {}
        freeb = diag.get("free_boundary", {}) if isinstance(diag, dict) else {}
        nestor = freeb.get("last_nestor_diagnostics", {}) if isinstance(freeb, dict) else {}
        summary.update(
            {
                "n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
                "fsqr": diag.get("final_fsqr"),
                "fsqz": diag.get("final_fsqz"),
                "fsql": diag.get("final_fsql"),
                "free_boundary_vacuum_stub": freeb.get("vacuum_stub") if isinstance(freeb, dict) else None,
                "free_boundary_nestor_model": freeb.get("nestor_model") if isinstance(freeb, dict) else None,
                "free_boundary_bnormal_rms": nestor.get("bnormal_rms") if isinstance(nestor, dict) else None,
                "free_boundary_bsqvac_rms": nestor.get("bsqvac_rms") if isinstance(nestor, dict) else None,
            }
        )
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
    if wout_path is not None and Path(wout_path).exists():
        try:
            wout = read_wout(wout_path)
            for name in ("fsqr", "fsqz", "fsql", "aspect", "wb", "wp"):
                if hasattr(wout, name):
                    summary[name] = float(getattr(wout, name))
            if float(getattr(wout, "wb", 0.0)) != 0.0:
                summary["beta_proxy_percent"] = 100.0 * float(wout.wp) / float(wout.wb)
        except Exception:
            pass
    return summary


def write_example_outputs(input_path: Path, indata: InData, summary_path: Path, summary: dict[str, Any]) -> None:
    """Write the VMEC input and JSON summary."""

    write_indata(input_path, indata)
    summary_path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n")

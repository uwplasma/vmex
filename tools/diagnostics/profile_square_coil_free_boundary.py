#!/usr/bin/env python
"""Profile square-coil free-boundary solves through direct, mgrid, and VMEC2000 paths."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.toroidal_stellarator_mirror_hybrid_square_coils_free_boundary import (
    ExampleConfig,
    _case_label,
    _run_budget,
    _stage_values,
    build_square_coils,
    make_free_boundary_indata,
)
from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.external_fields import write_mgrid_from_coils
from vmec_jax.namelist import write_indata
from vmec_jax.toroidal_hybrid import evaluate_toroidal_hybrid_indata_boundary, recommended_square_axis_nzeta
from vmec_jax.vmec2000_exec import find_vmec2000_exec, run_xvmec2000


DEFAULT_OUTDIR = REPO_ROOT / "results" / "square_coil_freeb_backend_profile"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--beta-percent", type=float, default=0.0)
    p.add_argument("--mpol", type=int, default=6)
    p.add_argument("--ntor", type=int, default=12)
    p.add_argument("--ns", type=int, default=9)
    p.add_argument("--nzeta", type=int, default=24)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--ftol", type=float, default=1.0e-8)
    p.add_argument("--ns-array", default=None, help="Comma-separated VMEC multigrid NS_ARRAY override.")
    p.add_argument("--niter-array", default=None, help="Comma-separated VMEC multigrid NITER_ARRAY override.")
    p.add_argument("--ftol-array", default=None, help="Comma-separated VMEC multigrid FTOL_ARRAY override.")
    p.add_argument("--phiedge", type=float, default=None)
    p.add_argument("--delt", type=float, default=0.05)
    p.add_argument("--activate-fsq", type=float, default=1.0e-3)
    p.add_argument("--axis-kind", default="spline", choices=("spline", "superellipse"))
    p.add_argument("--axis-corner-factor", type=float, default=1.14)
    p.add_argument("--enforce-recommended-nzeta", action="store_true")
    p.add_argument("--n-coils-per-side", type=int, default=4)
    p.add_argument("--coil-segments", type=int, default=96)
    p.add_argument("--mgrid-nr", type=int, default=36)
    p.add_argument("--mgrid-nz", type=int, default=28)
    p.add_argument("--mgrid-nphi", type=int, default=None)
    p.add_argument("--mgrid-padding-fraction", type=float, default=0.35)
    p.add_argument("--mgrid-min-padding", type=float, default=0.15)
    p.add_argument("--skip-direct", action="store_true")
    p.add_argument("--skip-mgrid", action="store_true")
    p.add_argument("--run-vmec2000", action="store_true")
    p.add_argument("--vmec2000-exec", type=Path, default=None)
    p.add_argument("--vmec2000-timeout", type=float, default=600.0)
    p.add_argument("--jit-forces", action="store_true")
    return p


def _parse_int_list(raw: str | None, *, name: str) -> tuple[int, ...] | None:
    if raw is None or str(raw).strip() == "":
        return None
    values = tuple(int(tok.strip()) for tok in str(raw).replace(";", ",").split(",") if tok.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _parse_float_list(raw: str | None, *, name: str) -> tuple[float, ...] | None:
    if raw is None or str(raw).strip() == "":
        return None
    values = tuple(float(tok.strip()) for tok in str(raw).replace(";", ",").split(",") if tok.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _resolve_schedule(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    ns_array = _parse_int_list(args.ns_array, name="ns-array")
    niter_array = _parse_int_list(args.niter_array, name="niter-array")
    ftol_array = _parse_float_list(args.ftol_array, name="ftol-array")
    provided = [value is not None for value in (ns_array, niter_array, ftol_array)]
    if any(provided) and not all(provided):
        raise ValueError("--ns-array, --niter-array, and --ftol-array must be provided together")
    if ns_array is None or niter_array is None or ftol_array is None:
        return (int(args.ns),), (int(args.max_iter),), (float(args.ftol),)
    if not (len(ns_array) == len(niter_array) == len(ftol_array)):
        raise ValueError("--ns-array, --niter-array, and --ftol-array must have matching lengths")
    return ns_array, niter_array, ftol_array


def _tail_lines(path: Path | None, *, lines: int = 60) -> list[str]:
    if path is None or not Path(path).exists():
        return []
    return Path(path).read_text(errors="replace").splitlines()[-int(lines) :]


def _last_finite(values: Any) -> float | None:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return None
    finite = arr[np.isfinite(arr)]
    return None if finite.size == 0 else float(finite[-1])


def _classify_run(diag: dict[str, Any], residuals: dict[str, Any]) -> str:
    if bool(residuals.get("converged_strict", False)):
        return "converged_strict"
    if not bool(residuals.get("free_boundary_active", False)):
        return "free_boundary_not_activated"
    bad_resets = int(residuals.get("bad_resets") or 0)
    n_iter = int(residuals.get("n_iter") or 0)
    if n_iter > 0 and bad_resets >= max(5, n_iter // 2):
        return "bad_jacobian_or_restart_limited"
    component_sum = residuals.get("final_fsq_component_sum")
    requested = residuals.get("requested_ftol")
    if component_sum is not None and requested is not None:
        try:
            if float(component_sum) > 100.0 * float(requested):
                return "underconverged"
        except Exception:
            pass
    return "incomplete"


def _final_residuals(run: Any) -> dict[str, Any]:
    diag = run.result.diagnostics if run.result is not None else {}
    diag = diag if isinstance(diag, dict) else {}
    fsqr = diag.get("final_fsqr")
    fsqz = diag.get("final_fsqz")
    fsql = diag.get("final_fsql")
    values = [v for v in (fsqr, fsqz, fsql) if v is not None and np.isfinite(float(v))]
    freeb = diag.get("free_boundary", {}) if isinstance(diag.get("free_boundary", {}), dict) else {}
    nestor = freeb.get("last_nestor_diagnostics", {}) if isinstance(freeb.get("last_nestor_diagnostics", {}), dict) else {}
    model = str(freeb.get("nestor_model", "none"))
    out = {
        "n_iter": None if run.result is None else int(getattr(run.result, "n_iter", -1)),
        "converged": bool(diag.get("converged", False)),
        "converged_strict": bool(diag.get("converged_strict", False)),
        "requested_ftol": diag.get("requested_ftol"),
        "final_fsqr": fsqr,
        "final_fsqz": fsqz,
        "final_fsql": fsql,
        "final_fsq_component_sum": float(sum(float(v) for v in values)) if values else None,
        "bad_resets": diag.get("bad_resets"),
        "ijacob": diag.get("ijacob"),
        "final_residual_recomputed_on_accepted_state": diag.get("final_residual_recomputed_on_accepted_state"),
        "free_boundary_nestor_model": model,
        "free_boundary_active": bool(model.strip() and model != "none"),
        "free_boundary_bnormal_rms": nestor.get("bnormal_rms"),
        "free_boundary_bsqvac_rms": nestor.get("bsqvac_rms"),
        "free_boundary_couple_edge": freeb.get("couple_edge"),
        "free_boundary_activate_fsq": freeb.get("activate_fsq"),
        "free_boundary_last_ivac": freeb.get("ivac"),
        "free_boundary_last_ivacskip": freeb.get("ivacskip"),
        "free_boundary_last_nvacskip": freeb.get("nvacskip"),
        "free_boundary_last_nestor_solve_time_s": _last_finite(diag.get("freeb_nestor_solve_time_history")),
        "free_boundary_last_nestor_sample_time_s": _last_finite(diag.get("freeb_nestor_sample_time_history")),
    }
    out["stall_classification"] = _classify_run(diag, out)
    return out


def _mgrid_bounds(indata: Any, *, padding_fraction: float, min_padding: float) -> dict[str, float]:
    samples = evaluate_toroidal_hybrid_indata_boundary(indata, ntheta=96, nzeta=128)
    rmin = float(np.min(samples.R))
    rmax = float(np.max(samples.R))
    zmin = float(np.min(samples.Z))
    zmax = float(np.max(samples.Z))
    rpad = max(float(min_padding), float(padding_fraction) * max(rmax - rmin, 1.0e-6))
    zpad = max(float(min_padding), float(padding_fraction) * max(zmax - zmin, 1.0e-6))
    return {
        "rmin": max(1.0e-3, rmin - rpad),
        "rmax": rmax + rpad,
        "zmin": zmin - zpad,
        "zmax": zmax + zpad,
        "boundary_rmin": rmin,
        "boundary_rmax": rmax,
        "boundary_zmin": zmin,
        "boundary_zmax": zmax,
    }


def _run_jax_backend(
    *,
    input_path: Path,
    wout_path: Path,
    config: ExampleConfig,
    direct_params: Any | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if direct_params is not None:
        kwargs = {
            "external_field_provider_kind": "direct_coils",
            "external_field_provider_params": direct_params,
        }
    t0 = time.perf_counter()
    run = run_free_boundary(
        input_path,
        max_iter=_run_budget(config, restart_state=None),
        multigrid=bool(config.use_multigrid_schedule),
        multigrid_use_input_niter=True,
        verbose=False,
        jit_forces=config.jit_forces,
        free_boundary_activate_fsq=None
        if config.free_boundary_activate_fsq is None
        else float(config.free_boundary_activate_fsq),
        **kwargs,
    )
    wall_s = time.perf_counter() - t0
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    return {
        "status": "completed",
        "wall_s": float(wall_s),
        "input": input_path,
        "wout": wout_path,
        **_final_residuals(run),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    mgrid_nphi = int(args.nzeta if args.mgrid_nphi is None else args.mgrid_nphi)
    ns_array, niter_array, ftol_array = _resolve_schedule(args)
    recommended_nzeta = recommended_square_axis_nzeta(int(args.ntor))
    if bool(args.enforce_recommended_nzeta) and int(args.nzeta) < recommended_nzeta:
        raise ValueError(
            f"NZETA={int(args.nzeta)} is underresolved for NTOR={int(args.ntor)}; use at least {recommended_nzeta}"
        )
    config = ExampleConfig(
        outdir=outdir,
        betas_percent=(float(args.beta_percent),),
        n_coils_per_side=int(args.n_coils_per_side),
        coil_segments=int(args.coil_segments),
        plasma_axis_kind=str(args.axis_kind),
        plasma_axis_spline_corner_radius_factor=float(args.axis_corner_factor),
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        ns=int(ns_array[-1]),
        ns_array=ns_array,
        nzeta=int(args.nzeta),
        max_iter=int(niter_array[-1]),
        ftol=float(ftol_array[-1]),
        phiedge=float(args.phiedge) if args.phiedge is not None else ExampleConfig().phiedge,
        niter_array=niter_array,
        ftol_array=ftol_array,
        use_multigrid_schedule=len(ns_array) > 1,
        delt=float(args.delt),
        free_boundary_activate_fsq=float(args.activate_fsq),
        beta_continuation_restart=False,
        jit_forces=bool(args.jit_forces),
        write_plots=False,
    )
    ns_values, niter_values, ftol_values = _stage_values(config)
    coils = build_square_coils(config)
    label = _case_label(float(args.beta_percent))
    direct_input = outdir / f"input.square_{label}_direct"
    mgrid_input = outdir / f"input.square_{label}_mgrid"
    direct_wout = outdir / f"wout_square_{label}_direct.nc"
    mgrid_wout = outdir / f"wout_square_{label}_mgrid.nc"
    mgrid_path = outdir / "mgrid_square_coils.nc"

    base_indata = make_free_boundary_indata(config, beta_percent=float(args.beta_percent))
    write_indata(direct_input, base_indata)
    bounds = _mgrid_bounds(
        base_indata,
        padding_fraction=float(args.mgrid_padding_fraction),
        min_padding=float(args.mgrid_min_padding),
    )
    write_mgrid_from_coils(
        mgrid_path,
        coils.params,
        rmin=bounds["rmin"],
        rmax=bounds["rmax"],
        zmin=bounds["zmin"],
        zmax=bounds["zmax"],
        nr=int(args.mgrid_nr),
        nz=int(args.mgrid_nz),
        nphi=mgrid_nphi,
        nfp=int(config.nfp),
    )
    mgrid_indata = deepcopy(base_indata)
    mgrid_indata.scalars["MGRID_FILE"] = mgrid_path.name
    write_indata(mgrid_input, mgrid_indata)

    payload: dict[str, Any] = {
        "schema": "square_coil_free_boundary_backend_profile",
        "configuration": {
            "beta_percent": float(args.beta_percent),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "ns": int(ns_array[-1]),
            "nzeta": int(args.nzeta),
            "recommended_nzeta": int(recommended_nzeta),
            "nzeta_underrecommended": bool(int(args.nzeta) < int(recommended_nzeta)),
            "max_iter": int(niter_array[-1]),
            "ftol": float(ftol_array[-1]),
            "phiedge": float(config.phiedge),
            "delt": float(args.delt),
            "activate_fsq": float(args.activate_fsq),
            "axis_kind": str(args.axis_kind),
            "axis_corner_factor": float(args.axis_corner_factor),
            "use_multigrid_schedule": bool(len(ns_array) > 1),
            "ns_array": ns_values,
            "niter_array": niter_values,
            "ftol_array": ftol_values,
        },
        "mgrid": {
            "path": mgrid_path,
            "nr": int(args.mgrid_nr),
            "nz": int(args.mgrid_nz),
            "nphi": int(mgrid_nphi),
            **bounds,
        },
        "backends": {},
    }
    if not args.skip_direct:
        payload["backends"]["vmec_jax_direct"] = _run_jax_backend(
            input_path=direct_input,
            wout_path=direct_wout,
            config=config,
            direct_params=coils.params,
        )
    if not args.skip_mgrid:
        payload["backends"]["vmec_jax_mgrid"] = _run_jax_backend(
            input_path=mgrid_input,
            wout_path=mgrid_wout,
            config=config,
            direct_params=None,
        )
    if bool(args.run_vmec2000):
        exe = args.vmec2000_exec or find_vmec2000_exec()
        if exe is None:
            payload["backends"]["vmec2000_mgrid"] = {"status": "skipped_missing_xvmec2000"}
        else:
            t0 = time.perf_counter()
            try:
                run = run_xvmec2000(
                    mgrid_input,
                    exec_path=exe,
                    workdir=outdir / "vmec2000_mgrid",
                    timeout_s=float(args.vmec2000_timeout),
                    keep_workdir=True,
                )
                rows = [row for stage in run.stages for row in stage.rows]
                last = rows[-1] if rows else None
                payload["backends"]["vmec2000_mgrid"] = {
                    "status": "completed" if run.returncode == 0 else "nonzero_exit",
                    "returncode": int(run.returncode),
                    "wall_s": float(time.perf_counter() - t0),
                    "exec": exe,
                    "workdir": run.workdir,
                    "threed1": run.threed1_path,
                    "stdout_tail": run.stdout.splitlines()[-40:],
                    "stderr_tail": run.stderr.splitlines()[-40:],
                    "threed1_tail": _tail_lines(run.threed1_path, lines=80),
                    "iteration_row_count": len(rows),
                    "last_row": None
                    if last is None
                    else {
                        "it": int(last.it),
                        "fsqr": float(last.fsqr),
                        "fsqz": float(last.fsqz),
                        "fsql": float(last.fsql),
                        "delt0r": last.delt0r,
                        "delbsq": last.delbsq,
                        "fedge": last.fedge,
                    },
                }
            except Exception as exc:
                payload["backends"]["vmec2000_mgrid"] = {"status": "failed", "error": repr(exc)}

    report = outdir / "square_coil_free_boundary_backend_profile.json"
    report.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

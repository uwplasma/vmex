#!/usr/bin/env python
"""Global-to-local QI basin scan by interpolating VMEC boundary modes.

This diagnostic is intended for far-seed QI robustness work.  It makes large,
deterministic moves from a seed boundary toward a known same-NFP QI reference,
solves each interpolated boundary, and ranks the solved states with the same QI
diagnostics used by the public examples.  Promising interpolation points can
then be passed to ``qi_constraint_policy_scan.py`` for local QI/mirror cleanup.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_INPUT = REPO_ROOT / "examples" / "data" / "input.QI_stel_seed_3127"
DEFAULT_REFERENCE_INPUT = REPO_ROOT / "examples" / "data" / "input.nfp3_QI_fixed_resolution_final"
DEFAULT_OUT_ROOT = Path("/tmp/vmec_jax_qi_boundary_interpolation_scan")


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _parse_lambdas(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("At least one interpolation lambda is required.")
    for value in values:
        if not np.isfinite(value):
            raise ValueError(f"Interpolation lambda must be finite, got {value!r}.")
    return values


def _diagnose(vj: Any, run: Any, *, args: argparse.Namespace) -> dict[str, Any]:
    from vmec_jax.quasi_isodynamic.diagnostics import QIDiagnosticOptions, QISeedSuitabilityTargets, annotate_qi_seed_suitability

    options = QIDiagnosticOptions(
        surfaces=np.asarray(args.surfaces, dtype=float),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        nphi=int(args.nphi),
        nalpha=int(args.nalpha),
        n_bounce=int(args.n_bounce),
        include_bounce_endpoints=True,
        phimin=0.0,
        mirror_threshold=float(args.max_mirror_ratio),
        mirror_ntheta=int(args.mirror_ntheta),
        mirror_nphi=int(args.mirror_nphi),
        elongation_threshold=float(args.max_elongation),
        elongation_ntheta=int(args.elongation_ntheta),
        elongation_nphi=int(args.elongation_nphi),
        jit_booz=True,
    )
    diagnostics = vj.qi_diagnostics_from_state(
        state=run.state,
        static=run.static,
        indata=run.indata,
        signgs=run.signgs,
        surfaces=np.asarray(args.surfaces, dtype=float),
        options=options,
    )
    return annotate_qi_seed_suitability(
        diagnostics,
        targets=QISeedSuitabilityTargets(
            smooth_qi_max=float(args.smooth_qi_max),
            legacy_qi_max=float(args.legacy_qi_max),
            target_aspect=float(args.target_aspect),
            abs_iota_min=float(args.abs_iota_min),
            mirror_ratio_max=float(args.max_mirror_ratio),
            max_elongation=float(args.max_elongation),
        ),
    )


def _record_float(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def write_summary(records: list[dict[str, Any]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    fields = [
        "lambda",
        "smooth_qi",
        "legacy_qi",
        "mirror",
        "elongation",
        "iota",
        "aspect",
        "selection",
        "selection_reason",
        "wall_time_s",
        "input",
        "wout",
    ]
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def run_scan(args: argparse.Namespace) -> list[dict[str, Any]]:
    _ensure_repo_on_path()
    import vmec_jax as vj
    from vmec_jax._compat import enable_x64
    from vmec_jax.namelist import read_indata, write_indata

    enable_x64(True)
    seed = read_indata(args.seed_input)
    reference = read_indata(args.reference_input)
    if int(seed.get_int("NFP", -1)) != int(reference.get_int("NFP", -2)):
        raise ValueError(
            "Boundary interpolation requires same-NFP inputs; "
            f"got seed NFP={seed.get_int('NFP')} and reference NFP={reference.get_int('NFP')}."
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    keys = tuple(key.strip().upper() for key in args.keys.split(",") if key.strip())
    plan = {
        "seed_input": str(args.seed_input),
        "reference_input": str(args.reference_input),
        "out_root": str(args.out_root),
        "lambdas": list(args.lambdas),
        "keys": list(keys),
        "max_mode": args.max_mode,
        "diagnostics": {
            "surfaces": list(args.surfaces),
            "mboz": args.mboz,
            "nboz": args.nboz,
            "nphi": args.nphi,
            "nalpha": args.nalpha,
            "n_bounce": args.n_bounce,
        },
    }
    (args.out_root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    records: list[dict[str, Any]] = []
    for lam in args.lambdas:
        start = time.perf_counter()
        case_dir = args.out_root / f"lambda_{lam:.3f}".replace(".", "p").replace("-", "m")
        input_path = case_dir / "input.interpolated"
        wout_path = case_dir / "wout_interpolated.nc"
        try:
            candidate = vj.interpolate_indata_boundary(seed, reference, lam, keys=keys, max_mode=args.max_mode)
            write_indata(input_path, candidate)
            run = vj.run_fixed_boundary(
                input_path,
                max_iter=int(args.max_iter),
                solver_device=args.solver_device,
                verbose=False,
            )
            vj.write_wout_from_fixed_boundary_run(wout_path, run)
            diagnostics = _diagnose(vj, run, args=args)
            selected = bool(diagnostics.get("qi_engineering_gate_passed"))
            record = {
                "lambda": float(lam),
                "smooth_qi": _record_float(diagnostics, "qi_smooth_total"),
                "legacy_qi": _record_float(diagnostics, "qi_legacy_total"),
                "mirror": _record_float(diagnostics, "qi_mirror_ratio_max"),
                "elongation": _record_float(diagnostics, "qi_max_elongation"),
                "iota": _record_float(diagnostics, "mean_iota"),
                "aspect": _record_float(diagnostics, "aspect"),
                "selection": "selected" if selected else "rejected",
                "selection_reason": "; ".join(diagnostics.get("qi_failure_reasons", [])),
                "wall_time_s": time.perf_counter() - start,
                "input": str(input_path),
                "wout": str(wout_path),
            }
            (case_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
        except Exception as exc:  # noqa: BLE001 - keep the scan moving.
            record = {
                "lambda": float(lam),
                "smooth_qi": None,
                "legacy_qi": None,
                "mirror": None,
                "elongation": None,
                "iota": None,
                "aspect": None,
                "selection": "error",
                "selection_reason": f"{type(exc).__name__}: {exc}",
                "wall_time_s": time.perf_counter() - start,
                "input": str(input_path),
                "wout": str(wout_path),
            }
        records.append(record)
        write_summary(records, args.out_root)
        print(
            f"lambda={lam:.3f}: {record['selection']} "
            f"smooth={record['smooth_qi']} legacy={record['legacy_qi']} "
            f"mirror={record['mirror']} iota={record['iota']} aspect={record['aspect']}",
            flush=True,
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-input", type=Path, default=DEFAULT_SEED_INPUT)
    parser.add_argument("--reference-input", type=Path, default=DEFAULT_REFERENCE_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--lambdas", type=_parse_lambdas, default=_parse_lambdas("0,0.25,0.5,0.75,1"))
    parser.add_argument("--keys", default="RBC,ZBS,RBS,ZBC")
    parser.add_argument("--max-mode", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--surfaces", type=_parse_lambdas, default=_parse_lambdas("0.5,1.0"))
    parser.add_argument("--mboz", type=int, default=6)
    parser.add_argument("--nboz", type=int, default=6)
    parser.add_argument("--nphi", type=int, default=31)
    parser.add_argument("--nalpha", type=int, default=7)
    parser.add_argument("--n-bounce", type=int, default=9)
    parser.add_argument("--mirror-ntheta", type=int, default=32)
    parser.add_argument("--mirror-nphi", type=int, default=32)
    parser.add_argument("--elongation-ntheta", type=int, default=24)
    parser.add_argument("--elongation-nphi", type=int, default=8)
    parser.add_argument("--target-aspect", type=float, default=4.0)
    parser.add_argument("--abs-iota-min", type=float, default=0.41)
    parser.add_argument("--smooth-qi-max", type=float, default=2.0e-3)
    parser.add_argument("--legacy-qi-max", type=float, default=2.0e-3)
    parser.add_argument("--max-mirror-ratio", type=float, default=0.35)
    parser.add_argument("--max-elongation", type=float, default=8.0)
    args = parser.parse_args(argv)
    args.seed_input = args.seed_input.expanduser()
    args.reference_input = args.reference_input.expanduser()
    args.out_root = args.out_root.expanduser()
    run_scan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

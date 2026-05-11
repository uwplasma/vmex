#!/usr/bin/env python
"""Audit solved seed equilibria for QI optimization suitability.

This tool is intentionally no-optimization: it reads existing ``input``/``wout``
pairs, evaluates QI diagnostics on the solved state, and writes JSON/CSV records
that are easy to compare before launching expensive QI seed sweeps.

Examples:

  python examples/optimization/audit_qi_seed_suitability.py --list-defaults
  python examples/optimization/audit_qi_seed_suitability.py --quick --csv results/qi_seed_audit.csv
  python examples/optimization/audit_qi_seed_suitability.py \
    --case qi_nfp2:qi:/path/to/input.nfp2_QI:/path/to/wout_nfp2_QI.nc
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.config import config_from_indata
from vmec_jax.energy import flux_profiles_from_indata
from vmec_jax.namelist import read_indata
from vmec_jax.qi_diagnostics import QIDiagnosticOptions, qi_diagnostics_from_state
from vmec_jax.static import build_static
from vmec_jax.wout import (
    equilibrium_aspect_ratio_from_state,
    equilibrium_iota_profiles_from_state,
    read_wout,
    state_from_wout,
)


enable_x64(True)

DATA_DIR = REPO_ROOT / "examples" / "data"
OMNIGENITY_ROOT = Path(os.environ.get("OMNIGENITY_OPTIMIZATION_ROOT", "~/local/omnigenity_optimization")).expanduser()
DEFAULT_OUTPUT = Path("results/qi_seed_audit/summary.json")

DEFAULT_TARGET_ASPECT = 5.0
DEFAULT_ABS_IOTA_MIN = 0.41
DEFAULT_MAX_MIRROR_RATIO = 0.21
DEFAULT_MAX_ELONGATION = 8.0
DEFAULT_SURFACES = (0.1, 0.35, 0.6, 0.85)


@dataclass(frozen=True)
class SeedCase:
    label: str
    family: str
    input_path: Path
    wout_path: Path


@dataclass(frozen=True)
class SuitabilityTargets:
    target_aspect: float = DEFAULT_TARGET_ASPECT
    abs_iota_min: float = DEFAULT_ABS_IOTA_MIN
    max_mirror_ratio: float = DEFAULT_MAX_MIRROR_RATIO
    max_elongation: float = DEFAULT_MAX_ELONGATION


def _local_default_cases() -> list[SeedCase]:
    return [
        SeedCase(
            "qi_nfp3_fixed_resolution",
            "qi",
            DATA_DIR / "input.nfp3_QI_fixed_resolution_final",
            DATA_DIR / "wout_nfp3_QI_fixed_resolution_final.nc",
        ),
        SeedCase(
            "qh_nfp4_warm_start",
            "qh",
            DATA_DIR / "input.nfp4_QH_warm_start",
            DATA_DIR / "wout_nfp4_QH_warm_start.nc",
        ),
        SeedCase(
            "qa_landreman_paul_lowres",
            "qa",
            DATA_DIR / "input.LandremanPaul2021_QA_lowres",
            DATA_DIR / "wout_LandremanPaul2021_QA_lowres.nc",
        ),
        SeedCase(
            "simple_circular_tokamak",
            "simple",
            DATA_DIR / "input.circular_tokamak",
            DATA_DIR / "wout_circular_tokamak.nc",
        ),
    ]


def _omnigenity_default_cases() -> list[SeedCase]:
    return [
        SeedCase(
            "qp_from_omnigenity_nfp2_qi",
            "qp",
            OMNIGENITY_ROOT / "inputs_QI" / "input.nfp2_QI_fixed_resolution_final",
            OMNIGENITY_ROOT / "wouts_QI" / "wout_nfp2_QI_fixed_resolution_final.nc",
        ),
        SeedCase(
            "qi_omnigenity_nfp1",
            "qi",
            OMNIGENITY_ROOT / "inputs_QI" / "input.nfp1_QI_fixed_resolution_final",
            OMNIGENITY_ROOT / "wouts_QI" / "wout_nfp1_QI_fixed_resolution_final.nc",
        ),
        SeedCase(
            "qi_omnigenity_nfp3",
            "qi",
            OMNIGENITY_ROOT / "inputs_QI" / "input.nfp3_QI_fixed_resolution_final",
            OMNIGENITY_ROOT / "wouts_QI" / "wout_nfp3_QI_fixed_resolution_final.nc",
        ),
    ]


def default_seed_cases() -> tuple[list[SeedCase], list[dict[str, str]]]:
    """Return existing default cases and a record of unavailable optional cases."""

    cases = _local_default_cases() + _omnigenity_default_cases()
    available: list[SeedCase] = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        missing = [
            str(path)
            for path in (case.input_path, case.wout_path)
            if not path.expanduser().exists()
        ]
        if missing:
            skipped.append({"label": case.label, "family": case.family, "missing": ";".join(missing)})
        else:
            available.append(case)
    return available, skipped


def parse_case(raw: str) -> SeedCase:
    parts = raw.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--case must have format label:family:input_path:wout_path"
        )
    label, family, input_path, wout_path = parts
    if not label:
        raise argparse.ArgumentTypeError("--case label must be non-empty")
    if family.lower() not in {"qi", "qp", "qh", "qa", "simple"}:
        raise argparse.ArgumentTypeError("family must be one of qi, qp, qh, qa, simple")
    return SeedCase(label, family.lower(), Path(input_path).expanduser(), Path(wout_path).expanduser())


def parse_surfaces(raw: str) -> tuple[float, ...]:
    surfaces = tuple(float(part) for part in raw.split(",") if part.strip())
    if not surfaces:
        raise argparse.ArgumentTypeError("--surfaces must contain at least one value")
    for surface in surfaces:
        if surface <= 0.0 or surface > 1.0:
            raise argparse.ArgumentTypeError("QI audit surfaces must be in (0, 1]")
    return surfaces


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return float(arr.ravel()[0])


def _mean_iota(iotas: Any) -> float | None:
    arr = np.asarray(iotas, dtype=float)
    if arr.size == 0:
        return None
    values = arr.ravel()[1:] if arr.size > 1 else arr.ravel()
    if values.size == 0:
        return None
    return float(np.mean(values))


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _constraint_status(record: dict[str, Any], targets: SuitabilityTargets) -> dict[str, Any]:
    aspect = _finite_or_none(record.get("aspect"))
    mean_iota = _finite_or_none(record.get("mean_iota"))
    mirror = _finite_or_none(record.get("qi_mirror_ratio_max"))
    elongation = _finite_or_none(record.get("qi_max_elongation"))
    smooth = _finite_or_none(record.get("qi_smooth_total"))
    legacy = _finite_or_none(record.get("qi_legacy_total"))

    aspect_relative_error = None if aspect is None else abs(aspect - targets.target_aspect) / targets.target_aspect
    iota_shortfall = (
        None
        if mean_iota is None
        else max(0.0, targets.abs_iota_min - abs(mean_iota))
    )
    mirror_excess = None if mirror is None else max(0.0, mirror - targets.max_mirror_ratio)
    elongation_excess = None if elongation is None else max(0.0, elongation - targets.max_elongation)
    diagnostic_errors = sorted(key for key in record if key.endswith("_error"))

    penalties = [
        0.0 if aspect_relative_error is None else aspect_relative_error,
        1.0 if iota_shortfall is None else iota_shortfall / targets.abs_iota_min,
        1.0 if mirror_excess is None else mirror_excess / targets.max_mirror_ratio,
        1.0 if elongation_excess is None else elongation_excess / targets.max_elongation,
    ]
    failed_constraints = []
    if aspect_relative_error is None or aspect_relative_error > 0.35:
        failed_constraints.append("aspect")
    if iota_shortfall is None or iota_shortfall > 0.0:
        failed_constraints.append("iota")
    if mirror_excess is None or mirror_excess > 0.0:
        failed_constraints.append("mirror")
    if elongation_excess is None or elongation_excess > 0.0:
        failed_constraints.append("elongation")
    if smooth is None:
        failed_constraints.append("smooth_qi")
    if legacy is None:
        failed_constraints.append("legacy_qi")
    failed_constraints.extend(diagnostic_errors)

    return {
        "aspect_relative_error": aspect_relative_error,
        "iota_shortfall": iota_shortfall,
        "mirror_excess": mirror_excess,
        "elongation_excess": elongation_excess,
        "failed_constraints": failed_constraints,
        "constraint_score": float(np.dot(penalties, penalties)),
        "seed_suitability": "pass" if not failed_constraints else "needs_attention",
    }


def evaluate_seed_case(
    case: SeedCase,
    *,
    surfaces: tuple[float, ...],
    targets: SuitabilityTargets,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    nphi_out: int,
    mboz: int,
    nboz: int,
    phimin: float,
    mirror_ntheta: int,
    mirror_nphi: int,
    elongation_ntheta: int,
    elongation_nphi: int,
    fail_on_error: bool,
) -> dict[str, Any]:
    input_path = case.input_path.expanduser().resolve()
    wout_path = case.wout_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not wout_path.exists():
        raise FileNotFoundError(wout_path)

    indata = read_indata(input_path)
    cfg = config_from_indata(indata)
    static = build_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)
    signgs = int(wout.signgs)
    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)

    options = QIDiagnosticOptions(
        surfaces=surfaces,
        mboz=mboz,
        nboz=nboz,
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        legacy_nphi_out=nphi_out,
        mirror_threshold=targets.max_mirror_ratio,
        mirror_ntheta=mirror_ntheta,
        mirror_nphi=mirror_nphi,
        mirror_surface_index=0,
        elongation_threshold=targets.max_elongation,
        elongation_ntheta=elongation_ntheta,
        elongation_nphi=elongation_nphi,
        phimin=phimin,
        fail_on_error=fail_on_error,
    )
    qi_record = qi_diagnostics_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        options=options,
        flux_local=flux,
    )

    _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    aspect = _first_float(equilibrium_aspect_ratio_from_state(state=state, static=static))

    record = {
        "label": case.label,
        "family": case.family,
        "input": str(input_path),
        "wout": str(wout_path),
        "nfp": int(wout.nfp),
        "mpol": int(wout.mpol),
        "ntor": int(wout.ntor),
        "ns": int(wout.ns),
        "aspect": aspect,
        "target_aspect": float(targets.target_aspect),
        "mean_iota": _mean_iota(iotas),
        "abs_iota_min": float(targets.abs_iota_min),
        **qi_record,
    }
    record.update(_constraint_status(record, targets))
    return record


def _qi_seed_score(record: dict[str, Any]) -> float:
    smooth = _finite_or_none(record.get("qi_smooth_total"))
    legacy = _finite_or_none(record.get("qi_legacy_total"))
    if smooth is None and legacy is None:
        return float("inf")
    return float((0.0 if smooth is None else smooth) + (0.0 if legacy is None else legacy))


def _sort_key(record: dict[str, Any]) -> tuple[float, float, float, str]:
    qi_score = _qi_seed_score(record)
    failed = len(record.get("failed_constraints", []))
    constraint_score = float(record.get("constraint_score", np.inf))
    return (qi_score, float(failed), constraint_score, str(record.get("label", "")))


def _with_ranks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        record["qi_seed_score"] = _qi_seed_score(record)
    ranked = sorted(records, key=_sort_key)
    for index, record in enumerate(ranked, start=1):
        record["suitability_rank"] = index

    for key, rank_key in (
        ("qi_smooth_total", "qi_smooth_rank"),
        ("qi_legacy_total", "qi_legacy_rank"),
    ):
        finite = [record for record in records if record.get(key) is not None]
        for index, record in enumerate(sorted(finite, key=lambda row: float(row[key])), start=1):
            record[rank_key] = index
    return ranked


def build_seed_audit(
    *,
    cases: list[SeedCase],
    skipped_defaults: list[dict[str, str]] | None,
    surfaces: tuple[float, ...],
    targets: SuitabilityTargets,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    nphi_out: int,
    mboz: int,
    nboz: int,
    phimin: float,
    mirror_ntheta: int,
    mirror_nphi: int,
    elongation_ntheta: int,
    elongation_nphi: int,
    fail_on_error: bool = False,
) -> dict[str, Any]:
    records = [
        evaluate_seed_case(
            case,
            surfaces=surfaces,
            targets=targets,
            nphi=nphi,
            nalpha=nalpha,
            n_bounce=n_bounce,
            nphi_out=nphi_out,
            mboz=mboz,
            nboz=nboz,
            phimin=phimin,
            mirror_ntheta=mirror_ntheta,
            mirror_nphi=mirror_nphi,
            elongation_ntheta=elongation_ntheta,
            elongation_nphi=elongation_nphi,
            fail_on_error=fail_on_error,
        )
        for case in cases
    ]
    records = _with_ranks(records)
    return {
        "mode": "qi_seed_suitability_audit",
        "no_optimization": True,
        "targets": asdict(targets),
        "resolution": {
            "surfaces": [float(surface) for surface in surfaces],
            "mboz": int(mboz),
            "nboz": int(nboz),
            "nphi": int(nphi),
            "nalpha": int(nalpha),
            "n_bounce": int(n_bounce),
            "nphi_out": int(nphi_out),
            "mirror_ntheta": int(mirror_ntheta),
            "mirror_nphi": int(mirror_nphi),
            "elongation_ntheta": int(elongation_ntheta),
            "elongation_nphi": int(elongation_nphi),
            "phimin": float(phimin),
        },
        "skipped_defaults": skipped_defaults or [],
        "cases": records,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")


def _write_csv(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "suitability_rank",
        "label",
        "family",
        "seed_suitability",
        "failed_constraints",
        "constraint_score",
        "qi_seed_score",
        "qi_smooth_rank",
        "qi_smooth_total",
        "qi_legacy_rank",
        "qi_legacy_total",
        "qi_mirror_ratio_max",
        "qi_mirror_excess_max",
        "qi_max_elongation",
        "qi_elongation_excess",
        "aspect",
        "aspect_relative_error",
        "mean_iota",
        "iota_shortfall",
        "nfp",
        "mpol",
        "ntor",
        "ns",
        "input",
        "wout",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["failed_constraints"] = ";".join(str(item) for item in record.get("failed_constraints", []))
            writer.writerow(row)


def _print_defaults(cases: list[SeedCase], skipped: list[dict[str, str]]) -> None:
    for case in cases:
        print(f"{case.label}:{case.family}:{case.input_path}:{case.wout_path}")
    if skipped:
        print("\nSkipped unavailable optional defaults:", file=sys.stderr)
        for row in skipped:
            print(f"  {row['label']} ({row['family']}): {row['missing']}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, help="label:family:input_path:wout_path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON output path")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV summary path")
    parser.add_argument("--list-defaults", action="store_true", help="Print default cases and exit")
    parser.add_argument("--quick", action="store_true", help="Use lower diagnostic resolution for smoke checks")
    parser.add_argument("--surfaces", type=parse_surfaces, default=DEFAULT_SURFACES)
    parser.add_argument("--mboz", type=int, default=18)
    parser.add_argument("--nboz", type=int, default=18)
    parser.add_argument("--nphi", type=int, default=151)
    parser.add_argument("--nalpha", type=int, default=31)
    parser.add_argument("--n-bounce", type=int, default=51)
    parser.add_argument("--nphi-out", type=int, default=401)
    parser.add_argument("--phimin", type=float, default=0.0)
    parser.add_argument("--mirror-ntheta", type=int, default=96)
    parser.add_argument("--mirror-nphi", type=int, default=96)
    parser.add_argument("--elongation-ntheta", type=int, default=48)
    parser.add_argument("--elongation-nphi", type=int, default=16)
    parser.add_argument("--target-aspect", type=float, default=DEFAULT_TARGET_ASPECT)
    parser.add_argument("--abs-iota-min", type=float, default=DEFAULT_ABS_IOTA_MIN)
    parser.add_argument("--max-mirror-ratio", type=float, default=DEFAULT_MAX_MIRROR_RATIO)
    parser.add_argument("--max-elongation", type=float, default=DEFAULT_MAX_ELONGATION)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    default_cases, skipped_defaults = default_seed_cases()
    if args.list_defaults:
        _print_defaults(default_cases, skipped_defaults)
        return 0

    cases = list(args.case) if args.case else default_cases
    if not cases:
        raise SystemExit("No seed cases available. Provide at least one --case.")

    nphi = 51 if args.quick else args.nphi
    nalpha = 11 if args.quick else args.nalpha
    n_bounce = 15 if args.quick else args.n_bounce
    nphi_out = 101 if args.quick else args.nphi_out
    mirror_ntheta = 32 if args.quick else args.mirror_ntheta
    mirror_nphi = 32 if args.quick else args.mirror_nphi
    elongation_ntheta = 24 if args.quick else args.elongation_ntheta
    elongation_nphi = 8 if args.quick else args.elongation_nphi

    targets = SuitabilityTargets(
        target_aspect=args.target_aspect,
        abs_iota_min=args.abs_iota_min,
        max_mirror_ratio=args.max_mirror_ratio,
        max_elongation=args.max_elongation,
    )
    report = build_seed_audit(
        cases=cases,
        skipped_defaults=[] if args.case else skipped_defaults,
        surfaces=args.surfaces,
        targets=targets,
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        nphi_out=nphi_out,
        mboz=args.mboz,
        nboz=args.nboz,
        phimin=args.phimin,
        mirror_ntheta=mirror_ntheta,
        mirror_nphi=mirror_nphi,
        elongation_ntheta=elongation_ntheta,
        elongation_nphi=elongation_nphi,
        fail_on_error=args.fail_on_error,
    )
    _write_json(report, args.output)
    if args.csv is not None:
        _write_csv(report["cases"], args.csv)

    print(f"Wrote {args.output} with {len(report['cases'])} seed records.")
    if args.csv is not None:
        print(f"Wrote {args.csv}.")
    best = report["cases"][0]
    print(
        "Best current seed: "
        f"{best['label']} ({best['family']}), suitability={best['seed_suitability']}, "
        f"smooth={best.get('qi_smooth_total')}, legacy={best.get('qi_legacy_total')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

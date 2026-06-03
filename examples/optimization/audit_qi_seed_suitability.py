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
import re
import shlex
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
DEFAULT_PREFINE_MANIFEST = Path("results/qi_seed_audit/prefine_manifest.json")
DEFAULT_PREFINE_OUTPUT_DIR = Path("results/qi_seed_audit/prefine_probes")

DEFAULT_TARGET_ASPECT = 10.0
DEFAULT_ABS_IOTA_MIN = 0.41
DEFAULT_MAX_MIRROR_RATIO = 0.21
DEFAULT_MAX_ELONGATION = 8.0
DEFAULT_SURFACES = (0.1, 0.35, 0.6, 0.85)
DEFAULT_PREFINE_SURFACES = (0.35, 0.65)
SEED_FAMILY_ORDER = ("qi", "qp", "qh", "qa", "simple")
PHIMIN_POLICIES = ("fixed", "well-phase")

MAX_PREFINE_TOP_N = 5
MAX_PREFINE_FAMILY_REPRESENTATIVES = len(SEED_FAMILY_ORDER)
MAX_PREFINE_MAX_NFEV = 5
MAX_PREFINE_CONTINUATION_NFEV = 3
MAX_PREFINE_STAGE_COUNT = 5
MAX_PREFINE_MODE = 3
MAX_PREFINE_VMEC_MODE = 5
MAX_PREFINE_INNER_ITER = 40
MAX_PREFINE_QI_NPHI = 51
MAX_PREFINE_QI_NALPHA = 11
MAX_PREFINE_QI_N_BOUNCE = 15
MAX_PREFINE_BOOZ_MODE = 10
DEFAULT_PREFINE_STAGE_MODES = (1, 1, 2, 2, 3)
OBJECTIVE_REGRESSION_RTOL = 1.0e-12
OBJECTIVE_REGRESSION_ATOL = 1.0e-14
PREFINE_STABLE_LOW_OBJECTIVE_THRESHOLD = 5.0e-2


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
    smooth_qi_max: float | None = 2.0e-3
    legacy_qi_max: float | None = 2.0e-3
    max_mirror_ratio: float = DEFAULT_MAX_MIRROR_RATIO
    max_elongation: float = DEFAULT_MAX_ELONGATION


@dataclass(frozen=True)
class QIPrefineProbeConfig:
    """Hard-capped settings for tiny optional QI-only prefine probes."""

    top_n: int = 1
    include_family_representatives: bool = True
    representative_families: tuple[str, ...] = SEED_FAMILY_ORDER
    max_nfev: int = 2
    continuation_nfev: int = 1
    max_mode: int = 3
    min_vmec_mode: int = 3
    stage_modes: tuple[int, ...] = DEFAULT_PREFINE_STAGE_MODES
    output_dir: Path = DEFAULT_PREFINE_OUTPUT_DIR
    surfaces: tuple[float, ...] = DEFAULT_PREFINE_SURFACES
    mboz: int = 8
    nboz: int = 8
    nphi: int = 31
    nalpha: int = 7
    n_bounce: int = 9
    include_bounce_endpoints: bool = True
    phimin: float = 0.0
    qi_weight: float = 1.0
    qi_ceiling_weight: float = 100.0
    qi_ceiling_max: float = 2.0e-3
    qi_ceiling_smooth_penalty: float = 2.0e-3
    mirror_weight: float = 0.0
    mirror_threshold: float = DEFAULT_MAX_MIRROR_RATIO
    mirror_ntheta: int = 32
    mirror_nphi: int = 32
    mirror_surface_index: int | None = None
    elongation_weight: float = 0.0
    elongation_threshold: float = DEFAULT_MAX_ELONGATION
    elongation_ntheta: int = 24
    elongation_nphi: int = 8
    method: str = "scipy"
    ftol: float = 1.0e-3
    gtol: float = 1.0e-3
    xtol: float = 1.0e-3
    use_ess: bool = True
    ess_alpha: float = 1.2
    inner_max_iter: int = 20
    trial_max_iter: int = 20
    inner_ftol: float = 1.0e-7
    trial_ftol: float = 1.0e-7
    scipy_tr_solver: str = "lsmr"
    scipy_lsmr_maxiter: int | None = 5


def _local_default_cases() -> list[SeedCase]:
    return [
        SeedCase(
            "qi_nfp3_fixed_resolution",
            "qi",
            DATA_DIR / "input.nfp3_QI_fixed_resolution_final",
            DATA_DIR / "wout_nfp3_QI_fixed_resolution_final.nc",
        ),
        SeedCase(
            "qi_stel_seed_3127",
            "qi",
            DATA_DIR / "input.QI_stel_seed_3127",
            DATA_DIR / "wout_QI_stel_seed_3127.nc",
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
        missing = [str(path) for path in (case.input_path, case.wout_path) if not path.expanduser().exists()]
        if missing:
            skipped.append({"label": case.label, "family": case.family, "missing": ";".join(missing)})
        else:
            available.append(case)
    return available, skipped


def parse_case(raw: str) -> SeedCase:
    parts = raw.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--case must have format label:family:input_path:wout_path")
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


def parse_stage_modes(raw: str) -> tuple[int, ...]:
    modes = tuple(int(part) for part in raw.split(",") if part.strip())
    if not modes:
        raise argparse.ArgumentTypeError("stage modes must contain at least one value")
    if any(mode <= 0 for mode in modes):
        raise argparse.ArgumentTypeError("stage modes must be positive integers")
    return modes


def parse_seed_families(raw: str) -> tuple[str, ...]:
    families = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not families:
        raise argparse.ArgumentTypeError("seed families must contain at least one value")
    unknown = sorted(set(families) - set(SEED_FAMILY_ORDER))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"seed families must be drawn from {', '.join(SEED_FAMILY_ORDER)}; got {', '.join(unknown)}"
        )
    return families


def parse_optional_surface_index(raw: str) -> int | None:
    value = raw.strip().lower()
    if value in {"all", "none", "*"}:
        return None
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("surface index must be a non-negative integer or 'all'") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("surface index must be a non-negative integer or 'all'")
    return index


def _surface_index_label(surface_index: int | None) -> str:
    return "all" if surface_index is None else str(int(surface_index))


def _unique_float_sequence(values: tuple[float, ...], *, atol: float = 1.0e-14) -> tuple[float, ...]:
    unique: list[float] = []
    for value in values:
        value_f = float(value)
        if not any(abs(value_f - existing) <= atol for existing in unique):
            unique.append(value_f)
    return tuple(unique)


def _phimin_candidates_for_case(case: SeedCase, *, phimin: float, phimin_policy: str) -> tuple[float, ...]:
    """Return QI well-phase candidates for a seed case.

    The Goodman/omnigenity QI diagnostic uses either ``0`` or ``pi/nfp`` as the
    first well maximum depending on the seed.  Auditing both phases catches
    seeds that otherwise look artificially poor solely because the interval is
    shifted by half a well.
    """

    if phimin_policy not in PHIMIN_POLICIES:
        raise ValueError(f"phimin_policy must be one of {PHIMIN_POLICIES}, got {phimin_policy!r}")
    if phimin_policy == "fixed":
        return (float(phimin),)

    indata = read_indata(case.input_path.expanduser().resolve())
    cfg = config_from_indata(indata)
    return _unique_float_sequence((0.0, float(np.pi) / float(cfg.nfp)))


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
    iota_shortfall = None if mean_iota is None else max(0.0, targets.abs_iota_min - abs(mean_iota))
    mirror_excess = None if mirror is None else max(0.0, mirror - targets.max_mirror_ratio)
    elongation_excess = None if elongation is None else max(0.0, elongation - targets.max_elongation)
    smooth_excess = (
        None
        if smooth is None or targets.smooth_qi_max is None
        else max(0.0, smooth - targets.smooth_qi_max)
    )
    legacy_excess = (
        None
        if legacy is None or targets.legacy_qi_max is None
        else max(0.0, legacy - targets.legacy_qi_max)
    )
    diagnostic_errors = sorted(key for key in record if key.endswith("_error"))

    penalties = [
        0.0 if aspect_relative_error is None else aspect_relative_error,
        1.0 if iota_shortfall is None else iota_shortfall / targets.abs_iota_min,
        1.0 if mirror_excess is None else mirror_excess / targets.max_mirror_ratio,
        1.0 if elongation_excess is None else elongation_excess / targets.max_elongation,
        1.0
        if smooth_excess is None
        else smooth_excess / max(float(targets.smooth_qi_max), 1.0e-16),
        1.0
        if legacy_excess is None
        else legacy_excess / max(float(targets.legacy_qi_max), 1.0e-16),
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
    if smooth is None or (smooth_excess is not None and smooth_excess > 0.0):
        failed_constraints.append("smooth_qi")
    if legacy is None or (legacy_excess is not None and legacy_excess > 0.0):
        failed_constraints.append("legacy_qi")
    failed_constraints.extend(diagnostic_errors)

    return {
        "aspect_relative_error": aspect_relative_error,
        "iota_shortfall": iota_shortfall,
        "mirror_excess": mirror_excess,
        "elongation_excess": elongation_excess,
        "smooth_qi_excess": smooth_excess,
        "legacy_qi_excess": legacy_excess,
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
    include_bounce_endpoints: bool,
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
        include_bounce_endpoints=bool(include_bounce_endpoints),
        legacy_nphi_out=nphi_out,
        mirror_threshold=targets.max_mirror_ratio,
        mirror_ntheta=mirror_ntheta,
        mirror_nphi=mirror_nphi,
        mirror_surface_index=None,
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


def _compact_phimin_candidate_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "qi_phimin",
        "qi_seed_score",
        "qi_smooth_total",
        "qi_legacy_total",
        "qi_mirror_ratio_max",
        "qi_max_elongation",
        "aspect",
        "mean_iota",
        "constraint_score",
        "failed_constraints",
    )
    return {key: record.get(key) for key in keys if key in record}


def _select_best_phimin_record(
    case: SeedCase,
    *,
    phimin_candidates: tuple[float, ...],
    surfaces: tuple[float, ...],
    targets: SuitabilityTargets,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    nphi_out: int,
    mboz: int,
    nboz: int,
    include_bounce_endpoints: bool,
    mirror_ntheta: int,
    mirror_nphi: int,
    elongation_ntheta: int,
    elongation_nphi: int,
    fail_on_error: bool,
) -> dict[str, Any]:
    candidates = [
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
            phimin=phimin_candidate,
            include_bounce_endpoints=include_bounce_endpoints,
            mirror_ntheta=mirror_ntheta,
            mirror_nphi=mirror_nphi,
            elongation_ntheta=elongation_ntheta,
            elongation_nphi=elongation_nphi,
            fail_on_error=fail_on_error,
        )
        for phimin_candidate in phimin_candidates
    ]
    for record in candidates:
        record["qi_seed_score"] = _qi_seed_score(record)
    selected = dict(min(candidates, key=_sort_key))
    selected["selected_phimin"] = float(selected.get("qi_phimin", phimin_candidates[0]))
    selected["phimin_candidates"] = [float(value) for value in phimin_candidates]
    selected["phimin_candidate_metrics"] = [_compact_phimin_candidate_record(record) for record in candidates]
    return selected


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


def _safe_label(label: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_")
    return safe or "seed"


def _validate_prefine_probe_config(config: QIPrefineProbeConfig) -> None:
    if not config.stage_modes:
        raise ValueError("prefine stage_modes must contain at least one mode")
    if not config.surfaces:
        raise ValueError("prefine surfaces must contain at least one value")
    if not config.representative_families:
        raise ValueError("prefine representative_families must contain at least one family")
    if any(surface <= 0.0 or surface > 1.0 for surface in config.surfaces):
        raise ValueError("prefine surfaces must be in (0, 1]")
    if len(set(config.representative_families)) != len(config.representative_families):
        raise ValueError("prefine representative_families must not contain duplicates")
    unknown_families = sorted(set(config.representative_families) - set(SEED_FAMILY_ORDER))
    if unknown_families:
        raise ValueError(f"prefine representative_families contains unsupported families: {unknown_families}")
    if any(mode <= 0 for mode in config.stage_modes):
        raise ValueError("prefine stage_modes must be positive integers")
    if any(next_mode < mode for mode, next_mode in zip(config.stage_modes, config.stage_modes[1:], strict=False)):
        raise ValueError("prefine stage_modes must be nondecreasing")
    checks = [
        (1 <= config.top_n <= MAX_PREFINE_TOP_N, f"prefine top_n must be in [1, {MAX_PREFINE_TOP_N}]"),
        (
            len(config.representative_families) <= MAX_PREFINE_FAMILY_REPRESENTATIVES,
            f"prefine representative_families must contain at most {MAX_PREFINE_FAMILY_REPRESENTATIVES} families",
        ),
        (
            1 <= config.max_nfev <= MAX_PREFINE_MAX_NFEV,
            f"prefine max_nfev must be in [1, {MAX_PREFINE_MAX_NFEV}]",
        ),
        (
            1 <= config.continuation_nfev <= MAX_PREFINE_CONTINUATION_NFEV,
            f"prefine continuation_nfev must be in [1, {MAX_PREFINE_CONTINUATION_NFEV}]",
        ),
        (1 <= config.max_mode <= MAX_PREFINE_MODE, f"prefine max_mode must be in [1, {MAX_PREFINE_MODE}]"),
        (
            1 <= config.min_vmec_mode <= MAX_PREFINE_VMEC_MODE,
            f"prefine min_vmec_mode must be in [1, {MAX_PREFINE_VMEC_MODE}]",
        ),
        (
            1 <= len(config.stage_modes) <= MAX_PREFINE_STAGE_COUNT,
            f"prefine stage_modes must contain 1-{MAX_PREFINE_STAGE_COUNT} modes",
        ),
        (
            max(config.stage_modes) <= config.max_mode,
            "prefine stage_modes must not exceed prefine max_mode",
        ),
        (
            max(config.stage_modes) == config.max_mode,
            "prefine stage_modes must include prefine max_mode",
        ),
        (
            int(config.stage_modes[-1]) == int(config.max_mode),
            "prefine stage_modes must end at prefine max_mode",
        ),
        (
            1 <= config.inner_max_iter <= MAX_PREFINE_INNER_ITER,
            f"prefine inner_max_iter must be in [1, {MAX_PREFINE_INNER_ITER}]",
        ),
        (
            1 <= config.trial_max_iter <= MAX_PREFINE_INNER_ITER,
            f"prefine trial_max_iter must be in [1, {MAX_PREFINE_INNER_ITER}]",
        ),
        (1 <= config.nphi <= MAX_PREFINE_QI_NPHI, f"prefine nphi must be in [1, {MAX_PREFINE_QI_NPHI}]"),
        (
            1 <= config.nalpha <= MAX_PREFINE_QI_NALPHA,
            f"prefine nalpha must be in [1, {MAX_PREFINE_QI_NALPHA}]",
        ),
        (
            1 <= config.n_bounce <= MAX_PREFINE_QI_N_BOUNCE,
            f"prefine n_bounce must be in [1, {MAX_PREFINE_QI_N_BOUNCE}]",
        ),
        (1 <= config.mboz <= MAX_PREFINE_BOOZ_MODE, f"prefine mboz must be in [1, {MAX_PREFINE_BOOZ_MODE}]"),
        (1 <= config.nboz <= MAX_PREFINE_BOOZ_MODE, f"prefine nboz must be in [1, {MAX_PREFINE_BOOZ_MODE}]"),
    ]
    if config.mirror_weight < 0.0:
        raise ValueError("prefine mirror_weight must be non-negative")
    if config.elongation_weight < 0.0:
        raise ValueError("prefine elongation_weight must be non-negative")
    if config.mirror_threshold <= 0.0:
        raise ValueError("prefine mirror_threshold must be positive")
    if config.elongation_threshold <= 0.0:
        raise ValueError("prefine elongation_threshold must be positive")
    if config.mirror_ntheta <= 0 or config.mirror_nphi <= 0:
        raise ValueError("prefine mirror grid sizes must be positive")
    if config.elongation_ntheta <= 0 or config.elongation_nphi <= 0:
        raise ValueError("prefine elongation grid sizes must be positive")
    if config.mirror_surface_index is not None and config.mirror_surface_index < 0:
        raise ValueError("prefine mirror_surface_index must be non-negative")
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def _prefine_probe_config_dict(config: QIPrefineProbeConfig) -> dict[str, Any]:
    row = asdict(config)
    row["output_dir"] = str(config.output_dir)
    row["surfaces"] = [float(surface) for surface in config.surfaces]
    row["stage_modes"] = [int(mode) for mode in config.stage_modes]
    row["representative_families"] = [str(family) for family in config.representative_families]
    row["endpoint_mode"] = _prefine_endpoint_mode(config.include_bounce_endpoints)
    row["total_nfev_cap"] = _prefine_total_nfev_cap(config)
    return row


def _prefine_endpoint_mode(include_bounce_endpoints: bool) -> str:
    return "include_bounce_endpoints" if bool(include_bounce_endpoints) else "interior_only"


def _prefine_stage_budget(config: QIPrefineProbeConfig, *, stage_mode: int) -> int:
    if int(stage_mode) == int(config.max_mode):
        return int(config.max_nfev)
    return int(config.continuation_nfev)


def _prefine_total_nfev_cap(config: QIPrefineProbeConfig) -> int:
    return sum(_prefine_stage_budget(config, stage_mode=int(mode)) for mode in config.stage_modes)


def _prefine_stage_plan(config: QIPrefineProbeConfig, *, probe_dir: Path | None = None) -> list[dict[str, Any]]:
    repeat_counts: dict[int, int] = {}
    stages: list[dict[str, Any]] = []
    for index, raw_mode in enumerate(config.stage_modes, start=1):
        mode = int(raw_mode)
        repeat_counts[mode] = repeat_counts.get(mode, 0) + 1
        stage = {
            "index": index,
            "mode": mode,
            "repeat_index_for_mode": repeat_counts[mode],
            "is_final_mode": mode == int(config.max_mode),
            "nfev_cap": _prefine_stage_budget(config, stage_mode=mode),
        }
        if probe_dir is not None:
            stage["output_dir"] = str(probe_dir / f"stage_{index:02d}_mode{mode:02d}")
        stages.append(stage)
    return stages


def _prefine_effective_caps(config: QIPrefineProbeConfig) -> dict[str, Any]:
    return {
        "per_probe_stage_count": len(config.stage_modes),
        "per_probe_total_nfev": _prefine_total_nfev_cap(config),
        "per_final_stage_nfev": int(config.max_nfev),
        "per_continuation_stage_nfev": int(config.continuation_nfev),
        "max_mode": int(config.max_mode),
        "min_vmec_mode": int(config.min_vmec_mode),
        "inner_max_iter": int(config.inner_max_iter),
        "trial_max_iter": int(config.trial_max_iter),
        "mboz": int(config.mboz),
        "nboz": int(config.nboz),
        "nphi": int(config.nphi),
        "nalpha": int(config.nalpha),
        "n_bounce": int(config.n_bounce),
    }


def _prefine_phimin_for_record(record: dict[str, Any], config: QIPrefineProbeConfig) -> tuple[float, str]:
    selected_phimin = record.get("selected_phimin")
    if selected_phimin is None:
        return float(config.phimin), "prefine_config_phimin"
    return float(selected_phimin), "audit_selected_phimin"


def _prefine_endpoint_alignment(audit_report: dict[str, Any], config: QIPrefineProbeConfig) -> dict[str, Any]:
    audit_value = audit_report.get("resolution", {}).get("include_bounce_endpoints")
    prefine_value = bool(config.include_bounce_endpoints)
    aligned = None if audit_value is None else bool(audit_value) == prefine_value
    return {
        "audit_include_bounce_endpoints": None if audit_value is None else bool(audit_value),
        "prefine_include_bounce_endpoints": prefine_value,
        "aligned": aligned,
        "endpoint_mode": _prefine_endpoint_mode(prefine_value),
    }


def _prefine_run_command(record: dict[str, Any], config: QIPrefineProbeConfig, manifest_path: Path) -> str:
    case = f"{record['label']}:{record['family']}:{record['input']}:{record['wout']}"
    selected_phimin, _phimin_source = _prefine_phimin_for_record(record, config)
    command = [
        sys.executable,
        str(Path(__file__).relative_to(REPO_ROOT)),
        "--case",
        case,
        "--quick",
        "--phimin-policy",
        "fixed",
        "--phimin",
        str(selected_phimin),
        "--include-bounce-endpoints" if bool(config.include_bounce_endpoints) else "--no-include-bounce-endpoints",
        "--prefine-probes",
        "run",
        "--prefine-reviewed",
        "--prefine-top-n",
        "1",
        "--prefine-manifest",
        str(manifest_path),
        "--prefine-output-dir",
        str(config.output_dir),
        "--prefine-max-nfev",
        str(config.max_nfev),
        "--prefine-continuation-nfev",
        str(config.continuation_nfev),
        "--prefine-max-mode",
        str(config.max_mode),
        "--prefine-min-vmec-mode",
        str(config.min_vmec_mode),
        "--prefine-stage-modes",
        ",".join(str(mode) for mode in config.stage_modes),
        "--prefine-surfaces",
        ",".join(str(surface) for surface in config.surfaces),
        "--prefine-mboz",
        str(config.mboz),
        "--prefine-nboz",
        str(config.nboz),
        "--prefine-nphi",
        str(config.nphi),
        "--prefine-nalpha",
        str(config.nalpha),
        "--prefine-n-bounce",
        str(config.n_bounce),
        "--prefine-phimin",
        str(selected_phimin),
        "--prefine-mirror-weight",
        str(config.mirror_weight),
        "--prefine-mirror-threshold",
        str(config.mirror_threshold),
        "--prefine-mirror-ntheta",
        str(config.mirror_ntheta),
        "--prefine-mirror-nphi",
        str(config.mirror_nphi),
        "--prefine-mirror-surface-index",
        _surface_index_label(config.mirror_surface_index),
        "--prefine-elongation-weight",
        str(config.elongation_weight),
        "--prefine-elongation-threshold",
        str(config.elongation_threshold),
        "--prefine-elongation-ntheta",
        str(config.elongation_ntheta),
        "--prefine-elongation-nphi",
        str(config.elongation_nphi),
        "--prefine-inner-max-iter",
        str(config.inner_max_iter),
        "--prefine-trial-max-iter",
        str(config.trial_max_iter),
        "--prefine-ess-alpha",
        str(config.ess_alpha),
    ]
    command.append("--prefine-use-ess" if bool(config.use_ess) else "--no-prefine-use-ess")
    if config.scipy_lsmr_maxiter is not None:
        command.extend(["--prefine-scipy-lsmr-maxiter", str(config.scipy_lsmr_maxiter)])
    command.append(
        "--prefine-include-bounce-endpoints"
        if bool(config.include_bounce_endpoints)
        else "--no-prefine-include-bounce-endpoints"
    )
    return " ".join(shlex.quote(part) for part in command)


def _prefine_selection_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("label", "")),
        str(record.get("family", "")),
        str(record.get("input", "")),
        str(record.get("wout", "")),
    )


def _select_prefine_probe_records(
    audit_report: dict[str, Any],
    config: QIPrefineProbeConfig,
) -> list[dict[str, Any]]:
    """Select top-ranked seeds plus one best-ranked representative per family."""

    cases = list(audit_report.get("cases", []))
    selected: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add_record(record: dict[str, Any], *, reason: str, representative_family: str | None = None) -> None:
        key = _prefine_selection_key(record)
        selection = by_key.get(key)
        if selection is None:
            selection = {
                "record": record,
                "selection_reasons": [],
                "representative_families": [],
            }
            by_key[key] = selection
            selected.append(selection)
        selection["selection_reasons"].append(reason)
        if representative_family is not None:
            selection["representative_families"].append(representative_family)

    for record in cases[: config.top_n]:
        add_record(record, reason="top_n")

    if config.include_family_representatives:
        for family in config.representative_families:
            representative = next(
                (record for record in cases if str(record.get("family", "")).lower() == family),
                None,
            )
            if representative is not None:
                add_record(
                    representative,
                    reason="family_representative",
                    representative_family=family,
                )

    return selected


def build_qi_prefine_probe_manifest(
    audit_report: dict[str, Any],
    *,
    config: QIPrefineProbeConfig,
    manifest_path: Path,
    dry_run: bool,
    reviewed: bool = False,
) -> dict[str, Any]:
    """Build a bounded manifest from top-ranked and family-representative audit rows."""

    _validate_prefine_probe_config(config)
    selected = _select_prefine_probe_records(audit_report, config)
    effective_caps = _prefine_effective_caps(config)
    endpoint_mode = _prefine_endpoint_mode(config.include_bounce_endpoints)
    endpoint_alignment = _prefine_endpoint_alignment(audit_report, config)
    plans = []
    for index, selection in enumerate(selected, start=1):
        record = selection["record"]
        label = str(record.get("label", f"seed_{index}"))
        probe_dir = config.output_dir / f"{index:02d}_{_safe_label(label)}"
        selected_phimin, phimin_source = _prefine_phimin_for_record(record, config)
        stage_plan = _prefine_stage_plan(config, probe_dir=probe_dir)
        plan = {
            "status": "planned" if dry_run else "pending",
            "review": {
                "required_before_run": True,
                "operator_confirmed": bool(reviewed),
                "status": "reviewed" if bool(reviewed) else "requires_review",
            },
            "audit_rank": record.get("suitability_rank", index),
            "selection_reasons": selection["selection_reasons"],
            "representative_families": selection["representative_families"],
            "label": label,
            "family": record.get("family"),
            "input": record.get("input"),
            "wout": record.get("wout"),
            "output_dir": str(probe_dir),
            "phimin": {
                "value": selected_phimin,
                "source": phimin_source,
                "audit_policy": record.get("phimin_policy"),
                "audit_candidates": [float(value) for value in record.get("phimin_candidates") or []],
            },
            "endpoint_mode": endpoint_mode,
            "endpoint_alignment": endpoint_alignment,
            "stages": stage_plan,
            "caps": effective_caps,
            "audit_metrics": {
                "qi_seed_score": record.get("qi_seed_score"),
                "qi_smooth_total": record.get("qi_smooth_total"),
                "qi_legacy_total": record.get("qi_legacy_total"),
                "qi_mirror_ratio_max": record.get("qi_mirror_ratio_max"),
                "qi_mirror_ratio_target": record.get("qi_mirror_ratio_target", config.mirror_threshold),
                "qi_mirror_excess_max": record.get("qi_mirror_excess_max"),
                "qi_max_elongation": record.get("qi_max_elongation"),
                "qi_elongation_target": record.get("qi_elongation_target", config.elongation_threshold),
                "qi_elongation_excess": record.get("qi_elongation_excess"),
                "aspect": record.get("aspect"),
                "target_aspect": record.get("target_aspect"),
                "mean_iota": record.get("mean_iota"),
                "abs_iota_min": record.get("abs_iota_min"),
                "selected_phimin": record.get("selected_phimin"),
                "constraint_score": record.get("constraint_score"),
                "failed_constraints": record.get("failed_constraints", []),
            },
            "optimization": {
                "objective": (
                    "qi_constrained_prefine_probe"
                    if float(config.mirror_weight) > 0.0 or float(config.elongation_weight) > 0.0
                    else "qi_only_prefine_probe"
                ),
                "max_nfev": int(config.max_nfev),
                "continuation_nfev": int(config.continuation_nfev),
                "max_mode": int(config.max_mode),
                "min_vmec_mode": int(config.min_vmec_mode),
                "stage_modes": [int(mode) for mode in config.stage_modes],
                "stage_count": len(config.stage_modes),
                "stage_plan": stage_plan,
                "total_nfev_cap": _prefine_total_nfev_cap(config),
                "method": config.method,
                "ftol": float(config.ftol),
                "gtol": float(config.gtol),
                "xtol": float(config.xtol),
                "use_ess": bool(config.use_ess),
                "ess_alpha": float(config.ess_alpha),
                "inner_max_iter": int(config.inner_max_iter),
                "trial_max_iter": int(config.trial_max_iter),
                "inner_ftol": float(config.inner_ftol),
                "trial_ftol": float(config.trial_ftol),
                "scipy_tr_solver": config.scipy_tr_solver,
                "scipy_lsmr_maxiter": config.scipy_lsmr_maxiter,
            },
            "qi_options": {
                "surfaces": [float(surface) for surface in config.surfaces],
                "mboz": int(config.mboz),
                "nboz": int(config.nboz),
                "nphi": int(config.nphi),
                "nalpha": int(config.nalpha),
                "n_bounce": int(config.n_bounce),
                "include_bounce_endpoints": bool(config.include_bounce_endpoints),
                "endpoint_mode": endpoint_mode,
                "phimin": selected_phimin,
                "phimin_source": phimin_source,
                "weight": float(config.qi_weight),
                "qi_ceiling_weight": float(config.qi_ceiling_weight),
                "qi_ceiling_max": float(config.qi_ceiling_max),
                "qi_ceiling_smooth_penalty": float(config.qi_ceiling_smooth_penalty),
                "mirror_weight": float(config.mirror_weight),
                "mirror_threshold": float(config.mirror_threshold),
                "mirror_ntheta": int(config.mirror_ntheta),
                "mirror_nphi": int(config.mirror_nphi),
                "mirror_surface_index": config.mirror_surface_index,
                "mirror_surface_mode": _surface_index_label(config.mirror_surface_index),
                "elongation_weight": float(config.elongation_weight),
                "elongation_threshold": float(config.elongation_threshold),
                "elongation_ntheta": int(config.elongation_ntheta),
                "elongation_nphi": int(config.elongation_nphi),
            },
            "would_write": [
                str(probe_dir / "input.initial"),
                str(probe_dir / "input.final"),
                str(probe_dir / "wout_initial.nc"),
                str(probe_dir / "wout_final.nc"),
                str(probe_dir / "history.json"),
            ],
            "would_write_stage_dirs": [str(probe_dir / f"stage_{stage['index']:02d}_mode{stage['mode']:02d}") for stage in stage_plan],
            "run_command": _prefine_run_command(record, config, manifest_path),
        }
        plans.append(plan)

    manifest = {
        "mode": "qi_prefine_probe_manifest",
        "dry_run": bool(dry_run),
        "review": {
            "required_before_run": True,
            "operator_confirmed": bool(reviewed),
            "status": "reviewed" if bool(reviewed) else "requires_review",
        },
        "hard_caps": {
            "top_n": MAX_PREFINE_TOP_N,
            "family_representatives": MAX_PREFINE_FAMILY_REPRESENTATIVES,
            "max_nfev": MAX_PREFINE_MAX_NFEV,
            "continuation_nfev": MAX_PREFINE_CONTINUATION_NFEV,
            "stage_count": MAX_PREFINE_STAGE_COUNT,
            "total_nfev_per_probe": MAX_PREFINE_STAGE_COUNT * MAX_PREFINE_MAX_NFEV,
            "max_mode": MAX_PREFINE_MODE,
            "min_vmec_mode": MAX_PREFINE_VMEC_MODE,
            "inner_iter": MAX_PREFINE_INNER_ITER,
            "nphi": MAX_PREFINE_QI_NPHI,
            "nalpha": MAX_PREFINE_QI_NALPHA,
            "n_bounce": MAX_PREFINE_QI_N_BOUNCE,
            "booz_mode": MAX_PREFINE_BOOZ_MODE,
        },
        "effective_caps": effective_caps,
        "endpoint_alignment": endpoint_alignment,
        "config": _prefine_probe_config_dict(config),
        "selection": {
            "requested_top_n": int(config.top_n),
            "include_family_representatives": bool(config.include_family_representatives),
            "requested_representative_families": [str(family) for family in config.representative_families],
            "available_rows": len(audit_report.get("cases", [])),
            "top_rows": min(int(config.top_n), len(audit_report.get("cases", []))),
            "family_representative_rows": sum(
                1 for plan in plans if "family_representative" in plan.get("selection_reasons", [])
            ),
            "covered_families": sorted(
                {str(family) for plan in plans for family in plan.get("representative_families", [])}
            ),
            "planned_rows": len(plans),
        },
        "plans": plans,
    }
    manifest["summary"] = summarize_qi_prefine_probe_manifest(manifest)
    return manifest


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(np.asarray(value))
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _prefine_metric_delta(initial: float | None, final: float | None) -> float | None:
    if initial is None or final is None:
        return None
    return float(final - initial)


def _prefine_metric_worsened(initial: float | None, final: float | None) -> bool | None:
    delta = _prefine_metric_delta(initial, final)
    if delta is None:
        return None
    return bool(delta > _prefine_objective_regression_tolerance(initial, final))


def _prefine_first_finite(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in record:
            value = _finite_float(record.get(key))
            if value is not None:
                return value
    return None


def _prefine_excess(
    record: dict[str, Any],
    *,
    value_keys: tuple[str, ...],
    target_keys: tuple[str, ...],
    explicit_keys: tuple[str, ...],
    target_default: Any = None,
) -> float | None:
    explicit = _prefine_first_finite(record, *explicit_keys)
    if explicit is not None:
        return max(0.0, explicit)
    value = _prefine_first_finite(record, *value_keys)
    target = _prefine_first_finite(record, *target_keys)
    if target is None:
        target = _finite_float(target_default)
    if value is None or target is None:
        return None
    return max(0.0, value - target)


def _prefine_snapshot_from_record(record: dict[str, Any] | None, qi_options_raw: dict[str, Any]) -> dict[str, Any]:
    record = {} if record is None else dict(record)
    mirror_threshold = _prefine_first_finite(
        record,
        "mirror_threshold",
        "qi_mirror_ratio_target",
        "mirror_ratio_target",
    )
    if mirror_threshold is None:
        mirror_threshold = _finite_float(qi_options_raw.get("mirror_threshold"))
    elongation_threshold = _prefine_first_finite(
        record,
        "elongation_threshold",
        "qi_elongation_target",
        "elongation_target",
    )
    if elongation_threshold is None:
        elongation_threshold = _finite_float(qi_options_raw.get("elongation_threshold"))
    mirror_excess = _prefine_excess(
        record,
        value_keys=("mirror_ratio", "qi_mirror_ratio_max", "qi_mirror_ratio"),
        target_keys=("mirror_threshold", "qi_mirror_ratio_target", "mirror_ratio_target"),
        explicit_keys=("mirror_excess", "qi_mirror_excess_max"),
        target_default=mirror_threshold,
    )
    elongation_excess = _prefine_excess(
        record,
        value_keys=("elongation", "qi_max_elongation", "qi_elongation"),
        target_keys=("elongation_threshold", "qi_elongation_target", "elongation_target"),
        explicit_keys=("elongation_excess", "qi_elongation_excess"),
        target_default=elongation_threshold,
    )
    mirror_weight = _finite_float(qi_options_raw.get("mirror_weight")) or 0.0
    elongation_weight = _finite_float(qi_options_raw.get("elongation_weight")) or 0.0
    snapshot = {
        "qi_residual": _prefine_first_finite(record, "qi_residual", "qi_smooth_total", "smooth_qi"),
        "qi_legacy_total": _prefine_first_finite(record, "qi_legacy_total", "legacy_qi"),
        "mirror_ratio": _prefine_first_finite(record, "mirror_ratio", "qi_mirror_ratio_max", "qi_mirror_ratio"),
        "mirror_threshold": mirror_threshold,
        "mirror_excess": mirror_excess,
        "mirror_penalty": None if mirror_excess is None else float(mirror_excess**2),
        "mirror_weighted_penalty": None if mirror_excess is None else float(mirror_weight * mirror_excess**2),
        "elongation": _prefine_first_finite(record, "elongation", "qi_max_elongation", "qi_elongation"),
        "elongation_threshold": elongation_threshold,
        "elongation_excess": elongation_excess,
        "elongation_penalty": None if elongation_excess is None else float(elongation_excess**2),
        "elongation_weighted_penalty": None
        if elongation_excess is None
        else float(elongation_weight * elongation_excess**2),
        "aspect": _finite_float(record.get("aspect")),
        "target_aspect": _finite_float(record.get("target_aspect")),
        "mean_iota": _finite_float(record.get("mean_iota")),
        "abs_iota_min": _finite_float(record.get("abs_iota_min")),
    }
    return {key: value for key, value in snapshot.items() if value is not None}


def _prefine_snapshot_delta(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "qi_residual",
        "qi_legacy_total",
        "mirror_ratio",
        "mirror_excess",
        "mirror_penalty",
        "mirror_weighted_penalty",
        "elongation",
        "elongation_excess",
        "elongation_penalty",
        "elongation_weighted_penalty",
        "aspect",
        "mean_iota",
    )
    delta = {}
    for key in keys:
        value = _prefine_metric_delta(_finite_float(initial.get(key)), _finite_float(final.get(key)))
        if value is not None:
            delta[key] = value
    return delta


def _prefine_record_from_flat_result(result: dict[str, Any], prefix: str) -> dict[str, Any]:
    aliases = {
        "qi_smooth_total": (f"qi_smooth_{prefix}", f"qi_residual_{prefix}", f"smooth_qi_{prefix}"),
        "qi_legacy_total": (f"qi_legacy_{prefix}", f"legacy_qi_{prefix}"),
        "qi_mirror_ratio_max": (f"mirror_ratio_{prefix}", f"qi_mirror_ratio_{prefix}"),
        "qi_mirror_excess_max": (f"mirror_excess_{prefix}", f"qi_mirror_excess_{prefix}"),
        "qi_max_elongation": (f"elongation_{prefix}", f"qi_elongation_{prefix}"),
        "qi_elongation_excess": (f"elongation_excess_{prefix}", f"qi_elongation_excess_{prefix}"),
        "aspect": (f"aspect_{prefix}",),
        "mean_iota": (f"mean_iota_{prefix}", f"iota_{prefix}"),
    }
    out: dict[str, Any] = {}
    for target_key, source_keys in aliases.items():
        for source_key in source_keys:
            if source_key in result:
                out[target_key] = result[source_key]
                break
    return out


def _prefine_normalized_objective_diagnostics(
    plan: dict[str, Any],
    result: dict[str, Any],
    *,
    objective_initial: float | None,
    objective_final: float | None,
) -> dict[str, Any]:
    raw_qi_options = plan.get("qi_options", {})
    qi_options_raw = raw_qi_options if isinstance(raw_qi_options, dict) else {}
    existing = result.get("objective_diagnostics") or result.get("diagnostic_decomposition")
    existing_initial = existing.get("initial") if isinstance(existing, dict) else None
    existing_final = existing.get("final") if isinstance(existing, dict) else None

    raw_audit_metrics = plan.get("audit_metrics", {})
    initial_record = dict(raw_audit_metrics) if isinstance(raw_audit_metrics, dict) else {}
    initial_record.update(_prefine_record_from_flat_result(result, "initial"))
    if isinstance(existing_initial, dict):
        initial_record.update(existing_initial)

    raw_final_diagnostics = result.get("final_diagnostics", {})
    final_record = dict(raw_final_diagnostics) if isinstance(raw_final_diagnostics, dict) else {}
    final_record.update(_prefine_record_from_flat_result(result, "final"))
    if isinstance(existing_final, dict):
        final_record.update(existing_final)

    initial = _prefine_snapshot_from_record(initial_record, qi_options_raw)
    final = _prefine_snapshot_from_record(final_record, qi_options_raw)

    delta = _prefine_snapshot_delta(initial, final)
    scalar_improved = None
    if objective_initial is not None and objective_final is not None:
        scalar_improved = bool(
            objective_initial > objective_final + _prefine_objective_regression_tolerance(objective_initial, objective_final)
        )
    smooth_worsened = _prefine_metric_worsened(
        _finite_float(initial.get("qi_residual")),
        _finite_float(final.get("qi_residual")),
    )
    legacy_worsened = _prefine_metric_worsened(
        _finite_float(initial.get("qi_legacy_total")),
        _finite_float(final.get("qi_legacy_total")),
    )
    worsened_terms = []
    if smooth_worsened is True:
        worsened_terms.append("smooth_qi")
    if legacy_worsened is True:
        worsened_terms.append("legacy_qi")
    scalar_improved_qi_worsened = bool(scalar_improved is True and worsened_terms)
    return {
        "initial": initial,
        "final": final,
        "delta": delta,
        "flags": {
            "scalar_objective_improved": scalar_improved,
            "smooth_qi_worsened": smooth_worsened,
            "legacy_qi_worsened": legacy_worsened,
            "worsened_qi_terms": worsened_terms,
            "scalar_improved_but_qi_worsened": scalar_improved_qi_worsened,
        },
    }


def _prefine_probe_diagnostic_record_from_files(
    *,
    input_path: Path,
    wout_path: Path,
    qi_options_raw: dict[str, Any],
) -> dict[str, Any] | None:
    """Evaluate final probe diagnostics from already-written artifacts only."""

    if not input_path.exists() or not wout_path.exists():
        return None
    indata = read_indata(input_path)
    cfg = config_from_indata(indata)
    static = build_static(cfg)
    wout = read_wout(wout_path)
    state = state_from_wout(wout)
    signgs = int(wout.signgs)
    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    options = QIDiagnosticOptions(
        surfaces=tuple(float(surface) for surface in qi_options_raw.get("surfaces", ())),
        mboz=int(qi_options_raw.get("mboz", 8)),
        nboz=int(qi_options_raw.get("nboz", 8)),
        nphi=int(qi_options_raw.get("nphi", 31)),
        nalpha=int(qi_options_raw.get("nalpha", 7)),
        n_bounce=int(qi_options_raw.get("n_bounce", 9)),
        include_bounce_endpoints=bool(qi_options_raw.get("include_bounce_endpoints", False)),
        mirror_threshold=float(qi_options_raw.get("mirror_threshold", DEFAULT_MAX_MIRROR_RATIO)),
        mirror_ntheta=int(qi_options_raw.get("mirror_ntheta", 32)),
        mirror_nphi=int(qi_options_raw.get("mirror_nphi", 32)),
        mirror_surface_index=(
            None
            if qi_options_raw.get("mirror_surface_index", None) is None
            else int(qi_options_raw.get("mirror_surface_index"))
        ),
        elongation_threshold=float(qi_options_raw.get("elongation_threshold", DEFAULT_MAX_ELONGATION)),
        elongation_ntheta=int(qi_options_raw.get("elongation_ntheta", 24)),
        elongation_nphi=int(qi_options_raw.get("elongation_nphi", 8)),
        phimin=float(qi_options_raw.get("phimin", 0.0)),
        fail_on_error=False,
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
    return {
        "aspect": _first_float(equilibrium_aspect_ratio_from_state(state=state, static=static)),
        "mean_iota": _mean_iota(iotas),
        **qi_record,
    }


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items: Any = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, np.ndarray):
        raw_items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = [value]
    out: list[int] = []
    for item in raw_items:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _prefine_objective_regression_tolerance(previous: float, current: float) -> float:
    scale = max(1.0, abs(float(previous)), abs(float(current)))
    return max(OBJECTIVE_REGRESSION_ATOL, OBJECTIVE_REGRESSION_RTOL * scale)


def _objective_values_from_history_payload(history_payload: Any) -> list[float | None]:
    if history_payload is None:
        return []
    if isinstance(history_payload, dict):
        for key in ("history", "objective_history", "objective_trace", "objectives"):
            if key in history_payload:
                return _objective_values_from_history_payload(history_payload[key])
        return []
    if not isinstance(history_payload, (list, tuple, np.ndarray)):
        return []

    values: list[float | None] = []
    for entry in history_payload:
        raw_value: Any = entry
        if isinstance(entry, dict):
            raw_value = None
            for key in ("objective", "total_objective", "objective_total", "cost"):
                if key in entry:
                    raw_value = entry[key]
                    break
        values.append(_finite_float(raw_value))
    return values


def _prefine_history_summary(
    history_payload: Any,
    *,
    objective_initial: Any = None,
    objective_final: Any = None,
) -> dict[str, Any]:
    raw_values = _objective_values_from_history_payload(history_payload)
    finite_values = [float(value) for value in raw_values if value is not None]
    nonfinite_count = len(raw_values) - len(finite_values)
    summary: dict[str, Any] = {
        "history_present": bool(raw_values),
        "objective_sample_count": len(raw_values),
        "finite_objective_sample_count": len(finite_values),
        "nonfinite_objective_sample_count": nonfinite_count,
        "objective_first": finite_values[0] if finite_values else None,
        "objective_last": finite_values[-1] if finite_values else None,
        "objective_min": min(finite_values) if finite_values else None,
        "objective_max": max(finite_values) if finite_values else None,
        "objective_decrease": (finite_values[0] - finite_values[-1]) if len(finite_values) >= 2 else None,
        "objective_monotonic_nonincreasing": None,
        "objective_regression_count": 0,
        "max_objective_increase": None,
        "final_worse_than_initial": None,
    }
    if len(finite_values) >= 2:
        increases = []
        for previous, current in zip(finite_values, finite_values[1:], strict=False):
            if current > previous + _prefine_objective_regression_tolerance(previous, current):
                increases.append(current - previous)
        summary["objective_regression_count"] = len(increases)
        summary["max_objective_increase"] = max(increases) if increases else 0.0
        summary["objective_monotonic_nonincreasing"] = bool(not increases and nonfinite_count == 0)

    initial = _finite_float(objective_initial)
    final = _finite_float(objective_final)
    if initial is None and finite_values:
        initial = finite_values[0]
    if final is None and finite_values:
        final = finite_values[-1]
    if initial is not None and final is not None:
        summary["final_worse_than_initial"] = bool(
            final > initial + _prefine_objective_regression_tolerance(initial, final)
        )
    return summary


def _prefine_existing_history_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary = dict(value)
    if "history_present" not in summary:
        return None
    for key in (
        "objective_sample_count",
        "finite_objective_sample_count",
        "nonfinite_objective_sample_count",
        "objective_regression_count",
    ):
        if key in summary and summary[key] is not None:
            summary[key] = int(summary[key])
    for key in (
        "objective_first",
        "objective_last",
        "objective_min",
        "objective_max",
        "objective_decrease",
        "max_objective_increase",
    ):
        summary[key] = _finite_float(summary.get(key))
    if "objective_monotonic_nonincreasing" in summary and summary["objective_monotonic_nonincreasing"] is not None:
        summary["objective_monotonic_nonincreasing"] = bool(summary["objective_monotonic_nonincreasing"])
    if "final_worse_than_initial" in summary and summary["final_worse_than_initial"] is not None:
        summary["final_worse_than_initial"] = bool(summary["final_worse_than_initial"])
    return summary


def _prefine_history_payload(plan: dict[str, Any], result: dict[str, Any]) -> Any:
    for payload in (
        result.get("history_dump"),
        result.get("_history_dump"),
        result.get("history"),
        result.get("objective_history"),
        result.get("objective_trace"),
        plan.get("history_dump"),
        plan.get("_history_dump"),
        plan.get("history"),
        plan.get("objective_history"),
        plan.get("objective_trace"),
    ):
        if payload is not None:
            return payload
    return None


def _prefine_stage_modes_from_plan(plan: dict[str, Any], result: dict[str, Any], key: str) -> list[int]:
    candidates: list[Any]
    if key == "requested":
        candidates = [
            result.get("requested_stage_modes"),
            plan.get("requested_stage_modes"),
            plan.get("optimization", {}).get("stage_modes"),
            [stage.get("mode") for stage in plan.get("stages", []) if isinstance(stage, dict)],
        ]
    else:
        history_payload = _prefine_history_payload(plan, result)
        history_modes = history_payload.get("stage_modes") if isinstance(history_payload, dict) else None
        candidates = [
            result.get("completed_stage_modes"),
            plan.get("completed_stage_modes"),
            history_modes,
        ]
    for candidate in candidates:
        values = _int_list(candidate)
        if values:
            return values
    return []


def _prefine_plan_status(plan: dict[str, Any]) -> str:
    status = plan.get("status")
    if status is not None:
        return str(status).strip().lower() or "unknown"
    if bool(plan.get("crashed", False)):
        return "failed"
    if plan.get("success") is True:
        return "completed"
    if plan.get("success") is False:
        return "failed"
    return "unknown"


def _prefine_plan_timed_out(plan: dict[str, Any], result: dict[str, Any], status: str) -> bool:
    if status in {"timeout", "timed_out"}:
        return True
    if bool(plan.get("timed_out", False) or result.get("timed_out", False)):
        return True
    pieces = [
        plan.get("error_type"),
        plan.get("error"),
        plan.get("message"),
        result.get("error_type"),
        result.get("error"),
        result.get("message"),
        result.get("optimizer_message"),
    ]
    text = " ".join(str(piece) for piece in pieces if piece is not None).lower()
    return "timeout" in text or "timed out" in text


def _prefine_plan_acceptance(row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("status", "unknown"))
    reasons: list[str] = []
    if bool(row.get("timed_out", False)):
        return {"accepted": False, "decision": "timeout", "reasons": ["probe timed out"]}
    if bool(row.get("failed", False)):
        return {"accepted": False, "decision": "failed", "reasons": ["probe failed"]}
    if status in {"planned", "pending"}:
        return {"accepted": False, "decision": "not_run", "reasons": [f"probe is {status}"]}
    if status != "completed":
        return {"accepted": False, "decision": "unknown_status", "reasons": [f"status is {status}"]}

    if row.get("objective_final") is None:
        reasons.append("missing finite final objective")
    if row.get("completed_all_requested_stages") is False:
        reasons.append("did not complete all requested stage modes")
    if row.get("objective_final_regressed") is True:
        reasons.append("final objective is worse than initial objective")
    if bool(row.get("scalar_improved_but_qi_worsened", False)):
        reasons.append("scalar objective improved while smooth/legacy QI worsened")

    history = row.get("history_summary", {})
    if bool(history.get("nonfinite_objective_sample_count", 0)):
        reasons.append("history contains non-finite objective samples")
    if int(history.get("objective_regression_count") or 0) > 0:
        reasons.append("history contains objective regressions")

    improvement = row.get("objective_improvement")
    if improvement is not None and improvement <= _prefine_objective_regression_tolerance(
        float(row.get("objective_initial") or 0.0),
        float(row.get("objective_final") or 0.0),
    ):
        objective_final = _finite_float(row.get("objective_final"))
        if objective_final is not None and objective_final <= PREFINE_STABLE_LOW_OBJECTIVE_THRESHOLD:
            return {
                "accepted": True,
                "decision": "accepted_stable_low_objective",
                "reasons": [
                    "completed with finite stable low objective",
                    f"objective_final <= {PREFINE_STABLE_LOW_OBJECTIVE_THRESHOLD:g}",
                ],
            }
        reasons.append("no positive objective improvement")

    if reasons:
        return {"accepted": False, "decision": "needs_review", "reasons": reasons}
    return {"accepted": True, "decision": "accepted", "reasons": ["completed with finite improved objective"]}


def _prefine_plan_result_summary(plan: dict[str, Any], *, index: int) -> dict[str, Any]:
    result = plan.get("result") if isinstance(plan.get("result"), dict) else {}
    status = _prefine_plan_status(plan)
    objective_initial = _finite_float(result.get("objective_initial", plan.get("objective_initial")))
    objective_final = _finite_float(result.get("objective_final", plan.get("objective_final")))
    objective_improvement = None
    objective_relative_improvement = None
    objective_final_regressed = None
    if objective_initial is not None and objective_final is not None:
        objective_improvement = objective_initial - objective_final
        if abs(objective_initial) > 0.0:
            objective_relative_improvement = objective_improvement / abs(objective_initial)
        objective_final_regressed = bool(
            objective_final > objective_initial + _prefine_objective_regression_tolerance(objective_initial, objective_final)
        )

    history_summary = _prefine_existing_history_summary(result.get("history_summary"))
    if history_summary is None:
        history_summary = _prefine_history_summary(
            _prefine_history_payload(plan, result),
            objective_initial=objective_initial,
            objective_final=objective_final,
        )
    objective_diagnostics = _prefine_normalized_objective_diagnostics(
        plan,
        result,
        objective_initial=objective_initial,
        objective_final=objective_final,
    )
    diagnostic_flags = objective_diagnostics.get("flags", {})

    requested_stage_modes = _prefine_stage_modes_from_plan(plan, result, "requested")
    completed_stage_modes = _prefine_stage_modes_from_plan(plan, result, "completed")
    stage_count_requested = int(result.get("stage_count_requested") or len(requested_stage_modes))
    stage_count_completed = int(result.get("stage_count_completed") or len(completed_stage_modes))
    completed_all_requested_stages = None
    if requested_stage_modes and completed_stage_modes:
        completed_all_requested_stages = completed_stage_modes == requested_stage_modes
    elif stage_count_requested and stage_count_completed:
        completed_all_requested_stages = stage_count_completed == stage_count_requested

    timed_out = _prefine_plan_timed_out(plan, result, status)
    failed = bool(timed_out or status in {"failed", "error", "crashed"} or plan.get("crashed", False))
    row = {
        "index": int(index),
        "label": str(plan.get("label", f"probe_{index}")),
        "family": plan.get("family"),
        "audit_rank": plan.get("audit_rank"),
        "status": status,
        "failed": failed,
        "timed_out": timed_out,
        "error_type": plan.get("error_type", result.get("error_type")),
        "error": plan.get("error", result.get("error")),
        "objective_initial": objective_initial,
        "objective_final": objective_final,
        "objective_improvement": objective_improvement,
        "objective_relative_improvement": objective_relative_improvement,
        "objective_final_regressed": objective_final_regressed,
        "requested_stage_modes": requested_stage_modes,
        "completed_stage_modes": completed_stage_modes,
        "stage_count_requested": stage_count_requested,
        "stage_count_completed": stage_count_completed,
        "completed_all_requested_stages": completed_all_requested_stages,
        "history_summary": history_summary,
        "objective_diagnostics": objective_diagnostics,
        "scalar_improved_but_qi_worsened": bool(diagnostic_flags.get("scalar_improved_but_qi_worsened", False)),
        "worsened_qi_terms": diagnostic_flags.get("worsened_qi_terms", []),
        "output_dir": plan.get("output_dir"),
        "phimin": result.get("phimin", plan.get("qi_options", {}).get("phimin")),
        "endpoint_mode": result.get("endpoint_mode", plan.get("endpoint_mode")),
    }
    row["acceptance"] = _prefine_plan_acceptance(row)
    return row


def _compact_prefine_summary_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "index": row.get("index"),
        "label": row.get("label"),
        "family": row.get("family"),
        "audit_rank": row.get("audit_rank"),
        "status": row.get("status"),
        "objective_initial": row.get("objective_initial"),
        "objective_final": row.get("objective_final"),
        "objective_improvement": row.get("objective_improvement"),
        "objective_relative_improvement": row.get("objective_relative_improvement"),
        "objective_diagnostics": row.get("objective_diagnostics"),
        "scalar_improved_but_qi_worsened": row.get("scalar_improved_but_qi_worsened"),
        "worsened_qi_terms": row.get("worsened_qi_terms", []),
        "requested_stage_modes": row.get("requested_stage_modes", []),
        "completed_stage_modes": row.get("completed_stage_modes", []),
        "completed_all_requested_stages": row.get("completed_all_requested_stages"),
        "output_dir": row.get("output_dir"),
        "acceptance": row.get("acceptance"),
    }


def _prefine_best_final_key(row: dict[str, Any]) -> tuple[float, int, str]:
    return (float(row["objective_final"]), int(row.get("audit_rank") or row.get("index") or 0), str(row.get("label", "")))


def _prefine_best_improvement_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
    final = row.get("objective_final")
    final_key = float(final) if final is not None else float("inf")
    return (
        -float(row["objective_improvement"]),
        final_key,
        int(row.get("audit_rank") or row.get("index") or 0),
        str(row.get("label", "")),
    )


def _prefine_next_pending_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("status") in {"planned", "pending"}), None)


def _prefine_probe_recommendation(
    manifest: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    accepted_candidate: dict[str, Any] | None,
    finite_completed: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    timeouts: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    qi_worsening_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pending = _prefine_next_pending_row(rows)
    if bool(manifest.get("dry_run", True)):
        label = pending["label"] if pending is not None else None
        message = (
            f"Review the manifest, then run the planned probe '{label}' with --prefine-reviewed; keep caps unchanged."
            if label is not None
            else "Review the manifest before adding or running prefine probes."
        )
        return {"action": "review_manifest", "label": label, "message": message}
    if timeouts:
        label = str(timeouts[0]["label"])
        return {
            "action": "inspect_timeout",
            "label": label,
            "message": f"Inspect timeout '{label}' before expanding the sweep; rerun only with reviewed bounded settings.",
        }
    if failures:
        label = str(failures[0]["label"])
        return {
            "action": "inspect_failure",
            "label": label,
            "message": f"Inspect failed probe '{label}' before promoting a seed or broadening family coverage.",
        }
    if regression_rows:
        label = str(regression_rows[0]["label"])
        return {
            "action": "review_objective_regression",
            "label": label,
            "message": f"Review objective history for '{label}' because it contains objective regressions.",
        }
    if qi_worsening_rows:
        label = str(qi_worsening_rows[0]["label"])
        terms = ", ".join(str(term) for term in qi_worsening_rows[0].get("worsened_qi_terms", []))
        return {
            "action": "review_qi_worsening",
            "label": label,
            "message": (
                f"Review '{label}' because the scalar objective improved while "
                f"{terms or 'smooth/legacy QI'} worsened."
            ),
        }
    if accepted_candidate is not None:
        label = str(accepted_candidate["label"])
        if pending is not None:
            return {
                "action": "run_next_pending_probe",
                "label": str(pending["label"]),
                "accepted_label": label,
                "message": (
                    f"Keep '{label}' as the current accepted prefine candidate; run pending probe "
                    f"'{pending['label']}' next if family coverage is still required."
                ),
            }
        return {
            "action": "promote_best_candidate",
            "label": label,
            "message": f"Promote '{label}' as the current prefine candidate; do not expand caps without a new reviewed plan.",
        }
    if pending is not None:
        label = str(pending["label"])
        return {
            "action": "run_pending_probe",
            "label": label,
            "message": f"Run pending probe '{label}' with the reviewed bounded manifest before drawing conclusions.",
        }
    if finite_completed:
        label = str(finite_completed[0]["label"])
        return {
            "action": "manual_review",
            "label": label,
            "message": f"Review completed probe '{label}' manually; no candidate passed automatic acceptance checks.",
        }
    return {
        "action": "no_actionable_result",
        "label": None,
        "message": "No finite completed prefine result is available; keep robustness claims deferred.",
    }


def summarize_qi_prefine_probe_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Summarize bounded prefine probe plans/results without reading large artifacts."""

    rows = [
        _prefine_plan_result_summary(plan, index=index)
        for index, plan in enumerate(manifest.get("plans", []), start=1)
        if isinstance(plan, dict)
    ]
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1

    completed = [row for row in rows if row.get("status") == "completed"]
    finite_completed = [row for row in completed if row.get("objective_final") is not None]
    finite_completed = sorted(finite_completed, key=_prefine_best_final_key)
    improvement_rows = [row for row in rows if row.get("objective_improvement") is not None]
    improvement_rows = sorted(improvement_rows, key=_prefine_best_improvement_key)
    failures = [row for row in rows if bool(row.get("failed", False))]
    timeouts = [row for row in rows if bool(row.get("timed_out", False))]
    regression_rows = [
        row
        for row in rows
        if bool(row.get("objective_final_regressed", False))
        or int(row.get("history_summary", {}).get("objective_regression_count") or 0) > 0
    ]
    qi_worsening_rows = [row for row in rows if bool(row.get("scalar_improved_but_qi_worsened", False))]
    accepted_rows = [row for row in finite_completed if bool(row.get("acceptance", {}).get("accepted", False))]
    accepted_candidate = min(accepted_rows, key=_prefine_best_final_key) if accepted_rows else None
    best_final = finite_completed[0] if finite_completed else None
    best_improvement = improvement_rows[0] if improvement_rows else None
    recommendation = _prefine_probe_recommendation(
        manifest,
        rows=rows,
        accepted_candidate=accepted_candidate,
        finite_completed=finite_completed,
        failures=failures,
        timeouts=timeouts,
        regression_rows=regression_rows,
        qi_worsening_rows=qi_worsening_rows,
    )
    blocking_issues = []
    if timeouts:
        blocking_issues.append(f"{len(timeouts)} timeout(s)")
    if failures:
        blocking_issues.append(f"{len(failures)} failed probe(s)")
    if regression_rows:
        blocking_issues.append(f"{len(regression_rows)} objective-regression probe(s)")
    if qi_worsening_rows:
        blocking_issues.append(f"{len(qi_worsening_rows)} scalar-improved QI-worsening probe(s)")

    return {
        "schema_version": 1,
        "dry_run": bool(manifest.get("dry_run", True)),
        "planned_rows": int(manifest.get("selection", {}).get("planned_rows", len(rows))),
        "statuses": statuses,
        "completed_count": len(completed),
        "completed_stage_modes": [
            {
                "label": row["label"],
                "family": row.get("family"),
                "requested_stage_modes": row.get("requested_stage_modes", []),
                "completed_stage_modes": row.get("completed_stage_modes", []),
                "completed_all_requested_stages": row.get("completed_all_requested_stages"),
            }
            for row in completed
        ],
        "best_candidate_by_final_objective": _compact_prefine_summary_row(best_final),
        "best_improvement": _compact_prefine_summary_row(best_improvement),
        "failure_count": len(failures),
        "failures": [_compact_prefine_summary_row(row) for row in failures],
        "timeout_count": len(timeouts),
        "timeouts": [_compact_prefine_summary_row(row) for row in timeouts],
        "history_regression_plan_count": len(regression_rows),
        "history_regressions": [
            {
                **(_compact_prefine_summary_row(row) or {}),
                "history_summary": row.get("history_summary", {}),
                "objective_final_regressed": row.get("objective_final_regressed"),
            }
            for row in regression_rows
        ],
        "scalar_improved_qi_worsened_count": len(qi_worsening_rows),
        "scalar_improved_qi_worsened": [_compact_prefine_summary_row(row) for row in qi_worsening_rows],
        "accepted_candidate": _compact_prefine_summary_row(accepted_candidate),
        "acceptance": {
            "decision": "accepted" if accepted_candidate is not None and not blocking_issues else "needs_review",
            "accepted": bool(accepted_candidate is not None and not blocking_issues),
            "accepted_candidate": _compact_prefine_summary_row(accepted_candidate),
            "blocking_issues": blocking_issues,
        },
        "recommendation": recommendation,
        "plan_summaries": rows,
    }


def run_qi_prefine_probe(plan: dict[str, Any], *, workflow: Any | None = None) -> dict[str, Any]:
    """Run one explicit tiny QI-only probe from a manifest plan."""

    if workflow is None:
        import vmec_jax as workflow

    qi_options_raw = plan["qi_options"]
    opt = plan["optimization"]
    requested_stage_modes = tuple(int(mode) for mode in opt["stage_modes"])
    qi_surfaces = tuple(float(surface) for surface in qi_options_raw["surfaces"])
    qi_mboz = int(qi_options_raw["mboz"])
    qi_nboz = int(qi_options_raw["nboz"])
    qi_phimin = float(qi_options_raw["phimin"])
    qi_jit_booz = bool(qi_options_raw.get("jit_booz", True))
    qi_options = workflow.QuasiIsodynamicOptions(
        surfaces=qi_surfaces,
        mboz=qi_mboz,
        nboz=qi_nboz,
        nphi=int(qi_options_raw["nphi"]),
        nalpha=int(qi_options_raw["nalpha"]),
        n_bounce=int(qi_options_raw["n_bounce"]),
        include_bounce_endpoints=bool(qi_options_raw.get("include_bounce_endpoints", False)),
        phimin=qi_phimin,
        jit_booz=qi_jit_booz,
    )
    qi_surfaces = tuple(float(surface) for surface in getattr(qi_options, "surfaces", qi_options_raw["surfaces"]))
    qi_mboz = int(getattr(qi_options, "mboz", qi_options_raw["mboz"]))
    qi_nboz = int(getattr(qi_options, "nboz", qi_options_raw["nboz"]))
    qi_phimin = float(getattr(qi_options, "phimin", qi_options_raw["phimin"]))
    qi_jit_booz = bool(getattr(qi_options, "jit_booz", False))
    objective_tuples = []
    qi = workflow.QuasiIsodynamicResidual(qi_options)
    objective_tuples.append((qi.J, 0.0, float(qi_options_raw["weight"])))
    if (
        float(qi_options_raw.get("qi_ceiling_weight", 0.0)) > 0.0
        and (
            float(qi_options_raw.get("mirror_weight", 0.0)) > 0.0
            or float(qi_options_raw.get("elongation_weight", 0.0)) > 0.0
        )
    ):
        qi_ceiling = workflow.QuasiIsodynamicResidualCeiling(
            maximum=float(qi_options_raw.get("qi_ceiling_max", 2.0e-3)),
            smooth_penalty=float(qi_options_raw.get("qi_ceiling_smooth_penalty", 2.0e-3)),
            qi_options=qi_options,
        )
        objective_tuples.append((qi_ceiling.J, 0.0, float(qi_options_raw["qi_ceiling_weight"])))
    if float(qi_options_raw.get("mirror_weight", 0.0)) > 0.0:
        mirror = workflow.MirrorRatio(
            threshold=float(qi_options_raw.get("mirror_threshold", DEFAULT_MAX_MIRROR_RATIO)),
            surfaces=qi_surfaces,
            mboz=qi_mboz,
            nboz=qi_nboz,
            ntheta=int(qi_options_raw.get("mirror_ntheta", 32)),
            nphi=int(qi_options_raw.get("mirror_nphi", 32)),
            surface_index=(
                None
                if qi_options_raw.get("mirror_surface_index", None) is None
                else int(qi_options_raw.get("mirror_surface_index"))
            ),
            phimin=qi_phimin,
            jit_booz=qi_jit_booz,
        )
        objective_tuples.append((mirror.J, 0.0, float(qi_options_raw["mirror_weight"])))
    if float(qi_options_raw.get("elongation_weight", 0.0)) > 0.0:
        elongation = workflow.MaxElongation(
            threshold=float(qi_options_raw.get("elongation_threshold", DEFAULT_MAX_ELONGATION)),
            ntheta=int(qi_options_raw.get("elongation_ntheta", 24)),
            nphi=int(qi_options_raw.get("elongation_nphi", 8)),
        )
        objective_tuples.append((elongation.J, 0.0, float(qi_options_raw["elongation_weight"])))
    problem = workflow.LeastSquaresProblem.from_tuples(objective_tuples)
    vmec = workflow.FixedBoundaryVMEC.from_input(
        plan["input"],
        max_mode=int(opt["max_mode"]),
        min_vmec_mode=int(opt["min_vmec_mode"]),
        output_dir=Path(plan["output_dir"]),
        project_input_boundary_to_max_mode=True,
    )
    result = workflow.least_squares_solve(
        vmec,
        problem,
        stage_modes=requested_stage_modes,
        max_nfev=int(opt["max_nfev"]),
        continuation_nfev=int(opt["continuation_nfev"]),
        method=str(opt["method"]),
        ftol=float(opt["ftol"]),
        gtol=float(opt["gtol"]),
        xtol=float(opt["xtol"]),
        use_ess=bool(opt["use_ess"]),
        ess_alpha=float(opt["ess_alpha"]),
        label=f"QI prefine probe: {plan['label']}",
        use_mode_continuation=len(opt["stage_modes"]) > 1,
        inner_max_iter=int(opt["inner_max_iter"]),
        trial_max_iter=int(opt["trial_max_iter"]),
        inner_ftol=float(opt["inner_ftol"]),
        trial_ftol=float(opt["trial_ftol"]),
        scipy_tr_solver=str(opt["scipy_tr_solver"]),
        scipy_lsmr_maxiter=opt["scipy_lsmr_maxiter"],
        save_stage_inputs=True,
        save_stage_wouts=False,
    )
    history = dict(result.final_result.get("_history_dump", {}))
    completed_stage_modes = [int(mode) for mode in getattr(result, "stage_modes", requested_stage_modes)]
    objective_initial = history.get("objective_initial")
    objective_final = history.get("objective_final")
    final_diagnostics = None
    diagnostic_error = None
    try:
        final_diagnostics = _prefine_probe_diagnostic_record_from_files(
            input_path=Path(plan["output_dir"]) / "input.final",
            wout_path=Path(plan["output_dir"]) / "wout_final.nc",
            qi_options_raw=qi_options_raw,
        )
    except Exception as exc:
        diagnostic_error = _format_error(exc)
    history_objective_diagnostics = (
        history.get("objective_diagnostics") if isinstance(history.get("objective_diagnostics"), dict) else None
    )
    completed = dict(plan)
    completed["status"] = "completed"
    completed["review"] = {
        **dict(plan.get("review", {})),
        "operator_confirmed": bool(plan.get("review", {}).get("operator_confirmed", False)),
    }
    completed["result"] = {
        "objective_initial": objective_initial,
        "objective_final": objective_final,
        "qi_final": history.get("qs_final"),
        "wall_time_s": history.get("total_wall_time_s"),
        "optimizer_success": history.get("success"),
        "optimizer_message": history.get("message"),
        "nfev": history.get("nfev"),
        "njev": history.get("njev"),
        "requested_stage_modes": list(requested_stage_modes),
        "completed_stage_modes": completed_stage_modes,
        "stage_count_requested": len(requested_stage_modes),
        "stage_count_completed": len(completed_stage_modes),
        "stage_plan": plan.get("stages", opt.get("stage_plan")),
        "total_nfev_cap": opt.get("total_nfev_cap"),
        "endpoint_mode": qi_options_raw.get(
            "endpoint_mode",
            _prefine_endpoint_mode(bool(qi_options_raw.get("include_bounce_endpoints", False))),
        ),
        "phimin": float(qi_options_raw["phimin"]),
        "history_path": str(Path(plan["output_dir"]) / "history.json"),
        "wout_final": str(Path(plan["output_dir"]) / "wout_final.nc"),
        "history_summary": _prefine_history_summary(
            history,
            objective_initial=objective_initial,
            objective_final=objective_final,
        ),
    }
    if history_objective_diagnostics is not None:
        completed["result"]["objective_diagnostics"] = history_objective_diagnostics
    if final_diagnostics is not None:
        completed["result"]["final_diagnostics"] = final_diagnostics
    if diagnostic_error is not None:
        completed["result"]["diagnostic_error"] = diagnostic_error
    return completed


def _prefine_manifest_has_review(manifest: dict[str, Any]) -> bool:
    review = manifest.get("review", {})
    return bool(review.get("operator_confirmed", False) and review.get("status") == "reviewed")


def run_qi_prefine_probe_manifest(
    manifest: dict[str, Any],
    *,
    fail_on_error: bool = False,
    require_review: bool = False,
    workflow: Any | None = None,
) -> dict[str, Any]:
    """Execute all pending plans in a manifest and record bounded outcomes."""

    if bool(require_review) and not _prefine_manifest_has_review(manifest):
        raise ValueError("prefine probe execution requires a reviewed manifest")
    executed = dict(manifest)
    executed["dry_run"] = False
    plans = []
    for plan in manifest.get("plans", []):
        try:
            plans.append(run_qi_prefine_probe(plan, workflow=workflow))
        except Exception as exc:
            failed = dict(plan)
            failed["status"] = "failed"
            failed["error_type"] = type(exc).__name__
            failed["error"] = str(exc)
            plans.append(failed)
            if fail_on_error:
                raise
    executed["plans"] = plans
    executed["summary"] = summarize_qi_prefine_probe_manifest(executed)
    executed["result_summary"] = summarize_qi_prefine_results(executed)
    return executed


def summarize_qi_prefine_results(manifest: dict[str, Any]) -> dict[str, Any]:
    summary = dict(summarize_qi_prefine_probe_manifest(manifest))
    stage_coverage: dict[str, int] = {}
    for row in summary.get("plan_summaries", []):
        if row.get("status") != "completed":
            continue
        for mode in row.get("completed_stage_modes", []):
            key = str(mode)
            stage_coverage[key] = stage_coverage.get(key, 0) + 1
    objective_regressions = [
        {
            "index": row.get("index"),
            "label": row.get("label"),
            "family": row.get("family"),
            "checked": bool(row.get("history_summary", {}).get("history_present", False)),
            "regression_count": int(row.get("history_summary", {}).get("objective_regression_count") or 0),
            "max_increase": row.get("history_summary", {}).get("max_objective_increase"),
        }
        for row in summary.get("plan_summaries", [])
        if int(row.get("history_summary", {}).get("objective_regression_count") or 0) > 0
    ]
    qi_worsening = [
        {
            "index": row.get("index"),
            "label": row.get("label"),
            "family": row.get("family"),
            "worsened_qi_terms": row.get("worsened_qi_terms", []),
            "objective_diagnostics": row.get("objective_diagnostics", {}),
        }
        for row in summary.get("plan_summaries", [])
        if bool(row.get("scalar_improved_but_qi_worsened", False))
    ]
    completed_count = int(summary.get("completed_count", 0))
    failure_count = int(summary.get("failure_count", 0))
    if completed_count == 0:
        legacy_recommendation = "run_reviewed_prefine_probes" if manifest.get("plans") else "build_prefine_manifest"
    elif objective_regressions:
        legacy_recommendation = "inspect_nonmonotone_histories_before_promoting_seed"
    elif qi_worsening:
        legacy_recommendation = "inspect_scalar_improved_qi_worsening_before_promoting_seed"
    elif failure_count:
        legacy_recommendation = "rerun_failed_probe_or_lower_probe_budget"
    else:
        legacy_recommendation = "promote_best_final_objective_seed"

    summary.update(
        {
            "executed_rows": completed_count + failure_count,
            "stage_coverage": stage_coverage,
            "best_by_final_objective": summary.get("best_candidate_by_final_objective"),
            "best_by_objective_improvement": summary.get("best_improvement"),
            "objective_regressions": objective_regressions,
            "scalar_improved_qi_worsened": qi_worsening,
            "legacy_recommendation": legacy_recommendation,
        }
    )
    return summary


def _prefine_probe_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return summarize_qi_prefine_probe_manifest(manifest)


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
    phimin_policy: str = "fixed",
    include_bounce_endpoints: bool = True,
) -> dict[str, Any]:
    records = []
    for case in cases:
        phimin_candidates = _phimin_candidates_for_case(
            case,
            phimin=phimin,
            phimin_policy=phimin_policy,
        )
        record = _select_best_phimin_record(
            case,
            phimin_candidates=phimin_candidates,
            surfaces=surfaces,
            targets=targets,
            nphi=nphi,
            nalpha=nalpha,
            n_bounce=n_bounce,
            nphi_out=nphi_out,
            mboz=mboz,
            nboz=nboz,
            include_bounce_endpoints=include_bounce_endpoints,
            mirror_ntheta=mirror_ntheta,
            mirror_nphi=mirror_nphi,
            elongation_ntheta=elongation_ntheta,
            elongation_nphi=elongation_nphi,
            fail_on_error=fail_on_error,
        )
        record["phimin_policy"] = phimin_policy
        records.append(record)
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
            "include_bounce_endpoints": bool(include_bounce_endpoints),
            "nphi_out": int(nphi_out),
            "mirror_ntheta": int(mirror_ntheta),
            "mirror_nphi": int(mirror_nphi),
            "elongation_ntheta": int(elongation_ntheta),
            "elongation_nphi": int(elongation_nphi),
            "phimin": float(phimin),
            "phimin_policy": phimin_policy,
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
        "phimin_policy",
        "selected_phimin",
        "qi_smooth_rank",
        "qi_smooth_total",
        "smooth_qi_excess",
        "qi_legacy_rank",
        "qi_legacy_total",
        "legacy_qi_excess",
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
    parser.add_argument(
        "--include-bounce-endpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include normalized bounce levels 0 and 1 in the smooth QI metric, "
            "matching the legacy Goodman-style diagnostic."
        ),
    )
    parser.add_argument(
        "--phimin-policy",
        choices=PHIMIN_POLICIES,
        default="well-phase",
        help=(
            "'fixed' audits only --phimin; 'well-phase' audits both 0 and pi/nfp "
            "and ranks the better QI well phase for each seed."
        ),
    )
    parser.add_argument("--mirror-ntheta", type=int, default=96)
    parser.add_argument("--mirror-nphi", type=int, default=96)
    parser.add_argument("--elongation-ntheta", type=int, default=48)
    parser.add_argument("--elongation-nphi", type=int, default=16)
    parser.add_argument("--target-aspect", type=float, default=DEFAULT_TARGET_ASPECT)
    parser.add_argument("--abs-iota-min", type=float, default=DEFAULT_ABS_IOTA_MIN)
    parser.add_argument(
        "--smooth-qi-max",
        type=float,
        default=2.0e-3,
        help="Maximum accepted smooth differentiable QI diagnostic; use a negative value to disable this gate.",
    )
    parser.add_argument(
        "--legacy-qi-max",
        type=float,
        default=2.0e-3,
        help="Maximum accepted legacy Goodman-style QI diagnostic; use a negative value to disable this gate.",
    )
    parser.add_argument("--max-mirror-ratio", type=float, default=DEFAULT_MAX_MIRROR_RATIO)
    parser.add_argument("--max-elongation", type=float, default=DEFAULT_MAX_ELONGATION)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument(
        "--prefine-probes",
        choices=("none", "plan", "run"),
        default="none",
        help=(
            "Optional bounded QI-only prefine workflow: 'plan' writes a dry-run manifest; "
            "'run' executes the tiny capped probes."
        ),
    )
    parser.add_argument("--prefine-manifest", type=Path, default=DEFAULT_PREFINE_MANIFEST)
    parser.add_argument("--prefine-output-dir", type=Path, default=DEFAULT_PREFINE_OUTPUT_DIR)
    parser.add_argument("--prefine-top-n", type=int, default=1)
    parser.add_argument(
        "--prefine-family-representatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also select the best-ranked available representative from each requested seed family.",
    )
    parser.add_argument(
        "--prefine-representative-families",
        type=parse_seed_families,
        default=SEED_FAMILY_ORDER,
        help="Comma-separated families to cover with representative prefine probes.",
    )
    parser.add_argument("--prefine-max-nfev", type=int, default=2)
    parser.add_argument("--prefine-continuation-nfev", type=int, default=1)
    parser.add_argument("--prefine-max-mode", type=int, default=3)
    parser.add_argument("--prefine-min-vmec-mode", type=int, default=3)
    parser.add_argument("--prefine-stage-modes", type=parse_stage_modes, default=DEFAULT_PREFINE_STAGE_MODES)
    parser.add_argument("--prefine-surfaces", type=parse_surfaces, default=DEFAULT_PREFINE_SURFACES)
    parser.add_argument("--prefine-mboz", type=int, default=8)
    parser.add_argument("--prefine-nboz", type=int, default=8)
    parser.add_argument("--prefine-nphi", type=int, default=31)
    parser.add_argument("--prefine-nalpha", type=int, default=7)
    parser.add_argument("--prefine-n-bounce", type=int, default=9)
    parser.add_argument(
        "--prefine-include-bounce-endpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use legacy-style bounce endpoint levels in bounded QI prefine probes. "
            "Defaults to the audit --include-bounce-endpoints setting."
        ),
    )
    parser.add_argument("--prefine-phimin", type=float, default=0.0)
    parser.add_argument(
        "--prefine-qi-ceiling-weight",
        type=float,
        default=100.0,
        help=(
            "QI residual ceiling weight used when mirror or elongation cleanup "
            "is enabled in constrained prefine probes."
        ),
    )
    parser.add_argument("--prefine-qi-ceiling-max", type=float, default=2.0e-3)
    parser.add_argument("--prefine-qi-ceiling-smooth-penalty", type=float, default=2.0e-3)
    parser.add_argument(
        "--prefine-mirror-weight",
        type=float,
        default=0.0,
        help="Optional mirror-ratio penalty weight for constrained prefine probes. Default 0 keeps QI-only behavior.",
    )
    parser.add_argument("--prefine-mirror-threshold", type=float, default=DEFAULT_MAX_MIRROR_RATIO)
    parser.add_argument("--prefine-mirror-ntheta", type=int, default=32)
    parser.add_argument("--prefine-mirror-nphi", type=int, default=32)
    parser.add_argument(
        "--prefine-mirror-surface-index",
        type=parse_optional_surface_index,
        default=None,
        help="Mirror-ratio surface index for constrained prefine probes, or 'all' to use all Boozer surfaces.",
    )
    parser.add_argument(
        "--prefine-elongation-weight",
        type=float,
        default=0.0,
        help="Optional max-elongation penalty weight for constrained prefine probes. Default 0 keeps QI-only behavior.",
    )
    parser.add_argument("--prefine-elongation-threshold", type=float, default=DEFAULT_MAX_ELONGATION)
    parser.add_argument("--prefine-elongation-ntheta", type=int, default=24)
    parser.add_argument("--prefine-elongation-nphi", type=int, default=8)
    parser.add_argument("--prefine-inner-max-iter", type=int, default=20)
    parser.add_argument("--prefine-trial-max-iter", type=int, default=20)
    parser.add_argument(
        "--prefine-use-ess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ESS parameter scaling in bounded QI prefine probes.",
    )
    parser.add_argument("--prefine-ess-alpha", type=float, default=1.2)
    parser.add_argument("--prefine-scipy-lsmr-maxiter", type=int, default=5)
    parser.add_argument(
        "--prefine-fail-on-error",
        action="store_true",
        help="Raise immediately if an explicit prefine probe run fails.",
    )
    parser.add_argument(
        "--prefine-reviewed",
        action="store_true",
        help="Confirm that the prefine manifest/run command has been reviewed before executing probes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.prefine_probes == "run" and not args.prefine_reviewed:
        raise SystemExit("Refusing to run prefine probes without --prefine-reviewed.")
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
        smooth_qi_max=None if args.smooth_qi_max < 0.0 else args.smooth_qi_max,
        legacy_qi_max=None if args.legacy_qi_max < 0.0 else args.legacy_qi_max,
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
        include_bounce_endpoints=args.include_bounce_endpoints,
        mirror_ntheta=mirror_ntheta,
        mirror_nphi=mirror_nphi,
        elongation_ntheta=elongation_ntheta,
        elongation_nphi=elongation_nphi,
        fail_on_error=args.fail_on_error,
        phimin_policy=args.phimin_policy,
    )

    prefine_manifest = None
    if args.prefine_probes != "none":
        prefine_include_bounce_endpoints = (
            args.include_bounce_endpoints
            if args.prefine_include_bounce_endpoints is None
            else args.prefine_include_bounce_endpoints
        )
        prefine_config = QIPrefineProbeConfig(
            top_n=args.prefine_top_n,
            include_family_representatives=args.prefine_family_representatives,
            representative_families=args.prefine_representative_families,
            max_nfev=args.prefine_max_nfev,
            continuation_nfev=args.prefine_continuation_nfev,
            max_mode=args.prefine_max_mode,
            min_vmec_mode=args.prefine_min_vmec_mode,
            stage_modes=args.prefine_stage_modes,
            output_dir=args.prefine_output_dir,
            surfaces=args.prefine_surfaces,
            mboz=args.prefine_mboz,
            nboz=args.prefine_nboz,
            nphi=args.prefine_nphi,
            nalpha=args.prefine_nalpha,
            n_bounce=args.prefine_n_bounce,
            include_bounce_endpoints=prefine_include_bounce_endpoints,
            phimin=args.prefine_phimin,
            qi_ceiling_weight=args.prefine_qi_ceiling_weight,
            qi_ceiling_max=args.prefine_qi_ceiling_max,
            qi_ceiling_smooth_penalty=args.prefine_qi_ceiling_smooth_penalty,
            mirror_weight=args.prefine_mirror_weight,
            mirror_threshold=args.prefine_mirror_threshold,
            mirror_ntheta=args.prefine_mirror_ntheta,
            mirror_nphi=args.prefine_mirror_nphi,
            mirror_surface_index=args.prefine_mirror_surface_index,
            elongation_weight=args.prefine_elongation_weight,
            elongation_threshold=args.prefine_elongation_threshold,
            elongation_ntheta=args.prefine_elongation_ntheta,
            elongation_nphi=args.prefine_elongation_nphi,
            use_ess=args.prefine_use_ess,
            ess_alpha=args.prefine_ess_alpha,
            inner_max_iter=args.prefine_inner_max_iter,
            trial_max_iter=args.prefine_trial_max_iter,
            scipy_lsmr_maxiter=args.prefine_scipy_lsmr_maxiter,
        )
        try:
            prefine_manifest = build_qi_prefine_probe_manifest(
                report,
                config=prefine_config,
                manifest_path=args.prefine_manifest,
                dry_run=args.prefine_probes == "plan",
                reviewed=args.prefine_reviewed,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.prefine_probes == "run":
            prefine_manifest = run_qi_prefine_probe_manifest(
                prefine_manifest,
                fail_on_error=args.prefine_fail_on_error,
                require_review=True,
            )
            report["no_optimization"] = False
        report["prefine_probe_mode"] = args.prefine_probes
        report["prefine_probe_manifest"] = str(args.prefine_manifest)
        report["prefine_probe_summary"] = _prefine_probe_summary(prefine_manifest)

    _write_json(report, args.output)
    if args.csv is not None:
        _write_csv(report["cases"], args.csv)
    if prefine_manifest is not None:
        _write_json(prefine_manifest, args.prefine_manifest)

    print(f"Wrote {args.output} with {len(report['cases'])} seed records.")
    if args.csv is not None:
        print(f"Wrote {args.csv}.")
    if prefine_manifest is not None:
        summary = _prefine_probe_summary(prefine_manifest)
        print(
            f"Wrote {args.prefine_manifest} with {summary['planned_rows']} "
            f"{'planned' if summary['dry_run'] else 'executed'} prefine probes."
        )
    best = report["cases"][0]
    print(
        "Best current seed: "
        f"{best['label']} ({best['family']}), suitability={best['seed_suitability']}, "
        f"smooth={best.get('qi_smooth_total')}, legacy={best.get('qi_legacy_total')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

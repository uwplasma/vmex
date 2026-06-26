#!/usr/bin/env python
"""Fast QI objective-component report for synthetic spectra or existing wouts.

This diagnostic does not run VMEC optimization.  In the default real-case mode
it reads existing ``input`` + ``wout`` pairs, evaluates the vmec_jax smooth QI
objective, the legacy branch/shuffle diagnostic, mirror ratio, LCFS elongation,
and LgradB components, then writes a compact JSON report.

It also retains a synthetic Boozer-spectrum mode for checking local-basin
assumptions before sweeps: single-well topology, the selected one-period
``phimin`` interval, and ranking agreement between smooth and legacy metrics.
"""

from __future__ import annotations

import argparse
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
from vmec_jax.quasi_isodynamic.legacy import legacy_qi_branch_shuffle_diagnostic_from_boozer_output
from vmec_jax.quasi_isodynamic import (
    lgradb_penalty_from_state,
    max_elongation_penalty_from_state,
    mirror_ratio_penalty_from_boozer_output,
    quasi_isodynamic_residual_from_boozer_output,
    quasi_isodynamic_residual_from_state,
)
from vmec_jax.static import build_static
from vmec_jax.wout import read_wout, state_from_wout


enable_x64(True)

DEFAULT_OUTPUT = Path("results/diagnostics/qi_objective_component_report.json")
DEFAULT_CURRENT_INPUT = REPO_ROOT / "examples/data/input.nfp3_QI_fixed_resolution_final"
DEFAULT_CURRENT_WOUT = REPO_ROOT / "examples/data/wout_nfp3_QI_fixed_resolution_final.nc"
DEFAULT_REFERENCE_ROOT = Path(
    os.environ.get("OMNIGENITY_OPTIMIZATION_ROOT", Path.home() / "local" / "omnigenity_optimization")
)
DEFAULT_REFERENCE_INPUT = DEFAULT_REFERENCE_ROOT / "inputs_QI/input.nfp3_QI_fixed_resolution_final"
DEFAULT_REFERENCE_WOUT = DEFAULT_REFERENCE_ROOT / "wouts_QI/wout_nfp3_QI_fixed_resolution_final.nc"


def _booz_like(
    *,
    xm: list[int],
    xn: list[int],
    bmnc: list[float],
    bmns: list[float] | None = None,
    iota: float = 0.4,
    nfp: int = 2,
) -> dict[str, np.ndarray]:
    out = {
        "bmnc_b": np.asarray([bmnc], dtype=float),
        "ixm_b": np.asarray(xm, dtype=float),
        "ixn_b": np.asarray(xn, dtype=float),
        "iota_b": np.asarray([iota], dtype=float),
        "nfp_b": np.asarray(nfp),
    }
    if bmns is not None:
        out["bmns_b"] = np.asarray([bmns], dtype=float)
    return out


def synthetic_cases() -> dict[str, dict[str, np.ndarray]]:
    """Return spectra that exercise QI-like, shifted-QI, and false-positive cases."""

    return {
        "qi_cosine_well": _booz_like(
            xm=[0, 0],
            xn=[0, 2],
            bmnc=[1.0, 0.10],
        ),
        "qi_shifted_well": _booz_like(
            xm=[0, 0],
            xn=[0, 2],
            bmnc=[1.0, -0.10],
        ),
        "helical_qh_like": _booz_like(
            xm=[0, 1],
            xn=[0, 2],
            bmnc=[1.0, 0.10],
        ),
        "mixed_qi_helical": _booz_like(
            xm=[0, 0, 1],
            xn=[0, 2, 2],
            bmnc=[1.0, 0.10, 0.04],
        ),
        "asymmetric_mixed": _booz_like(
            xm=[0, 0, 1, 1],
            xn=[0, 2, 0, 2],
            bmnc=[1.0, 0.08, 0.025, 0.035],
            bmns=[0.0, 0.025, -0.015, 0.020],
        ),
    }


def _component_total(out: dict[str, Any], key: str) -> float:
    values = np.asarray(out.get(key, []), dtype=float).ravel()
    return float(np.dot(values, values))


def _component_size(out: dict[str, Any], key: str) -> int:
    return int(np.asarray(out.get(key, []), dtype=float).size)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return float(arr.ravel()[0])


def _max_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return float(np.max(arr))


def _min_float(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return None
    return float(np.min(arr))


def _fieldline_boozer_grid(
    booz: dict[str, np.ndarray],
    *,
    nphi: int,
    nalpha: int,
    phimin: float,
) -> tuple[np.ndarray, np.ndarray]:
    nfp = int(np.asarray(booz["nfp_b"]).ravel()[0])
    iota = float(np.asarray(booz["iota_b"]).ravel()[0])
    xm = np.asarray(booz["ixm_b"], dtype=float)
    xn = np.asarray(booz["ixn_b"], dtype=float)
    bmnc = np.asarray(booz["bmnc_b"], dtype=float)[0]
    bmns_raw = booz.get("bmns_b")
    bmns = np.zeros_like(bmnc) if bmns_raw is None else np.asarray(bmns_raw, dtype=float)[0]

    phi = np.linspace(float(phimin), float(phimin) + 2.0 * np.pi / nfp, int(nphi), endpoint=True)
    alpha = np.linspace(0.0, 2.0 * np.pi, int(nalpha), endpoint=False)
    theta = alpha[None, :] + iota * phi[:, None]
    angle = theta[:, :, None] * xm[None, None, :] - phi[:, None, None] * xn[None, None, :]
    bmag = np.sum(bmnc[None, None, :] * np.cos(angle) + bmns[None, None, :] * np.sin(angle), axis=-1)
    d_b_d_theta = np.sum(
        -xm[None, None, :] * bmnc[None, None, :] * np.sin(angle)
        + xm[None, None, :] * bmns[None, None, :] * np.cos(angle),
        axis=-1,
    )
    return bmag, d_b_d_theta


def reference_style_proxy(
    booz: dict[str, np.ndarray],
    *,
    nphi: int,
    nalpha: int,
    phimin: float,
    trap_fraction: float = 0.5,
    delta_trap: float = 0.1,
) -> dict[str, float]:
    """Evaluate Boozer-grid analogues of the homotopy QI stage-1 proxies."""

    bmag, d_b_d_theta = _fieldline_boozer_grid(
        booz,
        nphi=nphi,
        nalpha=nalpha,
        phimin=phimin,
    )
    eps = 1.0e-12
    b_ref = max(float(np.mean(bmag)), eps)
    bmin_surface = float(np.min(bmag))
    bmax_surface = float(np.max(bmag))
    span = max(bmax_surface - bmin_surface, eps)
    bcut = bmin_surface + float(trap_fraction) * span
    w_trap = 1.0 / (1.0 + np.exp((bmag - bcut) / (float(delta_trap) * span + eps)))
    sigma_min = 0.20 * span
    w_min = np.exp(-((bmag - bmin_surface) / (sigma_min + eps)) ** 2)

    bmin_line = np.min(bmag, axis=0)
    bmax_line = np.max(bmag, axis=0)
    closure = w_trap * w_min * d_b_d_theta / b_ref
    return {
        "reference_bmin_std_over_alpha": float(np.std(bmin_line) / b_ref),
        "reference_bmax_std_over_alpha": float(np.std(bmax_line) / b_ref),
        "reference_qi_closure_rms": float(np.sqrt(np.mean(closure * closure))),
    }


def _smooth_component_record(smooth: dict[str, Any]) -> dict[str, Any]:
    return {
        "smooth_total": float(np.asarray(smooth["total"])),
        "smooth_width_total": _component_total(smooth, "width_residuals1d"),
        "smooth_branch_width_total": _component_total(smooth, "branch_width_residuals1d"),
        "smooth_profile_total": _component_total(smooth, "profile_residuals1d"),
        "smooth_shuffle_profile_total": _component_total(smooth, "shuffle_profile_residuals1d"),
        "smooth_weighted_shuffle_profile_total": _component_total(
            smooth,
            "weighted_shuffle_profile_residuals1d",
        ),
        "smooth_aligned_profile_total": _component_total(smooth, "aligned_profile_residuals1d"),
        "smooth_residual_size": _component_size(smooth, "residuals1d"),
        "smooth_width_size": _component_size(smooth, "width_residuals1d"),
        "smooth_branch_width_size": _component_size(smooth, "branch_width_residuals1d"),
        "smooth_profile_size": _component_size(smooth, "profile_residuals1d"),
        "smooth_shuffle_profile_size": _component_size(smooth, "shuffle_profile_residuals1d"),
        "smooth_weighted_shuffle_profile_size": _component_size(
            smooth,
            "weighted_shuffle_profile_residuals1d",
        ),
        "smooth_aligned_profile_size": _component_size(smooth, "aligned_profile_residuals1d"),
    }


def _mean_iota_from_wout(wout: Any) -> float | None:
    iotas = getattr(wout, "iotas", None)
    if iotas is None:
        return None
    arr = np.asarray(iotas, dtype=float).ravel()
    if arr.size <= 1:
        return 0.0
    return float(np.mean(arr[1:]))


def _passes_leq(value: Any, limit: float) -> bool:
    if value is None:
        return False
    return float(value) <= float(limit)


def _passes_abs_geq(value: Any, limit: float) -> bool:
    if value is None:
        return False
    return abs(float(value)) >= float(limit)


def _annotate_wout_case_gates(
    row: dict[str, Any],
    *,
    smooth_gate: float,
    legacy_gate: float,
    abs_iota_min: float,
) -> dict[str, Any]:
    """Attach flat promotion-gate fields used for mirror-cleanup ranking."""

    out = dict(row)
    mean_iota = out.get("mean_iota")
    out["abs_mean_iota"] = None if mean_iota is None else abs(float(mean_iota))
    out["qi_smooth_gate_limit"] = float(smooth_gate)
    out["qi_legacy_gate_limit"] = float(legacy_gate)
    out["abs_iota_min"] = float(abs_iota_min)

    smooth_ok = _passes_leq(out.get("smooth_total"), smooth_gate)
    legacy_ok = _passes_leq(out.get("legacy_total"), legacy_gate)
    iota_ok = _passes_abs_geq(mean_iota, abs_iota_min)
    mirror_ok = _passes_leq(out.get("mirror_ratio_max"), out.get("mirror_ratio_target", np.inf))
    elongation_ok = _passes_leq(out.get("max_elongation"), out.get("elongation_target", np.inf))
    qi_iota_ok = smooth_ok and legacy_ok and iota_ok
    engineering_ok = qi_iota_ok and mirror_ok and elongation_ok

    failures = []
    if not smooth_ok:
        failures.append("smooth_qi")
    if not legacy_ok:
        failures.append("legacy_qi")
    if not iota_ok:
        failures.append("abs_iota")
    if not mirror_ok:
        failures.append("mirror_ratio")
    if not elongation_ok:
        failures.append("elongation")

    out.update(
        {
            "qi_smooth_gate_passed": smooth_ok,
            "qi_legacy_gate_passed": legacy_ok,
            "qi_iota_gate_passed": qi_iota_ok,
            "qi_mirror_gate_passed": mirror_ok,
            "qi_elongation_gate_passed": elongation_ok,
            "qi_engineering_gate_passed": engineering_ok,
            "qi_mirror_cleanup_candidate": qi_iota_ok and not engineering_ok,
            "qi_gate_failures": failures,
        }
    )
    return out


def _ranking_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "case",
        "smooth_total",
        "legacy_total",
        "mirror_ratio_max",
        "mirror_excess_max",
        "max_elongation",
        "aspect",
        "mean_iota",
        "abs_mean_iota",
        "qi_iota_gate_passed",
        "qi_engineering_gate_passed",
        "qi_mirror_cleanup_candidate",
        "qi_gate_failures",
    )
    return {key: row.get(key) for key in keys}


def _sort_value(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _wout_rankings(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mirror_cleanup = sorted(
        rows,
        key=lambda row: (
            not bool(row.get("qi_iota_gate_passed")),
            _sort_value(row.get("mirror_ratio_max")),
            _sort_value(row.get("legacy_total")),
            _sort_value(row.get("smooth_total")),
        ),
    )
    engineering = sorted(
        (row for row in rows if bool(row.get("qi_engineering_gate_passed"))),
        key=lambda row: (
            _sort_value(row.get("mirror_ratio_max")),
            _sort_value(row.get("legacy_total")),
            _sort_value(row.get("smooth_total")),
        ),
    )
    qi_iota = sorted(
        (row for row in rows if bool(row.get("qi_iota_gate_passed"))),
        key=lambda row: (
            _sort_value(row.get("legacy_total")),
            _sort_value(row.get("smooth_total")),
            _sort_value(row.get("mirror_ratio_max")),
        ),
    )
    return {
        "mirror_cleanup": [_ranking_row(row) for row in mirror_cleanup],
        "qi_iota_candidates": [_ranking_row(row) for row in qi_iota],
        "engineering_candidates": [_ranking_row(row) for row in engineering],
    }


def evaluate_synthetic_case(
    name: str,
    booz: dict[str, np.ndarray],
    *,
    nphi: int,
    nalpha: int,
    n_bounce: int,
    nphi_out: int,
    phimin_factors: tuple[float, ...] = (0.0, 0.5),
) -> dict[str, Any]:
    nfp = int(np.asarray(booz["nfp_b"]).ravel()[0])
    period = 2.0 * np.pi / nfp
    rows = []
    for factor in phimin_factors:
        phimin = float(factor) * period
        smooth = quasi_isodynamic_residual_from_boozer_output(
            booz,
            nfp=nfp,
            nphi=nphi,
            nalpha=nalpha,
            n_bounce=n_bounce,
            softness=2.0e-2,
            width_weight=1.0,
            branch_width_weight=0.5,
            branch_width_softness=2.0e-2,
            profile_weight=0.1,
            shuffle_profile_weight=1.0,
            shuffle_profile_softness=2.0e-2,
            phimin=phimin,
        )
        legacy = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
            booz,
            nfp=nfp,
            nphi=nphi,
            nalpha=nalpha,
            n_bounce=n_bounce,
            nphi_out=nphi_out,
            phimin=phimin,
        )
        rows.append(
            {
                "phimin_factor": float(factor),
                "phimin": phimin,
                **_smooth_component_record(smooth),
                "legacy_total": float(legacy["total"]),
                **reference_style_proxy(
                    booz,
                    nphi=nphi,
                    nalpha=nalpha,
                    phimin=phimin,
                ),
            }
        )

    best_smooth = min(rows, key=lambda row: row["smooth_total"])
    best_legacy = min(rows, key=lambda row: row["legacy_total"])
    return {
        "case": name,
        "kind": "synthetic_boozer",
        "nfp": nfp,
        "rows": rows,
        "best_smooth_phimin_factor": best_smooth["phimin_factor"],
        "best_legacy_phimin_factor": best_legacy["phimin_factor"],
    }


def _parse_surfaces(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def _parse_case(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--case must have format label:input_path:wout_path")
    label, input_path, wout_path = parts
    if not label:
        raise argparse.ArgumentTypeError("--case label must be non-empty")
    return label, Path(input_path).expanduser(), Path(wout_path).expanduser()


def _default_real_cases() -> list[tuple[str, Path, Path]]:
    cases = [("vmec_jax_nfp3_qi", DEFAULT_CURRENT_INPUT, DEFAULT_CURRENT_WOUT)]
    if DEFAULT_REFERENCE_INPUT.exists() and DEFAULT_REFERENCE_WOUT.exists():
        cases.append(("omnigenity_reference_nfp3_qi", DEFAULT_REFERENCE_INPUT, DEFAULT_REFERENCE_WOUT))
    return cases


def evaluate_wout_case(
    label: str,
    input_path: Path,
    wout_path: Path,
    *,
    surfaces: list[float],
    nphi: int,
    nalpha: int,
    n_bounce: int,
    nphi_out: int,
    mboz: int,
    nboz: int,
    include_bounce_endpoints: bool = False,
    softness: float = 2.0e-2,
    width_weight: float = 1.0,
    branch_width_weight: float = 0.5,
    branch_width_softness: float = 2.0e-2,
    profile_weight: float = 0.1,
    shuffle_profile_weight: float = 1.0,
    shuffle_profile_softness: float = 2.0e-2,
    weighted_shuffle_profile_weight: float = 0.0,
    weighted_shuffle_profile_softness: float = 2.0e-2,
    aligned_profile_weight: float = 0.0,
    aligned_profile_softness: float = 2.0e-2,
    aligned_profile_trap_level: float = 0.65,
    aligned_profile_trap_softness: float = 5.0e-2,
    phimin: float = 0.0,
    mirror_threshold: float = 0.21,
    mirror_ntheta: int = 32,
    mirror_nphi: int = 32,
    elongation_threshold: float = 8.0,
    elongation_ntheta: int = 32,
    elongation_nphi: int = 12,
    lgradb_threshold: float = 0.30,
    lgradb_surface_index: int = -1,
    lgradb_ntheta: int = 8,
    lgradb_nphi: int = 8,
) -> dict[str, Any]:
    """Evaluate all no-solve QI components on a solved VMEC state from wout."""

    input_path = input_path.resolve()
    wout_path = wout_path.resolve()
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

    smooth = quasi_isodynamic_residual_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        surfaces=surfaces,
        mboz=mboz,
        nboz=nboz,
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        include_bounce_endpoints=include_bounce_endpoints,
        softness=softness,
        width_weight=width_weight,
        branch_width_weight=branch_width_weight,
        branch_width_softness=branch_width_softness,
        profile_weight=profile_weight,
        shuffle_profile_weight=shuffle_profile_weight,
        shuffle_profile_softness=shuffle_profile_softness,
        weighted_shuffle_profile_weight=weighted_shuffle_profile_weight,
        weighted_shuffle_profile_softness=weighted_shuffle_profile_softness,
        aligned_profile_weight=aligned_profile_weight,
        aligned_profile_softness=aligned_profile_softness,
        aligned_profile_trap_level=aligned_profile_trap_level,
        aligned_profile_trap_softness=aligned_profile_trap_softness,
        phimin=phimin,
        flux_local=flux,
    )
    booz = smooth["booz"]
    legacy = legacy_qi_branch_shuffle_diagnostic_from_boozer_output(
        booz,
        nfp=int(wout.nfp),
        nphi=nphi,
        nalpha=nalpha,
        n_bounce=n_bounce,
        nphi_out=nphi_out,
        phimin=phimin,
    )
    mirror = mirror_ratio_penalty_from_boozer_output(
        booz,
        nfp=int(wout.nfp),
        threshold=mirror_threshold,
        ntheta=mirror_ntheta,
        nphi=mirror_nphi,
        phimin=phimin,
    )
    elongation = max_elongation_penalty_from_state(
        state=state,
        static=static,
        threshold=elongation_threshold,
        ntheta=elongation_ntheta,
        nphi=elongation_nphi,
    )
    lgradb = lgradb_penalty_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        threshold=lgradb_threshold,
        s_index=lgradb_surface_index,
        ntheta=lgradb_ntheta,
        nphi=lgradb_nphi,
        smooth_penalty=0.0,
        flux_local=flux,
    )

    mirror_ratio = _max_float(mirror.get("mirror_ratio"))
    elongation_max = _as_float(elongation.get("max_elongation"))
    lgradb_min = _min_float(lgradb.get("L_grad_B"))
    lgradb_excess_max = _max_float(lgradb.get("excess"))
    return {
        "case": label,
        "kind": "wout_state",
        "input": str(input_path),
        "wout": str(wout_path),
        "nfp": int(wout.nfp),
        "mpol": int(wout.mpol),
        "ntor": int(wout.ntor),
        "ns": int(wout.ns),
        "signgs": signgs,
        "aspect": _as_float(getattr(wout, "aspect", None)),
        "mean_iota": _mean_iota_from_wout(wout),
        "surfaces": [float(s) for s in surfaces],
        "phimin": float(phimin),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "include_bounce_endpoints": bool(include_bounce_endpoints),
        "softness": float(softness),
        "width_weight": float(width_weight),
        "branch_width_weight": float(branch_width_weight),
        "branch_width_softness": float(branch_width_softness),
        "profile_weight": float(profile_weight),
        "shuffle_profile_weight": float(shuffle_profile_weight),
        "shuffle_profile_softness": float(shuffle_profile_softness),
        "weighted_shuffle_profile_weight": float(weighted_shuffle_profile_weight),
        "weighted_shuffle_profile_softness": float(weighted_shuffle_profile_softness),
        "aligned_profile_weight": float(aligned_profile_weight),
        "aligned_profile_softness": float(aligned_profile_softness),
        "aligned_profile_trap_level": float(aligned_profile_trap_level),
        "aligned_profile_trap_softness": float(aligned_profile_trap_softness),
        **_smooth_component_record(smooth),
        "legacy_total": float(legacy["total"]),
        "legacy_residual_size": int(legacy.get("residual_size", 0)),
        "mirror_total": float(np.asarray(mirror["total"])),
        "mirror_ratio_max": mirror_ratio,
        "mirror_ratio_target": float(mirror_threshold),
        "mirror_excess_max": None if mirror_ratio is None else max(0.0, mirror_ratio - float(mirror_threshold)),
        "elongation_total": float(np.asarray(elongation["total"])),
        "max_elongation": elongation_max,
        "elongation_target": float(elongation_threshold),
        "elongation_excess": None if elongation_max is None else max(0.0, elongation_max - float(elongation_threshold)),
        "lgradb_total": float(np.asarray(lgradb["total"])),
        "lgradb_min": lgradb_min,
        "lgradb_threshold": float(lgradb_threshold),
        "lgradb_excess_max": None if lgradb_excess_max is None else max(0.0, lgradb_excess_max),
    }


def _comparison_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(cases) < 2:
        return []
    base = cases[0]
    rows = []
    keys = (
        "smooth_total",
        "legacy_total",
        "mirror_ratio_max",
        "mirror_excess_max",
        "max_elongation",
        "elongation_excess",
        "lgradb_min",
        "lgradb_excess_max",
    )
    for case in cases[1:]:
        row: dict[str, Any] = {"case": case["case"], "baseline": base["case"]}
        for key in keys:
            lhs = case.get(key)
            rhs = base.get(key)
            if lhs is None or rhs is None:
                continue
            row[f"{key}_delta"] = float(lhs) - float(rhs)
            if abs(float(rhs)) > 0.0:
                row[f"{key}_ratio_to_baseline"] = float(lhs) / float(rhs)
        rows.append(row)
    return rows


def build_synthetic_report(
    *,
    nphi: int = 33,
    nalpha: int = 9,
    n_bounce: int = 7,
    nphi_out: int = 101,
) -> dict[str, Any]:
    cases = [
        evaluate_synthetic_case(
            name,
            booz,
            nphi=nphi,
            nalpha=nalpha,
            n_bounce=n_bounce,
            nphi_out=nphi_out,
        )
        for name, booz in synthetic_cases().items()
    ]
    best_by_smooth = sorted(
        (min(case["rows"], key=lambda row: row["smooth_total"]) | {"case": case["case"]} for case in cases),
        key=lambda row: row["smooth_total"],
    )
    best_by_legacy = sorted(
        (min(case["rows"], key=lambda row: row["legacy_total"]) | {"case": case["case"]} for case in cases),
        key=lambda row: row["legacy_total"],
    )
    return {
        "mode": "synthetic",
        "resolution": {
            "nphi": int(nphi),
            "nalpha": int(nalpha),
            "n_bounce": int(n_bounce),
            "nphi_out": int(nphi_out),
        },
        "cases": cases,
        "rankings": {
            "smooth": [{"case": row["case"], "total": row["smooth_total"]} for row in best_by_smooth],
            "legacy": [{"case": row["case"], "total": row["legacy_total"]} for row in best_by_legacy],
        },
    }


def build_wout_report(
    *,
    cases: list[tuple[str, Path, Path]],
    surfaces: list[float],
    nphi: int,
    nalpha: int,
    n_bounce: int,
    nphi_out: int,
    mboz: int,
    nboz: int,
    include_bounce_endpoints: bool = False,
    softness: float = 2.0e-2,
    width_weight: float = 1.0,
    branch_width_weight: float = 0.5,
    branch_width_softness: float = 2.0e-2,
    profile_weight: float = 0.1,
    shuffle_profile_weight: float = 1.0,
    shuffle_profile_softness: float = 2.0e-2,
    weighted_shuffle_profile_weight: float = 0.0,
    weighted_shuffle_profile_softness: float = 2.0e-2,
    aligned_profile_weight: float = 0.0,
    aligned_profile_softness: float = 2.0e-2,
    aligned_profile_trap_level: float = 0.65,
    aligned_profile_trap_softness: float = 5.0e-2,
    phimin: float = 0.0,
    mirror_threshold: float = 0.21,
    mirror_ntheta: int = 32,
    mirror_nphi: int = 32,
    elongation_threshold: float = 8.0,
    elongation_ntheta: int = 32,
    elongation_nphi: int = 12,
    lgradb_threshold: float = 0.30,
    lgradb_surface_index: int = -1,
    lgradb_ntheta: int = 8,
    lgradb_nphi: int = 8,
    smooth_gate: float = 2.0e-3,
    legacy_gate: float = 2.0e-3,
    abs_iota_min: float = 0.41,
) -> dict[str, Any]:
    rows = []
    for label, input_path, wout_path in cases:
        row = evaluate_wout_case(
            label,
            input_path,
            wout_path,
            surfaces=surfaces,
            nphi=nphi,
            nalpha=nalpha,
            n_bounce=n_bounce,
            nphi_out=nphi_out,
            mboz=mboz,
            nboz=nboz,
            include_bounce_endpoints=include_bounce_endpoints,
            softness=softness,
            width_weight=width_weight,
            branch_width_weight=branch_width_weight,
            branch_width_softness=branch_width_softness,
            profile_weight=profile_weight,
            shuffle_profile_weight=shuffle_profile_weight,
            shuffle_profile_softness=shuffle_profile_softness,
            weighted_shuffle_profile_weight=weighted_shuffle_profile_weight,
            weighted_shuffle_profile_softness=weighted_shuffle_profile_softness,
            aligned_profile_weight=aligned_profile_weight,
            aligned_profile_softness=aligned_profile_softness,
            aligned_profile_trap_level=aligned_profile_trap_level,
            aligned_profile_trap_softness=aligned_profile_trap_softness,
            phimin=phimin,
            mirror_threshold=mirror_threshold,
            mirror_ntheta=mirror_ntheta,
            mirror_nphi=mirror_nphi,
            elongation_threshold=elongation_threshold,
            elongation_ntheta=elongation_ntheta,
            elongation_nphi=elongation_nphi,
            lgradb_threshold=lgradb_threshold,
            lgradb_surface_index=lgradb_surface_index,
            lgradb_ntheta=lgradb_ntheta,
            lgradb_nphi=lgradb_nphi,
        )
        rows.append(
            _annotate_wout_case_gates(
                row,
                smooth_gate=smooth_gate,
                legacy_gate=legacy_gate,
                abs_iota_min=abs_iota_min,
            )
        )
    return {
        "mode": "wout",
        "resolution": {
            "surfaces": [float(s) for s in surfaces],
            "mboz": int(mboz),
            "nboz": int(nboz),
            "nphi": int(nphi),
            "nalpha": int(nalpha),
            "n_bounce": int(n_bounce),
            "nphi_out": int(nphi_out),
            "include_bounce_endpoints": bool(include_bounce_endpoints),
            "mirror_ntheta": int(mirror_ntheta),
            "mirror_nphi": int(mirror_nphi),
            "elongation_ntheta": int(elongation_ntheta),
            "elongation_nphi": int(elongation_nphi),
            "lgradb_ntheta": int(lgradb_ntheta),
            "lgradb_nphi": int(lgradb_nphi),
        },
        "smooth_qi_settings": {
            "softness": float(softness),
            "width_weight": float(width_weight),
            "branch_width_weight": float(branch_width_weight),
            "branch_width_softness": float(branch_width_softness),
            "profile_weight": float(profile_weight),
            "shuffle_profile_weight": float(shuffle_profile_weight),
            "shuffle_profile_softness": float(shuffle_profile_softness),
            "weighted_shuffle_profile_weight": float(weighted_shuffle_profile_weight),
            "weighted_shuffle_profile_softness": float(weighted_shuffle_profile_softness),
            "aligned_profile_weight": float(aligned_profile_weight),
            "aligned_profile_softness": float(aligned_profile_softness),
            "aligned_profile_trap_level": float(aligned_profile_trap_level),
            "aligned_profile_trap_softness": float(aligned_profile_trap_softness),
        },
        "reference_notes": {
            "classic_qi_script": str(DEFAULT_REFERENCE_ROOT / "QI_fixed_resolution.py"),
            "homotopy_qi_script": str(DEFAULT_REFERENCE_ROOT / "homotopy_QI.py"),
            "thresholds": {
                "smooth_qi": float(smooth_gate),
                "legacy_qi": float(legacy_gate),
                "abs_iota_min": float(abs_iota_min),
                "mirror_ratio": float(mirror_threshold),
                "elongation": float(elongation_threshold),
                "lgradb": float(lgradb_threshold),
            },
            "classic_reference_weights": {
                "lgradb_residual_weight": 0.01,
                "smooth_or_legacy_qi_weight": 1.0,
                "mirror_weight": 10.0,
                "elongation_weight": 10.0,
            },
        },
        "cases": rows,
        "comparisons_to_first_case": _comparison_rows(rows),
        "rankings": _wout_rankings(rows),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _print_wout_summary(report: dict[str, Any]) -> None:
    for row in report["cases"]:
        gate = "eng" if row.get("qi_engineering_gate_passed") else "qi" if row.get("qi_iota_gate_passed") else "fail"
        print(
            f"{row['case']}: smooth={row['smooth_total']:.3e} "
            f"legacy={row['legacy_total']:.3e} mirror={row['mirror_ratio_max']:.3f} "
            f"elong={row['max_elongation']:.3f} iota={row['mean_iota']:.3f} "
            f"gate={gate} lgradb_min={row['lgradb_min']:.3e}"
        )
    ranked = report.get("rankings", {}).get("mirror_cleanup", [])
    if ranked:
        labels = ", ".join(row["case"] for row in ranked[:5])
        print(f"mirror-cleanup ranking: {labels}")
    for row in report["comparisons_to_first_case"]:
        print(f"comparison {row['case']} vs {row['baseline']}: {json.dumps(row, sort_keys=True)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic Boozer-spectrum report.")
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        default=[],
        help="Real no-solve case as label:input_path:wout_path. Can be repeated.",
    )
    parser.add_argument("--surfaces", type=_parse_surfaces, default=_parse_surfaces("0.1,0.28,0.46,0.64,0.82,1.0"))
    parser.add_argument("--mboz", type=int, default=8)
    parser.add_argument("--nboz", type=int, default=8)
    parser.add_argument("--nphi", type=int, default=33)
    parser.add_argument("--nalpha", type=int, default=9)
    parser.add_argument("--n-bounce", type=int, default=7)
    parser.add_argument("--nphi-out", type=int, default=101)
    parser.add_argument("--include-bounce-endpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--softness", type=float, default=2.0e-2)
    parser.add_argument("--width-weight", type=float, default=1.0)
    parser.add_argument("--branch-width-weight", type=float, default=0.5)
    parser.add_argument("--branch-width-softness", type=float, default=2.0e-2)
    parser.add_argument("--profile-weight", type=float, default=0.1)
    parser.add_argument("--shuffle-profile-weight", type=float, default=1.0)
    parser.add_argument("--shuffle-profile-softness", type=float, default=2.0e-2)
    parser.add_argument("--weighted-shuffle-profile-weight", type=float, default=0.0)
    parser.add_argument("--weighted-shuffle-profile-softness", type=float, default=2.0e-2)
    parser.add_argument("--aligned-profile-weight", type=float, default=0.0)
    parser.add_argument("--aligned-profile-softness", type=float, default=2.0e-2)
    parser.add_argument("--aligned-profile-trap-level", type=float, default=0.65)
    parser.add_argument("--aligned-profile-trap-softness", type=float, default=5.0e-2)
    parser.add_argument("--phimin", type=float, default=0.0)
    parser.add_argument("--smooth-gate", type=float, default=2.0e-3)
    parser.add_argument("--legacy-gate", type=float, default=2.0e-3)
    parser.add_argument("--abs-iota-min", type=float, default=0.41)
    parser.add_argument("--mirror-threshold", type=float, default=0.21)
    parser.add_argument("--mirror-ntheta", type=int, default=32)
    parser.add_argument("--mirror-nphi", type=int, default=32)
    parser.add_argument("--elongation-threshold", type=float, default=8.0)
    parser.add_argument("--elongation-ntheta", type=int, default=32)
    parser.add_argument("--elongation-nphi", type=int, default=12)
    parser.add_argument("--lgradb-threshold", type=float, default=0.30)
    parser.add_argument("--lgradb-surface-index", type=int, default=-1)
    parser.add_argument("--lgradb-ntheta", type=int, default=8)
    parser.add_argument("--lgradb-nphi", type=int, default=8)
    args = parser.parse_args(argv)

    if args.synthetic:
        report = build_synthetic_report(
            nphi=args.nphi,
            nalpha=args.nalpha,
            n_bounce=args.n_bounce,
            nphi_out=args.nphi_out,
        )
    else:
        cases = list(args.case) if args.case else _default_real_cases()
        report = build_wout_report(
            cases=cases,
            surfaces=args.surfaces,
            nphi=args.nphi,
            nalpha=args.nalpha,
            n_bounce=args.n_bounce,
            nphi_out=args.nphi_out,
            mboz=args.mboz,
            nboz=args.nboz,
            include_bounce_endpoints=args.include_bounce_endpoints,
            softness=args.softness,
            width_weight=args.width_weight,
            branch_width_weight=args.branch_width_weight,
            branch_width_softness=args.branch_width_softness,
            profile_weight=args.profile_weight,
            shuffle_profile_weight=args.shuffle_profile_weight,
            shuffle_profile_softness=args.shuffle_profile_softness,
            weighted_shuffle_profile_weight=args.weighted_shuffle_profile_weight,
            weighted_shuffle_profile_softness=args.weighted_shuffle_profile_softness,
            aligned_profile_weight=args.aligned_profile_weight,
            aligned_profile_softness=args.aligned_profile_softness,
            aligned_profile_trap_level=args.aligned_profile_trap_level,
            aligned_profile_trap_softness=args.aligned_profile_trap_softness,
            phimin=args.phimin,
            mirror_threshold=args.mirror_threshold,
            mirror_ntheta=args.mirror_ntheta,
            mirror_nphi=args.mirror_nphi,
            elongation_threshold=args.elongation_threshold,
            elongation_ntheta=args.elongation_ntheta,
            elongation_nphi=args.elongation_nphi,
            lgradb_threshold=args.lgradb_threshold,
            lgradb_surface_index=args.lgradb_surface_index,
            lgradb_ntheta=args.lgradb_ntheta,
            lgradb_nphi=args.lgradb_nphi,
            smooth_gate=args.smooth_gate,
            legacy_gate=args.legacy_gate,
            abs_iota_min=args.abs_iota_min,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print(f"wrote {args.output}")
    if args.synthetic:
        print("smooth ranking:", ", ".join(row["case"] for row in report["rankings"]["smooth"]))
        print("legacy ranking:", ", ".join(row["case"] for row in report["rankings"]["legacy"]))
    else:
        _print_wout_summary(report)


if __name__ == "__main__":
    main()

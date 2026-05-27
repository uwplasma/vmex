#!/usr/bin/env python
"""Run bounded common-minimal-seed optimization showcase cases.

The README/docs showcase uses the same deliberately simple VMEC seed template
for each field period count:

- ``RBC(0,0)``
- ``RBC(0,1)``
- ``ZBS(0,1)``

This script maps that seed family to the configured targets.  Publication
claims should use the renderer output, which skips stale rows and may include
only the subset of cases with current non-stale provenance:

- QI with NFP=1, 2, 3, and a finite-beta NFP=4 reference lane
- an NFP=1 circular-torus QI stress lane
- QA with NFP=2 and 3
- QH with NFP=3 and 4
- QP with NFP=1, 2, 3, and 4

The implementation intentionally reuses ``generate_qs_ess_sweep.py`` so the
showcase follows the same exact optimizer, diagnostics, and per-case output
format as the larger benchmark matrix.  Defaults are bounded enough for local
representative runs; increase ``--max-nfev`` and ``--continuation-nfev`` for
publication-quality panels.

Examples:

  # One quick representative case.
  python examples/optimization/generate_minimal_seed_showcase.py --cases qa_nfp2 --max-nfev 2 --continuation-nfev 2

  # Full aspect-5 README/docs production lane.  Use the CUDA variant on
  # production GPU hosts; use cpu/cpu/cpu for a slower local reproduction.
  PYTHONPATH=. JAX_PLATFORMS=cuda python3 examples/optimization/generate_minimal_seed_showcase.py \\
    --cases qa_nfp2,qa_nfp3,qh_nfp3,qh_nfp4,qp_nfp2,qp_nfp3,qp_nfp4,qi_nfp1,qi_nfp2,qi_nfp3,qi_nfp4 \\
    --backend-label gpu --solver-device gpu --worker-jax-platforms cuda \\
    --policy continuation --max-mode 5 --ess on \\
    --max-nfev 60 --continuation-nfev 20 \\
    --inner-max-iter 550 --inner-ftol 1e-10 \\
    --trial-max-iter 550 --trial-ftol 1e-10 \\
    --ess-alpha 1.2 --case-timeout-s 7200 --rerun

  # Render completed cases.
  python examples/optimization/render_minimal_seed_showcase.py --publication-matrix
"""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "examples" / "data"
OUTPUT_ROOT = SCRIPT_DIR / "results" / "minimal_seed_showcase"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_qs_ess_sweep as sweep
from qi_optimization_cases import QI_CASES
import qi_staged_runner
from vmec_jax.namelist import InData, read_indata, write_indata


@dataclass(frozen=True)
class MinimalSeedCase:
    """One target problem started from the common minimal boundary template."""

    name: str
    problem: str
    nfp: int
    input_file: Path
    qi_qp_preseed: bool | None = None
    qi_jit_booz: bool | None = None
    qi_policy_case: str | None = None
    qi_reference_input: Path | None = None
    reference_preseed_input: Path | None = None
    reference_preseed_blend: float = 0.0


SHOWCASE_CASES: dict[str, MinimalSeedCase] = {
    "qi_nfp1": MinimalSeedCase(
        name="qi_nfp1",
        problem="qi",
        nfp=1,
        input_file=DATA_DIR / "input.minimal_seed_nfp1",
        qi_qp_preseed=False,
        qi_jit_booz=True,
        qi_policy_case="minimal_nfp1_qi",
        qi_reference_input=DATA_DIR / "input.nfp1_QI",
    ),
    "qi_nfp2": MinimalSeedCase(
        name="qi_nfp2",
        problem="qi",
        nfp=2,
        input_file=DATA_DIR / "input.minimal_seed_nfp2",
        qi_qp_preseed=False,
        qi_jit_booz=True,
        qi_policy_case="minimal_nfp2_qi",
        qi_reference_input=DATA_DIR / "input.nfp2_QI",
    ),
    "qi_nfp3": MinimalSeedCase(
        name="qi_nfp3",
        problem="qi",
        nfp=3,
        input_file=DATA_DIR / "input.minimal_seed_nfp3",
        qi_qp_preseed=False,
        qi_jit_booz=True,
        qi_policy_case="minimal_nfp3_qi",
        qi_reference_input=DATA_DIR / "input.nfp3_QI_fixed_resolution_final",
    ),
    "qi_nfp4": MinimalSeedCase(
        name="qi_nfp4",
        problem="qi",
        nfp=4,
        input_file=DATA_DIR / "input.minimal_seed_nfp4",
        qi_qp_preseed=False,
        qi_jit_booz=True,
        qi_policy_case="minimal_nfp4_qi",
        qi_reference_input=DATA_DIR / "input.nfp4_QI_finite_beta",
    ),
    "qi_circular_nfp1": MinimalSeedCase(
        name="qi_circular_nfp1",
        problem="qi",
        nfp=1,
        input_file=DATA_DIR / "input.circular_tokamak",
        qi_qp_preseed=False,
        qi_jit_booz=True,
        qi_policy_case="circular_nfp1_qi",
        qi_reference_input=DATA_DIR / "input.nfp1_QI",
    ),
    "qa_nfp2": MinimalSeedCase(
        name="qa_nfp2",
        problem="qa",
        nfp=2,
        input_file=DATA_DIR / "input.minimal_seed_nfp2",
        reference_preseed_input=DATA_DIR / "input.nfp2_QA_omnigenity",
        reference_preseed_blend=0.25,
    ),
    "qa_nfp3": MinimalSeedCase(
        name="qa_nfp3",
        problem="qa",
        nfp=3,
        input_file=DATA_DIR / "input.minimal_seed_nfp3",
    ),
    "qh_nfp3": MinimalSeedCase(
        name="qh_nfp3",
        problem="qh",
        nfp=3,
        input_file=DATA_DIR / "input.minimal_seed_nfp3",
    ),
    "qh_nfp4": MinimalSeedCase(
        name="qh_nfp4",
        problem="qh",
        nfp=4,
        input_file=DATA_DIR / "input.minimal_seed_nfp4",
    ),
    "qp_nfp1": MinimalSeedCase(
        name="qp_nfp1",
        problem="qp",
        nfp=1,
        input_file=DATA_DIR / "input.minimal_seed_nfp1",
        reference_preseed_input=DATA_DIR / "input.nfp1_QI",
        reference_preseed_blend=0.10,
    ),
    "qp_nfp2": MinimalSeedCase(
        name="qp_nfp2",
        problem="qp",
        nfp=2,
        input_file=DATA_DIR / "input.minimal_seed_nfp2",
        reference_preseed_input=DATA_DIR / "input.nfp2_QI",
        reference_preseed_blend=0.10,
    ),
    "qp_nfp3": MinimalSeedCase(
        name="qp_nfp3",
        problem="qp",
        nfp=3,
        input_file=DATA_DIR / "input.minimal_seed_nfp3",
        reference_preseed_input=DATA_DIR / "input.nfp3_QI_fixed_resolution_final",
        reference_preseed_blend=0.10,
    ),
    "qp_nfp4": MinimalSeedCase(
        name="qp_nfp4",
        problem="qp",
        nfp=4,
        input_file=DATA_DIR / "input.minimal_seed_nfp4",
        reference_preseed_input=DATA_DIR / "input.nfp4_QI_finite_beta",
        reference_preseed_blend=0.10,
    ),
}

DEFAULT_CASE_ORDER = (
    "qi_circular_nfp1",
    "qi_nfp1",
    "qi_nfp2",
    "qi_nfp3",
    "qi_nfp4",
    "qa_nfp2",
    "qa_nfp3",
    "qh_nfp3",
    "qh_nfp4",
    "qp_nfp1",
    "qp_nfp2",
    "qp_nfp3",
    "qp_nfp4",
)

PHYSICS_IOTA_FLOOR = 0.35
PHYSICS_QA_IOTA_TARGET = 0.42
PHYSICS_QA_IOTA_TOL = 0.08
PHYSICS_QI_LEGACY_MAX = 2.0e-3
PHYSICS_QI_MIRROR_MAX = 0.40
PHYSICS_QI_ELONGATION_MAX = 12.0
TARGET_HELICITY_SEED_AMPLITUDE = 1.0e-3
TARGET_HELICITY_SEED_MODE_TERMS = (
    ("RBC", (1, 0)),
    ("ZBS", (1, 0)),
    ("RBC", (-1, 1)),
    ("ZBS", (-1, 1)),
    ("RBC", (1, 1)),
    ("ZBS", (1, 1)),
)
REFERENCE_PRESEED_FAMILIES = ("RBC", "ZBS")


@dataclass(frozen=True)
class MinimalSeedBudget:
    """Bounded optimizer/VMEC budgets for a showcase run."""

    max_nfev: int
    continuation_nfev: int
    inner_max_iter: int
    inner_ftol: float
    trial_max_iter: int
    trial_ftol: float
    ess_alpha: float = 1.2


def _parse_case_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not names or names == ("all",):
        return DEFAULT_CASE_ORDER
    unknown = sorted(set(names) - set(SHOWCASE_CASES))
    if unknown:
        known = ", ".join(DEFAULT_CASE_ORDER)
        raise ValueError(f"Unknown minimal-seed case(s): {', '.join(unknown)}. Known cases: {known}")
    return names


def _bool_from_choice(value: str) -> bool:
    return str(value).strip().lower() in {"on", "true", "1", "yes"}


def _target_helicity_seed_terms(
    *,
    max_mode: int,
    amplitude: float = TARGET_HELICITY_SEED_AMPLITUDE,
) -> tuple[tuple[str, tuple[int, int], float], ...]:
    """Small deterministic boundary terms that avoid the zero-transform branch.

    The common minimal seed intentionally contains only ``RBC(0,0)``,
    ``RBC(0,1)``, and ``ZBS(0,1)``.  For QA/QH/QP/QI stress tests this can leave
    target-helicity derivatives exactly zero at the first optimization point.
    Adding the same tiny mode-1 perturbations used in the QA example keeps the
    run in the intended differentiable basin without changing the source input
    fixtures.
    """

    if float(amplitude) == 0.0 or int(max_mode) < 1:
        return ()
    return tuple(
        (name, index, float(amplitude))
        for name, index in TARGET_HELICITY_SEED_MODE_TERMS
        if max(abs(int(index[0])), abs(int(index[1]))) <= int(max_mode)
    )


def _write_target_helicity_seeded_input(
    source_file: Path,
    output_dir: Path,
    *,
    max_mode: int,
    amplitude: float = TARGET_HELICITY_SEED_AMPLITUDE,
) -> tuple[Path, tuple[tuple[str, tuple[int, int], float], ...]]:
    """Write the per-run seeded input deck and return inserted coefficients."""

    terms = _target_helicity_seed_terms(max_mode=max_mode, amplitude=amplitude)
    if not terms:
        return Path(source_file), ()
    source = read_indata(source_file)
    indexed = {name: dict(values) for name, values in source.indexed.items()}
    inserted: list[tuple[str, tuple[int, int], float]] = []
    for name, index, value in terms:
        coeffs = indexed.setdefault(name, {})
        existing = coeffs.get(index)
        try:
            existing_abs = abs(float(existing)) if existing is not None else 0.0
        except (TypeError, ValueError):
            existing_abs = 0.0
        if existing_abs == 0.0:
            coeffs[index] = float(value)
            inserted.append((name, index, float(value)))

    seeded = InData(scalars=dict(source.scalars), indexed=indexed, source_path=str(source_file))
    output_path = Path(output_dir) / "input.target_helicity_seed"
    write_indata(output_path, seeded)
    return output_path, tuple(inserted)


def _write_reference_preseeded_input(
    source_file: Path,
    reference_file: Path | None,
    output_dir: Path,
    *,
    max_mode: int,
    blend: float,
) -> tuple[Path, list[dict[str, Any]]]:
    """Blend low-order active-space boundary modes toward a same-NFP reference family.

    The raw minimal seed remains untouched.  This is an optimization-time
    preconditioner for the zero-transform branch: a small blend toward a known
    target family gives the iota residual a usable derivative before the local
    exact optimizer starts.  Only RBC/ZBS coefficients inside the active
    ``max_mode`` space are blended or inserted, and ``RBC(0,0)`` is left fixed.
    """

    if reference_file is None or float(blend) == 0.0:
        return Path(source_file), []
    source = read_indata(source_file)
    reference = read_indata(reference_file)
    if int(source.scalars.get("NFP", 0)) != int(reference.scalars.get("NFP", 0)):
        raise ValueError(
            f"Reference preseed NFP mismatch: {source_file} has NFP={source.scalars.get('NFP')}, "
            f"{reference_file} has NFP={reference.scalars.get('NFP')}"
        )
    indexed = {name: dict(values) for name, values in source.indexed.items()}
    changes: list[dict[str, Any]] = []
    alpha = float(blend)
    for family in REFERENCE_PRESEED_FAMILIES:
        coeffs = indexed.setdefault(family, {})
        for index, ref_value in sorted(reference.indexed.get(family, {}).items()):
            n_i, m_i = (int(index[0]), int(index[1]))
            if family == "RBC" and (n_i, m_i) == (0, 0):
                continue
            if max(abs(n_i), abs(m_i)) > int(max_mode):
                continue
            old_value = float(coeffs.get((n_i, m_i), 0.0))
            new_value = old_value + alpha * (float(ref_value) - old_value)
            coeffs[(n_i, m_i)] = new_value
            if abs(new_value - old_value) > 0.0:
                changes.append(
                    {
                        "family": family,
                        "n": n_i,
                        "m": m_i,
                        "old": old_value,
                        "reference": float(ref_value),
                        "new": new_value,
                    }
                )

    preseeded = InData(scalars=dict(source.scalars), indexed=indexed, source_path=str(source_file))
    output_path = Path(output_dir) / "input.reference_preseed"
    write_indata(output_path, preseeded)
    return output_path, changes


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _physics_gate_failures(case: MinimalSeedCase, result: sweep.CaseResult) -> list[str]:
    """Return failed promotion gates for the minimal-seed stress-test lane."""

    failures: list[str] = []
    iota = _finite_float(result.iota_final)
    if case.problem == "qa":
        if iota is None or abs(iota - PHYSICS_QA_IOTA_TARGET) > PHYSICS_QA_IOTA_TOL:
            failures.append(f"iota={iota!r} outside {PHYSICS_QA_IOTA_TARGET:.2f}+/-{PHYSICS_QA_IOTA_TOL:.2f}")
    else:
        if iota is None or abs(iota) < PHYSICS_IOTA_FLOOR:
            abs_iota = None if iota is None else abs(iota)
            failures.append(f"|iota|={abs_iota!r} below {PHYSICS_IOTA_FLOOR:.2f}")

    if case.problem == "qi":
        qi_legacy = _finite_float(result.qi_legacy_total)
        mirror = _finite_float(result.qi_mirror_ratio_max)
        elongation = _finite_float(result.qi_max_elongation)
        if qi_legacy is None or qi_legacy > PHYSICS_QI_LEGACY_MAX:
            failures.append(f"legacy QI={qi_legacy!r} above {PHYSICS_QI_LEGACY_MAX:.1e}")
        if mirror is None or mirror > PHYSICS_QI_MIRROR_MAX:
            failures.append(f"mirror={mirror!r} above {PHYSICS_QI_MIRROR_MAX:.2f}")
        if elongation is None or elongation > PHYSICS_QI_ELONGATION_MAX:
            failures.append(f"elongation={elongation!r} above {PHYSICS_QI_ELONGATION_MAX:.1f}")
    return failures


def _apply_physics_gate(case: MinimalSeedCase, result: sweep.CaseResult) -> bool:
    """Mark optimizer-success records failed when they miss stress-test physics gates."""

    if not result.success or result.crashed:
        return False
    failures = _physics_gate_failures(case, result)
    if not failures:
        return False
    result.success = False
    message = "; ".join(failures)
    if result.message:
        result.message = f"{result.message}; physics gate failed: {message}"
    else:
        result.message = f"physics gate failed: {message}"
    return True


def _case_output_dir(
    output_root: Path,
    *,
    case: MinimalSeedCase,
    backend_label: str,
    policy: str,
    max_mode: int,
    use_ess: bool,
) -> Path:
    qi_part = ""
    if case.problem == "qi":
        qi_part = case.qi_policy_case or ("qp_preseed" if bool(case.qi_qp_preseed) else "no_qp_preseed")
    parts = [output_root, backend_label, case.name, policy]
    if qi_part:
        parts.append(qi_part)
    parts.extend([f"mode{int(max_mode)}", sweep._ess_label(bool(use_ess))])
    path = Path(parts[0])
    for part in parts[1:]:
        path = path / str(part)
    return path


def _problem_config_for_case(
    case: MinimalSeedCase,
    *,
    max_mode: int,
    budget: MinimalSeedBudget,
    input_file: Path | None = None,
) -> sweep.ProblemConfig:
    """Return a sweep config patched to use the common minimal seed."""

    base = sweep.PROBLEM_CONFIGS[case.problem]
    min_vmec_mode = max(5, int(max_mode) + 2, int(base.min_vmec_mode))
    updates = {
        "input_file": Path(input_file) if input_file is not None else case.input_file,
        "max_nfev": int(budget.max_nfev),
        "continuation_nfev": int(budget.continuation_nfev),
        "inner_max_iter": int(budget.inner_max_iter),
        "inner_ftol": float(budget.inner_ftol),
        "trial_max_iter": int(budget.trial_max_iter),
        "trial_ftol": float(budget.trial_ftol),
        "ess_alpha": float(budget.ess_alpha),
        "project_input_boundary_to_max_mode": True,
        "min_vmec_mode": min_vmec_mode,
    }
    if case.problem == "qi":
        updates.update(
            qi_preseed_qp=bool(case.qi_qp_preseed),
            qi_jit_booz=True if case.qi_jit_booz is None else bool(case.qi_jit_booz),
        )
    return replace(base, **updates)


def _qp_preseed_config_for_qi_case(
    case: MinimalSeedCase,
    *,
    max_mode: int,
    budget: MinimalSeedBudget,
    input_file: Path | None = None,
) -> sweep.ProblemConfig:
    """Return a QP preseed config that uses the same NFP/minimal input as QI."""

    base = sweep.PROBLEM_CONFIGS["qp"]
    min_vmec_mode = max(5, int(max_mode) + 2, int(base.min_vmec_mode))
    return replace(
        base,
        input_file=Path(input_file) if input_file is not None else case.input_file,
        max_nfev=int(budget.max_nfev),
        continuation_nfev=int(budget.continuation_nfev),
        inner_max_iter=int(budget.inner_max_iter),
        inner_ftol=float(budget.inner_ftol),
        trial_max_iter=int(budget.trial_max_iter),
        trial_ftol=float(budget.trial_ftol),
        ess_alpha=float(budget.ess_alpha),
        project_input_boundary_to_max_mode=True,
        min_vmec_mode=min_vmec_mode,
    )


def _patch_qi_stage_budget(stage: dict[str, Any], *, budget: MinimalSeedBudget, max_mode: int) -> dict[str, Any]:
    """Return a QI stage dictionary with showcase budgets applied.

    The QI catalog contains reviewed stage policies.  The showcase command line
    still owns the actual budget, so stages that opt in with
    ``use_showcase_max_nfev`` inherit the requested production budget instead
    of their quick local default.
    """

    out = dict(stage)
    if bool(out.pop("use_showcase_max_nfev", False)):
        out["max_nfev"] = int(budget.max_nfev)
    if bool(out.pop("use_showcase_max_mode", False)) and "stage_mode_limits" not in out:
        out["stage_modes"] = (int(max_mode),)
        out["use_mode_continuation"] = False
    return out


def _write_showcase_metadata(
    output_dir: Path,
    *,
    case: MinimalSeedCase,
    policy: str,
    max_mode: int,
    use_ess: bool,
    budget: MinimalSeedBudget,
    seeded_input_file: Path | None = None,
    seed_terms: tuple[tuple[str, tuple[int, int], float], ...] = (),
    seed_amplitude: float = TARGET_HELICITY_SEED_AMPLITUDE,
    reference_preseeded_input_file: Path | None = None,
    reference_preseed_changes: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_metadata = asdict(case)
    case_metadata["input_file"] = str(case.input_file)
    if case.qi_reference_input is not None:
        case_metadata["qi_reference_input"] = str(case.qi_reference_input)
    if case.reference_preseed_input is not None:
        case_metadata["reference_preseed_input"] = str(case.reference_preseed_input)
    metadata = {
        "minimal_seed_case": case_metadata,
        "policy": str(policy),
        "max_mode": int(max_mode),
        "use_ess": bool(use_ess),
        "budget": asdict(budget),
        "reference_preseed": {
            "enabled": bool(case.reference_preseed_input is not None and float(case.reference_preseed_blend) != 0.0),
            "blend": float(case.reference_preseed_blend),
            "reference_input": None
            if case.reference_preseed_input is None
            else str(case.reference_preseed_input),
            "preseeded_input_file": None
            if reference_preseeded_input_file is None
            else str(reference_preseeded_input_file),
            "changes": list(reference_preseed_changes or []),
        },
        "target_helicity_seed": {
            "enabled": bool(seed_terms),
            "amplitude": float(seed_amplitude),
            "seeded_input_file": None if seeded_input_file is None else str(seeded_input_file),
            "terms": [
                {"family": name, "n": int(index[0]), "m": int(index[1]), "value": float(value)}
                for name, index, value in seed_terms
            ],
        },
    }
    (output_dir / "showcase_case.json").write_text(json.dumps(metadata, indent=2))


def _run_showcase_case(
    case: MinimalSeedCase,
    output_dir: Path,
    *,
    backend_label: str,
    solver_device: str | None,
    worker_jax_platforms: str | None,
    policy: str,
    max_mode: int,
    use_ess: bool,
    budget: MinimalSeedBudget,
    input_file: Path | None = None,
    case_timeout_s: float | None = None,
) -> sweep.CaseResult:
    """Run one minimal-seed case with temporary sweep config overrides."""

    if case.problem == "qi":
        qi_policy = QI_CASES.get(case.qi_policy_case or "qi_stel_seed_3127", {})
        mirror_ramp_stages = tuple(
            _patch_qi_stage_budget(stage, budget=budget, max_mode=max_mode)
            for stage in qi_policy.get("mirror_ramp_stages", ())
        )
        boundary_reference = qi_policy.get("boundary_reference_preconditioner", {})
        # Give the inner QI subprocess room to catch TimeoutExpired, harvest
        # partial artifacts, and let the outer worker write case_result.json.
        qi_timeout_s = None
        if case_timeout_s not in (None, 0):
            qi_timeout_s = max(1.0, min(0.95 * float(case_timeout_s), float(case_timeout_s) - 30.0))
        return qi_staged_runner.run_qi_staged_case(
            qi_staged_runner.QIStagedCaseConfig(
                name=case.name,
                input_file=Path(input_file) if input_file is not None else case.input_file,
                output_dir=output_dir,
                max_mode=int(max_mode),
                policy=str(policy),
                policy_case=case.qi_policy_case or "qi_stel_seed_3127",
                reference_input=case.qi_reference_input,
                reference_accept_as_baseline=bool(boundary_reference.get("accept_as_baseline", False)),
                backend_label=str(backend_label),
                solver_device=solver_device,
                worker_jax_platforms=worker_jax_platforms,
                use_ess=bool(use_ess),
                max_nfev=int(budget.max_nfev),
                continuation_nfev=int(budget.continuation_nfev),
                inner_max_iter=int(budget.inner_max_iter),
                inner_ftol=float(budget.inner_ftol),
                trial_max_iter=int(budget.trial_max_iter),
                trial_ftol=float(budget.trial_ftol),
                ess_alpha=float(budget.ess_alpha),
                target_aspect=sweep.TARGET_ASPECT,
                target_abs_iota_min=0.41,
                max_mirror_ratio=0.30,
                max_elongation=10.0,
                mirror_ramp_stages=mirror_ramp_stages,
                make_plots=False,
                timeout_s=qi_timeout_s,
            )
        )

    old_configs = dict(sweep.PROBLEM_CONFIGS)
    sweep.PROBLEM_CONFIGS[case.problem] = _problem_config_for_case(
        case,
        max_mode=max_mode,
        budget=budget,
        input_file=input_file,
    )
    if case.problem == "qi":
        sweep.PROBLEM_CONFIGS["qp"] = _qp_preseed_config_for_qi_case(
            case,
            max_mode=max_mode,
            budget=budget,
            input_file=input_file,
        )
    try:
        result = sweep._run_case(
            case.problem,
            int(max_mode),
            bool(use_ess),
            output_dir,
            use_mode_continuation=str(policy) == "continuation",
            policy=str(policy),
            backend=str(backend_label),
            solver_device=solver_device,
            jax_platforms=worker_jax_platforms,
            diagnostic_budgets=False,
            stellarator_asymmetric=False,
            qi_qp_preseed=case.qi_qp_preseed if case.problem == "qi" else None,
            qi_jit_booz=case.qi_jit_booz if case.problem == "qi" else None,
        )
    finally:
        sweep.PROBLEM_CONFIGS.clear()
        sweep.PROBLEM_CONFIGS.update(old_configs)
    return result


def _worker(
    case_name: str,
    output_dir_str: str,
    result_path_str: str,
    backend_label: str,
    solver_device: str | None,
    worker_jax_platforms: str | None,
    policy: str,
    max_mode: int,
    use_ess: bool,
    budget_dict: dict,
    target_helicity_seed_amplitude: float,
    reference_preseed_blend: float | None = None,
    case_timeout_s: float | None = None,
) -> None:
    output_dir = Path(output_dir_str)
    result_path = Path(result_path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep._start_worker_session()

    # Keep per-case logs even if the worker dies before writing case_result.json.
    # This is especially useful for JAX/XLA crashes, OOM kills, and timeout
    # diagnostics in the full production run.
    stdout_path = output_dir / "worker_stdout.log"
    stderr_path = output_dir / "worker_stderr.log"
    with stdout_path.open("a", buffering=1) as stdout, stderr_path.open("a", buffering=1) as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            _worker_impl(
                case_name,
                output_dir,
                result_path,
                backend_label,
                solver_device,
                worker_jax_platforms,
                policy,
                max_mode,
                use_ess,
                budget_dict,
                target_helicity_seed_amplitude,
                reference_preseed_blend,
                case_timeout_s,
            )


def _failure_result(
    case: MinimalSeedCase,
    output_dir: Path,
    *,
    backend_label: str,
    solver_device: str | None,
    worker_jax_platforms: str | None,
    policy: str,
    max_mode: int,
    use_ess: bool,
    message: str,
    total_wall_time_s: float | None = None,
    input_file: Path | None = None,
) -> sweep.CaseResult:
    """Create a consistent failed result record for worker errors/timeouts."""

    return sweep.CaseResult(
        backend=str(backend_label),
        problem=case.problem,
        max_mode=int(max_mode),
        use_ess=bool(use_ess),
        success=False,
        crashed=True,
        message=str(message),
        policy=str(policy),
        total_wall_time_s=total_wall_time_s,
        output_dir=str(output_dir),
        solver_device=solver_device,
        jax_platforms=worker_jax_platforms,
        input_file=str(input_file or case.input_file),
        input_nfp=int(case.nfp),
        qi_qp_preseed=case.qi_qp_preseed if case.problem == "qi" else None,
        qi_jit_booz=case.qi_jit_booz if case.problem == "qi" else None,
    )


def _annotate_qi_partial_result(
    case: MinimalSeedCase,
    result: sweep.CaseResult,
    output_dir: Path,
) -> bool:
    """Attach QI checkpoint metrics and mark timeout partials as non-crash failures."""

    if case.problem != "qi":
        return False
    changed = qi_staged_runner.annotate_case_result_from_partial_artifacts(result, output_dir)
    message = str(result.message or "").lower()
    timeout_partial = changed and "partial" in message and ("timeout" in message or "timed out" in message)
    if timeout_partial and result.crashed:
        result.crashed = False
        changed = True
    return changed


def _worker_impl(
    case_name: str,
    output_dir: Path,
    result_path: Path,
    backend_label: str,
    solver_device: str | None,
    worker_jax_platforms: str | None,
    policy: str,
    max_mode: int,
    use_ess: bool,
    budget_dict: dict,
    target_helicity_seed_amplitude: float,
    reference_preseed_blend: float | None = None,
    case_timeout_s: float | None = None,
) -> None:
    case = SHOWCASE_CASES[case_name]
    if reference_preseed_blend is not None and case.reference_preseed_input is not None:
        case = replace(case, reference_preseed_blend=float(reference_preseed_blend))
    budget = MinimalSeedBudget(**budget_dict)
    reference_preseeded_input_file: Path | None = None
    reference_preseed_changes: list[dict[str, Any]] = []
    seeded_input_file: Path | None = None
    seed_terms: tuple[tuple[str, tuple[int, int], float], ...] = ()
    try:
        reference_preseeded_input_file, reference_preseed_changes = _write_reference_preseeded_input(
            case.input_file,
            case.reference_preseed_input,
            output_dir,
            max_mode=int(max_mode),
            blend=float(case.reference_preseed_blend),
        )
        seeded_input_file, seed_terms = _write_target_helicity_seeded_input(
            reference_preseeded_input_file,
            output_dir,
            max_mode=int(max_mode),
            amplitude=float(target_helicity_seed_amplitude),
        )
        _write_showcase_metadata(
            output_dir,
            case=case,
            policy=policy,
            max_mode=max_mode,
            use_ess=use_ess,
            budget=budget,
            seeded_input_file=seeded_input_file,
            seed_terms=seed_terms,
            seed_amplitude=float(target_helicity_seed_amplitude),
            reference_preseeded_input_file=reference_preseeded_input_file,
            reference_preseed_changes=reference_preseed_changes,
        )
        result = _run_showcase_case(
            case,
            output_dir,
            backend_label=backend_label,
            solver_device=solver_device,
            worker_jax_platforms=worker_jax_platforms,
            policy=policy,
            max_mode=max_mode,
            use_ess=use_ess,
            budget=budget,
            input_file=seeded_input_file,
            case_timeout_s=case_timeout_s,
        )
        _annotate_qi_partial_result(case, result, output_dir)
        _apply_physics_gate(case, result)
    except Exception as exc:  # pragma: no cover - exercised by integration failures.
        (output_dir / "traceback.txt").write_text(traceback.format_exc())
        result = _failure_result(
            case,
            output_dir,
            backend_label=backend_label,
            solver_device=solver_device,
            worker_jax_platforms=worker_jax_platforms,
            policy=policy,
            max_mode=max_mode,
            use_ess=use_ess,
            message=f"{type(exc).__name__}: {exc}",
            input_file=seeded_input_file,
        )
        _annotate_qi_partial_result(case, result, output_dir)
    result_path.write_text(json.dumps(asdict(result), indent=2))


def _read_result(path: Path) -> sweep.CaseResult:
    return sweep.CaseResult(**json.loads(path.read_text()))


def _prepare_output_dir_for_run(output_dir: Path, *, rerun: bool) -> None:
    """Clear stale per-case artifacts before an explicit rerun."""

    if bool(rerun) and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_showcase_summary(results: list[sweep.CaseResult], output_root: Path, summary_name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    records = [asdict(result) for result in results]
    (output_root / f"{summary_name}.json").write_text(json.dumps(records, indent=2))
    sweep._write_summary_csv(results, output_root / f"{summary_name}.csv")

    case_csv = output_root / f"{summary_name}_with_cases.csv"
    fieldnames = ["minimal_seed_case", "minimal_seed_nfp"] + list(records[0].keys()) if records else []
    if not fieldnames:
        case_csv.write_text("")
        return
    with case_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            row = asdict(result)
            output_dir = Path(str(result.output_dir or ""))
            meta_path = output_dir / "showcase_case.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            case_meta = meta.get("minimal_seed_case", {})
            row["minimal_seed_case"] = case_meta.get("name", "")
            row["minimal_seed_nfp"] = case_meta.get("nfp", "")
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cases", type=str, default="all", help="Comma-separated cases or 'all'.")
    parser.add_argument("--backend-label", type=str, default="cpu")
    parser.add_argument("--solver-device", type=str, default="cpu", help="Use 'cpu', 'gpu', or 'none'.")
    parser.add_argument(
        "--worker-jax-platforms",
        type=str,
        default="cpu",
        help="Use 'inherit', 'cpu', or 'cuda'. The user-facing alias 'gpu' maps to 'cuda'.",
    )
    parser.add_argument("--policy", choices=("continuation", "direct"), default="continuation")
    parser.add_argument("--max-mode", type=int, default=5)
    parser.add_argument("--ess", choices=("on", "off"), default="on")
    parser.add_argument("--max-nfev", type=int, default=8)
    parser.add_argument("--continuation-nfev", type=int, default=8)
    parser.add_argument("--inner-max-iter", type=int, default=120)
    parser.add_argument("--inner-ftol", type=float, default=1e-9)
    parser.add_argument("--trial-max-iter", type=int, default=120)
    parser.add_argument("--trial-ftol", type=float, default=1e-9)
    parser.add_argument("--ess-alpha", type=float, default=1.2, help="ESS high-mode scaling strength.")
    parser.add_argument("--case-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--reference-preseed-blend",
        type=float,
        default=None,
        help="Override the same-NFP reference-preseed blend for cases that use one.",
    )
    parser.add_argument(
        "--target-helicity-seed-amplitude",
        type=float,
        default=TARGET_HELICITY_SEED_AMPLITUDE,
        help="Per-run RBC/ZBS mode-1 seed amplitude; use 0 to disable.",
    )
    parser.add_argument("--rerun", action="store_true", help="Recompute cases even when case_result.json exists.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    case_names = _parse_case_names(args.cases)
    solver_device = None if str(args.solver_device).lower() in {"", "none", "default"} else str(args.solver_device)
    worker_jax_platforms = sweep._normalize_worker_jax_platforms(args.worker_jax_platforms)
    budget = MinimalSeedBudget(
        max_nfev=int(args.max_nfev),
        continuation_nfev=int(args.continuation_nfev),
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
        ess_alpha=float(args.ess_alpha),
    )
    use_ess = _bool_from_choice(args.ess)
    max_mode = int(args.max_mode)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")

    results: list[sweep.CaseResult] = []
    for case_name in case_names:
        case = SHOWCASE_CASES[case_name]
        reference_preseed_blend = None if args.reference_preseed_blend is None else float(args.reference_preseed_blend)
        if reference_preseed_blend is not None and case.reference_preseed_input is not None:
            case = replace(case, reference_preseed_blend=reference_preseed_blend)
        output_dir = _case_output_dir(
            output_root,
            case=case,
            backend_label=str(args.backend_label),
            policy=str(args.policy),
            max_mode=max_mode,
            use_ess=use_ess,
        )
        result_path = output_dir / "case_result.json"
        label = f"{case.name} problem={case.problem} nfp={case.nfp} policy={args.policy} mode={max_mode} ess={use_ess}"
        if result_path.exists() and not bool(args.rerun):
            result = _read_result(result_path)
            print(f"[{label}] skip existing success={result.success} crashed={result.crashed}", flush=True)
            results.append(result)
            continue
        _prepare_output_dir_for_run(output_dir, rerun=bool(args.rerun))
        case_timeout_s = None if args.case_timeout_s in (None, 0) else float(args.case_timeout_s)

        old_platforms = os.environ.get("JAX_PLATFORMS")
        if worker_jax_platforms is not None:
            os.environ["JAX_PLATFORMS"] = worker_jax_platforms
        try:
            proc = ctx.Process(
                target=_worker,
                args=(
                    case_name,
                    str(output_dir),
                    str(result_path),
                    str(args.backend_label),
                    solver_device,
                    worker_jax_platforms,
                    str(args.policy),
                    max_mode,
                    use_ess,
                    asdict(budget),
                    float(args.target_helicity_seed_amplitude),
                    reference_preseed_blend,
                    case_timeout_s,
                ),
            )
            t0 = time.perf_counter()
            proc.start()
        finally:
            if old_platforms is None:
                os.environ.pop("JAX_PLATFORMS", None)
            else:
                os.environ["JAX_PLATFORMS"] = old_platforms

        proc.join(timeout=case_timeout_s)
        elapsed_s = time.perf_counter() - t0
        timed_out = proc.is_alive()
        if timed_out:
            sweep._terminate_worker_process(proc)

        if result_path.exists():
            result = _read_result(result_path)
            result_needs_write = sweep._set_missing_wall_time(result, elapsed_s)
        elif timed_out:
            result = _failure_result(
                case,
                output_dir,
                backend_label=str(args.backend_label),
                solver_device=solver_device,
                worker_jax_platforms=worker_jax_platforms,
                policy=str(args.policy),
                max_mode=max_mode,
                use_ess=use_ess,
                message=f"worker timed out after {case_timeout_s:.1f} s",
                total_wall_time_s=elapsed_s,
            )
            result_needs_write = True
        else:
            result = _failure_result(
                case,
                output_dir,
                backend_label=str(args.backend_label),
                solver_device=solver_device,
                worker_jax_platforms=worker_jax_platforms,
                policy=str(args.policy),
                max_mode=max_mode,
                use_ess=use_ess,
                message=f"worker exit code {proc.exitcode} without result file",
                total_wall_time_s=elapsed_s,
            )
            result_needs_write = True

        qi_partial = _annotate_qi_partial_result(case, result, output_dir)
        if qi_partial:
            result_needs_write = True

        if proc.exitcode not in (0, None):
            if not (timed_out and qi_partial):
                result.crashed = True
                if "worker exit code" not in result.message and "exit code" not in result.message:
                    result.message = f"exit code {proc.exitcode}; {result.message}"
                result_needs_write = True
            elif "partial" not in str(result.message).lower():
                result.message = f"{result.message}; partial QI stage checkpoint metrics recorded"
                result_needs_write = True

        if result_needs_write or not result_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            if not (output_dir / "showcase_case.json").exists():
                seed_terms = _target_helicity_seed_terms(
                    max_mode=max_mode,
                    amplitude=float(args.target_helicity_seed_amplitude),
                )
                seeded_input_file = output_dir / "input.target_helicity_seed" if seed_terms else case.input_file
                _write_showcase_metadata(
                    output_dir,
                    case=case,
                    policy=str(args.policy),
                    max_mode=max_mode,
                    use_ess=use_ess,
                    budget=budget,
                    seeded_input_file=seeded_input_file,
                    seed_terms=seed_terms,
                    seed_amplitude=float(args.target_helicity_seed_amplitude),
                    reference_preseeded_input_file=(
                        output_dir / "input.reference_preseed"
                        if case.reference_preseed_input is not None
                        else None
                    ),
                    reference_preseed_changes=[],
                )
            result_path.write_text(json.dumps(asdict(result), indent=2))

        results.append(result)
        print(
            f"[{label}] success={result.success} crashed={result.crashed} objective={result.objective_final}",
            flush=True,
        )

    summary_name = f"summary_{args.backend_label}_{args.policy}_mode{max_mode}_{sweep._ess_label(use_ess)}"
    _write_showcase_summary(results, output_root, summary_name)
    print(f"Wrote {output_root / (summary_name + '.csv')}")


if __name__ == "__main__":
    main()

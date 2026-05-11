#!/usr/bin/env python
"""Build the optional QI seed-robustness validation plan.

The plan is intentionally declarative: it records commands, prerequisites,
acceptance gates, and deferred lanes without executing VMEC2000, SIMSOPT, or
full optimization sweeps.  This keeps required CI light while making the manual
robustness workflow concrete and reviewable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


LATEST_GREEN_CI = {
    "workflow": "CI",
    "status": "success",
    "branch": "main",
    "head_sha": "5ca8216699c766621a1fe30e47db9b68befd36c2",
    "completed_at_utc": "2026-05-11T17:14:59Z",
    "url": "https://github.com/uwplasma/vmec_jax/actions/runs/25684339586",
}

QI_FAMILY_REPRESENTATIVES = [
    {
        "family": "qi",
        "label": "qi_nfp3_fixed_resolution",
        "required_for_family_probe": True,
        "source": "bundled examples/data input+wout",
    },
    {
        "family": "qp",
        "label": "qp_from_omnigenity_nfp2_qi",
        "required_for_family_probe": False,
        "source": "optional OMNIGENITY_OPTIMIZATION_ROOT checkout",
    },
    {
        "family": "qh",
        "label": "qh_nfp4_warm_start",
        "required_for_family_probe": True,
        "source": "bundled examples/data input+wout",
    },
    {
        "family": "qa",
        "label": "qa_landreman_paul_lowres",
        "required_for_family_probe": True,
        "source": "bundled examples/data input+wout",
    },
    {
        "family": "simple",
        "label": "simple_circular_tokamak",
        "required_for_family_probe": True,
        "source": "bundled examples/data input+wout",
    },
]


@dataclass(frozen=True)
class ValidationLane:
    lane_id: str
    title: str
    required_ci: bool
    prerequisites: list[str]
    command: str
    acceptance: list[str]
    artifact_paths: list[str]


def _lanes() -> list[ValidationLane]:
    return [
        ValidationLane(
            lane_id="required-fast-ci",
            title="Required fast semantic gate",
            required_ci=True,
            prerequisites=["standard test dependencies"],
            command='JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000"',
            acceptance=[
                "All required tests pass.",
                "No VMEC2000 executable, SIMSOPT install, or optimization sweep is required.",
            ],
            artifact_paths=["coverage.xml in the Python 3.11 CI job"],
        ),
        ValidationLane(
            lane_id="required-docs-ci",
            title="Required docs gates",
            required_ci=True,
            prerequisites=["Sphinx and furo docs dependencies"],
            command=(
                "SPHINX_FAST=1 python -m sphinx -W -j auto -b html docs docs/_build/html && "
                "READTHEDOCS=True python -m sphinx -W -j auto -b html docs docs/_build/html_full"
            ),
            acceptance=["Fast and full guide builds complete with warnings treated as errors."],
            artifact_paths=["docs/_build/html", "docs/_build/html_full"],
        ),
        ValidationLane(
            lane_id="qi-family-audit",
            title="Family-representative QI solved-state audit",
            required_ci=False,
            prerequisites=[
                "bundled examples/data inputs and wouts",
                "optional OMNIGENITY_OPTIMIZATION_ROOT for QP and extra QI representatives",
            ],
            command=(
                "PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py "
                "--quick --output results/qi_seed_audit/summary.json "
                "--csv results/qi_seed_audit/summary.csv"
            ),
            acceptance=[
                "Audit records QI, QH, QA, and simple families; QP is recorded when the optional checkout exists.",
                "Rows include smooth QI, legacy QI, mirror ratio, elongation, aspect ratio, mean iota, and skipped defaults.",
                "No optimization is launched.",
            ],
            artifact_paths=[
                "results/qi_seed_audit/summary.json",
                "results/qi_seed_audit/summary.csv",
            ],
        ),
        ValidationLane(
            lane_id="qi-prefine-manifest",
            title="Bounded QI prefine probe manifest",
            required_ci=False,
            prerequisites=["qi-family-audit output", "reviewer approval before any run mode"],
            command=(
                "PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py "
                "--quick --prefine-probes plan "
                "--prefine-manifest results/qi_seed_audit/prefine_manifest.json"
            ),
            acceptance=[
                "Manifest selects top-ranked rows plus one best-ranked representative per available seed family.",
                "Hard caps keep max_nfev, continuation_nfev, stage count, mode count, and Boozer resolution small.",
                "Dry-run manifest lists exact commands and expected outputs before any expensive probe runs.",
            ],
            artifact_paths=["results/qi_seed_audit/prefine_manifest.json"],
        ),
        ValidationLane(
            lane_id="qi-prefine-run",
            title="Explicit tiny QI prefine probes",
            required_ci=False,
            prerequisites=["reviewed prefine manifest", "deliberate local/manual execution"],
            command=(
                "PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py "
                "--quick --prefine-probes run "
                "--prefine-manifest results/qi_seed_audit/prefine_manifest.json"
            ),
            acceptance=[
                "Each executed plan reports completed or failed status.",
                "A family can only support robustness claims after final diagnostics and Boozer contour plots are audited.",
            ],
            artifact_paths=["results/qi_seed_audit/prefine_probes"],
        ),
        ValidationLane(
            lane_id="simsopt-optional",
            title="Optional SIMSOPT formula parity",
            required_ci=False,
            prerequisites=["SIMSOPT installed locally", "RUN_SIMSOPT_VALIDATION=1"],
            command="RUN_SIMSOPT_VALIDATION=1 pytest -q tests/test_simsopt_optional_validation.py",
            acceptance=[
                "VMEC-only QS residuals match SIMSOPT diagnostics on the bundled QH wout.",
                "The test skips instead of failing when SIMSOPT is unavailable.",
            ],
            artifact_paths=[],
        ),
        ValidationLane(
            lane_id="vmec2000-optional",
            title="Optional VMEC2000 executable parity",
            required_ci=False,
            prerequisites=["local VMEC2000 executable", "VMEC2000_EXEC", "VMEC2000_INTEGRATION=1"],
            command=(
                "VMEC2000_EXEC=/path/to/xvmec2000 VMEC2000_INTEGRATION=1 "
                "pytest -q tests/test_vmec2000_exec_fast_validation.py"
            ),
            acceptance=[
                "Executable-backed parity checks pass on local VMEC2000 output.",
                "The broader vmec2000 marker suite remains manual/scheduled, not required PR CI.",
            ],
            artifact_paths=[],
        ),
    ]


def build_plan(*, ci: dict[str, str] | None = None) -> dict[str, Any]:
    """Return a JSON-serializable optional validation plan."""

    lanes = _lanes()
    return {
        "mode": "optional_qi_seed_robustness_validation_plan",
        "required_ci_baseline": ci or LATEST_GREEN_CI,
        "required_ci_policy": {
            "heavy_external_validation_required": False,
            "required_ci_lanes": [lane.lane_id for lane in lanes if lane.required_ci],
            "optional_lanes": [lane.lane_id for lane in lanes if not lane.required_ci],
        },
        "family_representatives": QI_FAMILY_REPRESENTATIVES,
        "lanes": [asdict(lane) for lane in lanes],
        "next_parity_gates": [
            {
                "gate": "QI solved-state fixture",
                "status": "next",
                "criterion": (
                    "Add one small solved-state fixture around qi_diagnostics_from_state "
                    "before claiming optimizer seed robustness."
                ),
            },
            {
                "gate": "Family-prefine probes",
                "status": "manual/nightly",
                "criterion": "Run reviewed dry-run plans across QI, QP, QH, QA, and simple seeds.",
            },
            {
                "gate": "VMEC2000 executable parity",
                "status": "manual/nightly",
                "criterion": "Keep smoke executable parity green before broadening the manifest matrix.",
            },
            {
                "gate": "SIMSOPT diagnostic parity",
                "status": "optional",
                "criterion": "Keep formula-level SIMSOPT comparisons green when SIMSOPT is installed.",
            },
        ],
        "deferred_validation_lanes": [
            "Full multi-seed constrained-QI optimization sweep with visual Boozer |B| contour audit.",
            "Full VMEC2000 parity manifest against a local executable and fetched large assets.",
            "SIMSOPT finite-difference optimization comparison beyond formula-level residual parity.",
            "GPU-specific QI prefine/optimization robustness matrix.",
        ],
    }


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Optional QI Seed Robustness Validation Plan",
        "",
        f"Latest required CI baseline: {plan['required_ci_baseline']['status']} "
        f"({plan['required_ci_baseline']['completed_at_utc']}, "
        f"{plan['required_ci_baseline']['head_sha'][:12]}).",
        "",
        "## Family Representatives",
    ]
    for row in plan["family_representatives"]:
        required = "required" if row["required_for_family_probe"] else "optional"
        lines.append(f"- {row['family']}: {row['label']} ({required}; {row['source']})")
    lines.extend(["", "## Lanes"])
    for lane in plan["lanes"]:
        marker = "required CI" if lane["required_ci"] else "optional"
        lines.extend(
            [
                f"- {lane['lane_id']}: {lane['title']} ({marker})",
                f"  Command: `{lane['command']}`",
            ]
        )
    lines.extend(["", "## Deferred Lanes"])
    lines.extend(f"- {lane}" for lane in plan["deferred_validation_lanes"])
    return "\n".join(lines) + "\n"


def _write_output(plan: dict[str, Any], output: Path, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    else:
        output.write_text(render_markdown(plan))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/qi_seed_audit/validation_plan.json"))
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--ci-status", default=None, help="Override the recorded CI status for local experiments.")
    parser.add_argument("--ci-head-sha", default=None, help="Override the recorded CI head SHA.")
    parser.add_argument("--ci-url", default=None, help="Override the recorded CI run URL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ci = dict(LATEST_GREEN_CI)
    if args.ci_status is not None:
        ci["status"] = args.ci_status
    if args.ci_head_sha is not None:
        ci["head_sha"] = args.ci_head_sha
    if args.ci_url is not None:
        ci["url"] = args.ci_url

    plan = build_plan(ci=ci)
    _write_output(plan, args.output, args.format)
    print(f"Wrote {args.output} with {len(plan['lanes'])} validation lanes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

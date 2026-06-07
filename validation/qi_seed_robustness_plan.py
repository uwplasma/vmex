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


CI_VERIFICATION_PLACEHOLDER = {
    "workflow": "CI",
    "status": "unverified",
    "branch": "main",
    "head_sha": "",
    "completed_at_utc": "",
    "url": "",
    "verification_required": True,
    "verification_command": (
        "gh run list --repo uwplasma/vmec_jax --branch main "
        "--workflow CI --limit 5"
    ),
}

QI_FAMILY_REPRESENTATIVES = [
    {
        "family": "qi",
        "label": "qi_nfp3_fixed_resolution",
        "required_for_family_probe": True,
        "source": "bundled examples/data input+wout",
    },
    {
        "family": "qi",
        "label": "qi_stel_seed_3127",
        "required_for_family_probe": True,
        "source": "bundled examples/data input+wout robustness seed",
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

FAST_QI_DIAGNOSTIC_ARTIFACTS = [
    {
        "artifact_id": "qi-seed-suitability-annotation",
        "required_ci": True,
        "tests": [
            "tests/test_qi_diagnostics.py::test_qi_seed_suitability_annotation_reports_gate_failures",
            "tests/test_qi_diagnostics.py::test_qi_seed_ranking_tracks_legacy_goodman_order_on_synthetic_modes",
            "tests/test_qi_diagnostics.py::test_qi_diagnostics_from_bundled_solved_qi_seed_records_state_metrics",
        ],
        "validates": [
            "Smooth and legacy Goodman-style QI totals are combined into a deterministic rank score.",
            "Mirror, iota, aspect, and elongation gates emit explicit failure reasons.",
            "The bundled solved QI seed fixture records scalar state metrics without launching optimization.",
        ],
    }
]

QI_SEED_ROBUSTNESS_PROMOTION_GATES = [
    {
        "gate_id": "solved-state-family-audit",
        "title": "Solved-state family audit",
        "status": "manual-required",
        "required_before_robustness_claim": True,
        "pass_criteria": [
            "Audit rows cover QI, QH, QA, and simple bundled representatives; QP is included when the optional checkout exists.",
            "Each available row records smooth QI, raw QI, legacy QI, mirror ratio, elongation, aspect, mean iota, and failed constraints.",
            "Skipped optional representatives remain explicit in the report and keep that family deferred.",
        ],
        "artifact_paths": [
            "results/qi_seed_audit/summary.json",
            "results/qi_seed_audit/summary.csv",
        ],
    },
    {
        "gate_id": "reviewed-constrained-prefine-manifest",
        "title": "Reviewed constrained prefine manifest",
        "status": "manual-required",
        "required_before_robustness_claim": True,
        "pass_criteria": [
            "Dry-run plans select top-ranked rows plus one best-ranked representative per available seed family.",
            "Default objectives include smooth QI, a QI ceiling, all-surface mirror ratio, and elongation; QI-only ablations use explicit zero-weight flags.",
            "Already-low-QI seeds are labeled with a near-QI preservation policy that removes auxiliary mirror/elongation cleanup from the bounded first pass.",
            "Each plan records exact run commands, stage modes, ESS controls, Boozer/QI resolution, phimin policy, and hard nfev caps before any run mode.",
        ],
        "artifact_paths": ["results/qi_seed_audit/prefine_manifest.json"],
    },
    {
        "gate_id": "bounded-prefine-probe-results",
        "title": "Bounded constrained prefine probe results",
        "status": "not-complete",
        "required_before_robustness_claim": True,
        "pass_criteria": [
            "Reviewed run-mode probes finish with completed, failed, or timed-out status rather than silent promotion.",
            "Summaries identify best final objective, best improvement, objective-history regressions, and recommended next action.",
            "Scalar-objective improvement is insufficient when smooth or legacy QI worsens or engineering diagnostics fail.",
        ],
        "artifact_paths": ["results/qi_seed_audit/prefine_probes"],
    },
    {
        "gate_id": "independent-final-diagnostics",
        "title": "Independent final diagnostics",
        "status": "not-complete",
        "required_before_robustness_claim": True,
        "pass_criteria": [
            "Final candidates pass smooth QI, legacy QI, abs(mean iota), mirror ratio, elongation, and aspect gates at audit resolution.",
            "Diagnostics are computed from final input and wout artifacts, not from optimizer scalar objectives alone.",
            "Higher-resolution Boozer/QI re-evaluation does not qualitatively change QI ranking or engineering gate status.",
        ],
        "artifact_paths": [
            "results/qi_seed_audit/prefine_probes/*/diagnostics.json",
            "results/qi_seed_audit/prefine_probes/*/wout_final.nc",
        ],
    },
    {
        "gate_id": "boozer-contour-review",
        "title": "Boozer contour review",
        "status": "not-complete",
        "required_before_robustness_claim": True,
        "pass_criteria": [
            "Final Boozer |B| contour plots are generated from the same final wout used for diagnostics.",
            "Reviewed contours are poloidally closed or otherwise flagged as failed QI evidence.",
            "VMEC-angle plots alone are not accepted as a QI visual gate.",
        ],
        "artifact_paths": ["results/qi_seed_audit/prefine_probes/*/boozer_bmag_contours.png"],
    },
    {
        "gate_id": "multi-family-convergence-matrix",
        "title": "Multi-family convergence matrix",
        "status": "deferred",
        "required_before_robustness_claim": True,
        "pass_criteria": [
            "The constrained objective is run from QI, QP, QH, QA, and simple non-omnigenous seed families.",
            "Every accepted family row passes the independent numerical gates and Boozer contour review.",
            "Failing or missing family rows are documented as robustness gaps instead of being hidden by passing rows.",
        ],
        "artifact_paths": ["results/qi_seed_audit/multi_family_matrix.json"],
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


@dataclass(frozen=True)
class OptionalParityCommand:
    command_id: str
    backend: str
    required_ci: bool
    env: list[str]
    command: str
    bounded_by: list[str]
    validates: list[str]


def _optional_parity_commands() -> list[OptionalParityCommand]:
    return [
        OptionalParityCommand(
            command_id="simsopt-qs-family-formula",
            backend="SIMSOPT",
            required_ci=False,
            env=["RUN_SIMSOPT_VALIDATION=1"],
            command=(
                "RUN_SIMSOPT_VALIDATION=1 pytest -q "
                "tests/test_simsopt_optional_validation.py::"
                "test_quasisymmetry_residual_family_matches_simsopt_wout_formula"
            ),
            bounded_by=[
                "uses bundled QA and QH wout fixtures",
                "uses three radial surfaces and a 15x16 angular grid",
                "skips unless RUN_SIMSOPT_VALIDATION=1 and SIMSOPT is importable",
            ],
            validates=[
                "VMEC-only QA/QH quasisymmetry residual formulas match SIMSOPT diagnostics",
                "formula-level parity remains available without launching optimization",
            ],
        ),
        OptionalParityCommand(
            command_id="simsopt-qs-family-state",
            backend="SIMSOPT",
            required_ci=False,
            env=["RUN_SIMSOPT_VALIDATION=1"],
            command=(
                "RUN_SIMSOPT_VALIDATION=1 pytest -q "
                "tests/test_simsopt_optional_validation.py::"
                "test_quasisymmetry_state_diagnostic_family_matches_simsopt_converged_wout"
            ),
            bounded_by=[
                "uses bundled QA and QH converged wout fixtures",
                "uses the same low angular grid as the formula smoke",
                "skips unless RUN_SIMSOPT_VALIDATION=1 and SIMSOPT is importable",
            ],
            validates=[
                "State-derived VMEC diagnostics stay consistent with SIMSOPT QS residuals",
                "QA and QH optimization objectives use the same residual convention as the reference code",
            ],
        ),
        OptionalParityCommand(
            command_id="simsopt-redl-formula",
            backend="SIMSOPT",
            required_ci=False,
            env=["RUN_SIMSOPT_VALIDATION=1"],
            command=(
                "RUN_SIMSOPT_VALIDATION=1 pytest -q "
                "tests/test_finite_beta_helpers_unit.py::"
                "test_redl_bootstrap_formula_matches_simsopt_when_available"
            ),
            bounded_by=[
                "uses synthetic three-point profile and geometry arrays only",
                "launches no VMEC, BOOZ_XFORM, or optimization solve",
                "skips unless RUN_SIMSOPT_VALIDATION=1 and SIMSOPT is importable",
            ],
            validates=[
                "Redl bootstrap jdotB algebra matches SIMSOPT formula output",
                "finite-beta current objective wiring keeps a reference-code parity gate",
            ],
        ),
        OptionalParityCommand(
            command_id="vmec2000-converged-wout-smoke",
            backend="VMEC2000 executable",
            required_ci=False,
            env=["VMEC2000_EXEC=/path/to/xvmec2000", "VMEC2000_INTEGRATION=1"],
            command=(
                "VMEC2000_EXEC=/path/to/xvmec2000 VMEC2000_INTEGRATION=1 pytest -q "
                "tests/test_vmec2000_exec_fast_validation.py::"
                "test_vmec2000_converged_wout_diagnostics_validation"
            ),
            bounded_by=[
                "uses three low-resolution fixed-boundary inputs, including 3D QH",
                "patches NS/NITER/FTOL to bounded converged end-state runs",
                "uses a 120s executable timeout per case",
            ],
            validates=[
                "converged wout geometry, profiles, field coefficients, and scalar diagnostics against VMEC2000",
                "end-state parity remains available without brittle finite-step trace matching",
            ],
        ),
        OptionalParityCommand(
            command_id="vmec2000-stage-trace-smoke",
            backend="VMEC2000 executable",
            required_ci=False,
            env=["VMEC2000_EXEC=/path/to/xvmec2000", "VMEC2000_INTEGRATION=1"],
            command=(
                "VMEC2000_EXEC=/path/to/xvmec2000 VMEC2000_INTEGRATION=1 pytest -q "
                "tests/test_vmec2000_exec_fast_validation.py::"
                "test_fast_vmec2000_stage_trace_validation_cases"
            ),
            bounded_by=[
                "uses two bundled fixed-boundary inputs",
                "forces --single-ns 13 and --max-iter 2",
                "uses lite VMEC2000 dump output and a 60s executable timeout per case",
            ],
            validates=[
                "early-stage fsq trace parity against a local VMEC2000 executable",
                "axisymmetric and lasym=True pressure inputs stay covered before broadening the manifest",
            ],
        ),
        OptionalParityCommand(
            command_id="vmec2000-cli-five-iter",
            backend="VMEC2000 executable",
            required_ci=False,
            env=[
                "VMEC2000_EXEC=/path/to/xvmec2000",
                "VMEC2000_INTEGRATION=1",
                "VMEC2000_CLI_NITER=5",
            ],
            command=(
                "VMEC2000_EXEC=/path/to/xvmec2000 VMEC2000_INTEGRATION=1 "
                "VMEC2000_CLI_NITER=5 pytest -q tests/test_cli_vmec2000_exec.py"
            ),
            bounded_by=[
                "caps both VMEC2000 and vmec_jax CLI runs at five iterations",
                "uses each input deck's current grid only, with multigrid disabled in vmec_jax",
                "skips missing optional QA input instead of failing required CI",
            ],
            validates=[
                "CLI parity wiring produces comparable wout geometry modes",
                "VMEC2000 executable comparison remains available through the public command path",
            ],
        ),
        OptionalParityCommand(
            command_id="vmec2000-w7x-generated-mgrid-wout",
            backend="VMEC2000 executable + SIMSOPT generated mgrid",
            required_ci=False,
            env=["VMEC2000_EXEC=/path/to/xvmec2000", "VMEC2000_INTEGRATION=1"],
            command=(
                "VMEC2000_EXEC=/path/to/xvmec2000 VMEC2000_INTEGRATION=1 pytest -q "
                "tests/test_free_boundary_essos_coil_parity.py::"
                "test_vmec2000_w7x_generated_mgrid_fixture_reaches_active_vacuum_and_finite_wout"
            ),
            bounded_by=[
                "regenerates the W7-X mgrid in a temporary directory from SIMSOPT coil data",
                "uses a low-resolution two-stage free-boundary VMEC2000 schedule",
                "promotes only active vacuum, finite residuals, and finite positive geometry WOUTs",
            ],
            validates=[
                "stock VMEC2000 can consume a generated mgrid and produce a physical free-boundary WOUT",
                "generated mgrid WOUT-level promotion stays optional and executable-backed",
                "binary mgrid and WOUT assets remain outside the git repository",
            ],
        ),
        OptionalParityCommand(
            command_id="bundled-wout-two-case-smoke",
            backend="bundled VMEC2000 wout fixtures",
            required_ci=False,
            env=["RUN_FULL=1"],
            command=(
                "RUN_FULL=1 pytest -q "
                "tests/test_wout_comprehensive_parity.py::test_wout_comprehensive_parity[circular_tokamak] "
                "tests/test_wout_comprehensive_parity.py::test_wout_comprehensive_parity[nfp4_QH_warm_start]"
            ),
            bounded_by=[
                "runs two representative bundled wout fixtures only",
                "requires no local VMEC2000 executable or SIMSOPT install",
                "kept outside required CI because it launches full fixed-boundary solves",
            ],
            validates=[
                "static VMEC2000 reference parity for simple axisymmetric and QH 3D cases",
                "future external parity failures can be separated from solver/reference drift",
            ],
        ),
    ]


def _lanes() -> list[ValidationLane]:
    return [
        ValidationLane(
            lane_id="required-fast-ci",
            title="Required fast semantic gate",
            required_ci=True,
            prerequisites=["standard test dependencies"],
            command='JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000 and not simsopt"',
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
                "--prefine-manifest results/qi_seed_audit/prefine_manifest.json "
                "--prefine-mirror-weight 2.0 --prefine-elongation-weight 0.5 "
                "--prefine-mirror-surface-index all"
            ),
            acceptance=[
                "Manifest selects top-ranked rows plus one best-ranked representative per available seed family.",
                "Default prefine objective includes smooth QI, a QI ceiling, all-surface mirror ratio, and elongation terms.",
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
                "--quick --prefine-probes run --prefine-reviewed "
                "--prefine-manifest results/qi_seed_audit/prefine_manifest.json "
                "--prefine-mirror-weight 2.0 --prefine-elongation-weight 0.5 "
                "--prefine-mirror-surface-index all"
            ),
            acceptance=[
                "Each executed plan reports completed, failed, or timed-out status with completed stage modes.",
                "Constrained probes must report independent QI, mirror, elongation, iota, and aspect diagnostics.",
                "Manifest summary identifies best final objective, best improvement, objective-history regressions, and next action.",
                "A family can only support robustness claims after final diagnostics and Boozer contour plots are audited.",
            ],
            artifact_paths=["results/qi_seed_audit/prefine_probes"],
        ),
        ValidationLane(
            lane_id="simsopt-optional",
            title="Optional SIMSOPT formula and state parity",
            required_ci=False,
            prerequisites=["SIMSOPT installed locally", "RUN_SIMSOPT_VALIDATION=1"],
            command=(
                "RUN_SIMSOPT_VALIDATION=1 pytest -q "
                "tests/test_simsopt_optional_validation.py "
                "tests/test_redl_bootstrap_simsopt_parity.py "
                "tests/test_finite_beta_helpers_unit.py::"
                "test_redl_bootstrap_formula_matches_simsopt_when_available"
            ),
            acceptance=[
                "VMEC-only QS residuals match SIMSOPT diagnostics on bundled QA and QH wouts.",
                "State-derived VMEC diagnostics match SIMSOPT on the same converged fixtures.",
                "Redl bootstrap current formulas match SIMSOPT on synthetic and bundled finite-beta cases.",
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
                "Executable-backed stage-trace and converged-wout parity checks pass on local VMEC2000 output.",
                "The broader vmec2000 marker suite remains manual/scheduled, not required PR CI.",
            ],
            artifact_paths=[],
        ),
    ]


def build_plan(*, ci: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a JSON-serializable optional validation plan."""

    lanes = _lanes()
    optional_parity_commands = _optional_parity_commands()
    return {
        "schema_version": 2,
        "mode": "optional_qi_seed_robustness_validation_plan",
        "robustness_claim_status": "incomplete_until_promotion_gates_pass",
        "required_ci_baseline": ci or CI_VERIFICATION_PLACEHOLDER,
        "required_ci_policy": {
            "heavy_external_validation_required": False,
            "required_ci_lanes": [lane.lane_id for lane in lanes if lane.required_ci],
            "optional_lanes": [lane.lane_id for lane in lanes if not lane.required_ci],
        },
        "family_representatives": QI_FAMILY_REPRESENTATIVES,
        "fast_diagnostic_artifacts": FAST_QI_DIAGNOSTIC_ARTIFACTS,
        "qi_seed_robustness_promotion_gates": QI_SEED_ROBUSTNESS_PROMOTION_GATES,
        "lanes": [asdict(lane) for lane in lanes],
        "optional_parity_commands": [asdict(command) for command in optional_parity_commands],
        "next_parity_gates": [
            {
                "gate": "QI solved-state fixture",
                "status": "covered-fast-ci",
                "criterion": (
                    "Keep the bundled low-resolution qi_diagnostics_from_state fixture green; "
                    "add more families before claiming broad optimizer seed robustness."
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
                "criterion": (
                    "Keep stock-executable smoke green before broadening the manifest matrix; "
                    "strict external LASYM parity remains optional/instrumented until known gaps clear."
                ),
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
    ci = plan["required_ci_baseline"]
    if ci.get("verification_required"):
        ci_summary = (
            "Required CI baseline: unverified. Verify the latest main-branch CI "
            f"with `{ci['verification_command']}` before using this plan for release validation."
        )
    else:
        ci_summary = (
            f"Required CI baseline: {ci['status']} "
            f"({ci.get('completed_at_utc') or 'completion time not recorded'}, "
            f"{ci.get('head_sha', '')[:12] or 'SHA not recorded'})."
        )
    lines = [
        "# Optional QI Seed Robustness Validation Plan",
        "",
        ci_summary,
        "",
        "## Family Representatives",
    ]
    for row in plan["family_representatives"]:
        required = "required" if row["required_for_family_probe"] else "optional"
        lines.append(f"- {row['family']}: {row['label']} ({required}; {row['source']})")
    lines.extend(["", "## Fast Diagnostic Artifacts"])
    for artifact in plan["fast_diagnostic_artifacts"]:
        marker = "required CI" if artifact["required_ci"] else "optional"
        lines.append(f"- {artifact['artifact_id']} ({marker})")
    lines.extend(["", "## QI Seed Robustness Promotion Gates"])
    lines.append(f"Status: {plan['robustness_claim_status']}")
    for gate in plan["qi_seed_robustness_promotion_gates"]:
        marker = "required" if gate["required_before_robustness_claim"] else "optional"
        lines.extend(
            [
                f"- {gate['gate_id']}: {gate['title']} ({marker}; {gate['status']})",
                f"  Criteria: {'; '.join(gate['pass_criteria'])}",
            ]
        )
    lines.extend(["", "## Lanes"])
    for lane in plan["lanes"]:
        marker = "required CI" if lane["required_ci"] else "optional"
        lines.extend(
            [
                f"- {lane['lane_id']}: {lane['title']} ({marker})",
                f"  Command: `{lane['command']}`",
            ]
        )
    lines.extend(["", "## Optional Parity Commands"])
    for command in plan["optional_parity_commands"]:
        lines.extend(
            [
                f"- {command['command_id']}: {command['backend']}",
                f"  Command: `{command['command']}`",
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
    parser.add_argument("--ci-completed-at-utc", default=None, help="Override the recorded CI completion time.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ci = dict(CI_VERIFICATION_PLACEHOLDER)
    if args.ci_status is not None:
        ci["status"] = args.ci_status
        ci["verification_required"] = False
    if args.ci_head_sha is not None:
        ci["head_sha"] = args.ci_head_sha
        ci["verification_required"] = False
    if args.ci_url is not None:
        ci["url"] = args.ci_url
        ci["verification_required"] = False
    if args.ci_completed_at_utc is not None:
        ci["completed_at_utc"] = args.ci_completed_at_utc
        ci["verification_required"] = False

    plan = build_plan(ci=ci)
    _write_output(plan, args.output, args.format)
    print(f"Wrote {args.output} with {len(plan['lanes'])} validation lanes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

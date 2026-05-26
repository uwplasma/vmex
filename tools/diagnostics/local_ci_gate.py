#!/usr/bin/env python
"""Run the opt-in local CI gate before pushing.

The gate mirrors the required GitHub Actions lanes closely enough to catch
ordinary failures before spending hosted CI minutes.  It intentionally avoids
external VMEC2000, SIMSOPT, GPU, and nightly/full-physics lanes.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
COVERAGE_FAIL_UNDER = 95

PHYSICS_SMOKE_TESTS = (
    "tests/test_wout_comprehensive_parity.py::test_wout_comprehensive_parity[circular_tokamak]",
    "tests/test_wout_comprehensive_parity.py::test_wout_comprehensive_parity[nfp4_QH_warm_start]",
    "tests/test_wout_comprehensive_parity.py::test_convergence_only[cth_like_free_bdy]",
    "tests/test_wout_comprehensive_parity.py::test_convergence_only[nfp2_QA_highres]",
    (
        "tests/test_wout_bfield_bundled_parity.py::"
        "test_bundled_wout_stored_bsup_cartesian_magnitude_matches_bmnc"
        "[basic_non_stellsym_pressure_single_grid-"
        "examples_single_grid/data/input.basic_non_stellsym_pressure-"
        "examples_single_grid/data/wout_basic_non_stellsym_pressure_reference.nc-0.012]"
    ),
    (
        "tests/test_wout_profiles_currents_bundled_parity.py::"
        "test_bundled_wout_surface_averaged_currents_follow_ampere_radial_difference"
        "[single_grid_lasym_pressure]"
    ),
    (
        "tests/test_converged_wout_matrix_parity.py::"
        "test_bundled_converged_wout_matrix_physics_gates"
        "[fixed_nonaxis_lasym_single_basic_non_stellsym_pressure]"
    ),
    "tests/test_driver_api.py::test_run_free_boundary_smoke_on_bundled_small_case",
    "tests/test_qs_ess_render_smoke.py",
)


@dataclass(frozen=True)
class Stage:
    name: str
    commands: tuple[tuple[str, ...], ...]
    env: dict[str, str] | None = None


def _python(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)


def _stages(cli_outdir: Path) -> tuple[Stage, ...]:
    cli_wout = cli_outdir / "wout_circular_tokamak.nc"
    return (
        Stage("cli-smoke-help", (("vmec_jax", "--help"),), env={"JAX_ENABLE_X64": "1"}),
        Stage(
            "cli-smoke-solve",
            (
                (
                    "vmec_jax",
                    str(ROOT / "examples/data/input.circular_tokamak"),
                    "--max-iter",
                    "2",
                    "--no-multigrid",
                    "--no-use-input-niter",
                    "--quiet",
                    "--outdir",
                    str(cli_outdir),
                ),
                _python(
                    "-c",
                    (
                        "from pathlib import Path; "
                        f"path = Path({str(cli_wout)!r}); "
                        "raise SystemExit(0 if path.is_file() else f'missing {path}')"
                    ),
                ),
            ),
            env={"JAX_ENABLE_X64": "1"},
        ),
        Stage(
            "compile",
            (_python("-m", "compileall", "-q", "vmec_jax", "examples", "tests", "tools", "validation"),),
        ),
        Stage(
            "repo-size-audit",
            (
                _python(
                    "tools/diagnostics/repo_size_audit.py",
                    "--top",
                    "20",
                    "--max-total-mib",
                    "50",
                    "--max-file-mib",
                    "2",
                ),
            ),
        ),
        Stage("fetch-assets", (_python("tools/fetch_assets.py"),)),
        Stage(
            "fast-pytest-coverage",
            (
                (
                    "pytest",
                    "-q",
                    "-m",
                    "not full and not vmec2000 and not simsopt",
                    "--cov=vmec_jax",
                    "--cov-report=xml",
                    "--cov-report=term:skip-covered",
                    f"--cov-fail-under={COVERAGE_FAIL_UNDER}",
                ),
            ),
            env={"JAX_ENABLE_X64": "1"},
        ),
        Stage(
            "physics-smoke",
            (("pytest", "-q", *PHYSICS_SMOKE_TESTS),),
            env={"JAX_ENABLE_X64": "1", "RUN_FULL": "1"},
        ),
        Stage("build", (_python("-m", "build"),)),
        Stage(
            "docs-fast",
            (_python("-m", "sphinx", "-W", "-j", "auto", "-b", "html", "docs", "docs/_build/html"),),
            env={"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "SPHINX_FAST": "1"},
        ),
        Stage(
            "docs-full",
            (_python("-m", "sphinx", "-W", "-j", "auto", "-b", "html", "docs", "docs/_build/html_full"),),
            env={"LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "READTHEDOCS": "True"},
        ),
    )


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _selected_stages(stages: Sequence[Stage], *, only: set[str], skip: set[str]) -> list[Stage]:
    known = {stage.name for stage in stages}
    unknown_only = only - known
    unknown_skip = skip - known
    if unknown_only or unknown_skip:
        unknown = sorted(unknown_only | unknown_skip)
        raise SystemExit(f"Unknown stage name(s): {', '.join(unknown)}")
    return [stage for stage in stages if (not only or stage.name in only) and stage.name not in skip]


def _run_stage(stage: Stage, *, dry_run: bool) -> None:
    env = os.environ.copy()
    if stage.env:
        env.update(stage.env)
    print(f"\n==> {stage.name}")
    for command in stage.commands:
        if stage.env:
            env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(stage.env.items()))
            print(f"+ {env_prefix} {_format_command(command)}")
        else:
            print(f"+ {_format_command(command)}")
        if not dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    stage_names = [stage.name for stage in _stages(Path("/tmp/vmec_jax_local_ci_smoke"))]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--list", action="store_true", help="List stage names and exit.")
    parser.add_argument(
        "--only",
        action="append",
        choices=stage_names,
        default=[],
        help="Run only this stage. May be supplied more than once.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        choices=stage_names,
        default=[],
        help="Skip this stage. May be supplied more than once.",
    )
    parser.add_argument(
        "--cli-outdir",
        type=Path,
        default=None,
        help="Output directory for the CLI solve smoke. Defaults to a temporary directory.",
    )
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.list:
        for stage in _stages(Path("/tmp/vmec_jax_local_ci_smoke")):
            print(stage.name)
        return 0

    if args.cli_outdir is None:
        with tempfile.TemporaryDirectory(prefix="vmec_jax_cli_smoke_") as tmpdir:
            stages = _selected_stages(_stages(Path(tmpdir)), only=set(args.only), skip=set(args.skip))
            for stage in stages:
                _run_stage(stage, dry_run=args.dry_run)
    else:
        args.cli_outdir.mkdir(parents=True, exist_ok=True)
        stages = _selected_stages(_stages(args.cli_outdir), only=set(args.only), skip=set(args.skip))
        for stage in stages:
            _run_stage(stage, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

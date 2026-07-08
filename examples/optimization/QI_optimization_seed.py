#!/usr/bin/env python
"""Run the diagnostic NFP=3 seed-3127 QI optimization preset.

This is a compact stress-case preset for the far-seed
``input.QI_stel_seed_3127`` lane. Public README QI rows use the bundled
``input.minimal_seed_nfp*`` decks instead. This script keeps the scientific
controls visible here, then delegates to ``QI_optimization.py`` with explicit
command-line overrides so users do not need to manually edit the larger QI
driver for the seed-3127 diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "examples" / "data"
QI_DRIVER = REPO_ROOT / "examples" / "optimization" / "QI_optimization.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.optimization.qi_optimization_cases import QI_CASES

# Edit these ordinary variables for related same-NFP seed/reference experiments.
INPUT_FILE = DATA_DIR / "input.QI_stel_seed_3127"
REFERENCE_INPUT_FILE = DATA_DIR / "input.nfp3_QI_fixed_resolution_final"
OUTPUT_DIR = Path("results/qi_opt/ess/nfp3_seed3127")
POLICY_CASE = QI_CASES["qi_stel_seed_3127"]

MAX_MODE = int(POLICY_CASE["max_mode"])
MIN_VMEC_MODE = int(POLICY_CASE["min_vmec_mode"])
BOUNDARY_REFERENCE_POLICY = POLICY_CASE["boundary_reference_preconditioner"]
TARGET_ASPECT = float(POLICY_CASE["target_aspect"])
TARGET_ABS_IOTA_MIN = float(POLICY_CASE["target_abs_iota_min"])
MAX_MIRROR_RATIO = float(POLICY_CASE["mirror_threshold"])
MAX_ELONGATION = float(BOUNDARY_REFERENCE_POLICY["max_elongation"])

METHOD = "scipy_matrix_free"  # Also try "auto_scalar" or "scalar_trust" for performance studies.
SCIPY_LSMR_MAXITER = 4  # Cap matrix-free Jv/J.Tv products per trust-region subproblem.
FTOL = 1.0e-5
GTOL = 1.0e-5
XTOL = 1.0e-6
USE_ESS = True
ALPHA = 1.2
USE_MODE_CONTINUATION = False
CONTINUATION_NFEV = 0
MAX_NFEV = int(POLICY_CASE["max_nfev"])
STAGE_MODE_POLICY = "repeat"
STAGE_REPEATS = 1

# The reference-family scan is intentionally cheap; the accepted point is then
# replayed with the normal inner budget below so final diagnostics are not taken
# from an underconverged baseline.
BOUNDARY_REFERENCE_MAX_ITER = 80
INNER_MAX_ITER = 450
INNER_FTOL = 1.0e-9
TRIAL_MAX_ITER = 450
TRIAL_FTOL = 1.0e-9
SOLVER_DEVICE = None  # Set "cpu" or "gpu" to force a backend; None inherits JAX.

REFERENCE_LAMBDAS = tuple(BOUNDARY_REFERENCE_POLICY["lambdas"])
QI_GATE_SMOOTH_MAX = float(POLICY_CASE["qi_gate_smooth_max"])
QI_GATE_LEGACY_MAX = float(POLICY_CASE["qi_gate_legacy_max"])
QI_RESOLUTION = dict(POLICY_CASE["audit_qi_resolution"])
MIRROR_SELECTION_WEIGHT = 10.0

MAKE_PLOTS = True  # Set False for a faster diagnostics-only reproduction.
DRY_RUN = False  # Set True to print the delegated QI_optimization.py command only.


def _path_arg(path: Path) -> str:
    """Return a path argument relative to the repository when possible."""

    path = Path(path)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _csv(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:.16g}" for value in values)


def boundary_reference_config() -> dict:
    """Return the same-NFP preconditioner configuration for seed 3127."""

    return {
        "enabled": True,
        "reference_input": _path_arg(REFERENCE_INPUT_FILE),
        "lambdas": list(REFERENCE_LAMBDAS),
        "keys": ["RBC", "ZBS", "RBS", "ZBC"],
        "max_mode": MAX_MODE,
        "max_iter": BOUNDARY_REFERENCE_MAX_ITER,
        "target_aspect": TARGET_ASPECT,
        "abs_iota_min": TARGET_ABS_IOTA_MIN,
        "max_mirror_ratio": MAX_MIRROR_RATIO,
        "max_elongation": MAX_ELONGATION,
        "smooth_qi_max": QI_GATE_SMOOTH_MAX,
        "legacy_qi_max": QI_GATE_LEGACY_MAX,
        "diagnostic_qi_resolution": QI_RESOLUTION,
        "mirror_selection_weight": MIRROR_SELECTION_WEIGHT,
        "prefer_non_endpoint": True,
        "prefer_lowest_qi_candidate": True,
        "accept_as_baseline": True,
    }


def write_boundary_reference_config(path: Path) -> Path:
    path.write_text(json.dumps(boundary_reference_config(), indent=2, sort_keys=True) + "\n")
    return path


def mirror_ramp_stages_config() -> list[dict]:
    """Return the reviewed guarded cleanup stages for the seed-3127 lane."""

    return [dict(stage) for stage in POLICY_CASE["mirror_ramp_stages"]]


def write_mirror_ramp_stages_config(path: Path) -> Path:
    path.write_text(json.dumps(mirror_ramp_stages_config(), indent=2, sort_keys=True) + "\n")
    return path


def build_qi_optimization_command(boundary_reference_json: Path, mirror_ramp_stages_json: Path) -> list[str]:
    """Build the delegated ``QI_optimization.py`` command."""

    command = [
        sys.executable,
        _path_arg(QI_DRIVER),
        "--input-file",
        _path_arg(INPUT_FILE),
        "--output-dir",
        str(OUTPUT_DIR),
        "--max-mode",
        str(MAX_MODE),
        "--min-vmec-mode",
        str(MIN_VMEC_MODE),
        "--method",
        METHOD,
        "--scipy-lsmr-maxiter",
        str(SCIPY_LSMR_MAXITER),
        "--ftol",
        f"{FTOL:.16g}",
        "--gtol",
        f"{GTOL:.16g}",
        "--xtol",
        f"{XTOL:.16g}",
        "--target-aspect",
        f"{TARGET_ASPECT:.16g}",
        "--target-abs-iota-min",
        f"{TARGET_ABS_IOTA_MIN:.16g}",
        "--max-mirror-ratio",
        f"{MAX_MIRROR_RATIO:.16g}",
        "--max-elongation",
        f"{MAX_ELONGATION:.16g}",
        "--no-use-simple-seed",
        "--no-use-target-helicity-seed",
        "--use-reference-family-seed",
        "--reference-input",
        _path_arg(REFERENCE_INPUT_FILE),
        "--reference-lambdas",
        _csv(REFERENCE_LAMBDAS),
        "--boundary-reference-json",
        str(boundary_reference_json),
        "--mirror-ramp-stages-json",
        str(mirror_ramp_stages_json),
        "--accept-boundary-reference-baseline",
        _bool_flag("use-mode-continuation", USE_MODE_CONTINUATION),
        "--max-nfev",
        str(MAX_NFEV),
        "--continuation-nfev",
        str(CONTINUATION_NFEV),
        "--stage-mode-policy",
        STAGE_MODE_POLICY,
        "--stage-repeats",
        str(STAGE_REPEATS),
        "--inner-max-iter",
        str(INNER_MAX_ITER),
        "--inner-ftol",
        f"{INNER_FTOL:.16g}",
        "--trial-max-iter",
        str(TRIAL_MAX_ITER),
        "--trial-ftol",
        f"{TRIAL_FTOL:.16g}",
        "--ess-alpha",
        f"{ALPHA:.16g}",
        _bool_flag("use-ess", USE_ESS),
        _bool_flag("make-plots", MAKE_PLOTS),
    ]
    if SOLVER_DEVICE is not None:
        command.extend(["--solver-device", str(SOLVER_DEVICE)])
    for prefix in ("qi", "audit-qi"):
        command.extend(
            [
                f"--{prefix}-mboz",
                str(QI_RESOLUTION["mboz"]),
                f"--{prefix}-nboz",
                str(QI_RESOLUTION["nboz"]),
                f"--{prefix}-nphi",
                str(QI_RESOLUTION["nphi"]),
                f"--{prefix}-nalpha",
                str(QI_RESOLUTION["nalpha"]),
                f"--{prefix}-n-bounce",
                str(QI_RESOLUTION["n_bounce"]),
            ]
        )
    return command


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vmec_jax_qi3127_") as tmpdir:
        boundary_reference_json = write_boundary_reference_config(Path(tmpdir) / "boundary_reference.json")
        mirror_ramp_stages_json = write_mirror_ramp_stages_config(Path(tmpdir) / "mirror_ramp_stages.json")
        command = build_qi_optimization_command(boundary_reference_json, mirror_ramp_stages_json)
        print("Running the diagnostic NFP=3 seed-3127 QI preset.")
        print(f"  raw seed:        {_path_arg(INPUT_FILE)}")
        print(f"  QI reference:    {_path_arg(REFERENCE_INPUT_FILE)}")
        print(f"  output dir:      {OUTPUT_DIR}")
        print(f"  delegated driver:{_path_arg(QI_DRIVER)}")
        print("  command:")
        print("    " + " ".join(shlex.quote(item) for item in command))
        if DRY_RUN:
            return 0
        sys.stdout.flush()
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

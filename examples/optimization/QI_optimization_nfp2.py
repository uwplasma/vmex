#!/usr/bin/env python
"""Minimal-seed NFP=2 quasi-isodynamic optimization example."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qi_minimal_seed_example_common import (
    DATA_DIR,
    MinimalSeedQIExample,
    build_qi_optimization_command,
    example_from_cli,
    run_minimal_seed_qi_example,
)


# This is the reviewed balanced NFP=2 preset used by the README QI panel.
INPUT_FILE = DATA_DIR / "input.minimal_seed_nfp2"
REFERENCE_INPUT_FILE = DATA_DIR / "input.nfp2_QI"
OUTPUT_DIR = Path("results/qi_opt/ess/minimal_nfp2_qi_balanced_mirror035")
POLICY_CASE = "minimal_nfp2_qi_balanced_mirror035"

EXAMPLE = MinimalSeedQIExample(
    nfp=2,
    policy_case=POLICY_CASE,
    input_file=INPUT_FILE,
    reference_input=REFERENCE_INPUT_FILE,
    output_dir=OUTPUT_DIR,
)
BUILD_QI_OPTIMIZATION_COMMAND = build_qi_optimization_command


if __name__ == "__main__":
    raise SystemExit(run_minimal_seed_qi_example(example_from_cli(EXAMPLE)))

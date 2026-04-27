#!/usr/bin/env python
"""Finite-beta stage-one QA fixed-boundary optimization with vmec_jax."""

from pathlib import Path

try:
    from finite_beta_stage1_common import FiniteBetaStage1Config, run_stage1
except ModuleNotFoundError:
    from examples.optimization.finite_beta_stage1_common import FiniteBetaStage1Config, run_stage1


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# User-editable parameters.
MAX_MODE = 1
MAX_NFEV = 8
CONTINUATION_NFEV = 8
USE_ESS = True
USE_MODE_CONTINUATION = True
SOLVER_DEVICE = None  # set to "cpu" or "gpu" to force one backend


CONFIG = FiniteBetaStage1Config(
    input_file=DATA_DIR / "input.nfp2_QA_finite_beta",
    output_dir=Path("results/qa_finite_beta"),
    objective_kind="qa",
    helicity_m=1,
    helicity_n=0,
    max_mode=MAX_MODE,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    vmec_mpol=5,
    vmec_ntor=5,
    target_aspect=5.0,
    min_iota=0.31,
    min_average_iota=0.33,
    max_iota=0.49,
    field_weight=1.0e3,
    use_ess=USE_ESS,
    use_mode_continuation=USE_MODE_CONTINUATION,
    solver_device=SOLVER_DEVICE,
)


if __name__ == "__main__":
    run_stage1(CONFIG)

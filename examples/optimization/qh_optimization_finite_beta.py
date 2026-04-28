#!/usr/bin/env python
"""Finite-beta stage-one QH fixed-boundary optimization with vmec_jax."""

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
INNER_MAX_ITER = 0  # 0 uses NITER from the input deck
INNER_FTOL = 0.0  # 0 uses FTOL from the input deck
TRIAL_MAX_ITER = 300
TRIAL_FTOL = 1.0e-10


CONFIG = FiniteBetaStage1Config(
    input_file=DATA_DIR / "input.nfp4_QH_finite_beta",
    output_dir=Path("results/qh_finite_beta"),
    objective_kind="qh",
    helicity_m=1,
    helicity_n=-1,
    max_mode=MAX_MODE,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    vmec_mpol=5,
    vmec_ntor=5,
    target_aspect=5.0,
    min_iota=1.02,
    min_average_iota=1.05,
    max_iota=1.9,
    field_weight=1.0e3,
    use_ess=USE_ESS,
    use_mode_continuation=USE_MODE_CONTINUATION,
    solver_device=SOLVER_DEVICE,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
)


if __name__ == "__main__":
    run_stage1(CONFIG)

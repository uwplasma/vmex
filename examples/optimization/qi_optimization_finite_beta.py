#!/usr/bin/env python
"""Finite-beta stage-one QI fixed-boundary optimization with vmec_jax."""

from pathlib import Path

try:
    from finite_beta_stage1_common import FiniteBetaStage1Config, run_stage1
except ModuleNotFoundError:
    from examples.optimization.finite_beta_stage1_common import FiniteBetaStage1Config, run_stage1


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# User-editable parameters.
MAX_MODE = 1
MAX_NFEV = 6
CONTINUATION_NFEV = 6
USE_ESS = True
USE_MODE_CONTINUATION = True
SOLVER_DEVICE = None  # set to "cpu" or "gpu" to force one backend
INNER_MAX_ITER = 0  # 0 uses NITER from the input deck
INNER_FTOL = 0.0  # 0 uses FTOL from the input deck
TRIAL_MAX_ITER = 300
TRIAL_FTOL = 1.0e-10

# Boozer/QI residual resolution. These defaults are diagnostic-friendly for a
# first run; increase them for final research-quality QI refinements.
QI_MBOZ = 10
QI_NBOZ = 10
QI_NPHI = 32
QI_NALPHA = 8
QI_N_BOUNCE = 12
QI_SOFTNESS = 30.0
QI_PROFILE_WEIGHT = 0.15


CONFIG = FiniteBetaStage1Config(
    input_file=DATA_DIR / "input.nfp4_QI_finite_beta",
    output_dir=Path("results/qi_finite_beta"),
    objective_kind="qi",
    max_mode=MAX_MODE,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    vmec_mpol=5,
    vmec_ntor=5,
    target_aspect=6.0,
    min_iota=1.04,
    min_average_iota=1.06,
    max_iota=1.9,
    field_weight=2.0e5,
    use_ess=USE_ESS,
    use_mode_continuation=USE_MODE_CONTINUATION,
    solver_device=SOLVER_DEVICE,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    qi_mboz=QI_MBOZ,
    qi_nboz=QI_NBOZ,
    qi_nphi=QI_NPHI,
    qi_nalpha=QI_NALPHA,
    qi_n_bounce=QI_N_BOUNCE,
    qi_softness=QI_SOFTNESS,
    qi_profile_weight=QI_PROFILE_WEIGHT,
)


if __name__ == "__main__":
    run_stage1(CONFIG)

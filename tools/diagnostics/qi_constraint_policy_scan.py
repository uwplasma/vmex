#!/usr/bin/env python
"""Bounded QI constraint-policy scan for ``input.QI_stel_seed_3127``.

This helper intentionally reuses the same public workflow as
``examples/optimization/QI_optimization.py`` while keeping the scan small:
low Boozer/QI grids, explicit ``max_nfev`` caps, and a short policy matrix.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "examples" / "data"
DEFAULT_INPUT = DATA_DIR / "input.QI_stel_seed_3127"
DEFAULT_OUT_ROOT = Path("/tmp/vmec_jax_qi_constraint_policy_scan")
TARGET_ASPECT = 10.0
TARGET_ABS_IOTA_MIN = 0.41
MAX_ELONGATION = 8.0
QI_GATE_SMOOTH_MAX = 2.0e-3
QI_GATE_LEGACY_MAX = 2.0e-3


@dataclass(frozen=True)
class ScanResolution:
    surfaces: tuple[float, ...] = (0.35, 0.7, 1.0)
    mboz: int = 8
    nboz: int = 8
    nphi: int = 41
    nalpha: int = 9
    n_bounce: int = 11
    mirror_ntheta: int = 32
    mirror_nphi: int = 32
    elongation_ntheta: int = 24
    elongation_nphi: int = 8


@dataclass(frozen=True)
class StagePolicy:
    name: str
    method: str = "scipy"
    max_nfev: int = 2
    stage_modes: tuple[Any, ...] = (3,)
    vmec_max_mode: int | None = None
    min_vmec_mode: int | None = None
    target_aspect: float = TARGET_ASPECT
    aspect_weight: float = 0.25
    iota_abs_min: float | None = TARGET_ABS_IOTA_MIN
    target_iota: float | None = None
    target_iota_weight: float = 0.0
    iota_weight: float = 200.0**2
    iota_abs_max: float | None = None
    iota_ceiling_weight: float = 0.0
    qi_weight: float = 10.0
    boozer_target_wout: str | None = None
    boozer_target_weight: float = 0.0
    boozer_target_normalize: bool = True
    boozer_target_include_b00: bool = False
    mirror_threshold: float = 0.21
    promotion_mirror_threshold: float | None = None
    mirror_surface_index: int | None = None
    mirror_weight: float = 0.0
    elongation_weight: float = 0.0
    use_augmented_lagrangian: bool = False
    al_mirror_multiplier: float = 0.0
    al_mirror_penalty: float = 1.0
    al_elongation_multiplier: float = 0.0
    al_elongation_penalty: float = 1.0
    qi_ceiling_weight: float = 0.0
    qi_ceiling_max: float = 2.0e-3
    qi_ceiling_smooth_penalty: float = 2.0e-3
    branch_width_weight: float = 0.5
    weighted_shuffle_profile_weight: float = 0.0
    continue_if_qi_aspect_pass: bool = False
    scalar_step_bound: float | None = None
    lbfgs_step_bound: float | None = None
    continue_on_failure: bool = False


@dataclass(frozen=True)
class Policy:
    name: str
    description: str
    stages: tuple[StagePolicy, ...]


def default_policies(*, max_nfev: int = 2) -> tuple[Policy, ...]:
    """Return the bounded policy matrix requested for the seed-robustness probe."""

    return (
        Policy(
            "scipy_qi_iota",
            "Baseline SciPy trust-region QI+iota/aspect objective; no engineering cleanup.",
            (StagePolicy("qi_iota", method="scipy", max_nfev=max_nfev),),
        ),
        Policy(
            "scalar_trust_qi_iota",
            "Scalar-adjoint safeguarded trust probe with the same QI+iota/aspect objective.",
            (StagePolicy("qi_iota", method="scalar_trust", max_nfev=max_nfev),),
        ),
        Policy(
            "matrix_free_qi_iota",
            "SciPy matrix-free trust-region probe with the same QI+iota/aspect objective.",
            (StagePolicy("qi_iota", method="scipy_matrix_free", max_nfev=max_nfev),),
        ),
        Policy(
            "large_mirror_weights",
            "Direct hard mirror cleanup pressure with large mirror and elongation weights.",
            (
                StagePolicy(
                    "large_mirror",
                    method="scipy",
                    max_nfev=max_nfev,
                    mirror_threshold=0.21,
                    mirror_weight=50.0,
                    elongation_weight=10.0,
                ),
            ),
        ),
        Policy(
            "staged_mirror_relax_tight",
            "Two short cleanup stages: relaxed mirror threshold, then tighter mirror threshold.",
            (
                StagePolicy(
                    "mirror_relaxed",
                    method="scipy",
                    max_nfev=max(1, max_nfev),
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.35,
                    mirror_weight=10.0,
                    elongation_weight=5.0,
                    qi_ceiling_weight=100.0,
                    qi_ceiling_max=2.0e-2,
                ),
                StagePolicy(
                    "mirror_tight",
                    method="scipy",
                    max_nfev=max(1, max_nfev),
                    mirror_threshold=0.21,
                    promotion_mirror_threshold=0.30,
                    mirror_weight=30.0,
                    elongation_weight=10.0,
                    qi_ceiling_weight=250.0,
                    qi_ceiling_max=1.0e-2,
                ),
            ),
        ),
        Policy(
            "softplus_barriers",
            "Smooth softplus QI ceiling plus moderate mirror/elongation penalties.",
            (
                StagePolicy(
                    "softplus_barrier",
                    method="scipy",
                    max_nfev=max_nfev,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_weight=15.0,
                    elongation_weight=5.0,
                    qi_ceiling_weight=250.0,
                    qi_ceiling_max=2.0e-2,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "balanced_qi_mirror",
            "High-QI-weight mirror cleanup: preserve the QI basin while applying stronger mirror pressure.",
            (
                StagePolicy(
                    "balanced_qi_mirror",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=250.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_weight=20.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "balanced_qi_mirror_tight",
            "Higher mirror pressure with the same QI ceiling, used to check whether mirror can be lowered before QI fails.",
            (
                StagePolicy(
                    "balanced_qi_mirror_tight",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=250.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "balanced_qi500_mirror_tight",
            "Same tight mirror policy with doubled QI weight to test whether the low-mirror basin can pass QI gates.",
            (
                StagePolicy(
                    "balanced_qi500_mirror_tight",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=500.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "balanced_qi1000_mirror_tight",
            "Same tight mirror policy with quadrupled QI weight for the high-QI/low-mirror tradeoff check.",
            (
                StagePolicy(
                    "balanced_qi1000_mirror_tight",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=1000.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "target_lcfs_mirror_balanced",
            "Target only the LCFS mirror ratio while preserving the all-surface QI basin.",
            (
                StagePolicy(
                    "target_lcfs_mirror_balanced",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=500.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "target_lcfs_mirror_strong",
            "Higher QI weight plus LCFS-only mirror cleanup to test a less stiff engineering constraint.",
            (
                StagePolicy(
                    "target_lcfs_mirror_strong",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=50.0**2,
                    qi_weight=1000.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "target_lcfs_mirror_iota_guard",
            "LCFS-only mirror cleanup with stronger iota preservation for repeated cleanup passes.",
            (
                StagePolicy(
                    "target_lcfs_mirror_iota_guard",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=100.0**2,
                    qi_weight=1000.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "target_lcfs_mirror_high_iota_guard",
            "LCFS-only mirror cleanup with high QI and iota weights to probe the conservative tradeoff boundary.",
            (
                StagePolicy(
                    "target_lcfs_mirror_high_iota_guard",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=150.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=30.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=6.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "all_surface_mirror_ramp_iota_guard",
            "All-surface mirror ramp with strong QI and iota guards; tests whether mirror can be lowered without LCFS-only bias.",
            (
                StagePolicy(
                    "mirror_ramp_060",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=100.0**2,
                    qi_weight=1000.0,
                    mirror_threshold=0.60,
                    promotion_mirror_threshold=0.70,
                    mirror_surface_index=None,
                    mirror_weight=15.0,
                    elongation_weight=1.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=5.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
                StagePolicy(
                    "mirror_ramp_052",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=100.0**2,
                    qi_weight=1250.0,
                    mirror_threshold=0.52,
                    promotion_mirror_threshold=0.60,
                    mirror_surface_index=None,
                    mirror_weight=20.0,
                    elongation_weight=1.5,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=5.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
                StagePolicy(
                    "mirror_ramp_050",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=100.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=None,
                    mirror_weight=25.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "al_all_surface_mirror_iota_guard",
            "All-surface augmented-Lagrangian mirror cleanup with QI and iota guards.",
            (
                StagePolicy(
                    "al_all_surface_mirror",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=125.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=None,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=0.0,
                    al_mirror_penalty=500.0,
                    al_elongation_multiplier=0.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "al_lcfs_mirror_iota_guard",
            "LCFS augmented-Lagrangian mirror cleanup with QI and iota guards.",
            (
                StagePolicy(
                    "al_lcfs_mirror",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=125.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=0.0,
                    al_mirror_penalty=500.0,
                    al_elongation_multiplier=0.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "al_lcfs_mirror_scalar_escape",
            "Scalar-adjoint LCFS augmented-Lagrangian cleanup with a larger safeguarded step.",
            (
                StagePolicy(
                    "al_lcfs_scalar_escape",
                    method="scalar_trust",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=125.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=0.0,
                    al_mirror_penalty=500.0,
                    al_elongation_multiplier=0.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    scalar_step_bound=2.0e-2,
                ),
            ),
        ),
        Policy(
            "al_lcfs_mirror_lbfgs_escape",
            "L-BFGS LCFS augmented-Lagrangian cleanup to test whether trust-region linearization is limiting mirror progress.",
            (
                StagePolicy(
                    "al_lcfs_lbfgs_escape",
                    method="lbfgs_adjoint",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=125.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=0.0,
                    al_mirror_penalty=500.0,
                    al_elongation_multiplier=0.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    lbfgs_step_bound=2.0e-2,
                ),
            ),
        ),
        Policy(
            "al_lcfs_mirror_iota_window",
            "LCFS augmented-Lagrangian mirror cleanup with both lower and upper |iota| guards for low-mirror basin recovery.",
            (
                StagePolicy(
                    "al_lcfs_iota_window",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=125.0**2,
                    iota_abs_max=0.65,
                    iota_ceiling_weight=125.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=0.0,
                    al_mirror_penalty=500.0,
                    al_elongation_multiplier=0.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "target_lcfs_mirror_iota_window",
            "Soft LCFS mirror cleanup with lower and upper |iota| guards for low-mirror basin recovery.",
            (
                StagePolicy(
                    "target_lcfs_iota_window",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=125.0**2,
                    iota_abs_max=0.65,
                    iota_ceiling_weight=125.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "al_lcfs_mirror_iota_window_scalar",
            "Small-step scalar-trust AL mirror recovery with an |iota| window for rugged low-mirror basins.",
            (
                StagePolicy(
                    "al_lcfs_iota_window_scalar",
                    method="scalar_trust",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=125.0**2,
                    iota_abs_max=0.65,
                    iota_ceiling_weight=125.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=0.0,
                    al_mirror_penalty=500.0,
                    al_elongation_multiplier=0.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    scalar_step_bound=5.0e-3,
                ),
            ),
        ),
        Policy(
            "target_lcfs_mirror_iota_window_scalar",
            "Small-step scalar-trust soft mirror recovery with an |iota| window for rugged low-mirror basins.",
            (
                StagePolicy(
                    "target_lcfs_iota_window_scalar",
                    method="scalar_trust",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    iota_weight=125.0**2,
                    iota_abs_max=0.65,
                    iota_ceiling_weight=125.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    scalar_step_bound=5.0e-3,
                ),
            ),
        ),
        Policy(
            "al_lcfs_mirror_iota_target",
            "LCFS AL mirror recovery with an explicit signed iota target for the known negative-transform QI branch.",
            (
                StagePolicy(
                    "al_lcfs_iota_target",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    target_iota=-0.41,
                    target_iota_weight=250.0**2,
                    iota_weight=50.0**2,
                    iota_abs_max=0.65,
                    iota_ceiling_weight=250.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=0.0,
                    al_mirror_penalty=500.0,
                    al_elongation_multiplier=0.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "target_lcfs_mirror_iota_target",
            "Soft LCFS mirror recovery with an explicit signed iota target for the known negative-transform QI branch.",
            (
                StagePolicy(
                    "target_lcfs_iota_target",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.25,
                    target_iota=-0.41,
                    target_iota_weight=250.0**2,
                    iota_weight=50.0**2,
                    iota_abs_max=0.65,
                    iota_ceiling_weight=250.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=40.0,
                    elongation_weight=2.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "iota_homotopy_mirror_preserve",
            (
                "Exploratory signed-iota homotopy for low-mirror basins: ramp "
                "the transform toward -0.41 while preserving mirror before the "
                "final QI/iota cleanup."
            ),
            (
                StagePolicy(
                    "iota020_mirror",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.10,
                    iota_abs_min=0.18,
                    target_iota=-0.20,
                    target_iota_weight=75.0**2,
                    iota_weight=50.0**2,
                    iota_abs_max=0.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=300.0,
                    mirror_threshold=0.45,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=300.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=500.0,
                    qi_ceiling_max=2.0e-2,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "iota030_mirror",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.10,
                    iota_abs_min=0.28,
                    target_iota=-0.30,
                    target_iota_weight=100.0**2,
                    iota_weight=75.0**2,
                    iota_abs_max=0.45,
                    iota_ceiling_weight=75.0**2,
                    qi_weight=700.0,
                    mirror_threshold=0.48,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=400.0,
                    al_elongation_penalty=35.0,
                    qi_ceiling_weight=1000.0,
                    qi_ceiling_max=1.0e-2,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "iota041_mirror",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.10,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    target_iota=-0.41,
                    target_iota_weight=150.0**2,
                    iota_weight=100.0**2,
                    iota_abs_max=0.60,
                    iota_ceiling_weight=125.0**2,
                    qi_weight=1200.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=500.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=2500.0,
                    qi_ceiling_max=5.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "qi_cleanup",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    aspect_weight=0.10,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    target_iota=-0.41,
                    target_iota_weight=200.0**2,
                    iota_weight=100.0**2,
                    iota_abs_max=0.60,
                    iota_ceiling_weight=125.0**2,
                    qi_weight=1800.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=500.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "mode4_iota_homotopy_mirror_preserve",
            (
                "Same signed-iota homotopy with mode-4 boundary DOFs enabled, "
                "used to test whether the QI/mirror tradeoff is a mode-3 "
                "resolution limit."
            ),
            tuple(
                StagePolicy(
                    **{
                        **asdict(stage),
                        "name": f"mode4_{stage.name}",
                        "stage_modes": (4,),
                        "vmec_max_mode": 4,
                        "min_vmec_mode": 7,
                    }
                )
                for stage in (
                    StagePolicy(
                        "iota020_mirror",
                        method="scipy_matrix_free",
                        max_nfev=max(1, max_nfev // 2),
                        aspect_weight=0.10,
                        iota_abs_min=0.18,
                        target_iota=-0.20,
                        target_iota_weight=75.0**2,
                        iota_weight=50.0**2,
                        iota_abs_max=0.35,
                        iota_ceiling_weight=50.0**2,
                        qi_weight=300.0,
                        mirror_threshold=0.45,
                        promotion_mirror_threshold=0.50,
                        mirror_surface_index=-1,
                        mirror_weight=1.0,
                        elongation_weight=1.0,
                        use_augmented_lagrangian=True,
                        al_mirror_penalty=300.0,
                        al_elongation_penalty=25.0,
                        qi_ceiling_weight=500.0,
                        qi_ceiling_max=2.0e-2,
                        continue_on_failure=True,
                    ),
                    StagePolicy(
                        "iota041_mirror",
                        method="scipy_matrix_free",
                        max_nfev=max(1, max_nfev // 2),
                        aspect_weight=0.10,
                        iota_abs_min=TARGET_ABS_IOTA_MIN,
                        target_iota=-0.41,
                        target_iota_weight=150.0**2,
                        iota_weight=100.0**2,
                        iota_abs_max=0.60,
                        iota_ceiling_weight=125.0**2,
                        qi_weight=1200.0,
                        mirror_threshold=0.50,
                        promotion_mirror_threshold=0.50,
                        mirror_surface_index=-1,
                        mirror_weight=1.0,
                        elongation_weight=1.0,
                        use_augmented_lagrangian=True,
                        al_mirror_penalty=500.0,
                        al_elongation_penalty=50.0,
                        qi_ceiling_weight=2500.0,
                        qi_ceiling_max=5.0e-3,
                        continue_on_failure=True,
                    ),
                    StagePolicy(
                        "qi_cleanup",
                        method="scipy_matrix_free",
                        max_nfev=max(1, max_nfev // 2),
                        aspect_weight=0.10,
                        iota_abs_min=TARGET_ABS_IOTA_MIN,
                        target_iota=-0.41,
                        target_iota_weight=200.0**2,
                        iota_weight=100.0**2,
                        iota_abs_max=0.60,
                        iota_ceiling_weight=125.0**2,
                        qi_weight=1800.0,
                        mirror_threshold=0.50,
                        promotion_mirror_threshold=0.50,
                        mirror_surface_index=-1,
                        mirror_weight=1.0,
                        elongation_weight=1.0,
                        use_augmented_lagrangian=True,
                        al_mirror_penalty=500.0,
                        al_elongation_penalty=50.0,
                        qi_ceiling_weight=3000.0,
                        qi_ceiling_max=4.0e-3,
                    ),
                )
            ),
        ),
        Policy(
            "mode4_small_step_qi_recovery",
            (
                "Small-step mode-4 QI cleanup for low-mirror mode-4 states; "
                "uses scalar and L-BFGS adjoint paths to avoid failed "
                "matrix-free trust steps."
            ),
            (
                StagePolicy(
                    "mode4_scalar_qi_cleanup",
                    method="scalar_trust",
                    max_nfev=max_nfev,
                    stage_modes=(4,),
                    vmec_max_mode=4,
                    min_vmec_mode=7,
                    aspect_weight=0.10,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    target_iota=-0.41,
                    target_iota_weight=200.0**2,
                    iota_weight=100.0**2,
                    iota_abs_max=0.60,
                    iota_ceiling_weight=125.0**2,
                    qi_weight=2000.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=500.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    scalar_step_bound=1.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "mode4_lbfgs_qi_cleanup",
                    method="lbfgs_adjoint",
                    max_nfev=max_nfev,
                    stage_modes=(4,),
                    vmec_max_mode=4,
                    min_vmec_mode=7,
                    aspect_weight=0.10,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    target_iota=-0.41,
                    target_iota_weight=200.0**2,
                    iota_weight=100.0**2,
                    iota_abs_max=0.60,
                    iota_ceiling_weight=125.0**2,
                    qi_weight=2500.0,
                    mirror_threshold=0.50,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=-1,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=500.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    lbfgs_step_bound=1.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_reference_like_qi_mirror",
            (
                "Mode-4 NFP=3 reference-like QI policy: lower aspect target, "
                "no upper-iota ceiling, and mirror cleanup near the known "
                "Goodman-style NFP=3 QI basin."
            ),
            (
                StagePolicy(
                    "aspect4_qi_basin",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=(4,),
                    vmec_max_mode=4,
                    min_vmec_mode=7,
                    target_aspect=4.0,
                    aspect_weight=1.0,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=1500.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.35,
                    mirror_surface_index=None,
                    mirror_weight=0.0,
                    elongation_weight=0.0,
                    qi_ceiling_weight=1000.0,
                    qi_ceiling_max=8.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "aspect4_mirror_cleanup",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=(4,),
                    vmec_max_mode=4,
                    min_vmec_mode=7,
                    target_aspect=4.0,
                    aspect_weight=1.0,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=2000.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.35,
                    mirror_surface_index=None,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=500.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_boozer_target_homotopy",
            (
                "Reference-guided homotopy for the NFP=3 seed: first approach "
                "the known NFP=3 QI Boozer spectrum, then release the target "
                "and polish QI/mirror with bounded local steps."
            ),
            (
                StagePolicy(
                    "target_reference_boozer",
                    method="scalar_trust",
                    max_nfev=max_nfev,
                    stage_modes=(4,),
                    vmec_max_mode=4,
                    min_vmec_mode=7,
                    target_aspect=4.0,
                    aspect_weight=0.25,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=50.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=100.0,
                    boozer_target_wout="examples/data/wout_nfp3_QI_fixed_resolution_final.nc",
                    boozer_target_weight=500.0,
                    mirror_threshold=0.40,
                    promotion_mirror_threshold=0.50,
                    mirror_surface_index=None,
                    mirror_weight=0.0,
                    elongation_weight=0.0,
                    qi_ceiling_weight=200.0,
                    qi_ceiling_max=3.0e-2,
                    scalar_step_bound=5.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "mixed_reference_qi_mirror",
                    method="scalar_trust",
                    max_nfev=max_nfev,
                    stage_modes=(4,),
                    vmec_max_mode=4,
                    min_vmec_mode=7,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=75.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=75.0**2,
                    qi_weight=1000.0,
                    boozer_target_wout="examples/data/wout_nfp3_QI_fixed_resolution_final.nc",
                    boozer_target_weight=100.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.40,
                    mirror_surface_index=None,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=300.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=1500.0,
                    qi_ceiling_max=1.0e-2,
                    scalar_step_bound=3.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "release_reference_qi_mirror",
                    method="lbfgs_adjoint",
                    max_nfev=max_nfev,
                    stage_modes=(4,),
                    vmec_max_mode=4,
                    min_vmec_mode=7,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=75.0**2,
                    qi_weight=2000.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.35,
                    mirror_surface_index=None,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=500.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=3000.0,
                    qi_ceiling_max=4.0e-3,
                    lbfgs_step_bound=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_reference_scalar_polish",
            (
                "Small-step scalar/L-BFGS polish for NFP=3 states that already "
                "satisfy mirror, elongation, aspect, and iota gates but need "
                "lower smooth/legacy QI residuals."
            ),
            (
                StagePolicy(
                    "scalar_qi_polish",
                    method="scalar_trust",
                    max_nfev=max_nfev,
                    stage_modes=(4,),
                    vmec_max_mode=4,
                    min_vmec_mode=7,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=3000.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.35,
                    mirror_surface_index=None,
                    mirror_weight=0.25,
                    elongation_weight=0.25,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=200.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=2.0e-3,
                    scalar_step_bound=2.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "lbfgs_qi_polish",
                    method="lbfgs_adjoint",
                    max_nfev=max_nfev,
                    stage_modes=(4,),
                    vmec_max_mode=4,
                    min_vmec_mode=7,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=4000.0,
                    mirror_threshold=0.35,
                    promotion_mirror_threshold=0.35,
                    mirror_surface_index=None,
                    mirror_weight=0.25,
                    elongation_weight=0.25,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=200.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=2.0e-3,
                    lbfgs_step_bound=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_mode5_toroidal_first_local",
            (
                "Mode-5 local mirror cleanup from an NFP=3 QI candidate: "
                "unlock toroidal structure first, then the full mode-5 boundary."
            ),
            (
                StagePolicy(
                    "mode5_nfirst",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 5, "max_m": 1, "max_n": 5, "label": "nfirst"},),
                    vmec_max_mode=5,
                    min_vmec_mode=8,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=3000.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.50,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=200.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=2.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "mode5_full",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 5, "max_m": 5, "max_n": 5, "label": "full"},),
                    vmec_max_mode=5,
                    min_vmec_mode=8,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=3500.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.75,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=250.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=6000.0,
                    qi_ceiling_max=1.5e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_mode5_toroidal_first_soft_probe",
            (
                "Single-stage mode-5 toroidal-first probe with the lighter "
                "QI-ceiling/mirror cleanup style used by QI_optimization.py."
            ),
            (
                StagePolicy(
                    "mode5_nfirst_soft",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 5, "max_m": 1, "max_n": 5, "label": "nfirst"},),
                    vmec_max_mode=5,
                    min_vmec_mode=8,
                    target_aspect=4.0,
                    aspect_weight=0.05,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=50.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=25.0**2,
                    qi_weight=500.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=8.0,
                    elongation_weight=0.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=5.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_mode5_poltor_soft_probe",
            (
                "Single-stage mode-5 poloidal/toroidal anisotropic probe with "
                "moderate high-mode freedom in both directions."
            ),
            (
                StagePolicy(
                    "mode5_m3n5_soft",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 5, "max_m": 3, "max_n": 5, "label": "m3n5"},),
                    vmec_max_mode=5,
                    min_vmec_mode=8,
                    target_aspect=4.0,
                    aspect_weight=0.05,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=50.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=25.0**2,
                    qi_weight=500.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=8.0,
                    elongation_weight=0.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=5.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_mode6_toroidal_first_local",
            (
                "Mode-6 local mirror cleanup from an NFP=3 QI candidate: "
                "unlock toroidal structure first, then the full mode-6 boundary."
            ),
            (
                StagePolicy(
                    "mode6_nfirst",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 6, "max_m": 1, "max_n": 6, "label": "nfirst"},),
                    vmec_max_mode=6,
                    min_vmec_mode=9,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=3000.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.50,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=200.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=2.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "mode6_full",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 6, "max_m": 6, "max_n": 6, "label": "full"},),
                    vmec_max_mode=6,
                    min_vmec_mode=9,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=3500.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.75,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=250.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=6000.0,
                    qi_ceiling_max=1.5e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_mode6_toroidal_first_soft_probe",
            (
                "Single-stage mode-6 toroidal-first probe with the lighter "
                "QI-ceiling/mirror cleanup style used by QI_optimization.py."
            ),
            (
                StagePolicy(
                    "mode6_nfirst_soft",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 6, "max_m": 1, "max_n": 6, "label": "nfirst"},),
                    vmec_max_mode=6,
                    min_vmec_mode=9,
                    target_aspect=4.0,
                    aspect_weight=0.05,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=50.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=25.0**2,
                    qi_weight=500.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=8.0,
                    elongation_weight=0.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=5.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_mode6_poltor_soft_probe",
            (
                "Single-stage mode-6 poloidal/toroidal anisotropic probe with "
                "moderate high-mode freedom in both directions."
            ),
            (
                StagePolicy(
                    "mode6_m3n6_soft",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 6, "max_m": 3, "max_n": 6, "label": "m3n6"},),
                    vmec_max_mode=6,
                    min_vmec_mode=9,
                    target_aspect=4.0,
                    aspect_weight=0.05,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=50.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=25.0**2,
                    qi_weight=500.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=8.0,
                    elongation_weight=0.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=5.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_mode5_poltor_anisotropic_local",
            (
                "Mode-5 local anisotropic cleanup: compare poloidal-heavy and "
                "toroidal-heavy boundary subsets before the full mode-5 stage."
            ),
            (
                StagePolicy(
                    "mode5_mheavy",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 5, "max_m": 5, "max_n": 2, "label": "mheavy"},),
                    vmec_max_mode=5,
                    min_vmec_mode=8,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=3000.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.50,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=200.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=2.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "mode5_nheavy",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 5, "max_m": 2, "max_n": 5, "label": "nheavy"},),
                    vmec_max_mode=5,
                    min_vmec_mode=8,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=3500.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.75,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=250.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=6000.0,
                    qi_ceiling_max=1.5e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "mode5_full",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 5, "max_m": 5, "max_n": 5, "label": "full"},),
                    vmec_max_mode=5,
                    min_vmec_mode=8,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=4000.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.75,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=250.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=6000.0,
                    qi_ceiling_max=1.5e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "nfp3_mode6_poltor_anisotropic_local",
            (
                "Mode-6 local anisotropic cleanup: compare poloidal-heavy and "
                "toroidal-heavy boundary subsets before the full mode-6 stage."
            ),
            (
                StagePolicy(
                    "mode6_mheavy",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 6, "max_m": 6, "max_n": 2, "label": "mheavy"},),
                    vmec_max_mode=6,
                    min_vmec_mode=9,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=3000.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.50,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=200.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=5000.0,
                    qi_ceiling_max=2.0e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "mode6_nheavy",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 6, "max_m": 2, "max_n": 6, "label": "nheavy"},),
                    vmec_max_mode=6,
                    min_vmec_mode=9,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=3500.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.75,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=250.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=6000.0,
                    qi_ceiling_max=1.5e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                    continue_on_failure=True,
                ),
                StagePolicy(
                    "mode6_full",
                    method="scipy_matrix_free",
                    max_nfev=max_nfev,
                    stage_modes=({"mode": 6, "max_m": 6, "max_n": 6, "label": "full"},),
                    vmec_max_mode=6,
                    min_vmec_mode=9,
                    target_aspect=4.0,
                    aspect_weight=0.50,
                    iota_abs_min=TARGET_ABS_IOTA_MIN,
                    iota_weight=100.0**2,
                    iota_abs_max=1.35,
                    iota_ceiling_weight=50.0**2,
                    qi_weight=4000.0,
                    mirror_threshold=0.30,
                    promotion_mirror_threshold=0.30,
                    mirror_surface_index=None,
                    mirror_weight=0.75,
                    elongation_weight=0.50,
                    use_augmented_lagrangian=True,
                    al_mirror_penalty=250.0,
                    al_elongation_penalty=25.0,
                    qi_ceiling_weight=6000.0,
                    qi_ceiling_max=1.5e-3,
                    qi_ceiling_smooth_penalty=2.0e-3,
                ),
            ),
        ),
        Policy(
            "augmented_lagrangian_mirror",
            "Projected augmented-Lagrangian mirror/elongation constraints with a QI ceiling guard.",
            (
                StagePolicy(
                    "al_mirror_elongation",
                    method="scalar_trust",
                    max_nfev=max(2, max_nfev),
                    mirror_threshold=0.75,
                    promotion_mirror_threshold=0.75,
                    mirror_weight=1.0,
                    elongation_weight=1.0,
                    use_augmented_lagrangian=True,
                    al_mirror_multiplier=20.0,
                    al_mirror_penalty=400.0,
                    al_elongation_multiplier=5.0,
                    al_elongation_penalty=50.0,
                    qi_ceiling_weight=1000.0,
                    qi_ceiling_max=2.0e-3,
                ),
            ),
        ),
        Policy(
            "mode_continuation_repeat",
            "Cheap mode-continuation repeat: mode 2 warmup followed by two mode 3 passes.",
            (
                StagePolicy(
                    "mode_repeat",
                    method="scipy",
                    max_nfev=max(1, max_nfev),
                    stage_modes=(2, 3, 3),
                ),
            ),
        ),
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def _ensure_repo_on_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _build_qi_options(vj: Any, resolution: ScanResolution, stage: StagePolicy):
    return vj.QuasiIsodynamicOptions(
        surfaces=np.asarray(resolution.surfaces, dtype=float),
        mboz=int(resolution.mboz),
        nboz=int(resolution.nboz),
        nphi=int(resolution.nphi),
        nalpha=int(resolution.nalpha),
        n_bounce=int(resolution.n_bounce),
        include_bounce_endpoints=True,
        softness=2.0e-2,
        width_weight=1.0,
        branch_width_weight=float(stage.branch_width_weight),
        branch_width_softness=2.0e-2,
        profile_weight=0.1,
        shuffle_profile_weight=1.0,
        shuffle_profile_softness=2.0e-2,
        weighted_shuffle_profile_weight=float(stage.weighted_shuffle_profile_weight),
        weighted_shuffle_profile_softness=2.0e-2,
        phimin=0.0,
        jit_booz=True,
    )


def _make_problem(vj: Any, resolution: ScanResolution, stage: StagePolicy):
    qi_options = _build_qi_options(vj, resolution, stage)
    tuples = [
        (vj.AspectRatio().J, float(stage.target_aspect), float(stage.aspect_weight)),
        (vj.QuasiIsodynamicResidual(qi_options).J, 0.0, float(stage.qi_weight)),
    ]
    if stage.iota_abs_min is not None and stage.iota_weight > 0.0:
        tuples.insert(
            1,
            (
                vj.AbsMeanIotaFloor(float(stage.iota_abs_min)).J,
                0.0,
                float(stage.iota_weight),
            ),
        )
    if stage.target_iota is not None and stage.target_iota_weight > 0.0:
        tuples.append((vj.MeanIota().J, float(stage.target_iota), float(stage.target_iota_weight)))
    if stage.iota_abs_max is not None and stage.iota_ceiling_weight > 0.0:
        tuples.append(
            (
                vj.AbsMeanIotaCeiling(float(stage.iota_abs_max)).J,
                0.0,
                float(stage.iota_ceiling_weight),
            )
        )
    if stage.boozer_target_wout is not None and stage.boozer_target_weight > 0.0:
        target_path = Path(stage.boozer_target_wout)
        if not target_path.is_absolute():
            target_path = REPO_ROOT / target_path
        target = vj.boozer_b_target_from_wout(
            target_path,
            surfaces=qi_options.surfaces,
            mboz=qi_options.mboz,
            nboz=qi_options.nboz,
        )
        tuples.insert(
            2,
            (
                vj.BoozerBTarget(
                    target_bmnc=target["bmnc_b"],
                    target_bmns=target["bmns_b"],
                    normalize=bool(stage.boozer_target_normalize),
                    include_b00=bool(stage.boozer_target_include_b00),
                    qi_options=qi_options,
                ).J,
                0.0,
                float(stage.boozer_target_weight),
            ),
        )
    if stage.qi_ceiling_weight > 0.0:
        tuples.append(
            (
                vj.QuasiIsodynamicResidualCeiling(
                    maximum=float(stage.qi_ceiling_max),
                    smooth_penalty=float(stage.qi_ceiling_smooth_penalty),
                    qi_options=qi_options,
                ).J,
                0.0,
                float(stage.qi_ceiling_weight),
            )
        )
    if stage.mirror_weight > 0.0:
        mirror = vj.MirrorRatio(
            threshold=float(stage.mirror_threshold),
            ntheta=int(resolution.mirror_ntheta),
            nphi=int(resolution.mirror_nphi),
            surface_index=stage.mirror_surface_index,
            phimin=0.0,
            smooth_extrema=2.0e-2,
            smooth_penalty=2.0e-2,
            qi_options=qi_options,
        )
        if stage.use_augmented_lagrangian:
            mirror = vj.AugmentedLagrangianConstraint(
                mirror,
                multiplier=float(stage.al_mirror_multiplier),
                penalty=float(stage.al_mirror_penalty),
                softness=2.0e-2,
                name="al_mirror_ratio",
            )
        tuples.append(
            (
                mirror.J,
                0.0,
                float(stage.mirror_weight),
            )
        )
    if stage.elongation_weight > 0.0:
        elongation = vj.MaxElongation(
            threshold=MAX_ELONGATION,
            ntheta=int(resolution.elongation_ntheta),
            nphi=int(resolution.elongation_nphi),
            smooth_extrema=2.0e-2,
            smooth_penalty=2.0e-2,
            qi_options=qi_options,
        )
        if stage.use_augmented_lagrangian:
            elongation = vj.AugmentedLagrangianConstraint(
                elongation,
                multiplier=float(stage.al_elongation_multiplier),
                penalty=float(stage.al_elongation_penalty),
                softness=2.0e-2,
                name="al_max_elongation",
            )
        tuples.append(
            (
                elongation.J,
                0.0,
                float(stage.elongation_weight),
            )
        )
    return vj.LeastSquaresProblem.from_tuples(tuples), qi_options


def _diagnose(
    vj: Any,
    result: Any,
    resolution: ScanResolution,
    qi_options: Any,
    *,
    target_aspect: float,
    abs_iota_min: float | None,
    mirror_threshold: float,
) -> dict[str, Any]:
    from vmec_jax.quasi_isodynamic.diagnostics import QISeedSuitabilityTargets, annotate_qi_seed_suitability

    options = vj.QIDiagnosticOptions(
        surfaces=np.asarray(resolution.surfaces, dtype=float),
        mboz=int(resolution.mboz),
        nboz=int(resolution.nboz),
        nphi=int(resolution.nphi),
        nalpha=int(resolution.nalpha),
        n_bounce=int(resolution.n_bounce),
        include_bounce_endpoints=True,
        softness=qi_options.softness,
        width_weight=qi_options.width_weight,
        branch_width_weight=qi_options.branch_width_weight,
        branch_width_softness=qi_options.branch_width_softness,
        profile_weight=qi_options.profile_weight,
        shuffle_profile_weight=qi_options.shuffle_profile_weight,
        shuffle_profile_softness=qi_options.shuffle_profile_softness,
        weighted_shuffle_profile_weight=qi_options.weighted_shuffle_profile_weight,
        weighted_shuffle_profile_softness=qi_options.weighted_shuffle_profile_softness,
        phimin=0.0,
        mirror_threshold=float(mirror_threshold),
        mirror_ntheta=int(resolution.mirror_ntheta),
        mirror_nphi=int(resolution.mirror_nphi),
        elongation_threshold=MAX_ELONGATION,
    )
    diagnostics = vj.qi_diagnostics_from_state(
        state=result.final_state,
        static=result.final_optimizer.static,
        indata=result.final_optimizer.indata,
        signgs=result.final_optimizer.signgs,
        surfaces=np.asarray(resolution.surfaces, dtype=float),
        options=options,
    )
    return annotate_qi_seed_suitability(
        diagnostics,
        targets=QISeedSuitabilityTargets(
            smooth_qi_max=QI_GATE_SMOOTH_MAX,
            legacy_qi_max=QI_GATE_LEGACY_MAX,
            target_aspect=float(target_aspect),
            abs_iota_min=None if abs_iota_min is None else float(abs_iota_min),
            mirror_ratio_max=float(mirror_threshold),
            max_elongation=MAX_ELONGATION,
        ),
    )


def _stage_vmec_max_mode(stage: StagePolicy) -> int:
    if stage.vmec_max_mode is not None:
        return int(stage.vmec_max_mode)
    import vmec_jax as vj

    return max(int(vj.normalize_boundary_mode_limits(mode).mode) for mode in stage.stage_modes)


def _jsonable_stage_mode(mode: Any) -> Any:
    if isinstance(mode, dict):
        return dict(mode)
    if isinstance(mode, tuple):
        return list(mode)
    return mode


def run_policy(
    policy: Policy,
    *,
    input_file: Path,
    out_root: Path,
    resolution: ScanResolution,
    inner_max_iter: int,
    trial_max_iter: int,
) -> dict[str, Any]:
    _ensure_repo_on_path()
    import vmec_jax as vj
    from vmec_jax._compat import enable_x64

    enable_x64(True)
    active_input = input_file
    policy_dir = out_root / policy.name
    stage_records: list[dict[str, Any]] = []
    selected_output = False
    selected_reason = "no stage completed"
    final_result = None
    final_diagnostics: dict[str, Any] | None = None
    final_qi_options = None
    start = time.perf_counter()

    for index, stage in enumerate(policy.stages, start=1):
        stage_dir = policy_dir / f"{index:02d}_{stage.name}"
        problem, qi_options = _make_problem(vj, resolution, stage)
        stage_max_mode = _stage_vmec_max_mode(stage)
        vmec = vj.FixedBoundaryVMEC.from_input(
            active_input,
            max_mode=stage_max_mode,
            min_vmec_mode=int(stage.min_vmec_mode) if stage.min_vmec_mode is not None else max(6, stage_max_mode + 3),
            output_dir=stage_dir,
            project_input_boundary_to_max_mode=True,
        )
        result = vj.least_squares_solve(
            vmec,
            problem,
            stage_modes=list(stage.stage_modes),
            max_nfev=int(stage.max_nfev),
            continuation_nfev=0,
            method=str(stage.method),
            ftol=1.0e-4,
            gtol=1.0e-4,
            xtol=1.0e-8,
            use_ess=True,
            ess_alpha=1.2,
            label=f"{policy.name}:{stage.name}",
            inner_max_iter=int(inner_max_iter),
            inner_ftol=1.0e-8,
            trial_max_iter=int(trial_max_iter),
            trial_ftol=1.0e-8,
            solver_device="cpu",
            scipy_tr_solver="lsmr",
            scipy_lsmr_maxiter=5,
            scalar_step_bound=stage.scalar_step_bound,
            lbfgs_step_bound=stage.lbfgs_step_bound,
            save_stage_inputs=True,
            save_stage_wouts=False,
        )
        stage_dir.mkdir(parents=True, exist_ok=True)
        result.final_optimizer.save_input(stage_dir / "input.final", result.final_params)
        result.final_optimizer.save_wout(stage_dir / "wout_final.nc", result.final_params, state=result.final_state)
        result.final_optimizer.save_history(stage_dir / "history.json", result.final_result)
        mirror_gate = (
            float(stage.promotion_mirror_threshold)
            if stage.promotion_mirror_threshold is not None
            else float(stage.mirror_threshold)
        )
        diagnostics = _diagnose(
            vj,
            result,
            resolution,
            qi_options,
            target_aspect=float(stage.target_aspect),
            abs_iota_min=stage.iota_abs_min,
            mirror_threshold=mirror_gate,
        )
        (stage_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
        stage_selected = bool(diagnostics.get("qi_seed_gate_passed"))
        stage_record = {
            "stage": index,
            "stage_name": stage.name,
            "method": stage.method,
            "stage_modes": [_jsonable_stage_mode(mode) for mode in stage.stage_modes],
            "stage_mode_labels": [
                vj.describe_boundary_mode_limits(mode) for mode in stage.stage_modes
            ],
            "vmec_max_mode": stage_max_mode,
            "max_nfev": stage.max_nfev,
            "output_dir": str(stage_dir),
            "selected": stage_selected,
            "engineering_selected": bool(diagnostics.get("qi_engineering_gate_passed")),
            "smooth_qi": _float_or_none(diagnostics.get("qi_smooth_total")),
            "legacy_qi": _float_or_none(diagnostics.get("qi_legacy_total")),
            "mirror": _float_or_none(diagnostics.get("qi_mirror_ratio_max")),
            "elongation": _float_or_none(diagnostics.get("qi_max_elongation")),
            "iota": _float_or_none(diagnostics.get("mean_iota")),
            "aspect": _float_or_none(diagnostics.get("aspect")),
            "wall_time_s": _float_or_none(result.timing_summary.get("total_wall_time_s")),
            "gate_failures": diagnostics.get("qi_gate_failures", []),
        }
        stage_records.append(stage_record)
        final_result = result
        final_diagnostics = diagnostics
        final_qi_options = qi_options
        if stage_selected:
            selected_output = True
            selected_reason = "QI+iota seed gate passed"
            active_input = stage_dir / "input.final"
        elif stage.continue_on_failure:
            selected_output = False
            selected_reason = "; ".join(diagnostics.get("qi_failure_reasons", [])) or "QI+iota seed gate failed"
            active_input = stage_dir / "input.final"
            continue
        else:
            selected_output = False
            selected_reason = "; ".join(diagnostics.get("qi_failure_reasons", [])) or "QI+iota seed gate failed"
            break

    if final_result is not None and final_diagnostics is not None and final_qi_options is not None:
        policy_dir.mkdir(parents=True, exist_ok=True)
        final_result.final_optimizer.save_input(policy_dir / "input.final", final_result.final_params)
        final_result.final_optimizer.save_wout(policy_dir / "wout_final.nc", final_result.final_params, state=final_result.final_state)
        (policy_dir / "diagnostics.json").write_text(json.dumps(final_diagnostics, indent=2, sort_keys=True) + "\n")

    wall = time.perf_counter() - start
    last = stage_records[-1] if stage_records else {}
    record = {
        "policy": policy.name,
        "description": policy.description,
        "selected": bool(selected_output),
        "engineering_selected": bool(
            final_diagnostics is not None and final_diagnostics.get("qi_engineering_gate_passed")
        ),
        "selection": "selected" if selected_output else "rejected",
        "selection_reason": selected_reason,
        "smooth_qi": last.get("smooth_qi"),
        "legacy_qi": last.get("legacy_qi"),
        "mirror": last.get("mirror"),
        "elongation": last.get("elongation"),
        "iota": last.get("iota"),
        "aspect": last.get("aspect"),
        "wall_time_s": wall,
        "output_dir": str(policy_dir),
        "stages": stage_records,
    }
    (policy_dir / "policy_result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def write_summary(records: list[dict[str, Any]], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    fields = [
        "policy",
        "smooth_qi",
        "legacy_qi",
        "mirror",
        "elongation",
        "iota",
        "aspect",
        "wall_time_s",
        "selection",
        "engineering_selected",
        "selection_reason",
        "output_dir",
    ]
    with (out_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def _policy_subset(all_policies: tuple[Policy, ...], names: list[str] | None) -> tuple[Policy, ...]:
    if not names:
        return all_policies
    wanted = set(names)
    missing = wanted - {policy.name for policy in all_policies}
    if missing:
        raise ValueError(f"Unknown policies: {', '.join(sorted(missing))}")
    return tuple(policy for policy in all_policies if policy.name in wanted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--execute", action="store_true", help="Run optimizations. Default only writes the plan.")
    parser.add_argument("--policy", action="append", help="Run one named policy; repeat for multiple.")
    parser.add_argument("--max-nfev", type=int, default=2)
    parser.add_argument("--inner-max-iter", type=int, default=40)
    parser.add_argument("--trial-max-iter", type=int, default=25)
    parser.add_argument("--quick", action="store_true", help="Use an even smaller diagnostic grid.")
    args = parser.parse_args(argv)

    resolution = ScanResolution()
    if args.quick:
        resolution = ScanResolution(surfaces=(0.5, 1.0), mboz=6, nboz=6, nphi=31, nalpha=7, n_bounce=9)
    policies = _policy_subset(default_policies(max_nfev=max(1, int(args.max_nfev))), args.policy)

    args.out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "input": str(args.input),
        "out_root": str(args.out_root),
        "target_aspect": TARGET_ASPECT,
        "target_abs_iota_min": TARGET_ABS_IOTA_MIN,
        "resolution": asdict(resolution),
        "policies": [
            {
                "name": policy.name,
                "description": policy.description,
                "stages": [asdict(stage) for stage in policy.stages],
            }
            for policy in policies
        ],
    }
    (args.out_root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if not args.execute:
        print(f"Wrote bounded QI policy scan plan: {args.out_root / 'plan.json'}")
        return 0

    records: list[dict[str, Any]] = []
    for policy in policies:
        try:
            print(f"\n[qi-policy-scan] running {policy.name}", flush=True)
            records.append(
                run_policy(
                    policy,
                    input_file=args.input,
                    out_root=args.out_root,
                    resolution=resolution,
                    inner_max_iter=int(args.inner_max_iter),
                    trial_max_iter=int(args.trial_max_iter),
                )
            )
        except Exception as exc:  # noqa: BLE001 - policy scans should continue.
            records.append(
                {
                    "policy": policy.name,
                    "description": policy.description,
                    "selected": False,
                    "selection": "rejected",
                    "selection_reason": f"{type(exc).__name__}: {exc}",
                    "smooth_qi": None,
                    "legacy_qi": None,
                    "mirror": None,
                    "elongation": None,
                    "iota": None,
                    "aspect": None,
                    "wall_time_s": None,
                    "output_dir": str(args.out_root / policy.name),
                    "stages": [],
                }
            )
            print(f"[qi-policy-scan] {policy.name} failed: {type(exc).__name__}: {exc}", flush=True)
        write_summary(records, args.out_root)
    print(f"\nWrote QI policy scan summary: {args.out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

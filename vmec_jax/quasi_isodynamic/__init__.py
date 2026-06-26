"""Quasi-isodynamic objectives, diagnostics, and staged optimization helpers.

This package keeps all QI-specific code behind one domain name:

- `objectives.py` contains differentiable smooth-QI, mirror, elongation, and
  LgradB objective terms.
- `diagnostics.py` turns solved states and Boozer outputs into ranked QI
  diagnostic records.
- `legacy.py` keeps the NumPy/SciPy Goodman-style branch diagnostic used for
  validation and ranking, not autodiff optimization.
- `optimization.py` holds reusable staged-QI workflow helpers used by the
  example scripts.
"""

from .diagnostics import (
    QI_DIAGNOSTIC_VERSION,
    QIDiagnosticOptions,
    QISeedSuitabilityTargets,
    annotate_qi_seed_suitability,
    qi_cleanup_candidate_promotable,
    qi_diagnostics_from_boozer_output,
    qi_diagnostics_from_state,
    qi_promotion_score,
    rank_qi_seed_records,
)
from .legacy import legacy_qi_branch_shuffle_diagnostic_from_boozer_output
from .objectives import (
    boozer_output_from_state,
    boundary_max_elongation_from_rz,
    lgradb_from_state,
    lgradb_penalty_from_state,
    max_elongation_penalty_from_state,
    mirror_ratio_penalty_from_boozer_modes,
    mirror_ratio_penalty_from_boozer_output,
    mirror_ratio_penalty_from_state,
    quasi_isodynamic_residual_from_boozer_modes,
    quasi_isodynamic_residual_from_boozer_output,
    quasi_isodynamic_residual_from_state,
)

__all__ = [
    "QI_DIAGNOSTIC_VERSION",
    "QIDiagnosticOptions",
    "QISeedSuitabilityTargets",
    "annotate_qi_seed_suitability",
    "boozer_output_from_state",
    "boundary_max_elongation_from_rz",
    "lgradb_from_state",
    "lgradb_penalty_from_state",
    "legacy_qi_branch_shuffle_diagnostic_from_boozer_output",
    "max_elongation_penalty_from_state",
    "mirror_ratio_penalty_from_boozer_modes",
    "mirror_ratio_penalty_from_boozer_output",
    "mirror_ratio_penalty_from_state",
    "qi_cleanup_candidate_promotable",
    "qi_diagnostics_from_boozer_output",
    "qi_diagnostics_from_state",
    "qi_promotion_score",
    "quasi_isodynamic_residual_from_boozer_modes",
    "quasi_isodynamic_residual_from_boozer_output",
    "quasi_isodynamic_residual_from_state",
    "rank_qi_seed_records",
]

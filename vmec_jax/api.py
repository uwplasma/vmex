"""Public, user-facing API for vmec_jax.

This module intentionally re-exports a small set of functions that cover the
common workflows:

- I/O: load input, read/write wout
- Solve: fixed- and free-boundary drivers
- Plotting: VMEC-style surfaces and magnetic-field magnitude

Advanced users can import lower-level kernels directly from submodules.
"""

# ruff: noqa: F401
#
# This module intentionally imports the stable public API into one namespace.
# The export list is derived from these imports to avoid a second manual list
# that can drift from the actual facade.

from __future__ import annotations

import types as _types

from .namelist import minimal_fixed_boundary_indata, read_indata, write_indata
from .solvers.free_boundary import (
    ReducedControlMap,
    ReducedControlStep,
    reduced_control_decode,
    reduced_control_least_squares_step,
    reduced_control_pullback,
)
from .toroidal_hybrid import (
    SquareAxisControlBasis,
    SquareAxisControlFourierMatrix,
    SquareAxisControlProjection,
    SquareAxisSplineControls,
    ToroidalHybridBoundarySamples,
    evaluate_toroidal_hybrid_indata_boundary,
    recommend_square_axis_stellarator_mirror_hybrid_resolution,
    recommended_square_axis_ntheta,
    recommended_square_axis_nzeta,
    sample_square_axis_stellarator_mirror_hybrid_boundary,
    sample_toroidal_stellarator_mirror_hybrid_boundary,
    square_axis_free_boundary_edge_control_projection_payload,
    square_axis_resolution_deck_status,
    square_axis_strict_convergence_assessment,
    square_axis_spline_control_fourier_map_status,
    square_axis_spline_control_fourier_matrix,
    square_axis_spline_radius,
    square_axis_spline_radius_matrix,
    square_axis_spline_symmetric_control_basis,
    square_axis_strict_schedule_status,
    square_axis_stellarator_mirror_hybrid_indata,
    square_axis_stellarator_mirror_hybrid_projection_error,
    toroidal_hybrid_cross_section_anisotropy,
    toroidal_hybrid_cross_section_orientation,
    toroidal_stellarator_mirror_hybrid_indata,
    toroidal_stellarator_mirror_hybrid_metrics,
)
from .driver import (
    ExampleData,
    FixedBoundaryRun,
    FixedBoundarySolvedState,
    example_paths,
    fixed_boundary_solved_state,
    load_example,
    load_input,
    load_wout,
    run_free_boundary,
    run_fixed_boundary,
    residual_scalars_from_state,
    wout_from_fixed_boundary_run,
    write_wout_from_fixed_boundary_run,
)
from .boundary import boundary_from_input_convention, boundary_input_from_indata
from .booz import (
    BoozConfig,
    parse_booz_surfaces,
    read_booz_config,
    resolve_boozmn_path,
    run_booz_xform,
)
from .booz_input import booz_xform_inputs_from_state
from .field import b_cartesian_from_state, signgs_from_sqrtg
from .energy import flux_profiles_from_indata
from .finite_beta import (
    FiniteBetaTargets,
    finite_beta_global_residuals_from_state,
    finite_beta_scalars_from_state,
    magnetic_well_from_state,
)
from .implicit import solve_fixed_boundary_state_implicit_vmec_residual
from .bootstrap_current import (
    BootstrapCurrentIteration,
    BootstrapCurrentOptions,
    BootstrapCurrentResult,
    apply_current_profile_to_indata,
    bootstrap_current_fixed_point,
    bootstrap_current_update_to_indata,
    damp_current_profile,
    dpsi_ds_from_vmec_phiedge,
    integrate_current_derivative,
    redl_current_derivative_update,
    redl_current_integrating_factor_update,
    redl_current_rhs,
    vmec_current_profile_from_bootstrap_update,
)
from .mercier import glasser_resistive_interchange_from_mercier_terms
from .optimization_workflow import (
    AbsMeanIotaFloor,
    AbsMeanIotaCeiling,
    AspectRatio,
    AugmentedLagrangianConstraint,
    BDotB,
    BDotGradV,
    BVector,
    BetaTotal,
    BoozerBTarget,
    BoundaryModeLimits,
    DMerc,
    FixedBoundaryOptimizationResult,
    FixedBoundaryVMEC,
    GlasserResistiveInterchange,
    JDotB,
    JVector,
    LeastSquaresProblem,
    LgradB,
    MagneticWell,
    MaxElongation,
    MeanIota,
    MirrorRatio,
    VMECMirrorRatio,
    ObjectiveTerm,
    OptimizationOutputPaths,
    QuasiIsodynamicOptions,
    QuasiIsodynamicResidual,
    QuasiIsodynamicResidualCeiling,
    QuasisymmetryRatioResidual,
    QIObjectiveTerm,
    RedlBootstrapMismatch,
    ToroidalCurrent,
    ToroidalCurrentGradient,
    VolavgB,
    boozer_b_target_from_wout,
    least_squares_solve,
    optimization_output_paths,
    prepare_simple_omnigenity_seed_input,
    interpolate_indata_boundary,
    qi_boozer_b_target_objective,
    qi_max_elongation_constraint,
    qi_mirror_ratio_constraint,
    qs_stage_modes,
    repeated_stage_modes,
    save_optimization_result,
    simple_omnigenity_seed_indata,
)
from .quasi_isodynamic import (
    lgradb_from_state,
    lgradb_penalty_from_state,
    max_elongation_penalty_from_state,
    mirror_ratio_penalty_from_boozer_modes,
    mirror_ratio_penalty_from_boozer_output,
    quasi_isodynamic_residual_from_boozer_modes,
    quasi_isodynamic_residual_from_boozer_output,
    quasi_isodynamic_residual_from_state,
)
from .plotting import (
    bmag_from_state_physical,
    bmag_from_wout,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
    plot_3d_boundary_comparison,
    plot_bmag_contours,
    plot_boozmn,
    plot_boozmn_bmag_contours,
    plot_boozmn_mode_families,
    plot_boozmn_spectrum,
    plot_boozer_bmag_contours_from_state,
    plot_boozer_lcfs_bmag_comparison,
    plot_objective_history,
    plot_wout,
    surface_rz_from_wout_physical,
    surface_stack,
    vmecplot2_bmag_grid,
    vmecplot2_cross_section_indices,
    vmecplot2_lcfs_3d_grid,
    vmecplot2_surface_grid,
    write_axisym_overview,
    write_bmag_parity_figures,
    write_bsub_parity_figures,
    write_bsup_parity_figures,
    zeta_grid,
    zeta_grid_field_period,
)
from .quasi_isodynamic.diagnostics import (
    QIDiagnosticOptions,
    QISeedSuitabilityTargets,
    annotate_qi_seed_suitability,
    qi_cleanup_candidate_promotable,
    qi_diagnostics_from_boozer_output,
    qi_diagnostics_from_state,
    qi_promotion_score,
    rank_qi_seed_records,
)
from .quasi_isodynamic.optimization import (
    QIOptimizationContext,
    diagnostic_float,
    jsonable,
    make_qi_optimization_context,
    qi_diagnostics_for_result,
    qi_engineering_constraint_tuples,
    qi_mirror_objective_for_stage,
    qi_stage_modes,
    run_boundary_reference_preconditioner,
    run_qi_stage_policy,
    run_target_helicity_seed_preconditioner,
    save_raw_seed_initial_artifacts,
    target_helicity_seed_terms,
)
from .visualization import export_vtk_surface_and_fieldline
from .wout import read_wout, state_from_wout

__all__ = [
    name
    for name, value in globals().items()
    if not name.startswith("_")
    and name != "annotations"
    and not isinstance(value, _types.ModuleType)
]

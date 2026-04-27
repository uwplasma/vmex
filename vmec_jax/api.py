"""Public, user-facing API for vmec_jax.

This module intentionally re-exports a small set of functions that cover the
common workflows:

- I/O: load input, read/write wout
- Solve: fixed- and free-boundary drivers
- Plotting: VMEC-style surfaces and magnetic-field magnitude

Advanced users can import lower-level kernels directly from submodules.
"""

from __future__ import annotations

from .namelist import read_indata, write_indata
from .driver import (
    FixedBoundaryRun,
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
from .booz_input import booz_xform_inputs_from_state
from .field import b_cartesian_from_state, signgs_from_sqrtg
from .energy import flux_profiles_from_indata
from .plotting import (
    bmag_from_state_physical,
    bmag_from_wout,
    closed_theta_grid,
    fix_matplotlib_3d,
    profiles_from_wout,
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
from .visualization import export_vtk_surface_and_fieldline
from .wout import read_wout, state_from_wout

__all__ = [
    # Driver / solve
    "FixedBoundaryRun",
    "load_example",
    "load_input",
    "load_wout",
    "run_free_boundary",
    "run_fixed_boundary",
    "residual_scalars_from_state",
    "wout_from_fixed_boundary_run",
    "write_wout_from_fixed_boundary_run",
    "boundary_input_from_indata",
    "boundary_from_input_convention",
    "read_indata",
    "write_indata",
    "booz_xform_inputs_from_state",
    # Plotting helpers
    "bmag_from_state_physical",
    "bmag_from_wout",
    "closed_theta_grid",
    "fix_matplotlib_3d",
    "profiles_from_wout",
    "surface_rz_from_wout_physical",
    "surface_stack",
    "vmecplot2_bmag_grid",
    "vmecplot2_cross_section_indices",
    "vmecplot2_lcfs_3d_grid",
    "vmecplot2_surface_grid",
    "write_axisym_overview",
    "write_bmag_parity_figures",
    "write_bsub_parity_figures",
    "write_bsup_parity_figures",
    "zeta_grid",
    "zeta_grid_field_period",
    # Field helpers
    "b_cartesian_from_state",
    "signgs_from_sqrtg",
    "flux_profiles_from_indata",
    # Visualization helpers
    "export_vtk_surface_and_fieldline",
    # Low-friction wout access for plotting
    "read_wout",
    "state_from_wout",
]

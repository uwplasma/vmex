"""Compatibility facade for fixed-boundary solver implementations.

The implementation lives in domain modules under
``vmec_jax.solvers.fixed_boundary`` so this historical public module stays
small while existing imports and internal monkeypatch seams continue to work.
"""

from __future__ import annotations

import importlib
import sys
import types

_COMPAT_MODULE_NAMES = (
    "vmec_jax.field",
    "vmec_jax.fourier",
    "vmec_jax.geom",
    "vmec_jax.grids",
    "vmec_jax.state",
    "vmec_jax.vmec_forces",
    "vmec_jax.vmec_residue",
    "vmec_jax._solve_runtime",
    "vmec_jax.solvers.fixed_boundary.jit_cache",
    "vmec_jax.solvers.fixed_boundary.options",
    "vmec_jax.solvers.fixed_boundary.profiles",
    "vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset",
    "vmec_jax.solvers.fixed_boundary.diagnostics.first_step",
    "vmec_jax.solvers.fixed_boundary.diagnostics.force",
    "vmec_jax.solvers.fixed_boundary.diagnostics.hlo",
    "vmec_jax.solvers.fixed_boundary.diagnostics.io",
    "vmec_jax.solvers.fixed_boundary.optimization.constraints",
    "vmec_jax.solvers.fixed_boundary.optimization.energy",
    "vmec_jax.solvers.fixed_boundary.optimization.gd",
    "vmec_jax.solvers.fixed_boundary.optimization.gradient",
    "vmec_jax.solvers.fixed_boundary.optimization.lambda_gd",
    "vmec_jax.solvers.fixed_boundary.optimization.lbfgs",
    "vmec_jax.solvers.fixed_boundary.optimization.quasi_newton",
    "vmec_jax.solvers.fixed_boundary.optimization.residual_gn",
    "vmec_jax.solvers.fixed_boundary.optimization.residual_lbfgs",
    "vmec_jax.solvers.fixed_boundary.optimization.residual_objective",
    "vmec_jax.solvers.fixed_boundary.optimization.tolerances",
    "vmec_jax.solvers.fixed_boundary.preconditioning.operators",
    "vmec_jax.solvers.fixed_boundary.residual.config",
    "vmec_jax.solvers.fixed_boundary.residual.force_payload",
    "vmec_jax.solvers.fixed_boundary.residual.geometry",
    "vmec_jax.solvers.fixed_boundary.residual.host_diagnostics",
    "vmec_jax.solvers.fixed_boundary.residual.mode_transform",
    "vmec_jax.solvers.fixed_boundary.residual.payload_blocks",
    "vmec_jax.solvers.fixed_boundary.residual.policy",
    "vmec_jax.solvers.fixed_boundary.residual.preconditioner_payload",
    "vmec_jax.solvers.fixed_boundary.residual.ptau",
    "vmec_jax.solvers.fixed_boundary.residual.scan_adapters",
    "vmec_jax.solvers.fixed_boundary.scan.debug",
    "vmec_jax.solvers.fixed_boundary.scan.math",
    "vmec_jax.solvers.fixed_boundary.scan.output",
    "vmec_jax.solvers.fixed_boundary.scan.payload",
    "vmec_jax.solvers.fixed_boundary.scan.planning",
    "vmec_jax.solvers.fixed_boundary.scan.resume",
    "vmec_jax.solvers.fixed_boundary.scan.runtime",
    "vmec_jax.solvers.free_boundary.control",
)

_COMPAT_MODULES = tuple(importlib.import_module(name) for name in _COMPAT_MODULE_NAMES)
_MODULE_BY_NAME = {module.__name__: module for module in _COMPAT_MODULES}
_iteration = importlib.import_module("vmec_jax.solvers.fixed_boundary.residual.iteration")
_fixed_boundary_api = importlib.import_module("vmec_jax.solvers.fixed_boundary.api")
_MODULE_BY_NAME.update(
    {
        _iteration.__name__: _iteration,
        _fixed_boundary_api.__name__: _fixed_boundary_api,
    }
)

_LEGACY_ALIASES = {
    "_dump_array": ("vmec_jax.solvers.fixed_boundary.diagnostics.force", "dump_array"),
    "_enforce_field_rows": ("vmec_jax.solvers.fixed_boundary.optimization.constraints", "enforce_field_rows"),
    "_enforce_field_rows_np": ("vmec_jax.solvers.fixed_boundary.optimization.constraints", "enforce_field_rows_np"),
    "_enforce_lambda_gauge": ("vmec_jax.solvers.fixed_boundary.optimization.constraints", "enforce_lambda_gauge"),
    "_gc_from_frzl": ("vmec_jax.solvers.fixed_boundary.diagnostics.force", "gc_from_frzl"),
    "_free_boundary_iter_controls": ("vmec_jax.solvers.free_boundary.control", "free_boundary_iter_controls"),
    "_initial_axis_reset_decision": ("vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset", "initial_axis_reset_decision"),
    "_jit_cache_limit": ("vmec_jax.solvers.fixed_boundary.jit_cache", "jit_cache_limit"),
    "_mask_scan_restart_force_payload": ("vmec_jax.solvers.fixed_boundary.scan.payload", "mask_scan_restart_force_payload"),
    "_merge_axis_reset_state": ("vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset", "merge_axis_reset_state"),
    "_metric_surface_precond_scales_jax": ("vmec_jax.solvers.fixed_boundary.preconditioning.operators", "metric_surface_precond_scales_jax"),
    "_metric_surface_precond_scales_np": ("vmec_jax.solvers.fixed_boundary.preconditioning.operators", "metric_surface_precond_scales_np"),
    "_normalize_debug_print_mode": ("vmec_jax.solvers.fixed_boundary.residual.config", "normalize_debug_print_mode"),
    "_parse_bad_jacobian_config": ("vmec_jax.solvers.fixed_boundary.residual.config", "parse_bad_jacobian_config"),
    "_replace_mode_slice": ("vmec_jax.solvers.fixed_boundary.optimization.constraints", "replace_mode_slice"),
    "_replace_mode_slice_np": ("vmec_jax.solvers.fixed_boundary.optimization.constraints", "replace_mode_slice_np"),
    "_resolve_chunked_scan_config": ("vmec_jax.solvers.fixed_boundary.residual.config", "resolve_chunked_scan_config"),
    "_resolve_lbfgs_curvature_tol": ("vmec_jax.solvers.fixed_boundary.optimization.quasi_newton", "lbfgs_curvature_tolerance"),
    "_scale_mode_slice": ("vmec_jax.solvers.fixed_boundary.optimization.constraints", "scale_mode_slice"),
    "_scale_mode_slice_np": ("vmec_jax.solvers.fixed_boundary.optimization.constraints", "scale_mode_slice_np"),
    "_sm_sp_from_s_np": ("vmec_jax.solvers.fixed_boundary.preconditioning.operators", "sm_sp_from_s_np"),
    "_vmec2000_scan_options_from_env": ("vmec_jax.solvers.fixed_boundary.residual.policy", "vmec2000_scan_options_from_env"),
    "_vmec_scale_m1_factors_from_mats": ("vmec_jax.solvers.fixed_boundary.preconditioning.operators", "vmec_scale_m1_factors_from_mats"),
    "_vmec_scale_m1_factors_from_mats_np": ("vmec_jax.solvers.fixed_boundary.preconditioning.operators", "vmec_scale_m1_factors_from_mats_np"),
    "_write_axis_reset_dump": ("vmec_jax.solvers.fixed_boundary.diagnostics.axis_reset", "write_axis_reset_dump"),
    "_zero_coeff_column": ("vmec_jax.solvers.fixed_boundary.optimization.constraints", "zero_coeff_column"),
    "_zero_coeff_column_np": ("vmec_jax.solvers.fixed_boundary.optimization.constraints", "zero_coeff_column_np"),
}


def _export_symbols(module) -> None:
    for name, value in vars(module).items():
        if name.startswith("__") and name.endswith("__"):
            continue
        globals()[name] = value


for _module in _COMPAT_MODULES:
    _export_symbols(_module)
_export_symbols(_iteration)
_export_symbols(_fixed_boundary_api)
for _legacy_name, (_module_name, _target_name) in _LEGACY_ALIASES.items():
    globals()[_legacy_name] = getattr(_MODULE_BY_NAME[_module_name], _target_name)


class _SolveFacadeModule(types.ModuleType):
    """Forward assignments to the implementation module.

    A number of internal tests and downstream debugging workflows monkeypatch
    private ``vmec_jax.solve`` symbols.  The exported solver functions execute
    in the implementation module's global namespace, so assignments on this
    facade must be mirrored there to preserve legacy behavior.
    """

    def __setattr__(self, name, value):
        if not (name.startswith("__") and name.endswith("__")):
            if name in _LEGACY_ALIASES:
                module_name, target_name = _LEGACY_ALIASES[name]
                setattr(_MODULE_BY_NAME[module_name], target_name, value)
            for module in (*_COMPAT_MODULES, _iteration, _fixed_boundary_api):
                if hasattr(module, name):
                    setattr(module, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SolveFacadeModule

__all__ = tuple(name for name in globals() if not (name.startswith("__") and name.endswith("__")))

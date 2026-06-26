"""Optimization-oriented helpers for vmec_jax workflows."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, nullcontext
from dataclasses import fields, is_dataclass, replace
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import _compat as _compat_module
from ._compat import jax, jnp
from .boundary import BoundaryCoeffs
from .energy import flux_profiles_from_indata
from .field import signgs_from_sqrtg
from .geom import eval_geom
from .init_guess import initial_guess_from_boundary
from .namelist import write_indata
from .optimizers.fixed_boundary.linear_guards import (
    finite_linear_operator_output, linear_operator_matrix_arg, linear_operator_vector_arg,
)
from .optimizers.fixed_boundary.history import (
    ResidualHistoryPolicy, build_run_history_dump, history_entry_from_residuals,
    qs_objective_from_residuals,
)
from .optimizers.fixed_boundary.gauss_newton import gauss_newton_least_squares
from .optimizers.fixed_boundary.exact_replay import (
    jvp_only_basepoint_carries_enabled, jvp_only_exact_tape_enabled,
    scan_exact_helpers, solve_exact_with_tape_for_jvp, solve_scan_exact_state,
)
from .optimizers.fixed_boundary.matrix_free import build_residual_linear_operator
from .optimizers.fixed_boundary.parameterization import (
    BoundaryParamSpec, apply_boundary_params, apply_boundary_params_numpy,
    boundary_param_names, boundary_param_specs, coeff_label, create_x_scale,
    extend_boundary_for_max_mode, indexed_boundary_maps_from_boundary,
    lift_boundary_params, rebuild_indata_with_resolution, truncate_indata_boundary_modes,
)
from .optimizers.fixed_boundary import profiling as _profiling
from .optimizers.fixed_boundary.qs_residuals import (
    FixedBoundaryContext,  # noqa: F401 - public re-export
    _pressure_profile_for_static,  # noqa: F401 - private compatibility re-export
    make_qh_residuals_fn,  # noqa: F401 - public re-export
    make_qs_residuals_fn,  # noqa: F401 - public re-export
    parse_surface_list,  # noqa: F401 - public re-export
    prepare_fixed_boundary_context,  # noqa: F401 - public re-export
    smooth_min_abs_iota_residual,  # noqa: F401 - public re-export
    surface_indices_from_s,  # noqa: F401 - public re-export
    surface_indices_from_static,  # noqa: F401 - public re-export
)
from .optimizers.fixed_boundary.replay_policy import (
    chunked_projected_replay_projection_enabled, fused_projected_replay_enabled,
    lasym_replay_column_chunk, optimizer_backend_name,
    precompute_linear_operator_initial_tangents_enabled, projected_replay_residuals_enabled,
    scalar_gradient_initial_tangents_enabled,
)
from .optimizers.fixed_boundary.scalar_gradient import exact_objective_and_gradient
from .optimizers.fixed_boundary.scalar_lbfgs import run_lbfgs_adjoint_exact_optimizer
from .optimizers.fixed_boundary.scalar_trust import run_scalar_trust_exact_optimizer
from .optimizers.fixed_boundary.scipy_least_squares import (
    run_scipy_dense_exact_optimizer, run_scipy_matrix_free_exact_optimizer,
)
from .optimizers.fixed_boundary import state_cache as _state_cache
from .state import VMECState
from .static import VMECStatic

# Backwards-compatible private helper names used by older tests and profiling
# scripts.  Implementations live in optimizers.fixed_boundary.linear_guards.
_finite_linear_operator_output = finite_linear_operator_output
_linear_operator_matrix_arg = linear_operator_matrix_arg
_linear_operator_vector_arg = linear_operator_vector_arg
_apply_boundary_params_numpy = apply_boundary_params_numpy
_coeff_label = coeff_label
_indexed_boundary_maps_from_boundary = indexed_boundary_maps_from_boundary

__all__ = [
    "BoundaryParamSpec", "FixedBoundaryContext", "FixedBoundaryExactOptimizer",
    "apply_boundary_params", "boundary_param_names", "boundary_param_specs",
    "create_x_scale", "extend_boundary_for_max_mode", "gauss_newton_least_squares",
    "lift_boundary_params", "prepare_fixed_boundary_context", "rebuild_indata_with_resolution",
    "smooth_min_abs_iota_residual", "truncate_indata_boundary_modes",
]


def _optimizer_backend_name(solver_device_name: str | None) -> str:
    """Return the active optimizer backend name without changing device policy."""

    return optimizer_backend_name(solver_device_name)


class FixedBoundaryExactOptimizer:
    """Exact fixed-boundary optimizer built on VMEC solves and AD callbacks.

    The optimizer owns the boundary parameterization, accepted/trial VMEC solve
    budgets, exact-state caches, and outer optimization method dispatch.  The
    objective is supplied as ``residuals_fn(state)`` so examples can assemble
    SIMSOPT-like least-squares tuples while staying entirely inside vmec_jax.

    Use ``inner_*`` settings for accepted exact solves, ``trial_*`` settings for
    relaxed trial residuals, ``solver_device`` to select CPU/GPU callbacks, and
    ``exact_path`` to choose tape or scan replay.  Full workflow examples live
    in ``examples/optimization`` and the user documentation.
    """

    _DICT_CACHE_ATTRS = (
        "_exact_cache",
        "_exact_state_cache",
        "_exact_state_key_by_id",
        "_exact_residual_cache",
        "_exact_jacobian_cache",
        "_discrete_jacobian_helper_cache",
        "_scan_exact_helper_cache",
    )
    _ORDERED_CACHE_ATTRS = ("_initial_state_cache", "_trial_residual_cache")

    def __init__(
        self,
        static: VMECStatic,
        indata,
        boundary: BoundaryCoeffs,
        specs: Sequence[BoundaryParamSpec],
        residuals_fn: Callable,
        boundary_input: BoundaryCoeffs | None = None,
        *,
        inner_max_iter: int | None = None,
        inner_ftol: float | None = None,
        trial_max_iter: int | None = None,
        trial_ftol: float | None = None,
        solver_device: str | None = None,
        exact_path: str | None = None,
        freeze_initial_axis: bool = False,
    ) -> None:
        self._solver_device_name = self._resolve_solver_device(solver_device)
        self._exact_path_request = self._resolve_exact_path_request(exact_path)
        self._inside_solver_device_context = False
        if self._solver_device_name is not None:
            static = self._move_to_solver_device(static)
            boundary = self._move_to_solver_device(boundary)
            if boundary_input is not None:
                boundary_input = self._move_to_solver_device(boundary_input)

        self._static = static
        self._indata = indata
        self._boundary = boundary
        self._boundary_input = boundary_input
        self._specs = list(specs)
        self._residuals_fn = residuals_fn
        self._freeze_initial_axis = bool(freeze_initial_axis)
        self._residuals_eval_fn = self._make_residuals_eval_fn(residuals_fn)
        self._n_qs: int | None = getattr(residuals_fn, "_n_qs", None)
        self._n_non_qs: int = int(getattr(residuals_fn, "_n_non_qs", 1))
        self._has_residual_block_metadata = hasattr(residuals_fn, "_n_qs") or hasattr(residuals_fn, "_n_non_qs")
        self._qs_total_from_state_fn = getattr(residuals_fn, "_qs_total_from_state", None)
        self._aspect_target = getattr(residuals_fn, "_aspect_target", None)
        self._aspect_weight = float(getattr(residuals_fn, "_aspect_weight", 1.0))
        self._objective_family = getattr(residuals_fn, "_objective_family", None)
        self._helicity_m = getattr(residuals_fn, "_helicity_m", None)
        self._helicity_n = getattr(residuals_fn, "_helicity_n", None)

        # Derive signgs from the initial guess.
        state0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
        geom0 = eval_geom(state0, static)
        self._signgs = int(signgs_from_sqrtg(np.asarray(geom0.sqrtg), axis_index=1))
        self._flux = flux_profiles_from_indata(indata, static.s, signgs=self._signgs)
        if self._freeze_initial_axis:
            from .init_guess import extract_axis_override_from_state

            self._initial_axis_override = extract_axis_override_from_state(state0, static)
        else:
            self._initial_axis_override = None

        self._layout = state0.layout

        # Solver settings derived from indata.
        self._inner_max_iter = self._read_last_array("NITER_ARRAY", "NITER", 1500, int)
        self._inner_ftol = self._read_last_array("FTOL_ARRAY", "FTOL", 1e-13, float)
        self._step_size = float(indata.get_float("DELT", 1.0))
        if inner_max_iter is not None and int(inner_max_iter) > 0:
            self._inner_max_iter = int(inner_max_iter)
        if inner_ftol is not None and float(inner_ftol) > 0.0:
            self._inner_ftol = float(inner_ftol)

        _base = dict(
            indata=indata,
            signgs=self._signgs,
            step_size=self._step_size,
            include_constraint_force=True,
            apply_m1_constraints=True,
            precond_radial_alpha=0.5,
            precond_lambda_alpha=0.5,
            mode_diag_exponent=0.0,
            auto_flip_force=False,
            divide_by_scalxc_for_update=False,
            lambda_update_scale=1.0,
            enforce_vmec_lambda_axis=True,
            vmec2000_control=True,
            strict_update=True,
            backtracking=False,
            reference_mode=False,
            use_restart_triggers=True,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces="auto",
            use_scan=False,
            light_history=True,
            # The optimizer only ever consumes `result.state` from these inner
            # solves. Keeping the full resume_state payload alive in diagnostics
            # needlessly retains large cached arrays/checkpoints across SciPy
            # callbacks and is a major source of RSS growth on converged runs.
            resume_state_mode="none",
        )
        self._exact_solver_kwargs = dict(
            _base,
            preconditioner_use_precomputed_tridi=self._use_precomputed_tridi_for_exact_tape(),
        )
        self._trial_solver_kwargs = dict(
            _base,
            # Trial-point residuals do not need an adjoint tape.  Use a
            # backend-aware policy: CPU stays on the VMEC-control loop for
            # convergence/control parity, while accelerator backends use scan
            # to reduce launch overhead. VMEC_JAX_OPT_TRIAL_SCAN overrides this
            # for diagnostics.
            jit_forces="auto",
            use_scan=self._use_scan_for_trial_solves(),
        )
        self._trial_max_iter = min(self._inner_max_iter, 800)
        if trial_max_iter is not None:
            if int(trial_max_iter) > 0:
                self._trial_max_iter = int(trial_max_iter)
            else:
                self._trial_max_iter = int(self._inner_max_iter)
        if trial_ftol is None:
            self._trial_ftol = max(self._inner_ftol, 1e-10)
        elif float(trial_ftol) > 0.0:
            self._trial_ftol = float(trial_ftol)
        else:
            self._trial_ftol = float(self._inner_ftol)

        self._initialize_run_caches(state0)

    def _initialize_run_caches(self, state0: VMECState) -> None:
        for cache_attr in self._DICT_CACHE_ATTRS:
            setattr(self, cache_attr, {})
        self._scan_exact_path = self._select_exact_path()
        self._initial_state_cache: OrderedDict[bytes, VMECState] = OrderedDict()
        self._initial_state_cache_max = 4
        self._remember_initial_state(np.zeros(len(self._specs), dtype=float), state0)
        self._initial_state_packed_helper = None
        self._initial_tangent_cache: dict = {}
        self._initial_tangent_direction_cache: dict = {}
        self._last_jacobian_residual: np.ndarray | None = None
        self._last_jacobian_source = "exact_tape_replay"
        self._trial_residual_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._trial_residual_cache_max = 8
        self._profile: dict[str, dict[str, float | int]] = {}
        self._callback_trace_enabled = False
        self._callback_trace: list[dict] = []
        self._callback_point_ids: dict[bytes, int] = {}
        self._callback_previous_key: bytes | None = None
        self._history: list[dict] = []
        self._wall_t0: float = 0.0
        self._last_jacobian_key: list = [None]
        self._iota_fn = None
        self._best_exact_params: np.ndarray | None = None
        self._best_exact_state: VMECState | None = None
        self._best_exact_residual: np.ndarray | None = None
        self._best_exact_cost: float = math.inf
        self._exact_history_rejected_count: int = 0

    @property
    def static(self):
        """VMEC static configuration used by this optimizer stage."""

        return self._static

    @property
    def indata(self):
        """VMEC input data used by this optimizer stage."""

        return self._indata

    @property
    def signgs(self) -> int:
        """VMEC Jacobian sign used for profile and Boozer adapters."""

        return int(self._signgs)

    @property
    def flux(self):
        """Half/full-mesh flux-profile data used by objective callbacks."""

        return self._flux

    def _resolve_solver_device(self, solver_device: str | None) -> str | None:
        name = "auto" if solver_device is None else str(solver_device).strip().lower()
        if name in ("", "none", "auto", "default"):
            return None
        try:
            jax_module = _compat_module.jax
            current_backend = str(jax_module.default_backend()).strip().lower() if jax_module is not None else ""
        except Exception:
            current_backend = ""
        aliases = {
            "gpu": {"gpu", "cuda", "rocm", "tpu"},
            "cuda": {"gpu", "cuda"},
            "rocm": {"gpu", "rocm"},
            "tpu": {"tpu"},
            "cpu": {"cpu"},
        }
        if current_backend in aliases.get(name, {name}):
            # Explicitly requesting the already-active backend should not wrap
            # every callback in a default_device context or move static data a
            # second time.  On GPU this path is materially slower for new
            # accepted-point exact tapes.
            return None
        return name

    def _resolve_exact_path_request(self, exact_path: str | None) -> str | None:
        """Validate the optional accepted-point differentiation path request."""

        if exact_path is None:
            return None
        name = str(exact_path).strip().lower().replace("-", "_")
        if name in ("", "none", "auto", "default"):
            return None
        if name not in ("tape", "scan"):
            raise ValueError("exact_path must be one of None, 'auto', 'tape', or 'scan'")
        return name

    def _spec_max_mode(self) -> int:
        if not self._specs:
            return 0
        return max(max(abs(int(spec.m)), abs(int(spec.n))) for spec in self._specs)

    def _has_stellarator_asymmetric_parameter_specs(self) -> bool:
        return any(str(spec.kind).lower() in ("rs", "zc") for spec in getattr(self, "_specs", ()))

    def _has_stellarator_asymmetric_configuration(self) -> bool:
        if self._has_stellarator_asymmetric_parameter_specs():
            return True
        get_bool = getattr(self._indata, "get_bool", None)
        if callable(get_bool):
            try:
                return bool(get_bool("LASYM", False))
            except Exception:
                pass
        return bool(getattr(getattr(self._static, "cfg", None), "lasym", False))

    def _resolve_optimizer_method(self, method: str, scipy_lsmr_maxiter: int | None) -> tuple[str, int | None, str | None]:
        """Resolve optimizer method aliases and the opt-in automatic policy.

        ``method="auto"`` is intentionally conservative and device-preserving:
        it chooses the matrix-free trust-region path for profiled high-mode,
        stellarator-symmetric QS CPU/default-backend lanes where cold-process
        and memory-pressure profiles motivated the option. QI currently stays
        on dense SciPy unless matrix-free is requested explicitly because QI
        Boozer/bounce residual JVPs can be non-finite in cleanup stages. It does
        not guarantee the fastest warm wall time for every run, and it never
        moves work between CPU and GPU; explicit device choices are preserved.
        """

        method_key = str(method).strip().lower().replace("-", "_")
        aliases = {
            "matrix_free": "scipy_matrix_free",
            "scipy_mf": "scipy_matrix_free",
            "trf": "scipy",
        }
        method_key = aliases.get(method_key, method_key)
        scalar_auto_requested = method_key in ("auto_scalar", "auto_adjoint", "adaptive_scalar", "adaptive_adjoint")
        if method_key not in ("auto", "adaptive") and not scalar_auto_requested:
            return method_key, scipy_lsmr_maxiter, None

        if self._has_stellarator_asymmetric_configuration():
            prefix = "auto_scalar" if scalar_auto_requested else "auto"
            return "scipy", scipy_lsmr_maxiter, f"{prefix}:dense-lasym"

        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        helicity_m = None if self._helicity_m is None else int(self._helicity_m)
        helicity_n = None if self._helicity_n is None else int(self._helicity_n)
        if self._spec_max_mode() >= 3 and self._objective_family in ("qs", "qi"):
            if scalar_auto_requested:
                suffix = f"{backend}-" if backend in ("gpu", "cuda", "rocm", "tpu", "metal") else ""
                return "scalar_trust", scipy_lsmr_maxiter, f"auto_scalar:{suffix}high-mode-scalar-trust"
            if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
                return "scipy", scipy_lsmr_maxiter, f"auto:dense-preserves-{backend}"
            if self._objective_family == "qi":
                return "scipy", scipy_lsmr_maxiter, "auto:qi-dense-default"
            lsmr_maxiter = 4 if scipy_lsmr_maxiter is None else scipy_lsmr_maxiter
            if helicity_m == 1 and helicity_n == 0:
                family = "qa"
            elif helicity_m == 0 and helicity_n not in (None, 0):
                family = "qp"
            elif helicity_m == 1 and helicity_n not in (None, 0):
                family = "qh"
            else:
                family = "qs"
            return "scipy_matrix_free", lsmr_maxiter, f"auto:{family}-high-mode-matrix-free"

        if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
            prefix = "auto_scalar" if scalar_auto_requested else "auto"
            return "scipy", scipy_lsmr_maxiter, f"{prefix}:dense-preserves-{backend}"
        prefix = "auto_scalar" if scalar_auto_requested else "auto"
        return "scipy", scipy_lsmr_maxiter, f"{prefix}:dense-default"

    def _select_exact_path(self) -> str:
        """Choose the accepted-point differentiation path.

        The established non-scan discrete-adjoint tape is the default on CPU
        and GPU. May 2026 cold and warm ``office`` RTX A4000 profiling showed
        the scan-differentiated exact path can be useful for targeted parity
        studies but is not a robust GPU default for accepted-point Jacobians.
        The environment override ``VMEC_JAX_OPT_EXACT_PATH={tape,scan}``
        remains available for profiling and parity studies.
        """
        requested = getattr(self, "_exact_path_request", None)
        if requested in ("scan", "tape"):
            return str(requested)
        forced = os.getenv("VMEC_JAX_OPT_EXACT_PATH", "").strip().lower()
        if forced in ("scan", "tape"):
            return forced
        return "tape"

    def _use_precomputed_tridi_for_exact_tape(self) -> bool | None:
        """Use precomputed Thomas coefficients for accepted GPU tape solves.

        This is deliberately scoped to accepted-point exact solves. May 2026
        office RTX A4000 profiles show it reduces dense-Jacobian tape cost for
        mode-2 and mode-3 stellarator-symmetric tapes (24 and 48 DOFs), while
        larger parameter spaces can lose more in replay payload cost than they
        gain in preconditioner cost. ``None`` preserves the solver's legacy
        environment-controlled default for CPU/default backends.
        """

        forced = self._env_bool_override("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE")
        if forced is not None:
            return forced
        backend = self._exact_tape_backend_name()
        if backend not in ("gpu", "cuda", "tpu", "rocm"):
            return None
        try:
            max_dofs = int(os.getenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS", "48"))
        except ValueError:
            max_dofs = 48
        if max_dofs < 0:
            return False
        return True if len(self._specs) <= max_dofs else None

    def _use_scan_for_trial_solves(self) -> bool:
        """Return whether trial residual solves should use the scan loop.

        Exact-optimizer trial residuals are short VMEC solves called repeatedly
        by SciPy's trust-region line search.  They do not need an adjoint tape.
        CPU and current ``office`` GPU/CUDA profiles showed the non-scan loop is
        materially faster for high-mode QS trial points because scan pays a
        large cold compile/dispatch cost.  Environment overrides always win.
        """
        forced = os.getenv("VMEC_JAX_OPT_TRIAL_SCAN", "").strip().lower()
        if forced in ("1", "true", "yes", "on", "scan"):
            return True
        if forced in ("0", "false", "no", "off", "loop", "none"):
            return False
        return self._exact_tape_backend_name() in ("tpu",)

    def _exact_tape_backend_name(self) -> str:
        """Return the backend name used for exact-tape optimization policy."""

        backend = str(getattr(self, "_solver_device_name", None) or "").strip().lower()
        if backend:
            return backend
        try:
            jax_module = _compat_module.jax
            return str(jax_module.default_backend()).strip().lower() if jax_module is not None else "cpu"
        except Exception:
            return "cpu"

    def _env_bool_override(self, name: str) -> bool | None:
        value = os.getenv(str(name), "").strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
        return None

    def _gpu_like_exact_tape_backend(self) -> bool:
        return self._exact_tape_backend_name() in ("gpu", "cuda", "rocm", "tpu", "metal")

    def _solver_device_context(self):
        if self._solver_device_name is None:
            return nullcontext()
        try:
            jax_module = _compat_module.jax
            if jax_module is None:
                return nullcontext()
            devices = jax_module.devices(self._solver_device_name)
            if not devices:
                return nullcontext()
            return jax_module.default_device(devices[0])
        except Exception:
            return nullcontext()

    def _move_to_solver_device(self, value):
        if self._solver_device_name is None:
            return value
        try:
            jax_module = _compat_module.jax
            if jax_module is None:
                return value
            device = jax_module.devices(self._solver_device_name)[0]
            jax_array_type = jax_module.Array
        except Exception:
            return value

        def _move(obj):
            if obj is None or isinstance(obj, (str, bytes, int, float, complex, bool)):
                return obj
            if isinstance(obj, (np.ndarray, jax_array_type)):
                return jax_module.device_put(obj, device)
            if is_dataclass(obj) and not isinstance(obj, type):
                return replace(
                    obj,
                    **{field.name: _move(getattr(obj, field.name)) for field in fields(obj)},
                )
            if isinstance(obj, dict):
                return {key: _move(val) for key, val in obj.items()}
            if isinstance(obj, list):
                return [_move(item) for item in obj]
            if isinstance(obj, tuple):
                moved = tuple(_move(item) for item in obj)
                if hasattr(obj, "_fields"):
                    return type(obj)(*moved)
                return moved
            return obj

        return _move(value)

    def _run_in_solver_device_context(self, fn, *args, **kwargs):
        if self._solver_device_name is None or self._inside_solver_device_context:
            return fn(*args, **kwargs)
        from .kernels.tomnsp import tomnsps_fft_policy_override

        backend_name = str(self._solver_device_name).strip().lower()
        tomnsps_fft_override = (
            backend_name in ("gpu", "cuda", "rocm", "tpu")
            if os.getenv("VMEC_JAX_TOMNSPS_FFT") is None
            else None
        )
        with ExitStack() as stack:
            stack.enter_context(self._solver_device_context())
            stack.enter_context(tomnsps_fft_policy_override(tomnsps_fft_override))
            self._inside_solver_device_context = True
            try:
                return fn(*args, **kwargs)
            finally:
                self._inside_solver_device_context = False

    def _read_last_array(self, array_key: str, scalar_key: str, default, cast):
        value = self._indata.get(array_key, None)
        if value is not None:
            array_value = np.asarray(value).reshape(-1)
            if int(array_value.size) > 0:
                return cast(array_value[-1])
        return cast(self._indata.get(scalar_key, default))

    def _profile_add(self, name: str, dt: float) -> None:
        if not hasattr(self, "_profile"):
            self._profile = {}
        rec = self._profile.setdefault(name, {"count": 0, "wall_time_s": 0.0})
        rec["count"] = int(rec["count"]) + 1
        rec["wall_time_s"] = float(rec["wall_time_s"]) + float(dt)

    def _profile_add_counter(self, name: str, value: int | float) -> None:
        """Record a diagnostic counter in the profile schema without timing it."""
        self._profile_add(name, float(value))

    _profile_solver_free_boundary_timing = _profiling.profile_solver_free_boundary_timing
    _profile_solver_timing = _profiling.profile_solver_timing
    _profile_exact_tape_solver_timing = _profiling.profile_exact_tape_solver_timing
    _profile_dump = _profiling.profile_dump
    _sync_replay_timing_enabled = staticmethod(_profiling.sync_replay_timing_enabled)
    _profile_async_phase = _profiling.profile_async_phase
    _profile_blocking_phase = _profiling.profile_blocking_phase

    def _make_residuals_eval_fn(self, residuals_fn: Callable) -> Callable:
        """Return the non-differentiating residual evaluator used by callbacks."""
        flag = os.getenv("VMEC_JAX_OPT_JIT_RESIDUALS", "1").strip().lower()
        if flag in ("", "0", "false", "no", "off"):
            return residuals_fn

        @jax.jit
        def _eval(state):
            return jnp.asarray(residuals_fn(state), dtype=jnp.float64)

        return _eval

    def _evaluate_residuals_from_state(self, state: VMECState) -> np.ndarray:
        fn = getattr(self, "_residuals_eval_fn", self._residuals_fn)
        return np.asarray(fn(state), dtype=float)

    _callback_point_id = _state_cache.callback_point_id
    _trace_callback_event = _profiling.trace_callback_event
    _callback_trace_dump = _profiling.callback_trace_dump

    _exact_cache_key = staticmethod(_state_cache.exact_cache_key)
    _remember_initial_state = _state_cache.remember_initial_state

    def _initial_state_from_params(self, params, *, profile_name: str) -> VMECState:
        return _state_cache.initial_state_from_params(
            self,
            params,
            profile_name=profile_name,
            initial_guess_from_boundary_func=initial_guess_from_boundary,
        )

    def _use_jit_initial_state(self) -> bool:
        flag = os.getenv("VMEC_JAX_OPT_JIT_INITIAL_STATE")
        if flag is not None:
            return flag.strip().lower() not in ("", "0", "false", "no", "off")
        # The projected initial-state map is small enough that JIT compile and
        # dispatch overhead dominates cold CPU exact callbacks.  Keep the JIT
        # helper opt-in until a workload has enough same-shape reuse to amortize
        # compilation.
        return False

    def _initial_state_from_params_jit(self, params) -> VMECState | None:
        """Return the projected initial state using a cached JIT helper when safe."""

        if not self._use_jit_initial_state():
            return None
        try:
            from .init_guess import initial_guess_from_boundary as _ig
            from .state import pack_state, unpack_state
        except Exception:
            return None

        helper = getattr(self, "_initial_state_packed_helper", None)
        if helper is None:

            @jax.jit
            def _packed_initial_state(p):
                bdy = self._boundary_from_params(p)
                axis_override = getattr(self, "_initial_axis_override", None)
                if axis_override is None:
                    state = _ig(
                        self._static,
                        bdy,
                        self._indata,
                        vmec_project=True,
                    )
                else:
                    state = _ig(
                        self._static,
                        bdy,
                        self._indata,
                        vmec_project=True,
                        axis_override=axis_override,
                    )
                return jnp.asarray(pack_state(state), dtype=jnp.float64)

            helper = _packed_initial_state
            self._initial_state_packed_helper = helper

        try:
            packed = helper(jnp.asarray(params, dtype=jnp.float64))
            if self._sync_initial_state_projection_enabled():
                packed = jax.block_until_ready(packed)
            return unpack_state(packed, self._layout)
        except Exception:
            return None

    def _sync_initial_state_projection_enabled(self) -> bool:
        """Return whether the JIT initial-state projection should synchronize."""

        flag = os.getenv("VMEC_JAX_OPT_SYNC_INITIAL_STATE", "").strip().lower()
        return flag in ("1", "true", "yes", "on")

    _remember_exact_state = _state_cache.remember_exact_state
    _state_matches_params = _state_cache.state_matches_params
    _remember_exact_residual = _state_cache.remember_exact_residual
    _remember_exact_jacobian = _state_cache.remember_exact_jacobian
    _remember_best_exact_point = _state_cache.remember_best_exact_point
    _append_exact_history_entry = _state_cache.append_exact_history_entry
    _reset_run_state = _state_cache.reset_run_state
    _attach_run_private_payload = _state_cache.attach_run_private_payload
    _initial_run_evaluation = _state_cache.initial_run_evaluation

    def _exact_history_accepts(self, cost: float) -> bool:
        """Return whether an exact callback row should enter accepted history."""

        if not np.isfinite(float(cost)):
            return False
        best_cost = float(getattr(self, "_best_exact_cost", math.inf))
        if not np.isfinite(best_cost):
            return True
        tol = max(1.0e-14, 1.0e-9 * max(1.0, abs(best_cost), abs(float(cost))))
        return float(cost) <= best_cost + tol

    _cached_exact_residual = _state_cache.cached_exact_residual
    _cached_exact_state = _state_cache.cached_exact_state
    _cached_trial_residual = _state_cache.cached_trial_residual
    _remember_trial_residual = _state_cache.remember_trial_residual
    _boundary_from_params = _state_cache.boundary_from_params
    _boundary_from_params_numpy = _state_cache.boundary_from_params_numpy
    _boundary_input_from_params = _state_cache.boundary_input_from_params
    _initial_tangent_cache_key = _state_cache.initial_tangent_cache_key
    _indata_from_params = _state_cache.indata_from_params
    _base_params_vector = _state_cache.base_params_vector

    def _solve_forward(self, params, *, trial: bool = False):
        """Run a forward equilibrium solve."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self._solve_forward, params, trial=trial)
        from .solve import solve_fixed_boundary_residual_iter  # noqa: PLC0415

        t_total = time.perf_counter()
        state0 = self._initial_state_from_params(
            params,
            profile_name="initial_guess_trial" if trial else "initial_guess_forward",
        )
        t_solve = time.perf_counter()
        if trial:
            solver_kwargs = dict(self._trial_solver_kwargs)
            solver_kwargs.setdefault("state_only", bool(solver_kwargs.get("use_scan", False)))
            max_iter, ftol = self._trial_max_iter, self._trial_ftol
        else:
            solver_kwargs = self._exact_solver_kwargs
            max_iter, ftol = self._inner_max_iter, self._inner_ftol
        result = solve_fixed_boundary_residual_iter(
            state0,
            self._static,
            max_iter=max_iter,
            ftol=ftol,
            **solver_kwargs,
        )
        solve_wall_s = time.perf_counter() - t_solve
        self._profile_solver_timing(
            getattr(result, "diagnostics", None),
            profile_prefix="trial_solver" if trial else "forward_exact_solver",
            phase_wall_s=solve_wall_s,
            unattributed_name="solve_forward_trial_unattributed" if trial else "solve_forward_exact_unattributed",
        )
        self._profile_add(
            "solve_forward_trial" if trial else "solve_forward_exact",
            solve_wall_s,
        )
        self._profile_add(
            "solve_forward_trial_total" if trial else "solve_forward_exact_total",
            time.perf_counter() - t_total,
        )
        return result.state

    def _scan_exact_helpers(self):
        """Return JIT-compiled scan residual/Jacobian helpers for accelerator solves."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self._scan_exact_helpers)
        return scan_exact_helpers(self, initial_guess_from_boundary_func=initial_guess_from_boundary)

    def _solve_scan_exact_state(self, params):
        """Run the scan accepted-point solve and remember the final state."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self._solve_scan_exact_state, params)
        return solve_scan_exact_state(self, params)

    def _solve_exact_state(self, params, *, return_payload: bool = False):
        """Run the selected accepted-point exact solve."""
        if self._scan_exact_path == "scan":
            state = self._solve_scan_exact_state(params)
            return (state, None) if return_payload else state
        if return_payload:
            return self._solve_exact_with_tape(params, return_payload=True)
        return self._solve_exact_with_tape(params)

    def _best_exact_state_or_solve(self, params, retained_state: VMECState | None = None):
        state = self._cached_exact_state(params)
        if state is None:
            state = retained_state
        return state if state is not None else self._solve_exact_state(params)

    def _solve_exact_with_tape(self, params, *, return_payload: bool = False, jvp_only: bool = False):
        """Run exact solve + build adjoint tape, with caching."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(
                self._solve_exact_with_tape,
                params,
                return_payload=return_payload,
                jvp_only=jvp_only,
            )
        from .discrete_adjoint import build_residual_checkpoint_tape_direct
        from .init_guess import extract_axis_override_from_state
        from .state import unpack_state

        cache_key = self._exact_cache_key(params)
        if cache_key in self._exact_cache:
            self._profile_add("exact_cache_hit", 0.0)
            state, payload = self._exact_cache[cache_key]
            self._remember_exact_state(cache_key, state)
            return (state, payload) if return_payload else state

        t_total = time.perf_counter()
        state0 = self._initial_state_from_params(params, profile_name="initial_guess_exact")
        axis_override = (
            getattr(self, "_initial_axis_override", None)
            if getattr(self, "_initial_axis_override", None) is not None
            else extract_axis_override_from_state(state0, self._static)
        )
        t_tape = time.perf_counter()
        tape = build_residual_checkpoint_tape_direct(
            state0,
            self._static,
            max_iter=self._inner_max_iter,
            solver_kwargs=self._exact_solver_kwargs,
            indata=self._indata,
            signgs=self._signgs,
            ftol=self._inner_ftol,
            step_size=self._step_size,
            light_history=True,
            store_trace=False,
            store_full_step_traces=False,
            jvp_only=bool(jvp_only),
        )
        tape_build_wall_s = time.perf_counter() - t_tape
        self._profile_add("exact_tape_build", tape_build_wall_s)
        if jvp_only:
            self._profile_add("exact_tape_build_jvp_only", tape_build_wall_s)
        self._profile_exact_tape_solver_timing(tape, tape_build_wall_s)
        t_unpack = time.perf_counter()
        packed_final = jnp.asarray(tape.final_packed_state, dtype=jnp.float64)
        state = unpack_state(packed_final, self._layout)
        payload = {"tape": tape, "axis_override": axis_override, "packed_final": packed_final}
        self._exact_cache.clear()
        if not jvp_only:
            self._exact_cache[cache_key] = (state, payload)
        self._remember_exact_state(cache_key, state)
        self._profile_add("exact_unpack_cache", time.perf_counter() - t_unpack)
        self._profile_add("exact_solve_with_tape_total", time.perf_counter() - t_total)
        if jvp_only:
            self._profile_add("exact_solve_with_tape_jvp_only_total", time.perf_counter() - t_total)
        return (state, payload) if return_payload else state

    def _packed_final_from_exact_payload(self, state, payload):
        """Return the accepted packed state already carried by an exact tape payload."""

        from .state import pack_state

        packed = None
        if isinstance(payload, dict):
            packed = payload.get("packed_final")
            if packed is None:
                tape = payload.get("tape")
                packed = getattr(tape, "final_packed_state", None)
        if packed is None:
            packed = pack_state(state)
        return jnp.asarray(packed, dtype=jnp.float64)

    def _store_jacobian_result(self, exact_param_key, residuals, jac=None, *, source=None, t_total: float):
        """Materialize, profile, and cache an exact Jacobian result."""

        self._last_jacobian_residual = np.asarray(residuals, dtype=float)
        self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
        if jac is None:
            out = np.zeros((int(self._last_jacobian_residual.size), 0), dtype=float)
        else:
            t_host = time.perf_counter()
            out = np.asarray(jac, dtype=float)
            self._profile_add("jacobian_host_materialize", time.perf_counter() - t_host)
            self._last_jacobian_source = source
        self._remember_exact_jacobian(exact_param_key, out, self._last_jacobian_residual)
        self._profile_add("jacobian_total", time.perf_counter() - t_total)
        return out

    def residual_fun(self, params) -> np.ndarray:
        """Exact residual at *params* (builds adjoint tape, cached)."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.residual_fun, params)
        cache_key = self._exact_cache_key(params)
        cached = self._cached_exact_residual(params, cache_key=cache_key)
        if cached is not None:
            self._profile_add("residual_exact_cache_hit", 0.0)
            return cached
        if self._scan_exact_path == "scan":
            # Avoid compiling a second residual-only scan executable.  The exact
            # optimizer immediately needs the same accepted-point state for
            # history/cached residuals, so solve once and evaluate residuals from
            # that state.
            state = self._solve_scan_exact_state(params)
            t0 = time.perf_counter()
            out = self._evaluate_residuals_from_state(state)
            self._profile_add("scan_residual_eval_exact", time.perf_counter() - t0)
            self._remember_exact_residual(cache_key, out)
            return out
        state = self._solve_exact_with_tape(params)
        t_res = time.perf_counter()
        out = self._evaluate_residuals_from_state(state)
        self._profile_add("residual_eval_exact", time.perf_counter() - t_res)
        self._remember_exact_residual(cache_key, out)
        return out

    def forward_residual_fun(self, params) -> np.ndarray:
        """Relaxed residual for line-search trial evaluations."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.forward_residual_fun, params)
        cached = self._cached_trial_residual(params)
        if cached is not None:
            return cached
        exact_cached = self._cached_exact_residual(params)
        if exact_cached is not None:
            self._profile_add("trial_residual_exact_cache_hit", 0.0)
            self._remember_trial_residual(params, exact_cached)
            return np.asarray(exact_cached, dtype=float).reshape(-1)
        state = self._solve_forward(params, trial=True)
        t_res = time.perf_counter()
        out = self._evaluate_residuals_from_state(state)
        self._profile_add("residual_eval_trial", time.perf_counter() - t_res)
        self._remember_trial_residual(params, out)
        return out

    def _state_and_tangent_columns(self, params, *, profile_prefix: str):
        """Return accepted-point state and packed tangent columns as JAX arrays."""
        from .discrete_adjoint import checkpoint_tape_state_jvp_columns

        params = jnp.asarray(params, dtype=jnp.float64)
        state, payload = self._solve_exact_with_tape_for_jvp(params)
        if int(params.size) == 0:
            empty = jnp.zeros((0, int(self._layout.size)), dtype=jnp.float64)
            return state, empty

        initial_tangents = self._initial_tangent_columns(
            params,
            payload["axis_override"],
            profile_prefix=profile_prefix,
        )
        column_chunk = self._lasym_replay_column_chunk(int(params.size))
        if column_chunk is not None:
            self._profile_add(f"{profile_prefix}_replay_column_chunk_{column_chunk}", 0.0)
        t_replay = time.perf_counter()
        final_tangents = checkpoint_tape_state_jvp_columns(
            tape=payload["tape"],
            static=self._static,
            initial_tangents=initial_tangents,
            rebuild_preconditioner=True,
            column_chunk=column_chunk,
        )
        final_tangents = self._profile_async_phase(
            f"{profile_prefix}_tape_replay",
            t_replay,
            final_tangents,
        )
        return state, final_tangents

    _solve_exact_with_tape_for_jvp = solve_exact_with_tape_for_jvp
    _jvp_only_exact_tape_enabled = jvp_only_exact_tape_enabled
    _jvp_only_basepoint_carries_enabled = jvp_only_basepoint_carries_enabled

    def _initial_tangent_columns(self, params, axis_override, *, profile_prefix: str):
        """Return cached packed initial-state tangents for boundary parameters."""

        params = jnp.asarray(params, dtype=jnp.float64)
        if int(params.size) == 0:
            return jnp.zeros((0, int(self._layout.size)), dtype=jnp.float64)

        t_initial = time.perf_counter()
        t_key = time.perf_counter()
        cache_key = self._initial_tangent_cache_key(params)
        self._profile_add(
            f"{profile_prefix}_initial_tangents_cache_key",
            time.perf_counter() - t_key,
        )
        initial_tangents = self._initial_tangent_cache.get(cache_key) if cache_key is not None else None
        if initial_tangents is None:
            self._profile_add(f"{profile_prefix}_initial_tangents_cache_miss", 0.0)
            axis_override = {
                key: jnp.asarray(value, dtype=params.dtype) for key, value in axis_override.items()
            }

            t_linearize = time.perf_counter()
            _, initial_state_linear = jax.linearize(
                lambda p: self._solver_initial_state_packed_from_params(p, axis_override),
                params,
            )
            self._profile_add(
                f"{profile_prefix}_initial_tangents_linearize",
                time.perf_counter() - t_linearize,
            )
            if int(params.size) == 1:
                t_jvp = time.perf_counter()
                initial_tangents = initial_state_linear(jnp.ones_like(params))[None, :]
                initial_tangents = self._profile_async_phase(
                    f"{profile_prefix}_initial_tangents_single_jvp",
                    t_jvp,
                    initial_tangents,
                )
            else:
                directions = self._initial_tangent_directions(params, profile_prefix=profile_prefix)
                t_vmap = time.perf_counter()
                initial_tangents = jax.vmap(initial_state_linear)(directions)
                initial_tangents = self._profile_async_phase(
                    f"{profile_prefix}_initial_tangents_vmap",
                    t_vmap,
                    initial_tangents,
                )
            if cache_key is not None:
                t_store = time.perf_counter()
                self._initial_tangent_cache[cache_key] = initial_tangents
                self._profile_add(
                    f"{profile_prefix}_initial_tangents_cache_store",
                    time.perf_counter() - t_store,
                )
        else:
            self._profile_add(f"{profile_prefix}_initial_tangents_cache_hit", 0.0)
        self._profile_add(
            f"{profile_prefix}_initial_tangents",
            time.perf_counter() - t_initial,
        )
        return initial_tangents

    def _solver_initial_state_packed_from_params(self, params, axis_override):
        """Packed initial state after the solver's setup-time constraints.

        ``solve_fixed_boundary_residual_iter`` applies fixed-boundary edge
        enforcement, axis regularity, and VMEC lambda-axis rules before it
        records the first adjoint trace.  Replay tangents must enter at that
        same ``state_pre`` point rather than at the raw initial-guess state.
        """

        from .init_guess import initial_guess_from_boundary as _ig
        from .solve import (
            _apply_vmec_lambda_axis_rules_to_state,
            _enforce_fixed_boundary_and_axis,
            _mode00_index,
        )
        from .state import pack_state

        bdy = self._boundary_from_params(params)
        state0 = _ig(
            self._static,
            bdy,
            self._indata,
            vmec_project=True,
            axis_override=axis_override,
        )
        modes = getattr(self._static, "modes", None)
        if modes is None:
            return jnp.asarray(pack_state(state0), dtype=jnp.float64)

        idx00 = _mode00_index(modes)
        state0 = _enforce_fixed_boundary_and_axis(
            state0,
            self._static,
            edge_Rcos=jnp.asarray(state0.Rcos)[-1, :],
            edge_Rsin=jnp.asarray(state0.Rsin)[-1, :],
            edge_Zcos=jnp.asarray(state0.Zcos)[-1, :],
            edge_Zsin=jnp.asarray(state0.Zsin)[-1, :],
            enforce_edge=True,
            enforce_lambda_axis=True,
            idx00=idx00,
        )
        state0 = _apply_vmec_lambda_axis_rules_to_state(
            state0,
            enforce_vmec_lambda_axis=True,
            host_update_assembly=False,
            idx00=idx00,
        )
        return jnp.asarray(pack_state(state0), dtype=jnp.float64)

    def _initial_tangent_directions(self, params, *, profile_prefix: str):
        """Return cached identity directions used for dense initial-state JVPs."""

        if not hasattr(self, "_initial_tangent_direction_cache"):
            self._initial_tangent_direction_cache = {}
        dtype = jnp.asarray(params).dtype
        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        cache_key = (int(jnp.asarray(params).size), str(dtype), backend)
        directions = self._initial_tangent_direction_cache.get(cache_key)
        if directions is not None:
            self._profile_add(f"{profile_prefix}_initial_tangents_eye_cache_hit", 0.0)
            return directions

        self._profile_add(f"{profile_prefix}_initial_tangents_eye_cache_miss", 0.0)
        t_eye = time.perf_counter()
        directions = jnp.eye(cache_key[0], dtype=dtype)
        self._initial_tangent_direction_cache[cache_key] = directions
        self._profile_add(f"{profile_prefix}_initial_tangents_eye", time.perf_counter() - t_eye)
        return directions

    _lasym_replay_column_chunk = lasym_replay_column_chunk
    _precompute_linear_operator_initial_tangents_enabled = precompute_linear_operator_initial_tangents_enabled
    _scalar_gradient_initial_tangents_enabled = scalar_gradient_initial_tangents_enabled
    _projected_replay_residuals_enabled = projected_replay_residuals_enabled
    _fused_projected_replay_enabled = staticmethod(fused_projected_replay_enabled)
    _chunked_projected_replay_projection_enabled = chunked_projected_replay_projection_enabled

    def _discrete_jacobian_residual_helper(self, params_size: int, residuals_from_packed, *, jax):
        """Return cached residual/Jacobian projection helper for packed tangents."""

        helper_key = (
            int(params_size),
            int(self._layout.size),
            id(self._residuals_fn),
        )
        t_helper = time.perf_counter()
        helper_cache = self._discrete_jacobian_helper_cache.get(helper_key)
        if helper_cache is None:

            @jax.jit
            def _residual_tangent_jacobian(packed_state, packed_tangents):
                residuals, residual_linear = jax.linearize(residuals_from_packed, packed_state)
                # Keep the residual Jacobian transpose on device. Materializing
                # columns on the host and transposing there is especially costly
                # for GPU exact callbacks.
                return residuals, jax.vmap(residual_linear)(packed_tangents).T

            helper_cache = {
                "residual_tangent_jacobian": _residual_tangent_jacobian,
            }
            self._discrete_jacobian_helper_cache[helper_key] = helper_cache
            self._profile_add("jacobian_residual_tangent_helper_build", time.perf_counter() - t_helper)
        else:
            self._profile_add("jacobian_residual_tangent_helper_cache_hit", time.perf_counter() - t_helper)
        return helper_cache

    def _fused_dynamic_basepoint_projected_replay_helper(
        self,
        *,
        tape,
        params_size: int,
        residuals_from_packed,
        initial_tangents,
        column_chunk: int | None,
        jax,
    ):
        """Return a fused dynamic-basepoint replay/projection helper when eligible."""

        if column_chunk is not None or not self._fused_projected_replay_enabled():
            return None
        stacked = getattr(tape, "stacked_step_traces", None)
        stacked_base_carries = getattr(tape, "dynamic_base_carries_stacked", None)
        static_flags = getattr(tape, "step_trace_static_flags", None)
        if stacked is None or stacked_base_carries is None or static_flags is None:
            return None

        from .discrete_adjoint import (
            _checkpoint_tape_dynamic_basepoint_scan_runner,
            _dynamic_basepoint_payload_shapes_match,
            _replay_column_chunk_default,
            _tridi_policy_cache_value,
            _stacked_trace_signature,
        )

        if not _dynamic_basepoint_payload_shapes_match(stacked, stacked_base_carries):
            return None
        # The fused path intentionally bypasses checkpoint_tape_state_jvp_columns,
        # so defer to the standard replay path whenever explicit or automatic
        # chunking would be active.
        if os.environ.get("VMEC_JAX_REPLAY_COLUMN_CHUNK") is not None:
            return None

        auto_chunk = _replay_column_chunk_default(
            tape=tape,
            tangents=jnp.asarray(initial_tangents),
        )
        if auto_chunk is not None and int(params_size) > int(auto_chunk):
            return None

        helper_key = (
            "fused_dynamic_basepoint_projected_replay",
            int(params_size),
            int(self._layout.size),
            id(self._residuals_fn),
            id(self._static),
            _stacked_trace_signature(stacked),
            _stacked_trace_signature(stacked_base_carries),
            bool(static_flags["apply_lforbal"]),
            bool(static_flags["include_edge_residual"]),
            bool(static_flags["apply_m1_constraints"]),
            bool(static_flags["limit_update_rms"]),
            bool(static_flags["limit_dt_from_force"]),
            bool(static_flags["vmec2000_control"]),
            bool(static_flags["divide_by_scalxc_for_update"]),
            int(static_flags["signgs"]),
            int(static_flags["precond_jmax"]),
            _tridi_policy_cache_value(static_flags.get("preconditioner_use_precomputed_tridi", None)),
            _tridi_policy_cache_value(static_flags.get("preconditioner_use_lax_tridi", None)),
        )
        t_helper = time.perf_counter()
        helper_cache = self._discrete_jacobian_helper_cache.get(helper_key)
        if helper_cache is not None:
            self._profile_add(
                "jacobian_fused_projected_replay_helper_cache_hit",
                time.perf_counter() - t_helper,
            )
            return helper_cache

        run_scan = _checkpoint_tape_dynamic_basepoint_scan_runner(
            static=self._static,
            stacked=stacked,
            stacked_base_carries=stacked_base_carries,
            static_flags=static_flags,
        )

        @jax.jit
        def _fused_project(initial_tangents, packed_state, stacked_base_carries_in, stacked_traces_in):
            tangents = jnp.asarray(initial_tangents)
            carry0 = jax.tree_util.tree_map(lambda x: x[0], stacked_base_carries_in)

            def _zeros_like(arr):
                arr = jnp.asarray(arr)
                return jnp.zeros((tangents.shape[0],) + arr.shape, dtype=arr.dtype)

            carry_tangents0 = (tangents,) + tuple(_zeros_like(arr) for arr in carry0[1:])
            final_carry_tangents = run_scan(
                carry_tangents0,
                stacked_base_carries_in,
                stacked_traces_in,
            )
            residuals, residual_linear = jax.linearize(residuals_from_packed, packed_state)
            return residuals, jax.vmap(residual_linear)(final_carry_tangents[0]).T

        helper_cache = {
            "fused_project": _fused_project,
        }
        self._discrete_jacobian_helper_cache[helper_key] = helper_cache
        self._profile_add("jacobian_fused_projected_replay_helper_build", time.perf_counter() - t_helper)
        return helper_cache

    def _jacobian_fun_projected_replay(self, params, exact_param_key, *, t_total: float) -> np.ndarray:
        """Dense exact Jacobian path that avoids synchronizing full state tangents."""

        from .discrete_adjoint import checkpoint_tape_state_jvp_columns
        from .state import unpack_state

        state, payload = self._solve_exact_with_tape_for_jvp(params)
        packed_final = self._packed_final_from_exact_payload(state, payload)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        if int(params.size) == 0:
            helper_cache = self._discrete_jacobian_residual_helper(
                int(params.size),
                _residuals_from_packed,
                jax=jax,
            )
            residuals = helper_cache["residual_tangent_jacobian"](
                packed_final,
                jnp.zeros((0, int(self._layout.size)), dtype=jnp.float64),
            )[0]
            residuals = jax.block_until_ready(residuals)
            return self._store_jacobian_result(exact_param_key, residuals, t_total=t_total)

        initial_tangents = self._initial_tangent_columns(
            params,
            payload["axis_override"],
            profile_prefix="jacobian",
        )
        column_chunk = self._lasym_replay_column_chunk(int(params.size))
        if column_chunk is not None:
            self._profile_add(f"jacobian_projected_replay_column_chunk_{column_chunk}", 0.0)
        fused_helper = self._fused_dynamic_basepoint_projected_replay_helper(
            tape=payload["tape"],
            params_size=int(params.size),
            residuals_from_packed=_residuals_from_packed,
            initial_tangents=initial_tangents,
            column_chunk=column_chunk,
            jax=jax,
        )
        if fused_helper is not None:
            t_replay = time.perf_counter()
            residuals, jac = fused_helper["fused_project"](
                initial_tangents,
                packed_final,
                payload["tape"].dynamic_base_carries_stacked,
                payload["tape"].stacked_step_traces,
            )
            residuals, jac = self._profile_blocking_phase(
                "jacobian_fused_projected_replay_total",
                t_replay,
                (residuals, jac),
            )
            return self._store_jacobian_result(
                exact_param_key, residuals, jac, source="exact_tape_fused_projected_replay", t_total=t_total
            )
        helper_cache = self._discrete_jacobian_residual_helper(
            int(params.size),
            _residuals_from_packed,
            jax=jax,
        )
        if self._chunked_projected_replay_projection_enabled(column_chunk, int(params.size)):
            t_replay = time.perf_counter()
            jac_blocks = []
            residuals = None
            for start in range(0, int(params.size), int(column_chunk)):
                stop = min(start + int(column_chunk), int(params.size))
                final_tangents_chunk = checkpoint_tape_state_jvp_columns(
                    tape=payload["tape"],
                    static=self._static,
                    initial_tangents=initial_tangents[start:stop],
                    rebuild_preconditioner=True,
                    column_chunk=column_chunk,
                    _allow_chunking=False,
                )
                residuals, jac_chunk = helper_cache["residual_tangent_jacobian"](
                    packed_final,
                    final_tangents_chunk,
                )
                jac_blocks.append(jac_chunk)
            jac = jnp.concatenate(jac_blocks, axis=1)
            residuals, jac = self._profile_blocking_phase(
                "jacobian_chunked_projected_replay_projection_total",
                t_replay,
                (residuals, jac),
            )
            source = "exact_tape_chunked_projected_replay_projection"
            return self._store_jacobian_result(exact_param_key, residuals, jac, source=source, t_total=t_total)

        t_replay = time.perf_counter()
        final_tangents = checkpoint_tape_state_jvp_columns(
            tape=payload["tape"],
            static=self._static,
            initial_tangents=initial_tangents,
            rebuild_preconditioner=True,
            column_chunk=column_chunk,
        )
        # Intentionally do not block here.  Let the residual projection consume
        # the device value and block once after projection so GPU callbacks avoid
        # an extra host synchronization between replay and residual tangents.
        self._profile_add("jacobian_projected_tape_replay_dispatch", time.perf_counter() - t_replay)

        t_res = time.perf_counter()
        residuals, jac = helper_cache["residual_tangent_jacobian"](packed_final, final_tangents)
        residuals, jac = self._profile_blocking_phase(
            "jacobian_projected_replay_residual_tangents",
            t_res,
            (residuals, jac),
        )
        self._profile_add("jacobian_projected_replay_total", time.perf_counter() - t_replay)
        return self._store_jacobian_result(
            exact_param_key, residuals, jac, source="exact_tape_projected_replay", t_total=t_total
        )

    def jacobian_fun(self, params) -> np.ndarray:
        """Exact discrete-adjoint Jacobian at *params*."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.jacobian_fun, params)
        self._last_jacobian_source = "exact_tape_replay"
        exact_param_key = self._exact_cache_key(params)
        if self._scan_exact_path == "scan":
            helpers = self._scan_exact_helpers()
            t0 = time.perf_counter()
            residuals, jac = helpers["residual_and_jacobian"](jnp.asarray(params, dtype=jnp.float64))
            self._last_jacobian_residual = np.asarray(residuals, dtype=float)
            self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
            # Avoid a second accepted-point scan solve when the history metrics
            # can be reconstructed from the residual vector.  This is the common
            # QA/QH/QP/QI fixed-boundary optimization path; the state cache is
            # still populated for custom residuals or iota-tracked histories.
            if not self._can_build_history_from_residuals():
                self._solve_scan_exact_state(params)
            out = np.asarray(jac, dtype=float)
            self._last_jacobian_source = "scan_exact_replay"
            self._profile_add("scan_jacobian_total", time.perf_counter() - t0)
            return out
        from .state import pack_state, unpack_state

        t_total = time.perf_counter()
        cached_jacobian = getattr(self, "_exact_jacobian_cache", {}).get(exact_param_key)
        if cached_jacobian is not None:
            jac_cached, residual_cached = cached_jacobian
            self._last_jacobian_residual = np.asarray(residual_cached, dtype=float).reshape(-1)
            self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
            self._last_jacobian_source = "jacobian_cache_hit"
            self._profile_add("jacobian_cache_hit", 0.0)
            self._profile_add("jacobian_total", time.perf_counter() - t_total)
            return np.asarray(jac_cached, dtype=float).copy()

        params = jnp.asarray(params, dtype=jnp.float64)
        if self._projected_replay_residuals_enabled(int(params.size)):
            return self._jacobian_fun_projected_replay(params, exact_param_key, t_total=t_total)

        state, final_tangents = self._state_and_tangent_columns(
            params,
            profile_prefix="jacobian",
        )
        packed_final = jnp.asarray(pack_state(state), dtype=jnp.float64)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        helper_cache = self._discrete_jacobian_residual_helper(
            int(params.size),
            _residuals_from_packed,
            jax=jax,
        )

        t_res = time.perf_counter()
        residuals, jac = helper_cache["residual_tangent_jacobian"](packed_final, final_tangents)
        residuals, jac = self._profile_blocking_phase(
            "jacobian_residual_tangents",
            t_res,
            (residuals, jac),
        )
        return self._store_jacobian_result(exact_param_key, residuals, jac, source="exact_tape_replay", t_total=t_total)

    def state_tangent_columns_fun(self, params) -> tuple[VMECState, np.ndarray]:
        """Return the accepted-point state and packed state tangent columns.

        The tangent columns use the same frozen-axis initial-state convention
        and checkpoint tape replay as :meth:`jacobian_fun`. The returned array
        has shape ``(n_parameters, state.layout.size)``.
        """
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(
                self.state_tangent_columns_fun,
                params,
            )

        t_total = time.perf_counter()
        state, final_tangents = self._state_and_tangent_columns(
            params,
            profile_prefix="state_tangent",
        )
        out = np.asarray(final_tangents, dtype=float)
        self._profile_add("state_tangent_columns_total", time.perf_counter() - t_total)
        return state, out

    def b_cartesian_tangent_columns_fun(
        self,
        params,
        static: VMECStatic | None = None,
        *,
        s_index: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return boundary Cartesian field values and exact tangent columns.

        ``static`` supplies the angular grid for the field evaluation. If it is
        omitted, the optimizer's solve grid is used. The field has shape
        ``(ntheta, nzeta, 3)`` and the tangent columns have shape
        ``(ntheta, nzeta, 3, n_parameters)``.
        """
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(
                self.b_cartesian_tangent_columns_fun,
                params,
                static,
                s_index=s_index,
            )
        from .field import b_cartesian_from_state
        from .state import pack_state, unpack_state

        if static is None:
            static = self._static
        params = jnp.asarray(params, dtype=jnp.float64)
        state, state_tangents = self._state_and_tangent_columns(
            params,
            profile_prefix="b_cartesian_tangent",
        )
        packed_final = jnp.asarray(pack_state(state), dtype=jnp.float64)

        def _field_from_packed(packed):
            state_arg = unpack_state(packed, self._layout)
            field = b_cartesian_from_state(
                state_arg,
                static,
                indata=self._indata,
                signgs=self._signgs,
                s_index=s_index,
            )
            return jnp.ravel(field)

        field_flat, field_linear = jax.linearize(_field_from_packed, packed_final)
        nparams = int(params.size)
        if nparams == 0:
            columns = jnp.zeros((0, field_flat.size), dtype=field_flat.dtype)
        else:
            columns = jax.vmap(field_linear)(state_tangents)

        ntheta = int(static.grid.ntheta)
        nzeta = int(static.grid.nzeta)
        field = np.asarray(field_flat).reshape((ntheta, nzeta, 3))
        tangent_columns = np.asarray(columns, dtype=float).reshape((nparams, ntheta, nzeta, 3))
        tangent_columns = np.transpose(tangent_columns, (1, 2, 3, 0))
        return field, tangent_columns

    def objective_and_gradient_fun(self, params) -> tuple[float, np.ndarray]:
        """Exact scalar objective and reverse-discrete-adjoint gradient.

        This computes the gradient of ``0.5 * ||residual_fun(params)||**2``
        by one reverse replay through the VMEC iteration tape, instead of
        replaying one forward tangent column per boundary parameter.  It is
        intentionally exposed separately from :meth:`jacobian_fun` so we can
        profile and validate the adjoint-gradient path before changing the
        default least-squares optimizer.
        """
        return exact_objective_and_gradient(self, params)

    def gradient_fun(self, params) -> np.ndarray:
        """Exact reverse-discrete-adjoint gradient of the scalar objective."""
        return self.objective_and_gradient_fun(params)[1]

    def residual_linear_operator(self, params):
        """Return a matrix-free exact residual Jacobian at ``params``.

        The returned :class:`scipy.sparse.linalg.LinearOperator` implements
        ``J @ v`` with one forward tangent replay and ``J.T @ w`` with one
        reverse replay through the same converged VMEC iteration tape. This is
        the trust-region counterpart to :meth:`objective_and_gradient_fun` and
        avoids materializing the dense ``n_residuals x n_parameters`` Jacobian.
        """

        return build_residual_linear_operator(self, params)

    def _jacobian_fun_tracked(self, params):
        self._last_jacobian_key[0] = self._exact_cache_key(params)
        self._last_jacobian_residual = None
        self._last_jacobian_source = "exact_tape_replay"
        jac = self.jacobian_fun(params)
        key = self._last_jacobian_key[0]
        exact_residual = (
            np.asarray(self._last_jacobian_residual, dtype=float)
            if self._last_jacobian_residual is not None
            else self._cached_exact_residual(cache_key=key)
        )
        cached_state = None
        if self._last_jacobian_residual is not None and self._can_build_history_from_residuals():
            entry = self._history_entry_from_residuals(
                self._last_jacobian_residual,
                wall_time_s=time.perf_counter() - self._wall_t0,
            )
        elif self._scan_exact_path == "scan" and key is not None and key in self._exact_state_cache:
            cached_state = self._exact_state_cache[key]
            entry = self._history_entry_from_state_or_residual(
                cached_state,
                exact_residual,
                wall_time_s=time.perf_counter() - self._wall_t0,
                cache_key=key,
            )
        elif key is not None and key in self._exact_cache:
            cached_state, _ = self._exact_cache[key]
            entry = self._history_entry_from_state_or_residual(
                cached_state,
                exact_residual,
                wall_time_s=time.perf_counter() - self._wall_t0,
                cache_key=key,
            )
        else:
            entry = None
        if entry is not None:
            self._append_exact_history_entry(
                params,
                entry,
                exact_residual=exact_residual,
                state=cached_state,
            )
        elif exact_residual is not None:
            self._remember_best_exact_point(params, exact_residual)
        return jac

    def _exact_residual_after_jacobian(self):
        key = self._last_jacobian_key[0]
        if key is None and self._last_jacobian_residual is not None:
            return np.asarray(self._last_jacobian_residual, dtype=float)
        cached_residual = self._cached_exact_residual(cache_key=key)
        if cached_residual is not None:
            return cached_residual
        if key is None or key not in self._exact_cache:
            return None
        cached_state, _ = self._exact_cache[key]
        residual = self._evaluate_residuals_from_state(cached_state)
        self._remember_exact_residual(key, residual)
        return residual

    def _post_jacobian_clear(self, *, clear_compiled: bool = False):
        """Optionally release compiled replay helpers.

        Exact tapes and solved states are managed by the optimizer caches.  The
        replay/preconditioner JIT helpers are shape-keyed and LRU-bounded, so
        keeping them across accepted points avoids repeated CPU/GPU
        recompilation in long optimizations.  Full release is still available
        through clear_caches().
        """
        from .preconditioner_1d_jax import clear_preconditioner_jit_caches
        from .discrete_adjoint import clear_replay_scan_caches
        from .kernels.numpy_forces import clear_numpy_force_caches

        if clear_compiled:
            clear_replay_scan_caches()
            clear_preconditioner_jit_caches()
            clear_numpy_force_caches()

    def clear_caches(self) -> None:
        """Release JIT and exact-solve caches."""
        for cache_attr in (*self._DICT_CACHE_ATTRS, *self._ORDERED_CACHE_ATTRS):
            cache = getattr(self, cache_attr, None)
            if cache is not None:
                cache.clear()
        self._initial_state_packed_helper = None
        self._initial_tangent_cache.clear()
        if hasattr(self, "_initial_tangent_direction_cache"):
            self._initial_tangent_direction_cache.clear()
        self._last_jacobian_residual = None
        self._post_jacobian_clear(clear_compiled=True)

    def aspect_ratio(self, params) -> float:
        """Return the aspect ratio at *params* (uses exact solve cache)."""
        from .wout import equilibrium_aspect_ratio_from_state

        state = self._solve_exact_state(params)
        return float(np.asarray(equilibrium_aspect_ratio_from_state(state=state, static=self._static)))

    def _qs_from_res(self, res: np.ndarray) -> float:
        """Sum of squared QS residuals only (excludes aspect and iota)."""
        return qs_objective_from_residuals(res, self._residual_history_policy())

    def _residual_history_policy(self) -> ResidualHistoryPolicy:
        """Return the residual-block metadata used for history reconstruction."""

        return ResidualHistoryPolicy(
            aspect_target=getattr(self, "_aspect_target", None),
            aspect_weight=float(getattr(self, "_aspect_weight", 1.0)),
            n_non_qs=int(getattr(self, "_n_non_qs", 1)),
            n_qs=getattr(self, "_n_qs", None),
            has_residual_block_metadata=getattr(self, "_has_residual_block_metadata", None),
            has_iota_callback=getattr(self, "_iota_fn", None) is not None,
        )

    def _can_build_qs_from_residuals(self) -> bool:
        """Return true when residual block metadata identifies QS/objective blocks."""
        return self._residual_history_policy().can_build_qs_from_residuals()

    def _can_build_aspect_from_residuals(self) -> bool:
        """Return true when the first residual encodes weighted aspect error."""
        return self._residual_history_policy().can_build_aspect_from_residuals()

    def _can_build_history_from_residuals(self) -> bool:
        """Return true when residual metadata is enough for history metrics."""
        return self._residual_history_policy().can_build_history_from_residuals()

    def _history_entry_from_residuals(self, res: np.ndarray, *, wall_time_s: float) -> dict:
        """Build a history row without re-solving the accepted scan state."""
        return history_entry_from_residuals(
            res,
            wall_time_s=wall_time_s,
            policy=self._residual_history_policy(),
        )

    def _qs_total_from_residual_or_state(
        self,
        state: VMECState,
        res: np.ndarray | None = None,
    ) -> float:
        """Use residual block metadata for QS totals before expensive state callbacks."""
        if res is not None and self._can_build_qs_from_residuals():
            return self._qs_from_res(np.asarray(res, dtype=float).reshape(-1))
        return self._qs_total_from_state(state, res)

    def _history_entry_from_state_or_residual(
        self,
        state: VMECState,
        res: np.ndarray | None = None,
        *,
        wall_time_s: float,
        cost: float | None = None,
        cache_key: bytes | None = None,
    ) -> dict:
        """Build a history row, preferring exact cached residual data when safe."""
        res_arr = None if res is None else np.asarray(res, dtype=float).reshape(-1)
        if res_arr is not None and self._can_build_history_from_residuals():
            return self._history_entry_from_residuals(res_arr, wall_time_s=wall_time_s)

        if res_arr is None:
            res_arr = self._evaluate_residuals_from_state(state)
            if cache_key is not None:
                self._remember_exact_residual(cache_key, res_arr)

        cost_val = float(0.5 * np.dot(res_arr, res_arr)) if cost is None else float(cost)
        if self._can_build_aspect_from_residuals():
            aspect = float(getattr(self, "_aspect_target")) + float(res_arr[0]) / float(
                getattr(self, "_aspect_weight", 1.0)
            )
        else:
            from .wout import equilibrium_aspect_ratio_from_state

            aspect = float(np.asarray(equilibrium_aspect_ratio_from_state(state=state, static=self._static)))

        entry: dict = {
            "wall_time_s": float(wall_time_s),
            "cost": cost_val,
            "objective": 2.0 * cost_val,
            "qs_objective": self._qs_total_from_residual_or_state(state, res_arr),
            "aspect": aspect,
        }
        iota_fn = getattr(self, "_iota_fn", None)
        if iota_fn is not None:
            entry["iota"] = float(iota_fn(state))
        return entry

    def _qs_total_from_state(self, state: VMECState, res: np.ndarray | None = None) -> float:
        """QS-only objective from a solved state, with metadata-aware fallback."""
        if self._qs_total_from_state_fn is not None:
            return float(self._qs_total_from_state_fn(state))
        if res is not None:
            return self._qs_from_res(np.asarray(res, dtype=float))
        res = self._evaluate_residuals_from_state(state)
        return self._qs_from_res(np.asarray(res, dtype=float))

    def quasisymmetry_objective(self, params) -> float:
        """Return the total QS objective at *params*."""
        state = self._solve_exact_state(params)
        res = self._evaluate_residuals_from_state(state)
        return self._qs_total_from_state(state, res)

    def save_wout(self, path, params=None, *, state: VMECState | None = None) -> None:
        """Write a wout NetCDF file for the equilibrium at *params*.

        Parameters
        ----------
        path:
            Output path for the ``.nc`` file.
        params:
            Boundary parameter vector (zeros = reference boundary). Optional
            when ``state`` is provided.
        state:
            Already-solved exact VMEC state to write. Passing this avoids
            rerunning the equilibrium solve when the optimizer can verify that
            the state was solved for ``params``.

        Notes
        -----
        Uses the exact-solve cache when *params* was previously evaluated.  On
        a cache miss the accepted-point solver settings are used; the relaxed
        trial solver is never used for persisted wout artifacts.
        """
        t0 = time.perf_counter()
        from .driver import FixedBoundaryRun
        from .driver import write_wout_from_fixed_boundary_run

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if state is None:
            if params is None:
                raise ValueError("save_wout requires either params or state")
            state = self._cached_or_solve_exact_state(params)
        elif params is not None and not self._state_matches_params(state, params):
            state = self._cached_or_solve_exact_state(params)
        run = FixedBoundaryRun(
            cfg=self._static.cfg,
            indata=self._indata,
            static=self._static,
            state=state,
            result=None,
            flux=self._flux if hasattr(self._flux, "chipf") else None,
            profiles=None,
            signgs=self._signgs,
        )
        write_wout_from_fixed_boundary_run(str(path), run, include_fsq=False, fast_bcovar=True)
        self._profile_add("write_wout", time.perf_counter() - t0)
        print(f"  Wrote {path}")

    def _cached_or_solve_exact_state(self, params):
        state = self._cached_exact_state(params)
        if state is None:
            state = self._solve_forward(params, trial=False)
            self._remember_exact_state(self._exact_cache_key(params), state)
        return state

    def save_input(self, path, params) -> None:
        """Write a VMEC ``input.*`` namelist for the boundary at ``params``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_indata(path, self._indata_from_params(params))
        print(f"  Wrote {path}")

    def save_history(self, path, result: dict) -> None:
        """Persist the optimisation history to a JSON file.

        Parameters
        ----------
        path:
            Output JSON path.
        result:
            Dict returned by :meth:`run`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result["_history_dump"], f, indent=2)
        print(f"  Wrote {path}")

    def _record_cached_exact_history_entry(
        self,
        params,
        *,
        last_history_key: list,
        cost: float | None = None,
    ) -> bool:
        """Append history from an exact cached state when this is a new accepted point."""

        key = self._exact_cache_key(params)
        if key == last_history_key[0] or key not in self._exact_cache:
            return False
        cached_state, _ = self._exact_cache[key]
        exact_residual = self._cached_exact_residual(cache_key=key)
        kwargs = {
            "wall_time_s": time.perf_counter() - self._wall_t0,
            "cache_key": key,
        }
        if cost is not None:
            kwargs["cost"] = float(cost)
        entry = self._history_entry_from_state_or_residual(
            cached_state,
            exact_residual,
            **kwargs,
        )
        if exact_residual is None:
            exact_residual = self._cached_exact_residual(cache_key=key)
        if self._append_exact_history_entry(
            params,
            entry,
            exact_residual=exact_residual,
            state=cached_state,
        ):
            last_history_key[0] = key
            return True
        return False

    def run(
        self,
        params0,
        *,
        method: str = "gauss_newton",
        max_nfev: int = 10,
        ftol: float = 1e-3,
        gtol: float = 1e-3,
        xtol: float = 1e-3,
        x_scale=None,
        verbose: int = 1,
        iota_fn=None,
        target_iota: float | None = None,
        target_aspect: float | None = None,
        scipy_tr_solver: str | None = "lsmr",
        scipy_lsmr_maxiter: int | None = None,
        lbfgs_step_bound: float | None = 0.01,
        scalar_step_bound: float | None = 0.01,
        scalar_cost_only_trials: bool | None = None,
        trace_callbacks: bool | None = None,
    ) -> dict:
        """Run the configured exact fixed-boundary optimization stage.

        The method dispatch supports dense, matrix-free, scalar-adjoint, and
        Gauss-Newton paths.  The returned dictionary includes the optimizer
        result plus private exact VMEC states and a serializable
        ``"_history_dump"`` payload for :meth:`save_history`.
        """
        self._reset_run_state(trace_callbacks=trace_callbacks, iota_fn=iota_fn)

        params0_arr = np.asarray(params0, dtype=float)
        scalar_cost_only_trials_used: bool | None = None

        state0, entry0, cost0, qs_total0, aspect0 = self._initial_run_evaluation(params0_arr)

        method_requested = str(method).strip().lower().replace("-", "_")
        method_key, scipy_lsmr_maxiter, method_auto_reason = self._resolve_optimizer_method(
            method_requested,
            scipy_lsmr_maxiter,
        )
        if method_auto_reason is not None:
            self._profile_add(f"method_auto_{method_key}", 0.0)
        common = dict(x_scale=x_scale, max_nfev=max_nfev, ftol=ftol, gtol=gtol)
        match method_key:
            case "gauss_newton":
                result = gauss_newton_least_squares(
                    self.residual_fun,
                    self._jacobian_fun_tracked,
                    params0_arr,
                    forward_residual_fun=self.forward_residual_fun,
                    post_jacobian_callback=self._post_jacobian_clear,
                    exact_residual_after_jacobian_fun=self._exact_residual_after_jacobian,
                    **common,
                    xtol=xtol,
                    verbose=verbose,
                )
            case "scalar_trust" | "adjoint_trust" | "gradient_trust":
                result, scalar_cost_only_trials_used = run_scalar_trust_exact_optimizer(
                    self,
                    params0_arr,
                    **common,
                    scalar_step_bound=scalar_step_bound,
                    scalar_cost_only_trials=scalar_cost_only_trials,
                )
            case "lbfgs" | "lbfgs_adjoint":
                result = run_lbfgs_adjoint_exact_optimizer(
                    self,
                    params0_arr,
                    **common,
                    verbose=verbose,
                    lbfgs_step_bound=lbfgs_step_bound,
                )
            case "scipy_matrix_free" | "matrix_free" | "scipy_mf":
                result, scipy_lsmr_maxiter = run_scipy_matrix_free_exact_optimizer(
                    self,
                    params0_arr,
                    **common,
                    xtol=xtol,
                    verbose=verbose,
                    scipy_lsmr_maxiter=scipy_lsmr_maxiter,
                )
            case "scipy":
                result = run_scipy_dense_exact_optimizer(
                    self,
                    params0_arr,
                    **common,
                    xtol=xtol,
                    verbose=verbose,
                    scipy_tr_solver=scipy_tr_solver,
                    scipy_lsmr_maxiter=scipy_lsmr_maxiter,
                )
            case _:
                raise ValueError(f"Unknown optimization method '{method}'.")
        result["method"] = method_key
        result["method_requested"] = method_requested
        result["method_auto_reason"] = method_auto_reason
        self._post_jacobian_clear()

        selected_best_exact = bool(result.pop("_selected_best_exact_point", False))
        optimizer_exception = result.pop("_optimizer_exception", None)
        (
            state_final,
            entry_final,
            cost_final,
            qs_total_final,
            aspect_final,
            final_wall_time_s,
            selected_best_exact,
        ) = _state_cache.evaluate_and_record_final_exact_point(
            self,
            result,
            selected_best_exact=selected_best_exact,
        )

        history_dump = build_run_history_dump(
            label="Optimisation",
            max_nfev=max_nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            method_key=method_key,
            method_requested=method_requested,
            method_auto_reason=method_auto_reason,
            exact_path=self._scan_exact_path,
            scipy_tr_solver=scipy_tr_solver,
            scipy_lsmr_maxiter=scipy_lsmr_maxiter,
            lbfgs_step_bound=lbfgs_step_bound,
            scalar_step_bound=scalar_step_bound,
            scalar_cost_only_trials_used=scalar_cost_only_trials_used,
            solver_device=self._solver_device_name or "default",
            inner_max_iter=int(self._inner_max_iter),
            inner_ftol=float(self._inner_ftol),
            trial_max_iter=int(self._trial_max_iter),
            trial_ftol=float(self._trial_ftol),
            final_wall_time_s=final_wall_time_s,
            result=result,
            cost0=cost0,
            cost_final=cost_final,
            qs_total0=qs_total0,
            qs_total_final=qs_total_final,
            aspect0=aspect0,
            aspect_final=aspect_final,
            history=self._history,
            profile=self._profile_dump(),
            selected_best_exact=selected_best_exact,
            rejected_trial_exact_history_count=int(self._exact_history_rejected_count),
            optimizer_exception=optimizer_exception,
            iota_fn_present=iota_fn is not None,
            entry0=entry0,
            entry_final=entry_final,
            target_iota=target_iota,
            target_aspect=target_aspect,
            callback_trace=(self._callback_trace_dump() if self._callback_trace_enabled else None),
        )

        return self._attach_run_private_payload(
            result,
            state_initial=state0,
            state_final=state_final,
            history_dump=history_dump,
        )

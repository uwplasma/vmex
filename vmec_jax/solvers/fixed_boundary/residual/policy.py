"""Pure policy helpers for the residual-iteration VMEC solve.

This module contains host-side decisions used by
``solve_fixed_boundary_residual_iter``.  It deliberately avoids imports from
``solve.py`` so the large numerical routine can delegate control-flow policy
without creating import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple

import numpy as np

from .... import _solve_runtime
from .config import (
    HEAVY_DUMP_ENVS,
    LIGHT_DUMP_ENVS,
    bad_jacobian_tau_tolerance,
    parse_bad_jacobian_config,
    resolve_chunked_scan_config,
    resolve_dump_history_config,
    resolve_host_residual_metric_config,
)


class HostUpdateAssemblyPolicy(NamedTuple):
    """Host-side selection for assembling update blocks outside JAX."""

    enabled: bool
    auto_enabled: bool


class NumpyPreconditionerApplyPolicy(NamedTuple):
    """Host-side selection for NumPy preconditioner application."""

    enabled: bool
    mode_count: int
    max_iter_cutoff: int
    min_mode_count: int


class RestartFlagPolicy(NamedTuple):
    """Resolved restart/fallback feature toggles for VMEC-style iteration."""

    use_restart_triggers: bool
    use_direct_fallback: bool
    vmecpp_restart: bool


class HostRestartDecision(NamedTuple):
    """One host-loop restart decision and the scalar evidence behind it."""

    fsq: float
    fsq_res: float
    res0: float
    res0_old: float
    bad_growth_streak: int
    pre_restart_reason: str
    huge_initial_forces: bool
    store_checkpoint: bool
    vmecpp_bad_progress: bool


class Vmec2000TimeControlDecision(NamedTuple):
    """VMEC2000-style timestep/restart control decision for one iteration."""

    fsq: float
    fsq0: float
    res0: float
    res1: float
    trace_irst: int
    irst: int
    initialized: bool
    store_checkpoint: bool
    restart: bool
    pre_restart_reason: str


class BadJacobianTauDecision(NamedTuple):
    """Bad-Jacobian decision from tau extrema and finite checks."""

    min_tau: float
    max_tau: float
    bad_jacobian: bool
    finite: bool


class BadJacobianSelection(NamedTuple):
    """Selected bad-Jacobian state/tau verdict after policy arbitration."""

    bad_jacobian: bool
    min_tau: float
    max_tau: float


class ResidualIterHistoryRecord(NamedTuple):
    """Host-loop residual iteration history row used for diagnostics and tests."""

    step: float
    dt_eff: float
    update_rms: Any
    w_curr: float
    w_try: float
    w_try_ratio: float
    restart_path: str
    step_status: str
    restart_reason: str
    pre_restart_reason: str
    time_step: float
    res0: float
    res1: float
    fsq_prev: float
    bad_growth_streak: int
    iter1: int
    iter2: int
    grad_rms: float
    freeb_ivac: int | None
    freeb_ivacskip: int | None
    freeb_full_update: int | None


ResidualFsqSample = tuple[float, float, float, float]

_RESIDUAL_ITER_HISTORY_RECORD_KEYS = (
    "step_history dt_eff_history update_rms_history w_curr_history w_try_history w_try_ratio_history "
    "restart_path_history step_status_history restart_reason_history pre_restart_reason_history "
    "time_step_history res0_history res1_history fsq_prev_history bad_growth_streak_history "
    "iter1_history iter2_history grad_rms_history freeb_ivac_history freeb_ivacskip_history "
    "freeb_full_update_history"
).split()

_RESIDUAL_ITER_TERMINAL_HISTORY_KEYS = _RESIDUAL_ITER_HISTORY_RECORD_KEYS[7:] + (
    "freeb_nestor_reused_history freeb_nestor_solve_time_history freeb_nestor_sample_time_history"
).split()
_RESIDUAL_ITER_STEP_SAMPLE_HISTORY_KEYS = (
    "step_history dt_eff_history update_rms_history w_curr_history w_try_history "
    "w_try_ratio_history restart_path_history"
).split()

_RESIDUAL_ITER_ROLLBACK_HISTORY_KEYS = (
    "include_edge_history zero_m1_history bcovar_update_history "
    "w_history fsqr2_history fsqz2_history fsql2_history "
    "r00_history z00_history wb_history wp_history w_vmec_history "
    "rz_norm_history f_norm1_history gcr2_p_history gcz2_p_history gcl2_p_history "
    "fsq1_history fsqr1_history fsqz1_history fsql1_history "
    "min_tau_history max_tau_history bad_jacobian_history step_history "
    "dt_eff_history update_rms_history w_curr_history w_try_history w_try_ratio_history "
    "restart_path_history step_status_history restart_reason_history pre_restart_reason_history "
    "time_step_history res0_history res1_history fsq_prev_history "
    "bad_growth_streak_history iter1_history iter2_history "
    "freeb_ivac_history freeb_ivacskip_history freeb_full_update_history grad_rms_history"
).split()


def pop_residual_iter_rollback_histories(histories: tuple[list[Any], ...]) -> None:
    """Pop one optimistic history row from every aligned residual history list."""

    for hist in histories:
        if hist:
            hist.pop()


_RESIDUAL_ITER_OBJECT_HISTORY_DIAGNOSTICS = (
    "step_status_history restart_reason_history pre_restart_reason_history restart_path_history"
).split()
_RESIDUAL_ITER_FLOAT_HISTORY_DIAGNOSTICS = (
    "time_step_history res0_history res1_history fsq_prev_history dt_eff_history w_curr_history "
    "w_try_history w_try_ratio_history min_tau_history max_tau_history r00_history z00_history "
    "wb_history wp_history w_vmec_history freeb_nestor_bnormal_rms_history "
    "freeb_nestor_gsource_rms_history freeb_nestor_bsqvac_rms_history freeb_nestor_solve_time_history "
    "freeb_nestor_sample_time_history freeb_nestor_trial_solve_time_history "
    "freeb_nestor_trial_sample_time_history"
).split()
_RESIDUAL_ITER_INT_HISTORY_DIAGNOSTICS = (
    "bad_growth_streak_history iter1_history iter2_history bcovar_update_history include_edge_history "
    "zero_m1_history bad_jacobian_history freeb_ivac_history freeb_ivacskip_history "
    "freeb_full_update_history freeb_nestor_reused_history freeb_nestor_source_reused_history "
    "freeb_nestor_provider_allows_source_reuse_history freeb_nestor_trial_reused_history "
    "freeb_nestor_trial_failed_history"
).split()
_RESIDUAL_ITER_SCALAR_HISTORY_DIAGNOSTICS = (
    "update_rms_history fsq1_history fsqr1_history fsqz1_history fsql1_history rz_norm_history "
    "f_norm1_history gcr2_p_history gcz2_p_history gcl2_p_history"
).split()

_RESIDUAL_ITER_ALL_HISTORY_KEYS = tuple(
    dict.fromkeys(
        (
            "adjoint_step_trace_history",
            *_RESIDUAL_ITER_HISTORY_RECORD_KEYS,
            *_RESIDUAL_ITER_TERMINAL_HISTORY_KEYS,
            *_RESIDUAL_ITER_ROLLBACK_HISTORY_KEYS,
            *_RESIDUAL_ITER_OBJECT_HISTORY_DIAGNOSTICS,
            *_RESIDUAL_ITER_FLOAT_HISTORY_DIAGNOSTICS,
            *_RESIDUAL_ITER_INT_HISTORY_DIAGNOSTICS,
            *_RESIDUAL_ITER_SCALAR_HISTORY_DIAGNOSTICS,
        )
    )
)

def _empty_residual_iter_history_lists() -> dict[str, list[Any]]:
    return {key: [] for key in _RESIDUAL_ITER_ALL_HISTORY_KEYS}


@dataclass(slots=True)
class ResidualIterationHistories:
    """Aligned host-loop history lists for residual-iteration diagnostics."""

    lists: dict[str, list[Any]] = field(default_factory=_empty_residual_iter_history_lists)

    def __getitem__(self, key: str) -> list[Any]:
        return self.lists[key]

    def many(self, *keys: str) -> tuple[list[Any], ...]:
        return tuple(self.lists[key] for key in keys)

    def record_lists(self, *, free_boundary_enabled: bool) -> dict[str, Any]:
        record_lists = {key: self.lists[key] for key in _RESIDUAL_ITER_HISTORY_RECORD_KEYS}
        record_lists["free_boundary_enabled"] = bool(free_boundary_enabled)
        return record_lists

    def terminal_lists(self, *, free_boundary_enabled: bool) -> dict[str, Any]:
        terminal_lists = {key: self.lists[key] for key in _RESIDUAL_ITER_TERMINAL_HISTORY_KEYS}
        terminal_lists["free_boundary_enabled"] = bool(free_boundary_enabled)
        return terminal_lists

    def step_sample_lists(self) -> dict[str, Any]:
        return {key: self.lists[key] for key in _RESIDUAL_ITER_STEP_SAMPLE_HISTORY_KEYS}

    def rollback_lists(self) -> tuple[list[Any], ...]:
        return tuple(self.lists[key] for key in _RESIDUAL_ITER_ROLLBACK_HISTORY_KEYS)

    def append_bad_jacobian(self, track_history: bool, min_tau: float, max_tau: float, bad_flag: bool) -> None:
        if bool(track_history):
            self.lists["min_tau_history"].append(float(min_tau))
            self.lists["max_tau_history"].append(float(max_tau))
            self.lists["bad_jacobian_history"].append(int(bool(bad_flag)))

    def append_physical_sample(self, *, track_history: bool, fsq: ResidualFsqSample, vmec_scalars: Any) -> None:
        for key, value in zip(("w_history", "fsqr2_history", "fsqz2_history", "fsql2_history"), fsq, strict=True):
            self.lists[key].append(float(value))
        if bool(track_history):
            vmec_values = (vmec_scalars.r00, vmec_scalars.z00, vmec_scalars.wb, vmec_scalars.wp, vmec_scalars.w_vmec)
            for key, value in zip(
                ("r00_history", "z00_history", "wb_history", "wp_history", "w_vmec_history"), vmec_values, strict=True
            ):
                self.lists[key].append(float(value))

    def freeb_source_history_lists(self) -> dict[str, list[Any]]:
        return {
            "source_reused_history": self.lists["freeb_nestor_source_reused_history"],
            "provider_allows_source_reuse_history": self.lists["freeb_nestor_provider_allows_source_reuse_history"],
            "bnormal_rms_history": self.lists["freeb_nestor_bnormal_rms_history"],
            "gsource_rms_history": self.lists["freeb_nestor_gsource_rms_history"],
            "bsqvac_rms_history": self.lists["freeb_nestor_bsqvac_rms_history"],
        }

    def freeb_trial_history_lists(self) -> dict[str, list[Any]]:
        return {
            "trial_reused_history": self.lists["freeb_nestor_trial_reused_history"],
            "trial_solve_time_history": self.lists["freeb_nestor_trial_solve_time_history"],
            "trial_sample_time_history": self.lists["freeb_nestor_trial_sample_time_history"],
            "trial_failed_history": self.lists["freeb_nestor_trial_failed_history"],
        }

    def append_preconditioned(
        self,
        *,
        track_history: bool,
        rz_norm: Any,
        f_norm1: Any,
        gcr2_p: Any,
        gcz2_p: Any,
        gcl2_p: Any,
        fsq1: Any,
        fsqr1_safe: Any,
        fsqz1_safe: Any,
        fsql1_safe: Any,
    ) -> bool:
        """Append preconditioned residual channels to this history bundle."""

        return append_preconditioned_residual_history(
            track_history=bool(track_history),
            rz_norm=rz_norm,
            f_norm1=f_norm1,
            gcr2_p=gcr2_p,
            gcz2_p=gcz2_p,
            gcl2_p=gcl2_p,
            fsq1=fsq1,
            fsqr1_safe=fsqr1_safe,
            fsqz1_safe=fsqz1_safe,
            fsql1_safe=fsql1_safe,
            rz_norm_history=self.lists["rz_norm_history"],
            f_norm1_history=self.lists["f_norm1_history"],
            gcr2_p_history=self.lists["gcr2_p_history"],
            gcz2_p_history=self.lists["gcz2_p_history"],
            gcl2_p_history=self.lists["gcl2_p_history"],
            fsq1_history=self.lists["fsq1_history"],
            fsqr1_history=self.lists["fsqr1_history"],
            fsqz1_history=self.lists["fsqz1_history"],
            fsql1_history=self.lists["fsql1_history"],
        )

    def append_zero_update(
        self,
        *,
        track_history: bool,
        restart_path: str,
        step_status: str,
        restart_reason: str,
        pre_restart_reason: str,
        time_step_value: float,
        fsqr: float,
        fsqz: float,
        fsql: float,
        res0: float,
        res1: float,
        fsq_prev: float,
        bad_growth_streak: int,
        iter1: int,
        iter2: int,
        free_boundary_enabled: bool,
        freeb_ivac: int,
        freeb_ivacskip: int,
    ) -> bool:
        """Append a zero-update row to this history bundle."""

        return append_zero_update_history_record(
            track_history=bool(track_history),
            restart_path=restart_path,
            step_status=step_status,
            restart_reason=restart_reason,
            pre_restart_reason=pre_restart_reason,
            time_step_value=time_step_value,
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            res0=res0,
            res1=res1,
            fsq_prev=fsq_prev,
            bad_growth_streak=bad_growth_streak,
            iter1=iter1,
            iter2=iter2,
            free_boundary_enabled=bool(free_boundary_enabled),
            freeb_ivac=freeb_ivac,
            freeb_ivacskip=freeb_ivacskip,
            history_record_lists=self.record_lists(free_boundary_enabled=bool(free_boundary_enabled)),
        )

    def append_step_sample(
        self,
        *,
        track_history: bool,
        step: float,
        dt_eff: float,
        update_rms: Any,
        w_curr: float,
        w_try: float,
        w_try_ratio: float,
        restart_path: str,
    ) -> bool:
        """Append compact accepted-step sample channels to this history bundle."""

        return append_residual_iter_step_sample(
            track_history=bool(track_history),
            step=step,
            dt_eff=dt_eff,
            update_rms=update_rms,
            w_curr=w_curr,
            w_try=w_try,
            w_try_ratio=w_try_ratio,
            restart_path=restart_path,
            history_step_sample_lists=self.step_sample_lists(),
        )

    def append_terminal(
        self,
        *,
        track_history: bool,
        step_status: str,
        restart_reason: str,
        pre_restart_reason: str,
        time_step: float,
        res0: float,
        res1: float,
        fsq_prev: float,
        bad_growth_streak: int,
        iter1: int,
        iter2: int,
        fsqr: float,
        fsqz: float,
        fsql: float,
        free_boundary_enabled: bool,
        freeb_ivac: int,
        freeb_ivacskip: int,
        freeb_reused: bool,
        freeb_solve_time: float,
        freeb_sample_time: float,
    ) -> bool:
        """Append terminal per-iteration channels to this history bundle."""

        if not bool(track_history):
            return False
        append_residual_iter_terminal_history(
            step_status=step_status,
            restart_reason=restart_reason,
            pre_restart_reason=pre_restart_reason,
            time_step=float(time_step),
            res0=float(res0),
            res1=float(res1),
            fsq_prev=float(fsq_prev),
            bad_growth_streak=int(bad_growth_streak),
            iter1=int(iter1),
            iter2=int(iter2),
            fsqr=fsqr,
            fsqz=fsqz,
            fsql=fsql,
            freeb_ivac=freeb_ivac,
            freeb_ivacskip=freeb_ivacskip,
            freeb_reused=freeb_reused,
            freeb_solve_time=freeb_solve_time,
            freeb_sample_time=freeb_sample_time,
            **self.terminal_lists(free_boundary_enabled=bool(free_boundary_enabled)),
        )
        return True

    def diagnostics(self) -> dict[str, Any]:
        diag: dict[str, Any] = {"adjoint_step_trace": self.lists["adjoint_step_trace_history"]}
        diag.update(
            {key: np.asarray(self.lists[key], dtype=object) for key in _RESIDUAL_ITER_OBJECT_HISTORY_DIAGNOSTICS}
        )
        diag.update(
            {key: np.asarray(self.lists[key], dtype=float) for key in _RESIDUAL_ITER_FLOAT_HISTORY_DIAGNOSTICS}
        )
        diag.update(
            {key: np.asarray(self.lists[key], dtype=int) for key in _RESIDUAL_ITER_INT_HISTORY_DIAGNOSTICS}
        )
        scalar_history_array = _solve_runtime._scalar_history_array
        diag.update({key: scalar_history_array(self.lists[key]) for key in _RESIDUAL_ITER_SCALAR_HISTORY_DIAGNOSTICS})
        return diag


def new_residual_iter_histories() -> ResidualIterationHistories:
    """Create aligned residual-iteration history lists."""

    return ResidualIterationHistories()


def append_preconditioned_residual_history(
    *,
    track_history: bool,
    rz_norm: Any,
    f_norm1: Any,
    gcr2_p: Any,
    gcz2_p: Any,
    gcl2_p: Any,
    fsq1: Any,
    fsqr1_safe: Any,
    fsqz1_safe: Any,
    fsql1_safe: Any,
    rz_norm_history: list[Any],
    f_norm1_history: list[Any],
    gcr2_p_history: list[Any],
    gcz2_p_history: list[Any],
    gcl2_p_history: list[Any],
    fsq1_history: list[Any],
    fsqr1_history: list[Any],
    fsqz1_history: list[Any],
    fsql1_history: list[Any],
) -> bool:
    """Append preconditioned residual scalar channels when history is enabled."""

    if not bool(track_history):
        return False
    rz_norm_history.append(rz_norm)
    f_norm1_history.append(f_norm1)
    gcr2_p_history.append(gcr2_p)
    gcz2_p_history.append(gcz2_p)
    gcl2_p_history.append(gcl2_p)
    fsq1_history.append(fsq1)
    fsqr1_history.append(fsqr1_safe)
    fsqz1_history.append(fsqz1_safe)
    fsql1_history.append(fsql1_safe)
    return True


def append_zero_update_history_record(
    *,
    track_history: bool,
    restart_path: str,
    step_status: str,
    restart_reason: str,
    pre_restart_reason: str,
    time_step_value: float,
    fsqr: float,
    fsqz: float,
    fsql: float,
    res0: float,
    res1: float,
    fsq_prev: float,
    bad_growth_streak: int,
    iter1: int,
    iter2: int,
    free_boundary_enabled: bool,
    freeb_ivac: int,
    freeb_ivacskip: int,
    history_record_lists: Mapping[str, Any],
) -> bool:
    """Append a zero-update residual history row for early exits."""

    if not bool(track_history):
        return False
    rec = residual_iter_history_record(
        step=0.0,
        dt_eff=0.0,
        update_rms=0.0,
        w_curr=float(fsqr) + float(fsqz) + float(fsql),
        w_try=float("nan"),
        w_try_ratio=float("nan"),
        restart_path=restart_path,
        step_status=step_status,
        restart_reason=restart_reason,
        pre_restart_reason=pre_restart_reason,
        time_step=time_step_value,
        res0=res0,
        res1=res1,
        fsq_prev=fsq_prev,
        bad_growth_streak=bad_growth_streak,
        iter1=iter1,
        iter2=iter2,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        free_boundary_enabled=free_boundary_enabled,
        freeb_ivac=freeb_ivac,
        freeb_ivacskip=freeb_ivacskip,
    )
    append_residual_iter_history_record(rec, **history_record_lists)
    return True


def append_residual_iter_step_sample(
    *,
    track_history: bool,
    step: float,
    dt_eff: float,
    update_rms: Any,
    w_curr: float,
    w_try: float,
    w_try_ratio: float,
    restart_path: str,
    history_step_sample_lists: Mapping[str, Any],
) -> bool:
    """Append the residual-update proposal channels for one host iteration."""

    if not bool(track_history):
        return False
    history_step_sample_lists["step_history"].append(float(step))
    history_step_sample_lists["dt_eff_history"].append(float(dt_eff))
    history_step_sample_lists["update_rms_history"].append(update_rms)
    history_step_sample_lists["w_curr_history"].append(float(w_curr))
    history_step_sample_lists["w_try_history"].append(float(w_try))
    history_step_sample_lists["w_try_ratio_history"].append(float(w_try_ratio))
    history_step_sample_lists["restart_path_history"].append(str(restart_path))
    return True


def bad_jacobian_tau_decision(
    *,
    min_tau: float | None,
    max_tau: float | None,
    vmec2000_control: bool,
    ptau_tol: float,
) -> BadJacobianTauDecision:
    """Classify a VMEC Jacobian sign change from tau bounds.

    VMEC2000 uses the configured absolute ``ptau`` tolerance.  The modern
    residual controller keeps the historical relative guard used in the solver
    loop to avoid flagging tiny numerical sign changes around zero.
    """

    if min_tau is None or max_tau is None:
        return BadJacobianTauDecision(float("nan"), float("nan"), False, False)

    min_tau_f = float(min_tau)
    max_tau_f = float(max_tau)
    finite = bool(np.isfinite(min_tau_f) and np.isfinite(max_tau_f))
    if not finite:
        return BadJacobianTauDecision(min_tau_f, max_tau_f, False, False)

    if bool(vmec2000_control):
        tau_tol = bad_jacobian_tau_tolerance(ptau_tol=ptau_tol, ptau_tol_rel=0.0, tau_scale=0.0)
    else:
        tau_scale = max(abs(min_tau_f), abs(max_tau_f))
        tau_tol = max(1.0e-12, 1.0e-3 * tau_scale)
    bad_jacobian = (min_tau_f < -tau_tol) and (max_tau_f > tau_tol)
    return BadJacobianTauDecision(min_tau_f, max_tau_f, bool(bad_jacobian), True)


def bad_jacobian_requires_state_jacobian(
    *,
    badjac_use_state: bool,
    dump_ptau_state: bool,
    state_probe: bool,
    ptau_decision: BadJacobianTauDecision | None,
) -> bool:
    """Return whether the expensive state-Jacobian path is needed."""

    return (
        bool(badjac_use_state)
        or bool(dump_ptau_state)
        or bool(state_probe)
        or ptau_decision is None
        or bool(ptau_decision.bad_jacobian)
    )


def select_bad_jacobian_decision(
    *,
    badjac_use_state: bool,
    ptau_decision: BadJacobianTauDecision | None,
    state_decision: BadJacobianTauDecision,
) -> BadJacobianSelection:
    """Select the authoritative bad-Jacobian result for this controller mode."""

    if bool(badjac_use_state):
        return BadJacobianSelection(
            bool(state_decision.bad_jacobian),
            float(state_decision.min_tau),
            float(state_decision.max_tau),
        )
    if ptau_decision is None:
        return BadJacobianSelection(False, float("nan"), float("nan"))
    return BadJacobianSelection(
        bool(ptau_decision.bad_jacobian),
        float(ptau_decision.min_tau),
        float(ptau_decision.max_tau),
    )


def residual_iter_history_diagnostics(namespace: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize residual iteration history lists for the result diagnostics."""

    if isinstance(namespace, ResidualIterationHistories):
        return namespace.diagnostics()

    diag: dict[str, Any] = {"adjoint_step_trace": namespace["adjoint_step_trace_history"]}
    diag.update({key: np.asarray(namespace[key], dtype=object) for key in _RESIDUAL_ITER_OBJECT_HISTORY_DIAGNOSTICS})
    diag.update({key: np.asarray(namespace[key], dtype=float) for key in _RESIDUAL_ITER_FLOAT_HISTORY_DIAGNOSTICS})
    diag.update({key: np.asarray(namespace[key], dtype=int) for key in _RESIDUAL_ITER_INT_HISTORY_DIAGNOSTICS})
    scalar_history_array = _solve_runtime._scalar_history_array
    diag.update({key: scalar_history_array(namespace[key]) for key in _RESIDUAL_ITER_SCALAR_HISTORY_DIAGNOSTICS})
    return diag


class ScanFallbackDecision(NamedTuple):
    fallback: bool
    reasons: tuple[str, ...]
    reason_text: str
    probe_message: str
    bad_jac_count: int
    accepted_frac: float | None
    fsq_min_full: float | None
    fsq_max_full: float | None
    fsq_all_finite: bool


@dataclass(frozen=True)
class ResidualIterStartupPolicy:
    """Resolved non-numerical options for one residual-iteration solve.

    The large residual loop needs many host-side policy decisions before it can
    build grids, profiles, and JAX kernels.  Keeping these decisions in one
    immutable object makes the loop easier to audit and lets tests validate
    environment-driven behavior without executing the VMEC update loop.
    """

    max_iter: int
    step_size: float
    precompile_only: bool
    host_update_assembly: bool
    host_fsq1_norms_on_accelerator: bool
    host_residual_metrics_on_accelerator: bool
    adjoint_trace: bool
    adjoint_trace_mode: str
    preconditioner_use_precomputed_tridi_policy: bool
    preconditioner_use_lax_tridi_policy: bool
    signgs: int
    fsq_total_target: float | None
    lambda_update_scale: float
    enforce_vmec_lambda_axis: bool
    vmec2000_control: bool
    badjac_mode: str
    badjac_use_state: bool
    dump_ptau_state: bool
    light_history: bool
    resume_state_mode: str
    badjac_state_probe: bool
    badjac_initial_state_probe_iters: int
    ptau_tol: float
    ptau_tol_rel: float
    reference_mode: bool
    jit_precompile: bool
    use_restart_triggers: bool
    use_direct_fallback: bool
    vmecpp_restart: bool
    verbose_vmec2000_table: bool
    scan_fallback_enabled: bool
    scan_fallback_iters: int
    scan_fallback_badjac_limit: int
    scan_fallback_fsq_abs: float
    scan_fallback_accept_frac: float
    scan_fallback_fsq_factor: float
    scan_fallback_improve: float
    stage_transition_factor: float
    stage_transition_scale: float
    stage_prev_fsq: float | None
    auto_flip_force: bool
    jit_forces: bool
    use_scan: bool
    force_chunked_scan: bool
    differentiating_scan: bool
    limit_dt_from_force: bool
    limit_update_rms: bool
    backtracking: bool
    strict_update: bool
    dumps_enabled: bool
    dump_any: bool
    track_history: bool
    disabled_jit_for_dumps: bool


def scan_fallback_message(decision: ScanFallbackDecision) -> str:
    """Return the user-facing VMEC2000 scan fallback message.

    The solver owns the actual branch transition back to the host loop.  This
    helper only centralizes the formatting so tests can validate the diagnostic
    contract without executing the full residual iteration.
    """

    return (
        "[solve_fixed_boundary_residual_iter] "
        f"scan fallback -> non-scan ({decision.reason_text})"
        f"{decision.probe_message}"
    )


def resolve_residual_iter_startup_policy(
    *,
    max_iter: Any,
    step_size: Any,
    precompile_only: Any,
    signgs: Any,
    lambda_update_scale: Any,
    enforce_vmec_lambda_axis: Any,
    vmec2000_control: Any,
    reference_mode: Any,
    limit_dt_from_force: Any,
    limit_update_rms: Any,
    backtracking: Any,
    strict_update: Any,
    jit_precompile: Any,
    use_scan: Any,
    host_update_assembly: bool | None,
    backend_name: str,
    scan_backend_name: str,
    state_has_tracer: bool,
    env: Mapping[str, str | None],
    validate_options: Callable[..., Any],
    resolve_tridi_policies: Callable[..., tuple[bool, bool]],
    normalize_adjoint_trace_mode: Callable[[str], str],
    normalize_resume_state_mode: Callable[[str | None], str],
    resolve_scan_fallback_policy: Callable[..., Any],
    preconditioner_use_precomputed_tridi: bool | None,
    preconditioner_use_lax_tridi: bool | None,
    adjoint_trace: bool,
    adjoint_trace_mode: str,
    fsq_total_target: float | None,
    light_history: bool | None,
    resume_state_mode: str | None,
    use_restart_triggers: bool | None,
    use_direct_fallback: bool | None,
    vmecpp_restart: bool,
    verbose_vmec2000_table: bool,
    stage_prev_fsq: float | None,
    stage_transition_factor: float,
    stage_transition_scale: float,
    auto_flip_force: bool,
    jit_forces: bool,
    heavy_dump_envs: tuple[str, ...] = HEAVY_DUMP_ENVS,
    light_dump_envs: tuple[str, ...] = LIGHT_DUMP_ENVS,
) -> ResidualIterStartupPolicy:
    """Resolve host-side startup policy for ``solve_fixed_boundary_residual_iter``.

    The numerical residual iteration should not own stringly-typed environment
    parsing.  This helper keeps those decisions together while accepting a few
    callables from the solver layer to avoid cycles with validation and runtime
    helpers that already have compatibility aliases.
    """

    opts = validate_options(
        max_iter=max_iter,
        step_size=step_size,
        precompile_only=precompile_only,
        signgs=signgs,
        lambda_update_scale=lambda_update_scale,
        enforce_vmec_lambda_axis=enforce_vmec_lambda_axis,
        vmec2000_control=vmec2000_control,
        reference_mode=reference_mode,
        limit_dt_from_force=limit_dt_from_force,
        limit_update_rms=limit_update_rms,
        backtracking=backtracking,
        strict_update=strict_update,
        jit_precompile=jit_precompile,
        use_scan=use_scan,
    )
    allow_host_update_on_accelerator = str(env.get("VMEC_JAX_HOST_UPDATE_ON_ACCELERATOR", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    host_update = host_update_assembly_policy(
        requested=host_update_assembly,
        use_scan=opts.use_scan,
        backend_name=backend_name,
        state_has_tracer=state_has_tracer,
        allow_accelerator=allow_host_update_on_accelerator,
    ).enabled
    host_metric_config = resolve_host_residual_metric_config(
        backend_name=backend_name,
        fsq1_norms_env=env.get("VMEC_JAX_HOST_FSQ1_NORMS", "auto"),
        residual_metrics_env=env.get("VMEC_JAX_HOST_RESIDUAL_METRICS", "auto"),
    )
    adjoint_trace_mode_resolved = normalize_adjoint_trace_mode(adjoint_trace_mode)
    (
        preconditioner_use_precomputed_tridi_policy,
        preconditioner_use_lax_tridi_policy,
    ) = resolve_tridi_policies(
        use_precomputed=preconditioner_use_precomputed_tridi,
        use_lax_tridi=preconditioner_use_lax_tridi,
    )
    badjac_config = parse_bad_jacobian_config(env)
    light_history_resolved = resolve_light_history(
        light_history,
        env_value=str(env.get("VMEC_JAX_LIGHT_HISTORY", "0")),
    )
    restart_flags = resolve_restart_flags(
        use_restart_triggers=use_restart_triggers,
        use_direct_fallback=use_direct_fallback,
        vmecpp_restart=vmecpp_restart,
    )
    scan_fallback_policy = resolve_scan_fallback_policy(
        backend_name=scan_backend_name,
        enabled_env=env.get("VMEC_JAX_SCAN_FALLBACK"),
        iters_env=str(env.get("VMEC_JAX_SCAN_FALLBACK_ITERS", "50")),
        badjac_limit_env=str(env.get("VMEC_JAX_SCAN_FALLBACK_BJAC_LIMIT", "10")),
        fsq_abs_env=str(env.get("VMEC_JAX_SCAN_FALLBACK_FSQ_ABS", "1.0e-2")),
        accept_frac_env=str(env.get("VMEC_JAX_SCAN_FALLBACK_ACCEPTED_FRAC", "0.5")),
        fsq_factor_env=str(env.get("VMEC_JAX_SCAN_FALLBACK_FSQ_FACTOR", "50")),
        improve_env=str(env.get("VMEC_JAX_SCAN_FALLBACK_IMPROVE", "0.1")),
    )

    stage_transition_factor_resolved = float(stage_transition_factor)
    stage_transition_scale_resolved = float(stage_transition_scale)
    stage_prev_fsq_resolved = stage_prev_fsq
    if stage_transition_factor_resolved <= 0.0 or stage_transition_scale_resolved <= 0.0:
        stage_prev_fsq_resolved = None

    auto_flip_force_resolved = bool(auto_flip_force)
    if bool(opts.use_scan) and bool(opts.vmec2000_control) and auto_flip_force_resolved:
        auto_flip_force_resolved = False

    chunked_scan_config = resolve_chunked_scan_config(
        use_scan=bool(opts.use_scan),
        state_has_tracer=state_has_tracer,
        scan_fallback_enabled=bool(scan_fallback_policy.enabled),
        chunked_env=env.get("VMEC_JAX_VMEC2000_CHUNKED", "1"),
    )
    dump_history_config = resolve_dump_history_config(
        env=env,
        jit_forces=bool(jit_forces),
        light_history=bool(light_history_resolved),
        heavy_dump_envs=heavy_dump_envs,
        light_dump_envs=light_dump_envs,
    )

    return ResidualIterStartupPolicy(
        max_iter=int(opts.max_iter),
        step_size=float(opts.step_size),
        precompile_only=bool(opts.precompile_only),
        host_update_assembly=bool(host_update),
        host_fsq1_norms_on_accelerator=bool(host_metric_config.fsq1_norms_on_accelerator),
        host_residual_metrics_on_accelerator=bool(host_metric_config.residual_metrics_on_accelerator),
        adjoint_trace=bool(adjoint_trace),
        adjoint_trace_mode=str(adjoint_trace_mode_resolved),
        preconditioner_use_precomputed_tridi_policy=bool(preconditioner_use_precomputed_tridi_policy),
        preconditioner_use_lax_tridi_policy=bool(preconditioner_use_lax_tridi_policy),
        signgs=int(opts.signgs),
        fsq_total_target=None if fsq_total_target is None else max(0.0, float(fsq_total_target)),
        lambda_update_scale=float(opts.lambda_update_scale),
        enforce_vmec_lambda_axis=bool(opts.enforce_vmec_lambda_axis),
        vmec2000_control=bool(opts.vmec2000_control),
        badjac_mode=str(badjac_config.mode),
        badjac_use_state=bool(badjac_config.use_state),
        dump_ptau_state=bool(badjac_config.dump_ptau_state),
        light_history=bool(dump_history_config.light_history),
        resume_state_mode=normalize_resume_state_mode(resume_state_mode),
        badjac_state_probe=bool(badjac_config.state_probe),
        badjac_initial_state_probe_iters=int(badjac_config.initial_state_probe_iters),
        ptau_tol=float(badjac_config.ptau_tol),
        ptau_tol_rel=float(badjac_config.ptau_tol_rel),
        reference_mode=bool(opts.reference_mode),
        jit_precompile=bool(opts.jit_precompile),
        use_restart_triggers=bool(restart_flags.use_restart_triggers),
        use_direct_fallback=bool(restart_flags.use_direct_fallback),
        vmecpp_restart=bool(restart_flags.vmecpp_restart),
        verbose_vmec2000_table=bool(verbose_vmec2000_table),
        scan_fallback_enabled=bool(chunked_scan_config.scan_fallback_enabled),
        scan_fallback_iters=int(scan_fallback_policy.iters),
        scan_fallback_badjac_limit=int(scan_fallback_policy.badjac_limit),
        scan_fallback_fsq_abs=float(scan_fallback_policy.fsq_abs),
        scan_fallback_accept_frac=float(scan_fallback_policy.accept_frac),
        scan_fallback_fsq_factor=float(scan_fallback_policy.fsq_factor),
        scan_fallback_improve=float(scan_fallback_policy.improve),
        stage_transition_factor=float(stage_transition_factor_resolved),
        stage_transition_scale=float(stage_transition_scale_resolved),
        stage_prev_fsq=stage_prev_fsq_resolved,
        auto_flip_force=bool(auto_flip_force_resolved),
        jit_forces=bool(dump_history_config.jit_forces),
        use_scan=bool(opts.use_scan),
        force_chunked_scan=bool(chunked_scan_config.force_chunked_scan),
        differentiating_scan=bool(chunked_scan_config.differentiating_scan),
        limit_dt_from_force=bool(opts.limit_dt_from_force),
        limit_update_rms=bool(opts.limit_update_rms),
        backtracking=bool(opts.backtracking),
        strict_update=bool(opts.strict_update),
        dumps_enabled=bool(dump_history_config.dumps_enabled),
        dump_any=bool(dump_history_config.dump_any),
        track_history=bool(dump_history_config.track_history),
        disabled_jit_for_dumps=bool(dump_history_config.disabled_jit_for_dumps),
    )


@dataclass(frozen=True)
class Vmec2000ScanOptions:
    scan_print_env: str
    scan_print_mode: str
    scan_print_ordered: bool
    scan_print_chunked: bool
    scan_light: bool
    scan_minimal: bool
    scan_collect_scalars: bool
    scan_collect_print: bool
    scan_core: bool
    scan_trace: bool
    abort_scan_on_badjac: bool
    scan_use_precomputed: bool
    scan_use_lax_tridi: bool
    scan_use_restart_payload: bool
    print_in_scan: bool
    chunked_print: bool


def resolve_light_history(light_history: bool | None, *, env_value: str) -> bool:
    """Resolve the light-history option using the solver's legacy env parsing."""
    if light_history is None:
        light_hist_env = str(env_value).strip().lower()
        return light_hist_env not in ("", "0", "false", "no")
    return bool(light_history)


def resolve_restart_flags(
    *,
    use_restart_triggers: bool | None,
    use_direct_fallback: bool | None,
    vmecpp_restart: bool,
) -> RestartFlagPolicy:
    """Apply residual-iteration restart option defaults."""
    if use_restart_triggers is None:
        use_restart_triggers = True
    if use_direct_fallback is None:
        use_direct_fallback = False
    return RestartFlagPolicy(
        use_restart_triggers=bool(use_restart_triggers),
        use_direct_fallback=bool(use_direct_fallback),
        vmecpp_restart=bool(vmecpp_restart),
    )


def host_update_assembly_policy(
    *,
    requested: bool | None,
    use_scan: bool,
    backend_name: str,
    state_has_tracer: bool,
    allow_accelerator: bool = False,
) -> HostUpdateAssemblyPolicy:
    """Resolve whether the non-scan CPU solve should assemble updates on host."""
    backend = str(backend_name).strip().lower()
    backend_allowed = (backend == "cpu") or (
        bool(allow_accelerator) and backend in ("gpu", "cuda", "rocm", "tpu")
    )
    auto_enabled = (not bool(use_scan)) and (backend == "cpu") and (not bool(state_has_tracer))
    enabled = auto_enabled if requested is None else bool(requested)
    enabled = bool(enabled) and (not bool(use_scan)) and bool(backend_allowed)
    return HostUpdateAssemblyPolicy(enabled=bool(enabled), auto_enabled=bool(auto_enabled))


def numpy_preconditioner_apply_policy(
    *,
    host_update_assembly: bool,
    max_iter: int,
    mpol: int,
    ntor: int,
    max_iter_env: str,
    min_mode_count_env: str,
) -> NumpyPreconditionerApplyPolicy:
    """Resolve when the CPU host loop should use NumPy R/Z preconditioner apply.

    Short CPU solves avoid repeated JAX dispatch by using the NumPy apply path.
    Larger spectral problems also benefit from this path even when the stage
    iteration budget is large; small spectral problems are faster with the JAX
    apply path after compilation. The defaults preserve the short-solve behavior
    and enable NumPy apply for moderate/high mode counts.
    """

    try:
        max_iter_cutoff = max(0, int(str(max_iter_env).strip()))
    except Exception:
        max_iter_cutoff = 240
    try:
        min_mode_count = max(0, int(str(min_mode_count_env).strip()))
    except Exception:
        min_mode_count = 16
    try:
        mode_count = max(0, int(mpol)) * (max(0, int(ntor)) + 1)
    except Exception:
        mode_count = 0

    short_stage = max_iter_cutoff > 0 and int(max_iter) <= max_iter_cutoff
    spectral_stage = min_mode_count > 0 and mode_count >= min_mode_count
    enabled = bool(host_update_assembly) and (short_stage or spectral_stage)
    return NumpyPreconditionerApplyPolicy(
        enabled=bool(enabled),
        mode_count=int(mode_count),
        max_iter_cutoff=int(max_iter_cutoff),
        min_mode_count=int(min_mode_count),
    )


def stage_transition_restart_reason(
    *,
    iter2: int,
    fsq: float,
    pre_restart_reason: str,
    stage_prev_fsq: float | None,
    stage_transition_factor: float,
) -> str:
    """Return a stage-transition restart reason when first-step growth is too large."""
    reason = str(pre_restart_reason)
    if stage_prev_fsq is not None and int(iter2) == 1 and reason == "none":
        try:
            prev_stage_fsq_val = float(stage_prev_fsq)
        except Exception:
            prev_stage_fsq_val = None
        if prev_stage_fsq_val is not None and np.isfinite(prev_stage_fsq_val):
            if float(fsq) > (prev_stage_fsq_val * float(stage_transition_factor)):
                reason = "stage_transition"
    return reason


def vmec2000_scan_options_from_env(
    *,
    verbose: bool,
    vmec2000_control: bool,
    verbose_vmec2000_table: bool,
    light_history: bool,
    scan_minimal_default: bool | None,
    dump_any: bool,
    fsq_total_target: float | None,
    backend_name: str,
    force_chunked_scan_run: bool,
    scan_print_env: str,
    scan_print_mode_env: str,
    scan_print_ordered_env: str,
    scan_print_chunked_env: str,
    scan_light_env: str,
    scan_minimal_env: str,
    scan_core_env: str,
    scan_trace_env: str,
    abort_scan_env: str,
    scan_precompute_env: str,
    tridi_precompute_env: str,
    scan_lax_env: str,
    tridi_solve_env: str,
    scan_restart_payload_env: str,
) -> Vmec2000ScanOptions:
    scan_print_env = str(scan_print_env).strip().lower()
    scan_print_mode = str(scan_print_mode_env).strip().lower()
    scan_print_ordered = _solve_runtime._runtime_env_enabled(scan_print_ordered_env)
    scan_print_chunked = _solve_runtime._runtime_env_enabled(scan_print_chunked_env)
    scan_light = _solve_runtime._runtime_env_enabled(scan_light_env) or bool(light_history)
    scan_minimal_env_l = str(scan_minimal_env).strip().lower()
    if scan_minimal_env_l:
        scan_minimal = _solve_runtime._runtime_env_enabled(scan_minimal_env_l)
    elif scan_minimal_default is not None:
        scan_minimal = bool(scan_minimal_default)
    else:
        # Quiet runs default to the minimal scan history to reduce host traffic.
        scan_minimal = not bool(verbose)
    if dump_any:
        scan_minimal = False
        scan_light = False
    scan_collect_scalars = not scan_minimal
    scan_collect_print = (
        bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table) and scan_collect_scalars
    )
    scan_core = _solve_runtime._default_scan_core(
        scan_core_env=str(scan_core_env).strip().lower(),
        scan_minimal=bool(scan_minimal),
        fsq_total_target=fsq_total_target,
    )
    scan_trace = _solve_runtime._runtime_env_enabled(scan_trace_env)
    abort_scan_on_badjac = _solve_runtime._runtime_env_enabled(abort_scan_env)

    scan_precompute_env_l = str(scan_precompute_env).strip().lower()
    if scan_precompute_env_l:
        scan_use_precomputed = _solve_runtime._runtime_env_enabled(scan_precompute_env_l)
    else:
        tridi_precompute_env_l = str(tridi_precompute_env).strip().lower()
        if tridi_precompute_env_l:
            scan_use_precomputed = _solve_runtime._runtime_env_enabled(tridi_precompute_env_l)
        else:
            # VMEC2000 scan parity is most robust when the Thomas coefficients
            # are materialized once outside the loop.  This is required for
            # converged finite-beta CPU scan solves and is also the fast GPU
            # path; explicit env overrides remain available for bisection.
            scan_use_precomputed = True
    scan_lax_env_l = str(scan_lax_env).strip().lower()
    if scan_lax_env_l:
        scan_use_lax_tridi = _solve_runtime._runtime_env_enabled(scan_lax_env_l)
    elif str(tridi_solve_env).strip():
        scan_use_lax_tridi = str(tridi_solve_env).strip().lower() in (
            "1",
            "true",
            "yes",
            "lax",
            "force",
        )
    else:
        # The fused lax tridiagonal solver is useful for bisection but does not
        # preserve convergence on all VMEC2000 scan decks. Keep the robust
        # Thomas path as the default on every backend.
        scan_use_lax_tridi = False

    scan_restart_payload_env_l = str(scan_restart_payload_env).strip().lower()
    if scan_restart_payload_env_l in ("1", "true", "yes"):
        scan_use_restart_payload = True
    elif scan_restart_payload_env_l in ("0", "false", "no"):
        scan_use_restart_payload = False
    else:
        scan_use_restart_payload = str(backend_name).strip().lower() == "cpu"

    print_in_scan = (
        bool(verbose)
        and bool(vmec2000_control)
        and bool(verbose_vmec2000_table)
        and _solve_runtime._runtime_env_enabled(scan_print_env)
    )
    if scan_minimal:
        print_in_scan = False
    chunked_print = False
    if print_in_scan and scan_print_chunked:
        # Avoid host callbacks inside the scan: we'll print per chunk on host.
        chunked_print = True
        print_in_scan = False
    if force_chunked_scan_run:
        chunked_print = True
        print_in_scan = False
    if scan_print_mode not in ("debug_print", "debug_callback", "io_callback"):
        scan_print_mode = "debug_print"

    return Vmec2000ScanOptions(
        scan_print_env=scan_print_env,
        scan_print_mode=scan_print_mode,
        scan_print_ordered=bool(scan_print_ordered),
        scan_print_chunked=bool(scan_print_chunked),
        scan_light=bool(scan_light),
        scan_minimal=bool(scan_minimal),
        scan_collect_scalars=bool(scan_collect_scalars),
        scan_collect_print=bool(scan_collect_print),
        scan_core=bool(scan_core),
        scan_trace=bool(scan_trace),
        abort_scan_on_badjac=bool(abort_scan_on_badjac),
        scan_use_precomputed=bool(scan_use_precomputed),
        scan_use_lax_tridi=bool(scan_use_lax_tridi),
        scan_use_restart_payload=bool(scan_use_restart_payload),
        print_in_scan=bool(print_in_scan),
        chunked_print=bool(chunked_print),
    )


def scan_fallback_decision(
    *,
    diagnostics: Mapping[str, Any],
    fsqr_history: Any,
    fsqz_history: Any,
    fsql_history: Any,
    max_iter: int,
    fallback_iters: int,
    badjac_limit: int,
    fsq_abs: float,
    accept_frac: float,
    fsq_factor: float,
) -> ScanFallbackDecision:
    """Decide whether a VMEC2000 scan result should fall back to host iteration."""
    try:
        bad_jac_full = diagnostics.get("bad_jacobian_full", None)
    except Exception:
        bad_jac_full = None
    try:
        abort_scan_flag = bool(diagnostics.get("abort_scan", False))
    except Exception:
        abort_scan_flag = False

    probe_iters = min(int(max_iter), int(fallback_iters))
    bad_jac_count = 0
    if bad_jac_full is not None:
        try:
            bad_jac_arr = np.asarray(bad_jac_full).astype(int)
            if probe_iters > 0:
                bad_jac_count = int(np.sum(bad_jac_arr[:probe_iters]))
        except Exception:
            bad_jac_count = 0

    accepted_frac = None
    try:
        accepted_mask = diagnostics.get("accepted_mask", None)
    except Exception:
        accepted_mask = None
    if accepted_mask is not None:
        try:
            accepted_arr = np.asarray(accepted_mask).astype(float)
            if probe_iters > 0 and accepted_arr.size >= probe_iters:
                accepted_frac = float(np.mean(accepted_arr[:probe_iters]))
        except Exception:
            accepted_frac = None

    fsq_min_full = None
    fsq_max_full = None
    fsq_all_finite = True
    try:
        fsqr_diag = diagnostics.get("fsqr_full", None)
        fsqz_diag = diagnostics.get("fsqz_full", None)
        fsql_diag = diagnostics.get("fsql_full", None)
        if fsqr_diag is None or np.asarray(fsqr_diag).size == 0:
            fsqr_diag = fsqr_history
            fsqz_diag = fsqz_history
            fsql_diag = fsql_history
        fsq_full_arr = np.asarray(fsqr_diag) + np.asarray(fsqz_diag) + np.asarray(fsql_diag)
        fsq_min_full = float(np.min(fsq_full_arr))
        fsq_max_full = float(np.max(fsq_full_arr))
        fsq_all_finite = bool(np.all(np.isfinite(fsq_full_arr)))
    except Exception:
        fsq_min_full = None
        fsq_max_full = None
        fsq_all_finite = False

    fsq_min_ok = True if fsq_min_full is None else bool(fsq_min_full > float(fsq_abs))
    fsq_ratio_ok = True
    if fsq_min_full is not None and fsq_max_full is not None:
        if fsq_min_full > 0.0:
            fsq_ratio_ok = (fsq_max_full / fsq_min_full) > float(fsq_factor)

    fallback_reasons: list[str] = []
    if abort_scan_flag and (not fsq_all_finite or fsq_ratio_ok):
        fallback_reasons.append("abort_scan")
    if bad_jac_count > int(badjac_limit) and fsq_min_ok and fsq_ratio_ok:
        fallback_reasons.append(f"bad_jac_count={bad_jac_count} > {int(badjac_limit)}")
    # Note: ijacob alone is not a reliable failure signal; do not fall back
    # solely on ijacob growth.
    if accepted_frac is not None and accepted_frac < float(accept_frac) and fsq_min_ok and fsq_ratio_ok:
        fallback_reasons.append(f"accepted_frac={accepted_frac:.2f} < {float(accept_frac):.2f}")

    probe_msg = ""
    if fallback_reasons:
        try:
            probe_count = int(diagnostics.get("probe_count", 0))
            if probe_count > 0:
                probe_msg = (
                    " "
                    f"(probe_count={probe_count} "
                    f"probe_accept_frac={diagnostics.get('probe_accept_frac', float('nan')):.2f} "
                    f"probe_ratio={diagnostics.get('probe_ratio', float('nan')):.2f} "
                    f"probe_fsq_min={diagnostics.get('probe_fsq_min', float('nan')):.3e})"
                )
        except Exception:
            probe_msg = ""

    reasons = tuple(fallback_reasons)
    return ScanFallbackDecision(
        fallback=bool(reasons),
        reasons=reasons,
        reason_text=", ".join(reasons),
        probe_message=probe_msg,
        bad_jac_count=int(bad_jac_count),
        accepted_frac=accepted_frac,
        fsq_min_full=fsq_min_full,
        fsq_max_full=fsq_max_full,
        fsq_all_finite=bool(fsq_all_finite),
    )


def host_restart_decision(
    *,
    iter2: int,
    iter1: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    fsq1: float,
    fsq_prev: float,
    res0: float,
    bad_growth_streak: int,
    pre_restart_reason: str,
    reference_mode: bool,
    vmec2000_control: bool,
    bad_jacobian: bool,
    stage_prev_fsq: float | None,
    stage_transition_factor: float,
    lmove_axis: bool,
    vmecpp_restart: bool,
    k_preconditioner_update_interval: int,
) -> HostRestartDecision:
    """Evaluate host-loop residual trackers and pre-restart reason."""

    i2 = int(iter2)
    i1 = int(iter1)
    fsq = float(fsqr) + float(fsqz) + float(fsql)
    fsq1_f = float(fsq1)
    fsq_prev_f = float(fsq_prev)
    res0_f = float(res0)
    fsq_res = fsq if bool(reference_mode) else fsq1_f

    if bool(vmec2000_control):
        if (fsq_res <= fsq_prev_f) and np.isfinite(fsq_res):
            res0_f = min(res0_f, fsq_res)
        res0_old = res0_f
    else:
        if (i2 == i1) or (res0_f < 0.0):
            res0_f = fsq_res
        res0_old = res0_f
        res0_f = min(res0_f, fsq_res)

    store_checkpoint = (not bool(vmec2000_control)) and (fsq1_f <= res0_old) and ((i2 - i1) > 10)
    reason = stage_transition_restart_reason(
        iter2=i2,
        fsq=fsq,
        pre_restart_reason=pre_restart_reason,
        stage_prev_fsq=stage_prev_fsq,
        stage_transition_factor=stage_transition_factor,
    )

    huge_initial_forces = False
    if i2 == 1 and bool(lmove_axis):
        huge_initial_forces = (not np.isfinite(fsq)) or (fsq > 1.0e2)

    if fsq_res > 100.0 * max(res0_f, 1e-30):
        bad_growth_next = int(bad_growth_streak) + 1
    else:
        bad_growth_next = 0

    vmecpp_bad_progress = False
    k_update = int(k_preconditioner_update_interval)
    if bool(vmecpp_restart):
        vmecpp_bad_progress = (
            (i2 - i1) > (k_update // 2)
            and (i2 > 2 * k_update)
            and ((float(fsqr) + float(fsqz)) > 1.0e-2)
        )

    if bool(reference_mode):
        if bool(bad_jacobian) and (fsq > 1.0e1):
            reason = "bad_jacobian"
        elif (i2 > i1) and (fsq > 100.0 * max(res0_f, 1e-30)):
            reason = "bad_jacobian"
        elif (
            (i2 - i1) > (k_update // 2)
            and (i2 > 2 * k_update)
            and ((float(fsqr) + float(fsqz)) > 1.0e-2)
        ):
            reason = "bad_progress"
    elif bool(vmec2000_control):
        if bool(bad_jacobian) and (i2 > i1):
            reason = "bad_jacobian"
        elif vmecpp_bad_progress:
            reason = "bad_progress_vmecpp"
    else:
        if vmecpp_bad_progress:
            reason = "bad_progress_vmecpp"
        elif (i2 > (i1 + 8)) and (bad_growth_next >= 2):
            reason = "bad_jacobian"
        elif (
            (i2 - i1) > (k_update // 2)
            and (i2 > 2 * k_update)
            and (fsq1_f > 5.0 * max(res0_f, 1e-30))
            and (fsq1_f > 0.95 * max(fsq_prev_f, 1e-30))
        ):
            reason = "bad_progress"

    return HostRestartDecision(
        fsq=float(fsq),
        fsq_res=float(fsq_res),
        res0=float(res0_f),
        res0_old=float(res0_old),
        bad_growth_streak=int(bad_growth_next),
        pre_restart_reason=reason,
        huge_initial_forces=bool(huge_initial_forces),
        store_checkpoint=bool(store_checkpoint),
        vmecpp_bad_progress=bool(vmecpp_bad_progress),
    )


def vmec2000_time_control_decision(
    *,
    iter2: int,
    iter1: int,
    fsq_prev: float,
    fsq0_curr: float,
    fsq0_prev: float,
    res0: float,
    res1: float,
    bad_jacobian: bool,
    vmec2000_fact: float,
) -> Vmec2000TimeControlDecision:
    """Return host-side VMEC2000 TimeStepControl scalar decisions."""

    i2 = int(iter2)
    i1 = int(iter1)
    fsq = float(fsq_prev)
    fsq0 = float(fsq0_curr)
    res0_f = float(res0)
    res1_f = float(res1)

    irst = 1
    if bool(bad_jacobian) and (i2 > i1):
        # VMEC's irst=2 path uses the previous physical residual.
        irst = 2
        fsq0 = float(fsq0_prev)

    initialized = (i2 == i1) or (res0_f < 0.0) or (res1_f < 0.0)
    if initialized:
        res0_f = fsq
        res1_f = fsq0

    res0_f = min(res0_f, fsq)
    res1_f = min(res1_f, fsq0)
    store_checkpoint = (fsq <= res0_f) and (fsq0 <= res1_f) and (irst == 1)
    trace_irst = irst

    fact = float(vmec2000_fact)
    bad_progress = (fsq > fact * max(res0_f, 1e-30)) or (fsq0 > fact * max(res1_f, 1e-30))
    if (irst == 1) and ((i2 - i1) > 10) and bad_progress:
        irst = 3

    restart = irst != 1
    pre_restart_reason = "none"
    if restart:
        pre_restart_reason = "bad_jacobian" if irst == 2 else "time_control"

    return Vmec2000TimeControlDecision(
        fsq=float(fsq),
        fsq0=float(fsq0),
        res0=float(res0_f),
        res1=float(res1_f),
        trace_irst=int(trace_irst),
        irst=int(irst),
        initialized=bool(initialized),
        store_checkpoint=bool(store_checkpoint),
        restart=bool(restart),
        pre_restart_reason=pre_restart_reason,
    )


def residual_iter_history_record(
    *,
    step: float,
    dt_eff: float,
    update_rms: Any,
    w_curr: float,
    w_try: float,
    w_try_ratio: float,
    restart_path: str,
    step_status: str,
    restart_reason: str,
    pre_restart_reason: str,
    time_step: float,
    res0: float,
    res1: float,
    fsq_prev: float,
    bad_growth_streak: int,
    iter1: int,
    iter2: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    free_boundary_enabled: bool,
    freeb_ivac: int = 0,
    freeb_ivacskip: int = 0,
) -> ResidualIterHistoryRecord:
    """Pack one host residual-iteration history row without mutating lists."""

    freeb_ivac_out = None
    freeb_ivacskip_out = None
    freeb_full_update = None
    if bool(free_boundary_enabled):
        freeb_ivac_out = int(freeb_ivac)
        freeb_ivacskip_out = int(freeb_ivacskip)
        freeb_full_update = 1 if (freeb_ivac_out >= 0 and freeb_ivacskip_out == 0) else 0

    return ResidualIterHistoryRecord(
        step=float(step),
        dt_eff=float(dt_eff),
        update_rms=update_rms,
        w_curr=float(w_curr),
        w_try=float(w_try),
        w_try_ratio=float(w_try_ratio),
        restart_path=str(restart_path),
        step_status=str(step_status),
        restart_reason=str(restart_reason),
        pre_restart_reason=str(pre_restart_reason),
        time_step=float(time_step),
        res0=float(res0),
        res1=float(res1),
        fsq_prev=float(fsq_prev),
        bad_growth_streak=int(bad_growth_streak),
        iter1=int(iter1),
        iter2=int(iter2),
        grad_rms=float(np.sqrt(max(float(fsqr) + float(fsqz) + float(fsql), 0.0))),
        freeb_ivac=freeb_ivac_out,
        freeb_ivacskip=freeb_ivacskip_out,
        freeb_full_update=freeb_full_update,
    )


def append_residual_iter_history_record(
    rec: ResidualIterHistoryRecord,
    *,
    step_history: list,
    dt_eff_history: list,
    update_rms_history: list,
    w_curr_history: list,
    w_try_history: list,
    w_try_ratio_history: list,
    restart_path_history: list,
    step_status_history: list,
    restart_reason_history: list,
    pre_restart_reason_history: list,
    time_step_history: list,
    res0_history: list,
    res1_history: list,
    fsq_prev_history: list,
    bad_growth_streak_history: list,
    iter1_history: list,
    iter2_history: list,
    grad_rms_history: list,
    free_boundary_enabled: bool,
    freeb_ivac_history: list,
    freeb_ivacskip_history: list,
    freeb_full_update_history: list,
) -> None:
    """Append one residual-iteration history record to aligned host lists."""

    step_history.append(rec.step)
    dt_eff_history.append(rec.dt_eff)
    update_rms_history.append(rec.update_rms)
    w_curr_history.append(rec.w_curr)
    w_try_history.append(rec.w_try)
    w_try_ratio_history.append(rec.w_try_ratio)
    restart_path_history.append(rec.restart_path)
    step_status_history.append(rec.step_status)
    restart_reason_history.append(rec.restart_reason)
    pre_restart_reason_history.append(rec.pre_restart_reason)
    time_step_history.append(rec.time_step)
    res0_history.append(rec.res0)
    res1_history.append(rec.res1)
    fsq_prev_history.append(rec.fsq_prev)
    bad_growth_streak_history.append(rec.bad_growth_streak)
    iter1_history.append(rec.iter1)
    iter2_history.append(rec.iter2)
    grad_rms_history.append(rec.grad_rms)
    if bool(free_boundary_enabled):
        freeb_ivac_history.append(rec.freeb_ivac)
        freeb_ivacskip_history.append(rec.freeb_ivacskip)
        freeb_full_update_history.append(rec.freeb_full_update)


def append_residual_iter_terminal_history(
    *,
    step_status: str,
    restart_reason: str,
    pre_restart_reason: str,
    time_step: float,
    res0: float,
    res1: float,
    fsq_prev: float,
    bad_growth_streak: int,
    iter1: int,
    iter2: int,
    fsqr: float,
    fsqz: float,
    fsql: float,
    step_status_history: list,
    restart_reason_history: list,
    pre_restart_reason_history: list,
    time_step_history: list,
    res0_history: list,
    res1_history: list,
    fsq_prev_history: list,
    bad_growth_streak_history: list,
    iter1_history: list,
    iter2_history: list,
    grad_rms_history: list,
    free_boundary_enabled: bool,
    freeb_ivac: int,
    freeb_ivacskip: int,
    freeb_reused: bool,
    freeb_solve_time: float,
    freeb_sample_time: float,
    freeb_ivac_history: list,
    freeb_ivacskip_history: list,
    freeb_full_update_history: list,
    freeb_nestor_reused_history: list,
    freeb_nestor_solve_time_history: list,
    freeb_nestor_sample_time_history: list,
) -> None:
    """Append per-iteration terminal channels that are aligned with force histories."""

    step_status_history.append(step_status)
    restart_reason_history.append(restart_reason)
    pre_restart_reason_history.append(pre_restart_reason)
    time_step_history.append(float(time_step))
    res0_history.append(float(res0))
    res1_history.append(float(res1))
    fsq_prev_history.append(float(fsq_prev))
    bad_growth_streak_history.append(int(bad_growth_streak))
    iter1_history.append(int(iter1))
    iter2_history.append(int(iter2))
    if bool(free_boundary_enabled):
        freeb_ivac_history.append(int(freeb_ivac))
        freeb_ivacskip_history.append(int(freeb_ivacskip))
        freeb_full_update_history.append(1 if (int(freeb_ivac) >= 0 and int(freeb_ivacskip) == 0) else 0)
        freeb_nestor_reused_history.append(1 if bool(freeb_reused) else 0)
        freeb_nestor_solve_time_history.append(float(freeb_solve_time))
        freeb_nestor_sample_time_history.append(float(freeb_sample_time))
    grad_rms_history.append(float(np.sqrt(max(float(fsqr) + float(fsqz) + float(fsql), 0.0))))

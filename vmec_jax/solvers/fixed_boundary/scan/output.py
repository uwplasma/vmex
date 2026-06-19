"""Host-side VMEC2000 scan result postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np

from ...._compat import jnp
from ..results import ScanCarry, SolveVmecResidualResult
from .time_control import scan_fallback_probe_update


@dataclass(frozen=True)
class Vmec2000ScanHistories:
    fsqr: Any
    fsqz: Any
    fsql: Any
    accepted: Any | None = None
    r00: Any | None = None
    z00: Any | None = None
    w_mhd: Any | None = None
    dt: Any | None = None
    bad_jac: Any | None = None
    fsqr1: Any | None = None
    fsqz1: Any | None = None
    fsql1: Any | None = None
    zero_m1: Any | None = None
    include_edge: Any | None = None
    res0: Any | None = None
    res1: Any | None = None
    iter1: Any | None = None
    min_tau: Any | None = None
    max_tau: Any | None = None
    ptau_min: Any | None = None
    ptau_max: Any | None = None
    tau_min_state: Any | None = None
    tau_max_state: Any | None = None
    badjac_ptau: Any | None = None
    badjac_state: Any | None = None


@dataclass(frozen=True)
class Vmec2000ScanPostprocessResult:
    fsqr_full: np.ndarray
    fsqz_full: np.ndarray
    fsql_full: np.ndarray
    accepted_mask: np.ndarray
    accepted_idx: np.ndarray
    conv_mask: np.ndarray
    conv_idx: int
    conv_idx_print: int
    fsqr_history: np.ndarray
    fsqz_history: np.ndarray
    fsql_history: np.ndarray
    w_history: np.ndarray
    final_fsqr: float
    final_fsqz: float
    final_fsql: float
    converged_strict: bool
    converged_total: bool
    fsqr1_history: np.ndarray
    fsqz1_history: np.ndarray
    fsql1_history: np.ndarray
    zero_m1_history: np.ndarray
    include_edge_history: np.ndarray
    time_step_history: np.ndarray
    r00_history: np.ndarray
    w_vmec_history: np.ndarray
    res0_full: np.ndarray
    res1_full: np.ndarray
    iter1_full: np.ndarray
    min_tau_full: np.ndarray
    max_tau_full: np.ndarray
    ptau_min_full: np.ndarray
    ptau_max_full: np.ndarray
    tau_min_state_full: np.ndarray
    tau_max_state_full: np.ndarray
    badjac_ptau_full: np.ndarray
    badjac_state_full: np.ndarray
    bad_jacobian_full: np.ndarray
    fsqr_full_diag: np.ndarray
    fsqz_full_diag: np.ndarray
    fsql_full_diag: np.ndarray
    n_iter_hist: int
    resume_iter_offset: int
    resume_state: dict[str, Any] | None
    freeb_ivac_full: np.ndarray
    freeb_ivacskip_full: np.ndarray
    free_boundary_diag: dict[str, Any]
    probe_count: int
    probe_bad_jac: int
    probe_accept: int
    probe_fsq_start: float
    probe_fsq_min: float
    probe_fsq_max: float
    probe_ratio: float
    probe_accept_frac: float

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "fsqr_full": self.fsqr_full_diag,
            "fsqz_full": self.fsqz_full_diag,
            "fsql_full": self.fsql_full_diag,
            "accepted_mask": self.accepted_mask,
            "converged_iter": self.conv_idx,
            "converged": bool(np.any(self.conv_mask)),
            "converged_strict": self.converged_strict,
            "converged_by_total_fsq": self.converged_total,
            "final_fsqr": self.final_fsqr,
            "final_fsqz": self.final_fsqz,
            "final_fsql": self.final_fsql,
            "fsqr1_history": self.fsqr1_history,
            "fsqz1_history": self.fsqz1_history,
            "fsql1_history": self.fsql1_history,
            "time_step_history": self.time_step_history,
            "r00_history": self.r00_history,
            "w_vmec_history": self.w_vmec_history,
            "zero_m1_history": self.zero_m1_history.astype(int),
            "include_edge_history": self.include_edge_history.astype(int),
            "res0_full": self.res0_full,
            "res1_full": self.res1_full,
            "iter1_full": self.iter1_full,
            "bad_jacobian_full": self.bad_jacobian_full,
            "min_tau_full": self.min_tau_full,
            "max_tau_full": self.max_tau_full,
            "ptau_min_full": self.ptau_min_full,
            "ptau_max_full": self.ptau_max_full,
            "tau_min_state_full": self.tau_min_state_full,
            "tau_max_state_full": self.tau_max_state_full,
            "badjac_ptau_full": self.badjac_ptau_full.astype(int),
            "badjac_state_full": self.badjac_state_full.astype(int),
            "probe_count": self.probe_count,
            "probe_bad_jac": self.probe_bad_jac,
            "probe_accept": self.probe_accept,
            "probe_fsq_start": self.probe_fsq_start,
            "probe_fsq_min": self.probe_fsq_min,
            "probe_fsq_max": self.probe_fsq_max,
            "probe_ratio": self.probe_ratio,
            "probe_accept_frac": self.probe_accept_frac,
            "free_boundary": self.free_boundary_diag,
            "freeb_ivac_full": self.freeb_ivac_full,
            "freeb_ivacskip_full": self.freeb_ivacskip_full,
            "freeb_full_update_full": (self.freeb_ivacskip_full == 0).astype(int),
            "resume_state": self.resume_state,
        }


class Vmec2000ScanStepResult(NamedTuple):
    """Carry and history row emitted by one VMEC2000 scan step."""

    carry: ScanCarry
    history_row: Any


def vmec2000_scan_residual_result(
    *,
    state: Any,
    scan_output: Vmec2000ScanPostprocessResult,
    ftol: float,
    scan_light: bool,
    scan_minimal: bool,
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    resume_state_mode: str,
    fsq_total_target: float | None,
    badjac_use_state: bool,
    badjac_mode: str,
    badjac_state_probe: bool,
    badjac_initial_state_probe_iters: int,
    ijacob: int,
    abort_scan: bool,
    timing_report: dict[str, Any] | None,
) -> SolveVmecResidualResult:
    """Assemble the public residual result for a materialized VMEC2000 scan."""

    return SolveVmecResidualResult(
        state=state,
        n_iter=int(scan_output.w_history.shape[0]),
        w_history=np.asarray(scan_output.w_history),
        fsqr2_history=np.asarray(scan_output.fsqr_history),
        fsqz2_history=np.asarray(scan_output.fsqz_history),
        fsql2_history=np.asarray(scan_output.fsql_history),
        grad_rms_history=np.asarray([], dtype=float),
        step_history=np.asarray([], dtype=float),
        diagnostics={
            "use_scan": True,
            "vmec2000_scan": True,
            "scan_path": "vmec2000",
            "ftol": float(ftol),
            "requested_ftol": float(ftol),
            "light_history": bool(scan_light),
            "scan_minimal": bool(scan_minimal),
            "scan_use_precomputed": bool(scan_use_precomputed),
            "scan_use_lax_tridi": bool(scan_use_lax_tridi),
            "resume_state_mode": str(resume_state_mode),
            "fsq_total_target": fsq_total_target,
            "badjac_use_state": bool(badjac_use_state),
            "badjac_mode": str(badjac_mode),
            "badjac_state_probe": bool(badjac_state_probe),
            "badjac_initial_state_probe_iters": int(badjac_initial_state_probe_iters),
            "ijacob": int(ijacob),
            "abort_scan": bool(abort_scan),
            **scan_output.diagnostics,
            **({"timing": timing_report} if timing_report is not None else {}),
        },
    )


def vmec2000_scan_minimal_history_row(fsqr: Any, fsqz: Any, fsql: Any) -> tuple[Any, Any, Any]:
    """Return the compact scan-history row used for residual-only scans."""
    return (fsqr, fsqz, fsql)


def vmec2000_scan_light_history_row(
    fsqr: Any,
    fsqz: Any,
    fsql: Any,
    accepted: Any,
    r00: Any,
    z00: Any,
    w_mhd: Any,
    time_step: Any,
    bad_jacobian: Any,
) -> tuple[Any, ...]:
    """Return the light scan-history row consumed by ``unpack_vmec2000_scan_histories``."""
    return (fsqr, fsqz, fsql, accepted, r00, z00, w_mhd, time_step, bad_jacobian)


def vmec2000_scan_full_history_row(
    fsqr: Any,
    fsqz: Any,
    fsql: Any,
    fsqr1: Any,
    fsqz1: Any,
    fsql1: Any,
    accepted: Any,
    r00: Any,
    z00: Any,
    w_mhd: Any,
    time_step: Any,
    zero_m1: Any,
    include_edge: Any,
    res0: Any,
    res1: Any,
    iter1: Any,
    bad_jacobian: Any,
    min_tau: Any,
    max_tau: Any,
    min_tau_ptau: Any,
    max_tau_ptau: Any,
    min_tau_state: Any,
    max_tau_state: Any,
    badjac_ptau: Any,
    badjac_state: Any,
) -> tuple[Any, ...]:
    """Return the full VMEC2000 scan-history row in unpacker field order."""
    return (
        fsqr,
        fsqz,
        fsql,
        fsqr1,
        fsqz1,
        fsql1,
        accepted,
        r00,
        z00,
        w_mhd,
        time_step,
        zero_m1,
        include_edge,
        res0,
        res1,
        iter1,
        bad_jacobian,
        min_tau,
        max_tau,
        min_tau_ptau,
        max_tau_ptau,
        min_tau_state,
        max_tau_state,
        badjac_ptau,
        badjac_state,
    )


def vmec2000_scan_step_result(
    *,
    carry_adv: ScanCarry,
    step_fields: Any,
    current_payload: Any,
    selected_payload: Any,
    probe_update: Any,
    checkpoint_update: Any,
    vmec2000_control: bool,
    scan_core: bool,
    do_restart: Any,
    state_only_scan: bool,
    scan_minimal: bool,
    scan_light: bool,
    fsq0_prev_post: Any,
    force_bcovar_post: Any,
    flip_sign: Any,
    iter_offset_post: Any,
    iter1_post: Any,
    res0: Any,
    res1: Any,
    ijacob_post: Any,
    bad_resets_post: Any,
    bad_growth_post: Any,
    r00: Any,
    z00: Any,
    w_mhd: Any,
    conv_now: Any,
    time_step_report: Any,
    zero_m1: Any,
    include_edge: Any,
    bad_jacobian: Any,
    min_tau: Any,
    max_tau: Any,
    min_tau_ptau: Any,
    max_tau_ptau: Any,
    min_tau_state: Any,
    max_tau_state: Any,
    badjac_ptau: Any,
    badjac_state: Any,
) -> Vmec2000ScanStepResult:
    """Build the next scan carry and selected history row for a scan step."""

    fsqr_out = selected_payload.fsqr
    fsqz_out = selected_payload.fsqz
    fsql_out = selected_payload.fsql
    fsqr1_out = selected_payload.fsqr1
    fsqz1_out = selected_payload.fsqz1
    fsql1_out = selected_payload.fsql1
    restart_effective = do_restart if not bool(vmec2000_control) else jnp.asarray(False)
    accepted = jnp.logical_not(do_restart)
    accepted_count = jnp.where(
        jnp.asarray(scan_core),
        carry_adv.accepted_count,
        carry_adv.accepted_count + jnp.asarray(accepted, dtype=jnp.int32),
    )
    cache_valid = selected_payload.cache_valid if bool(vmec2000_control) else jnp.where(
        do_restart,
        jnp.asarray(False),
        current_payload.cache_valid,
    )
    new_carry = ScanCarry(
        state=step_fields.state,
        time_step=time_step_report,
        inv_tau=step_fields.inv_tau,
        fsq_prev=step_fields.fsq_prev,
        fsq0_prev=fsq0_prev_post,
        accepted_count=accepted_count,
        abort_scan=probe_update.abort_scan,
        skip_timecontrol=jnp.asarray(False) if bool(vmec2000_control) else jnp.asarray(do_restart),
        vRcc=step_fields.vRcc,
        vRss=step_fields.vRss,
        vZsc=step_fields.vZsc,
        vZcs=step_fields.vZcs,
        vLsc=step_fields.vLsc,
        vLcs=step_fields.vLcs,
        vRsc=step_fields.vRsc,
        vRcs=step_fields.vRcs,
        vZcc=step_fields.vZcc,
        vZss=step_fields.vZss,
        vLcc=step_fields.vLcc,
        vLss=step_fields.vLss,
        flip_sign=flip_sign,
        iter_offset=iter_offset_post,
        iter1=iter1_post,
        res0=res0,
        res1=res1,
        state_checkpoint=checkpoint_update.state_checkpoint,
        cache_valid=cache_valid,
        cache_precond_diag=current_payload.cache_precond_diag,
        cache_tcon=current_payload.cache_tcon,
        cache_norms=current_payload.cache_norms,
        cache_rz_scale=current_payload.cache_rz_scale,
        cache_l_scale=current_payload.cache_l_scale,
        cache_rz_norm=current_payload.cache_rz_norm,
        cache_f_norm1=current_payload.cache_f_norm1,
        cache_prec_rz_mats=current_payload.cache_rz_mats,
        cache_prec_lam_prec=current_payload.cache_lam_prec,
        force_bcovar_update=jnp.asarray(False) if bool(vmec2000_control) else force_bcovar_post,
        ijacob=ijacob_post,
        bad_resets=bad_resets_post,
        bad_growth=bad_growth_post,
        fsqz_prev=jnp.where(restart_effective, carry_adv.fsqz_prev, fsqz_out),
        r00_prev=r00,
        z00_prev=z00,
        w_mhd_prev=w_mhd,
        converged=carry_adv.converged | conv_now,
        probe_count=probe_update.probe_count,
        probe_bad_jac=probe_update.probe_bad_jac,
        probe_accept=probe_update.probe_accept,
        probe_fsq_min=probe_update.probe_fsq_min,
        probe_fsq_max=probe_update.probe_fsq_max,
        probe_fsq_start=probe_update.probe_fsq_start,
        fallback_active=carry_adv.fallback_active,
        fsqr_prev_phys=jnp.where(restart_effective, carry_adv.fsqr_prev_phys, fsqr_out),
        fsqz_prev_phys=jnp.where(restart_effective, carry_adv.fsqz_prev_phys, fsqz_out),
        fsql_prev_phys=jnp.where(restart_effective, carry_adv.fsql_prev_phys, fsql_out),
        fsqr1_prev=jnp.where(restart_effective, carry_adv.fsqr1_prev, fsqr1_out),
        fsqz1_prev=jnp.where(restart_effective, carry_adv.fsqz1_prev, fsqz1_out),
        fsql1_prev=jnp.where(restart_effective, carry_adv.fsql1_prev, fsql1_out),
        fsqr_checkpoint=checkpoint_update.residuals.fsqr,
        fsqz_checkpoint=checkpoint_update.residuals.fsqz,
        fsql_checkpoint=checkpoint_update.residuals.fsql,
        fsqr1_checkpoint=checkpoint_update.residuals.fsqr1,
        fsqz1_checkpoint=checkpoint_update.residuals.fsqz1,
        fsql1_checkpoint=checkpoint_update.residuals.fsql1,
        edge_Rcos=carry_adv.edge_Rcos,
        edge_Rsin=carry_adv.edge_Rsin,
        edge_Zcos=carry_adv.edge_Zcos,
        edge_Zsin=carry_adv.edge_Zsin,
    )
    if bool(state_only_scan):
        return Vmec2000ScanStepResult(carry=new_carry, history_row=())
    if bool(scan_minimal):
        return Vmec2000ScanStepResult(
            carry=new_carry,
            history_row=vmec2000_scan_minimal_history_row(fsqr_out, fsqz_out, fsql_out),
        )
    if bool(scan_light):
        return Vmec2000ScanStepResult(
            carry=new_carry,
            history_row=vmec2000_scan_light_history_row(
                fsqr_out,
                fsqz_out,
                fsql_out,
                accepted,
                r00,
                z00,
                w_mhd,
                time_step_report,
                bad_jacobian,
            ),
        )
    return Vmec2000ScanStepResult(
        carry=new_carry,
        history_row=vmec2000_scan_full_history_row(
            fsqr_out,
            fsqz_out,
            fsql_out,
            fsqr1_out,
            fsqz1_out,
            fsql1_out,
            accepted,
            r00,
            z00,
            w_mhd,
            time_step_report,
            zero_m1,
            include_edge,
            res0,
            res1,
            iter1_post,
            bad_jacobian,
            min_tau,
            max_tau,
            min_tau_ptau,
            max_tau_ptau,
            min_tau_state,
            max_tau_state,
            badjac_ptau,
            badjac_state,
        ),
    )


def finalize_vmec2000_scan_step(
    *,
    carry_adv: ScanCarry,
    step_fields: Any,
    current_payload: Any,
    selected_payload: Any,
    checkpoint_update: Any,
    scan_fallback_enabled_run: bool,
    scan_core: bool,
    fsq_phys: Any,
    fsq1: Any,
    bad_jacobian: Any,
    abort_scan_on_badjac: bool,
    scan_fallback_iters: Any,
    scan_fallback_badjac_limit: Any,
    scan_fallback_accept_frac: Any,
    scan_fallback_fsq_factor: Any,
    scan_fallback_fsq_abs: Any,
    scan_fallback_improve: Any,
    dtype: Any,
    vmec2000_control: bool,
    do_restart: Any,
    state_only_scan: bool,
    scan_minimal: bool,
    scan_light: bool,
    fsq0_prev_post: Any,
    force_bcovar_post: Any,
    flip_sign: Any,
    iter_offset_post: Any,
    iter1_post: Any,
    res0: Any,
    res1: Any,
    ijacob_post: Any,
    bad_resets_post: Any,
    bad_growth_post: Any,
    r00: Any,
    z00: Any,
    w_mhd: Any,
    conv_now: Any,
    time_step_report: Any,
    zero_m1: Any,
    include_edge: Any,
    min_tau: Any,
    max_tau: Any,
    min_tau_ptau: Any,
    max_tau_ptau: Any,
    min_tau_state: Any,
    max_tau_state: Any,
    badjac_ptau: Any,
    badjac_state: Any,
) -> Vmec2000ScanStepResult:
    """Update fallback probes and build the carry/history row for a scan step."""

    accepted = jnp.logical_not(do_restart)
    probe_update = scan_fallback_probe_update(
        enabled=scan_fallback_enabled_run,
        scan_core=bool(scan_core),
        probe_count=carry_adv.probe_count,
        probe_bad_jac=carry_adv.probe_bad_jac,
        probe_accept=carry_adv.probe_accept,
        probe_fsq_start=carry_adv.probe_fsq_start,
        probe_fsq_min=carry_adv.probe_fsq_min,
        probe_fsq_max=carry_adv.probe_fsq_max,
        fallback_active=carry_adv.fallback_active,
        abort_scan=carry_adv.abort_scan,
        fsq_phys=fsq_phys,
        fsq1=fsq1,
        bad_jacobian=bad_jacobian,
        accepted=accepted,
        abort_scan_on_badjac=abort_scan_on_badjac,
        fallback_iters=scan_fallback_iters,
        badjac_limit=scan_fallback_badjac_limit,
        accept_frac=scan_fallback_accept_frac,
        fsq_factor=scan_fallback_fsq_factor,
        fsq_abs=scan_fallback_fsq_abs,
        improve=scan_fallback_improve,
        dtype=dtype,
    )
    return vmec2000_scan_step_result(
        carry_adv=carry_adv,
        step_fields=step_fields,
        current_payload=current_payload,
        selected_payload=selected_payload,
        probe_update=probe_update,
        checkpoint_update=checkpoint_update,
        vmec2000_control=bool(vmec2000_control),
        scan_core=bool(scan_core),
        do_restart=do_restart,
        state_only_scan=bool(state_only_scan),
        scan_minimal=bool(scan_minimal),
        scan_light=bool(scan_light),
        fsq0_prev_post=fsq0_prev_post,
        force_bcovar_post=force_bcovar_post,
        flip_sign=flip_sign,
        iter_offset_post=iter_offset_post,
        iter1_post=iter1_post,
        res0=res0,
        res1=res1,
        ijacob_post=ijacob_post,
        bad_resets_post=bad_resets_post,
        bad_growth_post=bad_growth_post,
        r00=r00,
        z00=z00,
        w_mhd=w_mhd,
        conv_now=conv_now,
        time_step_report=time_step_report,
        zero_m1=zero_m1,
        include_edge=include_edge,
        bad_jacobian=bad_jacobian,
        min_tau=min_tau,
        max_tau=max_tau,
        min_tau_ptau=min_tau_ptau,
        max_tau_ptau=max_tau_ptau,
        min_tau_state=min_tau_state,
        max_tau_state=max_tau_state,
        badjac_ptau=badjac_ptau,
        badjac_state=badjac_state,
    )


def vmec2000_state_only_scan_diagnostics(
    *,
    carry_final: Any,
    traced: bool,
    ftol: float,
    scan_minimal: bool,
    scan_light: bool,
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    timing_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return diagnostics for a VMEC2000 scan that only returns the final state."""
    diagnostics = {
        "use_scan": True,
        "vmec2000_scan": True,
        "scan_path": "vmec2000",
        "state_only": True,
        "history_mode": "none",
        "history_none": True,
        "ftol": float(ftol),
        "requested_ftol": float(ftol),
        "scan_minimal": bool(scan_minimal),
        "light_history": bool(scan_light),
        "scan_use_precomputed": bool(scan_use_precomputed),
        "scan_use_lax_tridi": bool(scan_use_lax_tridi),
    }
    if timing_report is not None:
        diagnostics["timing"] = timing_report
    if not bool(traced):
        diagnostics.update(
            {
                "abort_scan": bool(np.asarray(carry_final.abort_scan)),
                "converged": bool(np.asarray(carry_final.converged)),
                "ijacob": int(np.asarray(carry_final.ijacob)),
            }
        )
    return diagnostics


def vmec2000_traced_scan_diagnostics(
    *,
    resume_state: dict[str, Any],
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
) -> dict[str, Any]:
    """Return diagnostics for a traced VMEC2000 scan result with no host history."""
    return {
        "use_scan": True,
        "vmec2000_scan": True,
        "scan_path": "vmec2000",
        "traced_scan": True,
        "scan_use_precomputed": bool(scan_use_precomputed),
        "scan_use_lax_tridi": bool(scan_use_lax_tridi),
        "resume_state": resume_state,
    }


def unpack_vmec2000_scan_histories(
    hist: Any,
    *,
    scan_minimal: bool,
    scan_light: bool,
) -> Vmec2000ScanHistories:
    if scan_minimal:
        fsqr_hist, fsqz_hist, fsql_hist = hist
        return Vmec2000ScanHistories(fsqr=fsqr_hist, fsqz=fsqz_hist, fsql=fsql_hist)
    if scan_light:
        (
            fsqr_hist,
            fsqz_hist,
            fsql_hist,
            accepted,
            r00_hist,
            z00_hist,
            w_mhd_hist,
            dt_hist,
            bad_jac_hist,
        ) = hist
        return Vmec2000ScanHistories(
            fsqr=fsqr_hist,
            fsqz=fsqz_hist,
            fsql=fsql_hist,
            accepted=accepted,
            r00=r00_hist,
            z00=z00_hist,
            w_mhd=w_mhd_hist,
            dt=dt_hist,
            bad_jac=bad_jac_hist,
        )
    (
        fsqr_hist,
        fsqz_hist,
        fsql_hist,
        fsqr1_hist,
        fsqz1_hist,
        fsql1_hist,
        accepted,
        r00_hist,
        z00_hist,
        w_mhd_hist,
        dt_hist,
        zero_m1_hist,
        include_edge_hist,
        res0_hist,
        res1_hist,
        iter1_hist,
        bad_jac_hist,
        min_tau_hist,
        max_tau_hist,
        ptau_min_hist,
        ptau_max_hist,
        tau_min_state_hist,
        tau_max_state_hist,
        badjac_ptau_hist,
        badjac_state_hist,
    ) = hist
    return Vmec2000ScanHistories(
        fsqr=fsqr_hist,
        fsqz=fsqz_hist,
        fsql=fsql_hist,
        accepted=accepted,
        r00=r00_hist,
        z00=z00_hist,
        w_mhd=w_mhd_hist,
        dt=dt_hist,
        bad_jac=bad_jac_hist,
        fsqr1=fsqr1_hist,
        fsqz1=fsqz1_hist,
        fsql1=fsql1_hist,
        zero_m1=zero_m1_hist,
        include_edge=include_edge_hist,
        res0=res0_hist,
        res1=res1_hist,
        iter1=iter1_hist,
        min_tau=min_tau_hist,
        max_tau=max_tau_hist,
        ptau_min=ptau_min_hist,
        ptau_max=ptau_max_hist,
        tau_min_state=tau_min_state_hist,
        tau_max_state=tau_max_state_hist,
        badjac_ptau=badjac_ptau_hist,
        badjac_state=badjac_state_hist,
    )


def postprocess_vmec2000_scan_result(
    histories: Vmec2000ScanHistories,
    carry_final: Any,
    *,
    vmec2000_control: bool,
    ftol: float,
    fsq_total_target: float | None,
    max_iter: int,
    scan_minimal: bool,
    scan_light: bool,
    resume_state_mode: str,
    pack_resume_state: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any] | None],
    free_boundary_enabled: bool,
    freeb_nvacskip: int,
    freeb_nvskip0: int,
    iter_offset0: int,
    free_boundary_iter_controls: Callable[[int, int, int], tuple[int, int]],
) -> Vmec2000ScanPostprocessResult:
    fsqr_full = np.asarray(histories.fsqr)
    fsqz_full = np.asarray(histories.fsqz)
    fsql_full = np.asarray(histories.fsql)
    if bool(vmec2000_control) or (histories.accepted is None):
        accepted_mask = np.ones_like(fsqr_full, dtype=bool)
    else:
        accepted_mask = np.asarray(histories.accepted).astype(bool)
    strict_mask = (fsqr_full <= float(ftol)) & (fsqz_full <= float(ftol)) & (fsql_full <= float(ftol))
    conv_mask = strict_mask.copy()
    if fsq_total_target is not None:
        total_mask = (fsqr_full + fsqz_full + fsql_full) <= float(fsq_total_target)
        conv_mask = conv_mask | total_mask
    if bool(np.any(conv_mask)):
        conv_idx = int(np.argmax(conv_mask)) + 1
        conv_idx_print = conv_idx
    else:
        conv_idx = int(fsqr_full.size)
        conv_idx_print = int(max_iter)
    if conv_idx < int(fsqr_full.size):
        accepted_mask = accepted_mask & (np.arange(int(fsqr_full.size)) < int(conv_idx))
    accepted_idx = np.flatnonzero(accepted_mask)
    if accepted_idx.size > int(max_iter):
        accepted_idx = accepted_idx[: int(max_iter)]
    fsqr_hist_np = fsqr_full[accepted_idx]
    fsqz_hist_np = fsqz_full[accepted_idx]
    fsql_hist_np = fsql_full[accepted_idx]
    w_hist = fsqr_hist_np + fsqz_hist_np + fsql_hist_np
    if fsqr_hist_np.size > 0:
        final_fsqr = float(fsqr_hist_np[-1])
        final_fsqz = float(fsqz_hist_np[-1])
        final_fsql = float(fsql_hist_np[-1])
    elif fsqr_full.size > 0:
        final_idx = min(max(int(conv_idx) - 1, 0), int(fsqr_full.size) - 1)
        final_fsqr = float(fsqr_full[final_idx])
        final_fsqz = float(fsqz_full[final_idx])
        final_fsql = float(fsql_full[final_idx])
    else:
        final_fsqr = float("inf")
        final_fsqz = float("inf")
        final_fsql = float("inf")
    converged_strict = bool((final_fsqr <= float(ftol)) and (final_fsqz <= float(ftol)) and (final_fsql <= float(ftol)))
    converged_total = (
        bool((final_fsqr + final_fsqz + final_fsql) <= float(fsq_total_target))
        if fsq_total_target is not None
        else False
    )

    if scan_minimal or scan_light:
        fsqr1_hist_np = np.zeros((0,), dtype=float)
        fsqz1_hist_np = np.zeros((0,), dtype=float)
        fsql1_hist_np = np.zeros((0,), dtype=float)
        zero_m1_hist_np = np.zeros((0,), dtype=int)
        include_edge_hist_np = np.zeros((0,), dtype=int)
    else:
        fsqr1_hist_np = np.asarray(histories.fsqr1)[accepted_idx]
        fsqz1_hist_np = np.asarray(histories.fsqz1)[accepted_idx]
        fsql1_hist_np = np.asarray(histories.fsql1)[accepted_idx]
        zero_m1_hist_np = np.asarray(histories.zero_m1)[accepted_idx]
        include_edge_hist_np = np.asarray(histories.include_edge)[accepted_idx]
    dt_hist_np = np.asarray(histories.dt)[accepted_idx] if histories.dt is not None else np.zeros((0,), dtype=float)
    r00_hist_np = np.asarray(histories.r00)[accepted_idx] if histories.r00 is not None else np.zeros((0,), dtype=float)
    w_mhd_hist_np = (
        np.asarray(histories.w_mhd)[accepted_idx] if histories.w_mhd is not None else np.zeros((0,), dtype=float)
    )
    res0_full = np.asarray(histories.res0) if histories.res0 is not None else np.zeros((0,), dtype=float)
    res1_full = np.asarray(histories.res1) if histories.res1 is not None else np.zeros((0,), dtype=float)
    iter1_full = np.asarray(histories.iter1) if histories.iter1 is not None else np.zeros((0,), dtype=int)
    min_tau_full = np.asarray(histories.min_tau) if histories.min_tau is not None else np.zeros((0,), dtype=float)
    max_tau_full = np.asarray(histories.max_tau) if histories.max_tau is not None else np.zeros((0,), dtype=float)
    ptau_min_full = np.asarray(histories.ptau_min) if histories.ptau_min is not None else np.zeros((0,), dtype=float)
    ptau_max_full = np.asarray(histories.ptau_max) if histories.ptau_max is not None else np.zeros((0,), dtype=float)
    tau_min_state_full = (
        np.asarray(histories.tau_min_state) if histories.tau_min_state is not None else np.zeros((0,), dtype=float)
    )
    tau_max_state_full = (
        np.asarray(histories.tau_max_state) if histories.tau_max_state is not None else np.zeros((0,), dtype=float)
    )
    badjac_ptau_full = (
        np.asarray(histories.badjac_ptau) if histories.badjac_ptau is not None else np.zeros((0,), dtype=int)
    )
    badjac_state_full = (
        np.asarray(histories.badjac_state) if histories.badjac_state is not None else np.zeros((0,), dtype=int)
    )
    bad_jac_full = (
        np.asarray(histories.bad_jac).astype(int) if histories.bad_jac is not None else np.zeros((0,), dtype=int)
    )
    probe_count_final = int(np.asarray(carry_final.probe_count))
    probe_bad_jac_final = int(np.asarray(carry_final.probe_bad_jac))
    probe_accept_final = int(np.asarray(carry_final.probe_accept))
    probe_fsq_start_final = float(np.asarray(carry_final.probe_fsq_start))
    probe_fsq_min_final = float(np.asarray(carry_final.probe_fsq_min))
    probe_fsq_max_final = float(np.asarray(carry_final.probe_fsq_max))
    probe_ratio_final = (
        probe_fsq_max_final / max(probe_fsq_start_final, 1.0e-30) if probe_count_final > 0 else float("nan")
    )
    probe_accept_frac_final = (
        float(probe_accept_final) / max(float(probe_count_final), 1.0) if probe_count_final > 0 else float("nan")
    )
    fsqr_full_diag = fsqr_full if not scan_minimal else np.zeros((0,), dtype=float)
    fsqz_full_diag = fsqz_full if not scan_minimal else np.zeros((0,), dtype=float)
    fsql_full_diag = fsql_full if not scan_minimal else np.zeros((0,), dtype=float)
    n_iter_hist = int(np.asarray(w_hist).shape[0])
    resume_iter_offset = int(np.asarray(carry_final.iter_offset)) + n_iter_hist
    if free_boundary_enabled:
        freeb_iter1_final = int(np.asarray(carry_final.iter1))
        freeb_ivac_final, freeb_ivacskip_final = free_boundary_iter_controls(
            int(resume_iter_offset), int(freeb_iter1_final), int(freeb_nvacskip)
        )
        if iter1_full.size > 0:
            iter2_full = np.arange(1, int(iter1_full.size) + 1, dtype=int) + int(iter_offset0)
            freeb_ivacskip_full = np.mod(iter2_full - iter1_full.astype(int), int(freeb_nvacskip)).astype(int)
            freeb_ivac_full = np.where(freeb_ivacskip_full == 0, 1, 2).astype(int)
        else:
            freeb_ivacskip_full = np.zeros((0,), dtype=int)
            freeb_ivac_full = np.zeros((0,), dtype=int)
    else:
        freeb_ivac_final = 0
        freeb_ivacskip_final = 0
        freeb_ivacskip_full = np.zeros((0,), dtype=int)
        freeb_ivac_full = np.zeros((0,), dtype=int)

    resume_state_scan_payload = None
    if resume_state_mode != "none":
        resume_state_scan_base = {
            "time_step": float(np.asarray(carry_final.time_step)),
            "inv_tau": np.asarray(carry_final.inv_tau),
            "fsq_prev": float(np.asarray(carry_final.fsq_prev)),
            "fsq0_prev": float(np.asarray(carry_final.fsq0_prev)),
            "flip_sign": float(np.asarray(carry_final.flip_sign)),
            "iter1": int(np.asarray(carry_final.iter1)),
            "iter_offset": int(resume_iter_offset),
            "res0": float(np.asarray(carry_final.res0)),
            "res1": float(np.asarray(carry_final.res1)),
            "prev_rz_fsq": float(np.asarray(carry_final.fsqr_prev_phys + carry_final.fsqz_prev_phys)),
            "vmec2000_cache_valid": bool(np.asarray(carry_final.cache_valid)),
            "ijacob": int(np.asarray(carry_final.ijacob)),
            "bad_resets": int(np.asarray(carry_final.bad_resets)),
            "bad_growth_streak": int(np.asarray(carry_final.bad_growth)),
            "fsqz_prev": float(np.asarray(carry_final.fsqz_prev)),
            "r00_prev": float(np.asarray(carry_final.r00_prev)),
            "z00_prev": float(np.asarray(carry_final.z00_prev)),
            "w_mhd_prev": float(np.asarray(carry_final.w_mhd_prev)),
            "force_bcovar_update": bool(np.asarray(carry_final.force_bcovar_update)),
            "freeb_ivac": int(freeb_ivac_final),
            "freeb_ivacskip": int(freeb_ivacskip_final),
            "freeb_nvacskip": int(freeb_nvacskip),
            "freeb_nvskip0": int(freeb_nvskip0),
        }
        resume_state_scan_heavy = None
        if resume_state_mode == "full":
            resume_state_scan_heavy = {
                "state_checkpoint": carry_final.state_checkpoint,
                "vRcc": np.asarray(carry_final.vRcc),
                "vRss": np.asarray(carry_final.vRss),
                "vZsc": np.asarray(carry_final.vZsc),
                "vZcs": np.asarray(carry_final.vZcs),
                "vLsc": np.asarray(carry_final.vLsc),
                "vLcs": np.asarray(carry_final.vLcs),
                "vRsc": np.asarray(carry_final.vRsc),
                "vRcs": np.asarray(carry_final.vRcs),
                "vZcc": np.asarray(carry_final.vZcc),
                "vZss": np.asarray(carry_final.vZss),
                "vLcc": np.asarray(carry_final.vLcc),
                "vLss": np.asarray(carry_final.vLss),
                "cache_precond_diag": carry_final.cache_precond_diag,
                "cache_tcon": carry_final.cache_tcon,
                "cache_norms": carry_final.cache_norms,
                "cache_rz_scale": carry_final.cache_rz_scale,
                "cache_l_scale": carry_final.cache_l_scale,
                "cache_rz_norm": np.asarray(carry_final.cache_rz_norm),
                "cache_f_norm1": np.asarray(carry_final.cache_f_norm1),
                "cache_prec_rz_mats": carry_final.cache_prec_rz_mats,
                "cache_prec_lam_prec": np.asarray(carry_final.cache_prec_lam_prec),
            }
        resume_state_scan_payload = pack_resume_state(
            base=resume_state_scan_base,
            heavy=resume_state_scan_heavy,
        )
    free_boundary_diag = {
        "enabled": bool(free_boundary_enabled),
        "nvacskip": int(freeb_nvacskip),
        "nvskip0": int(freeb_nvskip0),
        "ivac": int(freeb_ivac_final),
        "ivacskip": int(freeb_ivacskip_final),
        "vacuum_stub": True,
    }
    return Vmec2000ScanPostprocessResult(
        fsqr_full=fsqr_full,
        fsqz_full=fsqz_full,
        fsql_full=fsql_full,
        accepted_mask=np.asarray(accepted_mask),
        accepted_idx=accepted_idx,
        conv_mask=conv_mask,
        conv_idx=int(conv_idx),
        conv_idx_print=int(conv_idx_print),
        fsqr_history=fsqr_hist_np,
        fsqz_history=fsqz_hist_np,
        fsql_history=fsql_hist_np,
        w_history=w_hist,
        final_fsqr=float(final_fsqr),
        final_fsqz=float(final_fsqz),
        final_fsql=float(final_fsql),
        converged_strict=bool(converged_strict),
        converged_total=bool(converged_total),
        fsqr1_history=fsqr1_hist_np,
        fsqz1_history=fsqz1_hist_np,
        fsql1_history=fsql1_hist_np,
        zero_m1_history=zero_m1_hist_np,
        include_edge_history=include_edge_hist_np,
        time_step_history=dt_hist_np,
        r00_history=r00_hist_np,
        w_vmec_history=w_mhd_hist_np,
        res0_full=res0_full,
        res1_full=res1_full,
        iter1_full=iter1_full,
        min_tau_full=min_tau_full,
        max_tau_full=max_tau_full,
        ptau_min_full=ptau_min_full,
        ptau_max_full=ptau_max_full,
        tau_min_state_full=tau_min_state_full,
        tau_max_state_full=tau_max_state_full,
        badjac_ptau_full=badjac_ptau_full,
        badjac_state_full=badjac_state_full,
        bad_jacobian_full=bad_jac_full,
        fsqr_full_diag=fsqr_full_diag,
        fsqz_full_diag=fsqz_full_diag,
        fsql_full_diag=fsql_full_diag,
        n_iter_hist=n_iter_hist,
        resume_iter_offset=int(resume_iter_offset),
        resume_state=resume_state_scan_payload,
        freeb_ivac_full=freeb_ivac_full,
        freeb_ivacskip_full=freeb_ivacskip_full,
        free_boundary_diag=free_boundary_diag,
        probe_count=probe_count_final,
        probe_bad_jac=probe_bad_jac_final,
        probe_accept=probe_accept_final,
        probe_fsq_start=probe_fsq_start_final,
        probe_fsq_min=probe_fsq_min_final,
        probe_fsq_max=probe_fsq_max_final,
        probe_ratio=probe_ratio_final,
        probe_accept_frac=probe_accept_frac_final,
    )


def finalize_vmec2000_scan_run(
    *,
    carry_final: Any,
    history: Any,
    state0: Any,
    result_type: type[SolveVmecResidualResult],
    state_only_scan: bool,
    scan_minimal: bool,
    scan_light: bool,
    scan_use_precomputed: bool,
    scan_use_lax_tridi: bool,
    vmec2000_control: bool,
    ftol: float,
    fsq_total_target: float | None,
    max_iter: int,
    resume_state_mode: str,
    pack_resume_state: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any] | None],
    free_boundary_enabled: bool,
    freeb_nvacskip: int,
    freeb_nvskip0: int,
    iter_offset0: int,
    free_boundary_iter_controls: Callable[[int, int, int], tuple[int, int]],
    scan_timing_enabled: bool,
    scan_timing_stats: dict[str, Any],
    scan_postprocess_start: float | None,
    scan_total_start: float | None,
    perf_counter: Callable[[], float],
    build_timing_report: Callable[..., dict[str, float | int]],
    tree_has_tracer: Callable[[Any], bool],
    build_traced_scan_resume_state: Callable[..., dict[str, Any]],
    state_only_scan_result: Callable[..., SolveVmecResidualResult],
    traced_scan_result: Callable[..., SolveVmecResidualResult],
    attach_free_boundary_diagnostics: Callable[[SolveVmecResidualResult], SolveVmecResidualResult],
    emit_post_scan_rows: Callable[..., Any],
    post_scan_print_enabled: bool,
    should_print: Callable[[int, int], bool],
    print_row: Callable[..., Any],
    dump_ptau_rows: Callable[..., Any],
    dump_ptau_enabled: bool,
    badjac_mode: str,
    dump_ptau: Callable[..., Any],
    badjac_use_state: bool,
    badjac_state_probe: bool,
    badjac_initial_state_probe_iters: int,
) -> SolveVmecResidualResult:
    """Finalize a VMEC2000 scan run into the public residual result."""

    scan_timing_report = None
    if bool(state_only_scan):
        traced = tree_has_tracer(carry_final.state)
        hist_dtype = jnp.asarray(state0.Rcos).dtype
        empty = jnp.zeros((0,), dtype=hist_dtype) if traced else np.asarray([], dtype=float)
        if bool(scan_timing_enabled):
            if scan_postprocess_start is not None:
                scan_timing_stats["scan_postprocess_s"] = float(
                    scan_timing_stats.get("scan_postprocess_s", 0.0)
                ) + (perf_counter() - float(scan_postprocess_start))
            scan_total_s = (
                perf_counter() - float(scan_total_start) if scan_total_start is not None else sum(scan_timing_stats.values())
            )
            scan_timing_report = build_timing_report(
                iterations=int(max_iter),
                stats=scan_timing_stats,
                scan_total_s=float(scan_total_s),
            )
        diagnostics = vmec2000_state_only_scan_diagnostics(
            carry_final=carry_final,
            traced=bool(traced),
            ftol=float(ftol),
            scan_minimal=bool(scan_minimal),
            scan_light=bool(scan_light),
            scan_use_precomputed=bool(scan_use_precomputed),
            scan_use_lax_tridi=bool(scan_use_lax_tridi),
            timing_report=scan_timing_report,
        )
        return state_only_scan_result(
            result_type=result_type,
            carry_final=carry_final,
            empty_history=empty,
            max_iter=int(max_iter),
            diagnostics=diagnostics,
            attach_free_boundary_diagnostics=attach_free_boundary_diagnostics,
        )

    scan_histories = unpack_vmec2000_scan_histories(
        history,
        scan_minimal=bool(scan_minimal),
        scan_light=bool(scan_light),
    )
    if tree_has_tracer(history) or tree_has_tracer(carry_final.state):
        hist_dtype = jnp.asarray(state0.Rcos).dtype
        empty = jnp.zeros((0,), dtype=hist_dtype)
        traced_resume_state = build_traced_scan_resume_state(carry_final, max_iter=int(max_iter))
        return traced_scan_result(
            result_type=result_type,
            carry_final=carry_final,
            empty_history=empty,
            max_iter=int(max_iter),
            resume_state=traced_resume_state,
            scan_use_precomputed=bool(scan_use_precomputed),
            scan_use_lax_tridi=bool(scan_use_lax_tridi),
            attach_free_boundary_diagnostics=attach_free_boundary_diagnostics,
            traced_diagnostics_func=vmec2000_traced_scan_diagnostics,
        )

    scan_output = postprocess_vmec2000_scan_result(
        scan_histories,
        carry_final,
        vmec2000_control=bool(vmec2000_control),
        ftol=float(ftol),
        fsq_total_target=fsq_total_target,
        max_iter=int(max_iter),
        scan_minimal=bool(scan_minimal),
        scan_light=bool(scan_light),
        resume_state_mode=str(resume_state_mode),
        pack_resume_state=pack_resume_state,
        free_boundary_enabled=bool(free_boundary_enabled),
        freeb_nvacskip=int(freeb_nvacskip),
        freeb_nvskip0=int(freeb_nvskip0),
        iter_offset0=int(iter_offset0),
        free_boundary_iter_controls=free_boundary_iter_controls,
    )
    emit_post_scan_rows(
        enabled=bool(post_scan_print_enabled),
        scan_histories=scan_histories,
        fsqr_full=scan_output.fsqr_full,
        fsqz_full=scan_output.fsqz_full,
        fsql_full=scan_output.fsql_full,
        conv_idx_print=int(scan_output.conv_idx_print),
        max_iter=int(max_iter),
        should_print=should_print,
        print_row=print_row,
    )
    dump_ptau_rows(
        enabled=bool(dump_ptau_enabled),
        scan_histories=scan_histories,
        conv_idx_print=int(scan_output.conv_idx_print),
        max_iter=int(max_iter),
        iter_offset0=int(iter_offset0),
        badjac_mode=badjac_mode,
        dump_ptau=dump_ptau,
    )
    if bool(scan_timing_enabled):
        if scan_postprocess_start is not None:
            scan_timing_stats["scan_postprocess_s"] = float(scan_timing_stats.get("scan_postprocess_s", 0.0)) + (
                perf_counter() - float(scan_postprocess_start)
            )
        scan_total_s = (
            perf_counter() - float(scan_total_start) if scan_total_start is not None else sum(scan_timing_stats.values())
        )
        scan_timing_report = build_timing_report(
            iterations=int(scan_output.n_iter_hist),
            stats=scan_timing_stats,
            scan_total_s=float(scan_total_s),
        )
    res_scan = vmec2000_scan_residual_result(
        state=carry_final.state,
        scan_output=scan_output,
        ftol=float(ftol),
        scan_light=bool(scan_light),
        scan_minimal=bool(scan_minimal),
        scan_use_precomputed=bool(scan_use_precomputed),
        scan_use_lax_tridi=bool(scan_use_lax_tridi),
        resume_state_mode=str(resume_state_mode),
        fsq_total_target=fsq_total_target,
        badjac_use_state=bool(badjac_use_state),
        badjac_mode=badjac_mode,
        badjac_state_probe=bool(badjac_state_probe),
        badjac_initial_state_probe_iters=int(badjac_initial_state_probe_iters),
        ijacob=int(np.asarray(carry_final.ijacob)),
        abort_scan=bool(np.asarray(carry_final.abort_scan)),
        timing_report=scan_timing_report,
    )
    return attach_free_boundary_diagnostics(res_scan)

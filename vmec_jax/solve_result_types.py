"""Result and scan-carry containers used by VMEC solve routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple

import numpy as np

from .state import VMECState


@dataclass(frozen=True)
class SolveLambdaResult:
    state: VMECState
    n_iter: int
    wb_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class SolveFixedBoundaryResult:
    state: VMECState
    n_iter: int
    w_history: np.ndarray
    wb_history: np.ndarray
    wp_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class SolveVmecResidualResult:
    state: VMECState
    n_iter: int
    w_history: np.ndarray
    fsqr2_history: np.ndarray
    fsqz2_history: np.ndarray
    fsql2_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


class ScanCarry(NamedTuple):
    state: VMECState
    time_step: Any
    inv_tau: Any
    fsq_prev: Any
    fsq0_prev: Any
    accepted_count: Any
    probe_count: Any
    probe_bad_jac: Any
    probe_accept: Any
    probe_fsq_min: Any
    probe_fsq_max: Any
    probe_fsq_start: Any
    fallback_active: Any
    abort_scan: Any
    skip_timecontrol: Any
    vRcc: Any
    vRss: Any
    vZsc: Any
    vZcs: Any
    vLsc: Any
    vLcs: Any
    vRsc: Any
    vRcs: Any
    vZcc: Any
    vZss: Any
    vLcc: Any
    vLss: Any
    flip_sign: Any
    iter_offset: Any
    iter1: Any
    res0: Any
    res1: Any
    state_checkpoint: VMECState
    cache_valid: Any
    cache_precond_diag: Any
    cache_tcon: Any
    cache_norms: Any
    cache_rz_scale: Any
    cache_l_scale: Any
    cache_rz_norm: Any
    cache_f_norm1: Any
    cache_prec_rz_mats: Any
    cache_prec_lam_prec: Any
    force_bcovar_update: Any
    ijacob: Any
    bad_resets: Any
    bad_growth: Any
    fsqz_prev: Any
    r00_prev: Any
    z00_prev: Any
    w_mhd_prev: Any
    converged: Any
    fsqr_prev_phys: Any
    fsqz_prev_phys: Any
    fsql_prev_phys: Any
    fsqr1_prev: Any
    fsqz1_prev: Any
    fsql1_prev: Any
    fsqr_checkpoint: Any
    fsqz_checkpoint: Any
    fsql_checkpoint: Any
    fsqr1_checkpoint: Any
    fsqz1_checkpoint: Any
    fsql1_checkpoint: Any
    edge_Rcos: Any
    edge_Rsin: Any
    edge_Zcos: Any
    edge_Zsin: Any


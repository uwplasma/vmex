"""Result and scan-carry containers used by VMEC solve routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple

import numpy as np

from ._compat import tree_util
from .state import VMECState


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WoutLikeVmecForces:
    """Minimal ``wout``-like container for VMEC force/residual kernels."""

    nfp: int
    mpol: int
    ntor: int
    lasym: bool
    signgs: int

    phipf: Any  # (ns,)
    phips: Any  # (ns,)
    chipf: Any  # (ns,) (VMEC `wout` half-mesh averaged convention)
    pres: Any  # (ns,) (half mesh, VMEC internal units mu0*Pa)
    mass: Any | None = None  # (ns,) mass profile on half mesh (VMEC internal units)
    gamma: float | None = None
    ncurr: int = 0
    lcurrent: bool = True
    icurv: Any | None = None  # (ns,) integrated toroidal current profile
    flux_is_internal: bool = True
    phipf_internal: Any | None = None
    chipf_internal: Any | None = None
    chips_eff: Any | None = None

    def tree_flatten(self):
        children = (
            self.phipf,
            self.phips,
            self.chipf,
            self.pres,
            self.mass,
            self.icurv,
            self.phipf_internal,
            self.chipf_internal,
            self.chips_eff,
        )
        aux = (
            int(self.nfp),
            int(self.mpol),
            int(self.ntor),
            bool(self.lasym),
            int(self.signgs),
            None if self.gamma is None else float(self.gamma),
            int(self.ncurr),
            bool(self.lcurrent),
            bool(self.flux_is_internal),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            nfp,
            mpol,
            ntor,
            lasym,
            signgs,
            gamma,
            ncurr,
            lcurrent,
            flux_is_internal,
        ) = aux_data
        return cls(
            nfp=int(nfp),
            mpol=int(mpol),
            ntor=int(ntor),
            lasym=bool(lasym),
            signgs=int(signgs),
            gamma=gamma,
            ncurr=int(ncurr),
            lcurrent=bool(lcurrent),
            flux_is_internal=bool(flux_is_internal),
            phipf=children[0],
            phips=children[1],
            chipf=children[2],
            pres=children[3],
            mass=children[4],
            icurv=children[5],
            phipf_internal=children[6],
            chipf_internal=children[7],
            chips_eff=children[8],
        )


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

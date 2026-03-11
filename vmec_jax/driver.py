"""High-level helpers for VMEC driver scripts.

These functions provide a thin, convenient layer over the core modules so
simple scripts can be written with minimal boilerplate, while still allowing
power users to drop down to lower-level APIs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import os
import time
from typing import Optional

import numpy as np
from .boundary import boundary_from_indata
from .config import VMECConfig, load_config
from .energy import FluxProfiles, _iotaf_from_iotas, flux_profiles_from_indata
from .free_boundary import (
    MGridMetadata,
    PreparedMGrid,
    prepare_mgrid_for_config,
    validate_free_boundary_config,
)
from .geom import eval_geom
from .init_guess import initial_guess_from_boundary
from .multigrid import interp_vmec_state
from .profiles import eval_profiles
from .solve import (
    SolveVmecResidualResult,
    solve_fixed_boundary_gd,
    solve_fixed_boundary_lbfgs,
    solve_fixed_boundary_residual_iter,
)
from .static import VMECStatic, build_static
from .wout import WoutData, read_wout, state_from_wout


@dataclass(frozen=True)
class ExampleData:
    input_path: Path
    wout_path: Optional[Path]
    cfg: VMECConfig
    indata: any
    static: VMECStatic
    wout: Optional[WoutData]
    state: Optional[any]


@dataclass(frozen=True)
class FixedBoundaryRun:
    """Container returned by ``run_fixed_boundary``."""

    cfg: VMECConfig
    indata: any
    static: VMECStatic
    state: any
    result: any | None
    flux: any
    profiles: dict
    signgs: int


def _default_backend_name() -> str:
    try:
        import jax

        return str(jax.default_backend()).strip().lower() or "cpu"
    except Exception:
        return "cpu"


def _dynamic_scan_probe_settings(niter_i: int) -> tuple[int, bool, str]:
    backend = _default_backend_name()
    timed_env = os.getenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "").strip().lower()
    if timed_env in ("1", "true", "yes", "on"):
        timed_probe = True
    elif timed_env in ("0", "false", "no", "off"):
        timed_probe = False
    else:
        timed_probe = backend == "cpu"

    default_probe_iters = 10 if timed_probe else 3
    try:
        pre_iters_env = os.getenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "").strip()
        pre_iters = max(1, int(pre_iters_env)) if pre_iters_env else default_probe_iters
    except Exception:
        pre_iters = default_probe_iters

    if pre_iters >= int(niter_i):
        pre_iters = max(1, int(niter_i) - 1)
    return pre_iters, timed_probe, backend


_VALID_SOLVER_MODES = frozenset(("default", "parity", "accelerated"))
_FSQ_COMPONENT_NAMES = ("fsqr", "fsqz", "fsql")


def _normalize_solver_mode(*, solver_mode: str | None, performance_mode: bool) -> str:
    if solver_mode is None:
        return "default" if bool(performance_mode) else "parity"
    mode = str(solver_mode).strip().lower()
    aliases = {
        "fast": "default",
        "safe": "parity",
        "reference": "parity",
        "perf": "accelerated",
    }
    mode = aliases.get(mode, mode)
    if mode not in _VALID_SOLVER_MODES:
        valid = ", ".join(sorted(_VALID_SOLVER_MODES))
        raise ValueError(f"Unknown solver_mode {solver_mode!r}. Expected one of: {valid}.")
    return mode


def _accelerated_fsq_total_target_from_ftol(ftol: float) -> float:
    """Collapse per-component FTOL into an equivalent total-residual target.

    The accelerated path still treats the input FTOL as the user truth. The
    total objective is `fsqr + fsqz + fsql`, so the corresponding scalar target
    is the same per-channel budget summed across the active residual channels.
    """
    return max(0.0, float(ftol)) * float(len(_FSQ_COMPONENT_NAMES))


def _allocate_integer_budget(*, total: int, weights: list[int]) -> list[int]:
    total = max(0, int(total))
    if total <= 0:
        return [0] * len(weights)
    if not weights:
        return []
    weights_eff = [max(0, int(w)) for w in weights]
    if sum(weights_eff) <= 0:
        out = [0] * len(weights_eff)
        out[-1] = total
        return out
    raw = [float(total) * float(w) / float(sum(weights_eff)) for w in weights_eff]
    out = [int(v) for v in raw]
    remaining = int(total - sum(out))
    if remaining > 0:
        order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - float(out[idx])), reverse=True)
        for idx in order[:remaining]:
            out[idx] += 1
    elif remaining < 0:
        order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - float(out[idx])))
        for idx in order[: abs(remaining)]:
            if out[idx] > 0:
                out[idx] -= 1
    return out


def _accelerated_cli_budgeted_total_iters(*, total_budget: int, ns_stages: list[int]) -> int:
    """Reduce oversized parity-era budgets for CLI accelerated warm starts.

    When an input provides multiple NS stages but no explicit NITER_ARRAY, the
    VMEC-style NITER value is typically sized for a conservative final-grid
    iteration. For the CLI-only accelerated path, use a radial-work-equivalent
    total budget scaled by the square root of the coarsest-to-finest NS ratio.
    That keeps aggressive multigrid speedups while avoiding the severe
    under-budgeting seen on the hardest staged fixed-boundary cases.
    """
    total_budget = max(1, int(total_budget))
    if not ns_stages:
        return total_budget
    ns0 = max(1, int(ns_stages[0]))
    nsf = max(ns0, int(ns_stages[-1]))
    return max(1, int(round(float(total_budget) * float(np.sqrt(float(ns0) / float(nsf))))))


def _accelerated_cli_budgeted_stage_iters(*, total_budget: int, ns_stages: list[int]) -> list[int]:
    """Distribute a reduced CLI accelerated budget across multigrid stages.

    Use the number of newly introduced radial degrees of freedom per stage as
    the structural signal. Squaring that increment biases the budget toward the
    fine stages where most of the unresolved detail enters.
    """
    if not ns_stages:
        return [max(1, int(total_budget))]
    ns_int = [max(1, int(v)) for v in ns_stages]
    increments = [ns_int[0]]
    for prev_ns, curr_ns in zip(ns_int[:-1], ns_int[1:]):
        increments.append(max(1, int(curr_ns - prev_ns)))
    weights = [int(v) * int(v) for v in increments]
    out = _allocate_integer_budget(total=max(1, int(total_budget)), weights=weights)
    if out:
        out[-1] = max(1, int(out[-1]))
    return out


def _final_flux_profiles_from_state(
    *,
    indata,
    static_in: VMECStatic,
    state,
    signgs: int,
    flux_local,
    prof_local: dict,
    pressure_local,
):
    """Return post-solve flux/profile payloads consistent with the solved state.

    `flux_profiles_from_indata()` is input-only. For current-driven runs
    (`NCURR=1`), VMEC updates the rotational transform from the solved force
    balance. Mirror that here so the driver returns the same effective flux/iota
    channels that `wout` reconstruction uses.
    """
    ncurr = int(indata.get_int("NCURR", 0))
    if ncurr != 1:
        return flux_local, prof_local
    if os.getenv("VMEC_JAX_DISABLE_WOUT_NCURR_RECOMPUTE", "0") not in ("", "0"):
        return flux_local, prof_local

    from types import SimpleNamespace

    from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout
    from .vmec_residue import vmec_pwint_from_trig
    from .vmec_tomnsp import vmec_trig_tables
    from .wout import _chipf_from_chips, _icurv_full_mesh_from_indata

    s = np.asarray(static_in.s, dtype=float)
    try:
        state_ns = int(np.asarray(state.Rcos).shape[0])
    except Exception:
        state_ns = int(s.shape[0])
    if int(state_ns) != int(s.shape[0]):
        return flux_local, prof_local
    phipf = np.asarray(flux_local.phipf, dtype=float)
    phips = np.asarray(flux_local.phips, dtype=float)
    if phips.size:
        phips = phips.copy()
        phips[0] = 0.0

    pressure_out = np.asarray(pressure_local, dtype=float)
    if pressure_out.size:
        pressure_out = pressure_out.copy()
        pressure_out[0] = 0.0

    gamma = float(indata.get_float("GAMMA", 0.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    boundary = boundary_from_indata(indata, static_in.modes)
    idx00 = np.where((np.asarray(static_in.modes.m) == 0) & (np.asarray(static_in.modes.n) == 0))[0]
    r00 = float(boundary.R_cos[int(idx00[0])]) if idx00.size else float(np.asarray(boundary.R_cos)[0])
    vnorm = phips
    if lrfp:
        chipf_in = np.asarray(flux_local.chipf, dtype=float)
        if chipf_in.size:
            chips_in = np.concatenate([chipf_in[:1], 0.5 * (chipf_in[1:] + chipf_in[:-1])], axis=0)
            vnorm = chips_in
    mass = pressure_out * (np.abs(vnorm) * r00) ** gamma
    if mass.size:
        mass = mass.copy()
        mass[0] = 0.0

    trig = getattr(static_in, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(np.asarray(static_in.grid.theta).shape[0]),
            nzeta=int(np.asarray(static_in.grid.zeta).shape[0]),
            nfp=int(static_in.cfg.nfp),
            mmax=max(0, int(static_in.cfg.mpol) - 1),
            nmax=max(0, int(static_in.cfg.ntor)),
            lasym=bool(static_in.cfg.lasym),
        )
    icurv = np.asarray(_icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=int(signgs)), dtype=float)
    wout_like_pre = SimpleNamespace(
        phipf=phipf,
        phips=phips,
        chipf=np.zeros_like(phipf),
        signgs=int(signgs),
        nfp=int(static_in.cfg.nfp),
        mpol=int(static_in.cfg.mpol),
        ntor=int(static_in.cfg.ntor),
        lasym=bool(static_in.cfg.lasym),
        ncurr=0,
        lcurrent=False,
        icurv=np.zeros_like(phipf),
        flux_is_internal=True,
        mass=mass,
        gamma=gamma,
    )
    bc_pre = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static_in,
        wout=wout_like_pre,
        pres=pressure_out,
        use_vmec_synthesis=True,
        trig=trig,
    )

    sqrtg = np.asarray(bc_pre.jac.sqrtg, dtype=float)
    overg = np.zeros_like(sqrtg)
    np.divide(1.0, sqrtg, out=overg, where=(sqrtg != 0.0))
    pwint = np.asarray(
        vmec_pwint_from_trig(trig, ns=int(overg.shape[0]), nzeta=int(overg.shape[2])),
        dtype=overg.dtype,
    )
    guu = np.asarray(bc_pre.guu, dtype=float)
    guv = np.asarray(bc_pre.guv, dtype=float)
    bsupu = np.asarray(bc_pre.bsupu, dtype=float)
    bsupv = np.asarray(bc_pre.bsupv, dtype=float)
    top = icurv - np.sum(pwint * ((guu * bsupu) + (guv * bsupv)), axis=(1, 2))
    bot = np.sum(pwint * (overg * guu), axis=(1, 2))
    chips = np.zeros_like(top)
    np.divide(top, bot, out=chips, where=(bot != 0.0))
    if chips.size:
        chips[0] = 0.0

    iotas = np.zeros_like(chips)
    np.divide(chips, phips, out=iotas, where=(phips != 0.0))
    if iotas.size:
        iotas[0] = 0.0
    iotaf = np.asarray(_iotaf_from_iotas(iotas, lrfp=lrfp), dtype=float)
    chipf = np.asarray(_chipf_from_chips(chips), dtype=float)

    prof_out = dict(prof_local)
    prof_out["iota"] = iotas
    prof_out["iotaf"] = iotaf
    flux_out = FluxProfiles(
        phipf=phipf,
        chipf=chipf,
        phips=phips,
        signgs=int(signgs),
        lamscale=np.asarray(flux_local.lamscale),
    )
    return flux_out, prof_out


def residual_scalars_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    wout=None,
    use_vmec_synthesis: bool = True,
):
    """Compute VMEC-style invariant residual scalars (fsqr/fsqz/fsql) from a state.

    This uses the residual pipeline:
      bcovar -> forces -> tomnsps -> getfsq

    and is intentionally input-only: flux profiles and pressure are derived from
    `indata` rather than a reference `wout`.
    """
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import vmec_force_norms_from_bcovar_dynamic, vmec_fsq_from_tomnsps_dynamic
    from .vmec_tomnsp import TomnspsRZL, vmec_trig_tables

    class _WoutLike:
        __slots__ = ("nfp", "mpol", "ntor", "lasym", "signgs")

        def __init__(self, *, nfp: int, mpol: int, ntor: int, lasym: bool, signgs: int):
            self.nfp = int(nfp)
            self.mpol = int(mpol)
            self.ntor = int(ntor)
            self.lasym = bool(lasym)
            self.signgs = int(signgs)

    wout_like = wout
    if wout_like is None:
        wout_like = _WoutLike(
            nfp=int(static.cfg.nfp),
            mpol=int(static.cfg.mpol),
            ntor=int(static.cfg.ntor),
            lasym=bool(static.cfg.lasym),
            signgs=int(signgs),
        )

    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(wout_like.nfp),
        mmax=int(wout_like.mpol) - 1,
        nmax=int(wout_like.ntor),
        lasym=bool(wout_like.lasym),
    )

    k = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        indata=indata,
        use_wout_bsup=False,
        use_vmec_synthesis=bool(use_vmec_synthesis),
        trig=trig,
    )
    rzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(static.cfg.ntheta),
        cfg_nzeta=int(static.cfg.nzeta),
        wout=wout_like,
        trig=trig,
    )
    frzl = TomnspsRZL(
        frcc=rzl.frcc,
        frss=rzl.frss,
        fzsc=rzl.fzsc,
        fzcs=rzl.fzcs,
        flsc=rzl.flsc,
        flcs=rzl.flcs,
        frsc=rzl.frsc,
        frcs=rzl.frcs,
        fzcc=rzl.fzcc,
        fzss=rzl.fzss,
        flcc=rzl.flcc,
        flss=rzl.flss,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=static.s, signgs=int(signgs))
    scal = vmec_fsq_from_tomnsps_dynamic(
        frzl=frzl, norms=norms, lconm1=bool(getattr(static.cfg, "lconm1", True))
    )
    return float(scal.fsqr), float(scal.fsqz), float(scal.fsql)


def solve_fixed_boundary_from_boundary(
    *,
    boundary,
    static: VMECStatic,
    indata,
    flux,
    pressure,
    signgs: int,
    max_iter: int = 2,
    step_size: float = 5e-3,
    jacobian_penalty: float = 1e3,
    jit_grad: bool = False,
    differentiable: bool = True,
    stop_grad_in_update: bool = True,
    verbose: bool = False,
    vmec_project: bool = False,
):
    """Solve VMEC fixed-boundary starting from a boundary coefficient set.

    This helper wraps `initial_guess_from_boundary` and `solve_fixed_boundary_gd`
    so optimization scripts can call a single function.
    """
    st_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=vmec_project)
    res = solve_fixed_boundary_gd(
        st_guess,
        static,
        phipf=flux.phipf,
        chipf=flux.chipf,
        signgs=signgs,
        lamscale=flux.lamscale,
        pressure=pressure,
        gamma=float(indata.get_float("GAMMA", 0.0)),
        max_iter=int(max_iter),
        step_size=float(step_size),
        jacobian_penalty=float(jacobian_penalty),
        jit_grad=bool(jit_grad),
        differentiable=bool(differentiable),
        stop_grad_in_update=bool(stop_grad_in_update),
        verbose=bool(verbose),
    )
    return res.state


def wout_from_fixed_boundary_run(
    run: FixedBoundaryRun,
    *,
    include_fsq: bool = True,
    path: str | Path | None = None,
    fast_bcovar: bool | None = None,
) -> WoutData:
    """Build a minimal VMEC-style ``WoutData`` from a fixed-boundary run.

    This is the in-memory counterpart to :func:`write_wout_from_fixed_boundary_run`.
    Set ``fast_bcovar=True`` to enable the fast bcovar path for this call.
    """
    from .wout import wout_minimal_from_fixed_boundary

    path = Path(path) if path is not None else Path("wout_vmec_jax.nc")

    prev_fast_bcovar = None
    if fast_bcovar is not None:
        prev_fast_bcovar = os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR")
        os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] = "1" if fast_bcovar else "0"

    try:
        fsqt = None
        converged = None
        if include_fsq:
            fsqr = fsqz = fsql = None
            res = getattr(run, "result", None)
            if res is not None:
                converged = getattr(res, "diagnostics", {}).get("converged", None)
                fsqr_hist = getattr(res, "fsqr2_history", None)
                fsqz_hist = getattr(res, "fsqz2_history", None)
                fsql_hist = getattr(res, "fsql2_history", None)
                if fsqr_hist is not None and fsqz_hist is not None:
                    fsqr_hist = np.asarray(fsqr_hist, dtype=float)
                    fsqz_hist = np.asarray(fsqz_hist, dtype=float)
                    fsqt_hist = fsqr_hist + fsqz_hist
                    nstore = 100
                    niter = int(fsqt_hist.size)
                    stride = int(niter // nstore) + 1 if niter > 0 else 1
                    fsqt = np.zeros((nstore,), dtype=float)
                    count = 0
                    for iter2 in range(1, niter + 1):
                        if iter2 % stride != 0:
                            continue
                        fsqt[count] = float(fsqt_hist[iter2 - 1])
                        count += 1
                        if count >= nstore:
                            break
                if fsqr_hist is not None and fsqz_hist is not None and fsql_hist is not None:
                    try:
                        fsqr = float(np.asarray(fsqr_hist)[-1])
                        fsqz = float(np.asarray(fsqz_hist)[-1])
                        fsql = float(np.asarray(fsql_hist)[-1])
                    except Exception:
                        fsqr = fsqz = fsql = None
            if fsqr is None or fsqz is None or fsql is None:
                fsqr, fsqz, fsql = residual_scalars_from_state(
                    state=run.state, static=run.static, indata=run.indata, signgs=int(run.signgs), use_vmec_synthesis=True
                )
        else:
            fsqr = fsqz = fsql = 0.0

        wout = wout_minimal_from_fixed_boundary(
            path=path,
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
            fsqr=float(fsqr),
            fsqz=float(fsqz),
            fsql=float(fsql),
            fsqt=fsqt,
            converged=converged,
        )
    finally:
        if fast_bcovar is not None:
            if prev_fast_bcovar is None:
                os.environ.pop("VMEC_JAX_WOUT_FAST_BCOVAR", None)
            else:
                os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] = prev_fast_bcovar
    return wout


def write_wout_from_fixed_boundary_run(
    path: str | Path,
    run: FixedBoundaryRun,
    *,
    include_fsq: bool = True,
    fast_bcovar: bool | None = None,
):
    """Write a minimal VMEC-style `wout_*.nc` from a fixed-boundary run."""
    from .wout import write_wout

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wout = wout_from_fixed_boundary_run(run, include_fsq=include_fsq, path=path, fast_bcovar=fast_bcovar)
    write_wout(path, wout, overwrite=True)
    return wout


def example_paths(case: str, *, root: str | Path | None = None) -> tuple[Path, Optional[Path]]:
    """Return (input_path, wout_path) for a bundled example case."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    data_dir = root / "examples" / "data"
    input_path = data_dir / f"input.{case}"
    wout_path = data_dir / f"wout_{case}_reference.nc"
    if not wout_path.exists():
        wout_path = data_dir / f"wout_{case}.nc"
    if not wout_path.exists():
        wout_path = None
    return input_path, wout_path


def load_example(
    case: str,
    *,
    root: str | Path | None = None,
    with_wout: bool = True,
    grid=None,
) -> ExampleData:
    """Load a bundled example case (config + static + optional wout/state)."""
    input_path, wout_path = example_paths(case, root=root)
    cfg, indata = load_config(str(input_path))
    prepared_fb = prepare_mgrid_for_config(cfg, load_fields=False, strict=False)
    fb_meta = prepared_fb.metadata if isinstance(prepared_fb, PreparedMGrid) else None
    fb_extcur = prepared_fb.extcur if isinstance(prepared_fb, PreparedMGrid) else None
    static = build_static(cfg, grid=grid, mgrid_metadata=fb_meta, free_boundary_extcur=fb_extcur)
    if with_wout and wout_path is not None:
        wout = read_wout(wout_path)
        state = state_from_wout(wout)
    else:
        wout = None
        state = None
    return ExampleData(
        input_path=input_path,
        wout_path=wout_path,
        cfg=cfg,
        indata=indata,
        static=static,
        wout=wout,
        state=state,
    )


def load_input(path: str | Path):
    """Convenience wrapper around `load_config`."""
    return load_config(str(path))


def load_wout(path: str | Path) -> WoutData:
    """Convenience wrapper around `read_wout`."""
    return read_wout(path)


def save_npz(path: str | Path, **arrays) -> Path:
    """Save arrays into a NumPy `.npz` file and return the path."""
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


_STEP_SIZE_SENTINEL = object()
_MAX_ITER_SENTINEL = object()


def run_fixed_boundary(
    input_path: str | Path,
    *,
    solver: str = "vmec2000_iter",
    solver_mode: str | None = None,
    max_iter: int | object = _MAX_ITER_SENTINEL,
    step_size: float | object = _STEP_SIZE_SENTINEL,
    history_size: int = 10,
    # vmec_gn tuning (Gauss-Newton on VMEC residual vector)
    gn_damping: float | None = None,
    gn_cg_tol: float | None = None,
    gn_cg_maxiter: int = 80,
    use_initial_guess: bool = False,
    vmec_project: bool = True,
    use_restart_triggers: bool | None = None,
    vmecpp_restart: bool = False,
    use_direct_fallback: bool | None = None,
    multigrid: bool | None = None,
    multigrid_use_input_niter: bool = True,
    verbose: bool = True,
    jit_forces: bool | str = True,
    jit_precompile: bool | None = None,
    use_scan: bool = True,
    performance_mode: bool = True,
    scan_wout_corrector: bool | None = None,
    stage_transition_heuristic: bool | None = None,
    stage_transition_factor: float = 50.0,
    stage_transition_scale: float = 0.5,
    grid=None,
    ns_override: int | None = None,
    restart_state: any | None = None,
    restart_wout_path: str | Path | None = None,
    restart_solver_state: dict | None = None,
    cli_fixed_boundary_mode: bool = False,
):
    t_start = time.perf_counter()
    max_iter_overridden = max_iter is not _MAX_ITER_SENTINEL

    def _maybe_enable_compilation_cache() -> None:
        cache_env = os.getenv("VMEC_JAX_COMPILATION_CACHE", "1").strip().lower()
        if cache_env in ("", "0", "false", "no", "off"):
            return
        if os.getenv("VMEC_JAX_DISABLE_COMPILATION_CACHE", "") not in ("", "0"):
            return
        cache_dir = os.getenv("VMEC_JAX_COMPILATION_CACHE_DIR") or os.getenv("JAX_COMPILATION_CACHE_DIR")
        if not cache_dir:
            # Default to a user-writable cache to reduce first-run JIT latency
            # on repeated executions (especially in examples/CI runs).
            try:
                cache_dir = str(Path.home() / ".cache" / "vmec_jax" / "jax_compilation_cache")
            except Exception:
                cache_dir = ""
        if not cache_dir:
            return
        try:
            import jax
            from jax.experimental.compilation_cache import compilation_cache

            cache_path = Path(cache_dir)
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                # Fall back to /tmp when the home cache is not writable.
                try:
                    cache_path = Path("/tmp/vmec_jax/jax_compilation_cache")
                    cache_path.mkdir(parents=True, exist_ok=True)
                except Exception:
                    return
            cache_dir = str(cache_path)
            compilation_cache.set_cache_dir(cache_dir)
            try:
                jax.config.update("jax_enable_compilation_cache", True)
            except Exception:
                pass
            try:
                min_compile = os.getenv("VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS", "0")
                jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile))
                min_entry = os.getenv("VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
                jax.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry))
                max_size = os.getenv("VMEC_JAX_COMPILATION_CACHE_MAX_SIZE", "")
                if max_size:
                    jax.config.update("jax_compilation_cache_max_size", int(max_size))
                explain = os.getenv("VMEC_JAX_EXPLAIN_CACHE_MISSES", "")
                if explain.strip().lower() not in ("", "0", "false", "no"):
                    jax.config.update("jax_explain_cache_misses", True)
            except Exception:
                pass
        except Exception:
            return

    def _maybe_dump_xc_init(*, state, static, label: str) -> None:
        env = os.getenv("VMEC_JAX_DUMP_XC_INIT", "")
        if not env or env == "0":
            return
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        ns = int(static.cfg.ns)
        suffix = f"_{label}" if label else ""
        path = outdir / f"xc_init{suffix}_ns{ns}.dat"
        from .diagnostics import vmec_internal_mn_from_state, vmec_xc_from_mn_blocks

        blocks = vmec_internal_mn_from_state(state, static, apply_basis_norm=False, apply_m1_constraint=False)
        xc_kwargs = dict(
            rcc=blocks["rcc"],
            rss=blocks["rss"],
            zsc=blocks["zsc"],
            zcs=blocks["zcs"],
            lsc=blocks["lsc"],
            lcs=blocks["lcs"],
        )
        if "rsc" in blocks:
            xc_kwargs.update(
                rsc=blocks.get("rsc"),
                rcs=blocks.get("rcs"),
                zcc=blocks.get("zcc"),
                zss=blocks.get("zss"),
                lcc=blocks.get("lcc"),
                lss=blocks.get("lss"),
            )
        xc = vmec_xc_from_mn_blocks(cfg=static.cfg, **xc_kwargs)
        xcdot = np.zeros_like(xc)
        with path.open("w") as f:
            f.write("# xc/xcdot dump (init guess)\n")
            f.write(f"neqs={xc.size}\n")
            f.write("columns: i xc xcdot\n")
            for i, (x, xd) in enumerate(zip(xc, xcdot), start=1):
                f.write(f"{i:8d}{x:24.16e}{xd:24.16e}\n")
    """Run a fixed-boundary vmec_jax solve with minimal boilerplate.

    Parameters
    ----------
    solver:
        ``"vmec2000_iter"`` (VMEC-style multigrid iteration; default),
        ``"gd"`` (gradient descent), ``"lbfgs"``, ``"vmec_lbfgs"``, or
        ``"vmec_gn"`` (VMEC residual objective).
    use_initial_guess:
        If True, skip the solve and return the initialized state.
    ns_override:
        If provided, overrides the radial resolution (ns) used to build the state.
    restart_state:
        If provided, use this VMECState as the initial condition instead of
        building a new boundary-based guess. This disables multigrid staging.
    restart_wout_path:
        If provided, load the `wout_*.nc` file and use its state as the initial
        condition (same effect as `restart_state`). This disables multigrid
        staging.
    restart_solver_state:
        Optional solver-state dictionary returned by ``solve_fixed_boundary_residual_iter``
        (``diagnostics["resume_state"]``). When supplied with ``solver="vmec2000_iter"``,
        the time-step/momentum/preconditioner cache is resumed. This disables multigrid
        staging.
    cli_fixed_boundary_mode:
        Internal CLI-only flag for non-differentiable fixed-boundary policy
        overrides. Library callers should leave this as False.
    vmec_project:
        If True (default), re-project the initial guess through the VMEC
        internal grid/weights before returning or solving.
    verbose:
        If True (default), print VMEC-style iteration progress and a summary.
    jit_forces:
        If True (default), JIT the force kernels. If ``"auto"``, disable JIT
        for very small workloads to reduce first-iteration latency.
    performance_mode:
        If True, force the fast scan-based iteration path (no VMEC2000 control
        logic). This delivers order-of-magnitude speedups but does not preserve
        per-iteration VMEC2000 parity.
    solver_mode:
        Optional explicit solver policy. Supported values:
        ``"default"`` (current parity-guarded fast path),
        ``"parity"`` (strict VMEC2000-style control path), and
        ``"accelerated"`` (experimental non-parity path that prioritizes
        final residual/quality and device residency).
    """
    # Default to 64-bit for VMEC parity; users can opt out via JAX_ENABLE_X64=0.
    try:
        from ._compat import enable_x64

        enable_x64(True)
    except Exception:
        pass
    _maybe_enable_compilation_cache()
    solver_mode_eff = _normalize_solver_mode(solver_mode=solver_mode, performance_mode=bool(performance_mode))
    accelerated_mode = solver_mode_eff == "accelerated"
    performance_mode = solver_mode_eff != "parity"

    cfg, indata = load_config(str(input_path))
    restart_state_eff = restart_state
    restart_wout = None
    if restart_wout_path is not None:
        restart_wout = read_wout(Path(restart_wout_path))
        restart_state_eff = state_from_wout(restart_wout)

    if restart_state_eff is not None:
        restart_ns = int(restart_state_eff.layout.ns)
        if ns_override is not None and int(ns_override) != restart_ns:
            raise ValueError(
                f"restart_state ns={restart_ns} does not match ns_override={ns_override}"
            )
        cfg = replace(cfg, ns=int(restart_ns))
        if restart_solver_state is not None:
            # Ensure resume checkpoints align with the provided restart state.
            try:
                restart_solver_state = dict(restart_solver_state)
                restart_solver_state["state_checkpoint"] = restart_state_eff
            except Exception:
                pass
    elif ns_override is not None:
        cfg = replace(cfg, ns=int(ns_override))
    solver_lower = str(solver).lower()
    # VMEC starts from the input axis coefficients and only recomputes the
    # axis (guess_axis) after a bad-Jacobian trigger. For vmec2000_iter we
    # follow that behavior by default and allow opt-in axis inference via env.
    axis_infer_missing = solver_lower != "vmec2000_iter"
    if solver_lower == "vmec2000_iter":
        enable_axis_infer = os.getenv("VMEC_JAX_ENABLE_AXIS_INFER", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        disable_axis_infer = os.getenv("VMEC_JAX_DISABLE_AXIS_INFER", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if enable_axis_infer:
            axis_infer_missing = True
        if disable_axis_infer:
            axis_infer_missing = False
        if (not disable_axis_infer) and bool(performance_mode) and bool(cfg.lasym):
            # LASYM scan probes are more stable and much faster once the initial
            # axis guess is inferred up front. Keep the conservative VMEC-style
            # raw-axis start in parity mode.
            axis_infer_missing = True
    if grid is None and solver_lower in ("vmec_lbfgs", "vmec_gn", "vmec2000_iter"):
        from .vmec_tomnsp import vmec_angle_grid

        grid = vmec_angle_grid(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(cfg.nfp),
            lasym=bool(cfg.lasym),
        )
    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        try:
            if isinstance(value, np.ndarray):
                return list(value.tolist())
        except Exception:
            pass
        if isinstance(value, (int, float, np.integer, np.floating)):
            return [value]
        return None

    ns_list_input = _as_list(indata.get("NS_ARRAY", None))
    niter_list_input = _as_list(indata.get("NITER_ARRAY", None))
    ftol_list_input = _as_list(indata.get("FTOL_ARRAY", None))
    cli_budgeted_multigrid_requested = (
        bool(cli_fixed_boundary_mode)
        and bool(accelerated_mode)
        and (solver_lower in ("vmec2000_iter", "vmec2000_scan", "vmec2000_iter_fast"))
        and (not bool(cfg.lfreeb))
        and (restart_state_eff is None)
        and (restart_solver_state is None)
        and (multigrid is None)
        and (ns_list_input is not None)
        and (len(ns_list_input) > 1)
        and (niter_list_input is None)
    )
    cli_fixed_boundary_finish_enabled = (
        bool(cli_fixed_boundary_mode)
        and (solver_lower == "vmec2000_iter")
        and (not bool(cfg.lfreeb))
    )

    def _run_cli_accelerated_budgeted_multigrid(
        *,
        ns_stage_list: list[int],
        warm_start_budget: int,
        final_stage_budget: int,
    ):
        stage_budgets = _accelerated_cli_budgeted_stage_iters(
            total_budget=int(warm_start_budget),
            ns_stages=ns_stage_list,
        )
        if stage_budgets:
            stage_budgets[-1] = max(int(stage_budgets[-1]), int(final_stage_budget))
        stage_runs: list[FixedBoundaryRun] = []
        stage_state = None
        stage_static_prev = None
        stage_modes: list[str] = []
        for idx, (ns_i, niter_i) in enumerate(zip(ns_stage_list, stage_budgets)):
            is_final_stage = idx == (len(ns_stage_list) - 1)
            stage_mode_i = "parity" if bool(is_final_stage) else "accelerated"
            if stage_state is not None and int(stage_static_prev.cfg.ns) != int(ns_i):
                stage_state = interp_vmec_state(
                    stage_state,
                    m=stage_static_prev.modes.m,
                    n=stage_static_prev.modes.n,
                    lthreed=bool(stage_static_prev.cfg.lthreed),
                    lconm1=bool(getattr(stage_static_prev.cfg, "lconm1", True)),
                    ns_new=int(ns_i),
                )
            kwargs = dict(
                solver="vmec2000_iter",
                solver_mode=stage_mode_i,
                max_iter=int(niter_i),
                step_size=step_size,
                history_size=int(history_size),
                gn_damping=gn_damping,
                gn_cg_tol=gn_cg_tol,
                gn_cg_maxiter=int(gn_cg_maxiter),
                use_initial_guess=False,
                vmec_project=bool(vmec_project),
                use_restart_triggers=use_restart_triggers,
                vmecpp_restart=bool(vmecpp_restart),
                use_direct_fallback=use_direct_fallback,
                multigrid=False,
                multigrid_use_input_niter=False,
                verbose=bool(verbose),
                jit_forces=jit_forces,
                jit_precompile=jit_precompile,
                use_scan=False if bool(is_final_stage) else bool(use_scan),
                performance_mode=False if bool(is_final_stage) else True,
                scan_wout_corrector=scan_wout_corrector,
                stage_transition_heuristic=stage_transition_heuristic,
                stage_transition_factor=float(stage_transition_factor),
                stage_transition_scale=float(stage_transition_scale),
                grid=grid,
                cli_fixed_boundary_mode=False,
            )
            if stage_state is None:
                kwargs["ns_override"] = int(ns_i)
            else:
                kwargs["restart_state"] = stage_state
            stage_run = run_fixed_boundary(input_path, **kwargs)
            stage_runs.append(stage_run)
            stage_modes.append(str(stage_mode_i))
            stage_state = stage_run.state
            stage_static_prev = stage_run.static

        final_run = stage_runs[-1]
        if final_run.result is None:
            return final_run
        diag = dict(final_run.result.diagnostics)
        diag["solver_mode"] = str(solver_mode_eff)
        diag["accelerated_mode"] = True
        diag["cli_fixed_boundary_mode"] = True
        diag["cli_accelerated_fixed_policy"] = "budgeted_multigrid"
        diag["cli_accelerated_stage_ns"] = np.asarray(ns_stage_list, dtype=int)
        diag["cli_accelerated_stage_niter"] = np.asarray(stage_budgets, dtype=int)
        diag["cli_accelerated_stage_modes"] = np.asarray(stage_modes, dtype=object)
        diag["cli_accelerated_stage_fsq"] = np.asarray(
            [float(np.asarray(stage_run.result.w_history)[-1]) for stage_run in stage_runs],
            dtype=float,
        )
        diag["cli_accelerated_budget_total"] = int(warm_start_budget)
        diag["cli_accelerated_final_stage_budget"] = int(final_stage_budget)
        diag["multigrid_ns_stages"] = np.asarray(ns_stage_list, dtype=int)
        diag["multigrid_niter_stages"] = np.asarray(stage_budgets, dtype=int)
        diag["accelerated_single_grid_default"] = False
        final_run = replace(final_run, result=replace(final_run.result, diagnostics=diag))
        return _maybe_finish_cli_fixed_boundary_run(
            final_run,
            initial_policy="budgeted_multigrid",
            enabled=bool(cli_fixed_boundary_finish_enabled),
        )

    def _run_cli_explicit_staged_followup(
        *,
        ns_stage_list: list[int],
        niter_stage_list: list[int],
        ftol_stage_list: list[float],
    ) -> FixedBoundaryRun:
        stage_runs: list[FixedBoundaryRun] = []
        stage_state = None
        stage_static_prev = None
        stage_modes: list[str] = []
        for idx, (ns_i, niter_i, ftol_i) in enumerate(zip(ns_stage_list, niter_stage_list, ftol_stage_list)):
            if int(niter_i) <= 0:
                continue
            is_final_stage = idx == (len(ns_stage_list) - 1)
            if bool(is_final_stage):
                stage_mode_i = "parity"
            elif bool(cfg.lthreed) and (len(ns_stage_list) >= 3) and int(idx) == 0:
                # On staged 3D fixed-boundary cases, the coarsest continuation
                # stage determines which solution branch the later accelerated
                # continuation follows. Keep the entry and final stages on the
                # conservative VMEC-like controller, and accelerate only the
                # interior stages.
                stage_mode_i = "parity"
            else:
                stage_mode_i = "accelerated"
            if stage_state is not None and int(stage_static_prev.cfg.ns) != int(ns_i):
                stage_state = interp_vmec_state(
                    stage_state,
                    m=stage_static_prev.modes.m,
                    n=stage_static_prev.modes.n,
                    lthreed=bool(stage_static_prev.cfg.lthreed),
                    lconm1=bool(getattr(stage_static_prev.cfg, "lconm1", True)),
                    ns_new=int(ns_i),
                )
            kwargs = dict(
                solver="vmec2000_iter",
                solver_mode=stage_mode_i,
                max_iter=int(niter_i),
                step_size=step_size,
                history_size=int(history_size),
                gn_damping=gn_damping,
                gn_cg_tol=gn_cg_tol,
                gn_cg_maxiter=int(gn_cg_maxiter),
                use_initial_guess=False,
                vmec_project=bool(vmec_project),
                use_restart_triggers=use_restart_triggers,
                vmecpp_restart=bool(vmecpp_restart),
                use_direct_fallback=use_direct_fallback,
                multigrid=False,
                multigrid_use_input_niter=False,
                verbose=bool(verbose),
                jit_forces=jit_forces,
                jit_precompile=jit_precompile,
                use_scan=False if bool(is_final_stage) else bool(use_scan),
                performance_mode=False if bool(is_final_stage) else True,
                scan_wout_corrector=scan_wout_corrector,
                stage_transition_heuristic=stage_transition_heuristic,
                stage_transition_factor=float(stage_transition_factor),
                stage_transition_scale=float(stage_transition_scale),
                grid=grid,
                cli_fixed_boundary_mode=False,
            )
            if stage_state is None:
                kwargs["ns_override"] = int(ns_i)
            else:
                kwargs["restart_state"] = stage_state
            stage_run = run_fixed_boundary(input_path, **kwargs)
            stage_runs.append(stage_run)
            stage_modes.append(str(stage_mode_i))
            stage_state = stage_run.state
            stage_static_prev = stage_run.static

        final_run = stage_runs[-1]
        if final_run.result is None:
            return final_run
        diag = dict(final_run.result.diagnostics)
        diag["solver_mode"] = str(solver_mode_eff)
        diag["accelerated_mode"] = True
        diag["cli_fixed_boundary_mode"] = True
        diag["cli_staged_followup_policy"] = "input_multigrid"
        diag["cli_staged_followup_stage_ns"] = np.asarray(ns_stage_list, dtype=int)
        diag["cli_staged_followup_stage_niter"] = np.asarray(niter_stage_list, dtype=int)
        diag["cli_staged_followup_stage_modes"] = np.asarray(stage_modes, dtype=object)
        diag["cli_staged_followup_stage_fsq"] = np.asarray(
            [float(np.asarray(stage_run.result.w_history)[-1]) for stage_run in stage_runs],
            dtype=float,
        )
        final_run = replace(final_run, result=replace(final_run.result, diagnostics=diag))
        return final_run

    def _maybe_finish_cli_fixed_boundary_run(
        run_in: FixedBoundaryRun,
        *,
        initial_policy: str,
        enabled: bool,
    ) -> FixedBoundaryRun:
        if not bool(enabled):
            return run_in
        if run_in.result is None:
            return run_in
        base_diag = dict(run_in.result.diagnostics)
        base_diag["solver_mode"] = str(solver_mode_eff)
        base_diag["accelerated_mode"] = bool(accelerated_mode)
        base_diag["cli_fixed_boundary_mode"] = True
        base_diag["cli_fixed_boundary_initial_policy"] = str(initial_policy)
        target_fsq = _accelerated_fsq_total_target_from_ftol(float(indata.get_float("FTOL", 1.0e-13)))
        base_diag["fsq_total_target"] = float(target_fsq)
        staged_input = (ns_list_input is not None) and (len(ns_list_input) > 1)
        explicit_niter_stages = (
            [int(v) for v in niter_list_input]
            if (niter_list_input is not None) and (len(niter_list_input) == len(ns_list_input or []))
            else None
        )
        require_staged_followup = (
            bool(accelerated_mode)
            and str(initial_policy) == "single_grid"
            and bool(staged_input)
            and (explicit_niter_stages is not None)
            and bool(cfg.lthreed)
        )
        if bool(run_in.result.diagnostics.get("converged", False)) and (not bool(require_staged_followup)):
            base_diag["cli_fixed_boundary_finish_budgets"] = np.zeros((0,), dtype=int)
            base_diag["cli_fixed_boundary_finish_fsq"] = np.zeros((0,), dtype=float)
            base_diag["cli_fixed_boundary_finish_converged"] = np.zeros((0,), dtype=bool)
            base_diag["cli_fixed_boundary_finish_modes"] = np.asarray([], dtype=object)
            base_diag["cli_fixed_boundary_full_parity_fallback"] = False
            return replace(run_in, result=replace(run_in.result, diagnostics=base_diag))
        run_in_fsq = float(np.asarray(run_in.result.w_history)[-1])
        if (run_in_fsq <= float(target_fsq)) and (not bool(require_staged_followup)):
            base_diag["converged"] = True
            base_diag["converged_by_total_fsq"] = True
            base_diag["cli_fixed_boundary_finish_budgets"] = np.zeros((0,), dtype=int)
            base_diag["cli_fixed_boundary_finish_fsq"] = np.zeros((0,), dtype=float)
            base_diag["cli_fixed_boundary_finish_converged"] = np.zeros((0,), dtype=bool)
            base_diag["cli_fixed_boundary_finish_modes"] = np.asarray([], dtype=object)
            base_diag["cli_fixed_boundary_full_parity_fallback"] = False
            return replace(run_in, result=replace(run_in.result, diagnostics=base_diag))

        base_total_budget = max(1, int(max_iter))

        best_run = run_in
        best_fsq = float(np.asarray(run_in.result.w_history)[-1])
        attempt_budgets: list[int] = []
        attempt_fsq: list[float] = []
        attempt_converged: list[bool] = []
        attempt_modes: list[str] = []
        fallback_used = False
        staged_followup_used = False
        staged_followup_policy = ""
        staged_followup_ns = np.zeros((0,), dtype=int)
        staged_followup_niter = np.zeros((0,), dtype=int)
        staged_followup_modes = np.asarray([], dtype=object)
        staged_followup_fsq = np.zeros((0,), dtype=float)

        def _resolve_finish_jit_forces(static_i: VMECStatic, niter_i: int) -> bool:
            if isinstance(jit_forces, str):
                if jit_forces.strip().lower() != "auto":
                    return True
                try:
                    nmodes_i = int(np.asarray(static_i.modes.m).size)
                    nrzt = int(static_i.cfg.ns) * int(static_i.cfg.ntheta) * int(static_i.cfg.nzeta)
                    work = nmodes_i * nrzt
                except Exception:
                    return True
                if int(niter_i) >= 5:
                    return True
                return bool(work >= 2_000_000)
            return bool(jit_forces)

        def _run_finish_attempt(*, budget_i: int, mode_i: str, use_scan_i: bool, performance_mode_i: bool):
            static_i = best_run.static
            mode_i_l = str(mode_i).strip().lower()
            scan_minimal_default_i = True if (bool(performance_mode_i) and (not bool(verbose))) else None
            if step_size is _STEP_SIZE_SENTINEL or step_size is None:
                step_size_finish = float(indata.get_float("DELT", 5e-3))
            else:
                step_size_finish = float(step_size)
            finish_fsq_total_target = float(target_fsq) if mode_i_l == "accelerated" else None
            finish_resume_state_mode = "minimal" if mode_i_l == "accelerated" else "full"
            res_i = solve_fixed_boundary_residual_iter(
                best_run.state,
                static_i,
                indata=indata,
                signgs=best_run.signgs,
                ftol=float(indata.get_float("FTOL", 1.0e-13)),
                max_iter=int(budget_i),
                step_size=float(step_size_finish),
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
                use_restart_triggers=True if use_restart_triggers is None else bool(use_restart_triggers),
                vmecpp_restart=bool(vmecpp_restart),
                stage_prev_fsq=None,
                stage_transition_factor=float(stage_transition_factor),
                stage_transition_scale=float(stage_transition_scale),
                use_direct_fallback=use_direct_fallback,
                # CLI finish attempts deliberately restart from the current
                # equilibrium state only. Reusing the nonlinear controller
                # caches was materially less robust on the hard staged inputs.
                resume_state=None,
                verbose=False,
                verbose_vmec2000_table=False,
                jit_precompile=False,
                jit_warmup_iters=0,
                use_scan=bool(use_scan_i),
                scan_minimal_default=scan_minimal_default_i,
                light_history=True,
                resume_state_mode=finish_resume_state_mode,
                fsq_total_target=finish_fsq_total_target,
                jit_forces=_resolve_finish_jit_forces(static_i, int(budget_i)),
            )
            return replace(best_run, state=res_i.state, result=res_i)

        if staged_input and bool(accelerated_mode) and str(initial_policy) == "single_grid":
            explicit_ftol_stages = (
                [float(v) for v in ftol_list_input]
                if (ftol_list_input is not None) and (len(ftol_list_input) == len(ns_list_input))
                else [float(indata.get_float("FTOL", 1.0e-13))] * len(ns_list_input)
            )
            if explicit_niter_stages is not None:
                staged_followup = _run_cli_explicit_staged_followup(
                    ns_stage_list=[int(v) for v in ns_list_input],
                    niter_stage_list=explicit_niter_stages,
                    ftol_stage_list=explicit_ftol_stages,
                )
                staged_followup_used = True
                staged_followup_policy = "input_multigrid"
                staged_diag = dict(staged_followup.result.diagnostics)
                staged_followup_ns = np.asarray(staged_diag.get("cli_staged_followup_stage_ns", []), dtype=int)
                staged_followup_niter = np.asarray(staged_diag.get("cli_staged_followup_stage_niter", []), dtype=int)
                staged_followup_modes = np.asarray(staged_diag.get("cli_staged_followup_stage_modes", []), dtype=object)
                staged_followup_fsq = np.asarray(staged_diag.get("cli_staged_followup_stage_fsq", []), dtype=float)
                staged_fsq_val = float(np.asarray(staged_followup.result.w_history)[-1])
                staged_conv = bool(staged_followup.result.diagnostics.get("converged", False)) or (
                    staged_fsq_val <= float(target_fsq)
                )
                if staged_conv or (staged_fsq_val < float(best_fsq)):
                    best_run = staged_followup
                    best_fsq = float(staged_fsq_val)

        max_fallback_budget = int(2 * base_total_budget)
        improvement_floor = np.finfo(float).eps * max(1.0, abs(float(best_fsq)), abs(float(target_fsq)))
        if (
            bool(accelerated_mode)
            and str(initial_policy) == "single_grid"
            and (not bool(staged_followup_used))
            and not (bool(best_run.result.diagnostics.get("converged", False)) or float(best_fsq) <= float(target_fsq))
        ):
            accel_budget_i = int(base_total_budget)
            accel_budget_used = 0
            while int(accel_budget_i) >= 1 and int(accel_budget_used) < int(max_fallback_budget):
                prev_best_fsq = float(best_fsq)
                trial = _run_finish_attempt(
                    budget_i=accel_budget_i,
                    mode_i="accelerated",
                    use_scan_i=True,
                    performance_mode_i=True,
                )
                trial_fsq = float(np.asarray(trial.result.w_history)[-1])
                trial_conv = bool(trial.result.diagnostics.get("converged", False)) or (
                    float(trial_fsq) <= float(target_fsq)
                )
                attempt_budgets.append(int(accel_budget_i))
                attempt_fsq.append(float(trial_fsq))
                attempt_converged.append(bool(trial_conv))
                attempt_modes.append("accelerated")
                accel_budget_used += int(accel_budget_i)
                improved = trial_conv or (float(trial_fsq) < float(prev_best_fsq - improvement_floor))
                if improved:
                    best_run = trial
                    best_fsq = float(trial_fsq)
                if trial_conv or (not improved):
                    break
        if not (bool(best_run.result.diagnostics.get("converged", False)) or float(best_fsq) <= float(target_fsq)):
            budget_i = int(base_total_budget)
            while int(budget_i) >= 1:
                prev_best_fsq = float(best_fsq)
                trial = _run_finish_attempt(
                    budget_i=budget_i,
                    mode_i="parity",
                    use_scan_i=False,
                    performance_mode_i=False,
                )
                trial_fsq = float(np.asarray(trial.result.w_history)[-1])
                trial_conv = bool(trial.result.diagnostics.get("converged", False)) or (
                    float(trial_fsq) <= float(target_fsq)
                )
                attempt_budgets.append(int(budget_i))
                attempt_fsq.append(float(trial_fsq))
                attempt_converged.append(bool(trial_conv))
                attempt_modes.append("parity")
                improved = trial_conv or (float(trial_fsq) < float(prev_best_fsq - improvement_floor))
                if improved:
                    best_run = trial
                    best_fsq = float(trial_fsq)
                if trial_conv:
                    break
                if improved:
                    continue
                next_budget = max(1, int(np.ceil(float(budget_i) / 2.0)))
                if int(next_budget) == int(budget_i):
                    break
                budget_i = int(next_budget)

        if staged_input and not (
            bool(best_run.result.diagnostics.get("converged", False)) or float(best_fsq) <= float(target_fsq)
        ) and bool(accelerated_mode):
            fallback_used = True
            fallback = run_fixed_boundary(
                input_path,
                solver="vmec2000_iter",
                solver_mode="parity",
                max_iter=int(max_fallback_budget),
                step_size=step_size,
                history_size=int(history_size),
                gn_damping=gn_damping,
                gn_cg_tol=gn_cg_tol,
                gn_cg_maxiter=int(gn_cg_maxiter),
                use_initial_guess=False,
                vmec_project=bool(vmec_project),
                use_restart_triggers=use_restart_triggers,
                vmecpp_restart=bool(vmecpp_restart),
                use_direct_fallback=use_direct_fallback,
                multigrid=multigrid if bool(multigrid_user_provided) else None,
                multigrid_use_input_niter=bool(multigrid_use_input_niter),
                verbose=bool(verbose),
                jit_forces=jit_forces,
                jit_precompile=jit_precompile,
                use_scan=False,
                performance_mode=False,
                scan_wout_corrector=scan_wout_corrector,
                stage_transition_heuristic=stage_transition_heuristic,
                stage_transition_factor=float(stage_transition_factor),
                stage_transition_scale=float(stage_transition_scale),
                grid=grid,
                cli_fixed_boundary_mode=False,
            )
            fallback_fsq = float(np.asarray(fallback.result.w_history)[-1])
            if bool(fallback.result.diagnostics.get("converged", False)) or fallback_fsq < best_fsq:
                best_run = fallback
                best_fsq = float(fallback_fsq)

        diag = dict(base_diag)
        diag.update(best_run.result.diagnostics)
        diag["solver_mode"] = str(solver_mode_eff)
        diag["accelerated_mode"] = bool(accelerated_mode)
        diag["cli_fixed_boundary_mode"] = True
        diag["cli_fixed_boundary_initial_policy"] = str(initial_policy)
        if float(best_fsq) <= float(target_fsq):
            diag["converged"] = True
            diag["converged_by_total_fsq"] = True
        diag["cli_fixed_boundary_finish_budgets"] = np.asarray(attempt_budgets, dtype=int)
        diag["cli_fixed_boundary_finish_fsq"] = np.asarray(attempt_fsq, dtype=float)
        diag["cli_fixed_boundary_finish_converged"] = np.asarray(attempt_converged, dtype=bool)
        diag["cli_fixed_boundary_finish_modes"] = np.asarray(attempt_modes)
        diag["cli_fixed_boundary_full_parity_fallback"] = bool(fallback_used)
        diag["cli_fixed_boundary_staged_followup_used"] = bool(staged_followup_used)
        diag["cli_fixed_boundary_staged_followup_policy"] = str(staged_followup_policy)
        diag["cli_fixed_boundary_staged_followup_ns"] = staged_followup_ns
        diag["cli_fixed_boundary_staged_followup_niter"] = staged_followup_niter
        diag["cli_fixed_boundary_staged_followup_modes"] = staged_followup_modes
        diag["cli_fixed_boundary_staged_followup_fsq"] = staged_followup_fsq
        best_run = replace(best_run, result=replace(best_run.result, diagnostics=diag))
        return best_run

    multigrid_use_input_niter = bool(multigrid_use_input_niter)
    multigrid_user_provided = multigrid is not None
    accelerated_single_grid_default = False
    if multigrid is None:
        multigrid = solver_lower == "vmec2000_iter"
        if bool(cli_budgeted_multigrid_requested):
            multigrid = True
        elif accelerated_mode and (not bool(cfg.lfreeb)):
            # In accelerated fixed-boundary mode, direct final-grid solves avoid
            # per-stage interpolation/recompilation overhead and have been more
            # efficient across the heavy bundled cases tested so far.
            multigrid = False
            accelerated_single_grid_default = True
    if max_iter is _MAX_ITER_SENTINEL:
        if solver_lower in ("vmec2000_iter", "vmec2000_scan", "vmec2000_iter_fast"):
            niter_list = _as_list(indata.get("NITER_ARRAY", None))
            if niter_list:
                # VMEC2000 behavior: when NITER_ARRAY is present, it defines
                # the stage budgets even if there is only one stage.
                max_iter = int(sum(int(v) for v in niter_list))
            else:
                max_iter = int(indata.get_int("NITER", 10))
        else:
            max_iter = 10
    max_iter = int(max_iter)
    if restart_state_eff is not None:
        multigrid = False
    if restart_solver_state is not None:
        multigrid = False
    multigrid = bool(multigrid) and (ns_override is None)
    if stage_transition_heuristic is None:
        env_stage = os.getenv("VMEC_JAX_STAGE_HEURISTIC", "").strip().lower()
        if env_stage in ("1", "true", "yes"):
            stage_transition_heuristic = True
        elif env_stage in ("0", "false", "no"):
            stage_transition_heuristic = False
        else:
            stage_transition_heuristic = False
    stage_transition_heuristic = bool(stage_transition_heuristic)

    fb_strict_env = os.getenv("VMEC_JAX_FREEB_STRICT", "1").strip().lower()
    fb_strict = fb_strict_env not in ("", "0", "false", "no")
    validate_free_boundary_config(cfg, strict=fb_strict)
    prepared_fb = prepare_mgrid_for_config(cfg, load_fields=False, strict=fb_strict)
    fb_meta: MGridMetadata | None = None
    fb_extcur: tuple[float, ...] | None = None
    if isinstance(prepared_fb, PreparedMGrid):
        fb_meta = prepared_fb.metadata
        fb_extcur = prepared_fb.extcur

    # Build the initial state on either the final grid (single-grid solvers and
    # use_initial_guess) or on the first multigrid stage for VMEC-style solves.
    ns_stages = [int(cfg.ns)]
    if multigrid:
        if ns_list_input:
            ns_stages = [int(v) for v in ns_list_input]

    # When NITER_ARRAY is present, treat it as the authoritative total unless
    # the caller explicitly overrides max_iter.
    niter_list = niter_list_input
    if niter_list:
        niter_sum = int(sum(int(v) for v in niter_list))
        niter_default = int(indata.get_int("NITER", max_iter))
        if (not max_iter_overridden) and int(max_iter) == niter_default:
            max_iter = niter_sum

    if bool(cli_budgeted_multigrid_requested):
        budget_total = _accelerated_cli_budgeted_total_iters(total_budget=int(max_iter), ns_stages=ns_stages)
        return _run_cli_accelerated_budgeted_multigrid(
            ns_stage_list=list(ns_stages),
            warm_start_budget=int(budget_total),
            final_stage_budget=int(max_iter),
        )

    # Precompute boundary coefficients without triggering JAX initialization.
    boundary_coeffs = None
    if restart_state_eff is None:
        from .modes import vmec_mode_table

        boundary_modes = vmec_mode_table(cfg.mpol, cfg.ntor)
        boundary_coeffs = boundary_from_indata(indata, boundary_modes)

    # VMEC readin.f hard-codes signgs = -1 (then flips theta if needed).
    # For VMEC2000-iter parity, ignore input SIGNGS and match VMEC behavior.
    if solver_lower in ("vmec2000_iter", "vmec2000_scan", "vmec2000_iter_fast"):
        signgs = -1
    else:
        signgs = int(indata.get_int("SIGNGS", -1))
        if signgs not in (-1, 1):
            signgs = -1
    if solver_lower in ("vmec2000_iter", "vmec2000_scan", "vmec2000_iter_fast"):
        force_jit_env = os.getenv("VMEC_JAX_VMEC2000_FORCE_JIT", "").strip().lower()
        force_nojit_env = os.getenv("VMEC_JAX_VMEC2000_FORCE_NOJIT", "").strip().lower()
        if force_jit_env not in ("", "0", "false", "no"):
            jit_forces = True
        elif force_nojit_env not in ("", "0", "false", "no"):
            jit_forces = False
        elif isinstance(jit_forces, str):
            # default to JIT for vmec2000 unless explicitly disabled
            if jit_forces.strip().lower() == "auto":
                jit_forces = True

    gamma = indata.get_float("GAMMA", 0.0)
    static = None
    static_final = None
    bdy = None
    flux = None
    prof = None
    pressure = None

    def _build_static_cfg(cfg_in: VMECConfig) -> VMECStatic:
        if bool(cfg_in.lfreeb):
            return build_static(
                cfg_in,
                grid=grid,
                mgrid_metadata=fb_meta,
                free_boundary_extcur=fb_extcur,
            )
        return build_static(cfg_in, grid=grid)

    def _profiles_from_static(static_in: VMECStatic):
        flux_local = flux_profiles_from_indata(indata, static_in.s, signgs=signgs)
        # VMEC evaluates pressure/iota/current profiles on the radial half mesh.
        if int(cfg.ns) < 2:
            s_half = np.asarray(static_in.s)
        else:
            s_full = np.asarray(static_in.s)
            s_half = np.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])], axis=0)
        prof_local = eval_profiles(indata, s_half)
        pressure_local = prof_local.get("pressure", np.zeros_like(np.asarray(static_in.s)))
        return flux_local, prof_local, pressure_local

    def _ensure_static_profiles() -> None:
        nonlocal static, bdy, flux, prof, pressure
        if static is None:
            static = _build_static_cfg(cfg)
        if bdy is None:
            bdy = boundary_from_indata(indata, static.modes)
        if flux is None or prof is None or pressure is None:
            flux, prof, pressure = _profiles_from_static(static)

    if step_size is _STEP_SIZE_SENTINEL or step_size is None:
        if solver_lower in ("vmec2000_iter", "vmec2000_scan", "vmec2000_iter_fast"):
            step_size_val = float(indata.get_float("DELT", 5e-3))
        else:
            step_size_val = 5e-3
    else:
        step_size_val = float(step_size)

    if verbose and (solver_lower != "vmec2000_iter" or use_initial_guess):
        mode = "initial guess" if use_initial_guess else f"{solver} solve"
        print(f"[vmec_jax] fixed-boundary run ({mode})", flush=True)
        print(f"[vmec_jax] input={input_path}", flush=True)
        print(f"[vmec_jax] ns={cfg.ns} mpol={cfg.mpol} ntor={cfg.ntor} nfp={cfg.nfp}", flush=True)
        if not use_initial_guess:
            print(f"[vmec_jax] max_iter={max_iter} step_size={step_size_val} history_size={history_size}", flush=True)
    elif verbose and (solver_lower == "vmec2000_iter") and (not use_initial_guess):
        from datetime import datetime

        now = datetime.now()
        date_str = now.strftime("%b %d,%Y")
        time_str = now.strftime("%H:%M:%S")
        input_name = Path(input_path).name.upper()
        version = os.getenv("VMEC_JAX_VMEC2000_VERSION", "vmec_jax")
        print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -", flush=True)
        print("  SEQ =    1 TIME SLICE  0.0000E+00", flush=True)
        print(f"  PROCESSING {input_name}", flush=True)
        print(f"  THIS IS PARVMEC (PARALLEL VMEC), VERSION {version}", flush=True)
        print("  Lambda: Full Radial Mesh. L-Force: hybrid full/half.", flush=True)
        print("", flush=True)
        print(f"  COMPUTER:    OS:    RELEASE:   DATE = {date_str}  TIME = {time_str}", flush=True)
        print("", flush=True)

    def _initial_guess_with_optional_nojit(static_in, bdy_in, *, force_disable_jit: bool = False):
        disable_env = os.getenv("VMEC_JAX_DISABLE_JIT_INIT", "") not in ("", "0")
        if not (disable_env or force_disable_jit):
            return initial_guess_from_boundary(
                static_in,
                bdy_in,
                indata,
                vmec_project=vmec_project,
                infer_axis_if_missing=axis_infer_missing,
            )
        try:
            import jax

            with jax.disable_jit():
                return initial_guess_from_boundary(
                    static_in,
                    bdy_in,
                    indata,
                    vmec_project=vmec_project,
                    infer_axis_if_missing=axis_infer_missing,
                )
        except Exception:
            return initial_guess_from_boundary(
                static_in,
                bdy_in,
                indata,
                vmec_project=vmec_project,
                infer_axis_if_missing=axis_infer_missing,
            )

    if use_initial_guess:
        _ensure_static_profiles()
        if restart_state_eff is not None:
            st0 = restart_state_eff
        else:
            st0 = _initial_guess_with_optional_nojit(static, bdy)
            _maybe_dump_xc_init(state=st0, static=static, label="init")
        return FixedBoundaryRun(
            cfg=cfg,
            indata=indata,
            static=static,
            state=st0,
            result=None,
            flux=flux,
            profiles=prof,
            signgs=signgs,
        )

    if performance_mode:
        if solver_lower == "vmec2000_iter":
            solver_lower = "vmec2000_iter_fast"

    # Fast mode keeps minimal history only when not printing (verbose=False).
    scan_minimal_default = True if (bool(performance_mode) and (not bool(verbose))) else None

    solver = solver_lower
    if solver in ("vmec2000_iter_fast", "vmec2000_scan"):
        use_scan = True
        solver = "vmec2000_iter"
    # Parity mode defaults to the VMEC2000 non-scan control path unless
    # explicitly forced via environment variables.
    if solver == "vmec2000_iter" and (not bool(performance_mode)):
        use_scan = False
    if os.getenv("VMEC_JAX_USE_SCAN", "") not in ("", "0"):
        use_scan = True
    if solver == "gd":
        _ensure_static_profiles()
        st0 = restart_state_eff if restart_state_eff is not None else _initial_guess_with_optional_nojit(static, bdy)
        res = solve_fixed_boundary_gd(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=gamma,
            max_iter=int(max_iter),
            step_size=float(step_size_val),
            jacobian_penalty=1e3,
            jit_grad=True,
            verbose=bool(verbose),
        )
    elif solver == "lbfgs":
        _ensure_static_profiles()
        st0 = restart_state_eff if restart_state_eff is not None else _initial_guess_with_optional_nojit(static, bdy)
        res = solve_fixed_boundary_lbfgs(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=gamma,
            max_iter=int(max_iter),
            step_size=float(step_size_val),
            history_size=int(history_size),
            jit_grad=True,
            verbose=bool(verbose),
        )
    elif solver == "vmec_lbfgs":
        from .solve import solve_fixed_boundary_lbfgs_vmec_residual
        _ensure_static_profiles()
        st0 = restart_state_eff if restart_state_eff is not None else _initial_guess_with_optional_nojit(static, bdy)

        res = solve_fixed_boundary_lbfgs_vmec_residual(
            st0,
            static,
            indata=indata,
            signgs=signgs,
            history_size=int(history_size),
            max_iter=int(max_iter),
            step_size=float(step_size_val),
            jit_grad=True,
            preconditioner="mode_diag+radial_tridi",
            precond_exponent=1.0,
            precond_radial_alpha=0.2,
            verbose=bool(verbose),
        )
    elif solver == "vmec_gn":
        from .solve import solve_fixed_boundary_gn_vmec_residual
        _ensure_static_profiles()
        st0 = restart_state_eff if restart_state_eff is not None else _initial_guess_with_optional_nojit(static, bdy)

        res = solve_fixed_boundary_gn_vmec_residual(
            st0,
            static,
            indata=indata,
            signgs=signgs,
            max_iter=int(max_iter),
            step_size=float(step_size_val),
            damping=None if gn_damping is None else float(gn_damping),
            cg_tol=None if gn_cg_tol is None else float(gn_cg_tol),
            cg_maxiter=int(gn_cg_maxiter),
            jit_kernels=True,
            verbose=bool(verbose),
        )
    elif solver == "vmec2000_iter":
        def _distribute_iters(*, iters: int, nstep: int) -> list[int]:
            iters = int(iters)
            nstep = int(nstep)
            if iters <= 0:
                return [0]
            if nstep <= 1:
                return [iters]
            base, rem = divmod(iters, nstep)
            if base == 0:
                return [iters]
            return [base + (1 if i < rem else 0) for i in range(nstep)]

        # Stage controls.
        nstep = len(ns_stages)
        niter_array = indata.get("NITER_ARRAY", None)
        ftol_array = indata.get("FTOL_ARRAY", None)
        niter_list = _as_list(niter_array)
        ftol_list = _as_list(ftol_array)
        niter_stages_input = [int(v) for v in niter_list] if niter_list and len(niter_list) == nstep else None
        ftol_stages_input = [float(v) for v in ftol_list] if ftol_list and len(ftol_list) == nstep else None
        accelerated_single_grid_budget = (
            bool(accelerated_single_grid_default)
            and (not bool(multigrid_user_provided))
            and int(nstep) == 1
            and bool(niter_list)
            and (not bool(max_iter_overridden))
        )
        if multigrid_use_input_niter:
            niter_stages = niter_stages_input
            ftol_stages = ftol_stages_input
            if niter_stages is None:
                if accelerated_single_grid_budget:
                    # Accelerated single-grid runs collapse the staged solve to
                    # a single final-grid solve, but they should still honor
                    # the total input iteration budget implied by NITER_ARRAY.
                    niter_stages = [int(max_iter)]
                elif max_iter_overridden:
                    # Explicit caller budget: distribute total iterations across stages.
                    niter_stages = _distribute_iters(iters=int(max_iter), nstep=int(nstep))
                else:
                    # VMEC2000 semantics: when NITER_ARRAY is absent, NITER applies
                    # to each stage (not split across stages).
                    niter_stage = int(indata.get_int("NITER", int(max_iter)))
                    niter_stages = [niter_stage] * nstep
            else:
                # Respect caller `max_iter` as a total budget while preserving
                # VMEC2000 stage order semantics: consume iterations on coarse
                # stages first, then proceed to finer stages.
                budget = int(max_iter)
                remaining = max(0, budget)
                out = [0] * nstep
                for i in range(nstep):
                    if remaining <= 0:
                        break
                    cap = max(0, int(niter_stages[i]))
                    take = min(cap, remaining)
                    out[i] = take
                    remaining -= take
                niter_stages = out
            if ftol_stages is None:
                if bool(accelerated_single_grid_default) and int(nstep) == 1 and bool(ftol_list):
                    ftol_stages = [float(ftol_list[-1])]
                else:
                    ftol_stages = [float(indata.get_float("FTOL", 1e-13))] * nstep
        else:
            niter_stages = _distribute_iters(iters=int(max_iter), nstep=int(nstep))
            # VMEC2000 uses FTOL_ARRAY when present, even for single-stage runs.
            if ftol_stages_input is not None:
                ftol_stages = ftol_stages_input
            else:
                ftol_stages = [float(indata.get_float("FTOL", 1e-13))] * nstep

        # Run coarse -> fine stages with VMEC `interp.f` interpolation.
        stage_results: list[SolveVmecResidualResult] = []
        stage_offsets: list[int] = []
        from .modes import vmec_mode_table

        header_modes = vmec_mode_table(cfg.mpol, cfg.ntor)
        nmodes_header = int(np.asarray(header_modes.m).size)

        state = restart_state_eff
        static_prev = None
        static_final = None
        resume_state_stage = restart_solver_state
        multigrid_resume = False
        if multigrid:
            # Default to VMEC2000 behavior (reset time-step state per stage).
            env_resume = os.getenv("VMEC_JAX_MULTIGRID_RESUME", "0")
            multigrid_resume = env_resume.strip().lower() not in ("", "0", "false", "no")

        def _sanitize_resume_state_for_stage(resume_state):
            if resume_state is None:
                return None
            # Keep only time-step/momentum scalars that are safe across ns changes.
            time_step = resume_state.get("time_step", None)
            if time_step is None:
                return None
            # Clamp to the nominal DELT for stability when changing resolution.
            try:
                time_step = min(float(time_step), float(step_size_val))
            except Exception:
                time_step = float(time_step)
            inv_tau = [0.15 / float(time_step)] * 10
            out = {
                "time_step": float(time_step),
                "inv_tau": list(inv_tau),
            }
            if "flip_sign" in resume_state:
                out["flip_sign"] = float(resume_state["flip_sign"])
            out["iter_offset"] = 0
            out["vmec2000_cache_valid"] = False
            return out
        def _resolve_jit_forces(flag: bool | str, static_i: VMECStatic, niter_i: int) -> bool:
            if isinstance(flag, str):
                if flag.strip().lower() != "auto":
                    return True
                try:
                    nmodes_i = int(np.asarray(static_i.modes.m).size)
                    nrzt = int(static_i.cfg.ns) * int(static_i.cfg.ntheta) * int(static_i.cfg.nzeta)
                    work = nmodes_i * nrzt
                except Exception:
                    return True
                # Heuristic: avoid JIT for very small workloads unless the stage
                # will run long enough to amortize compilation cost.
                if int(niter_i) >= 5:
                    return True
                return bool(work >= 2_000_000)
            return bool(flag)

        env_precompile_stages = os.getenv("VMEC_JAX_PRECOMPILE_STAGES", "0")
        precompile_stages = env_precompile_stages.strip().lower() not in ("", "0", "false", "no")

        prev_stage_fsq = None
        ftol_last = None
        step_size_last = None
        for i, (ns_i, niter_i, ftol_i) in enumerate(zip(ns_stages, niter_stages, ftol_stages)):
            if int(niter_i) <= 0:
                continue
            if verbose:
                print(
                    f"  NS = {int(ns_i):4d} NO. FOURIER MODES = {nmodes_header:4d} "
                    f"FTOLV = {float(ftol_i):10.3E} NITER = {int(niter_i):6d}",
                    flush=True,
                )
                print("  PROCESSOR COUNT - RADIAL:    1", flush=True)
                print("", flush=True)
                if bool(cfg.lasym):
                    print(
                        "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)  ZAX(v=0)    DELT       WMHD",
                        flush=True,
                    )
                else:
                    print(
                        "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                        flush=True,
                    )

            cfg_i = replace(cfg, ns=int(ns_i))
            static_i = _build_static_cfg(cfg_i)
            scan_mode = bool(use_scan)
            if accelerated_mode:
                scan_mode = not bool(cfg_i.lfreeb)
            if bool(cfg.lasym):
                # For LASYM fixed-boundary stages, allow scan as a candidate in
                # the default fast path and let the automatic selector decide
                # whether the warmed scan route is both safe and worthwhile.
                lasym_scan_env = os.getenv("VMEC_JAX_LASYM_USE_SCAN", "auto").strip().lower()
                if lasym_scan_env in ("0", "false", "no", "off"):
                    scan_mode = False
                elif lasym_scan_env not in ("", "auto"):
                    scan_mode = True
            # Optional scan-parity guard: probe a few iterations and disable scan
            # if it diverges from the non-scan VMEC2000 path.
            scan_guard_default = "0"
            scan_guard_env = os.getenv("VMEC_JAX_SCAN_PARITY_GUARD", scan_guard_default).strip().lower()
            scan_guard_enabled = scan_guard_env not in ("", "0", "false", "no")
            if (not accelerated_mode) and scan_mode and scan_guard_enabled and int(niter_i) >= 3:
                probe_iters = min(10, int(niter_i))
                try:
                    guard_rtol = float(os.getenv("VMEC_JAX_SCAN_GUARD_RTOL", "1e-3"))
                    guard_atol = float(os.getenv("VMEC_JAX_SCAN_GUARD_ATOL", "1e-12"))
                    probe_kwargs = dict(
                        indata=indata,
                        signgs=signgs,
                        ftol=float(ftol_i),
                        max_iter=int(probe_iters),
                        step_size=float(step_size_val),
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
                        use_restart_triggers=True if use_restart_triggers is None else bool(use_restart_triggers),
                        vmecpp_restart=bool(vmecpp_restart),
                        use_direct_fallback=False,
                        stage_prev_fsq=None,
                        stage_transition_factor=float(stage_transition_factor),
                        stage_transition_scale=float(stage_transition_scale),
                        resume_state=None,
                        verbose=False,
                        verbose_vmec2000_table=False,
                        jit_precompile=False,
                        jit_warmup_iters=0,
                        scan_minimal_default=scan_minimal_default,
                    )
                    res_probe_scan = solve_fixed_boundary_residual_iter(
                        state,
                        static_i,
                        jit_forces=_resolve_jit_forces(jit_forces, static_i, int(probe_iters)),
                        use_scan=True,
                        **probe_kwargs,
                    )
                    res_probe_direct = solve_fixed_boundary_residual_iter(
                        state,
                        static_i,
                        jit_forces=_resolve_jit_forces(jit_forces, static_i, int(probe_iters)),
                        use_scan=False,
                        **probe_kwargs,
                    )
                    fsqr_scan = np.asarray(res_probe_scan.fsqr2_history)
                    fsqz_scan = np.asarray(res_probe_scan.fsqz2_history)
                    fsql_scan = np.asarray(res_probe_scan.fsql2_history)
                    fsqr_ref = np.asarray(res_probe_direct.fsqr2_history)
                    fsqz_ref = np.asarray(res_probe_direct.fsqz2_history)
                    fsql_ref = np.asarray(res_probe_direct.fsql2_history)
                    mismatch = False
                    if fsqr_scan.size == fsqr_ref.size == probe_iters:
                        if not np.allclose(fsqr_scan, fsqr_ref, rtol=guard_rtol, atol=guard_atol):
                            mismatch = True
                        if not np.allclose(fsqz_scan, fsqz_ref, rtol=guard_rtol, atol=guard_atol):
                            mismatch = True
                        if not np.allclose(fsql_scan, fsql_ref, rtol=guard_rtol, atol=guard_atol):
                            mismatch = True
                    else:
                        mismatch = True
                    if mismatch:
                        scan_mode = False
                        if bool(verbose):
                            print(
                                "[vmec_jax] scan parity guard: disabling scan for this stage (probe mismatch)",
                                flush=True,
                            )
                except Exception as exc:
                    # If probe fails, fall back to the safe (non-scan) path.
                    scan_mode = False
                    if bool(verbose):
                        print(
                            f"[vmec_jax] scan parity guard probe failed ({type(exc).__name__}); using non-scan for this stage.",
                            flush=True,
                        )
            jit_forces_base = _resolve_jit_forces(jit_forces, static_i, int(niter_i))
            jit_forces_eff = jit_forces_base
            if scan_mode and solver == "vmec2000_iter":
                scan_jit_env = os.getenv("VMEC_JAX_SCAN_JIT_FORCES")
                if scan_jit_env is None:
                    # Fast mode keeps JIT enabled for scan; parity mode disables by default.
                    if not bool(performance_mode):
                        jit_forces_eff = False
                elif scan_jit_env.strip().lower() in ("", "0", "false", "no"):
                    jit_forces_eff = False
                else:
                    jit_forces_eff = True
            jit_precompile_eff = False
            if bool(jit_forces_eff) and (not bool(scan_mode)):
                if jit_precompile is None:
                    val = os.getenv("VMEC_JAX_JIT_PRECOMPILE", "1").strip().lower()
                    jit_precompile_eff = val not in ("", "0", "false", "no")
                else:
                    jit_precompile_eff = bool(jit_precompile)
            jit_warmup_iters = 0
            if bool(jit_forces_eff) and (not bool(scan_mode)):
                env_warmup = os.getenv("VMEC_JAX_JIT_WARMUP_ITERS")
                if env_warmup is not None:
                    try:
                        jit_warmup_iters = max(0, int(env_warmup))
                    except Exception:
                        jit_warmup_iters = 2
                else:
                    jit_warmup_iters = 0 if bool(jit_precompile_eff) else 2
            # Precompute non-scan JIT settings for fast-fallback.
            jit_precompile_noscan = False
            if bool(jit_forces_base):
                if jit_precompile is None:
                    val = os.getenv("VMEC_JAX_JIT_PRECOMPILE", "1").strip().lower()
                    jit_precompile_noscan = val not in ("", "0", "false", "no")
                else:
                    jit_precompile_noscan = bool(jit_precompile)
            jit_warmup_noscan = 0
            if bool(jit_forces_base):
                env_warmup = os.getenv("VMEC_JAX_JIT_WARMUP_ITERS")
                if env_warmup is not None:
                    try:
                        jit_warmup_noscan = max(0, int(env_warmup))
                    except Exception:
                        jit_warmup_noscan = 2
                else:
                    jit_warmup_noscan = 0 if bool(jit_precompile_noscan) else 2
            if i == 0:
                if state is None:
                    if boundary_coeffs is None:
                        raise ValueError("boundary_coeffs missing; cannot build initial guess")
                    state = _initial_guess_with_optional_nojit(
                        static_i,
                        boundary_coeffs,
                        force_disable_jit=bool(jit_warmup_iters > 0),
                    )
                    _maybe_dump_xc_init(state=state, static=static_i, label="stage0")
            else:
                state = interp_vmec_state(
                    state,
                    m=static_prev.modes.m,
                    n=static_prev.modes.n,
                    lthreed=bool(static_prev.cfg.lthreed),
                    lconm1=bool(getattr(static_prev.cfg, "lconm1", True)),
                    ns_new=int(ns_i),
                )
            state_stage_start = state
            static_prev = static_i
            static_final = static_i

            stage_offsets.append(sum(int(np.asarray(r.w_history).size) for r in stage_results))
            vmec2000_ctrl = True
            stage_prev_fsq = prev_stage_fsq if bool(stage_transition_heuristic) else None
            stage_light_history = True if accelerated_mode else None
            stage_resume_state_mode = "minimal" if accelerated_mode else None
            stage_fsq_total_target = (
                _accelerated_fsq_total_target_from_ftol(float(ftol_i)) if accelerated_mode else None
            )
            solve_kwargs = dict(
                indata=indata,
                signgs=signgs,
                ftol=float(ftol_i),
                max_iter=int(niter_i),
                step_size=float(step_size_val),
                include_constraint_force=True,
                apply_m1_constraints=True,
                precond_radial_alpha=0.5,
                precond_lambda_alpha=0.5,
                mode_diag_exponent=0.0,
                auto_flip_force=False,
                divide_by_scalxc_for_update=False,
                lambda_update_scale=1.0,
                enforce_vmec_lambda_axis=True,
                vmec2000_control=vmec2000_ctrl,
                strict_update=True,
                backtracking=False,
                reference_mode=False,
                use_restart_triggers=True if use_restart_triggers is None else bool(use_restart_triggers),
                vmecpp_restart=bool(vmecpp_restart),
                use_direct_fallback=False,
                stage_prev_fsq=stage_prev_fsq,
                stage_transition_factor=float(stage_transition_factor),
                stage_transition_scale=float(stage_transition_scale),
                resume_state=resume_state_stage,
                verbose=bool(verbose),
                verbose_vmec2000_table=bool(verbose),
                use_scan=bool(scan_mode),
                jit_warmup_iters=int(jit_warmup_iters),
                jit_precompile=bool(jit_precompile_eff),
                scan_minimal_default=scan_minimal_default,
                light_history=stage_light_history,
                resume_state_mode=stage_resume_state_mode,
                fsq_total_target=stage_fsq_total_target,
            )
            dynamic_scan_default = "1" if bool(cfg.lasym) else "0"
            dynamic_scan_env = os.getenv("VMEC_JAX_DYNAMIC_SCAN", dynamic_scan_default).strip().lower()
            dynamic_scan = dynamic_scan_env not in ("", "0", "false", "no")
            if (
                (not accelerated_mode)
                and
                dynamic_scan
                and bool(performance_mode)
                and bool(scan_mode)
                and bool(vmec2000_ctrl)
                and int(niter_i) > 1
            ):
                pre_iters, timed_probe, probe_backend = _dynamic_scan_probe_settings(int(niter_i))
                if pre_iters > 0:
                    fsq_tol_env = os.getenv("VMEC_JAX_DYNAMIC_SCAN_FSQ_RTOL", "1e-6").strip()
                    try:
                        fsq_tol = float(fsq_tol_env)
                    except Exception:
                        fsq_tol = 1e-6
                    hist_atol_env = os.getenv("VMEC_JAX_DYNAMIC_SCAN_ATOL", "1e-12").strip()
                    try:
                        hist_atol = float(hist_atol_env)
                    except Exception:
                        hist_atol = 1e-12
                    pre_kwargs = dict(solve_kwargs)
                    pre_kwargs.update(
                        {
                            "max_iter": int(pre_iters),
                            "verbose": False,
                            "verbose_vmec2000_table": False,
                            "jit_warmup_iters": 0,
                            "jit_precompile": False,
                            # Keep full histories in the probe so we can compare
                            # the warmed scan/non-scan traces, not just a single
                            # terminal residual scalar.
                            "scan_minimal_default": False,
                        }
                    )

                    def _history_relerr(lhs_hist, rhs_hist) -> float:
                        lhs_hist = np.asarray(lhs_hist)
                        rhs_hist = np.asarray(rhs_hist)
                        if lhs_hist.shape != rhs_hist.shape:
                            return float("inf")
                        diff = np.max(np.abs(lhs_hist - rhs_hist))
                        scale = max(float(np.max(np.abs(rhs_hist))), 1e-30)
                        return float(diff / scale)

                    def _histories_match(lhs, rhs) -> bool:
                        keys = ("w_history", "fsqr2_history", "fsqz2_history", "fsql2_history")
                        for key in keys:
                            lhs_hist = np.asarray(getattr(lhs, key))
                            rhs_hist = np.asarray(getattr(rhs, key))
                            if lhs_hist.shape != rhs_hist.shape:
                                return False
                            if not np.allclose(lhs_hist, rhs_hist, rtol=float(fsq_tol), atol=float(hist_atol)):
                                return False
                        return True

                    def _run_pref(*, use_scan_flag: bool):
                        kwargs = dict(pre_kwargs)
                        kwargs["use_scan"] = bool(use_scan_flag)
                        kwargs["resume_state"] = deepcopy(resume_state_stage)
                        state_probe = deepcopy(state_stage_start)
                        if not bool(jit_forces_base):
                            try:
                                import jax
                                with jax.disable_jit():
                                    return solve_fixed_boundary_residual_iter(
                                        state_probe,
                                        static_i,
                                        jit_forces=False,
                                        **kwargs,
                                    )
                            except Exception:
                                return solve_fixed_boundary_residual_iter(
                                    state_probe,
                                    static_i,
                                    jit_forces=False,
                                    **kwargs,
                                )
                        return solve_fixed_boundary_residual_iter(
                            state_probe,
                            static_i,
                            jit_forces=True,
                            **kwargs,
                        )

                    if timed_probe:
                        # Warm both variants before timing so the selector compares
                        # steady-state iteration cost rather than one-off compile cost.
                        _ = _run_pref(use_scan_flag=False)
                        _ = _run_pref(use_scan_flag=True)

                        t0 = time.perf_counter()
                        res_pref_noscan = _run_pref(use_scan_flag=False)
                        t_noscan = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        res_pref_scan = _run_pref(use_scan_flag=True)
                        t_scan = time.perf_counter() - t0
                    else:
                        t_noscan = None
                        t_scan = None
                        res_pref_scan = _run_pref(use_scan_flag=True)
                        res_pref_noscan = _run_pref(use_scan_flag=False)

                    fsq_ok = _histories_match(res_pref_scan, res_pref_noscan)
                    choose_scan = bool(fsq_ok) and ((not timed_probe) or (t_scan < t_noscan))
                    scan_mode = bool(choose_scan)
                    solve_kwargs["use_scan"] = bool(scan_mode)
                    if bool(verbose):
                        if not bool(fsq_ok):
                            print(
                                "[vmec_jax] dynamic scan probe mismatch: "
                                f"w={_history_relerr(res_pref_scan.w_history, res_pref_noscan.w_history):.3e} "
                                f"fsqr={_history_relerr(res_pref_scan.fsqr2_history, res_pref_noscan.fsqr2_history):.3e} "
                                f"fsqz={_history_relerr(res_pref_scan.fsqz2_history, res_pref_noscan.fsqz2_history):.3e} "
                                f"fsql={_history_relerr(res_pref_scan.fsql2_history, res_pref_noscan.fsql2_history):.3e}",
                                flush=True,
                            )
                        if timed_probe:
                            print(
                                "[vmec_jax] dynamic scan selection: "
                                f"backend={probe_backend} scan={t_scan:.3f}s noscan={t_noscan:.3f}s "
                                f"fsq_ok={fsq_ok} -> use_scan={scan_mode}",
                                flush=True,
                            )
                        else:
                            print(
                                "[vmec_jax] dynamic scan parity probe: "
                                f"backend={probe_backend} iters={pre_iters} "
                                f"fsq_ok={fsq_ok} -> use_scan={scan_mode}",
                                flush=True,
                            )
            if bool(precompile_stages) and bool(jit_forces_eff):
                try:
                    precompile_kwargs = dict(solve_kwargs)
                    precompile_kwargs.update(
                        {
                            "precompile_only": True,
                            "verbose": False,
                            "verbose_vmec2000_table": False,
                            "jit_warmup_iters": 0,
                            "jit_precompile": True,
                            "max_iter": 1,
                        }
                    )
                    solve_fixed_boundary_residual_iter(
                        state,
                        static_i,
                        jit_forces=True,
                        **precompile_kwargs,
                    )
                except Exception:
                    pass
            if not bool(jit_forces_eff):
                try:
                    import jax
                    with jax.disable_jit():
                        res_i = solve_fixed_boundary_residual_iter(
                            state,
                            static_i,
                            jit_forces=False,
                            **solve_kwargs,
                        )
                except Exception:
                    res_i = solve_fixed_boundary_residual_iter(
                        state,
                        static_i,
                        jit_forces=False,
                        **solve_kwargs,
                    )
            else:
                res_i = solve_fixed_boundary_residual_iter(
                    state,
                    static_i,
                    jit_forces=True,
                    **solve_kwargs,
                )
            # Auto-fast fallback: if scan hits a bad-Jacobian path, rerun the stage
            # in the parity-safe non-scan mode.
            if (not accelerated_mode) and bool(performance_mode) and bool(scan_mode):
                try:
                    if bool(res_i.diagnostics.get("vmec2000_scan", False)) and bool(res_i.diagnostics.get("abort_scan", False)):
                        if bool(verbose):
                            print(
                                "[vmec_jax] scan abort detected; rerunning stage in parity mode.",
                                flush=True,
                            )
                        solve_kwargs_fallback = dict(solve_kwargs)
                        solve_kwargs_fallback.update(
                            {
                                "use_scan": False,
                                "resume_state": resume_state_stage,
                                "jit_warmup_iters": int(jit_warmup_noscan),
                                "jit_precompile": bool(jit_precompile_noscan),
                            }
                        )
                        if not bool(jit_forces_base):
                            try:
                                import jax
                                with jax.disable_jit():
                                    res_i = solve_fixed_boundary_residual_iter(
                                        state_stage_start,
                                        static_i,
                                        jit_forces=False,
                                        **solve_kwargs_fallback,
                                    )
                            except Exception:
                                res_i = solve_fixed_boundary_residual_iter(
                                    state_stage_start,
                                    static_i,
                                    jit_forces=False,
                                    **solve_kwargs_fallback,
                                )
                        else:
                            res_i = solve_fixed_boundary_residual_iter(
                                state_stage_start,
                                static_i,
                                jit_forces=True,
                                **solve_kwargs_fallback,
                            )
                except Exception:
                    pass
            stage_results.append(res_i)
            try:
                w_hist = np.asarray(res_i.w_history)
                prev_stage_fsq = float(w_hist[-1]) if w_hist.size else None
            except Exception:
                prev_stage_fsq = None
            if multigrid_resume and i < (nstep - 1):
                resume_state_stage = _sanitize_resume_state_for_stage(res_i.diagnostics.get("resume_state"))
            state = stage_results[-1].state
            static_prev = static_i
            ftol_last = float(ftol_i)
            step_size_last = float(step_size_val)

        # Merge per-stage histories into one VMEC-style trace object.
        def _cat(attr: str) -> np.ndarray:
            parts = [np.asarray(getattr(r, attr)) for r in stage_results if getattr(r, attr) is not None]
            return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=float)

        diag = dict(stage_results[-1].diagnostics)
        diag["solver_mode"] = str(solver_mode_eff)
        diag["accelerated_mode"] = bool(accelerated_mode)
        diag["accelerated_scan"] = bool(accelerated_mode) and bool(diag.get("use_scan", False))
        diag["multigrid_user_provided"] = bool(multigrid_user_provided)
        diag["accelerated_single_grid_default"] = bool(accelerated_single_grid_default)
        diag["multigrid_ns_stages"] = np.asarray(ns_stages, dtype=int)
        diag["multigrid_niter_stages"] = np.asarray(niter_stages, dtype=int)
        diag["multigrid_ftol_stages"] = np.asarray(ftol_stages, dtype=float)
        diag["multigrid_stage_offsets"] = np.asarray(stage_offsets, dtype=int)

        # Concatenate the common history keys that are useful for parity debugging.
        for k in (
            "step_status_history",
            "restart_reason_history",
            "pre_restart_reason_history",
            "time_step_history",
            "res0_history",
            "res1_history",
            "fsq_prev_history",
            "bad_growth_streak_history",
            "iter1_history",
            "bcovar_update_history",
            "include_edge_history",
            "zero_m1_history",
            "dt_eff_history",
            "update_rms_history",
            "w_curr_history",
            "w_try_history",
            "w_try_ratio_history",
            "restart_path_history",
            "min_tau_history",
            "max_tau_history",
            "bad_jacobian_history",
            "fsq1_history",
            "fsqr1_history",
            "fsqz1_history",
            "fsql1_history",
            "r00_history",
            "z00_history",
            "wb_history",
            "wp_history",
            "w_vmec_history",
            "rz_norm_history",
            "f_norm1_history",
            "gcr2_p_history",
            "gcz2_p_history",
            "gcl2_p_history",
        ):
            if any(k in r.diagnostics for r in stage_results):
                diag[k] = np.concatenate(
                    [np.asarray(r.diagnostics.get(k, np.zeros((0,), dtype=float))) for r in stage_results]
                )

        res = SolveVmecResidualResult(
            state=state,
            n_iter=int(sum(int(r.n_iter) + 1 for r in stage_results) - 1),
            w_history=_cat("w_history"),
            fsqr2_history=_cat("fsqr2_history"),
            fsqz2_history=_cat("fsqz2_history"),
            fsql2_history=_cat("fsql2_history"),
            grad_rms_history=_cat("grad_rms_history"),
            step_history=_cat("step_history"),
            diagnostics=diag,
        )
        # Optional scan corrector: run a single non-scan VMEC2000 step to
        # re-anchor the final state before writing wout outputs.
        try:
            use_scan_any = any(bool(r.diagnostics.get("vmec2000_scan", False)) for r in stage_results)
        except Exception:
            use_scan_any = False
        if accelerated_mode and scan_wout_corrector is None:
            scan_wout_corrector = False
        if scan_wout_corrector is None:
            scan_wout_env = os.getenv("VMEC_JAX_SCAN_WOUT_CORRECTOR", "0").strip().lower()
            scan_wout_corrector = scan_wout_env not in ("", "0", "false", "no")
        if use_scan_any and bool(scan_wout_corrector):
            try:
                resume_state_corr = res.diagnostics.get("resume_state", None)
                static_corr = static_prev if static_prev is not None else _build_static_cfg(cfg)
                ftol_corr = float(ftol_last) if ftol_last is not None else float(indata.get_float("FTOL", 1e-13))
                step_corr = float(step_size_last) if step_size_last is not None else 1.0
                corr_kwargs = dict(
                    indata=indata,
                    signgs=signgs,
                    ftol=ftol_corr,
                    max_iter=1,
                    step_size=step_corr,
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
                    use_restart_triggers=True if use_restart_triggers is None else bool(use_restart_triggers),
                    vmecpp_restart=bool(vmecpp_restart),
                    stage_prev_fsq=None,
                    stage_transition_factor=float(stage_transition_factor),
                    stage_transition_scale=float(stage_transition_scale),
                    use_direct_fallback=False,
                    resume_state=resume_state_corr,
                    verbose=False,
                    verbose_vmec2000_table=False,
                    jit_precompile=False,
                    jit_warmup_iters=0,
                    use_scan=False,
                    scan_minimal_default=scan_minimal_default,
                    light_history=True if accelerated_mode else None,
                    resume_state_mode="minimal" if accelerated_mode else None,
                    fsq_total_target=(
                        _accelerated_fsq_total_target_from_ftol(float(ftol_corr)) if accelerated_mode else None
                    ),
                )
                res_corr = solve_fixed_boundary_residual_iter(
                    res.state,
                    static_corr,
                    jit_forces=_resolve_jit_forces(jit_forces, static_corr, 1),
                    **corr_kwargs,
                )
                diag = dict(res.diagnostics)
                diag["scan_wout_corrector"] = True
                diag["scan_wout_corrector_iters"] = int(res_corr.n_iter)
                res = SolveVmecResidualResult(
                    state=res_corr.state,
                    n_iter=res.n_iter,
                    w_history=res.w_history,
                    fsqr2_history=res.fsqr2_history,
                    fsqz2_history=res.fsqz2_history,
                    fsql2_history=res.fsql2_history,
                    grad_rms_history=res.grad_rms_history,
                    step_history=res.step_history,
                    diagnostics=diag,
                )
            except Exception:
                pass
        static = _build_static_cfg(cfg)
        if verbose and solver == "vmec2000_iter":
            converged = bool(res.diagnostics.get("converged", False))
            if not converged:
                print(" Try increasing NITER or PRE_NITER if the preconditioner is on.", flush=True)
            print("", flush=True)
            print(" EXECUTION TERMINATED NORMALLY", flush=True)
            print("", flush=True)
            case_name = Path(input_path).name
            if case_name.startswith("input."):
                case_name = case_name.split("input.", 1)[-1]
            print(f" FILE : {case_name}", flush=True)
            ijacob = int(res.diagnostics.get("ijacob", 0))
            print(f" NUMBER OF JACOBIAN RESETS = {ijacob:4d}", flush=True)
            total_time = max(0.0, time.perf_counter() - t_start)
            print("", flush=True)
            print(f"    TOTAL COMPUTATIONAL TIME (SEC)         {total_time:8.2f}", flush=True)
            print("    TIME TO INPUT/OUTPUT                   0.00", flush=True)
            print("       READ IN DATA                        0.00", flush=True)
            print("       WRITE OUT DATA TO WOUT              0.00", flush=True)
            print(f"    TIME IN FUNCT3D                        {total_time:8.2f}", flush=True)
            print("       BCOVAR FIELDS                       0.00", flush=True)
            print("       FOURIER TRANSFORM                   0.00", flush=True)
            print("       INVERSE FOURIER TRANSFORM           0.00", flush=True)
            print("       FORCES AND SYMMETRIZE               0.00", flush=True)
            print("       RESIDUE                             0.00", flush=True)
            print("       EQFORCE                             0.00", flush=True)
            print("", flush=True)
            print(" NO. OF PROCS:     1", flush=True)
            print(" PARVMEC     :     T", flush=True)
            print(" LPRECOND    :     F", flush=True)
            print(" LV3FITCALL  :     F", flush=True)
    else:
        raise ValueError(
            f"Unknown solver: {solver!r} (expected 'gd', 'lbfgs', 'vmec_lbfgs', 'vmec_gn', or 'vmec2000_iter')"
        )

    if verbose and solver != "vmec2000_iter":
        n_iter = int(getattr(res, "n_iter", -1))
        w_final = float(res.w_history[-1]) if getattr(res, "w_history", None) is not None else float("nan")
        if getattr(res, "grad_rms_history", None) is not None and len(res.grad_rms_history) > 0:
            grad_final = float(res.grad_rms_history[-1])
        else:
            grad_final = float("nan")
        print(f"[vmec_jax] finished: n_iter={n_iter} w={w_final:.8e} grad_rms={grad_final:.3e}")

    if flux is None or prof is None or pressure is None:
        if static is None:
            static = _build_static_cfg(cfg)
        flux, prof, pressure = _profiles_from_static(static)
    flux, prof = _final_flux_profiles_from_state(
        indata=indata,
        static_in=static,
        state=res.state,
        signgs=signgs,
        flux_local=flux,
        prof_local=prof,
        pressure_local=pressure,
    )

    run_out = FixedBoundaryRun(
        cfg=cfg,
        indata=indata,
        static=static,
        state=res.state,
        result=res,
        flux=flux,
        profiles=prof,
        signgs=signgs,
    )
    cli_initial_policy = "multigrid" if bool(multigrid) and (len(ns_stages) > 1) else "single_grid"
    return _maybe_finish_cli_fixed_boundary_run(
        run_out,
        initial_policy=cli_initial_policy,
        enabled=bool(cli_fixed_boundary_finish_enabled),
    )

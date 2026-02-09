"""High-level helpers for VMEC driver scripts.

These functions provide a thin, convenient layer over the core modules so
simple scripts can be written with minimal boilerplate, while still allowing
power users to drop down to lower-level APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from .boundary import boundary_from_indata
from .config import VMECConfig, load_config
from .energy import flux_profiles_from_indata
from .field import signgs_from_sqrtg
from .geom import eval_geom
from .init_guess import initial_guess_from_boundary
from .multigrid import interp_vmec_state
from .profiles import eval_profiles
from .solve import solve_fixed_boundary_gd, solve_fixed_boundary_lbfgs
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


def residual_scalars_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
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
    scal = vmec_fsq_from_tomnsps_dynamic(frzl=frzl, norms=norms, lconm1=bool(getattr(static.cfg, "lconm1", True)))
    return float(scal.fsqr), float(scal.fsqz), float(scal.fsql)


def write_wout_from_fixed_boundary_run(
    path: str | Path,
    run: FixedBoundaryRun,
    *,
    include_fsq: bool = True,
):
    """Write a minimal VMEC-style `wout_*.nc` from a fixed-boundary run."""
    from .wout import write_wout, wout_minimal_from_fixed_boundary

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if include_fsq:
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
    )
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
    static = build_static(cfg, grid=grid)
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


def run_fixed_boundary(
    input_path: str | Path,
    *,
    solver: str = "gd",
    max_iter: int = 20,
    step_size: float | object = _STEP_SIZE_SENTINEL,
    history_size: int = 10,
    # vmec_gn tuning (Gauss-Newton on VMEC residual vector)
    gn_damping: float = 1e-3,
    gn_cg_tol: float = 1e-6,
    gn_cg_maxiter: int = 80,
    use_initial_guess: bool = False,
    vmec_project: bool = True,
    use_restart_triggers: bool | None = None,
    use_direct_fallback: bool | None = None,
    multigrid: bool | None = None,
    multigrid_use_input_niter: bool = False,
    verbose: bool = True,
    grid=None,
    ns_override: int | None = None,
):
    """Run a fixed-boundary vmec_jax solve with minimal boilerplate.

    Parameters
    ----------
    solver:
        ``"gd"`` (gradient descent), ``"lbfgs"``, ``"vmec_lbfgs"``,
        ``"vmec_gn"`` (VMEC residual objective), or
        ``"vmec2000_iter"`` (VMEC-style multigrid iteration).
    use_initial_guess:
        If True, skip the solve and return the initialized state.
    ns_override:
        If provided, overrides the radial resolution (ns) used to build the state.
    vmec_project:
        If True (default), re-project the initial guess through the VMEC
        internal grid/weights before returning or solving.
    verbose:
        If True (default), print VMEC-style iteration progress and a summary.
    """
    cfg, indata = load_config(str(input_path))
    if ns_override is not None:
        cfg = replace(cfg, ns=int(ns_override))
    solver_lower = str(solver).lower()
    if grid is None and solver_lower in ("vmec_lbfgs", "vmec_gn", "vmec2000_iter"):
        from .vmec_tomnsp import vmec_angle_grid

        grid = vmec_angle_grid(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(cfg.nfp),
            lasym=bool(cfg.lasym),
        )
    multigrid_use_input_niter = bool(multigrid_use_input_niter)
    if multigrid is None:
        multigrid = solver_lower == "vmec2000_iter"
    multigrid = bool(multigrid) and (ns_override is None)

    # Build the initial state on either the final grid (single-grid solvers and
    # use_initial_guess) or on the first multigrid stage for VMEC-style solves.
    ns_stages = [int(cfg.ns)]
    if multigrid:
        ns_array = indata.get("NS_ARRAY", None)
        if isinstance(ns_array, list) and ns_array:
            ns_stages = [int(v) for v in ns_array]
        elif isinstance(ns_array, (tuple, np.ndarray)) and len(ns_array):
            ns_stages = [int(v) for v in ns_array]

    # Stage-0 (coarsest) static + initial guess drives signgs selection.
    cfg0 = replace(cfg, ns=int(ns_stages[0]))
    static0 = build_static(cfg0, grid=grid)
    bdy = boundary_from_indata(indata, static0.modes)
    st0_coarse = initial_guess_from_boundary(static0, bdy, indata, vmec_project=vmec_project)

    g0 = eval_geom(st0_coarse, static0)
    signgs = signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    static = build_static(cfg, grid=grid)

    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    # VMEC evaluates pressure/iota/current profiles on the radial half mesh.
    if int(cfg.ns) < 2:
        s_half = np.asarray(static.s)
    else:
        s_full = np.asarray(static.s)
        s_half = np.concatenate([s_full[:1], 0.5 * (s_full[1:] + s_full[:-1])], axis=0)
    prof = eval_profiles(indata, s_half)
    pressure = prof.get("pressure", np.zeros_like(np.asarray(static.s)))
    gamma = indata.get_float("GAMMA", 0.0)

    if step_size is _STEP_SIZE_SENTINEL or step_size is None:
        if solver_lower in ("vmec2000_iter",):
            step_size_val = float(indata.get_float("DELT", 5e-3))
        else:
            step_size_val = 5e-3
    else:
        step_size_val = float(step_size)

    if verbose:
        mode = "initial guess" if use_initial_guess else f"{solver} solve"
        print(f"[vmec_jax] fixed-boundary run ({mode})")
        print(f"[vmec_jax] input={input_path}")
        print(f"[vmec_jax] ns={cfg.ns} mpol={cfg.mpol} ntor={cfg.ntor} nfp={cfg.nfp}")
        if not use_initial_guess:
            print(f"[vmec_jax] max_iter={max_iter} step_size={step_size_val} history_size={history_size}")

    if use_initial_guess:
        static = build_static(cfg, grid=grid)
        st0 = initial_guess_from_boundary(static, bdy, indata, vmec_project=vmec_project)
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

    solver = solver_lower
    if solver == "gd":
        static = build_static(cfg, grid=grid)
        st0 = initial_guess_from_boundary(static, bdy, indata, vmec_project=vmec_project)
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
        static = build_static(cfg, grid=grid)
        st0 = initial_guess_from_boundary(static, bdy, indata, vmec_project=vmec_project)
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
        static = build_static(cfg, grid=grid)
        st0 = initial_guess_from_boundary(static, bdy, indata, vmec_project=vmec_project)

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
        static = build_static(cfg, grid=grid)
        st0 = initial_guess_from_boundary(static, bdy, indata, vmec_project=vmec_project)

        res = solve_fixed_boundary_gn_vmec_residual(
            st0,
            static,
            indata=indata,
            signgs=signgs,
            max_iter=int(max_iter),
            step_size=float(step_size_val),
            damping=float(gn_damping),
            cg_tol=float(gn_cg_tol),
            cg_maxiter=int(gn_cg_maxiter),
            jit_kernels=True,
            verbose=bool(verbose),
        )
    elif solver == "vmec2000_iter":
        from .solve import SolveVmecResidualResult, solve_fixed_boundary_residual_iter

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
        if multigrid_use_input_niter:
            niter_array = indata.get("NITER_ARRAY", None)
            ftol_array = indata.get("FTOL_ARRAY", None)
            niter_stages = (
                [int(v) for v in niter_array] if isinstance(niter_array, list) and len(niter_array) == nstep else None
            )
            ftol_stages = (
                [float(v) for v in ftol_array] if isinstance(ftol_array, list) and len(ftol_array) == nstep else None
            )
            if niter_stages is None:
                niter_stages = _distribute_iters(iters=int(max_iter), nstep=int(nstep))
            else:
                # Respect the caller's `max_iter` as a total budget, but keep at
                # least 1 iteration per stage when possible (so staging still
                # happens in short debugging runs).
                budget = int(max_iter)
                if budget < nstep:
                    # Too few iterations to meaningfully stage; collapse to the
                    # final grid only.
                    ns_stages = [int(ns_stages[-1])]
                    nstep = 1
                    niter_stages = [int(max(budget, 1))]
                    if ftol_stages is not None:
                        ftol_stages = [float(ftol_stages[-1])]
                else:
                    base = [1] * nstep
                    remaining = budget - nstep
                    caps = [max(0, int(n) - 1) for n in niter_stages]
                    out = base[:]
                    for i in range(nstep):
                        if remaining <= 0:
                            break
                        take = min(caps[i], remaining)
                        out[i] += take
                        remaining -= take
                    if remaining > 0:
                        out[-1] += remaining
                    niter_stages = out
            if ftol_stages is None:
                ftol_stages = [float(indata.get_float("FTOL", 1e-10))] * nstep
        else:
            niter_stages = _distribute_iters(iters=int(max_iter), nstep=int(nstep))
            ftol_stages = [float(indata.get_float("FTOL", 1e-10))] * nstep

        # Run coarse -> fine stages with VMEC `interp.f` interpolation.
        stage_results: list[SolveVmecResidualResult] = []
        stage_offsets: list[int] = []

        state = st0_coarse
        static_prev = static0
        for i, (ns_i, niter_i, ftol_i) in enumerate(zip(ns_stages, niter_stages, ftol_stages)):
            cfg_i = replace(cfg, ns=int(ns_i))
            static_i = build_static(cfg_i, grid=grid)
            if i > 0:
                state = interp_vmec_state(state, m=static_prev.modes.m, ns_new=int(ns_i))
                static_prev = static_i

            if verbose:
                print(f"[vmec_jax] multigrid stage {i+1}/{nstep}: ns={ns_i} niter={niter_i} ftol={ftol_i:.2e}")

            stage_offsets.append(sum(int(np.asarray(r.w_history).size) for r in stage_results))
            stage_results.append(
                solve_fixed_boundary_residual_iter(
                    state,
                    static_i,
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
                    auto_flip_force=True,
                    divide_by_scalxc_for_update=False,
                    lambda_update_scale=float(2.0 * np.pi * float(signgs)),
                    enforce_vmec_lambda_axis=True,
                    vmec2000_control=True,
                    strict_update=True,
                    backtracking=False,
                    reference_mode=False,
                    use_restart_triggers=bool(use_restart_triggers) if use_restart_triggers is not None else False,
                    use_direct_fallback=False,
                    verbose=bool(verbose),
                )
            )
            state = stage_results[-1].state
            static_prev = static_i

        # Merge per-stage histories into one VMEC-style trace object.
        def _cat(attr: str) -> np.ndarray:
            parts = [np.asarray(getattr(r, attr)) for r in stage_results if getattr(r, attr) is not None]
            return np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=float)

        diag = dict(stage_results[-1].diagnostics)
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
        static = build_static(cfg, grid=grid)
    else:
        raise ValueError(
            f"Unknown solver: {solver!r} (expected 'gd', 'lbfgs', 'vmec_lbfgs', 'vmec_gn', or 'vmec2000_iter')"
        )

    if verbose:
        n_iter = int(getattr(res, "n_iter", -1))
        w_final = float(res.w_history[-1]) if getattr(res, "w_history", None) is not None else float("nan")
        grad_final = float(res.grad_rms_history[-1]) if getattr(res, "grad_rms_history", None) is not None else float("nan")
        print(f"[vmec_jax] finished: n_iter={n_iter} w={w_final:.8e} grad_rms={grad_final:.3e}")

    return FixedBoundaryRun(
        cfg=cfg,
        indata=indata,
        static=static,
        state=res.state,
        result=res,
        flux=flux,
        profiles=prof,
        signgs=signgs,
    )

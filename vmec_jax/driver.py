"""High-level helpers for VMEC driver scripts.

These functions provide a thin, convenient layer over the core modules so
simple scripts can be written with minimal boilerplate, while still allowing
power users to drop down to lower-level APIs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import os
from typing import Optional

import numpy as np

from .boundary import boundary_from_indata
from .config import VMECConfig, load_config
from .energy import flux_profiles_from_indata
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
        fsqr = fsqz = fsql = None
        res = getattr(run, "result", None)
        if res is not None:
            fsqr_hist = getattr(res, "fsqr2_history", None)
            fsqz_hist = getattr(res, "fsqz2_history", None)
            fsql_hist = getattr(res, "fsql2_history", None)
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
    solver: str = "vmec2000_iter",
    max_iter: int = 10,
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
    multigrid_use_input_niter: bool = True,
    verbose: bool = True,
    jit_forces: bool | str = "auto",
    use_scan: bool = False,
    grid=None,
    ns_override: int | None = None,
    restart_state: any | None = None,
    restart_wout_path: str | Path | None = None,
    restart_solver_state: dict | None = None,
):
    def _maybe_enable_compilation_cache() -> None:
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

            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            compilation_cache.set_cache_dir(cache_dir)
            try:
                jax.config.update("jax_enable_compilation_cache", True)
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
        xc = vmec_xc_from_mn_blocks(
            rcc=blocks["rcc"],
            rss=blocks["rss"],
            zsc=blocks["zsc"],
            zcs=blocks["zcs"],
            lsc=blocks["lsc"],
            lcs=blocks["lcs"],
            cfg=static.cfg,
        )
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
    vmec_project:
        If True (default), re-project the initial guess through the VMEC
        internal grid/weights before returning or solving.
    verbose:
        If True (default), print VMEC-style iteration progress and a summary.
    jit_forces:
        If True, JIT the force kernels. If ``"auto"`` (default), disable JIT
        for very small workloads to reduce first-iteration latency.
    """
    # Default to 64-bit for VMEC parity; users can opt out via JAX_ENABLE_X64=0.
    try:
        from ._compat import enable_x64

        enable_x64(True)
    except Exception:
        pass
    _maybe_enable_compilation_cache()
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
    elif ns_override is not None:
        cfg = replace(cfg, ns=int(ns_override))
    solver_lower = str(solver).lower()
    # VMEC uses `guess_axis` to build a usable initial axis when the input
    # axis arrays are missing/zero. Keep this enabled for vmec2000_iter parity.
    axis_infer_missing = True
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
    if restart_state_eff is not None:
        multigrid = False
    if restart_solver_state is not None:
        multigrid = False
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

    # Precompute boundary coefficients without triggering JAX initialization.
    boundary_coeffs = None
    if restart_state_eff is None:
        from .modes import vmec_mode_table

        boundary_modes = vmec_mode_table(cfg.mpol, cfg.ntor)
        boundary_coeffs = boundary_from_indata(indata, boundary_modes)

    # VMEC readin.f hard-codes signgs = -1 (then flips theta if needed).
    # For VMEC2000-iter parity, ignore input SIGNGS and match VMEC behavior.
    if solver_lower == "vmec2000_iter":
        signgs = -1
    else:
        signgs = int(indata.get_int("SIGNGS", -1))
        if signgs not in (-1, 1):
            signgs = -1

    gamma = indata.get_float("GAMMA", 0.0)
    static = None
    static_final = None
    bdy = None
    flux = None
    prof = None
    pressure = None

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
            static = build_static(cfg, grid=grid)
        if bdy is None:
            bdy = boundary_from_indata(indata, static.modes)
        if flux is None or prof is None or pressure is None:
            flux, prof, pressure = _profiles_from_static(static)

    if step_size is _STEP_SIZE_SENTINEL or step_size is None:
        if solver_lower in ("vmec2000_iter",):
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

    if use_initial_guess:
        _ensure_static_profiles()
        if restart_state_eff is not None:
            st0 = restart_state_eff
        else:
            st0 = initial_guess_from_boundary(
                static,
                bdy,
                indata,
                vmec_project=vmec_project,
                infer_axis_if_missing=axis_infer_missing,
            )
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

    solver = solver_lower
    if solver in ("vmec2000_iter_fast", "vmec2000_scan"):
        use_scan = True
        solver = "vmec2000_iter"
    if os.getenv("VMEC_JAX_USE_SCAN", "") not in ("", "0"):
        use_scan = True
    if solver == "gd":
        _ensure_static_profiles()
        st0 = restart_state_eff if restart_state_eff is not None else initial_guess_from_boundary(
            static,
            bdy,
            indata,
            vmec_project=vmec_project,
            infer_axis_if_missing=axis_infer_missing,
        )
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
        st0 = restart_state_eff if restart_state_eff is not None else initial_guess_from_boundary(
            static,
            bdy,
            indata,
            vmec_project=vmec_project,
            infer_axis_if_missing=axis_infer_missing,
        )
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
        st0 = restart_state_eff if restart_state_eff is not None else initial_guess_from_boundary(
            static,
            bdy,
            indata,
            vmec_project=vmec_project,
            infer_axis_if_missing=axis_infer_missing,
        )

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
        st0 = restart_state_eff if restart_state_eff is not None else initial_guess_from_boundary(
            static,
            bdy,
            indata,
            vmec_project=vmec_project,
            infer_axis_if_missing=axis_infer_missing,
        )

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

                    # When the total budget is smaller than the sum of the
                    # input's NITER_ARRAY, prioritize iterations on the final
                    # (finest) grid. This keeps short debugging runs focused
                    # on the physically relevant resolution instead of spending
                    # nearly the entire budget on coarse stages.
                    for i in range(nstep - 1, -1, -1):
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
        from .modes import vmec_mode_table

        header_modes = vmec_mode_table(cfg.mpol, cfg.ntor)
        nmodes_header = int(np.asarray(header_modes.m).size)

        state = restart_state_eff
        static_prev = None
        static_final = None
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

        for i, (ns_i, niter_i, ftol_i) in enumerate(zip(ns_stages, niter_stages, ftol_stages)):
            if verbose:
                print(
                    f"  NS = {int(ns_i):4d} NO. FOURIER MODES = {nmodes_header:4d} "
                    f"FTOLV = {float(ftol_i):10.3E} NITER = {int(niter_i):6d}",
                    flush=True,
                )
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
            static_i = build_static(cfg_i, grid=grid)
            scan_mode = bool(use_scan)
            jit_forces_eff = _resolve_jit_forces(jit_forces, static_i, int(niter_i))
            jit_warmup_iters = 0
            if bool(jit_forces_eff) and (not bool(scan_mode)):
                try:
                    jit_warmup_iters = max(0, int(os.getenv("VMEC_JAX_JIT_WARMUP_ITERS", "2")))
                except Exception:
                    jit_warmup_iters = 2
            if i == 0:
                if state is None:
                    if boundary_coeffs is None:
                        raise ValueError("boundary_coeffs missing; cannot build initial guess")
                    if jit_warmup_iters > 0:
                        try:
                            import jax
                            with jax.disable_jit():
                                state = initial_guess_from_boundary(
                                    static_i,
                                    boundary_coeffs,
                                    indata,
                                    vmec_project=vmec_project,
                                    infer_axis_if_missing=axis_infer_missing,
                                )
                        except Exception:
                            state = initial_guess_from_boundary(
                                static_i,
                                boundary_coeffs,
                                indata,
                                vmec_project=vmec_project,
                                infer_axis_if_missing=axis_infer_missing,
                            )
                    else:
                        state = initial_guess_from_boundary(
                            static_i,
                            boundary_coeffs,
                            indata,
                            vmec_project=vmec_project,
                            infer_axis_if_missing=axis_infer_missing,
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
            static_prev = static_i
            static_final = static_i

            stage_offsets.append(sum(int(np.asarray(r.w_history).size) for r in stage_results))
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
                vmec2000_control=not scan_mode,
                strict_update=False if scan_mode else True,
                backtracking=False,
                reference_mode=False,
                use_restart_triggers=False if scan_mode else (True if use_restart_triggers is None else bool(use_restart_triggers)),
                use_direct_fallback=False,
                resume_state=restart_solver_state,
                verbose=bool(verbose),
                verbose_vmec2000_table=bool(verbose),
                use_scan=bool(use_scan),
                jit_warmup_iters=int(jit_warmup_iters),
            )
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
            stage_results.append(res_i)
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
        static = build_static(cfg, grid=grid)
    else:
        raise ValueError(
            f"Unknown solver: {solver!r} (expected 'gd', 'lbfgs', 'vmec_lbfgs', 'vmec_gn', or 'vmec2000_iter')"
        )

    if verbose and solver_lower != "vmec2000_iter":
        n_iter = int(getattr(res, "n_iter", -1))
        w_final = float(res.w_history[-1]) if getattr(res, "w_history", None) is not None else float("nan")
        grad_final = float(res.grad_rms_history[-1]) if getattr(res, "grad_rms_history", None) is not None else float("nan")
        print(f"[vmec_jax] finished: n_iter={n_iter} w={w_final:.8e} grad_rms={grad_final:.3e}")

    if flux is None or prof is None or pressure is None:
        if static is None:
            static = build_static(cfg, grid=grid)
        flux, prof, pressure = _profiles_from_static(static)

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

"""Initial magnetic-axis reset helpers for VMEC residual solves."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Callable, NamedTuple

import numpy as np

from ...._compat import jnp
from ....state import VMECState


class InitialAxisResetDecision(NamedTuple):
    """Pure control decision for VMEC-style initial magnetic-axis resets."""

    bad_jacobian: bool
    force_reset: bool
    reset: bool


class InitialAxisResetRuntimeDecision(NamedTuple):
    """Pure in-loop decision for the VMEC2000 first-step axis retry."""

    bad_jacobian: bool
    huge_initial_forces: bool
    force_reset: bool
    reset: bool


class InitialAxisResetEvaluation(NamedTuple):
    """Initial force/Jacobian diagnostics and reset decision."""

    decision: InitialAxisResetDecision
    fsq_phys: float | None
    bad_jacobian_ptau: bool | None
    bad_jacobian_state: bool


class InitialAxisResetSetupResult(NamedTuple):
    """State returned after optional setup-time magnetic-axis reset."""

    state: VMECState
    axis_reset_done: bool
    ijacob: int
    state_checkpoint: VMECState
    velocities: tuple[Any, ...]
    res0: float
    res1: float
    prev_rz_fsq: float
    reset_applied: bool
    force_probe: tuple[Any, ...] | None


def initial_force_physical_fsq(*, norms: Any, gcr2: Any, gcz2: Any, gcl2: Any) -> float | None:
    """Return the physical initial residual used to gate axis reset attempts."""

    try:
        fsqr = norms.r1 * norms.fnorm * gcr2
        fsqz = norms.r1 * norms.fnorm * gcz2
        fsql = norms.fnormL * gcl2
        return float(np.asarray(fsqr + fsqz + fsql))
    except Exception:
        return None


def bad_jacobian_from_tau_range(*, min_tau: float, max_tau: float, abs_tol: float = 0.0) -> bool:
    """Return whether a Jacobian-sign change is present across a tau range."""

    tol = max(0.0, float(abs_tol))
    return bool(float(min_tau) < -tol and float(max_tau) > tol)


def bad_jacobian_ptau_from_minmax(
    *,
    ptau_min: Any | None,
    ptau_max: Any | None,
    ptau_tol: float,
    ptau_tol_rel: float,
) -> bool | None:
    """Return the bad-Jacobian decision from VMEC ptau min/max diagnostics."""

    if ptau_min is None or ptau_max is None:
        return None
    try:
        min_tau = float(np.asarray(ptau_min))
        max_tau = float(np.asarray(ptau_max))
        tau_scale = max(abs(min_tau), abs(max_tau))
        tau_tol = max(abs(float(ptau_tol)), max(float(ptau_tol_rel), 0.0) * float(tau_scale))
        return bad_jacobian_from_tau_range(min_tau=min_tau, max_tau=max_tau, abs_tol=tau_tol)
    except Exception:
        return None


def merge_axis_reset_state(*, st: VMECState, st_axis: VMECState, static, full_reset: bool) -> VMECState:
    """Return an axis-reset state, preserving non-axis coefficients unless requested."""

    if full_reset:
        return st_axis
    if getattr(static, "m_is_m0", None) is None:
        mask_m0 = jnp.asarray(np.asarray(static.modes.m, dtype=int) == 0, dtype=jnp.asarray(st.Rcos).dtype)
    else:
        mask_m0 = jnp.asarray(static.m_is_m0, dtype=jnp.asarray(st.Rcos).dtype)
    Rcos = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Rcos), jnp.asarray(st.Rcos))
    Rsin = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Rsin), jnp.asarray(st.Rsin))
    Zcos = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Zcos), jnp.asarray(st.Zcos))
    Zsin = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Zsin), jnp.asarray(st.Zsin))
    return VMECState(layout=st.layout, Rcos=Rcos, Rsin=Rsin, Zcos=Zcos, Zsin=Zsin, Lcos=st.Lcos, Lsin=st.Lsin)


def initial_axis_reset_decision(
    *,
    bad_jacobian_ptau: bool | None,
    bad_jacobian_state: bool,
    badjac_use_state: bool,
    fsq_phys: float | None,
    axis_reset_fsq_min: float,
    force_axis_reset: bool,
    axis_reset_always_3d: bool,
    lthreed: bool,
    vmec2000_control: bool = True,
    lmove_axis: bool = True,
    axis_reset_enabled: bool = True,
) -> InitialAxisResetDecision:
    """Pure control-flow gate for VMEC-style initial magnetic-axis resets."""

    if bad_jacobian_ptau is None:
        bad_jacobian = bool(bad_jacobian_state)
    elif bool(badjac_use_state):
        bad_jacobian = bool(bad_jacobian_ptau) and bool(bad_jacobian_state)
    else:
        bad_jacobian = bool(bad_jacobian_ptau)

    fsq_min = max(0.0, float(axis_reset_fsq_min))
    if bad_jacobian and fsq_min > 0.0:
        if fsq_phys is None:
            bad_jacobian = False
        else:
            fsq_val = float(fsq_phys)
            if (not np.isfinite(fsq_val)) or (fsq_val < fsq_min):
                bad_jacobian = False

    force_reset = bool(force_axis_reset) or (
        bool(vmec2000_control) and bool(lmove_axis) and bool(lthreed) and bool(axis_reset_always_3d)
    )
    return InitialAxisResetDecision(
        bool(bad_jacobian), bool(force_reset), bool(axis_reset_enabled) and (bool(bad_jacobian) or bool(force_reset))
    )


def initial_axis_reset_runtime_decision(
    *,
    bad_jacobian: bool,
    fsq_phys: float,
    axis_reset_fsq_min: float,
    force_axis_reset: bool,
    axis_reset_always_3d: bool,
    lthreed: bool,
    vmec2000_control: bool = True,
    lmove_axis: bool = True,
    axis_reset_enabled: bool = True,
) -> InitialAxisResetRuntimeDecision:
    """Return the in-loop VMEC2000 first-step axis-reset decision."""

    fsq_curr = float(fsq_phys)
    huge_initial_forces = (not np.isfinite(fsq_curr)) or (fsq_curr > 1.0e2)
    force_reset = bool(force_axis_reset) or (
        bool(vmec2000_control) and bool(lmove_axis) and bool(lthreed) and bool(axis_reset_always_3d)
    )
    bad_jacobian_next = bool(bad_jacobian)
    if (not force_reset) and float(axis_reset_fsq_min) > 0.0:
        if np.isfinite(fsq_curr) and (fsq_curr < float(axis_reset_fsq_min)):
            bad_jacobian_next = False
            huge_initial_forces = False
    reset = bool(axis_reset_enabled) and (bool(bad_jacobian_next) or bool(huge_initial_forces) or bool(force_reset))
    return InitialAxisResetRuntimeDecision(
        bool(bad_jacobian_next),
        bool(huge_initial_forces),
        bool(force_reset),
        bool(reset),
    )


def evaluate_initial_axis_reset(
    *,
    axis_reset_enabled: bool, norms: Any, gcr2: Any, gcz2: Any, gcl2: Any,
    k: Any, state: VMECState, static: Any, trig: Any, s: Any,
    badjac_use_state: bool, ptau_tol: float, ptau_tol_rel: float,
    axis_reset_fsq_min: float, force_axis_reset: bool, axis_reset_always_3d: bool,
    vmec2000_control: bool, lmove_axis: bool, debug_enabled: bool = False,
    state_check_on_missing_ptau: bool = False,
    ptau_minmax_from_k_host: Callable[[Any], tuple[Any | None, Any | None]],
    vmec_half_mesh_jacobian_from_state_func: Callable[..., Any],
) -> InitialAxisResetEvaluation:
    """Evaluate VMEC-style initial magnetic-axis reset diagnostics."""

    fsq_phys = initial_force_physical_fsq(norms=norms, gcr2=gcr2, gcz2=gcz2, gcl2=gcl2)
    fsq_floor = max(0.0, float(axis_reset_fsq_min))
    force_reset = bool(force_axis_reset) or (
        bool(vmec2000_control)
        and bool(lmove_axis)
        and bool(getattr(static.cfg, "lthreed", True))
        and bool(axis_reset_always_3d)
    )
    if (
        bool(axis_reset_enabled)
        and (not bool(debug_enabled))
        and (not force_reset)
        and fsq_floor > 0.0
        and fsq_phys is not None
    ):
        fsq_val = float(fsq_phys)
        if np.isfinite(fsq_val) and fsq_val < fsq_floor:
            return InitialAxisResetEvaluation(
                InitialAxisResetDecision(False, False, False),
                fsq_phys,
                None,
                False,
            )
    bad_jacobian_ptau = None
    bad_jacobian_state = False
    if bool(axis_reset_enabled):
        try:
            ptau_min, ptau_max = ptau_minmax_from_k_host(k)
        except Exception:
            ptau_min, ptau_max = None, None
        bad_jacobian_ptau = bad_jacobian_ptau_from_minmax(
            ptau_min=ptau_min,
            ptau_max=ptau_max,
            ptau_tol=ptau_tol,
            ptau_tol_rel=ptau_tol_rel,
        )
        if bool(badjac_use_state) or (bool(state_check_on_missing_ptau) and bad_jacobian_ptau is None):
            try:
                jac = vmec_half_mesh_jacobian_from_state_func(
                    state=state, modes=static.modes, trig=trig, s=s,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    lthreed=bool(getattr(static.cfg, "lthreed", True)),
                    mask_even=getattr(static, "m_is_even", None),
                    mask_odd=getattr(static, "m_is_odd", None),
                )
                tau = jnp.asarray(jac.tau)
                tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
                min_tau = float(np.asarray(jnp.min(tau_use)))
                max_tau = float(np.asarray(jnp.max(tau_use)))
                tau_scale = max(abs(min_tau), abs(max_tau))
                bad_jacobian_state = bad_jacobian_from_tau_range(
                    min_tau=min_tau,
                    max_tau=max_tau,
                    abs_tol=max(1.0e-12, 1.0e-2 * tau_scale),
                )
            except Exception:
                bad_jacobian_state = False

    decision = initial_axis_reset_decision(
        bad_jacobian_ptau=bad_jacobian_ptau, bad_jacobian_state=bad_jacobian_state,
        badjac_use_state=badjac_use_state, fsq_phys=fsq_phys,
        axis_reset_fsq_min=axis_reset_fsq_min, force_axis_reset=force_axis_reset,
        axis_reset_always_3d=axis_reset_always_3d,
        lthreed=bool(getattr(static.cfg, "lthreed", True)),
        vmec2000_control=vmec2000_control, lmove_axis=lmove_axis,
        axis_reset_enabled=axis_reset_enabled,
    )
    if bool(debug_enabled):
        try:
            fsq_debug = float("nan") if fsq_phys is None else float(fsq_phys)
            print(
                "[axis_reset] fsq0="
                f"{fsq_debug:.6e} axis_reset_fsq_min={axis_reset_fsq_min:.3e} "
                f"badjac_ptau={bad_jacobian_ptau} badjac_state={bad_jacobian_state} "
                f"badjac_used={bool(decision.bad_jacobian)}",
                flush=True,
            )
        except Exception:
            pass
    return InitialAxisResetEvaluation(decision, fsq_phys, bad_jacobian_ptau, bool(bad_jacobian_state))


def run_initial_axis_reset_setup(
    *,
    state: VMECState,
    axis_reset_done: bool,
    ijacob: int,
    state_checkpoint: VMECState,
    velocities: tuple[Any, ...],
    res0: float,
    res1: float,
    prev_rz_fsq: float,
    vmec2000_control: bool,
    lmove_axis: bool,
    verbose: bool,
    verbose_vmec2000_table: bool,
    timing_enabled: bool,
    timing_stats: dict[str, float],
    force_axis_reset: bool,
    axis_reset_always_3d: bool,
    axis_reset_fsq_min: float,
    badjac_use_state: bool,
    static: Any,
    trig: Any,
    s: Any,
    zero_precond_diag: Any,
    zero_tcon: Any,
    compute_forces_iter_func: Callable[..., Any],
    reset_axis_from_boundary_func: Callable[..., VMECState],
    zero_velocity_blocks_like_func: Callable[..., tuple[Any, ...]],
    ptau_minmax_from_k_host_func: Callable[[Any], tuple[Any | None, Any | None]],
    vmec_half_mesh_jacobian_from_state_func: Callable[..., Any],
    print_axis_guess_func: Callable[[Any, Any], None],
    axis_reset_coeffs_func: Callable[[], tuple[Any, Any, Any, Any] | None],
    env_enabled_func: Callable[[str], bool],
    getenv_func: Callable[[str, str], str],
    perf_counter_func: Callable[[], float],
    has_jax_func: Callable[[], bool],
    block_until_ready_func: Callable[[Any], Any] | None,
    jnp_module: Any = jnp,
) -> InitialAxisResetSetupResult:
    """Run the setup-time VMEC magnetic-axis reset with host-loop semantics."""

    reset_applied = False
    force_probe = None
    t_setup_axis_reset_start = perf_counter_func() if timing_enabled else None
    if bool(vmec2000_control) and (not bool(axis_reset_done)) and bool(lmove_axis):
        try:
            t_setup_axis_force_start = perf_counter_func() if timing_enabled else None
            k0, _frzl0, gcr2_0, gcz2_0, gcl2_0, _rz_scale0, _l_scale0, norms0 = (
                compute_forces_iter_func(
                    state,
                    include_edge=False,
                    zero_m1=jnp_module.asarray(1.0, dtype=jnp_module.asarray(state.Rcos).dtype),
                    constraint_precond_diag=zero_precond_diag,
                    constraint_tcon=zero_tcon,
                    constraint_precond_active=jnp_module.asarray(False),
                    constraint_tcon_active=jnp_module.asarray(False),
                    iter_idx=None,
                    iter2=1,
                )
            )
            force_probe = (k0, _frzl0, gcr2_0, gcz2_0, gcl2_0, _rz_scale0, _l_scale0, norms0)
            if timing_enabled and t_setup_axis_force_start is not None:
                try:
                    if has_jax_func() and block_until_ready_func is not None:
                        block_until_ready_func((gcr2_0, gcz2_0, gcl2_0))
                except Exception:
                    pass
                timing_stats["setup_axis_reset_compute_forces"] += (
                    perf_counter_func() - float(t_setup_axis_force_start)
                )
            axis_reset_eval = evaluate_initial_axis_reset(
                axis_reset_enabled=True,
                norms=norms0,
                gcr2=gcr2_0,
                gcz2=gcz2_0,
                gcl2=gcl2_0,
                k=k0,
                state=state,
                static=static,
                trig=trig,
                s=s,
                badjac_use_state=bool(badjac_use_state),
                ptau_tol=0.0,
                ptau_tol_rel=0.0,
                axis_reset_fsq_min=axis_reset_fsq_min,
                force_axis_reset=bool(force_axis_reset),
                axis_reset_always_3d=bool(axis_reset_always_3d),
                vmec2000_control=True,
                lmove_axis=True,
                debug_enabled=env_enabled_func(getenv_func("VMEC_JAX_AXIS_RESET_DEBUG", "")),
                state_check_on_missing_ptau=True,
                ptau_minmax_from_k_host=ptau_minmax_from_k_host_func,
                vmec_half_mesh_jacobian_from_state_func=vmec_half_mesh_jacobian_from_state_func,
            )
            axis_reset_decision = axis_reset_eval.decision
            bad_jacobian0 = bool(axis_reset_decision.bad_jacobian)
            force_axis_reset_init = bool(axis_reset_decision.force_reset)
            if axis_reset_decision.reset:
                if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    if bad_jacobian0 or force_axis_reset_init:
                        print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                    print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
                state = reset_axis_from_boundary_func(
                    state,
                    k_guess=k0,
                    full_reset=False,
                    refine_axis_guess=False,
                )
                if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    coeffs = axis_reset_coeffs_func()
                    if coeffs is not None:
                        raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = coeffs
                        print_axis_guess_func(raxis_cc, zaxis_cs)
                axis_reset_done = True
                ijacob = 1
                state_checkpoint = state
                velocities = zero_velocity_blocks_like_func(*velocities)
                res0 = -1.0
                res1 = -1.0
                prev_rz_fsq = 2.0
                reset_applied = True
                force_probe = None
        except Exception:
            pass
    if timing_enabled and t_setup_axis_reset_start is not None:
        timing_stats["setup_axis_reset"] += perf_counter_func() - float(t_setup_axis_reset_start)
    return InitialAxisResetSetupResult(
        state=state,
        axis_reset_done=bool(axis_reset_done),
        ijacob=int(ijacob),
        state_checkpoint=state_checkpoint,
        velocities=velocities,
        res0=float(res0),
        res1=float(res1),
        prev_rz_fsq=float(prev_rz_fsq),
        reset_applied=bool(reset_applied),
        force_probe=force_probe,
    )


def write_axis_reset_dump(
    *, axis_dump_dir: str | os.PathLike[str] | None, ns: int, ntor: int, used_state_guess: bool,
    raxis_cc, raxis_cs, zaxis_cc, zaxis_cs,
) -> bool:
    """Write optional magnetic-axis reset coefficients for diagnostics."""

    if axis_dump_dir is None or str(axis_dump_dir).strip() == "":
        return False
    try:
        p = Path(axis_dump_dir).expanduser().resolve()
        ntor_i = int(ntor)
        rcc = np.asarray(raxis_cc)
        rcs = np.asarray(raxis_cs)
        zcc = np.asarray(zaxis_cc)
        zcs = np.asarray(zaxis_cs)
        if min(rcc.size, rcs.size, zcc.size, zcs.size) < ntor_i + 1:
            return False
        p.mkdir(parents=True, exist_ok=True)
        out = p / f"axis_reset_ns{int(ns)}.dat"
        with out.open("w", encoding="utf-8") as f:
            f.write(f"# used_state_guess={int(bool(used_state_guess))}\n")
            f.write("n raxis_cc raxis_cs zaxis_cc zaxis_cs\n")
            for n in range(ntor_i + 1):
                f.write(
                    f"{n:4d} "
                    f"{float(rcc[n]): .16e} "
                    f"{float(rcs[n]): .16e} "
                    f"{float(zcc[n]): .16e} "
                    f"{float(zcs[n]): .16e}\n"
                )
        return True
    except Exception:
        return False


def reset_axis_from_boundary(
    st: VMECState,
    *,
    boundary_for_axis: Any,
    static: Any,
    indata: Any,
    signgs: int,
    trig: Any,
    k_guess: Any = None,
    full_reset: bool = False,
    refine_axis_guess: bool = True,
    zero_precond_diag: Any,
    zero_tcon: Any,
    constraint_active_false: Any,
    compute_forces_iter_func: Callable[..., Any],
    apply_vmec_lambda_axis_rules_func: Callable[[VMECState], VMECState],
    initial_guess_from_boundary_func: Callable[..., VMECState],
    read_axis_coeffs_func: Callable[[Any], dict[str, Any]],
    recompute_axis_from_state_vmec_func: Callable[..., tuple[Any, Any, Any, Any]],
    recompute_axis_from_boundary_func: Callable[..., tuple[Any, Any]],
    axis_dump_dir: str | os.PathLike[str] | None = None,
) -> tuple[VMECState, tuple[Any, Any, Any, Any] | None]:
    """Return a VMEC-style initial magnetic-axis reset state and coefficients."""

    if boundary_for_axis is None:
        return st, None

    ntor = int(static.cfg.ntor)
    raxis_cc = np.zeros((ntor + 1,), dtype=float)
    raxis_cs = np.zeros((ntor + 1,), dtype=float)
    zaxis_cc = np.zeros((ntor + 1,), dtype=float)
    zaxis_cs = np.zeros((ntor + 1,), dtype=float)

    used_state_guess = False
    if k_guess is not None:
        try:
            raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = recompute_axis_from_state_vmec_func(
                static,
                pr1_even=k_guess.pr1_even,
                pr1_odd=k_guess.pr1_odd,
                pz1_even=k_guess.pz1_even,
                pz1_odd=k_guess.pz1_odd,
                pru_even=k_guess.pru_even,
                pru_odd=k_guess.pru_odd,
                pzu_even=k_guess.pzu_even,
                pzu_odd=k_guess.pzu_odd,
                signgs=int(signgs),
                trig=trig,
            )
            used_state_guess = True
        except Exception:
            used_state_guess = False

    def _state_from_axis_coeffs(
        rcc: np.ndarray,
        rcs: np.ndarray,
        zcc: np.ndarray,
        zcs: np.ndarray,
        *,
        dtype,
    ) -> VMECState:
        scalars_local = dict(indata.scalars)
        scalars_local["RAXIS_CC"] = [float(v) for v in np.ravel(rcc)]
        scalars_local["RAXIS_CS"] = [float(v) for v in np.ravel(rcs)]
        scalars_local["ZAXIS_CC"] = [float(v) for v in np.ravel(zcc)]
        scalars_local["ZAXIS_CS"] = [float(v) for v in np.ravel(zcs)]
        indata_local = type(indata)(scalars=scalars_local, indexed=indata.indexed)
        return initial_guess_from_boundary_func(
            static,
            boundary_for_axis,
            indata_local,
            dtype=dtype,
            infer_axis_if_missing=False,
        )

    if used_state_guess and bool(refine_axis_guess):
        try:
            st_tmp = _state_from_axis_coeffs(
                raxis_cc,
                raxis_cs,
                zaxis_cc,
                zaxis_cs,
                dtype=jnp.asarray(st.Rcos).dtype,
            )
            k_tmp, _, _, _, _, _, _, _ = compute_forces_iter_func(
                st_tmp,
                include_edge=False,
                zero_m1=jnp.asarray(1.0, dtype=jnp.asarray(st.Rcos).dtype),
                constraint_precond_diag=zero_precond_diag,
                constraint_tcon=zero_tcon,
                constraint_precond_active=constraint_active_false,
                constraint_tcon_active=constraint_active_false,
                iter_idx=None,
                iter2=1,
            )
            raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = recompute_axis_from_state_vmec_func(
                static,
                pr1_even=k_tmp.pr1_even,
                pr1_odd=k_tmp.pr1_odd,
                pz1_even=k_tmp.pz1_even,
                pz1_odd=k_tmp.pz1_odd,
                pru_even=k_tmp.pru_even,
                pru_odd=k_tmp.pru_odd,
                pzu_even=k_tmp.pzu_even,
                pzu_odd=k_tmp.pzu_odd,
                signgs=int(signgs),
                trig=trig,
            )
        except Exception:
            pass

    if not used_state_guess:
        axis_vals = read_axis_coeffs_func(indata)
        raxis_cc = np.asarray(axis_vals.get("RAXIS_CC", 0.0), dtype=float)
        zaxis_cs = np.asarray(axis_vals.get("ZAXIS_CS", 0.0), dtype=float)
        if raxis_cc.ndim == 0:
            raxis_cc = np.asarray([float(raxis_cc)], dtype=float)
        if zaxis_cs.ndim == 0:
            zaxis_cs = np.asarray([float(zaxis_cs)], dtype=float)
        if raxis_cc.size < ntor + 1:
            raxis_cc = np.pad(raxis_cc, (0, ntor + 1 - raxis_cc.size))
        if zaxis_cs.size < ntor + 1:
            zaxis_cs = np.pad(zaxis_cs, (0, ntor + 1 - zaxis_cs.size))
        raxis_cc, zaxis_cs = recompute_axis_from_boundary_func(
            static,
            boundary_for_axis,
            raxis_cc=raxis_cc,
            zaxis_cs=zaxis_cs,
            signgs=int(signgs),
        )

    write_axis_reset_dump(
        axis_dump_dir=axis_dump_dir,
        ns=int(static.cfg.ns),
        ntor=int(static.cfg.ntor),
        used_state_guess=bool(used_state_guess),
        raxis_cc=raxis_cc,
        raxis_cs=raxis_cs,
        zaxis_cc=zaxis_cc,
        zaxis_cs=zaxis_cs,
    )

    st_axis = _state_from_axis_coeffs(
        raxis_cc,
        raxis_cs,
        zaxis_cc,
        zaxis_cs,
        dtype=jnp.asarray(st.Rcos).dtype,
    )
    st_out = merge_axis_reset_state(st=st, st_axis=st_axis, static=static, full_reset=full_reset)
    return apply_vmec_lambda_axis_rules_func(st_out), (raxis_cc, raxis_cs, zaxis_cc, zaxis_cs)

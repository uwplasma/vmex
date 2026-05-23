"""Implicit-diff optimization of boundary modes for target iota, aspect, and volume.

This example optimizes a small set of boundary coefficients (RBC/ZBS) while
holding the major-radius mode R00 fixed. The equilibrium solve uses the
implicit fixed-boundary solver so gradients do not backpropagate through the
inner iterations.

For the exact discrete-adjoint fixed-resolution quasisymmetry route used in the
current QH optimization work, see ``qh_fixed_resolution_exact.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import time

import numpy as np

import vmec_jax as vj


def main() -> None:
    os.environ.setdefault("VMEC_JAX_SCAN_PRINT", "0")

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="cth_like_fixed_bdy")
    parser.add_argument("--opt-steps", type=int, default=6, help="Outer optimization steps.")
    parser.add_argument(
        "--opt-lr",
        type=float,
        default=1e-4,
        help="Trust-region style outer step size applied after gradient normalization.",
    )
    parser.add_argument("--max-backtracks", type=int, default=8, help="Backtracking attempts for each outer step.")
    parser.add_argument("--target-iota", type=float, default=None, help="Target mean iota.")
    parser.add_argument("--target-aspect", type=float, default=None, help="Target equilibrium aspect ratio.")
    parser.add_argument("--target-volume", type=float, default=None, help="Target total volume.")
    parser.add_argument("--ns", type=int, default=None, help="Radial resolution (NS). Defaults to input file.")
    parser.add_argument("--niter", type=int, default=None, help="Inner VMEC iterations per objective eval.")
    parser.add_argument("--ftol", type=float, default=None, help="Inner solver grad tolerance.")
    args = parser.parse_args()

    from vmec_jax._compat import enable_x64, has_jax, jax, jnp
    from vmec_jax.geom import eval_geom
    from vmec_jax.implicit import ImplicitFixedBoundaryOptions, solve_fixed_boundary_state_implicit
    from vmec_jax.integrals import volume_from_sqrtg
    from vmec_jax.static import build_static
    from vmec_jax.vmec_parity import vmec_m1_physical_to_internal_signed

    if not has_jax():
        raise SystemExit("This example requires JAX (pip install -e .).")
    enable_x64(True)

    root = Path(__file__).resolve().parents[2]
    input_path = root / "examples" / "data" / f"input.{args.case}"

    cfg0, indata = vj.load_config(input_path)

    def _last_array_value(key, fallback):
        value = indata.get(key, None)
        if isinstance(value, list) and value:
            return value[-1]
        if value is None:
            return fallback
        return value

    niter_default = min(int(_last_array_value("NITER_ARRAY", indata.get_int("NITER", 50))), 120)
    ns_use = int(args.ns) if args.ns is not None else int(cfg0.ns)
    niter_use = int(args.niter) if args.niter is not None else int(niter_default)
    ftol_use = float(args.ftol) if args.ftol is not None else float(
        _last_array_value("FTOL_ARRAY", indata.get_float("FTOL", 1e-11))
    )
    cfg = replace(cfg0, ns=int(ns_use))
    static = build_static(cfg)

    boundary0 = vj.boundary_from_indata(indata, static.modes, apply_m1_constraint=False)
    Rcos0 = jnp.asarray(boundary0.R_cos)
    Rsin0 = jnp.asarray(boundary0.R_sin)
    Zcos0 = jnp.asarray(boundary0.Z_cos)
    Zsin0 = jnp.asarray(boundary0.Z_sin)

    def _select_boundary_modes(coeffs, label_prefix):
        m_arr = np.asarray(static.modes.m, dtype=int)
        n_arr = np.asarray(static.modes.n, dtype=int)
        sel_idx = []
        sel_names = []
        for index, (m_val, n_val) in enumerate(zip(m_arr, n_arr)):
            if m_val == 0 and n_val == 0:
                continue
            if abs(m_val) > 1 or abs(n_val) > 1:
                continue
            if abs(float(np.asarray(coeffs)[index])) <= 0.0:
                continue
            sel_idx.append(int(index))
            sel_names.append(f"{label_prefix}({int(m_val)},{int(n_val)})")
        return sel_idx, sel_names

    idx_R, names_R = _select_boundary_modes(Rcos0, "RBC")
    idx_Z, names_Z = _select_boundary_modes(Zsin0, "ZBS")
    if not idx_R and not idx_Z:
        raise SystemExit("No boundary modes selected for optimization.")

    m_arr = np.asarray(static.modes.m, dtype=int)
    n_arr = np.asarray(static.modes.n, dtype=int)
    k00 = None
    k00_match = np.where((m_arr == 0) & (n_arr == 0))[0]
    if k00_match.size:
        k00 = int(k00_match[0])

    st0 = vj.initial_guess_from_boundary(static, boundary0, indata, vmec_project=False)
    g0 = eval_geom(st0, static)
    signgs = int(vj.signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1))

    flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
    phipf = jnp.asarray(flux.phipf)
    chipf = jnp.asarray(flux.chipf)
    lamscale = jnp.asarray(flux.lamscale)
    pressure = jnp.asarray(vj.eval_profiles(indata, static.s).get("pressure", jnp.zeros_like(static.s)))
    use_iota_target = int(indata.get_int("NCURR", 0)) != 0
    if not use_iota_target:
        print("note: NCURR=0 -> omitting iota from the optimization objective.")

    mode_scale = static.mode_scale_internal
    if mode_scale is None:
        raise SystemExit("static.mode_scale_internal is required for this example.")
    mode_scale = jnp.asarray(mode_scale)

    implicit_opts = ImplicitFixedBoundaryOptions(cg_max_iter=60, cg_tol=1e-10, damping=1e-5)

    def _build_boundary(params):
        params = jnp.asarray(params)
        Rcos = Rcos0
        Zsin = Zsin0
        if idx_R:
            Rcos = Rcos.at[jnp.asarray(idx_R, dtype=jnp.int32)].add(params[: len(idx_R)])
        if idx_Z:
            Zsin = Zsin.at[jnp.asarray(idx_Z, dtype=jnp.int32)].add(params[len(idx_R) :])
        if k00 is not None:
            Rcos = Rcos.at[int(k00)].set(Rcos0[int(k00)])
        return Rcos, Zsin

    def _boundary_to_edge(Rcos, Zsin):
        Rcos_i = (jnp.asarray(Rcos) * mode_scale)[None, :]
        Rsin_i = (jnp.asarray(Rsin0) * mode_scale)[None, :]
        Zcos_i = (jnp.asarray(Zcos0) * mode_scale)[None, :]
        Zsin_i = (jnp.asarray(Zsin) * mode_scale)[None, :]
        Rcos_i, Zsin_i, Rsin_i, Zcos_i = vmec_m1_physical_to_internal_signed(
            Rcos=Rcos_i,
            Zsin=Zsin_i,
            Rsin=Rsin_i,
            Zcos=Zcos_i,
            modes=static.modes,
            lthreed=bool(cfg.ntor > 0),
            lasym=bool(cfg.lasym),
            lconm1=bool(cfg.lconm1),
        )
        return Rcos_i[0], Rsin_i[0], Zcos_i[0], Zsin_i[0]

    def _state_guess_from_boundary(Rcos, Zsin):
        boundary_guess = vj.BoundaryCoeffs(
            R_cos=jax.lax.stop_gradient(jnp.asarray(Rcos)),
            R_sin=jax.lax.stop_gradient(jnp.asarray(Rsin0)),
            Z_cos=jax.lax.stop_gradient(jnp.asarray(Zcos0)),
            Z_sin=jax.lax.stop_gradient(jnp.asarray(Zsin)),
        )
        return vj.initial_guess_from_boundary(static, boundary_guess, indata, vmec_project=False)

    def _iota_mean(state):
        _chips, _iotas, iotaf = vj.equilibrium_iota_profiles_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
        )
        return jnp.mean(iotaf[1:]) if iotaf.size > 1 else iotaf[0]

    def _aspect_ratio(state):
        return vj.equilibrium_aspect_ratio_from_state(state=state, static=static)

    def _volume_total(state):
        geom = eval_geom(state, static)
        _dvds, vol = volume_from_sqrtg(
            geom.sqrtg,
            static.s,
            static.grid.theta,
            static.grid.zeta,
            nfp=int(cfg.nfp),
        )
        return vol[-1]

    def _solve_with_solver(state_guess, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin, solver_name, step_size):
        return solve_fixed_boundary_state_implicit(
            state_guess,
            static,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs,
            lamscale=lamscale,
            pressure=pressure,
            gamma=float(indata.get_float("GAMMA", 0.0)),
            jacobian_penalty=1e3,
            solver=solver_name,
            max_iter=int(niter_use),
            step_size=float(step_size),
            history_size=8,
            grad_tol=float(ftol_use),
            preconditioner="mode_diag+radial_tridi",
            precond_exponent=1.0,
            precond_radial_alpha=0.5,
            implicit=implicit_opts,
            implicit_converge_tol=max(float(ftol_use), 1e-6),
            implicit_zero_unconverged=False,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
        )

    def _solve_state(params):
        Rcos, Zsin = _build_boundary(params)
        state_guess = _state_guess_from_boundary(Rcos, Zsin)
        edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin = _boundary_to_edge(Rcos, Zsin)
        try:
            return _solve_with_solver(state_guess, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin, "lbfgs", 1.0)
        except ValueError:
            return _solve_with_solver(state_guess, edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin, "gd", 0.2)

    params0 = jnp.zeros((len(idx_R) + len(idx_Z),), dtype=jnp.float64)
    print("computing base equilibrium for target values...")
    st_base = _solve_state(params0)
    iota_base = float(np.asarray(_iota_mean(st_base))) if use_iota_target else None
    aspect_base = float(np.asarray(_aspect_ratio(st_base)))
    vol_base = float(np.asarray(_volume_total(st_base)))

    target_iota = None
    if use_iota_target:
        target_iota = float(args.target_iota) if args.target_iota is not None else (1.05 * float(iota_base))
    target_aspect = float(args.target_aspect) if args.target_aspect is not None else (1.05 * aspect_base)
    target_volume = float(args.target_volume) if args.target_volume is not None else (1.05 * vol_base)

    iota_scale = max(abs(target_iota), 1e-3) if target_iota is not None else 1.0
    aspect_scale = max(abs(target_aspect), 1e-3)
    volume_scale = max(abs(target_volume), 1e-6)

    def objective(params):
        st = _solve_state(params)
        iota_mean = _iota_mean(st) if use_iota_target else jnp.asarray(0.0, dtype=jnp.asarray(params).dtype)
        aspect = _aspect_ratio(st)
        vol_total = _volume_total(st)
        loss = ((aspect - target_aspect) / aspect_scale) ** 2 + ((vol_total - target_volume) / volume_scale) ** 2
        if use_iota_target:
            loss = loss + ((iota_mean - target_iota) / iota_scale) ** 2
        return loss, (iota_mean, aspect, vol_total)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    params = params0

    print(f"NS={int(cfg.ns)} NITER={int(niter_use)} FTOL={float(ftol_use):.3e}")
    if target_iota is not None:
        print(f"target_iota={target_iota:.6e}")
    print(f"target_aspect={target_aspect:.6e} target_volume={target_volume:.6e}")
    print("params:", " ".join(names_R + names_Z))
    if k00 is not None:
        print(f"R00 fixed at {float(np.asarray(Rcos0[int(k00)])):.6e}")

    for step in range(int(args.opt_steps)):
        t0 = time.perf_counter()
        (val, aux), grad = value_and_grad(params)
        dt = time.perf_counter() - t0
        val_f = float(np.asarray(val))
        iota_mean = float(np.asarray(aux[0]))
        aspect = float(np.asarray(aux[1]))
        vol_total = float(np.asarray(aux[2]))
        grad_np = np.asarray(grad, dtype=float)
        grad_norm = float(np.linalg.norm(grad_np))
        grad_max = float(np.max(np.abs(grad_np))) if grad_np.size else 0.0

        accepted = False
        accepted_loss = val_f
        accepted_step = 0.0
        step_size = float(args.opt_lr)
        if grad_norm > 0.0:
            for _backtrack in range(int(args.max_backtracks) + 1):
                trial_params = params - (step_size / grad_norm) * grad
                try:
                    trial_val, _trial_aux = objective(trial_params)
                    trial_val_f = float(np.asarray(trial_val))
                except ValueError:
                    trial_val_f = float("inf")
                if np.isfinite(trial_val_f) and trial_val_f < val_f:
                    params = trial_params
                    accepted = True
                    accepted_loss = trial_val_f
                    accepted_step = step_size
                    break
                step_size *= 0.5

        msg = (
            f"step {step:02d}: loss={val_f:.6e} next_loss={accepted_loss:.6e} "
            f"aspect={aspect:.6e} volume={vol_total:.6e} grad_norm={grad_norm:.6e} "
            f"grad_max={grad_max:.6e} step={accepted_step:.6e} dt={dt:.3f}s"
        )
        if target_iota is not None:
            msg = (
                f"step {step:02d}: loss={val_f:.6e} next_loss={accepted_loss:.6e} "
                f"iota_mean={iota_mean:.6e} aspect={aspect:.6e} volume={vol_total:.6e} "
                f"grad_norm={grad_norm:.6e} grad_max={grad_max:.6e} step={accepted_step:.6e} dt={dt:.3f}s"
            )
        print(msg)
        if not accepted:
            print("no decreasing trial step found; stopping")
            break


if __name__ == "__main__":
    main()

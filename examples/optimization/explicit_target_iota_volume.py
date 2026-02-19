"""Explicit-diff optimization of boundary modes for target iota + volume.

This example differentiates *through* the inner VMEC iterations (no implicit
solve). It is intentionally similar to the implicit example for comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import time

import numpy as np

import vmec_jax as vj


def _pick_mode_indices(modes, targets):
    idx = []
    names = []
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    for m_val, n_val, label in targets:
        mask = (m_arr == int(m_val)) & (n_arr == int(n_val))
        if not np.any(mask):
            continue
        k = int(np.where(mask)[0][0])
        idx.append(k)
        names.append(label)
    return idx, names


def main() -> None:
    os.environ.setdefault("VMEC_JAX_SCAN_PRINT", "0")

    p = argparse.ArgumentParser()
    p.add_argument("--case", default="circular_tokamak")
    p.add_argument("--opt-steps", type=int, default=10, help="Outer optimization steps.")
    p.add_argument("--opt-lr", type=float, default=5e-2, help="Outer optimization learning rate.")
    p.add_argument("--target-iota", type=float, default=None, help="Target mid-radius iota.")
    p.add_argument("--target-volume", type=float, default=None, help="Target mid-radius volume.")
    p.add_argument("--ns", type=int, default=31, help="Radial resolution (NS).")
    p.add_argument("--niter", type=int, default=1000, help="Inner VMEC iterations per objective eval.")
    p.add_argument("--ftol", type=float, default=1e-13, help="Inner solver grad tolerance.")
    p.add_argument("--step-size", type=float, default=5e-3, help="Inner solver step size.")
    args = p.parse_args()

    from vmec_jax._compat import enable_x64, has_jax, jax, jnp
    from vmec_jax.geom import eval_geom
    from vmec_jax.integrals import volume_from_sqrtg
    from vmec_jax.profiles import eval_profiles
    from vmec_jax.static import build_static
    from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
    from vmec_jax.vmec_parity import vmec_m1_physical_to_internal_signed
    from vmec_jax.vmec_residue import vmec_pwint_from_trig
    from vmec_jax.vmec_tomnsp import vmec_trig_tables
    from vmec_jax.wout import _icurv_full_mesh_from_indata

    if not has_jax():
        raise SystemExit("This example requires JAX (pip install -e '.[jax]').")
    enable_x64(True)

    root = Path(__file__).resolve().parents[2]
    input_path = root / "examples" / "data" / f"input.{args.case}"

    cfg0, indata = vj.load_config(input_path)
    cfg = replace(cfg0, ns=int(args.ns))
    static = build_static(cfg)

    boundary0 = vj.boundary_from_indata(indata, static.modes, apply_m1_constraint=False)
    Rcos0 = jnp.asarray(boundary0.R_cos)
    Rsin0 = jnp.asarray(boundary0.R_sin)
    Zcos0 = jnp.asarray(boundary0.Z_cos)
    Zsin0 = jnp.asarray(boundary0.Z_sin)

    targets = [
        (1, 0, "RBC(1,0)"),
        (2, 0, "RBC(2,0)"),
    ]
    if int(cfg.ntor) > 0:
        targets.append((1, 1, "RBC(1,1)"))
    idx_R, names_R = _pick_mode_indices(static.modes, targets)

    targets = [
        (1, 0, "ZBS(1,0)"),
        (2, 0, "ZBS(2,0)"),
    ]
    if int(cfg.ntor) > 0:
        targets.append((1, 1, "ZBS(1,1)"))
    idx_Z, names_Z = _pick_mode_indices(static.modes, targets)

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
    signgs = vj.signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
    phipf = jnp.asarray(flux.phipf)
    chipf = jnp.asarray(flux.chipf)
    phips = jnp.asarray(flux.phips)
    lamscale = jnp.asarray(flux.lamscale)

    prof = eval_profiles(indata, static.s)
    pressure = jnp.asarray(prof.get("pressure", jnp.zeros_like(static.s)))
    iota_prof = jnp.asarray(prof.get("iota", jnp.zeros_like(static.s)))
    if iota_prof.size:
        iota_prof = iota_prof.at[0].set(jnp.asarray(0.0, dtype=iota_prof.dtype))

    s_idx = int(len(static.s) // 2)

    ncurr = int(indata.get_int("NCURR", 0))
    icurv = None
    trig = None
    if ncurr != 0:
        icurv = jnp.asarray(_icurv_full_mesh_from_indata(indata=indata, s_full=np.asarray(static.s), signgs=signgs))
        trig = static.trig_vmec
        if trig is None:
            trig = vmec_trig_tables(
                ntheta=int(cfg.ntheta),
                nzeta=int(cfg.nzeta),
                nfp=int(cfg.nfp),
                mmax=int(cfg.mpol) - 1,
                nmax=int(cfg.ntor),
                lasym=bool(cfg.lasym),
                dtype=np.asarray(static.s).dtype,
            )
    else:
        print("note: NCURR=0 -> iota is prescribed by the input profile (boundary does not change iota).")

    mode_scale = static.mode_scale_internal
    if mode_scale is None:
        raise SystemExit("static.mode_scale_internal is required for this example.")
    mode_scale = jnp.asarray(mode_scale)

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
        return Rcos, Rsin0, Zcos0, Zsin

    def _boundary_to_edge(Rcos, Rsin, Zcos, Zsin):
        Rcos_i = (jnp.asarray(Rcos) * mode_scale)[None, :]
        Rsin_i = (jnp.asarray(Rsin) * mode_scale)[None, :]
        Zcos_i = (jnp.asarray(Zcos) * mode_scale)[None, :]
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

    def _iota_mid(state):
        if ncurr == 0:
            return iota_prof[s_idx]
        from types import SimpleNamespace

        wout_like = SimpleNamespace(
            phipf=phipf,
            phips=phips,
            chipf=jnp.zeros_like(phipf),
            signgs=int(signgs),
            nfp=int(cfg.nfp),
            mpol=int(cfg.mpol),
            ntor=int(cfg.ntor),
            lasym=bool(cfg.lasym),
            ncurr=int(ncurr),
            lcurrent=True,
            icurv=icurv,
            flux_is_internal=True,
        )
        bc = vmec_bcovar_half_mesh_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            pres=pressure,
            use_vmec_synthesis=True,
            trig=trig,
        )
        sqrtg = jnp.asarray(bc.jac.sqrtg)
        overg = jnp.where(sqrtg != 0.0, 1.0 / sqrtg, 0.0)
        pwint = vmec_pwint_from_trig(trig, ns=int(overg.shape[0]), nzeta=int(overg.shape[2])).astype(overg.dtype)
        guu = jnp.asarray(bc.guu)
        guv = jnp.asarray(bc.guv)
        bsupu = jnp.asarray(bc.bsupu)
        bsupv = jnp.asarray(bc.bsupv)
        top = jnp.asarray(icurv, dtype=overg.dtype) - jnp.sum(
            pwint * ((guu * bsupu) + (guv * bsupv)),
            axis=(1, 2),
        )
        bot = jnp.sum(pwint * (overg * guu), axis=(1, 2))
        chips = jnp.where(bot != 0.0, top / bot, jnp.zeros_like(top))
        chips = chips.at[0].set(jnp.asarray(0.0, dtype=chips.dtype))
        iotas = jnp.where(phips != 0.0, chips / phips, jnp.zeros_like(chips))
        iotas = iotas.at[0].set(jnp.asarray(0.0, dtype=iotas.dtype))
        return iotas[s_idx]

    def _volume_mid(state):
        geom = eval_geom(state, static)
        _dvds, vol = volume_from_sqrtg(
            geom.sqrtg,
            static.s,
            static.grid.theta,
            static.grid.zeta,
            nfp=int(cfg.nfp),
        )
        return vol[s_idx]

    def _solve_state(params):
        Rcos, Rsin, Zcos, Zsin = _build_boundary(params)
        edge_Rcos, edge_Rsin, edge_Zcos, edge_Zsin = _boundary_to_edge(Rcos, Rsin, Zcos, Zsin)
        res = vj.solve_fixed_boundary_gd(
            st0,
            static,
            phipf=phipf,
            chipf=chipf,
            signgs=int(signgs),
            lamscale=lamscale,
            pressure=pressure,
            gamma=float(indata.get_float("GAMMA", 0.0)),
            jacobian_penalty=1e3,
            max_iter=int(args.niter),
            step_size=float(args.step_size),
            grad_tol=float(args.ftol),
            max_backtracks=12,
            bt_factor=0.5,
            preconditioner="mode_diag+radial_tridi",
            precond_exponent=1.0,
            precond_radial_alpha=0.5,
            differentiable=True,
            stop_grad_in_update=False,
            verbose=False,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
        )
        return res.state

    params0 = jnp.zeros((len(idx_R) + len(idx_Z),), dtype=jnp.float64)
    if args.target_iota is None or args.target_volume is None:
        print("computing base equilibrium for target values...")
        st_base = _solve_state(params0)
        iota_base = float(np.asarray(_iota_mid(st_base)))
        vol_base = float(np.asarray(_volume_mid(st_base)))
    else:
        iota_base = float(args.target_iota)
        vol_base = float(args.target_volume)

    target_iota = float(args.target_iota) if args.target_iota is not None else (1.05 * iota_base)
    target_volume = float(args.target_volume) if args.target_volume is not None else (0.98 * vol_base)

    def objective(params):
        st = _solve_state(params)
        iota_mid = _iota_mid(st)
        vol_mid = _volume_mid(st)
        loss = (iota_mid - target_iota) ** 2 + (vol_mid - target_volume) ** 2
        return loss, (iota_mid, vol_mid)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    params = params0

    print(f"target_iota={target_iota:.6e} target_volume={target_volume:.6e}")
    print("params:", " ".join(names_R + names_Z))
    if k00 is not None:
        print(f"R00 fixed at {float(np.asarray(Rcos0[int(k00)])):.6e}")

    for step in range(int(args.opt_steps)):
        t0 = time.perf_counter()
        (val, aux), grad = value_and_grad(params)
        dt = time.perf_counter() - t0
        iota_mid, vol_mid = aux
        grad_abs = float(jnp.sum(jnp.abs(grad)))
        grad_max = float(jnp.max(jnp.abs(grad)))
        params = params - float(args.opt_lr) * grad
        print(
            f"step {step:02d}: loss={float(val):.6e} "
            f"iota_mid={float(iota_mid):.6e} volume_mid={float(vol_mid):.6e} "
            f"grad_abs={grad_abs:.6e} grad_max={grad_max:.6e} dt={dt:.3f}s"
        )


if __name__ == "__main__":
    main()

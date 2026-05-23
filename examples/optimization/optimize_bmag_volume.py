"""Differentiate through a fixed-boundary solve w.r.t. boundary coefficients.

This tutorial shows how to:
- build a tiny differentiable fixed-boundary solve,
- optimize boundary coefficients with `jax.grad`,
- track mean(|B|) on the LCFS and total volume,
- optionally *report* iota from a VMEC-style wout (non-differentiable).

Note: iota reporting uses `wout_minimal_from_fixed_boundary`, which is a
NumPy-heavy diagnostic path. It is included as a *reporting* tool and is not
part of the differentiable objective by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vmec_jax as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="circular_tokamak")
    p.add_argument("--max-iter", type=int, default=2, help="Inner VMEC iterations per objective eval.")
    p.add_argument("--step-size", type=float, default=5e-3, help="Inner VMEC update step size.")
    p.add_argument("--opt-steps", type=int, default=3, help="Outer optimization steps on boundary params.")
    p.add_argument("--opt-lr", type=float, default=1e-2, help="Outer optimization learning rate.")
    p.add_argument("--target-volume", type=float, default=None, help="Target total volume (optional).")
    p.add_argument("--volume-weight", type=float, default=1e-2, help="Weight for volume penalty.")
    p.add_argument("--target-iota", type=float, default=None, help="Target mean iota (reporting only).")
    args = p.parse_args()

    from vmec_jax._compat import enable_x64, has_jax, jax, jnp
    from vmec_jax.integrals import volume_from_sqrtg_vmec

    if not has_jax():
        raise SystemExit("This example requires JAX (pip install -e .).")
    enable_x64(True)

    root = Path(__file__).resolve().parents[2]
    input_path = root / "examples" / "data" / f"input.{args.case}"

    cfg, indata = vj.load_config(input_path)
    static = vj.build_static(cfg)
    boundary0 = vj.boundary_from_indata(indata, static.modes)

    # Pick a few boundary coefficients to treat as parameters.
    # RBC(0,0), RBC(1,0), ZBS(1,0) for a tokamak-like shape.
    modes = static.modes
    k00 = int(np.where((np.asarray(modes.m) == 0) & (np.asarray(modes.n) == 0))[0][0])
    k10 = int(np.where((np.asarray(modes.m) == 1) & (np.asarray(modes.n) == 0))[0][0])

    # signgs is a fixed convention; infer it once from the unperturbed guess.
    st0 = vj.initial_guess_from_boundary(static, boundary0, indata, vmec_project=False)
    g0 = vj.eval_geom(st0, static)
    signgs0 = vj.signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)

    # Precompute flux profiles (fixed for this example).
    flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs0)
    pressure = jnp.zeros_like(jnp.asarray(static.s))

    def _build_boundary(params):
        dR00, dR10, dZ10 = params
        Rcos = jnp.asarray(boundary0.R_cos).at[k00].add(dR00).at[k10].add(dR10)
        Zsin = jnp.asarray(boundary0.Z_sin).at[k10].add(dZ10)
        return vj.BoundaryCoeffs(
            R_cos=Rcos,
            R_sin=jnp.asarray(boundary0.R_sin),
            Z_cos=jnp.asarray(boundary0.Z_cos),
            Z_sin=Zsin,
        )

    def objective(params):
        # params = [dRBC(0,0), dRBC(1,0), dZBS(1,0)]
        bdy = _build_boundary(params)
        st0 = vj.initial_guess_from_boundary(static, bdy, indata, vmec_project=False)

        res = vj.solve_fixed_boundary_gd(
            st0,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs0,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=float(indata.get_float("GAMMA", 0.0)),
            max_iter=int(args.max_iter),
            step_size=float(args.step_size),
            jacobian_penalty=1e3,
            jit_grad=False,
            differentiable=True,
            stop_grad_in_update=True,
            verbose=False,
        )

        theta = vj.closed_theta_grid(128)
        phi = jnp.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        B = vj.bmag_from_state_physical(
            res.state,
            static,
            indata=indata,
            theta=theta,
            phi=phi,
            s_index=int(cfg.ns) - 1,
            signgs=signgs0,
            phipf=flux.phipf,
            chipf=flux.chipf,
            lamscale=flux.lamscale,
            bmag_floor=1e-12,
        )
        bmag_mean = jnp.mean(B)

        geom = vj.eval_geom(res.state, static)
        _dvds, vol = volume_from_sqrtg_vmec(
            geom.sqrtg,
            static.s,
            static.grid.theta,
            static.grid.zeta,
            signgs=signgs0,
        )
        vol_total = vol[-1]

        obj = bmag_mean
        if args.target_volume is not None:
            obj = obj + float(args.volume_weight) * (vol_total - float(args.target_volume)) ** 2

        return obj, (bmag_mean, vol_total, res.state)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    params = jnp.zeros((3,), dtype=jnp.float64)

    print("params = [dRBC(0,0), dRBC(1,0), dZBS(1,0)]")
    for step in range(int(args.opt_steps)):
        (val, aux), grad = value_and_grad(params)
        bmag_mean, vol_total, state = aux
        params = params - float(args.opt_lr) * grad
        print(
            f"step {step:02d}: obj={float(val):.6e} "
            f"bmag_mean={float(bmag_mean):.6e} volume={float(vol_total):.6e}"
        )

        if args.target_iota is not None:
            # Reporting-only iota (non-differentiable path).
            from vmec_jax.wout import wout_minimal_from_fixed_boundary

            wout = wout_minimal_from_fixed_boundary(
                path=Path("/tmp/wout_tmp.nc"),
                state=state,
                static=static,
                indata=indata,
                signgs=signgs0,
                fsqr=0.0,
                fsqz=0.0,
                fsql=0.0,
            )
            iota_mean = float(np.mean(np.asarray(wout.iotaf)[1:]))
            print(f"  iota_mean={iota_mean:.6e} target={float(args.target_iota):.6e}")

    print("final params:", np.asarray(params))


if __name__ == "__main__":
    main()

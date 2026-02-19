"""Implicit-diff optimization of profile scalings for target iota + volume.

This example uses the implicit fixed-boundary solver so gradients do not
backpropagate through the inner equilibrium iterations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vmec_jax as vj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="circular_tokamak")
    p.add_argument("--opt-steps", type=int, default=4, help="Outer optimization steps.")
    p.add_argument("--opt-lr", type=float, default=5e-2, help="Outer optimization learning rate.")
    p.add_argument("--target-iota", type=float, default=None, help="Target mid-radius iota.")
    p.add_argument("--target-volume", type=float, default=None, help="Target mid-radius volume.")
    args = p.parse_args()

    from vmec_jax._compat import enable_x64, has_jax, jax, jnp
    from vmec_jax.energy import flux_profiles_from_indata
    from vmec_jax.field import chips_from_chipf
    from vmec_jax.geom import eval_geom
    from vmec_jax.implicit import ImplicitFixedBoundaryOptions, solve_fixed_boundary_state_implicit
    from vmec_jax.integrals import volume_from_sqrtg
    from vmec_jax.static import build_static
    from vmec_jax.wout import read_wout, state_from_wout

    if not has_jax():
        raise SystemExit("This example requires JAX (pip install -e '.[jax]').")
    enable_x64(True)

    root = Path(__file__).resolve().parents[2]
    input_path = root / "examples" / "data" / f"input.{args.case}"
    wout_path = root / "examples" / "data" / f"wout_{args.case}_reference.nc"

    cfg, indata = vj.load_config(input_path)
    static = build_static(cfg)
    wout = read_wout(wout_path)
    st0 = state_from_wout(wout)

    signgs = int(wout.signgs)
    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    phipf0 = jnp.asarray(flux.phipf)
    chipf0 = jnp.asarray(flux.chipf)
    phips0 = jnp.asarray(flux.phips)
    lamscale0 = jnp.asarray(flux.lamscale)
    pressure0 = jnp.asarray(wout.presf)

    s_idx = int(len(static.s) // 2)

    def _iota_from_chipf(chipf):
        chips = chips_from_chipf(chipf)
        den = jnp.where(phips0 != 0.0, phips0, jnp.ones_like(phips0))
        iotas = jnp.where(phips0 != 0.0, chips / den, jnp.zeros_like(chips))
        return iotas

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

    # Baseline targets if not provided by the user.
    iota_base = float(np.asarray(_iota_from_chipf(chipf0)[s_idx]))
    vol_base = float(np.asarray(_volume_mid(st0)))
    target_iota = float(args.target_iota) if args.target_iota is not None else (1.05 * iota_base)
    target_volume = float(args.target_volume) if args.target_volume is not None else (0.98 * vol_base)

    implicit_opts = ImplicitFixedBoundaryOptions(cg_max_iter=60, cg_tol=1e-10, damping=1e-5)

    def _solve_state(alpha, beta):
        return solve_fixed_boundary_state_implicit(
            st0,
            static,
            phipf=phipf0,
            chipf=alpha * chipf0,
            signgs=signgs,
            lamscale=lamscale0,
            pressure=beta * pressure0,
            gamma=float(wout.gamma),
            jacobian_penalty=1e3,
            solver="lbfgs",
            max_iter=20,
            step_size=1.0,
            history_size=8,
            grad_tol=1e-10,
            preconditioner="mode_diag+radial_tridi",
            precond_exponent=1.0,
            precond_radial_alpha=0.5,
            implicit=implicit_opts,
        )

    def objective(params):
        alpha, beta = params
        st = _solve_state(alpha, beta)
        iota_mid = _iota_from_chipf(alpha * chipf0)[s_idx]
        vol_mid = _volume_mid(st)
        loss = (iota_mid - target_iota) ** 2 + (vol_mid - target_volume) ** 2
        return loss, (iota_mid, vol_mid)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    params = jnp.asarray([1.0, 1.0], dtype=jnp.float64)

    print(f"target_iota={target_iota:.6e} target_volume={target_volume:.6e}")
    for step in range(int(args.opt_steps)):
        (val, aux), grad = value_and_grad(params)
        iota_mid, vol_mid = aux
        params = params - float(args.opt_lr) * grad
        print(
            f"step {step:02d}: loss={float(val):.6e} "
            f"iota_mid={float(iota_mid):.6e} volume_mid={float(vol_mid):.6e} "
            f"alpha={float(params[0]):.6e} beta={float(params[1]):.6e}"
        )


if __name__ == "__main__":
    main()

"""Small QH derivative/runtime probe for the discrete-adjoint recovery plan.

This script keeps the workload intentionally small:
- exact bundled `input.nfp4_QH_warm_start`
- one boundary DOF at a time
- one implicit-residual solve budget
- aspect derivative as the first stable gate
- lambda-state scalar derivative as the known hard diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, has_jax, jax, jnp
from vmec_jax.boundary import boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.implicit import solve_fixed_boundary_state_implicit_vmec_residual
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.wout import equilibrium_aspect_ratio_from_state


def _mode_index(modes, m: int, n: int) -> int:
    for k, (mm, nn) in enumerate(zip(np.asarray(modes.m), np.asarray(modes.n))):
        if int(mm) == int(m) and int(nn) == int(n):
            return k
    raise KeyError((m, n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="examples/data/input.nfp4_QH_warm_start")
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--surface-index", type=int, default=5)
    args = parser.parse_args()

    if not has_jax():
        raise SystemExit("This script requires JAX (pip install -e '.[jax]').")
    enable_x64(True)

    cfg, indata = load_config(REPO_ROOT / args.input)
    from vmec_jax.static import build_static

    static = build_static(cfg)
    boundary = boundary_from_indata(indata, static.modes)
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))

    edge_Rcos0 = np.asarray(boundary.R_cos, dtype=float)
    edge_Rsin0 = np.asarray(boundary.R_sin, dtype=float)
    edge_Zcos0 = np.asarray(boundary.Z_cos, dtype=float)
    edge_Zsin0 = np.asarray(boundary.Z_sin, dtype=float)
    k_rc01 = _mode_index(static.modes, 0, 1)
    k_l01 = _mode_index(static.modes, 0, 1)
    alpha0 = float(edge_Rcos0[k_rc01])

    def _solve_from_alpha(alpha):
        edge_Rcos = jnp.asarray(edge_Rcos0).at[k_rc01].set(alpha)
        return solve_fixed_boundary_state_implicit_vmec_residual(
            state_guess,
            static,
            indata=indata,
            signgs=signgs,
            state0_host=state_guess,
            max_iter=int(args.max_iter),
            step_size=float(indata.get_float("DELT", 1.0)),
            ftol=float(indata.get_float("FTOL", 1e-14)),
            edge_Rcos=edge_Rcos,
            edge_Rsin=jnp.asarray(edge_Rsin0),
            edge_Zcos=jnp.asarray(edge_Zcos0),
            edge_Zsin=jnp.asarray(edge_Zsin0),
        )

    def _aspect(alpha):
        state = _solve_from_alpha(alpha)
        return equilibrium_aspect_ratio_from_state(state=state, static=static)

    def _lambda_scalar(alpha):
        state = _solve_from_alpha(alpha)
        return jnp.asarray(state.Lsin)[int(args.surface_index), int(k_l01)]

    t0 = time.perf_counter()
    aspect_ad = float(np.asarray(jax.grad(_aspect)(alpha0)))
    aspect_ad_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    aspect_p = float(np.asarray(_aspect(alpha0 + args.eps)))
    aspect_m = float(np.asarray(_aspect(alpha0 - args.eps)))
    aspect_fd_s = time.perf_counter() - t0
    aspect_fd = (aspect_p - aspect_m) / (2.0 * float(args.eps))

    t0 = time.perf_counter()
    lambda_ad = float(np.asarray(jax.grad(_lambda_scalar)(alpha0)))
    lambda_ad_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    lambda_p = float(np.asarray(_lambda_scalar(alpha0 + args.eps)))
    lambda_m = float(np.asarray(_lambda_scalar(alpha0 - args.eps)))
    lambda_fd_s = time.perf_counter() - t0
    lambda_fd = (lambda_p - lambda_m) / (2.0 * float(args.eps))

    out = {
        "input": str(REPO_ROOT / args.input),
        "max_iter": int(args.max_iter),
        "eps": float(args.eps),
        "boundary_mode": {"m": 0, "n": 1, "index": int(k_rc01)},
        "lambda_probe": {"surface_index": int(args.surface_index), "mode_index": int(k_l01)},
        "aspect": {
            "ad": aspect_ad,
            "fd": aspect_fd,
            "abs_err": abs(aspect_ad - aspect_fd),
            "rel_err": abs(aspect_ad - aspect_fd) / max(1.0e-14, abs(aspect_fd)),
            "grad_time_s": aspect_ad_s,
            "fd_time_s": aspect_fd_s,
        },
        "lambda_scalar": {
            "ad": lambda_ad,
            "fd": lambda_fd,
            "abs_err": abs(lambda_ad - lambda_fd),
            "rel_err": abs(lambda_ad - lambda_fd) / max(1.0e-14, abs(lambda_fd)),
            "grad_time_s": lambda_ad_s,
            "fd_time_s": lambda_fd_s,
        },
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

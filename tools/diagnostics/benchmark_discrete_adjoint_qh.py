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
from vmec_jax.boundary import BoundaryCoeffs, boundary_from_indata
from vmec_jax.config import load_config
from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape
from vmec_jax.discrete_adjoint import preconditioned_force_channels_from_raw_forces
from vmec_jax.discrete_adjoint import preconditioned_force_channels_from_rz_output
from vmec_jax.discrete_adjoint import raw_force_residual_from_state
from vmec_jax.discrete_adjoint import strict_update_accepted_step
from vmec_jax.discrete_adjoint import strict_update_one_step_from_state
from vmec_jax.discrete_adjoint import strict_update_velocity_state_advance
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.implicit import solve_fixed_boundary_state_implicit_vmec_residual
from vmec_jax.init_guess import extract_axis_override_from_state, initial_guess_from_boundary
from vmec_jax.solve import solve_fixed_boundary_residual_iter
from vmec_jax.state import pack_state
from vmec_jax.kernels.tomnsp import TomnspsRZL
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
        raise SystemExit("This script requires JAX (pip install -e .).")
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
    axis_override = extract_axis_override_from_state(state_guess, static)

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

    def _projected_mid_rcos(alpha):
        boundary_alpha = BoundaryCoeffs(
            R_cos=jnp.asarray(edge_Rcos0).at[k_rc01].set(alpha),
            R_sin=jnp.asarray(edge_Rsin0),
            Z_cos=jnp.asarray(edge_Zcos0),
            Z_sin=jnp.asarray(edge_Zsin0),
        )
        state = initial_guess_from_boundary(
            static,
            boundary_alpha,
            indata,
            vmec_project=True,
            axis_override=axis_override,
        )
        return state.Rcos[int(args.surface_index), int(k_rc01)]

    projected_axis_ad = float(np.asarray(jax.grad(_projected_mid_rcos)(alpha0)))
    projected_axis_fd = float(
        (
            np.asarray(_projected_mid_rcos(alpha0 + args.eps))
            - np.asarray(_projected_mid_rcos(alpha0 - args.eps))
        )
        / (2.0 * float(args.eps))
    )

    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
    )
    t0 = time.perf_counter()
    direct = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        max_iter=int(args.max_iter),
        light_history=False,
        resume_state_mode="full",
        **common_kwargs,
    )
    direct_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=int(args.max_iter),
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
    )
    tape_s = time.perf_counter() - t0
    final_state_diff = float(
        np.max(np.abs(np.asarray(pack_state(direct.state)) - np.asarray(tape.packed_states[-1])))
    ) if tape.packed_states.shape[0] else float("nan")
    strict_update_runtime_s = float("nan")
    strict_update_state_diff = float("nan")
    residual_block_runtime_s = float("nan")
    residual_block_force_diff = float("nan")
    raw_force_block_runtime_s = float("nan")
    raw_force_block_force_diff = float("nan")
    force_block_runtime_s = float("nan")
    force_block_force_diff = float("nan")
    strict_step_runtime_s = float("nan")
    strict_step_state_diff = float("nan")
    one_step_axis_ad = float("nan")
    one_step_axis_fd = float("nan")
    if int(args.max_iter) >= 1 and tape.resume_states:
        resume = direct.diagnostics.get("resume_state")
        if resume is not None and len(np.asarray(direct.diagnostics.get("dt_eff_history", []))) >= 1:
            t0 = time.perf_counter()
            reconstructed = strict_update_velocity_state_advance(
                state_guess,
                static,
                dt_eff=float(np.asarray(direct.diagnostics["dt_eff_history"])[0]),
                vRcc=resume["vRcc"],
                vRss=resume["vRss"],
                vZsc=resume["vZsc"],
                vZcs=resume["vZcs"],
                vLsc=resume["vLsc"],
                vLcs=resume["vLcs"],
                edge_Rcos=np.asarray(state_guess.Rcos)[-1, :],
                edge_Rsin=np.asarray(state_guess.Rsin)[-1, :],
                edge_Zcos=np.asarray(state_guess.Zcos)[-1, :],
                edge_Zsin=np.asarray(state_guess.Zsin)[-1, :],
            )
            strict_update_runtime_s = time.perf_counter() - t0
            strict_update_state_diff = float(
                np.max(np.abs(np.asarray(pack_state(reconstructed)) - np.asarray(pack_state(direct.state))))
            )
        if tape.step_traces:
            trace = tape.step_traces[0]
            t0 = time.perf_counter()
            residual_out = raw_force_residual_from_state(
                trace["state_pre"],
                static,
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=trace["apply_lforbal"],
                include_edge_residual=trace["include_edge_residual"],
                apply_m1_constraints=trace["apply_m1_constraints"],
                zero_m1=trace["zero_m1"],
            )
            residual_block_runtime_s = time.perf_counter() - t0
            residual_block_force_diff = float(
                np.max(np.abs(np.asarray(residual_out["frzl"].flsc) - np.asarray(trace["frzl_flsc"])))
            )
            frzl = TomnspsRZL(
                frcc=residual_out["frzl"].frcc,
                frss=residual_out["frzl"].frss,
                fzsc=residual_out["frzl"].fzsc,
                fzcs=residual_out["frzl"].fzcs,
                flsc=residual_out["frzl"].flsc,
                flcs=residual_out["frzl"].flcs,
                frsc=residual_out["frzl"].frsc,
                frcs=residual_out["frzl"].frcs,
                fzcc=residual_out["frzl"].fzcc,
                fzss=residual_out["frzl"].fzss,
                flcc=residual_out["frzl"].flcc,
                flss=residual_out["frzl"].flss,
            )
            t0 = time.perf_counter()
            raw_force_out = preconditioned_force_channels_from_raw_forces(
                frzl=frzl,
                mats=trace["precond_mats"],
                jmax=trace["precond_jmax"],
                cfg=static.cfg,
                lam_prec=trace["lam_prec"],
                w_mode_mn=trace["w_mode_mn"],
                lambda_update_scale=trace["lambda_update_scale"],
            )
            raw_force_block_runtime_s = time.perf_counter() - t0
            raw_force_block_force_diff = float(
                np.max(np.abs(np.asarray(raw_force_out["flsc_u"]) - np.asarray(trace["flsc_u"])))
            )
            frzl_rz = TomnspsRZL(
                frcc=trace["frzl_rz_frcc"],
                frss=trace["frzl_rz_frss"],
                fzsc=trace["frzl_rz_fzsc"],
                fzcs=trace["frzl_rz_fzcs"],
                flsc=trace["frzl_rz_flsc"],
                flcs=trace["frzl_rz_flcs"],
                frsc=trace["frzl_rz_frsc"],
                frcs=trace["frzl_rz_frcs"],
                fzcc=trace["frzl_rz_fzcc"],
                fzss=trace["frzl_rz_fzss"],
                flcc=trace["frzl_rz_flcc"],
                flss=trace["frzl_rz_flss"],
            )
            t0 = time.perf_counter()
            force_out = preconditioned_force_channels_from_rz_output(
                frzl_rz=frzl_rz,
                lam_prec=trace["lam_prec"],
                w_mode_mn=trace["w_mode_mn"],
                lambda_update_scale=trace["lambda_update_scale"],
            )
            force_block_runtime_s = time.perf_counter() - t0
            force_block_force_diff = float(
                np.max(np.abs(np.asarray(force_out["flsc_u"]) - np.asarray(trace["flsc_u"])))
            )
            t0 = time.perf_counter()
            out = strict_update_accepted_step(
                trace["state_pre"],
                static,
                dt_eff=trace["dt_eff"],
                b1=trace["b1"],
                fac=trace["fac"],
                force_scale=trace["force_scale"],
                flip_sign=trace["flip_sign"],
                vRcc_before=trace["vRcc_before"],
                vRss_before=trace["vRss_before"],
                vZsc_before=trace["vZsc_before"],
                vZcs_before=trace["vZcs_before"],
                vLsc_before=trace["vLsc_before"],
                vLcs_before=trace["vLcs_before"],
                frcc_u=force_out["frcc_u"],
                frss_u=force_out["frss_u"],
                fzsc_u=force_out["fzsc_u"],
                fzcs_u=force_out["fzcs_u"],
                flsc_u=force_out["flsc_u"],
                flcs_u=force_out["flcs_u"],
                max_update_rms=trace["max_update_rms_pre"],
                limit_update_rms=trace["limit_update_rms"],
                divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
            )
            strict_step_runtime_s = time.perf_counter() - t0
            strict_step_state_diff = float(
                np.max(np.abs(np.asarray(pack_state(out["state_post"])) - np.asarray(pack_state(trace["state_post"]))))
            )
            def _one_step_lambda(alpha):
                boundary_alpha = BoundaryCoeffs(
                    R_cos=jnp.asarray(edge_Rcos0).at[k_rc01].set(alpha),
                    R_sin=jnp.asarray(edge_Rsin0),
                    Z_cos=jnp.asarray(edge_Zcos0),
                    Z_sin=jnp.asarray(edge_Zsin0),
                )
                state_pre = initial_guess_from_boundary(
                    static,
                    boundary_alpha,
                    indata,
                    vmec_project=True,
                    axis_override=axis_override,
                )
                step_out = strict_update_one_step_from_state(
                    state_pre,
                    static,
                    wout_like=trace["wout_like"],
                    trig=trace["trig"],
                    apply_lforbal=trace["apply_lforbal"],
                    include_edge_residual=trace["include_edge_residual"],
                    apply_m1_constraints=trace["apply_m1_constraints"],
                    zero_m1=trace["zero_m1"],
                    lambda_update_scale=trace["lambda_update_scale"],
                    dt_eff=trace["dt_eff"],
                    b1=trace["b1"],
                    fac=trace["fac"],
                    force_scale=trace["force_scale"],
                    flip_sign=trace["flip_sign"],
                    vRcc_before=trace["vRcc_before"],
                    vRss_before=trace["vRss_before"],
                    vZsc_before=trace["vZsc_before"],
                    vZcs_before=trace["vZcs_before"],
                    vLsc_before=trace["vLsc_before"],
                    vLcs_before=trace["vLcs_before"],
                    max_update_rms=trace["max_update_rms_pre"],
                    limit_update_rms=trace["limit_update_rms"],
                    divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
                )
                return step_out["step"]["state_post"].Lsin[int(args.surface_index), int(k_l01)]

            one_step_axis_ad = float(np.asarray(jax.grad(_one_step_lambda)(alpha0)))
            one_step_axis_fd = float(
                (
                    np.asarray(_one_step_lambda(alpha0 + args.eps))
                    - np.asarray(_one_step_lambda(alpha0 - args.eps))
                )
                / (2.0 * float(args.eps))
            )

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
        "projected_initial_guess": {
            "ad": projected_axis_ad,
            "fd": projected_axis_fd,
            "abs_err": abs(projected_axis_ad - projected_axis_fd),
            "rel_err": abs(projected_axis_ad - projected_axis_fd) / max(1.0e-14, abs(projected_axis_fd)),
        },
        "replay": {
            "direct_runtime_s": direct_s,
            "checkpoint_tape_runtime_s": tape_s,
            "checkpoint_count": int(tape.packed_states.shape[0]),
            "trace_len": int(tape.trace.iter2.shape[0]),
            "final_state_linf": final_state_diff,
            "residual_block_runtime_s": residual_block_runtime_s,
            "residual_block_flsc_linf": residual_block_force_diff,
            "raw_force_block_runtime_s": raw_force_block_runtime_s,
            "raw_force_block_flsc_linf": raw_force_block_force_diff,
            "force_block_runtime_s": force_block_runtime_s,
            "force_block_flsc_linf": force_block_force_diff,
            "strict_update_block_runtime_s": strict_update_runtime_s,
            "strict_update_block_state_linf": strict_update_state_diff,
            "strict_step_block_runtime_s": strict_step_runtime_s,
            "strict_step_block_state_linf": strict_step_state_diff,
            "one_step_boundary_ad": one_step_axis_ad,
            "one_step_boundary_fd": one_step_axis_fd,
        },
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

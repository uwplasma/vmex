#!/usr/bin/env python
# ruff: noqa: E402
"""Shared finite-beta stage-one optimization utilities for examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmec_jax as vj
from vmec_jax._compat import enable_x64, jnp
from vmec_jax.config import config_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.optimization import rebuild_indata_with_resolution
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.wout import equilibrium_iota_profiles_from_state


enable_x64(True)


@dataclass(frozen=True)
class FiniteBetaStage1Config:
    input_file: Path
    output_dir: Path
    objective_kind: str
    max_mode: int
    max_nfev: int
    continuation_nfev: int
    vmec_mpol: int
    vmec_ntor: int
    helicity_m: int = 1
    helicity_n: int = -1
    target_aspect: float = 5.0
    min_iota: float = 1.0
    min_average_iota: float = 1.0
    max_iota: float = 1.9
    target_volavgB: float = 5.86461221551616
    target_beta: float = 0.025
    aspect_weight: float = 1.0e3
    iota_weight: float = 1.0e5
    max_iota_weight: float = 1.0e8
    volavgB_weight: float = 1.0e3
    beta_weight: float = 1.0e1
    field_weight: float = 1.0e3
    use_ess: bool = True
    ess_alpha: float = 2.5
    use_mode_continuation: bool = True
    method: str = "scipy"
    ftol: float = 1.0e-3
    gtol: float = 1.0e-3
    xtol: float = 1.0e-3
    inner_max_iter: int = 0
    inner_ftol: float = 0.0
    trial_max_iter: int = 300
    trial_ftol: float = 1.0e-10
    solver_device: str | None = None
    surfaces: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 10, endpoint=True))
    qi_mboz: int = 18
    qi_nboz: int = 18
    qi_nphi: int = 96
    qi_nalpha: int = 24
    qi_n_bounce: int = 32
    qi_softness: float = 30.0
    qi_profile_weight: float = 0.15


def _pressure_profile(indata, static):
    prof = vj.eval_profiles(indata, jnp.asarray(static.s))
    return jnp.asarray(prof.get("pressure", jnp.zeros_like(jnp.asarray(static.s))))


def _mean_abs_iota(state, *, static, indata, signgs: int):
    _chips, _iotas, iotaf = equilibrium_iota_profiles_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(signgs),
    )
    iotaf = jnp.asarray(iotaf, dtype=jnp.float64)
    return jnp.mean(jnp.abs(iotaf[1:])) if int(iotaf.shape[0]) > 1 else jnp.asarray(0.0)


def _build_stage(cfg: FiniteBetaStage1Config, base_cfg, base_indata, max_mode: int):
    stage_static = vj.build_static(base_cfg)
    stage_boundary = vj.boundary_from_indata(base_indata, stage_static.modes, apply_m1_constraint=False)
    stage_indata, stage_static, stage_boundary = vj.extend_boundary_for_max_mode(
        base_indata,
        stage_static,
        stage_boundary,
        max_mode,
    )
    stage_boundary_input = vj.boundary_input_from_indata(stage_indata, stage_static.modes)
    stage_specs = vj.boundary_param_specs(
        stage_boundary_input,
        stage_static.modes,
        max_mode=max_mode,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    stage_guess = initial_guess_from_boundary(stage_static, stage_boundary, stage_indata, vmec_project=True)
    stage_geom = eval_geom(stage_guess, stage_static)
    stage_signgs = int(signgs_from_sqrtg(np.asarray(stage_geom.sqrtg), axis_index=1))
    stage_flux = vj.flux_profiles_from_indata(stage_indata, stage_static.s, signgs=stage_signgs)
    stage_pressure = _pressure_profile(stage_indata, stage_static)
    global_targets = vj.FiniteBetaTargets(
        aspect_ratio=cfg.target_aspect,
        min_iota=cfg.min_iota,
        min_average_iota=cfg.min_average_iota,
        max_iota=cfg.max_iota,
        volavgB=cfg.target_volavgB,
        beta_total=cfg.target_beta,
        aspect_weight=cfg.aspect_weight,
        iota_weight=cfg.iota_weight,
        max_iota_weight=cfg.max_iota_weight,
        volavgB_weight=cfg.volavgB_weight,
        beta_weight=cfg.beta_weight,
    )

    if cfg.objective_kind == "qi":
        from booz_xform_jax import prepare_booz_xform_constants
        from vmec_jax.modes import nyquist_mode_table_from_grid, vmec_mode_table
        from vmec_jax.quasi_isodynamic import _nearest_half_mesh_indices, quasi_isodynamic_residual_from_state

        main_modes = vmec_mode_table(int(stage_static.cfg.mpol), int(stage_static.cfg.ntor))
        nyq_modes = nyquist_mode_table_from_grid(
            mpol=int(stage_static.cfg.mpol),
            ntor=int(stage_static.cfg.ntor),
            ntheta=int(stage_static.cfg.ntheta),
            nzeta=int(stage_static.cfg.nzeta),
        )
        booz_constants, booz_grids = prepare_booz_xform_constants(
            nfp=int(stage_static.cfg.nfp),
            mboz=cfg.qi_mboz,
            nboz=cfg.qi_nboz,
            asym=bool(stage_static.cfg.lasym),
            xm=np.asarray(main_modes.m, dtype=int),
            xn=np.asarray(main_modes.n * int(stage_static.cfg.nfp), dtype=int),
            xm_nyq=np.asarray(nyq_modes.m, dtype=int),
            xn_nyq=np.asarray(nyq_modes.n * int(stage_static.cfg.nfp), dtype=int),
        )
        surface_indices = _nearest_half_mesh_indices(
            cfg.surfaces,
            n_half=max(int(np.asarray(stage_static.s).shape[0]) - 1, 1),
        )

        def field_eval(state):
            return quasi_isodynamic_residual_from_state(
                state=state,
                static=stage_static,
                indata=stage_indata,
                signgs=stage_signgs,
                flux_local=stage_flux,
                prof_local={"pressure": stage_pressure},
                pressure_local=stage_pressure,
                surfaces=cfg.surfaces,
                mboz=cfg.qi_mboz,
                nboz=cfg.qi_nboz,
                nphi=cfg.qi_nphi,
                nalpha=cfg.qi_nalpha,
                n_bounce=cfg.qi_n_bounce,
                softness=cfg.qi_softness,
                profile_weight=cfg.qi_profile_weight,
                jit_booz=False,
                booz_constants=booz_constants,
                booz_grids=booz_grids,
                surface_indices=surface_indices,
            )

    else:

        def field_eval(state):
            return quasisymmetry_ratio_residual_from_state(
                state=state,
                static=stage_static,
                indata=stage_indata,
                signgs=stage_signgs,
                flux_local=stage_flux,
                prof_local={"pressure": stage_pressure},
                pressure_local=stage_pressure,
                surfaces=cfg.surfaces,
                helicity_m=cfg.helicity_m,
                helicity_n=cfg.helicity_n,
            )

    def residuals_from_state(state):
        global_res = vj.finite_beta_global_residuals_from_state(
            state=state,
            static=stage_static,
            indata=stage_indata,
            signgs=stage_signgs,
            targets=global_targets,
        )
        field = field_eval(state)
        return jnp.concatenate(
            [
                global_res,
                jnp.asarray(field["residuals1d"], dtype=jnp.float64) * float(cfg.field_weight),
            ]
        )

    residuals_from_state._n_non_qs = 6
    residuals_from_state._qs_total_from_state = (
        lambda state: float(cfg.field_weight) ** 2 * float(field_eval(state)["total"])
    )

    opt = vj.FixedBoundaryExactOptimizer(
        stage_static,
        stage_indata,
        stage_boundary,
        stage_specs,
        residuals_from_state,
        boundary_input=stage_boundary_input,
        inner_max_iter=cfg.inner_max_iter,
        inner_ftol=cfg.inner_ftol,
        trial_max_iter=cfg.trial_max_iter,
        trial_ftol=cfg.trial_ftol,
        solver_device=cfg.solver_device,
    )
    x_scale = vj.create_x_scale(stage_specs, alpha=cfg.ess_alpha) if cfg.use_ess else np.ones(len(stage_specs))
    iota_fn = lambda state: float(_mean_abs_iota(state, static=stage_static, indata=stage_indata, signgs=stage_signgs))
    return stage_specs, opt, x_scale, iota_fn


def run_stage1(cfg: FiniteBetaStage1Config):
    print(f"Loading {cfg.input_file.name} ...")
    base_cfg, indata = vj.load_config(str(cfg.input_file))
    indata = rebuild_indata_with_resolution(indata, mpol=cfg.vmec_mpol, ntor=cfg.vmec_ntor)
    base_cfg = config_from_indata(indata)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stage_modes = (
        list(range(1, int(cfg.max_mode) + 1))
        if (cfg.use_mode_continuation and int(cfg.max_mode) > 1)
        else [int(cfg.max_mode)]
    )
    stage_records = []
    params_stage = None
    prev_specs = None
    final_opt = None
    final_params0 = None
    final_result = None

    for stage_mode in stage_modes:
        stage_specs, opt, x_scale, iota_fn = _build_stage(cfg, base_cfg, indata, stage_mode)
        params0 = (
            np.zeros(len(stage_specs), dtype=float)
            if params_stage is None
            else vj.lift_boundary_params(prev_specs, params_stage, stage_specs)
        )
        nfev = int(cfg.max_nfev) if stage_mode == int(cfg.max_mode) else int(cfg.continuation_nfev)
        print(f"Running finite-beta {cfg.objective_kind.upper()} stage mode={stage_mode}, nfev={nfev}")
        result = opt.run(
            params0,
            method=cfg.method,
            max_nfev=nfev,
            ftol=cfg.ftol,
            gtol=cfg.gtol,
            xtol=cfg.xtol,
            x_scale=x_scale,
            verbose=2,
            iota_fn=iota_fn,
            target_iota=cfg.min_average_iota,
            target_aspect=cfg.target_aspect,
        )
        stage_dir = cfg.output_dir / f"stage_mode{stage_mode}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        opt.save_input(stage_dir / "input.initial", params0)
        opt.save_input(stage_dir / "input.final", result["x"])
        opt.save_history(stage_dir / "history.json", result)
        stage_records.append((stage_mode, opt, params0, result))
        params_stage = result["x"]
        prev_specs = stage_specs
        final_opt = opt
        final_params0 = params0
        final_result = result

    assert final_opt is not None and final_params0 is not None and final_result is not None
    final_opt.save_input(cfg.output_dir / "input.initial", final_params0)
    final_opt.save_wout(cfg.output_dir / "wout_initial.nc", final_params0, state=final_result.get("_state_initial"))
    final_opt.save_input(cfg.output_dir / "input.final", final_result["x"])
    final_opt.save_wout(cfg.output_dir / "wout_final.nc", final_result["x"], state=final_result.get("_state_final"))
    final_opt.save_history(cfg.output_dir / "history.json", final_result)
    try:
        vj.plot_qh_optimization(
            cfg.output_dir / "wout_initial.nc",
            cfg.output_dir / "wout_final.nc",
            cfg.output_dir / "history.json",
            outdir=cfg.output_dir,
        )
    except Exception as exc:
        (cfg.output_dir / "plotting_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")

    hist = final_result["_history_dump"]
    print(f"Final objective: {hist['objective_final']:.6e}")
    print(f"Final aspect:    {hist['aspect_final']:.6f}")
    if hist["history"] and "iota" in hist["history"][-1]:
        print(f"Final |iota|:    {hist['history'][-1]['iota']:.6f}")
    return final_result

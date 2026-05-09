#!/usr/bin/env python3
"""Profile vmec_jax exact fixed-boundary optimization callbacks.

This is intentionally a diagnostics tool, not a tutorial example.  It mirrors
the QA/QH fixed-resolution examples but keeps the run short and prints a timing
breakdown from :class:`vmec_jax.FixedBoundaryExactOptimizer`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--problem", choices=("qa", "qh"), default="qa")
    p.add_argument("--max-mode", type=int, default=1)
    p.add_argument("--max-nfev", type=int, default=3)
    p.add_argument("--inner-max-iter", type=int, default=0)
    p.add_argument("--inner-ftol", type=float, default=0.0)
    p.add_argument("--trial-max-iter", type=int, default=300)
    p.add_argument("--trial-ftol", type=float, default=1e-10)
    p.add_argument("--solver-device", choices=("auto", "cpu", "gpu", "default"), default="auto")
    p.add_argument("--mpol", type=int, default=5)
    p.add_argument("--ntor", type=int, default=5)
    p.add_argument(
        "--stellarator-asymmetric",
        action="store_true",
        help="Set LASYM=T, include RBS/ZBC boundary parameters, and seed zero asymmetric modes.",
    )
    p.add_argument(
        "--asymmetric-seed",
        type=float,
        default=1.0e-7,
        help="Seed value applied to zero RBS/ZBC optimization parameters when --stellarator-asymmetric is set.",
    )
    p.add_argument("--ess", action="store_true")
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument(
        "--method",
        choices=(
            "scipy",
            "scipy_matrix_free",
            "gauss_newton",
            "lbfgs_adjoint",
            "scalar_trust",
        ),
        default="scipy",
    )
    p.add_argument("--scipy-tr-solver", choices=("lsmr", "exact", "none"), default="lsmr")
    p.add_argument("--lsmr-maxiter", type=int, default=0)
    p.add_argument(
        "--lbfgs-step-bound",
        type=float,
        default=0.01,
        help="Scaled-space half-width for method=lbfgs_adjoint.",
    )
    p.add_argument(
        "--scalar-step-bound",
        type=float,
        default=0.01,
        help="Initial/max scaled-space trust radius for method=scalar_trust.",
    )
    p.add_argument("--trace-outdir", type=str, default="")
    p.add_argument(
        "--device-memory-profile-out",
        type=str,
        default="",
        help=(
            "Optional path for jax.profiler.save_device_memory_profile(). "
            "Use with pprof/XProf to inspect live device buffers after the run."
        ),
    )
    p.add_argument("--json-out", type=str, default="")
    p.add_argument(
        "--callback",
        choices=("trial", "exact", "jacobian", "gradient", "linear", "run"),
        default="run",
        help=(
            "Profile one callback family and exit. 'run' executes the short "
            "optimizer run controlled by --max-nfev."
        ),
    )
    p.add_argument("--repeats", type=int, default=1, help="Callback repetitions for --callback modes.")
    p.add_argument(
        "--perturb-scale",
        type=float,
        default=0.0,
        help=(
            "When profiling callback modes, use deterministic distinct parameter "
            "vectors with this RMS perturbation scale. This measures realistic "
            "new accepted-point tape/replay cost instead of same-point cache hits."
        ),
    )
    p.add_argument(
        "--perturb-seed",
        type=int,
        default=1234,
        help="Random seed for --perturb-scale callback points.",
    )
    p.add_argument(
        "--clear-between-repeats",
        action="store_true",
        help="Clear exact optimizer caches between callback repetitions.",
    )
    p.add_argument(
        "--gradient-only",
        action="store_true",
        help="Profile one exact reverse-adjoint scalar-gradient callback instead of running the optimizer.",
    )
    p.add_argument(
        "--check-gradient",
        action="store_true",
        help="Also build the dense exact Jacobian and compare the reverse gradient to J.T @ r.",
    )
    p.add_argument(
        "--check-linear-operator",
        action="store_true",
        help="Build the dense exact Jacobian and compare matrix-free Jv/J.Tv products at the initial point.",
    )
    p.add_argument(
        "--linear-operator-repeats",
        type=int,
        default=1,
        help="Number of same-point matvec/rmatvec products to time in --check-linear-operator mode.",
    )
    p.add_argument(
        "--trial-use-scan",
        action="store_true",
        help="Force relaxed trial residual solves onto the lax.scan path; exact adjoint solves remain trace-capable non-scan.",
    )
    p.add_argument(
        "--trace-callbacks",
        action="store_true",
        help="Include SciPy residual/Jacobian callback source timings in the JSON history.",
    )
    p.add_argument(
        "--run-repeats",
        type=int,
        default=1,
        help=(
            "Repeat the short optimizer run in the same Python process. Between "
            "repeats, point/tape caches are cleared while compiled JAX/XLA "
            "executables remain warm. This separates in-process warm runtime "
            "from first-run compilation overhead."
        ),
    )
    p.add_argument(
        "--vmec-timing",
        action="store_true",
        help="Enable VMEC_JAX_TIMING so exact tape profiles include solver phase timings.",
    )
    return p.parse_args()


def _iota_mean_fn(vj, state, *, static, indata, signgs):
    chips, iotas, iotaf = vj.equilibrium_iota_profiles_from_state(
        state=state, static=static, indata=indata, signgs=signgs
    )
    del chips, iotaf
    iotas = np.asarray(iotas, dtype=float)
    return 0.0 if iotas.size <= 1 else float(np.mean(iotas[1:]))


def _print_profile(profile: dict[str, dict]) -> None:
    rows = sorted(
        (
            name,
            int(rec.get("count", 0)),
            float(rec.get("wall_time_s", 0.0)),
            float(rec.get("mean_wall_time_s", 0.0)),
        )
        for name, rec in profile.items()
    )
    rows.sort(key=lambda row: row[2], reverse=True)
    print("\nCallback timing profile:")
    print(f"{'name':48s} {'count':>7s} {'total_s':>12s} {'mean_s':>12s}")
    for name, count, total, mean in rows:
        print(f"{name:48s} {count:7d} {total:12.3f} {mean:12.3f}")


def _runtime_info() -> dict[str, object]:
    try:
        import jax

        return {
            "jax_version": getattr(jax, "__version__", None),
            "default_backend": str(jax.default_backend()),
            "devices": [str(device) for device in jax.devices()],
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"error": repr(exc)}


def _clear_optimizer_point_caches(opt) -> None:
    """Clear solved-state/tape caches without dropping compiled executables."""
    opt._exact_cache.clear()
    opt._exact_state_cache.clear()
    opt._trial_residual_cache.clear()
    opt._initial_tangent_cache.clear()
    opt._last_jacobian_residual = None


def main() -> int:
    args = _parse_args()
    if args.gradient_only:
        args.callback = "gradient"
    if args.vmec_timing:
        os.environ["VMEC_JAX_TIMING"] = "1"

    import vmec_jax as vj
    from vmec_jax._compat import enable_x64
    from vmec_jax.config import config_from_indata
    from vmec_jax.optimization import rebuild_indata_with_resolution

    enable_x64(True)

    root = Path(__file__).resolve().parents[2]
    input_file = root / "examples" / "data" / (
        "input.nfp2_QA" if args.problem == "qa" else "input.nfp4_QH_warm_start"
    )
    helicity_n = 0 if args.problem == "qa" else -1
    target_aspect = 6.0 if args.problem == "qa" else 7.0
    target_iota = 0.41 if args.problem == "qa" else None

    cfg, indata = vj.load_config(str(input_file))
    indata = rebuild_indata_with_resolution(indata, mpol=args.mpol, ntor=args.ntor)
    if args.stellarator_asymmetric:
        scalars = dict(indata.scalars)
        scalars["LASYM"] = True
        indexed = {key: dict(value) for key, value in indata.indexed.items()}
        from vmec_jax.namelist import InData

        indata = InData(scalars=scalars, indexed=indexed, source_path=indata.source_path)
    cfg = config_from_indata(indata)
    static = vj.build_static(cfg)
    boundary = vj.boundary_from_indata(indata, static.modes, apply_m1_constraint=False)
    indata, static, boundary = vj.extend_boundary_for_max_mode(
        indata, static, boundary, int(args.max_mode)
    )
    boundary_input = vj.boundary_input_from_indata(indata, static.modes)
    specs = vj.boundary_param_specs(
        boundary_input,
        static.modes,
        max_mode=int(args.max_mode),
        min_coeff=0.0,
        include=("rc", "zs", "rs", "zc") if args.stellarator_asymmetric else ("rc", "zs"),
        fix=("rc00",),
    )
    residuals_fn = vj.make_qs_residuals_fn(
        static,
        indata,
        helicity_m=1,
        helicity_n=helicity_n,
        target_aspect=target_aspect,
        target_iota=target_iota,
        surfaces=np.arange(0.0, 1.01, 0.1),
        aspect_weight=1.0,
        iota_weight=1.0,
        qs_weight=1.0,
    )
    opt = vj.FixedBoundaryExactOptimizer(
        static,
        indata,
        boundary,
        specs,
        residuals_fn,
        boundary_input=boundary_input,
        inner_max_iter=args.inner_max_iter,
        inner_ftol=args.inner_ftol,
        trial_max_iter=args.trial_max_iter,
        trial_ftol=args.trial_ftol,
        solver_device=args.solver_device,
    )
    if args.trial_use_scan:
        opt._trial_solver_kwargs["use_scan"] = True
    params0 = np.zeros(len(specs))
    if args.stellarator_asymmetric and float(args.asymmetric_seed) != 0.0:
        for index, spec in enumerate(specs):
            if spec.kind in ("rs", "zc"):
                params0[index] = float(args.asymmetric_seed)
    x_scale = vj.create_x_scale(specs, alpha=float(args.alpha)) if args.ess else np.ones(len(specs))

    print(
        f"Problem={args.problem} max_mode={args.max_mode} dofs={len(specs)} "
        f"lasym={args.stellarator_asymmetric} "
        f"inner=({args.inner_max_iter}, {args.inner_ftol:g}) "
        f"trial=({args.trial_max_iter}, {args.trial_ftol:g})"
    )
    print(f"Requested solver_device={args.solver_device} resolved={opt._solver_device_name or 'default'}")
    print(f"Runtime={json.dumps(_runtime_info(), sort_keys=True)}")
    print(f"Initial aspect={opt.aspect_ratio(params0):.6f} qs={opt.quasisymmetry_objective(params0):.6e}")
    opt.clear_caches()
    opt._profile = {}

    if args.callback != "run":
        repeats = max(1, int(args.repeats))
        perturb_scale = float(args.perturb_scale)
        rng = np.random.default_rng(int(args.perturb_seed))
        samples: list[dict[str, object]] = []
        for repeat in range(repeats):
            if repeat > 0 and args.clear_between_repeats:
                opt.clear_caches()
            if perturb_scale > 0.0:
                params = params0 + perturb_scale * rng.standard_normal(params0.shape)
            else:
                params = params0
            t0 = time.perf_counter()
            if args.callback == "trial":
                value = opt.forward_residual_fun(params)
                metric = float(np.linalg.norm(value))
                shape = list(np.asarray(value).shape)
            elif args.callback == "exact":
                value = opt.residual_fun(params)
                metric = float(np.linalg.norm(value))
                shape = list(np.asarray(value).shape)
            elif args.callback == "jacobian":
                value = opt.jacobian_fun(params)
                metric = float(np.linalg.norm(value))
                shape = list(np.asarray(value).shape)
            elif args.callback == "gradient":
                cost, grad = opt.objective_and_gradient_fun(params)
                metric = float(np.linalg.norm(grad))
                shape = [int(np.asarray(grad).size)]
                samples.append(
                    {
                        "repeat": repeat,
                        "wall_time_s": time.perf_counter() - t0,
                        "cost": float(cost),
                        "metric_norm": metric,
                        "param_step_norm": float(np.linalg.norm(params - params0)),
                        "shape": shape,
                    }
                )
                continue
            elif args.callback == "linear":
                op = opt.residual_linear_operator(params)
                direction = np.ones(len(specs), dtype=float)
                cotangent = np.ones(op.shape[0], dtype=float)
                jv = op.matvec(direction)
                jtw = op.rmatvec(cotangent)
                metric = float(np.linalg.norm(jv) + np.linalg.norm(jtw))
                shape = [int(op.shape[0]), int(op.shape[1])]
            else:  # pragma: no cover - guarded by argparse
                raise ValueError(args.callback)
            samples.append(
                {
                    "repeat": repeat,
                    "wall_time_s": time.perf_counter() - t0,
                    "metric_norm": metric,
                    "param_step_norm": float(np.linalg.norm(params - params0)),
                    "shape": shape,
                }
            )
        profile = opt._profile_dump()
        print(f"\nCallback={args.callback} repeats={repeats}")
        for sample in samples:
            print(
                f"  repeat={sample['repeat']} wall={float(sample['wall_time_s']):.3f}s "
                f"norm={float(sample['metric_norm']):.6e} "
                f"||dx||={float(sample['param_step_norm']):.3e} "
                f"shape={sample['shape']}"
            )
        _print_profile(profile)
        if args.json_out:
            out = Path(args.json_out).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "problem": args.problem,
                        "max_mode": int(args.max_mode),
                        "dofs": len(specs),
                        "callback": args.callback,
                        "perturb_scale": perturb_scale,
                        "perturb_seed": int(args.perturb_seed),
                        "solver_device_requested": args.solver_device,
                        "solver_device_resolved": opt._solver_device_name or "default",
                        "runtime": _runtime_info(),
                        "samples": samples,
                        "profile": profile,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Wrote {out}")
        return 0

    if args.check_linear_operator:
        rng = np.random.default_rng(1234)
        jac = opt.jacobian_fun(params0)
        op = opt.residual_linear_operator(params0)
        v = rng.standard_normal(jac.shape[1])
        w = rng.standard_normal(jac.shape[0])
        jv_ref = np.asarray(jac, dtype=float) @ v
        jv_op = op.matvec(v)
        vmat = rng.standard_normal((jac.shape[1], min(3, jac.shape[1])))
        jvmat_ref = np.asarray(jac, dtype=float) @ vmat
        jvmat_op = op.matmat(vmat)
        jtw_ref = np.asarray(jac, dtype=float).T @ w
        jtw_op = op.rmatvec(w)
        for _ in range(max(1, int(args.linear_operator_repeats)) - 1):
            _ = op.matvec(v)
            _ = op.rmatvec(w)
        jv_err = float(np.linalg.norm(jv_op - jv_ref) / max(np.linalg.norm(jv_ref), 1.0))
        jvmat_err = float(np.linalg.norm(jvmat_op - jvmat_ref) / max(np.linalg.norm(jvmat_ref), 1.0))
        jtw_err = float(np.linalg.norm(jtw_op - jtw_ref) / max(np.linalg.norm(jtw_ref), 1.0))
        print(f"LinearOperator check: rel ||Jv - dense Jv||={jv_err:.6e}")
        print(f"LinearOperator check: rel ||JX - dense JX||={jvmat_err:.6e}")
        print(f"LinearOperator check: rel ||J.Tw - dense J.Tw||={jtw_err:.6e}")
        _print_profile(opt._profile_dump())
        return 0

    if args.gradient_only:
        cost, grad = opt.objective_and_gradient_fun(params0)
        print(
            f"\nReverse-gradient callback: cost={cost:.6e} "
            f"||grad||={float(np.linalg.norm(grad)):.6e}"
        )
        if args.check_gradient:
            res = opt.residual_fun(params0)
            jac = opt.jacobian_fun(params0)
            dense_grad = np.asarray(jac, dtype=float).T @ np.asarray(res, dtype=float)
            abs_err = float(np.linalg.norm(grad - dense_grad))
            rel_err = abs_err / max(float(np.linalg.norm(dense_grad)), 1.0)
            print(
                f"Gradient check: ||g_adj - J.T r||={abs_err:.6e} "
                f"relative={rel_err:.6e}"
            )
        profile = opt._profile_dump()
        _print_profile(profile)
        if args.json_out:
            out = Path(args.json_out).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "problem": args.problem,
                        "max_mode": int(args.max_mode),
                        "dofs": len(specs),
                        "cost": float(cost),
                        "gradient_norm": float(np.linalg.norm(grad)),
                        "profile": profile,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Wrote {out}")
        return 0

    trace_out = Path(args.trace_outdir).expanduser().resolve() if args.trace_outdir else None
    if trace_out is not None:
        import jax

        trace_out.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(trace_out))

    run_repeats = max(1, int(args.run_repeats))
    histories: list[dict[str, object]] = []
    result = None
    try:
        tr_solver = None if args.scipy_tr_solver == "none" else args.scipy_tr_solver
        for repeat in range(run_repeats):
            if repeat > 0:
                _clear_optimizer_point_caches(opt)
                opt._profile = {}
            if run_repeats > 1:
                print(f"\n=== optimizer run repeat {repeat + 1}/{run_repeats} ===")
            result = opt.run(
                params0,
                method=args.method,
                max_nfev=args.max_nfev,
                ftol=1e-3,
                gtol=1e-3,
                xtol=1e-3,
                x_scale=x_scale,
                verbose=1,
                iota_fn=(
                    None
                    if args.problem != "qa"
                    else lambda state: _iota_mean_fn(vj, state, static=static, indata=indata, signgs=opt._signgs)
                ),
                target_iota=target_iota,
                target_aspect=target_aspect,
                scipy_tr_solver=tr_solver,
                scipy_lsmr_maxiter=None if args.lsmr_maxiter <= 0 else int(args.lsmr_maxiter),
                lbfgs_step_bound=float(args.lbfgs_step_bound),
                scalar_step_bound=float(args.scalar_step_bound),
                trace_callbacks=args.trace_callbacks,
            )
            hist_repeat = dict(result["_history_dump"])
            hist_repeat["repeat"] = repeat
            histories.append(hist_repeat)
    finally:
        if trace_out is not None:
            import jax

            jax.profiler.stop_trace()
            print(f"Trace written to {trace_out}")

    if result is None:  # pragma: no cover - defensive
        raise RuntimeError("optimizer run did not produce a result")
    hist = result["_history_dump"]
    print(
        f"\nFinal objective={hist['objective_final']:.6e} qs={hist['qs_final']:.6e} "
        f"aspect={hist['aspect_final']:.6f} wall={hist['total_wall_time_s']:.3f}s "
        f"nfev={hist['nfev']} njev={hist['njev']}"
    )
    if "iota_final" in hist:
        print(f"Final iota={hist['iota_final']:.6f} target={hist.get('target_iota'):.6f}")
    _print_profile(hist.get("profile", {}))

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = hist if run_repeats == 1 else {
            "problem": args.problem,
            "max_mode": int(args.max_mode),
            "dofs": len(specs),
            "method": args.method,
            "solver_device_requested": args.solver_device,
            "solver_device_resolved": opt._solver_device_name or "default",
            "runtime": _runtime_info(),
            "run_repeats": run_repeats,
            "runs": histories,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {out}")

    if args.device_memory_profile_out:
        import jax

        mem_out = Path(args.device_memory_profile_out).expanduser().resolve()
        mem_out.parent.mkdir(parents=True, exist_ok=True)
        jax.profiler.save_device_memory_profile(str(mem_out))
        print(f"Device memory profile written to {mem_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

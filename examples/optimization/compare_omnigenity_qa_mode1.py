#!/usr/bin/env python
"""Compare a max-mode-1 QA optimization against omnigenity_optimization.

This is a diagnostic script, not a polished gallery example.  It uses the same
input, objectives, weights, targets, and ESS scale as
``~/local/omnigenity_optimization/QA_fixed_resolution.py`` for the first
``max_mode=1`` stage, then writes enough data to audit objective values,
trajectories, derivative consistency, and runtime hot spots.

Outputs are written under ``examples/optimization/results/`` and are ignored by
git.
"""

from __future__ import annotations

import cProfile
import contextlib
import io
import json
import pstats
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vmec_jax as vj
from vmec_jax.optimization_workflow import build_fixed_boundary_objective_stage


# ---------------------------------------------------------------------------
# User-editable study parameters
# ---------------------------------------------------------------------------

INPUT_FILE = Path("~/local/omnigenity_optimization/inputs/input.nfp20_QA").expanduser()
OUTPUT_DIR = Path("results/omnigenity_compare/qa_mode1")

MAX_MODE = 1
MIN_VMEC_MODE = 5
MAX_NFEV = 6

ASPECT_TARGET = 2.5
IOTA_TARGET = 0.71
LGRADB_THRESHOLD = 0.35
QS_SURFACES = np.arange(0.0, 1.01, 0.1)

ESS_ALPHA = 1.2
FTOL = 1.0e-4
GTOL = 1.0e-4
XTOL = 1.0e-4
FINITE_DIFF_STEP = 1.0e-6

INNER_MAX_ITER = 120
INNER_FTOL = 1.0e-9
TRIAL_MAX_ITER = 120
TRIAL_FTOL = 1.0e-9

RUN_VMEC_JAX = True
RUN_SIMSOPT = True
RUN_FINITE_DIFFERENCE = True

# Leave as ``None`` to use the active Python environment.  Set to a local
# source tree if you want to force a specific SIMSOPT checkout.
SIMSOPT_SOURCE = None


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")
    print(f"wrote {path}")


def _profiled(label: str, outdir: Path, fn):
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    try:
        value = profiler.runcall(fn)
    finally:
        wall_s = time.perf_counter() - t0
        outdir.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(outdir / f"{label}.prof")
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(60)
        (outdir / f"{label}.txt").write_text(stream.getvalue())
    return value, wall_s


def _fd_jacobian(fun, x: np.ndarray, *, step: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x), dtype=float)
    jac = np.zeros((f0.size, x.size), dtype=float)
    for j in range(x.size):
        dx = np.zeros_like(x)
        dx[j] = step
        fp = np.asarray(fun(x + dx), dtype=float)
        fm = np.asarray(fun(x - dx), dtype=float)
        jac[:, j] = (fp - fm) / (2.0 * step)
        print(f"  finite-difference column {j + 1}/{x.size}")
    return jac


def _objective_from_residual(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float)
    return float(np.dot(residual, residual))


def _summarize_jacobian(jac_ad: np.ndarray | None, jac_fd: np.ndarray | None) -> dict:
    if jac_ad is None or jac_fd is None:
        return {}
    diff = np.asarray(jac_ad, dtype=float) - np.asarray(jac_fd, dtype=float)
    denom = max(float(np.linalg.norm(jac_fd)), np.finfo(float).tiny)
    return {
        "relative_frobenius_error": float(np.linalg.norm(diff) / denom),
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "ad_norm": float(np.linalg.norm(jac_ad)),
        "fd_norm": float(np.linalg.norm(jac_fd)),
    }


def _summarize_jacobian_blocks(
    term_report: list[dict],
    jac_ad: np.ndarray | None,
    jac_fd: np.ndarray | None,
) -> list[dict]:
    if jac_ad is None or jac_fd is None:
        return []
    rows = []
    offset = 0
    for term in term_report:
        size = int(term["residual_size"])
        block_ad = np.asarray(jac_ad[offset : offset + size], dtype=float)
        block_fd = np.asarray(jac_fd[offset : offset + size], dtype=float)
        diff = block_ad - block_fd
        denom = max(float(np.linalg.norm(block_fd)), np.finfo(float).tiny)
        rows.append(
            {
                "name": term["name"],
                "residual_size": size,
                "relative_frobenius_error": float(np.linalg.norm(diff) / denom),
                "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
                "ad_norm": float(np.linalg.norm(block_ad)),
                "fd_norm": float(np.linalg.norm(block_fd)),
            }
        )
        offset += size
    return rows


def _vmec_jax_term_report(problem: vj.LeastSquaresProblem, ctx, state) -> list[dict]:
    rows = []
    for term in problem.objective_terms:
        value = np.asarray(term.evaluate(ctx, state), dtype=float).ravel()
        residual = np.asarray(term.residual(ctx, state), dtype=float).ravel()
        total = None
        if term.total is not None:
            total = float(term.total(ctx, state))
        rows.append(
            {
                "name": term.name,
                "value_size": int(value.size),
                "value_norm": float(np.linalg.norm(value)),
                "value_first": value[:8].tolist(),
                "residual_size": int(residual.size),
                "residual_norm": float(np.linalg.norm(residual)),
                "residual_objective": float(np.dot(residual, residual)),
                "reported_total": total,
            }
        )
    return rows


def run_vmec_jax(outdir: Path) -> dict:
    print("\n=== vmec_jax QA max_mode=1 ===")
    vmec = vj.FixedBoundaryVMEC.from_input(
        INPUT_FILE,
        max_mode=MAX_MODE,
        min_vmec_mode=MIN_VMEC_MODE,
        output_dir=outdir,
        project_input_boundary_to_max_mode=True,
    )

    aspect = vj.AspectRatio()
    qs = vj.QuasisymmetryRatioResidual(helicity_m=1, helicity_n=0, surfaces=QS_SURFACES)
    iota = vj.MeanIota()
    lgradb = vj.LgradB(
        threshold=LGRADB_THRESHOLD,
        s_index=-1,
        ntheta=9,
        nphi=7,
    )
    problem = vj.LeastSquaresProblem.from_tuples(
        [
            (aspect.J, ASPECT_TARGET, 1.0),
            (qs.J, 0.0, 1.0),
            (iota.J, IOTA_TARGET, 1.0e2),
            (lgradb.J, 0.0, 1.0),
        ]
    )

    stage = build_fixed_boundary_objective_stage(
        vmec.cfg,
        vmec.indata,
        stage_mode=MAX_MODE,
        objectives=problem.objective_terms,
        include=vmec.include,
        fix=vmec.fix,
        project_input_boundary_to_max_mode=vmec.project_input_boundary_to_max_mode,
        inner_max_iter=INNER_MAX_ITER,
        inner_ftol=INNER_FTOL,
        trial_max_iter=TRIAL_MAX_ITER,
        trial_ftol=TRIAL_FTOL,
    )
    names = vj.boundary_param_names(stage.specs)
    x0 = np.zeros(len(stage.specs), dtype=float)
    x_scale = vj.create_x_scale(stage.specs, alpha=ESS_ALPHA)

    def initial_residual():
        return stage.optimizer.residual_fun(x0)

    residual0, residual_wall_s = _profiled("vmec_jax_initial_residual", outdir, initial_residual)

    def exact_jacobian():
        return stage.optimizer.jacobian_fun(x0)

    jac_ad, jac_wall_s = _profiled("vmec_jax_exact_jacobian", outdir, exact_jacobian)
    jac_fd = None
    fd_wall_s = None
    if RUN_FINITE_DIFFERENCE:
        jac_fd, fd_wall_s = _profiled(
            "vmec_jax_finite_difference_jacobian",
            outdir,
            lambda: _fd_jacobian(stage.optimizer.residual_fun, x0, step=FINITE_DIFF_STEP),
        )

    state0 = stage.optimizer._solve_exact_with_tape(x0)
    terms0 = _vmec_jax_term_report(problem, stage.ctx, state0)
    np.savez(
        outdir / "vmec_jax_derivatives.npz",
        x0=x0,
        names=np.asarray(names, dtype=object),
        residual0=np.asarray(residual0, dtype=float),
        jac_ad=np.asarray(jac_ad, dtype=float),
        jac_fd=np.asarray(jac_fd, dtype=float) if jac_fd is not None else np.empty((0, 0)),
        x_scale=x_scale,
    )

    trace: list[dict] = []

    def residual_logged(x):
        res = stage.optimizer.residual_fun(x)
        trace.append(
            {
                "callback": len(trace),
                "x": np.asarray(x, dtype=float).tolist(),
                "residual_norm": float(np.linalg.norm(res)),
                "objective": _objective_from_residual(res),
            }
        )
        return res

    def jacobian_logged(x):
        return stage.optimizer.jacobian_fun(x)

    def solve():
        return least_squares(
            residual_logged,
            x0,
            jac=jacobian_logged,
            x_scale=x_scale,
            max_nfev=MAX_NFEV,
            ftol=FTOL,
            gtol=GTOL,
            xtol=XTOL,
            method="trf",
            tr_solver="lsmr",
            verbose=2,
        )

    result, solve_wall_s = _profiled("vmec_jax_scipy_solve", outdir, solve)
    residual_final = stage.optimizer.residual_fun(result.x)
    state_final = stage.optimizer._solve_exact_with_tape(result.x)
    terms_final = _vmec_jax_term_report(problem, stage.ctx, state_final)

    stage.optimizer.save_input(outdir / "input.initial", x0)
    stage.optimizer.save_input(outdir / "input.final", result.x)
    stage.optimizer.save_wout(outdir / "wout_final.nc", result.x, state=state_final)

    summary = {
        "backend": "vmec_jax",
        "input_file": INPUT_FILE,
        "dof_names": names,
        "x0": x0,
        "x_final": result.x,
        "x_scale": x_scale,
        "initial_objective": _objective_from_residual(residual0),
        "final_objective": _objective_from_residual(residual_final),
        "initial_terms": terms0,
        "final_terms": terms_final,
        "jacobian_summary": _summarize_jacobian(jac_ad, jac_fd),
        "jacobian_block_summary": _summarize_jacobian_blocks(terms0, jac_ad, jac_fd),
        "timing_s": {
            "initial_residual": residual_wall_s,
            "exact_jacobian": jac_wall_s,
            "finite_difference_jacobian": fd_wall_s,
            "solve": solve_wall_s,
        },
        "scipy_result": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
        },
        "trace": trace,
    }
    _write_json(outdir / "vmec_jax_summary.json", summary)
    return summary


def _maybe_prepend_simsopt_source() -> None:
    if SIMSOPT_SOURCE is None:
        return
    source = Path(SIMSOPT_SOURCE).expanduser()
    if source.exists():
        sys.path.insert(0, str(source))


def _simsopt_x_scale(dof_names, alpha: float = 1.2, min_scale: float = 1.0e-9):
    pattern = r"[rz][cs]\((\d+),(-?\d+)\)"
    x_scale = np.ones(len(dof_names), dtype=float)
    for i, name in enumerate(dof_names):
        match = re.search(pattern, name)
        if match is None:
            continue
        mode_level = max(abs(int(match.group(1))), abs(int(match.group(2))))
        x_scale[i] = max(np.exp(-alpha * mode_level) / np.exp(-alpha), min_scale)
    return x_scale


def run_simsopt(outdir: Path) -> dict:
    print("\n=== SIMSOPT/omnigenity QA max_mode=1 ===")
    _maybe_prepend_simsopt_source()
    import simsopt
    from simsopt._core.optimizable import Optimizable
    from simsopt._core.util import Struct
    from simsopt.mhd import QuasisymmetryRatioResidual, Vmec, vmec_compute_geometry
    from simsopt.objectives import LeastSquaresProblem
    from simsopt.util import MpiPartition

    class LgradBresidual(Optimizable):
        def __init__(
            self,
            vmec: Vmec,
            s: int = 1,
            ntheta: int = 9,
            nphi: int = 7,
            LgradBthreshold: float = 0.35,
        ) -> None:
            self.vmec = vmec
            self.s = s
            self.ntheta = ntheta
            self.nphi = nphi
            self.theta = np.linspace(0, 2.0 * np.pi, ntheta, endpoint=True)
            self.phi = np.linspace(0, 2.0 * np.pi / vmec.indata.nfp, nphi, endpoint=True)
            self.threshold = LgradBthreshold
            super().__init__(depends_on=[vmec])

        def compute(self):
            self.vmec.run()
            data = vmec_compute_geometry(self.vmec, self.s, self.theta, self.phi)
            lgradb = data.L_grad_B.reshape(self.ntheta * self.nphi)
            filtered = np.maximum(1.0 / lgradb - 1.0 / self.threshold, 0.0)
            residuals1d = filtered / np.sqrt(self.ntheta * self.nphi)
            total = float(np.dot(residuals1d, residuals1d))
            results = Struct()
            results.residuals1d = residuals1d
            results.total = total
            return results

        def residuals(self):
            return self.compute().residuals1d

        def total(self):
            return self.compute().total

    mpi = MpiPartition()
    vmec = Vmec(str(INPUT_FILE), mpi=mpi, verbose=False)
    vmec.indata.mpol = max(2 + MAX_MODE, MIN_VMEC_MODE)
    vmec.indata.ntor = vmec.indata.mpol
    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=MAX_MODE, nmin=-MAX_MODE, nmax=MAX_MODE, fixed=False)
    surf.fix("rc(0,0)")

    qs = QuasisymmetryRatioResidual(vmec, QS_SURFACES, helicity_m=1, helicity_n=0)
    lgradb = LgradBresidual(vmec, s=1, ntheta=9, nphi=7, LgradBthreshold=LGRADB_THRESHOLD)
    problem = LeastSquaresProblem.from_tuples(
        [
            (vmec.aspect, ASPECT_TARGET, 1.0),
            (qs.residuals, 0.0, 1.0),
            (vmec.mean_iota, IOTA_TARGET, 1.0e2),
            (lgradb.residuals, 0.0, 1.0),
        ]
    )
    x0 = np.asarray(problem.x, dtype=float)
    names = list(problem.dof_names)
    x_scale = _simsopt_x_scale(names, alpha=ESS_ALPHA, min_scale=1.0e-9)

    def residual_fun(x):
        return np.asarray(problem.residuals(np.asarray(x, dtype=float)), dtype=float)

    residual0, residual_wall_s = _profiled("simsopt_initial_residual", outdir, lambda: residual_fun(x0))
    jac_fd = None
    fd_wall_s = None
    if RUN_FINITE_DIFFERENCE:
        jac_fd, fd_wall_s = _profiled(
            "simsopt_finite_difference_jacobian",
            outdir,
            lambda: _fd_jacobian(residual_fun, x0, step=FINITE_DIFF_STEP),
        )

    np.savez(
        outdir / "simsopt_derivatives.npz",
        x0=x0,
        names=np.asarray(names, dtype=object),
        residual0=np.asarray(residual0, dtype=float),
        jac_fd=np.asarray(jac_fd, dtype=float) if jac_fd is not None else np.empty((0, 0)),
        x_scale=x_scale,
    )

    trace: list[dict] = []

    def residual_logged(x):
        res = residual_fun(x)
        trace.append(
            {
                "callback": len(trace),
                "x": np.asarray(x, dtype=float).tolist(),
                "residual_norm": float(np.linalg.norm(res)),
                "objective": _objective_from_residual(res),
            }
        )
        return res

    def solve():
        return least_squares(
            residual_logged,
            x0,
            jac="2-point",
            diff_step=FINITE_DIFF_STEP,
            x_scale=x_scale,
            max_nfev=MAX_NFEV,
            ftol=FTOL,
            gtol=GTOL,
            xtol=XTOL,
            method="trf",
            tr_solver="lsmr",
            verbose=2,
        )

    result, solve_wall_s = _profiled("simsopt_scipy_solve", outdir, solve)
    residual_final = residual_fun(result.x)

    with contextlib.suppress(Exception):
        vmec.write_input(str(outdir / "input.final"))
    with contextlib.suppress(Exception):
        if getattr(vmec, "output_file", None):
            Path(vmec.output_file).replace(outdir / "wout_final.nc")

    summary = {
        "backend": "simsopt",
        "simsopt_module": getattr(simsopt, "__file__", "unknown"),
        "input_file": INPUT_FILE,
        "dof_names": names,
        "x0": x0,
        "x_final": result.x,
        "x_scale": x_scale,
        "initial_objective": _objective_from_residual(residual0),
        "final_objective": _objective_from_residual(residual_final),
        "initial_terms": {
            "aspect": float(vmec.aspect()),
            "mean_iota": float(vmec.mean_iota()),
            "qs_total": float(qs.total()),
            "LgradB_total": float(lgradb.total()),
        },
        "timing_s": {
            "initial_residual": residual_wall_s,
            "finite_difference_jacobian": fd_wall_s,
            "solve": solve_wall_s,
        },
        "scipy_result": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
        },
        "trace": trace,
    }
    _write_json(outdir / "simsopt_summary.json", summary)
    return summary


def compare_summaries(outdir: Path, vmec_jax: dict | None, simsopt: dict | None) -> None:
    comparison: dict = {
        "input_file": INPUT_FILE,
        "max_mode": MAX_MODE,
        "objective_policy": {
            "aspect_target": ASPECT_TARGET,
            "iota_target": IOTA_TARGET,
            "lgradb_threshold": LGRADB_THRESHOLD,
            "qs_surfaces": QS_SURFACES,
            "ess_alpha": ESS_ALPHA,
        },
    }
    if vmec_jax is not None:
        comparison["vmec_jax"] = {
            "initial_objective": vmec_jax["initial_objective"],
            "final_objective": vmec_jax["final_objective"],
            "jacobian_summary": vmec_jax["jacobian_summary"],
            "dof_names": vmec_jax["dof_names"],
        }
    if simsopt is not None:
        comparison["simsopt"] = {
            "initial_objective": simsopt["initial_objective"],
            "final_objective": simsopt["final_objective"],
            "dof_names": simsopt["dof_names"],
        }
    if vmec_jax is not None and simsopt is not None:
        comparison["cross_backend"] = {
            "same_dof_names": list(vmec_jax["dof_names"]) == list(simsopt["dof_names"]),
            "initial_objective_ratio_vmec_jax_over_simsopt": float(
                vmec_jax["initial_objective"] / max(simsopt["initial_objective"], np.finfo(float).tiny)
            ),
            "final_objective_ratio_vmec_jax_over_simsopt": float(
                vmec_jax["final_objective"] / max(simsopt["final_objective"], np.finfo(float).tiny)
            ),
        }
    _write_json(outdir / "comparison_summary.json", comparison)


if __name__ == "__main__":
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vmec_jax_summary = run_vmec_jax(OUTPUT_DIR / "vmec_jax") if RUN_VMEC_JAX else None
    simsopt_summary = None
    if RUN_SIMSOPT:
        try:
            simsopt_summary = run_simsopt(OUTPUT_DIR / "simsopt")
        except Exception as exc:
            failure = {
                "backend": "simsopt",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "input_file": INPUT_FILE,
                "max_mode": MAX_MODE,
                "min_vmec_mode": MIN_VMEC_MODE,
            }
            _write_json(OUTPUT_DIR / "simsopt_failure.json", failure)
            print(f"SIMSOPT comparison failed: {type(exc).__name__}: {exc}")
    compare_summaries(OUTPUT_DIR, vmec_jax_summary, simsopt_summary)
    print(f"\nComparison outputs: {OUTPUT_DIR.resolve()}")

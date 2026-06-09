#!/usr/bin/env python
"""Compare QI exact-optimizer JVPs against central finite differences.

This is a diagnostic for QI cleanup stalls: if direct fixed-boundary
perturbations change the QI metric but the matrix-free optimizer does not move,
run this tool at the same input deck and coefficient direction.

Example
-------
Compare the VMEC input coefficient ``RBC(n=0,m=1)`` (optimizer kind ``rc``):

.. code-block:: bash

   PYTHONPATH=. JAX_PLATFORMS=cuda python tools/diagnostics/qi_optimizer_jvp_fd_compare.py \
     --input examples/data/input.nfp2_QI \
     --kind rc --m 1 --n 0 --max-mode 3 --solver-device gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64
from vmec_jax.optimization_workflow import build_quasi_isodynamic_objective_stage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="VMEC input deck to diagnose.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--kind", choices=("rc", "rs", "zc", "zs"), default="rc")
    parser.add_argument("--m", type=int, default=1, help="Optimizer poloidal mode number.")
    parser.add_argument("--n", type=int, default=0, help="Optimizer toroidal mode number.")
    parser.add_argument("--max-mode", type=int, default=3)
    parser.add_argument("--min-vmec-mode", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--inner-max-iter", type=int, default=450)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-9)
    parser.add_argument("--trial-max-iter", type=int, default=450)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-9)
    parser.add_argument("--solver-device", choices=("cpu", "gpu", "none", "default"), default="default")
    parser.add_argument("--exact-path", choices=("auto", "tape", "scan"), default="auto")
    parser.add_argument("--mboz", type=int, default=7)
    parser.add_argument("--nboz", type=int, default=7)
    parser.add_argument("--nphi", type=int, default=61)
    parser.add_argument("--nalpha", type=int, default=13)
    parser.add_argument("--n-bounce", type=int, default=17)
    parser.add_argument("--surfaces", type=str, default="0.1,0.28,0.46,0.64,0.82,1.0")
    parser.add_argument(
        "--dense-jacobian",
        action="store_true",
        help="Also materialize the dense exact Jacobian and compare the selected column.",
    )
    return parser.parse_args()


def _surfaces(text: str) -> np.ndarray:
    values = [float(item) for item in str(text).replace(",", " ").split()]
    if not values:
        raise ValueError("--surfaces must contain at least one value.")
    return np.asarray(values, dtype=float)


def _spec_index(specs, *, kind: str, m: int, n: int) -> int:
    matches = [
        i
        for i, spec in enumerate(specs)
        if spec.kind == kind and int(spec.m) == int(m) and int(spec.n) == int(n)
    ]
    if not matches:
        available = ", ".join(f"{spec.kind}(m={spec.m},n={spec.n})" for spec in specs[:40])
        raise ValueError(
            f"No active parameter found for {kind}(m={m}, n={n}). "
            f"First active parameters: {available}"
        )
    return int(matches[0])


def main() -> None:
    args = _parse_args()
    enable_x64(True)

    solver_device = None if args.solver_device in {"none", "default"} else str(args.solver_device)
    exact_path = None if args.exact_path == "auto" else str(args.exact_path)
    input_file = Path(args.input).expanduser()
    surfaces = _surfaces(args.surfaces)

    vmec = vj.FixedBoundaryVMEC.from_input(
        input_file,
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        project_input_boundary_to_max_mode=True,
    )
    qi_options = vj.QuasiIsodynamicOptions(
        surfaces=surfaces,
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        nphi=int(args.nphi),
        nalpha=int(args.nalpha),
        n_bounce=int(args.n_bounce),
        include_bounce_endpoints=True,
        softness=2.0e-2,
        width_weight=1.0,
        branch_width_weight=0.5,
        branch_width_softness=2.0e-2,
        profile_weight=0.1,
        shuffle_profile_weight=1.0,
        shuffle_profile_softness=2.0e-2,
        weighted_shuffle_profile_weight=0.0,
        weighted_shuffle_profile_softness=2.0e-2,
        phimin=0.0,
        jit_booz=True,
    )
    problem = vj.LeastSquaresProblem.from_tuples(
        [(vj.QuasiIsodynamicResidual(qi_options).J, 0.0, 1.0)]
    )
    stage = build_quasi_isodynamic_objective_stage(
        vmec.cfg,
        vmec.indata,
        stage_mode=int(args.max_mode),
        scalar_objectives=problem.objective_terms,
        qi_objectives=problem.qi_objective_terms,
        surfaces=qi_options.surfaces,
        mboz=qi_options.mboz,
        nboz=qi_options.nboz,
        nphi=qi_options.nphi,
        nalpha=qi_options.nalpha,
        n_bounce=qi_options.n_bounce,
        include_bounce_endpoints=qi_options.include_bounce_endpoints,
        softness=qi_options.softness,
        width_weight=qi_options.width_weight,
        branch_width_weight=qi_options.branch_width_weight,
        branch_width_softness=qi_options.branch_width_softness,
        profile_weight=qi_options.profile_weight,
        shuffle_profile_weight=qi_options.shuffle_profile_weight,
        shuffle_profile_softness=qi_options.shuffle_profile_softness,
        shuffle_profile_nphi_out=qi_options.shuffle_profile_nphi_out,
        weighted_shuffle_profile_weight=qi_options.weighted_shuffle_profile_weight,
        weighted_shuffle_profile_softness=qi_options.weighted_shuffle_profile_softness,
        aligned_profile_weight=qi_options.aligned_profile_weight,
        aligned_profile_softness=qi_options.aligned_profile_softness,
        aligned_profile_trap_level=qi_options.aligned_profile_trap_level,
        aligned_profile_trap_softness=qi_options.aligned_profile_trap_softness,
        phimin=qi_options.phimin,
        jit_booz=qi_options.jit_booz,
        project_input_boundary_to_max_mode=vmec.project_input_boundary_to_max_mode,
        include=vmec.include,
        fix=vmec.fix,
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
        solver_device=solver_device,
        exact_path=exact_path,
    )

    params0 = np.zeros(len(stage.specs), dtype=float)
    direction = np.zeros_like(params0)
    idx = _spec_index(stage.specs, kind=str(args.kind), m=int(args.m), n=int(args.n))
    direction[idx] = 1.0
    eps = float(args.epsilon)

    r0 = np.asarray(stage.optimizer.residual_fun(params0), dtype=float)
    linear_operator = stage.optimizer.residual_linear_operator(params0)
    jvp = np.asarray(linear_operator.matvec(direction), dtype=float)
    r_plus = np.asarray(stage.optimizer.residual_fun(params0 + eps * direction), dtype=float)
    r_minus = np.asarray(stage.optimizer.residual_fun(params0 - eps * direction), dtype=float)
    fd = (r_plus - r_minus) / (2.0 * eps)

    diff = jvp - fd
    fd_norm = float(np.linalg.norm(fd))
    jvp_norm = float(np.linalg.norm(jvp))
    diff_norm = float(np.linalg.norm(diff))
    denom = max(fd_norm, jvp_norm, np.finfo(float).eps)
    dot = float(np.vdot(jvp, fd))
    cosine = dot / max(fd_norm * jvp_norm, np.finfo(float).eps)
    report = {
        "input": str(input_file),
        "max_mode": int(args.max_mode),
        "parameter": {
            "index": idx,
            "name": stage.specs[idx].name,
            "kind": stage.specs[idx].kind,
            "m": int(stage.specs[idx].m),
            "n": int(stage.specs[idx].n),
        },
        "epsilon": eps,
        "residual_size": int(r0.size),
        "residual_norm": float(np.linalg.norm(r0)),
        "jvp_norm": jvp_norm,
        "fd_norm": fd_norm,
        "diff_norm": diff_norm,
        "relative_diff_norm": diff_norm / denom,
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "cosine_similarity": cosine,
        "solver_device": solver_device or "default",
        "exact_path": exact_path or "auto",
    }
    if bool(args.dense_jacobian):
        jac = np.asarray(stage.optimizer.jacobian_fun(params0), dtype=float)
        dense_col = jac[:, idx]
        dense_diff = dense_col - fd
        dense_norm = float(np.linalg.norm(dense_col))
        dense_diff_norm = float(np.linalg.norm(dense_diff))
        dense_dot = float(np.vdot(dense_col, fd))
        report["dense_jacobian"] = {
            "shape": [int(jac.shape[0]), int(jac.shape[1])],
            "column_norm": dense_norm,
            "diff_norm": dense_diff_norm,
            "relative_diff_norm": dense_diff_norm / max(dense_norm, fd_norm, np.finfo(float).eps),
            "max_abs_diff": float(np.max(np.abs(dense_diff))) if dense_diff.size else 0.0,
            "cosine_similarity": dense_dot / max(dense_norm * fd_norm, np.finfo(float).eps),
            "matrix_free_column_diff_norm": float(np.linalg.norm(dense_col - jvp)),
        }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()

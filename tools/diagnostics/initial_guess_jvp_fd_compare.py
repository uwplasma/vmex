#!/usr/bin/env python
"""Compare VMEC initial-guess JVPs against finite differences.

This diagnostic is narrower than the full QI optimizer JVP comparison: it stops
before any VMEC residual iteration and isolates nonsmoothness in
``initial_guess_from_boundary`` itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64, jax, jnp
from vmec_jax.boundary import boundary_from_input_convention, boundary_input_from_indata
from vmec_jax.init_guess import extract_axis_override_from_state, initial_guess_from_boundary
from vmec_jax.optimization import (
    _apply_boundary_params_numpy,
    apply_boundary_params,
    boundary_param_specs,
    extend_boundary_for_max_mode,
    truncate_indata_boundary_modes,
)
from vmec_jax.state import pack_state
from vmec_jax.static import build_static


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="VMEC input deck to diagnose.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--kind", choices=("rc", "rs", "zc", "zs"), default="rc")
    parser.add_argument("--m", type=int, default=1, help="Optimizer poloidal mode number.")
    parser.add_argument("--n", type=int, default=0, help="Optimizer toroidal mode number.")
    parser.add_argument("--max-mode", type=int, default=3)
    parser.add_argument("--min-vmec-mode", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=1.0e-4)
    parser.add_argument("--include", type=str, default="rc,zs")
    parser.add_argument("--fix", type=str, default="rc00")
    parser.add_argument(
        "--no-project-input-boundary",
        action="store_true",
        help="Keep fixed high-mode input boundary coefficients instead of truncating to max-mode.",
    )
    return parser.parse_args()


def _csv_tuple(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def _spec_index(specs, *, kind: str, m: int, n: int) -> int:
    for i, spec in enumerate(specs):
        if spec.kind == kind and int(spec.m) == int(m) and int(spec.n) == int(n):
            return int(i)
    available = ", ".join(f"{spec.kind}(m={spec.m},n={spec.n})" for spec in specs[:40])
    raise ValueError(
        f"No active parameter found for {kind}(m={m}, n={n}). "
        f"First active parameters: {available}"
    )


def _state_component_report(layout, tangent, fd) -> dict[str, dict[str, float]]:
    names = ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin")
    tangent_blocks = layout.split(jnp.asarray(tangent))
    fd_blocks = layout.split(jnp.asarray(fd))
    out: dict[str, dict[str, float]] = {}
    for name, tangent_block, fd_block in zip(names, tangent_blocks, fd_blocks, strict=True):
        tangent_np = np.asarray(tangent_block, dtype=float)
        fd_np = np.asarray(fd_block, dtype=float)
        diff = tangent_np - fd_np
        tangent_norm = float(np.linalg.norm(tangent_np))
        fd_norm = float(np.linalg.norm(fd_np))
        out[name] = {
            "jvp_norm": tangent_norm,
            "fd_norm": fd_norm,
            "diff_norm": float(np.linalg.norm(diff)),
            "relative_diff_norm": float(np.linalg.norm(diff))
            / max(tangent_norm, fd_norm, np.finfo(float).eps),
            "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "cosine_similarity": float(np.vdot(tangent_np.reshape(-1), fd_np.reshape(-1)))
            / max(tangent_norm * fd_norm, np.finfo(float).eps),
        }
    return out


def _rtest_ztest(static, boundary) -> dict[str, float | bool]:
    m = np.asarray(static.modes.m, dtype=int)
    mask = m == 1
    rtest = float(np.sum(np.asarray(boundary.R_cos, dtype=float)[mask])) if np.any(mask) else 0.0
    ztest = float(np.sum(np.asarray(boundary.Z_sin, dtype=float)[mask])) if np.any(mask) else 0.0
    product = rtest * ztest
    return {
        "rtest": rtest,
        "ztest": ztest,
        "product": product,
        "lflip": bool(product < 0.0),
    }


def main() -> None:
    args = _parse_args()
    enable_x64(True)

    input_file = Path(args.input).expanduser()
    vmec = vj.FixedBoundaryVMEC.from_input(
        input_file,
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        project_input_boundary_to_max_mode=not bool(args.no_project_input_boundary),
    )
    stage_indata0 = (
        truncate_indata_boundary_modes(vmec.indata, max_mode=int(args.max_mode))
        if not bool(args.no_project_input_boundary)
        else vmec.indata
    )
    static = build_static(vmec.cfg)
    boundary = vj.boundary_from_indata(stage_indata0, static.modes, apply_m1_constraint=False)
    stage_indata, static, boundary = extend_boundary_for_max_mode(
        stage_indata0,
        static,
        boundary,
        int(args.max_mode),
    )
    boundary_input = boundary_input_from_indata(stage_indata, static.modes)
    specs = boundary_param_specs(
        boundary_input,
        static.modes,
        max_mode=int(args.max_mode),
        min_coeff=0.0,
        include=_csv_tuple(args.include),
        fix=_csv_tuple(args.fix),
    )
    idx = _spec_index(specs, kind=str(args.kind), m=int(args.m), n=int(args.n))
    params0 = np.zeros(len(specs), dtype=float)
    direction = np.zeros_like(params0)
    direction[idx] = 1.0
    eps = float(args.epsilon)

    def _boundary_from_params(params):
        boundary_i = apply_boundary_params(
            boundary_input,
            specs,
            jnp.asarray(params, dtype=jnp.float64),
        )
        return boundary_from_input_convention(
            boundary_i,
            static.modes,
            lasym=bool(static.cfg.lasym),
            apply_m1_constraint=False,
        )

    def _boundary_from_params_np(params):
        boundary_i = _apply_boundary_params_numpy(boundary_input, specs, np.asarray(params, dtype=float))
        return boundary_from_input_convention(
            boundary_i,
            static.modes,
            lasym=bool(static.cfg.lasym),
            apply_m1_constraint=False,
        )

    boundary0 = _boundary_from_params_np(params0)
    base_state = initial_guess_from_boundary(
        static,
        boundary0,
        stage_indata,
        vmec_project=True,
        axis_override=None,
    )
    base_axis_override = extract_axis_override_from_state(base_state, static)

    def _variant_report(*, vmec_project: bool, axis_mode: str) -> dict:
        axis_override = None
        if axis_mode == "base":
            axis_override = base_axis_override
        elif axis_mode != "inferred":
            raise ValueError(f"Unknown axis mode {axis_mode!r}")

        def _packed(params):
            state = initial_guess_from_boundary(
                static,
                _boundary_from_params(params),
                stage_indata,
                vmec_project=bool(vmec_project),
                axis_override=axis_override,
            )
            return jnp.asarray(pack_state(state), dtype=jnp.float64)

        params_j = jnp.asarray(params0, dtype=jnp.float64)
        direction_j = jnp.asarray(direction, dtype=jnp.float64)
        packed0, tangent = jax.jvp(_packed, (params_j,), (direction_j,))
        packed_plus = _packed(params_j + eps * direction_j)
        packed_minus = _packed(params_j - eps * direction_j)
        fd = (packed_plus - packed_minus) / (2.0 * eps)
        tangent_np = np.asarray(tangent, dtype=float)
        fd_np = np.asarray(fd, dtype=float)
        diff = tangent_np - fd_np
        tangent_norm = float(np.linalg.norm(tangent_np))
        fd_norm = float(np.linalg.norm(fd_np))
        return {
            "vmec_project": bool(vmec_project),
            "axis_mode": axis_mode,
            "packed_state_size": int(np.asarray(packed0).size),
            "jvp_norm": tangent_norm,
            "fd_norm": fd_norm,
            "diff_norm": float(np.linalg.norm(diff)),
            "relative_diff_norm": float(np.linalg.norm(diff))
            / max(tangent_norm, fd_norm, np.finfo(float).eps),
            "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
            "cosine_similarity": float(np.vdot(tangent_np, fd_np))
            / max(tangent_norm * fd_norm, np.finfo(float).eps),
            "components": _state_component_report(base_state.layout, tangent, fd),
        }

    report = {
        "input": str(input_file),
        "max_mode": int(args.max_mode),
        "epsilon": eps,
        "parameter": {
            "index": int(idx),
            "name": specs[idx].name,
            "kind": specs[idx].kind,
            "m": int(specs[idx].m),
            "n": int(specs[idx].n),
        },
        "rtest_ztest": {
            "base": _rtest_ztest(static, boundary0),
            "plus": _rtest_ztest(static, _boundary_from_params_np(params0 + eps * direction)),
            "minus": _rtest_ztest(static, _boundary_from_params_np(params0 - eps * direction)),
        },
        "variants": [
            _variant_report(vmec_project=False, axis_mode="inferred"),
            _variant_report(vmec_project=False, axis_mode="base"),
            _variant_report(vmec_project=True, axis_mode="inferred"),
            _variant_report(vmec_project=True, axis_mode="base"),
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
